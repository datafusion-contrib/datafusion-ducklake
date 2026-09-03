#![cfg(feature = "metadata-duckdb")]
//! Parity with official DuckLake's own filter-pushdown assertions.
//!
//! `tests/sqllogictests/sql/stats/filter_pushdown.test` is DuckDB's filter
//! pushdown test, vendored here and on the `EXPECTED_PASS` ratchet. It carries
//! nine `EXPLAIN ANALYZE ... Total Files Read: N` assertions that the
//! sqllogictest runner cannot check: `preprocess_test_file` skips every
//! `EXPLAIN` block, because DuckDB's analyzed plan and DataFusion's share no
//! format. So the vendored test checks the values and nothing about pruning.
//! This file checks the nine counts directly.
//!
//! The fixture is built by official DuckLake, through a real DuckDB `ATTACH
//! 'ducklake:...'` and the same `INSERT ... FROM range(...)` statements the
//! official test runs. `ducklake_file_column_stats` is therefore written by
//! official code and read back by [`datafusion_ducklake::stats_filter`]'s SQL,
//! which is what makes this a parity test: building the fixture with this
//! crate's own writer would only show that its two halves agree with each other.
//!
//! # Two assertions per predicate
//!
//! 1. **Catalog-level.** Lower the predicate and count the rows
//!    `get_table_file_metadata_page_filtered` returns. This is the only
//!    measurement that isolates the SQL-side pruning: a `PartitionedFile` count
//!    cannot separate it from the in-memory `PruningPredicate` that has always
//!    run over every listed file.
//! 2. **Correctness.** Run the official query through DataFusion and assert
//!    official's value. A file wrongly pruned shows up here as a short count,
//!    and this assertion is never weakened to make a count match.
//!
//! Each predicate is also measured against the same page call with no filter,
//! which returns every live file. That is the number the catalog-level
//! assertion would read if the pushdown did nothing, and it differs from
//! official's count for all nine, so none of them can pass with the mechanism
//! disabled.
//!
//! # Backends
//!
//! Every backend that can host a DuckLake catalog runs the same nine numbers:
//! DuckDB, SQLite, PostgreSQL and MySQL. Identical counts across four SQL
//! dialects is what shows the dialect-specific rendering — the `TRY_CAST`
//! replacements and the forced binary collation — preserved official's
//! semantics rather than quietly changing them. `MulticatalogProvider` is the
//! one provider absent here: its catalog is keyed by `ducklake_catalog_id` and
//! headed through `ducklake_catalog_snapshot_map`, a layout official DuckLake
//! does not write, so it cannot host an officially-built fixture at all.

use std::path::Path;
use std::sync::Arc;

use arrow::array::{
    Array, Date32Array, Decimal128Array, Int32Array, Int64Array, RecordBatch, StringViewArray,
};
use arrow::datatypes::{DataType, Schema};
use datafusion::catalog::CatalogProvider;
use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion::common::{DFSchema, ScalarValue};
use datafusion::logical_expr::LogicalPlan;
use datafusion::logical_expr::expr_rewriter::unnormalize_cols;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::prelude::*;
use datafusion_ducklake::stats_filter::lower_predicate;
use datafusion_ducklake::{DuckLakeCatalog, MetadataProvider};
use tempfile::TempDir;

use crate::common;

/// The official test's table definition, verbatim.
const CREATE_TABLE: &str =
    "CREATE TABLE lake.filter_pushdown(v INTEGER, i INTEGER, d DATE, k DECIMAL(9, 3), s VARCHAR)";

/// The official test's four `INSERT`s, one data file each. The last two offset
/// `d` from 1970 rather than 2000, which is what puts the 500000-hour range in
/// 2027; that is how the official test writes them.
const INSERTS: [&str; 4] = [
    "INSERT INTO lake.filter_pushdown
     SELECT i % 1000 v, i, (TIMESTAMP '2000-01-01' + interval (i) hour)::DATE, i / 10,
            printf('%06d', i)
     FROM range(1000) t(i)",
    "INSERT INTO lake.filter_pushdown
     SELECT i % 1000 v, i, (TIMESTAMP '2000-01-01' + interval (i) hour)::DATE, i / 10,
            printf('%06d', i)
     FROM range(100000,101000) t(i)",
    "INSERT INTO lake.filter_pushdown
     SELECT i % 1000 v, i, (TIMESTAMP '1970-01-01' + interval (i) hour)::DATE, i / 10,
            printf('%06d', i)
     FROM range(500000,501000) t(i)",
    "INSERT INTO lake.filter_pushdown
     SELECT i % 1000 v, i, (TIMESTAMP '1970-01-01' + interval (i) hour)::DATE, i / 10,
            printf('%06d', i)
     FROM range(501000, 501001) t(i)",
];

/// One page is large enough to hold every file in this fixture, so a returned
/// row count is the whole answer rather than one page of it.
const PAGE_LIMIT: usize = 1024;

/// Run `statements` against `attach_target` through official DuckLake.
///
/// `extensions` are the DuckDB catalog extensions the target needs beyond
/// `ducklake` itself — `sqlite`, `postgres`, `mysql`.
///
/// `DATA_INLINING_ROW_LIMIT 0` keeps the fourth `INSERT` a data file. DuckDB
/// 1.5.5 inlines a single-row `INSERT` into the catalog by default, which leaves
/// the table with three data files and nothing for `i != 501000` to prune —
/// official's own comment on that statement ("Single row so it should be able to
/// be pruned with != filter") says a fourth file is what it means to create.
/// Inlined rows are also a separate read path from the one under test here.
fn with_official_ducklake(
    attach_target: &str,
    extensions: &[&str],
    data_path: &Path,
    statements: &[&str],
) -> anyhow::Result<()> {
    common::ensure_ducklake_installed();
    for extension in extensions {
        common::ensure_extension_installed(extension);
    }
    std::fs::create_dir_all(data_path)?;

    let conn = duckdb::Connection::open_in_memory()?;
    for extension in extensions {
        conn.execute(&format!("LOAD {extension};"), [])?;
    }
    conn.execute("LOAD ducklake;", [])?;
    conn.execute(
        &format!(
            "ATTACH '{attach_target}' AS lake \
             (DATA_PATH '{}', DATA_INLINING_ROW_LIMIT 0);",
            data_path.display()
        ),
        [],
    )?;
    for statement in statements {
        conn.execute(statement, [])?;
    }
    conn.execute("DETACH lake;", [])?;
    Ok(())
}

/// Create the table and its first three data files, the state the official test
/// makes its first eight assertions against.
fn build_fixture(attach_target: &str, extensions: &[&str], data_path: &Path) -> anyhow::Result<()> {
    let mut statements = vec![CREATE_TABLE];
    statements.extend_from_slice(&INSERTS[..3]);
    with_official_ducklake(attach_target, extensions, data_path, &statements)
}

/// Add the fourth, single-row file. The official test does this only after its
/// first eight assertions, and every one of those eight would read a different
/// number of files with it present.
fn insert_fourth_file(
    attach_target: &str,
    extensions: &[&str],
    data_path: &Path,
) -> anyhow::Result<()> {
    with_official_ducklake(attach_target, extensions, data_path, &INSERTS[3..])
}

/// Bring a catalog written by the DuckLake extension that ships with `duckdb`
/// 1.4.1 — the version this fixture is written by — up to the shape a released
/// DuckDB 1.5.x writes.
///
/// Only `DuckdbMetadataProvider` probes for the newer catalog columns.
/// `SqliteMetadataProvider`, `PostgresMetadataProvider` and
/// `MySqlMetadataProvider` select `ducklake_column.default_value_type` /
/// `default_value_dialect` and `ducklake_schema_versions.table_id`
/// unconditionally, so on the older shape they fail the query outright rather
/// than degrading. `compaction_sqlite_tests::migrate_pinned_duckdb_fixture`
/// tops up the same two tables for the same reason.
///
/// Nothing here touches `ducklake_data_file` or `ducklake_file_column_stats`:
/// every file and every statistic this test reads is exactly as official wrote
/// it. `VARCHAR(255)` and `BIGINT` are accepted by all three dialects, and a
/// statement that fails because the column is already there is discarded at the
/// call site.
const MIGRATE_PINNED_DUCKDB_CATALOG: [&str; 4] = [
    "ALTER TABLE ducklake_column ADD COLUMN default_value_type VARCHAR(255)",
    "ALTER TABLE ducklake_column ADD COLUMN default_value_dialect VARCHAR(255)",
    "ALTER TABLE ducklake_schema_versions ADD COLUMN table_id BIGINT",
    "UPDATE ducklake_schema_versions
     SET table_id = (SELECT table_id FROM ducklake_table WHERE table_name = 'filter_pushdown')",
];

/// The row the official test prints for a `SELECT *` assertion.
struct OfficialRow {
    v: i32,
    i: i32,
    /// `d`, spelled as the official test spells the date.
    d: &'static str,
    /// `k` unscaled: `DECIMAL(9,3)` stores `25.300` as `25300`.
    k: i128,
    s: &'static str,
}

enum Expected {
    /// The value of `SELECT COUNT(*) ... WHERE <sql>`.
    Count(i64),
    /// The single row of `SELECT * ... WHERE <sql>`.
    Row(OfficialRow),
}

/// One of the nine assertions the official test makes about this fixture.
struct Case {
    /// The predicate, spelled as the official test spells it.
    sql: &'static str,
    /// `Total Files Read` from the official `EXPLAIN ANALYZE` assertion.
    official_files: usize,
    /// The result the official test asserts for the same predicate.
    expected: Expected,
    /// The predicate as a DataFusion expression. Each literal takes its own
    /// column's type, which is the form DataFusion's type coercion and
    /// `UnwrapCastInComparison` leave behind by the time filters reach `scan`.
    predicate: fn(&Schema) -> Expr,
}

/// A literal of `column`'s own type, parsed from the text the official test
/// writes.
fn literal(schema: &Schema, column: &str, text: &str) -> Expr {
    let field = schema
        .field_with_name(column)
        .unwrap_or_else(|e| panic!("column `{column}` in the fixture schema: {e}"));
    lit(
        ScalarValue::try_from_string(text.to_string(), field.data_type())
            .unwrap_or_else(|e| panic!("`{text}` as {}: {e}", field.data_type())),
    )
}

/// The eight assertions the official test makes against the three-file table.
fn official_cases() -> [Case; 8] {
    [
        Case {
            sql: "i > 100998",
            official_files: 2,
            expected: Expected::Count(1001),
            predicate: |schema| col("i").gt(literal(schema, "i", "100998")),
        },
        Case {
            sql: "i >= 100999",
            official_files: 2,
            expected: Expected::Count(1001),
            predicate: |schema| col("i").gt_eq(literal(schema, "i", "100999")),
        },
        Case {
            sql: "d = DATE '2000-01-23'",
            official_files: 1,
            expected: Expected::Count(24),
            predicate: |schema| col("d").eq(literal(schema, "d", "2000-01-23")),
        },
        // The fixed-point decimal encoding: `k`'s bounds are stored as
        // `0.000` / `99.900`, and the literal has to render the same way to
        // compare against them.
        Case {
            sql: "k = 25.3",
            official_files: 1,
            expected: Expected::Row(OfficialRow {
                v: 253,
                i: 253,
                d: "2000-01-11",
                k: 25_300,
                s: "000253",
            }),
            predicate: |schema| col("k").eq(literal(schema, "k", "25.3")),
        },
        // The raw, uncast, collation-sensitive string path.
        Case {
            sql: "s >= '500023'",
            official_files: 1,
            expected: Expected::Count(977),
            predicate: |schema| col("s").gt_eq(literal(schema, "s", "500023")),
        },
        // `AND` across two columns: two CTEs, intersected.
        Case {
            sql: "d >= DATE '2011-05-29' AND k < 50000",
            official_files: 1,
            expected: Expected::Count(1000),
            predicate: |schema| {
                col("d")
                    .gt_eq(literal(schema, "d", "2011-05-29"))
                    .and(col("k").lt(literal(schema, "k", "50000")))
            },
        },
        // `OR` within one column, which is all-or-nothing: a branch that failed
        // to lower would have to abandon the whole disjunction.
        Case {
            sql: "i = 527 OR i = 100527",
            official_files: 2,
            expected: Expected::Count(2),
            predicate: |schema| {
                col("i")
                    .eq(literal(schema, "i", "527"))
                    .or(col("i").eq(literal(schema, "i", "100527")))
            },
        },
        Case {
            sql: "i IN (500, 600, 700)",
            official_files: 1,
            expected: Expected::Count(3),
            predicate: |schema| {
                col("i").in_list(
                    vec![
                        literal(schema, "i", "500"),
                        literal(schema, "i", "600"),
                        literal(schema, "i", "700"),
                    ],
                    false,
                )
            },
        },
    ]
}

/// The ninth assertion, which only means anything once the fourth file exists:
/// `NOT (min = C AND max = C)` prunes that file precisely because every row in
/// it holds the same value.
fn not_equal_case() -> [Case; 1] {
    [Case {
        sql: "i != 501000",
        official_files: 3,
        expected: Expected::Count(3000),
        predicate: |schema| col("i").not_eq(literal(schema, "i", "501000")),
    }]
}

/// The filters DataFusion's planner pushes into the `TableScan` for `query`,
/// in the form `DuckLakeTable::scan` receives them.
///
/// Lowering this form is what makes the mechanism reachable from SQL rather
/// than only from a hand-built predicate. `unnormalize_cols` mirrors the
/// physical planner, which strips the `lake.main.filter_pushdown` qualifier
/// before handing filters to a `TableProvider`; `physical_conjuncts` resolves
/// them against an unqualified schema and would reject them otherwise.
async fn pushed_down_filters(ctx: &SessionContext, query: &str) -> Vec<Expr> {
    let plan = ctx
        .sql(query)
        .await
        .expect("query plans")
        .into_optimized_plan()
        .expect("query optimizes");
    let mut filters = Vec::new();
    plan.apply(|node| {
        if let LogicalPlan::TableScan(scan) = node {
            filters.extend(scan.filters.iter().cloned());
        }
        Ok(TreeNodeRecursion::Continue)
    })
    .expect("walking the plan does not fail");
    unnormalize_cols(filters)
}

fn count_value(batches: &[RecordBatch]) -> i64 {
    let batch = batches
        .iter()
        .find(|batch| batch.num_rows() > 0)
        .expect("COUNT(*) returns a row");
    batch
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("COUNT(*) is Int64")
        .value(0)
}

fn assert_official_row(label: &str, batches: &[RecordBatch], expected: &OfficialRow) {
    let rows: usize = batches.iter().map(RecordBatch::num_rows).sum();
    assert_eq!(rows, 1, "{label}: official prints exactly one row");
    let batch = batches
        .iter()
        .find(|batch| batch.num_rows() == 1)
        .expect("the one row");

    let int32 = |index: usize| {
        batch
            .column(index)
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("INTEGER column")
            .value(0)
    };
    assert_eq!(int32(0), expected.v, "{label}: v");
    assert_eq!(int32(1), expected.i, "{label}: i");

    let days = batch
        .column(2)
        .as_any()
        .downcast_ref::<Date32Array>()
        .expect("DATE column")
        .value(0);
    assert_eq!(
        ScalarValue::Date32(Some(days)),
        ScalarValue::try_from_string(expected.d.to_string(), &DataType::Date32)
            .expect("expected date parses"),
        "{label}: d"
    );

    let decimal = batch
        .column(3)
        .as_any()
        .downcast_ref::<Decimal128Array>()
        .expect("DECIMAL column")
        .value(0);
    assert_eq!(decimal, expected.k, "{label}: k, unscaled");

    let text = batch
        .column(4)
        .as_any()
        .downcast_ref::<StringViewArray>()
        .expect("VARCHAR column")
        .value(0);
    assert_eq!(text, expected.s, "{label}: s");
}

/// Check `cases` against a catalog holding `live_files` data files.
///
/// `divergences` names the predicates this backend prunes less than official
/// does, as `(predicate, files this backend lists)`. A divergence may only ever
/// keep *more* files than official; fewer is a lost row and is rejected here
/// rather than recorded. Every case's measured numbers are collected and
/// reported together, so one run says what the whole backend does rather than
/// stopping at the first difference.
async fn assert_official_parity(
    backend: &str,
    provider: Arc<dyn MetadataProvider>,
    cases: &[Case],
    live_files: usize,
    divergences: &[(&str, usize)],
) {
    let snapshot_id = provider.get_current_snapshot().expect("current snapshot");
    let schema_metadata = provider
        .get_schema_by_name("main", snapshot_id)
        .expect("schema lookup")
        .expect("`main` exists");
    let table_metadata = provider
        .get_table_by_name(schema_metadata.schema_id, "filter_pushdown", snapshot_id)
        .expect("table lookup")
        .expect("`filter_pushdown` exists");
    let table_id = table_metadata.table_id;
    let columns = provider
        .get_table_structure(table_id, snapshot_id)
        .expect("table structure");

    let catalog = DuckLakeCatalog::with_snapshot(Arc::clone(&provider), snapshot_id)
        .expect("catalog binds to the snapshot");
    let ctx = SessionContext::new();
    ctx.register_catalog("lake", Arc::new(catalog) as Arc<dyn CatalogProvider>);
    let table = ctx
        .catalog("lake")
        .expect("catalog registered")
        .schema("main")
        .expect("`main` schema")
        .table("filter_pushdown")
        .await
        .expect("table lookup")
        .expect("`filter_pushdown` present");
    let schema = table.schema();

    // `lower_predicate` maps an Arrow field index onto a `column_id` by
    // position, so the two lists have to stay in step.
    assert_eq!(
        schema.fields().len(),
        columns.len(),
        "{backend}: column count"
    );
    for (field, column) in schema.fields().iter().zip(&columns) {
        assert_eq!(
            field.name(),
            &column.column_name,
            "{backend}: schema and catalog columns are positionally aligned"
        );
    }

    let state = ctx.state();
    let df_schema = DFSchema::try_from(schema.as_ref().clone()).expect("schema converts");

    let mut measured: Vec<String> = Vec::new();
    let mut failures: Vec<String> = Vec::new();

    for case in cases {
        let label = format!("{backend}: `{}`", case.sql);
        let expected_files = divergences
            .iter()
            .find(|(sql, _)| *sql == case.sql)
            .map(|(_, files)| *files)
            .unwrap_or(case.official_files);
        assert!(
            expected_files >= case.official_files,
            "{label}: a recorded divergence may only keep more files than official, never fewer"
        );

        // What the listing returns with no statistics filter: every live file.
        // That is the number the assertions below would read if the pushdown
        // did nothing, and official's count is lower for all nine, so none of
        // them can pass with the mechanism disabled.
        let unfiltered = provider
            .get_table_file_metadata_page_filtered(table_id, snapshot_id, None, PAGE_LIMIT, None)
            .unwrap_or_else(|e| panic!("{label}: unfiltered page: {e}"));
        assert_eq!(
            unfiltered.len(),
            live_files,
            "{label}: an unfiltered listing returns every live file"
        );
        assert!(
            case.official_files < live_files,
            "{label}: official prunes at least one file here, so the counts below are not vacuous"
        );

        let predicate = state
            .create_physical_expr((case.predicate)(schema.as_ref()), &df_schema)
            .unwrap_or_else(|e| panic!("{label}: physical expression: {e}"));
        let filter = lower_predicate(&predicate, schema.as_ref(), &columns)
            .unwrap_or_else(|| panic!("{label}: does not lower to a statistics filter"));
        let filtered = provider
            .get_table_file_metadata_page_filtered(
                table_id,
                snapshot_id,
                None,
                PAGE_LIMIT,
                Some(&filter),
            )
            .unwrap_or_else(|e| panic!("{label}: filtered page: {e}"));

        let query = match case.expected {
            Expected::Count(_) => {
                format!(
                    "SELECT COUNT(*) FROM lake.main.filter_pushdown WHERE {}",
                    case.sql
                )
            },
            Expected::Row(_) => {
                format!("SELECT * FROM lake.main.filter_pushdown WHERE {}", case.sql)
            },
        };

        // The same predicate as it arrives from the planner. A form that
        // survives lowering by hand but not from SQL would leave the mechanism
        // unreachable for the queries it exists to speed up.
        let planned: Vec<Arc<dyn PhysicalExpr>> = pushed_down_filters(&ctx, &query)
            .await
            .into_iter()
            .map(|expr| {
                state
                    .create_physical_expr(expr, &df_schema)
                    .unwrap_or_else(|e| panic!("{label}: planned physical expression: {e}"))
            })
            .collect();
        let planned = datafusion::physical_expr::conjunction_opt(planned)
            .unwrap_or_else(|| panic!("{label}: the planner pushes no filter into the scan"));
        let planned_filter = lower_predicate(&planned, schema.as_ref(), &columns)
            .unwrap_or_else(|| panic!("{label}: the planner's own form does not lower"));
        let planned_files = provider
            .get_table_file_metadata_page_filtered(
                table_id,
                snapshot_id,
                None,
                PAGE_LIMIT,
                Some(&planned_filter),
            )
            .unwrap_or_else(|e| panic!("{label}: filtered page from the planned form: {e}"));

        let batches = ctx
            .sql(&query)
            .await
            .unwrap_or_else(|e| panic!("{label}: {query}: {e}"))
            .collect()
            .await
            .unwrap_or_else(|e| panic!("{label}: {query}: {e}"));
        let value = match &case.expected {
            Expected::Count(expected) => {
                let count = count_value(&batches);
                if count != *expected {
                    failures.push(format!(
                        "{label}: query returned {count}, official asserts {expected}"
                    ));
                }
                count.to_string()
            },
            Expected::Row(expected) => {
                assert_official_row(&label, &batches, expected);
                "row".to_string()
            },
        };

        measured.push(format!(
            "  {:<38} official {}  catalog {}  planner {}  value {}",
            case.sql,
            case.official_files,
            filtered.len(),
            planned_files.len(),
            value
        ));
        for (source, count) in [("catalog", filtered.len()), ("planner", planned_files.len())] {
            if count < case.official_files {
                failures.push(format!(
                    "{label}: {source} listed {count} files, fewer than the {} official reads \
                     — a file that may hold matching rows was pruned",
                    case.official_files
                ));
            } else if count != expected_files {
                failures.push(format!(
                    "{label}: {source} listed {count} files, expected {expected_files}"
                ));
            }
        }
    }

    // Visible under `--nocapture`: the whole per-backend table, which is what
    // this test exists to produce.
    println!(
        "{backend} against official DuckLake:\n{}",
        measured.join("\n")
    );
    assert!(
        failures.is_empty(),
        "{backend} against official DuckLake:\n{}\n\n{}",
        measured.join("\n"),
        failures.join("\n")
    );
}

/// A DuckDB catalog: official DuckLake writing its own native metadata format.
#[tokio::test(flavor = "multi_thread")]
async fn duckdb_catalog_matches_official_pushdown() {
    use datafusion_ducklake::DuckdbMetadataProvider;

    let temp = TempDir::new().unwrap();
    let catalog_path = temp.path().join("filter_pushdown.ducklake");
    let data_path = temp.path().join("data");
    let target = format!("ducklake:{}", catalog_path.display());
    let open = || {
        Arc::new(
            DuckdbMetadataProvider::new(catalog_path.to_string_lossy().to_string())
                .expect("duckdb provider"),
        ) as Arc<dyn MetadataProvider>
    };

    build_fixture(&target, &[], &data_path).unwrap();
    assert_official_parity("duckdb", open(), &official_cases(), 3, &[]).await;

    insert_fourth_file(&target, &[], &data_path).unwrap();
    assert_official_parity("duckdb", open(), &not_equal_case(), 4, &[]).await;
}

/// The two predicates SQLite prunes less than official on, and how many files
/// it lists for each.
///
/// `SqliteStatsDialect::try_cast` declines `DECIMAL` outright — SQLite's only
/// fractional numeric is `REAL`, and a decimal constant can carry more
/// significant digits than a double, so two values this engine orders apart can
/// round together and compare equal. A declined cast drops the comparison, so
/// `k = 25.3` contributes no condition and prunes nothing, and
/// `d >= DATE '2011-05-29' AND k < 50000` prunes on `d` alone. Both keep more
/// files than official reads, never fewer, so neither can lose a row — the
/// correctness assertions still hold official's values.
#[cfg(feature = "metadata-sqlite")]
const SQLITE_DIVERGENCES: [(&str, usize); 2] =
    [("k = 25.3", 3), ("d >= DATE '2011-05-29' AND k < 50000", 2)];

/// A SQLite catalog, written by official DuckLake through its `sqlite`
/// extension and read natively by `SqliteMetadataProvider`.
#[cfg(feature = "metadata-sqlite")]
#[tokio::test(flavor = "multi_thread")]
async fn sqlite_catalog_matches_official_pushdown() {
    use datafusion_ducklake::SqliteMetadataProvider;

    let temp = TempDir::new().unwrap();
    let catalog_path = temp.path().join("filter_pushdown.db");
    let data_path = temp.path().join("data");
    let target = format!("ducklake:sqlite:{}", catalog_path.display());
    let url = format!("sqlite:{}", catalog_path.display());

    build_fixture(&target, &["sqlite"], &data_path).unwrap();
    let pool = sqlx::SqlitePool::connect(&url).await.unwrap();
    for statement in MIGRATE_PINNED_DUCKDB_CATALOG {
        // A column a newer extension already wrote is not an error here.
        sqlx::query(statement).execute(&pool).await.ok();
    }
    pool.close().await;

    let provider = Arc::new(SqliteMetadataProvider::new(&url).await.unwrap());
    assert_official_parity(
        "sqlite",
        provider,
        &official_cases(),
        3,
        &SQLITE_DIVERGENCES,
    )
    .await;

    insert_fourth_file(&target, &["sqlite"], &data_path).unwrap();
    let provider = Arc::new(SqliteMetadataProvider::new(&url).await.unwrap());
    assert_official_parity(
        "sqlite",
        provider,
        &not_equal_case(),
        4,
        &SQLITE_DIVERGENCES,
    )
    .await;
}

/// A PostgreSQL catalog. The dialect that has no `TRY_CAST` and aborts the whole
/// listing query on a malformed `CAST`, so its rendering is the furthest from
/// official's.
#[cfg(feature = "metadata-postgres")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
#[tokio::test(flavor = "multi_thread")]
async fn postgres_catalog_matches_official_pushdown() {
    use datafusion_ducklake::PostgresMetadataProvider;
    use testcontainers::runners::AsyncRunner;
    use testcontainers_modules::postgres::Postgres;

    let container = Postgres::default().start().await.unwrap();
    let port = container.get_host_port_ipv4(5432).await.unwrap();
    let url = format!("postgresql://postgres:postgres@127.0.0.1:{port}/postgres");
    let target = format!(
        "ducklake:postgres:host=127.0.0.1 port={port} dbname=postgres \
         user=postgres password=postgres"
    );

    let temp = TempDir::new().unwrap();
    let data_path = temp.path().join("data");

    build_fixture(&target, &["postgres"], &data_path).unwrap();
    let pool = sqlx::PgPool::connect(&url).await.unwrap();
    for statement in MIGRATE_PINNED_DUCKDB_CATALOG {
        // A column a newer extension already wrote is not an error here.
        sqlx::query(statement).execute(&pool).await.ok();
    }
    pool.close().await;

    // No divergences, on any server version. A temporal comparison is made on
    // the encoded text rather than by casting, so it needs neither
    // `pg_input_is_valid` — PostgreSQL 16 and later — nor a regular expression
    // that decides a calendar. `testcontainers-modules` 0.11 starts
    // `postgres:11-alpine`, so this asserts full parity on the oldest server
    // the suite runs against.
    let provider = Arc::new(PostgresMetadataProvider::new(&url).await.unwrap());
    assert_official_parity("postgres", provider, &official_cases(), 3, &[]).await;

    insert_fourth_file(&target, &["postgres"], &data_path).unwrap();
    let provider = Arc::new(PostgresMetadataProvider::new(&url).await.unwrap());
    assert_official_parity("postgres", provider, &not_equal_case(), 4, &[]).await;
}

/// Copy every `ducklake_*` table of the DuckDB catalog at `catalog_path` into
/// the MySQL database `dsn` names, replacing whatever is there.
///
/// MySQL is the one backend that cannot have this fixture written into it
/// directly. DuckLake updates its rollup statistics on commit with an
/// `UPDATE ... JOIN`, and DuckDB's MySQL connector refuses that:
/// "Unsupported operator type HASH_JOIN in UPDATE statement — only simple
/// deletes are supported in the MySQL connector". The first `INSERT` into a
/// MySQL-hosted DuckLake catalog succeeds; the second aborts at commit. That is
/// upstream and on the write path, and reproduces on DuckDB 1.5.5 as well as on
/// the bundled 1.4.1.
///
/// So official DuckLake writes the catalog where it can, into its own DuckDB
/// format, and DuckDB copies the rows across unchanged. Every statistic the
/// MySQL provider then reads is the text official wrote, and `min_value` /
/// `max_value` land in `utf8mb4_0900_ai_ci` `TEXT` columns — byte for byte the
/// shape a direct write produces, which is what makes this backend worth
/// running: that collation is case- and accent-insensitive, and the forced
/// binary collation in the rendered SQL is what keeps `s >= '500023'` from
/// matching the wrong bounds.
fn transport_catalog_to_mysql(catalog_path: &Path, dsn: &str) -> anyhow::Result<()> {
    common::ensure_ducklake_installed();
    common::ensure_extension_installed("mysql");

    let conn = duckdb::Connection::open_in_memory()?;
    conn.execute("LOAD mysql;", [])?;
    conn.execute(
        &format!("ATTACH '{}' AS src (READ_ONLY);", catalog_path.display()),
        [],
    )?;
    conn.execute(&format!("ATTACH '{dsn}' AS dst (TYPE mysql);"), [])?;

    let tables: Vec<String> = {
        let mut statement = conn.prepare(
            "SELECT table_name FROM duckdb_tables()
             WHERE database_name = 'src' ORDER BY table_name",
        )?;
        let rows = statement.query_map([], |row| row.get::<_, String>(0))?;
        rows.collect::<Result<_, _>>()?
    };
    for table in tables {
        conn.execute(&format!("DROP TABLE IF EXISTS dst.{table};"), [])?;
        conn.execute(
            &format!("CREATE TABLE dst.{table} AS SELECT * FROM src.{table};"),
            [],
        )?;
    }
    Ok(())
}

/// Top up a freshly transported catalog and open a provider on it. The
/// transport replaces every table, so the migration has to be reapplied each
/// time.
#[cfg(feature = "metadata-mysql")]
async fn open_mysql(url: &str) -> Arc<datafusion_ducklake::MySqlMetadataProvider> {
    let pool = sqlx::MySqlPool::connect(url).await.unwrap();
    for statement in MIGRATE_PINNED_DUCKDB_CATALOG {
        // A column a newer extension already wrote is not an error here.
        sqlx::query(statement).execute(&pool).await.ok();
    }
    pool.close().await;
    Arc::new(
        datafusion_ducklake::MySqlMetadataProvider::new(url)
            .await
            .unwrap(),
    )
}

/// A MySQL catalog. Its `ducklake_file_column_stats.min_value` / `max_value`
/// are `utf8mb4_0900_ai_ci`, so `s >= '500023'` here is the assertion that the
/// forced binary collation is doing its job.
#[cfg(feature = "metadata-mysql")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
#[tokio::test(flavor = "multi_thread")]
async fn mysql_catalog_matches_official_pushdown() {
    use testcontainers::runners::AsyncRunner;
    use testcontainers_modules::mysql::Mysql;

    let container = Mysql::default().start().await.unwrap();
    let port = container.get_host_port_ipv4(3306).await.unwrap();
    let url = format!("mysql://root@127.0.0.1:{port}/test");
    let dsn = format!("host=127.0.0.1 port={port} user=root database=test");

    let temp = TempDir::new().unwrap();
    let catalog_path = temp.path().join("filter_pushdown.ducklake");
    let data_path = temp.path().join("data");
    let target = format!("ducklake:{}", catalog_path.display());

    build_fixture(&target, &[], &data_path).unwrap();
    transport_catalog_to_mysql(&catalog_path, &dsn).unwrap();
    assert_official_parity("mysql", open_mysql(&url).await, &official_cases(), 3, &[]).await;

    insert_fourth_file(&target, &[], &data_path).unwrap();
    transport_catalog_to_mysql(&catalog_path, &dsn).unwrap();
    assert_official_parity("mysql", open_mysql(&url).await, &not_equal_case(), 4, &[]).await;
}
