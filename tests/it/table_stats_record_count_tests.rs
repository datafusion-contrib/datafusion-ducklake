//! `ducklake_table_stats.record_count` against official DuckLake (DuckDB 1.5.5).
//!
//! Official keeps `record_count` as the gross count of rows ever inserted: only
//! inserts raise it, and deletes, truncates, flushes and compaction leave it.
//! DuckDB answers `MIN`/`MAX` from the table-wide column bounds, which only
//! widen, exactly when `record_count` equals the live row count
//! (`ducklake_scan.cpp`, `DuckLakeGetPartitionStats`). A writer that lowers
//! `record_count` when rows leave therefore makes DuckDB answer from bounds
//! wider than the live rows.
//!
//! Each scenario applies one history with official DuckLake to its own catalog
//! and with this crate to another, then requires the same `record_count` and
//! the same answers from DuckDB over both catalogs and from this crate.

#![cfg(all(feature = "write", feature = "metadata-duckdb"))]

use std::path::Path;
use std::sync::Arc;

use arrow::array::{Int32Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::prelude::SessionContext;
use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeTable, DuckLakeTableWriter, DuckLakeWriteOptions, MergeOptions,
    MetadataProvider, MetadataWriter, RewriteOptions, execute_ducklake_sql,
};
use object_store::local::LocalFileSystem;

use crate::inlined_mutation_differential_tests::{
    Metadata, attach_lake, duckdb_connection, official, query_lines,
};

/// One step of a history.
#[derive(Clone, Copy)]
enum Op {
    /// Append rows (`INSERT` for official).
    Append(&'static [i32]),
    /// Append one NULL row.
    AppendNull,
    /// Set the data inlining row limit.
    Limit(usize),
    /// Rewrite the crate's `ducklake_table_column_stats` rows for `a` to the
    /// shape earlier releases of this crate stored when a live file of the
    /// column held only NULLs: neither bound, meaning unknown. Official does
    /// nothing; its own rows already cover the values.
    LegacyNullBounds,
    /// Replace the table's rows (`DELETE` and `INSERT` in one transaction for
    /// official).
    Replace(&'static [i32]),
    /// A statement both run.
    Sql(&'static str),
    /// `merge_adjacent_files` / `ducklake_merge_adjacent_files`.
    Merge,
    /// `rewrite_data_files` / `ducklake_rewrite_data_files`, threshold 0.
    Rewrite,
    /// Remember the current snapshot for [`Op::RetireAppends`].
    MarkBase,
    /// `retire_appends_since` the marked snapshot; official runs the statement,
    /// which removes the same rows.
    RetireAppends(&'static str),
}

struct Scenario {
    name: &'static str,
    limit: usize,
    ops: &'static [Op],
}

const QUERIES: [&str; 4] = [
    "SELECT min(a), max(a), count(*) FROM lake.main.t",
    "SELECT count(*) FROM lake.main.t WHERE a = 5",
    "SELECT count(*) FROM lake.main.t WHERE a = 50",
    "SELECT count(*) FROM lake.main.t WHERE a >= 100",
];

/// The statement [`Op::LegacyNullBounds`] runs on the crate's metadata.
const LEGACY_NULL_BOUNDS: &str = "UPDATE ducklake_table_column_stats \
     SET min_value = NULL, max_value = NULL, contains_nan = NULL \
     WHERE table_id = (SELECT table_id FROM ducklake_table \
                       WHERE table_name = 't' AND end_snapshot IS NULL)";

fn scenarios() -> Vec<Scenario> {
    vec![
        Scenario {
            name: "replace then append",
            limit: 0,
            ops: &[Op::Append(&[100, 200]), Op::Replace(&[1, 2]), Op::Append(&[5])],
        },
        Scenario {
            name: "replace inlined then append",
            limit: 10,
            ops: &[Op::Append(&[100, 200]), Op::Replace(&[1, 2]), Op::Append(&[5])],
        },
        Scenario {
            name: "truncate then append",
            limit: 0,
            ops: &[Op::Append(&[100, 200]), Op::Sql("DELETE FROM lake.main.t"), Op::Append(&[5])],
        },
        Scenario {
            name: "delete then append",
            limit: 0,
            ops: &[
                Op::Append(&[100, 200, 1]),
                Op::Sql("DELETE FROM lake.main.t WHERE a >= 100"),
                Op::Append(&[5]),
            ],
        },
        Scenario {
            name: "inlined delete then append",
            limit: 10,
            ops: &[
                Op::Append(&[100, 200, 1]),
                Op::Sql("DELETE FROM lake.main.t WHERE a >= 100"),
                Op::Append(&[5]),
            ],
        },
        Scenario {
            name: "inlined delete, flush, append",
            limit: 10,
            ops: &[
                Op::Append(&[100, 200, 1]),
                Op::Sql("DELETE FROM lake.main.t WHERE a >= 100"),
                Op::Sql("CALL ducklake_flush_inlined_data('lake')"),
                Op::Append(&[5]),
            ],
        },
        Scenario {
            name: "merge after a dropped file",
            limit: 0,
            ops: &[
                Op::Append(&[100]),
                Op::Append(&[1]),
                Op::Append(&[2]),
                Op::Sql("DELETE FROM lake.main.t WHERE a = 100"),
                Op::Merge,
            ],
        },
        Scenario {
            name: "rewrite after delete",
            limit: 0,
            ops: &[
                Op::Append(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                Op::Append(&[50]),
                Op::Sql("DELETE FROM lake.main.t WHERE a >= 8"),
                Op::Rewrite,
            ],
        },
        Scenario {
            name: "retire appends",
            limit: 0,
            ops: &[
                Op::Append(&[1]),
                Op::MarkBase,
                Op::Append(&[200]),
                Op::RetireAppends("DELETE FROM lake.main.t WHERE a = 200"),
            ],
        },
        // A catalog an earlier release wrote, whose roll-up for `a` says
        // nothing about its bounds: the next write must not narrow it to its
        // own values.
        Scenario {
            name: "legacy unknown bounds then append",
            limit: 0,
            ops: &[Op::AppendNull, Op::Append(&[50, 60]), Op::LegacyNullBounds, Op::Append(&[5])],
        },
        Scenario {
            name: "legacy unknown bounds then inlined append",
            limit: 0,
            ops: &[
                Op::AppendNull,
                Op::Append(&[50, 60]),
                Op::LegacyNullBounds,
                Op::Limit(10),
                Op::Append(&[5]),
            ],
        },
        // A roll-up with neither bound over files holding only NULLs means no
        // valid value, as official reads it: the next write takes its bounds.
        Scenario {
            name: "all-NULL column then append",
            limit: 0,
            ops: &[Op::AppendNull, Op::AppendNull, Op::Append(&[5])],
        },
    ]
}

/// Runs a statement on the crate's metadata database.
type MetadataExec =
    Box<dyn Fn(&'static str) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()>>>>;

/// The crate's writer and provider for one backend.
struct Side {
    writer: Arc<dyn MetadataWriter>,
    provider: Arc<dyn MetadataProvider>,
    /// `None` where the crate holds the only handle to its metadata.
    metadata_exec: Option<MetadataExec>,
}

fn batch(values: &[i32]) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, true)])),
        vec![Arc::new(Int32Array::from(values.to_vec()))],
    )
    .unwrap()
}

fn values(values: &[i32]) -> String {
    values
        .iter()
        .map(|v| format!("({v})"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn official_statements(scenario: &Scenario) -> Vec<String> {
    let mut out = vec![
        format!(
            "CALL lake.set_option('data_inlining_row_limit', {})",
            scenario.limit
        ),
        "CREATE TABLE lake.main.t (a INTEGER)".to_string(),
    ];
    for op in scenario.ops {
        match op {
            Op::Append(v) => out.push(format!("INSERT INTO lake.main.t VALUES {}", values(v))),
            Op::AppendNull => out.push("INSERT INTO lake.main.t VALUES (NULL)".to_string()),
            Op::LegacyNullBounds => {},
            Op::Limit(limit) => out.push(format!(
                "CALL lake.set_option('data_inlining_row_limit', {limit})"
            )),
            Op::Replace(v) => out.push(format!(
                "BEGIN; DELETE FROM lake.main.t; INSERT INTO lake.main.t VALUES {}; COMMIT",
                values(v)
            )),
            Op::Sql(sql) | Op::RetireAppends(sql) => out.push((*sql).to_string()),
            Op::Merge => out.push("CALL ducklake_merge_adjacent_files('lake')".to_string()),
            Op::Rewrite => out.push(
                "CALL ducklake_rewrite_data_files('lake', delete_threshold => 0)".to_string(),
            ),
            Op::MarkBase => {},
        }
    }
    out
}

fn table_id(side: &Side) -> i64 {
    let snapshot = side.provider.get_current_snapshot().unwrap();
    let schema = side
        .provider
        .get_schema_by_name("main", snapshot)
        .unwrap()
        .unwrap();
    side.provider
        .get_table_by_name(schema.schema_id, "t", snapshot)
        .unwrap()
        .unwrap()
        .table_id
}

async fn writable_table(side: &Side) -> (SessionContext, DuckLakeTable) {
    let ctx = SessionContext::new();
    let catalog =
        DuckLakeCatalog::with_writer(Arc::clone(&side.provider), Arc::clone(&side.writer)).unwrap();
    ctx.register_catalog("lake", Arc::new(catalog));
    let provider = ctx
        .catalog("lake")
        .unwrap()
        .schema("main")
        .unwrap()
        .table("t")
        .await
        .unwrap()
        .unwrap();
    let table = (provider.as_ref() as &dyn std::any::Any)
        .downcast_ref::<DuckLakeTable>()
        .expect("provider is a DuckLakeTable")
        .clone();
    (ctx, table)
}

async fn run_crate_sql(side: &Side, statement: &str) {
    let ctx = SessionContext::new();
    let catalog =
        DuckLakeCatalog::with_writer(Arc::clone(&side.provider), Arc::clone(&side.writer)).unwrap();
    DuckLakeCatalog::register(&ctx, "lake", catalog);
    let handle =
        DuckLakeCatalog::with_writer(Arc::clone(&side.provider), Arc::clone(&side.writer)).unwrap();
    execute_ducklake_sql(&ctx, &handle, statement)
        .await
        .unwrap_or_else(|e| panic!("crate `{statement}`: {e}"))
        .collect()
        .await
        .unwrap_or_else(|e| panic!("crate `{statement}`: {e}"));
}

async fn apply_crate(side: &Side, scenario: &Scenario) {
    // A writer without global settings (MySQL) takes the limit at table scope
    // once the table exists.
    let mut table_scoped = side
        .writer
        .set_global_setting("data_inlining_row_limit", &scenario.limit.to_string())
        .is_err();
    let writer = |limit: usize| {
        DuckLakeTableWriter::new(Arc::clone(&side.writer), Arc::new(LocalFileSystem::new()))
            .unwrap()
            .with_options(&DuckLakeWriteOptions::default().with_data_inlining_row_limit(limit))
    };
    let mut limit = scenario.limit;
    let mut base = None;
    for op in scenario.ops {
        match op {
            Op::AppendNull => {
                let batch = RecordBatch::try_new(
                    Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, true)])),
                    vec![Arc::new(Int32Array::from(vec![None::<i32>]))],
                )
                .unwrap();
                writer(limit)
                    .append_table("main", "t", &[batch])
                    .await
                    .unwrap();
            },
            Op::LegacyNullBounds => {
                let exec = side
                    .metadata_exec
                    .as_ref()
                    .expect("the backend runs metadata statements");
                exec(LEGACY_NULL_BOUNDS).await;
            },
            Op::Limit(new) => {
                limit = *new;
                if side
                    .writer
                    .set_global_setting("data_inlining_row_limit", &limit.to_string())
                    .is_err()
                {
                    side.writer
                        .set_table_setting(
                            table_id(side),
                            "data_inlining_row_limit",
                            &limit.to_string(),
                        )
                        .unwrap();
                }
            },
            Op::Append(v) => {
                writer(limit)
                    .append_table("main", "t", &[batch(v)])
                    .await
                    .unwrap();
            },
            Op::Replace(v) => {
                writer(limit)
                    .write_table("main", "t", &[batch(v)])
                    .await
                    .unwrap();
            },
            Op::Sql(sql) => run_crate_sql(side, sql).await,
            Op::Merge => {
                let (ctx, table) = writable_table(side).await;
                let opts = MergeOptions {
                    min_file_size: 0,
                    ..MergeOptions::default()
                };
                let merged = table
                    .merge_adjacent_files(&ctx.state(), opts)
                    .await
                    .unwrap();
                assert!(merged.did_work(), "{}: nothing merged", scenario.name);
            },
            Op::Rewrite => {
                let (ctx, table) = writable_table(side).await;
                let opts = RewriteOptions {
                    delete_threshold: 0.0,
                    ..RewriteOptions::default()
                };
                let rewritten = table.rewrite_data_files(&ctx.state(), opts).await.unwrap();
                assert!(rewritten.did_work(), "{}: nothing rewritten", scenario.name);
            },
            Op::MarkBase => base = Some(side.provider.get_current_snapshot().unwrap()),
            Op::RetireAppends(_) => {
                side.writer
                    .retire_appends_since(table_id(side), base.expect("MarkBase first"))
                    .unwrap()
                    .expect("appends retired");
            },
        }
        if table_scoped {
            side.writer
                .set_table_setting(
                    table_id(side),
                    "data_inlining_row_limit",
                    &scenario.limit.to_string(),
                )
                .unwrap();
            table_scoped = false;
        }
    }
}

/// Answers of [`QUERIES`] by this crate, cells joined by `,`.
async fn crate_answers(side: &Side) -> Vec<Vec<String>> {
    let ctx = SessionContext::new();
    let snapshot = side.provider.get_current_snapshot().unwrap();
    let catalog = DuckLakeCatalog::with_snapshot(Arc::clone(&side.provider), snapshot).unwrap();
    ctx.register_catalog("lake", Arc::new(catalog));
    let mut out = Vec::new();
    for sql in QUERIES {
        let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
        let mut lines = Vec::new();
        for batch in &batches {
            for row in 0..batch.num_rows() {
                lines.push(
                    (0..batch.num_columns())
                        .map(|c| {
                            let array = batch.column(c);
                            if arrow::array::Array::is_null(array, row) {
                                "NULL".to_string()
                            } else {
                                arrow::util::display::array_value_to_string(array, row).unwrap()
                            }
                        })
                        .collect::<Vec<_>>()
                        .join(","),
                );
            }
        }
        out.push(lines);
    }
    out
}

/// Answers of [`QUERIES`] by DuckDB over the lake `metadata` describes.
fn duckdb_answers(metadata: &Metadata, data: &Path) -> Vec<Vec<String>> {
    let connection = duckdb_connection();
    attach_lake(&connection, metadata, data, true);
    QUERIES
        .iter()
        .map(|sql| query_lines(&connection, sql))
        .collect()
}

/// `contains_null, contains_nan, min_value, max_value` of the table-wide
/// statistics of `a`, booleans as `0`/`1`.
fn column_stats(metadata: &Metadata) -> Vec<String> {
    let connection = duckdb_connection();
    metadata.attach_raw(&connection);
    query_lines(
        &connection,
        "SELECT s.contains_null, s.contains_nan, s.min_value, s.max_value \
         FROM m.ducklake_table_column_stats s \
         JOIN m.ducklake_table t ON t.table_id = s.table_id \
         WHERE t.table_name = 't' AND t.end_snapshot IS NULL",
    )
    .into_iter()
    .map(|line| line.replace("false", "0").replace("true", "1"))
    .collect()
}

fn record_count(metadata: &Metadata) -> String {
    let connection = duckdb_connection();
    metadata.attach_raw(&connection);
    query_lines(
        &connection,
        "SELECT s.record_count FROM m.ducklake_table_stats s \
         JOIN m.ducklake_table t ON t.table_id = s.table_id \
         WHERE t.table_name = 't' AND t.end_snapshot IS NULL",
    )
    .join(",")
}

/// How a backend's catalog is checked once the crate's handles are dropped.
struct Check<'a> {
    /// The crate's catalog, for DuckDB to read; `None` when DuckDB has no
    /// reader for the layout.
    duckdb_reads: Option<&'a Metadata>,
    /// The crate's catalog metadata, for its `record_count`.
    raw: &'a Metadata,
    crate_data: &'a Path,
}

/// Run `scenario` with official on SQLite metadata in `dir` and with the
/// crate on `side`, then compare, returning each difference. `before_duckdb`
/// runs between the crate's writes and DuckDB's reads of the crate's catalog.
async fn run(
    scenario: &Scenario,
    dir: &Path,
    side: Side,
    check: Check<'_>,
    before_duckdb: impl std::future::Future<Output = ()>,
) -> Vec<String> {
    let official_metadata = Metadata::Sqlite(dir.join("official.db"));
    let official_data = dir.join("official_data");
    std::fs::create_dir_all(&official_data).unwrap();
    official(
        &official_metadata,
        &official_data,
        &official_statements(scenario),
    );
    let expected = duckdb_answers(&official_metadata, &official_data);

    let legacy = scenario
        .ops
        .iter()
        .any(|op| matches!(op, Op::LegacyNullBounds));
    if legacy && side.metadata_exec.is_none() {
        return Vec::new();
    }
    apply_crate(&side, scenario).await;
    let mut failures = Vec::new();
    let got = crate_answers(&side).await;
    if got != expected {
        failures.push(format!(
            "{}: the crate answers {got:?}, official {expected:?}",
            scenario.name
        ));
    }
    drop(side);
    before_duckdb.await;

    if let Some(metadata) = check.duckdb_reads {
        let got = duckdb_answers(metadata, check.crate_data);
        if got != expected {
            failures.push(format!(
                "{}: DuckDB answers {got:?} over the crate's catalog, {expected:?} over official's",
                scenario.name
            ));
        }
    }
    // Official narrows `record_count` (with the bounds) after a rewrite leaves
    // the table free of deletes; this crate keeps both as they were.
    let rewrite = scenario.ops.iter().any(|op| matches!(op, Op::Rewrite));
    if !rewrite {
        let (got, want) = (column_stats(check.raw), column_stats(&official_metadata));
        // A legacy roll-up may also be dropped: no row is unknown.
        if got != want && !(legacy && got.is_empty()) {
            failures.push(format!(
                "{}: ducklake_table_column_stats is {got:?}, official's {want:?}",
                scenario.name
            ));
        }
    }
    if !rewrite {
        let (got, want) = (record_count(check.raw), record_count(&official_metadata));
        if got != want {
            failures.push(format!(
                "{}: ducklake_table_stats.record_count is {got}, official's {want}",
                scenario.name
            ));
        }
    }
    failures
}

#[cfg(feature = "write-sqlite")]
#[tokio::test(flavor = "multi_thread")]
async fn sqlite_matches_official() {
    use datafusion_ducklake::{SqliteMetadataProvider, SqliteMetadataWriter};
    let mut failures = Vec::new();
    for scenario in scenarios() {
        let temp = tempfile::TempDir::new().unwrap();
        let dir = temp.path();
        let path = dir.join("crate.db");
        let data = dir.join("crate_data");
        std::fs::create_dir_all(&data).unwrap();
        let url = format!("sqlite:{}?mode=rwc", path.display());
        let writer = SqliteMetadataWriter::new_with_init(&url).await.unwrap();
        writer.set_data_path(data.to_str().unwrap()).unwrap();
        let exec_url = url.clone();
        let side = Side {
            writer: Arc::new(writer),
            provider: Arc::new(SqliteMetadataProvider::new(&url).await.unwrap()),
            metadata_exec: Some(Box::new(move |sql| {
                let url = exec_url.clone();
                Box::pin(async move {
                    let pool = sqlx::SqlitePool::connect(&url).await.unwrap();
                    sqlx::query(sql).execute(&pool).await.unwrap();
                    pool.close().await;
                })
            })),
        };
        let metadata = Metadata::Sqlite(path.clone());
        failures.extend(
            run(
                &scenario,
                dir,
                side,
                Check {
                    duckdb_reads: Some(&metadata),
                    raw: &metadata,
                    crate_data: &data,
                },
                crate::inlined_mutation_differential_tests::make_sqlite_catalog_duckdb_readable(
                    &path,
                ),
            )
            .await,
        );
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[cfg(feature = "write-duckdb")]
#[tokio::test(flavor = "multi_thread")]
async fn duckdb_matches_official() {
    use datafusion_ducklake::DuckdbMetadataWriter;
    let mut failures = Vec::new();
    for scenario in scenarios() {
        let temp = tempfile::TempDir::new().unwrap();
        let dir = temp.path();
        let path = dir.join("crate.duckdb");
        let data = dir.join("crate_data");
        std::fs::create_dir_all(&data).unwrap();
        let writer = DuckdbMetadataWriter::new_with_init(path.display().to_string()).unwrap();
        writer.set_data_path(data.to_str().unwrap()).unwrap();
        let provider = writer.metadata_provider();
        let side = Side {
            writer: Arc::new(writer),
            provider: Arc::new(provider),
            metadata_exec: None,
        };
        let metadata = Metadata::Duckdb(path.clone());
        failures.extend(
            run(
                &scenario,
                dir,
                side,
                Check {
                    duckdb_reads: Some(&metadata),
                    raw: &metadata,
                    crate_data: &data,
                },
                async {},
            )
            .await,
        );
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[cfg(feature = "write-mysql")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
#[tokio::test(flavor = "multi_thread")]
async fn mysql_matches_official() {
    use datafusion_ducklake::{MySqlMetadataProvider, MySqlMetadataWriter};
    use testcontainers::runners::AsyncRunner;
    use testcontainers_modules::mysql::Mysql;
    let container = Mysql::default().start().await.unwrap();
    let port = container.get_host_port_ipv4(3306).await.unwrap();
    let admin = sqlx::MySqlPool::connect(&format!("mysql://root@127.0.0.1:{port}/test"))
        .await
        .unwrap();
    let mut failures = Vec::new();
    for (index, scenario) in scenarios().into_iter().enumerate() {
        let temp = tempfile::TempDir::new().unwrap();
        let dir = temp.path();
        let database = format!("stats_{index}");
        sqlx::query(sqlx::AssertSqlSafe(format!("CREATE DATABASE {database}")))
            .execute(&admin)
            .await
            .unwrap();
        let url = format!("mysql://root@127.0.0.1:{port}/{database}");
        let data = dir.join("crate_data");
        std::fs::create_dir_all(&data).unwrap();
        let writer = MySqlMetadataWriter::new_with_init(&url).await.unwrap();
        writer.set_data_path(data.to_str().unwrap()).unwrap();
        let exec_url = url.clone();
        let side = Side {
            writer: Arc::new(writer),
            provider: Arc::new(MySqlMetadataProvider::new(&url).await.unwrap()),
            metadata_exec: Some(Box::new(move |sql| {
                let url = exec_url.clone();
                Box::pin(async move {
                    let pool = sqlx::MySqlPool::connect(&url).await.unwrap();
                    sqlx::query(sql).execute(&pool).await.unwrap();
                    pool.close().await;
                })
            })),
        };
        let metadata = Metadata::Mysql {
            host: "127.0.0.1".to_string(),
            port,
            database,
        };
        failures.extend(
            run(
                &scenario,
                dir,
                side,
                // DuckDB 1.5.5 cannot attach this crate's MySQL catalogs (their
                // `ducklake_snapshot` lacks `next_catalog_id`); its MIN/MAX shortcut
                // depends on `record_count` alone, which is compared.
                Check {
                    duckdb_reads: None,
                    raw: &metadata,
                    crate_data: &data,
                },
                async {},
            )
            .await,
        );
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[cfg(feature = "write-postgres")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
#[tokio::test(flavor = "multi_thread")]
async fn postgres_single_catalog_matches_official() {
    use datafusion_ducklake::{PostgresMetadataProvider, PostgresSingleCatalogMetadataWriter};
    use testcontainers::runners::AsyncRunner;
    use testcontainers_modules::postgres::Postgres;
    let container = Postgres::default().start().await.unwrap();
    let port = container.get_host_port_ipv4(5432).await.unwrap();
    let admin = sqlx::PgPool::connect(&format!(
        "postgresql://postgres:postgres@127.0.0.1:{port}/postgres"
    ))
    .await
    .unwrap();
    let mut failures = Vec::new();
    for (index, scenario) in scenarios().into_iter().enumerate() {
        // The single-catalog writer appends and replaces; it has no `DELETE`,
        // compaction or `retire_appends_since`.
        if !scenario.ops.iter().all(|op| {
            matches!(
                op,
                Op::Append(_)
                    | Op::AppendNull
                    | Op::Replace(_)
                    | Op::Limit(_)
                    | Op::LegacyNullBounds
            )
        }) {
            continue;
        }
        let temp = tempfile::TempDir::new().unwrap();
        let dir = temp.path();
        let database = format!("stats_{index}");
        sqlx::query(sqlx::AssertSqlSafe(format!("CREATE DATABASE {database}")))
            .execute(&admin)
            .await
            .unwrap();
        let url = format!("postgresql://postgres:postgres@127.0.0.1:{port}/{database}");
        let data = dir.join("crate_data");
        std::fs::create_dir_all(&data).unwrap();
        let writer = PostgresSingleCatalogMetadataWriter::new_with_init(&url)
            .await
            .unwrap();
        writer.set_data_path(data.to_str().unwrap()).unwrap();
        let exec_url = url.clone();
        let side = Side {
            writer: Arc::new(writer),
            provider: Arc::new(PostgresMetadataProvider::new(&url).await.unwrap()),
            metadata_exec: Some(Box::new(move |sql| {
                let url = exec_url.clone();
                Box::pin(async move {
                    let pool = sqlx::PgPool::connect(&url).await.unwrap();
                    sqlx::query(sql).execute(&pool).await.unwrap();
                    pool.close().await;
                })
            })),
        };
        let metadata = Metadata::Postgres(format!(
            "host=127.0.0.1 port={port} user=postgres password=postgres dbname={database}"
        ));
        failures.extend(
            run(
                &scenario,
                dir,
                side,
                // As for MySQL, DuckDB cannot attach this crate's PostgreSQL
                // catalogs; `record_count` is compared.
                Check {
                    duckdb_reads: None,
                    raw: &metadata,
                    crate_data: &data,
                },
                async {},
            )
            .await,
        );
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[cfg(feature = "multicatalog-postgres")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
#[tokio::test(flavor = "multi_thread")]
async fn multicatalog_matches_official() {
    use datafusion_ducklake::{MulticatalogManager, MulticatalogProvider, PostgresMetadataWriter};
    use testcontainers::runners::AsyncRunner;
    use testcontainers_modules::postgres::Postgres;
    let container = Postgres::default().start().await.unwrap();
    let port = container.get_host_port_ipv4(5432).await.unwrap();
    let admin = sqlx::PgPool::connect(&format!(
        "postgresql://postgres:postgres@127.0.0.1:{port}/postgres"
    ))
    .await
    .unwrap();
    let mut failures = Vec::new();
    for (index, scenario) in scenarios().into_iter().enumerate() {
        let temp = tempfile::TempDir::new().unwrap();
        let dir = temp.path();
        let database = format!("stats_{index}");
        sqlx::query(sqlx::AssertSqlSafe(format!("CREATE DATABASE {database}")))
            .execute(&admin)
            .await
            .unwrap();
        let url = format!("postgresql://postgres:postgres@127.0.0.1:{port}/{database}");
        let data = dir.join("crate_data");
        std::fs::create_dir_all(&data).unwrap();
        let pool = sqlx::PgPool::connect(&url).await.unwrap();
        datafusion_ducklake::initialize_multicatalog_schema(&pool)
            .await
            .unwrap();
        let catalog = MulticatalogManager::new(pool.clone())
            .create_catalog("lake")
            .await
            .unwrap();
        let writer = PostgresMetadataWriter::with_pool(pool.clone(), catalog)
            .await
            .unwrap();
        writer.set_data_path(data.to_str().unwrap()).unwrap();
        let exec_url = url.clone();
        let side = Side {
            writer: Arc::new(writer),
            provider: Arc::new(MulticatalogProvider::with_pool(pool, "lake").await.unwrap()),
            metadata_exec: Some(Box::new(move |sql| {
                let url = exec_url.clone();
                Box::pin(async move {
                    let pool = sqlx::PgPool::connect(&url).await.unwrap();
                    sqlx::query(sql).execute(&pool).await.unwrap();
                    pool.close().await;
                })
            })),
        };
        // The multi-catalog layout has no official reader.
        let metadata = Metadata::Postgres(format!(
            "host=127.0.0.1 port={port} user=postgres password=postgres dbname={database}"
        ));
        failures.extend(
            run(
                &scenario,
                dir,
                side,
                Check {
                    duckdb_reads: None,
                    raw: &metadata,
                    crate_data: &data,
                },
                async {},
            )
            .await,
        );
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
