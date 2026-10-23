//! Sort keys that are expressions are evaluated with DataFusion's semantics.
//!
//! Where DataFusion computes differently from DuckDB (integer division,
//! overflow, type promotion, function behaviour), rows may be ordered
//! differently inside files than official orders them; query results are
//! unaffected. A key DataFusion cannot plan, or one that fails over the rows
//! being written, leaves the rows in arrival order (a flush: row id order)
//! instead of failing the write. Bare column keys still order rows as
//! official does, checked against the pinned DuckDB CLI.

#![cfg(feature = "write-sqlite")]

use std::path::Path;
use std::process::Command;
use std::sync::Arc;

use arrow::array::AsArray;
use arrow::datatypes::{Int32Type, Int64Type};
use datafusion::prelude::SessionContext;
use datafusion_ducklake::{
    DuckLakeCatalog, MergeOptions, MetadataProvider, MetadataWriter, RewriteOptions,
    SqliteMetadataProvider, SqliteMetadataWriter, execute_ducklake_sql,
};

use crate::inlining_abstraction_tests::{Lake, create, sqlite};

/// Run `statement` against `lake`'s SQLite catalog directly.
async fn raw(lake: &Lake, statement: &str) {
    let url = format!(
        "sqlite:{}?mode=rwc",
        lake.data.parent().unwrap().join("lake.db").display()
    );
    let pool = sqlx::SqlitePool::connect(&url).await.unwrap();
    sqlx::query(sqlx::AssertSqlSafe(statement.to_string()))
        .execute(&pool)
        .await
        .unwrap();
    pool.close().await;
}

/// The ids of `main.t` as its data files hold them: the files in data file
/// id order, each file's rows in their physical order. (A flush keeps each
/// row's row id, so the physical order is what a flush's sort changes.)
fn physical_ids(lake: &Lake) -> Vec<i64> {
    fn find(dir: &Path, name: &std::ffi::OsStr) -> Option<std::path::PathBuf> {
        for entry in std::fs::read_dir(dir).ok()? {
            let path = entry.ok()?.path();
            if path.is_dir() {
                if let Some(found) = find(&path, name) {
                    return Some(found);
                }
            } else if path.file_name() == Some(name) {
                return Some(path);
            }
        }
        None
    }
    let mut files = lake
        .provider
        .get_table_files_for_select(lake.table_id("t"), lake.head())
        .unwrap();
    files.sort_by_key(|file| file.data_file_id);
    let mut ids = Vec::new();
    for file in &files {
        let name = Path::new(&file.file.path).file_name().unwrap().to_owned();
        let path = find(&lake.data, &name).expect("data file on disk");
        let reader = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(
            std::fs::File::open(path).unwrap(),
        )
        .unwrap()
        .build()
        .unwrap();
        for batch in reader {
            let batch = batch.unwrap();
            let column = batch.column_by_name("id").unwrap();
            ids.extend(column.as_primitive::<Int64Type>().values().iter().copied());
        }
    }
    ids
}

/// `main.t(id BIGINT, v VARCHAR)` sorted by `expression` (stored as DuckDB
/// would store it: this crate's `SET SORTED BY` records bare columns only),
/// with `ids` inserted through SQL, each `v` its id's text unless `v` is
/// given. With `flush`, the rows are inlined unsorted (`sort_on_insert` off)
/// and then flushed, so the flush orders them. The ids in physical order.
async fn write_sorted(expression: &str, ids: &[i64], v: Option<&str>, flush: bool) -> Vec<i64> {
    let lake = sqlite()(Some(if flush {
        100
    } else {
        0
    }))
    .await;
    create(&lake, "t").await;
    lake.sql("ALTER TABLE lake.main.t SET SORTED BY (id)")
        .await
        .unwrap();
    raw(
        &lake,
        &format!(
            "UPDATE ducklake_sort_expression SET expression = '{}'",
            expression.replace('\'', "''")
        ),
    )
    .await;
    if flush {
        lake.writer
            .set_table_setting(lake.table_id("t"), "sort_on_insert", "false")
            .unwrap();
    }
    let rows = ids
        .iter()
        .map(|id| format!("({id}, '{}')", v.map_or(id.to_string(), str::to_string)))
        .collect::<Vec<_>>()
        .join(", ");
    lake.sql(&format!("INSERT INTO lake.main.t VALUES {rows}"))
        .await
        .unwrap_or_else(|e| panic!("`{expression}` insert: {e}"));
    if flush {
        assert_eq!(lake.inlined_rows("t"), ids.len(), "`{expression}` inlined");
        lake.catalog()
            .flush_inlined_data(&SessionContext::new().state(), None, None)
            .await
            .unwrap_or_else(|e| panic!("`{expression}` flush: {e}"));
        assert_eq!(lake.inlined_rows("t"), 0, "`{expression}` flushed");
    }
    physical_ids(&lake)
}

/// An expression key orders an insert and a flush by DataFusion's result:
/// `id / 2` of integers is integer division (DuckDB's divides as `DOUBLE`),
/// so `3` ties with `2` and `1` with `0`, ties keeping their arrival order,
/// and the same rows written again order the same way.
#[tokio::test(flavor = "multi_thread")]
async fn expression_keys_order_by_datafusion() {
    let ids = [3, 1, 2, 0, 5, 4];
    for flush in [false, true] {
        let first = write_sorted("(id / 2)", &ids, None, flush).await;
        assert_eq!(first, vec![1, 0, 3, 2, 5, 4], "flush {flush}");
        assert_eq!(
            write_sorted("(id / 2)", &ids, None, flush).await,
            first,
            "flush {flush}: deterministic"
        );
        // DuckDB's `main.` qualifier is dropped so the function resolves.
        assert_eq!(
            write_sorted("main.abs((id - 2))", &ids, None, flush).await,
            vec![2, 3, 1, 0, 4, 5],
            "flush {flush}"
        );
    }
}

/// A key DataFusion cannot plan (a function it lacks, an operator its
/// parser does not know) or that fails over the rows (a division by zero, a
/// failing cast) writes the rows in arrival order (a flush: row id order) instead of
/// failing the write.
#[tokio::test(flavor = "multi_thread")]
async fn unevaluable_expression_keys_keep_arrival_order() {
    let ids = [3, 1, 2];
    for flush in [false, true] {
        for expression in ["main.list_value(id)", "(id ** 2)", "(id / (id - 1))"] {
            assert_eq!(
                write_sorted(expression, &ids, None, flush).await,
                ids,
                "`{expression}`, flush {flush}"
            );
        }
        assert_eq!(
            write_sorted("CAST(v AS BIGINT)", &ids, Some("x"), flush).await,
            ids,
            "failing cast, flush {flush}"
        );
    }
}

/// Compaction orders its output by an expression key as an insert does
/// (DataFusion's semantics), and a key DataFusion cannot plan or that fails
/// over the rows (DuckDB's `//`, which DataFusion's parser does not know, a
/// division by zero) writes them in the order an unsorted compaction writes
/// them: merge, rewrite and checkpoint all complete, every value intact.
#[tokio::test(flavor = "multi_thread")]
async fn compaction_orders_by_expression_keys() {
    let key = |id: i64| -(id / 2);
    for expression in ["-((id / 2))", "-((id // 2))", "main.list_value(id)", "(id / (id - 1))"] {
        let lake = sqlite()(Some(0)).await;
        create(&lake, "t").await;
        lake.sql("ALTER TABLE lake.main.t SET SORTED BY (id)")
            .await
            .unwrap();
        raw(
            &lake,
            &format!(
                "UPDATE ducklake_sort_expression SET expression = '{}'",
                expression.replace('\'', "''")
            ),
        )
        .await;
        // Written one file per insert, each in arrival order: `sort_on_insert`
        // off orders nothing when nothing inlines.
        lake.writer
            .set_table_setting(lake.table_id("t"), "sort_on_insert", "false")
            .unwrap();
        for rows in ["(3, '3'), (1, '1'), (2, '2')", "(0, '0'), (5, '5'), (4, '4')"] {
            lake.sql(&format!("INSERT INTO lake.main.t VALUES {rows}"))
                .await
                .unwrap();
        }
        assert_eq!(physical_ids(&lake), [3, 1, 2, 0, 5, 4], "`{expression}`");

        let (ctx, table) = lake.table("t").await;
        let merged = table
            .merge_adjacent_files(&ctx.state(), MergeOptions::default())
            .await
            .unwrap_or_else(|e| panic!("`{expression}` merge: {e}"));
        assert_eq!(merged.files_created, 1, "`{expression}`");
        let ids = physical_ids(&lake);
        if expression == "-((id / 2))" {
            assert!(
                ids.windows(2).all(|pair| key(pair[0]) <= key(pair[1])),
                "`{expression}` merge orders by the key: {ids:?}"
            );
        } else {
            // Rowid-adjacent sources, written unsorted in source order.
            assert_eq!(ids, [3, 1, 2, 0, 5, 4], "`{expression}`");
        }

        lake.sql("DELETE FROM lake.main.t WHERE id IN (2, 5)")
            .await
            .unwrap();
        let (ctx, table) = lake.table("t").await;
        let rewritten = table
            .rewrite_data_files(
                &ctx.state(),
                RewriteOptions {
                    delete_threshold: 0.0,
                    data_file_ids: None,
                },
            )
            .await
            .unwrap_or_else(|e| panic!("`{expression}` rewrite: {e}"));
        assert_eq!(rewritten.files_created, 1, "`{expression}`");
        let ids = physical_ids(&lake);
        if expression == "-((id / 2))" {
            assert!(
                ids.windows(2).all(|pair| key(pair[0]) <= key(pair[1])),
                "`{expression}` rewrite orders by the key: {ids:?}"
            );
        }

        lake.sql("INSERT INTO lake.main.t VALUES (7, '7'), (6, '6')")
            .await
            .unwrap();
        lake.sql("DELETE FROM lake.main.t WHERE id = 7")
            .await
            .unwrap();
        let ctx = SessionContext::new();
        lake.catalog()
            .checkpoint(&ctx.state())
            .await
            .unwrap_or_else(|e| panic!("`{expression}` checkpoint: {e}"));
        let mut rows = lake.rows("t").await;
        rows.sort();
        assert_eq!(rows.len(), 5, "`{expression}`: {rows:?}");
        for id in [0, 1, 3, 4, 6] {
            assert!(
                rows.iter().any(|row| row.ends_with(&format!(",{id},{id}"))),
                "`{expression}`: {id} in {rows:?}"
            );
        }
    }
}

/// Run `sql` with the DuckDB CLI: its output lines, or its error.
fn cli(sql: &str) -> Result<Vec<String>, String> {
    let out = Command::new("duckdb")
        .args(["-csv", "-noheader", ":memory:", "-c", sql])
        .output()
        .unwrap();
    if out.status.success() {
        Ok(String::from_utf8_lossy(&out.stdout)
            .lines()
            .map(str::to_string)
            .collect())
    } else {
        Err(String::from_utf8_lossy(&out.stderr).to_string())
    }
}

fn attach(catalog: &Path, data: &Path) -> String {
    format!(
        "LOAD ducklake; LOAD sqlite; ATTACH 'ducklake:sqlite:{}' AS lake (DATA_PATH '{}/');",
        catalog.display(),
        data.display()
    )
}

const COLUMNS: &str = "id INTEGER, ti TINYINT, ubi UBIGINT, d DOUBLE, s VARCHAR";

/// Rows over [`COLUMNS`], every value a string literal (`NULL` bare), which
/// DuckDB and DataFusion both cast to the column's type: ties, `NULL`s,
/// `nan`, `-0.0`, infinities and an unsigned value past the signed range.
const ROWS: &str = "('1', '3', '10', '2.5', 'b'), ('2', '-2', '18000000000000000000', '-0.0', 'a'), \
                    ('3', '0', '0', 'nan', NULL), ('4', NULL, '1', 'inf', 'B'), \
                    ('5', '3', '9223372036854775808', '0.0', 'a'), ('6', '-1', NULL, '-inf', ''), \
                    ('7', '0', '2', NULL, 'b'), ('8', '-2', '2', '-nan', 'a')";

/// Bare column keys order an insert as official's does: two tables sorted by
/// `keys`, one written by the DuckDB CLI and one by this crate, hold the same
/// ids in `rowid` order.
#[tokio::test(flavor = "multi_thread")]
async fn bare_column_keys_order_inserts_as_the_duckdb_cli() {
    let Ok(version) = Command::new("duckdb").arg("--version").output() else {
        eprintln!("skipping: no `duckdb` on PATH (CI installs the pinned CLI)");
        return;
    };
    let version = String::from_utf8_lossy(&version.stdout).to_string();
    assert!(
        version.contains("v1.5.5"),
        "the oracle is the pinned DuckDB CLI (CI installs v1.5.5), found: {version}"
    );
    for keys in [
        "ti",
        "ti DESC NULLS FIRST, id",
        "ubi",
        "d",
        "d DESC",
        "s, id DESC",
        "S NULLS FIRST",
        "\"ti\" DESC, d",
    ] {
        let temp = tempfile::TempDir::new().unwrap();
        let catalog = temp.path().join("c.db");
        let data = temp.path().join("data");
        std::fs::create_dir_all(&data).unwrap();
        let url = format!("sqlite:{}?mode=rwc", catalog.display());
        // This crate creates the catalog; the DuckDB CLI writes it first.
        crate::common::crate_sqlite_catalog(&url, &data).await;
        cli(&format!(
            "{} CALL lake.set_option('data_inlining_row_limit', 0); \
             CREATE TABLE lake.o({COLUMNS}); ALTER TABLE lake.o SET SORTED BY ({keys}); \
             CREATE TABLE lake.c({COLUMNS}); ALTER TABLE lake.c SET SORTED BY ({keys}); \
             INSERT INTO lake.o VALUES {ROWS};",
            attach(&catalog, &data)
        ))
        .unwrap_or_else(|e| panic!("`{keys}` official: {e}"));
        let official = cli(&format!(
            "{} SELECT id FROM lake.main.o ORDER BY rowid;",
            attach(&catalog, &data)
        ))
        .unwrap();

        let writer: Arc<dyn MetadataWriter> =
            Arc::new(SqliteMetadataWriter::new_with_init(&url).await.unwrap());
        let provider: Arc<dyn MetadataProvider> =
            Arc::new(SqliteMetadataProvider::new(&url).await.unwrap());
        let catalog_of =
            || DuckLakeCatalog::with_writer(Arc::clone(&provider), Arc::clone(&writer)).unwrap();
        let ctx = SessionContext::new();
        DuckLakeCatalog::register(&ctx, "lake", catalog_of());
        let insert = format!("INSERT INTO lake.main.c VALUES {ROWS}");
        execute_ducklake_sql(&ctx, &catalog_of(), &insert)
            .await
            .unwrap()
            .collect()
            .await
            .unwrap_or_else(|e| panic!("`{keys}` ours: {e}"));
        // Read by this crate: the DuckDB extension cannot read a snapshot this
        // crate's writer recorded.
        let ctx = SessionContext::new();
        DuckLakeCatalog::register(&ctx, "lake", catalog_of().with_row_lineage(true));
        let batches = ctx
            .sql("SELECT id FROM lake.main.c ORDER BY rowid")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let ours: Vec<String> = batches
            .iter()
            .flat_map(|batch| {
                batch
                    .column(0)
                    .as_primitive::<Int32Type>()
                    .iter()
                    .map(|id| id.unwrap().to_string())
                    .collect::<Vec<_>>()
            })
            .collect();
        assert_eq!(ours, official, "`{keys}`");
    }
}
