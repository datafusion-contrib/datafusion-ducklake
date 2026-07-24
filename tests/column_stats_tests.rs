//! End-to-end validation of the write-side column-statistics pipeline: write
//! real Parquet through the crate, then read `ducklake_file_column_stats` /
//! `ducklake_table_column_stats` back out of the SQLite catalog and assert the
//! stored values are byte-identical to DuckDB's canonical encodings.
//!
//! This exercises the whole chain — Parquet footer harvest
//! (`stats_collect`) → DuckDB-canonical encoding (`stats_encode`) → per-backend
//! persistence — that the in-crate unit tests only cover in pieces.
#![cfg(feature = "write-sqlite")]
// 3.14 etc. below are deliberate float test data, not approximations of π.
#![allow(clippy::approx_constant)]

use std::sync::Arc;

use arrow::array::{
    BooleanArray, Date32Array, Decimal128Array, Float64Array, Int32Array, Int64Array, StringArray,
    TimestampMicrosecondArray,
};
use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use arrow::record_batch::RecordBatch;
use datafusion_ducklake::{DuckLakeTableWriter, MetadataWriter, SqliteMetadataWriter};
use object_store::local::LocalFileSystem;
use sqlx::{Row, SqlitePool};
use tempfile::TempDir;

/// (min_value, max_value, null_count, value_count) per column, ordered by
/// column_id (i.e. by declared column order).
type FileStatRow = (Option<String>, Option<String>, Option<i64>, Option<i64>);

#[tokio::test(flavor = "multi_thread")]
async fn crate_write_produces_duckdb_canonical_column_stats() {
    let temp = TempDir::new().unwrap();
    let db_path = temp.path().join("stats.db");
    let data_path = temp.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();

    let conn_str = format!("sqlite:{}?mode=rwc", db_path.display());
    let writer = SqliteMetadataWriter::new_with_init(&conn_str)
        .await
        .unwrap();
    writer.set_data_path(data_path.to_str().unwrap()).unwrap();

    // id: 1..3 (no nulls); name: Alice/Bob/NULL; d: 2020-01-01/-05/-03 (day
    // numbers since the epoch: 18262 = 2020-01-01, 18266 = 2020-01-05).
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true),
        Field::new("d", DataType::Date32, true),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(StringArray::from(vec![Some("Alice"), Some("Bob"), None])),
            Arc::new(Date32Array::from(vec![
                Some(18262),
                Some(18266),
                Some(18264),
            ])),
        ],
    )
    .unwrap();

    let table_writer =
        DuckLakeTableWriter::new(Arc::new(writer), Arc::new(LocalFileSystem::new())).unwrap();
    table_writer
        .write_table("main", "t", &[batch])
        .await
        .unwrap();

    // Read the persisted stats straight out of the catalog.
    let pool = SqlitePool::connect(&format!("sqlite:{}", db_path.display()))
        .await
        .unwrap();

    let file_stats: Vec<FileStatRow> = sqlx::query(
        "SELECT min_value, max_value, null_count, value_count
         FROM ducklake_file_column_stats ORDER BY column_id",
    )
    .fetch_all(&pool)
    .await
    .unwrap()
    .into_iter()
    .map(|r| {
        (
            r.try_get(0).unwrap(),
            r.try_get(1).unwrap(),
            r.try_get(2).unwrap(),
            r.try_get(3).unwrap(),
        )
    })
    .collect();

    assert_eq!(
        file_stats,
        vec![
            (
                Some("1".to_string()),
                Some("3".to_string()),
                Some(0),
                Some(3)
            ),
            (
                Some("Alice".to_string()),
                Some("Bob".to_string()),
                Some(1),
                Some(2)
            ),
            (
                Some("2020-01-01".to_string()),
                Some("2020-01-05".to_string()),
                Some(0),
                Some(3)
            ),
        ],
        "per-file zone maps must match DuckDB-canonical encodings"
    );

    // column_size_bytes: compressed on-disk size per column, harvested from the
    // parquet footer like official DuckLake. Present and positive for every
    // written column. Sizes are data-dependent, so assert presence + positivity
    // rather than exact bytes.
    let sizes: Vec<Option<i64>> =
        sqlx::query("SELECT column_size_bytes FROM ducklake_file_column_stats ORDER BY column_id")
            .fetch_all(&pool)
            .await
            .unwrap()
            .into_iter()
            .map(|r| r.try_get(0).unwrap())
            .collect();
    assert_eq!(
        sizes.len(),
        3,
        "one column_size_bytes row per written column"
    );
    for (i, s) in sizes.iter().enumerate() {
        assert!(
            matches!(s, Some(n) if *n > 0),
            "column {i} must have a positive column_size_bytes, got {s:?}"
        );
    }

    // Global roll-up: one row per column, contains_null true only for `name`.
    let table_stats: Vec<(Option<bool>, Option<String>, Option<String>)> = sqlx::query(
        "SELECT contains_null, min_value, max_value
         FROM ducklake_table_column_stats ORDER BY column_id",
    )
    .fetch_all(&pool)
    .await
    .unwrap()
    .into_iter()
    .map(|r| {
        (
            r.try_get(0).unwrap(),
            r.try_get(1).unwrap(),
            r.try_get(2).unwrap(),
        )
    })
    .collect();

    assert_eq!(
        table_stats,
        vec![
            (Some(false), Some("1".to_string()), Some("3".to_string())),
            (
                Some(true),
                Some("Alice".to_string()),
                Some("Bob".to_string())
            ),
            (
                Some(false),
                Some("2020-01-01".to_string()),
                Some("2020-01-05".to_string())
            ),
        ],
        "table-wide roll-up must reflect the single file's bounds"
    );
}

/// Differential dump vs official DuckLake: writes the SAME diverse-typed data
/// the `duckdb` CLI reference used, then prints the persisted per-file and
/// table-wide stats so they can be diffed against official. Run with:
///   cargo test --features write-sqlite --test column_stats_tests -- --nocapture differential_dump
#[tokio::test(flavor = "multi_thread")]
async fn differential_dump() {
    let temp = TempDir::new().unwrap();
    let db_path = temp.path().join("stats.db");
    let data_path = temp.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();
    let conn_str = format!("sqlite:{}?mode=rwc", db_path.display());
    let writer = SqliteMetadataWriter::new_with_init(&conn_str)
        .await
        .unwrap();
    writer.set_data_path(data_path.to_str().unwrap()).unwrap();

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("big", DataType::Int64, false),
        Field::new("price", DataType::Float64, true),
        Field::new("amt", DataType::Decimal128(10, 2), true),
        Field::new("d", DataType::Date32, true),
        Field::new("ts", DataType::Timestamp(TimeUnit::Microsecond, None), true),
        Field::new("name", DataType::Utf8, true),
        Field::new("flag", DataType::Boolean, true),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(Int64Array::from(vec![100000000000, -100000000000, 0])),
            Arc::new(Float64Array::from(vec![1.5, 3.14, -0.5])),
            Arc::new(
                Decimal128Array::from(vec![12345, 5, 10000])
                    .with_precision_and_scale(10, 2)
                    .unwrap(),
            ),
            Arc::new(Date32Array::from(vec![18262, 18264, 18266])),
            Arc::new(TimestampMicrosecondArray::from(vec![
                1_578_227_696_123_456,
                1_578_268_800_000_000,
                1_578_125_700_000_000,
            ])),
            Arc::new(StringArray::from(vec![Some("Alice"), Some("Bob"), None])),
            Arc::new(BooleanArray::from(vec![true, false, true])),
        ],
    )
    .unwrap();

    DuckLakeTableWriter::new(Arc::new(writer), Arc::new(LocalFileSystem::new()))
        .unwrap()
        .write_table("main", "t", &[batch])
        .await
        .unwrap();

    let pool = SqlitePool::connect(&format!("sqlite:{}", db_path.display()))
        .await
        .unwrap();

    eprintln!("--- CRATE FILE_STATS (name|min|max|null|value|contains_nan) ---");
    for row in sqlx::query(
        "SELECT c.column_name, s.min_value, s.max_value, s.null_count, s.value_count, s.contains_nan
         FROM ducklake_file_column_stats s
         JOIN ducklake_column c ON c.column_id = s.column_id AND c.end_snapshot IS NULL
         ORDER BY c.column_order",
    )
    .fetch_all(&pool)
    .await
    .unwrap()
    {
        let name: String = row.try_get(0).unwrap();
        let mn: Option<String> = row.try_get(1).unwrap();
        let mx: Option<String> = row.try_get(2).unwrap();
        let nc: Option<i64> = row.try_get(3).unwrap();
        let vc: Option<i64> = row.try_get(4).unwrap();
        let nan: Option<bool> = row.try_get(5).unwrap();
        eprintln!("{name}|{mn:?}|{mx:?}|{nc:?}|{vc:?}|{nan:?}");
    }

    eprintln!("--- CRATE TABLE_STATS (name|min|max|contains_null|contains_nan) ---");
    for row in sqlx::query(
        "SELECT c.column_name, g.min_value, g.max_value, g.contains_null, g.contains_nan
         FROM ducklake_table_column_stats g
         JOIN ducklake_column c ON c.column_id = g.column_id AND c.end_snapshot IS NULL
         ORDER BY c.column_order",
    )
    .fetch_all(&pool)
    .await
    .unwrap()
    {
        let name: String = row.try_get(0).unwrap();
        let mn: Option<String> = row.try_get(1).unwrap();
        let mx: Option<String> = row.try_get(2).unwrap();
        let cn: Option<bool> = row.try_get(3).unwrap();
        let nan: Option<bool> = row.try_get(4).unwrap();
        eprintln!("{name}|{mn:?}|{mx:?}|{cn:?}|{nan:?}");
    }
}

/// Emit a crate-written catalog to $LAKE_OUT (skipped if unset) so an external
/// DuckDB can attach it — the reverse round-trip check.
#[tokio::test(flavor = "multi_thread")]
async fn emit_catalog_for_duckdb() {
    let Ok(out) = std::env::var("LAKE_OUT") else {
        return;
    };
    let data_path = format!("{out}/data");
    std::fs::create_dir_all(&data_path).unwrap();
    let conn_str = format!("sqlite:{out}/meta.sqlite?mode=rwc");
    let writer = SqliteMetadataWriter::new_with_init(&conn_str)
        .await
        .unwrap();
    writer.set_data_path(&data_path).unwrap();

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("price", DataType::Float64, true),
        Field::new("amt", DataType::Decimal128(10, 2), true),
        Field::new("d", DataType::Date32, true),
        Field::new("name", DataType::Utf8, true),
        Field::new("flag", DataType::Boolean, true),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5])),
            Arc::new(Float64Array::from(vec![1.5, 3.14, -0.5, 9.0, 2.0])),
            Arc::new(
                Decimal128Array::from(vec![12345, 5, 10000, 200, 999])
                    .with_precision_and_scale(10, 2)
                    .unwrap(),
            ),
            Arc::new(Date32Array::from(vec![18262, 18264, 18266, 18263, 18265])),
            Arc::new(StringArray::from(vec![
                Some("Alice"),
                Some("Bob"),
                None,
                Some("Dave"),
                Some("Eve"),
            ])),
            Arc::new(BooleanArray::from(vec![true, false, true, false, true])),
        ],
    )
    .unwrap();
    DuckLakeTableWriter::new(Arc::new(writer), Arc::new(LocalFileSystem::new()))
        .unwrap()
        .write_table("main", "t", &[batch])
        .await
        .unwrap();
    eprintln!("wrote crate catalog to {out}");
}

/// Finding-1 regression: a float file containing NaN must store NULL min/max
/// (never a NaN-excluded finite bound) with contains_nan = true, so no reader
/// can prune it — matching official DuckLake.
#[tokio::test(flavor = "multi_thread")]
async fn float_with_nan_suppresses_minmax() {
    let temp = TempDir::new().unwrap();
    let db_path = temp.path().join("stats.db");
    let data_path = temp.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();
    let conn_str = format!("sqlite:{}?mode=rwc", db_path.display());
    let writer = SqliteMetadataWriter::new_with_init(&conn_str)
        .await
        .unwrap();
    writer.set_data_path(data_path.to_str().unwrap()).unwrap();

    let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Float64, true)]));
    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(Float64Array::from(vec![f64::NAN, 1.0, 2.0]))],
    )
    .unwrap();
    DuckLakeTableWriter::new(Arc::new(writer), Arc::new(LocalFileSystem::new()))
        .unwrap()
        .write_table("main", "t", &[batch])
        .await
        .unwrap();

    let pool = SqlitePool::connect(&format!("sqlite:{}", db_path.display()))
        .await
        .unwrap();
    let row = sqlx::query(
        "SELECT min_value, max_value, contains_nan, value_count
         FROM ducklake_file_column_stats",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    let mn: Option<String> = row.try_get(0).unwrap();
    let mx: Option<String> = row.try_get(1).unwrap();
    let nan: Option<bool> = row.try_get(2).unwrap();
    let vc: Option<i64> = row.try_get(3).unwrap();
    assert_eq!(mn, None, "min must be NULL when NaN present");
    assert_eq!(mx, None, "max must be NULL when NaN present");
    assert_eq!(nan, Some(true), "contains_nan must be true");
    assert_eq!(vc, Some(3), "NaN counts as a non-null value");
}

/// NaN rows must survive a filtered scan end-to-end. Parquet footer float
/// bounds exclude NaN while DataFusion evaluates `NaN > C` as true (IEEE
/// totalOrder), so a float `>` predicate pushed into the parquet reader can
/// row-group-prune the group holding the NaN row. `NanPruningBarrierExec`
/// keeps such predicates out of the scan whenever the file's NaN state isn't
/// known false — this asserts the query results, not just the plan shape.
#[tokio::test(flavor = "multi_thread")]
async fn nan_rows_survive_filtered_scan() {
    use datafusion::prelude::SessionContext;
    use datafusion_ducklake::{DuckLakeCatalog, SqliteMetadataProvider};

    let temp = TempDir::new().unwrap();
    let db_path = temp.path().join("stats.db");
    let data_path = temp.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();
    let conn_str = format!("sqlite:{}?mode=rwc", db_path.display());
    let writer = SqliteMetadataWriter::new_with_init(&conn_str)
        .await
        .unwrap();
    writer.set_data_path(data_path.to_str().unwrap()).unwrap();

    let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Float64, true)]));
    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(Float64Array::from(vec![f64::NAN, 1.0, 2.0]))],
    )
    .unwrap();
    DuckLakeTableWriter::new(Arc::new(writer), Arc::new(LocalFileSystem::new()))
        .unwrap()
        .write_table("main", "t", &[batch])
        .await
        .unwrap();

    let provider = SqliteMetadataProvider::new(&format!("sqlite:{}", db_path.display()))
        .await
        .unwrap();
    let catalog = DuckLakeCatalog::new(provider).unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("test", Arc::new(catalog));

    let count = |sql: &str| {
        let ctx = ctx.clone();
        let sql = sql.to_string();
        async move {
            let batches = ctx.sql(&sql).await.unwrap().collect().await.unwrap();
            batches.iter().map(|b| b.num_rows()).sum::<usize>()
        }
    };

    // NaN sorts above every value, so it matches any `>` bound the finite
    // values fail; the footer max (2.0) must not prune it away.
    assert_eq!(
        count("SELECT x FROM test.main.t WHERE x > 100").await,
        1,
        "the NaN row must not be pruned by footer max"
    );
    assert_eq!(count("SELECT x FROM test.main.t WHERE x > 1.5").await, 2);
    // `<` predicates never match NaN; finite semantics unchanged.
    assert_eq!(count("SELECT x FROM test.main.t WHERE x < 1.5").await, 1);
    assert_eq!(count("SELECT x FROM test.main.t").await, 3);

    // The barrier must be present in the plan for the NaN-unsafe column.
    let explain = ctx
        .sql("EXPLAIN SELECT x FROM test.main.t WHERE x > 100")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let plan = datafusion::arrow::util::pretty::pretty_format_batches(&explain)
        .unwrap()
        .to_string();
    assert!(
        plan.contains("NanPruningBarrierExec: unsafe_columns=[x]"),
        "expected NaN pruning barrier in plan:\n{plan}"
    );
}
