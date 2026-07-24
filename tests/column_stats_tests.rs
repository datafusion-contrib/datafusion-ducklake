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

// ---------------------------------------------------------------------------
// NaN pruning-safety scenarios
//
// Parquet footer float bounds exclude NaN while DataFusion evaluates `NaN > C`
// as true (IEEE totalOrder), so any pruning that trusts a NaN-blind max can
// silently drop NaN rows. Two guards exist: the catalog gate (`contains_nan`
// must be false for a float max to drive plan-time file pruning) and
// `NanPruningBarrierExec` (float predicates must not reach the parquet
// reader's row-group pruning unless every scanned file is known NaN-free).
// These tests pin the behavior for each writer/NaN combination end-to-end,
// asserting query RESULTS first and plan shape second.
// ---------------------------------------------------------------------------

/// Write `t(id INT, x DOUBLE)` with one data file per `files` element (first
/// is a Replace, the rest Append) and return the query context. The returned
/// sqlite path allows tests to doctor catalog stats before querying.
async fn setup_float_table(
    files: &[(Vec<i32>, Vec<f64>)],
) -> (TempDir, String, datafusion::prelude::SessionContext) {
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

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("x", DataType::Float64, true),
    ]));
    let table_writer =
        DuckLakeTableWriter::new(Arc::new(writer), Arc::new(LocalFileSystem::new())).unwrap();
    for (index, (ids, xs)) in files.iter().enumerate() {
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(ids.clone())), Arc::new(Float64Array::from(xs.clone()))],
        )
        .unwrap();
        if index == 0 {
            table_writer
                .write_table("main", "t", &[batch])
                .await
                .unwrap();
        } else {
            table_writer
                .append_table("main", "t", &[batch])
                .await
                .unwrap();
        }
    }

    let db_url = format!("sqlite:{}", db_path.display());
    let provider = SqliteMetadataProvider::new(&db_url).await.unwrap();
    let catalog = DuckLakeCatalog::new(provider).unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("test", Arc::new(catalog));
    (temp, db_url, ctx)
}

async fn row_count(ctx: &datafusion::prelude::SessionContext, sql: &str) -> usize {
    ctx.sql(sql)
        .await
        .unwrap()
        .collect()
        .await
        .unwrap()
        .iter()
        .map(|batch| batch.num_rows())
        .sum()
}

async fn physical_plan(ctx: &datafusion::prelude::SessionContext, sql: &str) -> String {
    let batches = ctx
        .sql(&format!("EXPLAIN {sql}"))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    datafusion::arrow::util::pretty::pretty_format_batches(&batches)
        .unwrap()
        .to_string()
}

/// The original wrong-results repro: a write-through file containing NaN.
/// NaN rows must survive filtered scans — the footer max must not row-group-
/// prune them away.
#[tokio::test(flavor = "multi_thread")]
async fn nan_rows_survive_filtered_scan() {
    let (_temp, _db, ctx) = setup_float_table(&[(vec![7, 8, 9], vec![f64::NAN, 1.0, 2.0])]).await;

    // NaN sorts above every value, so it matches any `>` bound the finite
    // values fail; the footer max (2.0) must not prune it away.
    assert_eq!(
        row_count(&ctx, "SELECT x FROM test.main.t WHERE x > 100").await,
        1,
        "the NaN row must not be pruned by footer max"
    );
    assert_eq!(
        row_count(&ctx, "SELECT x FROM test.main.t WHERE x > 1.5").await,
        2
    );
    // Equality probes for NaN itself: min <= NaN always holds, so the file
    // must be kept and the row found.
    assert_eq!(
        row_count(
            &ctx,
            "SELECT x FROM test.main.t WHERE x = CAST('NaN' AS DOUBLE)"
        )
        .await,
        1
    );
    // `<` predicates never match NaN; finite semantics unchanged.
    assert_eq!(
        row_count(&ctx, "SELECT x FROM test.main.t WHERE x < 1.5").await,
        1
    );
    assert_eq!(row_count(&ctx, "SELECT x FROM test.main.t").await, 3);

    // The barrier must be present in the plan for the NaN-unsafe column.
    let plan = physical_plan(&ctx, "SELECT x FROM test.main.t WHERE x > 100").await;
    assert!(
        plan.contains("NanPruningBarrierExec: unsafe_columns=[x]"),
        "expected NaN pruning barrier in plan:\n{plan}"
    );
}

/// NaN-free floats must keep BOTH pruning levels — the fix must not tax the
/// common case. File-level: a `>` predicate above the max prunes the file
/// outright. Row-group level: an in-range predicate is pushed into the
/// parquet scan, with no barrier in the plan.
#[tokio::test(flavor = "multi_thread")]
async fn nan_free_floats_keep_full_pruning() {
    let (_temp, _db, ctx) = setup_float_table(&[(vec![1, 2], vec![1.0, 2.0])]).await;

    // contains_nan = false was recorded, so the catalog float max is trusted
    // and the single file is pruned at plan time.
    let plan = physical_plan(&ctx, "SELECT x FROM test.main.t WHERE x > 100").await;
    assert!(
        plan.contains("EmptyExec"),
        "NaN-free file should be pruned via its float max:\n{plan}"
    );
    assert_eq!(
        row_count(&ctx, "SELECT x FROM test.main.t WHERE x > 100").await,
        0
    );

    // An in-range predicate keeps the file but reaches the parquet reader
    // unimpeded: no barrier, predicate attached to the scan.
    let plan = physical_plan(&ctx, "SELECT x FROM test.main.t WHERE x >= 1.5").await;
    assert!(
        !plan.contains("NanPruningBarrierExec"),
        "no barrier expected for a NaN-free file:\n{plan}"
    );
    assert!(
        plan.contains("predicate="),
        "float predicate should be pushed into the parquet scan:\n{plan}"
    );
    assert_eq!(
        row_count(&ctx, "SELECT x FROM test.main.t WHERE x >= 1.5").await,
        1
    );
}

/// The barrier is per-column: predicates on non-float columns keep full
/// parquet pushdown even when a float column in the same file is NaN-unsafe.
#[tokio::test(flavor = "multi_thread")]
async fn barrier_blocks_only_float_predicates() {
    let (_temp, _db, ctx) = setup_float_table(&[(vec![7, 8, 9], vec![f64::NAN, 1.0, 2.0])]).await;

    // Integer predicate: barrier present (x is unsafe) but the predicate
    // passes through it into the scan.
    let plan = physical_plan(&ctx, "SELECT id FROM test.main.t WHERE id >= 8").await;
    assert!(
        plan.contains("NanPruningBarrierExec: unsafe_columns=[x]"),
        "barrier expected while x is NaN-unsafe:\n{plan}"
    );
    assert!(
        plan.contains("predicate="),
        "integer predicate should pass the barrier into the scan:\n{plan}"
    );
    assert_eq!(
        row_count(&ctx, "SELECT id FROM test.main.t WHERE id >= 8").await,
        2
    );

    // Float predicate: rejected by the barrier — no predicate on the scan.
    let plan = physical_plan(&ctx, "SELECT x FROM test.main.t WHERE x > 100").await;
    assert!(
        !plan.contains("predicate="),
        "float predicate must not reach the parquet scan:\n{plan}"
    );

    // Mixed conjunction still finds the NaN row (id 7).
    assert_eq!(
        row_count(&ctx, "SELECT id FROM test.main.t WHERE id >= 0 AND x > 100").await,
        1
    );
}

/// Catalog rows shaped like official DuckLake's `ducklake_add_data_files`
/// (register-by-reference): float min/max present, contains_nan NULL. The max
/// must not be trusted at either pruning level, while the min still prunes —
/// NaN can only hide above the max, never below the min.
#[tokio::test(flavor = "multi_thread")]
async fn nan_unknown_bounds_prune_only_below_min() {
    let (_temp, db_url, ctx) =
        setup_float_table(&[(vec![7, 8, 9], vec![f64::NAN, 1.0, 2.0])]).await;

    // The crate's write path blanked the bounds and set contains_nan = true.
    // Rewrite the row to the official add_data_files shape: footer bounds
    // present (they exclude the NaN), NaN state unknown.
    let pool = SqlitePool::connect(&db_url).await.unwrap();
    sqlx::query(
        "UPDATE ducklake_file_column_stats
         SET min_value = '1.0', max_value = '2.0', contains_nan = NULL
         WHERE column_id IN
             (SELECT column_id FROM ducklake_column WHERE column_name = 'x')",
    )
    .execute(&pool)
    .await
    .unwrap();

    // max present but NaN unknown: `x > 100` must keep the file AND keep the
    // predicate away from the parquet reader — the NaN row survives.
    assert_eq!(
        row_count(&ctx, "SELECT x FROM test.main.t WHERE x > 100").await,
        1,
        "catalog max with unknown NaN state must not prune the NaN row"
    );
    let plan = physical_plan(&ctx, "SELECT x FROM test.main.t WHERE x > 100").await;
    assert!(
        plan.contains("NanPruningBarrierExec: unsafe_columns=[x]"),
        "barrier expected while NaN state is unknown:\n{plan}"
    );

    // min stays trustworthy: a predicate strictly below it prunes the file
    // outright (EmptyExec), NaN notwithstanding.
    let plan = physical_plan(&ctx, "SELECT x FROM test.main.t WHERE x < 0.5").await;
    assert!(
        plan.contains("EmptyExec"),
        "file should be pruned via its float min even with NaN unknown:\n{plan}"
    );
    assert_eq!(
        row_count(&ctx, "SELECT x FROM test.main.t WHERE x < 0.5").await,
        0
    );
}

/// One NaN-unsafe file poisons float pushdown for the whole multi-file scan
/// group (the unsafe set is a union), and NaN rows still surface from the
/// unsafe file while finite rows come from both.
#[tokio::test(flavor = "multi_thread")]
async fn nan_multi_file_scan_blocks_float_pruning() {
    let (_temp, _db, ctx) =
        setup_float_table(&[(vec![1, 2], vec![1.0, 2.0]), (vec![3, 4], vec![f64::NAN, 5.0])]).await;

    assert_eq!(row_count(&ctx, "SELECT x FROM test.main.t").await, 4);
    assert_eq!(
        row_count(&ctx, "SELECT x FROM test.main.t WHERE x > 100").await,
        1,
        "the NaN row in the second file must survive"
    );
    assert_eq!(
        row_count(&ctx, "SELECT x FROM test.main.t WHERE x > 4").await,
        2
    );
    assert_eq!(
        row_count(&ctx, "SELECT x FROM test.main.t WHERE x < 1.5").await,
        1
    );

    let plan = physical_plan(&ctx, "SELECT x FROM test.main.t WHERE x > 100").await;
    assert!(
        plan.contains("NanPruningBarrierExec: unsafe_columns=[x]"),
        "one unsafe file must make x unsafe for the scan group:\n{plan}"
    );
}
