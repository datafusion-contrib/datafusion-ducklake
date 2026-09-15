//! A scan must publish the catalog's row counts and column bounds as its own
//! `Statistics`, so DataFusion's `AggregateStatistics` rule can fold an
//! unfiltered `count(*)` / `min` / `max` into a literal instead of reading the
//! data to recompute what the catalog already records.
//!
//! Per-file statistics were always attached, so pruning and row-group filtering
//! worked; the scan-level summary was missing, and that summary is the only
//! thing the aggregate rule consults. The gap was invisible because every query
//! still returned the RIGHT answer — just by scanning. These tests therefore
//! assert the plan shape *and* the values: a summary that folds a wrong count
//! into a literal is far worse than the slow scan it replaces.
//!
//! Official DuckLake answers the same aggregates from its catalog
//! (`DuckLakeGetPartitionStats`), including its rule that column bounds are
//! exact only while no row has ever been deleted — so `deletes_*` below pins the
//! behaviour that keeps us from over-claiming.

#![cfg(all(feature = "write-sqlite", feature = "metadata-sqlite"))]

use std::sync::Arc;

use arrow::array::{Int32Array, Int64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::prelude::*;
use object_store::local::LocalFileSystem;
use tempfile::TempDir;

use datafusion_ducklake::partition::PartitionTransform;
use datafusion_ducklake::{
    ColumnDef, DuckLakeCatalog, DuckLakeTableWriter, DuckLakeWriteOptions, MetadataWriter,
    SqliteMetadataProvider, SqliteMetadataWriter, WriteMode,
};

fn table_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("val", DataType::Int32, false),
    ]))
}

fn object_store() -> Arc<dyn object_store::ObjectStore> {
    Arc::new(LocalFileSystem::new())
}

fn batch(ids: Vec<i32>, vals: Vec<i32>) -> RecordBatch {
    RecordBatch::try_new(
        table_schema(),
        vec![Arc::new(Int32Array::from(ids)), Arc::new(Int32Array::from(vals))],
    )
    .unwrap()
}

fn conn_str(temp_dir: &TempDir, writable: bool) -> String {
    let db_path = temp_dir.path().join("test.db");
    if writable {
        format!("sqlite:{}?mode=rwc", db_path.display())
    } else {
        format!("sqlite:{}", db_path.display())
    }
}

/// Two data files: ids 1..=3 and 10..=12, so the table holds 6 rows with
/// `min(id) = 1` and `max(id) = 12` spanning BOTH files — a per-file bound
/// alone cannot produce that pair, only the merge across files can.
async fn seed_two_files(temp_dir: &TempDir) {
    let data_path = temp_dir.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();
    let writer = SqliteMetadataWriter::new_with_init(&conn_str(temp_dir, true))
        .await
        .unwrap();
    writer.set_data_path(data_path.to_str().unwrap()).unwrap();
    DuckLakeTableWriter::new(Arc::new(writer), object_store())
        .unwrap()
        .write_table("main", "t", &[batch(vec![1, 2, 3], vec![10, 20, 30])])
        .await
        .unwrap();

    let writer = SqliteMetadataWriter::new(&conn_str(temp_dir, true))
        .await
        .unwrap();
    DuckLakeTableWriter::new(Arc::new(writer), object_store())
        .unwrap()
        .append_table("main", "t", &[batch(vec![10, 11, 12], vec![40, 50, 60])])
        .await
        .unwrap();
}

async fn session(temp_dir: &TempDir) -> SessionContext {
    let provider = SqliteMetadataProvider::new(&conn_str(temp_dir, false))
        .await
        .unwrap();
    let catalog = DuckLakeCatalog::new(provider).unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));
    ctx
}

/// The rendered physical plan for `sql`.
async fn physical_plan(ctx: &SessionContext, sql: &str) -> String {
    let plan = ctx
        .sql(sql)
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    format!(
        "{}",
        datafusion::physical_plan::displayable(plan.as_ref()).indent(false)
    )
}

async fn scalar_i64(ctx: &SessionContext, sql: &str, column: usize) -> i64 {
    let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
    let array = batches[0].column(column);
    if let Some(v) = array.as_any().downcast_ref::<Int64Array>() {
        return v.value(0);
    }
    i64::from(
        array
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("int column")
            .value(0),
    )
}

#[tokio::test(flavor = "multi_thread")]
async fn unfiltered_count_is_answered_without_scanning() {
    let temp = TempDir::new().unwrap();
    seed_two_files(&temp).await;
    let ctx = session(&temp).await;

    let plan = physical_plan(&ctx, "SELECT count(*) FROM ducklake.main.t").await;
    assert!(
        plan.contains("PlaceholderRowExec"),
        "count(*) should fold to a literal from catalog statistics, got:\n{plan}"
    );
    assert!(
        !plan.contains("DataSourceExec"),
        "count(*) should not read any file, got:\n{plan}"
    );

    // The literal must be the truth, not merely a literal.
    assert_eq!(
        scalar_i64(&ctx, "SELECT count(*) FROM ducklake.main.t", 0).await,
        6
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn unfiltered_min_max_is_answered_without_scanning() {
    let temp = TempDir::new().unwrap();
    seed_two_files(&temp).await;
    let ctx = session(&temp).await;

    let sql = "SELECT min(id), max(id) FROM ducklake.main.t";
    let plan = physical_plan(&ctx, sql).await;
    assert!(
        plan.contains("PlaceholderRowExec"),
        "min/max should fold from catalog column bounds, got:\n{plan}"
    );

    // Bounds must be merged ACROSS files: 1 lives in the first, 12 in the second.
    assert_eq!(scalar_i64(&ctx, sql, 0).await, 1);
    assert_eq!(scalar_i64(&ctx, sql, 1).await, 12);
}

#[tokio::test(flavor = "multi_thread")]
async fn filtered_count_still_reads_the_data() {
    let temp = TempDir::new().unwrap();
    seed_two_files(&temp).await;
    let ctx = session(&temp).await;

    // A filter makes the row count unknowable from statistics; DataFusion marks
    // the scan's statistics inexact once a filter is pushed down, so the rule
    // must decline rather than fold a pre-filter count into a literal.
    let sql = "SELECT count(*) FROM ducklake.main.t WHERE id > 2";
    let plan = physical_plan(&ctx, sql).await;
    assert!(
        plan.contains("DataSourceExec"),
        "a filtered count must read the data, got:\n{plan}"
    );
    assert_eq!(scalar_i64(&ctx, sql, 0).await, 4);
}

#[tokio::test(flavor = "multi_thread")]
async fn deletes_do_not_corrupt_count_or_bounds() {
    let temp = TempDir::new().unwrap();
    seed_two_files(&temp).await;

    // Remove the row holding the table's maximum id. The catalog's column
    // bounds are not tightened by a delete, so answering max(id) from them
    // would now return a row that no longer exists.
    let writer = SqliteMetadataWriter::new(&conn_str(&temp, true))
        .await
        .unwrap();
    let provider = SqliteMetadataProvider::new(&conn_str(&temp, true))
        .await
        .unwrap();
    let catalog = DuckLakeCatalog::with_writer(Arc::new(provider), Arc::new(writer)).unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));
    ctx.sql("DELETE FROM ducklake.main.t WHERE id = 12")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    let ctx = session(&temp).await;

    // Official folds count(*) unconditionally, subtracting deletes. So do we:
    // the delete filter knows how many positions it drops.
    let count_sql = "SELECT count(*) FROM ducklake.main.t";
    let count_plan = physical_plan(&ctx, count_sql).await;
    assert!(
        count_plan.contains("PlaceholderRowExec"),
        "count(*) must still fold after a DELETE:\n{count_plan}"
    );
    assert_eq!(
        scalar_i64(&ctx, count_sql, 0).await,
        5,
        "count must exclude deleted rows"
    );

    // Bounds are a different matter: the delete may have removed the extreme,
    // and nothing in the plan knows whether it did.
    let max_sql = "SELECT max(id) FROM ducklake.main.t";
    let max_plan = physical_plan(&ctx, max_sql).await;
    assert!(
        max_plan.contains("DataSourceExec"),
        "max must be read from the data once rows are deleted:\n{max_plan}"
    );
    assert_eq!(
        scalar_i64(&ctx, max_sql, 0).await,
        11,
        "max must not report a deleted row's value"
    );
}

/// An INLINED delete leaves `delete_file` and `delete_count` both NULL, so the
/// catalog counters cannot see it. The count must still come out right, because
/// the delete filter counts the positions it actually drops rather than trusting
/// those counters.
#[tokio::test(flavor = "multi_thread")]
async fn inlined_delete_is_subtracted_from_the_folded_count() {
    let temp = TempDir::new().unwrap();
    seed_two_files(&temp).await;

    let writer = SqliteMetadataWriter::new(&conn_str(&temp, true))
        .await
        .unwrap();
    let provider = SqliteMetadataProvider::new(&conn_str(&temp, true))
        .await
        .unwrap();
    let catalog = DuckLakeCatalog::with_writer(Arc::new(provider), Arc::new(writer)).unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));
    // Small enough to be recorded as an inlined delete rather than a delete file.
    ctx.sql("DELETE FROM ducklake.main.t WHERE id = 2")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    let ctx = session(&temp).await;
    assert_eq!(
        scalar_i64(&ctx, "SELECT count(*) FROM ducklake.main.t", 0).await,
        5,
        "one row deleted from six"
    );
}

/// A catalog bound on a string column may be a truncated, rounded-up prefix —
/// the DuckLake spec only requires bounds to BOUND the column. Official refuses
/// string MIN/MAX from statistics for exactly this reason. Truncate the stored
/// bound the way a parquet writer would and assert we return the real value.
#[tokio::test(flavor = "multi_thread")]
async fn widened_string_bound_never_answers_max() {
    let temp = TempDir::new().unwrap();
    let data_path = temp.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();

    let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Utf8, false)]));
    let long = "z".repeat(400) + "bbb";
    let rows = vec!["aaa".to_string(), long.clone()];
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(arrow::array::StringArray::from(rows))],
    )
    .unwrap();

    let writer = SqliteMetadataWriter::new_with_init(&conn_str(&temp, true))
        .await
        .unwrap();
    writer.set_data_path(data_path.to_str().unwrap()).unwrap();
    DuckLakeTableWriter::new(Arc::new(writer), object_store())
        .unwrap()
        .write_table("main", "s", &[batch])
        .await
        .unwrap();

    // Stand in for a writer that truncated the bound to a 256-byte prefix and
    // rounded the last byte up, which is what parquet does by default.
    let pool = sqlx::SqlitePool::connect(&conn_str(&temp, true))
        .await
        .unwrap();
    let truncated = format!("{}{{", "z".repeat(255));
    sqlx::query("UPDATE ducklake_file_column_stats SET max_value = ?")
        .bind(&truncated)
        .execute(&pool)
        .await
        .unwrap();
    pool.close().await;

    let ctx = session(&temp).await;
    let sql = "SELECT max(v) FROM ducklake.main.s";
    let plan = physical_plan(&ctx, sql).await;
    assert!(
        plan.contains("DataSourceExec"),
        "a string max must be read from the data, not folded from a bound:\n{plan}"
    );

    let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
    // The result may arrive as Utf8, LargeUtf8 or Utf8View depending on how the
    // scan renders it; normalise before comparing.
    let column = arrow::compute::cast(batches[0].column(0), &DataType::Utf8).unwrap();
    let got = column
        .as_any()
        .downcast_ref::<arrow::array::StringArray>()
        .unwrap()
        .value(0)
        .to_string();
    assert_eq!(got, long, "max(v) must be the row's value, not the bound");
}

/// `record_count` is nullable, and foreign catalogs omit it. The pruning path
/// falls back to a column's `value_count`, which counts NON-NULL values — a
/// smaller number on a nullable column. That must never become the answer to
/// `count(*)`.
///
/// The fixture is deliberately nullable with NULLs present, so `value_count`
/// (3) differs from the row count (5): the old behaviour would have folded
/// `count(*)` to 3.
#[tokio::test(flavor = "multi_thread")]
async fn null_record_count_never_folds_a_wrong_count() {
    let temp = TempDir::new().unwrap();
    let data_path = temp.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();

    let schema = Arc::new(Schema::new(vec![Field::new("n", DataType::Int32, true)]));
    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(Int32Array::from(vec![
            Some(1),
            None,
            Some(3),
            None,
            Some(5),
        ]))],
    )
    .unwrap();

    let writer = SqliteMetadataWriter::new_with_init(&conn_str(&temp, true))
        .await
        .unwrap();
    writer.set_data_path(data_path.to_str().unwrap()).unwrap();
    DuckLakeTableWriter::new(Arc::new(writer), object_store())
        .unwrap()
        .write_table("main", "n", &[batch])
        .await
        .unwrap();

    let pool = sqlx::SqlitePool::connect(&conn_str(&temp, true))
        .await
        .unwrap();
    // Confirm the fixture really does expose a misleading fallback before
    // relying on it: value_count must be smaller than the true row count.
    let value_count: Option<i64> =
        sqlx::query_scalar("SELECT value_count FROM ducklake_file_column_stats LIMIT 1")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(
        value_count,
        Some(3),
        "fixture must have NULLs to be meaningful"
    );
    sqlx::query("UPDATE ducklake_data_file SET record_count = NULL")
        .execute(&pool)
        .await
        .unwrap();
    pool.close().await;

    let ctx = session(&temp).await;
    let sql = "SELECT count(*) FROM ducklake.main.n";
    let plan = physical_plan(&ctx, sql).await;
    assert!(
        plan.contains("DataSourceExec"),
        "an unknown record_count must fall back to reading the data:\n{plan}"
    );
    assert_eq!(
        scalar_i64(&ctx, sql, 0).await,
        5,
        "must not report value_count"
    );
}

/// Rows small enough to be inlined live in the catalog, not in a data file. A
/// summary built only from data files would omit them, so `count(*)` must
/// either account for them or decline to fold. Either is acceptable; reporting
/// the data-file count alone is not.
#[tokio::test(flavor = "multi_thread")]
async fn inlined_rows_are_not_omitted_from_count() {
    let temp = TempDir::new().unwrap();
    seed_two_files(&temp).await;

    // Append two more rows under an inlining limit that captures them, so the
    // table holds 6 rows in parquet plus 2 inlined.
    let writer = SqliteMetadataWriter::new(&conn_str(&temp, true))
        .await
        .unwrap();
    DuckLakeTableWriter::new(Arc::new(writer), object_store())
        .unwrap()
        .with_options(&DuckLakeWriteOptions::default().with_data_inlining_row_limit(8))
        .append_table("main", "t", &[batch(vec![100, 101], vec![70, 80])])
        .await
        .unwrap();

    let ctx = session(&temp).await;
    assert_eq!(
        scalar_i64(&ctx, "SELECT count(*) FROM ducklake.main.t", 0).await,
        8,
        "inlined rows must be counted"
    );
    assert_eq!(
        scalar_i64(&ctx, "SELECT max(id) FROM ducklake.main.t", 0).await,
        101,
        "an inlined row may hold the maximum"
    );
}

/// A nested column makes this scan's read schema differ from the table's
/// physical schema, because the read-schema mapping rebuilds nested child
/// fields. `count(*)` must still fold: it is answered from `record_count` and
/// has nothing to do with column positions. Only the BOUNDS are positional, and
/// only those are given up.
#[tokio::test(flavor = "multi_thread")]
async fn nested_column_does_not_cost_the_count_fold() {
    use arrow::array::{Array, Int32Builder, ListBuilder};

    let temp = TempDir::new().unwrap();
    let data_path = temp.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();

    let mut list = ListBuilder::new(Int32Builder::new());
    for row in [vec![1, 2], vec![3], vec![4, 5, 6]] {
        for v in row {
            list.values().append_value(v);
        }
        list.append(true);
    }
    let list = list.finish();
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("tags", list.data_type().clone(), true),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(Int32Array::from(vec![1, 2, 3])), Arc::new(list)],
    )
    .unwrap();

    let writer = SqliteMetadataWriter::new_with_init(&conn_str(&temp, true))
        .await
        .unwrap();
    writer.set_data_path(data_path.to_str().unwrap()).unwrap();
    DuckLakeTableWriter::new(Arc::new(writer), object_store())
        .unwrap()
        .write_table("main", "nested", &[batch])
        .await
        .unwrap();

    let ctx = session(&temp).await;
    let sql = "SELECT count(*) FROM ducklake.main.nested";
    let plan = physical_plan(&ctx, sql).await;
    assert!(
        plan.contains("PlaceholderRowExec"),
        "a nested column must not cost the count(*) fold:\n{plan}"
    );
    assert_eq!(scalar_i64(&ctx, sql, 0).await, 3);

    // Whether the bounds fold is not the contract; being right is.
    assert_eq!(
        scalar_i64(&ctx, "SELECT max(id) FROM ducklake.main.nested", 0).await,
        3
    );
}

/// A partition value is not a column statistic. `apply_partition_bounds`
/// synthesises per-file bounds from `ducklake_file_partition_value` so partition
/// columns prune, and official never derives a column bound that way. Those
/// synthesised bounds must never reach the summary and answer `max()`.
///
/// The summary is built before `apply_partition_bounds` runs, so this is safe by
/// construction — this test is what keeps it that way.
#[tokio::test(flavor = "multi_thread")]
async fn partition_derived_bounds_never_answer_max() {
    let temp = TempDir::new().unwrap();
    let data_path = temp.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();

    let writer = SqliteMetadataWriter::new_with_init(&conn_str(&temp, true))
        .await
        .unwrap();
    writer.set_data_path(data_path.to_str().unwrap()).unwrap();

    let cols = vec![
        ColumnDef::from_arrow("id", &DataType::Int32, false).unwrap(),
        ColumnDef::from_arrow("region", &DataType::Utf8, true).unwrap(),
    ];
    let s = writer
        .begin_write_transaction("main", "p", &cols, WriteMode::Replace)
        .unwrap();
    writer
        .publish_snapshot(
            s.table_id,
            "main",
            "p",
            s.snapshot_id,
            WriteMode::Replace,
            s.base_snapshot_id,
            &cols,
            &s.column_ids,
        )
        .unwrap();
    writer
        .set_partition_spec(
            s.table_id,
            &[("region".to_string(), PartitionTransform::Identity)],
        )
        .unwrap();

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("region", DataType::Utf8, true),
    ]));
    let rows = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4])),
            Arc::new(arrow::array::StringArray::from(vec![
                "us", "us", "eu", "eu",
            ])),
        ],
    )
    .unwrap();
    let writer = SqliteMetadataWriter::new(&conn_str(&temp, true))
        .await
        .unwrap();
    DuckLakeTableWriter::new(Arc::new(writer), object_store())
        .unwrap()
        .append_table("main", "p", &[rows])
        .await
        .unwrap();

    let ctx = session(&temp).await;
    assert_eq!(
        scalar_i64(&ctx, "SELECT count(*) FROM ducklake.main.p", 0).await,
        4
    );
    assert_eq!(
        scalar_i64(&ctx, "SELECT max(id) FROM ducklake.main.p", 0).await,
        4
    );

    // `region` is the partition column, so every file carries a single-valued
    // partition bound for it. The answer must still come from the data.
    let batches = ctx
        .sql("SELECT max(region) FROM ducklake.main.p")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let column = arrow::compute::cast(batches[0].column(0), &DataType::Utf8).unwrap();
    let got = column
        .as_any()
        .downcast_ref::<arrow::array::StringArray>()
        .unwrap()
        .value(0)
        .to_string();
    assert_eq!(got, "us", "max(region) must come from the rows");
}

/// A delete file records physical positions, and its `file_path` column is
/// documentation this reader ignores, so one delete file referenced by two data
/// files contributes the other file's positions. Execution already ignored the
/// unmatched ones — it drops a row only when that row's own position is in the
/// set — but the set's SIZE is now the published row count, so an unmatched
/// position would subtract a row that was never removed and `count(*)` would
/// answer below what a scan returns.
#[tokio::test(flavor = "multi_thread")]
async fn delete_position_outside_a_file_is_not_subtracted() {
    let temp = TempDir::new().unwrap();
    let data_path = temp.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();

    // A SMALL file (3 rows, positions 0..2) and a BIG one (6 rows, 0..5), so a
    // delete at position 5 of the big file cannot exist in the small one.
    let writer = SqliteMetadataWriter::new_with_init(&conn_str(&temp, true))
        .await
        .unwrap();
    writer.set_data_path(data_path.to_str().unwrap()).unwrap();
    DuckLakeTableWriter::new(Arc::new(writer), object_store())
        .unwrap()
        .write_table("main", "t", &[batch(vec![1, 2, 3], vec![10, 20, 30])])
        .await
        .unwrap();
    let writer = SqliteMetadataWriter::new(&conn_str(&temp, true))
        .await
        .unwrap();
    DuckLakeTableWriter::new(Arc::new(writer), object_store())
        .unwrap()
        .append_table(
            "main",
            "t",
            &[batch(vec![10, 11, 12, 13, 14, 15], vec![1, 2, 3, 4, 5, 6])],
        )
        .await
        .unwrap();

    let writer = SqliteMetadataWriter::new(&conn_str(&temp, true))
        .await
        .unwrap();
    let provider = SqliteMetadataProvider::new(&conn_str(&temp, true))
        .await
        .unwrap();
    let catalog = DuckLakeCatalog::with_writer(Arc::new(provider), Arc::new(writer)).unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));
    // Last row of the six-row file: physical position 5.
    ctx.sql("DELETE FROM ducklake.main.t WHERE id = 15")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    // Point the SAME delete file at the three-row file as well. Position 5 is
    // not a row it holds.
    let pool = sqlx::SqlitePool::connect(&conn_str(&temp, true))
        .await
        .unwrap();
    let shared = sqlx::query(
        "INSERT INTO ducklake_delete_file \
           (data_file_id, table_id, path, path_is_relative, file_size_bytes, footer_size, \
            encryption_key, delete_count, begin_snapshot, end_snapshot) \
         SELECT (SELECT MIN(data_file_id) FROM ducklake_data_file), d.table_id, d.path, \
                d.path_is_relative, d.file_size_bytes, d.footer_size, d.encryption_key, \
                d.delete_count, d.begin_snapshot, d.end_snapshot \
         FROM ducklake_delete_file d WHERE d.end_snapshot IS NULL",
    )
    .execute(&pool)
    .await
    .map(|r| r.rows_affected())
    .unwrap_or(0);
    pool.close().await;
    assert_eq!(
        shared, 1,
        "fixture must actually share the delete file, or this test proves nothing"
    );

    let ctx = session(&temp).await;
    let folded = scalar_i64(&ctx, "SELECT count(*) FROM ducklake.main.t", 0).await;
    let scanned = scalar_i64(&ctx, "SELECT count(*) FROM ducklake.main.t WHERE id > 0", 0).await;
    assert_eq!(
        folded, scanned,
        "the folded count must equal what a scan returns"
    );
    assert_eq!(folded, 8, "nine rows less the one actually deleted");
}

/// A catalog written by the REAL DuckDB extension, not by this crate.
///
/// Both wrong-result bugs this module exists to prevent were invisible to a
/// suite that only reads catalogs we wrote. Our writer stores NULL for an
/// over-long string bound; DuckDB truncates it to a rounded-up prefix, which is
/// what the spec permits and what `add_data_files` ingests from any parquet
/// writer. So the fixture that breaks the reader is one we cannot produce.
///
/// Two tables, deliberately: `s` carries the long strings and NO deletes, `d`
/// carries the delete. Putting both in one table hides the string case, because
/// a file with deletes already has its bounds suppressed for a different reason
/// and the assertion would pass whether or not string bounds are handled.
///
/// The CLI is pinned by CI (`DUCKDB_CLI_VERSION`, checksummed). The version is
/// asserted here so a developer machine with a different `duckdb` on PATH fails
/// loudly instead of quietly testing another vintage against a reader that has
/// never seen its catalog columns.
#[tokio::test(flavor = "multi_thread")]
async fn duckdb_written_catalog_is_read_correctly() {
    use std::process::Command;

    let version = Command::new("duckdb").arg("--version").output().unwrap();
    let version = String::from_utf8_lossy(&version.stdout).to_string();
    assert!(
        version.contains("v1.5.5"),
        "fixture must be built by the pinned DuckDB CLI (CI installs v1.5.5), found: {version}"
    );

    let temp = TempDir::new().unwrap();
    let catalog_path = temp.path().join("duckdb.db");
    let data_path = temp.path().join("duckdb_data");
    std::fs::create_dir_all(&data_path).unwrap();

    let long = "z".repeat(5000) + "bbb";
    let sql = format!(
        "INSTALL ducklake; LOAD ducklake; \
         ATTACH 'ducklake:sqlite:{cat}' AS lake (DATA_PATH '{data}/', DATA_INLINING_ROW_LIMIT 0); \
         CREATE TABLE lake.main.s(id INTEGER, n INTEGER, v VARCHAR); \
         INSERT INTO lake.main.s VALUES (1, 10, 'aaa'), (2, NULL, '{long}'), (3, 30, 'mmm'); \
         CREATE TABLE lake.main.d(id INTEGER); \
         INSERT INTO lake.main.d VALUES (1), (2), (3), (4); \
         DELETE FROM lake.main.d WHERE id = 4;",
        cat = catalog_path.to_string_lossy(),
        data = data_path.to_string_lossy(),
    );
    let out = Command::new("duckdb")
        .args(["-csv", "-noheader", ":memory:", "-c", &sql])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "duckdb fixture build failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    // The bound DuckDB stored must be a truncated prefix — otherwise this
    // fixture is not exercising the case that broke the reader.
    let pool = sqlx::SqlitePool::connect(&format!("sqlite:{}", catalog_path.display()))
        .await
        .unwrap();
    let stored: Vec<String> = sqlx::query_scalar(
        "SELECT max_value FROM ducklake_file_column_stats WHERE max_value LIKE 'zz%'",
    )
    .fetch_all(&pool)
    .await
    .unwrap();
    pool.close().await;
    assert!(
        stored.iter().any(|bound| bound.len() < long.len()),
        "expected a TRUNCATED string bound from DuckDB, got lengths {:?}",
        stored.iter().map(String::len).collect::<Vec<_>>()
    );

    let provider = SqliteMetadataProvider::new(&format!("sqlite:{}", catalog_path.display()))
        .await
        .unwrap();
    let catalog = DuckLakeCatalog::new(provider).unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));

    // Integer bounds are stored whole, so they may fold — and must be right.
    assert_eq!(
        scalar_i64(&ctx, "SELECT count(*) FROM ducklake.main.s", 0).await,
        3
    );
    assert_eq!(
        scalar_i64(&ctx, "SELECT max(id) FROM ducklake.main.s", 0).await,
        3
    );
    assert_eq!(
        scalar_i64(&ctx, "SELECT min(id) FROM ducklake.main.s", 0).await,
        1
    );

    // The string bound is a truncated prefix on a DELETE-FREE file, so nothing
    // else suppresses it. It must still never answer max(v).
    let sql = "SELECT max(v) FROM ducklake.main.s";
    let plan = physical_plan(&ctx, sql).await;
    assert!(
        plan.contains("DataSourceExec"),
        "a foreign-written string bound must not be folded:\n{plan}"
    );
    let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
    let column = arrow::compute::cast(batches[0].column(0), &DataType::Utf8).unwrap();
    let got = column
        .as_any()
        .downcast_ref::<arrow::array::StringArray>()
        .unwrap()
        .value(0)
        .to_string();
    assert_eq!(
        got, long,
        "max(v) must be the row's value, not the stored bound"
    );

    // A delete recorded by a foreign writer: the count must still be right.
    assert_eq!(
        scalar_i64(&ctx, "SELECT count(*) FROM ducklake.main.d", 0).await,
        3,
        "count must account for DuckDB's own delete"
    );
    assert_eq!(
        scalar_i64(&ctx, "SELECT count(*) FROM ducklake.main.d WHERE id > 0", 0).await,
        3,
        "and a scan must agree with it"
    );
}
