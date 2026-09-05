//! Integration tests for reading DuckLake **data inlining** on the SQLite
//! backend.
//!
//! DuckDB's ducklake extension stores small INSERTs directly in the catalog
//! database (in `ducklake_inlined_data_<tid>_<sv>` tables registered in
//! `ducklake_inlined_data_tables`) instead of Parquet. A reader that only scans
//! `ducklake_data_file` silently undercounts. These tests hand-craft inlined
//! tables exactly as DuckDB would and assert that `SELECT` / `COUNT(*)` include
//! the inlined rows, that inlined-row deletes (`end_snapshot`) are respected, and
//! that time travel is correct — while catalogs with no inlined data are
//! unaffected.

#![cfg(all(feature = "write-sqlite", feature = "metadata-sqlite"))]

use std::process::Command;
use std::sync::Arc;

use arrow::array::types::IntervalMonthDayNano;
use arrow::array::{
    Array, ArrayRef, BinaryViewArray, Date32Array, Decimal128Array, FixedSizeBinaryArray,
    Float32Array, Float64Array, Int32Array, Int64Array, IntervalMonthDayNanoArray,
    LargeBinaryArray, LargeStringArray, ListArray, MapArray, StringViewArray, StructArray,
    Time64MicrosecondArray, TimestampMicrosecondArray, TimestampNanosecondArray, UInt32Array,
    UInt64Array,
};
use arrow::buffer::{NullBuffer, OffsetBuffer, ScalarBuffer};
use arrow::datatypes::{DataType, Field, Fields, IntervalUnit, Schema, TimeUnit};
use arrow::record_batch::RecordBatch;
use datafusion::common::ScalarValue;
use datafusion::prelude::*;
use object_store::local::LocalFileSystem;
use sqlx::sqlite::SqlitePool;
use sqlx::{AssertSqlSafe, Row};
use tempfile::TempDir;

use datafusion_ducklake::{
    ColumnDef, DeleteFileEntry, DuckLakeCatalog, DuckLakeError, DuckLakeTableWriter,
    DuckLakeWriteOptions, InlinedRowRef, MetadataProvider, MetadataWriter, SnapshotCommitMetadata,
    SqliteMetadataProvider, SqliteMetadataWriter, TableWriteOptions, WriteMode,
};

fn table_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("val", DataType::Int32, false),
    ]))
}

#[tokio::test(flavor = "multi_thread")]
async fn sqlite_writer_round_trips_nested_inlined_rows() {
    let temp = TempDir::new().unwrap();
    let writer = Arc::new(make_writer(&temp).await);

    let item_fields = Fields::from(vec![
        Field::new("price", DataType::Decimal128(10, 2), true),
        Field::new(
            "event_time",
            DataType::Timestamp(TimeUnit::Nanosecond, Some("UTC".into())),
            true,
        ),
        Field::new("label", DataType::Utf8View, true),
        Field::new("count", DataType::UInt32, true),
        Field::new("order_id", DataType::UInt64, true),
    ]);
    let items = StructArray::new(
        item_fields.clone(),
        vec![
            Arc::new(
                Decimal128Array::from(vec![Some(12_345), None])
                    .with_precision_and_scale(10, 2)
                    .unwrap(),
            ) as ArrayRef,
            Arc::new(
                TimestampNanosecondArray::from(vec![Some(1_000_002), None]).with_timezone("UTC"),
            ) as ArrayRef,
            Arc::new(StringViewArray::from(vec![Some("a,b'c"), None])) as ArrayRef,
            Arc::new(UInt32Array::from(vec![Some(1), None])) as ArrayRef,
            Arc::new(UInt64Array::from(vec![Some(11), None])) as ArrayRef,
        ],
        None,
    );
    let depths = ListArray::new(
        Arc::new(Field::new("item", DataType::Struct(item_fields), true)),
        OffsetBuffer::new(ScalarBuffer::from(vec![0_i32, 2, 2, 2])),
        Arc::new(items),
        Some(NullBuffer::from(vec![true, true, false])),
    );

    let state_fields = Fields::from(vec![
        Field::new("count", DataType::Int32, true),
        Field::new("note", DataType::Utf8View, true),
    ]);
    let state = StructArray::new(
        state_fields.clone(),
        vec![
            Arc::new(Int32Array::from(vec![Some(7), None, None])) as ArrayRef,
            Arc::new(StringViewArray::from(vec![Some("set"), None, None])) as ArrayRef,
        ],
        Some(NullBuffer::from(vec![true, true, false])),
    );

    let map_fields = Fields::from(vec![
        Field::new("key", DataType::Utf8View, false),
        Field::new("value", DataType::Int32, true),
    ]);
    let attributes = MapArray::new(
        Arc::new(Field::new(
            "entries",
            DataType::Struct(map_fields.clone()),
            false,
        )),
        OffsetBuffer::new(ScalarBuffer::from(vec![0_i32, 2, 2, 2])),
        StructArray::new(
            map_fields,
            vec![
                Arc::new(StringViewArray::from(vec!["a,b", "q'x"])) as ArrayRef,
                Arc::new(Int32Array::from(vec![Some(10), None])) as ArrayRef,
            ],
            None,
        ),
        Some(NullBuffer::from(vec![true, true, false])),
        false,
    );

    let schema = Arc::new(Schema::new(vec![
        Field::new("depths", depths.data_type().clone(), true),
        Field::new("state", DataType::Struct(state_fields), true),
        Field::new("attributes", attributes.data_type().clone(), true),
    ]));
    let expected = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(depths), Arc::new(state), Arc::new(attributes)],
    )
    .unwrap();

    let result = DuckLakeTableWriter::new(writer, object_store())
        .unwrap()
        .with_options(&DuckLakeWriteOptions::default().with_data_inlining_row_limit(3))
        .write_table("main", "nested", std::slice::from_ref(&expected))
        .await
        .unwrap();
    assert_eq!(result.files_written, 0);

    let provider = SqliteMetadataProvider::new(&ro_url(&temp)).await.unwrap();
    let snapshot_id = provider.get_current_snapshot().unwrap();
    let catalog_schema = provider
        .get_schema_by_name("main", snapshot_id)
        .unwrap()
        .unwrap();
    let table = provider
        .get_table_by_name(catalog_schema.schema_id, "nested", snapshot_id)
        .unwrap()
        .unwrap();
    let columns = provider
        .get_table_structure(table.table_id, snapshot_id)
        .unwrap();
    let batches = provider
        .get_inlined_data(table.table_id, snapshot_id, &columns)
        .unwrap();
    assert_eq!(batches, vec![expected]);
}

fn object_store() -> Arc<dyn object_store::ObjectStore> {
    Arc::new(LocalFileSystem::new())
}

fn rw_url(t: &TempDir) -> String {
    format!("sqlite:{}?mode=rwc", t.path().join("test.db").display())
}
fn ro_url(t: &TempDir) -> String {
    format!("sqlite:{}", t.path().join("test.db").display())
}

fn batch(ids: Vec<i32>, vals: Vec<i32>) -> RecordBatch {
    RecordBatch::try_new(
        table_schema(),
        vec![Arc::new(Int32Array::from(ids)), Arc::new(Int32Array::from(vals))],
    )
    .unwrap()
}

async fn make_writer(t: &TempDir) -> SqliteMetadataWriter {
    let data = t.path().join("data");
    std::fs::create_dir_all(&data).unwrap();
    let w = SqliteMetadataWriter::new_with_init(&rw_url(t))
        .await
        .unwrap();
    w.set_data_path(data.to_str().unwrap()).unwrap();
    w
}

fn create_empty_table(writer: &SqliteMetadataWriter, table_name: &str) {
    let columns = vec![
        ColumnDef::from_arrow("id", &DataType::Int32, false).unwrap(),
        ColumnDef::from_arrow("val", &DataType::Int32, false).unwrap(),
    ];
    let setup = writer
        .begin_write_transaction("main", table_name, &columns, WriteMode::Append)
        .unwrap();
    writer
        .publish_snapshot(
            setup.table_id,
            "main",
            table_name,
            setup.snapshot_id,
            WriteMode::Append,
            setup.base_snapshot_id,
            &columns,
            &setup.column_ids,
        )
        .unwrap();
}

/// `(id, val)` from `main.t`, ascending, as of `snapshot` (or latest).
async fn read_rows(t: &TempDir, snapshot: Option<i64>) -> Vec<(i32, i32)> {
    let provider = SqliteMetadataProvider::new(&ro_url(t)).await.unwrap();
    let catalog = match snapshot {
        Some(s) => DuckLakeCatalog::with_snapshot(Arc::new(provider), s).unwrap(),
        None => DuckLakeCatalog::new(provider).unwrap(),
    };
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));
    let batches = ctx
        .sql("SELECT id, val FROM ducklake.main.t ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let mut rows = Vec::new();
    for b in &batches {
        let ids = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        let vals = b.column(1).as_any().downcast_ref::<Int32Array>().unwrap();
        for i in 0..b.num_rows() {
            rows.push((ids.value(i), vals.value(i)));
        }
    }
    rows
}

/// Create the inlining registry + a physical inlined-insert table for `t`, laid
/// out exactly as DuckDB's extension would: `ducklake_inlined_data_<tid>_1(
/// row_id, begin_snapshot, end_snapshot, id, val)`.
async fn seed_inlined(
    pool: &SqlitePool,
    table_id: i64,
    rows: &[(i64, i64, Option<i64>, i32, i32)],
) {
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS ducklake_inlined_data_tables
             (table_id BIGINT, table_name VARCHAR, schema_version BIGINT)",
    )
    .execute(pool)
    .await
    .unwrap();
    let phys = format!("ducklake_inlined_data_{table_id}_1");
    sqlx::query(AssertSqlSafe(format!(
        "CREATE TABLE IF NOT EXISTS {phys}
             (row_id BIGINT, begin_snapshot BIGINT, end_snapshot BIGINT, id INTEGER, val INTEGER)"
    )))
    .execute(pool)
    .await
    .unwrap();
    sqlx::query(
        "INSERT INTO ducklake_inlined_data_tables (table_id, table_name, schema_version)
         VALUES (?, ?, 1)",
    )
    .bind(table_id)
    .bind(&phys)
    .execute(pool)
    .await
    .unwrap();
    for (row_id, begin, end, id, val) in rows {
        sqlx::query(AssertSqlSafe(format!(
            "INSERT INTO {phys} (row_id, begin_snapshot, end_snapshot, id, val) VALUES (?,?,?,?,?)"
        )))
        .bind(row_id)
        .bind(begin)
        .bind(*end)
        .bind(id)
        .bind(val)
        .execute(pool)
        .await
        .unwrap();
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn inlined_rows_are_unioned_into_reads_with_visibility_and_time_travel() {
    let t = TempDir::new().unwrap();
    // Parquet-backed rows: file1 at snapshot 1, file2 at snapshot 2.
    let w = Arc::new(make_writer(&t).await);
    DuckLakeTableWriter::new(w, object_store())
        .unwrap()
        .write_table("main", "t", &[batch(vec![1, 2], vec![10, 20])])
        .await
        .unwrap();
    let w2 = Arc::new(SqliteMetadataWriter::new(&rw_url(&t)).await.unwrap());
    DuckLakeTableWriter::new(w2, object_store())
        .unwrap()
        .append_table("main", "t", &[batch(vec![7, 8], vec![70, 80])])
        .await
        .unwrap();

    let pool = SqlitePool::connect(&rw_url(&t)).await.unwrap();
    let table_id: i64 = sqlx::query_scalar("SELECT table_id FROM ducklake_table LIMIT 1")
        .fetch_one(&pool)
        .await
        .unwrap();

    // Baseline (no inlined data yet): only the Parquet rows.
    assert_eq!(
        read_rows(&t, None).await,
        vec![(1, 10), (2, 20), (7, 70), (8, 80)]
    );

    // Inlined rows (as DuckDB would store them):
    //  - (3,30): live from snapshot 1 (end_snapshot NULL)
    //  - (5,50): inserted at snapshot 1, DELETED at snapshot 2 (end_snapshot = 2)
    seed_inlined(
        &pool,
        table_id,
        &[(100, 1, None, 3, 30), (101, 1, Some(2), 5, 50)],
    )
    .await;

    // At the latest snapshot (2): Parquet rows + the live inlined (3,30); the
    // inlined (5,50) is excluded because it was deleted at snapshot 2.
    assert_eq!(
        read_rows(&t, None).await,
        vec![(1, 10), (2, 20), (3, 30), (7, 70), (8, 80)],
        "inlined live row included; deleted inlined row excluded"
    );

    // Time travel to snapshot 1: only file1's Parquet rows, plus BOTH inlined
    // rows (neither deleted yet at snapshot 1; file2 not yet visible).
    assert_eq!(
        read_rows(&t, Some(1)).await,
        vec![(1, 10), (2, 20), (3, 30), (5, 50)],
        "time travel sees the inlined rows as of that snapshot"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn catalog_without_inlining_is_unaffected() {
    let t = TempDir::new().unwrap();
    let w = Arc::new(make_writer(&t).await);
    DuckLakeTableWriter::new(w, object_store())
        .unwrap()
        .write_table("main", "t", &[batch(vec![1, 2, 3], vec![10, 20, 30])])
        .await
        .unwrap();
    // No ducklake_inlined_data_tables exists -> get_inlined_data returns empty,
    // reads are exactly the Parquet rows.
    assert_eq!(read_rows(&t, None).await, vec![(1, 10), (2, 20), (3, 30)]);
}

#[tokio::test(flavor = "multi_thread")]
async fn flush_inlined_data_preserves_current_and_pinned_reads() {
    let temp = TempDir::new().unwrap();
    let writer = Arc::new(make_writer(&temp).await);
    let options = DuckLakeWriteOptions::default().with_data_inlining_row_limit(10);
    let table_writer = DuckLakeTableWriter::new(writer.clone(), object_store())
        .unwrap()
        .with_options(&options);
    let inline_write = table_writer
        .append_table("main", "t", &[batch(vec![1, 2], vec![10, 20])])
        .await
        .unwrap();
    assert_eq!(inline_write.records_written, 2);
    assert_eq!(read_rows(&temp, None).await, vec![(1, 10), (2, 20)]);

    let provider = SqliteMetadataProvider::new(&ro_url(&temp)).await.unwrap();
    let schema = provider
        .get_schema_by_name("main", inline_write.snapshot_id)
        .unwrap()
        .unwrap();
    let table = provider
        .get_table_by_name(schema.schema_id, "t", inline_write.snapshot_id)
        .unwrap()
        .unwrap();
    let columns = provider
        .get_table_structure(table.table_id, inline_write.snapshot_id)
        .unwrap();
    let inlined = provider
        .get_inlined_data_with_row_ids(table.table_id, inline_write.snapshot_id, &columns)
        .unwrap();
    assert_eq!(
        inlined
            .iter()
            .map(|data| data.batch.num_rows())
            .sum::<usize>(),
        2,
    );

    let flush_writer = DuckLakeTableWriter::new(writer.clone(), object_store()).unwrap();
    let flushed = flush_writer
        .flush_inlined_data("main", "t", &inlined, inline_write.snapshot_id)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(flushed.records_written, 2);
    assert_eq!(flushed.files_written, 1);
    assert_eq!(read_rows(&temp, None).await, vec![(1, 10), (2, 20)]);
    assert_eq!(
        read_rows(&temp, Some(inline_write.snapshot_id)).await,
        vec![(1, 10), (2, 20)],
    );

    let provider = SqliteMetadataProvider::new(&ro_url(&temp)).await.unwrap();
    assert!(
        provider
            .get_inlined_data_with_row_ids(table.table_id, flushed.snapshot_id, &columns)
            .unwrap()
            .is_empty(),
    );
    assert_eq!(
        provider
            .get_inlined_data_with_row_ids(table.table_id, inline_write.snapshot_id, &columns,)
            .unwrap()
            .iter()
            .map(|data| data.batch.num_rows())
            .sum::<usize>(),
        2,
    );
    assert_eq!(
        provider
            .get_table_files_for_select(table.table_id, flushed.snapshot_id)
            .unwrap()
            .len(),
        1,
    );
    assert!(
        flush_writer
            .flush_inlined_data("main", "t", &[], flushed.snapshot_id)
            .await
            .unwrap()
            .is_none(),
    );
    assert_eq!(
        provider.get_current_snapshot().unwrap(),
        flushed.snapshot_id
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn flush_inlined_data_preserves_deletion_history() {
    let temp = TempDir::new().unwrap();
    let writer = Arc::new(make_writer(&temp).await);
    let options = DuckLakeWriteOptions::default().with_data_inlining_row_limit(10);
    let table_writer = DuckLakeTableWriter::new(writer.clone(), object_store())
        .unwrap()
        .with_options(&options);
    let inline_write = table_writer
        .append_table("main", "t", &[batch(vec![1, 2, 3], vec![10, 20, 30])])
        .await
        .unwrap();

    let provider = SqliteMetadataProvider::new(&ro_url(&temp)).await.unwrap();
    let schema = provider
        .get_schema_by_name("main", inline_write.snapshot_id)
        .unwrap()
        .unwrap();
    let table = provider
        .get_table_by_name(schema.schema_id, "t", inline_write.snapshot_id)
        .unwrap()
        .unwrap();
    let columns = provider
        .get_table_structure(table.table_id, inline_write.snapshot_id)
        .unwrap();
    let before_delete = provider
        .get_inlined_data_with_row_ids(table.table_id, inline_write.snapshot_id, &columns)
        .unwrap();
    let ids = before_delete[0]
        .batch
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    let row = (0..ids.len()).find(|index| ids.value(*index) == 2).unwrap();
    let deleted = InlinedRowRef {
        table_name: before_delete[0].table_name.clone(),
        row_id: before_delete[0].row_ids[row],
    };
    let delete_commit = writer
        .commit_inlined_deletes(
            table.table_id,
            "main",
            "t",
            inline_write.snapshot_id,
            &[deleted],
        )
        .unwrap();
    assert_eq!(
        read_rows(&temp, Some(inline_write.snapshot_id)).await,
        vec![(1, 10), (2, 20), (3, 30)],
    );
    assert_eq!(
        read_rows(&temp, Some(delete_commit.snapshot_id)).await,
        vec![(1, 10), (3, 30)],
    );

    let live = provider
        .get_inlined_data_with_row_ids(table.table_id, delete_commit.snapshot_id, &columns)
        .unwrap();
    assert_eq!(
        live.iter().map(|data| data.batch.num_rows()).sum::<usize>(),
        2,
    );
    let flush_writer = DuckLakeTableWriter::new(writer, object_store()).unwrap();
    let flushed = flush_writer
        .flush_inlined_data("main", "t", &live, delete_commit.snapshot_id)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(flushed.records_written, 2);
    assert_eq!(read_rows(&temp, None).await, vec![(1, 10), (3, 30)]);
    assert_eq!(
        read_rows(&temp, Some(inline_write.snapshot_id)).await,
        vec![(1, 10), (2, 20), (3, 30)],
    );
    assert_eq!(
        read_rows(&temp, Some(delete_commit.snapshot_id)).await,
        vec![(1, 10), (3, 30)],
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn flush_inlined_data_conflicts_and_retries_from_fresh_snapshot() {
    let temp = TempDir::new().unwrap();
    let writer = Arc::new(make_writer(&temp).await);
    let inline_options = DuckLakeWriteOptions::default().with_data_inlining_row_limit(10);
    let inline_writer = DuckLakeTableWriter::new(writer.clone(), object_store())
        .unwrap()
        .with_options(&inline_options);
    let inline_write = inline_writer
        .append_table("main", "t", &[batch(vec![1, 2], vec![10, 20])])
        .await
        .unwrap();
    let provider = SqliteMetadataProvider::new(&ro_url(&temp)).await.unwrap();
    let schema = provider
        .get_schema_by_name("main", inline_write.snapshot_id)
        .unwrap()
        .unwrap();
    let table = provider
        .get_table_by_name(schema.schema_id, "t", inline_write.snapshot_id)
        .unwrap()
        .unwrap();
    let columns = provider
        .get_table_structure(table.table_id, inline_write.snapshot_id)
        .unwrap();
    let stale_rows = provider
        .get_inlined_data_with_row_ids(table.table_id, inline_write.snapshot_id, &columns)
        .unwrap();

    let concurrent = DuckLakeTableWriter::new(writer.clone(), object_store())
        .unwrap()
        .append_table("main", "t", &[batch(vec![3], vec![30])])
        .await
        .unwrap();
    let flush_writer = DuckLakeTableWriter::new(writer, object_store()).unwrap();
    let error = flush_writer
        .flush_inlined_data("main", "t", &stale_rows, inline_write.snapshot_id)
        .await
        .unwrap_err();
    assert!(matches!(error, DuckLakeError::Conflict(_)), "{error}");
    assert_eq!(
        read_rows(&temp, None).await,
        vec![(1, 10), (2, 20), (3, 30)],
    );

    let fresh_rows = provider
        .get_inlined_data_with_row_ids(table.table_id, concurrent.snapshot_id, &columns)
        .unwrap();
    let flushed = flush_writer
        .flush_inlined_data("main", "t", &fresh_rows, concurrent.snapshot_id)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(flushed.records_written, 2);
    assert_eq!(
        read_rows(&temp, None).await,
        vec![(1, 10), (2, 20), (3, 30)],
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn writer_inlines_at_limit_and_uses_parquet_outside_limit() {
    let t = TempDir::new().unwrap();
    let writer = Arc::new(make_writer(&t).await);
    let options = DuckLakeWriteOptions::default().with_data_inlining_row_limit(2);
    let result = DuckLakeTableWriter::new(writer, object_store())
        .unwrap()
        .with_options(&options)
        .write_table("main", "t", &[batch(vec![1, 2], vec![10, 20])])
        .await
        .unwrap();
    assert_eq!(result.files_written, 0);
    assert_eq!(result.records_written, 2);
    assert_eq!(read_rows(&t, None).await, vec![(1, 10), (2, 20)]);

    let pool = SqlitePool::connect(&rw_url(&t)).await.unwrap();
    let physical_name: String = sqlx::query_scalar(
        "SELECT table_name FROM ducklake_inlined_data_tables WHERE table_id = ?",
    )
    .bind(result.table_id)
    .fetch_one(&pool)
    .await
    .unwrap();
    let inline_rows: i64 = sqlx::query_scalar(AssertSqlSafe(format!(
        "SELECT COUNT(*) FROM {physical_name} WHERE end_snapshot IS NULL"
    )))
    .fetch_one(&pool)
    .await
    .unwrap();
    let data_files: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM ducklake_data_file WHERE table_id = ?")
            .bind(result.table_id)
            .fetch_one(&pool)
            .await
            .unwrap();
    let stats: (i64, i64, i64) = sqlx::query_as(
        "SELECT record_count, next_row_id, file_size_bytes
         FROM ducklake_table_stats WHERE table_id = ?",
    )
    .bind(result.table_id)
    .fetch_one(&pool)
    .await
    .unwrap();
    let changes: String = sqlx::query_scalar(
        "SELECT changes_made FROM ducklake_snapshot_changes WHERE snapshot_id = ?",
    )
    .bind(result.snapshot_id)
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(inline_rows, 2);
    assert_eq!(data_files, 0);
    assert_eq!(stats, (2, 2, 0));
    // The inline commit records the same composed ledger the Parquet path
    // records: the DDL entries for the snapshot plus the write change.
    assert_eq!(
        changes,
        format!(
            "created_schema:\"main\",created_table:\"main\".\"t\",inserted_into_table:{}",
            result.table_id
        )
    );

    let writer = Arc::new(SqliteMetadataWriter::new(&rw_url(&t)).await.unwrap());
    let over_limit = DuckLakeTableWriter::new(writer, object_store())
        .unwrap()
        .with_options(&options)
        .write_rows(
            "main",
            "parquet_t",
            table_schema().as_ref(),
            WriteMode::Append,
            &[batch(vec![1, 2, 3], vec![10, 20, 30])],
        )
        .await
        .unwrap();
    assert_eq!(over_limit.files_written, 1);
    assert_eq!(over_limit.records_written, 3);

    let disabled = DuckLakeWriteOptions::default().with_data_inlining_row_limit(0);
    let writer = Arc::new(SqliteMetadataWriter::new(&rw_url(&t)).await.unwrap());
    let disabled_result = DuckLakeTableWriter::new(writer, object_store())
        .unwrap()
        .with_options(&disabled)
        .write_rows(
            "main",
            "disabled_t",
            table_schema().as_ref(),
            WriteMode::Append,
            &[batch(vec![1], vec![10])],
        )
        .await
        .unwrap();
    assert_eq!(disabled_result.files_written, 1);
    assert_eq!(disabled_result.records_written, 1);
}

/// A fence-rejected multi-table commit is a DEFINITE rollback: nothing is
/// visible on either table and the staged Parquet objects are removed (only an
/// ambiguous commit failure leaves them to the guarded vacuum).
#[tokio::test(flavor = "multi_thread")]
async fn conflicted_multi_table_commit_leaves_no_partial_state_and_no_staged_files() {
    let temp = TempDir::new().unwrap();
    let writer = Arc::new(make_writer(&temp).await);
    create_empty_table(&writer, "data");
    create_empty_table(&writer, "coverage");
    let provider = SqliteMetadataProvider::new(&rw_url(&temp)).await.unwrap();
    let base = provider.get_current_snapshot().unwrap();

    let writer: Arc<dyn MetadataWriter> = writer;
    let table_writer = DuckLakeTableWriter::new(Arc::clone(&writer), object_store()).unwrap();
    let options = TableWriteOptions::new().with_expected_base_snapshot_id(base);
    let mut transaction = table_writer.transaction().with_options(&options);
    transaction
        .stage_write(
            "main",
            "data",
            table_schema().as_ref(),
            WriteMode::Append,
            &[batch(vec![1], vec![10])],
        )
        .await
        .unwrap();

    // A concurrent commit moves the staged table's generation past the
    // fenced base (the fence is per staged table).
    let concurrent = DuckLakeTableWriter::new(Arc::clone(&writer), object_store()).unwrap();
    concurrent
        .append_table("main", "data", &[batch(vec![9], vec![90])])
        .await
        .unwrap();

    let error = transaction.commit().await.unwrap_err();
    assert!(
        matches!(error, DuckLakeError::Conflict(_)),
        "expected Conflict, got {error:?}"
    );
    // Only the concurrent append's file survives; the rejected stage's
    // Parquet object is removed (definite rollback -> cleanup).
    let staged_parquet = walkdir_parquet(&temp.path().join("data").join("main").join("data"));
    assert_eq!(
        staged_parquet, 1,
        "the fence-rejected stage's Parquet object must be removed"
    );
}

fn walkdir_parquet(dir: &std::path::Path) -> usize {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    entries
        .flatten()
        .map(|entry| {
            let path = entry.path();
            if path.is_dir() {
                walkdir_parquet(&path)
            } else {
                usize::from(path.extension().is_some_and(|ext| ext == "parquet"))
            }
        })
        .sum()
}

/// One multi-table commit creating two tables in a fresh schema records the
/// schema's creation exactly once in the snapshot ledger.
#[tokio::test(flavor = "multi_thread")]
async fn multi_table_commit_records_created_schema_once() {
    let temp = TempDir::new().unwrap();
    let writer = Arc::new(make_writer(&temp).await);
    let table_writer = DuckLakeTableWriter::new(writer, object_store()).unwrap();
    let mut transaction = table_writer.transaction();
    for table_name in ["t1", "t2"] {
        transaction
            .stage_write(
                "s2",
                table_name,
                table_schema().as_ref(),
                WriteMode::Append,
                &[batch(vec![1], vec![10])],
            )
            .await
            .unwrap();
    }

    let results = transaction.commit().await.unwrap();

    assert_eq!(results.len(), 2);
    let pool = SqlitePool::connect(&rw_url(&temp)).await.unwrap();
    let changes: String = sqlx::query_scalar(
        "SELECT changes_made FROM ducklake_snapshot_changes WHERE snapshot_id = ?",
    )
    .bind(results[0].snapshot_id)
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(
        changes.matches("created_schema:\"s2\"").count(),
        1,
        "one commit records one schema creation: {changes}"
    );
    assert_eq!(
        changes,
        format!(
            "created_schema:\"s2\",created_table:\"s2\".\"t1\",created_table:\"s2\".\"t2\",\
             inserted_into_table:{},inserted_into_table:{}",
            results[0].table_id, results[1].table_id
        )
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn multi_table_write_commits_parquet_and_inline_rows_in_one_snapshot() {
    let temp = TempDir::new().unwrap();
    let writer = Arc::new(make_writer(&temp).await);
    create_empty_table(&writer, "data");
    create_empty_table(&writer, "coverage");
    let data_options = DuckLakeWriteOptions::default().with_data_inlining_row_limit(0);
    let coverage_options = DuckLakeWriteOptions::default().with_data_inlining_row_limit(2);
    let table_writer = DuckLakeTableWriter::new(writer, object_store()).unwrap();
    let mut transaction = table_writer.transaction();
    transaction
        .stage_write_with_options(
            "main",
            "data",
            table_schema().as_ref(),
            WriteMode::Append,
            &[batch(vec![1], vec![10])],
            &data_options,
        )
        .await
        .unwrap();
    transaction
        .stage_write_with_options(
            "main",
            "coverage",
            table_schema().as_ref(),
            WriteMode::Append,
            &[batch(vec![1], vec![10])],
            &coverage_options,
        )
        .await
        .unwrap();

    let results = transaction.commit().await.unwrap();

    assert_eq!(results.len(), 2);
    assert_eq!(results[0].snapshot_id, results[1].snapshot_id);
    assert_eq!(results[0].files_written, 1);
    assert_eq!(results[1].files_written, 0);
    let pool = SqlitePool::connect(&rw_url(&temp)).await.unwrap();
    let snapshots: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM ducklake_snapshot")
        .fetch_one(&pool)
        .await
        .unwrap();
    let files: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM ducklake_data_file")
        .fetch_one(&pool)
        .await
        .unwrap();
    let inlined_tables: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM ducklake_inlined_data_tables")
            .fetch_one(&pool)
            .await
            .unwrap();
    let changes: String = sqlx::query_scalar(
        "SELECT changes_made FROM ducklake_snapshot_changes WHERE snapshot_id = ?",
    )
    .bind(results[0].snapshot_id)
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(snapshots, 3);
    assert_eq!(files, 1);
    assert_eq!(inlined_tables, 1);
    assert_eq!(
        changes,
        format!(
            "inserted_into_table:{},inserted_into_table:{}",
            results[0].table_id, results[1].table_id
        )
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn multi_table_fence_rejection_removes_staged_parquet_file() {
    let temp = TempDir::new().unwrap();
    let writer = Arc::new(make_writer(&temp).await);
    create_empty_table(&writer, "coverage");
    let options = DuckLakeWriteOptions::default().with_data_inlining_row_limit(2);
    let table_writer = DuckLakeTableWriter::new(writer, object_store())
        .unwrap()
        .with_options(&options);
    let initial = table_writer
        .append_table("main", "data", &[batch(vec![1, 2, 3], vec![10, 20, 30])])
        .await
        .unwrap();
    let pool = SqlitePool::connect(&rw_url(&temp)).await.unwrap();
    let (data_file_id, data_file_path): (i64, String) = sqlx::query_as(
        "SELECT data_file_id, path FROM ducklake_data_file
         WHERE table_id = ? AND end_snapshot IS NULL",
    )
    .bind(initial.table_id)
    .fetch_one(&pool)
    .await
    .unwrap();
    let delete = table_writer
        .write_delete_file("main", "data", &data_file_path, &[0])
        .await
        .unwrap();
    let deletes = [DeleteFileEntry {
        data_file_id,
        expected_prev_delete_file: None,
        delete,
    }];
    let transaction_options =
        TableWriteOptions::new().with_expected_base_snapshot_id(initial.snapshot_id);
    let mut transaction = table_writer
        .transaction()
        .with_options(&transaction_options);
    transaction
        .stage_write_with_deletes(
            "main",
            "data",
            table_schema().as_ref(),
            WriteMode::Append,
            &[batch(vec![4, 5, 6], vec![40, 50, 60])],
            &deletes,
            &[],
        )
        .await
        .unwrap();
    transaction
        .stage_write(
            "main",
            "coverage",
            table_schema().as_ref(),
            WriteMode::Append,
            &[batch(vec![1], vec![10])],
        )
        .await
        .unwrap();
    table_writer
        .append_table("main", "data", &[batch(vec![7, 8, 9], vec![70, 80, 90])])
        .await
        .unwrap();

    let error = transaction.commit().await.unwrap_err();

    assert!(error.to_string().contains("conflict"));
    let data_dir = temp.path().join("data/main/data");
    let parquet_files = std::fs::read_dir(data_dir)
        .unwrap()
        .filter_map(std::result::Result::ok)
        .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "parquet"))
        .count();
    assert_eq!(parquet_files, 2);
    let pool = SqlitePool::connect(&rw_url(&temp)).await.unwrap();
    let coverage_rows: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM ducklake_inlined_data_tables tables
         JOIN ducklake_table table_meta ON table_meta.table_id = tables.table_id
         WHERE table_meta.table_name = 'coverage'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(coverage_rows, 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn multi_table_delete_only_stage_ends_inline_row() {
    let temp = TempDir::new().unwrap();
    let writer = Arc::new(make_writer(&temp).await);
    create_empty_table(&writer, "data");
    let options = DuckLakeWriteOptions::default().with_data_inlining_row_limit(2);
    let table_writer = DuckLakeTableWriter::new(writer, object_store())
        .unwrap()
        .with_options(&options);
    let coverage = table_writer
        .append_table("main", "coverage", &[batch(vec![1, 2], vec![10, 20])])
        .await
        .unwrap();
    let pool = SqlitePool::connect(&rw_url(&temp)).await.unwrap();
    let physical_name: String = sqlx::query_scalar(
        "SELECT table_name FROM ducklake_inlined_data_tables WHERE table_id = ?",
    )
    .bind(coverage.table_id)
    .fetch_one(&pool)
    .await
    .unwrap();
    let row_id: i64 = sqlx::query_scalar(AssertSqlSafe(format!(
        "SELECT row_id FROM {physical_name} WHERE id = 1 AND end_snapshot IS NULL"
    )))
    .fetch_one(&pool)
    .await
    .unwrap();
    let mut transaction = table_writer.transaction();
    transaction
        .stage_write(
            "main",
            "data",
            table_schema().as_ref(),
            WriteMode::Append,
            &[batch(vec![3, 4, 5], vec![30, 40, 50])],
        )
        .await
        .unwrap();
    transaction
        .stage_deletes(
            "main",
            "coverage",
            table_schema().as_ref(),
            &[],
            &[InlinedRowRef {
                table_name: physical_name.clone(),
                row_id,
            }],
        )
        .unwrap();

    let results = transaction.commit().await.unwrap();

    assert_eq!(results.len(), 2);
    assert_eq!(results[0].snapshot_id, results[1].snapshot_id);
    let live_ids: Vec<i64> = sqlx::query_scalar(AssertSqlSafe(format!(
        "SELECT id FROM {physical_name} WHERE end_snapshot IS NULL ORDER BY id"
    )))
    .fetch_all(&pool)
    .await
    .unwrap();
    assert_eq!(live_ids, vec![2]);
}

#[tokio::test(flavor = "multi_thread")]
async fn sqlite_inlined_uint64_round_trips_text_storage() {
    let temp = TempDir::new().unwrap();
    let writer = Arc::new(make_writer(&temp).await);
    let options = DuckLakeWriteOptions::default().with_data_inlining_row_limit(3);
    let schema = Arc::new(Schema::new(vec![
        Field::new("identifier", DataType::Utf8View, false),
        Field::new("value", DataType::UInt64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringViewArray::from(vec!["A", "A", "B"])),
            Arc::new(UInt64Array::from(vec![0, i64::MAX as u64 + 1, u64::MAX])),
        ],
    )
    .unwrap();
    DuckLakeTableWriter::new(writer, object_store())
        .unwrap()
        .with_options(&options)
        .append_table("main", "uint64_values", &[batch])
        .await
        .unwrap();
    let provider = SqliteMetadataProvider::new(&ro_url(&temp)).await.unwrap();
    let snapshot = provider.get_current_snapshot().unwrap();
    let catalog_schema = provider
        .get_schema_by_name("main", snapshot)
        .unwrap()
        .unwrap();
    let table = provider
        .get_table_by_name(catalog_schema.schema_id, "uint64_values", snapshot)
        .unwrap()
        .unwrap();
    let pool = SqlitePool::connect(&format!(
        "sqlite:{}?mode=rwc",
        temp.path().join("test.db").display()
    ))
    .await
    .unwrap();
    let physical: String = sqlx::query_scalar(
        "SELECT table_name FROM ducklake_inlined_data_tables WHERE table_id = ?",
    )
    .bind(table.table_id)
    .fetch_one(&pool)
    .await
    .unwrap();
    let declared_type: String = sqlx::query_scalar(AssertSqlSafe(format!(
        "SELECT type FROM pragma_table_info('{}') WHERE name = 'value'",
        physical
    )))
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(declared_type, "TEXT");
    let stored: Vec<String> = sqlx::query_scalar(AssertSqlSafe(format!(
        "SELECT value FROM {physical} ORDER BY row_id"
    )))
    .fetch_all(&pool)
    .await
    .unwrap();
    assert_eq!(
        stored,
        vec!["00000000000000000000", "09223372036854775808", "18446744073709551615",]
    );

    let index_writer = SqliteMetadataWriter::new(&format!(
        "sqlite:{}?mode=rwc",
        temp.path().join("test.db").display()
    ))
    .await
    .unwrap();
    index_writer
        .set_inlined_index_columns(
            table.table_id,
            &["identifier".to_string(), "value".to_string()],
        )
        .unwrap();
    index_writer.ensure_inlined_indexes(table.table_id).unwrap();
    index_writer.ensure_inlined_indexes(table.table_id).unwrap();
    let indexes: Vec<String> = sqlx::query_scalar(
        "SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = ? ORDER BY name",
    )
    .bind(&physical)
    .fetch_all(&pool)
    .await
    .unwrap();
    assert_eq!(
        indexes,
        vec![
            format!("{physical}_identifier_idx"),
            format!("{physical}_row_id_idx"),
            format!("{physical}_value_idx"),
        ]
    );
    sqlx::query(AssertSqlSafe(format!(
        "WITH RECURSIVE seq(value) AS (
           SELECT 1 UNION ALL SELECT value + 1 FROM seq WHERE value < 1000
         )
         INSERT INTO {physical}(row_id, begin_snapshot, end_snapshot, identifier, value)
         SELECT value + 100, ?, NULL, 'A', printf('%020d', value) FROM seq"
    )))
    .bind(snapshot)
    .execute(&pool)
    .await
    .unwrap();
    sqlx::query(AssertSqlSafe(format!("ANALYZE {physical}")))
        .execute(&pool)
        .await
        .unwrap();
    let coverage_plan = sqlx::query(AssertSqlSafe(format!(
        "EXPLAIN QUERY PLAN SELECT row_id FROM {physical}
         WHERE identifier = ?
           AND value >= substr('00000000000000000000' || ?, -20, 20)"
    )))
    .bind("A")
    .bind((i64::MAX as u64 + 1).to_string())
    .fetch_all(&pool)
    .await
    .unwrap()
    .into_iter()
    .map(|row| row.try_get::<String, _>(3).unwrap())
    .collect::<Vec<_>>();
    assert!(
        coverage_plan.iter().any(|detail| {
            detail.contains(&format!("{physical}_identifier_idx"))
                || detail.contains(&format!("{physical}_value_idx"))
        }),
        "{coverage_plan:?}"
    );
    let row_id_plan = sqlx::query(AssertSqlSafe(format!(
        "EXPLAIN QUERY PLAN UPDATE {physical} SET end_snapshot = ? WHERE row_id = ?"
    )))
    .bind(snapshot)
    .bind(1_i64)
    .fetch_all(&pool)
    .await
    .unwrap()
    .into_iter()
    .map(|row| row.try_get::<String, _>(3).unwrap())
    .collect::<Vec<_>>();
    assert!(
        row_id_plan
            .iter()
            .any(|detail| detail.contains(&format!("{physical}_row_id_idx"))),
        "{row_id_plan:?}"
    );
    sqlx::query(AssertSqlSafe(format!(
        "DELETE FROM {physical} WHERE row_id >= 101"
    )))
    .execute(&pool)
    .await
    .unwrap();

    let catalog = DuckLakeCatalog::new(provider).unwrap();
    let context = SessionContext::new();
    context.register_catalog("ducklake", Arc::new(catalog));

    let batches = context
        .sql("SELECT value FROM ducklake.main.uint64_values")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 3);
    let values = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap();
    assert_eq!(values.values(), &[0, i64::MAX as u64 + 1, u64::MAX]);
}

#[tokio::test(flavor = "multi_thread")]
async fn sqlite_uint64_reads_mixed_legacy_and_padded_tables() {
    let temp = TempDir::new().unwrap();
    let writer = Arc::new(make_writer(&temp).await);
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::UInt64,
            false,
        )])),
        vec![Arc::new(UInt64Array::from(vec![0, u64::MAX]))],
    )
    .unwrap();
    let result = DuckLakeTableWriter::new(writer, object_store())
        .unwrap()
        .with_options(&DuckLakeWriteOptions::default().with_data_inlining_row_limit(2))
        .append_table("main", "mixed_uint64", &[batch])
        .await
        .unwrap();

    let pool = SqlitePool::connect(&format!(
        "sqlite:{}?mode=rwc",
        temp.path().join("test.db").display()
    ))
    .await
    .unwrap();
    let legacy = format!("ducklake_inlined_data_{}_0", result.table_id);
    sqlx::query(AssertSqlSafe(format!(
        "CREATE TABLE {legacy}(
           row_id BIGINT NOT NULL,
           begin_snapshot BIGINT NOT NULL,
           end_snapshot BIGINT,
           value VARCHAR
         )"
    )))
    .execute(&pool)
    .await
    .unwrap();
    sqlx::query(AssertSqlSafe(format!(
        "INSERT INTO {legacy} VALUES (100, ?, NULL, '7'), (101, ?, NULL, '9223372036854775808')"
    )))
    .bind(result.snapshot_id)
    .bind(result.snapshot_id)
    .execute(&pool)
    .await
    .unwrap();
    sqlx::query(
        "INSERT INTO ducklake_inlined_data_tables(table_id, table_name, schema_version)
         VALUES (?, ?, 0)",
    )
    .bind(result.table_id)
    .bind(&legacy)
    .execute(&pool)
    .await
    .unwrap();
    // A post-upgrade writer insert lands zero-padded text in the legacy
    // VARCHAR table; equality pushdown must still find it.
    sqlx::query(AssertSqlSafe(format!(
        "INSERT INTO {legacy} VALUES (102, ?, NULL, '00000000000000000042')"
    )))
    .bind(result.snapshot_id)
    .execute(&pool)
    .await
    .unwrap();

    let provider = SqliteMetadataProvider::new(&ro_url(&temp)).await.unwrap();
    let snapshot = provider.get_current_snapshot().unwrap();
    let columns = provider
        .get_table_structure(result.table_id, snapshot)
        .unwrap();
    let batches = provider
        .get_inlined_data(result.table_id, snapshot, &columns)
        .unwrap();
    let mut actual = batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(0)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap()
                .values()
                .iter()
                .copied()
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    actual.sort_unstable();
    assert_eq!(actual, vec![0, 7, 42, i64::MAX as u64 + 1, u64::MAX]);
}

#[tokio::test(flavor = "multi_thread")]
async fn sqlite_temporal_values_round_trip_across_native_and_legacy_tables() {
    let temp = TempDir::new().unwrap();
    let writer = Arc::new(make_writer(&temp).await);
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("event_date", DataType::Date32, false),
            Field::new("event_time", DataType::Time64(TimeUnit::Microsecond), false),
            Field::new(
                "event_us",
                DataType::Timestamp(TimeUnit::Microsecond, None),
                false,
            ),
            Field::new(
                "event_ns",
                DataType::Timestamp(TimeUnit::Nanosecond, None),
                false,
            ),
        ])),
        vec![
            Arc::new(Date32Array::from(vec![1])),
            Arc::new(Time64MicrosecondArray::from(vec![1_000_002])),
            Arc::new(TimestampMicrosecondArray::from(vec![1_000_002])),
            Arc::new(TimestampNanosecondArray::from(vec![1_000_002_003])),
        ],
    )
    .unwrap();
    let result = DuckLakeTableWriter::new(writer, object_store())
        .unwrap()
        .with_options(&DuckLakeWriteOptions::default().with_data_inlining_row_limit(1))
        .append_table("main", "mixed_temporal", &[batch])
        .await
        .unwrap();

    let pool = SqlitePool::connect(&rw_url(&temp)).await.unwrap();
    let legacy = format!("ducklake_inlined_data_{}_legacy", result.table_id);
    sqlx::query(AssertSqlSafe(format!(
        "CREATE TABLE {legacy}(\
             row_id BIGINT NOT NULL,\
             begin_snapshot BIGINT NOT NULL,\
             end_snapshot BIGINT,\
             event_date VARCHAR,\
             event_time VARCHAR,\
             event_us VARCHAR,\
             event_ns VARCHAR\
         )"
    )))
    .execute(&pool)
    .await
    .unwrap();
    sqlx::query(AssertSqlSafe(format!(
        "INSERT INTO {legacy} VALUES (\
             100, ?, NULL, '1970-01-03', '00:00:02.000003',\
             '1970-01-01T00:00:02.000003', '1970-01-01T00:00:02.000003004'\
         )"
    )))
    .bind(result.snapshot_id)
    .execute(&pool)
    .await
    .unwrap();
    sqlx::query(
        "INSERT INTO ducklake_inlined_data_tables(table_id, table_name, schema_version) \
         VALUES (?, ?, 0)",
    )
    .bind(result.table_id)
    .bind(&legacy)
    .execute(&pool)
    .await
    .unwrap();

    let provider = SqliteMetadataProvider::new(&ro_url(&temp)).await.unwrap();
    let columns = provider
        .get_table_structure(result.table_id, result.snapshot_id)
        .unwrap();
    let batches = provider
        .get_inlined_data(result.table_id, result.snapshot_id, &columns)
        .unwrap();
    let mut dates = batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(0)
                .as_any()
                .downcast_ref::<Date32Array>()
                .unwrap()
                .values()
                .iter()
                .copied()
        })
        .collect::<Vec<_>>();
    dates.sort_unstable();
    assert_eq!(dates, vec![1, 2]);
    let mut timestamps = batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(3)
                .as_any()
                .downcast_ref::<TimestampNanosecondArray>()
                .unwrap()
                .values()
                .iter()
                .copied()
        })
        .collect::<Vec<_>>();
    timestamps.sort_unstable();
    assert_eq!(timestamps, vec![1_000_002_003, 2_000_003_004]);
}

#[tokio::test(flavor = "multi_thread")]
async fn sqlite_declared_indexes_apply_when_inline_table_is_created_later() {
    let temp = TempDir::new().unwrap();
    let writer = Arc::new(make_writer(&temp).await);
    let schema = Arc::new(Schema::new(vec![Field::new(
        "value",
        DataType::UInt64,
        false,
    )]));
    let initial = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(UInt64Array::from(vec![1]))],
    )
    .unwrap();
    let created = DuckLakeTableWriter::new(writer.clone(), object_store())
        .unwrap()
        .with_options(&DuckLakeWriteOptions::default().with_data_inlining_row_limit(0))
        .write_table("main", "declared_indexes", &[initial])
        .await
        .unwrap();
    writer
        .set_inlined_index_columns(created.table_id, &["value".to_string()])
        .unwrap();
    let inlined =
        RecordBatch::try_new(schema, vec![Arc::new(UInt64Array::from(vec![u64::MAX]))]).unwrap();
    DuckLakeTableWriter::new(writer, object_store())
        .unwrap()
        .with_options(&DuckLakeWriteOptions::default().with_data_inlining_row_limit(1))
        .append_table("main", "declared_indexes", &[inlined])
        .await
        .unwrap();

    let pool = SqlitePool::connect(&format!(
        "sqlite:{}?mode=rwc",
        temp.path().join("test.db").display()
    ))
    .await
    .unwrap();
    let physical: String = sqlx::query_scalar(
        "SELECT table_name FROM ducklake_inlined_data_tables WHERE table_id = ?",
    )
    .bind(created.table_id)
    .fetch_one(&pool)
    .await
    .unwrap();
    let indexes: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM sqlite_master
         WHERE type = 'index' AND tbl_name = ? AND name IN (?, ?)",
    )
    .bind(&physical)
    .bind(format!("{physical}_row_id_idx"))
    .bind(format!("{physical}_value_idx"))
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(indexes, 2);
}

#[tokio::test(flavor = "multi_thread")]
async fn inline_snapshot_column_uses_the_committed_snapshot() {
    let temp = TempDir::new().unwrap();
    let writer = Arc::new(make_writer(&temp).await);
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("version", DataType::Int64, true),
    ]));
    let columns = vec![
        ColumnDef::from_arrow("id", &DataType::Int32, false).unwrap(),
        ColumnDef::from_arrow("version", &DataType::Int64, true).unwrap(),
    ];
    let setup = writer
        .begin_write_transaction("main", "versioned", &columns, WriteMode::Append)
        .unwrap();
    writer
        .publish_snapshot(
            setup.table_id,
            "main",
            "versioned",
            setup.snapshot_id,
            WriteMode::Append,
            setup.base_snapshot_id,
            &columns,
            &setup.column_ids,
        )
        .unwrap();
    let options = DuckLakeWriteOptions::default().with_data_inlining_row_limit(2);
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2])),
            Arc::new(Int64Array::from(vec![None, Some(77)])),
        ],
    )
    .unwrap();
    let table_writer = DuckLakeTableWriter::new(writer, object_store()).unwrap();
    let mut transaction = table_writer.transaction();
    transaction
        .stage_write_with_snapshot_columns(
            "main",
            "versioned",
            schema.as_ref(),
            WriteMode::Append,
            &[batch],
            &options,
            &["version"],
        )
        .await
        .unwrap();

    let committed = transaction.commit().await.unwrap();

    let provider = SqliteMetadataProvider::new(&ro_url(&temp)).await.unwrap();
    let catalog = DuckLakeCatalog::new(provider).unwrap();
    let context = SessionContext::new();
    context.register_catalog("ducklake", Arc::new(catalog));
    let batches = context
        .sql("SELECT version FROM ducklake.main.versioned ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(batches.len(), 1);
    let versions = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(versions.values(), &[committed[0].snapshot_id, 77]);
}

#[tokio::test(flavor = "multi_thread")]
async fn delete_ends_inlined_rows_and_updates_stats() {
    let t = TempDir::new().unwrap();
    let writer = Arc::new(make_writer(&t).await);
    let options = DuckLakeWriteOptions::default().with_data_inlining_row_limit(10);
    let written = DuckLakeTableWriter::new(writer, object_store())
        .unwrap()
        .with_options(&options)
        .write_table("main", "t", &[batch(vec![1, 2], vec![10, 20])])
        .await
        .unwrap();

    let writer = SqliteMetadataWriter::new(&rw_url(&t)).await.unwrap();
    let provider = SqliteMetadataProvider::new(&rw_url(&t)).await.unwrap();
    let catalog = DuckLakeCatalog::with_writer(Arc::new(provider), Arc::new(writer)).unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));
    let deleted = ctx
        .sql("DELETE FROM ducklake.main.t WHERE id = 2")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let count = deleted[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap();
    assert_eq!(count.value(0), 1);
    assert_eq!(read_rows(&t, None).await, vec![(1, 10)]);

    let pool = SqlitePool::connect(&rw_url(&t)).await.unwrap();
    let physical_name: String = sqlx::query_scalar(
        "SELECT table_name FROM ducklake_inlined_data_tables WHERE table_id = ?",
    )
    .bind(written.table_id)
    .fetch_one(&pool)
    .await
    .unwrap();
    let ended: (Option<i64>, Option<i64>) = sqlx::query_as(AssertSqlSafe(format!(
        "SELECT MIN(end_snapshot), MAX(end_snapshot) FROM {physical_name} WHERE id = 2"
    )))
    .fetch_one(&pool)
    .await
    .unwrap();
    let stats: (i64, i64) = sqlx::query_as(
        "SELECT record_count, next_row_id FROM ducklake_table_stats WHERE table_id = ?",
    )
    .bind(written.table_id)
    .fetch_one(&pool)
    .await
    .unwrap();
    let changes: String = sqlx::query_scalar(
        "SELECT changes_made FROM ducklake_snapshot_changes ORDER BY snapshot_id DESC LIMIT 1",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    let snapshot: i64 = sqlx::query_scalar("SELECT MAX(snapshot_id) FROM ducklake_snapshot")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(ended, (Some(snapshot), Some(snapshot)));
    assert_eq!(stats, (1, 2));
    assert_eq!(changes, format!("deleted_from_table:{}", written.table_id));
}

#[tokio::test(flavor = "multi_thread")]
async fn delete_all_ends_every_inlined_row() {
    let t = TempDir::new().unwrap();
    let writer = Arc::new(make_writer(&t).await);
    let options = DuckLakeWriteOptions::default().with_data_inlining_row_limit(10);
    DuckLakeTableWriter::new(writer, object_store())
        .unwrap()
        .with_options(&options)
        .write_table("main", "t", &[batch(vec![1, 2], vec![10, 20])])
        .await
        .unwrap();
    let writer = SqliteMetadataWriter::new(&rw_url(&t)).await.unwrap();
    let provider = SqliteMetadataProvider::new(&rw_url(&t)).await.unwrap();
    let catalog = DuckLakeCatalog::with_writer(Arc::new(provider), Arc::new(writer)).unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));
    let deleted = ctx
        .sql("DELETE FROM ducklake.main.t")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let count = deleted[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap();
    assert_eq!(count.value(0), 2);
    assert!(read_rows(&t, None).await.is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn delete_commits_parquet_and_inlined_rows_atomically() {
    let t = TempDir::new().unwrap();
    let options = DuckLakeWriteOptions::default().with_data_inlining_row_limit(2);
    let writer = Arc::new(make_writer(&t).await);
    DuckLakeTableWriter::new(writer, object_store())
        .unwrap()
        .with_options(&options)
        .write_table("main", "t", &[batch(vec![1, 2], vec![10, 20])])
        .await
        .unwrap();
    let writer = Arc::new(SqliteMetadataWriter::new(&rw_url(&t)).await.unwrap());
    DuckLakeTableWriter::new(writer, object_store())
        .unwrap()
        .with_options(&options)
        .append_table("main", "t", &[batch(vec![3, 4, 5], vec![30, 40, 50])])
        .await
        .unwrap();

    let writer = SqliteMetadataWriter::new(&rw_url(&t)).await.unwrap();
    let provider = SqliteMetadataProvider::new(&rw_url(&t)).await.unwrap();
    let catalog = DuckLakeCatalog::with_writer(Arc::new(provider), Arc::new(writer)).unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));
    let deleted = ctx
        .sql("DELETE FROM ducklake.main.t WHERE id IN (2, 3)")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let count = deleted[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap();
    assert_eq!(count.value(0), 2);
    assert_eq!(read_rows(&t, None).await, vec![(1, 10), (4, 40), (5, 50)]);

    let pool = SqlitePool::connect(&rw_url(&t)).await.unwrap();
    let delete_snapshots: (i64, i64) =
        sqlx::query_as("SELECT MIN(begin_snapshot), MAX(begin_snapshot) FROM ducklake_delete_file")
            .fetch_one(&pool)
            .await
            .unwrap();
    let inlined_end: i64 =
        sqlx::query_scalar("SELECT end_snapshot FROM ducklake_inlined_data_1_1 WHERE id = 2")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(delete_snapshots.0, delete_snapshots.1);
    assert_eq!(inlined_end, delete_snapshots.0);
}

#[tokio::test(flavor = "multi_thread")]
async fn inlined_float_and_binary_view_columns_round_trip() {
    let t = TempDir::new().unwrap();
    let writer = Arc::new(make_writer(&t).await);
    let options = DuckLakeWriteOptions::default().with_data_inlining_row_limit(10);
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("double_val", DataType::Float64, true),
        Field::new("float_val", DataType::Float32, true),
        Field::new("payload", DataType::BinaryView, true),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![1, 2])),
            Arc::new(Float64Array::from(vec![Some(1.5), None])),
            Arc::new(Float32Array::from(vec![Some(-2.25_f32), None])),
            Arc::new(
                vec![Some(&[0x00_u8, 0xff][..]), None]
                    .into_iter()
                    .collect::<BinaryViewArray>(),
            ),
        ],
    )
    .unwrap();
    let result = DuckLakeTableWriter::new(writer, object_store())
        .unwrap()
        .with_options(&options)
        .write_table("main", "typed", &[batch])
        .await
        .unwrap();
    assert_eq!(result.files_written, 0);
    assert_eq!(result.records_written, 2);

    let provider = SqliteMetadataProvider::new(&ro_url(&t)).await.unwrap();
    let catalog = DuckLakeCatalog::new(provider).unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));
    let batches = ctx
        .sql("SELECT id, double_val, float_val, payload FROM ducklake.main.typed ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(batches.len(), 1);
    let batch = &batches[0];
    assert_eq!(batch.num_rows(), 2);
    let doubles = batch
        .column(1)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    let floats = batch
        .column(2)
        .as_any()
        .downcast_ref::<Float32Array>()
        .unwrap();
    let payloads = batch
        .column(3)
        .as_any()
        .downcast_ref::<BinaryViewArray>()
        .unwrap();
    assert_eq!(doubles.value(0), 1.5);
    assert!(doubles.is_null(1));
    assert_eq!(floats.value(0), -2.25_f32);
    assert!(floats.is_null(1));
    assert_eq!(payloads.value(0), &[0x00_u8, 0xff]);
    assert!(payloads.is_null(1));
}

#[tokio::test(flavor = "multi_thread")]
async fn timestamp_columns_round_trip_inlined() {
    let t = TempDir::new().unwrap();
    let writer = Arc::new(make_writer(&t).await);
    let options = DuckLakeWriteOptions::default().with_data_inlining_row_limit(10);
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("ts", DataType::Timestamp(TimeUnit::Microsecond, None), true),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![1])),
            Arc::new(TimestampMicrosecondArray::from(vec![Some(1_000_002)])),
        ],
    )
    .unwrap();
    let result = DuckLakeTableWriter::new(writer, object_store())
        .unwrap()
        .with_options(&options)
        .write_table("main", "events", &[batch])
        .await
        .unwrap();
    assert_eq!(
        result.files_written, 0,
        "a small write with a timestamp column should stay inlined"
    );
    assert_eq!(result.records_written, 1);

    let pool = SqlitePool::connect(&rw_url(&t)).await.unwrap();
    let inline_tables: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM ducklake_inlined_data_tables")
            .fetch_one(&pool)
            .await
            .unwrap();
    let data_files: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM ducklake_data_file")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(inline_tables, 1);
    assert_eq!(data_files, 0);

    let provider = SqliteMetadataProvider::new(&ro_url(&t)).await.unwrap();
    let snapshot_id = provider.get_current_snapshot().unwrap();
    let schema = provider
        .get_schema_by_name("main", snapshot_id)
        .unwrap()
        .unwrap();
    let table = provider
        .get_table_by_name(schema.schema_id, "events", snapshot_id)
        .unwrap()
        .unwrap();
    let columns = provider
        .get_table_structure(table.table_id, snapshot_id)
        .unwrap();
    let batches = provider
        .get_inlined_data(table.table_id, snapshot_id, &columns)
        .unwrap();
    let timestamps = batches[0]
        .column(1)
        .as_any()
        .downcast_ref::<TimestampMicrosecondArray>()
        .unwrap();
    assert_eq!(timestamps.value(0), 1_000_002);
}

#[tokio::test(flavor = "multi_thread")]
async fn sqlite_writer_round_trips_supported_scalar_inlined_rows() {
    let temp = TempDir::new().unwrap();
    let writer = Arc::new(make_writer(&temp).await);
    let decimal = Decimal128Array::from(vec![Some(12_345)])
        .with_precision_and_scale(10, 2)
        .unwrap();
    let uuid = [
        0x55, 0x0e, 0x84, 0x00, 0xe2, 0x9b, 0x41, 0xd4, 0xa7, 0x16, 0x44, 0x66, 0x55, 0x44, 0x00,
        0x00,
    ];
    let schema = Arc::new(Schema::new(vec![
        Field::new("decimal", DataType::Decimal128(10, 2), false),
        Field::new("date", DataType::Date32, false),
        Field::new("time", DataType::Time64(TimeUnit::Microsecond), false),
        Field::new(
            "timestamp_ns",
            DataType::Timestamp(TimeUnit::Nanosecond, Some("UTC".into())),
            false,
        ),
        Field::new(
            "interval",
            DataType::Interval(IntervalUnit::MonthDayNano),
            false,
        ),
        Field::new("large_text", DataType::LargeUtf8, false),
        Field::new("large_binary", DataType::LargeBinary, false),
        Field::new("uuid", DataType::FixedSizeBinary(16), false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(decimal),
            Arc::new(Date32Array::from(vec![1])),
            Arc::new(Time64MicrosecondArray::from(vec![1_000_002])),
            Arc::new(TimestampNanosecondArray::from(vec![1_000_002]).with_timezone("UTC")),
            Arc::new(IntervalMonthDayNanoArray::from(vec![
                IntervalMonthDayNano::new(1, 2, 3_000),
            ])),
            Arc::new(LargeStringArray::from(vec!["large"])),
            Arc::new(LargeBinaryArray::from(vec![&[0_u8, 0xff][..]])),
            Arc::new(FixedSizeBinaryArray::try_from_iter([uuid.as_slice()].into_iter()).unwrap()),
        ],
    )
    .unwrap();
    let result = DuckLakeTableWriter::new(writer, object_store())
        .unwrap()
        .with_options(&DuckLakeWriteOptions::default().with_data_inlining_row_limit(1))
        .write_table("main", "scalar_values", &[batch])
        .await
        .unwrap();
    assert_eq!(result.files_written, 0);

    let provider = SqliteMetadataProvider::new(&ro_url(&temp)).await.unwrap();
    let snapshot_id = provider.get_current_snapshot().unwrap();
    let schema = provider
        .get_schema_by_name("main", snapshot_id)
        .unwrap()
        .unwrap();
    let table = provider
        .get_table_by_name(schema.schema_id, "scalar_values", snapshot_id)
        .unwrap()
        .unwrap();
    let columns = provider
        .get_table_structure(table.table_id, snapshot_id)
        .unwrap();
    let batches = provider
        .get_inlined_data(table.table_id, snapshot_id, &columns)
        .unwrap();
    let expected = vec![
        ScalarValue::Decimal128(Some(12_345), 10, 2),
        ScalarValue::Date32(Some(1)),
        ScalarValue::Time64Microsecond(Some(1_000_002)),
        ScalarValue::TimestampNanosecond(Some(1_000_002), Some("UTC".into())),
        ScalarValue::new_interval_mdn(1, 2, 3_000),
        ScalarValue::Utf8View(Some("large".to_string())),
        ScalarValue::BinaryView(Some(vec![0, 0xff])),
        ScalarValue::FixedSizeBinary(16, Some(uuid.to_vec())),
    ];
    for (index, expected) in expected.into_iter().enumerate() {
        assert_eq!(
            ScalarValue::try_from_array(batches[0].column(index), 0).unwrap(),
            expected,
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn stale_expected_base_snapshot_conflicts_and_commits_nothing() {
    let t = TempDir::new().unwrap();
    let writer = make_writer(&t).await;
    let seed = Arc::new(SqliteMetadataWriter::new(&rw_url(&t)).await.unwrap());
    DuckLakeTableWriter::new(seed, object_store())
        .unwrap()
        .write_table("main", "t", &[batch(vec![1, 2], vec![10, 20])])
        .await
        .unwrap();

    let cols = vec![
        ColumnDef::from_arrow("id", &DataType::Int32, false).unwrap(),
        ColumnDef::from_arrow("val", &DataType::Int32, false).unwrap(),
    ];
    // Pin a base, then let a concurrent append publish a newer generation.
    let stale = writer
        .begin_write_transaction("main", "t", &cols, WriteMode::Append)
        .unwrap();
    let concurrent = Arc::new(SqliteMetadataWriter::new(&rw_url(&t)).await.unwrap());
    DuckLakeTableWriter::new(concurrent, object_store())
        .unwrap()
        .append_table("main", "t", &[batch(vec![3], vec![30])])
        .await
        .unwrap();

    let pool = SqlitePool::connect(&rw_url(&t)).await.unwrap();
    let head_before: i64 = sqlx::query_scalar("SELECT MAX(snapshot_id) FROM ducklake_snapshot")
        .fetch_one(&pool)
        .await
        .unwrap();

    let error = writer
        .register_inlined_data(
            stale.table_id,
            "main",
            "t",
            stale.snapshot_id,
            &[batch(vec![9], vec![90])],
            WriteMode::Append,
            stale.base_snapshot_id,
            &cols,
            &stale.column_ids,
            &SnapshotCommitMetadata::new(),
            Some(stale.base_snapshot_id),
        )
        .unwrap_err();
    assert!(matches!(error, DuckLakeError::Conflict(_)), "{error}");

    let head_after: i64 = sqlx::query_scalar("SELECT MAX(snapshot_id) FROM ducklake_snapshot")
        .fetch_one(&pool)
        .await
        .unwrap();
    let inline_tables: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM ducklake_inlined_data_tables")
            .fetch_one(&pool)
            .await
            .unwrap();
    let record_count: i64 =
        sqlx::query_scalar("SELECT record_count FROM ducklake_table_stats WHERE table_id = ?")
            .bind(stale.table_id)
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(head_after, head_before, "conflict must commit no snapshot");
    assert_eq!(inline_tables, 0);
    assert_eq!(record_count, 3);
    assert_eq!(read_rows(&t, None).await, vec![(1, 10), (2, 20), (3, 30)]);
}

#[tokio::test(flavor = "multi_thread")]
async fn update_refuses_tables_with_inlined_rows() {
    let t = TempDir::new().unwrap();
    let writer = Arc::new(make_writer(&t).await);
    let options = DuckLakeWriteOptions::default().with_data_inlining_row_limit(10);
    DuckLakeTableWriter::new(writer, object_store())
        .unwrap()
        .with_options(&options)
        .write_table("main", "t", &[batch(vec![1, 2], vec![10, 20])])
        .await
        .unwrap();

    // Default write options: inlining stays enabled on the writable catalog.
    let writer = SqliteMetadataWriter::new(&rw_url(&t)).await.unwrap();
    let provider = SqliteMetadataProvider::new(&rw_url(&t)).await.unwrap();
    let catalog = DuckLakeCatalog::with_writer(Arc::new(provider), Arc::new(writer)).unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));
    let error = match ctx
        .sql("UPDATE ducklake.main.t SET val = 99 WHERE id = 1")
        .await
    {
        Ok(df) => df.collect().await.expect_err("UPDATE must refuse"),
        Err(e) => e,
    };
    let message = error.to_string();
    assert!(
        message.contains("UPDATE on a table with inlined rows is not supported")
            && message.contains("flush inlined data to Parquet"),
        "{message}"
    );
    // Nothing changed.
    assert_eq!(read_rows(&t, None).await, vec![(1, 10), (2, 20)]);
}

#[tokio::test(flavor = "multi_thread")]
async fn row_lineage_scan_refuses_tables_with_inlined_rows() {
    let t = TempDir::new().unwrap();
    let writer = Arc::new(make_writer(&t).await);
    let options = DuckLakeWriteOptions::default().with_data_inlining_row_limit(10);
    DuckLakeTableWriter::new(writer, object_store())
        .unwrap()
        .with_options(&options)
        .write_table("main", "t", &[batch(vec![1, 2], vec![10, 20])])
        .await
        .unwrap();

    let provider = SqliteMetadataProvider::new(&ro_url(&t)).await.unwrap();
    let catalog = DuckLakeCatalog::new(provider)
        .unwrap()
        .with_row_lineage(true);
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));
    let error = match ctx.sql("SELECT rowid, id FROM ducklake.main.t").await {
        Ok(df) => df.collect().await.expect_err("rowid scan must refuse"),
        Err(e) => e,
    };
    let message = error.to_string();
    assert!(
        message.contains("row-lineage (rowid) scan on a table with inlined rows is not supported")
            && message.contains("flush inlined data to Parquet"),
        "{message}"
    );

    // The same catalog still serves non-rowid reads, inlined rows included.
    let batches = ctx
        .sql("SELECT id, val FROM ducklake.main.t ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total, 2);
}

#[tokio::test(flavor = "multi_thread")]
async fn crate_and_duckdb_round_trip_inlined_rows() {
    let t = TempDir::new().unwrap();
    let writer = Arc::new(make_writer(&t).await);
    let options = DuckLakeWriteOptions::default().with_data_inlining_row_limit(10);
    DuckLakeTableWriter::new(writer, object_store())
        .unwrap()
        .with_options(&options)
        .write_table("main", "t", &[batch(vec![1, 2], vec![10, 20])])
        .await
        .unwrap();

    let list_field = Arc::new(Field::new("item", DataType::Int32, true));
    let nested = ListArray::new(
        Arc::clone(&list_field),
        OffsetBuffer::new(ScalarBuffer::from(vec![0_i32, 2])),
        Arc::new(Int32Array::from(vec![1, 2])),
        None,
    );
    let nested_batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "depths",
            DataType::List(list_field),
            false,
        )])),
        vec![Arc::new(nested)],
    )
    .unwrap();
    DuckLakeTableWriter::new(Arc::new(make_writer(&t).await), object_store())
        .unwrap()
        .with_options(&options)
        .write_table("main", "nested", &[nested_batch])
        .await
        .unwrap();
    let pool = SqlitePool::connect(&rw_url(&t)).await.unwrap();
    sqlx::query(
        "ALTER TABLE ducklake_snapshot
         ADD COLUMN next_catalog_id BIGINT NOT NULL DEFAULT 1000",
    )
    .execute(&pool)
    .await
    .unwrap();
    sqlx::query(
        "ALTER TABLE ducklake_snapshot
         ADD COLUMN next_file_id BIGINT NOT NULL DEFAULT 1000",
    )
    .execute(&pool)
    .await
    .unwrap();
    sqlx::query("ALTER TABLE ducklake_schema ADD COLUMN schema_uuid VARCHAR")
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query("UPDATE ducklake_schema SET schema_uuid = '00000000-0000-0000-0000-000000000001'")
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query("ALTER TABLE ducklake_table ADD COLUMN table_uuid VARCHAR")
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query("UPDATE ducklake_table SET table_uuid = '00000000-0000-0000-0000-000000000002'")
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query("ALTER TABLE ducklake_data_file ADD COLUMN file_order BIGINT")
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query("ALTER TABLE ducklake_data_file ADD COLUMN file_format VARCHAR")
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query("ALTER TABLE ducklake_delete_file ADD COLUMN format VARCHAR")
        .execute(&pool)
        .await
        .unwrap();
    for ddl in [
        "CREATE TABLE ducklake_column_mapping (mapping_id BIGINT, table_id BIGINT, type VARCHAR)",
        "CREATE TABLE ducklake_column_tag (table_id BIGINT, column_id BIGINT, begin_snapshot BIGINT, end_snapshot BIGINT, key VARCHAR, value VARCHAR)",
        "CREATE TABLE ducklake_file_variant_stats (data_file_id BIGINT, table_id BIGINT, column_id BIGINT, variant_path VARCHAR, shredded_type VARCHAR, column_size_bytes BIGINT, value_count BIGINT, null_count BIGINT, min_value VARCHAR, max_value VARCHAR, contains_nan BOOLEAN, extra_stats VARCHAR)",
        "CREATE TABLE ducklake_macro (schema_id BIGINT, macro_id BIGINT, macro_name VARCHAR, begin_snapshot BIGINT, end_snapshot BIGINT)",
        "CREATE TABLE ducklake_macro_impl (macro_id BIGINT, impl_id BIGINT, dialect VARCHAR, sql VARCHAR, type VARCHAR)",
        "CREATE TABLE ducklake_macro_parameters (macro_id BIGINT, impl_id BIGINT, column_id BIGINT, parameter_name VARCHAR, parameter_type VARCHAR, default_value VARCHAR, default_value_type VARCHAR)",
        "CREATE TABLE ducklake_name_mapping (mapping_id BIGINT, column_id BIGINT, source_name VARCHAR, target_field_id BIGINT, parent_column BIGINT, is_partition BOOLEAN)",
        "CREATE TABLE ducklake_tag (object_id BIGINT, begin_snapshot BIGINT, end_snapshot BIGINT, key VARCHAR, value VARCHAR)",
        "CREATE TABLE IF NOT EXISTS ducklake_view (view_id BIGINT, view_uuid VARCHAR, begin_snapshot BIGINT, end_snapshot BIGINT, schema_id BIGINT, view_name VARCHAR, dialect VARCHAR, sql VARCHAR, column_aliases VARCHAR)",
    ] {
        sqlx::query(ddl).execute(&pool).await.unwrap();
    }
    pool.close().await;

    let attach = format!(
        "LOAD ducklake; LOAD sqlite; ATTACH 'ducklake:sqlite:{}' AS lake;",
        t.path().join("test.db").display()
    );
    let output = Command::new("duckdb")
        .args([
            "-csv",
            "-noheader",
            ":memory:",
            "-c",
            &format!(
                "{attach} SELECT id, val FROM lake.main.t ORDER BY id; \
             SELECT depths[1], depths[2] FROM lake.main.nested;"
            ),
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "DuckDB read failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        String::from_utf8(output.stdout).unwrap(),
        "1,10\n2,20\n1,2\n"
    );

    let output = Command::new("duckdb")
        .args([
            ":memory:",
            "-c",
            &format!(
                "{attach} INSERT INTO lake.main.t VALUES (3, 30); \
                 INSERT INTO lake.main.nested VALUES ([3, 4]);"
            ),
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "DuckDB insert failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    assert_eq!(read_rows(&t, None).await, vec![(1, 10), (2, 20), (3, 30)]);

    let provider = SqliteMetadataProvider::new(&ro_url(&t)).await.unwrap();
    let snapshot_id = provider.get_current_snapshot().unwrap();
    let schema = provider
        .get_schema_by_name("main", snapshot_id)
        .unwrap()
        .unwrap();
    let table = provider
        .get_table_by_name(schema.schema_id, "nested", snapshot_id)
        .unwrap()
        .unwrap();
    let columns = provider
        .get_table_structure(table.table_id, snapshot_id)
        .unwrap();
    let values = provider
        .get_inlined_data(table.table_id, snapshot_id, &columns)
        .unwrap()
        .into_iter()
        .flat_map(|batch| {
            let lists = batch
                .column(0)
                .as_any()
                .downcast_ref::<ListArray>()
                .unwrap();
            (0..lists.len())
                .map(|row| {
                    lists
                        .value(row)
                        .as_any()
                        .downcast_ref::<Int32Array>()
                        .unwrap()
                        .values()
                        .to_vec()
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    assert_eq!(values, vec![vec![1, 2], vec![3, 4]]);
}
