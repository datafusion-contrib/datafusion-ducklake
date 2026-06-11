//! Integration tests for write support.
//!
//! These tests verify that data written using DuckLakeTableWriter can be read back
//! via the existing DuckLakeCatalog read path.

#![cfg(all(feature = "write-sqlite", feature = "metadata-sqlite"))]

use std::sync::Arc;

use arrow::array::{
    Array, BooleanArray, Date32Array, Float64Array, Int32Array, Int64Array, StringArray,
    TimestampMicrosecondArray, TimestampNanosecondArray,
};
use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use arrow::record_batch::RecordBatch;
use datafusion::prelude::*;
use object_store::local::LocalFileSystem;
use tempfile::TempDir;

use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeTableWriter, MetadataWriter, SqliteMetadataProvider,
    SqliteMetadataWriter, WriteMode,
};

/// Create a local filesystem object store
fn create_object_store() -> Arc<dyn object_store::ObjectStore> {
    Arc::new(LocalFileSystem::new())
}

/// Helper to create a test environment with writer and data directory
async fn create_test_env() -> (SqliteMetadataWriter, TempDir) {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("test.db");
    let data_path = temp_dir.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();

    let conn_str = format!("sqlite:{}?mode=rwc", db_path.display());
    let writer = SqliteMetadataWriter::new_with_init(&conn_str)
        .await
        .unwrap();
    writer.set_data_path(data_path.to_str().unwrap()).unwrap();

    (writer, temp_dir)
}

/// Helper to create a SessionContext with a DuckLakeCatalog
async fn create_read_context(temp_dir: &TempDir) -> SessionContext {
    let db_path = temp_dir.path().join("test.db");
    let conn_str = format!("sqlite:{}", db_path.display());

    let provider = SqliteMetadataProvider::new(&conn_str).await.unwrap();
    let catalog = DuckLakeCatalog::new(provider).unwrap();

    let ctx = SessionContext::new();
    ctx.register_catalog("test", Arc::new(catalog));
    ctx
}

#[tokio::test(flavor = "multi_thread")]
async fn test_write_and_read_basic_types() {
    let (writer, temp_dir) = create_test_env().await;
    let object_store = create_object_store();

    // Create test data with various types
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true),
        Field::new("age", DataType::Int64, true),
        Field::new("score", DataType::Float64, true),
        Field::new("active", DataType::Boolean, true),
    ]));

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(StringArray::from(vec![Some("Alice"), Some("Bob"), None])),
            Arc::new(Int64Array::from(vec![Some(25), Some(30), Some(35)])),
            Arc::new(Float64Array::from(vec![Some(95.5), None, Some(88.0)])),
            Arc::new(BooleanArray::from(vec![
                Some(true),
                Some(false),
                Some(true),
            ])),
        ],
    )
    .unwrap();

    // Write data
    let table_writer = DuckLakeTableWriter::new(Arc::new(writer), object_store).unwrap();
    let result = table_writer
        .write_table("main", "users", &[batch])
        .await
        .unwrap();

    assert_eq!(result.records_written, 3);
    assert_eq!(result.files_written, 1);
    assert!(result.snapshot_id > 0);
    assert!(result.table_id > 0);
    assert!(result.schema_id > 0);

    // Read back via DuckLakeCatalog
    let ctx = create_read_context(&temp_dir).await;
    let df = ctx
        .sql("SELECT * FROM test.main.users ORDER BY id")
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();

    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 3);
    assert_eq!(batches[0].num_columns(), 5);

    // Verify data
    let ids = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    assert_eq!(ids.values(), &[1, 2, 3]);

    let names = batches[0]
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(names.value(0), "Alice");
    assert_eq!(names.value(1), "Bob");
    assert!(names.is_null(2));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_write_temporal_types() {
    let (writer, temp_dir) = create_test_env().await;
    let object_store = create_object_store();

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("date", DataType::Date32, true),
        Field::new(
            "timestamp",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            true,
        ),
    ]));

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![1, 2])),
            Arc::new(Date32Array::from(vec![Some(19000), Some(19001)])), // Days since epoch
            Arc::new(TimestampMicrosecondArray::from(vec![
                Some(1640000000000000),
                Some(1640000001000000),
            ])),
        ],
    )
    .unwrap();

    let table_writer = DuckLakeTableWriter::new(Arc::new(writer), object_store).unwrap();
    let result = table_writer
        .write_table("main", "events", &[batch])
        .await
        .unwrap();
    assert_eq!(result.records_written, 2);

    // Read back
    let ctx = create_read_context(&temp_dir).await;
    let df = ctx
        .sql("SELECT COUNT(*) as cnt FROM test.main.events")
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let count = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!(count, 2);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_write_multiple_batches() {
    let (writer, temp_dir) = create_test_env().await;
    let object_store = create_object_store();

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("value", DataType::Utf8, true),
    ]));

    let batch1 = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(vec![1, 2])), Arc::new(StringArray::from(vec!["a", "b"]))],
    )
    .unwrap();

    let batch2 = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(vec![3, 4])), Arc::new(StringArray::from(vec!["c", "d"]))],
    )
    .unwrap();

    let table_writer = DuckLakeTableWriter::new(Arc::new(writer), object_store).unwrap();
    let result = table_writer
        .write_table("main", "data", &[batch1, batch2])
        .await
        .unwrap();
    assert_eq!(result.records_written, 4);

    // Read back
    let ctx = create_read_context(&temp_dir).await;
    let df = ctx
        .sql("SELECT COUNT(*) as cnt FROM test.main.data")
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let count = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!(count, 4);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_replace_semantics() {
    let (writer, temp_dir) = create_test_env().await;
    let object_store = create_object_store();

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("value", DataType::Int32, true),
    ]));

    // Write initial data
    let batch1 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(Int32Array::from(vec![100, 200, 300])),
        ],
    )
    .unwrap();

    let table_writer =
        DuckLakeTableWriter::new(Arc::new(writer.clone()), Arc::clone(&object_store)).unwrap();
    table_writer
        .write_table("main", "replace_test", &[batch1])
        .await
        .unwrap();

    // Write replacement data
    let batch2 = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(vec![4, 5])), Arc::new(Int32Array::from(vec![400, 500]))],
    )
    .unwrap();

    let table_writer2 =
        DuckLakeTableWriter::new(Arc::new(writer), Arc::clone(&object_store)).unwrap();
    let result = table_writer2
        .write_table("main", "replace_test", &[batch2])
        .await
        .unwrap();
    assert_eq!(result.records_written, 2);

    // Read back - should only have the replacement data
    let ctx = create_read_context(&temp_dir).await;
    let df = ctx
        .sql("SELECT id, value FROM test.main.replace_test ORDER BY id")
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();

    assert_eq!(batches[0].num_rows(), 2);

    let ids = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    assert_eq!(ids.values(), &[4, 5]);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_append_semantics() {
    let (writer, temp_dir) = create_test_env().await;
    let object_store = create_object_store();

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("value", DataType::Int32, true),
    ]));

    // Write initial data
    let batch1 = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(vec![1, 2])), Arc::new(Int32Array::from(vec![100, 200]))],
    )
    .unwrap();

    let table_writer =
        DuckLakeTableWriter::new(Arc::new(writer.clone()), Arc::clone(&object_store)).unwrap();
    table_writer
        .write_table("main", "append_test", &[batch1])
        .await
        .unwrap();

    // Append more data
    let batch2 = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(vec![3, 4])), Arc::new(Int32Array::from(vec![300, 400]))],
    )
    .unwrap();

    let table_writer2 =
        DuckLakeTableWriter::new(Arc::new(writer), Arc::clone(&object_store)).unwrap();
    let result = table_writer2
        .append_table("main", "append_test", &[batch2])
        .await
        .unwrap();
    assert_eq!(result.records_written, 2);

    // Read back - should have all data
    let ctx = create_read_context(&temp_dir).await;
    let df = ctx
        .sql("SELECT COUNT(*) as cnt FROM test.main.append_test")
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();

    let count = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!(count, 4);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_multiple_tables_same_schema() {
    let (writer, temp_dir) = create_test_env().await;
    let object_store = create_object_store();

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true),
    ]));

    let batch1 = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(vec![1])), Arc::new(StringArray::from(vec!["table1"]))],
    )
    .unwrap();

    let batch2 = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(vec![2])), Arc::new(StringArray::from(vec!["table2"]))],
    )
    .unwrap();

    let table_writer = DuckLakeTableWriter::new(Arc::new(writer), object_store).unwrap();
    table_writer
        .write_table("main", "t1", &[batch1])
        .await
        .unwrap();
    table_writer
        .write_table("main", "t2", &[batch2])
        .await
        .unwrap();

    // Read back both tables
    let ctx = create_read_context(&temp_dir).await;

    let df1 = ctx.sql("SELECT name FROM test.main.t1").await.unwrap();
    let batches1 = df1.collect().await.unwrap();
    let names1 = batches1[0]
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(names1.value(0), "table1");

    let df2 = ctx.sql("SELECT name FROM test.main.t2").await.unwrap();
    let batches2 = df2.collect().await.unwrap();
    let names2 = batches2[0]
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(names2.value(0), "table2");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_field_ids_preserved_on_roundtrip() {
    let (writer, temp_dir) = create_test_env().await;
    let object_store = create_object_store();

    let schema = Arc::new(Schema::new(vec![
        Field::new("col_a", DataType::Int32, false),
        Field::new("col_b", DataType::Utf8, true),
    ]));

    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(Int32Array::from(vec![1])), Arc::new(StringArray::from(vec!["test"]))],
    )
    .unwrap();

    let table_writer = DuckLakeTableWriter::new(Arc::new(writer), object_store).unwrap();
    table_writer
        .write_table("main", "field_id_test", &[batch])
        .await
        .unwrap();

    // Find the Parquet file and verify field_ids
    let data_path = temp_dir
        .path()
        .join("data")
        .join("main")
        .join("field_id_test");
    let parquet_files: Vec<_> = std::fs::read_dir(&data_path)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "parquet"))
        .collect();

    assert_eq!(parquet_files.len(), 1);

    // Read Parquet file and check field_ids
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    let file = std::fs::File::open(parquet_files[0].path()).unwrap();
    let reader = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
    let metadata = reader.metadata();

    let schema_descr = metadata.file_metadata().schema_descr();
    let mut field_ids = Vec::new();
    for i in 0..schema_descr.num_columns() {
        let column = schema_descr.column(i);
        let basic_info = column.self_type().get_basic_info();
        if basic_info.has_id() {
            field_ids.push(basic_info.id());
        }
    }

    // Should have field_ids for both columns
    assert_eq!(field_ids.len(), 2);
    // Field IDs should be sequential starting from 1
    assert!(field_ids.contains(&1));
    assert!(field_ids.contains(&2));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_streaming_write_api() {
    let (writer, temp_dir) = create_test_env().await;
    let object_store = create_object_store();

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("value", DataType::Utf8, true),
    ]));

    // Use streaming API
    let table_writer = DuckLakeTableWriter::new(Arc::new(writer), object_store).unwrap();
    let mut session = table_writer
        .begin_write("main", "streaming_test", &schema, WriteMode::Replace)
        .unwrap();

    // Write multiple batches incrementally
    for i in 0..3 {
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![i * 10, i * 10 + 1])),
                Arc::new(StringArray::from(vec![
                    format!("val_{}", i * 10),
                    format!("val_{}", i * 10 + 1),
                ])),
            ],
        )
        .unwrap();
        session.write_batch(&batch).unwrap();
    }

    assert_eq!(session.row_count(), 6); // 3 batches * 2 rows

    let result = session.finish().await.unwrap();
    assert_eq!(result.records_written, 6);
    assert_eq!(result.files_written, 1);

    // Read back via DuckLakeCatalog
    let ctx = create_read_context(&temp_dir).await;
    let df = ctx
        .sql("SELECT COUNT(*) as cnt FROM test.main.streaming_test")
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let count = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!(count, 6);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_streaming_write_to_custom_path() {
    let (writer, temp_dir) = create_test_env().await;
    let object_store = create_object_store();

    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));

    // Use custom path (simulating external storage manager)
    let custom_dir = temp_dir.path().join("data").join("custom").join("location");
    let custom_dir_str = custom_dir.to_str().unwrap().to_string();
    let file_name = "my_data.parquet".to_string();

    let table_writer = DuckLakeTableWriter::new(Arc::new(writer), object_store).unwrap();
    let mut session = table_writer
        .begin_write_to_path(
            "main",
            "custom_path_test",
            &schema,
            &custom_dir_str,
            file_name.clone(),
            WriteMode::Replace,
        )
        .unwrap();

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
    )
    .unwrap();
    session.write_batch(&batch).unwrap();

    let result = session.finish().await.unwrap();
    assert_eq!(result.records_written, 3);

    // Verify file exists at custom path
    assert!(custom_dir.join(&file_name).exists());

    // Read back via DuckLakeCatalog
    let ctx = create_read_context(&temp_dir).await;
    let df = ctx
        .sql("SELECT COUNT(*) as cnt FROM test.main.custom_path_test")
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let count = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!(count, 3);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_streaming_empty_write() {
    let (writer, temp_dir) = create_test_env().await;
    let object_store = create_object_store();

    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));

    let table_writer = DuckLakeTableWriter::new(Arc::new(writer), object_store).unwrap();
    let session = table_writer
        .begin_write("main", "empty_test", &schema, WriteMode::Replace)
        .unwrap();

    // Finish without writing any batches
    let result = session.finish().await.unwrap();
    assert_eq!(result.records_written, 0);
    assert_eq!(result.files_written, 1);

    // Read back - should have 0 rows
    let ctx = create_read_context(&temp_dir).await;
    let df = ctx
        .sql("SELECT COUNT(*) as cnt FROM test.main.empty_test")
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let count = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!(count, 0);
}

// Schema Evolution Tests

#[tokio::test(flavor = "multi_thread")]
async fn test_append_add_nullable_column() {
    let (writer, temp_dir) = create_test_env().await;
    let object_store = create_object_store();

    // Initial schema with 2 columns
    let schema1 = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true),
    ]));

    let batch1 = RecordBatch::try_new(
        schema1.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2])),
            Arc::new(StringArray::from(vec!["Alice", "Bob"])),
        ],
    )
    .unwrap();

    let table_writer =
        DuckLakeTableWriter::new(Arc::new(writer.clone()), Arc::clone(&object_store)).unwrap();
    table_writer
        .write_table("main", "evolve_add", &[batch1])
        .await
        .unwrap();

    // Append with new nullable column
    let schema2 = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true),
        Field::new("age", DataType::Int32, true), // New nullable column
    ]));

    let batch2 = RecordBatch::try_new(
        schema2.clone(),
        vec![
            Arc::new(Int32Array::from(vec![3, 4])),
            Arc::new(StringArray::from(vec!["Charlie", "Diana"])),
            Arc::new(Int32Array::from(vec![30, 40])),
        ],
    )
    .unwrap();

    let table_writer2 =
        DuckLakeTableWriter::new(Arc::new(writer), Arc::clone(&object_store)).unwrap();
    let result = table_writer2
        .append_table("main", "evolve_add", &[batch2])
        .await;
    assert!(result.is_ok(), "Adding nullable column should succeed");
    assert_eq!(result.unwrap().records_written, 2);

    // Read back - should have all 4 rows
    let ctx = create_read_context(&temp_dir).await;
    let df = ctx
        .sql("SELECT COUNT(*) as cnt FROM test.main.evolve_add")
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let count = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!(count, 4);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_append_remove_column() {
    let (writer, temp_dir) = create_test_env().await;
    let object_store = create_object_store();

    // Initial schema with 3 columns
    let schema1 = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true),
        Field::new("extra", DataType::Utf8, true),
    ]));

    let batch1 = RecordBatch::try_new(
        schema1.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2])),
            Arc::new(StringArray::from(vec!["Alice", "Bob"])),
            Arc::new(StringArray::from(vec!["x", "y"])),
        ],
    )
    .unwrap();

    let table_writer =
        DuckLakeTableWriter::new(Arc::new(writer.clone()), Arc::clone(&object_store)).unwrap();
    table_writer
        .write_table("main", "evolve_remove", &[batch1])
        .await
        .unwrap();

    // Append without the 'extra' column
    let schema2 = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true),
    ]));

    let batch2 = RecordBatch::try_new(
        schema2.clone(),
        vec![
            Arc::new(Int32Array::from(vec![3, 4])),
            Arc::new(StringArray::from(vec!["Charlie", "Diana"])),
        ],
    )
    .unwrap();

    let table_writer2 =
        DuckLakeTableWriter::new(Arc::new(writer), Arc::clone(&object_store)).unwrap();
    let result = table_writer2
        .append_table("main", "evolve_remove", &[batch2])
        .await;
    assert!(result.is_ok(), "Removing column should succeed");
    assert_eq!(result.unwrap().records_written, 2);

    // Read back - should have all 4 rows
    let ctx = create_read_context(&temp_dir).await;
    let df = ctx
        .sql("SELECT COUNT(*) as cnt FROM test.main.evolve_remove")
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let count = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!(count, 4);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_append_type_mismatch_fails() {
    let (writer, _temp_dir) = create_test_env().await;
    let object_store = create_object_store();

    // Initial schema
    let schema1 = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("value", DataType::Int32, true),
    ]));

    let batch1 = RecordBatch::try_new(
        schema1.clone(),
        vec![Arc::new(Int32Array::from(vec![1, 2])), Arc::new(Int32Array::from(vec![100, 200]))],
    )
    .unwrap();

    let table_writer =
        DuckLakeTableWriter::new(Arc::new(writer.clone()), Arc::clone(&object_store)).unwrap();
    table_writer
        .write_table("main", "evolve_type", &[batch1])
        .await
        .unwrap();

    // Try to append with different type for 'value'
    let schema2 = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("value", DataType::Utf8, true), // Changed from Int32 to Utf8
    ]));

    let batch2 = RecordBatch::try_new(
        schema2.clone(),
        vec![Arc::new(Int32Array::from(vec![3])), Arc::new(StringArray::from(vec!["text"]))],
    )
    .unwrap();

    let table_writer2 =
        DuckLakeTableWriter::new(Arc::new(writer), Arc::clone(&object_store)).unwrap();
    let result = table_writer2
        .append_table("main", "evolve_type", &[batch2])
        .await;
    assert!(result.is_err(), "Type mismatch should fail");
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("type") && err.contains("value"),
        "Error should mention type mismatch for 'value' column: {}",
        err
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_append_non_nullable_column_fails() {
    let (writer, _temp_dir) = create_test_env().await;
    let object_store = create_object_store();

    // Initial schema
    let schema1 = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true),
    ]));

    let batch1 = RecordBatch::try_new(
        schema1.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2])),
            Arc::new(StringArray::from(vec!["Alice", "Bob"])),
        ],
    )
    .unwrap();

    let table_writer =
        DuckLakeTableWriter::new(Arc::new(writer.clone()), Arc::clone(&object_store)).unwrap();
    table_writer
        .write_table("main", "evolve_nonnull", &[batch1])
        .await
        .unwrap();

    // Try to add a non-nullable column
    let schema2 = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true),
        Field::new("required_field", DataType::Int32, false), // New non-nullable column
    ]));

    let batch2 = RecordBatch::try_new(
        schema2.clone(),
        vec![
            Arc::new(Int32Array::from(vec![3])),
            Arc::new(StringArray::from(vec!["Charlie"])),
            Arc::new(Int32Array::from(vec![999])),
        ],
    )
    .unwrap();

    let table_writer2 =
        DuckLakeTableWriter::new(Arc::new(writer), Arc::clone(&object_store)).unwrap();
    let result = table_writer2
        .append_table("main", "evolve_nonnull", &[batch2])
        .await;
    assert!(result.is_err(), "Adding non-nullable column should fail");
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("nullable") && err.contains("required_field"),
        "Error should mention that new column must be nullable: {}",
        err
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_append_reorder_columns() {
    let (writer, temp_dir) = create_test_env().await;
    let object_store = create_object_store();

    // Initial schema: id, name, value
    let schema1 = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true),
        Field::new("value", DataType::Int32, true),
    ]));

    let batch1 = RecordBatch::try_new(
        schema1.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2])),
            Arc::new(StringArray::from(vec!["Alice", "Bob"])),
            Arc::new(Int32Array::from(vec![100, 200])),
        ],
    )
    .unwrap();

    let table_writer =
        DuckLakeTableWriter::new(Arc::new(writer.clone()), Arc::clone(&object_store)).unwrap();
    table_writer
        .write_table("main", "evolve_reorder", &[batch1])
        .await
        .unwrap();

    // Append with reordered columns: value, id, name
    let schema2 = Arc::new(Schema::new(vec![
        Field::new("value", DataType::Int32, true),
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true),
    ]));

    let batch2 = RecordBatch::try_new(
        schema2.clone(),
        vec![
            Arc::new(Int32Array::from(vec![300, 400])),
            Arc::new(Int32Array::from(vec![3, 4])),
            Arc::new(StringArray::from(vec!["Charlie", "Diana"])),
        ],
    )
    .unwrap();

    let table_writer2 =
        DuckLakeTableWriter::new(Arc::new(writer), Arc::clone(&object_store)).unwrap();
    let result = table_writer2
        .append_table("main", "evolve_reorder", &[batch2])
        .await;
    assert!(result.is_ok(), "Reordering columns should succeed");
    assert_eq!(result.unwrap().records_written, 2);

    // Read back - should have all 4 rows
    let ctx = create_read_context(&temp_dir).await;
    let df = ctx
        .sql("SELECT COUNT(*) as cnt FROM test.main.evolve_reorder")
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let count = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!(count, 4);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_zero_column_table_rejected() {
    let (writer, _temp_dir) = create_test_env().await;
    let object_store = create_object_store();

    // An empty Arrow schema (zero columns)
    let schema = Arc::new(Schema::empty());

    let batch = RecordBatch::new_empty(schema.clone());

    let table_writer = DuckLakeTableWriter::new(Arc::new(writer), object_store).unwrap();
    let result = table_writer
        .write_table("main", "empty_cols", &[batch])
        .await;
    assert!(result.is_err(), "Zero-column table should be rejected");
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("at least one column"),
        "Error should mention needing at least one column: {}",
        err
    );
}

/// Write a table larger than `object_store::buffered::BufWriter`'s 10 MiB
/// capacity so `finish()` takes the multipart-upload branch rather than a
/// single PUT, then read it back. This is the path that lifts the object
/// store's single-PUT size ceiling, so exercise it end to end.
#[tokio::test(flavor = "multi_thread")]
async fn test_streaming_write_large_file_uses_multipart() {
    let (writer, temp_dir) = create_test_env().await;
    let object_store = create_object_store();

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("payload", DataType::Int64, false),
    ]));

    // ~1.5M rows * 16 bytes ≈ 24 MB of high-entropy (uncompressed) parquet,
    // comfortably above the 10 MiB BufWriter capacity that triggers multipart.
    const ROWS: i64 = 1_500_000;
    let ids: Vec<i64> = (0..ROWS).collect();
    let payload: Vec<i64> = (0..ROWS).map(|i| i.wrapping_mul(2_654_435_761)).collect();
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int64Array::from(ids)), Arc::new(Int64Array::from(payload))],
    )
    .unwrap();

    let table_writer = DuckLakeTableWriter::new(Arc::new(writer), object_store).unwrap();
    let result = table_writer
        .write_table("main", "big", &[batch])
        .await
        .unwrap();
    assert_eq!(result.records_written, ROWS);
    assert_eq!(result.files_written, 1);

    // Read back: a complete, valid parquet means the multipart upload assembled
    // correctly. Verify both the row count and an aggregate over every row.
    let ctx = create_read_context(&temp_dir).await;
    let batches = ctx
        .sql("SELECT COUNT(*) AS n, SUM(id) AS s FROM test.main.big")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let n = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    let s = batches[0]
        .column(1)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!(n, ROWS);
    assert_eq!(s, ROWS * (ROWS - 1) / 2);
}

/// End-to-end regression for nanosecond tz-aware timestamps.
///
/// A `Timestamp(Nanosecond, Some(tz))` column (the pandas/PyArrow default for
/// tz-aware datetimes) used to be cataloged as µs `timestamptz`, so the served
/// schema disagreed with the physical parquet and the read path silently
/// truncated the sub-microsecond fraction on every scan. This writes ns values
/// with non-zero sub-µs digits and asserts they survive the full
/// write -> catalog -> read round-trip at full precision.
#[tokio::test(flavor = "multi_thread")]
async fn test_write_and_read_nanosecond_timestamptz() {
    let (writer, temp_dir) = create_test_env().await;
    let object_store = create_object_store();

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new(
            "event_ts",
            DataType::Timestamp(TimeUnit::Nanosecond, Some("UTC".into())),
            true,
        ),
    ]));

    // Values with a non-zero sub-microsecond fraction: truncation to µs would
    // zero the last three digits, so exact equality proves no precision loss.
    let ns_values = vec![1_000_000_000_123_456_789i64, 1_700_000_000_987_654_321i64];
    let ts_array = TimestampNanosecondArray::from(ns_values.clone()).with_timezone("UTC");

    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(Int32Array::from(vec![1, 2])), Arc::new(ts_array)],
    )
    .unwrap();

    let table_writer = DuckLakeTableWriter::new(Arc::new(writer), object_store).unwrap();
    table_writer
        .write_table("main", "events", &[batch])
        .await
        .unwrap();

    let ctx = create_read_context(&temp_dir).await;
    let df = ctx
        .sql("SELECT event_ts FROM test.main.events ORDER BY id")
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();

    // Served schema must keep nanosecond precision, not collapse to µs.
    assert_eq!(
        batches[0].schema().field(0).data_type(),
        &DataType::Timestamp(TimeUnit::Nanosecond, Some("UTC".into())),
        "served column must round-trip as nanosecond tz-aware"
    );

    let col = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<TimestampNanosecondArray>()
        .expect("column reads back as TimestampNanosecondArray");
    assert_eq!(
        col.values(),
        ns_values.as_slice(),
        "sub-microsecond fraction must survive the round-trip"
    );
}
