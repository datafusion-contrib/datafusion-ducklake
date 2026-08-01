//! NOT NULL enforcement across SQL and low-level DuckLake writes.

#![cfg(all(feature = "write-sqlite", feature = "metadata-sqlite"))]

use std::path::Path;
use std::sync::Arc;

use arrow::array::{ArrayRef, Int32Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::prelude::SessionContext;
use object_store::local::LocalFileSystem;
use sqlx::sqlite::SqlitePool;
use tempfile::TempDir;

use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeTableWriter, MetadataWriter, SqliteMetadataProvider,
    SqliteMetadataWriter,
};

fn table_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("required", DataType::Int32, false),
        Field::new("optional", DataType::Int32, true),
    ]))
}

fn object_store() -> Arc<dyn object_store::ObjectStore> {
    Arc::new(LocalFileSystem::new())
}

async fn make_writer(temp_dir: &TempDir) -> SqliteMetadataWriter {
    let db_path = temp_dir.path().join("test.db");
    let data_path = temp_dir.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();
    let conn_str = format!("sqlite:{}?mode=rwc", db_path.display());
    let writer = SqliteMetadataWriter::new_with_init(&conn_str)
        .await
        .unwrap();
    writer.set_data_path(data_path.to_str().unwrap()).unwrap();
    writer
}

async fn seed_table(temp_dir: &TempDir) {
    let writer = Arc::new(make_writer(temp_dir).await);
    let batch = RecordBatch::try_new(
        table_schema(),
        vec![
            Arc::new(Int32Array::from(vec![1])),
            Arc::new(Int32Array::from(vec![10])),
            Arc::new(Int32Array::from(vec![Some(100)])),
        ],
    )
    .unwrap();
    DuckLakeTableWriter::new(writer, object_store())
        .unwrap()
        .write_table("main", "t", &[batch])
        .await
        .unwrap();
}

async fn writable_context(temp_dir: &TempDir) -> SessionContext {
    let conn_str = format!(
        "sqlite:{}?mode=rwc",
        temp_dir.path().join("test.db").display()
    );
    let writer = SqliteMetadataWriter::new(&conn_str).await.unwrap();
    let provider = SqliteMetadataProvider::new(&conn_str).await.unwrap();
    let catalog = DuckLakeCatalog::with_writer(Arc::new(provider), Arc::new(writer)).unwrap();
    let context = SessionContext::new();
    context.register_catalog("ducklake", Arc::new(catalog));
    context
}

async fn execute_error(context: &SessionContext, sql: &str) -> String {
    match context.sql(sql).await {
        Ok(frame) => frame.collect().await.unwrap_err().to_string(),
        Err(e) => e.to_string(),
    }
}

async fn catalog_state(temp_dir: &TempDir) -> (i64, i64, i64, i64) {
    let conn_str = format!("sqlite:{}", temp_dir.path().join("test.db").display());
    let pool = SqlitePool::connect(&conn_str).await.unwrap();
    let head = sqlx::query_scalar("SELECT COALESCE(MAX(snapshot_id), 0) FROM ducklake_snapshot")
        .fetch_one(&pool)
        .await
        .unwrap();
    let tables = sqlx::query_scalar("SELECT COUNT(*) FROM ducklake_table")
        .fetch_one(&pool)
        .await
        .unwrap();
    let data_files = sqlx::query_scalar("SELECT COUNT(*) FROM ducklake_data_file")
        .fetch_one(&pool)
        .await
        .unwrap();
    let delete_files = sqlx::query_scalar("SELECT COUNT(*) FROM ducklake_delete_file")
        .fetch_one(&pool)
        .await
        .unwrap();
    (head, tables, data_files, delete_files)
}

fn file_count(path: &Path) -> usize {
    if !path.exists() {
        return 0;
    }
    std::fs::read_dir(path)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .map(|entry| {
            if entry.is_dir() {
                file_count(&entry)
            } else {
                1
            }
        })
        .sum()
}

#[tokio::test(flavor = "multi_thread")]
async fn sql_insert_rejects_not_null_and_accepts_nullable_column() {
    let temp_dir = TempDir::new().unwrap();
    seed_table(&temp_dir).await;
    let context = writable_context(&temp_dir).await;
    let before = catalog_state(&temp_dir).await;
    let files_before = file_count(&temp_dir.path().join("data"));
    let input_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, true),
        Field::new("required", DataType::Int32, true),
        Field::new("optional", DataType::Int32, true),
    ]));
    let input = RecordBatch::try_new(
        input_schema,
        vec![
            Arc::new(Int32Array::from(vec![Some(2)])),
            Arc::new(Int32Array::from(vec![None])),
            Arc::new(Int32Array::from(vec![Some(200)])),
        ],
    )
    .unwrap();
    context.register_batch("invalid_input", input).unwrap();

    let error = execute_error(
        &context,
        "INSERT INTO ducklake.main.t SELECT id, required, optional FROM invalid_input",
    )
    .await;

    assert!(
        error.contains("NOT NULL constraint failed: required"),
        "unexpected error: {error}"
    );
    assert_eq!(catalog_state(&temp_dir).await, before);
    assert_eq!(file_count(&temp_dir.path().join("data")), files_before);

    let result = context
        .sql("INSERT INTO ducklake.main.t VALUES (2, 20, CAST(NULL AS INT))")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    assert_eq!(result.len(), 1);
    assert_eq!(catalog_state(&temp_dir).await.0, before.0 + 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn sql_update_rejects_not_null_before_delete_file_write() {
    let temp_dir = TempDir::new().unwrap();
    seed_table(&temp_dir).await;
    let context = writable_context(&temp_dir).await;
    let before = catalog_state(&temp_dir).await;
    let files_before = file_count(&temp_dir.path().join("data"));

    let error = execute_error(
        &context,
        "UPDATE ducklake.main.t SET required = CAST(NULL AS INT) WHERE id = 1",
    )
    .await;

    assert!(
        error.contains("NOT NULL constraint failed: required"),
        "unexpected error: {error}"
    );
    assert_eq!(catalog_state(&temp_dir).await, before);
    assert_eq!(file_count(&temp_dir.path().join("data")), files_before);
}

#[tokio::test(flavor = "multi_thread")]
async fn low_level_write_uses_catalog_not_incoming_nullability() {
    let temp_dir = TempDir::new().unwrap();
    seed_table(&temp_dir).await;
    let conn_str = format!(
        "sqlite:{}?mode=rwc",
        temp_dir.path().join("test.db").display()
    );
    let writer = Arc::new(SqliteMetadataWriter::new(&conn_str).await.unwrap());
    let before = catalog_state(&temp_dir).await;
    let files_before = file_count(&temp_dir.path().join("data"));
    let incoming_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, true),
        Field::new("required", DataType::Int32, true),
        Field::new("optional", DataType::Int32, true),
    ]));
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int32Array::from(vec![Some(2)])),
        Arc::new(Int32Array::from(vec![None])),
        Arc::new(Int32Array::from(vec![Some(200)])),
    ];
    let batch = RecordBatch::try_new(incoming_schema, columns).unwrap();

    let error = DuckLakeTableWriter::new(writer, object_store())
        .unwrap()
        .append_table("main", "t", &[batch])
        .await
        .unwrap_err();

    assert_eq!(
        error.to_string(),
        "Invalid configuration: NOT NULL constraint failed: required"
    );
    assert_eq!(catalog_state(&temp_dir).await, before);
    assert_eq!(file_count(&temp_dir.path().join("data")), files_before);
}
