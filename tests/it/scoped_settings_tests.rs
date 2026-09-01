//! Scoped DuckLake catalog-setting integration tests.

#![cfg(all(feature = "write-sqlite", feature = "metadata-sqlite", feature = "metadata-duckdb"))]

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow::array::Int64Array;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::prelude::SessionContext;
use datafusion_ducklake::metadata_provider::MetadataProvider;
use datafusion_ducklake::{
    ColumnDef, DuckLakeCatalog, DuckdbMetadataProvider, MetadataWriter, SqliteMetadataProvider,
    SqliteMetadataWriter, WriteMode,
};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::basic::Compression;
use rstest::rstest;
use sqlx::SqlitePool;
use tempfile::TempDir;

fn assert_parquet_v1_zstd(path: &Path) {
    let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(path).unwrap()).unwrap();
    let metadata = reader.metadata();
    assert_eq!(
        metadata.file_metadata().version(),
        1,
        "unexpected Parquet version for {}",
        path.display()
    );
    let compression = metadata.row_groups()[0].columns()[0].compression();
    assert!(
        matches!(compression, Compression::ZSTD(_)),
        "unexpected compression {compression:?} for {}",
        path.display()
    );
}

async fn writable_context(connection: &str) -> SessionContext {
    let provider = SqliteMetadataProvider::new(connection).await.unwrap();
    let writer = SqliteMetadataWriter::new(connection).await.unwrap();
    let catalog = DuckLakeCatalog::with_writer(Arc::new(provider), Arc::new(writer)).unwrap();
    let context = SessionContext::new();
    context.register_catalog("lake", Arc::new(catalog));
    context
}

#[rstest]
#[tokio::test(flavor = "multi_thread")]
async fn writable_open_adds_scope_id_once_and_preserves_global_settings() {
    let temp = TempDir::new().unwrap();
    let database = temp.path().join("legacy.db");
    let connection = format!("sqlite:{}?mode=rwc", database.display());
    let pool = SqlitePool::connect(&connection).await.unwrap();
    sqlx::query("CREATE TABLE ducklake_metadata (key VARCHAR NOT NULL, value VARCHAR NOT NULL)")
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query(
        "INSERT INTO ducklake_metadata (key, value) VALUES ('data_path', '/preserved/path')",
    )
    .execute(&pool)
    .await
    .unwrap();
    pool.close().await;

    let provider = SqliteMetadataProvider::new(&connection).await.unwrap();
    assert_eq!(
        provider
            .get_metadata_settings(None, None)
            .unwrap()
            .get("data_path")
            .map(String::as_str),
        Some("/preserved/path")
    );

    let writer = SqliteMetadataWriter::new_with_init(&connection)
        .await
        .unwrap();
    writer
        .set_global_setting("parquet_compression", "zstd")
        .unwrap();
    SqliteMetadataWriter::new_with_init(&connection)
        .await
        .unwrap();

    let pool = SqlitePool::connect(&connection).await.unwrap();
    let scope_columns: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM pragma_table_info('ducklake_metadata') \
         WHERE name IN ('scope', 'scope_id')",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    let data_path: String = sqlx::query_scalar(
        "SELECT value FROM ducklake_metadata WHERE key = 'data_path' AND scope IS NULL",
    )
    .fetch_one(&pool)
    .await
    .unwrap();

    assert_eq!(scope_columns, 2);
    assert_eq!(data_path, "/preserved/path");
    let provider = SqliteMetadataProvider::new(&connection).await.unwrap();
    assert_eq!(
        provider
            .get_metadata_settings(None, None)
            .unwrap()
            .get("parquet_compression")
            .map(String::as_str),
        Some("zstd")
    );
}

#[rstest]
#[tokio::test(flavor = "multi_thread")]
async fn table_scoped_parquet_options_control_insert_update_and_delete_files() {
    let temp = TempDir::new().unwrap();
    let database = temp.path().join("catalog.db");
    let data = temp.path().join("data");
    std::fs::create_dir(&data).unwrap();
    let connection = format!("sqlite:{}?mode=rwc", database.display());

    let writer = SqliteMetadataWriter::new_with_init(&connection)
        .await
        .unwrap();
    writer.set_data_path(data.to_str().unwrap()).unwrap();
    let columns = vec![ColumnDef::new("id", "int64", false).unwrap()];
    let setup = writer
        .begin_write_transaction("main", "events", &columns, WriteMode::Replace)
        .unwrap();
    writer
        .publish_snapshot(
            setup.table_id,
            "main",
            "events",
            setup.snapshot_id,
            WriteMode::Replace,
            setup.base_snapshot_id,
            &columns,
            &setup.column_ids,
        )
        .unwrap();

    let pool = SqlitePool::connect(&connection).await.unwrap();
    sqlx::query(
        "INSERT INTO ducklake_metadata (key, value, scope, scope_id) \
         VALUES ('parquet_compression', 'zstd', 'table', ?), \
                ('parquet_version', 'V1', 'table', ?), \
                ('data_inlining_row_limit', '0', 'table', ?)",
    )
    .bind(setup.table_id)
    .bind(setup.table_id)
    .bind(setup.table_id)
    .execute(&pool)
    .await
    .unwrap();
    pool.close().await;

    let provider = SqliteMetadataProvider::new(&connection).await.unwrap();
    let catalog = DuckLakeCatalog::with_writer(Arc::new(provider), Arc::new(writer)).unwrap();
    let context = SessionContext::new();
    context.register_catalog("lake", Arc::new(catalog));
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
    context
        .register_batch(
            "source",
            RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(vec![1, 2, 3]))]).unwrap(),
        )
        .unwrap();
    context
        .sql("INSERT INTO lake.main.events SELECT * FROM source")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    let provider = SqliteMetadataProvider::new(&connection).await.unwrap();
    let snapshot = provider.get_current_snapshot().unwrap();
    let files = provider
        .get_table_files_for_select(setup.table_id, snapshot)
        .unwrap();
    assert_eq!(files.len(), 1);
    let table_path = data.join("main/events");
    assert_parquet_v1_zstd(&table_path.join(&files[0].file.path));

    let context = writable_context(&connection).await;
    context
        .sql("UPDATE lake.main.events SET id = id + 10 WHERE id = 1")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let provider = SqliteMetadataProvider::new(&connection).await.unwrap();
    let snapshot = provider.get_current_snapshot().unwrap();
    let files = provider
        .get_table_files_for_select(setup.table_id, snapshot)
        .unwrap();
    assert_eq!(files.len(), 2);
    for file in &files {
        assert_parquet_v1_zstd(&table_path.join(&file.file.path));
        if let Some(delete_file) = &file.delete_file {
            assert_parquet_v1_zstd(&table_path.join(&delete_file.path));
        }
    }
    assert_eq!(
        files
            .iter()
            .filter(|file| file.delete_file.is_some())
            .count(),
        1
    );

    let context = writable_context(&connection).await;
    context
        .sql("DELETE FROM lake.main.events WHERE id = 2")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let provider = SqliteMetadataProvider::new(&connection).await.unwrap();
    let snapshot = provider.get_current_snapshot().unwrap();
    let files = provider
        .get_table_files_for_select(setup.table_id, snapshot)
        .unwrap();
    let delete_files: Vec<_> = files
        .iter()
        .filter_map(|file| file.delete_file.as_ref())
        .collect();
    assert_eq!(delete_files.len(), 1);
    assert_parquet_v1_zstd(&table_path.join(&delete_files[0].path));
}

#[rstest]
#[tokio::test(flavor = "multi_thread")]
async fn invalid_write_setting_does_not_block_reads() {
    let temp = TempDir::new().unwrap();
    let database = temp.path().join("catalog.db");
    let data = temp.path().join("data");
    std::fs::create_dir(&data).unwrap();
    let connection = format!("sqlite:{}?mode=rwc", database.display());
    let writer = SqliteMetadataWriter::new_with_init(&connection)
        .await
        .unwrap();
    writer.set_data_path(data.to_str().unwrap()).unwrap();
    let columns = vec![ColumnDef::new("id", "int64", false).unwrap()];
    let setup = writer
        .begin_write_transaction("main", "events", &columns, WriteMode::Replace)
        .unwrap();
    writer
        .publish_snapshot(
            setup.table_id,
            "main",
            "events",
            setup.snapshot_id,
            WriteMode::Replace,
            setup.base_snapshot_id,
            &columns,
            &setup.column_ids,
        )
        .unwrap();
    let pool = SqlitePool::connect(&connection).await.unwrap();
    sqlx::query(
        "INSERT INTO ducklake_metadata (key, value, scope, scope_id) \
         VALUES ('parquet_compression', 'zstd', 'table', ?), \
                ('parquet_compression_level', '23', 'table', ?)",
    )
    .bind(setup.table_id)
    .bind(setup.table_id)
    .execute(&pool)
    .await
    .unwrap();
    pool.close().await;

    let provider = SqliteMetadataProvider::new(&connection).await.unwrap();
    let catalog = DuckLakeCatalog::with_writer(Arc::new(provider), Arc::new(writer)).unwrap();
    let context = SessionContext::new();
    context.register_catalog("lake", Arc::new(catalog));
    let batches = context
        .sql("SELECT count(*) FROM lake.main.events")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(
        batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0),
        0
    );

    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
    context
        .register_batch(
            "source",
            RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(vec![1]))]).unwrap(),
        )
        .unwrap();
    let error = match context
        .sql("INSERT INTO lake.main.events SELECT * FROM source")
        .await
    {
        Ok(frame) => frame.collect().await.unwrap_err().to_string(),
        Err(e) => e.to_string(),
    };
    assert!(error.contains("Invalid DuckLake write settings"), "{error}");
}

#[tokio::test(flavor = "multi_thread")]
async fn table_scoped_row_limit_controls_sql_inlining() {
    let temp = TempDir::new().unwrap();
    let database = temp.path().join("catalog.db");
    let data = temp.path().join("data");
    std::fs::create_dir(&data).unwrap();
    let connection = format!("sqlite:{}?mode=rwc", database.display());
    let writer = SqliteMetadataWriter::new_with_init(&connection)
        .await
        .unwrap();
    writer.set_data_path(data.to_str().unwrap()).unwrap();
    let columns = vec![ColumnDef::new("id", "int64", false).unwrap()];
    let setup = writer
        .begin_write_transaction("main", "events", &columns, WriteMode::Replace)
        .unwrap();
    writer
        .publish_snapshot(
            setup.table_id,
            "main",
            "events",
            setup.snapshot_id,
            WriteMode::Replace,
            setup.base_snapshot_id,
            &columns,
            &setup.column_ids,
        )
        .unwrap();
    let pool = SqlitePool::connect(&connection).await.unwrap();
    sqlx::query(
        "INSERT INTO ducklake_metadata (key, value, scope, scope_id)
         VALUES ('data_inlining_row_limit', '2', 'table', ?)",
    )
    .bind(setup.table_id)
    .execute(&pool)
    .await
    .unwrap();
    pool.close().await;

    let provider = SqliteMetadataProvider::new(&connection).await.unwrap();
    let catalog = DuckLakeCatalog::with_writer(Arc::new(provider), Arc::new(writer)).unwrap();
    let context = SessionContext::new();
    context.register_catalog("lake", Arc::new(catalog));
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
    context
        .register_batch(
            "source",
            RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(vec![1, 2]))]).unwrap(),
        )
        .unwrap();
    context
        .sql("INSERT INTO lake.main.events SELECT * FROM source")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    let pool = SqlitePool::connect(&connection).await.unwrap();
    let files: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM ducklake_data_file WHERE table_id = ?")
            .bind(setup.table_id)
            .fetch_one(&pool)
            .await
            .unwrap();
    let physical_name: String = sqlx::query_scalar(
        "SELECT table_name FROM ducklake_inlined_data_tables WHERE table_id = ?",
    )
    .bind(setup.table_id)
    .fetch_one(&pool)
    .await
    .unwrap();
    let rows: i64 = sqlx::query_scalar(sqlx::AssertSqlSafe(format!(
        "SELECT COUNT(*) FROM {physical_name} WHERE end_snapshot IS NULL"
    )))
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(files, 0);
    assert_eq!(rows, 2);
}

#[test]
fn official_duckdb_settings_resolve_with_table_precedence() {
    let temp = TempDir::new().unwrap();
    let catalog_path = temp.path().join("official.ducklake");
    let data_path = temp.path().join("data");
    let connection = duckdb::Connection::open_in_memory().unwrap();
    connection.execute("INSTALL ducklake", []).unwrap();
    connection.execute("LOAD ducklake", []).unwrap();
    connection
        .execute(
            &format!(
                "ATTACH 'ducklake:{}' AS lake (DATA_PATH '{}')",
                catalog_path.display(),
                data_path.display()
            ),
            [],
        )
        .unwrap();
    connection
        .execute("CREATE TABLE lake.events (id BIGINT)", [])
        .unwrap();
    connection
        .execute(
            "CALL lake.set_option('parquet_compression', 'uncompressed')",
            [],
        )
        .unwrap();
    connection
        .execute(
            "CALL lake.set_option('parquet_compression', 'lz4', schema => 'main')",
            [],
        )
        .unwrap();
    connection
        .execute(
            "CALL lake.set_option('parquet_compression', 'zstd', table_name => 'events')",
            [],
        )
        .unwrap();
    connection.execute("DETACH lake", []).unwrap();
    drop(connection);

    let provider = DuckdbMetadataProvider::new(catalog_path.to_str().unwrap()).unwrap();
    let snapshot = provider.get_current_snapshot().unwrap();
    let schema = provider
        .get_schema_by_name("main", snapshot)
        .unwrap()
        .unwrap();
    let table = provider
        .get_table_by_name(schema.schema_id, "events", snapshot)
        .unwrap()
        .unwrap();
    let settings = provider
        .get_metadata_settings(Some(schema.schema_id), Some(table.table_id))
        .unwrap();

    assert_eq!(
        settings.get("parquet_compression").map(String::as_str),
        Some("zstd")
    );
}
