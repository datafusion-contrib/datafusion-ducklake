//! Tests for PME (Parquet Modular Encryption) support
//!
//! These tests verify that datafusion-ducklake can read PME-compliant encrypted
//! Parquet files when encryption keys are stored in the DuckLake catalog.

#![cfg(all(feature = "metadata-duckdb", feature = "encryption"))]

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{Int32Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::prelude::*;
use parquet::arrow::ArrowWriter;
use parquet::encryption::encrypt::FileEncryptionProperties;
use parquet::file::properties::WriterProperties;
use tempfile::TempDir;

use datafusion_ducklake::MetadataProvider;
use datafusion_ducklake::catalog::DuckLakeCatalog;
use datafusion_ducklake::metadata_provider_duckdb::DuckdbMetadataProvider;

/// Test encryption key (128-bit / 16 bytes)
const TEST_ENCRYPTION_KEY: &[u8; 16] = b"0123456789abcdef";

/// Creates a PME-compliant encrypted Parquet file using parquet-rs
fn create_encrypted_parquet_file(path: &Path, key: &[u8]) -> anyhow::Result<u64> {
    // Create test data
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
    ]));

    let id_array = Int32Array::from(vec![1, 2, 3]);
    let name_array = StringArray::from(vec!["Alice", "Bob", "Charlie"]);

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(id_array), Arc::new(name_array)],
    )?;

    // Create encryption properties (uniform encryption - same key for footer and all columns)
    let encryption_properties = FileEncryptionProperties::builder(key.to_vec()).build()?;

    // Create writer properties with encryption
    let props = WriterProperties::builder()
        .with_file_encryption_properties(encryption_properties)
        .build();

    // Write encrypted parquet file
    let file = File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
    writer.write(&batch)?;
    writer.close()?;

    // Return file size
    let metadata = std::fs::metadata(path)?;
    Ok(metadata.len())
}

/// Creates a DuckLake catalog database with references to our encrypted file
fn create_catalog_with_encrypted_file(
    catalog_path: &Path,
    parquet_path: &Path,
    file_size: u64,
    encryption_key: &str,
) -> anyhow::Result<()> {
    let conn = duckdb::Connection::open(catalog_path)?;

    // Create DuckLake catalog tables
    conn.execute_batch(
        "
        -- Metadata table
        CREATE TABLE ducklake_metadata (
            key VARCHAR NOT NULL,
            value VARCHAR NOT NULL,
            scope VARCHAR
        );

        -- Snapshot table
        CREATE TABLE ducklake_snapshot (
            snapshot_id BIGINT PRIMARY KEY,
            snapshot_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Schema table
        CREATE TABLE ducklake_schema (
            schema_id BIGINT PRIMARY KEY,
            schema_name VARCHAR NOT NULL,
            path VARCHAR NOT NULL DEFAULT '',
            path_is_relative BOOLEAN NOT NULL DEFAULT true,
            begin_snapshot BIGINT NOT NULL,
            end_snapshot BIGINT
        );

        -- Table table
        CREATE TABLE ducklake_table (
            table_id BIGINT PRIMARY KEY,
            schema_id BIGINT NOT NULL,
            table_name VARCHAR NOT NULL,
            path VARCHAR NOT NULL DEFAULT '',
            path_is_relative BOOLEAN NOT NULL DEFAULT true,
            begin_snapshot BIGINT NOT NULL,
            end_snapshot BIGINT
        );

        -- Column table
        CREATE TABLE ducklake_column (
            column_id BIGINT PRIMARY KEY,
            table_id BIGINT NOT NULL,
            column_name VARCHAR NOT NULL,
            column_type VARCHAR NOT NULL,
            column_order INTEGER NOT NULL,
            nulls_allowed BOOLEAN DEFAULT true,
            parent_column BIGINT,
            begin_snapshot BIGINT NOT NULL,
            end_snapshot BIGINT
        );

        -- Data file table
        CREATE TABLE ducklake_data_file (
            data_file_id BIGINT PRIMARY KEY,
            table_id BIGINT NOT NULL,
            path VARCHAR NOT NULL,
            path_is_relative BOOLEAN NOT NULL DEFAULT false,
            file_size_bytes BIGINT NOT NULL,
            footer_size BIGINT,
            encryption_key VARCHAR,
            record_count BIGINT,
            row_id_start BIGINT,
            begin_snapshot BIGINT NOT NULL,
            end_snapshot BIGINT
        );

        -- Delete file table (empty for this test)
        CREATE TABLE ducklake_delete_file (
            delete_file_id BIGINT PRIMARY KEY,
            data_file_id BIGINT NOT NULL,
            table_id BIGINT NOT NULL,
            path VARCHAR NOT NULL,
            path_is_relative BOOLEAN NOT NULL DEFAULT false,
            file_size_bytes BIGINT NOT NULL,
            footer_size BIGINT,
            encryption_key VARCHAR,
            delete_count BIGINT,
            begin_snapshot BIGINT NOT NULL,
            end_snapshot BIGINT
        );
        ",
    )?;

    // Get absolute path for the parquet file
    let parquet_abs_path = parquet_path.canonicalize()?.display().to_string();
    let data_path = parquet_path
        .parent()
        .unwrap()
        .canonicalize()?
        .display()
        .to_string();

    // Insert metadata
    conn.execute(
        "INSERT INTO ducklake_metadata (key, value, scope) VALUES ('data_path', ?, NULL)",
        [&data_path],
    )?;

    // Insert snapshot
    conn.execute("INSERT INTO ducklake_snapshot (snapshot_id) VALUES (1)", [])?;

    // Insert schema (main schema)
    conn.execute(
        "INSERT INTO ducklake_schema (schema_id, schema_name, path, path_is_relative, begin_snapshot)
         VALUES (1, 'main', '', true, 1)",
        [],
    )?;

    // Insert table
    conn.execute(
        "INSERT INTO ducklake_table (table_id, schema_id, table_name, path, path_is_relative, begin_snapshot)
         VALUES (1, 1, 'encrypted_users', '', true, 1)",
        [],
    )?;

    // Insert columns
    conn.execute(
        "INSERT INTO ducklake_column (column_id, table_id, column_name, column_type, column_order, begin_snapshot)
         VALUES (1, 1, 'id', 'INTEGER', 0, 1)",
        [],
    )?;
    conn.execute(
        "INSERT INTO ducklake_column (column_id, table_id, column_name, column_type, column_order, begin_snapshot)
         VALUES (2, 1, 'name', 'VARCHAR', 1, 1)",
        [],
    )?;

    // Insert data file with encryption key (base64 encoded)
    let key_base64 = base64::Engine::encode(
        &base64::engine::general_purpose::STANDARD,
        encryption_key.as_bytes(),
    );
    conn.execute(
        "INSERT INTO ducklake_data_file (data_file_id, table_id, path, path_is_relative, file_size_bytes, encryption_key, begin_snapshot)
         VALUES (1, 1, ?, false, ?, ?, 1)",
        duckdb::params![parquet_abs_path, file_size as i64, key_base64],
    )?;

    Ok(())
}

#[tokio::test]
async fn test_read_pme_encrypted_parquet() -> anyhow::Result<()> {
    // Create temp directory for test files
    let temp_dir = TempDir::new()?;
    let parquet_path = temp_dir.path().join("encrypted_data.parquet");
    let catalog_path = temp_dir.path().join("catalog.duckdb");

    // Step 1: Create PME-compliant encrypted Parquet file
    let file_size = create_encrypted_parquet_file(&parquet_path, TEST_ENCRYPTION_KEY)?;
    println!("Created encrypted parquet file: {} bytes", file_size);

    // Step 2: Create DuckLake catalog with encryption key
    // Note: We store the raw key, the encryption module will decode it
    let key_str = std::str::from_utf8(TEST_ENCRYPTION_KEY)?;
    create_catalog_with_encrypted_file(&catalog_path, &parquet_path, file_size, key_str)?;
    println!("Created DuckLake catalog at: {}", catalog_path.display());

    // Debug: Query the catalog to see what's stored
    {
        let debug_conn = duckdb::Connection::open(&catalog_path)?;
        let mut stmt = debug_conn.prepare("SELECT path, encryption_key FROM ducklake_data_file")?;
        let rows = stmt.query_map([], |row| {
            let path: String = row.get(0)?;
            let key: Option<String> = row.get(1)?;
            Ok((path, key))
        })?;
        for row in rows {
            let (path, key) = row?;
            println!("DEBUG: Data file path={}, encryption_key={:?}", path, key);
        }
    }

    // Step 3: Create DataFusion context and register DuckLake catalog
    let ctx = SessionContext::new();

    let provider = DuckdbMetadataProvider::new(catalog_path.to_str().unwrap())?;

    // Debug: Check what files are returned by the provider
    let snapshot_id = provider.get_current_snapshot()?;
    println!("DEBUG: Current snapshot_id={}", snapshot_id);

    // Debug: Get table files directly from provider
    // First need to get table_id
    let schemas = provider.list_schemas(snapshot_id)?;
    println!("DEBUG: Schemas={:?}", schemas);
    let tables = provider.list_tables(schemas[0].schema_id, snapshot_id)?;
    println!("DEBUG: Tables={:?}", tables);
    let table_files = provider.get_table_files_for_select(tables[0].table_id, snapshot_id)?;
    println!("DEBUG: Table files count={}", table_files.len());
    for tf in &table_files {
        println!(
            "DEBUG: File path={}, encryption_key={:?}",
            tf.file.path, tf.file.encryption_key
        );
    }

    let catalog = DuckLakeCatalog::new(provider)?;
    ctx.register_catalog("ducklake", Arc::new(catalog));

    // Step 4: Query the encrypted table
    let df = ctx
        .sql("SELECT * FROM ducklake.main.encrypted_users ORDER BY id")
        .await?;
    let batches = df.collect().await?;

    // Step 5: Verify results
    assert_eq!(batches.len(), 1);
    let batch = &batches[0];
    assert_eq!(batch.num_rows(), 3);

    // Check id column
    let id_col = batch
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    assert_eq!(id_col.value(0), 1);
    assert_eq!(id_col.value(1), 2);
    assert_eq!(id_col.value(2), 3);

    // Check name column. DuckLake string columns scan as Utf8View; cast to Utf8
    // so the assertions read the values through a StringArray.
    let name_col_arr = arrow::compute::cast(batch.column(1), &DataType::Utf8).unwrap();
    let name_col = name_col_arr.as_any().downcast_ref::<StringArray>().unwrap();
    assert_eq!(name_col.value(0), "Alice");
    assert_eq!(name_col.value(1), "Bob");
    assert_eq!(name_col.value(2), "Charlie");

    println!("Successfully read encrypted Parquet file through DuckLake!");

    Ok(())
}

#[tokio::test]
async fn test_read_encrypted_parquet_without_key_fails() -> anyhow::Result<()> {
    // Create temp directory for test files
    let temp_dir = TempDir::new()?;
    let parquet_path = temp_dir.path().join("encrypted_data.parquet");
    let catalog_path = temp_dir.path().join("catalog.duckdb");

    // Create PME-compliant encrypted Parquet file
    let file_size = create_encrypted_parquet_file(&parquet_path, TEST_ENCRYPTION_KEY)?;

    // Create DuckLake catalog WITHOUT encryption key (empty string)
    create_catalog_with_encrypted_file(&catalog_path, &parquet_path, file_size, "")?;

    // Create DataFusion context and register DuckLake catalog
    let ctx = SessionContext::new();

    let provider = DuckdbMetadataProvider::new(catalog_path.to_str().unwrap())?;
    let catalog = DuckLakeCatalog::new(provider)?;
    ctx.register_catalog("ducklake", Arc::new(catalog));

    // Query should fail because file is encrypted but no key provided
    let result = ctx.sql("SELECT * FROM ducklake.main.encrypted_users").await;

    // The query planning might succeed, but execution should fail
    if let Ok(df) = result {
        let exec_result = df.collect().await;
        assert!(
            exec_result.is_err(),
            "Expected error when reading encrypted file without key"
        );
        let err_msg = exec_result.unwrap_err().to_string();
        println!("Got expected error: {}", err_msg);
        // Should indicate encryption-related failure
        assert!(
            err_msg.contains("encrypted")
                || err_msg.contains("decrypt")
                || err_msg.contains("Parquet"),
            "Error should mention encryption: {}",
            err_msg
        );
    }

    Ok(())
}

#[tokio::test]
async fn test_read_encrypted_parquet_with_wrong_key_fails() -> anyhow::Result<()> {
    // Create temp directory for test files
    let temp_dir = TempDir::new()?;
    let parquet_path = temp_dir.path().join("encrypted_data.parquet");
    let catalog_path = temp_dir.path().join("catalog.duckdb");

    // Create PME-compliant encrypted Parquet file with one key
    let file_size = create_encrypted_parquet_file(&parquet_path, TEST_ENCRYPTION_KEY)?;

    // Create DuckLake catalog with WRONG encryption key
    let wrong_key = "wrongkey12345678"; // Different 16-byte key
    create_catalog_with_encrypted_file(&catalog_path, &parquet_path, file_size, wrong_key)?;

    // Create DataFusion context and register DuckLake catalog
    let ctx = SessionContext::new();

    let provider = DuckdbMetadataProvider::new(catalog_path.to_str().unwrap())?;
    let catalog = DuckLakeCatalog::new(provider)?;
    ctx.register_catalog("ducklake", Arc::new(catalog));

    // Query should fail because key is wrong
    let result = ctx.sql("SELECT * FROM ducklake.main.encrypted_users").await;

    if let Ok(df) = result {
        let exec_result = df.collect().await;
        assert!(
            exec_result.is_err(),
            "Expected error when reading with wrong key"
        );
        println!("Got expected error: {}", exec_result.unwrap_err());
    }

    Ok(())
}
