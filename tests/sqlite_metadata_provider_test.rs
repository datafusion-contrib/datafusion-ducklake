#![cfg(feature = "metadata-sqlite")]
//! SQLite metadata provider tests
//!
//! This test suite verifies the SQLite metadata provider implementation,
//! including all MetadataProvider trait methods, schema initialization,
//! concurrent access, and error handling.
//!
//! ## Test Setup
//!
//! Tests use in-memory SQLite databases for fast, isolated testing.
//! No Docker or external services required.
//!
//! ## Coverage
//!
//! - Schema initialization (idempotent)
//! - All MetadataProvider trait methods
//! - Snapshot isolation and temporal queries
//! - Concurrent access and thread safety
//! - Error handling and edge cases

mod common;

use datafusion::prelude::*;
use datafusion_ducklake::{
    DuckLakeCatalog, DuckdbMetadataProvider, SqliteMetadataProvider,
    metadata_provider::MetadataProvider,
};
use sqlx::SqlitePool;
use std::sync::Arc;
use tempfile::TempDir;

/// Initialize DuckLake catalog schema in SQLite (for tests only)
async fn init_schema(pool: &SqlitePool) -> anyhow::Result<()> {
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS ducklake_snapshot (
            snapshot_id INTEGER PRIMARY KEY,
            snapshot_time TEXT
        )",
    )
    .execute(pool)
    .await?;

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS ducklake_schema (
            schema_id INTEGER PRIMARY KEY,
            schema_name TEXT NOT NULL,
            path TEXT NOT NULL,
            path_is_relative INTEGER NOT NULL,
            begin_snapshot INTEGER NOT NULL,
            end_snapshot INTEGER
        )",
    )
    .execute(pool)
    .await?;

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS ducklake_table (
            table_id INTEGER PRIMARY KEY,
            schema_id INTEGER NOT NULL,
            table_name TEXT NOT NULL,
            path TEXT NOT NULL,
            path_is_relative INTEGER NOT NULL,
            begin_snapshot INTEGER NOT NULL,
            end_snapshot INTEGER,
            FOREIGN KEY (schema_id) REFERENCES ducklake_schema(schema_id)
        )",
    )
    .execute(pool)
    .await?;

    // Schema must match SQL_CREATE_SCHEMA in metadata_writer_sqlite.rs.
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS ducklake_column (
            column_id INTEGER PRIMARY KEY,
            table_id INTEGER NOT NULL,
            column_name TEXT NOT NULL,
            column_type TEXT NOT NULL,
            column_order INTEGER NOT NULL,
            nulls_allowed INTEGER,
            initial_default TEXT,
            default_value TEXT,
            parent_column INTEGER,
            default_value_type TEXT,
            default_value_dialect TEXT,
            begin_snapshot INTEGER NOT NULL DEFAULT 1,
            end_snapshot INTEGER,
            FOREIGN KEY (table_id) REFERENCES ducklake_table(table_id)
        )",
    )
    .execute(pool)
    .await?;

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS ducklake_data_file (
            data_file_id INTEGER PRIMARY KEY,
            table_id INTEGER NOT NULL,
            path TEXT NOT NULL,
            path_is_relative INTEGER NOT NULL,
            file_size_bytes INTEGER NOT NULL,
            footer_size INTEGER,
            encryption_key TEXT,
            record_count INTEGER,
            row_id_start INTEGER,
            mapping_id INTEGER,
            begin_snapshot INTEGER NOT NULL DEFAULT 1,
            end_snapshot INTEGER,
            FOREIGN KEY (table_id) REFERENCES ducklake_table(table_id)
        )",
    )
    .execute(pool)
    .await?;

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS ducklake_delete_file (
            delete_file_id INTEGER PRIMARY KEY,
            data_file_id INTEGER NOT NULL,
            table_id INTEGER NOT NULL,
            path TEXT NOT NULL,
            path_is_relative INTEGER NOT NULL,
            file_size_bytes INTEGER NOT NULL,
            footer_size INTEGER,
            encryption_key TEXT,
            delete_count INTEGER,
            begin_snapshot INTEGER NOT NULL,
            end_snapshot INTEGER,
            FOREIGN KEY (data_file_id) REFERENCES ducklake_data_file(data_file_id),
            FOREIGN KEY (table_id) REFERENCES ducklake_table(table_id)
        )",
    )
    .execute(pool)
    .await?;

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS ducklake_metadata (
            key TEXT NOT NULL PRIMARY KEY,
            value TEXT NOT NULL,
            scope TEXT,
            scope_id INTEGER
        )",
    )
    .execute(pool)
    .await?;

    // SQLite indexes
    sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_schema_snapshot ON ducklake_schema(begin_snapshot, end_snapshot)",
    )
    .execute(pool)
    .await?;

    sqlx::query("CREATE INDEX IF NOT EXISTS idx_table_schema ON ducklake_table(schema_id)")
        .execute(pool)
        .await?;

    sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_table_snapshot ON ducklake_table(begin_snapshot, end_snapshot)",
    )
    .execute(pool)
    .await?;

    Ok(())
}

/// Helper to create a SQLite provider with initialized schema (in-memory)
async fn create_sqlite_provider() -> anyhow::Result<SqliteMetadataProvider> {
    // Use a unique in-memory database for each test
    let provider = SqliteMetadataProvider::new("sqlite::memory:")
        .await
        .expect("Failed to create provider");
    init_schema(&provider.pool).await?;

    Ok(provider)
}

/// Helper to populate test data in SQLite
async fn populate_test_data(provider: &SqliteMetadataProvider) -> anyhow::Result<()> {
    let pool = &provider.pool;

    // Insert snapshots
    sqlx::query(
        "INSERT INTO ducklake_snapshot (snapshot_id, snapshot_time) VALUES (?, datetime('now'))",
    )
    .bind(1i64)
    .execute(pool)
    .await?;

    sqlx::query(
        "INSERT INTO ducklake_snapshot (snapshot_id, snapshot_time) VALUES (?, datetime('now'))",
    )
    .bind(2i64)
    .execute(pool)
    .await?;

    // Insert metadata (data_path)
    sqlx::query(
        "INSERT INTO ducklake_metadata (key, value, scope, scope_id) VALUES (?, ?, NULL, NULL)",
    )
    .bind("data_path")
    .bind("file:///tmp/ducklake_data/")
    .execute(pool)
    .await?;

    // Insert schema
    sqlx::query(
        "INSERT INTO ducklake_schema (schema_id, schema_name, path, path_is_relative, begin_snapshot, end_snapshot)
         VALUES (?, ?, ?, ?, ?, ?)"
    )
    .bind(1i64)
    .bind("test_schema")
    .bind("test_schema/")
    .bind(1i32)
    .bind(1i64)
    .bind(None::<i64>)
    .execute(pool)
    .await?;

    // Insert another schema (only in snapshot 2)
    sqlx::query(
        "INSERT INTO ducklake_schema (schema_id, schema_name, path, path_is_relative, begin_snapshot, end_snapshot)
         VALUES (?, ?, ?, ?, ?, ?)"
    )
    .bind(2i64)
    .bind("schema2")
    .bind("schema2/")
    .bind(1i32)
    .bind(2i64)
    .bind(None::<i64>)
    .execute(pool)
    .await?;

    // Insert table
    sqlx::query(
        "INSERT INTO ducklake_table (table_id, schema_id, table_name, path, path_is_relative, begin_snapshot, end_snapshot)
         VALUES (?, ?, ?, ?, ?, ?, ?)"
    )
    .bind(1i64)
    .bind(1i64)
    .bind("users")
    .bind("users/")
    .bind(1i32)
    .bind(1i64)
    .bind(None::<i64>)
    .execute(pool)
    .await?;

    // Insert another table (only in snapshot 2)
    sqlx::query(
        "INSERT INTO ducklake_table (table_id, schema_id, table_name, path, path_is_relative, begin_snapshot, end_snapshot)
         VALUES (?, ?, ?, ?, ?, ?, ?)"
    )
    .bind(2i64)
    .bind(1i64)
    .bind("products")
    .bind("products/")
    .bind(1i32)
    .bind(2i64)
    .bind(None::<i64>)
    .execute(pool)
    .await?;

    // Insert columns for users table
    sqlx::query(
        "INSERT INTO ducklake_column (column_id, table_id, column_name, column_type, column_order, nulls_allowed)
         VALUES (?, ?, ?, ?, ?, ?)",
    )
    .bind(1i64)
    .bind(1i64)
    .bind("id")
    .bind("INT")
    .bind(0i32)
    .bind(0i32) // false
    .execute(pool)
    .await?;

    sqlx::query(
        "INSERT INTO ducklake_column (column_id, table_id, column_name, column_type, column_order, nulls_allowed)
         VALUES (?, ?, ?, ?, ?, ?)",
    )
    .bind(2i64)
    .bind(1i64)
    .bind("name")
    .bind("VARCHAR")
    .bind(1i32)
    .bind(1i32) // true
    .execute(pool)
    .await?;

    sqlx::query(
        "INSERT INTO ducklake_column (column_id, table_id, column_name, column_type, column_order, nulls_allowed)
         VALUES (?, ?, ?, ?, ?, ?)",
    )
    .bind(3i64)
    .bind(1i64)
    .bind("email")
    .bind("VARCHAR")
    .bind(2i32)
    .bind(1i32) // true
    .execute(pool)
    .await?;

    // Insert data file
    sqlx::query(
        "INSERT INTO ducklake_data_file (data_file_id, table_id, path, path_is_relative, file_size_bytes, footer_size, begin_snapshot)
         VALUES (?, ?, ?, ?, ?, ?, ?)"
    )
    .bind(1i64)
    .bind(1i64)
    .bind("data_001.parquet")
    .bind(1i32)
    .bind(1024i64)
    .bind(Some(128i64))
    .bind(1i64)
    .execute(pool)
    .await?;

    sqlx::query(
        "INSERT INTO ducklake_data_file (data_file_id, table_id, path, path_is_relative, file_size_bytes, footer_size, begin_snapshot)
         VALUES (?, ?, ?, ?, ?, ?, ?)"
    )
    .bind(2i64)
    .bind(1i64)
    .bind("data_002.parquet")
    .bind(1i32)
    .bind(2048i64)
    .bind(Some(256i64))
    .bind(1i64)
    .execute(pool)
    .await?;

    // Insert delete file for first data file
    sqlx::query(
        "INSERT INTO ducklake_delete_file (delete_file_id, data_file_id, table_id, path, path_is_relative,
                                           file_size_bytes, footer_size, delete_count, begin_snapshot, end_snapshot)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    )
    .bind(1i64)
    .bind(1i64)
    .bind(1i64)
    .bind("data_001.delete.parquet")
    .bind(1i32)
    .bind(512i64)
    .bind(Some(64i64))
    .bind(Some(5i64))
    .bind(1i64)
    .bind(None::<i64>)
    .execute(pool)
    .await?;

    Ok(())
}

/// Helper to populate SQLite with metadata from a DuckDB-created catalog
async fn populate_from_duckdb_catalog(
    provider: &SqliteMetadataProvider,
) -> anyhow::Result<(String, TempDir)> {
    // Step 1: Create temporary directory and DuckDB catalog with real Parquet files
    let temp_dir = TempDir::new()?;
    let catalog_path = temp_dir.path().join("source.ducklake");
    common::create_catalog_no_deletes(&catalog_path)?;

    // Step 2: Read metadata from DuckDB catalog
    let duckdb_provider = DuckdbMetadataProvider::new(catalog_path.to_string_lossy().to_string())?;

    let data_path = duckdb_provider.get_data_path()?;
    let snapshots = duckdb_provider.list_snapshots()?;
    let current_snapshot = snapshots
        .last()
        .ok_or_else(|| anyhow::anyhow!("No snapshots found"))?;

    let schemas = duckdb_provider.list_schemas(current_snapshot.snapshot_id)?;

    // Step 3: Populate SQLite with metadata from DuckDB
    let pool = &provider.pool;

    // Insert snapshots
    for snapshot in &snapshots {
        sqlx::query("INSERT INTO ducklake_snapshot (snapshot_id, snapshot_time) VALUES (?, ?)")
            .bind(snapshot.snapshot_id)
            .bind(&snapshot.timestamp)
            .execute(pool)
            .await?;
    }

    // Insert data_path metadata
    sqlx::query(
        "INSERT INTO ducklake_metadata (key, value, scope, scope_id) VALUES (?, ?, NULL, NULL)",
    )
    .bind("data_path")
    .bind(&data_path)
    .execute(pool)
    .await?;

    // Insert schemas, tables, columns, and files
    for schema in &schemas {
        sqlx::query(
            "INSERT INTO ducklake_schema (schema_id, schema_name, path, path_is_relative, begin_snapshot, end_snapshot)
             VALUES (?, ?, ?, ?, ?, ?)"
        )
        .bind(schema.schema_id)
        .bind(&schema.schema_name)
        .bind(&schema.path)
        .bind(schema.path_is_relative as i32)
        .bind(1i64)
        .bind(None::<i64>)
        .execute(pool)
        .await?;

        let tables = duckdb_provider.list_tables(schema.schema_id, current_snapshot.snapshot_id)?;

        for table in &tables {
            sqlx::query(
                "INSERT INTO ducklake_table (table_id, schema_id, table_name, path, path_is_relative, begin_snapshot, end_snapshot)
                 VALUES (?, ?, ?, ?, ?, ?, ?)"
            )
            .bind(table.table_id)
            .bind(schema.schema_id)
            .bind(&table.table_name)
            .bind(&table.path)
            .bind(table.path_is_relative as i32)
            .bind(1i64)
            .bind(None::<i64>)
            .execute(pool)
            .await?;

            let columns = duckdb_provider
                .get_table_structure(table.table_id, duckdb_provider.get_current_snapshot()?)?;

            for (order, column) in columns.iter().enumerate() {
                sqlx::query(
                    "INSERT INTO ducklake_column (column_id, table_id, column_name, column_type, column_order, nulls_allowed)
                     VALUES (?, ?, ?, ?, ?, ?)"
                )
                .bind(column.column_id)
                .bind(table.table_id)
                .bind(&column.column_name)
                .bind(&column.column_type)
                .bind(order as i32)
                .bind(column.is_nullable as i32)
                .execute(pool)
                .await?;
            }

            let files = duckdb_provider
                .get_table_files_for_select(table.table_id, current_snapshot.snapshot_id)?;

            for (file_idx, file) in files.iter().enumerate() {
                let data_file_id = table.table_id * 1000 + file_idx as i64 + 1;

                sqlx::query(
                    "INSERT INTO ducklake_data_file (data_file_id, table_id, path, path_is_relative, file_size_bytes, footer_size, begin_snapshot)
                     VALUES (?, ?, ?, ?, ?, ?, ?)"
                )
                .bind(data_file_id)
                .bind(table.table_id)
                .bind(&file.file.path)
                .bind(file.file.path_is_relative as i32)
                .bind(file.file.file_size_bytes)
                .bind(file.file.footer_size)
                .bind(1i64)
                .execute(pool)
                .await?;

                if let Some(delete_file) = &file.delete_file {
                    let delete_file_id = data_file_id;

                    sqlx::query(
                        "INSERT INTO ducklake_delete_file (delete_file_id, data_file_id, table_id, path, path_is_relative,
                                                           file_size_bytes, footer_size, delete_count, begin_snapshot, end_snapshot)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                    )
                    .bind(delete_file_id)
                    .bind(data_file_id)
                    .bind(table.table_id)
                    .bind(&delete_file.path)
                    .bind(delete_file.path_is_relative as i32)
                    .bind(delete_file.file_size_bytes)
                    .bind(delete_file.footer_size)
                    .bind(None::<i64>)
                    .bind(1i64)
                    .bind(None::<i64>)
                    .execute(pool)
                    .await?;
                }
            }
        }
    }

    Ok((data_path, temp_dir))
}

#[tokio::test(flavor = "multi_thread")]
async fn test_schema_initialization_idempotent() {
    let provider = create_sqlite_provider().await.unwrap();

    // Initialize schema again - should be idempotent
    init_schema(&provider.pool)
        .await
        .expect("Schema initialization should be idempotent");

    // Verify tables exist by querying them
    let result = provider.get_current_snapshot();
    assert!(result.is_ok(), "Should be able to query after init");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_current_snapshot() {
    let provider = create_sqlite_provider().await.unwrap();

    // Initially should be 0 (no snapshots)
    let snapshot_id = provider
        .get_current_snapshot()
        .expect("Should get current snapshot");
    assert_eq!(snapshot_id, 0, "Should be 0 when no snapshots exist");

    // Populate test data
    populate_test_data(&provider)
        .await
        .expect("Failed to populate test data");

    // Should now return 2 (max snapshot_id)
    let snapshot_id = provider
        .get_current_snapshot()
        .expect("Should get current snapshot");
    assert_eq!(snapshot_id, 2, "Should return max snapshot_id");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_data_path() {
    let provider = create_sqlite_provider().await.unwrap();

    populate_test_data(&provider)
        .await
        .expect("Failed to populate test data");

    let data_path = provider.get_data_path().expect("Should get data path");

    assert_eq!(data_path, "file:///tmp/ducklake_data/");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_list_snapshots() {
    let provider = create_sqlite_provider().await.unwrap();

    populate_test_data(&provider)
        .await
        .expect("Failed to populate test data");

    let snapshots = provider.list_snapshots().expect("Should list snapshots");

    assert_eq!(snapshots.len(), 2, "Should have 2 snapshots");
    assert_eq!(snapshots[0].snapshot_id, 1);
    assert_eq!(snapshots[1].snapshot_id, 2);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_list_schemas_snapshot_isolation() {
    let provider = create_sqlite_provider().await.unwrap();

    populate_test_data(&provider)
        .await
        .expect("Failed to populate test data");

    // Snapshot 1 should only see test_schema
    let schemas = provider
        .list_schemas(1)
        .expect("Should list schemas for snapshot 1");

    assert_eq!(schemas.len(), 1, "Snapshot 1 should have 1 schema");
    assert_eq!(schemas[0].schema_name, "test_schema");

    // Snapshot 2 should see both schemas
    let schemas = provider
        .list_schemas(2)
        .expect("Should list schemas for snapshot 2");

    assert_eq!(schemas.len(), 2, "Snapshot 2 should have 2 schemas");

    let schema_names: Vec<_> = schemas.iter().map(|s| s.schema_name.as_str()).collect();
    assert!(schema_names.contains(&"test_schema"));
    assert!(schema_names.contains(&"schema2"));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_schema_by_name() {
    let provider = create_sqlite_provider().await.unwrap();

    populate_test_data(&provider)
        .await
        .expect("Failed to populate test data");

    // Should find test_schema
    let schema = provider
        .get_schema_by_name("test_schema", 1)
        .expect("Should get schema by name");

    assert!(schema.is_some(), "Should find test_schema");
    let schema = schema.unwrap();
    assert_eq!(schema.schema_name, "test_schema");
    assert_eq!(schema.schema_id, 1);

    // Should not find non-existent schema
    let schema = provider
        .get_schema_by_name("nonexistent", 1)
        .expect("Should handle non-existent schema");

    assert!(schema.is_none(), "Should not find nonexistent schema");

    // schema2 should not be visible in snapshot 1
    let schema = provider
        .get_schema_by_name("schema2", 1)
        .expect("Should handle schema not in snapshot");

    assert!(
        schema.is_none(),
        "schema2 should not be visible in snapshot 1"
    );

    // schema2 should be visible in snapshot 2
    let schema = provider
        .get_schema_by_name("schema2", 2)
        .expect("Should get schema by name");

    assert!(schema.is_some(), "schema2 should be visible in snapshot 2");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_list_tables() {
    let provider = create_sqlite_provider().await.unwrap();

    populate_test_data(&provider)
        .await
        .expect("Failed to populate test data");

    // Snapshot 1 should only see users table
    let tables = provider.list_tables(1, 1).expect("Should list tables");

    assert_eq!(tables.len(), 1, "Snapshot 1 should have 1 table");
    assert_eq!(tables[0].table_name, "users");

    // Snapshot 2 should see both tables
    let tables = provider.list_tables(1, 2).expect("Should list tables");

    assert_eq!(tables.len(), 2, "Snapshot 2 should have 2 tables");

    let table_names: Vec<_> = tables.iter().map(|t| t.table_name.as_str()).collect();
    assert!(table_names.contains(&"users"));
    assert!(table_names.contains(&"products"));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_table_by_name() {
    let provider = create_sqlite_provider().await.unwrap();

    populate_test_data(&provider)
        .await
        .expect("Failed to populate test data");

    // Should find users table
    let table = provider
        .get_table_by_name(1, "users", 1)
        .expect("Should get table by name");

    assert!(table.is_some(), "Should find users table");
    let table = table.unwrap();
    assert_eq!(table.table_name, "users");
    assert_eq!(table.table_id, 1);

    // Should not find non-existent table
    let table = provider
        .get_table_by_name(1, "nonexistent", 1)
        .expect("Should handle non-existent table");

    assert!(table.is_none(), "Should not find nonexistent table");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_table_exists() {
    let provider = create_sqlite_provider().await.unwrap();

    populate_test_data(&provider)
        .await
        .expect("Failed to populate test data");

    // users table should exist
    let exists = provider
        .table_exists(1, "users", 1)
        .expect("Should check if table exists");

    assert!(exists, "users table should exist");

    // nonexistent table should not exist
    let exists = provider
        .table_exists(1, "nonexistent", 1)
        .expect("Should check if table exists");

    assert!(!exists, "nonexistent table should not exist");

    // products table should not exist in snapshot 1
    let exists = provider
        .table_exists(1, "products", 1)
        .expect("Should check if table exists");

    assert!(!exists, "products table should not exist in snapshot 1");

    // products table should exist in snapshot 2
    let exists = provider
        .table_exists(1, "products", 2)
        .expect("Should check if table exists");

    assert!(exists, "products table should exist in snapshot 2");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_table_structure() {
    let provider = create_sqlite_provider().await.unwrap();

    populate_test_data(&provider)
        .await
        .expect("Failed to populate test data");

    let snapshot = provider
        .get_current_snapshot()
        .expect("Should get snapshot");
    let columns = provider
        .get_table_structure(1, snapshot)
        .expect("Should get table structure");

    assert_eq!(columns.len(), 3, "users table should have 3 columns");

    assert_eq!(columns[0].column_name, "id");
    assert_eq!(columns[0].column_type, "INT");
    assert!(!columns[0].is_nullable);

    assert_eq!(columns[1].column_name, "name");
    assert_eq!(columns[1].column_type, "VARCHAR");
    assert!(columns[1].is_nullable);

    assert_eq!(columns[2].column_name, "email");
    assert_eq!(columns[2].column_type, "VARCHAR");
    assert!(columns[2].is_nullable);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_table_files_for_select() {
    let provider = create_sqlite_provider().await.unwrap();

    populate_test_data(&provider)
        .await
        .expect("Failed to populate test data");

    let files = provider
        .get_table_files_for_select(1, 1)
        .expect("Should get table files");

    assert_eq!(files.len(), 2, "Should have 2 data files");

    // First file should have a delete file
    assert_eq!(files[0].file.path, "data_001.parquet");
    assert_eq!(files[0].file.file_size_bytes, 1024);
    assert_eq!(files[0].file.footer_size, Some(128));
    assert!(
        files[0].delete_file.is_some(),
        "First file should have delete file"
    );

    let delete_file = files[0].delete_file.as_ref().unwrap();
    assert_eq!(delete_file.path, "data_001.delete.parquet");
    assert_eq!(delete_file.file_size_bytes, 512);

    // Second file should not have a delete file
    assert_eq!(files[1].file.path, "data_002.parquet");
    assert_eq!(files[1].file.file_size_bytes, 2048);
    assert_eq!(files[1].file.footer_size, Some(256));
    assert!(
        files[1].delete_file.is_none(),
        "Second file should not have delete file"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_list_all_tables() {
    let provider = create_sqlite_provider().await.unwrap();

    populate_test_data(&provider)
        .await
        .expect("Failed to populate test data");

    // Snapshot 1 should only see 1 table
    let tables = provider.list_all_tables(1).expect("Should list all tables");

    assert_eq!(tables.len(), 1, "Snapshot 1 should have 1 table");
    assert_eq!(tables[0].schema_name, "test_schema");
    assert_eq!(tables[0].table.table_name, "users");

    // Snapshot 2 should see 2 tables
    let tables = provider.list_all_tables(2).expect("Should list all tables");

    assert_eq!(tables.len(), 2, "Snapshot 2 should have 2 tables");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_list_all_columns() {
    let provider = create_sqlite_provider().await.unwrap();

    populate_test_data(&provider)
        .await
        .expect("Failed to populate test data");

    let columns = provider
        .list_all_columns(1)
        .expect("Should list all columns");

    assert_eq!(columns.len(), 3, "Should have 3 columns from users table");

    assert_eq!(columns[0].schema_name, "test_schema");
    assert_eq!(columns[0].table_name, "users");
    assert_eq!(columns[0].column.column_name, "id");

    assert_eq!(columns[1].column.column_name, "name");
    assert_eq!(columns[2].column.column_name, "email");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_list_all_files() {
    let provider = create_sqlite_provider().await.unwrap();

    populate_test_data(&provider)
        .await
        .expect("Failed to populate test data");

    let files = provider.list_all_files(1).expect("Should list all files");

    assert_eq!(files.len(), 2, "Should have 2 files");

    assert_eq!(files[0].schema_name, "test_schema");
    assert_eq!(files[0].table_name, "users");
    assert_eq!(files[0].file.file.path, "data_001.parquet");
    assert!(files[0].file.delete_file.is_some());

    assert_eq!(files[1].file.file.path, "data_002.parquet");
    assert!(files[1].file.delete_file.is_none());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_concurrent_access() {
    let provider = create_sqlite_provider().await.unwrap();

    populate_test_data(&provider)
        .await
        .expect("Failed to populate test data");

    let provider = Arc::new(provider);

    // Spawn 10 concurrent tasks
    let mut tasks = Vec::new();
    for _ in 0..10 {
        let provider = provider.clone();
        let task = tokio::spawn(async move {
            let snapshot = provider
                .get_current_snapshot()
                .expect("Should get snapshot");
            let _schemas = provider.list_schemas(1).expect("Should list schemas");
            let _tables = provider.list_tables(1, 1).expect("Should list tables");
            let _columns = provider
                .get_table_structure(1, snapshot)
                .expect("Should get structure");
        });
        tasks.push(task);
    }

    for task in tasks {
        task.await.expect("Task should complete successfully");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_datafusion_integration() {
    let provider = create_sqlite_provider().await.unwrap();

    populate_test_data(&provider)
        .await
        .expect("Failed to populate test data");

    let catalog = DuckLakeCatalog::new(provider).expect("Should create catalog");

    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));

    // Query information_schema
    let df = ctx
        .sql("SELECT schema_name FROM ducklake.information_schema.schemata")
        .await
        .expect("Should query information_schema");

    let results = df.collect().await.expect("Should collect results");
    assert!(!results.is_empty(), "Should have schema results");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_query_real_parquet_files() {
    let provider = create_sqlite_provider().await.unwrap();

    let (_data_path, _temp_dir) = populate_from_duckdb_catalog(&provider)
        .await
        .expect("Failed to populate from DuckDB catalog");

    let catalog = DuckLakeCatalog::new(provider).expect("Should create catalog");

    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));

    // Query actual table data
    let df = ctx
        .sql("SELECT * FROM ducklake.main.users ORDER BY id")
        .await
        .expect("Should query table data");

    let results = df.collect().await.expect("Should collect results");

    assert_eq!(results.len(), 1, "Should have one batch");
    let batch = &results[0];
    assert_eq!(batch.num_rows(), 4, "Should have 4 rows");

    // Verify schema
    assert_eq!(batch.num_columns(), 3, "Should have 3 columns");
    let schema = batch.schema();
    assert_eq!(schema.field(0).name(), "id");
    assert_eq!(schema.field(1).name(), "name");
    assert_eq!(schema.field(2).name(), "email");

    // Verify first row data
    use datafusion::arrow::array::{Int32Array, StringArray};
    let id_col = batch
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    // DuckLake string columns scan as Utf8View; cast to Utf8 to read via StringArray.
    let name_col_arr = datafusion::arrow::compute::cast(
        batch.column(1),
        &datafusion::arrow::datatypes::DataType::Utf8,
    )
    .unwrap();
    let name_col = name_col_arr.as_any().downcast_ref::<StringArray>().unwrap();
    let email_col_arr = datafusion::arrow::compute::cast(
        batch.column(2),
        &datafusion::arrow::datatypes::DataType::Utf8,
    )
    .unwrap();
    let email_col = email_col_arr
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    assert_eq!(id_col.value(0), 1);
    assert_eq!(name_col.value(0), "Alice");
    assert_eq!(email_col.value(0), "alice@example.com");

    assert_eq!(id_col.value(1), 2);
    assert_eq!(name_col.value(1), "Bob");
    assert_eq!(email_col.value(1), "bob@example.com");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_query_with_filter() {
    let provider = create_sqlite_provider().await.unwrap();

    let (_data_path, _temp_dir) = populate_from_duckdb_catalog(&provider)
        .await
        .expect("Failed to populate from DuckDB catalog");

    let catalog = DuckLakeCatalog::new(provider).expect("Should create catalog");
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));

    // Query with WHERE filter
    let df = ctx
        .sql("SELECT name, email FROM ducklake.main.users WHERE id > 2 ORDER BY id")
        .await
        .expect("Should query with filter");

    let results = df.collect().await.expect("Should collect results");

    assert_eq!(results.len(), 1, "Should have one batch");
    let batch = &results[0];
    assert_eq!(batch.num_rows(), 2, "Should have 2 rows (Charlie, Diana)");

    use datafusion::arrow::array::StringArray;
    let name_col_arr = datafusion::arrow::compute::cast(
        batch.column(0),
        &datafusion::arrow::datatypes::DataType::Utf8,
    )
    .unwrap();
    let name_col = name_col_arr.as_any().downcast_ref::<StringArray>().unwrap();

    assert_eq!(name_col.value(0), "Charlie");
    assert_eq!(name_col.value(1), "Diana");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_schema_capability_probe_memoized_positive_only() {
    let provider = create_sqlite_provider().await.unwrap();

    populate_test_data(&provider)
        .await
        .expect("Failed to populate test data");

    // Minimal fixture: every optional capability is absent, so the
    // positive-only memo must stay empty and repeated calls keep re-probing
    // (the status quo for legacy catalogs) while returning identical results.
    assert!(!provider.schema_capabilities_cached());
    let first = provider
        .get_table_files_for_select(1, 1)
        .expect("Should get table files");
    assert!(
        !provider.schema_capabilities_cached(),
        "a negative probe result must not be cached"
    );
    let second = provider
        .get_table_files_for_select(1, 1)
        .expect("Should get table files");
    assert_eq!(first.len(), 2);
    assert_eq!(format!("{first:?}"), format!("{second:?}"));

    // Upgrade the catalog mid-flight: add every capability the provider
    // probes. Because false was never cached, the very next call must see
    // the upgrade and memoize the all-true answer.
    let pool = &provider.pool;
    sqlx::query("ALTER TABLE ducklake_data_file ADD COLUMN partial_max INTEGER")
        .execute(pool)
        .await
        .unwrap();
    sqlx::query("ALTER TABLE ducklake_delete_file ADD COLUMN partial_max INTEGER")
        .execute(pool)
        .await
        .unwrap();
    sqlx::query("ALTER TABLE ducklake_data_file ADD COLUMN partition_id INTEGER")
        .execute(pool)
        .await
        .unwrap();
    sqlx::query(
        "CREATE TABLE ducklake_schema_versions (
            begin_snapshot INTEGER NOT NULL,
            schema_version INTEGER NOT NULL,
            table_id INTEGER NOT NULL
        )",
    )
    .execute(pool)
    .await
    .unwrap();
    sqlx::query(
        "CREATE TABLE ducklake_inlined_data_tables (
            table_id INTEGER NOT NULL,
            table_name TEXT NOT NULL,
            schema_snapshot INTEGER
        )",
    )
    .execute(pool)
    .await
    .unwrap();

    let migrated_first = provider
        .get_table_files_for_select(1, 1)
        .expect("Should get table files");
    assert!(
        provider.schema_capabilities_cached(),
        "an all-true probe result must be cached after the first call"
    );
    let migrated_second = provider
        .get_table_files_for_select(1, 1)
        .expect("Should get table files");
    assert_eq!(migrated_first.len(), 2);
    assert_eq!(
        format!("{migrated_first:?}"),
        format!("{migrated_second:?}")
    );

    // Clones share the memo (the cell is Arc-shared).
    assert!(provider.clone().schema_capabilities_cached());
}
