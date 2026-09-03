#![cfg(all(feature = "metadata-postgres", feature = "metadata-duckdb"))]
//! PostgreSQL metadata provider tests
//!
//! This test suite verifies the PostgreSQL metadata provider implementation,
//! including all MetadataProvider trait methods, schema initialization,
//! concurrent access, and error handling.
//!
//! ## Test Setup
//!
//! Tests use testcontainers to spin up a temporary PostgreSQL instance.
//! Each test creates its own database with test data to ensure isolation.
//!
//! ## Coverage
//!
//! - Schema initialization (idempotent)
//! - All MetadataProvider trait methods
//! - Snapshot isolation and temporal queries
//! - Concurrent access and thread safety
//! - Error handling and edge cases

use crate::common;

use arrow::datatypes::{DataType, Field, Schema};
use datafusion::logical_expr::Operator;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_expr::expressions::{BinaryExpr, Column as PhysColumn, lit};
use datafusion::prelude::*;
use datafusion_ducklake::metadata_provider::DuckLakeTableColumn;
use datafusion_ducklake::stats_filter::{StatsFilter, lower_predicate};
use datafusion_ducklake::{
    DuckLakeCatalog, DuckdbMetadataProvider, PostgresMetadataProvider,
    metadata_provider::MetadataProvider,
};
use sqlx::PgPool;
use std::sync::Arc;
use tempfile::TempDir;
use testcontainers::ImageExt;
use testcontainers::runners::AsyncRunner;
use testcontainers_modules::postgres::Postgres;

/// Initialize DuckLake catalog schema in PostgreSQL (for tests only)
async fn init_schema(pool: &PgPool) -> anyhow::Result<()> {
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS ducklake_snapshot (
            snapshot_id BIGINT PRIMARY KEY,
            snapshot_time TIMESTAMP
        )",
    )
    .execute(pool)
    .await?;

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS ducklake_column_mapping (
            mapping_id BIGINT PRIMARY KEY,
            table_id BIGINT NOT NULL,
            type TEXT NOT NULL
        )",
    )
    .execute(pool)
    .await?;
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS ducklake_name_mapping (
            mapping_id BIGINT NOT NULL,
            column_id BIGINT NOT NULL,
            source_name TEXT NOT NULL,
            target_field_id BIGINT NOT NULL,
            parent_column BIGINT,
            is_partition BOOLEAN NOT NULL
        )",
    )
    .execute(pool)
    .await?;

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS ducklake_schema (
            schema_id BIGINT PRIMARY KEY,
            schema_name VARCHAR NOT NULL,
            path VARCHAR NOT NULL,
            path_is_relative BOOLEAN NOT NULL,
            begin_snapshot BIGINT NOT NULL,
            end_snapshot BIGINT
        )",
    )
    .execute(pool)
    .await?;

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS ducklake_table (
            table_id BIGINT PRIMARY KEY,
            schema_id BIGINT NOT NULL,
            table_name VARCHAR NOT NULL,
            path VARCHAR NOT NULL,
            path_is_relative BOOLEAN NOT NULL,
            begin_snapshot BIGINT NOT NULL,
            end_snapshot BIGINT,
            FOREIGN KEY (schema_id) REFERENCES ducklake_schema(schema_id)
        )",
    )
    .execute(pool)
    .await?;

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS ducklake_view (
            view_id BIGINT,
            view_uuid UUID,
            schema_id BIGINT NOT NULL,
            view_name VARCHAR NOT NULL,
            dialect VARCHAR NOT NULL,
            sql VARCHAR NOT NULL,
            column_aliases VARCHAR,
            begin_snapshot BIGINT NOT NULL,
            end_snapshot BIGINT
        )",
    )
    .execute(pool)
    .await?;

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS ducklake_column (
            column_id BIGINT PRIMARY KEY,
            table_id BIGINT NOT NULL,
            column_name VARCHAR NOT NULL,
            column_type VARCHAR NOT NULL,
            column_order INTEGER NOT NULL,
            FOREIGN KEY (table_id) REFERENCES ducklake_table(table_id)
        )",
    )
    .execute(pool)
    .await?;

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS ducklake_data_file (
            data_file_id BIGINT PRIMARY KEY,
            table_id BIGINT NOT NULL,
            path VARCHAR NOT NULL,
            path_is_relative BOOLEAN NOT NULL,
            file_size_bytes BIGINT NOT NULL,
            footer_size BIGINT,
            FOREIGN KEY (table_id) REFERENCES ducklake_table(table_id)
        )",
    )
    .execute(pool)
    .await?;

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS ducklake_delete_file (
            delete_file_id BIGINT PRIMARY KEY,
            data_file_id BIGINT NOT NULL,
            table_id BIGINT NOT NULL,
            path VARCHAR NOT NULL,
            path_is_relative BOOLEAN NOT NULL,
            file_size_bytes BIGINT NOT NULL,
            footer_size BIGINT,
            delete_count BIGINT,
            begin_snapshot BIGINT NOT NULL,
            end_snapshot BIGINT,
            FOREIGN KEY (data_file_id) REFERENCES ducklake_data_file(data_file_id),
            FOREIGN KEY (table_id) REFERENCES ducklake_table(table_id)
        )",
    )
    .execute(pool)
    .await?;

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS ducklake_metadata (
            key VARCHAR NOT NULL PRIMARY KEY,
            value VARCHAR NOT NULL,
            scope VARCHAR,
            scope_id BIGINT
        )",
    )
    .execute(pool)
    .await?;

    sqlx::query("CREATE INDEX IF NOT EXISTS idx_schema_snapshot ON ducklake_schema(begin_snapshot, end_snapshot)")
        .execute(pool)
        .await?;

    sqlx::query("CREATE INDEX IF NOT EXISTS idx_table_schema ON ducklake_table(schema_id)")
        .execute(pool)
        .await?;

    sqlx::query("CREATE INDEX IF NOT EXISTS idx_table_snapshot ON ducklake_table(begin_snapshot, end_snapshot)")
        .execute(pool)
        .await?;

    sqlx::query(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_schema_name_active
         ON ducklake_schema(schema_name) WHERE end_snapshot IS NULL",
    )
    .execute(pool)
    .await?;

    sqlx::query(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_table_name_active
         ON ducklake_table(schema_id, table_name) WHERE end_snapshot IS NULL",
    )
    .execute(pool)
    .await?;

    sqlx::query(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_column_name_unique
         ON ducklake_column(table_id, column_name)",
    )
    .execute(pool)
    .await?;

    Ok(())
}

/// Helper to create a PostgreSQL provider with initialized schema
async fn create_postgres_provider() -> anyhow::Result<(
    PostgresMetadataProvider,
    testcontainers::ContainerAsync<Postgres>,
)> {
    let container = Postgres::default().start().await?;

    let host = "127.0.0.1";
    let port = container.get_host_port_ipv4(5432).await?;
    let conn_str = format!("postgresql://postgres:postgres@{}:{}/postgres", host, port);

    let provider = PostgresMetadataProvider::new(&conn_str)
        .await
        .expect("Failed to create provider");
    init_schema(&provider.pool).await?;

    Ok((provider, container))
}

/// Helper to populate test data in PostgreSQL
async fn populate_test_data(provider: &PostgresMetadataProvider) -> anyhow::Result<()> {
    // Get the pool for direct SQL access
    let pool = &provider.pool;

    // Insert snapshots
    sqlx::query("INSERT INTO ducklake_snapshot (snapshot_id, snapshot_time) VALUES ($1, NOW())")
        .bind(1i64)
        .execute(pool)
        .await?;

    sqlx::query("INSERT INTO ducklake_snapshot (snapshot_id, snapshot_time) VALUES ($1, NOW())")
        .bind(2i64)
        .execute(pool)
        .await?;

    // Insert metadata (data_path)
    sqlx::query(
        "INSERT INTO ducklake_metadata (key, value, scope, scope_id) VALUES ($1, $2, NULL, NULL)",
    )
    .bind("data_path")
    .bind("file:///tmp/ducklake_data/")
    .execute(pool)
    .await?;

    // Insert schema
    sqlx::query(
        "INSERT INTO ducklake_schema (schema_id, schema_name, path, path_is_relative, begin_snapshot, end_snapshot)
         VALUES ($1, $2, $3, $4, $5, $6)"
    )
    .bind(1i64)
    .bind("test_schema")
    .bind("test_schema/")
    .bind(true)
    .bind(1i64)
    .bind(None::<i64>)
    .execute(pool)
    .await?;

    // Insert another schema (only in snapshot 2)
    sqlx::query(
        "INSERT INTO ducklake_schema (schema_id, schema_name, path, path_is_relative, begin_snapshot, end_snapshot)
         VALUES ($1, $2, $3, $4, $5, $6)"
    )
    .bind(2i64)
    .bind("schema2")
    .bind("schema2/")
    .bind(true)
    .bind(2i64)
    .bind(None::<i64>)
    .execute(pool)
    .await?;

    // Insert table
    sqlx::query(
        "INSERT INTO ducklake_table (table_id, schema_id, table_name, path, path_is_relative, begin_snapshot, end_snapshot)
         VALUES ($1, $2, $3, $4, $5, $6, $7)"
    )
    .bind(1i64)
    .bind(1i64)
    .bind("users")
    .bind("users/")
    .bind(true)
    .bind(1i64)
    .bind(None::<i64>)
    .execute(pool)
    .await?;

    // Insert another table (only in snapshot 2)
    sqlx::query(
        "INSERT INTO ducklake_table (table_id, schema_id, table_name, path, path_is_relative, begin_snapshot, end_snapshot)
         VALUES ($1, $2, $3, $4, $5, $6, $7)"
    )
    .bind(2i64)
    .bind(1i64)
    .bind("products")
    .bind("products/")
    .bind(true)
    .bind(2i64)
    .bind(None::<i64>)
    .execute(pool)
    .await?;

    // Insert columns for users table
    sqlx::query(
        "INSERT INTO ducklake_column (column_id, table_id, column_name, column_type, column_order)
         VALUES ($1, $2, $3, $4, $5)",
    )
    .bind(1i64)
    .bind(1i64)
    .bind("id")
    .bind("INT")
    .bind(0i32)
    .execute(pool)
    .await?;

    sqlx::query(
        "INSERT INTO ducklake_column (column_id, table_id, column_name, column_type, column_order)
         VALUES ($1, $2, $3, $4, $5)",
    )
    .bind(2i64)
    .bind(1i64)
    .bind("name")
    .bind("VARCHAR")
    .bind(1i32)
    .execute(pool)
    .await?;

    sqlx::query(
        "INSERT INTO ducklake_column (column_id, table_id, column_name, column_type, column_order)
         VALUES ($1, $2, $3, $4, $5)",
    )
    .bind(3i64)
    .bind(1i64)
    .bind("email")
    .bind("VARCHAR")
    .bind(2i32)
    .execute(pool)
    .await?;

    // Insert data file
    sqlx::query(
        "INSERT INTO ducklake_data_file (data_file_id, table_id, path, path_is_relative, file_size_bytes, footer_size)
         VALUES ($1, $2, $3, $4, $5, $6)"
    )
    .bind(1i64)
    .bind(1i64)
    .bind("data_001.parquet")
    .bind(true)
    .bind(1024i64)
    .bind(Some(128i64))
    .execute(pool)
    .await?;

    sqlx::query(
        "INSERT INTO ducklake_data_file (data_file_id, table_id, path, path_is_relative, file_size_bytes, footer_size)
         VALUES ($1, $2, $3, $4, $5, $6)"
    )
    .bind(2i64)
    .bind(1i64)
    .bind("data_002.parquet")
    .bind(true)
    .bind(2048i64)
    .bind(Some(256i64))
    .execute(pool)
    .await?;

    // Insert delete file for first data file
    sqlx::query(
        "INSERT INTO ducklake_delete_file (delete_file_id, data_file_id, table_id, path, path_is_relative,
                                           file_size_bytes, footer_size, delete_count, begin_snapshot, end_snapshot)
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)"
    )
    .bind(1i64)
    .bind(1i64)
    .bind(1i64)
    .bind("data_001.delete.parquet")
    .bind(true)
    .bind(512i64)
    .bind(Some(64i64))
    .bind(Some(5i64))
    .bind(1i64)
    .bind(None::<i64>)
    .execute(pool)
    .await?;

    Ok(())
}

/// Helper to populate PostgreSQL with metadata from a DuckDB-created catalog
///
/// This creates actual Parquet files using DuckDB + DuckLake extension,
/// then reads the metadata from DuckDB and populates PostgreSQL with it.
/// Both providers can then query the same real Parquet files.
///
/// Returns the data_path and TempDir. The TempDir must be kept alive for the
/// duration of the test to prevent cleanup of Parquet files.
async fn populate_from_duckdb_catalog(
    provider: &PostgresMetadataProvider,
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

    // Step 3: Populate PostgreSQL with metadata from DuckDB
    // Use a transaction for atomicity (all-or-nothing)
    let mut tx = provider.pool.begin().await?;

    // Insert snapshots
    for snapshot in &snapshots {
        // Parse timestamp string to NaiveDateTime if present
        let timestamp_value: Option<sqlx::types::chrono::NaiveDateTime> =
            snapshot.timestamp.as_ref().and_then(|ts_str| {
                sqlx::types::chrono::NaiveDateTime::parse_from_str(ts_str, "%Y-%m-%d %H:%M:%S%.6f")
                    .ok()
            });

        sqlx::query("INSERT INTO ducklake_snapshot (snapshot_id, snapshot_time) VALUES ($1, $2)")
            .bind(snapshot.snapshot_id)
            .bind(timestamp_value)
            .execute(&mut *tx)
            .await?;
    }

    // Insert data_path metadata
    sqlx::query(
        "INSERT INTO ducklake_metadata (key, value, scope, scope_id) VALUES ($1, $2, NULL, NULL)",
    )
    .bind("data_path")
    .bind(&data_path)
    .execute(&mut *tx)
    .await?;

    // Insert schemas, tables, columns, and files
    for schema in &schemas {
        sqlx::query(
            "INSERT INTO ducklake_schema (schema_id, schema_name, path, path_is_relative, begin_snapshot, end_snapshot)
             VALUES ($1, $2, $3, $4, $5, $6)"
        )
        .bind(schema.schema_id)
        .bind(&schema.schema_name)
        .bind(&schema.path)
        .bind(schema.path_is_relative)
        // NOTE: Hardcoded to snapshot 1 - this assumes single-snapshot catalogs.
        // For multi-snapshot testing, DuckDB metadata would need to expose
        // begin_snapshot/end_snapshot for schemas and tables.
        .bind(1i64) // begin_snapshot
        .bind(None::<i64>) // end_snapshot (active)
        .execute(&mut *tx)
        .await?;

        // Get tables for this schema
        let tables = duckdb_provider.list_tables(schema.schema_id, current_snapshot.snapshot_id)?;

        for table in &tables {
            sqlx::query(
                "INSERT INTO ducklake_table (table_id, schema_id, table_name, path, path_is_relative, begin_snapshot, end_snapshot)
                 VALUES ($1, $2, $3, $4, $5, $6, $7)"
            )
            .bind(table.table_id)
            .bind(schema.schema_id)
            .bind(&table.table_name)
            .bind(&table.path)
            .bind(table.path_is_relative)
            .bind(1i64) // begin_snapshot
            .bind(None::<i64>) // end_snapshot (active)
            .execute(&mut *tx)
            .await?;

            // Get columns for this table
            let columns = duckdb_provider
                .get_table_structure(table.table_id, duckdb_provider.get_current_snapshot()?)?;

            for (order, column) in columns.iter().enumerate() {
                sqlx::query(
                    "INSERT INTO ducklake_column (column_id, table_id, column_name, column_type, column_order)
                     VALUES ($1, $2, $3, $4, $5)"
                )
                .bind(column.column_id)
                .bind(table.table_id)
                .bind(&column.column_name)
                .bind(&column.column_type)
                .bind(order as i32)
                .execute(&mut *tx)
                .await?;
            }

            // Get data files for this table
            let files = duckdb_provider
                .get_table_files_for_select(table.table_id, current_snapshot.snapshot_id)?;

            for (file_idx, file) in files.iter().enumerate() {
                let data_file_id = table.table_id * 1000 + file_idx as i64 + 1;

                sqlx::query(
                    "INSERT INTO ducklake_data_file (data_file_id, table_id, path, path_is_relative, file_size_bytes, footer_size)
                     VALUES ($1, $2, $3, $4, $5, $6)"
                )
                .bind(data_file_id)
                .bind(table.table_id)
                .bind(&file.file.path)
                .bind(file.file.path_is_relative)
                .bind(file.file.file_size_bytes)
                .bind(file.file.footer_size)
                .execute(&mut *tx)
                .await?;

                // Insert delete file if present
                if let Some(delete_file) = &file.delete_file {
                    let delete_file_id = data_file_id;

                    sqlx::query(
                        "INSERT INTO ducklake_delete_file (delete_file_id, data_file_id, table_id, path, path_is_relative,
                                                           file_size_bytes, footer_size, delete_count, begin_snapshot, end_snapshot)
                         VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)"
                    )
                    .bind(delete_file_id)
                    .bind(data_file_id)
                    .bind(table.table_id)
                    .bind(&delete_file.path)
                    .bind(delete_file.path_is_relative)
                    .bind(delete_file.file_size_bytes)
                    .bind(delete_file.footer_size)
                    .bind(None::<i64>) // delete_count
                    .bind(1i64) // begin_snapshot
                    .bind(None::<i64>) // end_snapshot (active)
                    .execute(&mut *tx)
                    .await?;
                }
            }
        }
    }

    // Commit the transaction atomically
    tx.commit().await?;

    // Return temp_dir so caller can keep it alive during the test
    // When temp_dir is dropped, Parquet files are automatically cleaned up
    Ok((data_path, temp_dir))
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn test_schema_initialization_idempotent() {
    let (provider, _container) = create_postgres_provider().await.unwrap();

    // Initialize schema again - should be idempotent
    init_schema(&provider.pool)
        .await
        .expect("Schema initialization should be idempotent");

    // Verify tables exist by querying them
    let result = provider.get_current_snapshot();
    assert!(result.is_ok(), "Should be able to query after init");
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn test_get_current_snapshot() {
    let (provider, _container) = create_postgres_provider().await.unwrap();

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
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn test_get_data_path() {
    let (provider, _container) = create_postgres_provider().await.unwrap();

    populate_test_data(&provider)
        .await
        .expect("Failed to populate test data");

    let data_path = provider.get_data_path().expect("Should get data path");

    assert_eq!(data_path, "file:///tmp/ducklake_data/");
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn test_list_snapshots() {
    let (provider, _container) = create_postgres_provider().await.unwrap();

    populate_test_data(&provider)
        .await
        .expect("Failed to populate test data");

    let snapshots = provider.list_snapshots().expect("Should list snapshots");

    assert_eq!(snapshots.len(), 2, "Should have 2 snapshots");
    assert_eq!(snapshots[0].snapshot_id, 1);
    assert_eq!(snapshots[1].snapshot_id, 2);
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn test_list_schemas_snapshot_isolation() {
    let (provider, _container) = create_postgres_provider().await.unwrap();

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
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn test_get_schema_by_name() {
    let (provider, _container) = create_postgres_provider().await.unwrap();

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
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn test_list_tables() {
    let (provider, _container) = create_postgres_provider().await.unwrap();

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
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn test_list_views() -> anyhow::Result<()> {
    let (provider, _container) = create_postgres_provider().await?;
    populate_test_data(&provider).await?;
    sqlx::query(
        "INSERT INTO ducklake_view
         (view_id, schema_id, view_name, dialect, sql, column_aliases, begin_snapshot, end_snapshot)
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8),
                ($9, $10, $11, $12, $13, $14, $15, $16)",
    )
    .bind(7i64)
    .bind(1i64)
    .bind("active_view")
    .bind("duckdb")
    .bind("SELECT id FROM users")
    .bind("\"identifier\"")
    .bind(2i64)
    .bind(None::<i64>)
    .bind(8i64)
    .bind(1i64)
    .bind("expired_view")
    .bind("duckdb")
    .bind("SELECT name FROM users")
    .bind("")
    .bind(1i64)
    .bind(2i64)
    .execute(&provider.pool)
    .await?;

    assert_eq!(provider.list_views(1, 1)?[0].view_name, "expired_view");
    let views = provider.list_views(1, 2)?;
    assert_eq!(views.len(), 1);
    assert_eq!(views[0].view_name, "active_view");
    assert_eq!(views[0].begin_snapshot, 2);
    assert_eq!(views[0].column_aliases.as_deref(), Some("\"identifier\""));
    assert_eq!(
        provider.get_view_by_name(1, "active_view", 2)?,
        Some(views[0].clone())
    );
    let all_views = provider.list_all_views(2)?;
    assert_eq!(all_views.len(), 1);
    assert_eq!(all_views[0].schema_name, "test_schema");
    assert_eq!(all_views[0].view, views[0]);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn test_get_table_by_name() {
    let (provider, _container) = create_postgres_provider().await.unwrap();

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
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn test_table_exists() {
    let (provider, _container) = create_postgres_provider().await.unwrap();

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
#[ignore = "provider-test fixture drift vs current schema; fixed with the provider rework"]
async fn test_get_table_structure() {
    let (provider, _container) = create_postgres_provider().await.unwrap();

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
    assert_eq!(columns[0].column_type, "int32");

    assert_eq!(columns[1].column_name, "name");
    assert_eq!(columns[1].column_type, "varchar");

    assert_eq!(columns[2].column_name, "email");
    assert_eq!(columns[2].column_type, "varchar");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_name_mapping() -> anyhow::Result<()> {
    let (provider, _container) = create_postgres_provider().await?;
    sqlx::query(
        "INSERT INTO ducklake_column_mapping(mapping_id, table_id, type)
         VALUES (7, 1, 'map_by_name')",
    )
    .execute(&provider.pool)
    .await?;
    sqlx::query(
        "INSERT INTO ducklake_name_mapping(
            mapping_id, column_id, source_name, target_field_id, parent_column, is_partition
         ) VALUES
            (7, 1, 'nested', 3, NULL, false),
            (7, 2, 'child', 4, 1, false),
            (7, 3, 'part', 5, NULL, true)",
    )
    .execute(&provider.pool)
    .await?;

    let mapping = provider.get_name_mapping(7)?;
    assert_eq!(mapping.mapping_id, 7);
    assert_eq!(mapping.table_id, 1);
    assert_eq!(mapping.mapping_type, "map_by_name");
    assert_eq!(mapping.entries.len(), 3);
    assert!(mapping.entries[1].is_partition);
    assert_eq!(mapping.entries[2].parent_column, Some(1));
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "provider-test fixture drift vs current schema; fixed with the provider rework"]
async fn test_get_table_files_for_select() {
    let (provider, _container) = create_postgres_provider().await.unwrap();

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
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn test_list_all_tables() {
    let (provider, _container) = create_postgres_provider().await.unwrap();

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
#[ignore = "provider-test fixture drift vs current schema; fixed with the provider rework"]
async fn test_list_all_columns() {
    let (provider, _container) = create_postgres_provider().await.unwrap();

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
#[ignore = "provider-test fixture drift vs current schema; fixed with the provider rework"]
async fn test_list_all_files() {
    let (provider, _container) = create_postgres_provider().await.unwrap();

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
#[ignore = "provider-test fixture drift vs current schema; fixed with the provider rework"]
async fn test_concurrent_access() {
    let (provider, _container) = create_postgres_provider().await.unwrap();

    populate_test_data(&provider)
        .await
        .expect("Failed to populate test data");

    // Clone provider for concurrent access (Arc allows sharing)
    let provider = Arc::new(provider);

    // Spawn 10 concurrent tasks
    let mut tasks = Vec::new();
    for _ in 0..10 {
        let provider = provider.clone();
        let task = tokio::spawn(async move {
            // Each task performs multiple operations
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

    // Wait for all tasks to complete
    for task in tasks {
        task.await.expect("Task should complete successfully");
    }
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn test_datafusion_integration() {
    let (provider, _container) = create_postgres_provider().await.unwrap();

    populate_test_data(&provider)
        .await
        .expect("Failed to populate test data");

    // Create DuckLake catalog with PostgreSQL provider
    let catalog = DuckLakeCatalog::new(provider).expect("Should create catalog");

    // Register with DataFusion
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
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn test_error_invalid_connection_string() {
    let result = PostgresMetadataProvider::new("invalid://connection:string").await;
    assert!(
        result.is_err(),
        "Should fail with invalid connection string"
    );
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn test_error_connection_refused() {
    let result =
        PostgresMetadataProvider::new("postgresql://postgres:postgres@localhost:9999/db").await;
    assert!(result.is_err(), "Should fail when connection is refused");
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "provider-test fixture drift vs current schema; fixed with the provider rework"]
async fn test_query_real_parquet_files() {
    let (provider, _container) = create_postgres_provider().await.unwrap();

    // Populate PostgreSQL with metadata from DuckDB-created catalog
    let (_data_path, _temp_dir) = populate_from_duckdb_catalog(&provider)
        .await
        .expect("Failed to populate from DuckDB catalog");

    // Create DuckLake catalog with PostgreSQL provider
    let catalog = DuckLakeCatalog::new(provider).expect("Should create catalog");

    // Register with DataFusion
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));

    // Query actual table data (not just information_schema)
    let df = ctx
        .sql("SELECT * FROM ducklake.main.users ORDER BY id")
        .await
        .expect("Should query table data");

    let results = df.collect().await.expect("Should collect results");

    // Verify we got the expected 4 rows from create_catalog_no_deletes
    assert_eq!(results.len(), 1, "Should have one batch");
    let batch = &results[0];
    assert_eq!(batch.num_rows(), 4, "Should have 4 rows");

    // Verify schema
    assert_eq!(batch.num_columns(), 3, "Should have 3 columns");
    let schema = batch.schema();
    assert_eq!(schema.field(0).name(), "id");
    assert_eq!(schema.field(1).name(), "name");
    assert_eq!(schema.field(2).name(), "email");

    // Verify first row data (Alice)
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
#[ignore = "provider-test fixture drift vs current schema; fixed with the provider rework"]
async fn test_query_with_filter() {
    let (provider, _container) = create_postgres_provider().await.unwrap();

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

/// Bring the minimal fixture schema up to a fully-migrated catalog: every
/// optional capability the provider probes (`partial_max` columns, the
/// `ducklake_schema_versions` ledger, and the inlined-data registry) plus the
/// file columns the scan projection reads.
async fn migrate_fixture_to_current_schema(pool: &PgPool) -> anyhow::Result<()> {
    sqlx::query(
        "ALTER TABLE ducklake_data_file
             ADD COLUMN IF NOT EXISTS encryption_key VARCHAR,
             ADD COLUMN IF NOT EXISTS record_count BIGINT,
             ADD COLUMN IF NOT EXISTS row_id_start BIGINT,
             ADD COLUMN IF NOT EXISTS begin_snapshot BIGINT NOT NULL DEFAULT 1,
             ADD COLUMN IF NOT EXISTS end_snapshot BIGINT,
             ADD COLUMN IF NOT EXISTS partial_max BIGINT,
             ADD COLUMN IF NOT EXISTS partition_id BIGINT,
             ADD COLUMN IF NOT EXISTS mapping_id BIGINT",
    )
    .execute(pool)
    .await?;
    sqlx::query(
        "ALTER TABLE ducklake_delete_file
             ADD COLUMN IF NOT EXISTS encryption_key VARCHAR,
             ADD COLUMN IF NOT EXISTS partial_max BIGINT",
    )
    .execute(pool)
    .await?;
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS ducklake_schema_versions (
             begin_snapshot BIGINT NOT NULL,
             schema_version BIGINT NOT NULL,
             table_id BIGINT NOT NULL
         )",
    )
    .execute(pool)
    .await?;
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS ducklake_inlined_data_tables (
             table_id BIGINT NOT NULL,
             table_name VARCHAR NOT NULL,
             schema_version BIGINT NOT NULL
         )",
    )
    .execute(pool)
    .await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn test_schema_capability_probe_memoized_positive_only() {
    let (provider, _container) = create_postgres_provider().await.unwrap();

    populate_test_data(&provider)
        .await
        .expect("Failed to populate test data");
    migrate_fixture_to_current_schema(&provider.pool)
        .await
        .expect("Failed to migrate fixture schema");

    // Fully-migrated catalog: the first scan probes once (all capabilities
    // true) and memoizes; the second scan reuses the memo and must return
    // identical results.
    assert!(!provider.schema_capabilities_cached());
    let first = provider
        .get_table_files_for_select(1, 1)
        .expect("Should get table files");
    assert!(
        provider.schema_capabilities_cached(),
        "an all-true probe result must be cached after the first call"
    );
    let second = provider
        .get_table_files_for_select(1, 1)
        .expect("Should get table files");
    assert_eq!(first.len(), 2);
    assert_eq!(format!("{first:?}"), format!("{second:?}"));

    // Clones share the memo (the cell is Arc-shared).
    assert!(provider.clone().schema_capabilities_cached());
}

// ---------------------------------------------------------------------------
// Filter pushdown into the file-listing SQL
// ---------------------------------------------------------------------------

/// An `Int32` column (`column_id` 7) and a `TIMESTAMP` one (`column_id` 9) on
/// `table_id` 1, and five data files covering every case the listing query has
/// to get right:
///
/// | file | statistics row      | must be         |
/// |------|---------------------|-----------------|
/// | 1    | min `0`, max `10`   | kept (matches)  |
/// | 2    | min `100`, max `200`| pruned          |
/// | 3    | min `not-a-number`  | kept, fail-open |
/// | 4    | none at all         | kept, LEFT JOIN |
/// | 5    | min/max NULL        | kept, fail-open |
///
/// File 2 sits between two matching files on purpose: it is what makes the
/// keyset-pagination test able to fail.
async fn init_filter_fixture(pool: &PgPool) -> anyhow::Result<()> {
    for statement in [
        "CREATE TABLE ducklake_snapshot (snapshot_id BIGINT PRIMARY KEY, snapshot_time TIMESTAMP)",
        "CREATE TABLE ducklake_data_file (
             data_file_id BIGINT PRIMARY KEY, table_id BIGINT NOT NULL, path VARCHAR NOT NULL,
             path_is_relative BOOLEAN NOT NULL, file_size_bytes BIGINT NOT NULL,
             footer_size BIGINT, encryption_key VARCHAR, record_count BIGINT,
             row_id_start BIGINT, begin_snapshot BIGINT NOT NULL, end_snapshot BIGINT,
             partial_max BIGINT, partition_id BIGINT, mapping_id BIGINT)",
        "CREATE TABLE ducklake_delete_file (
             delete_file_id BIGINT PRIMARY KEY, data_file_id BIGINT NOT NULL,
             table_id BIGINT NOT NULL, path VARCHAR NOT NULL, path_is_relative BOOLEAN NOT NULL,
             file_size_bytes BIGINT NOT NULL, footer_size BIGINT, delete_count BIGINT,
             begin_snapshot BIGINT NOT NULL, end_snapshot BIGINT, encryption_key VARCHAR,
             partial_max BIGINT)",
        "CREATE TABLE ducklake_file_column_stats (
             data_file_id BIGINT NOT NULL, table_id BIGINT NOT NULL, column_id BIGINT NOT NULL,
             column_size_bytes BIGINT, value_count BIGINT, null_count BIGINT,
             min_value VARCHAR, max_value VARCHAR, contains_nan BOOLEAN)",
        "CREATE TABLE ducklake_file_partition_value (
             data_file_id BIGINT NOT NULL, table_id BIGINT NOT NULL,
             partition_key_index BIGINT NOT NULL, partition_value VARCHAR)",
        "CREATE TABLE ducklake_schema_versions (
             begin_snapshot BIGINT NOT NULL, schema_version BIGINT NOT NULL,
             table_id BIGINT NOT NULL)",
        "INSERT INTO ducklake_snapshot VALUES (1, NOW())",
        "INSERT INTO ducklake_data_file
             (data_file_id, table_id, path, path_is_relative, file_size_bytes, footer_size,
              record_count, row_id_start, begin_snapshot)
         VALUES (1, 1, 'f1.parquet', true, 10, 1, 100, 0, 1),
                (2, 1, 'f2.parquet', true, 10, 1, 100, 100, 1),
                (3, 1, 'f3.parquet', true, 10, 1, 100, 200, 1),
                (4, 1, 'f4.parquet', true, 10, 1, 100, 300, 1),
                (5, 1, 'f5.parquet', true, 10, 1, 100, 400, 1)",
        // File 3's bounds are text no numeric parser accepts — what a foreign
        // writer, or a corrupted row, leaves behind.
        "INSERT INTO ducklake_file_column_stats
             (data_file_id, table_id, column_id, column_size_bytes, value_count, null_count,
              min_value, max_value, contains_nan)
         VALUES (1, 1, 7, 8, 100, 0, '0', '10', NULL),
                (2, 1, 7, 8, 100, 0, '100', '200', NULL),
                (3, 1, 7, 8, 100, 0, 'not-a-number', 'also-not-a-number', NULL),
                (5, 1, 7, 8, 100, 0, NULL, NULL, NULL),
                (1, 1, 9, 8, 100, 0, '2020-01-01 00:00:00', '2020-06-01 00:00:00', NULL),
                (2, 1, 9, 8, 100, 0, '2021-01-01 00:00:00', '2021-06-01 00:00:00', NULL),
                (3, 1, 9, 8, 100, 0, 'not-a-timestamp', 'also-not-a-timestamp', NULL)",
    ] {
        sqlx::query(statement).execute(pool).await?;
    }
    Ok(())
}

/// Lower `predicate` against the fixture's schema. An Arrow field index maps
/// to the `column_id` `ducklake_file_column_stats` keys on by position, so the
/// two lists must stay in step.
fn fixture_filter(predicate: Arc<dyn PhysicalExpr>) -> StatsFilter {
    let schema = Schema::new(vec![
        Field::new("a", DataType::Int32, true),
        Field::new(
            "t",
            DataType::Timestamp(arrow::datatypes::TimeUnit::Microsecond, None),
            true,
        ),
    ]);
    let columns = [
        DuckLakeTableColumn::new(7, "a".to_string(), "int32".to_string(), true),
        DuckLakeTableColumn::new(9, "t".to_string(), "timestamp".to_string(), true),
    ];
    lower_predicate(&predicate, &schema, &columns).expect("predicate lowers")
}

/// `a > 5 AND a < 10`, lowered for pushdown.
fn int_range_filter() -> StatsFilter {
    let a = Arc::new(PhysColumn::new("a", 0)) as Arc<dyn PhysicalExpr>;
    fixture_filter(Arc::new(BinaryExpr::new(
        Arc::new(BinaryExpr::new(Arc::clone(&a), Operator::Gt, lit(5i32))),
        Operator::And,
        Arc::new(BinaryExpr::new(a, Operator::Lt, lit(10i32))),
    )))
}

/// `t >= 2021-01-01`, lowered for pushdown.
fn timestamp_filter() -> StatsFilter {
    let t = Arc::new(PhysColumn::new("t", 1)) as Arc<dyn PhysicalExpr>;
    fixture_filter(Arc::new(BinaryExpr::new(
        t,
        Operator::GtEq,
        lit(datafusion::common::ScalarValue::TimestampMicrosecond(
            Some(1_609_459_200_000_000),
            None,
        )),
    )))
}

type FilterFixture = (
    PostgresMetadataProvider,
    testcontainers::ContainerAsync<Postgres>,
);

async fn filter_fixture_provider() -> anyhow::Result<FilterFixture> {
    filter_fixture_on(Postgres::default().start().await?).await
}

/// The same fixture pinned to PostgreSQL 16, where the dialect validates a stat
/// with `pg_input_is_valid` instead of the pattern the default container falls
/// back to. Pinned rather than left to the testcontainers default because that
/// default is older than 16, so nothing else covers this path.
async fn filter_fixture_provider_pg16() -> anyhow::Result<FilterFixture> {
    filter_fixture_on(Postgres::default().with_tag("16-alpine").start().await?).await
}

async fn filter_fixture_on(
    container: testcontainers::ContainerAsync<Postgres>,
) -> anyhow::Result<FilterFixture> {
    let port = container.get_host_port_ipv4(5432).await?;
    let conn_str = format!("postgresql://postgres:postgres@127.0.0.1:{port}/postgres");
    let provider = PostgresMetadataProvider::new(&conn_str).await?;
    init_filter_fixture(&provider.pool).await?;
    Ok((provider, container))
}

fn listed_ids(files: &[datafusion_ducklake::metadata_provider::DuckLakeFileMetadata]) -> Vec<i64> {
    files.iter().map(|file| file.file.data_file_id).collect()
}

/// A file whose recorded range cannot hold a matching row never reaches the
/// caller, and every file whose statistics are absent, incomplete or
/// unreadable does.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn test_file_listing_filter_prunes_only_on_provable_statistics() {
    let (provider, _container) = filter_fixture_provider().await.unwrap();
    let filter = int_range_filter();

    let unfiltered = provider
        .get_table_file_metadata_page(1, 1, None, 4096)
        .expect("unfiltered listing");
    assert_eq!(listed_ids(&unfiltered), vec![1, 2, 3, 4, 5]);

    let filtered = provider
        .get_table_file_metadata_page_filtered(1, 1, None, 4096, Some(&filter))
        .expect("filtered listing");
    assert_eq!(
        listed_ids(&filtered),
        vec![1, 3, 4, 5],
        "only file 2, whose range 100..200 excludes 5 < a < 10, may be pruned"
    );

    // The statistics of the files that survived still come back, so the
    // in-memory pruning this pre-filters for has what it needs.
    let first = filtered.first().expect("file 1 survives");
    assert_eq!(
        first.column_statistics.len(),
        2,
        "both columns' stats come back"
    );
    assert_eq!(first.column_statistics[0].column_id, 7);
    assert_eq!(first.column_statistics[0].min_value.as_deref(), Some("0"));
}

/// A `min_value` no numeric parser accepts must neither abort the listing nor
/// prune the file. PostgreSQL raises on `CAST('not-a-number' AS numeric)`, so
/// the query is built to test the text before casting it; the comparison then
/// yields NULL, and a NULL comparison keeps the file rather than dropping it.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn test_file_listing_filter_keeps_a_file_with_a_malformed_min_value() {
    let (provider, _container) = filter_fixture_provider().await.unwrap();

    // The bare cast the filter must never emit: proof that this row is what
    // would take the whole listing query down.
    let bare_cast: Result<i64, sqlx::Error> = sqlx::query_scalar(
        "SELECT count(*) FROM ducklake_file_column_stats WHERE CAST(min_value AS numeric) > 0",
    )
    .fetch_one(&provider.pool)
    .await;
    assert!(
        bare_cast.is_err(),
        "the fixture must contain a stat a plain CAST cannot parse"
    );

    let filtered = provider
        .get_table_file_metadata_page_filtered(1, 1, None, 4096, Some(&int_range_filter()))
        .expect("a malformed stat must not fail the listing query");
    assert!(
        listed_ids(&filtered).contains(&3),
        "file 3's unreadable bounds prove nothing, so it must be kept: {:?}",
        listed_ids(&filtered)
    );
}

/// The filter is applied inside the query, ahead of `LIMIT`, so the keyset
/// cursor still walks every matching file. Applied after the fetch, the page
/// landing on the pruned file 2 would come back empty and iteration would stop
/// there, silently hiding files 3, 4 and 5.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn test_file_listing_filter_paginates_past_pruned_files() {
    let (provider, _container) = filter_fixture_provider().await.unwrap();
    let filter = int_range_filter();

    let mut seen = Vec::new();
    let mut cursor = None;
    loop {
        let page = provider
            .get_table_file_metadata_page_filtered(1, 1, cursor, 1, Some(&filter))
            .expect("filtered page");
        if page.is_empty() {
            break;
        }
        seen.extend(listed_ids(&page));
        cursor = seen.last().copied();
    }
    assert_eq!(seen, vec![1, 3, 4, 5]);
}

/// A catalog with no `ducklake_file_column_stats` at all cannot be joined
/// against it. Listing every file is always correct, so the filter is dropped
/// and the scan proceeds instead of failing.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn test_file_listing_filter_falls_open_on_a_catalog_without_statistics() {
    let (provider, _container) = filter_fixture_provider().await.unwrap();
    sqlx::query("DROP TABLE ducklake_file_column_stats")
        .execute(&provider.pool)
        .await
        .expect("drop the statistics table");

    let filtered = provider
        .get_table_file_metadata_page_filtered(1, 1, None, 4096, Some(&int_range_filter()))
        .expect("a legacy catalog must still list its files");
    assert_eq!(listed_ids(&filtered), vec![1, 2, 3, 4, 5]);
}

/// On PostgreSQL 16 and later the dialect validates a stat with
/// `pg_input_is_valid`, which is exact where the older servers' pattern is
/// conservative. Pruning, fail-open on a malformed stat and the no-stats-row
/// LEFT JOIN must all behave identically, and a temporal comparison — which no
/// pattern can validate, so it pushes down nothing on an older server — must
/// prune here.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn test_file_listing_filter_uses_soft_input_validation_on_postgresql_16() {
    let (provider, _container) = filter_fixture_provider_pg16().await.unwrap();
    let soft_input_validation: bool =
        sqlx::query_scalar("SELECT to_regprocedure('pg_input_is_valid(text,text)') IS NOT NULL")
            .fetch_one(&provider.pool)
            .await
            .expect("probe the server");
    assert!(
        soft_input_validation,
        "this test only means anything on PostgreSQL 16 or later"
    );

    let filtered = provider
        .get_table_file_metadata_page_filtered(1, 1, None, 4096, Some(&int_range_filter()))
        .expect("filtered listing");
    assert_eq!(listed_ids(&filtered), vec![1, 3, 4, 5]);

    // `t >= 2021-01-01`: file 1's range ends in 2020-06 and is the only one
    // proven not to match. File 3's bounds are not timestamps, file 4 has no
    // stats row and file 5 has none for this column, so all three are kept.
    let filtered = provider
        .get_table_file_metadata_page_filtered(1, 1, None, 4096, Some(&timestamp_filter()))
        .expect("filtered listing");
    assert_eq!(listed_ids(&filtered), vec![2, 3, 4, 5]);
}

/// The server version does not change which files a temporal filter keeps.
///
/// Nothing is cast, so nothing needs `pg_input_is_valid` — which exists only on
/// PostgreSQL 16 and later, and which a pre-16 server used to lack any
/// substitute for, leaving a timestamp predicate to prune nothing at all. The
/// comparison is made on the encoded text, which every version orders the same
/// way, so this asserts the identical result to the 16+ case above.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn test_temporal_filter_prunes_the_same_without_soft_input_validation() {
    let (provider, _container) = filter_fixture_provider().await.unwrap();
    let soft_input_validation: bool =
        sqlx::query_scalar("SELECT to_regprocedure('pg_input_is_valid(text,text)') IS NOT NULL")
            .fetch_one(&provider.pool)
            .await
            .expect("probe the server");
    assert!(
        !soft_input_validation,
        "this fixture is meant to run on a pre-16 server; the 16+ path is covered above"
    );
    let filtered = provider
        .get_table_file_metadata_page_filtered(1, 1, None, 4096, Some(&timestamp_filter()))
        .expect("filtered listing");
    assert_eq!(listed_ids(&filtered), vec![2, 3, 4, 5]);
}

/// A temporal constant is never resolved by PostgreSQL, so one its input
/// function refuses cannot abort the listing.
///
/// A cast would resolve it: the constant is spliced in as a literal on the other
/// side of the comparison, and the server reads it against the cast target
/// before any row is examined, so an unreadable one turns a scan that planned
/// fine without pushdown into a hard failure. Comparing the encoded text instead
/// removes that failure mode entirely — and it makes year zero, which
/// PostgreSQL's calendar does not have, an ordinary comparison rather than
/// something to decline.
///
/// What still declines is an encoding whose bytes do not order chronologically.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn test_out_of_range_temporal_constant_does_not_fail_the_listing() {
    let (provider, _container) = filter_fixture_provider_pg16().await.unwrap();

    let listing_for = |micros: i64| {
        let constant = datafusion::common::ScalarValue::TimestampMicrosecond(Some(micros), None);
        let filter = fixture_filter(Arc::new(BinaryExpr::new(
            Arc::new(PhysColumn::new("t", 1)) as Arc<dyn PhysicalExpr>,
            Operator::Lt,
            lit(constant),
        )));
        provider
            .get_table_file_metadata_page_filtered(1, 1, None, 4096, Some(&filter))
            .expect("a constant PostgreSQL cannot read must not fail the listing")
    };

    // Year zero: `0000-01-01 00:00:00`. Four digits and no sign, so it orders as
    // text like any other timestamp — but PostgreSQL's calendar runs 1 BC
    // straight into 1 AD, so casting it raises `date/time field value out of
    // range`. Proof that this is a constant PostgreSQL refuses, and that it
    // refuses it while *parsing*: `WHERE false AND ...` reads no row and the
    // statement still fails.
    let year_zero = -62_167_219_200_000_000i64;
    let encoded = datafusion_ducklake::stats_encode::encode_scalar(
        &datafusion::common::ScalarValue::TimestampMicrosecond(Some(year_zero), None),
    )
    .expect("the constant has a canonical encoding");
    let refused: Result<i64, sqlx::Error> = sqlx::query_scalar(sqlx::AssertSqlSafe(format!(
        "SELECT count(*) FROM ducklake_file_column_stats
         WHERE false AND CAST(min_value AS timestamp) < '{encoded}'"
    )))
    .fetch_one(&provider.pool)
    .await;
    assert!(
        refused.is_err(),
        "PostgreSQL must refuse the constant {encoded} in a cast, or this test proves nothing"
    );

    // Compared as text it prunes normally: files 1 and 2 carry real bounds that
    // cannot be below year zero, and 3 to 5 have no usable statistics, so the
    // guards keep them.
    assert_eq!(
        listed_ids(&listing_for(year_zero)),
        vec![3, 4, 5],
        "year zero is a comparison like any other once nothing is cast"
    );

    // A year past 9999, which chrono renders with an explicit `+`. That sorts
    // below every digit, so as text it would order *before* every ordinary
    // timestamp — the comparison is declined and prunes nothing.
    assert_eq!(
        listed_ids(&listing_for(400_000_000_000_000_000i64)),
        vec![1, 2, 3, 4, 5],
        "a sign-prefixed year does not order as text, so it is declined"
    );
}
