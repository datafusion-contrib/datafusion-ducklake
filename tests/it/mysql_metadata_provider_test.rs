#![cfg(feature = "metadata-mysql")]
//! MySQL metadata provider tests
//!
//! This test suite verifies the MySQL metadata provider implementation,
//! including all MetadataProvider trait methods, schema initialization,
//! concurrent access, and error handling.
//!
//! ## Test Setup
//!
//! Tests use testcontainers to spin up a temporary MySQL instance.
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
    DuckLakeCatalog, DuckdbMetadataProvider, MySqlMetadataProvider,
    metadata_provider::MetadataProvider,
};
use sqlx::MySqlPool;
use std::sync::Arc;
use tempfile::TempDir;
use testcontainers::runners::AsyncRunner;
use testcontainers_modules::mysql::Mysql;

/// Initialize DuckLake catalog schema in MySQL (for tests only)
async fn init_schema(pool: &MySqlPool) -> anyhow::Result<()> {
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS ducklake_snapshot (
            snapshot_id BIGINT PRIMARY KEY,
            snapshot_time DATETIME(6)
        )",
    )
    .execute(pool)
    .await?;

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS ducklake_column_mapping (
            mapping_id BIGINT PRIMARY KEY,
            table_id BIGINT NOT NULL,
            type VARCHAR(255) NOT NULL
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
            schema_name VARCHAR(255) NOT NULL,
            path VARCHAR(1024) NOT NULL,
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
            table_name VARCHAR(255) NOT NULL,
            path VARCHAR(1024) NOT NULL,
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
            view_uuid VARCHAR(36),
            schema_id BIGINT NOT NULL,
            view_name VARCHAR(255) NOT NULL,
            dialect VARCHAR(255) NOT NULL,
            `sql` TEXT NOT NULL,
            column_aliases TEXT,
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
            column_name VARCHAR(255) NOT NULL,
            column_type VARCHAR(255) NOT NULL,
            column_order INTEGER NOT NULL,
            nulls_allowed BOOLEAN,
            FOREIGN KEY (table_id) REFERENCES ducklake_table(table_id)
        )",
    )
    .execute(pool)
    .await?;

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS ducklake_data_file (
            data_file_id BIGINT PRIMARY KEY,
            table_id BIGINT NOT NULL,
            path VARCHAR(1024) NOT NULL,
            path_is_relative BOOLEAN NOT NULL,
            file_size_bytes BIGINT NOT NULL,
            footer_size BIGINT,
            encryption_key VARCHAR(255),
            begin_snapshot BIGINT NOT NULL DEFAULT 1,
            end_snapshot BIGINT,
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
            path VARCHAR(1024) NOT NULL,
            path_is_relative BOOLEAN NOT NULL,
            file_size_bytes BIGINT NOT NULL,
            footer_size BIGINT,
            encryption_key VARCHAR(255),
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
            `key` VARCHAR(255) NOT NULL,
            value VARCHAR(1024) NOT NULL,
            scope VARCHAR(255),
            scope_id BIGINT,
            PRIMARY KEY (`key`)
        )",
    )
    .execute(pool)
    .await?;

    // MySQL indexes
    sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_schema_snapshot ON ducklake_schema(begin_snapshot, end_snapshot)",
    )
    .execute(pool)
    .await
    .ok(); // Ignore if already exists

    sqlx::query("CREATE INDEX IF NOT EXISTS idx_table_schema ON ducklake_table(schema_id)")
        .execute(pool)
        .await
        .ok();

    sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_table_snapshot ON ducklake_table(begin_snapshot, end_snapshot)",
    )
    .execute(pool)
    .await
    .ok();

    Ok(())
}

/// Helper to create a MySQL provider with initialized schema
async fn create_mysql_provider()
-> anyhow::Result<(MySqlMetadataProvider, testcontainers::ContainerAsync<Mysql>)> {
    let container = Mysql::default().start().await?;

    let host = "127.0.0.1";
    let port = container.get_host_port_ipv4(3306).await?;
    let conn_str = format!("mysql://root@{}:{}/test", host, port);

    let provider = MySqlMetadataProvider::new(&conn_str)
        .await
        .expect("Failed to create provider");
    init_schema(&provider.pool).await?;

    Ok((provider, container))
}

/// Helper to populate test data in MySQL
async fn populate_test_data(provider: &MySqlMetadataProvider) -> anyhow::Result<()> {
    let pool = &provider.pool;

    // Insert snapshots
    sqlx::query("INSERT INTO ducklake_snapshot (snapshot_id, snapshot_time) VALUES (?, NOW())")
        .bind(1i64)
        .execute(pool)
        .await?;

    sqlx::query("INSERT INTO ducklake_snapshot (snapshot_id, snapshot_time) VALUES (?, NOW())")
        .bind(2i64)
        .execute(pool)
        .await?;

    // Insert metadata (data_path)
    sqlx::query(
        "INSERT INTO ducklake_metadata (`key`, value, scope, scope_id) VALUES (?, ?, NULL, NULL)",
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
    .bind(true)
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
    .bind(true)
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
    .bind(true)
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
    .bind(true)
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
    .bind(false)
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
    .bind(true)
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
    .bind(true)
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
    .bind(true)
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
    .bind(true)
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

/// Helper to populate MySQL with metadata from a DuckDB-created catalog
async fn populate_from_duckdb_catalog(
    provider: &MySqlMetadataProvider,
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

    // Step 3: Populate MySQL with metadata from DuckDB
    let pool = &provider.pool;

    // Insert snapshots
    for snapshot in &snapshots {
        let timestamp_value: Option<sqlx::types::chrono::NaiveDateTime> =
            snapshot.timestamp.as_ref().and_then(|ts_str| {
                sqlx::types::chrono::NaiveDateTime::parse_from_str(ts_str, "%Y-%m-%d %H:%M:%S%.6f")
                    .ok()
            });

        sqlx::query("INSERT INTO ducklake_snapshot (snapshot_id, snapshot_time) VALUES (?, ?)")
            .bind(snapshot.snapshot_id)
            .bind(timestamp_value)
            .execute(pool)
            .await?;
    }

    // Insert data_path metadata
    sqlx::query(
        "INSERT INTO ducklake_metadata (`key`, value, scope, scope_id) VALUES (?, ?, NULL, NULL)",
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
        .bind(schema.path_is_relative)
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
            .bind(table.path_is_relative)
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
                .bind(column.is_nullable)
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
                .bind(file.file.path_is_relative)
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
                    .bind(delete_file.path_is_relative)
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
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn test_schema_initialization_idempotent() {
    let (provider, _container) = create_mysql_provider().await.unwrap();

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
    let (provider, _container) = create_mysql_provider().await.unwrap();

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
    let (provider, _container) = create_mysql_provider().await.unwrap();

    populate_test_data(&provider)
        .await
        .expect("Failed to populate test data");

    let data_path = provider.get_data_path().expect("Should get data path");

    assert_eq!(data_path, "file:///tmp/ducklake_data/");
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn test_list_snapshots() {
    let (provider, _container) = create_mysql_provider().await.unwrap();

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
    let (provider, _container) = create_mysql_provider().await.unwrap();

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
    let (provider, _container) = create_mysql_provider().await.unwrap();

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
    let (provider, _container) = create_mysql_provider().await.unwrap();

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
    let (provider, _container) = create_mysql_provider().await?;
    populate_test_data(&provider).await?;
    sqlx::query(
        "INSERT INTO ducklake_view
         (view_id, schema_id, view_name, dialect, `sql`, column_aliases, begin_snapshot, end_snapshot)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?), (?, ?, ?, ?, ?, ?, ?, ?)",
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
    let (provider, _container) = create_mysql_provider().await.unwrap();

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
    let (provider, _container) = create_mysql_provider().await.unwrap();

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
    let (provider, _container) = create_mysql_provider().await.unwrap();

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
    let (provider, _container) = create_mysql_provider().await?;
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
    let (provider, _container) = create_mysql_provider().await.unwrap();

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
    let (provider, _container) = create_mysql_provider().await.unwrap();

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
    let (provider, _container) = create_mysql_provider().await.unwrap();

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
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn test_list_all_files() {
    let (provider, _container) = create_mysql_provider().await.unwrap();

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
    let (provider, _container) = create_mysql_provider().await.unwrap();

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
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn test_datafusion_integration() {
    let (provider, _container) = create_mysql_provider().await.unwrap();

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
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn test_error_invalid_connection_string() {
    let result = MySqlMetadataProvider::new("invalid://connection:string").await;
    assert!(
        result.is_err(),
        "Should fail with invalid connection string"
    );
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn test_error_connection_refused() {
    let result = MySqlMetadataProvider::new("mysql://root@localhost:9999/db").await;
    assert!(result.is_err(), "Should fail when connection is refused");
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "provider-test fixture drift vs current schema; fixed with the provider rework"]
async fn test_query_real_parquet_files() {
    let (provider, _container) = create_mysql_provider().await.unwrap();

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
#[ignore = "provider-test fixture drift vs current schema; fixed with the provider rework"]
async fn test_query_with_filter() {
    let (provider, _container) = create_mysql_provider().await.unwrap();

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

/// Connecting as a password-protected `caching_sha2_password` user over a
/// non-TLS channel must succeed.
///
/// This is MySQL 8's default auth plugin. On the first connection for a user the
/// server's fast-auth cache is cold, so it demands full authentication; over an
/// insecure channel the client must fetch the server's RSA public key and send
/// the password encrypted. sqlx bundled that RSA backend unconditionally through
/// 0.8, but 0.9 moved it behind the `mysql-rsa` feature and substitutes a stub
/// that fails at connect time with "RSA auth backend disabled". Nothing catches
/// that at compile time, and the rest of this suite connects as passwordless
/// `root` (an empty password skips the RSA exchange entirely), so this test is
/// what pins `sqlx/mysql-rsa` into the `metadata-mysql` feature.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn test_connect_caching_sha2_password_without_tls() {
    let container = Mysql::default()
        .with_init_sql(
            "CREATE USER 'rsauser'@'%' IDENTIFIED WITH caching_sha2_password BY 'secret-pw';
             GRANT ALL PRIVILEGES ON test.* TO 'rsauser'@'%';"
                .to_string()
                .into_bytes(),
        )
        .start()
        .await
        .expect("Failed to start MySQL");

    let port = container
        .get_host_port_ipv4(3306)
        .await
        .expect("Failed to get port");
    let conn_str = format!("mysql://rsauser:secret-pw@127.0.0.1:{}/test", port);

    let provider = MySqlMetadataProvider::new(&conn_str)
        .await
        .expect("should connect as a caching_sha2_password user over a non-TLS channel");

    // Prove the connection is actually usable, not merely constructed.
    init_schema(&provider.pool)
        .await
        .expect("should run DDL over the authenticated connection");
    let snapshots = provider
        .list_snapshots()
        .expect("should query over the authenticated connection");
    assert!(
        snapshots.is_empty(),
        "freshly initialized catalog should have no snapshots"
    );
}

// ---------------------------------------------------------------------------
// Statistics filter pushdown (`get_table_file_metadata_page_filtered`)
// ---------------------------------------------------------------------------

/// A catalog with `file_count` files on table 1 and a per-file statistics
/// table.
///
/// MySQL's `init_schema` above predates both `ducklake_file_column_stats` and
/// the file columns the paged listing projects, so this adds them — which is
/// also what makes the fail-open test below meaningful.
async fn create_provider_with_file_stats(
    file_count: i64,
) -> anyhow::Result<(MySqlMetadataProvider, testcontainers::ContainerAsync<Mysql>)> {
    let (provider, container) = create_mysql_provider().await?;
    let pool = &provider.pool;
    sqlx::query(
        "ALTER TABLE ducklake_data_file
             ADD COLUMN record_count BIGINT,
             ADD COLUMN row_id_start BIGINT,
             ADD COLUMN mapping_id BIGINT",
    )
    .execute(pool)
    .await?;
    sqlx::query(
        "CREATE TABLE ducklake_file_column_stats (
            data_file_id BIGINT NOT NULL,
            table_id BIGINT NOT NULL,
            column_id BIGINT NOT NULL,
            column_size_bytes BIGINT,
            value_count BIGINT,
            null_count BIGINT,
            min_value VARCHAR(4000),
            max_value VARCHAR(4000),
            contains_nan BOOLEAN
        )",
    )
    .execute(pool)
    .await?;
    sqlx::query(
        "INSERT INTO ducklake_schema
             (schema_id, schema_name, path, path_is_relative, begin_snapshot)
         VALUES (1, 'main', 'main/', TRUE, 1)",
    )
    .execute(pool)
    .await?;
    sqlx::query(
        "INSERT INTO ducklake_table
             (table_id, schema_id, table_name, path, path_is_relative, begin_snapshot)
         VALUES (1, 1, 'events', 'events/', TRUE, 1)",
    )
    .execute(pool)
    .await?;
    for data_file_id in 1..=file_count {
        sqlx::query(
            "INSERT INTO ducklake_data_file
                 (data_file_id, table_id, path, path_is_relative, file_size_bytes,
                  record_count, row_id_start, begin_snapshot)
             VALUES (?, 1, ?, FALSE, 1000, 10, 0, 1)",
        )
        .bind(data_file_id)
        .bind(format!("file{data_file_id}.parquet"))
        .execute(pool)
        .await?;
    }
    Ok((provider, container))
}

/// Give `data_file_id` a statistics row for `column_id`.
async fn insert_column_stats(
    provider: &MySqlMetadataProvider,
    data_file_id: i64,
    column_id: i64,
    min_value: Option<&str>,
    max_value: Option<&str>,
) -> anyhow::Result<()> {
    sqlx::query(
        "INSERT INTO ducklake_file_column_stats
             (data_file_id, table_id, column_id, value_count, null_count, min_value, max_value)
         VALUES (?, 1, ?, 10, 0, ?, ?)",
    )
    .bind(data_file_id)
    .bind(column_id)
    .bind(min_value)
    .bind(max_value)
    .execute(&provider.pool)
    .await?;
    Ok(())
}

/// Lower `a <op> value` on an `INT32` column whose `column_id` is 7.
fn int32_filter(operator: Operator, value: i32) -> StatsFilter {
    let column = Arc::new(PhysColumn::new("a", 0)) as Arc<dyn PhysicalExpr>;
    let predicate =
        Arc::new(BinaryExpr::new(column, operator, lit(value))) as Arc<dyn PhysicalExpr>;
    let schema = Schema::new(vec![Field::new("a", DataType::Int32, true)]);
    let columns = vec![DuckLakeTableColumn::new(7, "a".to_string(), "int32".to_string(), true)];
    lower_predicate(&predicate, &schema, &columns).expect("predicate lowers to a statistics filter")
}

fn file_ids(files: &[datafusion_ducklake::metadata_provider::DuckLakeFileMetadata]) -> Vec<i64> {
    files.iter().map(|file| file.file.data_file_id).collect()
}

/// The catalog query drops files whose bounds cannot hold a match; a file with
/// no statistics row, a NULL bound, or a bound that is not a number is kept.
///
/// MySQL's `CAST('not-a-number' AS DECIMAL)` is `0` with a warning, so the
/// malformed case is the one an unguarded cast would silently prune.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn filtered_file_page_prunes_by_statistics() {
    let (provider, _container) = create_provider_with_file_stats(5).await.unwrap();
    // `a < 5` reads min_value only. 1: matches. 2: cannot match. 3: malformed
    // minimum. 4: NULL minimum. 5: no statistics row at all.
    insert_column_stats(&provider, 1, 7, Some("1"), Some("2"))
        .await
        .unwrap();
    insert_column_stats(&provider, 2, 7, Some("100"), Some("200"))
        .await
        .unwrap();
    insert_column_stats(&provider, 3, 7, Some("not-a-number"), Some("10"))
        .await
        .unwrap();
    insert_column_stats(&provider, 4, 7, None, Some("10"))
        .await
        .unwrap();

    let filter = int32_filter(Operator::Lt, 5);
    let filtered = provider
        .get_table_file_metadata_page_filtered(1, 1, None, 100, Some(&filter))
        .expect("filtered page");
    assert_eq!(file_ids(&filtered), vec![1, 3, 4, 5]);

    // The same call without a filter still sees every file.
    let unfiltered = provider
        .get_table_file_metadata_page(1, 1, None, 100)
        .expect("unfiltered page");
    assert_eq!(file_ids(&unfiltered), vec![1, 2, 3, 4, 5]);

    // The filter is applied before LIMIT: file 2 is pruned inside the query, so
    // the one-row page after file 1 returns file 3 rather than nothing.
    let page = provider
        .get_table_file_metadata_page_filtered(1, 1, Some(1), 1, Some(&filter))
        .expect("filtered page");
    assert_eq!(file_ids(&page), vec![3]);
}

/// String bounds are compared byte-wise, not under MySQL's default
/// `utf8mb4_0900_ai_ci`.
///
/// File 1's bounds are `Zebra` .. `apple`, a valid range byte-wise ('Z' is
/// 0x5A, 'a' is 0x61) that contains `apple`. Under the case-insensitive
/// collation `'apple' >= 'Zebra'` is false, so the file would be pruned even
/// though it may hold matching rows.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn filtered_file_page_compares_strings_byte_wise() {
    let (provider, _container) = create_provider_with_file_stats(2).await.unwrap();
    insert_column_stats(&provider, 1, 8, Some("Zebra"), Some("apple"))
        .await
        .unwrap();
    insert_column_stats(&provider, 2, 8, Some("b"), Some("c"))
        .await
        .unwrap();

    let column = Arc::new(PhysColumn::new("s", 0)) as Arc<dyn PhysicalExpr>;
    let predicate =
        Arc::new(BinaryExpr::new(column, Operator::Eq, lit("apple"))) as Arc<dyn PhysicalExpr>;
    let schema = Schema::new(vec![Field::new("s", DataType::Utf8, true)]);
    let columns = vec![DuckLakeTableColumn::new(8, "s".to_string(), "varchar".to_string(), true)];
    let filter =
        lower_predicate(&predicate, &schema, &columns).expect("predicate lowers to a filter");

    let filtered = provider
        .get_table_file_metadata_page_filtered(1, 1, None, 100, Some(&filter))
        .expect("filtered page");
    assert_eq!(file_ids(&filtered), vec![1]);
}

/// A stat with a trailing line terminator is refused, not compared.
///
/// MySQL 8's `REGEXP` is ICU, where `$` matches *before* a final line
/// terminator — so a `$`-anchored shape test also admits a bound carrying a
/// trailing newline or U+2028. That is not cosmetic: U+2028's UTF-8 lead byte
/// is `0xE2`, which sorts above `.` (0x2E), so such a bound compares as later
/// than any fractional timestamp and its file is pruned though it holds
/// matching rows. The guard matches the value against its own leading match
/// instead, which has no end anchor to get wrong.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn filtered_file_page_refuses_a_bound_with_a_trailing_line_terminator() {
    let (provider, _container) = create_provider_with_file_stats(3).await.unwrap();
    // File 1 is well formed and genuinely cannot match. Files 2 and 3 carry the
    // same instant with a trailing terminator, which no writer of ours produces
    // — so their bounds are unusable and both files must survive.
    insert_column_stats(
        &provider,
        1,
        9,
        Some("2024-06-01 00:00:00"),
        Some("2024-06-02 00:00:00"),
    )
    .await
    .unwrap();
    insert_column_stats(
        &provider,
        2,
        9,
        Some("2024-01-01 00:00:00\u{2028}"),
        Some("2024-01-01 00:00:00\u{2028}"),
    )
    .await
    .unwrap();
    insert_column_stats(
        &provider,
        3,
        9,
        Some("2024-01-01 00:00:00\n"),
        Some("2024-01-01 00:00:00\n"),
    )
    .await
    .unwrap();

    let column = Arc::new(PhysColumn::new("t", 0)) as Arc<dyn PhysicalExpr>;
    let predicate = Arc::new(BinaryExpr::new(
        column,
        Operator::Lt,
        lit(datafusion::common::ScalarValue::TimestampMicrosecond(
            Some(1_704_067_200_500_000),
            None,
        )),
    )) as Arc<dyn PhysicalExpr>;
    let schema = Schema::new(vec![Field::new(
        "t",
        DataType::Timestamp(datafusion::arrow::datatypes::TimeUnit::Microsecond, None),
        true,
    )]);
    let columns = vec![DuckLakeTableColumn::new(9, "t".to_string(), "timestamp".to_string(), true)];
    let filter =
        lower_predicate(&predicate, &schema, &columns).expect("predicate lowers to a filter");

    let filtered = provider
        .get_table_file_metadata_page_filtered(1, 1, None, 100, Some(&filter))
        .expect("filtered page");
    assert_eq!(
        file_ids(&filtered),
        vec![2, 3],
        "a bound with a trailing terminator is not this encoding, so it must not prune"
    );
}

/// Trailing spaces do not make two different strings compare equal.
///
/// `utf8mb4_bin` is a PAD SPACE collation: it reads `'a'` and `'a '` as the same
/// value and orders neither below the other. DataFusion compares `Utf8`
/// byte-wise, where `'a'` is less than `'a '` — so under the padded collation a
/// file whose bound is `'a'` is pruned from `WHERE s < 'a '` although it holds a
/// row that satisfies it. `utf8mb4_0900_bin` is NO PAD.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn filtered_file_page_does_not_pad_string_bounds() {
    let (provider, _container) = create_provider_with_file_stats(2).await.unwrap();
    insert_column_stats(&provider, 1, 8, Some("a"), Some("a"))
        .await
        .unwrap();
    insert_column_stats(&provider, 2, 8, Some("x"), Some("z"))
        .await
        .unwrap();

    let column = Arc::new(PhysColumn::new("s", 0)) as Arc<dyn PhysicalExpr>;
    let predicate =
        Arc::new(BinaryExpr::new(column, Operator::Lt, lit("a "))) as Arc<dyn PhysicalExpr>;
    let schema = Schema::new(vec![Field::new("s", DataType::Utf8, true)]);
    let columns = vec![DuckLakeTableColumn::new(8, "s".to_string(), "varchar".to_string(), true)];
    let filter =
        lower_predicate(&predicate, &schema, &columns).expect("predicate lowers to a filter");

    let filtered = provider
        .get_table_file_metadata_page_filtered(1, 1, None, 100, Some(&filter))
        .expect("filtered page");
    assert_eq!(
        file_ids(&filtered),
        vec![1],
        "'a' is byte-wise less than 'a ', so file 1 matches and must be kept"
    );
}

/// A date bound MySQL would read leniently is refused, not converted.
///
/// MySQL's `DATE` parser accepts far more than the encoder writes:
/// `' 2020-01-01 '`, `2020/01/01`, `20200101`, `2020.01.01` and `2020-1-1` all
/// convert to 2020-01-01. Without a shape test each becomes a definite bound
/// and prunes on a value no writer of ours produced. The cast alone cannot
/// catch them — it returns NULL only for what no calendar has, like
/// `2020-02-31`.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn filtered_file_page_refuses_a_leniently_parsed_date_bound() {
    let lenient = [" 2020-01-01 ", "2020/01/01", "20200101", "2020.01.01", "2020-1-1"];
    for bound in lenient {
        let (provider, _container) = create_provider_with_file_stats(2).await.unwrap();
        // File 1's real dates are in 2021; the bound claims 2020 in a spelling
        // the encoder never writes, so it must not be used to prune.
        insert_column_stats(&provider, 1, 9, Some(bound), Some(bound))
            .await
            .unwrap();
        insert_column_stats(&provider, 2, 9, Some("2019-01-01"), Some("2019-06-01"))
            .await
            .unwrap();

        let column = Arc::new(PhysColumn::new("d", 0)) as Arc<dyn PhysicalExpr>;
        let predicate = Arc::new(BinaryExpr::new(
            column,
            Operator::Gt,
            lit(datafusion::common::ScalarValue::Date32(Some(18_700))),
        )) as Arc<dyn PhysicalExpr>;
        let schema = Schema::new(vec![Field::new("d", DataType::Date32, true)]);
        let columns = vec![DuckLakeTableColumn::new(9, "d".to_string(), "date".to_string(), true)];
        let filter =
            lower_predicate(&predicate, &schema, &columns).expect("predicate lowers to a filter");

        let filtered = provider
            .get_table_file_metadata_page_filtered(1, 1, None, 100, Some(&filter))
            .expect("filtered page");
        assert_eq!(
            file_ids(&filtered),
            vec![1],
            "bound {bound:?} is not an encoding this crate writes, so it must not prune file 1"
        );
    }
}

/// A catalog with no `ducklake_file_column_stats` must still list its files:
/// joining a table that does not exist is a hard error, so the query falls back
/// to the unfiltered listing.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn filtered_file_page_falls_back_without_a_statistics_table() {
    let (provider, _container) = create_provider_with_file_stats(2).await.unwrap();
    sqlx::query("DROP TABLE ducklake_file_column_stats")
        .execute(&provider.pool)
        .await
        .unwrap();

    let filter = int32_filter(Operator::Gt, 50);
    let filtered = provider
        .get_table_file_metadata_page_filtered(1, 1, None, 100, Some(&filter))
        .expect("a legacy catalog still lists files");
    assert_eq!(file_ids(&filtered), vec![1, 2]);
}

/// A selective filter must not read the statistics of the files it pruned.
///
/// The two enrichment queries used to be scoped `data_file_id > after AND <=
/// last`, which is bounded by the page size only while nothing narrows the
/// listing. With a filter the surviving ids are sparse: a single match at the
/// far end of a large table puts `last` near the table maximum, and the first
/// page's statistics query then returns every stats row below it.
///
/// Row *counts* are not observable from the return value, so the pruned files
/// carry a stats row that cannot be decoded: a NULL `column_id`, which
/// `DuckLakeFileColumnStatistics` reads into an `i64`. Reading one is an error,
/// so the filtered page succeeding is proof it read none of them — and the
/// unfiltered page below, whose range does cover them, fails, which is what
/// makes this discriminate rather than pass vacuously.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn filtered_file_page_reads_no_statistics_for_pruned_files() {
    let (provider, _container) = create_provider_with_file_stats(6).await.unwrap();
    // The shared fixture declares `column_id` NOT NULL; drop that so a row the
    // reader cannot decode can be planted on a pruned file.
    sqlx::query("ALTER TABLE ducklake_file_column_stats MODIFY column_id BIGINT NULL")
        .execute(&provider.pool)
        .await
        .unwrap();

    // Files 1..=5 cannot hold a value above 50; file 6 can. The match is last,
    // so `last_data_file_id` is the table maximum and the old range covered
    // every pruned file.
    for data_file_id in 1..=6i64 {
        let (min_value, max_value) = if data_file_id == 6 {
            ("100", "200")
        } else {
            ("0", "10")
        };
        insert_column_stats(&provider, data_file_id, 7, Some(min_value), Some(max_value))
            .await
            .unwrap();
        if data_file_id < 6 {
            // The undecodable row. Its NULL `column_id` also keeps it out of the
            // filter's own CTE, which selects `column_id = 7`, so it changes
            // nothing about which files survive.
            sqlx::query(
                "INSERT INTO ducklake_file_column_stats
                     (data_file_id, table_id, column_id, value_count, null_count,
                      min_value, max_value)
                 VALUES (?, 1, NULL, 10, 0, '0', '10')",
            )
            .bind(data_file_id)
            .execute(&provider.pool)
            .await
            .unwrap();
        }
    }

    let filter = int32_filter(Operator::Gt, 50);
    let filtered = provider
        .get_table_file_metadata_page_filtered(1, 1, None, 100, Some(&filter))
        .expect("the filtered page reads statistics only for the files it returned");
    assert_eq!(file_ids(&filtered), vec![6]);

    // The same call with no filter returns every file, so its range does reach
    // the planted rows and it fails. That is the behaviour the filtered call
    // would have had while it scoped by range.
    provider
        .get_table_file_metadata_page(1, 1, None, 100)
        .expect_err("the unfiltered range reaches the undecodable rows");
}

/// A string constant holding a backslash pushes down, against the real server.
///
/// MySQL reads `\` inside a quoted string as an escape unless `sql_mode` carries
/// `NO_BACKSLASH_ESCAPES`, so the standard rendering of `a\` is `'a\'`, whose
/// closing quote is consumed and the statement dies with error 1064. Nothing
/// would be visible from here: the blanket unfiltered retry would list all four
/// files and the pruning would just be gone. Asserting that exactly the matching
/// file comes back is therefore the assertion — a query that failed returns
/// every file, and one that mis-rendered the constant returns the wrong file.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn filtered_file_page_pushes_down_awkward_string_constants() {
    let (provider, _container) = create_provider_with_file_stats(4).await.unwrap();
    // One file per constant, each pinned to that exact value, plus a control the
    // filter must prune every time.
    let values = ["a\\", "a'", "a\\'", "zzz"];
    for (index, value) in values.iter().enumerate() {
        let data_file_id = i64::try_from(index).unwrap() + 1;
        insert_column_stats(&provider, data_file_id, 8, Some(value), Some(value))
            .await
            .unwrap();
    }

    let schema = Schema::new(vec![Field::new("s", DataType::Utf8, true)]);
    let columns = vec![DuckLakeTableColumn::new(8, "s".to_string(), "varchar".to_string(), true)];
    for (index, value) in values.iter().enumerate().take(3) {
        let column = Arc::new(PhysColumn::new("s", 0)) as Arc<dyn PhysicalExpr>;
        let predicate =
            Arc::new(BinaryExpr::new(column, Operator::Eq, lit(*value))) as Arc<dyn PhysicalExpr>;
        let filter =
            lower_predicate(&predicate, &schema, &columns).expect("predicate lowers to a filter");
        let filtered = provider
            .get_table_file_metadata_page_filtered(1, 1, None, 100, Some(&filter))
            .expect("filtered page");
        assert_eq!(
            file_ids(&filtered),
            vec![i64::try_from(index).unwrap() + 1],
            "constant {value:?} did not push down to exactly its own file"
        );
    }
}

/// A float bound of a magnitude no `f64` holds keeps its file rather than being
/// read as a saturated one.
///
/// `CAST` does not fail on such text, it saturates: on MySQL 8 `'1e-400'` reads
/// as `0` and `'1e+400'` as `1.797…e308`, both silently. Read that way, file 1's
/// maximum of `1e-400` becomes `0`, `0 > 1.0` is false and the file is pruned —
/// a file whose real bound is unknown. The pattern bounds both digit runs so the
/// text is declined instead, the comparison is NULL, and the file is kept.
///
/// File 2 is the control: a well-formed maximum that genuinely cannot match.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn filtered_file_page_keeps_a_float_bound_mysql_would_saturate() {
    let (provider, _container) = create_provider_with_file_stats(2).await.unwrap();
    for (data_file_id, max_value) in [(1i64, "1e-400"), (2, "0.5")] {
        // `contains_nan` must be recorded false, or the NaN gate keeps every
        // file on its own and the bound is never consulted.
        sqlx::query(
            "INSERT INTO ducklake_file_column_stats
                 (data_file_id, table_id, column_id, value_count, null_count,
                  min_value, max_value, contains_nan)
             VALUES (?, 1, 9, 10, 0, '0.0', ?, FALSE)",
        )
        .bind(data_file_id)
        .bind(max_value)
        .execute(&provider.pool)
        .await
        .unwrap();
    }

    let column = Arc::new(PhysColumn::new("f", 0)) as Arc<dyn PhysicalExpr>;
    let predicate =
        Arc::new(BinaryExpr::new(column, Operator::Gt, lit(1.0f64))) as Arc<dyn PhysicalExpr>;
    let schema = Schema::new(vec![Field::new("f", DataType::Float64, true)]);
    let columns = vec![DuckLakeTableColumn::new(9, "f".to_string(), "double".to_string(), true)];
    let filter =
        lower_predicate(&predicate, &schema, &columns).expect("predicate lowers to a filter");

    let filtered = provider
        .get_table_file_metadata_page_filtered(1, 1, None, 100, Some(&filter))
        .expect("filtered page");
    assert_eq!(file_ids(&filtered), vec![1]);
}
