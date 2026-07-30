#![cfg(feature = "metadata-duckdb")]
//! Integration tests for information_schema virtual tables and table functions

use datafusion::prelude::*;
use datafusion_ducklake::{DuckLakeCatalog, DuckdbMetadataProvider, register_ducklake_functions};
use std::sync::Arc;

use crate::common;

#[tokio::test]
#[ignore] // Snapshots table requires ducklake_snapshot table which test catalogs don't create
async fn test_information_schema_snapshots() -> Result<(), Box<dyn std::error::Error>> {
    // NOTE: This test is ignored because the test helper uses DuckDB's DuckLake extension
    // which doesn't expose the ducklake_snapshot table directly.
    // In production catalogs created by other means, this table would exist.

    let temp_dir = tempfile::tempdir()?;
    let catalog_path = temp_dir.path().join("test.ducklake");

    common::create_catalog_no_deletes(&catalog_path)?;

    let provider = DuckdbMetadataProvider::new(catalog_path.to_str().unwrap())?;
    let catalog = DuckLakeCatalog::new(provider)?;
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));

    // Query snapshots
    let df = ctx
        .sql("SELECT * FROM ducklake.information_schema.snapshots")
        .await?;

    let results = df.collect().await?;

    assert!(!results.is_empty(), "Should have at least one snapshot");
    println!(
        "✓ Snapshots table test passed - found {} row(s)",
        results[0].num_rows()
    );
    Ok(())
}

#[tokio::test]
async fn test_information_schema_schemata() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = tempfile::tempdir()?;
    let catalog_path = temp_dir.path().join("test.ducklake");

    common::create_catalog_no_deletes(&catalog_path)?;

    let provider = DuckdbMetadataProvider::new(catalog_path.to_str().unwrap())?;
    let catalog = DuckLakeCatalog::new(provider)?;
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));

    // Query schemata
    let df = ctx
        .sql("SELECT schema_name, path FROM ducklake.information_schema.schemata ORDER BY schema_name")
        .await?;

    let results = df.collect().await?;

    // Should have the 'main' schema
    assert!(!results.is_empty(), "Should have at least one schema");

    println!("✓ Schemata table test passed");
    Ok(())
}

#[tokio::test]
async fn test_information_schema_tables() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = tempfile::tempdir()?;
    let catalog_path = temp_dir.path().join("test.ducklake");

    common::create_catalog_no_deletes(&catalog_path)?;

    let provider = DuckdbMetadataProvider::new(catalog_path.to_str().unwrap())?;
    let catalog = DuckLakeCatalog::new(provider)?;
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));

    // Query tables
    let df = ctx
        .sql("SELECT schema_name, table_name FROM ducklake.information_schema.tables ORDER BY schema_name, table_name")
        .await?;

    let results = df.collect().await?;

    // Should have at least the 'users' table from test data
    assert!(!results.is_empty(), "Should have at least one table");

    println!("✓ Tables table test passed");
    Ok(())
}

#[tokio::test]
async fn test_information_schema_columns() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = tempfile::tempdir()?;
    let catalog_path = temp_dir.path().join("test.ducklake");

    common::create_catalog_no_deletes(&catalog_path)?;

    let provider = DuckdbMetadataProvider::new(catalog_path.to_str().unwrap())?;
    let catalog = DuckLakeCatalog::new(provider)?;
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));

    // Query columns
    let df = ctx
        .sql("SELECT schema_name, table_name, column_name, column_type FROM ducklake.information_schema.columns ORDER BY schema_name, table_name, column_name")
        .await?;

    let results = df.collect().await?;

    // Should have columns from test tables
    assert!(!results.is_empty(), "Should have columns");
    assert!(results[0].num_rows() > 0, "Should have column data");

    println!("✓ Columns table test passed");
    Ok(())
}

#[tokio::test]
async fn test_information_schema_filtering() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = tempfile::tempdir()?;
    let catalog_path = temp_dir.path().join("test.ducklake");

    common::create_catalog_no_deletes(&catalog_path)?;

    let provider = DuckdbMetadataProvider::new(catalog_path.to_str().unwrap())?;
    let catalog = DuckLakeCatalog::new(provider)?;
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));

    // Test filtering on schema_name
    let df = ctx
        .sql("SELECT table_name FROM ducklake.information_schema.tables WHERE schema_name = 'main'")
        .await?;

    let results = df.collect().await?;
    assert!(!results.is_empty(), "Should have tables in main schema");

    println!("✓ Filtering test passed");
    Ok(())
}

#[tokio::test]
async fn test_information_schema_aggregation() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = tempfile::tempdir()?;
    let catalog_path = temp_dir.path().join("test.ducklake");

    common::create_catalog_no_deletes(&catalog_path)?;

    let provider = DuckdbMetadataProvider::new(catalog_path.to_str().unwrap())?;
    let catalog = DuckLakeCatalog::new(provider)?;
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));

    // Test aggregation - count tables per schema
    let df = ctx
        .sql("SELECT schema_name, COUNT(*) as table_count FROM ducklake.information_schema.tables GROUP BY schema_name")
        .await?;

    let results = df.collect().await?;
    assert!(!results.is_empty(), "Should have aggregation results");

    println!("✓ Aggregation test passed");
    Ok(())
}

#[tokio::test]
async fn test_information_schema_join() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = tempfile::tempdir()?;
    let catalog_path = temp_dir.path().join("test.ducklake");

    common::create_catalog_no_deletes(&catalog_path)?;

    let provider = DuckdbMetadataProvider::new(catalog_path.to_str().unwrap())?;
    let catalog = DuckLakeCatalog::new(provider)?;
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));

    // Test joining tables with columns
    let df = ctx
        .sql(
            "SELECT t.schema_name, t.table_name, COUNT(c.column_name) as column_count \
             FROM ducklake.information_schema.tables t \
             JOIN ducklake.information_schema.columns c \
               ON t.schema_name = c.schema_name AND t.table_name = c.table_name \
             GROUP BY t.schema_name, t.table_name \
             ORDER BY t.schema_name, t.table_name",
        )
        .await?;

    let results = df.collect().await?;
    assert!(!results.is_empty(), "Should have join results");

    println!("✓ Join test passed");
    Ok(())
}

#[tokio::test]
async fn test_information_schema_projection() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = tempfile::tempdir()?;
    let catalog_path = temp_dir.path().join("test.ducklake");

    common::create_catalog_no_deletes(&catalog_path)?;

    let provider = DuckdbMetadataProvider::new(catalog_path.to_str().unwrap())?;
    let catalog = DuckLakeCatalog::new(provider)?;
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));

    // Test projection - only select specific columns
    let df = ctx
        .sql("SELECT column_name FROM ducklake.information_schema.columns")
        .await?;

    let results = df.collect().await?;
    assert!(!results.is_empty(), "Should have projection results");

    // Verify only one column returned
    assert_eq!(results[0].num_columns(), 1, "Should have only one column");

    println!("✓ Projection test passed");
    Ok(())
}

#[tokio::test]
async fn test_information_schema_all_tables_exist() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = tempfile::tempdir()?;
    let catalog_path = temp_dir.path().join("test.ducklake");

    common::create_catalog_no_deletes(&catalog_path)?;

    let provider = DuckdbMetadataProvider::new(catalog_path.to_str().unwrap())?;
    let catalog = DuckLakeCatalog::new(provider)?;
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));

    // Verify core tables exist (excluding snapshots which requires special setup)
    let tables = vec!["schemata", "tables", "columns"];

    for table in tables {
        let query = format!("SELECT * FROM ducklake.information_schema.{}", table);
        let df = ctx.sql(&query).await?;
        let results = df.collect().await?;
        assert!(!results.is_empty(), "Table {} should be queryable", table);
        println!(
            "✓ Table {} exists and is queryable with {} rows",
            table,
            results[0].num_rows()
        );
    }

    Ok(())
}

#[tokio::test]
async fn test_information_schema_in_schema_list() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = tempfile::tempdir()?;
    let catalog_path = temp_dir.path().join("test.ducklake");

    common::create_catalog_no_deletes(&catalog_path)?;

    let provider = DuckdbMetadataProvider::new(catalog_path.to_str().unwrap())?;
    let catalog = DuckLakeCatalog::new(provider)?;
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));

    // Test that we can query information_schema tables
    // (DataFusion may not support SHOW SCHEMAS in all versions)
    let df = ctx
        .sql("SELECT * FROM ducklake.information_schema.schemata")
        .await?;
    let results = df.collect().await?;

    assert!(
        !results.is_empty(),
        "Should be able to query information_schema"
    );

    println!("✓ information_schema is accessible");
    Ok(())
}

#[tokio::test]
async fn test_information_schema_files() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = tempfile::tempdir()?;
    let catalog_path = temp_dir.path().join("test.ducklake");

    common::create_catalog_no_deletes(&catalog_path)?;

    let provider = DuckdbMetadataProvider::new(catalog_path.to_str().unwrap())?;
    let catalog = DuckLakeCatalog::new(provider)?;
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));

    // Query files table
    let df = ctx
        .sql("SELECT schema_name, table_name, file_path, file_size_bytes, has_delete_file FROM ducklake.information_schema.files")
        .await?;

    let results = df.collect().await?;

    // Should have files from test tables
    assert!(!results.is_empty(), "Should have files");
    assert!(results[0].num_rows() > 0, "Should have file data");

    println!(
        "✓ Files table test passed - found {} file(s)",
        results[0].num_rows()
    );
    Ok(())
}

#[tokio::test]
async fn test_information_schema_files_filtering() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = tempfile::tempdir()?;
    let catalog_path = temp_dir.path().join("test.ducklake");

    common::create_catalog_no_deletes(&catalog_path)?;

    let provider = DuckdbMetadataProvider::new(catalog_path.to_str().unwrap())?;
    let catalog = DuckLakeCatalog::new(provider)?;
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));

    // Test filtering by table_name
    let df = ctx
        .sql("SELECT file_path FROM ducklake.information_schema.files WHERE table_name = 'users'")
        .await?;

    let results = df.collect().await?;
    assert!(!results.is_empty(), "Should have files for users table");

    println!("✓ Files filtering test passed");
    Ok(())
}

// Table Functions Tests

#[tokio::test]
async fn test_ducklake_snapshots_function() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = tempfile::tempdir()?;
    let catalog_path = temp_dir.path().join("test.ducklake");

    common::create_catalog_no_deletes(&catalog_path)?;

    let provider = DuckdbMetadataProvider::new(catalog_path.to_str().unwrap())?;
    let ctx = SessionContext::new();

    // Register table functions
    register_ducklake_functions(&ctx, Arc::new(provider));

    // Query using function syntax
    let df = ctx.sql("SELECT * FROM ducklake_snapshots()").await?;
    let results = df.collect().await?;

    assert!(!results.is_empty(), "Should have snapshots");
    println!("✓ ducklake_snapshots() function test passed");
    Ok(())
}

#[tokio::test]
async fn test_ducklake_table_info_function() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = tempfile::tempdir()?;
    let catalog_path = temp_dir.path().join("test.ducklake");

    common::create_catalog_no_deletes(&catalog_path)?;

    let provider = DuckdbMetadataProvider::new(catalog_path.to_str().unwrap())?;
    let ctx = SessionContext::new();

    register_ducklake_functions(&ctx, Arc::new(provider));

    let df = ctx
        .sql("SELECT table_name, file_count, file_size_bytes FROM ducklake_table_info()")
        .await?;
    let results = df.collect().await?;

    assert!(!results.is_empty(), "Should have table info");
    println!("✓ ducklake_table_info() function test passed");
    Ok(())
}

#[tokio::test]
async fn test_ducklake_list_files_function() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = tempfile::tempdir()?;
    let catalog_path = temp_dir.path().join("test.ducklake");

    common::create_catalog_no_deletes(&catalog_path)?;

    let provider = DuckdbMetadataProvider::new(catalog_path.to_str().unwrap())?;
    let ctx = SessionContext::new();

    register_ducklake_functions(&ctx, Arc::new(provider));

    let df = ctx
        .sql("SELECT file_path, file_size_bytes FROM ducklake_list_files()")
        .await?;
    let results = df.collect().await?;

    assert!(!results.is_empty(), "Should have files");
    println!("✓ ducklake_list_files() function test passed");
    Ok(())
}

#[tokio::test]
async fn test_table_info_aggregation() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = tempfile::tempdir()?;
    let catalog_path = temp_dir.path().join("test.ducklake");

    common::create_catalog_no_deletes(&catalog_path)?;

    let provider = DuckdbMetadataProvider::new(catalog_path.to_str().unwrap())?;
    let ctx = SessionContext::new();

    register_ducklake_functions(&ctx, Arc::new(provider));

    // Get total storage
    let df = ctx
        .sql("SELECT SUM(file_size_bytes) as total_bytes FROM ducklake_table_info()")
        .await?;
    let results = df.collect().await?;

    assert!(!results.is_empty(), "Should have aggregation results");
    println!("✓ Table info aggregation test passed");
    Ok(())
}

#[tokio::test]
async fn test_function_rejects_arguments() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = tempfile::tempdir()?;
    let catalog_path = temp_dir.path().join("test.ducklake");

    common::create_catalog_no_deletes(&catalog_path)?;

    let provider = DuckdbMetadataProvider::new(catalog_path.to_str().unwrap())?;
    let ctx = SessionContext::new();

    register_ducklake_functions(&ctx, Arc::new(provider));

    // These functions should reject arguments
    let result = ctx.sql("SELECT * FROM ducklake_snapshots('arg')").await;
    assert!(result.is_err(), "Should reject arguments");

    println!("✓ Function argument validation test passed");
    Ok(())
}
