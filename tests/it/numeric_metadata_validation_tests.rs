#![cfg(feature = "metadata-duckdb")]
//! Tests for numeric metadata validation (#58, #59)
//!
//! Verifies that negative file_size_bytes and footer_size values in catalog
//! metadata are caught early with clear error messages instead of wrapping
//! to huge values via unchecked `as u64`/`as usize` casts.

use std::sync::Arc;

use datafusion::error::Result as DataFusionResult;
use datafusion::prelude::*;
use datafusion_ducklake::{DuckLakeCatalog, DuckdbMetadataProvider};
use tempfile::TempDir;

/// Creates a catalog with data, then corrupts file_size_bytes to a negative value
fn create_catalog_with_negative_file_size(catalog_path: &std::path::Path) -> anyhow::Result<()> {
    let conn = duckdb::Connection::open_in_memory()?;
    conn.execute("INSTALL ducklake;", [])?;
    conn.execute("LOAD ducklake;", [])?;

    let ducklake_path = format!("ducklake:{}", catalog_path.display());
    conn.execute(&format!("ATTACH '{}' AS test_catalog;", ducklake_path), [])?;

    conn.execute(
        "CREATE TABLE test_catalog.items (id INT, name VARCHAR);",
        [],
    )?;
    conn.execute(
        "INSERT INTO test_catalog.items VALUES (1, 'Widget'), (2, 'Gadget');",
        [],
    )?;

    // Detach the DuckLake catalog so we can tamper with the raw metadata
    conn.execute("DETACH test_catalog;", [])?;

    // Now open the catalog DB directly and corrupt file_size_bytes
    let meta_conn = duckdb::Connection::open(catalog_path)?;
    meta_conn.execute("UPDATE ducklake_data_file SET file_size_bytes = -1;", [])?;

    Ok(())
}

/// Creates a catalog with data, then corrupts footer_size to a negative value
fn create_catalog_with_negative_footer_size(catalog_path: &std::path::Path) -> anyhow::Result<()> {
    let conn = duckdb::Connection::open_in_memory()?;
    conn.execute("INSTALL ducklake;", [])?;
    conn.execute("LOAD ducklake;", [])?;

    let ducklake_path = format!("ducklake:{}", catalog_path.display());
    conn.execute(&format!("ATTACH '{}' AS test_catalog;", ducklake_path), [])?;

    conn.execute(
        "CREATE TABLE test_catalog.items (id INT, name VARCHAR);",
        [],
    )?;
    conn.execute(
        "INSERT INTO test_catalog.items VALUES (1, 'Widget'), (2, 'Gadget');",
        [],
    )?;

    // Detach the DuckLake catalog so we can tamper with the raw metadata
    conn.execute("DETACH test_catalog;", [])?;

    // Now open the catalog DB directly and set footer_size to negative
    let meta_conn = duckdb::Connection::open(catalog_path)?;
    meta_conn.execute("UPDATE ducklake_data_file SET footer_size = -42;", [])?;

    Ok(())
}

fn create_catalog(path: &str) -> DataFusionResult<Arc<DuckLakeCatalog>> {
    let provider = DuckdbMetadataProvider::new(path)
        .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
    let catalog = DuckLakeCatalog::new(provider)
        .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
    Ok(Arc::new(catalog))
}

#[tokio::test]
async fn test_negative_file_size_produces_clear_error() -> DataFusionResult<()> {
    let temp_dir =
        TempDir::new().map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
    let catalog_path = temp_dir.path().join("neg_size.ducklake");

    create_catalog_with_negative_file_size(&catalog_path)
        .map_err(|e| datafusion::error::DataFusionError::External(e.into()))?;

    let catalog = create_catalog(&catalog_path.to_string_lossy())?;
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", catalog);

    let result = ctx
        .sql("SELECT * FROM ducklake.main.items")
        .await?
        .collect()
        .await;

    assert!(
        result.is_err(),
        "Query should fail with negative file_size_bytes"
    );
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("Invalid file_size_bytes"),
        "Error should mention invalid file_size_bytes, got: {}",
        err_msg
    );
    assert!(
        err_msg.contains("-1"),
        "Error should contain the negative value, got: {}",
        err_msg
    );

    Ok(())
}

#[tokio::test]
async fn test_negative_footer_size_is_gracefully_skipped() -> DataFusionResult<()> {
    let temp_dir =
        TempDir::new().map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
    let catalog_path = temp_dir.path().join("neg_footer.ducklake");

    create_catalog_with_negative_footer_size(&catalog_path)
        .map_err(|e| datafusion::error::DataFusionError::External(e.into()))?;

    let catalog = create_catalog(&catalog_path.to_string_lossy())?;
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", catalog);

    // Negative footer_size should be skipped (not used as hint), query should succeed
    let df = ctx.sql("SELECT * FROM ducklake.main.items").await?;
    let batches = df.collect().await?;

    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(
        total_rows, 2,
        "Should still return all rows when footer_size is negative"
    );

    Ok(())
}
