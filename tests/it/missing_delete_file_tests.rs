#![cfg(feature = "metadata-duckdb")]
//! Tests for missing delete file error handling (issue #52)

use crate::common;

use std::sync::Arc;

use datafusion::error::Result as DataFusionResult;
use datafusion::prelude::*;
use datafusion_ducklake::{DuckLakeCatalog, DuckdbMetadataProvider};
use tempfile::TempDir;

fn create_catalog(path: &str) -> DataFusionResult<Arc<DuckLakeCatalog>> {
    let provider = DuckdbMetadataProvider::new(path)
        .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
    let catalog = DuckLakeCatalog::new(provider)
        .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
    Ok(Arc::new(catalog))
}

fn remove_delete_files(data_dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut removed = Vec::new();
    if let Ok(entries) = std::fs::read_dir(data_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                removed.extend(remove_delete_files(&path));
            } else if let Some(name) = path.file_name().and_then(|n| n.to_str())
                && name.contains("delete")
                && name.ends_with(".parquet")
            {
                std::fs::remove_file(&path).expect("Failed to remove delete file");
                removed.push(path);
            }
        }
    }
    removed
}

#[tokio::test]
async fn test_missing_delete_file_returns_error() -> DataFusionResult<()> {
    let temp_dir =
        TempDir::new().map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
    let catalog_path = temp_dir.path().join("missing_delete.ducklake");
    common::create_catalog_with_deletes(&catalog_path).map_err(common::to_datafusion_error)?;
    let removed = remove_delete_files(temp_dir.path());
    assert!(
        !removed.is_empty(),
        "Should have removed at least one delete file"
    );
    let catalog = create_catalog(&catalog_path.to_string_lossy())?;
    let ctx = SessionContext::new();
    ctx.register_catalog("test", catalog);
    let df = ctx
        .sql("SELECT * FROM test.main.products ORDER BY id")
        .await?;
    let result = df.collect().await;
    assert!(result.is_err(), "Expected error for missing delete file");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("not found"),
        "Error should mention not found, got: {}",
        err_msg
    );
    assert!(
        err_msg.contains("Delete file"),
        "Error should mention Delete file, got: {}",
        err_msg
    );
    Ok(())
}

#[tokio::test]
async fn test_delete_files_work_normally() -> DataFusionResult<()> {
    let temp_dir =
        TempDir::new().map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
    let catalog_path = temp_dir.path().join("normal_deletes.ducklake");
    common::create_catalog_with_deletes(&catalog_path).map_err(common::to_datafusion_error)?;
    let catalog = create_catalog(&catalog_path.to_string_lossy())?;
    let ctx = SessionContext::new();
    ctx.register_catalog("test", catalog);
    let df = ctx
        .sql("SELECT id FROM test.main.products ORDER BY id")
        .await?;
    let results = df.collect().await?;
    let total_rows: usize = results.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 3, "Should have 3 rows after deletes");
    Ok(())
}

#[tokio::test]
async fn test_missing_delete_file_count_query_errors() -> DataFusionResult<()> {
    let temp_dir =
        TempDir::new().map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
    let catalog_path = temp_dir.path().join("missing_delete_count.ducklake");
    common::create_catalog_with_deletes(&catalog_path).map_err(common::to_datafusion_error)?;
    let removed = remove_delete_files(temp_dir.path());
    assert!(!removed.is_empty());
    let catalog = create_catalog(&catalog_path.to_string_lossy())?;
    let ctx = SessionContext::new();
    ctx.register_catalog("test", catalog);
    let df = ctx.sql("SELECT COUNT(*) FROM test.main.products").await?;
    let result = df.collect().await;
    assert!(result.is_err(), "COUNT should error on missing delete file");
    Ok(())
}
