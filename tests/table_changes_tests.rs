#![cfg(feature = "metadata-duckdb")]
//! Integration tests for ducklake_table_changes() and ducklake_table_deletions() functions
//!
//! These tests verify that the table_changes function correctly returns
//! files added between snapshots for CDC (Change Data Capture) queries,
//! and that table_deletions correctly returns deleted rows.

mod common;

use std::sync::Arc;

use arrow::array::{Array, Int32Array, StringArray};
use arrow::record_batch::RecordBatch;
use datafusion::error::Result as DataFusionResult;
use datafusion::prelude::*;
use datafusion_ducklake::{DuckLakeCatalog, DuckdbMetadataProvider, register_ducklake_functions};
use tempfile::TempDir;

/// Helper to get i32 values from a column
fn get_int32_column(batch: &RecordBatch, col_idx: usize) -> Vec<i32> {
    let column = batch.column(col_idx);
    let array = column
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("Expected Int32Array");

    (0..array.len())
        .filter_map(|i| {
            if array.is_null(i) {
                None
            } else {
                Some(array.value(i))
            }
        })
        .collect()
}

/// Helper to get string values from a column (supports both Utf8 and LargeUtf8)
fn get_string_column(batch: &RecordBatch, col_idx: usize) -> Vec<String> {
    let column = batch.column(col_idx);

    // Try StringArray (Utf8) first
    if let Some(array) = column.as_any().downcast_ref::<StringArray>() {
        return (0..array.len())
            .filter_map(|i| {
                if array.is_null(i) {
                    None
                } else {
                    Some(array.value(i).to_string())
                }
            })
            .collect();
    }

    // Try LargeStringArray (LargeUtf8)
    if let Some(array) = column
        .as_any()
        .downcast_ref::<arrow::array::LargeStringArray>()
    {
        return (0..array.len())
            .filter_map(|i| {
                if array.is_null(i) {
                    None
                } else {
                    Some(array.value(i).to_string())
                }
            })
            .collect();
    }

    // Try StringViewArray (Utf8View) — the layout scanned user string columns use.
    if let Some(array) = column
        .as_any()
        .downcast_ref::<arrow::array::StringViewArray>()
    {
        return (0..array.len())
            .filter_map(|i| {
                if array.is_null(i) {
                    None
                } else {
                    Some(array.value(i).to_string())
                }
            })
            .collect();
    }

    panic!(
        "Expected StringArray, LargeStringArray, or StringViewArray, got {:?}",
        column.data_type()
    );
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    /// Helper to create a context with catalog and register functions
    async fn create_context_with_functions(path: &str) -> DataFusionResult<SessionContext> {
        let provider = DuckdbMetadataProvider::new(path)?;
        let provider_arc: Arc<dyn datafusion_ducklake::MetadataProvider> =
            Arc::new(DuckdbMetadataProvider::new(path)?);

        let catalog = DuckLakeCatalog::new(provider)?;

        let ctx = SessionContext::new();
        ctx.register_catalog("ducklake", Arc::new(catalog));

        // Register the table functions including ducklake_table_changes
        register_ducklake_functions(&ctx, provider_arc);

        Ok(ctx)
    }

    /// Test that ducklake_table_changes returns empty when no changes between snapshots
    #[tokio::test]
    async fn test_table_changes_no_changes() -> DataFusionResult<()> {
        let temp_dir = TempDir::new().unwrap();
        let catalog_path = temp_dir.path().join("multi_snapshot.ducklake");

        common::create_catalog_multiple_snapshots(&catalog_path)
            .map_err(common::to_datafusion_error)?;

        let ctx = create_context_with_functions(catalog_path.to_str().unwrap()).await?;

        // Query changes between same snapshot (should be empty)
        let df = ctx
            .sql("SELECT * FROM ducklake_table_changes('main.events', 1, 1)")
            .await?;

        let batches: Vec<RecordBatch> = df.collect().await?;
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();

        assert_eq!(total_rows, 0, "No changes expected between same snapshot");

        Ok(())
    }

    /// Test that ducklake_table_changes returns insert changes with actual row data
    #[tokio::test]
    async fn test_table_changes_inserts() -> DataFusionResult<()> {
        let temp_dir = TempDir::new().unwrap();
        let catalog_path = temp_dir.path().join("multi_snapshot.ducklake");

        common::create_catalog_multiple_snapshots(&catalog_path)
            .map_err(common::to_datafusion_error)?;

        let ctx = create_context_with_functions(catalog_path.to_str().unwrap()).await?;

        // Query all columns including table data + CDC metadata; the fixture's
        // final snapshot also deletes id=2, so filter to the insert rows.
        let df = ctx
            .sql(
                "SELECT id, event_type, value, snapshot_id, change_type \
                 FROM ducklake_table_changes('main.events', 0, 10) \
                 WHERE change_type = 'insert' ORDER BY id",
            )
            .await?;

        let batches: Vec<RecordBatch> = df.collect().await?;

        // Should have rows from both inserts (5 total rows inserted)
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 5, "Should have 5 inserted rows");

        // Verify actual row data
        let mut all_ids: Vec<i32> = Vec::new();
        let mut all_change_types: Vec<String> = Vec::new();
        for batch in &batches {
            all_ids.extend(get_int32_column(batch, 0)); // id column
            all_change_types.extend(get_string_column(batch, 4)); // change_type column
        }

        // All rows should have ids 1-5
        assert_eq!(all_ids, vec![1, 2, 3, 4, 5], "Should have ids 1-5");

        for change_type in all_change_types {
            assert_eq!(change_type, "insert", "filtered to insert rows");
        }

        Ok(())
    }

    /// Test that ducklake_table_changes returns pure deletes as `delete` rows
    /// carrying the deleted rows' old values, matching official DuckLake.
    ///
    /// The fixture deletes the row with id=2 ('view', 50) in its final snapshot.
    #[tokio::test]
    async fn test_table_changes_deletes() -> DataFusionResult<()> {
        let temp_dir = TempDir::new().unwrap();
        let catalog_path = temp_dir.path().join("multi_snapshot.ducklake");

        common::create_catalog_multiple_snapshots(&catalog_path)
            .map_err(common::to_datafusion_error)?;

        let ctx = create_context_with_functions(catalog_path.to_str().unwrap()).await?;

        let df = ctx
            .sql(
                "SELECT id, event_type, value FROM \
                 ducklake_table_changes('main.events', 0, 100) \
                 WHERE change_type = 'delete'",
            )
            .await?;

        let batches: Vec<RecordBatch> = df.collect().await?;
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 1, "exactly the one deleted row surfaces");

        let b = batches.iter().find(|b| b.num_rows() > 0).unwrap();
        let ids = b
            .column(0)
            .as_any()
            .downcast_ref::<arrow::array::Int32Array>()
            .unwrap();
        assert_eq!(ids.value(0), 2, "the delete carries the old row's values");

        Ok(())
    }

    /// Test that ducklake_table_changes works with table name without schema
    #[tokio::test]
    async fn test_table_changes_default_schema() -> DataFusionResult<()> {
        let temp_dir = TempDir::new().unwrap();
        let catalog_path = temp_dir.path().join("multi_snapshot.ducklake");

        common::create_catalog_multiple_snapshots(&catalog_path)
            .map_err(common::to_datafusion_error)?;

        let ctx = create_context_with_functions(catalog_path.to_str().unwrap()).await?;

        // Query using just table name (should default to 'main' schema)
        let df = ctx
            .sql("SELECT * FROM ducklake_table_changes('events', 0, 10)")
            .await?;

        let batches: Vec<RecordBatch> = df.collect().await?;
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();

        assert!(total_rows > 0, "Should find changes using default schema");

        Ok(())
    }

    /// Test that ducklake_table_changes returns correct schema
    #[tokio::test]
    async fn test_table_changes_schema() -> DataFusionResult<()> {
        let temp_dir = TempDir::new().unwrap();
        let catalog_path = temp_dir.path().join("multi_snapshot.ducklake");

        common::create_catalog_multiple_snapshots(&catalog_path)
            .map_err(common::to_datafusion_error)?;

        let ctx = create_context_with_functions(catalog_path.to_str().unwrap()).await?;

        let df = ctx
            .sql("SELECT * FROM ducklake_table_changes('main.events', 0, 10)")
            .await?;

        let schema = df.schema();

        // Verify the expected columns are present (Phase 2: table columns + CDC columns)
        let field_names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();

        // Table columns (from events table)
        assert!(
            field_names.contains(&"id"),
            "Schema should contain 'id' table column"
        );
        assert!(
            field_names.contains(&"event_type"),
            "Schema should contain 'event_type' table column"
        );
        assert!(
            field_names.contains(&"value"),
            "Schema should contain 'value' table column"
        );

        // CDC metadata columns
        assert!(
            field_names.contains(&"rowid"),
            "Schema should contain 'rowid' CDC column"
        );
        assert!(
            field_names.contains(&"snapshot_id"),
            "Schema should contain 'snapshot_id' CDC column"
        );
        assert!(
            field_names.contains(&"change_type"),
            "Schema should contain 'change_type' CDC column"
        );

        // Verify column order: CDC columns leading (official DuckLake order),
        // then table columns.
        assert_eq!(field_names.len(), 6, "Should have 6 columns total");
        assert_eq!(
            field_names,
            vec!["snapshot_id", "rowid", "change_type", "id", "event_type", "value"],
            "Columns should be in order: CDC columns (official order), then table columns"
        );

        Ok(())
    }

    /// Test error handling for non-existent table
    #[tokio::test]
    async fn test_table_changes_nonexistent_table() -> DataFusionResult<()> {
        let temp_dir = TempDir::new().unwrap();
        let catalog_path = temp_dir.path().join("multi_snapshot.ducklake");

        common::create_catalog_multiple_snapshots(&catalog_path)
            .map_err(common::to_datafusion_error)?;

        let ctx = create_context_with_functions(catalog_path.to_str().unwrap()).await?;

        // Query a non-existent table
        let result = ctx
            .sql("SELECT * FROM ducklake_table_changes('main.nonexistent', 0, 10)")
            .await;

        assert!(result.is_err(), "Should error for non-existent table");

        Ok(())
    }

    /// Test error handling for non-existent schema
    #[tokio::test]
    async fn test_table_changes_nonexistent_schema() -> DataFusionResult<()> {
        let temp_dir = TempDir::new().unwrap();
        let catalog_path = temp_dir.path().join("multi_snapshot.ducklake");

        common::create_catalog_multiple_snapshots(&catalog_path)
            .map_err(common::to_datafusion_error)?;

        let ctx = create_context_with_functions(catalog_path.to_str().unwrap()).await?;

        // Query a non-existent schema
        let result = ctx
            .sql("SELECT * FROM ducklake_table_changes('nonexistent.events', 0, 10)")
            .await;

        assert!(result.is_err(), "Should error for non-existent schema");

        Ok(())
    }

    /// Test error handling for invalid snapshot range (start > end)
    #[tokio::test]
    async fn test_table_changes_invalid_snapshot_range() -> DataFusionResult<()> {
        let temp_dir = TempDir::new().unwrap();
        let catalog_path = temp_dir.path().join("multi_snapshot.ducklake");

        common::create_catalog_multiple_snapshots(&catalog_path)
            .map_err(common::to_datafusion_error)?;

        let ctx = create_context_with_functions(catalog_path.to_str().unwrap()).await?;

        // Query with start_snapshot > end_snapshot
        let result = ctx
            .sql("SELECT * FROM ducklake_table_changes('main.events', 10, 5)")
            .await;

        assert!(
            result.is_err(),
            "Should error when start_snapshot > end_snapshot"
        );

        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("start_snapshot") && err_msg.contains("end_snapshot"),
            "Error message should mention snapshot range, got: {}",
            err_msg
        );

        Ok(())
    }
}

#[cfg(test)]
mod table_deletions_tests {
    use super::*;

    /// Helper to create a context with catalog and register functions
    async fn create_context_with_functions(path: &str) -> DataFusionResult<SessionContext> {
        let provider = DuckdbMetadataProvider::new(path)?;
        let provider_arc: Arc<dyn datafusion_ducklake::MetadataProvider> =
            Arc::new(DuckdbMetadataProvider::new(path)?);

        let catalog = DuckLakeCatalog::new(provider)?;

        let ctx = SessionContext::new();
        ctx.register_catalog("ducklake", Arc::new(catalog));

        // Register the table functions including ducklake_table_deletions
        register_ducklake_functions(&ctx, provider_arc);

        Ok(ctx)
    }

    /// Test that ducklake_table_deletions returns empty when no rows are deleted
    #[tokio::test]
    async fn test_table_deletions_no_deletions() -> DataFusionResult<()> {
        let temp_dir = TempDir::new().unwrap();
        let catalog_path = temp_dir.path().join("no_deletes.ducklake");

        // Create catalog with only inserts, no deletes
        common::create_catalog_no_deletes(&catalog_path).map_err(common::to_datafusion_error)?;

        let ctx = create_context_with_functions(catalog_path.to_str().unwrap()).await?;

        // Query deletions - should be empty since no rows were deleted
        let df = ctx
            .sql("SELECT * FROM ducklake_table_deletions('main.users', 0, 100)")
            .await?;

        let batches: Vec<RecordBatch> = df.collect().await?;
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();

        assert_eq!(
            total_rows, 0,
            "Should have no deleted rows when no deletions occurred"
        );

        Ok(())
    }

    /// Test that ducklake_table_deletions returns deleted rows correctly
    #[tokio::test]
    async fn test_table_deletions_with_deletes() -> DataFusionResult<()> {
        let temp_dir = TempDir::new().unwrap();
        let catalog_path = temp_dir.path().join("with_deletes.ducklake");

        // Create catalog with 5 rows, 2 deleted (ids 2 and 4)
        common::create_catalog_with_deletes(&catalog_path).map_err(common::to_datafusion_error)?;

        let ctx = create_context_with_functions(catalog_path.to_str().unwrap()).await?;

        // Query all deletions - products table has: id, name, price, in_stock, snapshot_id, change_type
        let df = ctx
            .sql("SELECT * FROM ducklake_table_deletions('main.products', 0, 100) ORDER BY id")
            .await?;

        let batches: Vec<RecordBatch> = df.collect().await?;
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();

        // Should have 2 deleted rows (ids 2 and 4)
        assert_eq!(total_rows, 2, "Should have 2 deleted rows");

        // Get column indices by name
        let schema = batches[0].schema();
        let id_idx = schema.index_of("id").expect("id column");
        let change_type_idx = schema.index_of("change_type").expect("change_type column");

        // Verify deleted row ids
        let mut all_ids: Vec<i32> = Vec::new();
        let mut all_change_types: Vec<String> = Vec::new();
        for batch in &batches {
            all_ids.extend(get_int32_column(batch, id_idx));
            all_change_types.extend(get_string_column(batch, change_type_idx));
        }

        assert_eq!(all_ids, vec![2, 4], "Should have deleted ids 2 and 4");

        // All change_types should be 'delete'
        for change_type in &all_change_types {
            assert_eq!(change_type, "delete", "All change_types should be 'delete'");
        }

        Ok(())
    }

    /// Test ducklake_table_deletions with complex scenario:
    /// delete all rows, insert more, delete few more
    #[tokio::test]
    async fn test_table_deletions_complex_scenario() -> DataFusionResult<()> {
        let temp_dir = TempDir::new().unwrap();
        let catalog_path = temp_dir.path().join("complex_deletes.ducklake");

        // Create catalog with complex deletion scenario
        common::create_catalog_complex_deletions(&catalog_path)
            .map_err(common::to_datafusion_error)?;

        let ctx = create_context_with_functions(catalog_path.to_str().unwrap()).await?;

        // Query all deletions across all snapshots - items table has: id, snapshot_id, change_type
        let df = ctx
            .sql("SELECT * FROM ducklake_table_deletions('main.items', 0, 100) ORDER BY id")
            .await?;

        let batches: Vec<RecordBatch> = df.collect().await?;
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();

        // We deleted: initial 3 rows (1,2,3), then from second batch deleted 2 rows (5,6)
        // Total: 5 deletions
        assert_eq!(total_rows, 5, "Should have 5 total deleted rows");

        // Get column index by name
        let schema = batches[0].schema();
        let change_type_idx = schema.index_of("change_type").expect("change_type column");

        // Verify all change_types are 'delete'
        let mut all_change_types: Vec<String> = Vec::new();
        for batch in &batches {
            all_change_types.extend(get_string_column(batch, change_type_idx));
        }

        for change_type in &all_change_types {
            assert_eq!(change_type, "delete", "All change_types should be 'delete'");
        }

        Ok(())
    }

    /// Test ducklake_table_deletions schema includes table columns + CDC columns
    #[tokio::test]
    async fn test_table_deletions_schema() -> DataFusionResult<()> {
        let temp_dir = TempDir::new().unwrap();
        let catalog_path = temp_dir.path().join("schema_test.ducklake");

        common::create_catalog_with_deletes(&catalog_path).map_err(common::to_datafusion_error)?;

        let ctx = create_context_with_functions(catalog_path.to_str().unwrap()).await?;

        let df = ctx
            .sql("SELECT * FROM ducklake_table_deletions('main.products', 0, 100)")
            .await?;

        let schema = df.schema();
        let field_names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();

        // Table columns (from products table)
        assert!(
            field_names.contains(&"id"),
            "Schema should contain 'id' table column"
        );
        assert!(
            field_names.contains(&"name"),
            "Schema should contain 'name' table column"
        );
        assert!(
            field_names.contains(&"price"),
            "Schema should contain 'price' table column"
        );
        assert!(
            field_names.contains(&"in_stock"),
            "Schema should contain 'in_stock' table column"
        );

        // CDC metadata columns
        assert!(
            field_names.contains(&"snapshot_id"),
            "Schema should contain 'snapshot_id' CDC column"
        );
        assert!(
            field_names.contains(&"change_type"),
            "Schema should contain 'change_type' CDC column"
        );

        Ok(())
    }
}
