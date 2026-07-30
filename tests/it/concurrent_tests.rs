#![cfg(feature = "metadata-duckdb")]
//! Concurrent query safety tests
//!
//! This test suite verifies that DataFusion-DuckLake is thread-safe and can handle
//! multiple concurrent queries without race conditions or data corruption.
//!
//! ## Concurrency Pattern
//!
//! The tests use Tokio's async runtime with `tokio::spawn` to create true concurrent
//! tasks that may run on different threads. Each task:
//! 1. Creates its own SQL query (SELECT, COUNT, aggregation, etc.)
//! 2. Executes against the same shared catalog and table
//! 3. Collects and verifies results independently
//!
//! ## Thread Safety Guarantees
//!
//! The DuckLake implementation is designed to be thread-safe:
//! - **MetadataProvider**: Uses a single shared connection protected by Mutex
//! - **Catalog/Schema**: Dynamic metadata lookup with no shared mutable state
//! - **Table**: Immutable metadata cached at creation time
//! - **ObjectStore**: DataFusion's object stores are Arc<dyn ObjectStore> (thread-safe)
//!
//! These tests verify these guarantees hold under concurrent load.

use crate::common;

use std::sync::Arc;

use arrow::array::{Array, Int32Array, Int64Array};
use datafusion::common::DataFusionError;
use datafusion::error::Result as DataFusionResult;
use datafusion::prelude::*;
use datafusion_ducklake::{DuckLakeCatalog, DuckdbMetadataProvider};
use tempfile::TempDir;

/// Test helper to extract integer values from a RecordBatch column
/// Supports both Int32 and Int64
fn get_int_column(batch: &arrow::record_batch::RecordBatch, col_idx: usize) -> Vec<i32> {
    let column = batch.column(col_idx);

    // Try Int32 first
    if let Some(array) = column.as_any().downcast_ref::<Int32Array>() {
        return (0..array.len())
            .filter_map(|i| {
                if array.is_null(i) {
                    None
                } else {
                    Some(array.value(i))
                }
            })
            .collect();
    }

    // Try Int64
    if let Some(array) = column.as_any().downcast_ref::<Int64Array>() {
        return (0..array.len())
            .filter_map(|i| {
                if array.is_null(i) {
                    None
                } else {
                    Some(array.value(i) as i32)
                }
            })
            .collect();
    }

    panic!(
        "Column should be Int32Array or Int64Array, got {:?}",
        column.data_type()
    );
}

/// Test concurrent queries on the same table
///
/// This test spawns 10 concurrent Tokio tasks that all query the same table
/// simultaneously. Each task executes a SELECT query and verifies the results.
///
/// **Expected behavior**:
/// - All tasks should complete successfully without panics or deadlocks
/// - All tasks should return identical, correct results
/// - No race conditions in metadata provider or catalog implementation
#[tokio::test]
async fn test_concurrent_select_queries() -> DataFusionResult<()> {
    let temp_dir =
        TempDir::new().map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
    let catalog_path = temp_dir.path().join("no_deletes.ducklake");

    // Generate test data
    common::create_catalog_no_deletes(&catalog_path).map_err(common::to_datafusion_error)?;

    // Create shared catalog (Arc allows sharing across tasks)
    let provider = DuckdbMetadataProvider::new(catalog_path.to_string_lossy().to_string())
        .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
    let catalog = Arc::new(
        DuckLakeCatalog::new(provider)
            .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?,
    );

    // Create 10 concurrent query tasks
    let mut tasks = Vec::new();
    for task_id in 0..10 {
        let catalog_clone = Arc::clone(&catalog);

        let task = tokio::spawn(async move {
            // Each task creates its own SessionContext (thread-local)
            let ctx = SessionContext::new();
            ctx.register_catalog("no_deletes", catalog_clone);

            // Execute query
            let df = ctx
                .sql("SELECT * FROM no_deletes.main.users ORDER BY id")
                .await?;
            let results = df.collect().await?;

            // Verify results
            let total_rows: usize = results.iter().map(|b| b.num_rows()).sum();
            assert_eq!(total_rows, 4, "Task {} got wrong row count", task_id);

            // Collect all IDs to verify data integrity
            let mut all_ids = Vec::new();
            for batch in &results {
                all_ids.extend(get_int_column(batch, 0));
            }
            assert_eq!(all_ids, vec![1, 2, 3, 4], "Task {} got wrong IDs", task_id);

            Ok::<_, DataFusionError>((task_id, total_rows, all_ids))
        });

        tasks.push(task);
    }

    // Wait for all tasks to complete
    let mut results = Vec::new();
    for task in tasks {
        let result = task.await.expect("Task panicked")?;
        results.push(result);
    }

    // Verify all tasks got identical results
    assert_eq!(results.len(), 10, "Should have 10 completed tasks");
    for (task_id, row_count, ids) in results {
        assert_eq!(row_count, 4, "Task {} row count mismatch", task_id);
        assert_eq!(ids, vec![1, 2, 3, 4], "Task {} IDs mismatch", task_id);
    }

    eprintln!("✓ All 10 concurrent SELECT queries returned correct results");

    Ok(())
}

/// Test concurrent COUNT queries
///
/// This test verifies that aggregation queries (COUNT) work correctly
/// when executed concurrently. COUNT queries follow a different execution
/// path with optimizations for zero-column batches.
#[tokio::test]
async fn test_concurrent_count_queries() -> DataFusionResult<()> {
    let temp_dir =
        TempDir::new().map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
    let catalog_path = temp_dir.path().join("with_deletes.ducklake");

    // Generate test data
    common::create_catalog_with_deletes(&catalog_path).map_err(common::to_datafusion_error)?;

    let provider = DuckdbMetadataProvider::new(catalog_path.to_string_lossy().to_string())
        .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
    let catalog = Arc::new(
        DuckLakeCatalog::new(provider)
            .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?,
    );

    // Create 10 concurrent COUNT query tasks
    let mut tasks = Vec::new();
    for task_id in 0..10 {
        let catalog_clone = Arc::clone(&catalog);

        let task = tokio::spawn(async move {
            let ctx = SessionContext::new();
            ctx.register_catalog("with_deletes", catalog_clone);

            // Execute COUNT query (should exclude deleted rows)
            let df = ctx
                .sql("SELECT COUNT(*) as count FROM with_deletes.main.products")
                .await?;
            let results = df.collect().await?;

            assert!(!results.is_empty());
            let batch = &results[0];
            let counts = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();

            let count = counts.value(0);
            assert_eq!(count, 3, "Task {} got wrong count", task_id);

            Ok::<_, DataFusionError>((task_id, count))
        });

        tasks.push(task);
    }

    // Wait for all tasks
    let mut results = Vec::new();
    for task in tasks {
        let result = task.await.expect("Task panicked")?;
        results.push(result);
    }

    // Verify all tasks got the same count
    assert_eq!(results.len(), 10);
    for (task_id, count) in results {
        assert_eq!(count, 3, "Task {} count mismatch", task_id);
    }

    eprintln!("✓ All 10 concurrent COUNT queries returned correct results");

    Ok(())
}

/// Test concurrent mixed queries (SELECT, COUNT, aggregations)
///
/// This test executes different types of queries concurrently to verify
/// that the system remains stable under heterogeneous query workloads.
#[tokio::test]
async fn test_concurrent_mixed_queries() -> DataFusionResult<()> {
    let temp_dir =
        TempDir::new().map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
    let catalog_path = temp_dir.path().join("with_updates.ducklake");

    // Generate test data
    common::create_catalog_with_updates(&catalog_path).map_err(common::to_datafusion_error)?;

    let provider = DuckdbMetadataProvider::new(catalog_path.to_string_lossy().to_string())
        .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
    let catalog = Arc::new(
        DuckLakeCatalog::new(provider)
            .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?,
    );

    // Define different query types
    let queries = [
        (
            "SELECT COUNT(*) as count FROM with_updates.main.inventory",
            "count",
        ),
        (
            "SELECT SUM(quantity) as total FROM with_updates.main.inventory",
            "sum",
        ),
        (
            "SELECT * FROM with_updates.main.inventory WHERE id = 1",
            "filter_1",
        ),
        (
            "SELECT * FROM with_updates.main.inventory WHERE id = 3",
            "filter_3",
        ),
        (
            "SELECT id, quantity FROM with_updates.main.inventory ORDER BY id",
            "ordered",
        ),
    ];

    // Execute each query type 2 times concurrently (10 total tasks)
    let mut tasks = Vec::new();
    for i in 0..10 {
        let (query, query_type) = queries[i % queries.len()];
        let catalog_clone = Arc::clone(&catalog);
        let query_string = query.to_string();
        let query_type_string = query_type.to_string();

        let task = tokio::spawn(async move {
            let ctx = SessionContext::new();
            ctx.register_catalog("with_updates", catalog_clone);

            let df = ctx.sql(&query_string).await?;
            let results = df.collect().await?;

            // Basic validation based on query type
            let validation = match query_type_string.as_str() {
                "count" => {
                    let batch = &results[0];
                    let counts = batch
                        .column(0)
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .unwrap();
                    counts.value(0) == 3
                },
                "sum" => {
                    let batch = &results[0];
                    let totals = batch
                        .column(0)
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .unwrap();
                    totals.value(0) == 500 // 120 + 200 + 180
                },
                "filter_1" | "filter_3" => {
                    let total_rows: usize = results.iter().map(|b| b.num_rows()).sum();
                    total_rows == 1
                },
                "ordered" => {
                    let total_rows: usize = results.iter().map(|b| b.num_rows()).sum();
                    total_rows == 3
                },
                _ => false,
            };

            Ok::<_, DataFusionError>((i, query_type_string, validation))
        });

        tasks.push(task);
    }

    // Wait for all tasks
    let mut results = Vec::new();
    for task in tasks {
        let result = task.await.expect("Task panicked")?;
        results.push(result);
    }

    // Verify all tasks succeeded
    assert_eq!(results.len(), 10);
    for (task_id, query_type, valid) in results {
        assert!(valid, "Task {} ({}) validation failed", task_id, query_type);
    }

    eprintln!("✓ All 10 concurrent mixed queries returned correct results");

    Ok(())
}

/// Test concurrent queries with delete file filtering
///
/// This test specifically targets the delete filter execution path,
/// verifying that concurrent queries correctly apply delete filters
/// without race conditions.
#[tokio::test]
async fn test_concurrent_delete_filtering() -> DataFusionResult<()> {
    let temp_dir =
        TempDir::new().map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
    let catalog_path = temp_dir.path().join("with_deletes.ducklake");

    // Generate test data
    common::create_catalog_with_deletes(&catalog_path).map_err(common::to_datafusion_error)?;

    let provider = DuckdbMetadataProvider::new(catalog_path.to_string_lossy().to_string())
        .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
    let catalog = Arc::new(
        DuckLakeCatalog::new(provider)
            .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?,
    );

    // Create 10 concurrent tasks querying table with deletes
    let mut tasks = Vec::new();
    for task_id in 0..10 {
        let catalog_clone = Arc::clone(&catalog);

        let task = tokio::spawn(async move {
            let ctx = SessionContext::new();
            ctx.register_catalog("with_deletes", catalog_clone);

            // Query all rows (should exclude deleted IDs 2 and 4)
            let df = ctx
                .sql("SELECT * FROM with_deletes.main.products ORDER BY id")
                .await?;
            let results = df.collect().await?;

            let total_rows: usize = results.iter().map(|b| b.num_rows()).sum();
            assert_eq!(total_rows, 3, "Task {} got wrong row count", task_id);

            // Collect IDs to verify deleted rows are excluded
            let mut all_ids = Vec::new();
            for batch in &results {
                all_ids.extend(get_int_column(batch, 0));
            }
            assert_eq!(all_ids, vec![1, 3, 5], "Task {} got wrong IDs", task_id);

            // Also verify that deleted rows are truly absent
            let df_deleted = ctx
                .sql("SELECT * FROM with_deletes.main.products WHERE id = 2")
                .await?;
            let deleted_results = df_deleted.collect().await?;
            let deleted_count: usize = deleted_results.iter().map(|b| b.num_rows()).sum();
            assert_eq!(deleted_count, 0, "Task {} found deleted row", task_id);

            Ok::<_, DataFusionError>((task_id, total_rows, all_ids))
        });

        tasks.push(task);
    }

    // Wait for all tasks
    let mut results = Vec::new();
    for task in tasks {
        let result = task.await.expect("Task panicked")?;
        results.push(result);
    }

    // Verify all tasks got identical results
    assert_eq!(results.len(), 10);
    for (task_id, row_count, ids) in results {
        assert_eq!(row_count, 3, "Task {} row count mismatch", task_id);
        assert_eq!(ids, vec![1, 3, 5], "Task {} IDs mismatch", task_id);
    }

    eprintln!("✓ All 10 concurrent queries correctly filtered deleted rows");

    Ok(())
}

/// Test concurrent catalog metadata access
///
/// This test verifies that concurrent access to catalog metadata
/// (schema listing, table listing) is thread-safe.
#[tokio::test]
async fn test_concurrent_metadata_access() -> DataFusionResult<()> {
    let temp_dir =
        TempDir::new().map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
    let catalog_path = temp_dir.path().join("no_deletes.ducklake");

    // Generate test data
    common::create_catalog_no_deletes(&catalog_path).map_err(common::to_datafusion_error)?;

    let provider = DuckdbMetadataProvider::new(catalog_path.to_string_lossy().to_string())
        .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
    let catalog = Arc::new(
        DuckLakeCatalog::new(provider)
            .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?,
    );

    // Create 10 tasks that access catalog metadata concurrently
    let mut tasks = Vec::new();
    for task_id in 0..10 {
        let catalog_clone = Arc::clone(&catalog);

        let task = tokio::spawn(async move {
            let ctx = SessionContext::new();
            let catalog_name = "no_deletes";
            ctx.register_catalog(catalog_name, catalog_clone);

            let schema_names = ctx.catalog(catalog_name).unwrap().schema_names();

            // Should have information_schema and main
            assert_eq!(2, schema_names.len());
            assert_eq!(schema_names, vec!["information_schema", "main"]);

            // Check main schema has users table
            let main_table_names = ctx
                .catalog(catalog_name)
                .unwrap()
                .schema("main")
                .unwrap()
                .table_names();
            assert_eq!(1, main_table_names.len());
            assert_eq!(main_table_names, vec!["users"]);

            Ok::<_, DataFusionError>(task_id)
        });

        tasks.push(task);
    }

    // Wait for all tasks
    let mut results = Vec::new();
    for task in tasks {
        let result = task.await.expect("Task panicked")?;
        results.push(result);
    }

    assert_eq!(results.len(), 10);
    eprintln!("✓ All 10 concurrent metadata access operations completed successfully");

    Ok(())
}

/// Stress test: Many concurrent queries with different access patterns
///
/// This test creates 20 concurrent tasks with varied query patterns
/// to stress-test the system under high concurrent load.
#[tokio::test]
async fn test_stress_concurrent_queries() -> DataFusionResult<()> {
    let temp_dir =
        TempDir::new().map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
    let catalog_path = temp_dir.path().join("with_deletes.ducklake");

    // Generate test data
    common::create_catalog_with_deletes(&catalog_path).map_err(common::to_datafusion_error)?;

    let provider = DuckdbMetadataProvider::new(catalog_path.to_string_lossy().to_string())
        .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
    let catalog = Arc::new(
        DuckLakeCatalog::new(provider)
            .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?,
    );

    // Create 20 concurrent tasks
    let mut tasks = Vec::new();
    for task_id in 0..20 {
        let catalog_clone = Arc::clone(&catalog);

        let task = tokio::spawn(async move {
            let ctx = SessionContext::new();
            ctx.register_catalog("with_deletes", catalog_clone);

            // Vary query patterns based on task_id
            let result = match task_id % 4 {
                0 => {
                    // Full table scan
                    let df = ctx
                        .sql("SELECT * FROM with_deletes.main.products ORDER BY id")
                        .await?;
                    let results = df.collect().await?;
                    let row_count: usize = results.iter().map(|b| b.num_rows()).sum();
                    row_count == 3
                },
                1 => {
                    // Count query
                    let df = ctx
                        .sql("SELECT COUNT(*) FROM with_deletes.main.products")
                        .await?;
                    let results = df.collect().await?;
                    let batch = &results[0];
                    let counts = batch
                        .column(0)
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .unwrap();
                    counts.value(0) == 3
                },
                2 => {
                    // Filtered query
                    let df = ctx
                        .sql("SELECT * FROM with_deletes.main.products WHERE id > 2")
                        .await?;
                    let results = df.collect().await?;
                    let row_count: usize = results.iter().map(|b| b.num_rows()).sum();
                    row_count == 2 // ids 3 and 5
                },
                3 => {
                    // Query for deleted row (should return 0)
                    let df = ctx
                        .sql("SELECT * FROM with_deletes.main.products WHERE id = 4")
                        .await?;
                    let results = df.collect().await?;
                    let row_count: usize = results.iter().map(|b| b.num_rows()).sum();
                    row_count == 0
                },
                _ => unreachable!(),
            };

            Ok::<_, DataFusionError>((task_id, result))
        });

        tasks.push(task);
    }

    // Wait for all tasks
    let mut results = Vec::new();
    for task in tasks {
        let result = task.await.expect("Task panicked")?;
        results.push(result);
    }

    // Verify all tasks succeeded
    assert_eq!(results.len(), 20);
    for (task_id, valid) in results {
        assert!(valid, "Task {} validation failed", task_id);
    }

    eprintln!("✓ All 20 concurrent stress test queries completed successfully");

    Ok(())
}
