#![cfg(feature = "metadata-duckdb")]
//! Common test utilities for integration tests
//!
//! This module provides helper functions to generate DuckLake catalogs
//! for testing purposes. Each function creates a persistent DuckDB database
//! file with DuckLake extension, populates it with test data, and returns
//! the path for use with DuckdbMetadataProvider.

#![allow(dead_code)]

use anyhow::Result;
use std::path::Path;
use std::sync::Once;

static INSTALL_DUCKLAKE: Once = Once::new();

/// Install the ducklake and parquet extensions exactly once (thread-safe across parallel tests).
///
/// Pre-installing parquet avoids race conditions when multiple concurrent tests
/// trigger DuckDB's auto-install simultaneously (observed on macOS CI).
fn ensure_ducklake_installed() {
    INSTALL_DUCKLAKE.call_once(|| {
        let conn = duckdb::Connection::open_in_memory().expect("open in-memory duckdb");
        conn.execute("INSTALL ducklake;", [])
            .expect("install ducklake extension");
        conn.execute("INSTALL parquet;", [])
            .expect("install parquet extension");
    });
}

/// Creates a catalog with a simple table (no deletes)
///
/// Table schema:
/// - users (id INT, name VARCHAR, email VARCHAR)
/// - 4 rows: Alice, Bob, Charlie, Diana
///
/// # Example
///
/// ```
/// use tempfile::TempDir;
/// let temp_dir = TempDir::new()?;
/// let catalog_path = temp_dir.path().join("no_deletes.ducklake");
/// create_catalog_no_deletes(&catalog_path)?;
/// ```
pub fn create_catalog_no_deletes(catalog_path: &Path) -> Result<()> {
    // Use in-memory database to avoid file locking issues
    let conn = duckdb::Connection::open_in_memory()?;

    ensure_ducklake_installed();
    conn.execute("LOAD ducklake;", [])?;

    let ducklake_path = format!("ducklake:{}", catalog_path.display());
    conn.execute(&format!("ATTACH '{}' AS test_catalog;", ducklake_path), [])?;

    conn.execute(
        "CREATE TABLE test_catalog.users (
            id INT,
            name VARCHAR,
            email VARCHAR
        );",
        [],
    )?;

    conn.execute(
        "INSERT INTO test_catalog.users VALUES
            (1, 'Alice', 'alice@example.com'),
            (2, 'Bob', 'bob@example.com'),
            (3, 'Charlie', 'charlie@example.com'),
            (4, 'Diana', 'diana@example.com');",
        [],
    )?;

    Ok(())
}

/// Creates a catalog with DELETE operations
///
/// Table schema:
/// - products (id INT, name VARCHAR, price DECIMAL(10,2), in_stock BOOLEAN)
/// - 5 rows inserted, 2 deleted (ids 2 and 4)
/// - Final result: 3 rows (ids 1, 3, 5)
///
/// This catalog demonstrates delete file functionality where some rows
/// are marked as deleted via DuckLake's delete files.
pub fn create_catalog_with_deletes(catalog_path: &Path) -> Result<()> {
    // Use in-memory database to avoid file locking issues
    let conn = duckdb::Connection::open_in_memory()?;

    ensure_ducklake_installed();
    conn.execute("LOAD ducklake;", [])?;

    let ducklake_path = format!("ducklake:{}", catalog_path.display());
    conn.execute(&format!("ATTACH '{}' AS test_catalog;", ducklake_path), [])?;

    conn.execute(
        "CREATE TABLE test_catalog.products (
            id INT,
            name VARCHAR,
            price DECIMAL(10,2),
            in_stock BOOLEAN
        );",
        [],
    )?;

    conn.execute(
        "INSERT INTO test_catalog.products VALUES
            (1, 'Laptop', 999.99, true),
            (2, 'Mouse', 25.50, true),
            (3, 'Keyboard', 75.00, true),
            (4, 'Monitor', 299.99, false),
            (5, 'Webcam', 89.99, true);",
        [],
    )?;

    // Create delete files
    conn.execute("DELETE FROM test_catalog.products WHERE id = 2;", [])?;
    conn.execute("DELETE FROM test_catalog.products WHERE id = 4;", [])?;

    Ok(())
}

/// Creates a catalog with UPDATE operations
///
/// Table schema:
/// - inventory (id INT, product_name VARCHAR, quantity INT, last_updated TIMESTAMP)
/// - 3 rows inserted, 2 updated (ids 1 and 3)
/// - Updates create delete files for old versions
///
/// This demonstrates the MOR (Merge-On-Read) pattern where UPDATEs
/// create delete files for old row versions.
pub fn create_catalog_with_updates(catalog_path: &Path) -> Result<()> {
    // Use in-memory database to avoid file locking issues
    let conn = duckdb::Connection::open_in_memory()?;

    ensure_ducklake_installed();
    conn.execute("LOAD ducklake;", [])?;

    let ducklake_path = format!("ducklake:{}", catalog_path.display());
    conn.execute(&format!("ATTACH '{}' AS test_catalog;", ducklake_path), [])?;

    conn.execute(
        "CREATE TABLE test_catalog.inventory (
            id INT,
            product_name VARCHAR,
            quantity INT,
            last_updated TIMESTAMP
        );",
        [],
    )?;

    conn.execute(
        "INSERT INTO test_catalog.inventory VALUES
            (1, 'Widget A', 100, '2024-01-01 10:00:00'),
            (2, 'Widget B', 200, '2024-01-01 10:00:00'),
            (3, 'Widget C', 150, '2024-01-01 10:00:00');",
        [],
    )?;

    // Create delete files via updates
    conn.execute(
        "UPDATE test_catalog.inventory
         SET quantity = 120, last_updated = '2024-01-02 15:30:00'
         WHERE id = 1;",
        [],
    )?;

    conn.execute(
        "UPDATE test_catalog.inventory
         SET quantity = 180, last_updated = '2024-01-02 16:00:00'
         WHERE id = 3;",
        [],
    )?;

    Ok(())
}

/// Creates a catalog for filter pushdown correctness testing
///
/// Table schema:
/// - items (id INT, value VARCHAR)
/// - 5 rows: [1,2,3,4,5]
/// - Row with id=3 deleted
///
/// This tests that WHERE filters are applied AFTER delete filtering,
/// ensuring correct query semantics.
pub fn create_catalog_filter_pushdown(catalog_path: &Path) -> Result<()> {
    // Use in-memory database to avoid file locking issues
    let conn = duckdb::Connection::open_in_memory()?;

    ensure_ducklake_installed();
    conn.execute("LOAD ducklake;", [])?;

    let ducklake_path = format!("ducklake:{}", catalog_path.display());
    conn.execute(&format!("ATTACH '{}' AS test_catalog;", ducklake_path), [])?;

    conn.execute(
        "CREATE TABLE test_catalog.items (
            id INT,
            value VARCHAR
        );",
        [],
    )?;

    conn.execute(
        "INSERT INTO test_catalog.items VALUES
            (1, 'one'),
            (2, 'two'),
            (3, 'three'),
            (4, 'four'),
            (5, 'five');",
        [],
    )?;

    // Delete id=3 to test filter application order
    conn.execute("DELETE FROM test_catalog.items WHERE id = 3;", [])?;

    Ok(())
}

/// Creates a catalog with an empty table (no data)
///
/// Table schema:
/// - tbl (i INT)
/// - 0 rows (after inserting and deleting)
pub fn create_catalog_empty_table(catalog_path: &Path) -> Result<()> {
    let conn = duckdb::Connection::open_in_memory()?;

    ensure_ducklake_installed();
    conn.execute("LOAD ducklake;", [])?;

    let ducklake_path = format!("ducklake:{}", catalog_path.display());
    conn.execute(&format!("ATTACH '{}' AS test_catalog;", ducklake_path), [])?;

    conn.execute(
        "CREATE TABLE test_catalog.tbl (
            i INTEGER
        );",
        [],
    )?;

    // Insert a dummy row to create the file structure
    conn.execute("INSERT INTO test_catalog.tbl VALUES (1);", [])?;

    // Delete it to make the table effectively empty
    conn.execute("DELETE FROM test_catalog.tbl WHERE i = 1;", [])?;

    Ok(())
}

/// Creates a catalog matching ducklake_basic.test scenario
///
/// Tables:
/// - test (i INT, j INT) with 4 rows: (1,2), (NULL,3), (4,5), (6,7)
/// - test2 (j VARCHAR, date DATE) with 1 row: ('hello world', '1992-01-01')
pub fn create_catalog_basic_test(catalog_path: &Path) -> Result<()> {
    let conn = duckdb::Connection::open_in_memory()?;

    ensure_ducklake_installed();
    conn.execute("LOAD ducklake;", [])?;

    let ducklake_path = format!("ducklake:{}", catalog_path.display());
    conn.execute(&format!("ATTACH '{}' AS test_catalog;", ducklake_path), [])?;

    // Create first table: test(i INTEGER, j INTEGER)
    conn.execute(
        "CREATE TABLE test_catalog.test (
            i INTEGER,
            j INTEGER
        );",
        [],
    )?;

    // Insert data in two batches (as in original test)
    conn.execute(
        "INSERT INTO test_catalog.test VALUES (1, 2), (NULL, 3);",
        [],
    )?;

    conn.execute("INSERT INTO test_catalog.test VALUES (4, 5), (6, 7);", [])?;

    // Create second table: test2 with VARCHAR and DATE
    conn.execute(
        "CREATE TABLE test_catalog.test2 AS
         SELECT 'hello world' AS j, DATE '1992-01-01' AS date;",
        [],
    )?;

    Ok(())
}

/// Helper to convert anyhow errors to DataFusion errors
///
/// This is useful for converting anyhow::Error to DataFusionError in test code.
pub fn to_datafusion_error(e: anyhow::Error) -> datafusion::error::DataFusionError {
    datafusion::error::DataFusionError::External(e.into())
}

/// Creates a catalog with complex deletion scenario for testing table_deletions
///
/// This creates a table and performs:
/// - Insert initial rows (1, 2, 3)
/// - Delete all rows
/// - Insert more rows (4, 5, 6, 7)
/// - Delete some rows (5, 6)
///
/// Total deletions: 5 rows (1, 2, 3, 5, 6)
pub fn create_catalog_complex_deletions(catalog_path: &Path) -> Result<()> {
    let conn = duckdb::Connection::open_in_memory()?;

    ensure_ducklake_installed();
    conn.execute("LOAD ducklake;", [])?;

    let ducklake_path = format!("ducklake:{}", catalog_path.display());
    conn.execute(&format!("ATTACH '{}' AS test_catalog;", ducklake_path), [])?;

    // Create table
    conn.execute(
        "CREATE TABLE test_catalog.items (
            id INT
        );",
        [],
    )?;

    // Insert initial rows (1, 2, 3)
    conn.execute("INSERT INTO test_catalog.items VALUES (1), (2), (3);", [])?;

    // Delete all rows
    conn.execute("DELETE FROM test_catalog.items;", [])?;

    // Insert more rows (4, 5, 6, 7)
    conn.execute(
        "INSERT INTO test_catalog.items VALUES (4), (5), (6), (7);",
        [],
    )?;

    // Delete some rows (5, 6)
    conn.execute("DELETE FROM test_catalog.items WHERE id IN (5, 6);", [])?;

    Ok(())
}

/// Creates a catalog with multiple snapshots for testing table_changes
///
/// This creates a table and performs operations across multiple snapshots:
/// - Snapshot 1: Initial table creation + first batch insert (rows 1-3)
/// - Snapshot 2: Second batch insert (rows 4-5)
/// - Snapshot 3: Delete row 2
///
/// This allows testing ducklake_table_changes() to see:
/// - Files added between snapshots 0-1 (initial insert)
/// - Files added between snapshots 1-2 (second insert)
/// - Files added between snapshots 2-3 (delete file)
pub fn create_catalog_multiple_snapshots(catalog_path: &Path) -> Result<()> {
    let conn = duckdb::Connection::open_in_memory()?;

    ensure_ducklake_installed();
    conn.execute("LOAD ducklake;", [])?;

    let ducklake_path = format!("ducklake:{}", catalog_path.display());
    conn.execute(&format!("ATTACH '{}' AS test_catalog;", ducklake_path), [])?;

    // Snapshot 1: Create table and insert first batch
    conn.execute(
        "CREATE TABLE test_catalog.events (
            id INT,
            event_type VARCHAR,
            value INT
        );",
        [],
    )?;

    conn.execute(
        "INSERT INTO test_catalog.events VALUES
            (1, 'click', 100),
            (2, 'view', 50),
            (3, 'purchase', 500);",
        [],
    )?;

    // Snapshot 2: Insert second batch
    conn.execute(
        "INSERT INTO test_catalog.events VALUES
            (4, 'click', 75),
            (5, 'view', 25);",
        [],
    )?;

    // Snapshot 3: Delete a row
    conn.execute("DELETE FROM test_catalog.events WHERE id = 2;", [])?;

    Ok(())
}
