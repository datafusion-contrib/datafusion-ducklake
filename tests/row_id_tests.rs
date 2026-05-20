#![cfg(feature = "metadata-duckdb")]
//! End-to-end tests for DuckLake row-lineage (`rowid` virtual column).
//!
//! These tests build small DuckLake catalogs via the DuckDB CLI (so the
//! catalog tables, including `data_file.row_id_start`, are populated by the
//! official extension) and then query through DataFusion to verify our
//! injected `rowid` column matches what DuckDB itself would return.

mod common;

use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use arrow::array::{Array, Int32Array, Int64Array};
use arrow::record_batch::RecordBatch;
use datafusion::error::Result as DataFusionResult;
use datafusion::prelude::*;
use datafusion_ducklake::{DuckLakeCatalog, DuckdbMetadataProvider};
use tempfile::TempDir;

/// Build a small two-file catalog: two separate INSERT statements produce two
/// data files, each with its own `row_id_start`.
fn create_catalog_rowid_two_files(catalog_path: &Path) -> Result<()> {
    let conn = duckdb::Connection::open_in_memory()?;
    conn.execute("INSTALL ducklake;", [])?;
    conn.execute("INSTALL parquet;", [])?;
    conn.execute("LOAD ducklake;", [])?;

    let ducklake_path = format!("ducklake:{}", catalog_path.display());
    conn.execute(&format!("ATTACH '{}' AS c;", ducklake_path), [])?;

    conn.execute("CREATE TABLE c.t(i INTEGER);", [])?;
    // First file: rows 0..3
    conn.execute(
        "INSERT INTO c.t SELECT i FROM range(0, 3) t(i);",
        [],
    )?;
    // Second file: rows 10..15
    conn.execute(
        "INSERT INTO c.t SELECT i FROM range(10, 15) t(i);",
        [],
    )?;
    Ok(())
}

/// Same as the two-file catalog, but DELETE the rows where i is odd.
/// Verifies that DeleteFilterExec runs after RowIdExec so deleted rowids
/// are correctly elided from the output.
fn create_catalog_rowid_with_deletes(catalog_path: &Path) -> Result<()> {
    let conn = duckdb::Connection::open_in_memory()?;
    conn.execute("INSTALL ducklake;", [])?;
    conn.execute("INSTALL parquet;", [])?;
    conn.execute("LOAD ducklake;", [])?;

    let ducklake_path = format!("ducklake:{}", catalog_path.display());
    conn.execute(&format!("ATTACH '{}' AS c;", ducklake_path), [])?;

    conn.execute("CREATE TABLE c.t(i INTEGER);", [])?;
    conn.execute("INSERT INTO c.t SELECT i FROM range(0, 3) t(i);", [])?;
    conn.execute("INSERT INTO c.t SELECT i FROM range(10, 15) t(i);", [])?;
    // Same DELETE pattern as the reference DuckLake rowid test.
    conn.execute("DELETE FROM c.t WHERE i % 2 = 1;", [])?;
    Ok(())
}

/// Mirrors the UPDATE portion of test/sql/rowid/ducklake_row_id.test in the
/// upstream DuckLake extension. The UPDATE rewrites surviving rows into a
/// new data file whose parquet stores `_ducklake_internal_row_id` (field-id
/// 2147483540) so the original rowids survive the rewrite.
///
/// Trajectory (rowid in parentheses):
///   INSERT range(0, 3)   → (0,0) (1,1) (2,2)
///   INSERT range(10, 15) → (3,10) (4,11) (5,12) (6,13) (7,14)
///   DELETE WHERE i % 2 = 1 → drops rowids {1, 4, 6}
///     Surviving: (0,0) (2,2) (3,10) (5,12) (7,14)
///   UPDATE i = i+1000 WHERE i<3 OR i>10 → touches rowids {0, 2, 5, 7}:
///     → (0,1000) (2,1002) (3,10) (5,1012) (7,1014)
///
/// The expected result is only achievable if the rewrite file's embedded
/// `_ducklake_internal_row_id` column is read; `row_id_start + position`
/// for the rewrite file would yield a fresh sequential range, not {0,2,5,7}.
fn create_catalog_rowid_with_update(catalog_path: &Path) -> Result<()> {
    let conn = duckdb::Connection::open_in_memory()?;
    conn.execute("INSTALL ducklake;", [])?;
    conn.execute("INSTALL parquet;", [])?;
    conn.execute("LOAD ducklake;", [])?;

    let ducklake_path = format!("ducklake:{}", catalog_path.display());
    conn.execute(&format!("ATTACH '{}' AS c;", ducklake_path), [])?;

    conn.execute("CREATE TABLE c.t(i INTEGER);", [])?;
    conn.execute("INSERT INTO c.t SELECT i FROM range(0, 3) t(i);", [])?;
    conn.execute("INSERT INTO c.t SELECT i FROM range(10, 15) t(i);", [])?;
    conn.execute("DELETE FROM c.t WHERE i % 2 = 1;", [])?;
    // Force a file rewrite that preserves original rowids.
    conn.execute("UPDATE c.t SET i = i + 1000 WHERE i < 3 OR i > 10;", [])?;
    Ok(())
}

fn make_catalog(path: &str, row_lineage: bool) -> DataFusionResult<Arc<DuckLakeCatalog>> {
    let provider = DuckdbMetadataProvider::new(path)
        .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
    let catalog = DuckLakeCatalog::new(provider)
        .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?
        .with_row_lineage(row_lineage);
    Ok(Arc::new(catalog))
}

fn collect_rowid_i_sorted(batches: &[RecordBatch]) -> Vec<(i64, i32)> {
    let mut out = Vec::new();
    for batch in batches {
        let rowids = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("first column should be rowid (Int64)");
        let vals = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("second column should be i (Int32)");
        for r in 0..batch.num_rows() {
            assert!(!rowids.is_null(r), "rowid should not be null");
            out.push((rowids.value(r), vals.value(r)));
        }
    }
    out.sort_by_key(|&(rowid, _)| rowid);
    out
}

#[tokio::test]
async fn rowid_disabled_by_default() -> DataFusionResult<()> {
    let temp = TempDir::new()
        .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
    let path = temp.path().join("rowid_default.ducklake");
    create_catalog_rowid_two_files(&path).map_err(common::to_datafusion_error)?;

    // Default: row lineage is OFF.
    let catalog = make_catalog(&path.to_string_lossy(), false)?;
    let ctx = SessionContext::new();
    ctx.register_catalog("c", catalog);

    // SELECT * should not expose rowid.
    let df = ctx.sql("SELECT * FROM c.main.t ORDER BY i").await?;
    let schema = df.schema().clone();
    let field_names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    assert_eq!(
        field_names,
        vec!["i"],
        "rowid must not appear when row lineage is disabled (got {:?})",
        field_names
    );

    // Explicitly referencing rowid should fail to plan.
    let err = ctx
        .sql("SELECT rowid FROM c.main.t")
        .await
        .err()
        .expect("should fail to plan rowid reference with lineage off");
    assert!(
        format!("{err}").to_lowercase().contains("rowid"),
        "expected error to mention rowid, got: {err}"
    );

    Ok(())
}

#[tokio::test]
async fn rowid_sequential_across_files() -> DataFusionResult<()> {
    let temp = TempDir::new()
        .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
    let path = temp.path().join("rowid_two_files.ducklake");
    create_catalog_rowid_two_files(&path).map_err(common::to_datafusion_error)?;

    let catalog = make_catalog(&path.to_string_lossy(), true)?;
    let ctx = SessionContext::new();
    ctx.register_catalog("c", catalog);

    // Verify rowid IS in the table schema and is BIGINT.
    let df = ctx.sql("SELECT * FROM c.main.t").await?;
    let schema = df.schema().clone();
    let field_names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    assert_eq!(field_names, vec!["i", "rowid"]);

    // Query rowid + i; rowid values should be globally unique and contiguous
    // starting at 0 (DuckLake assigns sequentially across files inserted in
    // order).
    let df = ctx
        .sql("SELECT rowid, i FROM c.main.t ORDER BY rowid")
        .await?;
    let batches = df.collect().await?;

    let pairs = collect_rowid_i_sorted(&batches);
    assert_eq!(
        pairs,
        vec![
            (0, 0),
            (1, 1),
            (2, 2),
            (3, 10),
            (4, 11),
            (5, 12),
            (6, 13),
            (7, 14),
        ],
        "rowids should be contiguous 0..8 across the two files",
    );

    // Spot-check WHERE rowid = N
    let df = ctx
        .sql("SELECT rowid, i FROM c.main.t WHERE rowid = 4")
        .await?;
    let batches = df.collect().await?;
    let pairs = collect_rowid_i_sorted(&batches);
    assert_eq!(pairs, vec![(4, 11)]);

    Ok(())
}

#[tokio::test]
async fn rowid_preserved_under_deletes() -> DataFusionResult<()> {
    let temp = TempDir::new()
        .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
    let path = temp.path().join("rowid_with_deletes.ducklake");
    create_catalog_rowid_with_deletes(&path).map_err(common::to_datafusion_error)?;

    let catalog = make_catalog(&path.to_string_lossy(), true)?;
    let ctx = SessionContext::new();
    ctx.register_catalog("c", catalog);

    let df = ctx
        .sql("SELECT rowid, i FROM c.main.t ORDER BY rowid")
        .await?;
    let batches = df.collect().await?;
    let pairs = collect_rowid_i_sorted(&batches);

    // Source rows (rowid, i): (0,0) (1,1) (2,2) (3,10) (4,11) (5,12) (6,13) (7,14)
    // Deleted where i is odd: i ∈ {1, 11, 13}, leaving rowids {0, 2, 3, 5, 7}.
    assert_eq!(
        pairs,
        vec![(0, 0), (2, 2), (3, 10), (5, 12), (7, 14)],
        "deleted rows' rowids must be elided, surviving rowids unchanged",
    );

    Ok(())
}

#[tokio::test]
async fn rowid_only_projection() -> DataFusionResult<()> {
    // Edge case: physical projection is empty (just rowid). Verifies that
    // RowIdExec works when ParquetExec emits zero-column count batches.
    let temp = TempDir::new()
        .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
    let path = temp.path().join("rowid_only.ducklake");
    create_catalog_rowid_two_files(&path).map_err(common::to_datafusion_error)?;

    let catalog = make_catalog(&path.to_string_lossy(), true)?;
    let ctx = SessionContext::new();
    ctx.register_catalog("c", catalog);

    let df = ctx.sql("SELECT rowid FROM c.main.t ORDER BY rowid").await?;
    let batches = df.collect().await?;
    let mut all = Vec::new();
    for batch in &batches {
        let rowids = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("rowid column should be Int64");
        for r in 0..batch.num_rows() {
            assert!(!rowids.is_null(r));
            all.push(rowids.value(r));
        }
    }
    all.sort();
    assert_eq!(all, vec![0, 1, 2, 3, 4, 5, 6, 7]);
    Ok(())
}

#[tokio::test]
async fn rowid_preserved_across_update_rewrite() -> DataFusionResult<()> {
    // The critical test: UPDATE rewrites the file with embedded rowids.
    // Our scan must read those embedded values rather than compute
    // `row_id_start + position`, otherwise the rowids drift.
    let temp = TempDir::new()
        .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
    let path = temp.path().join("rowid_update.ducklake");
    create_catalog_rowid_with_update(&path).map_err(common::to_datafusion_error)?;

    let catalog = make_catalog(&path.to_string_lossy(), true)?;
    let ctx = SessionContext::new();
    ctx.register_catalog("c", catalog);

    let df = ctx
        .sql("SELECT rowid, i FROM c.main.t ORDER BY rowid")
        .await?;
    let batches = df.collect().await?;
    let pairs = collect_rowid_i_sorted(&batches);

    assert_eq!(
        pairs,
        vec![(0, 1000), (2, 1002), (3, 10), (5, 1012), (7, 1014)],
        "rowids must survive the UPDATE rewrite; values reflect i+1000 \
         applied to original rowids 0, 2, 5, 7 (file-rewritten rows)",
    );

    // Also verify per-rowid lookup is correct after the rewrite.
    let df = ctx
        .sql("SELECT rowid, i FROM c.main.t WHERE rowid = 5")
        .await?;
    let batches = df.collect().await?;
    let pairs = collect_rowid_i_sorted(&batches);
    assert_eq!(pairs, vec![(5, 1012)]);

    Ok(())
}

#[tokio::test]
async fn rowid_count_star_unaffected() -> DataFusionResult<()> {
    // COUNT(*) should not trigger the rowid path (it asks for zero columns).
    let temp = TempDir::new()
        .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
    let path = temp.path().join("rowid_count_star.ducklake");
    create_catalog_rowid_two_files(&path).map_err(common::to_datafusion_error)?;

    let catalog = make_catalog(&path.to_string_lossy(), true)?;
    let ctx = SessionContext::new();
    ctx.register_catalog("c", catalog);

    let df = ctx.sql("SELECT COUNT(*) FROM c.main.t").await?;
    let batches = df.collect().await?;
    let total: i64 = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!(total, 8);
    Ok(())
}
