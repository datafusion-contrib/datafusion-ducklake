#![cfg(feature = "metadata-duckdb")]
//! Regression tests for issue #178: `ducklake_table_deletions` must return the
//! correct deleted rows regardless of how DataFusion parallelizes the scans.
//!
//! Pre-fix, `DeletedRowsExec` inherited its data scan's partitioning and
//! matched deleted positions by stream arrival order. On a multicore machine
//! the physical optimizer inserts round-robin repartitions above the internal
//! scans (no special settings needed), sending the delete-position set to one
//! partition and the matching data batches to others — deletions were silently
//! missed, and positions matched in the wrong partition emitted the wrong
//! row's content. All tests here FAIL on the pre-fix code.

mod common;

use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{Int32Array, Int64Array, StringArray};
use datafusion::config::ConfigOptions;
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::prelude::*;
use datafusion_ducklake::{DuckdbMetadataProvider, register_ducklake_functions};
use tempfile::TempDir;

/// Rows-per-file large enough to span many record batches and several Parquet
/// row groups (DuckDB's default row-group size is 122_880).
const BIG: i64 = 600_000;

fn box_err<E: std::error::Error + Send + Sync + 'static>(e: E) -> DataFusionError {
    DataFusionError::External(Box::new(e))
}

/// Build a catalog: one 600k-row insert (row_id_start = 0, so rowid == id),
/// then delete `targets`.
fn build_catalog(path: &Path, targets: &[i64]) -> DataFusionResult<()> {
    let conn = duckdb::Connection::open_in_memory().map_err(box_err)?;
    conn.execute("INSTALL ducklake;", []).map_err(box_err)?;
    conn.execute("INSTALL parquet;", []).map_err(box_err)?;
    conn.execute("LOAD ducklake;", []).map_err(box_err)?;
    conn.execute(&format!("ATTACH 'ducklake:{}' AS c;", path.display()), [])
        .map_err(box_err)?;
    conn.execute("CREATE TABLE c.t(id INTEGER);", [])
        .map_err(box_err)?;
    conn.execute(
        &format!("INSERT INTO c.t SELECT i FROM range(0, {BIG}) t(i);"),
        [],
    )
    .map_err(box_err)?;
    let list = targets
        .iter()
        .map(|t| t.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    conn.execute(&format!("DELETE FROM c.t WHERE id IN ({list});"), [])
        .map_err(box_err)?;
    Ok(())
}

/// Query the deletions feed over the full history and return the sorted
/// `(id, rowid, change_type)` rows.
async fn deletions(ctx: &SessionContext, path: &Path) -> DataFusionResult<Vec<(i64, i64, String)>> {
    let provider =
        Arc::new(DuckdbMetadataProvider::new(path.to_str().expect("utf8 path")).map_err(box_err)?);
    register_ducklake_functions(ctx, provider);
    let batches = ctx
        .sql("SELECT id, rowid, change_type FROM ducklake_table_deletions('main.t', 0, 3)")
        .await?
        .collect()
        .await?;
    let mut rows = Vec::new();
    for b in &batches {
        let id = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        let rid = b.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
        let ct = b.column(2).as_any().downcast_ref::<StringArray>().unwrap();
        for r in 0..b.num_rows() {
            rows.push((id.value(r) as i64, rid.value(r), ct.value(r).to_string()));
        }
    }
    rows.sort();
    Ok(rows)
}

fn expected(targets: &[i64]) -> Vec<(i64, i64, String)> {
    let sorted: BTreeSet<i64> = targets.iter().copied().collect();
    sorted
        .into_iter()
        .map(|t| (t, t, "delete".to_string()))
        .collect()
}

/// The configuration from issue #178: aggressively split single files into
/// multiple byte-range scan partitions.
fn split_ctx() -> SessionContext {
    let mut cfg = ConfigOptions::new();
    cfg.execution.target_partitions = 8;
    cfg.optimizer.repartition_file_scans = true;
    cfg.optimizer.repartition_file_min_size = 1;
    SessionContext::new_with_config(SessionConfig::from(cfg))
}

// ---------------------------------------------------------------------------
// 1. DEFAULT context (FAILS pre-fix): the physical optimizer's round-robin
//    repartitioning alone desynchronized positions from rows.
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn deletions_correct_under_default_context() -> DataFusionResult<()> {
    let temp = TempDir::new().map_err(box_err)?;
    let path = temp.path().join("default.ducklake");
    let targets = [300_005i64];
    build_catalog(&path, &targets)?;

    let ctx = SessionContext::new();
    assert_eq!(deletions(&ctx, &path).await?, expected(&targets));
    Ok(())
}

// ---------------------------------------------------------------------------
// 2. Aggressive byte-range splitting, the issue's original repro (FAILS pre-fix)
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn deletions_correct_under_file_scan_splitting() -> DataFusionResult<()> {
    let temp = TempDir::new().map_err(box_err)?;
    let path = temp.path().join("split.ducklake");
    let targets = [300_005i64];
    build_catalog(&path, &targets)?;

    assert_eq!(deletions(&split_ctx(), &path).await?, expected(&targets));
    Ok(())
}

// ---------------------------------------------------------------------------
// 3. Deletes scattered across the whole file — first row, batch boundaries,
//    row-group interiors, last row — with both contexts. Also guards against
//    emitting the WRONG row's content for a matched position.
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn scattered_deletions_correct_under_both_contexts() -> DataFusionResult<()> {
    let temp = TempDir::new().map_err(box_err)?;
    let path = temp.path().join("scattered.ducklake");
    let targets = [0i64, 8_191, 8_192, 122_880, 245_765, 599_999];
    build_catalog(&path, &targets)?;

    assert_eq!(
        deletions(&SessionContext::new(), &path).await?,
        expected(&targets)
    );
    assert_eq!(deletions(&split_ctx(), &path).await?, expected(&targets));
    Ok(())
}

// ---------------------------------------------------------------------------
// 4. rowid projected away (no rowid synthesis) still matches by true position.
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn deletions_without_rowid_projection() -> DataFusionResult<()> {
    let temp = TempDir::new().map_err(box_err)?;
    let path = temp.path().join("norowid.ducklake");
    build_catalog(&path, &[300_005])?;

    let ctx = SessionContext::new();
    let provider =
        Arc::new(DuckdbMetadataProvider::new(path.to_str().expect("utf8 path")).map_err(box_err)?);
    register_ducklake_functions(&ctx, provider);
    let batches = ctx
        .sql("SELECT id FROM ducklake_table_deletions('main.t', 0, 3)")
        .await?
        .collect()
        .await?;
    let mut ids = Vec::new();
    for b in &batches {
        let id = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        for r in 0..b.num_rows() {
            ids.push(id.value(r));
        }
    }
    assert_eq!(ids, vec![300_005]);
    Ok(())
}
