#![cfg(feature = "metadata-duckdb")]
//! Regression tests for row-id lineage & delete filtering using TRUE physical
//! file position (not stream arrival order).
//!
//! These tests use files large enough that DataFusion splits the scan across
//! multiple byte-range partitions — the condition under which the legacy
//! arrival-order counters in `RowIdExec` / `DeleteFilterExec` produce wrong
//! answers. They are the gate for the physical-position fix; several FAIL on
//! the pre-fix code (documented per-test).
//!
//! See docs/rowid-lineage-physical-position-plan.md.

mod common;

use std::path::Path;
use std::sync::Arc;

use arrow::array::{Array, Int32Array, Int64Array};
use datafusion::config::ConfigOptions;
use datafusion::error::Result as DataFusionResult;
use datafusion::prelude::*;
use datafusion_ducklake::{DuckLakeCatalog, DuckdbMetadataProvider};
use tempfile::TempDir;

/// Rows-per-file large enough to span several Parquet row groups (DuckDB's
/// default row-group size is 122_880).
const BIG: i64 = 600_000;

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn open(catalog_path: &Path) -> anyhow::Result<duckdb::Connection> {
    let conn = duckdb::Connection::open_in_memory()?;
    conn.execute("INSTALL ducklake;", [])?;
    conn.execute("INSTALL parquet;", [])?;
    conn.execute("LOAD ducklake;", [])?;
    conn.execute(
        &format!("ATTACH 'ducklake:{}' AS c;", catalog_path.display()),
        [],
    )?;
    Ok(conn)
}

fn catalog(path: &str, lineage: bool) -> DataFusionResult<Arc<DuckLakeCatalog>> {
    let provider = DuckdbMetadataProvider::new(path)
        .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
    Ok(Arc::new(
        DuckLakeCatalog::new(provider)
            .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?
            .with_row_lineage(lineage),
    ))
}

/// SessionContext that aggressively splits single files into multiple scan
/// partitions — the configuration that exposes the arrival-order bug.
fn split_ctx() -> SessionContext {
    let mut cfg = ConfigOptions::new();
    cfg.execution.target_partitions = 8;
    cfg.optimizer.repartition_file_scans = true;
    cfg.optimizer.repartition_file_min_size = 1;
    SessionContext::new_with_config(SessionConfig::from(cfg))
}

fn temp() -> DataFusionResult<TempDir> {
    TempDir::new().map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))
}

// ---------------------------------------------------------------------------
// 1. rowid == i over a large single file (FAILS pre-fix)
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn rowid_matches_physical_position_on_split_file() -> DataFusionResult<()> {
    let temp = temp()?;
    let path = temp.path().join("rowid_big.ducklake");
    {
        let conn = open(&path).map_err(common::to_datafusion_error)?;
        conn.execute("CREATE TABLE c.t(i INTEGER);", [])
            .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
        // Single INSERT => one data file => row_id_start = 0 => rowid must equal i.
        conn.execute(
            &format!("INSERT INTO c.t SELECT i FROM range(0, {BIG}) t(i);"),
            [],
        )
        .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
    }

    let ctx = split_ctx();
    ctx.register_catalog("c", catalog(&path.to_string_lossy(), true)?);

    let batches = ctx
        .sql("SELECT rowid, i FROM c.main.t")
        .await?
        .collect()
        .await?;
    let mut total = 0usize;
    let mut mismatches = 0usize;
    for b in &batches {
        let rid = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        let v = b.column(1).as_any().downcast_ref::<Int32Array>().unwrap();
        for r in 0..b.num_rows() {
            total += 1;
            if rid.value(r) != v.value(r) as i64 {
                mismatches += 1;
            }
        }
    }
    assert_eq!(total, BIG as usize, "row count");
    assert_eq!(mismatches, 0, "{mismatches}/{total} rows have rowid != i");
    Ok(())
}

// ---------------------------------------------------------------------------
// 2. deleted physical positions are absent over a large file (FAILS pre-fix)
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn deletes_apply_to_correct_physical_rows_on_split_file() -> DataFusionResult<()> {
    let temp = temp()?;
    let path = temp.path().join("del_big.ducklake");
    let (lo, hi) = (245_760i64, 245_770i64); // a block inside a non-first row group
    {
        let conn = open(&path).map_err(common::to_datafusion_error)?;
        for s in [
            "CREATE TABLE c.t(i INTEGER);".to_string(),
            format!("INSERT INTO c.t SELECT i FROM range(0, {BIG}) t(i);"),
            format!("DELETE FROM c.t WHERE i >= {lo} AND i < {hi};"),
        ] {
            conn.execute(&s, [])
                .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
        }
    }

    let ctx = split_ctx();
    ctx.register_catalog("c", catalog(&path.to_string_lossy(), false)?);

    let batches = ctx.sql("SELECT i FROM c.main.t").await?.collect().await?;
    let mut total = 0usize;
    let mut survived_deleted = 0usize;
    for b in &batches {
        let v = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        for r in 0..b.num_rows() {
            total += 1;
            let x = v.value(r) as i64;
            if x >= lo && x < hi {
                survived_deleted += 1;
            }
        }
    }
    assert_eq!(
        survived_deleted, 0,
        "deleted rows [{lo},{hi}) must not appear"
    );
    assert_eq!(total, (BIG - (hi - lo)) as usize, "row count after delete");
    Ok(())
}

// ---------------------------------------------------------------------------
// 3. rowid + deletes together on a split file (FAILS pre-fix)
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn rowid_and_deletes_together_on_split_file() -> DataFusionResult<()> {
    let temp = temp()?;
    let path = temp.path().join("rid_del.ducklake");
    let (lo, hi) = (300_000i64, 300_005i64);
    {
        let conn = open(&path).map_err(common::to_datafusion_error)?;
        for s in [
            "CREATE TABLE c.t(i INTEGER);".to_string(),
            format!("INSERT INTO c.t SELECT i FROM range(0, {BIG}) t(i);"),
            format!("DELETE FROM c.t WHERE i >= {lo} AND i < {hi};"),
        ] {
            conn.execute(&s, [])
                .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
        }
    }

    let ctx = split_ctx();
    ctx.register_catalog("c", catalog(&path.to_string_lossy(), true)?);

    let batches = ctx
        .sql("SELECT rowid, i FROM c.main.t")
        .await?
        .collect()
        .await?;
    for b in &batches {
        let rid = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        let v = b.column(1).as_any().downcast_ref::<Int32Array>().unwrap();
        for r in 0..b.num_rows() {
            assert_eq!(rid.value(r), v.value(r) as i64, "rowid must equal i");
            let x = v.value(r) as i64;
            assert!(!(x >= lo && x < hi), "deleted row {x} must not appear");
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// 4. filtered rowid query keeps correct rowids (guard for piece D)
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn filtered_rowid_query_keeps_correct_rowids() -> DataFusionResult<()> {
    let temp = temp()?;
    let path = temp.path().join("filt.ducklake");
    {
        let conn = open(&path).map_err(common::to_datafusion_error)?;
        conn.execute("CREATE TABLE c.t(i INTEGER);", [])
            .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
        conn.execute(
            &format!("INSERT INTO c.t SELECT i FROM range(0, {BIG}) t(i);"),
            [],
        )
        .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
    }

    let ctx = split_ctx();
    ctx.register_catalog("c", catalog(&path.to_string_lossy(), true)?);

    let batches = ctx
        .sql("SELECT rowid, i FROM c.main.t WHERE i >= 590000")
        .await?
        .collect()
        .await?;
    let mut total = 0usize;
    for b in &batches {
        let rid = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        let v = b.column(1).as_any().downcast_ref::<Int32Array>().unwrap();
        for r in 0..b.num_rows() {
            total += 1;
            assert!(v.value(r) >= 590000, "filter must hold");
            assert_eq!(rid.value(r), v.value(r) as i64, "rowid must equal i");
        }
    }
    assert_eq!(total, (BIG - 590000) as usize);
    Ok(())
}

// ---------------------------------------------------------------------------
// 5. ORDER BY on a positional path must not corrupt rowids
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn order_by_does_not_corrupt_rowids() -> DataFusionResult<()> {
    let temp = temp()?;
    let path = temp.path().join("ord.ducklake");
    {
        let conn = open(&path).map_err(common::to_datafusion_error)?;
        conn.execute("CREATE TABLE c.t(i INTEGER);", [])
            .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
        conn.execute(
            &format!("INSERT INTO c.t SELECT i FROM range(0, {BIG}) t(i);"),
            [],
        )
        .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
    }

    let ctx = split_ctx();
    ctx.register_catalog("c", catalog(&path.to_string_lossy(), true)?);

    let batches = ctx
        .sql("SELECT rowid, i FROM c.main.t ORDER BY i DESC LIMIT 5")
        .await?
        .collect()
        .await?;
    for b in &batches {
        let rid = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        let v = b.column(1).as_any().downcast_ref::<Int32Array>().unwrap();
        for r in 0..b.num_rows() {
            assert_eq!(
                rid.value(r),
                v.value(r) as i64,
                "rowid must equal i after ORDER BY"
            );
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// 6. LIMIT over a file whose first physical row is deleted (FAILS pre-fix)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn limit_after_delete_returns_survivor_not_zero() -> DataFusionResult<()> {
    let temp = temp()?;
    let path = temp.path().join("lim.ducklake");
    {
        let conn = open(&path).map_err(common::to_datafusion_error)?;
        for s in [
            "CREATE TABLE c.t(i INTEGER);",
            "INSERT INTO c.t SELECT i FROM range(0, 10) t(i);",
            "DELETE FROM c.t WHERE i = 0;", // delete the FIRST physical row
        ] {
            conn.execute(s, [])
                .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
        }
    }

    let ctx = SessionContext::new();
    ctx.register_catalog("c", catalog(&path.to_string_lossy(), false)?);

    let batches = ctx
        .sql("SELECT i FROM c.main.t LIMIT 1")
        .await?
        .collect()
        .await?;
    let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(rows, 1, "LIMIT 1 must return one survivor, not {rows}");
    Ok(())
}

// ---------------------------------------------------------------------------
// 7. COUNT(*) with deletes on a split file preserves the count
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn count_star_with_deletes_on_split_file() -> DataFusionResult<()> {
    let temp = temp()?;
    let path = temp.path().join("cnt.ducklake");
    let deleted = 7i64;
    {
        let conn = open(&path).map_err(common::to_datafusion_error)?;
        for s in [
            "CREATE TABLE c.t(i INTEGER);".to_string(),
            format!("INSERT INTO c.t SELECT i FROM range(0, {BIG}) t(i);"),
            format!("DELETE FROM c.t WHERE i < {deleted};"),
        ] {
            conn.execute(&s, [])
                .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
        }
    }

    let ctx = split_ctx();
    ctx.register_catalog("c", catalog(&path.to_string_lossy(), false)?);

    let batches = ctx
        .sql("SELECT COUNT(*) FROM c.main.t")
        .await?
        .collect()
        .await?;
    let cnt = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!(cnt, BIG - deleted, "COUNT(*) must reflect deletes");
    Ok(())
}

// ---------------------------------------------------------------------------
// 8b. plan-shape invariant: nothing drops/reorders/re-splits rows below
//     FileRowNumberExec on a positional path.
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn positional_plan_shape_is_safe() -> DataFusionResult<()> {
    let temp = temp()?;
    let path = temp.path().join("plan.ducklake");
    {
        let conn = open(&path).map_err(common::to_datafusion_error)?;
        conn.execute("CREATE TABLE c.t(i INTEGER);", [])
            .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
        conn.execute(
            &format!("INSERT INTO c.t SELECT i FROM range(0, {BIG}) t(i);"),
            [],
        )
        .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
    }

    let ctx = split_ctx();
    ctx.register_catalog("c", catalog(&path.to_string_lossy(), true)?);

    let mut plan = String::new();
    for b in ctx
        .sql("EXPLAIN SELECT rowid, i FROM c.main.t")
        .await?
        .collect()
        .await?
    {
        if let Some(c) = b
            .column(1)
            .as_any()
            .downcast_ref::<arrow::array::StringArray>()
        {
            for r in 0..b.num_rows() {
                plan.push_str(c.value(r));
                plan.push('\n');
            }
        }
    }

    let lines: Vec<&str> = plan.lines().collect();
    let frn = lines
        .iter()
        .position(|l| l.contains("FileRowNumberExec"))
        .unwrap_or_else(|| panic!("plan has no FileRowNumberExec:\n{plan}"));

    // The file was split into multiple row-group-aligned partitions (parallel).
    assert!(
        plan.contains(" groups:"),
        "scan should be split into >1 group:\n{plan}"
    );

    // CRITICAL invariant: the scan is *directly* below FileRowNumberExec — nothing
    // reshuffles, coalesces, or re-splits rows between them (which would break the
    // per-partition seed). A RepartitionExec ABOVE FileRowNumberExec is fine: the
    // position is already materialized into the data and travels with each row.
    assert!(
        lines
            .get(frn + 1)
            .is_some_and(|l| l.contains("DataSourceExec")),
        "DataSourceExec must be the direct child of FileRowNumberExec:\n{plan}"
    );

    // No reader-side predicate (would prune rows before position synthesis).
    assert!(
        !plan.contains("predicate="),
        "no reader predicate on positional scan:\n{plan}"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// 8. single-row-group small file stays correct (regression for small files)
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn small_single_row_group_file_unchanged() -> DataFusionResult<()> {
    let temp = temp()?;
    let path = temp.path().join("small.ducklake");
    {
        let conn = open(&path).map_err(common::to_datafusion_error)?;
        conn.execute("CREATE TABLE c.t(i INTEGER);", [])
            .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
        conn.execute("INSERT INTO c.t SELECT i FROM range(0, 5) t(i);", [])
            .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
    }

    let ctx = split_ctx();
    ctx.register_catalog("c", catalog(&path.to_string_lossy(), true)?);

    let batches = ctx
        .sql("SELECT rowid, i FROM c.main.t ORDER BY rowid")
        .await?
        .collect()
        .await?;
    let mut pairs = Vec::new();
    for b in &batches {
        let rid = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        let v = b.column(1).as_any().downcast_ref::<Int32Array>().unwrap();
        for r in 0..b.num_rows() {
            pairs.push((rid.value(r), v.value(r)));
        }
    }
    assert_eq!(pairs, vec![(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]);
    Ok(())
}
