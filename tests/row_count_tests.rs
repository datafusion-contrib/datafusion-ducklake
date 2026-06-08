#![cfg(feature = "metadata-duckdb")]
//! Integration tests for `MetadataProvider::get_table_row_count`.
//!
//! The metadata-only row count (`SUM(record_count) - SUM(delete_count)` over
//! the files visible at a snapshot) is cross-checked against two independent
//! ground truths for every catalog shape:
//!
//! 1. DuckDB/DuckLake's own `SELECT COUNT(*)` (run through a `duckdb`
//!    connection against the same catalog) — the upstream answer.
//! 2. DataFusion's `SELECT COUNT(*)` through this crate's delete-filtering scan.
//!
//! Shapes covered include merge-on-read updates, a truncate-then-reinsert
//! sequence (rewritten/superseded data files), and an all-deleted table — the
//! cases where naively summing physical `record_count` would overcount.

mod common;

use std::path::Path;
use std::sync::Arc;

use arrow::array::Int64Array;
use datafusion::error::Result as DataFusionResult;
use datafusion::prelude::*;
use datafusion_ducklake::{DuckLakeCatalog, DuckdbMetadataProvider, MetadataProvider};
use tempfile::TempDir;

fn to_df<E: std::error::Error + Send + Sync + 'static>(e: E) -> datafusion::error::DataFusionError {
    datafusion::error::DataFusionError::External(Box::new(e))
}

/// Resolve `(table_id, snapshot)` for `main.<table>` at the catalog head.
fn resolve_table(provider: &DuckdbMetadataProvider, table: &str) -> DataFusionResult<(i64, i64)> {
    let snapshot = provider.get_current_snapshot().map_err(to_df)?;
    let schema = provider
        .get_schema_by_name("main", snapshot)
        .map_err(to_df)?
        .expect("schema 'main' should exist");
    let table_meta = provider
        .get_table_by_name(schema.schema_id, table, snapshot)
        .map_err(to_df)?
        .unwrap_or_else(|| panic!("table '{table}' should exist"));
    Ok((table_meta.table_id, snapshot))
}

/// Upstream ground truth: `SELECT COUNT(*)` via DuckDB's own DuckLake reader.
fn duckdb_count_star(catalog_path: &Path, table: &str) -> DataFusionResult<i64> {
    let conn = duckdb::Connection::open_in_memory().map_err(to_df)?;
    conn.execute("INSTALL ducklake;", []).map_err(to_df)?;
    conn.execute("LOAD ducklake;", []).map_err(to_df)?;
    conn.execute(
        &format!("ATTACH 'ducklake:{}' AS c;", catalog_path.display()),
        [],
    )
    .map_err(to_df)?;
    conn.query_row(&format!("SELECT COUNT(*) FROM c.main.{table}"), [], |r| {
        r.get::<_, i64>(0)
    })
    .map_err(to_df)
}

/// `SELECT COUNT(*)` through this crate's DataFusion scan.
async fn datafusion_count_star(catalog_path: &str, table: &str) -> DataFusionResult<i64> {
    let provider = DuckdbMetadataProvider::new(catalog_path).map_err(to_df)?;
    let lake = DuckLakeCatalog::new(provider).map_err(to_df)?;
    let ctx = SessionContext::new();
    ctx.register_catalog("c", Arc::new(lake));
    let batches = ctx
        .sql(&format!("SELECT COUNT(*) FROM c.main.{table}"))
        .await?
        .collect()
        .await?;
    Ok(batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("COUNT(*) is Int64")
        .value(0))
}

/// For one catalog shape, assert the metadata count equals DuckLake's own
/// COUNT(*), this crate's COUNT(*), and the expected value.
async fn assert_counts_agree(
    create: fn(&Path) -> anyhow::Result<()>,
    table: &str,
    expected: u64,
) -> DataFusionResult<()> {
    let temp_dir = TempDir::new().map_err(to_df)?;
    let catalog_path = temp_dir.path().join(format!("{table}.ducklake"));
    create(&catalog_path).map_err(common::to_datafusion_error)?;
    let path = catalog_path.to_string_lossy().to_string();

    let provider = DuckdbMetadataProvider::new(&path).map_err(to_df)?;
    let (table_id, snapshot) = resolve_table(&provider, table)?;
    let metadata_count = provider
        .get_table_row_count(table_id, snapshot)
        .map_err(to_df)?;

    let upstream = duckdb_count_star(&catalog_path, table)?;
    let datafusion = datafusion_count_star(&path, table).await?;

    assert_eq!(
        metadata_count, expected,
        "metadata row count for '{table}' should be {expected}"
    );
    assert_eq!(
        metadata_count as i64, upstream,
        "metadata count must match DuckLake's own COUNT(*) for '{table}'"
    );
    assert_eq!(
        metadata_count as i64, datafusion,
        "metadata count must match DataFusion COUNT(*) for '{table}'"
    );
    Ok(())
}

#[tokio::test]
async fn test_row_count_no_deletes() -> DataFusionResult<()> {
    assert_counts_agree(common::create_catalog_no_deletes, "users", 4).await
}

#[tokio::test]
async fn test_row_count_with_deletes() -> DataFusionResult<()> {
    // 5 inserted - 2 deleted = 3 (physical SUM(record_count) would be 5).
    assert_counts_agree(common::create_catalog_with_deletes, "products", 3).await
}

#[tokio::test]
async fn test_row_count_merge_on_read_updates() -> DataFusionResult<()> {
    // 3 rows, 2 updated: new versions written + old versions delete-marked,
    // net still 3.
    assert_counts_agree(common::create_catalog_with_updates, "inventory", 3).await
}

#[tokio::test]
async fn test_row_count_truncate_then_reinsert() -> DataFusionResult<()> {
    // insert(1,2,3) -> delete all -> insert(4,5,6,7) -> delete(5,6): net {4,7}.
    assert_counts_agree(common::create_catalog_complex_deletions, "items", 2).await
}

#[tokio::test]
async fn test_row_count_all_deleted() -> DataFusionResult<()> {
    assert_counts_agree(common::create_catalog_empty_table, "tbl", 0).await
}
