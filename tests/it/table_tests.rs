#![cfg(feature = "metadata-duckdb")]
//! Table provider tests
//!
//! Tests for DuckLakeTable functionality.

use std::sync::Arc;

use arrow::array::Int64Array;
use datafusion::common::stats::Precision;
use datafusion::common::{ScalarValue, Statistics};
use datafusion::datasource::physical_plan::FileScanConfig;
use datafusion::datasource::source::DataSourceExec;
use datafusion::error::Result as DataFusionResult;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::prelude::*;
use datafusion_ducklake::metadata_provider::MetadataProvider;
use datafusion_ducklake::{DuckLakeCatalog, DuckdbMetadataProvider};
use tempfile::TempDir;

/// Creates a catalog with an empty table (no data files)
fn create_empty_table_catalog(catalog_path: &std::path::Path) -> anyhow::Result<()> {
    let conn = duckdb::Connection::open_in_memory()?;

    conn.execute("INSTALL ducklake;", [])?;
    conn.execute("LOAD ducklake;", [])?;

    // Create data directory (DuckLake only creates it on first INSERT)
    let data_dir = catalog_path.with_extension("ducklake.files");
    std::fs::create_dir_all(&data_dir)?;

    let ducklake_path = format!("ducklake:{}", catalog_path.display());
    conn.execute(&format!("ATTACH '{}' AS test_catalog;", ducklake_path), [])?;

    // Multiple columns for projection tests
    conn.execute(
        "CREATE TABLE test_catalog.tbl (a INTEGER, b VARCHAR, c DOUBLE);",
        [],
    )?;

    // No INSERT - table has no data files

    Ok(())
}

fn create_catalog(path: &str) -> DataFusionResult<Arc<DuckLakeCatalog>> {
    let provider = DuckdbMetadataProvider::new(path)
        .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
    let catalog = DuckLakeCatalog::new(provider)
        .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
    Ok(Arc::new(catalog))
}

/// Helper to setup test context
async fn setup_empty_table_context(name: &str) -> DataFusionResult<SessionContext> {
    let temp_dir =
        TempDir::new().map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
    let catalog_path = temp_dir.path().join(format!("{}.ducklake", name));

    create_empty_table_catalog(&catalog_path)
        .map_err(|e| datafusion::error::DataFusionError::External(e.into()))?;

    let catalog = create_catalog(&catalog_path.to_string_lossy())?;
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", catalog);

    // Keep temp_dir alive by leaking it (test cleanup handles it)
    std::mem::forget(temp_dir);

    Ok(ctx)
}

/// Test basic empty table scan
#[tokio::test]
async fn test_empty_table_basic_scan() -> DataFusionResult<()> {
    let ctx = setup_empty_table_context("basic").await?;

    let df = ctx.sql("SELECT * FROM ducklake.main.tbl").await?;
    let batches = df.collect().await?;

    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 0);

    Ok(())
}

/// Test empty table with projection
#[tokio::test]
async fn test_empty_table_projection() -> DataFusionResult<()> {
    let ctx = setup_empty_table_context("proj").await?;

    let df = ctx.sql("SELECT a FROM ducklake.main.tbl").await?;
    let schema = df.schema().clone();
    let batches = df.collect().await?;

    // Verify schema has only projected column
    assert_eq!(schema.fields().len(), 1);
    assert_eq!(schema.field(0).name(), "a");

    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 0);

    Ok(())
}

/// Test empty table with reordered projection
#[tokio::test]
async fn test_empty_table_reordered_projection() -> DataFusionResult<()> {
    let ctx = setup_empty_table_context("reorder").await?;

    let df = ctx.sql("SELECT c, a FROM ducklake.main.tbl").await?;
    let schema = df.schema().clone();
    let batches = df.collect().await?;

    // Verify schema has columns in correct order
    assert_eq!(schema.fields().len(), 2);
    assert_eq!(schema.field(0).name(), "c");
    assert_eq!(schema.field(1).name(), "a");

    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 0);

    Ok(())
}

/// Test empty table with filter
#[tokio::test]
async fn test_empty_table_with_filter() -> DataFusionResult<()> {
    let ctx = setup_empty_table_context("filter").await?;

    let df = ctx
        .sql("SELECT * FROM ducklake.main.tbl WHERE a > 10")
        .await?;
    let batches = df.collect().await?;

    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 0);

    Ok(())
}

/// Test empty table with aggregate (COUNT)
#[tokio::test]
async fn test_empty_table_aggregate() -> DataFusionResult<()> {
    let ctx = setup_empty_table_context("agg").await?;

    let df = ctx
        .sql("SELECT COUNT(*) as cnt FROM ducklake.main.tbl")
        .await?;
    let batches = df.collect().await?;

    // COUNT on empty table should return 1 row with value 0
    assert_eq!(batches.len(), 1);
    let batch = &batches[0];
    assert_eq!(batch.num_rows(), 1);

    let cnt = batch
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("COUNT should return Int64")
        .value(0);
    assert_eq!(cnt, 0, "COUNT(*) on empty table should be 0");

    Ok(())
}

/// Creates a catalog with a table populated with rows so DuckLake writes a real
/// data file. Used to validate that `DuckLakeTable::statistics()` agrees with
/// the catalog's per-file sizes.
fn create_populated_table_catalog(catalog_path: &std::path::Path) -> anyhow::Result<()> {
    let conn = duckdb::Connection::open_in_memory()?;
    conn.execute("INSTALL ducklake;", [])?;
    conn.execute("LOAD ducklake;", [])?;

    let data_dir = catalog_path.with_extension("ducklake.files");
    std::fs::create_dir_all(&data_dir)?;

    let ducklake_path = format!("ducklake:{}", catalog_path.display());
    conn.execute(&format!("ATTACH '{}' AS test_catalog;", ducklake_path), [])?;
    conn.execute("CREATE TABLE test_catalog.tbl (a INTEGER, b VARCHAR);", [])?;
    // Insert enough rows that DuckLake actually emits a data file.
    conn.execute(
        "INSERT INTO test_catalog.tbl SELECT i, repeat('x', 100) FROM range(0, 1000) t(i);",
        [],
    )?;
    Ok(())
}

fn collect_partitioned_file_statistics(
    plan: &Arc<dyn ExecutionPlan>,
    output: &mut Vec<Arc<Statistics>>,
) {
    if let Some(exec) = plan.downcast_ref::<DataSourceExec>()
        && let Some(config) = exec.data_source().downcast_ref::<FileScanConfig>()
    {
        for group in &config.file_groups {
            output.extend(
                group
                    .iter()
                    .filter_map(|file| file.statistics.as_ref().map(Arc::clone)),
            );
        }
    }
    for child in plan.children() {
        collect_partitioned_file_statistics(child, output);
    }
}

/// Collect the paths of every scanned data file in the plan, regardless of
/// whether statistics are attached. Unlike `collect_partitioned_file_statistics`
/// (which the delete/positional paths would undercount, since those build files
/// without statistics), this counts pruning on any scan path.
fn collect_partitioned_file_paths(plan: &Arc<dyn ExecutionPlan>, output: &mut Vec<String>) {
    if let Some(exec) = plan.downcast_ref::<DataSourceExec>()
        && let Some(config) = exec.data_source().downcast_ref::<FileScanConfig>()
    {
        for group in &config.file_groups {
            output.extend(
                group
                    .iter()
                    .map(|file| file.object_meta.location.to_string()),
            );
        }
    }
    for child in plan.children() {
        collect_partitioned_file_paths(child, output);
    }
}

/// DuckLake's table-, table-column-, and file-column statistic rows are all
/// surfaced through their corresponding DataFusion statistics objects.
#[tokio::test]
async fn test_catalog_statistics_are_exposed_to_datafusion() -> DataFusionResult<()> {
    let temp_dir =
        TempDir::new().map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
    let catalog_path = temp_dir.path().join("column_stats.ducklake");
    create_populated_table_catalog(&catalog_path)
        .map_err(|e| datafusion::error::DataFusionError::External(e.into()))?;

    let catalog = create_catalog(&catalog_path.to_string_lossy())?;
    let schema = datafusion::catalog::CatalogProvider::schema(catalog.as_ref(), "main")
        .expect("main schema exists");
    let table = schema
        .table("tbl")
        .await
        .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?
        .expect("tbl present in main schema");

    // Verify the provider loaded all three DuckLake statistics tables before
    // checking their DataFusion representation below.
    let provider = DuckdbMetadataProvider::new(catalog_path.to_string_lossy().to_string())
        .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
    let snapshot = provider
        .get_current_snapshot()
        .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
    let schema_meta = provider
        .get_schema_by_name("main", snapshot)
        .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?
        .expect("main schema metadata");
    let table_meta = provider
        .get_table_by_name(schema_meta.schema_id, "tbl", snapshot)
        .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?
        .expect("tbl metadata");
    let columns = provider
        .get_table_structure(table_meta.table_id, snapshot)
        .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
    let a_id = columns
        .iter()
        .find(|column| column.column_name == "a")
        .expect("a column metadata")
        .column_id;
    let catalog_stats = provider
        .get_table_statistics(table_meta.table_id, snapshot)
        .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
    assert_eq!(
        catalog_stats
            .table
            .as_ref()
            .and_then(|stats| stats.record_count),
        Some(1000)
    );
    let table_a = catalog_stats
        .columns
        .iter()
        .find(|stats| stats.column_id == a_id)
        .expect("table_column_stats for a");
    assert_eq!(table_a.contains_null, Some(false));
    assert_eq!(table_a.min_value.as_deref(), Some("0"));
    assert_eq!(table_a.max_value.as_deref(), Some("999"));
    let file_a = catalog_stats
        .files
        .iter()
        .find(|stats| stats.column_id == a_id)
        .expect("file_column_stats for a");
    assert_eq!(file_a.value_count, Some(1000));
    assert_eq!(file_a.null_count, Some(0));
    assert_eq!(file_a.min_value.as_deref(), Some("0"));
    assert_eq!(file_a.max_value.as_deref(), Some("999"));

    let table_stats = table.statistics().expect("table statistics");
    assert_eq!(table_stats.num_rows, Precision::Exact(1000));
    assert!(matches!(
        table_stats.total_byte_size,
        Precision::Inexact(bytes) if bytes > 0
    ));
    assert_eq!(table_stats.column_statistics.len(), 2);
    let a = &table_stats.column_statistics[0];
    assert_eq!(a.null_count, Precision::Exact(0));
    assert_eq!(a.min_value, Precision::Exact(ScalarValue::Int32(Some(0))));
    assert_eq!(a.max_value, Precision::Exact(ScalarValue::Int32(Some(999))));
    assert!(matches!(a.byte_size, Precision::Inexact(bytes) if bytes > 0));

    let state = SessionContext::new().state();
    let plan = table.scan(&state, None, &[], None).await?;
    let mut file_stats = Vec::new();
    collect_partitioned_file_statistics(&plan, &mut file_stats);
    assert_eq!(file_stats.len(), 1, "the data file should carry statistics");
    let file = &file_stats[0];
    assert_eq!(file.num_rows, Precision::Exact(1000));
    assert_eq!(
        file.column_statistics[0].min_value,
        Precision::Exact(ScalarValue::Int32(Some(0)))
    );
    assert_eq!(
        file.column_statistics[0].max_value,
        Precision::Exact(ScalarValue::Int32(Some(999)))
    );

    Ok(())
}

/// Validates that `DuckLakeTable::statistics()` returns the same byte total as
/// directly summing `file_size_bytes - delete_file_size_bytes` from the
/// catalog's per-file metadata — i.e. the same aggregate `ducklake_table_info`
/// produces.
#[tokio::test]
async fn test_statistics_total_byte_size_matches_catalog_aggregate() -> DataFusionResult<()> {
    let temp_dir =
        TempDir::new().map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
    let catalog_path = temp_dir.path().join("stats.ducklake");
    create_populated_table_catalog(&catalog_path)
        .map_err(|e| datafusion::error::DataFusionError::External(e.into()))?;

    let catalog = create_catalog(&catalog_path.to_string_lossy())?;
    let schema = datafusion::catalog::CatalogProvider::schema(catalog.as_ref(), "main")
        .expect("main schema exists");
    let table = schema
        .table("tbl")
        .await
        .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?
        .expect("tbl present in main schema");

    // What our impl reports.
    let stats = table.statistics().expect("statistics() returned None");
    let our_bytes = match stats.total_byte_size {
        Precision::Exact(b) | Precision::Inexact(b) => b as i64,
        Precision::Absent => {
            panic!("total_byte_size was Absent for a populated table")
        },
    };

    // Canonical: sum file_size_bytes - delete_file_size_bytes directly from
    // the catalog at the latest snapshot. Same aggregate ducklake_table_info
    // computes; same source rows our statistics() impl reads.
    let provider = DuckdbMetadataProvider::new(catalog_path.to_string_lossy().to_string())
        .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
    let snapshot_id = provider
        .get_current_snapshot()
        .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
    let schema_meta = provider
        .get_schema_by_name("main", snapshot_id)
        .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?
        .expect("main schema metadata");
    let table_meta = provider
        .get_table_by_name(schema_meta.schema_id, "tbl", snapshot_id)
        .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?
        .expect("tbl metadata");
    let files = provider
        .get_table_files_for_select(table_meta.table_id, snapshot_id)
        .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
    let canonical_bytes: i64 = files
        .iter()
        .map(|f| {
            let data = f.file.file_size_bytes;
            let dels = f.delete_file.as_ref().map_or(0, |d| d.file_size_bytes);
            data - dels
        })
        .sum();

    assert!(
        our_bytes > 0,
        "expected populated table to report non-zero bytes, got {}",
        our_bytes
    );
    assert_eq!(
        our_bytes, canonical_bytes,
        "statistics().total_byte_size must equal SUM(file_size) - SUM(delete_file_size) \
         from the catalog (our_bytes={}, canonical={})",
        our_bytes, canonical_bytes
    );

    // Hold temp_dir to outlive the catalog handle.
    std::mem::forget(temp_dir);
    Ok(())
}

/// Empty tables — no data files yet — should still return Statistics with
/// total_byte_size == 0, not Absent or None.
#[tokio::test]
async fn test_statistics_zero_for_empty_table() -> DataFusionResult<()> {
    let ctx = setup_empty_table_context("stats_empty").await?;
    let cat = ctx
        .catalog("ducklake")
        .expect("ducklake catalog registered");
    let schema = cat.schema("main").expect("main schema");
    let table = schema
        .table("tbl")
        .await
        .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?
        .expect("tbl present");
    let stats = table.statistics().expect("statistics() returned None");
    match stats.total_byte_size {
        Precision::Exact(0) | Precision::Inexact(0) => {},
        other => panic!("expected zero bytes for empty table, got {:?}", other),
    }
    Ok(())
}

/// Two data files with disjoint `a` ranges ([0, 999] and [10000, 10999]),
/// written as two separate INSERTs so DuckLake emits one file per range.
fn create_two_file_catalog(catalog_path: &std::path::Path) -> anyhow::Result<()> {
    let conn = duckdb::Connection::open_in_memory()?;
    conn.execute("INSTALL ducklake;", [])?;
    conn.execute("LOAD ducklake;", [])?;

    let data_dir = catalog_path.with_extension("ducklake.files");
    std::fs::create_dir_all(&data_dir)?;

    let ducklake_path = format!("ducklake:{}", catalog_path.display());
    conn.execute(&format!("ATTACH '{}' AS test_catalog;", ducklake_path), [])?;
    conn.execute("CREATE TABLE test_catalog.tbl (a INTEGER, b VARCHAR);", [])?;
    conn.execute(
        "INSERT INTO test_catalog.tbl SELECT i, repeat('x', 100) FROM range(0, 1000) t(i);",
        [],
    )?;
    conn.execute(
        "INSERT INTO test_catalog.tbl SELECT i, repeat('x', 100) FROM range(10000, 11000) t(i);",
        [],
    )?;
    Ok(())
}

/// Files whose min/max cannot satisfy a filter are dropped from the scan plan
/// at planning time, while a non-selective scan keeps every file. Establishes
/// the two-file baseline, then checks a selective filter, a filter matching
/// both files, and a filter matching none.
#[tokio::test]
async fn test_scan_prunes_files_by_statistics() -> DataFusionResult<()> {
    let temp_dir =
        TempDir::new().map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
    let catalog_path = temp_dir.path().join("prune.ducklake");
    create_two_file_catalog(&catalog_path)
        .map_err(|e| datafusion::error::DataFusionError::External(e.into()))?;

    let catalog = create_catalog(&catalog_path.to_string_lossy())?;
    let schema = datafusion::catalog::CatalogProvider::schema(catalog.as_ref(), "main")
        .expect("main schema exists");
    let table = schema
        .table("tbl")
        .await
        .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?
        .expect("tbl present in main schema");

    let state = SessionContext::new().state();

    let files_for = |plan: &Arc<dyn ExecutionPlan>| {
        let mut stats = Vec::new();
        collect_partitioned_file_statistics(plan, &mut stats);
        stats
    };

    // No filter: both files present in the plan.
    let plan = table.scan(&state, None, &[], None).await?;
    assert_eq!(
        files_for(&plan).len(),
        2,
        "unfiltered scan keeps every file"
    );

    // Selective filter `a < 5000` can only match the [0, 999] file; the
    // [10000, 10999] file is dropped at planning time.
    let plan = table
        .scan(&state, None, &[col("a").lt(lit(5000_i32))], None)
        .await?;
    let stats = files_for(&plan);
    assert_eq!(stats.len(), 1, "selective filter prunes the disjoint file");
    assert_eq!(
        stats[0].column_statistics[0].max_value,
        Precision::Exact(ScalarValue::Int32(Some(999))),
        "the surviving file is the low-range one"
    );

    // Filter spanning both ranges keeps both files.
    let plan = table
        .scan(&state, None, &[col("a").gt(lit(-1_i32))], None)
        .await?;
    assert_eq!(files_for(&plan).len(), 2, "non-selective filter keeps both");

    // Filter matching neither file prunes everything.
    let plan = table
        .scan(&state, None, &[col("a").gt(lit(999_999_i32))], None)
        .await?;
    assert_eq!(files_for(&plan).len(), 0, "unmatchable filter prunes all");

    // Correctness: pruning must not change results. Count through a full SQL
    // scan (filter reapplied on top) should equal the low file's row count.
    let ctx = SessionContext::new();
    ctx.register_catalog("c", catalog);
    let batches = ctx
        .sql("SELECT COUNT(*) FROM c.main.tbl WHERE a < 5000")
        .await?
        .collect()
        .await?;
    let count = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("count is Int64")
        .value(0);
    assert_eq!(
        count, 1000,
        "pruned query returns the full low-range result"
    );

    Ok(())
}

/// Pruning composes correctly with the with-deletes scan path: a table where
/// one file carries a live delete returns correct, delete-applied results under
/// a selective filter. (A file with a live delete has inexact statistics and is
/// intentionally never pruned — and its presence suppresses pruning by that
/// column for the whole scan — so this asserts correctness, not a file-count
/// reduction. See `prune_table_files`.)
#[tokio::test]
async fn test_scan_with_live_deletes_is_correct() -> DataFusionResult<()> {
    let temp_dir =
        TempDir::new().map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
    let catalog_path = temp_dir.path().join("prune_deletes.ducklake");

    // Two disjoint files, then delete one row from the low-range file so it
    // acquires a live delete file.
    {
        let conn = duckdb::Connection::open_in_memory()
            .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
        let exec = |sql: &str| -> DataFusionResult<()> {
            conn.execute(sql, [])
                .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
            Ok(())
        };
        exec("INSTALL ducklake;")?;
        exec("LOAD ducklake;")?;
        std::fs::create_dir_all(catalog_path.with_extension("ducklake.files"))
            .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
        exec(&format!(
            "ATTACH 'ducklake:{}' AS test_catalog;",
            catalog_path.display()
        ))?;
        exec("CREATE TABLE test_catalog.tbl (a INTEGER, b VARCHAR);")?;
        exec("INSERT INTO test_catalog.tbl SELECT i, repeat('x', 100) FROM range(0, 1000) t(i);")?;
        exec(
            "INSERT INTO test_catalog.tbl SELECT i, repeat('x', 100) FROM range(10000, 11000) t(i);",
        )?;
        exec("DELETE FROM test_catalog.tbl WHERE a = 42;")?;
    }

    let catalog = create_catalog(&catalog_path.to_string_lossy())?;
    let schema = datafusion::catalog::CatalogProvider::schema(catalog.as_ref(), "main")
        .expect("main schema exists");
    let table = schema
        .table("tbl")
        .await
        .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?
        .expect("tbl present in main schema");

    // Pruning runs over the with-deletes path without error, and the plan still
    // includes the file that carries the live delete (it is never pruned).
    let state = SessionContext::new().state();
    let plan = table
        .scan(&state, None, &[col("a").lt(lit(5000_i32))], None)
        .await?;
    let mut paths = Vec::new();
    collect_partitioned_file_paths(&plan, &mut paths);
    assert!(
        !paths.is_empty(),
        "the file with a live delete is kept, not pruned"
    );

    // Correctness across the full range: the delete is applied exactly once, and
    // no live rows are lost, whether or not a filter is present.
    let ctx = SessionContext::new();
    ctx.register_catalog("c", catalog);
    for (sql, expected, msg) in [
        (
            "SELECT COUNT(*) FROM c.main.tbl WHERE a < 5000",
            999_i64,
            "one row deleted from the low-range file",
        ),
        (
            "SELECT COUNT(*) FROM c.main.tbl WHERE a > 5000",
            1000,
            "the unaffected high-range file is unchanged",
        ),
        (
            "SELECT COUNT(*) FROM c.main.tbl",
            1999,
            "total reflects exactly one deleted row",
        ),
    ] {
        let batches = ctx.sql(sql).await?.collect().await?;
        let count = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("count is Int64")
            .value(0);
        assert_eq!(count, expected, "{msg}");
    }

    Ok(())
}

/// Pruning also applies on the row-lineage scan path (rowid projected), which
/// builds a separate per-file exec for each surviving file. Verified by file
/// path since that path does not always attach statistics.
#[tokio::test]
async fn test_scan_prunes_files_on_rowid_path() -> DataFusionResult<()> {
    let temp_dir =
        TempDir::new().map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
    let catalog_path = temp_dir.path().join("prune_rowid.ducklake");
    create_two_file_catalog(&catalog_path)
        .map_err(|e| datafusion::error::DataFusionError::External(e.into()))?;

    // Enable row lineage so the table exposes a synthetic `rowid` column and
    // `scan` routes through the per-file row-lineage path.
    let provider = DuckdbMetadataProvider::new(catalog_path.to_string_lossy().to_string())
        .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
    let catalog = Arc::new(
        DuckLakeCatalog::new(provider)
            .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?
            .with_row_lineage(true),
    );
    let schema = datafusion::catalog::CatalogProvider::schema(catalog.as_ref(), "main")
        .expect("main schema exists");
    let table = schema
        .table("tbl")
        .await
        .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?
        .expect("tbl present in main schema");

    let state = SessionContext::new().state();
    let paths_for = |plan: &Arc<dyn ExecutionPlan>| {
        let mut paths = Vec::new();
        collect_partitioned_file_paths(plan, &mut paths);
        paths
    };

    // Projection `None` with row lineage on means "all columns including rowid",
    // routing through the row-lineage path. A selective filter still prunes.
    let plan = table
        .scan(&state, None, &[col("a").lt(lit(5000_i32))], None)
        .await?;
    assert_eq!(
        paths_for(&plan).len(),
        1,
        "row-lineage scan prunes the disjoint file"
    );

    let plan = table.scan(&state, None, &[], None).await?;
    assert_eq!(
        paths_for(&plan).len(),
        2,
        "unfiltered row-lineage scan keeps every file"
    );

    // Correctness: rowid projection over the pruned scan returns every low-range
    // row exactly once.
    let ctx = SessionContext::new();
    ctx.register_catalog("c", catalog);
    let batches = ctx
        .sql("SELECT COUNT(*), COUNT(DISTINCT rowid) FROM c.main.tbl WHERE a < 5000")
        .await?
        .collect()
        .await?;
    let rows = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("count is Int64")
        .value(0);
    let distinct_rowids = batches[0]
        .column(1)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("count is Int64")
        .value(0);
    assert_eq!(rows, 1000, "all low-range rows returned");
    assert_eq!(
        distinct_rowids, 1000,
        "each surviving row has a distinct rowid"
    );

    Ok(())
}
