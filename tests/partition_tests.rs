#![cfg(feature = "metadata-duckdb")]
//! DuckLake partitioning tests.
//!
//! Read side is validated against a real DuckDB-produced partitioned catalog
//! (`LOAD ducklake; ALTER TABLE ... SET PARTITIONED BY (...)`), the ground-truth
//! oracle: this proves we correctly read catalogs DuckDB partitioned, parse the
//! spec, surface per-file partition values, and prune.

use std::sync::Arc;

use datafusion::prelude::*;
use datafusion_ducklake::metadata_provider::MetadataProvider;
use datafusion_ducklake::partition::PartitionTransform;
use datafusion_ducklake::{DuckLakeCatalog, DuckdbMetadataProvider};
use tempfile::TempDir;

/// Create a DuckLake catalog with an `events` table partitioned by
/// `(region, year(ts))` and four rows spanning four partitions
/// `(region × year)`, so DuckDB writes one data file per partition.
fn create_partitioned_catalog(catalog_path: &std::path::Path) -> anyhow::Result<()> {
    let conn = duckdb::Connection::open_in_memory()?;
    conn.execute("INSTALL ducklake;", [])?;
    conn.execute("LOAD ducklake;", [])?;
    conn.execute("INSTALL parquet;", [])?;

    let ducklake_path = format!("ducklake:{}", catalog_path.display());
    conn.execute(&format!("ATTACH '{}' AS test_catalog;", ducklake_path), [])?;

    conn.execute(
        "CREATE TABLE test_catalog.events (id INTEGER, region VARCHAR, ts TIMESTAMP);",
        [],
    )?;
    conn.execute(
        "ALTER TABLE test_catalog.events SET PARTITIONED BY (region, year(ts));",
        [],
    )?;
    conn.execute(
        "INSERT INTO test_catalog.events VALUES
            (1, 'us', TIMESTAMP '2023-01-15 10:00:00'),
            (2, 'us', TIMESTAMP '2024-06-20 12:00:00'),
            (3, 'eu', TIMESTAMP '2023-03-10 08:00:00'),
            (4, 'eu', TIMESTAMP '2024-11-05 18:00:00');",
        [],
    )?;
    Ok(())
}

fn setup(name: &str) -> anyhow::Result<(SessionContext, String, TempDir)> {
    let temp_dir = TempDir::new()?;
    let catalog_path = temp_dir.path().join(format!("{name}.ducklake"));
    create_partitioned_catalog(&catalog_path)?;
    let path = catalog_path.to_string_lossy().to_string();

    let provider = DuckdbMetadataProvider::new(&path)?;
    let catalog = DuckLakeCatalog::new(provider)?;
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));
    Ok((ctx, path, temp_dir))
}

#[tokio::test(flavor = "multi_thread")]
async fn read_partitioned_table_returns_all_rows() -> anyhow::Result<()> {
    let (ctx, _path, _tmp) = setup("read_all")?;
    let batches = ctx
        .sql("SELECT id FROM ducklake.main.events ORDER BY id")
        .await?
        .collect()
        .await?;
    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total, 4, "expected 4 rows across the 4 partitions");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn get_partition_spec_parses_transforms() -> anyhow::Result<()> {
    let (_ctx, path, _tmp) = setup("spec")?;
    let provider = DuckdbMetadataProvider::new(&path)?;
    let snapshot = provider.get_current_snapshot()?;
    let schema = provider.get_schema_by_name("main", snapshot)?.unwrap();
    let table = provider
        .get_table_by_name(schema.schema_id, "events", snapshot)?
        .unwrap();

    let spec = provider
        .get_partition_spec(table.table_id, snapshot)?
        .expect("events should have a partition spec");
    assert_eq!(spec.columns.len(), 2, "two partition keys");
    // Key 0 = region (identity), key 1 = year(ts).
    assert_eq!(spec.columns[0].partition_key_index, 0);
    assert_eq!(spec.columns[0].transform, PartitionTransform::Identity);
    assert_eq!(spec.columns[1].partition_key_index, 1);
    assert_eq!(spec.columns[1].transform, PartitionTransform::Year);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn per_file_partition_values_are_surfaced() -> anyhow::Result<()> {
    let (_ctx, path, _tmp) = setup("values")?;
    let provider = DuckdbMetadataProvider::new(&path)?;
    let snapshot = provider.get_current_snapshot()?;
    let schema = provider.get_schema_by_name("main", snapshot)?.unwrap();
    let table = provider
        .get_table_by_name(schema.schema_id, "events", snapshot)?
        .unwrap();

    let page = provider.get_table_file_metadata_page(table.table_id, snapshot, None, 4096)?;
    assert_eq!(page.len(), 4, "one data file per (region, year) partition");
    // Every file carries two partition values (region, year), and the set of
    // region values across files is exactly {us, eu}.
    let mut regions: Vec<String> = Vec::new();
    for meta in &page {
        assert_eq!(
            meta.file.partition_values.len(),
            2,
            "each file has a value for both partition keys"
        );
        let region = meta
            .file
            .partition_values
            .iter()
            .find(|(key_index, _)| *key_index == 0)
            .and_then(|(_, value)| value.clone())
            .expect("region partition value present");
        regions.push(region);
    }
    regions.sort();
    regions.dedup();
    assert_eq!(regions, vec!["eu".to_string(), "us".to_string()]);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn filter_on_partition_column_is_correct_and_prunes() -> anyhow::Result<()> {
    let (ctx, _path, _tmp) = setup("prune")?;

    // Correctness: filtering on the partition column returns exactly the matching rows.
    let batches = ctx
        .sql("SELECT id FROM ducklake.main.events WHERE region = 'us' ORDER BY id")
        .await?
        .collect()
        .await?;
    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total, 2, "two 'us' rows");

    // Pruning: the physical plan should reference only the two 'us' partition
    // files, not all four.
    let plan = ctx
        .sql("SELECT id FROM ducklake.main.events WHERE region = 'us'")
        .await?
        .create_physical_plan()
        .await?;
    let display = datafusion::physical_plan::displayable(plan.as_ref())
        .indent(true)
        .to_string();
    let files = display.matches(".parquet").count();
    assert!(
        files <= 2,
        "partition/stats pruning should keep at most 2 of 4 files, got {files}:\n{display}"
    );
    Ok(())
}
