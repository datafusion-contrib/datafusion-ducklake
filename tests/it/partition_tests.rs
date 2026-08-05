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
    crate::common::ensure_ducklake_installed();
    conn.execute("LOAD ducklake;", [])?;

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

/// Ground truth for a partition-moving `UPDATE` on a partitioned table, taken from
/// official DuckLake: it is the shape this crate's multi-file append+delete commit has
/// to reproduce, so it is pinned here rather than assumed.
///
/// Mirrors the upstream partition-moving-update scenario: two rows are updated so their
/// partition key changes, which moves them to two NEW partitions.
///
/// The spec uses two identity keys rather than a `day()` transform: on the DuckDB
/// version this crate links, a partition-moving UPDATE against a spec containing a
/// transform trips an internal error inside the reference implementation's update sink.
/// The commit SHAPE being pinned here is the same either way.
///
/// `DATA_INLINING_ROW_LIMIT 0` keeps the small write on the data-file path — with
/// inlining on, the rows never reach `ducklake_data_file` and there is nothing to
/// compare.
fn create_partition_moving_update_catalog(catalog_path: &std::path::Path) -> anyhow::Result<()> {
    let conn = duckdb::Connection::open_in_memory()?;
    crate::common::ensure_ducklake_installed();
    conn.execute("LOAD ducklake;", [])?;
    conn.execute(
        &format!(
            "ATTACH 'ducklake:{}' AS test_catalog (DATA_INLINING_ROW_LIMIT 0);",
            catalog_path.display()
        ),
        [],
    )?;
    conn.execute(
        "CREATE TABLE test_catalog.t (p VARCHAR, q VARCHAR, v VARCHAR);",
        [],
    )?;
    conn.execute("ALTER TABLE test_catalog.t SET PARTITIONED BY (p, q);", [])?;
    conn.execute(
        "INSERT INTO test_catalog.t VALUES
            ('p1', 'q1', 'va'),
            ('p2', 'q2', 'vb'),
            ('p1', 'q1', 'vc'),
            ('p2', 'q2', 'vd');",
        [],
    )?;
    // Moves 'va' to (p3, q1) and 'vb' to (p3, q2): two output partitions, one
    // superseded row in each of the two input files.
    conn.execute(
        "UPDATE test_catalog.t SET p = 'p3' WHERE v IN ('va','vb');",
        [],
    )?;
    Ok(())
}

/// The reference implementation commits a partition-moving UPDATE as: N appended data
/// files (one per output partition, each with its own partition values, row-id range
/// and per-column statistics) plus M delete files (one per superseded input file), ALL
/// stamped with ONE snapshot — the input files stay live, superseded by their delete
/// files rather than retired.
///
/// Ignored because it exercises the reference implementation, not this crate, and the
/// `ducklake` extension version this crate links cannot perform a partitioned UPDATE
/// reliably: it aborts with an internal assertion failure inside `DuckLakeUpdate::Sink`
/// (seen on macOS; the `day(ts)` transform variant fails on every platform, which is why
/// the spec here uses identity keys). Newer DuckDB releases run the same scenario fine,
/// so this is expected to pass once the linked version is bumped — hence ignored rather
/// than deleted. Run it explicitly with `--ignored`.
///
/// The behaviour it documents was verified by hand against a newer DuckDB CLI and the
/// observed catalog state is recorded in the commit that introduced this test, so the
/// evidence is not lost while the test is dormant. This crate's own equivalent of the
/// same scenario is covered by `sql_update_tests`, which does not depend on the
/// extension.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "linked ducklake extension aborts on partitioned UPDATE; passes on newer DuckDB"]
async fn upstream_partitioned_update_is_one_snapshot_of_n_files_and_m_deletes() -> anyhow::Result<()>
{
    let temp_dir = TempDir::new()?;
    let catalog_path = temp_dir.path().join("partition_moving_update.ducklake");
    create_partition_moving_update_catalog(&catalog_path)?;

    // Read the catalog rows directly: the claim is about the metadata shape.
    let conn = duckdb::Connection::open(&catalog_path)?;
    let update_snapshot: i64 = conn.query_row(
        "SELECT MAX(snapshot_id) FROM ducklake_snapshot",
        [],
        |row| row.get(0),
    )?;

    // Appended side: two data files, one per NEW partition, each holding one row.
    let mut stmt = conn.prepare(
        "SELECT f.data_file_id, f.record_count, f.row_id_start,
                (SELECT string_agg(v.partition_value, '/' ORDER BY v.partition_key_index)
                 FROM ducklake_file_partition_value v WHERE v.data_file_id = f.data_file_id),
                (SELECT COUNT(*) FROM ducklake_file_column_stats s
                 WHERE s.data_file_id = f.data_file_id)
         FROM ducklake_data_file f
         WHERE f.begin_snapshot = ?
         ORDER BY f.data_file_id",
    )?;
    let appended: Vec<(i64, i64, i64, String, i64)> = stmt
        .query_map([update_snapshot], |row| {
            Ok((
                row.get(0)?,
                row.get(1)?,
                row.get(2)?,
                row.get(3)?,
                row.get(4)?,
            ))
        })?
        .collect::<Result<_, _>>()?;
    assert_eq!(
        appended.len(),
        2,
        "one appended data file per output partition: {appended:?}"
    );
    let mut partitions: Vec<&str> = appended.iter().map(|f| f.3.as_str()).collect();
    partitions.sort_unstable();
    assert_eq!(
        partitions,
        vec!["p3/q1", "p3/q2"],
        "each appended file carries its OWN partition values"
    );
    assert!(
        appended.iter().all(|f| f.1 == 1),
        "one moved row per appended file: {appended:?}"
    );
    assert_ne!(
        appended[0].2, appended[1].2,
        "each appended file draws its own rowid range: {appended:?}"
    );
    assert!(
        appended.iter().all(|f| f.4 == 3),
        "per-column statistics for EVERY appended file (3 columns each): {appended:?}"
    );

    // Delete side: one delete file per superseded input file, on the SAME snapshot.
    let mut stmt = conn.prepare(
        "SELECT data_file_id, begin_snapshot, delete_count FROM ducklake_delete_file
         WHERE end_snapshot IS NULL ORDER BY data_file_id",
    )?;
    let deletes: Vec<(i64, i64, i64)> = stmt
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))?
        .collect::<Result<_, _>>()?;
    assert_eq!(
        deletes.len(),
        2,
        "one delete file per input file: {deletes:?}"
    );
    assert!(
        deletes.iter().all(|d| d.1 == update_snapshot),
        "the delete files share the appended files' snapshot: {deletes:?}"
    );
    assert!(
        deletes.iter().all(|d| d.2 == 1),
        "one superseded row per input file: {deletes:?}"
    );
    let appended_ids: Vec<i64> = appended.iter().map(|f| f.0).collect();
    assert!(
        deletes.iter().all(|d| !appended_ids.contains(&d.0)),
        "the deletes target the INPUT files, not the appended ones"
    );

    // The input files are superseded, not retired: they stay live at the new head.
    let retired: i64 = conn.query_row(
        "SELECT COUNT(*) FROM ducklake_data_file WHERE end_snapshot IS NOT NULL",
        [],
        |row| row.get(0),
    )?;
    assert_eq!(retired, 0, "a keyed mutation retires no data file");

    // Gross record_count: the appended rows are added, the deletes are accounted at
    // read time rather than subtracted here.
    let (record_count, next_row_id): (i64, i64) = conn.query_row(
        "SELECT record_count, next_row_id FROM ducklake_table_stats",
        [],
        |row| Ok((row.get(0)?, row.get(1)?)),
    )?;
    assert_eq!(record_count, 6, "4 inserted + 2 new versions, gross");
    assert_eq!(next_row_id, 6, "rowids are never reused");

    // And this crate reads that catalog back as an in-place update.
    let provider = DuckdbMetadataProvider::new(catalog_path.to_string_lossy().as_ref())?;
    let catalog = DuckLakeCatalog::new(provider)?;
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));
    let batches = ctx
        .sql("SELECT p, v FROM ducklake.main.t ORDER BY v")
        .await?
        .collect()
        .await?;
    let mut pairs = Vec::new();
    for b in &batches {
        let p = b
            .column(0)
            .as_any()
            .downcast_ref::<arrow::array::StringViewArray>()
            .unwrap();
        let v = b
            .column(1)
            .as_any()
            .downcast_ref::<arrow::array::StringViewArray>()
            .unwrap();
        for i in 0..b.num_rows() {
            pairs.push((p.value(i).to_string(), v.value(i).to_string()));
        }
    }
    assert_eq!(
        pairs,
        vec![
            ("p3".to_string(), "va".to_string()),
            ("p3".to_string(), "vb".to_string()),
            ("p1".to_string(), "vc".to_string()),
            ("p2".to_string(), "vd".to_string()),
        ],
        "'va' and 'vb' moved to p3; 'vc' and 'vd' untouched"
    );
    Ok(())
}
