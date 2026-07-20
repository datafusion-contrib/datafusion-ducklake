//! Partitioned write validation for the DuckDB backend.
//!
//! Exercises the DuckDB writer's partition path directly (`set_partition_spec` +
//! atomic `register_data_files` with per-file partition values), then reads it
//! back through the DuckDB provider. The DataFusion INSERT splitting itself is
//! backend-agnostic and covered by the SQLite tests; this pins the
//! DuckDB-specific catalog SQL (sequence-allocated `partition_id`, `RETURNING`).

#![cfg(all(feature = "write-duckdb", feature = "metadata-duckdb"))]

use datafusion_ducklake::metadata_provider::MetadataProvider;
use datafusion_ducklake::partition::PartitionTransform;
use datafusion_ducklake::{
    ColumnDef, DataFileInfo, DuckdbMetadataProvider, DuckdbMetadataWriter, MetadataWriter,
    WriteMode,
};
use tempfile::TempDir;

#[test]
fn duckdb_set_spec_and_register_partition_files() {
    let temp = TempDir::new().unwrap();
    let db = temp.path().join("catalog.ducklake");
    let db_str = db.to_str().unwrap().to_string();
    let data = temp.path().join("data");

    let table_id;
    {
        let writer = DuckdbMetadataWriter::new_with_init(&db_str).unwrap();
        writer.set_data_path(data.to_str().unwrap()).unwrap();

        let cols = vec![
            ColumnDef::new("id", "int64", false).unwrap(),
            ColumnDef::new("region", "varchar", true).unwrap(),
        ];
        // Create the empty table.
        let setup = writer
            .begin_write_transaction("main", "events", &cols, WriteMode::Replace)
            .unwrap();
        writer
            .publish_snapshot(
                setup.table_id,
                "main",
                "events",
                setup.snapshot_id,
                WriteMode::Replace,
                setup.base_snapshot_id,
                &cols,
                &setup.column_ids,
            )
            .unwrap();
        table_id = setup.table_id;

        // Partition by region (identity). Fresh catalog → the sequence assigns
        // partition_id = 1.
        writer
            .set_partition_spec(
                table_id,
                &[("region".to_string(), PartitionTransform::Identity)],
            )
            .unwrap();

        // Register two partition files atomically in one snapshot.
        let files = vec![
            DataFileInfo::new("region=us/a.parquet", 100, 2)
                .with_partition(1, vec![(0, Some("us".to_string()))]),
            DataFileInfo::new("region=eu/b.parquet", 100, 3)
                .with_partition(1, vec![(0, Some("eu".to_string()))]),
        ];
        let setup2 = writer
            .begin_write_transaction("main", "events", &cols, WriteMode::Append)
            .unwrap();
        writer
            .register_data_files(
                setup2.table_id,
                "main",
                "events",
                setup2.snapshot_id,
                &files,
                WriteMode::Append,
                setup2.base_snapshot_id,
                &cols,
                &setup2.column_ids,
            )
            .unwrap();
        // Writer (and its lock on the DuckDB file) dropped here.
    }

    let provider = DuckdbMetadataProvider::new(&db_str).unwrap();
    let snap = provider.get_current_snapshot().unwrap();

    let spec = provider
        .get_partition_spec(table_id, snap)
        .unwrap()
        .expect("partition spec present");
    assert_eq!(spec.columns.len(), 1);
    assert_eq!(spec.columns[0].transform, PartitionTransform::Identity);

    let page = provider
        .get_table_file_metadata_page(table_id, snap, None, 4096)
        .unwrap();
    assert_eq!(page.len(), 2, "two partition files");
    let mut regions: Vec<String> = page
        .iter()
        .filter_map(|m| {
            m.file
                .partition_values
                .iter()
                .find(|(k, _)| *k == 0)
                .and_then(|(_, v)| v.clone())
        })
        .collect();
    regions.sort();
    assert_eq!(regions, vec!["eu".to_string(), "us".to_string()]);

    // Row count = 2 + 3 across the two partition files.
    assert_eq!(provider.get_table_row_count(table_id, snap).unwrap(), 5);
}

#[test]
fn duckdb_register_files_with_retired_partition_id_conflicts() {
    // Concurrency fence: registering files stamped with a partition_id whose spec
    // generation was retired (by a concurrent RESET/SET) must abort with Conflict,
    // never commit a file pointing at a retired partition_id.
    let temp = TempDir::new().unwrap();
    let db = temp.path().join("catalog.ducklake");
    let db_str = db.to_str().unwrap().to_string();
    let data = temp.path().join("data");

    let writer = DuckdbMetadataWriter::new_with_init(&db_str).unwrap();
    writer.set_data_path(data.to_str().unwrap()).unwrap();
    let cols = vec![
        ColumnDef::new("id", "int64", false).unwrap(),
        ColumnDef::new("region", "varchar", true).unwrap(),
    ];
    let setup = writer
        .begin_write_transaction("main", "events", &cols, WriteMode::Replace)
        .unwrap();
    writer
        .publish_snapshot(
            setup.table_id,
            "main",
            "events",
            setup.snapshot_id,
            WriteMode::Replace,
            setup.base_snapshot_id,
            &cols,
            &setup.column_ids,
        )
        .unwrap();
    let table_id = setup.table_id;

    // Set spec (region) → partition_id 1, then RESET → retire generation 1.
    writer
        .set_partition_spec(
            table_id,
            &[("region".to_string(), PartitionTransform::Identity)],
        )
        .unwrap();
    writer.reset_partition_spec(table_id).unwrap();

    // Register a file stamped with the now-retired partition_id 1.
    let files = vec![
        DataFileInfo::new("region=us/a.parquet", 100, 2)
            .with_partition(1, vec![(0, Some("us".to_string()))]),
    ];
    let setup2 = writer
        .begin_write_transaction("main", "events", &cols, WriteMode::Append)
        .unwrap();
    let result = writer.register_data_files(
        setup2.table_id,
        "main",
        "events",
        setup2.snapshot_id,
        &files,
        WriteMode::Append,
        setup2.base_snapshot_id,
        &cols,
        &setup2.column_ids,
    );
    let err = result.expect_err("registering a retired partition_id must conflict");
    let msg = err.to_string().to_lowercase();
    assert!(
        msg.contains("partition spec") || msg.contains("concurrent"),
        "expected a partition-spec conflict, got: {err}"
    );

    // Nothing committed: the table still has no data files.
    let provider = DuckdbMetadataProvider::new(&db_str).unwrap();
    let snap = provider.get_current_snapshot().unwrap();
    let page = provider
        .get_table_file_metadata_page(table_id, snap, None, 4096)
        .unwrap();
    assert!(
        page.is_empty(),
        "a conflicting register must not commit files"
    );
}

#[test]
fn duckdb_register_unpartitioned_file_into_partitioned_table_conflicts() {
    // Inverse fence: registering an unpartitioned file (no partition_id) into a table
    // that has a live partition spec must conflict — never leave a partition_id-less
    // file in a partitioned table.
    let temp = TempDir::new().unwrap();
    let db = temp.path().join("catalog.ducklake");
    let db_str = db.to_str().unwrap().to_string();
    let data = temp.path().join("data");

    let writer = DuckdbMetadataWriter::new_with_init(&db_str).unwrap();
    writer.set_data_path(data.to_str().unwrap()).unwrap();
    let cols = vec![
        ColumnDef::new("id", "int64", false).unwrap(),
        ColumnDef::new("region", "varchar", true).unwrap(),
    ];
    let setup = writer
        .begin_write_transaction("main", "events", &cols, WriteMode::Replace)
        .unwrap();
    writer
        .publish_snapshot(
            setup.table_id,
            "main",
            "events",
            setup.snapshot_id,
            WriteMode::Replace,
            setup.base_snapshot_id,
            &cols,
            &setup.column_ids,
        )
        .unwrap();
    let table_id = setup.table_id;

    // Make the table partitioned.
    writer
        .set_partition_spec(
            table_id,
            &[("region".to_string(), PartitionTransform::Identity)],
        )
        .unwrap();

    // Register an UNPARTITIONED file (no partition_id) → must conflict.
    let file = DataFileInfo::new("plain.parquet", 100, 2);
    let setup2 = writer
        .begin_write_transaction("main", "events", &cols, WriteMode::Append)
        .unwrap();
    let result = writer.register_data_file(
        setup2.table_id,
        "main",
        "events",
        setup2.snapshot_id,
        &file,
        WriteMode::Append,
        setup2.base_snapshot_id,
        &cols,
        &setup2.column_ids,
    );
    let err = result.expect_err("unpartitioned file into a partitioned table must conflict");
    let msg = err.to_string().to_lowercase();
    assert!(
        msg.contains("partition spec") || msg.contains("concurrent"),
        "expected a partition-spec conflict, got: {err}"
    );
}

#[test]
fn duckdb_empty_replace_on_partitioned_table_does_not_conflict() {
    // A 0-row Replace (empty overwrite truncate) on a partitioned table must NOT trip
    // the inverse fence — a 0-row file carries no partition data, so exempting it
    // preserves the empty-overwrite truncate behavior.
    let temp = TempDir::new().unwrap();
    let db = temp.path().join("catalog.ducklake");
    let db_str = db.to_str().unwrap().to_string();
    let data = temp.path().join("data");

    let writer = DuckdbMetadataWriter::new_with_init(&db_str).unwrap();
    writer.set_data_path(data.to_str().unwrap()).unwrap();
    let cols = vec![
        ColumnDef::new("id", "int64", false).unwrap(),
        ColumnDef::new("region", "varchar", true).unwrap(),
    ];
    let setup = writer
        .begin_write_transaction("main", "events", &cols, WriteMode::Replace)
        .unwrap();
    writer
        .publish_snapshot(
            setup.table_id,
            "main",
            "events",
            setup.snapshot_id,
            WriteMode::Replace,
            setup.base_snapshot_id,
            &cols,
            &setup.column_ids,
        )
        .unwrap();
    let table_id = setup.table_id;

    writer
        .set_partition_spec(
            table_id,
            &[("region".to_string(), PartitionTransform::Identity)],
        )
        .unwrap();

    // 0-row Replace (truncate marker) → must succeed, not conflict.
    let file = DataFileInfo::new("empty-marker.parquet", 0, 0);
    let setup2 = writer
        .begin_write_transaction("main", "events", &cols, WriteMode::Replace)
        .unwrap();
    let result = writer.register_data_file(
        setup2.table_id,
        "main",
        "events",
        setup2.snapshot_id,
        &file,
        WriteMode::Replace,
        setup2.base_snapshot_id,
        &cols,
        &setup2.column_ids,
    );
    assert!(
        result.is_ok(),
        "0-row Replace on a partitioned table must not conflict (truncate): {result:?}"
    );

    let provider = DuckdbMetadataProvider::new(&db_str).unwrap();
    let snap = provider.get_current_snapshot().unwrap();
    assert_eq!(
        provider.get_table_row_count(table_id, snap).unwrap(),
        0,
        "truncate marker leaves zero live rows"
    );
}

#[test]
fn duckdb_register_multiple_unpartitioned_files_into_partitioned_table_conflicts() {
    // P2 (public multi-file API): register_data_files with NON-EMPTY partition_id-less
    // files into a partitioned table must conflict — the multi-file path fences the
    // unpartitioned direction too, not just the retired-partition_id case.
    let temp = TempDir::new().unwrap();
    let db = temp.path().join("catalog.ducklake");
    let db_str = db.to_str().unwrap().to_string();
    let data = temp.path().join("data");

    let writer = DuckdbMetadataWriter::new_with_init(&db_str).unwrap();
    writer.set_data_path(data.to_str().unwrap()).unwrap();
    let cols = vec![
        ColumnDef::new("id", "int64", false).unwrap(),
        ColumnDef::new("region", "varchar", true).unwrap(),
    ];
    let setup = writer
        .begin_write_transaction("main", "events", &cols, WriteMode::Replace)
        .unwrap();
    writer
        .publish_snapshot(
            setup.table_id,
            "main",
            "events",
            setup.snapshot_id,
            WriteMode::Replace,
            setup.base_snapshot_id,
            &cols,
            &setup.column_ids,
        )
        .unwrap();
    let table_id = setup.table_id;
    writer
        .set_partition_spec(
            table_id,
            &[("region".to_string(), PartitionTransform::Identity)],
        )
        .unwrap();

    // Two non-empty files with NO partition_id (DataFileInfo::new default).
    let files =
        vec![DataFileInfo::new("a.parquet", 100, 2), DataFileInfo::new("b.parquet", 100, 3)];
    let setup2 = writer
        .begin_write_transaction("main", "events", &cols, WriteMode::Append)
        .unwrap();
    let result = writer.register_data_files(
        setup2.table_id,
        "main",
        "events",
        setup2.snapshot_id,
        &files,
        WriteMode::Append,
        setup2.base_snapshot_id,
        &cols,
        &setup2.column_ids,
    );
    let err =
        result.expect_err("non-empty unpartitioned files into a partitioned table must conflict");
    let msg = err.to_string().to_lowercase();
    assert!(
        msg.contains("partition spec") || msg.contains("concurrent"),
        "expected a partition-spec conflict, got: {err}"
    );
}
