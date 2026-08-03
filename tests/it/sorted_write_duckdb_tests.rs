//! Sort-spec validation for the DuckDB backend.
//!
//! Pins the DuckDB-specific catalog SQL for sort order (sequence-allocated
//! `sort_id` via `nextval` + `RETURNING`-free insert, SET-time column validation,
//! and the deliberate *absence* of a schema_version bump on sort changes). The
//! DataFusion sort/rollover machinery itself is backend-agnostic and covered by
//! the SQLite tests; this exercises `set_sort_spec`/`reset_sort_spec`/
//! `get_sort_spec` on the DuckDB writer + provider.

#![cfg(all(feature = "write-duckdb", feature = "metadata-duckdb"))]

use datafusion_ducklake::metadata_provider::MetadataProvider;
use datafusion_ducklake::sort::{NullOrder, SortDirection, SortField};
use datafusion_ducklake::{
    ColumnDef, DataFileInfo, DuckdbMetadataProvider, DuckdbMetadataWriter, MetadataWriter,
    SnapshotCommitMetadata, WriteMode,
};
use tempfile::TempDir;

#[test]
fn duckdb_set_get_reset_sort_spec() {
    let temp = TempDir::new().unwrap();
    let db_str = temp
        .path()
        .join("catalog.ducklake")
        .to_str()
        .unwrap()
        .to_string();
    let data = temp.path().join("data");

    let table_id;
    {
        let writer = DuckdbMetadataWriter::new_with_init(&db_str).unwrap();
        writer.set_data_path(data.to_str().unwrap()).unwrap();
        let cols = vec![
            ColumnDef::new("id", "int64", false).unwrap(),
            ColumnDef::new("ts", "int64", true).unwrap(),
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
        table_id = setup.table_id;

        writer
            .set_sort_spec(
                table_id,
                &[
                    SortField::column(0, "id", SortDirection::Asc, NullOrder::NullsLast),
                    SortField::column(1, "ts", SortDirection::Desc, NullOrder::NullsFirst),
                ],
            )
            .unwrap();

        // An unknown sort column is rejected at SET time.
        let err = writer.set_sort_spec(
            table_id,
            &[SortField::column(0, "nope", SortDirection::Asc, NullOrder::NullsLast)],
        );
        assert!(err.is_err(), "unknown sort column must be rejected");
        // Writer (and its lock on the DuckDB file) dropped here.
    }

    let provider = DuckdbMetadataProvider::new(&db_str).unwrap();
    let snap = provider.get_current_snapshot().unwrap();
    let spec = provider
        .get_sort_spec(table_id, snap)
        .unwrap()
        .expect("sort spec present after SET");
    assert_eq!(spec.fields.len(), 2);
    assert_eq!(spec.fields[0].expression, "id");
    assert_eq!(spec.fields[0].direction, SortDirection::Asc);
    assert_eq!(spec.fields[0].null_order, NullOrder::NullsLast);
    assert_eq!(spec.fields[1].expression, "ts");
    assert_eq!(spec.fields[1].direction, SortDirection::Desc);
    assert_eq!(spec.fields[1].null_order, NullOrder::NullsFirst);

    {
        let writer = DuckdbMetadataWriter::new_with_init(&db_str).unwrap();
        writer.reset_sort_spec(table_id).unwrap();
    }
    let provider = DuckdbMetadataProvider::new(&db_str).unwrap();
    let snap = provider.get_current_snapshot().unwrap();
    assert!(
        provider.get_sort_spec(table_id, snap).unwrap().is_none(),
        "sort spec cleared after RESET"
    );
}

#[test]
fn duckdb_records_snapshot_changes_and_commit_metadata() {
    let temp = TempDir::new().unwrap();
    let db_str = temp
        .path()
        .join("snapshot_changes.ducklake")
        .to_str()
        .unwrap()
        .to_string();
    let columns = vec![ColumnDef::new("id", "int64", false).unwrap()];
    let bare_snapshot;
    let first_commit;
    let second_commit;
    let table_id;

    {
        let writer = DuckdbMetadataWriter::new_with_init(&db_str).unwrap();
        bare_snapshot = writer.create_snapshot().unwrap();
        let first = writer
            .begin_write_transaction("main", "events", &columns, WriteMode::Replace)
            .unwrap();
        table_id = first.table_id;
        first_commit = writer
            .register_data_file_with_commit_metadata(
                first.table_id,
                "main",
                "events",
                first.snapshot_id,
                &DataFileInfo::new("first.parquet", 100, 3),
                WriteMode::Replace,
                first.base_snapshot_id,
                &columns,
                &first.column_ids,
                &SnapshotCommitMetadata::new()
                    .with_author("Jane Doe")
                    .with_message("Initial import")
                    .with_extra_info("opaque-import-id"),
                None,
            )
            .unwrap();

        let second = writer
            .begin_write_transaction("main", "events", &columns, WriteMode::Replace)
            .unwrap();
        second_commit = writer
            .register_data_file(
                second.table_id,
                "main",
                "events",
                second.snapshot_id,
                &DataFileInfo::new("second.parquet", 200, 5),
                WriteMode::Replace,
                second.base_snapshot_id,
                &columns,
                &second.column_ids,
            )
            .unwrap();
    }

    let connection = duckdb::Connection::open(&db_str).unwrap();
    let mut statement = connection
        .prepare(
            "SELECT snapshot_id, changes_made, author, commit_message, commit_extra_info
             FROM ducklake_snapshot_changes
             ORDER BY snapshot_id",
        )
        .unwrap();
    let rows = statement
        .query_map([], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, Option<String>>(1)?,
                row.get::<_, Option<String>>(2)?,
                row.get::<_, Option<String>>(3)?,
                row.get::<_, Option<String>>(4)?,
            ))
        })
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    assert_eq!(
        rows,
        vec![
            (bare_snapshot, None, None, None, None),
            (
                first_commit.snapshot_id,
                Some(format!(
                    "created_schema:\"main\",created_table:\"main\".\"events\",\
                     inserted_into_table:{table_id}"
                )),
                Some("Jane Doe".to_string()),
                Some("Initial import".to_string()),
                Some("opaque-import-id".to_string()),
            ),
            (
                second_commit.snapshot_id,
                Some(format!(
                    "deleted_from_table:{table_id},inserted_into_table:{table_id}"
                )),
                None,
                None,
                None,
            ),
        ],
    );
}
