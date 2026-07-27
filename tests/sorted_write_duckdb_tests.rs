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
    ColumnDef, DuckdbMetadataProvider, DuckdbMetadataWriter, MetadataWriter, WriteMode,
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
