#![cfg(feature = "write-mysql")]
//! MySQL metadata WRITER round-trip tests.
//!
//! Writes a catalog with [`MySqlMetadataWriter`] and reads it back with
//! [`MySqlMetadataProvider`], asserting the write path produced exactly the
//! snapshot / schema / table / column / data-file rows the provider resolves.
//!
//! Uses testcontainers to spin up a throwaway MySQL, so it is gated the same way
//! as `tests/mysql_metadata_provider_test.rs`: it is ignored under
//! `skip-tests-with-docker` on macOS (Docker unavailable there).

use datafusion_ducklake::{
    ColumnDef, DataFileInfo, MetadataProvider, MetadataWriter, MySqlMetadataProvider,
    MySqlMetadataWriter, WriteMode,
};
use testcontainers::runners::AsyncRunner;
use testcontainers_modules::mysql::Mysql;

/// Full write -> read round-trip through both a data-file commit
/// (`register_data_file`) and a fileless CREATE-TABLE commit
/// (`publish_snapshot`).
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn mysql_writer_roundtrip_write_then_read() {
    let container = Mysql::default().start().await.unwrap();
    let host = "127.0.0.1";
    let port = container.get_host_port_ipv4(3306).await.unwrap();
    let conn_str = format!("mysql://root@{}:{}/test", host, port);

    // --- Write side --------------------------------------------------------
    let writer = MySqlMetadataWriter::new_with_init(&conn_str).await.unwrap();
    writer.set_data_path("file:///tmp/ducklake_data/").unwrap();

    let columns = vec![
        ColumnDef::new("id", "int64", false).unwrap(),
        ColumnDef::new("name", "varchar", true).unwrap(),
    ];

    // Real write path: begin (reserve ids, get-or-create schema/table) then
    // commit by registering a data file.
    let setup = writer
        .begin_write_transaction("main", "users", &columns, WriteMode::Replace)
        .unwrap();
    let file = DataFileInfo::new("data_001.parquet", 1024, 4).with_footer_size(128);
    let committed = writer
        .register_data_file(
            setup.table_id,
            "main",
            "users",
            setup.snapshot_id,
            &file,
            WriteMode::Replace,
            setup.base_snapshot_id,
            &columns,
            &setup.column_ids,
        )
        .unwrap();
    // First write commits snapshot 1 on a fresh catalog.
    assert_eq!(committed.snapshot_id, 1, "first write commits snapshot 1");

    // A fileless CREATE TABLE exercises the publish_snapshot override.
    let cols2 = vec![ColumnDef::new("c1", "int32", true).unwrap()];
    let setup2 = writer
        .begin_write_transaction("main", "empty_t", &cols2, WriteMode::Replace)
        .unwrap();
    let committed2 = writer
        .publish_snapshot(
            setup2.table_id,
            "main",
            "empty_t",
            setup2.snapshot_id,
            WriteMode::Replace,
            setup2.base_snapshot_id,
            &cols2,
            &setup2.column_ids,
        )
        .unwrap();
    assert!(
        committed2.snapshot_id > committed.snapshot_id,
        "second commit advances the head"
    );

    // --- Read side ---------------------------------------------------------
    let provider = MySqlMetadataProvider::new(&conn_str).await.unwrap();

    let snap = provider.get_current_snapshot().unwrap();
    assert_eq!(snap, committed2.snapshot_id, "head is the latest commit");
    assert_eq!(
        provider.get_data_path().unwrap(),
        "file:///tmp/ducklake_data/"
    );

    // The schema written by the first commit is visible at the head.
    let schemas = provider.list_schemas(snap).unwrap();
    assert_eq!(schemas.len(), 1, "one schema");
    assert_eq!(schemas[0].schema_name, "main");

    // Both tables live under it.
    let tables = provider.list_tables(committed.schema_id, snap).unwrap();
    let names: Vec<_> = tables.iter().map(|t| t.table_name.as_str()).collect();
    assert!(names.contains(&"users"), "users table present");
    assert!(names.contains(&"empty_t"), "empty_t table present");

    // Column generation of `users` reads back in order with the written types.
    let structure = provider
        .get_table_structure(committed.table_id, snap)
        .unwrap();
    assert_eq!(structure.len(), 2, "users has two columns");
    assert_eq!(structure[0].column_name, "id");
    assert_eq!(structure[0].column_type, "int64");
    assert!(!structure[0].is_nullable);
    assert_eq!(structure[1].column_name, "name");
    assert_eq!(structure[1].column_type, "varchar");
    assert!(structure[1].is_nullable);

    // The registered data file reads back with its metadata; no delete file.
    let files = provider
        .get_table_files_for_select(committed.table_id, snap)
        .unwrap();
    assert_eq!(files.len(), 1, "one data file");
    assert_eq!(files[0].file.path, "data_001.parquet");
    assert_eq!(files[0].file.file_size_bytes, 1024);
    assert_eq!(files[0].file.footer_size, Some(128));
    assert!(files[0].delete_file.is_none(), "no delete file");

    // The fileless CREATE TABLE published a table with columns but no files.
    let empty_id = tables
        .iter()
        .find(|t| t.table_name == "empty_t")
        .unwrap()
        .table_id;
    let empty_structure = provider.get_table_structure(empty_id, snap).unwrap();
    assert_eq!(empty_structure.len(), 1, "empty_t has one column");
    assert_eq!(empty_structure[0].column_name, "c1");
    let empty_files = provider.get_table_files_for_select(empty_id, snap).unwrap();
    assert!(empty_files.is_empty(), "empty_t has no data files");
}
