#![cfg(feature = "write-mysql")]
//! MySQL metadata WRITER round-trip tests.
//!
//! Writes a catalog with [`MySqlMetadataWriter`] and reads it back with
//! [`MySqlMetadataProvider`], asserting the write path produced exactly the
//! snapshot / schema / table / column / data-file rows the provider resolves.
//!
//! Uses testcontainers to spin up a throwaway MySQL, so it is gated the same way
//! as `tests/it/mysql_metadata_provider_test.rs`: it is ignored under
//! `skip-tests-with-docker` on macOS (Docker unavailable there).

use arrow::array::Int32Array;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion_ducklake::maintenance::ExpireCriteria;
use datafusion_ducklake::metadata_writer::InlinedRowRef;
use datafusion_ducklake::{
    ColumnDef, CommitIds, CompactionOutputFile, CompactionSourceFile, DataFileInfo,
    DeleteFileEntry, DeleteFileInfo, MetadataProvider, MetadataWriter, MySqlMetadataProvider,
    MySqlMetadataWriter, SnapshotCommitMetadata, SourceRetirement, WriteMode, WriteSetupResult,
};
use sqlx::{AssertSqlSafe, Row};
use std::sync::Arc;
use testcontainers::ContainerAsync;
use testcontainers::runners::AsyncRunner;
use testcontainers_modules::mysql::Mysql;

/// Start a throwaway MySQL and open an initialized writer against it. The
/// container is returned so it stays alive for the test's duration.
async fn start_writer() -> (ContainerAsync<Mysql>, MySqlMetadataWriter, sqlx::MySqlPool) {
    let container = Mysql::default().start().await.unwrap();
    let port = container.get_host_port_ipv4(3306).await.unwrap();
    let conn_str = format!("mysql://root@127.0.0.1:{port}/test");
    let writer = MySqlMetadataWriter::new_with_init(&conn_str).await.unwrap();
    writer.set_data_path("file:///tmp/ducklake_data/").unwrap();
    let pool = sqlx::MySqlPool::connect(&conn_str).await.unwrap();
    (container, writer, pool)
}

fn int_column() -> Vec<ColumnDef> {
    vec![ColumnDef::new("id", "int32", false).unwrap()]
}

fn append(writer: &MySqlMetadataWriter, path: &str, rows: i64) -> (WriteSetupResult, CommitIds) {
    let columns = int_column();
    let setup = writer
        .begin_write_transaction("main", "t", &columns, WriteMode::Append)
        .unwrap();
    let commit = writer
        .register_data_file(
            setup.table_id,
            "main",
            "t",
            setup.snapshot_id,
            &DataFileInfo::new(path, rows * 10, rows),
            WriteMode::Append,
            setup.base_snapshot_id,
            &columns,
            &setup.column_ids,
        )
        .unwrap();
    (setup, commit)
}

fn append_inlined(writer: &MySqlMetadataWriter, values: &[i32]) -> (WriteSetupResult, CommitIds) {
    let columns = int_column();
    let setup = writer
        .begin_write_transaction("main", "t", &columns, WriteMode::Append)
        .unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let batch =
        RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(values.to_vec()))]).unwrap();
    let commit = writer
        .register_inlined_data(
            setup.table_id,
            "main",
            "t",
            setup.snapshot_id,
            &[batch],
            WriteMode::Append,
            setup.base_snapshot_id,
            &columns,
            &setup.column_ids,
            &SnapshotCommitMetadata::default(),
            None,
        )
        .unwrap();
    (setup, commit)
}

async fn file_id(pool: &sqlx::MySqlPool, path: &str) -> i64 {
    sqlx::query_scalar("SELECT data_file_id FROM ducklake_data_file WHERE path = ?")
        .bind(path)
        .fetch_one(pool)
        .await
        .unwrap()
}

async fn inlined_table(pool: &sqlx::MySqlPool, table_id: i64) -> String {
    sqlx::query_scalar("SELECT table_name FROM ducklake_inlined_data_tables WHERE table_id = ?")
        .bind(table_id)
        .fetch_one(pool)
        .await
        .unwrap()
}

async fn changes_made(pool: &sqlx::MySqlPool, snapshot_id: i64) -> Option<String> {
    sqlx::query_scalar("SELECT changes_made FROM ducklake_snapshot_changes WHERE snapshot_id = ?")
        .bind(snapshot_id)
        .fetch_one(pool)
        .await
        .unwrap()
}

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
    let bare_snapshot = writer.create_snapshot().unwrap();
    assert_eq!(bare_snapshot, 1, "bare snapshot is snapshot 1");

    // Real write path: begin (reserve ids, get-or-create schema/table) then
    // commit by registering a data file.
    let setup = writer
        .begin_write_transaction("main", "users", &columns, WriteMode::Replace)
        .unwrap();
    let file = DataFileInfo::new("data_001.parquet", 1024, 4).with_footer_size(128);
    let committed = writer
        .register_data_file_with_commit_metadata(
            setup.table_id,
            "main",
            "users",
            setup.snapshot_id,
            &file,
            WriteMode::Replace,
            setup.base_snapshot_id,
            &columns,
            &setup.column_ids,
            &SnapshotCommitMetadata::new()
                .with_author("Jane Doe")
                .with_message("Initial import")
                .with_extra_info("opaque-import-id"),
            None,
        )
        .unwrap();
    assert_eq!(committed.snapshot_id, 2, "first write commits snapshot 2");

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

    let pool = sqlx::MySqlPool::connect(&conn_str).await.unwrap();
    let rows = sqlx::query(
        "SELECT snapshot_id, changes_made, author, commit_message, commit_extra_info
         FROM ducklake_snapshot_changes
         ORDER BY snapshot_id",
    )
    .fetch_all(&pool)
    .await
    .unwrap();
    let changes = rows
        .iter()
        .map(|row| {
            (
                row.try_get::<i64, _>("snapshot_id").unwrap(),
                row.try_get::<Option<String>, _>("changes_made").unwrap(),
                row.try_get::<Option<String>, _>("author").unwrap(),
                row.try_get::<Option<String>, _>("commit_message").unwrap(),
                row.try_get::<Option<String>, _>("commit_extra_info")
                    .unwrap(),
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        changes,
        vec![
            (bare_snapshot, None, None, None, None),
            (
                committed.snapshot_id,
                Some(format!(
                    "created_schema:\"main\",created_table:\"main\".\"users\",\
                     inserted_into_table:{}",
                    setup.table_id
                )),
                Some("Jane Doe".to_string()),
                Some("Initial import".to_string()),
                Some("opaque-import-id".to_string()),
            ),
            (
                committed2.snapshot_id,
                Some(format!(
                    "created_table:\"main\".\"empty_t\",inserted_into_table:{}",
                    setup2.table_id
                )),
                None,
                None,
                None,
            ),
        ],
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

    let levels = DataType::List(Arc::new(Field::new(
        "item",
        DataType::Struct(
            vec![
                Arc::new(Field::new("price", DataType::Decimal128(38, 16), false)),
                Arc::new(Field::new("count", DataType::UInt32, false)),
            ]
            .into(),
        ),
        false,
    )));
    let nested_columns = vec![ColumnDef::from_arrow("bids", &levels, false).unwrap()];
    let nested_setup = writer
        .begin_write_transaction("main", "depths", &nested_columns, WriteMode::Replace)
        .unwrap();
    assert_eq!(nested_setup.column_ids, vec![nested_setup.field_ids[0]]);
    assert_eq!(nested_setup.field_ids.len(), 4);
    let nested_commit = writer
        .register_data_file(
            nested_setup.table_id,
            "main",
            "depths",
            nested_setup.snapshot_id,
            &DataFileInfo::new("depths.parquet", 1, 1),
            WriteMode::Replace,
            nested_setup.base_snapshot_id,
            &nested_columns,
            &nested_setup.field_ids,
        )
        .unwrap();
    let nested_structure = provider
        .get_table_structure(nested_setup.table_id, nested_commit.snapshot_id)
        .unwrap();
    assert_eq!(nested_structure.len(), 1);
    assert_eq!(nested_structure[0].column_name, "bids");
    assert_eq!(
        nested_structure[0].column_type,
        "list<struct<price:decimal(38, 16),count:uint32>>"
    );

    let nested_rows = sqlx::query(
        "SELECT column_id, column_name, column_type, parent_column
         FROM ducklake_column
         WHERE table_id = ? AND end_snapshot IS NULL
         ORDER BY column_order",
    )
    .bind(nested_setup.table_id)
    .fetch_all(&pool)
    .await
    .unwrap();
    let nested_actual = nested_rows
        .iter()
        .map(|row| {
            (
                row.try_get::<i64, _>("column_id").unwrap(),
                row.try_get::<String, _>("column_name").unwrap(),
                row.try_get::<String, _>("column_type").unwrap(),
                row.try_get::<Option<i64>, _>("parent_column").unwrap(),
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        nested_actual,
        vec![
            (
                nested_setup.field_ids[0],
                "bids".into(),
                "list".into(),
                None
            ),
            (
                nested_setup.field_ids[1],
                "element".into(),
                "struct".into(),
                Some(nested_setup.field_ids[0]),
            ),
            (
                nested_setup.field_ids[2],
                "price".into(),
                "decimal(38, 16)".into(),
                Some(nested_setup.field_ids[1]),
            ),
            (
                nested_setup.field_ids[3],
                "count".into(),
                "uint32".into(),
                Some(nested_setup.field_ids[1]),
            ),
        ]
    );
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn mysql_writer_completion_matches_sqlite_contracts() {
    let container = Mysql::default().start().await.unwrap();
    let port = container.get_host_port_ipv4(3306).await.unwrap();
    let conn_str = format!("mysql://root@127.0.0.1:{port}/test");
    let writer = MySqlMetadataWriter::new_with_init(&conn_str).await.unwrap();
    writer.set_data_path("file:///tmp/ducklake_data/").unwrap();
    let pool = sqlx::MySqlPool::connect(&conn_str).await.unwrap();
    let columns = vec![ColumnDef::new("id", "int32", false).unwrap()];

    let first = writer
        .begin_write_transaction("main", "t", &columns, WriteMode::Append)
        .unwrap();
    let first_commit = writer
        .register_data_file(
            first.table_id,
            "main",
            "t",
            first.snapshot_id,
            &DataFileInfo::new("source.parquet", 40, 4),
            WriteMode::Append,
            first.base_snapshot_id,
            &columns,
            &first.column_ids,
        )
        .unwrap();
    let source_id: i64 =
        sqlx::query_scalar("SELECT data_file_id FROM ducklake_data_file WHERE path = ?")
            .bind("source.parquet")
            .fetch_one(&pool)
            .await
            .unwrap();
    let delete_commit = writer
        .set_delete_file(
            first.table_id,
            "main",
            "t",
            first_commit.snapshot_id + 1,
            source_id,
            None,
            first_commit.snapshot_id,
            &DeleteFileInfo::new("delete-1.parquet", 10, 1),
        )
        .unwrap();
    let first_delete_id: i64 = sqlx::query_scalar(
        "SELECT delete_file_id FROM ducklake_delete_file
         WHERE data_file_id = ? AND end_snapshot IS NULL",
    )
    .bind(source_id)
    .fetch_one(&pool)
    .await
    .unwrap();
    let update = writer
        .begin_write_transaction("main", "t", &columns, WriteMode::Append)
        .unwrap();
    let update_commit = writer
        .register_data_file_with_deletes(
            update.table_id,
            "main",
            "t",
            update.snapshot_id,
            &DataFileInfo::new("updated.parquet", 10, 1),
            &[DeleteFileEntry {
                data_file_id: source_id,
                expected_prev_delete_file: Some(first_delete_id),
                delete: DeleteFileInfo::new("delete-2.parquet", 20, 2),
            }],
            WriteMode::Append,
            update.base_snapshot_id,
            &columns,
            &update.column_ids,
        )
        .unwrap();
    assert!(writer.supports_update());
    assert_eq!(update_commit.snapshot_id, delete_commit.snapshot_id + 1);
    let update_state: (i64, Option<i64>, i64) = sqlx::query_as(
        "SELECT
            (SELECT begin_snapshot FROM ducklake_data_file WHERE path = 'updated.parquet'),
            (SELECT end_snapshot FROM ducklake_delete_file WHERE delete_file_id = ?),
            (SELECT begin_snapshot FROM ducklake_delete_file WHERE path = 'delete-2.parquet')",
    )
    .bind(first_delete_id)
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(
        update_state,
        (
            update_commit.snapshot_id,
            Some(update_commit.snapshot_id),
            update_commit.snapshot_id,
        )
    );
    assert_eq!(
        writer
            .commit_truncate(first.table_id, "main", "t", update_commit.snapshot_id,)
            .unwrap(),
        3
    );

    let compact_first = writer
        .begin_write_transaction("main", "compact", &columns, WriteMode::Append)
        .unwrap();
    let compact_first_commit = writer
        .register_data_file(
            compact_first.table_id,
            "main",
            "compact",
            compact_first.snapshot_id,
            &DataFileInfo::new("compact-a.parquet", 20, 2),
            WriteMode::Append,
            compact_first.base_snapshot_id,
            &columns,
            &compact_first.column_ids,
        )
        .unwrap();
    let compact_second = writer
        .begin_write_transaction("main", "compact", &columns, WriteMode::Append)
        .unwrap();
    let compact_second_commit = writer
        .register_data_file(
            compact_second.table_id,
            "main",
            "compact",
            compact_second.snapshot_id,
            &DataFileInfo::new("compact-b.parquet", 30, 3),
            WriteMode::Append,
            compact_second.base_snapshot_id,
            &columns,
            &compact_second.column_ids,
        )
        .unwrap();
    let compact_ids: Vec<i64> = sqlx::query_scalar(
        "SELECT data_file_id FROM ducklake_data_file
         WHERE table_id = ? AND end_snapshot IS NULL ORDER BY data_file_id",
    )
    .bind(compact_first.table_id)
    .fetch_all(&pool)
    .await
    .unwrap();
    let compact_commit = writer
        .commit_compaction(
            compact_first.table_id,
            compact_second_commit.snapshot_id,
            &compact_ids
                .iter()
                .map(|data_file_id| CompactionSourceFile {
                    data_file_id: *data_file_id,
                    delete_file_id: None,
                })
                .collect::<Vec<_>>(),
            &[CompactionOutputFile {
                file: DataFileInfo::new("merged.parquet", 45, 5),
                begin_snapshot: Some(compact_first_commit.snapshot_id),
                partial_max: Some(compact_second_commit.snapshot_id),
            }],
            SourceRetirement::Remove,
        )
        .unwrap();
    let compact_state: (i64, i64, Option<i64>, String) = sqlx::query_as(
        "SELECT
            (SELECT COUNT(*) FROM ducklake_data_file WHERE table_id = ?),
            (SELECT COUNT(*) FROM ducklake_files_scheduled_for_deletion),
            (SELECT partial_max FROM ducklake_data_file WHERE path = 'merged.parquet'),
            (SELECT changes_made FROM ducklake_snapshot_changes WHERE snapshot_id = ?)",
    )
    .bind(compact_first.table_id)
    .bind(compact_commit.snapshot_id)
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(
        compact_state,
        (
            1,
            2,
            Some(compact_second_commit.snapshot_id),
            format!("compacted_table:{}", compact_first.table_id),
        )
    );

    let restore_base = writer
        .begin_write_transaction("main", "restore", &columns, WriteMode::Append)
        .unwrap();
    let restore_base_commit = writer
        .register_data_file(
            restore_base.table_id,
            "main",
            "restore",
            restore_base.snapshot_id,
            &DataFileInfo::new("restore-base.parquet", 20, 2),
            WriteMode::Append,
            restore_base.base_snapshot_id,
            &columns,
            &restore_base.column_ids,
        )
        .unwrap();
    let restore_append = writer
        .begin_write_transaction("main", "restore", &columns, WriteMode::Append)
        .unwrap();
    writer
        .register_data_file(
            restore_append.table_id,
            "main",
            "restore",
            restore_append.snapshot_id,
            &DataFileInfo::new("restore-append.parquet", 30, 3),
            WriteMode::Append,
            restore_append.base_snapshot_id,
            &columns,
            &restore_append.column_ids,
        )
        .unwrap();
    let restore_snapshot = writer
        .retire_appends_since(restore_base.table_id, restore_base_commit.snapshot_id)
        .unwrap()
        .unwrap();
    let restore_state: (Option<i64>, i64, i64) = sqlx::query_as(
        "SELECT
            (SELECT end_snapshot FROM ducklake_data_file WHERE path = 'restore-append.parquet'),
            (SELECT record_count FROM ducklake_table_stats WHERE table_id = ?),
            (SELECT next_row_id FROM ducklake_table_stats WHERE table_id = ?)",
    )
    .bind(restore_base.table_id)
    .bind(restore_base.table_id)
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(restore_state, (Some(restore_snapshot), 2, 5));

    let promote = writer
        .begin_write_transaction("main", "promote", &columns, WriteMode::Append)
        .unwrap();
    writer
        .register_data_file(
            promote.table_id,
            "main",
            "promote",
            promote.snapshot_id,
            &DataFileInfo::new("promote.parquet", 10, 1),
            WriteMode::Append,
            promote.base_snapshot_id,
            &columns,
            &promote.column_ids,
        )
        .unwrap();
    let old_column_id: i64 = sqlx::query_scalar(
        "SELECT column_id FROM ducklake_column
         WHERE table_id = ? AND end_snapshot IS NULL",
    )
    .bind(promote.table_id)
    .fetch_one(&pool)
    .await
    .unwrap();
    let promote_snapshot = writer
        .promote_column_type(promote.table_id, "id", "int64")
        .unwrap();
    let promoted: (i64, String, i64) = sqlx::query_as(
        "SELECT column_id, column_type, begin_snapshot FROM ducklake_column
         WHERE table_id = ? AND end_snapshot IS NULL",
    )
    .bind(promote.table_id)
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(
        promoted,
        (old_column_id, "int64".to_string(), promote_snapshot)
    );

    let expiry = writer
        .begin_write_transaction("main", "expiry", &columns, WriteMode::Append)
        .unwrap();
    let expiry_commit = writer
        .register_data_file(
            expiry.table_id,
            "main",
            "expiry",
            expiry.snapshot_id,
            &DataFileInfo::new("expiry.parquet", 10, 1),
            WriteMode::Append,
            expiry.base_snapshot_id,
            &columns,
            &expiry.column_ids,
        )
        .unwrap();
    writer
        .commit_truncate(expiry.table_id, "main", "expiry", expiry_commit.snapshot_id)
        .unwrap();
    let expired = writer
        .expire_snapshots(ExpireCriteria::Versions(vec![expiry_commit.snapshot_id]))
        .unwrap();
    assert_eq!(expired.len(), 1);
    assert_eq!(expired[0].snapshot_id, expiry_commit.snapshot_id);
    let expiry_state: (i64, i64) = sqlx::query_as(
        "SELECT
            (SELECT COUNT(*) FROM ducklake_data_file WHERE path = 'expiry.parquet'),
            (SELECT COUNT(*) FROM ducklake_files_scheduled_for_deletion
             WHERE path LIKE '%expiry.parquet')",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(expiry_state, (0, 1));
}

/// TRUNCATE of a table whose only live data is inlined rows must end those rows
/// (not no-op on the "no live parquet files" guard) and count them.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn mysql_truncate_ends_inline_only_table() {
    let (_container, writer, pool) = start_writer().await;
    let (setup, commit) = append_inlined(&writer, &[1, 2]);
    let removed = writer
        .commit_truncate(setup.table_id, "main", "t", commit.snapshot_id)
        .unwrap();
    assert_eq!(removed, 2, "the two inline rows are the whole live table");
    let truncate_snapshot = commit.snapshot_id + 1;
    let physical = inlined_table(&pool, setup.table_id).await;
    let state: (i64, i64, i64, i64) = sqlx::query_as(AssertSqlSafe(format!(
        "SELECT
            (SELECT MAX(snapshot_id) FROM ducklake_snapshot),
            (SELECT COUNT(*) FROM `{physical}` WHERE end_snapshot IS NULL),
            (SELECT COUNT(*) FROM `{physical}` WHERE end_snapshot = {truncate_snapshot}),
            (SELECT record_count FROM ducklake_table_stats WHERE table_id = {})",
        setup.table_id
    )))
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(state, (truncate_snapshot, 0, 2, 0));
    assert_eq!(
        changes_made(&pool, truncate_snapshot).await,
        Some(format!("deleted_from_table:{}", setup.table_id))
    );
    // Nothing left to truncate: idempotent no-op with no new snapshot.
    assert_eq!(
        writer
            .commit_truncate(setup.table_id, "main", "t", truncate_snapshot)
            .unwrap(),
        0
    );
}

/// TRUNCATE of a mixed table retires the parquet files AND ends the inlined
/// rows, counting both.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn mysql_truncate_retires_files_and_ends_inline_rows() {
    let (_container, writer, pool) = start_writer().await;
    let (setup, _) = append(&writer, "base.parquet", 3);
    let (_, inline_commit) = append_inlined(&writer, &[10, 11]);
    let removed = writer
        .commit_truncate(setup.table_id, "main", "t", inline_commit.snapshot_id)
        .unwrap();
    assert_eq!(removed, 5, "3 parquet rows + 2 inline rows");
    let truncate_snapshot = inline_commit.snapshot_id + 1;
    let physical = inlined_table(&pool, setup.table_id).await;
    let state: (Option<i64>, i64, i64, i64) = sqlx::query_as(AssertSqlSafe(format!(
        "SELECT
            (SELECT end_snapshot FROM ducklake_data_file WHERE path = 'base.parquet'),
            (SELECT COUNT(*) FROM `{physical}` WHERE end_snapshot IS NULL),
            (SELECT COUNT(*) FROM `{physical}` WHERE end_snapshot = {truncate_snapshot}),
            (SELECT record_count FROM ducklake_table_stats WHERE table_id = {})",
        setup.table_id
    )))
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(state, (Some(truncate_snapshot), 0, 2, 0));
    assert_eq!(
        changes_made(&pool, truncate_snapshot).await,
        Some(format!("deleted_from_table:{}", setup.table_id))
    );
}

/// A DELETE spanning parquet rows (positional delete file) and inlined rows
/// commits in ONE snapshot via the commit_deletes override.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn mysql_commit_deletes_mixes_positional_and_inlined_in_one_snapshot() {
    let (_container, writer, pool) = start_writer().await;
    let (setup, _) = append(&writer, "base.parquet", 3);
    let (_, inline_commit) = append_inlined(&writer, &[10, 11]);
    let source_id = file_id(&pool, "base.parquet").await;
    let physical = inlined_table(&pool, setup.table_id).await;
    // Inline rows follow the parquet file's 3 rows: row_ids 3 and 4.
    let commit = writer
        .commit_deletes(
            setup.table_id,
            "main",
            "t",
            inline_commit.snapshot_id,
            &[DeleteFileEntry {
                data_file_id: source_id,
                expected_prev_delete_file: None,
                delete: DeleteFileInfo::new("delete-1.parquet", 10, 1),
            }],
            &[InlinedRowRef {
                table_name: physical.clone(),
                row_id: 3,
            }],
        )
        .unwrap();
    assert_eq!(commit.snapshot_id, inline_commit.snapshot_id + 1);
    let state: (i64, Option<i64>, Option<i64>, i64) = sqlx::query_as(AssertSqlSafe(format!(
        "SELECT
            (SELECT begin_snapshot FROM ducklake_delete_file WHERE path = 'delete-1.parquet'),
            (SELECT end_snapshot FROM `{physical}` WHERE row_id = 3),
            (SELECT end_snapshot FROM `{physical}` WHERE row_id = 4),
            (SELECT record_count FROM ducklake_table_stats WHERE table_id = {})",
        setup.table_id
    )))
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(
        state,
        (commit.snapshot_id, Some(commit.snapshot_id), None, 4),
        "delete file begins at the commit, inline row 3 ends there, row 4 stays \
         live, and only the inline delete adjusts the gross count"
    );
    assert_eq!(
        changes_made(&pool, commit.snapshot_id).await,
        Some(format!("deleted_from_table:{}", setup.table_id))
    );
}

/// A pure positional-delete snapshot records `deleted_from_table` rather than
/// leaving `changes_made` NULL.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn mysql_positional_delete_snapshot_records_changes_made() {
    let (_container, writer, pool) = start_writer().await;
    let (setup, base) = append(&writer, "a.parquet", 2);
    let commit = writer
        .commit_positional_deletes(
            setup.table_id,
            "main",
            "t",
            base.snapshot_id,
            &[DeleteFileEntry {
                data_file_id: file_id(&pool, "a.parquet").await,
                expected_prev_delete_file: None,
                delete: DeleteFileInfo::new("d.parquet", 10, 1),
            }],
        )
        .unwrap();
    assert_eq!(
        changes_made(&pool, commit.snapshot_id).await,
        Some(format!("deleted_from_table:{}", setup.table_id))
    );
}

/// Staging a table (begin_write_transaction) and abandoning the write leaves a
/// table row with zero live columns; the next write must create the columns at
/// its commit instead of failing with a retryable-looking Conflict.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn mysql_first_write_commits_after_abandoned_staging() {
    let (_container, writer, pool) = start_writer().await;
    let columns = int_column();
    // Stage the table but never write: it persists with zero live columns.
    let staged = writer
        .begin_write_transaction("main", "t", &columns, WriteMode::Append)
        .unwrap();
    // The next write's commit creates the columns rather than conflicting.
    let (setup, commit) = append(&writer, "a.parquet", 2);
    assert_eq!(setup.table_id, staged.table_id);
    assert_eq!(commit.snapshot_id, 1, "abandoned staging left no snapshot");
    let live_columns: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM ducklake_column
         WHERE table_id = ? AND end_snapshot IS NULL",
    )
    .bind(setup.table_id)
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(live_columns, 1);
}

/// Every data/delete-file insert path draws from the explicit counters, so a
/// sequence mixing appends (formerly AUTO_INCREMENT) with UPDATE-style and
/// compaction registrations (always explicit) allocates distinct ids. Under the
/// old split allocators the UPDATE's explicit id would have collided with the
/// first append's auto-increment id on the primary key.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn mysql_unified_file_id_allocation_survives_mixed_paths() {
    let (_container, writer, pool) = start_writer().await;
    let columns = int_column();
    let (setup, _) = append(&writer, "a.parquet", 2);
    append(&writer, "b.parquet", 2);
    let a_id = file_id(&pool, "a.parquet").await;
    let b_id = file_id(&pool, "b.parquet").await;
    assert_eq!((a_id, b_id), (1, 2), "appends allocate from the counter");

    // UPDATE-style commit: register the rewritten file and swap a's delete file
    // in one snapshot. Its explicit ids must not collide with the appends.
    let update = writer
        .begin_write_transaction("main", "t", &columns, WriteMode::Append)
        .unwrap();
    let update_commit = writer
        .register_data_file_with_deletes(
            update.table_id,
            "main",
            "t",
            update.snapshot_id,
            &DataFileInfo::new("c.parquet", 20, 2),
            &[DeleteFileEntry {
                data_file_id: a_id,
                expected_prev_delete_file: None,
                delete: DeleteFileInfo::new("delete-a.parquet", 10, 2),
            }],
            WriteMode::Append,
            update.base_snapshot_id,
            &columns,
            &update.column_ids,
        )
        .unwrap();
    let c_id = file_id(&pool, "c.parquet").await;
    let delete_id: i64 =
        sqlx::query_scalar("SELECT delete_file_id FROM ducklake_delete_file WHERE path = ?")
            .bind("delete-a.parquet")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(
        (c_id, delete_id),
        (3, 1),
        "the UPDATE's data file continues the shared space; delete files have \
         their own counter"
    );
    assert_eq!(
        changes_made(&pool, update_commit.snapshot_id).await,
        Some(format!(
            "deleted_from_table:{0},inserted_into_table:{0}",
            setup.table_id
        ))
    );

    // Compaction outputs draw from the same counter as the appends they retire.
    let compact_commit = writer
        .commit_compaction(
            setup.table_id,
            update_commit.snapshot_id,
            &[
                CompactionSourceFile {
                    data_file_id: a_id,
                    delete_file_id: Some(delete_id),
                },
                CompactionSourceFile {
                    data_file_id: b_id,
                    delete_file_id: None,
                },
                CompactionSourceFile {
                    data_file_id: c_id,
                    delete_file_id: None,
                },
            ],
            &[CompactionOutputFile {
                file: DataFileInfo::new("merged.parquet", 40, 4),
                begin_snapshot: None,
                partial_max: None,
            }],
            SourceRetirement::Retire,
        )
        .unwrap();
    let merged_id = file_id(&pool, "merged.parquet").await;
    assert_eq!(merged_id, 4, "compaction output continues the shared space");
    let (rows, distinct_rows, live): (i64, i64, i64) = sqlx::query_as(
        "SELECT COUNT(*), COUNT(DISTINCT data_file_id),
                COUNT(CASE WHEN end_snapshot IS NULL THEN 1 END)
         FROM ducklake_data_file",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(
        (rows, distinct_rows, live),
        (4, 4, 1),
        "four distinct file ids; only the compaction output is live"
    );
    assert_eq!(
        changes_made(&pool, compact_commit.snapshot_id).await,
        Some(format!("compacted_table:{}", setup.table_id))
    );
}
