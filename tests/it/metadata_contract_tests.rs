#![cfg(feature = "write-sqlite")]
#[cfg(feature = "write-duckdb")]
use std::process::Command;
#[cfg(any(feature = "write-duckdb", feature = "write-postgres"))]
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(any(feature = "write-duckdb", feature = "write-postgres"))]
use arrow::array::{ArrayRef, Int64Array};
use arrow::record_batch::RecordBatch;
#[cfg(any(feature = "write-duckdb", feature = "write-postgres"))]
use datafusion::prelude::SessionContext;
#[cfg(any(feature = "write-duckdb", feature = "write-postgres"))]
use datafusion_ducklake::DuckLakeCatalog;
#[cfg(any(feature = "write-duckdb", feature = "write-postgres"))]
use datafusion_ducklake::DuckLakeTableWriter;
#[cfg(feature = "write-postgres")]
use datafusion_ducklake::MulticatalogManager;
#[cfg(feature = "write-duckdb")]
use datafusion_ducklake::maintenance::{
    CleanupCriteria, ExpireCriteria, cleanup_old_files_duckdb, delete_orphaned_files_duckdb,
};
use datafusion_ducklake::{
    ColumnDef, DataFileInfo, DuckLakeError, MetadataProvider, MetadataWriter,
    SnapshotChangeMetadata, SnapshotCommitMetadata, SqliteMetadataProvider, SqliteMetadataWriter,
    WriteMode,
};
#[cfg(feature = "write-duckdb")]
use datafusion_ducklake::{DuckdbMetadataProvider, DuckdbMetadataWriter};
#[cfg(any(feature = "write-duckdb", feature = "write-postgres"))]
use object_store::ObjectStore;
#[cfg(any(feature = "write-duckdb", feature = "write-postgres"))]
use object_store::local::LocalFileSystem;
#[cfg(any(feature = "write-duckdb", feature = "write-postgres"))]
use std::slice::from_ref;
use tempfile::TempDir;
#[cfg(any(feature = "write-postgres", feature = "write-mysql"))]
use testcontainers::runners::AsyncRunner;
#[cfg(feature = "write-postgres")]
use testcontainers_modules::postgres::Postgres;

#[cfg(any(feature = "write-duckdb", feature = "write-postgres"))]
use datafusion_ducklake::WriteResult;
#[cfg(feature = "write-mysql")]
use datafusion_ducklake::{MySqlMetadataProvider, MySqlMetadataWriter};
#[cfg(feature = "write-postgres")]
use datafusion_ducklake::{PostgresMetadataProvider, PostgresSingleCatalogMetadataWriter};
#[cfg(feature = "write-mysql")]
use testcontainers_modules::mysql::Mysql;

#[cfg(feature = "write-postgres")]
use arrow::array::{
    Array, Decimal128Array, ListArray, StructArray, TimestampNanosecondArray, UInt32Array,
    UInt64Array,
};
#[cfg(feature = "write-postgres")]
use arrow::buffer::{OffsetBuffer, ScalarBuffer};
#[cfg(feature = "write-postgres")]
use arrow::datatypes::{DataType, Field, Fields, Schema, TimeUnit};
#[cfg(any(feature = "write-duckdb", feature = "write-postgres"))]
use datafusion_ducklake::DuckLakeWriteOptions;

fn columns() -> Vec<ColumnDef> {
    vec![ColumnDef::new("value", "BIGINT", false).unwrap()]
}

#[cfg(any(feature = "write-duckdb", feature = "write-postgres"))]
fn batch(values: Vec<i64>) -> RecordBatch {
    RecordBatch::try_from_iter(vec![(
        "value",
        Arc::new(Int64Array::from(values)) as ArrayRef,
    )])
    .unwrap()
}

fn write_contract_data(writer: &dyn MetadataWriter, identity: &str) -> (i64, i64) {
    let setup = writer
        .begin_write_transaction("main", "events", &columns(), WriteMode::Replace)
        .unwrap();
    let commit = writer
        .register_data_file_with_commit_metadata(
            setup.table_id,
            "main",
            "events",
            setup.snapshot_id,
            &DataFileInfo::new("events.parquet", 128, 1),
            WriteMode::Replace,
            setup.base_snapshot_id,
            &columns(),
            &setup.column_ids,
            &SnapshotCommitMetadata::new()
                .with_author("contract")
                .with_message("metadata contract")
                .with_extra_info(identity),
            None,
        )
        .unwrap();
    (setup.table_id, commit.snapshot_id)
}

fn assert_metadata_contract(
    provider: &dyn MetadataProvider,
    writer: &dyn MetadataWriter,
    identity: &str,
    table_id: i64,
    snapshot_id: i64,
    test_global_settings: bool,
) {
    let error = writer
        .set_table_setting(table_id, "misspelled_option", "42")
        .unwrap_err();
    assert_eq!(
        error.to_string(),
        "Unsupported feature: unsupported table setting 'misspelled_option'"
    );
    assert_eq!(
        provider
            .find_snapshot_by_commit_extra_info(&identity.to_ascii_uppercase())
            .unwrap(),
        None
    );
    let changes = provider.list_snapshot_changes().unwrap();
    let change = changes
        .iter()
        .find(|change| change.snapshot_id == snapshot_id)
        .unwrap();
    assert_eq!(change.author.as_deref(), Some("contract"));
    assert_eq!(change.commit_message.as_deref(), Some("metadata contract"));
    assert_eq!(change.commit_extra_info.as_deref(), Some(identity));
    assert_eq!(
        provider
            .find_snapshot_by_commit_extra_info(identity)
            .unwrap(),
        Some(snapshot_id),
    );

    if test_global_settings {
        writer
            .set_global_setting("data_inlining_row_limit", "17")
            .unwrap();
        assert_eq!(
            provider
                .get_metadata_settings(None, None)
                .unwrap()
                .get("data_inlining_row_limit")
                .map(String::as_str),
            Some("17"),
        );
    }
    writer
        .set_table_setting(table_id, "DATA_INLINING_ROW_LIMIT", "42")
        .unwrap();
    assert_eq!(
        provider
            .get_metadata_settings(None, Some(table_id))
            .unwrap()
            .get("data_inlining_row_limit")
            .map(String::as_str),
        Some("42"),
    );

    let called = AtomicBool::new(false);
    writer
        .with_commit_lock(
            identity,
            Box::new(|| {
                assert_eq!(provider.get_current_snapshot()?, snapshot_id);
                called.store(true, Ordering::SeqCst);
                Ok(())
            }),
        )
        .unwrap();
    assert!(called.load(Ordering::SeqCst));

    let error = writer
        .with_commit_lock(
            identity,
            Box::new(|| Err(DuckLakeError::Internal("operation failed".to_string()))),
        )
        .unwrap_err();
    assert_eq!(error.to_string(), "Internal error: operation failed");

    // The failed operation released the lock: re-acquiring under the same
    // identity succeeds.
    writer
        .with_commit_lock(identity, Box::new(|| Ok(())))
        .unwrap();
}

#[cfg(feature = "write-duckdb")]
#[tokio::test]
async fn duckdb_metadata_contract() {
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("catalog.ducklake");
    let writer = DuckdbMetadataWriter::new_with_init(path.to_str().unwrap()).unwrap();
    writer.set_data_path(temp.path().to_str().unwrap()).unwrap();
    let (table_id, snapshot_id) = write_contract_data(&writer, "duckdb-contract");

    writer
        .set_global_setting("data_inlining_row_limit", "17")
        .unwrap();
    writer
        .set_table_setting(table_id, "data_inlining_row_limit", "42")
        .unwrap();
    let called = AtomicBool::new(false);
    writer
        .with_commit_lock(
            "duckdb-contract",
            Box::new(|| {
                called.store(true, Ordering::SeqCst);
                Ok(())
            }),
        )
        .unwrap();
    assert!(called.load(Ordering::SeqCst));
    let error = writer
        .with_commit_lock(
            "duckdb-contract",
            Box::new(|| Err(DuckLakeError::Internal("operation failed".to_string()))),
        )
        .unwrap_err();
    assert_eq!(error.to_string(), "Internal error: operation failed");
    writer
        .with_commit_lock("duckdb-contract", Box::new(|| Ok(())))
        .unwrap();

    let setup = writer
        .begin_write_transaction("main", "events", &columns(), WriteMode::Append)
        .unwrap();
    let inline_commit = writer
        .register_inlined_data(
            setup.table_id,
            "main",
            "events",
            setup.snapshot_id,
            &[batch(vec![99])],
            WriteMode::Append,
            setup.base_snapshot_id,
            &columns(),
            &setup.column_ids,
            &SnapshotCommitMetadata::new(),
            Some(setup.base_snapshot_id),
        )
        .unwrap();
    let shared_provider = writer.metadata_provider();
    assert_eq!(
        shared_provider
            .list_snapshot_changes()
            .unwrap()
            .last()
            .unwrap()
            .changes_made,
        Some(format!("inlined_insert:{table_id}"))
    );
    assert_eq!(
        shared_provider.get_current_snapshot().unwrap(),
        inline_commit.snapshot_id,
    );
    drop(shared_provider);
    drop(writer);

    let provider = DuckdbMetadataProvider::new(path.to_str().unwrap()).unwrap();
    let changes = provider.list_snapshot_changes().unwrap();
    let change = changes
        .iter()
        .find(|change| change.snapshot_id == snapshot_id)
        .unwrap();
    assert_eq!(change.author.as_deref(), Some("contract"));
    assert_eq!(change.commit_message.as_deref(), Some("metadata contract"));
    assert_eq!(change.commit_extra_info.as_deref(), Some("duckdb-contract"));
    assert_eq!(
        provider
            .find_snapshot_by_commit_extra_info("duckdb-contract")
            .unwrap(),
        Some(snapshot_id),
    );
    assert_eq!(
        provider
            .get_metadata_settings(None, None)
            .unwrap()
            .get("data_inlining_row_limit")
            .map(String::as_str),
        Some("17"),
    );
    assert_eq!(
        provider
            .get_metadata_settings(None, Some(table_id))
            .unwrap()
            .get("data_inlining_row_limit")
            .map(String::as_str),
        Some("42"),
    );
    let table_columns = provider
        .get_table_structure(table_id, inline_commit.snapshot_id)
        .unwrap();
    let inlined = provider
        .get_inlined_data_with_row_ids(table_id, inline_commit.snapshot_id, &table_columns)
        .unwrap();
    assert_eq!(inlined.len(), 1);
    assert_eq!(inlined[0].row_ids, vec![1]);
    assert_eq!(inlined[0].begin_snapshots, vec![inline_commit.snapshot_id],);
    assert_eq!(
        inlined[0]
            .batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .values(),
        &[99],
    );
}

#[cfg(feature = "write-duckdb")]
#[tokio::test]
async fn duckdb_commit_lock_child() {
    let Ok(path) = std::env::var("DUCKLAKE_CRASH_LOCK_PATH") else {
        return;
    };
    let writer = DuckdbMetadataWriter::new(path).unwrap();
    writer
        .with_commit_lock("crashed-holder", Box::new(|| std::process::exit(17)))
        .unwrap();
}

#[cfg(feature = "write-duckdb")]
#[tokio::test]
async fn duckdb_commit_lock_survives_crashed_holder() {
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("crash-lock.ducklake");
    drop(DuckdbMetadataWriter::new_with_init(path.to_str().unwrap()).unwrap());
    let status = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "metadata_contract_tests::duckdb_commit_lock_child", "--nocapture"])
        .env("DUCKLAKE_CRASH_LOCK_PATH", path.as_os_str())
        .status()
        .unwrap();
    assert_eq!(status.code(), Some(17));

    let writer = DuckdbMetadataWriter::new(path.to_str().unwrap()).unwrap();
    writer
        .with_commit_lock("crashed-holder", Box::new(|| Ok(())))
        .unwrap();
}

#[cfg(feature = "write-duckdb")]
#[tokio::test]
async fn duckdb_maintenance_contract() {
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("maintenance.ducklake");
    let data_path = temp.path().join("data");
    let table_path = data_path.join("main").join("events");
    std::fs::create_dir_all(&table_path).unwrap();
    let writer = DuckdbMetadataWriter::new_with_init(path.to_str().unwrap()).unwrap();
    writer.set_data_path(data_path.to_str().unwrap()).unwrap();
    let first = writer
        .begin_write_transaction("main", "events", &columns(), WriteMode::Replace)
        .unwrap();
    let first_commit = writer
        .register_data_file(
            first.table_id,
            "main",
            "events",
            first.snapshot_id,
            &DataFileInfo::new("first.parquet", 5, 1),
            WriteMode::Replace,
            first.base_snapshot_id,
            &columns(),
            &first.column_ids,
        )
        .unwrap();
    std::fs::write(table_path.join("first.parquet"), b"first").unwrap();
    let second = writer
        .begin_write_transaction("main", "events", &columns(), WriteMode::Replace)
        .unwrap();
    writer
        .register_data_file(
            second.table_id,
            "main",
            "events",
            second.snapshot_id,
            &DataFileInfo::new("second.parquet", 6, 1),
            WriteMode::Replace,
            second.base_snapshot_id,
            &columns(),
            &second.column_ids,
        )
        .unwrap();
    std::fs::write(table_path.join("second.parquet"), b"second").unwrap();
    assert_eq!(
        writer
            .expire_snapshots(ExpireCriteria::Versions(vec![first_commit.snapshot_id]))
            .unwrap()
            .len(),
        1,
    );
    let object_store: Arc<dyn ObjectStore> = Arc::new(LocalFileSystem::new());
    let dry_run = cleanup_old_files_duckdb(
        &writer,
        Arc::clone(&object_store),
        CleanupCriteria::All,
        true,
    )
    .await
    .unwrap();
    assert_eq!(dry_run.len(), 1);
    assert!(table_path.join("first.parquet").exists());
    let deleted = cleanup_old_files_duckdb(
        &writer,
        Arc::clone(&object_store),
        CleanupCriteria::All,
        false,
    )
    .await
    .unwrap();
    assert_eq!(deleted, dry_run);
    assert!(!table_path.join("first.parquet").exists());
    assert!(table_path.join("second.parquet").exists());

    let orphan = table_path.join("orphan.parquet");
    std::fs::write(&orphan, b"orphan").unwrap();
    let deleted = delete_orphaned_files_duckdb(&writer, object_store, CleanupCriteria::All, false)
        .await
        .unwrap();
    assert_eq!(deleted.len(), 1);
    assert!(deleted[0].ends_with("main/events/orphan.parquet"));
    assert!(!orphan.exists());
    assert!(table_path.join("second.parquet").exists());
}

#[tokio::test(flavor = "multi_thread")]
async fn sqlite_metadata_contract() {
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("catalog.sqlite");
    let url = format!("sqlite:{}?mode=rwc", path.display());
    let writer = SqliteMetadataWriter::new_with_init(&url).await.unwrap();
    writer.set_data_path(temp.path().to_str().unwrap()).unwrap();
    let provider = SqliteMetadataProvider::new(&url).await.unwrap();
    let (table_id, snapshot_id) = write_contract_data(&writer, "sqlite-contract");

    assert_metadata_contract(
        &provider,
        &writer,
        "sqlite-contract",
        table_id,
        snapshot_id,
        true,
    );
    let pool = sqlx::SqlitePool::connect(&url).await.unwrap();
    sqlx::query("UPDATE ducklake_table SET end_snapshot = ? WHERE table_id = ?")
        .bind(snapshot_id)
        .bind(table_id)
        .execute(&pool)
        .await
        .unwrap();
    assert!(
        matches!(writer.set_table_setting(table_id, "auto_compact", "true"), Err(DuckLakeError::TableNotFound(id)) if id == table_id.to_string())
    );
}

#[cfg(feature = "write-postgres")]
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn postgres_metadata_contract() {
    let container = Postgres::default().start().await.unwrap();
    let port = container.get_host_port_ipv4(5432).await.unwrap();
    let url = format!("postgresql://postgres:postgres@127.0.0.1:{port}/postgres");
    let manager = MulticatalogManager::connect(&url, 5).await.unwrap();
    let catalog_id = manager.create_catalog("metadata_contract").await.unwrap();
    let pool = sqlx::PgPool::connect(&url).await.unwrap();
    sqlx::query("DROP TABLE ducklake_inlined_data_tables")
        .execute(&pool)
        .await
        .unwrap();
    let writer = manager.writer(catalog_id).await.unwrap();
    writer.set_data_path("/tmp/metadata-contract").unwrap();
    let provider = manager.provider(catalog_id).await.unwrap();
    let (table_id, snapshot_id) = write_contract_data(&writer, "postgres-contract");

    assert_metadata_contract(
        &provider,
        &writer,
        "postgres-contract",
        table_id,
        snapshot_id,
        true,
    );
    let second_id = manager
        .create_catalog("metadata_contract_two")
        .await
        .unwrap();
    let second_writer = manager.writer(second_id).await.unwrap();
    second_writer
        .set_global_setting("data_inlining_row_limit", "99")
        .unwrap();
    let second_provider = manager.provider(second_id).await.unwrap();
    assert_eq!(
        provider
            .get_metadata_settings(None, None)
            .unwrap()
            .get("data_inlining_row_limit")
            .map(String::as_str),
        Some("17"),
    );
    assert_eq!(
        second_provider
            .get_metadata_settings(None, None)
            .unwrap()
            .get("data_inlining_row_limit")
            .map(String::as_str),
        Some("99"),
    );
}

#[cfg(feature = "write-postgres")]
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn postgres_flush_inlined_data() {
    let container = Postgres::default().start().await.unwrap();
    let port = container.get_host_port_ipv4(5432).await.unwrap();
    let url = format!("postgresql://postgres:postgres@127.0.0.1:{port}/postgres");
    let manager = MulticatalogManager::connect(&url, 5).await.unwrap();
    let catalog_id = manager.create_catalog("flush_contract").await.unwrap();
    let writer = Arc::new(manager.writer(catalog_id).await.unwrap());
    let temp = TempDir::new().unwrap();
    let data_path = temp.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();
    writer.set_data_path(data_path.to_str().unwrap()).unwrap();
    let object_store: Arc<dyn ObjectStore> = Arc::new(LocalFileSystem::new());
    let setup = writer
        .begin_write_transaction("main", "events", &columns(), WriteMode::Append)
        .unwrap();
    let inline_write = writer
        .register_inlined_data(
            setup.table_id,
            "main",
            "events",
            setup.snapshot_id,
            &[batch(vec![5, 8])],
            WriteMode::Append,
            setup.base_snapshot_id,
            &columns(),
            &setup.column_ids,
            &SnapshotCommitMetadata::default(),
            None,
        )
        .unwrap();
    let provider = manager.provider(catalog_id).await.unwrap();
    let schema = provider
        .get_schema_by_name("main", inline_write.snapshot_id)
        .unwrap()
        .unwrap();
    let table = provider
        .get_table_by_name(schema.schema_id, "events", inline_write.snapshot_id)
        .unwrap()
        .unwrap();
    let columns = provider
        .get_table_structure(table.table_id, inline_write.snapshot_id)
        .unwrap();
    let inlined = provider
        .get_inlined_data_with_row_ids(table.table_id, inline_write.snapshot_id, &columns)
        .unwrap();
    let flush_writer = DuckLakeTableWriter::new(writer, object_store).unwrap();
    let flushed = flush_writer
        .flush_inlined_data("main", "events", &inlined, inline_write.snapshot_id)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        provider
            .list_snapshot_changes()
            .unwrap()
            .last()
            .unwrap()
            .changes_made,
        Some(format!("inline_flush:{}", table.table_id))
    );
    assert_eq!(flushed.records_written, 2);
    assert_eq!(flushed.files_written, 1);
    assert!(
        provider
            .get_inlined_data_with_row_ids(table.table_id, flushed.snapshot_id, &columns)
            .unwrap()
            .is_empty(),
    );
    assert_eq!(
        provider
            .get_inlined_data_with_row_ids(table.table_id, inline_write.snapshot_id, &columns,)
            .unwrap()
            .iter()
            .map(|data| data.batch.num_rows())
            .sum::<usize>(),
        2,
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn sqlite_snapshot_changes_preserve_null_and_missing_ledger_rows() {
    let temp = TempDir::new().unwrap();
    let url = format!(
        "sqlite:{}?mode=rwc",
        temp.path().join("catalog.sqlite").display()
    );
    let writer = SqliteMetadataWriter::new_with_init(&url).await.unwrap();
    let first = writer.create_snapshot().unwrap();
    let second = writer.create_snapshot().unwrap();
    let pool = sqlx::SqlitePool::connect(&url).await.unwrap();
    sqlx::query("DELETE FROM ducklake_snapshot_changes WHERE snapshot_id = ?")
        .bind(second)
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query("UPDATE ducklake_snapshot SET snapshot_time = '2026-01-02 03:04:05.123456'")
        .execute(&pool)
        .await
        .unwrap();
    let provider = SqliteMetadataProvider::new(&url).await.unwrap();
    let snapshots = provider.list_snapshots().unwrap();
    let changes = provider.list_snapshot_changes().unwrap();
    let expected: Vec<_> = snapshots
        .into_iter()
        .map(|snapshot| SnapshotChangeMetadata {
            snapshot_id: snapshot.snapshot_id,
            timestamp: snapshot.timestamp,
            changes_made: None,
            author: None,
            commit_message: None,
            commit_extra_info: None,
        })
        .collect();
    assert_eq!(second, first + 1);
    assert_eq!(changes, expected);
}

#[tokio::test(flavor = "multi_thread")]
async fn sqlite_reopens_legacy_catalog_without_schema_initialization() {
    let temp = TempDir::new().unwrap();
    let url = format!(
        "sqlite:{}?mode=rwc",
        temp.path().join("catalog.sqlite").display()
    );
    let writer = SqliteMetadataWriter::new_with_init(&url).await.unwrap();
    let (table_id, _) = write_contract_data(&writer, "legacy");
    let empty_snapshot = writer.create_snapshot().unwrap();
    drop(writer);
    let pool = sqlx::SqlitePool::connect(&url).await.unwrap();
    sqlx::query("DROP TABLE ducklake_inlined_data_tables")
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query("ALTER TABLE ducklake_snapshot_changes RENAME TO legacy_changes")
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query("CREATE TABLE ducklake_snapshot_changes (snapshot_id INTEGER PRIMARY KEY, changes_made VARCHAR NOT NULL, author VARCHAR, commit_message VARCHAR, commit_extra_info VARCHAR)").execute(&pool).await.unwrap();
    sqlx::query("INSERT INTO ducklake_snapshot_changes SELECT snapshot_id, COALESCE(changes_made, ''), author, commit_message, commit_extra_info FROM legacy_changes").execute(&pool).await.unwrap();
    sqlx::query("DROP TABLE legacy_changes")
        .execute(&pool)
        .await
        .unwrap();
    let writer = SqliteMetadataWriter::new(&url).await.unwrap();
    let (replaced_table_id, replaced_snapshot) = write_contract_data(&writer, "replacement");
    let removed = writer
        .commit_truncate(table_id, "main", "events", replaced_snapshot)
        .unwrap();
    let provider = SqliteMetadataProvider::new(&url).await.unwrap();
    let changes = provider.list_snapshot_changes().unwrap();
    let migrated = changes
        .iter()
        .find(|change| change.snapshot_id == empty_snapshot)
        .unwrap();
    assert_eq!(replaced_table_id, table_id);
    assert_eq!(removed, 1);
    assert_eq!(migrated.changes_made, None);
    assert_eq!(
        provider
            .get_inlined_data(
                table_id,
                provider.get_current_snapshot().unwrap(),
                &provider
                    .get_table_structure(table_id, provider.get_current_snapshot().unwrap())
                    .unwrap()
            )
            .unwrap(),
        Vec::<RecordBatch>::new()
    );
}

#[cfg(feature = "write-duckdb")]
#[rstest::rstest]
#[case::existing_tables(true)]
#[case::new_tables(false)]
#[tokio::test(flavor = "multi_thread")]
async fn duckdb_multi_table_commit_reads_both_tables(#[case] existing: bool) {
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("catalog.ducklake");
    let writer = Arc::new(DuckdbMetadataWriter::new_with_init(path.to_str().unwrap()).unwrap());
    writer.set_data_path(temp.path().to_str().unwrap()).unwrap();
    let results = commit_tables(writer.clone(), existing).await;
    let provider = writer.metadata_provider();
    let changes = provider.list_snapshot_changes().unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("test", Arc::new(DuckLakeCatalog::new(provider).unwrap()));
    let actual = ctx.sql("SELECT value FROM test.main.first UNION ALL SELECT value FROM test.main.second ORDER BY value").await.unwrap().collect().await.unwrap();
    let values: Vec<i64> = actual
        .iter()
        .flat_map(|batch| {
            batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .values()
                .to_vec()
        })
        .collect();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].snapshot_id, results[1].snapshot_id);
    assert_eq!(
        (results[0].files_written, results[0].records_written),
        (1, 2)
    );
    assert_eq!(
        (results[1].files_written, results[1].records_written),
        (1, 1)
    );
    assert_eq!(
        values,
        if existing {
            vec![11, 11, 23, 23, 37, 37]
        } else {
            vec![11, 23, 37]
        }
    );
    let expected = if existing {
        format!(
            "inserted_into_table:{},inserted_into_table:{}",
            results[0].table_id, results[1].table_id
        )
    } else {
        format!(
            "created_schema:\"main\",created_table:\"main\".\"first\",inserted_into_table:{},created_table:\"main\".\"second\",inserted_into_table:{}",
            results[0].table_id, results[1].table_id
        )
    };
    assert_eq!(
        changes.last().unwrap().changes_made.as_deref(),
        Some(expected.as_str())
    );
}

#[cfg(feature = "write-postgres")]
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn postgres_single_metadata_contract() {
    let container = Postgres::default().start().await.unwrap();
    let port = container.get_host_port_ipv4(5432).await.unwrap();
    let url = format!("postgresql://postgres:postgres@127.0.0.1:{port}/postgres");
    let writer = PostgresSingleCatalogMetadataWriter::new_with_init(&url)
        .await
        .unwrap();
    let provider = PostgresMetadataProvider::new(&url).await.unwrap();
    let (table_id, snapshot) = write_contract_data(&writer, "single-contract");
    assert_metadata_contract(
        &provider,
        &writer,
        "single-contract",
        table_id,
        snapshot,
        true,
    );
}

#[cfg(feature = "write-mysql")]
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn mysql_metadata_contract() {
    let container = Mysql::default().start().await.unwrap();
    let port = container.get_host_port_ipv4(3306).await.unwrap();
    let url = format!("mysql://root@127.0.0.1:{port}/test");
    let writer = MySqlMetadataWriter::new_with_init(&url).await.unwrap();
    let provider = MySqlMetadataProvider::new(&url).await.unwrap();
    let (table_id, snapshot) = write_contract_data(&writer, "mysql-contract");
    assert_metadata_contract(
        &provider,
        &writer,
        "mysql-contract",
        table_id,
        snapshot,
        false,
    );
}

#[cfg(feature = "write-postgres")]
#[rstest::rstest]
#[case::existing_tables(true)]
#[case::new_tables(false)]
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn postgres_single_multi_table_commit_reads_both_tables(#[case] existing: bool) {
    let container = Postgres::default().start().await.unwrap();
    let port = container.get_host_port_ipv4(5432).await.unwrap();
    let url = format!("postgresql://postgres:postgres@127.0.0.1:{port}/postgres");
    let writer = Arc::new(
        PostgresSingleCatalogMetadataWriter::new_with_init(&url)
            .await
            .unwrap(),
    );
    let temp = TempDir::new().unwrap();
    writer.set_data_path(temp.path().to_str().unwrap()).unwrap();
    let results = commit_tables(writer, existing).await;
    let provider = PostgresMetadataProvider::new(&url).await.unwrap();
    let changes = provider.list_snapshot_changes().unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("test", Arc::new(DuckLakeCatalog::new(provider).unwrap()));
    let actual = ctx.sql("SELECT value FROM test.main.first UNION ALL SELECT value FROM test.main.second ORDER BY value").await.unwrap().collect().await.unwrap();
    let values: Vec<i64> = actual
        .iter()
        .flat_map(|batch| {
            batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .values()
                .to_vec()
        })
        .collect();
    let expected = if existing {
        format!(
            "inserted_into_table:{},inserted_into_table:{}",
            results[0].table_id, results[1].table_id
        )
    } else {
        format!(
            "created_schema:\"main\",created_table:\"main\".\"first\",inserted_into_table:{},created_table:\"main\".\"second\",inserted_into_table:{}",
            results[0].table_id, results[1].table_id
        )
    };
    assert_eq!(results[0].snapshot_id, results[1].snapshot_id);
    assert_eq!(
        values,
        if existing {
            vec![11, 11, 23, 23, 37, 37]
        } else {
            vec![11, 23, 37]
        }
    );
    assert_eq!(
        changes.last().unwrap().changes_made.as_deref(),
        Some(expected.as_str())
    );
}

#[cfg(any(feature = "write-duckdb", feature = "write-postgres"))]
async fn commit_tables(writer: Arc<dyn MetadataWriter>, existing: bool) -> Vec<WriteResult> {
    let table_writer =
        DuckLakeTableWriter::new(writer.clone(), Arc::new(LocalFileSystem::new())).unwrap();
    let first = batch(vec![11, 23]);
    let second = batch(vec![37]);
    if existing {
        for (name, rows) in [("first", &first), ("second", &second)] {
            table_writer
                .append_table("main", name, from_ref(rows))
                .await
                .unwrap();
        }
    }
    let mut transaction = table_writer.transaction();
    for (name, rows) in [("first", &first), ("second", &second)] {
        transaction
            .stage_write(
                "main",
                name,
                rows.schema().as_ref(),
                WriteMode::Append,
                from_ref(rows),
            )
            .await
            .unwrap();
    }
    transaction.commit().await.unwrap()
}

#[cfg(feature = "write-postgres")]
fn nested_batch() -> RecordBatch {
    let fields = Fields::from(vec![
        Field::new("price", DataType::Decimal128(10, 2), true),
        Field::new(
            "event_time",
            DataType::Timestamp(TimeUnit::Nanosecond, Some("UTC".into())),
            true,
        ),
        Field::new("count", DataType::UInt32, true),
        Field::new("order_id", DataType::UInt64, true),
    ]);
    let values = StructArray::new(
        fields.clone(),
        vec![
            Arc::new(
                Decimal128Array::from(vec![12_345, 67_890])
                    .with_precision_and_scale(10, 2)
                    .unwrap(),
            ) as ArrayRef,
            Arc::new(
                TimestampNanosecondArray::from(vec![1_000_002, 2_000_003]).with_timezone("UTC"),
            ) as ArrayRef,
            Arc::new(UInt32Array::from(vec![1, 2])) as ArrayRef,
            Arc::new(UInt64Array::from(vec![11, 22])) as ArrayRef,
        ],
        None,
    );
    let depths = ListArray::new(
        Arc::new(Field::new("item", DataType::Struct(fields), true)),
        OffsetBuffer::new(ScalarBuffer::from(vec![0_i32, 2])),
        Arc::new(values),
        None,
    );
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "depths",
            depths.data_type().clone(),
            true,
        )])),
        vec![Arc::new(depths)],
    )
    .unwrap()
}

#[cfg(feature = "write-duckdb")]
#[tokio::test]
async fn duckdb_flush_inlined_data() {
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("flush.ducklake");
    let data_path = temp.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();
    let writer = Arc::new(DuckdbMetadataWriter::new_with_init(path.to_str().unwrap()).unwrap());
    writer.set_data_path(data_path.to_str().unwrap()).unwrap();
    let object_store: Arc<dyn ObjectStore> = Arc::new(LocalFileSystem::new());
    let options = DuckLakeWriteOptions::default().with_data_inlining_row_limit(10);
    let table_writer = DuckLakeTableWriter::new(writer.clone(), Arc::clone(&object_store))
        .unwrap()
        .with_options(&options);
    let inline_write = table_writer
        .append_table("main", "events", &[batch(vec![7, 11])])
        .await
        .unwrap();
    drop(table_writer);
    drop(writer);

    let provider = DuckdbMetadataProvider::new(path.to_str().unwrap()).unwrap();
    let schema = provider
        .get_schema_by_name("main", inline_write.snapshot_id)
        .unwrap()
        .unwrap();
    let table = provider
        .get_table_by_name(schema.schema_id, "events", inline_write.snapshot_id)
        .unwrap()
        .unwrap();
    let columns = provider
        .get_table_structure(table.table_id, inline_write.snapshot_id)
        .unwrap();
    let inlined = provider
        .get_inlined_data_with_row_ids(table.table_id, inline_write.snapshot_id, &columns)
        .unwrap();
    assert_eq!(
        inlined
            .iter()
            .map(|data| data.batch.num_rows())
            .sum::<usize>(),
        2,
    );
    drop(provider);

    let writer = Arc::new(DuckdbMetadataWriter::new(path.to_str().unwrap()).unwrap());
    let table_writer = DuckLakeTableWriter::new(writer.clone(), object_store).unwrap();
    let flushed = table_writer
        .flush_inlined_data("main", "events", &inlined, inline_write.snapshot_id)
        .await
        .unwrap()
        .unwrap();
    drop(table_writer);
    drop(writer);

    let provider = DuckdbMetadataProvider::new(path.to_str().unwrap()).unwrap();
    assert!(
        provider
            .get_inlined_data_with_row_ids(table.table_id, flushed.snapshot_id, &columns)
            .unwrap()
            .is_empty(),
    );
    assert_eq!(
        provider
            .get_inlined_data_with_row_ids(table.table_id, inline_write.snapshot_id, &columns,)
            .unwrap()
            .iter()
            .map(|data| data.batch.num_rows())
            .sum::<usize>(),
        2,
    );
    assert_eq!(
        provider
            .get_table_files_for_select(table.table_id, flushed.snapshot_id)
            .unwrap()
            .len(),
        1,
    );
    let catalog = DuckLakeCatalog::new(provider).unwrap();
    let context = SessionContext::new();
    context.register_catalog("ducklake", Arc::new(catalog));
    let batches = context
        .sql("SELECT value FROM ducklake.main.events ORDER BY value")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(batches.len(), 1);
    assert_eq!(
        batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .values(),
        &[7, 11],
    );
}

#[cfg(feature = "write-postgres")]
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn postgres_round_trips_nested_inlined_rows() {
    let container = Postgres::default().start().await.unwrap();
    let port = container.get_host_port_ipv4(5432).await.unwrap();
    let url = format!("postgresql://postgres:postgres@127.0.0.1:{port}/postgres");
    let manager = MulticatalogManager::connect(&url, 5).await.unwrap();
    let catalog_id = manager.create_catalog("nested_contract").await.unwrap();
    let writer = Arc::new(manager.writer(catalog_id).await.unwrap());
    let temp = TempDir::new().unwrap();
    let data_path = temp.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();
    writer.set_data_path(data_path.to_str().unwrap()).unwrap();
    let expected = nested_batch();
    let written = DuckLakeTableWriter::new(writer, Arc::new(LocalFileSystem::new()))
        .unwrap()
        .with_options(&DuckLakeWriteOptions::default().with_data_inlining_row_limit(1))
        .write_table("main", "depths", std::slice::from_ref(&expected))
        .await
        .unwrap();
    assert_eq!(written.files_written, 0);

    let provider = manager.provider(catalog_id).await.unwrap();
    let schema = provider
        .get_schema_by_name("main", written.snapshot_id)
        .unwrap()
        .unwrap();
    let table = provider
        .get_table_by_name(schema.schema_id, "depths", written.snapshot_id)
        .unwrap()
        .unwrap();
    let columns = provider
        .get_table_structure(table.table_id, written.snapshot_id)
        .unwrap();
    let batches = provider
        .get_inlined_data(table.table_id, written.snapshot_id, &columns)
        .unwrap();
    assert_eq!(batches, vec![expected]);
}
