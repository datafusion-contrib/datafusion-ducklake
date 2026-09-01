#![cfg(all(feature = "write-sqlite", feature = "write-postgres"))]
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use arrow::array::{ArrayRef, Int64Array};
use arrow::record_batch::RecordBatch;
use datafusion_ducklake::maintenance::{
    CleanupCriteria, ExpireCriteria, cleanup_old_files_duckdb, delete_orphaned_files_duckdb,
};
use datafusion_ducklake::{
    ColumnDef, DataFileInfo, DuckLakeError, DuckLakeTableWriter, DuckLakeWriteOptions,
    DuckdbMetadataProvider, DuckdbMetadataWriter, MetadataProvider, MetadataWriter,
    MulticatalogManager, SnapshotCommitMetadata, SqliteMetadataProvider, SqliteMetadataWriter,
    WriteMode,
};
use object_store::ObjectStore;
use object_store::local::LocalFileSystem;
use tempfile::TempDir;
use testcontainers::runners::AsyncRunner;
use testcontainers_modules::postgres::Postgres;

fn columns() -> Vec<ColumnDef> {
    vec![ColumnDef::new("value", "BIGINT", false).unwrap()]
}

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
) {
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
    writer
        .set_table_setting(table_id, "data_inlining_row_limit", "42")
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

    assert_metadata_contract(&provider, &writer, "sqlite-contract", table_id, snapshot_id);
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn postgres_metadata_contract() {
    let container = Postgres::default().start().await.unwrap();
    let port = container.get_host_port_ipv4(5432).await.unwrap();
    let url = format!("postgresql://postgres:postgres@127.0.0.1:{port}/postgres");
    let manager = MulticatalogManager::connect(&url, 5).await.unwrap();
    let catalog_id = manager.create_catalog("metadata_contract").await.unwrap();
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
    let options = DuckLakeWriteOptions::default().with_data_inlining_row_limit(10);
    let table_writer = DuckLakeTableWriter::new(writer.clone(), Arc::clone(&object_store))
        .unwrap()
        .with_options(&options);
    let inline_write = table_writer
        .append_table("main", "events", &[batch(vec![5, 8])])
        .await
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
