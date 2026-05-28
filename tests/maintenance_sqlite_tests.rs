//! Integration tests for single-catalog maintenance on the SQLite writer:
//! `drop_table`, `expire_snapshots`, and `cleanup_old_files_sqlite`.
//!
//! Mirrors the upstream DuckLake suite
//! (`test/sql/catalog/drop_table.test`, `test/sql/compaction/expire_snapshots*.test`).

#![cfg(all(feature = "write-sqlite", feature = "metadata-sqlite"))]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use object_store::ObjectStore;
use object_store::local::LocalFileSystem;
use sqlx::Row;
use sqlx::sqlite::SqlitePool;
use tempfile::TempDir;

use datafusion_ducklake::SqliteMetadataWriter;
use datafusion_ducklake::maintenance::{
    CleanupCriteria, ExpireCriteria, cleanup_old_files_sqlite, delete_orphaned_files_sqlite,
};
use datafusion_ducklake::metadata_writer::{ColumnDef, DataFileInfo, MetadataWriter, WriteMode};

fn cols() -> Vec<ColumnDef> {
    vec![
        ColumnDef::new("id", "int64", false).unwrap(),
        ColumnDef::new("name", "varchar", true).unwrap(),
    ]
}

/// A writable single-catalog SQLite environment plus the bits needed to assert on it.
struct Harness {
    writer: SqliteMetadataWriter,
    conn_str: String,
    data_path: PathBuf,
    _temp: TempDir,
}

async fn setup() -> Harness {
    let temp = TempDir::new().unwrap();
    let db_path = temp.path().join("test.db");
    let data_path = temp.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();
    let conn_str = format!("sqlite:{}?mode=rwc", db_path.display());

    let writer = SqliteMetadataWriter::new_with_init(&conn_str)
        .await
        .unwrap();
    writer.set_data_path(data_path.to_str().unwrap()).unwrap();

    Harness {
        writer,
        conn_str,
        data_path,
        _temp: temp,
    }
}

/// Open a fresh read connection to the metadata DB for raw assertions.
async fn pool(h: &Harness) -> SqlitePool {
    SqlitePool::connect(&h.conn_str).await.unwrap()
}

async fn scalar_i64(pool: &SqlitePool, sql: &str) -> i64 {
    sqlx::query(sql)
        .fetch_one(pool)
        .await
        .unwrap()
        .try_get::<i64, _>(0)
        .unwrap()
}

/// Register a data file in the catalog AND create the matching physical file on disk
/// at `<data_path>/<schema>/<table>/<name>`, mirroring the real write layout.
fn write_physical_file(data_path: &Path, schema: &str, table: &str, name: &str) {
    let dir = data_path.join(schema).join(table);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join(name), b"parquet-bytes").unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn drop_table_tombstones_children_and_is_idempotent() {
    let h = setup().await;
    let s = h
        .writer
        .begin_write_transaction("main", "users", &cols(), WriteMode::Replace)
        .unwrap();
    h.writer
        .register_data_file(
            s.table_id,
            s.snapshot_id,
            &DataFileInfo::new("f1.parquet", 100, 5),
        )
        .unwrap();

    let dropped = h.writer.drop_table("main", "users").unwrap();
    assert!(dropped, "live table should drop");

    let p = pool(&h).await;
    // No live rows remain for the table; children carry the drop snapshot.
    for tbl in ["ducklake_table", "ducklake_column", "ducklake_data_file"] {
        let live = scalar_i64(
            &p,
            &format!(
                "SELECT COUNT(*) FROM {tbl} WHERE table_id = {} AND end_snapshot IS NULL",
                s.table_id
            ),
        )
        .await;
        assert_eq!(live, 0, "no live rows in {tbl} after drop");
    }

    // The stats row is intentionally preserved (monotonic next_row_id).
    let stats = scalar_i64(
        &p,
        &format!(
            "SELECT COUNT(*) FROM ducklake_table_stats WHERE table_id = {}",
            s.table_id
        ),
    )
    .await;
    assert_eq!(stats, 1, "table_stats row preserved across drop");
    let next_row_id = scalar_i64(
        &p,
        &format!(
            "SELECT next_row_id FROM ducklake_table_stats WHERE table_id = {}",
            s.table_id
        ),
    )
    .await;
    assert_eq!(next_row_id, 5, "next_row_id preserved");

    // Second drop is a no-op.
    let dropped_again = h.writer.drop_table("main", "users").unwrap();
    assert!(!dropped_again, "second drop returns false");
}

#[tokio::test(flavor = "multi_thread")]
async fn drop_table_unknown_and_empty_names() {
    let h = setup().await;
    assert!(
        !h.writer.drop_table("main", "ghost").unwrap(),
        "unknown table -> false"
    );
    assert!(
        h.writer.drop_table("", "users").is_err(),
        "empty schema rejected"
    );
    assert!(
        h.writer.drop_table("main", "").is_err(),
        "empty table rejected"
    );
}

/// Write three Replace generations of one table so the first two data files become
/// superseded (end-snapshotted), then return the writer + snapshot ids.
fn three_generations(writer: &SqliteMetadataWriter) -> (i64, i64, i64, i64) {
    let s1 = writer
        .begin_write_transaction("main", "t", &cols(), WriteMode::Replace)
        .unwrap();
    writer
        .register_data_file(
            s1.table_id,
            s1.snapshot_id,
            &DataFileInfo::new("f1.parquet", 100, 5),
        )
        .unwrap();
    let s2 = writer
        .begin_write_transaction("main", "t", &cols(), WriteMode::Replace)
        .unwrap();
    writer
        .register_data_file(
            s2.table_id,
            s2.snapshot_id,
            &DataFileInfo::new("f2.parquet", 100, 5),
        )
        .unwrap();
    let s3 = writer
        .begin_write_transaction("main", "t", &cols(), WriteMode::Replace)
        .unwrap();
    writer
        .register_data_file(
            s3.table_id,
            s3.snapshot_id,
            &DataFileInfo::new("f3.parquet", 100, 5),
        )
        .unwrap();
    (s1.table_id, s1.snapshot_id, s2.snapshot_id, s3.snapshot_id)
}

#[tokio::test(flavor = "multi_thread")]
async fn expire_by_version_schedules_orphaned_file() {
    let h = setup().await;
    let (_tid, s1, _s2, _s3) = three_generations(&h.writer);

    // f1 lives in [s1, s2); expiring s1 leaves no surviving snapshot in that range.
    let expired = h
        .writer
        .expire_snapshots(ExpireCriteria::Versions(vec![s1]))
        .unwrap();
    assert_eq!(expired.len(), 1);
    assert_eq!(expired[0].snapshot_id, s1);

    let p = pool(&h).await;
    assert_eq!(
        scalar_i64(
            &p,
            &format!("SELECT COUNT(*) FROM ducklake_snapshot WHERE snapshot_id = {s1}")
        )
        .await,
        0,
        "expired snapshot row removed"
    );
    // Exactly f1 is scheduled, with the data_path-relative resolved path.
    let scheduled: Vec<(String, i64)> =
        sqlx::query("SELECT path, path_is_relative FROM ducklake_files_scheduled_for_deletion")
            .fetch_all(&p)
            .await
            .unwrap()
            .into_iter()
            .map(|r| {
                (
                    r.try_get::<String, _>(0).unwrap(),
                    r.try_get::<i64, _>(1).unwrap(),
                )
            })
            .collect();
    assert_eq!(scheduled.len(), 1);
    assert_eq!(scheduled[0].0, "main/t/f1.parquet");
    assert_eq!(scheduled[0].1, 1, "path is relative to data_path");
    // f1's catalog row is gone; f2/f3 remain.
    assert_eq!(
        scalar_i64(&p, "SELECT COUNT(*) FROM ducklake_data_file").await,
        2
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn expire_full_after_drop_reclaims_all_table_metadata() {
    let h = setup().await;
    let s = h
        .writer
        .begin_write_transaction("main", "t", &cols(), WriteMode::Replace)
        .unwrap();
    h.writer
        .register_data_file(
            s.table_id,
            s.snapshot_id,
            &DataFileInfo::new("f1.parquet", 100, 5),
        )
        .unwrap();
    // Drop allocates snapshot 2; the table is now fully tombstoned in [1, 2).
    assert!(h.writer.drop_table("main", "t").unwrap());

    // Expire snapshot 1 (the drop snapshot 2 is the most recent and is kept).
    let expired = h
        .writer
        .expire_snapshots(ExpireCriteria::Versions(vec![s.snapshot_id]))
        .unwrap();
    assert_eq!(expired.len(), 1);

    let p = pool(&h).await;
    let tid = s.table_id;
    for tbl in ["ducklake_table", "ducklake_column", "ducklake_data_file", "ducklake_table_stats"] {
        let cnt = scalar_i64(
            &p,
            &format!("SELECT COUNT(*) FROM {tbl} WHERE table_id = {tid}"),
        )
        .await;
        assert_eq!(cnt, 0, "{tbl} fully reclaimed after expire");
    }
    // The orphaned file was scheduled for physical deletion.
    assert_eq!(
        scalar_i64(
            &p,
            "SELECT COUNT(*) FROM ducklake_files_scheduled_for_deletion"
        )
        .await,
        1
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn cleanup_old_files_deletes_physical_file() {
    let h = setup().await;
    let (_tid, s1, _s2, _s3) = three_generations(&h.writer);
    // Materialize the physical files referenced by the catalog.
    for name in ["f1.parquet", "f2.parquet", "f3.parquet"] {
        write_physical_file(&h.data_path, "main", "t", name);
    }

    h.writer
        .expire_snapshots(ExpireCriteria::Versions(vec![s1]))
        .unwrap();

    let store: Arc<dyn ObjectStore> = Arc::new(LocalFileSystem::new());
    let f1 = h.data_path.join("main").join("t").join("f1.parquet");

    // Dry run reports the path without deleting.
    let dry = cleanup_old_files_sqlite(&h.writer, store.clone(), CleanupCriteria::All, true)
        .await
        .unwrap();
    assert_eq!(dry.len(), 1);
    assert!(f1.exists(), "dry run must not delete");

    // Real run deletes the object and clears the bookkeeping row.
    let done = cleanup_old_files_sqlite(&h.writer, store.clone(), CleanupCriteria::All, false)
        .await
        .unwrap();
    assert_eq!(done.len(), 1);
    assert!(!f1.exists(), "f1 physically removed");
    assert!(
        h.data_path
            .join("main")
            .join("t")
            .join("f2.parquet")
            .exists(),
        "live file f2 untouched"
    );

    let p = pool(&h).await;
    assert_eq!(
        scalar_i64(
            &p,
            "SELECT COUNT(*) FROM ducklake_files_scheduled_for_deletion"
        )
        .await,
        0,
        "scheduled rows cleared"
    );

    // Idempotent: nothing left to clean.
    let again = cleanup_old_files_sqlite(&h.writer, store, CleanupCriteria::All, false)
        .await
        .unwrap();
    assert!(again.is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn expire_older_than_uses_datetime_utc() {
    // Cover the OlderThan public API (chrono::DateTime<Utc>) end-to-end: a
    // far-future cutoff means every non-most-recent snapshot is expirable.
    let h = setup().await;
    let (_tid, _s1, _s2, _s3) = three_generations(&h.writer);

    let cutoff = chrono::Utc::now() + chrono::Duration::days(1);
    let expired = h
        .writer
        .expire_snapshots(ExpireCriteria::OlderThan(cutoff))
        .unwrap();
    // three_generations creates s1, s2, s3. s3 is most-recent and kept;
    // the other two are expired.
    assert_eq!(expired.len(), 2);
}

#[tokio::test(flavor = "multi_thread")]
async fn expire_and_cleanup_no_op_paths() {
    let h = setup().await;
    // Single snapshot: nothing is expirable (the most recent is always kept).
    let s = h
        .writer
        .begin_write_transaction("main", "t", &cols(), WriteMode::Replace)
        .unwrap();
    let expired = h
        .writer
        .expire_snapshots(ExpireCriteria::Versions(vec![s.snapshot_id]))
        .unwrap();
    assert!(
        expired.is_empty(),
        "cannot expire the only/most-recent snapshot"
    );

    // Nothing scheduled -> cleanup returns empty.
    let store: Arc<dyn ObjectStore> = Arc::new(LocalFileSystem::new());
    let cleaned = cleanup_old_files_sqlite(&h.writer, store, CleanupCriteria::All, false)
        .await
        .unwrap();
    assert!(cleaned.is_empty());
}

// ---------------------------------------------------------------------------
// delete_orphaned_files
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn delete_orphaned_files_removes_unreferenced_keeps_referenced() {
    let h = setup().await;
    // One registered, referenced file.
    let s = h
        .writer
        .begin_write_transaction("main", "t", &cols(), WriteMode::Replace)
        .unwrap();
    h.writer
        .register_data_file(
            s.table_id,
            s.snapshot_id,
            &DataFileInfo::new("referenced.parquet", 100, 5),
        )
        .unwrap();
    write_physical_file(&h.data_path, "main", "t", "referenced.parquet");

    // Drop a stray file alongside it that's NOT in the catalog.
    write_physical_file(&h.data_path, "main", "t", "orphan.parquet");
    // And a non-parquet file that must be ignored (only .parquet is swept).
    std::fs::write(
        h.data_path.join("main").join("t").join("README.txt"),
        b"keep me",
    )
    .unwrap();

    let store: Arc<dyn ObjectStore> = Arc::new(LocalFileSystem::new());
    let orphan_path = h.data_path.join("main").join("t").join("orphan.parquet");
    let ref_path = h
        .data_path
        .join("main")
        .join("t")
        .join("referenced.parquet");
    let readme = h.data_path.join("main").join("t").join("README.txt");

    // Dry run reports the orphan without touching disk.
    let dry = delete_orphaned_files_sqlite(&h.writer, store.clone(), CleanupCriteria::All, true)
        .await
        .unwrap();
    assert_eq!(dry.len(), 1);
    assert!(
        dry[0].ends_with("main/t/orphan.parquet"),
        "got {:?}",
        dry[0]
    );
    assert!(orphan_path.exists());

    // Real run deletes only the orphan.
    let done = delete_orphaned_files_sqlite(&h.writer, store, CleanupCriteria::All, false)
        .await
        .unwrap();
    assert_eq!(done.len(), 1);
    assert!(!orphan_path.exists(), "orphan should be gone");
    assert!(ref_path.exists(), "referenced file must survive");
    assert!(readme.exists(), "non-parquet file must be ignored");
}

#[tokio::test(flavor = "multi_thread")]
async fn delete_orphaned_files_older_than_skips_recent_files() {
    // Critical safety check: a file just written (in-flight) is newer than the
    // cutoff and must be kept even though it's unreferenced. Matches the
    // upstream `last_modified < older_than` guard.
    let h = setup().await;
    write_physical_file(&h.data_path, "main", "t", "fresh_orphan.parquet");
    let fresh = h
        .data_path
        .join("main")
        .join("t")
        .join("fresh_orphan.parquet");
    assert!(fresh.exists());

    let store: Arc<dyn ObjectStore> = Arc::new(LocalFileSystem::new());
    let cutoff = chrono::Utc::now() - chrono::Duration::hours(1);
    let deleted =
        delete_orphaned_files_sqlite(&h.writer, store, CleanupCriteria::OlderThan(cutoff), false)
            .await
            .unwrap();
    assert!(
        deleted.is_empty(),
        "files newer than cutoff must not be deleted (got {deleted:?})"
    );
    assert!(fresh.exists(), "fresh orphan survives older_than filter");
}

#[tokio::test(flavor = "multi_thread")]
async fn delete_orphaned_files_spares_files_pending_in_scheduled_table() {
    // Files in `ducklake_files_scheduled_for_deletion` are "queued for
    // cleanup_old_files" — they may still exist on disk. delete_orphaned_files
    // must treat them as referenced so it doesn't race ahead and delete them
    // (cleanup_old_files would then double-delete and remove the bookkeeping).
    let h = setup().await;
    write_physical_file(&h.data_path, "main", "t", "scheduled.parquet");

    // Seed the scheduled-for-deletion table directly. This matches what
    // `expire_snapshots` would have done — but we shortcut it for the test.
    let p = pool(&h).await;
    sqlx::query(
        "INSERT INTO ducklake_files_scheduled_for_deletion
             (data_file_id, path, path_is_relative, schedule_start)
         VALUES (?, ?, 1, CURRENT_TIMESTAMP)",
    )
    .bind(42_i64)
    .bind("main/t/scheduled.parquet")
    .execute(&p)
    .await
    .unwrap();

    let store: Arc<dyn ObjectStore> = Arc::new(LocalFileSystem::new());
    let deleted = delete_orphaned_files_sqlite(&h.writer, store, CleanupCriteria::All, false)
        .await
        .unwrap();
    assert!(
        deleted.is_empty(),
        "files in scheduled-for-deletion must be considered referenced (got {deleted:?})"
    );
    assert!(
        h.data_path
            .join("main")
            .join("t")
            .join("scheduled.parquet")
            .exists()
    );
}
