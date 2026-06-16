//! Integration tests for single-catalog maintenance on the SQLite writer:
//! `drop_table`, `expire_snapshots`, and `cleanup_old_files_sqlite`.
//!
//! Mirrors the upstream DuckLake suite
//! (`test/sql/catalog/drop_table.test`, `test/sql/compaction/expire_snapshots*.test`).

#![cfg(all(feature = "write-sqlite", feature = "metadata-sqlite"))]

use std::collections::HashSet;
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
            WriteMode::Replace,
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
            WriteMode::Replace,
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
            WriteMode::Replace,
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
            WriteMode::Replace,
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
            WriteMode::Replace,
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
            WriteMode::Replace,
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

// ---------------------------------------------------------------------------
// delete_orphaned_files — edge-case contract coverage
//
// The orphan sweep relies on three implicit invariants that the happy-path
// tests above don't exercise:
//
//   * `object_store::Path` normalises consecutive slashes, so a relative path
//     starting with `/` (which the `RESOLVED_PATH` SQL CASE can emit when
//     `s.path = ''`) still matches the listing for the same physical file.
//   * `resolve_path` short-circuits when `path_is_relative = false`, so a
//     data-file row whose `path` is absolute is keyed by that absolute path
//     verbatim — and must still match `object_store.list`'s key for the same
//     file when the file happens to live under `data_path`.
//   * `object_store.list(Some(&prefix))` recurses into subdirectories — a
//     hive-partitioned layout (`year=…/month=…/file.parquet`) must be fully
//     visible.
//
// Plus two operational corners worth pinning down:
//
//   * `dry_run` and the real run return identical path sets — they take
//     separate code paths inside `run_orphan_cleanup`.
//   * `CleanupCriteria::All` against an empty catalog deletes every
//     `.parquet` in `data_path`. Documented behaviour (mirrors the official
//     `cleanup_all => true`), made explicit here so anyone changing the
//     safety story sees the footgun directly.
// ---------------------------------------------------------------------------

/// Schema rows with `path = ''` (the DDL default) make the `RESOLVED_PATH`
/// SQL emit a leading-slash relative path (`'' || '/' || t || '/' || f`).
/// After `join_paths(base_key, "/t/f")` we get a path with a double slash.
/// `ObjectPath::from(...)` normalises that to the single-slash form, which
/// matches what `object_store.list` returns for the same file. This test
/// locks the contract in case ObjectPath's normalisation ever weakens.
#[tokio::test(flavor = "multi_thread")]
async fn delete_orphaned_files_handles_empty_schema_path() {
    let h = setup().await;
    let p = pool(&h).await;

    // Bypass `begin_write_transaction` — it always passes `schema_name` as
    // `path`, so we'd never naturally produce an empty `s.path`. Insert
    // the catalog skeleton directly to exercise the DDL default.
    sqlx::query("INSERT INTO ducklake_snapshot (snapshot_time) VALUES (CURRENT_TIMESTAMP)")
        .execute(&p)
        .await
        .unwrap();
    sqlx::query(
        "INSERT INTO ducklake_schema (schema_name, path, path_is_relative, begin_snapshot)
         VALUES ('s', '', 1, 1)",
    )
    .execute(&p)
    .await
    .unwrap();
    sqlx::query(
        "INSERT INTO ducklake_table (schema_id, table_name, path, path_is_relative, begin_snapshot)
         VALUES (1, 't', 't', 1, 1)",
    )
    .execute(&p)
    .await
    .unwrap();
    sqlx::query(
        "INSERT INTO ducklake_data_file (table_id, path, path_is_relative, file_size_bytes, record_count, begin_snapshot)
         VALUES (1, 'f.parquet', 1, 100, 5, 1)",
    )
    .execute(&p)
    .await
    .unwrap();

    // With s.path='' the resolved path collapses to `<data_path>/t/f.parquet`
    // (the empty schema contributes nothing). The physical file lives there.
    let dir = h.data_path.join("t");
    std::fs::create_dir_all(&dir).unwrap();
    let referenced = dir.join("f.parquet");
    std::fs::write(&referenced, b"data").unwrap();

    let store: Arc<dyn ObjectStore> = Arc::new(LocalFileSystem::new());
    let deleted = delete_orphaned_files_sqlite(&h.writer, store, CleanupCriteria::All, false)
        .await
        .unwrap();
    assert!(
        deleted.is_empty(),
        "referenced file under empty-schema-path resolution must survive (got {deleted:?})"
    );
    assert!(referenced.exists(), "referenced file must still exist");
}

/// `path_is_relative = false` makes `RESOLVED_PATH` return `df.path` verbatim
/// (no prepending). The orphan-cleanup resolver short-circuits in
/// `resolve_path` and uses the absolute key as-is. A file at that absolute
/// key — even if it happens to live under `data_path` — must be excluded
/// from deletion. Our writer doesn't emit absolute paths, so the SQL CASE
/// branch is otherwise untested by integration tests.
#[tokio::test(flavor = "multi_thread")]
async fn delete_orphaned_files_handles_absolute_file_paths() {
    let h = setup().await;
    let p = pool(&h).await;
    let s = h
        .writer
        .begin_write_transaction("main", "t", &cols(), WriteMode::Replace)
        .unwrap();

    // Materialise the file at an absolute path that happens to fall under
    // data_path — that's the configuration where the SQL CASE's absolute
    // branch interacts with the listing.
    write_physical_file(&h.data_path, "main", "t", "abs.parquet");
    let abs_file = h.data_path.join("main").join("t").join("abs.parquet");
    let abs_str = abs_file.to_str().unwrap().to_string();

    // Register it with path_is_relative = false (= 0 in SQLite).
    sqlx::query(
        "INSERT INTO ducklake_data_file (table_id, path, path_is_relative, file_size_bytes, record_count, begin_snapshot)
         VALUES (?, ?, 0, 100, 5, ?)",
    )
    .bind(s.table_id)
    .bind(&abs_str)
    .bind(s.snapshot_id)
    .execute(&p)
    .await
    .unwrap();

    // Drop a stray IN data_path so we know the sweep IS running.
    write_physical_file(&h.data_path, "main", "t", "stray.parquet");
    let stray = h.data_path.join("main").join("t").join("stray.parquet");

    let store: Arc<dyn ObjectStore> = Arc::new(LocalFileSystem::new());
    let deleted = delete_orphaned_files_sqlite(&h.writer, store, CleanupCriteria::All, false)
        .await
        .unwrap();
    assert_eq!(
        deleted.len(),
        1,
        "stray is the only orphan (got {deleted:?})"
    );
    assert!(
        deleted[0].ends_with("main/t/stray.parquet"),
        "got {:?}",
        deleted[0]
    );
    assert!(!stray.exists(), "stray deleted");
    assert!(abs_file.exists(), "absolute-path registered file survives");
}

/// `object_store.list(Some(&prefix))` must recurse into subdirectories so
/// partition-style layouts (`year=…/month=…/file.parquet`) are reachable.
/// Our writer doesn't produce nested layouts today, but other tools might —
/// and a future change to the listing call (e.g. switching to a delimiter
/// to optimise paginating) could silently stop recursing.
#[tokio::test(flavor = "multi_thread")]
async fn delete_orphaned_files_recurses_into_nested_directories() {
    let h = setup().await;
    let s = h
        .writer
        .begin_write_transaction("main", "t", &cols(), WriteMode::Replace)
        .unwrap();
    h.writer
        .register_data_file(
            s.table_id,
            s.snapshot_id,
            &DataFileInfo::new("ref.parquet", 100, 5),
            WriteMode::Replace,
        )
        .unwrap();
    write_physical_file(&h.data_path, "main", "t", "ref.parquet");

    // Two orphan files at progressively deeper nesting.
    let deep_dir = h
        .data_path
        .join("main")
        .join("t")
        .join("year=2024")
        .join("month=01");
    std::fs::create_dir_all(&deep_dir).unwrap();
    let level2 = deep_dir.join("partition.parquet");
    std::fs::write(&level2, b"orphan-2").unwrap();
    let deeper_dir = deep_dir.join("day=15");
    std::fs::create_dir_all(&deeper_dir).unwrap();
    let level3 = deeper_dir.join("nested.parquet");
    std::fs::write(&level3, b"orphan-3").unwrap();

    let store: Arc<dyn ObjectStore> = Arc::new(LocalFileSystem::new());
    let deleted: HashSet<String> =
        delete_orphaned_files_sqlite(&h.writer, store, CleanupCriteria::All, false)
            .await
            .unwrap()
            .into_iter()
            .collect();
    assert_eq!(
        deleted.len(),
        2,
        "both nested orphans must be discovered (got {deleted:?})"
    );
    assert!(!level2.exists());
    assert!(!level3.exists());
    assert!(
        h.data_path
            .join("main")
            .join("t")
            .join("ref.parquet")
            .exists(),
        "referenced survives"
    );
}

/// dry_run and real-run must return identical path strings (callers might
/// compare them or feed dry_run into a confirmation prompt then expect the
/// same paths to be deleted on confirm). They take separate code paths
/// (dry_run formats from the pre-delete `orphans` Vec; real-run formats
/// per-orphan after `object_store.delete` succeeds), so a future refactor
/// could diverge them.
#[tokio::test(flavor = "multi_thread")]
async fn delete_orphaned_files_dry_run_matches_real_run() {
    let h = setup().await;
    let s = h
        .writer
        .begin_write_transaction("main", "t", &cols(), WriteMode::Replace)
        .unwrap();
    h.writer
        .register_data_file(
            s.table_id,
            s.snapshot_id,
            &DataFileInfo::new("ref.parquet", 100, 5),
            WriteMode::Replace,
        )
        .unwrap();
    write_physical_file(&h.data_path, "main", "t", "ref.parquet");
    for name in ["o1.parquet", "o2.parquet", "o3.parquet"] {
        write_physical_file(&h.data_path, "main", "t", name);
    }

    let store: Arc<dyn ObjectStore> = Arc::new(LocalFileSystem::new());
    let dry: HashSet<String> =
        delete_orphaned_files_sqlite(&h.writer, store.clone(), CleanupCriteria::All, true)
            .await
            .unwrap()
            .into_iter()
            .collect();
    assert_eq!(dry.len(), 3);
    // Dry-run must not touch disk.
    for name in ["o1.parquet", "o2.parquet", "o3.parquet"] {
        assert!(h.data_path.join("main").join("t").join(name).exists());
    }

    let real: HashSet<String> =
        delete_orphaned_files_sqlite(&h.writer, store, CleanupCriteria::All, false)
            .await
            .unwrap()
            .into_iter()
            .collect();
    assert_eq!(
        dry, real,
        "dry_run and real_run must return identical path sets"
    );
}

/// `CleanupCriteria::All` against an empty catalog deletes every `.parquet`
/// under `data_path`. This mirrors the official `cleanup_all => true`
/// semantics but is the classic operator-misuse footgun — pointing the API
/// at a data_path that has files but no catalog rows wipes everything.
/// `OlderThan` is the only safeguard at the crate level; callers should
/// enforce it as a policy invariant (e.g. an automated worker should never
/// pass `All`). This test makes the contract explicit so anyone changing
/// the safety story has to update it deliberately.
#[tokio::test(flavor = "multi_thread")]
async fn delete_orphaned_files_all_on_empty_catalog_wipes_data_path() {
    let h = setup().await;
    // No begin_write_transaction — the catalog is completely empty.
    write_physical_file(&h.data_path, "main", "t", "innocent.parquet");
    write_physical_file(&h.data_path, "main", "t", "innocent2.parquet");
    let innocent1 = h.data_path.join("main").join("t").join("innocent.parquet");
    let innocent2 = h.data_path.join("main").join("t").join("innocent2.parquet");

    let store: Arc<dyn ObjectStore> = Arc::new(LocalFileSystem::new());

    // `All` on an empty catalog → everything in data_path is "unreferenced"
    // → everything is deleted. Documented contract.
    let deleted =
        delete_orphaned_files_sqlite(&h.writer, store.clone(), CleanupCriteria::All, false)
            .await
            .unwrap();
    assert_eq!(
        deleted.len(),
        2,
        "All on empty catalog deletes every .parquet"
    );
    assert!(!innocent1.exists());
    assert!(!innocent2.exists());

    // Restore + verify `OlderThan` is the safeguard against the same scenario.
    write_physical_file(&h.data_path, "main", "t", "fresh.parquet");
    let fresh = h.data_path.join("main").join("t").join("fresh.parquet");
    let cutoff = chrono::Utc::now() - chrono::Duration::hours(1);
    let kept =
        delete_orphaned_files_sqlite(&h.writer, store, CleanupCriteria::OlderThan(cutoff), false)
            .await
            .unwrap();
    assert!(
        kept.is_empty(),
        "OlderThan filter saves a fresh unreferenced file"
    );
    assert!(
        fresh.exists(),
        "fresh file survives OlderThan sweep on empty catalog"
    );
}
