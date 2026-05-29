//! Maintenance operations for DuckLake catalogs: snapshot expiration, physical
//! file cleanup, and orphaned-file reclamation.
//!
//! These port the three official DuckLake maintenance commands:
//!
//! 1. **expire** ([`crate::metadata_writer_sqlite::SqliteMetadataWriter::expire_snapshots`],
//!    [`crate::multicatalog::MulticatalogManager::expire_snapshots_in_catalog`]) deletes the
//!    chosen snapshots, garbage-collects every table / data file / delete file that is no
//!    longer reachable by any surviving snapshot, and records the orphaned physical paths in
//!    `ducklake_files_scheduled_for_deletion`. No object storage is touched.
//! 2. **cleanup_old_files** ([`cleanup_old_files_sqlite`], [`cleanup_old_files_in_catalog`])
//!    reads the scheduled rows, deletes the objects from the object store, and removes the rows.
//! 3. **delete_orphaned_files** ([`delete_orphaned_files_sqlite`],
//!    [`delete_orphaned_files_multicatalog`]) lists the data path, subtracts everything
//!    referenced by the catalog (data files + delete files + still-scheduled-for-deletion),
//!    and deletes whatever's left. Catches files left by aborted writes and by
//!    `drop_catalog` (which hard-deletes catalog metadata without scheduling its files).
//!
//! The metadata writers deliberately hold no object store — physical I/O lives here so the
//! catalog layer stays storage-agnostic (the object store comes from the caller, e.g. the
//! same one a [`crate::table_writer::DuckLakeTableWriter`] was built with).

use crate::Result;
use crate::path_resolver::{parse_object_store_url, resolve_path};
use chrono::{DateTime, Utc};
use futures::TryStreamExt;
use object_store::path::Path as ObjectPath;
use object_store::{ObjectStore, ObjectStoreExt};
use std::collections::HashSet;
use std::sync::Arc;

/// Which snapshots to expire.
#[derive(Debug, Clone)]
pub enum ExpireCriteria {
    /// Expire exactly these snapshot ids (the most recent snapshot is always kept,
    /// even if listed here).
    Versions(Vec<i64>),
    /// Expire every snapshot older than this timestamp. The most recent snapshot
    /// is always kept regardless.
    OlderThan(DateTime<Utc>),
}

/// Which scheduled files to physically delete.
#[derive(Debug, Clone)]
pub enum CleanupCriteria {
    /// Delete every scheduled file regardless of when it was scheduled.
    All,
    /// Delete only files scheduled before this timestamp.
    OlderThan(DateTime<Utc>),
}

/// Render a UTC timestamp as a SQL literal both backends parse and compare correctly.
///
/// SQLite stores `CURRENT_TIMESTAMP` as `'YYYY-MM-DD HH:MM:SS'` text — lexicographic
/// comparison with this format works because the components are zero-padded and in
/// big-endian order. Postgres parses the same text into both `TIMESTAMP` and
/// `TIMESTAMPTZ` (we explicitly cast at the bind site).
pub(crate) fn format_sql_timestamp(dt: &DateTime<Utc>) -> String {
    dt.format("%Y-%m-%d %H:%M:%S%.6f").to_string()
}

/// A snapshot that was expired, as returned by the expire operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpiredSnapshot {
    /// The expired snapshot id.
    pub snapshot_id: i64,
    /// The snapshot timestamp, as stored in `ducklake_snapshot.snapshot_time`.
    pub snapshot_time: String,
}

/// A row of `ducklake_files_scheduled_for_deletion`. `path` is relative to the
/// catalog `data_path` root when `path_is_relative` is set (see the table docs).
#[derive(Debug, Clone)]
pub struct ScheduledFile {
    /// The `data_file_id` of the (already-deleted) data/delete file row.
    pub data_file_id: i64,
    /// Physical path, relative to `data_path` when `path_is_relative`.
    pub path: String,
    /// Whether `path` is relative to the catalog `data_path` root.
    pub path_is_relative: bool,
}

/// Resolve scheduled rows against `data_path`, delete the objects (unless `dry_run`),
/// and return the resolved absolute paths that were (or would be) deleted. Shared
/// by both backends; the row listing / row removal is backend-specific and passed in.
async fn run_cleanup<RemoveFut>(
    data_path: &str,
    files: Vec<ScheduledFile>,
    object_store: Arc<dyn ObjectStore>,
    dry_run: bool,
    remove_rows: impl FnOnce(Vec<i64>) -> RemoveFut,
) -> Result<Vec<String>>
where
    RemoveFut: std::future::Future<Output = Result<()>>,
{
    if files.is_empty() {
        return Ok(Vec::new());
    }
    let (_, base_key) = parse_object_store_url(data_path)?;

    let mut resolved = Vec::with_capacity(files.len());
    let mut ids = Vec::with_capacity(files.len());
    for file in &files {
        let abs = resolve_path(&base_key, &file.path, file.path_is_relative)?;
        resolved.push(abs);
        ids.push(file.data_file_id);
    }

    if dry_run {
        return Ok(resolved);
    }

    for abs in &resolved {
        // object_store keys are relative (no leading slash) — same transform the
        // writer uses when it puts a file (see table_writer.rs).
        let key = ObjectPath::from(abs.trim_start_matches('/'));
        match object_store.delete(&key).await {
            Ok(()) => {},
            // A missing object means a prior partial cleanup already removed it —
            // idempotent, so we still drop the scheduled row.
            Err(object_store::Error::NotFound {
                ..
            }) => {},
            Err(e) => return Err(e.into()),
        }
    }

    remove_rows(ids).await?;
    Ok(resolved)
}

/// Physically delete files scheduled by [`SqliteMetadataWriter::expire_snapshots`] and
/// remove their bookkeeping rows. Returns the resolved absolute paths deleted (or, for
/// `dry_run`, the paths that would be deleted).
///
/// [`SqliteMetadataWriter::expire_snapshots`]: crate::metadata_writer_sqlite::SqliteMetadataWriter::expire_snapshots
#[cfg(feature = "write-sqlite")]
pub async fn cleanup_old_files_sqlite(
    writer: &crate::metadata_writer_sqlite::SqliteMetadataWriter,
    object_store: Arc<dyn ObjectStore>,
    criteria: CleanupCriteria,
    dry_run: bool,
) -> Result<Vec<String>> {
    let data_path = crate::metadata_writer::MetadataWriter::get_data_path(writer)?;
    let files = writer.list_scheduled_for_deletion(&criteria)?;
    run_cleanup(&data_path, files, object_store, dry_run, |ids| async move {
        writer.remove_scheduled(&ids)
    })
    .await
}

/// Physically delete files scheduled by
/// [`MulticatalogManager::expire_snapshots_in_catalog`] for `catalog_name` and remove their
/// bookkeeping rows. Returns the resolved absolute paths deleted (or, for `dry_run`, the
/// paths that would be deleted).
///
/// [`MulticatalogManager::expire_snapshots_in_catalog`]: crate::multicatalog::MulticatalogManager::expire_snapshots_in_catalog
#[cfg(feature = "write-postgres")]
pub async fn cleanup_old_files_in_catalog(
    mgr: &crate::multicatalog::MulticatalogManager,
    catalog_name: &str,
    object_store: Arc<dyn ObjectStore>,
    criteria: CleanupCriteria,
    dry_run: bool,
) -> Result<Vec<String>> {
    let data_path = mgr.get_data_path().await?;
    let files = mgr
        .list_scheduled_for_deletion_in_catalog(catalog_name, &criteria)
        .await?;
    run_cleanup(&data_path, files, object_store, dry_run, |ids| async move {
        mgr.remove_scheduled_in_catalog(catalog_name, &ids).await
    })
    .await
}

/// List the object store under `data_path`, subtract everything referenced by the
/// catalog (passed in as `(path, path_is_relative)` pairs), filter by `.parquet`
/// suffix + `last_modified < older_than` (when set), and delete the leftovers.
///
/// Shared between SQLite and multicatalog Postgres — the only backend-specific
/// piece is producing `referenced`, which is passed in.
async fn run_orphan_cleanup(
    data_path: &str,
    referenced: Vec<(String, bool)>,
    object_store: Arc<dyn ObjectStore>,
    criteria: CleanupCriteria,
    dry_run: bool,
) -> Result<Vec<String>> {
    let (_, base_key) = parse_object_store_url(data_path)?;

    // Build the set of referenced object_store keys, normalised the same way
    // the listing produces them (no leading slash, via ObjectPath canon).
    let mut referenced_set: HashSet<ObjectPath> = HashSet::with_capacity(referenced.len());
    for (path, rel) in referenced {
        let abs = resolve_path(&base_key, &path, rel)?;
        referenced_set.insert(ObjectPath::from(abs.trim_start_matches('/')));
    }

    // List every file under the data path. The prefix is the data_path's key
    // part (same transform we use everywhere else).
    let prefix = ObjectPath::from(base_key.trim_start_matches('/'));
    let entries: Vec<object_store::ObjectMeta> =
        object_store.list(Some(&prefix)).try_collect().await?;

    // Apply the official filters: only `.parquet`, and only files whose
    // `last_modified < older_than` when a cutoff was given. Skipping in-flight
    // writes via the timestamp filter is what makes this safe to schedule.
    let mut orphans: Vec<ObjectPath> = Vec::new();
    for meta in entries {
        if !meta.location.as_ref().ends_with(".parquet") {
            continue;
        }
        if let CleanupCriteria::OlderThan(cutoff) = &criteria
            && meta.last_modified >= *cutoff
        {
            continue;
        }
        if !referenced_set.contains(&meta.location) {
            orphans.push(meta.location);
        }
    }

    // Return absolute-style paths (leading `/`) to match `cleanup_old_files`'s
    // return shape and the official `ducklake_delete_orphaned_files` output.
    // ObjectPath strips the leading slash for object-store canonical form, so
    // we add it back at the API boundary. The OS-level delete still uses the
    // ObjectPath directly.
    if dry_run {
        return Ok(orphans.into_iter().map(|p| format!("/{p}")).collect());
    }

    let mut deleted = Vec::with_capacity(orphans.len());
    for orphan in orphans {
        match object_store.delete(&orphan).await {
            Ok(()) => {},
            // Already gone (another vacuumer, or out-of-band deletion) — fine.
            Err(object_store::Error::NotFound {
                ..
            }) => {},
            Err(e) => return Err(e.into()),
        }
        deleted.push(format!("/{orphan}"));
    }
    Ok(deleted)
}

/// List the catalog's `data_path` and delete every `.parquet` not referenced by
/// the metadata (data files, delete files, or still-scheduled-for-deletion rows).
///
/// Returns the absolute paths deleted, or — for `dry_run` — the paths that would
/// be deleted. Matches the official `ducklake_delete_orphaned_files` semantics:
/// the `OlderThan` filter compares against the file's `last_modified` so files
/// being written by in-flight transactions are skipped. `CleanupCriteria::All`
/// is allowed (matching the official `cleanup_all => true`) but should be used
/// only when the catalog is known to be idle.
#[cfg(feature = "write-sqlite")]
pub async fn delete_orphaned_files_sqlite(
    writer: &crate::metadata_writer_sqlite::SqliteMetadataWriter,
    object_store: Arc<dyn ObjectStore>,
    criteria: CleanupCriteria,
    dry_run: bool,
) -> Result<Vec<String>> {
    let data_path = crate::metadata_writer::MetadataWriter::get_data_path(writer)?;
    let referenced = writer.list_referenced_paths()?;
    run_orphan_cleanup(&data_path, referenced, object_store, criteria, dry_run).await
}

/// List the multicatalog's shared `data_path` and delete every `.parquet` not
/// referenced by any catalog in the metadata DB.
///
/// **Global, not per-catalog.** In our multicatalog Postgres model every catalog
/// shares one `data_path`, so the natural unit for orphan reclamation is the
/// whole data path. The referenced set is built across every catalog's data
/// files, delete files, and still-pending scheduled-for-deletion rows. This
/// closes the file-leak left by [`MulticatalogManager::drop_catalog`] — which
/// hard-deletes catalog metadata in one shot, so its files become unreferenced
/// rather than scheduled.
///
/// [`MulticatalogManager::drop_catalog`]: crate::multicatalog::MulticatalogManager::drop_catalog
#[cfg(feature = "write-postgres")]
pub async fn delete_orphaned_files_multicatalog(
    mgr: &crate::multicatalog::MulticatalogManager,
    object_store: Arc<dyn ObjectStore>,
    criteria: CleanupCriteria,
    dry_run: bool,
) -> Result<Vec<String>> {
    let data_path = mgr.get_data_path().await?;
    let referenced = mgr.list_referenced_paths().await?;
    run_orphan_cleanup(&data_path, referenced, object_store, criteria, dry_run).await
}
