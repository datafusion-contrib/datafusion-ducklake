//! Maintenance operations for DuckLake catalogs: snapshot expiration and
//! physical file cleanup.
//!
//! These port the official DuckLake two-phase vacuum
//! (`ducklake_expire_snapshots` + `ducklake_cleanup_old_files`):
//!
//! 1. **expire** ([`crate::metadata_writer_sqlite::SqliteMetadataWriter::expire_snapshots`],
//!    [`crate::multicatalog::MulticatalogManager::expire_snapshots_in_catalog`]) deletes the
//!    chosen snapshots, garbage-collects every table / data file / delete file that is no
//!    longer reachable by any surviving snapshot, and records the orphaned physical paths in
//!    `ducklake_files_scheduled_for_deletion`. No object storage is touched.
//! 2. **cleanup** ([`cleanup_old_files_sqlite`], [`cleanup_old_files_in_catalog`]) reads the
//!    scheduled rows, deletes the objects from the object store, and removes the rows.
//!
//! The metadata writers deliberately hold no object store — physical I/O lives here so the
//! catalog layer stays storage-agnostic (the object store comes from the caller, e.g. the
//! same one a [`crate::table_writer::DuckLakeTableWriter`] was built with).

use crate::Result;
use crate::path_resolver::{parse_object_store_url, resolve_path};
use chrono::{DateTime, Utc};
use object_store::path::Path as ObjectPath;
use object_store::{ObjectStore, ObjectStoreExt};
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
