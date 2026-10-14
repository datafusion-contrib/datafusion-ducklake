//! Maintenance operations for DuckLake catalogs: snapshot expiration, physical
//! file cleanup, and orphaned-file reclamation.
//!
//! These port the three official DuckLake maintenance commands:
//!
//! 1. **expire** ([`crate::metadata_writer_sqlite::SqliteMetadataWriter::expire_snapshots`],
//!    [`crate::multicatalog::MulticatalogManager::expire_snapshots_in_catalog`]) deletes the
//!    chosen snapshots, garbage-collects every table / data file / delete file that is no
//!    longer reachable by any surviving snapshot, and records the orphaned physical paths in
//!    `ducklake_files_scheduled_for_deletion`. No object storage is touched. A dead table's
//!    catalog-inlined storage goes too, in the metadata database: its inlined data and
//!    inlined delete tables are dropped.
//! 2. **cleanup_old_files** ([`cleanup_old_files_sqlite`], [`cleanup_old_files_in_catalog`])
//!    reads the scheduled rows, deletes the objects from the object store, and removes the rows.
//! 3. **delete_orphaned_files** ([`delete_orphaned_files_sqlite`],
//!    [`delete_orphaned_files_in_catalog`], [`delete_orphaned_files_multicatalog`]) lists a data
//!    path, subtracts referenced data files, delete files, and files still scheduled for deletion,
//!    and deletes whatever is left. The catalog-scoped form catches files left by aborted writes.
//!    The global form also reclaims dropped-catalog files from retained data-path tombstones.
//!
//! [`CatalogMaintenance`] dispatches the three per backend; a writer exposes it through
//! [`MetadataWriter::maintenance`](crate::metadata_writer::MetadataWriter::maintenance), which
//! is how [`DuckLakeCatalog::checkpoint`](crate::DuckLakeCatalog::checkpoint) runs them.
//!
//! The metadata writers deliberately hold no object store — physical I/O lives here so the
//! catalog layer stays storage-agnostic (the object store comes from the caller, e.g. the
//! same one a [`crate::table_writer::DuckLakeTableWriter`] was built with).

use crate::Result;
#[cfg(feature = "write-postgres")]
use crate::path_resolver::join_paths;
use crate::path_resolver::{parse_object_store_url, resolve_path};
use chrono::{DateTime, Utc};
#[cfg(feature = "write-postgres")]
use datafusion::datasource::object_store::ObjectStoreUrl;
use futures::TryStreamExt;
use object_store::path::Path as ObjectPath;
use object_store::{ObjectStore, ObjectStoreExt};
#[cfg(feature = "write-postgres")]
use std::collections::BTreeMap;
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

/// The backend-specific maintenance commands, dispatched per metadata writer
/// through [`MetadataWriter::maintenance`](crate::metadata_writer::MetadataWriter::maintenance).
///
/// Each method is the backend's own entry point (for example
/// [`cleanup_old_files_sqlite`]); the trait only lets a caller holding a
/// `dyn MetadataWriter` reach them.
#[async_trait::async_trait]
pub trait CatalogMaintenance: Send + Sync {
    /// Expire snapshots; see the module docs.
    async fn expire_snapshots(&self, criteria: ExpireCriteria) -> Result<Vec<ExpiredSnapshot>>;

    /// Physically delete files scheduled for deletion and drop their rows.
    async fn cleanup_old_files(
        &self,
        object_store: Arc<dyn ObjectStore>,
        criteria: CleanupCriteria,
        dry_run: bool,
    ) -> Result<Vec<String>>;

    /// Delete the Parquet files under the catalog's data path that no metadata
    /// row references.
    async fn delete_orphaned_files(
        &self,
        object_store: Arc<dyn ObjectStore>,
        criteria: CleanupCriteria,
        dry_run: bool,
    ) -> Result<Vec<String>>;
}

#[cfg(feature = "write-sqlite")]
#[async_trait::async_trait]
impl CatalogMaintenance for crate::metadata_writer_sqlite::SqliteMetadataWriter {
    async fn expire_snapshots(&self, criteria: ExpireCriteria) -> Result<Vec<ExpiredSnapshot>> {
        crate::metadata_writer_sqlite::SqliteMetadataWriter::expire_snapshots(self, criteria)
    }

    async fn cleanup_old_files(
        &self,
        object_store: Arc<dyn ObjectStore>,
        criteria: CleanupCriteria,
        dry_run: bool,
    ) -> Result<Vec<String>> {
        cleanup_old_files_sqlite(self, object_store, criteria, dry_run).await
    }

    async fn delete_orphaned_files(
        &self,
        object_store: Arc<dyn ObjectStore>,
        criteria: CleanupCriteria,
        dry_run: bool,
    ) -> Result<Vec<String>> {
        delete_orphaned_files_sqlite(self, object_store, criteria, dry_run).await
    }
}

#[cfg(feature = "write-duckdb")]
#[async_trait::async_trait]
impl CatalogMaintenance for crate::metadata_writer_duckdb::DuckdbMetadataWriter {
    async fn expire_snapshots(&self, criteria: ExpireCriteria) -> Result<Vec<ExpiredSnapshot>> {
        crate::metadata_writer_duckdb::DuckdbMetadataWriter::expire_snapshots(self, criteria)
    }

    async fn cleanup_old_files(
        &self,
        object_store: Arc<dyn ObjectStore>,
        criteria: CleanupCriteria,
        dry_run: bool,
    ) -> Result<Vec<String>> {
        cleanup_old_files_duckdb(self, object_store, criteria, dry_run).await
    }

    async fn delete_orphaned_files(
        &self,
        object_store: Arc<dyn ObjectStore>,
        criteria: CleanupCriteria,
        dry_run: bool,
    ) -> Result<Vec<String>> {
        delete_orphaned_files_duckdb(self, object_store, criteria, dry_run).await
    }
}

#[cfg(feature = "write-mysql")]
#[async_trait::async_trait]
impl CatalogMaintenance for crate::metadata_writer_mysql::MySqlMetadataWriter {
    async fn expire_snapshots(&self, criteria: ExpireCriteria) -> Result<Vec<ExpiredSnapshot>> {
        crate::metadata_writer_mysql::MySqlMetadataWriter::expire_snapshots(self, criteria)
    }

    async fn cleanup_old_files(
        &self,
        object_store: Arc<dyn ObjectStore>,
        criteria: CleanupCriteria,
        dry_run: bool,
    ) -> Result<Vec<String>> {
        cleanup_old_files_mysql(self, object_store, criteria, dry_run).await
    }

    async fn delete_orphaned_files(
        &self,
        object_store: Arc<dyn ObjectStore>,
        criteria: CleanupCriteria,
        dry_run: bool,
    ) -> Result<Vec<String>> {
        delete_orphaned_files_mysql(self, object_store, criteria, dry_run).await
    }
}

/// The multicatalog layout's commands run on the writer's own catalog, through
/// [`crate::multicatalog::MulticatalogManager`] and its catalog-scoped functions.
#[cfg(feature = "write-postgres")]
#[async_trait::async_trait]
impl CatalogMaintenance for crate::metadata_writer_postgres::PostgresMetadataWriter {
    async fn expire_snapshots(&self, criteria: ExpireCriteria) -> Result<Vec<ExpiredSnapshot>> {
        let (manager, name) = self.catalog_manager().await?;
        manager.expire_snapshots_in_catalog(&name, criteria).await
    }

    async fn cleanup_old_files(
        &self,
        object_store: Arc<dyn ObjectStore>,
        criteria: CleanupCriteria,
        dry_run: bool,
    ) -> Result<Vec<String>> {
        let (manager, name) = self.catalog_manager().await?;
        cleanup_old_files_in_catalog(&manager, &name, object_store, criteria, dry_run).await
    }

    async fn delete_orphaned_files(
        &self,
        object_store: Arc<dyn ObjectStore>,
        criteria: CleanupCriteria,
        dry_run: bool,
    ) -> Result<Vec<String>> {
        let (manager, name) = self.catalog_manager().await?;
        delete_orphaned_files_in_catalog(&manager, &name, object_store, criteria, dry_run).await
    }
}

#[cfg(feature = "write-postgres")]
struct DataPathGroup {
    object_store_url: ObjectStoreUrl,
    object_path: ObjectPath,
    data_paths: Vec<String>,
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

/// The instant `interval` before `now`, for the `delete_older_than` catalog
/// option, which holds a DuckDB `INTERVAL` string (see [`interval_before`]).
///
/// A negative interval is refused, where official accepts it: its cutoff lies
/// in the future, so cleanup would delete the files of commits still in flight.
pub(crate) fn older_than(interval: &str, now: DateTime<Utc>) -> Result<DateTime<Utc>> {
    let cutoff = interval_before(interval, now)?;
    if cutoff > now {
        return Err(crate::DuckLakeError::InvalidConfig(format!(
            "interval '{interval}' is negative; an older-than cutoff must not be in the future"
        )));
    }
    Ok(cutoff)
}

/// The instant `interval` before `now`, for the `expire_older_than` and
/// `delete_older_than` catalog options, which hold a DuckDB `INTERVAL` string
/// (`'2 days'`, `'1h30m'`, `'-00:30:00'`, `'2 days ago'`); official parses it
/// with DuckDB's `Interval::FromString` (see [`parse_duckdb_interval`]) and
/// subtracts it from the current timestamp. A negative interval gives an
/// instant after `now`, as in DuckDB. Months shift the calendar month,
/// clamping the day, then days and microseconds follow, as DuckDB's
/// timestamp arithmetic does.
pub(crate) fn interval_before(interval: &str, now: DateTime<Utc>) -> Result<DateTime<Utc>> {
    let invalid =
        || crate::DuckLakeError::InvalidConfig(format!("Failed to parse interval: '{interval}'"));
    let parsed = parse_duckdb_interval(interval).ok_or_else(invalid)?;
    let months = i64::from(parsed.months);
    let shifted = if months >= 0 {
        now.checked_sub_months(chrono::Months::new(
            u32::try_from(months).map_err(|_| invalid())?,
        ))
    } else {
        now.checked_add_months(chrono::Months::new(
            u32::try_from(-months).map_err(|_| invalid())?,
        ))
    }
    .ok_or_else(invalid)?;
    shifted
        .checked_sub_signed(chrono::Duration::days(i64::from(parsed.days)))
        .and_then(|shifted| {
            shifted.checked_sub_signed(chrono::Duration::microseconds(parsed.micros))
        })
        .ok_or_else(invalid)
}

/// A DuckDB `INTERVAL` value: months, days and microseconds, kept apart as
/// DuckDB keeps them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct DuckdbInterval {
    pub(crate) months: i32,
    pub(crate) days: i32,
    pub(crate) micros: i64,
}

/// Parse `text` as DuckDB 1.5.5's `Interval::FromCString` does, `None` where
/// it fails:
///
/// - an optional leading `@`, then `<number> <unit>` pairs separated by
///   optional whitespace; a number is digits, with an optional fraction, and
///   `-` for a negative one (no `+`, and no bare `.5`);
/// - a unit is one of DuckDB's date part names, any case; a fraction of a
///   month or quarter becomes days of 30, of a week or day the time, of a
///   year or longer whole months, each truncated;
/// - a lone number with no unit is seconds;
/// - a number followed by `:` starts an `HH:MM[:SS[.ffffff]]` time, which
///   ends the interval (anything after it is ignored); a `-` before it negates
///   every time part read so far;
/// - a trailing `ago` negates the whole interval.
pub(crate) fn parse_duckdb_interval(text: &str) -> Option<DuckdbInterval> {
    const MICROS_PER_SEC: i64 = 1_000_000;
    const MICROS_PER_MINUTE: i64 = 60 * MICROS_PER_SEC;
    const MICROS_PER_HOUR: i64 = 60 * MICROS_PER_MINUTE;
    const MICROS_PER_DAY: i64 = 24 * MICROS_PER_HOUR;
    const DAYS_PER_MONTH: i64 = 30;
    const DAYS_PER_WEEK: i64 = 7;

    fn add_i32(target: &mut i32, input: i64, multiplier: i64, fraction: i64) -> Option<()> {
        let addition = i32::try_from(input.checked_mul(multiplier)?).ok()?;
        *target = target.checked_add(addition)?;
        if fraction != 0 {
            let addition = i32::try_from((fraction * multiplier) / MICROS_PER_SEC).ok()?;
            *target = target.checked_add(addition)?;
        }
        Some(())
    }
    fn add_i64(target: &mut i64, input: i64, multiplier: i64, fraction: i64) -> Option<()> {
        *target = target.checked_add(input.checked_mul(multiplier)?)?;
        if fraction != 0 {
            *target = target.checked_add((fraction * multiplier) / MICROS_PER_SEC)?;
        }
        Some(())
    }
    let is_space = |c: u8| matches!(c, b' ' | b'\t' | b'\n');

    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut result = DuckdbInterval::default();
    if len == 0 {
        return None;
    }
    let mut pos = 0;
    match bytes[0] {
        b'@' => pos += 1,
        b'P' | b'p' => return None,
        _ => {},
    }
    let mut found_any = false;
    loop {
        // A number, `ago`, or the end.
        while pos < len && is_space(bytes[pos]) {
            pos += 1;
        }
        if pos >= len {
            break;
        }
        let negative = match bytes[pos] {
            b'0'..=b'9' => false,
            b'-' => {
                pos += 1;
                true
            },
            b'a' | b'A' => {
                // `ago`, then only whitespace.
                if len - pos < 3
                    || !bytes[pos + 1].eq_ignore_ascii_case(&b'g')
                    || !bytes[pos + 2].eq_ignore_ascii_case(&b'o')
                    || !bytes[pos + 3..].iter().all(|c| is_space(*c))
                {
                    return None;
                }
                if result.months == i32::MIN || result.days == i32::MIN {
                    return None;
                }
                result.months = -result.months;
                result.days = -result.days;
                // DuckDB's negation of the microseconds wraps.
                result.micros = result.micros.wrapping_neg();
                break;
            },
            _ => return None,
        };
        let start = pos;
        while pos < len && bytes[pos].is_ascii_digit() {
            pos += 1;
        }
        if pos < len && bytes[pos] == b':' {
            // `HH:MM[:SS[.ffffff]]`, at most 9 hour digits.
            let time = parse_interval_time(&bytes[start..])?;
            result.micros = result.micros.checked_add(time)?;
            if negative {
                result.micros = result.micros.wrapping_neg();
            }
            found_any = true;
            break;
        }
        if pos == start {
            return None;
        }
        let mut number: i64 = text[start..pos].parse().ok()?;
        let mut fraction: i64 = 0;
        if pos < len && bytes[pos] == b'.' {
            let mut mult = 100_000;
            pos += 1;
            while pos < len && bytes[pos].is_ascii_digit() {
                if mult > 0 {
                    fraction += i64::from(bytes[pos] - b'0') * mult;
                }
                mult /= 10;
                pos += 1;
            }
        }
        if negative {
            number = -number;
            fraction = -fraction;
        }
        while pos < len && is_space(bytes[pos]) {
            pos += 1;
        }
        let unit_start = pos;
        while pos < len && bytes[pos].is_ascii_alphabetic() {
            pos += 1;
        }
        let unit = text[unit_start..pos].to_ascii_lowercase();
        if unit.is_empty() && !found_any {
            // A lone number is seconds; only whitespace may follow.
            add_i64(&mut result.micros, number, MICROS_PER_SEC, 0)?;
            result.micros = result.micros.checked_add(fraction)?;
            if !bytes[pos..].iter().all(|c| is_space(*c)) {
                return None;
            }
            found_any = true;
            break;
        }
        match unit.as_str() {
            "millennium" | "mil" | "millenniums" | "millennia" | "mils" | "millenium" => {
                add_i32(&mut result.months, number, 12_000, fraction)?
            },
            "century" | "cent" | "centuries" | "c" => {
                add_i32(&mut result.months, number, 1_200, fraction)?
            },
            "decade" | "dec" | "decades" | "decs" => {
                add_i32(&mut result.months, number, 120, fraction)?
            },
            "year" | "yr" | "y" | "years" | "yrs" => {
                add_i32(&mut result.months, number, 12, fraction)?
            },
            "quarter" | "quarters" => {
                add_i32(&mut result.months, number, 3, fraction)?;
                let fraction = (fraction * 3) % MICROS_PER_SEC;
                add_i32(&mut result.days, 0, DAYS_PER_MONTH, fraction)?;
            },
            "month" | "mon" | "months" | "mons" => {
                add_i32(&mut result.months, number, 1, 0)?;
                add_i32(&mut result.days, 0, DAYS_PER_MONTH, fraction)?;
            },
            "day" | "days" | "d" | "dayofmonth" => {
                add_i32(&mut result.days, number, 1, 0)?;
                add_i64(&mut result.micros, 0, MICROS_PER_DAY, fraction)?;
            },
            "week" | "weeks" | "w" | "weekofyear" => {
                add_i32(&mut result.days, number, DAYS_PER_WEEK, fraction)?;
                let fraction = (fraction * DAYS_PER_WEEK) % MICROS_PER_SEC;
                add_i64(&mut result.micros, 0, MICROS_PER_DAY, fraction)?;
            },
            "microseconds" | "microsecond" | "us" | "usec" | "usecs" | "usecond" | "useconds" => {
                let number = number.checked_add((fraction * 2) / MICROS_PER_SEC)?;
                add_i64(&mut result.micros, number, 1, 0)?;
            },
            "milliseconds" | "millisecond" | "ms" | "msec" | "msecs" | "msecond" | "mseconds" => {
                add_i64(&mut result.micros, number, 1_000, fraction)?
            },
            "second" | "sec" | "seconds" | "secs" | "s" => {
                add_i64(&mut result.micros, number, MICROS_PER_SEC, fraction)?
            },
            "minute" | "min" | "minutes" | "mins" | "m" => {
                add_i64(&mut result.micros, number, MICROS_PER_MINUTE, fraction)?
            },
            "hour" | "hr" | "hours" | "hrs" | "h" => {
                add_i64(&mut result.micros, number, MICROS_PER_HOUR, fraction)?
            },
            _ => return None,
        }
        found_any = true;
    }
    found_any.then_some(result)
}

/// The microseconds of an `HH:MM[:SS[.ffffff]]` interval time at the start of
/// `bytes`, as DuckDB's non-strict `Time::TryConvertInterval` reads it: up to
/// 9 hour digits, minutes and seconds of one or two digits below 60, and
/// whatever follows the seconds ignored.
fn parse_interval_time(bytes: &[u8]) -> Option<i64> {
    let len = bytes.len();
    let mut pos = 0;
    let mut hour: i64 = 0;
    let mut digits = 0;
    while pos < len && bytes[pos].is_ascii_digit() {
        digits += 1;
        if digits > 9 {
            return None;
        }
        hour = hour * 10 + i64::from(bytes[pos] - b'0');
        pos += 1;
    }
    if digits == 0 || pos >= len || bytes[pos] != b':' {
        return None;
    }
    pos += 1;
    let double_digit = |pos: &mut usize| -> Option<i64> {
        if *pos < len && bytes[*pos].is_ascii_digit() {
            let mut value = i64::from(bytes[*pos] - b'0');
            *pos += 1;
            if *pos < len && bytes[*pos].is_ascii_digit() {
                value = value * 10 + i64::from(bytes[*pos] - b'0');
                *pos += 1;
            }
            Some(value)
        } else {
            None
        }
    };
    let minute = if pos == len {
        0
    } else {
        double_digit(&mut pos)?
    };
    if minute >= 60 {
        return None;
    }
    let second = if pos >= len {
        0
    } else if bytes[pos] != b':' {
        return None;
    } else {
        pos += 1;
        if pos == len {
            0
        } else {
            double_digit(&mut pos)?
        }
    };
    if second >= 60 {
        return None;
    }
    let mut micros: i64 = 0;
    if pos < len && bytes[pos] == b'.' {
        pos += 1;
        let mut mult = 100_000;
        while pos < len && bytes[pos].is_ascii_digit() {
            if mult > 0 {
                micros += i64::from(bytes[pos] - b'0') * mult;
            }
            mult /= 10;
            pos += 1;
        }
    }
    Some(((hour * 60 + minute) * 60 + second) * 1_000_000 + micros)
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
    /// Whether the catalog that scheduled this row owned the object, on the
    /// multicatalog Postgres layout. `Some(true)` on every row the ownership rule
    /// produced — a catalog only ever schedules what it owns — so its job is to mark
    /// the row's provenance, not to name a second party. `None` on every other
    /// backend, and on a multicatalog row written before the rule existed, which
    /// [`cleanup_old_files_in_catalog`] therefore treats as unverified.
    ///
    /// Deliberately NOT called `owner_catalog_id` like the column on
    /// `ducklake_catalog_data_file_reference`: there, a value names ANOTHER catalog, and one name
    /// carrying opposite meanings in two tables is what a later query copies across
    /// and gets backwards.
    pub scheduled_by_owner: Option<bool>,
}

/// Resolve scheduled rows against `data_path`, delete the objects (unless `dry_run`),
/// and return the resolved absolute paths that were (or would be) deleted. Shared
/// by both backends; the row listing / row removal is backend-specific and passed in.
async fn run_cleanup<RemoveFut>(
    data_path: &str,
    files: Vec<ScheduledFile>,
    object_store: Arc<dyn ObjectStore>,
    dry_run: bool,
    remove_rows: impl FnOnce(Vec<ScheduledFile>) -> RemoveFut,
) -> Result<Vec<String>>
where
    RemoveFut: std::future::Future<Output = Result<()>>,
{
    if files.is_empty() {
        return Ok(Vec::new());
    }
    let (_, base_key) = parse_object_store_url(data_path)?;

    let mut resolved = Vec::with_capacity(files.len());
    for file in &files {
        let abs = resolve_path(&base_key, &file.path, file.path_is_relative)?;
        resolved.push(abs);
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

    remove_rows(files).await?;
    Ok(resolved)
}

/// The object-store key a resolved path actually names — the same transform
/// [`run_cleanup`] applies before deleting and [`run_orphan_cleanup`] applies before
/// comparing.
///
/// Reclaim safety turns on "does any catalog still reference this object", and the two
/// sides of that question are built by different code: a referencing row's path comes
/// from the registering engine's own nested resolve, a scheduled row's from the SQL
/// resolvers, which always insert a separator where [`crate::path_resolver::join_paths`]
/// omits one after a base that already ends in it. Comparing canonical keys rather than
/// raw strings keeps a one-character spelling difference from reading as "unreferenced"
/// and deleting a live file.
#[cfg(feature = "write-postgres")]
fn canonical_key(resolved: &str) -> String {
    ObjectPath::from(resolved.trim_start_matches('/')).to_string()
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
    run_cleanup(
        &data_path,
        files,
        object_store,
        dry_run,
        |files| async move { writer.remove_scheduled(&scheduled_ids(&files)) },
    )
    .await
}

/// The `data_file_id` column of every scheduled row, for backends that key their
/// bookkeeping removal by id.
///
/// The multicatalog Postgres backend deliberately does not: see
/// [`crate::multicatalog::MulticatalogManager::remove_scheduled_paths_in_catalog`].
#[cfg(any(feature = "write-sqlite", feature = "write-duckdb", feature = "write-mysql"))]
fn scheduled_ids(files: &[ScheduledFile]) -> Vec<i64> {
    files.iter().map(|f| f.data_file_id).collect()
}

/// Physically delete files scheduled by
/// [`DuckdbMetadataWriter::expire_snapshots`] and remove their bookkeeping rows.
/// Returns resolved absolute paths deleted (or, for `dry_run`, paths that would
/// be deleted).
///
/// [`DuckdbMetadataWriter::expire_snapshots`]: crate::metadata_writer_duckdb::DuckdbMetadataWriter::expire_snapshots
#[cfg(feature = "write-duckdb")]
pub async fn cleanup_old_files_duckdb(
    writer: &crate::metadata_writer_duckdb::DuckdbMetadataWriter,
    object_store: Arc<dyn ObjectStore>,
    criteria: CleanupCriteria,
    dry_run: bool,
) -> Result<Vec<String>> {
    let data_path = crate::metadata_writer::MetadataWriter::get_data_path(writer)?;
    let files = writer.list_scheduled_for_deletion(&criteria)?;
    run_cleanup(
        &data_path,
        files,
        object_store,
        dry_run,
        |files| async move { writer.remove_scheduled(&scheduled_ids(&files)) },
    )
    .await
}

/// Physically delete files scheduled by
/// [`MySqlMetadataWriter::expire_snapshots`] and remove their bookkeeping rows.
/// Returns resolved absolute paths deleted (or, for `dry_run`, paths that would
/// be deleted).
///
/// [`MySqlMetadataWriter::expire_snapshots`]: crate::metadata_writer_mysql::MySqlMetadataWriter::expire_snapshots
#[cfg(feature = "write-mysql")]
pub async fn cleanup_old_files_mysql(
    writer: &crate::metadata_writer_mysql::MySqlMetadataWriter,
    object_store: Arc<dyn ObjectStore>,
    criteria: CleanupCriteria,
    dry_run: bool,
) -> Result<Vec<String>> {
    let data_path = crate::metadata_writer::MetadataWriter::get_data_path(writer)?;
    let files = writer.list_scheduled_for_deletion(&criteria)?;
    run_cleanup(
        &data_path,
        files,
        object_store,
        dry_run,
        |files| async move { writer.remove_scheduled(&scheduled_ids(&files)) },
    )
    .await
}

/// Physically delete files scheduled by
/// [`MulticatalogManager::expire_snapshots_in_catalog`] for `catalog_name` and remove their
/// bookkeeping rows. Returns the resolved absolute paths deleted (or, for `dry_run`, the
/// paths that would be deleted).
///
/// A scheduled file that an absolute-path data or delete file row in ANY catalog still
/// points at is not deleted: another catalog registered that object by reference (a
/// database fork does this instead of copying), so the owner's reclaim waits. The row
/// stays scheduled and its `schedule_start` is reset, so an `OlderThan` sweep leaves it
/// alone for another grace period; once the last reference row is gone the next sweep
/// deletes it. Referencing catalogs never schedule such rows themselves (see
/// `schedule_pg_files`), so the owner's scheduled row is the only reclaim intent — and
/// when the owner catalog is dropped first, the object is reclaimed by the orphan sweep
/// once no row anywhere resolves to it.
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
    let data_path = mgr.get_data_path_in_catalog(catalog_name).await?;
    let files = mgr
        .list_scheduled_for_deletion_in_catalog(catalog_name, &criteria)
        .await?;
    let files = retain_unreferenced(mgr, catalog_name, &data_path, files, dry_run).await?;
    run_cleanup(
        &data_path,
        files,
        object_store,
        dry_run,
        |files| async move {
            let paths: Vec<String> = files.into_iter().map(|f| f.path).collect();
            mgr.remove_scheduled_paths_in_catalog(catalog_name, &paths)
                .await
        },
    )
    .await
}

/// Drop from `files` every scheduled row this catalog must not delete: one whose
/// object a reference row in some catalog still names (deferred, so the owner
/// reclaims it once the last reference goes), and one whose object this catalog does
/// not own at all (skipped and warned about, never deleted).
///
/// Both comparisons are on canonical object-store keys, not raw strings — see
/// [`canonical_key`] for why a raw comparison is unsafe here.
///
/// Two guards make this fail closed, and the first holds unconditionally.
///
/// **An object inside another catalog's `cat_{id}/` layout is never reclaimed here**,
/// however the row was scheduled and whenever. That catalog owns everything under its
/// own prefix by construction, so a scheduled row of ours naming one is wrong whatever
/// produced it. The check cannot be waived by the row's own claim of ownership,
/// because the case that most needs catching is a row whose claim is the thing that is
/// wrong: a reference registered while `path_is_relative = false` still meant
/// "reference" reads as owned once ownership is recorded explicitly (such a row got no
/// reference record), so this catalog's expire schedules the owner's live object
/// and stamps it as its own to reclaim. Trusting that stamp deletes another catalog's
/// data. A legacy pre-`cat_{id}` layout is unaffected: its files sit under no
/// `cat_{id}/` prefix at all, so nothing matches.
///
/// **A row of unknown provenance is reclaimed only under this catalog's own layout.**
/// A row this code scheduled carries `scheduled_by_owner` — a catalog only ever
/// schedules what it owns — which is what lets a catalog reclaim its own file written
/// outside its layout (a [`crate::DuckLakeTableWriter::begin_write_to_path`] target,
/// which may sit outside `data_path` entirely). A row without that marker predates the
/// ownership rule: a build without the rule did schedule files it merely referenced,
/// so a metadata database written by a mixed pair of versions can hold one naming a
/// live file another catalog reads, with the reference row that would have protected
/// it already gone. Absolute rows like that are refused, with a warning, anywhere but
/// this catalog's own prefix. Relative rows are exempt from this second guard only:
/// they resolve through this catalog's own schema and table rows by construction,
/// which is what a legacy pre-`cat_{id}` layout registers.
#[cfg(feature = "write-postgres")]
async fn retain_unreferenced(
    mgr: &crate::multicatalog::MulticatalogManager,
    catalog_name: &str,
    data_path: &str,
    files: Vec<ScheduledFile>,
    dry_run: bool,
) -> Result<Vec<ScheduledFile>> {
    if files.is_empty() {
        return Ok(files);
    }
    let (_, base_key) = parse_object_store_url(data_path)?;
    let own_catalog_id = mgr.find_catalog_id(catalog_name).await?;
    let own_prefix = match own_catalog_id {
        Some(id) => Some(canonical_key(&join_paths(&base_key, &format!("cat_{id}"))?)),
        None => None,
    };
    // Every OTHER catalog's own-layout prefix, each resolved against ITS data_path
    // rather than ours — nothing makes a referrer share its owner's root. A parse
    // failure propagates rather than dropping that catalog from the set: losing a
    // prefix here silently re-opens the delete this guard exists to refuse.
    let mut foreign_prefixes: Vec<String> = Vec::new();
    for (id, path) in mgr.list_catalog_data_paths().await? {
        if Some(id) == own_catalog_id {
            continue;
        }
        let (_, foreign_base) = parse_object_store_url(&path)?;
        foreign_prefixes.push(canonical_key(&join_paths(
            &foreign_base,
            &format!("cat_{id}"),
        )?));
    }

    let mut resolved = Vec::with_capacity(files.len());
    for file in &files {
        resolved.push(resolve_path(&base_key, &file.path, file.path_is_relative)?);
    }
    // Canonicalise the whole stored side, rather than probing it with the spellings
    // this side happens to produce. A reference row names its object in whatever form
    // its registrar wrote; `ObjectPath` folds those forms together, SQL equality does
    // not. Selecting by equality would return only the rows whose spelling was guessed,
    // and a reference that does not come back is indistinguishable from one that does
    // not exist — which deletes a file another catalog is reading.
    let referenced: HashSet<String> = mgr
        .all_reference_paths()
        .await?
        .iter()
        .map(|p| canonical_key(p))
        .collect();

    let mut kept = Vec::with_capacity(files.len());
    let mut deferred = Vec::new();
    let under = |key: &str, prefix: &str| key == prefix || key.starts_with(&format!("{prefix}/"));
    for (file, abs) in files.into_iter().zip(resolved) {
        let key = canonical_key(&abs);
        if let Some(foreign) = foreign_prefixes.iter().find(|p| under(&key, p)) {
            tracing::warn!(
                catalog = catalog_name,
                path = %abs,
                owner_prefix = %foreign,
                "refusing to reclaim a scheduled file inside another catalog's own \
                 layout; that catalog owns it and is left to reclaim it"
            );
            continue;
        }
        if file.scheduled_by_owner != Some(true)
            && !file.path_is_relative
            && let Some(prefix) = &own_prefix
            && !under(&key, prefix)
        {
            tracing::warn!(
                catalog = catalog_name,
                path = %abs,
                own_prefix = %prefix,
                "refusing to reclaim a scheduled file outside this catalog's own layout \
                 that was scheduled before file ownership was recorded; it may name an \
                 object another catalog owns and is left scheduled"
            );
            continue;
        }
        if referenced.contains(&key) {
            deferred.push(file.path);
        } else {
            kept.push(file);
        }
    }
    if !dry_run {
        mgr.defer_scheduled_in_catalog(catalog_name, &deferred)
            .await?;
    }
    Ok(kept)
}

/// Reclaim metadata rows whose owning row is already gone, on a Postgres store.
///
/// Step 3 of a sweep, after expire and cleanup. Expire deletes the children of the
/// owners it retires; this collects children whose owners were retired by an
/// earlier version that did not, and which no future expire can reach because it
/// deletes by the ids it is retiring. Bounded per call, so a large backlog drains
/// over successive sweeps rather than stalling one.
///
/// Safe by construction rather than by heuristic: a row whose owner is absent is
/// unreachable -- every read joins through the owner -- and owner and children are
/// written in one transaction, so no snapshot sees a child whose owner has not
/// landed.
#[cfg(feature = "write-postgres")]
pub async fn purge_orphaned_metadata_postgres(pool: &sqlx::PgPool) -> Result<()> {
    crate::metadata_writer_postgres::purge_orphaned_metadata(pool).await
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
    // writes via the timestamp filter is what makes this safe to schedule, and
    // only while the cutoff is earlier than the start of the longest write still
    // running: a streaming write uploads each file as it finishes, so its first
    // file is as old as the write itself, and is unreferenced until it commits.
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
/// being written by in-flight transactions are skipped. That holds only when the
/// cutoff is earlier than the start of the longest write that may still be
/// running: a streaming write puts each file in the data path as soon as it is
/// finished and commits them all at the end, so its first file is as old as the
/// write. `CleanupCriteria::All` is allowed (matching the official
/// `cleanup_all => true`) but should be used only when the catalog is known to be
/// idle.
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

/// List the DuckDB catalog's `data_path` and delete every unreferenced Parquet
/// file.
///
/// An `OlderThan` cutoff must be earlier than the start of the longest write that may still
/// be running: a streaming write puts each file in the data path as soon as it is finished
/// and commits them all at the end.
#[cfg(feature = "write-duckdb")]
pub async fn delete_orphaned_files_duckdb(
    writer: &crate::metadata_writer_duckdb::DuckdbMetadataWriter,
    object_store: Arc<dyn ObjectStore>,
    criteria: CleanupCriteria,
    dry_run: bool,
) -> Result<Vec<String>> {
    let data_path = crate::metadata_writer::MetadataWriter::get_data_path(writer)?;
    let referenced = writer.list_referenced_paths()?;
    run_orphan_cleanup(&data_path, referenced, object_store, criteria, dry_run).await
}

/// List the MySQL catalog's `data_path` and delete every unreferenced Parquet
/// file.
#[cfg(feature = "write-mysql")]
pub async fn delete_orphaned_files_mysql(
    writer: &crate::metadata_writer_mysql::MySqlMetadataWriter,
    object_store: Arc<dyn ObjectStore>,
    criteria: CleanupCriteria,
    dry_run: bool,
) -> Result<Vec<String>> {
    let data_path = crate::metadata_writer::MetadataWriter::get_data_path(writer)?;
    let referenced = writer.list_referenced_paths()?;
    run_orphan_cleanup(&data_path, referenced, object_store, criteria, dry_run).await
}

/// Sweep every registered multicatalog data path and delete unreferenced Parquet files.
///
/// All roots must resolve to the same object-store authority because this API
/// accepts one object store. Use [`delete_orphaned_files_in_data_path`] with the
/// matching store when one metadata database spans multiple authorities.
///
/// An `OlderThan` cutoff must be earlier than the start of the longest write that may still
/// be running: a streaming write puts each file in the data path as soon as it is finished
/// and commits them all at the end.
#[cfg(feature = "write-postgres")]
pub async fn delete_orphaned_files_multicatalog(
    mgr: &crate::multicatalog::MulticatalogManager,
    object_store: Arc<dyn ObjectStore>,
    criteria: CleanupCriteria,
    dry_run: bool,
) -> Result<Vec<String>> {
    let data_path_groups = merge_data_path_groups(group_data_paths(mgr.list_data_paths().await?)?);
    if data_path_groups.is_empty() {
        return Err(crate::DuckLakeError::InvalidConfig(
            "Missing required catalog metadata: 'data_path' not configured.".to_string(),
        ));
    }
    let mut authority = None;
    for group in &data_path_groups {
        if authority
            .as_ref()
            .is_some_and(|expected| expected != &group.object_store_url)
        {
            return Err(crate::DuckLakeError::InvalidConfig(
                "Multicatalog data paths span multiple object-store authorities; clean each path with delete_orphaned_files_in_data_path"
                    .to_string(),
            ));
        }
        authority = Some(group.object_store_url.clone());
    }

    let mut deleted = Vec::new();
    for group in data_path_groups {
        deleted.extend(
            delete_orphaned_files_in_data_path_inner(
                mgr,
                &group,
                Arc::clone(&object_store),
                criteria.clone(),
                dry_run,
            )
            .await?,
        );
    }
    deleted.sort();
    deleted.dedup();
    Ok(deleted)
}

/// Delete orphaned Parquet files within one catalog's canonical data path.
///
/// The sweep includes every catalog root nested under that path because object-store listing is
/// recursive. References from every included root are retained.
///
/// An `OlderThan` cutoff must be earlier than the start of the longest write that may still
/// be running: a streaming write puts each file in the data path as soon as it is finished
/// and commits them all at the end.
#[cfg(feature = "write-postgres")]
pub async fn delete_orphaned_files_in_catalog(
    mgr: &crate::multicatalog::MulticatalogManager,
    catalog_name: &str,
    object_store: Arc<dyn ObjectStore>,
    criteria: CleanupCriteria,
    dry_run: bool,
) -> Result<Vec<String>> {
    let data_path = mgr.get_data_path_in_catalog(catalog_name).await?;
    let group = registered_data_path_group(mgr, &data_path).await?;
    delete_orphaned_files_in_data_path_inner(mgr, &group, object_store, criteria, dry_run).await
}

/// Delete orphaned Parquet files within one retained multicatalog data path.
///
/// `object_store` must serve the authority parsed from `data_path`. The object-store trait does not
/// expose its authority for runtime validation.
///
/// An `OlderThan` cutoff must be earlier than the start of the longest write that may still
/// be running: a streaming write puts each file in the data path as soon as it is finished
/// and commits them all at the end.
#[cfg(feature = "write-postgres")]
pub async fn delete_orphaned_files_in_data_path(
    mgr: &crate::multicatalog::MulticatalogManager,
    data_path: &str,
    object_store: Arc<dyn ObjectStore>,
    criteria: CleanupCriteria,
    dry_run: bool,
) -> Result<Vec<String>> {
    let group = registered_data_path_group(mgr, data_path).await?;
    delete_orphaned_files_in_data_path_inner(mgr, &group, object_store, criteria, dry_run).await
}

#[cfg(feature = "write-postgres")]
async fn delete_orphaned_files_in_data_path_inner(
    mgr: &crate::multicatalog::MulticatalogManager,
    group: &DataPathGroup,
    object_store: Arc<dyn ObjectStore>,
    criteria: CleanupCriteria,
    dry_run: bool,
) -> Result<Vec<String>> {
    let clear_tombstone = !dry_run && matches!(&criteria, CleanupCriteria::All);
    let dropped_before = mgr.current_timestamp().await?;
    let mut referenced = Vec::new();
    for data_path in &group.data_paths {
        let (_, base_key) = parse_object_store_url(data_path)?;
        for (path, relative) in mgr
            .list_referenced_paths_in_data_paths(std::slice::from_ref(data_path))
            .await?
        {
            referenced.push((resolve_path(&base_key, &path, relative)?, false));
        }
    }
    let deleted = run_orphan_cleanup(
        &group.data_paths[0],
        referenced,
        object_store,
        criteria,
        dry_run,
    )
    .await?;
    if clear_tombstone {
        mgr.clear_dropped_data_paths(&group.data_paths, dropped_before)
            .await?;
    }
    Ok(deleted)
}

#[cfg(feature = "write-postgres")]
async fn registered_data_path_group(
    mgr: &crate::multicatalog::MulticatalogManager,
    data_path: &str,
) -> Result<DataPathGroup> {
    let mut groups = group_data_paths(mgr.list_data_paths().await?)?;
    let index = groups
        .iter()
        .position(|group| group.data_paths.iter().any(|path| path == data_path))
        .ok_or_else(|| {
            crate::DuckLakeError::InvalidConfig(format!(
                "Data path {data_path:?} is not registered for multicatalog cleanup"
            ))
        })?;
    let mut root = groups.remove(index);
    for group in groups {
        if group.object_store_url == root.object_store_url
            && group.object_path.prefix_matches(&root.object_path)
        {
            root.data_paths.extend(group.data_paths);
        }
    }
    Ok(root)
}

#[cfg(feature = "write-postgres")]
fn group_data_paths(data_paths: Vec<String>) -> Result<Vec<DataPathGroup>> {
    let mut groups = BTreeMap::new();
    for data_path in data_paths {
        let (object_store_url, base_key) = parse_object_store_url(&data_path)?;
        let object_path = ObjectPath::from(base_key.trim_start_matches('/'));
        groups
            .entry((object_store_url, object_path))
            .or_insert_with(Vec::new)
            .push(data_path);
    }
    Ok(groups
        .into_iter()
        .map(
            |((object_store_url, object_path), data_paths)| DataPathGroup {
                object_store_url,
                object_path,
                data_paths,
            },
        )
        .collect())
}

#[cfg(feature = "write-postgres")]
fn merge_data_path_groups(groups: Vec<DataPathGroup>) -> Vec<DataPathGroup> {
    let mut roots: Vec<DataPathGroup> = Vec::new();
    for group in groups {
        if let Some(root) = roots.iter_mut().find(|root| {
            group.object_store_url == root.object_store_url
                && group.object_path.prefix_matches(&root.object_path)
        }) {
            root.data_paths.extend(group.data_paths);
        } else {
            roots.push(group);
        }
    }
    roots
}

#[cfg(test)]
mod older_than_tests {
    use super::older_than;
    use chrono::{DateTime, Duration, TimeZone, Utc};

    fn now() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 3, 31, 12, 0, 0).unwrap()
    }

    /// `TIMESTAMP '2026-03-31 12:00:00' - TRY_CAST(<text> AS INTERVAL)` on the
    /// DuckDB 1.5.5 CLI, `None` where the cast fails: what official's
    /// `Interval::FromString` makes of each option value.
    #[test]
    fn interval_before_matches_duckdb() {
        for (text, expected) in [
            ("-00:30:00", Some("2026-03-31 12:30:00")),
            ("-01:30:00", Some("2026-03-31 13:30:00")),
            ("00:30:00", Some("2026-03-31 11:30:00")),
            ("1 day 02:00:00", Some("2026-03-30 10:00:00")),
            ("1 hour -00:30:00", Some("2026-03-31 13:30:00")),
            ("2 days ago", Some("2026-04-02 12:00:00")),
            ("1 day ago", Some("2026-04-01 12:00:00")),
            ("1", Some("2026-03-31 11:59:59")),
            ("1.5", Some("2026-03-31 11:59:58.5")),
            ("-1", Some("2026-03-31 12:00:01")),
            ("-1.5", Some("2026-03-31 12:00:01.5")),
            ("+1 day", None),
            (".5 days", None),
            ("0.5 days", Some("2026-03-31 00:00:00")),
            ("1.5 hours", Some("2026-03-31 10:30:00")),
            ("1h30m", Some("2026-03-31 10:30:00")),
            ("90 min", Some("2026-03-31 10:30:00")),
            ("2 weeks", Some("2026-03-17 12:00:00")),
            ("1.5 weeks", Some("2026-03-21 00:00:00")),
            ("1 mon", Some("2026-02-28 12:00:00")),
            ("1.5 months", Some("2026-02-13 12:00:00")),
            ("1.7 years", Some("2024-07-31 12:00:00")),
            ("0.5 quarter", Some("2026-02-13 12:00:00")),
            ("@ 2 days", Some("2026-03-29 12:00:00")),
            ("@2 days", Some("2026-03-29 12:00:00")),
            ("2 DAYS", Some("2026-03-29 12:00:00")),
            ("2days", Some("2026-03-29 12:00:00")),
            ("1 day garbage", None),
            ("00:30:00 garbage", Some("2026-03-31 11:30:00")),
            ("00:30:00 2 days", Some("2026-03-31 11:30:00")),
            ("- 1 day", None),
            ("--1 day", None),
            ("1 day 1 day", Some("2026-03-29 12:00:00")),
            ("1 fortnight", None),
            ("25:00:00", Some("2026-03-30 11:00:00")),
            ("00:60:00", None),
            ("1:2:3", Some("2026-03-31 10:57:57")),
            ("10:5", Some("2026-03-31 01:55:00")),
            ("1 yr 2 mons 3 d", Some("2025-01-28 12:00:00")),
            ("1 millenium", Some("1026-03-31 12:00:00")),
            ("1 dayofmonth", Some("2026-03-30 12:00:00")),
            ("1 epoch", None),
            ("3 usecond", Some("2026-03-31 11:59:59.999997")),
            ("1.5 microseconds", Some("2026-03-31 11:59:59.999998")),
            ("1 ago", None),
            ("ago", None),
            ("1 day  ago  ", Some("2026-04-01 12:00:00")),
            ("P1D", None),
            (" 2 days", Some("2026-03-29 12:00:00")),
            ("2 days ", Some("2026-03-29 12:00:00")),
            ("00:30 garbage", None),
            ("10:", Some("2026-03-31 02:00:00")),
            ("1234567890:00:00", None),
            ("1.5 quarters", Some("2025-11-15 12:00:00")),
            ("1 day 2 hours ago", Some("2026-04-01 14:00:00")),
            ("-1 day ago", Some("2026-03-30 12:00:00")),
            ("1.5 days", Some("2026-03-30 00:00:00")),
            ("3 ms", Some("2026-03-31 11:59:59.997")),
            ("1.0000005 seconds", Some("2026-03-31 11:59:59")),
            ("-2 days 03:00:00", Some("2026-04-02 09:00:00")),
        ] {
            let got = super::interval_before(text, now()).ok();
            let expected = expected.map(|expected| {
                chrono::NaiveDateTime::parse_from_str(expected, "%Y-%m-%d %H:%M:%S%.f")
                    .unwrap()
                    .and_utc()
            });
            assert_eq!(got, expected, "{text:?}");
        }
    }

    #[test]
    fn parses_duckdb_interval_strings() {
        for (text, expected) in [
            ("2 days", now() - Duration::days(2)),
            ("1 week", now() - Duration::days(7)),
            ("1d", now() - Duration::days(1)),
            ("90 min", now() - Duration::minutes(90)),
            ("1h30m", now() - Duration::minutes(90)),
            ("1.5 hours", now() - Duration::minutes(90)),
            ("00:30:00", now() - Duration::minutes(30)),
            ("1 day 02:00:00", now() - Duration::hours(26)),
            (
                "1 mon",
                Utc.with_ymd_and_hms(2026, 2, 28, 12, 0, 0).unwrap(),
            ),
            (
                "2 years",
                Utc.with_ymd_and_hms(2024, 3, 31, 12, 0, 0).unwrap(),
            ),
        ] {
            assert_eq!(older_than(text, now()).unwrap(), expected, "{text}");
        }
    }

    /// Negating the microseconds for `ago` or a negative time wraps as
    /// DuckDB 1.5.5's does: `'-1 us -9223372036854775807 us ago'` and
    /// `'-1 us -9223372036854775807 us -00:00'` are both `-9223372036854775808`
    /// microseconds.
    #[test]
    fn negating_the_smallest_micros_wraps_as_duckdb_does() {
        for text in ["-1 us -9223372036854775807 us ago", "-1 us -9223372036854775807 us -00:00"] {
            assert_eq!(
                super::parse_duckdb_interval(text),
                Some(super::DuckdbInterval {
                    months: 0,
                    days: 0,
                    micros: i64::MIN,
                }),
                "{text}"
            );
            assert!(older_than(text, now()).is_err(), "{text}");
        }
    }

    #[test]
    fn rejects_what_duckdb_rejects() {
        for text in ["", "soon", "2 fortnights", "1 q", "1 qtr"] {
            assert!(older_than(text, now()).is_err(), "{text}");
        }
    }

    /// Fractions of month-or-longer units as DuckDB 1.5.5 parses them:
    /// `INTERVAL '1.5 months'` is `1 month 15 days`, `'1.04 months'` is `1 month
    /// 1 day`, `'1.7 years'` is `1 year 8 months`, `'0.5 quarter'` is `1 month
    /// 15 days`, `'0.001 centuries'` is `1 month`.
    #[test]
    fn parses_fractional_months_as_duckdb_does() {
        let months_days = |months: u32, days: i64| {
            now()
                .checked_sub_months(chrono::Months::new(months))
                .unwrap()
                - Duration::days(days)
        };
        for (text, expected) in [
            ("1.5 months", months_days(1, 15)),
            ("1.04 months", months_days(1, 1)),
            ("1.99 months", months_days(1, 29)),
            ("1.7 years", months_days(20, 0)),
            ("1.25 years", months_days(15, 0)),
            ("0.15 decades", months_days(18, 0)),
            ("0.5 quarter", months_days(1, 15)),
            ("0.001 centuries", months_days(1, 0)),
            ("1.5 weeks", now() - Duration::hours(252)),
        ] {
            assert_eq!(older_than(text, now()).unwrap(), expected, "{text}");
        }
    }

    /// `expire_older_than` takes a negative interval, as official does: its
    /// cutoff is after now.
    #[test]
    fn interval_before_accepts_negative_intervals() {
        assert_eq!(
            super::interval_before("-1 day", now()).unwrap(),
            now() + Duration::days(1)
        );
        assert_eq!(
            super::interval_before("-1.5 months", now()).unwrap(),
            now().checked_add_months(chrono::Months::new(1)).unwrap() + Duration::days(15)
        );
    }

    /// Values beyond any representable interval fail with a typed error rather
    /// than overflowing.
    #[test]
    fn rejects_absurd_intervals() {
        for text in [
            "99999999999999999999 years",
            "9223372036854775807 months",
            "768614336404564650 years",
            "99999999999999999999 days",
            "9223372036854775807 years 9223372036854775807 years",
            "999999999999999999999999:00:00",
        ] {
            assert!(
                matches!(
                    older_than(text, now()),
                    Err(crate::DuckLakeError::InvalidConfig(_))
                ),
                "{text}"
            );
        }
    }

    /// A negative interval would give a cutoff in the future.
    #[test]
    fn rejects_a_future_cutoff() {
        for text in ["-1 day", "-1 mon", "1 day -2 days"] {
            assert!(
                matches!(
                    older_than(text, now()),
                    Err(crate::DuckLakeError::InvalidConfig(_))
                ),
                "{text}"
            );
        }
        assert_eq!(older_than("0 days", now()).unwrap(), now());
    }
}
