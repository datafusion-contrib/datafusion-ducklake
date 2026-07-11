//! Metadata writer trait and common types for DuckLake catalog writes.
//!
//! This module provides the `MetadataWriter` trait for writing metadata to DuckLake catalogs,
//! along with helper types for column definitions and data file registration.

use crate::{DuckLakeError, Result};

/// Maximum allowed length for catalog entity names (schemas, tables, columns).
pub const MAX_NAME_LENGTH: usize = 1024;

/// Validate a catalog entity name (schema, table, or column).
///
/// Rejects names that are:
/// - Empty or whitespace-only
/// - Contain ASCII control characters (0x00-0x1F, 0x7F)
/// - Exceed [`MAX_NAME_LENGTH`] characters
pub fn validate_name(name: &str, kind: &str) -> Result<()> {
    if name.trim().is_empty() {
        return Err(DuckLakeError::InvalidConfig(format!(
            "{kind} name cannot be empty or whitespace-only"
        )));
    }
    if let Some(pos) = name.find(|c: char| c.is_ascii_control()) {
        let byte = name.as_bytes()[pos];
        return Err(DuckLakeError::InvalidConfig(format!(
            "{kind} name contains control character 0x{byte:02X} at position {pos}"
        )));
    }
    if name.len() > MAX_NAME_LENGTH {
        return Err(DuckLakeError::InvalidConfig(format!(
            "{kind} name exceeds maximum length of {MAX_NAME_LENGTH} characters (got {})",
            name.len()
        )));
    }
    Ok(())
}

/// Write mode for table operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriteMode {
    /// Drop existing data and replace with new data
    Replace,
    /// Keep existing data and append new records
    Append,
}
use crate::types::{arrow_to_ducklake_type, ducklake_to_arrow_type};
use arrow::datatypes::DataType;

/// Column definition for creating or updating a table's schema.
///
/// Unlike `DuckLakeTableColumn` (used for reading), this struct doesn't have a `column_id`
/// field since IDs are assigned by the catalog during write operations.
#[derive(Debug, Clone)]
pub struct ColumnDef {
    /// Column name
    pub(crate) name: String,
    /// DuckLake type string (e.g., "varchar", "int64", "decimal(10,2)")
    pub(crate) ducklake_type: String,
    /// Whether this column allows NULL values
    pub(crate) is_nullable: bool,
}

impl ColumnDef {
    /// Returns the column name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the DuckLake type string.
    pub fn ducklake_type(&self) -> &str {
        &self.ducklake_type
    }

    /// Returns whether this column allows NULL values.
    pub fn is_nullable(&self) -> bool {
        self.is_nullable
    }

    /// Create a new column definition.
    ///
    /// Validates that `ducklake_type` is a recognized DuckLake type string by converting
    /// it to an Arrow DataType. Returns an error if the type is invalid or unsupported.
    pub fn new(
        name: impl Into<String>,
        ducklake_type: impl Into<String>,
        is_nullable: bool,
    ) -> Result<Self> {
        let name = name.into();
        validate_name(&name, "Column")?;
        let ducklake_type = ducklake_type.into();
        // Validate the type string by attempting to convert it to an Arrow type.
        // We discard the result; we only care that the conversion succeeds.
        ducklake_to_arrow_type(&ducklake_type)?;
        Ok(Self {
            name,
            ducklake_type,
            is_nullable,
        })
    }

    /// Create a column definition from an Arrow DataType.
    ///
    /// This is a convenience constructor that converts the Arrow type to a DuckLake type string.
    /// The resulting DuckLake type is guaranteed to be valid since it was derived from a known
    /// Arrow type.
    pub fn from_arrow(
        name: impl Into<String>,
        data_type: &DataType,
        is_nullable: bool,
    ) -> Result<Self> {
        let name = name.into();
        validate_name(&name, "Column")?;
        let ducklake_type = arrow_to_ducklake_type(data_type)?;
        // We use direct struct construction here since the ducklake_type was just
        // produced by arrow_to_ducklake_type, so it is guaranteed to be valid.
        Ok(Self {
            name,
            ducklake_type,
            is_nullable,
        })
    }
}

/// Whether `proposed` is a *schema change* relative to `existing` — i.e. whether a
/// commit carrying it is DDL (and must bump `schema_version`) rather than a pure
/// data write (which carries `schema_version` forward).
///
/// `existing` is the table's currently-live columns as `(name, ducklake_type,
/// nullable)`, ordered by `column_order`; `proposed` is the incoming schema. The
/// comparison is positional, mirroring upstream's per-column diff.
///
/// A same-name type difference is NOT treated as a change when it's the benign
/// Append-vs-promote race: a data write that PASSED the begin-time type reject (its
/// staged type matched the type AT BEGIN) but whose column a concurrent promote
/// widened before this commit. The staged (narrower) type losslessly widens to the
/// committed type and is served via cast-on-read, so it must NOT bump
/// `schema_version`. We accept canonical-equal OR staged-widens-to-committed;
/// anything else is real DDL. (Not `types_compatible`, which would also accept
/// committed-widens-to-staged and wrongly classify the race as DDL.)
///
/// Shared by the SQLite and Postgres writers so the DDL/DML classification can't
/// drift between backends.
pub(crate) fn columns_differ(existing: &[(String, String, bool)], proposed: &[ColumnDef]) -> bool {
    if existing.len() != proposed.len() {
        return true;
    }
    for ((ex_name, ex_type, ex_nullable), new_col) in existing.iter().zip(proposed.iter()) {
        if ex_name != &new_col.name {
            return true;
        }
        let same_type = crate::types::types_equal_canonical(ex_type, &new_col.ducklake_type)
            || crate::types::is_promotable(&new_col.ducklake_type, ex_type);
        if !same_type {
            return true;
        }
        if *ex_nullable != new_col.is_nullable {
            return true;
        }
    }
    false
}

/// Information about a data file to register in the catalog.
///
/// This struct contains the metadata needed to register a Parquet file in the DuckLake catalog.
#[derive(Debug, Clone)]
pub struct DataFileInfo {
    /// Path to the file (relative to table path or absolute)
    pub path: String,
    /// Whether the path is relative to the table's path
    pub path_is_relative: bool,
    /// Size of the file in bytes
    pub file_size_bytes: i64,
    /// Size of the Parquet footer in bytes (optimization hint for reads)
    pub footer_size: Option<i64>,
    /// Number of records in the file
    pub record_count: i64,
}

impl DataFileInfo {
    /// Create a new data file info with relative path.
    ///
    /// # Panics
    ///
    /// Panics if `record_count` is negative. Record counts originate from
    /// `RecordBatch::num_rows()` (always non-negative), so a negative value
    /// indicates a programming error.
    pub fn new(path: impl Into<String>, file_size_bytes: i64, record_count: i64) -> Self {
        assert!(
            record_count >= 0,
            "record_count must be non-negative, got {}",
            record_count
        );
        Self {
            path: path.into(),
            path_is_relative: true,
            file_size_bytes,
            footer_size: None,
            record_count,
        }
    }

    /// Set the footer size for read optimization.
    pub fn with_footer_size(mut self, footer_size: i64) -> Self {
        self.footer_size = Some(footer_size);
        self
    }

    /// Mark this file as having an absolute path.
    pub fn with_absolute_path(mut self) -> Self {
        self.path_is_relative = false;
        self
    }
}

/// A positional delete file to register via [`MetadataWriter::set_delete_file`].
/// Mirrors [`DataFileInfo`]; the parquet has the standard `(file_path, pos)`
/// schema. Must be cumulative for its data file (all still-deleted positions),
/// since at most one delete file is live per data file at a time.
#[derive(Debug, Clone)]
pub struct DeleteFileInfo {
    /// Path to the delete file (relative to the table path, or absolute).
    pub path: String,
    /// Whether the path is relative to the table's path.
    pub path_is_relative: bool,
    /// Size of the delete file in bytes.
    pub file_size_bytes: i64,
    /// Size of the Parquet footer in bytes (read optimization hint).
    pub footer_size: Option<i64>,
    /// Number of deleted positions in this file.
    pub delete_count: i64,
}

impl DeleteFileInfo {
    /// Create a new delete-file info with a relative path.
    ///
    /// # Panics
    /// Panics if `delete_count` is negative.
    pub fn new(path: impl Into<String>, file_size_bytes: i64, delete_count: i64) -> Self {
        assert!(
            delete_count >= 0,
            "delete_count must be non-negative, got {delete_count}"
        );
        Self {
            path: path.into(),
            path_is_relative: true,
            file_size_bytes,
            footer_size: None,
            delete_count,
        }
    }

    /// Set the footer size for read optimization.
    pub fn with_footer_size(mut self, footer_size: i64) -> Self {
        self.footer_size = Some(footer_size);
        self
    }

    /// Mark this delete file as having an absolute path.
    pub fn with_absolute_path(mut self) -> Self {
        self.path_is_relative = false;
        self
    }
}

/// One data file's positional delete, applied as part of a combined
/// [`MetadataWriter::register_data_file_with_deletes`] commit. Supersedes the
/// live delete file for `data_file_id` with `delete` (which must be cumulative),
/// guarded by the same compare-and-swap as
/// [`MetadataWriter::set_delete_file`].
#[derive(Debug, Clone)]
pub struct DeleteFileEntry {
    /// The existing data file whose rows are being (partly) deleted.
    pub data_file_id: i64,
    /// The live delete file the caller resolved against for `data_file_id`
    /// (compare-and-swap guard), or `None` if none was live.
    pub expected_prev_delete_file: Option<i64>,
    /// The new cumulative delete file (all still-deleted positions for the file).
    pub delete: DeleteFileInfo,
}

/// Validate the `deletes` of a
/// [`MetadataWriter::register_data_file_with_deletes`] call before any work.
///
/// Positional deletes require [`WriteMode::Append`]: a `Replace` retires the
/// very data files the deletes target, so the fence could never find them and
/// the commit would abort with a misleading "retired by a concurrent write"
/// error. Each entry must also target a distinct data file — positions are
/// cumulative per file, so the caller unions them into one entry per file;
/// duplicates would otherwise abort on the second entry's compare-and-swap.
pub(crate) fn validate_delete_entries(mode: WriteMode, deletes: &[DeleteFileEntry]) -> Result<()> {
    if deletes.is_empty() {
        return Ok(());
    }
    if mode == WriteMode::Replace {
        return Err(DuckLakeError::InvalidConfig(
            "register_data_file_with_deletes: positional deletes require WriteMode::Append; \
             Replace retires the data files the deletes target"
                .to_string(),
        ));
    }
    let mut seen = std::collections::HashSet::with_capacity(deletes.len());
    for entry in deletes {
        if !seen.insert(entry.data_file_id) {
            return Err(DuckLakeError::InvalidConfig(format!(
                "register_data_file_with_deletes: duplicate delete entry for data file {}; \
                 each entry must target a distinct data file",
                entry.data_file_id
            )));
        }
    }
    Ok(())
}

/// Result of a write operation.
#[derive(Debug)]
pub struct WriteResult {
    /// Snapshot ID of the write operation
    pub snapshot_id: i64,
    /// Table ID (may be newly created)
    pub table_id: i64,
    /// Schema ID (may be newly created)
    pub schema_id: i64,
    /// Number of files written
    pub files_written: usize,
    /// Total records written
    pub records_written: i64,
}

/// The ids actually committed by `register_data_file` / `publish_snapshot`.
///
/// On multicatalog Postgres all metadata is written at the commit point, so the
/// committed `snapshot_id` is assigned there and the `schema_id`/`table_id` are
/// the real committed ids (which may differ from the begin-time reservations in
/// [`WriteSetupResult`] if a concurrent writer created the schema/table first).
/// Callers should use these for the authoritative result rather than the
/// begin-time reservations.
#[derive(Debug, Clone, Copy)]
pub struct CommitIds {
    /// Snapshot id assigned at commit (the new catalog head for this write).
    pub snapshot_id: i64,
    /// Committed schema id.
    pub schema_id: i64,
    /// Committed table id.
    pub table_id: i64,
}

/// Result of a transactional write setup operation.
#[derive(Debug)]
pub struct WriteSetupResult {
    /// Snapshot ID created for this write
    pub snapshot_id: i64,
    /// The catalog head observed at `begin_write_transaction` (the base for
    /// `Replace` conflict detection), threaded back to the commit step. If a
    /// concurrent writer committed a newer generation of the table since this base
    /// — i.e. any data file or column with `begin_snapshot`/`end_snapshot > base`
    /// — the commit aborts with [`crate::DuckLakeError::Conflict`]. Both backends
    /// now share this model: snapshot ids are assigned at *commit* (single-catalog
    /// SQLite `MAX(snapshot_id)+1`; multicatalog Postgres a plain `IDENTITY`
    /// insert), so per-catalog id order == commit order and the scalar
    /// `> base` test is exact.
    pub base_snapshot_id: i64,
    /// Schema ID (may be newly created)
    pub schema_id: i64,
    /// Table ID (may be newly created)
    pub table_id: i64,
    /// Column IDs in order
    pub column_ids: Vec<i64>,
}

/// Trait for writing metadata to DuckLake catalogs.
///
/// Implementations must be thread-safe (`Send + Sync`).
pub trait MetadataWriter: Send + Sync + std::fmt::Debug {
    /// Create a new snapshot and return its ID.
    fn create_snapshot(&self) -> Result<i64>;

    /// Get or create a schema, returning `(schema_id, was_created)`.
    fn get_or_create_schema(
        &self,
        name: &str,
        path: Option<&str>,
        snapshot_id: i64,
    ) -> Result<(i64, bool)>;

    /// Get or create a table, returning `(table_id, was_created)`.
    fn get_or_create_table(
        &self,
        schema_id: i64,
        name: &str,
        path: Option<&str>,
        snapshot_id: i64,
    ) -> Result<(i64, bool)>;

    /// Set columns for a table, returning assigned column IDs.
    /// Ends existing columns using end_snapshot pattern for time travel.
    fn set_columns(
        &self,
        table_id: i64,
        columns: &[ColumnDef],
        snapshot_id: i64,
    ) -> Result<Vec<i64>>;

    /// Promote (widen) an existing column's type in place — DuckLake schema
    /// evolution, distinct from a data write (which *rejects* type changes; see
    /// [`MetadataWriter::begin_write_transaction`]).
    ///
    /// In a single transaction: validate the change is a lossless widening
    /// ([`crate::types::is_promotable`]), create a new snapshot, retire the live
    /// `ducklake_column` row (set its `end_snapshot`), and insert a new row with
    /// the **same `column_id`**, the new `column_type`, and `begin_snapshot` = the
    /// new snapshot. The stable `column_id` keeps Parquet field-ids valid, so
    /// files written before and after both resolve to their snapshot's version
    /// (the read path casts old narrow values up to the widened type). Returns the
    /// new snapshot id.
    ///
    /// Default impl errors — backends that don't support promotion yet return
    /// [`crate::DuckLakeError::InvalidConfig`].
    fn promote_column_type(
        &self,
        _table_id: i64,
        _column_name: &str,
        _new_ducklake_type: &str,
    ) -> Result<i64> {
        Err(DuckLakeError::InvalidConfig(
            "promote_column_type is not supported on this metadata backend".to_string(),
        ))
    }

    /// Register a new data file and publish its snapshot as the catalog head,
    /// atomically. For `Replace`, retires the prior generation in the same
    /// transaction. Returns the committed snapshot id: assigned at this commit
    /// for SQLite (so it may differ from `WriteSetupResult::snapshot_id` under
    /// concurrency), reserved at begin for Postgres.
    ///
    /// `columns` / `column_ids` describe the snapshot's column generation (in
    /// `column_order`, ids matching `WriteSetupResult::column_ids`). Backends
    /// that finalize columns in `begin_write_transaction` (multicatalog
    /// Postgres) ignore them; single-catalog backends (SQLite) defer the
    /// column generation to this commit and use them to insert the column rows.
    ///
    /// `base_snapshot` is the catalog head observed at `begin_write_transaction`
    /// ([`WriteSetupResult::base_snapshot_id`]). For `Replace`, the commit aborts
    /// with [`crate::DuckLakeError::Conflict`] if any data file of the table has
    /// `begin_snapshot` or `end_snapshot` greater than `base_snapshot` — i.e.
    /// another writer published a newer generation since this write began — so
    /// concurrent replaces never silently union or clobber each other.
    ///
    /// `schema_name` / `table_name` identify the target. Multicatalog Postgres
    /// writes ALL metadata at this commit (the schema/table get-or-create happens
    /// here, keyed by these names) so it needs them; single-catalog SQLite already
    /// created the schema/table at begin and ignores them.
    /// Returns the [`CommitIds`] actually committed (the snapshot id assigned at
    /// commit, and the real schema/table ids — which may differ from the
    /// begin-time reservations if a concurrent writer created them first).
    #[allow(clippy::too_many_arguments)]
    fn register_data_file(
        &self,
        table_id: i64,
        schema_name: &str,
        table_name: &str,
        snapshot_id: i64,
        file: &DataFileInfo,
        mode: WriteMode,
        base_snapshot: i64,
        columns: &[ColumnDef],
        column_ids: &[i64],
    ) -> Result<CommitIds>;

    /// Register a positional delete file for a single data file, superseding any
    /// prior live delete file for it (at most one is live per data file).
    ///
    /// In one transaction, abort with [`crate::DuckLakeError::Conflict`] if either
    /// the target `data_file_id` is no longer the live data file (a concurrent
    /// Replace/compaction retired it since `base_snapshot`, invalidating the
    /// resolved positions) or the currently-live delete file for it no longer
    /// matches `expected_prev_delete_file` (a concurrent delete on the same file
    /// won the race). A concurrent *append* that only adds other files does NOT
    /// conflict — it never moves this file's rows. Otherwise end the prior delete
    /// file and insert `delete`, which must carry the cumulative position set.
    ///
    /// Default: unsupported; backends override it.
    #[allow(clippy::too_many_arguments)]
    fn set_delete_file(
        &self,
        _table_id: i64,
        _schema_name: &str,
        _table_name: &str,
        _snapshot_id: i64,
        _data_file_id: i64,
        _expected_prev_delete_file: Option<i64>,
        _base_snapshot: i64,
        _delete: &DeleteFileInfo,
    ) -> Result<CommitIds> {
        Err(DuckLakeError::InvalidConfig(
            "set_delete_file is not supported by this metadata writer".to_string(),
        ))
    }

    /// Atomically register one new data file AND apply positional deletes to
    /// existing data files, in a SINGLE snapshot — the primitive behind an
    /// update/upsert (supersede rows and insert their new versions in one commit).
    ///
    /// In one transaction: allocate one snapshot; insert `file` and advance the
    /// stats/row-lineage counter exactly as
    /// [`register_data_file`](MetadataWriter::register_data_file); then, for each
    /// [`DeleteFileEntry`], apply the same target-file fence + compare-and-swap +
    /// retire-prior + insert-cumulative as
    /// [`set_delete_file`](MetadataWriter::set_delete_file), all stamped with that
    /// one snapshot. Advance the catalog head LAST, so the append and every delete
    /// become visible together — never a half-applied intermediate state. Aborts
    /// with [`crate::DuckLakeError::Conflict`] on the first entry whose target
    /// data file was retired since `base_snapshot`, or whose live delete file no
    /// longer matches `expected_prev_delete_file`.
    ///
    /// `deletes` may be empty (equivalent to
    /// [`register_data_file`](MetadataWriter::register_data_file)); each entry
    /// must target a distinct `data_file_id`.
    ///
    /// Default: unsupported; backends override it.
    #[allow(clippy::too_many_arguments)]
    fn register_data_file_with_deletes(
        &self,
        _table_id: i64,
        _schema_name: &str,
        _table_name: &str,
        _snapshot_id: i64,
        _file: &DataFileInfo,
        _deletes: &[DeleteFileEntry],
        _mode: WriteMode,
        _base_snapshot: i64,
        _columns: &[ColumnDef],
        _column_ids: &[i64],
    ) -> Result<CommitIds> {
        Err(DuckLakeError::InvalidConfig(
            "register_data_file_with_deletes is not supported by this metadata writer".to_string(),
        ))
    }

    /// Apply positional deletes to one or more existing data files in a SINGLE
    /// new snapshot, WITHOUT appending any data file — the commit behind a SQL
    /// `DELETE ... WHERE`. This is [`register_data_file_with_deletes`] minus the
    /// append: it does not require (and never writes) a new data file, so a pure
    /// delete does not create a spurious empty data file.
    ///
    /// [`register_data_file_with_deletes`]: MetadataWriter::register_data_file_with_deletes
    ///
    /// In one transaction: allocate one snapshot (carrying `schema_version`
    /// forward — a delete is not DDL); then for each [`DeleteFileEntry`] apply the
    /// same target-file fence + compare-and-swap + retire-prior + insert-cumulative
    /// as [`set_delete_file`](MetadataWriter::set_delete_file), all stamped with
    /// that one snapshot; advance the catalog head LAST, so every file's delete
    /// becomes visible together (atomic multi-file DELETE) — never a half-applied
    /// state. Aborts with [`crate::DuckLakeError::Conflict`] on the first entry
    /// whose target data file was retired since `base_snapshot`, or whose live
    /// delete file no longer matches `expected_prev_delete_file`.
    ///
    /// `deletes` must be non-empty (an empty delete is a caller-side no-op that
    /// must NOT reach here — it would create an empty snapshot); each entry must
    /// target a distinct `data_file_id`.
    ///
    /// Default: unsupported; backends override it.
    fn commit_positional_deletes(
        &self,
        _table_id: i64,
        _schema_name: &str,
        _table_name: &str,
        _base_snapshot: i64,
        _deletes: &[DeleteFileEntry],
    ) -> Result<CommitIds> {
        Err(DuckLakeError::InvalidConfig(
            "positional DELETE is not supported on this metadata backend".to_string(),
        ))
    }

    /// Truncate a table: end EVERY live data file (and its live delete file) in
    /// one new snapshot and zero the visible stat totals, WITHOUT rewriting any
    /// data — the commit behind a SQL `DELETE FROM t` with no `WHERE`. Mirrors the
    /// file-ending drop_table performs, but leaves the table's schema live.
    /// `next_row_id` is deliberately preserved (rowids stay monotonic).
    ///
    /// Returns the number of rows removed (the table's live row count immediately
    /// before the truncate: gross `record_count` minus still-live delete counts),
    /// which the SQL `DELETE` reports as rows affected. The count is computed
    /// inside the same transaction that ends the files, so it is consistent with
    /// what was removed.
    ///
    /// Default: unsupported; backends override it.
    fn commit_truncate(
        &self,
        _table_id: i64,
        _schema_name: &str,
        _table_name: &str,
        _base_snapshot: i64,
    ) -> Result<u64> {
        Err(DuckLakeError::InvalidConfig(
            "DELETE (truncate) is not supported on this metadata backend".to_string(),
        ))
    }

    /// Publish a write's snapshot as the catalog head with no data file (CREATE
    /// TABLE, zero-row Replace). For `Replace`, retires the prior generation.
    /// See [`MetadataWriter::register_data_file`] for the parameters.
    ///
    /// Default no-op. Backends that advance the head in
    /// `begin_write_transaction` could rely on it, but both shipped backends
    /// override: multicatalog Postgres writes the snapshot/schema/table/column
    /// metadata and inserts the `ducklake_catalog_snapshot_map` head row, and
    /// SQLite (which defers the `ducklake_snapshot` row insert out of
    /// `begin_write_transaction`) inserts the snapshot row + column generation here.
    #[allow(clippy::too_many_arguments)]
    fn publish_snapshot(
        &self,
        _table_id: i64,
        _schema_name: &str,
        _table_name: &str,
        _snapshot_id: i64,
        _mode: WriteMode,
        _base_snapshot: i64,
        _columns: &[ColumnDef],
        _column_ids: &[i64],
    ) -> Result<CommitIds> {
        Ok(CommitIds {
            snapshot_id: _snapshot_id,
            schema_id: 0,
            table_id: _table_id,
        })
    }

    /// End all existing data files for a table. Returns count of files ended.
    fn end_table_files(&self, table_id: i64, snapshot_id: i64) -> Result<u64>;

    /// Get the data path from catalog metadata.
    fn get_data_path(&self) -> Result<String>;

    /// Set the data path in catalog metadata.
    fn set_data_path(&self, path: &str) -> Result<()>;

    /// Initialize DuckLake schema tables if they don't exist.
    fn initialize_schema(&self) -> Result<()>;

    /// Atomically set up catalog metadata for a write operation.
    /// Creates snapshot, schema, table, columns in a single transaction.
    /// If mode is `WriteMode::Replace`, ends existing data files.
    fn begin_write_transaction(
        &self,
        schema_name: &str,
        table_name: &str,
        columns: &[ColumnDef],
        mode: WriteMode,
    ) -> Result<WriteSetupResult>;

    /// The catalog id this writer is scoped to, when the backend has a notion
    /// of catalogs (multicatalog Postgres). Single-catalog backends (SQLite)
    /// return `None`, which keeps `DuckLakeTableWriter` from inserting a
    /// per-catalog directory segment into newly-written file paths and so
    /// preserves today's `{data_path}/{schema}/{table}/…` layout.
    fn catalog_id(&self) -> Option<i64> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DuckLakeError;

    #[test]
    fn test_column_def_new() {
        let col = ColumnDef::new("test_col", "int32", true).unwrap();
        assert_eq!(col.name, "test_col");
        assert_eq!(col.ducklake_type, "int32");
        assert!(col.is_nullable);
    }

    #[test]
    fn test_column_def_new_valid_types() {
        // Various valid type strings should be accepted
        assert!(ColumnDef::new("a", "int32", true).is_ok());
        assert!(ColumnDef::new("b", "varchar", false).is_ok());
        assert!(ColumnDef::new("c", "boolean", true).is_ok());
        assert!(ColumnDef::new("d", "float64", true).is_ok());
        assert!(ColumnDef::new("e", "decimal(10,2)", true).is_ok());
        assert!(ColumnDef::new("f", "timestamp", true).is_ok());
        assert!(ColumnDef::new("g", "date", true).is_ok());
        assert!(ColumnDef::new("h", "bigint", true).is_ok());
        assert!(ColumnDef::new("i", "text", true).is_ok());
    }

    #[test]
    fn test_column_def_new_invalid_type_rejected() {
        let result = ColumnDef::new("col", "not_a_type", true);
        assert!(result.is_err());
        match result {
            Err(DuckLakeError::UnsupportedType(msg)) => {
                assert_eq!(msg, "not_a_type");
            },
            other => panic!("Expected UnsupportedType error, got {:?}", other),
        }
    }

    #[test]
    fn test_column_def_new_empty_type_rejected() {
        let result = ColumnDef::new("col", "", true);
        assert!(result.is_err());
        match result {
            Err(DuckLakeError::UnsupportedType(_)) => {},
            other => panic!("Expected UnsupportedType error, got {:?}", other),
        }
    }

    #[test]
    fn test_column_def_from_arrow() {
        let col = ColumnDef::from_arrow("id", &DataType::Int64, false).unwrap();
        assert_eq!(col.name, "id");
        assert_eq!(col.ducklake_type, "int64");
        assert!(!col.is_nullable);
    }

    #[test]
    fn test_data_file_info_new() {
        let file = DataFileInfo::new("test.parquet", 1024, 100);
        assert_eq!(file.path, "test.parquet");
        assert!(file.path_is_relative);
        assert_eq!(file.file_size_bytes, 1024);
        assert_eq!(file.record_count, 100);
        assert!(file.footer_size.is_none());
    }

    #[test]
    fn test_data_file_info_with_footer_size() {
        let file = DataFileInfo::new("test.parquet", 1024, 100).with_footer_size(256);
        assert_eq!(file.footer_size, Some(256));
    }

    #[test]
    fn test_data_file_info_with_absolute_path() {
        let file = DataFileInfo::new("/absolute/path.parquet", 1024, 100).with_absolute_path();
        assert!(!file.path_is_relative);
    }

    #[test]
    fn test_column_def_empty_name_rejected() {
        let result = ColumnDef::new("", "int32", true);
        assert!(result.is_err());
        match result {
            Err(DuckLakeError::InvalidConfig(msg)) => {
                assert!(msg.contains("empty"), "Expected 'empty' in: {msg}");
            },
            other => panic!("Expected InvalidConfig, got {:?}", other),
        }
    }

    #[test]
    fn test_column_def_control_char_name_rejected() {
        let result = ColumnDef::new("col\0name", "int32", true);
        assert!(result.is_err());
        match result {
            Err(DuckLakeError::InvalidConfig(msg)) => {
                assert!(
                    msg.contains("control character"),
                    "Expected 'control character' in: {msg}"
                );
            },
            other => panic!("Expected InvalidConfig, got {:?}", other),
        }
    }

    #[test]
    fn test_column_def_from_arrow_empty_name_rejected() {
        let result = ColumnDef::from_arrow("", &DataType::Int64, false);
        assert!(result.is_err());
        match result {
            Err(DuckLakeError::InvalidConfig(msg)) => {
                assert!(msg.contains("empty"), "Expected 'empty' in: {msg}");
            },
            other => panic!("Expected InvalidConfig, got {:?}", other),
        }
    }

    #[test]
    fn test_column_def_from_arrow_control_char_rejected() {
        let result = ColumnDef::from_arrow("col\nnewline", &DataType::Int64, false);
        assert!(result.is_err());
        match result {
            Err(DuckLakeError::InvalidConfig(msg)) => {
                assert!(
                    msg.contains("control character"),
                    "Expected 'control character' in: {msg}"
                );
            },
            other => panic!("Expected InvalidConfig, got {:?}", other),
        }
    }

    #[test]
    fn test_validate_name_valid() {
        assert!(validate_name("users", "Table").is_ok());
        assert!(validate_name("my_column", "Column").is_ok());
        assert!(validate_name("Schema123", "Schema").is_ok());
        assert!(validate_name("a", "Column").is_ok());
    }

    #[test]
    fn test_validate_name_empty() {
        let result = validate_name("", "Table");
        assert!(result.is_err());
        let result = validate_name("   ", "Table");
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_name_control_chars() {
        // Null byte
        assert!(validate_name("col\0", "Column").is_err());
        // Newline
        assert!(validate_name("col\n", "Column").is_err());
        // Tab
        assert!(validate_name("col\t", "Column").is_err());
        // DEL (0x7F)
        assert!(validate_name("col\x7F", "Column").is_err());
    }

    #[test]
    fn test_validate_name_length_limit() {
        // Exactly at limit should succeed
        let at_limit = "a".repeat(MAX_NAME_LENGTH);
        assert!(validate_name(&at_limit, "Table").is_ok());

        // One over should fail
        let over_limit = "a".repeat(MAX_NAME_LENGTH + 1);
        assert!(validate_name(&over_limit, "Table").is_err());
    }

    #[test]
    fn test_column_def_long_name_rejected() {
        let long_name = "x".repeat(MAX_NAME_LENGTH + 1);
        let result = ColumnDef::new(long_name, "int32", true);
        assert!(result.is_err());
        match result {
            Err(DuckLakeError::InvalidConfig(msg)) => {
                assert!(
                    msg.contains("exceeds maximum length"),
                    "Expected 'exceeds maximum length' in: {msg}"
                );
            },
            other => panic!("Expected InvalidConfig, got {:?}", other),
        }
    }

    #[test]
    fn test_data_file_info_zero_record_count() {
        let file = DataFileInfo::new("empty.parquet", 0, 0);
        assert_eq!(file.record_count, 0);
    }

    #[test]
    #[should_panic(expected = "record_count must be non-negative")]
    fn test_data_file_info_negative_record_count_panics() {
        DataFileInfo::new("test.parquet", 1024, -1);
    }
}
