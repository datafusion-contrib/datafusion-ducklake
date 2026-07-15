use crate::Result;

// SQL queries for DuckLake catalog tables
// These queries are database-agnostic and work with DuckDB, SQLite, PostgreSQL, MySQL
pub const SQL_GET_LATEST_SNAPSHOT: &str =
    "SELECT COALESCE(MAX(snapshot_id), 0) FROM ducklake_snapshot";

pub const SQL_LIST_SNAPSHOTS: &str = "SELECT snapshot_id, CAST(snapshot_time AS VARCHAR) as timestamp FROM ducklake_snapshot ORDER BY snapshot_id";

pub const SQL_LIST_SCHEMAS: &str =
    "SELECT schema_id, schema_name, path, path_is_relative FROM ducklake_schema
     WHERE ? >= begin_snapshot AND (? < end_snapshot OR end_snapshot IS NULL)";

pub const SQL_LIST_TABLES: &str =
    "SELECT table_id, table_name, path, path_is_relative FROM ducklake_table
     WHERE schema_id = ?
       AND ? >= begin_snapshot
       AND (? < end_snapshot OR end_snapshot IS NULL)";

pub const SQL_GET_TABLE_COLUMNS: &str =
    "SELECT column_id, column_name, column_type, nulls_allowed, parent_column
     FROM ducklake_column
     WHERE table_id = ?
       AND ? >= begin_snapshot
       AND (? < end_snapshot OR end_snapshot IS NULL)
     ORDER BY column_order";

pub const SQL_GET_DATA_FILES: &str = "
    SELECT
        data.data_file_id,
        data.path AS data_file_path,
        data.path_is_relative AS data_path_is_relative,
        data.file_size_bytes AS data_file_size,
        data.footer_size AS data_footer_size,
        data.encryption_key AS data_encryption_key,
        data.row_id_start AS data_row_id_start,
        data.record_count AS data_record_count,
        del.delete_file_id,
        del.path AS delete_file_path,
        del.path_is_relative AS delete_path_is_relative,
        del.file_size_bytes AS delete_file_size,
        del.footer_size AS delete_footer_size,
        del.encryption_key AS delete_encryption_key,
        del.delete_count
    FROM ducklake_data_file AS data
    LEFT JOIN ducklake_delete_file AS del
        ON data.data_file_id = del.data_file_id
        AND del.table_id = ?
        AND ? >= del.begin_snapshot
        AND (? < del.end_snapshot OR del.end_snapshot IS NULL)
    WHERE data.table_id = ?
      AND ? >= data.begin_snapshot
      AND (? < data.end_snapshot OR data.end_snapshot IS NULL)";

pub const SQL_GET_TABLE_STATS: &str =
    "SELECT record_count, file_size_bytes FROM ducklake_table_stats WHERE table_id = ?";

pub const SQL_GET_TABLE_COLUMN_STATS: &str = "
    SELECT column_id, contains_null, min_value, max_value
    FROM ducklake_table_column_stats
    WHERE table_id = ?";

pub const SQL_GET_FILE_COLUMN_STATS: &str = "
    SELECT
        stats.data_file_id,
        stats.column_id,
        stats.column_size_bytes,
        stats.value_count,
        stats.null_count,
        stats.min_value,
        stats.max_value
    FROM ducklake_file_column_stats AS stats
    INNER JOIN ducklake_data_file AS data
        ON data.data_file_id = stats.data_file_id
        AND data.table_id = stats.table_id
    WHERE stats.table_id = ?
      AND ? >= data.begin_snapshot
      AND (? < data.end_snapshot OR data.end_snapshot IS NULL)";

pub const SQL_GET_DATA_PATH: &str =
    "SELECT value FROM ducklake_metadata WHERE key = 'data_path' AND scope IS NULL";

pub const SQL_GET_SCHEMA_BY_NAME: &str =
    "SELECT schema_id, schema_name, path, path_is_relative FROM ducklake_schema
     WHERE schema_name = ?
       AND ? >= begin_snapshot
       AND (? < end_snapshot OR end_snapshot IS NULL)";

pub const SQL_GET_TABLE_BY_NAME: &str =
    "SELECT table_id, table_name, path, path_is_relative FROM ducklake_table
     WHERE schema_id = ?
       AND table_name = ?
       AND ? >= begin_snapshot
       AND (? < end_snapshot OR end_snapshot IS NULL)";

pub const SQL_TABLE_EXISTS: &str = "SELECT EXISTS(
       SELECT 1 FROM ducklake_table
       WHERE schema_id = ?
         AND table_name = ?
         AND ? >= begin_snapshot
         AND (? < end_snapshot OR end_snapshot IS NULL)
     )";

// Queries for table_changes (CDC) - files added/removed between snapshots

pub const SQL_GET_DATA_FILES_ADDED_BETWEEN_SNAPSHOTS: &str = "
    SELECT
        data.begin_snapshot,
        data.path,
        data.path_is_relative,
        data.file_size_bytes,
        data.footer_size,
        data.encryption_key
    FROM ducklake_data_file AS data
    WHERE data.table_id = ?
      AND data.begin_snapshot > ?
      AND data.begin_snapshot <= ?
    ORDER BY data.begin_snapshot";

pub const SQL_GET_DELETE_FILES_ADDED_BETWEEN_SNAPSHOTS: &str = "
WITH params AS (
    SELECT
        ? AS table_identifier,
        ? AS start_snapshot,
        ? AS finish_snapshot
),

current_delete AS (
    SELECT
        df.data_file_id,
        df.begin_snapshot,
        df.path,
        df.path_is_relative,
        df.file_size_bytes,
        df.footer_size,
        df.encryption_key
    FROM ducklake_delete_file df
    CROSS JOIN params p
    WHERE df.table_id = p.table_identifier
      AND df.begin_snapshot BETWEEN p.start_snapshot AND p.finish_snapshot
),

all_deletes AS (
    SELECT
        df.data_file_id,
        df.begin_snapshot,
        df.path,
        df.path_is_relative,
        df.file_size_bytes,
        df.footer_size,
        df.encryption_key
    FROM ducklake_delete_file df
    CROSS JOIN params p
    WHERE df.table_id = p.table_identifier
)

SELECT
    data.path,
    data.path_is_relative,
    data.file_size_bytes,
    data.footer_size,
    data.row_id_start,
    data.record_count,
    data.mapping_id,

    cd.path AS current_delete_path,
    cd.path_is_relative AS current_delete_path_is_relative,
    cd.file_size_bytes AS current_delete_file_size_bytes,
    cd.footer_size AS current_delete_footer_size,

    pd.path AS previous_delete_path,
    pd.path_is_relative AS previous_delete_path_is_relative,
    pd.file_size_bytes AS previous_delete_file_size_bytes,
    pd.footer_size AS previous_delete_footer_size,

    cd.begin_snapshot
FROM current_delete cd
JOIN ducklake_data_file data
  ON data.data_file_id = cd.data_file_id
LEFT JOIN LATERAL (
    SELECT path, path_is_relative, file_size_bytes, footer_size
    FROM all_deletes ad
    WHERE ad.data_file_id = cd.data_file_id
      AND ad.begin_snapshot < cd.begin_snapshot
    ORDER BY ad.begin_snapshot DESC
    LIMIT 1
) pd ON true
CROSS JOIN params p
WHERE data.table_id = p.table_identifier

UNION ALL

SELECT
    data.path,
    data.path_is_relative,
    data.file_size_bytes,
    data.footer_size,
    data.row_id_start,
    data.record_count,
    data.mapping_id,

    NULL,
    NULL,
    NULL,
    NULL,

    pd.path,
    pd.path_is_relative,
    pd.file_size_bytes,
    pd.footer_size,

    data.end_snapshot
FROM ducklake_data_file data
LEFT JOIN LATERAL (
    SELECT path, path_is_relative, file_size_bytes, footer_size
    FROM all_deletes ad
    WHERE ad.data_file_id = data.data_file_id
      AND ad.begin_snapshot < data.end_snapshot
    ORDER BY ad.begin_snapshot DESC
    LIMIT 1
) pd ON true
CROSS JOIN params p
WHERE data.table_id = p.table_identifier
  AND data.end_snapshot BETWEEN p.start_snapshot AND p.finish_snapshot;
";

// Bulk queries for information_schema (avoids N+1 query problem)

pub const SQL_LIST_ALL_TABLES: &str = "
    SELECT
        s.schema_name,
        t.table_id,
        t.table_name,
        t.path,
        t.path_is_relative
    FROM ducklake_schema s
    JOIN ducklake_table t ON s.schema_id = t.schema_id
    WHERE ? >= s.begin_snapshot
      AND (? < s.end_snapshot OR s.end_snapshot IS NULL)
      AND ? >= t.begin_snapshot
      AND (? < t.end_snapshot OR t.end_snapshot IS NULL)
    ORDER BY s.schema_name, t.table_name";

pub const SQL_LIST_ALL_COLUMNS: &str = "
    SELECT
        s.schema_name,
        t.table_name,
        c.column_id,
        c.column_name,
        c.column_type,
        c.nulls_allowed,
        c.parent_column
    FROM ducklake_schema s
    JOIN ducklake_table t ON s.schema_id = t.schema_id
    JOIN ducklake_column c ON t.table_id = c.table_id
    WHERE ? >= s.begin_snapshot
      AND (? < s.end_snapshot OR s.end_snapshot IS NULL)
      AND ? >= t.begin_snapshot
      AND (? < t.end_snapshot OR t.end_snapshot IS NULL)
      AND ? >= c.begin_snapshot
      AND (? < c.end_snapshot OR c.end_snapshot IS NULL)
    ORDER BY s.schema_name, t.table_name, c.column_order";

pub const SQL_LIST_ALL_FILES: &str = "
    SELECT
        s.schema_name,
        t.table_name,
        data.data_file_id,
        data.path AS data_file_path,
        data.path_is_relative AS data_path_is_relative,
        data.file_size_bytes AS data_file_size,
        data.footer_size AS data_footer_size,
        data.encryption_key AS data_encryption_key,
        del.delete_file_id,
        del.path AS delete_file_path,
        del.path_is_relative AS delete_path_is_relative,
        del.file_size_bytes AS delete_file_size,
        del.footer_size AS delete_footer_size,
        del.encryption_key AS delete_encryption_key,
        del.delete_count
    FROM ducklake_schema s
    JOIN ducklake_table t ON s.schema_id = t.schema_id
    JOIN ducklake_data_file data ON t.table_id = data.table_id
    LEFT JOIN ducklake_delete_file del
        ON data.data_file_id = del.data_file_id
        AND del.table_id = t.table_id
        AND ? >= del.begin_snapshot
        AND (? < del.end_snapshot OR del.end_snapshot IS NULL)
    WHERE ? >= s.begin_snapshot
      AND (? < s.end_snapshot OR s.end_snapshot IS NULL)
      AND ? >= t.begin_snapshot
      AND (? < t.end_snapshot OR t.end_snapshot IS NULL)
      AND ? >= data.begin_snapshot
      AND (? < data.end_snapshot OR data.end_snapshot IS NULL)
    ORDER BY s.schema_name, t.table_name, data.path";

/// Metadata for a snapshot in the DuckLake catalog
#[derive(Debug, Clone)]
pub struct SnapshotMetadata {
    /// Unique identifier for this snapshot
    pub snapshot_id: i64,
    /// Timestamp when the snapshot was created (optional)
    pub timestamp: Option<String>,
}

/// Metadata for a schema in the DuckLake catalog
#[derive(Debug, Clone)]
pub struct SchemaMetadata {
    /// Unique identifier for this schema in the catalog
    pub schema_id: i64,
    /// Name of the schema as it appears in SQL queries
    pub schema_name: String,
    /// Path to the schema's data directory (may be relative or absolute)
    pub path: String,
    /// Whether the path is relative to the catalog's data_path
    pub path_is_relative: bool,
}

/// Metadata for a table in the DuckLake catalog
#[derive(Debug, Clone)]
pub struct TableMetadata {
    /// Unique identifier for this table in the catalog
    pub table_id: i64,
    /// Name of the table as it appears in SQL queries
    pub table_name: String,
    /// Path to the table's data directory (may be relative or absolute)
    pub path: String,
    /// Whether the path is relative to the schema's path
    pub path_is_relative: bool,
}

/// Table metadata with its schema name (for bulk queries)
#[derive(Debug, Clone)]
pub struct TableWithSchema {
    /// Name of the schema this table belongs to
    pub schema_name: String,
    /// Table metadata
    pub table: TableMetadata,
}

/// Column metadata with its schema and table names (for bulk queries)
#[derive(Debug, Clone)]
pub struct ColumnWithTable {
    /// Name of the schema this column's table belongs to
    pub schema_name: String,
    /// Name of the table this column belongs to
    pub table_name: String,
    /// Column metadata
    pub column: DuckLakeTableColumn,
}

/// File metadata with its schema and table names (for bulk queries)
#[derive(Debug, Clone)]
pub struct FileWithTable {
    /// Name of the schema this file's table belongs to
    pub schema_name: String,
    /// Name of the table this file belongs to
    pub table_name: String,
    /// File metadata
    pub file: DuckLakeTableFile,
}

/// Column definition for a DuckLake table
#[derive(Debug, Clone)]
pub struct DuckLakeTableColumn {
    /// Unique identifier for this column in the catalog
    pub column_id: i64,
    /// Name of the column
    pub column_name: String,
    /// DuckLake type string (e.g., "varchar", "int64", "decimal(10,2)")
    pub column_type: String,
    /// Whether this column allows NULL values
    pub is_nullable: bool,
}

impl DuckLakeTableColumn {
    pub fn new(
        column_id: i64,
        column_name: String,
        column_type: String,
        is_nullable: bool,
    ) -> Self {
        Self {
            column_id,
            column_name,
            column_type,
            is_nullable,
        }
    }
}

/// Reconstruct list types from parent-child column rows.
///
/// DuckLake stores list columns as two rows in `ducklake_column`:
/// - Parent row: `column_type = "list"`, `parent_column = NULL`
/// - Child row:  `column_type = "<element_type>"`, `parent_column = <parent_column_id>`
///
/// This function rewrites the parent's `column_type` to `list<element_type>`
/// and removes child rows from the result.
///
/// Only handles `list` parent types. Struct, map, etc. are left unchanged.
pub fn reconstruct_list_columns(
    rows: Vec<(DuckLakeTableColumn, Option<i64>)>,
) -> Vec<DuckLakeTableColumn> {
    use std::collections::HashMap;

    // Index: column_id -> position in rows
    let id_to_index: HashMap<i64, usize> = rows
        .iter()
        .enumerate()
        .map(|(i, (col, _))| (col.column_id, i))
        .collect();

    // Separate into columns and parent_column arrays
    let mut columns: Vec<DuckLakeTableColumn> = Vec::with_capacity(rows.len());
    let mut parent_columns: Vec<Option<i64>> = Vec::with_capacity(rows.len());
    for (col, parent) in rows {
        columns.push(col);
        parent_columns.push(parent);
    }

    // Find children of list parents and rewrite parent types
    let mut skip: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for (i, parent_id) in parent_columns.iter().enumerate() {
        if let Some(pid) = parent_id
            && let Some(&parent_idx) = id_to_index.get(pid)
            && columns[parent_idx].column_type == "list"
        {
            columns[parent_idx].column_type = format!("list<{}>", columns[i].column_type);
            skip.insert(i);
        }
    }

    // Return only top-level columns (not children)
    columns
        .into_iter()
        .enumerate()
        .filter(|(i, _)| !skip.contains(i))
        .map(|(_, col)| col)
        .collect()
}

/// Same as [`reconstruct_list_columns`] but for [`ColumnWithTable`] rows.
pub fn reconstruct_list_columns_with_table(
    rows: Vec<(ColumnWithTable, Option<i64>)>,
) -> Vec<ColumnWithTable> {
    use std::collections::HashMap;

    let id_to_index: HashMap<i64, usize> = rows
        .iter()
        .enumerate()
        .map(|(i, (cwt, _))| (cwt.column.column_id, i))
        .collect();

    let mut entries: Vec<ColumnWithTable> = Vec::with_capacity(rows.len());
    let mut parent_columns: Vec<Option<i64>> = Vec::with_capacity(rows.len());
    for (cwt, parent) in rows {
        entries.push(cwt);
        parent_columns.push(parent);
    }

    let mut skip: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for (i, parent_id) in parent_columns.iter().enumerate() {
        if let Some(pid) = parent_id
            && let Some(&parent_idx) = id_to_index.get(pid)
            && entries[parent_idx].column.column_type == "list"
        {
            entries[parent_idx].column.column_type =
                format!("list<{}>", entries[i].column.column_type);
            skip.insert(i);
        }
    }

    entries
        .into_iter()
        .enumerate()
        .filter(|(i, _)| !skip.contains(i))
        .map(|(_, e)| e)
        .collect()
}

/// Metadata for a data file or delete file in DuckLake
#[derive(Debug, Clone)]
pub struct DuckLakeFileData {
    /// Path to the file (may be relative or absolute)
    pub path: String,
    /// Whether the path is relative to the table's path
    pub path_is_relative: bool,
    /// Encryption key for the file (used for Parquet Modular Encryption)
    pub encryption_key: Option<String>,
    /// Size of the file in bytes
    pub file_size_bytes: i64,
    /// Size of the Parquet footer in bytes (optional optimization hint)
    pub footer_size: Option<i64>,
}

impl DuckLakeFileData {
    pub fn new(path: String, path_is_relative: bool, file_size_bytes: i64) -> Self {
        Self {
            path,
            path_is_relative,
            encryption_key: None,
            file_size_bytes,
            footer_size: None,
        }
    }
}

/// Represents a data file and its associated delete file (if any) for a DuckLake table
#[derive(Debug, Clone)]
pub struct DuckLakeTableFile {
    /// Catalog `data_file_id` — the identity a positional-delete write targets
    /// (`MetadataWriter::set_delete_file`). Needed by the mutation path; the read
    /// path ignores it.
    pub data_file_id: i64,
    /// Metadata for the data file
    pub file: DuckLakeFileData,
    /// Catalog `delete_file_id` of the currently-live delete file for this data
    /// file, or `None` if none is live. The compare-and-swap `expected_prev`
    /// when superseding it with a cumulative delete file.
    pub delete_file_id: Option<i64>,
    /// Optional associated delete file containing deleted row positions
    pub delete_file: Option<DuckLakeFileData>,
    /// Starting row ID for this file. Combined with each row's position in the
    /// file, this gives a globally unique `rowid` (DuckLake row lineage).
    /// `None` for files where the metadata column is unset (e.g. older catalogs).
    pub row_id_start: Option<i64>,
    /// Snapshot ID when this file was created (reserved for future use)
    pub snapshot_id: Option<i64>,
    /// The file's own `begin_snapshot` (origin snapshot). Distinct from
    /// `snapshot_id` (the QUERIED snapshot). Compaction uses it as each row's
    /// origin for the merged partial file's per-row `_ducklake_internal_snapshot_id`
    /// column and `partial_max`. `None` when the provider does not surface it.
    pub begin_snapshot: Option<i64>,
    /// The catalog `schema_version` in effect at `begin_snapshot`. Compaction
    /// merges only files sharing one schema version (never across a DDL
    /// boundary). `None` when the provider does not surface it.
    pub schema_version: Option<i64>,
    /// `partial_max` from `ducklake_data_file`: for a merged **partial data
    /// file**, the maximum origin snapshot id among its rows (their per-row
    /// origin is embedded in the `_ducklake_internal_snapshot_id` column).
    /// `None` for ordinary files. When reading at a snapshot below this, the
    /// read path drops the file's rows whose embedded origin exceeds the read
    /// snapshot (per-row time-travel visibility).
    pub partial_max: Option<i64>,
    /// Total rows in this file (`record_count` from the catalog), before any
    /// delete files are applied. Used for synthetic `rowid` generation.
    pub max_row_count: Option<i64>,
    /// Number of rows removed by the associated `delete_file` visible at the
    /// queried snapshot (`delete_count` from `ducklake_delete_file`). `None`
    /// when there is no visible delete file. Net live rows for this file are
    /// `max_row_count - delete_count`.
    pub delete_count: Option<i64>,
}

/// Statistics cached for a table in the DuckLake catalog.
///
/// `table` and `columns` describe the current table generation. Per-file
/// statistics are filtered to the data files visible at the requested
/// snapshot so callers can also use them for time-travel scans.
#[derive(Debug, Clone, Default)]
pub struct DuckLakeStatistics {
    pub table: Option<DuckLakeTableStatistics>,
    pub columns: Vec<DuckLakeTableColumnStatistics>,
    pub files: Vec<DuckLakeFileColumnStatistics>,
}

/// A row from `ducklake_table_stats`.
#[derive(Debug, Clone)]
pub struct DuckLakeTableStatistics {
    pub record_count: Option<i64>,
    pub file_size_bytes: Option<i64>,
}

/// A row from `ducklake_table_column_stats` containing the fields DataFusion
/// can represent in [`datafusion::common::ColumnStatistics`].
#[derive(Debug, Clone)]
pub struct DuckLakeTableColumnStatistics {
    pub column_id: i64,
    pub contains_null: Option<bool>,
    pub min_value: Option<String>,
    pub max_value: Option<String>,
}

/// A row from `ducklake_file_column_stats` containing the fields DataFusion
/// can represent in [`datafusion::common::ColumnStatistics`].
#[derive(Debug, Clone)]
pub struct DuckLakeFileColumnStatistics {
    pub data_file_id: i64,
    pub column_id: i64,
    pub column_size_bytes: Option<i64>,
    pub value_count: Option<i64>,
    pub null_count: Option<i64>,
    pub min_value: Option<String>,
    pub max_value: Option<String>,
}

impl DuckLakeTableFile {
    pub fn new(file: DuckLakeFileData) -> Self {
        Self {
            // A bare file with no catalog context: `data_file_id` is unset (0)
            // and there is no associated delete file. Not for the mutation path,
            // which reads files (with real ids) via `get_table_files_for_select`.
            data_file_id: 0,
            file,
            delete_file_id: None,
            delete_file: None,
            row_id_start: None,
            snapshot_id: None,
            begin_snapshot: None,
            schema_version: None,
            partial_max: None,
            max_row_count: None,
            delete_count: None,
        }
    }
}

// Change tracking structures for table_changes (CDC) functionality

#[derive(Debug, Clone)]
pub struct DataFileChange {
    pub begin_snapshot: i64,
    pub path: String,
    pub path_is_relative: bool,
    pub file_size_bytes: i64,
    pub footer_size: Option<i64>,
    pub encryption_key: Option<String>,
}

#[derive(Debug, Clone)]
pub struct DeleteFileChange {
    /* -------- Data file being affected -------- */
    pub data_file_path: String,
    pub data_file_path_is_relative: bool,
    pub data_file_size_bytes: i64,
    pub data_file_footer_size: i64,
    pub data_row_id_start: i64,
    pub data_record_count: i64,
    pub data_mapping_id: Option<i64>,

    /* -------- Delete file added at this snapshot (None for full file deletes) -------- */
    pub current_delete_path: Option<String>,
    pub current_delete_path_is_relative: Option<bool>,
    pub current_delete_file_size_bytes: Option<i64>,
    pub current_delete_footer_size: Option<i64>,

    /* -------- Delete file replaced (if any) -------- */
    pub previous_delete_path: Option<String>,
    pub previous_delete_path_is_relative: Option<bool>,
    pub previous_delete_file_size_bytes: Option<i64>,
    pub previous_delete_footer_size: Option<i64>,

    /* -------- Snapshot where change occurred -------- */
    pub snapshot_id: i64,
}

pub trait MetadataProvider: Send + Sync + std::fmt::Debug {
    /// Get the current snapshot ID (dynamic, not cached)
    fn get_current_snapshot(&self) -> Result<i64>;

    /// Get the data path from catalog metadata (not snapshot-dependent)
    fn get_data_path(&self) -> Result<String>;

    /// List all snapshots in the catalog
    fn list_snapshots(&self) -> Result<Vec<SnapshotMetadata>>;

    /// List schemas for a specific snapshot
    fn list_schemas(&self, snapshot_id: i64) -> Result<Vec<SchemaMetadata>>;

    /// List tables for a specific snapshot
    fn list_tables(&self, schema_id: i64, snapshot_id: i64) -> Result<Vec<TableMetadata>>;

    /// Get table structure (columns) visible at `snapshot_id`. Columns are
    /// snapshot-scoped (`snapshot_id >= begin_snapshot AND (snapshot_id <
    /// end_snapshot OR end_snapshot IS NULL)`), matching upstream DuckLake and
    /// the catalog's own snapshot-scoped `list_tables`/`list_schemas`. This is
    /// required for correct reads under schema evolution and to hide
    /// uncommitted/dormant column generations on the multicatalog write path.
    fn get_table_structure(
        &self,
        table_id: i64,
        snapshot_id: i64,
    ) -> Result<Vec<DuckLakeTableColumn>>;

    /// Get table files for a specific snapshot
    fn get_table_files_for_select(
        &self,
        table_id: i64,
        snapshot_id: i64,
    ) -> Result<Vec<DuckLakeTableFile>>;
    //     todo: support select with file pruning

    /// Load table-, column-, and file-level statistics from the DuckLake
    /// catalog. Implementations should return unknown statistics when the
    /// optional statistics tables do not exist (for compatibility with older
    /// catalogs) rather than making the table unreadable.
    fn get_table_statistics(
        &self,
        _table_id: i64,
        _snapshot_id: i64,
    ) -> Result<DuckLakeStatistics> {
        Ok(DuckLakeStatistics::default())
    }

    /// Read rows that DuckDB's *data-inlining* optimization stored directly in
    /// the catalog database (not in Parquet), for `table_id` visible at
    /// `snapshot_id`, materialized as Arrow batches in `columns`' physical
    /// schema (same column order as [`get_table_structure`](Self::get_table_structure)).
    ///
    /// DuckLake inlines small INSERTs into per-`(table, schema_version)` catalog
    /// tables `ducklake_inlined_data_<id>_<sv>(row_id, begin_snapshot,
    /// end_snapshot, <data cols>)`, registered in `ducklake_inlined_data_tables`.
    /// A row is visible when `snapshot_id >= begin_snapshot AND (end_snapshot IS
    /// NULL OR snapshot_id < end_snapshot)`; a deleted inlined row simply carries
    /// `end_snapshot`, so the predicate handles inlined-row deletes.
    ///
    /// The default returns empty, so catalogs without inlined data (no
    /// `ducklake_inlined_data_tables`) and backends that don't implement this are
    /// unaffected. Implementations that return empty when the registry is absent
    /// keep older catalogs readable.
    ///
    /// NOTE: this surfaces inlined INSERT rows only. Inlined deletions of rows
    /// that live in Parquet data files (`ducklake_inlined_delete_<id>`) are a
    /// separate mechanism and are not yet applied here.
    fn get_inlined_data(
        &self,
        _table_id: i64,
        _snapshot_id: i64,
        _columns: &[DuckLakeTableColumn],
    ) -> Result<Vec<arrow::record_batch::RecordBatch>> {
        Ok(Vec::new())
    }

    /// Net number of live rows in a table at a snapshot, accounting for delete
    /// files: `SUM(record_count) - SUM(delete_count)` over the files visible at
    /// `snapshot_id`. This matches a `SELECT COUNT(*)` against the table at that
    /// snapshot without scanning any data — the counts come from catalog
    /// metadata.
    ///
    /// The default implementation derives the count from
    /// [`get_table_files_for_select`](Self::get_table_files_for_select), so it
    /// is computed from exactly the file set a scan would read and stays correct
    /// across deletes, replacements, and compaction. A file whose
    /// `max_row_count` is unset (foreign catalogs that omit `record_count`)
    /// contributes 0 and cannot be counted from metadata alone.
    fn get_table_row_count(&self, table_id: i64, snapshot_id: i64) -> Result<u64> {
        let files = self.get_table_files_for_select(table_id, snapshot_id)?;
        let net: i64 = files
            .iter()
            .map(|f| f.max_row_count.unwrap_or(0) - f.delete_count.unwrap_or(0))
            .sum();
        Ok(net.max(0) as u64)
    }

    // Dynamic lookup methods for on-demand metadata retrieval

    /// Get schema by name for a specific snapshot
    fn get_schema_by_name(&self, name: &str, snapshot_id: i64) -> Result<Option<SchemaMetadata>>;

    /// Get table by name for a specific snapshot
    fn get_table_by_name(
        &self,
        schema_id: i64,
        name: &str,
        snapshot_id: i64,
    ) -> Result<Option<TableMetadata>>;

    /// Check if table exists for a specific snapshot
    fn table_exists(&self, schema_id: i64, name: &str, snapshot_id: i64) -> Result<bool>;

    // Bulk query methods for information_schema

    /// List all tables across all schemas for a snapshot
    fn list_all_tables(&self, snapshot_id: i64) -> Result<Vec<TableWithSchema>>;

    /// List all columns across all tables for a snapshot
    fn list_all_columns(&self, snapshot_id: i64) -> Result<Vec<ColumnWithTable>>;

    /// List all files across all tables for a snapshot
    fn list_all_files(&self, snapshot_id: i64) -> Result<Vec<FileWithTable>>;

    // Change tracking methods for table_changes (CDC) functionality

    /// Get data files added between two snapshots (exclusive start, inclusive end)
    /// Returns files where begin_snapshot > start_snapshot AND begin_snapshot <= end_snapshot
    /// These represent INSERT changes - new rows added to the table
    fn get_data_files_added_between_snapshots(
        &self,
        table_id: i64,
        start_snapshot: i64,
        end_snapshot: i64,
    ) -> Result<Vec<DataFileChange>>;

    /// Get delete files added between two snapshots (exclusive start, inclusive end)
    /// Returns delete files where begin_snapshot > start_snapshot AND begin_snapshot <= end_snapshot
    /// These represent DELETE changes - rows removed from the table
    fn get_delete_files_added_between_snapshots(
        &self,
        table_id: i64,
        start_snapshot: i64,
        end_snapshot: i64,
    ) -> Result<Vec<DeleteFileChange>>;
}

#[cfg(any(feature = "metadata-postgres", feature = "metadata-mysql", feature = "metadata-sqlite"))]
/// Helper function to bridge async sqlx operations to sync MetadataProvider trait
pub(crate) fn block_on<F, T>(f: F) -> T
where
    F: std::future::Future<Output = T>,
{
    tokio::task::block_in_place(|| tokio::runtime::Handle::current().block_on(f))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reconstruct_list_columns_basic() {
        let rows = vec![
            (
                DuckLakeTableColumn::new(1, "id".into(), "int64".into(), false),
                None,
            ),
            (
                DuckLakeTableColumn::new(6, "vector".into(), "list".into(), true),
                None,
            ),
            (
                DuckLakeTableColumn::new(7, "element".into(), "float64".into(), true),
                Some(6),
            ),
        ];

        let result = reconstruct_list_columns(rows);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].column_name, "id");
        assert_eq!(result[0].column_type, "int64");
        assert_eq!(result[1].column_name, "vector");
        assert_eq!(result[1].column_type, "list<float64>");
    }

    #[test]
    fn test_reconstruct_list_columns_no_lists() {
        let rows = vec![
            (
                DuckLakeTableColumn::new(1, "id".into(), "int64".into(), false),
                None,
            ),
            (
                DuckLakeTableColumn::new(2, "name".into(), "varchar".into(), true),
                None,
            ),
        ];

        let result = reconstruct_list_columns(rows);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].column_type, "int64");
        assert_eq!(result[1].column_type, "varchar");
    }

    #[test]
    fn test_reconstruct_list_columns_struct_parent_unchanged() {
        // Struct parents should NOT be rewritten — child stays in result
        let rows = vec![
            (
                DuckLakeTableColumn::new(1, "data".into(), "struct".into(), true),
                None,
            ),
            (
                DuckLakeTableColumn::new(2, "field_a".into(), "int32".into(), true),
                Some(1),
            ),
        ];

        let result = reconstruct_list_columns(rows);
        assert_eq!(result.len(), 2); // both remain
        assert_eq!(result[0].column_type, "struct"); // unchanged
    }

    #[test]
    fn test_reconstruct_list_columns_multiple_lists() {
        let rows = vec![
            (
                DuckLakeTableColumn::new(1, "tags".into(), "list".into(), true),
                None,
            ),
            (
                DuckLakeTableColumn::new(2, "element".into(), "varchar".into(), true),
                Some(1),
            ),
            (
                DuckLakeTableColumn::new(3, "scores".into(), "list".into(), true),
                None,
            ),
            (
                DuckLakeTableColumn::new(4, "element".into(), "float64".into(), true),
                Some(3),
            ),
        ];

        let result = reconstruct_list_columns(rows);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].column_type, "list<varchar>");
        assert_eq!(result[1].column_type, "list<float64>");
    }

    #[test]
    fn test_reconstruct_list_columns_with_table_basic() {
        let rows = vec![
            (
                ColumnWithTable {
                    schema_name: "main".into(),
                    table_name: "t".into(),
                    column: DuckLakeTableColumn::new(6, "vector".into(), "list".into(), true),
                },
                None,
            ),
            (
                ColumnWithTable {
                    schema_name: "main".into(),
                    table_name: "t".into(),
                    column: DuckLakeTableColumn::new(7, "element".into(), "float64".into(), true),
                },
                Some(6),
            ),
        ];

        let result = reconstruct_list_columns_with_table(rows);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].column.column_type, "list<float64>");
    }
}
