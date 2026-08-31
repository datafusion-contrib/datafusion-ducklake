use crate::Result;
use crate::types::{arrow_to_ducklake_type, ducklake_to_arrow_type};
use arrow::datatypes::{DataType, Field, SchemaRef};
use arrow::record_batch::RecordBatch;
use datafusion::common::ScalarValue;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

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

pub const SQL_LIST_VIEWS: &str =
    "SELECT view_id, schema_id, begin_snapshot, view_name, dialect, sql, column_aliases FROM ducklake_view
     WHERE schema_id = ?
       AND ? >= begin_snapshot
       AND (? < end_snapshot OR end_snapshot IS NULL)";

pub const SQL_GET_VIEW_BY_NAME: &str =
    "SELECT view_id, schema_id, begin_snapshot, view_name, dialect, sql, column_aliases FROM ducklake_view
     WHERE schema_id = ?
       AND view_name = ?
       AND ? >= begin_snapshot
       AND (? < end_snapshot OR end_snapshot IS NULL)";

pub const SQL_GET_TABLE_COLUMNS: &str =
    "SELECT column_id, column_name, column_type, nulls_allowed, parent_column,
            initial_default, default_value, default_value_type, default_value_dialect
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

/// Read a table's active partition spec (partition_info joined to its key
/// columns) visible at a snapshot. `?` placeholders (duckdb/sqlite/mysql style):
/// `table_id, snapshot_id, snapshot_id`. Postgres builds a `$N` variant inline.
pub const SQL_GET_PARTITION_SPEC: &str = "
    SELECT pi.partition_id, pc.partition_key_index, pc.column_id, pc.transform
    FROM ducklake_partition_info AS pi
    JOIN ducklake_partition_column AS pc
        ON pc.partition_id = pi.partition_id AND pc.table_id = pi.table_id
    WHERE pi.table_id = ?
      AND ? >= pi.begin_snapshot
      AND (? < pi.end_snapshot OR pi.end_snapshot IS NULL)
    ORDER BY pc.partition_key_index";

/// Read a table's active sort spec (sort_info joined to its expressions) visible at
/// a snapshot. `?` placeholders (duckdb/sqlite/mysql style): `table_id, snapshot_id,
/// snapshot_id`. Postgres builds a `$N` variant inline.
pub const SQL_GET_SORT_SPEC: &str = "
    SELECT si.sort_id, se.sort_key_index, se.expression, se.dialect,
           se.sort_direction, se.null_order
    FROM ducklake_sort_info AS si
    JOIN ducklake_sort_expression AS se
        ON se.sort_id = si.sort_id AND se.table_id = si.table_id
    WHERE si.table_id = ?
      AND ? >= si.begin_snapshot
      AND (? < si.end_snapshot OR si.end_snapshot IS NULL)
    ORDER BY se.sort_key_index";

/// Read per-file partition values for a `data_file_id` range (the planning page
/// window). `?` placeholders: `table_id, after_data_file_id (exclusive),
/// last_data_file_id (inclusive)`. Rows for files outside the page are harmless
/// (grouped by id and matched only to files actually in the page).
pub const SQL_GET_FILE_PARTITION_VALUES: &str = "
    SELECT data_file_id, partition_key_index, partition_value
    FROM ducklake_file_partition_value
    WHERE table_id = ?
      AND data_file_id > ?
      AND data_file_id <= ?";

pub const SQL_GET_TABLE_STATS: &str =
    "SELECT record_count, file_size_bytes FROM ducklake_table_stats WHERE table_id = ?";

pub const SQL_GET_TABLE_COLUMN_STATS: &str = "
    SELECT column_id, contains_null, min_value, max_value, contains_nan
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
        stats.max_value,
        stats.contains_nan
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
        data.encryption_key,
        data.row_id_start,
        data.partial_max
    FROM ducklake_data_file AS data
    WHERE data.table_id = $1
      AND data.begin_snapshot <= $3
      AND (data.begin_snapshot >= $2
           OR (data.partial_max IS NOT NULL AND data.partial_max >= $2))
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
      AND df.begin_snapshot <= p.finish_snapshot
      AND (df.begin_snapshot >= p.start_snapshot
           OR (df.partial_max IS NOT NULL AND df.partial_max >= p.start_snapshot))
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
  AND data.end_snapshot >= p.start_snapshot
  AND data.end_snapshot <= p.finish_snapshot;
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

pub const SQL_LIST_ALL_VIEWS: &str = "
    SELECT s.schema_name, v.view_id, v.schema_id, v.begin_snapshot, v.view_name, v.dialect, v.sql,
           v.column_aliases
    FROM ducklake_schema s
    JOIN ducklake_view v ON s.schema_id = v.schema_id
    WHERE ? >= s.begin_snapshot
      AND (? < s.end_snapshot OR s.end_snapshot IS NULL)
      AND ? >= v.begin_snapshot
      AND (? < v.end_snapshot OR v.end_snapshot IS NULL)
    ORDER BY s.schema_name, v.view_name";

pub const SQL_LIST_ALL_COLUMNS: &str = "
    SELECT
        s.schema_name,
        t.table_name,
        c.column_id,
        c.column_name,
        c.column_type,
        c.nulls_allowed,
        c.parent_column,
        c.initial_default,
        c.default_value,
        c.default_value_type,
        c.default_value_dialect
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

pub(crate) fn parse_snapshot_timestamp(raw: &str) -> Option<chrono::NaiveDateTime> {
    let mut timestamp = raw.trim();
    for suffix in ["Z", " UTC", "+00:00", "+00"] {
        if let Some(stripped) = timestamp.strip_suffix(suffix) {
            timestamp = stripped.trim();
            break;
        }
    }
    for format in ["%Y-%m-%d %H:%M:%S%.f", "%Y-%m-%dT%H:%M:%S%.f"] {
        if let Ok(parsed) = chrono::NaiveDateTime::parse_from_str(timestamp, format) {
            return Some(parsed);
        }
    }
    chrono::NaiveDate::parse_from_str(timestamp, "%Y-%m-%d")
        .ok()
        .and_then(|date| date.and_hms_opt(0, 0, 0))
}

pub(crate) fn resolve_snapshot_at_or_before(
    provider: &dyn MetadataProvider,
    timestamp: chrono::NaiveDateTime,
) -> Result<i64> {
    resolve_snapshot_at(provider, timestamp, false)
}

pub(crate) fn resolve_snapshot_at_or_after(
    provider: &dyn MetadataProvider,
    timestamp: chrono::NaiveDateTime,
) -> Result<i64> {
    resolve_snapshot_at(provider, timestamp, true)
}

fn resolve_snapshot_at(
    provider: &dyn MetadataProvider,
    timestamp: chrono::NaiveDateTime,
    at_or_after: bool,
) -> Result<i64> {
    let mut best: Option<(chrono::NaiveDateTime, i64)> = None;
    for snapshot in provider.list_snapshots()? {
        let Some(candidate_time) = snapshot
            .timestamp
            .as_deref()
            .and_then(parse_snapshot_timestamp)
        else {
            continue;
        };
        if (at_or_after && candidate_time < timestamp)
            || (!at_or_after && candidate_time > timestamp)
        {
            continue;
        }
        let replace = match best {
            None => true,
            Some((best_time, best_id)) if at_or_after => {
                candidate_time < best_time
                    || (candidate_time == best_time && snapshot.snapshot_id < best_id)
            },
            Some((best_time, best_id)) => {
                candidate_time > best_time
                    || (candidate_time == best_time && snapshot.snapshot_id > best_id)
            },
        };
        if replace {
            best = Some((candidate_time, snapshot.snapshot_id));
        }
    }
    best.map(|(_, snapshot_id)| snapshot_id).ok_or_else(|| {
        crate::error::DuckLakeError::InvalidSnapshot(format!(
            "No snapshot found {} timestamp {timestamp}",
            if at_or_after {
                "at or after"
            } else {
                "at or before"
            }
        ))
    })
}

pub(crate) fn require_snapshot(provider: &dyn MetadataProvider, snapshot_id: i64) -> Result<i64> {
    if provider
        .list_snapshots()?
        .iter()
        .any(|snapshot| snapshot.snapshot_id == snapshot_id)
    {
        Ok(snapshot_id)
    } else {
        Err(crate::error::DuckLakeError::InvalidSnapshot(format!(
            "Snapshot {snapshot_id} does not exist"
        )))
    }
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

/// Metadata for a view in the DuckLake catalog.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ViewMetadata {
    /// Unique identifier for this view in the catalog.
    pub view_id: i64,
    /// Schema containing the view.
    pub schema_id: i64,
    /// Snapshot that created this view generation.
    pub begin_snapshot: i64,
    /// Name of the view as it appears in SQL queries.
    pub view_name: String,
    /// SQL dialect used by the stored definition.
    pub dialect: String,
    /// Query defining the view.
    pub sql: String,
    /// DuckLake's quoted, comma-separated output aliases.
    pub column_aliases: Option<String>,
}

/// View metadata with its schema name for information-schema queries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ViewWithSchema {
    /// Name of the schema this view belongs to.
    pub schema_name: String,
    /// View metadata.
    pub view: ViewMetadata,
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

/// Column definition for a DuckLake table.
///
/// Built-in providers retain the reconstructed nested Arrow type and descendant
/// IDs so reads and rewrites preserve the recursive catalog representation.
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
    pub(crate) data_type: Option<DataType>,
    pub(crate) nested_column_ids: Vec<i64>,
    /// Value substituted for this column in files that predate it
    pub initial_default: Option<String>,
    /// Value applied when a new write omits this column
    pub default_value: Option<String>,
    /// How to interpret the stored defaults (`literal` or `expression`)
    pub default_value_type: Option<String>,
    /// SQL dialect used to encode the stored defaults
    pub default_value_dialect: Option<String>,
}

/// A positional deletion stored directly in a DuckLake metadata catalog.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DuckLakeInlinedDelete {
    /// Catalog id of the Parquet data file containing the deleted row.
    pub data_file_id: i64,
    /// Zero-based physical row position within the Parquet data file.
    pub row_id: i64,
}

pub(crate) fn is_inlined_data_table(name: &str) -> bool {
    name.starts_with("ducklake_inlined_data_")
        && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

pub(crate) const INLINED_DATA_REMEDIATION: &str =
    "flush inlined data to Parquet (or disable data inlining at write time)";

/// Return the specification-defined physical table name for a table's inlined
/// Parquet-row deletions.
///
/// External metadata providers can use this helper to implement
/// [`MetadataProvider::get_inlined_deletes`] without duplicating identifier
/// validation.
pub fn inlined_delete_table_name(table_id: i64) -> Result<String> {
    if table_id < 0 {
        return Err(crate::DuckLakeError::InvalidConfig(format!(
            "DuckLake table id must be non-negative, was {table_id}"
        )));
    }
    Ok(format!("ducklake_inlined_delete_{table_id}"))
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum InlinedDataBackend {
    Postgres,
    MySql,
}

pub(crate) fn inlined_text_projection(
    backend: InlinedDataBackend,
    column: &DuckLakeTableColumn,
    data_type: &arrow::datatypes::DataType,
    ident: &str,
) -> String {
    use arrow::datatypes::DataType;

    match backend {
        InlinedDataBackend::Postgres => match data_type {
            DataType::FixedSizeBinary(_)
                if column.column_type.trim().eq_ignore_ascii_case("uuid") =>
            {
                format!("CAST({ident} AS TEXT)")
            },
            DataType::Binary
            | DataType::LargeBinary
            | DataType::BinaryView
            | DataType::FixedSizeBinary(_) => format!("encode({ident}, 'hex')"),
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
                if matches!(
                    column.column_type.trim().to_ascii_lowercase().as_str(),
                    "varchar" | "text" | "string" | "json"
                ) =>
            {
                format!("convert_from({ident}, 'UTF8')")
            },
            _ => format!("CAST({ident} AS TEXT)"),
        },
        InlinedDataBackend::MySql => match data_type {
            DataType::FixedSizeBinary(_)
                if column.column_type.trim().eq_ignore_ascii_case("uuid") =>
            {
                format!("CAST({ident} AS CHAR CHARACTER SET utf8mb4)")
            },
            DataType::Binary
            | DataType::LargeBinary
            | DataType::BinaryView
            | DataType::FixedSizeBinary(_) => format!("HEX({ident})"),
            _ => format!("CAST({ident} AS CHAR CHARACTER SET utf8mb4)"),
        },
    }
}

pub(crate) fn build_inlined_batch(
    schema: SchemaRef,
    columns: &[DuckLakeTableColumn],
    rows: &[Vec<ScalarValue>],
) -> Result<RecordBatch> {
    if rows.iter().any(|row| row.len() != columns.len()) {
        return Err(crate::DuckLakeError::Internal(
            "inlined data row does not match the catalog column count".to_string(),
        ));
    }
    let arrays = (0..columns.len())
        .map(|index| ScalarValue::iter_to_array(rows.iter().map(|row| row[index].clone())))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(RecordBatch::try_new(schema, arrays)?)
}

pub(crate) fn parse_inlined_rows(
    schema: SchemaRef,
    columns: &[DuckLakeTableColumn],
    rows: Vec<Vec<Option<String>>>,
) -> Result<RecordBatch> {
    let rows = rows
        .into_iter()
        .map(|row| {
            if row.len() != columns.len() {
                return Err(crate::DuckLakeError::Internal(
                    "inlined data row does not match the catalog column count".to_string(),
                ));
            }
            row.into_iter()
                .enumerate()
                .map(|(index, value)| {
                    let field = schema.field(index);
                    match value {
                        Some(value) => crate::types::parse_ducklake_scalar(
                            &value,
                            field.data_type(),
                        )
                        .ok_or_else(|| {
                            crate::DuckLakeError::Unsupported(format!(
                                "inlined data for column '{}' cannot decode value '{}' as {}; \
                                 {INLINED_DATA_REMEDIATION}",
                                columns[index].column_name,
                                value,
                                field.data_type()
                            ))
                        }),
                        None => Ok(ScalarValue::try_from(field.data_type())?),
                    }
                })
                .collect::<Result<Vec<_>>>()
        })
        .collect::<Result<Vec<_>>>()?;
    build_inlined_batch(schema, columns, &rows)
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
            data_type: None,
            nested_column_ids: Vec::new(),
            initial_default: None,
            default_value: None,
            default_value_type: None,
            default_value_dialect: None,
        }
    }

    pub(crate) fn data_type(&self) -> Result<DataType> {
        match &self.data_type {
            Some(data_type) => Ok(data_type.clone()),
            None => ducklake_to_arrow_type(&self.column_type),
        }
    }

    /// Attach the default metadata stored in `ducklake_column`.
    pub(crate) fn with_defaults(
        mut self,
        initial_default: Option<String>,
        default_value: Option<String>,
        default_value_type: Option<String>,
        default_value_dialect: Option<String>,
    ) -> Self {
        self.initial_default = initial_default;
        // DuckLake stores a missing operational default as the string sentinel `NULL`
        self.default_value = default_value.filter(|value| value != "NULL");
        self.default_value_type = default_value_type;
        self.default_value_dialect = default_value_dialect;
        self
    }
}

/// Reconstruct nested Arrow types from DuckLake's recursive column rows.
pub fn reconstruct_columns(
    rows: Vec<(DuckLakeTableColumn, Option<i64>)>,
) -> Result<Vec<DuckLakeTableColumn>> {
    let id_to_index: HashMap<i64, usize> = rows
        .iter()
        .enumerate()
        .map(|(index, (column, _))| (column.column_id, index))
        .collect();
    if id_to_index.len() != rows.len() {
        return Err(crate::DuckLakeError::InvalidConfig(
            "DuckLake column metadata contains duplicate column ids".into(),
        ));
    }
    let mut children: HashMap<i64, Vec<usize>> = HashMap::new();
    for (index, (_, parent_id)) in rows.iter().enumerate() {
        if let Some(parent_id) = parent_id {
            if !id_to_index.contains_key(parent_id) {
                return Err(crate::DuckLakeError::InvalidConfig(format!(
                    "Nested column {} references missing parent column {parent_id}",
                    rows[index].0.column_id
                )));
            }
            children.entry(*parent_id).or_default().push(index);
        }
    }

    fn build_type(
        index: usize,
        rows: &[(DuckLakeTableColumn, Option<i64>)],
        children: &HashMap<i64, Vec<usize>>,
        visiting: &mut HashSet<i64>,
    ) -> Result<DataType> {
        let column = &rows[index].0;
        if visiting.len() >= crate::types::MAX_NESTED_TYPE_DEPTH {
            return Err(crate::DuckLakeError::InvalidConfig(format!(
                "Nested column metadata exceeds maximum depth {}",
                crate::types::MAX_NESTED_TYPE_DEPTH
            )));
        }
        if !visiting.insert(column.column_id) {
            return Err(crate::DuckLakeError::InvalidConfig(format!(
                "Nested column cycle includes column {}",
                column.column_id
            )));
        }
        let child_indices = children
            .get(&column.column_id)
            .map(Vec::as_slice)
            .unwrap_or_default();
        let data_type = match column.column_type.to_ascii_lowercase().as_str() {
            "list" => {
                let [child_index] = child_indices else {
                    return Err(crate::DuckLakeError::InvalidConfig(format!(
                        "List column '{}' must have exactly one child",
                        column.column_name
                    )));
                };
                let child = &rows[*child_index].0;
                DataType::List(Arc::new(Field::new(
                    "item",
                    build_type(*child_index, rows, children, visiting)?,
                    child.is_nullable,
                )))
            },
            "struct" => {
                let fields = child_indices
                    .iter()
                    .map(|child_index| {
                        let child = &rows[*child_index].0;
                        Ok(Arc::new(Field::new(
                            &child.column_name,
                            build_type(*child_index, rows, children, visiting)?,
                            child.is_nullable,
                        )))
                    })
                    .collect::<Result<Vec<_>>>()?;
                DataType::Struct(fields.into())
            },
            "map" => {
                let [key_index, value_index] = child_indices else {
                    return Err(crate::DuckLakeError::InvalidConfig(format!(
                        "Map column '{}' must have key and value children",
                        column.column_name
                    )));
                };
                let key = &rows[*key_index].0;
                let value = &rows[*value_index].0;
                if key.column_name != "key" || value.column_name != "value" {
                    return Err(crate::DuckLakeError::InvalidConfig(format!(
                        "Map column '{}' children must be named key then value",
                        column.column_name
                    )));
                }
                let entries = DataType::Struct(
                    vec![
                        Arc::new(Field::new(
                            "key",
                            build_type(*key_index, rows, children, visiting)?,
                            false,
                        )),
                        Arc::new(Field::new(
                            "value",
                            build_type(*value_index, rows, children, visiting)?,
                            value.is_nullable,
                        )),
                    ]
                    .into(),
                );
                DataType::Map(Arc::new(Field::new("entries", entries, false)), false)
            },
            _ if child_indices.is_empty() => ducklake_to_arrow_type(&column.column_type)?,
            _ => {
                return Err(crate::DuckLakeError::InvalidConfig(format!(
                    "Non-nested column '{}' has child columns",
                    column.column_name
                )));
            },
        };
        visiting.remove(&column.column_id);
        Ok(data_type)
    }

    let mut result = Vec::new();
    for (index, (column, parent_id)) in rows.iter().enumerate() {
        if parent_id.is_some() {
            continue;
        }
        let mut column = column.clone();
        let data_type = build_type(index, &rows, &children, &mut HashSet::new())?;
        column.column_type = reconstructed_column_type(&column.column_type, &data_type)?;
        column.data_type = Some(data_type);
        fn collect_ids(
            column_id: i64,
            rows: &[(DuckLakeTableColumn, Option<i64>)],
            children: &HashMap<i64, Vec<usize>>,
            ids: &mut Vec<i64>,
        ) {
            if let Some(child_indices) = children.get(&column_id) {
                for child_index in child_indices {
                    let child_id = rows[*child_index].0.column_id;
                    ids.push(child_id);
                    collect_ids(child_id, rows, children, ids);
                }
            }
        }
        collect_ids(
            column.column_id,
            &rows,
            &children,
            &mut column.nested_column_ids,
        );
        result.push(column);
    }
    let reconstructed_count = result
        .iter()
        .map(|column| 1 + column.nested_column_ids.len())
        .sum::<usize>();
    if reconstructed_count != rows.len() {
        return Err(crate::DuckLakeError::InvalidConfig(
            "DuckLake column metadata contains a parent cycle or unreachable nested column".into(),
        ));
    }
    Ok(result)
}

fn reconstructed_column_type(catalog_type: &str, data_type: &DataType) -> Result<String> {
    let normalized = catalog_type.trim().to_ascii_lowercase();
    let preserves_logical_name = matches!(data_type, DataType::Binary)
        || matches!(data_type, DataType::Utf8View)
            && !matches!(normalized.as_str(), "varchar" | "text" | "string");
    if preserves_logical_name {
        Ok(catalog_type.to_string())
    } else {
        arrow_to_ducklake_type(data_type)
    }
}

/// Same as [`reconstruct_columns`] but for [`ColumnWithTable`] rows.
pub fn reconstruct_columns_with_table(
    rows: Vec<(ColumnWithTable, Option<i64>)>,
) -> Result<Vec<ColumnWithTable>> {
    type ColumnRowsByTable = HashMap<(String, String), Vec<(DuckLakeTableColumn, Option<i64>)>>;
    let mut grouped = ColumnRowsByTable::new();
    let mut order = Vec::new();
    for (entry, parent_id) in rows {
        let key = (entry.schema_name, entry.table_name);
        if !grouped.contains_key(&key) {
            order.push(key.clone());
        }
        grouped
            .entry(key)
            .or_default()
            .push((entry.column, parent_id));
    }

    let mut result = Vec::new();
    for (schema_name, table_name) in order {
        let columns = reconstruct_columns(
            grouped
                .remove(&(schema_name.clone(), table_name.clone()))
                .unwrap_or_default(),
        )?;
        result.extend(columns.into_iter().map(|column| ColumnWithTable {
            schema_name: schema_name.clone(),
            table_name: table_name.clone(),
            column,
        }));
    }
    Ok(result)
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
    /// `partition_id` from `ducklake_data_file`: the partition spec generation
    /// this file was written under, or `None` for a file of an unpartitioned
    /// table (or a catalog without partition support). Used to associate the
    /// file with a [`crate::partition::PartitionSpec`] and by the write path's
    /// GC / conflict checks.
    pub partition_id: Option<i64>,
    /// The file's partition values, one per partition key, as `(partition_key_index,
    /// value)` where `value` is the DuckDB-canonical VARCHAR every row in the file
    /// shares for that key (`None` == SQL NULL). Populated only on the read/planning
    /// path (`get_table_file_metadata_page`); left empty by `get_table_files_for_select`
    /// (the write/delete path does not need it). Drives partition pruning.
    pub partition_values: Vec<(i32, Option<String>)>,
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
    /// Tri-state NaN flag for float columns: `Some(false)` = known NaN-free,
    /// `Some(true)` = contains NaN, `None` = unknown (e.g. register-by-reference
    /// loads, where the parquet footer carries no NaN signal). Stored min/max
    /// exclude NaN, and DataFusion orders floats with `total_cmp`, which is
    /// signed on NaN: positive NaN sorts above every value and negative NaN
    /// below every value, `-Infinity` included. Since this flag does not record
    /// the sign, **neither** bound is trustworthy unless it is `Some(false)` —
    /// see `float_bound_is_usable` in `table.rs`.
    pub contains_nan: Option<bool>,
    /// Sum of compressed bytes reported by every visible file for this column.
    pub column_size_bytes: Option<i64>,
    /// Whether the table-wide bounds are exact for the requested snapshot.
    ///
    /// DuckLake's rollup is complete for live data files, but positional
    /// deletes can remove an extremal value without tightening the rollup.
    pub bounds_are_exact: bool,
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
    /// Tri-state NaN flag; see [`DuckLakeTableColumnStatistics::contains_nan`].
    pub contains_nan: Option<bool>,
}

/// One visible data file and its catalog column statistics.
///
/// Table scans consume these records in bounded pages so selective predicates
/// can discard files without first materializing the table's full file set.
#[derive(Debug, Clone)]
pub struct DuckLakeFileMetadata {
    pub file: DuckLakeTableFile,
    pub column_statistics: Vec<DuckLakeFileColumnStatistics>,
}

/// Maximum number of file metadata records retained by the planning iterator.
///
/// This keeps planning memory bounded while avoiding tens of thousands of
/// catalog round trips for million-file tables.
pub const FILE_METADATA_BATCH_SIZE: usize = 4_096;

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
            partition_id: None,
            partition_values: Vec::new(),
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
    /// First rowid assigned to this file (`row_id_start` in the catalog), or
    /// `None` when the catalog does not carry one (e.g. an embedded-rowid
    /// compaction/rewrite output, or an older catalog). Required only to
    /// synthesize a plain insert's rowid (`row_id_start + physical_position`);
    /// files with an embedded rowid do not need it.
    pub row_id_start: Option<i64>,
    /// For a compaction-merged partial file: the maximum origin snapshot id of
    /// its rows (`partial_max` in the catalog). `Some` means the file spans
    /// snapshots `begin_snapshot..=partial_max` and each row's snapshot must be
    /// read from the embedded `_ducklake_internal_snapshot_id` column, with
    /// rows outside the query window filtered out. `None` for ordinary files
    /// (all rows belong to `begin_snapshot`).
    pub partial_max: Option<i64>,
}

#[derive(Debug, Clone)]
pub struct DeleteFileChange {
    /* -------- Data file being affected -------- */
    pub data_file_path: String,
    pub data_file_path_is_relative: bool,
    pub data_file_size_bytes: i64,
    // Nullable in `ducklake_data_file` (mirrors `DataFileChange::footer_size`);
    // a data file registered without a footer-size hint has NULL here.
    pub data_file_footer_size: Option<i64>,
    // Nullable (mirrors `DataFileChange::row_id_start`): NULL for embedded-rowid
    // rewrite/compaction outputs. Required only when a deleted row's rowid must
    // be synthesized as `row_id_start + position` (i.e. the source file has no
    // embedded rowid).
    pub data_row_id_start: Option<i64>,
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

    /// List views visible in a schema at a specific snapshot.
    fn list_views(&self, _schema_id: i64, _snapshot_id: i64) -> Result<Vec<ViewMetadata>> {
        Ok(Vec::new())
    }

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

    /// Read the table's active partition spec visible at `snapshot_id`
    /// (`ducklake_partition_info` + `ducklake_partition_column`), or `None` if
    /// the table is unpartitioned or the catalog has no partition tables. Drives
    /// partition pruning together with each file's `partition_values`.
    ///
    /// The default returns `None`, so external providers and catalogs without
    /// partition support are unaffected. Built-in providers override this.
    fn get_partition_spec(
        &self,
        _table_id: i64,
        _snapshot_id: i64,
    ) -> Result<Option<crate::partition::PartitionSpec>> {
        Ok(None)
    }

    /// Read the table's active sort spec visible at `snapshot_id`
    /// (`ducklake_sort_info` + `ducklake_sort_expression`), or `None` if the table
    /// is unsorted or the catalog has no sort tables. The write path uses this to
    /// order rows within each data file; it does not affect read correctness.
    ///
    /// The default returns `None`, so external providers and catalogs without sort
    /// support are unaffected. Built-in providers override this.
    fn get_sort_spec(
        &self,
        _table_id: i64,
        _snapshot_id: i64,
    ) -> Result<Option<crate::sort::SortSpec>> {
        Ok(None)
    }

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

    /// Load only table- and table-column statistics.
    ///
    /// Built-in providers override this to avoid touching per-file statistics
    /// while constructing a [`crate::DuckLakeTable`]. The default preserves
    /// compatibility for external providers.
    fn get_table_summary_statistics(
        &self,
        table_id: i64,
        snapshot_id: i64,
    ) -> Result<DuckLakeStatistics> {
        let mut statistics = self.get_table_statistics(table_id, snapshot_id)?;
        statistics.files.clear();
        Ok(statistics)
    }

    /// Load one keyset-paginated batch of visible files and their statistics.
    ///
    /// `after_data_file_id` is exclusive. Implementations must return records
    /// ordered by `data_file_id` and no more than `limit` records. Built-in SQL
    /// providers override this with bounded catalog queries; the default keeps
    /// external providers source-compatible, although it is not memory-bounded.
    fn get_table_file_metadata_page(
        &self,
        table_id: i64,
        snapshot_id: i64,
        after_data_file_id: Option<i64>,
        limit: usize,
    ) -> Result<Vec<DuckLakeFileMetadata>> {
        let mut files = self.get_table_files_for_select(table_id, snapshot_id)?;
        files.sort_by_key(|file| file.data_file_id);
        let statistics = self.get_table_statistics(table_id, snapshot_id)?.files;
        let mut by_file: HashMap<i64, Vec<DuckLakeFileColumnStatistics>> = HashMap::new();
        for statistic in statistics {
            by_file
                .entry(statistic.data_file_id)
                .or_default()
                .push(statistic);
        }
        Ok(files
            .into_iter()
            .filter(|file| after_data_file_id.is_none_or(|after| file.data_file_id > after))
            .take(limit)
            .map(|file| DuckLakeFileMetadata {
                column_statistics: by_file.remove(&file.data_file_id).unwrap_or_default(),
                file,
            })
            .collect())
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

    /// Read positional deletions stored in `ducklake_inlined_delete_<table_id>`
    /// that are visible at `snapshot_id`.
    ///
    /// The default keeps legacy catalogs and providers without deletion inlining
    /// source-compatible. Implementations return an empty vector when the
    /// physical table does not exist.
    fn get_inlined_deletes(
        &self,
        _table_id: i64,
        _snapshot_id: i64,
    ) -> Result<Vec<DuckLakeInlinedDelete>> {
        Ok(Vec::new())
    }

    /// Net number of live rows in a table at a snapshot, accounting for delete
    /// files and inlined positions: `SUM(record_count) - SUM(delete_count) -
    /// distinct inlined positions` over the files visible at `snapshot_id`.
    /// This matches a `SELECT COUNT(*)` against the table at that snapshot
    /// without scanning any data — the counts come from catalog metadata.
    ///
    /// The default implementation derives the count from
    /// [`get_table_files_for_select`](Self::get_table_files_for_select), so it
    /// is computed from exactly the file set a scan would read and stays correct
    /// across deletes, replacements, and compaction. A file whose
    /// `max_row_count` is unset (foreign catalogs that omit `record_count`)
    /// contributes 0 and cannot be counted from metadata alone.
    fn get_table_row_count(&self, table_id: i64, snapshot_id: i64) -> Result<u64> {
        let files = self.get_table_files_for_select(table_id, snapshot_id)?;
        let visible_file_ids = files
            .iter()
            .map(|file| file.data_file_id)
            .collect::<std::collections::HashSet<_>>();
        let inlined_count = self
            .get_inlined_deletes(table_id, snapshot_id)?
            .into_iter()
            .filter(|delete| visible_file_ids.contains(&delete.data_file_id))
            .map(|delete| (delete.data_file_id, delete.row_id))
            .collect::<std::collections::HashSet<_>>()
            .len() as i64;
        let net: i64 = files
            .iter()
            .map(|f| f.max_row_count.unwrap_or(0) - f.delete_count.unwrap_or(0))
            .sum::<i64>()
            - inlined_count;
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

    /// Get a visible view by name.
    fn get_view_by_name(
        &self,
        _schema_id: i64,
        _name: &str,
        _snapshot_id: i64,
    ) -> Result<Option<ViewMetadata>> {
        Ok(None)
    }

    /// Check if table exists for a specific snapshot
    fn table_exists(&self, schema_id: i64, name: &str, snapshot_id: i64) -> Result<bool>;

    // Bulk query methods for information_schema

    /// List all tables across all schemas for a snapshot
    fn list_all_tables(&self, snapshot_id: i64) -> Result<Vec<TableWithSchema>>;

    /// List all views across all visible schemas for a snapshot.
    fn list_all_views(&self, snapshot_id: i64) -> Result<Vec<ViewWithSchema>> {
        let mut views = Vec::new();
        for schema in self.list_schemas(snapshot_id)? {
            views.extend(
                self.list_views(schema.schema_id, snapshot_id)?
                    .into_iter()
                    .map(|view| ViewWithSchema {
                        schema_name: schema.schema_name.clone(),
                        view,
                    }),
            );
        }
        Ok(views)
    }

    /// List all columns across all tables for a snapshot
    fn list_all_columns(&self, snapshot_id: i64) -> Result<Vec<ColumnWithTable>>;

    /// List all files across all tables for a snapshot
    fn list_all_files(&self, snapshot_id: i64) -> Result<Vec<FileWithTable>>;

    // Change tracking methods for table_changes (CDC) functionality

    /// Get data files added between two snapshots (inclusive on both ends, matching official DuckLake)
    /// Returns files where begin_snapshot >= start_snapshot AND begin_snapshot <= end_snapshot
    /// These represent INSERT changes - new rows added to the table
    fn get_data_files_added_between_snapshots(
        &self,
        table_id: i64,
        start_snapshot: i64,
        end_snapshot: i64,
    ) -> Result<Vec<DataFileChange>>;

    /// Get delete files added between two snapshots (inclusive on both ends, matching official DuckLake)
    /// Returns delete files where begin_snapshot >= start_snapshot AND begin_snapshot <= end_snapshot
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
    use std::sync::Arc;

    use arrow::datatypes::{DataType, Field, Schema};

    use super::*;

    fn column(name: &str, column_type: &str) -> DuckLakeTableColumn {
        DuckLakeTableColumn::new(1, name.to_string(), column_type.to_string(), true)
    }

    #[test]
    fn inlined_text_projections_match_backend_encodings() {
        assert_eq!(
            inlined_text_projection(
                InlinedDataBackend::Postgres,
                &column("note", "varchar"),
                &DataType::Utf8View,
                "\"note\"",
            ),
            "convert_from(\"note\", 'UTF8')"
        );
        assert_eq!(
            inlined_text_projection(
                InlinedDataBackend::Postgres,
                &column("payload", "blob"),
                &DataType::BinaryView,
                "\"payload\"",
            ),
            "encode(\"payload\", 'hex')"
        );
        assert_eq!(
            inlined_text_projection(
                InlinedDataBackend::Postgres,
                &column("token", "uuid"),
                &DataType::FixedSizeBinary(16),
                "\"token\"",
            ),
            "CAST(\"token\" AS TEXT)"
        );
        assert_eq!(
            inlined_text_projection(
                InlinedDataBackend::MySql,
                &column("payload", "blob"),
                &DataType::BinaryView,
                "`payload`",
            ),
            "HEX(`payload`)"
        );
        assert_eq!(
            inlined_text_projection(
                InlinedDataBackend::MySql,
                &column("note", "varchar"),
                &DataType::Utf8View,
                "`note`",
            ),
            "CAST(`note` AS CHAR CHARACTER SET utf8mb4)"
        );
    }

    #[test]
    fn inlined_rows_reject_unsupported_nested_encoding() {
        let columns = vec![column("items", "list<int32>")];
        let schema = Arc::new(Schema::new(vec![Field::new_list(
            "items",
            Field::new("item", DataType::Int32, true),
            true,
        )]));
        let error = parse_inlined_rows(schema, &columns, vec![vec![Some("[1, 2]".to_string())]])
            .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("inlined data for column 'items' cannot decode value '[1, 2]' as List"),
            "unexpected error: {error}"
        );
        assert!(
            error.to_string().contains(INLINED_DATA_REMEDIATION),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn test_reconstruct_columns_list() {
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

        let result = reconstruct_columns(rows).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].column_name, "id");
        assert_eq!(result[0].column_type, "int64");
        assert_eq!(result[1].column_name, "vector");
        assert_eq!(result[1].column_type, "list<float64>");
    }

    #[test]
    fn test_reconstruct_columns_scalars() {
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

        let result = reconstruct_columns(rows).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].column_type, "int64");
        assert_eq!(result[1].column_type, "varchar");
    }

    #[test]
    fn test_reconstruct_columns_preserves_scalar_catalog_names() {
        let rows = vec![
            (
                DuckLakeTableColumn::new(1, "shape".into(), "geometry".into(), true),
                None,
            ),
            (
                DuckLakeTableColumn::new(2, "details".into(), "json".into(), true),
                None,
            ),
            (
                DuckLakeTableColumn::new(3, "at".into(), "timetz".into(), true),
                None,
            ),
            (
                DuckLakeTableColumn::new(4, "count".into(), "INT".into(), true),
                None,
            ),
        ];

        let result = reconstruct_columns(rows).unwrap();

        assert_eq!(result[0].column_type, "geometry");
        assert_eq!(result[1].column_type, "json");
        assert_eq!(result[2].column_type, "timetz");
        assert_eq!(result[3].column_type, "int32");
    }

    #[test]
    fn test_reconstruct_columns_depth_is_bounded() {
        let count = crate::types::MAX_NESTED_TYPE_DEPTH + 2;
        let rows = (0..count)
            .map(|index| {
                let column_id = index as i64 + 1;
                let column_type = if index + 1 == count {
                    "int32"
                } else {
                    "struct"
                };
                (
                    DuckLakeTableColumn::new(
                        column_id,
                        format!("field_{index}"),
                        column_type.to_string(),
                        true,
                    ),
                    (index > 0).then_some(column_id - 1),
                )
            })
            .collect();

        let error = reconstruct_columns(rows).unwrap_err();

        assert!(error.to_string().contains("maximum depth"));
    }

    #[test]
    fn test_reconstruct_columns_struct() {
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

        let result = reconstruct_columns(rows).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].column_type, "struct<field_a:int32>");
        assert_eq!(result[0].nested_column_ids, vec![2]);
    }

    #[test]
    fn test_reconstruct_columns_arbitrary_nesting() {
        let rows = vec![
            (
                DuckLakeTableColumn::new(1, "payload".into(), "struct".into(), false),
                None,
            ),
            (
                DuckLakeTableColumn::new(2, "levels".into(), "list".into(), false),
                Some(1),
            ),
            (
                DuckLakeTableColumn::new(3, "element".into(), "struct".into(), false),
                Some(2),
            ),
            (
                DuckLakeTableColumn::new(4, "price".into(), "decimal(38, 16)".into(), false),
                Some(3),
            ),
            (
                DuckLakeTableColumn::new(5, "attrs".into(), "map".into(), true),
                Some(1),
            ),
            (
                DuckLakeTableColumn::new(6, "key".into(), "varchar".into(), false),
                Some(5),
            ),
            (
                DuckLakeTableColumn::new(7, "value".into(), "list".into(), true),
                Some(5),
            ),
            (
                DuckLakeTableColumn::new(8, "element".into(), "int32".into(), true),
                Some(7),
            ),
        ];

        let result = reconstruct_columns(rows).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(
            result[0].column_type,
            "struct<levels:list<struct<price:decimal(38, 16)>>,attrs:map<varchar,list<int32>>>"
        );
        assert_eq!(result[0].nested_column_ids, vec![2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn test_reconstruct_columns_rejects_invalid_map_children() {
        let rows = vec![
            (
                DuckLakeTableColumn::new(1, "attrs".into(), "map".into(), true),
                None,
            ),
            (
                DuckLakeTableColumn::new(2, "value".into(), "int32".into(), true),
                Some(1),
            ),
            (
                DuckLakeTableColumn::new(3, "key".into(), "varchar".into(), false),
                Some(1),
            ),
        ];

        assert!(reconstruct_columns(rows).is_err());
    }

    #[test]
    fn test_reconstruct_columns_rejects_duplicate_ids() {
        let rows = vec![
            (
                DuckLakeTableColumn::new(1, "id".into(), "int64".into(), false),
                None,
            ),
            (
                DuckLakeTableColumn::new(1, "name".into(), "varchar".into(), true),
                None,
            ),
        ];

        assert!(reconstruct_columns(rows).is_err());
    }

    #[test]
    fn test_reconstruct_columns_rejects_parent_cycle() {
        let rows = vec![
            (
                DuckLakeTableColumn::new(1, "left".into(), "struct".into(), false),
                Some(2),
            ),
            (
                DuckLakeTableColumn::new(2, "right".into(), "struct".into(), false),
                Some(1),
            ),
        ];

        assert!(reconstruct_columns(rows).is_err());
    }

    #[test]
    fn test_reconstruct_columns_multiple_lists() {
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

        let result = reconstruct_columns(rows).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].column_type, "list<varchar>");
        assert_eq!(result[1].column_type, "list<float64>");
    }

    #[test]
    fn test_reconstruct_columns_with_table_list() {
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

        let result = reconstruct_columns_with_table(rows).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].column.column_type, "list<float64>");
    }
}
