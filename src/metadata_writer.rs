//! Metadata writer trait and common types for DuckLake catalog writes.
//!
//! This module provides the `MetadataWriter` trait for writing metadata to DuckLake catalogs,
//! along with helper types for column definitions and data file registration.

use crate::types::{arrow_to_ducklake_type, ducklake_to_arrow_type};
use crate::{DuckLakeError, Result};
use arrow::array::{Array, FixedSizeBinaryArray};
use arrow::datatypes::{DataType, Schema as ArrowSchema};
use arrow::record_batch::RecordBatch;
use std::collections::{HashMap, HashSet};

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

pub(crate) fn table_write_changes(
    table_id: i64,
    mode: WriteMode,
    has_deletes: bool,
    replaced_existing_data: bool,
) -> String {
    match (mode, has_deletes, replaced_existing_data) {
        (WriteMode::Append, false, _) | (WriteMode::Replace, false, false) => {
            format!("inserted_into_table:{table_id}")
        },
        (WriteMode::Append | WriteMode::Replace, true, _) | (WriteMode::Replace, false, true) => {
            format!("deleted_from_table:{table_id},inserted_into_table:{table_id}")
        },
    }
}

pub(crate) fn quote_snapshot_name(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

pub(crate) fn quote_snapshot_table(schema_name: &str, table_name: &str) -> String {
    format!(
        "{}.{}",
        quote_snapshot_name(schema_name),
        quote_snapshot_name(table_name)
    )
}

/// Optional values stored in a DuckLake snapshot change row.
///
/// The fields map to `author`, `commit_message`, and `commit_extra_info` in
/// `ducklake_snapshot_changes`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SnapshotCommitMetadata {
    author: Option<String>,
    message: Option<String>,
    extra_info: Option<String>,
}

impl SnapshotCommitMetadata {
    /// Creates empty commit metadata.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            author: None,
            message: None,
            extra_info: None,
        }
    }

    /// Sets the snapshot author.
    #[must_use]
    pub fn with_author(mut self, author: impl Into<String>) -> Self {
        self.author = Some(author.into());
        self
    }

    /// Sets the snapshot commit message.
    #[must_use]
    pub fn with_message(mut self, message: impl Into<String>) -> Self {
        self.message = Some(message.into());
        self
    }

    /// Sets opaque extra information for the snapshot.
    #[must_use]
    pub fn with_extra_info(mut self, extra_info: impl Into<String>) -> Self {
        self.extra_info = Some(extra_info.into());
        self
    }

    /// Returns the snapshot author.
    #[must_use]
    pub fn author(&self) -> Option<&str> {
        self.author.as_deref()
    }

    /// Returns the snapshot commit message.
    #[must_use]
    pub fn message(&self) -> Option<&str> {
        self.message.as_deref()
    }

    /// Returns the snapshot extra information.
    #[must_use]
    pub fn extra_info(&self) -> Option<&str> {
        self.extra_info.as_deref()
    }

    fn ensure_supported_by_default(&self) -> Result<()> {
        if self.author.is_some() || self.message.is_some() || self.extra_info.is_some() {
            return Err(DuckLakeError::InvalidConfig(
                "commit metadata is not supported by this metadata writer".to_string(),
            ));
        }
        Ok(())
    }
}

pub(crate) fn inlined_text_value(array: &dyn Array, row: usize) -> Result<String> {
    if array.data_type() == &DataType::FixedSizeBinary(16) {
        let bytes = array
            .as_any()
            .downcast_ref::<FixedSizeBinaryArray>()
            .expect("Arrow data type and array implementation agree")
            .value(row);
        return uuid::Uuid::from_slice(bytes)
            .map(|value| value.to_string())
            .map_err(|error| DuckLakeError::InvalidConfig(format!("invalid UUID bytes: {error}")));
    }
    Ok(arrow::util::display::array_value_to_string(array, row)?)
}

pub(crate) fn scalar_type_supports_inlining(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Null
            | DataType::Boolean
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float32
            | DataType::Float64
            | DataType::Decimal128(_, _)
            | DataType::Date32
            | DataType::Time64(arrow::datatypes::TimeUnit::Microsecond)
            | DataType::Timestamp(_, _)
            | DataType::Interval(arrow::datatypes::IntervalUnit::MonthDayNano)
            | DataType::Utf8
            | DataType::LargeUtf8
            | DataType::Utf8View
            | DataType::Binary
            | DataType::LargeBinary
            | DataType::BinaryView
            | DataType::FixedSizeBinary(_)
    )
}

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
    /// Arrow type retained so nested child names and nullability survive catalog flattening.
    pub(crate) data_type: DataType,
    /// Literal used for files that predate a newly added column
    pub(crate) initial_default: Option<String>,
    /// Literal applied when a write omits this column
    pub(crate) default_value: Option<String>,
    /// Default encoding kind (`literal` when a default is set)
    pub(crate) default_value_type: Option<String>,
    /// SQL dialect recorded for interoperability
    pub(crate) default_value_dialect: Option<String>,
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
        let data_type = ducklake_to_arrow_type(&ducklake_type)?;
        Ok(Self {
            name,
            ducklake_type,
            is_nullable,
            data_type,
            initial_default: None,
            default_value: None,
            default_value_type: None,
            default_value_dialect: None,
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
            data_type: data_type.clone(),
            initial_default: None,
            default_value: None,
            default_value_type: None,
            default_value_dialect: None,
        })
    }

    /// Set a DuckLake literal default for this column.
    pub fn with_default(mut self, value: impl Into<String>) -> Result<Self> {
        let value = value.into();
        let data_type = ducklake_to_arrow_type(&self.ducklake_type)?;
        if crate::types::parse_ducklake_default_scalar(&value, &data_type).is_none() {
            return Err(DuckLakeError::InvalidConfig(format!(
                "Cannot decode default '{value}' for column '{}' as {data_type}",
                self.name
            )));
        }
        self.initial_default = Some(value.clone());
        self.default_value = Some(value);
        self.default_value_type = Some("literal".to_string());
        self.default_value_dialect = Some("duckdb".to_string());
        Ok(self)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct CatalogColumnDef {
    pub name: String,
    pub ducklake_type: String,
    pub logical_type: String,
    pub is_nullable: bool,
    pub parent_index: Option<usize>,
    pub initial_default: Option<String>,
    pub default_value: Option<String>,
    pub default_value_type: Option<String>,
    pub default_value_dialect: Option<String>,
}

#[derive(Debug, Clone)]
pub(crate) struct ExistingCatalogColumn {
    pub column_id: i64,
    pub name: String,
    pub ducklake_type: String,
    pub parent_column: Option<i64>,
}

pub(crate) fn catalog_column_defs(columns: &[ColumnDef]) -> Result<Vec<CatalogColumnDef>> {
    let mut result = Vec::new();
    for column in columns {
        let root_index = result.len();
        append_column_def(
            &column.name,
            &column.data_type,
            column.is_nullable,
            None,
            &mut result,
        )?;
        let root = &mut result[root_index];
        root.initial_default.clone_from(&column.initial_default);
        root.default_value.clone_from(&column.default_value);
        root.default_value_type
            .clone_from(&column.default_value_type);
        root.default_value_dialect
            .clone_from(&column.default_value_dialect);
    }
    Ok(result)
}

fn append_column_def(
    name: &str,
    data_type: &DataType,
    is_nullable: bool,
    parent_index: Option<usize>,
    result: &mut Vec<CatalogColumnDef>,
) -> Result<()> {
    let index = result.len();
    result.push(CatalogColumnDef {
        name: name.to_string(),
        ducklake_type: catalog_type_name(data_type)?,
        logical_type: arrow_to_ducklake_type(data_type)?,
        is_nullable,
        parent_index,
        initial_default: None,
        default_value: None,
        default_value_type: None,
        default_value_dialect: None,
    });
    match data_type {
        DataType::List(field) | DataType::LargeList(field) | DataType::FixedSizeList(field, _) => {
            append_column_def(
                "element",
                field.data_type(),
                field.is_nullable(),
                Some(index),
                result,
            )
        },
        DataType::Struct(fields) => {
            for field in fields {
                append_column_def(
                    field.name(),
                    field.data_type(),
                    field.is_nullable(),
                    Some(index),
                    result,
                )?;
            }
            Ok(())
        },
        DataType::Map(entries, _) => {
            let DataType::Struct(fields) = entries.data_type() else {
                return Err(DuckLakeError::UnsupportedType(
                    "Arrow map entries must be a struct".to_string(),
                ));
            };
            let [key, value] = fields.as_ref() else {
                return Err(DuckLakeError::UnsupportedType(
                    "Arrow maps must have key and value fields".to_string(),
                ));
            };
            append_column_def("key", key.data_type(), false, Some(index), result)?;
            append_column_def(
                "value",
                value.data_type(),
                value.is_nullable(),
                Some(index),
                result,
            )
        },
        _ => Ok(()),
    }
}

pub(crate) fn catalog_column_type_equal(existing_type: &str, proposed: &CatalogColumnDef) -> bool {
    existing_type.eq_ignore_ascii_case(&proposed.ducklake_type)
        || crate::types::types_equal_canonical(existing_type, &proposed.ducklake_type)
        || catalog_column_type_requires_migration(existing_type, proposed)
}

pub(crate) fn catalog_column_type_requires_migration(
    existing_type: &str,
    proposed: &CatalogColumnDef,
) -> bool {
    proposed.ducklake_type == "list"
        && existing_type
            .trim()
            .to_ascii_lowercase()
            .starts_with("list<")
        && crate::types::types_equal_canonical(existing_type, &proposed.logical_type)
}

fn catalog_type_name(data_type: &DataType) -> Result<String> {
    match data_type {
        DataType::List(_) | DataType::LargeList(_) | DataType::FixedSizeList(_, _) => {
            Ok("list".to_string())
        },
        DataType::Struct(_) => Ok("struct".to_string()),
        DataType::Map(_, _) => Ok("map".to_string()),
        _ => arrow_to_ducklake_type(data_type),
    }
}

pub(crate) fn top_level_column_ids(
    columns: &[CatalogColumnDef],
    field_ids: &[i64],
) -> Result<Vec<i64>> {
    if columns.len() != field_ids.len() {
        return Err(DuckLakeError::Internal(format!(
            "catalog column count {} does not match field id count {}",
            columns.len(),
            field_ids.len()
        )));
    }
    Ok(columns
        .iter()
        .zip(field_ids)
        .filter_map(|(column, field_id)| column.parent_index.is_none().then_some(*field_id))
        .collect())
}

pub(crate) fn assign_column_ids(
    proposed: &[CatalogColumnDef],
    existing: &[ExistingCatalogColumn],
    fresh_ids: &[i64],
) -> Result<Vec<i64>> {
    if proposed.len() != fresh_ids.len() {
        return Err(DuckLakeError::Internal(format!(
            "proposed column count {} does not match reserved id count {}",
            proposed.len(),
            fresh_ids.len()
        )));
    }
    let mut existing_paths: HashMap<i64, Vec<String>> = HashMap::new();
    let mut unresolved = existing.iter().collect::<Vec<_>>();
    while !unresolved.is_empty() {
        let before = unresolved.len();
        unresolved.retain(|column| {
            let parent_path = match column.parent_column {
                Some(parent_id) => match existing_paths.get(&parent_id) {
                    Some(path) => path.clone(),
                    None => return true,
                },
                None => Vec::new(),
            };
            let mut path = parent_path;
            path.push(column.name.clone());
            existing_paths.insert(column.column_id, path);
            false
        });
        if unresolved.len() == before {
            return Err(DuckLakeError::InvalidConfig(
                "Catalog contains an orphaned or cyclic nested column".to_string(),
            ));
        }
    }
    if existing_paths.len() != existing.len() {
        return Err(DuckLakeError::InvalidConfig(
            "Catalog contains duplicate column ids".to_string(),
        ));
    }
    let mut existing_by_path = HashMap::new();
    for column in existing {
        let path = existing_paths
            .get(&column.column_id)
            .expect("every existing column path was resolved")
            .clone();
        if existing_by_path.insert(path, column.column_id).is_some() {
            return Err(DuckLakeError::InvalidConfig(
                "Catalog contains duplicate nested column paths".to_string(),
            ));
        }
    }
    let mut proposed_paths: Vec<Vec<String>> = Vec::with_capacity(proposed.len());
    let mut proposed_path_set = HashSet::with_capacity(proposed.len());
    for column in proposed {
        let mut path = column
            .parent_index
            .map(|parent_index| proposed_paths[parent_index].clone())
            .unwrap_or_default();
        path.push(column.name.clone());
        if !proposed_path_set.insert(path.clone()) {
            return Err(DuckLakeError::InvalidConfig(
                "Proposed schema contains duplicate nested column paths".to_string(),
            ));
        }
        proposed_paths.push(path);
    }
    Ok(proposed_paths
        .iter()
        .zip(fresh_ids)
        .map(|(path, fresh_id)| existing_by_path.get(path).copied().unwrap_or(*fresh_id))
        .collect())
}

pub(crate) fn catalog_columns_differ(
    existing: &[ExistingCatalogColumn],
    existing_nullability: &[bool],
    proposed: &[CatalogColumnDef],
    field_ids: &[i64],
) -> bool {
    if existing.len() != proposed.len()
        || existing.len() != existing_nullability.len()
        || proposed.len() != field_ids.len()
    {
        return true;
    }
    existing
        .iter()
        .zip(existing_nullability)
        .zip(proposed.iter().zip(field_ids))
        .any(|((existing, existing_nullable), (proposed, field_id))| {
            let parent_id = proposed.parent_index.map(|index| field_ids[index]);
            let same_type = catalog_column_type_equal(&existing.ducklake_type, proposed)
                || crate::types::is_promotable(&proposed.ducklake_type, &existing.ducklake_type);
            existing.column_id != *field_id
                || existing.name != proposed.name
                || !same_type
                || *existing_nullable != proposed.is_nullable
                || existing.parent_column != parent_id
        })
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
#[cfg(test)]
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

/// Per-column statistics for one data file, persisted to
/// `ducklake_file_column_stats` and used for file-level pruning.
///
/// Mirrors the official DuckLake row shape. `min_value` / `max_value` are the
/// DuckDB-canonical `VARCHAR` encoding of the bounds (see
/// [`crate::stats_encode`]); `None` means SQL `NULL` — DuckLake keeps (never
/// prunes) a file whose bound is NULL, so `None` is always the safe value for a
/// type or value we cannot faithfully encode.
#[derive(Debug, Clone, PartialEq)]
pub struct ColumnStat {
    /// Catalog `column_id` (DuckLake field id) this stat describes.
    pub column_id: i64,
    /// Minimum value, DuckDB-canonical `VARCHAR`, or `None` for SQL `NULL`.
    pub min_value: Option<String>,
    /// Maximum value, DuckDB-canonical `VARCHAR`, or `None` for SQL `NULL`.
    pub max_value: Option<String>,
    /// Number of NULL values in this column across the file.
    pub null_count: Option<i64>,
    /// Number of non-NULL values in this column across the file.
    pub value_count: Option<i64>,
    /// For `FLOAT`/`DOUBLE` columns: whether any NaN is present. `None` for
    /// non-floating columns, and for float columns whose NaN state is unknown
    /// (footer-only harvests — the parquet footer carries no NaN signal). When
    /// `true`, `min_value`/`max_value` are omitted (matching DuckLake); readers
    /// may only trust a float `max_value` as an upper bound when this is
    /// `false`, since NaN sorts above every value.
    pub contains_nan: Option<bool>,
    /// Total compressed size of the column in bytes, summed over the file's
    /// column chunks (parquet `total_compressed_size`).
    pub column_size_bytes: Option<i64>,
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
    /// Per-column statistics (min/max/null counts) to persist to
    /// `ducklake_file_column_stats`. Empty when stats were not computed (e.g. a
    /// caller that predates stats support); backends must treat an empty vector
    /// as "no stats rows for this file", which is spec-safe.
    pub column_stats: Vec<ColumnStat>,
    /// The partition spec generation (`ducklake_partition_info.partition_id`) this
    /// file was written under, or `None` for a file of an unpartitioned table.
    /// When set, the backend stores it on the `ducklake_data_file` row.
    pub partition_id: Option<i64>,
    /// The file's partition values as `(partition_key_index, value)` — the single
    /// transformed value every row in the file shares for each partition key,
    /// DuckDB-canonical VARCHAR (`None` == SQL NULL). Persisted to
    /// `ducklake_file_partition_value`. Empty for an unpartitioned file.
    pub partition_values: Vec<(i32, Option<String>)>,
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
            column_stats: Vec::new(),
            partition_id: None,
            partition_values: Vec::new(),
        }
    }

    /// Set the footer size for read optimization.
    pub fn with_footer_size(mut self, footer_size: i64) -> Self {
        self.footer_size = Some(footer_size);
        self
    }

    /// Attach per-column statistics to persist to `ducklake_file_column_stats`.
    pub fn with_column_stats(mut self, column_stats: Vec<ColumnStat>) -> Self {
        self.column_stats = column_stats;
        self
    }

    /// Attach the partition spec generation and per-key partition values for this
    /// file (persisted to `ducklake_data_file.partition_id` and
    /// `ducklake_file_partition_value`). Values are `(partition_key_index, value)`.
    pub fn with_partition(
        mut self,
        partition_id: i64,
        partition_values: Vec<(i32, Option<String>)>,
    ) -> Self {
        self.partition_id = Some(partition_id);
        self.partition_values = partition_values;
        self
    }

    /// Mark this file as having an absolute path.
    pub fn with_absolute_path(mut self) -> Self {
        self.path_is_relative = false;
        self
    }
}

/// Enforce the partition-spec invariant for one file being committed, given the
/// table's currently-live partition generation (`live_partition_id`, `None` when
/// the table has no live spec). Every backend's `register_data_file` /
/// `register_data_files` commit path calls this to fence the partition DDL race in
/// BOTH directions:
///
/// - a file carrying a `partition_id` must reference the generation live *now* — a
///   concurrent `RESET`/`SET PARTITIONED BY` that retired it (or replaced it with a
///   new generation) since the write was planned makes the stamped id stale; and
/// - a *non-empty* file WITHOUT a `partition_id` must not land in a table that now
///   has a live spec (an unpartitioned write planned before a concurrent
///   `SET PARTITIONED BY`).
///
/// A 0-row file is exempt: an empty `Replace`/`Overwrite` truncate marker carries
/// no partitioned data, so it violates neither direction. Returns
/// [`DuckLakeError::Conflict`] on a violation so the caller aborts (rolls back) the
/// commit and the write is retried against the current spec.
pub(crate) fn enforce_partition_fence(
    table_id: i64,
    live_partition_id: Option<i64>,
    file: &DataFileInfo,
) -> Result<()> {
    match file.partition_id {
        Some(pid) if live_partition_id != Some(pid) => Err(DuckLakeError::Conflict(format!(
            "partition spec (partition_id {pid}) for table {table_id} was changed by a concurrent \
             SET/RESET PARTITIONED BY during this commit; re-open the catalog and retry"
        ))),
        None if file.record_count > 0 && live_partition_id.is_some() => {
            Err(DuckLakeError::Conflict(format!(
                "table {table_id} gained a partition spec (concurrent SET PARTITIONED BY) after this \
                 unpartitioned write was planned; re-open the catalog and retry"
            )))
        },
        _ => Ok(()),
    }
}

/// Validate the partition values a caller attached to a file it is *promoting*
/// (registering an already-written parquet, rather than one this crate wrote)
/// against the live spec's `transforms`, in key order.
///
/// A promoted file is registered byte-for-byte, so whether its rows really do all
/// share these values cannot be established from the catalog — that is the caller's
/// assertion. Official DuckLake makes the same assumption, taking the values from
/// the file's Hive path and validating only their shape
/// (`IsValidTransformedHivePartitionValue` checks bucket range; `MapHiveColumn`
/// checks castability). This mirrors that, checking what the catalog can prove:
///
/// - one value per partition key, no more and no fewer;
/// - key indices exactly `0..n-1`, each once (a duplicated or out-of-range index
///   would silently drop or mis-assign a key on read);
/// - for `bucket(N)`, a value that parses as an integer in `0..N`, or SQL NULL.
///   NULL is legal for every transform — a NULL input yields a NULL partition value,
///   which is a partition in its own right (DuckDB's `__HIVE_DEFAULT_PARTITION__`).
///   Official agrees: `IsValidTransformedHivePartitionValue` returns early for a NULL
///   hive value before its bucket-range check.
///
/// Values for `identity` and the temporal transforms are opaque strings here: the
/// column type they must cast to is not part of the spec, so type checking belongs
/// to whoever produced them. A caller that cannot vouch for the file's contents
/// should verify against the parquet footer's per-column min/max before promoting —
/// for `identity`, `min == max == value` proves the file is single-partition.
pub(crate) fn validate_promoted_partition_values(
    table_id: i64,
    transforms: &[String],
    key_column_types: &[Option<DataType>],
    file: &DataFileInfo,
) -> Result<()> {
    use crate::partition::PartitionTransform;

    if file.partition_values.len() != transforms.len() {
        return Err(DuckLakeError::InvalidConfig(format!(
            "promoted file for table {table_id} carries {} partition value(s) but the table's \
             live partition spec has {} key(s)",
            file.partition_values.len(),
            transforms.len()
        )));
    }
    let mut seen = vec![false; transforms.len()];
    for (key_index, value) in &file.partition_values {
        let index = usize::try_from(*key_index).ok().filter(|i| *i < seen.len());
        let Some(index) = index else {
            return Err(DuckLakeError::InvalidConfig(format!(
                "promoted file for table {table_id} has partition_key_index {key_index}, outside \
                 the live spec's 0..{} keys",
                transforms.len()
            )));
        };
        if seen[index] {
            return Err(DuckLakeError::InvalidConfig(format!(
                "promoted file for table {table_id} repeats partition_key_index {key_index}"
            )));
        }
        seen[index] = true;

        // Check the value is well-formed for the transform and, for `identity`, that
        // it casts to the key column's type — official does the same
        // (`MapHiveColumn` errors when the Hive value will not cast). A NULL value is
        // legitimate under any transform and always passes. An unknown column type
        // (a key column absent from the promoted schema) skips the type check rather
        // than guessing.
        let transform = PartitionTransform::parse(&transforms[index]);
        let column_type = key_column_types
            .get(index)
            .and_then(|t| t.clone())
            .unwrap_or(DataType::Utf8);
        if !transform.value_is_well_formed(value.as_deref(), &column_type) {
            return Err(DuckLakeError::InvalidConfig(format!(
                "promoted file for table {table_id} has partition value {value:?} for key \
                 {key_index} with transform '{}' on a {column_type} column; the value is not \
                 valid for that key",
                transform.to_catalog_string()
            )));
        }
    }
    Ok(())
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

/// One source (data) file being retired by a compaction commit
/// ([`MetadataWriter::commit_compaction`]).
///
/// Its rows have been rewritten into a [`CompactionOutputFile`]; the commit
/// retires this data file (sets `end_snapshot`) and its live delete file (if
/// any) and schedules BOTH for physical deletion. `delete_file_id` is a
/// compare-and-swap guard: the commit aborts if the live delete file for
/// `data_file_id` no longer matches it (a concurrent DELETE/UPDATE moved the
/// file's live rows since they were read, which would otherwise resurrect
/// deleted rows into the rewritten output).
#[derive(Debug, Clone)]
pub struct CompactionSourceFile {
    /// The data file being retired + scheduled for deletion.
    pub data_file_id: i64,
    /// The live delete file the caller resolved the source's live rows against
    /// (retired + scheduled with the data file), or `None` if none was live.
    pub delete_file_id: Option<i64>,
}

/// How a compaction commit retires its source files
/// ([`MetadataWriter::commit_compaction`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceRetirement {
    /// The outputs fully represent the sources for EVERY snapshot (a merged
    /// partial file, visible from its min origin snapshot with per-row
    /// filtering), so the source catalog rows are removed and their physical
    /// files scheduled for deletion — `cleanup_old_files` may reclaim them at
    /// once. Used by `merge_adjacent_files`.
    Remove,
    /// The outputs only hold currently-live rows (a `rewrite_data_files`
    /// output), so the sources still serve time travel to pre-compaction
    /// snapshots: retire them (set `end_snapshot`) but do NOT schedule them.
    /// `expire_snapshots` schedules them once their snapshots are expired, so
    /// disk reclamation is deferred but time travel stays correct.
    Retire,
}

/// One new file to register in a compaction commit
/// ([`MetadataWriter::commit_compaction`]).
///
/// The parquet has already been written (embedding each row's original rowid,
/// and for a merged partial file the per-row `_ducklake_internal_snapshot_id`
/// column); this is the catalog registration. `row_id_start` is stored NULL
/// because the file's rowids are served from its embedded rowid column, not
/// synthesized from a start.
#[derive(Debug, Clone)]
pub struct CompactionOutputFile {
    /// The written parquet's location / size / record count.
    pub file: DataFileInfo,
    /// `partial_max` for a merged partial file: the maximum origin snapshot id
    /// among its rows (so a reader knows the file is partial and, below this
    /// snapshot, filters its rows per-origin). `None` for a rewrite output or a
    /// merge whose rows all share one origin snapshot.
    pub partial_max: Option<i64>,
    /// `begin_snapshot` for this output. For a merged partial file, the MINIMUM
    /// origin snapshot among its rows, so historical reads back to that point
    /// see it (row-filtered by origin). `None` means "use the new compaction
    /// snapshot" — the correct choice for a rewrite output, whose rows are all
    /// currently-live and whose pre-compaction history is served by the retained
    /// sources.
    pub begin_snapshot: Option<i64>,
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

/// Stable identity of one visible inlined row selected for deletion.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct InlinedRowRef {
    /// Physical `ducklake_inlined_data_*` table registered for the table.
    pub table_name: String,
    /// Stable DuckLake row id within that table.
    pub row_id: i64,
}

/// Row storage staged for one table in a multi-table write.
#[derive(Debug, Clone)]
pub enum StagedTableData {
    /// Parquet data files already uploaded to object storage.
    Files(Vec<DataFileInfo>),
    /// Record batches to store in DuckLake's metadata catalog.
    Inlined(Vec<RecordBatch>),
    /// No inserted rows; the stage contains only deletes.
    None,
}

/// One table's staged changes in a multi-table write.
#[derive(Debug, Clone)]
pub struct StagedTableWrite {
    pub(crate) table_id: i64,
    pub(crate) schema_name: String,
    pub(crate) table_name: String,
    pub(crate) base_snapshot_id: i64,
    pub(crate) mode: WriteMode,
    pub(crate) columns: Vec<ColumnDef>,
    pub(crate) column_ids: Vec<i64>,
    pub(crate) data: StagedTableData,
    pub(crate) snapshot_id_columns: Vec<String>,
    pub(crate) positional_deletes: Vec<DeleteFileEntry>,
    pub(crate) inlined_deletes: Vec<InlinedRowRef>,
}

impl StagedTableWrite {
    /// Returns the target table id reserved during write setup.
    #[must_use]
    pub const fn table_id(&self) -> i64 {
        self.table_id
    }

    /// Returns the target schema name.
    #[must_use]
    pub fn schema_name(&self) -> &str {
        &self.schema_name
    }

    /// Returns the target table name.
    #[must_use]
    pub fn table_name(&self) -> &str {
        &self.table_name
    }

    /// Returns the table snapshot observed during write setup.
    #[must_use]
    pub const fn base_snapshot_id(&self) -> i64 {
        self.base_snapshot_id
    }

    /// Returns the staged write mode.
    #[must_use]
    pub const fn mode(&self) -> WriteMode {
        self.mode
    }

    /// Returns the staged catalog columns.
    #[must_use]
    pub fn columns(&self) -> &[ColumnDef] {
        &self.columns
    }

    /// Returns the catalog column ids paired with the staged columns.
    #[must_use]
    pub fn column_ids(&self) -> &[i64] {
        &self.column_ids
    }

    /// Returns the staged row storage.
    #[must_use]
    pub const fn data(&self) -> &StagedTableData {
        &self.data
    }

    /// Returns the staged positional deletes.
    #[must_use]
    pub fn positional_deletes(&self) -> &[DeleteFileEntry] {
        &self.positional_deletes
    }

    /// Returns the staged inlined-row deletes.
    #[must_use]
    pub fn inlined_deletes(&self) -> &[InlinedRowRef] {
        &self.inlined_deletes
    }
}

/// Result of one atomic multi-table write.
#[derive(Debug, Clone)]
pub struct MultiTableCommit {
    /// Snapshot shared by every committed table change.
    pub snapshot_id: i64,
    /// Authoritative schema and table ids for each staged table.
    pub tables: Vec<CommitIds>,
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
    /// Top-level column IDs in Arrow schema order.
    pub column_ids: Vec<i64>,
    /// Every typed field ID in DuckLake depth-first preorder.
    pub field_ids: Vec<i64>,
}

/// Trait for writing metadata to DuckLake catalogs.
///
/// Implementations must be thread-safe (`Send + Sync`).
pub trait MetadataWriter: Send + Sync + std::fmt::Debug {
    /// Create a new snapshot and return its ID.
    fn create_snapshot(&self) -> Result<i64>;

    /// Replace a global-scoped catalog setting.
    fn set_global_setting(&self, _key: &str, _value: &str) -> Result<()> {
        Err(DuckLakeError::Unsupported(
            "global-scoped settings are not supported by this metadata backend".to_string(),
        ))
    }

    /// Replace a table-scoped catalog setting.
    fn set_table_setting(&self, _table_id: i64, _key: &str, _value: &str) -> Result<()> {
        Err(DuckLakeError::Unsupported(
            "table-scoped settings are not supported by this metadata backend".to_string(),
        ))
    }

    /// Run an operation while holding a backend-appropriate commit lock.
    ///
    /// A coordination utility for callers that want to serialize a multi-step
    /// commit workflow (e.g. read staged state, dedup, commit) under one
    /// `identity` across processes. Correctness does not depend on it: the
    /// optimistic `expected_base_snapshot_id` fence remains the conflict
    /// mechanism, and this lock only avoids duplicate concurrent work.
    ///
    /// Contract: the lock is held for the duration of `operation` and released
    /// on both success and error before this returns; an `operation` error
    /// propagates and takes precedence over a release error. A crashed holder
    /// must not leave the lock held (backends use self-releasing mechanisms:
    /// an advisory transaction lock, a file lock released on close). Keep
    /// critical sections short — implementations may pin a connection while
    /// the lock is held.
    fn with_commit_lock(
        &self,
        _identity: &str,
        _operation: Box<dyn FnOnce() -> Result<()> + '_>,
    ) -> Result<()> {
        Err(DuckLakeError::Unsupported(
            "commit locking is not supported by this metadata backend".to_string(),
        ))
    }

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

    /// Set (or replace) the table's partition spec — DuckLake partitioning DDL,
    /// the commit behind `ALTER TABLE … SET PARTITIONED BY (…)`.
    ///
    /// In one transaction: create a new snapshot; end the currently-live
    /// `ducklake_partition_info` row (and its `ducklake_partition_column` rows) if
    /// one exists; insert a new generation with a fresh `partition_id` and one
    /// `ducklake_partition_column` row per `(column_name, transform)` in order
    /// (each column NAME resolved to the table's live `column_id`); and
    /// bump/record `schema_version` (setting a spec is DDL). Existing data files
    /// are left untouched. `columns` must be non-empty (use
    /// [`reset_partition_spec`](MetadataWriter::reset_partition_spec) to remove a
    /// spec) and each column must exist; otherwise
    /// [`crate::DuckLakeError::InvalidConfig`]. Returns the new snapshot id.
    ///
    /// Default: unsupported; writable backends override it.
    fn set_partition_spec(
        &self,
        _table_id: i64,
        _columns: &[(String, crate::partition::PartitionTransform)],
    ) -> Result<i64> {
        Err(DuckLakeError::InvalidConfig(
            "SET PARTITIONED BY is not supported on this metadata backend".to_string(),
        ))
    }

    /// The table's currently-live partition spec (`ducklake_partition_info` with
    /// `end_snapshot IS NULL`, joined to its key columns), or `None` when the table
    /// is unpartitioned.
    ///
    /// The write paths need this from the *writer* side: a caller reaching
    /// [`crate::table_writer::DuckLakeTableWriter`] directly (rather than through SQL
    /// `INSERT`) has no [`crate::metadata_provider::MetadataProvider`] to ask, but
    /// still must lay its files out per the spec and stamp the right `partition_id` —
    /// otherwise `enforce_partition_fence` rejects the commit. Resolved against the
    /// write schema by [`crate::partition::PartitionWriteSpec::resolve`].
    ///
    /// The returned spec is for WRITING only: `prune_safe` is always `false`, since
    /// deciding whether a mapping may prune arbitrary live files needs the full
    /// generation history that the read path loads.
    ///
    /// Reading it at write-planning time is inherently racy against a concurrent
    /// `SET`/`RESET PARTITIONED BY` — `enforce_partition_fence` is what makes the
    /// commit safe, by re-checking the live generation inside the commit transaction.
    ///
    /// Default: `None` (backends without partition support are never partitioned).
    fn live_partition_spec(
        &self,
        _table_id: i64,
    ) -> Result<Option<crate::partition::PartitionSpec>> {
        Ok(None)
    }

    /// Remove the table's partition spec — the commit behind `ALTER TABLE …
    /// RESET PARTITIONED BY`.
    ///
    /// In one transaction: create a new snapshot, end the currently-live
    /// `ducklake_partition_info` row (a no-op if none is live), and bump/record
    /// `schema_version`. Existing partitioned data files keep their `partition_id`
    /// and values (they stay readable); only subsequent writes are unpartitioned.
    /// Returns the new snapshot id (or the current head if there was nothing to
    /// reset).
    ///
    /// Default: unsupported; writable backends override it.
    fn reset_partition_spec(&self, _table_id: i64) -> Result<i64> {
        Err(DuckLakeError::InvalidConfig(
            "RESET PARTITIONED BY is not supported on this metadata backend".to_string(),
        ))
    }

    /// The table's currently-live sort spec (`ducklake_sort_info` with
    /// `end_snapshot IS NULL`, joined to its expressions), or `None` when the table
    /// has no sort order.
    ///
    /// The counterpart of `live_partition_spec` for sort: a caller reaching
    /// [`crate::table_writer::DuckLakeTableWriter`] directly has no
    /// [`crate::metadata_provider::MetadataProvider`] to ask, but a bulk write should
    /// still lay its rows out in the table's sort order so successive files cover
    /// contiguous, non-overlapping ranges.
    ///
    /// Unlike partitioning, getting this wrong is not a correctness problem — an
    /// unsorted file reads back fine, it just prunes less well — so there is no
    /// commit-time fence for sort.
    ///
    /// Default: `None` (backends without sort support are never sorted).
    fn live_sort_spec(&self, _table_id: i64) -> Result<Option<crate::sort::SortSpec>> {
        Ok(None)
    }

    /// Set the table's sort spec — the commit behind `ALTER TABLE … SET SORTED BY
    /// (…)`.
    ///
    /// In one transaction: create a new snapshot; end the currently-live
    /// `ducklake_sort_info` row if one exists; insert a new generation with a fresh
    /// `sort_id` and one `ducklake_sort_expression` row per `SortField` in order.
    /// `fields` must be non-empty (use
    /// [`reset_sort_spec`](MetadataWriter::reset_sort_spec) to remove a spec).
    ///
    /// Unlike [`set_partition_spec`](MetadataWriter::set_partition_spec), this does
    /// **not** bump `schema_version`: DuckLake treats a sort-order change as
    /// metadata that does not alter the logical schema (existing readers are
    /// unaffected — sort order only influences how *future* writes are laid out).
    /// Existing data files are left untouched. Returns the new snapshot id.
    ///
    /// Default: unsupported; writable backends override it.
    fn set_sort_spec(&self, _table_id: i64, _fields: &[crate::sort::SortField]) -> Result<i64> {
        Err(DuckLakeError::InvalidConfig(
            "SET SORTED BY is not supported on this metadata backend".to_string(),
        ))
    }

    /// Remove the table's sort spec — the commit behind `ALTER TABLE … RESET SORTED
    /// BY`.
    ///
    /// In one transaction: create a new snapshot and end the currently-live
    /// `ducklake_sort_info` row (a no-op if none is live). Does not bump
    /// `schema_version` (see [`set_sort_spec`](MetadataWriter::set_sort_spec)).
    /// Existing data files keep whatever order they were written in; only
    /// subsequent writes are unsorted. Returns the new snapshot id (or the current
    /// head if there was nothing to reset).
    ///
    /// Default: unsupported; writable backends override it.
    fn reset_sort_spec(&self, _table_id: i64) -> Result<i64> {
        Err(DuckLakeError::InvalidConfig(
            "RESET SORTED BY is not supported on this metadata backend".to_string(),
        ))
    }

    /// Register a new data file and publish its snapshot as the catalog head,
    /// atomically. For `Replace`, retires the prior generation in the same
    /// transaction. Returns the committed snapshot id: assigned at this commit
    /// for SQLite (so it may differ from `WriteSetupResult::snapshot_id` under
    /// concurrency), reserved at begin for Postgres.
    ///
    /// `columns` carries the top-level Arrow schema. `column_ids` carries every
    /// typed node ID in depth-first `column_order`, matching
    /// `WriteSetupResult::field_ids`. Backends
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

    /// Register MULTIPLE new data files in ONE snapshot — the atomic commit behind
    /// a partitioned INSERT / CTAS, which writes one file per partition. Like
    /// [`register_data_file`](MetadataWriter::register_data_file) but commits all
    /// `files` together: `Replace` retires the prior generation once, then every
    /// file is added; `Append` just adds them. Each file is assigned a distinct
    /// `row_id_start` from the advancing row-lineage counter, and its
    /// `partition_id` / `partition_values` (when set) are persisted. `files` must
    /// be non-empty. Returns the committed ids.
    ///
    /// Default: falls back to a single [`register_data_file`](MetadataWriter::register_data_file) when exactly one file
    /// is given (so a non-partitioned write works everywhere), and otherwise errors
    /// — backends that support partitioned writes override this to commit N files
    /// atomically in one snapshot.
    #[allow(clippy::too_many_arguments)]
    fn register_data_files(
        &self,
        table_id: i64,
        schema_name: &str,
        table_name: &str,
        snapshot_id: i64,
        files: &[DataFileInfo],
        mode: WriteMode,
        base_snapshot: i64,
        columns: &[ColumnDef],
        column_ids: &[i64],
    ) -> Result<CommitIds> {
        match files {
            [file] => self.register_data_file(
                table_id,
                schema_name,
                table_name,
                snapshot_id,
                file,
                mode,
                base_snapshot,
                columns,
                column_ids,
            ),
            _ => Err(DuckLakeError::InvalidConfig(
                "register_data_files (atomic multi-file / partitioned write) is not \
                 supported on this metadata backend"
                    .to_string(),
            )),
        }
    }

    /// Registers one data file and attaches commit metadata.
    ///
    /// The default implementation accepts empty metadata and otherwise returns
    /// an error. Metadata backends that support these fields override it.
    #[allow(clippy::too_many_arguments)]
    fn register_data_file_with_commit_metadata(
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
        commit_metadata: &SnapshotCommitMetadata,
        expected_base_snapshot_id: Option<i64>,
    ) -> Result<CommitIds> {
        if expected_base_snapshot_id.is_some() {
            return Err(DuckLakeError::InvalidConfig(
                "conditional writes are not supported by this metadata writer".to_string(),
            ));
        }
        commit_metadata.ensure_supported_by_default()?;
        self.register_data_file(
            table_id,
            schema_name,
            table_name,
            snapshot_id,
            file,
            mode,
            base_snapshot,
            columns,
            column_ids,
        )
    }

    /// Registers multiple data files and attaches commit metadata.
    ///
    /// The default implementation accepts empty metadata and otherwise returns
    /// an error. Metadata backends that support these fields override it.
    #[allow(clippy::too_many_arguments)]
    fn register_data_files_with_commit_metadata(
        &self,
        table_id: i64,
        schema_name: &str,
        table_name: &str,
        snapshot_id: i64,
        files: &[DataFileInfo],
        mode: WriteMode,
        base_snapshot: i64,
        columns: &[ColumnDef],
        column_ids: &[i64],
        commit_metadata: &SnapshotCommitMetadata,
        expected_base_snapshot_id: Option<i64>,
    ) -> Result<CommitIds> {
        if expected_base_snapshot_id.is_some() {
            return Err(DuckLakeError::InvalidConfig(
                "conditional multi-file writes are not supported by this metadata writer"
                    .to_string(),
            ));
        }
        commit_metadata.ensure_supported_by_default()?;
        self.register_data_files(
            table_id,
            schema_name,
            table_name,
            snapshot_id,
            files,
            mode,
            base_snapshot,
            columns,
            column_ids,
        )
    }

    /// Whether this writer can store a small insert with `schema`'s column
    /// types inlined in the catalog ([`Self::register_inlined_data`]) such that
    /// its own inline read path decodes them back exactly. The write path
    /// consults this before routing a row-bearing write at or below the scoped
    /// `data_inlining_row_limit`: a writer that returns `false` keeps the
    /// Parquet path instead of failing the write (or corrupting a round trip),
    /// since when to inline is writer policy under the DuckLake specification.
    fn supports_data_inlining(&self, _schema: &ArrowSchema) -> bool {
        false
    }

    /// Store a small insert in the catalog instead of writing a Parquet file.
    ///
    /// Implementations create or reuse the per-`(table_id, schema_version)`
    /// physical table, register it in `ducklake_inlined_data_tables`, insert the
    /// rows with their stable row ids and snapshot bounds, update table stats,
    /// record `commit_metadata` and the snapshot change ledger (appending to any
    /// DDL entries already recorded for the snapshot), and — when
    /// `expected_base_snapshot_id` is set and `mode` is not Replace — abort with
    /// [`crate::DuckLakeError::Conflict`] if the table's data-file generation
    /// changed since that snapshot, all in the same transaction as the snapshot
    /// commit.
    #[allow(clippy::too_many_arguments)]
    fn register_inlined_data(
        &self,
        _table_id: i64,
        _schema_name: &str,
        _table_name: &str,
        _snapshot_id: i64,
        _batches: &[RecordBatch],
        _mode: WriteMode,
        _base_snapshot: i64,
        _columns: &[ColumnDef],
        _column_ids: &[i64],
        _commit_metadata: &SnapshotCommitMetadata,
        _expected_base_snapshot_id: Option<i64>,
    ) -> Result<CommitIds> {
        Err(DuckLakeError::InvalidConfig(
            "data inlining is not supported by this metadata writer".to_string(),
        ))
    }

    /// Commit staged changes for multiple tables in one metadata transaction.
    ///
    /// Implementations allocate one snapshot, evaluate the optional table-state
    /// fence once for the complete write, and make every table change visible
    /// together. A returned error must leave no staged metadata visible.
    fn commit_multi_table(
        &self,
        _writes: &[StagedTableWrite],
        _commit_metadata: &SnapshotCommitMetadata,
        _expected_base_snapshot_id: Option<i64>,
    ) -> Result<MultiTableCommit> {
        Err(DuckLakeError::InvalidConfig(
            "multi-table writes are not supported by this metadata writer".to_string(),
        ))
    }

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

    /// Registers a data file with positional deletes and commit metadata.
    ///
    /// The default implementation accepts empty metadata and otherwise returns
    /// an error. Metadata backends that support these fields override it.
    #[allow(clippy::too_many_arguments)]
    fn register_data_file_with_deletes_and_commit_metadata(
        &self,
        table_id: i64,
        schema_name: &str,
        table_name: &str,
        snapshot_id: i64,
        file: &DataFileInfo,
        deletes: &[DeleteFileEntry],
        mode: WriteMode,
        base_snapshot: i64,
        columns: &[ColumnDef],
        column_ids: &[i64],
        commit_metadata: &SnapshotCommitMetadata,
        expected_base_snapshot_id: Option<i64>,
    ) -> Result<CommitIds> {
        if expected_base_snapshot_id.is_some() {
            return Err(DuckLakeError::InvalidConfig(
                "conditional combined writes are not supported by this metadata writer".to_string(),
            ));
        }
        commit_metadata.ensure_supported_by_default()?;
        self.register_data_file_with_deletes(
            table_id,
            schema_name,
            table_name,
            snapshot_id,
            file,
            deletes,
            mode,
            base_snapshot,
            columns,
            column_ids,
        )
    }

    /// Atomically register N new data files AND apply positional deletes to
    /// existing data files, in a SINGLE snapshot — the multi-file form of
    /// [`register_data_file_with_deletes`], for a keyed mutation whose appended
    /// side spans several files (one per partition, or a rolled sequence).
    ///
    /// [`register_data_file_with_deletes`]: MetadataWriter::register_data_file_with_deletes
    ///
    /// In one transaction: allocate one snapshot; insert every file in `files`,
    /// each drawing a distinct `row_id_start` from the advancing row-lineage
    /// counter and carrying its own per-column stats and partition assignment,
    /// exactly as [`register_data_files`](MetadataWriter::register_data_files);
    /// then, for each [`DeleteFileEntry`], apply the same target-file fence +
    /// compare-and-swap + retire-prior + insert-cumulative as
    /// [`set_delete_file`](MetadataWriter::set_delete_file), all stamped with
    /// that one snapshot. Advance the catalog head LAST, so every appended file
    /// and every delete become visible together — never a half-applied
    /// intermediate state. Aborts with [`crate::DuckLakeError::Conflict`] on the
    /// first entry whose target data file was retired since `base_snapshot`, or
    /// whose live delete file no longer matches `expected_prev_delete_file`.
    ///
    /// `files` must be non-empty; `deletes` may be empty (equivalent to
    /// [`register_data_files`](MetadataWriter::register_data_files)) and each
    /// entry must target a distinct `data_file_id`.
    ///
    /// Default: unsupported; backends override it.
    #[allow(clippy::too_many_arguments)]
    fn register_data_files_with_deletes(
        &self,
        _table_id: i64,
        _schema_name: &str,
        _table_name: &str,
        _snapshot_id: i64,
        _files: &[DataFileInfo],
        _deletes: &[DeleteFileEntry],
        _mode: WriteMode,
        _base_snapshot: i64,
        _columns: &[ColumnDef],
        _column_ids: &[i64],
    ) -> Result<CommitIds> {
        Err(DuckLakeError::InvalidConfig(
            "register_data_files_with_deletes is not supported by this metadata writer".to_string(),
        ))
    }

    /// Registers multiple data files with positional deletes and commit metadata.
    ///
    /// The default implementation accepts empty metadata and otherwise returns
    /// an error. Metadata backends that support these fields override it.
    #[allow(clippy::too_many_arguments)]
    fn register_data_files_with_deletes_and_commit_metadata(
        &self,
        table_id: i64,
        schema_name: &str,
        table_name: &str,
        snapshot_id: i64,
        files: &[DataFileInfo],
        deletes: &[DeleteFileEntry],
        mode: WriteMode,
        base_snapshot: i64,
        columns: &[ColumnDef],
        column_ids: &[i64],
        commit_metadata: &SnapshotCommitMetadata,
        expected_base_snapshot_id: Option<i64>,
    ) -> Result<CommitIds> {
        if expected_base_snapshot_id.is_some() {
            return Err(DuckLakeError::InvalidConfig(
                "conditional combined multi-file writes are not supported by this metadata writer"
                    .to_string(),
            ));
        }
        commit_metadata.ensure_supported_by_default()?;
        self.register_data_files_with_deletes(
            table_id,
            schema_name,
            table_name,
            snapshot_id,
            files,
            deletes,
            mode,
            base_snapshot,
            columns,
            column_ids,
        )
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

    /// End visible inlined rows in one new snapshot. Implementations must fence
    /// every row against `base_snapshot`, set `end_snapshot` only while it is
    /// still live, and decrement the current table row count atomically.
    fn commit_inlined_deletes(
        &self,
        _table_id: i64,
        _schema_name: &str,
        _table_name: &str,
        _base_snapshot: i64,
        _rows: &[InlinedRowRef],
    ) -> Result<CommitIds> {
        Err(DuckLakeError::InvalidConfig(
            "inlined-row DELETE is not supported on this metadata backend".to_string(),
        ))
    }

    /// Apply positional and inlined-row deletes in one snapshot. The default
    /// dispatches when only one storage form is present; backends supporting a
    /// table that mixes both forms override this for an atomic combined commit.
    fn commit_deletes(
        &self,
        table_id: i64,
        schema_name: &str,
        table_name: &str,
        base_snapshot: i64,
        positional: &[DeleteFileEntry],
        inlined: &[InlinedRowRef],
    ) -> Result<CommitIds> {
        match (positional.is_empty(), inlined.is_empty()) {
            (false, true) => self.commit_positional_deletes(
                table_id,
                schema_name,
                table_name,
                base_snapshot,
                positional,
            ),
            (true, false) => self.commit_inlined_deletes(
                table_id,
                schema_name,
                table_name,
                base_snapshot,
                inlined,
            ),
            (false, false) => Err(DuckLakeError::InvalidConfig(
                "combined positional and inlined-row DELETE is not supported on this metadata backend"
                    .to_string(),
            )),
            (true, true) => Err(DuckLakeError::InvalidConfig(
                "commit_deletes requires at least one delete".to_string(),
            )),
        }
    }

    /// Commit a compaction (`merge_adjacent_files` / `rewrite_data_files`) in
    /// ONE new snapshot: register the rewritten `outputs`, retire every `source`
    /// data file AND its live delete file, recompute the table's visible stat
    /// totals from the surviving files, and record `changes_made =
    /// compacted_table:<id>` in `ducklake_snapshot_changes`.
    ///
    /// `retirement` decides how the sources are retired:
    /// [`SourceRetirement::Remove`] (merge — the partial output covers every
    /// snapshot, so remove the source rows AND schedule their files for
    /// deletion) or [`SourceRetirement::Retire`] (rewrite — the sources still
    /// serve time travel, so set `end_snapshot` but do NOT schedule them).
    /// Either way, no file is physically deleted here.
    ///
    /// Compaction changes the physical file layout, not the logical rows, so the
    /// commit is designed NOT to conflict with a concurrent append (which adds
    /// unrelated files): for each `source`, the data file must still be live,
    /// the live delete file must still match
    /// [`CompactionSourceFile::delete_file_id`] (a compare-and-swap), and no new
    /// distinct inlined-delete position may have appeared after `base_snapshot`.
    /// A concurrent Replace/compaction that retired the file or a concurrent
    /// DELETE/UPDATE (positional or inlined) that changed its live rows since
    /// they were read — aborts with [`crate::DuckLakeError::Conflict`] so
    /// retired rows can never be resurrected into an output. The new snapshot carries `schema_version`
    /// forward (compaction is not DDL). `base_snapshot` is the catalog head the
    /// sources were read at, used only for the conflict diagnostic.
    ///
    /// Each output is registered with `end_snapshot` NULL, `row_id_start` NULL
    /// (rowids come from the embedded column), its
    /// [`CompactionOutputFile::partial_max`], and `begin_snapshot` =
    /// [`CompactionOutputFile::begin_snapshot`] (or the new snapshot when that is
    /// `None`).
    ///
    /// Default: unsupported; the SQLite and Postgres backends override it.
    fn commit_compaction(
        &self,
        _table_id: i64,
        _base_snapshot: i64,
        _sources: &[CompactionSourceFile],
        _outputs: &[CompactionOutputFile],
        _retirement: SourceRetirement,
    ) -> Result<CommitIds> {
        Err(DuckLakeError::InvalidConfig(
            "compaction is not supported on this metadata backend".to_string(),
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

    /// Roll back a *pure-append* delta committed after `base_snapshot`: end
    /// (retire) every live data file whose `begin_snapshot > base_snapshot` — the
    /// files an append added on top of `base_snapshot` — in ONE new snapshot,
    /// WITHOUT rewriting data, returning the table's live content to exactly what
    /// it was at `base_snapshot`. This is the commit behind undoing an append that
    /// committed its DuckLake snapshot but whose post-commit step (e.g. a search
    /// index build) failed before the higher layer published it: the appended rows
    /// sit at the catalog head, unreferenced by any published generation, and a
    /// blind retry would stack a second copy on them. Retiring them here leaves a
    /// clean base so the retry appends exactly once.
    ///
    /// Refuses with [`crate::DuckLakeError::Conflict`] if the delta since
    /// `base_snapshot` is NOT a pure append — specifically if any of these exist
    /// for the table: a delete file with `begin_snapshot > base_snapshot`, a data
    /// file present at/before `base_snapshot` that was ended after it
    /// (`begin_snapshot <= base_snapshot AND end_snapshot > base_snapshot`), or a
    /// column version changed after it (`begin_snapshot > base_snapshot OR
    /// end_snapshot > base_snapshot`). Forward-only file retirement cannot
    /// faithfully revert a delete / replace / update / schema-promotion, so in
    /// those cases the caller must keep its read freeze and surface the state
    /// rather than expose a half-reverted table. The purity checks run BEFORE the
    /// no-op return, so a delete-only orphan (no appended data file) still yields
    /// `Conflict`, not a silent no-op.
    ///
    /// The guard covers the snapshot-visible change tables that can make a delta
    /// non-append: data files, delete files, and columns. Per-file partition values
    /// need no check — they hang off `data_file_id`, so retiring a file leaves its
    /// values attached exactly as any other retired file's are. This crate writes no
    /// inlined-data rows, so there is nothing else to check; if inlined-data support
    /// is added, extend the guard to reject a post-base row there too.
    ///
    /// Not covered: a `SET`/`RESET PARTITIONED BY` committed after `base_snapshot`.
    /// It changes no column, so the delta still reads as a pure append and the
    /// appended files are retired — leaving the new spec in place with no data under
    /// it. That is a consistent state (the retry re-appends under the live spec), but
    /// it is not a *revert* to `base_snapshot`; a caller needing exact reversion must
    /// serialize partition DDL against this, as the contract below already requires
    /// for writes.
    ///
    /// Recomputes the visible stat totals (`record_count`, `file_size_bytes`, and
    /// the per-column stats) from the surviving live files, and preserves
    /// `next_row_id` (rowids stay monotonic — retired ranges are never reused).
    /// Advances the catalog head LAST. Returns the new snapshot id, or `None` if no
    /// appended files exist (a no-op — no snapshot is created).
    ///
    /// # Caller contract
    /// `base_snapshot` MUST be a real `ducklake_snapshot.snapshot_id` — the table's
    /// current published generation — because `begin_snapshot > base_snapshot` is
    /// what distinguishes the orphaned append from published data. The caller MUST
    /// serialize writes to the table (e.g. a per-table write lock) so no concurrent
    /// legitimate append has a live file with `begin_snapshot > base_snapshot` that
    /// this would wrongly retire.
    ///
    /// Default: unsupported; the SQLite and Postgres backends override it.
    fn retire_appends_since(&self, _table_id: i64, _base_snapshot: i64) -> Result<Option<i64>> {
        Err(DuckLakeError::InvalidConfig(
            "retire_appends_since is not supported on this metadata backend".to_string(),
        ))
    }

    /// Register a data file that already carries DuckLake field-ids — e.g. a
    /// parquet copied verbatim from another catalog — adopting `column_ids` as
    /// the destination table's column ids so the file's embedded field-ids
    /// resolve on read. Self-contained (no `begin_write_transaction`): creates
    /// the table/columns on first write, appends after.
    ///
    /// `column_ids` must be non-empty and 1:1 with the recursive catalog nodes
    /// flattened from `columns`, in depth-first preorder. A mismatch is rejected
    /// with [`crate::DuckLakeError::InvalidConfig`]. Rowids are freshly assigned
    /// (the source `row_id_start` is not preserved), so indexes keyed on the
    /// source's rowids do not carry over.
    ///
    /// # Partitioning
    ///
    /// Nothing is rewritten here, so a partition assignment can only be *carried*,
    /// not derived: set it with [`DataFileInfo::with_partition`] and it is persisted
    /// (`ducklake_data_file.partition_id` + `ducklake_file_partition_value`).
    /// Promoting into a partitioned table without it is refused by the partition
    /// fence — an unpartitioned file cannot satisfy a live spec.
    ///
    /// The caller asserts the file holds rows of exactly ONE partition; only that
    /// caller can know, since the values are not checked against the file's contents.
    /// Official DuckLake's `ducklake_add_data_files` works the same way, reading the
    /// values from the file's Hive path and validating only their shape. What IS
    /// checked here is the shape against the live spec (see
    /// `validate_promoted_partition_values`).
    ///
    /// Two ways to satisfy the contract safely: copy the values from the source
    /// catalog when promoting files that a partitioned DuckLake table already wrote
    /// (they are single-partition by construction), or, for a file of unknown
    /// provenance, check the parquet footer's per-column min/max first — for an
    /// `identity` key, `min == max == value` proves the file is single-partition
    /// without reading a row.
    ///
    /// Promoting into a partitioned table WITHOUT an assignment is refused with a
    /// message naming the fix — not the partition fence's concurrency wording, which
    /// would misdescribe what went wrong.
    ///
    /// # Sort order
    ///
    /// A table's sort order is NOT enforced here: promoting an unsorted file into a
    /// sorted table is allowed, not an error. Sort order only affects how tight a
    /// file's min/max statistics are — an unsorted file reads back correctly, it just
    /// prunes less well, and a later compaction re-sorts it. Contrast partitioning,
    /// where a wrong value makes the read path prune away live rows, which is why
    /// that IS enforced. Official DuckLake's `ducklake_add_data_files` likewise
    /// ignores sort order entirely.
    ///
    /// Default: unsupported; only multicatalog Postgres, whose column ids are
    /// reusable across catalogs, implements it.
    #[allow(clippy::too_many_arguments)]
    fn register_existing_data_file(
        &self,
        _schema_name: &str,
        _table_name: &str,
        _columns: &[ColumnDef],
        _column_ids: &[i64],
        _file: &DataFileInfo,
        _mode: WriteMode,
    ) -> Result<CommitIds> {
        Err(DuckLakeError::InvalidConfig(
            "register_existing_data_file is not supported by this metadata writer".to_string(),
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

    /// Whether this backend supports row-level `UPDATE` (append the rewritten
    /// rows + apply positional deletes in one snapshot via
    /// [`register_data_file_with_deletes`](MetadataWriter::register_data_file_with_deletes)).
    ///
    /// Default `false`. Backends that implement the atomic append-with-deletes
    /// commit (SQLite, multicatalog Postgres) override it to `true`. The `UPDATE`
    /// planner path (`DuckLakeTable::update`) checks this up front and returns a
    /// clean "not supported" error for backends that don't (DuckDB, MySQL),
    /// rather than doing the file rewrites and only failing at commit.
    fn supports_update(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DuckLakeError;
    use arrow::datatypes::Field;
    use std::sync::Arc;

    fn promoted(values: Vec<(i32, Option<String>)>) -> DataFileInfo {
        DataFileInfo::new("f.parquet", 1024, 10).with_partition(7, values)
    }

    /// Key column types for a spec whose keys are all string-typed — the permissive
    /// case, so these tests exercise arity/index/transform rules in isolation.
    fn utf8_types(n: usize) -> Vec<Option<DataType>> {
        vec![Some(DataType::Utf8); n]
    }

    #[test]
    fn scalar_inlining_support_matches_round_trip_contract() {
        for data_type in [
            DataType::Int8,
            DataType::Float64,
            DataType::Decimal128(38, 16),
            DataType::Date32,
            DataType::Time64(arrow::datatypes::TimeUnit::Microsecond),
            DataType::Timestamp(arrow::datatypes::TimeUnit::Nanosecond, Some("UTC".into())),
            DataType::Interval(arrow::datatypes::IntervalUnit::MonthDayNano),
            DataType::LargeUtf8,
            DataType::LargeBinary,
            DataType::FixedSizeBinary(16),
        ] {
            assert!(scalar_type_supports_inlining(&data_type), "{data_type}");
        }
        assert!(!scalar_type_supports_inlining(&DataType::Time64(
            arrow::datatypes::TimeUnit::Nanosecond,
        )));
        assert!(!scalar_type_supports_inlining(&DataType::List(Arc::new(
            Field::new("item", DataType::Int64, true),
        ))));
    }

    #[test]
    fn promoted_values_must_match_the_live_key_count() {
        let transforms = vec!["identity".to_string(), "year".to_string()];
        // One value for a two-key spec: the second key would silently have none.
        let err = validate_promoted_partition_values(
            1,
            &transforms,
            &utf8_types(transforms.len()),
            &promoted(vec![(0, Some("us".into()))]),
        )
        .unwrap_err();
        assert!(matches!(err, DuckLakeError::InvalidConfig(_)), "got {err}");
        // Correct count passes.
        assert!(
            validate_promoted_partition_values(
                1,
                &transforms,
                &utf8_types(transforms.len()),
                &promoted(vec![(0, Some("us".into())), (1, Some("2024".into()))]),
            )
            .is_ok()
        );
    }

    #[test]
    fn promoted_values_reject_duplicate_or_out_of_range_key_index() {
        let transforms = vec!["identity".to_string(), "year".to_string()];
        // Same key twice: one spec key ends up unassigned.
        assert!(
            validate_promoted_partition_values(
                1,
                &transforms,
                &utf8_types(transforms.len()),
                &promoted(vec![(0, Some("us".into())), (0, Some("eu".into()))]),
            )
            .is_err()
        );
        // Index past the end of the spec.
        assert!(
            validate_promoted_partition_values(
                1,
                &transforms,
                &utf8_types(transforms.len()),
                &promoted(vec![(0, Some("us".into())), (5, Some("2024".into()))]),
            )
            .is_err()
        );
    }

    #[test]
    fn promoted_bucket_values_must_be_in_range() {
        let transforms = vec!["bucket(8)".to_string()];
        assert!(
            validate_promoted_partition_values(
                1,
                &transforms,
                &utf8_types(transforms.len()),
                &promoted(vec![(0, Some("3".into()))])
            )
            .is_ok()
        );
        for bad in ["8", "-1", "abc"] {
            assert!(
                validate_promoted_partition_values(
                    1,
                    &transforms,
                    &utf8_types(transforms.len()),
                    &promoted(vec![(0, Some(bad.to_string()))]),
                )
                .is_err(),
                "bucket value {bad} must be rejected"
            );
        }
        // NULL IS valid for a bucket key: a NULL input yields a NULL partition
        // value, which official accepts too (IsValidTransformedHivePartitionValue
        // returns early on a NULL hive value, before its range check).
        assert!(
            validate_promoted_partition_values(
                1,
                &transforms,
                &utf8_types(transforms.len()),
                &promoted(vec![(0, None)])
            )
            .is_ok()
        );
    }

    #[test]
    fn promoted_identity_value_must_cast_to_the_key_column_type() {
        let transforms = vec!["identity".to_string()];
        let int_key = vec![Some(DataType::Int32)];
        // A value that cannot be an Int32 partition key is impossible for the file to
        // hold, and would be persisted then used as an EXACT pruning bound.
        let err = validate_promoted_partition_values(
            1,
            &transforms,
            &int_key,
            &promoted(vec![(0, Some("abc".into()))]),
        )
        .unwrap_err();
        assert!(matches!(err, DuckLakeError::InvalidConfig(_)), "got {err}");
        // A well-formed integer passes, as does NULL.
        for value in [Some("42".to_string()), None] {
            assert!(
                validate_promoted_partition_values(
                    1,
                    &transforms,
                    &int_key,
                    &promoted(vec![(0, value.clone())]),
                )
                .is_ok(),
                "value {value:?} must be accepted for an Int32 identity key"
            );
        }
    }

    #[test]
    fn promoted_temporal_value_must_parse_as_an_integer_and_no_more() {
        // Official types a temporal partition key as BIGINT and only casts
        // (GetPartitionKeyType / MapPartitionColumns) — it does NOT range-check the
        // calendar component. So a non-integer is rejected, but an out-of-range
        // month like 13 must be ACCEPTED, or we would refuse a value official takes.
        let transforms = vec!["month".to_string()];
        let date_key = vec![Some(DataType::Date32)];
        assert!(
            validate_promoted_partition_values(
                1,
                &transforms,
                &date_key,
                &promoted(vec![(0, Some("2024-06".into()))]),
            )
            .is_err(),
            "a non-integer month value must be rejected"
        );
        for accepted in ["6", "13", "0"] {
            assert!(
                validate_promoted_partition_values(
                    1,
                    &transforms,
                    &date_key,
                    &promoted(vec![(0, Some(accepted.to_string()))]),
                )
                .is_ok(),
                "month value {accepted} must be accepted (official does not range-check)"
            );
        }
    }

    #[test]
    fn promoted_null_value_is_legal_for_identity() {
        // A NULL partition value is a legitimate partition (DuckDB's
        // __HIVE_DEFAULT_PARTITION__), so it must pass for a non-bucket key.
        let transforms = vec!["identity".to_string()];
        assert!(
            validate_promoted_partition_values(
                1,
                &transforms,
                &utf8_types(transforms.len()),
                &promoted(vec![(0, None)])
            )
            .is_ok()
        );
    }

    #[test]
    fn fence_exempts_empty_file_but_rejects_rows_without_partition() {
        // A 0-row Replace truncate marker carries no partitioned data.
        let empty = DataFileInfo::new("empty.parquet", 0, 0);
        assert!(enforce_partition_fence(1, Some(7), &empty).is_ok());
        // A row-bearing file with no partition cannot live in a partitioned table.
        let rows = DataFileInfo::new("f.parquet", 1024, 10);
        assert!(matches!(
            enforce_partition_fence(1, Some(7), &rows),
            Err(DuckLakeError::Conflict(_))
        ));
        // ...but is fine when the table has no live spec.
        assert!(enforce_partition_fence(1, None, &rows).is_ok());
        // A file stamped with a retired generation is rejected.
        assert!(matches!(
            enforce_partition_fence(1, Some(9), &promoted(vec![(0, Some("us".into()))])),
            Err(DuckLakeError::Conflict(_))
        ));
    }

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
    fn test_catalog_column_defs_use_depth_first_preorder() {
        let levels = DataType::List(Arc::new(Field::new(
            "item",
            DataType::Struct(
                vec![
                    Arc::new(Field::new("price", DataType::Decimal128(38, 16), false)),
                    Arc::new(Field::new(
                        "tags",
                        DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
                        true,
                    )),
                ]
                .into(),
            ),
            false,
        )));
        let attrs = DataType::Map(
            Arc::new(Field::new(
                "entries",
                DataType::Struct(
                    vec![
                        Arc::new(Field::new("key", DataType::Utf8, false)),
                        Arc::new(Field::new(
                            "value",
                            DataType::Struct(
                                vec![Arc::new(Field::new("active", DataType::Boolean, false))]
                                    .into(),
                            ),
                            true,
                        )),
                    ]
                    .into(),
                ),
                false,
            )),
            false,
        );
        let columns = vec![
            ColumnDef::from_arrow("id", &DataType::Int32, false)
                .unwrap()
                .with_default("7")
                .unwrap(),
            ColumnDef::from_arrow("levels", &levels, false).unwrap(),
            ColumnDef::from_arrow("attrs", &attrs, true).unwrap(),
        ];

        let definitions = catalog_column_defs(&columns).unwrap();
        let actual = definitions
            .iter()
            .map(|column| {
                (
                    column.name.as_str(),
                    column.ducklake_type.as_str(),
                    column.is_nullable,
                    column.parent_index,
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            actual,
            vec![
                ("id", "int32", false, None),
                ("levels", "list", false, None),
                ("element", "struct", false, Some(1)),
                ("price", "decimal(38, 16)", false, Some(2)),
                ("tags", "list", true, Some(2)),
                ("element", "varchar", true, Some(4)),
                ("attrs", "map", true, None),
                ("key", "varchar", false, Some(6)),
                ("value", "struct", true, Some(6)),
                ("active", "boolean", false, Some(8)),
            ]
        );
        assert_eq!(
            top_level_column_ids(&definitions, &[10, 11, 12, 13, 14, 15, 16, 17, 18, 19]).unwrap(),
            vec![10, 11, 16]
        );
        let defaults = definitions
            .iter()
            .enumerate()
            .filter_map(|(index, column)| {
                column.default_value.as_deref().map(|value| (index, value))
            })
            .collect::<Vec<_>>();
        assert_eq!(defaults, vec![(0, "7")]);
    }

    #[test]
    fn test_assign_column_ids_rejects_duplicate_paths() {
        let columns = vec![
            ColumnDef::from_arrow("value", &DataType::Int32, false).unwrap(),
            ColumnDef::from_arrow("value", &DataType::Int64, false).unwrap(),
        ];
        let definitions = catalog_column_defs(&columns).unwrap();

        assert!(assign_column_ids(&definitions, &[], &[10, 11]).is_err());
    }

    #[test]
    fn catalog_type_name_returns_unsupported_scalar_error() {
        let error =
            catalog_type_name(&DataType::Duration(arrow::datatypes::TimeUnit::Second)).unwrap_err();

        assert!(matches!(error, DuckLakeError::UnsupportedType(_)));
    }

    #[test]
    fn test_catalog_columns_differ_accepts_nested_type_alias() {
        let columns = vec![
            ColumnDef::from_arrow(
                "values",
                &DataType::List(Arc::new(Field::new("item", DataType::Int64, true))),
                false,
            )
            .unwrap(),
        ];
        let definitions = catalog_column_defs(&columns).unwrap();
        let existing = vec![
            ExistingCatalogColumn {
                column_id: 10,
                name: "values".into(),
                ducklake_type: "list".into(),
                parent_column: None,
            },
            ExistingCatalogColumn {
                column_id: 11,
                name: "element".into(),
                ducklake_type: "bigint".into(),
                parent_column: Some(10),
            },
        ];

        assert!(!catalog_columns_differ(
            &existing,
            &[false, true],
            &definitions,
            &[10, 11],
        ));
    }

    #[test]
    fn legacy_list_type_matches_only_the_same_recursive_type() {
        let columns = vec![
            ColumnDef::from_arrow(
                "values",
                &DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
                true,
            )
            .unwrap(),
        ];
        let definitions = catalog_column_defs(&columns).unwrap();

        assert!(catalog_column_type_equal("list<float32>", &definitions[0]));
        assert!(catalog_column_type_requires_migration(
            "list<float32>",
            &definitions[0]
        ));
        assert!(!catalog_column_type_equal("list<int32>", &definitions[0]));
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
