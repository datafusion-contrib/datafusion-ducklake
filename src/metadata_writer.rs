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

/// Result of a transactional write setup operation.
#[derive(Debug)]
pub struct WriteSetupResult {
    /// Snapshot ID created for this write
    pub snapshot_id: i64,
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

    /// Register a new data file and publish its snapshot as the catalog head,
    /// atomically. For `Replace`, retires the prior generation in the same
    /// transaction. Returns the assigned data_file_id.
    fn register_data_file(
        &self,
        table_id: i64,
        snapshot_id: i64,
        file: &DataFileInfo,
        mode: WriteMode,
    ) -> Result<i64>;

    /// Publish a write's snapshot as the catalog head with no data file (CREATE
    /// TABLE, zero-row Replace). For `Replace`, retires the prior generation.
    ///
    /// Default no-op: single-catalog backends advance the head by inserting the
    /// snapshot row in `begin_write_transaction`. Multicatalog Postgres
    /// overrides this to insert the `ducklake_catalog_snapshot_map` head row.
    fn publish_snapshot(&self, _table_id: i64, _snapshot_id: i64, _mode: WriteMode) -> Result<()> {
        Ok(())
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
