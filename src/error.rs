//! Error types for the DuckLake DataFusion extension

use std::fmt;

use thiserror::Error;

/// The data-write mode that attempted an unsupported column type change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeChangeWriteMode {
    /// Drop existing data and replace with new data.
    Replace,
    /// Keep existing data and append new records.
    Append,
}

impl fmt::Display for TypeChangeWriteMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TypeChangeWriteMode::Replace => write!(f, "Replace"),
            TypeChangeWriteMode::Append => write!(f, "Append"),
        }
    }
}

/// The operation that attempted an unsupported column type change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeChangeOperation {
    /// Explicit metadata-only schema evolution through `promote_column_type`.
    PromoteColumnType,
    /// A data write tried to change the type of an existing same-name column.
    DataWrite {
        mode: TypeChangeWriteMode,
    },
}

impl fmt::Display for TypeChangeOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TypeChangeOperation::PromoteColumnType => write!(f, "promote_column_type"),
            TypeChangeOperation::DataWrite {
                mode,
            } => write!(f, "{mode} data write"),
        }
    }
}

/// Error type for DuckLake operations
#[derive(Error, Debug)]
pub enum DuckLakeError {
    /// Error from DataFusion
    #[error("DataFusion error: {0}")]
    DataFusion(#[from] datafusion::error::DataFusionError),

    /// Error from Arrow
    #[error("Arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),

    /// DuckDB error
    #[cfg(feature = "metadata-duckdb")]
    #[error("DuckDB error: {0}")]
    DuckDb(#[from] duckdb::Error),

    /// sqlx database error (for PostgreSQL/MySQL/SQLite metadata providers)
    #[cfg(any(
        feature = "metadata-postgres",
        feature = "metadata-mysql",
        feature = "metadata-sqlite"
    ))]
    #[error("Database error: {0}")]
    Sqlx(#[from] sqlx::Error),

    /// Catalog not found
    #[error("Catalog not found: {0}")]
    CatalogNotFound(String),

    /// Schema not found
    #[error("Schema not found: {0}")]
    SchemaNotFound(String),

    /// Table not found
    #[error("Table not found: {0}")]
    TableNotFound(String),

    /// Invalid snapshot
    #[error("Invalid snapshot: {0}")]
    InvalidSnapshot(String),

    /// Invalid catalog configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// A write or promotion tried to change an existing column to a type that
    /// DuckLake cannot adopt through metadata-only schema evolution.
    #[error(
        "Unsupported type change during {operation}: column '{column}' from '{from}' to '{to}'"
    )]
    UnsupportedTypeChange {
        operation: TypeChangeOperation,
        column: String,
        from: String,
        to: String,
    },

    /// Unsupported DuckLake type
    #[error("Unsupported DuckLake type: {0}")]
    UnsupportedType(String),

    /// Unsupported feature
    #[error("Unsupported feature: {0}")]
    Unsupported(String),

    /// ObjectStore error
    #[error("ObjectStore error: {0}")]
    ObjectStore(#[from] object_store::Error),

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Parquet error
    #[error("Parquet error: {0}")]
    Parquet(#[from] parquet::errors::ParquetError),

    /// A concurrent write conflict detected at commit time: another writer
    /// published a newer generation of the table since this write began. The
    /// loser aborts (DuckLake-style optimistic concurrency) rather than silently
    /// unioning or clobbering the concurrent commit. Callers may retry.
    #[error("Write conflict: {0}")]
    Conflict(String),

    /// Generic error
    #[error("Internal error: {0}")]
    Internal(String),
}

impl From<DuckLakeError> for datafusion::error::DataFusionError {
    fn from(err: DuckLakeError) -> Self {
        match err {
            // If it's already a DataFusion error, unwrap it
            DuckLakeError::DataFusion(e) => e,
            // For all other errors, wrap them as External
            other => datafusion::error::DataFusionError::External(Box::new(other)),
        }
    }
}
