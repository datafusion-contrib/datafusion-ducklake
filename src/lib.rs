//! # DataFusion-DuckLake
//!
//! A DataFusion extension that adds support for DuckLake, an integrated data lake and catalog format.
//!
//! ## Overview
//!
//! DuckLake uses:
//! - **Catalog Database**: SQL database (DuckDB, SQLite, PostgreSQL, MySQL) storing metadata as SQL tables
//! - **Data Storage**: Apache Parquet files stored on disk/object storage
//!
//! This extension provides read and write access to DuckLake catalogs through DataFusion's
//! catalog and table provider interfaces. Writes are feature-gated (`write-sqlite`,
//! `write-postgres`); see `COMPATIBILITY.md` for the full backend/feature matrix.
//!
//! ## Example
//!
//! ```no_run
//! # async fn example() -> datafusion_ducklake::Result<()> {
//! use datafusion::prelude::*;
//! use datafusion_ducklake::{DuckLakeCatalog, DuckdbMetadataProvider};
//!
//! // Create a DataFusion session context
//! let ctx = SessionContext::new();
//!
//! // Create a DuckDB metadata provider
//! let provider = DuckdbMetadataProvider::new("path/to/catalog.ducklake")?;
//!
//! // Register a DuckLake catalog with the provider
//! let catalog = DuckLakeCatalog::new(provider)?;
//! ctx.register_catalog("ducklake", std::sync::Arc::new(catalog));
//!
//! // Query tables from the catalog
//! let df = ctx.sql("SELECT * FROM ducklake.main.my_table").await?;
//! df.show().await?;
//! # Ok(())
//! # }
//! ```

pub mod catalog;
pub mod column_rename;
pub mod delete_filter;
pub mod encryption;
pub mod error;
pub mod information_schema;
pub mod metadata_provider;
pub(crate) mod nan_pruning_barrier;
pub mod partition;
pub mod path_resolver;
pub(crate) mod positional_source;
pub mod row_id;
pub mod schema;
pub mod snapshot_filter;
pub mod table;
pub mod table_changes;
pub mod table_deletions;
pub mod table_functions;
pub mod types;

// Metadata providers (feature-gated)
#[cfg(feature = "metadata-duckdb")]
pub mod metadata_provider_duckdb;
#[cfg(feature = "metadata-mysql")]
pub mod metadata_provider_mysql;
#[cfg(feature = "metadata-postgres")]
pub mod metadata_provider_postgres;
#[cfg(feature = "metadata-sqlite")]
pub mod metadata_provider_sqlite;

// Write support (feature-gated)
#[cfg(feature = "write")]
pub mod compaction;
#[cfg(feature = "write")]
pub mod delete_exec;
#[cfg(feature = "write")]
pub mod insert_exec;
#[cfg(feature = "write")]
pub mod maintenance;
#[cfg(feature = "write")]
pub mod metadata_writer;
#[cfg(feature = "write-duckdb")]
pub mod metadata_writer_duckdb;
#[cfg(feature = "write-mysql")]
pub mod metadata_writer_mysql;
#[cfg(feature = "write-postgres")]
pub mod metadata_writer_postgres;
#[cfg(feature = "write-sqlite")]
pub mod metadata_writer_sqlite;
#[cfg(feature = "write-postgres")]
pub mod multicatalog;
#[cfg(feature = "multicatalog-postgres")]
pub mod multicatalog_provider;
#[cfg(feature = "write")]
pub mod sql;
#[cfg(feature = "write")]
pub mod stats_collect;
#[cfg(feature = "write")]
pub mod stats_encode;
#[cfg(feature = "write")]
pub mod table_writer;
#[cfg(feature = "write")]
pub mod update_exec;

// Result type for DuckLake operations
pub type Result<T> = std::result::Result<T, DuckLakeError>;

// Re-export main types for convenience
pub use catalog::DuckLakeCatalog;
pub use error::{DuckLakeError, TypeChangeOperation, TypeChangeWriteMode};
pub use metadata_provider::{DuckLakeFileData, DuckLakeTableFile, MetadataProvider};
pub use partition::{PartitionSpec, PartitionSpecColumn, PartitionTransform};
pub use schema::DuckLakeSchema;
pub use table::DuckLakeTable;
pub use table_functions::register_ducklake_functions;

// Re-export metadata providers (feature-gated)
#[cfg(feature = "metadata-duckdb")]
pub use metadata_provider_duckdb::DuckdbMetadataProvider;
#[cfg(feature = "metadata-mysql")]
pub use metadata_provider_mysql::MySqlMetadataProvider;
#[cfg(feature = "metadata-postgres")]
pub use metadata_provider_postgres::PostgresMetadataProvider;
#[cfg(feature = "metadata-sqlite")]
pub use metadata_provider_sqlite::SqliteMetadataProvider;

// Re-export write types (feature-gated)
#[cfg(feature = "write")]
pub use compaction::{CompactionResult, MergeOptions, RewriteOptions};
#[cfg(feature = "write")]
pub use delete_exec::DuckLakeDeleteExec;
#[cfg(feature = "write")]
pub use insert_exec::DuckLakeInsertExec;
#[cfg(feature = "write")]
pub use metadata_writer::{
    ColumnDef, ColumnStat, CommitIds, CompactionOutputFile, CompactionSourceFile, DataFileInfo,
    DeleteFileEntry, DeleteFileInfo, MetadataWriter, SourceRetirement, WriteMode, WriteResult,
    WriteSetupResult,
};
#[cfg(feature = "write-duckdb")]
pub use metadata_writer_duckdb::DuckdbMetadataWriter;
#[cfg(feature = "write-mysql")]
pub use metadata_writer_mysql::MySqlMetadataWriter;
#[cfg(feature = "write-postgres")]
pub use metadata_writer_postgres::PostgresMetadataWriter;
#[cfg(feature = "write-sqlite")]
pub use metadata_writer_sqlite::SqliteMetadataWriter;
#[cfg(feature = "write-postgres")]
pub use multicatalog::{MulticatalogManager, initialize_multicatalog_schema};
#[cfg(feature = "multicatalog-postgres")]
pub use multicatalog_provider::MulticatalogProvider;
#[cfg(feature = "write")]
pub use sql::execute_ducklake_sql;
#[cfg(feature = "write")]
pub use table_writer::{DuckLakeTableWriter, TableWriteSession};
#[cfg(feature = "write")]
pub use update_exec::DuckLakeUpdateExec;
