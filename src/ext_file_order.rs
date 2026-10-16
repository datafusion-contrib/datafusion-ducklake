//! EXPERIMENTAL, opt-in: the `ext_file_order` sidecar catalog table.
//!
//! One row per data file whose rows were verified to follow a sort key:
//!
//! ```sql
//! CREATE TABLE ext_ducklake_file_order (
//!     table_id        BIGINT NOT NULL,
//!     path            VARCHAR NOT NULL,   -- ducklake_data_file.path
//!     file_size_bytes BIGINT NOT NULL,    -- ducklake_data_file.file_size_bytes
//!     sort_key        VARCHAR NOT NULL,   -- crate::file_ordering format, by field id
//!     verified_by     VARCHAR NOT NULL,   -- 'write' (the writer) | 'verify' (the job)
//!     verified_at     TIMESTAMP NOT NULL,
//!     PRIMARY KEY (table_id, path)
//! )
//! ```
//!
//! Outside the spec's `ducklake_*` namespace on purpose: DuckLake clients
//! (the DuckDB extension included) only touch the tables they name, so they
//! neither read nor disturb it, and no future spec table can collide with it.
//!
//! A row states a fact about the parquet object at `path`, which DuckLake
//! never rewrites in place, so it stays true whatever snapshot reads it and
//! needs no transaction with the catalog. It is keyed by path, not by
//! `data_file_id`: an id names a catalog row, and a writer that reuses ids
//! (SQLite row ids before the shared file-id counter) would hand a stale row
//! to an unrelated file. A file applies a row only when its table, path and
//! size all match, so a row can never vouch for any other file. A missing row
//! means "unknown" and the file is scanned as it always was. A row whose file
//! never reached, or has left, the catalog is harmless (no file lists that
//! path), and `MetadataWriter::prune_file_orders` drops them.

/// The sidecar table's name.
pub const TABLE: &str = "ext_ducklake_file_order";

/// One sidecar row: the verified sort key of the parquet file at `path`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileOrder {
    /// The data file's catalog path (`ducklake_data_file.path`).
    pub path: String,
    /// The data file's size; a file of another size at `path` is not this one.
    pub file_size_bytes: i64,
    /// The verified key, `field_id:asc|desc:nulls_first|nulls_last`, comma
    /// separated, primary key first.
    pub sort_key: String,
}

#[cfg(feature = "metadata-sqlite")]
pub(crate) const SQLITE_DDL: &str = "CREATE TABLE IF NOT EXISTS ext_ducklake_file_order (
    table_id BIGINT NOT NULL,
    path VARCHAR NOT NULL,
    file_size_bytes BIGINT NOT NULL,
    sort_key VARCHAR NOT NULL,
    verified_by VARCHAR NOT NULL,
    verified_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (table_id, path)
)";

#[cfg(feature = "metadata-postgres")]
pub(crate) const POSTGRES_DDL: &str = "CREATE TABLE IF NOT EXISTS ext_ducklake_file_order (
    table_id BIGINT NOT NULL,
    path VARCHAR NOT NULL,
    file_size_bytes BIGINT NOT NULL,
    sort_key VARCHAR NOT NULL,
    verified_by VARCHAR NOT NULL,
    verified_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (table_id, path)
)";

/// Recorded by the writer, for a file it wrote.
pub(crate) const BY_WRITE: &str = "write";
/// Recorded by `DuckLakeTable::verify_file_orders`, after reading the file.
pub(crate) const BY_VERIFY: &str = "verify";

/// Whether `error` says the sidecar table does not exist (never created: the
/// option was never turned on for this catalog), which reads as "no rows".
#[cfg(any(feature = "metadata-sqlite", feature = "metadata-postgres"))]
pub(crate) fn is_missing_table(error: &sqlx::Error) -> bool {
    match error {
        sqlx::Error::Database(db) => {
            db.code().as_deref() == Some("42P01") || db.message().contains("no such table")
        },
        _ => false,
    }
}
