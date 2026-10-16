//! EXPERIMENTAL, opt-in: the `ext_file_order` sidecar catalog table.
//!
//! One row per data file whose rows were verified to follow a sort key:
//!
//! ```sql
//! CREATE TABLE ext_ducklake_file_order (
//!     data_file_id BIGINT PRIMARY KEY,  -- ducklake_data_file.data_file_id
//!     table_id     BIGINT NOT NULL,
//!     sort_key     VARCHAR NOT NULL,    -- crate::file_ordering format, by field id
//!     verified_by  VARCHAR NOT NULL,    -- 'write' (the writer) | 'verify' (the job)
//!     verified_at  TIMESTAMP NOT NULL
//! )
//! ```
//!
//! Outside the spec's `ducklake_*` namespace on purpose: DuckLake clients
//! (the DuckDB extension included) only touch the tables they name, so they
//! neither read nor disturb it, and no future spec table can collide with it.
//! A data file is immutable and its id is never reused, so a row stays true for
//! as long as the file exists, whatever snapshot reads it; a missing row means
//! "unknown" and the file is scanned as it always was. A row for a file that
//! no longer exists is harmless (nothing lists that id), and
//! `MetadataWriter::prune_file_orders` drops them.

/// The sidecar table's name.
pub const TABLE: &str = "ext_ducklake_file_order";

#[cfg(feature = "metadata-sqlite")]
pub(crate) const SQLITE_DDL: &str = "CREATE TABLE IF NOT EXISTS ext_ducklake_file_order (
    data_file_id BIGINT PRIMARY KEY,
    table_id BIGINT NOT NULL,
    sort_key VARCHAR NOT NULL,
    verified_by VARCHAR NOT NULL,
    verified_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
)";

#[cfg(feature = "metadata-postgres")]
pub(crate) const POSTGRES_DDL: &str = "CREATE TABLE IF NOT EXISTS ext_ducklake_file_order (
    data_file_id BIGINT PRIMARY KEY,
    table_id BIGINT NOT NULL,
    sort_key VARCHAR NOT NULL,
    verified_by VARCHAR NOT NULL,
    verified_at TIMESTAMPTZ NOT NULL DEFAULT now()
)";

/// Recorded by the writer, in the file's own commit.
pub(crate) const BY_WRITE: &str = "write";
/// Recorded by `DuckLakeTable::verify_file_orders`, after reading the file.
pub(crate) const BY_VERIFY: &str = "verify";

/// Whether `error` says the sidecar table does not exist (never created: the
/// option was never turned on for this catalog), which reads as "no rows".
pub(crate) fn is_missing_table(error: &sqlx::Error) -> bool {
    match error {
        sqlx::Error::Database(db) => {
            db.code().as_deref() == Some("42P01") || db.message().contains("no such table")
        },
        _ => false,
    }
}
