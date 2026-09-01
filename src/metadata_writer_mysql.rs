//! MySQL implementation of [`MetadataWriter`].
//!
//! Single-catalog only — the legacy DuckLake v1.0 layout, mirroring
//! [`crate::metadata_writer_sqlite::SqliteMetadataWriter`] rather than the
//! multicatalog Postgres writer. It supports the crate's write primitives,
//! inlined data, positional deletes, updates, compaction, truncate, append
//! restoration, type promotion, partition and sort metadata, and snapshot
//! expiry. Multiple catalogs remain specific to PostgreSQL;
//! [`MetadataWriter::catalog_id`] inherits the `None` default and keeps file
//! paths in the `{data_path}/{schema}/{table}/…` layout.
//!
//! Requires a multi-threaded Tokio runtime (`#[tokio::test(flavor =
//! "multi_thread")]`): the sync trait methods bridge async sqlx via
//! `crate::metadata_provider::block_on`, exactly like the SQLite writer.
//!
//! ## MySQL dialect adaptations vs the SQLite template
//!
//! 1. **No `RETURNING`.** Auto-increment PK ids (`schema_id`, `table_id`) are
//!    read back with `MySqlQueryResult::last_insert_id()`; counter-allocated ids
//!    (`column_id`, `snapshot_id`, `data_file_id`, `delete_file_id`) are read
//!    back with an `UPDATE` followed by a `SELECT` in the same transaction
//!    (`reserve_ids`). File ids come exclusively from the counters — never from
//!    auto-increment — so every insert path shares one id space per table.
//! 2. **DDL type mapping.** `INTEGER`→`BIGINT`, bounded names→`VARCHAR(1024)`,
//!    long/path values→`TEXT`, `BOOLEAN`→`TINYINT(1)`. Every table is InnoDB so
//!    transactions + `SELECT … FOR UPDATE`-style row locks actually serialize.
//! 3. **Reserved words.** `ducklake_metadata`'s `key`/`value` columns are
//!    backticked everywhere.
//! 4. **`INSERT OR IGNORE`→`INSERT IGNORE`.**
//! 5. **No self-referential `INSERT … SELECT`.** MySQL rejects `INSERT INTO t …
//!    SELECT … FROM t` (error 1093), so `snapshot_id` is allocated from a
//!    monotonic counter row rather than `SELECT MAX(snapshot_id)+1`.

use crate::Result;
use crate::error::{TypeChangeOperation, TypeChangeWriteMode};
use crate::maintenance::{ExpireCriteria, ExpiredSnapshot, format_sql_timestamp};
use crate::metadata_provider::block_on;
use crate::metadata_writer::{
    ColumnDef, ColumnStat, CommitIds, CompactionOutputFile, CompactionSourceFile, DataFileInfo,
    DeleteFileEntry, DeleteFileInfo, ExistingCatalogColumn, INLINED_INDEX_COLUMNS_SETTING,
    InlinedRowRef, MetadataWriter, MultiTableCommit, SnapshotCommitMetadata, SourceRetirement,
    StagedTableData, StagedTableWrite, WriteMode, WriteSetupResult, assign_column_ids,
    catalog_column_defs, catalog_column_type_equal, catalog_column_type_requires_migration,
    catalog_columns_differ, encode_inlined_index_columns, live_inlined_index_columns,
    parse_inlined_index_columns, quote_snapshot_name, quote_snapshot_table, table_write_changes,
    top_level_column_ids, validate_delete_entries, validate_inlined_index_columns, validate_name,
};
use crate::partition::PartitionTransform;
use arrow::array::{
    Array, BinaryArray, BinaryViewArray, BooleanArray, FixedSizeBinaryArray, Float32Array,
    Float64Array, Int8Array, Int16Array, Int32Array, Int64Array, LargeBinaryArray,
    LargeStringArray, StringArray, TimestampNanosecondArray, UInt8Array, UInt16Array, UInt32Array,
    UInt64Array,
};
use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;
use sqlx::Row;
use sqlx::mysql::{MySql, MySqlPool, MySqlPoolOptions};
use sqlx::{AssertSqlSafe, QueryBuilder};

const DEFAULT_MAX_CONNECTIONS: u32 = 5;

fn id_list(ids: &[i64]) -> String {
    ids.iter()
        .map(i64::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

const RESOLVED_PATH: &str = "CASE
    WHEN NOT df.path_is_relative THEN df.path
    WHEN NOT t.path_is_relative THEN CONCAT(t.path, '/', df.path)
    ELSE CONCAT(s.path, '/', t.path, '/', df.path)
END";

const REL_FLAG: &str =
    "(CASE WHEN df.path_is_relative AND t.path_is_relative AND s.path_is_relative
           THEN 1 ELSE 0 END)";

fn quote_ident(name: &str) -> String {
    format!("`{}`", name.replace('`', "``"))
}

fn mysql_type_supports_index(physical_type: &str) -> bool {
    !matches!(
        physical_type.to_ascii_lowercase().as_str(),
        "longtext" | "longblob"
    )
}

fn inlined_mysql_type(data_type: &DataType) -> String {
    match data_type {
        DataType::Boolean
        | DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32 => "BIGINT".to_string(),
        DataType::UInt64 => "BIGINT UNSIGNED".to_string(),
        DataType::Float32 => "FLOAT".to_string(),
        DataType::Float64 => "DOUBLE".to_string(),
        DataType::Decimal32(precision, scale)
        | DataType::Decimal64(precision, scale)
        | DataType::Decimal128(precision, scale) => format!("DECIMAL({precision},{scale})"),
        DataType::Date32 => "DATE".to_string(),
        DataType::Time64(arrow::datatypes::TimeUnit::Microsecond) => "TIME(6)".to_string(),
        DataType::Timestamp(arrow::datatypes::TimeUnit::Nanosecond, _) => "BIGINT".to_string(),
        // DATETIME, not TIMESTAMP: TIMESTAMP is capped to [1970-01-01 00:00:01,
        // 2038-01-19] UTC and converts through the session time zone on
        // store/retrieve, so epoch-adjacent and post-2038 values would fail or
        // shift between sessions.
        DataType::Timestamp(_, _) => "DATETIME(6)".to_string(),
        DataType::Binary | DataType::LargeBinary | DataType::BinaryView => "LONGBLOB".to_string(),
        DataType::FixedSizeBinary(size) if *size != 16 => "LONGBLOB".to_string(),
        _ => "LONGTEXT".to_string(),
    }
}

fn mysql_type_inlines(data_type: &DataType) -> bool {
    crate::nested_inline::type_supports_inlining(data_type)
}

fn push_inlined_mysql_value(
    query: &mut QueryBuilder<MySql>,
    array: &dyn Array,
    row: usize,
    sql_type: &str,
) -> Result<()> {
    if array.is_null(row) {
        query.push_bind(Option::<String>::None);
        return Ok(());
    }
    macro_rules! signed {
        ($array:ty) => {{
            query.push_bind(
                array
                    .as_any()
                    .downcast_ref::<$array>()
                    .expect("Arrow data type and array implementation agree")
                    .value(row) as i64,
            );
        }};
    }
    macro_rules! unsigned {
        ($array:ty) => {{
            query.push_bind(
                array
                    .as_any()
                    .downcast_ref::<$array>()
                    .expect("Arrow data type and array implementation agree")
                    .value(row) as i64,
            );
        }};
    }
    match array.data_type() {
        DataType::Boolean => {
            let value = array
                .as_any()
                .downcast_ref::<BooleanArray>()
                .expect("Arrow data type and array implementation agree")
                .value(row);
            query.push_bind(i64::from(value));
        },
        DataType::Int8 => signed!(Int8Array),
        DataType::Int16 => signed!(Int16Array),
        DataType::Int32 => signed!(Int32Array),
        DataType::Int64 => signed!(Int64Array),
        DataType::UInt8 => unsigned!(UInt8Array),
        DataType::UInt16 => unsigned!(UInt16Array),
        DataType::UInt32 => unsigned!(UInt32Array),
        DataType::UInt64 => {
            let value = array
                .as_any()
                .downcast_ref::<UInt64Array>()
                .expect("Arrow data type and array implementation agree")
                .value(row);
            query
                .push("CAST(CAST(")
                .push_bind(value.to_string())
                .push(" AS DECIMAL(20,0)) AS UNSIGNED)");
        },
        DataType::Float32 => {
            let value = array
                .as_any()
                .downcast_ref::<Float32Array>()
                .expect("Arrow data type and array implementation agree")
                .value(row);
            query.push_bind(value);
        },
        DataType::Float64 => {
            let value = array
                .as_any()
                .downcast_ref::<Float64Array>()
                .expect("Arrow data type and array implementation agree")
                .value(row);
            query.push_bind(value);
        },
        DataType::Decimal32(_, _) | DataType::Decimal64(_, _) | DataType::Decimal128(_, _) => {
            query
                .push("CAST(")
                .push_bind(crate::metadata_writer::inlined_text_value(array, row)?)
                .push(" AS ")
                .push(sql_type)
                .push(')');
        },
        DataType::Timestamp(arrow::datatypes::TimeUnit::Nanosecond, _) => {
            let value = array
                .as_any()
                .downcast_ref::<TimestampNanosecondArray>()
                .expect("Arrow data type and array implementation agree")
                .value(row);
            query.push_bind(value);
        },
        DataType::Utf8 => {
            query.push_bind(
                array
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .expect("Arrow data type and array implementation agree")
                    .value(row)
                    .to_owned(),
            );
        },
        DataType::LargeUtf8 => {
            query.push_bind(
                array
                    .as_any()
                    .downcast_ref::<LargeStringArray>()
                    .expect("Arrow data type and array implementation agree")
                    .value(row)
                    .to_owned(),
            );
        },
        DataType::Binary => {
            query.push_bind(
                array
                    .as_any()
                    .downcast_ref::<BinaryArray>()
                    .expect("Arrow data type and array implementation agree")
                    .value(row)
                    .to_vec(),
            );
        },
        DataType::LargeBinary => {
            query.push_bind(
                array
                    .as_any()
                    .downcast_ref::<LargeBinaryArray>()
                    .expect("Arrow data type and array implementation agree")
                    .value(row)
                    .to_vec(),
            );
        },
        DataType::BinaryView => {
            query.push_bind(
                array
                    .as_any()
                    .downcast_ref::<BinaryViewArray>()
                    .expect("Arrow data type and array implementation agree")
                    .value(row)
                    .to_vec(),
            );
        },
        DataType::FixedSizeBinary(size) if *size != 16 => {
            query.push_bind(
                array
                    .as_any()
                    .downcast_ref::<FixedSizeBinaryArray>()
                    .expect("Arrow data type and array implementation agree")
                    .value(row)
                    .to_vec(),
            );
        },
        _ => {
            query.push_bind(crate::metadata_writer::inlined_text_value(array, row)?);
        },
    }
    Ok(())
}

/// The DuckLake v1.0 catalog tables in MySQL dialect, one `CREATE TABLE` per
/// entry. sqlx runs each `query()` as a single prepared statement on MySQL
/// (unlike the SQLite driver's multi-statement exec), so — like the Postgres
/// writer — the DDL must be split rather than sent as one `;`-joined script.
///
/// Columns and their order match the SQLite writer's `SQL_CREATE_SCHEMA` (and so
/// upstream DuckLake) for catalog compatibility; only the SQL types are mapped
/// to MySQL. Auto-increment PKs back the ids read via `last_insert_id()`
/// (`schema_id`/`table_id`). `data_file_id`/`delete_file_id` keep their
/// auto-increment declarations for pre-existing-catalog compatibility, but every
/// insert passes an explicit id from the `next_file_id`/`next_delete_file_id`
/// counters so the update/delete/compaction paths and appends share one id
/// space. `snapshot_id` is a plain PK assigned from the `next_snapshot_id`
/// counter, and `ducklake_column` is a bare table (no PK) so a versioned column
/// can hold multiple rows sharing a `column_id`.
const SQL_CREATE_TABLES: &[&str] = &[
    r#"CREATE TABLE IF NOT EXISTS ducklake_metadata (
        `key` VARCHAR(1024) NOT NULL,
        `value` TEXT NOT NULL,
        scope VARCHAR(1024),
        scope_id BIGINT
    ) ENGINE = InnoDB"#,
    r#"CREATE TABLE IF NOT EXISTS ducklake_snapshot (
        snapshot_id BIGINT NOT NULL PRIMARY KEY,
        snapshot_time DATETIME(6) DEFAULT CURRENT_TIMESTAMP(6),
        schema_version BIGINT NOT NULL DEFAULT 0
    ) ENGINE = InnoDB"#,
    r#"CREATE TABLE IF NOT EXISTS ducklake_snapshot_changes (
        snapshot_id BIGINT NOT NULL PRIMARY KEY,
        changes_made TEXT,
        author TEXT,
        commit_message TEXT,
        commit_extra_info TEXT
    ) ENGINE = InnoDB"#,
    r#"CREATE TABLE IF NOT EXISTS ducklake_schema_versions (
        begin_snapshot BIGINT NOT NULL,
        schema_version BIGINT NOT NULL,
        table_id BIGINT NOT NULL,
        UNIQUE (table_id, begin_snapshot)
    ) ENGINE = InnoDB"#,
    r#"CREATE TABLE IF NOT EXISTS ducklake_schema (
        schema_id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
        schema_name VARCHAR(1024) NOT NULL,
        path TEXT NOT NULL,
        path_is_relative TINYINT(1) NOT NULL DEFAULT 1,
        begin_snapshot BIGINT NOT NULL,
        end_snapshot BIGINT
    ) ENGINE = InnoDB"#,
    r#"CREATE TABLE IF NOT EXISTS ducklake_table (
        table_id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
        schema_id BIGINT NOT NULL,
        table_name VARCHAR(1024) NOT NULL,
        path TEXT NOT NULL,
        path_is_relative TINYINT(1) NOT NULL DEFAULT 1,
        begin_snapshot BIGINT NOT NULL,
        end_snapshot BIGINT
    ) ENGINE = InnoDB"#,
    r#"CREATE TABLE IF NOT EXISTS ducklake_view (
        view_id BIGINT,
        view_uuid VARCHAR(36),
        begin_snapshot BIGINT,
        end_snapshot BIGINT,
        schema_id BIGINT,
        view_name VARCHAR(1024),
        dialect VARCHAR(1024),
        `sql` TEXT,
        column_aliases TEXT
    ) ENGINE = InnoDB"#,
    // Bare table (no PRIMARY KEY), mirroring upstream `ducklake_column`: a column
    // is versioned by `[begin_snapshot, end_snapshot)`, so type promotion can
    // preserve its field id while retiring the prior version.
    r#"CREATE TABLE IF NOT EXISTS ducklake_column (
        column_id BIGINT,
        begin_snapshot BIGINT,
        end_snapshot BIGINT,
        table_id BIGINT,
        column_order BIGINT,
        column_name VARCHAR(1024),
        column_type VARCHAR(1024),
        initial_default TEXT,
        default_value TEXT,
        nulls_allowed TINYINT(1),
        parent_column BIGINT,
        default_value_type VARCHAR(1024),
        default_value_dialect VARCHAR(1024)
    ) ENGINE = InnoDB"#,
    r#"CREATE TABLE IF NOT EXISTS ducklake_data_file (
        data_file_id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
        table_id BIGINT NOT NULL,
        path TEXT NOT NULL,
        path_is_relative TINYINT(1) NOT NULL DEFAULT 1,
        file_size_bytes BIGINT NOT NULL,
        footer_size BIGINT,
        encryption_key VARCHAR(1024),
        record_count BIGINT,
        row_id_start BIGINT,
        mapping_id BIGINT,
        begin_snapshot BIGINT NOT NULL,
        end_snapshot BIGINT,
        partial_max BIGINT,
        partition_id BIGINT
    ) ENGINE = InnoDB"#,
    // Per-table row-lineage + running totals. `next_row_id` allocates rowids
    // monotonically over the table's lifetime; `record_count`/`file_size_bytes`
    // mirror the currently-visible totals for DuckDB's `ducklake_table_info`.
    r#"CREATE TABLE IF NOT EXISTS ducklake_table_stats (
        table_id BIGINT NOT NULL PRIMARY KEY,
        record_count BIGINT NOT NULL DEFAULT 0,
        next_row_id BIGINT NOT NULL DEFAULT 0,
        file_size_bytes BIGINT NOT NULL DEFAULT 0
    ) ENGINE = InnoDB"#,
    r#"CREATE TABLE IF NOT EXISTS ducklake_inlined_data_tables (
        table_id BIGINT NOT NULL,
        table_name VARCHAR(1024) NOT NULL,
        schema_version BIGINT NOT NULL
    ) ENGINE = InnoDB"#,
    // Per-file, per-column zone maps (DuckLake spec) — powers file pruning.
    // min/max use TEXT (bounds can be up to the encoder's length cap). Column
    // set mirrors the official extension and the other backends.
    r#"CREATE TABLE IF NOT EXISTS ducklake_file_column_stats (
        data_file_id BIGINT NOT NULL,
        table_id BIGINT NOT NULL,
        column_id BIGINT NOT NULL,
        column_size_bytes BIGINT,
        value_count BIGINT,
        null_count BIGINT,
        min_value TEXT,
        max_value TEXT,
        contains_nan BOOLEAN,
        extra_stats TEXT
    ) ENGINE = InnoDB"#,
    // Table-wide per-column roll-up (DuckLake spec) — feeds the optimizer.
    r#"CREATE TABLE IF NOT EXISTS ducklake_table_column_stats (
        table_id BIGINT NOT NULL,
        column_id BIGINT NOT NULL,
        contains_null BOOLEAN,
        contains_nan BOOLEAN,
        min_value TEXT,
        max_value TEXT,
        extra_stats TEXT
    ) ENGINE = InnoDB"#,
    // Delete files registered by the UPDATE/DELETE paths; ids come from the
    // next_delete_file_id counter (the auto-increment is never consulted).
    r#"CREATE TABLE IF NOT EXISTS ducklake_delete_file (
        delete_file_id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
        data_file_id BIGINT NOT NULL,
        table_id BIGINT NOT NULL,
        path TEXT NOT NULL,
        path_is_relative TINYINT(1) NOT NULL DEFAULT 1,
        file_size_bytes BIGINT NOT NULL,
        footer_size BIGINT,
        encryption_key VARCHAR(1024),
        delete_count BIGINT,
        begin_snapshot BIGINT NOT NULL,
        end_snapshot BIGINT,
        partial_max BIGINT
    ) ENGINE = InnoDB"#,
    r#"CREATE TABLE IF NOT EXISTS ducklake_files_scheduled_for_deletion (
        data_file_id BIGINT NOT NULL,
        path TEXT NOT NULL,
        path_is_relative TINYINT(1) NOT NULL DEFAULT 1,
        schedule_start DATETIME(6) DEFAULT CURRENT_TIMESTAMP(6)
    ) ENGINE = InnoDB"#,
    // Partition spec generations (DuckLake spec); end_snapshot NULL == active.
    r#"CREATE TABLE IF NOT EXISTS ducklake_partition_info (
        partition_id BIGINT NOT NULL,
        table_id BIGINT NOT NULL,
        begin_snapshot BIGINT NOT NULL,
        end_snapshot BIGINT
    ) ENGINE = InnoDB"#,
    // Partition-key columns for a spec (DuckLake spec), ordered by partition_key_index.
    r#"CREATE TABLE IF NOT EXISTS ducklake_partition_column (
        partition_id BIGINT NOT NULL,
        table_id BIGINT NOT NULL,
        partition_key_index BIGINT NOT NULL,
        column_id BIGINT NOT NULL,
        transform VARCHAR(1024) NOT NULL
    ) ENGINE = InnoDB"#,
    // Per-file partition values (DuckLake spec): the value every row in the file
    // shares for a partition key, DuckDB-canonical VARCHAR (NULL is legal).
    r#"CREATE TABLE IF NOT EXISTS ducklake_file_partition_value (
        data_file_id BIGINT NOT NULL,
        table_id BIGINT NOT NULL,
        partition_key_index BIGINT NOT NULL,
        partition_value TEXT
    ) ENGINE = InnoDB"#,
    // Sort spec generations (DuckLake spec); end_snapshot NULL == active. sort_id
    // is allocated from the next_sort_id counter (like partition_id).
    r#"CREATE TABLE IF NOT EXISTS ducklake_sort_info (
        sort_id BIGINT NOT NULL,
        table_id BIGINT NOT NULL,
        begin_snapshot BIGINT NOT NULL,
        end_snapshot BIGINT
    ) ENGINE = InnoDB"#,
    // Sort-key expressions for a spec (DuckLake spec), ordered by sort_key_index.
    // expression is a sort expression in `dialect` (this crate produces bare column
    // names under `duckdb`); sort_direction ASC/DESC; null_order NULLS_FIRST/LAST.
    r#"CREATE TABLE IF NOT EXISTS ducklake_sort_expression (
        sort_id BIGINT NOT NULL,
        table_id BIGINT NOT NULL,
        sort_key_index BIGINT NOT NULL,
        expression VARCHAR(1024) NOT NULL,
        dialect VARCHAR(256) NOT NULL,
        sort_direction VARCHAR(16) NOT NULL,
        null_order VARCHAR(16) NOT NULL
    ) ENGINE = InnoDB"#,
];

/// MySQL-based metadata writer for DuckLake catalogs.
#[derive(Debug, Clone)]
pub struct MySqlMetadataWriter {
    pool: MySqlPool,
}

impl MySqlMetadataWriter {
    /// Open a writer against an existing MySQL DuckLake catalog. Does not create
    /// the catalog tables — call [`Self::initialize_schema`] (or use
    /// [`Self::new_with_init`]) for a fresh database.
    pub async fn new(connection_string: &str) -> Result<Self> {
        Self::with_max_connections(connection_string, DEFAULT_MAX_CONNECTIONS).await
    }

    /// Open a writer with a bounded connection pool.
    pub async fn with_max_connections(
        connection_string: &str,
        max_connections: u32,
    ) -> Result<Self> {
        let pool = MySqlPoolOptions::new()
            .max_connections(max_connections)
            .connect(connection_string)
            .await?;
        Ok(Self {
            pool,
        })
    }

    /// Open a writer and create/upgrade the DuckLake catalog tables.
    pub async fn new_with_init(connection_string: &str) -> Result<Self> {
        let writer = Self::new(connection_string).await?;
        writer.initialize_schema()?;
        Ok(writer)
    }

    /// Expire snapshots and remove catalog rows no longer reachable from a
    /// surviving snapshot. Physical files are queued for later cleanup.
    pub fn expire_snapshots(&self, criteria: ExpireCriteria) -> Result<Vec<ExpiredSnapshot>> {
        block_on(async {
            let mut tx = self.pool.begin().await?;
            let most_recent: Option<i64> =
                sqlx::query_scalar("SELECT MAX(snapshot_id) FROM ducklake_snapshot")
                    .fetch_one(&mut *tx)
                    .await?;
            let Some(most_recent) = most_recent else {
                tx.commit().await?;
                return Ok(Vec::new());
            };
            let rows = match criteria {
                ExpireCriteria::Versions(versions) => {
                    let versions = versions
                        .into_iter()
                        .filter(|snapshot| *snapshot != 0 && *snapshot != most_recent)
                        .collect::<Vec<_>>();
                    if versions.is_empty() {
                        tx.commit().await?;
                        return Ok(Vec::new());
                    }
                    sqlx::query(AssertSqlSafe(format!(
                        "SELECT snapshot_id, CAST(snapshot_time AS CHAR)
                         FROM ducklake_snapshot WHERE snapshot_id IN ({}) ORDER BY snapshot_id",
                        id_list(&versions)
                    )))
                    .fetch_all(&mut *tx)
                    .await?
                },
                ExpireCriteria::OlderThan(cutoff) => {
                    sqlx::query(
                        "SELECT snapshot_id, CAST(snapshot_time AS CHAR)
                         FROM ducklake_snapshot
                         WHERE snapshot_id != 0 AND snapshot_id != ? AND snapshot_time < ? ORDER BY snapshot_id",
                    )
                    .bind(most_recent)
                    .bind(format_sql_timestamp(&cutoff))
                    .fetch_all(&mut *tx)
                    .await?
                },
            };
            let candidates = rows
                .into_iter()
                .map(|row| {
                    Ok(ExpiredSnapshot {
                        snapshot_id: row.try_get(0)?,
                        snapshot_time: row.try_get(1)?,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            if candidates.is_empty() {
                tx.commit().await?;
                return Ok(Vec::new());
            }
            let expire_ids = candidates
                .iter()
                .map(|snapshot| snapshot.snapshot_id)
                .collect::<Vec<_>>();
            sqlx::query(AssertSqlSafe(format!(
                "DELETE FROM ducklake_snapshot WHERE snapshot_id IN ({})",
                id_list(&expire_ids)
            )))
            .execute(&mut *tx)
            .await?;
            let dead_tables = sqlx::query(
                "SELECT t.table_id FROM ducklake_table t
                 WHERE t.end_snapshot IS NOT NULL AND NOT EXISTS (
                     SELECT 1 FROM ducklake_snapshot
                     WHERE snapshot_id >= t.begin_snapshot AND snapshot_id < t.end_snapshot)
                 AND NOT EXISTS (
                     SELECT 1 FROM ducklake_table t2
                     WHERE t2.table_id = t.table_id
                       AND (t2.end_snapshot IS NULL OR EXISTS (
                           SELECT 1 FROM ducklake_snapshot
                           WHERE snapshot_id >= t2.begin_snapshot
                             AND snapshot_id < t2.end_snapshot)))",
            )
            .fetch_all(&mut *tx)
            .await?
            .into_iter()
            .map(|row| row.try_get::<i64, _>(0))
            .collect::<std::result::Result<Vec<_>, _>>()?;
            let dead_table_filter = if dead_tables.is_empty() {
                "false".to_string()
            } else {
                format!("df.table_id IN ({})", id_list(&dead_tables))
            };
            let dead_data_files = sqlx::query(AssertSqlSafe(format!(
                "SELECT df.data_file_id FROM ducklake_data_file df
                 WHERE ({dead_table_filter}) OR (df.end_snapshot IS NOT NULL AND NOT EXISTS (
                     SELECT 1 FROM ducklake_snapshot
                     WHERE snapshot_id >= df.begin_snapshot AND snapshot_id < df.end_snapshot))"
            )))
            .fetch_all(&mut *tx)
            .await?
            .into_iter()
            .map(|row| row.try_get::<i64, _>(0))
            .collect::<std::result::Result<Vec<_>, _>>()?;
            if !dead_data_files.is_empty() {
                let ids = id_list(&dead_data_files);
                sqlx::query(AssertSqlSafe(format!(
                    "INSERT INTO ducklake_files_scheduled_for_deletion
                         (data_file_id, path, path_is_relative)
                     SELECT df.data_file_id, {RESOLVED_PATH}, {REL_FLAG}
                     FROM ducklake_data_file df
                     JOIN ducklake_table t ON t.table_id = df.table_id
                     JOIN ducklake_schema s ON s.schema_id = t.schema_id
                     WHERE df.data_file_id IN ({ids})"
                )))
                .execute(&mut *tx)
                .await?;
                sqlx::query(AssertSqlSafe(format!(
                    "DELETE FROM ducklake_data_file WHERE data_file_id IN ({ids})"
                )))
                .execute(&mut *tx)
                .await?;
            }
            let dead_data_filter = if dead_data_files.is_empty() {
                "false".to_string()
            } else {
                format!("df.data_file_id IN ({})", id_list(&dead_data_files))
            };
            let dead_delete_table_filter = if dead_tables.is_empty() {
                "false".to_string()
            } else {
                format!("df.table_id IN ({})", id_list(&dead_tables))
            };
            let dead_delete_files = sqlx::query(AssertSqlSafe(format!(
                "SELECT df.delete_file_id FROM ducklake_delete_file df
                 WHERE ({dead_data_filter}) OR ({dead_delete_table_filter})
                    OR (df.end_snapshot IS NOT NULL AND NOT EXISTS (
                        SELECT 1 FROM ducklake_snapshot
                        WHERE snapshot_id >= df.begin_snapshot AND snapshot_id < df.end_snapshot))"
            )))
            .fetch_all(&mut *tx)
            .await?
            .into_iter()
            .map(|row| row.try_get::<i64, _>(0))
            .collect::<std::result::Result<Vec<_>, _>>()?;
            if !dead_delete_files.is_empty() {
                let ids = id_list(&dead_delete_files);
                sqlx::query(AssertSqlSafe(format!(
                    "INSERT INTO ducklake_files_scheduled_for_deletion
                         (data_file_id, path, path_is_relative)
                     SELECT df.delete_file_id, {RESOLVED_PATH}, {REL_FLAG}
                     FROM ducklake_delete_file df
                     JOIN ducklake_table t ON t.table_id = df.table_id
                     JOIN ducklake_schema s ON s.schema_id = t.schema_id
                     WHERE df.delete_file_id IN ({ids})"
                )))
                .execute(&mut *tx)
                .await?;
                sqlx::query(AssertSqlSafe(format!(
                    "DELETE FROM ducklake_delete_file WHERE delete_file_id IN ({ids})"
                )))
                .execute(&mut *tx)
                .await?;
            }
            if !dead_tables.is_empty() {
                let ids = id_list(&dead_tables);
                for table in [
                    "ducklake_table",
                    "ducklake_table_stats",
                    "ducklake_column",
                    "ducklake_schema_versions",
                ] {
                    sqlx::query(AssertSqlSafe(format!(
                        "DELETE FROM {table} WHERE table_id IN ({ids})"
                    )))
                    .execute(&mut *tx)
                    .await?;
                }
            }
            sqlx::query(
                "DELETE FROM ducklake_schema
                 WHERE end_snapshot IS NOT NULL AND NOT EXISTS (
                     SELECT 1 FROM ducklake_snapshot
                     WHERE snapshot_id >= ducklake_schema.begin_snapshot
                       AND snapshot_id < ducklake_schema.end_snapshot)",
            )
            .execute(&mut *tx)
            .await?;
            tx.commit().await?;
            Ok(candidates)
        })
    }
}

/// Atomically reserve `n` consecutive ids from a monotonic counter stored in
/// `ducklake_metadata` (seeded by `initialize_schema`), returning the LAST id of
/// the block — the reserved ids are `last - n + 1 ..= last`.
///
/// MySQL has no `UPDATE … RETURNING`, so this bumps the counter then reads it
/// back within the same transaction. The `UPDATE` takes an exclusive InnoDB row
/// lock held until commit, so a concurrent `reserve_ids` on the same `key`
/// blocks here rather than handing out an overlapping id — the same
/// serialization SQLite gets from its single-writer lock. Used for `column_id`
/// and `snapshot_id`.
async fn reserve_ids(
    tx: &mut sqlx::Transaction<'_, sqlx::MySql>,
    key: &str,
    n: i64,
) -> Result<i64> {
    sqlx::query(
        "UPDATE ducklake_metadata
         SET `value` = CAST(CAST(`value` AS SIGNED) + ? AS CHAR)
         WHERE `key` = ? AND scope IS NULL",
    )
    .bind(n)
    .bind(key)
    .execute(&mut **tx)
    .await?;
    let last: i64 = sqlx::query(
        "SELECT CAST(`value` AS SIGNED) FROM ducklake_metadata WHERE `key` = ? AND scope IS NULL",
    )
    .bind(key)
    .fetch_one(&mut **tx)
    .await?
    .try_get(0)?;
    Ok(last)
}

async fn reserve_file_ids(tx: &mut sqlx::Transaction<'_, sqlx::MySql>, n: i64) -> Result<Vec<i64>> {
    let last = reserve_ids(tx, "next_file_id", n).await?;
    Ok(((last - n + 1)..=last).collect())
}

async fn reserve_delete_file_ids(
    tx: &mut sqlx::Transaction<'_, sqlx::MySql>,
    n: i64,
) -> Result<Vec<i64>> {
    let last = reserve_ids(tx, "next_delete_file_id", n).await?;
    Ok(((last - n + 1)..=last).collect())
}

/// Seed a monotonic id counter row if it does not already exist, starting from
/// the current MAX of its backing column so a pre-existing catalog keeps
/// allocating without reuse; an existing counter is raised to that MAX if it
/// fell behind (the file-id counters of a catalog written before the writer
/// unified on explicit ids sit below the auto-increment-assigned MAX). Done as
/// check-then-insert (two statements) rather than `INSERT … SELECT … WHERE NOT
/// EXISTS`, because that self-referential `INSERT … SELECT` against
/// `ducklake_metadata` is rejected by MySQL (1093).
async fn seed_counter(pool: &MySqlPool, key: &str, max_sql: &'static str) -> Result<()> {
    let exists: i64 =
        sqlx::query("SELECT COUNT(*) FROM ducklake_metadata WHERE `key` = ? AND scope IS NULL")
            .bind(key)
            .fetch_one(pool)
            .await?
            .try_get(0)?;
    let start: i64 = sqlx::query(max_sql).fetch_one(pool).await?.try_get(0)?;
    if exists == 0 {
        sqlx::query("INSERT INTO ducklake_metadata (`key`, `value`, scope) VALUES (?, ?, NULL)")
            .bind(key)
            .bind(start.to_string())
            .execute(pool)
            .await?;
    } else {
        sqlx::query(
            "UPDATE ducklake_metadata
             SET `value` = CAST(GREATEST(CAST(`value` AS SIGNED), ?) AS CHAR)
             WHERE `key` = ? AND scope IS NULL",
        )
        .bind(start)
        .bind(key)
        .execute(pool)
        .await?;
    }
    Ok(())
}

/// Optimistic-concurrency check for a `Replace` commit (mirrors the SQLite /
/// Postgres writers). Run before retiring the prior generation: if any data file
/// of the table has `begin_snapshot` or `end_snapshot` newer than
/// `base_snapshot` (the head observed when this write began), another writer
/// published a newer generation in the meantime, so this `Replace` aborts with
/// [`crate::DuckLakeError::Conflict`] rather than clobbering it. (`Append` does
/// not call this: concurrent appends commute.)
async fn detect_replace_conflict(
    tx: &mut sqlx::Transaction<'_, sqlx::MySql>,
    table_id: i64,
    base_snapshot: i64,
) -> Result<()> {
    let conflict: Option<i64> = sqlx::query(
        "SELECT 1 FROM ducklake_data_file
         WHERE table_id = ? AND (begin_snapshot > ? OR end_snapshot > ?)
         LIMIT 1",
    )
    .bind(table_id)
    .bind(base_snapshot)
    .bind(base_snapshot)
    .fetch_optional(&mut **tx)
    .await?
    .map(|row| row.try_get(0))
    .transpose()?;
    if conflict.is_some() {
        return Err(crate::DuckLakeError::Conflict(format!(
            "Replace on table {table_id} conflicts with a concurrent write committed since \
             snapshot {base_snapshot}; aborting (retry the write against the new generation)"
        )));
    }
    let inlined_tables =
        sqlx::query("SELECT table_name FROM ducklake_inlined_data_tables WHERE table_id = ?")
            .bind(table_id)
            .fetch_all(&mut **tx)
            .await?;
    for row in inlined_tables {
        let table_name: String = row.try_get(0)?;
        let sql = format!(
            "SELECT 1 FROM {} WHERE begin_snapshot > ? OR end_snapshot > ? LIMIT 1",
            quote_ident(&table_name)
        );
        if sqlx::query(AssertSqlSafe(sql))
            .bind(base_snapshot)
            .bind(base_snapshot)
            .fetch_optional(&mut **tx)
            .await?
            .is_some()
        {
            return Err(crate::DuckLakeError::Conflict(format!(
                "Replace on table {table_id} conflicts with inlined data committed since \
                 snapshot {base_snapshot}; aborting"
            )));
        }
    }
    Ok(())
}

async fn detect_new_inlined_deletes(
    tx: &mut sqlx::Transaction<'_, sqlx::MySql>,
    table_id: i64,
    base_snapshot: i64,
    data_file_ids: &[i64],
) -> Result<()> {
    if data_file_ids.is_empty() {
        return Ok(());
    }

    let table = crate::metadata_provider::inlined_delete_table_name(table_id)?;
    let exists: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM information_schema.tables
         WHERE table_schema = DATABASE() AND table_name = ?",
    )
    .bind(&table)
    .fetch_one(&mut **tx)
    .await?;
    if exists == 0 {
        return Ok(());
    }

    let conflict: i64 = sqlx::query_scalar(AssertSqlSafe(format!(
        "SELECT COUNT(*) FROM {} newer
         WHERE newer.file_id IN ({})
           AND (newer.begin_snapshot IS NULL OR (
             newer.begin_snapshot > ?
             AND (newer.row_id IS NULL OR NOT EXISTS (
               SELECT 1 FROM {} prior
               WHERE prior.file_id = newer.file_id
                 AND prior.row_id = newer.row_id
                 AND prior.begin_snapshot <= ?
             ))
           ))",
        quote_ident(&table),
        id_list(data_file_ids),
        quote_ident(&table),
    )))
    .bind(base_snapshot)
    .bind(base_snapshot)
    .fetch_one(&mut **tx)
    .await?;
    if conflict > 0 {
        return Err(crate::DuckLakeError::Conflict(format!(
            "table {table_id} gained an inlined delete on a source file since snapshot \
             {base_snapshot}; re-open the catalog and re-plan"
        )));
    }
    Ok(())
}

/// Retire the prior generation's still-visible data files at `snapshot_id` and
/// zero the visible stat totals. The `begin_snapshot < snapshot_id` guard spares
/// files registered for *this* snapshot, so a multi-file write does not retire
/// its own siblings. `next_row_id` is left untouched (rowids stay monotonic).
async fn retire_prior_generation(
    tx: &mut sqlx::Transaction<'_, sqlx::MySql>,
    table_id: i64,
    snapshot_id: i64,
) -> Result<()> {
    sqlx::query(
        "UPDATE ducklake_data_file SET end_snapshot = ?
         WHERE table_id = ? AND end_snapshot IS NULL AND begin_snapshot < ?",
    )
    .bind(snapshot_id)
    .bind(table_id)
    .bind(snapshot_id)
    .execute(&mut **tx)
    .await?;

    let inlined_tables =
        sqlx::query("SELECT table_name FROM ducklake_inlined_data_tables WHERE table_id = ?")
            .bind(table_id)
            .fetch_all(&mut **tx)
            .await?;
    for row in inlined_tables {
        let table_name: String = row.try_get(0)?;
        let sql = format!(
            "UPDATE {} SET end_snapshot = ? \
             WHERE end_snapshot IS NULL AND begin_snapshot < ?",
            quote_ident(&table_name)
        );
        sqlx::query(AssertSqlSafe(sql))
            .bind(snapshot_id)
            .bind(snapshot_id)
            .execute(&mut **tx)
            .await?;
    }

    sqlx::query(
        "UPDATE ducklake_table_stats SET record_count = 0, file_size_bytes = 0 WHERE table_id = ?",
    )
    .bind(table_id)
    .execute(&mut **tx)
    .await?;
    Ok(())
}

/// Insert the next `ducklake_snapshot` row, carrying `schema_version` forward
/// (the pure-data-write default), and return `(snapshot_id, schema_version)`.
///
/// `snapshot_id` is allocated from the `next_snapshot_id` counter. [`reserve_ids`]
/// takes an exclusive InnoDB lock on that counter row held until this transaction
/// commits, so this is the "write-lock-first" serialization point of a commit —
/// every commit transaction contends on the single counter row, so per-catalog
/// id order equals commit order and the scalar `> base_snapshot` conflict test is
/// exact. The counter is used (rather than `SELECT MAX(snapshot_id)+1`) because
/// MySQL rejects `INSERT … SELECT` from the table being inserted into (error
/// 1093) and has no `RETURNING`; a counter both serializes writers and hands the
/// id back directly. A DDL commit follows this with [`bump_schema_version`].
async fn insert_snapshot(tx: &mut sqlx::Transaction<'_, sqlx::MySql>) -> Result<(i64, i64)> {
    let snapshot_id = reserve_ids(tx, "next_snapshot_id", 1).await?;
    // Carry the current per-catalog schema_version forward; a DDL commit corrects
    // this to a bump via `bump_schema_version` below. Read before the INSERT so
    // the MAX is over the pre-existing rows only (matches the SQLite writer).
    let schema_version: i64 =
        sqlx::query("SELECT COALESCE(MAX(schema_version), 0) FROM ducklake_snapshot")
            .fetch_one(&mut **tx)
            .await?
            .try_get(0)?;
    sqlx::query(
        "INSERT INTO ducklake_snapshot (snapshot_id, snapshot_time, schema_version)
         VALUES (?, NOW(6), ?)",
    )
    .bind(snapshot_id)
    .bind(schema_version)
    .execute(&mut **tx)
    .await?;
    sqlx::query(
        "INSERT INTO ducklake_snapshot_changes (snapshot_id, changes_made)
         VALUES (?, NULL)",
    )
    .bind(snapshot_id)
    .execute(&mut **tx)
    .await?;
    Ok((snapshot_id, schema_version))
}

async fn record_snapshot_changes(
    tx: &mut sqlx::Transaction<'_, sqlx::MySql>,
    snapshot_id: i64,
    changes_made: &str,
    commit_metadata: &SnapshotCommitMetadata,
) -> Result<()> {
    let changes_made = (!changes_made.is_empty()).then_some(changes_made);
    sqlx::query(
        "UPDATE ducklake_snapshot_changes
         SET changes_made = CASE
                 WHEN changes_made IS NULL THEN ?
                 WHEN ? IS NULL THEN changes_made
                 ELSE CONCAT(changes_made, ',', ?)
             END,
             author = ?,
             commit_message = ?,
             commit_extra_info = ?
         WHERE snapshot_id = ?",
    )
    .bind(changes_made)
    .bind(changes_made)
    .bind(changes_made)
    .bind(commit_metadata.author())
    .bind(commit_metadata.message())
    .bind(commit_metadata.extra_info())
    .bind(snapshot_id)
    .execute(&mut **tx)
    .await?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn record_table_write_changes(
    tx: &mut sqlx::Transaction<'_, sqlx::MySql>,
    snapshot_id: i64,
    table_id: i64,
    schema_name: &str,
    table_name: &str,
    mode: WriteMode,
    has_deletes: bool,
    commit_metadata: &SnapshotCommitMetadata,
) -> Result<()> {
    let row = sqlx::query(
        "SELECT s.begin_snapshot AS schema_begin_snapshot,
                t.begin_snapshot AS table_begin_snapshot
         FROM ducklake_table t
         JOIN ducklake_schema s ON s.schema_id = t.schema_id
         WHERE t.table_id = ?",
    )
    .bind(table_id)
    .fetch_one(&mut **tx)
    .await?;
    let schema_begin_snapshot: i64 = row.try_get("schema_begin_snapshot")?;
    let table_begin_snapshot: i64 = row.try_get("table_begin_snapshot")?;
    let altered: bool = sqlx::query_scalar(
        "SELECT EXISTS(
            SELECT 1 FROM ducklake_schema_versions
            WHERE table_id = ? AND begin_snapshot = ?
         )",
    )
    .bind(table_id)
    .bind(snapshot_id)
    .fetch_one(&mut **tx)
    .await?;
    let mut replaced_existing_data: bool = sqlx::query_scalar(
        "SELECT EXISTS(
            SELECT 1 FROM ducklake_data_file
            WHERE table_id = ? AND end_snapshot = ?
         )",
    )
    .bind(table_id)
    .bind(snapshot_id)
    .fetch_one(&mut **tx)
    .await?;
    if !replaced_existing_data {
        // A Replace over inline-only prior data ends inline rows, not files.
        let inlined_tables: Vec<String> = sqlx::query_scalar(
            "SELECT table_name FROM ducklake_inlined_data_tables WHERE table_id = ?",
        )
        .bind(table_id)
        .fetch_all(&mut **tx)
        .await?;
        for inlined_table in inlined_tables {
            let sql = format!(
                "SELECT EXISTS(SELECT 1 FROM {} WHERE end_snapshot = ?)",
                quote_ident(&inlined_table)
            );
            if sqlx::query_scalar(AssertSqlSafe(sql))
                .bind(snapshot_id)
                .fetch_one(&mut **tx)
                .await?
            {
                replaced_existing_data = true;
                break;
            }
        }
    }

    let mut changes = Vec::new();
    if schema_begin_snapshot == snapshot_id {
        changes.push(format!(
            "created_schema:{}",
            quote_snapshot_name(schema_name)
        ));
    }
    if table_begin_snapshot == snapshot_id {
        changes.push(format!(
            "created_table:{}",
            quote_snapshot_table(schema_name, table_name)
        ));
    } else if altered {
        changes.push(format!("altered_table:{table_id}"));
    }
    changes.push(table_write_changes(
        table_id,
        mode,
        has_deletes,
        replaced_existing_data,
    ));
    record_snapshot_changes(tx, snapshot_id, &changes.join(","), commit_metadata).await
}

/// Bump the per-catalog monotonic `schema_version` on a DDL snapshot to
/// `prev_max + 1` (max over the OTHER snapshots, so re-running is stable) and
/// return the new value. Mirrors upstream `if (SchemaChangesMade()) schema_version++`.
async fn bump_schema_version(
    tx: &mut sqlx::Transaction<'_, sqlx::MySql>,
    snapshot_id: i64,
) -> Result<i64> {
    let prev_max: i64 = sqlx::query(
        "SELECT COALESCE(MAX(schema_version), 0) FROM ducklake_snapshot WHERE snapshot_id <> ?",
    )
    .bind(snapshot_id)
    .fetch_one(&mut **tx)
    .await?
    .try_get(0)?;
    let new_version = prev_max + 1;
    sqlx::query("UPDATE ducklake_snapshot SET schema_version = ? WHERE snapshot_id = ?")
        .bind(new_version)
        .bind(snapshot_id)
        .execute(&mut **tx)
        .await?;
    Ok(new_version)
}

/// Record a `ducklake_schema_versions` ledger row for a DDL that leaves the table
/// live (create, column add/remove/reorder). Not called for a drop.
async fn record_schema_version(
    tx: &mut sqlx::Transaction<'_, sqlx::MySql>,
    snapshot_id: i64,
    schema_version: i64,
    table_id: i64,
) -> Result<()> {
    sqlx::query(
        "INSERT INTO ducklake_schema_versions (begin_snapshot, schema_version, table_id)
         VALUES (?, ?, ?)",
    )
    .bind(snapshot_id)
    .bind(schema_version)
    .bind(table_id)
    .execute(&mut **tx)
    .await?;
    Ok(())
}

/// The atomic commit point for a single-catalog write. Inserts the deferred
/// `ducklake_snapshot` row (its counter id was reserved conceptually at begin but
/// the row is inserted here), finalizes the column generation, and — for
/// `Replace` — retires the prior data generation. All within the caller's
/// transaction, so a reader never sees a half-published head.
///
/// The column generation is deferred to here (rather than written in
/// `begin_write_transaction`) because the read path resolves a table's columns by
/// `end_snapshot IS NULL` only (not snapshot-scoped), so inserting the new
/// generation at begin would leak it to concurrent reads during the upload
/// window. `column_ids` are the ids reserved at begin and already baked into the
/// staged parquet's `field_id` metadata.
/// Persist the harvested per-column stats for a just-registered data file
/// (per-file zone maps). See the SQLite writer's equivalent for the rationale.
async fn insert_file_column_stats(
    tx: &mut sqlx::Transaction<'_, sqlx::MySql>,
    table_id: i64,
    data_file_id: i64,
    column_stats: &[ColumnStat],
) -> Result<()> {
    for stat in column_stats {
        sqlx::query(
            "INSERT INTO ducklake_file_column_stats
                 (data_file_id, table_id, column_id, column_size_bytes,
                  value_count, null_count, min_value, max_value, contains_nan, extra_stats)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)",
        )
        .bind(data_file_id)
        .bind(table_id)
        .bind(stat.column_id)
        .bind(stat.column_size_bytes)
        .bind(stat.value_count)
        .bind(stat.null_count)
        .bind(stat.min_value.as_deref())
        .bind(stat.max_value.as_deref())
        .bind(stat.contains_nan)
        .execute(&mut **tx)
        .await?;
    }
    Ok(())
}

/// Recompute `ducklake_table_column_stats` from the table's live files and
/// replace the stored rows. See the SQLite writer's equivalent for the rationale.
/// Persist a partitioned data file's partition metadata within the commit
/// transaction: set `ducklake_data_file.partition_id` and insert one
/// `ducklake_file_partition_value` row per partition key. A no-op for an
/// unpartitioned file.
async fn insert_partition_metadata(
    tx: &mut sqlx::Transaction<'_, sqlx::MySql>,
    table_id: i64,
    data_file_id: i64,
    file: &DataFileInfo,
) -> Result<()> {
    if let Some(partition_id) = file.partition_id {
        sqlx::query("UPDATE ducklake_data_file SET partition_id = ? WHERE data_file_id = ?")
            .bind(partition_id)
            .bind(data_file_id)
            .execute(&mut **tx)
            .await?;
    }
    for (key_index, value) in &file.partition_values {
        sqlx::query(
            "INSERT INTO ducklake_file_partition_value
                 (data_file_id, table_id, partition_key_index, partition_value)
             VALUES (?, ?, ?, ?)",
        )
        .bind(data_file_id)
        .bind(table_id)
        .bind(i64::from(*key_index))
        .bind(value.clone())
        .execute(&mut **tx)
        .await?;
    }
    Ok(())
}

async fn live_columns_for_stats(
    tx: &mut sqlx::Transaction<'_, sqlx::MySql>,
    table_id: i64,
) -> Result<(Vec<ColumnDef>, Vec<i64>)> {
    let rows = sqlx::query(
        "SELECT column_id, column_name, column_type
         FROM ducklake_column
         WHERE table_id = ? AND end_snapshot IS NULL
         ORDER BY column_order",
    )
    .bind(table_id)
    .fetch_all(&mut **tx)
    .await?;
    let mut columns = Vec::with_capacity(rows.len());
    let mut column_ids = Vec::with_capacity(rows.len());
    for row in rows {
        column_ids.push(row.try_get(0)?);
        columns.push(ColumnDef::new(
            row.try_get::<String, _>(1)?,
            row.try_get::<String, _>(2)?,
            true,
        )?);
    }
    Ok((columns, column_ids))
}

async fn apply_delete_entry(
    tx: &mut sqlx::Transaction<'_, sqlx::MySql>,
    table_id: i64,
    base_snapshot: i64,
    snapshot_id: i64,
    entry: &DeleteFileEntry,
) -> Result<()> {
    let target_live: Option<i64> = sqlx::query_scalar(
        "SELECT 1 FROM ducklake_data_file
         WHERE data_file_id = ? AND end_snapshot IS NULL",
    )
    .bind(entry.data_file_id)
    .fetch_optional(&mut **tx)
    .await?;
    if target_live.is_none() {
        return Err(crate::DuckLakeError::Conflict(format!(
            "UPDATE/DELETE on data file {} could not commit: the file is no longer live as of \
             the catalog's current head (retired since snapshot {base_snapshot}); re-open the \
             catalog at the latest snapshot and retry",
            entry.data_file_id
        )));
    }
    let current_delete: Option<i64> = sqlx::query_scalar(
        "SELECT delete_file_id FROM ducklake_delete_file
         WHERE data_file_id = ? AND end_snapshot IS NULL",
    )
    .bind(entry.data_file_id)
    .fetch_optional(&mut **tx)
    .await?;
    if current_delete != entry.expected_prev_delete_file {
        return Err(crate::DuckLakeError::Conflict(format!(
            "UPDATE/DELETE on data file {} could not commit: its live delete file changed from \
             {:?} to {current_delete:?} since snapshot {base_snapshot}; re-open the catalog at \
             the latest snapshot and retry",
            entry.data_file_id, entry.expected_prev_delete_file
        )));
    }
    if let Some(delete_file_id) = entry.expected_prev_delete_file {
        sqlx::query(
            "UPDATE ducklake_delete_file SET end_snapshot = ?
             WHERE delete_file_id = ? AND end_snapshot IS NULL",
        )
        .bind(snapshot_id)
        .bind(delete_file_id)
        .execute(&mut **tx)
        .await?;
    }
    let delete_file_id = reserve_delete_file_ids(tx, 1).await?[0];
    sqlx::query(
        "INSERT INTO ducklake_delete_file
             (delete_file_id, data_file_id, table_id, path, path_is_relative,
              file_size_bytes, footer_size, delete_count, begin_snapshot)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
    )
    .bind(delete_file_id)
    .bind(entry.data_file_id)
    .bind(table_id)
    .bind(&entry.delete.path)
    .bind(entry.delete.path_is_relative)
    .bind(entry.delete.file_size_bytes)
    .bind(entry.delete.footer_size)
    .bind(entry.delete.delete_count)
    .bind(snapshot_id)
    .execute(&mut **tx)
    .await?;
    Ok(())
}

/// The physical `ducklake_inlined_data_*` tables registered for the table.
/// Mirrors the SQLite writer's `inlined_table_names`.
async fn inlined_table_names(
    tx: &mut sqlx::Transaction<'_, sqlx::MySql>,
    table_id: i64,
) -> Result<Vec<String>> {
    let rows =
        sqlx::query("SELECT table_name FROM ducklake_inlined_data_tables WHERE table_id = ?")
            .bind(table_id)
            .fetch_all(&mut **tx)
            .await?;
    rows.into_iter().map(|row| Ok(row.try_get(0)?)).collect()
}

/// Count the table's still-visible inlined rows across its registered physical
/// tables. Mirrors the SQLite writer's `live_inlined_row_count`.
async fn live_inlined_row_count(
    tx: &mut sqlx::Transaction<'_, sqlx::MySql>,
    table_id: i64,
) -> Result<i64> {
    let mut total = 0;
    for table_name in inlined_table_names(tx, table_id).await? {
        let sql = format!(
            "SELECT COUNT(*) FROM {} WHERE end_snapshot IS NULL",
            quote_ident(&table_name)
        );
        total += sqlx::query_scalar::<_, i64>(AssertSqlSafe(sql))
            .fetch_one(&mut **tx)
            .await?;
    }
    Ok(total)
}

/// End the referenced inlined rows at `snapshot_id`, fencing each against
/// `base_snapshot`. Mirrors the SQLite writer's `apply_inlined_deletes`
/// (used by the combined `commit_deletes` path).
async fn apply_inlined_deletes(
    tx: &mut sqlx::Transaction<'_, sqlx::MySql>,
    table_id: i64,
    snapshot_id: i64,
    base_snapshot: i64,
    rows: &[InlinedRowRef],
) -> Result<()> {
    let registered = inlined_table_names(tx, table_id)
        .await?
        .into_iter()
        .collect::<std::collections::HashSet<_>>();
    for row in rows {
        if !registered.contains(&row.table_name) {
            return Err(crate::DuckLakeError::Conflict(format!(
                "inlined row {} belongs to an unregistered table '{}'",
                row.row_id, row.table_name
            )));
        }
        let sql = format!(
            "UPDATE {} SET end_snapshot = ? \
             WHERE row_id = ? AND begin_snapshot <= ? AND end_snapshot IS NULL",
            quote_ident(&row.table_name)
        );
        let affected = sqlx::query(AssertSqlSafe(sql))
            .bind(snapshot_id)
            .bind(row.row_id)
            .bind(base_snapshot)
            .execute(&mut **tx)
            .await?
            .rows_affected();
        if affected != 1 {
            return Err(crate::DuckLakeError::Conflict(format!(
                "inlined row {} in '{}' is no longer live at snapshot {base_snapshot}",
                row.row_id, row.table_name
            )));
        }
    }
    Ok(())
}

async fn recompute_table_column_stats(
    tx: &mut sqlx::Transaction<'_, sqlx::MySql>,
    table_id: i64,
    columns: &[ColumnDef],
    column_ids: &[i64],
) -> Result<()> {
    use crate::stats_encode::{FileColumnStat, aggregate_global_column_stats};
    let catalog_columns = catalog_column_defs(columns)?;
    let column_ids = top_level_column_ids(&catalog_columns, column_ids)?;

    let live_file_count: i64 = sqlx::query(
        "SELECT COUNT(*) FROM ducklake_data_file WHERE table_id = ? AND end_snapshot IS NULL",
    )
    .bind(table_id)
    .fetch_one(&mut **tx)
    .await?
    .try_get(0)?;

    let mut per_file: Vec<FileColumnStat> = Vec::new();
    for row in sqlx::query(
        "SELECT s.column_id, s.min_value, s.max_value, s.null_count, s.contains_nan
         FROM ducklake_file_column_stats s
         JOIN ducklake_data_file d ON d.data_file_id = s.data_file_id
         WHERE d.table_id = ? AND d.end_snapshot IS NULL",
    )
    .bind(table_id)
    .fetch_all(&mut **tx)
    .await?
    {
        per_file.push(FileColumnStat {
            column_id: row.try_get(0)?,
            min_value: row.try_get(1)?,
            max_value: row.try_get(2)?,
            null_count: row.try_get(3)?,
            contains_nan: row.try_get(4)?,
        });
    }

    let numeric_of = |column_id: i64| -> bool {
        column_ids
            .iter()
            .position(|id| *id == column_id)
            .and_then(|i| columns.get(i))
            .map(|c| crate::stats_encode::is_numeric_ducklake_type(c.ducklake_type()))
            .unwrap_or(false)
    };
    let globals = aggregate_global_column_stats(&per_file, live_file_count, numeric_of);

    sqlx::query("DELETE FROM ducklake_table_column_stats WHERE table_id = ?")
        .bind(table_id)
        .execute(&mut **tx)
        .await?;
    for g in globals {
        sqlx::query(
            "INSERT INTO ducklake_table_column_stats
                 (table_id, column_id, contains_null, contains_nan, min_value, max_value, extra_stats)
             VALUES (?, ?, ?, ?, ?, ?, NULL)",
        )
        .bind(table_id)
        .bind(g.column_id)
        .bind(g.contains_null)
        .bind(g.contains_nan)
        .bind(g.min_value)
        .bind(g.max_value)
        .execute(&mut **tx)
        .await?;
    }
    Ok(())
}

async fn finalize_snapshot(
    tx: &mut sqlx::Transaction<'_, sqlx::MySql>,
    table_id: i64,
    columns: &[ColumnDef],
    column_ids: &[i64],
    mode: WriteMode,
    base_snapshot: i64,
) -> Result<i64> {
    // Allocate the snapshot FIRST (carrying schema_version forward): this takes
    // the counter lock up front, serializing concurrent commits. schema_version is
    // corrected to a DDL bump below once we've classified the commit.
    let (snapshot_id, mut schema_version) = insert_snapshot(tx).await?;

    // Classify this commit as DDL vs pure data write. `current` is the table's
    // live columns ordered by `column_order`; an empty set means a brand-new table
    // (the creating write is DDL). Mirrors upstream `SchemaChangesMade()`.
    use std::collections::{HashMap, HashSet};
    let proposed = catalog_column_defs(columns)?;
    if proposed.len() != column_ids.len() {
        return Err(crate::DuckLakeError::InvalidConfig(format!(
            "column_ids has {} entries for {} catalog column nodes",
            column_ids.len(),
            proposed.len()
        )));
    }
    let current = sqlx::query(
        "SELECT column_id, column_name, column_type, column_order, nulls_allowed, parent_column
         FROM ducklake_column
         WHERE table_id = ? AND end_snapshot IS NULL
         ORDER BY column_order",
    )
    .bind(table_id)
    .fetch_all(&mut **tx)
    .await?;

    let existing_catalog_columns = current
        .iter()
        .map(|row| {
            Ok::<_, sqlx::Error>(ExistingCatalogColumn {
                column_id: row.try_get("column_id")?,
                name: row.try_get("column_name")?,
                ducklake_type: row.try_get("column_type")?,
                parent_column: row.try_get("parent_column")?,
            })
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let existing_nullability = current
        .iter()
        .map(|row| {
            Ok::<_, sqlx::Error>(
                row.try_get::<Option<bool>, _>("nulls_allowed")?
                    .unwrap_or(true),
            )
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let committed_ids = assign_column_ids(&proposed, &existing_catalog_columns, column_ids)?;
    if committed_ids != column_ids {
        return Err(crate::DuckLakeError::Conflict(
            "table columns were created concurrently with different field ids; retry the write"
                .to_string(),
        ));
    }
    let is_ddl = current.is_empty()
        || catalog_columns_differ(
            &existing_catalog_columns,
            &existing_nullability,
            &proposed,
            column_ids,
        );
    if is_ddl {
        // A DDL commit bumps the per-catalog schema_version (the insert above only
        // carried it forward). A pure data write keeps the carried value.
        schema_version = bump_schema_version(tx, snapshot_id).await?;
    }

    // Reconcile the column generation SURGICALLY so each column keeps a stable
    // column_id (== parquet field_id) across writes: end only removed columns,
    // insert only new ones, and leave unchanged columns (and their ids) in place.
    let proposed_ids = column_ids.iter().copied().collect::<HashSet<_>>();
    let mut current_by_id: HashMap<i64, (i64, bool, String)> = HashMap::new();
    for row in &current {
        let column_id: i64 = row.try_get("column_id")?;
        let order: i64 = row.try_get("column_order")?;
        let nullable: bool = row
            .try_get::<Option<bool>, _>("nulls_allowed")?
            .unwrap_or(true);
        let ducklake_type: String = row.try_get("column_type")?;
        if !proposed_ids.contains(&column_id) {
            sqlx::query(
                "UPDATE ducklake_column SET end_snapshot = ?
                 WHERE table_id = ? AND column_id = ? AND end_snapshot IS NULL",
            )
            .bind(snapshot_id)
            .bind(table_id)
            .bind(column_id)
            .execute(&mut **tx)
            .await?;
        }
        current_by_id.insert(column_id, (order, nullable, ducklake_type));
    }

    for (order, (column, column_id)) in proposed.iter().zip(column_ids).enumerate() {
        let parent_id = column.parent_index.map(|index| column_ids[index]);
        match current_by_id.get(column_id) {
            Some((cur_order, cur_nullable, cur_type)) => {
                let migrate_type = catalog_column_type_requires_migration(cur_type, column);
                if migrate_type {
                    sqlx::query(
                        "UPDATE ducklake_column SET end_snapshot = ?
                         WHERE table_id = ? AND column_id = ? AND end_snapshot IS NULL",
                    )
                    .bind(snapshot_id)
                    .bind(table_id)
                    .bind(column_id)
                    .execute(&mut **tx)
                    .await?;
                    sqlx::query(
                        "INSERT INTO ducklake_column
                             (column_id, table_id, column_name, column_type, column_order,
                              nulls_allowed, parent_column, begin_snapshot)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    )
                    .bind(column_id)
                    .bind(table_id)
                    .bind(&column.name)
                    .bind(&column.ducklake_type)
                    .bind(order as i64)
                    .bind(column.is_nullable)
                    .bind(parent_id)
                    .bind(snapshot_id)
                    .execute(&mut **tx)
                    .await?;
                } else if *cur_order != order as i64 || *cur_nullable != column.is_nullable {
                    sqlx::query(
                        "UPDATE ducklake_column
                         SET column_order = ?, nulls_allowed = ?
                         WHERE table_id = ? AND column_id = ? AND end_snapshot IS NULL",
                    )
                    .bind(order as i64)
                    .bind(column.is_nullable)
                    .bind(table_id)
                    .bind(column_id)
                    .execute(&mut **tx)
                    .await?;
                }
            },
            // Newly added column: insert it with its reserved id.
            None => {
                sqlx::query(
                    "INSERT INTO ducklake_column
                          (column_id, table_id, column_name, column_type, column_order,
                           nulls_allowed, parent_column, begin_snapshot, initial_default,
                           default_value, default_value_type, default_value_dialect)
                      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                )
                .bind(column_id)
                .bind(table_id)
                .bind(&column.name)
                .bind(&column.ducklake_type)
                .bind(order as i64)
                .bind(column.is_nullable)
                .bind(parent_id)
                .bind(snapshot_id)
                .bind(&column.initial_default)
                .bind(&column.default_value)
                .bind(&column.default_value_type)
                .bind(&column.default_value_dialect)
                .execute(&mut **tx)
                .await?;
            },
        }
    }

    if mode == WriteMode::Replace {
        // Abort if a concurrent writer published a newer generation since this
        // write began.
        detect_replace_conflict(tx, table_id, base_snapshot).await?;
        // Seed the stats row (first write to a brand-new table) so retire's
        // zero-update has a row, then retire the prior data generation.
        sqlx::query(
            "INSERT IGNORE INTO ducklake_table_stats
                 (table_id, record_count, next_row_id, file_size_bytes)
             VALUES (?, 0, 0, 0)",
        )
        .bind(table_id)
        .execute(&mut **tx)
        .await?;
        retire_prior_generation(tx, table_id, snapshot_id).await?;
    }

    // Record the schema-change ledger row for a DDL commit. A pure data write
    // carries schema_version forward and writes no row.
    if is_ddl {
        record_schema_version(tx, snapshot_id, schema_version, table_id).await?;
    }
    Ok(snapshot_id)
}

async fn validate_staged_table(
    tx: &mut sqlx::Transaction<'_, sqlx::MySql>,
    write: &StagedTableWrite,
) -> Result<i64> {
    let schema_id: i64 = sqlx::query_scalar(
        "SELECT schema_id FROM ducklake_table
         WHERE table_id = ? AND end_snapshot IS NULL",
    )
    .bind(write.table_id)
    .fetch_optional(&mut **tx)
    .await?
    .ok_or_else(|| {
        crate::DuckLakeError::Conflict(format!(
            "multi-table write target {}.{} is no longer live",
            write.schema_name, write.table_name
        ))
    })?;
    let proposed = catalog_column_defs(&write.columns)?;
    let current = sqlx::query(
        "SELECT column_id, column_type, nulls_allowed, parent_column
         FROM ducklake_column
         WHERE table_id = ? AND end_snapshot IS NULL
         ORDER BY column_order",
    )
    .bind(write.table_id)
    .fetch_all(&mut **tx)
    .await?;
    if current.len() != proposed.len() || proposed.len() != write.column_ids.len() {
        return Err(crate::DuckLakeError::Conflict(format!(
            "multi-table write target {}.{} changed schema after staging",
            write.schema_name, write.table_name
        )));
    }
    for (index, (row, proposed)) in current.iter().zip(&proposed).enumerate() {
        let column_id: i64 = row.try_get(0)?;
        let column_type: String = row.try_get(1)?;
        let nullable = row.try_get::<Option<bool>, _>(2)?.unwrap_or(true);
        let parent_id: Option<i64> = row.try_get(3)?;
        let proposed_parent = proposed.parent_index.map(|parent| write.column_ids[parent]);
        if column_id != write.column_ids[index]
            || !catalog_column_type_equal(&column_type, proposed)
            || nullable != proposed.is_nullable
            || parent_id != proposed_parent
        {
            return Err(crate::DuckLakeError::Conflict(format!(
                "multi-table write target {}.{} changed schema after staging",
                write.schema_name, write.table_name
            )));
        }
    }
    Ok(schema_id)
}

async fn has_live_data(tx: &mut sqlx::Transaction<'_, sqlx::MySql>, table_id: i64) -> Result<bool> {
    let has_files: bool = sqlx::query_scalar(
        "SELECT EXISTS(
             SELECT 1 FROM ducklake_data_file
             WHERE table_id = ? AND end_snapshot IS NULL
         )",
    )
    .bind(table_id)
    .fetch_one(&mut **tx)
    .await?;
    if has_files {
        return Ok(true);
    }
    let inline_tables =
        sqlx::query("SELECT table_name FROM ducklake_inlined_data_tables WHERE table_id = ?")
            .bind(table_id)
            .fetch_all(&mut **tx)
            .await?;
    for row in inline_tables {
        let table_name: String = row.try_get(0)?;
        let has_rows: bool = sqlx::query_scalar(AssertSqlSafe(format!(
            "SELECT EXISTS(SELECT 1 FROM {} WHERE end_snapshot IS NULL)",
            quote_ident(&table_name)
        )))
        .fetch_one(&mut **tx)
        .await?;
        if has_rows {
            return Ok(true);
        }
    }
    Ok(false)
}

async fn commit_staged_files(
    tx: &mut sqlx::Transaction<'_, sqlx::MySql>,
    snapshot_id: i64,
    write: &StagedTableWrite,
    files: &[DataFileInfo],
) -> Result<()> {
    if files.is_empty() {
        return Err(crate::DuckLakeError::InvalidConfig(
            "multi-table file stage requires at least one file".to_string(),
        ));
    }
    let live_partition_id: Option<i64> = sqlx::query_scalar(
        "SELECT partition_id FROM ducklake_partition_info
         WHERE table_id = ? AND end_snapshot IS NULL",
    )
    .bind(write.table_id)
    .fetch_optional(&mut **tx)
    .await?;
    for file in files {
        crate::metadata_writer::enforce_partition_fence(write.table_id, live_partition_id, file)?;
    }
    sqlx::query(
        "INSERT IGNORE INTO ducklake_table_stats
             (table_id, record_count, next_row_id, file_size_bytes)
         VALUES (?, 0, 0, 0)",
    )
    .bind(write.table_id)
    .execute(&mut **tx)
    .await?;
    let mut next_row_id: i64 =
        sqlx::query_scalar("SELECT next_row_id FROM ducklake_table_stats WHERE table_id = ?")
            .bind(write.table_id)
            .fetch_one(&mut **tx)
            .await?;
    let data_file_ids = reserve_file_ids(tx, files.len() as i64).await?;
    let mut total_records = 0i64;
    let mut total_bytes = 0i64;
    for (file, data_file_id) in files.iter().zip(data_file_ids) {
        sqlx::query(
            "INSERT INTO ducklake_data_file
                 (data_file_id, table_id, path, path_is_relative, file_size_bytes,
                  footer_size, record_count, row_id_start, begin_snapshot)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(data_file_id)
        .bind(write.table_id)
        .bind(&file.path)
        .bind(file.path_is_relative)
        .bind(file.file_size_bytes)
        .bind(file.footer_size)
        .bind(file.record_count)
        .bind(next_row_id)
        .bind(snapshot_id)
        .execute(&mut **tx)
        .await?;
        insert_file_column_stats(tx, write.table_id, data_file_id, &file.column_stats).await?;
        insert_partition_metadata(tx, write.table_id, data_file_id, file).await?;
        next_row_id += file.record_count;
        total_records += file.record_count;
        total_bytes += file.file_size_bytes;
    }
    recompute_table_column_stats(tx, write.table_id, &write.columns, &write.column_ids).await?;
    sqlx::query(
        "UPDATE ducklake_table_stats
         SET next_row_id = next_row_id + ?,
             record_count = record_count + ?,
             file_size_bytes = file_size_bytes + ?
         WHERE table_id = ?",
    )
    .bind(total_records)
    .bind(total_records)
    .bind(total_bytes)
    .bind(write.table_id)
    .execute(&mut **tx)
    .await?;
    Ok(())
}

/// Build the DDL for a physical inline table. MySQL DDL implicitly commits the
/// surrounding transaction, so callers MUST execute this on the pool BEFORE the
/// write transaction opens (CREATE TABLE IF NOT EXISTS is idempotent, and an
/// existing physical table without registry rows is inert).
fn inlined_mysql_table_ddl(
    physical_name: &str,
    fields: &arrow::datatypes::Fields,
    columns: &[ColumnDef],
) -> String {
    let mut ddl = format!(
        "CREATE TABLE IF NOT EXISTS {} (\
         row_id BIGINT NOT NULL, begin_snapshot BIGINT NOT NULL, end_snapshot BIGINT",
        quote_ident(physical_name)
    );
    for (field, column) in fields.iter().zip(columns) {
        ddl.push_str(", ");
        ddl.push_str(&quote_ident(column.name()));
        ddl.push(' ');
        ddl.push_str(&inlined_mysql_type(field.data_type()));
    }
    ddl.push_str(") ENGINE = InnoDB");
    ddl
}

async fn mysql_inlined_index_columns(pool: &MySqlPool, table_id: i64) -> Result<Vec<String>> {
    let value: Option<String> = sqlx::query_scalar(
        "SELECT value FROM ducklake_metadata
         WHERE `key` = ? AND scope = 'table' AND scope_id = ?",
    )
    .bind(INLINED_INDEX_COLUMNS_SETTING)
    .bind(table_id)
    .fetch_optional(pool)
    .await?;
    let declared = parse_inlined_index_columns(value.as_deref().unwrap_or_default())?;
    let available = sqlx::query_scalar::<_, String>(
        "SELECT column_name FROM ducklake_column
         WHERE table_id = ? AND end_snapshot IS NULL AND parent_column IS NULL",
    )
    .bind(table_id)
    .fetch_all(pool)
    .await?
    .into_iter()
    .collect::<std::collections::HashSet<_>>();
    Ok(live_inlined_index_columns(declared, &available))
}

async fn ensure_mysql_physical_indexes(
    pool: &MySqlPool,
    table_id: i64,
    physical_name: &str,
) -> Result<()> {
    let declared = mysql_inlined_index_columns(pool, table_id).await?;
    let physical_types = sqlx::query_as::<_, (String, String)>(
        "SELECT column_name, column_type FROM information_schema.columns
         WHERE table_schema = DATABASE() AND table_name = ?",
    )
    .bind(physical_name)
    .fetch_all(pool)
    .await?
    .into_iter()
    .collect::<std::collections::HashMap<_, _>>();
    for column in std::iter::once("row_id").chain(declared.iter().map(String::as_str)) {
        let Some(physical_type) = physical_types.get(column) else {
            continue;
        };
        if !mysql_type_supports_index(physical_type) {
            continue;
        }
        let index_name = format!("{physical_name}_{column}_idx");
        let exists: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM information_schema.statistics
             WHERE table_schema = DATABASE() AND table_name = ? AND index_name = ?",
        )
        .bind(physical_name)
        .bind(&index_name)
        .fetch_one(pool)
        .await?;
        if exists > 0 {
            continue;
        }
        let create = sqlx::query(AssertSqlSafe(format!(
            "CREATE INDEX {} ON {}({})",
            quote_ident(&index_name),
            quote_ident(physical_name),
            quote_ident(column),
        )))
        .execute(pool)
        .await;
        if let Err(error) = create {
            let exists: i64 = sqlx::query_scalar(
                "SELECT COUNT(*) FROM information_schema.statistics
                 WHERE table_schema = DATABASE() AND table_name = ? AND index_name = ?",
            )
            .bind(physical_name)
            .bind(&index_name)
            .fetch_one(pool)
            .await?;
            if exists == 0 {
                return Err(error.into());
            }
        }
    }
    Ok(())
}

async fn commit_staged_inline(
    tx: &mut sqlx::Transaction<'_, sqlx::MySql>,
    snapshot_id: i64,
    write: &StagedTableWrite,
    batches: &[RecordBatch],
) -> Result<()> {
    let record_count: usize = batches.iter().map(RecordBatch::num_rows).sum();
    if record_count == 0 {
        return Err(crate::DuckLakeError::InvalidConfig(
            "multi-table inline stage requires at least one row".to_string(),
        ));
    }
    let schema_version: i64 =
        sqlx::query_scalar("SELECT schema_version FROM ducklake_snapshot WHERE snapshot_id = ?")
            .bind(snapshot_id)
            .fetch_one(&mut **tx)
            .await?;
    let physical_name = format!(
        "ducklake_inlined_data_{}_{}",
        write.table_id, schema_version
    );
    // The physical table was created before this transaction opened (MySQL DDL
    // would implicitly commit it). A missing table means the schema version
    // moved between pre-creation and this commit; abort so the caller retries.
    let physical_exists: bool = sqlx::query_scalar(
        "SELECT COUNT(*) > 0 FROM information_schema.tables
         WHERE table_schema = DATABASE() AND table_name = ?",
    )
    .bind(&physical_name)
    .fetch_one(&mut **tx)
    .await?;
    if !physical_exists {
        return Err(crate::DuckLakeError::Conflict(format!(
            "multi-table inline stage for table {} resolved schema version {schema_version}, \
             whose inline table does not exist yet; retry the commit",
            write.table_id
        )));
    }
    sqlx::query(
        "INSERT INTO ducklake_inlined_data_tables (table_id, table_name, schema_version)
         SELECT ?, ?, ? FROM DUAL WHERE NOT EXISTS (
             SELECT 1 FROM ducklake_inlined_data_tables
             WHERE table_id = ? AND schema_version = ?
         )",
    )
    .bind(write.table_id)
    .bind(&physical_name)
    .bind(schema_version)
    .bind(write.table_id)
    .bind(schema_version)
    .execute(&mut **tx)
    .await?;
    sqlx::query(
        "INSERT IGNORE INTO ducklake_table_stats
             (table_id, record_count, next_row_id, file_size_bytes)
         VALUES (?, 0, 0, 0)",
    )
    .bind(write.table_id)
    .execute(&mut **tx)
    .await?;
    let mut row_id: i64 =
        sqlx::query_scalar("SELECT next_row_id FROM ducklake_table_stats WHERE table_id = ?")
            .bind(write.table_id)
            .fetch_one(&mut **tx)
            .await?;
    let column_list = write
        .columns
        .iter()
        .map(|column| quote_ident(column.name()))
        .collect::<Vec<_>>()
        .join(", ");
    for batch in batches {
        for batch_row in 0..batch.num_rows() {
            let mut query = QueryBuilder::<MySql>::new(format!(
                "INSERT INTO {} (row_id, begin_snapshot, end_snapshot, {}) VALUES (",
                quote_ident(&physical_name),
                column_list
            ));
            query.push_bind(row_id);
            query.push(", ").push_bind(snapshot_id);
            query.push(", NULL");
            for (array, column) in batch.columns().iter().zip(&write.columns) {
                query.push(", ");
                if write
                    .snapshot_id_columns
                    .iter()
                    .any(|name| name == column.name())
                    && array.is_null(batch_row)
                {
                    query.push_bind(snapshot_id);
                } else {
                    push_inlined_mysql_value(
                        &mut query,
                        array.as_ref(),
                        batch_row,
                        &inlined_mysql_type(array.data_type()),
                    )?;
                }
            }
            query.push(')');
            query.build().execute(&mut **tx).await?;
            row_id += 1;
        }
    }
    let record_count = i64::try_from(record_count).map_err(|_| {
        crate::DuckLakeError::InvalidConfig("multi-table inline row count exceeds i64".to_string())
    })?;
    sqlx::query(
        "UPDATE ducklake_table_stats
         SET next_row_id = next_row_id + ?, record_count = record_count + ?
         WHERE table_id = ?",
    )
    .bind(record_count)
    .bind(record_count)
    .bind(write.table_id)
    .execute(&mut **tx)
    .await?;
    Ok(())
}

async fn apply_staged_inlined_deletes(
    tx: &mut sqlx::Transaction<'_, sqlx::MySql>,
    snapshot_id: i64,
    write: &StagedTableWrite,
) -> Result<()> {
    if write.inlined_deletes.is_empty() {
        return Ok(());
    }
    let registered =
        sqlx::query("SELECT table_name FROM ducklake_inlined_data_tables WHERE table_id = ?")
            .bind(write.table_id)
            .fetch_all(&mut **tx)
            .await?
            .into_iter()
            .map(|row| row.try_get(0))
            .collect::<std::result::Result<std::collections::HashSet<String>, _>>()?;
    for row in &write.inlined_deletes {
        if !registered.contains(&row.table_name) {
            return Err(crate::DuckLakeError::Conflict(format!(
                "inlined row {} belongs to an unregistered table '{}'",
                row.row_id, row.table_name
            )));
        }
        let affected = sqlx::query(AssertSqlSafe(format!(
            "UPDATE {} SET end_snapshot = ? \
             WHERE row_id = ? AND begin_snapshot <= ? AND end_snapshot IS NULL",
            quote_ident(&row.table_name)
        )))
        .bind(snapshot_id)
        .bind(row.row_id)
        .bind(write.base_snapshot_id)
        .execute(&mut **tx)
        .await?
        .rows_affected();
        if affected != 1 {
            return Err(crate::DuckLakeError::Conflict(format!(
                "inlined row {} in '{}' is no longer live at snapshot {}",
                row.row_id, row.table_name, write.base_snapshot_id
            )));
        }
    }
    let deleted = i64::try_from(write.inlined_deletes.len()).map_err(|_| {
        crate::DuckLakeError::InvalidConfig(
            "multi-table inline delete count exceeds i64".to_string(),
        )
    })?;
    sqlx::query(
        "UPDATE ducklake_table_stats
         SET record_count = GREATEST(record_count - ?, 0) WHERE table_id = ?",
    )
    .bind(deleted)
    .bind(write.table_id)
    .execute(&mut **tx)
    .await?;
    Ok(())
}

impl MetadataWriter for MySqlMetadataWriter {
    fn supports_update(&self) -> bool {
        true
    }
    fn set_table_setting(&self, table_id: i64, key: &str, value: &str) -> Result<()> {
        block_on(async {
            let mut transaction = self.pool.begin().await?;
            let exists: bool = sqlx::query_scalar(
                "SELECT COUNT(*) > 0 FROM ducklake_table
                 WHERE table_id = ? AND end_snapshot IS NULL",
            )
            .bind(table_id)
            .fetch_one(&mut *transaction)
            .await?;
            if !exists {
                return Err(crate::DuckLakeError::TableNotFound(table_id.to_string()));
            }
            sqlx::query(
                "DELETE FROM ducklake_metadata
                 WHERE `key` = ? AND scope = 'table' AND scope_id = ?",
            )
            .bind(key)
            .bind(table_id)
            .execute(&mut *transaction)
            .await?;
            sqlx::query(
                "INSERT INTO ducklake_metadata (`key`, value, scope, scope_id)
                 VALUES (?, ?, 'table', ?)",
            )
            .bind(key)
            .bind(value)
            .bind(table_id)
            .execute(&mut *transaction)
            .await?;
            transaction.commit().await?;
            Ok(())
        })
    }

    fn set_inlined_index_columns(&self, table_id: i64, columns: &[String]) -> Result<()> {
        let value = encode_inlined_index_columns(columns)?;
        block_on(async {
            let mut transaction = self.pool.begin().await?;
            let available = sqlx::query_scalar::<_, String>(
                "SELECT column_name FROM ducklake_column
                 WHERE table_id = ? AND end_snapshot IS NULL AND parent_column IS NULL",
            )
            .bind(table_id)
            .fetch_all(&mut *transaction)
            .await?
            .into_iter()
            .collect::<std::collections::HashSet<_>>();
            if available.is_empty() {
                return Err(crate::DuckLakeError::TableNotFound(table_id.to_string()));
            }
            validate_inlined_index_columns(columns, &available)?;
            let existing: Option<String> = sqlx::query_scalar(
                "SELECT value FROM ducklake_metadata
                 WHERE `key` = ? AND scope = 'table' AND scope_id = ?",
            )
            .bind(INLINED_INDEX_COLUMNS_SETTING)
            .bind(table_id)
            .fetch_optional(&mut *transaction)
            .await?;
            if existing.as_deref() == Some(value.as_str()) {
                transaction.commit().await?;
                return Ok(());
            }
            sqlx::query(
                "DELETE FROM ducklake_metadata
                 WHERE `key` = ? AND scope = 'table' AND scope_id = ?",
            )
            .bind(INLINED_INDEX_COLUMNS_SETTING)
            .bind(table_id)
            .execute(&mut *transaction)
            .await?;
            sqlx::query(
                "INSERT INTO ducklake_metadata (`key`, value, scope, scope_id)
                 VALUES (?, ?, 'table', ?)",
            )
            .bind(INLINED_INDEX_COLUMNS_SETTING)
            .bind(value)
            .bind(table_id)
            .execute(&mut *transaction)
            .await?;
            transaction.commit().await?;
            Ok(())
        })
    }

    fn ensure_inlined_indexes(&self, table_id: i64) -> Result<()> {
        block_on(async {
            let physical_tables = sqlx::query_scalar::<_, String>(
                "SELECT table_name FROM ducklake_inlined_data_tables WHERE table_id = ?",
            )
            .bind(table_id)
            .fetch_all(&self.pool)
            .await?;
            mysql_inlined_index_columns(&self.pool, table_id).await?;
            for physical_table in physical_tables {
                ensure_mysql_physical_indexes(&self.pool, table_id, &physical_table).await?;
            }
            Ok(())
        })
    }

    fn create_snapshot(&self) -> Result<i64> {
        block_on(async {
            let mut tx = self.pool.begin().await?;
            // A bare snapshot carries no schema change of its own → carry
            // schema_version forward (no DDL bump, no ledger row).
            let (snapshot_id, _schema_version) = insert_snapshot(&mut tx).await?;
            tx.commit().await?;
            Ok(snapshot_id)
        })
    }

    fn get_or_create_schema(
        &self,
        name: &str,
        path: Option<&str>,
        snapshot_id: i64,
    ) -> Result<(i64, bool)> {
        validate_name(name, "Schema")?;
        block_on(async {
            let mut tx = self.pool.begin().await?;
            let existing = sqlx::query(
                "SELECT schema_id FROM ducklake_schema
                 WHERE schema_name = ? AND end_snapshot IS NULL",
            )
            .bind(name)
            .fetch_optional(&mut *tx)
            .await?;

            if let Some(row) = existing {
                tx.commit().await?;
                return Ok((row.try_get(0)?, false));
            }

            let schema_path = path.unwrap_or(name);
            // No RETURNING: read the new auto-increment id via last_insert_id().
            let result = sqlx::query(
                "INSERT INTO ducklake_schema (schema_name, path, path_is_relative, begin_snapshot)
                 VALUES (?, ?, 1, ?)",
            )
            .bind(name)
            .bind(schema_path)
            .bind(snapshot_id)
            .execute(&mut *tx)
            .await?;

            record_snapshot_changes(
                &mut tx,
                snapshot_id,
                &format!("created_schema:{}", quote_snapshot_name(name)),
                &SnapshotCommitMetadata::default(),
            )
            .await?;
            tx.commit().await?;
            Ok((result.last_insert_id() as i64, true))
        })
    }

    fn get_or_create_table(
        &self,
        schema_id: i64,
        name: &str,
        path: Option<&str>,
        snapshot_id: i64,
    ) -> Result<(i64, bool)> {
        validate_name(name, "Table")?;
        block_on(async {
            let mut tx = self.pool.begin().await?;
            let existing = sqlx::query(
                "SELECT table_id FROM ducklake_table
                 WHERE schema_id = ? AND table_name = ? AND end_snapshot IS NULL",
            )
            .bind(schema_id)
            .bind(name)
            .fetch_optional(&mut *tx)
            .await?;

            if let Some(row) = existing {
                tx.commit().await?;
                return Ok((row.try_get(0)?, false));
            }

            let schema_name: String =
                sqlx::query_scalar("SELECT schema_name FROM ducklake_schema WHERE schema_id = ?")
                    .bind(schema_id)
                    .fetch_one(&mut *tx)
                    .await?;

            let table_path = path.unwrap_or(name);
            let result = sqlx::query(
                "INSERT INTO ducklake_table (schema_id, table_name, path, path_is_relative, begin_snapshot)
                 VALUES (?, ?, ?, 1, ?)",
            )
            .bind(schema_id)
            .bind(name)
            .bind(table_path)
            .bind(snapshot_id)
            .execute(&mut *tx)
            .await?;

            record_snapshot_changes(
                &mut tx,
                snapshot_id,
                &format!("created_table:{}", quote_snapshot_table(&schema_name, name)),
                &SnapshotCommitMetadata::default(),
            )
            .await?;
            tx.commit().await?;
            Ok((result.last_insert_id() as i64, true))
        })
    }

    fn promote_column_type(
        &self,
        table_id: i64,
        column_name: &str,
        new_ducklake_type: &str,
    ) -> Result<i64> {
        crate::types::ducklake_to_arrow_type(new_ducklake_type)?;
        block_on(async {
            let mut tx = self.pool.begin().await?;
            let (snapshot_id, _carried) = insert_snapshot(&mut tx).await?;
            let row = sqlx::query(
                "SELECT column_id, column_type, column_order, nulls_allowed
                 FROM ducklake_column
                 WHERE table_id = ? AND column_name = ? AND end_snapshot IS NULL",
            )
            .bind(table_id)
            .bind(column_name)
            .fetch_optional(&mut *tx)
            .await?
            .ok_or_else(|| {
                crate::DuckLakeError::InvalidConfig(format!(
                    "promote_column_type: no live column '{column_name}' in table {table_id}"
                ))
            })?;
            let column_id: i64 = row.try_get("column_id")?;
            let current_type: String = row.try_get("column_type")?;
            let column_order: i64 = row.try_get("column_order")?;
            let nulls_allowed = row
                .try_get::<Option<bool>, _>("nulls_allowed")?
                .unwrap_or(true);
            if crate::types::types_equal_canonical(&current_type, new_ducklake_type) {
                return Err(crate::DuckLakeError::InvalidConfig(format!(
                    "promote_column_type: column '{column_name}' is already type '{current_type}' (no change)"
                )));
            }
            if !crate::types::is_promotable(&current_type, new_ducklake_type) {
                return Err(crate::DuckLakeError::UnsupportedTypeChange {
                    operation: TypeChangeOperation::PromoteColumnType,
                    column: column_name.to_string(),
                    from: current_type,
                    to: new_ducklake_type.to_string(),
                });
            }
            let schema_version = bump_schema_version(&mut tx, snapshot_id).await?;
            sqlx::query(
                "UPDATE ducklake_column SET end_snapshot = ?
                 WHERE table_id = ? AND column_id = ? AND end_snapshot IS NULL",
            )
            .bind(snapshot_id)
            .bind(table_id)
            .bind(column_id)
            .execute(&mut *tx)
            .await?;
            sqlx::query(
                "INSERT INTO ducklake_column
                     (column_id, begin_snapshot, end_snapshot, table_id, column_order,
                      column_name, column_type, nulls_allowed)
                 VALUES (?, ?, NULL, ?, ?, ?, ?, ?)",
            )
            .bind(column_id)
            .bind(snapshot_id)
            .bind(table_id)
            .bind(column_order)
            .bind(column_name)
            .bind(new_ducklake_type)
            .bind(nulls_allowed)
            .execute(&mut *tx)
            .await?;
            record_schema_version(&mut tx, snapshot_id, schema_version, table_id).await?;
            record_snapshot_changes(
                &mut tx,
                snapshot_id,
                &format!("altered_table:{table_id}"),
                &SnapshotCommitMetadata::default(),
            )
            .await?;
            tx.commit().await?;
            Ok(snapshot_id)
        })
    }

    fn set_columns(
        &self,
        table_id: i64,
        columns: &[ColumnDef],
        snapshot_id: i64,
    ) -> Result<Vec<i64>> {
        if columns.is_empty() {
            return Err(crate::DuckLakeError::InvalidConfig(
                "Table must have at least one column".to_string(),
            ));
        }
        block_on(async {
            // Transaction for atomicity: if column insertion fails, we don't leave
            // existing columns marked as ended.
            let mut tx = self.pool.begin().await?;

            sqlx::query(
                "UPDATE ducklake_column SET end_snapshot = ?
                 WHERE table_id = ? AND end_snapshot IS NULL",
            )
            .bind(snapshot_id)
            .bind(table_id)
            .execute(&mut *tx)
            .await?;

            // Reserve a contiguous column_id block from the monotonic counter and
            // insert with explicit ids, keeping the allocator authoritative.
            let catalog_columns = catalog_column_defs(columns)?;
            let n = catalog_columns.len() as i64;
            let last_column_id = reserve_ids(&mut tx, "next_column_id", n).await?;
            let first_column_id = last_column_id - n + 1;
            let field_ids = (first_column_id..=last_column_id).collect::<Vec<_>>();
            for (order, (column, column_id)) in
                catalog_columns.iter().zip(field_ids.iter()).enumerate()
            {
                let parent_id = column.parent_index.map(|index| field_ids[index]);
                sqlx::query(
                    "INSERT INTO ducklake_column
                          (column_id, table_id, column_name, column_type, column_order,
                           nulls_allowed, parent_column, begin_snapshot, initial_default,
                           default_value, default_value_type, default_value_dialect)
                      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                )
                .bind(column_id)
                .bind(table_id)
                .bind(&column.name)
                .bind(&column.ducklake_type)
                .bind(order as i64)
                .bind(column.is_nullable)
                .bind(parent_id)
                .bind(snapshot_id)
                .bind(&column.initial_default)
                .bind(&column.default_value)
                .bind(&column.default_value_type)
                .bind(&column.default_value_dialect)
                .execute(&mut *tx)
                .await?;
            }

            let table_begin_snapshot: i64 =
                sqlx::query_scalar("SELECT begin_snapshot FROM ducklake_table WHERE table_id = ?")
                    .bind(table_id)
                    .fetch_one(&mut *tx)
                    .await?;
            if table_begin_snapshot != snapshot_id {
                record_snapshot_changes(
                    &mut tx,
                    snapshot_id,
                    &format!("altered_table:{table_id}"),
                    &SnapshotCommitMetadata::default(),
                )
                .await?;
            }
            tx.commit().await?;
            top_level_column_ids(&catalog_columns, &field_ids)
        })
    }

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
    ) -> Result<CommitIds> {
        self.register_data_file_with_commit_metadata(
            table_id,
            schema_name,
            table_name,
            snapshot_id,
            file,
            mode,
            base_snapshot,
            columns,
            column_ids,
            &SnapshotCommitMetadata::default(),
            None,
        )
    }

    fn register_data_file_with_commit_metadata(
        &self,
        table_id: i64,
        schema_name: &str,
        table_name: &str,
        _snapshot_id: i64,
        file: &DataFileInfo,
        mode: WriteMode,
        base_snapshot: i64,
        columns: &[ColumnDef],
        column_ids: &[i64],
        commit_metadata: &SnapshotCommitMetadata,
        expected_base_snapshot_id: Option<i64>,
    ) -> Result<CommitIds> {
        if expected_base_snapshot_id.is_some() {
            return Err(crate::DuckLakeError::InvalidConfig(
                "conditional writes are not supported by the MySQL metadata writer".to_string(),
            ));
        }
        block_on(async {
            // Single atomic commit: insert the deferred snapshot row + finalize the
            // column generation + retire the prior generation (Replace), then
            // register this file and advance the monotonic row-lineage counter —
            // all in one transaction, so the head only ever resolves to
            // fully-populated data.
            let mut tx = self.pool.begin().await?;

            let snapshot_id =
                finalize_snapshot(&mut tx, table_id, columns, column_ids, mode, base_snapshot)
                    .await?;

            // Partition-spec fence: this file must be consistent with the table's live
            // partition generation at commit time (both directions — see
            // enforce_partition_fence). The tx rolls back on a Conflict.
            let live_partition_id: Option<i64> = sqlx::query_scalar(
                "SELECT partition_id FROM ducklake_partition_info
                 WHERE table_id = ? AND end_snapshot IS NULL",
            )
            .bind(table_id)
            .fetch_optional(&mut *tx)
            .await?;
            crate::metadata_writer::enforce_partition_fence(table_id, live_partition_id, file)?;

            // Seed the stats row for the Append path (Replace already seeded it in
            // finalize_snapshot); INSERT IGNORE is a no-op if it exists.
            sqlx::query(
                "INSERT IGNORE INTO ducklake_table_stats
                     (table_id, record_count, next_row_id, file_size_bytes)
                 VALUES (?, 0, 0, 0)",
            )
            .bind(table_id)
            .execute(&mut *tx)
            .await?;

            let row_id_start: i64 =
                sqlx::query("SELECT next_row_id FROM ducklake_table_stats WHERE table_id = ?")
                    .bind(table_id)
                    .fetch_one(&mut *tx)
                    .await?
                    .try_get(0)?;

            // The id comes from the shared next_file_id counter (never the
            // auto-increment), so appends and the update/delete/compaction paths
            // allocate from one id space and cannot collide on the PK.
            let data_file_id = reserve_file_ids(&mut tx, 1).await?[0];
            sqlx::query(
                "INSERT INTO ducklake_data_file
                     (data_file_id, table_id, path, path_is_relative, file_size_bytes,
                      footer_size, record_count, row_id_start, begin_snapshot)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            )
            .bind(data_file_id)
            .bind(table_id)
            .bind(&file.path)
            .bind(file.path_is_relative)
            .bind(file.file_size_bytes)
            .bind(file.footer_size)
            .bind(file.record_count)
            .bind(row_id_start)
            .bind(snapshot_id)
            .execute(&mut *tx)
            .await?;

            // Persist the file's zone maps + refresh the roll-up.
            insert_file_column_stats(&mut tx, table_id, data_file_id, &file.column_stats).await?;
            insert_partition_metadata(&mut tx, table_id, data_file_id, file).await?;
            recompute_table_column_stats(&mut tx, table_id, columns, column_ids).await?;

            // Advance the counter and accumulate stats. `next_row_id`
            // monotonically increases over the table's lifetime.
            sqlx::query(
                "UPDATE ducklake_table_stats
                 SET next_row_id     = next_row_id + ?,
                     record_count    = record_count + ?,
                     file_size_bytes = file_size_bytes + ?
                 WHERE table_id = ?",
            )
            .bind(file.record_count)
            .bind(file.record_count)
            .bind(file.file_size_bytes)
            .bind(table_id)
            .execute(&mut *tx)
            .await?;

            record_table_write_changes(
                &mut tx,
                snapshot_id,
                table_id,
                schema_name,
                table_name,
                mode,
                false,
                commit_metadata,
            )
            .await?;
            let schema_id: i64 =
                sqlx::query("SELECT schema_id FROM ducklake_table WHERE table_id = ?")
                    .bind(table_id)
                    .fetch_one(&mut *tx)
                    .await?
                    .try_get(0)?;

            tx.commit().await?;
            Ok(CommitIds {
                snapshot_id,
                schema_id,
                table_id,
            })
        })
    }

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
        self.register_data_files_with_commit_metadata(
            table_id,
            schema_name,
            table_name,
            snapshot_id,
            files,
            mode,
            base_snapshot,
            columns,
            column_ids,
            &SnapshotCommitMetadata::default(),
            None,
        )
    }

    fn register_data_files_with_commit_metadata(
        &self,
        table_id: i64,
        schema_name: &str,
        table_name: &str,
        _snapshot_id: i64,
        files: &[DataFileInfo],
        mode: WriteMode,
        base_snapshot: i64,
        columns: &[ColumnDef],
        column_ids: &[i64],
        commit_metadata: &SnapshotCommitMetadata,
        expected_base_snapshot_id: Option<i64>,
    ) -> Result<CommitIds> {
        if expected_base_snapshot_id.is_some() {
            return Err(crate::DuckLakeError::InvalidConfig(
                "conditional multi-file writes are not supported by the MySQL metadata writer"
                    .to_string(),
            ));
        }
        if files.is_empty() {
            return Err(crate::DuckLakeError::InvalidConfig(
                "register_data_files: files must be non-empty".to_string(),
            ));
        }
        block_on(async {
            let mut tx = self.pool.begin().await?;
            let snapshot_id =
                finalize_snapshot(&mut tx, table_id, columns, column_ids, mode, base_snapshot)
                    .await?;
            // Partition-spec fence (both directions, every file): each file must be
            // consistent with the table's live partition generation at commit time.
            // The tx rolls back on a Conflict.
            let live_partition_id: Option<i64> = sqlx::query_scalar(
                "SELECT partition_id FROM ducklake_partition_info
                 WHERE table_id = ? AND end_snapshot IS NULL",
            )
            .bind(table_id)
            .fetch_optional(&mut *tx)
            .await?;
            for file in files {
                crate::metadata_writer::enforce_partition_fence(table_id, live_partition_id, file)?;
            }
            sqlx::query(
                "INSERT IGNORE INTO ducklake_table_stats
                     (table_id, record_count, next_row_id, file_size_bytes)
                 VALUES (?, 0, 0, 0)",
            )
            .bind(table_id)
            .execute(&mut *tx)
            .await?;
            let mut next_row_id: i64 =
                sqlx::query("SELECT next_row_id FROM ducklake_table_stats WHERE table_id = ?")
                    .bind(table_id)
                    .fetch_one(&mut *tx)
                    .await?
                    .try_get(0)?;
            let mut total_records: i64 = 0;
            let mut total_bytes: i64 = 0;
            let file_count = i64::try_from(files.len()).map_err(|_| {
                crate::DuckLakeError::InvalidConfig(
                    "register_data_files file count exceeds i64".to_string(),
                )
            })?;
            // Explicit ids from the shared next_file_id counter (never the
            // auto-increment) keep every insert path in one id space.
            let data_file_ids = reserve_file_ids(&mut tx, file_count).await?;
            for (file, data_file_id) in files.iter().zip(data_file_ids) {
                sqlx::query(
                    "INSERT INTO ducklake_data_file
                         (data_file_id, table_id, path, path_is_relative, file_size_bytes,
                          footer_size, record_count, row_id_start, begin_snapshot)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                )
                .bind(data_file_id)
                .bind(table_id)
                .bind(&file.path)
                .bind(file.path_is_relative)
                .bind(file.file_size_bytes)
                .bind(file.footer_size)
                .bind(file.record_count)
                .bind(next_row_id)
                .bind(snapshot_id)
                .execute(&mut *tx)
                .await?;
                insert_file_column_stats(&mut tx, table_id, data_file_id, &file.column_stats)
                    .await?;
                insert_partition_metadata(&mut tx, table_id, data_file_id, file).await?;
                next_row_id += file.record_count;
                total_records += file.record_count;
                total_bytes += file.file_size_bytes;
            }
            recompute_table_column_stats(&mut tx, table_id, columns, column_ids).await?;
            sqlx::query(
                "UPDATE ducklake_table_stats
                 SET next_row_id     = next_row_id + ?,
                     record_count    = record_count + ?,
                     file_size_bytes = file_size_bytes + ?
                 WHERE table_id = ?",
            )
            .bind(total_records)
            .bind(total_records)
            .bind(total_bytes)
            .bind(table_id)
            .execute(&mut *tx)
            .await?;
            record_table_write_changes(
                &mut tx,
                snapshot_id,
                table_id,
                schema_name,
                table_name,
                mode,
                false,
                commit_metadata,
            )
            .await?;
            let schema_id: i64 =
                sqlx::query("SELECT schema_id FROM ducklake_table WHERE table_id = ?")
                    .bind(table_id)
                    .fetch_one(&mut *tx)
                    .await?
                    .try_get(0)?;
            tx.commit().await?;
            Ok(CommitIds {
                snapshot_id,
                schema_id,
                table_id,
            })
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn supports_data_inlining(&self, schema: &arrow::datatypes::Schema) -> bool {
        schema
            .fields()
            .iter()
            .all(|field| mysql_type_inlines(field.data_type()))
    }

    #[allow(clippy::too_many_arguments)]
    fn set_delete_file(
        &self,
        table_id: i64,
        _schema_name: &str,
        _table_name: &str,
        _snapshot_id: i64,
        data_file_id: i64,
        expected_prev_delete_file: Option<i64>,
        base_snapshot: i64,
        delete: &DeleteFileInfo,
    ) -> Result<CommitIds> {
        block_on(async {
            let mut tx = self.pool.begin().await?;
            let (snapshot_id, _schema_version) = insert_snapshot(&mut tx).await?;
            apply_delete_entry(
                &mut tx,
                table_id,
                base_snapshot,
                snapshot_id,
                &DeleteFileEntry {
                    data_file_id,
                    expected_prev_delete_file,
                    delete: delete.clone(),
                },
            )
            .await?;
            record_snapshot_changes(
                &mut tx,
                snapshot_id,
                &format!("deleted_from_table:{table_id}"),
                &SnapshotCommitMetadata::default(),
            )
            .await?;
            let schema_id: i64 =
                sqlx::query_scalar("SELECT schema_id FROM ducklake_table WHERE table_id = ?")
                    .bind(table_id)
                    .fetch_one(&mut *tx)
                    .await?;
            tx.commit().await?;
            Ok(CommitIds {
                snapshot_id,
                schema_id,
                table_id,
            })
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn register_data_file_with_deletes(
        &self,
        table_id: i64,
        schema_name: &str,
        table_name: &str,
        _snapshot_id: i64,
        file: &DataFileInfo,
        deletes: &[DeleteFileEntry],
        mode: WriteMode,
        base_snapshot: i64,
        columns: &[ColumnDef],
        column_ids: &[i64],
    ) -> Result<CommitIds> {
        validate_delete_entries(mode, deletes)?;
        block_on(async {
            let mut tx = self.pool.begin().await?;
            let snapshot_id =
                finalize_snapshot(&mut tx, table_id, columns, column_ids, mode, base_snapshot)
                    .await?;
            let data_file_ids = deletes
                .iter()
                .map(|entry| entry.data_file_id)
                .collect::<Vec<_>>();
            detect_new_inlined_deletes(&mut tx, table_id, base_snapshot, &data_file_ids).await?;
            let live_partition_id: Option<i64> = sqlx::query_scalar(
                "SELECT partition_id FROM ducklake_partition_info
                 WHERE table_id = ? AND end_snapshot IS NULL",
            )
            .bind(table_id)
            .fetch_optional(&mut *tx)
            .await?;
            crate::metadata_writer::enforce_partition_fence(table_id, live_partition_id, file)?;
            sqlx::query(
                "INSERT IGNORE INTO ducklake_table_stats
                     (table_id, record_count, next_row_id, file_size_bytes)
                 VALUES (?, 0, 0, 0)",
            )
            .bind(table_id)
            .execute(&mut *tx)
            .await?;
            let row_id_start: i64 = sqlx::query_scalar(
                "SELECT next_row_id FROM ducklake_table_stats WHERE table_id = ?",
            )
            .bind(table_id)
            .fetch_one(&mut *tx)
            .await?;
            let data_file_id = reserve_file_ids(&mut tx, 1).await?[0];
            sqlx::query(
                "INSERT INTO ducklake_data_file
                     (data_file_id, table_id, path, path_is_relative, file_size_bytes,
                      footer_size, record_count, row_id_start, begin_snapshot)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            )
            .bind(data_file_id)
            .bind(table_id)
            .bind(&file.path)
            .bind(file.path_is_relative)
            .bind(file.file_size_bytes)
            .bind(file.footer_size)
            .bind(file.record_count)
            .bind(row_id_start)
            .bind(snapshot_id)
            .execute(&mut *tx)
            .await?;
            insert_file_column_stats(&mut tx, table_id, data_file_id, &file.column_stats).await?;
            insert_partition_metadata(&mut tx, table_id, data_file_id, file).await?;
            recompute_table_column_stats(&mut tx, table_id, columns, column_ids).await?;
            sqlx::query(
                "UPDATE ducklake_table_stats
                 SET next_row_id = next_row_id + ?, record_count = record_count + ?,
                     file_size_bytes = file_size_bytes + ? WHERE table_id = ?",
            )
            .bind(file.record_count)
            .bind(file.record_count)
            .bind(file.file_size_bytes)
            .bind(table_id)
            .execute(&mut *tx)
            .await?;
            for entry in deletes {
                apply_delete_entry(&mut tx, table_id, base_snapshot, snapshot_id, entry).await?;
            }
            record_table_write_changes(
                &mut tx,
                snapshot_id,
                table_id,
                schema_name,
                table_name,
                mode,
                !deletes.is_empty(),
                &SnapshotCommitMetadata::default(),
            )
            .await?;
            let schema_id: i64 =
                sqlx::query_scalar("SELECT schema_id FROM ducklake_table WHERE table_id = ?")
                    .bind(table_id)
                    .fetch_one(&mut *tx)
                    .await?;
            tx.commit().await?;
            Ok(CommitIds {
                snapshot_id,
                schema_id,
                table_id,
            })
        })
    }

    fn commit_positional_deletes(
        &self,
        table_id: i64,
        _schema_name: &str,
        _table_name: &str,
        base_snapshot: i64,
        deletes: &[DeleteFileEntry],
    ) -> Result<CommitIds> {
        if deletes.is_empty() {
            return Err(crate::DuckLakeError::InvalidConfig(
                "commit_positional_deletes requires at least one delete entry".to_string(),
            ));
        }
        validate_delete_entries(WriteMode::Append, deletes)?;
        block_on(async {
            let mut tx = self.pool.begin().await?;
            let (snapshot_id, _schema_version) = insert_snapshot(&mut tx).await?;
            let data_file_ids = deletes
                .iter()
                .map(|entry| entry.data_file_id)
                .collect::<Vec<_>>();
            detect_new_inlined_deletes(&mut tx, table_id, base_snapshot, &data_file_ids).await?;
            for entry in deletes {
                apply_delete_entry(&mut tx, table_id, base_snapshot, snapshot_id, entry).await?;
            }
            record_snapshot_changes(
                &mut tx,
                snapshot_id,
                &format!("deleted_from_table:{table_id}"),
                &SnapshotCommitMetadata::default(),
            )
            .await?;
            let schema_id: i64 =
                sqlx::query_scalar("SELECT schema_id FROM ducklake_table WHERE table_id = ?")
                    .bind(table_id)
                    .fetch_one(&mut *tx)
                    .await?;
            tx.commit().await?;
            Ok(CommitIds {
                snapshot_id,
                schema_id,
                table_id,
            })
        })
    }

    fn commit_deletes(
        &self,
        table_id: i64,
        schema_name: &str,
        table_name: &str,
        base_snapshot: i64,
        positional: &[DeleteFileEntry],
        inlined: &[InlinedRowRef],
    ) -> Result<CommitIds> {
        // Mixed positional + inlined DELETE in ONE snapshot/transaction,
        // mirroring the SQLite writer's override (the trait default refuses the
        // combination). Single-form deletes keep their dedicated paths.
        if inlined.is_empty() {
            return self.commit_positional_deletes(
                table_id,
                schema_name,
                table_name,
                base_snapshot,
                positional,
            );
        }
        if positional.is_empty() {
            return self.commit_inlined_deletes(
                table_id,
                schema_name,
                table_name,
                base_snapshot,
                inlined,
            );
        }
        validate_delete_entries(WriteMode::Append, positional)?;
        if inlined
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len()
            != inlined.len()
        {
            return Err(crate::DuckLakeError::InvalidConfig(
                "commit_deletes contains duplicate inlined rows".to_string(),
            ));
        }
        block_on(async {
            let mut tx = self.pool.begin().await?;
            let (snapshot_id, _schema_version) = insert_snapshot(&mut tx).await?;
            let data_file_ids = positional
                .iter()
                .map(|entry| entry.data_file_id)
                .collect::<Vec<_>>();
            detect_new_inlined_deletes(&mut tx, table_id, base_snapshot, &data_file_ids).await?;
            for entry in positional {
                apply_delete_entry(&mut tx, table_id, base_snapshot, snapshot_id, entry).await?;
            }
            apply_inlined_deletes(&mut tx, table_id, snapshot_id, base_snapshot, inlined).await?;
            let deleted = i64::try_from(inlined.len()).map_err(|_| {
                crate::DuckLakeError::InvalidConfig(
                    "commit_deletes row count exceeds i64".to_string(),
                )
            })?;
            sqlx::query(
                "UPDATE ducklake_table_stats
                 SET record_count = GREATEST(record_count - ?, 0) WHERE table_id = ?",
            )
            .bind(deleted)
            .bind(table_id)
            .execute(&mut *tx)
            .await?;
            record_snapshot_changes(
                &mut tx,
                snapshot_id,
                &format!("deleted_from_table:{table_id}"),
                &SnapshotCommitMetadata::default(),
            )
            .await?;
            let schema_id: i64 =
                sqlx::query_scalar("SELECT schema_id FROM ducklake_table WHERE table_id = ?")
                    .bind(table_id)
                    .fetch_one(&mut *tx)
                    .await?;
            tx.commit().await?;
            Ok(CommitIds {
                snapshot_id,
                schema_id,
                table_id,
            })
        })
    }

    fn commit_compaction(
        &self,
        table_id: i64,
        base_snapshot: i64,
        sources: &[CompactionSourceFile],
        outputs: &[CompactionOutputFile],
        retirement: SourceRetirement,
    ) -> Result<CommitIds> {
        if sources.is_empty() {
            return Err(crate::DuckLakeError::InvalidConfig(
                "commit_compaction requires at least one source file".to_string(),
            ));
        }
        block_on(async {
            let mut tx = self.pool.begin().await?;
            let (snapshot_id, _schema_version) = insert_snapshot(&mut tx).await?;
            let source_ids = sources
                .iter()
                .map(|source| source.data_file_id)
                .collect::<Vec<_>>();
            detect_new_inlined_deletes(&mut tx, table_id, base_snapshot, &source_ids).await?;
            for source in sources {
                let live: Option<i64> = sqlx::query_scalar(
                    "SELECT 1 FROM ducklake_data_file
                     WHERE data_file_id = ? AND table_id = ? AND end_snapshot IS NULL",
                )
                .bind(source.data_file_id)
                .bind(table_id)
                .fetch_optional(&mut *tx)
                .await?;
                if live.is_none() {
                    return Err(crate::DuckLakeError::Conflict(format!(
                        "compaction of table {table_id} could not commit: source data file {} is \
                         no longer live since snapshot {base_snapshot}; re-open the catalog and \
                         re-plan",
                        source.data_file_id
                    )));
                }
                let current_delete: Option<i64> = sqlx::query_scalar(
                    "SELECT delete_file_id FROM ducklake_delete_file
                     WHERE data_file_id = ? AND end_snapshot IS NULL",
                )
                .bind(source.data_file_id)
                .fetch_optional(&mut *tx)
                .await?;
                if current_delete != source.delete_file_id {
                    return Err(crate::DuckLakeError::Conflict(format!(
                        "compaction of table {table_id} could not commit: the live delete file \
                         of source data file {} changed from {:?} to {current_delete:?} since \
                         snapshot {base_snapshot}; re-open the catalog and re-plan",
                        source.data_file_id, source.delete_file_id
                    )));
                }
            }
            let source_ids = sources
                .iter()
                .map(|source| source.data_file_id)
                .collect::<Vec<_>>();
            let source_ids = id_list(&source_ids);
            match retirement {
                SourceRetirement::Remove => {
                    for query in [
                        format!(
                            "INSERT INTO ducklake_files_scheduled_for_deletion
                                 (data_file_id, path, path_is_relative)
                             SELECT df.data_file_id, {RESOLVED_PATH}, {REL_FLAG}
                             FROM ducklake_data_file df
                             JOIN ducklake_table t ON t.table_id = df.table_id
                             JOIN ducklake_schema s ON s.schema_id = t.schema_id
                             WHERE df.data_file_id IN ({source_ids})"
                        ),
                        format!(
                            "INSERT INTO ducklake_files_scheduled_for_deletion
                                 (data_file_id, path, path_is_relative)
                             SELECT df.delete_file_id, {RESOLVED_PATH}, {REL_FLAG}
                             FROM ducklake_delete_file df
                             JOIN ducklake_table t ON t.table_id = df.table_id
                             JOIN ducklake_schema s ON s.schema_id = t.schema_id
                             WHERE df.data_file_id IN ({source_ids})"
                        ),
                    ] {
                        sqlx::query(AssertSqlSafe(query)).execute(&mut *tx).await?;
                    }
                    for table in [
                        "ducklake_delete_file",
                        "ducklake_data_file",
                        "ducklake_file_column_stats",
                        "ducklake_file_partition_value",
                    ] {
                        sqlx::query(AssertSqlSafe(format!(
                            "DELETE FROM {table} WHERE data_file_id IN ({source_ids})"
                        )))
                        .execute(&mut *tx)
                        .await?;
                    }
                },
                SourceRetirement::Retire => {
                    for table in ["ducklake_data_file", "ducklake_delete_file"] {
                        sqlx::query(AssertSqlSafe(format!(
                            "UPDATE {table} SET end_snapshot = ?
                             WHERE data_file_id IN ({source_ids}) AND end_snapshot IS NULL"
                        )))
                        .bind(snapshot_id)
                        .execute(&mut *tx)
                        .await?;
                    }
                },
            }
            let output_count = i64::try_from(outputs.len()).map_err(|_| {
                crate::DuckLakeError::InvalidConfig(
                    "commit_compaction output count exceeds i64".to_string(),
                )
            })?;
            let output_ids = reserve_file_ids(&mut tx, output_count).await?;
            for (output, data_file_id) in outputs.iter().zip(output_ids) {
                let begin_snapshot = output.begin_snapshot.unwrap_or(snapshot_id);
                sqlx::query(
                    "INSERT INTO ducklake_data_file
                         (data_file_id, table_id, path, path_is_relative, file_size_bytes,
                          footer_size, record_count, row_id_start, begin_snapshot, partial_max)
                     VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)",
                )
                .bind(data_file_id)
                .bind(table_id)
                .bind(&output.file.path)
                .bind(output.file.path_is_relative)
                .bind(output.file.file_size_bytes)
                .bind(output.file.footer_size)
                .bind(output.file.record_count)
                .bind(begin_snapshot)
                .bind(output.partial_max)
                .execute(&mut *tx)
                .await?;
                insert_file_column_stats(
                    &mut tx,
                    table_id,
                    data_file_id,
                    &output.file.column_stats,
                )
                .await?;
                insert_partition_metadata(&mut tx, table_id, data_file_id, &output.file).await?;
            }
            sqlx::query(
                "INSERT IGNORE INTO ducklake_table_stats
                     (table_id, record_count, next_row_id, file_size_bytes)
                 VALUES (?, 0, 0, 0)",
            )
            .bind(table_id)
            .execute(&mut *tx)
            .await?;
            sqlx::query(
                "UPDATE ducklake_table_stats SET
                     record_count = (SELECT COALESCE(SUM(record_count), 0)
                                     FROM ducklake_data_file
                                     WHERE table_id = ? AND end_snapshot IS NULL),
                     file_size_bytes = (SELECT COALESCE(SUM(file_size_bytes), 0)
                                        FROM ducklake_data_file
                                        WHERE table_id = ? AND end_snapshot IS NULL)
                 WHERE table_id = ?",
            )
            .bind(table_id)
            .bind(table_id)
            .bind(table_id)
            .execute(&mut *tx)
            .await?;
            record_snapshot_changes(
                &mut tx,
                snapshot_id,
                &format!("compacted_table:{table_id}"),
                &SnapshotCommitMetadata::new().with_message("datafusion compaction"),
            )
            .await?;
            let schema_id: i64 =
                sqlx::query_scalar("SELECT schema_id FROM ducklake_table WHERE table_id = ?")
                    .bind(table_id)
                    .fetch_one(&mut *tx)
                    .await?;
            tx.commit().await?;
            Ok(CommitIds {
                snapshot_id,
                schema_id,
                table_id,
            })
        })
    }

    fn retire_appends_since(&self, table_id: i64, base_snapshot: i64) -> Result<Option<i64>> {
        block_on(async {
            let mut tx = self.pool.begin().await?;
            let (snapshot_id, _schema_version) = insert_snapshot(&mut tx).await?;
            let impure_delete: Option<i64> = sqlx::query_scalar(
                "SELECT 1 FROM ducklake_delete_file
                 WHERE table_id = ? AND begin_snapshot > ? LIMIT 1",
            )
            .bind(table_id)
            .bind(base_snapshot)
            .fetch_optional(&mut *tx)
            .await?;
            let impure_ended: Option<i64> = sqlx::query_scalar(
                "SELECT 1 FROM ducklake_data_file
                 WHERE table_id = ? AND begin_snapshot <= ?
                   AND end_snapshot IS NOT NULL AND end_snapshot > ? LIMIT 1",
            )
            .bind(table_id)
            .bind(base_snapshot)
            .bind(base_snapshot)
            .fetch_optional(&mut *tx)
            .await?;
            let impure_column: Option<i64> = sqlx::query_scalar(
                "SELECT 1 FROM ducklake_column
                 WHERE table_id = ? AND (begin_snapshot > ? OR end_snapshot > ?) LIMIT 1",
            )
            .bind(table_id)
            .bind(base_snapshot)
            .bind(base_snapshot)
            .fetch_optional(&mut *tx)
            .await?;
            let impure_partition: Option<i64> = sqlx::query_scalar(
                "SELECT 1 FROM ducklake_partition_info
                 WHERE table_id = ? AND (begin_snapshot > ? OR end_snapshot > ?) LIMIT 1",
            )
            .bind(table_id)
            .bind(base_snapshot)
            .bind(base_snapshot)
            .fetch_optional(&mut *tx)
            .await?;
            if impure_delete.is_some()
                || impure_ended.is_some()
                || impure_column.is_some()
                || impure_partition.is_some()
            {
                return Err(crate::DuckLakeError::Conflict(format!(
                    "table {table_id}: changes since snapshot {base_snapshot} are not a pure \
                     append (delete/replace/update or schema/partition change present); refusing \
                     to retire"
                )));
            }
            let has_appended: Option<i64> = sqlx::query_scalar(
                "SELECT 1 FROM ducklake_data_file
                 WHERE table_id = ? AND end_snapshot IS NULL AND begin_snapshot > ? LIMIT 1",
            )
            .bind(table_id)
            .bind(base_snapshot)
            .fetch_optional(&mut *tx)
            .await?;
            if has_appended.is_none() {
                return Ok(None);
            }
            sqlx::query(
                "UPDATE ducklake_data_file SET end_snapshot = ?
                 WHERE table_id = ? AND end_snapshot IS NULL AND begin_snapshot > ?",
            )
            .bind(snapshot_id)
            .bind(table_id)
            .bind(base_snapshot)
            .execute(&mut *tx)
            .await?;
            sqlx::query(
                "UPDATE ducklake_table_stats SET
                     record_count = (SELECT COALESCE(SUM(record_count), 0)
                                     FROM ducklake_data_file
                                     WHERE table_id = ? AND end_snapshot IS NULL),
                     file_size_bytes = (SELECT COALESCE(SUM(file_size_bytes), 0)
                                        FROM ducklake_data_file
                                        WHERE table_id = ? AND end_snapshot IS NULL)
                 WHERE table_id = ?",
            )
            .bind(table_id)
            .bind(table_id)
            .bind(table_id)
            .execute(&mut *tx)
            .await?;
            let (columns, column_ids) = live_columns_for_stats(&mut tx, table_id).await?;
            recompute_table_column_stats(&mut tx, table_id, &columns, &column_ids).await?;
            record_snapshot_changes(
                &mut tx,
                snapshot_id,
                &format!("deleted_from_table:{table_id}"),
                &SnapshotCommitMetadata::default(),
            )
            .await?;
            tx.commit().await?;
            Ok(Some(snapshot_id))
        })
    }

    fn commit_truncate(
        &self,
        table_id: i64,
        _schema_name: &str,
        _table_name: &str,
        _base_snapshot: i64,
    ) -> Result<u64> {
        block_on(async {
            let mut tx = self.pool.begin().await?;
            let (snapshot_id, _schema_version) = insert_snapshot(&mut tx).await?;
            let has_live_data: Option<i64> = sqlx::query_scalar(
                "SELECT 1 FROM ducklake_data_file
                 WHERE table_id = ? AND end_snapshot IS NULL LIMIT 1",
            )
            .bind(table_id)
            .fetch_optional(&mut *tx)
            .await?;
            let live_inlined = live_inlined_row_count(&mut tx, table_id).await?;
            if has_live_data.is_none() && live_inlined == 0 {
                return Ok(0);
            }
            let gross: Option<i64> = sqlx::query_scalar(
                "SELECT record_count FROM ducklake_table_stats WHERE table_id = ?",
            )
            .bind(table_id)
            .fetch_optional(&mut *tx)
            .await?;
            let deleted: i64 = sqlx::query_scalar(
                "SELECT CAST(COALESCE(SUM(delete_count), 0) AS SIGNED)
                 FROM ducklake_delete_file
                 WHERE table_id = ? AND end_snapshot IS NULL",
            )
            .bind(table_id)
            .fetch_one(&mut *tx)
            .await?;
            let live_rows = (gross.unwrap_or(0) - deleted).max(0) as u64;
            sqlx::query(
                "UPDATE ducklake_data_file SET end_snapshot = ?
                 WHERE table_id = ? AND end_snapshot IS NULL",
            )
            .bind(snapshot_id)
            .bind(table_id)
            .execute(&mut *tx)
            .await?;
            for table_name in inlined_table_names(&mut tx, table_id).await? {
                let sql = format!(
                    "UPDATE {} SET end_snapshot = ? WHERE end_snapshot IS NULL",
                    quote_ident(&table_name)
                );
                sqlx::query(AssertSqlSafe(sql))
                    .bind(snapshot_id)
                    .execute(&mut *tx)
                    .await?;
            }
            sqlx::query(
                "UPDATE ducklake_delete_file SET end_snapshot = ?
                 WHERE table_id = ? AND end_snapshot IS NULL",
            )
            .bind(snapshot_id)
            .bind(table_id)
            .execute(&mut *tx)
            .await?;
            sqlx::query(
                "UPDATE ducklake_table_stats SET record_count = 0, file_size_bytes = 0
                 WHERE table_id = ?",
            )
            .bind(table_id)
            .execute(&mut *tx)
            .await?;
            record_snapshot_changes(
                &mut tx,
                snapshot_id,
                &format!("deleted_from_table:{table_id}"),
                &SnapshotCommitMetadata::default(),
            )
            .await?;
            tx.commit().await?;
            Ok(live_rows)
        })
    }

    fn register_inlined_data(
        &self,
        table_id: i64,
        schema_name: &str,
        table_name: &str,
        _snapshot_id: i64,
        batches: &[RecordBatch],
        mode: WriteMode,
        base_snapshot: i64,
        columns: &[ColumnDef],
        column_ids: &[i64],
        commit_metadata: &SnapshotCommitMetadata,
        expected_base_snapshot_id: Option<i64>,
    ) -> Result<CommitIds> {
        let record_count: usize = batches.iter().map(RecordBatch::num_rows).sum();
        if record_count == 0 {
            return Err(crate::DuckLakeError::InvalidConfig(
                "register_inlined_data: batches must contain at least one row".to_string(),
            ));
        }
        if batches
            .iter()
            .any(|batch| batch.num_columns() != columns.len())
        {
            return Err(crate::DuckLakeError::InvalidConfig(
                "register_inlined_data: batch schema does not match table columns".to_string(),
            ));
        }

        block_on(async {
            // MySQL DDL implicitly commits the surrounding transaction, so the
            // physical inline table must exist BEFORE the write transaction
            // opens (CREATE TABLE IF NOT EXISTS is idempotent, and an existing
            // physical table without registry rows is inert). Its name embeds
            // the schema version the commit will allocate, which is only known
            // inside the transaction — predict it, and when the commit resolves
            // a different version (a schema change in this write), roll back,
            // create the table for the observed version, and retry once.
            let mut create_for: Option<i64> = None;
            let mut settled = None;
            for _ in 0..3 {
                let version_to_create = match create_for {
                    Some(version) => version,
                    None => {
                        sqlx::query_scalar(
                            "SELECT COALESCE(MAX(schema_version), 0) FROM ducklake_snapshot",
                        )
                        .fetch_one(&self.pool)
                        .await?
                    },
                };
                let physical_name = format!("ducklake_inlined_data_{table_id}_{version_to_create}");
                let ddl =
                    inlined_mysql_table_ddl(&physical_name, batches[0].schema().fields(), columns);
                sqlx::query(AssertSqlSafe(ddl)).execute(&self.pool).await?;
                ensure_mysql_physical_indexes(&self.pool, table_id, &physical_name).await?;

                let mut tx = self.pool.begin().await?;
                let snapshot_id =
                    finalize_snapshot(&mut tx, table_id, columns, column_ids, mode, base_snapshot)
                        .await?;
                if mode != WriteMode::Replace
                    && let Some(expected_base_snapshot_id) = expected_base_snapshot_id
                {
                    detect_replace_conflict(&mut tx, table_id, expected_base_snapshot_id).await?;
                }
                let schema_version: i64 = sqlx::query_scalar(
                    "SELECT schema_version FROM ducklake_snapshot WHERE snapshot_id = ?",
                )
                .bind(snapshot_id)
                .fetch_one(&mut *tx)
                .await?;
                if schema_version != version_to_create {
                    tx.rollback().await?;
                    create_for = Some(schema_version);
                    continue;
                }
                settled = Some((tx, snapshot_id, schema_version, physical_name));
                break;
            }
            let Some((mut tx, snapshot_id, schema_version, physical_name)) = settled else {
                return Err(crate::DuckLakeError::Conflict(format!(
                    "register_inlined_data on table {table_id} could not settle on a schema \
                     version for the inline table; retry the write"
                )));
            };
            sqlx::query(
                "INSERT INTO ducklake_inlined_data_tables
                     (table_id, table_name, schema_version)
                 SELECT ?, ?, ? FROM DUAL
                 WHERE NOT EXISTS (
                     SELECT 1 FROM ducklake_inlined_data_tables
                     WHERE table_id = ? AND schema_version = ?)",
            )
            .bind(table_id)
            .bind(&physical_name)
            .bind(schema_version)
            .bind(table_id)
            .bind(schema_version)
            .execute(&mut *tx)
            .await?;
            sqlx::query(
                "INSERT IGNORE INTO ducklake_table_stats
                     (table_id, record_count, next_row_id, file_size_bytes)
                 VALUES (?, 0, 0, 0)",
            )
            .bind(table_id)
            .execute(&mut *tx)
            .await?;
            let mut row_id: i64 = sqlx::query_scalar(
                "SELECT next_row_id FROM ducklake_table_stats WHERE table_id = ?",
            )
            .bind(table_id)
            .fetch_one(&mut *tx)
            .await?;

            let column_list = columns
                .iter()
                .map(|column| quote_ident(column.name()))
                .collect::<Vec<_>>()
                .join(", ");
            for batch in batches {
                for batch_row in 0..batch.num_rows() {
                    let mut query = QueryBuilder::<MySql>::new(format!(
                        "INSERT INTO {} (row_id, begin_snapshot, end_snapshot, {}) VALUES (",
                        quote_ident(&physical_name),
                        column_list
                    ));
                    query.push_bind(row_id);
                    query.push(", ").push_bind(snapshot_id);
                    query.push(", NULL");
                    for array in batch.columns() {
                        query.push(", ");
                        push_inlined_mysql_value(
                            &mut query,
                            array.as_ref(),
                            batch_row,
                            &inlined_mysql_type(array.data_type()),
                        )?;
                    }
                    query.push(')');
                    query.build().execute(&mut *tx).await?;
                    row_id += 1;
                }
            }

            let record_count = i64::try_from(record_count).map_err(|_| {
                crate::DuckLakeError::InvalidConfig(
                    "register_inlined_data: record count exceeds i64".to_string(),
                )
            })?;
            sqlx::query(
                "UPDATE ducklake_table_stats
                 SET next_row_id = next_row_id + ?, record_count = record_count + ?
                 WHERE table_id = ?",
            )
            .bind(record_count)
            .bind(record_count)
            .bind(table_id)
            .execute(&mut *tx)
            .await?;
            // The same composed ledger recording the Parquet path uses: DDL
            // entries plus the write change, appended rather than clobbered.
            record_table_write_changes(
                &mut tx,
                snapshot_id,
                table_id,
                schema_name,
                table_name,
                mode,
                false,
                commit_metadata,
            )
            .await?;
            let schema_id: i64 =
                sqlx::query_scalar("SELECT schema_id FROM ducklake_table WHERE table_id = ?")
                    .bind(table_id)
                    .fetch_one(&mut *tx)
                    .await?;
            tx.commit().await?;
            Ok(CommitIds {
                snapshot_id,
                schema_id,
                table_id,
            })
        })
    }

    fn commit_multi_table(
        &self,
        writes: &[StagedTableWrite],
        commit_metadata: &SnapshotCommitMetadata,
        expected_base_snapshot_id: Option<i64>,
    ) -> Result<MultiTableCommit> {
        if writes.is_empty() {
            return Err(crate::DuckLakeError::InvalidConfig(
                "commit_multi_table requires at least one table stage".to_string(),
            ));
        }
        let table_ids = writes
            .iter()
            .map(|write| write.table_id)
            .collect::<std::collections::HashSet<_>>();
        if table_ids.len() != writes.len() {
            return Err(crate::DuckLakeError::InvalidConfig(
                "commit_multi_table requires one stage per table".to_string(),
            ));
        }

        block_on(async {
            // MySQL DDL implicitly commits, so physical inline tables must exist
            // before the write transaction opens. Predict the schema version the
            // commit will resolve; commit_staged_inline aborts with Conflict if
            // it moved, and the caller's retry recreates for the new version.
            let predicted_version: i64 = sqlx::query_scalar(
                "SELECT COALESCE(MAX(schema_version), 0) FROM ducklake_snapshot",
            )
            .fetch_one(&self.pool)
            .await?;
            for write in writes {
                if let StagedTableData::Inlined(batches) = &write.data {
                    if batches.is_empty() {
                        continue;
                    }
                    let physical_name = format!(
                        "ducklake_inlined_data_{}_{predicted_version}",
                        write.table_id
                    );
                    let ddl = inlined_mysql_table_ddl(
                        &physical_name,
                        batches[0].schema().fields(),
                        &write.columns,
                    );
                    sqlx::query(AssertSqlSafe(ddl)).execute(&self.pool).await?;
                    ensure_mysql_physical_indexes(&self.pool, write.table_id, &physical_name)
                        .await?;
                }
            }

            let mut tx = self.pool.begin().await?;
            // Allocate the snapshot FIRST: its counter update takes the
            // serializing row lock before any plain SELECT establishes this
            // transaction's InnoDB read view, so every fence and stats read
            // below sees all commits serialized before ours. Reading first
            // would freeze a pre-lock view and silently bypass the fences.
            let (snapshot_id, _schema_version) = insert_snapshot(&mut tx).await?;
            if let Some(expected) = expected_base_snapshot_id {
                for write in writes {
                    detect_replace_conflict(&mut tx, write.table_id, expected).await?;
                }
            }
            let mut had_live_data = Vec::with_capacity(writes.len());
            for write in writes {
                had_live_data.push(has_live_data(&mut tx, write.table_id).await?);
            }
            let mut tables = Vec::with_capacity(writes.len());
            for write in writes {
                let schema_id = validate_staged_table(&mut tx, write).await?;
                if write.mode == WriteMode::Replace {
                    detect_replace_conflict(&mut tx, write.table_id, write.base_snapshot_id)
                        .await?;
                    retire_prior_generation(&mut tx, write.table_id, snapshot_id).await?;
                }
                tables.push(CommitIds {
                    snapshot_id,
                    schema_id,
                    table_id: write.table_id,
                });
            }
            for (write, replaced_existing_data) in writes.iter().zip(had_live_data) {
                match &write.data {
                    StagedTableData::Files(files) => {
                        commit_staged_files(&mut tx, snapshot_id, write, files).await?;
                    },
                    StagedTableData::Inlined(batches) => {
                        commit_staged_inline(&mut tx, snapshot_id, write, batches).await?;
                    },
                    StagedTableData::None => {},
                }
                for entry in &write.positional_deletes {
                    apply_delete_entry(
                        &mut tx,
                        write.table_id,
                        write.base_snapshot_id,
                        snapshot_id,
                        entry,
                    )
                    .await?;
                }
                apply_staged_inlined_deletes(&mut tx, snapshot_id, write).await?;
                let has_deletes =
                    !write.positional_deletes.is_empty() || !write.inlined_deletes.is_empty();
                let changes_made = if matches!(&write.data, StagedTableData::None) {
                    format!("deleted_from_table:{}", write.table_id)
                } else {
                    table_write_changes(
                        write.table_id,
                        write.mode,
                        has_deletes,
                        replaced_existing_data,
                    )
                };
                record_snapshot_changes(&mut tx, snapshot_id, &changes_made, commit_metadata)
                    .await?;
            }
            tx.commit().await?;
            Ok(MultiTableCommit {
                snapshot_id,
                tables,
            })
        })
    }

    fn commit_inlined_deletes(
        &self,
        table_id: i64,
        _schema_name: &str,
        _table_name: &str,
        base_snapshot: i64,
        rows: &[InlinedRowRef],
    ) -> Result<CommitIds> {
        if rows.is_empty() {
            return Err(crate::DuckLakeError::InvalidConfig(
                "commit_inlined_deletes requires at least one row".to_string(),
            ));
        }
        if rows.iter().collect::<std::collections::HashSet<_>>().len() != rows.len() {
            return Err(crate::DuckLakeError::InvalidConfig(
                "commit_inlined_deletes contains duplicate rows".to_string(),
            ));
        }
        block_on(async {
            let mut tx = self.pool.begin().await?;
            let (snapshot_id, _schema_version) = insert_snapshot(&mut tx).await?;
            let registered = sqlx::query(
                "SELECT table_name FROM ducklake_inlined_data_tables WHERE table_id = ?",
            )
            .bind(table_id)
            .fetch_all(&mut *tx)
            .await?
            .into_iter()
            .map(|row| row.try_get(0))
            .collect::<std::result::Result<std::collections::HashSet<String>, _>>()?;
            for row in rows {
                if !registered.contains(&row.table_name) {
                    return Err(crate::DuckLakeError::Conflict(format!(
                        "inlined row {} belongs to an unregistered table '{}'",
                        row.row_id, row.table_name
                    )));
                }
                let sql = format!(
                    "UPDATE {} SET end_snapshot = ? \
                     WHERE row_id = ? AND begin_snapshot <= ? AND end_snapshot IS NULL",
                    quote_ident(&row.table_name)
                );
                let affected = sqlx::query(AssertSqlSafe(sql))
                    .bind(snapshot_id)
                    .bind(row.row_id)
                    .bind(base_snapshot)
                    .execute(&mut *tx)
                    .await?
                    .rows_affected();
                if affected != 1 {
                    return Err(crate::DuckLakeError::Conflict(format!(
                        "inlined row {} in '{}' is no longer live at snapshot {base_snapshot}",
                        row.row_id, row.table_name
                    )));
                }
            }
            let deleted = i64::try_from(rows.len()).map_err(|_| {
                crate::DuckLakeError::InvalidConfig(
                    "commit_inlined_deletes row count exceeds i64".to_string(),
                )
            })?;
            sqlx::query(
                "UPDATE ducklake_table_stats
                 SET record_count = GREATEST(record_count - ?, 0) WHERE table_id = ?",
            )
            .bind(deleted)
            .bind(table_id)
            .execute(&mut *tx)
            .await?;
            let changes_made = format!("deleted_from_table:{table_id}");
            record_snapshot_changes(
                &mut tx,
                snapshot_id,
                &changes_made,
                &SnapshotCommitMetadata::default(),
            )
            .await?;
            let schema_id: i64 =
                sqlx::query_scalar("SELECT schema_id FROM ducklake_table WHERE table_id = ?")
                    .bind(table_id)
                    .fetch_one(&mut *tx)
                    .await?;
            tx.commit().await?;
            Ok(CommitIds {
                snapshot_id,
                schema_id,
                table_id,
            })
        })
    }

    fn set_partition_spec(
        &self,
        table_id: i64,
        columns: &[(String, PartitionTransform)],
    ) -> Result<i64> {
        if columns.is_empty() {
            return Err(crate::DuckLakeError::InvalidConfig(
                "set_partition_spec: partition spec must have at least one column; \
                 use reset_partition_spec to remove partitioning"
                    .to_string(),
            ));
        }
        block_on(async {
            let mut tx = self.pool.begin().await?;
            let partition_id = reserve_ids(&mut tx, "next_partition_id", 1).await?;
            let (new_snapshot, _carried) = insert_snapshot(&mut tx).await?;

            let mut column_ids: Vec<i64> = Vec::with_capacity(columns.len());
            for (name, _transform) in columns {
                let column_id: i64 = sqlx::query_scalar(
                    "SELECT column_id FROM ducklake_column
                     WHERE table_id = ? AND column_name = ? AND end_snapshot IS NULL
                       AND parent_column IS NULL",
                )
                .bind(table_id)
                .bind(name)
                .fetch_optional(&mut *tx)
                .await?
                .ok_or_else(|| {
                    crate::DuckLakeError::InvalidConfig(format!(
                        "set_partition_spec: no live column '{name}' in table {table_id}"
                    ))
                })?;
                column_ids.push(column_id);
            }

            sqlx::query(
                "UPDATE ducklake_partition_info SET end_snapshot = ?
                 WHERE table_id = ? AND end_snapshot IS NULL",
            )
            .bind(new_snapshot)
            .bind(table_id)
            .execute(&mut *tx)
            .await?;
            sqlx::query(
                "INSERT INTO ducklake_partition_info
                     (partition_id, table_id, begin_snapshot, end_snapshot)
                 VALUES (?, ?, ?, NULL)",
            )
            .bind(partition_id)
            .bind(table_id)
            .bind(new_snapshot)
            .execute(&mut *tx)
            .await?;
            for (key_index, column_id) in column_ids.iter().enumerate() {
                sqlx::query(
                    "INSERT INTO ducklake_partition_column
                         (partition_id, table_id, partition_key_index, column_id, transform)
                     VALUES (?, ?, ?, ?, ?)",
                )
                .bind(partition_id)
                .bind(table_id)
                .bind(key_index as i64)
                .bind(*column_id)
                .bind(columns[key_index].1.to_catalog_string())
                .execute(&mut *tx)
                .await?;
            }

            let new_schema_version = bump_schema_version(&mut tx, new_snapshot).await?;
            record_schema_version(&mut tx, new_snapshot, new_schema_version, table_id).await?;
            record_snapshot_changes(
                &mut tx,
                new_snapshot,
                &format!("altered_table:{table_id}"),
                &SnapshotCommitMetadata::default(),
            )
            .await?;
            tx.commit().await?;
            Ok(new_snapshot)
        })
    }

    fn live_partition_spec(
        &self,
        table_id: i64,
    ) -> Result<Option<crate::partition::PartitionSpec>> {
        block_on(async {
            let rows = sqlx::query(
                "SELECT pi.partition_id, pc.partition_key_index, pc.column_id, pc.transform
                 FROM ducklake_partition_info AS pi
                 JOIN ducklake_partition_column AS pc
                   ON pc.partition_id = pi.partition_id AND pc.table_id = pi.table_id
                 WHERE pi.table_id = ? AND pi.end_snapshot IS NULL
                 ORDER BY pc.partition_key_index",
            )
            .bind(table_id)
            .fetch_all(&self.pool)
            .await?;
            let parsed = rows
                .iter()
                .map(|row| {
                    Ok::<_, crate::DuckLakeError>((
                        row.try_get::<i64, _>(0)?,
                        i32::try_from(row.try_get::<i64, _>(1)?).unwrap_or(0),
                        row.try_get::<i64, _>(2)?,
                        row.try_get::<String, _>(3)?,
                    ))
                })
                .collect::<Result<Vec<_>>>()?;
            // prune_safe = false: this spec is for laying out a write, never pruning.
            Ok(crate::partition::PartitionSpec::from_rows(parsed, false))
        })
    }

    fn reset_partition_spec(&self, table_id: i64) -> Result<i64> {
        block_on(async {
            let mut tx = self.pool.begin().await?;
            let (new_snapshot, _carried) = insert_snapshot(&mut tx).await?;
            let ended = sqlx::query(
                "UPDATE ducklake_partition_info SET end_snapshot = ?
                 WHERE table_id = ? AND end_snapshot IS NULL",
            )
            .bind(new_snapshot)
            .bind(table_id)
            .execute(&mut *tx)
            .await?
            .rows_affected();
            if ended == 0 {
                drop(tx);
                let head: i64 = sqlx::query_scalar(
                    "SELECT COALESCE(MAX(snapshot_id), 0) FROM ducklake_snapshot",
                )
                .fetch_one(&self.pool)
                .await?;
                return Ok(head);
            }
            let new_schema_version = bump_schema_version(&mut tx, new_snapshot).await?;
            record_schema_version(&mut tx, new_snapshot, new_schema_version, table_id).await?;
            record_snapshot_changes(
                &mut tx,
                new_snapshot,
                &format!("altered_table:{table_id}"),
                &SnapshotCommitMetadata::default(),
            )
            .await?;
            tx.commit().await?;
            Ok(new_snapshot)
        })
    }

    fn live_sort_spec(&self, table_id: i64) -> Result<Option<crate::sort::SortSpec>> {
        block_on(async {
            let rows = sqlx::query(
                "SELECT si.sort_id, se.sort_key_index, se.expression, se.dialect,
                        se.sort_direction, se.null_order
                 FROM ducklake_sort_info AS si
                 JOIN ducklake_sort_expression AS se
                   ON se.sort_id = si.sort_id AND se.table_id = si.table_id
                 WHERE si.table_id = ? AND si.end_snapshot IS NULL
                 ORDER BY se.sort_key_index",
            )
            .bind(table_id)
            .fetch_all(&self.pool)
            .await?;
            let parsed = rows
                .iter()
                .map(|row| {
                    Ok::<_, crate::DuckLakeError>((
                        row.try_get::<i64, _>(0)?,
                        i32::try_from(row.try_get::<i64, _>(1)?).unwrap_or(0),
                        row.try_get::<String, _>(2)?,
                        row.try_get::<String, _>(3)?,
                        row.try_get::<String, _>(4)?,
                        row.try_get::<String, _>(5)?,
                    ))
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(crate::sort::SortSpec::from_rows(parsed))
        })
    }

    fn set_sort_spec(&self, table_id: i64, fields: &[crate::sort::SortField]) -> Result<i64> {
        if fields.is_empty() {
            return Err(crate::DuckLakeError::InvalidConfig(
                "set_sort_spec: at least one sort key is required (use reset_sort_spec to clear)"
                    .to_string(),
            ));
        }
        block_on(async {
            let mut tx = self.pool.begin().await?;
            let sort_id = reserve_ids(&mut tx, "next_sort_id", 1).await?;
            let (new_snapshot, _carried) = insert_snapshot(&mut tx).await?;

            // Validate every sort key resolves to a live column (v1 supports bare
            // column keys only), so a bad SET fails here rather than silently
            // producing unsorted writes later.
            for field in fields {
                let column = field.column_candidate().ok_or_else(|| {
                    crate::DuckLakeError::InvalidConfig(format!(
                        "set_sort_spec: sort key '{}' is not a bare column; only column \
                         sort keys are supported",
                        field.expression
                    ))
                })?;
                let exists: Option<i64> = sqlx::query_scalar(
                    "SELECT column_id FROM ducklake_column
                     WHERE table_id = ? AND column_name = ? AND end_snapshot IS NULL
                       AND parent_column IS NULL",
                )
                .bind(table_id)
                .bind(&column)
                .fetch_optional(&mut *tx)
                .await?;
                if exists.is_none() {
                    return Err(crate::DuckLakeError::InvalidConfig(format!(
                        "set_sort_spec: no live column '{column}' in table {table_id}"
                    )));
                }
            }

            sqlx::query(
                "UPDATE ducklake_sort_info SET end_snapshot = ?
                 WHERE table_id = ? AND end_snapshot IS NULL",
            )
            .bind(new_snapshot)
            .bind(table_id)
            .execute(&mut *tx)
            .await?;
            sqlx::query(
                "INSERT INTO ducklake_sort_info
                     (sort_id, table_id, begin_snapshot, end_snapshot)
                 VALUES (?, ?, ?, NULL)",
            )
            .bind(sort_id)
            .bind(table_id)
            .bind(new_snapshot)
            .execute(&mut *tx)
            .await?;
            for field in fields {
                sqlx::query(
                    "INSERT INTO ducklake_sort_expression
                         (sort_id, table_id, sort_key_index, expression, dialect,
                          sort_direction, null_order)
                     VALUES (?, ?, ?, ?, ?, ?, ?)",
                )
                .bind(sort_id)
                .bind(table_id)
                .bind(field.sort_key_index as i64)
                .bind(&field.expression)
                .bind(&field.dialect)
                .bind(field.direction.to_catalog_string())
                .bind(field.null_order.to_catalog_string())
                .execute(&mut *tx)
                .await?;
            }

            // A sort-order change does NOT bump schema_version.
            record_snapshot_changes(
                &mut tx,
                new_snapshot,
                &format!("altered_table:{table_id}"),
                &SnapshotCommitMetadata::default(),
            )
            .await?;
            tx.commit().await?;
            Ok(new_snapshot)
        })
    }

    fn reset_sort_spec(&self, table_id: i64) -> Result<i64> {
        block_on(async {
            let mut tx = self.pool.begin().await?;
            let (new_snapshot, _carried) = insert_snapshot(&mut tx).await?;
            let ended = sqlx::query(
                "UPDATE ducklake_sort_info SET end_snapshot = ?
                 WHERE table_id = ? AND end_snapshot IS NULL",
            )
            .bind(new_snapshot)
            .bind(table_id)
            .execute(&mut *tx)
            .await?
            .rows_affected();
            if ended == 0 {
                drop(tx);
                let head: i64 = sqlx::query_scalar(
                    "SELECT COALESCE(MAX(snapshot_id), 0) FROM ducklake_snapshot",
                )
                .fetch_one(&self.pool)
                .await?;
                return Ok(head);
            }
            // A sort-order change does NOT bump schema_version.
            record_snapshot_changes(
                &mut tx,
                new_snapshot,
                &format!("altered_table:{table_id}"),
                &SnapshotCommitMetadata::default(),
            )
            .await?;
            tx.commit().await?;
            Ok(new_snapshot)
        })
    }

    fn publish_snapshot(
        &self,
        table_id: i64,
        schema_name: &str,
        table_name: &str,
        _snapshot_id: i64,
        mode: WriteMode,
        base_snapshot: i64,
        columns: &[ColumnDef],
        column_ids: &[i64],
    ) -> Result<CommitIds> {
        // Fileless commit point (CREATE TABLE, zero-row Replace). Single-catalog
        // MySQL defers the snapshot-row insert out of begin_write_transaction, so
        // the trait's default no-op is insufficient: insert the deferred snapshot
        // row + column generation and, for Replace, retire the prior generation —
        // making the new head visible atomically.
        block_on(async {
            let mut tx = self.pool.begin().await?;
            let snapshot_id =
                finalize_snapshot(&mut tx, table_id, columns, column_ids, mode, base_snapshot)
                    .await?;
            record_table_write_changes(
                &mut tx,
                snapshot_id,
                table_id,
                schema_name,
                table_name,
                mode,
                false,
                &SnapshotCommitMetadata::default(),
            )
            .await?;
            let schema_id: i64 =
                sqlx::query("SELECT schema_id FROM ducklake_table WHERE table_id = ?")
                    .bind(table_id)
                    .fetch_one(&mut *tx)
                    .await?
                    .try_get(0)?;
            tx.commit().await?;
            Ok(CommitIds {
                snapshot_id,
                schema_id,
                table_id,
            })
        })
    }

    fn end_table_files(&self, table_id: i64, snapshot_id: i64) -> Result<u64> {
        // Used by WriteMode::Replace. End-snapshotting every visible file drops the
        // table's currently-visible row count and byte total to zero. `next_row_id`
        // is deliberately NOT reset: rowids must stay monotonic across the table's
        // lifetime so historical snapshots still resolve uniquely.
        block_on(async {
            let mut tx = self.pool.begin().await?;

            let result = sqlx::query(
                "UPDATE ducklake_data_file SET end_snapshot = ?
                 WHERE table_id = ? AND end_snapshot IS NULL",
            )
            .bind(snapshot_id)
            .bind(table_id)
            .execute(&mut *tx)
            .await?;

            sqlx::query(
                "UPDATE ducklake_table_stats
                 SET record_count = 0, file_size_bytes = 0
                 WHERE table_id = ?",
            )
            .bind(table_id)
            .execute(&mut *tx)
            .await?;

            tx.commit().await?;
            Ok(result.rows_affected())
        })
    }

    fn get_data_path(&self) -> Result<String> {
        block_on(async {
            let row = sqlx::query(
                "SELECT `value` FROM ducklake_metadata WHERE `key` = ? AND scope IS NULL",
            )
            .bind("data_path")
            .fetch_optional(&self.pool)
            .await?;

            match row {
                Some(r) => Ok(r.try_get(0)?),
                None => Err(crate::error::DuckLakeError::InvalidConfig(
                    "Missing required catalog metadata: 'data_path' not configured.".to_string(),
                )),
            }
        })
    }

    fn set_data_path(&self, path: &str) -> Result<()> {
        block_on(async {
            sqlx::query(
                "DELETE FROM ducklake_metadata WHERE `key` = 'data_path' AND scope IS NULL",
            )
            .execute(&self.pool)
            .await?;

            sqlx::query(
                "INSERT INTO ducklake_metadata (`key`, `value`, scope)
                 VALUES ('data_path', ?, NULL)",
            )
            .bind(path)
            .execute(&self.pool)
            .await?;

            Ok(())
        })
    }

    fn initialize_schema(&self) -> Result<()> {
        block_on(async {
            // sqlx runs each query() as a single prepared statement on MySQL, so
            // create each table separately (see SQL_CREATE_TABLES).
            for ddl in SQL_CREATE_TABLES {
                sqlx::query(*ddl).execute(&self.pool).await?;
            }
            let default_columns: Vec<String> = sqlx::query_scalar(
                "SELECT column_name FROM information_schema.columns
                 WHERE table_schema = DATABASE()
                   AND table_name = 'ducklake_column'
                   AND column_name IN ('initial_default', 'default_value',
                                       'default_value_type', 'default_value_dialect')",
            )
            .fetch_all(&self.pool)
            .await?;
            for (name, ddl) in [
                (
                    "initial_default",
                    "ALTER TABLE ducklake_column ADD COLUMN initial_default VARCHAR(1024)",
                ),
                (
                    "default_value",
                    "ALTER TABLE ducklake_column ADD COLUMN default_value VARCHAR(1024)",
                ),
                (
                    "default_value_type",
                    "ALTER TABLE ducklake_column ADD COLUMN default_value_type VARCHAR(1024)",
                ),
                (
                    "default_value_dialect",
                    "ALTER TABLE ducklake_column ADD COLUMN default_value_dialect VARCHAR(1024)",
                ),
            ] {
                if !default_columns.iter().any(|column| column == name) {
                    sqlx::query(ddl).execute(&self.pool).await?;
                }
            }
            let has_scope_id: i64 = sqlx::query_scalar(
                "SELECT COUNT(*) FROM information_schema.columns \
                 WHERE table_schema = DATABASE() \
                   AND table_name = 'ducklake_metadata' \
                   AND column_name = 'scope_id'",
            )
            .fetch_one(&self.pool)
            .await?;
            let has_scope: i64 = sqlx::query_scalar(
                "SELECT COUNT(*) FROM information_schema.columns \
                 WHERE table_schema = DATABASE() \
                   AND table_name = 'ducklake_metadata' \
                   AND column_name = 'scope'",
            )
            .fetch_one(&self.pool)
            .await?;
            if has_scope == 0 {
                sqlx::query("ALTER TABLE ducklake_metadata ADD COLUMN scope VARCHAR(255)")
                    .execute(&self.pool)
                    .await?;
            }
            if has_scope_id == 0 {
                sqlx::query("ALTER TABLE ducklake_metadata ADD COLUMN scope_id BIGINT")
                    .execute(&self.pool)
                    .await?;
            }
            // Upgrade a pre-existing catalog to carry ducklake_data_file.partition_id.
            // MySQL has no `ADD COLUMN IF NOT EXISTS`, so probe information_schema first
            // (idempotent, lossless — NULL means "not partitioned").
            let has_partition_id: i64 = sqlx::query_scalar(
                "SELECT COUNT(*) FROM information_schema.columns \
                 WHERE table_schema = DATABASE() \
                   AND table_name = 'ducklake_data_file' \
                   AND column_name = 'partition_id'",
            )
            .fetch_one(&self.pool)
            .await?;
            if has_partition_id == 0 {
                sqlx::query("ALTER TABLE ducklake_data_file ADD COLUMN partition_id BIGINT")
                    .execute(&self.pool)
                    .await?;
            }
            let changes_nullable: String = sqlx::query_scalar(
                "SELECT is_nullable FROM information_schema.columns
                 WHERE table_schema = DATABASE()
                   AND table_name = 'ducklake_snapshot_changes'
                   AND column_name = 'changes_made'",
            )
            .fetch_one(&self.pool)
            .await?;
            if changes_nullable == "NO" {
                sqlx::query("ALTER TABLE ducklake_snapshot_changes MODIFY changes_made TEXT NULL")
                    .execute(&self.pool)
                    .await?;
            }
            for table in ["ducklake_data_file", "ducklake_delete_file"] {
                let has_partial_max: i64 = sqlx::query_scalar(
                    "SELECT COUNT(*) FROM information_schema.columns
                     WHERE table_schema = DATABASE() AND table_name = ?
                       AND column_name = 'partial_max'",
                )
                .bind(table)
                .fetch_one(&self.pool)
                .await?;
                if has_partial_max == 0 {
                    sqlx::query(AssertSqlSafe(format!(
                        "ALTER TABLE {table} ADD COLUMN partial_max BIGINT"
                    )))
                    .execute(&self.pool)
                    .await?;
                }
            }
            // Seed the monotonic id allocators. snapshot_id and column_id are
            // reserved inside a transaction and read back (no RETURNING and no
            // auto-increment for these), so they live in ducklake_metadata. Seeded
            // from the current MAX so a pre-existing catalog continues without
            // reusing ids; idempotent on re-open.
            seed_counter(
                &self.pool,
                "next_column_id",
                "SELECT COALESCE(MAX(column_id), 0) FROM ducklake_column",
            )
            .await?;
            seed_counter(
                &self.pool,
                "next_snapshot_id",
                "SELECT COALESCE(MAX(snapshot_id), 0) FROM ducklake_snapshot",
            )
            .await?;
            seed_counter(
                &self.pool,
                "next_file_id",
                "SELECT COALESCE(MAX(data_file_id), 0) FROM ducklake_data_file",
            )
            .await?;
            seed_counter(
                &self.pool,
                "next_delete_file_id",
                "SELECT COALESCE(MAX(delete_file_id), 0) FROM ducklake_delete_file",
            )
            .await?;
            seed_counter(
                &self.pool,
                "next_partition_id",
                "SELECT COALESCE(MAX(partition_id), 0) FROM ducklake_partition_info",
            )
            .await?;
            seed_counter(
                &self.pool,
                "next_sort_id",
                "SELECT COALESCE(MAX(sort_id), 0) FROM ducklake_sort_info",
            )
            .await?;
            Ok(())
        })
    }

    fn begin_write_transaction(
        &self,
        schema_name: &str,
        table_name: &str,
        columns: &[ColumnDef],
        mode: WriteMode,
    ) -> Result<WriteSetupResult> {
        validate_name(schema_name, "Schema")?;
        validate_name(table_name, "Table")?;
        if columns.is_empty() {
            return Err(crate::DuckLakeError::InvalidConfig(
                "Table must have at least one column".to_string(),
            ));
        }
        block_on(async {
            let mut tx = self.pool.begin().await?;

            // Reserve the column ids first so the counter UPDATE takes the write
            // lock up front. These ids match the staged parquet field ids.
            let catalog_columns = catalog_column_defs(columns)?;
            let n = catalog_columns.len() as i64;
            let last_column_id = reserve_ids(&mut tx, "next_column_id", n).await?;
            // Freshly reserved ids. Only a genuinely-new column actually consumes
            // one below; an existing column keeps its current id, so some of these
            // may go unused (harmless monotonic-counter gaps).
            let fresh_ids: Vec<i64> = ((last_column_id - n + 1)..=last_column_id).collect();

            // The catalog head this write is based on; a Replace commit aborts if
            // another writer published a newer generation of the table past it.
            let base_snapshot_id: i64 =
                sqlx::query("SELECT COALESCE(MAX(snapshot_id), 0) FROM ducklake_snapshot")
                    .fetch_one(&mut *tx)
                    .await?
                    .try_get(0)?;

            // Tentative id for WriteSetupResult; the real one is assigned at the
            // commit (finalize_snapshot), so it may differ under concurrency.
            let snapshot_id: i64 = base_snapshot_id + 1;

            let schema_id: i64 = {
                let existing = sqlx::query(
                    "SELECT schema_id FROM ducklake_schema
                     WHERE schema_name = ? AND end_snapshot IS NULL",
                )
                .bind(schema_name)
                .fetch_optional(&mut *tx)
                .await?;

                if let Some(row) = existing {
                    row.try_get(0)?
                } else {
                    let result = sqlx::query(
                        "INSERT INTO ducklake_schema (schema_name, path, path_is_relative, begin_snapshot)
                         VALUES (?, ?, 1, ?)",
                    )
                    .bind(schema_name)
                    .bind(schema_name)
                    .bind(snapshot_id)
                    .execute(&mut *tx)
                    .await?;
                    result.last_insert_id() as i64
                }
            };

            let table_id: i64 = {
                let existing = sqlx::query(
                    "SELECT table_id FROM ducklake_table
                     WHERE schema_id = ? AND table_name = ? AND end_snapshot IS NULL",
                )
                .bind(schema_id)
                .bind(table_name)
                .fetch_optional(&mut *tx)
                .await?;

                if let Some(row) = existing {
                    row.try_get(0)?
                } else {
                    let result = sqlx::query(
                        "INSERT INTO ducklake_table (schema_id, table_name, path, path_is_relative, begin_snapshot)
                         VALUES (?, ?, ?, 1, ?)",
                    )
                    .bind(schema_id)
                    .bind(table_name)
                    .bind(table_name)
                    .bind(snapshot_id)
                    .execute(&mut *tx)
                    .await?;
                    result.last_insert_id() as i64
                }
            };

            // Get existing columns to (a) check schema compatibility for appends
            // and (b) REUSE each column's id (column_id == parquet field_id; an
            // unchanged column must keep its id, or files already written would
            // read back as NULL).
            let rows = sqlx::query(
                "SELECT column_name, column_type, column_id, parent_column
                 FROM ducklake_column
                 WHERE table_id = ? AND end_snapshot IS NULL
                 ORDER BY column_order",
            )
            .bind(table_id)
            .fetch_all(&mut *tx)
            .await?;

            let mut existing_catalog_columns = Vec::with_capacity(rows.len());
            for row in rows {
                let name: String = row.try_get(0)?;
                let ducklake_type: String = row.try_get(1)?;
                let column_id: i64 = row.try_get(2)?;
                let parent_column: Option<i64> = row.try_get(3)?;
                existing_catalog_columns.push(ExistingCatalogColumn {
                    column_id,
                    name,
                    ducklake_type,
                    parent_column,
                });
            }
            let field_ids =
                assign_column_ids(&catalog_columns, &existing_catalog_columns, &fresh_ids)?;

            // Data-write policy (§5): a data write — Replace OR Append — must NOT
            // change a column's type (that is schema evolution and must go through
            // promote_column_type). The
            // comparison is canonical (`int64` ≡ `bigint`) so an alias-only
            // restatement is a no-op. Append additionally requires a genuinely new
            // column to be nullable.
            if !existing_catalog_columns.is_empty() {
                use std::collections::HashMap;

                let existing_map: HashMap<i64, &ExistingCatalogColumn> = existing_catalog_columns
                    .iter()
                    .map(|column| (column.column_id, column))
                    .collect();

                for (new_column, column_id) in catalog_columns.iter().zip(&field_ids) {
                    if let Some(existing_column) = existing_map.get(column_id) {
                        let same_type =
                            catalog_column_type_equal(&existing_column.ducklake_type, new_column);
                        if !same_type {
                            return Err(crate::error::DuckLakeError::UnsupportedTypeChange {
                                operation: TypeChangeOperation::DataWrite {
                                    mode: match mode {
                                        WriteMode::Replace => TypeChangeWriteMode::Replace,
                                        WriteMode::Append => TypeChangeWriteMode::Append,
                                    },
                                },
                                column: new_column.name.clone(),
                                from: existing_column.ducklake_type.clone(),
                                to: new_column.ducklake_type.clone(),
                            });
                        }
                    } else if mode == WriteMode::Append
                        && new_column.parent_index.is_none()
                        && !new_column.is_nullable
                    {
                        return Err(crate::error::DuckLakeError::InvalidConfig(format!(
                            "Schema evolution error: new column '{}' must be nullable. Adding non-nullable columns is not allowed.",
                            new_column.name
                        )));
                    }
                }
            }

            // Final per-column ids: reuse the existing id for a column already in
            // the table, consume a freshly reserved id only for a genuinely new
            // column. These are baked into the staged parquet's field_id metadata,
            // so they must equal the ids finalize_snapshot commits. Column rows
            // themselves are written at the commit point (not here): the read path
            // resolves columns by `end_snapshot IS NULL` only, so inserting at begin
            // would leak the new generation to concurrent reads.
            // No snapshot row, no column rows, and no Replace retirement are written
            // here — all are deferred to the atomic commit so the head never
            // resolves to an incomplete snapshot. This TX commits only the
            // idempotent get-or-create schema/table rows; they carry begin_snapshot
            // = the reserved id and stay invisible until the snapshot publishes,
            // since schema/table reads ARE snapshot-scoped.
            tx.commit().await?;

            Ok(WriteSetupResult {
                snapshot_id,
                base_snapshot_id,
                schema_id,
                table_id,
                column_ids: top_level_column_ids(&catalog_columns, &field_ids)?,
                field_ids,
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mysql_inlining_gate_accepts_nested_types() {
        let nested = DataType::List(std::sync::Arc::new(arrow::datatypes::Field::new(
            "item",
            DataType::Int32,
            true,
        )));

        assert!(mysql_type_inlines(&nested));
        assert_eq!(inlined_mysql_type(&nested), "LONGTEXT");
    }

    #[test]
    fn mysql_inlined_types_follow_sqlite_style_encodings() {
        assert_eq!(inlined_mysql_type(&DataType::Int32), "BIGINT");
        assert_eq!(inlined_mysql_type(&DataType::UInt64), "BIGINT UNSIGNED");
        assert_eq!(inlined_mysql_type(&DataType::Float32), "FLOAT");
        assert_eq!(inlined_mysql_type(&DataType::Float64), "DOUBLE");
        assert_eq!(
            inlined_mysql_type(&DataType::Decimal128(20, 4)),
            "DECIMAL(20,4)"
        );
        assert_eq!(inlined_mysql_type(&DataType::Binary), "LONGBLOB");
        assert_eq!(
            inlined_mysql_type(&DataType::FixedSizeBinary(16)),
            "LONGTEXT"
        );
        assert_eq!(
            inlined_mysql_type(&DataType::FixedSizeBinary(32)),
            "LONGBLOB"
        );
        assert_eq!(inlined_mysql_type(&DataType::Date32), "DATE");
        assert_eq!(
            inlined_mysql_type(&DataType::Timestamp(
                arrow::datatypes::TimeUnit::Nanosecond,
                None,
            )),
            "BIGINT"
        );
        assert_eq!(
            inlined_mysql_type(&DataType::Timestamp(
                arrow::datatypes::TimeUnit::Microsecond,
                None,
            )),
            "DATETIME(6)"
        );
    }

    #[test]
    fn mysql_indexes_skip_unbounded_text_and_binary_columns() {
        assert!(!mysql_type_supports_index("LONGTEXT"));
        assert!(!mysql_type_supports_index("longblob"));
        assert!(mysql_type_supports_index("BIGINT UNSIGNED"));
    }

    #[test]
    fn mysql_native_inline_inserts_cast_parameters_only_when_needed() {
        let uint64 = UInt64Array::from(vec![u64::MAX]);
        let mut query = QueryBuilder::<MySql>::new("VALUES (");
        push_inlined_mysql_value(&mut query, &uint64, 0, "BIGINT UNSIGNED").unwrap();
        query.push(')');
        assert_eq!(
            query.sql(),
            "VALUES (CAST(CAST(? AS DECIMAL(20,0)) AS UNSIGNED))"
        );

        let decimal = arrow::array::Decimal128Array::from(vec![123_456_i128])
            .with_precision_and_scale(20, 4)
            .unwrap();
        let mut query = QueryBuilder::<MySql>::new("VALUES (");
        push_inlined_mysql_value(&mut query, &decimal, 0, "DECIMAL(20,4)").unwrap();
        query.push(')');
        assert_eq!(query.sql(), "VALUES (CAST(? AS DECIMAL(20,4)))");
    }
}
