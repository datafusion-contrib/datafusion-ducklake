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
use crate::inlined_reclaim::{SUPERSEDED_INLINED_TABLES_SQL, dead_table_inlined_storage};
use crate::maintenance::{
    CleanupCriteria, ExpireCriteria, ExpiredSnapshot, ScheduledFile, format_sql_timestamp,
};
use crate::metadata_provider::{InlinedSystemColumns, block_on};
use crate::metadata_provider_mysql::{catalog_inlined_system_columns, inlined_table_layout};
use crate::metadata_writer::directory_path;
use crate::metadata_writer::{
    ColumnDef, ColumnStat, CommitIds, CompactionOutputFile, CompactionSourceFile, ConflictScope,
    DataFileInfo, DeleteFileEntry, DeleteFileInfo, ExistingCatalogColumn,
    INLINED_INDEX_COLUMNS_SETTING, InlinedFlushDeleteFile, InlinedFlushTable, InlinedRowRef,
    MetadataWriter, MultiTableCommit, SnapshotCommitMetadata, SourceRetirement, StagedTableData,
    StagedTableWrite, WriteMode, WriteSetupResult, assign_column_ids, carried_row_id_start,
    catalog_column_defs, catalog_column_type_equal, catalog_column_type_requires_migration,
    catalog_columns_differ, encode_inlined_index_columns, inlined_delete_groups,
    inlined_flush_changes, inlined_flush_conflict, live_inlined_index_columns,
    parse_inlined_index_columns, quote_snapshot_name, quote_snapshot_table, snapshot_has_change,
    staged_table_write_changes, table_storage_changes, top_level_column_ids,
    validate_delete_entries, validate_inlined_flush, validate_inlined_index_columns, validate_name,
    validate_table_setting,
};
use crate::metadata_writer::{
    DroppedDataFile, InlinedFileDelete, InlinedFlushFileDelete, SQL_TABLE_INLINING_COLUMNS,
    SourceFile, TableChanges, enables_inlining, inlined_deletion_conflict,
    reject_reserved_inlining_columns, rewritten_delete_conflict, rewritten_deletes_snapshot,
    row_mutation_source_files, source_file_id_list, staged_conflict, table_conflict,
    validate_row_mutation_stage,
};
use crate::partition::PartitionTransform;
use arrow::array::{
    Array, BinaryArray, BinaryViewArray, BooleanArray, Date32Array, FixedSizeBinaryArray,
    Float32Array, Float64Array, Int8Array, Int16Array, Int32Array, Int64Array, LargeBinaryArray,
    LargeStringArray, StringArray, TimestampNanosecondArray, UInt8Array, UInt16Array, UInt32Array,
    UInt64Array,
};
use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;
use sqlx::Row;
use sqlx::mysql::{MySql, MySqlConnection, MySqlPool, MySqlPoolOptions};
use sqlx::{AssertSqlSafe, Connection, QueryBuilder};

const DEFAULT_MAX_CONNECTIONS: u32 = 5;

fn id_list(ids: &[i64]) -> String {
    ids.iter()
        .map(i64::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

const RESOLVED_PATH: &str = "CASE
    WHEN NOT df.path_is_relative THEN df.path
    WHEN NOT t.path_is_relative THEN CONCAT(TRIM(TRAILING '/' FROM t.path), '/', TRIM(LEADING '/' FROM df.path))
    ELSE CONCAT(TRIM(TRAILING '/' FROM s.path), '/', TRIM(TRAILING '/' FROM t.path), '/', TRIM(LEADING '/' FROM df.path))
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

/// Whether every value of `array` fits the MySQL column [`inlined_mysql_type`]
/// stores it in. MySQL has no NaN or infinity in `FLOAT`/`DOUBLE`, and
/// `DATE`/`DATETIME(6)` span the years 1000 to 9999 only; a write holding
/// another value is written to Parquet instead, which holds it exactly.
fn mysql_values_inline(array: &dyn Array) -> bool {
    use arrow::datatypes::TimeUnit;
    let epoch = chrono::NaiveDate::from_ymd_opt(1970, 1, 1).expect("valid date");
    let first = chrono::NaiveDate::from_ymd_opt(1000, 1, 1).expect("valid date");
    let last = chrono::NaiveDate::from_ymd_opt(9999, 12, 31).expect("valid date");
    let first_day = first.signed_duration_since(epoch).num_days();
    let last_day = last.signed_duration_since(epoch).num_days();
    let all = |valid: &dyn Fn(usize) -> bool| {
        (0..array.len()).all(|row| array.is_null(row) || valid(row))
    };
    match array.data_type() {
        DataType::Float32 => {
            let values = array
                .as_any()
                .downcast_ref::<Float32Array>()
                .expect("Arrow data type and array implementation agree");
            all(&|row| values.value(row).is_finite())
        },
        DataType::Float64 => {
            let values = array
                .as_any()
                .downcast_ref::<Float64Array>()
                .expect("Arrow data type and array implementation agree");
            all(&|row| values.value(row).is_finite())
        },
        DataType::Date32 => {
            let values = array
                .as_any()
                .downcast_ref::<Date32Array>()
                .expect("Arrow data type and array implementation agree");
            all(&|row| (first_day..=last_day).contains(&i64::from(values.value(row))))
        },
        DataType::Timestamp(unit, _) if *unit != TimeUnit::Nanosecond => {
            let per_second: i64 = match unit {
                TimeUnit::Second => 1,
                TimeUnit::Millisecond => 1_000,
                TimeUnit::Microsecond => 1_000_000,
                TimeUnit::Nanosecond => unreachable!("guarded above"),
            };
            let low = first_day * 86_400 * per_second;
            let high = (last_day + 1) * 86_400 * per_second - 1;
            let values = arrow::array::cast::as_primitive_array::<arrow::datatypes::Int64Type>(
                &arrow::compute::cast(array, &DataType::Int64).expect("timestamps cast to Int64"),
            )
            .clone();
            all(&|row| (low..=high).contains(&values.value(row)))
        },
        _ => true,
    }
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
        // `DATETIME(6)` holds the UTC instant without an offset, which MySQL
        // rejects in either spelling (`Z`, or DuckDB's `+00`).
        DataType::Timestamp(unit, Some(_)) => {
            // The same instants, read without their zone.
            let utc = arrow::array::make_array(
                array
                    .slice(row, 1)
                    .to_data()
                    .into_builder()
                    .data_type(DataType::Timestamp(*unit, None))
                    .build()?,
            );
            query.push_bind(crate::metadata_writer::inlined_text_value(utc.as_ref(), 0)?);
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
        let url = connection_string.to_string();
        let pool = crate::metadata_provider::connect_on_catalog_runtime(async move {
            MySqlPoolOptions::new()
                .max_connections(max_connections)
                .connect(&url)
                .await
        })
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
    /// surviving snapshot. Physical files are queued for later cleanup. A fully
    /// expired table's inlined data and inlined delete tables are dropped with its
    /// registry rows.
    ///
    /// MySQL commits DDL implicitly, so the drops cannot share the metadata
    /// transaction. The expire commits first and keeps a dead table's
    /// `ducklake_table` and registry rows; the drops follow, and only then are
    /// those rows deleted. The dead-table query that finds the work runs on every
    /// call, so an expire interrupted between those steps is finished by the next
    /// one, whether or not that one expires a snapshot.
    pub fn expire_snapshots(&self, criteria: ExpireCriteria) -> Result<Vec<ExpiredSnapshot>> {
        let candidates = self.expire_snapshot_rows(criteria)?;
        block_on(self.drop_dead_table_inlined_storage())?;
        Ok(candidates)
    }

    /// The transactional part of [`Self::expire_snapshots`]: everything except a
    /// dead table's `ducklake_table` rows, its inlined storage and registry rows.
    fn expire_snapshot_rows(&self, criteria: ExpireCriteria) -> Result<Vec<ExpiredSnapshot>> {
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
            let dead_tables = sqlx::query_scalar::<_, i64>(DEAD_TABLES_SQL)
                .fetch_all(&mut *tx)
                .await?;
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
            // A dead table's `ducklake_table` and registry rows outlive this
            // transaction: they are what lists its inlined storage until it is dropped
            // (`drop_dead_table_inlined_storage`).
            if !dead_tables.is_empty() {
                let ids = id_list(&dead_tables);
                for table in ["ducklake_table_stats", "ducklake_column", "ducklake_schema_versions"]
                {
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

    /// Drop the inlined data and inlined delete tables of every fully expired
    /// table, then delete its registry and `ducklake_table` rows. The tables are
    /// found by official's dead-table query, which still lists them until those
    /// last rows go, so a run interrupted after any drop is completed by the next.
    async fn drop_dead_table_inlined_storage(&self) -> Result<()> {
        let dead_tables = sqlx::query_scalar::<_, i64>(DEAD_TABLES_SQL)
            .fetch_all(&self.pool)
            .await?;
        if dead_tables.is_empty() {
            return Ok(());
        }
        let ids = id_list(&dead_tables);
        // A catalog this crate's writer created before data inlining has no registry.
        let has_registry = self.has_inlined_data_registry().await?;
        let registered = if has_registry {
            sqlx::query_scalar::<_, String>(AssertSqlSafe(format!(
                "SELECT table_name FROM ducklake_inlined_data_tables WHERE table_id IN ({ids})"
            )))
            .fetch_all(&self.pool)
            .await?
        } else {
            Vec::new()
        };
        for table in dead_table_inlined_storage(registered, &dead_tables)? {
            sqlx::query(AssertSqlSafe(format!(
                "DROP TABLE IF EXISTS {}",
                quote_ident(&table)
            )))
            .execute(&self.pool)
            .await?;
        }
        let mut tx = self.pool.begin().await?;
        let mut registries = vec!["ducklake_table"];
        if has_registry {
            registries.insert(0, "ducklake_inlined_data_tables");
        }
        for table in registries {
            sqlx::query(AssertSqlSafe(format!(
                "DELETE FROM {table} WHERE table_id IN ({ids})"
            )))
            .execute(&mut *tx)
            .await?;
        }
        tx.commit().await?;
        Ok(())
    }

    /// Rows of `ducklake_files_scheduled_for_deletion` the criteria select.
    pub(crate) fn list_scheduled_for_deletion(
        &self,
        criteria: &CleanupCriteria,
    ) -> Result<Vec<ScheduledFile>> {
        block_on(async {
            let rows = match criteria {
                CleanupCriteria::All => {
                    sqlx::query(
                        "SELECT data_file_id, path, path_is_relative
                         FROM ducklake_files_scheduled_for_deletion",
                    )
                    .fetch_all(&self.pool)
                    .await?
                },
                CleanupCriteria::OlderThan(cutoff) => {
                    sqlx::query(
                        "SELECT data_file_id, path, path_is_relative
                         FROM ducklake_files_scheduled_for_deletion
                         WHERE schedule_start < ?",
                    )
                    .bind(format_sql_timestamp(cutoff))
                    .fetch_all(&self.pool)
                    .await?
                },
            };
            rows.into_iter()
                .map(|row| {
                    Ok(ScheduledFile {
                        data_file_id: row.try_get(0)?,
                        path: row.try_get(1)?,
                        path_is_relative: row.try_get::<bool, _>(2)?,
                        // Single-catalog layout — see the SQLite writer.
                        scheduled_by_owner: None,
                    })
                })
                .collect()
        })
    }

    /// Remove scheduled-deletion bookkeeping rows after their objects are gone.
    pub(crate) fn remove_scheduled(&self, ids: &[i64]) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        block_on(async {
            sqlx::query(AssertSqlSafe(format!(
                "DELETE FROM ducklake_files_scheduled_for_deletion WHERE data_file_id IN ({})",
                id_list(ids)
            )))
            .execute(&self.pool)
            .await?;
            Ok(())
        })
    }

    /// Every path the catalog still references: data files, delete files and
    /// files scheduled for deletion.
    pub(crate) fn list_referenced_paths(&self) -> Result<Vec<(String, bool)>> {
        block_on(async {
            let rows = sqlx::query(AssertSqlSafe(format!(
                "SELECT {RESOLVED_PATH} AS path, {REL_FLAG} AS path_is_relative
                 FROM ducklake_data_file AS df
                 JOIN ducklake_table AS t ON t.table_id = df.table_id
                 JOIN ducklake_schema AS s ON s.schema_id = t.schema_id
                 UNION ALL
                 SELECT {RESOLVED_PATH} AS path, {REL_FLAG} AS path_is_relative
                 FROM ducklake_delete_file AS df
                 JOIN ducklake_table AS t ON t.table_id = df.table_id
                 JOIN ducklake_schema AS s ON s.schema_id = t.schema_id
                 UNION ALL
                 SELECT path, path_is_relative
                 FROM ducklake_files_scheduled_for_deletion"
            )))
            .fetch_all(&self.pool)
            .await?;
            rows.into_iter()
                .map(|row| Ok((row.try_get::<String, _>(0)?, row.try_get::<bool, _>(1)?)))
                .collect()
        })
    }

    /// Whether the catalog has the `ducklake_inlined_data_tables` registry, from
    /// the capability probe every inlined read uses.
    async fn has_inlined_data_registry(&self) -> Result<bool> {
        crate::MySqlMetadataProvider::from_pool(self.pool.clone())
            .has_inlined_data_registry()
            .await
    }
}

/// Official's dead-table query (`DeleteSnapshots`): tables no surviving snapshot
/// can read and with no version a surviving snapshot can read.
const DEAD_TABLES_SQL: &str = "SELECT t.table_id FROM ducklake_table t
     WHERE t.end_snapshot IS NOT NULL AND NOT EXISTS (
         SELECT 1 FROM ducklake_snapshot
         WHERE snapshot_id >= t.begin_snapshot AND snapshot_id < t.end_snapshot)
     AND NOT EXISTS (
         SELECT 1 FROM ducklake_table t2
         WHERE t2.table_id = t.table_id
           AND (t2.end_snapshot IS NULL OR EXISTS (
               SELECT 1 FROM ducklake_snapshot
               WHERE snapshot_id >= t2.begin_snapshot
                 AND snapshot_id < t2.end_snapshot)))";

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

/// Run `alter`, which adds `column` to `table`. MySQL has no `ADD COLUMN IF
/// NOT EXISTS`, so the caller probes first; an initialization that passed the
/// probe while another added the column fails with a duplicate column, which
/// is accepted when the column now exists.
async fn add_column(pool: &MySqlPool, table: &str, column: &str, alter: &str) -> Result<()> {
    let Err(error) = sqlx::query(AssertSqlSafe(alter.to_string()))
        .execute(pool)
        .await
    else {
        return Ok(());
    };
    let exists: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM information_schema.columns
         WHERE table_schema = DATABASE() AND table_name = ? AND column_name = ?",
    )
    .bind(table)
    .bind(column)
    .fetch_one(pool)
    .await?;
    if exists == 0 {
        return Err(error.into());
    }
    Ok(())
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
/// or delete file of the table has `begin_snapshot` or `end_snapshot` newer than
/// `base_snapshot` (the head observed when this write began), or an inlined data
/// or inlined delete row was committed after it, another writer changed the
/// table in the meantime, so this `Replace` aborts with
/// [`crate::DuckLakeError::Conflict`] rather than clobbering it. (`Append` does
/// not call this unless the write set an expected base snapshot: concurrent
/// appends commute.)
/// An inlined-data flush ([`ConflictScope::InlinedFlush`]) does not count delete
/// files: official DuckLake commits a flush over a concurrent delete file.
async fn detect_replace_conflict(
    tx: &mut sqlx::Transaction<'_, sqlx::MySql>,
    table_id: i64,
    base_snapshot: i64,
    scope: ConflictScope,
) -> Result<()> {
    let conflict: Option<i64> = sqlx::query(
        "SELECT 1 FROM DUAL WHERE EXISTS (SELECT 1 FROM ducklake_data_file
             WHERE table_id = ? AND (begin_snapshot > ? OR end_snapshot > ?))
           OR (? AND EXISTS (SELECT 1 FROM ducklake_delete_file
             WHERE table_id = ? AND (begin_snapshot > ? OR end_snapshot > ?)))",
    )
    .bind(table_id)
    .bind(base_snapshot)
    .bind(base_snapshot)
    .bind(scope.counts_delete_files())
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
    for (table_name, system) in inlined_tables(tx, table_id).await? {
        let sql = format!(
            "SELECT 1 FROM {} WHERE {} > ? OR {} > ? LIMIT 1",
            quote_ident(&table_name),
            system.begin_snapshot,
            system.end_snapshot,
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
    let inlined_deletes = crate::metadata_provider::inlined_delete_table_name(table_id)?;
    let exists: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM information_schema.tables
         WHERE table_schema = DATABASE() AND table_name = ?",
    )
    .bind(&inlined_deletes)
    .fetch_one(&mut **tx)
    .await?;
    if exists > 0 {
        let sql = format!(
            "SELECT 1 FROM {} WHERE begin_snapshot > ? LIMIT 1",
            quote_ident(&inlined_deletes)
        );
        if sqlx::query(AssertSqlSafe(sql))
            .bind(base_snapshot)
            .fetch_optional(&mut **tx)
            .await?
            .is_some()
        {
            return Err(crate::DuckLakeError::Conflict(format!(
                "Replace on table {table_id} conflicts with an inlined delete committed since \
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
    sources: &[SourceFile<'_>],
) -> Result<()> {
    if sources.is_empty() {
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

    let concurrent: Vec<(i64, Option<i64>)> = sqlx::query_as(AssertSqlSafe(format!(
        "SELECT newer.file_id, newer.row_id FROM {} newer
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
        source_file_id_list(sources),
        quote_ident(&table),
    )))
    .bind(base_snapshot)
    .bind(base_snapshot)
    .fetch_all(&mut **tx)
    .await?;
    inlined_deletion_conflict(table_id, base_snapshot, sources, &concurrent)
}

/// Retire the prior generation's still-visible data files and inlined rows at
/// `snapshot_id`. The `begin_snapshot < snapshot_id` guard spares files
/// registered for *this* snapshot, so a multi-file write does not retire its
/// own siblings. `ducklake_table_stats` is left as it is, as official's `DELETE`
/// leaves it: `record_count` counts every row ever inserted.
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

    for (table_name, system) in inlined_tables(tx, table_id).await? {
        let sql = format!(
            "UPDATE {} SET {end} = ? WHERE {end} IS NULL AND {begin} < ?",
            quote_ident(&table_name),
            end = system.end_snapshot,
            begin = system.begin_snapshot,
        );
        sqlx::query(AssertSqlSafe(sql))
            .bind(snapshot_id)
            .bind(snapshot_id)
            .execute(&mut **tx)
            .await?;
    }
    Ok(())
}

/// Register a delete file an inlined flush wrote for `data_file_id`.
/// Register each of `entries`, a delete file that records its positions'
/// snapshots, against its data file. One that replaces the file's live delete
/// file carries every position of it, so the replaced row is removed and its
/// object scheduled for deletion, as official's `DeleteOverwrittenDeleteFiles`
/// does. Fails with `conflict()` when a data file's `end_snapshot` is not the
/// entry's, or the delete file it replaces (a live file's live one, a retired
/// file's latest one) is not the entry's.
async fn replace_delete_files(
    tx: &mut sqlx::Transaction<'_, sqlx::MySql>,
    table_id: i64,
    entries: &[InlinedFlushFileDelete],
    conflict: impl Fn() -> crate::DuckLakeError,
) -> Result<()> {
    for entry in entries {
        let target_end: Option<Option<i64>> = sqlx::query_scalar(
            "SELECT end_snapshot FROM ducklake_data_file WHERE data_file_id = ?",
        )
        .bind(entry.data_file_id)
        .fetch_optional(&mut **tx)
        .await?;
        // A live file's live delete file, or a retired file's latest one.
        let current: Option<i64> = sqlx::query_scalar(
            "SELECT delete_file_id FROM ducklake_delete_file
             WHERE data_file_id = ? AND (end_snapshot IS NULL OR ?)
             ORDER BY begin_snapshot DESC LIMIT 1",
        )
        .bind(entry.data_file_id)
        .bind(entry.data_file_end.is_some())
        .fetch_optional(&mut **tx)
        .await?;
        if target_end != Some(entry.data_file_end) || current != entry.expected_prev_delete_file {
            return Err(conflict());
        }
        if let Some(previous) = entry.expected_prev_delete_file {
            sqlx::query(AssertSqlSafe(format!(
                "INSERT INTO ducklake_files_scheduled_for_deletion
                     (data_file_id, path, path_is_relative)
                 SELECT df.delete_file_id, {RESOLVED_PATH}, {REL_FLAG}
                 FROM ducklake_delete_file df
                 JOIN ducklake_table t ON t.table_id = df.table_id
                 JOIN ducklake_schema s ON s.schema_id = t.schema_id
                 WHERE df.delete_file_id = ?"
            )))
            .bind(previous)
            .execute(&mut **tx)
            .await?;
            sqlx::query("DELETE FROM ducklake_delete_file WHERE delete_file_id = ?")
                .bind(previous)
                .execute(&mut **tx)
                .await?;
        }
        insert_flush_delete_file(tx, table_id, entry.data_file_id, &entry.delete).await?;
    }
    Ok(())
}

async fn insert_flush_delete_file(
    tx: &mut sqlx::Transaction<'_, sqlx::MySql>,
    table_id: i64,
    data_file_id: i64,
    delete: &InlinedFlushDeleteFile,
) -> Result<()> {
    let delete_file_id = reserve_delete_file_ids(tx, 1).await?[0];
    sqlx::query(
        "INSERT INTO ducklake_delete_file
             (delete_file_id, data_file_id, table_id, path, path_is_relative,
              file_size_bytes, footer_size, delete_count, begin_snapshot, partial_max,
              end_snapshot)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
    )
    .bind(delete_file_id)
    .bind(data_file_id)
    .bind(table_id)
    .bind(&delete.delete.path)
    .bind(delete.delete.path_is_relative)
    .bind(delete.delete.file_size_bytes)
    .bind(delete.delete.footer_size)
    .bind(delete.delete.delete_count)
    .bind(delete.begin_snapshot)
    .bind(delete.partial_max)
    .bind(delete.end_snapshot)
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
    inlined: bool,
    commit_metadata: &SnapshotCommitMetadata,
) -> Result<()> {
    let replaced_existing_data: bool = sqlx::query_scalar(
        "SELECT EXISTS(
            SELECT 1 FROM ducklake_data_file
            WHERE table_id = ? AND end_snapshot = ?
         )",
    )
    .bind(table_id)
    .bind(snapshot_id)
    .fetch_one(&mut **tx)
    .await?;
    let mut replaced_inlined = false;
    if mode == WriteMode::Replace {
        // A Replace over inline-only prior data ends inline rows, not files.
        for (inlined_table, system) in inlined_tables(tx, table_id).await? {
            let sql = format!(
                "SELECT EXISTS(SELECT 1 FROM {} WHERE {} = ?)",
                quote_ident(&inlined_table),
                system.end_snapshot,
            );
            if sqlx::query_scalar(AssertSqlSafe(sql))
                .bind(snapshot_id)
                .fetch_one(&mut **tx)
                .await?
            {
                replaced_inlined = true;
                break;
            }
        }
    }

    let write_changes = table_storage_changes(
        table_id,
        Some(inlined),
        has_deletes || (mode == WriteMode::Replace && replaced_existing_data),
        replaced_inlined,
    );
    record_table_changes(
        tx,
        snapshot_id,
        table_id,
        schema_name,
        table_name,
        &write_changes,
        commit_metadata,
    )
    .await
}

async fn record_table_changes(
    tx: &mut sqlx::Transaction<'_, sqlx::MySql>,
    snapshot_id: i64,
    table_id: i64,
    schema_name: &str,
    table_name: &str,
    write_changes: &str,
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
    let mut changes = Vec::new();
    if schema_begin_snapshot == snapshot_id {
        let entry = format!("created_schema:{}", quote_snapshot_name(schema_name));
        let recorded: Option<String> = sqlx::query_scalar(
            "SELECT changes_made FROM ducklake_snapshot_changes WHERE snapshot_id = ?",
        )
        .bind(snapshot_id)
        .fetch_optional(&mut **tx)
        .await?
        .flatten();
        if !snapshot_has_change(recorded.as_deref().unwrap_or_default(), &entry) {
            changes.push(entry);
        }
    }
    if table_begin_snapshot == snapshot_id {
        changes.push(format!(
            "created_table:{}",
            quote_snapshot_table(schema_name, table_name)
        ));
    } else if altered {
        changes.push(format!("altered_table:{table_id}"));
    }
    changes.push(write_changes.to_string());
    record_snapshot_changes(tx, snapshot_id, &changes.join(","), commit_metadata).await
}

/// Give legacy `data_path`, schema path, and table path values the trailing `/`
/// DuckLake readers expect. Idempotent.
async fn migrate_directory_paths(pool: &MySqlPool) -> Result<()> {
    let mut tx = pool.begin().await?;
    sqlx::query(
        "UPDATE ducklake_metadata SET `value` = CONCAT(`value`, '/')
         WHERE `key` = 'data_path' AND scope IS NULL AND `value` <> '' AND `value` NOT LIKE '%/'",
    )
    .execute(&mut *tx)
    .await?;
    sqlx::query("UPDATE ducklake_schema SET path = CONCAT(path, '/') WHERE path <> '' AND path NOT LIKE '%/'")
        .execute(&mut *tx)
        .await?;
    sqlx::query("UPDATE ducklake_table SET path = CONCAT(path, '/') WHERE path <> '' AND path NOT LIKE '%/'")
        .execute(&mut *tx)
        .await?;
    tx.commit().await?;
    Ok(())
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
    crate::metadata_writer::reject_duplicate_column_stats(column_stats)?;
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

/// The field ids and DuckLake types of the table's live columns and nested
/// fields, by which the table-wide statistics merge compares bounds.
async fn live_stat_fields(
    tx: &mut sqlx::Transaction<'_, sqlx::MySql>,
    table_id: i64,
) -> Result<Vec<(i64, String)>> {
    Ok(sqlx::query_as(
        "SELECT column_id, column_type FROM ducklake_column
         WHERE table_id = ? AND end_snapshot IS NULL",
    )
    .bind(table_id)
    .fetch_all(&mut **tx)
    .await?)
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

/// The physical `ducklake_inlined_data_*` tables registered for the table, with
/// their system column names. Mirrors the SQLite writer's `inlined_tables`.
async fn inlined_tables(
    tx: &mut sqlx::Transaction<'_, sqlx::MySql>,
    table_id: i64,
) -> Result<Vec<(String, InlinedSystemColumns)>> {
    let names: Vec<String> = sqlx::query_scalar(
        "SELECT table_name FROM ducklake_inlined_data_tables WHERE table_id = ?",
    )
    .bind(table_id)
    .fetch_all(&mut **tx)
    .await?;
    let mut tables = Vec::with_capacity(names.len());
    for name in names {
        // MySQL shows a transaction no table created after its snapshot, and
        // fails a read of one: such a table is one this commit created, on a
        // connection of its own, and holds no row the commit reads.
        let created_since: bool = sqlx::query_scalar(
            "SELECT NOT EXISTS(SELECT 1 FROM information_schema.columns
                               WHERE table_schema = DATABASE() AND table_name = ?)",
        )
        .bind(&name)
        .fetch_one(&mut **tx)
        .await?;
        if created_since {
            continue;
        }
        let system = inlined_table_layout(&mut **tx, &name).await?.system;
        tables.push((name, system));
    }
    Ok(tables)
}

/// Count the table's still-visible inlined rows across its registered physical
/// tables. Mirrors the SQLite writer's `live_inlined_row_count`.
async fn live_inlined_row_count(
    tx: &mut sqlx::Transaction<'_, sqlx::MySql>,
    table_id: i64,
) -> Result<i64> {
    let mut total = 0;
    for (table_name, system) in inlined_tables(tx, table_id).await? {
        let sql = format!(
            "SELECT COUNT(*) FROM {} WHERE {} IS NULL",
            quote_ident(&table_name),
            system.end_snapshot,
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
/// Official's table-level conflict check for a staged insert, `DELETE` or
/// `UPDATE` against every snapshot committed after its base snapshot and before
/// `snapshot_id` (see [`staged_conflict`]).
async fn check_staged_conflicts(
    tx: &mut sqlx::Transaction<'_, sqlx::MySql>,
    write: &StagedTableWrite,
    snapshot_id: i64,
) -> Result<()> {
    let changes = changes_between(tx, write.base_snapshot_id, snapshot_id).await?;
    staged_conflict(write, &changes).map_or(Ok(()), Err)
}

/// The `changes_made` of every snapshot after `base_snapshot` and before
/// `snapshot_id`, the commit's own.
async fn changes_between(
    tx: &mut sqlx::Transaction<'_, sqlx::MySql>,
    base_snapshot: i64,
    snapshot_id: i64,
) -> Result<Vec<Option<String>>> {
    Ok(sqlx::query_scalar(
        "SELECT changes_made FROM ducklake_snapshot_changes
         WHERE snapshot_id > ? AND snapshot_id < ?",
    )
    .bind(base_snapshot)
    .bind(snapshot_id)
    .fetch_all(&mut **tx)
    .await?)
}

/// The table the data files `deletes` delete from belong to: the table the
/// statement read, which a concurrent drop may have replaced under its name.
async fn deleted_files_table(
    tx: &mut sqlx::Transaction<'_, sqlx::MySql>,
    table_id: i64,
    deletes: &[DeleteFileEntry],
) -> Result<i64> {
    let Some(entry) = deletes.first() else {
        return Ok(table_id);
    };
    let source: Option<i64> =
        sqlx::query_scalar("SELECT table_id FROM ducklake_data_file WHERE data_file_id = ?")
            .bind(entry.data_file_id)
            .fetch_optional(&mut **tx)
            .await?;
    Ok(source.unwrap_or(table_id))
}

/// Official's table-level conflict check for a commit at `snapshot_id` making
/// `ours` to `table_id` (see [`table_conflict`]).
async fn check_table_conflicts(
    tx: &mut sqlx::Transaction<'_, sqlx::MySql>,
    table_id: i64,
    base_snapshot: i64,
    snapshot_id: i64,
    ours: TableChanges,
) -> Result<()> {
    let changes = changes_between(tx, base_snapshot, snapshot_id).await?;
    table_conflict(table_id, ours, &changes).map_or(Ok(()), Err)
}

async fn apply_inlined_deletes(
    tx: &mut sqlx::Transaction<'_, sqlx::MySql>,
    table_id: i64,
    snapshot_id: i64,
    base_snapshot: i64,
    rows: &[InlinedRowRef],
) -> Result<()> {
    if rows.is_empty() {
        return Ok(());
    }
    let registered = inlined_tables(tx, table_id).await?;
    for (name, system) in &registered {
        let sql = format!(
            "SELECT EXISTS(SELECT 1 FROM {} WHERE ({begin} > ? AND {begin} < ?) OR ({end} > ? AND {end} < ?))",
            quote_ident(name),
            begin = system.begin_snapshot,
            end = system.end_snapshot,
        );
        let changed: bool = sqlx::query_scalar(AssertSqlSafe(sql))
            .bind(base_snapshot)
            .bind(snapshot_id)
            .bind(base_snapshot)
            .bind(snapshot_id)
            .fetch_one(&mut **tx)
            .await?;
        if changed {
            return Err(crate::DuckLakeError::Conflict(
                "inlined rows changed since the delete snapshot".to_string(),
            ));
        }
    }
    for (name, row_ids) in inlined_delete_groups(rows) {
        let Some((_, system)) = registered.iter().find(|(table, _)| table == name) else {
            return Err(crate::DuckLakeError::Conflict(format!(
                "inlined deletes reference unregistered table '{name}'"
            )));
        };
        let ids = row_ids
            .iter()
            .map(i64::to_string)
            .collect::<Vec<_>>()
            .join(",");
        let sql = format!(
            "UPDATE {} SET {end} = ? WHERE {row_id} IN ({ids}) AND {end} IS NULL AND {begin} <> ?",
            quote_ident(name),
            end = system.end_snapshot,
            row_id = system.row_id,
            begin = system.begin_snapshot,
        );
        sqlx::query(AssertSqlSafe(sql))
            .bind(snapshot_id)
            .bind(snapshot_id)
            .execute(&mut **tx)
            .await?;
    }
    Ok(())
}

/// Merge the statistics of `files`, newly committed to the table, into its
/// `ducklake_table_column_stats` roll-up, as official DuckLake does when it
/// commits data files (see [`crate::stats_encode::merge_file_column_stats`]).
/// Called before the write advances `ducklake_table_stats`, which says whether
/// the table had statistics before it.
async fn merge_file_table_column_stats(
    tx: &mut sqlx::Transaction<'_, sqlx::MySql>,
    table_id: i64,
    fields: &[(i64, String)],
    files: &[Vec<crate::stats_encode::FileColumnStat>],
) -> Result<()> {
    let initialized = table_stats_initialized(tx, table_id).await?;
    let live: Vec<i64> = fields.iter().map(|(id, _)| *id).collect();
    let existing = load_table_column_stats(tx, table_id).await?;
    let merged =
        crate::stats_encode::merge_file_column_stats(existing, initialized, files, &live, |id| {
            crate::metadata_writer::stat_kind_of(fields, id)
        });
    store_table_column_stats(tx, table_id, merged).await
}

/// The table's stored `ducklake_table_column_stats` rows, less those whose
/// bounds are unknown. A row with neither bound reads as holding no valid value
/// (see [`crate::stats_encode::ColumnStats::from_global`]). Official writes one
/// while the column's live files hold only NULLs or only NaN (whose bounds no
/// reader uses once `contains_nan` is set), after a rewrite recomputes the
/// statistics from such files, and for a `GEOMETRY` column, whose bounds are in
/// `extra_stats`; this crate keeps those rows, as official merges into them.
/// Releases of this crate up to 0.9 also wrote one, without NaN, when a live
/// file's bounds were missing: a row with neither bound, no NaN and no
/// `extra_stats` over a live file of an earlier commit not known to hold only
/// NULLs or only NaN is dropped here, so the merge leaves the field with no row
/// (unknown) instead of narrowing it to the new values. The commit's own
/// snapshot is the newest, so its new files are not counted.
async fn load_table_column_stats(
    tx: &mut sqlx::Transaction<'_, sqlx::MySql>,
    table_id: i64,
) -> Result<Vec<crate::stats_encode::GlobalColumnStat>> {
    let mut existing = Vec::new();
    for row in sqlx::query(
        "SELECT s.column_id, s.contains_null, s.contains_nan, s.min_value, s.max_value
         FROM ducklake_table_column_stats s
         WHERE s.table_id = ?
           AND NOT (s.min_value IS NULL AND s.max_value IS NULL
                    AND (s.contains_nan IS NULL OR NOT s.contains_nan)
                    AND s.extra_stats IS NULL AND EXISTS (
               SELECT 1 FROM ducklake_data_file d
               WHERE d.table_id = s.table_id AND d.end_snapshot IS NULL
                 AND d.begin_snapshot < (SELECT MAX(snapshot_id) FROM ducklake_snapshot)
                 AND NOT EXISTS (
                     SELECT 1 FROM ducklake_file_column_stats f
                     WHERE f.data_file_id = d.data_file_id AND f.column_id = s.column_id
                       AND ((f.null_count IS NOT NULL AND f.value_count = 0)
                            OR (f.contains_nan AND f.min_value IS NULL
                                AND f.max_value IS NULL)))))",
    )
    .bind(table_id)
    .fetch_all(&mut **tx)
    .await?
    {
        existing.push(crate::stats_encode::GlobalColumnStat {
            column_id: row.try_get(0)?,
            contains_null: row.try_get(1)?,
            contains_nan: row.try_get(2)?,
            min_value: row.try_get(3)?,
            max_value: row.try_get(4)?,
        });
    }
    Ok(existing)
}

/// Replace the table's `ducklake_table_column_stats` rows with `globals`.
async fn store_table_column_stats(
    tx: &mut sqlx::Transaction<'_, sqlx::MySql>,
    table_id: i64,
    globals: Vec<crate::stats_encode::GlobalColumnStat>,
) -> Result<()> {
    // A kept row keeps its `extra_stats` (a `GEOMETRY` column's bounds), which
    // this crate does not compute.
    let extra: std::collections::HashMap<i64, String> = sqlx::query_as(
        "SELECT column_id, extra_stats FROM ducklake_table_column_stats
         WHERE table_id = ? AND extra_stats IS NOT NULL",
    )
    .bind(table_id)
    .fetch_all(&mut **tx)
    .await?
    .into_iter()
    .collect();
    sqlx::query("DELETE FROM ducklake_table_column_stats WHERE table_id = ?")
        .bind(table_id)
        .execute(&mut **tx)
        .await?;
    for g in globals {
        sqlx::query(
            "INSERT INTO ducklake_table_column_stats
                 (table_id, column_id, contains_null, contains_nan, min_value, max_value, extra_stats)
             VALUES (?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(table_id)
        .bind(g.column_id)
        .bind(g.contains_null)
        .bind(g.contains_nan)
        .bind(g.min_value)
        .bind(g.max_value)
        .bind(extra.get(&g.column_id))
        .execute(&mut **tx)
        .await?;
    }
    Ok(())
}

/// Whether the table's statistics were initialized before this write, as
/// official keys an inlined write's statistics on an existing
/// `ducklake_table_stats` row. This crate seeds that row with zeros when it
/// creates a table, so a seeded row counts once it has recorded a row
/// (`next_row_id > 0`) or the table has column statistics.
async fn table_stats_initialized(
    tx: &mut sqlx::Transaction<'_, sqlx::MySql>,
    table_id: i64,
) -> Result<bool> {
    let initialized: i64 = sqlx::query_scalar(
        "SELECT EXISTS (SELECT 1 FROM ducklake_table_stats WHERE table_id = ? AND next_row_id > 0)
             OR EXISTS (SELECT 1 FROM ducklake_table_column_stats WHERE table_id = ?)",
    )
    .bind(table_id)
    .bind(table_id)
    .fetch_one(&mut **tx)
    .await?;
    Ok(initialized != 0)
}

/// Merge the statistics of `batches`, written inline to the table, into its
/// `ducklake_table_column_stats`, as official DuckLake does on an inlined write
/// (see [`crate::stats_encode::merge_inlined_column_stats`]). A NULL in one of
/// `snapshot_id_columns` is stored as `snapshot_id`.
#[expect(
    clippy::too_many_arguments,
    reason = "the statistics merge needs the table, its fields, the rows and the stats state"
)]
pub(crate) async fn merge_inlined_table_column_stats(
    tx: &mut sqlx::Transaction<'_, sqlx::MySql>,
    table_id: i64,
    columns: &[ColumnDef],
    column_ids: &[i64],
    batches: &[RecordBatch],
    snapshot_id_columns: &[String],
    snapshot_id: i64,
    initialized: bool,
) -> Result<()> {
    let fields = crate::metadata_writer::catalog_fields(columns, column_ids)?;
    let inlined = crate::stats_encode::inlined_column_stats(batches, &fields, |i| {
        columns
            .get(i)
            .filter(|column| snapshot_id_columns.iter().any(|name| name == column.name()))
            .map(|_| datafusion::common::ScalarValue::Int64(Some(snapshot_id)))
    })?;
    let existing = load_table_column_stats(tx, table_id).await?;
    let merged =
        crate::stats_encode::merge_inlined_column_stats(existing, inlined, initialized, |id| {
            crate::metadata_writer::stat_kind_of(&fields, id)
        });
    store_table_column_stats(tx, table_id, merged).await
}

async fn finalize_snapshot(
    tx: &mut sqlx::Transaction<'_, sqlx::MySql>,
    table_id: i64,
    columns: &[ColumnDef],
    column_ids: &[i64],
    mode: WriteMode,
    base_snapshot: i64,
) -> Result<i64> {
    let (snapshot_id, _) = insert_snapshot(tx).await?;
    finalize_table_snapshot(
        tx,
        snapshot_id,
        table_id,
        columns,
        column_ids,
        mode,
        base_snapshot,
        true,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn finalize_table_snapshot(
    tx: &mut sqlx::Transaction<'_, sqlx::MySql>,
    snapshot_id: i64,
    table_id: i64,
    columns: &[ColumnDef],
    column_ids: &[i64],
    mode: WriteMode,
    base_snapshot: i64,
    inserts: bool,
) -> Result<i64> {
    // Allocate the snapshot FIRST (carrying schema_version forward): this takes
    // the counter lock up front, serializing concurrent commits. schema_version is
    // corrected to a DDL bump below once we've classified the commit.
    let mut schema_version: i64 =
        sqlx::query_scalar("SELECT schema_version FROM ducklake_snapshot WHERE snapshot_id = ?")
            .bind(snapshot_id)
            .fetch_one(&mut **tx)
            .await?;

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
    if current.is_empty() {
        sqlx::query("UPDATE ducklake_schema AS s JOIN ducklake_table AS t ON t.schema_id = s.schema_id SET s.begin_snapshot = ? WHERE t.table_id = ? AND s.begin_snapshot = t.begin_snapshot").bind(snapshot_id).bind(table_id).execute(&mut **tx).await?;
        sqlx::query("UPDATE ducklake_table SET begin_snapshot = ? WHERE table_id = ?")
            .bind(snapshot_id)
            .bind(table_id)
            .execute(&mut **tx)
            .await?;
    }
    let is_ddl = current.is_empty()
        || catalog_columns_differ(
            &existing_catalog_columns,
            &existing_nullability,
            &proposed,
            column_ids,
        );
    if is_ddl {
        reject_new_reserved_columns(tx, table_id, &existing_catalog_columns, &proposed).await?;
    }
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
        detect_replace_conflict(tx, table_id, base_snapshot, ConflictScope::Table).await?;
        // Retire the prior data generation. A table gets its stats row with
        // its first rows, as official's `CREATE TABLE` writes none.
        retire_prior_generation(tx, table_id, snapshot_id).await?;
    } else if inserts {
        // An insert refuses a drop, alter or delete of the table since its
        // base snapshot, as official's does; a stage that only deletes is
        // checked by its own rules.
        check_table_conflicts(
            tx,
            table_id,
            base_snapshot,
            snapshot_id,
            TableChanges::append(),
        )
        .await?;
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
    if current.is_empty() && proposed.len() == write.column_ids.len() {
        return Ok(schema_id);
    }
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

async fn has_live_data(
    tx: &mut sqlx::Transaction<'_, sqlx::MySql>,
    table_id: i64,
) -> Result<(bool, bool)> {
    let has_files: bool = sqlx::query_scalar(
        "SELECT EXISTS(
             SELECT 1 FROM ducklake_data_file
             WHERE table_id = ? AND end_snapshot IS NULL
         )",
    )
    .bind(table_id)
    .fetch_one(&mut **tx)
    .await?;
    for (table_name, system) in inlined_tables(tx, table_id).await? {
        let has_rows: bool = sqlx::query_scalar(AssertSqlSafe(format!(
            "SELECT EXISTS(SELECT 1 FROM {} WHERE {} IS NULL)",
            quote_ident(&table_name),
            system.end_snapshot,
        )))
        .fetch_one(&mut **tx)
        .await?;
        if has_rows {
            return Ok((has_files, true));
        }
    }
    Ok((has_files, false))
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
    let mut total_advance = 0i64;
    let mut total_records = 0i64;
    let mut total_bytes = 0i64;
    for (file, data_file_id) in files.iter().zip(data_file_ids) {
        let row_ids = crate::metadata_writer::allocate_row_id_start(next_row_id, file);
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
        .bind(row_ids.stored)
        .bind(snapshot_id)
        .execute(&mut **tx)
        .await?;
        insert_file_column_stats(tx, write.table_id, data_file_id, &file.column_stats).await?;
        insert_partition_metadata(tx, write.table_id, data_file_id, file).await?;
        next_row_id += row_ids.advance;
        total_advance += row_ids.advance;
        total_records += file.record_count;
        total_bytes += file.file_size_bytes;
    }
    merge_file_table_column_stats(
        tx,
        write.table_id,
        &crate::metadata_writer::catalog_fields(&write.columns, &write.column_ids)?,
        &crate::metadata_writer::file_column_stats(files),
    )
    .await?;
    sqlx::query(
        "UPDATE ducklake_table_stats
         SET next_row_id = next_row_id + ?,
             record_count = record_count + ?,
             file_size_bytes = file_size_bytes + ?
         WHERE table_id = ?",
    )
    .bind(total_advance)
    .bind(total_records)
    .bind(total_bytes)
    .bind(write.table_id)
    .execute(&mut **tx)
    .await?;
    Ok(())
}

/// Build the DDL for a physical inline table whose system columns are named by
/// `system`. MySQL DDL implicitly commits the surrounding transaction, so callers
/// MUST execute this on the pool BEFORE the write transaction opens (CREATE TABLE
/// IF NOT EXISTS is idempotent, and an existing physical table without registry
/// rows is inert).
fn inlined_mysql_table_ddl(
    physical_name: &str,
    fields: &[(String, DataType, String)],
    system: InlinedSystemColumns,
) -> String {
    let mut ddl = format!(
        "CREATE TABLE IF NOT EXISTS {} ({} BIGINT, {} BIGINT, {} BIGINT",
        quote_ident(physical_name),
        system.row_id,
        system.begin_snapshot,
        system.end_snapshot,
    );
    for (name, data_type, _) in fields {
        ddl.push_str(", ");
        ddl.push_str(&quote_ident(name));
        ddl.push(' ');
        ddl.push_str(&inlined_mysql_type(data_type));
    }
    ddl.push_str(") ENGINE = InnoDB");
    ddl
}

/// `(name, Arrow type, DuckLake type)` of the columns rows of `batches` hold.
fn inlined_fields(
    batches: &[RecordBatch],
    columns: &[ColumnDef],
) -> Vec<(String, DataType, String)> {
    columns
        .iter()
        .zip(batches[0].schema().fields())
        .map(|(column, field)| {
            (
                column.name().to_string(),
                field.data_type().clone(),
                column.ducklake_type().to_string(),
            )
        })
        .collect()
}

/// The live column rows of `table_id`, each with its parent column id.
async fn live_table_columns(
    tx: &mut sqlx::Transaction<'_, sqlx::MySql>,
    table_id: i64,
) -> Result<Vec<(crate::metadata_provider::DuckLakeTableColumn, Option<i64>)>> {
    let rows = sqlx::query(
        "SELECT column_id, column_name, column_type, nulls_allowed, parent_column
         FROM ducklake_column
         WHERE table_id = ? AND end_snapshot IS NULL
         ORDER BY column_order",
    )
    .bind(table_id)
    .fetch_all(&mut **tx)
    .await?;
    let rows = rows
        .into_iter()
        .map(|row| {
            let nullable: Option<bool> = row.try_get(3)?;
            Ok((
                crate::metadata_provider::DuckLakeTableColumn::new(
                    row.try_get(0)?,
                    row.try_get(1)?,
                    row.try_get(2)?,
                    nullable.unwrap_or(true),
                ),
                row.try_get::<Option<i64>, _>(4)?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(rows)
}

/// Create the inlined data table `ducklake_inlined_data_{table_id}_{schema_version}`
/// holding `fields` after the system columns, as official's DDL does, and
/// register it in `tx`. Returns its name and system columns.
///
/// MySQL DDL commits implicitly, so the table and its declared indexes are
/// created on a separate connection and outlive a rollback of `tx`: a commit
/// that fails after this leaves the table behind, unregistered and empty (its
/// rows were inserted in `tx`). Readers find inlined tables through the
/// registry only, so such a table is never read. The schema version it was
/// created for is reused by the next commit that changes the table, which may
/// declare other columns; so a physical table of this name that no committed
/// registry row names is dropped here and created afresh. The counter lock
/// `tx` holds keeps the version stable, so no other commit is creating it.
async fn create_inlined_data_table(
    tx: &mut sqlx::Transaction<'_, sqlx::MySql>,
    pool: &MySqlPool,
    table_id: i64,
    schema_version: i64,
    fields: &[(String, DataType, String)],
) -> Result<(String, InlinedSystemColumns)> {
    let physical_name = format!("ducklake_inlined_data_{table_id}_{schema_version}");
    let mut ddl_connection = MySqlConnection::connect_with(&pool.connect_options()).await?;
    const REGISTERED: &str = "SELECT EXISTS(SELECT 1 FROM ducklake_inlined_data_tables
                                       WHERE table_id = ? AND schema_version = ?)";
    // Registered by a committed commit, or earlier in this one.
    let mut registered = false;
    for registry in [&mut ddl_connection, &mut **tx] {
        registered |= sqlx::query_scalar::<_, bool>(REGISTERED)
            .bind(table_id)
            .bind(schema_version)
            .fetch_one(registry)
            .await?;
    }
    if !registered {
        sqlx::query(AssertSqlSafe(format!(
            "DROP TABLE IF EXISTS {}",
            quote_ident(&physical_name)
        )))
        .execute(&mut ddl_connection)
        .await?;
    }
    let ddl = inlined_mysql_table_ddl(
        &physical_name,
        fields,
        catalog_inlined_system_columns(&mut ddl_connection).await?,
    );
    sqlx::query(AssertSqlSafe(ddl))
        .execute(&mut ddl_connection)
        .await?;
    let system = inlined_table_layout(&mut ddl_connection, &physical_name)
        .await?
        .system;
    create_declared_indexes(tx, &mut ddl_connection, table_id, &physical_name, fields).await?;
    sqlx::query(
        "INSERT INTO ducklake_inlined_data_tables (table_id, table_name, schema_version)
         SELECT ?, ?, ? FROM DUAL WHERE NOT EXISTS (
             SELECT 1 FROM ducklake_inlined_data_tables
             WHERE table_id = ? AND schema_version = ?
         )",
    )
    .bind(table_id)
    .bind(&physical_name)
    .bind(schema_version)
    .bind(table_id)
    .bind(schema_version)
    .execute(&mut **tx)
    .await?;
    Ok((physical_name, system))
}

/// Create on `ddl_connection` the indexes declared for `table_id` over the
/// columns of `fields` of the inlined data table `physical_name`.
async fn create_declared_indexes(
    tx: &mut sqlx::Transaction<'_, sqlx::MySql>,
    ddl_connection: &mut MySqlConnection,
    table_id: i64,
    physical_name: &str,
    fields: &[(String, DataType, String)],
) -> Result<()> {
    let value: Option<String> = sqlx::query_scalar(
        "SELECT value FROM ducklake_metadata WHERE `key` = ? AND scope = 'table' AND scope_id = ?",
    )
    .bind(INLINED_INDEX_COLUMNS_SETTING)
    .bind(table_id)
    .fetch_optional(&mut **tx)
    .await?;
    let declared = parse_inlined_index_columns(value.as_deref().unwrap_or_default())?;
    let available = fields.iter().map(|(name, _, _)| name.clone()).collect();
    let declared = live_inlined_index_columns(declared, &available);
    create_mysql_physical_indexes(ddl_connection, physical_name, &declared).await
}

/// The `data_inlining_row_limit` in effect for the live table `table_id`;
/// `0` when it is not live.
async fn table_inlining_limit(
    tx: &mut sqlx::Transaction<'_, sqlx::MySql>,
    table_id: i64,
) -> Result<usize> {
    let Some(schema_id): Option<i64> = sqlx::query_scalar(
        "SELECT schema_id FROM ducklake_table WHERE table_id = ? AND end_snapshot IS NULL",
    )
    .bind(table_id)
    .fetch_optional(&mut **tx)
    .await?
    else {
        return Ok(0);
    };
    let settings: Vec<(Option<String>, Option<i64>, String)> = sqlx::query_as(
        "SELECT scope, scope_id, value FROM ducklake_metadata
         WHERE `key` = 'data_inlining_row_limit'",
    )
    .fetch_all(&mut **tx)
    .await?;
    crate::metadata_writer::effective_inlining_limit(&settings, schema_id, table_id)
}

/// Refuse a commit that creates `table_id` or adds a column to it named like
/// an inlined data table's system column while its inlining limit is above
/// `0`, as official refuses `CREATE TABLE` and `ADD COLUMN`.
async fn reject_new_reserved_columns(
    tx: &mut sqlx::Transaction<'_, sqlx::MySql>,
    table_id: i64,
    existing: &[crate::metadata_writer::ExistingCatalogColumn],
    proposed: &[crate::metadata_writer::CatalogColumnDef],
) -> Result<()> {
    let system = catalog_inlined_system_columns(&mut **tx).await?;
    let Some(column) = crate::metadata_writer::new_reserved_column(existing, proposed, system)
    else {
        return Ok(());
    };
    if table_inlining_limit(tx, table_id).await? == 0 {
        return Ok(());
    }
    Err(crate::metadata_writer::reserved_inlining_column_error(
        &column,
    ))
}

/// Official's eager inlined data table: a commit that creates `table_id` or
/// changes its columns or partitioning creates the inlined data table of its
/// `schema_version` while the table's `data_inlining_row_limit` is above `0`
/// and its columns can be inlined (`WriteNewInlinedTables`).
async fn create_eager_inlined_table(
    tx: &mut sqlx::Transaction<'_, sqlx::MySql>,
    pool: &MySqlPool,
    table_id: i64,
    schema_version: i64,
) -> Result<Option<String>> {
    if table_inlining_limit(tx, table_id).await? == 0 {
        return Ok(None);
    }
    let columns = live_table_columns(tx, table_id).await?;
    let system = catalog_inlined_system_columns(&mut **tx).await?;
    if let Some(fields) =
        crate::metadata_writer::inlinable_columns(columns, system, None, mysql_type_inlines)?
    {
        let (name, _) =
            create_inlined_data_table(tx, pool, table_id, schema_version, &fields).await?;
        return Ok(Some(name));
    }
    Ok(None)
}

/// [`create_eager_inlined_table`] for every table a DDL of the commit at
/// `snapshot_id` recorded a schema version for. MySQL reads a table created
/// after a transaction's snapshot only to fail it, so a commit runs this once
/// its own reads of inlined tables are done. Returns the tables it created.
async fn create_eager_inlined_tables(
    tx: &mut sqlx::Transaction<'_, sqlx::MySql>,
    pool: &MySqlPool,
    snapshot_id: i64,
) -> Result<Vec<String>> {
    let versions: Vec<(i64, i64)> = sqlx::query_as(
        "SELECT table_id, schema_version FROM ducklake_schema_versions
         WHERE begin_snapshot = ? ORDER BY table_id",
    )
    .bind(snapshot_id)
    .fetch_all(&mut **tx)
    .await?;
    let mut created = Vec::new();
    for (table_id, schema_version) in versions {
        if let Some(name) = create_eager_inlined_table(tx, pool, table_id, schema_version).await? {
            created.push(name);
        }
    }
    Ok(created)
}

/// The inlined data table rows of `table_id` written at `snapshot_id` go to:
/// its latest, as official's `LatestInlinedTableQuery` picks it, or, when it
/// has none, a new one for a schema version the commit takes for it, as
/// official's `WriteNewInlinedData` creates one. Official inserts into the
/// latest table whatever its columns; when they are not `fields` the insert
/// fails as official's does, unless this commit itself changed the table's
/// columns (a schema-changing append), which gets a new table.
async fn inlined_insert_table(
    tx: &mut sqlx::Transaction<'_, sqlx::MySql>,
    pool: &MySqlPool,
    table_id: i64,
    snapshot_id: i64,
    fields: &[(String, DataType, String)],
) -> Result<(String, InlinedSystemColumns)> {
    let latest: Option<String> = sqlx::query_scalar(
        "SELECT table_name FROM ducklake_inlined_data_tables
         WHERE table_id = ? ORDER BY schema_version DESC LIMIT 1",
    )
    .bind(table_id)
    .fetch_optional(&mut **tx)
    .await?;
    if let Some(name) = latest {
        // The transaction's snapshot may predate the table's DDL, so its
        // layout is read on a connection of its own.
        let mut connection = MySqlConnection::connect_with(&pool.connect_options()).await?;
        let present: Vec<String> = sqlx::query_scalar(
            "SELECT column_name FROM information_schema.columns
             WHERE table_schema = DATABASE() AND table_name = ?
             ORDER BY ordinal_position",
        )
        .bind(&name)
        .fetch_all(&mut connection)
        .await?;
        let matches = present.len() == fields.len() + 3
            && present[3..]
                .iter()
                .zip(fields)
                .all(|(stored, (name, _, _))| stored == name);
        if matches {
            let system = inlined_table_layout(&mut connection, &name).await?.system;
            create_declared_indexes(tx, &mut connection, table_id, &name, fields).await?;
            return Ok((name, system));
        }
        let changed_now: bool = sqlx::query_scalar(
            "SELECT EXISTS(SELECT 1 FROM ducklake_column
                           WHERE table_id = ? AND (begin_snapshot = ? OR end_snapshot = ?))",
        )
        .bind(table_id)
        .bind(snapshot_id)
        .bind(snapshot_id)
        .fetch_one(&mut **tx)
        .await?;
        if !changed_now {
            return Err(crate::metadata_writer::inlined_table_shape_error(
                &name,
                present.len(),
                fields.len() + 3,
            ));
        }
    }
    sqlx::query(
        "UPDATE ducklake_snapshot SET schema_version = schema_version + 1 WHERE snapshot_id = ?",
    )
    .bind(snapshot_id)
    .execute(&mut **tx)
    .await?;
    let schema_version: i64 =
        sqlx::query_scalar("SELECT schema_version FROM ducklake_snapshot WHERE snapshot_id = ?")
            .bind(snapshot_id)
            .fetch_one(&mut **tx)
            .await?;
    create_inlined_data_table(tx, pool, table_id, schema_version, fields).await
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
    let mut connection = pool.acquire().await?;
    create_mysql_physical_indexes(&mut connection, physical_name, &declared).await
}

async fn create_mysql_physical_indexes(
    connection: &mut MySqlConnection,
    physical_name: &str,
    declared: &[String],
) -> Result<()> {
    let physical_types = sqlx::query_as::<_, (String, String)>(
        "SELECT column_name, column_type FROM information_schema.columns
         WHERE table_schema = DATABASE() AND table_name = ?",
    )
    .bind(physical_name)
    .fetch_all(&mut *connection)
    .await?
    .into_iter()
    .collect::<std::collections::HashMap<_, _>>();
    for column in declared.iter().map(String::as_str) {
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
        .fetch_one(&mut *connection)
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
        .execute(&mut *connection)
        .await;
        if let Err(error) = create {
            let exists: i64 = sqlx::query_scalar(
                "SELECT COUNT(*) FROM information_schema.statistics
                 WHERE table_schema = DATABASE() AND table_name = ? AND index_name = ?",
            )
            .bind(physical_name)
            .bind(&index_name)
            .fetch_one(&mut *connection)
            .await?;
            if exists == 0 {
                return Err(error.into());
            }
        }
    }
    Ok(())
}

/// `system` names the system columns of the inlined table the rows go to.
async fn commit_staged_inline(
    tx: &mut sqlx::Transaction<'_, sqlx::MySql>,
    snapshot_id: i64,
    write: &StagedTableWrite,
    batches: &[RecordBatch],
    physical_name: &str,
    system: InlinedSystemColumns,
) -> Result<()> {
    enforce_inline_partition_fence(tx, write.table_id, write.base_snapshot_id).await?;
    let record_count: usize = batches.iter().map(RecordBatch::num_rows).sum();
    if record_count == 0 {
        return Err(crate::DuckLakeError::InvalidConfig(
            "multi-table inline stage requires at least one row".to_string(),
        ));
    }
    let stats_initialized = table_stats_initialized(tx, write.table_id).await?;
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
    let mut written = 0usize;
    for batch in batches {
        for batch_row in 0..batch.num_rows() {
            let mut query = QueryBuilder::<MySql>::new(format!(
                "INSERT INTO {} ({}, {}, {}, {}) VALUES (",
                quote_ident(physical_name),
                system.row_id,
                system.begin_snapshot,
                system.end_snapshot,
                column_list
            ));
            let stored_row_id = match &write.inlined_row_ids {
                Some(row_ids) => row_ids[written],
                None => row_id,
            };
            query.push_bind(stored_row_id);
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
            written += 1;
        }
    }
    merge_inlined_table_column_stats(
        tx,
        write.table_id,
        &write.columns,
        &write.column_ids,
        batches,
        &write.snapshot_id_columns,
        snapshot_id,
        stats_initialized,
    )
    .await?;
    let record_count = i64::try_from(record_count).map_err(|_| {
        crate::DuckLakeError::InvalidConfig("multi-table inline row count exceeds i64".to_string())
    })?;
    // Preserved row ids were drawn when the rows were first written.
    let advance = if write.inlined_row_ids.is_some() {
        0
    } else {
        record_count
    };
    sqlx::query(
        "UPDATE ducklake_table_stats
         SET next_row_id = next_row_id + ?, record_count = record_count + ?
         WHERE table_id = ?",
    )
    .bind(advance)
    .bind(record_count)
    .bind(write.table_id)
    .execute(&mut **tx)
    .await?;
    Ok(())
}

/// Create `ducklake_inlined_delete_<table_id>` as official DuckLake does. MySQL
/// commits DDL implicitly, so this runs on its own connection before the
/// metadata transaction begins.
async fn create_inlined_delete_table(pool: &MySqlPool, table_id: i64) -> Result<()> {
    let table = crate::metadata_provider::inlined_delete_table_name(table_id)?;
    sqlx::query(AssertSqlSafe(format!(
        "CREATE TABLE IF NOT EXISTS {} (file_id BIGINT, row_id BIGINT, begin_snapshot BIGINT)",
        quote_ident(&table)
    )))
    .execute(pool)
    .await?;
    Ok(())
}

/// Record `deletes` in `ducklake_inlined_delete_<table_id>`, which
/// [`create_inlined_delete_table`] created. Aborts when a target data file is
/// no longer live; the caller has checked the table-level conflicts
/// ([`table_conflict`]).
async fn apply_inlined_file_deletes(
    tx: &mut sqlx::Transaction<'_, sqlx::MySql>,
    table_id: i64,
    snapshot_id: i64,
    base_snapshot: i64,
    deletes: &[InlinedFileDelete],
) -> Result<()> {
    if deletes.is_empty() {
        return Ok(());
    }
    for delete in deletes {
        let live: Option<i64> = sqlx::query_scalar(
            "SELECT data_file_id FROM ducklake_data_file
             WHERE data_file_id = ? AND table_id = ? AND end_snapshot IS NULL",
        )
        .bind(delete.data_file_id)
        .bind(table_id)
        .fetch_optional(&mut **tx)
        .await?;
        if live.is_none() {
            return Err(crate::DuckLakeError::Conflict(format!(
                "data file {} is no longer live since snapshot {base_snapshot}; re-open the \
                 catalog and retry the statement",
                delete.data_file_id
            )));
        }
        let delete_file: Option<i64> = sqlx::query_scalar(
            "SELECT delete_file_id FROM ducklake_delete_file
             WHERE data_file_id = ? AND end_snapshot IS NULL",
        )
        .bind(delete.data_file_id)
        .fetch_optional(&mut **tx)
        .await?;
        if delete_file != delete.expected_delete_file {
            return Err(crate::metadata_writer::changed_delete_file_conflict(
                delete.data_file_id,
                base_snapshot,
            ));
        }
    }
    let table = crate::metadata_provider::inlined_delete_table_name(table_id)?;
    let rows = deletes
        .iter()
        .flat_map(|delete| {
            delete
                .positions
                .iter()
                .map(move |position| (delete.data_file_id, *position))
        })
        .collect::<Vec<_>>();
    for chunk in rows.chunks(1000) {
        let mut query = QueryBuilder::<MySql>::new(format!(
            "INSERT INTO {} (file_id, row_id, begin_snapshot) ",
            quote_ident(&table)
        ));
        query.push_values(chunk, |mut row, (file_id, position)| {
            row.push_bind(*file_id)
                .push_bind(*position)
                .push_bind(snapshot_id);
        });
        query.build().execute(&mut **tx).await?;
    }
    Ok(())
}

/// End each of `files` at `snapshot_id`, leaving its live delete file as it
/// is, as official's `DropDataFiles` does. Aborts when a file is no longer
/// live or its live delete file changed.
async fn drop_data_files(
    tx: &mut sqlx::Transaction<'_, sqlx::MySql>,
    table_id: i64,
    snapshot_id: i64,
    base_snapshot: i64,
    files: &[DroppedDataFile],
) -> Result<()> {
    for file in files {
        let current: Option<i64> = sqlx::query_scalar(
            "SELECT delete_file_id FROM ducklake_delete_file
             WHERE data_file_id = ? AND end_snapshot IS NULL",
        )
        .bind(file.data_file_id)
        .fetch_optional(&mut **tx)
        .await?;
        let ended = sqlx::query(
            "UPDATE ducklake_data_file SET end_snapshot = ?
             WHERE data_file_id = ? AND table_id = ? AND end_snapshot IS NULL",
        )
        .bind(snapshot_id)
        .bind(file.data_file_id)
        .bind(table_id)
        .execute(&mut **tx)
        .await?
        .rows_affected();
        if ended != 1 || current != file.expected_delete_file {
            return Err(crate::DuckLakeError::Conflict(format!(
                "data file {} or its delete file changed since snapshot {base_snapshot}; \
                 re-open the catalog and retry the statement",
                file.data_file_id
            )));
        }
    }
    Ok(())
}

async fn apply_staged_inlined_deletes(
    tx: &mut sqlx::Transaction<'_, sqlx::MySql>,
    snapshot_id: i64,
    write: &StagedTableWrite,
) -> Result<()> {
    apply_inlined_deletes(
        tx,
        write.table_id,
        snapshot_id,
        write.base_snapshot_id,
        &write.inlined_deletes,
    )
    .await
}

impl MetadataWriter for MySqlMetadataWriter {
    fn maintenance(&self) -> Option<&dyn crate::maintenance::CatalogMaintenance> {
        Some(self)
    }

    fn supports_update(&self) -> bool {
        true
    }

    fn supports_inlined_row_mutations(&self) -> bool {
        true
    }

    fn drop_empty_superseded_inlined_tables(&self) -> Result<()> {
        block_on(async {
            // A catalog this crate's writer created before data inlining has no
            // registry, and so nothing to drop.
            if !self.has_inlined_data_registry().await? {
                return Ok(());
            }
            let targets = sqlx::query_as::<_, (i64, i64, String)>(SUPERSEDED_INLINED_TABLES_SQL)
                .fetch_all(&self.pool)
                .await?;
            for (table_id, schema_version, table_name) in targets {
                let quoted = quote_ident(&table_name);
                // The registry row goes first, in its own committed transaction, and the
                // DROP after it: MySQL commits DDL implicitly, and a registry row naming a
                // dropped table would fail every read of its table. An interruption
                // between the two leaves an empty table no reader lists — official's
                // order and its window on MySQL.
                let mut tx = self.pool.begin().await?;
                let rows: i64 =
                    sqlx::query_scalar(AssertSqlSafe(format!("SELECT COUNT(*) FROM {quoted}")))
                        .fetch_one(&mut *tx)
                        .await?;
                if rows != 0 {
                    tx.commit().await?;
                    continue;
                }
                sqlx::query(
                    "DELETE FROM ducklake_inlined_data_tables
                     WHERE table_id = ? AND schema_version = ?",
                )
                .bind(table_id)
                .bind(schema_version)
                .execute(&mut *tx)
                .await?;
                tx.commit().await?;
                sqlx::query(AssertSqlSafe(format!("DROP TABLE IF EXISTS {quoted}")))
                    .execute(&self.pool)
                    .await?;
            }
            Ok(())
        })
    }

    fn set_table_setting(&self, table_id: i64, key: &str, value: &str) -> Result<()> {
        let key = validate_table_setting(key)?;
        block_on(async {
            let mut tx = self.pool.begin().await?;
            let live: Option<i64> = sqlx::query_scalar("SELECT table_id FROM ducklake_table WHERE table_id = ? AND end_snapshot IS NULL FOR UPDATE").bind(table_id).fetch_optional(&mut *tx).await?;
            if live.is_none() {
                return Err(crate::DuckLakeError::TableNotFound(table_id.to_string()));
            }
            if enables_inlining(&key, value) {
                let columns: Vec<(String, String)> = sqlx::query_as(SQL_TABLE_INLINING_COLUMNS)
                    .bind(table_id)
                    .fetch_all(&mut *tx)
                    .await?;
                let system =
                    crate::metadata_provider_mysql::catalog_inlined_system_columns(&mut *tx)
                        .await?;
                reject_reserved_inlining_columns(&columns, system)?;
            }
            sqlx::query("DELETE FROM ducklake_metadata WHERE `key` = ? AND scope = 'table' AND scope_id = ?").bind(&key).bind(table_id).execute(&mut *tx).await?;
            sqlx::query("INSERT INTO ducklake_metadata (`key`, value, scope, scope_id) VALUES (?, ?, 'table', ?)").bind(&key).bind(value).bind(table_id).execute(&mut *tx).await?;
            tx.commit().await?;
            Ok(())
        })
    }

    fn with_commit_lock(
        &self,
        identity: &str,
        operation: Box<dyn FnOnce() -> Result<()> + '_>,
    ) -> Result<()> {
        block_on(async {
            let mut connection = self.pool.acquire().await?;
            // Named locks outlive transactions; close also releases the lock after a panic or failed unlock.
            connection.close_on_drop();
            let acquired: Option<i64> =
                sqlx::query_scalar("SELECT GET_LOCK(SHA2(CONCAT(DATABASE(), ':', ?), 256), 30)")
                    .bind(identity)
                    .fetch_one(&mut *connection)
                    .await?;
            if acquired != Some(1) {
                return Err(crate::DuckLakeError::Conflict(
                    "commit coordination lock was not acquired".to_string(),
                ));
            }
            let result = operation();
            let unlock = sqlx::query_scalar::<_, Option<i64>>(
                "SELECT RELEASE_LOCK(SHA2(CONCAT(DATABASE(), ':', ?), 256))",
            )
            .bind(identity)
            .fetch_one(&mut *connection)
            .await;
            match (result, unlock) {
                (Err(e), _) => Err(e),
                (Ok(()), Ok(Some(1))) => Ok(()),
                (Ok(()), Ok(_)) => Err(crate::DuckLakeError::Internal(
                    "commit coordination lock was not released".to_string(),
                )),
                (Ok(()), Err(e)) => Err(e.into()),
            }
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

    fn catalog_encrypted(&self) -> Result<bool> {
        block_on(async {
            let values: Vec<String> = sqlx::query_scalar(
                "SELECT `value` FROM ducklake_metadata \
                 WHERE `key` = 'encrypted' AND scope IS NULL",
            )
            .fetch_all(&self.pool)
            .await?;
            crate::metadata_writer::encrypted_option(&values)
        })
    }

    fn head_snapshot(&self) -> Result<Option<i64>> {
        block_on(async {
            Ok(
                sqlx::query_scalar("SELECT MAX(snapshot_id) FROM ducklake_snapshot")
                    .fetch_one(&self.pool)
                    .await?,
            )
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

            let schema_path = directory_path(path.unwrap_or(name));
            // No RETURNING: read the new auto-increment id via last_insert_id().
            let result = sqlx::query(
                "INSERT INTO ducklake_schema (schema_name, path, path_is_relative, begin_snapshot)
                 VALUES (?, ?, 1, ?)",
            )
            .bind(name)
            .bind(&schema_path)
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

            let table_path = directory_path(path.unwrap_or(name));
            let result = sqlx::query(
                "INSERT INTO ducklake_table (schema_id, table_name, path, path_is_relative, begin_snapshot)
                 VALUES (?, ?, ?, 1, ?)",
            )
            .bind(schema_id)
            .bind(name)
            .bind(&table_path)
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
            // The new version keeps everything but the type: defaults
            // included, so rows written before the column was added still
            // read its default.
            let row = sqlx::query(
                "SELECT column_id, column_type, column_order, nulls_allowed, initial_default,
                        default_value, default_value_type, default_value_dialect
                 FROM ducklake_column
                 WHERE table_id = ? AND column_name = ? AND end_snapshot IS NULL
                   AND parent_column IS NULL",
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
            let initial_default: Option<String> = row.try_get("initial_default")?;
            let default_value: Option<String> = row.try_get("default_value")?;
            let default_value_type: Option<String> = row.try_get("default_value_type")?;
            let default_value_dialect: Option<String> = row.try_get("default_value_dialect")?;
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
                      column_name, column_type, nulls_allowed, initial_default, default_value,
                      default_value_type, default_value_dialect)
                 VALUES (?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            )
            .bind(column_id)
            .bind(snapshot_id)
            .bind(table_id)
            .bind(column_order)
            .bind(column_name)
            .bind(new_ducklake_type)
            .bind(nulls_allowed)
            .bind(initial_default)
            .bind(default_value)
            .bind(default_value_type)
            .bind(default_value_dialect)
            .execute(&mut *tx)
            .await?;
            record_schema_version(&mut tx, snapshot_id, schema_version, table_id).await?;
            create_eager_inlined_table(&mut tx, &self.pool, table_id, schema_version).await?;
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

            let next_row_id: i64 =
                sqlx::query("SELECT next_row_id FROM ducklake_table_stats WHERE table_id = ?")
                    .bind(table_id)
                    .fetch_one(&mut *tx)
                    .await?
                    .try_get(0)?;
            let row_ids = crate::metadata_writer::allocate_row_id_start(next_row_id, file);

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
            .bind(row_ids.stored)
            .bind(snapshot_id)
            .execute(&mut *tx)
            .await?;

            // Persist the file's zone maps + refresh the roll-up.
            insert_file_column_stats(&mut tx, table_id, data_file_id, &file.column_stats).await?;
            insert_partition_metadata(&mut tx, table_id, data_file_id, file).await?;
            merge_file_table_column_stats(
                &mut tx,
                table_id,
                &crate::metadata_writer::catalog_fields(columns, column_ids)?,
                &crate::metadata_writer::file_column_stats([file]),
            )
            .await?;

            // Advance the counter and accumulate stats. `next_row_id`
            // monotonically increases over the table's lifetime.
            sqlx::query(
                "UPDATE ducklake_table_stats
                 SET next_row_id     = next_row_id + ?,
                     record_count    = record_count + ?,
                     file_size_bytes = file_size_bytes + ?
                 WHERE table_id = ?",
            )
            .bind(row_ids.advance)
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

            create_eager_inlined_tables(&mut tx, &self.pool, snapshot_id).await?;
            crate::metadata_writer::commit_networked(tx).await?;
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
            let mut total_advance: i64 = 0;
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
                let row_ids = crate::metadata_writer::allocate_row_id_start(next_row_id, file);
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
                .bind(row_ids.stored)
                .bind(snapshot_id)
                .execute(&mut *tx)
                .await?;
                insert_file_column_stats(&mut tx, table_id, data_file_id, &file.column_stats)
                    .await?;
                insert_partition_metadata(&mut tx, table_id, data_file_id, file).await?;
                next_row_id += row_ids.advance;
                total_advance += row_ids.advance;
                total_records += file.record_count;
                total_bytes += file.file_size_bytes;
            }
            merge_file_table_column_stats(
                &mut tx,
                table_id,
                &crate::metadata_writer::catalog_fields(columns, column_ids)?,
                &crate::metadata_writer::file_column_stats(files),
            )
            .await?;
            sqlx::query(
                "UPDATE ducklake_table_stats
                 SET next_row_id     = next_row_id + ?,
                     record_count    = record_count + ?,
                     file_size_bytes = file_size_bytes + ?
                 WHERE table_id = ?",
            )
            .bind(total_advance)
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
            create_eager_inlined_tables(&mut tx, &self.pool, snapshot_id).await?;
            crate::metadata_writer::commit_networked(tx).await?;
            Ok(CommitIds {
                snapshot_id,
                schema_id,
                table_id,
            })
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn reserved_inlining_column(&self, columns: &[String]) -> Result<Option<String>> {
        let system = block_on(catalog_inlined_system_columns(&self.pool))?;
        Ok(columns
            .iter()
            .find(|column| system.conflicts_with(column))
            .cloned())
    }

    fn supports_data_inlining(&self, schema: &arrow::datatypes::Schema) -> bool {
        let Ok(system) = block_on(catalog_inlined_system_columns(&self.pool)) else {
            return false;
        };
        if schema
            .fields()
            .iter()
            .any(|field| system.conflicts_with(field.name()))
        {
            return false;
        }
        schema
            .fields()
            .iter()
            .all(|field| mysql_type_inlines(field.data_type()))
    }

    fn supports_data_inlining_values(&self, batches: &[RecordBatch]) -> bool {
        batches
            .iter()
            .flat_map(RecordBatch::columns)
            .all(|array| mysql_values_inline(array.as_ref()))
    }

    fn table_types_support_inlining(&self, schema_name: &str, table_name: &str) -> Result<bool> {
        block_on(async {
            let refused: i64 = sqlx::query_scalar(AssertSqlSafe(format!(
                "SELECT EXISTS(
                     SELECT 1 FROM ducklake_column c
                     JOIN ducklake_table t ON t.table_id = c.table_id
                     JOIN ducklake_schema s ON s.schema_id = t.schema_id
                     WHERE s.schema_name = ? AND s.end_snapshot IS NULL
                       AND t.table_name = ? AND t.end_snapshot IS NULL
                       AND c.end_snapshot IS NULL AND LOWER(c.column_type) IN ({}))",
                crate::types::sql_uninlinable_types()
            )))
            .bind(schema_name)
            .bind(table_name)
            .fetch_one(&self.pool)
            .await?;
            Ok(refused == 0)
        })
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
            // Official's table-level check: a drop, alter, compaction or insert
            // of the table since `base_snapshot` refuses the delete.
            let ours = TableChanges {
                deleted_from: true,
                ..TableChanges::default()
            };
            check_table_conflicts(&mut tx, table_id, base_snapshot, snapshot_id, ours).await?;
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
        snapshot_id: i64,
        file: &DataFileInfo,
        deletes: &[DeleteFileEntry],
        mode: WriteMode,
        base_snapshot: i64,
        columns: &[ColumnDef],
        column_ids: &[i64],
    ) -> Result<CommitIds> {
        self.register_data_files_with_deletes(
            table_id,
            schema_name,
            table_name,
            snapshot_id,
            std::slice::from_ref(file),
            deletes,
            mode,
            base_snapshot,
            columns,
            column_ids,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn register_data_files_with_deletes(
        &self,
        table_id: i64,
        schema_name: &str,
        table_name: &str,
        _snapshot_id: i64,
        files: &[DataFileInfo],
        deletes: &[DeleteFileEntry],
        mode: WriteMode,
        base_snapshot: i64,
        columns: &[ColumnDef],
        column_ids: &[i64],
    ) -> Result<CommitIds> {
        if files.is_empty() {
            return Err(crate::DuckLakeError::InvalidConfig(
                "register_data_files_with_deletes: files must be non-empty".to_string(),
            ));
        }
        validate_delete_entries(mode, deletes)?;
        block_on(async {
            let mut tx = self.pool.begin().await?;
            let snapshot_id =
                finalize_snapshot(&mut tx, table_id, columns, column_ids, mode, base_snapshot)
                    .await?;
            if !deletes.is_empty() {
                let ours = TableChanges {
                    inserted_into: true,
                    deleted_from: true,
                    ..TableChanges::default()
                };
                let source = deleted_files_table(&mut tx, table_id, deletes).await?;
                check_table_conflicts(&mut tx, source, base_snapshot, snapshot_id, ours).await?;
            }
            let sources = SourceFile::of_entries(deletes, false);
            detect_new_inlined_deletes(&mut tx, table_id, base_snapshot, &sources).await?;
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
            // Each file draws its own row ids from the advancing counter.
            for file in files {
                let next_row_id: i64 = sqlx::query_scalar(
                    "SELECT next_row_id FROM ducklake_table_stats WHERE table_id = ?",
                )
                .bind(table_id)
                .fetch_one(&mut *tx)
                .await?;
                let row_ids = crate::metadata_writer::allocate_row_id_start(next_row_id, file);
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
                .bind(row_ids.stored)
                .bind(snapshot_id)
                .execute(&mut *tx)
                .await?;
                insert_file_column_stats(&mut tx, table_id, data_file_id, &file.column_stats)
                    .await?;
                insert_partition_metadata(&mut tx, table_id, data_file_id, file).await?;
                sqlx::query(
                    "UPDATE ducklake_table_stats
                     SET next_row_id = next_row_id + ?, record_count = record_count + ?,
                         file_size_bytes = file_size_bytes + ? WHERE table_id = ?",
                )
                .bind(row_ids.advance)
                .bind(file.record_count)
                .bind(file.file_size_bytes)
                .bind(table_id)
                .execute(&mut *tx)
                .await?;
            }
            merge_file_table_column_stats(
                &mut tx,
                table_id,
                &crate::metadata_writer::catalog_fields(columns, column_ids)?,
                &crate::metadata_writer::file_column_stats(files),
            )
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
                false,
                &SnapshotCommitMetadata::default(),
            )
            .await?;
            let schema_id: i64 =
                sqlx::query_scalar("SELECT schema_id FROM ducklake_table WHERE table_id = ?")
                    .bind(table_id)
                    .fetch_one(&mut *tx)
                    .await?;
            create_eager_inlined_tables(&mut tx, &self.pool, snapshot_id).await?;
            crate::metadata_writer::commit_networked(tx).await?;
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
            let ours = TableChanges {
                deleted_from: true,
                ..TableChanges::default()
            };
            check_table_conflicts(&mut tx, table_id, base_snapshot, snapshot_id, ours).await?;
            let sources = SourceFile::of_entries(deletes, true);
            detect_new_inlined_deletes(&mut tx, table_id, base_snapshot, &sources).await?;
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
            let ours = TableChanges {
                deleted_from: !positional.is_empty(),
                deleted_inlined: !inlined.is_empty(),
                ..TableChanges::default()
            };
            check_table_conflicts(&mut tx, table_id, base_snapshot, snapshot_id, ours).await?;
            let sources = SourceFile::of_entries(positional, true);
            detect_new_inlined_deletes(&mut tx, table_id, base_snapshot, &sources).await?;
            for entry in positional {
                apply_delete_entry(&mut tx, table_id, base_snapshot, snapshot_id, entry).await?;
            }
            apply_inlined_deletes(&mut tx, table_id, snapshot_id, base_snapshot, inlined).await?;
            record_snapshot_changes(
                &mut tx,
                snapshot_id,
                &crate::metadata_writer::table_storage_changes(
                    table_id,
                    None,
                    ours.deleted_from,
                    ours.deleted_inlined,
                ),
                &SnapshotCommitMetadata::default(),
            )
            .await?;
            let schema_id: i64 =
                sqlx::query_scalar("SELECT schema_id FROM ducklake_table WHERE table_id = ?")
                    .bind(table_id)
                    .fetch_one(&mut *tx)
                    .await?;
            crate::metadata_writer::commit_networked(tx).await?;
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
            let source_files = sources
                .iter()
                .map(|source| SourceFile::any(source.data_file_id))
                .collect::<Vec<_>>();
            detect_new_inlined_deletes(&mut tx, table_id, base_snapshot, &source_files).await?;
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
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                )
                .bind(data_file_id)
                .bind(table_id)
                .bind(&output.file.path)
                .bind(output.file.path_is_relative)
                .bind(output.file.file_size_bytes)
                .bind(output.file.footer_size)
                .bind(output.file.record_count)
                .bind(crate::metadata_writer::carried_row_id_start(&output.file))
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
            // `ducklake_table_stats` is left as it is, as official's compaction
            // leaves it: `record_count` stays the gross count of rows ever inserted.
            record_snapshot_changes(
                &mut tx,
                snapshot_id,
                &crate::metadata_writer::compaction_change(retirement, table_id),
                &SnapshotCommitMetadata::new(),
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

    fn commit_inlined_flush(
        &self,
        base_snapshot: i64,
        tables: &[InlinedFlushTable],
    ) -> Result<i64> {
        validate_inlined_flush(tables)?;
        block_on(async {
            let mut tx = self.pool.begin().await?;
            // A flush is not DDL: schema_version carries forward.
            let (snapshot_id, _schema_version) = insert_snapshot(&mut tx).await?;
            let changes: Vec<Option<String>> = sqlx::query_scalar(
                "SELECT changes_made FROM ducklake_snapshot_changes
                 WHERE snapshot_id > ? AND snapshot_id < ?",
            )
            .bind(base_snapshot)
            .bind(snapshot_id)
            .fetch_all(&mut *tx)
            .await?;
            if let Some(conflict) =
                crate::metadata_writer::inlined_flush_table_conflict(tables, &changes)
            {
                return Err(conflict);
            }
            for table in tables {
                let live: Option<i64> = sqlx::query_scalar(
                    "SELECT 1 FROM ducklake_table WHERE table_id = ? AND end_snapshot IS NULL",
                )
                .bind(table.table_id)
                .fetch_optional(&mut *tx)
                .await?;
                if live.is_none() {
                    return Err(inlined_flush_conflict(table.table_id, base_snapshot));
                }
                sqlx::query(
                    "INSERT IGNORE INTO ducklake_table_stats
                         (table_id, record_count, next_row_id, file_size_bytes)
                     VALUES (?, 0, 0, 0)",
                )
                .bind(table.table_id)
                .execute(&mut *tx)
                .await?;
                let output_count = i64::try_from(table.files.len()).map_err(|_| {
                    crate::DuckLakeError::InvalidConfig("flush file count exceeds i64".to_string())
                })?;
                let output_ids = if output_count == 0 {
                    Vec::new()
                } else {
                    reserve_file_ids(&mut tx, output_count).await?
                };
                let mut added_bytes = 0;
                for (out, data_file_id) in table.files.iter().zip(output_ids) {
                    sqlx::query(
                        "INSERT INTO ducklake_data_file
                             (data_file_id, table_id, path, path_is_relative, file_size_bytes,
                              footer_size, record_count, row_id_start, begin_snapshot, partial_max)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    )
                    .bind(data_file_id)
                    .bind(table.table_id)
                    .bind(&out.file.path)
                    .bind(out.file.path_is_relative)
                    .bind(out.file.file_size_bytes)
                    .bind(out.file.footer_size)
                    .bind(out.file.record_count)
                    .bind(carried_row_id_start(&out.file))
                    .bind(out.begin_snapshot)
                    .bind(out.partial_max)
                    .execute(&mut *tx)
                    .await?;
                    insert_file_column_stats(
                        &mut tx,
                        table.table_id,
                        data_file_id,
                        &out.file.column_stats,
                    )
                    .await?;
                    insert_partition_metadata(&mut tx, table.table_id, data_file_id, &out.file)
                        .await?;
                    if let Some(delete) = &out.delete {
                        insert_flush_delete_file(&mut tx, table.table_id, data_file_id, delete)
                            .await?;
                    }
                    added_bytes += out.file.file_size_bytes;
                }
                replace_delete_files(&mut tx, table.table_id, &table.file_deletes, || {
                    inlined_flush_conflict(table.table_id, base_snapshot)
                })
                .await?;
                if !table.file_deletes.is_empty() {
                    let inlined_delete =
                        crate::metadata_provider::inlined_delete_table_name(table.table_id)?;
                    let ids = id_list(
                        &table
                            .file_deletes
                            .iter()
                            .map(|entry| entry.data_file_id)
                            .collect::<Vec<_>>(),
                    );
                    sqlx::query(AssertSqlSafe(format!(
                        "DELETE FROM {} WHERE file_id IN ({ids}) AND begin_snapshot <= ?",
                        quote_ident(&inlined_delete)
                    )))
                    .bind(base_snapshot)
                    .execute(&mut *tx)
                    .await?;
                }
                for name in &table.inlined_tables {
                    let system = inlined_table_layout(&mut *tx, name).await?.system;
                    sqlx::query(AssertSqlSafe(format!(
                        "DELETE FROM {} WHERE {} <= ?",
                        quote_ident(name),
                        system.begin_snapshot,
                    )))
                    .bind(base_snapshot)
                    .execute(&mut *tx)
                    .await?;
                }
                if !table.files.is_empty() {
                    sqlx::query(
                        "UPDATE ducklake_table_stats SET file_size_bytes = file_size_bytes + ?
                         WHERE table_id = ?",
                    )
                    .bind(added_bytes)
                    .bind(table.table_id)
                    .execute(&mut *tx)
                    .await?;
                    let fields = live_stat_fields(&mut tx, table.table_id).await?;
                    merge_file_table_column_stats(
                        &mut tx,
                        table.table_id,
                        &fields,
                        &crate::metadata_writer::file_column_stats(
                            table.files.iter().map(|flushed| &flushed.file),
                        ),
                    )
                    .await?;
                }
            }
            record_snapshot_changes(
                &mut tx,
                snapshot_id,
                &inlined_flush_changes(tables),
                &SnapshotCommitMetadata::default(),
            )
            .await?;
            crate::metadata_writer::commit_networked(tx).await?;
            Ok(snapshot_id)
        })
    }

    fn retire_appends_since(&self, table_id: i64, base_snapshot: i64) -> Result<Option<i64>> {
        block_on(async {
            let mut tx = self.pool.begin().await?;
            let (snapshot_id, _schema_version) = insert_snapshot(&mut tx).await?;
            // A delete file rewritten after base begins at its earliest deletion,
            // which may precede base; its `partial_max` is its latest one.
            let impure_delete: Option<i64> = sqlx::query_scalar(
                "SELECT 1 FROM ducklake_delete_file
                 WHERE table_id = ? AND (begin_snapshot > ? OR partial_max > ?) LIMIT 1",
            )
            .bind(table_id)
            .bind(base_snapshot)
            .bind(base_snapshot)
            .fetch_optional(&mut *tx)
            .await?;
            // A base-era file ended since, or a live file holding rows of both
            // sides of base (a flush's file begins at its earliest row).
            let impure_ended: Option<i64> = sqlx::query_scalar(
                "SELECT 1 FROM ducklake_data_file
                 WHERE table_id = ? AND begin_snapshot <= ?
                   AND ((end_snapshot IS NOT NULL AND end_snapshot > ?)
                        OR (end_snapshot IS NULL AND partial_max > ?)) LIMIT 1",
            )
            .bind(table_id)
            .bind(base_snapshot)
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
            // Inlined rows: a row ended after base is a deletion (or an update's
            // old version), as is any inlined deletion of a Parquet row.
            let table_exists = "SELECT COUNT(*) FROM information_schema.tables
                 WHERE table_schema = DATABASE() AND table_name = ?";
            let has_registry: i64 = sqlx::query_scalar(table_exists)
                .bind("ducklake_inlined_data_tables")
                .fetch_one(&mut *tx)
                .await?;
            let inlined = if has_registry > 0 {
                inlined_tables(&mut tx, table_id).await?
            } else {
                Vec::new()
            };
            let mut impure_inlined = false;
            for (name, system) in &inlined {
                let ended: Option<i64> = sqlx::query_scalar(AssertSqlSafe(format!(
                    "SELECT 1 FROM {} WHERE {} > ? LIMIT 1",
                    quote_ident(name),
                    system.end_snapshot,
                )))
                .bind(base_snapshot)
                .fetch_optional(&mut *tx)
                .await?;
                impure_inlined |= ended.is_some();
            }
            let inlined_deletes = crate::metadata_provider::inlined_delete_table_name(table_id)?;
            let has_inlined_deletes: i64 = sqlx::query_scalar(table_exists)
                .bind(&inlined_deletes)
                .fetch_one(&mut *tx)
                .await?;
            if has_inlined_deletes > 0 {
                let deleted: Option<i64> = sqlx::query_scalar(AssertSqlSafe(format!(
                    "SELECT 1 FROM {} WHERE begin_snapshot > ? LIMIT 1",
                    quote_ident(&inlined_deletes),
                )))
                .bind(base_snapshot)
                .fetch_optional(&mut *tx)
                .await?;
                impure_inlined |= deleted.is_some();
            }
            if impure_delete.is_some()
                || impure_ended.is_some()
                || impure_column.is_some()
                || impure_partition.is_some()
                || impure_inlined
            {
                return Err(crate::DuckLakeError::Conflict(format!(
                    "table {table_id}: changes since snapshot {base_snapshot} are not a pure \
                     append (delete/replace/update or schema/partition change present); refusing \
                     to retire"
                )));
            }
            let appended_rows: Option<i64> = sqlx::query_scalar(
                "SELECT CAST(SUM(record_count) AS SIGNED) FROM ducklake_data_file
                 WHERE table_id = ? AND end_snapshot IS NULL AND begin_snapshot > ?",
            )
            .bind(table_id)
            .bind(base_snapshot)
            .fetch_one(&mut *tx)
            .await?;
            let mut appended_inlined = 0_i64;
            for (name, system) in &inlined {
                appended_inlined += sqlx::query_scalar::<_, i64>(AssertSqlSafe(format!(
                    "SELECT COUNT(*) FROM {} WHERE {} > ? AND {} IS NULL",
                    quote_ident(name),
                    system.begin_snapshot,
                    system.end_snapshot,
                )))
                .bind(base_snapshot)
                .fetch_one(&mut *tx)
                .await?;
            }
            if appended_rows.is_none() && appended_inlined == 0 {
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
            for (name, system) in &inlined {
                sqlx::query(AssertSqlSafe(format!(
                    "UPDATE {} SET {} = ? WHERE {} > ? AND {} IS NULL",
                    quote_ident(name),
                    system.end_snapshot,
                    system.begin_snapshot,
                    system.end_snapshot,
                )))
                .bind(snapshot_id)
                .bind(base_snapshot)
                .execute(&mut *tx)
                .await?;
            }
            // `ducklake_table_stats` is left as it is, as official's `DELETE`
            // leaves it: `record_count` stays the gross count of rows ever inserted.
            let changes_made = match (appended_rows.is_some(), appended_inlined > 0) {
                (true, true) => format!("deleted_from_table:{table_id},inlined_delete:{table_id}"),
                (false, true) => format!("inlined_delete:{table_id}"),
                _ => format!("deleted_from_table:{table_id}"),
            };
            record_snapshot_changes(
                &mut tx,
                snapshot_id,
                &changes_made,
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
        base_snapshot: i64,
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
            let ours = TableChanges {
                deleted_from: has_live_data.is_some(),
                deleted_inlined: live_inlined > 0,
                ..TableChanges::default()
            };
            check_table_conflicts(&mut tx, table_id, base_snapshot, snapshot_id, ours).await?;
            let gross: Option<i64> = sqlx::query_scalar(
                "SELECT CAST(COALESCE(SUM(record_count), 0) AS SIGNED) FROM ducklake_data_file WHERE table_id = ? AND end_snapshot IS NULL",
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
            let live_rows = (gross.unwrap_or(0) + live_inlined - deleted).max(0) as u64;
            sqlx::query(
                "UPDATE ducklake_data_file SET end_snapshot = ?
                 WHERE table_id = ? AND end_snapshot IS NULL",
            )
            .bind(snapshot_id)
            .bind(table_id)
            .execute(&mut *tx)
            .await?;
            for (table_name, system) in inlined_tables(&mut tx, table_id).await? {
                let sql = format!(
                    "UPDATE {} SET {end} = ? WHERE {end} IS NULL",
                    quote_ident(&table_name),
                    end = system.end_snapshot,
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
            record_snapshot_changes(
                &mut tx,
                snapshot_id,
                &crate::metadata_writer::table_storage_changes(
                    table_id,
                    None,
                    ours.deleted_from,
                    ours.deleted_inlined,
                ),
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
            let mut tx = self.pool.begin().await?;
            enforce_inline_partition_fence(&mut tx, table_id, base_snapshot).await?;
            let snapshot_id =
                finalize_snapshot(&mut tx, table_id, columns, column_ids, mode, base_snapshot)
                    .await?;
            if mode != WriteMode::Replace
                && let Some(expected_base_snapshot_id) = expected_base_snapshot_id
            {
                detect_replace_conflict(
                    &mut tx,
                    table_id,
                    expected_base_snapshot_id,
                    ConflictScope::Table,
                )
                .await?;
            }
            create_eager_inlined_tables(&mut tx, &self.pool, snapshot_id).await?;
            let fields = inlined_fields(batches, columns);
            let (physical_name, system) =
                inlined_insert_table(&mut tx, &self.pool, table_id, snapshot_id, &fields).await?;
            let stats_initialized = table_stats_initialized(&mut tx, table_id).await?;
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
                        "INSERT INTO {} ({}, {}, {}, {}) VALUES (",
                        quote_ident(&physical_name),
                        system.row_id,
                        system.begin_snapshot,
                        system.end_snapshot,
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

            merge_inlined_table_column_stats(
                &mut tx,
                table_id,
                columns,
                column_ids,
                batches,
                &[],
                snapshot_id,
                stats_initialized,
            )
            .await?;
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
                true,
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
            for write in writes {
                if let StagedTableData::Inlined(batches) = &write.data
                    && batches.iter().all(|batch| batch.num_rows() == 0)
                {
                    return Err(crate::DuckLakeError::InvalidConfig(
                        "multi-table inline stage requires at least one row".to_string(),
                    ));
                }
            }
            for write in writes {
                validate_row_mutation_stage(write)?;
                if !write.inlined_file_deletes.is_empty() {
                    create_inlined_delete_table(&self.pool, write.table_id).await?;
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
                    detect_replace_conflict(
                        &mut tx,
                        write.table_id,
                        expected,
                        ConflictScope::of(write),
                    )
                    .await?;
                }
            }
            for write in writes {
                check_staged_conflicts(&mut tx, write, snapshot_id).await?;
            }
            for write in writes {
                if let Some(dated) = rewritten_deletes_snapshot(write) {
                    let between: bool = sqlx::query_scalar(
                        "SELECT EXISTS(SELECT 1 FROM ducklake_snapshot
                                       WHERE snapshot_id >= ? AND snapshot_id < ?)",
                    )
                    .bind(dated)
                    .bind(snapshot_id)
                    .fetch_one(&mut *tx)
                    .await?;
                    if between {
                        return Err(rewritten_delete_conflict(write.table_id, dated));
                    }
                }
            }
            for write in writes {
                detect_new_inlined_deletes(
                    &mut tx,
                    write.table_id,
                    write.base_snapshot_id,
                    &row_mutation_source_files(write),
                )
                .await?;
            }
            let mut had_live_data = Vec::with_capacity(writes.len());
            for write in writes {
                had_live_data.push(has_live_data(&mut tx, write.table_id).await?);
            }
            let mut tables = Vec::with_capacity(writes.len());
            for write in writes {
                let schema_id = validate_staged_table(&mut tx, write).await?;
                finalize_table_snapshot(
                    &mut tx,
                    snapshot_id,
                    write.table_id,
                    &write.columns,
                    &write.column_ids,
                    write.mode,
                    write.base_snapshot_id,
                    !matches!(write.data, StagedTableData::None),
                )
                .await?;
                tables.push(CommitIds {
                    snapshot_id,
                    schema_id,
                    table_id: write.table_id,
                });
            }
            create_eager_inlined_tables(&mut tx, &self.pool, snapshot_id).await?;
            let mut inline_layouts = std::collections::HashMap::new();
            for write in writes {
                if let StagedTableData::Inlined(batches) = &write.data {
                    let fields = inlined_fields(batches, &write.columns);
                    let target = inlined_insert_table(
                        &mut tx,
                        &self.pool,
                        write.table_id,
                        snapshot_id,
                        &fields,
                    )
                    .await?;
                    inline_layouts.insert(write.table_id, target);
                }
            }
            for (write, (had_files, had_inlined_rows)) in writes.iter().zip(had_live_data) {
                match &write.data {
                    StagedTableData::Files(files) => {
                        commit_staged_files(&mut tx, snapshot_id, write, files).await?;
                    },
                    StagedTableData::Inlined(batches) => {
                        let (physical_name, system) = inline_layouts
                            .get(&write.table_id)
                            .cloned()
                            .ok_or_else(|| {
                                crate::DuckLakeError::Internal(format!(
                                    "no inlined table was prepared for table {}",
                                    write.table_id
                                ))
                            })?;
                        commit_staged_inline(
                            &mut tx,
                            snapshot_id,
                            write,
                            batches,
                            &physical_name,
                            system,
                        )
                        .await?;
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
                replace_delete_files(&mut tx, write.table_id, &write.rewritten_deletes, || {
                    crate::DuckLakeError::Conflict(format!(
                        "a data file of table {} or its delete file changed since snapshot {}; \
                         re-open the catalog and retry the statement",
                        write.table_id, write.base_snapshot_id
                    ))
                })
                .await?;
                apply_staged_inlined_deletes(&mut tx, snapshot_id, write).await?;
                apply_inlined_file_deletes(
                    &mut tx,
                    write.table_id,
                    snapshot_id,
                    write.base_snapshot_id,
                    &write.inlined_file_deletes,
                )
                .await?;
                drop_data_files(
                    &mut tx,
                    write.table_id,
                    snapshot_id,
                    write.base_snapshot_id,
                    &write.dropped_files,
                )
                .await?;
                let changes_made = staged_table_write_changes(write, had_files, had_inlined_rows);
                record_table_changes(
                    &mut tx,
                    snapshot_id,
                    write.table_id,
                    &write.schema_name,
                    &write.table_name,
                    &changes_made,
                    commit_metadata,
                )
                .await?;
            }
            crate::metadata_writer::commit_networked(tx).await?;
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
            let ours = TableChanges {
                deleted_inlined: true,
                ..TableChanges::default()
            };
            check_table_conflicts(&mut tx, table_id, base_snapshot, snapshot_id, ours).await?;
            let registered = inlined_tables(&mut tx, table_id)
                .await?
                .into_iter()
                .collect::<std::collections::HashMap<_, _>>();
            for row in rows {
                let Some(system) = registered.get(&row.table_name) else {
                    return Err(crate::DuckLakeError::Conflict(format!(
                        "inlined row {} belongs to an unregistered table '{}'",
                        row.row_id, row.table_name
                    )));
                };
                let sql = format!(
                    "UPDATE {} SET {end} = ? \
                     WHERE {row_id} = ? AND {begin} <= ? AND {end} IS NULL",
                    quote_ident(&row.table_name),
                    end = system.end_snapshot,
                    row_id = system.row_id,
                    begin = system.begin_snapshot,
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
            let changes_made = crate::metadata_writer::table_storage_changes(
                table_id,
                None,
                ours.deleted_from,
                ours.deleted_inlined,
            );
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
            create_eager_inlined_table(&mut tx, &self.pool, table_id, new_schema_version).await?;
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
            create_eager_inlined_table(&mut tx, &self.pool, table_id, new_schema_version).await?;
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
            create_eager_inlined_tables(&mut tx, &self.pool, snapshot_id).await?;
            tx.commit().await?;
            Ok(CommitIds {
                snapshot_id,
                schema_id,
                table_id,
            })
        })
    }

    fn end_table_files(&self, table_id: i64, snapshot_id: i64) -> Result<u64> {
        // Used by WriteMode::Replace. `ducklake_table_stats` is left as it is, as
        // official's `DELETE` leaves it: `record_count` stays the gross count of
        // rows ever inserted, and `next_row_id` stays monotonic so historical
        // snapshots still resolve uniquely.
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

    fn get_table_column_nullability(
        &self,
        schema_name: &str,
        table_name: &str,
    ) -> Result<Option<Vec<(String, bool)>>> {
        block_on(async {
            let rows = sqlx::query(
                "SELECT c.column_name, c.nulls_allowed
                 FROM ducklake_column c
                 JOIN ducklake_table t ON t.table_id = c.table_id
                 JOIN ducklake_schema s ON s.schema_id = t.schema_id
                 WHERE s.schema_name = ? AND s.end_snapshot IS NULL
                   AND t.table_name = ? AND t.end_snapshot IS NULL
                   AND c.end_snapshot IS NULL AND c.parent_column IS NULL
                 ORDER BY c.column_order",
            )
            .bind(schema_name)
            .bind(table_name)
            .fetch_all(&self.pool)
            .await?;
            if rows.is_empty() {
                return Ok(None);
            }
            rows.into_iter()
                .map(|row| {
                    let name: String = row.try_get(0)?;
                    let nullable = row.try_get::<Option<bool>, _>(1)?.unwrap_or(true);
                    Ok((name, nullable))
                })
                .collect::<std::result::Result<Vec<_>, sqlx::Error>>()
                .map(Some)
                .map_err(Into::into)
        })
    }

    fn set_data_path(&self, path: &str) -> Result<()> {
        let path = directory_path(path);
        block_on(async {
            let current: Option<String> = sqlx::query_scalar(
                "SELECT `value` FROM ducklake_metadata WHERE `key` = 'data_path' AND scope IS NULL",
            )
            .fetch_optional(&self.pool)
            .await?;
            if current.as_deref() == Some(path.as_str()) {
                return Ok(());
            }

            sqlx::query(
                "DELETE FROM ducklake_metadata WHERE `key` = 'data_path' AND scope IS NULL",
            )
            .execute(&self.pool)
            .await?;

            sqlx::query(
                "INSERT INTO ducklake_metadata (`key`, `value`, scope)
                 VALUES ('data_path', ?, NULL)",
            )
            .bind(&path)
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
                add_column(
                    &self.pool,
                    "ducklake_metadata",
                    "scope",
                    "ALTER TABLE ducklake_metadata ADD COLUMN scope VARCHAR(255)",
                )
                .await?;
            }
            if has_scope_id == 0 {
                add_column(
                    &self.pool,
                    "ducklake_metadata",
                    "scope_id",
                    "ALTER TABLE ducklake_metadata ADD COLUMN scope_id BIGINT",
                )
                .await?;
            }
            // The partition pre-filter (`crate::stats_filter`) reads
            // ducklake_file_partition_value once per listing, restricted to one
            // table's rows for one partition key; unindexed that is a scan of
            // every table's partition values. MySQL has no
            // `CREATE INDEX IF NOT EXISTS`, so probe information_schema first,
            // exactly as the column upgrades above do. Both columns are
            // fixed-width deliberately — `partition_value` is TEXT, which MySQL
            // cannot index without a prefix length at all.
            let has_partition_value_index: i64 = sqlx::query_scalar(
                "SELECT COUNT(*) FROM information_schema.statistics \
                 WHERE table_schema = DATABASE() \
                   AND table_name = 'ducklake_file_partition_value' \
                   AND index_name = 'idx_file_partition_value_table_key'",
            )
            .fetch_one(&self.pool)
            .await?;
            if has_partition_value_index == 0 {
                // Check-then-create races another process running the same
                // bootstrap, and the loser gets ER_DUP_KEYNAME. Re-probe on
                // failure and accept an index that is now there, exactly as
                // `create_mysql_physical_indexes` does; a real failure still
                // surfaces.
                let created = sqlx::query(
                    "CREATE INDEX idx_file_partition_value_table_key \
                     ON ducklake_file_partition_value (table_id, partition_key_index)",
                )
                .execute(&self.pool)
                .await;
                if let Err(error) = created {
                    let now_exists: i64 = sqlx::query_scalar(
                        "SELECT COUNT(*) FROM information_schema.statistics \
                         WHERE table_schema = DATABASE() \
                           AND table_name = 'ducklake_file_partition_value' \
                           AND index_name = 'idx_file_partition_value_table_key'",
                    )
                    .fetch_one(&self.pool)
                    .await?;
                    if now_exists == 0 {
                        return Err(error.into());
                    }
                }
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
                    add_column(
                        &self.pool,
                        table,
                        "partial_max",
                        &format!("ALTER TABLE {table} ADD COLUMN partial_max BIGINT"),
                    )
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
            migrate_directory_paths(&self.pool).await?;
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
                    .bind(directory_path(schema_name))
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
                    .bind(directory_path(table_name))
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

async fn enforce_inline_partition_fence(
    tx: &mut sqlx::Transaction<'_, sqlx::MySql>,
    table_id: i64,
    base_snapshot: i64,
) -> Result<()> {
    let changed: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM ducklake_partition_info WHERE table_id = ? AND (begin_snapshot > ? OR end_snapshot > ?)").bind(table_id).bind(base_snapshot).bind(base_snapshot).fetch_one(&mut **tx).await?;
    if changed != 0 {
        return Err(crate::DuckLakeError::Conflict(format!(
            "partition spec for table {table_id} changed during inline write; retry the write"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Int64Array;
    use std::sync::Arc;
    use testcontainers::runners::AsyncRunner;
    use testcontainers_modules::mysql::Mysql;

    #[rstest::rstest]
    #[tokio::test(flavor = "multi_thread")]
    #[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
    async fn mysql_staged_inline_uses_the_locked_schema_version() {
        let container = Mysql::default().start().await.unwrap();
        let port = container.get_host_port_ipv4(3306).await.unwrap();
        let url = format!("mysql://root@127.0.0.1:{port}/test");
        let writer = MySqlMetadataWriter::with_max_connections(&url, 1)
            .await
            .unwrap();
        writer.initialize_schema().unwrap();
        let columns = vec![ColumnDef::new("value", "BIGINT", false).unwrap()];
        let batch = RecordBatch::try_from_iter(vec![(
            "value",
            Arc::new(Int64Array::from(vec![7, 11])) as Arc<dyn Array>,
        )])
        .unwrap();
        let setup = writer
            .begin_write_transaction("main", "events", &columns, WriteMode::Append)
            .unwrap();
        let write = StagedTableWrite {
            table_id: setup.table_id,
            schema_name: "main".to_string(),
            table_name: "events".to_string(),
            base_snapshot_id: setup.base_snapshot_id,
            mode: WriteMode::Append,
            columns,
            column_ids: setup.column_ids,
            data: StagedTableData::Inlined(vec![batch]),
            snapshot_id_columns: Vec::new(),
            positional_deletes: Vec::new(),
            inlined_deletes: Vec::new(),
            inlined_row_ids: None,
            inlined_file_deletes: Vec::new(),
            rewritten_deletes: Vec::new(),
            dropped_files: Vec::new(),
            inlined_flush: false,
        };
        let committed = writer
            .commit_multi_table(
                std::slice::from_ref(&write),
                &SnapshotCommitMetadata::default(),
                None,
            )
            .unwrap();
        let schema_version: i64 = sqlx::query_scalar(
            "SELECT schema_version FROM ducklake_snapshot WHERE snapshot_id = ?",
        )
        .bind(committed.snapshot_id)
        .fetch_one(&writer.pool)
        .await
        .unwrap();
        let physical: String = sqlx::query_scalar(
            "SELECT table_name FROM ducklake_inlined_data_tables WHERE table_id = ?",
        )
        .bind(write.table_id)
        .fetch_one(&writer.pool)
        .await
        .unwrap();
        let sql = format!(
            "SELECT row_id, begin_snapshot, value FROM {} ORDER BY row_id",
            quote_ident(&physical)
        );
        let rows: Vec<(i64, i64, i64)> = sqlx::query_as(AssertSqlSafe(sql))
            .fetch_all(&writer.pool)
            .await
            .unwrap();
        assert_eq!(schema_version, 1);
        assert_eq!(
            physical,
            format!("ducklake_inlined_data_{}_1", write.table_id)
        );
        assert_eq!(
            rows,
            vec![(0, committed.snapshot_id, 7), (1, committed.snapshot_id, 11)]
        );
        let mut empty = write;
        empty.data = StagedTableData::Inlined(Vec::new());
        let error = writer
            .commit_multi_table(&[empty], &SnapshotCommitMetadata::default(), None)
            .unwrap_err();
        let state: (i64, i64) = sqlx::query_as("SELECT (SELECT COUNT(*) FROM ducklake_snapshot), (SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = DATABASE() AND table_name LIKE 'ducklake_inlined_data_%')").fetch_one(&writer.pool).await.unwrap();
        assert!(matches!(error, crate::DuckLakeError::InvalidConfig(_)));
        assert_eq!(state, (1, 2));
    }

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
