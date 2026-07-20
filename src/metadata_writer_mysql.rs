//! MySQL implementation of [`MetadataWriter`].
//!
//! Single-catalog only — the legacy DuckLake v1.0 layout, mirroring
//! [`crate::metadata_writer_sqlite::SqliteMetadataWriter`] rather than the
//! multicatalog Postgres writer. It supports the write primitives the crate's
//! table-write path needs (INSERT / REPLACE / CREATE TABLE) and deliberately
//! does NOT support deletes, upserts, compaction, partitioning, type promotion,
//! or multiple catalogs: [`MetadataWriter::promote_column_type`] and
//! [`MetadataWriter::set_delete_file`] inherit their erroring defaults, and
//! [`MetadataWriter::catalog_id`] inherits the `None` default (which keeps
//! newly-written file paths in the `{data_path}/{schema}/{table}/…` layout).
//!
//! Requires a multi-threaded Tokio runtime (`#[tokio::test(flavor =
//! "multi_thread")]`): the sync trait methods bridge async sqlx via
//! `crate::metadata_provider::block_on`, exactly like the SQLite writer.
//!
//! ## MySQL dialect adaptations vs the SQLite template
//!
//! 1. **No `RETURNING`.** Auto-increment PK ids (`schema_id`, `table_id`,
//!    `data_file_id`) are read back with `MySqlQueryResult::last_insert_id()`;
//!    counter-allocated ids (`column_id`, `snapshot_id`) are read back with an
//!    `UPDATE` followed by a `SELECT` in the same transaction (`reserve_ids`).
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
use crate::metadata_provider::block_on;
use crate::metadata_writer::{
    ColumnDef, ColumnStat, CommitIds, DataFileInfo, MetadataWriter, WriteMode, WriteSetupResult,
    columns_differ, validate_name,
};
use crate::partition::PartitionTransform;
use sqlx::Row;
use sqlx::mysql::{MySqlPool, MySqlPoolOptions};

const DEFAULT_MAX_CONNECTIONS: u32 = 5;

/// The DuckLake v1.0 catalog tables in MySQL dialect, one `CREATE TABLE` per
/// entry. sqlx runs each `query()` as a single prepared statement on MySQL
/// (unlike the SQLite driver's multi-statement exec), so — like the Postgres
/// writer — the DDL must be split rather than sent as one `;`-joined script.
///
/// Columns and their order match the SQLite writer's `SQL_CREATE_SCHEMA` (and so
/// upstream DuckLake) for catalog compatibility; only the SQL types are mapped
/// to MySQL. Auto-increment PKs back the ids read via `last_insert_id()`
/// (`schema_id`/`table_id`/`data_file_id`/`delete_file_id`). `snapshot_id` is a
/// plain PK assigned from the `next_snapshot_id` counter, and `ducklake_column`
/// is a bare table (no PK) so a versioned column can hold multiple rows sharing
/// a `column_id`.
const SQL_CREATE_TABLES: &[&str] = &[
    r#"CREATE TABLE IF NOT EXISTS ducklake_metadata (
        `key` VARCHAR(1024) NOT NULL,
        `value` TEXT NOT NULL,
        scope VARCHAR(1024)
    ) ENGINE = InnoDB"#,
    r#"CREATE TABLE IF NOT EXISTS ducklake_snapshot (
        snapshot_id BIGINT NOT NULL PRIMARY KEY,
        snapshot_time DATETIME(6) DEFAULT CURRENT_TIMESTAMP(6),
        schema_version BIGINT NOT NULL DEFAULT 0
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
    // Bare table (no PRIMARY KEY), mirroring upstream `ducklake_column`: a column
    // is versioned by `[begin_snapshot, end_snapshot)` and — although this writer
    // never promotes types — the shape stays identical to SQLite/upstream so
    // catalogs interoperate. The four `*default*` columns and `parent_column` are
    // left NULL (no nested-type / column-default writes).
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
    // Created for catalog-shape parity and so the provider's LEFT JOINs resolve;
    // this writer never inserts delete files (`set_delete_file` is unsupported).
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
        end_snapshot BIGINT
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

/// Seed a monotonic id counter row if it does not already exist, starting from
/// the current MAX of its backing column so a pre-existing catalog keeps
/// allocating without reuse. Done as check-then-insert (two statements) rather
/// than `INSERT … SELECT … WHERE NOT EXISTS`, because that self-referential
/// `INSERT … SELECT` against `ducklake_metadata` is rejected by MySQL (1093).
async fn seed_counter(pool: &MySqlPool, key: &str, max_sql: &str) -> Result<()> {
    let exists: i64 =
        sqlx::query("SELECT COUNT(*) FROM ducklake_metadata WHERE `key` = ? AND scope IS NULL")
            .bind(key)
            .fetch_one(pool)
            .await?
            .try_get(0)?;
    if exists == 0 {
        let start: i64 = sqlx::query(max_sql).fetch_one(pool).await?.try_get(0)?;
        sqlx::query("INSERT INTO ducklake_metadata (`key`, `value`, scope) VALUES (?, ?, NULL)")
            .bind(key)
            .bind(start.to_string())
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
    Ok((snapshot_id, schema_version))
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

async fn recompute_table_column_stats(
    tx: &mut sqlx::Transaction<'_, sqlx::MySql>,
    table_id: i64,
    columns: &[ColumnDef],
    column_ids: &[i64],
) -> Result<()> {
    use crate::stats_encode::{FileColumnStat, aggregate_global_column_stats};

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
    let current = sqlx::query(
        "SELECT column_name, column_type, column_order, nulls_allowed
         FROM ducklake_column
         WHERE table_id = ? AND end_snapshot IS NULL
         ORDER BY column_order",
    )
    .bind(table_id)
    .fetch_all(&mut **tx)
    .await?;

    let mut existing: Vec<(String, String, bool)> = Vec::with_capacity(current.len());
    for row in &current {
        let name: String = row.try_get("column_name")?;
        let ty: String = row.try_get("column_type")?;
        let nullable: bool = row
            .try_get::<Option<bool>, _>("nulls_allowed")?
            .unwrap_or(true);
        existing.push((name, ty, nullable));
    }
    let is_ddl = existing.is_empty() || columns_differ(&existing, columns);
    if is_ddl {
        // A DDL commit bumps the per-catalog schema_version (the insert above only
        // carried it forward). A pure data write keeps the carried value.
        schema_version = bump_schema_version(tx, snapshot_id).await?;
    }

    // Reconcile the column generation SURGICALLY so each column keeps a stable
    // column_id (== parquet field_id) across writes: end only removed columns,
    // insert only new ones, and leave unchanged columns (and their ids) in place.
    let new_names: HashSet<&str> = columns.iter().map(|c| c.name()).collect();
    let mut current_by_name: HashMap<String, (i64, bool)> = HashMap::new();
    for row in &current {
        let name: String = row.try_get("column_name")?;
        let order: i64 = row.try_get("column_order")?;
        let nullable: bool = row
            .try_get::<Option<bool>, _>("nulls_allowed")?
            .unwrap_or(true);
        if !new_names.contains(name.as_str()) {
            // Column dropped in the new schema: end its generation.
            sqlx::query(
                "UPDATE ducklake_column SET end_snapshot = ?
                 WHERE table_id = ? AND column_name = ? AND end_snapshot IS NULL",
            )
            .bind(snapshot_id)
            .bind(table_id)
            .bind(&name)
            .execute(&mut **tx)
            .await?;
        }
        current_by_name.insert(name, (order, nullable));
    }

    for (order, (col, column_id)) in columns.iter().zip(column_ids.iter()).enumerate() {
        match current_by_name.get(col.name()) {
            // Existing column kept: its id stays stable. Sync order/nullability
            // only if they changed (type changes are rejected at begin).
            Some(&(cur_order, cur_nullable)) => {
                if cur_order != order as i64 || cur_nullable != col.is_nullable() {
                    sqlx::query(
                        "UPDATE ducklake_column SET column_order = ?, nulls_allowed = ?
                         WHERE table_id = ? AND column_name = ? AND end_snapshot IS NULL",
                    )
                    .bind(order as i64)
                    .bind(col.is_nullable())
                    .bind(table_id)
                    .bind(col.name())
                    .execute(&mut **tx)
                    .await?;
                }
            },
            // Newly added column: insert it with its reserved id.
            None => {
                sqlx::query(
                    "INSERT INTO ducklake_column
                         (column_id, table_id, column_name, column_type, column_order, nulls_allowed, begin_snapshot)
                     VALUES (?, ?, ?, ?, ?, ?, ?)",
                )
                .bind(column_id)
                .bind(table_id)
                .bind(col.name())
                .bind(col.ducklake_type())
                .bind(order as i64)
                .bind(col.is_nullable())
                .bind(snapshot_id)
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

impl MetadataWriter for MySqlMetadataWriter {
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
            let existing = sqlx::query(
                "SELECT schema_id FROM ducklake_schema
                 WHERE schema_name = ? AND end_snapshot IS NULL",
            )
            .bind(name)
            .fetch_optional(&self.pool)
            .await?;

            if let Some(row) = existing {
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
            .execute(&self.pool)
            .await?;

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
            let existing = sqlx::query(
                "SELECT table_id FROM ducklake_table
                 WHERE schema_id = ? AND table_name = ? AND end_snapshot IS NULL",
            )
            .bind(schema_id)
            .bind(name)
            .fetch_optional(&self.pool)
            .await?;

            if let Some(row) = existing {
                return Ok((row.try_get(0)?, false));
            }

            let table_path = path.unwrap_or(name);
            let result = sqlx::query(
                "INSERT INTO ducklake_table (schema_id, table_name, path, path_is_relative, begin_snapshot)
                 VALUES (?, ?, ?, 1, ?)",
            )
            .bind(schema_id)
            .bind(name)
            .bind(table_path)
            .bind(snapshot_id)
            .execute(&self.pool)
            .await?;

            Ok((result.last_insert_id() as i64, true))
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
            let n = columns.len() as i64;
            let last_column_id = reserve_ids(&mut tx, "next_column_id", n).await?;
            let first_column_id = last_column_id - n + 1;
            let mut column_ids = Vec::with_capacity(columns.len());
            for (order, col) in columns.iter().enumerate() {
                let column_id = first_column_id + order as i64;
                sqlx::query(
                    "INSERT INTO ducklake_column (column_id, table_id, column_name, column_type, column_order, nulls_allowed, begin_snapshot)
                     VALUES (?, ?, ?, ?, ?, ?, ?)",
                )
                .bind(column_id)
                .bind(table_id)
                .bind(col.name())
                .bind(col.ducklake_type())
                .bind(order as i64)
                .bind(col.is_nullable())
                .bind(snapshot_id)
                .execute(&mut *tx)
                .await?;
                column_ids.push(column_id);
            }

            tx.commit().await?;
            Ok(column_ids)
        })
    }

    fn register_data_file(
        &self,
        table_id: i64,
        // MySQL created the schema/table at begin, so the names are unused here;
        // accepted only to satisfy the trait shared with multicatalog Postgres.
        _schema_name: &str,
        _table_name: &str,
        _snapshot_id: i64,
        file: &DataFileInfo,
        mode: WriteMode,
        base_snapshot: i64,
        columns: &[ColumnDef],
        column_ids: &[i64],
    ) -> Result<CommitIds> {
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

            let inserted = sqlx::query(
                "INSERT INTO ducklake_data_file
                     (table_id, path, path_is_relative, file_size_bytes,
                      footer_size, record_count, row_id_start, begin_snapshot)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            )
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

            // MySQL has no RETURNING: the auto-increment PK is read via
            // last_insert_id(). Persist the file's zone maps + refresh the roll-up.
            let data_file_id = inserted.last_insert_id() as i64;
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
        _schema_name: &str,
        _table_name: &str,
        _snapshot_id: i64,
        files: &[DataFileInfo],
        mode: WriteMode,
        base_snapshot: i64,
        columns: &[ColumnDef],
        column_ids: &[i64],
    ) -> Result<CommitIds> {
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
            for file in files {
                let inserted = sqlx::query(
                    "INSERT INTO ducklake_data_file
                         (table_id, path, path_is_relative, file_size_bytes,
                          footer_size, record_count, row_id_start, begin_snapshot)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                )
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
                let data_file_id = inserted.last_insert_id() as i64;
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
                     WHERE table_id = ? AND column_name = ? AND end_snapshot IS NULL",
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
            tx.commit().await?;
            Ok(new_snapshot)
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
            tx.commit().await?;
            Ok(new_snapshot)
        })
    }

    fn publish_snapshot(
        &self,
        table_id: i64,
        // MySQL created the schema/table at begin; names unused (trait parity).
        _schema_name: &str,
        _table_name: &str,
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
                sqlx::query(ddl).execute(&self.pool).await?;
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
                "next_partition_id",
                "SELECT COALESCE(MAX(partition_id), 0) FROM ducklake_partition_info",
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
            let n = columns.len() as i64;
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
                "SELECT column_name, column_type, nulls_allowed, column_id
                 FROM ducklake_column
                 WHERE table_id = ? AND end_snapshot IS NULL
                 ORDER BY column_order",
            )
            .bind(table_id)
            .fetch_all(&mut *tx)
            .await?;

            let mut existing_columns: Vec<(String, String, bool)> = Vec::with_capacity(rows.len());
            let mut existing_ids: std::collections::HashMap<String, i64> =
                std::collections::HashMap::new();
            for row in rows {
                let name: String = row.try_get(0)?;
                let col_type: String = row.try_get(1)?;
                let nullable: bool = row.try_get::<Option<bool>, _>(2)?.unwrap_or(true);
                let cid: i64 = row.try_get(3)?;
                existing_ids.insert(name.clone(), cid);
                existing_columns.push((name, col_type, nullable));
            }

            // Data-write policy (§5): a data write — Replace OR Append — must NOT
            // change a column's type (that is schema evolution and must go through
            // promote_column_type, which this backend does not support). The
            // comparison is canonical (`int64` ≡ `bigint`) so an alias-only
            // restatement is a no-op. Append additionally requires a genuinely new
            // column to be nullable.
            if !existing_columns.is_empty() {
                use std::collections::HashMap;

                let existing_map: HashMap<&str, (&str, bool)> = existing_columns
                    .iter()
                    .map(|(name, col_type, nullable)| {
                        (name.as_str(), (col_type.as_str(), *nullable))
                    })
                    .collect();

                for new_col in columns.iter() {
                    if let Some((existing_type, _existing_nullable)) =
                        existing_map.get(new_col.name())
                    {
                        if !crate::types::types_equal_canonical(
                            existing_type,
                            new_col.ducklake_type(),
                        ) {
                            return Err(crate::error::DuckLakeError::UnsupportedTypeChange {
                                operation: TypeChangeOperation::DataWrite {
                                    mode: match mode {
                                        WriteMode::Replace => TypeChangeWriteMode::Replace,
                                        WriteMode::Append => TypeChangeWriteMode::Append,
                                    },
                                },
                                column: new_col.name().to_string(),
                                from: (*existing_type).to_string(),
                                to: new_col.ducklake_type().to_string(),
                            });
                        }
                    } else if mode == WriteMode::Append && !new_col.is_nullable() {
                        return Err(crate::error::DuckLakeError::InvalidConfig(format!(
                            "Schema evolution error: new column '{}' must be nullable. Adding non-nullable columns is not allowed.",
                            new_col.name()
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
            let column_ids: Vec<i64> = columns
                .iter()
                .zip(fresh_ids.iter())
                .map(|(col, &fresh)| existing_ids.get(col.name()).copied().unwrap_or(fresh))
                .collect();

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
                column_ids,
            })
        })
    }
}
