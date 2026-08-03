//! Single-catalog PostgreSQL implementation of [`MetadataWriter`] — the standard,
//! spec-compliant DuckLake layout, readable by DuckDB's `ducklake` extension.
//!
//! Separate from [`crate::metadata_writer_postgres::PostgresMetadataWriter`], which
//! writes this crate's multicatalog layout (`catalog_id` columns plus
//! `ducklake_catalog*` map tables) and is not part of the DuckLake spec.
//!
//! Modelled on [`crate::metadata_writer_mysql::MySqlMetadataWriter`]: same
//! single-catalog structure and supported surface, SQL mapped to Postgres. Deletes,
//! upserts, compaction and type promotion inherit their erroring trait defaults, and
//! [`MetadataWriter::catalog_id`] inherits `None` so paths stay unscoped. Sync trait
//! methods bridge async sqlx via `crate::metadata_provider::block_on`, so a
//! multi-threaded Tokio runtime is required.
//!
//! `snapshot_id` is counter-allocated rather than IDENTITY. Upstream's column is a
//! plain `BIGINT PRIMARY KEY`, and the counter's `UPDATE` holds a row lock to commit,
//! so commits serialize and snapshot-id order equals commit order — what the
//! `Replace` conflict test relies on. Sequences are non-transactional and would not
//! give that; the multicatalog writer gets it from a per-catalog `FOR UPDATE` lock
//! that has no single-catalog equivalent.

use crate::Result;
use crate::error::{TypeChangeOperation, TypeChangeWriteMode};
use crate::metadata_provider::block_on;
use crate::metadata_writer::{
    ColumnDef, ColumnStat, CommitIds, DataFileInfo, MetadataWriter, SnapshotCommitMetadata,
    WriteMode, WriteSetupResult, columns_differ, quote_snapshot_name, quote_snapshot_table,
    table_write_changes, validate_name,
};
use crate::partition::PartitionTransform;
use sqlx::Row;
use sqlx::postgres::{PgPool, PgPoolOptions};

const DEFAULT_MAX_CONNECTIONS: u32 = 5;

/// The DuckLake catalog tables in Postgres dialect. Columns and their order match
/// the SQLite and MySQL writers (and so upstream); only the SQL types differ. Split
/// one per entry because sqlx runs each `query()` as a single prepared statement.
const SQL_CREATE_TABLES: &[&str] = &[
    r#"CREATE TABLE IF NOT EXISTS ducklake_metadata (
        key VARCHAR NOT NULL,
        value VARCHAR NOT NULL,
        scope VARCHAR
    )"#,
    r#"CREATE TABLE IF NOT EXISTS ducklake_snapshot (
        snapshot_id BIGINT NOT NULL PRIMARY KEY,
        snapshot_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        schema_version BIGINT NOT NULL DEFAULT 0
    )"#,
    // Per-snapshot change summary + commit metadata (DuckLake spec). `insert_snapshot`
    // seeds one row per snapshot with `changes_made` NULL; the commit paths fill it in
    // via `record_snapshot_changes`, appending as a comma-separated list.
    r#"CREATE TABLE IF NOT EXISTS ducklake_snapshot_changes (
        snapshot_id BIGINT NOT NULL PRIMARY KEY,
        changes_made VARCHAR,
        author VARCHAR,
        commit_message VARCHAR,
        commit_extra_info VARCHAR
    )"#,
    r#"CREATE TABLE IF NOT EXISTS ducklake_schema_versions (
        begin_snapshot BIGINT NOT NULL,
        schema_version BIGINT NOT NULL,
        table_id BIGINT NOT NULL,
        UNIQUE (table_id, begin_snapshot)
    )"#,
    r#"CREATE TABLE IF NOT EXISTS ducklake_schema (
        schema_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        schema_name VARCHAR NOT NULL,
        path VARCHAR NOT NULL DEFAULT '',
        path_is_relative BOOLEAN NOT NULL DEFAULT TRUE,
        begin_snapshot BIGINT NOT NULL,
        end_snapshot BIGINT
    )"#,
    r#"CREATE TABLE IF NOT EXISTS ducklake_table (
        table_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        schema_id BIGINT NOT NULL,
        table_name VARCHAR NOT NULL,
        path VARCHAR NOT NULL DEFAULT '',
        path_is_relative BOOLEAN NOT NULL DEFAULT TRUE,
        begin_snapshot BIGINT NOT NULL,
        end_snapshot BIGINT
    )"#,
    // Bare table (no PRIMARY KEY), mirroring upstream: a versioned column needs
    // several rows sharing one `column_id`. The `*default*` columns and
    // `parent_column` are left NULL.
    r#"CREATE TABLE IF NOT EXISTS ducklake_column (
        column_id BIGINT,
        begin_snapshot BIGINT,
        end_snapshot BIGINT,
        table_id BIGINT,
        column_order BIGINT,
        column_name VARCHAR,
        column_type VARCHAR,
        initial_default VARCHAR,
        default_value VARCHAR,
        nulls_allowed BOOLEAN,
        parent_column BIGINT,
        default_value_type VARCHAR,
        default_value_dialect VARCHAR
    )"#,
    r#"CREATE TABLE IF NOT EXISTS ducklake_data_file (
        data_file_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        table_id BIGINT NOT NULL,
        path VARCHAR NOT NULL,
        path_is_relative BOOLEAN NOT NULL DEFAULT TRUE,
        file_size_bytes BIGINT NOT NULL,
        footer_size BIGINT,
        encryption_key VARCHAR,
        record_count BIGINT,
        row_id_start BIGINT,
        mapping_id BIGINT,
        begin_snapshot BIGINT NOT NULL,
        end_snapshot BIGINT,
        partition_id BIGINT
    )"#,
    // Per-table row-lineage + running totals. `next_row_id` allocates rowids
    // monotonically over the table's lifetime; `record_count`/`file_size_bytes`
    // mirror the currently-visible totals for DuckDB's `ducklake_table_info`.
    r#"CREATE TABLE IF NOT EXISTS ducklake_table_stats (
        table_id BIGINT PRIMARY KEY,
        record_count BIGINT NOT NULL DEFAULT 0,
        next_row_id BIGINT NOT NULL DEFAULT 0,
        file_size_bytes BIGINT NOT NULL DEFAULT 0
    )"#,
    // Per-file, per-column zone maps (DuckLake spec) — powers file pruning.
    // Column set mirrors the official extension and the other backends.
    r#"CREATE TABLE IF NOT EXISTS ducklake_file_column_stats (
        data_file_id BIGINT NOT NULL,
        table_id BIGINT NOT NULL,
        column_id BIGINT NOT NULL,
        column_size_bytes BIGINT,
        value_count BIGINT,
        null_count BIGINT,
        min_value VARCHAR,
        max_value VARCHAR,
        contains_nan BOOLEAN,
        extra_stats VARCHAR
    )"#,
    // Table-wide per-column roll-up (DuckLake spec) — feeds the optimizer.
    r#"CREATE TABLE IF NOT EXISTS ducklake_table_column_stats (
        table_id BIGINT NOT NULL,
        column_id BIGINT NOT NULL,
        contains_null BOOLEAN,
        contains_nan BOOLEAN,
        min_value VARCHAR,
        max_value VARCHAR,
        extra_stats VARCHAR
    )"#,
    // Created for catalog-shape parity and so the provider's LEFT JOINs resolve;
    // this writer never inserts delete files (`set_delete_file` is unsupported).
    r#"CREATE TABLE IF NOT EXISTS ducklake_delete_file (
        delete_file_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        data_file_id BIGINT NOT NULL,
        table_id BIGINT NOT NULL,
        path VARCHAR NOT NULL,
        path_is_relative BOOLEAN NOT NULL DEFAULT TRUE,
        file_size_bytes BIGINT NOT NULL,
        footer_size BIGINT,
        encryption_key VARCHAR,
        delete_count BIGINT,
        begin_snapshot BIGINT NOT NULL,
        end_snapshot BIGINT
    )"#,
    r#"CREATE TABLE IF NOT EXISTS ducklake_files_scheduled_for_deletion (
        data_file_id BIGINT NOT NULL,
        path VARCHAR NOT NULL,
        path_is_relative BOOLEAN NOT NULL DEFAULT TRUE,
        schedule_start TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )"#,
    // Partition spec generations (DuckLake spec); end_snapshot NULL == active.
    r#"CREATE TABLE IF NOT EXISTS ducklake_partition_info (
        partition_id BIGINT NOT NULL,
        table_id BIGINT NOT NULL,
        begin_snapshot BIGINT NOT NULL,
        end_snapshot BIGINT
    )"#,
    // Partition-key columns for a spec (DuckLake spec), ordered by partition_key_index.
    r#"CREATE TABLE IF NOT EXISTS ducklake_partition_column (
        partition_id BIGINT NOT NULL,
        table_id BIGINT NOT NULL,
        partition_key_index BIGINT NOT NULL,
        column_id BIGINT NOT NULL,
        transform VARCHAR NOT NULL
    )"#,
    // Per-file partition values (DuckLake spec): the value every row in the file
    // shares for a partition key, DuckDB-canonical VARCHAR (NULL is legal).
    r#"CREATE TABLE IF NOT EXISTS ducklake_file_partition_value (
        data_file_id BIGINT NOT NULL,
        table_id BIGINT NOT NULL,
        partition_key_index BIGINT NOT NULL,
        partition_value VARCHAR
    )"#,
    // Sort spec generations (DuckLake spec); end_snapshot NULL == active. sort_id
    // is allocated from the next_sort_id counter (like partition_id).
    r#"CREATE TABLE IF NOT EXISTS ducklake_sort_info (
        sort_id BIGINT NOT NULL,
        table_id BIGINT NOT NULL,
        begin_snapshot BIGINT NOT NULL,
        end_snapshot BIGINT
    )"#,
    // Sort-key expressions for a spec (DuckLake spec), ordered by sort_key_index.
    // expression is a sort expression in `dialect` (this crate produces bare column
    // names under `duckdb`); sort_direction ASC/DESC; null_order NULLS_FIRST/LAST.
    r#"CREATE TABLE IF NOT EXISTS ducklake_sort_expression (
        sort_id BIGINT NOT NULL,
        table_id BIGINT NOT NULL,
        sort_key_index BIGINT NOT NULL,
        expression VARCHAR NOT NULL,
        dialect VARCHAR NOT NULL,
        sort_direction VARCHAR NOT NULL,
        null_order VARCHAR NOT NULL
    )"#,
];

/// Single-catalog PostgreSQL metadata writer for standard DuckLake catalogs.
///
/// Use this (not [`crate::metadata_writer_postgres::PostgresMetadataWriter`])
/// when the catalog must be readable and writable by other DuckLake
/// implementations, including DuckDB's `ducklake` extension.
#[derive(Debug, Clone)]
pub struct PostgresSingleCatalogMetadataWriter {
    pool: PgPool,
}

impl PostgresSingleCatalogMetadataWriter {
    /// Open a writer against an existing single-catalog Postgres DuckLake
    /// catalog. Does not create the catalog tables — call
    /// [`MetadataWriter::initialize_schema`] (or use [`Self::new_with_init`])
    /// for a fresh database.
    pub async fn new(connection_string: &str) -> Result<Self> {
        Self::with_max_connections(connection_string, DEFAULT_MAX_CONNECTIONS).await
    }

    /// Open a writer with a bounded connection pool.
    pub async fn with_max_connections(
        connection_string: &str,
        max_connections: u32,
    ) -> Result<Self> {
        let pool = PgPoolOptions::new()
            .max_connections(max_connections)
            .connect(connection_string)
            .await?;
        Ok(Self {
            pool,
        })
    }

    /// Adopt an existing pool (e.g. one shared with a
    /// [`crate::metadata_provider_postgres::PostgresMetadataProvider`]).
    pub fn from_pool(pool: PgPool) -> Self {
        Self {
            pool,
        }
    }

    /// Open a writer and create/upgrade the DuckLake catalog tables.
    pub async fn new_with_init(connection_string: &str) -> Result<Self> {
        let writer = Self::new(connection_string).await?;
        writer.initialize_schema()?;
        Ok(writer)
    }
}

/// Reserve `n` consecutive ids from a counter in `ducklake_metadata`, returning the
/// LAST of the block (`last - n + 1 ..= last`). The `UPDATE` holds an exclusive row
/// lock until commit, so concurrent reservations block rather than overlap.
async fn reserve_ids(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    key: &str,
    n: i64,
) -> Result<i64> {
    let last: i64 = sqlx::query(
        "UPDATE ducklake_metadata
         SET value = CAST(CAST(value AS BIGINT) + $1 AS VARCHAR)
         WHERE key = $2 AND scope IS NULL
         RETURNING CAST(value AS BIGINT)",
    )
    .bind(n)
    .bind(key)
    .fetch_one(&mut **tx)
    .await?
    .try_get(0)?;
    Ok(last)
}

/// Seed an id counter from the current MAX of its backing column, so a pre-existing
/// catalog keeps allocating without reuse. `WHERE NOT EXISTS` makes a re-open a no-op.
async fn seed_counter(pool: &PgPool, key: &str, max_sql: &'static str) -> Result<()> {
    let start: i64 = sqlx::query(max_sql).fetch_one(pool).await?.try_get(0)?;
    sqlx::query(
        "INSERT INTO ducklake_metadata (key, value, scope)
         SELECT $1, $2, NULL
         WHERE NOT EXISTS (
             SELECT 1 FROM ducklake_metadata WHERE key = $1 AND scope IS NULL
         )",
    )
    .bind(key)
    .bind(start.to_string())
    .execute(pool)
    .await?;
    Ok(())
}

/// Abort a `Replace` whose base is stale: any data file newer than `base_snapshot`
/// means another writer published in the meantime. `Append` does not call this —
/// concurrent appends commute.
async fn detect_replace_conflict(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    table_id: i64,
    base_snapshot: i64,
) -> Result<()> {
    let conflict: Option<i64> = sqlx::query(
        "SELECT 1 FROM ducklake_data_file
         WHERE table_id = $1 AND (begin_snapshot > $2 OR end_snapshot > $2)
         LIMIT 1",
    )
    .bind(table_id)
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
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    table_id: i64,
    snapshot_id: i64,
) -> Result<()> {
    sqlx::query(
        "UPDATE ducklake_data_file SET end_snapshot = $1
         WHERE table_id = $2 AND end_snapshot IS NULL AND begin_snapshot < $1",
    )
    .bind(snapshot_id)
    .bind(table_id)
    .execute(&mut **tx)
    .await?;

    sqlx::query(
        "UPDATE ducklake_table_stats SET record_count = 0, file_size_bytes = 0 WHERE table_id = $1",
    )
    .bind(table_id)
    .execute(&mut **tx)
    .await?;
    Ok(())
}

/// Seed the per-table stats row if absent. `ON CONFLICT DO NOTHING` on the
/// `table_id` PK is the Postgres spelling of MySQL's `INSERT IGNORE`.
async fn seed_table_stats(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    table_id: i64,
) -> Result<()> {
    sqlx::query(
        "INSERT INTO ducklake_table_stats (table_id, record_count, next_row_id, file_size_bytes)
         VALUES ($1, 0, 0, 0)
         ON CONFLICT (table_id) DO NOTHING",
    )
    .bind(table_id)
    .execute(&mut **tx)
    .await?;
    Ok(())
}

/// Insert the next `ducklake_snapshot` row, carrying `schema_version` forward.
///
/// Reserving `snapshot_id` takes the counter's row lock, making this the
/// serialization point of a commit — so id order equals commit order, which the
/// `> base_snapshot` conflict test depends on. A DDL commit follows with
/// [`bump_schema_version`].
async fn insert_snapshot(tx: &mut sqlx::Transaction<'_, sqlx::Postgres>) -> Result<(i64, i64)> {
    let snapshot_id = reserve_ids(tx, "next_snapshot_id", 1).await?;
    // Carry the current per-catalog schema_version forward; a DDL commit corrects
    // this to a bump via `bump_schema_version` below. Read before the INSERT so
    // the MAX is over the pre-existing rows only (matches the other writers).
    let schema_version: i64 =
        sqlx::query("SELECT COALESCE(MAX(schema_version), 0) FROM ducklake_snapshot")
            .fetch_one(&mut **tx)
            .await?
            .try_get(0)?;
    sqlx::query(
        "INSERT INTO ducklake_snapshot (snapshot_id, snapshot_time, schema_version)
         VALUES ($1, NOW(), $2)",
    )
    .bind(snapshot_id)
    .bind(schema_version)
    .execute(&mut **tx)
    .await?;
    // Seed the change row so the commit paths can UPDATE it (and so every snapshot
    // has exactly one row, as the spec expects).
    sqlx::query(
        "INSERT INTO ducklake_snapshot_changes (snapshot_id, changes_made)
         VALUES ($1, NULL)",
    )
    .bind(snapshot_id)
    .execute(&mut **tx)
    .await?;
    Ok((snapshot_id, schema_version))
}

/// Append to this snapshot's `changes_made` list and stamp its commit metadata.
/// Appends rather than overwrites: one snapshot can create a schema, a table, and
/// insert rows.
async fn record_snapshot_changes(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    snapshot_id: i64,
    changes_made: &str,
    commit_metadata: &SnapshotCommitMetadata,
) -> Result<()> {
    let changes_made = (!changes_made.is_empty()).then_some(changes_made);
    sqlx::query(
        "UPDATE ducklake_snapshot_changes
         SET changes_made = CASE
                 WHEN changes_made IS NULL THEN $1
                 WHEN $1 IS NULL THEN changes_made
                 ELSE changes_made || ',' || $1
             END,
             author = $2,
             commit_message = $3,
             commit_extra_info = $4
         WHERE snapshot_id = $5",
    )
    .bind(changes_made)
    .bind(commit_metadata.author())
    .bind(commit_metadata.message())
    .bind(commit_metadata.extra_info())
    .bind(snapshot_id)
    .execute(&mut **tx)
    .await?;
    Ok(())
}

/// Record the change summary for a data-write commit: whether this snapshot also
/// created the schema/table or altered it, plus the insert/delete shape from
/// [`table_write_changes`]. Mirrors the MySQL and SQLite writers.
async fn record_table_write_changes(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    snapshot_id: i64,
    table_id: i64,
    schema_name: &str,
    table_name: &str,
    mode: WriteMode,
    commit_metadata: &SnapshotCommitMetadata,
) -> Result<()> {
    let row = sqlx::query(
        "SELECT s.begin_snapshot AS schema_begin_snapshot,
                t.begin_snapshot AS table_begin_snapshot
         FROM ducklake_table t
         JOIN ducklake_schema s ON s.schema_id = t.schema_id
         WHERE t.table_id = $1",
    )
    .bind(table_id)
    .fetch_one(&mut **tx)
    .await?;
    let schema_begin_snapshot: i64 = row.try_get("schema_begin_snapshot")?;
    let table_begin_snapshot: i64 = row.try_get("table_begin_snapshot")?;
    let altered: bool = sqlx::query_scalar(
        "SELECT EXISTS(
            SELECT 1 FROM ducklake_schema_versions
            WHERE table_id = $1 AND begin_snapshot = $2
         )",
    )
    .bind(table_id)
    .bind(snapshot_id)
    .fetch_one(&mut **tx)
    .await?;
    let replaced_existing_data: bool = sqlx::query_scalar(
        "SELECT EXISTS(
            SELECT 1 FROM ducklake_data_file
            WHERE table_id = $1 AND end_snapshot = $2
         )",
    )
    .bind(table_id)
    .bind(snapshot_id)
    .fetch_one(&mut **tx)
    .await?;

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
        false,
        replaced_existing_data,
    ));
    record_snapshot_changes(tx, snapshot_id, &changes.join(","), commit_metadata).await
}

/// Bump the per-catalog monotonic `schema_version` on a DDL snapshot to
/// `prev_max + 1` (max over the OTHER snapshots, so re-running is stable) and
/// return the new value. Mirrors upstream `if (SchemaChangesMade()) schema_version++`.
async fn bump_schema_version(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    snapshot_id: i64,
) -> Result<i64> {
    let prev_max: i64 = sqlx::query(
        "SELECT COALESCE(MAX(schema_version), 0) FROM ducklake_snapshot WHERE snapshot_id <> $1",
    )
    .bind(snapshot_id)
    .fetch_one(&mut **tx)
    .await?
    .try_get(0)?;
    let new_version = prev_max + 1;
    sqlx::query("UPDATE ducklake_snapshot SET schema_version = $1 WHERE snapshot_id = $2")
        .bind(new_version)
        .bind(snapshot_id)
        .execute(&mut **tx)
        .await?;
    Ok(new_version)
}

/// Record a `ducklake_schema_versions` ledger row for a DDL that leaves the table
/// live (create, column add/remove/reorder). Not called for a drop.
async fn record_schema_version(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    snapshot_id: i64,
    schema_version: i64,
    table_id: i64,
) -> Result<()> {
    sqlx::query(
        "INSERT INTO ducklake_schema_versions (begin_snapshot, schema_version, table_id)
         VALUES ($1, $2, $3)",
    )
    .bind(snapshot_id)
    .bind(schema_version)
    .bind(table_id)
    .execute(&mut **tx)
    .await?;
    Ok(())
}

/// Persist the harvested per-column stats for a just-registered data file
/// (per-file zone maps). See the SQLite writer's equivalent for the rationale.
async fn insert_file_column_stats(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    table_id: i64,
    data_file_id: i64,
    column_stats: &[ColumnStat],
) -> Result<()> {
    for stat in column_stats {
        sqlx::query(
            "INSERT INTO ducklake_file_column_stats
                 (data_file_id, table_id, column_id, column_size_bytes,
                  value_count, null_count, min_value, max_value, contains_nan, extra_stats)
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NULL)",
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

/// Persist a partitioned data file's partition metadata within the commit
/// transaction: set `ducklake_data_file.partition_id` and insert one
/// `ducklake_file_partition_value` row per partition key. A no-op for an
/// unpartitioned file.
async fn insert_partition_metadata(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    table_id: i64,
    data_file_id: i64,
    file: &DataFileInfo,
) -> Result<()> {
    if let Some(partition_id) = file.partition_id {
        sqlx::query("UPDATE ducklake_data_file SET partition_id = $1 WHERE data_file_id = $2")
            .bind(partition_id)
            .bind(data_file_id)
            .execute(&mut **tx)
            .await?;
    }
    for (key_index, value) in &file.partition_values {
        sqlx::query(
            "INSERT INTO ducklake_file_partition_value
                 (data_file_id, table_id, partition_key_index, partition_value)
             VALUES ($1, $2, $3, $4)",
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

/// Recompute `ducklake_table_column_stats` from the table's live files and
/// replace the stored rows. See the SQLite writer's equivalent for the rationale.
async fn recompute_table_column_stats(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    table_id: i64,
    columns: &[ColumnDef],
    column_ids: &[i64],
) -> Result<()> {
    use crate::stats_encode::{FileColumnStat, aggregate_global_column_stats};

    let live_file_count: i64 = sqlx::query(
        "SELECT COUNT(*) FROM ducklake_data_file WHERE table_id = $1 AND end_snapshot IS NULL",
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
         WHERE d.table_id = $1 AND d.end_snapshot IS NULL",
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

    sqlx::query("DELETE FROM ducklake_table_column_stats WHERE table_id = $1")
        .bind(table_id)
        .execute(&mut **tx)
        .await?;
    for g in globals {
        sqlx::query(
            "INSERT INTO ducklake_table_column_stats
                 (table_id, column_id, contains_null, contains_nan, min_value, max_value, extra_stats)
             VALUES ($1, $2, $3, $4, $5, $6, NULL)",
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

/// Read the table's live partition generation for the commit-time fence.
async fn live_partition_id(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    table_id: i64,
) -> Result<Option<i64>> {
    Ok(sqlx::query_scalar(
        "SELECT partition_id FROM ducklake_partition_info
         WHERE table_id = $1 AND end_snapshot IS NULL",
    )
    .bind(table_id)
    .fetch_optional(&mut **tx)
    .await?)
}

/// The atomic commit point: insert the deferred snapshot row, finalize the column
/// generation, and for `Replace` retire the prior data generation — all in the
/// caller's transaction, so a reader never sees a half-published head.
///
/// The column generation is deferred to here rather than written at begin because
/// the read path resolves columns by `end_snapshot IS NULL` alone (not
/// snapshot-scoped), so writing them early would leak them to concurrent reads.
async fn finalize_snapshot(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
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
         WHERE table_id = $1 AND end_snapshot IS NULL
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
                "UPDATE ducklake_column SET end_snapshot = $1
                 WHERE table_id = $2 AND column_name = $3 AND end_snapshot IS NULL",
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
                        "UPDATE ducklake_column SET column_order = $1, nulls_allowed = $2
                         WHERE table_id = $3 AND column_name = $4 AND end_snapshot IS NULL",
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
                     VALUES ($1, $2, $3, $4, $5, $6, $7)",
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
        seed_table_stats(tx, table_id).await?;
        retire_prior_generation(tx, table_id, snapshot_id).await?;
    }

    // Record the schema-change ledger row for a DDL commit. A pure data write
    // carries schema_version forward and writes no row.
    if is_ddl {
        record_schema_version(tx, snapshot_id, schema_version, table_id).await?;
    }
    Ok(snapshot_id)
}

impl MetadataWriter for PostgresSingleCatalogMetadataWriter {
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
            // One transaction so the schema row and its change record publish together.
            let mut tx = self.pool.begin().await?;
            let existing = sqlx::query(
                "SELECT schema_id FROM ducklake_schema
                 WHERE schema_name = $1 AND end_snapshot IS NULL",
            )
            .bind(name)
            .fetch_optional(&mut *tx)
            .await?;

            if let Some(row) = existing {
                tx.commit().await?;
                return Ok((row.try_get(0)?, false));
            }

            // Unscoped, relative path: the file layout stays
            // `{data_path}/{schema}/{table}/…`, matching every other
            // single-catalog backend (the multicatalog writer scopes to
            // `cat_{id}/{schema}` instead).
            let schema_path = path.unwrap_or(name);
            let schema_id: i64 = sqlx::query(
                "INSERT INTO ducklake_schema (schema_name, path, path_is_relative, begin_snapshot)
                 VALUES ($1, $2, TRUE, $3) RETURNING schema_id",
            )
            .bind(name)
            .bind(schema_path)
            .bind(snapshot_id)
            .fetch_one(&mut *tx)
            .await?
            .try_get(0)?;

            record_snapshot_changes(
                &mut tx,
                snapshot_id,
                &format!("created_schema:{}", quote_snapshot_name(name)),
                &SnapshotCommitMetadata::default(),
            )
            .await?;
            tx.commit().await?;
            Ok((schema_id, true))
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
                 WHERE schema_id = $1 AND table_name = $2 AND end_snapshot IS NULL",
            )
            .bind(schema_id)
            .bind(name)
            .fetch_optional(&mut *tx)
            .await?;

            if let Some(row) = existing {
                tx.commit().await?;
                return Ok((row.try_get(0)?, false));
            }

            // Needed for the qualified name in the change record.
            let schema_name: String =
                sqlx::query_scalar("SELECT schema_name FROM ducklake_schema WHERE schema_id = $1")
                    .bind(schema_id)
                    .fetch_one(&mut *tx)
                    .await?;

            let table_path = path.unwrap_or(name);
            let table_id: i64 = sqlx::query(
                "INSERT INTO ducklake_table (schema_id, table_name, path, path_is_relative, begin_snapshot)
                 VALUES ($1, $2, $3, TRUE, $4) RETURNING table_id",
            )
            .bind(schema_id)
            .bind(name)
            .bind(table_path)
            .bind(snapshot_id)
            .fetch_one(&mut *tx)
            .await?
            .try_get(0)?;

            record_snapshot_changes(
                &mut tx,
                snapshot_id,
                &format!("created_table:{}", quote_snapshot_table(&schema_name, name)),
                &SnapshotCommitMetadata::default(),
            )
            .await?;
            tx.commit().await?;
            Ok((table_id, true))
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
                "UPDATE ducklake_column SET end_snapshot = $1
                 WHERE table_id = $2 AND end_snapshot IS NULL",
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
                     VALUES ($1, $2, $3, $4, $5, $6, $7)",
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

            // A column set on an already-existing table is an ALTER; on a table
            // created in this same snapshot it is part of the create, already
            // recorded by get_or_create_table.
            let table_begin_snapshot: i64 =
                sqlx::query_scalar("SELECT begin_snapshot FROM ducklake_table WHERE table_id = $1")
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
            Ok(column_ids)
        })
    }

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

    #[allow(clippy::too_many_arguments)]
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
                "conditional writes are not supported by the single-catalog Postgres metadata \
                 writer"
                    .to_string(),
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
            let partition_generation = live_partition_id(&mut tx, table_id).await?;
            crate::metadata_writer::enforce_partition_fence(table_id, partition_generation, file)?;

            // Seed the stats row for the Append path (Replace already seeded it in
            // finalize_snapshot); ON CONFLICT DO NOTHING is a no-op if it exists.
            seed_table_stats(&mut tx, table_id).await?;

            let row_id_start: i64 =
                sqlx::query("SELECT next_row_id FROM ducklake_table_stats WHERE table_id = $1")
                    .bind(table_id)
                    .fetch_one(&mut *tx)
                    .await?
                    .try_get(0)?;

            let data_file_id: i64 = sqlx::query(
                "INSERT INTO ducklake_data_file
                     (table_id, path, path_is_relative, file_size_bytes,
                      footer_size, record_count, row_id_start, begin_snapshot)
                 VALUES ($1, $2, $3, $4, $5, $6, $7, $8) RETURNING data_file_id",
            )
            .bind(table_id)
            .bind(&file.path)
            .bind(file.path_is_relative)
            .bind(file.file_size_bytes)
            .bind(file.footer_size)
            .bind(file.record_count)
            .bind(row_id_start)
            .bind(snapshot_id)
            .fetch_one(&mut *tx)
            .await?
            .try_get(0)?;

            // Persist the file's zone maps + refresh the roll-up.
            insert_file_column_stats(&mut tx, table_id, data_file_id, &file.column_stats).await?;
            insert_partition_metadata(&mut tx, table_id, data_file_id, file).await?;
            recompute_table_column_stats(&mut tx, table_id, columns, column_ids).await?;

            // Advance the counter and accumulate stats. `next_row_id`
            // monotonically increases over the table's lifetime.
            sqlx::query(
                "UPDATE ducklake_table_stats
                 SET next_row_id     = next_row_id + $1,
                     record_count    = record_count + $1,
                     file_size_bytes = file_size_bytes + $2
                 WHERE table_id = $3",
            )
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
                commit_metadata,
            )
            .await?;

            let schema_id: i64 =
                sqlx::query("SELECT schema_id FROM ducklake_table WHERE table_id = $1")
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

    #[allow(clippy::too_many_arguments)]
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
                "conditional multi-file writes are not supported by the single-catalog Postgres \
                 metadata writer"
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
            let partition_generation = live_partition_id(&mut tx, table_id).await?;
            for file in files {
                crate::metadata_writer::enforce_partition_fence(
                    table_id,
                    partition_generation,
                    file,
                )?;
            }
            seed_table_stats(&mut tx, table_id).await?;
            let mut next_row_id: i64 =
                sqlx::query("SELECT next_row_id FROM ducklake_table_stats WHERE table_id = $1")
                    .bind(table_id)
                    .fetch_one(&mut *tx)
                    .await?
                    .try_get(0)?;
            let mut total_records: i64 = 0;
            let mut total_bytes: i64 = 0;
            for file in files {
                let data_file_id: i64 = sqlx::query(
                    "INSERT INTO ducklake_data_file
                         (table_id, path, path_is_relative, file_size_bytes,
                          footer_size, record_count, row_id_start, begin_snapshot)
                     VALUES ($1, $2, $3, $4, $5, $6, $7, $8) RETURNING data_file_id",
                )
                .bind(table_id)
                .bind(&file.path)
                .bind(file.path_is_relative)
                .bind(file.file_size_bytes)
                .bind(file.footer_size)
                .bind(file.record_count)
                .bind(next_row_id)
                .bind(snapshot_id)
                .fetch_one(&mut *tx)
                .await?
                .try_get(0)?;
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
                 SET next_row_id     = next_row_id + $1,
                     record_count    = record_count + $1,
                     file_size_bytes = file_size_bytes + $2
                 WHERE table_id = $3",
            )
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
                commit_metadata,
            )
            .await?;
            let schema_id: i64 =
                sqlx::query("SELECT schema_id FROM ducklake_table WHERE table_id = $1")
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

            let mut resolved_column_ids: Vec<i64> = Vec::with_capacity(columns.len());
            for (name, _transform) in columns {
                let column_id: i64 = sqlx::query_scalar(
                    "SELECT column_id FROM ducklake_column
                     WHERE table_id = $1 AND column_name = $2 AND end_snapshot IS NULL",
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
                resolved_column_ids.push(column_id);
            }

            sqlx::query(
                "UPDATE ducklake_partition_info SET end_snapshot = $1
                 WHERE table_id = $2 AND end_snapshot IS NULL",
            )
            .bind(new_snapshot)
            .bind(table_id)
            .execute(&mut *tx)
            .await?;
            sqlx::query(
                "INSERT INTO ducklake_partition_info
                     (partition_id, table_id, begin_snapshot, end_snapshot)
                 VALUES ($1, $2, $3, NULL)",
            )
            .bind(partition_id)
            .bind(table_id)
            .bind(new_snapshot)
            .execute(&mut *tx)
            .await?;
            for (key_index, column_id) in resolved_column_ids.iter().enumerate() {
                sqlx::query(
                    "INSERT INTO ducklake_partition_column
                         (partition_id, table_id, partition_key_index, column_id, transform)
                     VALUES ($1, $2, $3, $4, $5)",
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
                 WHERE pi.table_id = $1 AND pi.end_snapshot IS NULL
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
                "UPDATE ducklake_partition_info SET end_snapshot = $1
                 WHERE table_id = $2 AND end_snapshot IS NULL",
            )
            .bind(new_snapshot)
            .bind(table_id)
            .execute(&mut *tx)
            .await?
            .rows_affected();
            if ended == 0 {
                // Nothing to reset: roll back the speculative snapshot rather than
                // publishing an empty one, and report the unchanged head.
                tx.rollback().await?;
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
                 WHERE si.table_id = $1 AND si.end_snapshot IS NULL
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
                     WHERE table_id = $1 AND column_name = $2 AND end_snapshot IS NULL",
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
                "UPDATE ducklake_sort_info SET end_snapshot = $1
                 WHERE table_id = $2 AND end_snapshot IS NULL",
            )
            .bind(new_snapshot)
            .bind(table_id)
            .execute(&mut *tx)
            .await?;
            sqlx::query(
                "INSERT INTO ducklake_sort_info
                     (sort_id, table_id, begin_snapshot, end_snapshot)
                 VALUES ($1, $2, $3, NULL)",
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
                     VALUES ($1, $2, $3, $4, $5, $6, $7)",
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
                "UPDATE ducklake_sort_info SET end_snapshot = $1
                 WHERE table_id = $2 AND end_snapshot IS NULL",
            )
            .bind(new_snapshot)
            .bind(table_id)
            .execute(&mut *tx)
            .await?
            .rows_affected();
            if ended == 0 {
                tx.rollback().await?;
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

    #[allow(clippy::too_many_arguments)]
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
        // Fileless commit point (CREATE TABLE, zero-row Replace). This writer
        // defers the snapshot-row insert out of begin_write_transaction, so the
        // trait's default no-op is insufficient: insert the deferred snapshot row +
        // column generation and, for Replace, retire the prior generation — making
        // the new head visible atomically.
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
                &SnapshotCommitMetadata::default(),
            )
            .await?;
            let schema_id: i64 =
                sqlx::query("SELECT schema_id FROM ducklake_table WHERE table_id = $1")
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
                "UPDATE ducklake_data_file SET end_snapshot = $1
                 WHERE table_id = $2 AND end_snapshot IS NULL",
            )
            .bind(snapshot_id)
            .bind(table_id)
            .execute(&mut *tx)
            .await?;

            sqlx::query(
                "UPDATE ducklake_table_stats
                 SET record_count = 0, file_size_bytes = 0
                 WHERE table_id = $1",
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
            let row =
                sqlx::query("SELECT value FROM ducklake_metadata WHERE key = $1 AND scope IS NULL")
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
            let mut tx = self.pool.begin().await?;
            sqlx::query("DELETE FROM ducklake_metadata WHERE key = 'data_path' AND scope IS NULL")
                .execute(&mut *tx)
                .await?;

            sqlx::query(
                "INSERT INTO ducklake_metadata (key, value, scope)
                 VALUES ('data_path', $1, NULL)",
            )
            .bind(path)
            .execute(&mut *tx)
            .await?;

            tx.commit().await?;
            Ok(())
        })
    }

    fn initialize_schema(&self) -> Result<()> {
        block_on(async {
            // sqlx runs each query() as a single prepared statement, so create each
            // table separately (see SQL_CREATE_TABLES).
            for ddl in SQL_CREATE_TABLES {
                sqlx::query(*ddl).execute(&self.pool).await?;
            }
            // Upgrade a pre-existing catalog to carry ducklake_data_file.partition_id
            // (idempotent, lossless — NULL means "not partitioned").
            sqlx::query(
                "ALTER TABLE ducklake_data_file ADD COLUMN IF NOT EXISTS partition_id BIGINT",
            )
            .execute(&self.pool)
            .await?;
            // A catalog created elsewhere may have `changes_made` NOT NULL, but the
            // row is seeded NULL and filled in at commit. Dropping an absent NOT
            // NULL is a no-op in Postgres, so no probe is needed.
            sqlx::query(
                "ALTER TABLE ducklake_snapshot_changes ALTER COLUMN changes_made DROP NOT NULL",
            )
            .execute(&self.pool)
            .await?;
            // snapshot_id, column_id, partition_id and sort_id have no IDENTITY —
            // they are reserved inside a transaction from these counters.
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
                     WHERE schema_name = $1 AND end_snapshot IS NULL",
                )
                .bind(schema_name)
                .fetch_optional(&mut *tx)
                .await?;

                if let Some(row) = existing {
                    row.try_get(0)?
                } else {
                    // Unscoped relative path — see get_or_create_schema.
                    sqlx::query(
                        "INSERT INTO ducklake_schema (schema_name, path, path_is_relative, begin_snapshot)
                         VALUES ($1, $2, TRUE, $3) RETURNING schema_id",
                    )
                    .bind(schema_name)
                    .bind(schema_name)
                    .bind(snapshot_id)
                    .fetch_one(&mut *tx)
                    .await?
                    .try_get(0)?
                }
            };

            let table_id: i64 = {
                let existing = sqlx::query(
                    "SELECT table_id FROM ducklake_table
                     WHERE schema_id = $1 AND table_name = $2 AND end_snapshot IS NULL",
                )
                .bind(schema_id)
                .bind(table_name)
                .fetch_optional(&mut *tx)
                .await?;

                if let Some(row) = existing {
                    row.try_get(0)?
                } else {
                    sqlx::query(
                        "INSERT INTO ducklake_table (schema_id, table_name, path, path_is_relative, begin_snapshot)
                         VALUES ($1, $2, $3, TRUE, $4) RETURNING table_id",
                    )
                    .bind(schema_id)
                    .bind(table_name)
                    .bind(table_name)
                    .bind(snapshot_id)
                    .fetch_one(&mut *tx)
                    .await?
                    .try_get(0)?
                }
            };

            // Get existing columns to (a) check schema compatibility for appends
            // and (b) REUSE each column's id (column_id == parquet field_id; an
            // unchanged column must keep its id, or files already written would
            // read back as NULL).
            let rows = sqlx::query(
                "SELECT column_name, column_type, nulls_allowed, column_id
                 FROM ducklake_column
                 WHERE table_id = $1 AND end_snapshot IS NULL
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

            // A data write must not change a column's type — that is schema
            // evolution, and this backend has no promote_column_type. The comparison
            // is canonical (`int64` ≡ `bigint`). Append also requires a new column
            // to be nullable.
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

            // Reuse an existing column's id, consume a fresh one only for a new
            // column. These are baked into the staged parquet's field_id metadata,
            // so they must equal what finalize_snapshot commits.
            let column_ids: Vec<i64> = columns
                .iter()
                .zip(fresh_ids.iter())
                .map(|(col, &fresh)| existing_ids.get(col.name()).copied().unwrap_or(fresh))
                .collect();

            // Commits only the idempotent get-or-create schema/table rows. Those
            // reads ARE snapshot-scoped, so the rows stay invisible until the
            // snapshot publishes; everything else is deferred to the atomic commit.
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
