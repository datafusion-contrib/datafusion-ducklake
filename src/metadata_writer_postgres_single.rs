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
use crate::metadata_provider::{TagObjectType, TagTarget};
use crate::metadata_writer::{
    ColumnDef, ColumnStat, CommitIds, ConflictScope, DataFileInfo, ExistingCatalogColumn,
    MetadataWriter, MultiTableCommit, ReservedTables, SnapshotCommitMetadata, StagedTableData,
    StagedTableWrite, StoredTablePaths, WriteMode, WriteSetupResult, assign_column_ids,
    catalog_column_defs, catalog_column_type_equal, catalog_column_type_requires_migration,
    catalog_columns_differ, quote_snapshot_name, quote_snapshot_table, snapshot_has_change,
    staged_table_write_changes, table_write_changes, tag_change, top_level_column_ids,
    validate_name,
};
use crate::metadata_writer::{
    InlinedFlushTable, LEGACY_CATALOG_ID_COUNTERS, ReservedTableId, inlined_flush_changes,
    inlined_inserts, positional_columns, refuse_changed_table_id, reserved_id_taken,
    validate_inlined_flush,
};
use crate::metadata_writer::{created_name, directory_path, table_directory};
use crate::metadata_writer_postgres::apply_inlined_flush_at_snapshot;
use crate::metadata_writer_postgres::{
    PostgresCommitGuard, SQL_CREATE_INLINED_DATA_TABLES, apply_inlined_deletes_at_snapshot,
    apply_positional_deletes_at_snapshot, apply_row_mutation_at_snapshot, check_staged_conflicts,
    commit_files_at_snapshot, commit_guarded, commit_inlined_at_snapshot, commit_networked_guarded,
    detect_inlined_delete_since, detect_replace_conflict as detect_staged_conflict,
    detect_row_mutation_conflicts, drop_empty_superseded_inlined_tables_in, quote_ident,
    set_postgres_table_setting, validate_staged_table, with_postgres_commit_lock,
};
use crate::partition::PartitionTransform;
use sqlx::postgres::{PgPool, PgPoolOptions};
use sqlx::{AssertSqlSafe, Row};
use std::sync::Arc;

const DEFAULT_MAX_CONNECTIONS: u32 = 5;

/// The DuckLake catalog tables in Postgres dialect. Columns and their order match
/// the SQLite and MySQL writers (and so upstream); only the SQL types differ. Split
/// one per entry because sqlx runs each `query()` as a single prepared statement.
const SQL_CREATE_TABLES: &[&str] = &[
    SQL_CREATE_INLINED_DATA_TABLES,
    r#"CREATE TABLE IF NOT EXISTS ducklake_metadata (
        key VARCHAR NOT NULL,
        value VARCHAR NOT NULL,
        scope VARCHAR,
        scope_id BIGINT
    )"#,
    r#"CREATE TABLE IF NOT EXISTS ducklake_snapshot (
        snapshot_id BIGINT NOT NULL PRIMARY KEY,
        snapshot_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        schema_version BIGINT NOT NULL DEFAULT 0,
        next_catalog_id BIGINT,
        next_file_id BIGINT
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
    // Bare table (no PRIMARY KEY), mirroring upstream: a versioned column needs
    // several rows sharing one `column_id`. The `*default*` columns and
    // `parent_column` are left NULL.
    r#"CREATE TABLE IF NOT EXISTS ducklake_view (
        view_id BIGINT,
        view_uuid UUID,
        begin_snapshot BIGINT,
        end_snapshot BIGINT,
        schema_id BIGINT,
        view_name VARCHAR,
        dialect VARCHAR,
        sql VARCHAR,
        column_aliases VARCHAR
    )"#,
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
    // The partition pre-filter (`crate::stats_filter`) reads this table once per
    // listing, restricted to one table's rows for one partition key. Unindexed
    // that is a scan of every table's partition values. Measured on PostgreSQL 18
    // over 409,600 rows (50 tables x 8,192 files), 72 ms without this index and
    // 6.8 ms with it; an independent rebuild of the same shape measured 79 ms and
    // 25 ms. The magnitude is fixture-bound, the direction is not, and either
    // index shape wins — the value column earns nothing here, because the
    // anti-join reads every row for the key regardless.
    // Both columns are fixed-width, deliberately — `partition_value`
    // is unbounded text, and indexing it would put a value's length between a
    // partitioned INSERT and success.
    r#"CREATE INDEX IF NOT EXISTS idx_file_partition_value_table_key
        ON ducklake_file_partition_value (table_id, partition_key_index)"#,
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
    // Tags on objects (tables and views) and on columns (DuckLake spec). A NULL
    // value is a live tag, which is how `COMMENT ON ... IS NULL` is stored.
    r#"CREATE TABLE IF NOT EXISTS ducklake_tag (
        object_id BIGINT,
        begin_snapshot BIGINT,
        end_snapshot BIGINT,
        key VARCHAR,
        value VARCHAR
    )"#,
    r#"CREATE TABLE IF NOT EXISTS ducklake_column_tag (
        table_id BIGINT,
        column_id BIGINT,
        begin_snapshot BIGINT,
        end_snapshot BIGINT,
        key VARCHAR,
        value VARCHAR
    )"#,
    // Tables this crate does not write, which official DuckDB reads when it
    // attaches.
    r#"CREATE TABLE IF NOT EXISTS ducklake_column_mapping (
        mapping_id BIGINT,
        table_id BIGINT,
        type VARCHAR
    )"#,
    r#"CREATE TABLE IF NOT EXISTS ducklake_name_mapping (
        mapping_id BIGINT,
        column_id BIGINT,
        source_name VARCHAR,
        target_field_id BIGINT,
        parent_column BIGINT,
        is_partition BOOLEAN
    )"#,
    r#"CREATE TABLE IF NOT EXISTS ducklake_file_variant_stats (
        data_file_id BIGINT,
        table_id BIGINT,
        column_id BIGINT,
        variant_path VARCHAR,
        shredded_type VARCHAR,
        column_size_bytes BIGINT,
        value_count BIGINT,
        null_count BIGINT,
        min_value VARCHAR,
        max_value VARCHAR,
        contains_nan BOOLEAN,
        extra_stats VARCHAR
    )"#,
    r#"CREATE TABLE IF NOT EXISTS ducklake_macro (
        schema_id BIGINT,
        macro_id BIGINT,
        macro_name VARCHAR,
        begin_snapshot BIGINT,
        end_snapshot BIGINT
    )"#,
    r#"CREATE TABLE IF NOT EXISTS ducklake_macro_impl (
        macro_id BIGINT,
        impl_id BIGINT,
        dialect VARCHAR,
        sql VARCHAR,
        type VARCHAR
    )"#,
    r#"CREATE TABLE IF NOT EXISTS ducklake_macro_parameters (
        macro_id BIGINT,
        impl_id BIGINT,
        column_id BIGINT,
        parameter_name VARCHAR,
        parameter_type VARCHAR,
        default_value VARCHAR,
        default_value_type VARCHAR
    )"#,
];

/// The catalog tables official DuckDB inserts into by position, with their
/// columns, keys and order as official has them. Schema and table ids default
/// from an identity sequence, which a release before 0.10 takes them from,
/// and file ids from [`SQL_CREATE_NEXT_FILE_ID`]; `BY DEFAULT` lets official
/// insert its own. `ducklake_table`
/// has no key: official keeps one row per version of a table, a rename's
/// included. A catalog this writer created before it followed them keeps its
/// layout, which a release before 0.10 still writes, until
/// [`PostgresSingleCatalogMetadataWriter::migrate_to_official_layout`] rebuilds
/// it.
const POSITIONAL_TABLES: [(&str, &str, &str); 4] = [
    (
        "ducklake_schema",
        "schema_id",
        "schema_id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
        schema_uuid UUID,
        begin_snapshot BIGINT NOT NULL,
        end_snapshot BIGINT,
        schema_name VARCHAR NOT NULL,
        path VARCHAR NOT NULL DEFAULT '',
        path_is_relative BOOLEAN NOT NULL DEFAULT TRUE",
    ),
    (
        "ducklake_table",
        "table_id",
        "table_id BIGINT GENERATED BY DEFAULT AS IDENTITY,
        table_uuid UUID,
        begin_snapshot BIGINT NOT NULL,
        end_snapshot BIGINT,
        schema_id BIGINT NOT NULL,
        table_name VARCHAR NOT NULL,
        path VARCHAR NOT NULL DEFAULT '',
        path_is_relative BOOLEAN NOT NULL DEFAULT TRUE",
    ),
    (
        "ducklake_data_file",
        "data_file_id",
        "data_file_id BIGINT PRIMARY KEY DEFAULT ducklake_next_file_id(),
        table_id BIGINT NOT NULL,
        begin_snapshot BIGINT NOT NULL,
        end_snapshot BIGINT,
        file_order BIGINT,
        path VARCHAR NOT NULL,
        path_is_relative BOOLEAN NOT NULL DEFAULT TRUE,
        file_format VARCHAR DEFAULT 'parquet',
        record_count BIGINT,
        file_size_bytes BIGINT NOT NULL,
        footer_size BIGINT,
        row_id_start BIGINT,
        partition_id BIGINT,
        encryption_key VARCHAR,
        mapping_id BIGINT,
        partial_max BIGINT",
    ),
    (
        "ducklake_delete_file",
        "delete_file_id",
        "delete_file_id BIGINT PRIMARY KEY DEFAULT ducklake_next_file_id(),
        table_id BIGINT NOT NULL,
        begin_snapshot BIGINT NOT NULL,
        end_snapshot BIGINT,
        data_file_id BIGINT NOT NULL,
        path VARCHAR NOT NULL,
        path_is_relative BOOLEAN NOT NULL DEFAULT TRUE,
        format VARCHAR DEFAULT 'parquet',
        delete_count BIGINT,
        file_size_bytes BIGINT NOT NULL,
        footer_size BIGINT,
        encryption_key VARCHAR,
        partial_max BIGINT",
    ),
];

/// `ducklake_next_file_id()`, the default of a data or delete file id: the next
/// id of the `next_file_id` counter, past every file id in use and below the
/// head snapshot's watermark, as official DuckDB allocates file ids. A writer
/// needs only `UPDATE` on `ducklake_metadata` for it, where an identity
/// sequence it had to advance past official's ids would take `UPDATE` on the
/// sequence. PL/pgSQL, whose body is not checked against tables created after
/// it.
const SQL_CREATE_NEXT_FILE_ID: &str = "CREATE FUNCTION ducklake_next_file_id() RETURNS BIGINT
LANGUAGE plpgsql AS $next_file_id$
DECLARE
    next_id BIGINT;
BEGIN
    UPDATE ducklake_metadata
    SET value = CAST(GREATEST(CAST(value AS BIGINT),
        COALESCE((SELECT MAX(data_file_id) FROM ducklake_data_file), -1),
        COALESCE((SELECT MAX(delete_file_id) FROM ducklake_delete_file), -1),
        COALESCE((SELECT next_file_id FROM ducklake_snapshot
                  ORDER BY snapshot_id DESC LIMIT 1) - 1, -1)) + 1 AS VARCHAR)
    WHERE key = 'next_file_id' AND scope IS NULL
    RETURNING CAST(value AS BIGINT) INTO next_id;
    RETURN next_id;
END
$next_file_id$";

/// The largest schema, table, view, partition, sort or macro id in use, `-1`
/// in an empty catalog.
const SQL_MAX_CATALOG_ID: &str = "SELECT GREATEST(
     COALESCE((SELECT MAX(schema_id) FROM ducklake_schema), -1),
     COALESCE((SELECT MAX(table_id) FROM ducklake_table), -1),
     COALESCE((SELECT MAX(view_id) FROM ducklake_view), -1),
     COALESCE((SELECT MAX(partition_id) FROM ducklake_partition_info), -1),
     COALESCE((SELECT MAX(sort_id) FROM ducklake_sort_info), -1),
     COALESCE((SELECT MAX(macro_id) FROM ducklake_macro), -1))";

/// The largest data or delete file id in use, `-1` in an empty catalog.
const SQL_MAX_FILE_ID: &str = "SELECT GREATEST(
     COALESCE((SELECT MAX(data_file_id) FROM ducklake_data_file), -1),
     COALESCE((SELECT MAX(delete_file_id) FROM ducklake_delete_file), -1))";

/// Single-catalog PostgreSQL metadata writer for standard DuckLake catalogs.
///
/// Use this (not [`crate::metadata_writer_postgres::PostgresMetadataWriter`])
/// when the catalog must be readable and writable by other DuckLake
/// implementations, including DuckDB's `ducklake` extension.
#[derive(Debug, Clone)]
pub struct PostgresSingleCatalogMetadataWriter {
    pool: PgPool,
    commit_guard: Option<Arc<dyn PostgresCommitGuard>>,
    data_inlining_row_limit: Option<usize>,
    reserved_tables: ReservedTables,
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
        let url = connection_string.to_string();
        let pool = crate::metadata_provider::connect_on_catalog_runtime(async move {
            PgPoolOptions::new()
                .max_connections(max_connections)
                .connect(&url)
                .await
        })
        .await?;
        upgrade_earlier_catalog(&pool).await?;
        Ok(Self {
            pool,
            commit_guard: None,
            data_inlining_row_limit: None,
            reserved_tables: ReservedTables::default(),
        })
    }

    /// Adopt an existing pool (e.g. one shared with a
    /// [`crate::metadata_provider_postgres::PostgresMetadataProvider`]),
    /// bringing a catalog an earlier release created up to what this writer
    /// uses, as [`Self::new`] does.
    pub async fn from_pool(pool: PgPool) -> Result<Self> {
        upgrade_earlier_catalog(&pool).await?;
        Ok(Self {
            pool,
            commit_guard: None,
            data_inlining_row_limit: None,
            reserved_tables: ReservedTables::default(),
        })
    }

    /// Open a writer and create/upgrade the DuckLake catalog tables.
    pub async fn new_with_init(connection_string: &str) -> Result<Self> {
        let writer = Self::new(connection_string).await?;
        writer.initialize_schema()?;
        Ok(writer)
    }

    /// Rebuild the catalog tables official DuckDB inserts into by position
    /// into official's layout, so official DuckDB can write the catalog; one
    /// this writer created already has it. A release before 0.10 does not keep
    /// the id watermarks official allocates from, so migrate once every writer
    /// of the catalog runs 0.10 or later. Idempotent.
    pub fn migrate_to_official_layout(&self) -> Result<()> {
        self.initialize_schema()?;
        self.block_on(order_positional_tables(&self.pool))
    }
    /// Runs `guard` inside every metadata transaction this writer commits, right
    /// before `COMMIT`; see [`PostgresCommitGuard`].
    pub fn with_commit_guard(mut self, guard: Arc<dyn PostgresCommitGuard>) -> Self {
        self.commit_guard = Some(guard);
        self
    }

    /// Set the `data_inlining_row_limit` this writer applies where the
    /// catalog stores none at table, schema or catalog scope, as official's
    /// `DATA_INLINING_ROW_LIMIT` attach option does; see
    /// [`MetadataWriter::data_inlining_row_limit`]. `0` stores every write in
    /// Parquet and creates no inlined data tables.
    #[must_use]
    pub fn with_data_inlining_row_limit(mut self, limit: usize) -> Self {
        self.data_inlining_row_limit = Some(limit);
        self
    }

    /// Drive `future`, a catalog call of this writer, with its
    /// `data_inlining_row_limit` in effect for the call.
    fn block_on<F: std::future::Future>(&self, future: F) -> F::Output {
        crate::metadata_writer::with_writer_inlining_limit(self.data_inlining_row_limit, || {
            crate::metadata_provider::block_on(future)
        })
    }

    async fn commit(&self, mut tx: sqlx::Transaction<'_, sqlx::Postgres>) -> Result<()> {
        raise_head_watermarks(&mut tx).await?;
        commit_guarded(tx, self.commit_guard.as_deref()).await
    }

    async fn commit_networked(&self, mut tx: sqlx::Transaction<'_, sqlx::Postgres>) -> Result<()> {
        raise_head_watermarks(&mut tx).await?;
        commit_networked_guarded(tx, self.commit_guard.as_deref()).await
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

/// Raise the `next_column_id` counter, which `n` column ids were just reserved
/// from, past `table_id`'s columns, returning its last id. Official numbers a
/// column it adds from the table's largest column id, never from this counter.
async fn raise_column_ids_past_table(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    table_id: i64,
    n: i64,
) -> Result<i64> {
    Ok(sqlx::query_scalar(
        "UPDATE ducklake_metadata
         SET value = CAST(GREATEST(CAST(value AS BIGINT),
                                   COALESCE((SELECT MAX(column_id) FROM ducklake_column
                                             WHERE table_id = $1), 0) + $2) AS VARCHAR)
         WHERE key = 'next_column_id' AND scope IS NULL
         RETURNING CAST(value AS BIGINT)",
    )
    .bind(table_id)
    .bind(n)
    .fetch_one(&mut **tx)
    .await?)
}

/// Advance the identity sequence of `table.column` past every id in the
/// table and below the head snapshot's `watermark` (`next_catalog_id` or
/// `next_file_id`), which official DuckDB allocates its ids from, so the
/// sequence never hands out one official used, on a catalog a release before
/// 0.10 laid out, whose ids default from one. Advancing it takes `UPDATE` on
/// it, and a role without that gets an error naming the grant; a catalog in
/// official's layout has no such sequence ([`SQL_CREATE_NEXT_FILE_ID`]).
async fn skip_official_ids(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    table: &str,
    column: &str,
    watermark: &str,
) -> Result<()> {
    let behind: Option<(String, i64, bool)> = sqlx::query_as(AssertSqlSafe(format!(
        "SELECT sequence, target, has_sequence_privilege(sequence, 'UPDATE')
         FROM (SELECT pg_get_serial_sequence($1, $2) AS sequence,
                      GREATEST(1, COALESCE((SELECT MAX({column}) FROM {table}), 0),
                               COALESCE((SELECT {watermark} FROM ducklake_snapshot
                                         ORDER BY snapshot_id DESC LIMIT 1) - 1, 0))
                          AS target) AS skip
         WHERE sequence IS NOT NULL
           AND target > COALESCE(pg_sequence_last_value(sequence::regclass), 0)"
    )))
    .bind(table)
    .bind(column)
    .fetch_optional(&mut **tx)
    .await?;
    match behind {
        None => Ok(()),
        Some((sequence, target, true)) => {
            sqlx::query("SELECT setval($1::regclass, $2)")
                .bind(&sequence)
                .bind(target)
                .execute(&mut **tx)
                .await?;
            Ok(())
        },
        Some((sequence, ..)) => Err(crate::DuckLakeError::InvalidConfig(format!(
            "official DuckDB allocated ids past {sequence}, which this writer takes {column} \
             values from; it must advance the sequence, so grant UPDATE ON SEQUENCE {sequence} \
             to its role"
        ))),
    }
}

/// Take the next schema, table, partition or sort id from the
/// `next_catalog_id` counter official DuckDB allocates the same ids from,
/// without inserting a row. The id is past every id in use, every id official
/// allocated (those below the head snapshot's watermark) and the
/// [`LEGACY_CATALOG_ID_COUNTERS`] a release before 0.10 took the same ids from.
/// Those counters, and the identity sequences that release took schema and
/// table ids from when this writer's role may update them, are raised to it,
/// so it never takes the id either.
async fn reserve_catalog_id(tx: &mut sqlx::Transaction<'_, sqlx::Postgres>) -> Result<i64> {
    let id: i64 = sqlx::query_scalar(AssertSqlSafe(format!(
        "UPDATE ducklake_metadata
         SET value = CAST(GREATEST(CAST(value AS BIGINT),
             COALESCE((SELECT next_catalog_id FROM ducklake_snapshot
                       ORDER BY snapshot_id DESC LIMIT 1) - 1, -1),
             ({SQL_MAX_CATALOG_ID}),
             COALESCE((SELECT MAX(CAST(value AS BIGINT)) FROM ducklake_metadata
                       WHERE key IN {LEGACY_CATALOG_ID_COUNTERS} AND scope IS NULL), -1))
             + 1 AS VARCHAR)
         WHERE key = 'next_catalog_id' AND scope IS NULL
         RETURNING CAST(value AS BIGINT)"
    )))
    .fetch_one(&mut **tx)
    .await?;
    sqlx::query(AssertSqlSafe(format!(
        "UPDATE ducklake_metadata SET value = CAST($1 AS VARCHAR)
         WHERE key IN {LEGACY_CATALOG_ID_COUNTERS} AND scope IS NULL"
    )))
    .bind(id)
    .execute(&mut **tx)
    .await?;
    // Only a release before 0.10 takes ids from them, so a role without
    // `UPDATE` on them leaves them, as on a catalog no such release writes.
    sqlx::query(
        "SELECT setval(sequence, $1)
         FROM (SELECT pg_get_serial_sequence(name, id) AS sequence
               FROM (VALUES ('ducklake_schema', 'schema_id'), ('ducklake_table', 'table_id'))
                   AS legacy(name, id)) AS legacy
         WHERE sequence IS NOT NULL
           AND $1 > COALESCE(pg_sequence_last_value(sequence::regclass), 0)
           AND has_sequence_privilege(sequence, 'UPDATE')",
    )
    .bind(id)
    .execute(&mut **tx)
    .await?;
    Ok(id)
}

/// Raise the head snapshot's `next_catalog_id` and `next_file_id` past every
/// id in use, so official DuckDB, which allocates from them, never reuses one.
/// Runs inside each commit, after its rows are written.
async fn raise_head_watermarks(tx: &mut sqlx::Transaction<'_, sqlx::Postgres>) -> Result<()> {
    sqlx::query(AssertSqlSafe(format!(
        "UPDATE ducklake_snapshot
         SET next_catalog_id = GREATEST(COALESCE(next_catalog_id, 0), catalog + 1),
             next_file_id = GREATEST(COALESCE(next_file_id, 0), file + 1)
         FROM ({SQL_MAX_CATALOG_ID}) AS catalog(catalog), ({SQL_MAX_FILE_ID}) AS file(file)
         WHERE snapshot_id = (SELECT MAX(snapshot_id) FROM ducklake_snapshot)
           AND (next_catalog_id IS NULL OR next_catalog_id <= catalog
                OR next_file_id IS NULL OR next_file_id <= file)"
    )))
    .execute(&mut **tx)
    .await?;
    Ok(())
}

/// Bring a catalog an earlier release created up to the tables, columns and
/// counters this writer uses, even when the caller skips `initialize_schema`.
/// A database without a catalog is left to `initialize_schema`.
async fn upgrade_earlier_catalog(pool: &PgPool) -> Result<()> {
    let initialized: bool =
        sqlx::query_scalar("SELECT to_regclass('ducklake_snapshot') IS NOT NULL")
            .fetch_one(pool)
            .await?;
    if initialized {
        create_missing_tables(pool).await?;
        add_official_columns(pool).await?;
        seed_counter(pool, "next_catalog_id", SQL_MAX_CATALOG_ID).await?;
        seed_counter(pool, "next_file_id", SQL_MAX_FILE_ID).await?;
    }
    Ok(())
}

/// Run each of [`SQL_CREATE_TABLES`] whose table or index the catalog lacks. A
/// role that reads and writes the catalog without owning it may not run a
/// `CREATE`, even one `IF NOT EXISTS` skips, so a current catalog runs none.
async fn create_missing_tables(pool: &PgPool) -> Result<()> {
    for ddl in SQL_CREATE_TABLES {
        let present: bool = sqlx::query_scalar("SELECT to_regclass($1) IS NOT NULL")
            .bind(created_name(ddl))
            .fetch_one(pool)
            .await?;
        if !present {
            sqlx::query(*ddl).execute(pool).await?;
        }
    }
    Ok(())
}

/// Add the columns a catalog an earlier release created lacks: the snapshot id
/// watermarks, each of [`POSITIONAL_TABLES`]' columns, and those of earlier
/// spec versions, last, as a release before 0.10, which names the columns it
/// writes, still writes the table. Only a missing column
/// is altered in, since `ADD COLUMN IF NOT EXISTS` takes the table's
/// `ACCESS EXCLUSIVE` lock even when it adds nothing.
async fn add_official_columns(pool: &PgPool) -> Result<()> {
    let watermarks =
        [("next_catalog_id", "next_catalog_id BIGINT"), ("next_file_id", "next_file_id BIGINT")];
    let earlier = [
        (
            "ducklake_metadata",
            vec![("scope", "scope VARCHAR"), ("scope_id", "scope_id BIGINT")],
        ),
        (
            "ducklake_column",
            vec![
                ("initial_default", "initial_default VARCHAR"),
                ("default_value", "default_value VARCHAR"),
                ("default_value_type", "default_value_type VARCHAR"),
                ("default_value_dialect", "default_value_dialect VARCHAR"),
            ],
        ),
        (
            "ducklake_data_file",
            vec![("partition_id", "partition_id BIGINT"), ("partial_max", "partial_max BIGINT")],
        ),
        (
            "ducklake_delete_file",
            vec![("partial_max", "partial_max BIGINT")],
        ),
        ("ducklake_snapshot", watermarks.to_vec()),
    ];
    let tables = earlier
        .into_iter()
        .chain(POSITIONAL_TABLES.iter().map(|(table, id, columns)| {
            // Every catalog has its id column, and adding an identity column
            // creates its sequence even when the column exists.
            let missing = positional_columns(columns)
                .into_iter()
                .filter(|(name, _, _)| name != id)
                .map(|(name, definition, _)| (name, definition))
                .collect::<Vec<_>>();
            (*table, missing)
        }));
    for (table, columns) in tables {
        let current: Vec<String> = sqlx::query_scalar(
            "SELECT attname::text FROM pg_attribute
             WHERE attrelid = to_regclass($1) AND attnum > 0 AND NOT attisdropped",
        )
        .bind(table)
        .fetch_all(pool)
        .await?;
        // `initialize_schema` creates a table a catalog lacks whole.
        if current.is_empty() {
            continue;
        }
        for (name, definition) in columns {
            if !current.iter().any(|column| column == name) {
                sqlx::query(AssertSqlSafe(format!(
                    "ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {definition}"
                )))
                .execute(pool)
                .await?;
            }
        }
    }
    Ok(())
}

/// Rebuild each of [`POSITIONAL_TABLES`] whose columns are those official
/// expects but in another order, with another key, or with an id official
/// cannot supply (`GENERATED ALWAYS`), as in a catalog this writer created
/// before it followed official's layout. Rows without a `schema_uuid` or
/// `table_uuid` get one, and the new identity sequence continues past the
/// copied ids. One transaction per table, which locks the table first, so a
/// row another writer commits meanwhile is copied or waits for the rebuilt
/// table, and a rebuild cut short leaves the original in place. The rebuilt
/// table keeps the original's owner, grants and indexes, and its identity
/// sequence the original's grants; a table views depend on is refused, as
/// dropping it would drop them.
async fn order_positional_tables(pool: &PgPool) -> Result<()> {
    for (table, id, columns) in POSITIONAL_TABLES {
        let expected = positional_columns(columns)
            .into_iter()
            .map(|(name, _, key)| (name.to_string(), key, false))
            .collect::<Vec<_>>();
        let current: Vec<(String, bool, bool)> = sqlx::query_as(
            "SELECT a.attname::text,
                    EXISTS (SELECT 1 FROM pg_index i
                            WHERE i.indrelid = a.attrelid AND i.indisprimary
                              AND a.attnum = ANY (i.indkey)),
                    a.attidentity = 'a'
             FROM pg_attribute a
             WHERE a.attrelid = to_regclass($1) AND a.attnum > 0 AND NOT a.attisdropped
             ORDER BY a.attnum",
        )
        .bind(table)
        .fetch_all(pool)
        .await?;
        if current == expected
            || !current
                .iter()
                .all(|(column, ..)| expected.iter().any(|(name, ..)| name == column))
        {
            continue;
        }
        let names = current
            .iter()
            .map(|(column, ..)| column.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        let ordered = format!("{table}__ordered");
        let mut tx = pool.begin().await?;
        sqlx::query(AssertSqlSafe(format!(
            "LOCK TABLE {table} IN ACCESS EXCLUSIVE MODE"
        )))
        .execute(&mut *tx)
        .await?;
        let views: Vec<String> = sqlx::query_scalar(
            "SELECT DISTINCT view.oid::regclass::text
             FROM pg_depend dependency
             JOIN pg_rewrite rule ON rule.oid = dependency.objid
             JOIN pg_class view ON view.oid = rule.ev_class
             WHERE dependency.refobjid = to_regclass($1) AND view.oid <> dependency.refobjid
             ORDER BY 1",
        )
        .bind(table)
        .fetch_all(&mut *tx)
        .await?;
        if !views.is_empty() {
            return Err(crate::DuckLakeError::InvalidConfig(format!(
                "migrate_to_official_layout rebuilds {table}, and views {} depend on it; drop \
                 them, migrate, then create them again",
                views.join(", ")
            )));
        }
        let (restore, sequence_grants) = table_privileges_and_indexes(&mut tx, table, id).await?;
        for statement in [
            format!("CREATE TABLE {ordered} ({columns})"),
            format!(
                "INSERT INTO {ordered} ({names}) OVERRIDING SYSTEM VALUE \
                 SELECT {names} FROM {table}"
            ),
            format!(
                "SELECT setval(pg_get_serial_sequence('{ordered}', '{id}'), \
                 GREATEST(1, COALESCE((SELECT MAX({id}) FROM {ordered}), 0)))"
            ),
            format!("DROP TABLE {table}"),
            format!("ALTER TABLE {ordered} RENAME TO {table}"),
        ]
        .into_iter()
        .chain(restore)
        {
            sqlx::query(AssertSqlSafe(statement))
                .execute(&mut *tx)
                .await?;
        }
        // The rebuilt table's sequence keeps the name it was created under, so
        // it is known once the table is renamed.
        let sequence: Option<String> = sqlx::query_scalar("SELECT pg_get_serial_sequence($1, $2)")
            .bind(table)
            .bind(id)
            .fetch_one(&mut *tx)
            .await?;
        if let Some(sequence) = sequence {
            for (privilege, grantee) in &sequence_grants {
                sqlx::query(AssertSqlSafe(format!(
                    "GRANT {privilege} ON SEQUENCE {sequence}{grantee}"
                )))
                .execute(&mut *tx)
                .await?;
            }
        }
        if let Some((uuid, ..)) = expected
            .iter()
            .find(|(column, ..)| column.ends_with("_uuid"))
        {
            backfill_uuids(&mut tx, table, uuid).await?;
        }
        tx.commit().await?;
    }
    Ok(())
}

/// Give each `{table}` id without a `uuid` one, a v7 UUID as official writes,
/// the same for every row of the id, as a renamed table keeps its UUID.
async fn backfill_uuids(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    table: &str,
    uuid: &str,
) -> Result<()> {
    let id = uuid.replace("_uuid", "_id");
    let ids: Vec<i64> = sqlx::query_scalar(AssertSqlSafe(format!(
        "SELECT DISTINCT {id} FROM {table} WHERE {uuid} IS NULL"
    )))
    .fetch_all(&mut **tx)
    .await?;
    for row_id in ids {
        sqlx::query(AssertSqlSafe(format!(
            "UPDATE {table} SET {uuid} = $1::uuid WHERE {id} = $2 AND {uuid} IS NULL"
        )))
        .bind(uuid::Uuid::now_v7().to_string())
        .bind(row_id)
        .execute(&mut **tx)
        .await?;
    }
    Ok(())
}

/// The statements that give `table`, once rebuilt under its name, the
/// original's owner, grants and non-key indexes, and the grants of its `id`
/// identity sequence, as [`grants`] lists them. Every role name is
/// quoted as an identifier by `quote_ident`, so none reaches a string literal.
async fn table_privileges_and_indexes(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    table: &str,
    id: &str,
) -> Result<(Vec<String>, Vec<(String, String)>)> {
    let owner: String = sqlx::query_scalar(
        "SELECT quote_ident(pg_get_userbyid(relowner)) FROM pg_class WHERE oid = to_regclass($1)",
    )
    .bind(table)
    .fetch_one(&mut **tx)
    .await?;
    let mut statements = vec![format!("ALTER TABLE {table} OWNER TO {owner}")];
    let table_grants = grants(tx, table).await?;
    statements.extend(
        table_grants
            .iter()
            .map(|grant| format!("GRANT {} ON TABLE {table}{}", grant.0, grant.1)),
    );
    let indexes: Vec<String> = sqlx::query_scalar(
        "SELECT pg_get_indexdef(indexrelid) FROM pg_index
         WHERE indrelid = to_regclass($1) AND NOT indisprimary",
    )
    .bind(table)
    .fetch_all(&mut **tx)
    .await?;
    statements.extend(indexes);
    let sequence: Option<String> = sqlx::query_scalar("SELECT pg_get_serial_sequence($1, $2)")
        .bind(table)
        .bind(id)
        .fetch_one(&mut **tx)
        .await?;
    let sequence_grants = match sequence {
        Some(sequence) => grants(tx, &sequence).await?,
        None => Vec::new(),
    };
    Ok((statements, sequence_grants))
}

/// The grants on `relation` to roles other than its owner, as
/// `(privilege, " TO <grantee>[ WITH GRANT OPTION]")`.
async fn grants(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    relation: &str,
) -> Result<Vec<(String, String)>> {
    Ok(sqlx::query_as(
        "SELECT privilege.privilege_type,
                ' TO ' || CASE WHEN privilege.grantee = 0 THEN 'PUBLIC'
                               ELSE quote_ident(pg_get_userbyid(privilege.grantee)) END
                       || CASE WHEN privilege.is_grantable THEN ' WITH GRANT OPTION' ELSE '' END
         FROM pg_class relation, aclexplode(relation.relacl) privilege
         WHERE relation.oid = to_regclass($1) AND privilege.grantee <> relation.relowner",
    )
    .bind(relation)
    .fetch_all(&mut **tx)
    .await?)
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

/// Abort a `Replace` whose base is stale: any data file or delete file begun or
/// ended after `base_snapshot`, or an inlined delete committed after it, means
/// another writer changed the table in the meantime. `Append` does not call this
/// unless the write set an expected base snapshot — concurrent appends commute.
/// An inlined-data flush ([`ConflictScope::InlinedFlush`]) does not count delete
/// files: official DuckLake commits a flush over a concurrent delete file.
async fn detect_replace_conflict(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    table_id: i64,
    base_snapshot: i64,
    scope: ConflictScope,
) -> Result<()> {
    let conflict: Option<i32> = sqlx::query(
        "SELECT 1 WHERE EXISTS (SELECT 1 FROM ducklake_data_file
             WHERE table_id = $1 AND (begin_snapshot > $2 OR end_snapshot > $2))
           OR ($3 AND EXISTS (SELECT 1 FROM ducklake_delete_file
             WHERE table_id = $1 AND (begin_snapshot > $2 OR end_snapshot > $2)))",
    )
    .bind(table_id)
    .bind(base_snapshot)
    .bind(scope.counts_delete_files())
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
    detect_inlined_delete_since(table_id, base_snapshot, tx).await
}

/// Conflict when the row of `table_id` began after `base_snapshot`: a write
/// that expects the catalog at that snapshot found the table's name free.
async fn detect_created_table_conflict(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    table_id: i64,
    base_snapshot: i64,
) -> Result<()> {
    let created: bool = sqlx::query_scalar(
        "SELECT EXISTS (SELECT 1 FROM ducklake_table WHERE table_id = $1 AND begin_snapshot > $2)",
    )
    .bind(table_id)
    .bind(base_snapshot)
    .fetch_one(&mut **tx)
    .await?;
    if created {
        return Err(crate::metadata_writer::created_table_conflict(
            table_id,
            base_snapshot,
        ));
    }
    Ok(())
}

/// Retire the prior generation's still-visible data files at `snapshot_id`. The `begin_snapshot < snapshot_id` guard spares files
/// registered for *this* snapshot, so a multi-file write does not retire its
/// own siblings. `ducklake_table_stats` is left as it is, as official's `DELETE`
/// leaves it: `record_count` counts every row ever inserted.
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
    // Past every snapshot, as official DuckDB, which takes `MAX(snapshot_id) +
    // 1`, may have committed some after this counter.
    let snapshot_id: i64 = sqlx::query_scalar(
        "UPDATE ducklake_metadata
         SET value = CAST(GREATEST(CAST(value AS BIGINT),
             (SELECT COALESCE(MAX(snapshot_id), 0) FROM ducklake_snapshot)) + 1 AS VARCHAR)
         WHERE key = 'next_snapshot_id' AND scope IS NULL
         RETURNING CAST(value AS BIGINT)",
    )
    .fetch_one(&mut **tx)
    .await?;
    // The commit inserts its files after this; their ids skip official's.
    skip_official_ids(tx, "ducklake_data_file", "data_file_id", "next_file_id").await?;
    skip_official_ids(tx, "ducklake_delete_file", "delete_file_id", "next_file_id").await?;
    // Carry the current per-catalog schema_version forward; a DDL commit corrects
    // this to a bump via `bump_schema_version` below. Read before the INSERT so
    // the MAX is over the pre-existing rows only (matches the other writers).
    let schema_version: i64 =
        sqlx::query("SELECT COALESCE(MAX(schema_version), 0) FROM ducklake_snapshot")
            .fetch_one(&mut **tx)
            .await?
            .try_get(0)?;
    crate::metadata_writer_postgres::utc_snapshot_time(tx).await?;
    sqlx::query(
        "INSERT INTO ducklake_snapshot
             (snapshot_id, snapshot_time, schema_version, next_catalog_id, next_file_id)
         SELECT $1, NOW(), $2, next_catalog_id, next_file_id
         FROM (SELECT 1) AS one
         LEFT JOIN (SELECT next_catalog_id, next_file_id FROM ducklake_snapshot
                    ORDER BY snapshot_id DESC LIMIT 1) AS head ON TRUE",
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
    let inlined = inlined_inserts(changes_made.unwrap_or_default());
    if inlined > 0 {
        sqlx::query(
            "UPDATE ducklake_snapshot SET next_file_id = COALESCE(next_file_id, 0) + $1
             WHERE snapshot_id = $2",
        )
        .bind(inlined as i64)
        .bind(snapshot_id)
        .execute(&mut **tx)
        .await?;
    }
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

    let write_changes = table_write_changes(table_id, mode, false, (replaced_existing_data, false));
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
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
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
    let mut changes = Vec::new();
    if schema_begin_snapshot == snapshot_id {
        let entry = format!("created_schema:{}", quote_snapshot_name(schema_name));
        let recorded: Option<String> = sqlx::query_scalar(
            "SELECT changes_made FROM ducklake_snapshot_changes WHERE snapshot_id = $1",
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
/// live (create, column add/remove/reorder, partitioning), and create the
/// table's inlined data table for it as official does. Not called for a drop.
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
    crate::metadata_writer_postgres::create_eager_inlined_table(tx, None, table_id, schema_version)
        .await
}

/// Persist the harvested per-column stats for a just-registered data file
/// (per-file zone maps). See the SQLite writer's equivalent for the rationale.
async fn insert_file_column_stats(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
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
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NULL)",
        )
        .bind(data_file_id)
        .bind(table_id)
        .bind(stat.column_id)
        .bind(stat.column_size_bytes)
        .bind(stat.value_count)
        .bind(stat.null_count)
        // PostgreSQL text holds no NUL byte: such a bound is stored as unknown.
        .bind(
            stat.min_value
                .as_deref()
                .filter(|text| !text.contains('\0')),
        )
        .bind(
            stat.max_value
                .as_deref()
                .filter(|text| !text.contains('\0')),
        )
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

/// The atomic commit point: insert the snapshot row, create the schema/table rows
/// if absent, finalize the column generation, and for `Replace` retire the prior
/// data generation — all in the caller's transaction, so a reader never sees a
/// half-published head.
///
/// Schema, table and column rows are all written here rather than at begin. The
/// read path resolves columns by `end_snapshot IS NULL` alone, and resolves
/// schemas/tables by `snapshot >= begin_snapshot` with no check that the snapshot
/// exists — so a row written at begin with a guessed id becomes visible as soon as
/// any writer reaches that id, even if this write never commits.
#[allow(clippy::too_many_arguments)]
async fn finalize_snapshot(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    schema_name: &str,
    table_name: &str,
    table_id_hint: i64,
    columns: &[ColumnDef],
    column_ids: &[i64],
    mode: WriteMode,
    base_snapshot: i64,
    reserved_at_setup: bool,
) -> Result<CommitIds> {
    let (snapshot_id, _) = insert_snapshot(tx).await?;
    finalize_table_snapshot(
        tx,
        snapshot_id,
        schema_name,
        table_name,
        table_id_hint,
        columns,
        column_ids,
        mode,
        base_snapshot,
        true,
        reserved_at_setup,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn finalize_table_snapshot(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    snapshot_id: i64,
    schema_name: &str,
    table_name: &str,
    table_id_hint: i64,
    columns: &[ColumnDef],
    column_ids: &[i64],
    mode: WriteMode,
    base_snapshot: i64,
    inserts: bool,
    reserved_at_setup: bool,
) -> Result<CommitIds> {
    // Allocate the snapshot FIRST (carrying schema_version forward): this takes
    // the counter lock up front, serializing concurrent commits. schema_version is
    // corrected to a DDL bump below once we've classified the commit.
    let mut schema_version: i64 =
        sqlx::query_scalar("SELECT schema_version FROM ducklake_snapshot WHERE snapshot_id = $1")
            .bind(snapshot_id)
            .fetch_one(&mut **tx)
            .await?;

    let schema_id: i64 = match sqlx::query_scalar(
        "SELECT schema_id FROM ducklake_schema
         WHERE schema_name = $1 AND end_snapshot IS NULL",
    )
    .bind(schema_name)
    .fetch_optional(&mut **tx)
    .await?
    {
        Some(id) => id,
        None => {
            let schema_id = reserve_catalog_id(tx).await?;
            sqlx::query_scalar(
                "INSERT INTO ducklake_schema
                     (schema_id, schema_uuid, schema_name, path, path_is_relative, begin_snapshot)
                 OVERRIDING SYSTEM VALUE
                 VALUES ($1, $5::uuid, $2, $3, TRUE, $4)
                 RETURNING schema_id",
            )
            .bind(schema_id)
            .bind(schema_name)
            .bind(directory_path(schema_name))
            .bind(snapshot_id)
            .bind(uuid::Uuid::now_v7().to_string())
            .fetch_one(&mut **tx)
            .await?
        },
    };

    // A new table keeps the id reserved at begin: the caller already holds it in
    // `WriteSetupResult` and passes it back here.
    let table_id: i64 = match sqlx::query_scalar(
        "SELECT table_id FROM ducklake_table
         WHERE schema_id = $1 AND table_name = $2 AND end_snapshot IS NULL",
    )
    .bind(schema_id)
    .bind(table_name)
    .fetch_optional(&mut **tx)
    .await?
    {
        Some(id) if id == table_id_hint => {
            // A setup that reserved the id found no table, and writes none, so a
            // live table of the id is another transaction's create that took it.
            if reserved_at_setup {
                refuse_changed_table_id(&reserved_id_taken(), mode, schema_name, table_name)?;
            }
            id
        },
        Some(_) => {
            return Err(crate::DuckLakeError::Conflict(format!(
                "table {} was created by another transaction; retry the write",
                crate::metadata_writer::quote_snapshot_table(schema_name, table_name)
            )));
        },
        None => {
            // As in the SQLite writer's `insert_new_table_rows`.
            let (renamed, taken, base_expired): (bool, bool, bool) = sqlx::query_as(
                "SELECT EXISTS (SELECT 1 FROM ducklake_table WHERE table_id = $1 AND table_name = $2),
                        EXISTS (SELECT 1 FROM ducklake_table WHERE table_id = $1 AND table_name <> $2)
                            OR EXISTS (SELECT 1 FROM ducklake_schema WHERE schema_id = $1)
                            OR EXISTS (SELECT 1 FROM ducklake_view WHERE view_id = $1)
                            OR EXISTS (SELECT 1 FROM ducklake_partition_info WHERE partition_id = $1)
                            OR EXISTS (SELECT 1 FROM ducklake_sort_info WHERE sort_id = $1)
                            OR EXISTS (SELECT 1 FROM ducklake_macro WHERE macro_id = $1),
                        $3 > 0 AND NOT EXISTS (SELECT 1 FROM ducklake_snapshot WHERE snapshot_id = $3)",
            )
            .bind(table_id_hint)
            .bind(table_name)
            .bind(base_snapshot)
            .fetch_one(&mut **tx)
            .await?;
            let reserved = ReservedTableId {
                renamed,
                taken,
                base_expired,
                reserved_at_setup,
            };
            refuse_changed_table_id(&reserved, mode, schema_name, table_name)?;
            sqlx::query(
                "INSERT INTO ducklake_table
                     (table_id, table_uuid, schema_id, table_name, path, path_is_relative,
                      begin_snapshot)
                 OVERRIDING SYSTEM VALUE
                 VALUES ($1, $6::uuid, $2, $3, $4, TRUE, $5)",
            )
            .bind(table_id_hint)
            .bind(schema_id)
            .bind(table_name)
            .bind(directory_path(table_name))
            .bind(snapshot_id)
            .bind(uuid::Uuid::now_v7().to_string())
            .execute(&mut **tx)
            .await?;
            table_id_hint
        },
    };

    // Classify this commit as DDL vs pure data write. `current` is the table's
    // live recursive column tree ordered by `column_order`.
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
         WHERE table_id = $1 AND end_snapshot IS NULL
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
        crate::metadata_writer_postgres::reject_new_reserved_columns(
            tx,
            None,
            table_id,
            &existing_catalog_columns,
            &proposed,
        )
        .await?;
    }
    if is_ddl {
        // A DDL commit bumps the per-catalog schema_version (the insert above only
        // carried it forward). A pure data write keeps the carried value.
        schema_version = bump_schema_version(tx, snapshot_id).await?;
    }

    // Reconcile the column generation SURGICALLY so each column keeps a stable
    // column_id (== parquet field_id) across writes: end only removed columns,
    // insert only new ones, and leave unchanged columns (and their ids) in place.
    // Official refuses to drop a column its sort order references.
    let dropped =
        crate::metadata_writer::dropped_top_level_columns(&existing_catalog_columns, column_ids);
    if !dropped.is_empty() {
        let keys: Vec<(String, String)> = sqlx::query_as(crate::sort::LIVE_SORT_KEYS_SQL_PG)
            .bind(table_id)
            .fetch_all(&mut **tx)
            .await?;
        crate::sort::refuse_dropping_sort_columns(&keys, &dropped)?;
    }
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
                "UPDATE ducklake_column SET end_snapshot = $1
                 WHERE table_id = $2 AND column_id = $3 AND end_snapshot IS NULL",
            )
            .bind(snapshot_id)
            .bind(table_id)
            .bind(column_id)
            .execute(&mut **tx)
            .await?;
        }
        current_by_id.insert(column_id, (order, nullable, ducklake_type));
    }

    let existing_orders = current
        .iter()
        .map(|row| row.try_get::<i64, _>("column_order"))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let orders = crate::metadata_writer::proposed_column_orders(
        &existing_catalog_columns,
        &existing_orders,
        &proposed,
        column_ids,
    );
    for ((column, column_id), order) in proposed.iter().zip(column_ids).zip(orders) {
        let parent_id = column.parent_index.map(|index| column_ids[index]);
        match current_by_id.get(column_id) {
            // A data write leaves a live column's row as it is, as official's
            // does. A schema change that retypes it, changes its top-level
            // nullability or moves it records a new row version, keeping the
            // old one for time travel.
            Some((cur_order, cur_nullable, cur_type)) => {
                if !is_ddl {
                    continue;
                }
                let migrate_type = catalog_column_type_requires_migration(cur_type, column);
                let nullable = if column.parent_index.is_none() {
                    column.is_nullable
                } else {
                    *cur_nullable
                };
                if !migrate_type && *cur_order == order && *cur_nullable == nullable {
                    continue;
                }
                let column_type = if migrate_type {
                    column.ducklake_type.as_str()
                } else {
                    cur_type.as_str()
                };
                sqlx::query(
                    "UPDATE ducklake_column SET end_snapshot = $1
                     WHERE table_id = $2 AND column_id = $3 AND end_snapshot IS NULL",
                )
                .bind(snapshot_id)
                .bind(table_id)
                .bind(column_id)
                .execute(&mut **tx)
                .await?;
                sqlx::query(
                    "INSERT INTO ducklake_column
                         (column_id, table_id, column_name, column_type, column_order,
                          nulls_allowed, parent_column, begin_snapshot, initial_default,
                          default_value, default_value_type, default_value_dialect)
                     SELECT column_id, table_id, column_name, $1, $2, $3, parent_column, $4,
                            initial_default, default_value, default_value_type,
                            default_value_dialect
                     FROM ducklake_column
                     WHERE table_id = $5 AND column_id = $6 AND end_snapshot = $4
                     LIMIT 1",
                )
                .bind(column_type)
                .bind(order)
                .bind(nullable)
                .bind(snapshot_id)
                .bind(table_id)
                .bind(column_id)
                .execute(&mut **tx)
                .await?;
            },
            None => {
                sqlx::query(
                    "INSERT INTO ducklake_column
                            (column_id, table_id, column_name, column_type, column_order,
                             nulls_allowed, parent_column, begin_snapshot, initial_default,
                             default_value, default_value_type, default_value_dialect)
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)",
                )
                .bind(column_id)
                .bind(table_id)
                .bind(&column.name)
                .bind(&column.ducklake_type)
                .bind(order)
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
        crate::metadata_writer_postgres::check_table_conflicts(
            tx,
            table_id,
            base_snapshot,
            crate::metadata_writer::TableChanges::append(),
        )
        .await?;
    }

    // Record the schema-change ledger row for a DDL commit. A pure data write
    // carries schema_version forward and writes no row.
    if is_ddl {
        record_schema_version(tx, snapshot_id, schema_version, table_id).await?;
    }
    Ok(CommitIds {
        snapshot_id,
        schema_id,
        table_id,
    })
}

impl MetadataWriter for PostgresSingleCatalogMetadataWriter {
    fn drop_empty_superseded_inlined_tables(&self) -> Result<()> {
        self.block_on(async {
            // Staged commits create the registry on demand; without it there is
            // nothing inlined to drop.
            if !crate::PostgresMetadataProvider::from_pool(self.pool.clone())
                .has_inlined_data_registry()
                .await?
            {
                return Ok(());
            }
            let mut tx = self.pool.begin().await?;
            drop_empty_superseded_inlined_tables_in(&mut tx, None).await?;
            self.commit(tx).await?;
            Ok(())
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
        let ids = writes
            .iter()
            .map(|write| write.table_id)
            .collect::<std::collections::HashSet<_>>();
        if ids.len() != writes.len() {
            return Err(crate::DuckLakeError::InvalidConfig(
                "commit_multi_table requires one stage per table".to_string(),
            ));
        }
        self.block_on(async {
            sqlx::query(SQL_CREATE_INLINED_DATA_TABLES)
                .execute(&self.pool)
                .await?;
            let mut tx = self.pool.begin().await?;
            let (snapshot_id, _) = insert_snapshot(&mut tx).await?;
            let mut had_data = Vec::with_capacity(writes.len());
            for write in writes {
                // Official checks the table-level conflicts first.
                check_staged_conflicts(&mut tx, write, Some(snapshot_id)).await?;
                validate_staged_table(&mut tx, write).await?;
                detect_row_mutation_conflicts(&mut tx, write).await?;
                if let Some(expected) = expected_base_snapshot_id {
                    detect_staged_conflict(
                        write.table_id,
                        expected,
                        ConflictScope::of(write),
                        &mut tx,
                    )
                    .await?;
                    detect_created_table_conflict(&mut tx, write.table_id, expected).await?;
                }
                let files: bool = sqlx::query_scalar("SELECT EXISTS(SELECT 1 FROM ducklake_data_file WHERE table_id = $1 AND end_snapshot IS NULL)").bind(write.table_id).fetch_one(&mut *tx).await?;
                let mut inline = false;
                for (name, system) in
                    crate::metadata_writer_postgres::inlined_tables(&mut tx, write.table_id).await?
                {
                    let sql = format!(
                        "SELECT EXISTS(SELECT 1 FROM {} WHERE {} IS NULL)",
                        quote_ident(&name),
                        system.end_snapshot,
                    );
                    inline |= sqlx::query_scalar::<_, bool>(AssertSqlSafe(sql))
                        .fetch_one(&mut *tx)
                        .await?;
                }
                had_data.push((files, inline));
            }
            let mut tables = Vec::with_capacity(writes.len());
            for write in writes {
                tables.push(
                    finalize_table_snapshot(
                        &mut tx,
                        snapshot_id,
                        &write.schema_name,
                        &write.table_name,
                        write.table_id,
                        &write.columns,
                        &write.column_ids,
                        write.mode,
                        write.base_snapshot_id,
                        !matches!(write.data, StagedTableData::None),
                        self.reserved_tables.take(write.table_id),
                    )
                    .await?,
                );
            }
            for ((write, ids), (files, inline)) in writes.iter().zip(&tables).zip(had_data) {
                let mut write = write.clone();
                write.table_id = ids.table_id;
                match &write.data {
                    StagedTableData::Files(files) => {
                        commit_files_at_snapshot(&mut tx, snapshot_id, &write, files).await?;
                    },
                    StagedTableData::Inlined(batches) => {
                        commit_inlined_at_snapshot(&mut tx, snapshot_id, &write, batches).await?;
                    },
                    StagedTableData::None => {},
                }
                apply_positional_deletes_at_snapshot(
                    &mut tx,
                    write.table_id,
                    snapshot_id,
                    write.base_snapshot_id,
                    &write.positional_deletes,
                )
                .await?;
                apply_inlined_deletes_at_snapshot(
                    &mut tx,
                    write.table_id,
                    snapshot_id,
                    write.base_snapshot_id,
                    &write.inlined_deletes,
                )
                .await?;
                apply_row_mutation_at_snapshot(&mut tx, snapshot_id, &write, None).await?;
                let changes = staged_table_write_changes(&write, files, inline);
                record_table_changes(
                    &mut tx,
                    snapshot_id,
                    write.table_id,
                    &write.schema_name,
                    &write.table_name,
                    &changes,
                    commit_metadata,
                )
                .await?;
            }
            self.commit_networked(tx).await?;
            Ok(MultiTableCommit {
                snapshot_id,
                tables,
            })
        })
    }

    fn commit_inlined_flush(
        &self,
        base_snapshot: i64,
        tables: &[InlinedFlushTable],
    ) -> Result<i64> {
        validate_inlined_flush(tables)?;
        self.block_on(async {
            let mut tx = self.pool.begin().await?;
            // A flush is not DDL: schema_version carries forward.
            let (snapshot_id, _) = insert_snapshot(&mut tx).await?;
            apply_inlined_flush_at_snapshot(&mut tx, snapshot_id, base_snapshot, tables, None)
                .await?;
            record_snapshot_changes(
                &mut tx,
                snapshot_id,
                &inlined_flush_changes(tables),
                &SnapshotCommitMetadata::default(),
            )
            .await?;
            self.commit_networked(tx).await?;
            Ok(snapshot_id)
        })
    }

    fn data_inlining_row_limit(&self) -> Option<usize> {
        self.data_inlining_row_limit
    }

    fn reserved_inlining_column(&self, columns: &[String]) -> Result<Option<String>> {
        let system = self.block_on(
            crate::metadata_provider_postgres::catalog_inlined_system_columns(&self.pool),
        )?;
        Ok(columns
            .iter()
            .find(|column| system.conflicts_with(column))
            .cloned())
    }

    fn table_data_inlining_row_limit(&self, schema_name: &str, table_name: &str) -> Result<usize> {
        self.block_on(
            crate::metadata_writer_postgres::postgres_inlining_limit_by_name(
                &self.pool,
                None,
                schema_name,
                table_name,
            ),
        )
    }

    fn table_setting(
        &self,
        schema_name: &str,
        table_name: &str,
        key: &str,
    ) -> Result<Option<String>> {
        self.block_on(
            crate::metadata_writer_postgres::postgres_table_setting_by_name(
                &self.pool,
                None,
                schema_name,
                table_name,
                key,
            ),
        )
    }

    fn set_schema_setting(&self, schema_id: i64, key: &str, value: &str) -> Result<()> {
        crate::metadata_writer_postgres::set_postgres_schema_setting(
            &self.pool,
            self.commit_guard.as_deref(),
            None,
            schema_id,
            key,
            value,
        )
    }

    fn set_table_setting(&self, table_id: i64, key: &str, value: &str) -> Result<()> {
        set_postgres_table_setting(
            &self.pool,
            self.commit_guard.as_deref(),
            None,
            table_id,
            key,
            value,
        )
    }

    fn with_commit_lock(
        &self,
        identity: &str,
        operation: Box<dyn FnOnce() -> Result<()> + '_>,
    ) -> Result<()> {
        with_postgres_commit_lock(&self.pool, &format!("single:{identity}"), operation)
    }

    fn catalog_encrypted(&self) -> Result<bool> {
        self.block_on(async {
            let values: Vec<String> = sqlx::query_scalar(
                "SELECT value FROM ducklake_metadata WHERE key = 'encrypted' AND scope IS NULL",
            )
            .fetch_all(&self.pool)
            .await?;
            crate::metadata_writer::encrypted_option(&values)
        })
    }

    fn head_snapshot(&self) -> Result<Option<i64>> {
        self.block_on(async {
            Ok(
                sqlx::query_scalar("SELECT MAX(snapshot_id) FROM ducklake_snapshot")
                    .fetch_one(&self.pool)
                    .await?,
            )
        })
    }

    fn create_snapshot(&self) -> Result<i64> {
        self.block_on(async {
            let mut tx = self.pool.begin().await?;
            // A bare snapshot carries no schema change of its own → carry
            // schema_version forward (no DDL bump, no ledger row).
            let (snapshot_id, _schema_version) = insert_snapshot(&mut tx).await?;
            self.commit(tx).await?;
            Ok(snapshot_id)
        })
    }

    fn set_global_setting(&self, key: &str, value: &str) -> Result<()> {
        // Option names are case-insensitive; official stores them lowercased.
        let key = key.to_ascii_lowercase();
        let key = key.as_str();
        let value = crate::metadata_writer::normalize_setting_value(key, value)?;
        let value = value.as_str();
        self.block_on(async {
            let mut transaction = self.pool.begin().await?;
            if crate::metadata_writer::enables_inlining(key, value) {
                let columns: Vec<(String, String)> =
                    sqlx::query_as(crate::metadata_writer::SQL_GLOBAL_INLINING_COLUMNS)
                        .fetch_all(&mut *transaction)
                        .await?;
                let system = crate::metadata_provider_postgres::catalog_inlined_system_columns(
                    &mut *transaction,
                )
                .await?;
                crate::metadata_writer::reject_reserved_inlining_columns(&columns, system)?;
            }
            sqlx::query("DELETE FROM ducklake_metadata WHERE key = $1 AND scope IS NULL")
                .bind(key)
                .execute(&mut *transaction)
                .await?;
            sqlx::query(
                "INSERT INTO ducklake_metadata (key, value, scope, scope_id)
                 VALUES ($1, $2, NULL, NULL)",
            )
            .bind(key)
            .bind(value)
            .execute(&mut *transaction)
            .await?;
            self.commit(transaction).await?;
            Ok(())
        })
    }

    fn get_or_create_schema(
        &self,
        name: &str,
        path: Option<&str>,
        snapshot_id: i64,
    ) -> Result<(i64, bool)> {
        validate_name(name, "Schema")?;
        self.block_on(async {
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
                self.commit(tx).await?;
                return Ok((row.try_get(0)?, false));
            }

            // Unscoped, relative path: the file layout stays
            // `{data_path}/{schema}/{table}/…`, matching every other
            // single-catalog backend (the multicatalog writer scopes to
            // `cat_{id}/{schema}` instead).
            let schema_path = directory_path(path.unwrap_or(name));
            let schema_id = reserve_catalog_id(&mut tx).await?;
            sqlx::query(
                "INSERT INTO ducklake_schema
                     (schema_id, schema_uuid, schema_name, path, path_is_relative, begin_snapshot)
                 OVERRIDING SYSTEM VALUE
                 VALUES ($1, $5::uuid, $2, $3, TRUE, $4)",
            )
            .bind(schema_id)
            .bind(name)
            .bind(&schema_path)
            .bind(snapshot_id)
            .bind(uuid::Uuid::now_v7().to_string())
            .execute(&mut *tx)
            .await?;

            record_snapshot_changes(
                &mut tx,
                snapshot_id,
                &format!("created_schema:{}", quote_snapshot_name(name)),
                &SnapshotCommitMetadata::default(),
            )
            .await?;
            self.commit(tx).await?;
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
        self.block_on(async {
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
                self.commit(tx).await?;
                return Ok((row.try_get(0)?, false));
            }

            // Needed for the qualified name in the change record.
            let schema_name: String =
                sqlx::query_scalar("SELECT schema_name FROM ducklake_schema WHERE schema_id = $1")
                    .bind(schema_id)
                    .fetch_one(&mut *tx)
                    .await?;

            let table_path = directory_path(path.unwrap_or(name));
            let table_id = reserve_catalog_id(&mut tx).await?;
            sqlx::query(
                "INSERT INTO ducklake_table
                     (table_id, table_uuid, schema_id, table_name, path, path_is_relative,
                      begin_snapshot)
                 OVERRIDING SYSTEM VALUE
                 VALUES ($1, $6::uuid, $2, $3, $4, TRUE, $5)",
            )
            .bind(table_id)
            .bind(schema_id)
            .bind(name)
            .bind(&table_path)
            .bind(snapshot_id)
            .bind(uuid::Uuid::now_v7().to_string())
            .execute(&mut *tx)
            .await?;

            record_snapshot_changes(
                &mut tx,
                snapshot_id,
                &format!("created_table:{}", quote_snapshot_table(&schema_name, name)),
                &SnapshotCommitMetadata::default(),
            )
            .await?;
            self.commit(tx).await?;
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
        self.block_on(async {
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
            let catalog_columns = catalog_column_defs(columns)?;
            let n = catalog_columns.len() as i64;
            reserve_ids(&mut tx, "next_column_id", n).await?;
            let last_column_id = raise_column_ids_past_table(&mut tx, table_id, n).await?;
            let first_column_id = last_column_id - n + 1;
            let field_ids = (first_column_id..=last_column_id).collect::<Vec<_>>();
            for (column, column_id) in catalog_columns.iter().zip(field_ids.iter()) {
                let parent_id = column.parent_index.map(|index| field_ids[index]);
                sqlx::query(
                    "INSERT INTO ducklake_column
                     (column_id, table_id, column_name, column_type, column_order,
                      nulls_allowed, parent_column, begin_snapshot, initial_default,
                      default_value, default_value_type, default_value_dialect)
                     VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)",
                )
                .bind(column_id)
                .bind(table_id)
                .bind(&column.name)
                .bind(&column.ducklake_type)
                // Official numbers a column it creates by its id.
                .bind(*column_id)
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

            self.commit(tx).await?;
            top_level_column_ids(&catalog_columns, &field_ids)
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
        self.block_on(async {
            // Single atomic commit: insert the snapshot row + create schema/table if
            // absent + finalize the column generation + retire the prior generation
            // (Replace), then register this file and advance the monotonic
            // row-lineage counter — all in one transaction, so the head only ever
            // resolves to fully-populated data.
            let mut tx = self.pool.begin().await?;

            let ids = finalize_snapshot(
                &mut tx,
                schema_name,
                table_name,
                table_id,
                columns,
                column_ids,
                mode,
                base_snapshot,
                self.reserved_tables.take(table_id),
            )
            .await?;
            let (snapshot_id, table_id) = (ids.snapshot_id, ids.table_id);

            // Caller-supplied precondition: the table's data-file generation must
            // not have moved past the snapshot the input was read at. Replace is
            // already fenced against `base_snapshot` inside finalize_snapshot.
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

            // Partition-spec fence: this file must be consistent with the table's live
            // partition generation at commit time (both directions — see
            // enforce_partition_fence). The tx rolls back on a Conflict.
            let partition_generation = live_partition_id(&mut tx, table_id).await?;
            crate::metadata_writer::enforce_partition_fence(table_id, partition_generation, file)?;

            // Seed the stats row for the Append path (Replace already seeded it in
            // finalize_snapshot); ON CONFLICT DO NOTHING is a no-op if it exists.
            seed_table_stats(&mut tx, table_id).await?;

            let next_row_id: i64 =
                sqlx::query("SELECT next_row_id FROM ducklake_table_stats WHERE table_id = $1")
                    .bind(table_id)
                    .fetch_one(&mut *tx)
                    .await?
                    .try_get(0)?;
            let row_ids = crate::metadata_writer::allocate_row_id_start(next_row_id, file);

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
            .bind(row_ids.stored)
            .bind(snapshot_id)
            .fetch_one(&mut *tx)
            .await?
            .try_get(0)?;

            // Persist the file's zone maps + refresh the roll-up.
            insert_file_column_stats(&mut tx, table_id, data_file_id, &file.column_stats).await?;
            insert_partition_metadata(&mut tx, table_id, data_file_id, file).await?;
            crate::metadata_writer_postgres::merge_file_table_column_stats(
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
                 SET next_row_id     = next_row_id + $1,
                     record_count    = record_count + $2,
                     file_size_bytes = file_size_bytes + $3
                 WHERE table_id = $4",
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
                commit_metadata,
            )
            .await?;

            self.commit_networked(tx).await?;
            Ok(ids)
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
        if files.is_empty() {
            return Err(crate::DuckLakeError::InvalidConfig(
                "register_data_files: files must be non-empty".to_string(),
            ));
        }
        self.block_on(async {
            let mut tx = self.pool.begin().await?;
            let ids = finalize_snapshot(
                &mut tx,
                schema_name,
                table_name,
                table_id,
                columns,
                column_ids,
                mode,
                base_snapshot,
                self.reserved_tables.take(table_id),
            )
            .await?;
            let (snapshot_id, table_id) = (ids.snapshot_id, ids.table_id);
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
            let mut total_advance: i64 = 0;
            let mut total_records: i64 = 0;
            let mut total_bytes: i64 = 0;
            for file in files {
                let row_ids = crate::metadata_writer::allocate_row_id_start(next_row_id, file);
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
                .bind(row_ids.stored)
                .bind(snapshot_id)
                .fetch_one(&mut *tx)
                .await?
                .try_get(0)?;
                insert_file_column_stats(&mut tx, table_id, data_file_id, &file.column_stats)
                    .await?;
                insert_partition_metadata(&mut tx, table_id, data_file_id, file).await?;
                next_row_id += row_ids.advance;
                total_advance += row_ids.advance;
                total_records += file.record_count;
                total_bytes += file.file_size_bytes;
            }
            crate::metadata_writer_postgres::merge_file_table_column_stats(
                &mut tx,
                table_id,
                &crate::metadata_writer::catalog_fields(columns, column_ids)?,
                &crate::metadata_writer::file_column_stats(files),
            )
            .await?;
            sqlx::query(
                "UPDATE ducklake_table_stats
                 SET next_row_id     = next_row_id + $1,
                     record_count    = record_count + $2,
                     file_size_bytes = file_size_bytes + $3
                 WHERE table_id = $4",
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
                commit_metadata,
            )
            .await?;
            self.commit_networked(tx).await?;
            Ok(ids)
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
        self.block_on(async {
            let mut tx = self.pool.begin().await?;
            let partition_id = reserve_catalog_id(&mut tx).await?;
            let (new_snapshot, _carried) = insert_snapshot(&mut tx).await?;

            let mut resolved_column_ids: Vec<i64> = Vec::with_capacity(columns.len());
            for (name, _transform) in columns {
                let column_id: i64 = sqlx::query_scalar(
                    "SELECT column_id FROM ducklake_column
                     WHERE table_id = $1 AND column_name = $2 AND end_snapshot IS NULL
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
            self.commit(tx).await?;
            Ok(new_snapshot)
        })
    }

    fn live_partition_spec(
        &self,
        table_id: i64,
    ) -> Result<Option<crate::partition::PartitionSpec>> {
        self.block_on(async {
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
        self.block_on(async {
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
            self.commit(tx).await?;
            Ok(new_snapshot)
        })
    }

    fn live_sort_spec(&self, table_id: i64) -> Result<Option<crate::sort::SortSpec>> {
        self.block_on(async {
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

    fn set_tag(&self, target: TagTarget, key: &str, value: Option<&str>) -> Result<i64> {
        validate_name(key, "Tag key")?;
        let change = tag_change(target, key)?;
        self.block_on(async {
            let mut tx = self.pool.begin().await?;
            let (snapshot_id, _carried) = insert_snapshot(&mut tx).await?;
            let live: bool = match target {
                TagTarget::Object {
                    object_type: TagObjectType::View,
                    object_id,
                } => {
                    sqlx::query_scalar(
                        "SELECT EXISTS(SELECT 1 FROM ducklake_view
                         WHERE view_id = $1 AND end_snapshot IS NULL)",
                    )
                    .bind(object_id)
                    .fetch_one(&mut *tx)
                    .await?
                },
                TagTarget::Object {
                    object_id,
                    ..
                } => {
                    sqlx::query_scalar(
                        "SELECT EXISTS(SELECT 1 FROM ducklake_table
                         WHERE table_id = $1 AND end_snapshot IS NULL)",
                    )
                    .bind(object_id)
                    .fetch_one(&mut *tx)
                    .await?
                },
                TagTarget::Column {
                    table_id,
                    column_id,
                } => {
                    sqlx::query_scalar(
                        "SELECT EXISTS(SELECT 1 FROM ducklake_column
                         WHERE table_id = $1 AND column_id = $2 AND end_snapshot IS NULL)",
                    )
                    .bind(table_id)
                    .bind(column_id)
                    .fetch_one(&mut *tx)
                    .await?
                },
            };
            if !live {
                return Err(crate::DuckLakeError::InvalidConfig(format!(
                    "set_tag: {target:?} is not live in this catalog"
                )));
            }
            // A comment is a schema change in official: it bumps `schema_version`, which
            // attached DuckDB sessions cache the loaded catalog by. Unlike a DDL commit it
            // writes no `ducklake_schema_versions` row.
            bump_schema_version(&mut tx, snapshot_id).await?;
            match target {
                TagTarget::Object {
                    object_id,
                    ..
                } => {
                    sqlx::query(
                        "UPDATE ducklake_tag SET end_snapshot = $1
                         WHERE object_id = $2 AND key = $3 AND end_snapshot IS NULL",
                    )
                    .bind(snapshot_id)
                    .bind(object_id)
                    .bind(key)
                    .execute(&mut *tx)
                    .await?;
                    sqlx::query(
                        "INSERT INTO ducklake_tag
                             (object_id, begin_snapshot, end_snapshot, key, value)
                         VALUES ($1, $2, NULL, $3, $4)",
                    )
                    .bind(object_id)
                    .bind(snapshot_id)
                    .bind(key)
                    .bind(value)
                    .execute(&mut *tx)
                    .await?;
                },
                TagTarget::Column {
                    table_id,
                    column_id,
                } => {
                    sqlx::query(
                        "UPDATE ducklake_column_tag SET end_snapshot = $1
                         WHERE table_id = $2 AND column_id = $3 AND key = $4
                           AND end_snapshot IS NULL",
                    )
                    .bind(snapshot_id)
                    .bind(table_id)
                    .bind(column_id)
                    .bind(key)
                    .execute(&mut *tx)
                    .await?;
                    sqlx::query(
                        "INSERT INTO ducklake_column_tag
                             (table_id, column_id, begin_snapshot, end_snapshot, key, value)
                         VALUES ($1, $2, $3, NULL, $4, $5)",
                    )
                    .bind(table_id)
                    .bind(column_id)
                    .bind(snapshot_id)
                    .bind(key)
                    .bind(value)
                    .execute(&mut *tx)
                    .await?;
                },
            }
            record_snapshot_changes(
                &mut tx,
                snapshot_id,
                &change,
                &SnapshotCommitMetadata::default(),
            )
            .await?;
            self.commit(tx).await?;
            Ok(snapshot_id)
        })
    }

    fn set_sort_spec(&self, table_id: i64, fields: &[crate::sort::SortField]) -> Result<i64> {
        if fields.is_empty() {
            return Err(crate::DuckLakeError::InvalidConfig(
                "set_sort_spec: at least one sort key is required (use reset_sort_spec to clear)"
                    .to_string(),
            ));
        }
        self.block_on(async {
            let mut tx = self.pool.begin().await?;
            let sort_id = reserve_catalog_id(&mut tx).await?;
            let (new_snapshot, _carried) = insert_snapshot(&mut tx).await?;
            // Official bumps the schema version on every sort-order change, with no
            // `ducklake_schema_versions` row (checked with the DuckDB CLI).
            bump_schema_version(&mut tx, new_snapshot).await?;

            // Validate every sort key resolves to a live column (v1 supports bare
            // column keys only), so a bad SET fails here rather than silently
            // producing unsorted writes later. A key binds to its column ignoring
            // case, as DuckDB's binder binds it.
            let names: Vec<String> = sqlx::query_scalar(
                "SELECT column_name FROM ducklake_column
                 WHERE table_id = $1 AND end_snapshot IS NULL AND parent_column IS NULL",
            )
            .bind(table_id)
            .fetch_all(&mut *tx)
            .await?;
            crate::sort::check_sort_key_columns(fields, &names)?;

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
            record_snapshot_changes(
                &mut tx,
                new_snapshot,
                &format!("altered_table:{table_id}"),
                &SnapshotCommitMetadata::default(),
            )
            .await?;
            self.commit(tx).await?;
            Ok(new_snapshot)
        })
    }

    fn reset_sort_spec(&self, table_id: i64) -> Result<i64> {
        self.block_on(async {
            let mut tx = self.pool.begin().await?;
            let (new_snapshot, _carried) = insert_snapshot(&mut tx).await?;
            bump_schema_version(&mut tx, new_snapshot).await?;
            sqlx::query(
                "UPDATE ducklake_sort_info SET end_snapshot = $1
                 WHERE table_id = $2 AND end_snapshot IS NULL",
            )
            .bind(new_snapshot)
            .bind(table_id)
            .execute(&mut *tx)
            .await?;
            record_snapshot_changes(
                &mut tx,
                new_snapshot,
                &format!("altered_table:{table_id}"),
                &SnapshotCommitMetadata::default(),
            )
            .await?;
            self.commit(tx).await?;
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
        self.block_on(async {
            let mut tx = self.pool.begin().await?;
            let ids = finalize_snapshot(
                &mut tx,
                schema_name,
                table_name,
                table_id,
                columns,
                column_ids,
                mode,
                base_snapshot,
                self.reserved_tables.take(table_id),
            )
            .await?;
            record_table_write_changes(
                &mut tx,
                ids.snapshot_id,
                ids.table_id,
                schema_name,
                table_name,
                mode,
                &SnapshotCommitMetadata::default(),
            )
            .await?;
            self.commit(tx).await?;
            Ok(ids)
        })
    }

    fn end_table_files(&self, table_id: i64, snapshot_id: i64) -> Result<u64> {
        // Used by WriteMode::Replace. `ducklake_table_stats` is left as it is, as
        // official's `DELETE` leaves it: `record_count` stays the gross count of
        // rows ever inserted, and `next_row_id` stays monotonic so historical
        // snapshots still resolve uniquely.
        self.block_on(async {
            let mut tx = self.pool.begin().await?;

            let result = sqlx::query(
                "UPDATE ducklake_data_file SET end_snapshot = $1
                 WHERE table_id = $2 AND end_snapshot IS NULL",
            )
            .bind(snapshot_id)
            .bind(table_id)
            .execute(&mut *tx)
            .await?;

            self.commit(tx).await?;
            Ok(result.rows_affected())
        })
    }

    fn get_data_path(&self) -> Result<String> {
        self.block_on(async {
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

    fn get_table_directory(&self, schema_name: &str, table_name: &str) -> Result<Option<String>> {
        let data_path = self.get_data_path()?;
        self.block_on(async {
            let row: Option<StoredTablePaths> = sqlx::query_as(
                "SELECT s.path, s.path_is_relative, t.path, t.path_is_relative
                 FROM ducklake_schema s
                 LEFT JOIN ducklake_table t
                     ON t.schema_id = s.schema_id AND t.table_name = $1 AND t.end_snapshot IS NULL
                 WHERE s.schema_name = $2 AND s.end_snapshot IS NULL",
            )
            .bind(table_name)
            .bind(schema_name)
            .fetch_optional(&self.pool)
            .await?;
            row.map(
                |(schema_path, schema_relative, table_path, table_relative)| {
                    table_directory(
                        &data_path,
                        (&schema_path, schema_relative.unwrap_or(true)),
                        (
                            &table_path.unwrap_or_else(|| directory_path(table_name)),
                            table_relative.unwrap_or(true),
                        ),
                    )
                },
            )
            .transpose()
        })
    }

    fn get_table_column_nullability(
        &self,
        schema_name: &str,
        table_name: &str,
    ) -> Result<Option<Vec<(String, bool)>>> {
        self.block_on(async {
            let rows = sqlx::query(
                "SELECT c.column_name, c.nulls_allowed
                 FROM ducklake_column c
                 JOIN ducklake_table t ON t.table_id = c.table_id
                 JOIN ducklake_schema s ON s.schema_id = t.schema_id
                 WHERE s.schema_name = $1 AND s.end_snapshot IS NULL
                   AND t.table_name = $2 AND t.end_snapshot IS NULL
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
        self.block_on(async {
            let current: Option<String> = sqlx::query_scalar(
                "SELECT value FROM ducklake_metadata WHERE key = 'data_path' AND scope IS NULL",
            )
            .fetch_optional(&self.pool)
            .await?;
            if current.as_deref() == Some(path.as_str()) {
                return Ok(());
            }

            let mut tx = self.pool.begin().await?;
            sqlx::query("DELETE FROM ducklake_metadata WHERE key = 'data_path' AND scope IS NULL")
                .execute(&mut *tx)
                .await?;

            sqlx::query(
                "INSERT INTO ducklake_metadata (key, value, scope)
                 VALUES ('data_path', $1, NULL)",
            )
            .bind(&path)
            .execute(&mut *tx)
            .await?;

            self.commit(tx).await?;
            Ok(())
        })
    }

    fn initialize_schema(&self) -> Result<()> {
        self.block_on(async {
            // sqlx runs each query() as a single prepared statement, so create each
            // table separately (see SQL_CREATE_TABLES).
            for ddl in SQL_CREATE_TABLES {
                sqlx::query(*ddl).execute(&self.pool).await?;
            }
            let function: bool =
                sqlx::query_scalar("SELECT to_regprocedure('ducklake_next_file_id()') IS NOT NULL")
                    .fetch_one(&self.pool)
                    .await?;
            if !function {
                sqlx::query(SQL_CREATE_NEXT_FILE_ID)
                    .execute(&self.pool)
                    .await?;
            }
            for (table, _, columns) in POSITIONAL_TABLES {
                sqlx::query(AssertSqlSafe(format!(
                    "CREATE TABLE IF NOT EXISTS {table} ({columns})"
                )))
                .execute(&self.pool)
                .await?;
            }
            // A catalog created elsewhere may have `changes_made` NOT NULL, but the
            // row is seeded NULL and filled in at commit.
            let changes_not_null: bool = sqlx::query_scalar(
                "SELECT attnotnull FROM pg_attribute
                 WHERE attrelid = to_regclass('ducklake_snapshot_changes')
                   AND attname = 'changes_made'",
            )
            .fetch_one(&self.pool)
            .await?;
            if changes_not_null {
                sqlx::query(
                    "ALTER TABLE ducklake_snapshot_changes ALTER COLUMN changes_made DROP NOT NULL",
                )
                .execute(&self.pool)
                .await?;
            }
            add_official_columns(&self.pool).await?;
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
            seed_counter(&self.pool, "next_catalog_id", SQL_MAX_CATALOG_ID).await?;
            seed_counter(&self.pool, "next_file_id", SQL_MAX_FILE_ID).await?;
            crate::metadata_writer_postgres::migrate_directory_paths(&self.pool).await?;
            sqlx::query(crate::metadata_writer::SQL_CANONICALIZE_LEGACY_BOOLEAN_SETTINGS)
                .execute(&self.pool)
                .await?;
            // A catalog this writer created before it kept the id watermarks has
            // none; raise the head's past every id in use.
            let mut tx = self.pool.begin().await?;
            raise_head_watermarks(&mut tx).await?;
            tx.commit().await?;
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
        self.block_on(async {
            let mut tx = self.pool.begin().await?;

            // Reserve the column ids first so the counter UPDATE takes the write
            // lock up front. These ids match the staged parquet field ids.
            let catalog_columns = catalog_column_defs(columns)?;
            let n = catalog_columns.len() as i64;
            reserve_ids(&mut tx, "next_column_id", n).await?;

            // The catalog head this write is based on; a Replace commit aborts if
            // another writer published a newer generation of the table past it.
            let base_snapshot_id: i64 =
                sqlx::query("SELECT COALESCE(MAX(snapshot_id), 0) FROM ducklake_snapshot")
                    .fetch_one(&mut *tx)
                    .await?
                    .try_get(0)?;

            // Informational only. The committed id is assigned by finalize_snapshot
            // from the counter, so under concurrency it may differ from this.
            let snapshot_id: i64 = base_snapshot_id + 1;

            // Look up, do NOT create. Reserving from the `next_catalog_id` counter
            // hands out an id without inserting a row, so a write that dies before
            // commit leaves nothing behind — schema/table visibility is
            // `snapshot >= begin_snapshot` with no check that the snapshot exists,
            // so a row written here against a guessed id would become readable the
            // moment any writer reached it. Unused reservations leave gaps, which
            // are expected and harmless.
            let schema_id: i64 = match sqlx::query_scalar(
                "SELECT schema_id FROM ducklake_schema
                 WHERE schema_name = $1 AND end_snapshot IS NULL",
            )
            .bind(schema_name)
            .fetch_optional(&mut *tx)
            .await?
            {
                Some(id) => id,
                None => reserve_catalog_id(&mut tx).await?,
            };

            let table_id: i64 = match sqlx::query_scalar(
                "SELECT t.table_id FROM ducklake_table t
                 JOIN ducklake_schema s ON s.schema_id = t.schema_id
                 WHERE s.schema_name = $1 AND s.end_snapshot IS NULL
                   AND t.table_name = $2 AND t.end_snapshot IS NULL",
            )
            .bind(schema_name)
            .bind(table_name)
            .fetch_optional(&mut *tx)
            .await?
            {
                Some(id) => id,
                None => {
                    let id = reserve_catalog_id(&mut tx).await?;
                    self.reserved_tables.insert(id);
                    id
                },
            };
            // Freshly reserved ids. Only a genuinely-new column actually consumes
            // one below; an existing column keeps its current id, so some of these
            // may go unused (harmless monotonic-counter gaps).
            let last_column_id = raise_column_ids_past_table(&mut tx, table_id, n).await?;
            let fresh_ids: Vec<i64> = ((last_column_id - n + 1)..=last_column_id).collect();

            // Get existing columns to (a) check schema compatibility for appends
            // and (b) REUSE each column's id (column_id == parquet field_id; an
            // unchanged column must keep its id, or files already written would
            // read back as NULL).
            let rows = sqlx::query(
                "SELECT column_name, column_type, column_id, parent_column
                 FROM ducklake_column
                 WHERE table_id = $1 AND end_snapshot IS NULL
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

            // A data write must not change a column's type — that is schema
            // evolution, and this backend has no promote_column_type. The comparison
            // is canonical (`int64` ≡ `bigint`). Append also requires a new column
            // to be nullable.
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

            let column_ids = top_level_column_ids(&catalog_columns, &field_ids)?;

            // Inserts nothing: this commit only persists the counter advance for the
            // column ids (and the sequence advances, which are non-transactional
            // anyway). Every metadata row is written by finalize_snapshot.
            self.commit(tx).await?;

            Ok(WriteSetupResult {
                snapshot_id,
                base_snapshot_id,
                schema_id,
                table_id,
                column_ids,
                field_ids,
            })
        })
    }
}
