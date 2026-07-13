//! PostgreSQL implementation of [`MetadataWriter`], multicatalog-aware from day one.
//!
//! Each `PostgresMetadataWriter` instance is bound to a single `catalog_id`. All
//! snapshot and schema inserts are paired with mapping-table inserts in the same
//! transaction, so cross-catalog isolation is enforced at write time.
//!
//! Schema-version allocation is per-catalog dense: a writer computes the next
//! `schema_version` under `FOR UPDATE` on the catalog's mapping rows, bumps on DDL
//! (table create or column-set change), and carries forward on DML (Append/Replace
//! with unchanged columns).

use crate::Result;
use crate::error::{TypeChangeOperation, TypeChangeWriteMode};
use crate::metadata_provider::block_on;
use crate::metadata_writer::{
    ColumnDef, CommitIds, DataFileInfo, DeleteFileEntry, DeleteFileInfo, MetadataWriter, WriteMode,
    WriteSetupResult, columns_differ, validate_delete_entries, validate_name,
};
use sqlx::Row;
use sqlx::postgres::{PgPool, PgPoolOptions};

const DEFAULT_MAX_CONNECTIONS: u32 = 5;

pub const DEFAULT_LOCK_TIMEOUT_MS: u32 = 30_000;

/// Each standard DuckLake table as a separate CREATE TABLE IF NOT EXISTS.
/// sqlx executes each `query()` as a single statement, so we split.
pub(crate) const SQL_CREATE_STANDARD_TABLES: &[&str] = &[
    r#"CREATE TABLE IF NOT EXISTS ducklake_metadata (
        key VARCHAR NOT NULL,
        value VARCHAR NOT NULL,
        scope VARCHAR
    )"#,
    r#"CREATE TABLE IF NOT EXISTS ducklake_snapshot (
        snapshot_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        snapshot_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        schema_version BIGINT NOT NULL DEFAULT 0
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
    // Multicatalog layout (NOT the upstream/DuckDB-readable single-catalog format,
    // so it is free to carry DB-level guarantees — design §4.1). `column_id` keeps
    // its IDENTITY (a global sequence the allocator reserves from), but is NO LONGER
    // a single-row PRIMARY KEY: a versioned / type-promoted column needs a second
    // row sharing the same `column_id`. Identity is the composite
    // (table_id, column_id, begin_snapshot); a partial unique index (below) enforces
    // at most one *live* version per field-id.
    r#"CREATE TABLE IF NOT EXISTS ducklake_column (
        column_id BIGINT GENERATED ALWAYS AS IDENTITY,
        table_id BIGINT NOT NULL,
        column_name VARCHAR NOT NULL,
        column_type VARCHAR NOT NULL,
        column_order BIGINT NOT NULL,
        nulls_allowed BOOLEAN DEFAULT TRUE,
        parent_column BIGINT,
        begin_snapshot BIGINT NOT NULL,
        end_snapshot BIGINT,
        PRIMARY KEY (table_id, column_id, begin_snapshot)
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
        partial_max BIGINT
    )"#,
    // Per-table running counters maintained inside the writer's transaction
    // so concurrent writes hand out non-overlapping rowid ranges. `next_row_id`
    // increases monotonically over the table's lifetime (rowids are never
    // reused, even after end-snapshot); `record_count` and `file_size_bytes`
    // mirror the currently-visible totals so DuckDB's `ducklake_table_info`
    // aggregate sees correct numbers for tables this writer produced. Mirrors
    // the sqlite writer's `ducklake_table_stats`.
    r#"CREATE TABLE IF NOT EXISTS ducklake_table_stats (
        table_id BIGINT PRIMARY KEY,
        record_count BIGINT NOT NULL DEFAULT 0,
        next_row_id BIGINT NOT NULL DEFAULT 0,
        file_size_bytes BIGINT NOT NULL DEFAULT 0
    )"#,
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
    // Idempotent guard: an existing single-catalog Postgres catalog populated by
    // another tool may not have schema_version on ducklake_snapshot.
    r#"ALTER TABLE ducklake_snapshot
        ADD COLUMN IF NOT EXISTS schema_version BIGINT NOT NULL DEFAULT 0"#,
    // Idempotent guard: an existing store may predate the v1.0 partial-file
    // marker. NULL means "not a partial file", correct for every pre-compaction
    // file.
    r#"ALTER TABLE ducklake_data_file
        ADD COLUMN IF NOT EXISTS partial_max BIGINT"#,
    // Per-snapshot change ledger (DuckLake spec). The compaction commit records
    // `compacted_table:<table_id>` here so DuckDB and other spec readers can
    // attribute the snapshot; other commit paths do not populate it yet.
    r#"CREATE TABLE IF NOT EXISTS ducklake_snapshot_changes (
        snapshot_id BIGINT PRIMARY KEY,
        changes_made VARCHAR NOT NULL,
        author VARCHAR,
        commit_message VARCHAR,
        commit_extra_info VARCHAR
    )"#,
];

/// Multicatalog scaffolding tables. Always run after the standard tables.
pub(crate) const SQL_CREATE_MULTICATALOG_TABLES: &[&str] = &[
    r#"CREATE TABLE IF NOT EXISTS ducklake_catalog (
        catalog_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        catalog_name VARCHAR NOT NULL UNIQUE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )"#,
    r#"CREATE TABLE IF NOT EXISTS ducklake_catalog_snapshot_map (
        catalog_id BIGINT NOT NULL,
        snapshot_id BIGINT NOT NULL,
        PRIMARY KEY (catalog_id, snapshot_id)
    )"#,
    r#"CREATE TABLE IF NOT EXISTS ducklake_catalog_schema_map (
        catalog_id BIGINT NOT NULL,
        schema_id BIGINT NOT NULL,
        PRIMARY KEY (catalog_id, schema_id)
    )"#,
    r#"CREATE TABLE IF NOT EXISTS ducklake_schema_versions (
        begin_snapshot BIGINT NOT NULL,
        schema_version BIGINT NOT NULL,
        table_id BIGINT NOT NULL,
        UNIQUE (table_id, begin_snapshot)
    )"#,
    // Files queued for physical deletion by the two-phase vacuum (DuckLake
    // spec). `expire_snapshots_in_catalog` GCs unreachable catalog rows and
    // records the orphaned physical paths here; `cleanup_old_files` deletes
    // the objects and removes these rows. `path` is stored relative to the
    // catalog `data_path` root (already resolved through schema/table) so
    // cleanup needs only a single-level join with `data_path`.
    //
    // Deviation from the single-catalog upstream schema: `catalog_id` scopes
    // each scheduled file to its catalog. Without it cleanup couldn't tell
    // catalogs apart — the data-file rows it would otherwise join against are
    // already deleted by the time the file is scheduled.
    r#"CREATE TABLE IF NOT EXISTS ducklake_files_scheduled_for_deletion (
        catalog_id BIGINT NOT NULL,
        data_file_id BIGINT NOT NULL,
        path VARCHAR NOT NULL,
        path_is_relative BOOLEAN NOT NULL DEFAULT TRUE,
        schedule_start TIMESTAMPTZ DEFAULT NOW()
    )"#,
    r#"CREATE INDEX IF NOT EXISTS idx_scheduled_for_deletion_catalog
        ON ducklake_files_scheduled_for_deletion(catalog_id)"#,
    r#"CREATE INDEX IF NOT EXISTS idx_catalog_snapshot_map_snapshot
        ON ducklake_catalog_snapshot_map(snapshot_id)"#,
    r#"CREATE INDEX IF NOT EXISTS idx_catalog_schema_map_schema
        ON ducklake_catalog_schema_map(schema_id)"#,
    r#"CREATE INDEX IF NOT EXISTS idx_schema_versions_table
        ON ducklake_schema_versions(table_id, begin_snapshot)"#,
    // Belt-and-suspenders: app-level lock_catalog should already prevent
    // duplicates, but a partial unique index catches anyone bypassing the
    // writer (manual SQL, external migrations).
    r#"CREATE UNIQUE INDEX IF NOT EXISTS idx_active_table_per_schema
        ON ducklake_table(schema_id, table_name) WHERE end_snapshot IS NULL"#,
    // At most one *live* version per field-id (design §4.1, reviews #2/#3). The
    // promote's retire-then-insert (end the old row, then insert the new live row,
    // in one txn) keeps this satisfied at every commit boundary.
    r#"CREATE UNIQUE INDEX IF NOT EXISTS idx_ducklake_column_live
        ON ducklake_column(table_id, column_id) WHERE end_snapshot IS NULL"#,
];

/// Run a slice of DDL statements against the pool. Each statement executes independently.
pub(crate) async fn execute_ddl_statements(pool: &PgPool, statements: &[&str]) -> Result<()> {
    for stmt in statements {
        sqlx::query(stmt).execute(pool).await?;
    }
    Ok(())
}

/// Upgrade an existing multicatalog store's `ducklake_column` from the legacy
/// single-row `column_id` PRIMARY KEY to the composite
/// `(table_id, column_id, begin_snapshot)` PK, so a versioned / type-promoted
/// column can have a second row sharing its `column_id`. `CREATE TABLE IF NOT
/// EXISTS` only shapes fresh stores; Postgres can `ALTER` a PK in place (unlike
/// SQLite). Idempotent (only acts when the current PK is the single-column one;
/// a no-op once composite) and lossless. The `IDENTITY` on `column_id` (the
/// allocator's sequence) is independent of the PK and survives the swap. The
/// partial unique index is created idempotently by `SQL_CREATE_STANDARD_TABLES`.
pub(crate) async fn migrate_ducklake_column_to_composite_pk(pool: &PgPool) -> Result<()> {
    sqlx::query(
        r#"DO $$
        DECLARE pk_name text;
        BEGIN
            -- Find the PRIMARY KEY iff it is a single-column PK (the legacy shape).
            SELECT conname INTO pk_name
            FROM pg_constraint
            WHERE conrelid = 'ducklake_column'::regclass
              AND contype = 'p'
              AND array_length(conkey, 1) = 1;
            IF pk_name IS NOT NULL THEN
                EXECUTE 'ALTER TABLE ducklake_column DROP CONSTRAINT ' || quote_ident(pk_name);
                EXECUTE 'ALTER TABLE ducklake_column ADD PRIMARY KEY (table_id, column_id, begin_snapshot)';
            END IF;
        END $$;"#,
    )
    .execute(pool)
    .await?;
    Ok(())
}

/// PostgreSQL-based metadata writer for DuckLake catalogs.
///
/// Bound to a single `catalog_id` at construction. To write to a different
/// catalog, construct a new writer with the desired `catalog_id`.
#[derive(Debug, Clone)]
pub struct PostgresMetadataWriter {
    pool: PgPool,
    catalog_id: i64,
    lock_timeout_ms: u32,
}

impl PostgresMetadataWriter {
    /// Bind a writer to the given pool and catalog id.
    ///
    /// Use [`crate::multicatalog::MulticatalogManager::create_catalog`] to obtain
    /// or create a catalog id by name.
    pub async fn with_pool(pool: PgPool, catalog_id: i64) -> Result<Self> {
        Ok(Self {
            pool,
            catalog_id,
            lock_timeout_ms: DEFAULT_LOCK_TIMEOUT_MS,
        })
    }

    pub async fn new(connection_string: &str, catalog_id: i64) -> Result<Self> {
        Self::with_max_connections(connection_string, catalog_id, DEFAULT_MAX_CONNECTIONS).await
    }

    pub async fn with_max_connections(
        connection_string: &str,
        catalog_id: i64,
        max_connections: u32,
    ) -> Result<Self> {
        let pool = PgPoolOptions::new()
            .max_connections(max_connections)
            .connect(connection_string)
            .await?;
        Ok(Self {
            pool,
            catalog_id,
            lock_timeout_ms: DEFAULT_LOCK_TIMEOUT_MS,
        })
    }

    /// Sets the Postgres `lock_timeout` (ms) applied before `FOR UPDATE`.
    /// `0` disables the timeout — not recommended for production.
    pub fn with_lock_timeout(mut self, ms: u32) -> Self {
        self.lock_timeout_ms = ms;
        self
    }

    pub fn catalog_id(&self) -> i64 {
        self.catalog_id
    }
}

async fn lock_catalog(
    catalog_id: i64,
    lock_timeout_ms: u32,
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
) -> Result<()> {
    if lock_timeout_ms > 0 {
        sqlx::query(&format!("SET LOCAL lock_timeout = {}", lock_timeout_ms))
            .execute(&mut **tx)
            .await?;
    }
    let row =
        sqlx::query("SELECT catalog_id FROM ducklake_catalog WHERE catalog_id = $1 FOR UPDATE")
            .bind(catalog_id)
            .fetch_optional(&mut **tx)
            .await?;
    if row.is_none() {
        return Err(crate::DuckLakeError::CatalogNotFound(format!(
            "catalog_id {}",
            catalog_id
        )));
    }
    Ok(())
}

async fn assert_schema_in_catalog(
    catalog_id: i64,
    schema_id: i64,
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
) -> Result<()> {
    let row = sqlx::query(
        "SELECT 1 FROM ducklake_catalog_schema_map
         WHERE catalog_id = $1 AND schema_id = $2",
    )
    .bind(catalog_id)
    .bind(schema_id)
    .fetch_optional(&mut **tx)
    .await?;
    if row.is_none() {
        return Err(crate::DuckLakeError::InvalidConfig(format!(
            "schema_id {} does not belong to catalog_id {}",
            schema_id, catalog_id
        )));
    }
    Ok(())
}

async fn assert_table_in_catalog(
    catalog_id: i64,
    table_id: i64,
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
) -> Result<()> {
    let row = sqlx::query(
        "SELECT m.catalog_id FROM ducklake_table t
         LEFT JOIN ducklake_catalog_schema_map m ON m.schema_id = t.schema_id
         WHERE t.table_id = $1",
    )
    .bind(table_id)
    .fetch_optional(&mut **tx)
    .await?;
    match row {
        None => Err(crate::DuckLakeError::TableNotFound(format!(
            "table_id {}",
            table_id
        ))),
        Some(r) => {
            let owner: Option<i64> = r.try_get(0)?;
            if owner != Some(catalog_id) {
                Err(crate::DuckLakeError::InvalidConfig(format!(
                    "table_id {} does not belong to catalog_id {}",
                    table_id, catalog_id
                )))
            } else {
                Ok(())
            }
        },
    }
}

/// Reject only a `table_id` hint that exists and belongs to ANOTHER catalog. A
/// hint that does not yet exist is fine — it is the id reserved at begin that
/// `finalize_snapshot` is about to create under this catalog (first write to a
/// new table). Used by the commit path (`register_data_file`/`publish_snapshot`),
/// which must tolerate a not-yet-created table while still catching a caller that
/// hands in a different catalog's table.
async fn assert_table_not_in_other_catalog(
    catalog_id: i64,
    table_id: i64,
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
) -> Result<()> {
    let row = sqlx::query(
        "SELECT m.catalog_id FROM ducklake_table t
         LEFT JOIN ducklake_catalog_schema_map m ON m.schema_id = t.schema_id
         WHERE t.table_id = $1",
    )
    .bind(table_id)
    .fetch_optional(&mut **tx)
    .await?;
    if let Some(r) = row {
        let owner: Option<i64> = r.try_get(0)?;
        if owner != Some(catalog_id) {
            return Err(crate::DuckLakeError::InvalidConfig(format!(
                "table_id {} does not belong to catalog_id {}",
                table_id, catalog_id
            )));
        }
    }
    Ok(())
}

/// Reserve `n` ids from the IDENTITY-backing sequence of `table.col` WITHOUT
/// inserting rows, so begin can hand out column/schema/table ids (column ids are
/// the parquet field-ids baked into the staged file) that the commit later
/// inserts explicitly via `OVERRIDING SYSTEM VALUE`. Sequences are
/// non-transactional, so gaps from an aborted write are fine and expected. The
/// ids come back in order.
async fn reserve_ids(
    table: &str,
    col: &str,
    n: i64,
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
) -> Result<Vec<i64>> {
    if n <= 0 {
        return Ok(Vec::new());
    }
    let rows =
        sqlx::query("SELECT nextval(pg_get_serial_sequence($1, $2)) FROM generate_series(1, $3)")
            .bind(table)
            .bind(col)
            .bind(n)
            .fetch_all(&mut **tx)
            .await?;
    rows.into_iter().map(|r| Ok(r.try_get(0)?)).collect()
}

/// Optimistic-concurrency check for a `Replace` commit. Run under the catalog
/// `FOR UPDATE` lock at the commit point, BEFORE this writer inserts its own
/// files/columns and before `advance_catalog_head`. Because snapshot ids are
/// assigned at commit (id order == commit order per catalog) and all metadata is
/// written at commit (no dormant rows), this scalar check is exact: if any data
/// file OR column of the table has `begin_snapshot` or `end_snapshot` > `base`
/// (the catalog head observed at begin), another writer committed a generation
/// of this table since this write began ⇒ [`DuckLakeError::Conflict`]. Catches a
/// data Replace (new file begin), a fileless `CREATE`/Replace (new column begin),
/// and a DROP (end-stamp). The writer's own rows are not written yet, so the
/// check never self-conflicts. (`Append` does not call this: concurrent appends
/// commute, matching upstream DuckLake.)
async fn detect_replace_conflict(
    table_id: i64,
    base_snapshot: i64,
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
) -> Result<()> {
    let conflict = sqlx::query(
        "SELECT 1 WHERE EXISTS (SELECT 1 FROM ducklake_data_file
             WHERE table_id = $1 AND (begin_snapshot > $2 OR end_snapshot > $2))
           OR EXISTS (SELECT 1 FROM ducklake_column
             WHERE table_id = $1 AND (begin_snapshot > $2 OR end_snapshot > $2))",
    )
    .bind(table_id)
    .bind(base_snapshot)
    .fetch_optional(&mut **tx)
    .await?;
    if conflict.is_some() {
        return Err(crate::DuckLakeError::Conflict(format!(
            "Replace on table {table_id} conflicts with a concurrent write committed since \
             snapshot {base_snapshot}; aborting (retry the write against the new generation)"
        )));
    }
    Ok(())
}

/// Retire the generation preceding `snapshot_id` for a Replace: end-snapshot
/// every still-live file from an earlier snapshot and zero the visible
/// record/byte totals. The `begin_snapshot < snapshot_id` guard leaves the
/// current write's own files untouched (multi-file safety); `next_row_id` stays
/// monotonic so rowids are never reused.
async fn retire_prior_generation(
    table_id: i64,
    snapshot_id: i64,
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
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
        "UPDATE ducklake_table_stats
         SET record_count = 0, file_size_bytes = 0
         WHERE table_id = $1",
    )
    .bind(table_id)
    .execute(&mut **tx)
    .await?;
    Ok(())
}

/// Publish `snapshot_id` as the catalog head by mapping it to the catalog.
/// Idempotent (the write path calls it once, but a retried/multi-file commit
/// must not fail on the PK).
async fn advance_catalog_head(
    catalog_id: i64,
    snapshot_id: i64,
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
) -> Result<()> {
    sqlx::query(
        "INSERT INTO ducklake_catalog_snapshot_map (catalog_id, snapshot_id)
         VALUES ($1, $2)
         ON CONFLICT (catalog_id, snapshot_id) DO NOTHING",
    )
    .bind(catalog_id)
    .bind(snapshot_id)
    .execute(&mut **tx)
    .await?;
    Ok(())
}

/// SQL expression resolving a `df`-aliased file row's path relative to the
/// catalog `data_path` root (file → table → schema). Mirrors the multicatalog
/// expire path's `PG_RESOLVED_PATH`; duplicated here (that one is private to
/// `multicatalog`) so the compaction commit can schedule retired files.
const COMPACTION_RESOLVED_PATH: &str = "CASE
    WHEN NOT df.path_is_relative THEN df.path
    WHEN NOT t.path_is_relative THEN t.path || '/' || df.path
    ELSE s.path || '/' || t.path || '/' || df.path
END";

/// Companion to [`COMPACTION_RESOLVED_PATH`]: true only when the whole chain is relative.
const COMPACTION_REL_FLAG: &str =
    "(df.path_is_relative AND t.path_is_relative AND s.path_is_relative)";

/// Insert `(id, resolved_path, rel)` rows (as produced by
/// [`COMPACTION_RESOLVED_PATH`] / [`COMPACTION_REL_FLAG`]) into
/// `ducklake_files_scheduled_for_deletion`, scoped to `catalog_id`. Mirrors the
/// multicatalog expire path's `schedule_pg_files`.
async fn schedule_compaction_files(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    catalog_id: i64,
    rows: Vec<sqlx::postgres::PgRow>,
) -> Result<()> {
    for row in rows {
        let id: i64 = row.try_get(0)?;
        let path: String = row.try_get(1)?;
        let rel: bool = row.try_get(2)?;
        sqlx::query(
            "INSERT INTO ducklake_files_scheduled_for_deletion
                 (catalog_id, data_file_id, path, path_is_relative, schedule_start)
             VALUES ($1, $2, $3, $4, NOW())",
        )
        .bind(catalog_id)
        .bind(id)
        .bind(&path)
        .bind(rel)
        .execute(&mut **tx)
        .await?;
    }
    Ok(())
}

/// The atomic commit point for a multicatalog Postgres write, shared by
/// `register_data_file` (with a data file) and `publish_snapshot` (fileless).
/// The CALLER already holds the catalog `FOR UPDATE` lock and an open tx.
///
/// All metadata — the snapshot row, the get-or-create schema/table rows, the
/// column generation, the `schema_versions` row, and the Replace retirement — is
/// written HERE so nothing is visible until `advance_catalog_head` maps the
/// snapshot (the caller runs that LAST). The `snapshot_id` is a plain IDENTITY
/// insert, so per-catalog id order == commit order, which is what makes the
/// scalar [`detect_replace_conflict`] and the dense schema_version computation
/// exact. The reserved schema/table/column ids from begin are inserted with
/// `OVERRIDING SYSTEM VALUE`; the reused column ids keep parquet field-ids stable.
///
/// Returns `(committed_snapshot_id, table_id)`.
///
/// `table_id_hint` is the id reserved for the table at begin; it is used only
/// when the table does not yet exist (first write). The schema id is re-derived
/// here — looked up if the schema already exists, else a fresh id is reserved
/// from the sequence — because the reserved schema id from begin is not threaded
/// through the commit (it is never baked into anything; the parquet path encodes
/// the catalog id, not the schema id).
#[allow(clippy::too_many_arguments)]
async fn finalize_snapshot(
    catalog_id: i64,
    schema_name: &str,
    table_name: &str,
    table_id_hint: i64,
    columns: &[ColumnDef],
    column_ids: &[i64],
    mode: WriteMode,
    base_snapshot: i64,
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
) -> Result<(i64, i64, i64)> {
    // 1. Resolve the live schema id under the lock. Reuse it if present; else
    //    reserve a fresh id from the sequence (the row is inserted in step 4 once
    //    the snapshot id exists for begin_snapshot).
    let (schema_id, schema_was_created): (i64, bool) = {
        let existing = sqlx::query(
            "SELECT s.schema_id FROM ducklake_schema s
             JOIN ducklake_catalog_schema_map m ON m.schema_id = s.schema_id
             WHERE m.catalog_id = $1 AND s.schema_name = $2 AND s.end_snapshot IS NULL",
        )
        .bind(catalog_id)
        .bind(schema_name)
        .fetch_optional(&mut **tx)
        .await?;
        if let Some(row) = existing {
            (row.try_get(0)?, false)
        } else {
            let id = reserve_ids("ducklake_schema", "schema_id", 1, tx).await?[0];
            (id, true)
        }
    };

    // 2. Conflict check for Replace, BEFORE inserting any of this writer's rows.
    //    Resolve the table id first; a brand-new table has no prior generation to
    //    conflict with, so skip the check (base == head, no rows exist yet).
    if mode == WriteMode::Replace && !schema_was_created {
        let existing_table_id: Option<i64> = sqlx::query(
            "SELECT table_id FROM ducklake_table
             WHERE schema_id = $1 AND table_name = $2 AND end_snapshot IS NULL",
        )
        .bind(schema_id)
        .bind(table_name)
        .fetch_optional(&mut **tx)
        .await?
        .map(|r| r.try_get(0))
        .transpose()?;
        if let Some(tid) = existing_table_id {
            detect_replace_conflict(tid, base_snapshot, tx).await?;
        }
    }

    // 3. Allocate the snapshot id in commit order (plain IDENTITY). schema_version
    //    is patched in below once we know it.
    let snapshot_id: i64 = sqlx::query(
        "INSERT INTO ducklake_snapshot (snapshot_time, schema_version)
         VALUES (NOW(), 0) RETURNING snapshot_id",
    )
    .fetch_one(&mut **tx)
    .await?
    .try_get(0)?;

    // 4. Insert the schema row (with its reserved id) if it is new. The catalog id
    //    is encoded into the schema's *path* (not its name) so two catalogs holding
    //    their own `public` land in disjoint physical subtrees: the reader's
    //    resolution chain (`data_path + schema.path + table.path + file.path`) then
    //    puts files under `cat_{id}/{schema}/{table}/…`, matching the upload
    //    location.
    if schema_was_created {
        let scoped_schema_path = format!("cat_{catalog_id}/{schema_name}");
        sqlx::query(
            "INSERT INTO ducklake_schema
                 (schema_id, schema_name, path, path_is_relative, begin_snapshot)
             OVERRIDING SYSTEM VALUE
             VALUES ($1, $2, $3, TRUE, $4)",
        )
        .bind(schema_id)
        .bind(schema_name)
        .bind(&scoped_schema_path)
        .bind(snapshot_id)
        .execute(&mut **tx)
        .await?;
        sqlx::query(
            "INSERT INTO ducklake_catalog_schema_map (catalog_id, schema_id)
             VALUES ($1, $2)",
        )
        .bind(catalog_id)
        .bind(schema_id)
        .execute(&mut **tx)
        .await?;
    }

    // 5. get-or-create table under the lock. Capture whether we created it.
    let (table_id, table_was_created): (i64, bool) = {
        let existing = sqlx::query(
            "SELECT table_id FROM ducklake_table
             WHERE schema_id = $1 AND table_name = $2 AND end_snapshot IS NULL",
        )
        .bind(schema_id)
        .bind(table_name)
        .fetch_optional(&mut **tx)
        .await?;
        if let Some(row) = existing {
            (row.try_get(0)?, false)
        } else {
            sqlx::query(
                "INSERT INTO ducklake_table
                     (table_id, schema_id, table_name, path, path_is_relative, begin_snapshot)
                 OVERRIDING SYSTEM VALUE
                 VALUES ($1, $2, $3, $4, TRUE, $5)",
            )
            .bind(table_id_hint)
            .bind(schema_id)
            .bind(table_name)
            .bind(table_name)
            .bind(snapshot_id)
            .execute(&mut **tx)
            .await?;
            (table_id_hint, true)
        }
    };

    // 6. Read the columns live AT COMMIT (under the lock) to classify DDL vs DML
    //    and drive the surgical column update below.
    let existing_column_rows = sqlx::query(
        "SELECT column_name, column_type, nulls_allowed, column_order, column_id
         FROM ducklake_column
         WHERE table_id = $1 AND end_snapshot IS NULL
         ORDER BY column_order",
    )
    .bind(table_id)
    .fetch_all(&mut **tx)
    .await?;
    let existing_columns: Vec<(String, String, bool)> = existing_column_rows
        .iter()
        .map(|row| {
            let name: String = row.try_get(0)?;
            let col_type: String = row.try_get(1)?;
            let nullable: bool = row.try_get::<Option<bool>, _>(2)?.unwrap_or(true);
            Ok::<_, sqlx::Error>((name, col_type, nullable))
        })
        .collect::<std::result::Result<_, _>>()?;
    // name -> (column_id, column_order, nullable) for the surgical update. The
    // column_id is used to detect field-id drift: if the caller's staged parquet
    // baked a column_id (at begin) that no longer matches the committed column
    // (e.g. an Append whose table was created by a concurrent writer with
    // different ids), the file's field-ids would resolve to NULL — so we abort.
    let mut current_by_name: std::collections::HashMap<String, (i64, i64, bool)> =
        std::collections::HashMap::new();
    for row in &existing_column_rows {
        let name: String = row.try_get(0)?;
        let nullable: bool = row.try_get::<Option<bool>, _>(2)?.unwrap_or(true);
        let order: i64 = row.try_get(3)?;
        let id: i64 = row.try_get(4)?;
        current_by_name.insert(name, (id, order, nullable));
    }

    let is_ddl = table_was_created || columns_differ(&existing_columns, columns);

    // No `< S` window: ids are commit-ordered, so MAX over mapped predecessors is
    // the immediately-preceding version. DDL bumps; DML carries forward (with a v1
    // floor for the very first write to the catalog).
    let prev_max: i64 = sqlx::query(
        "SELECT COALESCE(MAX(s.schema_version), 0) FROM ducklake_snapshot s
         JOIN ducklake_catalog_snapshot_map m ON m.snapshot_id = s.snapshot_id
         WHERE m.catalog_id = $1",
    )
    .bind(catalog_id)
    .fetch_one(&mut **tx)
    .await?
    .try_get(0)?;
    let new_schema_version = if is_ddl {
        prev_max + 1
    } else if prev_max == 0 {
        1
    } else {
        prev_max
    };
    sqlx::query("UPDATE ducklake_snapshot SET schema_version = $1 WHERE snapshot_id = $2")
        .bind(new_schema_version)
        .bind(snapshot_id)
        .execute(&mut **tx)
        .await?;

    // 7. Write the column generation SURGICALLY (mode-independent, matching the
    //    SQLite writer) so each kept column keeps a STABLE column_id (== parquet
    //    field_id). End only removed columns, insert only genuinely-new ones (with
    //    their reserved ids), and sync order/nullability on the rest in place.
    //    Stable ids are required even for Replace: a concurrent in-flight Append
    //    baked the kept columns' ids into its parquet, so re-minting them would make
    //    that Append's rows read back as all-NULL. The prior generation's retired
    //    files keep their old ids for time travel. (The Replace conflict check does
    //    not depend on a column re-mint — see the data-file/column scan above.)
    {
        use std::collections::HashSet;
        let new_names: HashSet<&str> = columns.iter().map(|c| c.name.as_str()).collect();
        for name in current_by_name.keys() {
            if !new_names.contains(name.as_str()) {
                sqlx::query(
                    "UPDATE ducklake_column SET end_snapshot = $1
                     WHERE table_id = $2 AND column_name = $3 AND end_snapshot IS NULL",
                )
                .bind(snapshot_id)
                .bind(table_id)
                .bind(name)
                .execute(&mut **tx)
                .await?;
            }
        }
        for (order, (col, column_id)) in columns.iter().zip(column_ids.iter()).enumerate() {
            match current_by_name.get(&col.name) {
                Some(&(cur_id, cur_order, cur_nullable)) => {
                    // Field-id drift: the staged parquet baked `*column_id` for this
                    // column at begin, but the committed column now has a different
                    // id (a concurrent writer created the table/column with other
                    // ids between this writer's begin and commit). Registering the
                    // file would make this column read back as all-NULL, so abort —
                    // the caller retries against the now-committed schema. (Append
                    // is otherwise not conflict-checked; this guards correctness,
                    // not isolation.)
                    if *column_id != cur_id {
                        return Err(crate::DuckLakeError::Conflict(format!(
                            "column '{}' of table {table_id} was created concurrently with a \
                             different field id ({cur_id}, staged {column_id}); aborting to avoid \
                             a NULL-filled read (retry the write)",
                            col.name
                        )));
                    }
                    if cur_order != order as i64 || cur_nullable != col.is_nullable {
                        sqlx::query(
                            "UPDATE ducklake_column SET column_order = $1, nulls_allowed = $2
                             WHERE table_id = $3 AND column_name = $4 AND end_snapshot IS NULL",
                        )
                        .bind(order as i64)
                        .bind(col.is_nullable)
                        .bind(table_id)
                        .bind(&col.name)
                        .execute(&mut **tx)
                        .await?;
                    }
                },
                None => {
                    sqlx::query(
                        "INSERT INTO ducklake_column
                             (column_id, table_id, column_name, column_type, column_order,
                              nulls_allowed, begin_snapshot)
                         OVERRIDING SYSTEM VALUE
                         VALUES ($1, $2, $3, $4, $5, $6, $7)",
                    )
                    .bind(column_id)
                    .bind(table_id)
                    .bind(&col.name)
                    .bind(&col.ducklake_type)
                    .bind(order as i64)
                    .bind(col.is_nullable)
                    .bind(snapshot_id)
                    .execute(&mut **tx)
                    .await?;
                },
            }
        }
    }

    // 8. One ducklake_schema_versions row per DDL (UNIQUE(table_id, begin_snapshot)).
    if is_ddl {
        sqlx::query(
            "INSERT INTO ducklake_schema_versions (begin_snapshot, schema_version, table_id)
             VALUES ($1, $2, $3)",
        )
        .bind(snapshot_id)
        .bind(new_schema_version)
        .bind(table_id)
        .execute(&mut **tx)
        .await?;
    }

    // 9. Replace retirement: end the prior generation's files + zero the visible
    //    totals. The `begin_snapshot < S` guard spares this write's own files.
    if mode == WriteMode::Replace {
        sqlx::query(
            "INSERT INTO ducklake_table_stats (table_id, record_count, next_row_id, file_size_bytes)
             VALUES ($1, 0, 0, 0)
             ON CONFLICT (table_id) DO NOTHING",
        )
        .bind(table_id)
        .execute(&mut **tx)
        .await?;
        retire_prior_generation(table_id, snapshot_id, tx).await?;
    }

    Ok((snapshot_id, schema_id, table_id))
}

impl MetadataWriter for PostgresMetadataWriter {
    fn create_snapshot(&self) -> Result<i64> {
        block_on(async {
            let mut tx = self.pool.begin().await?;

            let row = sqlx::query(
                "INSERT INTO ducklake_snapshot (snapshot_time, schema_version)
                 VALUES (CURRENT_TIMESTAMP, 0) RETURNING snapshot_id",
            )
            .fetch_one(&mut *tx)
            .await?;
            let snapshot_id: i64 = row.try_get(0)?;

            sqlx::query(
                "INSERT INTO ducklake_catalog_snapshot_map (catalog_id, snapshot_id)
                 VALUES ($1, $2)",
            )
            .bind(self.catalog_id)
            .bind(snapshot_id)
            .execute(&mut *tx)
            .await?;

            tx.commit().await?;
            Ok(snapshot_id)
        })
    }

    fn promote_column_type(
        &self,
        table_id: i64,
        column_name: &str,
        new_ducklake_type: &str,
    ) -> Result<i64> {
        // Reject an unknown target type before opening a transaction.
        crate::types::ducklake_to_arrow_type(new_ducklake_type)?;
        block_on(async {
            let mut tx = self.pool.begin().await?;
            lock_catalog(self.catalog_id, self.lock_timeout_ms, &mut tx).await?;
            // Ownership guard (matches every other table_id-taking mutator,
            // e.g. set_columns / end_table_files): table_ids are global across the
            // multicatalog store, so refuse a table_id that belongs to a different
            // catalog — otherwise a promote scoped to this catalog could silently
            // mutate another catalog's column.
            assert_table_in_catalog(self.catalog_id, table_id, &mut tx).await?;

            // Live version of the column.
            let row = sqlx::query(
                "SELECT column_id, column_type, column_order, nulls_allowed
                 FROM ducklake_column
                 WHERE table_id = $1 AND column_name = $2 AND end_snapshot IS NULL",
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
            let cur_type: String = row.try_get("column_type")?;
            let column_order: i64 = row.try_get("column_order")?;
            let nulls_allowed: bool = row
                .try_get::<Option<bool>, _>("nulls_allowed")?
                .unwrap_or(true);

            // No-op / not-a-widening guards (canonical first so an alias-only
            // restatement is "no change", not attempted).
            if crate::types::types_equal_canonical(&cur_type, new_ducklake_type) {
                return Err(crate::DuckLakeError::InvalidConfig(format!(
                    "promote_column_type: column '{column_name}' is already type '{cur_type}' (no change)"
                )));
            }
            if !crate::types::is_promotable(&cur_type, new_ducklake_type) {
                return Err(crate::DuckLakeError::UnsupportedTypeChange {
                    operation: TypeChangeOperation::PromoteColumnType,
                    column: column_name.to_string(),
                    from: cur_type,
                    to: new_ducklake_type.to_string(),
                });
            }

            // New snapshot + advance this catalog's head.
            let snapshot_id: i64 = sqlx::query(
                "INSERT INTO ducklake_snapshot (snapshot_time, schema_version)
                 VALUES (NOW(), 0) RETURNING snapshot_id",
            )
            .fetch_one(&mut *tx)
            .await?
            .try_get(0)?;
            sqlx::query(
                "INSERT INTO ducklake_catalog_snapshot_map (catalog_id, snapshot_id) VALUES ($1, $2)",
            )
            .bind(self.catalog_id)
            .bind(snapshot_id)
            .execute(&mut *tx)
            .await?;

            // A promote IS schema evolution → bump schema_version (per-catalog dense)
            // and record the ledger row (same model as a DDL data-write commit).
            let prev_max: i64 = sqlx::query(
                "SELECT COALESCE(MAX(s.schema_version), 0) FROM ducklake_snapshot s
                 JOIN ducklake_catalog_snapshot_map m ON m.snapshot_id = s.snapshot_id
                 WHERE m.catalog_id = $1 AND s.snapshot_id <> $2",
            )
            .bind(self.catalog_id)
            .bind(snapshot_id)
            .fetch_one(&mut *tx)
            .await?
            .try_get(0)?;
            let new_schema_version = prev_max + 1;
            sqlx::query("UPDATE ducklake_snapshot SET schema_version = $1 WHERE snapshot_id = $2")
                .bind(new_schema_version)
                .bind(snapshot_id)
                .execute(&mut *tx)
                .await?;
            sqlx::query(
                "INSERT INTO ducklake_schema_versions (begin_snapshot, schema_version, table_id)
                 VALUES ($1, $2, $3)",
            )
            .bind(snapshot_id)
            .bind(new_schema_version)
            .bind(table_id)
            .execute(&mut *tx)
            .await?;

            // Retire the live row, insert the new version with the SAME column_id
            // (OVERRIDING SYSTEM VALUE — column_id is IDENTITY). Retire-before-insert
            // keeps the live-version partial unique index satisfied at all times.
            sqlx::query(
                "UPDATE ducklake_column SET end_snapshot = $1
                 WHERE table_id = $2 AND column_id = $3 AND end_snapshot IS NULL",
            )
            .bind(snapshot_id)
            .bind(table_id)
            .bind(column_id)
            .execute(&mut *tx)
            .await?;
            sqlx::query(
                "INSERT INTO ducklake_column
                     (column_id, table_id, column_name, column_type, column_order, nulls_allowed, begin_snapshot)
                 OVERRIDING SYSTEM VALUE
                 VALUES ($1, $2, $3, $4, $5, $6, $7)",
            )
            .bind(column_id)
            .bind(table_id)
            .bind(column_name)
            .bind(new_ducklake_type)
            .bind(column_order)
            .bind(nulls_allowed)
            .bind(snapshot_id)
            .execute(&mut *tx)
            .await?;

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
            lock_catalog(self.catalog_id, self.lock_timeout_ms, &mut tx).await?;

            let existing = sqlx::query(
                "SELECT s.schema_id FROM ducklake_schema s
                 JOIN ducklake_catalog_schema_map m ON m.schema_id = s.schema_id
                 WHERE m.catalog_id = $1 AND s.schema_name = $2 AND s.end_snapshot IS NULL",
            )
            .bind(self.catalog_id)
            .bind(name)
            .fetch_optional(&mut *tx)
            .await?;

            if let Some(row) = existing {
                let id: i64 = row.try_get(0)?;
                tx.commit().await?;
                return Ok((id, false));
            }

            let schema_path = path.unwrap_or(name);
            let row = sqlx::query(
                "INSERT INTO ducklake_schema (schema_name, path, path_is_relative, begin_snapshot)
                 VALUES ($1, $2, TRUE, $3) RETURNING schema_id",
            )
            .bind(name)
            .bind(schema_path)
            .bind(snapshot_id)
            .fetch_one(&mut *tx)
            .await?;
            let schema_id: i64 = row.try_get(0)?;

            sqlx::query(
                "INSERT INTO ducklake_catalog_schema_map (catalog_id, schema_id)
                 VALUES ($1, $2)",
            )
            .bind(self.catalog_id)
            .bind(schema_id)
            .execute(&mut *tx)
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
            lock_catalog(self.catalog_id, self.lock_timeout_ms, &mut tx).await?;
            assert_schema_in_catalog(self.catalog_id, schema_id, &mut tx).await?;

            let existing = sqlx::query(
                "SELECT table_id FROM ducklake_table
                 WHERE schema_id = $1 AND table_name = $2 AND end_snapshot IS NULL",
            )
            .bind(schema_id)
            .bind(name)
            .fetch_optional(&mut *tx)
            .await?;

            if let Some(row) = existing {
                let id: i64 = row.try_get(0)?;
                tx.commit().await?;
                return Ok((id, false));
            }

            let table_path = path.unwrap_or(name);
            let row = sqlx::query(
                "INSERT INTO ducklake_table (schema_id, table_name, path, path_is_relative, begin_snapshot)
                 VALUES ($1, $2, $3, TRUE, $4) RETURNING table_id",
            )
            .bind(schema_id)
            .bind(name)
            .bind(table_path)
            .bind(snapshot_id)
            .fetch_one(&mut *tx)
            .await?;
            let id: i64 = row.try_get(0)?;

            tx.commit().await?;
            Ok((id, true))
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
            let mut tx = self.pool.begin().await?;
            lock_catalog(self.catalog_id, self.lock_timeout_ms, &mut tx).await?;
            assert_table_in_catalog(self.catalog_id, table_id, &mut tx).await?;

            sqlx::query(
                "UPDATE ducklake_column SET end_snapshot = $1
                 WHERE table_id = $2 AND end_snapshot IS NULL",
            )
            .bind(snapshot_id)
            .bind(table_id)
            .execute(&mut *tx)
            .await?;

            let mut column_ids = Vec::with_capacity(columns.len());
            for (order, col) in columns.iter().enumerate() {
                let row = sqlx::query(
                    "INSERT INTO ducklake_column (table_id, column_name, column_type, column_order, nulls_allowed, begin_snapshot)
                     VALUES ($1, $2, $3, $4, $5, $6) RETURNING column_id",
                )
                .bind(table_id)
                .bind(&col.name)
                .bind(&col.ducklake_type)
                .bind(order as i64)
                .bind(col.is_nullable)
                .bind(snapshot_id)
                .fetch_one(&mut *tx)
                .await?;
                column_ids.push(row.try_get(0)?);
            }

            tx.commit().await?;
            Ok(column_ids)
        })
    }

    fn register_data_file(
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
    ) -> Result<CommitIds> {
        block_on(async {
            // Single atomic commit. finalize_snapshot writes ALL metadata (the
            // snapshot row, get-or-create schema/table, the column generation, the
            // schema_versions row, and the Replace retirement) and returns the
            // committed snapshot id + real table id. We then register the file and
            // advance the catalog head LAST, so nothing is visible until the head
            // maps the snapshot. row_id_start is drawn from the table's monotonic
            // counter under the catalog lock so concurrent writers hand out
            // non-overlapping ranges; the stats row is seeded for tables created
            // before this writer maintained it. The passed `table_id` is the id
            // reserved at begin (== the committed id); we tolerate it not existing
            // yet (first write) but reject another catalog's id.
            let mut tx = self.pool.begin().await?;
            lock_catalog(self.catalog_id, self.lock_timeout_ms, &mut tx).await?;
            assert_table_not_in_other_catalog(self.catalog_id, table_id, &mut tx).await?;

            let (snapshot_id, schema_id, table_id) = finalize_snapshot(
                self.catalog_id,
                schema_name,
                table_name,
                table_id,
                columns,
                column_ids,
                mode,
                base_snapshot,
                &mut tx,
            )
            .await?;

            sqlx::query(
                "INSERT INTO ducklake_table_stats (table_id, record_count, next_row_id, file_size_bytes)
                 VALUES ($1, 0, 0, 0)
                 ON CONFLICT (table_id) DO NOTHING",
            )
            .bind(table_id)
            .execute(&mut *tx)
            .await?;

            let stats_row =
                sqlx::query("SELECT next_row_id FROM ducklake_table_stats WHERE table_id = $1")
                    .bind(table_id)
                    .fetch_one(&mut *tx)
                    .await?;
            let row_id_start: i64 = stats_row.try_get(0)?;

            let inserted = sqlx::query(
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
            .await?;
            let _data_file_id: i64 = inserted.try_get(0)?;

            // Advance the counter and accumulate stats. `next_row_id`
            // monotonically increases over the table's lifetime — rowids
            // are never reused, even after end-snapshot. For Replace the
            // record/byte totals were just zeroed, so this leaves them at the
            // new file's values.
            sqlx::query(
                "UPDATE ducklake_table_stats
                 SET next_row_id     = next_row_id + $1,
                     record_count    = record_count + $2,
                     file_size_bytes = file_size_bytes + $3
                 WHERE table_id = $4",
            )
            .bind(file.record_count)
            .bind(file.record_count)
            .bind(file.file_size_bytes)
            .bind(table_id)
            .execute(&mut *tx)
            .await?;

            // advance_catalog_head MUST be the last write before commit.
            advance_catalog_head(self.catalog_id, snapshot_id, &mut tx).await?;

            tx.commit().await?;
            Ok(CommitIds {
                snapshot_id,
                schema_id,
                table_id,
            })
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
            // Single atomic commit under the catalog lock: fence on
            // `base_snapshot`, compare-and-swap the currently-live delete file
            // for this data file, allocate the snapshot, retire the prior delete
            // file, insert the new cumulative one, and advance the catalog head
            // LAST — so at most one delete file is ever live per data file and
            // nothing is visible until the head maps the snapshot.
            let mut tx = self.pool.begin().await?;
            lock_catalog(self.catalog_id, self.lock_timeout_ms, &mut tx).await?;
            assert_table_not_in_other_catalog(self.catalog_id, table_id, &mut tx).await?;

            // Target-file fence: the resolved positions are physical row indices
            // in `data_file_id`, and a parquet data file is immutable — so only a
            // concurrent write that RETIRED this file (a Replace/compaction) since
            // `base_snapshot` can invalidate them. An append that adds *other*
            // files does not move this file's rows, and a concurrent delete on
            // THIS file is caught by the compare-and-swap below; neither must
            // block the delete. Abort iff the target is no longer the live file.
            // Select the BIGINT `data_file_id` (not a literal `1`, which Postgres
            // types as INT4 and cannot decode into i64) — we only need existence.
            let target_live: Option<i64> = sqlx::query_scalar(
                "SELECT data_file_id FROM ducklake_data_file
                 WHERE data_file_id = $1 AND end_snapshot IS NULL",
            )
            .bind(data_file_id)
            .fetch_optional(&mut *tx)
            .await?;
            if target_live.is_none() {
                return Err(crate::DuckLakeError::Conflict(format!(
                    "delete targets data file {data_file_id}, which was retired by a \
                     concurrent write since snapshot {base_snapshot}; retry against the \
                     new generation"
                )));
            }

            // Compare-and-swap on the currently-live delete file for this data
            // file (`end_snapshot IS NULL`); a concurrent delete on the same data
            // file makes it differ from what the caller saw.
            let current_prev: Option<i64> = sqlx::query_scalar(
                "SELECT delete_file_id FROM ducklake_delete_file
                 WHERE data_file_id = $1 AND end_snapshot IS NULL",
            )
            .bind(data_file_id)
            .fetch_optional(&mut *tx)
            .await?;
            if current_prev != expected_prev_delete_file {
                return Err(crate::DuckLakeError::Conflict(format!(
                    "delete on data file {data_file_id} conflicts with a concurrent delete \
                     (expected live delete file {expected_prev_delete_file:?}, found \
                     {current_prev:?}); retry against the new generation"
                )));
            }

            // Allocate the snapshot (commit-ordered IDENTITY). A delete is
            // non-DDL, so carry the per-catalog `schema_version` forward.
            let snapshot_id: i64 = sqlx::query(
                "INSERT INTO ducklake_snapshot (snapshot_time, schema_version)
                 VALUES (NOW(), 0) RETURNING snapshot_id",
            )
            .fetch_one(&mut *tx)
            .await?
            .try_get(0)?;
            let prev_max: i64 = sqlx::query(
                "SELECT COALESCE(MAX(s.schema_version), 0) FROM ducklake_snapshot s
                 JOIN ducklake_catalog_snapshot_map m ON m.snapshot_id = s.snapshot_id
                 WHERE m.catalog_id = $1",
            )
            .bind(self.catalog_id)
            .fetch_one(&mut *tx)
            .await?
            .try_get(0)?;
            sqlx::query("UPDATE ducklake_snapshot SET schema_version = $1 WHERE snapshot_id = $2")
                .bind(prev_max.max(1))
                .bind(snapshot_id)
                .execute(&mut *tx)
                .await?;

            // Retire the prior delete file (cumulative: the new file carries all
            // still-deleted positions, so the old one is superseded).
            if let Some(prev) = expected_prev_delete_file {
                sqlx::query(
                    "UPDATE ducklake_delete_file SET end_snapshot = $1
                     WHERE delete_file_id = $2 AND end_snapshot IS NULL",
                )
                .bind(snapshot_id)
                .bind(prev)
                .execute(&mut *tx)
                .await?;
            }

            sqlx::query(
                "INSERT INTO ducklake_delete_file
                     (data_file_id, table_id, path, path_is_relative, file_size_bytes,
                      footer_size, delete_count, begin_snapshot)
                 VALUES ($1, $2, $3, $4, $5, $6, $7, $8)",
            )
            .bind(data_file_id)
            .bind(table_id)
            .bind(&delete.path)
            .bind(delete.path_is_relative)
            .bind(delete.file_size_bytes)
            .bind(delete.footer_size)
            .bind(delete.delete_count)
            .bind(snapshot_id)
            .execute(&mut *tx)
            .await?;

            let schema_id: i64 =
                sqlx::query_scalar("SELECT schema_id FROM ducklake_table WHERE table_id = $1")
                    .bind(table_id)
                    .fetch_one(&mut *tx)
                    .await?;

            // advance_catalog_head MUST be the last write before commit.
            advance_catalog_head(self.catalog_id, snapshot_id, &mut tx).await?;

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
            // One atomic commit for a combined append + positional deletes (an
            // update/upsert). finalize_snapshot writes the snapshot + schema/columns
            // and returns the committed snapshot id; the new data file AND every
            // delete file are stamped with that one id, and advance_catalog_head runs
            // LAST — so the whole mutation becomes visible together, never a
            // half-applied intermediate.
            let mut tx = self.pool.begin().await?;
            lock_catalog(self.catalog_id, self.lock_timeout_ms, &mut tx).await?;
            assert_table_not_in_other_catalog(self.catalog_id, table_id, &mut tx).await?;

            let (snapshot_id, schema_id, table_id) = finalize_snapshot(
                self.catalog_id,
                schema_name,
                table_name,
                table_id,
                columns,
                column_ids,
                mode,
                base_snapshot,
                &mut tx,
            )
            .await?;

            // Register the new data file (the inserted row versions), exactly as
            // register_data_file: seed stats, draw the row-id range, insert, and
            // accumulate. Deletes are accounted at read time (delete_count), so the
            // stats record_count stays gross — do not adjust it for the deletes.
            sqlx::query(
                "INSERT INTO ducklake_table_stats (table_id, record_count, next_row_id, file_size_bytes)
                 VALUES ($1, 0, 0, 0)
                 ON CONFLICT (table_id) DO NOTHING",
            )
            .bind(table_id)
            .execute(&mut *tx)
            .await?;

            let stats_row =
                sqlx::query("SELECT next_row_id FROM ducklake_table_stats WHERE table_id = $1")
                    .bind(table_id)
                    .fetch_one(&mut *tx)
                    .await?;
            let row_id_start: i64 = stats_row.try_get(0)?;

            sqlx::query(
                "INSERT INTO ducklake_data_file
                     (table_id, path, path_is_relative, file_size_bytes,
                      footer_size, record_count, row_id_start, begin_snapshot)
                 VALUES ($1, $2, $3, $4, $5, $6, $7, $8)",
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

            sqlx::query(
                "UPDATE ducklake_table_stats
                 SET next_row_id     = next_row_id + $1,
                     record_count    = record_count + $2,
                     file_size_bytes = file_size_bytes + $3
                 WHERE table_id = $4",
            )
            .bind(file.record_count)
            .bind(file.record_count)
            .bind(file.file_size_bytes)
            .bind(table_id)
            .execute(&mut *tx)
            .await?;

            // Apply each positional delete with the same fence + compare-and-swap as
            // set_delete_file, stamped with this snapshot. Each entry targets a
            // distinct data file, so there is no intra-transaction CAS contention.
            for entry in deletes {
                let target_live: Option<i64> = sqlx::query_scalar(
                    "SELECT data_file_id FROM ducklake_data_file
                     WHERE data_file_id = $1 AND end_snapshot IS NULL",
                )
                .bind(entry.data_file_id)
                .fetch_optional(&mut *tx)
                .await?;
                if target_live.is_none() {
                    return Err(crate::DuckLakeError::Conflict(format!(
                        "UPDATE/DELETE on data file {} could not commit: the file is no longer \
                         live as of the catalog's current head (retired since snapshot \
                         {base_snapshot}). This happens when another writer committed a \
                         Replace/compaction, OR when an earlier write in THIS session already \
                         advanced the catalog (the catalog pins its snapshot at creation and does \
                         not refresh). Re-open the catalog at the latest snapshot and retry.",
                        entry.data_file_id
                    )));
                }

                let current_prev: Option<i64> = sqlx::query_scalar(
                    "SELECT delete_file_id FROM ducklake_delete_file
                     WHERE data_file_id = $1 AND end_snapshot IS NULL",
                )
                .bind(entry.data_file_id)
                .fetch_optional(&mut *tx)
                .await?;
                if current_prev != entry.expected_prev_delete_file {
                    return Err(crate::DuckLakeError::Conflict(format!(
                        "UPDATE/DELETE on data file {} could not commit: its live delete file \
                         changed from {:?} to {current_prev:?} since snapshot {base_snapshot}. \
                         Another writer committed a delete on this file, OR an earlier \
                         UPDATE/DELETE in THIS session did (the catalog pins its snapshot at \
                         creation and does not refresh). Re-open the catalog at the latest \
                         snapshot and retry.",
                        entry.data_file_id, entry.expected_prev_delete_file
                    )));
                }

                if let Some(prev) = entry.expected_prev_delete_file {
                    sqlx::query(
                        "UPDATE ducklake_delete_file SET end_snapshot = $1
                         WHERE delete_file_id = $2 AND end_snapshot IS NULL",
                    )
                    .bind(snapshot_id)
                    .bind(prev)
                    .execute(&mut *tx)
                    .await?;
                }

                sqlx::query(
                    "INSERT INTO ducklake_delete_file
                         (data_file_id, table_id, path, path_is_relative, file_size_bytes,
                          footer_size, delete_count, begin_snapshot)
                     VALUES ($1, $2, $3, $4, $5, $6, $7, $8)",
                )
                .bind(entry.data_file_id)
                .bind(table_id)
                .bind(&entry.delete.path)
                .bind(entry.delete.path_is_relative)
                .bind(entry.delete.file_size_bytes)
                .bind(entry.delete.footer_size)
                .bind(entry.delete.delete_count)
                .bind(snapshot_id)
                .execute(&mut *tx)
                .await?;
            }

            // advance_catalog_head MUST be the last write before commit.
            advance_catalog_head(self.catalog_id, snapshot_id, &mut tx).await?;

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
        // A positional delete never retires the data files it targets, so it is
        // Append-semantics for validation (also enforces distinct data files).
        validate_delete_entries(WriteMode::Append, deletes)?;
        block_on(async {
            // Single atomic commit under the catalog lock for an N-file positional
            // DELETE with no append: fence + compare-and-swap + retire + insert per
            // entry, all stamped with one snapshot; advance_catalog_head LAST so
            // the whole multi-file delete becomes visible together.
            let mut tx = self.pool.begin().await?;
            lock_catalog(self.catalog_id, self.lock_timeout_ms, &mut tx).await?;
            assert_table_not_in_other_catalog(self.catalog_id, table_id, &mut tx).await?;

            // Allocate the snapshot (commit-ordered IDENTITY). A delete is
            // non-DDL, so carry the per-catalog schema_version forward.
            let snapshot_id: i64 = sqlx::query(
                "INSERT INTO ducklake_snapshot (snapshot_time, schema_version)
                 VALUES (NOW(), 0) RETURNING snapshot_id",
            )
            .fetch_one(&mut *tx)
            .await?
            .try_get(0)?;
            let prev_max: i64 = sqlx::query(
                "SELECT COALESCE(MAX(s.schema_version), 0) FROM ducklake_snapshot s
                 JOIN ducklake_catalog_snapshot_map m ON m.snapshot_id = s.snapshot_id
                 WHERE m.catalog_id = $1",
            )
            .bind(self.catalog_id)
            .fetch_one(&mut *tx)
            .await?
            .try_get(0)?;
            sqlx::query("UPDATE ducklake_snapshot SET schema_version = $1 WHERE snapshot_id = $2")
                .bind(prev_max.max(1))
                .bind(snapshot_id)
                .execute(&mut *tx)
                .await?;

            for entry in deletes {
                // Target-file fence: abort iff the data file is no longer live.
                // Select the BIGINT data_file_id (not literal 1, which Postgres
                // types as INT4 and cannot decode into i64) — existence only.
                let target_live: Option<i64> = sqlx::query_scalar(
                    "SELECT data_file_id FROM ducklake_data_file
                     WHERE data_file_id = $1 AND end_snapshot IS NULL",
                )
                .bind(entry.data_file_id)
                .fetch_optional(&mut *tx)
                .await?;
                if target_live.is_none() {
                    return Err(crate::DuckLakeError::Conflict(format!(
                        "DELETE on data file {} could not commit: the file is no longer live as \
                         of the catalog's current head (retired since snapshot {base_snapshot}). \
                         This happens when another writer committed a Replace/compaction, OR when \
                         an earlier write in THIS session already advanced the catalog (the \
                         catalog pins its snapshot at creation and does not refresh). Re-open the \
                         catalog at the latest snapshot and retry.",
                        entry.data_file_id
                    )));
                }

                // Compare-and-swap on the currently-live delete file.
                let current_prev: Option<i64> = sqlx::query_scalar(
                    "SELECT delete_file_id FROM ducklake_delete_file
                     WHERE data_file_id = $1 AND end_snapshot IS NULL",
                )
                .bind(entry.data_file_id)
                .fetch_optional(&mut *tx)
                .await?;
                if current_prev != entry.expected_prev_delete_file {
                    return Err(crate::DuckLakeError::Conflict(format!(
                        "DELETE on data file {} could not commit: its live delete file changed \
                         from {:?} to {current_prev:?} since snapshot {base_snapshot}. Another \
                         writer committed a delete on this file, OR an earlier DELETE in THIS \
                         session did (the catalog pins its snapshot at creation and does not \
                         refresh). Re-open the catalog at the latest snapshot and retry.",
                        entry.data_file_id, entry.expected_prev_delete_file
                    )));
                }

                if let Some(prev) = entry.expected_prev_delete_file {
                    sqlx::query(
                        "UPDATE ducklake_delete_file SET end_snapshot = $1
                         WHERE delete_file_id = $2 AND end_snapshot IS NULL",
                    )
                    .bind(snapshot_id)
                    .bind(prev)
                    .execute(&mut *tx)
                    .await?;
                }

                sqlx::query(
                    "INSERT INTO ducklake_delete_file
                         (data_file_id, table_id, path, path_is_relative, file_size_bytes,
                          footer_size, delete_count, begin_snapshot)
                     VALUES ($1, $2, $3, $4, $5, $6, $7, $8)",
                )
                .bind(entry.data_file_id)
                .bind(table_id)
                .bind(&entry.delete.path)
                .bind(entry.delete.path_is_relative)
                .bind(entry.delete.file_size_bytes)
                .bind(entry.delete.footer_size)
                .bind(entry.delete.delete_count)
                .bind(snapshot_id)
                .execute(&mut *tx)
                .await?;
            }

            let schema_id: i64 =
                sqlx::query_scalar("SELECT schema_id FROM ducklake_table WHERE table_id = $1")
                    .bind(table_id)
                    .fetch_one(&mut *tx)
                    .await?;

            // advance_catalog_head MUST be the last write before commit.
            advance_catalog_head(self.catalog_id, snapshot_id, &mut tx).await?;

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
        sources: &[crate::metadata_writer::CompactionSourceFile],
        outputs: &[crate::metadata_writer::CompactionOutputFile],
        retirement: crate::metadata_writer::SourceRetirement,
    ) -> Result<CommitIds> {
        use crate::metadata_writer::SourceRetirement;
        if sources.is_empty() {
            return Err(crate::DuckLakeError::InvalidConfig(
                "commit_compaction requires at least one source file".to_string(),
            ));
        }
        block_on(async {
            // One atomic commit under the catalog lock. Mirrors
            // commit_positional_deletes' shape (allocate snapshot, carry
            // schema_version forward, fence + CAS), plus: schedule + retire the
            // sources (and their delete files), register the outputs, recompute
            // stats, record the change ledger, and advance_catalog_head LAST.
            let mut tx = self.pool.begin().await?;
            lock_catalog(self.catalog_id, self.lock_timeout_ms, &mut tx).await?;
            assert_table_not_in_other_catalog(self.catalog_id, table_id, &mut tx).await?;

            let snapshot_id: i64 = sqlx::query(
                "INSERT INTO ducklake_snapshot (snapshot_time, schema_version)
                 VALUES (NOW(), 0) RETURNING snapshot_id",
            )
            .fetch_one(&mut *tx)
            .await?
            .try_get(0)?;
            let prev_max: i64 = sqlx::query(
                "SELECT COALESCE(MAX(s.schema_version), 0) FROM ducklake_snapshot s
                 JOIN ducklake_catalog_snapshot_map m ON m.snapshot_id = s.snapshot_id
                 WHERE m.catalog_id = $1",
            )
            .bind(self.catalog_id)
            .fetch_one(&mut *tx)
            .await?
            .try_get(0)?;
            sqlx::query("UPDATE ducklake_snapshot SET schema_version = $1 WHERE snapshot_id = $2")
                .bind(prev_max.max(1))
                .bind(snapshot_id)
                .execute(&mut *tx)
                .await?;

            // Fence + compare-and-swap per source (see commit_compaction on the
            // SQLite writer for the rationale): abort — never resurrect rows —
            // if a source was retired or its live delete file changed since read.
            for src in sources {
                let target_live: Option<i64> = sqlx::query_scalar(
                    "SELECT data_file_id FROM ducklake_data_file
                     WHERE data_file_id = $1 AND table_id = $2 AND end_snapshot IS NULL",
                )
                .bind(src.data_file_id)
                .bind(table_id)
                .fetch_optional(&mut *tx)
                .await?;
                if target_live.is_none() {
                    return Err(crate::DuckLakeError::Conflict(format!(
                        "compaction of table {table_id} could not commit: source data file {} is \
                         no longer live (retired by a concurrent Replace/compaction since snapshot \
                         {base_snapshot}). Re-open the catalog at the latest snapshot and re-plan.",
                        src.data_file_id
                    )));
                }
                let current_delete: Option<i64> = sqlx::query_scalar(
                    "SELECT delete_file_id FROM ducklake_delete_file
                     WHERE data_file_id = $1 AND end_snapshot IS NULL",
                )
                .bind(src.data_file_id)
                .fetch_optional(&mut *tx)
                .await?;
                if current_delete != src.delete_file_id {
                    return Err(crate::DuckLakeError::Conflict(format!(
                        "compaction of table {table_id} could not commit: the live delete file of \
                         source data file {} changed from {:?} to {current_delete:?} since snapshot \
                         {base_snapshot} (a concurrent DELETE/UPDATE). Re-open the catalog at the \
                         latest snapshot and re-plan.",
                        src.data_file_id, src.delete_file_id
                    )));
                }
            }

            let source_data_ids: Vec<i64> = sources.iter().map(|s| s.data_file_id).collect();

            match retirement {
                SourceRetirement::Remove => {
                    // Merge: the partial output serves every snapshot the sources
                    // did, so schedule their physical files (resolving paths as the
                    // multicatalog expire path does) and REMOVE their catalog rows.
                    let dead_data = sqlx::query(&format!(
                        "SELECT df.data_file_id, {COMPACTION_RESOLVED_PATH} AS resolved_path,
                                {COMPACTION_REL_FLAG} AS rel
                         FROM ducklake_data_file df
                         JOIN ducklake_table t ON t.table_id = df.table_id
                         JOIN ducklake_schema s ON s.schema_id = t.schema_id
                         WHERE df.data_file_id = ANY($1)"
                    ))
                    .bind(&source_data_ids)
                    .fetch_all(&mut *tx)
                    .await?;
                    schedule_compaction_files(&mut tx, self.catalog_id, dead_data).await?;

                    let dead_del = sqlx::query(&format!(
                        "SELECT df.delete_file_id, {COMPACTION_RESOLVED_PATH} AS resolved_path,
                                {COMPACTION_REL_FLAG} AS rel
                         FROM ducklake_delete_file df
                         JOIN ducklake_table t ON t.table_id = df.table_id
                         JOIN ducklake_schema s ON s.schema_id = t.schema_id
                         WHERE df.data_file_id = ANY($1)"
                    ))
                    .bind(&source_data_ids)
                    .fetch_all(&mut *tx)
                    .await?;
                    schedule_compaction_files(&mut tx, self.catalog_id, dead_del).await?;

                    sqlx::query("DELETE FROM ducklake_delete_file WHERE data_file_id = ANY($1)")
                        .bind(&source_data_ids)
                        .execute(&mut *tx)
                        .await?;
                    sqlx::query("DELETE FROM ducklake_data_file WHERE data_file_id = ANY($1)")
                        .bind(&source_data_ids)
                        .execute(&mut *tx)
                        .await?;
                },
                SourceRetirement::Retire => {
                    // Rewrite: the sources still serve time travel to pre-compaction
                    // snapshots, so retire them (end_snapshot) but do NOT schedule
                    // them; expire_snapshots reclaims them once their snapshots are
                    // gone.
                    sqlx::query(
                        "UPDATE ducklake_data_file SET end_snapshot = $1
                         WHERE data_file_id = ANY($2) AND end_snapshot IS NULL",
                    )
                    .bind(snapshot_id)
                    .bind(&source_data_ids)
                    .execute(&mut *tx)
                    .await?;
                    sqlx::query(
                        "UPDATE ducklake_delete_file SET end_snapshot = $1
                         WHERE data_file_id = ANY($2) AND end_snapshot IS NULL",
                    )
                    .bind(snapshot_id)
                    .bind(&source_data_ids)
                    .execute(&mut *tx)
                    .await?;
                },
            }

            // Register each rewritten output. begin_snapshot = the file's min
            // origin snapshot for a merged partial file (so historical reads see
            // it), else this compaction snapshot; row_id_start NULL (rowids come
            // from the embedded column); partial_max marks a merged partial file.
            for out in outputs {
                let begin = out.begin_snapshot.unwrap_or(snapshot_id);
                sqlx::query(
                    "INSERT INTO ducklake_data_file
                         (table_id, path, path_is_relative, file_size_bytes,
                          footer_size, record_count, row_id_start, begin_snapshot, partial_max)
                     VALUES ($1, $2, $3, $4, $5, $6, NULL, $7, $8)",
                )
                .bind(table_id)
                .bind(&out.file.path)
                .bind(out.file.path_is_relative)
                .bind(out.file.file_size_bytes)
                .bind(out.file.footer_size)
                .bind(out.file.record_count)
                .bind(begin)
                .bind(out.partial_max)
                .execute(&mut *tx)
                .await?;
            }

            // Recompute the visible stat totals from the surviving files (see the
            // SQLite writer for why this is correct for both merge and rewrite).
            // next_row_id is deliberately not advanced (no new logical rows).
            sqlx::query(
                "INSERT INTO ducklake_table_stats (table_id, record_count, next_row_id, file_size_bytes)
                 VALUES ($1, 0, 0, 0) ON CONFLICT (table_id) DO NOTHING",
            )
            .bind(table_id)
            .execute(&mut *tx)
            .await?;
            sqlx::query(
                "UPDATE ducklake_table_stats SET
                     record_count = (SELECT COALESCE(SUM(record_count), 0)
                                     FROM ducklake_data_file
                                     WHERE table_id = $1 AND end_snapshot IS NULL),
                     file_size_bytes = (SELECT COALESCE(SUM(file_size_bytes), 0)
                                        FROM ducklake_data_file
                                        WHERE table_id = $1 AND end_snapshot IS NULL)
                 WHERE table_id = $1",
            )
            .bind(table_id)
            .execute(&mut *tx)
            .await?;

            sqlx::query(
                "INSERT INTO ducklake_snapshot_changes (snapshot_id, changes_made, commit_message)
                 VALUES ($1, $2, $3)",
            )
            .bind(snapshot_id)
            .bind(format!("compacted_table:{table_id}"))
            .bind("datafusion compaction")
            .execute(&mut *tx)
            .await?;

            let schema_id: i64 =
                sqlx::query_scalar("SELECT schema_id FROM ducklake_table WHERE table_id = $1")
                    .bind(table_id)
                    .fetch_one(&mut *tx)
                    .await?;

            // advance_catalog_head MUST be the last write before commit.
            advance_catalog_head(self.catalog_id, snapshot_id, &mut tx).await?;

            tx.commit().await?;
            Ok(CommitIds {
                snapshot_id,
                schema_id,
                table_id,
            })
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
            // Metadata-only truncate in one snapshot under the catalog lock: end
            // every live data file and its live delete file, zero the visible stat
            // totals, and advance the head LAST. next_row_id is preserved.
            let mut tx = self.pool.begin().await?;
            lock_catalog(self.catalog_id, self.lock_timeout_ms, &mut tx).await?;
            assert_table_in_catalog(self.catalog_id, table_id, &mut tx).await?;

            // No-op guard: nothing to truncate if the table has no live data file.
            // Return Ok(0) BEFORE allocating a snapshot, so a repeated
            // `DELETE FROM t` under a pinned snapshot does not create a
            // content-free snapshot. lock_catalog above already serializes, so
            // this read is stable.
            let has_live_data: Option<i64> = sqlx::query_scalar(
                "SELECT data_file_id FROM ducklake_data_file
                 WHERE table_id = $1 AND end_snapshot IS NULL LIMIT 1",
            )
            .bind(table_id)
            .fetch_optional(&mut *tx)
            .await?;
            if has_live_data.is_none() {
                return Ok(0);
            }

            let snapshot_id: i64 = sqlx::query(
                "INSERT INTO ducklake_snapshot (snapshot_time, schema_version)
                 VALUES (NOW(), 0) RETURNING snapshot_id",
            )
            .fetch_one(&mut *tx)
            .await?
            .try_get(0)?;
            let prev_max: i64 = sqlx::query(
                "SELECT COALESCE(MAX(s.schema_version), 0) FROM ducklake_snapshot s
                 JOIN ducklake_catalog_snapshot_map m ON m.snapshot_id = s.snapshot_id
                 WHERE m.catalog_id = $1",
            )
            .bind(self.catalog_id)
            .fetch_one(&mut *tx)
            .await?
            .try_get(0)?;
            sqlx::query("UPDATE ducklake_snapshot SET schema_version = $1 WHERE snapshot_id = $2")
                .bind(prev_max.max(1))
                .bind(snapshot_id)
                .execute(&mut *tx)
                .await?;

            // Rows removed = gross record_count minus still-live delete counts,
            // computed BEFORE ending anything (so it matches what we retire).
            // SUM(bigint) is NUMERIC in Postgres; cast back to BIGINT for i64.
            let gross: Option<i64> = sqlx::query_scalar(
                "SELECT COALESCE(record_count, 0) FROM ducklake_table_stats WHERE table_id = $1",
            )
            .bind(table_id)
            .fetch_optional(&mut *tx)
            .await?;
            let deleted: i64 = sqlx::query_scalar(
                "SELECT COALESCE(SUM(delete_count), 0)::BIGINT FROM ducklake_delete_file
                 WHERE table_id = $1 AND end_snapshot IS NULL",
            )
            .bind(table_id)
            .fetch_one(&mut *tx)
            .await?;
            let live_rows = (gross.unwrap_or(0) - deleted).max(0) as u64;

            sqlx::query(
                "UPDATE ducklake_data_file SET end_snapshot = $1
                 WHERE table_id = $2 AND end_snapshot IS NULL",
            )
            .bind(snapshot_id)
            .bind(table_id)
            .execute(&mut *tx)
            .await?;
            sqlx::query(
                "UPDATE ducklake_delete_file SET end_snapshot = $1
                 WHERE table_id = $2 AND end_snapshot IS NULL",
            )
            .bind(snapshot_id)
            .bind(table_id)
            .execute(&mut *tx)
            .await?;
            sqlx::query(
                "UPDATE ducklake_table_stats SET record_count = 0, file_size_bytes = 0
                 WHERE table_id = $1",
            )
            .bind(table_id)
            .execute(&mut *tx)
            .await?;

            // advance_catalog_head MUST be the last write before commit.
            advance_catalog_head(self.catalog_id, snapshot_id, &mut tx).await?;

            tx.commit().await?;
            Ok(live_rows)
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
        block_on(async {
            // Fileless commit point (CREATE TABLE, zero-row Replace). Same atomic
            // model as register_data_file minus the data-file insert: write all
            // metadata via finalize_snapshot, then advance the head LAST.
            let mut tx = self.pool.begin().await?;
            lock_catalog(self.catalog_id, self.lock_timeout_ms, &mut tx).await?;
            assert_table_not_in_other_catalog(self.catalog_id, table_id, &mut tx).await?;

            let (snapshot_id, schema_id, table_id) = finalize_snapshot(
                self.catalog_id,
                schema_name,
                table_name,
                table_id,
                columns,
                column_ids,
                mode,
                base_snapshot,
                &mut tx,
            )
            .await?;

            advance_catalog_head(self.catalog_id, snapshot_id, &mut tx).await?;

            tx.commit().await?;
            Ok(CommitIds {
                snapshot_id,
                schema_id,
                table_id,
            })
        })
    }

    fn end_table_files(&self, table_id: i64, snapshot_id: i64) -> Result<u64> {
        // Used by WriteMode::Replace. End-snapshotting every visible file
        // drops the table's currently-visible row count and byte total to
        // zero (the new files written next will rebuild them). `next_row_id`
        // is deliberately NOT reset: rowids must stay monotonic across the
        // table's lifetime so historical snapshots still resolve uniquely.
        block_on(async {
            let mut tx = self.pool.begin().await?;
            lock_catalog(self.catalog_id, self.lock_timeout_ms, &mut tx).await?;
            assert_table_in_catalog(self.catalog_id, table_id, &mut tx).await?;

            let result = sqlx::query(
                "UPDATE ducklake_data_file SET end_snapshot = $1
                 WHERE table_id = $2 AND end_snapshot IS NULL",
            )
            .bind(snapshot_id)
            .bind(table_id)
            .execute(&mut *tx)
            .await?;
            let n = result.rows_affected();

            sqlx::query(
                "UPDATE ducklake_table_stats
                 SET record_count = 0, file_size_bytes = 0
                 WHERE table_id = $1",
            )
            .bind(table_id)
            .execute(&mut *tx)
            .await?;

            tx.commit().await?;
            Ok(n)
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
        // data_path is global per Phase 1. Reject silent overwrites — if it's
        // already set to a different value, a concurrent tenant likely set it.
        block_on(async {
            let mut tx = self.pool.begin().await?;

            let existing: Option<String> = sqlx::query(
                "SELECT value FROM ducklake_metadata
                 WHERE key = 'data_path' AND scope IS NULL FOR UPDATE",
            )
            .fetch_optional(&mut *tx)
            .await?
            .map(|r| r.try_get(0))
            .transpose()?;

            match existing {
                Some(cur) if cur == path => {
                    tx.commit().await?;
                    return Ok(());
                },
                Some(cur) => {
                    return Err(crate::error::DuckLakeError::InvalidConfig(format!(
                        "data_path already set to {:?}, refusing to overwrite with {:?}",
                        cur, path
                    )));
                },
                None => {},
            }

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
            execute_ddl_statements(&self.pool, SQL_CREATE_STANDARD_TABLES).await?;
            execute_ddl_statements(&self.pool, SQL_CREATE_MULTICATALOG_TABLES).await?;
            // Upgrade a pre-existing store's ducklake_column to the composite PK
            // (legacy single-row column_id PK → versioned-capable). Idempotent.
            migrate_ducklake_column_to_composite_pk(&self.pool).await?;
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
        let catalog_id = self.catalog_id;
        let lock_timeout_ms = self.lock_timeout_ms;
        block_on(async {
            // RESERVE ONLY: this transaction inserts NOTHING into ducklake_snapshot
            // / ducklake_schema / ducklake_table / ducklake_column /
            // ducklake_schema_versions / either map table. All of that is written
            // atomically at the commit point (register_data_file /
            // publish_snapshot → finalize_snapshot) and assigned a commit-ordered
            // snapshot id there, so no metadata row is ever readable before its
            // write has published. Here we only reserve ids from the IDENTITY
            // sequences (gaps are fine) and capture the conflict base.
            //
            // The catalog FOR UPDATE lock is held so the existing-column read and
            // the id-reuse map are consistent and the returned field-ids are
            // stable; it is released at tx.commit (which commits only the
            // non-transactional sequence advances).
            let mut tx = self.pool.begin().await?;
            lock_catalog(catalog_id, lock_timeout_ms, &mut tx).await?;

            // Look up (do NOT create) the live schema; reserve a fresh id if
            // absent. setup.schema_id is informational (no caller bakes it into a
            // file — the parquet path encodes the catalog id, not the schema id),
            // so for a brand-new schema this reserved id is NOT the committed id:
            // finalize_snapshot re-derives/reserves the schema id at the commit.
            // The reservation here keeps setup.schema_id distinct & non-zero across
            // concurrent new schemas (sequence gaps from the unused reservation are
            // expected and harmless).
            let schema_id: i64 = {
                let existing = sqlx::query(
                    "SELECT s.schema_id FROM ducklake_schema s
                     JOIN ducklake_catalog_schema_map m ON m.schema_id = s.schema_id
                     WHERE m.catalog_id = $1 AND s.schema_name = $2 AND s.end_snapshot IS NULL",
                )
                .bind(catalog_id)
                .bind(schema_name)
                .fetch_optional(&mut *tx)
                .await?;
                match existing {
                    Some(row) => row.try_get(0)?,
                    None => reserve_ids("ducklake_schema", "schema_id", 1, &mut tx).await?[0],
                }
            };

            // Look up (do NOT create) the live table; reserve an id if absent. The
            // reserved id IS used (threaded to finalize as table_id_hint).
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
                    reserve_ids("ducklake_table", "table_id", 1, &mut tx).await?[0]
                }
            };

            // Read existing columns (name, type, nullable, id) to drive (a) the
            // Append schema-evolution check and (b) id reuse: an unchanged column
            // keeps its column_id (== parquet field_id), so an already-written
            // file's field-ids stay valid.
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

            // Data-write policy (§5, same rules as the SQLite writer): a data write
            // — Replace OR Append — must NOT change a column's type. A type change is
            // schema evolution and must go through `promote_column_type`, never a data
            // write; silently keeping the old catalog type (the "C" bug) corrupts
            // reads. Canonical comparison (`int64` ≡ `bigint`) so an alias-only
            // restatement is a no-op. Append additionally requires a genuinely new
            // column to be nullable (a Replace overwrites every row, so a new
            // non-nullable column is fine there).
            if !existing_columns.is_empty() {
                use std::collections::HashMap;
                let existing_map: HashMap<&str, (&str, bool)> = existing_columns
                    .iter()
                    .map(|(name, col_type, nullable)| {
                        (name.as_str(), (col_type.as_str(), *nullable))
                    })
                    .collect();

                for new_col in columns.iter() {
                    if let Some((existing_type, _)) = existing_map.get(new_col.name.as_str()) {
                        // Same-name column: a (canonical) type change is rejected in BOTH
                        // modes. Not `types_compatible` — that accepts widenings, the
                        // silent acceptance we are closing.
                        if !crate::types::types_equal_canonical(
                            existing_type,
                            &new_col.ducklake_type,
                        ) {
                            return Err(crate::error::DuckLakeError::UnsupportedTypeChange {
                                operation: TypeChangeOperation::DataWrite {
                                    mode: match mode {
                                        WriteMode::Replace => TypeChangeWriteMode::Replace,
                                        WriteMode::Append => TypeChangeWriteMode::Append,
                                    },
                                },
                                column: new_col.name.clone(),
                                from: (*existing_type).to_string(),
                                to: new_col.ducklake_type.clone(),
                            });
                        }
                    } else if mode == WriteMode::Append && !new_col.is_nullable {
                        return Err(crate::error::DuckLakeError::InvalidConfig(format!(
                            "Schema evolution error: new column '{}' must be nullable. Adding non-nullable columns is not allowed.",
                            new_col.name
                        )));
                    }
                }
            }

            // Reserve N column ids, then compute the final per-column ids. These
            // are baked into the staged parquet's field_id metadata, so they must
            // equal what finalize_snapshot inserts at commit.
            //
            // Mode-independent (matching the SQLite writer): REUSE the existing id
            // for a column whose NAME already exists, consume a freshly-reserved id
            // only for a genuinely-new column. Stable ids are required for BOTH
            // modes — a concurrent in-flight Append bakes the kept columns' ids into
            // its parquet, so a Replace must NOT re-mint them (re-minting would make
            // that Append's rows read back as all-NULL). The Replace conflict check
            // does not rely on a column re-mint: a data Replace leaves a new data
            // file (begin > base) and a schema-changing Replace ends/inserts the
            // changed columns (begin/end > base); a fileless same-schema Replace
            // leaves no trace and resolves last-writer-wins, exactly like SQLite.
            let fresh_ids = reserve_ids(
                "ducklake_column",
                "column_id",
                columns.len() as i64,
                &mut tx,
            )
            .await?;
            let column_ids: Vec<i64> = columns
                .iter()
                .zip(fresh_ids.iter())
                .map(|(col, &fresh)| existing_ids.get(&col.name).copied().unwrap_or(fresh))
                .collect();

            // base = catalog head observed at begin. The Replace commit aborts if
            // any file/column of the table moved past it (a concurrent writer
            // committed a newer generation). Snapshot ids are commit-ordered, so
            // this scalar head is an exact conflict base.
            let base_snapshot_id: i64 = sqlx::query(
                "SELECT COALESCE(MAX(snapshot_id), 0) FROM ducklake_catalog_snapshot_map
                 WHERE catalog_id = $1",
            )
            .bind(catalog_id)
            .fetch_one(&mut *tx)
            .await?
            .try_get(0)?;

            // Commit the (sequence-only) reservation transaction.
            tx.commit().await?;

            Ok(WriteSetupResult {
                // snapshot_id is vestigial here (like SQLite's): the real id is
                // assigned at the commit by finalize_snapshot.
                snapshot_id: 0,
                base_snapshot_id,
                schema_id,
                table_id,
                column_ids,
            })
        })
    }

    fn catalog_id(&self) -> Option<i64> {
        Some(self.catalog_id)
    }

    /// Multicatalog Postgres implements the atomic append-with-deletes commit,
    /// so it supports row-level `UPDATE`.
    fn supports_update(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use crate::metadata_writer::{ColumnDef, columns_differ};

    #[test]
    fn test_columns_differ_identical() {
        let existing =
            vec![("id".into(), "int64".into(), false), ("name".into(), "varchar".into(), true)];
        let proposed = vec![
            ColumnDef::new("id", "int64", false).unwrap(),
            ColumnDef::new("name", "varchar", true).unwrap(),
        ];
        assert!(!columns_differ(&existing, &proposed));
    }

    #[test]
    fn test_columns_differ_added_column() {
        let existing = vec![("id".into(), "int64".into(), false)];
        let proposed = vec![
            ColumnDef::new("id", "int64", false).unwrap(),
            ColumnDef::new("name", "varchar", true).unwrap(),
        ];
        assert!(columns_differ(&existing, &proposed));
    }

    #[test]
    fn test_columns_differ_renamed_column() {
        let existing = vec![("id".into(), "int64".into(), false)];
        let proposed = vec![ColumnDef::new("user_id", "int64", false).unwrap()];
        assert!(columns_differ(&existing, &proposed));
    }

    #[test]
    fn test_columns_differ_type_change() {
        let existing = vec![("id".into(), "int32".into(), false)];
        let proposed = vec![ColumnDef::new("id", "varchar", false).unwrap()];
        assert!(columns_differ(&existing, &proposed));
    }

    #[test]
    fn test_columns_differ_forward_widening_is_a_change() {
        // existing int32 -> proposed int64 is a forward widening. Since #149 this
        // is NOT "the same column": on a data write it's rejected at begin-time
        // (widenings must go through promote_column_type), and if it reaches
        // columns_differ it must classify as DDL. Only the benign promote-race
        // direction below is treated as same-type.
        let existing = vec![("id".into(), "int32".into(), false)];
        let proposed = vec![ColumnDef::new("id", "int64", false).unwrap()];
        assert!(columns_differ(&existing, &proposed));
    }

    #[test]
    fn test_columns_differ_benign_promote_race_is_not_ddl() {
        // The Append-vs-promote race: the committed column was already widened to
        // int64 by a concurrent promote, while this write staged the narrower
        // int32 (which passed the begin-time reject against the type AT BEGIN).
        // The staged int32 losslessly widens to the committed int64 and is served
        // via cast-on-read, so it is NOT a schema change and must not bump
        // schema_version.
        let existing = vec![("id".into(), "int64".into(), false)];
        let proposed = vec![ColumnDef::new("id", "int32", false).unwrap()];
        assert!(!columns_differ(&existing, &proposed));
    }

    #[test]
    fn test_columns_differ_nullability_change() {
        let existing = vec![("id".into(), "int64".into(), false)];
        let proposed = vec![ColumnDef::new("id", "int64", true).unwrap()];
        assert!(columns_differ(&existing, &proposed));
    }

    #[test]
    fn test_columns_differ_alias_canonical() {
        // bigint and int64 normalize to the same canonical type.
        let existing = vec![("id".into(), "bigint".into(), false)];
        let proposed = vec![ColumnDef::new("id", "int64", false).unwrap()];
        assert!(!columns_differ(&existing, &proposed));
    }
}
