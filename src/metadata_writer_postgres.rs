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
use crate::metadata_provider::block_on;
use crate::metadata_writer::{
    ColumnDef, DataFileInfo, MetadataWriter, WriteMode, WriteSetupResult, validate_name,
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
    r#"CREATE TABLE IF NOT EXISTS ducklake_column (
        column_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        table_id BIGINT NOT NULL,
        column_name VARCHAR NOT NULL,
        column_type VARCHAR NOT NULL,
        column_order BIGINT NOT NULL,
        nulls_allowed BOOLEAN DEFAULT TRUE,
        parent_column BIGINT,
        begin_snapshot BIGINT NOT NULL,
        end_snapshot BIGINT
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
        end_snapshot BIGINT
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
];

/// Run a slice of DDL statements against the pool. Each statement executes independently.
pub(crate) async fn execute_ddl_statements(pool: &PgPool, statements: &[&str]) -> Result<()> {
    for stmt in statements {
        sqlx::query(stmt).execute(pool).await?;
    }
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
        snapshot_id: i64,
        file: &DataFileInfo,
    ) -> Result<i64> {
        block_on(async {
            // Allocate row_id_start from the table's monotonic counter inside
            // the same transaction so concurrent writers can't hand out
            // overlapping ranges. The DuckLake row-lineage read path
            // (RowIdExec) hard-errors on files where row_id_start is NULL and
            // no embedded `_ducklake_internal_row_id` column is present, so
            // every file we write must carry a value. Falls back to inserting
            // a fresh stats row (next_row_id = 0) for tables created before
            // this writer started maintaining `ducklake_table_stats`.
            let mut tx = self.pool.begin().await?;
            lock_catalog(self.catalog_id, self.lock_timeout_ms, &mut tx).await?;
            assert_table_in_catalog(self.catalog_id, table_id, &mut tx).await?;

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

            let row = sqlx::query(
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
            let id: i64 = row.try_get(0)?;

            // Advance the counter and accumulate stats. `next_row_id`
            // monotonically increases over the table's lifetime — rowids
            // are never reused, even after end-snapshot.
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

            tx.commit().await?;
            Ok(id)
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
            let mut tx = self.pool.begin().await?;
            lock_catalog(catalog_id, lock_timeout_ms, &mut tx).await?;

            // schema_version is patched in below once we know it.
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
            .bind(catalog_id)
            .bind(snapshot_id)
            .execute(&mut *tx)
            .await?;

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

                if let Some(row) = existing {
                    row.try_get(0)?
                } else {
                    let row = sqlx::query(
                        "INSERT INTO ducklake_schema (schema_name, path, path_is_relative, begin_snapshot)
                         VALUES ($1, $2, TRUE, $3) RETURNING schema_id",
                    )
                    .bind(schema_name)
                    .bind(schema_name)
                    .bind(snapshot_id)
                    .fetch_one(&mut *tx)
                    .await?;
                    let id: i64 = row.try_get(0)?;

                    sqlx::query(
                        "INSERT INTO ducklake_catalog_schema_map (catalog_id, schema_id)
                         VALUES ($1, $2)",
                    )
                    .bind(catalog_id)
                    .bind(id)
                    .execute(&mut *tx)
                    .await?;

                    id
                }
            };

            let (table_id, table_was_created): (i64, bool) = {
                let existing = sqlx::query(
                    "SELECT table_id FROM ducklake_table
                     WHERE schema_id = $1 AND table_name = $2 AND end_snapshot IS NULL",
                )
                .bind(schema_id)
                .bind(table_name)
                .fetch_optional(&mut *tx)
                .await?;

                if let Some(row) = existing {
                    (row.try_get(0)?, false)
                } else {
                    let row = sqlx::query(
                        "INSERT INTO ducklake_table (schema_id, table_name, path, path_is_relative, begin_snapshot)
                         VALUES ($1, $2, $3, TRUE, $4) RETURNING table_id",
                    )
                    .bind(schema_id)
                    .bind(table_name)
                    .bind(table_name)
                    .bind(snapshot_id)
                    .fetch_one(&mut *tx)
                    .await?;
                    (row.try_get(0)?, true)
                }
            };

            // Fetch existing columns to drive (a) schema-evolution checks for Append,
            // (b) DDL/DML classification.
            let existing_column_rows = sqlx::query(
                "SELECT column_name, column_type, nulls_allowed
                 FROM ducklake_column
                 WHERE table_id = $1 AND end_snapshot IS NULL
                 ORDER BY column_order",
            )
            .bind(table_id)
            .fetch_all(&mut *tx)
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

            // Append-mode schema evolution validation (same rules as SQLite writer).
            if mode == WriteMode::Append && !existing_columns.is_empty() {
                use std::collections::HashMap;
                let existing_map: HashMap<&str, (&str, bool)> = existing_columns
                    .iter()
                    .map(|(name, col_type, nullable)| {
                        (name.as_str(), (col_type.as_str(), *nullable))
                    })
                    .collect();

                for new_col in columns.iter() {
                    if let Some((existing_type, _)) = existing_map.get(new_col.name.as_str()) {
                        if !crate::types::types_compatible(existing_type, &new_col.ducklake_type) {
                            return Err(crate::error::DuckLakeError::InvalidConfig(format!(
                                "Schema evolution error: column '{}' has type '{}' in existing table but '{}' in new schema. Type changes are not allowed.",
                                new_col.name, existing_type, new_col.ducklake_type
                            )));
                        }
                    } else if !new_col.is_nullable {
                        return Err(crate::error::DuckLakeError::InvalidConfig(format!(
                            "Schema evolution error: new column '{}' must be nullable. Adding non-nullable columns is not allowed.",
                            new_col.name
                        )));
                    }
                }
            }

            // Classify DDL vs DML. DDL = first write (table just created) or the
            // column set changed from what exists. DML = same schema, just data.
            let is_ddl = table_was_created || columns_differ(&existing_columns, columns);

            // Allocate schema_version under the catalog lock we already hold.
            // Per spec: per-catalog dense; DDL bumps, DML carries forward.
            let max_row = sqlx::query(
                "SELECT COALESCE(MAX(s.schema_version), 0) FROM ducklake_snapshot s
                 JOIN ducklake_catalog_snapshot_map m ON m.snapshot_id = s.snapshot_id
                 WHERE m.catalog_id = $1 AND s.snapshot_id < $2",
            )
            .bind(catalog_id)
            .bind(snapshot_id)
            .fetch_one(&mut *tx)
            .await?;
            let prev_max: i64 = max_row.try_get(0)?;
            let new_schema_version = if is_ddl {
                prev_max + 1
            } else if prev_max == 0 {
                // No prior snapshot for this catalog yet — even a DML-classified
                // path needs a valid v1.
                1
            } else {
                prev_max
            };

            sqlx::query("UPDATE ducklake_snapshot SET schema_version = $1 WHERE snapshot_id = $2")
                .bind(new_schema_version)
                .bind(snapshot_id)
                .execute(&mut *tx)
                .await?;

            // End existing columns and insert the new set.
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

            // Record a row in ducklake_schema_versions for every DDL — the
            // UNIQUE(table_id, begin_snapshot) constraint ensures no duplicates.
            if is_ddl {
                sqlx::query(
                    "INSERT INTO ducklake_schema_versions (begin_snapshot, schema_version, table_id)
                     VALUES ($1, $2, $3)",
                )
                .bind(snapshot_id)
                .bind(new_schema_version)
                .bind(table_id)
                .execute(&mut *tx)
                .await?;
            }

            if mode == WriteMode::Replace {
                sqlx::query(
                    "UPDATE ducklake_data_file SET end_snapshot = $1
                     WHERE table_id = $2 AND end_snapshot IS NULL",
                )
                .bind(snapshot_id)
                .bind(table_id)
                .execute(&mut *tx)
                .await?;

                // Mirror end_table_files: drop visible row/byte totals to
                // zero while keeping next_row_id monotonic. Seed the row
                // first in case this is a Replace on a brand-new table.
                sqlx::query(
                    "INSERT INTO ducklake_table_stats
                         (table_id, record_count, next_row_id, file_size_bytes)
                     VALUES ($1, 0, 0, 0)
                     ON CONFLICT (table_id) DO NOTHING",
                )
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
            }

            tx.commit().await?;

            Ok(WriteSetupResult {
                snapshot_id,
                schema_id,
                table_id,
                column_ids,
            })
        })
    }
}

/// Return true when the existing column set differs from the proposed one in a
/// way that requires a DDL bump: a different number of columns, a renamed
/// column, a changed type, or a changed nullability.
///
/// Columns are compared positionally so a pure reorder counts as DDL — matches
/// the conservative interpretation of the spec.
fn columns_differ(existing: &[(String, String, bool)], proposed: &[ColumnDef]) -> bool {
    if existing.len() != proposed.len() {
        return true;
    }
    for ((ex_name, ex_type, ex_nullable), new_col) in existing.iter().zip(proposed.iter()) {
        if ex_name != &new_col.name {
            return true;
        }
        if !crate::types::types_compatible(ex_type, &new_col.ducklake_type) {
            return true;
        }
        // Same-name same-compatible-type same-nullability ⇒ no DDL.
        if *ex_nullable != new_col.is_nullable {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata_writer::ColumnDef;

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
    fn test_columns_differ_type_widening_is_compatible() {
        // int32 -> int64 is a safe promotion per types_compatible(); we treat
        // that as the same column, no DDL bump needed.
        let existing = vec![("id".into(), "int32".into(), false)];
        let proposed = vec![ColumnDef::new("id", "int64", false).unwrap()];
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
