//! SQLite implementation of [`MetadataWriter`].
//!
//! Requires multi-threaded Tokio runtime (`#[tokio::test(flavor = "multi_thread")]`).

use crate::Result;
use crate::maintenance::{
    CleanupCriteria, ExpireCriteria, ExpiredSnapshot, ScheduledFile, format_sql_timestamp,
};
use crate::metadata_provider::block_on;
use crate::metadata_writer::{
    ColumnDef, DataFileInfo, MetadataWriter, WriteMode, WriteSetupResult, validate_name,
};
use sqlx::Row;
use sqlx::sqlite::{SqlitePool, SqlitePoolOptions};

const DEFAULT_MAX_CONNECTIONS: u32 = 5;

/// Render a slice of ids as a SQL `IN (...)` body. Safe to interpolate because
/// the values are `i64` (no injection surface) — same approach the upstream
/// DuckLake C++ takes, and the only option since SQLite lacks array binds.
fn id_list(ids: &[i64]) -> String {
    ids.iter()
        .map(|id| id.to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

const SQL_CREATE_SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS ducklake_metadata (
    key VARCHAR NOT NULL,
    value VARCHAR NOT NULL,
    scope VARCHAR
);

CREATE TABLE IF NOT EXISTS ducklake_snapshot (
    snapshot_id INTEGER PRIMARY KEY,
    snapshot_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ducklake_schema (
    schema_id INTEGER PRIMARY KEY,
    schema_name VARCHAR NOT NULL,
    path VARCHAR NOT NULL DEFAULT '',
    path_is_relative BOOLEAN NOT NULL DEFAULT 1,
    begin_snapshot INTEGER NOT NULL,
    end_snapshot INTEGER
);

CREATE TABLE IF NOT EXISTS ducklake_table (
    table_id INTEGER PRIMARY KEY,
    schema_id INTEGER NOT NULL,
    table_name VARCHAR NOT NULL,
    path VARCHAR NOT NULL DEFAULT '',
    path_is_relative BOOLEAN NOT NULL DEFAULT 1,
    begin_snapshot INTEGER NOT NULL,
    end_snapshot INTEGER
);

CREATE TABLE IF NOT EXISTS ducklake_column (
    column_id INTEGER PRIMARY KEY,
    table_id INTEGER NOT NULL,
    column_name VARCHAR NOT NULL,
    column_type VARCHAR NOT NULL,
    column_order INTEGER NOT NULL,
    nulls_allowed BOOLEAN DEFAULT 1,
    -- Mirror the upstream DuckLake spec (ducklake_metadata_manager.cpp):
    -- `parent_column` is projected by our reader's SQL_GET_TABLE_COLUMNS;
    -- the four `*default*` columns are projected by DuckDB when it reads
    -- catalogs we produce. We leave them NULL — no nested-type or
    -- column-default writes yet.
    initial_default VARCHAR,
    default_value VARCHAR,
    parent_column INTEGER,
    default_value_type VARCHAR,
    default_value_dialect VARCHAR,
    begin_snapshot INTEGER NOT NULL,
    end_snapshot INTEGER
);

CREATE TABLE IF NOT EXISTS ducklake_data_file (
    data_file_id INTEGER PRIMARY KEY,
    table_id INTEGER NOT NULL,
    path VARCHAR NOT NULL,
    path_is_relative BOOLEAN NOT NULL DEFAULT 1,
    file_size_bytes INTEGER NOT NULL,
    footer_size INTEGER,
    encryption_key VARCHAR,
    record_count INTEGER,
    row_id_start INTEGER,
    mapping_id INTEGER,
    begin_snapshot INTEGER NOT NULL,
    end_snapshot INTEGER
);

-- Per-table row-lineage counter (DuckLake spec). `next_row_id` is the
-- monotonic rowid allocator: a new data file gets its `row_id_start` from
-- the current value, then we advance by `record_count` in the same
-- transaction. `record_count` and `file_size_bytes` mirror the currently-
-- visible totals so DuckDB's `ducklake_table_info` aggregate sees correct
-- numbers for tables we wrote.
CREATE TABLE IF NOT EXISTS ducklake_table_stats (
    table_id INTEGER PRIMARY KEY,
    record_count INTEGER NOT NULL DEFAULT 0,
    next_row_id INTEGER NOT NULL DEFAULT 0,
    file_size_bytes INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS ducklake_delete_file (
    delete_file_id INTEGER PRIMARY KEY,
    data_file_id INTEGER NOT NULL,
    table_id INTEGER NOT NULL,
    path VARCHAR NOT NULL,
    path_is_relative BOOLEAN NOT NULL DEFAULT 1,
    file_size_bytes INTEGER NOT NULL,
    footer_size INTEGER,
    encryption_key VARCHAR,
    delete_count INTEGER,
    begin_snapshot INTEGER NOT NULL,
    end_snapshot INTEGER
);

-- Files queued for physical deletion by the two-phase vacuum (DuckLake spec).
-- `expire_snapshots` GCs unreachable catalog rows and records the orphaned
-- physical paths here; `cleanup_old_files` deletes the objects and removes
-- these rows. `path` is stored relative to the catalog `data_path` root
-- (i.e. already resolved through schema/table) so cleanup needs only a
-- single-level join with `data_path`. Mirrors the upstream
-- `ducklake_files_scheduled_for_deletion` table.
CREATE TABLE IF NOT EXISTS ducklake_files_scheduled_for_deletion (
    data_file_id INTEGER NOT NULL,
    path VARCHAR NOT NULL,
    path_is_relative BOOLEAN NOT NULL DEFAULT 1,
    schedule_start TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"#;

/// SQLite-based metadata writer for DuckLake catalogs.
#[derive(Debug, Clone)]
pub struct SqliteMetadataWriter {
    pool: SqlitePool,
}

impl SqliteMetadataWriter {
    pub async fn new(connection_string: &str) -> Result<Self> {
        Self::with_max_connections(connection_string, DEFAULT_MAX_CONNECTIONS).await
    }

    pub async fn with_max_connections(
        connection_string: &str,
        max_connections: u32,
    ) -> Result<Self> {
        let pool = SqlitePoolOptions::new()
            .max_connections(max_connections)
            .connect(connection_string)
            .await?;
        Ok(Self {
            pool,
        })
    }

    pub async fn new_with_init(connection_string: &str) -> Result<Self> {
        let writer = Self::new(connection_string).await?;
        writer.initialize_schema()?;
        Ok(writer)
    }

    /// Tombstone a live table at a new "drop snapshot".
    ///
    /// Allocates a fresh snapshot and sets `end_snapshot = <drop snapshot>` on the
    /// currently-live rows for the named table in `ducklake_table`,
    /// `ducklake_column`, `ducklake_data_file`, and `ducklake_delete_file`. Reads at
    /// any snapshot `>=` the drop snapshot no longer see the table; earlier snapshots
    /// are unaffected (time travel preserved). This is the single-catalog mirror of
    /// [`crate::multicatalog::MulticatalogManager::drop_table_in_catalog`].
    ///
    /// Idempotent: returns `Ok(false)` (no snapshot allocated) when no live
    /// `(schema, table)` pair exists. Physical data files are not removed — that is
    /// the job of [`Self::expire_snapshots`] + [`crate::maintenance::cleanup_old_files_sqlite`].
    ///
    /// The table's `ducklake_table_stats` row is intentionally left in place:
    /// `next_row_id` must stay monotonic across the table's lifetime, and a
    /// recreate-after-drop gets a fresh `table_id`. The orphaned stats row is
    /// reclaimed by [`Self::expire_snapshots`] once the table is fully expired.
    pub fn drop_table(&self, schema_name: &str, table_name: &str) -> Result<bool> {
        validate_name(schema_name, "Schema")?;
        validate_name(table_name, "Table")?;
        block_on(async {
            let mut tx = self.pool.begin().await?;

            // Resolve the live table_id. `end_snapshot IS NULL` on both schema and
            // table makes an already-dropped table a no-op (idempotent).
            let table_id: i64 = match sqlx::query(
                "SELECT t.table_id FROM ducklake_table t
                 JOIN ducklake_schema s ON s.schema_id = t.schema_id
                 WHERE s.schema_name = ? AND s.end_snapshot IS NULL
                   AND t.table_name = ? AND t.end_snapshot IS NULL",
            )
            .bind(schema_name)
            .bind(table_name)
            .fetch_optional(&mut *tx)
            .await?
            {
                Some(r) => r.try_get(0)?,
                None => {
                    tx.commit().await?;
                    return Ok(false);
                },
            };

            let drop_snapshot = reserve_ids(&mut tx, "next_snapshot_id", 1).await?;
            sqlx::query(
                "INSERT INTO ducklake_snapshot (snapshot_id, snapshot_time) VALUES (?, CURRENT_TIMESTAMP)",
            )
            .bind(drop_snapshot)
            .execute(&mut *tx)
            .await?;

            for child in
                ["ducklake_table", "ducklake_column", "ducklake_data_file", "ducklake_delete_file"]
            {
                sqlx::query(&format!(
                    "UPDATE {child} SET end_snapshot = ?
                     WHERE table_id = ? AND end_snapshot IS NULL"
                ))
                .bind(drop_snapshot)
                .bind(table_id)
                .execute(&mut *tx)
                .await?;
            }

            tx.commit().await?;
            Ok(true)
        })
    }

    /// Expire snapshots and garbage-collect the catalog metadata they leave behind.
    ///
    /// Ports the official `ducklake_expire_snapshots` / `DuckLakeMetadataManager::DeleteSnapshots`.
    /// The most recent snapshot is never expired. After deleting the chosen snapshot
    /// rows, every table / data file / delete file no longer reachable by any surviving
    /// snapshot is removed from the catalog and its physical path is recorded in
    /// `ducklake_files_scheduled_for_deletion` for later physical deletion via
    /// [`crate::maintenance::cleanup_old_files_sqlite`]. Returns the expired snapshots.
    ///
    /// Reachability is global here — correct for a single-catalog SQLite metadata DB.
    /// The whole operation runs in one transaction, so a crash can't leave scheduled
    /// rows out of sync with the catalog.
    pub fn expire_snapshots(&self, criteria: ExpireCriteria) -> Result<Vec<ExpiredSnapshot>> {
        block_on(async {
            let mut tx = self.pool.begin().await?;

            // The most recent snapshot is never expirable.
            let most_recent: Option<i64> =
                sqlx::query("SELECT MAX(snapshot_id) FROM ducklake_snapshot")
                    .fetch_one(&mut *tx)
                    .await?
                    .try_get(0)?;
            let most_recent = match most_recent {
                Some(id) => id,
                None => {
                    tx.commit().await?;
                    return Ok(Vec::new());
                },
            };

            // 1. Resolve the snapshots to expire (excluding the most recent).
            let candidates: Vec<ExpiredSnapshot> = match &criteria {
                ExpireCriteria::Versions(versions) => {
                    let ids: Vec<i64> = versions
                        .iter()
                        .copied()
                        .filter(|&v| v != most_recent)
                        .collect();
                    if ids.is_empty() {
                        tx.commit().await?;
                        return Ok(Vec::new());
                    }
                    let rows = sqlx::query(&format!(
                        "SELECT snapshot_id, snapshot_time FROM ducklake_snapshot
                         WHERE snapshot_id IN ({}) ORDER BY snapshot_id",
                        id_list(&ids)
                    ))
                    .fetch_all(&mut *tx)
                    .await?;
                    rows_to_snapshots(rows)?
                },
                ExpireCriteria::OlderThan(ts) => {
                    let rows = sqlx::query(
                        "SELECT snapshot_id, snapshot_time FROM ducklake_snapshot
                         WHERE snapshot_id != ? AND snapshot_time < ? ORDER BY snapshot_id",
                    )
                    .bind(most_recent)
                    .bind(format_sql_timestamp(ts))
                    .fetch_all(&mut *tx)
                    .await?;
                    rows_to_snapshots(rows)?
                },
            };
            if candidates.is_empty() {
                tx.commit().await?;
                return Ok(Vec::new());
            }
            let expire_ids: Vec<i64> = candidates.iter().map(|s| s.snapshot_id).collect();

            // 2. Delete the snapshot rows themselves.
            sqlx::query(&format!(
                "DELETE FROM ducklake_snapshot WHERE snapshot_id IN ({})",
                id_list(&expire_ids)
            ))
            .execute(&mut *tx)
            .await?;

            // 3. Tables whose lifetime is no longer covered by any surviving snapshot
            //    AND which have no surviving live version.
            let dead_tables: Vec<i64> = sqlx::query(
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
            .map(|r| r.try_get::<i64, _>(0))
            .collect::<std::result::Result<_, _>>()?;
            let dead_table_filter = if dead_tables.is_empty() {
                "0".to_string()
            } else {
                format!("df.table_id IN ({})", id_list(&dead_tables))
            };

            // 4. Data files no longer referenced by any surviving snapshot (or belonging
            //    to a dead table): schedule their physical paths, then drop the rows.
            let dead_data_files = sqlx::query(&format!(
                "SELECT df.data_file_id, {RESOLVED_PATH} AS resolved_path, {REL_FLAG} AS rel
                 FROM ducklake_data_file df
                 JOIN ducklake_table t ON t.table_id = df.table_id
                 JOIN ducklake_schema s ON s.schema_id = t.schema_id
                 WHERE ({dead_table_filter}) OR (df.end_snapshot IS NOT NULL AND NOT EXISTS (
                     SELECT 1 FROM ducklake_snapshot
                     WHERE snapshot_id >= df.begin_snapshot AND snapshot_id < df.end_snapshot))"
            ))
            .fetch_all(&mut *tx)
            .await?;
            let data_file_ids = schedule_files(&mut tx, dead_data_files).await?;
            if !data_file_ids.is_empty() {
                sqlx::query(&format!(
                    "DELETE FROM ducklake_data_file WHERE data_file_id IN ({})",
                    id_list(&data_file_ids)
                ))
                .execute(&mut *tx)
                .await?;
            }

            // 5. Delete files orphaned by the data files above, by a dead table, or no
            //    longer referenced by any surviving snapshot. (Our writer does not emit
            //    delete files yet, so this is a no-op for catalogs we produce.)
            let dead_data_filter = if data_file_ids.is_empty() {
                "0".to_string()
            } else {
                format!("df.data_file_id IN ({})", id_list(&data_file_ids))
            };
            let dead_delete_table_filter = if dead_tables.is_empty() {
                "0".to_string()
            } else {
                format!("df.table_id IN ({})", id_list(&dead_tables))
            };
            let dead_delete_files = sqlx::query(&format!(
                "SELECT df.delete_file_id, {RESOLVED_PATH} AS resolved_path, {REL_FLAG} AS rel
                 FROM ducklake_delete_file df
                 JOIN ducklake_table t ON t.table_id = df.table_id
                 JOIN ducklake_schema s ON s.schema_id = t.schema_id
                 WHERE ({dead_data_filter}) OR ({dead_delete_table_filter})
                    OR (df.end_snapshot IS NOT NULL AND NOT EXISTS (
                        SELECT 1 FROM ducklake_snapshot
                        WHERE snapshot_id >= df.begin_snapshot AND snapshot_id < df.end_snapshot))"
            ))
            .fetch_all(&mut *tx)
            .await?;
            let delete_file_ids = schedule_files(&mut tx, dead_delete_files).await?;
            if !delete_file_ids.is_empty() {
                sqlx::query(&format!(
                    "DELETE FROM ducklake_delete_file WHERE delete_file_id IN ({})",
                    id_list(&delete_file_ids)
                ))
                .execute(&mut *tx)
                .await?;
            }

            // 6. Reclaim per-table metadata for fully-expired tables (this is where a
            //    dropped table's orphaned ducklake_table_stats row is removed).
            if !dead_tables.is_empty() {
                let dead = id_list(&dead_tables);
                for table in ["ducklake_table", "ducklake_table_stats", "ducklake_column"] {
                    sqlx::query(&format!("DELETE FROM {table} WHERE table_id IN ({dead})"))
                        .execute(&mut *tx)
                        .await?;
                }
            }

            // 7. Reclaim schemas no longer covered by any surviving snapshot.
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

    /// List files scheduled for physical deletion, optionally filtered by schedule time.
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
                CleanupCriteria::OlderThan(ts) => {
                    sqlx::query(
                        "SELECT data_file_id, path, path_is_relative
                     FROM ducklake_files_scheduled_for_deletion
                     WHERE schedule_start < ?",
                    )
                    .bind(format_sql_timestamp(ts))
                    .fetch_all(&self.pool)
                    .await?
                },
            };
            rows.into_iter()
                .map(|r| {
                    Ok(ScheduledFile {
                        data_file_id: r.try_get(0)?,
                        path: r.try_get(1)?,
                        path_is_relative: r.try_get::<i64, _>(2)? != 0,
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
            sqlx::query(&format!(
                "DELETE FROM ducklake_files_scheduled_for_deletion WHERE data_file_id IN ({})",
                id_list(ids)
            ))
            .execute(&self.pool)
            .await?;
            Ok(())
        })
    }

    /// Every physical file currently referenced by the catalog: live + tombstoned
    /// data files, live + tombstoned delete files, and the still-scheduled
    /// deletion queue (those rows haven't been processed by `cleanup_old_files`
    /// yet, so the files might still exist on disk and must not be touched by
    /// orphan cleanup). Each row is `(path, path_is_relative)` resolved relative
    /// to the catalog `data_path` root; the caller resolves against the actual
    /// `data_path` for comparison with object_store keys.
    ///
    /// Used by [`crate::maintenance::delete_orphaned_files_sqlite`].
    pub(crate) fn list_referenced_paths(&self) -> Result<Vec<(String, bool)>> {
        block_on(async {
            let q = format!(
                "SELECT {RESOLVED_PATH} AS p, {REL_FLAG} AS rel
                 FROM ducklake_data_file df
                 JOIN ducklake_table t ON t.table_id = df.table_id
                 JOIN ducklake_schema s ON s.schema_id = t.schema_id
                 UNION ALL
                 SELECT {RESOLVED_PATH} AS p, {REL_FLAG} AS rel
                 FROM ducklake_delete_file df
                 JOIN ducklake_table t ON t.table_id = df.table_id
                 JOIN ducklake_schema s ON s.schema_id = t.schema_id
                 UNION ALL
                 SELECT path AS p, CAST(path_is_relative AS INTEGER) AS rel
                 FROM ducklake_files_scheduled_for_deletion"
            );
            let rows = sqlx::query(&q).fetch_all(&self.pool).await?;
            rows.into_iter()
                .map(|r| {
                    let p: String = r.try_get(0)?;
                    let rel: i64 = r.try_get(1)?;
                    Ok((p, rel != 0))
                })
                .collect()
        })
    }
}

/// SQL expression yielding a file path resolved relative to the catalog `data_path`
/// root, mirroring the read-side hierarchical resolution (file → table → schema →
/// data_path). An absolute path anywhere in the chain short-circuits to that absolute
/// path. Assumes `/`-joined relative names (everything our writer produces).
const RESOLVED_PATH: &str = "CASE
    WHEN NOT df.path_is_relative THEN df.path
    WHEN NOT t.path_is_relative THEN t.path || '/' || df.path
    ELSE s.path || '/' || t.path || '/' || df.path
END";

/// Companion to [`RESOLVED_PATH`]: 1 only when the whole chain is relative (so the
/// resolved path is relative to `data_path`), else 0 (the resolved path is absolute).
const REL_FLAG: &str =
    "(CASE WHEN df.path_is_relative AND t.path_is_relative AND s.path_is_relative
           THEN 1 ELSE 0 END)";

/// Map `(snapshot_id, snapshot_time)` rows to [`ExpiredSnapshot`]s.
fn rows_to_snapshots(rows: Vec<sqlx::sqlite::SqliteRow>) -> Result<Vec<ExpiredSnapshot>> {
    rows.into_iter()
        .map(|r| {
            Ok(ExpiredSnapshot {
                snapshot_id: r.try_get(0)?,
                snapshot_time: r.try_get(1)?,
            })
        })
        .collect()
}

/// Insert `(id, resolved_path, rel)` rows (as produced by [`RESOLVED_PATH`]/[`REL_FLAG`])
/// into `ducklake_files_scheduled_for_deletion` and return the ids scheduled.
async fn schedule_files(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    rows: Vec<sqlx::sqlite::SqliteRow>,
) -> Result<Vec<i64>> {
    let mut ids = Vec::with_capacity(rows.len());
    for row in rows {
        let id: i64 = row.try_get(0)?;
        let path: String = row.try_get(1)?;
        let rel: i64 = row.try_get(2)?;
        sqlx::query(
            "INSERT INTO ducklake_files_scheduled_for_deletion
                 (data_file_id, path, path_is_relative, schedule_start)
             VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
        )
        .bind(id)
        .bind(&path)
        .bind(rel)
        .execute(&mut **tx)
        .await?;
        ids.push(id);
    }
    Ok(ids)
}

/// Atomically reserve `n` consecutive ids from a monotonic counter stored in
/// `ducklake_metadata` (seeded by `initialize_schema`), returning the LAST id of
/// the block — the reserved ids are `last - n + 1 ..= last`. The `UPDATE`
/// serializes under SQLite's single-writer lock, so concurrent writers (even to
/// different tables) never hand out overlapping ids. Used for `snapshot_id` and
/// `column_id`, which are reserved in `begin_write_transaction` and inserted at
/// the commit: rowid autoincrement can't be used there because inserting the
/// `ducklake_snapshot` row would advance the head (`MAX(snapshot_id)`) before the
/// data is committed.
async fn reserve_ids(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    key: &str,
    n: i64,
) -> Result<i64> {
    let last: i64 = sqlx::query(
        "UPDATE ducklake_metadata
         SET value = CAST(CAST(value AS INTEGER) + ? AS TEXT)
         WHERE key = ? AND scope IS NULL
         RETURNING CAST(value AS INTEGER)",
    )
    .bind(n)
    .bind(key)
    .fetch_one(&mut **tx)
    .await?
    .try_get(0)?;
    Ok(last)
}

/// Retire the prior generation's still-visible data files at `snapshot_id` and
/// zero the visible stat totals. The `begin_snapshot < snapshot_id` guard
/// spares files registered for *this* snapshot, so a multi-file write does not
/// retire its own siblings. `next_row_id` is left untouched (rowids stay
/// monotonic across the table's lifetime). Mirrors the Postgres writer.
async fn retire_prior_generation(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
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

/// The atomic commit point for a single-catalog write. Inserts the deferred
/// `ducklake_snapshot` row (its reserved id was returned by
/// `begin_write_transaction` but NOT inserted there), finalizes the column
/// generation, and — for `Replace` — retires the prior data generation. All
/// within the caller's transaction, so `COALESCE(MAX(snapshot_id), 0)` only
/// ever resolves to a fully-populated head (no transient empty read).
///
/// The column generation is deferred to here (rather than written in
/// `begin_write_transaction` like the Postgres backend) because the SQLite read
/// path resolves a table's columns by `end_snapshot IS NULL` ONLY (not
/// snapshot-scoped), so inserting the new generation at begin would leak it to
/// concurrent reads during the upload window. `column_ids` are the ids reserved
/// at begin and already baked into the staged parquet's `field_id` metadata.
async fn finalize_snapshot(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    table_id: i64,
    snapshot_id: i64,
    columns: &[ColumnDef],
    column_ids: &[i64],
    mode: WriteMode,
) -> Result<()> {
    // Insert the deferred snapshot row with its reserved id — THIS is the head
    // advance.
    sqlx::query(
        "INSERT INTO ducklake_snapshot (snapshot_id, snapshot_time) VALUES (?, CURRENT_TIMESTAMP)",
    )
    .bind(snapshot_id)
    .execute(&mut **tx)
    .await?;

    // Finalize the column generation: end the prior columns and insert the new
    // set with the reserved column_ids (matching the parquet field_ids).
    sqlx::query(
        "UPDATE ducklake_column SET end_snapshot = ?
         WHERE table_id = ? AND end_snapshot IS NULL",
    )
    .bind(snapshot_id)
    .bind(table_id)
    .execute(&mut **tx)
    .await?;

    for (order, (col, column_id)) in columns.iter().zip(column_ids.iter()).enumerate() {
        sqlx::query(
            "INSERT INTO ducklake_column
                 (column_id, table_id, column_name, column_type, column_order, nulls_allowed, begin_snapshot)
             VALUES (?, ?, ?, ?, ?, ?, ?)",
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
    }

    if mode == WriteMode::Replace {
        // Seed the stats row (first write to a brand-new table) so retire's
        // zero-update has a row, then retire the prior data generation.
        sqlx::query(
            "INSERT OR IGNORE INTO ducklake_table_stats
                 (table_id, record_count, next_row_id, file_size_bytes)
             VALUES (?, 0, 0, 0)",
        )
        .bind(table_id)
        .execute(&mut **tx)
        .await?;
        retire_prior_generation(tx, table_id, snapshot_id).await?;
    }
    Ok(())
}

impl MetadataWriter for SqliteMetadataWriter {
    fn create_snapshot(&self) -> Result<i64> {
        block_on(async {
            let mut tx = self.pool.begin().await?;
            let snapshot_id = reserve_ids(&mut tx, "next_snapshot_id", 1).await?;
            sqlx::query(
                "INSERT INTO ducklake_snapshot (snapshot_id, snapshot_time) VALUES (?, CURRENT_TIMESTAMP)",
            )
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
            let row = sqlx::query(
                "INSERT INTO ducklake_schema (schema_name, path, path_is_relative, begin_snapshot)
                 VALUES (?, ?, 1, ?) RETURNING schema_id",
            )
            .bind(name)
            .bind(schema_path)
            .bind(snapshot_id)
            .fetch_one(&self.pool)
            .await?;

            Ok((row.try_get(0)?, true))
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
            let row = sqlx::query(
                "INSERT INTO ducklake_table (schema_id, table_name, path, path_is_relative, begin_snapshot)
                 VALUES (?, ?, ?, 1, ?) RETURNING table_id",
            )
            .bind(schema_id)
            .bind(name)
            .bind(table_path)
            .bind(snapshot_id)
            .fetch_one(&self.pool)
            .await?;

            Ok((row.try_get(0)?, true))
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
            // Use a transaction to ensure atomicity: if column insertion fails,
            // we don't leave existing columns marked as ended
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
            // insert with explicit ids, keeping the allocator authoritative (the
            // write path's begin/commit reserve column ids from the same counter).
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
                .bind(&col.name)
                .bind(&col.ducklake_type)
                .bind(order as i64)
                .bind(col.is_nullable)
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
        snapshot_id: i64,
        file: &DataFileInfo,
        mode: WriteMode,
        columns: &[ColumnDef],
        column_ids: &[i64],
    ) -> Result<i64> {
        block_on(async {
            // Single atomic commit: insert the deferred snapshot row + finalize
            // the column generation + retire the prior generation (Replace),
            // then register this file and advance the monotonic row-lineage
            // counter — all in one transaction, so the head (MAX(snapshot_id))
            // only ever resolves to fully-populated data (no empty-read window).
            let mut tx = self.pool.begin().await?;

            finalize_snapshot(&mut tx, table_id, snapshot_id, columns, column_ids, mode).await?;

            // Seed the stats row for the Append path (Replace already seeded it
            // in finalize_snapshot); INSERT OR IGNORE is a no-op if it exists.
            sqlx::query(
                "INSERT OR IGNORE INTO ducklake_table_stats
                     (table_id, record_count, next_row_id, file_size_bytes)
                 VALUES (?, 0, 0, 0)",
            )
            .bind(table_id)
            .execute(&mut *tx)
            .await?;

            let stats_row =
                sqlx::query("SELECT next_row_id FROM ducklake_table_stats WHERE table_id = ?")
                    .bind(table_id)
                    .fetch_one(&mut *tx)
                    .await?;
            let row_id_start: i64 = stats_row.try_get(0)?;

            let data_file_row = sqlx::query(
                "INSERT INTO ducklake_data_file
                     (table_id, path, path_is_relative, file_size_bytes,
                      footer_size, record_count, row_id_start, begin_snapshot)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?) RETURNING data_file_id",
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
            let data_file_id: i64 = data_file_row.try_get(0)?;

            // Advance the counter and accumulate stats. `next_row_id`
            // monotonically increases over the table's lifetime — rowids
            // are never reused, even after end-snapshot.
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

            tx.commit().await?;
            Ok(data_file_id)
        })
    }

    fn publish_snapshot(
        &self,
        table_id: i64,
        snapshot_id: i64,
        mode: WriteMode,
        columns: &[ColumnDef],
        column_ids: &[i64],
    ) -> Result<()> {
        // Fileless commit point. Single-catalog SQLite defers the snapshot-row
        // insert out of begin_write_transaction, so this is no longer a no-op:
        // it inserts the deferred snapshot row + column generation and, for
        // Replace, retires the prior generation — making the new head visible
        // atomically. Used by CREATE TABLE (schema.rs); the crate's own write
        // path always registers a file (even for zero rows) and so commits via
        // register_data_file instead.
        block_on(async {
            let mut tx = self.pool.begin().await?;
            finalize_snapshot(&mut tx, table_id, snapshot_id, columns, column_ids, mode).await?;
            tx.commit().await?;
            Ok(())
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
            let row =
                sqlx::query("SELECT value FROM ducklake_metadata WHERE key = ? AND scope IS NULL")
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
            sqlx::query("DELETE FROM ducklake_metadata WHERE key = 'data_path' AND scope IS NULL")
                .execute(&self.pool)
                .await?;

            sqlx::query(
                "INSERT INTO ducklake_metadata (key, value, scope)
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
            sqlx::query(SQL_CREATE_SCHEMA).execute(&self.pool).await?;
            // Seed the monotonic id allocators. snapshot_id and column_id are
            // reserved in begin_write_transaction and inserted at the commit, so
            // they can't use rowid autoincrement (inserting the ducklake_snapshot
            // row would advance the head before the data is committed). The
            // counters give collision-free allocation across concurrent writers.
            // Idempotent on re-open, and seeded from the current MAX so a
            // pre-existing catalog continues without reusing ids.
            sqlx::query(
                "INSERT INTO ducklake_metadata (key, value, scope)
                 SELECT 'next_snapshot_id',
                        CAST(COALESCE((SELECT MAX(snapshot_id) FROM ducklake_snapshot), 0) AS TEXT),
                        NULL
                 WHERE NOT EXISTS (
                     SELECT 1 FROM ducklake_metadata WHERE key = 'next_snapshot_id' AND scope IS NULL
                 )",
            )
            .execute(&self.pool)
            .await?;
            sqlx::query(
                "INSERT INTO ducklake_metadata (key, value, scope)
                 SELECT 'next_column_id',
                        CAST(COALESCE((SELECT MAX(column_id) FROM ducklake_column), 0) AS TEXT),
                        NULL
                 WHERE NOT EXISTS (
                     SELECT 1 FROM ducklake_metadata WHERE key = 'next_column_id' AND scope IS NULL
                 )",
            )
            .execute(&self.pool)
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

            // Reserve the snapshot id from the monotonic counter WITHOUT inserting
            // the row. The head is COALESCE(MAX(snapshot_id), 0), so inserting here
            // would advance the head before the data file exists — the
            // transient-empty-read bug. The row is inserted at the atomic commit
            // point (`finalize_snapshot`, from register_data_file /
            // publish_snapshot). The counter serializes under SQLite's write lock,
            // so concurrent writers get distinct ids and never collide.
            let snapshot_id = reserve_ids(&mut tx, "next_snapshot_id", 1).await?;

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
                    let row = sqlx::query(
                        "INSERT INTO ducklake_schema (schema_name, path, path_is_relative, begin_snapshot)
                         VALUES (?, ?, 1, ?) RETURNING schema_id",
                    )
                    .bind(schema_name)
                    .bind(schema_name)
                    .bind(snapshot_id)
                    .fetch_one(&mut *tx)
                    .await?;
                    row.try_get(0)?
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
                    let row = sqlx::query(
                        "INSERT INTO ducklake_table (schema_id, table_name, path, path_is_relative, begin_snapshot)
                         VALUES (?, ?, ?, 1, ?) RETURNING table_id",
                    )
                    .bind(schema_id)
                    .bind(table_name)
                    .bind(table_name)
                    .bind(snapshot_id)
                    .fetch_one(&mut *tx)
                    .await?;
                    row.try_get(0)?
                }
            };

            // Get existing columns to check schema compatibility for appends
            let rows = sqlx::query(
                "SELECT column_name, column_type, nulls_allowed
                 FROM ducklake_column
                 WHERE table_id = ? AND end_snapshot IS NULL
                 ORDER BY column_order",
            )
            .bind(table_id)
            .fetch_all(&mut *tx)
            .await?;

            let mut existing_columns: Vec<(String, String, bool)> = Vec::with_capacity(rows.len());
            for row in rows {
                let name: String = row.try_get(0)?;
                let col_type: String = row.try_get(1)?;
                let nullable: bool = row.try_get::<Option<bool>, _>(2)?.unwrap_or(true);
                existing_columns.push((name, col_type, nullable));
            }

            // For append mode, validate schema compatibility with evolution rules:
            // - Allowed: add nullable columns, remove columns, reorder columns
            // - Disallowed: add non-nullable columns, type changes for existing columns
            if mode == WriteMode::Append && !existing_columns.is_empty() {
                use std::collections::HashMap;

                // Build map of existing columns: name -> (type, nullable)
                let existing_map: HashMap<&str, (&str, bool)> = existing_columns
                    .iter()
                    .map(|(name, col_type, nullable)| {
                        (name.as_str(), (col_type.as_str(), *nullable))
                    })
                    .collect();

                for new_col in columns.iter() {
                    if let Some((existing_type, _existing_nullable)) =
                        existing_map.get(new_col.name.as_str())
                    {
                        // Column exists - check type compatibility (normalize aliases + allow promotions)
                        if !crate::types::types_compatible(existing_type, &new_col.ducklake_type) {
                            return Err(crate::error::DuckLakeError::InvalidConfig(format!(
                                "Schema evolution error: column '{}' has type '{}' in existing table but '{}' in new schema. Type changes are not allowed.",
                                new_col.name, existing_type, new_col.ducklake_type
                            )));
                        }
                        // Note: We allow nullable changes (strict -> nullable is safe for reads)
                    } else {
                        // New column - must be nullable
                        if !new_col.is_nullable {
                            return Err(crate::error::DuckLakeError::InvalidConfig(format!(
                                "Schema evolution error: new column '{}' must be nullable. Adding non-nullable columns is not allowed.",
                                new_col.name
                            )));
                        }
                    }
                }
                // Columns in existing but not in new schema are implicitly removed - this is allowed
            }

            // Reserve a contiguous column_id block from the monotonic counter
            // WITHOUT inserting the column rows. The new column generation is
            // written at the commit point (`finalize_snapshot`), not here, because
            // the SQLite read path resolves a table's columns by `end_snapshot IS
            // NULL` only (not snapshot-scoped) — inserting at begin would leak the
            // new generation to concurrent reads during the upload window. The
            // reserved ids are returned so the staged parquet's `field_id`
            // metadata matches the ids eventually committed, and the counter keeps
            // concurrent writers' blocks disjoint.
            let n = columns.len() as i64;
            let last_column_id = reserve_ids(&mut tx, "next_column_id", n).await?;
            let column_ids: Vec<i64> = ((last_column_id - n + 1)..=last_column_id).collect();

            // No snapshot row, no column rows, and no Replace retirement are
            // written here — all are deferred to the atomic commit so the head
            // never resolves to an incomplete snapshot. TX-A commits only the
            // idempotent get-or-create schema/table rows; they carry
            // begin_snapshot = the reserved id and stay invisible until the
            // snapshot publishes, since schema/table reads ARE snapshot-scoped.
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

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    async fn create_test_writer() -> (SqliteMetadataWriter, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let conn_str = format!("sqlite:{}?mode=rwc", db_path.display());
        let writer = SqliteMetadataWriter::new_with_init(&conn_str)
            .await
            .unwrap();
        (writer, temp_dir)
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_create_snapshot() {
        let (writer, _temp) = create_test_writer().await;

        let snap1 = writer.create_snapshot().unwrap();
        assert_eq!(snap1, 1);

        let snap2 = writer.create_snapshot().unwrap();
        assert_eq!(snap2, 2);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_get_or_create_schema() {
        let (writer, _temp) = create_test_writer().await;
        let snapshot_id = writer.create_snapshot().unwrap();

        // Create new schema
        let (schema_id, created) = writer
            .get_or_create_schema("main", None, snapshot_id)
            .unwrap();
        assert!(created);
        assert_eq!(schema_id, 1);

        // Get existing schema
        let (schema_id2, created2) = writer
            .get_or_create_schema("main", None, snapshot_id)
            .unwrap();
        assert!(!created2);
        assert_eq!(schema_id2, 1);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_get_or_create_table() {
        let (writer, _temp) = create_test_writer().await;
        let snapshot_id = writer.create_snapshot().unwrap();
        let (schema_id, _) = writer
            .get_or_create_schema("main", None, snapshot_id)
            .unwrap();

        // Create new table
        let (table_id, created) = writer
            .get_or_create_table(schema_id, "users", None, snapshot_id)
            .unwrap();
        assert!(created);
        assert_eq!(table_id, 1);

        // Get existing table
        let (table_id2, created2) = writer
            .get_or_create_table(schema_id, "users", None, snapshot_id)
            .unwrap();
        assert!(!created2);
        assert_eq!(table_id2, 1);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_set_columns() {
        let (writer, _temp) = create_test_writer().await;
        let snapshot_id = writer.create_snapshot().unwrap();
        let (schema_id, _) = writer
            .get_or_create_schema("main", None, snapshot_id)
            .unwrap();
        let (table_id, _) = writer
            .get_or_create_table(schema_id, "users", None, snapshot_id)
            .unwrap();

        let columns = vec![
            ColumnDef::new("id", "int64", false).unwrap(),
            ColumnDef::new("name", "varchar", true).unwrap(),
        ];

        let column_ids = writer.set_columns(table_id, &columns, snapshot_id).unwrap();
        assert_eq!(column_ids.len(), 2);
        assert_eq!(column_ids[0], 1);
        assert_eq!(column_ids[1], 2);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_register_data_file() {
        let (writer, _temp) = create_test_writer().await;
        // Reserve (do not create) the snapshot: register_data_file inserts it.
        let snapshot_id = reserve_snapshot(&writer).await;
        let (schema_id, _) = writer
            .get_or_create_schema("main", None, snapshot_id)
            .unwrap();
        let (table_id, _) = writer
            .get_or_create_table(schema_id, "users", None, snapshot_id)
            .unwrap();

        let file = DataFileInfo::new("data.parquet", 1024, 100).with_footer_size(256);

        let file_id = writer
            .register_data_file(table_id, snapshot_id, &file, WriteMode::Append, &[], &[])
            .unwrap();
        assert_eq!(file_id, 1);
    }

    /// Reserve the next snapshot id the way `begin_write_transaction` now does
    /// (bump the monotonic counter WITHOUT inserting the row), so a subsequent
    /// `register_data_file` can insert it atomically at the commit. Tests that
    /// drive `register_data_file` directly must reserve (not `create_snapshot`)
    /// the snapshot they register, since registration owns the snapshot insert.
    async fn reserve_snapshot(writer: &SqliteMetadataWriter) -> i64 {
        sqlx::query(
            "UPDATE ducklake_metadata SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT)
             WHERE key = 'next_snapshot_id' AND scope IS NULL
             RETURNING CAST(value AS INTEGER)",
        )
        .fetch_one(&writer.pool)
        .await
        .unwrap()
        .try_get(0)
        .unwrap()
    }

    /// Helper for the row-lineage tests: read back what `register_data_file`
    /// wrote into the catalog so we can assert on `row_id_start` and the
    /// stats counter directly.
    async fn read_row_id_start(writer: &SqliteMetadataWriter, file_id: i64) -> Option<i64> {
        let row = sqlx::query("SELECT row_id_start FROM ducklake_data_file WHERE data_file_id = ?")
            .bind(file_id)
            .fetch_one(&writer.pool)
            .await
            .unwrap();
        row.try_get(0).ok()
    }

    async fn read_table_stats(writer: &SqliteMetadataWriter, table_id: i64) -> (i64, i64, i64) {
        let row = sqlx::query(
            "SELECT record_count, next_row_id, file_size_bytes
             FROM ducklake_table_stats WHERE table_id = ?",
        )
        .bind(table_id)
        .fetch_one(&writer.pool)
        .await
        .unwrap();
        (
            row.try_get(0).unwrap(),
            row.try_get(1).unwrap(),
            row.try_get(2).unwrap(),
        )
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn row_id_start_advances_across_inserts() {
        // Two INSERTs against the same table should hand out non-overlapping
        // [row_id_start, row_id_start + record_count) ranges. Counter is
        // initialized lazily on first register_data_file.
        let (writer, _temp) = create_test_writer().await;
        // Two separate writes (each registers its own snapshot atomically).
        let snap1 = reserve_snapshot(&writer).await;
        let (schema_id, _) = writer.get_or_create_schema("main", None, snap1).unwrap();
        let (table_id, _) = writer
            .get_or_create_table(schema_id, "t", None, snap1)
            .unwrap();

        let f1_id = writer
            .register_data_file(
                table_id,
                snap1,
                &DataFileInfo::new("a.parquet", 100, 3),
                WriteMode::Append,
                &[],
                &[],
            )
            .unwrap();
        let snap2 = reserve_snapshot(&writer).await;
        let f2_id = writer
            .register_data_file(
                table_id,
                snap2,
                &DataFileInfo::new("b.parquet", 250, 7),
                WriteMode::Append,
                &[],
                &[],
            )
            .unwrap();

        assert_eq!(read_row_id_start(&writer, f1_id).await, Some(0));
        assert_eq!(read_row_id_start(&writer, f2_id).await, Some(3));

        let (records, next, bytes) = read_table_stats(&writer, table_id).await;
        assert_eq!(records, 10, "record_count = 3 + 7");
        assert_eq!(next, 10, "next_row_id advances by sum of record_counts");
        assert_eq!(bytes, 350, "file_size_bytes accumulates");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn end_table_files_preserves_next_row_id() {
        // Replace must reset record_count and file_size_bytes (those reflect
        // currently-visible rows) but keep next_row_id monotonic so the next
        // generation of files gets fresh, non-overlapping rowids.
        let (writer, _temp) = create_test_writer().await;
        let snap1 = reserve_snapshot(&writer).await;
        let (schema_id, _) = writer.get_or_create_schema("main", None, snap1).unwrap();
        let (table_id, _) = writer
            .get_or_create_table(schema_id, "t", None, snap1)
            .unwrap();

        writer
            .register_data_file(
                table_id,
                snap1,
                &DataFileInfo::new("a.parquet", 100, 5),
                WriteMode::Append,
                &[],
                &[],
            )
            .unwrap();

        let snap2 = writer.create_snapshot().unwrap();
        writer.end_table_files(table_id, snap2).unwrap();

        let (records, next, bytes) = read_table_stats(&writer, table_id).await;
        assert_eq!(records, 0, "record_count cleared after end_table_files");
        assert_eq!(next, 5, "next_row_id preserved (monotonic across lifetime)");
        assert_eq!(bytes, 0, "file_size_bytes cleared");

        let snap3 = reserve_snapshot(&writer).await;
        let f2_id = writer
            .register_data_file(
                table_id,
                snap3,
                &DataFileInfo::new("b.parquet", 200, 2),
                WriteMode::Append,
                &[],
                &[],
            )
            .unwrap();
        assert_eq!(
            read_row_id_start(&writer, f2_id).await,
            Some(5),
            "post-replace files must start at the preserved counter, not 0",
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn row_id_start_works_when_stats_row_missing() {
        // Defensive path: a table that existed before this writer started
        // maintaining ducklake_table_stats. First register_data_file must
        // self-initialize the stats row rather than fail.
        let (writer, _temp) = create_test_writer().await;
        let snapshot_id = reserve_snapshot(&writer).await;
        let (schema_id, _) = writer
            .get_or_create_schema("main", None, snapshot_id)
            .unwrap();
        let (table_id, _) = writer
            .get_or_create_table(schema_id, "legacy", None, snapshot_id)
            .unwrap();

        // Simulate a "legacy" table by deleting any stats row that
        // get_or_create_table may have written (it doesn't today, but be
        // explicit so the test stays meaningful if that changes).
        sqlx::query("DELETE FROM ducklake_table_stats WHERE table_id = ?")
            .bind(table_id)
            .execute(&writer.pool)
            .await
            .unwrap();

        let file_id = writer
            .register_data_file(
                table_id,
                snapshot_id,
                &DataFileInfo::new("a.parquet", 50, 4),
                WriteMode::Append,
                &[],
                &[],
            )
            .unwrap();
        assert_eq!(read_row_id_start(&writer, file_id).await, Some(0));
        let (records, next, _) = read_table_stats(&writer, table_id).await;
        assert_eq!(records, 4);
        assert_eq!(next, 4);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_end_table_files() {
        let (writer, _temp) = create_test_writer().await;
        let snapshot1 = reserve_snapshot(&writer).await;
        let (schema_id, _) = writer
            .get_or_create_schema("main", None, snapshot1)
            .unwrap();
        let (table_id, _) = writer
            .get_or_create_table(schema_id, "users", None, snapshot1)
            .unwrap();

        // Register a file
        let file = DataFileInfo::new("data1.parquet", 1024, 100);
        writer
            .register_data_file(table_id, snapshot1, &file, WriteMode::Append, &[], &[])
            .unwrap();

        // End files at new snapshot
        let snapshot2 = writer.create_snapshot().unwrap();
        let ended = writer.end_table_files(table_id, snapshot2).unwrap();
        assert_eq!(ended, 1);

        // End again should affect 0 files
        let ended2 = writer.end_table_files(table_id, snapshot2).unwrap();
        assert_eq!(ended2, 0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_data_path() {
        let (writer, _temp) = create_test_writer().await;

        // Set data path
        writer.set_data_path("/data/path").unwrap();

        // Get data path
        let path = writer.get_data_path().unwrap();
        assert_eq!(path, "/data/path");

        // Update data path
        writer.set_data_path("/new/path").unwrap();
        let path2 = writer.get_data_path().unwrap();
        assert_eq!(path2, "/new/path");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_get_or_create_schema_empty_name_rejected() {
        let (writer, _temp) = create_test_writer().await;
        let snapshot_id = writer.create_snapshot().unwrap();
        let result = writer.get_or_create_schema("", None, snapshot_id);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("empty"), "Expected 'empty' in: {err_msg}");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_get_or_create_schema_control_char_rejected() {
        let (writer, _temp) = create_test_writer().await;
        let snapshot_id = writer.create_snapshot().unwrap();
        let result = writer.get_or_create_schema("bad\0schema", None, snapshot_id);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("control character"),
            "Expected 'control character' in: {err_msg}"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_begin_write_transaction_empty_schema_name_rejected() {
        let (writer, _temp) = create_test_writer().await;
        let columns = vec![ColumnDef::new("id", "int64", false).unwrap()];
        let result = writer.begin_write_transaction("", "table", &columns, WriteMode::Replace);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("empty"), "Expected 'empty' in: {err_msg}");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_begin_write_transaction_empty_table_name_rejected() {
        let (writer, _temp) = create_test_writer().await;
        let columns = vec![ColumnDef::new("id", "int64", false).unwrap()];
        let result = writer.begin_write_transaction("main", "", &columns, WriteMode::Replace);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("empty"), "Expected 'empty' in: {err_msg}");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_begin_write_transaction_control_char_names_rejected() {
        let (writer, _temp) = create_test_writer().await;
        let columns = vec![ColumnDef::new("id", "int64", false).unwrap()];

        // Control char in schema name
        let result =
            writer.begin_write_transaction("bad\nschema", "table", &columns, WriteMode::Replace);
        assert!(result.is_err());

        // Control char in table name
        let result =
            writer.begin_write_transaction("main", "bad\ttable", &columns, WriteMode::Replace);
        assert!(result.is_err());
    }
}
