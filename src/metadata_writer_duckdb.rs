//! DuckDB implementation of [`MetadataWriter`].
//!
//! A byte-compatible sibling of [`crate::metadata_writer_sqlite::SqliteMetadataWriter`]:
//! it writes the *exact same* DuckLake catalog tables (same names, same columns) so
//! that DuckDB's own `ducklake` extension — and this crate's
//! [`crate::DuckdbMetadataProvider`] — can read back what it writes.
//!
//! Scope is deliberately narrow (see `write-duckdb` feature): **legacy /
//! single-catalog only** — INSERT / REPLACE / CREATE TABLE (including zero-row).
//! No deletes, no upsert, no compaction, no partitioning, no multicatalog. The
//! erroring trait defaults for [`MetadataWriter::promote_column_type`] and
//! [`MetadataWriter::set_delete_file`] are inherited, and
//! [`MetadataWriter::catalog_id`] returns `None`.
//!
//! ## How this differs from the SQLite writer
//!
//! 1. **Synchronous driver.** The `duckdb` crate is rusqlite-like and sync, not
//!    sqlx/async, so there is no `block_on`: the trait's sync `fn`s run directly
//!    against a `duckdb::Connection`. A single connection guarded by a `Mutex`
//!    makes the writer `Send + Sync` and — since DuckLake file catalogs are
//!    single-writer — serializes commits, which is exactly the "write lock" the
//!    SQLite writer leans on. Each mutating method opens one DuckDB transaction
//!    (`BEGIN`/`COMMIT`, auto-rollback on drop) for atomicity.
//! 2. **Id allocation.** SQLite relies on `INTEGER PRIMARY KEY` rowid
//!    autoincrement for `schema_id` / `table_id` / `data_file_id` /
//!    `delete_file_id`; a plain DuckDB PK does *not* autoincrement, so those
//!    columns default from DuckDB **sequences** (`nextval(...)`). `snapshot_id`
//!    keeps SQLite's commit-ordered `MAX(snapshot_id)+1` allocation (assigned at
//!    the commit, never reserved at begin) and `column_id` keeps SQLite's
//!    monotonic counter row in `ducklake_metadata` bumped with `UPDATE ...
//!    RETURNING`. The column *layouts* are otherwise identical to the SQLite
//!    DDL; only the id-default mechanism and the SQL type spellings
//!    (BIGINT/VARCHAR/BOOLEAN/TIMESTAMP) change.
//! 3. **Ordering + conflict detection preserved.** The snapshot is allocated
//!    before the table reads, and a `Replace` commit aborts with
//!    [`crate::DuckLakeError::Conflict`] if any data file of the table has a
//!    `begin_snapshot`/`end_snapshot` newer than the base observed at begin.

use crate::Result;
use crate::error::{TypeChangeOperation, TypeChangeWriteMode};
use crate::metadata_writer::{
    ColumnDef, ColumnStat, CommitIds, DataFileInfo, MetadataWriter, WriteMode, WriteSetupResult,
    columns_differ, validate_name,
};
use crate::partition::PartitionTransform;
use duckdb::{Connection, OptionalExt, Transaction, params};
use std::sync::{Arc, Mutex, MutexGuard};

/// DuckLake catalog DDL for DuckDB.
///
/// Column-for-column identical to `SQL_CREATE_SCHEMA` in the SQLite writer; the
/// only changes are DuckDB spellings (BIGINT/VARCHAR/BOOLEAN/TIMESTAMP,
/// `DEFAULT true`) and the id-default mechanism. The `*_id_seq` sequences supply
/// the autoincrement that a plain DuckDB `PRIMARY KEY` does not; they are created
/// *before* the tables that reference them via `DEFAULT nextval(...)`. DuckDB
/// persists a sequence's value in the database file, so re-opening a catalog this
/// writer created continues handing out fresh, non-overlapping ids.
const SQL_CREATE_SCHEMA: &str = r#"
CREATE SEQUENCE IF NOT EXISTS ducklake_schema_id_seq START 1;
CREATE SEQUENCE IF NOT EXISTS ducklake_table_id_seq START 1;
CREATE SEQUENCE IF NOT EXISTS ducklake_data_file_id_seq START 1;
CREATE SEQUENCE IF NOT EXISTS ducklake_delete_file_id_seq START 1;
CREATE SEQUENCE IF NOT EXISTS ducklake_partition_id_seq START 1;

CREATE TABLE IF NOT EXISTS ducklake_metadata (
    key VARCHAR NOT NULL,
    value VARCHAR NOT NULL,
    scope VARCHAR
);

CREATE TABLE IF NOT EXISTS ducklake_snapshot (
    snapshot_id BIGINT PRIMARY KEY,
    snapshot_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    schema_version BIGINT NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS ducklake_schema_versions (
    begin_snapshot BIGINT NOT NULL,
    schema_version BIGINT NOT NULL,
    table_id BIGINT NOT NULL,
    UNIQUE (table_id, begin_snapshot)
);

CREATE TABLE IF NOT EXISTS ducklake_schema (
    schema_id BIGINT PRIMARY KEY DEFAULT nextval('ducklake_schema_id_seq'),
    schema_name VARCHAR NOT NULL,
    path VARCHAR NOT NULL DEFAULT '',
    path_is_relative BOOLEAN NOT NULL DEFAULT true,
    begin_snapshot BIGINT NOT NULL,
    end_snapshot BIGINT
);

CREATE TABLE IF NOT EXISTS ducklake_table (
    table_id BIGINT PRIMARY KEY DEFAULT nextval('ducklake_table_id_seq'),
    schema_id BIGINT NOT NULL,
    table_name VARCHAR NOT NULL,
    path VARCHAR NOT NULL DEFAULT '',
    path_is_relative BOOLEAN NOT NULL DEFAULT true,
    begin_snapshot BIGINT NOT NULL,
    end_snapshot BIGINT
);

-- Bare table (no PK, no NOT NULL), upstream's exact column order, so a column
-- can be versioned by [begin_snapshot, end_snapshot) and a type promotion can
-- write a second row sharing the same column_id. Mirrors the SQLite writer's
-- `ducklake_column`. The four `*default*` columns + `parent_column` are left
-- NULL (no nested-type / column-default writes here).
CREATE TABLE IF NOT EXISTS ducklake_column (
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
);

CREATE TABLE IF NOT EXISTS ducklake_data_file (
    data_file_id BIGINT PRIMARY KEY DEFAULT nextval('ducklake_data_file_id_seq'),
    table_id BIGINT NOT NULL,
    path VARCHAR NOT NULL,
    path_is_relative BOOLEAN NOT NULL DEFAULT true,
    file_size_bytes BIGINT NOT NULL,
    footer_size BIGINT,
    encryption_key VARCHAR,
    record_count BIGINT,
    row_id_start BIGINT,
    mapping_id BIGINT,
    begin_snapshot BIGINT NOT NULL,
    end_snapshot BIGINT,
    -- References ducklake_partition_info.partition_id; NULL when unpartitioned.
    partition_id BIGINT
);

CREATE TABLE IF NOT EXISTS ducklake_table_stats (
    table_id BIGINT PRIMARY KEY,
    record_count BIGINT NOT NULL DEFAULT 0,
    next_row_id BIGINT NOT NULL DEFAULT 0,
    file_size_bytes BIGINT NOT NULL DEFAULT 0
);

-- Per-file, per-column zone maps (DuckLake spec) — powers file pruning.
-- Column set mirrors the official extension and the other backends.
CREATE TABLE IF NOT EXISTS ducklake_file_column_stats (
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
);

-- Table-wide per-column roll-up (DuckLake spec) — feeds the optimizer.
CREATE TABLE IF NOT EXISTS ducklake_table_column_stats (
    table_id BIGINT NOT NULL,
    column_id BIGINT NOT NULL,
    contains_null BOOLEAN,
    contains_nan BOOLEAN,
    min_value VARCHAR,
    max_value VARCHAR,
    extra_stats VARCHAR
);

CREATE TABLE IF NOT EXISTS ducklake_delete_file (
    delete_file_id BIGINT PRIMARY KEY DEFAULT nextval('ducklake_delete_file_id_seq'),
    data_file_id BIGINT NOT NULL,
    table_id BIGINT NOT NULL,
    path VARCHAR NOT NULL,
    path_is_relative BOOLEAN NOT NULL DEFAULT true,
    file_size_bytes BIGINT NOT NULL,
    footer_size BIGINT,
    encryption_key VARCHAR,
    delete_count BIGINT,
    begin_snapshot BIGINT NOT NULL,
    end_snapshot BIGINT
);

CREATE TABLE IF NOT EXISTS ducklake_files_scheduled_for_deletion (
    data_file_id BIGINT NOT NULL,
    path VARCHAR NOT NULL,
    path_is_relative BOOLEAN NOT NULL DEFAULT true,
    schedule_start TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Partition spec generations (DuckLake spec); end_snapshot NULL == active.
CREATE TABLE IF NOT EXISTS ducklake_partition_info (
    partition_id BIGINT NOT NULL,
    table_id BIGINT NOT NULL,
    begin_snapshot BIGINT NOT NULL,
    end_snapshot BIGINT
);

-- Partition-key columns for a spec (DuckLake spec), ordered by partition_key_index.
CREATE TABLE IF NOT EXISTS ducklake_partition_column (
    partition_id BIGINT NOT NULL,
    table_id BIGINT NOT NULL,
    partition_key_index BIGINT NOT NULL,
    column_id BIGINT NOT NULL,
    transform VARCHAR NOT NULL
);

-- Per-file partition values (DuckLake spec): the value every row in the file
-- shares for a partition key, DuckDB-canonical VARCHAR (NULL is legal).
CREATE TABLE IF NOT EXISTS ducklake_file_partition_value (
    data_file_id BIGINT NOT NULL,
    table_id BIGINT NOT NULL,
    partition_key_index BIGINT NOT NULL,
    partition_value VARCHAR
);
"#;

/// DuckDB-based metadata writer for DuckLake catalogs.
///
/// Holds a single read-write `duckdb::Connection` behind a `Mutex` (cloneable via
/// `Arc`, sharing the one connection). See the module docs for the differences
/// from the SQLite writer.
#[derive(Debug, Clone)]
pub struct DuckdbMetadataWriter {
    conn: Arc<Mutex<Connection>>,
    /// Path to the catalog database, retained for logging/debugging.
    #[allow(dead_code)]
    catalog_path: String,
}

impl DuckdbMetadataWriter {
    /// Open (creating if absent) a DuckDB catalog file for writing.
    pub fn new(path: impl Into<String>) -> Result<Self> {
        let catalog_path = path.into();
        let conn = Connection::open(&catalog_path)?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            catalog_path,
        })
    }

    /// Open the catalog and initialize the DuckLake schema tables.
    pub fn new_with_init(path: impl Into<String>) -> Result<Self> {
        let writer = Self::new(path)?;
        writer.initialize_schema()?;
        Ok(writer)
    }

    /// Lock the shared connection. Panics only if a previous holder panicked
    /// mid-write (poisoned mutex), which cannot leave partial SQL committed
    /// because every mutating method wraps its work in a transaction.
    fn connection(&self) -> MutexGuard<'_, Connection> {
        self.conn.lock().expect("DuckDB connection mutex poisoned")
    }
}

/// Atomically reserve `n` consecutive ids from a monotonic counter row in
/// `ducklake_metadata` (seeded by `initialize_schema`), returning the LAST id of
/// the block — the reserved ids are `last - n + 1 ..= last`. Mirrors the SQLite
/// writer's `reserve_ids`; used for `column_id`, which is reserved at begin and
/// baked into the staged parquet's `field_id` metadata. `value` is stored as
/// text and cast through BIGINT so the `UPDATE ... RETURNING` is exact.
fn reserve_ids(tx: &Transaction<'_>, key: &str, n: i64) -> Result<i64> {
    let last: i64 = tx.query_row(
        "UPDATE ducklake_metadata
         SET value = CAST(CAST(value AS BIGINT) + ? AS VARCHAR)
         WHERE key = ? AND scope IS NULL
         RETURNING CAST(value AS BIGINT)",
        params![n, key],
        |row| row.get(0),
    )?;
    Ok(last)
}

/// Insert the next `ducklake_snapshot` row in commit order, carrying
/// `schema_version` forward (the pure-data-write default), and return
/// `(snapshot_id, schema_version)`.
///
/// Mirrors the SQLite writer's `insert_snapshot` — `MAX(snapshot_id)+1` assigned
/// at the commit, never reserved at begin — but split into a read then an insert
/// (DuckDB's single-writer connection is held under the `Mutex`, so no
/// concurrent writer can slip between them; the two-statement form avoids relying
/// on `INSERT ... SELECT ... RETURNING`). A DDL commit follows this with
/// [`bump_schema_version`].
fn insert_snapshot(tx: &Transaction<'_>) -> Result<(i64, i64)> {
    let (snapshot_id, schema_version): (i64, i64) = tx.query_row(
        "SELECT COALESCE(MAX(snapshot_id), 0) + 1, COALESCE(MAX(schema_version), 0)
         FROM ducklake_snapshot",
        [],
        |row| Ok((row.get(0)?, row.get(1)?)),
    )?;
    tx.execute(
        "INSERT INTO ducklake_snapshot (snapshot_id, snapshot_time, schema_version)
         VALUES (?, CURRENT_TIMESTAMP, ?)",
        params![snapshot_id, schema_version],
    )?;
    Ok((snapshot_id, schema_version))
}

/// Bump the per-catalog monotonic `schema_version` on a DDL snapshot to
/// `prev_max + 1` (max over the OTHER snapshots, so re-running is stable) and
/// return the new value. Mirrors the SQLite writer's `bump_schema_version`.
fn bump_schema_version(tx: &Transaction<'_>, snapshot_id: i64) -> Result<i64> {
    let prev_max: i64 = tx.query_row(
        "SELECT COALESCE(MAX(schema_version), 0) FROM ducklake_snapshot WHERE snapshot_id <> ?",
        params![snapshot_id],
        |row| row.get(0),
    )?;
    let new_version = prev_max + 1;
    tx.execute(
        "UPDATE ducklake_snapshot SET schema_version = ? WHERE snapshot_id = ?",
        params![new_version, snapshot_id],
    )?;
    Ok(new_version)
}

/// Record a `ducklake_schema_versions` ledger row for a DDL that leaves the table
/// live (create, column add/remove/reorder). Mirrors the SQLite writer.
fn record_schema_version(
    tx: &Transaction<'_>,
    snapshot_id: i64,
    schema_version: i64,
    table_id: i64,
) -> Result<()> {
    tx.execute(
        "INSERT INTO ducklake_schema_versions (begin_snapshot, schema_version, table_id)
         VALUES (?, ?, ?)",
        params![snapshot_id, schema_version, table_id],
    )?;
    Ok(())
}

/// Seed the `ducklake_table_stats` row for a brand-new table if it is missing.
///
/// Replaces the SQLite writer's `INSERT OR IGNORE` with an explicit
/// exists-check, avoiding any dependency on DuckDB upsert syntax while producing
/// the identical result (a single zeroed stats row per table).
fn seed_stats_if_missing(tx: &Transaction<'_>, table_id: i64) -> Result<()> {
    let exists: Option<i64> = tx
        .query_row(
            "SELECT 1 FROM ducklake_table_stats WHERE table_id = ?",
            params![table_id],
            |row| row.get(0),
        )
        .optional()?;
    if exists.is_none() {
        tx.execute(
            "INSERT INTO ducklake_table_stats
                 (table_id, record_count, next_row_id, file_size_bytes)
             VALUES (?, 0, 0, 0)",
            params![table_id],
        )?;
    }
    Ok(())
}

/// Optimistic-concurrency check for a `Replace` commit (mirrors the SQLite
/// writer). If any data file of the table has `begin_snapshot` or `end_snapshot`
/// newer than `base_snapshot` (the head observed when this write began), another
/// writer published a newer generation in the meantime, so this `Replace` aborts
/// with [`crate::DuckLakeError::Conflict`] rather than clobbering it.
fn detect_replace_conflict(tx: &Transaction<'_>, table_id: i64, base_snapshot: i64) -> Result<()> {
    let conflicts: i64 = tx.query_row(
        "SELECT COUNT(*) FROM ducklake_data_file
         WHERE table_id = ? AND (begin_snapshot > ? OR end_snapshot > ?)",
        params![table_id, base_snapshot, base_snapshot],
        |row| row.get(0),
    )?;
    if conflicts > 0 {
        return Err(crate::DuckLakeError::Conflict(format!(
            "Replace on table {table_id} conflicts with a concurrent write committed since \
             snapshot {base_snapshot}; aborting (retry the write against the new generation)"
        )));
    }
    Ok(())
}

/// Retire the prior generation's still-visible data files at `snapshot_id` and
/// zero the visible stat totals. The `begin_snapshot < snapshot_id` guard spares
/// files registered for *this* snapshot. `next_row_id` is left untouched (rowids
/// stay monotonic across the table's lifetime). Mirrors the SQLite writer.
fn retire_prior_generation(tx: &Transaction<'_>, table_id: i64, snapshot_id: i64) -> Result<()> {
    tx.execute(
        "UPDATE ducklake_data_file SET end_snapshot = ?
         WHERE table_id = ? AND end_snapshot IS NULL AND begin_snapshot < ?",
        params![snapshot_id, table_id, snapshot_id],
    )?;
    tx.execute(
        "UPDATE ducklake_table_stats SET record_count = 0, file_size_bytes = 0 WHERE table_id = ?",
        params![table_id],
    )?;
    Ok(())
}

/// The atomic commit point for a single-catalog write. Inserts the deferred
/// `ducklake_snapshot` row, finalizes the column generation surgically (stable
/// `column_id`s), classifies the commit as DDL vs pure data write to bump/carry
/// `schema_version`, and — for `Replace` — fences on the base snapshot and
/// retires the prior data generation. A faithful port of the SQLite writer's
/// `finalize_snapshot`; the only structural change is reading the current
/// columns into an owned `Vec` (dropping the prepared statement) before issuing
/// further writes on the same connection.
/// Persist the harvested per-column stats for a just-registered data file
/// (per-file zone maps). See the SQLite writer's equivalent for the rationale.
fn insert_file_column_stats(
    tx: &Transaction<'_>,
    table_id: i64,
    data_file_id: i64,
    column_stats: &[ColumnStat],
) -> Result<()> {
    for stat in column_stats {
        tx.execute(
            "INSERT INTO ducklake_file_column_stats
                 (data_file_id, table_id, column_id, column_size_bytes,
                  value_count, null_count, min_value, max_value, contains_nan, extra_stats)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)",
            params![
                data_file_id,
                table_id,
                stat.column_id,
                stat.column_size_bytes,
                stat.value_count,
                stat.null_count,
                stat.min_value.as_deref(),
                stat.max_value.as_deref(),
                stat.contains_nan,
            ],
        )?;
    }
    Ok(())
}

/// Persist a partitioned data file's partition metadata within the commit
/// transaction: set `ducklake_data_file.partition_id` and insert one
/// `ducklake_file_partition_value` row per partition key. A no-op for an
/// unpartitioned file.
fn insert_partition_metadata(
    tx: &Transaction<'_>,
    table_id: i64,
    data_file_id: i64,
    file: &DataFileInfo,
) -> Result<()> {
    if let Some(partition_id) = file.partition_id {
        tx.execute(
            "UPDATE ducklake_data_file SET partition_id = ? WHERE data_file_id = ?",
            params![partition_id, data_file_id],
        )?;
    }
    for (key_index, value) in &file.partition_values {
        tx.execute(
            "INSERT INTO ducklake_file_partition_value
                 (data_file_id, table_id, partition_key_index, partition_value)
             VALUES (?, ?, ?, ?)",
            params![data_file_id, table_id, i64::from(*key_index), value.as_deref()],
        )?;
    }
    Ok(())
}

/// Recompute `ducklake_table_column_stats` from the table's live files and
/// replace the stored rows. See the SQLite writer's equivalent for the rationale.
fn recompute_table_column_stats(
    tx: &Transaction<'_>,
    table_id: i64,
    columns: &[ColumnDef],
    column_ids: &[i64],
) -> Result<()> {
    use crate::stats_encode::{FileColumnStat, aggregate_global_column_stats};

    let live_file_count: i64 = tx.query_row(
        "SELECT COUNT(*) FROM ducklake_data_file WHERE table_id = ? AND end_snapshot IS NULL",
        params![table_id],
        |row| row.get(0),
    )?;

    // Collect first so the prepared statement's borrow of `tx` is released
    // before we issue the DELETE/INSERT writes below.
    let per_file: Vec<FileColumnStat> = {
        let mut stmt = tx.prepare(
            "SELECT s.column_id, s.min_value, s.max_value, s.null_count, s.contains_nan
             FROM ducklake_file_column_stats s
             JOIN ducklake_data_file d ON d.data_file_id = s.data_file_id
             WHERE d.table_id = ? AND d.end_snapshot IS NULL",
        )?;
        let mapped = stmt.query_map(params![table_id], |row| {
            Ok(FileColumnStat {
                column_id: row.get::<_, i64>(0)?,
                min_value: row.get::<_, Option<String>>(1)?,
                max_value: row.get::<_, Option<String>>(2)?,
                null_count: row.get::<_, Option<i64>>(3)?,
                contains_nan: row.get::<_, Option<bool>>(4)?,
            })
        })?;
        mapped.collect::<duckdb::Result<Vec<_>>>()?
    };

    let numeric_of = |column_id: i64| -> bool {
        column_ids
            .iter()
            .position(|id| *id == column_id)
            .and_then(|i| columns.get(i))
            .map(|c| crate::stats_encode::is_numeric_ducklake_type(c.ducklake_type()))
            .unwrap_or(false)
    };
    let globals = aggregate_global_column_stats(&per_file, live_file_count, numeric_of);

    tx.execute(
        "DELETE FROM ducklake_table_column_stats WHERE table_id = ?",
        params![table_id],
    )?;
    for g in globals {
        tx.execute(
            "INSERT INTO ducklake_table_column_stats
                 (table_id, column_id, contains_null, contains_nan, min_value, max_value, extra_stats)
             VALUES (?, ?, ?, ?, ?, ?, NULL)",
            params![
                table_id,
                g.column_id,
                g.contains_null,
                g.contains_nan,
                g.min_value,
                g.max_value,
            ],
        )?;
    }
    Ok(())
}

fn finalize_snapshot(
    tx: &Transaction<'_>,
    table_id: i64,
    columns: &[ColumnDef],
    column_ids: &[i64],
    mode: WriteMode,
    base_snapshot: i64,
) -> Result<i64> {
    use std::collections::{HashMap, HashSet};

    // Allocate the snapshot FIRST (carrying schema_version forward); corrected to
    // a DDL bump below once the commit is classified.
    let (snapshot_id, mut schema_version) = insert_snapshot(tx)?;

    // The table's live columns ordered by column_order. Collected into an owned
    // Vec so the prepared statement is dropped before we mutate the same
    // connection. An empty set means a brand-new table (its creating write is DDL).
    let current: Vec<(String, String, i64, bool)> = {
        let mut stmt = tx.prepare(
            "SELECT column_name, column_type, column_order, nulls_allowed
             FROM ducklake_column
             WHERE table_id = ? AND end_snapshot IS NULL
             ORDER BY column_order",
        )?;
        let mapped = stmt.query_map(params![table_id], |row| {
            let name: String = row.get(0)?;
            let ty: String = row.get(1)?;
            let order: i64 = row.get(2)?;
            let nullable: Option<bool> = row.get(3)?;
            Ok((name, ty, order, nullable.unwrap_or(true)))
        })?;
        mapped.collect::<std::result::Result<Vec<_>, duckdb::Error>>()?
    };

    let existing: Vec<(String, String, bool)> = current
        .iter()
        .map(|(name, ty, _order, nullable)| (name.clone(), ty.clone(), *nullable))
        .collect();
    let is_ddl = existing.is_empty() || columns_differ(&existing, columns);
    if is_ddl {
        // A DDL commit bumps schema_version (the insert only carried it forward).
        schema_version = bump_schema_version(tx, snapshot_id)?;
    }

    // Reconcile the column generation SURGICALLY so each column keeps a stable
    // column_id (== parquet field_id): end only removed columns, insert only new
    // ones, and leave unchanged columns (and their ids) in place.
    let new_names: HashSet<&str> = columns.iter().map(|c| c.name()).collect();
    let mut current_by_name: HashMap<String, (i64, bool)> = HashMap::new();
    for (name, _ty, order, nullable) in &current {
        if !new_names.contains(name.as_str()) {
            tx.execute(
                "UPDATE ducklake_column SET end_snapshot = ?
                 WHERE table_id = ? AND column_name = ? AND end_snapshot IS NULL",
                params![snapshot_id, table_id, name.as_str()],
            )?;
        }
        current_by_name.insert(name.clone(), (*order, *nullable));
    }

    for (order, (col, column_id)) in columns.iter().zip(column_ids.iter()).enumerate() {
        match current_by_name.get(col.name()) {
            // Existing column kept: its id stays stable. Sync order/nullability
            // only if they changed (type changes are rejected at begin).
            Some(&(cur_order, cur_nullable)) => {
                if cur_order != order as i64 || cur_nullable != col.is_nullable() {
                    tx.execute(
                        "UPDATE ducklake_column SET column_order = ?, nulls_allowed = ?
                         WHERE table_id = ? AND column_name = ? AND end_snapshot IS NULL",
                        params![order as i64, col.is_nullable(), table_id, col.name()],
                    )?;
                }
            },
            // Newly added column: insert it with its reserved id.
            None => {
                tx.execute(
                    "INSERT INTO ducklake_column
                         (column_id, table_id, column_name, column_type, column_order, nulls_allowed, begin_snapshot)
                     VALUES (?, ?, ?, ?, ?, ?, ?)",
                    params![
                        column_id,
                        table_id,
                        col.name(),
                        col.ducklake_type(),
                        order as i64,
                        col.is_nullable(),
                        snapshot_id
                    ],
                )?;
            },
        }
    }

    if mode == WriteMode::Replace {
        // Abort if a concurrent writer published a newer generation since begin.
        detect_replace_conflict(tx, table_id, base_snapshot)?;
        // Seed the stats row (first write to a brand-new table) so retire's
        // zero-update has a row, then retire the prior data generation.
        seed_stats_if_missing(tx, table_id)?;
        retire_prior_generation(tx, table_id, snapshot_id)?;
    }

    // Record the schema-change ledger row for a DDL commit. A pure data write
    // carries schema_version forward and writes no row.
    if is_ddl {
        record_schema_version(tx, snapshot_id, schema_version, table_id)?;
    }
    Ok(snapshot_id)
}

impl MetadataWriter for DuckdbMetadataWriter {
    fn create_snapshot(&self) -> Result<i64> {
        let mut conn = self.connection();
        let tx = conn.transaction()?;
        // A bare snapshot carries no schema change of its own → carry
        // schema_version forward (no DDL bump, no ledger row).
        let (snapshot_id, _schema_version) = insert_snapshot(&tx)?;
        tx.commit()?;
        Ok(snapshot_id)
    }

    fn get_or_create_schema(
        &self,
        name: &str,
        path: Option<&str>,
        snapshot_id: i64,
    ) -> Result<(i64, bool)> {
        validate_name(name, "Schema")?;
        let mut conn = self.connection();
        let tx = conn.transaction()?;

        let existing: Option<i64> = tx
            .query_row(
                "SELECT schema_id FROM ducklake_schema
                 WHERE schema_name = ? AND end_snapshot IS NULL",
                params![name],
                |row| row.get(0),
            )
            .optional()?;

        if let Some(schema_id) = existing {
            tx.commit()?;
            return Ok((schema_id, false));
        }

        let schema_path = path.unwrap_or(name);
        let schema_id: i64 = tx.query_row(
            "INSERT INTO ducklake_schema (schema_name, path, path_is_relative, begin_snapshot)
             VALUES (?, ?, true, ?) RETURNING schema_id",
            params![name, schema_path, snapshot_id],
            |row| row.get(0),
        )?;
        tx.commit()?;
        Ok((schema_id, true))
    }

    fn get_or_create_table(
        &self,
        schema_id: i64,
        name: &str,
        path: Option<&str>,
        snapshot_id: i64,
    ) -> Result<(i64, bool)> {
        validate_name(name, "Table")?;
        let mut conn = self.connection();
        let tx = conn.transaction()?;

        let existing: Option<i64> = tx
            .query_row(
                "SELECT table_id FROM ducklake_table
                 WHERE schema_id = ? AND table_name = ? AND end_snapshot IS NULL",
                params![schema_id, name],
                |row| row.get(0),
            )
            .optional()?;

        if let Some(table_id) = existing {
            tx.commit()?;
            return Ok((table_id, false));
        }

        let table_path = path.unwrap_or(name);
        let table_id: i64 = tx.query_row(
            "INSERT INTO ducklake_table (schema_id, table_name, path, path_is_relative, begin_snapshot)
             VALUES (?, ?, ?, true, ?) RETURNING table_id",
            params![schema_id, name, table_path, snapshot_id],
            |row| row.get(0),
        )?;
        tx.commit()?;
        Ok((table_id, true))
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
        let mut conn = self.connection();
        let tx = conn.transaction()?;

        tx.execute(
            "UPDATE ducklake_column SET end_snapshot = ?
             WHERE table_id = ? AND end_snapshot IS NULL",
            params![snapshot_id, table_id],
        )?;

        // Reserve a contiguous column_id block from the monotonic counter and
        // insert with explicit ids, keeping the allocator authoritative.
        let n = columns.len() as i64;
        let last_column_id = reserve_ids(&tx, "next_column_id", n)?;
        let first_column_id = last_column_id - n + 1;
        let mut column_ids = Vec::with_capacity(columns.len());
        for (order, col) in columns.iter().enumerate() {
            let column_id = first_column_id + order as i64;
            tx.execute(
                "INSERT INTO ducklake_column (column_id, table_id, column_name, column_type, column_order, nulls_allowed, begin_snapshot)
                 VALUES (?, ?, ?, ?, ?, ?, ?)",
                params![
                    column_id,
                    table_id,
                    col.name(),
                    col.ducklake_type(),
                    order as i64,
                    col.is_nullable(),
                    snapshot_id
                ],
            )?;
            column_ids.push(column_id);
        }

        tx.commit()?;
        Ok(column_ids)
    }

    fn register_data_file(
        &self,
        table_id: i64,
        // The schema/table were created at begin, so the names are unused here;
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
        let mut conn = self.connection();
        let tx = conn.transaction()?;

        // Single atomic commit: insert the deferred snapshot row + finalize the
        // column generation + retire the prior generation (Replace), then
        // register this file and advance the monotonic row-lineage counter.
        let snapshot_id =
            finalize_snapshot(&tx, table_id, columns, column_ids, mode, base_snapshot)?;

        // Partition-spec fence: this file must be consistent with the table's live
        // partition generation at commit time (both directions — see
        // enforce_partition_fence). The scalar subquery yields NULL (→ None) when the
        // table is unpartitioned; the tx rolls back on a Conflict.
        let live_partition_id: Option<i64> = tx.query_row(
            "SELECT (SELECT partition_id FROM ducklake_partition_info
                     WHERE table_id = ? AND end_snapshot IS NULL LIMIT 1)",
            params![table_id],
            |row| row.get::<_, Option<i64>>(0),
        )?;
        crate::metadata_writer::enforce_partition_fence(table_id, live_partition_id, file)?;

        // Seed the stats row for the Append path (Replace already seeded it in
        // finalize_snapshot); a no-op if it exists.
        seed_stats_if_missing(&tx, table_id)?;

        let row_id_start: i64 = tx.query_row(
            "SELECT next_row_id FROM ducklake_table_stats WHERE table_id = ?",
            params![table_id],
            |row| row.get(0),
        )?;

        // RETURNING gives us the sequence-allocated data_file_id to tie the
        // per-column stats rows to, in the same transaction.
        let data_file_id: i64 = tx.query_row(
            "INSERT INTO ducklake_data_file
                 (table_id, path, path_is_relative, file_size_bytes,
                  footer_size, record_count, row_id_start, begin_snapshot)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?) RETURNING data_file_id",
            params![
                table_id,
                file.path.as_str(),
                file.path_is_relative,
                file.file_size_bytes,
                file.footer_size,
                file.record_count,
                row_id_start,
                snapshot_id
            ],
            |row| row.get(0),
        )?;

        insert_file_column_stats(&tx, table_id, data_file_id, &file.column_stats)?;
        insert_partition_metadata(&tx, table_id, data_file_id, file)?;
        recompute_table_column_stats(&tx, table_id, columns, column_ids)?;

        // Advance the counter and accumulate stats. next_row_id monotonically
        // increases over the table's lifetime — rowids are never reused.
        tx.execute(
            "UPDATE ducklake_table_stats
             SET next_row_id     = next_row_id + ?,
                 record_count    = record_count + ?,
                 file_size_bytes = file_size_bytes + ?
             WHERE table_id = ?",
            params![file.record_count, file.record_count, file.file_size_bytes, table_id],
        )?;

        let schema_id: i64 = tx.query_row(
            "SELECT schema_id FROM ducklake_table WHERE table_id = ?",
            params![table_id],
            |row| row.get(0),
        )?;

        tx.commit()?;
        Ok(CommitIds {
            snapshot_id,
            schema_id,
            table_id,
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
        let mut conn = self.connection();
        let tx = conn.transaction()?;
        // One atomic snapshot for all N partition files.
        let snapshot_id =
            finalize_snapshot(&tx, table_id, columns, column_ids, mode, base_snapshot)?;
        // Partition-spec fence (both directions, every file): each file must be
        // consistent with the table's live partition generation at commit time. The
        // scalar subquery yields NULL (→ None) when unpartitioned; the tx rolls back
        // on a Conflict.
        let live_partition_id: Option<i64> = tx.query_row(
            "SELECT (SELECT partition_id FROM ducklake_partition_info
                     WHERE table_id = ? AND end_snapshot IS NULL LIMIT 1)",
            params![table_id],
            |row| row.get::<_, Option<i64>>(0),
        )?;
        for file in files {
            crate::metadata_writer::enforce_partition_fence(table_id, live_partition_id, file)?;
        }
        seed_stats_if_missing(&tx, table_id)?;
        let mut next_row_id: i64 = tx.query_row(
            "SELECT next_row_id FROM ducklake_table_stats WHERE table_id = ?",
            params![table_id],
            |row| row.get(0),
        )?;
        let mut total_records: i64 = 0;
        let mut total_bytes: i64 = 0;
        for file in files {
            let data_file_id: i64 = tx.query_row(
                "INSERT INTO ducklake_data_file
                     (table_id, path, path_is_relative, file_size_bytes,
                      footer_size, record_count, row_id_start, begin_snapshot)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?) RETURNING data_file_id",
                params![
                    table_id,
                    file.path.as_str(),
                    file.path_is_relative,
                    file.file_size_bytes,
                    file.footer_size,
                    file.record_count,
                    next_row_id,
                    snapshot_id
                ],
                |row| row.get(0),
            )?;
            insert_file_column_stats(&tx, table_id, data_file_id, &file.column_stats)?;
            insert_partition_metadata(&tx, table_id, data_file_id, file)?;
            next_row_id += file.record_count;
            total_records += file.record_count;
            total_bytes += file.file_size_bytes;
        }
        recompute_table_column_stats(&tx, table_id, columns, column_ids)?;
        tx.execute(
            "UPDATE ducklake_table_stats
             SET next_row_id     = next_row_id + ?,
                 record_count    = record_count + ?,
                 file_size_bytes = file_size_bytes + ?
             WHERE table_id = ?",
            params![total_records, total_records, total_bytes, table_id],
        )?;
        let schema_id: i64 = tx.query_row(
            "SELECT schema_id FROM ducklake_table WHERE table_id = ?",
            params![table_id],
            |row| row.get(0),
        )?;
        tx.commit()?;
        Ok(CommitIds {
            snapshot_id,
            schema_id,
            table_id,
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
        let mut conn = self.connection();
        let tx = conn.transaction()?;
        let partition_id: i64 =
            tx.query_row("SELECT nextval('ducklake_partition_id_seq')", [], |row| {
                row.get(0)
            })?;
        let (new_snapshot, _carried) = insert_snapshot(&tx)?;

        // Resolve each partition-key column NAME to its live column_id.
        let mut column_ids: Vec<i64> = Vec::with_capacity(columns.len());
        for (name, _transform) in columns {
            let column_id: i64 = tx
                .query_row(
                    "SELECT column_id FROM ducklake_column
                     WHERE table_id = ? AND column_name = ? AND end_snapshot IS NULL",
                    params![table_id, name.as_str()],
                    |row| row.get(0),
                )
                .optional()?
                .ok_or_else(|| {
                    crate::DuckLakeError::InvalidConfig(format!(
                        "set_partition_spec: no live column '{name}' in table {table_id}"
                    ))
                })?;
            column_ids.push(column_id);
        }

        tx.execute(
            "UPDATE ducklake_partition_info SET end_snapshot = ?
             WHERE table_id = ? AND end_snapshot IS NULL",
            params![new_snapshot, table_id],
        )?;
        tx.execute(
            "INSERT INTO ducklake_partition_info
                 (partition_id, table_id, begin_snapshot, end_snapshot)
             VALUES (?, ?, ?, NULL)",
            params![partition_id, table_id, new_snapshot],
        )?;
        for (key_index, column_id) in column_ids.iter().enumerate() {
            tx.execute(
                "INSERT INTO ducklake_partition_column
                     (partition_id, table_id, partition_key_index, column_id, transform)
                 VALUES (?, ?, ?, ?, ?)",
                params![
                    partition_id,
                    table_id,
                    key_index as i64,
                    *column_id,
                    columns[key_index].1.to_catalog_string()
                ],
            )?;
        }

        let new_schema_version = bump_schema_version(&tx, new_snapshot)?;
        record_schema_version(&tx, new_snapshot, new_schema_version, table_id)?;
        tx.commit()?;
        Ok(new_snapshot)
    }

    fn reset_partition_spec(&self, table_id: i64) -> Result<i64> {
        let mut conn = self.connection();
        let tx = conn.transaction()?;
        let (new_snapshot, _carried) = insert_snapshot(&tx)?;
        let ended = tx.execute(
            "UPDATE ducklake_partition_info SET end_snapshot = ?
             WHERE table_id = ? AND end_snapshot IS NULL",
            params![new_snapshot, table_id],
        )?;
        if ended == 0 {
            // Nothing to reset: roll back the snapshot and report the head.
            drop(tx);
            let head: i64 = conn.query_row(
                "SELECT COALESCE(MAX(snapshot_id), 0) FROM ducklake_snapshot",
                [],
                |row| row.get(0),
            )?;
            return Ok(head);
        }
        let new_schema_version = bump_schema_version(&tx, new_snapshot)?;
        record_schema_version(&tx, new_snapshot, new_schema_version, table_id)?;
        tx.commit()?;
        Ok(new_snapshot)
    }

    fn publish_snapshot(
        &self,
        table_id: i64,
        // The schema/table were created at begin; names unused (trait parity).
        _schema_name: &str,
        _table_name: &str,
        _snapshot_id: i64,
        mode: WriteMode,
        base_snapshot: i64,
        columns: &[ColumnDef],
        column_ids: &[i64],
    ) -> Result<CommitIds> {
        // Fileless commit point (CREATE TABLE, zero-row Replace). Single-catalog
        // DuckDB defers the snapshot-row insert out of begin_write_transaction,
        // so this is NOT the trait's no-op default: it inserts the deferred
        // snapshot row + column generation and, for Replace, retires the prior
        // generation — making the new head visible atomically.
        let mut conn = self.connection();
        let tx = conn.transaction()?;
        let snapshot_id =
            finalize_snapshot(&tx, table_id, columns, column_ids, mode, base_snapshot)?;
        let schema_id: i64 = tx.query_row(
            "SELECT schema_id FROM ducklake_table WHERE table_id = ?",
            params![table_id],
            |row| row.get(0),
        )?;
        tx.commit()?;
        Ok(CommitIds {
            snapshot_id,
            schema_id,
            table_id,
        })
    }

    fn end_table_files(&self, table_id: i64, snapshot_id: i64) -> Result<u64> {
        // Used by WriteMode::Replace. End-snapshotting every visible file drops
        // the table's currently-visible row count and byte total to zero.
        // next_row_id is deliberately NOT reset (rowids stay monotonic).
        let mut conn = self.connection();
        let tx = conn.transaction()?;

        let rows_affected = tx.execute(
            "UPDATE ducklake_data_file SET end_snapshot = ?
             WHERE table_id = ? AND end_snapshot IS NULL",
            params![snapshot_id, table_id],
        )? as u64;

        tx.execute(
            "UPDATE ducklake_table_stats
             SET record_count = 0, file_size_bytes = 0
             WHERE table_id = ?",
            params![table_id],
        )?;

        tx.commit()?;
        Ok(rows_affected)
    }

    fn get_data_path(&self) -> Result<String> {
        let conn = self.connection();
        let row: Option<String> = conn
            .query_row(
                "SELECT value FROM ducklake_metadata WHERE key = ? AND scope IS NULL",
                params!["data_path"],
                |row| row.get(0),
            )
            .optional()?;
        match row {
            Some(path) => Ok(path),
            None => Err(crate::error::DuckLakeError::InvalidConfig(
                "Missing required catalog metadata: 'data_path' not configured.".to_string(),
            )),
        }
    }

    fn set_data_path(&self, path: &str) -> Result<()> {
        let mut conn = self.connection();
        let tx = conn.transaction()?;
        tx.execute(
            "DELETE FROM ducklake_metadata WHERE key = 'data_path' AND scope IS NULL",
            [],
        )?;
        tx.execute(
            "INSERT INTO ducklake_metadata (key, value, scope) VALUES ('data_path', ?, NULL)",
            params![path],
        )?;
        tx.commit()?;
        Ok(())
    }

    fn initialize_schema(&self) -> Result<()> {
        let conn = self.connection();
        conn.execute_batch(SQL_CREATE_SCHEMA)?;
        // Upgrade a pre-existing catalog to carry ducklake_data_file.partition_id
        // (CREATE TABLE IF NOT EXISTS never alters an existing table). DuckDB supports
        // ADD COLUMN IF NOT EXISTS, so this is idempotent and lossless (NULL means
        // "not partitioned").
        conn.execute_batch(
            "ALTER TABLE ducklake_data_file ADD COLUMN IF NOT EXISTS partition_id BIGINT",
        )?;
        // Seed the monotonic column_id allocator (snapshot_id uses MAX+1, and
        // schema/table/data_file/delete_file ids use sequences, so none of those
        // need a counter). Idempotent on re-open, and seeded from the current MAX
        // so a pre-existing catalog continues without reusing ids.
        conn.execute_batch(
            "INSERT INTO ducklake_metadata (key, value, scope)
             SELECT 'next_column_id',
                    CAST(COALESCE((SELECT MAX(column_id) FROM ducklake_column), 0) AS VARCHAR),
                    NULL
             WHERE NOT EXISTS (
                 SELECT 1 FROM ducklake_metadata WHERE key = 'next_column_id' AND scope IS NULL
             )",
        )?;
        Ok(())
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
        let mut conn = self.connection();
        let tx = conn.transaction()?;

        // Reserve the column ids first. These match the staged parquet field ids.
        let n = columns.len() as i64;
        let last_column_id = reserve_ids(&tx, "next_column_id", n)?;
        // Freshly reserved ids. Only a genuinely-new column consumes one below; an
        // existing column keeps its current id, so some may go unused (harmless
        // monotonic-counter gaps).
        let fresh_ids: Vec<i64> = ((last_column_id - n + 1)..=last_column_id).collect();

        // The catalog head this write is based on; a Replace commit aborts if
        // another writer published a newer generation of the table past it.
        let base_snapshot_id: i64 = tx.query_row(
            "SELECT COALESCE(MAX(snapshot_id), 0) FROM ducklake_snapshot",
            [],
            |row| row.get(0),
        )?;

        // Tentative id for WriteSetupResult; the real one is assigned at the
        // commit (finalize_snapshot), so it may differ under concurrency.
        let snapshot_id: i64 = base_snapshot_id + 1;

        let schema_id: i64 = {
            let existing: Option<i64> = tx
                .query_row(
                    "SELECT schema_id FROM ducklake_schema
                     WHERE schema_name = ? AND end_snapshot IS NULL",
                    params![schema_name],
                    |row| row.get(0),
                )
                .optional()?;
            match existing {
                Some(id) => id,
                None => tx.query_row(
                    "INSERT INTO ducklake_schema (schema_name, path, path_is_relative, begin_snapshot)
                     VALUES (?, ?, true, ?) RETURNING schema_id",
                    params![schema_name, schema_name, snapshot_id],
                    |row| row.get(0),
                )?,
            }
        };

        let table_id: i64 = {
            let existing: Option<i64> = tx
                .query_row(
                    "SELECT table_id FROM ducklake_table
                     WHERE schema_id = ? AND table_name = ? AND end_snapshot IS NULL",
                    params![schema_id, table_name],
                    |row| row.get(0),
                )
                .optional()?;
            match existing {
                Some(id) => id,
                None => tx.query_row(
                    "INSERT INTO ducklake_table (schema_id, table_name, path, path_is_relative, begin_snapshot)
                     VALUES (?, ?, ?, true, ?) RETURNING table_id",
                    params![schema_id, table_name, table_name, snapshot_id],
                    |row| row.get(0),
                )?,
            }
        };

        // Existing columns to (a) check schema compatibility and (b) REUSE each
        // column's id (column_id == parquet field_id == a column's stable
        // identity). Collected into owned Vec so the statement drops before the
        // commit that follows.
        let existing_rows: Vec<(String, String, bool, i64)> = {
            let mut stmt = tx.prepare(
                "SELECT column_name, column_type, nulls_allowed, column_id
                 FROM ducklake_column
                 WHERE table_id = ? AND end_snapshot IS NULL
                 ORDER BY column_order",
            )?;
            let mapped = stmt.query_map(params![table_id], |row| {
                let name: String = row.get(0)?;
                let col_type: String = row.get(1)?;
                let nullable: Option<bool> = row.get(2)?;
                let cid: i64 = row.get(3)?;
                Ok((name, col_type, nullable.unwrap_or(true), cid))
            })?;
            mapped.collect::<std::result::Result<Vec<_>, duckdb::Error>>()?
        };

        let mut existing_columns: Vec<(String, String, bool)> =
            Vec::with_capacity(existing_rows.len());
        let mut existing_ids: std::collections::HashMap<String, i64> =
            std::collections::HashMap::new();
        for (name, col_type, nullable, cid) in existing_rows {
            existing_ids.insert(name.clone(), cid);
            existing_columns.push((name, col_type, nullable));
        }

        // Data-write policy: a data write — Replace OR Append — must NOT change a
        // column's type (that is schema evolution, which must go through
        // promote_column_type). Comparison is canonical (int64 ≡ bigint) so an
        // alias-only restatement is a no-op. Append additionally requires a
        // genuinely new column to be nullable. Mirrors the SQLite writer.
        if !existing_columns.is_empty() {
            use std::collections::HashMap;
            let existing_map: HashMap<&str, (&str, bool)> = existing_columns
                .iter()
                .map(|(name, col_type, nullable)| (name.as_str(), (col_type.as_str(), *nullable)))
                .collect();

            for new_col in columns.iter() {
                if let Some((existing_type, _existing_nullable)) = existing_map.get(new_col.name())
                {
                    if !crate::types::types_equal_canonical(existing_type, new_col.ducklake_type())
                    {
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

        // Final per-column ids: reuse the existing id for a column already in the
        // table, consume a freshly reserved id only for a genuinely new column.
        // These are baked into the staged parquet's field_id metadata, so they
        // must equal the ids finalize_snapshot commits. The column rows are
        // written at the commit point (not here).
        let column_ids: Vec<i64> = columns
            .iter()
            .zip(fresh_ids.iter())
            .map(|(col, &fresh)| existing_ids.get(col.name()).copied().unwrap_or(fresh))
            .collect();

        // Only the idempotent get-or-create schema/table rows are committed here;
        // they carry begin_snapshot = the reserved id and stay invisible until the
        // snapshot publishes, since schema/table reads ARE snapshot-scoped.
        tx.commit()?;

        Ok(WriteSetupResult {
            snapshot_id,
            base_snapshot_id,
            schema_id,
            table_id,
            column_ids,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DuckdbMetadataProvider;
    use crate::metadata_provider::MetadataProvider;
    use tempfile::TempDir;

    /// End-to-end round trip: write a small table through the real write path
    /// (begin_write_transaction + register_data_file), then read every catalog
    /// facet back through the DuckdbMetadataProvider and assert the snapshot,
    /// data_path, table, columns and data-file rows are byte-compatible.
    #[test]
    fn duckdb_write_then_read_back_via_provider() {
        let temp = TempDir::new().unwrap();
        let db_path = temp.path().join("catalog.ducklake");
        let db_path_str = db_path.to_str().unwrap().to_string();
        let data_path = temp.path().join("data");
        let data_path_str = data_path.to_str().unwrap().to_string();

        // --- write ---
        {
            let writer = DuckdbMetadataWriter::new_with_init(&db_path_str).unwrap();
            writer.set_data_path(&data_path_str).unwrap();

            let columns = vec![
                ColumnDef::new("id", "int64", false).unwrap(),
                ColumnDef::new("name", "varchar", true).unwrap(),
            ];

            let setup = writer
                .begin_write_transaction("main", "users", &columns, WriteMode::Append)
                .unwrap();
            assert_eq!(setup.column_ids.len(), 2);
            assert_eq!(setup.base_snapshot_id, 0, "fresh catalog: no prior head");

            let file = DataFileInfo::new("users/data-0.parquet", 1024, 3).with_footer_size(256);
            let commit = writer
                .register_data_file(
                    setup.table_id,
                    "main",
                    "users",
                    setup.snapshot_id,
                    &file,
                    WriteMode::Append,
                    setup.base_snapshot_id,
                    &columns,
                    &setup.column_ids,
                )
                .unwrap();
            assert_eq!(commit.snapshot_id, 1, "first write commits snapshot 1");
            // writer (and its read-write lock on the file) dropped at block end.
        }

        // --- read back through the provider (read-only connection) ---
        let provider = DuckdbMetadataProvider::new(&db_path_str).unwrap();

        let snapshot = provider.get_current_snapshot().unwrap();
        assert_eq!(snapshot, 1, "committed head is snapshot 1");

        assert_eq!(provider.get_data_path().unwrap(), data_path_str);

        let schema = provider
            .get_schema_by_name("main", snapshot)
            .unwrap()
            .expect("schema 'main' must exist");
        let table = provider
            .get_table_by_name(schema.schema_id, "users", snapshot)
            .unwrap()
            .expect("table 'users' must exist");
        assert_eq!(table.table_name, "users");

        let cols = provider
            .get_table_structure(table.table_id, snapshot)
            .unwrap();
        assert_eq!(cols.len(), 2, "both columns must read back");
        assert_eq!(cols[0].column_name, "id");
        assert_eq!(cols[0].column_type, "int64");
        assert!(!cols[0].is_nullable);
        assert_eq!(cols[1].column_name, "name");
        assert_eq!(cols[1].column_type, "varchar");
        assert!(cols[1].is_nullable);

        let files = provider
            .get_table_files_for_select(table.table_id, snapshot)
            .unwrap();
        assert_eq!(files.len(), 1, "exactly one data file");
        assert_eq!(files[0].file.path, "users/data-0.parquet");
        assert!(files[0].file.path_is_relative);
        assert_eq!(files[0].file.file_size_bytes, 1024);
        assert_eq!(files[0].file.footer_size, Some(256));
        assert_eq!(files[0].max_row_count, Some(3));
        assert_eq!(
            files[0].row_id_start,
            Some(0),
            "first file starts at rowid 0"
        );
    }

    /// A second Append hands out a fresh, non-overlapping rowid range and
    /// accumulates the table stats — mirrors the SQLite writer's
    /// `row_id_start_advances_across_inserts`.
    #[test]
    fn duckdb_row_id_start_advances_across_appends() {
        let temp = TempDir::new().unwrap();
        let db_path = temp.path().join("catalog.ducklake");
        let db_path_str = db_path.to_str().unwrap().to_string();

        let writer = DuckdbMetadataWriter::new_with_init(&db_path_str).unwrap();
        writer.set_data_path("/tmp/does-not-matter").unwrap();
        let columns = vec![ColumnDef::new("id", "int64", false).unwrap()];

        let setup1 = writer
            .begin_write_transaction("main", "t", &columns, WriteMode::Append)
            .unwrap();
        writer
            .register_data_file(
                setup1.table_id,
                "main",
                "t",
                setup1.snapshot_id,
                &DataFileInfo::new("a.parquet", 100, 3),
                WriteMode::Append,
                setup1.base_snapshot_id,
                &columns,
                &setup1.column_ids,
            )
            .unwrap();

        let setup2 = writer
            .begin_write_transaction("main", "t", &columns, WriteMode::Append)
            .unwrap();
        let commit2 = writer
            .register_data_file(
                setup2.table_id,
                "main",
                "t",
                setup2.snapshot_id,
                &DataFileInfo::new("b.parquet", 250, 7),
                WriteMode::Append,
                setup2.base_snapshot_id,
                &columns,
                &setup2.column_ids,
            )
            .unwrap();
        assert_eq!(commit2.snapshot_id, 2, "second write commits snapshot 2");

        // Read the two files' row_id_start directly to assert non-overlapping ranges.
        let mut conn = writer.connection();
        let tx = conn.transaction().unwrap();
        let a_start: i64 = tx
            .query_row(
                "SELECT row_id_start FROM ducklake_data_file WHERE path = 'a.parquet'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        let b_start: i64 = tx
            .query_row(
                "SELECT row_id_start FROM ducklake_data_file WHERE path = 'b.parquet'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        let (records, next, bytes): (i64, i64, i64) = tx
            .query_row(
                "SELECT record_count, next_row_id, file_size_bytes
                 FROM ducklake_table_stats WHERE table_id = ?",
                params![setup1.table_id],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .unwrap();
        tx.commit().unwrap();

        assert_eq!(a_start, 0, "first file starts at 0");
        assert_eq!(b_start, 3, "second file starts after the first file's rows");
        assert_eq!(records, 10, "record_count = 3 + 7");
        assert_eq!(next, 10, "next_row_id advances by sum of record_counts");
        assert_eq!(bytes, 350, "file_size_bytes accumulates");
    }
}
