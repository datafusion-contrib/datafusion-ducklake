//! Multicatalog manager: create, look up, and bootstrap DuckLake catalogs.
//!
//! See [`crate::metadata_writer_postgres::PostgresMetadataWriter`] for the writer
//! that binds to a `catalog_id` produced here.

use crate::Result;
use crate::metadata_writer_postgres::execute_ddl_statements;
use sqlx::Row;
use sqlx::postgres::PgPool;

/// Bootstrap standard + multicatalog tables. Idempotent.
pub async fn initialize_multicatalog_schema(pool: &PgPool) -> Result<()> {
    execute_ddl_statements(
        pool,
        crate::metadata_writer_postgres::SQL_CREATE_STANDARD_TABLES,
    )
    .await?;
    execute_ddl_statements(
        pool,
        crate::metadata_writer_postgres::SQL_CREATE_MULTICATALOG_TABLES,
    )
    .await?;
    Ok(())
}

/// Manages DuckLake catalogs within a shared metadata database.
///
/// `MulticatalogManager` is stateless: it just executes SQL against the pool.
/// Multiple managers against the same pool are safe.
#[derive(Debug, Clone)]
pub struct MulticatalogManager {
    pool: PgPool,
}

impl MulticatalogManager {
    /// Construct a manager bound to a Postgres pool.
    pub fn new(pool: PgPool) -> Self {
        Self {
            pool,
        }
    }

    /// Create a catalog with the given name, returning its `catalog_id`.
    /// Idempotent. Concurrency-safety relies on the `UNIQUE (catalog_name)`
    /// constraint on `ducklake_catalog`.
    pub async fn create_catalog(&self, name: &str) -> Result<i64> {
        if name.trim().is_empty() {
            return Err(crate::DuckLakeError::InvalidConfig(
                "Catalog name cannot be empty".to_string(),
            ));
        }

        // Fast path: already exists.
        if let Some(id) = self.find_catalog_id(name).await? {
            return Ok(id);
        }

        // Try to insert; if a concurrent writer beat us, fall back to lookup.
        let insert = sqlx::query(
            "INSERT INTO ducklake_catalog (catalog_name) VALUES ($1)
             ON CONFLICT (catalog_name) DO NOTHING
             RETURNING catalog_id",
        )
        .bind(name)
        .fetch_optional(&self.pool)
        .await?;

        if let Some(row) = insert {
            return Ok(row.try_get(0)?);
        }

        // Conflict: row exists from a concurrent writer. Look it up.
        self.find_catalog_id(name).await?.ok_or_else(|| {
            crate::DuckLakeError::Internal(format!(
                "Catalog '{}' insert conflicted but lookup found nothing — \
                 the row was inserted and then deleted concurrently",
                name
            ))
        })
    }

    /// Look up a catalog by name, returning `None` if not found.
    pub async fn find_catalog_id(&self, name: &str) -> Result<Option<i64>> {
        let row = sqlx::query("SELECT catalog_id FROM ducklake_catalog WHERE catalog_name = $1")
            .bind(name)
            .fetch_optional(&self.pool)
            .await?;

        match row {
            Some(r) => Ok(Some(r.try_get(0)?)),
            None => Ok(None),
        }
    }

    /// List all catalogs as (id, name) pairs ordered by id.
    pub async fn list_catalogs(&self) -> Result<Vec<(i64, String)>> {
        let rows = sqlx::query(
            "SELECT catalog_id, catalog_name FROM ducklake_catalog ORDER BY catalog_id",
        )
        .fetch_all(&self.pool)
        .await?;

        let mut out = Vec::with_capacity(rows.len());
        for row in rows {
            let id: i64 = row.try_get(0)?;
            let name: String = row.try_get(1)?;
            out.push((id, name));
        }
        Ok(out)
    }

    /// Drop a catalog and every metadata row owned by it.
    ///
    /// Removes the catalog row, its mapping-table entries, and every
    /// entity reachable through those mappings: snapshots, schemas,
    /// tables, columns, data and delete file records, schema-version
    /// history. Idempotent — returns `false` if no catalog by that name
    /// exists.
    ///
    /// # Concurrency
    ///
    /// The catalog row is taken with `FOR UPDATE` before any deletion,
    /// so concurrent writers (which acquire the same lock in
    /// `lock_catalog`) serialise against the drop. A 30s `lock_timeout`
    /// bounds how long this waits for an in-progress writer to release
    /// the lock — matches
    /// [`crate::metadata_writer_postgres::DEFAULT_LOCK_TIMEOUT_MS`]. The
    /// entire teardown is one transaction; either everything is gone
    /// or nothing changes.
    ///
    /// # What's NOT cleaned up
    ///
    /// Data files on object storage are not removed here. Vacuum
    /// reclaims them later, matching the crate's drop-without-commit
    /// convention. Callers needing tighter storage cleanup should list
    /// the data file paths before calling this and schedule their
    /// removal externally.
    pub async fn drop_catalog(&self, name: &str) -> Result<bool> {
        if name.trim().is_empty() {
            return Err(crate::DuckLakeError::InvalidConfig(
                "Catalog name cannot be empty".to_string(),
            ));
        }

        let mut tx = self.pool.begin().await?;

        sqlx::query("SET LOCAL lock_timeout = 30000")
            .execute(&mut *tx)
            .await?;

        // Resolve catalog_id under FOR UPDATE. Concurrent writers (which
        // also take FOR UPDATE on this row in `lock_catalog`) serialise
        // against the drop. If the row has already been deleted by a
        // parallel drop, we treat this call as a no-op.
        let row = sqlx::query(
            "SELECT catalog_id FROM ducklake_catalog
             WHERE catalog_name = $1 FOR UPDATE",
        )
        .bind(name)
        .fetch_optional(&mut *tx)
        .await?;

        let catalog_id: i64 = match row {
            Some(r) => r.try_get(0)?,
            None => {
                tx.commit().await?;
                return Ok(false);
            },
        };

        // Order matters: anything that reads through the catalog mapping
        // tables must run before we delete those mapping rows. Within
        // catalog-owned entities the order is rows-that-reference-
        // `table_id` first, then tables, then schemas, then snapshots.

        // Catalog-scoped child rows that reference `table_id`. Filtered
        // transitively through the schema map so we only touch rows
        // owned by this catalog.
        for child_table in [
            "ducklake_data_file",
            "ducklake_delete_file",
            "ducklake_column",
            "ducklake_schema_versions",
        ] {
            sqlx::query(&format!(
                "DELETE FROM {} WHERE table_id IN (
                    SELECT t.table_id FROM ducklake_table t
                    JOIN ducklake_catalog_schema_map m ON m.schema_id = t.schema_id
                    WHERE m.catalog_id = $1
                )",
                child_table
            ))
            .bind(catalog_id)
            .execute(&mut *tx)
            .await?;
        }

        // Tables and schemas owned by this catalog.
        sqlx::query(
            "DELETE FROM ducklake_table WHERE schema_id IN (
                SELECT schema_id FROM ducklake_catalog_schema_map WHERE catalog_id = $1
            )",
        )
        .bind(catalog_id)
        .execute(&mut *tx)
        .await?;

        sqlx::query(
            "DELETE FROM ducklake_schema WHERE schema_id IN (
                SELECT schema_id FROM ducklake_catalog_schema_map WHERE catalog_id = $1
            )",
        )
        .bind(catalog_id)
        .execute(&mut *tx)
        .await?;

        // Snapshots are catalog-scoped only via the snapshot map; they
        // carry no catalog_id of their own, so the map is the only path
        // to locate them.
        sqlx::query(
            "DELETE FROM ducklake_snapshot WHERE snapshot_id IN (
                SELECT snapshot_id FROM ducklake_catalog_snapshot_map WHERE catalog_id = $1
            )",
        )
        .bind(catalog_id)
        .execute(&mut *tx)
        .await?;

        // Now safe to remove the mapping rows — nothing reads through
        // them anymore.
        sqlx::query("DELETE FROM ducklake_catalog_schema_map WHERE catalog_id = $1")
            .bind(catalog_id)
            .execute(&mut *tx)
            .await?;
        sqlx::query("DELETE FROM ducklake_catalog_snapshot_map WHERE catalog_id = $1")
            .bind(catalog_id)
            .execute(&mut *tx)
            .await?;

        // Finally the catalog row itself.
        sqlx::query("DELETE FROM ducklake_catalog WHERE catalog_id = $1")
            .bind(catalog_id)
            .execute(&mut *tx)
            .await?;

        tx.commit().await?;
        Ok(true)
    }

    /// Tombstone a single table inside a catalog at a new commit snapshot.
    ///
    /// Allocates a new snapshot (the "drop snapshot") and sets
    /// `end_snapshot = <drop snapshot>` on the currently-live rows for
    /// the named table in `ducklake_table`, `ducklake_column`,
    /// `ducklake_data_file`, and `ducklake_delete_file`. After commit,
    /// reads at any snapshot `>=` the drop snapshot will not see the
    /// table; reads at earlier snapshots are unaffected.
    ///
    /// Idempotent — returns `Ok(false)` when no live table by that
    /// `(catalog, schema, table)` triple exists. No snapshot is
    /// allocated in that case.
    ///
    /// # Concurrency
    ///
    /// The catalog row is taken with `FOR UPDATE` before any work, so
    /// concurrent writers (which acquire the same lock in
    /// `lock_catalog`) serialise against the drop. A 30s
    /// `lock_timeout` bounds how long this waits for an in-progress
    /// writer to release the lock — matches
    /// [`crate::metadata_writer_postgres::DEFAULT_LOCK_TIMEOUT_MS`].
    /// The entire teardown is one transaction; either the table is
    /// tombstoned at the new snapshot or nothing changes.
    ///
    /// # What's NOT cleaned up
    ///
    /// Data files on object storage are not removed here, and the
    /// tombstoned metadata rows remain in place under their
    /// `end_snapshot`. Physical reclamation is the job of a future
    /// expire-snapshots / cleanup-files vacuum (mirrors the official
    /// DuckLake design). Callers needing tighter storage cleanup
    /// should list the table's data file paths via
    /// [`crate::MetadataProvider::get_table_files_for_select`] BEFORE
    /// calling this and schedule their removal externally.
    pub async fn drop_table_in_catalog(
        &self,
        catalog_name: &str,
        schema_name: &str,
        table_name: &str,
    ) -> Result<bool> {
        if catalog_name.trim().is_empty() {
            return Err(crate::DuckLakeError::InvalidConfig(
                "Catalog name cannot be empty".to_string(),
            ));
        }
        if schema_name.trim().is_empty() {
            return Err(crate::DuckLakeError::InvalidConfig(
                "Schema name cannot be empty".to_string(),
            ));
        }
        if table_name.trim().is_empty() {
            return Err(crate::DuckLakeError::InvalidConfig(
                "Table name cannot be empty".to_string(),
            ));
        }

        let mut tx = self.pool.begin().await?;

        sqlx::query("SET LOCAL lock_timeout = 30000")
            .execute(&mut *tx)
            .await?;

        // Resolve catalog_id under FOR UPDATE. Concurrent writers (which
        // also take FOR UPDATE on this row in `lock_catalog`) serialise
        // against the drop. An unknown catalog is a no-op — same
        // idempotency contract as `drop_catalog`.
        let catalog_id: i64 = match sqlx::query(
            "SELECT catalog_id FROM ducklake_catalog
             WHERE catalog_name = $1 FOR UPDATE",
        )
        .bind(catalog_name)
        .fetch_optional(&mut *tx)
        .await?
        {
            Some(r) => r.try_get(0)?,
            None => {
                tx.commit().await?;
                return Ok(false);
            },
        };

        // Resolve the live table_id by `(catalog, schema, table)`.
        // Filters on `end_snapshot IS NULL` on both schema and table so
        // an already-tombstoned table is treated as not-found and the
        // call becomes an idempotent no-op.
        let table_id: i64 = match sqlx::query(
            "SELECT t.table_id FROM ducklake_table t
             JOIN ducklake_schema s ON s.schema_id = t.schema_id
             JOIN ducklake_catalog_schema_map m ON m.schema_id = s.schema_id
             WHERE m.catalog_id = $1
               AND s.schema_name = $2
               AND s.end_snapshot IS NULL
               AND t.table_name = $3
               AND t.end_snapshot IS NULL",
        )
        .bind(catalog_id)
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

        // Allocate the drop snapshot and register it under this catalog.
        // Every snapshot has a row in `ducklake_catalog_snapshot_map` —
        // same discipline as `MetadataWriter::create_snapshot` and
        // `begin_write_transaction`.
        //
        // DROP TABLE is a DDL change (the set of live tables in the
        // catalog shrinks by one), so the drop snapshot bumps
        // `schema_version` — `prev_max + 1` per the per-catalog dense
        // contract that `begin_write_transaction` enforces for DDL
        // commits. Computed AFTER the snapshot is inserted, filtered
        // strictly less than its own id, so concurrent writers (which
        // hold the same catalog FOR UPDATE lock we do) can't slip a
        // row in to invalidate the dense allocation.
        let drop_snapshot: i64 = sqlx::query(
            "INSERT INTO ducklake_snapshot (snapshot_time, schema_version)
             VALUES (CURRENT_TIMESTAMP, 0) RETURNING snapshot_id",
        )
        .fetch_one(&mut *tx)
        .await?
        .try_get(0)?;

        sqlx::query(
            "INSERT INTO ducklake_catalog_snapshot_map (catalog_id, snapshot_id)
             VALUES ($1, $2)",
        )
        .bind(catalog_id)
        .bind(drop_snapshot)
        .execute(&mut *tx)
        .await?;

        // Bump per-catalog dense schema_version: prior MAX + 1 (DDL).
        // Falls back to 1 when this is the catalog's first commit ever,
        // mirroring `begin_write_transaction`'s "no prior snapshot ⇒ v1"
        // guard so the schema_version stays monotone-from-1.
        let prev_max: i64 = sqlx::query(
            "SELECT COALESCE(MAX(s.schema_version), 0) FROM ducklake_snapshot s
             JOIN ducklake_catalog_snapshot_map m ON m.snapshot_id = s.snapshot_id
             WHERE m.catalog_id = $1 AND s.snapshot_id < $2",
        )
        .bind(catalog_id)
        .bind(drop_snapshot)
        .fetch_one(&mut *tx)
        .await?
        .try_get(0)?;
        let new_schema_version = if prev_max == 0 {
            1
        } else {
            prev_max + 1
        };
        sqlx::query("UPDATE ducklake_snapshot SET schema_version = $1 WHERE snapshot_id = $2")
            .bind(new_schema_version)
            .bind(drop_snapshot)
            .execute(&mut *tx)
            .await?;

        // Tombstone the table row and every currently-live child row
        // keyed by `table_id`. The `end_snapshot IS NULL` guard makes
        // each UPDATE a no-op for rows already tombstoned at an earlier
        // snapshot (e.g. data files superseded by a prior REPLACE).
        //
        // `ducklake_schema_versions` rows are not tombstoned — that
        // table has no `end_snapshot` column. They become unreferenced
        // after the table is fully expired; future vacuum reclaims
        // them. This mirrors the official DuckLake's `DropTables`
        // (which leaves `ducklake_schema_versions` to vacuum likewise).
        for child_table in
            ["ducklake_table", "ducklake_column", "ducklake_data_file", "ducklake_delete_file"]
        {
            sqlx::query(&format!(
                "UPDATE {} SET end_snapshot = $1
                 WHERE table_id = $2 AND end_snapshot IS NULL",
                child_table
            ))
            .bind(drop_snapshot)
            .bind(table_id)
            .execute(&mut *tx)
            .await?;
        }

        tx.commit().await?;
        Ok(true)
    }
}
