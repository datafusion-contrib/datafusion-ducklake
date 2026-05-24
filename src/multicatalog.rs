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
}
