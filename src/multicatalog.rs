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
}
