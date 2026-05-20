//! SQLite implementation of [`MetadataWriter`].
//!
//! Requires multi-threaded Tokio runtime (`#[tokio::test(flavor = "multi_thread")]`).

use crate::Result;
use crate::metadata_provider::block_on;
use crate::metadata_writer::{
    ColumnDef, DataFileInfo, MetadataWriter, WriteMode, WriteSetupResult, validate_name,
};
use sqlx::Row;
use sqlx::sqlite::{SqlitePool, SqlitePoolOptions};

const DEFAULT_MAX_CONNECTIONS: u32 = 5;

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
}

impl MetadataWriter for SqliteMetadataWriter {
    fn create_snapshot(&self) -> Result<i64> {
        block_on(async {
            let row = sqlx::query(
                "INSERT INTO ducklake_snapshot (snapshot_time) VALUES (CURRENT_TIMESTAMP) RETURNING snapshot_id",
            )
            .fetch_one(&self.pool)
            .await?;
            Ok(row.try_get(0)?)
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

            let mut column_ids = Vec::with_capacity(columns.len());
            for (order, col) in columns.iter().enumerate() {
                let row = sqlx::query(
                    "INSERT INTO ducklake_column (table_id, column_name, column_type, column_order, nulls_allowed, begin_snapshot)
                     VALUES (?, ?, ?, ?, ?, ?) RETURNING column_id",
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
            // overlapping ranges. Falls back to inserting a fresh stats row
            // (next_row_id = 0) for tables created before this writer
            // started maintaining the table_stats table.
            let mut tx = self.pool.begin().await?;

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

            let row = sqlx::query(
                "INSERT INTO ducklake_snapshot (snapshot_time) VALUES (CURRENT_TIMESTAMP) RETURNING snapshot_id",
            )
            .fetch_one(&mut *tx)
            .await?;
            let snapshot_id: i64 = row.try_get(0)?;

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

            sqlx::query(
                "UPDATE ducklake_column SET end_snapshot = ?
                 WHERE table_id = ? AND end_snapshot IS NULL",
            )
            .bind(snapshot_id)
            .bind(table_id)
            .execute(&mut *tx)
            .await?;

            let mut column_ids = Vec::with_capacity(columns.len());
            for (order, col) in columns.iter().enumerate() {
                let row = sqlx::query(
                    "INSERT INTO ducklake_column (table_id, column_name, column_type, column_order, nulls_allowed, begin_snapshot)
                     VALUES (?, ?, ?, ?, ?, ?) RETURNING column_id",
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

            if mode == WriteMode::Replace {
                sqlx::query(
                    "UPDATE ducklake_data_file SET end_snapshot = ?
                     WHERE table_id = ? AND end_snapshot IS NULL",
                )
                .bind(snapshot_id)
                .bind(table_id)
                .execute(&mut *tx)
                .await?;

                // Mirror end_table_files: drop visible row/byte totals to
                // zero while keeping next_row_id monotonic. INSERT OR IGNORE
                // first so we don't fall over if this is the first write
                // (Replace on a brand-new table).
                sqlx::query(
                    "INSERT OR IGNORE INTO ducklake_table_stats
                         (table_id, record_count, next_row_id, file_size_bytes)
                     VALUES (?, 0, 0, 0)",
                )
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
        let snapshot_id = writer.create_snapshot().unwrap();
        let (schema_id, _) = writer
            .get_or_create_schema("main", None, snapshot_id)
            .unwrap();
        let (table_id, _) = writer
            .get_or_create_table(schema_id, "users", None, snapshot_id)
            .unwrap();

        let file = DataFileInfo::new("data.parquet", 1024, 100).with_footer_size(256);

        let file_id = writer
            .register_data_file(table_id, snapshot_id, &file)
            .unwrap();
        assert_eq!(file_id, 1);
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
        let snapshot_id = writer.create_snapshot().unwrap();
        let (schema_id, _) = writer
            .get_or_create_schema("main", None, snapshot_id)
            .unwrap();
        let (table_id, _) = writer
            .get_or_create_table(schema_id, "t", None, snapshot_id)
            .unwrap();

        let f1_id = writer
            .register_data_file(
                table_id,
                snapshot_id,
                &DataFileInfo::new("a.parquet", 100, 3),
            )
            .unwrap();
        let f2_id = writer
            .register_data_file(
                table_id,
                snapshot_id,
                &DataFileInfo::new("b.parquet", 250, 7),
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
        let snap1 = writer.create_snapshot().unwrap();
        let (schema_id, _) = writer.get_or_create_schema("main", None, snap1).unwrap();
        let (table_id, _) = writer
            .get_or_create_table(schema_id, "t", None, snap1)
            .unwrap();

        writer
            .register_data_file(table_id, snap1, &DataFileInfo::new("a.parquet", 100, 5))
            .unwrap();

        let snap2 = writer.create_snapshot().unwrap();
        writer.end_table_files(table_id, snap2).unwrap();

        let (records, next, bytes) = read_table_stats(&writer, table_id).await;
        assert_eq!(records, 0, "record_count cleared after end_table_files");
        assert_eq!(next, 5, "next_row_id preserved (monotonic across lifetime)");
        assert_eq!(bytes, 0, "file_size_bytes cleared");

        let f2_id = writer
            .register_data_file(table_id, snap2, &DataFileInfo::new("b.parquet", 200, 2))
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
        let snapshot_id = writer.create_snapshot().unwrap();
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
        let snapshot1 = writer.create_snapshot().unwrap();
        let (schema_id, _) = writer
            .get_or_create_schema("main", None, snapshot1)
            .unwrap();
        let (table_id, _) = writer
            .get_or_create_table(schema_id, "users", None, snapshot1)
            .unwrap();

        // Register a file
        let file = DataFileInfo::new("data1.parquet", 1024, 100);
        writer
            .register_data_file(table_id, snapshot1, &file)
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
