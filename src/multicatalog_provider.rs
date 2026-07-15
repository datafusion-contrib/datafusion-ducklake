//! Catalog-scoped Postgres reader for DuckLake multicatalog.
//!
//! Bound to a single `catalog_id` at construction. All queries that touch
//! catalog-discriminated entities (snapshots, schemas, top-level lists) join
//! through `ducklake_catalog_snapshot_map` / `ducklake_catalog_schema_map`.
//! Queries keyed by a globally unique id (`schema_id`, `table_id`) need no
//! extra scoping because the caller already obtained the id through a
//! catalog-scoped lookup.
//!
//! This implementation is standalone; it does not wrap
//! [`crate::PostgresMetadataProvider`] — the existing single-catalog provider
//! is untouched.

use crate::Result;
use crate::metadata_provider::{
    ColumnWithTable, DataFileChange, DeleteFileChange, DuckLakeFileColumnStatistics,
    DuckLakeFileData, DuckLakeFileMetadata, DuckLakeStatistics, DuckLakeTableColumn,
    DuckLakeTableColumnStatistics, DuckLakeTableFile, DuckLakeTableStatistics, FileWithTable,
    MetadataProvider, SchemaMetadata, SnapshotMetadata, TableMetadata, TableWithSchema, block_on,
    reconstruct_list_columns, reconstruct_list_columns_with_table,
};
use sqlx::Row;
use sqlx::postgres::{PgPool, PgPoolOptions, PgRow};
use sqlx::types::chrono::NaiveDateTime;
use std::collections::HashMap;

fn is_missing_statistics_table(error: &sqlx::Error) -> bool {
    let message = error.to_string().to_ascii_lowercase();
    message.contains("does not exist") || message.contains("undefined table")
}

fn decode_table_file(row: &PgRow, snapshot_id: i64) -> Result<DuckLakeTableFile> {
    let delete_file_id: Option<i64> = row.try_get(8)?;
    let (delete_file, delete_count) = if delete_file_id.is_some() {
        (
            Some(DuckLakeFileData {
                path: row.try_get(9)?,
                path_is_relative: row.try_get(10)?,
                file_size_bytes: row.try_get(11)?,
                footer_size: row.try_get(12)?,
                encryption_key: row.try_get(13)?,
            }),
            row.try_get(14)?,
        )
    } else {
        (None, None)
    };
    Ok(DuckLakeTableFile {
        data_file_id: row.try_get(0)?,
        file: DuckLakeFileData {
            path: row.try_get(1)?,
            path_is_relative: row.try_get(2)?,
            file_size_bytes: row.try_get(3)?,
            footer_size: row.try_get(4)?,
            encryption_key: row.try_get(5)?,
        },
        delete_file_id,
        delete_file,
        row_id_start: row.try_get(6)?,
        snapshot_id: Some(snapshot_id),
        begin_snapshot: row.try_get(15)?,
        schema_version: row.try_get(17)?,
        partial_max: row.try_get(16)?,
        max_row_count: row.try_get(7)?,
        delete_count,
    })
}

const DEFAULT_MAX_CONNECTIONS: u32 = 5;

/// Catalog-scoped Postgres metadata reader.
///
/// Construct with [`Self::with_pool`] (name-keyed; resolves to `catalog_id` once
/// at construction) or [`Self::with_pool_and_id`] (id-keyed; skip the lookup).
#[derive(Debug, Clone)]
pub struct MulticatalogProvider {
    pool: PgPool,
    catalog_id: i64,
}

impl MulticatalogProvider {
    /// Build a pool from a connection string, then resolve the catalog by name.
    pub async fn new(connection_string: &str, catalog_name: &str) -> Result<Self> {
        let pool = PgPoolOptions::new()
            .max_connections(DEFAULT_MAX_CONNECTIONS)
            .connect(connection_string)
            .await?;
        Self::with_pool(pool, catalog_name).await
    }

    /// Bind to an existing pool, resolving the catalog by name.
    ///
    /// Returns [`crate::DuckLakeError::CatalogNotFound`] if no row in
    /// `ducklake_catalog` matches `catalog_name`.
    pub async fn with_pool(pool: PgPool, catalog_name: &str) -> Result<Self> {
        let row = sqlx::query("SELECT catalog_id FROM ducklake_catalog WHERE catalog_name = $1")
            .bind(catalog_name)
            .fetch_optional(&pool)
            .await?;
        let catalog_id: i64 = row
            .ok_or_else(|| crate::DuckLakeError::CatalogNotFound(catalog_name.to_string()))?
            .try_get(0)?;
        Ok(Self {
            pool,
            catalog_id,
        })
    }

    /// Bind to an existing pool with an already-known `catalog_id`. Skips the
    /// name lookup. Caller is responsible for ensuring the id exists.
    pub async fn with_pool_and_id(pool: PgPool, catalog_id: i64) -> Result<Self> {
        Ok(Self {
            pool,
            catalog_id,
        })
    }

    pub fn catalog_id(&self) -> i64 {
        self.catalog_id
    }
}

impl MetadataProvider for MulticatalogProvider {
    fn get_current_snapshot(&self) -> Result<i64> {
        block_on(async {
            let row = sqlx::query(
                "SELECT COALESCE(MAX(snapshot_id), 0)
                 FROM ducklake_catalog_snapshot_map
                 WHERE catalog_id = $1",
            )
            .bind(self.catalog_id)
            .fetch_one(&self.pool)
            .await?;
            Ok(row.try_get(0)?)
        })
    }

    fn get_data_path(&self) -> Result<String> {
        // data_path is global per Phase 1 default.
        block_on(async {
            let row =
                sqlx::query("SELECT value FROM ducklake_metadata WHERE key = $1 AND scope IS NULL")
                    .bind("data_path")
                    .fetch_optional(&self.pool)
                    .await?;

            match row {
                Some(r) => Ok(r.try_get(0)?),
                None => Err(crate::error::DuckLakeError::InvalidConfig(
                    "Missing required catalog metadata: 'data_path' not configured. \
                     The catalog may be uninitialized or corrupted."
                        .to_string(),
                )),
            }
        })
    }

    fn list_snapshots(&self) -> Result<Vec<SnapshotMetadata>> {
        block_on(async {
            let rows = sqlx::query(
                "SELECT s.snapshot_id, s.snapshot_time
                 FROM ducklake_snapshot s
                 JOIN ducklake_catalog_snapshot_map m ON m.snapshot_id = s.snapshot_id
                 WHERE m.catalog_id = $1
                 ORDER BY s.snapshot_id",
            )
            .bind(self.catalog_id)
            .fetch_all(&self.pool)
            .await?;

            rows.into_iter()
                .map(|row| {
                    let snapshot_id: i64 = row.try_get(0)?;
                    let timestamp: Option<NaiveDateTime> = row.try_get(1)?;
                    let timestamp_str =
                        timestamp.map(|ts| ts.format("%Y-%m-%d %H:%M:%S%.6f").to_string());
                    Ok(SnapshotMetadata {
                        snapshot_id,
                        timestamp: timestamp_str,
                    })
                })
                .collect()
        })
    }

    fn list_schemas(&self, snapshot_id: i64) -> Result<Vec<SchemaMetadata>> {
        block_on(async {
            let rows = sqlx::query(
                "SELECT s.schema_id, s.schema_name, s.path, s.path_is_relative
                 FROM ducklake_schema s
                 JOIN ducklake_catalog_schema_map m ON m.schema_id = s.schema_id
                 WHERE m.catalog_id = $1
                   AND $2 >= s.begin_snapshot
                   AND ($3 < s.end_snapshot OR s.end_snapshot IS NULL)",
            )
            .bind(self.catalog_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .fetch_all(&self.pool)
            .await?;

            rows.into_iter()
                .map(|row| {
                    Ok(SchemaMetadata {
                        schema_id: row.try_get(0)?,
                        schema_name: row.try_get(1)?,
                        path: row.try_get(2)?,
                        path_is_relative: row.try_get(3)?,
                    })
                })
                .collect()
        })
    }

    fn list_tables(&self, schema_id: i64, snapshot_id: i64) -> Result<Vec<TableMetadata>> {
        // schema_id is globally unique; caller has already resolved it via
        // get_schema_by_name (catalog-scoped). No additional scoping needed.
        block_on(async {
            let rows = sqlx::query(
                "SELECT table_id, table_name, path, path_is_relative
                 FROM ducklake_table
                 WHERE schema_id = $1
                   AND $2 >= begin_snapshot
                   AND ($3 < end_snapshot OR end_snapshot IS NULL)",
            )
            .bind(schema_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .fetch_all(&self.pool)
            .await?;

            rows.into_iter()
                .map(|row| {
                    Ok(TableMetadata {
                        table_id: row.try_get(0)?,
                        table_name: row.try_get(1)?,
                        path: row.try_get(2)?,
                        path_is_relative: row.try_get(3)?,
                    })
                })
                .collect()
        })
    }

    fn get_table_structure(
        &self,
        table_id: i64,
        snapshot_id: i64,
    ) -> Result<Vec<DuckLakeTableColumn>> {
        // Columns inherit catalog via table_id (a table belongs to exactly one
        // catalog), but must still be SNAPSHOT-scoped like list_tables /
        // list_schemas: reading by `end_snapshot IS NULL` alone leaks a
        // concurrent or aborted writer's begin-time column generation (which
        // commits before the head advances). Match the catalog head window.
        block_on(async {
            let rows = sqlx::query(
                "SELECT column_id, column_name, column_type, nulls_allowed, parent_column
                 FROM ducklake_column
                 WHERE table_id = $1
                   AND $2 >= begin_snapshot
                   AND ($3 < end_snapshot OR end_snapshot IS NULL)
                 ORDER BY column_order",
            )
            .bind(table_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .fetch_all(&self.pool)
            .await?;

            let raw: Result<Vec<(DuckLakeTableColumn, Option<i64>)>> = rows
                .into_iter()
                .map(|row| {
                    let nulls_allowed: Option<bool> = row.try_get(3)?;
                    let parent_column: Option<i64> = row.try_get(4)?;
                    Ok((
                        DuckLakeTableColumn {
                            column_id: row.try_get(0)?,
                            column_name: row.try_get(1)?,
                            column_type: row.try_get(2)?,
                            is_nullable: nulls_allowed.unwrap_or(true),
                        },
                        parent_column,
                    ))
                })
                .collect();
            Ok(reconstruct_list_columns(raw?))
        })
    }

    fn get_table_files_for_select(
        &self,
        table_id: i64,
        snapshot_id: i64,
    ) -> Result<Vec<DuckLakeTableFile>> {
        // Files inherit catalog via table_id. No scoping.
        block_on(async {
            // Backward compatibility (mirrors the single-catalog Postgres
            // provider): degrade `partial_max` / `schema_version` to NULL when a
            // catalog predates the v1.0 partial-file column or the
            // `ducklake_schema_versions` ledger. Surfacing these lets compaction
            // (which pairs its writer with THIS reader) select merge candidates
            // and lets time-travel reads filter partial files per-row.
            let has_partial_max: bool = sqlx::query_scalar(
                "SELECT EXISTS (SELECT 1 FROM information_schema.columns
                 WHERE table_name = 'ducklake_data_file' AND column_name = 'partial_max')",
            )
            .fetch_one(&self.pool)
            .await?;
            let has_schema_versions: bool =
                sqlx::query_scalar("SELECT to_regclass('ducklake_schema_versions') IS NOT NULL")
                    .fetch_one(&self.pool)
                    .await?;
            let partial_max_expr = if has_partial_max {
                "data.partial_max::bigint"
            } else {
                "NULL::bigint"
            };
            let schema_version_expr = if has_schema_versions {
                "(SELECT sv.schema_version::bigint
                  FROM ducklake_schema_versions sv
                  WHERE sv.table_id = data.table_id
                    AND sv.begin_snapshot <= data.begin_snapshot
                  ORDER BY sv.begin_snapshot DESC
                  LIMIT 1)"
            } else {
                "NULL::bigint"
            };
            let sql = format!(
                "SELECT
                    data.data_file_id,
                    data.path AS data_file_path,
                    data.path_is_relative AS data_path_is_relative,
                    data.file_size_bytes AS data_file_size,
                    data.footer_size AS data_footer_size,
                    data.encryption_key AS data_encryption_key,
                    data.row_id_start AS data_row_id_start,
                    data.record_count AS data_record_count,
                    del.delete_file_id,
                    del.path AS delete_file_path,
                    del.path_is_relative AS delete_path_is_relative,
                    del.file_size_bytes AS delete_file_size,
                    del.footer_size AS delete_footer_size,
                    del.encryption_key AS delete_encryption_key,
                    del.delete_count,
                    data.begin_snapshot::bigint AS data_begin_snapshot,
                    {partial_max_expr} AS data_partial_max,
                    {schema_version_expr} AS data_schema_version
                FROM ducklake_data_file AS data
                LEFT JOIN ducklake_delete_file AS del
                    ON data.data_file_id = del.data_file_id
                    AND del.table_id = $1
                    AND $2 >= del.begin_snapshot
                    AND ($3 < del.end_snapshot OR del.end_snapshot IS NULL)
                WHERE data.table_id = $4
                  AND $5 >= data.begin_snapshot
                  AND ($6 < data.end_snapshot OR data.end_snapshot IS NULL)"
            );
            let rows = sqlx::query(&sql)
                .bind(table_id)
                .bind(snapshot_id)
                .bind(snapshot_id)
                .bind(table_id)
                .bind(snapshot_id)
                .bind(snapshot_id)
                .fetch_all(&self.pool)
                .await?;

            rows.iter()
                .map(|row| decode_table_file(row, snapshot_id))
                .collect()
        })
    }

    fn get_table_file_metadata_page(
        &self,
        table_id: i64,
        snapshot_id: i64,
        after_data_file_id: Option<i64>,
        limit: usize,
    ) -> Result<Vec<DuckLakeFileMetadata>> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let limit = i64::try_from(limit).map_err(|_| {
            crate::DuckLakeError::InvalidConfig("file metadata page limit exceeds i64".to_string())
        })?;
        block_on(async {
            let has_partial_max: bool = sqlx::query_scalar(
                "SELECT EXISTS (SELECT 1 FROM information_schema.columns
                 WHERE table_name = 'ducklake_data_file' AND column_name = 'partial_max')",
            )
            .fetch_one(&self.pool)
            .await?;
            let has_schema_versions: bool =
                sqlx::query_scalar("SELECT to_regclass('ducklake_schema_versions') IS NOT NULL")
                    .fetch_one(&self.pool)
                    .await?;
            let partial_max_expr = if has_partial_max {
                "data.partial_max::bigint"
            } else {
                "NULL::bigint"
            };
            let schema_version_expr = if has_schema_versions {
                "(SELECT sv.schema_version::bigint
                  FROM ducklake_schema_versions sv
                  WHERE sv.table_id = data.table_id
                    AND sv.begin_snapshot <= data.begin_snapshot
                  ORDER BY sv.begin_snapshot DESC LIMIT 1)"
            } else {
                "NULL::bigint"
            };
            let sql = format!(
                "SELECT data.data_file_id, data.path, data.path_is_relative,
                        data.file_size_bytes, data.footer_size, data.encryption_key,
                        data.row_id_start, data.record_count,
                        del.delete_file_id, del.path, del.path_is_relative,
                        del.file_size_bytes, del.footer_size, del.encryption_key,
                        del.delete_count, data.begin_snapshot::bigint,
                        {partial_max_expr}, {schema_version_expr}
                 FROM ducklake_data_file AS data
                 LEFT JOIN ducklake_delete_file AS del
                   ON data.data_file_id = del.data_file_id
                  AND del.table_id = $1
                  AND $2 >= del.begin_snapshot
                  AND ($3 < del.end_snapshot OR del.end_snapshot IS NULL)
                 WHERE data.table_id = $4
                   AND $5 >= data.begin_snapshot
                   AND ($6 < data.end_snapshot OR data.end_snapshot IS NULL)
                   AND data.data_file_id > $7
                 ORDER BY data.data_file_id
                 LIMIT $8"
            );
            let rows = sqlx::query(&sql)
                .bind(table_id)
                .bind(snapshot_id)
                .bind(snapshot_id)
                .bind(table_id)
                .bind(snapshot_id)
                .bind(snapshot_id)
                .bind(after_data_file_id.unwrap_or(i64::MIN))
                .bind(limit)
                .fetch_all(&self.pool)
                .await?;
            let files = rows
                .iter()
                .map(|row| decode_table_file(row, snapshot_id))
                .collect::<Result<Vec<_>>>()?;
            let Some(last_data_file_id) = files.last().map(|file| file.data_file_id) else {
                return Ok(Vec::new());
            };
            let statistics = match sqlx::query(
                "SELECT stats.data_file_id, stats.column_id,
                        stats.column_size_bytes, stats.value_count, stats.null_count,
                        stats.min_value, stats.max_value
                 FROM ducklake_file_column_stats AS stats
                 INNER JOIN ducklake_data_file AS data
                   ON data.data_file_id = stats.data_file_id
                  AND data.table_id = stats.table_id
                 WHERE stats.table_id = $1
                   AND $2 >= data.begin_snapshot
                   AND ($3 < data.end_snapshot OR data.end_snapshot IS NULL)
                   AND stats.data_file_id > $4
                   AND stats.data_file_id <= $5
                 ORDER BY stats.data_file_id, stats.column_id",
            )
            .bind(table_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .bind(after_data_file_id.unwrap_or(i64::MIN))
            .bind(last_data_file_id)
            .fetch_all(&self.pool)
            .await
            {
                Ok(rows) => rows
                    .into_iter()
                    .map(|row| {
                        Ok(DuckLakeFileColumnStatistics {
                            data_file_id: row.try_get(0)?,
                            column_id: row.try_get(1)?,
                            column_size_bytes: row.try_get(2)?,
                            value_count: row.try_get(3)?,
                            null_count: row.try_get(4)?,
                            min_value: row.try_get(5)?,
                            max_value: row.try_get(6)?,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?,
                Err(error) if is_missing_statistics_table(&error) => Vec::new(),
                Err(error) => return Err(error.into()),
            };
            let mut statistics_by_file: HashMap<i64, Vec<_>> = HashMap::new();
            for statistic in statistics {
                statistics_by_file
                    .entry(statistic.data_file_id)
                    .or_default()
                    .push(statistic);
            }
            Ok(files
                .into_iter()
                .map(|file| DuckLakeFileMetadata {
                    column_statistics: statistics_by_file
                        .remove(&file.data_file_id)
                        .unwrap_or_default(),
                    file,
                })
                .collect())
        })
    }

    fn get_table_summary_statistics(
        &self,
        table_id: i64,
        snapshot_id: i64,
    ) -> Result<DuckLakeStatistics> {
        block_on(async {
            let table = match sqlx::query(
                "SELECT record_count, file_size_bytes
                 FROM ducklake_table_stats WHERE table_id = $1",
            )
            .bind(table_id)
            .fetch_optional(&self.pool)
            .await
            {
                Ok(row) => row
                    .map(|row| {
                        Ok::<_, sqlx::Error>(DuckLakeTableStatistics {
                            record_count: row.try_get(0)?,
                            file_size_bytes: row.try_get(1)?,
                        })
                    })
                    .transpose()?,
                Err(error) if is_missing_statistics_table(&error) => None,
                Err(error) => return Err(error.into()),
            };
            let column_sizes = match sqlx::query(
                "SELECT stats.column_id,
                        CASE
                          WHEN COUNT(*) = COUNT(stats.column_size_bytes)
                           AND COUNT(*) = (
                             SELECT COUNT(*) FROM ducklake_data_file visible
                             WHERE visible.table_id = $1
                               AND $2 >= visible.begin_snapshot
                               AND ($3 < visible.end_snapshot OR visible.end_snapshot IS NULL)
                           )
                          THEN CAST(SUM(stats.column_size_bytes) AS BIGINT)
                        END
                 FROM ducklake_file_column_stats stats
                 INNER JOIN ducklake_data_file data
                   ON data.data_file_id = stats.data_file_id
                  AND data.table_id = stats.table_id
                 WHERE stats.table_id = $4
                   AND $5 >= data.begin_snapshot
                   AND ($6 < data.end_snapshot OR data.end_snapshot IS NULL)
                 GROUP BY stats.column_id",
            )
            .bind(table_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .bind(table_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .fetch_all(&self.pool)
            .await
            {
                Ok(rows) => rows
                    .into_iter()
                    .filter_map(|row| match row.try_get::<Option<i64>, _>(1) {
                        Ok(Some(size)) => Some(row.try_get(0).map(|column_id| (column_id, size))),
                        Ok(None) => None,
                        Err(error) => Some(Err(error)),
                    })
                    .collect::<std::result::Result<HashMap<i64, i64>, _>>()?,
                Err(error) if is_missing_statistics_table(&error) => HashMap::new(),
                Err(error) => return Err(error.into()),
            };
            let bounds_are_exact: bool = sqlx::query_scalar(
                "SELECT NOT EXISTS (
                     SELECT 1 FROM ducklake_delete_file
                     WHERE table_id = $1
                       AND $2 >= begin_snapshot
                       AND ($3 < end_snapshot OR end_snapshot IS NULL)
                 )",
            )
            .bind(table_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .fetch_one(&self.pool)
            .await?;
            let columns = match sqlx::query(
                "SELECT column_id, contains_null, min_value, max_value
                 FROM ducklake_table_column_stats WHERE table_id = $1",
            )
            .bind(table_id)
            .fetch_all(&self.pool)
            .await
            {
                Ok(rows) => rows
                    .into_iter()
                    .map(|row| {
                        let column_id = row.try_get(0)?;
                        Ok(DuckLakeTableColumnStatistics {
                            column_id,
                            contains_null: row.try_get(1)?,
                            min_value: row.try_get(2)?,
                            max_value: row.try_get(3)?,
                            column_size_bytes: column_sizes.get(&column_id).copied(),
                            bounds_are_exact,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?,
                Err(error) if is_missing_statistics_table(&error) => Vec::new(),
                Err(error) => return Err(error.into()),
            };
            Ok(DuckLakeStatistics {
                table,
                columns,
                files: Vec::new(),
            })
        })
    }

    fn get_table_statistics(&self, table_id: i64, snapshot_id: i64) -> Result<DuckLakeStatistics> {
        block_on(async {
            let table = match sqlx::query(
                "SELECT record_count, file_size_bytes
                 FROM ducklake_table_stats WHERE table_id = $1",
            )
            .bind(table_id)
            .fetch_optional(&self.pool)
            .await
            {
                Ok(row) => row
                    .map(|row| {
                        Ok::<_, sqlx::Error>(DuckLakeTableStatistics {
                            record_count: row.try_get(0)?,
                            file_size_bytes: row.try_get(1)?,
                        })
                    })
                    .transpose()?,
                Err(error) if is_missing_statistics_table(&error) => None,
                Err(error) => return Err(error.into()),
            };

            let columns = match sqlx::query(
                "SELECT column_id, contains_null, min_value, max_value
                 FROM ducklake_table_column_stats WHERE table_id = $1",
            )
            .bind(table_id)
            .fetch_all(&self.pool)
            .await
            {
                Ok(rows) => rows
                    .into_iter()
                    .map(|row| {
                        Ok(DuckLakeTableColumnStatistics {
                            column_id: row.try_get(0)?,
                            contains_null: row.try_get(1)?,
                            min_value: row.try_get(2)?,
                            max_value: row.try_get(3)?,
                            column_size_bytes: None,
                            bounds_are_exact: false,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?,
                Err(error) if is_missing_statistics_table(&error) => Vec::new(),
                Err(error) => return Err(error.into()),
            };

            let files = match sqlx::query(
                "SELECT
                    stats.data_file_id,
                    stats.column_id,
                    stats.column_size_bytes,
                    stats.value_count,
                    stats.null_count,
                    stats.min_value,
                    stats.max_value
                 FROM ducklake_file_column_stats AS stats
                 INNER JOIN ducklake_data_file AS data
                    ON data.data_file_id = stats.data_file_id
                    AND data.table_id = stats.table_id
                 WHERE stats.table_id = $1
                   AND $2 >= data.begin_snapshot
                   AND ($3 < data.end_snapshot OR data.end_snapshot IS NULL)",
            )
            .bind(table_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .fetch_all(&self.pool)
            .await
            {
                Ok(rows) => rows
                    .into_iter()
                    .map(|row| {
                        Ok(DuckLakeFileColumnStatistics {
                            data_file_id: row.try_get(0)?,
                            column_id: row.try_get(1)?,
                            column_size_bytes: row.try_get(2)?,
                            value_count: row.try_get(3)?,
                            null_count: row.try_get(4)?,
                            min_value: row.try_get(5)?,
                            max_value: row.try_get(6)?,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?,
                Err(error) if is_missing_statistics_table(&error) => Vec::new(),
                Err(error) => return Err(error.into()),
            };

            Ok(DuckLakeStatistics {
                table,
                columns,
                files,
            })
        })
    }

    fn get_schema_by_name(&self, name: &str, snapshot_id: i64) -> Result<Option<SchemaMetadata>> {
        block_on(async {
            let row = sqlx::query(
                "SELECT s.schema_id, s.schema_name, s.path, s.path_is_relative
                 FROM ducklake_schema s
                 JOIN ducklake_catalog_schema_map m ON m.schema_id = s.schema_id
                 WHERE m.catalog_id = $1
                   AND s.schema_name = $2
                   AND $3 >= s.begin_snapshot
                   AND ($4 < s.end_snapshot OR s.end_snapshot IS NULL)",
            )
            .bind(self.catalog_id)
            .bind(name)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .fetch_optional(&self.pool)
            .await?;

            match row {
                Some(r) => Ok(Some(SchemaMetadata {
                    schema_id: r.try_get(0)?,
                    schema_name: r.try_get(1)?,
                    path: r.try_get(2)?,
                    path_is_relative: r.try_get(3)?,
                })),
                None => Ok(None),
            }
        })
    }

    fn get_table_by_name(
        &self,
        schema_id: i64,
        name: &str,
        snapshot_id: i64,
    ) -> Result<Option<TableMetadata>> {
        // schema_id catalog-scoped by caller.
        block_on(async {
            let row = sqlx::query(
                "SELECT table_id, table_name, path, path_is_relative
                 FROM ducklake_table
                 WHERE schema_id = $1
                   AND table_name = $2
                   AND $3 >= begin_snapshot
                   AND ($4 < end_snapshot OR end_snapshot IS NULL)",
            )
            .bind(schema_id)
            .bind(name)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .fetch_optional(&self.pool)
            .await?;

            match row {
                Some(r) => Ok(Some(TableMetadata {
                    table_id: r.try_get(0)?,
                    table_name: r.try_get(1)?,
                    path: r.try_get(2)?,
                    path_is_relative: r.try_get(3)?,
                })),
                None => Ok(None),
            }
        })
    }

    fn table_exists(&self, schema_id: i64, name: &str, snapshot_id: i64) -> Result<bool> {
        block_on(async {
            let row = sqlx::query(
                "SELECT EXISTS(
                    SELECT 1 FROM ducklake_table
                    WHERE schema_id = $1
                      AND table_name = $2
                      AND $3 >= begin_snapshot
                      AND ($4 < end_snapshot OR end_snapshot IS NULL)
                 )",
            )
            .bind(schema_id)
            .bind(name)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .fetch_one(&self.pool)
            .await?;
            Ok(row.try_get(0)?)
        })
    }

    fn list_all_tables(&self, snapshot_id: i64) -> Result<Vec<TableWithSchema>> {
        block_on(async {
            let rows = sqlx::query(
                "SELECT s.schema_name, t.table_id, t.table_name, t.path, t.path_is_relative
                 FROM ducklake_schema s
                 JOIN ducklake_catalog_schema_map m ON m.schema_id = s.schema_id
                 JOIN ducklake_table t ON s.schema_id = t.schema_id
                 WHERE m.catalog_id = $1
                   AND $2 >= s.begin_snapshot
                   AND ($3 < s.end_snapshot OR s.end_snapshot IS NULL)
                   AND $4 >= t.begin_snapshot
                   AND ($5 < t.end_snapshot OR t.end_snapshot IS NULL)
                 ORDER BY s.schema_name, t.table_name",
            )
            .bind(self.catalog_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .fetch_all(&self.pool)
            .await?;

            rows.into_iter()
                .map(|row| {
                    let schema_name: String = row.try_get(0)?;
                    let table = TableMetadata {
                        table_id: row.try_get(1)?,
                        table_name: row.try_get(2)?,
                        path: row.try_get(3)?,
                        path_is_relative: row.try_get(4)?,
                    };
                    Ok(TableWithSchema {
                        schema_name,
                        table,
                    })
                })
                .collect()
        })
    }

    fn list_all_columns(&self, snapshot_id: i64) -> Result<Vec<ColumnWithTable>> {
        // Note: unlike the single-catalog PostgresMetadataProvider, this filters
        // columns by snapshot range as well — needed for multicatalog because we
        // accumulate column history across catalogs and would otherwise return
        // ended columns alongside current ones.
        block_on(async {
            let rows = sqlx::query(
                "SELECT s.schema_name, t.table_name, c.column_id, c.column_name, c.column_type,
                        c.nulls_allowed, c.parent_column
                 FROM ducklake_schema s
                 JOIN ducklake_catalog_schema_map m ON m.schema_id = s.schema_id
                 JOIN ducklake_table t ON s.schema_id = t.schema_id
                 JOIN ducklake_column c ON t.table_id = c.table_id
                 WHERE m.catalog_id = $1
                   AND $2 >= s.begin_snapshot
                   AND ($3 < s.end_snapshot OR s.end_snapshot IS NULL)
                   AND $4 >= t.begin_snapshot
                   AND ($5 < t.end_snapshot OR t.end_snapshot IS NULL)
                   AND $6 >= c.begin_snapshot
                   AND ($7 < c.end_snapshot OR c.end_snapshot IS NULL)
                 ORDER BY s.schema_name, t.table_name, c.column_order",
            )
            .bind(self.catalog_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .fetch_all(&self.pool)
            .await?;

            let raw: Result<Vec<(ColumnWithTable, Option<i64>)>> = rows
                .into_iter()
                .map(|row| {
                    let schema_name: String = row.try_get(0)?;
                    let table_name: String = row.try_get(1)?;
                    let nulls_allowed: Option<bool> = row.try_get(5)?;
                    let parent_column: Option<i64> = row.try_get(6)?;
                    let column = DuckLakeTableColumn {
                        column_id: row.try_get(2)?,
                        column_name: row.try_get(3)?,
                        column_type: row.try_get(4)?,
                        is_nullable: nulls_allowed.unwrap_or(true),
                    };
                    Ok((
                        ColumnWithTable {
                            schema_name,
                            table_name,
                            column,
                        },
                        parent_column,
                    ))
                })
                .collect();
            Ok(reconstruct_list_columns_with_table(raw?))
        })
    }

    fn list_all_files(&self, snapshot_id: i64) -> Result<Vec<FileWithTable>> {
        block_on(async {
            let rows = sqlx::query(
                "SELECT
                    s.schema_name,
                    t.table_name,
                    data.data_file_id,
                    data.path AS data_file_path,
                    data.path_is_relative AS data_path_is_relative,
                    data.file_size_bytes AS data_file_size,
                    data.footer_size AS data_footer_size,
                    data.encryption_key AS data_encryption_key,
                    del.delete_file_id,
                    del.path AS delete_file_path,
                    del.path_is_relative AS delete_path_is_relative,
                    del.file_size_bytes AS delete_file_size,
                    del.footer_size AS delete_footer_size,
                    del.encryption_key AS delete_encryption_key,
                    del.delete_count
                FROM ducklake_schema s
                JOIN ducklake_catalog_schema_map m ON m.schema_id = s.schema_id
                JOIN ducklake_table t ON s.schema_id = t.schema_id
                JOIN ducklake_data_file data ON t.table_id = data.table_id
                LEFT JOIN ducklake_delete_file del
                    ON data.data_file_id = del.data_file_id
                    AND del.table_id = t.table_id
                    AND $1 >= del.begin_snapshot
                    AND ($2 < del.end_snapshot OR del.end_snapshot IS NULL)
                WHERE m.catalog_id = $3
                  AND $4 >= s.begin_snapshot
                  AND ($5 < s.end_snapshot OR s.end_snapshot IS NULL)
                  AND $6 >= t.begin_snapshot
                  AND ($7 < t.end_snapshot OR t.end_snapshot IS NULL)
                  AND $8 >= data.begin_snapshot
                  AND ($9 < data.end_snapshot OR data.end_snapshot IS NULL)
                ORDER BY s.schema_name, t.table_name, data.path",
            )
            .bind(snapshot_id)
            .bind(snapshot_id)
            .bind(self.catalog_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .fetch_all(&self.pool)
            .await?;

            rows.into_iter()
                .map(|row| {
                    let data_file = DuckLakeFileData {
                        path: row.try_get(3)?,
                        path_is_relative: row.try_get(4)?,
                        file_size_bytes: row.try_get(5)?,
                        footer_size: row.try_get(6)?,
                        encryption_key: row.try_get(7)?,
                    };
                    let delete_file = if row.try_get::<Option<i64>, _>(8)?.is_some() {
                        Some(DuckLakeFileData {
                            path: row.try_get(9)?,
                            path_is_relative: row.try_get(10)?,
                            file_size_bytes: row.try_get(11)?,
                            footer_size: row.try_get(12)?,
                            encryption_key: row.try_get(13)?,
                        })
                    } else {
                        None
                    };
                    Ok(FileWithTable {
                        schema_name: row.try_get(0)?,
                        table_name: row.try_get(1)?,
                        file: DuckLakeTableFile {
                            data_file_id: row.try_get(2)?,
                            file: data_file,
                            delete_file_id: row.try_get(8)?,
                            delete_file,
                            row_id_start: None,
                            snapshot_id: None,
                            begin_snapshot: None,
                            schema_version: None,
                            partial_max: None,
                            max_row_count: row.try_get(14)?,
                            delete_count: None,
                        },
                    })
                })
                .collect()
        })
    }

    fn get_data_files_added_between_snapshots(
        &self,
        table_id: i64,
        start_snapshot: i64,
        end_snapshot: i64,
    ) -> Result<Vec<DataFileChange>> {
        // CDC inherits catalog via table_id. No additional scoping.
        block_on(async {
            let rows = sqlx::query(
                "SELECT
                    data.begin_snapshot,
                    data.path,
                    data.path_is_relative,
                    data.file_size_bytes,
                    data.footer_size,
                    data.encryption_key,
                    data.row_id_start
                FROM ducklake_data_file AS data
                WHERE data.table_id = $1
                  AND data.begin_snapshot > $2
                  AND data.begin_snapshot <= $3
                ORDER BY data.begin_snapshot",
            )
            .bind(table_id)
            .bind(start_snapshot)
            .bind(end_snapshot)
            .fetch_all(&self.pool)
            .await?;

            rows.into_iter()
                .map(|row| {
                    Ok(DataFileChange {
                        begin_snapshot: row.try_get(0)?,
                        path: row.try_get(1)?,
                        path_is_relative: row.try_get(2)?,
                        file_size_bytes: row.try_get(3)?,
                        footer_size: row.try_get(4)?,
                        encryption_key: row.try_get(5)?,
                        row_id_start: row.try_get(6)?,
                    })
                })
                .collect()
        })
    }

    fn get_delete_files_added_between_snapshots(
        &self,
        table_id: i64,
        start_snapshot: i64,
        end_snapshot: i64,
    ) -> Result<Vec<DeleteFileChange>> {
        // Same shape as the single-catalog PostgresMetadataProvider — inherits
        // catalog via table_id.
        block_on(async {
            let rows = sqlx::query(
                r#"
WITH current_delete AS (
    SELECT
        ddf.data_file_id,
        ddf.begin_snapshot,
        ddf.path,
        ddf.path_is_relative,
        ddf.file_size_bytes,
        ddf.footer_size,
        ddf.encryption_key
    FROM ducklake_delete_file ddf
    WHERE ddf.table_id = $1
      AND ddf.begin_snapshot > $2
      AND ddf.begin_snapshot <= $3
),
data_files AS (
    SELECT df.*
    FROM ducklake_data_file df
    WHERE df.table_id = $1
)
SELECT
    data.path, data.path_is_relative, data.file_size_bytes, data.footer_size,
    data.row_id_start, data.record_count, data.mapping_id,
    current_delete.path, current_delete.path_is_relative,
    current_delete.file_size_bytes, current_delete.footer_size,
    prev.path, prev.path_is_relative, prev.file_size_bytes, prev.footer_size,
    current_delete.begin_snapshot
FROM current_delete
JOIN data_files data USING (data_file_id)
LEFT JOIN LATERAL (
    SELECT ddf.path, ddf.path_is_relative, ddf.file_size_bytes, ddf.footer_size
    FROM ducklake_delete_file ddf
    WHERE ddf.table_id = $1
      AND ddf.data_file_id = current_delete.data_file_id
      AND ddf.begin_snapshot < current_delete.begin_snapshot
    ORDER BY ddf.begin_snapshot DESC
    LIMIT 1
) prev ON true
UNION ALL
SELECT
    data.path, data.path_is_relative, data.file_size_bytes, data.footer_size,
    data.row_id_start, data.record_count, data.mapping_id,
    NULL::VARCHAR, NULL::BOOLEAN, NULL::BIGINT, NULL::BIGINT,
    prev.path, prev.path_is_relative, prev.file_size_bytes, prev.footer_size,
    data.end_snapshot
FROM ducklake_data_file data
LEFT JOIN LATERAL (
    SELECT ddf.path, ddf.path_is_relative, ddf.file_size_bytes, ddf.footer_size
    FROM ducklake_delete_file ddf
    WHERE ddf.table_id = $1
      AND ddf.data_file_id = data.data_file_id
      AND ddf.begin_snapshot < data.end_snapshot
    ORDER BY ddf.begin_snapshot DESC
    LIMIT 1
) prev ON true
WHERE data.table_id = $1
  AND data.end_snapshot > $2
  AND data.end_snapshot <= $3
"#,
            )
            .bind(table_id)
            .bind(start_snapshot)
            .bind(end_snapshot)
            .fetch_all(&self.pool)
            .await?;

            rows.into_iter()
                .map(|row| {
                    Ok(DeleteFileChange {
                        data_file_path: row.try_get(0)?,
                        data_file_path_is_relative: row.try_get(1)?,
                        data_file_size_bytes: row.try_get(2)?,
                        data_file_footer_size: row.try_get(3)?,
                        data_row_id_start: row.try_get(4)?,
                        data_record_count: row.try_get(5)?,
                        data_mapping_id: row.try_get(6)?,
                        current_delete_path: row.try_get(7)?,
                        current_delete_path_is_relative: row.try_get(8)?,
                        current_delete_file_size_bytes: row.try_get(9)?,
                        current_delete_footer_size: row.try_get(10)?,
                        previous_delete_path: row.try_get(11)?,
                        previous_delete_path_is_relative: row.try_get(12)?,
                        previous_delete_file_size_bytes: row.try_get(13)?,
                        previous_delete_footer_size: row.try_get(14)?,
                        snapshot_id: row.try_get(15)?,
                    })
                })
                .collect()
        })
    }
}
