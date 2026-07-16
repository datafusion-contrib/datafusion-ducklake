//! SQLite metadata provider for DuckLake catalogs.

use crate::Result;
use crate::metadata_provider::{
    ColumnWithTable, DataFileChange, DeleteFileChange, DuckLakeFileColumnStatistics,
    DuckLakeFileData, DuckLakeFileMetadata, DuckLakeStatistics, DuckLakeTableColumn,
    DuckLakeTableColumnStatistics, DuckLakeTableFile, DuckLakeTableStatistics, FileWithTable,
    MetadataProvider, SchemaMetadata, SnapshotMetadata, TableMetadata, TableWithSchema, block_on,
    reconstruct_list_columns, reconstruct_list_columns_with_table,
};
use arrow::array::{
    ArrayRef, BinaryArray, BooleanArray, Float32Array, Float64Array, Int8Array, Int16Array,
    Int32Array, Int64Array, RecordBatch, UInt8Array, UInt16Array, UInt32Array, UInt64Array,
    new_null_array,
};
use arrow::datatypes::{DataType, SchemaRef};
use sqlx::Row;
use sqlx::sqlite::{SqlitePool, SqlitePoolOptions, SqliteRow};
use sqlx::types::chrono::NaiveDateTime;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Quote a SQL identifier for SQLite (double-quote, doubling embedded quotes),
/// so catalog-supplied inlined-table / column names can't break the query.
fn quote_ident(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

fn decode_table_file(row: &SqliteRow, snapshot_id: i64) -> Result<DuckLakeTableFile> {
    let data_file = DuckLakeFileData {
        path: row.try_get(1)?,
        path_is_relative: row.try_get(2)?,
        file_size_bytes: row.try_get(3)?,
        footer_size: row.try_get(4)?,
        encryption_key: row.try_get(5)?,
    };
    let (delete_file, delete_count) = if row.try_get::<Option<i64>, _>(8)?.is_some() {
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
        file: data_file,
        delete_file_id: row.try_get(8)?,
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

/// Build one Arrow [`RecordBatch`] (in `schema`, the table's physical schema)
/// from inlined rows fetched out of a `ducklake_inlined_data_*` table. `present`
/// is the set of the physical table's data-column names; a table column absent
/// from it (added after this inlined table's schema version) is null-filled.
/// Errors on a column type not yet supported for inlined reads (loud, never
/// silent) — inlined values for those types must be flushed to Parquet first.
fn build_inlined_batch(
    schema: &SchemaRef,
    columns: &[DuckLakeTableColumn],
    present: &HashSet<String>,
    rows: &[sqlx::sqlite::SqliteRow],
) -> Result<RecordBatch> {
    let n = rows.len();
    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(columns.len());
    for (i, col) in columns.iter().enumerate() {
        let dt = schema.field(i).data_type();
        let name = col.column_name.as_str();
        if !present.contains(name) {
            arrays.push(new_null_array(dt, n));
            continue;
        }
        // SQLite stores INTEGER as i64 and REAL as f64; read at that width and
        // narrow/convert to the catalog's declared Arrow type.
        macro_rules! ints {
            ($arr:ty, $t:ty) => {{
                let mut b = Vec::with_capacity(n);
                for r in rows {
                    b.push(r.try_get::<Option<i64>, _>(name)?.map(|v| v as $t));
                }
                Arc::new(<$arr>::from(b)) as ArrayRef
            }};
        }
        let array: ArrayRef = match dt {
            DataType::Int8 => ints!(Int8Array, i8),
            DataType::Int16 => ints!(Int16Array, i16),
            DataType::Int32 => ints!(Int32Array, i32),
            DataType::Int64 => ints!(Int64Array, i64),
            DataType::UInt8 => ints!(UInt8Array, u8),
            DataType::UInt16 => ints!(UInt16Array, u16),
            DataType::UInt32 => ints!(UInt32Array, u32),
            DataType::UInt64 => ints!(UInt64Array, u64),
            DataType::Float32 => {
                let mut b = Vec::with_capacity(n);
                for r in rows {
                    b.push(r.try_get::<Option<f64>, _>(name)?.map(|v| v as f32));
                }
                Arc::new(Float32Array::from(b)) as ArrayRef
            },
            DataType::Float64 => {
                let mut b = Vec::with_capacity(n);
                for r in rows {
                    b.push(r.try_get::<Option<f64>, _>(name)?);
                }
                Arc::new(Float64Array::from(b)) as ArrayRef
            },
            DataType::Utf8 => {
                let mut b: Vec<Option<String>> = Vec::with_capacity(n);
                for r in rows {
                    b.push(r.try_get::<Option<String>, _>(name)?);
                }
                Arc::new(arrow::array::StringArray::from(b)) as ArrayRef
            },
            DataType::Boolean => {
                let mut b = Vec::with_capacity(n);
                for r in rows {
                    b.push(r.try_get::<Option<i64>, _>(name)?.map(|v| v != 0));
                }
                Arc::new(BooleanArray::from(b)) as ArrayRef
            },
            DataType::Binary => {
                let mut b: Vec<Option<Vec<u8>>> = Vec::with_capacity(n);
                for r in rows {
                    b.push(r.try_get::<Option<Vec<u8>>, _>(name)?);
                }
                Arc::new(BinaryArray::from(
                    b.iter().map(|o| o.as_deref()).collect::<Vec<_>>(),
                )) as ArrayRef
            },
            other => {
                return Err(crate::error::DuckLakeError::Unsupported(format!(
                    "inlined data for column '{name}' of type {other:?} is not yet supported; \
                     flush inlined data to Parquet (or disable data inlining at write time)"
                )));
            },
        };
        arrays.push(array);
    }
    Ok(RecordBatch::try_new(schema.clone(), arrays)?)
}

fn is_missing_statistics_table(error: &sqlx::Error) -> bool {
    let message = error.to_string().to_ascii_lowercase();
    message.contains("no such table") || message.contains("does not exist")
}

/// SQLite-based metadata provider for DuckLake catalogs.
#[derive(Debug, Clone)]
pub struct SqliteMetadataProvider {
    pub pool: SqlitePool,
}

impl SqliteMetadataProvider {
    /// Creates a new provider for an existing DuckLake catalog.
    ///
    /// Connection string format: `sqlite:///path/to/catalog.db` or `sqlite::memory:`
    pub async fn new(connection_string: &str) -> Result<Self> {
        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect(connection_string)
            .await?;

        Ok(Self {
            pool,
        })
    }
}

impl MetadataProvider for SqliteMetadataProvider {
    fn get_current_snapshot(&self) -> Result<i64> {
        block_on(async {
            let row = sqlx::query("SELECT COALESCE(MAX(snapshot_id), 0) FROM ducklake_snapshot")
                .fetch_one(&self.pool)
                .await?;
            Ok(row.try_get(0)?)
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
                "SELECT snapshot_id, snapshot_time
                 FROM ducklake_snapshot ORDER BY snapshot_id",
            )
            .fetch_all(&self.pool)
            .await?;

            rows.into_iter()
                .map(|row| {
                    let snapshot_id: i64 = row.try_get(0)?;
                    let timestamp: Option<NaiveDateTime> = row.try_get(1)?;
                    let timestamp_str = timestamp
                        .map(|ts: NaiveDateTime| ts.format("%Y-%m-%d %H:%M:%S%.6f").to_string());

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
                "SELECT schema_id, schema_name, path, path_is_relative FROM ducklake_schema
                 WHERE ? >= begin_snapshot AND (? < end_snapshot OR end_snapshot IS NULL)",
            )
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
        block_on(async {
            let rows = sqlx::query(
                "SELECT table_id, table_name, path, path_is_relative FROM ducklake_table
                 WHERE schema_id = ?
                   AND ? >= begin_snapshot
                   AND (? < end_snapshot OR end_snapshot IS NULL)",
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
        block_on(async {
            let rows = sqlx::query(
                "SELECT column_id, column_name, column_type, nulls_allowed, parent_column
                 FROM ducklake_column
                 WHERE table_id = ?
                   AND ? >= begin_snapshot
                   AND (? < end_snapshot OR end_snapshot IS NULL)
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
        block_on(async {
            // Backward compatibility: minimal / pre-v1.0 catalogs may lack the
            // `partial_max` column and the `ducklake_schema_versions` ledger.
            // Detect both and degrade those projections to NULL so plain reads
            // still work (both are consumed only by compaction; `partial_max`
            // also by time-travel reads of partial files, which such catalogs
            // never contain).
            let has_partial_max: i64 = sqlx::query_scalar(
                "SELECT COUNT(*) FROM pragma_table_info('ducklake_data_file') WHERE name = 'partial_max'",
            )
            .fetch_one(&self.pool)
            .await?;
            let has_schema_versions: i64 = sqlx::query_scalar(
                "SELECT COUNT(*) FROM sqlite_master
                 WHERE type = 'table' AND name = 'ducklake_schema_versions'",
            )
            .fetch_one(&self.pool)
            .await?;
            let partial_max_expr = if has_partial_max > 0 {
                "data.partial_max"
            } else {
                "NULL"
            };
            let schema_version_expr = if has_schema_versions > 0 {
                "(SELECT sv.schema_version
                  FROM ducklake_schema_versions sv
                  WHERE sv.table_id = data.table_id
                    AND sv.begin_snapshot <= data.begin_snapshot
                  ORDER BY sv.begin_snapshot DESC
                  LIMIT 1)"
            } else {
                "NULL"
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
                    data.begin_snapshot AS data_begin_snapshot,
                    {partial_max_expr} AS data_partial_max,
                    {schema_version_expr} AS data_schema_version
                FROM ducklake_data_file AS data
                LEFT JOIN ducklake_delete_file AS del
                    ON data.data_file_id = del.data_file_id
                    AND del.table_id = ?
                    AND ? >= del.begin_snapshot
                    AND (? < del.end_snapshot OR del.end_snapshot IS NULL)
                WHERE data.table_id = ?
                  AND ? >= data.begin_snapshot
                  AND (? < data.end_snapshot OR data.end_snapshot IS NULL)"
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
            let has_partial_max: i64 = sqlx::query_scalar(
                "SELECT COUNT(*) FROM pragma_table_info('ducklake_data_file') WHERE name = 'partial_max'",
            )
            .fetch_one(&self.pool)
            .await?;
            let has_schema_versions: i64 = sqlx::query_scalar(
                "SELECT COUNT(*) FROM sqlite_master
                 WHERE type = 'table' AND name = 'ducklake_schema_versions'",
            )
            .fetch_one(&self.pool)
            .await?;
            let partial_max_expr = if has_partial_max > 0 {
                "data.partial_max"
            } else {
                "NULL"
            };
            let schema_version_expr = if has_schema_versions > 0 {
                "(SELECT sv.schema_version
                  FROM ducklake_schema_versions sv
                  WHERE sv.table_id = data.table_id
                    AND sv.begin_snapshot <= data.begin_snapshot
                  ORDER BY sv.begin_snapshot DESC
                  LIMIT 1)"
            } else {
                "NULL"
            };
            let sql = format!(
                "SELECT
                    data.data_file_id, data.path, data.path_is_relative,
                    data.file_size_bytes, data.footer_size, data.encryption_key,
                    data.row_id_start, data.record_count,
                    del.delete_file_id, del.path, del.path_is_relative,
                    del.file_size_bytes, del.footer_size, del.encryption_key,
                    del.delete_count, data.begin_snapshot,
                    {partial_max_expr}, {schema_version_expr}
                 FROM ducklake_data_file AS data
                 LEFT JOIN ducklake_delete_file AS del
                   ON data.data_file_id = del.data_file_id
                  AND del.table_id = ?
                  AND ? >= del.begin_snapshot
                  AND (? < del.end_snapshot OR del.end_snapshot IS NULL)
                 WHERE data.table_id = ?
                   AND ? >= data.begin_snapshot
                   AND (? < data.end_snapshot OR data.end_snapshot IS NULL)
                   AND data.data_file_id > ?
                 ORDER BY data.data_file_id
                 LIMIT ?"
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
                 WHERE stats.table_id = ?
                   AND ? >= data.begin_snapshot
                   AND (? < data.end_snapshot OR data.end_snapshot IS NULL)
                   AND stats.data_file_id > ?
                   AND stats.data_file_id <= ?
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
                 FROM ducklake_table_stats WHERE table_id = ?",
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
                             WHERE visible.table_id = ?
                               AND ? >= visible.begin_snapshot
                               AND (? < visible.end_snapshot OR visible.end_snapshot IS NULL)
                           )
                          THEN CAST(SUM(stats.column_size_bytes) AS INTEGER)
                        END
                 FROM ducklake_file_column_stats stats
                 INNER JOIN ducklake_data_file data
                   ON data.data_file_id = stats.data_file_id
                  AND data.table_id = stats.table_id
                 WHERE stats.table_id = ?
                   AND ? >= data.begin_snapshot
                   AND (? < data.end_snapshot OR data.end_snapshot IS NULL)
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
                     WHERE table_id = ?
                       AND ? >= begin_snapshot
                       AND (? < end_snapshot OR end_snapshot IS NULL)
                 )",
            )
            .bind(table_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .fetch_one(&self.pool)
            .await?;
            let columns = match sqlx::query(
                "SELECT column_id, contains_null, min_value, max_value
                 FROM ducklake_table_column_stats WHERE table_id = ?",
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
                 FROM ducklake_table_stats WHERE table_id = ?",
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
                 FROM ducklake_table_column_stats WHERE table_id = ?",
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
                 WHERE stats.table_id = ?
                   AND ? >= data.begin_snapshot
                   AND (? < data.end_snapshot OR data.end_snapshot IS NULL)",
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

    fn get_inlined_data(
        &self,
        table_id: i64,
        snapshot_id: i64,
        columns: &[DuckLakeTableColumn],
    ) -> Result<Vec<RecordBatch>> {
        block_on(async {
            // Most catalogs have no inlined data — the registry table is absent.
            // Detect and return empty so they (and older catalogs) are unaffected.
            let has_registry: i64 = sqlx::query_scalar(
                "SELECT COUNT(*) FROM sqlite_master
                 WHERE type = 'table' AND name = 'ducklake_inlined_data_tables'",
            )
            .fetch_one(&self.pool)
            .await?;
            if has_registry == 0 {
                return Ok(Vec::new());
            }

            // Every physical inlined table for this table (one per schema version).
            let regs = sqlx::query(
                "SELECT table_name FROM ducklake_inlined_data_tables WHERE table_id = ?",
            )
            .bind(table_id)
            .fetch_all(&self.pool)
            .await?;
            if regs.is_empty() {
                return Ok(Vec::new());
            }

            let schema: SchemaRef = Arc::new(crate::types::build_arrow_schema(columns)?);
            let mut batches = Vec::new();
            for reg in regs {
                let phys: String = reg.try_get("table_name")?;
                // Defensive: only touch tables that look like DuckLake inline tables.
                if !phys.starts_with("ducklake_inlined_data_")
                    || !phys.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
                {
                    continue;
                }

                // Which of the table's columns this inline table physically has
                // (its layout matches the schema version it was created for).
                let info = sqlx::query(&format!(
                    "SELECT name FROM pragma_table_info({})",
                    // pragma wants a string literal; single-quote-escape the name.
                    format_args!("'{}'", phys.replace('\'', "''"))
                ))
                .fetch_all(&self.pool)
                .await?;
                let present: HashSet<String> = info
                    .iter()
                    .filter_map(|r| r.try_get::<String, _>("name").ok())
                    .collect();

                // Project the table columns this inline table actually has; rows
                // visible at the snapshot (this predicate also hides inlined-row
                // deletes, which set end_snapshot). ORDER BY row_id for stability.
                let projected: Vec<String> = columns
                    .iter()
                    .filter(|c| present.contains(c.column_name.as_str()))
                    .map(|c| quote_ident(&c.column_name))
                    .collect();
                let select_list = if projected.is_empty() {
                    "1".to_string()
                } else {
                    projected.join(", ")
                };
                let sql = format!(
                    "SELECT {select_list} FROM {} \
                     WHERE ? >= begin_snapshot AND (? < end_snapshot OR end_snapshot IS NULL) \
                     ORDER BY row_id",
                    quote_ident(&phys)
                );
                let rows = sqlx::query(&sql)
                    .bind(snapshot_id)
                    .bind(snapshot_id)
                    .fetch_all(&self.pool)
                    .await?;
                if rows.is_empty() {
                    continue;
                }
                batches.push(build_inlined_batch(&schema, columns, &present, &rows)?);
            }
            Ok(batches)
        })
    }

    fn get_schema_by_name(&self, name: &str, snapshot_id: i64) -> Result<Option<SchemaMetadata>> {
        block_on(async {
            let row = sqlx::query(
                "SELECT schema_id, schema_name, path, path_is_relative FROM ducklake_schema
                 WHERE schema_name = ?
                   AND ? >= begin_snapshot
                   AND (? < end_snapshot OR end_snapshot IS NULL)",
            )
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
        block_on(async {
            let row = sqlx::query(
                "SELECT table_id, table_name, path, path_is_relative FROM ducklake_table
                 WHERE schema_id = ?
                   AND table_name = ?
                   AND ? >= begin_snapshot
                   AND (? < end_snapshot OR end_snapshot IS NULL)",
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
                "SELECT COUNT(*) FROM ducklake_table
                 WHERE schema_id = ?
                   AND table_name = ?
                   AND ? >= begin_snapshot
                   AND (? < end_snapshot OR end_snapshot IS NULL)",
            )
            .bind(schema_id)
            .bind(name)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .fetch_one(&self.pool)
            .await?;

            let count: i64 = row.try_get(0)?;
            Ok(count > 0)
        })
    }

    fn list_all_tables(&self, snapshot_id: i64) -> Result<Vec<TableWithSchema>> {
        block_on(async {
            let rows = sqlx::query(
                "SELECT s.schema_name, t.table_id, t.table_name, t.path, t.path_is_relative
                 FROM ducklake_schema s
                 JOIN ducklake_table t ON s.schema_id = t.schema_id
                 WHERE ? >= s.begin_snapshot
                   AND (? < s.end_snapshot OR s.end_snapshot IS NULL)
                   AND ? >= t.begin_snapshot
                   AND (? < t.end_snapshot OR t.end_snapshot IS NULL)
                 ORDER BY s.schema_name, t.table_name",
            )
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
        block_on(async {
            let rows = sqlx::query(
                "SELECT s.schema_name, t.table_name, c.column_id, c.column_name, c.column_type, c.nulls_allowed, c.parent_column
                 FROM ducklake_schema s
                 JOIN ducklake_table t ON s.schema_id = t.schema_id
                 JOIN ducklake_column c ON t.table_id = c.table_id
                 WHERE ? >= s.begin_snapshot
                   AND (? < s.end_snapshot OR s.end_snapshot IS NULL)
                   AND ? >= t.begin_snapshot
                   AND (? < t.end_snapshot OR t.end_snapshot IS NULL)
                   AND ? >= c.begin_snapshot
                   AND (? < c.end_snapshot OR c.end_snapshot IS NULL)
                 ORDER BY s.schema_name, t.table_name, c.column_order",
            )
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
                JOIN ducklake_table t ON s.schema_id = t.schema_id
                JOIN ducklake_data_file data ON t.table_id = data.table_id
                LEFT JOIN ducklake_delete_file del
                    ON data.data_file_id = del.data_file_id
                    AND del.table_id = t.table_id
                    AND ? >= del.begin_snapshot
                    AND (? < del.end_snapshot OR del.end_snapshot IS NULL)
                WHERE ? >= s.begin_snapshot
                  AND (? < s.end_snapshot OR s.end_snapshot IS NULL)
                  AND ? >= t.begin_snapshot
                  AND (? < t.end_snapshot OR t.end_snapshot IS NULL)
                  AND ? >= data.begin_snapshot
                  AND (? < data.end_snapshot OR data.end_snapshot IS NULL)
                ORDER BY s.schema_name, t.table_name, data.path",
            )
            .bind(snapshot_id)
            .bind(snapshot_id)
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
                WHERE data.table_id = ?
                  AND data.begin_snapshot >= ?
                  AND data.begin_snapshot <= ?
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
        block_on(async {
            // SQLite doesn't support LATERAL JOIN, so we use correlated subqueries instead
            // This query has two parts:
            // 1. Incremental deletes: delete files added in the snapshot range
            // 2. Full file deletes: data files that were completely removed in the snapshot range
            let rows = sqlx::query(
                r#"
-- Part 1: Incremental deletes (delete file added)
SELECT
    data.path AS data_path,
    data.path_is_relative AS data_path_is_relative,
    data.file_size_bytes AS data_file_size,
    data.footer_size AS data_footer_size,
    data.row_id_start,
    data.record_count,
    data.mapping_id,

    cd.path AS current_delete_path,
    cd.path_is_relative AS current_delete_path_is_relative,
    cd.file_size_bytes AS current_delete_file_size,
    cd.footer_size AS current_delete_footer_size,

    -- Previous delete file (correlated subquery instead of LATERAL)
    (SELECT path FROM ducklake_delete_file pd
     WHERE pd.table_id = ?
       AND pd.data_file_id = cd.data_file_id
       AND pd.begin_snapshot < cd.begin_snapshot
     ORDER BY pd.begin_snapshot DESC LIMIT 1) AS prev_delete_path,
    (SELECT path_is_relative FROM ducklake_delete_file pd
     WHERE pd.table_id = ?
       AND pd.data_file_id = cd.data_file_id
       AND pd.begin_snapshot < cd.begin_snapshot
     ORDER BY pd.begin_snapshot DESC LIMIT 1) AS prev_delete_path_is_relative,
    (SELECT file_size_bytes FROM ducklake_delete_file pd
     WHERE pd.table_id = ?
       AND pd.data_file_id = cd.data_file_id
       AND pd.begin_snapshot < cd.begin_snapshot
     ORDER BY pd.begin_snapshot DESC LIMIT 1) AS prev_delete_file_size,
    (SELECT footer_size FROM ducklake_delete_file pd
     WHERE pd.table_id = ?
       AND pd.data_file_id = cd.data_file_id
       AND pd.begin_snapshot < cd.begin_snapshot
     ORDER BY pd.begin_snapshot DESC LIMIT 1) AS prev_delete_footer_size,

    cd.begin_snapshot AS snapshot_id
FROM ducklake_delete_file cd
JOIN ducklake_data_file data ON data.data_file_id = cd.data_file_id
WHERE cd.table_id = ?
  AND cd.begin_snapshot >= ?
  AND cd.begin_snapshot <= ?
  AND data.table_id = ?

UNION ALL

-- Part 2: Full file deletes (data file removed entirely)
SELECT
    data.path AS data_path,
    data.path_is_relative AS data_path_is_relative,
    data.file_size_bytes AS data_file_size,
    data.footer_size AS data_footer_size,
    data.row_id_start,
    data.record_count,
    data.mapping_id,

    NULL AS current_delete_path,
    NULL AS current_delete_path_is_relative,
    NULL AS current_delete_file_size,
    NULL AS current_delete_footer_size,

    -- Previous delete file
    (SELECT path FROM ducklake_delete_file pd
     WHERE pd.table_id = ?
       AND pd.data_file_id = data.data_file_id
       AND pd.begin_snapshot < data.end_snapshot
     ORDER BY pd.begin_snapshot DESC LIMIT 1) AS prev_delete_path,
    (SELECT path_is_relative FROM ducklake_delete_file pd
     WHERE pd.table_id = ?
       AND pd.data_file_id = data.data_file_id
       AND pd.begin_snapshot < data.end_snapshot
     ORDER BY pd.begin_snapshot DESC LIMIT 1) AS prev_delete_path_is_relative,
    (SELECT file_size_bytes FROM ducklake_delete_file pd
     WHERE pd.table_id = ?
       AND pd.data_file_id = data.data_file_id
       AND pd.begin_snapshot < data.end_snapshot
     ORDER BY pd.begin_snapshot DESC LIMIT 1) AS prev_delete_file_size,
    (SELECT footer_size FROM ducklake_delete_file pd
     WHERE pd.table_id = ?
       AND pd.data_file_id = data.data_file_id
       AND pd.begin_snapshot < data.end_snapshot
     ORDER BY pd.begin_snapshot DESC LIMIT 1) AS prev_delete_footer_size,

    data.end_snapshot AS snapshot_id
FROM ducklake_data_file data
WHERE data.table_id = ?
  AND data.end_snapshot >= ?
  AND data.end_snapshot <= ?
"#,
            )
            // Part 1 bindings: 4x table_id for prev subqueries, table_id for cd, start, end, table_id for data
            .bind(table_id)
            .bind(table_id)
            .bind(table_id)
            .bind(table_id)
            .bind(table_id)
            .bind(start_snapshot)
            .bind(end_snapshot)
            .bind(table_id)
            // Part 2 bindings: 4x table_id for prev subqueries, table_id for data, start, end
            .bind(table_id)
            .bind(table_id)
            .bind(table_id)
            .bind(table_id)
            .bind(table_id)
            .bind(start_snapshot)
            .bind(end_snapshot)
            .fetch_all(&self.pool)
            .await?;

            rows.into_iter()
                .map(|row| {
                    Ok(DeleteFileChange {
                        // data file
                        data_file_path: row.try_get(0)?,
                        data_file_path_is_relative: row.try_get(1)?,
                        data_file_size_bytes: row.try_get(2)?,
                        data_file_footer_size: row.try_get(3)?,
                        data_row_id_start: row.try_get(4)?,
                        data_record_count: row.try_get(5)?,
                        data_mapping_id: row.try_get(6)?,

                        // current delete
                        current_delete_path: row.try_get(7)?,
                        current_delete_path_is_relative: row.try_get(8)?,
                        current_delete_file_size_bytes: row.try_get(9)?,
                        current_delete_footer_size: row.try_get(10)?,

                        // previous delete
                        previous_delete_path: row.try_get(11)?,
                        previous_delete_path_is_relative: row.try_get(12)?,
                        previous_delete_file_size_bytes: row.try_get(13)?,
                        previous_delete_footer_size: row.try_get(14)?,

                        // snapshot
                        snapshot_id: row.try_get(15)?,
                    })
                })
                .collect()
        })
    }
}
