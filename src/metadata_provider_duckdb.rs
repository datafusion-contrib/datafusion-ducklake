use crate::DuckLakeError;
use crate::metadata_provider::{
    ColumnWithTable, DataFileChange, DeleteFileChange, DuckLakeFileColumnStatistics,
    DuckLakeFileData, DuckLakeFileMetadata, DuckLakeStatistics, DuckLakeTableColumn,
    DuckLakeTableColumnStatistics, DuckLakeTableFile, DuckLakeTableStatistics, FileWithTable,
    MetadataProvider, SQL_GET_DATA_FILES, SQL_GET_DATA_FILES_ADDED_BETWEEN_SNAPSHOTS,
    SQL_GET_DATA_PATH, SQL_GET_DELETE_FILES_ADDED_BETWEEN_SNAPSHOTS, SQL_GET_FILE_COLUMN_STATS,
    SQL_GET_LATEST_SNAPSHOT, SQL_GET_SCHEMA_BY_NAME, SQL_GET_TABLE_BY_NAME,
    SQL_GET_TABLE_COLUMN_STATS, SQL_GET_TABLE_COLUMNS, SQL_GET_TABLE_STATS, SQL_LIST_ALL_COLUMNS,
    SQL_LIST_ALL_FILES, SQL_LIST_ALL_TABLES, SQL_LIST_SCHEMAS, SQL_LIST_SNAPSHOTS, SQL_LIST_TABLES,
    SQL_TABLE_EXISTS, SchemaMetadata, SnapshotMetadata, TableMetadata, TableWithSchema,
    reconstruct_list_columns, reconstruct_list_columns_with_table,
};
use duckdb::AccessMode::ReadOnly;
use duckdb::{Config, Connection, params};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard};

fn is_missing_statistics_table(error: &duckdb::Error) -> bool {
    let message = error.to_string().to_ascii_lowercase();
    message.contains("does not exist") || message.contains("not found")
}

/// DuckDB metadata provider
///
/// Uses a single shared connection protected by a Mutex to avoid
/// the overhead of creating a new connection for each metadata query.
/// This is safe for read-only operations.
#[derive(Debug, Clone)]
pub struct DuckdbMetadataProvider {
    conn: Arc<Mutex<Connection>>,
    /// Path to the catalog database, retained for logging/debugging
    #[allow(dead_code)]
    catalog_path: String,
}

impl DuckdbMetadataProvider {
    /// Create a new DuckDB metadata provider
    pub fn new(catalog_path: impl Into<String>) -> crate::Result<Self> {
        let catalog_path = catalog_path.into();
        let conn = Self::create_connection(&catalog_path)?;

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            catalog_path,
        })
    }

    /// Get a reference to the shared connection
    fn connection(&self) -> MutexGuard<'_, Connection> {
        self.conn.lock().expect("DuckDB connection mutex poisoned")
    }

    /// Create a new read-only connection to the catalog database
    fn create_connection(catalog_path: &str) -> crate::Result<Connection> {
        let config = Config::default().access_mode(ReadOnly)?;
        match Connection::open_with_flags(catalog_path, config) {
            Ok(con) => Ok(con),
            Err(msg)
                if msg
                    .to_string()
                    .starts_with("IO Error: Could not set lock on file") =>
            {
                tracing::warn!(
                    error = %msg,
                    "DuckDB file likely already open in write mode. Cannot connect"
                );
                Err(DuckLakeError::DuckDb(msg))
            },
            Err(msg) => {
                tracing::error!(error = %msg, "Failed to open DuckDB catalog");
                Err(DuckLakeError::DuckDb(msg))
            },
        }
    }
}

impl MetadataProvider for DuckdbMetadataProvider {
    fn get_current_snapshot(&self) -> crate::Result<i64> {
        let conn = self.connection();
        let snapshot_id: i64 = conn.query_row(SQL_GET_LATEST_SNAPSHOT, [], |row| row.get(0))?;
        Ok(snapshot_id)
    }

    fn get_data_path(&self) -> crate::Result<String> {
        let conn = self.connection();
        let data_path: String = conn.query_row(SQL_GET_DATA_PATH, [], |row| row.get(0))?;
        Ok(data_path)
    }

    fn list_snapshots(&self) -> crate::Result<Vec<SnapshotMetadata>> {
        let conn = self.connection();
        let mut stmt = conn.prepare(SQL_LIST_SNAPSHOTS)?;

        let snapshots = stmt
            .query_map([], |row| {
                let snapshot_id: i64 = row.get(0)?;
                let timestamp: Option<String> = row.get(1)?;
                Ok(SnapshotMetadata {
                    snapshot_id,
                    timestamp,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(snapshots)
    }

    fn list_schemas(&self, snapshot_id: i64) -> crate::Result<Vec<SchemaMetadata>> {
        let conn = self.connection();
        let mut stmt = conn.prepare(SQL_LIST_SCHEMAS)?;

        let schemas = stmt
            .query_map([snapshot_id, snapshot_id], |row| {
                let schema_id: i64 = row.get(0)?;
                let schema_name: String = row.get(1)?;
                let path: String = row.get(2)?;
                let path_is_relative: bool = row.get(3)?;
                Ok(SchemaMetadata {
                    schema_id,
                    schema_name,
                    path,
                    path_is_relative,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(schemas)
    }

    fn list_tables(&self, schema_id: i64, snapshot_id: i64) -> crate::Result<Vec<TableMetadata>> {
        let conn = self.connection();
        let mut stmt = conn.prepare(SQL_LIST_TABLES)?;

        let tables = stmt
            .query_map([schema_id, snapshot_id, snapshot_id], |row| {
                let table_id: i64 = row.get(0)?;
                let table_name: String = row.get(1)?;
                let path: String = row.get(2)?;
                let path_is_relative: bool = row.get(3)?;
                Ok(TableMetadata {
                    table_id,
                    table_name,
                    path,
                    path_is_relative,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(tables)
    }

    fn get_table_structure(
        &self,
        table_id: i64,
        snapshot_id: i64,
    ) -> crate::Result<Vec<DuckLakeTableColumn>> {
        let conn = self.connection();
        let mut stmt = conn.prepare(SQL_GET_TABLE_COLUMNS)?;

        let raw_columns: Vec<(DuckLakeTableColumn, Option<i64>)> = stmt
            .query_map(duckdb::params![table_id, snapshot_id, snapshot_id], |row| {
                let column_id: i64 = row.get(0)?;
                let column_name: String = row.get(1)?;
                let column_type: String = row.get(2)?;
                let nulls_allowed: Option<bool> = row.get(3)?;
                let parent_column: Option<i64> = row.get(4)?;
                Ok((
                    DuckLakeTableColumn::new(
                        column_id,
                        column_name,
                        column_type,
                        nulls_allowed.unwrap_or(true),
                    ),
                    parent_column,
                ))
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(reconstruct_list_columns(raw_columns))
    }

    fn get_table_files_for_select(
        &self,
        table_id: i64,
        snapshot_id: i64,
    ) -> crate::Result<Vec<DuckLakeTableFile>> {
        let conn = self.connection();
        let mut stmt = conn.prepare(SQL_GET_DATA_FILES)?;

        let files = stmt
            .query_map(
                [table_id, snapshot_id, snapshot_id, table_id, snapshot_id, snapshot_id],
                |row| {
                    // Parse data file (columns 0-7)
                    let data_file_id: i64 = row.get(0)?;
                    let data_file = DuckLakeFileData {
                        path: row.get(1)?,
                        path_is_relative: row.get(2)?,
                        file_size_bytes: row.get(3)?,
                        footer_size: row.get(4)?,
                        encryption_key: row.get(5)?,
                    };
                    let row_id_start: Option<i64> = row.get(6)?;
                    let record_count: Option<i64> = row.get(7)?;

                    // Parse delete file (columns 8-14) if exists
                    let (delete_file, delete_count, delete_file_id) =
                        if let Ok(Some(dfid)) = row.get::<_, Option<i64>>(8) {
                            (
                                Some(DuckLakeFileData {
                                    path: row.get(9)?,
                                    path_is_relative: row.get(10)?,
                                    file_size_bytes: row.get(11)?,
                                    footer_size: row.get(12)?,
                                    encryption_key: row.get(13)?,
                                }),
                                row.get(14)?,
                                Some(dfid),
                            )
                        } else {
                            (None, None, None)
                        };

                    Ok(DuckLakeTableFile {
                        data_file_id,
                        file: data_file,
                        delete_file_id,
                        delete_file,
                        row_id_start,
                        snapshot_id: Some(snapshot_id),
                        begin_snapshot: None,
                        schema_version: None,
                        partial_max: None,
                        max_row_count: record_count,
                        delete_count,
                    })
                },
            )?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(files)
    }

    fn get_table_summary_statistics(
        &self,
        table_id: i64,
        snapshot_id: i64,
    ) -> crate::Result<DuckLakeStatistics> {
        let conn = self.connection();
        let table = match conn.prepare(SQL_GET_TABLE_STATS) {
            Ok(mut stmt) => {
                let mut rows = stmt.query([table_id])?;
                rows.next()?
                    .map(|row| {
                        Ok::<_, duckdb::Error>(DuckLakeTableStatistics {
                            record_count: row.get(0)?,
                            file_size_bytes: row.get(1)?,
                        })
                    })
                    .transpose()?
            },
            Err(error) if is_missing_statistics_table(&error) => None,
            Err(error) => return Err(error.into()),
        };
        let column_sizes: HashMap<i64, i64> = match conn.prepare(
            "SELECT stats.column_id,
                    CASE
                      WHEN COUNT(*) = COUNT(stats.column_size_bytes)
                       AND COUNT(*) = (
                         SELECT COUNT(*) FROM ducklake_data_file visible
                         WHERE visible.table_id = ?
                           AND ? >= visible.begin_snapshot
                           AND (? < visible.end_snapshot OR visible.end_snapshot IS NULL)
                       )
                      THEN CAST(SUM(stats.column_size_bytes) AS BIGINT)
                    END
             FROM ducklake_file_column_stats stats
             INNER JOIN ducklake_data_file data
               ON data.data_file_id = stats.data_file_id
              AND data.table_id = stats.table_id
             WHERE stats.table_id = ?
               AND ? >= data.begin_snapshot
               AND (? < data.end_snapshot OR data.end_snapshot IS NULL)
             GROUP BY stats.column_id",
        ) {
            Ok(mut stmt) => stmt
                .query_map(
                    params![table_id, snapshot_id, snapshot_id, table_id, snapshot_id, snapshot_id],
                    |row| Ok((row.get::<_, i64>(0)?, row.get::<_, Option<i64>>(1)?)),
                )?
                .filter_map(|row| match row {
                    Ok((column_id, Some(size))) => Some(Ok((column_id, size))),
                    Ok((_, None)) => None,
                    Err(error) => Some(Err(error)),
                })
                .collect::<Result<_, _>>()?,
            Err(error) if is_missing_statistics_table(&error) => HashMap::new(),
            Err(error) => return Err(error.into()),
        };
        let bounds_are_exact: bool = conn.query_row(
            "SELECT NOT EXISTS (
                 SELECT 1 FROM ducklake_delete_file
                 WHERE table_id = ?
                   AND ? >= begin_snapshot
                   AND (? < end_snapshot OR end_snapshot IS NULL)
             )",
            params![table_id, snapshot_id, snapshot_id],
            |row| row.get(0),
        )?;
        let columns = match conn.prepare(SQL_GET_TABLE_COLUMN_STATS) {
            Ok(mut stmt) => stmt
                .query_map([table_id], |row| {
                    let column_id = row.get(0)?;
                    Ok(DuckLakeTableColumnStatistics {
                        column_id,
                        contains_null: row.get(1)?,
                        min_value: row.get(2)?,
                        max_value: row.get(3)?,
                        column_size_bytes: column_sizes.get(&column_id).copied(),
                        bounds_are_exact,
                    })
                })?
                .collect::<Result<Vec<_>, _>>()?,
            Err(error) if is_missing_statistics_table(&error) => Vec::new(),
            Err(error) => return Err(error.into()),
        };
        Ok(DuckLakeStatistics {
            table,
            columns,
            files: Vec::new(),
        })
    }

    fn get_table_file_metadata_page(
        &self,
        table_id: i64,
        snapshot_id: i64,
        after_data_file_id: Option<i64>,
        limit: usize,
    ) -> crate::Result<Vec<DuckLakeFileMetadata>> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let limit = i64::try_from(limit).map_err(|_| {
            crate::DuckLakeError::InvalidConfig("file metadata page limit exceeds i64".to_string())
        })?;
        let conn = self.connection();
        let sql = format!(
            "{SQL_GET_DATA_FILES}
             AND data.data_file_id > ?
             ORDER BY data.data_file_id
             LIMIT ?"
        );
        let mut statement = conn.prepare(&sql)?;
        let files = statement
            .query_map(
                params![
                    table_id,
                    snapshot_id,
                    snapshot_id,
                    table_id,
                    snapshot_id,
                    snapshot_id,
                    after_data_file_id.unwrap_or(i64::MIN),
                    limit
                ],
                |row| {
                    let delete_file_id: Option<i64> = row.get(8)?;
                    let (delete_file, delete_count) = if delete_file_id.is_some() {
                        (
                            Some(DuckLakeFileData {
                                path: row.get(9)?,
                                path_is_relative: row.get(10)?,
                                file_size_bytes: row.get(11)?,
                                footer_size: row.get(12)?,
                                encryption_key: row.get(13)?,
                            }),
                            row.get(14)?,
                        )
                    } else {
                        (None, None)
                    };
                    Ok(DuckLakeTableFile {
                        data_file_id: row.get(0)?,
                        file: DuckLakeFileData {
                            path: row.get(1)?,
                            path_is_relative: row.get(2)?,
                            file_size_bytes: row.get(3)?,
                            footer_size: row.get(4)?,
                            encryption_key: row.get(5)?,
                        },
                        delete_file_id,
                        delete_file,
                        row_id_start: row.get(6)?,
                        snapshot_id: Some(snapshot_id),
                        begin_snapshot: None,
                        schema_version: None,
                        partial_max: None,
                        max_row_count: row.get(7)?,
                        delete_count,
                    })
                },
            )?
            .collect::<Result<Vec<_>, _>>()?;
        let Some(last_data_file_id) = files.last().map(|file| file.data_file_id) else {
            return Ok(Vec::new());
        };
        let statistics = match conn.prepare(
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
        ) {
            Ok(mut statement) => statement
                .query_map(
                    params![
                        table_id,
                        snapshot_id,
                        snapshot_id,
                        after_data_file_id.unwrap_or(i64::MIN),
                        last_data_file_id
                    ],
                    |row| {
                        Ok(DuckLakeFileColumnStatistics {
                            data_file_id: row.get(0)?,
                            column_id: row.get(1)?,
                            column_size_bytes: row.get(2)?,
                            value_count: row.get(3)?,
                            null_count: row.get(4)?,
                            min_value: row.get(5)?,
                            max_value: row.get(6)?,
                        })
                    },
                )?
                .collect::<Result<Vec<_>, _>>()?,
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
    }

    fn get_table_statistics(
        &self,
        table_id: i64,
        snapshot_id: i64,
    ) -> crate::Result<DuckLakeStatistics> {
        let conn = self.connection();

        let table = match conn.prepare(SQL_GET_TABLE_STATS) {
            Ok(mut stmt) => {
                let mut rows = stmt.query([table_id])?;
                rows.next()?
                    .map(|row| {
                        Ok::<_, duckdb::Error>(DuckLakeTableStatistics {
                            record_count: row.get(0)?,
                            file_size_bytes: row.get(1)?,
                        })
                    })
                    .transpose()?
            },
            Err(error) if is_missing_statistics_table(&error) => None,
            Err(error) => return Err(error.into()),
        };

        let columns = match conn.prepare(SQL_GET_TABLE_COLUMN_STATS) {
            Ok(mut stmt) => stmt
                .query_map([table_id], |row| {
                    Ok(DuckLakeTableColumnStatistics {
                        column_id: row.get(0)?,
                        contains_null: row.get(1)?,
                        min_value: row.get(2)?,
                        max_value: row.get(3)?,
                        column_size_bytes: None,
                        bounds_are_exact: false,
                    })
                })?
                .collect::<Result<Vec<_>, _>>()?,
            Err(error) if is_missing_statistics_table(&error) => Vec::new(),
            Err(error) => return Err(error.into()),
        };

        let files = match conn.prepare(SQL_GET_FILE_COLUMN_STATS) {
            Ok(mut stmt) => stmt
                .query_map([table_id, snapshot_id, snapshot_id], |row| {
                    Ok(DuckLakeFileColumnStatistics {
                        data_file_id: row.get(0)?,
                        column_id: row.get(1)?,
                        column_size_bytes: row.get(2)?,
                        value_count: row.get(3)?,
                        null_count: row.get(4)?,
                        min_value: row.get(5)?,
                        max_value: row.get(6)?,
                    })
                })?
                .collect::<Result<Vec<_>, _>>()?,
            Err(error) if is_missing_statistics_table(&error) => Vec::new(),
            Err(error) => return Err(error.into()),
        };

        Ok(DuckLakeStatistics {
            table,
            columns,
            files,
        })
    }

    fn get_schema_by_name(
        &self,
        name: &str,
        snapshot_id: i64,
    ) -> crate::Result<Option<SchemaMetadata>> {
        let conn = self.connection();
        let mut stmt = conn.prepare(SQL_GET_SCHEMA_BY_NAME)?;

        let mut rows = stmt.query(params![name, snapshot_id, snapshot_id])?;

        if let Some(row) = rows.next()? {
            let schema_id: i64 = row.get(0)?;
            let schema_name: String = row.get(1)?;
            let path: String = row.get(2)?;
            let path_is_relative: bool = row.get(3)?;
            Ok(Some(SchemaMetadata {
                schema_id,
                schema_name,
                path,
                path_is_relative,
            }))
        } else {
            Ok(None)
        }
    }

    fn get_table_by_name(
        &self,
        schema_id: i64,
        name: &str,
        snapshot_id: i64,
    ) -> crate::Result<Option<TableMetadata>> {
        let conn = self.connection();
        let mut stmt = conn.prepare(SQL_GET_TABLE_BY_NAME)?;

        let mut rows = stmt.query(params![&schema_id, &name, &snapshot_id, &snapshot_id])?;

        if let Some(row) = rows.next()? {
            let table_id: i64 = row.get(0)?;
            let table_name: String = row.get(1)?;
            let path: String = row.get(2)?;
            let path_is_relative: bool = row.get(3)?;
            Ok(Some(TableMetadata {
                table_id,
                table_name,
                path,
                path_is_relative,
            }))
        } else {
            Ok(None)
        }
    }

    fn table_exists(&self, schema_id: i64, name: &str, snapshot_id: i64) -> crate::Result<bool> {
        let conn = self.connection();
        let exists: bool = conn.query_row(
            SQL_TABLE_EXISTS,
            params![schema_id, &name, &snapshot_id, &snapshot_id],
            |row| row.get(0),
        )?;
        Ok(exists)
    }

    fn list_all_tables(&self, snapshot_id: i64) -> crate::Result<Vec<TableWithSchema>> {
        let conn = self.connection();
        let mut stmt = conn.prepare(SQL_LIST_ALL_TABLES)?;

        let tables = stmt
            .query_map(
                params![snapshot_id, snapshot_id, snapshot_id, snapshot_id],
                |row| {
                    let schema_name: String = row.get(0)?;
                    let table = TableMetadata {
                        table_id: row.get(1)?,
                        table_name: row.get(2)?,
                        path: row.get(3)?,
                        path_is_relative: row.get(4)?,
                    };
                    Ok(TableWithSchema {
                        schema_name,
                        table,
                    })
                },
            )?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(tables)
    }

    fn list_all_columns(&self, snapshot_id: i64) -> crate::Result<Vec<ColumnWithTable>> {
        let conn = self.connection();
        let mut stmt = conn.prepare(SQL_LIST_ALL_COLUMNS)?;

        let raw_columns: Vec<(ColumnWithTable, Option<i64>)> = stmt
            .query_map(
                params![
                    snapshot_id,
                    snapshot_id,
                    snapshot_id,
                    snapshot_id,
                    snapshot_id,
                    snapshot_id
                ],
                |row| {
                    let schema_name: String = row.get(0)?;
                    let table_name: String = row.get(1)?;
                    let nulls_allowed: Option<bool> = row.get(5)?;
                    let parent_column: Option<i64> = row.get(6)?;
                    let column = DuckLakeTableColumn {
                        column_id: row.get(2)?,
                        column_name: row.get(3)?,
                        column_type: row.get(4)?,
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
                },
            )?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(reconstruct_list_columns_with_table(raw_columns))
    }

    fn list_all_files(&self, snapshot_id: i64) -> crate::Result<Vec<FileWithTable>> {
        let conn = self.connection();
        let mut stmt = conn.prepare(SQL_LIST_ALL_FILES)?;

        let files = stmt
            .query_map(
                params![
                    snapshot_id,
                    snapshot_id,
                    snapshot_id,
                    snapshot_id,
                    snapshot_id,
                    snapshot_id,
                    snapshot_id,
                    snapshot_id
                ],
                |row| {
                    let schema_name: String = row.get(0)?;
                    let table_name: String = row.get(1)?;

                    // Column 2 is data_file_id; columns 3-7 are the data file.
                    let data_file_id: i64 = row.get(2)?;
                    let data_file = DuckLakeFileData {
                        path: row.get(3)?,
                        path_is_relative: row.get(4)?,
                        file_size_bytes: row.get(5)?,
                        footer_size: row.get(6)?,
                        encryption_key: row.get(7)?,
                    };

                    // Column 8 is delete_file_id (NULL when no live delete file).
                    let (delete_file, delete_file_id) =
                        if let Ok(Some(dfid)) = row.get::<_, Option<i64>>(8) {
                            (
                                Some(DuckLakeFileData {
                                    path: row.get(9)?,
                                    path_is_relative: row.get(10)?,
                                    file_size_bytes: row.get(11)?,
                                    footer_size: row.get(12)?,
                                    encryption_key: row.get(13)?,
                                }),
                                Some(dfid),
                            )
                        } else {
                            (None, None)
                        };

                    let max_row_count = row.get::<_, Option<i64>>(14)?;

                    Ok(FileWithTable {
                        schema_name,
                        table_name,
                        file: DuckLakeTableFile {
                            data_file_id,
                            file: data_file,
                            delete_file_id,
                            delete_file,
                            row_id_start: None,
                            snapshot_id: None,
                            begin_snapshot: None,
                            schema_version: None,
                            partial_max: None,
                            max_row_count,
                            delete_count: None,
                        },
                    })
                },
            )?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(files)
    }

    fn get_data_files_added_between_snapshots(
        &self,
        table_id: i64,
        start_snapshot: i64,
        end_snapshot: i64,
    ) -> crate::Result<Vec<DataFileChange>> {
        let conn = self.connection();

        // DuckLake's catalog schema renamed the merged-partial-file marker:
        // older catalogs (spec 0.2, written by earlier ducklake extensions)
        // carry `partial_file_info` (a cumulative `snapshot:rowcount|...`
        // string); current ones carry `partial_max` (BIGINT). Detect which
        // column this catalog has and query accordingly.
        let has_partial_max: bool = conn.query_row(
            "SELECT COUNT(*) > 0 FROM pragma_table_info('ducklake_data_file') \
             WHERE name = 'partial_max'",
            [],
            |row| row.get(0),
        )?;

        if has_partial_max {
            let mut stmt = conn.prepare(SQL_GET_DATA_FILES_ADDED_BETWEEN_SNAPSHOTS)?;
            let files = stmt
                .query_map(params![table_id, start_snapshot, end_snapshot], |row| {
                    Ok(DataFileChange {
                        begin_snapshot: row.get(0)?,
                        path: row.get(1)?,
                        path_is_relative: row.get(2)?,
                        file_size_bytes: row.get(3)?,
                        footer_size: row.get(4)?,
                        encryption_key: row.get(5)?,
                        row_id_start: row.get(6)?,
                        partial_max: row.get(7)?,
                    })
                })?
                .collect::<Result<Vec<_>, _>>()?;
            return Ok(files);
        }

        // Old-spec catalog: fetch candidate partial files broadly and apply the
        // `partial_max >= start` bound in Rust after parsing the info string.
        let mut stmt = conn.prepare(
            "SELECT
                data.begin_snapshot,
                data.path,
                data.path_is_relative,
                data.file_size_bytes,
                data.footer_size,
                data.encryption_key,
                data.row_id_start,
                data.partial_file_info
            FROM ducklake_data_file AS data
            WHERE data.table_id = $1
              AND data.begin_snapshot <= $3
              AND (data.begin_snapshot >= $2 OR data.partial_file_info IS NOT NULL)
            ORDER BY data.begin_snapshot",
        )?;
        let files = stmt
            .query_map(params![table_id, start_snapshot, end_snapshot], |row| {
                let info: Option<String> = row.get(7)?;
                Ok(DataFileChange {
                    begin_snapshot: row.get(0)?,
                    path: row.get(1)?,
                    path_is_relative: row.get(2)?,
                    file_size_bytes: row.get(3)?,
                    footer_size: row.get(4)?,
                    encryption_key: row.get(5)?,
                    row_id_start: row.get(6)?,
                    partial_max: info.as_deref().and_then(parse_partial_file_info_max),
                })
            })?
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .filter(|f: &DataFileChange| {
                f.begin_snapshot >= start_snapshot
                    || f.partial_max.is_some_and(|max| max >= start_snapshot)
            })
            .collect();

        Ok(files)
    }

    fn get_delete_files_added_between_snapshots(
        &self,
        table_id: i64,
        start_snapshot: i64,
        end_snapshot: i64,
    ) -> crate::Result<Vec<DeleteFileChange>> {
        let conn = self.connection();

        // Cumulative (current-spec) delete files can hold in-window deletions
        // even when their begin_snapshot predates the window; they are included
        // via `ducklake_delete_file.partial_max` (their max embedded snapshot).
        // Older catalogs have no such column — and no cumulative delete files —
        // so the predicate degrades to NULL there, keeping the plain
        // begin-snapshot window.
        let has_delete_partial_max: bool = conn.query_row(
            "SELECT COUNT(*) > 0 FROM pragma_table_info('ducklake_delete_file') \
             WHERE name = 'partial_max'",
            [],
            |row| row.get(0),
        )?;
        let sql = if has_delete_partial_max {
            SQL_GET_DELETE_FILES_ADDED_BETWEEN_SNAPSHOTS.to_string()
        } else {
            SQL_GET_DELETE_FILES_ADDED_BETWEEN_SNAPSHOTS.replace("df.partial_max", "NULL")
        };
        let mut stmt = conn.prepare(&sql)?;

        let files = stmt
            .query_map(params![table_id, start_snapshot, end_snapshot], |row| {
                Ok(DeleteFileChange {
                    // data file
                    data_file_path: row.get(0)?,
                    data_file_path_is_relative: row.get(1)?,
                    data_file_size_bytes: row.get(2)?,
                    data_file_footer_size: row.get(3)?,
                    data_row_id_start: row.get(4)?,
                    data_record_count: row.get(5)?,
                    data_mapping_id: row.get(6)?,

                    // current delete
                    current_delete_path: row.get(7)?,
                    current_delete_path_is_relative: row.get(8)?,
                    current_delete_file_size_bytes: row.get(9)?,
                    current_delete_footer_size: row.get(10)?,

                    // previous delete
                    previous_delete_path: row.get(11)?,
                    previous_delete_path_is_relative: row.get(12)?,
                    previous_delete_file_size_bytes: row.get(13)?,
                    previous_delete_footer_size: row.get(14)?,

                    // snapshot
                    snapshot_id: row.get(15)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(files)
    }
}

/// Parse the maximum origin snapshot id out of an old-spec `partial_file_info`
/// string — a `|`-separated list of cumulative `snapshot:rowcount` pairs (e.g.
/// `"2:1|3:2|4:3"`), whose last pair carries the file's maximum snapshot.
fn parse_partial_file_info_max(info: &str) -> Option<i64> {
    info.rsplit('|')
        .next()
        .and_then(|pair| pair.split(':').next())
        .and_then(|snap| snap.trim().parse::<i64>().ok())
}

#[cfg(test)]
mod partial_file_info_tests {
    use super::parse_partial_file_info_max;

    #[test]
    fn parses_multi_pair_info() {
        assert_eq!(parse_partial_file_info_max("2:1|3:2|4:3"), Some(4));
    }

    #[test]
    fn parses_single_pair_info() {
        assert_eq!(parse_partial_file_info_max("7:100"), Some(7));
    }

    #[test]
    fn malformed_info_is_none() {
        assert_eq!(parse_partial_file_info_max(""), None);
        assert_eq!(parse_partial_file_info_max("nonsense"), None);
    }
}
