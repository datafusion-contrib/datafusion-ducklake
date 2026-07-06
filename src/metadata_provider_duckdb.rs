use crate::DuckLakeError;
use crate::metadata_provider::{
    ColumnWithTable, DataFileChange, DeleteFileChange, DuckLakeFileData, DuckLakeTableColumn,
    DuckLakeTableFile, FileWithTable, MetadataProvider, SQL_GET_DATA_FILES,
    SQL_GET_DATA_FILES_ADDED_BETWEEN_SNAPSHOTS, SQL_GET_DATA_PATH,
    SQL_GET_DELETE_FILES_ADDED_BETWEEN_SNAPSHOTS, SQL_GET_LATEST_SNAPSHOT, SQL_GET_SCHEMA_BY_NAME,
    SQL_GET_TABLE_BY_NAME, SQL_GET_TABLE_COLUMNS, SQL_LIST_ALL_COLUMNS, SQL_LIST_ALL_FILES,
    SQL_LIST_ALL_TABLES, SQL_LIST_SCHEMAS, SQL_LIST_SNAPSHOTS, SQL_LIST_TABLES, SQL_TABLE_EXISTS,
    SchemaMetadata, SnapshotMetadata, TableMetadata, TableWithSchema, reconstruct_list_columns,
    reconstruct_list_columns_with_table,
};
use duckdb::AccessMode::ReadOnly;
use duckdb::{Config, Connection, params};
use std::sync::{Arc, Mutex, MutexGuard};

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
                        max_row_count: record_count,
                        delete_count,
                    })
                },
            )?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(files)
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
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(files)
    }

    fn get_delete_files_added_between_snapshots(
        &self,
        table_id: i64,
        start_snapshot: i64,
        end_snapshot: i64,
    ) -> crate::Result<Vec<DeleteFileChange>> {
        let conn = self.connection();
        let mut stmt = conn.prepare(SQL_GET_DELETE_FILES_ADDED_BETWEEN_SNAPSHOTS)?;

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
