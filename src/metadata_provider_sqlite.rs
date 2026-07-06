//! SQLite metadata provider for DuckLake catalogs.

use crate::Result;
use crate::metadata_provider::{
    ColumnWithTable, DataFileChange, DeleteFileChange, DuckLakeFileData, DuckLakeTableColumn,
    DuckLakeTableFile, FileWithTable, MetadataProvider, SchemaMetadata, SnapshotMetadata,
    TableMetadata, TableWithSchema, block_on, reconstruct_list_columns,
    reconstruct_list_columns_with_table,
};
use sqlx::Row;
use sqlx::sqlite::{SqlitePool, SqlitePoolOptions};
use sqlx::types::chrono::NaiveDateTime;

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
            let rows = sqlx::query(
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
                    del.delete_count
                FROM ducklake_data_file AS data
                LEFT JOIN ducklake_delete_file AS del
                    ON data.data_file_id = del.data_file_id
                    AND del.table_id = ?
                    AND ? >= del.begin_snapshot
                    AND (? < del.end_snapshot OR del.end_snapshot IS NULL)
                WHERE data.table_id = ?
                  AND ? >= data.begin_snapshot
                  AND (? < data.end_snapshot OR data.end_snapshot IS NULL)",
            )
            .bind(table_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .bind(table_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .fetch_all(&self.pool)
            .await?;

            rows.into_iter()
                .map(|row| {
                    let data_file = DuckLakeFileData {
                        path: row.try_get(1)?,
                        path_is_relative: row.try_get(2)?,
                        file_size_bytes: row.try_get(3)?,
                        footer_size: row.try_get(4)?,
                        encryption_key: row.try_get(5)?,
                    };
                    let row_id_start: Option<i64> = row.try_get(6)?;
                    let record_count: Option<i64> = row.try_get(7)?;

                    let (delete_file, delete_count) = if row.try_get::<Option<i64>, _>(8)?.is_some()
                    {
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
                        row_id_start,
                        snapshot_id: Some(snapshot_id),
                        max_row_count: record_count,
                        delete_count,
                    })
                })
                .collect()
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
                    data.encryption_key
                FROM ducklake_data_file AS data
                WHERE data.table_id = ?
                  AND data.begin_snapshot > ?
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
  AND cd.begin_snapshot > ?
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
  AND data.end_snapshot > ?
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
