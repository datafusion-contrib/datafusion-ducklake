#![cfg(feature = "metadata-duckdb")]
//! Hybrid AsyncDB adapter for running DuckDB DuckLake tests
//!
//! This adapter uses a hybrid approach:
//! - WRITE operations (CREATE/INSERT/UPDATE/DELETE) → DuckDB
//! - READ operations (SELECT) → DataFusion
//! - After each WRITE → Refresh DataFusion catalog to pick up metadata changes
//! - Table references rewritten for DataFusion: ducklake.table → ducklake.main.table
//!
//! This allows running DuckDB tests through DataFusion's read path.

use datafusion::arrow::array::*;
use datafusion::arrow::datatypes::DataType;
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::prelude::*;
use datafusion_ducklake::catalog::DuckLakeCatalog;
use datafusion_ducklake::{DuckdbMetadataProvider, MetadataProvider, register_ducklake_functions};
use duckdb::Connection;
use sqllogictest::{AsyncDB, DBOutput, DefaultColumnType};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// Custom error type for hybrid adapter
#[derive(Debug)]
pub struct HybridError(String);

impl std::fmt::Display for HybridError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Hybrid error: {}", self.0)
    }
}

impl std::error::Error for HybridError {}

impl From<duckdb::Error> for HybridError {
    fn from(e: duckdb::Error) -> Self {
        HybridError(format!("DuckDB error: {}", e))
    }
}

impl From<datafusion::error::DataFusionError> for HybridError {
    fn from(e: datafusion::error::DataFusionError) -> Self {
        HybridError(format!("DataFusion error: {}", e))
    }
}

impl From<datafusion_ducklake::error::DuckLakeError> for HybridError {
    fn from(e: datafusion_ducklake::error::DuckLakeError) -> Self {
        HybridError(format!("DuckLake error: {}", e))
    }
}

/// Hybrid database adapter: DuckDB for writes, DataFusion for reads
#[derive(Clone)]
pub struct HybridDuckLakeDB {
    /// DuckDB connection for WRITE operations
    duckdb_conn: Arc<Mutex<Connection>>,
    /// DataFusion context for READ operations
    datafusion_ctx: Arc<Mutex<SessionContext>>,
    /// Path to DuckLake catalog file
    catalog_path: PathBuf,
}

impl HybridDuckLakeDB {
    pub fn new(catalog_path: PathBuf) -> Result<Self, HybridError> {
        // Create data files directory
        let data_path = catalog_path.with_extension("files");
        std::fs::create_dir_all(&data_path)
            .map_err(|e| HybridError(format!("Failed to create data directory: {}", e)))?;

        // Create DuckDB connection for WRITE operations
        let conn = Connection::open_in_memory()?;
        conn.execute("INSTALL ducklake;", [])?;
        conn.execute("LOAD ducklake;", [])?;

        let ducklake_path = format!("ducklake:{}", catalog_path.display());
        let attach_sql = format!(
            "ATTACH '{}' AS ducklake (DATA_PATH '{}')",
            ducklake_path,
            data_path.display()
        );
        conn.execute(&attach_sql, [])?;

        // Create DataFusion context for READ operations
        let ctx = SessionContext::new();
        let metadata_provider = DuckdbMetadataProvider::new(catalog_path.to_str().unwrap())?;
        let catalog = Arc::new(DuckLakeCatalog::new(metadata_provider)?);
        ctx.register_catalog("ducklake", catalog);
        let provider: Arc<dyn MetadataProvider> =
            Arc::new(DuckdbMetadataProvider::new(catalog_path.to_str().unwrap())?);
        register_ducklake_functions(&ctx, provider);

        Ok(Self {
            duckdb_conn: Arc::new(Mutex::new(conn)),
            datafusion_ctx: Arc::new(Mutex::new(ctx)),
            catalog_path,
        })
    }

    /// Detect if SQL should be routed to DuckDB
    /// Includes WRITE operations and catalog management statements
    fn is_write_statement(sql: &str) -> bool {
        let trimmed = sql.trim().to_uppercase();
        trimmed.starts_with("CREATE ")
            || trimmed.starts_with("INSERT ")
            || trimmed.starts_with("UPDATE ")
            || trimmed.starts_with("DELETE ")
            || trimmed.starts_with("DROP ")
            || trimmed.starts_with("ALTER ")
            || trimmed.starts_with("MERGE ")
            || trimmed.starts_with("USE ")
            || trimmed.starts_with("SHOW ")
            || trimmed.starts_with("CALL ")
            || trimmed == "BEGIN"
            || trimmed.starts_with("BEGIN ")
            || trimmed == "COMMIT"
            || trimmed.starts_with("COMMIT ")
            || trimmed == "ROLLBACK"
            || trimmed.starts_with("ROLLBACK ")
    }

    /// Rewrite official CDC function call forms onto this crate's UDTFs:
    /// the catalog-scoped 3-arg sugar `ducklake.table_changes('t', a, b)` and
    /// the global 5-arg form `ducklake_table_changes('cat', 'schema', 't', a, b)`
    /// both become `ducklake_table_changes('schema.t', a, b)`.
    fn rewrite_cdc_function_calls(sql: &str) -> String {
        use std::sync::OnceLock;
        static SCOPED: OnceLock<regex::Regex> = OnceLock::new();
        static FIVE_ARG: OnceLock<regex::Regex> = OnceLock::new();
        let scoped = SCOPED.get_or_init(|| {
            regex::Regex::new(
                r"ducklake\.(table_changes|table_deletions|table_insertions)\(\s*'([^']+)'",
            )
            .unwrap()
        });
        let five_arg = FIVE_ARG.get_or_init(|| {
            regex::Regex::new(
                r"ducklake_(table_changes|table_deletions|table_insertions)\(\s*'[^']+'\s*,\s*'([^']+)'\s*,\s*'([^']+)'\s*,",
            )
            .unwrap()
        });
        let sql = scoped.replace_all(sql, "ducklake_$1('main.$2'");
        five_arg
            .replace_all(&sql, "ducklake_$1('$2.$3',")
            .into_owned()
    }

    /// Rewrite table references from 2-part to 3-part names
    fn rewrite_table_references(sql: &str) -> String {
        let sql = Self::rewrite_cdc_function_calls(sql);
        let sql: &str = sql.as_str();
        // Avoid double-conversion
        if sql.contains("ducklake.main.") {
            return sql.to_string();
        }

        // Simple string replacement
        let mut result = String::with_capacity(sql.len() + 100);
        let mut remaining = sql;

        while let Some(pos) = remaining.find("ducklake.") {
            result.push_str(&remaining[..pos]);
            result.push_str("ducklake.");
            let after = &remaining[pos + 9..]; // 9 = len("ducklake.")

            if after.starts_with("main.") {
                remaining = after;
            } else {
                result.push_str("main.");
                remaining = after;
            }
        }
        result.push_str(remaining);
        result
    }

    /// Refresh catalog snapshot after a write
    fn refresh_catalog(&self) -> Result<(), HybridError> {
        let mut ctx_guard = self.datafusion_ctx.lock().unwrap();

        // Create new session context with fresh catalog
        let new_ctx = SessionContext::new();
        let metadata_provider = DuckdbMetadataProvider::new(self.catalog_path.to_str().unwrap())?;
        let catalog = Arc::new(DuckLakeCatalog::new(metadata_provider)?);
        new_ctx.register_catalog("ducklake", catalog);
        let provider: Arc<dyn MetadataProvider> = Arc::new(DuckdbMetadataProvider::new(
            self.catalog_path.to_str().unwrap(),
        )?);
        register_ducklake_functions(&new_ctx, provider);

        // Replace the context
        *ctx_guard = new_ctx;

        Ok(())
    }

    /// Execute WRITE via DuckDB
    fn execute_write(&self, sql: &str) -> Result<(), HybridError> {
        let conn = self.duckdb_conn.lock().unwrap();
        conn.execute(sql, [])?;
        Ok(())
    }

    /// Execute READ via DataFusion
    async fn execute_read(&self, sql: &str) -> Result<Vec<RecordBatch>, HybridError> {
        let sql_rewritten = Self::rewrite_table_references(sql);

        // Clone the context to release the lock before await
        let ctx = {
            let ctx_guard = self.datafusion_ctx.lock().unwrap();
            ctx_guard.clone()
        };

        let df = ctx.sql(&sql_rewritten).await?;
        let batches = df.collect().await?;
        Ok(batches)
    }
}

#[async_trait::async_trait]
impl AsyncDB for HybridDuckLakeDB {
    type Error = HybridError;
    type ColumnType = DefaultColumnType;

    async fn run(&mut self, sql: &str) -> Result<DBOutput<Self::ColumnType>, Self::Error> {
        if Self::is_write_statement(sql) {
            // DuckDB path: WRITE operations and catalog management
            // Includes: CREATE, INSERT, UPDATE, DELETE, USE, SHOW, CALL, etc.
            self.execute_write(sql)?;

            // Refresh catalog to pick up changes
            self.refresh_catalog()?;

            // Return success
            Ok(DBOutput::StatementComplete(0))
        } else {
            // DataFusion path: READ operations (SELECT, etc.)
            let batches = self.execute_read(sql).await?;

            if batches.is_empty() {
                return Ok(DBOutput::StatementComplete(0));
            }

            // Convert to sqllogictest format
            let schema = batches[0].schema();
            let types = schema
                .fields()
                .iter()
                .map(|f| match f.data_type() {
                    DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 => {
                        DefaultColumnType::Integer
                    },
                    DataType::Float32 | DataType::Float64 => DefaultColumnType::FloatingPoint,
                    DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => {
                        DefaultColumnType::Text
                    },
                    _ => DefaultColumnType::Any,
                })
                .collect::<Vec<_>>();

            let mut rows = Vec::new();
            for batch in batches {
                rows.extend(convert_batch_to_strings(&batch)?);
            }

            Ok(DBOutput::Rows {
                types,
                rows,
            })
        }
    }

    fn engine_name(&self) -> &str {
        "HybridDuckLake(DuckDB+DataFusion)"
    }
}

/// Convert RecordBatch to string rows for sqllogictest
fn convert_batch_to_strings(batch: &RecordBatch) -> Result<Vec<Vec<String>>, HybridError> {
    let mut rows = Vec::new();

    for row_idx in 0..batch.num_rows() {
        let mut row = Vec::new();
        for col_idx in 0..batch.num_columns() {
            let column = batch.column(col_idx);
            let value = if column.is_null(row_idx) {
                "NULL".to_string()
            } else {
                match column.data_type() {
                    DataType::Int8 => {
                        let arr = column.as_any().downcast_ref::<Int8Array>().unwrap();
                        arr.value(row_idx).to_string()
                    },
                    DataType::Int16 => {
                        let arr = column.as_any().downcast_ref::<Int16Array>().unwrap();
                        arr.value(row_idx).to_string()
                    },
                    DataType::Int32 => {
                        let arr = column.as_any().downcast_ref::<Int32Array>().unwrap();
                        arr.value(row_idx).to_string()
                    },
                    DataType::Int64 => {
                        let arr = column.as_any().downcast_ref::<Int64Array>().unwrap();
                        arr.value(row_idx).to_string()
                    },
                    DataType::Float32 => {
                        let arr = column.as_any().downcast_ref::<Float32Array>().unwrap();
                        arr.value(row_idx).to_string()
                    },
                    DataType::Float64 => {
                        let arr = column.as_any().downcast_ref::<Float64Array>().unwrap();
                        arr.value(row_idx).to_string()
                    },
                    DataType::Utf8 => {
                        let arr = column.as_any().downcast_ref::<StringArray>().unwrap();
                        arr.value(row_idx).to_string()
                    },
                    DataType::LargeUtf8 => {
                        let arr = column.as_any().downcast_ref::<LargeStringArray>().unwrap();
                        arr.value(row_idx).to_string()
                    },
                    DataType::Utf8View => {
                        let arr = column.as_any().downcast_ref::<StringViewArray>().unwrap();
                        arr.value(row_idx).to_string()
                    },
                    DataType::Boolean => {
                        let arr = column.as_any().downcast_ref::<BooleanArray>().unwrap();
                        arr.value(row_idx).to_string()
                    },
                    DataType::Date32 => {
                        let arr = column.as_any().downcast_ref::<Date32Array>().unwrap();
                        arr.value_as_date(row_idx).unwrap().to_string()
                    },
                    DataType::Timestamp(_, _) => {
                        let arr = column
                            .as_any()
                            .downcast_ref::<TimestampMicrosecondArray>()
                            .unwrap();
                        format!("{}", arr.value_as_datetime(row_idx).unwrap())
                    },
                    DataType::Decimal128(_, scale) => {
                        let arr = column.as_any().downcast_ref::<Decimal128Array>().unwrap();
                        let value = arr.value(row_idx);
                        let scale_factor = 10_f64.powi(*scale as i32);
                        format!(
                            "{:.scale$}",
                            value as f64 / scale_factor,
                            scale = *scale as usize
                        )
                    },
                    _ => format!("{:?}", column),
                }
            };
            row.push(value);
        }
        rows.push(row);
    }

    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_detection() {
        // Write operations - routed to DuckDB
        assert!(HybridDuckLakeDB::is_write_statement(
            "CREATE TABLE foo (id INT)"
        ));
        assert!(HybridDuckLakeDB::is_write_statement(
            "INSERT INTO foo VALUES (1)"
        ));
        assert!(HybridDuckLakeDB::is_write_statement(
            "UPDATE foo SET id = 2"
        ));
        assert!(HybridDuckLakeDB::is_write_statement("DELETE FROM foo"));
        assert!(HybridDuckLakeDB::is_write_statement("DROP TABLE foo"));
        assert!(HybridDuckLakeDB::is_write_statement(
            "ALTER TABLE foo ADD COLUMN bar INT"
        ));

        // Transaction control - routed to DuckDB
        assert!(HybridDuckLakeDB::is_write_statement("BEGIN"));
        assert!(HybridDuckLakeDB::is_write_statement("COMMIT"));
        assert!(HybridDuckLakeDB::is_write_statement("ROLLBACK"));

        // Catalog management - routed to DuckDB
        assert!(HybridDuckLakeDB::is_write_statement("USE ducklake"));
        assert!(HybridDuckLakeDB::is_write_statement("SHOW TABLES"));
        assert!(HybridDuckLakeDB::is_write_statement(
            "CALL some_procedure()"
        ));

        // Read operations - routed to DataFusion
        assert!(!HybridDuckLakeDB::is_write_statement("SELECT * FROM foo"));
        assert!(!HybridDuckLakeDB::is_write_statement(
            "WITH cte AS (...) SELECT ..."
        ));
    }

    #[test]
    fn test_table_rewrite() {
        let result = HybridDuckLakeDB::rewrite_table_references("SELECT * FROM ducklake.test");
        assert!(
            result.contains("ducklake.main.test"),
            "Expected 'ducklake.main.test' in: {}",
            result
        );

        let result =
            HybridDuckLakeDB::rewrite_table_references("INSERT INTO ducklake.test VALUES (1)");
        assert!(
            result.contains("ducklake.main.test"),
            "Expected 'ducklake.main.test' in: {}",
            result
        );

        // Avoid double-rewrite
        let result = HybridDuckLakeDB::rewrite_table_references("SELECT * FROM ducklake.main.test");
        assert_eq!(result, "SELECT * FROM ducklake.main.test");
    }
}
