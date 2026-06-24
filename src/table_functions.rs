//! User-Defined Table Functions (UDTFs) for DuckLake catalog metadata

use datafusion::catalog::TableFunctionImpl;
use datafusion::common::{Result as DataFusionResult, ScalarValue, plan_err};
use datafusion::datasource::TableProvider;
use datafusion::logical_expr::Expr;
use std::sync::Arc;

use crate::information_schema::{FilesTable, SnapshotsTable, TableInfoTable};
use crate::metadata_provider::MetadataProvider;
use crate::path_resolver::{parse_object_store_url, resolve_path};
use crate::table_changes::TableChangesTable;
use crate::table_deletions::TableDeletionsTable;
use crate::types::build_arrow_schema;

#[derive(Debug)]
pub struct DucklakeSnapshotsFunction {
    provider: Arc<dyn MetadataProvider>,
}

impl DucklakeSnapshotsFunction {
    pub fn new(provider: Arc<dyn MetadataProvider>) -> Self {
        Self {
            provider,
        }
    }
}

impl TableFunctionImpl for DucklakeSnapshotsFunction {
    fn call(&self, exprs: &[Expr]) -> DataFusionResult<Arc<dyn TableProvider>> {
        if !exprs.is_empty() {
            return plan_err!("ducklake_snapshots() takes no arguments");
        }

        Ok(Arc::new(SnapshotsTable::new(self.provider.clone())))
    }
}

#[derive(Debug)]
pub struct DucklakeTableInfoFunction {
    provider: Arc<dyn MetadataProvider>,
}

impl DucklakeTableInfoFunction {
    pub fn new(provider: Arc<dyn MetadataProvider>) -> Self {
        Self {
            provider,
        }
    }
}

impl TableFunctionImpl for DucklakeTableInfoFunction {
    fn call(&self, exprs: &[Expr]) -> DataFusionResult<Arc<dyn TableProvider>> {
        if !exprs.is_empty() {
            return plan_err!("ducklake_table_info() takes no arguments");
        }

        Ok(Arc::new(TableInfoTable::new(self.provider.clone())))
    }
}

#[derive(Debug)]
pub struct DucklakeListFilesFunction {
    provider: Arc<dyn MetadataProvider>,
}

impl DucklakeListFilesFunction {
    pub fn new(provider: Arc<dyn MetadataProvider>) -> Self {
        Self {
            provider,
        }
    }
}

impl TableFunctionImpl for DucklakeListFilesFunction {
    fn call(&self, exprs: &[Expr]) -> DataFusionResult<Arc<dyn TableProvider>> {
        if !exprs.is_empty() {
            return plan_err!("ducklake_list_files() takes no arguments");
        }

        Ok(Arc::new(FilesTable::new(self.provider.clone())))
    }
}

#[derive(Debug)]
pub struct DucklakeTableChangesFunction {
    provider: Arc<dyn MetadataProvider>,
}

impl DucklakeTableChangesFunction {
    pub fn new(provider: Arc<dyn MetadataProvider>) -> Self {
        Self {
            provider,
        }
    }

    fn parse_table_name(table_name: &str) -> (&str, &str) {
        if let Some(dot_pos) = table_name.find('.') {
            let schema = &table_name[..dot_pos];
            let table = &table_name[dot_pos + 1..];
            (schema, table)
        } else {
            ("main", table_name)
        }
    }
}

impl TableFunctionImpl for DucklakeTableChangesFunction {
    fn call(&self, exprs: &[Expr]) -> DataFusionResult<Arc<dyn TableProvider>> {
        if exprs.len() != 3 {
            return plan_err!(
                "ducklake_table_changes() requires 3 arguments: \
                 ducklake_table_changes('schema.table', start_snapshot, end_snapshot)"
            );
        }

        // Parse table name argument
        let table_name = match &exprs[0] {
            Expr::Literal(ScalarValue::Utf8(Some(name)), _) => name.clone(),
            _ => {
                return plan_err!(
                    "First argument to ducklake_table_changes() must be a string literal \
                     (e.g., 'main.users' or 'users')"
                );
            },
        };

        // Parse start_snapshot argument
        let start_snapshot = match &exprs[1] {
            Expr::Literal(ScalarValue::Int64(Some(v)), _) => *v,
            Expr::Literal(ScalarValue::Int32(Some(v)), _) => *v as i64,
            _ => {
                return plan_err!(
                    "Second argument to ducklake_table_changes() must be an integer (start_snapshot)"
                );
            },
        };

        // Parse end_snapshot argument
        let end_snapshot = match &exprs[2] {
            Expr::Literal(ScalarValue::Int64(Some(v)), _) => *v,
            Expr::Literal(ScalarValue::Int32(Some(v)), _) => *v as i64,
            _ => {
                return plan_err!(
                    "Third argument to ducklake_table_changes() must be an integer (end_snapshot)"
                );
            },
        };

        // Validate snapshot range
        if start_snapshot > end_snapshot {
            return plan_err!(
                "start_snapshot ({}) must be less than or equal to end_snapshot ({})",
                start_snapshot,
                end_snapshot
            );
        }

        // Look up the table to get table_id
        let (schema_name, table_name_only) = Self::parse_table_name(&table_name);

        let snapshot_id = self
            .provider
            .get_current_snapshot()
            .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;

        let schema = self
            .provider
            .get_schema_by_name(schema_name, snapshot_id)
            .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?
            .ok_or_else(|| {
                datafusion::error::DataFusionError::Plan(format!(
                    "Schema '{}' not found in catalog",
                    schema_name
                ))
            })?;

        let table = self
            .provider
            .get_table_by_name(schema.schema_id, table_name_only, snapshot_id)
            .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?
            .ok_or_else(|| {
                datafusion::error::DataFusionError::Plan(format!(
                    "Table '{}.{}' not found in catalog",
                    schema_name, table_name_only
                ))
            })?;

        // Get data_path and parse ObjectStoreUrl
        let data_path = self
            .provider
            .get_data_path()
            .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;

        let (object_store_url, catalog_path) = parse_object_store_url(&data_path)
            .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;

        // Resolve full table path: catalog_path -> schema.path -> table.path
        let schema_path = resolve_path(&catalog_path, &schema.path, schema.path_is_relative)
            .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
        let table_path = resolve_path(&schema_path, &table.path, table.path_is_relative)
            .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;

        // Get table structure and build Arrow schema
        let columns = self
            .provider
            .get_table_structure(table.table_id, end_snapshot)
            .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;

        let table_schema = Arc::new(
            build_arrow_schema(&columns)
                .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?,
        );

        Ok(Arc::new(TableChangesTable::new(
            self.provider.clone(),
            table.table_id,
            start_snapshot,
            end_snapshot,
            Arc::new(object_store_url),
            table_path,
            table_schema,
        )))
    }
}

#[derive(Debug)]
pub struct DucklakeTableDeletionsFunction {
    provider: Arc<dyn MetadataProvider>,
}

impl DucklakeTableDeletionsFunction {
    pub fn new(provider: Arc<dyn MetadataProvider>) -> Self {
        Self {
            provider,
        }
    }

    fn parse_table_name(table_name: &str) -> (&str, &str) {
        if let Some(dot_pos) = table_name.find('.') {
            let schema = &table_name[..dot_pos];
            let table = &table_name[dot_pos + 1..];
            (schema, table)
        } else {
            ("main", table_name)
        }
    }
}

impl TableFunctionImpl for DucklakeTableDeletionsFunction {
    fn call(&self, exprs: &[Expr]) -> DataFusionResult<Arc<dyn TableProvider>> {
        if exprs.len() != 3 {
            return plan_err!(
                "ducklake_table_deletions() requires 3 arguments: \
                 ducklake_table_deletions('schema.table', start_snapshot, end_snapshot)"
            );
        }

        // Parse table name argument
        let table_name = match &exprs[0] {
            Expr::Literal(ScalarValue::Utf8(Some(name)), _) => name.clone(),
            _ => {
                return plan_err!(
                    "First argument to ducklake_table_deletions() must be a string literal \
                     (e.g., 'main.users' or 'users')"
                );
            },
        };

        // Parse start_snapshot argument
        let start_snapshot = match &exprs[1] {
            Expr::Literal(ScalarValue::Int64(Some(v)), _) => *v,
            Expr::Literal(ScalarValue::Int32(Some(v)), _) => *v as i64,
            _ => {
                return plan_err!(
                    "Second argument to ducklake_table_deletions() must be an integer (start_snapshot)"
                );
            },
        };

        // Parse end_snapshot argument
        let end_snapshot = match &exprs[2] {
            Expr::Literal(ScalarValue::Int64(Some(v)), _) => *v,
            Expr::Literal(ScalarValue::Int32(Some(v)), _) => *v as i64,
            _ => {
                return plan_err!(
                    "Third argument to ducklake_table_deletions() must be an integer (end_snapshot)"
                );
            },
        };

        // Validate snapshot range
        if start_snapshot > end_snapshot {
            return plan_err!(
                "start_snapshot ({}) must be less than or equal to end_snapshot ({})",
                start_snapshot,
                end_snapshot
            );
        }

        // Look up the table to get table_id
        let (schema_name, table_name_only) = Self::parse_table_name(&table_name);

        let snapshot_id = self
            .provider
            .get_current_snapshot()
            .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;

        let schema = self
            .provider
            .get_schema_by_name(schema_name, snapshot_id)
            .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?
            .ok_or_else(|| {
                datafusion::error::DataFusionError::Plan(format!(
                    "Schema '{}' not found in catalog",
                    schema_name
                ))
            })?;

        let table = self
            .provider
            .get_table_by_name(schema.schema_id, table_name_only, snapshot_id)
            .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?
            .ok_or_else(|| {
                datafusion::error::DataFusionError::Plan(format!(
                    "Table '{}.{}' not found in catalog",
                    schema_name, table_name_only
                ))
            })?;

        // Get data_path and parse ObjectStoreUrl
        let data_path = self
            .provider
            .get_data_path()
            .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;

        let (object_store_url, catalog_path) = parse_object_store_url(&data_path)
            .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;

        // Resolve full table path: catalog_path -> schema.path -> table.path
        let schema_path = resolve_path(&catalog_path, &schema.path, schema.path_is_relative)
            .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
        let table_path = resolve_path(&schema_path, &table.path, table.path_is_relative)
            .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;

        // Get table structure and build Arrow schema
        let columns = self
            .provider
            .get_table_structure(table.table_id, end_snapshot)
            .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;

        let table_schema = Arc::new(
            build_arrow_schema(&columns)
                .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?,
        );

        Ok(Arc::new(TableDeletionsTable::new(
            self.provider.clone(),
            table.table_id,
            start_snapshot,
            end_snapshot,
            Arc::new(object_store_url),
            table_path,
            table_schema,
        )))
    }
}

/// Registers all ducklake_*() table functions with a SessionContext.
pub fn register_ducklake_functions(
    ctx: &datafusion::execution::context::SessionContext,
    provider: Arc<dyn MetadataProvider>,
) {
    ctx.register_udtf(
        "ducklake_snapshots",
        Arc::new(DucklakeSnapshotsFunction::new(provider.clone())),
    );
    ctx.register_udtf(
        "ducklake_table_info",
        Arc::new(DucklakeTableInfoFunction::new(provider.clone())),
    );
    ctx.register_udtf(
        "ducklake_list_files",
        Arc::new(DucklakeListFilesFunction::new(provider.clone())),
    );
    ctx.register_udtf(
        "ducklake_table_changes",
        Arc::new(DucklakeTableChangesFunction::new(provider.clone())),
    );
    ctx.register_udtf(
        "ducklake_table_deletions",
        Arc::new(DucklakeTableDeletionsFunction::new(provider.clone())),
    );
}
