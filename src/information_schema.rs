//! Information schema implementation for DuckLake catalog metadata
//!
//! Provides SQL-queryable virtual tables exposing catalog metadata via the standard
//! `information_schema` pattern. Uses live querying - metadata is fetched fresh from
//! the catalog database on every query execution.
//!
//! # Available Tables
//!
//! - `information_schema.snapshots` - All snapshots in the catalog
//! - `information_schema.schemata` - Schemas at current snapshot
//! - `information_schema.tables` - Tables across all schemas at current snapshot
//! - `information_schema.columns` - Columns for all tables
//! - `information_schema.files` - Data files for all tables
//!
//! # Usage
//!
//! ```sql
//! -- List all snapshots
//! SELECT * FROM ducklake.information_schema.snapshots;
//!
//! -- List schemas
//! SELECT * FROM ducklake.information_schema.schemata;
//!
//! -- List tables
//! SELECT * FROM ducklake.information_schema.tables WHERE schema_name = 'public';
//! ```

use std::sync::Arc;

use arrow::array::{ArrayRef, BooleanArray, Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::catalog::{SchemaProvider, Session};
use datafusion::datasource::TableProvider;
use datafusion::datasource::memory::MemTable;
use datafusion::error::Result as DataFusionResult;
use datafusion::logical_expr::TableType;
use datafusion::physical_plan::ExecutionPlan;

use crate::metadata_provider::MetadataProvider;

/// Live table provider for snapshots - queries metadata on every scan
#[derive(Debug)]
pub struct SnapshotsTable {
    provider: Arc<dyn MetadataProvider>,
    schema: SchemaRef,
}

impl SnapshotsTable {
    pub fn new(provider: Arc<dyn MetadataProvider>) -> Self {
        let schema = Arc::new(Schema::new(vec![
            Field::new("snapshot_id", DataType::Int64, false),
            Field::new("timestamp", DataType::Utf8, true),
        ]));
        Self {
            provider,
            schema,
        }
    }

    fn query_snapshots(&self) -> DataFusionResult<RecordBatch> {
        let snapshots = self
            .provider
            .list_snapshots()
            .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;

        let snapshot_ids: ArrayRef = Arc::new(Int64Array::from(
            snapshots.iter().map(|s| s.snapshot_id).collect::<Vec<_>>(),
        ));

        let timestamps: ArrayRef = Arc::new(StringArray::from(
            snapshots
                .iter()
                .map(|s| s.timestamp.as_deref())
                .collect::<Vec<_>>(),
        ));

        RecordBatch::try_new(self.schema.clone(), vec![snapshot_ids, timestamps])
            .map_err(|e| datafusion::error::DataFusionError::ArrowError(Box::new(e), None))
    }
}

#[async_trait::async_trait]
impl TableProvider for SnapshotsTable {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn table_type(&self) -> TableType {
        TableType::View
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[datafusion::prelude::Expr],
        limit: Option<usize>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        // Query catalog database live
        let batch = self.query_snapshots()?;

        // Use MemTable for execution (MemTable handles projection/filters/limit)
        let mem_table = MemTable::try_new(self.schema.clone(), vec![vec![batch]])?;
        mem_table.scan(state, projection, filters, limit).await
    }
}

/// Live table provider for schemata - queries metadata on every scan
#[derive(Debug)]
pub struct SchemataTable {
    provider: Arc<dyn MetadataProvider>,
    schema: SchemaRef,
}

impl SchemataTable {
    pub fn new(provider: Arc<dyn MetadataProvider>) -> Self {
        let schema = Arc::new(Schema::new(vec![
            Field::new("snapshot_id", DataType::Int64, false),
            Field::new("schema_id", DataType::Int64, false),
            Field::new("schema_name", DataType::Utf8, false),
            Field::new("path", DataType::Utf8, false),
            Field::new("path_is_relative", DataType::Boolean, false),
        ]));
        Self {
            provider,
            schema,
        }
    }

    fn query_schemata(&self) -> DataFusionResult<RecordBatch> {
        let snapshot_id = self
            .provider
            .get_current_snapshot()
            .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;

        let schemas = self
            .provider
            .list_schemas(snapshot_id)
            .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;

        let snapshot_ids: ArrayRef = Arc::new(Int64Array::from(vec![snapshot_id; schemas.len()]));

        let schema_ids: ArrayRef = Arc::new(Int64Array::from(
            schemas.iter().map(|s| s.schema_id).collect::<Vec<_>>(),
        ));

        let schema_names: ArrayRef = Arc::new(StringArray::from(
            schemas
                .iter()
                .map(|s| s.schema_name.as_str())
                .collect::<Vec<_>>(),
        ));

        let paths: ArrayRef = Arc::new(StringArray::from(
            schemas.iter().map(|s| s.path.as_str()).collect::<Vec<_>>(),
        ));

        let path_is_relative: ArrayRef = Arc::new(BooleanArray::from(
            schemas
                .iter()
                .map(|s| s.path_is_relative)
                .collect::<Vec<_>>(),
        ));

        RecordBatch::try_new(
            self.schema.clone(),
            vec![snapshot_ids, schema_ids, schema_names, paths, path_is_relative],
        )
        .map_err(|e| datafusion::error::DataFusionError::ArrowError(Box::new(e), None))
    }
}

#[async_trait::async_trait]
impl TableProvider for SchemataTable {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn table_type(&self) -> TableType {
        TableType::View
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[datafusion::prelude::Expr],
        limit: Option<usize>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        // Query catalog database live
        let batch = self.query_schemata()?;

        // Use MemTable for execution
        let mem_table = MemTable::try_new(self.schema.clone(), vec![vec![batch]])?;
        mem_table.scan(state, projection, filters, limit).await
    }
}

/// Live table provider for tables - queries metadata on every scan
#[derive(Debug)]
pub struct TablesTable {
    provider: Arc<dyn MetadataProvider>,
    schema: SchemaRef,
}

impl TablesTable {
    pub fn new(provider: Arc<dyn MetadataProvider>) -> Self {
        let schema = Arc::new(Schema::new(vec![
            Field::new("snapshot_id", DataType::Int64, false),
            Field::new("schema_name", DataType::Utf8, false),
            Field::new("table_id", DataType::Int64, false),
            Field::new("table_name", DataType::Utf8, false),
            Field::new("path", DataType::Utf8, false),
            Field::new("path_is_relative", DataType::Boolean, false),
        ]));
        Self {
            provider,
            schema,
        }
    }

    fn query_tables(&self) -> DataFusionResult<RecordBatch> {
        let snapshot_id = self
            .provider
            .get_current_snapshot()
            .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;

        // Single bulk query instead of N+1 queries
        let all_tables = self
            .provider
            .list_all_tables(snapshot_id)
            .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;

        let snapshot_ids: ArrayRef =
            Arc::new(Int64Array::from(vec![snapshot_id; all_tables.len()]));

        let schema_names: ArrayRef = Arc::new(StringArray::from(
            all_tables
                .iter()
                .map(|t| t.schema_name.as_str())
                .collect::<Vec<_>>(),
        ));

        let table_ids: ArrayRef = Arc::new(Int64Array::from(
            all_tables
                .iter()
                .map(|t| t.table.table_id)
                .collect::<Vec<_>>(),
        ));

        let table_names: ArrayRef = Arc::new(StringArray::from(
            all_tables
                .iter()
                .map(|t| t.table.table_name.as_str())
                .collect::<Vec<_>>(),
        ));

        let paths: ArrayRef = Arc::new(StringArray::from(
            all_tables
                .iter()
                .map(|t| t.table.path.as_str())
                .collect::<Vec<_>>(),
        ));

        let path_is_relative: ArrayRef = Arc::new(BooleanArray::from(
            all_tables
                .iter()
                .map(|t| t.table.path_is_relative)
                .collect::<Vec<_>>(),
        ));

        RecordBatch::try_new(
            self.schema.clone(),
            vec![snapshot_ids, schema_names, table_ids, table_names, paths, path_is_relative],
        )
        .map_err(|e| datafusion::error::DataFusionError::ArrowError(Box::new(e), None))
    }
}

#[async_trait::async_trait]
impl TableProvider for TablesTable {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn table_type(&self) -> TableType {
        TableType::View
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[datafusion::prelude::Expr],
        limit: Option<usize>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        // Query catalog database live
        let batch = self.query_tables()?;

        // Use MemTable for execution
        let mem_table = MemTable::try_new(self.schema.clone(), vec![vec![batch]])?;
        mem_table.scan(state, projection, filters, limit).await
    }
}

/// Live table provider for views.
#[derive(Debug)]
pub struct ViewsTable {
    provider: Arc<dyn MetadataProvider>,
    schema: SchemaRef,
}

impl ViewsTable {
    pub fn new(provider: Arc<dyn MetadataProvider>) -> Self {
        let schema = Arc::new(Schema::new(vec![
            Field::new("snapshot_id", DataType::Int64, false),
            Field::new("schema_name", DataType::Utf8, false),
            Field::new("view_id", DataType::Int64, false),
            Field::new("view_name", DataType::Utf8, false),
            Field::new("dialect", DataType::Utf8, false),
            Field::new("sql", DataType::Utf8, false),
            Field::new("column_aliases", DataType::Utf8, true),
        ]));
        Self {
            provider,
            schema,
        }
    }

    fn query_views(&self) -> DataFusionResult<RecordBatch> {
        let snapshot_id = self
            .provider
            .get_current_snapshot()
            .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
        let views = self
            .provider
            .list_all_views(snapshot_id)
            .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;

        RecordBatch::try_new(
            Arc::clone(&self.schema),
            vec![
                Arc::new(Int64Array::from(vec![snapshot_id; views.len()])),
                Arc::new(StringArray::from_iter_values(
                    views.iter().map(|view| view.schema_name.as_str()),
                )),
                Arc::new(Int64Array::from_iter_values(
                    views.iter().map(|view| view.view.view_id),
                )),
                Arc::new(StringArray::from_iter_values(
                    views.iter().map(|view| view.view.view_name.as_str()),
                )),
                Arc::new(StringArray::from_iter_values(
                    views.iter().map(|view| view.view.dialect.as_str()),
                )),
                Arc::new(StringArray::from_iter_values(
                    views.iter().map(|view| view.view.sql.as_str()),
                )),
                Arc::new(StringArray::from(
                    views
                        .iter()
                        .map(|view| view.view.column_aliases.as_deref())
                        .collect::<Vec<_>>(),
                )),
            ],
        )
        .map_err(|e| datafusion::error::DataFusionError::ArrowError(Box::new(e), None))
    }
}

#[async_trait::async_trait]
impl TableProvider for ViewsTable {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn table_type(&self) -> TableType {
        TableType::View
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[datafusion::prelude::Expr],
        limit: Option<usize>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let batch = self.query_views()?;
        let mem_table = MemTable::try_new(Arc::clone(&self.schema), vec![vec![batch]])?;
        mem_table.scan(state, projection, filters, limit).await
    }
}

/// Live table provider for columns - queries metadata on every scan
#[derive(Debug)]
pub struct ColumnsTable {
    provider: Arc<dyn MetadataProvider>,
    schema: SchemaRef,
}

impl ColumnsTable {
    pub fn new(provider: Arc<dyn MetadataProvider>) -> Self {
        let schema = Arc::new(Schema::new(vec![
            Field::new("schema_name", DataType::Utf8, false),
            Field::new("table_name", DataType::Utf8, false),
            Field::new("column_id", DataType::Int64, false),
            Field::new("column_name", DataType::Utf8, false),
            Field::new("column_type", DataType::Utf8, false),
        ]));
        Self {
            provider,
            schema,
        }
    }

    fn query_columns(&self) -> DataFusionResult<RecordBatch> {
        let snapshot_id = self
            .provider
            .get_current_snapshot()
            .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;

        // Single bulk query instead of N*M queries
        let all_columns_data = self
            .provider
            .list_all_columns(snapshot_id)
            .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;

        let schema_names: ArrayRef = Arc::new(StringArray::from(
            all_columns_data
                .iter()
                .map(|c| c.schema_name.as_str())
                .collect::<Vec<_>>(),
        ));

        let table_names: ArrayRef = Arc::new(StringArray::from(
            all_columns_data
                .iter()
                .map(|c| c.table_name.as_str())
                .collect::<Vec<_>>(),
        ));

        let column_ids: ArrayRef = Arc::new(Int64Array::from(
            all_columns_data
                .iter()
                .map(|c| c.column.column_id)
                .collect::<Vec<_>>(),
        ));

        let column_names: ArrayRef = Arc::new(StringArray::from(
            all_columns_data
                .iter()
                .map(|c| c.column.column_name.as_str())
                .collect::<Vec<_>>(),
        ));

        let column_types: ArrayRef = Arc::new(StringArray::from(
            all_columns_data
                .iter()
                .map(|c| c.column.column_type.as_str())
                .collect::<Vec<_>>(),
        ));

        RecordBatch::try_new(
            self.schema.clone(),
            vec![schema_names, table_names, column_ids, column_names, column_types],
        )
        .map_err(|e| datafusion::error::DataFusionError::ArrowError(Box::new(e), None))
    }
}

#[async_trait::async_trait]
impl TableProvider for ColumnsTable {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn table_type(&self) -> TableType {
        TableType::View
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[datafusion::prelude::Expr],
        limit: Option<usize>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        // Query catalog database live
        let batch = self.query_columns()?;

        // Use MemTable for execution
        let mem_table = MemTable::try_new(self.schema.clone(), vec![vec![batch]])?;
        mem_table.scan(state, projection, filters, limit).await
    }
}

/// Live table provider for table_info - aggregates file information per table
#[derive(Debug)]
pub struct TableInfoTable {
    provider: Arc<dyn MetadataProvider>,
    schema: SchemaRef,
}

impl TableInfoTable {
    pub fn new(provider: Arc<dyn MetadataProvider>) -> Self {
        let schema = Arc::new(Schema::new(vec![
            Field::new("schema_name", DataType::Utf8, false),
            Field::new("table_name", DataType::Utf8, false),
            Field::new("table_id", DataType::Int64, false),
            Field::new("file_count", DataType::Int64, false),
            Field::new("file_size_bytes", DataType::Int64, false),
            Field::new("delete_file_count", DataType::Int64, false),
            Field::new("delete_file_size_bytes", DataType::Int64, false),
        ]));
        Self {
            provider,
            schema,
        }
    }

    fn query_table_info(&self) -> DataFusionResult<RecordBatch> {
        let snapshot_id = self
            .provider
            .get_current_snapshot()
            .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;

        // Single bulk query instead of N*M queries
        let all_files = self
            .provider
            .list_all_files(snapshot_id)
            .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;

        // Get all tables to include tables with no files
        let all_tables = self
            .provider
            .list_all_tables(snapshot_id)
            .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;

        // Group files by table and aggregate statistics
        use std::collections::HashMap;

        #[derive(Default, Debug)]
        struct TableStats {
            table_id: i64,
            file_count: i64,
            file_size: i64,
            delete_count: i64,
            delete_size: i64,
        }

        type TableKey = (String, String);

        let mut table_stats: HashMap<TableKey, TableStats> = HashMap::new();

        // Initialize all tables with zero stats
        for t in &all_tables {
            table_stats.insert(
                (t.schema_name.clone(), t.table.table_name.clone()),
                TableStats {
                    table_id: t.table.table_id,
                    file_count: 0,
                    file_size: 0,
                    delete_count: 0,
                    delete_size: 0,
                },
            );
        }

        // Aggregate file statistics
        for file in &all_files {
            let key = (file.schema_name.clone(), file.table_name.clone());
            let entry = table_stats.entry(key).or_default();
            entry.file_count += 1;
            entry.file_size += file.file.file.file_size_bytes;
            if file.file.delete_file.is_some() {
                entry.delete_count += 1;
                entry.delete_size += file
                    .file
                    .delete_file
                    .as_ref()
                    .map(|d| d.file_size_bytes)
                    .unwrap_or(0); // delete_file_size_bytes
            }
        }

        // Convert to vector and sort for deterministic output
        let mut all_table_info: Vec<_> = table_stats.into_iter().collect();
        all_table_info.sort_by(|a, b| {
            // Sort by schema_name, then table_name
            a.0.0.cmp(&b.0.0).then_with(|| a.0.1.cmp(&b.0.1))
        });

        // Build arrays in a single pass
        let mut table_names = Vec::with_capacity(all_table_info.len());
        let mut schema_names = Vec::with_capacity(all_table_info.len());
        let mut table_ids = Vec::with_capacity(all_table_info.len());
        let mut file_counts = Vec::with_capacity(all_table_info.len());
        let mut file_sizes = Vec::with_capacity(all_table_info.len());
        let mut delete_file_counts = Vec::with_capacity(all_table_info.len());
        let mut delete_file_sizes = Vec::with_capacity(all_table_info.len());

        for ((schema_name, table_name), stats) in all_table_info {
            schema_names.push(schema_name);
            table_names.push(table_name);
            table_ids.push(stats.table_id);
            file_counts.push(stats.file_count);
            file_sizes.push(stats.file_size);
            delete_file_counts.push(stats.delete_count);
            delete_file_sizes.push(stats.delete_size);
        }

        let schema_names: ArrayRef = Arc::new(StringArray::from(schema_names));
        let table_names: ArrayRef = Arc::new(StringArray::from(table_names));
        let table_ids: ArrayRef = Arc::new(Int64Array::from(table_ids));
        let file_counts: ArrayRef = Arc::new(Int64Array::from(file_counts));
        let file_sizes: ArrayRef = Arc::new(Int64Array::from(file_sizes));
        let delete_file_counts: ArrayRef = Arc::new(Int64Array::from(delete_file_counts));
        let delete_file_sizes: ArrayRef = Arc::new(Int64Array::from(delete_file_sizes));

        RecordBatch::try_new(
            self.schema.clone(),
            vec![
                schema_names,
                table_names,
                table_ids,
                file_counts,
                file_sizes,
                delete_file_counts,
                delete_file_sizes,
            ],
        )
        .map_err(|e| datafusion::error::DataFusionError::ArrowError(Box::new(e), None))
    }
}

#[async_trait::async_trait]
impl TableProvider for TableInfoTable {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn table_type(&self) -> TableType {
        TableType::View
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[datafusion::prelude::Expr],
        limit: Option<usize>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let batch = self.query_table_info()?;
        let mem_table = MemTable::try_new(self.schema.clone(), vec![vec![batch]])?;
        mem_table.scan(state, projection, filters, limit).await
    }
}

/// Live table provider for files - queries metadata on every scan
#[derive(Debug)]
pub struct FilesTable {
    provider: Arc<dyn MetadataProvider>,
    schema: SchemaRef,
}

impl FilesTable {
    pub fn new(provider: Arc<dyn MetadataProvider>) -> Self {
        let schema = Arc::new(Schema::new(vec![
            Field::new("schema_name", DataType::Utf8, false),
            Field::new("table_name", DataType::Utf8, false),
            Field::new("file_path", DataType::Utf8, false),
            Field::new("file_size_bytes", DataType::Int64, false),
            Field::new("record_count", DataType::Int64, true),
            Field::new("has_delete_file", DataType::Boolean, false),
        ]));
        Self {
            provider,
            schema,
        }
    }

    fn query_files(&self) -> DataFusionResult<RecordBatch> {
        let snapshot_id = self
            .provider
            .get_current_snapshot()
            .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;

        // Single bulk query instead of N*M queries
        let all_files_data = self
            .provider
            .list_all_files(snapshot_id)
            .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;

        let schema_names: ArrayRef = Arc::new(StringArray::from(
            all_files_data
                .iter()
                .map(|f| f.schema_name.as_str())
                .collect::<Vec<_>>(),
        ));

        let table_names: ArrayRef = Arc::new(StringArray::from(
            all_files_data
                .iter()
                .map(|f| f.table_name.as_str())
                .collect::<Vec<_>>(),
        ));

        let file_paths: ArrayRef = Arc::new(StringArray::from(
            all_files_data
                .iter()
                .map(|f| f.file.file.path.as_str())
                .collect::<Vec<_>>(),
        ));

        let file_sizes: ArrayRef = Arc::new(Int64Array::from(
            all_files_data
                .iter()
                .map(|f| f.file.file.file_size_bytes)
                .collect::<Vec<_>>(),
        ));

        // Note: record_count might not be available in all catalogs
        let record_counts: ArrayRef = Arc::new(Int64Array::from(
            all_files_data
                .iter()
                .map(|f| f.file.max_row_count)
                .collect::<Vec<_>>(),
        ));

        let has_delete_file: ArrayRef = Arc::new(BooleanArray::from(
            all_files_data
                .iter()
                .map(|f| f.file.delete_file.is_some())
                .collect::<Vec<_>>(),
        ));

        RecordBatch::try_new(
            self.schema.clone(),
            vec![schema_names, table_names, file_paths, file_sizes, record_counts, has_delete_file],
        )
        .map_err(|e| datafusion::error::DataFusionError::ArrowError(Box::new(e), None))
    }
}

#[async_trait::async_trait]
impl TableProvider for FilesTable {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn table_type(&self) -> TableType {
        TableType::View
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[datafusion::prelude::Expr],
        limit: Option<usize>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        // Query catalog database live
        let batch = self.query_files()?;

        // Use MemTable for execution
        let mem_table = MemTable::try_new(self.schema.clone(), vec![vec![batch]])?;
        mem_table.scan(state, projection, filters, limit).await
    }
}

/// Schema provider for information_schema
///
/// Provides live metadata tables that query the catalog database on every access.
/// No upfront data loading - all queries execute fresh against the metadata provider.
#[derive(Debug)]
pub(crate) struct InformationSchemaProvider {
    provider: Arc<dyn MetadataProvider>,
}

impl InformationSchemaProvider {
    pub fn new(provider: Arc<dyn MetadataProvider>) -> Self {
        Self {
            provider,
        }
    }
}

#[async_trait::async_trait]
impl SchemaProvider for InformationSchemaProvider {
    fn table_names(&self) -> Vec<String> {
        vec![
            "snapshots".to_string(),
            "schemata".to_string(),
            "tables".to_string(),
            "views".to_string(),
            "table_info".to_string(),
            "columns".to_string(),
            "files".to_string(),
        ]
    }

    async fn table(&self, name: &str) -> DataFusionResult<Option<Arc<dyn TableProvider>>> {
        // Create table provider on-demand - queries will be live
        let provider: Option<Arc<dyn TableProvider>> = match name {
            "snapshots" => Some(Arc::new(SnapshotsTable::new(self.provider.clone()))),
            "schemata" => Some(Arc::new(SchemataTable::new(self.provider.clone()))),
            "tables" => Some(Arc::new(TablesTable::new(self.provider.clone()))),
            "views" => Some(Arc::new(ViewsTable::new(self.provider.clone()))),
            "table_info" => Some(Arc::new(TableInfoTable::new(self.provider.clone()))),
            "columns" => Some(Arc::new(ColumnsTable::new(self.provider.clone()))),
            "files" => Some(Arc::new(FilesTable::new(self.provider.clone()))),
            _ => None,
        };
        Ok(provider)
    }

    fn table_exist(&self, name: &str) -> bool {
        self.table_names().iter().any(|t| t == name)
    }
}
