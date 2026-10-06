//! DuckLake schema provider implementation

use std::sync::Arc;

use async_trait::async_trait;
use datafusion::catalog::{SchemaProvider, TableProvider};
use datafusion::datasource::object_store::ObjectStoreUrl;
use datafusion::error::Result as DataFusionResult;
use datafusion::logical_expr::TableType;

use crate::catalog::SnapshotSelection;
use crate::metadata_provider::{MetadataProvider, TableMetadata, ViewMetadata};
use crate::path_resolver::resolve_path;
use crate::scan_memo::DuckLakeReadOptions;
use crate::snapshot_consistency::{SnapshotRebind, ViewGuard};
use crate::table::DuckLakeTable;
use crate::view::{UnplannableViewTable, plan_view, resolve_view_definition};

#[cfg(feature = "write")]
use crate::metadata_writer::{ColumnDef, MetadataWriter, WriteMode, validate_name};
#[cfg(feature = "write")]
use datafusion::datasource::MemTable;
use datafusion::error::DataFusionError;

/// Validate table name to prevent path traversal attacks and reject
/// empty, control-character, or overlength names.
///
/// Table names are used to construct file paths, so we must ensure they
/// don't contain path separators or parent directory references.
#[cfg(feature = "write")]
fn validate_table_name(name: &str) -> DataFusionResult<()> {
    // Shared name validation (empty, control chars, length)
    validate_name(name, "Table").map_err(|e| DataFusionError::External(Box::new(e)))?;
    if name.contains('/') || name.contains('\\') || name.contains("..") {
        return Err(DataFusionError::Plan(format!(
            "Invalid table name '{}': must not contain path separators or '..'",
            name
        )));
    }
    // Also reject names that are just dots
    if name.chars().all(|c| c == '.') {
        return Err(DataFusionError::Plan(format!(
            "Invalid table name '{}': must not be only dots",
            name
        )));
    }
    Ok(())
}

/// DuckLake schema provider
///
/// Represents a schema within a DuckLake catalog and provides access to tables.
/// Uses dynamic metadata lookup - tables are queried on-demand from the catalog database.
/// Each lookup resolves its snapshot once, so the metadata it reads is from one snapshot.
#[derive(Debug, Clone)]
pub struct DuckLakeSchema {
    schema_id: i64,
    schema_name: String,
    /// Object store URL for resolving file paths (e.g., s3://bucket/ or file:///)
    object_store_url: Arc<ObjectStoreUrl>,
    provider: Arc<dyn MetadataProvider>,
    /// Snapshot that table lookups read
    snapshot: SnapshotSelection,
    /// Set when tables can be rebuilt at another snapshot: the views they were planned through.
    view_guards: Option<Arc<Vec<ViewGuard>>>,
    /// Schema path for resolving relative table paths
    schema_path: String,
    /// Propagated from the catalog: when true, tables expose a `rowid` column.
    row_lineage: bool,
    /// Propagated from the catalog: how each table reuses its scans' reads.
    read_options: DuckLakeReadOptions,
    /// Metadata writer for write operations (when write feature is enabled)
    #[cfg(feature = "write")]
    writer: Option<Arc<dyn MetadataWriter>>,
    /// Write-layout options propagated from the catalog to each table's INSERT.
    #[cfg(feature = "write")]
    write_options: crate::table_writer::DuckLakeWriteOptions,
}

impl DuckLakeSchema {
    /// Create a new DuckLake schema
    pub fn new(
        schema_id: i64,
        schema_name: impl Into<String>,
        provider: Arc<dyn MetadataProvider>,
        snapshot_id: i64, // Received from catalog
        object_store_url: Arc<ObjectStoreUrl>,
        schema_path: String,
    ) -> Self {
        Self {
            schema_id,
            schema_name: schema_name.into(),
            provider,
            snapshot: SnapshotSelection::Fixed(snapshot_id),
            view_guards: None,
            object_store_url,
            schema_path,
            row_lineage: false,
            read_options: DuckLakeReadOptions::default(),
            #[cfg(feature = "write")]
            writer: None,
            #[cfg(feature = "write")]
            write_options: crate::table_writer::DuckLakeWriteOptions::default(),
        }
    }

    pub(crate) fn with_snapshot_selection(
        mut self,
        snapshot: SnapshotSelection,
        view_guards: Option<Arc<Vec<ViewGuard>>>,
    ) -> Self {
        self.snapshot = snapshot;
        self.view_guards = view_guards;
        self
    }

    pub(crate) fn provider_key(&self) -> usize {
        Arc::as_ptr(&self.provider) as *const () as usize
    }

    /// Enable the row-lineage virtual `rowid` column for all tables in this
    /// schema. Set by the parent catalog (see `DuckLakeCatalog::with_row_lineage`).
    pub fn with_row_lineage(mut self, enabled: bool) -> Self {
        self.row_lineage = enabled;
        self
    }

    /// Set how each table of this schema reuses what its scans read. Set by the
    /// parent catalog (see `DuckLakeCatalog::with_read_options`).
    pub fn with_read_options(mut self, options: DuckLakeReadOptions) -> Self {
        self.read_options = options;
        self
    }

    /// Configure this schema for write operations.
    ///
    /// This method enables write support by attaching a metadata writer.
    /// Once configured, the schema can handle CREATE TABLE AS and tables can handle INSERT INTO.
    ///
    /// # Arguments
    /// * `writer` - Metadata writer for catalog operations
    #[cfg(feature = "write")]
    pub fn with_writer(mut self, writer: Arc<dyn MetadataWriter>) -> Self {
        self.writer = Some(writer);
        self
    }

    /// Set the write-layout options propagated to each table's INSERT path.
    #[cfg(feature = "write")]
    pub fn with_write_options(
        mut self,
        options: crate::table_writer::DuckLakeWriteOptions,
    ) -> Self {
        self.write_options = options;
        self
    }
}

impl DuckLakeSchema {
    fn resolve_snapshot(&self) -> DataFusionResult<i64> {
        self.snapshot
            .resolve(self.provider.as_ref())
            .map_err(|e| DataFusionError::External(Box::new(e)))
    }

    fn build_table(
        &self,
        meta: &TableMetadata,
        snapshot_id: i64,
    ) -> DataFusionResult<DuckLakeTable> {
        // Resolve table path hierarchically using path_resolver utility
        let table_path = resolve_path(&self.schema_path, &meta.path, meta.path_is_relative)
            .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;

        // Pass snapshot_id to table
        let table = DuckLakeTable::new(
            meta.table_id,
            meta.table_name.clone(),
            self.provider.clone(),
            snapshot_id, // Propagate snapshot_id
            self.object_store_url.clone(),
            table_path,
        )
        .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?
        .with_row_lineage(self.row_lineage)
        .with_read_options(self.read_options.clone());

        // Configure writer if this schema is writable
        #[cfg(feature = "write")]
        let table = if let Some(writer) = self.writer.as_ref() {
            let settings = self
                .provider
                .get_metadata_settings(Some(self.schema_id), Some(meta.table_id))
                .map_err(|e| DataFusionError::External(Box::new(e)))?;
            let options =
                crate::table_writer::DuckLakeWriteOptions::from_metadata_settings_deferred(
                    &settings,
                )
                .with_overrides(&self.write_options);
            table
                .with_writer(self.schema_name.clone(), Arc::clone(writer))
                .with_write_options(options)
        } else {
            table
        };

        Ok(match &self.view_guards {
            Some(guards) => table.with_snapshot_rebind(SnapshotRebind::new(
                self.clone(),
                meta.table_name.clone(),
                Arc::clone(guards),
            )),
            None => table,
        })
    }

    pub(crate) fn view_at(
        &self,
        name: &str,
        snapshot_id: i64,
    ) -> DataFusionResult<Option<ViewMetadata>> {
        self.provider
            .get_view_by_name(self.schema_id, name, snapshot_id)
            .map_err(|e| DataFusionError::External(Box::new(e)))
    }

    /// Rebuild a table at another snapshot, for [`SnapshotRebind`].
    pub(crate) fn table_at(
        &self,
        name: &str,
        snapshot_id: i64,
    ) -> DataFusionResult<Option<DuckLakeTable>> {
        self.provider
            .get_table_by_name(self.schema_id, name, snapshot_id)
            .map_err(|e| DataFusionError::External(Box::new(e)))?
            .map(|meta| self.build_table(&meta, snapshot_id))
            .transpose()
    }
}

#[async_trait]
impl SchemaProvider for DuckLakeSchema {
    fn table_names(&self) -> Vec<String> {
        let snapshot_id = match self.snapshot.resolve(self.provider.as_ref()) {
            Ok(snapshot_id) => snapshot_id,
            Err(e) => {
                tracing::error!(error = %e, schema_name = %self.schema_name, "Failed to resolve snapshot");
                return Vec::new();
            },
        };
        let mut names = self
            .provider
            .list_tables(self.schema_id, snapshot_id)
            .inspect_err(|e| {
                tracing::error!(
                    error = %e,
                    schema_id = %self.schema_id,
                    snapshot_id,
                    schema_name = %self.schema_name,
                    "Failed to list tables from catalog"
                )
            })
            .unwrap_or_default()
            .into_iter()
            .map(|t| t.table_name)
            .collect::<Vec<_>>();
        names.extend(
            self.provider
                .list_views(self.schema_id, snapshot_id)
                .inspect_err(|e| {
                    tracing::error!(
                        error = %e,
                        schema_id = %self.schema_id,
                        snapshot_id,
                        schema_name = %self.schema_name,
                        "Failed to list views from catalog"
                    )
                })
                .unwrap_or_default()
                .into_iter()
                .map(|view| view.view_name),
        );
        names.sort();
        names.dedup();
        names
    }

    async fn table(&self, name: &str) -> DataFusionResult<Option<Arc<dyn TableProvider>>> {
        let snapshot_id = self.resolve_snapshot()?;
        match self
            .provider
            .get_table_by_name(self.schema_id, name, snapshot_id)
        {
            Ok(Some(meta)) => self
                .build_table(&meta, snapshot_id)
                .map(|table| Some(Arc::new(table) as Arc<dyn TableProvider>)),
            Ok(None) => match self
                .provider
                .get_view_by_name(self.schema_id, name, snapshot_id)
            {
                Ok(Some(view)) => {
                    let (definition, planned) = match resolve_view_definition(
                        &view,
                        self.provider.as_ref(),
                        snapshot_id,
                        &self.schema_name,
                    ) {
                        Ok(definition) => {
                            let planned = plan_view(
                                &view,
                                &definition,
                                Arc::clone(&self.provider),
                                snapshot_id,
                                &self.schema_name,
                                self.row_lineage,
                                self.read_options.clone(),
                                self.view_guards.as_ref().map(|guards| {
                                    let mut chain = guards.to_vec();
                                    chain.push(ViewGuard::new(view.clone()));
                                    Arc::new(chain)
                                }),
                            )
                            .await;
                            (definition, planned)
                        },
                        Err(e) => (view.sql.clone(), Err(e)),
                    };
                    let table = match planned {
                        Ok(table) => table,
                        Err(e) => Arc::new(UnplannableViewTable::new(definition, &e)),
                    };
                    Ok(Some(table))
                },
                Ok(None) => Ok(None),
                Err(e) => Err(datafusion::error::DataFusionError::External(Box::new(e))),
            },
            Err(e) => Err(datafusion::error::DataFusionError::External(Box::new(e))),
        }
    }

    async fn table_type(&self, name: &str) -> DataFusionResult<Option<TableType>> {
        let snapshot_id = self.resolve_snapshot()?;
        if self
            .provider
            .table_exists(self.schema_id, name, snapshot_id)
            .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?
        {
            return Ok(Some(TableType::Base));
        }
        self.provider
            .get_view_by_name(self.schema_id, name, snapshot_id)
            .map(|view| view.map(|_| TableType::View))
            .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))
    }

    fn table_exist(&self, name: &str) -> bool {
        let Ok(snapshot_id) = self.snapshot.resolve(self.provider.as_ref()) else {
            return false;
        };
        self.provider
            .table_exists(self.schema_id, name, snapshot_id)
            .unwrap_or(false)
            || self
                .provider
                .get_view_by_name(self.schema_id, name, snapshot_id)
                .map(|view| view.is_some())
                .unwrap_or(false)
    }

    /// Register a new table in this schema.
    ///
    /// This is called by DataFusion for CREATE TABLE AS SELECT statements.
    /// It creates the table metadata in the catalog and returns a writable table provider.
    #[cfg(feature = "write")]
    fn register_table(
        &self,
        name: String,
        table: Arc<dyn TableProvider>,
    ) -> DataFusionResult<Option<Arc<dyn TableProvider>>> {
        // Validate table name to prevent path traversal attacks
        validate_table_name(&name)?;

        let snapshot_id = self.resolve_snapshot()?;
        if self
            .provider
            .get_view_by_name(self.schema_id, &name, snapshot_id)
            .map_err(|e| DataFusionError::External(Box::new(e)))?
            .is_some()
        {
            return Err(DataFusionError::Plan(format!(
                "Cannot create table '{name}': a view with that name already exists"
            )));
        }

        let writer = self.writer.as_ref().ok_or_else(|| {
            DataFusionError::Plan(
                "Schema is read-only. Use DuckLakeCatalog::with_writer() to enable writes."
                    .to_string(),
            )
        })?;

        // DataFusion supplies materialized CTAS rows in a MemTable. Publishing only
        // its schema would discard those rows.
        if let Some(memory) = table.downcast_ref::<MemTable>() {
            for partition in &memory.batches {
                let batches = partition.try_read().map_err(|_| {
                    DataFusionError::Execution(
                        "Cannot register an in-memory table while its rows are being modified"
                            .to_string(),
                    )
                })?;
                if batches.iter().any(|batch| batch.num_rows() > 0) {
                    return Err(DataFusionError::NotImplemented(
                        "CREATE TABLE AS SELECT with rows is not supported; use CREATE TABLE followed by INSERT INTO ... SELECT".to_string(),
                    ));
                }
            }
        }

        // Convert Arrow schema to ColumnDefs
        let arrow_schema = table.schema();
        let columns: Vec<ColumnDef> = arrow_schema
            .fields()
            .iter()
            .map(|field| {
                ColumnDef::from_arrow(field.name(), field.data_type(), field.is_nullable())
                    .map_err(|e| DataFusionError::External(Box::new(e)))
            })
            .collect::<DataFusionResult<Vec<_>>>()?;

        // Create table in metadata (creates snapshot, table, columns in a transaction)
        let setup = writer
            .begin_write_transaction(&self.schema_name, &name, &columns, WriteMode::Replace)
            .map_err(|e| DataFusionError::External(Box::new(e)))?;

        // CREATE TABLE registers no data file, so publish the head here; without
        // this the new (empty) table never becomes visible on multicatalog
        // backends that defer the head advance out of begin_write_transaction.
        let committed = writer
            .publish_snapshot(
                setup.table_id,
                &self.schema_name,
                &name,
                setup.snapshot_id,
                WriteMode::Replace,
                setup.base_snapshot_id,
                &columns,
                &setup.field_ids,
            )
            .map_err(|e| DataFusionError::External(Box::new(e)))?;

        // Resolve table path
        let table_path = resolve_path(&self.schema_path, &name, true)
            .map_err(|e| DataFusionError::External(Box::new(e)))?;

        // Build the table provider from the COMMITTED ids/snapshot (the begin-time
        // `setup.snapshot_id` is vestigial on the commit-time path; the real
        // snapshot is assigned at commit).
        let settings = self
            .provider
            .get_metadata_settings(Some(self.schema_id), Some(committed.table_id))
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        let options =
            crate::table_writer::DuckLakeWriteOptions::from_metadata_settings_deferred(&settings)
                .with_overrides(&self.write_options);
        let writable_table = DuckLakeTable::new(
            committed.table_id,
            name,
            self.provider.clone(),
            committed.snapshot_id,
            self.object_store_url.clone(),
            table_path,
        )
        .map_err(|e| DataFusionError::External(Box::new(e)))?
        .with_read_options(self.read_options.clone())
        .with_writer(self.schema_name.clone(), Arc::clone(writer))
        .with_write_options(options);

        Ok(Some(Arc::new(writable_table) as Arc<dyn TableProvider>))
    }
}

#[cfg(all(test, feature = "write"))]
mod tests {
    use super::*;

    #[test]
    fn test_validate_table_name_valid() {
        assert!(validate_table_name("users").is_ok());
        assert!(validate_table_name("my_table").is_ok());
        assert!(validate_table_name("Table123").is_ok());
        assert!(validate_table_name("a").is_ok());
    }

    #[test]
    fn test_validate_table_name_empty() {
        let result = validate_table_name("");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("cannot be empty"));
    }

    #[test]
    fn test_validate_table_name_path_traversal() {
        // Forward slash
        let result = validate_table_name("../etc/passwd");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("path separators"));

        // Backslash
        let result = validate_table_name("..\\windows\\system32");
        assert!(result.is_err());

        // Double dot
        let result = validate_table_name("foo..bar");
        assert!(result.is_err());

        // Just slashes
        let result = validate_table_name("foo/bar");
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_table_name_only_dots() {
        assert!(validate_table_name(".").is_err());
        assert!(validate_table_name("..").is_err());
        assert!(validate_table_name("...").is_err());
    }

    #[test]
    fn test_validate_table_name_control_chars() {
        assert!(validate_table_name("table\0name").is_err());
        assert!(validate_table_name("table\nname").is_err());
        assert!(validate_table_name("table\tname").is_err());
        assert!(validate_table_name("\x7Ftable").is_err());
    }
}
