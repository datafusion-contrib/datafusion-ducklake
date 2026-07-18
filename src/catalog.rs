//! DuckLake catalog provider implementation

use std::sync::Arc;

use crate::Result;
use crate::information_schema::InformationSchemaProvider;
use crate::metadata_provider::MetadataProvider;
use crate::path_resolver::{parse_object_store_url, resolve_path};
use crate::schema::DuckLakeSchema;
use datafusion::catalog::{CatalogProvider, SchemaProvider};
use datafusion::datasource::object_store::ObjectStoreUrl;

#[cfg(feature = "write")]
use crate::metadata_writer::MetadataWriter;

/// Configuration for write operations (when write feature is enabled)
#[cfg(feature = "write")]
#[derive(Debug, Clone)]
struct WriteConfig {
    /// Metadata writer for catalog operations
    writer: Arc<dyn MetadataWriter>,
}

/// DuckLake catalog provider
///
/// Connects to a DuckLake catalog database and provides access to schemas and tables.
/// Uses dynamic metadata lookup - schemas are queried on-demand from the catalog database.
/// Bound to a specific snapshot ID for query consistency.
#[derive(Debug)]
pub struct DuckLakeCatalog {
    /// Metadata provider for querying catalog
    provider: Arc<dyn MetadataProvider>,
    /// Snapshot ID this catalog is bound to (for query consistency)
    snapshot_id: i64,
    /// Object store URL for resolving file paths (e.g., s3://bucket/ or file:///)
    object_store_url: Arc<ObjectStoreUrl>,
    /// Catalog base path component for resolving relative schema paths (e.g., /prefix/)
    catalog_path: String,
    /// When true, expose a virtual `rowid` BIGINT column on every table
    /// (DuckLake row-lineage feature). Default: false, to preserve existing
    /// `SELECT *` shape for callers that haven't opted in.
    row_lineage: bool,
    /// Write configuration (when write feature is enabled)
    #[cfg(feature = "write")]
    write_config: Option<WriteConfig>,
}

impl DuckLakeCatalog {
    /// Create a new DuckLake catalog with a metadata provider
    ///
    /// Gets the current snapshot ID at creation time and binds the catalog to it.
    /// For backward compatibility. For explicit snapshot control, use `with_snapshot()`.
    pub fn new(provider: impl MetadataProvider + 'static) -> Result<Self> {
        let provider = Arc::new(provider) as Arc<dyn MetadataProvider>;
        let snapshot_id = provider.get_current_snapshot()?;
        let data_path = provider.get_data_path()?;
        let (object_store_url, catalog_path) = parse_object_store_url(&data_path)?;

        Ok(Self {
            provider,
            snapshot_id,
            object_store_url: Arc::new(object_store_url),
            catalog_path,
            row_lineage: false,
            #[cfg(feature = "write")]
            write_config: None,
        })
    }

    /// Create a catalog bound to a specific snapshot ID
    ///
    /// All schemas and tables returned will use this snapshot, guaranteeing
    /// query consistency even if multiple catalog/schema/table lookups occur
    /// during query planning.
    pub fn with_snapshot(provider: Arc<dyn MetadataProvider>, snapshot_id: i64) -> Result<Self> {
        let data_path = provider.get_data_path()?;
        let (object_store_url, catalog_path) = parse_object_store_url(&data_path)?;

        Ok(Self {
            provider,
            snapshot_id,
            object_store_url: Arc::new(object_store_url),
            catalog_path,
            row_lineage: false,
            #[cfg(feature = "write")]
            write_config: None,
        })
    }

    /// Create a catalog with write support.
    ///
    /// This constructor enables write operations (INSERT INTO, CREATE TABLE AS)
    /// by attaching a metadata writer. The catalog will pass the writer to all
    /// schemas and tables it creates.
    ///
    /// # Arguments
    /// * `provider` - Metadata provider for reading catalog metadata
    /// * `writer` - Metadata writer for write operations
    ///
    /// # Example
    /// ```no_run
    /// # async fn example() -> datafusion_ducklake::Result<()> {
    /// use datafusion_ducklake::{DuckLakeCatalog, SqliteMetadataProvider, SqliteMetadataWriter};
    /// use std::sync::Arc;
    ///
    /// let provider = SqliteMetadataProvider::new("sqlite:catalog.db?mode=rwc").await?;
    /// // Use `new_with_init` for a writable catalog: it creates the schema if
    /// // absent AND runs idempotent migrations (e.g. upgrading a legacy
    /// // `ducklake_column` so type promotion works). Plain `new()` is connect-only
    /// // and skips migrations, so a pre-existing catalog opened that way is not upgraded.
    /// let writer = SqliteMetadataWriter::new_with_init("sqlite:catalog.db?mode=rwc").await?;
    ///
    /// let catalog = DuckLakeCatalog::with_writer(Arc::new(provider), Arc::new(writer))?;
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "write")]
    pub fn with_writer(
        provider: Arc<dyn MetadataProvider>,
        writer: Arc<dyn MetadataWriter>,
    ) -> Result<Self> {
        let snapshot_id = provider.get_current_snapshot()?;
        let data_path_str = provider.get_data_path()?;
        let (object_store_url, catalog_path) = parse_object_store_url(&data_path_str)?;

        Ok(Self {
            provider,
            snapshot_id,
            object_store_url: Arc::new(object_store_url),
            catalog_path,
            row_lineage: false,
            write_config: Some(WriteConfig {
                writer,
            }),
        })
    }

    /// Enable the DuckLake row-lineage feature: every table will expose a
    /// virtual `rowid` BIGINT column (assigned from each row's `row_id_start +
    /// position_in_file`). Off by default to preserve existing `SELECT *`
    /// shape.
    ///
    /// Note: DataFusion has no hidden-column concept, so the `rowid` column
    /// IS included in `SELECT *` once enabled — this differs from the DuckDB
    /// extension where `rowid` is hidden unless explicitly referenced.
    pub fn with_row_lineage(mut self, enabled: bool) -> Self {
        self.row_lineage = enabled;
        self
    }

    /// Get the metadata provider for this catalog
    ///
    /// This is useful when you need to register table functions separately.
    pub fn provider(&self) -> Arc<dyn MetadataProvider> {
        self.provider.clone()
    }

    /// The metadata writer this catalog was configured with, if any (i.e. it was
    /// built via [`DuckLakeCatalog::with_writer`]). Used by
    /// [`crate::execute_ducklake_sql`] to run partition DDL. Returns `None` for a
    /// read-only catalog.
    #[cfg(feature = "write")]
    pub fn writer(&self) -> Option<Arc<dyn MetadataWriter>> {
        self.write_config
            .as_ref()
            .map(|config| Arc::clone(&config.writer))
    }
}

impl CatalogProvider for DuckLakeCatalog {
    fn schema_names(&self) -> Vec<String> {
        // Start with information_schema
        let mut names = vec!["information_schema".to_string()];

        // Add data schemas from catalog using the pinned snapshot_id
        let data_schemas = self
            .provider
            .list_schemas(self.snapshot_id)
            .inspect_err(|e| {
                tracing::error!(
                    error = %e,
                    snapshot_id = %self.snapshot_id,
                    "Failed to list schemas from catalog"
                )
            })
            .unwrap_or_default()
            .into_iter()
            .map(|s| s.schema_name);

        names.extend(data_schemas);

        // Ensure deterministic order and no duplicates
        names.sort();
        names.dedup();

        names
    }

    fn schema(&self, name: &str) -> Option<Arc<dyn SchemaProvider>> {
        // Handle information_schema specially
        if name == "information_schema" {
            return Some(Arc::new(InformationSchemaProvider::new(Arc::clone(
                &self.provider,
            ))));
        }

        // Query database with the pinned snapshot_id for data schemas
        match self.provider.get_schema_by_name(name, self.snapshot_id) {
            Ok(Some(meta)) => {
                // Resolve schema path hierarchically using path_resolver utility
                let schema_path =
                    match resolve_path(&self.catalog_path, &meta.path, meta.path_is_relative) {
                        Ok(p) => p,
                        Err(e) => {
                            tracing::error!(
                                error = %e,
                                schema_name = %name,
                                "Failed to resolve schema path"
                            );
                            return None;
                        },
                    };

                // Pass the pinned snapshot_id to schema
                let schema = DuckLakeSchema::new(
                    meta.schema_id,
                    meta.schema_name,
                    Arc::clone(&self.provider),
                    self.snapshot_id, // Propagate pinned snapshot_id
                    self.object_store_url.clone(),
                    schema_path,
                )
                .with_row_lineage(self.row_lineage);

                // Configure writer if this catalog is writable
                #[cfg(feature = "write")]
                let schema = if let Some(ref config) = self.write_config {
                    schema.with_writer(Arc::clone(&config.writer))
                } else {
                    schema
                };

                Some(Arc::new(schema) as Arc<dyn SchemaProvider>)
            },
            _ => None,
        }
    }
}
