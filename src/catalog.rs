//! DuckLake catalog provider implementation

use std::sync::Arc;

use crate::Result;
use crate::information_schema::InformationSchemaProvider;
use crate::metadata_provider::{MetadataProvider, resolve_snapshot_at_or_before};
use crate::path_resolver::{parse_object_store_url, resolve_path};
use crate::schema::DuckLakeSchema;
use crate::snapshot_consistency::{ViewGuard, register_snapshot_consistency};
use datafusion::catalog::{CatalogProvider, SchemaProvider};
use datafusion::datasource::object_store::ObjectStoreUrl;
use datafusion::prelude::SessionContext;

#[cfg(feature = "write")]
use crate::metadata_writer::MetadataWriter;

/// Configuration for write operations (when write feature is enabled)
#[cfg(feature = "write")]
#[derive(Debug, Clone)]
struct WriteConfig {
    /// Metadata writer for catalog operations
    writer: Arc<dyn MetadataWriter>,
    /// Write-layout options (compression, row-group caps, file-rollover target)
    /// applied to the writer built for each INSERT.
    options: crate::table_writer::DuckLakeWriteOptions,
}

/// Snapshot that lookups read: one fixed ID, or the latest at each lookup.
#[derive(Debug, Clone, Copy)]
pub(crate) enum SnapshotSelection {
    Fixed(i64),
    Latest,
}

impl SnapshotSelection {
    pub(crate) fn resolve(self, provider: &dyn MetadataProvider) -> Result<i64> {
        match self {
            Self::Fixed(snapshot_id) => Ok(snapshot_id),
            Self::Latest => provider.get_current_snapshot(),
        }
    }
}

/// DuckLake catalog provider
///
/// Connects to a DuckLake catalog database and provides access to schemas and tables.
/// Uses dynamic metadata lookup - schemas are queried on-demand from the catalog database.
/// Catalogs from [`DuckLakeCatalog::new`] and [`DuckLakeCatalog::with_writer`] read the latest
/// snapshot at each lookup, as a new DuckDB transaction does, so commits from any writer are
/// visible to the next statement. [`DuckLakeCatalog::with_snapshot`] and
/// [`DuckLakeCatalog::with_snapshot_at`] stay bound to one snapshot ID.
#[derive(Debug)]
pub struct DuckLakeCatalog {
    /// Metadata provider for querying catalog
    provider: Arc<dyn MetadataProvider>,
    /// Snapshot that schema and table lookups read
    snapshot: SnapshotSelection,
    /// Set when tables can be rebuilt at another snapshot: the views they were planned through.
    view_guards: Option<Arc<Vec<ViewGuard>>>,
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
    /// Reads the latest snapshot at each lookup. For a fixed view, use `with_snapshot()`.
    pub fn new(provider: impl MetadataProvider + 'static) -> Result<Self> {
        let provider = Arc::new(provider) as Arc<dyn MetadataProvider>;
        provider.get_current_snapshot()?;
        let data_path = provider.get_data_path()?;
        let (object_store_url, catalog_path) = parse_object_store_url(&data_path)?;

        Ok(Self {
            provider,
            snapshot: SnapshotSelection::Latest,
            view_guards: Some(Arc::new(Vec::new())),
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
            snapshot: SnapshotSelection::Fixed(snapshot_id),
            view_guards: None,
            object_store_url: Arc::new(object_store_url),
            catalog_path,
            row_lineage: false,
            #[cfg(feature = "write")]
            write_config: None,
        })
    }

    /// Create a read-only catalog bound to the latest snapshot at or before a
    /// UTC timestamp. When snapshots share a timestamp, selects the highest
    /// snapshot ID so every commit recorded at that instant is visible.
    pub fn with_snapshot_at(
        provider: Arc<dyn MetadataProvider>,
        timestamp: chrono::DateTime<chrono::Utc>,
    ) -> Result<Self> {
        let snapshot_id = resolve_snapshot_at_or_before(provider.as_ref(), timestamp.naive_utc())?;
        Self::with_snapshot(provider, snapshot_id)
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
        provider.get_current_snapshot()?;
        let data_path_str = provider.get_data_path()?;
        let (object_store_url, catalog_path) = parse_object_store_url(&data_path_str)?;

        Ok(Self {
            provider,
            snapshot: SnapshotSelection::Latest,
            view_guards: Some(Arc::new(Vec::new())),
            object_store_url: Arc::new(object_store_url),
            catalog_path,
            row_lineage: false,
            write_config: Some(WriteConfig {
                writer,
                options: crate::table_writer::DuckLakeWriteOptions::default(),
            }),
        })
    }

    /// Set the write-layout options (compression, row-group caps, file-rollover
    /// target) applied to every `INSERT` through this catalog. No-op on a
    /// read-only catalog. Writes roll over at `target_file_size` by default, so
    /// with a sort order each INSERT lands as several files each covering a
    /// contiguous value range — enabling file-level pruning.
    #[cfg(feature = "write")]
    pub fn with_write_options(
        mut self,
        options: crate::table_writer::DuckLakeWriteOptions,
    ) -> Self {
        if let Some(config) = self.write_config.as_mut() {
            config.options = options;
        }
        self
    }

    /// Register a catalog on a session under `name`.
    ///
    /// A catalog from [`DuckLakeCatalog::new`] or [`DuckLakeCatalog::with_writer`] reads the
    /// latest snapshot at each lookup, so this also calls [`register_snapshot_consistency`]
    /// to keep a statement over several tables on one snapshot. A catalog bound to a snapshot
    /// is registered as it is.
    pub fn register(ctx: &SessionContext, name: &str, catalog: DuckLakeCatalog) {
        if catalog.view_guards.is_some() {
            register_snapshot_consistency(ctx);
        }
        ctx.register_catalog(name, Arc::new(catalog));
    }

    /// Make this catalog's tables rebuildable at another snapshot, under `view_guards`.
    pub(crate) fn with_view_guards(mut self, view_guards: Option<Arc<Vec<ViewGuard>>>) -> Self {
        self.view_guards = view_guards;
        self
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

        let data_schemas = self
            .snapshot
            .resolve(self.provider.as_ref())
            .and_then(|snapshot_id| self.provider.list_schemas(snapshot_id))
            .inspect_err(|e| tracing::error!(error = %e, "Failed to list schemas from catalog"))
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

        let snapshot_id = match self.snapshot.resolve(self.provider.as_ref()) {
            Ok(snapshot_id) => snapshot_id,
            Err(e) => {
                tracing::error!(error = %e, schema_name = %name, "Failed to resolve snapshot");
                return None;
            },
        };
        match self.provider.get_schema_by_name(name, snapshot_id) {
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

                let schema = DuckLakeSchema::new(
                    meta.schema_id,
                    meta.schema_name,
                    Arc::clone(&self.provider),
                    snapshot_id,
                    self.object_store_url.clone(),
                    schema_path,
                )
                .with_snapshot_selection(self.snapshot, self.view_guards.clone())
                .with_row_lineage(self.row_lineage);

                // Configure writer if this catalog is writable
                #[cfg(feature = "write")]
                let schema = if let Some(ref config) = self.write_config {
                    schema
                        .with_writer(Arc::clone(&config.writer))
                        .with_write_options(config.options.clone())
                } else {
                    schema
                };

                Some(Arc::new(schema) as Arc<dyn SchemaProvider>)
            },
            Ok(None) => None,
            Err(e) => {
                tracing::error!(error = %e, schema_name = %name, "Failed to look up schema");
                None
            },
        }
    }
}
