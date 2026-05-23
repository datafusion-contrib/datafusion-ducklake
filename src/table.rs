//! DuckLake table provider implementation

use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::Result;
use crate::column_rename::ColumnRenameExec;
use crate::delete_filter::DeleteFilterExec;
use crate::metadata_provider::{
    DuckLakeFileData, DuckLakeTableColumn, DuckLakeTableFile, MetadataProvider,
};
use crate::path_resolver::resolve_path;
use crate::row_id::{ROW_ID_PARQUET_FIELD_ID, ROWID_COLUMN_NAME, RowIdExec, rowid_field};
use crate::types::{
    build_arrow_schema, build_read_schema_with_field_id_mapping, extract_parquet_field_ids,
};

#[cfg(feature = "write")]
use crate::insert_exec::DuckLakeInsertExec;
#[cfg(feature = "write")]
use crate::metadata_writer::{MetadataWriter, WriteMode};

#[cfg(feature = "encryption")]
use crate::encryption::EncryptionFactoryBuilder;
use arrow::array::{Array, Int64Array};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use async_trait::async_trait;
use datafusion::catalog::{Session, TableProvider};
use datafusion::common::Statistics;
use datafusion::common::stats::Precision;
use datafusion::datasource::listing::PartitionedFile;
use datafusion::datasource::physical_plan::{FileGroup, FileScanConfigBuilder, ParquetSource};
use datafusion::datasource::source::DataSourceExec;
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::execution::object_store::ObjectStoreUrl;
#[cfg(feature = "write")]
use datafusion::logical_expr::dml::InsertOp;
use datafusion::logical_expr::{Expr, TableProviderFilterPushDown, TableType};
use datafusion::physical_plan::ExecutionPlan;
use futures::StreamExt;
use object_store::path::Path as ObjectPath;
use parquet::arrow::ParquetRecordBatchStreamBuilder;
use parquet::arrow::async_reader::ParquetObjectReader;
use tokio::sync::OnceCell;

#[cfg(feature = "encryption")]
use datafusion::execution::parquet_encryption::EncryptionFactory;

// Delete file schema constants (public for testing)
pub const DELETE_FILE_PATH_COL: &str = "file_path";
pub const DELETE_POS_COL: &str = "pos";

/// Validate and convert file_size_bytes from i64 (as stored in DuckLake metadata) to u64.
///
/// DuckLake stores file sizes as signed integers in SQL. A negative value indicates
/// corrupt or invalid metadata. Without this check, a negative i64 cast to u64 would
/// wrap to a huge value (e.g., -1 becomes u64::MAX), causing confusing downstream errors.
pub(crate) fn validated_file_size(file_size_bytes: i64, file_path: &str) -> DataFusionResult<u64> {
    u64::try_from(file_size_bytes).map_err(|_| {
        DataFusionError::Execution(format!(
            "Invalid file_size_bytes ({}) for file '{}': value must be non-negative",
            file_size_bytes, file_path
        ))
    })
}

/// Validate and convert record_count from i64 (as stored in DuckLake metadata) to u64.
///
/// DuckLake stores record counts as signed integers in SQL. A negative value indicates
/// corrupt or invalid metadata. Without this check, a negative record_count would cause
/// incorrect behavior (e.g., empty ranges in full-file deletes, or incorrect row filtering).
pub(crate) fn validated_record_count(record_count: i64, file_path: &str) -> DataFusionResult<u64> {
    u64::try_from(record_count).map_err(|_| {
        DataFusionError::Execution(format!(
            "Invalid record_count ({}) for file '{}': value must be non-negative",
            record_count, file_path
        ))
    })
}

/// Returns the expected schema for DuckLake delete files
///
/// Delete files have a standard schema: (file_path: VARCHAR, pos: INT64)
/// The file_path column is metadata/documentation only (for Iceberg compatibility).
/// The pos column contains the row positions to delete.
pub fn delete_file_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new(DELETE_FILE_PATH_COL, DataType::Utf8, false),
        Field::new(DELETE_POS_COL, DataType::Int64, false),
    ]))
}

/// Cached schema mapping for renamed columns
type SchemaMappingCache = (SchemaRef, HashMap<String, String>);

/// Per-file read configuration computed for the row-lineage scan path.
///
/// Encapsulates the decision made by `DuckLakeMultiFileReader::GetVirtualColumnExpression`
/// in the C++ extension: either the parquet file embeds a row-id column
/// (UPDATE/compaction case — surviving rowids preserved across file rewrite),
/// or it doesn't (INSERT-only case — synthesize from `row_id_start + position`).
#[derive(Debug, Clone)]
struct FileReadConfig {
    /// Schema we pass to `ParquetSource::new` for this file. When
    /// `embedded_rowid_parquet_name` is `Some`, this schema has the embedded
    /// rowid column appended at the end (under its parquet name).
    read_schema: SchemaRef,
    /// Parquet-name → user-facing-name renames. Includes the rowid rename
    /// (parquet column → `"rowid"`) when the file has an embedded column with
    /// a different name.
    name_mapping: HashMap<String, String>,
    /// `Some(parquet_column_name)` if the file embeds the rowid column
    /// (tagged with [`ROW_ID_PARQUET_FIELD_ID`]); `None` otherwise.
    embedded_rowid_parquet_name: Option<String>,
}

/// DuckLake table provider
///
/// Represents a table within a DuckLake schema and provides access to data via Parquet files.
/// Caches snapshot_id and uses it to load all metadata atomically.
pub struct DuckLakeTable {
    #[allow(dead_code)]
    table_id: i64,
    table_name: String,
    #[allow(dead_code)]
    provider: Arc<dyn MetadataProvider>,
    /// Object store URL for resolving file paths (e.g., s3://bucket/ or file:///)
    object_store_url: Arc<ObjectStoreUrl>,
    /// Table path for resolving relative file paths
    table_path: String,
    /// User-facing schema. Equals `physical_schema` when row lineage is off, or
    /// `physical_schema` with a `rowid` BIGINT appended at the end when on.
    schema: SchemaRef,
    /// Schema of the physical (parquet-backed) columns only — no rowid.
    physical_schema: SchemaRef,
    /// When true, `schema` includes a trailing `rowid` column and `scan()`
    /// injects it per-file via [`RowIdExec`].
    row_lineage: bool,
    /// Column metadata from DuckLake (needed for field_id mapping)
    columns: Vec<DuckLakeTableColumn>,
    /// Table files with paths as stored in metadata (resolved on-the-fly when needed)
    table_files: Vec<DuckLakeTableFile>,
    /// Cached schema mapping (read_schema, name_mapping) - computed once on first scan
    schema_mapping_cache: OnceCell<SchemaMappingCache>,
    /// Per-file row-lineage read config, populated lazily on the rowid scan
    /// path. Each file requires its own parquet metadata read to detect an
    /// embedded `_ducklake_internal_row_id` column; we memoize so repeated
    /// scans don't re-fetch.
    file_read_config_cache: std::sync::Mutex<HashMap<String, Arc<FileReadConfig>>>,
    /// Encryption factory for decrypting encrypted Parquet files (when encryption feature is enabled)
    #[cfg(feature = "encryption")]
    encryption_factory: Option<Arc<dyn EncryptionFactory>>,
    /// Schema name (needed for write operations)
    #[cfg(feature = "write")]
    schema_name: Option<String>,
    /// Metadata writer for write operations (when write feature is enabled)
    #[cfg(feature = "write")]
    writer: Option<Arc<dyn MetadataWriter>>,
}

impl std::fmt::Debug for DuckLakeTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DuckLakeTable")
            .field("table_id", &self.table_id)
            .field("table_name", &self.table_name)
            .field("table_path", &self.table_path)
            .field("schema", &self.schema)
            .field("columns", &self.columns)
            .field("table_files", &self.table_files)
            .finish_non_exhaustive()
    }
}

impl DuckLakeTable {
    /// Create a new DuckLake table
    pub fn new(
        table_id: i64,
        table_name: impl Into<String>,
        provider: Arc<dyn MetadataProvider>,
        snapshot_id: i64, // Received from schema
        object_store_url: Arc<ObjectStoreUrl>,
        table_path: String,
    ) -> Result<Self> {
        // Load ALL metadata with this snapshot_id
        let columns = provider.get_table_structure(table_id)?;
        let physical_schema = Arc::new(build_arrow_schema(&columns)?);
        let schema = physical_schema.clone();
        let table_files = provider.get_table_files_for_select(table_id, snapshot_id)?;

        // Build encryption factory from file encryption keys (when encryption feature is enabled)
        #[cfg(feature = "encryption")]
        let encryption_factory = {
            let mut builder = EncryptionFactoryBuilder::new();
            for table_file in &table_files {
                // Resolve the file path for the mapping
                let resolved_path = resolve_path(
                    &table_path,
                    &table_file.file.path,
                    table_file.file.path_is_relative,
                )?;
                builder.add_file(&resolved_path, table_file.file.encryption_key.as_deref());

                // Also add delete file encryption key if present
                if let Some(ref delete_file) = table_file.delete_file {
                    let resolved_delete_path =
                        resolve_path(&table_path, &delete_file.path, delete_file.path_is_relative)?;
                    builder.add_file(&resolved_delete_path, delete_file.encryption_key.as_deref());
                }
            }
            let factory = builder.build();
            if factory.has_encrypted_files() {
                Some(Arc::new(factory) as Arc<dyn EncryptionFactory>)
            } else {
                None
            }
        };

        Ok(Self {
            table_id,
            table_name: table_name.into(),
            provider,
            object_store_url,
            table_path,
            schema,
            physical_schema,
            row_lineage: false,
            columns,
            table_files,
            #[cfg(feature = "encryption")]
            encryption_factory,
            schema_mapping_cache: OnceCell::new(),
            file_read_config_cache: std::sync::Mutex::new(HashMap::new()),
            #[cfg(feature = "write")]
            schema_name: None,
            #[cfg(feature = "write")]
            writer: None,
        })
    }

    /// Enable / disable the row-lineage feature. When enabled, the table's
    /// public schema includes a trailing `rowid` BIGINT column synthesized
    /// from each row's catalog-recorded `row_id_start + position_in_file`.
    pub fn with_row_lineage(mut self, enabled: bool) -> Self {
        self.row_lineage = enabled;
        self.schema = if enabled {
            let mut fields: Vec<Arc<Field>> =
                self.physical_schema.fields().iter().cloned().collect();
            fields.push(Arc::new(rowid_field()));
            Arc::new(Schema::new(fields))
        } else {
            self.physical_schema.clone()
        };
        self
    }

    /// Index of the synthetic `rowid` column in `self.schema`, when enabled.
    fn rowid_index(&self) -> Option<usize> {
        self.row_lineage
            .then(|| self.physical_schema.fields().len())
    }

    /// Resolve a file path (data or delete file) to its absolute path
    fn resolve_file_path(&self, file: &DuckLakeFileData) -> DataFusionResult<String> {
        resolve_path(&self.table_path, &file.path, file.path_is_relative)
            .map_err(|e| DataFusionError::External(Box::new(e)))
    }

    /// Create a ParquetSource with encryption support if enabled and needed
    fn create_parquet_source(&self, schema: SchemaRef) -> ParquetSource {
        #[cfg(feature = "encryption")]
        if let Some(ref factory) = self.encryption_factory {
            return ParquetSource::new(schema).with_encryption_factory(Arc::clone(factory));
        }
        ParquetSource::new(schema)
    }

    /// Get the cached schema mapping, computing it once from the first file if needed.
    /// All files in a DuckLake table have the same schema structure, so we only need to check one.
    async fn get_schema_mapping(
        &self,
        state: &dyn Session,
    ) -> DataFusionResult<&SchemaMappingCache> {
        self.schema_mapping_cache
            .get_or_try_init(|| async {
                // If no files, use current schema with no rename mapping
                let Some(first_file) = self.table_files.first() else {
                    return Ok((self.schema.clone(), HashMap::new()));
                };

                let resolved_path = self.resolve_file_path(&first_file.file)?;
                let object_store = state
                    .runtime_env()
                    .object_store(self.object_store_url.as_ref())?;
                let object_path = ObjectPath::from(resolved_path.as_str());

                let reader = ParquetObjectReader::new(object_store, object_path);

                // Build the ParquetRecordBatchStreamBuilder with decryption if needed
                #[cfg(feature = "encryption")]
                let builder = {
                    use parquet::arrow::arrow_reader::ArrowReaderOptions;

                    // Check if file has encryption key
                    let options = if let Some(ref key) = first_file.file.encryption_key {
                        if !key.is_empty() {
                            let key_bytes =
                                crate::encryption::DuckLakeEncryptionFactory::decode_key(key)?;
                            let decryption_props =
                                parquet::encryption::decrypt::FileDecryptionProperties::builder(
                                    key_bytes,
                                )
                                .build()
                                .map_err(|e| {
                                    DataFusionError::Execution(format!(
                                        "Failed to create decryption properties: {}",
                                        e
                                    ))
                                })?;
                            ArrowReaderOptions::new()
                                .with_file_decryption_properties(decryption_props)
                        } else {
                            ArrowReaderOptions::new()
                        }
                    } else {
                        ArrowReaderOptions::new()
                    };

                    ParquetRecordBatchStreamBuilder::new_with_options(reader, options)
                        .await
                        .map_err(|e| DataFusionError::External(Box::new(e)))?
                };

                #[cfg(not(feature = "encryption"))]
                let builder = ParquetRecordBatchStreamBuilder::new(reader)
                    .await
                    .map_err(|e| DataFusionError::External(Box::new(e)))?;

                let field_id_map = extract_parquet_field_ids(builder.metadata());

                // No field_ids means external file - use current schema directly
                if field_id_map.is_empty() {
                    return Ok((self.schema.clone(), HashMap::new()));
                }

                let (read_schema, name_mapping) =
                    build_read_schema_with_field_id_mapping(&self.columns, &field_id_map)
                        .map_err(|e| DataFusionError::External(Box::new(e)))?;

                Ok((Arc::new(read_schema), name_mapping))
            })
            .await
    }

    /// Read a delete file and extract all deleted row positions
    ///
    /// The delete file is already associated with a specific data file via metadata.
    /// We only need to extract the "pos" column - the "file_path" column is
    /// metadata/documentation only (for Iceberg compatibility).
    async fn read_delete_file_positions(
        &self,
        state: &dyn Session,
        delete_file: &DuckLakeFileData,
    ) -> DataFusionResult<HashSet<i64>> {
        // Get the standard delete file schema
        let delete_schema = delete_file_schema();

        // Resolve the delete file path
        let resolved_delete_path = self.resolve_file_path(delete_file)?;

        // Create PartitionedFile with footer size hint if available
        let mut pf = PartitionedFile::new(
            &resolved_delete_path,
            validated_file_size(delete_file.file_size_bytes, &resolved_delete_path)?,
        );
        if let Some(hint) = crate::parquet_meta::metadata_size_hint_from_footer(delete_file.footer_size) {
            pf = pf.with_metadata_size_hint(hint);
        }

        // Create file scan config for the delete file
        let file_scan_config = FileScanConfigBuilder::new(
            self.object_store_url.as_ref().clone(),
            Arc::new(self.create_parquet_source(delete_schema)),
        )
        .with_file_group(FileGroup::new(vec![pf]))
        .build();

        // Use DataSourceExec directly to preserve our ParquetSource with encryption factory
        let exec = DataSourceExec::from_data_source(file_scan_config);

        // Execute and collect all batches
        let task_ctx = state.task_ctx();
        let stream = exec.execute(0, task_ctx)?;

        let batches: Vec<RecordBatch> = stream
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<DataFusionResult<Vec<_>>>()
            .map_err(|e| {
                if is_object_store_not_found(&e) {
                    DataFusionError::Execution(format!(
                        "Delete file '{}' referenced in catalog metadata was not found. This may indicate catalog corruption or that the file was deleted outside of DuckLake.",
                        resolved_delete_path
                    ))
                } else {
                    e
                }
            })?;

        // Extract all positions from all batches
        let mut positions = HashSet::new();
        for batch in batches {
            extract_deleted_positions_from_batch(&batch, &mut positions)?;
        }

        Ok(positions)
    }

    /// Build a single execution plan for all files without delete files
    ///
    /// Groups multiple files into a single efficient execution plan since they don't
    /// need delete filtering.
    async fn build_exec_for_files_without_deletes(
        &self,
        state: &dyn Session,
        files: &[&DuckLakeTableFile],
        projection: Option<&Vec<usize>>,
        limit: Option<usize>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let (read_schema, name_mapping) = self.get_schema_mapping(state).await?;

        let partitioned_files: Vec<PartitionedFile> = files
            .iter()
            .map(|table_file| {
                let resolved_path = self.resolve_file_path(&table_file.file)?;
                let mut pf = PartitionedFile::new(
                    &resolved_path,
                    validated_file_size(table_file.file.file_size_bytes, &resolved_path)?,
                );

                // Apply footer size hint if available from DuckLake metadata
                // This reduces I/O from 2 reads to 1 read per file (especially beneficial for S3/MinIO)
                if let Some(hint) = crate::parquet_meta::metadata_size_hint_from_footer(table_file.file.footer_size) {
                    pf = pf.with_metadata_size_hint(hint);
                }

                Ok(pf)
            })
            .collect::<DataFusionResult<Vec<_>>>()?;

        // Use read_schema (with original Parquet names) for reading
        let mut builder = FileScanConfigBuilder::new(
            self.object_store_url.as_ref().clone(),
            Arc::new(self.create_parquet_source(read_schema.clone())),
        )
        .with_limit(limit)
        .with_file_group(FileGroup::new(partitioned_files));

        // Apply projection if provided
        if let Some(proj) = projection {
            builder = builder.with_projection_indices(Some(proj.clone()))?;
        }

        let file_scan_config = builder.build();
        // Use DataSourceExec directly to preserve our ParquetSource with encryption factory
        let parquet_exec: Arc<dyn ExecutionPlan> =
            DataSourceExec::from_data_source(file_scan_config);

        // Wrap with ColumnRenameExec if column names differ
        if !name_mapping.is_empty() {
            let output_schema = match projection {
                Some(indices) => Arc::new(self.schema.project(indices)?),
                None => self.schema.clone(),
            };
            Ok(Arc::new(ColumnRenameExec::new(
                parquet_exec,
                output_schema,
                name_mapping.clone(),
            )))
        } else {
            Ok(parquet_exec)
        }
    }

    /// Configure this table for write operations.
    ///
    /// This method enables write support by attaching a metadata writer and data path.
    /// Once configured, the table can handle INSERT INTO operations.
    ///
    /// # Arguments
    /// * `schema_name` - Name of the schema this table belongs to
    /// * `writer` - Metadata writer for catalog operations
    #[cfg(feature = "write")]
    pub fn with_writer(mut self, schema_name: String, writer: Arc<dyn MetadataWriter>) -> Self {
        self.schema_name = Some(schema_name);
        self.writer = Some(writer);
        self
    }

    /// Build an execution plan for a single file with delete filtering
    ///
    /// Creates a Parquet scan wrapped with a delete filter to exclude deleted rows.
    async fn build_exec_for_file_with_deletes(
        &self,
        state: &dyn Session,
        table_file: &DuckLakeTableFile,
        projection: Option<&Vec<usize>>,
        limit: Option<usize>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let (read_schema, name_mapping) = self.get_schema_mapping(state).await?;

        // Resolve the data file path for scanning
        let resolved_path = self.resolve_file_path(&table_file.file)?;

        // Create PartitionedFile with footer size hint if available
        let mut pf = PartitionedFile::new(
            &resolved_path,
            validated_file_size(table_file.file.file_size_bytes, &resolved_path)?,
        );
        if let Some(hint) = crate::parquet_meta::metadata_size_hint_from_footer(table_file.file.footer_size) {
            pf = pf.with_metadata_size_hint(hint);
        }

        // Use read_schema (with original Parquet names) for reading
        let mut builder = FileScanConfigBuilder::new(
            self.object_store_url.as_ref().clone(),
            Arc::new(self.create_parquet_source(read_schema.clone())),
        )
        .with_limit(limit)
        .with_file_group(FileGroup::new(vec![pf]));

        // Apply projection if provided
        if let Some(proj) = projection {
            builder = builder.with_projection_indices(Some(proj.clone()))?;
        }

        let file_scan_config = builder.build();
        // Use DataSourceExec directly to preserve our ParquetSource with encryption factory
        let parquet_exec: Arc<dyn ExecutionPlan> =
            DataSourceExec::from_data_source(file_scan_config);

        // Wrap with delete filter - we know there's a delete file since we partitioned
        // The metadata already tells us which delete file goes with this data file
        let exec_after_delete: Arc<dyn ExecutionPlan> =
            if let Some(ref delete_file) = table_file.delete_file {
                let deleted_positions = self.read_delete_file_positions(state, delete_file).await?;

                if !deleted_positions.is_empty() {
                    Arc::new(DeleteFilterExec::new(
                        parquet_exec,
                        table_file.file.path.clone(),
                        Arc::new(deleted_positions),
                    ))
                } else {
                    parquet_exec
                }
            } else {
                parquet_exec
            };

        // Wrap with ColumnRenameExec if column names differ
        if !name_mapping.is_empty() {
            let output_schema = match projection {
                Some(indices) => Arc::new(self.schema.project(indices)?),
                None => self.schema.clone(),
            };
            Ok(Arc::new(ColumnRenameExec::new(
                exec_after_delete,
                output_schema,
                name_mapping.clone(),
            )))
        } else {
            Ok(exec_after_delete)
        }
    }

    /// Inspect a single file's parquet metadata for the row-lineage scan
    /// path. Mirrors the per-file logic in `DuckLakeMultiFileReader::
    /// GetVirtualColumnExpression` (ducklake C++): if the file embeds a
    /// column tagged with [`ROW_ID_PARQUET_FIELD_ID`], project that column;
    /// otherwise synthesize rowid from `row_id_start + position`.
    async fn build_file_read_config(
        &self,
        state: &dyn Session,
        file: &DuckLakeFileData,
    ) -> DataFusionResult<Arc<FileReadConfig>> {
        let resolved_path = self.resolve_file_path(file)?;

        {
            let cache = self.file_read_config_cache.lock().unwrap();
            if let Some(cfg) = cache.get(&resolved_path) {
                return Ok(cfg.clone());
            }
        }

        let object_store = state
            .runtime_env()
            .object_store(self.object_store_url.as_ref())?;
        let object_path = ObjectPath::from(resolved_path.as_str());
        let reader = ParquetObjectReader::new(object_store, object_path);

        #[cfg(feature = "encryption")]
        let builder = {
            use parquet::arrow::arrow_reader::ArrowReaderOptions;
            let options = if let Some(ref key) = file.encryption_key {
                if !key.is_empty() {
                    let key_bytes = crate::encryption::DuckLakeEncryptionFactory::decode_key(key)?;
                    let decryption_props =
                        parquet::encryption::decrypt::FileDecryptionProperties::builder(key_bytes)
                            .build()
                            .map_err(|e| {
                                DataFusionError::Execution(format!(
                                    "Failed to create decryption properties: {}",
                                    e
                                ))
                            })?;
                    ArrowReaderOptions::new().with_file_decryption_properties(decryption_props)
                } else {
                    ArrowReaderOptions::new()
                }
            } else {
                ArrowReaderOptions::new()
            };
            ParquetRecordBatchStreamBuilder::new_with_options(reader, options)
                .await
                .map_err(|e| DataFusionError::External(Box::new(e)))?
        };

        #[cfg(not(feature = "encryption"))]
        let builder = ParquetRecordBatchStreamBuilder::new(reader)
            .await
            .map_err(|e| DataFusionError::External(Box::new(e)))?;

        let field_id_map = extract_parquet_field_ids(builder.metadata());

        // Standard read_schema + name_mapping for physical columns.
        let (physical_read_schema, mut name_mapping) = if field_id_map.is_empty() {
            (self.physical_schema.as_ref().clone(), HashMap::new())
        } else {
            let (s, m) = build_read_schema_with_field_id_mapping(&self.columns, &field_id_map)
                .map_err(|e| DataFusionError::External(Box::new(e)))?;
            (s, m)
        };

        // Detect the embedded rowid column by reserved field-id.
        let embedded_rowid_parquet_name = field_id_map.get(&ROW_ID_PARQUET_FIELD_ID).cloned();

        let read_schema = if let Some(ref parquet_name) = embedded_rowid_parquet_name {
            // Append the embedded rowid column to read_schema under its
            // parquet name; ParquetExec will project it by name from the
            // file. We add a `parquet_name → "rowid"` rename so the user
            // sees the column as `rowid` (only needed if the names differ).
            let mut fields: Vec<Arc<Field>> =
                physical_read_schema.fields().iter().cloned().collect();
            fields.push(Arc::new(Field::new(
                parquet_name.clone(),
                DataType::Int64,
                true,
            )));
            if parquet_name != ROWID_COLUMN_NAME {
                name_mapping.insert(parquet_name.clone(), ROWID_COLUMN_NAME.to_string());
            }
            Arc::new(Schema::new(fields))
        } else {
            Arc::new(physical_read_schema)
        };

        let cfg = Arc::new(FileReadConfig {
            read_schema,
            name_mapping,
            embedded_rowid_parquet_name,
        });

        {
            let mut cache = self.file_read_config_cache.lock().unwrap();
            cache.entry(resolved_path).or_insert_with(|| cfg.clone());
        }

        Ok(cfg)
    }

    /// Build a plan for a single file when the synthetic `rowid` column is in
    /// the projection. Always uses per-file scans because each file may have a
    /// different layout (embedded rowid vs. synthesized) and a distinct
    /// `row_id_start`.
    ///
    /// Order: ParquetExec(physical_proj [+ rowid if embedded]) →
    ///   RowIdExec(?) → DeleteFilterExec(?) → ColumnRenameExec(?).
    async fn build_exec_for_file_with_rowid(
        &self,
        state: &dyn Session,
        table_file: &DuckLakeTableFile,
        user_proj: &[usize],
        rowid_idx: usize,
        limit: Option<usize>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let file_cfg = self.build_file_read_config(state, &table_file.file).await?;
        let has_embedded = file_cfg.embedded_rowid_parquet_name.is_some();

        // Decompose user projection: which physical columns to read, and where
        // rowid should appear in the output.
        let physical_proj: Vec<usize> = user_proj
            .iter()
            .filter(|&&i| i != rowid_idx)
            .copied()
            .collect();
        let rowid_insert_pos: usize =
            user_proj
                .iter()
                .position(|&i| i == rowid_idx)
                .ok_or_else(|| {
                    DataFusionError::Internal(
                        "build_exec_for_file_with_rowid called without rowid in projection".into(),
                    )
                })?;

        // Match the C++ extension: if the file embeds no rowid column AND the
        // catalog didn't record a `row_id_start`, lineage cannot be
        // reconstructed. Hard-error rather than silently emit NULL/garbage.
        if !has_embedded && table_file.row_id_start.is_none() {
            return Err(DataFusionError::Execution(format!(
                "File \"{}\" has no embedded `_ducklake_internal_row_id` column and no \
                 `row_id_start` set in the catalog — row lineage cannot be reconstructed",
                table_file.file.path
            )));
        }

        // Build the ParquetExec projection. When the file embeds rowid we
        // splice it into the projection at `rowid_insert_pos` directly, so
        // the parquet emits batches already in the user's requested column
        // order (no extra reorder pass needed).
        let parquet_projection: Vec<usize> = if has_embedded {
            let rowid_col_in_read_schema = file_cfg.read_schema.fields().len() - 1;
            let mut p = physical_proj.clone();
            p.insert(rowid_insert_pos, rowid_col_in_read_schema);
            p
        } else {
            physical_proj.clone()
        };

        // Resolve and configure the data file
        let resolved_path = self.resolve_file_path(&table_file.file)?;
        let mut pf = PartitionedFile::new(
            &resolved_path,
            validated_file_size(table_file.file.file_size_bytes, &resolved_path)?,
        );
        if let Some(hint) = crate::parquet_meta::metadata_size_hint_from_footer(table_file.file.footer_size) {
            pf = pf.with_metadata_size_hint(hint);
        }

        let mut builder = FileScanConfigBuilder::new(
            self.object_store_url.as_ref().clone(),
            Arc::new(self.create_parquet_source(file_cfg.read_schema.clone())),
        )
        .with_limit(limit)
        .with_file_group(FileGroup::new(vec![pf]));
        builder = builder.with_projection_indices(Some(parquet_projection))?;

        let parquet_exec: Arc<dyn ExecutionPlan> =
            DataSourceExec::from_data_source(builder.build());

        // Synthesize rowid only when the file doesn't already supply it.
        let after_rowid: Arc<dyn ExecutionPlan> = if has_embedded {
            parquet_exec
        } else {
            Arc::new(RowIdExec::new_at(
                parquet_exec,
                table_file.row_id_start,
                rowid_insert_pos,
            ))
        };

        // Apply delete filter if needed. DeleteFilterExec tracks file
        // position, which is preserved through both RowIdExec and an embedded
        // rowid projection (both leave row order untouched).
        let after_deletes: Arc<dyn ExecutionPlan> =
            if let Some(ref delete_file) = table_file.delete_file {
                let deleted_positions = self.read_delete_file_positions(state, delete_file).await?;
                if !deleted_positions.is_empty() {
                    Arc::new(DeleteFilterExec::new(
                        after_rowid,
                        table_file.file.path.clone(),
                        Arc::new(deleted_positions),
                    ))
                } else {
                    after_rowid
                }
            } else {
                after_rowid
            };

        // Apply column rename. Required when a physical column was renamed in
        // the catalog, or when the embedded rowid column's parquet name
        // differs from `"rowid"` (the common case — it's
        // `_ducklake_internal_row_id`).
        if !file_cfg.name_mapping.is_empty() {
            let output_schema = self.output_schema_for_projection(user_proj, rowid_idx);
            Ok(Arc::new(ColumnRenameExec::new(
                after_deletes,
                output_schema,
                file_cfg.name_mapping.clone(),
            )))
        } else {
            Ok(after_deletes)
        }
    }

    /// Output schema for the rowid-projected per-file plan: physical fields
    /// (using their user-facing renamed names from `self.schema`) interleaved
    /// with the synthetic `rowid` field at `rowid_idx`.
    fn output_schema_for_projection(&self, user_proj: &[usize], rowid_idx: usize) -> SchemaRef {
        let mut fields: Vec<Arc<Field>> = Vec::with_capacity(user_proj.len());
        for &i in user_proj {
            if i == rowid_idx {
                fields.push(Arc::new(rowid_field()));
            } else {
                fields.push(self.schema.fields()[i].clone());
            }
        }
        Arc::new(Schema::new(fields))
    }
}

#[async_trait]
impl TableProvider for DuckLakeTable {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    fn statistics(&self) -> Option<Statistics> {
        // Aggregate per-file byte sizes from the cached `table_files`. Mirrors
        // DuckLake's own `ducklake_table_info` aggregate exactly:
        //
        //     total_byte_size == SUM(data_file.file_size_bytes)
        //                       - SUM(delete_file.file_size_bytes)
        //
        // The values come from the ducklake catalog, so this is the same
        // source of truth `ducklake_table_info` uses — no extra round trips
        // and the numbers will match byte-for-byte.
        //
        // Marked `Precision::Inexact` because DataFusion documents
        // `total_byte_size` as the *uncompressed Arrow output* size, while
        // the catalog tracks *compressed parquet* bytes. For wide
        // column types (List(Float64) embeddings) the two are nearly
        // identical; for narrow scalar schemas the on-disk number is 3-5x
        // smaller than Arrow output. Reporting compressed bytes Inexact
        // gives consumers a useful lower-bound estimate without misleading
        // the optimiser into thinking it's exact Arrow size. When
        // `record_count` is plumbed through `DuckLakeFileData`, a follow-up
        // can populate `num_rows` and use `calculate_total_byte_size` for a
        // closer Arrow-side estimate.
        let data_bytes: i64 = self
            .table_files
            .iter()
            .map(|f| f.file.file_size_bytes)
            .sum();
        let delete_bytes: i64 = self
            .table_files
            .iter()
            .filter_map(|f| f.delete_file.as_ref())
            .map(|df| df.file_size_bytes)
            .sum();
        let net_bytes = (data_bytes - delete_bytes).max(0) as usize;

        let mut stats = Statistics::new_unknown(&self.schema);
        stats.total_byte_size = Precision::Inexact(net_bytes);
        Some(stats)
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> DataFusionResult<Vec<TableProviderFilterPushDown>> {
        // Mark all filters as Inexact because we apply delete filters after the scan.
        // DataFusion will reapply these filters after DeleteFilterExec to ensure
        // correctness, but Parquet can still use them for:
        // - Row group pruning via statistics
        // - Page-level filtering with late materialization
        // - Bloom filter lookups (if available)
        Ok(filters
            .iter()
            .map(|_| TableProviderFilterPushDown::Inexact)
            .collect())
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        // Filters are received here for informational purposes. DataFusion's optimizer
        // automatically pushes them down to the Parquet scanner for row group pruning and
        // page-level filtering since we declared support via supports_filters_pushdown().
        // We mark them as Inexact, so DataFusion will reapply them after our scan.
        _filters: &[Expr],
        limit: Option<usize>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        // Row-lineage detour: when the synthetic `rowid` column is projected,
        // every file needs its own scan because each has a distinct
        // `row_id_start`. `projection == None` with row lineage on means "all
        // columns including rowid", which also routes through this path.
        let rowid_idx = self.rowid_index();
        let rowid_in_proj = match (rowid_idx, projection) {
            (Some(r), Some(p)) => p.contains(&r),
            (Some(_), None) => true,
            (None, _) => false,
        };

        if rowid_in_proj {
            let rowid_idx = rowid_idx.unwrap();
            let user_proj: Vec<usize> = projection
                .cloned()
                .unwrap_or_else(|| (0..self.schema.fields().len()).collect());

            let mut execs: Vec<Arc<dyn ExecutionPlan>> = Vec::new();
            for tf in &self.table_files {
                let exec = self
                    .build_exec_for_file_with_rowid(state, tf, &user_proj, rowid_idx, limit)
                    .await?;
                execs.push(exec);
            }

            if execs.is_empty() {
                use datafusion::physical_plan::empty::EmptyExec;
                let projected_schema = self.output_schema_for_projection(&user_proj, rowid_idx);
                return Ok(Arc::new(EmptyExec::new(projected_schema)));
            }

            return combine_execution_plans(execs);
        }

        // Fast path: rowid not projected. All projection indices refer to
        // physical columns, so the existing logic works untouched.
        let (files_with_deletes, files_without_deletes): (Vec<_>, Vec<_>) = self
            .table_files
            .iter()
            .partition(|tf| tf.delete_file.is_some());

        let mut execs: Vec<Arc<dyn ExecutionPlan>> = Vec::new();

        // Create single exec for all files without deletes (more efficient)
        if !files_without_deletes.is_empty() {
            let exec = self
                .build_exec_for_files_without_deletes(
                    state,
                    &files_without_deletes,
                    projection,
                    limit,
                )
                .await?;
            execs.push(exec);
        }

        // Only create separate execs for files with deletes
        for table_file in files_with_deletes {
            let exec = self
                .build_exec_for_file_with_deletes(state, table_file, projection, limit)
                .await?;
            execs.push(exec);
        }

        // Handle empty tables (no data files)
        if execs.is_empty() {
            use datafusion::physical_plan::empty::EmptyExec;
            let projected_schema = match projection {
                Some(indices) => Arc::new(self.schema.project(indices)?),
                None => self.schema.clone(),
            };
            return Ok(Arc::new(EmptyExec::new(projected_schema)));
        }

        // Combine execution plans
        combine_execution_plans(execs)
    }

    #[cfg(feature = "write")]
    async fn insert_into(
        &self,
        _state: &dyn Session,
        input: Arc<dyn ExecutionPlan>,
        insert_op: InsertOp,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let writer = self.writer.as_ref().ok_or_else(|| {
            DataFusionError::Plan(
                "Table is read-only. Use DuckLakeCatalog::with_writer() to enable writes."
                    .to_string(),
            )
        })?;

        let schema_name = self.schema_name.as_ref().ok_or_else(|| {
            DataFusionError::Internal("Schema name not set for writable table".to_string())
        })?;

        let write_mode = match insert_op {
            InsertOp::Append => WriteMode::Append,
            InsertOp::Overwrite | InsertOp::Replace => WriteMode::Replace,
        };

        Ok(Arc::new(DuckLakeInsertExec::new(
            input,
            Arc::clone(writer),
            schema_name.clone(),
            self.table_name.clone(),
            self.schema(),
            write_mode,
            self.object_store_url.clone(),
        )))
    }
}

/// Combines multiple execution plans into a single plan
fn combine_execution_plans(
    execs: Vec<Arc<dyn ExecutionPlan>>,
) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
    if execs.len() == 1 {
        Ok(execs.into_iter().next().unwrap())
    } else {
        use datafusion::physical_plan::union::UnionExec;
        UnionExec::try_new(execs)
    }
}

/// Extract deleted row positions from a delete file RecordBatch
///
/// Delete files have schema: (file_path: VARCHAR, pos: INT64)
/// We only extract the "pos" column - the "file_path" column is metadata/documentation
/// only (for Iceberg compatibility). The metadata catalog already tells us which delete
/// file is associated with which data file.
fn extract_deleted_positions_from_batch(
    batch: &RecordBatch,
    positions: &mut HashSet<i64>,
) -> DataFusionResult<()> {
    // Get the pos column index by name (not magic number)
    let schema = batch.schema();
    let pos_idx = schema.index_of(DELETE_POS_COL)?;

    // Get the pos column
    let pos_array = batch
        .column(pos_idx)
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| {
            DataFusionError::Internal(format!("{} column not found or wrong type", DELETE_POS_COL))
        })?;

    // Extract all non-null positions
    for i in 0..batch.num_rows() {
        if !pos_array.is_null(i) {
            positions.insert(pos_array.value(i));
        }
    }

    Ok(())
}

/// Check if a DataFusion error is caused by an object store NotFound error.
fn is_object_store_not_found(err: &DataFusionError) -> bool {
    if let DataFusionError::ObjectStore(os_err) = err {
        return matches!(&**os_err, object_store::Error::NotFound { .. });
    }
    let mut source = std::error::Error::source(err);
    while let Some(e) = source {
        if let Some(os_err) = e.downcast_ref::<object_store::Error>() {
            return matches!(os_err, object_store::Error::NotFound { .. });
        }
        source = e.source();
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validated_file_size_positive() {
        assert_eq!(validated_file_size(0, "test.parquet").unwrap(), 0);
        assert_eq!(validated_file_size(1024, "test.parquet").unwrap(), 1024);
        assert_eq!(
            validated_file_size(i64::MAX, "test.parquet").unwrap(),
            i64::MAX as u64
        );
    }

    #[test]
    fn test_validated_file_size_negative() {
        let err = validated_file_size(-1, "data/test.parquet").unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("-1"),
            "Error should contain the negative value: {}",
            msg
        );
        assert!(
            msg.contains("data/test.parquet"),
            "Error should contain the file path: {}",
            msg
        );
    }

    #[test]
    fn test_validated_file_size_large_negative() {
        let err = validated_file_size(i64::MIN, "bad.parquet").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("bad.parquet"));
        assert!(msg.contains(&i64::MIN.to_string()));
    }

    #[test]
    fn test_validated_record_count_positive() {
        assert_eq!(validated_record_count(0, "test.parquet").unwrap(), 0);
        assert_eq!(validated_record_count(100, "test.parquet").unwrap(), 100);
        assert_eq!(
            validated_record_count(i64::MAX, "test.parquet").unwrap(),
            i64::MAX as u64
        );
    }

    #[test]
    fn test_validated_record_count_negative() {
        let err = validated_record_count(-1, "data/test.parquet").unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("-1"),
            "Error should contain the negative value: {}",
            msg
        );
        assert!(
            msg.contains("data/test.parquet"),
            "Error should contain the file path: {}",
            msg
        );
        assert!(
            msg.contains("record_count"),
            "Error should mention record_count: {}",
            msg
        );
    }

    #[test]
    fn test_validated_record_count_large_negative() {
        let err = validated_record_count(i64::MIN, "bad.parquet").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("bad.parquet"));
        assert!(msg.contains(&i64::MIN.to_string()));
    }
}
