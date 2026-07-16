//! Table changes (CDC) functionality for DuckLake
//!
//! This module provides the `ducklake_table_changes()` table function that returns
//! actual row data from Parquet files with additional CDC metadata columns —
//! inserts, deletes (with the deleted rows' old values), and UPDATEs correlated
//! into `update_preimage`/`update_postimage` pairs, matching official DuckLake.
//!
//! Note: Ordering across files is undefined unless explicitly requested via ORDER BY.

use std::collections::HashSet;
use std::fmt;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow::array::{Array, ArrayRef, BooleanArray, Int64Array, StringArray, UInt32Array};
use arrow::compute::take;
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use async_trait::async_trait;
use datafusion::catalog::Session;
use datafusion::common::Result as DataFusionResult;
use datafusion::datasource::listing::PartitionedFile;
use datafusion::datasource::physical_plan::{FileGroup, FileScanConfigBuilder, ParquetSource};
use datafusion::datasource::source::DataSourceExec;
use datafusion::datasource::{TableProvider, TableType};
use datafusion::error::DataFusionError;
use datafusion::execution::object_store::ObjectStoreUrl;
use datafusion::execution::{RecordBatchStream, SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_expr::{EquivalenceProperties, PhysicalExpr};
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::physical_plan::union::UnionExec;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, PlanProperties,
};
use futures::{Stream, StreamExt};
use object_store::path::Path as ObjectPath;
use parquet::arrow::ParquetRecordBatchStreamBuilder;
use parquet::arrow::async_reader::ParquetObjectReader;

use crate::metadata_provider::{DataFileChange, MetadataProvider};
use crate::path_resolver::resolve_path;
use crate::positional_source::PositionalFileSource;
use crate::row_id::{FileRowNumberExec, ROW_ID_PARQUET_FIELD_ID, ROW_POS_COLUMN_NAME};
use crate::table::{delete_file_schema, validated_file_size, validated_record_count};
use crate::types::extract_parquet_field_ids;

#[cfg(feature = "encryption")]
use crate::encryption::EncryptionFactoryBuilder;
#[cfg(feature = "encryption")]
use datafusion::execution::parquet_encryption::EncryptionFactory;

/// Type of change captured in CDC output.
///
/// [`UpdatePreimage`](ChangeType::UpdatePreimage) /
/// [`UpdatePostimage`](ChangeType::UpdatePostimage) are the paired old/new row
/// versions of an `UPDATE`: `ducklake_table_changes` correlates a same-snapshot
/// delete + insert that share a rowid into this pair (the DuckLake spirit of an
/// update in a change feed) instead of surfacing them as an unrelated delete and
/// insert. The `as_str` values match the DuckLake change-feed spec strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChangeType {
    Insert,
    Delete,
    /// The old version of a row that an `UPDATE` rewrote in this snapshot.
    UpdatePreimage,
    /// The new version of a row that an `UPDATE` rewrote in this snapshot.
    UpdatePostimage,
}

impl ChangeType {
    /// Returns the string representation for Arrow output
    fn as_str(&self) -> &'static str {
        match self {
            ChangeType::Insert => "insert",
            ChangeType::Delete => "delete",
            ChangeType::UpdatePreimage => "update_preimage",
            ChangeType::UpdatePostimage => "update_postimage",
        }
    }
}

impl fmt::Display for ChangeType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Positions of the CDC metadata columns in the feed's output schema. They
/// LEAD the table columns, matching official DuckLake's `ducklake_table_changes`
/// projection (`SELECT snapshot_id, rowid, change_type, ...`).
const SNAPSHOT_ID_IDX: usize = 0;
const ROWID_IDX: usize = 1;
const CHANGE_TYPE_IDX: usize = 2;
/// Number of CDC metadata columns preceding the table columns.
const CDC_COLS: usize = 3;

/// Custom execution plan that prepends CDC columns (snapshot_id, rowid, change_type) to each batch
///
/// This plan wraps a ParquetExec and appends CDC metadata columns to each output batch.
/// It supports projection pushdown by:
/// - Reading only requested table columns from Parquet
/// - Including only requested CDC columns in output
/// - Optionally skipping input columns entirely when only CDC columns are needed
#[derive(Debug)]
pub struct PrependCDCColumnsExec {
    /// The input execution plan (typically ParquetExec)
    input: Arc<dyn ExecutionPlan>,
    /// Snapshot ID for this file
    snapshot_id: i64,
    /// Change type for this file
    change_type: ChangeType,
    /// Whether to include a rowid column in output. On this insert-only path
    /// rowid cannot be synthesized, so it is emitted as an all-NULL column
    /// (used only for encrypted tables, where the correlated path can't run).
    include_rowid: bool,
    /// Whether to include snapshot_id in output
    include_snapshot_id: bool,
    /// Whether to include change_type in output
    include_change_type: bool,
    /// If true, input columns are dummy (for row count only) and should not be included
    skip_input_columns: bool,
    /// Output schema (projected input schema + requested CDC columns)
    output_schema: SchemaRef,
    /// Cached plan properties with updated schema
    properties: Arc<PlanProperties>,
}

impl PrependCDCColumnsExec {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input: Arc<dyn ExecutionPlan>,
        snapshot_id: i64,
        change_type: ChangeType,
        include_rowid: bool,
        include_snapshot_id: bool,
        include_change_type: bool,
        skip_input_columns: bool,
        output_schema: SchemaRef,
    ) -> Self {
        // Create new equivalence properties with the output schema.
        // We preserve partitioning and execution semantics from input.
        // Note: This resets equivalences which is pessimistic but correct.
        // Future optimization: carry forward equivalences for projected table columns.
        let eq_properties = EquivalenceProperties::new(output_schema.clone());

        let properties = Arc::new(PlanProperties::new(
            eq_properties,
            input.output_partitioning().clone(),
            input.pipeline_behavior(),
            input.boundedness(),
        ));

        Self {
            input,
            snapshot_id,
            change_type,
            include_rowid,
            include_snapshot_id,
            include_change_type,
            skip_input_columns,
            output_schema,
            properties,
        }
    }
}

impl DisplayAs for PrependCDCColumnsExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default
            | DisplayFormatType::Verbose
            | DisplayFormatType::TreeRender => {
                write!(
                    f,
                    "PrependCDCColumnsExec: snapshot_id={}, change_type={}, \
                     include_snapshot={}, include_change={}, skip_input={}",
                    self.snapshot_id,
                    self.change_type,
                    self.include_snapshot_id,
                    self.include_change_type,
                    self.skip_input_columns
                )
            },
        }
    }
}

impl ExecutionPlan for PrependCDCColumnsExec {
    fn name(&self) -> &str {
        "PrependCDCColumnsExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(DataFusionError::Internal(
                "PrependCDCColumnsExec expects exactly one child".into(),
            ));
        }

        Ok(Arc::new(PrependCDCColumnsExec::new(
            children[0].clone(),
            self.snapshot_id,
            self.change_type,
            self.include_rowid,
            self.include_snapshot_id,
            self.include_change_type,
            self.skip_input_columns,
            self.output_schema.clone(),
        )))
    }

    fn schema(&self) -> SchemaRef {
        self.output_schema.clone()
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        let input_stream = self.input.execute(partition, context)?;

        Ok(Box::pin(PrependCDCColumnsStream {
            input: input_stream,
            snapshot_id: self.snapshot_id,
            change_type: self.change_type,
            include_rowid: self.include_rowid,
            include_snapshot_id: self.include_snapshot_id,
            include_change_type: self.include_change_type,
            skip_input_columns: self.skip_input_columns,
            output_schema: self.output_schema.clone(),
        }))
    }
}

/// Stream that appends CDC columns to input batches
struct PrependCDCColumnsStream {
    input: SendableRecordBatchStream,
    snapshot_id: i64,
    change_type: ChangeType,
    include_rowid: bool,
    include_snapshot_id: bool,
    include_change_type: bool,
    skip_input_columns: bool,
    output_schema: SchemaRef,
}

impl Stream for PrependCDCColumnsStream {
    type Item = DataFusionResult<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match Pin::new(&mut self.input).poll_next(cx) {
            Poll::Ready(Some(Ok(batch))) => {
                let result = self.transform_batch(&batch);
                Poll::Ready(Some(result))
            },
            Poll::Ready(Some(Err(e))) => Poll::Ready(Some(Err(e))),
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

impl PrependCDCColumnsStream {
    fn transform_batch(&self, batch: &RecordBatch) -> DataFusionResult<RecordBatch> {
        let num_rows = batch.num_rows();
        let mut columns: Vec<ArrayRef> = Vec::new();

        // Prepend requested CDC columns, in the order snapshot_id, rowid,
        // change_type (official DuckLake order). rowid is all-NULL here: this
        // insert-only path can't synthesize it (used for encrypted tables).
        if self.include_snapshot_id {
            columns.push(Arc::new(Int64Array::from(vec![self.snapshot_id; num_rows])));
        }
        if self.include_rowid {
            columns.push(Arc::new(Int64Array::from(vec![None::<i64>; num_rows])));
        }
        if self.include_change_type {
            columns.push(Arc::new(StringArray::from(vec![
                self.change_type.as_str();
                num_rows
            ])));
        }

        // Then the input columns, unless we're skipping them
        if !self.skip_input_columns {
            columns.extend(batch.columns().iter().cloned());
        }

        RecordBatch::try_new(self.output_schema.clone(), columns)
            .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))
    }
}

impl RecordBatchStream for PrependCDCColumnsStream {
    fn schema(&self) -> SchemaRef {
        self.output_schema.clone()
    }
}

/// Projection analysis result: maps logical projection to physical components
struct ProjectionInfo {
    /// Table column indices to read from Parquet (in original order)
    table_indices: Vec<usize>,
    /// Whether rowid is requested
    need_rowid: bool,
    /// Whether snapshot_id is requested
    need_snapshot_id: bool,
    /// Whether change_type is requested
    need_change_type: bool,
    /// The projected output schema
    output_schema: SchemaRef,
}

#[derive(Debug)]
pub struct TableChangesTable {
    provider: Arc<dyn MetadataProvider>,
    table_id: i64,
    start_snapshot: i64,
    end_snapshot: i64,
    /// Object store URL for resolving file paths
    object_store_url: Arc<ObjectStoreUrl>,
    /// Table path for resolving relative file paths
    table_path: String,
    /// Original table schema (without CDC columns)
    table_schema: SchemaRef,
    /// Combined schema: table columns + snapshot_id + change_type
    output_schema: SchemaRef,
}

impl TableChangesTable {
    pub fn new(
        provider: Arc<dyn MetadataProvider>,
        table_id: i64,
        start_snapshot: i64,
        end_snapshot: i64,
        object_store_url: Arc<ObjectStoreUrl>,
        table_path: String,
        table_schema: SchemaRef,
    ) -> Self {
        // Build output schema: CDC metadata columns leading — (snapshot_id,
        // rowid, change_type), official DuckLake's ducklake_table_changes
        // column order — then the table columns.
        let mut fields: Vec<Field> = Vec::with_capacity(table_schema.fields().len() + CDC_COLS);
        fields.push(Field::new("snapshot_id", DataType::Int64, false));
        // rowid is nullable: it is NULL on encrypted (PME) tables, where the
        // correlated feed cannot decrypt footers to resolve rowids.
        fields.push(Field::new("rowid", DataType::Int64, true));
        fields.push(Field::new("change_type", DataType::Utf8, false));
        fields.extend(table_schema.fields().iter().map(|f| f.as_ref().clone()));
        let output_schema = Arc::new(Schema::new(fields));

        Self {
            provider,
            table_id,
            start_snapshot,
            end_snapshot,
            object_store_url,
            table_path,
            table_schema,
            output_schema,
        }
    }

    /// Analyze projection and split into table columns and CDC columns.
    /// CDC columns lead the table columns in the order `snapshot_id`, `rowid`,
    /// `change_type` (official DuckLake order).
    fn analyze_projection(&self, projection: Option<&Vec<usize>>) -> ProjectionInfo {
        let num_table_cols = self.table_schema.fields().len();

        match projection {
            None => {
                // No projection - read all columns
                ProjectionInfo {
                    table_indices: (0..num_table_cols).collect(),
                    need_rowid: true,
                    need_snapshot_id: true,
                    need_change_type: true,
                    output_schema: self.output_schema.clone(),
                }
            },
            Some(indices) => {
                // Split indices into table columns and CDC columns
                let mut table_indices: Vec<usize> = Vec::new();
                let mut need_rowid = false;
                let mut need_snapshot_id = false;
                let mut need_change_type = false;

                for &idx in indices {
                    match idx {
                        SNAPSHOT_ID_IDX => need_snapshot_id = true,
                        ROWID_IDX => need_rowid = true,
                        CHANGE_TYPE_IDX => need_change_type = true,
                        _ if idx < num_table_cols + CDC_COLS => {
                            table_indices.push(idx - CDC_COLS);
                        },
                        _ => {},
                    }
                }

                // Build projected output schema in the order requested
                let mut fields: Vec<Field> = Vec::with_capacity(indices.len());
                for &idx in indices {
                    fields.push(self.output_schema.field(idx).clone());
                }
                let output_schema = Arc::new(Schema::new(fields));

                ProjectionInfo {
                    table_indices,
                    need_rowid,
                    need_snapshot_id,
                    need_change_type,
                    output_schema,
                }
            },
        }
    }

    /// Build the schema that PrependCDCColumnsExec will output. On this
    /// (encryption-aware, insert-only) path rowid cannot be synthesized, so when
    /// requested it is emitted as a nullable, all-NULL column.
    fn build_cdc_exec_schema(
        &self,
        table_indices: &[usize],
        need_rowid: bool,
        need_snapshot_id: bool,
        need_change_type: bool,
    ) -> SchemaRef {
        let mut fields: Vec<Field> = Vec::with_capacity(table_indices.len() + CDC_COLS);

        if need_snapshot_id {
            fields.push(Field::new("snapshot_id", DataType::Int64, false));
        }
        if need_rowid {
            fields.push(Field::new("rowid", DataType::Int64, true));
        }
        if need_change_type {
            fields.push(Field::new("change_type", DataType::Utf8, false));
        }
        fields.extend(
            table_indices
                .iter()
                .map(|&i| self.table_schema.field(i).clone()),
        );

        Arc::new(Schema::new(fields))
    }

    /// Build a ParquetExec wrapped with PrependCDCColumnsExec for a single file
    #[cfg(feature = "encryption")]
    async fn build_exec_for_file(
        &self,
        state: &dyn Session,
        data_file: &DataFileChange,
        proj_info: &ProjectionInfo,
        encryption_factory: &Option<Arc<dyn EncryptionFactory>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let parquet_source = if let Some(factory) = encryption_factory {
            ParquetSource::new(self.table_schema.clone())
                .with_encryption_factory(Arc::clone(factory))
        } else {
            ParquetSource::new(self.table_schema.clone())
        };
        self.build_exec_for_file_impl(state, data_file, proj_info, parquet_source)
            .await
    }

    /// Build a ParquetExec wrapped with PrependCDCColumnsExec for a single file
    #[cfg(not(feature = "encryption"))]
    async fn build_exec_for_file(
        &self,
        state: &dyn Session,
        data_file: &DataFileChange,
        proj_info: &ProjectionInfo,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        self.build_exec_for_file_impl(
            state,
            data_file,
            proj_info,
            ParquetSource::new(self.table_schema.clone()),
        )
        .await
    }

    /// Internal implementation for building a ParquetExec wrapped with PrependCDCColumnsExec
    async fn build_exec_for_file_impl(
        &self,
        _state: &dyn Session,
        data_file: &DataFileChange,
        proj_info: &ProjectionInfo,
        parquet_source: ParquetSource,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        // Resolve file path
        let resolved_path = resolve_path(
            &self.table_path,
            &data_file.path,
            data_file.path_is_relative,
        )
        .map_err(|e| DataFusionError::External(Box::new(e)))?;

        // Create PartitionedFile with footer size hint if available
        let mut pf = PartitionedFile::new(
            &resolved_path,
            validated_file_size(data_file.file_size_bytes, &resolved_path)?,
        );
        if let Some(footer_size) = data_file.footer_size
            && footer_size > 0
            && let Ok(hint) = usize::try_from(footer_size)
        {
            pf = pf.with_metadata_size_hint(hint);
        }

        // Determine what to read from Parquet
        let parquet_projection = if proj_info.table_indices.is_empty() {
            // Only CDC columns requested - read minimal data for row counts
            Some(vec![0])
        } else {
            Some(proj_info.table_indices.clone())
        };

        // Create file scan config with projection pushdown
        let mut builder = FileScanConfigBuilder::new(
            self.object_store_url.as_ref().clone(),
            Arc::new(parquet_source),
        )
        .with_file_group(FileGroup::new(vec![pf]));

        if let Some(proj) = parquet_projection {
            builder = builder.with_projection_indices(Some(proj))?;
        }

        let file_scan_config = builder.build();

        // Use DataSourceExec directly to preserve our ParquetSource with encryption factory
        let parquet_exec: Arc<dyn ExecutionPlan> =
            DataSourceExec::from_data_source(file_scan_config);

        // Determine if we should skip input columns (only CDC columns requested)
        let skip_input_columns = proj_info.table_indices.is_empty();

        // Build output schema for PrependCDCColumnsExec
        let cdc_exec_schema = if skip_input_columns {
            // Only CDC columns - build schema with just those
            let mut fields = Vec::new();
            if proj_info.need_snapshot_id {
                fields.push(Field::new("snapshot_id", DataType::Int64, false));
            }
            if proj_info.need_rowid {
                fields.push(Field::new("rowid", DataType::Int64, true));
            }
            if proj_info.need_change_type {
                fields.push(Field::new("change_type", DataType::Utf8, false));
            }
            Arc::new(Schema::new(fields))
        } else {
            self.build_cdc_exec_schema(
                &proj_info.table_indices,
                proj_info.need_rowid,
                proj_info.need_snapshot_id,
                proj_info.need_change_type,
            )
        };

        Ok(Arc::new(PrependCDCColumnsExec::new(
            parquet_exec,
            data_file.begin_snapshot,
            ChangeType::Insert,
            proj_info.need_rowid,
            proj_info.need_snapshot_id,
            proj_info.need_change_type,
            skip_input_columns,
            cdc_exec_schema,
        )))
    }

    /// Read a file's parquet footer and return the physical name of its embedded
    /// row-id column ([`ROW_ID_PARQUET_FIELD_ID`]) when present. A file that
    /// carries one is the postimage side of an `UPDATE` / compaction.
    async fn detect_embedded_rowid_name(
        &self,
        state: &dyn Session,
        path: &str,
        is_relative: bool,
    ) -> DataFusionResult<Option<String>> {
        let resolved = resolve_path(&self.table_path, path, is_relative)
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        let object_store = state
            .runtime_env()
            .object_store(self.object_store_url.as_ref())?;
        let reader = ParquetObjectReader::new(object_store, ObjectPath::from(resolved.as_str()));
        let builder = ParquetRecordBatchStreamBuilder::new(reader)
            .await
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        let field_ids = extract_parquet_field_ids(builder.metadata());
        Ok(field_ids.get(&ROW_ID_PARQUET_FIELD_ID).cloned())
    }

    /// The read schema for a data file: the table columns, plus the embedded
    /// rowid column (under its physical name) when `embedded_name` is `Some`.
    fn read_schema_with_optional_rowid(&self, embedded_name: &Option<String>) -> SchemaRef {
        match embedded_name {
            Some(name) => {
                let mut fields: Vec<Field> = self
                    .table_schema
                    .fields()
                    .iter()
                    .map(|f| f.as_ref().clone())
                    .collect();
                fields.push(Field::new(name, DataType::Int64, true));
                Arc::new(Schema::new(fields))
            },
            None => self.table_schema.clone(),
        }
    }

    /// Scan of an inserted data file for the correlated feed. A file with an
    /// embedded rowid (an UPDATE / compaction postimage) is scanned plainly — its
    /// rowid IS the embedded column. A plain insert is scanned positionally
    /// (`PositionalFileSource` + `FileRowNumberExec`) so its rowid can be
    /// synthesized as `row_id_start + position` — but only when `need_rowid`;
    /// otherwise it is a plain scan with no position column.
    fn build_insert_scan(
        &self,
        data_file: &DataFileChange,
        embedded_name: &Option<String>,
        need_rowid: bool,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let resolved = resolve_path(
            &self.table_path,
            &data_file.path,
            data_file.path_is_relative,
        )
        .map_err(|e| DataFusionError::External(Box::new(e)))?;
        let mut pf = PartitionedFile::new(
            &resolved,
            validated_file_size(data_file.file_size_bytes, &resolved)?,
        );
        if let Some(footer) = data_file.footer_size
            && footer > 0
            && let Ok(hint) = usize::try_from(footer)
        {
            pf = pf.with_metadata_size_hint(hint);
        }
        let read_schema = self.read_schema_with_optional_rowid(embedded_name);
        let plain_scan = |pf: PartitionedFile, schema: SchemaRef| {
            let builder = FileScanConfigBuilder::new(
                self.object_store_url.as_ref().clone(),
                Arc::new(ParquetSource::new(schema)),
            )
            .with_file_group(FileGroup::new(vec![pf]));
            DataSourceExec::from_data_source(builder.build())
        };
        match embedded_name {
            // Postimage / rewrite: rowid IS the embedded column — a plain scan.
            Some(_) => Ok(plain_scan(pf, read_schema)),
            // Plain insert, rowid requested: scan positionally to synthesize
            // rowid = row_id_start + physical position.
            None if need_rowid => {
                let source = PositionalFileSource::wrap(Arc::new(ParquetSource::new(read_schema)));
                let builder =
                    FileScanConfigBuilder::new(self.object_store_url.as_ref().clone(), source)
                        .with_file_group(FileGroup::new(vec![pf]))
                        .with_partitioned_by_file_group(true);
                let scan = DataSourceExec::from_data_source(builder.build());
                Ok(Arc::new(FileRowNumberExec::new(scan, vec![0])))
            },
            // Plain insert, rowid not requested: a plain scan, no positions.
            None => Ok(plain_scan(pf, read_schema)),
        }
    }

    /// Positional scan of a delete's source data file: table columns, the
    /// embedded rowid column when present, and the internal physical-position
    /// column. `PositionalFileSource` + `FileRowNumberExec` guarantee true
    /// physical positions so deleted rows can be matched to the delete file's
    /// `pos` set regardless of scan partitioning.
    fn build_delete_data_scan(
        &self,
        resolved_path: &str,
        size_bytes: i64,
        footer_size: i64,
        embedded_name: &Option<String>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let mut pf = PartitionedFile::new(
            resolved_path,
            validated_file_size(size_bytes, resolved_path)?,
        );
        if footer_size > 0
            && let Ok(hint) = usize::try_from(footer_size)
        {
            pf = pf.with_metadata_size_hint(hint);
        }
        let read_schema = self.read_schema_with_optional_rowid(embedded_name);
        let source = PositionalFileSource::wrap(Arc::new(ParquetSource::new(read_schema)));
        let builder = FileScanConfigBuilder::new(self.object_store_url.as_ref().clone(), source)
            .with_file_group(FileGroup::new(vec![pf]))
            .with_partitioned_by_file_group(true);
        let scan = DataSourceExec::from_data_source(builder.build());
        Ok(Arc::new(FileRowNumberExec::new(scan, vec![0])))
    }

    /// Scan of a positional delete file (the standard `(file_path, pos)` schema);
    /// the correlation path reads its `pos` column to find newly-deleted rows.
    fn build_delete_file_scan(
        &self,
        path: &str,
        is_relative: bool,
        size_bytes: i64,
        footer_size: i64,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let resolved = resolve_path(&self.table_path, path, is_relative)
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        let mut pf = PartitionedFile::new(&resolved, validated_file_size(size_bytes, &resolved)?);
        if footer_size > 0
            && let Ok(hint) = usize::try_from(footer_size)
        {
            pf = pf.with_metadata_size_hint(hint);
        }
        let builder = FileScanConfigBuilder::new(
            self.object_store_url.as_ref().clone(),
            Arc::new(ParquetSource::new(delete_file_schema())),
        )
        .with_file_group(FileGroup::new(vec![pf]));
        Ok(DataSourceExec::from_data_source(builder.build()))
    }

    /// Build the correlated change feed: pair a same-snapshot delete + insert
    /// that share a rowid into `update_preimage` (old) + `update_postimage`
    /// (new); surface unmatched inserts as `insert` and unmatched deletes as
    /// `delete` (carrying the deleted rows' old values), matching official
    /// DuckLake's `ducklake_table_changes`.
    async fn build_correlated_changes(
        &self,
        state: &dyn Session,
        data_files: &[DataFileChange],
        delete_files: &[crate::metadata_provider::DeleteFileChange],
        embedded_names: &[Option<String>],
        projection: Option<&Vec<usize>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let table_len = self.table_schema.fields().len();
        // Whether the caller wants the rowid column (at ROWID_IDX among the
        // leading CDC columns). When it does not, plain inserts skip the
        // positional scan and rowid synthesis entirely.
        let need_rowid = projection.is_none_or(|idx| idx.contains(&ROWID_IDX));

        let mut insert_units = Vec::with_capacity(data_files.len());
        for (df, name) in data_files.iter().zip(embedded_names.iter()) {
            // Embedded file: rowid is the trailing embedded column (at `table_len`).
            // Plain insert (rowid requested): `build_insert_scan` appends a
            // physical-position column (at `table_len`); rowid = row_id_start + pos.
            let embedded = name.is_some();
            insert_units.push(InsertUnit {
                snapshot_id: df.begin_snapshot,
                scan: self.build_insert_scan(df, name, need_rowid)?,
                embedded_col_idx: name.as_ref().map(|_| table_len),
                pos_col_idx: if !embedded && need_rowid {
                    Some(table_len)
                } else {
                    None
                },
                row_id_start: df.row_id_start,
            });
        }

        // Every delete in range is read: unmatched ones surface as `delete`
        // rows, and those sharing a (snapshot_id, rowid) with an embedded-rowid
        // insert pair into update preimages.
        let delete_units = {
            let mut delete_units = Vec::with_capacity(delete_files.len());
            for dfc in delete_files {
                validated_record_count(dfc.data_record_count, &dfc.data_file_path)?;
                let resolved = resolve_path(
                    &self.table_path,
                    &dfc.data_file_path,
                    dfc.data_file_path_is_relative,
                )
                .map_err(|e| DataFusionError::External(Box::new(e)))?;
                let old_embedded = self
                    .detect_embedded_rowid_name(
                        state,
                        &dfc.data_file_path,
                        dfc.data_file_path_is_relative,
                    )
                    .await?;
                let data_scan = self.build_delete_data_scan(
                    &resolved,
                    dfc.data_file_size_bytes,
                    dfc.data_file_footer_size.unwrap_or(0),
                    &old_embedded,
                )?;
                let current_delete_scan = match &dfc.current_delete_path {
                    Some(p) => Some(self.build_delete_file_scan(
                        p,
                        dfc.current_delete_path_is_relative.unwrap_or(true),
                        dfc.current_delete_file_size_bytes.unwrap_or(0),
                        dfc.current_delete_footer_size.unwrap_or(0),
                    )?),
                    None => None,
                };
                let previous_delete_scan = match &dfc.previous_delete_path {
                    Some(p) => Some(self.build_delete_file_scan(
                        p,
                        dfc.previous_delete_path_is_relative.unwrap_or(true),
                        dfc.previous_delete_file_size_bytes.unwrap_or(0),
                        dfc.previous_delete_footer_size.unwrap_or(0),
                    )?),
                    None => None,
                };
                delete_units.push(DeleteUnit {
                    snapshot_id: dfc.snapshot_id,
                    data_scan,
                    embedded_col_idx: old_embedded.as_ref().map(|_| table_len),
                    current_delete_scan,
                    previous_delete_scan,
                    record_count: dfc.data_record_count,
                    row_id_start: dfc.data_row_id_start,
                });
            }
            delete_units
        };

        let full: Arc<dyn ExecutionPlan> = Arc::new(TableChangesExec::new(
            insert_units,
            delete_units,
            self.table_schema.clone(),
            self.output_schema.clone(),
            table_len,
            need_rowid,
        ));

        // The exec emits the full `[snapshot_id, rowid, change_type, table
        // columns]` schema; honor the requested projection with a
        // ProjectionExec on top.
        match projection {
            None => Ok(full),
            Some(indices) => {
                let exprs: Vec<(Arc<dyn PhysicalExpr>, String)> = indices
                    .iter()
                    .map(|&i| {
                        let f = self.output_schema.field(i);
                        (
                            Arc::new(Column::new(f.name(), i)) as Arc<dyn PhysicalExpr>,
                            f.name().to_string(),
                        )
                    })
                    .collect();
                Ok(Arc::new(ProjectionExec::try_new(exprs, full)?))
            },
        }
    }
}

#[async_trait]
impl TableProvider for TableChangesTable {
    fn schema(&self) -> SchemaRef {
        self.output_schema.clone()
    }

    fn table_type(&self) -> TableType {
        TableType::View
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        _filters: &[datafusion::prelude::Expr],
        _limit: Option<usize>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        // Analyze projection to determine what to read
        let proj_info = self.analyze_projection(projection);

        // Get data files added between snapshots (INSERT changes)
        let data_files = self
            .provider
            .get_data_files_added_between_snapshots(
                self.table_id,
                self.start_snapshot,
                self.end_snapshot,
            )
            .map_err(|e| DataFusionError::External(Box::new(e)))?;

        // Deletes applied in the window surface as `delete` rows (and pair into
        // update preimages), so they participate in both the empty check and
        // the path decision — a delete-only window is NOT empty.
        let delete_files = self
            .provider
            .get_delete_files_added_between_snapshots(
                self.table_id,
                self.start_snapshot,
                self.end_snapshot,
            )
            .map_err(|e| DataFusionError::External(Box::new(e)))?;

        // Handle empty case
        if data_files.is_empty() && delete_files.is_empty() {
            use datafusion::physical_plan::empty::EmptyExec;
            return Ok(Arc::new(EmptyExec::new(proj_info.output_schema)));
        }

        // Decide whether to take the correlated path (reading delete sources to
        // emit `delete` rows and pair an UPDATE's delete+insert into
        // preimage/postimage). Guards, all cheap and metadata-only:
        //
        //  1. Deletes-present OR rowid-requested: with neither, the window is
        //     plain inserts and needs no correlation — a plain-INSERT catalog
        //     does ZERO per-file parquet footer reads at plan time.
        //  2. Not encrypted: the correlated path reads parquet footers (to detect
        //     the embedded-rowid postimage) and the source rows of deletes, none
        //     of which it can decrypt (the delete-side change record carries no
        //     key). On a PME catalog we therefore stay on the insert-only path
        //     below — which IS encryption-aware — so CDC over inserts never
        //     fails; the tradeoff is that UPDATEs surface as plain inserts and
        //     pure deletes are missing there. See COMPATIBILITY.md. (A
        //     delete-only window carries no data file to detect encryption
        //     from, so on an encrypted catalog it fails at read rather than
        //     returning wrong results.)
        let any_encrypted = {
            #[cfg(feature = "encryption")]
            {
                data_files.iter().any(|d| d.encryption_key.is_some())
            }
            #[cfg(not(feature = "encryption"))]
            {
                false
            }
        };

        if (proj_info.need_rowid || !delete_files.is_empty()) && !any_encrypted {
            let mut embedded_names: Vec<Option<String>> = Vec::with_capacity(data_files.len());
            for data_file in &data_files {
                embedded_names.push(
                    self.detect_embedded_rowid_name(
                        state,
                        &data_file.path,
                        data_file.path_is_relative,
                    )
                    .await?,
                );
            }
            return self
                .build_correlated_changes(
                    state,
                    &data_files,
                    &delete_files,
                    &embedded_names,
                    projection,
                )
                .await;
        }

        // Build encryption factory from file encryption keys (when encryption feature is enabled)
        #[cfg(feature = "encryption")]
        let encryption_factory: Option<Arc<dyn EncryptionFactory>> = {
            let mut builder = EncryptionFactoryBuilder::new();
            for data_file in &data_files {
                let resolved_path = resolve_path(
                    &self.table_path,
                    &data_file.path,
                    data_file.path_is_relative,
                )
                .map_err(|e| DataFusionError::External(Box::new(e)))?;
                builder.add_file(&resolved_path, data_file.encryption_key.as_deref());
            }
            let factory = builder.build();
            if factory.has_encrypted_files() {
                Some(Arc::new(factory) as Arc<dyn EncryptionFactory>)
            } else {
                None
            }
        };

        // Build execution plan for each file with projection pushdown
        let mut execs: Vec<Arc<dyn ExecutionPlan>> = Vec::with_capacity(data_files.len());
        for data_file in &data_files {
            #[cfg(feature = "encryption")]
            let exec = self
                .build_exec_for_file(state, data_file, &proj_info, &encryption_factory)
                .await?;
            #[cfg(not(feature = "encryption"))]
            let exec = self
                .build_exec_for_file(state, data_file, &proj_info)
                .await?;
            execs.push(exec);
        }

        // Combine with UnionExec if multiple files
        if execs.len() == 1 {
            Ok(execs.into_iter().next().unwrap())
        } else {
            UnionExec::try_new(execs)
        }
    }
}

// ---------------------------------------------------------------------------
// Correlated change feed (insert / delete / update_preimage / update_postimage)
// ---------------------------------------------------------------------------

/// One inserted data file added in the snapshot range, with enough context to
/// derive each row's rowid. When `embedded_col_idx` is `Some`, the scan's column
/// at that index is the file's embedded rowid (an UPDATE / compaction postimage);
/// otherwise the file is a plain INSERT whose scan carries a physical-position
/// column at `pos_col_idx` and whose rowid is `row_id_start + position`.
#[derive(Clone)]
struct InsertUnit {
    snapshot_id: i64,
    scan: Arc<dyn ExecutionPlan>,
    /// Column index of the embedded rowid, or `None` for a plain insert.
    embedded_col_idx: Option<usize>,
    /// Column index of the physical row-position (plain inserts only).
    pos_col_idx: Option<usize>,
    /// First rowid of the file (`None` if the catalog carries none). Required
    /// only for a plain insert, whose rowid is `row_id_start + pos`.
    row_id_start: Option<i64>,
}

/// One delete applied in the snapshot range: enough to read the newly-deleted
/// rows of the source data file (the delete positions minus the previous
/// generation's) together with each row's rowid.
#[derive(Clone)]
struct DeleteUnit {
    snapshot_id: i64,
    /// Positional scan of the source data file: `[table columns..., (embedded
    /// rowid), __ducklake_row_pos]`.
    data_scan: Arc<dyn ExecutionPlan>,
    /// Column index of the source file's embedded rowid, or `None` (rowids are
    /// then `row_id_start + position`).
    embedded_col_idx: Option<usize>,
    /// Scan of the current delete file, or `None` for a full-file delete.
    current_delete_scan: Option<Arc<dyn ExecutionPlan>>,
    /// Scan of the delete file this one superseded, if any.
    previous_delete_scan: Option<Arc<dyn ExecutionPlan>>,
    record_count: i64,
    /// First rowid of the source file (`None` if the catalog carries none).
    /// Required only when a deleted row has no embedded rowid.
    row_id_start: Option<i64>,
}

/// Rows carrying their `(snapshot_id, rowid)` correlation key alongside the
/// table columns, ready to be tagged once update pairs are known.
struct KeyedRows {
    snapshot_id: i64,
    table_batch: RecordBatch,
    rowid: Int64Array,
}

/// Execution plan for the correlated `ducklake_table_changes` feed. Collects the
/// inserted rows (with embedded rowids) and the newly-deleted rows (with
/// synthesized/embedded rowids), pairs those sharing a `(snapshot_id, rowid)`
/// into preimage/postimage, and emits the tagged rows. Single output partition.
#[derive(Debug)]
pub struct TableChangesExec {
    insert_units: Vec<InsertUnit>,
    delete_units: Vec<DeleteUnit>,
    #[allow(dead_code)]
    table_schema: SchemaRef,
    output_schema: SchemaRef,
    table_len: usize,
    /// Whether the rowid column is actually requested. When false, plain inserts
    /// skip rowid synthesis (emitting a placeholder dropped by the projection),
    /// so a non-rowid projection never needs a plain insert's row_id_start.
    need_rowid: bool,
    properties: Arc<PlanProperties>,
}

impl std::fmt::Debug for InsertUnit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InsertUnit")
            .field("snapshot_id", &self.snapshot_id)
            .field("embedded_col_idx", &self.embedded_col_idx)
            .finish_non_exhaustive()
    }
}

impl std::fmt::Debug for DeleteUnit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeleteUnit")
            .field("snapshot_id", &self.snapshot_id)
            .field("embedded_col_idx", &self.embedded_col_idx)
            .finish_non_exhaustive()
    }
}

impl TableChangesExec {
    fn new(
        insert_units: Vec<InsertUnit>,
        delete_units: Vec<DeleteUnit>,
        table_schema: SchemaRef,
        output_schema: SchemaRef,
        table_len: usize,
        need_rowid: bool,
    ) -> Self {
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(output_schema.clone()),
            datafusion::physical_expr::Partitioning::UnknownPartitioning(1),
            datafusion::physical_plan::execution_plan::EmissionType::Final,
            datafusion::physical_plan::execution_plan::Boundedness::Bounded,
        ));
        Self {
            insert_units,
            delete_units,
            table_schema,
            output_schema,
            table_len,
            need_rowid,
            properties,
        }
    }
}

impl DisplayAs for TableChangesExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default
            | DisplayFormatType::Verbose
            | DisplayFormatType::TreeRender => {
                write!(
                    f,
                    "TableChangesExec: inserts={}, deletes={}",
                    self.insert_units.len(),
                    self.delete_units.len()
                )
            },
        }
    }
}

impl ExecutionPlan for TableChangesExec {
    fn name(&self) -> &str {
        "TableChangesExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    /// No DataFusion children: the per-file scans are internal and executed
    /// directly, so the optimizer never rewrites them.
    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        if !children.is_empty() {
            return Err(DataFusionError::Internal(
                "TableChangesExec has no children".to_string(),
            ));
        }
        Ok(self)
    }

    fn schema(&self) -> SchemaRef {
        self.output_schema.clone()
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(DataFusionError::Internal(format!(
                "TableChangesExec only supports partition 0, got {partition}"
            )));
        }

        let insert_units = self.insert_units.clone();
        let delete_units = self.delete_units.clone();
        let output_schema = self.output_schema.clone();
        let table_len = self.table_len;
        let need_rowid = self.need_rowid;

        let fut = async move {
            correlate_changes(
                insert_units,
                delete_units,
                output_schema,
                table_len,
                need_rowid,
                context,
            )
            .await
        };

        let schema = self.output_schema.clone();
        let stream = futures::stream::once(fut)
            .map(|res: DataFusionResult<Vec<RecordBatch>>| match res {
                Ok(batches) => futures::stream::iter(batches.into_iter().map(Ok)).boxed(),
                Err(e) => futures::stream::iter(std::iter::once(Err(e))).boxed(),
            })
            .flatten();

        Ok(Box::pin(
            datafusion::physical_plan::stream::RecordBatchStreamAdapter::new(schema, stream),
        ))
    }
}

/// Collect the inserted and deleted rows, correlate update pairs by
/// `(snapshot_id, rowid)`, and produce the tagged output batches.
async fn correlate_changes(
    insert_units: Vec<InsertUnit>,
    delete_units: Vec<DeleteUnit>,
    output_schema: SchemaRef,
    table_len: usize,
    need_rowid: bool,
    context: Arc<TaskContext>,
) -> DataFusionResult<Vec<RecordBatch>> {
    // Inserted rows split into postimage candidates (embedded rowid — can pair
    // with a delete into an UPDATE) and plain inserts (fresh rowid — never pair).
    // A plain insert's rowid is synthesized only when actually requested; when it
    // is projected away it is a placeholder, so a non-rowid query never needs a
    // plain insert's row_id_start.
    let mut postimages: Vec<KeyedRows> = Vec::new();
    let mut plain_inserts: Vec<KeyedRows> = Vec::new();
    for unit in &insert_units {
        let batches =
            datafusion::physical_plan::collect(Arc::clone(&unit.scan), context.clone()).await?;
        for b in batches {
            let n = b.num_rows();
            if n == 0 {
                continue;
            }
            let table_batch = b.project(&(0..table_len).collect::<Vec<_>>())?;
            match unit.embedded_col_idx {
                // UPDATE / compaction postimage: rowid IS the embedded column.
                Some(idx) => {
                    let rowid = b
                        .column(idx)
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .ok_or_else(|| {
                            DataFusionError::Internal(
                                "embedded rowid column is not Int64".to_string(),
                            )
                        })?
                        .clone();
                    postimages.push(KeyedRows {
                        snapshot_id: unit.snapshot_id,
                        table_batch,
                        rowid,
                    });
                },
                // Plain insert: rowid = row_id_start + physical position when
                // requested, else a placeholder dropped by the projection.
                None => {
                    let rowid = if need_rowid {
                        let pos_idx = unit.pos_col_idx.ok_or_else(|| {
                            DataFusionError::Internal(
                                "plain insert scan is missing its position column".to_string(),
                            )
                        })?;
                        let row_id_start = unit.row_id_start.ok_or_else(|| {
                            DataFusionError::Internal(
                                "cannot synthesize rowid: inserted file has neither an embedded \
                                 rowid nor a row_id_start"
                                    .to_string(),
                            )
                        })?;
                        let pos = b
                            .column(pos_idx)
                            .as_any()
                            .downcast_ref::<Int64Array>()
                            .ok_or_else(|| {
                                DataFusionError::Internal(format!(
                                    "{ROW_POS_COLUMN_NAME} column is not Int64"
                                ))
                            })?;
                        Int64Array::from(
                            (0..n)
                                .map(|i| row_id_start + pos.value(i))
                                .collect::<Vec<i64>>(),
                        )
                    } else {
                        Int64Array::from(vec![0i64; n])
                    };
                    plain_inserts.push(KeyedRows {
                        snapshot_id: unit.snapshot_id,
                        table_batch,
                        rowid,
                    });
                },
            }
        }
    }

    // Deleted rows: the positions newly masked at this snapshot, with each row's
    // rowid (embedded column when the source file has one, else row_id_start +
    // physical position). The rowid is required only when it is output
    // (`need_rowid`) or when an update pair is possible (some postimage exists
    // to correlate against); with neither, every delete is a pure delete and
    // its rowid a placeholder — so a non-rowid projection over a delete-only
    // window never fails on a source file whose rowid cannot be synthesized
    // (no embedded rowid and a NULL row_id_start), mirroring
    // `ducklake_table_deletions`.
    let preimage_rowids_required = need_rowid || !postimages.is_empty();
    let mut preimages: Vec<KeyedRows> = Vec::new();
    for unit in &delete_units {
        let current = collect_delete_positions(&unit.current_delete_scan, context.clone()).await?;
        let current: HashSet<i64> = match current {
            Some(set) => set,
            None => (0..unit.record_count).collect(),
        };
        let previous = collect_delete_positions(&unit.previous_delete_scan, context.clone())
            .await?
            .unwrap_or_default();

        let data_batches =
            datafusion::physical_plan::collect(Arc::clone(&unit.data_scan), context.clone())
                .await?;
        for b in data_batches {
            let n = b.num_rows();
            if n == 0 {
                continue;
            }
            let pos_idx = b.schema().index_of(ROW_POS_COLUMN_NAME)?;
            let pos = b
                .column(pos_idx)
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| {
                    DataFusionError::Internal(format!("{ROW_POS_COLUMN_NAME} column is not Int64"))
                })?;
            let embedded = match unit.embedded_col_idx {
                Some(idx) => Some(
                    b.column(idx)
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .ok_or_else(|| {
                            DataFusionError::Internal(
                                "embedded rowid column is not Int64".to_string(),
                            )
                        })?,
                ),
                None => None,
            };

            // With no embedded rowid, deleted rowids are row_id_start + position;
            // require row_id_start in that case rather than emitting wrong ids —
            // but only when the rowid is actually consumed (output or pairing).
            let synth_start: Option<i64> = if embedded.is_none() && preimage_rowids_required {
                Some(unit.row_id_start.ok_or_else(|| {
                    DataFusionError::Internal(
                        "cannot synthesize deleted rowid: source file has neither an embedded \
                         rowid nor a row_id_start"
                            .to_string(),
                    )
                })?)
            } else {
                None
            };

            let mut keep: Vec<u32> = Vec::new();
            let mut rowids: Vec<i64> = Vec::new();
            for i in 0..n {
                let p = pos.value(i);
                if current.contains(&p) && !previous.contains(&p) {
                    keep.push(i as u32);
                    rowids.push(match (embedded, synth_start) {
                        (Some(arr), _) => arr.value(i),
                        (None, Some(start)) => start + p,
                        // Unneeded rowid (not output, nothing to pair with):
                        // a placeholder that update_keys can never contain.
                        (None, None) => 0,
                    });
                }
            }
            if keep.is_empty() {
                continue;
            }
            let indices = UInt32Array::from(keep);
            let table_cols: Vec<ArrayRef> = (0..table_len)
                .map(|c| {
                    take(b.column(c).as_ref(), &indices, None)
                        .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))
                })
                .collect::<DataFusionResult<_>>()?;
            let table_batch = RecordBatch::try_new(
                Arc::new(Schema::new(
                    (0..table_len)
                        .map(|c| b.schema().field(c).clone())
                        .collect::<Vec<_>>(),
                )),
                table_cols,
            )?;
            preimages.push(KeyedRows {
                snapshot_id: unit.snapshot_id,
                table_batch,
                rowid: Int64Array::from(rowids),
            });
        }
    }

    // Update pairs = a postimage (embedded, preserved rowid) whose (snapshot,
    // rowid) also appears as a delete. Only postimages seed the key set — plain
    // inserts carry fresh rowids that never match a delete (and, when rowid is
    // projected away, only a placeholder), so they must not participate.
    let post_keys: HashSet<(i64, i64)> = postimages
        .iter()
        .flat_map(|k| (0..k.rowid.len()).map(move |i| (k.snapshot_id, k.rowid.value(i))))
        .collect();
    let update_keys: HashSet<(i64, i64)> = preimages
        .iter()
        .flat_map(|k| (0..k.rowid.len()).map(move |i| (k.snapshot_id, k.rowid.value(i))))
        .filter(|key| post_keys.contains(key))
        .collect();

    let mut out: Vec<RecordBatch> = Vec::new();
    // Plain inserts are always `insert` (they never pair with a delete).
    for k in &plain_inserts {
        out.push(prepend_cdc_columns(
            &k.table_batch,
            k.rowid.clone(),
            k.snapshot_id,
            ChangeType::Insert,
            &output_schema,
        )?);
    }
    for k in &postimages {
        // Rows whose key is an update pair become postimages; the rest are plain
        // inserts (embedded file with no matching delete, e.g. compaction).
        if let Some(b) = filter_and_tag(
            k,
            &key_mask(k, &update_keys, true),
            ChangeType::UpdatePostimage,
            &output_schema,
        )? {
            out.push(b);
        }
        if let Some(b) = filter_and_tag(
            k,
            &key_mask(k, &update_keys, false),
            ChangeType::Insert,
            &output_schema,
        )? {
            out.push(b);
        }
    }
    for k in &preimages {
        // Rows paired with an insert surface as update preimages; the rest are
        // pure deletes, emitted as `delete` rows carrying the old values
        // (matching official DuckLake's table_changes).
        if let Some(b) = filter_and_tag(
            k,
            &key_mask(k, &update_keys, true),
            ChangeType::UpdatePreimage,
            &output_schema,
        )? {
            out.push(b);
        }
        if let Some(b) = filter_and_tag(
            k,
            &key_mask(k, &update_keys, false),
            ChangeType::Delete,
            &output_schema,
        )? {
            out.push(b);
        }
    }
    Ok(out)
}

/// Collect the `pos` set from a delete-file scan (`None` scan => `None`).
async fn collect_delete_positions(
    scan: &Option<Arc<dyn ExecutionPlan>>,
    context: Arc<TaskContext>,
) -> DataFusionResult<Option<HashSet<i64>>> {
    let Some(scan) = scan else {
        return Ok(None);
    };
    let batches = datafusion::physical_plan::collect(Arc::clone(scan), context).await?;
    let mut set = HashSet::new();
    for b in &batches {
        if b.num_columns() < 2 {
            continue;
        }
        let pos = b
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| {
                DataFusionError::Internal("delete `pos` column is not Int64".to_string())
            })?;
        for i in 0..pos.len() {
            if !pos.is_null(i) {
                set.insert(pos.value(i));
            }
        }
    }
    Ok(Some(set))
}

/// Prepend the CDC `snapshot_id` + `rowid` + `change_type` columns to a
/// table-column batch (official DuckLake column order). `rowid` must have the
/// same length as `table_batch`.
fn prepend_cdc_columns(
    table_batch: &RecordBatch,
    rowid: Int64Array,
    snapshot_id: i64,
    change: ChangeType,
    output_schema: &SchemaRef,
) -> DataFusionResult<RecordBatch> {
    let n = table_batch.num_rows();
    let mut cols: Vec<ArrayRef> = Vec::with_capacity(table_batch.num_columns() + CDC_COLS);
    cols.push(Arc::new(Int64Array::from(vec![snapshot_id; n])));
    cols.push(Arc::new(rowid));
    cols.push(Arc::new(StringArray::from(vec![change.as_str(); n])));
    cols.extend(table_batch.columns().iter().cloned());
    RecordBatch::try_new(output_schema.clone(), cols)
        .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))
}

/// A row-selection mask over `keyed`: `want_update` selects rows whose
/// `(snapshot_id, rowid)` is (or is not) an update pair.
fn key_mask(
    keyed: &KeyedRows,
    update_keys: &HashSet<(i64, i64)>,
    want_update: bool,
) -> BooleanArray {
    BooleanArray::from(
        (0..keyed.rowid.len())
            .map(|i| {
                let is_update = update_keys.contains(&(keyed.snapshot_id, keyed.rowid.value(i)));
                is_update == want_update
            })
            .collect::<Vec<bool>>(),
    )
}

/// Filter `keyed`'s table columns by `mask`, tag with `change`, and append the
/// CDC columns. Returns `None` when the mask selects no rows.
fn filter_and_tag(
    keyed: &KeyedRows,
    mask: &BooleanArray,
    change: ChangeType,
    output_schema: &SchemaRef,
) -> DataFusionResult<Option<RecordBatch>> {
    if mask.true_count() == 0 {
        return Ok(None);
    }
    let cols: Vec<ArrayRef> = keyed
        .table_batch
        .columns()
        .iter()
        .map(|c| {
            arrow::compute::filter(c.as_ref(), mask)
                .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))
        })
        .collect::<DataFusionResult<_>>()?;
    let filtered = RecordBatch::try_new(keyed.table_batch.schema(), cols)?;
    let rowid = arrow::compute::filter(&keyed.rowid, mask)
        .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| DataFusionError::Internal("filtered rowid is not Int64".to_string()))?
        .clone();
    Ok(Some(prepend_cdc_columns(
        &filtered,
        rowid,
        keyed.snapshot_id,
        change,
        output_schema,
    )?))
}
