//! Table deletions functionality for DuckLake
//!
//! This module provides the `ducklake_table_deletions()` table function that returns
//! the actual deleted rows between snapshots, with CDC metadata columns.
//!
//! For each data file with deletions:
//! 1. Read positions from current delete file (or all positions for full file delete)
//! 2. Subtract positions from previous delete file (if exists)
//! 3. Read the data file and return only the rows at the newly deleted positions
//! 4. Append CDC columns (snapshot_id, change_type='delete')

use std::collections::HashSet;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow::array::{ArrayRef, Int64Array, StringArray, UInt32Array};
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
use futures::Stream;
use object_store::path::Path as ObjectPath;
use parquet::arrow::ParquetRecordBatchStreamBuilder;
use parquet::arrow::async_reader::ParquetObjectReader;

use crate::metadata_provider::{DeleteFileChange, MetadataProvider};
use crate::path_resolver::resolve_path;
use crate::row_id::ROW_ID_PARQUET_FIELD_ID;
use crate::table::{validated_file_size, validated_record_count};
use crate::types::extract_parquet_field_ids;

/// Delete file schema: (file_path: VARCHAR, pos: INT64)
fn delete_file_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("file_path", DataType::Utf8, false),
        Field::new("pos", DataType::Int64, false),
    ]))
}

/// TableProvider that exposes deleted rows between snapshots
///
/// For each data file with deletions:
/// 1. Read positions from current delete file (or generate all positions for full file delete)
/// 2. Subtract positions from previous delete file (if exists)
/// 3. Read the data file and filter to only deleted row positions
/// 4. Append snapshot_id and change_type columns
#[derive(Debug)]
pub struct TableDeletionsTable {
    provider: Arc<dyn MetadataProvider>,
    table_id: i64,
    start_snapshot: i64,
    end_snapshot: i64,
    object_store_url: Arc<ObjectStoreUrl>,
    table_path: String,
    /// Original table schema (without CDC columns)
    table_schema: SchemaRef,
    /// Combined schema: table columns + rowid + snapshot_id + change_type
    output_schema: SchemaRef,
}

impl TableDeletionsTable {
    pub fn new(
        provider: Arc<dyn MetadataProvider>,
        table_id: i64,
        start_snapshot: i64,
        end_snapshot: i64,
        object_store_url: Arc<ObjectStoreUrl>,
        table_path: String,
        table_schema: SchemaRef,
    ) -> Self {
        // Build output schema: table columns + CDC metadata columns
        // (rowid, snapshot_id, change_type), in that order.
        let mut fields: Vec<Field> = table_schema
            .fields()
            .iter()
            .map(|f| f.as_ref().clone())
            .collect();
        // rowid is nullable for symmetry with ducklake_table_changes (where it is
        // NULL on encrypted tables); the deletions path always synthesizes a
        // non-null value for the cases it supports.
        fields.push(Field::new("rowid", DataType::Int64, true));
        fields.push(Field::new("snapshot_id", DataType::Int64, false));
        fields.push(Field::new("change_type", DataType::Utf8, false));
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

    /// Build execution plan for a single delete file entry. `need_rowid` gates
    /// resolving the rowid (footer probe + embedded read + synthesis); when the
    /// caller projected rowid away, it is emitted as a placeholder and dropped.
    async fn build_exec_for_delete_entry(
        &self,
        state: &dyn Session,
        need_rowid: bool,
        delete_file: &DeleteFileChange,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        // Resolve data file path
        let data_file_path = resolve_path(
            &self.table_path,
            &delete_file.data_file_path,
            delete_file.data_file_path_is_relative,
        )
        .map_err(|e| DataFusionError::External(Box::new(e)))?;

        // Create scan for current delete file (if exists - None means full file delete)
        let current_delete_exec = if let Some(ref current_path) = delete_file.current_delete_path {
            Some(self.build_delete_file_scan(
                current_path,
                delete_file.current_delete_path_is_relative.unwrap_or(true),
                delete_file.current_delete_file_size_bytes.unwrap_or(0),
                delete_file.current_delete_footer_size.unwrap_or(0),
            )?)
        } else {
            None
        };

        // Create scan for previous delete file (if exists)
        let previous_delete_exec = if let Some(ref prev_path) = delete_file.previous_delete_path {
            Some(self.build_delete_file_scan(
                prev_path,
                delete_file.previous_delete_path_is_relative.unwrap_or(true),
                delete_file.previous_delete_file_size_bytes.unwrap_or(0),
                delete_file.previous_delete_footer_size.unwrap_or(0),
            )?)
        } else {
            None
        };

        // Detect the source file's embedded rowid column (present on UPDATE /
        // compaction outputs). When present, a deleted row's rowid is that
        // embedded value — NOT `row_id_start + position`, which would key the
        // delete differently from the row's insert/update_postimage. Only probe
        // when the rowid is actually requested.
        let table_len = self.table_schema.fields().len();
        let embedded_name = if need_rowid {
            self.detect_embedded_rowid_name(
                state,
                &delete_file.data_file_path,
                delete_file.data_file_path_is_relative,
            )
            .await?
        } else {
            None
        };
        let embedded_col_idx = embedded_name.as_ref().map(|_| table_len);

        // Create scan for data file (with the embedded rowid column when present)
        let data_file_exec = self.build_data_file_scan(
            &data_file_path,
            delete_file.data_file_size_bytes,
            delete_file.data_file_footer_size.unwrap_or(0),
            &embedded_name,
        )?;

        // Validate record_count before use — a negative value from corrupt metadata
        // would cause incorrect behavior (e.g., empty ranges in full-file deletes).
        validated_record_count(delete_file.data_record_count, &delete_file.data_file_path)?;

        Ok(Arc::new(DeletedRowsExec::new(
            current_delete_exec,
            previous_delete_exec,
            data_file_exec,
            delete_file.data_record_count,
            delete_file.snapshot_id,
            delete_file.data_row_id_start,
            table_len,
            embedded_col_idx,
            need_rowid,
            self.output_schema.clone(),
        )))
    }

    /// Read the source file's footer and return the physical name of its embedded
    /// row-id column ([`ROW_ID_PARQUET_FIELD_ID`]) when present. Such a file is an
    /// UPDATE / compaction output whose logical rowids come from that column.
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

    /// Build a ParquetExec for a delete file
    fn build_delete_file_scan(
        &self,
        path: &str,
        is_relative: bool,
        size_bytes: i64,
        footer_size: i64,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let resolved_path = resolve_path(&self.table_path, path, is_relative)
            .map_err(|e| DataFusionError::External(Box::new(e)))?;

        let mut pf = PartitionedFile::new(
            &resolved_path,
            validated_file_size(size_bytes, &resolved_path)?,
        );
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

    /// Build a scan of the source data file: table columns, plus the embedded
    /// rowid column when `embedded_name` is `Some`.
    ///
    /// NOTE: positions come from the stream's `row_offset` (arrival order), which
    /// is the file's physical position only for a non-repartitioned scan. Under
    /// file-scan repartitioning this exec is unsound (deletes missed / wrong
    /// rowids) — a pre-existing limitation tracked in
    /// <https://github.com/datafusion-contrib/datafusion-ducklake/issues/178>.
    fn build_data_file_scan(
        &self,
        path: &str,
        size_bytes: i64,
        footer_size: i64,
        embedded_name: &Option<String>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let mut pf = PartitionedFile::new(path, validated_file_size(size_bytes, path)?);
        if footer_size > 0
            && let Ok(hint) = usize::try_from(footer_size)
        {
            pf = pf.with_metadata_size_hint(hint);
        }

        let read_schema = match embedded_name {
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
        };

        let builder = FileScanConfigBuilder::new(
            self.object_store_url.as_ref().clone(),
            Arc::new(ParquetSource::new(read_schema)),
        )
        .with_file_group(FileGroup::new(vec![pf]));

        Ok(DataSourceExec::from_data_source(builder.build()))
    }
}

#[async_trait]
impl TableProvider for TableDeletionsTable {
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
        // Get delete files added between snapshots
        let delete_files = self
            .provider
            .get_delete_files_added_between_snapshots(
                self.table_id,
                self.start_snapshot,
                self.end_snapshot,
            )
            .map_err(|e| DataFusionError::External(Box::new(e)))?;

        // Handle empty case
        if delete_files.is_empty() {
            use datafusion::physical_plan::empty::EmptyExec;
            let output_schema = match projection {
                Some(indices) => {
                    let fields: Vec<Field> = indices
                        .iter()
                        .map(|&i| self.output_schema.field(i).clone())
                        .collect();
                    Arc::new(Schema::new(fields))
                },
                None => self.output_schema.clone(),
            };
            return Ok(Arc::new(EmptyExec::new(output_schema)));
        }

        // Does the caller actually want `rowid`? It follows the table columns in
        // the output schema. When it is projected away we skip resolving it (no
        // footer probe, no synthesis), so a query like `SELECT id, change_type
        // FROM ducklake_table_deletions(...)` never fails on a file whose rowid
        // cannot be synthesized (no embedded rowid and a NULL row_id_start).
        let rowid_idx = self.table_schema.fields().len();
        let need_rowid = projection.is_none_or(|indices| indices.contains(&rowid_idx));

        // Build execution plan for each delete entry
        let mut execs: Vec<Arc<dyn ExecutionPlan>> = Vec::with_capacity(delete_files.len());
        for delete_file in &delete_files {
            let exec = self
                .build_exec_for_delete_entry(state, need_rowid, delete_file)
                .await?;
            execs.push(exec);
        }

        // Combine with UnionExec if multiple
        let full: Arc<dyn ExecutionPlan> = if execs.len() == 1 {
            execs.into_iter().next().unwrap()
        } else {
            UnionExec::try_new(execs)?
        };

        // The exec emits the full `[table cols, rowid, snapshot_id, change_type]`
        // schema; honor the requested projection with a ProjectionExec on top.
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

/// Execution plan that reads deleted rows from a data file
///
/// 1. Reads current delete file to get deleted positions
/// 2. Reads previous delete file to get previously deleted positions (if exists)
/// 3. Computes delta: positions in current but not in previous
/// 4. Reads data file and filters to only include rows at deleted positions
/// 5. Appends CDC columns (snapshot_id, change_type='delete')
#[derive(Debug)]
pub struct DeletedRowsExec {
    /// Scan of current delete file (None for full file deletes)
    current_delete_scan: Option<Arc<dyn ExecutionPlan>>,
    /// Scan of previous delete file (if exists)
    previous_delete_scan: Option<Arc<dyn ExecutionPlan>>,
    /// Scan of data file
    data_file_scan: Arc<dyn ExecutionPlan>,
    /// Total record count in data file (used for full file deletes)
    record_count: i64,
    /// Snapshot ID for CDC column
    snapshot_id: i64,
    /// First rowid of the data file (`None` if the catalog carries none). Used
    /// to synthesize a deleted row's rowid as `row_id_start + physical position`
    /// when the source file has no embedded rowid.
    row_id_start: Option<i64>,
    /// Number of leading table columns in the data-file batch (the rest are the
    /// optional embedded rowid column).
    table_len: usize,
    /// Column index of the embedded rowid in the data-file batch, if present (an
    /// UPDATE / compaction output); its value IS the deleted row's rowid.
    embedded_col_idx: Option<usize>,
    /// Whether the rowid column is actually requested. When false, rowid is
    /// emitted as a placeholder (dropped by the projection above) and neither an
    /// embedded rowid nor a row_id_start is required.
    need_rowid: bool,
    /// Output schema (table columns + rowid + snapshot_id + change_type)
    output_schema: SchemaRef,
    /// Cached plan properties
    properties: Arc<PlanProperties>,
}

impl DeletedRowsExec {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        current_delete_scan: Option<Arc<dyn ExecutionPlan>>,
        previous_delete_scan: Option<Arc<dyn ExecutionPlan>>,
        data_file_scan: Arc<dyn ExecutionPlan>,
        record_count: i64,
        snapshot_id: i64,
        row_id_start: Option<i64>,
        table_len: usize,
        embedded_col_idx: Option<usize>,
        need_rowid: bool,
        output_schema: SchemaRef,
    ) -> Self {
        let eq_properties = EquivalenceProperties::new(output_schema.clone());
        let properties = Arc::new(PlanProperties::new(
            eq_properties,
            data_file_scan.output_partitioning().clone(),
            data_file_scan.pipeline_behavior(),
            data_file_scan.boundedness(),
        ));

        Self {
            current_delete_scan,
            previous_delete_scan,
            data_file_scan,
            record_count,
            snapshot_id,
            row_id_start,
            table_len,
            embedded_col_idx,
            need_rowid,
            output_schema,
            properties,
        }
    }
}

impl DisplayAs for DeletedRowsExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default
            | DisplayFormatType::Verbose
            | DisplayFormatType::TreeRender => {
                write!(
                    f,
                    "DeletedRowsExec: snapshot_id={}, full_delete={}, has_previous={}",
                    self.snapshot_id,
                    self.current_delete_scan.is_none(),
                    self.previous_delete_scan.is_some()
                )
            },
        }
    }
}

impl ExecutionPlan for DeletedRowsExec {
    fn name(&self) -> &str {
        "DeletedRowsExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        let mut children = Vec::new();
        if let Some(ref curr) = self.current_delete_scan {
            children.push(curr);
        }
        if let Some(ref prev) = self.previous_delete_scan {
            children.push(prev);
        }
        children.push(&self.data_file_scan);
        children
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let mut idx = 0;

        let current = if self.current_delete_scan.is_some() {
            let c = children
                .get(idx)
                .cloned()
                .ok_or_else(|| DataFusionError::Internal("Missing current delete child".into()))?;
            idx += 1;
            Some(c)
        } else {
            None
        };

        let previous = if self.previous_delete_scan.is_some() {
            let p = children
                .get(idx)
                .cloned()
                .ok_or_else(|| DataFusionError::Internal("Missing previous delete child".into()))?;
            idx += 1;
            Some(p)
        } else {
            None
        };

        let data = children
            .get(idx)
            .cloned()
            .ok_or_else(|| DataFusionError::Internal("Missing data file child".into()))?;

        Ok(Arc::new(DeletedRowsExec::new(
            current,
            previous,
            data,
            self.record_count,
            self.snapshot_id,
            self.row_id_start,
            self.table_len,
            self.embedded_col_idx,
            self.need_rowid,
            self.output_schema.clone(),
        )))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        let current_stream = self
            .current_delete_scan
            .as_ref()
            .map(|p| p.execute(partition, context.clone()))
            .transpose()?;
        let previous_stream = self
            .previous_delete_scan
            .as_ref()
            .map(|p| p.execute(partition, context.clone()))
            .transpose()?;
        let data_stream = self.data_file_scan.execute(partition, context)?;

        Ok(Box::pin(DeletedRowsStream::new(
            current_stream,
            previous_stream,
            data_stream,
            self.record_count,
            self.snapshot_id,
            self.row_id_start,
            self.table_len,
            self.embedded_col_idx,
            self.need_rowid,
            self.output_schema.clone(),
        )))
    }

    fn schema(&self) -> SchemaRef {
        self.output_schema.clone()
    }
}

/// Stream state machine
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamState {
    /// Reading current delete file
    ReadingCurrentDelete,
    /// Reading previous delete file
    ReadingPreviousDelete,
    /// Reading data file and filtering
    ReadingData,
    /// Done
    Done,
}

/// Stream that reads deleted rows from a data file
struct DeletedRowsStream {
    /// Current delete file stream (None for full file delete)
    current_delete_stream: Option<SendableRecordBatchStream>,
    /// Previous delete file stream (if exists)
    previous_delete_stream: Option<SendableRecordBatchStream>,
    /// Data file stream
    data_stream: SendableRecordBatchStream,
    /// Snapshot ID for CDC column
    snapshot_id: i64,
    /// First rowid of the data file (`None` if absent); used to synthesize a
    /// deleted row's rowid as `row_id_start + physical position` when there is
    /// no embedded rowid.
    row_id_start: Option<i64>,
    /// Number of leading table columns in each data-file batch.
    table_len: usize,
    /// Column index of the embedded rowid in the data-file batch, if present;
    /// its value IS the deleted row's rowid.
    embedded_col_idx: Option<usize>,
    /// Whether rowid is requested; when false it is emitted as a placeholder.
    need_rowid: bool,
    /// Output schema
    output_schema: SchemaRef,
    /// Collected current positions (or all positions for full delete)
    current_positions: HashSet<i64>,
    /// Collected previous positions
    previous_positions: HashSet<i64>,
    /// Computed delta positions (sorted)
    deleted_positions: Option<Vec<i64>>,
    /// Current row offset in data file (arrival order; see issue #178)
    row_offset: i64,
    /// State machine
    state: StreamState,
}

impl DeletedRowsStream {
    #[allow(clippy::too_many_arguments)]
    fn new(
        current_delete_stream: Option<SendableRecordBatchStream>,
        previous_delete_stream: Option<SendableRecordBatchStream>,
        data_stream: SendableRecordBatchStream,
        record_count: i64,
        snapshot_id: i64,
        row_id_start: Option<i64>,
        table_len: usize,
        embedded_col_idx: Option<usize>,
        need_rowid: bool,
        output_schema: SchemaRef,
    ) -> Self {
        // Determine initial state and compute positions if needed
        let (initial_state, current_positions, deleted_positions) =
            if current_delete_stream.is_some() {
                (StreamState::ReadingCurrentDelete, HashSet::new(), None)
            } else if previous_delete_stream.is_some() {
                // Full file delete but has previous - need to subtract previous positions
                let current: HashSet<i64> = (0..record_count).collect();
                (StreamState::ReadingPreviousDelete, current, None)
            } else {
                // Full file delete with no previous - all positions are deleted
                let positions: Vec<i64> = (0..record_count).collect();
                (
                    StreamState::ReadingData,
                    HashSet::new(),
                    Some(positions), // Pre-computed sorted positions
                )
            };

        Self {
            current_delete_stream,
            previous_delete_stream,
            data_stream,
            snapshot_id,
            row_id_start,
            table_len,
            embedded_col_idx,
            need_rowid,
            output_schema,
            current_positions,
            previous_positions: HashSet::new(),
            deleted_positions,
            row_offset: 0,
            state: initial_state,
        }
    }

    /// Extract positions from a delete file batch
    fn extract_positions(batch: &RecordBatch) -> HashSet<i64> {
        if batch.num_columns() < 2 {
            return HashSet::new();
        }

        let pos_array = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("pos column should be Int64");

        pos_array.values().iter().copied().collect()
    }

    /// Compute the delta and sort it
    fn compute_deleted_positions(&mut self) {
        let mut delta: Vec<i64> = self
            .current_positions
            .iter()
            .filter(|pos| !self.previous_positions.contains(pos))
            .copied()
            .collect();
        delta.sort_unstable();
        self.deleted_positions = Some(delta);
    }

    /// Filter batch to only include deleted rows and append CDC columns
    fn filter_batch(&mut self, batch: &RecordBatch) -> DataFusionResult<Option<RecordBatch>> {
        let deleted_positions = self.deleted_positions.as_ref().unwrap();
        let num_rows = batch.num_rows();

        // Resolve each deleted row's rowid: the embedded rowid column when the
        // source file has one (an UPDATE / compaction output), else
        // `row_id_start + physical position`.
        let embedded = match self.embedded_col_idx {
            Some(idx) => Some(
                batch
                    .column(idx)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .ok_or_else(|| {
                        DataFusionError::Internal("embedded rowid column is not Int64".to_string())
                    })?,
            ),
            None => None,
        };
        // Require a row_id_start only when a rowid is actually needed and there
        // is no embedded rowid to read.
        let synth_start: Option<i64> = if self.need_rowid && embedded.is_none() {
            Some(self.row_id_start.ok_or_else(|| {
                DataFusionError::Internal(
                    "cannot synthesize deleted rowid: source file has neither an embedded \
                     rowid nor a row_id_start"
                        .to_string(),
                )
            })?)
        } else {
            None
        };

        // Find which rows in this batch are deleted, capturing each one's rowid.
        // `global_pos` is the stream's `row_offset` (arrival order) — the file's
        // physical position for a non-repartitioned scan. See issue #178 for the
        // repartitioning limitation this shares with the delete-position match.
        let mut keep_indices: Vec<u32> = Vec::new();
        let mut rowids: Vec<i64> = Vec::new();
        for i in 0..num_rows {
            let global_pos = self.row_offset + i as i64;
            if deleted_positions.binary_search(&global_pos).is_ok() {
                keep_indices.push(i as u32);
                // When rowid is projected away, emit a placeholder (dropped by
                // the ProjectionExec above) so no rowid needs synthesizing.
                let rowid = if !self.need_rowid {
                    0
                } else {
                    match (embedded, synth_start) {
                        (Some(arr), _) => arr.value(i),
                        (None, Some(start)) => start + global_pos,
                        // synth_start is Some whenever rowid is needed and there
                        // is no embedded rowid (resolved above).
                        (None, None) => unreachable!("row_id_start resolved above"),
                    }
                };
                rowids.push(rowid);
            }
        }

        // Update row offset for next batch.
        self.row_offset += num_rows as i64;

        // If no deleted rows in this batch, return None
        if keep_indices.is_empty() {
            return Ok(None);
        }

        // Select the deleted rows' TABLE columns (excluding the embedded rowid
        // helper column, if the scan read one), then append the CDC columns.
        let indices = UInt32Array::from(keep_indices.clone());
        let mut columns: Vec<ArrayRef> = Vec::with_capacity(self.table_len + 3);
        for col in batch.columns().iter().take(self.table_len) {
            let filtered = take(col.as_ref(), &indices, None)
                .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?;
            columns.push(filtered);
        }

        // Append CDC columns: rowid, snapshot_id, change_type.
        let num_output_rows = keep_indices.len();
        columns.push(Arc::new(Int64Array::from(rowids)));
        columns.push(Arc::new(Int64Array::from(vec![
            self.snapshot_id;
            num_output_rows
        ])));
        columns.push(Arc::new(StringArray::from(vec!["delete"; num_output_rows])));

        RecordBatch::try_new(self.output_schema.clone(), columns)
            .map(Some)
            .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))
    }
}

impl Stream for DeletedRowsStream {
    type Item = DataFusionResult<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            match self.state {
                StreamState::ReadingCurrentDelete => {
                    let current = self.current_delete_stream.as_mut().unwrap();
                    match Pin::new(current).poll_next(cx) {
                        Poll::Ready(Some(Ok(batch))) => {
                            let positions = Self::extract_positions(&batch);
                            self.current_positions.extend(positions);
                        },
                        Poll::Ready(Some(Err(e))) => return Poll::Ready(Some(Err(e))),
                        Poll::Ready(None) => {
                            if self.previous_delete_stream.is_some() {
                                self.state = StreamState::ReadingPreviousDelete;
                            } else {
                                self.compute_deleted_positions();
                                self.state = StreamState::ReadingData;
                            }
                        },
                        Poll::Pending => return Poll::Pending,
                    }
                },
                StreamState::ReadingPreviousDelete => {
                    let prev = self.previous_delete_stream.as_mut().unwrap();
                    match Pin::new(prev).poll_next(cx) {
                        Poll::Ready(Some(Ok(batch))) => {
                            let positions = Self::extract_positions(&batch);
                            self.previous_positions.extend(positions);
                        },
                        Poll::Ready(Some(Err(e))) => return Poll::Ready(Some(Err(e))),
                        Poll::Ready(None) => {
                            self.compute_deleted_positions();
                            self.state = StreamState::ReadingData;
                        },
                        Poll::Pending => return Poll::Pending,
                    }
                },
                StreamState::ReadingData => {
                    match Pin::new(&mut self.data_stream).poll_next(cx) {
                        Poll::Ready(Some(Ok(batch))) => {
                            match self.filter_batch(&batch)? {
                                Some(filtered) => return Poll::Ready(Some(Ok(filtered))),
                                None => continue, // No deleted rows in this batch
                            }
                        },
                        Poll::Ready(Some(Err(e))) => return Poll::Ready(Some(Err(e))),
                        Poll::Ready(None) => {
                            self.state = StreamState::Done;
                            return Poll::Ready(None);
                        },
                        Poll::Pending => return Poll::Pending,
                    }
                },
                StreamState::Done => {
                    return Poll::Ready(None);
                },
            }
        }
    }
}

impl RecordBatchStream for DeletedRowsStream {
    fn schema(&self) -> SchemaRef {
        self.output_schema.clone()
    }
}
