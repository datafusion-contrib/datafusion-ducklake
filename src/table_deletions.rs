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

use std::any::Any;
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
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::union::UnionExec;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, PlanProperties,
};
use futures::Stream;

use crate::metadata_provider::{DeleteFileChange, MetadataProvider};
use crate::path_resolver::resolve_path;
use crate::table::{validated_file_size, validated_record_count};

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
    /// Combined schema: table columns + snapshot_id + change_type
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
        let mut fields: Vec<Field> = table_schema
            .fields()
            .iter()
            .map(|f| f.as_ref().clone())
            .collect();
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

    /// Build execution plan for a single delete file entry
    fn build_exec_for_delete_entry(
        &self,
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

        // Create scan for data file
        let data_file_exec = self.build_data_file_scan(
            &data_file_path,
            delete_file.data_file_size_bytes,
            delete_file.data_file_footer_size,
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
            self.output_schema.clone(),
        )))
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
        if let Some(hint) = crate::parquet_meta::metadata_size_hint_from_footer(Some(footer_size)) {
            pf = pf.with_metadata_size_hint(hint);
        }

        let builder = FileScanConfigBuilder::new(
            self.object_store_url.as_ref().clone(),
            Arc::new(ParquetSource::new(delete_file_schema())),
        )
        .with_file_group(FileGroup::new(vec![pf]));

        Ok(DataSourceExec::from_data_source(builder.build()))
    }

    /// Build a ParquetExec for a data file
    fn build_data_file_scan(
        &self,
        path: &str,
        size_bytes: i64,
        footer_size: i64,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let mut pf = PartitionedFile::new(path, validated_file_size(size_bytes, path)?);
        if let Some(hint) = crate::parquet_meta::metadata_size_hint_from_footer(Some(footer_size)) {
            pf = pf.with_metadata_size_hint(hint);
        }

        let builder = FileScanConfigBuilder::new(
            self.object_store_url.as_ref().clone(),
            Arc::new(ParquetSource::new(self.table_schema.clone())),
        )
        .with_file_group(FileGroup::new(vec![pf]));

        Ok(DataSourceExec::from_data_source(builder.build()))
    }
}

#[async_trait]
impl TableProvider for TableDeletionsTable {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.output_schema.clone()
    }

    fn table_type(&self) -> TableType {
        TableType::View
    }

    async fn scan(
        &self,
        _state: &dyn Session,
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

        // Build execution plan for each delete entry
        let mut execs: Vec<Arc<dyn ExecutionPlan>> = Vec::with_capacity(delete_files.len());
        for delete_file in &delete_files {
            let exec = self.build_exec_for_delete_entry(delete_file)?;
            execs.push(exec);
        }

        // Combine with UnionExec if multiple
        if execs.len() == 1 {
            Ok(execs.into_iter().next().unwrap())
        } else {
            UnionExec::try_new(execs)
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
    /// Output schema (table columns + snapshot_id + change_type)
    output_schema: SchemaRef,
    /// Cached plan properties
    properties: Arc<PlanProperties>,
}

impl DeletedRowsExec {
    pub fn new(
        current_delete_scan: Option<Arc<dyn ExecutionPlan>>,
        previous_delete_scan: Option<Arc<dyn ExecutionPlan>>,
        data_file_scan: Arc<dyn ExecutionPlan>,
        record_count: i64,
        snapshot_id: i64,
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

    fn as_any(&self) -> &dyn Any {
        self
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
    /// Output schema
    output_schema: SchemaRef,
    /// Collected current positions (or all positions for full delete)
    current_positions: HashSet<i64>,
    /// Collected previous positions
    previous_positions: HashSet<i64>,
    /// Computed delta positions (sorted)
    deleted_positions: Option<Vec<i64>>,
    /// Current row offset in data file
    row_offset: i64,
    /// State machine
    state: StreamState,
}

impl DeletedRowsStream {
    fn new(
        current_delete_stream: Option<SendableRecordBatchStream>,
        previous_delete_stream: Option<SendableRecordBatchStream>,
        data_stream: SendableRecordBatchStream,
        record_count: i64,
        snapshot_id: i64,
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

        // Find which rows in this batch are deleted
        let mut keep_indices: Vec<u32> = Vec::new();
        for i in 0..num_rows {
            let global_pos = self.row_offset + i as i64;
            if deleted_positions.binary_search(&global_pos).is_ok() {
                keep_indices.push(i as u32);
            }
        }

        // Update row offset for next batch
        self.row_offset += num_rows as i64;

        // If no deleted rows in this batch, return None
        if keep_indices.is_empty() {
            return Ok(None);
        }

        // Use Arrow's take kernel to select rows
        let indices = UInt32Array::from(keep_indices.clone());
        let mut columns: Vec<ArrayRef> = Vec::with_capacity(batch.num_columns() + 2);

        for col in batch.columns() {
            let filtered = take(col.as_ref(), &indices, None)
                .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?;
            columns.push(filtered);
        }

        // Append CDC columns
        let num_output_rows = keep_indices.len();
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
