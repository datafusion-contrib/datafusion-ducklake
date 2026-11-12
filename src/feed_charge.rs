//! The memory a change feed holds from planning to the end of execution,
//! charged to the session's memory pool.
//!
//! The catalog's inlined deletions of a window, and the first page of its
//! inlined rows (all of them, for a window that fits a page), are read at
//! planning, inside the plan's consistent catalog read, and the plan keeps them
//! until it is dropped; the inlined rows past the first page are read as the
//! plan executes ([`crate::inlined_feed`]). Each read is charged as soon as it
//! returns, so a window whose inlined history does not fit the pool fails with
//! a resources error there, before the plan is built on it; the charge then
//! follows what the plan keeps and is released with the plan.
//!
//! What one catalog statement returns is allocated before it can be charged.
//! The overshoot is bounded by one statement's result, which is freed as soon
//! as the charge refuses it: a page of inlined rows, on a provider that pages
//! them, and otherwise the whole selection.

use std::fmt;
use std::sync::Arc;

use arrow::array::{ArrayRef, Int64Array, RecordBatch, RecordBatchOptions};
use arrow::datatypes::SchemaRef;
use datafusion::catalog::Session;
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::common::{Result as DataFusionResult, Statistics};
use datafusion::error::DataFusionError;
use datafusion::execution::memory_pool::{MemoryConsumer, MemoryReservation};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::{EquivalenceProperties, PhysicalExpr};
use datafusion::physical_plan::common::compute_record_batch_statistics;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
};

use crate::metadata_provider::{DuckLakeInlinedRows, InlinedFileDeleteChange};
use crate::table_deletions::keyed_rows_schema;

/// The reservation one feed plan's planning-time memory is charged to. Every
/// node of the plan that holds that memory holds a clone, so it is released
/// when the last of them is dropped.
pub(crate) type PlanCharge = Arc<MemoryReservation>;

/// A reservation in the session's memory pool for the plan of `feed`.
pub(crate) fn plan_charge(state: &dyn Session, feed: &str) -> PlanCharge {
    Arc::new(
        MemoryConsumer::new(format!("{feed} planning")).register(&state.runtime_env().memory_pool),
    )
}

/// Release `bytes` of `charge`, no more than it holds.
pub(crate) fn release(charge: &PlanCharge, bytes: usize) {
    charge.shrink(bytes.min(charge.size()));
}

/// Heap and inline bytes of inlined rows as a catalog read returns them.
pub(crate) fn inlined_rows_bytes(rows: &Vec<DuckLakeInlinedRows>) -> usize {
    rows.capacity() * size_of::<DuckLakeInlinedRows>()
        + rows
            .iter()
            .map(|rows| {
                rows.table_name.capacity()
                    + rows.row_ids.capacity() * size_of::<i64>()
                    + rows.snapshots.capacity() * size_of::<i64>()
                    + rows.end_snapshots.capacity() * size_of::<Option<i64>>()
                    + rows.batch.get_array_memory_size()
            })
            .sum::<usize>()
}

/// Heap and inline bytes of inlined deletions as a catalog read returns them.
pub(crate) fn file_deletes_bytes(changes: &Vec<InlinedFileDeleteChange>) -> usize {
    changes.capacity() * size_of::<InlinedFileDeleteChange>()
        + changes
            .iter()
            .map(|change| change.data_file_path.capacity())
            .sum::<usize>()
}

/// `rows`, read under the full table schema, as one batch of `schema`
/// (`[snapshot_id, rowid, table columns]`). The row ids and snapshots move into
/// the key columns without a copy.
pub(crate) fn keyed_inlined_batch(
    rows: DuckLakeInlinedRows,
    schema: &SchemaRef,
) -> DataFusionResult<RecordBatch> {
    let num_rows = rows.batch.num_rows();
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(rows.batch.num_columns() + 2);
    columns.push(Arc::new(Int64Array::from(rows.snapshots)));
    columns.push(Arc::new(Int64Array::from(rows.row_ids)));
    columns.extend(rows.batch.columns().iter().cloned());
    Ok(RecordBatch::try_new_with_options(
        Arc::clone(schema),
        columns,
        &RecordBatchOptions::new().with_row_count(Some(num_rows)),
    )?)
}

/// Inlined row versions of a window as `[snapshot_id, rowid, table columns]`
/// batches, built once from the catalog's read and shared by every node of
/// the plan that reads them.
#[derive(Debug, Clone)]
pub(crate) struct KeyedInlinedRows {
    batches: Arc<Vec<RecordBatch>>,
    charge: PlanCharge,
}

impl KeyedInlinedRows {
    /// No rows.
    pub(crate) fn empty(charge: &PlanCharge) -> Self {
        Self {
            batches: Arc::new(Vec::new()),
            charge: Arc::clone(charge),
        }
    }

    /// Key `rows`, read under the full `table_schema`. The row ids and
    /// snapshots move into the key columns without a copy.
    pub(crate) fn new(
        rows: Vec<DuckLakeInlinedRows>,
        table_schema: &SchemaRef,
        charge: &PlanCharge,
    ) -> DataFusionResult<Self> {
        let all: Vec<usize> = (0..table_schema.fields().len()).collect();
        let schema = keyed_rows_schema(table_schema, &all);
        let mut batches = Vec::with_capacity(rows.len());
        for rows in rows {
            batches.push(keyed_inlined_batch(rows, &schema)?);
        }
        Ok(Self {
            batches: Arc::new(batches),
            charge: Arc::clone(charge),
        })
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.batches.is_empty()
    }

    /// The bytes the batches hold.
    pub(crate) fn bytes(&self) -> usize {
        self.batches.capacity() * size_of::<RecordBatch>()
            + self
                .batches
                .iter()
                .map(RecordBatch::get_array_memory_size)
                .sum::<usize>()
    }

    /// The rows as `[snapshot_id, rowid, the table columns at
    /// table_indices]`, sharing these batches' arrays.
    pub(crate) fn exec(&self, table_indices: &[usize]) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let indices: Vec<usize> = [0, 1]
            .into_iter()
            .chain(table_indices.iter().map(|i| i + 2))
            .collect();
        let mut batches = Vec::with_capacity(self.batches.len());
        for batch in self.batches.iter() {
            batches.push(batch.project(&indices)?);
        }
        let schema = match self.batches.first() {
            Some(batch) => Arc::new(batch.schema().project(&indices)?),
            None => {
                return Err(DataFusionError::Internal(
                    "keyed inlined rows planned with no batch".to_string(),
                ));
            },
        };
        Ok(Arc::new(ChargedMemoryExec::new(
            batches,
            schema,
            Arc::clone(&self.charge),
        )?))
    }
}

/// A single-partition source of batches the plan holds in memory, charged to
/// the plan's [`PlanCharge`] for as long as the node lives.
#[derive(Debug)]
pub(crate) struct ChargedMemoryExec {
    batches: Arc<Vec<RecordBatch>>,
    schema: SchemaRef,
    statistics: Statistics,
    properties: Arc<PlanProperties>,
    _charge: PlanCharge,
}

impl ChargedMemoryExec {
    fn new(
        batches: Vec<RecordBatch>,
        schema: SchemaRef,
        charge: PlanCharge,
    ) -> DataFusionResult<Self> {
        let statistics =
            compute_record_batch_statistics(std::slice::from_ref(&batches), &schema, None);
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(Arc::clone(&schema)),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ));
        Ok(Self {
            batches: Arc::new(batches),
            schema,
            statistics,
            properties,
            _charge: charge,
        })
    }
}

impl DisplayAs for ChargedMemoryExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "ChargedMemoryExec: batches={}", self.batches.len())
    }
}

impl ExecutionPlan for ChargedMemoryExec {
    fn name(&self) -> &str {
        "ChargedMemoryExec"
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> DataFusionResult<TreeNodeRecursion>,
    ) -> DataFusionResult<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        if !children.is_empty() {
            return Err(DataFusionError::Internal(
                "ChargedMemoryExec has no children".to_string(),
            ));
        }
        Ok(self)
    }

    fn execute(
        &self,
        partition: usize,
        _context: Arc<TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(DataFusionError::Internal(format!(
                "ChargedMemoryExec has 1 partition, got {partition}"
            )));
        }
        let batches = Arc::clone(&self.batches);
        let stream = futures::stream::iter((0..batches.len()).map(move |i| Ok(batches[i].clone())));
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&self.schema),
            stream,
        )))
    }

    fn partition_statistics(&self, partition: Option<usize>) -> DataFusionResult<Arc<Statistics>> {
        match partition {
            None | Some(0) => Ok(Arc::new(self.statistics.clone())),
            Some(p) => Err(DataFusionError::Internal(format!(
                "ChargedMemoryExec has 1 partition, got {p}"
            ))),
        }
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}
