//! The `rowid` of each row of a scan over many data files, resolved per file as
//! the reader opens it.
//!
//! Official DuckLake computes a row's id when a file is opened
//! (`DuckLakeMultiFileReader::GetVirtualColumnExpression`): the file's own
//! `_ducklake_internal_row_id` column when it carries one — found by its reserved
//! field id, as a file an UPDATE or compaction rewrote does — and otherwise the
//! file's catalog `row_id_start` plus the row's physical position. A file with
//! neither fails the read.
//!
//! The scan under [`RowLineageExec`] supplies, for every row, the embedded id as
//! the file stores it (NULL when it stores none), whether the file stores one at
//! all, the file's `row_id_start` and catalog id as per-file partition values, and
//! the row's physical position. The field-id adapter answers the first two per
//! file ([`crate::field_id_adapter::FieldIdExprAdapterFactory::with_row_lineage`]).

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, BooleanArray, Int64Array, Int64Builder, RecordBatch};
use arrow::datatypes::{Field, Schema, SchemaRef};
use datafusion::common::config::ConfigOptions;
use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_expr::projection::ProjectionMapping;
use datafusion::physical_expr::{EquivalenceProperties, PhysicalExpr, PhysicalSortExpr};
use datafusion::physical_plan::execution_plan::Boundedness;
use datafusion::physical_plan::filter_pushdown::{
    ChildFilterDescription, FilterDescription, FilterPushdownPhase,
};
use datafusion::physical_plan::sort_pushdown::SortOrderPushdownResult;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    ChildStats, ColumnStatistics, DisplayAs, DisplayFormatType, ExecutionPlan,
    ExecutionPlanProperties, Partitioning, PlanProperties, Statistics, StatisticsArgs,
};
use futures::StreamExt;

use crate::row_id::rowid_field;

/// Where [`RowLineageExec`] finds each of its inputs in the scan's output.
#[derive(Debug, Clone, Copy)]
pub(crate) struct LineageColumns {
    pub(crate) embedded: usize,
    pub(crate) has_embedded: usize,
    pub(crate) file_id: usize,
    pub(crate) row_id_start: usize,
    pub(crate) position: usize,
}

/// Computes `rowid` for each row.
///
/// The output is the input's first `data_columns` columns, then `rowid`, then
/// the data file id and the row position, which the delete filter above
/// consumes. Those columns keep their input positions, so the input's orderings
/// over them hold here.
#[derive(Debug)]
pub(crate) struct RowLineageExec {
    input: Arc<dyn ExecutionPlan>,
    columns: LineageColumns,
    data_columns: usize,
    /// Data file paths by catalog id, for the error a file without lineage raises.
    paths: Arc<HashMap<i64, String>>,
    schema: SchemaRef,
    properties: Arc<PlanProperties>,
}

impl RowLineageExec {
    pub(crate) fn try_new(
        input: Arc<dyn ExecutionPlan>,
        columns: LineageColumns,
        data_columns: usize,
        paths: HashMap<i64, String>,
    ) -> DataFusionResult<Self> {
        Self::assemble(input, columns, data_columns, Arc::new(paths))
    }

    fn assemble(
        input: Arc<dyn ExecutionPlan>,
        columns: LineageColumns,
        data_columns: usize,
        paths: Arc<HashMap<i64, String>>,
    ) -> DataFusionResult<Self> {
        let input_schema = input.schema();
        if data_columns > input_schema.fields().len() {
            return Err(DataFusionError::Internal(
                "RowLineageExec: more data columns than the input has".into(),
            ));
        }
        let mut fields: Vec<Arc<Field>> = input_schema.fields()[..data_columns].to_vec();
        fields.push(Arc::new(rowid_field()));
        fields.push(Arc::new(input_schema.field(columns.file_id).clone()));
        fields.push(Arc::new(input_schema.field(columns.position).clone()));
        let schema: SchemaRef = Arc::new(Schema::new(fields));
        let prefix: Vec<usize> = (0..data_columns).collect();
        let equivalence = match ProjectionMapping::from_indices(&prefix, &input_schema) {
            Ok(mapping) => input
                .equivalence_properties()
                .project(&mapping, Arc::clone(&schema)),
            Err(_) => EquivalenceProperties::new(Arc::clone(&schema)),
        };
        let properties = Arc::new(PlanProperties::new(
            equivalence,
            Partitioning::UnknownPartitioning(input.output_partitioning().partition_count()),
            input.pipeline_behavior(),
            Boundedness::Bounded,
        ));
        Ok(Self {
            input,
            columns,
            data_columns,
            paths,
            schema,
            properties,
        })
    }
}

impl DisplayAs for RowLineageExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "RowLineageExec")
    }
}

impl ExecutionPlan for RowLineageExec {
    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> DataFusionResult<TreeNodeRecursion>,
    ) -> DataFusionResult<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn name(&self) -> &str {
        "RowLineageExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    /// No: the output has `rowid` where the input has the embedded-row-id
    /// probe, so an ordering requirement the optimizer pushed through by column
    /// index would land on the wrong column. The orderings over the data
    /// columns are republished in the equivalence properties instead.
    fn maintains_input_order(&self) -> Vec<bool> {
        vec![false]
    }

    /// No repartition below this node, for the reason
    /// [`crate::lazy_delete_filter::LazyDeleteFilterExec`] gives.
    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![false]
    }

    fn supports_limit_pushdown(&self) -> bool {
        true
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let [child] = <[_; 1]>::try_from(children).map_err(|_| {
            DataFusionError::Internal("RowLineageExec expects exactly one child".into())
        })?;
        Ok(Arc::new(Self::assemble(
            child,
            self.columns,
            self.data_columns,
            Arc::clone(&self.paths),
        )?))
    }

    /// Forwards a filter only when it references no `rowid`: the scan below
    /// has no such column. [`ChildFilterDescription::from_child`] resolves
    /// columns by name, so a filter on `rowid` is reported unsupported and stays
    /// above.
    fn gather_filters_for_pushdown(
        &self,
        _phase: FilterPushdownPhase,
        parent_filters: Vec<Arc<dyn PhysicalExpr>>,
        _config: &ConfigOptions,
    ) -> DataFusionResult<FilterDescription> {
        Ok(
            FilterDescription::new().with_child(ChildFilterDescription::from_child(
                &parent_filters,
                &self.input,
            )?),
        )
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        let input = self.input.execute(partition, context)?;
        let columns = self.columns;
        let data_columns = self.data_columns;
        let paths = Arc::clone(&self.paths);
        let schema = Arc::clone(&self.schema);
        let stream_schema = Arc::clone(&schema);
        let stream = input.map(move |batch| {
            let batch = batch?;
            let rowid = row_ids(&batch, columns, &paths)?;
            let mut arrays: Vec<ArrayRef> = batch.columns()[..data_columns].to_vec();
            arrays.push(rowid);
            arrays.push(Arc::clone(batch.column(columns.file_id)));
            arrays.push(Arc::clone(batch.column(columns.position)));
            Ok(RecordBatch::try_new(Arc::clone(&stream_schema), arrays)?)
        });
        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream)))
    }

    /// An ordering over the data columns is served by the scan, which reads
    /// them at the same positions; one on `rowid` or a hidden column is not.
    /// `Inexact`, as the rename node above reports it: this node republishes the
    /// ordering, but the nodes above may not.
    fn try_pushdown_sort(
        &self,
        order: &[PhysicalSortExpr],
    ) -> DataFusionResult<SortOrderPushdownResult<Arc<dyn ExecutionPlan>>> {
        if !references_only_prefix(order, self.data_columns) {
            return Ok(SortOrderPushdownResult::Unsupported);
        }
        let columns = self.columns;
        let data_columns = self.data_columns;
        let paths = Arc::clone(&self.paths);
        self.input.try_pushdown_sort(order)?.into_inexact().try_map(
            |inner| -> DataFusionResult<Arc<dyn ExecutionPlan>> {
                Ok(Arc::new(Self::assemble(
                    inner,
                    columns,
                    data_columns,
                    paths,
                )?))
            },
        )
    }

    fn child_stats_requests(&self, partition: Option<usize>) -> Vec<ChildStats> {
        vec![ChildStats::At(partition)]
    }

    /// One output row per input row: the count carries through. Column bounds
    /// are not forwarded.
    fn statistics_from_inputs(
        &self,
        input_stats: &[Arc<Statistics>],
        _args: &StatisticsArgs,
    ) -> DataFusionResult<Arc<Statistics>> {
        let mut statistics = Statistics::new_unknown(&self.schema);
        if let Some(input) = input_stats.first() {
            statistics.num_rows = input.num_rows;
            statistics.total_byte_size = input.total_byte_size.to_inexact();
        }
        statistics.column_statistics =
            vec![ColumnStatistics::new_unknown(); self.schema.fields().len()];
        Ok(Arc::new(statistics))
    }
}

/// Each row's id: the file's embedded id when the file stores one, otherwise
/// `row_id_start` plus the row's position.
fn row_ids(
    batch: &RecordBatch,
    columns: LineageColumns,
    paths: &HashMap<i64, String>,
) -> DataFusionResult<ArrayRef> {
    let embedded = int64(batch, columns.embedded, "embedded row id")?;
    let has_embedded = batch
        .column(columns.has_embedded)
        .as_any()
        .downcast_ref::<BooleanArray>()
        .ok_or_else(|| {
            DataFusionError::Internal("the scan's embedded-row-id flag is not Boolean".into())
        })?;
    let file_ids = int64(batch, columns.file_id, "data file id")?;
    let starts = int64(batch, columns.row_id_start, "row_id_start")?;
    let positions = int64(batch, columns.position, "row position")?;
    let mut ids = Int64Builder::with_capacity(batch.num_rows());
    for row in 0..batch.num_rows() {
        if has_embedded.is_valid(row) && has_embedded.value(row) {
            ids.append_option((!embedded.is_null(row)).then(|| embedded.value(row)));
        } else if starts.is_valid(row) {
            ids.append_value(starts.value(row) + positions.value(row));
        } else {
            let path = paths
                .get(&file_ids.value(row))
                .map(String::as_str)
                .unwrap_or("<unknown>");
            return Err(DataFusionError::Execution(format!(
                "File \"{path}\" has no embedded `_ducklake_internal_row_id` column and no \
                 `row_id_start` set in the catalog — row lineage cannot be reconstructed"
            )));
        }
    }
    Ok(Arc::new(ids.finish()))
}

fn int64<'a>(batch: &'a RecordBatch, index: usize, what: &str) -> DataFusionResult<&'a Int64Array> {
    batch
        .column(index)
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| DataFusionError::Internal(format!("the scan's {what} column is not Int64")))
}

/// Whether every column `order` references sits among the first `prefix`
/// columns.
pub(crate) fn references_only_prefix(order: &[PhysicalSortExpr], prefix: usize) -> bool {
    order.iter().all(|sort| {
        let mut inside = true;
        sort.expr
            .apply(|expr| {
                if let Some(column) = expr.downcast_ref::<Column>()
                    && column.index() >= prefix
                {
                    inside = false;
                    return Ok(TreeNodeRecursion::Stop);
                }
                Ok(TreeNodeRecursion::Continue)
            })
            .expect("a column walk that never fails");
        inside
    })
}
