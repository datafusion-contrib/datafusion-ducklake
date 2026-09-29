//! Keeps a filter above a scan whose files are resolved as they open, unless the
//! reader is guaranteed to apply it.
//!
//! With `datafusion.execution.parquet.pushdown_filters` on, DataFusion removes
//! the `FilterExec` above a parquet scan for every predicate the scan accepts at
//! plan time, judging it against the table's schema. The reader then builds its
//! row filter per file, from the predicate rewritten to that file's columns, and
//! a conjunct the per-file form no longer suits is skipped rather than refused.
//! A predicate accepted on the table schema and skipped in a file is lost.
//!
//! The rewrite ([`crate::field_id_adapter`]) keeps a predicate in a form the row
//! filter accepts exactly when [`reader_keeps_predicate`] holds. This node passes
//! every predicate down — the scan still prunes with it — but reports as not
//! pushed each one for which that does not hold, so the `FilterExec` above keeps
//! it and applies it to what the scan returns.

use std::sync::Arc;

use datafusion::common::config::ConfigOptions;
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::{PhysicalExpr, PhysicalSortExpr};
use datafusion::physical_plan::filter_pushdown::{
    ChildFilterDescription, ChildPushdownResult, FilterDescription, FilterPushdownPhase,
    FilterPushdownPropagation, PushedDown,
};
use datafusion::physical_plan::sort_pushdown::SortOrderPushdownResult;
use datafusion::physical_plan::{
    ChildStats, DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, Statistics,
    StatisticsArgs,
};

use crate::field_id_adapter::reader_keeps_predicate;

/// Pass-through node over a scan resolved per file by field id.
#[derive(Debug)]
pub(crate) struct OpenTimeFilterBarrierExec {
    input: Arc<dyn ExecutionPlan>,
    properties: Arc<PlanProperties>,
}

impl OpenTimeFilterBarrierExec {
    pub(crate) fn new(input: Arc<dyn ExecutionPlan>) -> Self {
        let properties = Arc::clone(input.properties());
        Self {
            input,
            properties,
        }
    }
}

impl DisplayAs for OpenTimeFilterBarrierExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "OpenTimeFilterBarrierExec")
    }
}

impl ExecutionPlan for OpenTimeFilterBarrierExec {
    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> DataFusionResult<TreeNodeRecursion>,
    ) -> DataFusionResult<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn name(&self) -> &str {
        "OpenTimeFilterBarrierExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true]
    }

    fn supports_limit_pushdown(&self) -> bool {
        true
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let [child] = <[_; 1]>::try_from(children).map_err(|_| {
            DataFusionError::Internal("OpenTimeFilterBarrierExec expects exactly one child".into())
        })?;
        Ok(Arc::new(Self::new(child)))
    }

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

    fn handle_child_pushdown_result(
        &self,
        _phase: FilterPushdownPhase,
        child_pushdown_result: ChildPushdownResult,
        _config: &ConfigOptions,
    ) -> DataFusionResult<FilterPushdownPropagation<Arc<dyn ExecutionPlan>>> {
        let schema = self.input.schema();
        let filters = child_pushdown_result
            .parent_filters
            .iter()
            .map(|result| match result.all() {
                PushedDown::Yes if reader_keeps_predicate(&result.filter, &schema) => {
                    PushedDown::Yes
                },
                _ => PushedDown::No,
            })
            .collect();
        Ok(FilterPushdownPropagation::with_parent_pushdown_result(
            filters,
        ))
    }

    fn try_pushdown_sort(
        &self,
        order: &[PhysicalSortExpr],
    ) -> DataFusionResult<SortOrderPushdownResult<Arc<dyn ExecutionPlan>>> {
        Ok(self
            .input
            .try_pushdown_sort(order)?
            .map(|inner| Arc::new(Self::new(inner)) as Arc<dyn ExecutionPlan>))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        self.input.execute(partition, context)
    }

    fn child_stats_requests(&self, partition: Option<usize>) -> Vec<ChildStats> {
        vec![ChildStats::At(partition)]
    }

    fn statistics_from_inputs(
        &self,
        input_stats: &[Arc<Statistics>],
        _args: &StatisticsArgs,
    ) -> DataFusionResult<Arc<Statistics>> {
        Ok(Arc::clone(&input_stats[0]))
    }

    /// Kept for callers of the deprecated method, as `NanPruningBarrierExec` does.
    #[allow(deprecated)]
    fn partition_statistics(&self, partition: Option<usize>) -> DataFusionResult<Arc<Statistics>> {
        self.input.partition_statistics(partition)
    }
}
