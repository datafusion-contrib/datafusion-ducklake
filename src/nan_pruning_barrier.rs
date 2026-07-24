//! Filter-pushdown barrier for NaN-unsafe float columns
//!
//! Parquet footer min/max for float columns exclude NaN, and NaN sorts above
//! every value in DataFusion (IEEE 754 totalOrder, matching DuckDB). When a
//! file's NaN state is unknown or positive, a predicate like `x > C` pushed
//! into the parquet scan can row-group-prune a group whose footer max is below
//! `C` while it still holds matching NaN rows — silently dropping them. The
//! catalog-level gate (see `float_max_is_bound` in `table.rs`) protects only
//! plan-time file pruning; this node protects execution-time row-group/page/
//! bloom pruning by refusing to push such predicates into the scan.
//!
//! Purely a pass-through at execution: it forwards batches unchanged and only
//! participates in the filter-pushdown negotiation, rejecting predicates that
//! reference a NaN-unsafe float column. Rejected predicates stay in the
//! `FilterExec` above (all table filters are declared `Inexact`), so results
//! remain correct — the scan just reads more row groups for those predicates.

use std::collections::HashSet;
use std::sync::Arc;

use datafusion::common::config::ConfigOptions;
use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_expr::{PhysicalExpr, PhysicalSortExpr};
use datafusion::physical_plan::filter_pushdown::{
    ChildFilterDescription, FilterDescription, FilterPushdownPhase,
};
use datafusion::physical_plan::sort_pushdown::SortOrderPushdownResult;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, Statistics,
};

/// Pass-through node that blocks filter pushdown for predicates referencing
/// float columns whose NaN state is not known to be false for every file
/// beneath it.
#[derive(Debug)]
pub struct NanPruningBarrierExec {
    input: Arc<dyn ExecutionPlan>,
    /// Names (in this node's schema) of float columns whose NaN state is
    /// unknown or positive for at least one file under this scan.
    unsafe_columns: Arc<HashSet<String>>,
    /// Child's properties, reused verbatim (the schema is unchanged).
    properties: Arc<PlanProperties>,
}

impl NanPruningBarrierExec {
    pub fn new(input: Arc<dyn ExecutionPlan>, unsafe_columns: Arc<HashSet<String>>) -> Self {
        let properties = Arc::clone(input.properties());
        Self {
            input,
            unsafe_columns,
            properties,
        }
    }
}

impl DisplayAs for NanPruningBarrierExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut columns: Vec<&str> = self.unsafe_columns.iter().map(String::as_str).collect();
        columns.sort_unstable();
        write!(
            f,
            "NanPruningBarrierExec: unsafe_columns=[{}]",
            columns.join(", ")
        )
    }
}

impl ExecutionPlan for NanPruningBarrierExec {
    fn name(&self) -> &str {
        "NanPruningBarrierExec"
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
        if children.len() != 1 {
            return Err(DataFusionError::Internal(
                "NanPruningBarrierExec expects exactly one child".into(),
            ));
        }
        Ok(Arc::new(NanPruningBarrierExec::new(
            Arc::clone(&children[0]),
            Arc::clone(&self.unsafe_columns),
        )))
    }

    fn gather_filters_for_pushdown(
        &self,
        _phase: FilterPushdownPhase,
        parent_filters: Vec<Arc<dyn PhysicalExpr>>,
        _config: &ConfigOptions,
    ) -> DataFusionResult<FilterDescription> {
        // Allow a predicate through only when every column it references is
        // NaN-safe; the rest are reported unsupported and stay above us.
        let allowed: HashSet<usize> = self
            .input
            .schema()
            .fields()
            .iter()
            .enumerate()
            .filter(|(_, field)| !self.unsafe_columns.contains(field.name()))
            .map(|(index, _)| index)
            .collect();
        let child = ChildFilterDescription::from_child_with_allowed_indices(
            &parent_filters,
            allowed,
            &self.input,
        )?;
        Ok(FilterDescription::new().with_child(child))
    }

    fn try_pushdown_sort(
        &self,
        order: &[PhysicalSortExpr],
    ) -> DataFusionResult<SortOrderPushdownResult<Arc<dyn ExecutionPlan>>> {
        // Sort pushdown can reorder files/row groups from their statistics —
        // the same NaN-blind stats the filter barrier exists for — so only
        // delegate when no sort key touches an unsafe float column.
        let references_unsafe = order.iter().any(|sort| {
            let mut found = false;
            sort.expr
                .apply(|expr| {
                    if let Some(column) = expr.downcast_ref::<Column>()
                        && self.unsafe_columns.contains(column.name())
                    {
                        found = true;
                        return Ok(TreeNodeRecursion::Stop);
                    }
                    Ok(TreeNodeRecursion::Continue)
                })
                .expect("column scan over a sort expression cannot fail");
            found
        });
        if references_unsafe {
            return Ok(SortOrderPushdownResult::Unsupported);
        }
        Ok(self.input.try_pushdown_sort(order)?.map(|inner| {
            Arc::new(NanPruningBarrierExec::new(
                inner,
                Arc::clone(&self.unsafe_columns),
            )) as Arc<dyn ExecutionPlan>
        }))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        self.input.execute(partition, context)
    }

    fn partition_statistics(&self, partition: Option<usize>) -> DataFusionResult<Arc<Statistics>> {
        self.input.partition_statistics(partition)
    }
}
