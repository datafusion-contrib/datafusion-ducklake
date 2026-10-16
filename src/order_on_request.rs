//! Declare a DuckLake scan's sort order only when a sort asks for it.
//!
//! [`OrderOnRequestExec`] wraps a scan planned exactly as without verified
//! orders: same file groups, byte-range splits, everything. It changes nothing
//! for a query that does not sort by the table's key. When DataFusion's
//! `PushdownSort` rule asks it to satisfy an ordering (`try_pushdown_sort`), it
//! derives an ordered plan *from its current input* — so whatever filter,
//! projection and limit the optimizer already pushed into the scans is kept —
//! by giving the scan over the verified files the statistics-proven
//! non-overlapping chains as file groups plus `output_ordering`, and by sorting
//! the in-memory (inlined) rows and declaring that order. It answers `Exact`
//! only when that plan's equivalence properties satisfy the request;
//! otherwise it defers to its input's own answer (DataFusion's statistics
//! based Inexact pushdown), i.e. today's behaviour. On `Exact`, `PushdownSort`
//! removes the sort, pushes the fetch into the scan and adds its own buffer
//! (`sort_pushdown_buffer_capacity`) between the partitions and the merge.
//!
//! Eligibility is structural, checked on the plan it is given: every
//! file-backed scan under it must read exactly the verified files (a scan of
//! any other file — unverified, with deletes, name-mapped — makes it decline),
//! every in-memory source is sorted, and nothing under it may carry a limit
//! already (an inner `LIMIT` pushed into the scan bounds the rows; the
//! ordered plan would turn it into a per-chain limit). The final check against the
//! requested ordering makes a wrapper above (one that loses orderings) a
//! decline, never a wrong answer.

use std::collections::HashSet;
use std::fmt;
use std::sync::Arc;

use datafusion::common::Result;
use datafusion::common::config::ConfigOptions;
use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::datasource::memory::MemorySourceConfig;
use datafusion::datasource::physical_plan::{FileGroup, FileScanConfig};
use datafusion::datasource::source::DataSourceExec;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_expr::{LexOrdering, PhysicalExpr, PhysicalSortExpr};
use datafusion::physical_plan::execution_plan::CardinalityEffect;
use datafusion::physical_plan::filter_pushdown::{
    ChildPushdownResult, FilterDescription, FilterPushdownPhase, FilterPushdownPropagation,
};
use datafusion::physical_plan::metrics::MetricsSet;
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec;
use datafusion::physical_plan::statistics::{ChildStats, StatisticsArgs};
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, PlanProperties,
    SortOrderPushdownResult,
};

/// See the module docs.
#[derive(Debug)]
pub struct OrderOnRequestExec {
    input: Arc<dyn ExecutionPlan>,
    /// The verified files, as non-overlapping chains in `ordering`.
    chains: Arc<Vec<FileGroup>>,
    /// Their paths, to recognise the scan that reads them.
    paths: Arc<HashSet<String>>,
    /// The order the chains and the sorted inlined rows follow, over the
    /// table schema.
    ordering: LexOrdering,
    /// Read-ahead per chain when merging (the session's
    /// `sort_pushdown_buffer_capacity`), see [`MergedChainsExec`].
    buffer_capacity: usize,
    /// The session's file-scan split settings, `None` when
    /// `repartition_file_scans` is off: `(target_partitions, min_size)`.
    split: Option<(usize, usize)>,
}

impl OrderOnRequestExec {
    pub(crate) fn new(
        input: Arc<dyn ExecutionPlan>,
        chains: Vec<FileGroup>,
        ordering: LexOrdering,
        buffer_capacity: usize,
        split: Option<(usize, usize)>,
    ) -> Self {
        let paths = chains
            .iter()
            .flat_map(|group| group.iter())
            .map(|file| file.object_meta.location.to_string())
            .collect();
        Self {
            input,
            chains: Arc::new(chains),
            paths: Arc::new(paths),
            ordering,
            buffer_capacity,
            split,
        }
    }

    fn with_input(&self, input: Arc<dyn ExecutionPlan>) -> Self {
        Self {
            input,
            chains: Arc::clone(&self.chains),
            paths: Arc::clone(&self.paths),
            ordering: self.ordering.clone(),
            buffer_capacity: self.buffer_capacity,
            split: self.split,
        }
    }

    /// The ordered form of the input, or `None` when it is not eligible.
    fn ordered_plan(&self) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        let mut replaced = 0usize;
        let mut eligible = true;
        let plan = Arc::clone(&self.input).transform_up(|node| {
            // A limit already under this node (an inner `LIMIT` that
            // `LimitPushdown` moved into the scan before `PushdownSort` asked)
            // bounds the input's rows. The ordered plan would rebuild the scan
            // as several chains, where that limit holds per chain, and
            // `PushdownSort` then overwrites it with the outer fetch: decline.
            if node.fetch().is_some() {
                eligible = false;
                return Ok(Transformed::no(node));
            }
            let Some(scan) = node.downcast_ref::<DataSourceExec>() else {
                return Ok(Transformed::no(node));
            };
            let source = scan.data_source();
            if let Some(config) = source.as_ref().downcast_ref::<FileScanConfig>() {
                let paths: HashSet<String> = config
                    .file_groups
                    .iter()
                    .flat_map(|group| group.iter())
                    .map(|file| file.object_meta.location.to_string())
                    .collect();
                // Exactly the verified files (byte-range splits of them
                // included); anything else is a scan this wrapper cannot order.
                if paths != *self.paths {
                    eligible = false;
                    return Ok(Transformed::no(node));
                }
                replaced += 1;
                let mut ordered = config.clone();
                ordered.file_groups = self.chains.as_ref().clone();
                ordered.output_ordering = vec![self.ordering.clone()];
                // Fewer chains than partitions (a short window: few files)
                // would read on fewer threads than today's byte-range split.
                // DataFusion's order-preserving repartitioner splits
                // single-file chains into contiguous ranges, one partition
                // each, each still ordered; the merge above restores one order.
                if let Some((target, min_size)) = self.split
                    && let Some(split) = ordered.file_source().repartitioned(
                        target,
                        min_size,
                        Some(self.ordering.clone()),
                        &ordered,
                    )?
                {
                    ordered = split;
                }
                return Ok(Transformed::yes(
                    DataSourceExec::from_data_source(ordered) as Arc<dyn ExecutionPlan>
                ));
            }
            if let Some(memory) = source.as_ref().downcast_ref::<MemorySourceConfig>() {
                // Inlined rows: no file, never verified; sorted here (they are
                // bounded by the inlining limit) and declared as sorted.
                let schema = memory.original_schema();
                let batches: Vec<_> = memory.partitions().iter().flatten().cloned().collect();
                let batch = arrow::compute::concat_batches(&schema, &batches)?;
                let sort_columns = self
                    .ordering
                    .iter()
                    .map(|expr| {
                        let column = expr.expr.downcast_ref::<Column>().ok_or_else(|| {
                            datafusion::common::DataFusionError::Internal(
                                "ordered alternative: non-column sort key".to_string(),
                            )
                        })?;
                        Ok(arrow::compute::SortColumn {
                            values: Arc::clone(batch.column(column.index())),
                            options: Some(expr.options),
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                let indices = arrow::compute::lexsort_to_indices(&sort_columns, None)?;
                let sorted = arrow::compute::take_record_batch(&batch, &indices)?;
                let config = MemorySourceConfig::try_new(
                    &[vec![sorted]],
                    schema,
                    memory.projection().clone(),
                )?
                .try_with_sort_information(vec![self.ordering.clone()])?;
                return Ok(Transformed::yes(
                    DataSourceExec::from_data_source(config) as Arc<dyn ExecutionPlan>
                ));
            }
            eligible = false;
            Ok(Transformed::no(node))
        })?;
        Ok((eligible && replaced == 1).then_some(plan.data))
    }
}

impl DisplayAs for OrderOnRequestExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "OrderOnRequestExec: ordering=[{}], verified_files={}",
            self.ordering,
            self.paths.len()
        )
    }
}

impl ExecutionPlan for OrderOnRequestExec {
    fn name(&self) -> &str {
        "OrderOnRequestExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        self.input.properties()
    }

    // `false` keeps `EnforceSorting` from pushing a sort below this node onto
    // the scan, which would hide it from `try_pushdown_sort` here. The input
    // is unordered (the order is only ever offered on request), so there is
    // nothing to maintain.
    fn maintains_input_order(&self) -> Vec<bool> {
        vec![false]
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![false]
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(self.with_input(children.swap_remove(0))))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        self.input.execute(partition, context)
    }

    fn metrics(&self) -> Option<MetricsSet> {
        None
    }

    fn child_stats_requests(&self, partition: Option<usize>) -> Vec<ChildStats> {
        vec![ChildStats::At(partition)]
    }

    fn statistics_from_inputs(
        &self,
        input_stats: &[Arc<datafusion::common::Statistics>],
        _args: &StatisticsArgs,
    ) -> Result<Arc<datafusion::common::Statistics>> {
        Ok(Arc::clone(&input_stats[0]))
    }

    // Transparent to a limit with no sort above: it reaches the scan as today.
    fn supports_limit_pushdown(&self) -> bool {
        true
    }

    fn cardinality_effect(&self) -> CardinalityEffect {
        CardinalityEffect::Equal
    }

    fn try_swapping_with_projection(
        &self,
        projection: &ProjectionExec,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        Ok(self
            .input
            .try_swapping_with_projection(projection)?
            .map(|input| Arc::new(self.with_input(input)) as Arc<dyn ExecutionPlan>))
    }

    fn gather_filters_for_pushdown(
        &self,
        _phase: FilterPushdownPhase,
        parent_filters: Vec<Arc<dyn PhysicalExpr>>,
        _config: &ConfigOptions,
    ) -> Result<FilterDescription> {
        FilterDescription::from_children(parent_filters, &self.children())
    }

    fn handle_child_pushdown_result(
        &self,
        _phase: FilterPushdownPhase,
        child_pushdown_result: ChildPushdownResult,
        _config: &ConfigOptions,
    ) -> Result<FilterPushdownPropagation<Arc<dyn ExecutionPlan>>> {
        Ok(FilterPushdownPropagation::if_all(child_pushdown_result))
    }

    fn try_pushdown_sort(
        &self,
        order: &[PhysicalSortExpr],
    ) -> Result<SortOrderPushdownResult<Arc<dyn ExecutionPlan>>> {
        if let Some(ordered) = self.ordered_plan()?
            && ordered
                .equivalence_properties()
                .ordering_satisfy(order.iter().cloned())?
        {
            // `PushdownSort` puts what this returns where the sort was. A sort
            // with no merge above it (one input partition, or one target
            // partition over a union) needs a single ordered stream, and which
            // shape this is cannot be seen from here: always return one.
            let ordered = if ordered.output_partitioning().partition_count() > 1 {
                let order = LexOrdering::new(order.iter().cloned()).ok_or_else(|| {
                    datafusion::common::DataFusionError::Internal(
                        "ordered alternative: empty sort request".to_string(),
                    )
                })?;
                Arc::new(MergedChainsExec::new(
                    ordered,
                    order,
                    self.buffer_capacity,
                    None,
                )) as Arc<dyn ExecutionPlan>
            } else {
                ordered
            };
            return Ok(SortOrderPushdownResult::Exact {
                inner: ordered,
            });
        }
        // Not ours to serve: exactly what the input would answer without us
        // (DataFusion's statistics-based Inexact pushdown, for instance).
        self.input.try_pushdown_sort(order)
    }
}

/// The ordered chains merged into one stream: a `SortPreservingMergeExec`
/// over a DataFusion `BufferExec` per chain, holding a fetch it pushes into
/// the chains.
///
/// Both halves are load-bearing (lab, 7 partitions): without the buffer the
/// merge pulls the chains in lockstep, one batch deep each, and a rare filter
/// that must read every file runs 2-3x slower; a bare `BufferExec` stops
/// `LimitPushdown`/`PushdownSort` from reaching the scans, and an early-stop
/// query reads ahead into the buffer instead of stopping (10x slower). So
/// [`ExecutionPlan::with_fetch`] here rebuilds the plan with the fetch both in
/// each chain's scan and on the merge.
#[derive(Debug)]
pub struct MergedChainsExec {
    chains: Arc<dyn ExecutionPlan>,
    order: LexOrdering,
    capacity: usize,
    fetch: Option<usize>,
    merge: Arc<dyn ExecutionPlan>,
}

impl MergedChainsExec {
    fn new(
        chains: Arc<dyn ExecutionPlan>,
        order: LexOrdering,
        capacity: usize,
        fetch: Option<usize>,
    ) -> Self {
        let limited = match fetch {
            Some(fetch) => push_fetch(&chains, fetch),
            None => Arc::clone(&chains),
        };
        let limited_chains = Arc::clone(&limited);
        let buffered: Arc<dyn ExecutionPlan> = if capacity > 0 {
            Arc::new(datafusion::physical_plan::buffer::BufferExec::new(
                limited, capacity,
            ))
        } else {
            limited
        };
        let merge =
            Arc::new(SortPreservingMergeExec::new(order.clone(), buffered).with_fetch(fetch));
        Self {
            // The limited chains: what runs, and what `EXPLAIN` shows.
            chains: limited_chains,
            order,
            capacity,
            fetch,
            merge,
        }
    }
}

impl DisplayAs for MergedChainsExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "MergedChainsExec: capacity={}", self.capacity)?;
        if let Some(fetch) = self.fetch {
            write!(f, ", fetch={fetch}")?;
        }
        Ok(())
    }
}

impl ExecutionPlan for MergedChainsExec {
    fn name(&self) -> &str {
        "MergedChainsExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        self.merge.properties()
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true]
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![false]
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.chains]
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(Self::new(
            children.swap_remove(0),
            self.order.clone(),
            self.capacity,
            self.fetch,
        )))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        self.merge.execute(partition, context)
    }

    fn metrics(&self) -> Option<MetricsSet> {
        self.merge.metrics()
    }

    fn fetch(&self) -> Option<usize> {
        self.fetch
    }

    fn with_fetch(&self, limit: Option<usize>) -> Option<Arc<dyn ExecutionPlan>> {
        Some(Arc::new(Self::new(
            Arc::clone(&self.chains),
            self.order.clone(),
            self.capacity,
            limit,
        )))
    }

    fn cardinality_effect(&self) -> CardinalityEffect {
        CardinalityEffect::LowerEqual
    }
}

/// `plan` limited to `fetch` rows per partition, the limit as close to the
/// scans as it goes: into a scan that takes it, through a union into each
/// input (so the parquet scan of a files + inlined-rows union still stops
/// early), and otherwise as a `LocalLimitExec`.
fn push_fetch(plan: &Arc<dyn ExecutionPlan>, fetch: usize) -> Arc<dyn ExecutionPlan> {
    if let Some(limited) = plan.with_fetch(Some(fetch)) {
        return limited;
    }
    if plan
        .downcast_ref::<datafusion::physical_plan::union::UnionExec>()
        .is_some()
    {
        let children: Vec<_> = plan
            .children()
            .into_iter()
            .map(|c| push_fetch(c, fetch))
            .collect();
        if let Ok(union) =
            datafusion::physical_plan::replace_children_if_necessary(Arc::clone(plan), children)
        {
            return union;
        }
    }
    Arc::new(datafusion::physical_plan::limit::LocalLimitExec::new(
        Arc::clone(plan),
        fetch,
    ))
}
