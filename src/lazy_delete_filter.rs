//! Positional deletes applied to a scan over many data files, each file's delete
//! set read only when the scan first returns rows of that file.
//!
//! Official DuckLake attaches a data file's deletes as its reader opens
//! (`DuckLakeMultiFileReader::InitializeReader` builds a `DuckLakeDeleteFilter`
//! from the file's delete file and inlined deletions). [`LazyDeleteFilterExec`]
//! does the same above a DataFusion parquet scan: the scan emits, beside each row,
//! the row's physical position in its file and the file's catalog id, and this node
//! reads that file's delete set the first time it meets the id, then drops the rows
//! whose position it holds. Planning reads neither the data files nor their delete
//! files, and a scan that stops early reads the delete files of only the data files
//! it reached.
//!
//! The row count this node publishes is the catalog's, as official's
//! `DuckLakeGetPartitionStats` publishes it: the scan's `record_count` total less
//! each file's `delete_count` and its inlined deletions, a figure available
//! without reading any delete file.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow::array::{Array, BooleanArray, Int64Array, RecordBatch, RecordBatchOptions};
use arrow::compute::filter_record_batch;
use arrow::datatypes::SchemaRef;
use datafusion::common::config::ConfigOptions;
use datafusion::common::stats::Precision;
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::execution::object_store::ObjectStoreUrl;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::projection::ProjectionMapping;
use datafusion::physical_expr::{EquivalenceProperties, PhysicalExpr};
use datafusion::physical_plan::execution_plan::{Boundedness, reset_plan_states};
use datafusion::physical_plan::filter_pushdown::{
    ChildFilterDescription, FilterDescription, FilterPushdownPhase,
};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    ChildStats, ColumnStatistics, DisplayAs, DisplayFormatType, ExecutionPlan,
    ExecutionPlanProperties, Partitioning, PlanProperties, Statistics, StatisticsArgs,
};
use futures::TryStreamExt;
use object_store::ObjectStoreExt;
use object_store::path::Path as ObjectPath;
use tokio::sync::OnceCell;

use crate::table::{DELETE_POS_COL, DELETE_SNAPSHOT_COL, is_object_store_not_found};

/// One data file's deletes, as the catalog lists them.
#[derive(Debug)]
pub(crate) struct FileDeletes {
    /// The scan of the file's delete file, built without reading it, and the
    /// delete file's resolved path; `None` when its only deletes are inlined.
    pub(crate) delete_file: Option<(Arc<dyn ExecutionPlan>, String)>,
    /// Positions deleted by inlined deletions.
    pub(crate) inlined: HashSet<i64>,
    /// Rows the catalog counts as deleted from this file: its delete file's
    /// `delete_count` plus its inlined deletions. `None` when the delete file
    /// records no count.
    pub(crate) catalog_deleted: Option<usize>,
    /// The snapshot the scan reads: a delete file that records each deletion's
    /// snapshot applies only the deletions made at or before it.
    pub(crate) read_snapshot: Option<i64>,
}

/// Drops the rows of each data file that its deletes name.
///
/// A file's delete set, once read, is kept until the plan is dropped: a data file
/// can be split into byte ranges read by different partitions, and nothing here
/// learns when the last of them is done. The sets are not reserved against the
/// session's memory pool; they hold one `i64` per deleted row of each data file
/// the scan has reached.
///
/// The input carries, at `file_id_index`, the data file's catalog id — the same
/// on every row of a file — and at `pos_index` the row's physical position in the
/// file. Both are dropped from the output.
#[derive(Debug)]
pub(crate) struct LazyDeleteFilterExec {
    input: Arc<dyn ExecutionPlan>,
    file_id_index: usize,
    pos_index: usize,
    /// Input column indices the output keeps, in order.
    kept: Vec<usize>,
    schema: SchemaRef,
    files: Arc<HashMap<i64, FileDeletes>>,
    loaded: Arc<HashMap<i64, OnceCell<Arc<HashSet<i64>>>>>,
    object_store_url: ObjectStoreUrl,
    /// Per input partition, the data files with deletes in the order the scan
    /// is planned to read them: while the scan reads one file, the delete sets
    /// of the next [`PREFETCH_FILES`] are read. The order is approximate: the
    /// scan's partitions steal files from one another, so a partition may read
    /// a file planned for another, or skip one planned for it. The prefetch
    /// only warms the per-file cache; a file whose delete set is not ready when
    /// its rows arrive has it read then, so any order gives the same rows.
    read_order: Arc<Vec<Vec<i64>>>,
    properties: Arc<PlanProperties>,
}

/// How many data files ahead of the one being read have their delete sets
/// read in the background.
const PREFETCH_FILES: usize = 2;

impl LazyDeleteFilterExec {
    pub(crate) fn try_new(
        input: Arc<dyn ExecutionPlan>,
        file_id_index: usize,
        pos_index: usize,
        files: HashMap<i64, FileDeletes>,
        object_store_url: ObjectStoreUrl,
    ) -> DataFusionResult<Self> {
        let input_schema = input.schema();
        let kept: Vec<usize> = (0..input_schema.fields().len())
            .filter(|index| *index != file_id_index && *index != pos_index)
            .collect();
        let schema = Arc::new(input_schema.project(&kept)?);
        let loaded = files.keys().map(|id| (*id, OnceCell::new())).collect();
        Ok(Self::assemble(
            input,
            file_id_index,
            pos_index,
            kept,
            schema,
            Arc::new(files),
            Arc::new(loaded),
            object_store_url,
            Arc::new(Vec::new()),
        ))
    }

    /// Read ahead of the scan in `read_order`: per input partition, the data
    /// file ids in the order the scan is planned to read them. A cache-warming
    /// hint; see the field's docs.
    pub(crate) fn with_read_order(mut self, read_order: Vec<Vec<i64>>) -> Self {
        self.read_order = Arc::new(read_order);
        self
    }

    #[allow(clippy::too_many_arguments)]
    fn assemble(
        input: Arc<dyn ExecutionPlan>,
        file_id_index: usize,
        pos_index: usize,
        kept: Vec<usize>,
        schema: SchemaRef,
        files: Arc<HashMap<i64, FileDeletes>>,
        loaded: Arc<HashMap<i64, OnceCell<Arc<HashSet<i64>>>>>,
        object_store_url: ObjectStoreUrl,
        read_order: Arc<Vec<Vec<i64>>>,
    ) -> Self {
        // Rows are dropped, never reordered, and the dropped columns trail the
        // kept ones, so the input's orderings over the kept columns hold.
        let mapping = ProjectionMapping::from_indices(&kept, &input.schema()).ok();
        let equivalence = match &mapping {
            Some(mapping) => input
                .equivalence_properties()
                .project(mapping, Arc::clone(&schema)),
            None => EquivalenceProperties::new(Arc::clone(&schema)),
        };
        // The input's partitioning, restated over the kept columns: a hash
        // partitioning on a dropped column no longer describes the output.
        let partitioning = match &mapping {
            Some(mapping) => input
                .output_partitioning()
                .project(mapping, input.equivalence_properties()),
            None => {
                Partitioning::UnknownPartitioning(input.output_partitioning().partition_count())
            },
        };
        let properties = Arc::new(PlanProperties::new(
            equivalence,
            partitioning,
            input.pipeline_behavior(),
            Boundedness::Bounded,
        ));
        Self {
            input,
            file_id_index,
            pos_index,
            kept,
            schema,
            files,
            loaded,
            object_store_url,
            read_order,
            properties,
        }
    }
}

impl DisplayAs for LazyDeleteFilterExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "LazyDeleteFilterExec: files={}", self.files.len())
    }
}

impl ExecutionPlan for LazyDeleteFilterExec {
    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> DataFusionResult<TreeNodeRecursion>,
    ) -> DataFusionResult<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn name(&self) -> &str {
        "LazyDeleteFilterExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    /// Only when the kept columns are the input's leading ones, at the same
    /// positions: an ordering requirement the optimizer pushes through is
    /// mapped by column index.
    fn maintains_input_order(&self) -> Vec<bool> {
        vec![
            self.kept
                .iter()
                .enumerate()
                .all(|(index, kept)| index == *kept),
        ]
    }

    /// No repartition below this node. A round-robin repartition under it would
    /// drain the scan eagerly — every output partition wants a batch — so a
    /// `LIMIT` above could not stop the scan opening file after file. The scan
    /// under it is split into file groups where that pays (see
    /// `DuckLakeTable::split_scan`), which is what parallelizes the reading.
    ///
    /// This keeps a repartition from below this node only. A `FilterExec` above
    /// it still asks for one, so a filtered `LIMIT` whose filter stays above the
    /// scan (`pushdown_filters` off) runs over a round-robin repartition that
    /// reads every file, as it does over any DataFusion parquet scan of one file
    /// group. With `pushdown_filters` on, the filter moves into the reader, no
    /// `FilterExec` remains, and a filtered `LIMIT` stops as early as an
    /// unfiltered one.
    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![false]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let [child] = <[_; 1]>::try_from(children).map_err(|_| {
            DataFusionError::Internal("LazyDeleteFilterExec expects exactly one child".into())
        })?;
        Ok(Arc::new(Self::assemble(
            child,
            self.file_id_index,
            self.pos_index,
            self.kept.clone(),
            Arc::clone(&self.schema),
            Arc::clone(&self.files),
            Arc::clone(&self.loaded),
            self.object_store_url.clone(),
            Arc::clone(&self.read_order),
        )))
    }

    /// Forwards every filter: deletion is keyed by a row's physical position,
    /// which the reader derives from the file, so a filter evaluated below this
    /// node removes the same rows as one evaluated above it.
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
        let input = self.input.execute(partition, Arc::clone(&context))?;
        let filter = Arc::new(BatchFilter {
            file_id_index: self.file_id_index,
            pos_index: self.pos_index,
            kept: self.kept.clone(),
            schema: Arc::clone(&self.schema),
            files: Arc::clone(&self.files),
            loaded: Arc::clone(&self.loaded),
            object_store_url: self.object_store_url.clone(),
            read_order: self.read_order.get(partition).cloned().unwrap_or_default(),
        });
        // Owned by the stream alone, so dropping the stream — a `LIMIT` above
        // having what it needs — aborts the reads started ahead.
        let prefetching = Arc::new(std::sync::Mutex::new(PrefetchState::default()));
        let stream = input.and_then(move |batch| {
            let filter = Arc::clone(&filter);
            let context = Arc::clone(&context);
            let prefetching = Arc::clone(&prefetching);
            async move { filter.apply(batch, &context, &prefetching).await }
        });
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&self.schema),
            stream,
        )))
    }

    fn child_stats_requests(&self, partition: Option<usize>) -> Vec<ChildStats> {
        vec![ChildStats::At(partition)]
    }

    /// The input's row count less the catalog's count of deleted rows, exact
    /// when both are. A single output partition's count is inexact: the catalog
    /// counts a file's deletes, not a partition's.
    fn statistics_from_inputs(
        &self,
        input_stats: &[Arc<Statistics>],
        args: &StatisticsArgs,
    ) -> DataFusionResult<Arc<Statistics>> {
        let Some(input) = input_stats.first() else {
            return Ok(Arc::new(Statistics::new_unknown(&self.schema)));
        };
        let deleted = self
            .files
            .values()
            .map(|file| file.catalog_deleted)
            .sum::<Option<usize>>();
        let num_rows = match (input.num_rows, deleted, args.partition()) {
            (Precision::Exact(rows), Some(deleted), None) => rows
                .checked_sub(deleted)
                .map(Precision::Exact)
                .unwrap_or(Precision::Absent),
            (rows, _, _) => rows.to_inexact(),
        };
        Ok(Arc::new(Statistics {
            num_rows,
            total_byte_size: input.total_byte_size.to_inexact(),
            column_statistics: vec![ColumnStatistics::new_unknown(); self.schema.fields().len()],
        }))
    }
}

/// The per-batch work of [`LazyDeleteFilterExec`], shared by its partitions.
struct BatchFilter {
    file_id_index: usize,
    pos_index: usize,
    kept: Vec<usize>,
    schema: SchemaRef,
    files: Arc<HashMap<i64, FileDeletes>>,
    loaded: Arc<HashMap<i64, OnceCell<Arc<HashSet<i64>>>>>,
    object_store_url: ObjectStoreUrl,
    /// This partition's data files with deletes, in the scan's read order.
    read_order: Vec<i64>,
}

/// Delete-set reads a partition's stream has started ahead of the scan.
#[derive(Default)]
struct PrefetchState {
    tasks: tokio::task::JoinSet<()>,
    /// Index in `read_order` up to which reads have been started.
    started: usize,
}

impl BatchFilter {
    /// Start reading, in the background, the delete sets of the files that
    /// follow `file_id` in this partition's read order. A failed read is
    /// dropped here and repeated, with its error, when the file is reached.
    fn prefetch_after(
        self: &Arc<Self>,
        file_id: i64,
        context: &Arc<TaskContext>,
        prefetching: &std::sync::Mutex<PrefetchState>,
    ) {
        let Some(position) = self.read_order.iter().position(|id| *id == file_id) else {
            return;
        };
        let end = (position + 1 + PREFETCH_FILES).min(self.read_order.len());
        let mut state = prefetching.lock().unwrap();
        let start = state.started.max(position + 1);
        for &next in &self.read_order[start.min(end)..end] {
            let filter = Arc::clone(self);
            let context = Arc::clone(context);
            state.tasks.spawn(async move {
                let _ = filter.deleted(next, &context).await;
            });
        }
        state.started = state.started.max(end);
    }

    async fn apply(
        self: &Arc<Self>,
        batch: RecordBatch,
        context: &Arc<TaskContext>,
        prefetching: &std::sync::Mutex<PrefetchState>,
    ) -> DataFusionResult<RecordBatch> {
        let file_ids = int64_column(&batch, self.file_id_index, "data file id")?;
        let positions = int64_column(&batch, self.pos_index, "row position")?;
        let mut keep = Vec::with_capacity(batch.num_rows());
        let mut current: Option<(i64, Arc<HashSet<i64>>)> = None;
        for row in 0..batch.num_rows() {
            let file_id = file_ids.value(row);
            let deleted = match &current {
                Some((id, deleted)) if *id == file_id => Arc::clone(deleted),
                _ => {
                    self.prefetch_after(file_id, context, prefetching);
                    let deleted = self.deleted(file_id, context).await?;
                    current = Some((file_id, Arc::clone(&deleted)));
                    deleted
                },
            };
            keep.push(!deleted.contains(&positions.value(row)));
        }
        let filtered = filter_record_batch(&batch, &BooleanArray::from(keep))?;
        let columns = self
            .kept
            .iter()
            .map(|index| Arc::clone(filtered.column(*index)))
            .collect();
        Ok(RecordBatch::try_new_with_options(
            Arc::clone(&self.schema),
            columns,
            &RecordBatchOptions::new().with_row_count(Some(filtered.num_rows())),
        )?)
    }

    /// The positions deleted from data file `file_id`, read once per scan.
    async fn deleted(
        &self,
        file_id: i64,
        context: &Arc<TaskContext>,
    ) -> DataFusionResult<Arc<HashSet<i64>>> {
        let (Some(file), Some(cell)) = (self.files.get(&file_id), self.loaded.get(&file_id)) else {
            return Ok(Arc::new(HashSet::new()));
        };
        cell.get_or_try_init(|| async {
            let mut deleted = file.inlined.clone();
            if let Some((scan, path)) = &file.delete_file {
                deleted.extend(
                    read_delete_positions(
                        scan,
                        context,
                        &self.object_store_url,
                        path,
                        file.read_snapshot,
                    )
                    .await?,
                );
            }
            Ok::<_, DataFusionError>(Arc::new(deleted))
        })
        .await
        .cloned()
    }
}

fn int64_column<'a>(
    batch: &'a RecordBatch,
    index: usize,
    what: &str,
) -> DataFusionResult<&'a Int64Array> {
    batch
        .column(index)
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| DataFusionError::Internal(format!("the scan's {what} column is not Int64")))
}

/// Whether a pass-through node over `input` gains from DataFusion repartitioning
/// what sits under it: yes over a leaf scan, which DataFusion may split into
/// file groups, and no over a node that itself declines it (see
/// [`LazyDeleteFilterExec`]), so a round-robin repartition is placed above the
/// whole per-row stack rather than wedged inside it.
pub(crate) fn benefits_through(input: &Arc<dyn ExecutionPlan>) -> Vec<bool> {
    vec![input.children().is_empty() || input.benefits_from_input_partitioning().iter().all(|b| *b)]
}

/// Run the scan of one delete file and collect the positions it deletes (its
/// `pos` column). With `Some(read_snapshot)`, a delete file that records each
/// deletion's snapshot (`_ducklake_internal_snapshot_id`, which official DuckLake
/// writes when a file with committed deletes is deleted from again) contributes
/// only the deletions made at or before it, as official's
/// `DuckLakeDeleteFilter::SetSnapshotFilter` applies them to a read. With
/// `None`, every position is taken — what a write that supersedes the file
/// needs, since its replacement must carry deletions made after the writer's
/// snapshot too. A file that records no snapshots contributes all of them
/// either way. A failed read of a file that is absent from the
/// object store is reported as a missing delete file.
///
/// Each call executes its own copy of `scan` with fresh execution state. A
/// `DataSourceExec` keeps the work queue its partitions share for the life of
/// the plan, so a plan that has been executed once yields nothing when
/// executed again: re-running the stored plan after a failed read, or from a
/// second partition racing the same file, would read an empty delete set.
pub(crate) async fn read_delete_positions(
    scan: &Arc<dyn ExecutionPlan>,
    context: &Arc<TaskContext>,
    object_store_url: &ObjectStoreUrl,
    resolved_path: &str,
    read_snapshot: Option<i64>,
) -> DataFusionResult<HashSet<i64>> {
    let scan = reset_plan_states(Arc::clone(scan))?;
    let batches = match scan.execute(0, Arc::clone(context)) {
        Ok(stream) => stream.try_collect::<Vec<_>>().await,
        Err(error) => Err(error),
    };
    let batches = match batches {
        Ok(batches) => batches,
        Err(error) => {
            return Err(classify_delete_file_read_error(
                context,
                object_store_url,
                resolved_path,
                error,
            )
            .await);
        },
    };
    let mut positions = HashSet::new();
    for batch in &batches {
        let pos_index = batch.schema().index_of(DELETE_POS_COL)?;
        let pos = int64_column(batch, pos_index, DELETE_POS_COL)?;
        let snapshots = match batch.schema().index_of(DELETE_SNAPSHOT_COL) {
            Ok(index) => Some(int64_column(batch, index, DELETE_SNAPSHOT_COL)?),
            Err(_) => None,
        };
        for row in 0..batch.num_rows() {
            let applies = match (read_snapshot, snapshots) {
                (Some(read_snapshot), Some(snapshots)) => {
                    snapshots.is_null(row) || snapshots.value(row) <= read_snapshot
                },
                _ => true,
            };
            if applies && !pos.is_null(row) {
                positions.insert(pos.value(row));
            }
        }
    }
    Ok(positions)
}

/// Turn a failed delete-file read into a caller-facing error, replacing the raw
/// parquet failure with a "delete file is missing" message when the file really
/// is absent from the object store.
///
/// The absence cannot be read off the error alone: DataFusion's parquet reader
/// flattens the metadata-fetch failure into a `ParquetError::General` string, so
/// the underlying `object_store::Error::NotFound` is no longer in the source
/// chain. Probing the store is only done once a read has already failed, so the
/// happy path pays nothing.
pub(crate) async fn classify_delete_file_read_error(
    context: &TaskContext,
    object_store_url: &ObjectStoreUrl,
    resolved_path: &str,
    error: DataFusionError,
) -> DataFusionError {
    let missing = if is_object_store_not_found(&error) {
        true
    } else {
        match context.runtime_env().object_store(object_store_url) {
            Ok(store) => matches!(
                store.head(&ObjectPath::from(resolved_path)).await,
                Err(object_store::Error::NotFound { .. })
            ),
            Err(_) => false,
        }
    };
    if missing {
        DataFusionError::Execution(format!(
            "Delete file '{resolved_path}' referenced in catalog metadata was not found. This may indicate catalog corruption or that the file was deleted outside of DuckLake."
        ))
    } else {
        error
    }
}
