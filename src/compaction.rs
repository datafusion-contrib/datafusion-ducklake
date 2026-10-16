//! Explicit, triggered DuckLake compaction for a single table.
//!
//! Two maintenance operations, each invoked programmatically (never
//! automatically on write) and returning a [`CompactionResult`] with metrics:
//!
//! 1. [`DuckLakeTable::merge_adjacent_files`] coalesces several small data files
//!    of one table (of the SAME schema version — never across a DDL boundary)
//!    into fewer larger ones. A merged file keeps each row's original rowid:
//!    when the sources' rowid ranges are adjacent it writes the rows in rowid
//!    order and keeps the first source's `row_id_start`, otherwise it embeds
//!    each rowid as a column. A merged file that spans more than one origin
//!    snapshot is written as a DuckLake **partial data file**: it embeds a
//!    per-row `_ducklake_internal_snapshot_id` column, and its catalog row
//!    records `partial_max` (the maximum origin snapshot id among its rows), so
//!    time travel / change feeds can still attribute every merged row to its
//!    origin snapshot. A partial file is itself a merge
//!    candidate: its scan projects that embedded column, so each row keeps its
//!    own origin through a re-merge and a partition never accumulates a floor of
//!    files it can no longer reduce.
//! 2. [`DuckLakeTable::rewrite_data_files`] rewrites a data file whose deleted
//!    fraction exceeds a threshold (DuckDB's default is 0.95): it reads only the
//!    file's LIVE rows (delete-aware), writes them to a new file preserving each
//!    row's rowid, and retires BOTH the old data file and its delete file.
//!
//! Both operations commit ATOMICALLY in one snapshot via
//! `MetadataWriter::commit_compaction`: the rewritten outputs are registered, the
//! source files (and, for a rewrite, their delete files) are retired
//! (`end_snapshot` set) and scheduled for physical deletion, and
//! `ducklake_snapshot_changes.changes_made` records `merge_adjacent:<table_id>` or
//! `rewrite_delete:<table_id>`, as official's compaction does.
//! Compaction changes the physical layout, not the logical rows, so the commit is
//! structured NOT to conflict with a concurrent append; it aborts only if a
//! source file was retired, or its live rows changed, since it was read (the
//! `base_snapshot` conflict check), which prevents ever resurrecting a
//! retired/deleted row into an output.
//!
//! Retired files are only SCHEDULED for deletion, never removed here, so time
//! travel to a pre-compaction snapshot still reads them until
//! [`cleanup_old_files_sqlite`](crate::maintenance::cleanup_old_files_sqlite)
//! reclaims them.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicI64, Ordering};

use arrow::array::{Array, ArrayRef, Int64Array, RecordBatch};
use arrow::compute::SortOptions;
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::catalog::Session;
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::datasource::memory::MemorySourceConfig;
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_expr::{
    EquivalenceProperties, LexOrdering, PhysicalSortExpr, expressions::Column,
};
use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion::physical_plan::execution_plan::reset_plan_states;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::union::UnionExec;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, PlanProperties,
    sorts::sort::SortExec, sorts::sort_preserving_merge::SortPreservingMergeExec,
};
use futures::{StreamExt, TryStreamExt};

use crate::column_rename::ColumnRenameExec;
use crate::metadata_provider::DuckLakeTableFile;
use crate::metadata_writer::{
    CompactionOutputFile, CompactionSourceFile, RowIdStart, SourceRetirement,
};
use crate::partition::PartitionSpec;
use crate::row_id::EMBEDDED_SNAPSHOT_ID_COLUMN_NAME;
use crate::sort::SortSpec;
use crate::table::{
    DuckLakeTable, MergeSourceFacts, RewrittenBatch, UpdateSourceScan, rewrite_output_schema,
    rewrite_scanned_batch,
};
use crate::table_writer::DuckLakeTableWriter;
use crate::{DuckLakeError, Result};

/// Options for [`DuckLakeTable::merge_adjacent_files`].
#[derive(Debug, Clone)]
pub struct MergeOptions {
    /// Bin-pack adjacent small files (in `(schema_version, data_file_id)` order)
    /// until a bin reaches this many bytes, then emit it as one merged file.
    /// Files already at or above this size are left alone.
    pub target_file_size: u64,
    /// Cap on the number of source files considered in one call, to bound the
    /// memory and I/O of a single merge (candidates are taken in
    /// `(schema_version, data_file_id)` order).
    pub max_merged_files: usize,
    /// Skip files smaller than this many bytes. `0` makes every below-target file
    /// a candidate.
    pub min_file_size: u64,
}

impl Default for MergeOptions {
    fn default() -> Self {
        Self {
            // 512 MiB, matching official DuckLake's target_file_size default and
            // the write-path rollover default (DEFAULT_TARGET_FILE_SIZE), so merge
            // and insert target the same file size.
            target_file_size: crate::table_writer::DEFAULT_TARGET_FILE_SIZE as u64,
            max_merged_files: 1024,
            min_file_size: 0,
        }
    }
}

/// Options for [`DuckLakeTable::rewrite_data_files`].
#[derive(Debug, Clone)]
pub struct RewriteOptions {
    /// Rewrite a data file only when the fraction of its rows masked by its live
    /// delete file is not below this value. DuckDB's default is `0.95`. Must be
    /// in `[0.0, 1.0]`, or NaN, which rewrites every file with deletions, as
    /// official reads a stored `nan`.
    pub delete_threshold: f64,
    /// When set, rewrite only these currently-live data files, regardless of
    /// their delete fraction. This supports explicit physical maintenance such
    /// as re-applying a table sort order without changing logical rows.
    pub data_file_ids: Option<Vec<i64>>,
}

impl Default for RewriteOptions {
    fn default() -> Self {
        Self {
            delete_threshold: 0.95,
            data_file_ids: None,
        }
    }
}

/// Metrics returned by a compaction operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompactionResult {
    /// Number of source data files retired (merged or rewritten).
    pub files_processed: usize,
    /// Number of new (merged / rewritten) files written and registered.
    pub files_created: usize,
    /// Total rows written into the new files.
    pub rows_written: i64,
    /// The snapshot of the flush of the table's inlined data the operation
    /// ran first under an opt-in
    /// [`InlinedFlushPolicy`](crate::table_writer::InlinedFlushPolicy), or
    /// `None` when it flushed nothing (always, under the default policy).
    pub inlined_flush_snapshot: Option<i64>,
}

impl CompactionResult {
    /// A no-op result: nothing matched the operation's criteria.
    fn empty() -> Self {
        Self {
            files_processed: 0,
            files_created: 0,
            rows_written: 0,
            inlined_flush_snapshot: None,
        }
    }

    /// Whether the operation committed anything: retired a source file, or,
    /// under an opt-in
    /// [`InlinedFlushPolicy`](crate::table_writer::InlinedFlushPolicy),
    /// flushed the table's inlined data first. A `false` result committed no
    /// snapshot.
    pub fn did_work(&self) -> bool {
        self.files_processed > 0 || self.inlined_flush_snapshot.is_some()
    }
}

/// The Arrow field for the `_ducklake_internal_snapshot_id` column a merged
/// partial file embeds. Field-id metadata is deliberately absent: only
/// the column order matters here, and `write_compacted_file_stream` re-imposes
/// the field-id-tagged parquet schema.
fn snapshot_column_field() -> Field {
    Field::new(EMBEDDED_SNAPSHOT_ID_COLUMN_NAME, DataType::Int64, true)
}

/// Where one merge source's per-row origin snapshots come from.
///
/// Decided from the PHYSICAL presence of the embedded column in the source's
/// parquet footer, mirroring `DuckLakeMultiFileReader::GetVirtualColumnExpression`'s
/// rule for `COLUMN_IDENTIFIER_SNAPSHOT_ID` (`ducklake_multi_file_reader.cpp`):
/// a source carrying the column keeps its OWN per-row origins, and any other
/// source contributes its single catalog `begin_snapshot` as a constant.
#[derive(Debug, Clone, Copy)]
enum OriginSource {
    /// Every row of this source originates in one snapshot.
    Constant(i64),
    /// This source physically carries `_ducklake_internal_snapshot_id`: its
    /// rows' origins are read from that column, row by row.
    Embedded,
}

/// Resolve one rewritten batch's per-row origin snapshots into the column a
/// merged partial file embeds.
///
/// For [`OriginSource::Embedded`] the origins come from the source's own
/// embedded column, already filtered alongside its rows by
/// [`rewrite_scanned_batch`]; for [`OriginSource::Constant`] every row of the
/// source shares one origin.
fn resolve_origin_column(
    origin: OriginSource,
    carried: Option<&ArrayRef>,
    rows: usize,
    source_path: &str,
) -> DataFusionResult<ArrayRef> {
    if let OriginSource::Constant(origin) = origin {
        return Ok(Arc::new(Int64Array::from(vec![origin; rows])));
    }
    let carried = carried.ok_or_else(|| {
        DataFusionError::Internal(format!(
            "merge source \"{source_path}\" was planned with per-row origins but its scan did \
             not project the embedded snapshot-id column"
        ))
    })?;
    let origins = carried
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| {
            DataFusionError::Internal(format!(
                "embedded snapshot-id column of \"{source_path}\" is not Int64"
            ))
        })?;
    // A NULL origin leaves the row's visibility undefined — the time-travel and
    // change-feed comparisons on it disagree, so the same row can appear in one
    // and not the other. Refuse rather than write it: this same commit removes
    // the sources, so there would be nothing left to recover the origin from.
    if origins.null_count() > 0 {
        return Err(DataFusionError::Execution(format!(
            "merge source \"{source_path}\" has NULL values in its embedded \
             `_ducklake_internal_snapshot_id` column; refusing to merge rows whose origin \
             snapshot is unknown"
        )));
    }
    Ok(Arc::clone(carried))
}

/// Append `origins` to a `[data columns..., rowid]` batch, yielding the
/// `[data columns..., rowid, snapshot_id]` `schema` of a merged partial file.
fn append_snapshot_column(
    batch: &RecordBatch,
    origins: ArrayRef,
    schema: &SchemaRef,
) -> DataFusionResult<RecordBatch> {
    let mut cols: Vec<ArrayRef> = batch.columns().to_vec();
    cols.push(origins);
    Ok(RecordBatch::try_new(Arc::clone(schema), cols)?)
}

/// One source file of a compaction, as a leaf of the compaction plan.
///
/// Wraps that file's positional read plan and rewrites each batch it produces
/// into `[physical columns (catalog types)..., rowid]` — the same per-row
/// transformation an `UPDATE` applies, via the shared [`rewrite_scanned_batch`]
/// — appending the `_ducklake_internal_snapshot_id` column when `origin` is set
/// (a merge whose output is a partial file), resolved per row or as a per-file
/// constant according to [`OriginSource`].
///
/// Carrying provenance as columns OF THE SCAN is what lets a whole bin be read
/// by one execution rather than one file at a time: nothing downstream needs to
/// know which source a batch came from. Official DuckLake compaction has the
/// same shape — a single scan over the compaction set that projects the row-id
/// and snapshot-id virtual columns
/// (`InsertVirtualColumns::WRITE_ROW_ID_AND_SNAPSHOT_ID` in
/// `ducklake_compaction_functions.cpp`).
#[derive(Debug)]
struct CompactionSourceExec {
    /// The source file's positional read plan and its lineage metadata. Shared
    /// rather than cloned: every scan partition reads the same metadata, and it
    /// carries the file's already-deleted position set.
    scan: Arc<UpdateSourceScan>,
    /// The table's physical (data) columns in catalog types.
    physical_schema: SchemaRef,
    /// `[physical columns..., rowid]` — what [`rewrite_scanned_batch`] emits.
    rewrite_schema: SchemaRef,
    /// This exec's output schema: [`Self::rewrite_schema`] plus the embedded
    /// snapshot-id column when [`Self::origin`] is set.
    schema: SchemaRef,
    /// Where this file's rows' origin snapshots come from, for a partial merge
    /// output; `None` leaves the batch at `[physical columns..., rowid]`.
    origin: Option<OriginSource>,
    /// The greatest origin this exec has actually emitted, shared across every
    /// leaf of one bin so the bin can record what it wrote.
    ///
    /// The catalog cannot answer this. A source may physically embed origins
    /// above the `partial_max` its catalog row claims — providers substitute
    /// NULL on catalogs predating the column, the MySQL provider hardcodes it,
    /// and the migration that added it NULL-filled every existing row. Deriving
    /// the output's bound from those rows would record a maximum below one the
    /// merge physically wrote, and `needs_snapshot_filter` only filters below
    /// `partial_max` — so rows would be served at snapshots before they
    /// existed, with the sources retired in the same commit.
    observed_max_origin: Arc<AtomicI64>,
    /// Emit each row's rowid as the source's catalog `row_id_start` plus its
    /// physical position, even when the file embeds a rowid column. Set for
    /// a rowid-adjacent merge, whose output's rowids are positions. Official
    /// DuckLake never reads the sources' rowids for such a merge
    /// (`GenerateCompactionCommand` projects no rowid when the files are
    /// adjacent): each source's rows go into the output in file order.
    positional_rowids: bool,
    properties: Arc<PlanProperties>,
}

impl CompactionSourceExec {
    fn new(
        scan: Arc<UpdateSourceScan>,
        physical_schema: SchemaRef,
        origin: Option<OriginSource>,
        observed_max_origin: Arc<AtomicI64>,
        positional_rowids: bool,
    ) -> Self {
        let rewrite_schema = rewrite_output_schema(&physical_schema);
        let schema = if origin.is_some() {
            let mut fields: Vec<Arc<Field>> = rewrite_schema.fields().iter().cloned().collect();
            fields.push(Arc::new(snapshot_column_field()));
            Arc::new(Schema::new(fields))
        } else {
            Arc::clone(&rewrite_schema)
        };
        // Row-for-row: same partitioning as the file's scan, and no reordering.
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(Arc::clone(&schema)),
            scan.scan.output_partitioning().clone(),
            scan.scan.pipeline_behavior(),
            scan.scan.boundedness(),
        ));
        Self {
            scan,
            physical_schema,
            rewrite_schema,
            schema,
            origin,
            observed_max_origin,
            positional_rowids,
            properties,
        }
    }
}

impl DisplayAs for CompactionSourceExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "CompactionSourceExec: file={}, origin={}",
            self.scan.source_path,
            match self.origin {
                None => "none".to_string(),
                Some(OriginSource::Constant(origin)) => origin.to_string(),
                Some(OriginSource::Embedded) => "embedded".to_string(),
            }
        )
    }
}

impl ExecutionPlan for CompactionSourceExec {
    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> DataFusionResult<TreeNodeRecursion>,
    ) -> DataFusionResult<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn name(&self) -> &str {
        "CompactionSourceExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.scan.scan]
    }

    /// Order-preserving: every row in, one row out, in order.
    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let [child] = <[Arc<dyn ExecutionPlan>; 1]>::try_from(children).map_err(|_| {
            DataFusionError::Internal("CompactionSourceExec expects exactly one child".into())
        })?;
        let mut scan = UpdateSourceScan::clone(&self.scan);
        scan.scan = child;
        Ok(Arc::new(CompactionSourceExec::new(
            Arc::new(scan),
            Arc::clone(&self.physical_schema),
            self.origin,
            Arc::clone(&self.observed_max_origin),
            self.positional_rowids,
        )))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        let input = self.scan.scan.execute(partition, context)?;
        let scan = Arc::clone(&self.scan);
        let physical_schema = Arc::clone(&self.physical_schema);
        let rewrite_schema = Arc::clone(&self.rewrite_schema);
        let schema = Arc::clone(&self.schema);
        let origin = self.origin;
        let observed_max_origin = Arc::clone(&self.observed_max_origin);
        let positional_rowids = self.positional_rowids;
        let stream = input
            .map(move |batch| -> DataFusionResult<Option<RecordBatch>> {
                let batch = batch?;
                // Compaction keeps every live row exactly as it is: no predicate
                // to select with, no assignments to apply.
                let Some(RewrittenBatch {
                    batch,
                    positions,
                    origin_snapshots,
                }) = rewrite_scanned_batch(
                    &physical_schema,
                    &rewrite_schema,
                    &scan,
                    &batch,
                    None,
                    &[],
                )?
                else {
                    // An empty batch contributes nothing; drop it rather than
                    // pass it on, so the parquet writer only ever sees rows.
                    return Ok(None);
                };
                let batch = if positional_rowids {
                    with_positional_rowids(batch, &positions, &scan)?
                } else {
                    batch
                };
                Ok(Some(match origin {
                    Some(origin) => {
                        let origins = resolve_origin_column(
                            origin,
                            origin_snapshots.as_ref(),
                            batch.num_rows(),
                            &scan.source_path,
                        )?;
                        // What the output records has to be what it wrote, not
                        // what the catalog said the sources held.
                        let typed = origins
                            .as_any()
                            .downcast_ref::<arrow::array::Int64Array>()
                            .ok_or_else(|| {
                                // Understating the bound is the defect this
                                // whole path exists to prevent, so a type the
                                // max cannot be taken from is an error rather
                                // than a skipped update.
                                DataFusionError::Internal(format!(
                                    "compaction origin column for \"{}\" is {:?}, not Int64",
                                    scan.source_path,
                                    origins.data_type()
                                ))
                            })?;
                        if let Some(batch_max) = arrow::compute::max(typed) {
                            observed_max_origin.fetch_max(batch_max, Ordering::Relaxed);
                        }
                        append_snapshot_column(&batch, origins, &schema)?
                    },
                    None => batch,
                }))
            })
            .try_filter_map(|batch| std::future::ready(Ok(batch)));
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&self.schema),
            stream,
        )))
    }
}

/// Replace the rowid column of a rewritten `[physical columns..., rowid]` batch
/// with the source's catalog `row_id_start` plus each row's physical position.
fn with_positional_rowids(
    batch: RecordBatch,
    positions: &[i64],
    scan: &UpdateSourceScan,
) -> DataFusionResult<RecordBatch> {
    let start = scan.row_id_start.ok_or_else(|| {
        DataFusionError::Internal(format!(
            "rowid-adjacent merge source \"{}\" has no row_id_start",
            scan.source_path
        ))
    })?;
    let rowids: ArrayRef = Arc::new(Int64Array::from_iter_values(
        positions.iter().map(|position| start + position),
    ));
    let mut columns = batch.columns().to_vec();
    columns[scan.physical_len] = rowids;
    Ok(RecordBatch::try_new(batch.schema(), columns)?)
}

/// Pull from `stream` until the first batch that carries rows, then hand back
/// the stream with that batch put in front of it. `None` means the whole stream
/// was empty, so the compaction has nothing to write and must create no file.
///
/// Replaces the old "collect the whole output, then check the batches are not
/// all empty" guard: one batch is enough to decide.
async fn first_nonempty(
    mut stream: SendableRecordBatchStream,
) -> Result<Option<SendableRecordBatchStream>> {
    let schema = stream.schema();
    while let Some(batch) = stream.next().await {
        let batch = batch?;
        if batch.num_rows() == 0 {
            continue;
        }
        let head = futures::stream::once(std::future::ready(Ok(batch)));
        return Ok(Some(Box::pin(RecordBatchStreamAdapter::new(
            schema,
            head.chain(stream),
        ))));
    }
    Ok(None)
}

/// [`first_nonempty`] of `sorted`, a compaction output sorted by the table's
/// sort order, or — when it fails before emitting a row — of `unsorted`, the
/// same rows in the order an unsorted compaction writes them.
///
/// A sort is blocking: it evaluates every key over every row before it emits
/// one, so a key that fails to evaluate (a division by zero, a failing cast)
/// fails it here. The order is a layout preference, so such a key writes the
/// rows unsorted rather than fail the compaction, as an insert keeps their
/// arrival order. `unsorted` re-executes the sources from a fresh plan state
/// (`reset_plan_states`); an error that is not the key's (a source that
/// cannot be read) fails that run too.
async fn first_nonempty_sorted(
    sorted: Result<SendableRecordBatchStream>,
    unsorted: impl FnOnce() -> Result<SendableRecordBatchStream>,
) -> Result<Option<SendableRecordBatchStream>> {
    match sorted.map(first_nonempty) {
        Ok(first) => match first.await {
            Ok(first) => Ok(first),
            Err(_) => first_nonempty(unsorted()?).await,
        },
        Err(_) => first_nonempty(unsorted()?).await,
    }
}

/// A file's partition identity, normalized for grouping and comparison: the spec
/// generation it was written under (`None` for an unpartitioned file) and its
/// per-key values ordered by `partition_key_index`.
///
/// Two files may be merged only when this matches exactly. Ordering by it also
/// clusters same-partition files together, so bin-packing needs no extra pass.
fn partition_key(file: &DuckLakeTableFile) -> (Option<i64>, Vec<Option<String>>) {
    let mut values = file.partition_values.clone();
    values.sort_by_key(|(index, _)| *index);
    (
        file.partition_id,
        values.into_iter().map(|(_, value)| value).collect(),
    )
}

/// Re-key normalized partition values back to the `(partition_key_index, value)`
/// pairs [`DataFileInfo::with_partition`] persists.
fn partition_value_pairs(values: &[Option<String>]) -> Vec<(i32, Option<String>)> {
    values
        .iter()
        .enumerate()
        .map(|(index, value)| (index as i32, value.clone()))
        .collect()
}

/// Resolve a table's live sort specification into the ordering compaction and
/// `UPDATE` apply to their rewritten rows, against `data_schema` — the table's
/// data columns, which are the LEADING columns of a rewrite batch. `None` means
/// "write unsorted".
///
/// The keys are those an insert sorts by ([`crate::sort::sort_exprs`]): a bare
/// column by position (a float one as DuckDB orders it, `nan` last and `-0.0`
/// equal to `0.0`), any other expression evaluated with DataFusion's
/// semantics. No key of the `duckdb` dialect, or one DataFusion cannot plan,
/// is `None`: the order is layout only, so the rows keep the order an unsorted
/// compaction writes them in. A key naming a column the table lacks fails
/// with official's binder error.
///
/// Resolved ONCE per operation, before any source file is read, so a
/// specification official refuses fails the whole compaction rather than
/// depending on which bin happened to contain rows.
pub(crate) fn compaction_ordering(
    data_schema: &Schema,
    sort_spec: Option<&SortSpec>,
) -> Result<Option<LexOrdering>> {
    let Some(sort_spec) = sort_spec else {
        return Ok(None);
    };
    Ok(LexOrdering::new(crate::sort::sort_exprs(
        sort_spec,
        data_schema,
    )?))
}

/// Stream already-materialized rewrite output through the same path as a
/// compaction plan. `UPDATE` holds its rewritten rows in memory (it interleaves
/// them with per-file delete authoring), so it enters here rather than at
/// [`sorted_rewrite_output`].
pub(crate) fn sorted_rewrite_batches(
    context: Arc<TaskContext>,
    batches: Vec<RecordBatch>,
    ordering: Option<&LexOrdering>,
) -> Result<SendableRecordBatchStream> {
    let schema = batches
        .first()
        .ok_or_else(|| DuckLakeError::Internal("cannot sort empty compaction input".to_string()))?
        .schema();
    let input = MemorySourceConfig::try_new_exec(&[batches], schema, None)?;
    sorted_rewrite_output(context, input, ordering)
}

/// Stream compaction output through DataFusion's spilling sort.
///
/// `input` carries `[data columns..., rowid]` and, for a partial merge, the
/// embedded snapshot-id column. `ordering` (from [`compaction_ordering`])
/// resolves against the leading data columns, so the embedded row lineage stays
/// attached to its row; `None` streams the input through unsorted.
///
/// `input` is normally multi-partition — one partition per source file — because
/// that is how the compaction set is read in parallel.
/// `CoalescePartitionsExec` funnels those partitions back into the single stream
/// the compaction writer consumes.
///
/// That coalesce emits partitions interleaved by arrival, so an unsorted merge's
/// physical row order depends on which object-store request returned first. The
/// same now applies to a single-file `rewrite_data_files`, whose scan is split
/// across byte ranges: its output order is no longer the source order. Official
/// DuckLake preserves order on this copy (`PRESERVE_ORDER` in
/// `ducklake_compaction_functions.cpp`), but here the order costs nothing that
/// is load bearing: an unsorted output written through here carries each row's
/// rowid and origin snapshot as columns and records no `row_id_start`, so
/// nothing downstream derives lineage from where a row sits. (An unsorted
/// rowid-adjacent merge, whose rowids ARE positions, is written through
/// [`rowid_ordered_output`] instead. A sorted one is written through here, and
/// its rows take the rowids of their sorted positions.) Delete
/// files ARE written in position space, but a
/// mutation resolves those positions by rescanning the file it is targeting, so
/// they describe that file's actual layout whatever it turned out to be.
///
/// Preserving order instead means holding a partition until its turn to be
/// emitted. The custom operator that did so buffered each one into a `Vec`
/// outside the memory pool's accounting, which is memory a streaming coalesce
/// does not need and the pool could not see.
///
/// The coalesce starts every input partition at once rather than capping
/// concurrency, so the fan-out is worth being explicit about. Partitions are the
/// bin's files, each split across byte ranges, and two limits bound that from
/// opposite directions:
/// `MergeOptions::max_merged_files` caps how many files a pass considers, and
/// the bin's byte budget caps how much data one plan covers. They cannot both be
/// at maximum.
/// The batches in flight are bounded too: `CoalescePartitionsExec` feeds a
/// `tokio::sync::mpsc` channel sized to the partition count, so a producer that
/// outruns the writer blocks on send rather than accumulating.
pub(crate) fn sorted_rewrite_output(
    context: Arc<TaskContext>,
    input: Arc<dyn ExecutionPlan>,
    ordering: Option<&LexOrdering>,
) -> Result<SendableRecordBatchStream> {
    let schema = input.schema();
    let input: Arc<dyn ExecutionPlan> = if input.output_partitioning().partition_count() > 1 {
        Arc::new(CoalescePartitionsExec::new(input))
    } else {
        input
    };
    let Some(ordering) = ordering else {
        return Ok(input.execute(0, context)?);
    };
    let sorted: Arc<dyn ExecutionPlan> = Arc::new(SortExec::new(ordering.clone(), input));
    let output = Arc::new(ColumnRenameExec::new(sorted, schema, HashMap::new()));
    Ok(output.execute(0, context)?)
}

/// The `row_id_start` a merge of `bin` keeps, when its sources are
/// rowid-adjacent in the order their rows are written.
///
/// Official DuckLake's rule (`GenerateCompactionCommand` in
/// `ducklake_compaction_functions.cpp`): every source has a `row_id_start`,
/// and each one's equals the previous source's `row_id_start + row_count`.
/// The merged file then holds exactly the range `[first start, first start +
/// total rows)` in position order, so it embeds no rowid column and keeps the
/// first source's `row_id_start`. Any other bin returns `None` and embeds.
///
/// Merge sources have no deletes, so `row_count` is the catalog
/// `record_count` (`max_row_count`); a source without one cannot be shown
/// adjacent.
fn adjacent_row_id_start(bin: &[&DuckLakeTableFile]) -> Option<i64> {
    let mut next: Option<i64> = None;
    for tf in bin {
        let start = tf.row_id_start?;
        if next.is_some_and(|next| next != start) {
            return None;
        }
        next = Some(start.checked_add(tf.max_row_count?)?);
    }
    bin.first()?.row_id_start
}

/// Stream an unsorted rowid-adjacent merge's rows in source order, without
/// the rowid column, for a file whose rowids are `start` plus each row's
/// position.
///
/// `input` carries `[data columns..., rowid, (snapshot id)]` with the rowid at
/// `rowid_index`, where each source's rowid is its `row_id_start` plus the
/// row's position in it (`CompactionSourceExec::positional_rowids`). Each
/// partition of it reads one byte range of one source in file order, so each
/// is already in rowid order and a sort-preserving merge on the rowid restores
/// the sources' order across partitions. Official DuckLake gets the same order
/// from `PRESERVE_ORDER` on the merge's copy.
///
/// The position a row lands at IS its rowid in such a file, so the stream
/// checks that the rowids it passes on are exactly `start, start + 1, ...`
/// and fails the merge on the first one that is not, rather than write a file
/// whose rows are out of source order. The caller checks the total row count.
fn rowid_ordered_output(
    context: Arc<TaskContext>,
    input: Arc<dyn ExecutionPlan>,
    rowid_index: usize,
    start: i64,
) -> Result<SendableRecordBatchStream> {
    let schema = input.schema();
    let input: Arc<dyn ExecutionPlan> = if input.output_partitioning().partition_count() > 1 {
        let rowid = schema.field(rowid_index).name();
        let ordering = LexOrdering::new([PhysicalSortExpr::new(
            Arc::new(Column::new(rowid, rowid_index)),
            SortOptions {
                descending: false,
                nulls_first: false,
            },
        )])
        .ok_or_else(|| DuckLakeError::Internal("rowid ordering is empty".to_string()))?;
        Arc::new(SortPreservingMergeExec::new(ordering, input))
    } else {
        input
    };
    let kept: Vec<usize> = (0..schema.fields().len())
        .filter(|&index| index != rowid_index)
        .collect();
    let output_schema = Arc::new(schema.project(&kept)?);
    let mut next = start;
    let stream = input
        .execute(0, context)?
        .map(move |batch| -> DataFusionResult<RecordBatch> {
            let batch = batch?;
            let rowids = batch
                .column(rowid_index)
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| {
                    DataFusionError::Internal("merge rowid column is not Int64".to_string())
                })?;
            for index in 0..rowids.len() {
                if rowids.is_null(index) || rowids.value(index) != next {
                    return Err(DataFusionError::Internal(format!(
                        "rowid-adjacent merge expected rowid {next} at this position, found {}",
                        if rowids.is_null(index) {
                            "NULL".to_string()
                        } else {
                            rowids.value(index).to_string()
                        }
                    )));
                }
                next += 1;
            }
            Ok(batch.project(&kept)?)
        });
    Ok(Box::pin(RecordBatchStreamAdapter::new(
        output_schema,
        stream,
    )))
}

/// `stream` with the column at `index` projected away.
///
/// A sorted rowid-adjacent merge drops the rowid column only after the sort,
/// so the stream it sorts is the same `[data columns..., rowid, (snapshot id)]`
/// shape every other merge sorts.
fn without_column(
    stream: SendableRecordBatchStream,
    index: usize,
) -> Result<SendableRecordBatchStream> {
    let schema = stream.schema();
    let kept: Vec<usize> = (0..schema.fields().len())
        .filter(|&column| column != index)
        .collect();
    let output_schema = Arc::new(schema.project(&kept)?);
    let stream =
        stream.map(move |batch| -> DataFusionResult<RecordBatch> { Ok(batch?.project(&kept)?) });
    Ok(Box::pin(RecordBatchStreamAdapter::new(
        output_schema,
        stream,
    )))
}

impl DuckLakeTable {
    /// The Hive directory a compaction output of the partition `(partition_id,
    /// values)` lands in, whose sources include `source`: named as official
    /// names a partition's directory (`GetPartitionKeyName`) when
    /// `partition_id` is the live generation, else the source's own Hive
    /// directory, which its generation's writer named. Empty names (positional
    /// `key=…`) only when neither is known, as for a key whose column has since
    /// been dropped. The catalog is the authoritative source of partition
    /// values, so the directory costs readability only — never correctness.
    #[cfg(feature = "write")]
    fn partition_subpath(
        &self,
        live: Option<&PartitionSpec>,
        partition_id: i64,
        values: &[Option<String>],
        column_ids: &[i64],
        source: &DuckLakeTableFile,
    ) -> String {
        if let Some(spec) = live.filter(|spec| spec.partition_id == partition_id) {
            let schema = self.physical_schema();
            let keys: Option<Vec<(String, &crate::partition::PartitionTransform)>> = spec
                .columns
                .iter()
                .map(|column| {
                    let index = column_ids.iter().position(|id| *id == column.column_id)?;
                    Some((schema.field(index).name().to_string(), &column.transform))
                })
                .collect();
            if let Some(keys) = keys {
                let names = crate::partition::hive_key_names(
                    keys.iter()
                        .map(|(name, transform)| (name.as_str(), *transform)),
                );
                return crate::partition::hive_subpath(&names, values);
            }
        }
        if source.file.path_is_relative
            && let Some((directory, _)) = source.file.path.rsplit_once('/')
            && !directory.is_empty()
            && directory.split('/').all(|segment| segment.contains('='))
        {
            return directory.to_string();
        }
        crate::partition::hive_subpath(&[], values)
    }

    /// Merge several small adjacent data files of this table into fewer larger
    /// ones, committing the new layout in ONE snapshot.
    ///
    /// Candidates are the table's live files that have no live delete file, whose
    /// size is in `[min_file_size, target_file_size)`, and whose origin snapshot
    /// and schema version are known. They are grouped by schema version (so a DDL
    /// boundary is never crossed) AND by partition identity — matching official
    /// DuckLake, which merges only *within* a partition — and, within a group,
    /// bin-packed in official DuckLake's candidate order (`begin_snapshot`,
    /// `row_id_start`, `data_file_id`) until a bin reaches `target_file_size`;
    /// only bins of two or more files are merged. Delete-bearing files are
    /// deliberately left to [`rewrite_data_files`](Self::rewrite_data_files).
    ///
    /// A merged file inherits its sources' `partition_id` and partition values and
    /// lands in their Hive directory: every file in a bin shares one partition, so
    /// the output belongs to exactly that partition. The inherited generation may be
    /// a *retired* one (files written before a `SET`/`RESET PARTITIONED BY`); that is
    /// correct — the merged rows really do have that generation's layout, and
    /// preserving it keeps them prunable exactly as before.
    ///
    /// When the bin's sources are rowid-adjacent (each one's `row_id_start` is
    /// the previous one's `row_id_start + record_count`), the merged file has
    /// no rowid column and keeps the first source's `row_id_start`, as official
    /// DuckLake's merge does: each row's rowid is that start plus the row's
    /// position in the merged file. Without a sort order the rows are written
    /// in source order, so every rowid is unchanged. With one they are written
    /// in sorted order, so the rowids follow the sorted position, as in
    /// official. That breaks the row lineage guarantee that compaction
    /// preserves rowids; it is an open upstream bug (duckdb/ducklake#1602) and
    /// is matched here until official fixes it. Any other bin embeds each
    /// row's original rowid as a column. A merged file whose rows span more
    /// than one origin snapshot is
    /// written as a partial file (embedding the per-row
    /// `_ducklake_internal_snapshot_id` column and recording `partial_max`). The
    /// sources are retired and scheduled for deletion in the same commit.
    ///
    /// A source may itself be a partial file. Whether its rows carry their own
    /// origins is decided by the PHYSICAL presence of the embedded column in its
    /// parquet footer — never by the catalog's `partial_max`, which providers may
    /// leave NULL on a file that has the column — so those origins are copied
    /// into the output instead of being replaced by one file-level
    /// `begin_snapshot`. Merging a partial source requires this table to be
    /// opened at a snapshot at or above its `partial_max` (all its rows must be
    /// visible to the handle doing the merge); that is an error, while a bin
    /// whose catalog calls a source partial that carries no such column is
    /// skipped.
    ///
    /// Returns no-op metrics (and commits no snapshot) when nothing qualifies.
    /// Errors if the table is read-only (open the catalog with a writer) or if a
    /// source file's rowid lineage cannot be reconstructed.
    ///
    /// Inlined rows and inlined deletions stay where they are, as official's
    /// `ducklake_merge_adjacent_files` leaves them. A file with inlined
    /// deletions is not merged (its deletions would be lost with it); a flush
    /// ([`DuckLakeCatalog::flush_inlined_data`](crate::DuckLakeCatalog::flush_inlined_data)
    /// or [`DuckLakeCatalog::checkpoint`](crate::DuckLakeCatalog::checkpoint))
    /// turns them into a delete file, after which [`Self::rewrite_data_files`]
    /// reclaims it. An opt-in
    /// [`InlinedFlushPolicy`](crate::table_writer::InlinedFlushPolicy) on the
    /// table's write options, an addition of this crate, flushes the table
    /// first when it is due; the merge then plans over the table at the flush
    /// snapshot, which the result's `inlined_flush_snapshot` reports.
    pub async fn merge_adjacent_files(
        &self,
        state: &dyn Session,
        opts: MergeOptions,
    ) -> Result<CompactionResult> {
        match self
            .flush_inlined_if_due(state, "merge_adjacent_files")
            .await?
        {
            Some((flushed, snapshot)) => {
                let mut result = flushed.merge_adjacent_files_planned(state, opts).await?;
                result.inlined_flush_snapshot = Some(snapshot);
                Ok(result)
            },
            None => self.merge_adjacent_files_planned(state, opts).await,
        }
    }

    /// [`Self::merge_adjacent_files`] after its flush step.
    async fn merge_adjacent_files_planned(
        &self,
        state: &dyn Session,
        opts: MergeOptions,
    ) -> Result<CompactionResult> {
        let writer = self.writer().ok_or_else(|| {
            DuckLakeError::InvalidConfig(
                "merge_adjacent_files: table is read-only; open the catalog with a writer"
                    .to_string(),
            )
        })?;
        let schema_name = self.schema_name().ok_or_else(|| {
            DuckLakeError::Internal("writable table has no schema name".to_string())
        })?;
        crate::inlined_flush::refuse_encrypted_catalog(writer.as_ref(), "merge_adjacent_files")?;
        self.write_options().validate()?;
        if !self.write_options().auto_compact.unwrap_or(true) {
            return Ok(CompactionResult::empty());
        }

        // Candidates: live, delete-free, below-target files with a known origin
        // snapshot + schema version, ordered so adjacency and same-version
        // grouping fall out of the sort.
        //
        // A partial file IS a candidate: its scan projects the embedded
        // `_ducklake_internal_snapshot_id` column, so each of its rows keeps its
        // own origin through the merge. Official DuckLake likewise applies no
        // `partial_max` filter to its compaction candidates
        // (`GetFilesForCompaction`). Excluding them instead strands every merge
        // output on a table taking appends between merge passes, leaving each
        // partition a floor of files nothing can reduce.
        let table_files = self.files()?;
        let inlined_deletes = self.inlined_deletes_by_file()?;
        let mut candidates: Vec<&DuckLakeTableFile> = table_files
            .iter()
            .filter(|f| {
                f.delete_file_id.is_none()
                    // Removing a source with inlined deletes would erase its masked rows
                    // from every snapshot while leaving metadata that points at the file.
                    && !inlined_deletes.contains_key(&f.data_file_id)
                    && f.begin_snapshot.is_some()
                    && f.schema_version.is_some()
                    && (f.file.file_size_bytes as u64) >= opts.min_file_size
                    && (f.file.file_size_bytes as u64) < opts.target_file_size
            })
            .collect();
        // Group by (schema_version, partition identity) so both the DDL
        // boundary and the partition boundary fall out of the sort, and within
        // a group take official DuckLake's candidate order: `begin_snapshot`,
        // then `row_id_start` (NULL last), then `data_file_id`
        // (`GetFilesForCompaction`, `ducklake_metadata_manager.cpp`). The order
        // decides both the bins and whether a bin's ranges are adjacent: an
        // earlier merge's output takes a new, higher `data_file_id` but keeps
        // its sources' `begin_snapshot` and range, so it sorts before files
        // appended after those sources.
        candidates.sort_by_key(|f| {
            (
                f.schema_version.unwrap_or(0),
                partition_key(f),
                f.begin_snapshot,
                f.row_id_start.is_none(),
                f.row_id_start,
                f.data_file_id,
            )
        });
        candidates.truncate(opts.max_merged_files);

        // Bin-pack within each (schema-version, partition) run; only bins of >= 2
        // files merge. Merging across partitions would produce a file that belongs
        // to no single partition — unprunable, and unrepresentable in
        // `ducklake_file_partition_value`.
        let mut bins: Vec<Vec<&DuckLakeTableFile>> = Vec::new();
        let mut i = 0;
        while i < candidates.len() {
            let version = candidates[i].schema_version;
            let partition = partition_key(candidates[i]);
            let mut running: u64 = 0;
            let mut bin: Vec<&DuckLakeTableFile> = Vec::new();
            while i < candidates.len()
                && candidates[i].schema_version == version
                && partition_key(candidates[i]) == partition
            {
                bin.push(candidates[i]);
                running += candidates[i].file.file_size_bytes as u64;
                i += 1;
                if running >= opts.target_file_size {
                    break;
                }
            }
            if bin.len() >= 2 {
                bins.push(bin);
            }
        }
        if bins.is_empty() {
            return Ok(CompactionResult::empty());
        }

        // A partial source's rows are ALL visible only at a snapshot at or above
        // its `partial_max`, and a merge reads every live row of its sources. A
        // handle pinned below that would fold rows the handle itself cannot see
        // into a new output, and commit them against a base snapshot that never
        // contained them. Refuse loudly rather than compact from a historical
        // handle: the caller wants a head handle.

        // Safety, from each source's parquet footer:
        //
        // - The merged output is written at the table's CURRENT schema, so a
        //   source carrying a column dropped since it was written would lose that
        //   column's data (and its source is then removed).
        // - A source the catalog calls partial must physically carry the
        //   `_ducklake_internal_snapshot_id` column its rows' origins live in.
        //   Without it the catalog and the file disagree about the file's own
        //   history, and the read path already refuses such a file below
        //   `partial_max` (`build_exec_for_partial_file`); merging it would carry
        //   that disagreement into a new file and remove the evidence.
        //
        // Either way the bin is dropped entirely — those files are left
        // uncompacted rather than silently losing data or lineage. (The common
        // case — files at the current schema, or an older schema that only ADDED
        // columns — is unaffected.) Settled before anything else, so a table with
        // only such bins stays a pure no-op.
        //
        // The facts are kept because the per-bin planning below needs
        // `has_embedded_snapshot` again, and re-asking would re-walk the footers.
        let mut source_facts: HashMap<i64, MergeSourceFacts> = HashMap::new();
        let mut viable: Vec<Vec<&DuckLakeTableFile>> = Vec::with_capacity(bins.len());
        for bin in bins {
            let mut mergeable = true;
            for tf in &bin {
                let facts = self.merge_source_facts(state, &tf.file).await?;
                source_facts.insert(tf.data_file_id, facts);
                if facts.drops_current_columns {
                    mergeable = false;
                } else if tf.partial_max.is_some() && !facts.has_embedded_snapshot {
                    tracing::warn!(
                        file = %tf.file.path,
                        partial_max = ?tf.partial_max,
                        "skipping merge bin: the catalog records this file as partial but it \
                         carries no embedded snapshot-id column"
                    );
                    mergeable = false;
                } else if tf.partial_max.is_some_and(|pm| self.base_snapshot() < pm) {
                    // A handle opened below the file's reach would fold rows it
                    // cannot see into the output and commit them against a base
                    // that never held them. Cost the bin rather than the sweep:
                    // a compactor on a slightly stale handle should still make
                    // progress on everything else, which is the failure this
                    // branch exists to end.
                    tracing::warn!(
                        file = %tf.file.path,
                        partial_max = ?tf.partial_max,
                        base_snapshot = self.base_snapshot(),
                        "skipping merge bin: this table is opened below the snapshot the \
                         source's rows reach; re-open at the current snapshot to merge it"
                    );
                    mergeable = false;
                }
                if !mergeable {
                    break;
                }
            }
            if mergeable {
                viable.push(bin);
            }
        }
        let bins = viable;
        if bins.is_empty() {
            return Ok(CompactionResult::empty());
        }

        let object_store = state
            .runtime_env()
            .object_store(self.object_store_url().as_ref())?;
        // Inherit the table's write options, exactly as the insert path does
        // (`insert_exec.rs`). Compaction re-encodes data that already exists, so
        // writing with the format defaults does not merely fail to optimise — it
        // *undoes* the settings the data was written with. A table written LZ4
        // with a bounded row group comes back uncompressed and, below a million
        // rows, as a single row group nothing can prune into.
        //
        // Official DuckLake has no such gap: its compaction builds its copy
        // options through the same `DuckLakeInsert::GetCopyOptions` inserts use,
        // so a merged file inherits the catalog's configured
        // `parquet_compression` / `parquet_compression_level`
        // (`ducklake_compaction_functions.cpp:655`, `ducklake_insert.cpp:511`).
        // Taking them from the table rather than from a per-call option keeps
        // that single source of truth: one catalog setting, both paths.
        let table_writer = DuckLakeTableWriter::new(Arc::clone(writer), object_store)?
            .with_options(&self.write_options);
        let column_ids = self.column_ids();
        let top_level_column_ids = self.top_level_column_ids();
        let physical_schema = self.physical_schema();

        // Apply the table's live sort order to each merged file (mirroring official
        // DuckLake compaction), so the compacted file's rows are ordered and its
        // per-column min/max stay tight for range pruning. Bin-packing already
        // bounds each output near target_file_size, so no extra file rollover is
        // needed here.
        let sort_spec = self.live_sort_spec()?;
        let ordering = compaction_ordering(physical_schema.as_ref(), sort_spec.as_ref())?;
        // Only for naming the output's Hive directory; the partition identity a
        // merged file carries comes from its sources, not from this.
        let live_partition_spec = self.live_partition_spec()?;

        let mut sources: Vec<CompactionSourceFile> = Vec::new();
        let mut outputs: Vec<CompactionOutputFile> = Vec::new();
        let mut files_processed = 0usize;
        let mut rows_written = 0i64;

        for bin in &bins {
            // Every source's snapshot bounds are catalog metadata, so the shape
            // of the output is settled before a single row is read.
            //
            // A source spans the snapshot RANGE `[begin_snapshot, partial_max]` —
            // a point for an ordinary file, a genuine interval for a partial one
            // — so the output spans the union of those ranges: it begins at the
            // MINIMUM origin (so historical reads back to that point see it,
            // row-filtered by origin) and its `partial_max` is the maximum origin
            // any of its rows carries. `GetCompactionChanges`
            // (`ducklake_transaction_state.cpp`) derives the same two values the
            // same way. The sources are then redundant for every snapshot, so the
            // commit removes + schedules them.
            //
            // The output is partial whenever its rows do not all share one
            // origin, and ALSO whenever any source physically carries the
            // embedded column. That second disjunct is what keeps a re-merge
            // lossless when the catalog understates a source: writing a
            // non-partial output would drop the snapshot column AND record
            // `partial_max` NULL, so no reader would look for those origins
            // again — and this same commit removes the sources.
            let mut range_mins: Vec<i64> = Vec::with_capacity(bin.len());
            let mut range_maxes: Vec<i64> = Vec::with_capacity(bin.len());
            for tf in bin {
                let begin = tf.begin_snapshot.ok_or_else(|| {
                    DuckLakeError::Internal("merge candidate missing begin_snapshot".to_string())
                })?;
                range_mins.push(begin);
                range_maxes.push(tf.partial_max.unwrap_or(begin));
            }
            let min_origin = range_mins.iter().copied().min();
            let max_origin = range_maxes.iter().copied().max();
            let embeds_origins = |tf: &DuckLakeTableFile| {
                source_facts
                    .get(&tf.data_file_id)
                    .is_some_and(|facts| facts.has_embedded_snapshot)
            };
            let partial = min_origin != max_origin || bin.iter().any(|tf| embeds_origins(tf));

            // Rowid-adjacent sources: the output embeds no rowid column and
            // keeps the first source's `row_id_start`, so each row's rowid is
            // that start plus its position in the output, as in official
            // DuckLake's merge (`adjacent_row_id_start`). The rule reads only
            // the catalog ranges, so a source that physically embeds rowids
            // (an inlined-data flush records both the column and a range)
            // qualifies too, and its rows are taken in file order like any
            // other source's. Under a sort order the rows are written in
            // sorted order, so a row the sort moves takes the rowid of its new
            // position, as in official.
            let carried_row_id_start = adjacent_row_id_start(bin);
            // Seeded from the catalog so a bin that embeds nothing still reports
            // its range; every leaf raises it to what it actually emitted.
            let observed_max_origin = Arc::new(AtomicI64::new(max_origin.unwrap_or(i64::MIN)));

            // ONE execution over the whole bin, so DataFusion reads every source
            // concurrently instead of one object-store round trip at a time.
            // Each source contributes a leaf that carries its own rowid lineage
            // (and, for a partial merge, its origin snapshot) as columns, which
            // is what makes the single scan possible — the same shape official
            // DuckLake compaction uses.
            let mut leaves: Vec<Arc<dyn ExecutionPlan>> = Vec::with_capacity(bin.len());
            for (tf, begin) in bin.iter().zip(&range_mins) {
                // One fact — does this file physically carry the embedded column
                // — decides both the scan's projection and where the exec reads
                // origins from, so the two can never disagree and stamp a
                // per-row-origin source with a single `begin_snapshot`. A source
                // that carries it forces `partial`, so the column always has
                // somewhere to go.
                let source_embeds_origins = embeds_origins(tf);
                let origin = partial.then_some(if source_embeds_origins {
                    OriginSource::Embedded
                } else {
                    OriginSource::Constant(*begin)
                });
                let scan = self
                    .build_update_scan_with_snapshot(
                        state,
                        tf,
                        source_embeds_origins,
                        inlined_deletes.get(&tf.data_file_id),
                        None,
                    )
                    .await?;
                leaves.push(Arc::new(CompactionSourceExec::new(
                    Arc::new(scan),
                    Arc::clone(&physical_schema),
                    origin,
                    Arc::clone(&observed_max_origin),
                    carried_row_id_start.is_some(),
                )));
                sources.push(CompactionSourceFile {
                    data_file_id: tf.data_file_id,
                    delete_file_id: None,
                });
                files_processed += 1;
            }

            let rowid_index = physical_schema.fields().len();
            // Fresh plan state each time: a scan executed once hands its
            // files out from a shared queue, which a second execution of the
            // same plan finds drained.
            let unsorted = || -> Result<SendableRecordBatchStream> {
                let input = reset_plan_states(UnionExec::try_new(leaves.clone())?)?;
                match carried_row_id_start {
                    Some(start) => {
                        rowid_ordered_output(state.task_ctx(), input, rowid_index, start)
                    },
                    None => sorted_rewrite_output(state.task_ctx(), input, None),
                }
            };
            // A bin whose sources hold no rows at all writes no file; its
            // sources are still retired by the commit below.
            let merged = match ordering.as_ref() {
                None => first_nonempty(unsorted()?).await?,
                Some(ordering) => {
                    let sorted = UnionExec::try_new(leaves.clone())
                        .map_err(DuckLakeError::from)
                        .and_then(|input| {
                            sorted_rewrite_output(state.task_ctx(), input, Some(ordering))
                        })
                        .and_then(|sorted| match carried_row_id_start {
                            Some(_) => without_column(sorted, rowid_index),
                            None => Ok(sorted),
                        });
                    first_nonempty_sorted(sorted, unsorted).await?
                },
            };
            let Some(merged) = merged else {
                continue;
            };
            // Every file in the bin shares one partition identity (that is the
            // grouping key), so the merged output inherits it: same Hive directory,
            // same `partition_id` + values in the catalog.
            let (partition_id, partition_values) = partition_key(bin[0]);
            let subpath = partition_id
                .filter(|_| self.write_options().hive_file_pattern.unwrap_or(true))
                .map(|pid| {
                    self.partition_subpath(
                        live_partition_spec.as_ref(),
                        pid,
                        &partition_values,
                        &top_level_column_ids,
                        bin[0],
                    )
                });
            let mut file = table_writer
                .write_compacted_file_stream_with_lineage(
                    schema_name,
                    self.table_name(),
                    physical_schema.as_ref(),
                    &column_ids,
                    &top_level_column_ids,
                    merged,
                    carried_row_id_start.is_none(),
                    partial,
                    subpath.as_deref(),
                    self.ext_file_order_request(physical_schema.as_ref())?,
                )
                .await?;
            if let Some(start) = carried_row_id_start {
                // The count closes the range: the file holds exactly as many
                // rows as the sources' ranges cover. (Unsorted, the stream also
                // checked the rows arrived in source order.)
                let expected: i64 = bin.iter().filter_map(|tf| tf.max_row_count).sum();
                if file.record_count != expected {
                    return Err(DuckLakeError::Internal(format!(
                        "rowid-adjacent merge wrote {} rows, its sources hold {expected}",
                        file.record_count
                    )));
                }
                file.row_id_start = RowIdStart::Preserved(start);
            }
            rows_written += file.record_count;
            let file = match partition_id {
                Some(pid) => file.with_partition(pid, partition_value_pairs(&partition_values)),
                None => file,
            };
            // Read after the write has drained the stream: the bound the
            // catalog claimed, raised to whatever the merge actually emitted.
            let partial_max = partial.then(|| observed_max_origin.load(Ordering::Relaxed));
            outputs.push(CompactionOutputFile {
                file,
                partial_max,
                begin_snapshot: min_origin,
            });
        }

        if sources.is_empty() {
            return Ok(CompactionResult::empty());
        }
        if let Err(error) = writer.commit_compaction(
            self.table_id(),
            self.base_snapshot(),
            &sources,
            &outputs,
            SourceRetirement::Remove,
        ) {
            let files = outputs
                .iter()
                .map(|output| (output.file.path.as_str(), output.file.path_is_relative));
            return Err(table_writer
                .release_refused_files(schema_name, self.table_name(), error, files)
                .await);
        }
        Ok(CompactionResult {
            files_processed,
            files_created: outputs.len(),
            rows_written,
            inlined_flush_snapshot: None,
        })
    }

    /// Rewrite data files whose deleted fraction is at least
    /// `opts.delete_threshold`, dropping their deleted rows, in ONE snapshot.
    ///
    /// For each live file with a delete file masking at least that fraction of
    /// its rows, the file's LIVE rows are read (delete-aware) and written to a
    /// new file that preserves each row's original rowid; the old data file AND
    /// its delete file are retired and scheduled for deletion. A file whose rows
    /// are entirely deleted is retired with no replacement.
    ///
    /// Returns no-op metrics (and commits no snapshot) when no file exceeds the
    /// threshold. Errors if the table is read-only or `delete_threshold` is
    /// outside `[0.0, 1.0]`.
    ///
    /// Inlined deletions of a file count towards its deleted fraction, and a
    /// file whose rows they all delete is retired like any other. Inlined rows
    /// stay inlined, as official's `ducklake_rewrite_data_files` leaves them,
    /// unless an opt-in
    /// [`InlinedFlushPolicy`](crate::table_writer::InlinedFlushPolicy) flushes
    /// them first, as for [`Self::merge_adjacent_files`].
    pub async fn rewrite_data_files(
        &self,
        state: &dyn Session,
        opts: RewriteOptions,
    ) -> Result<CompactionResult> {
        match self
            .flush_inlined_if_due(state, "rewrite_data_files")
            .await?
        {
            Some((flushed, snapshot)) => {
                let mut result = flushed.rewrite_data_files_planned(state, opts).await?;
                result.inlined_flush_snapshot = Some(snapshot);
                Ok(result)
            },
            None => self.rewrite_data_files_planned(state, opts).await,
        }
    }

    /// Flush this table's inlined data when its opt-in
    /// [`InlinedFlushPolicy`](crate::table_writer::InlinedFlushPolicy) says so,
    /// returning the table reopened at the flush snapshot and the snapshot;
    /// `None` when it flushed nothing (the default policy, read-only,
    /// `auto_compact` off, an encrypted catalog, or no trigger fired).
    async fn flush_inlined_if_due(
        &self,
        state: &dyn Session,
        operation: &str,
    ) -> Result<Option<(DuckLakeTable, i64)>> {
        let Some(writer) = self.writer() else {
            return Ok(None);
        };
        let Some(schema_name) = self.schema_name() else {
            return Ok(None);
        };
        self.write_options().validate()?;
        // The policy is opt-in: without one, nothing is read from the catalog.
        let policy = self
            .write_options()
            .inlined_flush_policy
            .unwrap_or_default();
        if policy.min_inlined_rows.is_none() && policy.max_inlined_age.is_none() {
            return Ok(None);
        }
        if !self.write_options().auto_compact.unwrap_or(true) || writer.catalog_encrypted()? {
            return Ok(None);
        }
        let Some(backlog) = self.inlined_flush_backlog(state).await? else {
            return Ok(None);
        };
        let pending = backlog.rows + backlog.file_deletes;
        if pending == 0 {
            return Ok(None);
        }
        let due = policy.min_inlined_rows.is_some_and(|min| pending >= min)
            || match (policy.max_inlined_age, backlog.oldest_snapshot) {
                (Some(age), Some(oldest)) => self
                    .snapshot_age(oldest)?
                    .is_some_and(|elapsed| elapsed >= age),
                _ => false,
            };
        if !due {
            return Ok(None);
        }
        tracing::debug!(
            table = self.table_name(),
            pending,
            operation,
            "flushing inlined data before compaction"
        );
        // The flush writes with the options the compaction was asked to use
        // (compression, row groups, rollover, Hive paths), over the table's
        // stored ones as every write applies them.
        let catalog =
            crate::DuckLakeCatalog::with_writer(Arc::clone(self.provider()), Arc::clone(writer))?
                .with_write_options(self.write_options().clone());
        let outcome = catalog
            .flush_inlined_data_with(
                crate::inlined_flush::FlushObjectStores::Session(Arc::clone(state.runtime_env())),
                Some(schema_name),
                Some(self.table_name()),
                None,
            )
            .await?;
        let Some(snapshot) = outcome.snapshot_id else {
            return Ok(None);
        };
        Ok(Some((self.reopened_at_head()?, snapshot)))
    }

    /// What a flush of this table at the head would move: its inlined rows
    /// (those live at the table's snapshot), the inlined deletions of Parquet
    /// rows the flush moves (a retired data file's that cannot merge into its
    /// delete file history stay inlined, and do not count), and the smallest
    /// `begin_snapshot` among everything it moves, deleted inlined rows
    /// included. `None` when the metadata provider does not read inlined data
    /// for a flush (a third-party provider without the flush reads), which
    /// leaves nothing to flush.
    async fn inlined_flush_backlog(&self, state: &dyn Session) -> Result<Option<FlushBacklog>> {
        let provider = self.provider();
        let head = provider.get_current_snapshot()?;
        let unsupported = |error: &DuckLakeError| matches!(error, DuckLakeError::Unsupported(_));
        let oldest_row = match provider.oldest_inlined_row_snapshot(self.table_id(), head) {
            Err(error) if unsupported(&error) => {
                tracing::debug!(table = self.table_name(), %error, "not flushing inlined data");
                return Ok(None);
            },
            result => result?,
        };
        let file_deletes = match provider.get_inlined_file_deletes_for_flush(self.table_id(), head)
        {
            Err(error) if unsupported(&error) => {
                tracing::debug!(table = self.table_name(), %error, "not flushing inlined data");
                return Ok(None);
            },
            result => result?,
        };
        let rows: u64 = self
            .inlined_data_with_row_ids()?
            .iter()
            .map(|rows| rows.row_ids.len() as u64)
            .sum();
        let (file_deletes, oldest_delete) = if file_deletes.is_empty() {
            (0, None)
        } else {
            let store = state
                .runtime_env()
                .object_store(self.object_store_url().as_ref())?;
            crate::inlined_flush::flushed_file_deletes(self, store.as_ref(), file_deletes).await?
        };
        Ok(Some(FlushBacklog {
            rows,
            file_deletes,
            oldest_snapshot: oldest_row.into_iter().chain(oldest_delete).min(),
        }))
    }

    /// How long ago snapshot `snapshot` was committed. An expired snapshot
    /// takes the time of the first surviving snapshot after it, the earliest
    /// time known to follow it. `None` when no such snapshot has a time.
    fn snapshot_age(&self, snapshot: i64) -> Result<Option<std::time::Duration>> {
        let Some(timestamp) = self
            .provider()
            .list_snapshots()?
            .into_iter()
            .filter(|candidate| candidate.snapshot_id >= snapshot)
            .min_by_key(|candidate| candidate.snapshot_id)
            .and_then(|candidate| candidate.timestamp)
            .and_then(|raw| crate::metadata_provider::parse_snapshot_timestamp(&raw))
        else {
            return Ok(None);
        };
        // A snapshot time ahead of this clock (the catalog server's clock
        // runs ahead) is no age at all.
        let now = chrono::Utc::now().naive_utc();
        Ok(Some((now - timestamp).to_std().unwrap_or_default()))
    }

    /// [`Self::rewrite_data_files`] after its flush step.
    async fn rewrite_data_files_planned(
        &self,
        state: &dyn Session,
        mut opts: RewriteOptions,
    ) -> Result<CompactionResult> {
        self.write_options().validate()?;
        if !self.write_options().auto_compact.unwrap_or(true) {
            return Ok(CompactionResult::empty());
        }
        if opts.delete_threshold == RewriteOptions::default().delete_threshold {
            opts.delete_threshold = self
                .write_options()
                .rewrite_delete_threshold
                .unwrap_or(opts.delete_threshold);
        }
        // `threshold < 0 || threshold > 1`, which a NaN passes, as official's
        // `ducklake_set_option` checks it.
        #[allow(clippy::manual_range_contains)]
        if opts.delete_threshold < 0.0 || opts.delete_threshold > 1.0 {
            return Err(DuckLakeError::InvalidConfig(format!(
                "rewrite_data_files: delete_threshold must be in [0.0, 1.0], got {}",
                opts.delete_threshold
            )));
        }
        let writer = self.writer().ok_or_else(|| {
            DuckLakeError::InvalidConfig(
                "rewrite_data_files: table is read-only; open the catalog with a writer"
                    .to_string(),
            )
        })?;
        crate::inlined_flush::refuse_encrypted_catalog(writer.as_ref(), "rewrite_data_files")?;
        let schema_name = self.schema_name().ok_or_else(|| {
            DuckLakeError::Internal("writable table has no schema name".to_string())
        })?;

        let object_store = state
            .runtime_env()
            .object_store(self.object_store_url().as_ref())?;
        // Inherit the table's write options, for the reasons given in
        // `merge_adjacent_files`. A rewrite re-encodes just as a merge does, so
        // it has to carry them too — the two writer constructions are the only
        // places in this crate that could silently disagree about it.
        let table_writer = DuckLakeTableWriter::new(Arc::clone(writer), object_store)?
            .with_options(&self.write_options);
        let column_ids = self.column_ids();
        let top_level_column_ids = self.top_level_column_ids();
        let physical_schema = self.physical_schema();

        // Select the files to rewrite up front, so a table with nothing over the
        // threshold stays a pure no-op — it must not fail on, say, a sort order
        // this crate cannot honour.
        let selected_ids = opts
            .data_file_ids
            .map(|ids| ids.into_iter().collect::<HashSet<_>>());
        let table_files = self.files()?;
        let inlined_deletes = self.inlined_deletes_by_file()?;
        let candidates: Vec<&DuckLakeTableFile> = table_files
            .iter()
            .filter(|tf| match &selected_ids {
                Some(selected_ids) => selected_ids.contains(&tf.data_file_id),
                // Automatic selection counts both Parquet and inlined deletes.
                None => {
                    let record_count = tf.max_row_count.unwrap_or(0);
                    let inlined_count = inlined_deletes
                        .get(&tf.data_file_id)
                        .map_or(0, |positions| positions.len() as i64);
                    let delete_count = tf.delete_count.unwrap_or(0) + inlined_count;
                    // Official keeps a file only when `delete_ratio <
                    // threshold`, which a NaN threshold never is.
                    delete_count > 0
                        && record_count > 0
                        && (delete_count as f64 / record_count as f64)
                            .partial_cmp(&opts.delete_threshold)
                            != Some(std::cmp::Ordering::Less)
                },
            })
            .collect();
        if candidates.is_empty() {
            return Ok(CompactionResult::empty());
        }

        // Re-apply the table's live sort order to each rewritten file so its rows
        // stay ordered (tight min/max) after the delete-driven rewrite.
        let sort_spec = self.live_sort_spec()?;
        let ordering = compaction_ordering(physical_schema.as_ref(), sort_spec.as_ref())?;
        // Only for naming the output's Hive directory (see `partition_path_names`);
        // a rewritten file inherits its partition identity from the file it replaces.
        let live_partition_spec = self.live_partition_spec()?;

        let mut sources: Vec<CompactionSourceFile> = Vec::new();
        let mut outputs: Vec<CompactionOutputFile> = Vec::new();
        let mut files_processed = 0usize;
        let mut rows_written = 0i64;

        for tf in candidates {
            // A rewrite replaces ONE source file, so its scan is already the
            // whole set; `build_update_scan` splits it across partitions itself,
            // since this plan is executed directly and never reaches the
            // optimizer's repartition rule. Rowid lineage rides out of the scan
            // as a column, so the sort and the parquet write consume the plan
            // directly instead of a fully collected `Vec<RecordBatch>`.
            let scan = self
                .build_update_scan(state, tf, inlined_deletes.get(&tf.data_file_id))
                .await?;
            let source: Arc<dyn ExecutionPlan> = Arc::new(CompactionSourceExec::new(
                Arc::new(scan),
                Arc::clone(&physical_schema),
                // A delete-rewrite emits no origin column, so nothing raises
                // this and nothing reads it.
                None,
                Arc::new(AtomicI64::new(i64::MIN)),
                // A rewrite always embeds its rowids: deleted rows leave gaps.
                false,
            ));
            // Fresh plan state, as for a merge: the sorted run drained the
            // scan's shared file queue.
            let unsorted = || {
                sorted_rewrite_output(
                    state.task_ctx(),
                    reset_plan_states(Arc::clone(&source))?,
                    None,
                )
            };
            let sorted = match ordering.as_ref() {
                None => first_nonempty(unsorted()?).await?,
                Some(ordering) => {
                    let sorted = sorted_rewrite_output(
                        state.task_ctx(),
                        Arc::clone(&source),
                        Some(ordering),
                    );
                    first_nonempty_sorted(sorted, unsorted).await?
                },
            };

            files_processed += 1;
            sources.push(CompactionSourceFile {
                data_file_id: tf.data_file_id,
                delete_file_id: tf.delete_file_id,
            });

            // Every row deleted: the source is retired with no replacement.
            let Some(sorted) = sorted else {
                continue;
            };

            // The rewrite drops deleted rows from ONE source file, so the output
            // holds a subset of that file's rows and therefore its exact
            // partition: inherit the identity and the Hive directory.
            let (partition_id, partition_values) = partition_key(tf);
            let subpath = partition_id
                .filter(|_| self.write_options().hive_file_pattern.unwrap_or(true))
                .map(|pid| {
                    self.partition_subpath(
                        live_partition_spec.as_ref(),
                        pid,
                        &partition_values,
                        &top_level_column_ids,
                        tf,
                    )
                });
            let file = table_writer
                .write_compacted_file_stream_with_lineage(
                    schema_name,
                    self.table_name(),
                    physical_schema.as_ref(),
                    &column_ids,
                    &top_level_column_ids,
                    sorted,
                    true,
                    false,
                    subpath.as_deref(),
                    self.ext_file_order_request(physical_schema.as_ref())?,
                )
                .await?;
            rows_written += file.record_count;
            let file = match partition_id {
                Some(pid) => file.with_partition(pid, partition_value_pairs(&partition_values)),
                None => file,
            };
            // A rewrite output holds only currently-live rows and begins at
            // the compaction snapshot (begin_snapshot = None); its
            // pre-compaction history is served by the retained sources.
            outputs.push(CompactionOutputFile {
                file,
                partial_max: None,
                begin_snapshot: None,
            });
        }

        if sources.is_empty() {
            return Ok(CompactionResult::empty());
        }
        // Retire (do not remove) the sources: they still serve time travel to
        // pre-rewrite snapshots until their snapshots are expired.
        if let Err(error) = writer.commit_compaction(
            self.table_id(),
            self.base_snapshot(),
            &sources,
            &outputs,
            SourceRetirement::Retire,
        ) {
            let files = outputs
                .iter()
                .map(|output| (output.file.path.as_str(), output.file.path_is_relative));
            return Err(table_writer
                .release_refused_files(schema_name, self.table_name(), error, files)
                .await);
        }
        Ok(CompactionResult {
            files_processed,
            files_created: outputs.len(),
            rows_written,
            inlined_flush_snapshot: None,
        })
    }
}

/// What [`DuckLakeCatalog::checkpoint`](crate::DuckLakeCatalog::checkpoint) did,
/// phase by phase.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct CheckpointResult {
    /// Tables whose inlined rows were flushed to Parquet.
    pub flushed: Vec<crate::inlined_flush::FlushedInlinedTable>,
    /// Snapshots expired; empty when `expire_older_than` is not set.
    pub expired: Vec<crate::maintenance::ExpiredSnapshot>,
    /// `(schema, table, result)` for every table `merge_adjacent_files` compacted.
    pub merged: Vec<(String, String, CompactionResult)>,
    /// `(schema, table, result)` for every table `rewrite_data_files` rewrote.
    pub rewritten: Vec<(String, String, CompactionResult)>,
    /// Paths `cleanup_old_files` deleted.
    pub cleaned_up: Vec<String>,
    /// Paths `delete_orphaned_files` deleted.
    pub orphans_deleted: Vec<String>,
}

/// `delete_older_than`'s default, as official's cleanup functions read it.
const DEFAULT_DELETE_OLDER_THAN: &str = "2 days";

impl crate::DuckLakeCatalog {
    /// Run official DuckLake's `CHECKPOINT`: its six maintenance phases in its
    /// order, each committing on its own, with official's option defaults.
    ///
    /// 1. [`Self::flush_inlined_data`] on every table (it can produce many small
    ///    files, which compaction then merges).
    /// 2. Snapshot expiry of the snapshots older than the catalog's
    ///    `expire_older_than` option; skipped when the option is not set, as
    ///    official's `ducklake_expire_snapshots` without arguments is.
    /// 3. [`DuckLakeTable::merge_adjacent_files`] on every table, up to the
    ///    table's `target_file_size`, with no cap on the files merged.
    /// 4. [`DuckLakeTable::rewrite_data_files`] on every table, at the table's
    ///    `rewrite_delete_threshold` (default `0.95`).
    /// 5. File cleanup of the files scheduled for deletion before the catalog's
    ///    `delete_older_than` option (default `2 days`).
    /// 6. Deletion of the unreferenced Parquet files older than
    ///    `delete_older_than`.
    ///
    /// Tables with `auto_compact` set to `false` are skipped by phases 1, 3 and
    /// 4, as official does. Each phase stays callable on its own: the methods
    /// above, and the backend's
    /// [`CatalogMaintenance`](crate::maintenance::CatalogMaintenance) commands.
    /// A failing phase stops the checkpoint with the phases before it committed,
    /// as official's `CHECKPOINT` does.
    ///
    /// Phases 2, 5 and 6 are backend-specific and dispatched through
    /// [`MetadataWriter::maintenance`](crate::metadata_writer::MetadataWriter::maintenance).
    /// A writer without them (the single-catalog PostgreSQL writer) fails with
    /// [`DuckLakeError::Unsupported`] before any phase runs. On a catalog whose
    /// `encrypted` option is `true` the phases that write Parquet (1, 3 and 4)
    /// are skipped, since this crate writes no encrypted Parquet files, and
    /// expiry and cleanup run. Errors also when the catalog is read-only, or an
    /// interval option does not parse.
    pub async fn checkpoint(&self, state: &dyn Session) -> Result<CheckpointResult> {
        let writer = self.writer().ok_or_else(|| {
            DuckLakeError::InvalidConfig(
                "checkpoint: catalog is read-only; open it with a writer".to_string(),
            )
        })?;
        if writer.maintenance().is_none() {
            return Err(DuckLakeError::Unsupported(
                "checkpoint: this metadata writer implements no snapshot expiry or file \
                 cleanup; run flush_inlined_data, merge_adjacent_files and \
                 rewrite_data_files on their own"
                    .to_string(),
            ));
        }
        let provider = self.provider();
        let settings = provider.get_metadata_settings(None, None)?;
        let encrypted = writer.catalog_encrypted()?;
        if encrypted {
            tracing::warn!(
                "checkpoint: the catalog is encrypted; skipping the flush, merge and rewrite \
                 phases, which would write unencrypted Parquet"
            );
        }
        let expire_older_than = settings
            .get("expire_older_than")
            .filter(|value| !value.is_empty())
            .cloned();
        let delete_older_than = settings
            .get("delete_older_than")
            .cloned()
            .unwrap_or_else(|| DEFAULT_DELETE_OLDER_THAN.to_string());
        // As official's CHECKPOINT runs its phases one after another, an
        // option a phase cannot use fails that phase, after the earlier ones
        // committed.

        // 1. Flush.
        let flushed = if encrypted {
            Vec::new()
        } else {
            self.flush_inlined_data(state, None, None).await?
        };
        let mut result = CheckpointResult {
            flushed,
            ..CheckpointResult::default()
        };
        let maintenance = writer
            .maintenance()
            .expect("maintenance support checked above");

        // 2. Expire.
        if let Some(interval) = &expire_older_than {
            let cutoff = crate::maintenance::interval_before(interval, chrono::Utc::now())?;
            result.expired = maintenance
                .expire_snapshots(crate::maintenance::ExpireCriteria::OlderThan(cutoff))
                .await?;
        }

        // 3. Merge adjacent files.
        let compacted_tables = if encrypted {
            Vec::new()
        } else {
            self.live_table_names()?
        };
        for (schema, table) in compacted_tables.iter().cloned() {
            if let Some(handle) = self.open_table(&schema, &table).await? {
                let options = MergeOptions {
                    target_file_size: handle
                        .write_options()
                        .target_file_size
                        .unwrap_or(crate::table_writer::DEFAULT_TARGET_FILE_SIZE)
                        as u64,
                    max_merged_files: usize::MAX,
                    min_file_size: 0,
                };
                let merged = handle.merge_adjacent_files(state, options).await?;
                if merged.did_work() {
                    result.merged.push((schema, table, merged));
                }
            }
        }

        // 4. Rewrite data files.
        let rewritten_tables = if encrypted {
            Vec::new()
        } else {
            self.live_table_names()?
        };
        for (schema, table) in rewritten_tables {
            if let Some(handle) = self.open_table(&schema, &table).await? {
                let options = RewriteOptions {
                    delete_threshold: rewrite_delete_threshold(
                        &handle,
                        provider
                            .get_schema_by_name(&schema, provider.get_current_snapshot()?)?
                            .map(|meta| meta.schema_id),
                    )?,
                    data_file_ids: None,
                };
                let rewritten = handle.rewrite_data_files(state, options).await?;
                if rewritten.did_work() {
                    result.rewritten.push((schema, table, rewritten));
                }
            }
        }

        // 5. and 6. Clean up old and orphaned files.
        if delete_older_than.is_empty() {
            return Err(DuckLakeError::InvalidConfig(
                "ducklake_cleanup_old_files: either cleanup_all OR older_than must be specified"
                    .to_string(),
            ));
        }
        let data_path = provider.get_data_path()?;
        let (url, _) = crate::path_resolver::parse_object_store_url(&data_path)?;
        let object_store = state.runtime_env().object_store(&url)?;
        let cutoff = crate::maintenance::older_than(&delete_older_than, chrono::Utc::now())?;
        result.cleaned_up = maintenance
            .cleanup_old_files(
                Arc::clone(&object_store),
                crate::maintenance::CleanupCriteria::OlderThan(cutoff),
                false,
            )
            .await?;
        let cutoff = crate::maintenance::older_than(&delete_older_than, chrono::Utc::now())?;
        result.orphans_deleted = maintenance
            .delete_orphaned_files(
                object_store,
                crate::maintenance::CleanupCriteria::OlderThan(cutoff),
                false,
            )
            .await?;
        Ok(result)
    }

    /// `(schema, table)` for every table live at the catalog head.
    fn live_table_names(&self) -> Result<Vec<(String, String)>> {
        let provider = self.provider();
        let snapshot = provider.get_current_snapshot()?;
        let mut names = Vec::new();
        for schema in provider.list_schemas(snapshot)? {
            for table in provider.list_tables(schema.schema_id, snapshot)? {
                names.push((schema.schema_name.clone(), table.table_name));
            }
        }
        Ok(names)
    }

    /// Open `schema.table` at the catalog head; `None` for a view or a table
    /// dropped meanwhile.
    async fn open_table(&self, schema: &str, table: &str) -> Result<Option<Arc<DuckLakeTable>>> {
        use datafusion::catalog::CatalogProvider;
        let Some(schema) = self.schema(schema) else {
            return Ok(None);
        };
        let Some(provider) = schema.table(table).await? else {
            return Ok(None);
        };
        Ok(provider
            .downcast_ref::<DuckLakeTable>()
            .map(|table| Arc::new(table.clone())))
    }
}

/// The table's `rewrite_delete_threshold` option (table, then schema, then
/// global), `0.95` when unset, as official's rewrite reads it.
fn rewrite_delete_threshold(table: &DuckLakeTable, schema_id: Option<i64>) -> Result<f64> {
    let settings = table
        .provider()
        .get_metadata_settings(schema_id, Some(table.table_id()))?;
    let Some(value) = settings.get("rewrite_delete_threshold") else {
        return Ok(0.95);
    };
    // Read with DuckDB's `VARCHAR` cast, as official reads a stored `DOUBLE`
    // (`nan` included).
    let threshold = crate::duckdb_cast::varchar_to_double(value).ok_or_else(|| {
        DuckLakeError::InvalidConfig(format!(
            "rewrite_delete_threshold '{value}' is not a number"
        ))
    })?;
    // `threshold < 0 || threshold > 1`, which a NaN passes.
    #[allow(clippy::manual_range_contains)]
    if threshold < 0.0 || threshold > 1.0 {
        return Err(DuckLakeError::InvalidConfig(
            "The delete_threshold option must be between 0 and 1".to_string(),
        ));
    }
    Ok(threshold)
}

/// What a flush of one table would move; see
/// [`DuckLakeTable::inlined_flush_backlog`].
struct FlushBacklog {
    rows: u64,
    file_deletes: u64,
    oldest_snapshot: Option<i64>,
}

impl DuckLakeTable {
    /// What a maintenance write of this table (merge, rewrite, inlined flush)
    /// asks of its outputs under the experimental `ext_file_order` option;
    /// `schema` holds the written data columns.
    pub(crate) fn ext_file_order_request(
        &self,
        schema: &Schema,
    ) -> Result<Option<crate::file_ordering::FileOrderRequest>> {
        if !self.write_options().ext_file_order.unwrap_or(false) {
            return Ok(None);
        }
        let Some(writer) = self.writer() else {
            return Ok(None);
        };
        Ok(crate::file_ordering::FileOrderRequest::new(
            Arc::clone(writer),
            self.table_id(),
            self.live_sort_spec()?.as_ref(),
            schema,
        ))
    }
}

/// Options for [`DuckLakeTable::verify_file_orders`].
#[derive(Debug, Clone, Default)]
pub struct VerifyOrderOptions {
    /// Only these data files; `None` = every live file without a record.
    pub data_file_ids: Option<Vec<i64>>,
    /// Stop after reading this many files (bounds one pass); `None` = all.
    pub max_files: Option<usize>,
}

/// What [`DuckLakeTable::verify_file_orders`] did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VerifyOrderResult {
    /// Files whose key columns were read and checked.
    pub files_verified: usize,
    /// Of those, files recorded as ordered (rows inserted into the sidecar).
    pub files_recorded: u64,
    /// Of those, files whose rows are NOT in sort order in the catalog's
    /// column types, or whose key values do not cast to them (nothing
    /// recorded).
    pub files_unsorted: usize,
    /// Files not read: already recorded under the current key, name-mapped,
    /// encrypted, a footer without the key columns, or a key column whose
    /// type does not cast to the catalog's.
    pub files_skipped: usize,
    /// Compressed bytes of the key column chunks read.
    pub bytes_read: u64,
}

impl DuckLakeTable {
    /// EXPERIMENTAL (`ext_file_order`): record the verified sort order of live
    /// data files that have no record under the table's current sort key:
    /// files written before the option was on or under an older key, by
    /// another DuckLake client, uploaded parquet registered as is, or
    /// `ducklake_add_data_files`. The files are not rewritten and no snapshot
    /// is taken.
    ///
    /// Per file: read only the sort-key columns (located by field id), in file
    /// order, through the same check the writer runs; if every row is in order,
    /// record the file's path, size and key in `ext_ducklake_file_order`. A row
    /// describes the object at that path, so it needs no snapshot and is true
    /// at every snapshot that lists the file. Concurrent maintenance is
    /// harmless: a row for a file compacted away meanwhile matches no live
    /// file and is dropped by `MetadataWriter::prune_file_orders`. Idempotent.
    /// Rows are recorded once all candidates are read, so an error part way
    /// (a file vacuumed after listing, say) records nothing for the pass.
    pub async fn verify_file_orders(
        &self,
        state: &dyn Session,
        opts: VerifyOrderOptions,
    ) -> Result<VerifyOrderResult> {
        use datafusion::datasource::listing::PartitionedFile;
        use datafusion::datasource::physical_plan::ParquetFileReaderFactory;
        use datafusion::datasource::physical_plan::parquet::DefaultParquetFileReaderFactory;
        use datafusion::physical_plan::metrics::ExecutionPlanMetricsSet;
        use object_store::path::Path as ObjectPath;
        use parquet::arrow::ProjectionMask;
        use parquet::arrow::arrow_reader::ArrowReaderMetadata;
        use parquet::arrow::async_reader::{AsyncFileReader, ParquetRecordBatchStreamBuilder};
        use parquet::file::metadata::SortingColumn;

        let mut result = VerifyOrderResult::default();
        let writer = self.writer().ok_or_else(|| {
            DuckLakeError::InvalidConfig(
                "verify_file_orders: table is read-only; open the catalog with a writer"
                    .to_string(),
            )
        })?;
        let Some(spec) = self.live_sort_spec()? else {
            return Ok(result);
        };
        // Expression keys cannot be evaluated here, so they cannot be verified.
        let Some(keys) = spec.producible_columns() else {
            return Ok(result);
        };
        let physical_schema = self.physical_schema();
        let column_ids = self.top_level_column_ids();
        let mut key_ids: Vec<(i64, SortOptions)> = Vec::with_capacity(keys.len());
        // Each key's catalog type: the reader casts the file's column to it.
        let mut key_types: Vec<DataType> = Vec::with_capacity(keys.len());
        for (name, direction, nulls) in &keys {
            let Ok(index) = physical_schema.index_of(name) else {
                return Ok(result);
            };
            key_types.push(physical_schema.field(index).data_type().clone());
            key_ids.push((
                column_ids[index],
                SortOptions {
                    descending: *direction == crate::sort::SortDirection::Desc,
                    nulls_first: *nulls == crate::sort::NullOrder::NullsFirst,
                },
            ));
        }

        let object_store = state
            .runtime_env()
            .object_store(self.object_store_url().as_ref())?;
        let readers = DefaultParquetFileReaderFactory::new(Arc::clone(&object_store));
        let reader_metrics = ExecutionPlanMetricsSet::new();
        let selected = opts
            .data_file_ids
            .map(|ids| ids.into_iter().collect::<HashSet<_>>());
        let mut candidates = Vec::new();
        for page in self.file_metadata_pages("verify_file_orders", None) {
            candidates.extend(
                page?
                    .into_iter()
                    .map(|metadata| metadata.file)
                    .filter(|tf| {
                        selected
                            .as_ref()
                            .is_none_or(|ids| ids.contains(&tf.data_file_id))
                    }),
            );
        }
        // Files already recorded under the current key are skipped; one
        // recorded under an older key is checked again under this one.
        let current_key = key_ids
            .iter()
            .map(|(id, options)| {
                Some(crate::file_ordering::AttestedKey {
                    field_id: i32::try_from(*id).ok()?,
                    options: *options,
                })
            })
            .collect::<Option<Vec<_>>>()
            .map(|keys| crate::file_ordering::encode(&keys));
        let paths: Vec<&str> = candidates.iter().map(|tf| tf.file.path.as_str()).collect();
        let recorded: HashSet<(String, i64)> = self
            .metadata_provider()
            .verified_file_orders(self.table_id(), &paths)?
            .into_iter()
            .filter(|order| current_key.as_ref() == Some(&order.sort_key))
            .map(|order| (order.path, order.file_size_bytes))
            .collect();

        let mut orders: Vec<crate::ext_file_order::FileOrder> = Vec::new();
        for tf in candidates {
            if opts
                .max_files
                .is_some_and(|max| result.files_verified >= max)
            {
                break;
            }
            if recorded.contains(&(tf.file.path.clone(), tf.file.file_size_bytes))
                || tf.file.mapping_id.is_some()
                || tf
                    .file
                    .encryption_key
                    .as_deref()
                    .is_some_and(|k| !k.is_empty())
            {
                result.files_skipped += 1;
                continue;
            }
            let source = ObjectPath::from(crate::path_resolver::resolve_path(
                self.table_path(),
                &tf.file.path,
                tf.file.path_is_relative,
            )?);
            let size = u64::try_from(tf.file.file_size_bytes).map_err(|_| {
                DuckLakeError::Internal(format!("negative file size for {}", tf.file.path))
            })?;
            let open = || {
                readers.create_reader(
                    0,
                    PartitionedFile::new(source.to_string(), size),
                    crate::table::metadata_size_hint(tf.file.footer_size),
                    &reader_metrics,
                )
            };
            let footer = open()?.get_metadata(None).await?;
            let reader_metadata =
                ArrowReaderMetadata::try_new(Arc::clone(&footer), Default::default())?;
            let file_schema = reader_metadata.schema().clone();
            let field_id = |field: &arrow::datatypes::Field| {
                field
                    .metadata()
                    .get("PARQUET:field_id")
                    .and_then(|id| id.parse::<i64>().ok())
            };
            // By field id; a file with no field ids at all (an external or
            // pre-DuckLake upload) by name, as the reader resolves it.
            let by_name = file_schema.fields().iter().all(|f| field_id(f).is_none());
            let roots: Option<Vec<usize>> = key_ids
                .iter()
                .map(|(id, _)| {
                    if by_name {
                        let index = column_ids.iter().position(|c| c == id)?;
                        file_schema
                            .index_of(physical_schema.field(index).name())
                            .ok()
                    } else {
                        file_schema
                            .fields()
                            .iter()
                            .position(|f| field_id(f) == Some(*id))
                    }
                })
                .collect();
            let Some(roots) = roots else {
                result.files_skipped += 1;
                continue;
            };
            let mut sorted_roots = roots.clone();
            sorted_roots.sort_unstable();
            sorted_roots.dedup();
            let descr = footer.file_metadata().schema_descr();
            result.bytes_read += footer
                .row_groups()
                .iter()
                .flat_map(|rg| rg.columns().iter().enumerate())
                .filter(|(leaf, _)| sorted_roots.contains(&descr.get_column_root_idx(*leaf)))
                .map(|(_, column)| column.compressed_size().max(0) as u64)
                .sum::<u64>();
            let mask = ProjectionMask::roots(descr, sorted_roots.clone());
            let key_reader: Box<dyn AsyncFileReader> = open()?;
            let mut stream =
                ParquetRecordBatchStreamBuilder::new_with_metadata(key_reader, reader_metadata)
                    .with_projection(mask)
                    .build()?;
            let requested: Vec<SortingColumn> = roots
                .iter()
                .zip(&key_ids)
                .map(|(root, (_, options))| SortingColumn {
                    column_idx: sorted_roots.binary_search(root).unwrap() as i32,
                    descending: options.descending,
                    nulls_first: options.nulls_first,
                })
                .collect();
            // Check the rows as the reader sees them, in the catalog's type.
            // An order in the file's own type need not survive the cast: ISO
            // text with mixed UTC offsets read as a TIMESTAMP, for one.
            let file_schema = stream.schema();
            let catalog_typed: Option<Vec<Field>> = sorted_roots
                .iter()
                .zip(file_schema.fields())
                .map(|(root, field)| {
                    let key = roots.iter().position(|r| r == root)?;
                    let target = &key_types[key];
                    arrow::compute::can_cast_types(field.data_type(), target)
                        .then(|| field.as_ref().clone().with_data_type(target.clone()))
                })
                .collect();
            let catalog_ids: Option<Vec<i32>> = key_ids
                .iter()
                .map(|(id, _)| i32::try_from(*id).ok())
                .collect();
            let checked = catalog_typed.and_then(|fields| {
                let schema = Arc::new(Schema::new(fields));
                let check = crate::file_ordering::SortCheck::with_field_ids(
                    schema.as_ref(),
                    &requested,
                    &catalog_ids?,
                )?;
                Some((schema, check))
            });
            let Some((catalog_schema, mut check)) = checked else {
                result.files_skipped += 1;
                continue;
            };
            result.files_verified += 1;
            let mut castable = true;
            while let Some(batch) = stream.next().await {
                let batch = batch?;
                let columns = batch
                    .columns()
                    .iter()
                    .zip(catalog_schema.fields())
                    .map(|(column, field)| {
                        arrow::compute::cast_with_options(
                            column,
                            field.data_type(),
                            &datafusion::common::format::DEFAULT_CAST_OPTIONS,
                        )
                    })
                    .collect::<std::result::Result<Vec<_>, _>>();
                // A value the cast refuses fails the reader too: not verified.
                let Ok(columns) = columns else {
                    castable = false;
                    break;
                };
                check.observe(&RecordBatch::try_new(Arc::clone(&catalog_schema), columns)?)?;
            }
            if !castable {
                result.files_unsorted += 1;
                continue;
            }
            match check.attestation() {
                Some(sort_key) => orders.push(crate::ext_file_order::FileOrder {
                    path: tf.file.path.clone(),
                    file_size_bytes: tf.file.file_size_bytes,
                    sort_key,
                }),
                None => result.files_unsorted += 1,
            }
        }
        result.files_recorded = writer.record_file_orders(
            self.table_id(),
            crate::ext_file_order::BY_VERIFY,
            &orders,
        )?;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sort::{DUCKDB_DIALECT, NullOrder, SortDirection, SortField};

    fn key(expression: &str) -> SortSpec {
        SortSpec {
            sort_id: 7,
            fields: vec![SortField {
                sort_key_index: 0,
                expression: expression.to_string(),
                dialect: DUCKDB_DIALECT.to_string(),
                direction: SortDirection::Asc,
                null_order: NullOrder::NullsLast,
            }],
        }
    }

    #[test]
    fn compaction_orders_by_expression_sort_keys() {
        let data_schema = Schema::new(vec![Field::new("id", DataType::Int64, false)]);
        let ordering = compaction_ordering(&data_schema, Some(&key("-((id / 2))")))
            .unwrap()
            .expect("an expression key DataFusion plans orders the output");
        assert_eq!(ordering.len(), 1);
        // Keys DataFusion cannot plan (a function it lacks, DuckDB's `//`
        // operator) keep the unsorted order.
        for unplannable in ["main.list_value(id)", "-((id // 2))"] {
            assert!(
                compaction_ordering(&data_schema, Some(&key(unplannable)))
                    .unwrap()
                    .is_none(),
                "{unplannable}"
            );
        }
        // A key naming a missing column fails, as official's binder fails it.
        assert!(compaction_ordering(&data_schema, Some(&key("(gone + 1)"))).is_err());
    }
}
