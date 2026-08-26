//! Explicit, triggered DuckLake compaction for a single table.
//!
//! Two maintenance operations, each invoked programmatically (never
//! automatically on write) and returning a [`CompactionResult`] with metrics:
//!
//! 1. [`DuckLakeTable::merge_adjacent_files`] coalesces several small data files
//!    of one table (of the SAME schema version — never across a DDL boundary)
//!    into fewer larger ones. A merged file that spans more than one origin
//!    snapshot is written as a DuckLake **partial data file**: it embeds each
//!    row's original rowid AND a per-row `_ducklake_internal_snapshot_id` column,
//!    and its catalog row records `partial_max` (the maximum origin snapshot id
//!    among its rows), so time travel / change feeds can still attribute every
//!    merged row to its origin snapshot.
//! 2. [`DuckLakeTable::rewrite_data_files`] rewrites a data file whose deleted
//!    fraction exceeds a threshold (DuckDB's default is 0.95): it reads only the
//!    file's LIVE rows (delete-aware), writes them to a new file preserving each
//!    row's rowid, and retires BOTH the old data file and its delete file.
//!
//! Both operations commit ATOMICALLY in one snapshot via
//! `MetadataWriter::commit_compaction`: the rewritten outputs are registered, the
//! source files (and, for a rewrite, their delete files) are retired
//! (`end_snapshot` set) and scheduled for physical deletion, and
//! `ducklake_snapshot_changes.changes_made` records `compacted_table:<table_id>`.
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

use arrow::array::{ArrayRef, Int64Array, RecordBatch};
use arrow::compute::SortOptions;
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::catalog::Session;
use datafusion::datasource::memory::MemorySourceConfig;
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::{
    EquivalenceProperties, LexOrdering, PhysicalSortExpr, expressions::Column,
};
use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::union::UnionExec;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, PlanProperties,
    sorts::sort::SortExec,
};
use futures::{StreamExt, TryStreamExt};

use crate::column_rename::ColumnRenameExec;
use crate::metadata_provider::DuckLakeTableFile;
use crate::metadata_writer::{CompactionOutputFile, CompactionSourceFile, SourceRetirement};
use crate::partition::PartitionSpec;
use crate::row_id::EMBEDDED_SNAPSHOT_ID_COLUMN_NAME;
use crate::sort::{SortDirection, SortSpec};
use crate::table::{
    DuckLakeTable, RewrittenBatch, UpdateSourceScan, rewrite_output_schema, rewrite_scanned_batch,
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
    /// delete file is at least this value. DuckDB's default is `0.95`. Must be in
    /// `[0.0, 1.0]`.
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
}

impl CompactionResult {
    /// A no-op result: nothing matched the operation's criteria.
    fn empty() -> Self {
        Self {
            files_processed: 0,
            files_created: 0,
            rows_written: 0,
        }
    }

    /// Whether the operation actually compacted anything (retired a source file).
    /// A `false` result committed no snapshot.
    pub fn did_work(&self) -> bool {
        self.files_processed > 0
    }
}

/// The Arrow field for the constant `_ducklake_internal_snapshot_id` column a
/// merged partial file embeds. Field-id metadata is deliberately absent: only
/// the column order matters here, and `write_compacted_file_stream` re-imposes
/// the field-id-tagged parquet schema.
fn snapshot_column_field() -> Field {
    Field::new(EMBEDDED_SNAPSHOT_ID_COLUMN_NAME, DataType::Int64, true)
}

/// Append a constant `_ducklake_internal_snapshot_id` column (every value =
/// `origin`) to a `[data columns..., rowid]` batch, yielding the
/// `[data columns..., rowid, snapshot_id]` `schema` of a merged partial file.
fn append_snapshot_column(
    batch: &RecordBatch,
    origin: i64,
    schema: &SchemaRef,
) -> DataFusionResult<RecordBatch> {
    let snap: ArrayRef = Arc::new(Int64Array::from(vec![origin; batch.num_rows()]));
    let mut cols: Vec<ArrayRef> = batch.columns().to_vec();
    cols.push(snap);
    Ok(RecordBatch::try_new(Arc::clone(schema), cols)?)
}

/// One source file of a compaction, as a leaf of the compaction plan.
///
/// Wraps that file's positional read plan and rewrites each batch it produces
/// into `[physical columns (catalog types)..., rowid]` — the same per-row
/// transformation an `UPDATE` applies, via the shared [`rewrite_scanned_batch`]
/// — appending the constant `_ducklake_internal_snapshot_id` column when
/// `origin` is set (a merge whose bin spans more than one origin snapshot).
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
    /// Origin snapshot to stamp on every row of this file, for a partial merge
    /// output; `None` leaves the batch at `[physical columns..., rowid]`.
    origin: Option<i64>,
    properties: Arc<PlanProperties>,
}

impl CompactionSourceExec {
    fn new(scan: Arc<UpdateSourceScan>, physical_schema: SchemaRef, origin: Option<i64>) -> Self {
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
            self.origin
                .map_or_else(|| "none".to_string(), |origin| origin.to_string())
        )
    }
}

impl ExecutionPlan for CompactionSourceExec {
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
        let stream = input
            .map(move |batch| -> DataFusionResult<Option<RecordBatch>> {
                let batch = batch?;
                // Compaction keeps every live row exactly as it is: no predicate
                // to select with, no assignments to apply.
                let Some(RewrittenBatch {
                    batch,
                    ..
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
                Ok(Some(match origin {
                    Some(origin) => append_snapshot_column(&batch, origin, &schema)?,
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
/// Resolved ONCE per operation, before any source file is read, so a
/// specification this crate cannot honour fails the whole compaction rather
/// than depending on which bin happened to contain rows.
pub(crate) fn compaction_ordering(
    data_schema: &Schema,
    sort_spec: Option<&SortSpec>,
) -> Result<Option<LexOrdering>> {
    let Some(sort_spec) = sort_spec else {
        return Ok(None);
    };
    let keys = sort_spec.producible_columns().ok_or_else(|| {
        DuckLakeError::InvalidConfig(format!(
            "DuckLake sort order {} contains an unsupported expression; \
             datafusion-ducklake can write only bare-column sort keys",
            sort_spec.sort_id
        ))
    })?;
    // No usable keys means "write unsorted", not an error. `producible_columns` filters
    // out fields whose dialect is not `duckdb`, so a spec authored by another engine can
    // legitimately leave nothing behind. Official DuckLake skips such fields and
    // proceeds with whatever remains — an empty ORDER BY when that is all of them
    // (`ducklake_compaction_functions.cpp`, `ducklake_insert.cpp`). Falling through to
    // `LexOrdering::new` would instead fail a compaction that official completes; the
    // SQL INSERT path already returns "no ordering" for this case.
    if keys.is_empty() {
        return Ok(None);
    }

    let mut expressions = Vec::with_capacity(keys.len());
    for (name, direction, null_order) in keys {
        let index = data_schema.index_of(&name).map_err(|_| {
            DuckLakeError::InvalidConfig(format!(
                "DuckLake sort key '{name}' is not present in the rewrite schema"
            ))
        })?;
        expressions.push(PhysicalSortExpr::new(
            Arc::new(Column::new(&name, index)),
            SortOptions {
                descending: direction == SortDirection::Desc,
                nulls_first: null_order.nulls_first(),
            },
        ));
    }
    Ok(Some(LexOrdering::new(expressions).ok_or_else(|| {
        DuckLakeError::Internal("sort order is empty".to_string())
    })?))
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
/// `input` is normally multi-partition — one partition per source file, plus one
/// per row-group chunk of a large file — because that is how the compaction set
/// is read in parallel. `CoalescePartitionsExec` funnels those partitions back
/// into the single stream the compaction writer consumes.
///
/// That coalesce emits partitions interleaved by arrival, so an unsorted merge's
/// physical row order depends on which object-store request returned first. This
/// matches official DuckLake, which sets `DONT_PRESERVE_ORDER` on exactly this
/// copy (`ducklake_compaction_functions.cpp`), and costs nothing that is load
/// bearing: a compaction output carries each row's rowid and origin snapshot as
/// columns and records no `row_id_start`, so nothing downstream derives lineage
/// from where a row sits. Delete files ARE written in position space, but a
/// mutation resolves those positions by rescanning the file it is targeting, so
/// they describe that file's actual layout whatever it turned out to be.
///
/// Preserving order instead means holding a partition until its turn to be
/// emitted. The custom operator that did so buffered each one into a `Vec`
/// outside the memory pool's accounting, which is memory a streaming coalesce
/// does not need and the pool could not see. Note the coalesce starts every
/// input partition at once rather than capping concurrency; the count is
/// bounded by the bin, which `MergeOptions::max_merged_files` already limits.
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

impl DuckLakeTable {
    /// The live partition spec's key column names in key order, used only to build
    /// the readable Hive directory of a compaction output.
    ///
    /// Empty when the table is unpartitioned, when `partition_id` is a *retired*
    /// generation (whose key order may differ from the live one, so live names would
    /// mislabel the directory), or when a key's column has since been dropped. The
    /// catalog is the authoritative source of partition values, so degrading to
    /// positional `key=…` naming costs readability only — never correctness.
    #[cfg(feature = "write")]
    fn partition_path_names(
        &self,
        live: Option<&PartitionSpec>,
        partition_id: i64,
        column_ids: &[i64],
    ) -> Vec<String> {
        let Some(spec) = live.filter(|spec| spec.partition_id == partition_id) else {
            return Vec::new();
        };
        let schema = self.physical_schema();
        let names: Option<Vec<String>> = spec
            .columns
            .iter()
            .map(|column| {
                let index = column_ids.iter().position(|id| *id == column.column_id)?;
                Some(schema.field(index).name().to_string())
            })
            .collect();
        names.unwrap_or_default()
    }

    /// Merge several small adjacent data files of this table into fewer larger
    /// ones, committing the new layout in ONE snapshot.
    ///
    /// Candidates are the table's live files that have no live delete file, whose
    /// size is in `[min_file_size, target_file_size)`, and whose origin snapshot
    /// and schema version are known. They are grouped by schema version (so a DDL
    /// boundary is never crossed) AND by partition identity — matching official
    /// DuckLake, which merges only *within* a partition — and, within a group,
    /// bin-packed in `data_file_id` order until a bin reaches `target_file_size`;
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
    /// Each source file's live rows are read with their original rowids
    /// preserved; a merged file that spans more than one origin snapshot is
    /// written as a partial file (embedding the per-row
    /// `_ducklake_internal_snapshot_id` column and recording `partial_max`). The
    /// sources are retired and scheduled for deletion in the same commit.
    ///
    /// Returns no-op metrics (and commits no snapshot) when nothing qualifies.
    /// Errors if the table is read-only (open the catalog with a writer) or if a
    /// source file's rowid lineage cannot be reconstructed.
    pub async fn merge_adjacent_files(
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

        // Candidates: live, delete-free, below-target files with a known origin
        // snapshot + schema version, ordered so adjacency and same-version
        // grouping fall out of the sort.
        let table_files = self.files()?;
        let mut candidates: Vec<&DuckLakeTableFile> = table_files
            .iter()
            .filter(|f| {
                f.delete_file_id.is_none()
                    // Never re-merge an existing partial file: its rows carry
                    // per-row origins in the embedded `_ducklake_internal_snapshot_id`
                    // column, which the read path used to reconstruct them does NOT
                    // surface — re-merging would collapse every row onto the file's
                    // single begin_snapshot and corrupt time travel.
                    && f.partial_max.is_none()
                    && f.begin_snapshot.is_some()
                    && f.schema_version.is_some()
                    && (f.file.file_size_bytes as u64) >= opts.min_file_size
                    && (f.file.file_size_bytes as u64) < opts.target_file_size
            })
            .collect();
        // Sort by (schema_version, partition identity, data_file_id) so both the
        // DDL boundary and the partition boundary fall out of the sort, and files
        // stay in data_file_id order (adjacency) within a partition.
        candidates.sort_by_key(|f| {
            (
                f.schema_version.unwrap_or(0),
                partition_key(f),
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

        // Safety: the merged output is written at the table's CURRENT schema, so
        // a source carrying a column dropped since it was written would lose
        // that column's data (and its source is then removed). Drop any such bin
        // entirely — those files are left uncompacted rather than silently
        // losing data. (The common case — files at the current schema, or an
        // older schema that only ADDED columns — is unaffected.) Settled before
        // anything else, so a table with only such bins stays a pure no-op.
        let mut viable: Vec<Vec<&DuckLakeTableFile>> = Vec::with_capacity(bins.len());
        for bin in bins {
            let mut drops_columns = false;
            for tf in &bin {
                if self.file_drops_current_columns(state, &tf.file).await? {
                    drops_columns = true;
                    break;
                }
            }
            if !drops_columns {
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
            // Every source's origin snapshot is catalog metadata, so the shape
            // of the output is settled before a single row is read.
            //
            // A group spanning >1 origin snapshot is a partial file: embed the
            // per-row snapshot column, record the max origin as partial_max, and
            // set begin_snapshot to the MIN origin so historical reads back to
            // that point see it (row-filtered by origin). The sources are then
            // redundant for every snapshot, so the commit removes + schedules
            // them. A single-origin group needs no per-row column (all rows share
            // one origin), and begins at that origin.
            let mut file_origins: Vec<i64> = Vec::with_capacity(bin.len());
            for tf in bin {
                file_origins.push(tf.begin_snapshot.ok_or_else(|| {
                    DuckLakeError::Internal("merge candidate missing begin_snapshot".to_string())
                })?);
            }
            let origins: HashSet<i64> = file_origins.iter().copied().collect();
            let partial = origins.len() > 1;
            let min_origin = origins.iter().copied().min();
            let partial_max = if partial {
                origins.iter().copied().max()
            } else {
                None
            };

            // ONE execution over the whole bin, so DataFusion reads every source
            // concurrently instead of one object-store round trip at a time.
            // Each source contributes a leaf that carries its own rowid lineage
            // (and, for a partial merge, its origin snapshot) as columns, which
            // is what makes the single scan possible — the same shape official
            // DuckLake compaction uses.
            let mut leaves: Vec<Arc<dyn ExecutionPlan>> = Vec::with_capacity(bin.len());
            for (tf, origin) in bin.iter().zip(&file_origins) {
                let scan = self.build_update_scan(state, tf).await?;
                leaves.push(Arc::new(CompactionSourceExec::new(
                    Arc::new(scan),
                    Arc::clone(&physical_schema),
                    partial.then_some(*origin),
                )));
                sources.push(CompactionSourceFile {
                    data_file_id: tf.data_file_id,
                    delete_file_id: None,
                });
                files_processed += 1;
            }

            let merged = sorted_rewrite_output(
                state.task_ctx(),
                UnionExec::try_new(leaves)?,
                ordering.as_ref(),
            )?;
            // A bin whose sources hold no rows at all writes no file; its
            // sources are still retired by the commit below.
            let Some(merged) = first_nonempty(merged).await? else {
                continue;
            };
            // Every file in the bin shares one partition identity (that is the
            // grouping key), so the merged output inherits it: same Hive directory,
            // same `partition_id` + values in the catalog.
            let (partition_id, partition_values) = partition_key(bin[0]);
            let subpath = partition_id.map(|pid| {
                let names = self.partition_path_names(
                    live_partition_spec.as_ref(),
                    pid,
                    &top_level_column_ids,
                );
                crate::partition::hive_subpath(&names, &partition_values)
            });
            let file = table_writer
                .write_compacted_file_stream(
                    schema_name,
                    self.table_name(),
                    physical_schema.as_ref(),
                    &column_ids,
                    &top_level_column_ids,
                    merged,
                    partial,
                    subpath.as_deref(),
                )
                .await?;
            rows_written += file.record_count;
            let file = match partition_id {
                Some(pid) => file.with_partition(pid, partition_value_pairs(&partition_values)),
                None => file,
            };
            outputs.push(CompactionOutputFile {
                file,
                partial_max,
                begin_snapshot: min_origin,
            });
        }

        if sources.is_empty() {
            return Ok(CompactionResult::empty());
        }
        writer.commit_compaction(
            self.table_id(),
            self.base_snapshot(),
            &sources,
            &outputs,
            SourceRetirement::Remove,
        )?;
        Ok(CompactionResult {
            files_processed,
            files_created: outputs.len(),
            rows_written,
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
    pub async fn rewrite_data_files(
        &self,
        state: &dyn Session,
        opts: RewriteOptions,
    ) -> Result<CompactionResult> {
        if !(0.0..=1.0).contains(&opts.delete_threshold) {
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
        let candidates: Vec<&DuckLakeTableFile> = table_files
            .iter()
            .filter(|tf| match &selected_ids {
                Some(selected_ids) => selected_ids.contains(&tf.data_file_id),
                // Threshold selection only applies to files with live deletes.
                None => {
                    let record_count = tf.max_row_count.unwrap_or(0);
                    tf.delete_file_id.is_some()
                        && record_count > 0
                        && tf.delete_count.unwrap_or(0) as f64 / record_count as f64
                            >= opts.delete_threshold
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
            // whole set; the file's own row groups are what DataFusion reads in
            // parallel. Rowid lineage rides out of the scan as a column, so the
            // sort and the parquet write consume the plan directly instead of a
            // fully collected `Vec<RecordBatch>`.
            let scan = self.build_update_scan(state, tf).await?;
            let sorted = sorted_rewrite_output(
                state.task_ctx(),
                Arc::new(CompactionSourceExec::new(
                    Arc::new(scan),
                    Arc::clone(&physical_schema),
                    None,
                )),
                ordering.as_ref(),
            )?;

            files_processed += 1;
            sources.push(CompactionSourceFile {
                data_file_id: tf.data_file_id,
                delete_file_id: tf.delete_file_id,
            });

            // Every row deleted: the source is retired with no replacement.
            let Some(sorted) = first_nonempty(sorted).await? else {
                continue;
            };

            // The rewrite drops deleted rows from ONE source file, so the output
            // holds a subset of that file's rows and therefore its exact
            // partition: inherit the identity and the Hive directory.
            let (partition_id, partition_values) = partition_key(tf);
            let subpath = partition_id.map(|pid| {
                let names = self.partition_path_names(
                    live_partition_spec.as_ref(),
                    pid,
                    &top_level_column_ids,
                );
                crate::partition::hive_subpath(&names, &partition_values)
            });
            let file = table_writer
                .write_compacted_file_stream(
                    schema_name,
                    self.table_name(),
                    physical_schema.as_ref(),
                    &column_ids,
                    &top_level_column_ids,
                    sorted,
                    false,
                    subpath.as_deref(),
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
        writer.commit_compaction(
            self.table_id(),
            self.base_snapshot(),
            &sources,
            &outputs,
            SourceRetirement::Retire,
        )?;
        Ok(CompactionResult {
            files_processed,
            files_created: outputs.len(),
            rows_written,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sort::{DUCKDB_DIALECT, NullOrder, SortDirection, SortField};

    #[test]
    fn compaction_ordering_rejects_expression_sort_key() {
        let data_schema = Schema::new(vec![Field::new("id", DataType::Int64, false)]);
        let sort_spec = SortSpec {
            sort_id: 7,
            fields: vec![SortField {
                sort_key_index: 0,
                expression: "lower(id)".to_string(),
                dialect: DUCKDB_DIALECT.to_string(),
                direction: SortDirection::Asc,
                null_order: NullOrder::NullsLast,
            }],
        };

        let result = compaction_ordering(&data_schema, Some(&sort_spec));
        let err = match result {
            Ok(_) => panic!("expression sort key must be rejected"),
            Err(e) => e,
        };

        assert_eq!(
            err.to_string(),
            "Invalid configuration: DuckLake sort order 7 contains an unsupported expression; \
             datafusion-ducklake can write only bare-column sort keys",
        );
    }
}
