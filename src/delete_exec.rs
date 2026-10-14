//! DuckLake `DELETE` execution plan.
//!
//! Backs [`crate::table::DuckLakeTable`]'s `TableProvider::delete_from`. When
//! executed, it performs the delete and yields a single `count: UInt64` row
//! (rows affected), mirroring the count contract of
//! [`DuckLakeInsertExec`](crate::insert_exec::DuckLakeInsertExec).
//!
//! Two paths:
//! - **Filtered delete** (a `WHERE` clause): for each live data file, resolve the
//!   physical row positions matching the predicate and record them as official
//!   DuckLake's `DuckLakeDelete::FlushDelete` does: at most the scoped
//!   `data_inlining_row_limit` go to
//!   `ducklake_inlined_delete_<table_id>`; a file they empty is dropped; a file
//!   with a live delete file gets a new one carrying each position's snapshot
//!   in its place; any other gets a delete file of its new positions. Matching
//!   inlined rows are ended. Everything commits in a SINGLE snapshot (atomic
//!   multi-file delete). The read path is already delete-aware, so a subsequent
//!   `SELECT` excludes the deleted rows automatically.
//! - **Delete-all** (no `WHERE`): every live row of every file is recorded the
//!   same way — a file without inlined deletions over the limit is dropped
//!   without reading its delete file — and every inlined row is ended, in one
//!   snapshot. On a writer without staged row mutations it is a metadata-only
//!   truncate that ends every live data file.
//!
//! A file rewritten by an UPDATE or by compaction is handled like any other. A
//! delete file's `pos` is a row's physical index in the data file, which a
//! rewrite leaves meaningful: the rewritten rows sit at `0..n-1`. What a rewrite
//! disturbs is the rowid *sequence* — the surviving rowids carry holes where the
//! applied deletes were, and the catalog records no `row_id_start` — and rowids
//! are not what position resolution uses.
//!
//! # Session lifecycle
//!
//! A catalog from [`DuckLakeCatalog::with_writer`](crate::DuckLakeCatalog::with_writer) reads
//! the latest snapshot for each statement, so a later statement in the same `SessionContext`
//! sees this `DELETE` and any other writer's commit. A statement plans against one snapshot; a
//! commit by another writer between planning and execution aborts with a
//! [`Conflict`](crate::DuckLakeError::Conflict), which guards against a stale, non-cumulative
//! delete file resurrecting already-deleted rows.

use std::collections::{HashMap, HashSet};
use std::fmt::{self, Debug};
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, BooleanArray, RecordBatch, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::catalog::Session;
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::execution::object_store::ObjectStoreUrl;
use datafusion::execution::{SendableRecordBatchStream, SessionState, TaskContext};
use datafusion::physical_expr::{EquivalenceProperties, Partitioning, PhysicalExpr};
use datafusion::physical_plan::apply_expression_roots;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use futures::stream;

use crate::metadata_writer::{
    DeleteFileEntry, DroppedDataFile, InlinedFileDelete, InlinedFlushDeleteFile,
    InlinedFlushFileDelete, InlinedRowRef, MetadataWriter, StagedTableData,
};
use crate::table::DuckLakeTable;
use crate::table_writer::DuckLakeTableWriter;

/// Schema for the output of a delete operation: the count of rows deleted.
/// Same shape DataFusion expects from `insert_into`.
fn make_delete_count_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new(
        "count",
        DataType::UInt64,
        false,
    )]))
}

/// Execution plan that deletes rows from a DuckLake table.
///
/// Takes no input stream (unlike insert): the rows to delete are discovered by
/// scanning the table's own data files against `predicate`. `None` predicate
/// means delete ALL rows.
pub struct DuckLakeDeleteExec {
    /// A clone of the target table, used for its reader methods
    /// (`resolve_positions`, `read_delete_file_positions`).
    table: Arc<DuckLakeTable>,
    /// Session captured at plan time. A bare `TaskContext` cannot build physical
    /// exprs or drive the positional sub-plans; `SessionState` is the concrete
    /// `Session` impl that can, so the delete work can run at execute time.
    session_state: SessionState,
    /// Physical predicate over the table's physical columns. `None` => delete
    /// ALL rows (metadata-only truncate).
    predicate: Option<Arc<dyn PhysicalExpr>>,
    /// Metadata writer for the atomic commit.
    writer: Arc<dyn MetadataWriter>,
    /// Object store URL for writing positional delete files.
    object_store_url: Arc<ObjectStoreUrl>,
    schema_name: String,
    table_name: String,
    table_id: i64,
    /// Snapshot the table was opened at (the generation the positions were
    /// resolved against); threaded to the commit for conflict diagnostics.
    base_snapshot: i64,
    cache: Arc<PlanProperties>,
}

impl DuckLakeDeleteExec {
    /// Create a new `DuckLakeDeleteExec`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        table: Arc<DuckLakeTable>,
        session_state: SessionState,
        predicate: Option<Arc<dyn PhysicalExpr>>,
        writer: Arc<dyn MetadataWriter>,
        schema_name: String,
        table_name: String,
        table_id: i64,
        base_snapshot: i64,
        object_store_url: Arc<ObjectStoreUrl>,
    ) -> Self {
        let cache = Self::compute_properties();
        Self {
            table,
            session_state,
            predicate,
            writer,
            object_store_url,
            schema_name,
            table_name,
            table_id,
            base_snapshot,
            cache,
        }
    }

    fn compute_properties() -> Arc<PlanProperties> {
        Arc::new(PlanProperties::new(
            EquivalenceProperties::new(make_delete_count_schema()),
            Partitioning::UnknownPartitioning(1),
            datafusion::physical_plan::execution_plan::EmissionType::Final,
            datafusion::physical_plan::execution_plan::Boundedness::Bounded,
        ))
    }
}

impl Debug for DuckLakeDeleteExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DuckLakeDeleteExec")
            .field("schema_name", &self.schema_name)
            .field("table_name", &self.table_name)
            .field("table_id", &self.table_id)
            .field("delete_all", &self.predicate.is_none())
            .finish_non_exhaustive()
    }
}

impl DisplayAs for DuckLakeDeleteExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        match t {
            DisplayFormatType::Default
            | DisplayFormatType::Verbose
            | DisplayFormatType::TreeRender => {
                write!(
                    f,
                    "DuckLakeDeleteExec: table={}.{}, predicate={}",
                    self.schema_name,
                    self.table_name,
                    if self.predicate.is_some() {
                        "filter"
                    } else {
                        "all-rows"
                    }
                )
            },
        }
    }
}

impl ExecutionPlan for DuckLakeDeleteExec {
    fn apply_expressions(
        &self,
        f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> DataFusionResult<TreeNodeRecursion>,
    ) -> DataFusionResult<TreeNodeRecursion> {
        apply_expression_roots(self.predicate.iter(), f)
    }

    fn name(&self) -> &str {
        "DuckLakeDeleteExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.cache
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        // No input plan: rows to delete are found by scanning the table's files.
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        if !children.is_empty() {
            return Err(DataFusionError::Internal(
                "DuckLakeDeleteExec does not take any children".to_string(),
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
                "DuckLakeDeleteExec only supports partition 0, got {}",
                partition
            )));
        }

        let table = Arc::clone(&self.table);
        let session_state = self.session_state.clone();
        let predicate = self.predicate.clone();
        let writer = Arc::clone(&self.writer);
        let object_store_url = Arc::clone(&self.object_store_url);
        let schema_name = self.schema_name.clone();
        let table_name = self.table_name.clone();
        let table_id = self.table_id;
        let base_snapshot = self.base_snapshot;
        let output_schema = make_delete_count_schema();

        let stream = stream::once(async move {
            let deleted = run_delete(
                table,
                &session_state,
                predicate,
                writer,
                object_store_url,
                &schema_name,
                &table_name,
                table_id,
                base_snapshot,
            )
            .await?;
            let count: ArrayRef = Arc::new(UInt64Array::from(vec![deleted]));
            Ok(RecordBatch::try_new(output_schema, vec![count])?)
        });

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            make_delete_count_schema(),
            stream,
        )))
    }
}

/// Run the delete and return the number of rows deleted. See the module docs for
/// the two paths (filtered positional delete vs. delete-all truncate).
#[allow(clippy::too_many_arguments)]
async fn run_delete(
    table: Arc<DuckLakeTable>,
    session_state: &SessionState,
    predicate: Option<Arc<dyn PhysicalExpr>>,
    writer: Arc<dyn MetadataWriter>,
    object_store_url: Arc<ObjectStoreUrl>,
    schema_name: &str,
    table_name: &str,
    table_id: i64,
    base_snapshot: i64,
) -> DataFusionResult<u64> {
    let state: &dyn Session = session_state;
    let table_files = table
        .files()
        .map_err(|error| DataFusionError::External(Box::new(error)))?;
    let inlined_data = table
        .inlined_data_with_row_ids()
        .map_err(|error| DataFusionError::External(Box::new(error)))?;

    // A writer that stages row mutations records each file's deletions as
    // official DuckLake's `DuckLakeDelete::FlushDelete` does (see
    // [`FileDeletions::record`]); any other writes cumulative delete files.
    let capable = writer.supports_inlined_row_mutations();
    let inline_limit = if capable {
        table.write_options().data_inlining_row_limit.unwrap_or(0)
    } else {
        0
    };

    // Delete-all (no WHERE) on a writer without staged row mutations:
    // metadata-only truncate — end every live data file in one snapshot. Skip
    // the empty table entirely (no-op, no snapshot).
    if predicate.is_none() && !capable {
        if table_files.is_empty() && inlined_data.is_empty() {
            return Ok(0);
        }
        return writer
            .commit_truncate(table_id, schema_name, table_name, base_snapshot)
            .map_err(|e| DataFusionError::External(Box::new(e)));
    }

    // Object store for writing delete files — the same store the positional reads
    // resolve against.
    let object_store = state
        .runtime_env()
        .object_store(object_store_url.as_ref())?;
    let table_writer = DuckLakeTableWriter::new(Arc::clone(&writer), object_store)
        .map(|writer| writer.with_options(table.write_options()))
        .map_err(|e| DataFusionError::External(Box::new(e)))?;
    let inlined_deletes = table.inlined_deletes_by_file()?;

    let predicate = match predicate {
        None => {
            return delete_all(
                &table,
                state,
                &table_writer,
                inline_limit,
                &table_files,
                &inlined_data,
                &inlined_deletes,
                schema_name,
                table_name,
                base_snapshot,
            )
            .await;
        },
        Some(p) => p,
    };

    let mut entries: Vec<DeleteFileEntry> = Vec::new();
    let mut deletions = FileDeletions::default();
    let mut inlined_rows: Vec<InlinedRowRef> = Vec::new();
    let mut total_deleted: u64 = 0;

    for inlined in &inlined_data {
        let values = predicate.evaluate(&inlined.batch)?;
        let matches = values.into_array(inlined.batch.num_rows())?;
        let matches = matches
            .as_any()
            .downcast_ref::<BooleanArray>()
            .ok_or_else(|| {
                DataFusionError::Internal(
                    "DELETE predicate did not evaluate to a boolean array".to_string(),
                )
            })?;
        for (row, row_id) in inlined.row_ids.iter().enumerate() {
            if matches.is_valid(row) && matches.value(row) {
                inlined_rows.push(InlinedRowRef {
                    table_name: inlined.table_name.clone(),
                    row_id: *row_id,
                });
            }
        }
    }
    total_deleted += inlined_rows.len() as u64;

    for tf in &table_files {
        // A file rewritten by an UPDATE or by compaction is handled here like any
        // other. A delete file's `pos` is a row's physical index in the data file,
        // and a rewrite leaves that meaningful — the rewritten rows sit at
        // `0..n-1`. What a rewrite disturbs is the rowid sequence (deleted rows are
        // dropped, so the survivors' rowids carry holes and the catalog records no
        // `row_id_start`), and rowids are not what resolution uses.

        // Physical positions matching the predicate (raw scan; delete files NOT
        // applied — resolution is over the file's own rows).
        let matched = table
            .resolve_positions(state, &tf.file, Arc::clone(&predicate))
            .await?;
        if matched.is_empty() {
            continue;
        }

        // Already-deleted positions for this file: the live delete file plus any
        // inlined deletes. Rows already deleted are neither re-counted nor
        // re-deleted, and a predicate matching only such rows commits nothing.
        let existing = match tf.delete_file {
            Some(ref df) => table.read_delete_file_entries(state, df).await?,
            None => HashMap::new(),
        };
        let inlined = inlined_deletes.get(&tf.data_file_id);
        let mut newly_deleted: Vec<i64> = matched
            .into_iter()
            .filter(|pos| {
                !existing.contains_key(pos) && !inlined.is_some_and(|set| set.contains(pos))
            })
            .collect();
        if newly_deleted.is_empty() {
            // Every matched row was already deleted: nothing changes here.
            continue;
        }
        newly_deleted.sort_unstable();
        total_deleted += newly_deleted.len() as u64;

        if capable {
            deletions
                .record(
                    &table,
                    &table_writer,
                    schema_name,
                    table_name,
                    &DeletionTarget::of_file(tf, &existing),
                    newly_deleted,
                    inline_limit,
                    base_snapshot,
                )
                .await?;
            continue;
        }

        // Carry forward prior Parquet positions and add only newly matched live
        // rows.
        let mut cumulative: Vec<i64> = existing.keys().copied().chain(newly_deleted).collect();
        cumulative.sort_unstable();

        let delete_info = table_writer
            .write_delete_file(schema_name, table_name, &tf.file.path, &cumulative)
            .await
            .map_err(|e| DataFusionError::External(Box::new(e)))?;

        entries.push(DeleteFileEntry {
            data_file_id: tf.data_file_id,
            expected_prev_delete_file: tf.delete_file_id,
            delete: delete_info,
        });
    }

    if capable && (!inlined_rows.is_empty() || deletions.needs_staged_commit()) {
        commit_row_deletes(
            &table,
            state,
            &table_writer,
            schema_name,
            table_name,
            base_snapshot,
            deletions,
            inlined_rows,
        )
        .await?;
        return Ok(total_deleted);
    }
    entries.extend(deletions.positional);

    if entries.is_empty() && inlined_rows.is_empty() {
        // Predicate matched nothing new anywhere: no commit, no snapshot.
        return Ok(0);
    }

    // Commit every affected file's delete file in ONE snapshot (atomic
    // multi-file DELETE). No new data file is appended, so this uses the
    // dedicated delete-only commit rather than `register_data_file_with_deletes`
    // (which requires an appended file).
    if let Err(error) = writer.commit_deletes(
        table_id,
        schema_name,
        table_name,
        base_snapshot,
        &entries,
        &inlined_rows,
    ) {
        let error = table_writer
            .release_written_delete_files(schema_name, table_name, error, &entries)
            .await;
        return Err(DataFusionError::External(Box::new(error)));
    }

    Ok(total_deleted)
}

/// A live data file a statement deletes rows from.
pub(crate) struct DeletionTarget<'a> {
    /// The data file's catalog id.
    pub(crate) data_file_id: i64,
    /// The data file's catalog path, recorded in its delete files.
    pub(crate) data_file_path: &'a str,
    /// The data file's `record_count`.
    pub(crate) record_count: Option<i64>,
    /// The file's live delete file: its id and every position it deletes with
    /// the snapshot it records, or `None` where it records none.
    pub(crate) delete_file: Option<(i64, &'a HashMap<i64, Option<i64>>)>,
}

impl<'a> DeletionTarget<'a> {
    /// The target for `file`, whose live delete file deletes `existing`.
    fn of_file(
        file: &'a crate::metadata_provider::DuckLakeTableFile,
        existing: &'a HashMap<i64, Option<i64>>,
    ) -> Self {
        Self {
            data_file_id: file.data_file_id,
            data_file_path: &file.file.path,
            record_count: file.max_row_count,
            delete_file: file.delete_file_id.map(|id| (id, existing)),
        }
    }
}

/// How one statement records its deletions of Parquet rows, per data file.
#[derive(Default)]
pub(crate) struct FileDeletions {
    /// New delete files for files that had none.
    pub(crate) positional: Vec<DeleteFileEntry>,
    /// Delete files that replace a file's live one.
    pub(crate) rewritten: Vec<crate::table_writer::RewrittenDelete>,
    /// Deletions recorded in `ducklake_inlined_delete_<table_id>`.
    pub(crate) inlined: Vec<InlinedFileDelete>,
    /// Files whose rows are all deleted.
    pub(crate) dropped: Vec<DroppedDataFile>,
    /// The snapshot rewritten delete files date the statement's deletions to,
    /// once the first is written.
    commit_snapshot: Option<i64>,
}

impl FileDeletions {
    /// Whether the deletions need the staged commit: anything beyond new delete
    /// files for files that had none.
    pub(crate) fn needs_staged_commit(&self) -> bool {
        !self.rewritten.is_empty() || !self.inlined.is_empty() || !self.dropped.is_empty()
    }

    /// Record the deletion of `positions` (sorted, distinct, none of them
    /// deleted already) from `target`, as official DuckLake's
    /// `DuckLakeDelete::FlushDelete` does (`ducklake_delete.cpp`):
    ///
    /// - at most the scoped `inline_limit` (when positive) go to
    ///   `ducklake_inlined_delete_<table_id>`;
    /// - otherwise, when the file has a live delete file, a new one carries its
    ///   positions with their snapshots and the new ones at the snapshot the
    ///   commit is expected to take, replacing it; when that would delete
    ///   every row of the file, the file is dropped instead;
    /// - otherwise a new delete file holds the positions, or the file is
    ///   dropped when they are all of its rows.
    ///
    /// Deletions a file has in `ducklake_inlined_delete_<table_id>` are not
    /// counted towards dropping it, as official does not count them.
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn record(
        &mut self,
        table: &DuckLakeTable,
        table_writer: &DuckLakeTableWriter,
        schema_name: &str,
        table_name: &str,
        target: &DeletionTarget<'_>,
        positions: Vec<i64>,
        inline_limit: usize,
        base_snapshot: i64,
    ) -> DataFusionResult<()> {
        if positions.is_empty() {
            return Ok(());
        }
        if inline_limit > 0 && positions.len() <= inline_limit {
            self.inlined.push(InlinedFileDelete {
                data_file_id: target.data_file_id,
                positions,
                expected_delete_file: target.delete_file.map(|(id, _)| id),
            });
            return Ok(());
        }
        let deletes_every_row = |count: usize| {
            target
                .record_count
                .is_some_and(|rows| usize::try_from(rows).is_ok_and(|rows| rows == count))
        };
        let Some((delete_file_id, existing)) = target.delete_file else {
            if deletes_every_row(positions.len()) {
                self.dropped.push(DroppedDataFile {
                    data_file_id: target.data_file_id,
                    expected_delete_file: None,
                });
                return Ok(());
            }
            let delete = table_writer
                .write_delete_file(schema_name, table_name, target.data_file_path, &positions)
                .await
                .map_err(|e| DataFusionError::External(Box::new(e)))?;
            self.positional.push(DeleteFileEntry {
                data_file_id: target.data_file_id,
                expected_prev_delete_file: None,
                delete,
            });
            return Ok(());
        };
        if deletes_every_row(existing.len() + positions.len()) {
            self.dropped.push(DroppedDataFile {
                data_file_id: target.data_file_id,
                expected_delete_file: Some(delete_file_id),
            });
            return Ok(());
        }
        // A delete file that records no snapshot for a position deleted it at
        // the file's own `begin_snapshot`.
        let fallback = if existing.values().any(Option::is_none) {
            Some(table.delete_file_begin_snapshot(delete_file_id)?)
        } else {
            None
        };
        // The statement's deletions are dated to the id the catalog's next
        // snapshot is expected to take (on a multicatalog store, other
        // catalogs' snapshots draw from the same sequence), and at least the
        // snapshot after the one it read; the commit re-dates them should
        // another snapshot take that id first.
        let commit_snapshot = match self.commit_snapshot {
            Some(snapshot) => snapshot,
            None => {
                let next = table_writer
                    .next_snapshot_id()
                    .map_err(|e| DataFusionError::External(Box::new(e)))?
                    .map_or(base_snapshot + 1, |next| next.max(base_snapshot + 1));
                self.commit_snapshot = Some(next);
                next
            },
        };
        let source = crate::table_writer::RewrittenDeleteSource {
            data_file_path: target.data_file_path.to_string(),
            existing: existing
                .iter()
                .map(|(position, snapshot)| {
                    (*position, snapshot.or(fallback).unwrap_or(commit_snapshot))
                })
                .collect(),
            positions,
        };
        let (delete, begin_snapshot, partial_max) = table_writer
            .write_rewritten_delete_file(schema_name, table_name, &source, commit_snapshot)
            .await
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        self.rewritten.push(crate::table_writer::RewrittenDelete {
            entry: InlinedFlushFileDelete {
                data_file_id: target.data_file_id,
                data_file_end: None,
                expected_prev_delete_file: Some(delete_file_id),
                delete: InlinedFlushDeleteFile {
                    delete,
                    begin_snapshot,
                    partial_max: Some(partial_max),
                    end_snapshot: None,
                },
            },
            source,
        });
        Ok(())
    }
}

/// Reads the live delete file of one of `table`'s data files at the
/// catalog's latest snapshot, for a commit whose inlined deletions met a
/// delete file their statement did not read.
pub(crate) struct LatestDeleteFiles<'a> {
    pub(crate) table: &'a DuckLakeTable,
    pub(crate) state: &'a dyn Session,
}

impl crate::table_writer::DeleteFilePositions for LatestDeleteFiles<'_> {
    fn live_delete_file<'a>(
        &'a self,
        table_id: i64,
        data_file_id: i64,
    ) -> futures::future::BoxFuture<'a, crate::Result<Option<crate::table_writer::LiveDeleteFile>>>
    {
        Box::pin(async move {
            let provider = self.table.provider();
            let snapshot = provider.get_current_snapshot()?;
            let file = provider
                .get_table_files_for_select(table_id, snapshot)?
                .into_iter()
                .find(|file| file.data_file_id == data_file_id);
            let Some(file) = file else {
                return Ok(None);
            };
            let (Some(delete_file_id), Some(delete_file)) = (file.delete_file_id, file.delete_file)
            else {
                return Ok(None);
            };
            let positions = self
                .table
                .read_delete_file_entries(self.state, &delete_file)
                .await?
                .into_keys()
                .collect();
            Ok(Some((delete_file_id, positions)))
        })
    }
}

/// Commit a `DELETE` that ends inlined rows, records inlined file deletions,
/// rewrites delete files or drops data files, as one staged write in one
/// snapshot.
#[allow(clippy::too_many_arguments)]
async fn commit_row_deletes(
    table: &DuckLakeTable,
    state: &dyn Session,
    table_writer: &DuckLakeTableWriter,
    schema_name: &str,
    table_name: &str,
    base_snapshot: i64,
    deletions: FileDeletions,
    inlined_rows: Vec<InlinedRowRef>,
) -> DataFusionResult<()> {
    let latest = LatestDeleteFiles {
        table,
        state,
    };
    let mut transaction = table_writer.transaction();
    transaction.set_delete_file_positions(&latest);
    transaction
        .stage_row_mutation(
            schema_name,
            table_name,
            table.table_id(),
            table.column_ids(),
            table.physical_schema().as_ref(),
            base_snapshot,
            StagedTableData::None,
            None,
            deletions.positional,
            deletions.rewritten,
            inlined_rows,
            deletions.inlined,
            deletions.dropped,
        )
        .await
        .map_err(|e| DataFusionError::External(Box::new(e)))?;
    transaction
        .commit()
        .await
        .map_err(|e| DataFusionError::External(Box::new(e)))?;
    Ok(())
}

/// `DELETE` without `WHERE` on a writer that stages row mutations, as official
/// DuckLake deletes every row: each data file's live rows are recorded through
/// [`FileDeletions::record`] — inlined when they number at most
/// `inline_limit`, else the file is dropped when no deletion of it is inlined,
/// or gets a delete file when one is — and every live inlined row is ended, all
/// in one snapshot.
#[allow(clippy::too_many_arguments)]
async fn delete_all(
    table: &DuckLakeTable,
    state: &dyn Session,
    table_writer: &DuckLakeTableWriter,
    inline_limit: usize,
    table_files: &[crate::metadata_provider::DuckLakeTableFile],
    inlined_data: &[crate::metadata_provider::DuckLakeInlinedData],
    inlined_deletes: &HashMap<i64, HashSet<i64>>,
    schema_name: &str,
    table_name: &str,
    base_snapshot: i64,
) -> DataFusionResult<u64> {
    let inlined_rows: Vec<InlinedRowRef> = inlined_data
        .iter()
        .flat_map(|inlined| {
            inlined.row_ids.iter().map(|row_id| InlinedRowRef {
                table_name: inlined.table_name.clone(),
                row_id: *row_id,
            })
        })
        .collect();
    let mut total = inlined_rows.len() as u64;
    let mut deletions = FileDeletions::default();
    for tf in table_files {
        // A file whose catalog row records no count is counted from its
        // footer and its delete file, so every row of it is deleted too.
        let rows = match tf.max_row_count {
            Some(rows) => rows,
            None => table.data_file_row_count(state, &tf.file).await?,
        };
        let deleted = match (tf.delete_count, &tf.delete_file) {
            (Some(count), _) => count,
            (None, Some(delete_file)) => table
                .read_delete_file_entries(state, delete_file)
                .await?
                .len() as i64,
            (None, None) => 0,
        };
        let inlined = inlined_deletes.get(&tf.data_file_id);
        let live = rows
            .checked_sub(deleted)
            .and_then(|live| live.checked_sub(inlined.map_or(0, |set| set.len() as i64)))
            .ok_or_else(|| {
                DataFusionError::Execution(format!(
                    "data file {} row counts overflow",
                    tf.data_file_id
                ))
            })?;
        if live <= 0 {
            continue;
        }
        total += live as u64;
        if inlined.is_none() && (inline_limit == 0 || live > inline_limit as i64) {
            // Every remaining row is deleted, so the file is dropped; its
            // positions need not be read.
            deletions.dropped.push(DroppedDataFile {
                data_file_id: tf.data_file_id,
                expected_delete_file: tf.delete_file_id,
            });
            continue;
        }
        let existing = match tf.delete_file {
            Some(ref df) => table.read_delete_file_entries(state, df).await?,
            None => HashMap::new(),
        };
        let positions: Vec<i64> = (0..rows)
            .filter(|pos| {
                !existing.contains_key(pos) && !inlined.is_some_and(|set| set.contains(pos))
            })
            .collect();
        deletions
            .record(
                table,
                table_writer,
                schema_name,
                table_name,
                &DeletionTarget::of_file(tf, &existing),
                positions,
                inline_limit,
                base_snapshot,
            )
            .await?;
    }
    if inlined_rows.is_empty()
        && deletions.positional.is_empty()
        && !deletions.needs_staged_commit()
    {
        return Ok(0);
    }
    commit_row_deletes(
        table,
        state,
        table_writer,
        schema_name,
        table_name,
        base_snapshot,
        deletions,
        inlined_rows,
    )
    .await?;
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delete_count_schema() {
        let schema = make_delete_count_schema();
        assert_eq!(schema.fields().len(), 1);
        assert_eq!(schema.field(0).name(), "count");
        assert_eq!(schema.field(0).data_type(), &DataType::UInt64);
    }
}
