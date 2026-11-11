//! The inlined row versions a change feed reads, in bounded pages.
//!
//! Planning reads the first page of a selection (the versions written, or
//! deleted, in the window) inside the plan's consistent catalog read, and the
//! plan holds it, charged to the session's memory pool. A window whose
//! versions fit one page (the session's `batch_size` rows) is read whole
//! there, as it always was. The rest of a larger one is read while the plan
//! executes, a page at a time, by [`InlinedPagesExec`], so what a feed holds
//! of the catalog's inlined rows is bounded by a page rather than by the
//! window.
//!
//! Every page of a selection shows the catalog the plan read. The pages read
//! at execution go through one consistent read pinned to the state of the
//! plan's ([`MetadataProvider::pin_consistent_read`]), begun at planning and
//! shared by the selections of the feed, so a flush that commits after
//! planning moves no row out from under them: the feed answers as of the
//! catalog it planned against, as official DuckLake reads a whole feed in
//! one transaction. Inside that read the pages stream from one statement per
//! physical inlined-data table (a server-side cursor on PostgreSQL, a
//! temporary table on SQLite; see [`ConsistentRead`]'s cursor), since the
//! inlined tables have no index a keyset query per page could seek with: each
//! page would scan and sort the whole table again. A provider that cannot pin
//! a read, including one whose pool has no connection free to pin it on, has
//! the rest of a selection read at planning, one statement per physical
//! inlined-data table, inside the plan's consistent read.
//!
//! A selection's pages can be read by more than one node of the plan (an
//! update pairing reads them as rows and as keys); each node holds its own
//! handle to the pinned read and lets go of it when it has read its last
//! page, fails, or its stream is dropped, and a node never executed lets go
//! of it when the plan drops. The read ends with the last handle. Until then
//! it holds one catalog connection, and the database keeps the row versions
//! it shows: a feed read slowly holds it for as long as it reads.

use std::sync::{Arc, Mutex};

use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::execution::memory_pool::MemoryConsumer;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::{EquivalenceProperties, Partitioning, PhysicalExpr};
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use futures::{StreamExt, TryStreamExt};

use crate::feed_charge::{
    KeyedInlinedRows, PlanCharge, inlined_rows_bytes, keyed_inlined_batch, release,
};
use crate::inlined_filter::InlinedFilter;
use crate::metadata_provider::{
    ConsistentRead, DuckLakeInlinedRows, DuckLakeTableColumn, InlinedCursorRequest,
    InlinedRowSelection, InlinedRowsAfter, InlinedRowsCursor, MetadataProvider,
};

/// The state of an [`InlinedPagesExec`] stream with pages left: the cursor,
/// once opened, and the pinned read it reads in.
type PagesState = Option<(Option<Box<dyn InlinedRowsCursor>>, Arc<ConsistentRead>)>;
use crate::table_deletions::keyed_rows_schema;

fn external(e: crate::DuckLakeError) -> DataFusionError {
    DataFusionError::External(Box::new(e))
}

/// Where a feed reads its inlined row versions from.
pub(crate) struct InlinedSource<'a> {
    /// The provider of the plan's consistent read: the first page, and every
    /// page when no read can be pinned to its state.
    pub(crate) reader: &'a dyn MetadataProvider,
    pub(crate) table_id: i64,
    pub(crate) columns: &'a Arc<Vec<DuckLakeTableColumn>>,
    pub(crate) table_schema: &'a SchemaRef,
    /// Row versions per page.
    pub(crate) page_rows: usize,
    pub(crate) charge: &'a PlanCharge,
    /// The read the pages after the first are read through.
    pub(crate) pinned: PinnedRead,
}

/// The consistent read pinned to the state of a plan's
/// ([`MetadataProvider::pin_consistent_read`]) that the pages of a feed's
/// selections after their first are read through: begun by the first
/// selection with pages left, and shared by the rest.
#[derive(Default)]
pub(crate) struct PinnedRead {
    // `None` until a selection asks; then the read, or `None` when the
    // reader cannot pin one.
    read: Mutex<Option<Option<Arc<ConsistentRead>>>>,
}

impl PinnedRead {
    fn get(&self, reader: &dyn MetadataProvider) -> DataFusionResult<Option<Arc<ConsistentRead>>> {
        let mut read = self
            .read
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if read.is_none() {
            *read = Some(
                reader
                    .pin_consistent_read()
                    .map_err(external)?
                    .map(Arc::new),
            );
        }
        Ok(read.clone().flatten())
    }
}

/// Let go of one handle to a pinned read, ending it when it is the last: its
/// transaction ends and its connection goes back to the pool. A handle let go
/// of otherwise (dropped on an error or a cancelled stream) ends it as well,
/// rolled back from a task when it is the last.
fn release_pinned(read: Arc<ConsistentRead>) -> DataFusionResult<()> {
    match Arc::try_unwrap(read) {
        Ok(read) => read.end().map_err(external),
        Err(_) => Ok(()),
    }
}

/// The inlined row versions of one selection of a feed's window: the first
/// page, held by the plan, and where the pages after it are read from.
#[derive(Debug, Clone)]
pub(crate) struct InlinedFeedRows {
    held: KeyedInlinedRows,
    rest: Option<InlinedRest>,
}

/// The pages of a selection after those the plan holds, and the pinned read
/// they are read through.
#[derive(Debug, Clone)]
struct InlinedRest {
    pages: Arc<InlinedPages>,
    read: Arc<ConsistentRead>,
}

/// What the pages of a selection after those the plan holds are read with.
#[derive(Debug)]
struct InlinedPages {
    table_schema: SchemaRef,
    /// The versions after the last the held page ends with.
    request: InlinedCursorRequest,
}

fn row_count(rows: &[DuckLakeInlinedRows]) -> usize {
    rows.iter().map(|rows| rows.row_ids.len()).sum()
}

impl InlinedFeedRows {
    /// No row versions.
    pub(crate) fn empty(charge: &PlanCharge) -> Self {
        Self {
            held: KeyedInlinedRows::empty(charge),
            rest: None,
        }
    }

    /// Read the versions `selection` picks (narrowed by `filter`, which may
    /// only ever drop versions a filter above the feed rejects): the first page
    /// now, inside the plan's consistent read, and the rest when the plan
    /// executes, through the source's pinned read, or now as well when no read
    /// can be pinned. What is read now is charged as it returns.
    pub(crate) fn read(
        source: &InlinedSource<'_>,
        selection: InlinedRowSelection,
        filter: Option<&InlinedFilter>,
    ) -> DataFusionResult<Self> {
        let page_rows = source.page_rows.max(1);
        let read_page = |after: Option<&InlinedRowsAfter>, limit: usize| {
            let rows = source
                .reader
                .read_inlined_rows_page(
                    source.table_id,
                    source.columns,
                    selection,
                    filter,
                    after,
                    limit,
                )
                .map_err(external)?;
            source.charge.try_grow(inlined_rows_bytes(&rows))?;
            Ok::<_, DataFusionError>(rows)
        };
        let mut rows = read_page(None, page_rows)?;
        let mut rest = None;
        if row_count(&rows) >= page_rows
            && let Some(after) = InlinedRowsAfter::after_last(&rows)
        {
            match source.pinned.get(source.reader)? {
                Some(read) => {
                    rest = Some(InlinedRest {
                        pages: Arc::new(InlinedPages {
                            table_schema: Arc::clone(source.table_schema),
                            request: InlinedCursorRequest {
                                table_id: source.table_id,
                                columns: Arc::clone(source.columns),
                                selection,
                                filter: filter.cloned(),
                                after,
                                page_rows,
                            },
                        }),
                        read,
                    })
                },
                // No read to page through later: read the rest here, inside
                // the plan's consistent read, in one statement per inlined
                // table (one per page would scan the table once a page).
                None => rows.extend(read_page(Some(&after), usize::MAX)?),
            }
        }
        let read = inlined_rows_bytes(&rows);
        let held = KeyedInlinedRows::new(rows, source.table_schema, source.charge)?;
        release(source.charge, read.saturating_sub(held.bytes()));
        Ok(Self {
            held,
            rest,
        })
    }

    /// Whether the selection holds no version. A selection with pages left
    /// holds a full first page.
    pub(crate) fn is_empty(&self) -> bool {
        self.held.is_empty()
    }

    /// The bytes the plan holds.
    pub(crate) fn bytes(&self) -> usize {
        self.held.bytes()
    }

    /// The versions as `[snapshot_id, rowid, the table columns at
    /// table_indices]`: the held page, then the rest, read as the plan
    /// executes.
    pub(crate) fn exec(&self, table_indices: &[usize]) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let held = self.held.exec(table_indices)?;
        let Some(rest) = &self.rest else {
            return Ok(held);
        };
        let pages: Arc<dyn ExecutionPlan> = Arc::new(InlinedPagesExec::new(
            Arc::clone(&rest.pages),
            Arc::clone(&rest.read),
            table_indices.to_vec(),
        ));
        datafusion::physical_plan::union::UnionExec::try_new(vec![held, pages])
    }
}

/// Reads the pages of a selection's inlined row versions after those the
/// plan holds, one at a time, as `[snapshot_id, rowid, projected table
/// columns]`, through the read pinned to the plan's state (see the module
/// documentation), streaming them from the cursor that read opens.
///
/// The node executes once. Its first execution takes the node's handle to
/// the pinned read and lets go of it after the last page; executing it again
/// (re-running a collected plan, or a caller that executes a plan more than
/// once) fails with an error that asks for the query to be planned again,
/// since the pinned read, and with it the catalog state the plan was built
/// against, is gone.
#[derive(Debug)]
pub(crate) struct InlinedPagesExec {
    rest: Arc<InlinedPages>,
    /// This node's handle to the pinned read, until its execution takes it.
    read: Mutex<Option<Arc<ConsistentRead>>>,
    table_indices: Vec<usize>,
    schema: SchemaRef,
    properties: Arc<PlanProperties>,
}

impl InlinedPagesExec {
    fn new(rest: Arc<InlinedPages>, read: Arc<ConsistentRead>, table_indices: Vec<usize>) -> Self {
        let schema = keyed_rows_schema(&rest.table_schema, &table_indices);
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(Arc::clone(&schema)),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ));
        Self {
            rest,
            read: Mutex::new(Some(read)),
            table_indices,
            schema,
            properties,
        }
    }
}

impl DisplayAs for InlinedPagesExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "InlinedPagesExec: table={}, page_rows={}",
            self.rest.request.table_id, self.rest.request.page_rows
        )
    }
}

impl ExecutionPlan for InlinedPagesExec {
    fn name(&self) -> &str {
        "InlinedPagesExec"
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
                "InlinedPagesExec has no children".to_string(),
            ));
        }
        Ok(self)
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(DataFusionError::Internal(format!(
                "InlinedPagesExec has 1 partition, got {partition}"
            )));
        }
        let rest = Arc::clone(&self.rest);
        let read = self
            .read
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take()
            .ok_or_else(|| {
                DataFusionError::Execution(
                    "this change feed's plan has already read the inlined rows of its window, \
                     and reads them once; plan the query again to run it again"
                        .to_string(),
                )
            })?;
        // Each page is charged while the next is read, so at most one page of
        // the catalog's rows is held here beyond what the plan consumes.
        let reservation = MemoryConsumer::new("ducklake change feed inlined pages")
            .register(context.memory_pool());
        let all: Vec<usize> = (0..rest.table_schema.fields().len()).collect();
        let full_schema = keyed_rows_schema(&rest.table_schema, &all);
        let indices: Vec<usize> = [0, 1]
            .into_iter()
            .chain(self.table_indices.iter().map(|i| i + 2))
            .collect();
        // The stream owns the pinned read, and the cursor open in it, until
        // the last page: an error or a dropped stream drops both with the
        // stream's state. The cursor opens on the first poll.
        let start: PagesState = Some((None, read));
        let pages =
            futures::stream::try_unfold((start, reservation), move |(state, reservation)| {
                let rest = Arc::clone(&rest);
                let full_schema = Arc::clone(&full_schema);
                let indices = indices.clone();
                async move {
                    let Some((cursor, read)) = state else {
                        return Ok(None);
                    };
                    let mut cursor = match cursor {
                        Some(cursor) => cursor,
                        None => read
                            .open_inlined_rows_cursor(rest.request.clone())
                            .map_err(external)?,
                    };
                    let rows = cursor.next_page().map_err(external)?;
                    reservation.try_resize(inlined_rows_bytes(&rows))?;
                    let state = if row_count(&rows) >= rest.request.page_rows {
                        Some((Some(cursor), read))
                    } else {
                        // The last page: the read ends now, not when the
                        // stream is next polled or dropped.
                        drop(cursor);
                        release_pinned(read)?;
                        None
                    };
                    let mut batches: Vec<RecordBatch> = Vec::with_capacity(rows.len());
                    for rows in rows {
                        if rows.row_ids.is_empty() {
                            continue;
                        }
                        batches.push(keyed_inlined_batch(rows, &full_schema)?.project(&indices)?);
                    }
                    Ok::<_, DataFusionError>(Some((batches, (state, reservation))))
                }
            });
        let stream = pages
            .map_ok(|batches| futures::stream::iter(batches.into_iter().map(Ok)))
            .try_flatten();
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&self.schema),
            stream.boxed(),
        )))
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}
