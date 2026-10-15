//! [`CatalogReadMemo`]: an opt-in memo of the catalog reads of table scans,
//! kept across queries. Its documentation says how it differs from official
//! DuckLake, and which facts it depends on.
//!
//! For maintainers: keep the memo here, in the provider wrapper.
//! `DuckLakeTable::scan` has no memo code, and needs none. Forward each new
//! `MetadataProvider` method in [`MemoizedMetadataProvider`]: a method left out
//! runs the trait default instead of the wrapped provider's own method. The
//! test `the_bound_provider_overrides_every_trait_method` checks this.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::mem::size_of;
use std::sync::{Arc, Mutex, MutexGuard};

use arrow::record_batch::RecordBatch;

use crate::Result;
use crate::inlined_filter::{InlinedDataScan, InlinedFilter};
use crate::metadata_provider::{
    ColumnTag, ColumnWithTable, DataFileChange, DeleteFileChange, DuckLakeFileData,
    DuckLakeFileMetadata, DuckLakeInlinedData, DuckLakeInlinedDelete,
    DuckLakeInlinedFileDeleteForFlush, DuckLakeInlinedFlushData, DuckLakeInlinedRows,
    DuckLakeNameMapping, DuckLakeStatistics, DuckLakeTableColumn, DuckLakeTableField,
    DuckLakeTableFile, DuckLakeTag, FILE_METADATA_BATCH_SIZE, FileWithTable,
    InlinedFileDeleteChange, InlinedRowSelection, MetadataProvider, ObjectTag, SchemaMetadata,
    SnapshotChangeMetadata, SnapshotMetadata, TableMetadata, TableWithSchema, TagTarget,
    ViewMetadata, ViewWithSchema,
};
use crate::partition::PartitionSpec;
use crate::sort::SortSpec;
use crate::stats_filter::StatsFilter;

/// The catalog reads of table scans, kept across queries for one catalog.
///
/// A built [`DuckLakeTable`](crate::DuckLakeTable) reads the catalog on every
/// scan: the file listing with each file's column statistics and partition
/// values, the inlined deletions and the inlined rows (with their row ids for a
/// `rowid` scan). A caller that keeps built tables across queries pays these
/// reads on every query. The memo keeps them instead. The caller binds the memo
/// to its provider once per query, and gives the bound provider to the tables
/// of that query:
///
/// ```rust,ignore
/// let memo = CatalogReadMemo::new(CatalogReadMemo::DEFAULT_MAX_BYTES);
/// // For each query:
/// let provider = memo.bind(Arc::clone(&provider))?; // reads the catalog head once
/// let table = cached_table.with_metadata_provider(Arc::new(provider));
/// // or: let catalog = DuckLakeCatalog::new(provider)?;
/// ```
///
/// The memo lives in the provider and not in the scan. `DuckLakeTable::scan`
/// has no memo code: it makes the same provider calls with or without a memo.
///
/// # How this differs from DuckLake
///
/// Official DuckLake keeps no file listing between transactions. Each scan
/// lists its files again, at the snapshot that its transaction pinned. This
/// memo keeps a table's reads across queries, and checks only the catalog head
/// (the newest snapshot id) before it serves them. It is correct only while
/// these facts hold:
///
/// 1. Every commit that changes what a scan reads also moves the head. An
///    insert, a delete, an update, a merge of adjacent files, a rewrite of data
///    files and a flush of inlined data each commit a new snapshot, in official
///    DuckLake and in this crate. The merge, the rewrite and the flush also
///    change the listing of older snapshots. A writer that changes the catalog
///    tables without a new snapshot breaks the memo.
/// 2. Snapshot ids only grow. A bind that reads a newer head drops every
///    entry, because no entry can serve that head.
/// 3. Expiring snapshots and cleaning up old files commit no snapshot, so they
///    do not move the head. A memo entry of an expired snapshot keeps the files
///    of its fill, and a read fails once cleanup deletes them. Official DuckLake
///    refuses a read at an expired snapshot ("No snapshot found"). Entries of
///    the head snapshot cannot be affected, because the head cannot expire.
/// 4. A key is a table id and a snapshot id, so one memo serves one catalog.
///
/// # Consistency
///
/// A fill reads all of a table's scan-time reads together: the listing, the
/// inlined deletions and the inlined rows, with and without row ids. Every
/// served call of the table takes its answer from one fill. The reads are separate statements, so a
/// commit between two of them can split them: a flush can move a row from the
/// inlined rows into a file between the two reads. The memo keeps a fill only
/// when the head after the fill is still the head that the bind read. So
/// every kept fill shows the catalog at that one head. A fill during which the
/// head moved serves its own query and is not kept, which is the risk that
/// every scan without a memo takes too.
///
/// # What the bound provider serves
///
/// - `get_table_file_metadata_page` and `get_table_file_metadata_page_filtered`
///   serve pages of the whole listing. The statistics filter is advisory in the
///   trait, and the scan prunes the listing in memory.
/// - `get_inlined_deletes` serves the inlined deletions.
/// - `scan_inlined_data` and `scan_inlined_data_noting_absence` serve every
///   visible inlined row, unfiltered, which the trait permits. The scan applies
///   its filters above them.
/// - `read_inlined_rows` with [`InlinedRowSelection::VisibleAt`] serves the same
///   rows with their row ids, unfiltered, for a `rowid` scan. Other selections
///   go to the wrapped provider.
///
/// Every other method, `get_current_snapshot` included, goes to the wrapped
/// provider. A write must act on the current layout, so give a bound provider
/// to read-only tables only.
///
/// # Cost
///
/// A bind reads the head: one statement per query. A fill reads more than a
/// scan without a memo does: the whole listing without the scan's filters, the
/// table's columns, the inlined rows with their row ids (unless the provider
/// showed that there are none), and the head after the fill. A commit to any
/// table of the catalog moves the head and empties the memo. On a catalog that commits
/// between most queries, the memo costs more than it saves.
///
/// # Budget
///
/// [`Self::new`] takes the bytes the memo can keep, for all of its tables
/// together. Byte counts are estimates of the heap memory held. A table whose
/// reads do not fit the whole budget is not memoized: its scans read the
/// catalog, as without a memo, and the memo does not try to fill it again. A
/// table whose reads fit the budget, but not what is left of it, is read
/// again by each query until a newer head empties the memo.
pub struct CatalogReadMemo {
    max_bytes: usize,
    state: Mutex<MemoState>,
}

/// A table id and a snapshot id.
type TableKey = (i64, i64);

#[derive(Default)]
struct MemoState {
    /// The newest head a bind read. Every kept fill shows the catalog at it.
    head: Option<i64>,
    bytes: usize,
    tables: HashMap<TableKey, Arc<TableReads>>,
    /// Tables whose reads did not fit the whole budget.
    too_large: HashSet<TableKey>,
}

/// The catalog reads of one table at one snapshot, read together.
struct TableReads {
    /// The whole listing, ordered by data file id.
    listing: Vec<DuckLakeFileMetadata>,
    inlined_deletes: Vec<DuckLakeInlinedDelete>,
    /// The columns the inlined rows were read in.
    columns: Vec<DuckLakeTableColumn>,
    /// Every visible inlined row, unfiltered.
    inlined_rows: Vec<RecordBatch>,
    /// The same rows with their row ids, as `read_inlined_rows` reads them at
    /// this snapshot. `None` when the provider could not read them: a `rowid`
    /// scan then reads the catalog.
    visible_rows: Option<Vec<DuckLakeInlinedRows>>,
    /// What the provider said about inlined rows through this snapshot (see
    /// [`MetadataProvider::scan_inlined_data_noting_absence`]).
    inlined_absent_through: Option<i64>,
}

impl CatalogReadMemo {
    /// The budget for a memo of typical size: 64 MiB, about 20,000 data files
    /// with statistics on 20 columns each.
    pub const DEFAULT_MAX_BYTES: usize = 64 << 20;

    /// An empty memo that keeps up to `max_bytes` of catalog reads.
    #[must_use]
    pub fn new(max_bytes: usize) -> Arc<Self> {
        Arc::new(Self {
            max_bytes,
            state: Mutex::new(MemoState::default()),
        })
    }

    /// A provider for one query: it reads through `provider`, and serves the
    /// catalog reads of table scans from this memo.
    ///
    /// This reads the catalog head once, through `provider`. A head newer than
    /// the memo's drops every entry. Bind once per query, not once for many
    /// queries: the bound provider checks its entries against this head only.
    /// `provider` must read the catalog this memo serves.
    pub fn bind(
        self: &Arc<Self>,
        provider: Arc<dyn MetadataProvider>,
    ) -> Result<MemoizedMetadataProvider> {
        let head = provider.get_current_snapshot()?;
        {
            let mut state = self.lock();
            if state.head.is_none_or(|kept| kept < head) {
                state.head = Some(head);
                state.tables.clear();
                state.bytes = 0;
            }
        }
        Ok(MemoizedMetadataProvider {
            memo: Arc::clone(self),
            inner: provider,
            head,
            own_reads: Mutex::default(),
        })
    }

    fn lock(&self) -> MutexGuard<'_, MemoState> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

impl fmt::Debug for CatalogReadMemo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let state = self.lock();
        f.debug_struct("CatalogReadMemo")
            .field("max_bytes", &self.max_bytes)
            .field("head", &state.head)
            .field("bytes", &state.bytes)
            .field("tables", &state.tables.len())
            .finish()
    }
}

/// A provider for one query, which [`CatalogReadMemo::bind`] returns: it
/// serves the catalog reads of table scans from the memo, and passes every
/// other call to the provider it wraps.
pub struct MemoizedMetadataProvider {
    memo: Arc<CatalogReadMemo>,
    inner: Arc<dyn MetadataProvider>,
    /// The head the bind read.
    head: i64,
    /// Fills during which the head moved: they serve this query alone.
    own_reads: Mutex<HashMap<TableKey, Arc<TableReads>>>,
}

impl fmt::Debug for MemoizedMetadataProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MemoizedMetadataProvider")
            .field("head", &self.head)
            .field("memo", &self.memo)
            .field("inner", &self.inner)
            .finish()
    }
}

impl MemoizedMetadataProvider {
    /// The reads of `table_id` at `snapshot_id`: a fill kept at this query's
    /// head, else a fill of this query, else a new fill. `None` when they do
    /// not fit the memo's budget, and the caller reads the catalog itself.
    fn table_reads(&self, table_id: i64, snapshot_id: i64) -> Result<Option<Arc<TableReads>>> {
        let key = (table_id, snapshot_id);
        {
            let state = self.memo.lock();
            if state.too_large.contains(&key) {
                return Ok(None);
            }
            if state.head == Some(self.head)
                && let Some(reads) = state.tables.get(&key)
            {
                return Ok(Some(Arc::clone(reads)));
            }
        }
        if let Some(reads) = self.own_reads().get(&key) {
            return Ok(Some(Arc::clone(reads)));
        }
        let Some((reads, bytes)) = self.fill(table_id, snapshot_id)? else {
            self.memo.lock().too_large.insert(key);
            return Ok(None);
        };
        let reads = Arc::new(reads);
        // Every commit adds a snapshot, so a head that has not moved since the
        // bind means that no commit landed before the fill ended. A head that
        // cannot be read counts as moved.
        if self.inner.get_current_snapshot().ok() == Some(self.head) {
            let mut state = self.memo.lock();
            if state.head == Some(self.head) {
                if let Some(kept) = state.tables.get(&key) {
                    return Ok(Some(Arc::clone(kept)));
                }
                if state.bytes.saturating_add(bytes) <= self.memo.max_bytes {
                    state.bytes += bytes;
                    state.tables.insert(key, Arc::clone(&reads));
                    return Ok(Some(reads));
                }
            }
        }
        self.own_reads().insert(key, Arc::clone(&reads));
        Ok(Some(reads))
    }

    /// Read everything a scan of `table_id` at `snapshot_id` reads from the
    /// catalog, without the scan's filters, so that the reads serve any later
    /// scan. `None` once the reads exceed the memo's whole budget.
    fn fill(&self, table_id: i64, snapshot_id: i64) -> Result<Option<(TableReads, usize)>> {
        let max_bytes = self.memo.max_bytes;
        let mut bytes = 0usize;
        let mut listing = Vec::new();
        let mut after = None;
        // Page as `DuckLakeTable` does, until a short page.
        loop {
            let page = self.inner.get_table_file_metadata_page_filtered(
                table_id,
                snapshot_id,
                after,
                FILE_METADATA_BATCH_SIZE,
                None,
            )?;
            let Some(last) = page.last() else {
                break;
            };
            let next = last.file.data_file_id;
            if page.len() > FILE_METADATA_BATCH_SIZE || after.is_some_and(|after| next <= after) {
                return Err(crate::DuckLakeError::InvalidConfig(format!(
                    "metadata provider returned a {}-file page after file {after:?} for a \
                     {FILE_METADATA_BATCH_SIZE}-file page, ending at file {next}",
                    page.len(),
                )));
            }
            after = Some(next);
            bytes = bytes.saturating_add(page.iter().map(file_metadata_bytes).sum());
            if bytes > max_bytes {
                return Ok(None);
            }
            let short = page.len() < FILE_METADATA_BATCH_SIZE;
            listing.extend(page);
            if short {
                break;
            }
        }
        let inlined_deletes = self.inner.get_inlined_deletes(table_id, snapshot_id)?;
        bytes = bytes.saturating_add(inlined_deletes.len() * size_of::<DuckLakeInlinedDelete>());
        if bytes > max_bytes {
            return Ok(None);
        }
        // `DuckLakeTable` reads its inlined rows in these columns.
        let columns = self.inner.get_table_structure(table_id, snapshot_id)?;
        let (inlined, inlined_absent_through) =
            self.inner
                .scan_inlined_data_noting_absence(table_id, snapshot_id, &columns, None)?;
        bytes = bytes.saturating_add(
            inlined
                .batches
                .iter()
                .map(RecordBatch::get_array_memory_size)
                .sum(),
        );
        if bytes > max_bytes {
            return Ok(None);
        }
        // A `rowid` scan reads the same rows with their row ids. Read them here
        // too, unless the provider showed that there are none, so that every
        // read of the scan comes from this fill. A provider that cannot read them
        // fails the `rowid` scan alone, as it does without a memo.
        let visible_rows = if inlined_absent_through.is_some_and(|through| snapshot_id <= through) {
            Some(Vec::new())
        } else {
            self.inner
                .read_inlined_rows(
                    table_id,
                    &columns,
                    InlinedRowSelection::VisibleAt(snapshot_id),
                    None,
                )
                .ok()
        };
        bytes = bytes.saturating_add(visible_rows.iter().flatten().map(inlined_rows_bytes).sum());
        if bytes > max_bytes {
            return Ok(None);
        }
        Ok(Some((
            TableReads {
                listing,
                inlined_deletes,
                columns,
                inlined_rows: inlined.batches,
                inlined_absent_through,
                visible_rows,
            },
            bytes,
        )))
    }

    fn own_reads(&self) -> MutexGuard<'_, HashMap<TableKey, Arc<TableReads>>> {
        self.own_reads
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

/// The files of `listing` after `after`, at most `limit` of them.
fn listing_page(
    listing: &[DuckLakeFileMetadata],
    after: Option<i64>,
    limit: usize,
) -> Vec<DuckLakeFileMetadata> {
    let start = after.map_or(0, |after| {
        listing.partition_point(|metadata| metadata.file.data_file_id <= after)
    });
    listing[start..].iter().take(limit).cloned().collect()
}

/// Estimated memory one file of a listing holds.
fn file_metadata_bytes(metadata: &DuckLakeFileMetadata) -> usize {
    let file = &metadata.file;
    let delete_file = file.delete_file.as_ref().map_or(0, |delete_file| {
        size_of::<DuckLakeFileData>() + file_data_heap_bytes(delete_file)
    });
    let partition_values = file
        .partition_values
        .iter()
        .map(|(_, value)| {
            size_of::<(i32, Option<String>)>() + value.as_ref().map_or(0, String::len)
        })
        .sum::<usize>();
    let statistics = metadata
        .column_statistics
        .iter()
        .map(|statistic| {
            size_of_val(statistic)
                + statistic.min_value.as_ref().map_or(0, String::len)
                + statistic.max_value.as_ref().map_or(0, String::len)
        })
        .sum::<usize>();
    size_of::<DuckLakeFileMetadata>()
        + file_data_heap_bytes(&file.file)
        + delete_file
        + partition_values
        + statistics
}

/// Estimated memory one physical table's inlined rows hold, with their ids.
fn inlined_rows_bytes(rows: &DuckLakeInlinedRows) -> usize {
    size_of::<DuckLakeInlinedRows>()
        + rows.table_name.len()
        + rows.batch.get_array_memory_size()
        + (rows.row_ids.len() + rows.snapshots.len()) * size_of::<i64>()
        + rows.end_snapshots.len() * size_of::<Option<i64>>()
}

fn file_data_heap_bytes(file: &DuckLakeFileData) -> usize {
    file.path.len() + file.encryption_key.as_ref().map_or(0, String::len)
}

impl MetadataProvider for MemoizedMetadataProvider {
    fn get_current_snapshot(&self) -> Result<i64> {
        self.inner.get_current_snapshot()
    }

    fn get_data_path(&self) -> Result<String> {
        self.inner.get_data_path()
    }

    fn get_metadata_settings(
        &self,
        schema_id: Option<i64>,
        table_id: Option<i64>,
    ) -> Result<HashMap<String, String>> {
        self.inner.get_metadata_settings(schema_id, table_id)
    }

    fn list_snapshots(&self) -> Result<Vec<SnapshotMetadata>> {
        self.inner.list_snapshots()
    }

    fn list_snapshot_changes(&self) -> Result<Vec<SnapshotChangeMetadata>> {
        self.inner.list_snapshot_changes()
    }

    fn find_snapshot_by_commit_extra_info(&self, needle: &str) -> Result<Option<i64>> {
        self.inner.find_snapshot_by_commit_extra_info(needle)
    }

    fn list_schemas(&self, snapshot_id: i64) -> Result<Vec<SchemaMetadata>> {
        self.inner.list_schemas(snapshot_id)
    }

    fn list_tables(&self, schema_id: i64, snapshot_id: i64) -> Result<Vec<TableMetadata>> {
        self.inner.list_tables(schema_id, snapshot_id)
    }

    fn list_views(&self, schema_id: i64, snapshot_id: i64) -> Result<Vec<ViewMetadata>> {
        self.inner.list_views(schema_id, snapshot_id)
    }

    fn get_table_structure(
        &self,
        table_id: i64,
        snapshot_id: i64,
    ) -> Result<Vec<DuckLakeTableColumn>> {
        self.inner.get_table_structure(table_id, snapshot_id)
    }

    fn get_table_fields(&self, table_id: i64, snapshot_id: i64) -> Result<Vec<DuckLakeTableField>> {
        self.inner.get_table_fields(table_id, snapshot_id)
    }

    fn get_name_mapping(&self, mapping_id: i64) -> Result<DuckLakeNameMapping> {
        self.inner.get_name_mapping(mapping_id)
    }

    fn get_table_files_for_select(
        &self,
        table_id: i64,
        snapshot_id: i64,
    ) -> Result<Vec<DuckLakeTableFile>> {
        self.inner.get_table_files_for_select(table_id, snapshot_id)
    }

    fn get_partition_spec(&self, table_id: i64, snapshot_id: i64) -> Result<Option<PartitionSpec>> {
        self.inner.get_partition_spec(table_id, snapshot_id)
    }

    fn get_sort_spec(&self, table_id: i64, snapshot_id: i64) -> Result<Option<SortSpec>> {
        self.inner.get_sort_spec(table_id, snapshot_id)
    }

    fn get_table_statistics(&self, table_id: i64, snapshot_id: i64) -> Result<DuckLakeStatistics> {
        self.inner.get_table_statistics(table_id, snapshot_id)
    }

    fn get_table_summary_statistics(
        &self,
        table_id: i64,
        snapshot_id: i64,
    ) -> Result<DuckLakeStatistics> {
        self.inner
            .get_table_summary_statistics(table_id, snapshot_id)
    }

    fn get_table_file_metadata_page(
        &self,
        table_id: i64,
        snapshot_id: i64,
        after_data_file_id: Option<i64>,
        limit: usize,
    ) -> Result<Vec<DuckLakeFileMetadata>> {
        match self.table_reads(table_id, snapshot_id)? {
            Some(reads) => Ok(listing_page(&reads.listing, after_data_file_id, limit)),
            None => self.inner.get_table_file_metadata_page(
                table_id,
                snapshot_id,
                after_data_file_id,
                limit,
            ),
        }
    }

    fn get_table_file_metadata_page_filtered(
        &self,
        table_id: i64,
        snapshot_id: i64,
        after_data_file_id: Option<i64>,
        limit: usize,
        filter: Option<&StatsFilter>,
    ) -> Result<Vec<DuckLakeFileMetadata>> {
        // The filter is advisory: the whole listing is a correct answer.
        match self.table_reads(table_id, snapshot_id)? {
            Some(reads) => Ok(listing_page(&reads.listing, after_data_file_id, limit)),
            None => self.inner.get_table_file_metadata_page_filtered(
                table_id,
                snapshot_id,
                after_data_file_id,
                limit,
                filter,
            ),
        }
    }

    fn get_inlined_data(
        &self,
        table_id: i64,
        snapshot_id: i64,
        columns: &[DuckLakeTableColumn],
    ) -> Result<Vec<RecordBatch>> {
        self.inner.get_inlined_data(table_id, snapshot_id, columns)
    }

    fn scan_inlined_data(
        &self,
        table_id: i64,
        snapshot_id: i64,
        columns: &[DuckLakeTableColumn],
        filter: Option<&InlinedFilter>,
    ) -> Result<InlinedDataScan> {
        Ok(self
            .scan_inlined_data_noting_absence(table_id, snapshot_id, columns, filter)?
            .0)
    }

    fn scan_inlined_data_noting_absence(
        &self,
        table_id: i64,
        snapshot_id: i64,
        columns: &[DuckLakeTableColumn],
        filter: Option<&InlinedFilter>,
    ) -> Result<(InlinedDataScan, Option<i64>)> {
        // The filter is advisory: every visible row is a correct answer. Rows in
        // other columns than the fill's are read from the catalog.
        if let Some(reads) = self.table_reads(table_id, snapshot_id)?
            && reads.columns == columns
        {
            let scan = InlinedDataScan {
                batches: reads.inlined_rows.clone(),
                materialized_row_count: 0,
            };
            return Ok((scan, reads.inlined_absent_through));
        }
        self.inner
            .scan_inlined_data_noting_absence(table_id, snapshot_id, columns, filter)
    }

    fn get_inlined_deletes(
        &self,
        table_id: i64,
        snapshot_id: i64,
    ) -> Result<Vec<DuckLakeInlinedDelete>> {
        match self.table_reads(table_id, snapshot_id)? {
            Some(reads) => Ok(reads.inlined_deletes.clone()),
            None => self.inner.get_inlined_deletes(table_id, snapshot_id),
        }
    }

    fn get_delete_file_begin_snapshot(&self, delete_file_id: i64) -> Result<Option<i64>> {
        self.inner.get_delete_file_begin_snapshot(delete_file_id)
    }

    fn get_inlined_data_with_row_ids(
        &self,
        table_id: i64,
        snapshot_id: i64,
        columns: &[DuckLakeTableColumn],
    ) -> Result<Vec<DuckLakeInlinedData>> {
        self.inner
            .get_inlined_data_with_row_ids(table_id, snapshot_id, columns)
    }

    fn get_inlined_data_for_flush(
        &self,
        table_id: i64,
        snapshot_id: i64,
    ) -> Result<Vec<DuckLakeInlinedFlushData>> {
        self.inner.get_inlined_data_for_flush(table_id, snapshot_id)
    }

    fn get_inlined_file_deletes_for_flush(
        &self,
        table_id: i64,
        snapshot_id: i64,
    ) -> Result<Vec<DuckLakeInlinedFileDeleteForFlush>> {
        self.inner
            .get_inlined_file_deletes_for_flush(table_id, snapshot_id)
    }

    fn read_inlined_rows(
        &self,
        table_id: i64,
        columns: &[DuckLakeTableColumn],
        selection: InlinedRowSelection,
        filter: Option<&InlinedFilter>,
    ) -> Result<Vec<DuckLakeInlinedRows>> {
        // The filter is advisory: every visible row is a correct answer.
        if let InlinedRowSelection::VisibleAt(snapshot_id) = selection
            && let Some(reads) = self.table_reads(table_id, snapshot_id)?
            && reads.columns == columns
            && let Some(rows) = &reads.visible_rows
        {
            return Ok(rows.clone());
        }
        self.inner
            .read_inlined_rows(table_id, columns, selection, filter)
    }

    fn get_inlined_file_deletes_between(
        &self,
        table_id: i64,
        start_snapshot: i64,
        end_snapshot: i64,
    ) -> Result<Vec<InlinedFileDeleteChange>> {
        self.inner
            .get_inlined_file_deletes_between(table_id, start_snapshot, end_snapshot)
    }

    fn get_table_row_count(&self, table_id: i64, snapshot_id: i64) -> Result<u64> {
        self.inner.get_table_row_count(table_id, snapshot_id)
    }

    fn get_schema_by_name(&self, name: &str, snapshot_id: i64) -> Result<Option<SchemaMetadata>> {
        self.inner.get_schema_by_name(name, snapshot_id)
    }

    fn get_table_by_name(
        &self,
        schema_id: i64,
        name: &str,
        snapshot_id: i64,
    ) -> Result<Option<TableMetadata>> {
        self.inner.get_table_by_name(schema_id, name, snapshot_id)
    }

    fn get_view_by_name(
        &self,
        schema_id: i64,
        name: &str,
        snapshot_id: i64,
    ) -> Result<Option<ViewMetadata>> {
        self.inner.get_view_by_name(schema_id, name, snapshot_id)
    }

    fn table_exists(&self, schema_id: i64, name: &str, snapshot_id: i64) -> Result<bool> {
        self.inner.table_exists(schema_id, name, snapshot_id)
    }

    fn get_tags(&self, target: TagTarget, snapshot_id: i64) -> Result<Vec<DuckLakeTag>> {
        self.inner.get_tags(target, snapshot_id)
    }

    fn get_view_id_by_name(
        &self,
        schema_id: i64,
        name: &str,
        snapshot_id: i64,
    ) -> Result<Option<i64>> {
        self.inner.get_view_id_by_name(schema_id, name, snapshot_id)
    }

    fn list_all_tables(&self, snapshot_id: i64) -> Result<Vec<TableWithSchema>> {
        self.inner.list_all_tables(snapshot_id)
    }

    fn list_all_views(&self, snapshot_id: i64) -> Result<Vec<ViewWithSchema>> {
        self.inner.list_all_views(snapshot_id)
    }

    fn list_all_columns(&self, snapshot_id: i64) -> Result<Vec<ColumnWithTable>> {
        self.inner.list_all_columns(snapshot_id)
    }

    fn list_all_object_tags(&self, snapshot_id: i64) -> Result<Vec<ObjectTag>> {
        self.inner.list_all_object_tags(snapshot_id)
    }

    fn list_all_column_tags(&self, snapshot_id: i64) -> Result<Vec<ColumnTag>> {
        self.inner.list_all_column_tags(snapshot_id)
    }

    fn list_all_files(&self, snapshot_id: i64) -> Result<Vec<FileWithTable>> {
        self.inner.list_all_files(snapshot_id)
    }

    fn get_data_files_added_between_snapshots(
        &self,
        table_id: i64,
        start_snapshot: i64,
        end_snapshot: i64,
    ) -> Result<Vec<DataFileChange>> {
        self.inner
            .get_data_files_added_between_snapshots(table_id, start_snapshot, end_snapshot)
    }

    fn get_delete_files_added_between_snapshots(
        &self,
        table_id: i64,
        start_snapshot: i64,
        end_snapshot: i64,
    ) -> Result<Vec<DeleteFileChange>> {
        self.inner
            .get_delete_files_added_between_snapshots(table_id, start_snapshot, end_snapshot)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    /// The names of the methods declared in the block of `source` that starts
    /// with `header`, at the block's own indentation.
    fn method_names(source: &str, header: &str) -> BTreeSet<String> {
        let start = source.find(header).expect("the block exists");
        let block = &source[start..];
        let end = block.find("\n}\n").expect("the block ends");
        block[..end]
            .lines()
            .filter_map(|line| line.strip_prefix("    fn "))
            .map(|rest| {
                rest.split(|c: char| !(c.is_alphanumeric() || c == '_'))
                    .next()
                    .unwrap_or_default()
                    .to_string()
            })
            .collect()
    }

    /// A trait method with a default body that the bound provider does not
    /// override runs that default, not the wrapped provider's own method. For
    /// most methods the default returns nothing or an error, so a memoized
    /// read would silently differ from an unmemoized one. Add every new trait
    /// method to `impl MetadataProvider for MemoizedMetadataProvider`.
    #[test]
    fn the_bound_provider_overrides_every_trait_method() {
        let declared = method_names(
            include_str!("metadata_provider.rs"),
            "pub trait MetadataProvider",
        );
        let overridden = method_names(
            include_str!("catalog_read_memo.rs"),
            "impl MetadataProvider for MemoizedMetadataProvider",
        );
        assert!(
            declared.len() > 40,
            "found the trait's methods: {declared:?}"
        );
        assert_eq!(declared, overridden);
    }

    fn file(data_file_id: i64) -> DuckLakeFileMetadata {
        let mut file = DuckLakeTableFile::new(DuckLakeFileData::new(
            format!("{data_file_id}.parquet"),
            true,
            1,
        ));
        file.data_file_id = data_file_id;
        DuckLakeFileMetadata {
            file,
            column_statistics: Vec::new(),
        }
    }

    #[test]
    fn a_listing_page_starts_after_the_cursor() {
        let listing = [2, 5, 9, 12].map(file);
        let ids = |page: Vec<DuckLakeFileMetadata>| {
            page.iter()
                .map(|metadata| metadata.file.data_file_id)
                .collect::<Vec<_>>()
        };
        assert_eq!(ids(listing_page(&listing, None, 3)), [2, 5, 9]);
        assert_eq!(ids(listing_page(&listing, Some(5), 3)), [9, 12]);
        assert_eq!(ids(listing_page(&listing, Some(6), 1)), [9]);
        assert!(listing_page(&listing, Some(12), 3).is_empty());
    }
}
