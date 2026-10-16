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
/// 1. Every commit that changes what a scan reads also moves the head. SQL
///    writes (insert, delete, update) and the maintenance calls (merge adjacent
///    files, rewrite data files, flush inlined data) each commit a new snapshot
///    together with their changes, in official DuckLake and in this crate. The
///    merge, the rewrite and the flush also change the listing of older
///    snapshots.
///
///    The low-level [`MetadataWriter`](crate::metadata_writer::MetadataWriter)
///    API does not keep this fact. `create_snapshot` moves the head, and later
///    calls such as `register_data_file` or `end_table_files` change the
///    catalog at that snapshot without moving the head again. A memo filled
///    between them misses those changes until the next commit. Any other writer
///    that changes the catalog tables without a new snapshot breaks the memo in
///    the same way.
/// 2. Snapshot ids only grow. A bind that reads a newer head drops every
///    entry, because no entry can serve that head.
/// 3. Expiring snapshots and cleaning up old files commit no snapshot, so they
///    do not move the head. Official DuckLake refuses a read at an expired
///    snapshot ("No snapshot found"). Without a memo, this crate returns no rows
///    for it. With a memo, a kept entry of the expired snapshot returns the old
///    rows until cleanup deletes their files, and then the read fails. Entries
///    of the head snapshot are safe, because the head cannot expire.
/// 4. A key is a table id and a snapshot id, with no catalog identity. Use one
///    memo per catalog. A catalog that is restored to an older head and then
///    committed up to the same snapshot id again gets the old entries.
///
/// Two answers stay across heads, so that a fill after a commit reads less:
///
/// - A table's columns at a snapshot. A change to the columns commits a new
///   snapshot, so the columns at an older one do not change. The memo serves
///   inlined rows only to a scan whose columns are equal to the kept ones. A
///   scan with other columns reads the catalog, and its columns replace the
///   kept ones.
/// - The provider's answer that a table has no inlined rows through a snapshot.
///   A writer adds inlined rows only at its commit's snapshot, which lies above
///   the head at the read, so the answer holds for later reads too.
///   `DuckLakeTable` keeps the same answer for the same reason.
///
/// # Consistency
///
/// A query uses one set of reads for each table. The first call of the query
/// for a table picks them, and every later call of that query, from any of its
/// scans, uses the same reads.
///
/// A fill reads in the order of a scan without a memo: the inlined deletions,
/// then the listing, then the inlined rows. The reads are separate statements,
/// so a commit between two of them can split them. The memo keeps a fill only
/// when the head after the fill is still the head that the bind read. So every
/// kept fill shows the catalog at that one head. A query whose own fill a
/// commit split uses a fill that another query kept at its head, when there is
/// one. Otherwise it uses its own fill, which then has the same split as a scan
/// without a memo.
///
/// # One bound provider per query
///
/// A bound provider keeps the reads that it picked, and checks nothing after
/// its bind. A provider kept past its query serves old reads, and the reads
/// that only it uses have no budget. Bind again for each query. A catalog over
/// a bound provider must also be built for each query.
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
/// to read-only tables only. Nothing stops `DuckLakeCatalog::with_writer` from
/// taking one.
///
/// # Cost
///
/// A bind reads the head: one statement per query. A fill reads more than a
/// scan without a memo does:
///
/// - the whole listing, without the scan's filters;
/// - the table's columns, the first time;
/// - the inlined rows with their row ids, unless the provider showed that
///   there are none;
/// - the head after the fill.
///
/// A commit to any table of the catalog moves the head and empties the memo.
/// On a catalog that commits between most queries, the memo costs more than it
/// saves.
///
/// # Budget
///
/// [`Self::new`] takes the bytes the memo can keep, for all of its tables
/// together. Byte counts are estimates of the heap memory held. The columns
/// and the answers about inlined rows that stay across heads are not counted:
/// the memo keeps one of each per table.
///
/// - A table whose reads do not fit the whole budget is not memoized. Its
///   scans read the catalog, as without a memo, and this memo does not fill
///   that table again, at any snapshot.
/// - A table whose reads fit the budget, but not what is left of it, uses its
///   fill for the query that read it. Later queries at the same head read the
///   catalog for it, as without a memo. A newer head empties the memo, and the
///   table can then fill again.
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
    /// Tables whose fill at this head did not fit what was left of the budget.
    no_room: HashSet<TableKey>,
    /// Tables, by id, whose reads did not fit the whole budget.
    too_large: HashSet<i64>,
    /// Kept across heads: each table's columns, by id, at the snapshot they
    /// were read for. See [`CatalogReadMemo`] for why they stay valid.
    columns: HashMap<i64, (i64, Arc<Vec<DuckLakeTableColumn>>)>,
    /// Kept across heads: for each table, by id, the latest snapshot through
    /// which the provider reported no inlined rows (see
    /// [`MetadataProvider::scan_inlined_data_noting_absence`]).
    inlined_absent_through: HashMap<i64, i64>,
}

/// The catalog reads of one table at one snapshot, read together.
struct TableReads {
    inlined_deletes: Vec<DuckLakeInlinedDelete>,
    /// The whole listing, ordered by data file id.
    listing: Vec<DuckLakeFileMetadata>,
    /// The columns the inlined rows were read in.
    columns: Arc<Vec<DuckLakeTableColumn>>,
    /// Every visible inlined row, unfiltered.
    inlined_rows: Vec<RecordBatch>,
    /// What the provider said about inlined rows through this snapshot (see
    /// [`MetadataProvider::scan_inlined_data_noting_absence`]).
    inlined_absent_through: Option<i64>,
    /// The same rows with their row ids, as `read_inlined_rows` reads them at
    /// this snapshot. `None` when the provider could not read them: a `rowid`
    /// scan then reads the catalog.
    visible_rows: Option<Vec<DuckLakeInlinedRows>>,
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
    /// queries: see [`CatalogReadMemo`]. `provider` must read the catalog this
    /// memo serves.
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
                state.no_room.clear();
                state.bytes = 0;
            }
        }
        Ok(MemoizedMetadataProvider {
            memo: Arc::clone(self),
            inner: provider,
            head,
            picked: Mutex::default(),
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
    /// The reads this query uses for each table, or `None` where it reads the
    /// catalog for that table.
    picked: Mutex<HashMap<TableKey, Option<Arc<TableReads>>>>,
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
    /// The reads this query uses for `table_id` at `snapshot_id`, or `None`
    /// when the query reads the catalog for that table. The first call of the
    /// query for the table picks them, and later calls get the same answer.
    fn table_reads(&self, table_id: i64, snapshot_id: i64) -> Result<Option<Arc<TableReads>>> {
        let key = (table_id, snapshot_id);
        if let Some(picked) = self.picked().get(&key) {
            return Ok(picked.clone());
        }
        let reads = self.pick(table_id, snapshot_id)?;
        // Two scans of the query can pick at the same time: the first one wins.
        Ok(self.picked().entry(key).or_insert(reads).clone())
    }

    /// A fill kept at this query's head, else a new fill, else `None` when the
    /// table does not fit the memo.
    fn pick(&self, table_id: i64, snapshot_id: i64) -> Result<Option<Arc<TableReads>>> {
        let key = (table_id, snapshot_id);
        {
            let state = self.memo.lock();
            if state.too_large.contains(&table_id) {
                return Ok(None);
            }
            if state.head == Some(self.head) {
                if let Some(kept) = state.tables.get(&key) {
                    return Ok(Some(Arc::clone(kept)));
                }
                if state.no_room.contains(&key) {
                    return Ok(None);
                }
            }
        }
        let Some((reads, bytes)) = self.fill(table_id, snapshot_id)? else {
            self.memo.lock().too_large.insert(table_id);
            return Ok(None);
        };
        let reads = Arc::new(reads);
        // Every commit adds a snapshot, so a head that has not moved since the
        // bind means that no commit landed before the fill ended. A head that
        // cannot be read counts as moved.
        let unmoved = self.inner.get_current_snapshot().ok() == Some(self.head);
        let mut state = self.memo.lock();
        if state.head == Some(self.head) {
            // Another query kept a fill at this head while this one read: it
            // shows the catalog at the head, which this fill may not.
            if let Some(kept) = state.tables.get(&key) {
                return Ok(Some(Arc::clone(kept)));
            }
            if unmoved {
                if state.bytes.saturating_add(bytes) <= self.memo.max_bytes {
                    state.bytes += bytes;
                    state.tables.insert(key, Arc::clone(&reads));
                } else {
                    state.no_room.insert(key);
                }
            }
        }
        Ok(Some(reads))
    }

    /// Read everything a scan of `table_id` at `snapshot_id` reads from the
    /// catalog, without the scan's filters, so that the reads serve any later
    /// scan. `None` once the reads exceed the memo's whole budget.
    ///
    /// The reads run in the order of a scan without a memo, so that a commit
    /// between two of them splits them as it splits such a scan: the inlined
    /// deletions, then the listing, then the inlined rows.
    fn fill(&self, table_id: i64, snapshot_id: i64) -> Result<Option<(TableReads, usize)>> {
        let max_bytes = self.memo.max_bytes;
        let inlined_deletes = self.inner.get_inlined_deletes(table_id, snapshot_id)?;
        let mut bytes = inlined_deletes.len() * size_of::<DuckLakeInlinedDelete>();
        if bytes > max_bytes {
            return Ok(None);
        }
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
        let columns = self.columns(table_id, snapshot_id)?;
        let known_absent = self
            .memo
            .lock()
            .inlined_absent_through
            .get(&table_id)
            .copied()
            .filter(|&through| snapshot_id <= through);
        let (inlined_rows, inlined_absent_through, visible_rows) = match known_absent {
            // An earlier fill showed that no inlined row exists at this snapshot.
            Some(through) => (Vec::new(), Some(through), Some(Vec::new())),
            None => {
                let (inlined, absent_through) = self.inner.scan_inlined_data_noting_absence(
                    table_id,
                    snapshot_id,
                    &columns,
                    None,
                )?;
                if let Some(through) = absent_through {
                    let mut state = self.memo.lock();
                    let kept = state
                        .inlined_absent_through
                        .entry(table_id)
                        .or_insert(through);
                    *kept = (*kept).max(through);
                }
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
                // A `rowid` scan reads the same rows with their row ids. Read
                // them here too, unless the provider showed that there are none,
                // so that every read of the scan comes from this fill. A provider
                // that cannot read them fails the `rowid` scan alone, as it does
                // without a memo.
                let visible_rows = if absent_through.is_some_and(|through| snapshot_id <= through) {
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
                (inlined.batches, absent_through, visible_rows)
            },
        };
        bytes = bytes.saturating_add(visible_rows.iter().flatten().map(inlined_rows_bytes).sum());
        if bytes > max_bytes {
            return Ok(None);
        }
        Ok(Some((
            TableReads {
                inlined_deletes,
                listing,
                columns,
                inlined_rows,
                inlined_absent_through,
                visible_rows,
            },
            bytes,
        )))
    }

    /// The columns `DuckLakeTable` reads `table_id`'s inlined rows in at
    /// `snapshot_id`: kept from an earlier fill, else read from the catalog.
    fn columns(&self, table_id: i64, snapshot_id: i64) -> Result<Arc<Vec<DuckLakeTableColumn>>> {
        let kept = self
            .memo
            .lock()
            .columns
            .get(&table_id)
            .filter(|(snapshot, _)| *snapshot == snapshot_id)
            .map(|(_, columns)| Arc::clone(columns));
        if let Some(columns) = kept {
            return Ok(columns);
        }
        let columns = Arc::new(self.inner.get_table_structure(table_id, snapshot_id)?);
        self.keep_columns(table_id, snapshot_id, Arc::clone(&columns));
        Ok(columns)
    }

    /// Keep `columns` as `table_id`'s columns at `snapshot_id`, in place of the
    /// columns kept for it at any snapshot.
    fn keep_columns(
        &self,
        table_id: i64,
        snapshot_id: i64,
        columns: Arc<Vec<DuckLakeTableColumn>>,
    ) {
        self.memo
            .lock()
            .columns
            .insert(table_id, (snapshot_id, columns));
    }

    fn picked(&self) -> MutexGuard<'_, HashMap<TableKey, Option<Arc<TableReads>>>> {
        self.picked
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
        if let Some(reads) = self.table_reads(table_id, snapshot_id)? {
            if reads.columns.as_slice() == columns {
                let scan = InlinedDataScan {
                    batches: reads.inlined_rows.clone(),
                    materialized_row_count: 0,
                };
                return Ok((scan, reads.inlined_absent_through));
            }
            // The next fill reads the rows in the scan's own columns.
            self.keep_columns(table_id, snapshot_id, Arc::new(columns.to_vec()));
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
        {
            if reads.columns.as_slice() != columns {
                // The next fill reads the rows in the scan's own columns.
                self.keep_columns(table_id, snapshot_id, Arc::new(columns.to_vec()));
            } else if let Some(rows) = &reads.visible_rows {
                return Ok(rows.clone());
            }
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
    use std::collections::{BTreeMap, BTreeSet};
    use std::sync::Mutex as StdMutex;
    use std::sync::atomic::{AtomicI64, Ordering as AtomicOrdering};

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

    /// A catalog of one table, `1`, whose listing holds `files` data files at
    /// every snapshot, and which reports that it holds no inlined rows. It
    /// counts the reads of the columns and of the inlined rows.
    #[derive(Debug)]
    struct ManyFiles {
        files: i64,
        head: AtomicI64,
        calls: StdMutex<BTreeMap<&'static str, usize>>,
    }

    impl ManyFiles {
        fn new(files: i64) -> Self {
            Self {
                files,
                head: AtomicI64::new(1),
                calls: StdMutex::default(),
            }
        }

        fn count(&self, method: &'static str) {
            *self.calls.lock().unwrap().entry(method).or_default() += 1;
        }

        fn take(&self) -> BTreeMap<&'static str, usize> {
            std::mem::take(&mut *self.calls.lock().unwrap())
        }
    }

    impl MetadataProvider for ManyFiles {
        fn get_current_snapshot(&self) -> Result<i64> {
            Ok(self.head.load(AtomicOrdering::SeqCst))
        }
        fn get_data_path(&self) -> Result<String> {
            Ok("/data".to_string())
        }
        fn list_snapshots(&self) -> Result<Vec<SnapshotMetadata>> {
            unreachable!()
        }
        fn list_schemas(&self, _: i64) -> Result<Vec<SchemaMetadata>> {
            unreachable!()
        }
        fn list_tables(&self, _: i64, _: i64) -> Result<Vec<TableMetadata>> {
            unreachable!()
        }
        fn get_table_structure(&self, _: i64, _: i64) -> Result<Vec<DuckLakeTableColumn>> {
            self.count("get_table_structure");
            Ok(Vec::new())
        }
        fn scan_inlined_data_noting_absence(
            &self,
            _: i64,
            snapshot_id: i64,
            _: &[DuckLakeTableColumn],
            _: Option<&InlinedFilter>,
        ) -> Result<(InlinedDataScan, Option<i64>)> {
            self.count("scan_inlined_data_noting_absence");
            let head = self.head.load(AtomicOrdering::SeqCst);
            Ok((InlinedDataScan::default(), Some(snapshot_id.min(head))))
        }
        fn read_inlined_rows(
            &self,
            _: i64,
            _: &[DuckLakeTableColumn],
            _: InlinedRowSelection,
            _: Option<&InlinedFilter>,
        ) -> Result<Vec<DuckLakeInlinedRows>> {
            self.count("read_inlined_rows");
            Ok(Vec::new())
        }
        fn get_table_files_for_select(&self, _: i64, _: i64) -> Result<Vec<DuckLakeTableFile>> {
            unreachable!()
        }
        fn get_table_file_metadata_page(
            &self,
            _: i64,
            _: i64,
            after_data_file_id: Option<i64>,
            limit: usize,
        ) -> Result<Vec<DuckLakeFileMetadata>> {
            let first = after_data_file_id.map_or(1, |after| after + 1);
            Ok((first..=self.files).take(limit).map(file).collect())
        }
        fn get_schema_by_name(&self, _: &str, _: i64) -> Result<Option<SchemaMetadata>> {
            unreachable!()
        }
        fn get_table_by_name(&self, _: i64, _: &str, _: i64) -> Result<Option<TableMetadata>> {
            unreachable!()
        }
        fn table_exists(&self, _: i64, _: &str, _: i64) -> Result<bool> {
            unreachable!()
        }
        fn list_all_tables(&self, _: i64) -> Result<Vec<TableWithSchema>> {
            unreachable!()
        }
        fn list_all_columns(&self, _: i64) -> Result<Vec<ColumnWithTable>> {
            unreachable!()
        }
        fn list_all_files(&self, _: i64) -> Result<Vec<FileWithTable>> {
            unreachable!()
        }
        fn get_data_files_added_between_snapshots(
            &self,
            _: i64,
            _: i64,
            _: i64,
        ) -> Result<Vec<DataFileChange>> {
            unreachable!()
        }
        fn get_delete_files_added_between_snapshots(
            &self,
            _: i64,
            _: i64,
            _: i64,
        ) -> Result<Vec<DeleteFileChange>> {
            unreachable!()
        }
    }

    /// The fill pages through a listing of several full pages, and the bound
    /// provider serves all of it, page by page, as the scan reads it.
    #[test]
    fn a_fill_keeps_every_page_of_a_long_listing() {
        let files = 2 * FILE_METADATA_BATCH_SIZE as i64 + 5;
        let memo = CatalogReadMemo::new(CatalogReadMemo::DEFAULT_MAX_BYTES);
        let bound = memo.bind(Arc::new(ManyFiles::new(files))).unwrap();
        let mut ids = Vec::new();
        let mut after = None;
        loop {
            let page = bound
                .get_table_file_metadata_page_filtered(1, 1, after, FILE_METADATA_BATCH_SIZE, None)
                .unwrap();
            ids.extend(page.iter().map(|metadata| metadata.file.data_file_id));
            if page.len() < FILE_METADATA_BATCH_SIZE {
                break;
            }
            after = ids.last().copied();
        }
        assert_eq!(ids, (1..=files).collect::<Vec<_>>());
    }

    /// A fill after a commit takes the table's columns, and the provider's
    /// answer that it holds no inlined rows, from the fill before the commit.
    /// The answer covers snapshots up to the head it was read at, not above.
    #[test]
    fn a_fill_after_a_commit_reads_no_columns_and_no_inlined_rows() {
        let provider = Arc::new(ManyFiles::new(3));
        let memo = CatalogReadMemo::new(CatalogReadMemo::DEFAULT_MAX_BYTES);
        let query = |snapshot_id: i64| {
            memo.bind(Arc::clone(&provider) as Arc<dyn MetadataProvider>)
                .unwrap()
                .get_inlined_deletes(1, snapshot_id)
                .unwrap();
            provider.take()
        };
        assert_eq!(
            query(1),
            BTreeMap::from([("get_table_structure", 1), ("scan_inlined_data_noting_absence", 1),])
        );

        provider.head.store(2, AtomicOrdering::SeqCst);
        assert_eq!(query(1), BTreeMap::new(), "a commit moved the head");

        // Snapshot 3 lies above the head, 2: a later commit can still inline
        // rows at it.
        assert_eq!(
            query(3),
            BTreeMap::from([
                ("get_table_structure", 1),
                ("read_inlined_rows", 1),
                ("scan_inlined_data_noting_absence", 1),
            ])
        );
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
