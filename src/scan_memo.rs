//! What a [`DuckLakeTable`](crate::DuckLakeTable) keeps of its scans' reads for
//! the scans after them, configured by [`DuckLakeReadOptions`].
//!
//! A table is bound to one snapshot, but by default every scan of it asks the
//! catalog again for the same facts: the file listing with each file's column
//! statistics and partition values, the inlined deletions, the inlined rows, and
//! whether the snapshot is the current one. Every scan also reads again each
//! delete file it reaches. A caller that keeps a built table across queries pays
//! all of it on every query. The two memos here remove that cost:
//!
//! - The **catalog memo** keeps the catalog reads of the table's first scan. It
//!   is all or nothing: the listing, the inlined rows, the inlined deletions and
//!   the current-snapshot answer are captured by one fill and used together. A
//!   merge, a rewrite or a flush of inlined data changes how a past snapshot is
//!   stored, so a memoized listing used beside a fresh inlined read (or the
//!   reverse) could count a flushed row twice, or not at all. For the same
//!   reason the fill is kept only if no commit landed while it read. The fill
//!   records the catalog head, and each later scan reads the head and fills the
//!   memo again when a commit moved it, as official DuckLake keys its caches on
//!   the snapshot it reads.
//! - The **delete memo** keeps the positions each data file's delete file
//!   removes. A delete file is never rewritten in place: new deletions from a
//!   data file are written to a new file under a new path. So a path always
//!   names the same positions.
//!
//! Neither memo is shared between tables, and both are off by default.

use std::collections::{HashMap, HashSet};
use std::mem::size_of;
use std::sync::{Arc, Mutex};

use arrow::record_batch::RecordBatch;

use crate::metadata_provider::{
    DuckLakeFileColumnStatistics, DuckLakeFileData, DuckLakeFileMetadata, DuckLakeStatistics,
    DuckLakeTableFile,
};

/// How a [`DuckLakeTable`](crate::DuckLakeTable) reuses what its scans read.
///
/// By default (`DuckLakeReadOptions::default()`) a table reads the catalog on
/// every scan, and reads every delete file a scan reaches. Give a memo a byte
/// budget to keep those reads for the table's later scans instead;
/// [`Self::memoized`] turns both memos on with the default budgets. Apply the
/// options with
/// [`DuckLakeCatalog::with_read_options`](crate::DuckLakeCatalog::with_read_options)
/// or [`DuckLakeTable::with_read_options`](crate::DuckLakeTable::with_read_options).
///
/// Memoizing pays off for a caller that keeps a built table across queries, for
/// example one that caches tables by catalog, name and snapshot. A table that a
/// catalog lookup builds for one statement is seldom scanned twice.
///
/// # The head check
///
/// The rows at a table's snapshot never change, but the files that hold them
/// can. A commit can replace a file that an older snapshot still reads:
///
/// - a `DELETE` or `UPDATE` of rows in a data file that already has deletions
///   writes a new delete file in place of the old one;
/// - merging adjacent files, or rewriting files with many deletions, writes new
///   data files in place of the old ones;
/// - flushing inlined data moves inlined rows and deletions into new files.
///
/// Each replaced file is scheduled for deletion, and cleanup deletes it once it
/// is older than the grace period cleanup is given: two days by default in
/// official DuckLake (`delete_older_than`), or at once with
/// `ducklake_cleanup_old_files(.., cleanup_all => true)` or this crate's
/// `CleanupCriteria::All`.
///
/// Each of these replacements is a commit, and every commit moves the catalog
/// head. So a memoized table reads the head on each scan, one catalog query,
/// and fills the memo again when the head moved since the fill. A scan thus
/// never plans over a replaced file, and cleanup cannot delete a file that a
/// scan is about to read, except one that is already running, which is the
/// case cleanup's grace period exists for. A commit to any table of the
/// catalog moves the head, so on a busy catalog the memo fills again often.
///
/// A delete memo keeps no file names, so it needs no check.
///
/// Expiring snapshots commits nothing, so the head check does not notice that
/// the table's own snapshot expired. A table at an expired snapshot cannot be
/// read without a memo either, as in official DuckLake.
///
/// # Budgets
///
/// Each budget holds per table, so a caller that keeps many memoized tables
/// keeps up to that many budgets. Byte counts are estimates of the heap memory
/// held.
///
/// The catalog memo keeps a scan's reads only if they fit in
/// [`Self::catalog_memo_bytes`]; otherwise the table reads the catalog on every
/// scan, as without a memo, and does not try to fill it again. The scan that
/// fills the memo reads without its filters, so that the memo can serve any
/// later scan: it lists every file without the catalog-side statistics filter,
/// and reads every visible inlined row, as an unfiltered scan does. Later scans
/// prune the listing in memory, and apply their filters to the inlined rows
/// above the scan.
///
/// The delete memo keeps positions until they fill [`Self::delete_memo_bytes`].
/// It never evicts: once full, it keeps what it has, and other delete files are
/// read on each scan. A scan that lists the files again, without a catalog
/// memo or after the head moved, can find that a merge or a rewrite replaced a
/// data file. The old file's entry stays in the memo, and counts against its
/// budget, until the table is dropped.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct DuckLakeReadOptions {
    /// Bytes a table can keep of a scan's catalog reads. Zero, the default,
    /// turns the catalog memo off.
    pub catalog_memo_bytes: usize,
    /// Bytes of delete-file positions a table can keep across scans. Zero, the
    /// default, turns the delete memo off.
    pub delete_memo_bytes: usize,
}

impl DuckLakeReadOptions {
    /// The catalog memo's budget in [`Self::memoized`]: 32 MiB, about 10,000
    /// data files with statistics on 20 columns each.
    pub const DEFAULT_CATALOG_MEMO_BYTES: usize = 32 << 20;
    /// The delete memo's budget in [`Self::memoized`]: 16 MiB, about a million
    /// deleted rows.
    pub const DEFAULT_DELETE_MEMO_BYTES: usize = 16 << 20;

    /// Both memos on, with the default budgets.
    #[must_use]
    pub fn memoized() -> Self {
        Self::default()
            .with_catalog_memo(Self::DEFAULT_CATALOG_MEMO_BYTES)
            .with_delete_memo(Self::DEFAULT_DELETE_MEMO_BYTES)
    }

    /// Keep the first scan's catalog reads within `max_bytes`. Zero turns the
    /// catalog memo off.
    #[must_use]
    pub fn with_catalog_memo(mut self, max_bytes: usize) -> Self {
        self.catalog_memo_bytes = max_bytes;
        self
    }

    /// Keep delete-file positions within `max_bytes`. Zero turns the delete
    /// memo off.
    #[must_use]
    pub fn with_delete_memo(mut self, max_bytes: usize) -> Self {
        self.delete_memo_bytes = max_bytes;
        self
    }

    /// The catalog memo's budget, or `None` when it is off.
    pub(crate) fn catalog_memo_budget(&self) -> Option<MemoBudget> {
        (self.catalog_memo_bytes > 0).then_some(MemoBudget {
            max_bytes: self.catalog_memo_bytes,
            bytes: 0,
        })
    }
}

/// One page of a table's file listing: the files, and the column statistics the
/// catalog records for them.
#[derive(Debug, Clone, Default)]
pub(crate) struct ListingPage {
    pub(crate) files: Vec<DuckLakeTableFile>,
    /// Only `files` is set: the per-file column statistics of `files`.
    pub(crate) statistics: DuckLakeStatistics,
}

impl From<Vec<DuckLakeFileMetadata>> for ListingPage {
    fn from(metadata: Vec<DuckLakeFileMetadata>) -> Self {
        let mut page = Self {
            files: Vec::with_capacity(metadata.len()),
            statistics: DuckLakeStatistics::default(),
        };
        for DuckLakeFileMetadata {
            file,
            column_statistics,
        } in metadata
        {
            page.files.push(file);
            page.statistics.files.extend(column_statistics);
        }
        page
    }
}

/// The catalog reads of one scan, which later scans of the table use in place
/// of reading the catalog.
#[derive(Debug)]
pub(crate) struct CatalogMemo {
    /// The catalog head when the reads ran: a later scan uses the memo only
    /// while the head has not moved.
    pub(crate) head: i64,
    /// The complete file listing, in the pages the catalog returned it in.
    pub(crate) pages: Vec<ListingPage>,
    /// Inlined deletions: data file id to deleted row positions.
    pub(crate) inlined_deletes: HashMap<i64, HashSet<i64>>,
    /// Every visible inlined row, unfiltered.
    pub(crate) inlined_rows: Vec<RecordBatch>,
    /// Rows the catalog materialized to fill `inlined_rows`.
    pub(crate) inlined_materialized: usize,
    /// Whether the table's snapshot was the current one throughout the reads
    /// above. Decides, as it does for a scan without a memo, whether the
    /// catalog's delete counts are exact; see
    /// `DuckLakeTable::build_exec_for_files_with_deletes`.
    pub(crate) at_current_snapshot: bool,
}

/// A table's catalog memo.
#[derive(Debug, Default)]
pub(crate) enum MemoState {
    /// No scan has filled the memo yet, or the last fill was not kept.
    #[default]
    Unset,
    /// The reads of a scan, used by later scans while the head stays at
    /// [`CatalogMemo::head`].
    Filled(Arc<CatalogMemo>),
    /// A scan's reads did not fit the budget. They are not tried again: the
    /// table's snapshot holds the same rows on every scan, and trying again
    /// would repeat an unfiltered listing on every scan.
    TooLarge,
}

/// A table's catalog memo, shared with every clone of the table.
pub(crate) type CatalogMemoCell = Mutex<MemoState>;

/// The running size of a catalog memo being filled, against its budget.
#[derive(Debug)]
pub(crate) struct MemoBudget {
    max_bytes: usize,
    bytes: usize,
}

impl MemoBudget {
    /// Count `bytes` more. False once the memo is over budget.
    pub(crate) fn add(&mut self, bytes: usize) -> bool {
        self.bytes = self.bytes.saturating_add(bytes);
        self.bytes <= self.max_bytes
    }
}

/// Estimated memory one listing page holds.
pub(crate) fn listing_page_bytes(page: &ListingPage) -> usize {
    let files = page
        .files
        .iter()
        .map(|file| {
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
            size_of::<DuckLakeTableFile>()
                + file_data_heap_bytes(&file.file)
                + delete_file
                + partition_values
        })
        .sum::<usize>();
    let statistics = page
        .statistics
        .files
        .iter()
        .map(|statistic| {
            size_of::<DuckLakeFileColumnStatistics>()
                + statistic.min_value.as_ref().map_or(0, String::len)
                + statistic.max_value.as_ref().map_or(0, String::len)
        })
        .sum::<usize>();
    size_of::<ListingPage>() + files + statistics
}

fn file_data_heap_bytes(file: &DuckLakeFileData) -> usize {
    file.path.len() + file.encryption_key.as_ref().map_or(0, String::len)
}

/// Estimated memory a set of deleted positions holds: the slots of its hash
/// table, each an `i64` and a control byte, at the table's 7/8 load factor.
pub(crate) fn positions_bytes(positions: &HashSet<i64>) -> usize {
    positions.capacity() * (size_of::<i64>() + 1) * 8 / 7 + size_of::<HashSet<i64>>()
}

/// Estimated memory the inlined deletions of a table hold.
pub(crate) fn inlined_deletes_bytes(deletes: &HashMap<i64, HashSet<i64>>) -> usize {
    deletes
        .values()
        .map(|positions| size_of::<i64>() + positions_bytes(positions))
        .sum()
}

/// The positions each data file's delete file removes, kept for later scans of
/// the same table.
///
/// An entry is keyed by the data file and records the delete file's resolved
/// path and the snapshot it was read for, and it serves only a read of that path
/// for that snapshot. A path always names the same positions, because a delete
/// file is never rewritten in place. A file that records each deletion's
/// snapshot contributes only the deletions made at or before the read snapshot,
/// so that snapshot is part of the entry too. When a later scan lists a new
/// delete file for the data file, its positions replace the entry. An entry
/// whose data file no listing names any more, after a merge or a rewrite,
/// stays until the memo is dropped.
#[derive(Debug)]
pub(crate) struct DeletePositionMemo {
    max_bytes: usize,
    state: Mutex<DeleteMemoState>,
}

#[derive(Debug, Default)]
struct DeleteMemoState {
    bytes: usize,
    files: HashMap<i64, DeleteMemoEntry>,
}

#[derive(Debug)]
struct DeleteMemoEntry {
    path: String,
    read_snapshot: Option<i64>,
    positions: Arc<HashSet<i64>>,
    bytes: usize,
}

impl DeletePositionMemo {
    /// A memo that keeps up to `max_bytes` of positions, or `None` when
    /// `max_bytes` is zero.
    pub(crate) fn new(max_bytes: usize) -> Option<Arc<Self>> {
        (max_bytes > 0).then(|| {
            Arc::new(Self {
                max_bytes,
                state: Mutex::new(DeleteMemoState::default()),
            })
        })
    }

    /// The positions kept for data file `data_file_id`, if they were read from
    /// the delete file at `path` for `read_snapshot`.
    pub(crate) fn get(
        &self,
        data_file_id: i64,
        path: &str,
        read_snapshot: Option<i64>,
    ) -> Option<Arc<HashSet<i64>>> {
        let state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state
            .files
            .get(&data_file_id)
            .filter(|entry| entry.path == path && entry.read_snapshot == read_snapshot)
            .map(|entry| Arc::clone(&entry.positions))
    }

    /// Keep the positions read from the delete file at `path` for data file
    /// `data_file_id`, replacing what the memo kept for that data file, unless
    /// they would take the memo over its budget.
    pub(crate) fn insert(
        &self,
        data_file_id: i64,
        path: &str,
        read_snapshot: Option<i64>,
        positions: &Arc<HashSet<i64>>,
    ) {
        let bytes = size_of::<DeleteMemoEntry>() + path.len() + positions_bytes(positions);
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let replaced = state
            .files
            .get(&data_file_id)
            .map_or(0, |entry| entry.bytes);
        if (state.bytes - replaced).saturating_add(bytes) > self.max_bytes {
            return;
        }
        state.bytes = state.bytes - replaced + bytes;
        state.files.insert(
            data_file_id,
            DeleteMemoEntry {
                path: path.to_string(),
                read_snapshot,
                positions: Arc::clone(positions),
                bytes,
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn page(paths: &[&str], statistics_per_file: usize) -> ListingPage {
        let metadata = paths
            .iter()
            .enumerate()
            .map(|(index, path)| DuckLakeFileMetadata {
                file: DuckLakeTableFile::new(DuckLakeFileData::new(path.to_string(), true, 100)),
                column_statistics: (0..statistics_per_file)
                    .map(|column_id| DuckLakeFileColumnStatistics {
                        data_file_id: index as i64,
                        column_id: column_id as i64,
                        column_size_bytes: None,
                        value_count: None,
                        null_count: None,
                        min_value: Some("aaaa".to_string()),
                        max_value: Some("zzzzzz".to_string()),
                        contains_nan: None,
                    })
                    .collect(),
            })
            .collect::<Vec<_>>();
        ListingPage::from(metadata)
    }

    #[test]
    fn default_options_turn_both_memos_off() {
        let options = DuckLakeReadOptions::default();
        assert!(options.catalog_memo_budget().is_none());
        assert!(DeletePositionMemo::new(options.delete_memo_bytes).is_none());

        let memoized = DuckLakeReadOptions::memoized();
        assert!(memoized.catalog_memo_budget().is_some());
        assert!(DeletePositionMemo::new(memoized.delete_memo_bytes).is_some());

        // Setting the field alone is enough: no second knob to forget. Outside
        // this crate `#[non_exhaustive]` rules out a struct literal, so this
        // is how a caller sets it.
        #[allow(clippy::field_reassign_with_default)]
        let catalog_only = {
            let mut options = DuckLakeReadOptions::default();
            options.catalog_memo_bytes = 1;
            options
        };
        assert!(catalog_only.catalog_memo_budget().is_some());
    }

    #[test]
    fn a_listing_page_splits_files_from_their_statistics() {
        let page = page(&["a.parquet", "b.parquet"], 3);
        assert_eq!(page.files.len(), 2);
        assert_eq!(page.statistics.files.len(), 6);
        assert!(page.statistics.table.is_none());
        assert!(page.statistics.columns.is_empty());
    }

    #[test]
    fn listing_page_bytes_counts_paths_and_statistics() {
        let bare = listing_page_bytes(&page(&["a.parquet"], 0));
        assert!(bare >= size_of::<DuckLakeTableFile>() + "a.parquet".len());
        let with_statistics = listing_page_bytes(&page(&["a.parquet"], 3));
        assert_eq!(
            with_statistics - bare,
            3 * (size_of::<DuckLakeFileColumnStatistics>() + "aaaa".len() + "zzzzzz".len())
        );
    }

    #[test]
    fn memo_budget_stops_past_its_limit() {
        let mut budget = DuckLakeReadOptions::default()
            .with_catalog_memo(10)
            .catalog_memo_budget()
            .unwrap();
        assert!(budget.add(10));
        assert!(!budget.add(1), "one byte more exceeds the budget");
    }

    #[test]
    fn delete_memo_keeps_entries_within_its_budget() {
        let small = Arc::new(HashSet::from([1, 2, 3]));
        let budget = size_of::<DeleteMemoEntry>() + "a".len() + positions_bytes(&small);
        let memo = DeletePositionMemo::new(budget).unwrap();

        memo.insert(1, "a", Some(7), &small);
        assert_eq!(memo.get(1, "a", Some(7)).as_deref(), Some(&*small));
        // An entry serves only a read of its path, for its snapshot.
        assert!(memo.get(1, "a", Some(8)).is_none());
        assert!(memo.get(1, "a", None).is_none());
        assert!(memo.get(1, "b", Some(7)).is_none());
        assert!(memo.get(2, "a", Some(7)).is_none());

        // The budget is spent, so a second data file's deletes are read on
        // each scan instead.
        memo.insert(2, "a", Some(7), &small);
        assert!(memo.get(2, "a", Some(7)).is_none());
    }

    #[test]
    fn a_new_delete_file_replaces_the_entry_of_its_data_file() {
        let first = Arc::new(HashSet::from([1]));
        let second = Arc::new(HashSet::from([1, 2]));
        // Room for one entry, not two: replacing must give back the first
        // entry's bytes.
        let budget = size_of::<DeleteMemoEntry>() + "new".len() + positions_bytes(&second);
        let memo = DeletePositionMemo::new(budget).unwrap();
        memo.insert(1, "old", Some(1), &first);
        memo.insert(1, "new", Some(1), &second);
        assert!(memo.get(1, "old", Some(1)).is_none());
        assert_eq!(memo.get(1, "new", Some(1)).as_deref(), Some(&*second));
    }
}
