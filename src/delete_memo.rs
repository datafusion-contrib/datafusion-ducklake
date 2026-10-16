//! The delete positions a [`DuckLakeTable`](crate::DuckLakeTable) keeps for
//! its later scans, configured by [`DuckLakeReadOptions`].
//!
//! By default every scan reads again each delete file it reaches, because the
//! delete sets live for one plan. A caller that keeps a built table across
//! queries pays that read on every query. The delete memo keeps the positions
//! each data file's delete file removes. A delete file is never rewritten in
//! place: new deletions from a data file go to a new file under a new path, and
//! official DuckLake names each file with a new UUID. So a path always names the
//! same positions, and the memo needs no check against the catalog.
//!
//! The memo is not shared between tables, and it is off by default. For the
//! catalog reads of a scan, see [`crate::CatalogReadMemo`].

use std::collections::{HashMap, HashSet};
use std::mem::size_of;
use std::sync::{Arc, Mutex};

/// How a [`DuckLakeTable`](crate::DuckLakeTable) reuses what its scans read.
///
/// By default (`DuckLakeReadOptions::default()`) a table reads every delete file
/// a scan reaches, on every scan. Give the delete memo a byte budget to keep
/// those positions for the table's later scans instead; [`Self::memoized`]
/// turns the memo on with the default budget. Apply the options with
/// [`DuckLakeCatalog::with_read_options`](crate::DuckLakeCatalog::with_read_options)
/// or [`DuckLakeTable::with_read_options`](crate::DuckLakeTable::with_read_options).
///
/// The memo pays off for a caller that keeps a built table across queries, for
/// example one that caches tables by catalog, name and snapshot. A table that a
/// catalog lookup builds for one statement is seldom scanned twice.
///
/// # Budget
///
/// The budget holds per table, so a caller that keeps many tables keeps up to
/// that many budgets. Byte counts are estimates of the heap memory held.
///
/// The memo keeps positions until they fill [`Self::delete_memo_bytes`]. It
/// never evicts: once full, it keeps what it has, and other delete files are
/// read on each scan. A merge or a rewrite can replace a data file that the memo
/// holds positions for. The old file's entry stays in the memo, and counts
/// against its budget, until the table is dropped.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct DuckLakeReadOptions {
    /// Bytes of delete-file positions a table can keep across scans. Zero, the
    /// default, turns the delete memo off.
    pub delete_memo_bytes: usize,
    /// EXPERIMENTAL: whether scans read the `ext_ducklake_file_order` table
    /// (see [`crate::ext_file_order`]) and serve `ORDER BY <sort key> LIMIT n`
    /// from the files it records, in order and without a sort. Off by default:
    /// the table is never read, and scans plan as without this option.
    pub ext_file_order: bool,
}

impl DuckLakeReadOptions {
    /// The delete memo's budget in [`Self::memoized`]: 16 MiB, about a million
    /// deleted rows.
    pub const DEFAULT_DELETE_MEMO_BYTES: usize = 16 << 20;

    /// The delete memo on, with the default budget.
    #[must_use]
    pub fn memoized() -> Self {
        Self::default().with_delete_memo(Self::DEFAULT_DELETE_MEMO_BYTES)
    }

    /// Keep delete-file positions within `max_bytes`. Zero turns the delete
    /// memo off.
    #[must_use]
    pub fn with_delete_memo(mut self, max_bytes: usize) -> Self {
        self.delete_memo_bytes = max_bytes;
        self
    }

    /// EXPERIMENTAL: declare the verified sort order of recorded files to a
    /// sort that asks for it; see [`Self::ext_file_order`].
    #[must_use]
    pub fn with_ext_file_order(mut self, enabled: bool) -> Self {
        self.ext_file_order = enabled;
        self
    }
}

/// Estimated memory a set of deleted positions holds: the slots of its hash
/// table, each an `i64` and a control byte, at the table's 7/8 load factor.
pub(crate) fn positions_bytes(positions: &HashSet<i64>) -> usize {
    positions.capacity() * (size_of::<i64>() + 1) * 8 / 7 + size_of::<HashSet<i64>>()
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

    #[test]
    fn default_options_turn_the_memo_off() {
        let options = DuckLakeReadOptions::default();
        assert!(DeletePositionMemo::new(options.delete_memo_bytes).is_none());

        let memoized = DuckLakeReadOptions::memoized();
        assert!(DeletePositionMemo::new(memoized.delete_memo_bytes).is_some());
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
