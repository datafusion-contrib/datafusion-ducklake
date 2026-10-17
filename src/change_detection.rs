//! Whether a table's data changed between two snapshots, read from the
//! snapshot change ledger (`ducklake_snapshot_changes.changes_made`) as
//! official DuckLake records it.
//!
//! Every commit names the tables it touched in `changes_made`:
//! `inserted_into_table:<id>` and `deleted_from_table:<id>` for Parquet data
//! and delete files, `inlined_insert:<id>` and `inlined_delete:<id>` for rows
//! and deletions stored in the catalog, `inline_flush:<id>` (official's
//! `flushed_inlined`) when inlined data moves to Parquet,
//! `merge_adjacent:<id>`, `rewrite_delete:<id>` and `compacted_table:<id>` for
//! compaction, `altered_table:<id>` and `dropped_table:<id>`. Reading the
//! ledger answers "did this table change" without listing a single data file,
//! and sees changes that live only in the catalog (inlined inserts and
//! deletions), which a file listing cannot.
//!
//! Whether the rows changed is what official's change feed says:
//! [`TableDataChanges::rows_changed`] is set when this crate's port of
//! `ducklake_table_insertions` or `ducklake_table_deletions` over the window
//! returns a row (or the window's history is unknown). The ledger decides
//! without reading data wherever it can:
//!
//! - no token naming the table in the window (and no expired or unrecorded
//!   snapshot): unchanged. Every change the feed reports is a data file, a
//!   delete file or an inlined row or deletion committed in the window, and
//!   every commit that writes one names the table;
//! - an insert or a delete token (`inserted_into_table`, `deleted_from_table`,
//!   `inlined_insert`, `inlined_delete`), `dropped_table`, a token this crate
//!   does not know, an unrecorded snapshot or an expired one: changed. This
//!   errs towards a change in one case: official's flush of inlined deletions
//!   of Parquet rows (and a flush that writes delete files beside moving rows)
//!   records `deleted_from_table`, as a `DELETE` does, while the feed reports
//!   nothing for it;
//! - `altered_table` only: unchanged. An alteration writes no data file, delete
//!   file or inlined row, so the feed has nothing for it;
//! - otherwise the snapshots recorded only a flush or a compaction
//!   (`inline_flush`, `flushed_inlined`, `merge_adjacent`, `rewrite_delete`,
//!   `compacted_table`), which move rows or rewrite their files, and the feed
//!   is read over them, stopping at its first row. A `DELETE` committed in one
//!   transaction with official's flush records only `inline_flush`; the feed
//!   reports the deletion. Official's `ducklake_rewrite_data_files` feed
//!   reports every row it rewrote as an `UPDATE`, so a rewrite counts as a
//!   change; a merge of adjacent files reports nothing.
//!
//! [`TableDataChanges::layout_changed`] adds every flush and compaction from
//! the tokens alone. A cache keyed by row content needs the first; one keyed by
//! file and position (a row's location) needs both.
//!
//! # Example
//!
//! ```no_run
//! # async fn example(
//! #     ctx: &datafusion::prelude::SessionContext,
//! #     provider: &std::sync::Arc<dyn datafusion_ducklake::MetadataProvider>,
//! # ) -> datafusion_ducklake::Result<()> {
//! use datafusion_ducklake::change_detection::table_data_changes;
//!
//! let table_id = 7;
//! let cached_at = 41;
//! let head = provider.get_current_snapshot()?;
//! let changes = table_data_changes(&ctx.state(), provider, table_id, cached_at, head).await?;
//! if changes.rows_changed() {
//!     // rebuild what was derived from the table's rows at `cached_at`
//! }
//! # Ok(())
//! # }
//! ```

use std::sync::Arc;

use datafusion::catalog::Session;
use datafusion::datasource::TableProvider;
use datafusion::physical_plan::execute_stream;
use futures::StreamExt;

use crate::Result;
use crate::metadata_provider::{MetadataProvider, snapshot_change_tokens};
use crate::table_changes::TableInsertionsTable;
use crate::table_deletions::TableDeletionsTable;
use crate::table_functions::{CdcTableContext, cdc_table_context};

/// What the snapshots in a window did to one table, from the change ledger
/// and, where the ledger cannot tell, the change feed.
///
/// Built by [`table_data_changes`]. Each flag is set when at least one
/// snapshot in the window recorded that kind of change for the table, or
/// (for [`Self::inserted`] and [`Self::deleted`]) when the change feed
/// reported one over snapshots that recorded only a flush or a compaction.
/// The feed is read only over such snapshots after the last one whose tokens
/// recorded a row change, so the feed-derived part of these two flags covers
/// only those: [`Self::rows_changed`] is exact, while an insertion or deletion
/// the feed would report for an earlier flush or compaction leaves its flag
/// as the tokens set it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct TableDataChanges {
    /// Rows were written to Parquet data files (`inserted_into_table`), or
    /// `ducklake_table_insertions` reported a row over a flush or compaction.
    pub inserted: bool,
    /// Rows were deleted through delete files or by ending data files
    /// (`deleted_from_table`), a `DELETE`, `UPDATE`, truncate or `Replace`
    /// write among them, or `ducklake_table_deletions` reported a row over a
    /// flush or compaction (a `DELETE` committed with official's flush).
    pub deleted: bool,
    /// Rows were stored inlined in the catalog (`inlined_insert`).
    pub inlined_inserted: bool,
    /// Inlined rows were deleted, or deletions of Parquet rows were recorded
    /// inlined (`inlined_delete`).
    pub inlined_deleted: bool,
    /// Inlined rows or deletions moved to Parquet (`inline_flush`, official's
    /// `flushed_inlined`).
    pub flushed: bool,
    /// Data files were merged or rewritten (`merge_adjacent`,
    /// `rewrite_delete`, `compacted_table`).
    pub compacted: bool,
    /// The table's schema or layout changed (`altered_table`): a column added,
    /// dropped, renamed or retyped, its partitioning, sort order, comment or
    /// tags. [`Self::rows_changed`] leaves it out: which rows the table holds
    /// is the same, though what a row reads as may not be (a column added with
    /// a default, a type widened). Derived data that depends on the columns
    /// checks this too, or [`Self::any_changed`].
    pub altered: bool,
    /// The table was dropped (`dropped_table`).
    pub dropped: bool,
    /// A snapshot in the window has no `changes_made` recorded (a catalog
    /// written by a tool that keeps no change ledger), or records a change of
    /// the table this crate does not know, so what it did to the table is
    /// unknown. [`Self::rows_changed`] counts it as a change.
    pub unrecorded: bool,
    /// Snapshots of the window were expired, so their ledger rows are gone
    /// and what they did to the table is unknown. [`Self::rows_changed`]
    /// counts it as a change.
    ///
    /// Found by snapshot id: a catalog numbers its snapshots consecutively, so
    /// a missing id between the window's start and the catalog's head is an
    /// expired snapshot. A multi-catalog PostgreSQL store shares one id
    /// sequence among its catalogs; there only an expired starting snapshot,
    /// or an expired first snapshot of the table in the window, is found, and
    /// taken to mean the snapshots after it were expired too. That finds every
    /// expiry by age (`ExpireCriteria::OlderThan`), which removes a catalog's
    /// oldest snapshots, but not `ExpireCriteria::Versions` naming a snapshot
    /// strictly inside the window: a missing id there is as likely another
    /// catalog's snapshot or a rolled-back commit's, so it is not taken as one.
    pub history_incomplete: bool,
    /// The latest snapshot in the window that changed the table's rows (as
    /// [`Self::rows_changed`] counts them), if any.
    pub last_rows_changed_snapshot: Option<i64>,
    /// The latest snapshot in the window that recorded any change of the
    /// table, if any.
    pub last_changed_snapshot: Option<i64>,
}

impl TableDataChanges {
    /// Whether the rows the table holds may differ between the window's ends:
    /// an insert or delete, Parquet or inlined, recorded or reported by the
    /// change feed, a drop, a snapshot without a recorded ledger, or an
    /// incomplete history. A schema change alone ([`Self::altered`]) is not
    /// one: it changes no row's presence, though it can change the columns
    /// every row reads with.
    #[must_use]
    pub fn rows_changed(&self) -> bool {
        self.inserted
            || self.deleted
            || self.inlined_inserted
            || self.inlined_deleted
            || self.dropped
            || self.unrecorded
            || self.history_incomplete
    }

    /// Whether where the table's rows live changed: a flush or compaction, or
    /// any change that [`Self::rows_changed`] counts. A row's data file and
    /// position (and whether it is inlined) are stable only while this is
    /// `false`; its `rowid` survives a flush and a compaction.
    #[must_use]
    pub fn layout_changed(&self) -> bool {
        self.rows_changed() || self.flushed || self.compacted
    }

    /// Whether anything at all was recorded for the table, an alteration
    /// included.
    #[must_use]
    pub fn any_changed(&self) -> bool {
        self.layout_changed() || self.altered
    }

    /// Record the tokens of one snapshot's `changes_made`, returning what the
    /// tokens say of its rows.
    fn record(
        &mut self,
        snapshot_id: i64,
        changes_made: Option<&str>,
        table_id: i64,
    ) -> SnapshotRows {
        let Some(changes) = changes_made else {
            self.unrecorded = true;
            self.last_rows_changed_snapshot = Some(snapshot_id);
            self.last_changed_snapshot = Some(snapshot_id);
            return SnapshotRows::Changed;
        };
        let mut rows = false;
        let mut moved = false;
        let mut any = false;
        for (kind, value) in
            snapshot_change_tokens(changes).filter_map(|token| token.trim().split_once(':'))
        {
            if value.trim().parse::<i64>().ok() != Some(table_id) {
                continue;
            }
            let kind = kind.trim().to_ascii_lowercase();
            if crate::metadata_provider::change_kind_names_other_object(&kind) {
                continue;
            }
            any = true;
            match kind.as_str() {
                "inserted_into_table" => {
                    rows = true;
                    self.inserted = true;
                },
                "deleted_from_table" => {
                    rows = true;
                    self.deleted = true;
                },
                "inlined_insert" => {
                    rows = true;
                    self.inlined_inserted = true;
                },
                "inlined_delete" => {
                    rows = true;
                    self.inlined_deleted = true;
                },
                "dropped_table" => {
                    rows = true;
                    self.dropped = true;
                },
                "inline_flush" | "flushed_inlined" => {
                    moved = true;
                    self.flushed = true;
                },
                "merge_adjacent" | "rewrite_delete" | "compacted_table" => {
                    moved = true;
                    self.compacted = true;
                },
                "altered_table" => self.altered = true,
                // A change of this table the crate does not know: what it did
                // to the rows is unknown.
                _ => {
                    rows = true;
                    self.unrecorded = true;
                },
            }
        }
        if any {
            self.last_changed_snapshot = Some(snapshot_id);
        }
        if rows {
            self.last_rows_changed_snapshot = Some(snapshot_id);
            SnapshotRows::Changed
        } else if moved {
            SnapshotRows::Moved
        } else {
            SnapshotRows::Unchanged
        }
    }

    /// Record that the window's history is incomplete: snapshots up to
    /// `bound` (the first surviving snapshot after them) were expired.
    fn record_gap(&mut self, bound: i64) {
        self.history_incomplete = true;
        self.last_rows_changed_snapshot = Some(
            self.last_rows_changed_snapshot
                .map_or(bound, |snapshot| snapshot.max(bound)),
        );
        self.last_changed_snapshot = Some(
            self.last_changed_snapshot
                .map_or(bound, |snapshot| snapshot.max(bound)),
        );
    }
}

/// What one snapshot's tokens say of the table's rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SnapshotRows {
    /// Nothing that writes a data file, delete file or inlined row.
    Unchanged,
    /// Only a flush or a compaction: the change feed decides.
    Moved,
    /// An insert, a delete, a drop, or an unknown or unrecorded change.
    Changed,
}

/// What the snapshots after `after_snapshot`, through `through_snapshot`, did
/// to table `table_id`.
///
/// Reads the ledger rows of the window that can name the table (and those
/// with no ledger row) and the snapshot ids from the window's start up to its
/// end or the catalog's head, whichever is first
/// ([`MetadataProvider::snapshot_ids_between`]). Where the snapshots after the
/// last one whose tokens record a row change recorded only a flush or a
/// compaction, this crate's `ducklake_table_deletions` and
/// `ducklake_table_insertions` are read over them through `state`, which must
/// reach the table's object store, each stopping at its first row (and again
/// one snapshot at a time to find the latest one that changed rows, when they
/// report one). A negative `after_snapshot` starts the window at the catalog's
/// first snapshot. A window with no snapshot (`after_snapshot >=
/// through_snapshot`) reports nothing.
pub async fn table_data_changes(
    state: &dyn Session,
    provider: &Arc<dyn MetadataProvider>,
    table_id: i64,
    after_snapshot: i64,
    through_snapshot: i64,
) -> Result<TableDataChanges> {
    analyse(
        state,
        provider,
        table_id,
        after_snapshot,
        through_snapshot,
        true,
    )
    .await
}

/// [`table_data_changes`], stopping as soon as a row change is found: the
/// flags past it may be unset and `last_rows_changed_snapshot` may not be the
/// latest.
async fn analyse(
    state: &dyn Session,
    provider: &Arc<dyn MetadataProvider>,
    table_id: i64,
    after_snapshot: i64,
    through_snapshot: i64,
    complete: bool,
) -> Result<TableDataChanges> {
    let mut changes = TableDataChanges::default();
    if after_snapshot >= through_snapshot {
        return Ok(changes);
    }
    let mut moved = Vec::new();
    for (snapshot_id, changes_made) in
        provider.list_table_snapshot_changes(table_id, after_snapshot, through_snapshot)?
    {
        if changes.record(snapshot_id, changes_made.as_deref(), table_id) == SnapshotRows::Moved {
            moved.push(snapshot_id);
        }
    }
    if let Some(bound) = expired_in_window(
        provider.as_ref(),
        table_id,
        after_snapshot,
        through_snapshot,
    )? {
        changes.record_gap(bound);
    }
    if !complete && changes.rows_changed() {
        return Ok(changes);
    }
    // Snapshots up to the last recorded row change are counted already.
    let settled = changes.last_rows_changed_snapshot.unwrap_or(i64::MIN);
    moved.retain(|snapshot| *snapshot > settled);
    let (Some(&first), Some(&last)) = (moved.first(), moved.last()) else {
        return Ok(changes);
    };
    let Some(feed) = ChangeFeed::open(provider, table_id, through_snapshot)? else {
        // The table is not visible at the window's end while the ledger
        // recorded no drop of it: what happened to its rows is unknown.
        changes.unrecorded = true;
        changes.last_rows_changed_snapshot = Some(last);
        return Ok(changes);
    };
    let (deleted, inserted) = feed.reports(state, first, last).await?;
    if !deleted && !inserted {
        return Ok(changes);
    }
    changes.deleted |= deleted;
    changes.inserted |= inserted;
    let mut latest = last;
    if complete && moved.len() > 1 {
        for snapshot in moved.iter().rev() {
            let (deleted, inserted) = feed.reports(state, *snapshot, *snapshot).await?;
            if deleted || inserted {
                latest = *snapshot;
                break;
            }
        }
    }
    changes.last_rows_changed_snapshot = Some(latest);
    Ok(changes)
}

/// This crate's `ducklake_table_deletions` and `ducklake_table_insertions`
/// over one table, resolved at a window's end.
struct ChangeFeed<'a> {
    provider: &'a Arc<dyn MetadataProvider>,
    context: CdcTableContext,
    end_snapshot: i64,
}

impl<'a> ChangeFeed<'a> {
    /// The feeds of table `table_id` as of `end_snapshot`, or `None` when the
    /// table is not visible there.
    fn open(
        provider: &'a Arc<dyn MetadataProvider>,
        table_id: i64,
        end_snapshot: i64,
    ) -> Result<Option<Self>> {
        let Some(found) = provider
            .list_all_tables(end_snapshot)?
            .into_iter()
            .find(|entry| entry.table.table_id == table_id)
        else {
            return Ok(None);
        };
        let Some(schema) = provider.get_schema_by_name(&found.schema_name, end_snapshot)? else {
            return Ok(None);
        };
        let context = cdc_table_context(provider.as_ref(), &schema, &found.table, end_snapshot)?;
        Ok(Some(Self {
            provider,
            context,
            end_snapshot,
        }))
    }

    /// Whether the deletions and the insertions feeds over the inclusive
    /// window `[from, through]` report a row, each read up to its first row.
    async fn reports(&self, state: &dyn Session, from: i64, through: i64) -> Result<(bool, bool)> {
        let through = through.min(self.end_snapshot);
        let context = &self.context;
        let deletions = TableDeletionsTable::new(
            Arc::clone(self.provider),
            context.table_id,
            from,
            through,
            Arc::new(context.object_store_url.clone()),
            context.table_path.clone(),
            Arc::clone(&context.table_schema),
        )
        .with_columns(context.columns.clone());
        let deleted = has_row(state, &deletions).await?;
        let insertions = TableInsertionsTable::new(
            Arc::clone(self.provider),
            context.table_id,
            from,
            through,
            Arc::new(context.object_store_url.clone()),
            context.table_path.clone(),
            Arc::clone(&context.table_schema),
        )
        .with_columns(context.columns.clone());
        let inserted = has_row(state, &insertions).await?;
        Ok((deleted, inserted))
    }
}

/// Whether `feed` yields a row, reading only its leading `snapshot_id` column
/// and stopping at the first row.
async fn has_row(state: &dyn Session, feed: &dyn TableProvider) -> Result<bool> {
    let plan = feed.scan(state, Some(&vec![0]), &[], Some(1)).await?;
    let mut stream = execute_stream(plan, state.task_ctx())?;
    while let Some(batch) = stream.next().await {
        if batch?.num_rows() > 0 {
            return Ok(true);
        }
    }
    Ok(false)
}

/// When snapshots of the window `(after_snapshot, through_snapshot]` were
/// expired, the first surviving snapshot after the last of them: no change
/// before it is known. `None` when the window's history is complete, as far as
/// the snapshot ids tell.
///
/// With consecutive ids (see
/// [`MetadataProvider::snapshot_ids_are_consecutive`]) every id of the window
/// up to the catalog's head must survive; whether the snapshot the window
/// starts after survives does not matter, since its ledger row is outside the
/// window. With a shared id sequence a gap is invisible, so an expired
/// starting snapshot is taken as a sign that the snapshots after it were
/// expired too, and so is an expired first snapshot of table `table_id`
/// ([`MetadataProvider::table_first_snapshot`]) in the window: the table's
/// history before the first snapshot that survives it is gone.
fn expired_in_window(
    provider: &dyn MetadataProvider,
    table_id: i64,
    after_snapshot: i64,
    through_snapshot: i64,
) -> Result<Option<i64>> {
    // Ids past the catalog's head are not expired, only not yet taken; the
    // head itself is never expired.
    let end = through_snapshot.min(provider.get_current_snapshot()?);
    let ids = provider.snapshot_ids_between(after_snapshot.max(0), end)?;
    // The first surviving snapshot after `id`, else the window's end.
    let after = |id: i64| {
        ids.iter()
            .copied()
            .find(|candidate| *candidate > id)
            .unwrap_or(end)
    };
    if !provider.snapshot_ids_are_consecutive() {
        let mut bound = if after_snapshot < 0 {
            // A whole history with no snapshot left up to its end: every one
            // was expired, or none was taken yet, which the shared sequence
            // cannot tell apart.
            (ids.is_empty() && end >= 0).then_some(end)
        } else {
            (ids.first() != Some(&after_snapshot)).then(|| after(after_snapshot))
        };
        if let Some(created) = provider.table_first_snapshot(table_id)?
            && created > after_snapshot
            && created <= end
            && ids.binary_search(&created).is_err()
        {
            let gap_end = after(created);
            bound = Some(bound.map_or(gap_end, |bound| bound.max(gap_end)));
        }
        return Ok(bound);
    }
    // The window's first id: the snapshot after `after_snapshot`, or for a
    // whole history the catalog's first, which official numbers 0 and this
    // crate's writers 1; snapshot 0 of official creates only the schema.
    let mut bound = None;
    let first = if after_snapshot >= 0 {
        after_snapshot + 1
    } else {
        match ids.first() {
            Some(first) if *first <= 1 => *first,
            Some(first) => {
                bound = Some(*first);
                *first
            },
            // No snapshot is left up to the window's end, the head past it: a
            // catalog's first snapshot (`0` or `1`) was taken, and every one
            // through the end was expired. Snapshot `0` alone changes no table.
            None => return Ok((end >= 1).then_some(end)),
        }
    };
    if first > end {
        return Ok(bound);
    }
    // The latest id of `first..=end` that is missing, walking down from `end`.
    let present: std::collections::HashSet<i64> = ids.iter().copied().collect();
    if let Some(missing) = (first..=end).rev().find(|id| !present.contains(id)) {
        let gap_end = after(missing);
        bound = Some(bound.map_or(gap_end, |bound: i64| bound.max(gap_end)));
    }
    Ok(bound)
}

/// Whether table `table_id`'s rows may have changed after `after_snapshot`
/// through `through_snapshot` ([`TableDataChanges::rows_changed`]). Stops at
/// the first snapshot whose tokens record a row change, reading no data then.
pub async fn table_rows_changed(
    state: &dyn Session,
    provider: &Arc<dyn MetadataProvider>,
    table_id: i64,
    after_snapshot: i64,
    through_snapshot: i64,
) -> Result<bool> {
    Ok(analyse(
        state,
        provider,
        table_id,
        after_snapshot,
        through_snapshot,
        false,
    )
    .await?
    .rows_changed())
}

/// The latest snapshot at or before `through_snapshot` that changed table
/// `table_id`'s rows ([`TableDataChanges::rows_changed`]), or `None` when no
/// snapshot did. A snapshot whose ledger is unrecorded counts as one.
pub async fn last_rows_changed_snapshot(
    state: &dyn Session,
    provider: &Arc<dyn MetadataProvider>,
    table_id: i64,
    through_snapshot: i64,
) -> Result<Option<i64>> {
    Ok(
        table_data_changes(state, provider, table_id, -1, through_snapshot)
            .await?
            .last_rows_changed_snapshot,
    )
}

/// The `LIKE` pattern a ledger row naming table `table_id` matches: every
/// token naming a table by id ends in `:<id>`. It also matches longer ids
/// sharing the prefix, which [`table_data_changes`] filters out.
#[cfg_attr(
    not(any(
        feature = "metadata-sqlite",
        feature = "metadata-duckdb",
        feature = "metadata-postgres",
        feature = "metadata-mysql"
    )),
    allow(dead_code)
)]
pub(crate) fn table_change_like_pattern(table_id: i64) -> String {
    format!("%:{table_id}%")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn changes_of(rows: &[(i64, Option<&str>)], table_id: i64) -> TableDataChanges {
        let mut changes = TableDataChanges::default();
        for (snapshot, made) in rows {
            let _ = changes.record(*snapshot, *made, table_id);
        }
        changes
    }

    #[test]
    fn inlined_changes_count_as_row_changes() {
        let changes = changes_of(&[(3, Some("inlined_insert:1"))], 1);
        assert!(changes.inlined_inserted && changes.rows_changed());
        assert_eq!(changes.last_rows_changed_snapshot, Some(3));
        let changes = changes_of(&[(4, Some("inlined_delete:1"))], 1);
        assert!(changes.inlined_deleted && changes.rows_changed());
    }

    #[test]
    fn flush_and_compaction_change_only_the_layout() {
        let changes = changes_of(
            &[(5, Some("inline_flush:1")), (6, Some("merge_adjacent:1"))],
            1,
        );
        assert!(changes.flushed && changes.compacted && !changes.deleted);
        assert!(!changes.rows_changed() && changes.layout_changed());
        assert_eq!(changes.last_rows_changed_snapshot, None);
        assert_eq!(changes.last_changed_snapshot, Some(6));
        let flush_only = changes_of(&[(7, Some("flushed_inlined:1"))], 1);
        assert!(!flush_only.rows_changed() && flush_only.layout_changed());
    }

    #[test]
    fn deleted_from_table_is_always_a_deletion() {
        // A flush writing delete files, a `DELETE` committed with a flush, a
        // flush of inlined deletions alone and a `DELETE` record the same
        // tokens: each counts as a deletion without reading the feed.
        for made in [
            "deleted_from_table:1,inline_flush:1",
            "deleted_from_table:1,flushed_inlined:1",
            "deleted_from_table:1",
            "deleted_from_table:1,inlined_delete:1,inline_flush:1",
        ] {
            let changes = changes_of(&[(5, Some(made))], 1);
            assert!(changes.deleted && changes.rows_changed(), "{made}");
            assert_eq!(changes.last_rows_changed_snapshot, Some(5), "{made}");
        }
    }

    #[test]
    fn other_tables_and_prefix_ids_are_ignored() {
        let changes = changes_of(
            &[(
                2,
                Some("inserted_into_table:12,created_table:\"main\".\"t:1\""),
            )],
            1,
        );
        assert_eq!(changes, TableDataChanges::default());
    }

    #[test]
    fn flush_and_compaction_tokens_leave_the_rows_to_the_feed() {
        let mut changes = TableDataChanges::default();
        assert_eq!(
            changes.record(5, Some("inline_flush:1"), 1),
            SnapshotRows::Moved
        );
        assert_eq!(
            changes.record(6, Some("rewrite_delete:1,altered_table:2"), 1),
            SnapshotRows::Moved
        );
        assert_eq!(
            changes.record(7, Some("altered_table:1"), 1),
            SnapshotRows::Unchanged
        );
        assert_eq!(
            changes.record(8, Some("inline_flush:1,inlined_delete:1"), 1),
            SnapshotRows::Changed
        );
    }

    #[test]
    fn unknown_tokens_naming_the_table_count_as_changes() {
        let changes = changes_of(&[(4, Some("future_change:1"))], 1);
        assert!(changes.unrecorded && changes.rows_changed(), "{changes:?}");
        assert_eq!(changes.last_rows_changed_snapshot, Some(4));
    }

    #[test]
    fn unrecorded_snapshots_count_as_changes() {
        let changes = changes_of(&[(9, None)], 1);
        assert!(changes.unrecorded && changes.rows_changed());
        assert_eq!(changes.last_rows_changed_snapshot, Some(9));
    }
}
