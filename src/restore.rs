//! Restore a table's rows to an earlier snapshot in one forward commit.
//!
//! [`DuckLakeTable::restore_rows_to`] makes a table's live rows equal to its
//! rows at an earlier snapshot, the *mark*, by committing one new snapshot.
//! History is never rewritten: every snapshot, the ones between the mark and
//! the restore included, still reads what it read. A restored row keeps the
//! `rowid` it had at the mark, so anything keyed by `rowid` stays valid.
//!
//! This is a library extension; official DuckLake has no equivalent. The
//! restore commits what a SQL `UPDATE` commits (official's
//! `ducklake_update.cpp`: a delete of row versions plus an insert of row
//! versions that keep their row ids), through the same staged write, so it
//! leaves the catalog as an `UPDATE` leaves it, and the change ledger records
//! `inserted_into_table`/`deleted_from_table` or `inlined_insert`/
//! `inlined_delete` as an `UPDATE` would.
//!
//! # What changes
//!
//! The restore compares where the table's rows live at the mark and at the
//! snapshot the table was opened at (its *head*), reading nothing the catalog
//! had to keep for it besides the mark itself:
//!
//! - **Removed**: rows live at the head that were not live at the mark, deleted
//!   as a `DELETE` deletes them: the live rows of a data file added since the
//!   mark, rows a file live at both holds from after the mark (a partial file a
//!   flush or merge wrote), and inlined rows inserted since the mark. A file's
//!   deletions within the scoped `data_inlining_row_limit` go to
//!   `ducklake_inlined_delete_<table_id>`; above it a file added since the mark
//!   is ended whole, by metadata only (unless some of its rows were deleted
//!   through `ducklake_inlined_delete_<table_id>`, over which official never
//!   ends a file), and other files get a delete file.
//! - **Restored**: rows live at the mark that are not live at the head: the
//!   rows of data files ended since the mark, the rows of files live at both
//!   that were deleted since (each delete file's per-position snapshots where
//!   it records them, else the difference of the two delete sets, and inlined
//!   deletions since the mark), and inlined rows ended since. They are
//!   inserted again with their row ids, inlined when they number at most the
//!   table's scoped `data_inlining_row_limit` (as an `UPDATE`'s new versions
//!   are), else written to Parquet files that embed each row's id, each row
//!   routed to its partition on a partitioned table.
//!
//! An update since the mark is both: its new version is removed and its old one
//! restored. A flush or merge since the mark that wrote a partial file
//! leaves the rows it moved where they are; a rewrite that ended files and
//! wrote new ones has its new files ended and the rows of the files it ended
//! restored.
//!
//! # Requirements
//!
//! The table must have been opened at the head of a writable catalog
//! ([`DuckLakeCatalog::with_writer`](crate::DuckLakeCatalog::with_writer)), on
//! a writer that stages row mutations (SQLite, DuckDB, MySQL and multi-catalog
//! PostgreSQL; single-catalog PostgreSQL refuses with
//! [`DuckLakeError::Unsupported`], as it refuses `UPDATE` and `DELETE`). The
//! commit is based on the table's snapshot, and any commit to the table since
//! that snapshot (one the change ledger names it in) aborts it with
//! [`DuckLakeError::Conflict`], so the rows it leaves are the mark's. A
//! PostgreSQL commit guard approves it as it approves any other commit.
//!
//! # Cost
//!
//! The comparison reads the catalog's file lists at both snapshots, the delete
//! files of the data files live at both, and the embedded snapshots of partial
//! files read at the mark. The restored rows are read file by file and
//! streamed into the files they are written to; they are held in memory only
//! when they are few enough to inline.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow::array::{BooleanArray, RecordBatch};
use arrow::datatypes::SchemaRef;
use datafusion::catalog::Session;
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use futures::future::BoxFuture;
use futures::stream::{self, BoxStream};
use futures::{StreamExt, TryStreamExt};

use crate::DuckLakeError;
use crate::delete_exec::{DeletionTarget, FileDeletions, LatestDeleteFiles};
use crate::keyed_mutation::from_df;
use crate::metadata_provider::DuckLakeTableFile;
use crate::metadata_writer::{DroppedDataFile, InlinedRowRef, StagedTableData};
use crate::rowid_lookup::bounded;
use crate::table::{DuckLakeTable, rewrite_output_schema, rewrite_scanned_batch};
use crate::table_writer::DuckLakeTableWriter;

/// What [`DuckLakeTable::restore_rows_to`] did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct RestoreResult {
    /// The snapshot the restore committed, or `None` when the table's rows
    /// already equalled its rows at the mark and nothing was committed.
    pub snapshot_id: Option<i64>,
    /// Rows live at the head that were not live at the mark, now removed.
    pub rows_removed: u64,
    /// Rows live at the mark that were not live at the head, now inserted
    /// again with their row ids.
    pub rows_restored: u64,
    /// Data files the restore ended: those added since the mark, and any file
    /// whose remaining rows it deleted.
    pub files_ended: usize,
}

impl DuckLakeTable {
    /// Restore this table's rows to their state at `snapshot`, keeping each
    /// restored row's `rowid`, in one new snapshot; see the
    /// [`restore`](crate::restore) module docs.
    ///
    /// Errors:
    ///
    /// - [`DuckLakeError::InvalidConfig`] when the table was not opened from a
    ///   writable catalog, `snapshot` is not before the snapshot the table was
    ///   opened at, or the table does not exist at `snapshot`;
    /// - [`DuckLakeError::InvalidSnapshot`] when `snapshot` is not in the
    ///   catalog: it has expired (or was never one of this catalog's), so the
    ///   rows it read may be gone;
    /// - [`DuckLakeError::Unsupported`] when the writer does not stage row
    ///   mutations (single-catalog PostgreSQL), or the table's columns or
    ///   partition spec changed since `snapshot`;
    /// - [`DuckLakeError::Conflict`] when the table's rows changed since the
    ///   snapshot it was opened at; a PostgreSQL commit guard that refuses
    ///   it fails it with `CommitRefused`.
    ///
    /// None of these refusals commits anything.
    pub async fn restore_rows_to(
        &self,
        state: &dyn Session,
        snapshot: i64,
    ) -> crate::Result<RestoreResult> {
        let context = RestoreContext::new(self, state, snapshot)?;
        // Each step's state lives on the heap, keeping this future small for
        // a caller that awaits it inside its own.
        let diff = Box::pin(context.diff(state)).await?;
        // An expiry commits no snapshot, so the commit's checks cannot see
        // one: the mark surviving the diff's catalog reads shows they read it
        // whole. A data file cleaned up since fails the read of its rows.
        mark_survives(self, snapshot)?;
        if diff.rows_removed == 0 && diff.rows_restored == 0 {
            return Ok(RestoreResult::default());
        }
        Box::pin(context.commit(state, diff)).await
    }
}

/// Refuses a restore to `snapshot` when it is not in the catalog: it has
/// expired, or was never one of its snapshots.
fn mark_survives(head: &DuckLakeTable, snapshot: i64) -> crate::Result<()> {
    if head
        .provider()
        .snapshot_ids_between(snapshot, snapshot)?
        .is_empty()
    {
        return Err(DuckLakeError::InvalidSnapshot(format!(
            "restore of '{}': snapshot {snapshot} is not in the catalog; it has expired or was \
             never one of its snapshots, and the rows it read may be gone",
            head.table_name()
        )));
    }
    Ok(())
}

/// What a restore works with, resolved and checked once.
struct RestoreContext<'a> {
    head: &'a DuckLakeTable,
    /// The table at the mark.
    mark: DuckLakeTable,
    table_writer: DuckLakeTableWriter,
    schema_name: String,
    physical_schema: SchemaRef,
    /// The table's scoped `data_inlining_row_limit`.
    inline_limit: usize,
    concurrency: usize,
}

/// One data file's rows the restore removes, with what the deletion is
/// recorded against.
struct RemovedPositions {
    file: DuckLakeTableFile,
    /// Its live delete file's positions and their snapshots.
    existing: HashMap<i64, Option<i64>>,
    /// Sorted, distinct, each live at the head.
    positions: Vec<i64>,
}

/// One data file's rows (as read at the mark) the restore inserts again.
struct RestoredPositions {
    file: DuckLakeTableFile,
    positions: Positions,
    /// How many rows `positions` selects.
    rows: u64,
}

/// Positions of a data file, held as the smaller set: the deleted positions
/// when every other row of a file is restored, so a large file restored
/// whole costs no set of its rows.
enum Positions {
    Only(HashSet<i64>),
    AllBut(HashSet<i64>),
}

impl Positions {
    fn contains(&self, position: i64) -> bool {
        match self {
            Self::Only(positions) => positions.contains(&position),
            Self::AllBut(positions) => !positions.contains(&position),
        }
    }
}

/// How many of `rows` positions `removed` leaves.
fn remaining(rows: i64, removed: &HashSet<i64>) -> u64 {
    let gone = removed
        .iter()
        .filter(|pos| (0..rows).contains(*pos))
        .count();
    (rows.max(0) as u64).saturating_sub(gone as u64)
}

/// The rows a restore removes and inserts again.
#[derive(Default)]
struct RestoreDiff {
    ended: Vec<DroppedDataFile>,
    removed: Vec<RemovedPositions>,
    removed_inlined: Vec<InlinedRowRef>,
    rows_removed: u64,
    restored: Vec<RestoredPositions>,
    /// Inlined rows to insert again, `[physical columns..., rowid]`.
    restored_inlined: Vec<RecordBatch>,
    rows_restored: u64,
}

/// One data file's part of the diff.
enum FileDiff {
    Ended(DroppedDataFile, u64),
    Changed {
        removed: Option<Box<RemovedPositions>>,
        restored: Option<Box<RestoredPositions>>,
    },
}

impl<'a> RestoreContext<'a> {
    fn new(head: &'a DuckLakeTable, state: &dyn Session, snapshot: i64) -> crate::Result<Self> {
        let name = head.table_name();
        let writer = head.writer().cloned().ok_or_else(|| {
            DuckLakeError::InvalidConfig(format!(
                "restore of '{name}': the table was not opened from a writable catalog"
            ))
        })?;
        if !writer.supports_inlined_row_mutations() {
            return Err(DuckLakeError::Unsupported(
                "restoring rows needs a writer that stages row mutations (SQLite, DuckDB, \
                 MySQL or multi-catalog PostgreSQL)"
                    .to_string(),
            ));
        }
        let schema_name = head
            .schema_name()
            .ok_or_else(|| {
                DuckLakeError::InvalidConfig(format!(
                    "restore of '{name}': the table has no schema name"
                ))
            })?
            .to_string();
        let base = head.snapshot_id();
        if snapshot >= base {
            return Err(DuckLakeError::InvalidConfig(format!(
                "restore of '{name}': snapshot {snapshot} is not before snapshot {base}, the \
                 one the table was read at"
            )));
        }
        let provider = head.provider();
        mark_survives(head, snapshot)?;
        if provider
            .get_table_structure(head.table_id(), snapshot)?
            .is_empty()
        {
            return Err(DuckLakeError::InvalidConfig(format!(
                "restore of '{name}': the table does not exist at snapshot {snapshot}"
            )));
        }
        let mark = DuckLakeTable::new(
            head.table_id(),
            name,
            Arc::clone(provider),
            snapshot,
            Arc::clone(head.object_store_url()),
            head.table_path().to_string(),
        )?;
        let physical_schema = head.physical_schema();
        if mark.column_ids() != head.column_ids() || mark.physical_schema() != physical_schema {
            return Err(DuckLakeError::Unsupported(format!(
                "restore of '{name}': its columns changed since snapshot {snapshot}"
            )));
        }
        let partition_id = |at: i64| -> crate::Result<Option<i64>> {
            Ok(provider
                .get_partition_spec(head.table_id(), at)?
                .map(|spec| spec.partition_id))
        };
        if partition_id(snapshot)? != partition_id(base)? {
            return Err(DuckLakeError::Unsupported(format!(
                "restore of '{name}': its partition spec changed since snapshot {snapshot}"
            )));
        }
        let object_store = state
            .runtime_env()
            .object_store(head.object_store_url().as_ref())
            .map_err(|e| DuckLakeError::Internal(e.to_string()))?;
        let table_writer =
            DuckLakeTableWriter::new(writer, object_store)?.with_options(head.write_options());
        let inline_limit = table_writer.table_data_inlining_row_limit(&schema_name, name)?;
        Ok(Self {
            head,
            mark,
            table_writer,
            schema_name,
            physical_schema,
            inline_limit,
            concurrency: state.config().target_partitions().max(1),
        })
    }

    /// The rows the restore removes and inserts again.
    async fn diff(&self, state: &dyn Session) -> crate::Result<RestoreDiff> {
        let head = self.head;
        let mark = &self.mark;
        let head_files = head.files()?;
        let mut mark_files: HashMap<i64, DuckLakeTableFile> = mark
            .files()?
            .into_iter()
            .map(|file| (file.data_file_id, file))
            .collect();
        let head_inlined = head.inlined_deletes_by_file().map_err(from_df)?;
        let mark_inlined = mark.inlined_deletes_by_file().map_err(from_df)?;

        let mut pending: Vec<BoxFuture<'_, DataFusionResult<FileDiff>>> = Vec::new();
        for file in head_files {
            let at_mark = mark_files.remove(&file.data_file_id);
            let head_inlined = head_inlined.get(&file.data_file_id);
            let mark_inlined = mark_inlined.get(&file.data_file_id);
            pending.push(Box::pin(async move {
                match at_mark {
                    None => added_file(head, state, file, head_inlined, self.inline_limit).await,
                    Some(at_mark) => {
                        shared_file(head, mark, state, file, at_mark, head_inlined, mark_inlined)
                            .await
                    },
                }
            }));
        }
        for file in mark_files.into_values() {
            let mark_inlined = mark_inlined.get(&file.data_file_id);
            pending.push(Box::pin(async move {
                let removed = removed_at(mark, state, &file, mark_inlined, None).await?;
                let rows = remaining(row_count(mark, state, &file).await?, &removed);
                Ok(FileDiff::Changed {
                    removed: None,
                    restored: (rows > 0).then(|| {
                        Box::new(RestoredPositions {
                            file,
                            positions: Positions::AllBut(removed),
                            rows,
                        })
                    }),
                })
            }));
        }

        let mut diff = RestoreDiff::default();
        let mut files = bounded(pending, self.concurrency).await.map_err(from_df)?;
        // In catalog order, whatever order the reads finished in.
        files.sort_by_key(|file| match file {
            FileDiff::Ended(ended, _) => ended.data_file_id,
            FileDiff::Changed {
                removed,
                restored,
            } => removed
                .as_ref()
                .map(|removed| removed.file.data_file_id)
                .or(restored.as_ref().map(|restored| restored.file.data_file_id))
                .unwrap_or_default(),
        });
        for file in files {
            match file {
                FileDiff::Ended(ended, rows) => {
                    diff.rows_removed += rows;
                    diff.ended.push(ended);
                },
                FileDiff::Changed {
                    removed,
                    restored,
                } => {
                    if let Some(removed) = removed {
                        diff.rows_removed += removed.positions.len() as u64;
                        diff.removed.push(*removed);
                    }
                    if let Some(restored) = restored {
                        diff.rows_restored += restored.rows;
                        diff.restored.push(*restored);
                    }
                },
            }
        }

        // An inlined row version is the same version at both snapshots exactly
        // when its table, row id and begin snapshot agree: an update ends a
        // version and inserts its successor with the same row id.
        let mut at_head = HashSet::new();
        for rows in head.inlined_data_with_row_ids()? {
            check_aligned(&rows)?;
            for (row_id, begin) in rows.row_ids.iter().zip(&rows.begin_snapshots) {
                at_head.insert((rows.table_name.clone(), *row_id, *begin));
                if *begin > mark.snapshot_id() {
                    diff.removed_inlined.push(InlinedRowRef {
                        table_name: rows.table_name.clone(),
                        row_id: *row_id,
                    });
                }
            }
        }
        diff.rows_removed += diff.removed_inlined.len() as u64;
        let out_schema = rewrite_output_schema(&self.physical_schema);
        for rows in mark.inlined_data_with_row_ids()? {
            check_aligned(&rows)?;
            let keep = BooleanArray::from(
                rows.row_ids
                    .iter()
                    .zip(&rows.begin_snapshots)
                    .map(|(row_id, begin)| {
                        !at_head.contains(&(rows.table_name.clone(), *row_id, *begin))
                    })
                    .collect::<Vec<_>>(),
            );
            if keep.true_count() == 0 {
                continue;
            }
            let Some((batch, _)) = mark.apply_update_to_inlined(&rows, None, &[])? else {
                continue;
            };
            let kept = arrow::compute::filter_record_batch(&batch, &keep)?;
            diff.rows_restored += kept.num_rows() as u64;
            diff.restored_inlined.push(RecordBatch::try_new(
                Arc::clone(&out_schema),
                kept.columns().to_vec(),
            )?);
        }
        Ok(diff)
    }

    /// Commit `diff` in one snapshot based on the table's.
    async fn commit(&self, state: &dyn Session, diff: RestoreDiff) -> crate::Result<RestoreResult> {
        let RestoreDiff {
            ended,
            removed,
            removed_inlined,
            rows_removed,
            restored,
            restored_inlined,
            rows_restored,
        } = diff;
        let head = self.head;
        let base_snapshot = head.snapshot_id();
        let table_name = head.table_name();
        let (data, row_ids) =
            Box::pin(self.stage_restored(state, restored, restored_inlined, rows_restored)).await?;
        let data_files = |data: &StagedTableData| match data {
            StagedTableData::Files(files) => files
                .iter()
                .map(|file| (file.path.clone(), file.path_is_relative))
                .collect(),
            _ => Vec::new(),
        };

        // Each file's removed rows are deleted as official's
        // `DuckLakeDelete::FlushDelete` deletes them, every outcome kept so a
        // failure still knows the delete files the others wrote.
        let commit_snapshot = match self.table_writer.next_snapshot_id() {
            Ok(next) => next.map_or(base_snapshot + 1, |next| next.max(base_snapshot + 1)),
            Err(error) => {
                self.table_writer
                    .remove_unstaged(&self.schema_name, table_name, data_files(&data))
                    .await;
                return Err(error);
            },
        };
        let pending: Vec<BoxFuture<'_, DataFusionResult<DataFusionResult<FileDeletions>>>> =
            removed
                .iter()
                .map(
                    |file| -> BoxFuture<'_, DataFusionResult<DataFusionResult<FileDeletions>>> {
                        Box::pin(async move {
                            let mut deletions = FileDeletions::dated(commit_snapshot);
                            let recorded = deletions
                                .record(
                                    head,
                                    &self.table_writer,
                                    &self.schema_name,
                                    table_name,
                                    &DeletionTarget::of_file(&file.file, &file.existing),
                                    file.positions.clone(),
                                    self.inline_limit,
                                    base_snapshot,
                                )
                                .await;
                            Ok(recorded.map(|()| deletions))
                        })
                    },
                )
                .collect();
        let recorded = bounded(pending, self.concurrency).await.map_err(from_df)?;
        let mut deletions = FileDeletions::default();
        let mut failure = None;
        for recorded in recorded {
            match recorded {
                Ok(recorded) => deletions.absorb(recorded),
                Err(error) => failure = failure.or(Some(error)),
            }
        }
        if let Some(error) = failure {
            let mut written = data_files(&data);
            written.extend(
                deletions
                    .positional
                    .iter()
                    .map(|entry| &entry.delete)
                    .chain(
                        deletions
                            .rewritten
                            .iter()
                            .map(|rewritten| &rewritten.entry.delete.delete),
                    )
                    .map(|delete| (delete.path.clone(), delete.path_is_relative)),
            );
            self.table_writer
                .remove_unstaged(&self.schema_name, table_name, written)
                .await;
            return Err(from_df(error));
        }
        deletions.dropped.extend(ended);
        let files_ended = deletions.dropped.len();

        let latest = LatestDeleteFiles {
            table: head,
            state,
        };
        let mut transaction = self.table_writer.transaction();
        transaction.set_delete_file_positions(&latest);
        transaction
            .stage_row_mutation(
                &self.schema_name,
                table_name,
                head.table_id(),
                head.column_ids(),
                self.physical_schema.as_ref(),
                base_snapshot,
                data,
                row_ids,
                deletions.positional,
                deletions.rewritten,
                removed_inlined,
                deletions.inlined,
                deletions.dropped,
            )
            .await?;
        // Any commit to the table since its snapshot aborts the restore: the
        // rows it leaves are the mark's only if the head it compared against
        // is still the head. An `UPDATE`'s checks alone would admit a
        // concurrent insert, a second restore re-inserting the same rows, or a
        // rewrite whose new files begin before the snapshot.
        transaction.make_last_write_exclusive();
        let snapshot_id = transaction
            .commit()
            .await?
            .first()
            .map(|result| result.snapshot_id);
        Ok(RestoreResult {
            snapshot_id,
            rows_removed,
            rows_restored,
            files_ended,
        })
    }

    /// The restored rows as staged data: inlined with their row ids when they
    /// number at most the limit and the backend inlines them, else written to
    /// Parquet files that embed their row ids, streamed file by file.
    async fn stage_restored(
        &self,
        state: &dyn Session,
        restored: Vec<RestoredPositions>,
        restored_inlined: Vec<RecordBatch>,
        rows: u64,
    ) -> crate::Result<(StagedTableData, Option<Vec<i64>>)> {
        if rows == 0 {
            return Ok((StagedTableData::None, None));
        }
        let mark = &self.mark;
        let table_name = self.head.table_name();
        let source = || -> BoxStream<'_, DataFusionResult<RecordBatch>> {
            stream::iter(restored.iter())
                .map(move |file| file_rows(mark, state, &file.file, &file.positions))
                .flatten()
                .chain(stream::iter(restored_inlined.iter().cloned().map(Ok)))
                .boxed()
        };
        if self.inline_limit > 0 && rows <= self.inline_limit as u64 {
            let batches: Vec<RecordBatch> = source().try_collect().await.map_err(from_df)?;
            let width = self.physical_schema.fields().len();
            let new_versions = batches
                .iter()
                .map(|batch| {
                    RecordBatch::try_new(
                        Arc::clone(&self.physical_schema),
                        batch.columns()[..width].to_vec(),
                    )
                })
                .collect::<std::result::Result<Vec<_>, _>>()?;
            if self.table_writer.should_inline(
                &self.schema_name,
                table_name,
                rows as usize,
                self.physical_schema.as_ref(),
                &new_versions,
            )? {
                let row_ids = batches
                    .iter()
                    .flat_map(|batch| row_ids_of(batch).to_vec())
                    .collect();
                return Ok((StagedTableData::Inlined(new_versions), Some(row_ids)));
            }
            let files = crate::update_exec::write_rewritten_stream(
                self.restored_files_session()?,
                stream::iter(batches.into_iter().map(Ok)),
            )
            .await
            .map_err(from_df)?;
            return Ok((StagedTableData::Files(files), None));
        }
        let files =
            crate::update_exec::write_rewritten_stream(self.restored_files_session()?, source())
                .await
                .map_err(from_df)?;
        Ok((StagedTableData::Files(files), None))
    }

    /// A session writing restored rows to Parquet files that embed their row
    /// ids, rolling over at the target file size, based on the table's
    /// snapshot.
    fn restored_files_session(&self) -> crate::Result<crate::table_writer::TableWriteSession> {
        Ok(self
            .table_writer
            .begin_rolling_write_with_embedded_rowid(
                &self.schema_name,
                self.head.table_name(),
                self.physical_schema.as_ref(),
                crate::metadata_writer::WriteMode::Append,
            )?
            .with_base_snapshot_id(self.head.snapshot_id()))
    }
}

/// A head file not live at the mark: all its live rows are removed, as
/// official's `DELETE` removes every row of a file
/// (`DuckLakeDelete::FlushDelete`). At most the scoped `inline_limit` of them
/// are recorded in `ducklake_inlined_delete_<table_id>`; more end the file
/// whole, by metadata only. A file with deletions already in
/// `ducklake_inlined_delete_<table_id>` is never ended, as official does not
/// count them towards its rows deleted; its live rows are deleted as a
/// `DELETE` deletes them.
async fn added_file(
    head: &DuckLakeTable,
    state: &dyn Session,
    file: DuckLakeTableFile,
    inlined: Option<&HashSet<i64>>,
    inline_limit: usize,
) -> DataFusionResult<FileDiff> {
    let existing = match &file.delete_file {
        Some(delete_file) => head.read_delete_file_entries(state, delete_file).await?,
        None => HashMap::new(),
    };
    let removed = removed_at(head, state, &file, inlined, Some(&existing)).await?;
    let rows = row_count(head, state, &file).await?;
    let live = remaining(rows, &removed);
    if live == 0 {
        return Ok(FileDiff::Changed {
            removed: None,
            restored: None,
        });
    }
    let inlines = inline_limit > 0 && live <= inline_limit as u64;
    if inlines || inlined.is_some_and(|inlined| !inlined.is_empty()) {
        return Ok(FileDiff::Changed {
            removed: Some(Box::new(RemovedPositions {
                file,
                existing,
                positions: (0..rows).filter(|pos| !removed.contains(pos)).collect(),
            })),
            restored: None,
        });
    }
    Ok(FileDiff::Ended(
        DroppedDataFile {
            data_file_id: file.data_file_id,
            expected_delete_file: file.delete_file_id,
        },
        live,
    ))
}

/// A file live at both snapshots: the positions live at the head but not at
/// the mark are removed, those live at the mark but not at the head restored.
async fn shared_file(
    head: &DuckLakeTable,
    mark: &DuckLakeTable,
    state: &dyn Session,
    file: DuckLakeTableFile,
    at_mark: DuckLakeTableFile,
    head_inlined: Option<&HashSet<i64>>,
    mark_inlined: Option<&HashSet<i64>>,
) -> DataFusionResult<FileDiff> {
    let existing = match &file.delete_file {
        Some(delete_file) => head.read_delete_file_entries(state, delete_file).await?,
        None => HashMap::new(),
    };
    let removed_at_head = removed_at(head, state, &file, head_inlined, Some(&existing)).await?;
    // The same delete file at both snapshots dates each position it records,
    // so the mark's set needs no second read.
    let same_delete_file =
        file.delete_file_id.is_some() && file.delete_file_id == at_mark.delete_file_id;
    let removed_at_mark = removed_at(
        mark,
        state,
        &at_mark,
        mark_inlined,
        same_delete_file.then_some(&existing),
    )
    .await?;
    let rows = row_count(head, state, &file).await?;
    let mut removed: Vec<i64> = removed_at_mark
        .difference(&removed_at_head)
        .copied()
        .filter(|pos| (0..rows).contains(pos))
        .collect();
    removed.sort_unstable();
    let restored: HashSet<i64> = removed_at_head
        .difference(&removed_at_mark)
        .copied()
        .filter(|pos| (0..rows).contains(pos))
        .collect();
    Ok(FileDiff::Changed {
        removed: (!removed.is_empty()).then(|| {
            Box::new(RemovedPositions {
                file,
                existing,
                positions: removed,
            })
        }),
        restored: (!restored.is_empty()).then(|| {
            Box::new(RestoredPositions {
                file: at_mark,
                rows: restored.len() as u64,
                positions: Positions::Only(restored),
            })
        }),
    })
}

/// The positions of `file` holding no live row at `table`'s snapshot: those its
/// delete file (as `entries`, when already read) and `inlined` deletions
/// remove as of the snapshot, and those a partial file holds from after it.
async fn removed_at(
    table: &DuckLakeTable,
    state: &dyn Session,
    file: &DuckLakeTableFile,
    inlined: Option<&HashSet<i64>>,
    entries: Option<&HashMap<i64, Option<i64>>>,
) -> DataFusionResult<HashSet<i64>> {
    let mut removed = match entries {
        // A position the file records no snapshot for was deleted at the
        // file's own `begin_snapshot`, which a live file's is not after.
        Some(entries) => {
            let mut removed: HashSet<i64> = entries
                .iter()
                .filter(|(_, snapshot)| snapshot.is_none_or(|s| s <= table.snapshot_id()))
                .map(|(position, _)| *position)
                .collect();
            removed.extend(inlined.into_iter().flatten());
            removed
        },
        None => {
            table
                .deleted_positions_at_snapshot(state, file, inlined)
                .await?
        },
    };
    removed.extend(table.positions_inserted_later(state, file).await?);
    Ok(removed)
}

/// The number of rows `file` holds, from the catalog or else its footer.
async fn row_count(
    table: &DuckLakeTable,
    state: &dyn Session,
    file: &DuckLakeTableFile,
) -> DataFusionResult<i64> {
    match file.max_row_count {
        Some(rows) => Ok(rows),
        None => table.data_file_row_count(state, &file.file).await,
    }
}

/// The rows of `file` at `positions`, read at `table`'s snapshot as
/// `[physical columns..., rowid]` with each row's `rowid`, its delete files
/// not applied: the positions include rows deleted since.
fn file_rows<'a>(
    table: &'a DuckLakeTable,
    state: &'a dyn Session,
    file: &'a DuckLakeTableFile,
    positions: &'a Positions,
) -> BoxStream<'a, DataFusionResult<RecordBatch>> {
    let physical_schema = table.physical_schema();
    let out_schema = rewrite_output_schema(&physical_schema);
    stream::once(async move {
        let mut unmasked = file.clone();
        unmasked.delete_file = None;
        unmasked.delete_file_id = None;
        let scan = table
            .build_update_scan_with_snapshot(state, &unmasked, false, None, None)
            .await?;
        let rows =
            datafusion::physical_plan::execute_stream(Arc::clone(&scan.scan), state.task_ctx())?;
        Ok::<_, DataFusionError>(
            rows.map(move |batch| {
                let Some(rewritten) = rewrite_scanned_batch(
                    &physical_schema,
                    &out_schema,
                    &scan,
                    &batch?,
                    None,
                    &[],
                )?
                else {
                    return Ok(None);
                };
                let keep = BooleanArray::from(
                    rewritten
                        .positions
                        .iter()
                        .map(|position| positions.contains(*position))
                        .collect::<Vec<_>>(),
                );
                let kept = arrow::compute::filter_record_batch(&rewritten.batch, &keep)?;
                Ok((kept.num_rows() > 0).then_some(kept))
            })
            .try_filter_map(|batch| futures::future::ready(Ok(batch))),
        )
    })
    .try_flatten()
    .boxed()
}

/// The trailing `rowid` column of a `[physical columns..., rowid]` batch.
fn row_ids_of(batch: &RecordBatch) -> &[i64] {
    batch
        .column(batch.num_columns() - 1)
        .as_any()
        .downcast_ref::<arrow::array::Int64Array>()
        .expect("rowid column is Int64")
        .values()
}

fn check_aligned(rows: &crate::metadata_provider::DuckLakeInlinedData) -> crate::Result<()> {
    if rows.begin_snapshots.len() != rows.row_ids.len() {
        return Err(DuckLakeError::Internal(format!(
            "inlined rows of '{}' carry {} begin snapshots for {} row ids",
            rows.table_name,
            rows.begin_snapshots.len(),
            rows.row_ids.len()
        )));
    }
    Ok(())
}
