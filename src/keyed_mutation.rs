//! Keyed mutations of a table from a staged batch: delete, update or upsert
//! the rows whose key columns equal a staged row's.
//!
//! A caller that holds a batch of changed rows and knows the table's key
//! columns gets the semantics of a SQL statement keyed on them, without
//! knowing where the matching rows live:
//!
//! - [`KeyedMutation::delete_by_keys`] is `DELETE FROM t WHERE (k...) IN
//!   (SELECT k... FROM staged)`;
//! - [`KeyedMutation::update_by_keys`] is `MERGE INTO t USING staged ON k...
//!   WHEN MATCHED THEN UPDATE SET *`: each matched row takes the staged row's
//!   values, matched to the table's columns by name, and keeps its `rowid`;
//! - [`KeyedMutation::upsert_by_keys`] deletes the matched rows and appends
//!   every staged row, with new `rowid`s, in the same snapshot.
//!
//! Matches are found in the table's Parquet files and in its inlined rows,
//! and each is recorded the way official DuckLake's `DELETE` and `UPDATE`
//! record it (`DuckLakeDelete::FlushDelete`): a data file's deletions that
//! number at most the table's `data_inlining_row_limit` go to
//! `ducklake_inlined_delete_<table_id>`, even when they delete every row of
//! the file; above the limit, a file they empty is ended, a file with a live
//! delete file gets a new one carrying each position with the snapshot that
//! deleted it, and any other file gets a delete file of its new positions;
//! matched inlined rows are ended. New rows (the upsert's appends,
//! the update's new versions) are inlined when the statement writes at most
//! the limit's rows, as official's insert inlines them, else written to
//! Parquet. Everything commits in one snapshot, with the conflict checks and
//! retries of a SQL `DELETE` or `UPDATE`
//! ([`DuckLakeWriteTransaction::commit`](crate::DuckLakeWriteTransaction::commit)).
//!
//! A key is matched by value equality of every key column, each found in a
//! staged batch by name. A staged row with a NULL key matches nothing, as
//! `NULL = NULL` is not true in SQL; an upsert still appends it.
//!
//! # Staged types
//!
//! Staged batches are typed Arrow data, and no value is converted as DuckDB
//! converts text: every staged key and value column (found by name) must have
//! its table column's Arrow type, or a type that widens to it without loss:
//!
//! - a signed integer to a wider signed one; an unsigned integer to a wider
//!   unsigned or signed one; `Float32` to `Float64`;
//! - a decimal to one with at least as many integer digits and at least its
//!   scale;
//! - `Utf8`, `LargeUtf8` and `Utf8View` to one another; `Binary`,
//!   `LargeBinary` and `BinaryView` to one another, and `FixedSizeBinary` to
//!   any of them;
//! - `Date32` and `Date64` to one another, a `Date64` value that is not a
//!   whole day refused; a timestamp to the same time zone at the same or a
//!   finer unit;
//! - a list, struct (the same field names, in order) or map whose children
//!   widen under these rules, field ids and metadata ignored.
//!
//! A staged column may be nullable where the table's is `NOT NULL`; a NULL
//! written there fails the mutation with the `NOT NULL constraint failed`
//! error. Any other type fails it with [`DuckLakeError::InvalidConfig`]
//! (``column `id`: expected Int64, got Utf8``) before anything is written.
//!
//! Keys compare in the table column's type, after widening: exactly, and
//! floats numerically at any depth of a nested key, as DuckDB's `DISTINCT`
//! compares them (`-0.0` equals `0`, and NaN equals NaN).
//!
//! # Cost
//!
//! The staged keys are held in memory. Files are pruned by the catalog
//! statistics of the key columns (each key column's staged minimum and
//! maximum), so a mutation touching a few keys of a large table reads only the
//! files whose bounds admit them; each remaining file's key columns are read
//! once, `concurrency` files at a time
//! ([`KeyedMutation::with_concurrency`]).
//!
//! # Example
//!
//! ```no_run
//! # async fn example(
//! #     ctx: &datafusion::prelude::SessionContext,
//! #     table: &datafusion_ducklake::DuckLakeTable,
//! #     staged: Vec<arrow::record_batch::RecordBatch>,
//! # ) -> datafusion_ducklake::Result<()> {
//! let state = ctx.state();
//! let result = table
//!     .keyed_mutation(&["id"])
//!     .upsert_by_keys(&state, staged)
//!     .await?;
//! println!("{} replaced, {} written", result.rows_deleted, result.rows_inserted);
//! # Ok(())
//! # }
//! ```

use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, BooleanArray, BooleanBuilder, RecordBatch};
use arrow::compute::take;
use arrow::datatypes::{DataType, Schema, SchemaRef};
use arrow::row::{RowConverter, SortField};
use datafusion::catalog::Session;
use datafusion::common::ScalarValue;
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::logical_expr::{ColumnarValue, Operator};
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_expr::expressions::{BinaryExpr, Column, Literal};
use futures::future::BoxFuture;

use crate::rowid_lookup::bounded;

use crate::DuckLakeError;
use crate::delete_exec::{DeletionTarget, FileDeletions, LatestDeleteFiles};
use crate::metadata_provider::DuckLakeTableFile;
use crate::metadata_writer::{InlinedRowRef, StagedTableData, WriteMode};
use crate::table::{DuckLakeTable, UpdateSourceScan};
use crate::table_writer::DuckLakeTableWriter;

/// What a keyed mutation did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct KeyedMutationResult {
    /// The snapshot the mutation committed, or `None` when it changed nothing
    /// and committed no snapshot.
    pub snapshot_id: Option<i64>,
    /// Rows deleted: by [`KeyedMutation::delete_by_keys`], or replaced by
    /// [`KeyedMutation::upsert_by_keys`].
    pub rows_deleted: u64,
    /// Rows [`KeyedMutation::update_by_keys`] gave new values.
    pub rows_updated: u64,
    /// Rows [`KeyedMutation::upsert_by_keys`] appended.
    pub rows_inserted: u64,
}

/// A keyed mutation of one table, from [`DuckLakeTable::keyed_mutation`].
///
/// See the [module docs](self) for the semantics of each operation. The
/// table must have been opened from a writable catalog
/// ([`DuckLakeCatalog::with_writer`](crate::DuckLakeCatalog::with_writer)), on
/// a writer that stages row mutations (SQLite, DuckDB, MySQL and
/// multi-catalog PostgreSQL); it reads the table at the snapshot it was opened
/// at, and a conflicting commit since aborts the mutation with
/// [`DuckLakeError::Conflict`].
#[derive(Debug)]
pub struct KeyedMutation<'a> {
    table: &'a DuckLakeTable,
    key_columns: Vec<String>,
    concurrency: Option<usize>,
    /// Set for [`DuckLakeTable::delete_positions`] and
    /// [`DuckLakeTable::delete_rowids`], which name rows without keys.
    by_position: bool,
}

impl DuckLakeTable {
    /// A keyed mutation of this table on `key_columns` (names of its
    /// columns); see [`KeyedMutation`].
    pub fn keyed_mutation(&self, key_columns: &[&str]) -> KeyedMutation<'_> {
        KeyedMutation {
            table: self,
            key_columns: key_columns.iter().map(|name| (*name).to_string()).collect(),
            concurrency: None,
            by_position: false,
        }
    }

    /// Delete the rows at `positions`, `(data_file_id, physical positions)` of
    /// this table's live data files, in one snapshot, recording them as
    /// official DuckLake's `DELETE` records them (see the
    /// [`keyed_mutation`](crate::keyed_mutation) module docs): a file's
    /// positions that number at most the table's `data_inlining_row_limit` go
    /// to `ducklake_inlined_delete_<table_id>`, even when they are every row of
    /// the file; above the limit, a file they empty is ended, a file with a
    /// live delete file gets a new one carrying every position with its
    /// snapshot, and any other file a delete file of its positions.
    /// Positions already deleted are skipped; a file not live at this table's
    /// snapshot, or a position past its rows, is refused with
    /// [`DuckLakeError::InvalidConfig`].
    ///
    /// This is the inlining-aware form of writing a delete file
    /// ([`DuckLakeTableWriter::write_delete_file`]) and committing it with
    /// [`MetadataWriter::set_delete_file`](crate::MetadataWriter::set_delete_file).
    pub async fn delete_positions(
        &self,
        state: &dyn Session,
        positions: Vec<(i64, Vec<i64>)>,
    ) -> crate::Result<KeyedMutationResult> {
        self.delete_located(state, positions, Vec::new()).await
    }

    /// Delete the live rows with `row_ids` at this table's snapshot, wherever
    /// they live ([`DuckLakeTable::locate_rowids`]): inlined rows are ended,
    /// Parquet rows deleted as [`Self::delete_positions`] deletes them, in one
    /// snapshot. Ids no live row has are ignored.
    pub async fn delete_rowids(
        &self,
        state: &dyn Session,
        row_ids: &[i64],
    ) -> crate::Result<KeyedMutationResult> {
        let mut positions: HashMap<i64, Vec<i64>> = HashMap::new();
        let mut inlined = Vec::new();
        for row in self.locate_rowids(state, row_ids).await? {
            match row.location {
                crate::rowid_lookup::RowLocation::Inlined {
                    table_name,
                } => {
                    inlined.push(InlinedRowRef {
                        table_name,
                        row_id: row.row_id,
                    });
                },
                crate::rowid_lookup::RowLocation::File {
                    data_file_id,
                    position,
                    ..
                } => positions.entry(data_file_id).or_default().push(position),
            }
        }
        self.delete_located(state, positions.into_iter().collect(), inlined)
            .await
    }

    async fn delete_located(
        &self,
        state: &dyn Session,
        positions: Vec<(i64, Vec<i64>)>,
        inlined_rows: Vec<InlinedRowRef>,
    ) -> crate::Result<KeyedMutationResult> {
        let mutation = KeyedMutation {
            table: self,
            key_columns: Vec::new(),
            concurrency: None,
            by_position: true,
        };
        let context = mutation.context(state)?;
        let mut live: HashMap<i64, DuckLakeTableFile> = self
            .files()?
            .into_iter()
            .map(|file| (file.data_file_id, file))
            .collect();
        let inlined_deletes = self.inlined_deletes_by_file().map_err(from_df)?;
        let mut targets = Vec::with_capacity(positions.len());
        for (data_file_id, positions) in positions {
            let file = live.remove(&data_file_id).ok_or_else(|| {
                DuckLakeError::InvalidConfig(format!(
                    "data file {data_file_id} is not a live file of '{}' at snapshot {}",
                    self.table_name(),
                    self.snapshot_id()
                ))
            })?;
            // A file whose catalog row records no count is bounded by its
            // footer's.
            let rows = match file.max_row_count {
                Some(rows) => rows,
                None => self
                    .data_file_row_count(state, &file.file)
                    .await
                    .map_err(from_df)?,
            };
            if let Some(position) = positions.iter().find(|p| **p < 0 || **p >= rows) {
                return Err(DuckLakeError::InvalidConfig(format!(
                    "position {position} is outside data file {data_file_id}'s {rows} rows"
                )));
            }
            targets.push((file, positions));
        }
        let table = self;
        let pending: Vec<BoxFuture<'_, DataFusionResult<Option<FileMatch>>>> = targets
            .into_iter()
            .map(|(file, positions)| {
                let inlined = inlined_deletes.get(&file.data_file_id);
                let pending: BoxFuture<'_, DataFusionResult<Option<FileMatch>>> =
                    Box::pin(async move {
                        let existing = match &file.delete_file {
                            Some(delete_file) => {
                                table.read_delete_file_entries(state, delete_file).await?
                            },
                            None => HashMap::new(),
                        };
                        let mut positions: Vec<i64> = positions
                            .into_iter()
                            .filter(|position| {
                                !existing.contains_key(position)
                                    && !inlined.is_some_and(|set| set.contains(position))
                            })
                            .collect();
                        positions.sort_unstable();
                        positions.dedup();
                        Ok((!positions.is_empty()).then_some(FileMatch {
                            file,
                            existing,
                            positions,
                        }))
                    });
                pending
            })
            .collect();
        let files: Vec<FileMatch> = bounded(pending, mutation.concurrency(state))
            .await
            .map_err(from_df)?
            .into_iter()
            .flatten()
            .collect();
        let matched = MatchedRows {
            files,
            inlined_rows,
        };
        let rows_deleted = matched.row_count();
        if rows_deleted == 0 {
            return Ok(KeyedMutationResult::default());
        }
        let snapshot_id = mutation
            .commit(&context, state, matched, StagedTableData::None)
            .await?;
        Ok(KeyedMutationResult {
            snapshot_id,
            rows_deleted,
            ..KeyedMutationResult::default()
        })
    }
}

impl<'a> KeyedMutation<'a> {
    /// Read at most `files` data files at once while matching keys. Defaults
    /// to the session's `target_partitions`.
    #[must_use]
    pub fn with_concurrency(mut self, files: usize) -> Self {
        self.concurrency = Some(files.max(1));
        self
    }

    /// Delete every row whose key equals the key of a row of `keys`, which
    /// needs only the key columns, found by name (other columns are ignored).
    /// Each key column must have its table column's type or a lossless
    /// widening of it ([module docs](self)).
    pub async fn delete_by_keys(
        self,
        state: &dyn Session,
        keys: Vec<RecordBatch>,
    ) -> crate::Result<KeyedMutationResult> {
        let context = self.context(state)?;
        let keys = context.conform(&keys, &context.key_indices, false)?;
        let keys = KeySet::build(&context.physical_schema, &context.key_indices, &keys)?;
        if keys.is_empty() {
            return Ok(KeyedMutationResult::default());
        }
        let matched = self.match_rows(&context, state, &keys).await?;
        let rows_deleted = matched.row_count();
        if rows_deleted == 0 {
            return Ok(KeyedMutationResult::default());
        }
        let snapshot_id = self
            .commit(&context, state, matched, StagedTableData::None)
            .await?;
        Ok(KeyedMutationResult {
            snapshot_id,
            rows_deleted,
            ..KeyedMutationResult::default()
        })
    }

    /// Delete every row whose key equals the key of a row of `rows`, and
    /// append every row of `rows`, in one snapshot. `rows` must carry every
    /// column of the table, matched by name, each of its table column's type
    /// or a lossless widening of it ([module docs](self)). The appended rows
    /// are laid out as official's insert lays them out under the table's
    /// stored `sort_on_insert`
    /// ([`DuckLakeTableWriter::table_sort_on_insert`]): in its `SET SORTED
    /// BY` order with the option on; with it off, sorted only when they go to
    /// Parquet while inlining is enabled. Official has no upsert; this
    /// follows its `DELETE` followed by `INSERT`, the statements an upsert is
    /// checked against.
    pub async fn upsert_by_keys(
        self,
        state: &dyn Session,
        rows: Vec<RecordBatch>,
    ) -> crate::Result<KeyedMutationResult> {
        let context = self.context(state)?;
        let all: Vec<usize> = (0..context.physical_schema.fields().len()).collect();
        let rows = context.conform(&rows, &all, true)?;
        let keys = KeySet::build(&context.physical_schema, &context.key_indices, &rows)?;
        let rows_inserted: u64 = rows.iter().map(|batch| batch.num_rows() as u64).sum();
        if rows_inserted == 0 {
            return Ok(KeyedMutationResult::default());
        }
        let matched = if keys.is_empty() {
            MatchedRows::default()
        } else {
            self.match_rows(&context, state, &keys).await?
        };
        let rows_deleted = matched.row_count();
        let data = context.stage_new_rows(rows).await?;
        let snapshot_id = self.commit(&context, state, matched, data).await?;
        Ok(KeyedMutationResult {
            snapshot_id,
            rows_deleted,
            rows_inserted,
            ..KeyedMutationResult::default()
        })
    }

    /// Give every row whose key equals the key of a row of `rows` that row's
    /// values, keeping its `rowid`, in one snapshot, as `MERGE INTO t USING
    /// rows ON k... WHEN MATCHED THEN UPDATE SET *`, values matched to the
    /// table's columns by name. Rows of `rows` matching no row of the table are
    /// ignored. `rows` must carry every column of the table, each of its table
    /// column's type or a lossless widening of it ([module docs](self)), every
    /// row checked before anything is matched. Several rows of `rows` may share
    /// a key, as
    /// `MERGE` accepts them: a matched row takes the last one's values (official
    /// leaves the choice to its execution order; a small source updates in
    /// source order) and counts once in
    /// [`rows_updated`](KeyedMutationResult::rows_updated), where `MERGE`
    /// counts every source row. The new versions are not sorted by the
    /// table's `SET SORTED BY` order, as official's `UPDATE` and `MERGE` sort
    /// none.
    pub async fn update_by_keys(
        self,
        state: &dyn Session,
        rows: Vec<RecordBatch>,
    ) -> crate::Result<KeyedMutationResult> {
        let context = self.context(state)?;
        let all: Vec<usize> = (0..context.physical_schema.fields().len()).collect();
        let staged = context.conform(&rows, &all, true)?;
        let keys = KeySet::build(&context.physical_schema, &context.key_indices, &staged)?;
        if keys.is_empty() {
            return Ok(KeyedMutationResult::default());
        }
        let predicate = keys.predicate(&context.physical_schema, &context.key_indices)?;
        let files = self.table.files_matching(&predicate.range_or_true())?;
        let inlined_deletes = self.table.inlined_deletes_by_file().map_err(from_df)?;
        let task_ctx = state.task_ctx();
        let concurrency = self.concurrency(state);
        let table = self.table;
        let combined = predicate.combined();
        type Scanned = (UpdateSourceScan, Vec<RecordBatch>, Vec<i64>);
        let pending: Vec<BoxFuture<'_, DataFusionResult<Scanned>>> = files
            .into_iter()
            .map(|file| {
                let inlined = inlined_deletes.get(&file.data_file_id);
                let combined = Arc::clone(&combined);
                let task_ctx = Arc::clone(&task_ctx);
                let staged = &staged;
                let keys = &keys;
                let context = &context;
                let pending: BoxFuture<'_, DataFusionResult<Scanned>> = Box::pin(async move {
                    let scan = table
                        .build_update_scan_filtered(state, &file, inlined, Some(&combined))
                        .await?;
                    let batches =
                        datafusion::physical_plan::collect(Arc::clone(&scan.scan), task_ctx)
                            .await?;
                    let out =
                        table.apply_update_to_batches(&scan, &batches, Some(&combined), &[])?;
                    let mut versions = Vec::with_capacity(out.updated_batches.len());
                    for batch in &out.updated_batches {
                        versions.push(context.new_versions(batch, staged, keys)?);
                    }
                    Ok((scan, versions, out.matched_positions))
                });
                pending
            })
            .collect();
        let scans: Vec<Scanned> = bounded(pending, concurrency)
            .await
            .map_err(from_df)?
            .into_iter()
            .filter(|(_, _, positions)| !positions.is_empty())
            .collect();

        let mut updated_batches = Vec::new();
        let mut superseded_inlined = Vec::new();
        for rows in self.table.inlined_data_with_row_ids()? {
            if let Some((batch, superseded)) =
                self.table
                    .apply_update_to_inlined(&rows, Some(&combined), &[])?
            {
                updated_batches.push(
                    context
                        .new_versions(&batch, &staged, &keys)
                        .map_err(from_df)?,
                );
                superseded_inlined.extend(superseded);
            }
        }
        let mut superseded_positions = Vec::with_capacity(scans.len());
        for (scan, versions, positions) in &scans {
            updated_batches.extend(versions.iter().cloned());
            superseded_positions.push((scan, positions.clone()));
        }
        let rows_updated: u64 = updated_batches
            .iter()
            .map(|batch| batch.num_rows() as u64)
            .sum();
        if rows_updated == 0 {
            return Ok(KeyedMutationResult::default());
        }
        let snapshot_id = crate::update_exec::commit_row_versions(
            &crate::update_exec::RowVersionCommit {
                table: self.table,
                table_writer: &context.table_writer,
                context: state.task_ctx(),
                schema_name: &context.schema_name,
                table_name: self.table.table_name(),
                base_snapshot: self.table.snapshot_id(),
                inline_capable: true,
                inline_limit: context.inline_limit,
            },
            updated_batches,
            superseded_positions,
            Vec::new(),
            superseded_inlined,
            rows_updated,
        )
        .await
        .map_err(from_df)?;
        Ok(KeyedMutationResult {
            snapshot_id,
            rows_updated,
            ..KeyedMutationResult::default()
        })
    }

    fn concurrency(&self, state: &dyn Session) -> usize {
        self.concurrency
            .unwrap_or_else(|| state.config().target_partitions().max(1))
    }

    fn context(&self, state: &dyn Session) -> crate::Result<MutationContext> {
        let table = self.table;
        let writer = table.writer().cloned().ok_or_else(|| {
            DuckLakeError::InvalidConfig(format!(
                "keyed mutation of '{}': the table was not opened from a writable catalog",
                table.table_name()
            ))
        })?;
        if !writer.supports_inlined_row_mutations() {
            return Err(DuckLakeError::Unsupported(
                "keyed mutations need a writer that stages row mutations (SQLite, DuckDB, \
                 MySQL or multi-catalog PostgreSQL)"
                    .to_string(),
            ));
        }
        let schema_name = table
            .schema_name()
            .ok_or_else(|| {
                DuckLakeError::InvalidConfig(format!(
                    "keyed mutation of '{}': the table has no schema name",
                    table.table_name()
                ))
            })?
            .to_string();
        if self.key_columns.is_empty() && !self.by_position {
            return Err(DuckLakeError::InvalidConfig(
                "a keyed mutation needs at least one key column".to_string(),
            ));
        }
        let physical_schema = table.physical_schema();
        let key_indices = self
            .key_columns
            .iter()
            .map(|name| {
                physical_schema.index_of(name).map_err(|_| {
                    DuckLakeError::InvalidConfig(format!(
                        "key column '{name}' is not a column of '{}'",
                        table.table_name()
                    ))
                })
            })
            .collect::<crate::Result<Vec<_>>>()?;
        let object_store = state
            .runtime_env()
            .object_store(table.object_store_url().as_ref())
            .map_err(|e| DuckLakeError::Internal(e.to_string()))?;
        let table_writer = DuckLakeTableWriter::new(Arc::clone(&writer), object_store)?
            .with_options(table.write_options());
        // The stored table, schema or catalog-wide limit, else the writer's:
        // a table opened without the catalog's options (`DuckLakeTable::new`)
        // carries no resolved limit of its own.
        let inline_limit =
            table_writer.table_data_inlining_row_limit(&schema_name, table.table_name())?;
        let sort_spec = table.live_sort_spec()?;
        // Read only for a sorted table: they decide only how its rows sort.
        let (sort_on_insert, inlining_enabled) = match &sort_spec {
            Some(_) => {
                let on = table_writer.table_sort_on_insert(&schema_name, table.table_name())?;
                let enabled = !on
                    && table_writer.inlining_enabled(
                        &schema_name,
                        table.table_name(),
                        &physical_schema,
                    )?;
                (on, enabled)
            },
            None => (false, false),
        };
        Ok(MutationContext {
            table_writer,
            schema_name,
            physical_schema,
            key_indices,
            inline_limit,
            sort_spec,
            sort_on_insert,
            inlining_enabled,
            table_name: table.table_name().to_string(),
            base_snapshot: table.snapshot_id(),
        })
    }

    /// Every live row of the table whose key is in `keys`: per data file the
    /// positions not deleted already, and the inlined rows.
    async fn match_rows(
        &self,
        context: &MutationContext,
        state: &dyn Session,
        keys: &KeySet,
    ) -> crate::Result<MatchedRows> {
        let predicate = keys.predicate(&context.physical_schema, &context.key_indices)?;
        let files = self.table.files_matching(&predicate.range_or_true())?;
        let inlined_deletes = self.table.inlined_deletes_by_file().map_err(from_df)?;
        let combined = predicate.combined();
        let table = self.table;
        let pending: Vec<BoxFuture<'_, DataFusionResult<Option<FileMatch>>>> = files
            .into_iter()
            .map(|file| {
                let combined = Arc::clone(&combined);
                let inlined = inlined_deletes.get(&file.data_file_id);
                let pending: BoxFuture<'_, DataFusionResult<Option<FileMatch>>> =
                    Box::pin(async move {
                        let matched = table.resolve_positions(state, &file.file, combined).await?;
                        if matched.is_empty() {
                            return Ok(None);
                        }
                        // A merged file read before its `partial_max` holds rows
                        // of later snapshots, which are not rows of the table here,
                        // as the scan leaves them out.
                        let later = table.positions_inserted_later(state, &file).await?;
                        let existing = match &file.delete_file {
                            Some(delete_file) => {
                                table.read_delete_file_entries(state, delete_file).await?
                            },
                            None => HashMap::new(),
                        };
                        let mut positions: Vec<i64> = matched
                            .into_iter()
                            .filter(|position| {
                                !later.contains(position)
                                    && !existing.contains_key(position)
                                    && !inlined.is_some_and(|set| set.contains(position))
                            })
                            .collect();
                        if positions.is_empty() {
                            return Ok(None);
                        }
                        positions.sort_unstable();
                        Ok(Some(FileMatch {
                            file,
                            existing,
                            positions,
                        }))
                    });
                pending
            })
            .collect();
        let files: Vec<FileMatch> = bounded(pending, self.concurrency(state))
            .await
            .map_err(from_df)?
            .into_iter()
            .flatten()
            .collect();

        let mut inlined_rows = Vec::new();
        for rows in self.table.inlined_data_with_row_ids()? {
            let batch = coerce_to_schema(&rows.batch, &context.physical_schema)?;
            let mask = keys.contains(&batch)?;
            for (row, row_id) in rows.row_ids.iter().enumerate() {
                if mask.value(row) {
                    inlined_rows.push(InlinedRowRef {
                        table_name: rows.table_name.clone(),
                        row_id: *row_id,
                    });
                }
            }
        }
        Ok(MatchedRows {
            files,
            inlined_rows,
        })
    }

    /// Record `matched` as official records a `DELETE`'s deletions and commit
    /// them with `data`, the appended rows, in one snapshot.
    async fn commit(
        &self,
        context: &MutationContext,
        state: &dyn Session,
        matched: MatchedRows,
        data: StagedTableData,
    ) -> crate::Result<Option<i64>> {
        let table = self.table;
        let base_snapshot = context.base_snapshot;
        let commit_snapshot = context
            .table_writer
            .next_snapshot_id()?
            .map_or(base_snapshot + 1, |next| next.max(base_snapshot + 1));
        let pending: Vec<BoxFuture<'_, DataFusionResult<FileDeletions>>> = matched
            .files
            .iter()
            .map(|found| -> BoxFuture<'_, DataFusionResult<FileDeletions>> {
                Box::pin(async move {
                    let mut deletions = FileDeletions::dated(commit_snapshot);
                    deletions
                        .record(
                            table,
                            &context.table_writer,
                            &context.schema_name,
                            &context.table_name,
                            &DeletionTarget::of_file(&found.file, &found.existing),
                            found.positions.clone(),
                            context.inline_limit,
                            base_snapshot,
                        )
                        .await?;
                    Ok(deletions)
                })
            })
            .collect();
        let recorded = bounded(pending, self.concurrency(state))
            .await
            .map_err(from_df)?;
        let mut deletions = FileDeletions::default();
        for recorded in recorded {
            deletions.absorb(recorded);
        }
        let latest = LatestDeleteFiles {
            table,
            state,
        };
        let mut transaction = context.table_writer.transaction();
        transaction.set_delete_file_positions(&latest);
        transaction
            .stage_row_mutation(
                &context.schema_name,
                &context.table_name,
                table.table_id(),
                table.column_ids(),
                context.physical_schema.as_ref(),
                base_snapshot,
                data,
                None,
                deletions.positional,
                deletions.rewritten,
                matched.inlined_rows,
                deletions.inlined,
                deletions.dropped,
            )
            .await?;
        Ok(transaction
            .commit()
            .await?
            .first()
            .map(|result| result.snapshot_id))
    }
}

/// What a keyed mutation works with, resolved once.
struct MutationContext {
    table_writer: DuckLakeTableWriter,
    schema_name: String,
    table_name: String,
    physical_schema: SchemaRef,
    key_indices: Vec<usize>,
    /// The table's scoped `data_inlining_row_limit`.
    inline_limit: usize,
    /// The table's `SET SORTED BY` order at the head, if any.
    sort_spec: Option<crate::sort::SortSpec>,
    /// The table's `sort_on_insert`
    /// ([`DuckLakeTableWriter::table_sort_on_insert`]); read only when
    /// `sort_spec` is set.
    sort_on_insert: bool,
    /// Whether inlining is enabled for the table
    /// ([`DuckLakeTableWriter::inlining_enabled`]); read only when `sort_spec`
    /// is set and `sort_on_insert` off.
    inlining_enabled: bool,
    base_snapshot: i64,
}

impl MutationContext {
    /// The non-empty `batches`' columns `indices` (of the table's physical
    /// schema), found by name and conformed to the table's types
    /// ([`conform_column`]), in a schema of those columns made nullable. A
    /// missing column, or one whose type does not widen to the table's
    /// without loss, is refused for every batch before any is converted;
    /// `exact` refuses a batch with columns besides them too.
    fn conform(
        &self,
        batches: &[RecordBatch],
        indices: &[usize],
        exact: bool,
    ) -> crate::Result<Vec<RecordBatch>> {
        let mut found = Vec::with_capacity(batches.len());
        for batch in batches {
            let schema = batch.schema();
            if exact && schema.fields().len() != indices.len() {
                return Err(DuckLakeError::InvalidConfig(format!(
                    "staged rows have {} columns, table '{}' has {}",
                    schema.fields().len(),
                    self.table_name,
                    indices.len()
                )));
            }
            let columns = indices
                .iter()
                .map(|index| {
                    let field = self.physical_schema.field(*index);
                    let position = schema.index_of(field.name()).map_err(|_| {
                        DuckLakeError::InvalidConfig(format!(
                            "staged rows have no column '{}'",
                            field.name()
                        ))
                    })?;
                    check_staged_type(
                        field.name(),
                        schema.field(position).data_type(),
                        field.data_type(),
                    )?;
                    Ok(position)
                })
                .collect::<crate::Result<Vec<_>>>()?;
            found.push(columns);
        }
        let fields: Vec<_> = indices
            .iter()
            .map(|index| self.physical_schema.field(*index).clone())
            .collect();
        let schema = permissive(&Arc::new(Schema::new(fields)));
        batches
            .iter()
            .zip(found)
            .filter(|(batch, _)| batch.num_rows() > 0)
            .map(|(batch, positions)| {
                let columns = positions
                    .iter()
                    .zip(schema.fields())
                    .map(|(position, field)| conform_column(batch.column(*position), field))
                    .collect::<crate::Result<Vec<_>>>()?;
                Ok(RecordBatch::try_new(Arc::clone(&schema), columns)?)
            })
            .collect()
    }

    /// The appended rows as staged data: inlined when they number at most the
    /// limit and the backend inlines them, else written to Parquet files the
    /// commit registers.
    async fn stage_new_rows(&self, rows: Vec<RecordBatch>) -> crate::Result<StagedTableData> {
        // The shared validator names a NULL in a NOT NULL column before the
        // table's schema, which refuses it with an Arrow error, is applied.
        crate::table_writer::validate_not_null_batches(&self.physical_schema, &rows)?;
        let rows = rows
            .into_iter()
            .map(|batch| {
                RecordBatch::try_new(Arc::clone(&self.physical_schema), batch.columns().to_vec())
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let count: usize = rows.iter().map(RecordBatch::num_rows).sum();
        let inline = self.table_writer.should_inline(
            &self.schema_name,
            &self.table_name,
            count,
            &self.physical_schema,
            &rows,
        )?;
        // In the table's `SET SORTED BY` order, as official's insert lays them
        // out (`DuckLakeCatalog::PlanInsert`), its keys' columns validated
        // whenever that insert plans the sort, inlined rows or not.
        if let Some(spec) = &self.sort_spec
            && (self.sort_on_insert || self.inlining_enabled)
        {
            crate::sort::validate_sort_columns(spec, &self.physical_schema)?;
        }
        let sort =
            crate::table_writer::insert_sorts(self.sort_on_insert, self.inlining_enabled, inline);
        let rows = match self.sort_spec.as_ref().filter(|_| sort) {
            Some(spec) => crate::table_writer::sort_insert_rows(rows, &self.physical_schema, spec)?,
            None => rows,
        };
        if inline {
            return Ok(StagedTableData::Inlined(rows));
        }
        let mut session = self
            .table_writer
            .begin_write(
                &self.schema_name,
                &self.table_name,
                &self.physical_schema,
                WriteMode::Append,
            )?
            .with_base_snapshot_id(self.base_snapshot);
        for batch in &rows {
            session.write_batch_async(batch).await?;
        }
        Ok(session
            .finish_staged()
            .await?
            .unwrap_or(StagedTableData::None))
    }

    /// The new versions of the matched rows `matched` (`[physical
    /// columns..., rowid]`): each takes the values of the staged row (of
    /// `staged`, from [`Self::conform`]) with its key, and keeps its `rowid`.
    fn new_versions(
        &self,
        matched: &RecordBatch,
        staged: &[RecordBatch],
        keys: &KeySet,
    ) -> DataFusionResult<RecordBatch> {
        let width = matched.num_columns();
        let data = matched.project(&(0..width - 1).collect::<Vec<_>>())?;
        // Each matched row's staged row, as (batch, row in batch).
        let mut starts = Vec::with_capacity(staged.len());
        let mut start = 0usize;
        for batch in staged {
            starts.push(start);
            start += batch.num_rows();
        }
        let mut wanted: Vec<Vec<u64>> = vec![Vec::new(); staged.len()];
        let mut picks = Vec::with_capacity(data.num_rows());
        for index in keys.staged_rows(&data)? {
            let batch = starts.partition_point(|start| *start <= index) - 1;
            picks.push((batch, wanted[batch].len()));
            wanted[batch].push((index - starts[batch]) as u64);
        }
        let mut columns = Vec::with_capacity(width);
        for column in 0..self.physical_schema.fields().len() {
            let parts = staged
                .iter()
                .zip(&wanted)
                .map(|(batch, rows)| {
                    let rows = arrow::array::UInt64Array::from(rows.clone());
                    take(batch.column(column).as_ref(), &rows, None)
                })
                .collect::<Result<Vec<ArrayRef>, _>>()?;
            let parts: Vec<&dyn Array> = parts.iter().map(AsRef::as_ref).collect();
            columns.push(arrow::compute::interleave(&parts, &picks)?);
        }
        crate::table_writer::validate_not_null_batches(
            &self.physical_schema,
            &[RecordBatch::try_new(permissive(&self.physical_schema), columns.clone())?],
        )
        .map_err(to_df)?;
        columns.push(Arc::clone(matched.column(width - 1)));
        Ok(RecordBatch::try_new(matched.schema(), columns)?)
    }
}

/// `schema` with every field nullable, so staged rows with NULLs in a NOT
/// NULL column reach the shared validator, which names the column.
fn permissive(schema: &SchemaRef) -> SchemaRef {
    Arc::new(Schema::new(
        schema
            .fields()
            .iter()
            .map(|field| Arc::new(field.as_ref().clone().with_nullable(true)))
            .collect::<Vec<_>>(),
    ))
}

/// `batch` (inlined rows in catalog column order) in `schema`'s types.
fn coerce_to_schema(batch: &RecordBatch, schema: &SchemaRef) -> crate::Result<RecordBatch> {
    let columns = batch
        .columns()
        .iter()
        .zip(schema.fields())
        .map(|(column, field)| crate::column_rename::coerce_column(column, field.data_type()))
        .collect::<DataFusionResult<Vec<_>>>()
        .map_err(from_df)?;
    Ok(RecordBatch::try_new(permissive(schema), columns)?)
}

/// One data file's matched rows.
struct FileMatch {
    file: DuckLakeTableFile,
    /// The positions its live delete file deletes, with their snapshots.
    existing: HashMap<i64, Option<i64>>,
    /// Matched positions not deleted already, sorted.
    positions: Vec<i64>,
}

/// The live rows a keyed mutation matched.
#[derive(Default)]
struct MatchedRows {
    files: Vec<FileMatch>,
    inlined_rows: Vec<InlinedRowRef>,
}

impl MatchedRows {
    fn row_count(&self) -> u64 {
        self.files
            .iter()
            .map(|file| file.positions.len() as u64)
            .sum::<u64>()
            + self.inlined_rows.len() as u64
    }
}

/// The distinct non-NULL keys of a staged batch, encoded with Arrow's row
/// format, each with the staged row that carries it.
struct KeySet {
    keys: Arc<KeySetRef>,
    /// The key columns' names, by which every batch's key columns are found.
    names: Vec<String>,
    /// Per key column, the smallest and largest staged value, where the type
    /// orders values the way file statistics do.
    bounds: Vec<Option<(ScalarValue, ScalarValue)>>,
    /// The distinct staged values of a single key column: the membership test
    /// is then DataFusion's `IN` list, which the Parquet reader can prune row
    /// groups with.
    single: Option<Vec<ScalarValue>>,
}

impl fmt::Debug for KeySet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KeySet")
            .field("types", &self.keys.0.types)
            .field("keys", &self.keys.0.rows.len())
            .finish_non_exhaustive()
    }
}

impl KeySet {
    /// The keys of `batches` (from [`MutationContext::conform`], so in the
    /// table's types), whose key columns are `schema`'s `key_indices`. A key
    /// staged twice keeps its last row.
    fn build(
        schema: &SchemaRef,
        key_indices: &[usize],
        batches: &[RecordBatch],
    ) -> crate::Result<Self> {
        let names: Vec<String> = key_indices
            .iter()
            .map(|index| schema.field(*index).name().clone())
            .collect();
        let types: Vec<DataType> = key_indices
            .iter()
            .map(|index| schema.field(*index).data_type().clone())
            .collect();
        let converter = RowConverter::new(
            types
                .iter()
                .map(|data_type| SortField::new(data_type.clone()))
                .collect(),
        )?;
        let mut rows = HashMap::new();
        let accumulator = |max: bool| {
            types
                .iter()
                .map(|data_type| {
                    if !orderable(data_type) {
                        return Ok(None);
                    }
                    Ok(Some(if max {
                        Box::new(
                            datafusion::functions_aggregate::min_max::MaxAccumulator::try_new(
                                data_type,
                            )?,
                        ) as Box<dyn datafusion::logical_expr::Accumulator>
                    } else {
                        Box::new(
                            datafusion::functions_aggregate::min_max::MinAccumulator::try_new(
                                data_type,
                            )?,
                        )
                    }))
                })
                .collect::<DataFusionResult<Vec<_>>>()
                .map_err(from_df)
        };
        let mut min = accumulator(false)?;
        let mut max = accumulator(true)?;
        let mut offset: usize = 0;
        // An `IN` list compares floats by their bits, not as DuckDB does.
        let mut single = (key_indices.len() == 1 && !has_float(&types[0])).then(Vec::new);
        for batch in batches {
            let columns = staged_key_columns(batch, &names)?;
            let encoded = converter.convert_columns(&columns)?;
            let valid = all_valid(&columns, batch.num_rows());
            for row in 0..batch.num_rows() {
                if !valid.value(row) {
                    continue;
                }
                let index = offset + row;
                let previous =
                    rows.insert(encoded.row(row).as_ref().to_vec().into_boxed_slice(), index);
                if previous.is_none()
                    && let Some(values) = single.as_mut()
                {
                    values.push(ScalarValue::try_from_array(&columns[0], row).map_err(from_df)?);
                }
            }
            for (position, column) in columns.iter().enumerate() {
                if let Some(acc) = min[position].as_mut() {
                    acc.update_batch(std::slice::from_ref(column))
                        .map_err(from_df)?;
                }
                if let Some(acc) = max[position].as_mut() {
                    acc.update_batch(std::slice::from_ref(column))
                        .map_err(from_df)?;
                }
            }
            offset += batch.num_rows();
        }
        let bounds = min
            .iter_mut()
            .zip(max.iter_mut())
            .map(|(min, max)| match (min.as_mut(), max.as_mut()) {
                (Some(min), Some(max)) => {
                    let (min, max) = (min.evaluate()?, max.evaluate()?);
                    Ok((!min.is_null() && !max.is_null()).then_some((min, max)))
                },
                _ => Ok(None),
            })
            .collect::<DataFusionResult<Vec<_>>>()
            .map_err(from_df)?;
        Ok(Self {
            keys: Arc::new(KeySetRef(SharedKeys {
                converter,
                types,
                rows,
            })),
            names,
            bounds,
            single,
        })
    }

    fn is_empty(&self) -> bool {
        self.keys.0.rows.is_empty()
    }

    /// Which rows of `batch` (in the table's column order) carry a staged key.
    fn contains(&self, batch: &RecordBatch) -> DataFusionResult<BooleanArray> {
        let keys = &self.keys.0;
        let columns = table_key_columns(batch, &self.names, &keys.types).map_err(to_df)?;
        keys.contains(&columns, batch.num_rows())
    }

    /// For each row of `batch` (carrying a staged key), the staged row with it.
    fn staged_rows(&self, batch: &RecordBatch) -> DataFusionResult<Vec<usize>> {
        let keys = &self.keys.0;
        let columns = table_key_columns(batch, &self.names, &keys.types).map_err(to_df)?;
        let encoded = keys.converter.convert_columns(&columns)?;
        (0..batch.num_rows())
            .map(|row| {
                keys.rows
                    .get(encoded.row(row).as_ref())
                    .copied()
                    .ok_or_else(|| {
                        DataFusionError::Internal("a matched row carries no staged key".to_string())
                    })
            })
            .collect()
    }

    /// The predicate over the table's physical columns a matched row
    /// satisfies: each key column within the staged bounds, and the key in
    /// this set.
    fn predicate(&self, schema: &SchemaRef, key_indices: &[usize]) -> crate::Result<KeyPredicate> {
        let mut range: Option<Arc<dyn PhysicalExpr>> = None;
        for (position, index) in key_indices.iter().enumerate() {
            let Some((min, max)) = &self.bounds[position] else {
                continue;
            };
            let column: Arc<dyn PhysicalExpr> =
                Arc::new(Column::new(schema.field(*index).name(), *index));
            let lower: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
                Arc::clone(&column),
                Operator::GtEq,
                Arc::new(Literal::new(min.clone())),
            ));
            let upper: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
                column,
                Operator::LtEq,
                Arc::new(Literal::new(max.clone())),
            ));
            let bound: Arc<dyn PhysicalExpr> =
                Arc::new(BinaryExpr::new(lower, Operator::And, upper));
            range = Some(match range {
                None => bound,
                Some(range) => Arc::new(BinaryExpr::new(range, Operator::And, bound)),
            });
        }
        if let (Some(values), [index]) = (&self.single, key_indices) {
            // DataFusion's pruning rewrites an `IN` list of at most this many
            // values; a longer one prunes by the bounds beside it.
            const PRUNABLE_IN_LIST: usize = 20;
            let list = values
                .iter()
                .map(|value| Arc::new(Literal::new(value.clone())) as Arc<dyn PhysicalExpr>)
                .collect();
            let member = datafusion::physical_expr::expressions::InListExpr::try_new(
                Arc::new(Column::new(schema.field(*index).name(), *index)),
                list,
                false,
                schema,
            )
            .map_err(from_df)?;
            return Ok(KeyPredicate {
                in_list: values.len() <= PRUNABLE_IN_LIST || range.is_none(),
                range,
                member: Arc::new(member),
            });
        }
        Ok(KeyPredicate {
            in_list: false,
            range,
            member: Arc::new(KeyMembership {
                columns: key_indices
                    .iter()
                    .map(|index| {
                        Arc::new(Column::new(schema.field(*index).name(), *index))
                            as Arc<dyn PhysicalExpr>
                    })
                    .collect(),
                keys: Arc::clone(&self.keys),
            }),
        })
    }
}

/// The predicate a keyed mutation matches rows with.
struct KeyPredicate {
    /// Whether `member` stands alone: an `IN` list over a single key column
    /// short enough for DataFusion's pruning to rewrite, which prunes as well
    /// as the bounds would and needs none beside it.
    in_list: bool,
    /// The key columns' staged bounds, for file and row-group pruning; `None`
    /// when no key column has orderable bounds.
    range: Option<Arc<dyn PhysicalExpr>>,
    /// Membership of the key in the staged set.
    member: Arc<dyn PhysicalExpr>,
}

impl KeyPredicate {
    /// The bounds alone (or an `IN` list), or `true`, for pruning files by
    /// their statistics.
    fn range_or_true(&self) -> Arc<dyn PhysicalExpr> {
        if self.in_list {
            return Arc::clone(&self.member);
        }
        self.range
            .clone()
            .unwrap_or_else(|| Arc::new(Literal::new(ScalarValue::Boolean(Some(true)))))
    }

    /// The bounds and the membership test: what a matched row satisfies. An
    /// `IN` list stands alone.
    fn combined(&self) -> Arc<dyn PhysicalExpr> {
        if self.in_list {
            return Arc::clone(&self.member);
        }
        match &self.range {
            Some(range) => Arc::new(BinaryExpr::new(
                Arc::clone(range),
                Operator::And,
                Arc::clone(&self.member),
            )),
            None => Arc::clone(&self.member),
        }
    }
}

/// Whether file statistics order `data_type`'s values as comparison does, so
/// a staged minimum and maximum can prune files. Floats are left out: a NaN
/// sits outside a file's recorded bounds.
fn orderable(data_type: &DataType) -> bool {
    data_type.is_integer()
        || matches!(
            data_type,
            DataType::Utf8
                | DataType::LargeUtf8
                | DataType::Utf8View
                | DataType::Date32
                | DataType::Date64
                | DataType::Timestamp(_, _)
                | DataType::Decimal128(_, _)
                | DataType::Decimal256(_, _)
                | DataType::Boolean
        )
}

/// Refuse a staged column `name` of type `staged` for a table column of type
/// `table` unless its values widen to `table` without loss ([`widens`]).
fn check_staged_type(name: &str, staged: &DataType, table: &DataType) -> crate::Result<()> {
    if widens(staged, table) {
        return Ok(());
    }
    Err(DuckLakeError::InvalidConfig(format!(
        "column `{name}`: expected {table}, got {staged}"
    )))
}

/// Whether every value of `staged` converts to `table` without loss, by the
/// rules of the [module docs](self): equal types, and the lossless numeric,
/// string, binary, date and timestamp widenings, applied recursively to the
/// children of lists, structs and maps. Nullability, field ids and metadata
/// are not compared.
fn widens(staged: &DataType, table: &DataType) -> bool {
    use DataType::*;
    use arrow::datatypes::TimeUnit;
    let unit_rank = |unit: &TimeUnit| match unit {
        TimeUnit::Second => 0,
        TimeUnit::Millisecond => 1,
        TimeUnit::Microsecond => 2,
        TimeUnit::Nanosecond => 3,
    };
    let decimal = |data_type: &DataType| match data_type {
        Decimal32(precision, scale)
        | Decimal64(precision, scale)
        | Decimal128(precision, scale)
        | Decimal256(precision, scale) => Some((i16::from(*precision), i16::from(*scale))),
        _ => None,
    };
    match (staged, table) {
        (Int8, Int8 | Int16 | Int32 | Int64)
        | (Int16, Int16 | Int32 | Int64)
        | (Int32, Int32 | Int64)
        | (Int64, Int64)
        | (UInt8, UInt8 | UInt16 | UInt32 | UInt64 | Int16 | Int32 | Int64)
        | (UInt16, UInt16 | UInt32 | UInt64 | Int32 | Int64)
        | (UInt32, UInt32 | UInt64 | Int64)
        | (UInt64, UInt64)
        | (Float32, Float32 | Float64)
        | (Float64, Float64)
        | (Utf8 | LargeUtf8 | Utf8View, Utf8 | LargeUtf8 | Utf8View)
        | (
            Binary | LargeBinary | BinaryView | FixedSizeBinary(_),
            Binary | LargeBinary | BinaryView,
        )
        | (Date32 | Date64, Date32 | Date64) => true,
        (FixedSizeBinary(staged), FixedSizeBinary(table)) => staged == table,
        (Timestamp(staged_unit, staged_tz), Timestamp(table_unit, table_tz)) => {
            staged_tz == table_tz && unit_rank(staged_unit) <= unit_rank(table_unit)
        },
        (List(staged) | LargeList(staged), List(table) | LargeList(table)) => {
            widens(staged.data_type(), table.data_type())
        },
        (FixedSizeList(staged, staged_size), FixedSizeList(table, table_size)) => {
            staged_size == table_size && widens(staged.data_type(), table.data_type())
        },
        (Struct(staged), Struct(table)) => {
            staged.len() == table.len()
                && staged.iter().zip(table.iter()).all(|(staged, table)| {
                    staged.name() == table.name() && widens(staged.data_type(), table.data_type())
                })
        },
        (Map(staged, staged_sorted), Map(table, table_sorted)) if staged_sorted == table_sorted => {
            match (staged.data_type(), table.data_type()) {
                // Entries are `(key, value)` by position, whatever they are named.
                (Struct(staged), Struct(table)) => {
                    staged.len() == table.len()
                        && staged
                            .iter()
                            .zip(table.iter())
                            .all(|(staged, table)| widens(staged.data_type(), table.data_type()))
                },
                _ => false,
            }
        },
        _ => match (decimal(staged), decimal(table)) {
            (Some((precision, scale)), Some((table_precision, table_scale))) => {
                table_scale >= scale && table_precision - table_scale >= precision - scale
            },
            _ => staged == table,
        },
    }
}

/// A staged `column` (checked by [`check_staged_type`]) in `field`'s type. A
/// conversion can still fail on a value its type should not hold (a decimal
/// past its precision, a timestamp out of a finer unit's range, a NULL in a
/// non-nullable nested field), with [`DuckLakeError::InvalidConfig`].
fn conform_column(column: &ArrayRef, field: &arrow::datatypes::Field) -> crate::Result<ArrayRef> {
    let target = field.data_type();
    let invalid = |error: &dyn fmt::Display| {
        DuckLakeError::InvalidConfig(format!("column `{}`: {error}", field.name()))
    };
    let column = if column.data_type().equals_datatype(target) {
        Arc::clone(column)
    } else {
        // A `Date64` to `Date32` cast truncates a value that is not a whole
        // day: refused, as the widening is lossless only for whole days.
        if !date64_whole_days(column.as_ref()).map_err(|error| invalid(&error))? {
            return Err(invalid(&"a Date64 value is not a whole day"));
        }
        let options = arrow::compute::CastOptions {
            safe: false,
            ..Default::default()
        };
        arrow::compute::cast_with_options(column, target, &options)
            .map_err(|error| invalid(&error))?
    };
    crate::column_rename::coerce_column(&column, target).map_err(|error| invalid(&error))
}

/// Whether every valid `Date64` value of `column`, at any depth of a list,
/// struct or map, is a whole day. Only those values are read: a value under a
/// NULL parent, or of any other type, is not.
fn date64_whole_days(column: &dyn Array) -> Result<bool, arrow::error::ArrowError> {
    use arrow::array::{AsArray, make_array};
    use arrow::buffer::NullBuffer;
    use arrow::datatypes::Date64Type;
    const DAY_MILLIS: i64 = 86_400_000;
    if !has_date64(column.data_type()) {
        return Ok(true);
    }
    let valid = (0..column.len()).filter(|row| column.is_valid(*row));
    match column.data_type() {
        DataType::Date64 => Ok(column
            .as_primitive::<Date64Type>()
            .iter()
            .flatten()
            .all(|millis| millis % DAY_MILLIS == 0)),
        DataType::List(_) => {
            let list = column.as_list::<i32>();
            for row in valid {
                if !date64_whole_days(list.value(row).as_ref())? {
                    return Ok(false);
                }
            }
            Ok(true)
        },
        DataType::LargeList(_) => {
            let list = column.as_list::<i64>();
            for row in valid {
                if !date64_whole_days(list.value(row).as_ref())? {
                    return Ok(false);
                }
            }
            Ok(true)
        },
        DataType::FixedSizeList(..) => {
            let list = column.as_fixed_size_list();
            for row in valid {
                if !date64_whole_days(list.value(row).as_ref())? {
                    return Ok(false);
                }
            }
            Ok(true)
        },
        DataType::Map(..) => {
            let map = column.as_map();
            for row in valid {
                if !date64_whole_days(&map.value(row))? {
                    return Ok(false);
                }
            }
            Ok(true)
        },
        DataType::Struct(_) => {
            let parent = column.as_struct();
            for child in parent.columns() {
                // The child's values under a NULL row are not the row's.
                let nulls = NullBuffer::union(parent.nulls(), child.nulls());
                let child = make_array(child.to_data().into_builder().nulls(nulls).build()?);
                if !date64_whole_days(child.as_ref())? {
                    return Ok(false);
                }
            }
            Ok(true)
        },
        _ => Ok(true),
    }
}

/// Whether `data_type` holds a `Date64`, at any depth.
fn has_date64(data_type: &DataType) -> bool {
    match data_type {
        DataType::Date64 => true,
        DataType::List(field)
        | DataType::LargeList(field)
        | DataType::FixedSizeList(field, _)
        | DataType::Map(field, _) => has_date64(field.data_type()),
        DataType::Struct(fields) => fields.iter().any(|field| has_date64(field.data_type())),
        _ => false,
    }
}

/// Whether `data_type` holds a float, at any depth.
fn has_float(data_type: &DataType) -> bool {
    match data_type {
        DataType::Float16 | DataType::Float32 | DataType::Float64 => true,
        DataType::List(field)
        | DataType::LargeList(field)
        | DataType::ListView(field)
        | DataType::LargeListView(field)
        | DataType::FixedSizeList(field, _)
        | DataType::Map(field, _) => has_float(field.data_type()),
        DataType::Struct(fields) => fields.iter().any(|field| has_float(field.data_type())),
        DataType::Dictionary(_, value) => has_float(value),
        _ => false,
    }
}

/// `column` with every float zero positive and every NaN one NaN, at any
/// depth of a list, struct or map, so the row format, which orders floats by
/// their bits, compares them as DuckDB does.
fn canonical_floats(column: ArrayRef) -> DataFusionResult<ArrayRef> {
    use arrow::array::{
        AsArray, FixedSizeListArray, LargeListArray, ListArray, MapArray, StructArray,
    };
    use arrow::datatypes::{Float32Type, Float64Type};
    if !has_float(column.data_type()) {
        return Ok(column);
    }
    Ok(match column.data_type() {
        DataType::Float64 => Arc::new(arrow::compute::kernels::arity::unary::<_, _, Float64Type>(
            column.as_primitive::<Float64Type>(),
            |value| {
                if value.is_nan() {
                    f64::NAN
                } else {
                    value + 0.0
                }
            },
        )),
        DataType::Float32 => Arc::new(arrow::compute::kernels::arity::unary::<_, _, Float32Type>(
            column.as_primitive::<Float32Type>(),
            |value| {
                if value.is_nan() {
                    f32::NAN
                } else {
                    value + 0.0
                }
            },
        )),
        DataType::Struct(fields) => {
            let array = column.as_struct();
            let children = array
                .columns()
                .iter()
                .map(|child| canonical_floats(Arc::clone(child)))
                .collect::<DataFusionResult<Vec<_>>>()?;
            Arc::new(StructArray::try_new(
                fields.clone(),
                children,
                array.nulls().cloned(),
            )?)
        },
        DataType::List(field) => {
            let array = column.as_list::<i32>();
            Arc::new(ListArray::try_new(
                Arc::clone(field),
                array.offsets().clone(),
                canonical_floats(Arc::clone(array.values()))?,
                array.nulls().cloned(),
            )?)
        },
        DataType::LargeList(field) => {
            let array = column.as_list::<i64>();
            Arc::new(LargeListArray::try_new(
                Arc::clone(field),
                array.offsets().clone(),
                canonical_floats(Arc::clone(array.values()))?,
                array.nulls().cloned(),
            )?)
        },
        DataType::FixedSizeList(field, size) => {
            let array = column.as_fixed_size_list();
            Arc::new(FixedSizeListArray::try_new(
                Arc::clone(field),
                *size,
                canonical_floats(Arc::clone(array.values()))?,
                array.nulls().cloned(),
            )?)
        },
        DataType::Map(field, ordered) => {
            let array = column.as_map();
            let entries = canonical_floats(Arc::new(array.entries().clone()))?;
            Arc::new(MapArray::try_new(
                Arc::clone(field),
                array.offsets().clone(),
                entries.as_struct().clone(),
                array.nulls().cloned(),
                *ordered,
            )?)
        },
        // A float in a list view or dictionary is compared by its bits.
        _ => column,
    })
}

/// The key columns `names` of a staged batch (from
/// [`MutationContext::conform`]), found by name.
fn staged_key_columns(batch: &RecordBatch, names: &[String]) -> crate::Result<Vec<ArrayRef>> {
    names
        .iter()
        .map(|name| {
            let column = batch.column_by_name(name).ok_or_else(|| {
                DuckLakeError::InvalidConfig(format!("staged keys have no column '{name}'"))
            })?;
            canonical_floats(Arc::clone(column)).map_err(from_df)
        })
        .collect()
}

/// A key column of the table's rows in its table type `data_type`.
fn table_key_column(column: &ArrayRef, data_type: &DataType) -> DataFusionResult<ArrayRef> {
    canonical_floats(crate::column_rename::coerce_column(column, data_type)?)
}

/// The key columns `names` of a batch of the table's rows, found by name.
fn table_key_columns(
    batch: &RecordBatch,
    names: &[String],
    types: &[DataType],
) -> crate::Result<Vec<ArrayRef>> {
    names
        .iter()
        .zip(types)
        .map(|(name, data_type)| {
            let column = batch.column_by_name(name).ok_or_else(|| {
                DuckLakeError::Internal(format!("table rows have no key column '{name}'"))
            })?;
            table_key_column(column, data_type).map_err(from_df)
        })
        .collect()
}

fn all_valid(columns: &[ArrayRef], rows: usize) -> BooleanArray {
    let mut out = BooleanBuilder::with_capacity(rows);
    for row in 0..rows {
        out.append_value(columns.iter().all(|column| column.is_valid(row)));
    }
    out.finish()
}

/// The staged keys, each with the staged row that carries it, shared by
/// every copy of a [`KeyMembership`].
struct SharedKeys {
    converter: RowConverter,
    /// The key columns' types, in key order.
    types: Vec<DataType>,
    rows: HashMap<Box<[u8]>, usize>,
}

impl SharedKeys {
    /// Which rows of the key `columns` (in `types`) carry a staged key; a
    /// NULL in any key column matches nothing.
    fn contains(&self, columns: &[ArrayRef], rows: usize) -> DataFusionResult<BooleanArray> {
        let encoded = self.converter.convert_columns(columns)?;
        let valid = all_valid(columns, rows);
        let mut out = BooleanBuilder::with_capacity(rows);
        for row in 0..rows {
            out.append_value(valid.value(row) && self.rows.contains_key(encoded.row(row).as_ref()));
        }
        Ok(out.finish())
    }
}

/// [`SharedKeys`] compared and hashed by identity, so a [`KeyMembership`]
/// equals only its own copies.
struct KeySetRef(SharedKeys);

/// `(k...) IN staged keys`, over the key columns: true for a row whose key is
/// one of the staged ones, false otherwise (a NULL key included).
#[derive(Clone)]
struct KeyMembership {
    columns: Vec<Arc<dyn PhysicalExpr>>,
    keys: Arc<KeySetRef>,
}

impl fmt::Debug for KeyMembership {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KeyMembership")
            .field("columns", &self.columns)
            .field("keys", &self.keys.0.rows.len())
            .finish()
    }
}

impl fmt::Display for KeyMembership {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(")?;
        for (index, column) in self.columns.iter().enumerate() {
            if index > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{column}")?;
        }
        write!(f, ") IN <{} staged keys>", self.keys.0.rows.len())
    }
}

impl PartialEq for KeyMembership {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.keys, &other.keys) && self.columns == other.columns
    }
}

impl Eq for KeyMembership {}

impl Hash for KeyMembership {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (Arc::as_ptr(&self.keys) as usize).hash(state);
        self.columns.hash(state);
    }
}

impl PhysicalExpr for KeyMembership {
    fn data_type(&self, _input_schema: &Schema) -> DataFusionResult<DataType> {
        Ok(DataType::Boolean)
    }

    fn nullable(&self, _input_schema: &Schema) -> DataFusionResult<bool> {
        Ok(false)
    }

    fn evaluate(&self, batch: &RecordBatch) -> DataFusionResult<ColumnarValue> {
        let rows = batch.num_rows();
        let keys = &self.keys.0;
        let columns = self
            .columns
            .iter()
            .zip(&keys.types)
            .map(|(column, data_type)| {
                let values = column.evaluate(batch)?.into_array(rows)?;
                table_key_column(&values, data_type)
            })
            .collect::<DataFusionResult<Vec<_>>>()?;
        Ok(ColumnarValue::Array(Arc::new(
            keys.contains(&columns, rows)?,
        )))
    }

    fn children(&self) -> Vec<&Arc<dyn PhysicalExpr>> {
        self.columns.iter().collect()
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn PhysicalExpr>>,
    ) -> DataFusionResult<Arc<dyn PhysicalExpr>> {
        if children.len() != self.columns.len() {
            return Err(DataFusionError::Internal(format!(
                "key membership over {} columns given {} children",
                self.columns.len(),
                children.len()
            )));
        }
        Ok(Arc::new(Self {
            columns: children,
            keys: Arc::clone(&self.keys),
        }))
    }

    fn fmt_sql(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

fn from_df(error: DataFusionError) -> DuckLakeError {
    match error {
        DataFusionError::External(inner) => match inner.downcast::<DuckLakeError>() {
            Ok(error) => *error,
            Err(inner) => DuckLakeError::DataFusion(DataFusionError::External(inner)),
        },
        other => DuckLakeError::DataFusion(other),
    }
}

fn to_df(error: DuckLakeError) -> DataFusionError {
    DataFusionError::External(Box::new(error))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Int64Array as I64, StringArray};
    use arrow::datatypes::Field;

    /// The whole-day check reads only `Date64` values: a struct whose float
    /// sibling widens holds a signaling NaN, whose payload a round trip
    /// through `Float64` changes, and still conforms; a `Date64` that is not
    /// a whole day is refused, and one under a NULL struct row is not read.
    /// Guards against comparing the whole round-tripped column.
    #[test]
    fn date64_check_reads_only_dates() {
        use arrow::array::{Date64Array, Float32Array, StructArray};
        use arrow::buffer::NullBuffer;
        const DAY: i64 = 86_400_000;
        let staged_fields =
            vec![Field::new("d", DataType::Date64, true), Field::new("f", DataType::Float32, true)];
        let table_fields =
            vec![Field::new("d", DataType::Date32, true), Field::new("f", DataType::Float64, true)];
        let column = |days: Vec<i64>, nulls: Option<NullBuffer>| -> ArrayRef {
            let rows = days.len();
            Arc::new(StructArray::new(
                staged_fields.clone().into(),
                vec![
                    Arc::new(Date64Array::from(days)),
                    Arc::new(Float32Array::from(vec![f32::from_bits(0x7f80_0001); rows])),
                ],
                nulls,
            ))
        };
        let target = Field::new("s", DataType::Struct(table_fields.into()), true);

        let conformed = conform_column(&column(vec![DAY, -3 * DAY], None), &target).unwrap();
        assert_eq!(conformed.len(), 2);
        let refused = conform_column(&column(vec![DAY, DAY + 1], None), &target).unwrap_err();
        assert!(refused.to_string().contains("not a whole day"), "{refused}");
        let hidden = column(
            vec![DAY, DAY + 1],
            Some(NullBuffer::from(vec![true, false])),
        );
        conform_column(&hidden, &target).unwrap();
    }

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, true),
            Field::new("name", DataType::Utf8, true),
        ]))
    }

    fn batch(ids: Vec<Option<i64>>, names: Vec<Option<&str>>) -> RecordBatch {
        RecordBatch::try_new(
            schema(),
            vec![Arc::new(I64::from(ids)), Arc::new(StringArray::from(names))],
        )
        .unwrap()
    }

    #[test]
    fn membership_ignores_null_keys_and_matches_values() {
        let staged = batch(
            vec![Some(1), None, Some(3)],
            vec![Some("a"), Some("b"), None],
        );
        let keys = KeySet::build(&schema(), &[0], &[staged]).unwrap();
        assert_eq!(keys.keys.0.rows.len(), 2);
        let probe = batch(vec![Some(3), None, Some(2), Some(1)], vec![None; 4]);
        let mask = keys.contains(&probe).unwrap();
        assert_eq!(
            mask.iter().map(Option::unwrap).collect::<Vec<_>>(),
            vec![true, false, false, true]
        );
        let predicate = keys.predicate(&schema(), &[0]).unwrap().combined();
        let mask = predicate.evaluate(&probe).unwrap().into_array(4).unwrap();
        let mask = mask.as_any().downcast_ref::<BooleanArray>().unwrap();
        assert_eq!(
            mask.iter()
                .map(|value| value == Some(true))
                .collect::<Vec<_>>(),
            vec![true, false, false, true]
        );
    }

    #[test]
    fn composite_keys_match_on_every_column() {
        let staged = batch(vec![Some(1), Some(1)], vec![Some("a"), Some("b")]);
        let keys = KeySet::build(&schema(), &[0, 1], &[staged]).unwrap();
        let probe = batch(
            vec![Some(1), Some(1), Some(2)],
            vec![Some("b"), Some("c"), Some("a")],
        );
        let mask = keys.contains(&probe).unwrap();
        assert_eq!(
            mask.iter().map(Option::unwrap).collect::<Vec<_>>(),
            vec![true, false, false]
        );
        assert_eq!(keys.staged_rows(&probe.slice(0, 1)).unwrap()[0], 1);
    }

    #[test]
    fn a_key_staged_twice_keeps_its_last_row() {
        let staged = batch(vec![Some(1), Some(1)], vec![Some("a"), Some("b")]);
        let keys = KeySet::build(&schema(), &[0], &[staged]).unwrap();
        let probe = batch(vec![Some(1)], vec![None]);
        assert_eq!(keys.staged_rows(&probe).unwrap()[0], 1);
    }

    #[test]
    fn staged_types_widen_without_loss_only() {
        use arrow::datatypes::{Fields, TimeUnit};
        let utc = Some(Arc::from("UTC"));
        let accepted = [
            (DataType::Int8, DataType::Int64),
            (DataType::Int32, DataType::Int64),
            (DataType::UInt8, DataType::Int16),
            (DataType::UInt32, DataType::UInt64),
            (DataType::UInt32, DataType::Int64),
            (DataType::Float32, DataType::Float64),
            (DataType::Decimal128(5, 2), DataType::Decimal128(10, 3)),
            (DataType::Decimal256(10, 2), DataType::Decimal128(12, 2)),
            (DataType::Decimal32(4, 1), DataType::Decimal128(4, 1)),
            (DataType::Utf8View, DataType::Utf8),
            (DataType::LargeUtf8, DataType::Utf8View),
            (DataType::BinaryView, DataType::LargeBinary),
            (DataType::FixedSizeBinary(16), DataType::Binary),
            (DataType::Date64, DataType::Date32),
            (
                DataType::Timestamp(TimeUnit::Millisecond, utc.clone()),
                DataType::Timestamp(TimeUnit::Microsecond, utc.clone()),
            ),
            (
                DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
                DataType::LargeList(Arc::new(Field::new("element", DataType::Int64, false))),
            ),
            (
                DataType::Struct(Fields::from(vec![Field::new("a", DataType::Float32, true)])),
                DataType::Struct(Fields::from(vec![Field::new(
                    "a",
                    DataType::Float64,
                    false,
                )])),
            ),
        ];
        for (staged, table) in accepted {
            assert!(widens(&staged, &table), "{staged} -> {table}");
        }
        let refused = [
            (DataType::Utf8, DataType::Int64),
            (DataType::Int64, DataType::Int32),
            (DataType::Int8, DataType::UInt64),
            (DataType::UInt64, DataType::Int64),
            (DataType::Float64, DataType::Int64),
            (DataType::Int32, DataType::Float64),
            (DataType::Float64, DataType::Float32),
            (DataType::Decimal128(5, 2), DataType::Decimal128(5, 3)),
            (DataType::Decimal128(5, 2), DataType::Decimal128(4, 1)),
            (DataType::Decimal128(3, 1), DataType::Int32),
            (DataType::Utf8, DataType::Binary),
            (DataType::Binary, DataType::FixedSizeBinary(16)),
            (
                DataType::Date32,
                DataType::Timestamp(TimeUnit::Microsecond, None),
            ),
            (
                DataType::Timestamp(TimeUnit::Microsecond, None),
                DataType::Timestamp(TimeUnit::Millisecond, None),
            ),
            (
                DataType::Timestamp(TimeUnit::Microsecond, None),
                DataType::Timestamp(TimeUnit::Microsecond, utc),
            ),
            (
                DataType::Struct(Fields::from(vec![Field::new("b", DataType::Int64, true)])),
                DataType::Struct(Fields::from(vec![Field::new("a", DataType::Int64, true)])),
            ),
            (
                DataType::List(Arc::new(Field::new("item", DataType::Int64, true))),
                DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
            ),
            (DataType::Null, DataType::Int64),
        ];
        for (staged, table) in refused {
            assert!(!widens(&staged, &table), "{staged} -> {table}");
        }
        // A map's sorted flag is part of its type: Arrow casts no map to one
        // of the other flag, so a staged map of the other flag gets the
        // typed error before any value is converted.
        let entries = Arc::new(Field::new(
            "entries",
            DataType::Struct(Fields::from(vec![
                Field::new("key", DataType::Utf8, false),
                Field::new("value", DataType::Int32, true),
            ])),
            false,
        ));
        let sorted = DataType::Map(Arc::clone(&entries), true);
        let unsorted = DataType::Map(entries, false);
        assert!(widens(&unsorted, &unsorted));
        assert!(
            check_staged_type("m", &sorted, &unsorted)
                .unwrap_err()
                .to_string()
                .contains("column `m`: expected Map(")
        );
        assert_eq!(
            check_staged_type("id", &DataType::Utf8, &DataType::Int64)
                .unwrap_err()
                .to_string(),
            DuckLakeError::InvalidConfig("column `id`: expected Int64, got Utf8".to_string())
                .to_string()
        );
    }

    /// `-0.0` reads as `0.0` and every NaN as one NaN inside a list and a
    /// map, as at the top level and in a struct.
    #[test]
    fn canonical_floats_reach_list_and_map_values() {
        use arrow::array::{
            AsArray, Float64Array, Float64Builder, ListArray, MapBuilder, StringBuilder,
        };
        use arrow::datatypes::Float64Type;
        let canonical = |values: &Float64Array| -> Vec<u64> {
            values
                .values()
                .iter()
                .map(|value| value.to_bits())
                .collect()
        };
        let expected = vec![0.0f64.to_bits(), f64::NAN.to_bits(), 1.5f64.to_bits()];
        let negative_nan = f64::from_bits(f64::NAN.to_bits() | (1 << 63) | 1);

        let list = ListArray::from_iter_primitive::<Float64Type, _, _>(vec![
            Some(vec![Some(-0.0), Some(negative_nan)]),
            Some(vec![Some(1.5)]),
        ]);
        let canon = canonical_floats(Arc::new(list)).unwrap();
        assert_eq!(
            canonical(
                canon
                    .as_list::<i32>()
                    .values()
                    .as_primitive::<Float64Type>()
            ),
            expected
        );

        let mut map = MapBuilder::new(None, StringBuilder::new(), Float64Builder::new());
        map.keys().append_value("a");
        map.values().append_value(-0.0);
        map.keys().append_value("b");
        map.values().append_value(negative_nan);
        map.append(true).unwrap();
        map.keys().append_value("c");
        map.values().append_value(1.5);
        map.append(true).unwrap();
        let canon = canonical_floats(Arc::new(map.finish())).unwrap();
        assert_eq!(
            canonical(canon.as_map().values().as_primitive::<Float64Type>()),
            expected
        );
    }

    #[test]
    fn nested_float_keys_compare_numerically() {
        use arrow::array::{Float64Array, StructArray};
        use arrow::datatypes::Fields;
        let fields = Fields::from(vec![Field::new("f", DataType::Float64, true)]);
        let schema = Arc::new(Schema::new(vec![Field::new(
            "s",
            DataType::Struct(fields.clone()),
            true,
        )]));
        let rows = |values: Vec<f64>| {
            RecordBatch::try_new(
                Arc::clone(&schema),
                vec![Arc::new(StructArray::new(
                    fields.clone(),
                    vec![Arc::new(Float64Array::from(values))],
                    None,
                ))],
            )
            .unwrap()
        };
        let keys = KeySet::build(&schema, &[0], &[rows(vec![-0.0, f64::NAN])]).unwrap();
        assert!(keys.single.is_none(), "a nested float key is no IN list");
        let probe = rows(vec![0.0, -f64::NAN, 1.0]);
        assert_eq!(
            keys.contains(&probe)
                .unwrap()
                .iter()
                .map(Option::unwrap)
                .collect::<Vec<_>>(),
            vec![true, true, false]
        );
    }

    #[test]
    fn a_long_in_list_prunes_by_its_bounds() {
        let few = batch(vec![Some(1), Some(5)], vec![None; 2]);
        let keys = KeySet::build(&schema(), &[0], &[few]).unwrap();
        assert!(keys.predicate(&schema(), &[0]).unwrap().in_list);
        let many = batch((0..25).map(Some).collect(), vec![None; 25]);
        let keys = KeySet::build(&schema(), &[0], &[many]).unwrap();
        let predicate = keys.predicate(&schema(), &[0]).unwrap();
        assert!(!predicate.in_list && predicate.range.is_some());
        assert_eq!(
            predicate.range_or_true().to_string(),
            predicate.range.as_ref().unwrap().to_string()
        );
    }

    #[test]
    fn bounds_cover_the_staged_keys() {
        let staged = batch(vec![Some(5), Some(-2), None], vec![None; 3]);
        let keys = KeySet::build(&schema(), &[0], &[staged]).unwrap();
        assert_eq!(
            keys.bounds[0],
            Some((ScalarValue::Int64(Some(-2)), ScalarValue::Int64(Some(5))))
        );
    }
}
