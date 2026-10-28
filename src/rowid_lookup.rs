//! Find rows by `rowid`: where each lives, and the rows themselves.
//!
//! A row's `rowid` (DuckLake row lineage) is stable across updates, flushes and
//! compaction, but where the row lives is not. It may be
//!
//! - **inlined** in the catalog (`ducklake_inlined_data_<table_id>_<version>`),
//!   where its id is a column of the inlined table;
//! - in a Parquet file that **embeds** a `_ducklake_internal_row_id` column (a
//!   flush, an `UPDATE`'s new row versions, a merge or rewrite), where its id is
//!   that column's value. Such a file may still record a `row_id_start` (a flush
//!   records its smallest row id) and its ids need not be contiguous, so
//!   `row_id_start + position` names the wrong row;
//! - in any other Parquet file, at position `rowid - row_id_start`.
//!
//! [`DuckLakeTable::locate_rowids`] answers where each live row is,
//! [`DuckLakeTable::read_columns_by_rowid`] reads the columns a caller asks
//! for of each, with its `rowid`, and `DuckLakeTable::read_rows_by_rowid`
//! (with the `write` feature) reads them whole, all at the table's snapshot
//! and without the caller knowing which case applies.
//! [`DuckLakeTable::file_row_id_layout`] tells, for one file, which of the two
//! Parquet cases it is.

use std::collections::{HashMap, HashSet};
use std::ops::Range;
use std::sync::Arc;

use arrow::array::ArrayRef;
use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use datafusion::catalog::{Session, TableProvider};
use datafusion::datasource::listing::PartitionedFile;
use datafusion::datasource::physical_plan::ParquetFileReaderFactory;
use datafusion::parquet::arrow::arrow_reader::{ArrowReaderOptions, RowSelection, RowSelector};
use datafusion::parquet::arrow::async_reader::AsyncFileReader;
use datafusion::parquet::file::metadata::{PageIndexPolicy, ParquetMetaData};
use datafusion::physical_plan::metrics::ExecutionPlanMetricsSet;
use futures::future::BoxFuture;

use futures::{StreamExt, TryStreamExt};

use crate::DuckLakeError;
use crate::metadata_provider::{DuckLakeInlinedData, DuckLakeTableFile};
use crate::table::DuckLakeTable;

/// A selection of `positions` out of a file of `rows` rows.
pub(crate) fn positions_selection(positions: &HashSet<i64>, rows: i64) -> RowSelection {
    let mut sorted: Vec<i64> = positions
        .iter()
        .copied()
        .filter(|p| (0..rows).contains(p))
        .collect();
    sorted.sort_unstable();
    let mut selectors = Vec::with_capacity(sorted.len() * 2 + 1);
    let mut at = 0i64;
    for position in sorted {
        if position > at {
            selectors.push(RowSelector::skip((position - at) as usize));
        }
        selectors.push(RowSelector::select(1));
        at = position + 1;
    }
    if rows > at {
        selectors.push(RowSelector::skip((rows - at) as usize));
    }
    selectors.into()
}

/// Reads a file's metadata with its page index, whatever the scan asked for.
///
/// DataFusion loads the page index only for a page-pruning predicate. A scan
/// that selects rows by position has none, and without the offset index the
/// reader cannot tell which pages hold the selected rows, so it fetches each
/// selected row group's whole column chunks.
///
/// DataFusion loads the page index for an external row selection itself from
/// apache/datafusion#25460 on; once the pinned DataFusion includes it, this
/// wrapper has nothing left to do and should be removed.
#[derive(Debug)]
pub(crate) struct PageIndexReaderFactory(pub(crate) Arc<dyn ParquetFileReaderFactory>);

impl ParquetFileReaderFactory for PageIndexReaderFactory {
    fn create_reader(
        &self,
        partition_index: usize,
        partitioned_file: PartitionedFile,
        metadata_size_hint: Option<usize>,
        metrics: &ExecutionPlanMetricsSet,
    ) -> datafusion::common::Result<Box<dyn AsyncFileReader + Send>> {
        let inner = self.0.create_reader(
            partition_index,
            partitioned_file,
            metadata_size_hint,
            metrics,
        )?;
        Ok(Box::new(PageIndexReader {
            inner,
            options: None,
        }))
    }
}

struct PageIndexReader {
    inner: Box<dyn AsyncFileReader + Send>,
    options: Option<ArrowReaderOptions>,
}

impl AsyncFileReader for PageIndexReader {
    fn get_bytes(
        &mut self,
        range: Range<u64>,
    ) -> BoxFuture<'_, datafusion::parquet::errors::Result<bytes::Bytes>> {
        self.inner.get_bytes(range)
    }

    fn get_byte_ranges(
        &mut self,
        ranges: Vec<Range<u64>>,
    ) -> BoxFuture<'_, datafusion::parquet::errors::Result<Vec<bytes::Bytes>>> {
        self.inner.get_byte_ranges(ranges)
    }

    fn get_metadata<'a>(
        &'a mut self,
        options: Option<&'a ArrowReaderOptions>,
    ) -> BoxFuture<'a, datafusion::parquet::errors::Result<Arc<ParquetMetaData>>> {
        self.options = Some(
            options
                .cloned()
                .unwrap_or_default()
                .with_page_index_policy(PageIndexPolicy::Optional),
        );
        self.inner.get_metadata(self.options.as_ref())
    }
}

/// What [`DuckLakeTable::locate_rowids`] read: where each row is, and the
/// inlined rows, the files and their inlined deletions it found them in.
struct Located {
    rows: Vec<RowIdLocation>,
    inlined: Vec<DuckLakeInlinedData>,
    files: Vec<DuckLakeTableFile>,
    #[cfg_attr(not(feature = "write"), allow(dead_code))]
    inlined_deletes: std::collections::HashMap<i64, HashSet<i64>>,
}

/// Where a Parquet file's rows take their `rowid` from
/// ([`DuckLakeTable::file_row_id_layout`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RowIdLayout {
    /// The row at position `p` has `rowid` `row_id_start + p`.
    Positional {
        /// The file's catalog `row_id_start`.
        row_id_start: i64,
    },
    /// Each row carries its `rowid` in an embedded `_ducklake_internal_row_id`
    /// column; positions say nothing about it.
    Embedded,
}

/// Where a live row is stored.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum RowLocation {
    /// In the catalog's inlined data table `table_name`.
    Inlined {
        /// The physical inlined data table holding the row.
        table_name: String,
    },
    /// In a Parquet data file, at its physical `position` (the position a
    /// delete file records).
    File {
        /// The data file's catalog id.
        data_file_id: i64,
        /// The data file's catalog path.
        path: String,
        /// The row's physical position in the file.
        position: i64,
    },
}

/// One located row: its `rowid` and where it is.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct RowIdLocation {
    /// The row's `rowid`.
    pub row_id: i64,
    /// Where the row is.
    pub location: RowLocation,
}

impl DuckLakeTableFile {
    /// Whether the catalog says this file may embed its rows' `rowid`s (it
    /// records no `row_id_start`, or it is a partial file a flush or merge
    /// wrote), in which case `row_id_start + position` is not a row's `rowid`.
    /// `false` is not a proof that the file embeds none: only its footer says
    /// ([`DuckLakeTable::file_row_id_layout`]).
    #[must_use]
    pub fn may_embed_row_ids(&self) -> bool {
        self.row_id_start.is_none() || self.partial_max.is_some()
    }
}

impl DuckLakeTable {
    /// Where each live row with one of `row_ids` is at this table's snapshot,
    /// in no particular order. A `rowid` no live row has (deleted, or never
    /// assigned) is left out; one requested twice is reported once.
    ///
    /// Inlined rows come from the catalog. A Parquet file whose rows take
    /// their `rowid` from position ([`RowIdLayout::Positional`]) is located by
    /// arithmetic and read only for its deletions; a file that embeds them
    /// ([`RowIdLayout::Embedded`]) has its row id column read, once per call.
    /// Each file's footer is read once per table to learn which it is.
    pub async fn locate_rowids(
        &self,
        state: &dyn Session,
        row_ids: &[i64],
    ) -> crate::Result<Vec<RowIdLocation>> {
        Ok(self.locate(state, row_ids).await?.rows)
    }

    /// [`Self::locate_rowids`], with what it read the rows from. The inlined
    /// deletions, then the inlined rows, then the files are read, in the
    /// order a scan reads them, so that a flush between the reads moves no
    /// row out of all of them or into two
    /// ([`DuckLakeTable::inlined_rows_then_files`]).
    async fn locate(&self, state: &dyn Session, row_ids: &[i64]) -> crate::Result<Located> {
        let wanted: HashSet<i64> = row_ids.iter().copied().collect();
        if wanted.is_empty() {
            return Ok(Located {
                rows: Vec::new(),
                inlined: Vec::new(),
                files: Vec::new(),
                inlined_deletes: std::collections::HashMap::new(),
            });
        }
        let (inlined_deletes, (inlined, files)) = self.read_consistently(|table| {
            Ok::<_, DuckLakeError>((
                table.inlined_deletes_by_file().map_err(df_error)?,
                table.inlined_rows_then_files(|| table.files(), |file| file.data_file_id)?,
            ))
        })?;
        let mut located = Vec::new();
        for rows in &inlined {
            for row_id in &rows.row_ids {
                if wanted.contains(row_id) {
                    located.push(RowIdLocation {
                        row_id: *row_id,
                        location: RowLocation::Inlined {
                            table_name: rows.table_name.clone(),
                        },
                    });
                }
            }
        }
        let snapshot_id = self.snapshot_id();
        let wanted = &wanted;
        let inlined_deletes_ref = &inlined_deletes;
        let concurrency = state.config().target_partitions().max(1);
        let pending: Vec<BoxFuture<'_, crate::Result<Vec<RowIdLocation>>>> = files
            .iter()
            .map(|file| -> BoxFuture<'_, crate::Result<Vec<RowIdLocation>>> {
                let inlined = inlined_deletes_ref.get(&file.data_file_id);
                Box::pin(async move {
                    self.locate_in_file(state, file, wanted, inlined, snapshot_id)
                        .await
                })
            })
            .collect();
        let in_files = bounded(pending, concurrency).await?;
        located.extend(in_files.into_iter().flatten());
        Ok(Located {
            rows: located,
            inlined,
            files,
            inlined_deletes,
        })
    }

    async fn locate_in_file(
        &self,
        state: &dyn Session,
        file: &DuckLakeTableFile,
        wanted: &HashSet<i64>,
        inlined: Option<&HashSet<i64>>,
        snapshot_id: i64,
    ) -> crate::Result<Vec<RowIdLocation>> {
        // A file's footer says which layout it has: an embedded row id column
        // may hold ids below the file's `row_id_start` (an `UPDATE`'s new row
        // versions keep their rows' ids under a new `row_id_start`), so only a
        // positional file is skipped by its range.
        let layout = self
            .file_row_id_layout(state, file)
            .await
            .map_err(df_error)?;
        let candidates: Vec<(i64, i64)> = match layout {
            Some(RowIdLayout::Positional {
                row_id_start,
            }) => {
                // A catalog without `record_count` bounds the positions by
                // the file's footer, so no id past its rows is located.
                let rows = match file.max_row_count {
                    Some(rows) => rows,
                    None => self
                        .file_row_count(state, &file.file)
                        .await
                        .map_err(df_error)?,
                };
                let candidates: Vec<(i64, i64)> = wanted
                    .iter()
                    .filter_map(|row_id| {
                        let position = row_id.checked_sub(row_id_start)?;
                        (0..rows).contains(&position).then_some((position, *row_id))
                    })
                    .collect();
                // A merge of adjacent files keeps positional row ids but holds
                // rows of several snapshots: those inserted after this one are
                // not rows of the table here.
                if candidates.is_empty() {
                    candidates
                } else {
                    let later = self
                        .positions_inserted_later(state, file)
                        .await
                        .map_err(df_error)?;
                    candidates
                        .into_iter()
                        .filter(|(position, _)| !later.contains(position))
                        .collect()
                }
            },
            Some(RowIdLayout::Embedded) => {
                let partial = file.partial_max.is_some_and(|max| max > snapshot_id);
                self.embedded_row_ids(state, file, partial)
                    .await
                    .map_err(df_error)?
                    .unwrap_or_default()
                    .into_iter()
                    // Read at a snapshot before a partial file's last, a row
                    // shows when its snapshot is at most this one; a NULL one
                    // fails official's `<=` filter too.
                    .filter(|(_, row_id, snapshot)| {
                        wanted.contains(row_id)
                            && (!partial
                                || snapshot.is_some_and(|snapshot| snapshot <= snapshot_id))
                    })
                    .map(|(position, row_id, _)| (position, row_id))
                    .collect()
            },
            None => {
                return Err(DuckLakeError::InvalidConfig(format!(
                    "file \"{}\" has no embedded `_ducklake_internal_row_id` column and no \
                     `row_id_start`: its rows have no rowid",
                    file.file.path
                )));
            },
        };
        if candidates.is_empty() {
            return Ok(Vec::new());
        }
        let deleted = self
            .deleted_positions_at_snapshot(state, file, inlined)
            .await
            .map_err(df_error)?;
        Ok(candidates
            .into_iter()
            .filter(|(position, _)| !deleted.contains(position))
            .map(|(position, row_id)| RowIdLocation {
                row_id,
                location: RowLocation::File {
                    data_file_id: file.data_file_id,
                    path: file.file.path.clone(),
                    position,
                },
            })
            .collect())
    }

    /// The live rows with one of `row_ids` at this table's snapshot, as
    /// batches of the table's columns followed by a `rowid` column, in no
    /// particular order. Missing ids are left out, as
    /// [`Self::locate_rowids`] leaves them out.
    ///
    /// Only the files holding a requested row are read, each once.
    #[cfg(feature = "write")]
    pub async fn read_rows_by_rowid(
        &self,
        state: &dyn Session,
        row_ids: &[i64],
    ) -> crate::Result<Vec<arrow::record_batch::RecordBatch>> {
        let located = self.locate(state, row_ids).await?;
        let mut inlined_ids = HashSet::new();
        let mut by_file: std::collections::HashMap<i64, HashSet<i64>> =
            std::collections::HashMap::new();
        for row in &located.rows {
            match &row.location {
                RowLocation::Inlined {
                    ..
                } => {
                    inlined_ids.insert(row.row_id);
                },
                RowLocation::File {
                    data_file_id,
                    position,
                    ..
                } => {
                    by_file.entry(*data_file_id).or_default().insert(*position);
                },
            }
        }
        let mut batches = Vec::new();
        if !inlined_ids.is_empty() {
            for rows in &located.inlined {
                if !rows
                    .row_ids
                    .iter()
                    .any(|row_id| inlined_ids.contains(row_id))
                {
                    continue;
                }
                let predicate: std::sync::Arc<dyn datafusion::physical_expr::PhysicalExpr> =
                    std::sync::Arc::new(datafusion::physical_expr::expressions::Literal::new(
                        datafusion::common::ScalarValue::Boolean(Some(true)),
                    ));
                if let Some((batch, _)) =
                    self.apply_update_to_inlined(rows, Some(&predicate), &[])?
                {
                    let ids = batch
                        .column(batch.num_columns() - 1)
                        .as_any()
                        .downcast_ref::<arrow::array::Int64Array>()
                        .expect("rowid column is Int64")
                        .clone();
                    let keep = arrow::array::BooleanArray::from(
                        ids.values()
                            .iter()
                            .map(|row_id| inlined_ids.contains(row_id))
                            .collect::<Vec<_>>(),
                    );
                    batches.push(arrow::compute::filter_record_batch(&batch, &keep)?);
                }
            }
        }
        if by_file.is_empty() {
            return Ok(batches);
        }
        let files: Vec<DuckLakeTableFile> = located
            .files
            .into_iter()
            .filter(|file| by_file.contains_key(&file.data_file_id))
            .collect();
        let by_file = &by_file;
        let inlined_deletes = &located.inlined_deletes;
        let concurrency = state.config().target_partitions().max(1);
        type Read = datafusion::error::Result<Vec<arrow::record_batch::RecordBatch>>;
        let pending: Vec<BoxFuture<'_, Read>> = files
            .into_iter()
            .map(|file| -> BoxFuture<'_, Read> {
                Box::pin(async move {
                    self.read_file_rows_at(
                        state,
                        &file,
                        &by_file[&file.data_file_id],
                        inlined_deletes.get(&file.data_file_id),
                    )
                    .await
                })
            })
            .collect();
        let read = bounded(pending, concurrency).await.map_err(df_error)?;
        batches.extend(read.into_iter().flatten());
        Ok(batches)
    }
}

/// The rows [`DuckLakeTable::read_columns_by_rowid`] found.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct RowsByRowId {
    /// The `rowid` of each row of [`Self::batch`], row for row.
    pub row_ids: Vec<i64>,
    /// The requested columns of the rows found, under the table's names and
    /// types, in the order their `rowid`s were first requested.
    pub batch: RecordBatch,
}

impl RowsByRowId {
    /// The ids of `requested` that no row was found for, each once, in the
    /// order first requested: ids of rows deleted at the table's snapshot,
    /// inserted after it, or never assigned.
    #[must_use]
    pub fn missing(&self, requested: &[i64]) -> Vec<i64> {
        let found: HashSet<i64> = self.row_ids.iter().copied().collect();
        let mut seen = HashSet::new();
        requested
            .iter()
            .copied()
            .filter(|row_id| !found.contains(row_id) && seen.insert(*row_id))
            .collect()
    }
}

/// Rows read from one place, inlined rows or a data file: the values of the
/// read columns and each row's `rowid`.
struct ReadPart {
    columns: Vec<ArrayRef>,
    row_ids: Vec<i64>,
}

impl DuckLakeTable {
    /// The live rows with one of `row_ids` at this table's snapshot, with
    /// the columns `projection` names, as a scan with that projection reads
    /// them, and each row's `rowid`.
    ///
    /// `projection` holds indices into the table's schema (its
    /// `TableProvider::schema`, which ends with `rowid` when the table was
    /// opened with row lineage), as a scan takes them; `None` reads every
    /// column. Columns read under the table's current names and types: a
    /// renamed column reads its values from before the rename, a column a
    /// row predates reads its initial default, and a column stored narrower
    /// is widened.
    ///
    /// A row is found wherever it is: inlined in the catalog, or in a data
    /// file, with its `rowid` taken as the table's scans take it. A row the
    /// table does not hold at its snapshot (deleted by a delete file or an
    /// inlined deletion, inserted after the snapshot, including the later rows
    /// of a merged file, or never assigned) is not found. The result has one
    /// row per found id, in the order the ids were first requested, whatever
    /// order the reads finish in; [`RowsByRowId::missing`] lists the ids not
    /// found.
    ///
    /// Only the data files that hold a requested row are read, at most the
    /// session's `target_partitions` at once, and of each only the pages of
    /// the projected columns that hold the requested rows. A projection of
    /// only `rowid`, or of nothing, reads no data page.
    ///
    /// A table kept across calls keeps what it learnt of each file's footer,
    /// so a later call reads no footer again; its catalog reads (the file
    /// listing, the inlined rows and inlined deletions) are made on every
    /// call, unless the table's provider memoizes them
    /// ([`CatalogReadMemo`](crate::CatalogReadMemo)).
    pub async fn read_columns_by_rowid(
        &self,
        state: &dyn Session,
        row_ids: &[i64],
        projection: Option<&[usize]>,
    ) -> crate::Result<RowsByRowId> {
        let schema = TableProvider::schema(self);
        let physical_len = self.physical_schema().fields().len();
        let projection: Vec<usize> = match projection {
            Some(projection) => projection.to_vec(),
            None => (0..schema.fields().len()).collect(),
        };
        if let Some(index) = projection
            .iter()
            .find(|&&index| index >= schema.fields().len())
        {
            return Err(DuckLakeError::InvalidConfig(format!(
                "projection index {index} is out of range for table \"{}\" with {} columns",
                self.table_name(),
                schema.fields().len()
            )));
        }
        let output_schema = Arc::new(schema.project(&projection)?);
        // The physical columns to read, each once, in catalog order. An index
        // at or past them is the `rowid` column, which needs no read.
        let read: Vec<usize> = projection
            .iter()
            .copied()
            .filter(|&index| index < physical_len)
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();

        let located = self.locate(state, row_ids).await?;
        let mut parts: Vec<ReadPart> = Vec::new();
        let mut inlined_ids = HashSet::new();
        let mut by_file: HashMap<i64, HashMap<i64, i64>> = HashMap::new();
        for row in &located.rows {
            match &row.location {
                RowLocation::Inlined {
                    ..
                } => {
                    inlined_ids.insert(row.row_id);
                },
                RowLocation::File {
                    data_file_id,
                    position,
                    ..
                } => {
                    by_file
                        .entry(*data_file_id)
                        .or_default()
                        .insert(*position, row.row_id);
                },
            }
        }

        for rows in &located.inlined {
            let keep = arrow::array::BooleanArray::from(
                rows.row_ids
                    .iter()
                    .map(|row_id| inlined_ids.contains(row_id))
                    .collect::<Vec<_>>(),
            );
            if keep.true_count() == 0 {
                continue;
            }
            if rows.batch.num_columns() != physical_len {
                return Err(DuckLakeError::Internal(format!(
                    "inlined rows of \"{}\" have {} columns, the table has {physical_len}",
                    rows.table_name,
                    rows.batch.num_columns(),
                )));
            }
            let mut columns = Vec::with_capacity(read.len());
            for &index in &read {
                let column = arrow::compute::filter(rows.batch.column(index), &keep)?;
                columns.push(
                    crate::column_rename::coerce_column(&column, schema.field(index).data_type())
                        .map_err(df_error)?,
                );
            }
            let row_ids = rows
                .row_ids
                .iter()
                .copied()
                .filter(|row_id| inlined_ids.contains(row_id))
                .collect();
            parts.push(ReadPart {
                columns,
                row_ids,
            });
        }

        if read.is_empty() {
            // Nothing to read from the files: their rows' ids are all that is
            // asked for, and locating them found those.
            for positions in by_file.values() {
                parts.push(ReadPart {
                    columns: Vec::new(),
                    row_ids: positions.values().copied().collect(),
                });
            }
        } else if !by_file.is_empty() {
            let by_file = &by_file;
            let read = &read;
            let concurrency = state.config().target_partitions().max(1);
            let pending: Vec<BoxFuture<'_, crate::Result<Vec<ReadPart>>>> = located
                .files
                .iter()
                .filter_map(|file| {
                    let positions = by_file.get(&file.data_file_id)?;
                    Some(Box::pin(async move {
                        let wanted: HashSet<i64> = positions.keys().copied().collect();
                        let batches = self
                            .read_file_columns_at(state, file, &wanted, read)
                            .await
                            .map_err(df_error)?;
                        Ok(batches
                            .into_iter()
                            .map(|(columns, read_positions)| ReadPart {
                                columns,
                                row_ids: read_positions
                                    .values()
                                    .iter()
                                    .map(|position| positions[position])
                                    .collect(),
                            })
                            .collect())
                    })
                        as BoxFuture<'_, crate::Result<Vec<ReadPart>>>)
                })
                .collect();
            // Collected in listing order, whatever order the reads finish in.
            let files: Vec<Vec<ReadPart>> = futures::stream::iter(pending)
                .buffered(concurrency)
                .try_collect()
                .await?;
            parts.extend(files.into_iter().flatten());
        }

        assemble(&output_schema, &projection, &read, row_ids, &parts)
    }
}

/// The rows of `parts` with one of `requested`, one per id in the order the
/// ids were first requested, as `output_schema`: each of `projection`'s
/// columns taken from the part's column of the same index in `read`, or, past
/// the physical columns `read` names, the row's `rowid`.
fn assemble(
    output_schema: &SchemaRef,
    projection: &[usize],
    read: &[usize],
    requested: &[i64],
    parts: &[ReadPart],
) -> crate::Result<RowsByRowId> {
    let mut at: HashMap<i64, (usize, usize)> = HashMap::new();
    for (part_index, part) in parts.iter().enumerate() {
        for (row, row_id) in part.row_ids.iter().enumerate() {
            at.entry(*row_id).or_insert((part_index, row));
        }
    }
    let mut seen = HashSet::new();
    let mut indices = Vec::new();
    let mut row_ids = Vec::new();
    for row_id in requested {
        if let Some(&index) = at.get(row_id)
            && seen.insert(*row_id)
        {
            indices.push(index);
            row_ids.push(*row_id);
        }
    }
    if indices.is_empty() {
        return Ok(RowsByRowId {
            row_ids,
            batch: RecordBatch::new_empty(Arc::clone(output_schema)),
        });
    }
    let mut read_columns: Vec<ArrayRef> = Vec::with_capacity(read.len());
    for column in 0..read.len() {
        let arrays: Vec<&dyn arrow::array::Array> = parts
            .iter()
            .map(|part| part.columns[column].as_ref())
            .collect();
        read_columns.push(arrow::compute::interleave(&arrays, &indices)?);
    }
    let row_id_column: ArrayRef = Arc::new(arrow::array::Int64Array::from(row_ids.clone()));
    let columns: Vec<ArrayRef> = projection
        .iter()
        .map(|index| match read.binary_search(index) {
            Ok(position) => Arc::clone(&read_columns[position]),
            Err(_) => Arc::clone(&row_id_column),
        })
        .collect();
    let batch = RecordBatch::try_new_with_options(
        Arc::clone(output_schema),
        columns,
        &arrow::record_batch::RecordBatchOptions::new().with_row_count(Some(row_ids.len())),
    )?;
    Ok(RowsByRowId {
        row_ids,
        batch,
    })
}

/// Run `pending`, at most `limit` at once, and collect their results in
/// completion order. The futures are boxed so the caller's future stays
/// `Send` for every lifetime it borrows.
pub(crate) async fn bounded<'a, T, E>(
    pending: Vec<BoxFuture<'a, Result<T, E>>>,
    limit: usize,
) -> Result<Vec<T>, E> {
    futures::stream::iter(pending)
        .buffer_unordered(limit.max(1))
        .try_collect()
        .await
}

fn df_error(error: datafusion::error::DataFusionError) -> DuckLakeError {
    match error {
        datafusion::error::DataFusionError::External(inner) => {
            match inner.downcast::<DuckLakeError>() {
                Ok(error) => *error,
                Err(inner) => {
                    DuckLakeError::DataFusion(datafusion::error::DataFusionError::External(inner))
                },
            }
        },
        other => DuckLakeError::DataFusion(other),
    }
}
