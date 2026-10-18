//! Fork a table into another catalog of a multi-catalog PostgreSQL store.
//!
//! [`MulticatalogManager::fork_table`] creates (or replaces) a table in one
//! catalog whose content equals another catalog's table at the snapshot the
//! source [`DuckLakeTable`] was opened at, in one commit, copying no Parquet
//! data file:
//!
//! - every live data file is registered by reference
//!   ([`MetadataWriter::register_existing_data_files`]): its absolute path, with
//!   the catalog that owns the object (the source, or for a fork of a fork the
//!   root of the chain) named as owner, so neither catalog's maintenance
//!   deletes an object the other still reads;
//! - a file's delete file is referenced the same way, unless the file has
//!   deletions the source records inlined (`ducklake_inlined_delete_<id>`):
//!   then the fork gets a delete file of its own holding every deleted
//!   position, the referenced delete file's and the inlined ones, since an
//!   inlined deletion lives in the source's catalog and cannot be referenced;
//! - the source's inlined rows are written to a Parquet file of the fork's own,
//!   as official DuckLake's flush writes them (ordered by the source's sort
//!   keys at the snapshot, then `(row_id, begin_snapshot)`, each row keeping
//!   its `rowid` and insert snapshot in embedded columns), rolled over into
//!   further files at the target file size when the table is unpartitioned and
//!   split by partition when it is, and
//!   registered with the catalog row a flush records but for its
//!   `begin_snapshot`: `row_id_start` its smallest row id (NULL when the id of
//!   an inlined row deleted by the snapshot falls in its id range, since a
//!   fork writes only the live rows) and `partial_max` its largest
//!   insert snapshot, but the fork's
//!   snapshot as `begin_snapshot`, since the fork's catalog has no earlier one. The fork
//!   then holds what a flush of the source would have, without changing the
//!   source.
//!
//! A referenced partial file (one the source flushed or merged) keeps its
//! `partial_max`. The fork's change feed thus dates the rows of a file that
//! embeds insert snapshots by those snapshots (the source's, which precede
//! the fork's), as official reads such a file and as a merge of the fork's
//! files dates them; the rows of any other file are dated to the fork's
//! snapshot.
//!
//! The fork's columns copy the source's catalog rows: their ids, exact types
//! (`json`, `geometry`, `timetz`, ...), nullability and defaults, nested
//! fields' included. A list column in the legacy catalog shape (one
//! `list<...>` row, no `element` row) takes upstream's shape, its `element`
//! under a fresh id, as a write to the source would migrate it; a re-fork into
//! the same table keeps the id the earlier fork gave it. The source's
//! tags and comments, the table's and its columns', are the fork's, recorded
//! in the same commit.
//!
//! Row ids are preserved: each file keeps its `row_id_start` (or its embedded
//! row ids), and the fork's row id allocator starts past the source's, so a row
//! has the same `rowid` in both tables and new rows in either never collide
//! with carried ones.
//!
//! A source opened at an earlier snapshot than the catalog's head is forked as
//! it read then: a file's later deletions, and the rows of a partial file
//! inserted later, are masked by a delete file of the fork's own.
//!
//! Official DuckLake has no fork; the closest equivalent, `CREATE TABLE ... AS
//! SELECT`, copies the rows into new files with new row ids. A fork shares the
//! files and keeps the row ids instead, so a fork into the same catalog, where
//! one catalog would own a file twice, is refused.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow::array::{ArrayRef, Int64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::catalog::Session;
use object_store::path::Path as ObjectPath;

use crate::metadata_provider::DuckLakeTableFile;
use crate::metadata_writer::{
    CarriedTags, CatalogNode, ColumnDef, ColumnStat, CommitIds, DataFileInfo, DeleteFileInfo,
    MetadataWriter, PromoteLayout, PromotedFile, SourceFileRows, WriteMode,
};
use crate::multicatalog::MulticatalogManager;
use crate::table::DuckLakeTable;
use crate::table_writer::DuckLakeTableWriter;
use crate::{DuckLakeError, Result};

/// What [`MulticatalogManager::fork_table`] did.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct TableFork {
    /// The fork's commit.
    pub commit: CommitIds,
    /// Source data files the fork references.
    pub files_referenced: usize,
    /// Delete files the fork wrote of its own: for files with inlined
    /// deletions, or masking what a historical source did not see.
    pub delete_files_written: usize,
    /// Data files the fork wrote of its own, holding the source's inlined rows.
    pub inlined_files_written: usize,
    /// The source's inlined rows those files hold.
    pub inlined_rows: usize,
}

impl MulticatalogManager {
    /// Create `destination_schema.destination_table` in `destination`'s
    /// catalog holding exactly what `source` (a table of another catalog of
    /// this store, opened at the snapshot to fork) reads, in one commit; see
    /// the [module docs](crate::fork). An existing destination table is
    /// replaced ([`WriteMode::Replace`]) when its columns have the source's
    /// column ids (an earlier fork of the same source): a replace that changes
    /// its tags records them as `set_tag` does, bumping `schema_version` and
    /// recording `altered_table`.
    ///
    /// `state` supplies the object store the source's files and the fork's own
    /// files live in, which must be one store for both catalogs.
    ///
    /// Refused with [`DuckLakeError::InvalidConfig`], before anything is
    /// written, when `source` is a table of `destination`'s own catalog; when
    /// the destination table exists with a column of the source's under
    /// another column id, or one of the source's ids under another column (a
    /// table created there on its own): the fork's files carry the source's
    /// field ids and a replace keeps a column's id by name, so drop that table
    /// first; and when the source has a column tag other than `comment`, which
    /// [`MetadataWriter::set_tag`] refuses too. Refused with
    /// [`DuckLakeError::Unsupported`]: a source with a live file written under
    /// another partition spec than its current one (the fork establishes one
    /// partition generation for every file), with an encrypted data or delete
    /// file (a fork cannot carry the key), or with a file read through a name
    /// mapping (a fork does not carry the mapping). Refused with
    /// [`DuckLakeError::Conflict`], removing the files it wrote, when the
    /// source's expire or compaction removed a file it read before the fork
    /// committed (the commit locks the source's catalog to check); fork again.
    /// A dropped source catalog is [`DuckLakeError::CatalogNotFound`].
    ///
    /// # Example
    ///
    /// ```no_run
    /// # async fn example(
    /// #     state: &dyn datafusion::catalog::Session,
    /// #     manager: &datafusion_ducklake::MulticatalogManager,
    /// #     source: &datafusion_ducklake::DuckLakeTable,
    /// # ) -> datafusion_ducklake::Result<()> {
    /// let branch = manager.create_catalog("branch").await?;
    /// let writer = std::sync::Arc::new(manager.writer(branch).await?);
    /// let fork = manager
    ///     .fork_table(state, source, writer, "main", "orders")
    ///     .await?;
    /// println!("forked at snapshot {}", fork.commit.snapshot_id);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn fork_table(
        &self,
        state: &dyn Session,
        source: &DuckLakeTable,
        destination: Arc<dyn MetadataWriter>,
        destination_schema: &str,
        destination_table: &str,
    ) -> Result<TableFork> {
        let source_table_id = source.table_id();
        let source_catalog: Option<i64> = sqlx::query_scalar(
            "SELECT m.catalog_id FROM ducklake_table t
             JOIN ducklake_catalog_schema_map m ON m.schema_id = t.schema_id
             WHERE t.table_id = $1
             LIMIT 1",
        )
        .bind(source_table_id)
        .fetch_optional(self.pool())
        .await?;
        let source_catalog = source_catalog
            .ok_or_else(|| DuckLakeError::TableNotFound(format!("table_id {source_table_id}")))?;
        let destination_catalog = destination.catalog_id().ok_or_else(|| {
            DuckLakeError::InvalidConfig(
                "fork_table: the destination writer is not a multi-catalog PostgreSQL writer"
                    .to_string(),
            )
        })?;
        if source_catalog == destination_catalog {
            return Err(DuckLakeError::InvalidConfig(
                "fork_table: the destination is in the source's catalog; a fork references \
                 the source's files from another catalog of the store"
                    .to_string(),
            ));
        }
        let next_row_id: Option<i64> =
            sqlx::query_scalar("SELECT next_row_id FROM ducklake_table_stats WHERE table_id = $1")
                .bind(source_table_id)
                .fetch_optional(self.pool())
                .await?;
        let owners = self
            .table_file_owners(source_catalog, source_table_id)
            .await?;

        let provider = source.provider();
        let snapshot = source.snapshot_id();
        let at_head = provider.get_current_snapshot()? == snapshot;
        let physical_schema = source.physical_schema();
        let top_level_ids = source.top_level_column_ids();
        let source_column_ids = source.column_ids();
        // The destination table an earlier fork created, whose ids a legacy
        // list's `element` keeps.
        let existing = {
            let mut connection = self.pool().acquire().await?;
            crate::metadata_writer_postgres::live_named_table_columns(
                destination_catalog,
                destination_schema,
                destination_table,
                &mut connection,
            )
            .await?
            .map(|(_, columns)| columns)
            .unwrap_or_default()
        };
        let (columns, column_ids) = self
            .source_columns(
                source_table_id,
                snapshot,
                &physical_schema,
                &source_column_ids,
                &top_level_ids,
                &existing,
            )
            .await?;
        let tags = source_tags(
            provider.as_ref(),
            source_table_id,
            snapshot,
            &source_column_ids,
        )?;
        // Refused here, before any file is written, as the registration
        // would refuse them.
        crate::metadata_writer_postgres::validate_carried_tags(source_table_id, &tags)?;
        {
            let mut connection = self.pool().acquire().await?;
            crate::metadata_writer_postgres::refuse_foreign_column_ids(
                destination_catalog,
                destination_schema,
                destination_table,
                &columns,
                &column_ids,
                &mut connection,
            )
            .await?;
        }

        // The layout the fork establishes: the source's partition spec and
        // sort order at the forked snapshot.
        let partition_spec = provider.get_partition_spec(source_table_id, snapshot)?;
        let sort_spec = provider.get_sort_spec(source_table_id, snapshot)?;
        let name_of = |column_id: i64| -> Result<String> {
            top_level_ids
                .iter()
                .position(|id| *id == column_id)
                .map(|index| physical_schema.field(index).name().clone())
                .ok_or_else(|| {
                    DuckLakeError::Internal(format!(
                        "partition column {column_id} is not a live column of the source"
                    ))
                })
        };
        let layout = PromoteLayout {
            partition_by: match &partition_spec {
                Some(spec) => spec
                    .columns
                    .iter()
                    .map(|column| Ok((name_of(column.column_id)?, column.transform.clone())))
                    .collect::<Result<Vec<_>>>()?,
                None => Vec::new(),
            },
            sorted_by: sort_spec
                .as_ref()
                .map(|spec| spec.fields.clone())
                .unwrap_or_default(),
        };

        // Each live file with its statistics and partition values, listed
        // after the source's inlined deletions and inlined rows, so that a
        // flush of the source between the reads moves no row or deletion out
        // of all of them (`DuckLakeTable::inlined_rows_then_files`).
        let adopted: HashSet<i64> = column_ids.iter().copied().collect();
        let inlined_deletes = source.inlined_deletes_by_file().map_err(df_error)?;
        let (rows, mut metadata) = source.inlined_rows_then_files(
            || {
                let mut metadata = Vec::new();
                let mut after = None;
                loop {
                    let page = provider.get_table_file_metadata_page(
                        source_table_id,
                        snapshot,
                        after,
                        crate::metadata_provider::FILE_METADATA_BATCH_SIZE,
                    )?;
                    let Some(last) = page.last() else {
                        break;
                    };
                    after = Some(last.file.data_file_id);
                    let last_page = page.len() < crate::metadata_provider::FILE_METADATA_BATCH_SIZE;
                    metadata.extend(page);
                    if last_page {
                        break;
                    }
                }
                Ok(metadata)
            },
            |entry| entry.file.data_file_id,
        )?;
        // A file of no rows holds nothing the fork reads: the 0-row file a
        // Replace with no rows commits to retire the prior generation (with no
        // partition values on a partitioned table). Official writes no file for
        // an empty insert, so the fork leaves it out.
        metadata.retain(|entry| entry.file.max_row_count != Some(0));
        for entry in &metadata {
            let file = &entry.file;
            if file.file.encryption_key.is_some()
                || file
                    .delete_file
                    .as_ref()
                    .is_some_and(|delete| delete.encryption_key.is_some())
            {
                return Err(DuckLakeError::Unsupported(format!(
                    "fork_table: data file {} is encrypted; a fork cannot carry its key",
                    file.data_file_id
                )));
            }
            if file.file.mapping_id.is_some() {
                return Err(DuckLakeError::Unsupported(format!(
                    "fork_table: data file {} reads through a name mapping, which a fork \
                     does not carry",
                    file.data_file_id
                )));
            }
            if let Some(spec) = &partition_spec
                && file.partition_id != Some(spec.partition_id)
            {
                return Err(DuckLakeError::Unsupported(format!(
                    "fork_table: data file {} was written under another partition spec than \
                     the source's at snapshot {snapshot} (or before it was partitioned); a \
                     fork establishes that one spec for every file",
                    file.data_file_id
                )));
            }
        }

        let object_store = state
            .runtime_env()
            .object_store(source.object_store_url().as_ref())?;
        let table_writer =
            DuckLakeTableWriter::new(Arc::clone(&destination), Arc::clone(&object_store))?
                .with_options(source.write_options());

        // Every object the fork writes, removed when it commits nothing.
        let mut written: Vec<ObjectPath> = Vec::new();
        let result = async {
            let mut promoted = Vec::with_capacity(metadata.len() + 1);
            let mut delete_files_written = 0;
            for entry in &metadata {
                let file = &entry.file;
                let path = crate::path_resolver::resolve_path(
                    source.table_path(),
                    &file.file.path,
                    file.file.path_is_relative,
                )?;
                let owner = owners
                    .data_files
                    .get(&file.data_file_id)
                    .copied()
                    .unwrap_or(source_catalog);
                let mut info = DataFileInfo::new(
                    path.clone(),
                    file.file.file_size_bytes,
                    file.max_row_count.unwrap_or(0),
                )
                .with_absolute_path()
                .with_owner_catalog(owner)
                .with_column_stats(column_stats(&entry.column_statistics, &adopted))
                .with_source_row_id_start(file.row_id_start);
                if let Some(footer) = file.file.footer_size {
                    info = info.with_footer_size(footer);
                }
                if let Some(floor) = next_row_id {
                    info = info.with_row_id_floor(floor);
                }
                if partition_spec.is_some() {
                    info = info.with_partition_values(file.partition_values.clone());
                }
                let inlined = inlined_deletes.get(&file.data_file_id);
                let masked_later = !at_head
                    && (file.delete_file.is_some()
                        || file.partial_max.is_some_and(|max| max > snapshot));
                let referenced_delete = !(inlined.is_some() || masked_later);
                let delete = if !referenced_delete {
                    let positions = deleted_at(source, state, file, inlined, snapshot).await?;
                    if positions.is_empty() {
                        None
                    } else {
                        let delete = table_writer
                            .write_delete_file(
                                destination_schema,
                                destination_table,
                                &path,
                                &positions,
                            )
                            .await?;
                        written.push(table_writer.staged_object_path(
                            destination_schema,
                            destination_table,
                            &delete.path,
                            delete.path_is_relative,
                        )?);
                        delete_files_written += 1;
                        Some(delete)
                    }
                } else {
                    match (&file.delete_file, file.delete_file_id) {
                        (Some(delete_file), Some(delete_file_id)) => {
                            let delete_path = crate::path_resolver::resolve_path(
                                source.table_path(),
                                &delete_file.path,
                                delete_file.path_is_relative,
                            )?;
                            let owner = owners
                                .delete_files
                                .get(&delete_file_id)
                                .copied()
                                .unwrap_or(source_catalog);
                            let mut delete = DeleteFileInfo::new(
                                delete_path,
                                delete_file.file_size_bytes,
                                file.delete_count.unwrap_or(0),
                            )
                            .with_absolute_path()
                            .with_owner_catalog(owner);
                            if let Some(footer) = delete_file.footer_size {
                                delete = delete.with_footer_size(footer);
                            }
                            Some(delete)
                        },
                        _ => None,
                    }
                };
                // A partial file (flushed or merged) keeps its `partial_max`, so the
                // fork dates its rows by their embedded snapshots, as the source
                // does and as a merge of the fork's files would.
                let promoted_file = PromotedFile::new(info)
                    .with_partial_max(file.partial_max)
                    .with_source_rows(SourceFileRows {
                        catalog_id: source_catalog,
                        data_file_id: file.data_file_id,
                        delete_file_id: if referenced_delete {
                            file.delete_file_id
                        } else {
                            None
                        },
                    });
                promoted.push(match delete {
                    Some(delete) => promoted_file.with_delete(delete),
                    None => promoted_file,
                });
            }

            // The source's inlined rows, as a flush writes them.
            let mut inlined_rows = 0;
            let mut inlined_files_written = 0;
            if !rows.is_empty() {
                // The keys the fork's own files of inlined rows are sorted by, as
                // a flush of the fork's table sorts them.
                let sort_keys = crate::inlined_flush::sort_keys_of(source, sort_spec.clone())?;
                // The row ids of inlined rows deleted by the snapshot, which a
                // flush writes (with a delete file) and the fork leaves out.
                let live: HashSet<i64> = rows
                    .iter()
                    .flat_map(|data| data.row_ids.iter().copied())
                    .collect();
                let omitted: std::collections::BTreeSet<i64> = provider
                    .get_inlined_data_for_flush(source_table_id, snapshot)?
                    .iter()
                    .flat_map(|data| {
                        data.rows
                            .row_ids
                            .iter()
                            .zip(&data.rows.end_snapshots)
                            .filter(|(_, end)| end.is_some_and(|end| end <= snapshot))
                            .map(|(row_id, _)| *row_id)
                    })
                    .filter(|row_id| !live.contains(row_id))
                    .collect();
                let mut batches = Vec::with_capacity(rows.len());
                for data in &rows {
                    let mut columns = Vec::with_capacity(physical_schema.fields().len() + 2);
                    for (column, field) in data.batch.columns().iter().zip(physical_schema.fields())
                    {
                        columns.push(
                            crate::column_rename::coerce_column(column, field.data_type())
                                .map_err(DuckLakeError::DataFusion)?,
                        );
                    }
                    columns.push(Arc::new(Int64Array::from(data.row_ids.clone())) as ArrayRef);
                    columns
                        .push(Arc::new(Int64Array::from(data.begin_snapshots.clone())) as ArrayRef);
                    batches.push(RecordBatch::try_new(
                        lineage_schema(&physical_schema),
                        columns,
                    )?);
                    inlined_rows += data.row_ids.len();
                }
                let groups: Vec<PartitionedRows> = match &partition_spec {
                    Some(spec) => {
                        let write_spec = crate::partition::PartitionWriteSpec::resolve(
                            spec,
                            &top_level_ids,
                            &physical_schema,
                        )?;
                        crate::partition::split_batches_by_partition(
                            &lineage_schema(&physical_schema),
                            &batches,
                            &write_spec,
                        )?
                        .into_iter()
                        .map(|(values, batches)| (Some(values), batches))
                        .collect()
                    },
                    None => vec![(None, batches)],
                };
                let key_names = match &partition_spec {
                    Some(spec) => crate::partition::PartitionWriteSpec::resolve(
                        spec,
                        &top_level_ids,
                        &physical_schema,
                    )?
                    .key_names(),
                    None => Vec::new(),
                };
                let data_width = physical_schema.fields().len();
                for (values, batches) in groups {
                    let subpath = values
                        .as_ref()
                        .map(|values| crate::partition::hive_subpath(&key_names, values));
                    // The rows in the order official's flush writes them: by the
                    // table's sort keys, then `(row_id, begin_snapshot)`.
                    let rows = flush_ordered(
                        source,
                        sort_keys.as_deref(),
                        &top_level_ids,
                        &arrow::compute::concat_batches(
                            &lineage_schema(&physical_schema),
                            &batches,
                        )?,
                        data_width,
                    )?;
                    let batches: Vec<RecordBatch> = (0..rows.num_rows())
                        .step_by(crate::inlined_flush::FLUSH_BATCH_ROWS)
                        .map(|start| {
                            rows.slice(
                                start,
                                crate::inlined_flush::FLUSH_BATCH_ROWS.min(rows.num_rows() - start),
                            )
                        })
                        .collect();
                    let mut paths: Vec<ObjectPath> = Vec::new();
                    let files = table_writer
                        .write_rolled_lineage_files(
                            destination_schema,
                            destination_table,
                            &physical_schema,
                            &column_ids,
                            &top_level_ids,
                            &batches,
                            subpath.as_deref(),
                            // Official writes a partitioned flush as one file per
                            // partition, rolling over only an unpartitioned one.
                            values.is_none(),
                            &mut paths,
                        )
                        .await;
                    written.extend(paths);
                    // Each file holds the next `record_count` rows of the group.
                    // As official's flush records them: `row_id_start` the
                    // smallest row id it embeds and `partial_max` the largest
                    // snapshot.
                    let mut file_start = 0usize;
                    for file in files? {
                        let count = usize::try_from(file.record_count).map_err(|_| {
                            DuckLakeError::Internal("forked file has a negative row count".into())
                        })?;
                        if file_start + count > rows.num_rows() {
                            return Err(DuckLakeError::Internal(
                                "forked files hold more rows than were written".into(),
                            ));
                        }
                        let slice = rows.slice(file_start, count);
                        file_start += count;
                        let column = |index: usize| {
                            slice
                                .column(index)
                                .as_any()
                                .downcast_ref::<Int64Array>()
                                .cloned()
                                .ok_or_else(|| {
                                    DuckLakeError::Internal(
                                        "forked lineage column is not Int64".into(),
                                    )
                                })
                        };
                        let (Some(min_row_id), Some(partial_max)) = (
                            arrow::compute::min(&column(data_width)?),
                            arrow::compute::max(&column(data_width + 1)?),
                        ) else {
                            return Err(DuckLakeError::Internal(
                                "forked file holds no inlined row".into(),
                            ));
                        };
                        // As official's flush records it: `row_id_start` the
                        // smallest row id, whatever order the sort keys, the
                        // partitions or the rollover leave the ids in (each row's id
                        // is embedded). A flush writes deleted inlined rows too; the
                        // fork leaves them out, and records none for a file whose id
                        // range the id of one of them falls in, so an adjacent merge
                        // of the fork's files never renumbers rows across it. (In a
                        // partitioned table that row may belong to another partition's
                        // file; recording none keeps every rowid either way.)
                        let max_row_id =
                            arrow::compute::max(&column(data_width)?).unwrap_or(min_row_id);
                        let contiguous = omitted.range(min_row_id..=max_row_id).next().is_none();
                        let mut file = if contiguous {
                            file.with_source_row_id_start(Some(min_row_id))
                        } else {
                            file.with_embedded_row_ids()
                        };
                        if let Some(floor) = next_row_id {
                            file = file.with_row_id_floor(floor);
                        }
                        if let Some(values) = &values {
                            file = file.with_partition_values(
                                values
                                    .iter()
                                    .enumerate()
                                    .map(|(index, value)| (index as i32, value.clone()))
                                    .collect(),
                            );
                        }
                        inlined_files_written += 1;
                        promoted.push(PromotedFile::new(file).with_partial_max(Some(partial_max)));
                    }
                    if file_start != rows.num_rows() {
                        return Err(DuckLakeError::Internal(format!(
                            "forked files hold {file_start} of {} inlined rows",
                            rows.num_rows()
                        )));
                    }
                }
            }

            let commit = destination.register_existing_data_files_with_tags(
                destination_schema,
                destination_table,
                &columns,
                &column_ids,
                &promoted,
                (!layout.is_empty()).then_some(&layout),
                WriteMode::Replace,
                &tags,
            )?;
            Ok(TableFork {
                commit,
                files_referenced: metadata.len(),
                delete_files_written,
                inlined_files_written,
                inlined_rows,
            })
        }
        .await;
        if let Err(error) = &result
            && crate::table_writer::commit_definitely_rolled_back(error)
        {
            for failure in crate::table_writer::remove_objects(&object_store, written).await {
                tracing::warn!(error = %failure, "failed to remove a file of a failed fork");
            }
        }
        result
    }
}

/// `rows` (data columns, then the row id at `data_width` and the insert
/// snapshot after it) in the order official's flush writes inlined rows
/// ([`crate::inlined_flush::flush_order`]).
fn flush_ordered(
    source: &DuckLakeTable,
    sort_keys: Option<&[crate::inlined_flush::FlushSortKey]>,
    column_ids: &[i64],
    rows: &RecordBatch,
    data_width: usize,
) -> Result<RecordBatch> {
    let data = rows.project(&(0..data_width).collect::<Vec<_>>())?;
    let indices = crate::inlined_flush::flush_order(
        source,
        sort_keys,
        &data,
        column_ids,
        rows.column(data_width),
        rows.column(data_width + 1),
    )?;
    let columns = rows
        .columns()
        .iter()
        .map(|column| arrow::compute::take(column.as_ref(), &indices, None))
        .collect::<std::result::Result<Vec<ArrayRef>, _>>()?;
    Ok(RecordBatch::try_new(rows.schema(), columns)?)
}

impl MulticatalogManager {
    /// The source's live columns at `snapshot` as the fork creates them, with
    /// their column ids (depth first): each copying the source's catalog rows
    /// (its nested fields' included), so the fork keeps their exact types
    /// (`json`, `geometry`, `timetz`, ...), nullability and defaults.
    ///
    /// A list column in the legacy catalog shape (one `list<...>` row and no
    /// `element` row) takes upstream's shape, as a write to the source would
    /// migrate it: a `list` row and an `element` row (and its nested fields'),
    /// typed from the column's Arrow type. The nested rows keep the ids the
    /// destination's live table (`existing`, from an earlier fork) gives them
    /// under the same column, so a re-fork replaces it; else they take fresh
    /// ids.
    async fn source_columns(
        &self,
        table_id: i64,
        snapshot: i64,
        physical_schema: &Schema,
        column_ids: &[i64],
        top_level_ids: &[i64],
        existing: &[crate::metadata_writer::ExistingCatalogColumn],
    ) -> Result<(Vec<ColumnDef>, Vec<i64>)> {
        let rows = sqlx::query(
            "SELECT column_id, column_type, nulls_allowed, initial_default, default_value,
                    default_value_type, default_value_dialect
             FROM ducklake_column
             WHERE table_id = $1 AND $2 >= begin_snapshot
               AND ($2 < end_snapshot OR end_snapshot IS NULL)",
        )
        .bind(table_id)
        .bind(snapshot)
        .fetch_all(self.pool())
        .await?;
        let mut by_id: HashMap<i64, CatalogNode> = HashMap::with_capacity(rows.len());
        for row in &rows {
            use sqlx::Row;
            by_id.insert(
                row.try_get(0)?,
                CatalogNode {
                    ducklake_type: row.try_get(1)?,
                    is_nullable: row.try_get::<Option<bool>, _>(2)?.unwrap_or(true),
                    initial_default: row.try_get(3)?,
                    default_value: row.try_get(4)?,
                    default_value_type: row.try_get(5)?,
                    default_value_dialect: row.try_get(6)?,
                },
            );
        }
        let mut columns = Vec::with_capacity(top_level_ids.len());
        let mut fork_ids = Vec::with_capacity(column_ids.len());
        for (index, (field, root)) in physical_schema
            .fields()
            .iter()
            .zip(top_level_ids)
            .enumerate()
        {
            let start = column_ids
                .iter()
                .position(|id| id == root)
                .ok_or_else(|| DuckLakeError::Internal(format!("column {root} has no field id")))?;
            let end = top_level_ids
                .get(index + 1)
                .and_then(|next| column_ids.iter().position(|id| id == next))
                .unwrap_or(column_ids.len());
            let mut ids = column_ids
                .get(start..end)
                .ok_or_else(|| {
                    DuckLakeError::Internal(format!(
                        "the field ids of column '{}' are out of order",
                        field.name()
                    ))
                })?
                .to_vec();
            let mut nodes = ids
                .iter()
                .map(|id| by_id.get(id).cloned())
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| {
                    DuckLakeError::Internal(format!(
                        "column '{}' of the source has no catalog row at snapshot {snapshot}",
                        field.name()
                    ))
                })?;
            let shape = crate::metadata_writer::catalog_column_defs(&[ColumnDef::from_arrow(
                field.name(),
                field.data_type(),
                field.is_nullable(),
            )?])?;
            if nodes.len() != shape.len() {
                let legacy_list = nodes.len() == 1
                    && nodes[0]
                        .ducklake_type
                        .trim()
                        .to_ascii_lowercase()
                        .starts_with("list<");
                if !legacy_list {
                    return Err(DuckLakeError::Unsupported(format!(
                        "fork_table: column '{}' has {} catalog rows for the {} fields of its \
                         type {}; a fork cannot represent its shape",
                        field.name(),
                        nodes.len(),
                        shape.len(),
                        field.data_type()
                    )));
                }
                nodes[0].ducklake_type.clone_from(&shape[0].ducklake_type);
                for def in &shape[1..] {
                    nodes.push(CatalogNode {
                        ducklake_type: def.ducklake_type.clone(),
                        is_nullable: def.is_nullable,
                        initial_default: None,
                        default_value: None,
                        default_value_type: None,
                        default_value_dialect: None,
                    });
                }
                let nested =
                    match existing_descendants(existing, *root, &shape) {
                        Some(nested) => nested,
                        None => sqlx::query_scalar(
                            "SELECT nextval(pg_get_serial_sequence('ducklake_column', 'column_id'))
                             FROM generate_series(1, $1)",
                        )
                        .bind((shape.len() - 1) as i64)
                        .fetch_all(self.pool())
                        .await?,
                    };
                ids.extend(nested);
            }
            fork_ids.extend(ids);
            columns.push(ColumnDef::copying(field.name(), field.data_type(), nodes)?);
        }
        Ok((columns, fork_ids))
    }
}

/// The ids of the columns nested under column `root` of `existing` (a live
/// table's columns), depth first, when they have the names `shape` (the
/// column's catalog rows, `shape[0]` the column itself) gives them, in its
/// order; else `None`.
fn existing_descendants(
    existing: &[crate::metadata_writer::ExistingCatalogColumn],
    root: i64,
    shape: &[crate::metadata_writer::CatalogColumnDef],
) -> Option<Vec<i64>> {
    if !existing
        .iter()
        .any(|column| column.column_id == root && column.parent_column.is_none())
    {
        return None;
    }
    // Pre-order walk, children in `existing`'s (column order) order.
    let mut ids = Vec::with_capacity(shape.len() - 1);
    let mut names = Vec::with_capacity(shape.len() - 1);
    let mut stack = vec![root];
    while let Some(parent) = stack.pop() {
        if parent != root {
            ids.push(parent);
        }
        let children: Vec<&crate::metadata_writer::ExistingCatalogColumn> = existing
            .iter()
            .filter(|column| column.parent_column == Some(parent))
            .collect();
        for child in children.iter().rev() {
            stack.push(child.column_id);
        }
        if parent != root {
            let name = existing
                .iter()
                .find(|column| column.column_id == parent)
                .map(|column| column.name.as_str())?;
            names.push(name);
        }
    }
    let wanted: Vec<&str> = shape[1..].iter().map(|def| def.name.as_str()).collect();
    (names == wanted).then_some(ids)
}

/// The source table's tags at `snapshot`, its comment and its columns'
/// tags among them, as the fork records them.
fn source_tags(
    provider: &dyn crate::metadata_provider::MetadataProvider,
    table_id: i64,
    snapshot: i64,
    column_ids: &[i64],
) -> Result<CarriedTags> {
    use crate::metadata_provider::{TagObjectType, TagTarget};
    let table = provider
        .get_tags(
            TagTarget::Object {
                object_type: TagObjectType::Table,
                object_id: table_id,
            },
            snapshot,
        )?
        .into_iter()
        .map(|tag| (tag.key, tag.value))
        .collect();
    let live: HashSet<i64> = column_ids.iter().copied().collect();
    let columns = provider
        .list_all_column_tags(snapshot)?
        .into_iter()
        .filter(|tag| tag.table_id == table_id && live.contains(&tag.column_id))
        .map(|tag| (tag.column_id, tag.tag.key, tag.tag.value))
        .collect();
    Ok(CarriedTags {
        table,
        columns,
    })
}

/// Rows of one partition (its values; `None` for an unpartitioned table).
type PartitionedRows = (Option<Vec<Option<String>>>, Vec<RecordBatch>);

/// `schema`'s columns followed by the embedded row id and insert snapshot a
/// flush writes.
fn lineage_schema(schema: &Schema) -> Arc<Schema> {
    let mut fields: Vec<Field> = schema
        .fields()
        .iter()
        .map(|field| field.as_ref().clone().with_nullable(true))
        .collect();
    fields.push(Field::new("__ducklake_fork_row_id", DataType::Int64, true));
    fields.push(Field::new(
        "__ducklake_fork_snapshot",
        DataType::Int64,
        true,
    ));
    Arc::new(Schema::new(fields))
}

/// The positions of `file` its fork must mask: those deleted at `snapshot`
/// (its delete file's, as of the snapshot, and `inlined`), and, for a partial
/// file, the rows inserted after it. Sorted.
async fn deleted_at(
    source: &DuckLakeTable,
    state: &dyn Session,
    file: &DuckLakeTableFile,
    inlined: Option<&HashSet<i64>>,
    snapshot: i64,
) -> Result<Vec<i64>> {
    debug_assert_eq!(source.snapshot_id(), snapshot);
    let mut positions = source
        .deleted_positions_at_snapshot(state, file, inlined)
        .await
        .map_err(df_error)?;
    // The rows a partial file holds that were inserted after the snapshot, as
    // the scan masks them, whether or not the file embeds row ids.
    positions.extend(
        source
            .positions_inserted_later(state, file)
            .await
            .map_err(df_error)?,
    );
    let mut positions: Vec<i64> = positions.into_iter().collect();
    positions.sort_unstable();
    Ok(positions)
}

/// The catalog statistics of one file, as a promoted file carries them, for
/// the columns the fork adopts: a file written before a column was dropped
/// still has statistics for it.
fn column_stats(
    statistics: &[crate::metadata_provider::DuckLakeFileColumnStatistics],
    adopted: &HashSet<i64>,
) -> Vec<ColumnStat> {
    let mut by_column: HashMap<i64, ColumnStat> = HashMap::new();
    for statistic in statistics
        .iter()
        .filter(|statistic| adopted.contains(&statistic.column_id))
    {
        by_column.entry(statistic.column_id).or_insert(ColumnStat {
            column_id: statistic.column_id,
            min_value: statistic.min_value.clone(),
            max_value: statistic.max_value.clone(),
            null_count: statistic.null_count,
            value_count: statistic.value_count,
            contains_nan: statistic.contains_nan,
            column_size_bytes: statistic.column_size_bytes,
        });
    }
    let mut stats: Vec<ColumnStat> = by_column.into_values().collect();
    stats.sort_by_key(|stat| stat.column_id);
    stats
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
