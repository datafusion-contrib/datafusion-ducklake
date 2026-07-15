//! Explicit, triggered DuckLake compaction for a single table.
//!
//! Two maintenance operations, each invoked programmatically (never
//! automatically on write) and returning a [`CompactionResult`] with metrics:
//!
//! 1. [`DuckLakeTable::merge_adjacent_files`] coalesces several small data files
//!    of one table (of the SAME schema version — never across a DDL boundary)
//!    into fewer larger ones. A merged file that spans more than one origin
//!    snapshot is written as a DuckLake **partial data file**: it embeds each
//!    row's original rowid AND a per-row `_ducklake_internal_snapshot_id` column,
//!    and its catalog row records `partial_max` (the maximum origin snapshot id
//!    among its rows), so time travel / change feeds can still attribute every
//!    merged row to its origin snapshot.
//! 2. [`DuckLakeTable::rewrite_data_files`] rewrites a data file whose deleted
//!    fraction exceeds a threshold (DuckDB's default is 0.95): it reads only the
//!    file's LIVE rows (delete-aware), writes them to a new file preserving each
//!    row's rowid, and retires BOTH the old data file and its delete file.
//!
//! Both operations commit ATOMICALLY in one snapshot via
//! `MetadataWriter::commit_compaction`: the rewritten outputs are registered, the
//! source files (and, for a rewrite, their delete files) are retired
//! (`end_snapshot` set) and scheduled for physical deletion, and
//! `ducklake_snapshot_changes.changes_made` records `compacted_table:<table_id>`.
//! Compaction changes the physical layout, not the logical rows, so the commit is
//! structured NOT to conflict with a concurrent append; it aborts only if a
//! source file was retired, or its live rows changed, since it was read (the
//! `base_snapshot` conflict check), which prevents ever resurrecting a
//! retired/deleted row into an output.
//!
//! Retired files are only SCHEDULED for deletion, never removed here, so time
//! travel to a pre-compaction snapshot still reads them until
//! [`cleanup_old_files_sqlite`](crate::maintenance::cleanup_old_files_sqlite)
//! reclaims them.

use std::collections::HashSet;
use std::sync::Arc;

use arrow::array::{ArrayRef, Int64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::catalog::Session;

use crate::metadata_provider::DuckLakeTableFile;
use crate::metadata_writer::{CompactionOutputFile, CompactionSourceFile, SourceRetirement};
use crate::row_id::EMBEDDED_SNAPSHOT_ID_COLUMN_NAME;
use crate::table::DuckLakeTable;
use crate::table_writer::DuckLakeTableWriter;
use crate::{DuckLakeError, Result};

/// Options for [`DuckLakeTable::merge_adjacent_files`].
#[derive(Debug, Clone)]
pub struct MergeOptions {
    /// Bin-pack adjacent small files (in `(schema_version, data_file_id)` order)
    /// until a bin reaches this many bytes, then emit it as one merged file.
    /// Files already at or above this size are left alone.
    pub target_file_size: u64,
    /// Cap on the number of source files considered in one call, to bound the
    /// memory and I/O of a single merge (candidates are taken in
    /// `(schema_version, data_file_id)` order).
    pub max_merged_files: usize,
    /// Skip files smaller than this many bytes. `0` makes every below-target file
    /// a candidate.
    pub min_file_size: u64,
}

impl Default for MergeOptions {
    fn default() -> Self {
        Self {
            // 256 MiB: large enough to coalesce many small files into one, while
            // staying a reasonable single-file target.
            target_file_size: 256 * 1024 * 1024,
            max_merged_files: 1024,
            min_file_size: 0,
        }
    }
}

/// Options for [`DuckLakeTable::rewrite_data_files`].
#[derive(Debug, Clone)]
pub struct RewriteOptions {
    /// Rewrite a data file only when the fraction of its rows masked by its live
    /// delete file is at least this value. DuckDB's default is `0.95`. Must be in
    /// `[0.0, 1.0]`.
    pub delete_threshold: f64,
}

impl Default for RewriteOptions {
    fn default() -> Self {
        Self {
            delete_threshold: 0.95,
        }
    }
}

/// Metrics returned by a compaction operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompactionResult {
    /// Number of source data files retired (merged or rewritten).
    pub files_processed: usize,
    /// Number of new (merged / rewritten) files written and registered.
    pub files_created: usize,
    /// Total rows written into the new files.
    pub rows_written: i64,
}

impl CompactionResult {
    /// A no-op result: nothing matched the operation's criteria.
    fn empty() -> Self {
        Self {
            files_processed: 0,
            files_created: 0,
            rows_written: 0,
        }
    }

    /// Whether the operation actually compacted anything (retired a source file).
    /// A `false` result committed no snapshot.
    pub fn did_work(&self) -> bool {
        self.files_processed > 0
    }
}

/// Append a constant `_ducklake_internal_snapshot_id` column (every value =
/// `origin`) to a `[data columns..., rowid]` batch, yielding
/// `[data columns..., rowid, snapshot_id]` for a merged partial file. Only the
/// column order matters here; `write_compacted_file` re-imposes the
/// field-id-tagged parquet schema.
fn append_snapshot_column(batch: &RecordBatch, origin: i64) -> Result<RecordBatch> {
    let n = batch.num_rows();
    let snap: ArrayRef = Arc::new(Int64Array::from(vec![origin; n]));
    let mut cols: Vec<ArrayRef> = batch.columns().to_vec();
    cols.push(snap);
    let mut fields: Vec<Field> = batch
        .schema()
        .fields()
        .iter()
        .map(|f| f.as_ref().clone())
        .collect();
    fields.push(Field::new(
        EMBEDDED_SNAPSHOT_ID_COLUMN_NAME,
        DataType::Int64,
        true,
    ));
    Ok(RecordBatch::try_new(Arc::new(Schema::new(fields)), cols)?)
}

impl DuckLakeTable {
    /// Merge several small adjacent data files of this table into fewer larger
    /// ones, committing the new layout in ONE snapshot.
    ///
    /// Candidates are the table's live files that have no live delete file, whose
    /// size is in `[min_file_size, target_file_size)`, and whose origin snapshot
    /// and schema version are known. They are grouped by schema version (so a DDL
    /// boundary is never crossed) and, within a group, bin-packed in
    /// `data_file_id` order until a bin reaches `target_file_size`; only bins of
    /// two or more files are merged. Delete-bearing files are deliberately left
    /// to [`rewrite_data_files`](Self::rewrite_data_files).
    ///
    /// Each source file's live rows are read with their original rowids
    /// preserved; a merged file that spans more than one origin snapshot is
    /// written as a partial file (embedding the per-row
    /// `_ducklake_internal_snapshot_id` column and recording `partial_max`). The
    /// sources are retired and scheduled for deletion in the same commit.
    ///
    /// Returns no-op metrics (and commits no snapshot) when nothing qualifies.
    /// Errors if the table is read-only (open the catalog with a writer) or if a
    /// source file's rowid lineage cannot be reconstructed.
    pub async fn merge_adjacent_files(
        &self,
        state: &dyn Session,
        opts: MergeOptions,
    ) -> Result<CompactionResult> {
        let writer = self.writer().ok_or_else(|| {
            DuckLakeError::InvalidConfig(
                "merge_adjacent_files: table is read-only; open the catalog with a writer"
                    .to_string(),
            )
        })?;
        let schema_name = self.schema_name().ok_or_else(|| {
            DuckLakeError::Internal("writable table has no schema name".to_string())
        })?;

        // Candidates: live, delete-free, below-target files with a known origin
        // snapshot + schema version, ordered so adjacency and same-version
        // grouping fall out of the sort.
        let table_files = self.files()?;
        let mut candidates: Vec<&DuckLakeTableFile> = table_files
            .iter()
            .filter(|f| {
                f.delete_file_id.is_none()
                    // Never re-merge an existing partial file: its rows carry
                    // per-row origins in the embedded `_ducklake_internal_snapshot_id`
                    // column, which the read path used to reconstruct them does NOT
                    // surface — re-merging would collapse every row onto the file's
                    // single begin_snapshot and corrupt time travel.
                    && f.partial_max.is_none()
                    && f.begin_snapshot.is_some()
                    && f.schema_version.is_some()
                    && (f.file.file_size_bytes as u64) >= opts.min_file_size
                    && (f.file.file_size_bytes as u64) < opts.target_file_size
            })
            .collect();
        candidates.sort_by_key(|f| (f.schema_version.unwrap_or(0), f.data_file_id));
        candidates.truncate(opts.max_merged_files);

        // Bin-pack within each schema-version run; only bins of >= 2 files merge.
        let mut bins: Vec<Vec<&DuckLakeTableFile>> = Vec::new();
        let mut i = 0;
        while i < candidates.len() {
            let version = candidates[i].schema_version;
            let mut running: u64 = 0;
            let mut bin: Vec<&DuckLakeTableFile> = Vec::new();
            while i < candidates.len() && candidates[i].schema_version == version {
                bin.push(candidates[i]);
                running += candidates[i].file.file_size_bytes as u64;
                i += 1;
                if running >= opts.target_file_size {
                    break;
                }
            }
            if bin.len() >= 2 {
                bins.push(bin);
            }
        }
        if bins.is_empty() {
            return Ok(CompactionResult::empty());
        }

        let object_store = state
            .runtime_env()
            .object_store(self.object_store_url().as_ref())?;
        let table_writer = DuckLakeTableWriter::new(Arc::clone(writer), object_store)?;
        let column_ids = self.column_ids();
        let physical_schema = self.physical_schema();

        let mut sources: Vec<CompactionSourceFile> = Vec::new();
        let mut outputs: Vec<CompactionOutputFile> = Vec::new();
        let mut files_processed = 0usize;
        let mut rows_written = 0i64;

        for bin in &bins {
            // Safety: the merged output is written at the table's CURRENT schema,
            // so a source carrying a column dropped since it was written would
            // lose that column's data (and its source is then removed). Skip any
            // such group entirely — those files are left uncompacted rather than
            // silently losing data. (The common case — files at the current
            // schema, or an older schema that only ADDED columns — is unaffected.)
            let mut bin_would_drop_columns = false;
            for tf in bin {
                if self.file_drops_current_columns(state, &tf.file).await? {
                    bin_would_drop_columns = true;
                    break;
                }
            }
            if bin_would_drop_columns {
                continue;
            }

            // Read each source's live rows (with original rowids) and its origin.
            let mut per_source: Vec<(Vec<RecordBatch>, i64)> = Vec::with_capacity(bin.len());
            for tf in bin {
                let scan = self.build_update_scan(state, tf).await?;
                let batches =
                    datafusion::physical_plan::collect(Arc::clone(&scan.scan), state.task_ctx())
                        .await?;
                let out = self.apply_update_to_batches(&scan, &batches, None, &[])?;
                let origin = tf.begin_snapshot.ok_or_else(|| {
                    DuckLakeError::Internal("merge candidate missing begin_snapshot".to_string())
                })?;
                rows_written += out.matched_count as i64;
                per_source.push((out.updated_batches, origin));
                sources.push(CompactionSourceFile {
                    data_file_id: tf.data_file_id,
                    delete_file_id: None,
                });
                files_processed += 1;
            }

            // A group spanning >1 origin snapshot is a partial file: embed the
            // per-row snapshot column, record the max origin as partial_max, and
            // set begin_snapshot to the MIN origin so historical reads back to
            // that point see it (row-filtered by origin). The sources are then
            // redundant for every snapshot, so the commit removes + schedules
            // them. A single-origin group needs no per-row column (all rows share
            // one origin), and begins at that origin.
            let origins: HashSet<i64> = per_source.iter().map(|(_, o)| *o).collect();
            let partial = origins.len() > 1;
            let min_origin = origins.iter().copied().min();
            let partial_max = if partial {
                origins.iter().copied().max()
            } else {
                None
            };

            let mut merged: Vec<RecordBatch> = Vec::new();
            for (batches, origin) in per_source {
                for b in batches {
                    if b.num_rows() == 0 {
                        continue;
                    }
                    merged.push(if partial {
                        append_snapshot_column(&b, origin)?
                    } else {
                        b
                    });
                }
            }
            if merged.is_empty() {
                continue;
            }
            let file = table_writer
                .write_compacted_file(
                    schema_name,
                    self.table_name(),
                    physical_schema.as_ref(),
                    &column_ids,
                    &merged,
                    partial,
                )
                .await?;
            outputs.push(CompactionOutputFile {
                file,
                partial_max,
                begin_snapshot: min_origin,
            });
        }

        if sources.is_empty() {
            return Ok(CompactionResult::empty());
        }
        writer.commit_compaction(
            self.table_id(),
            self.base_snapshot(),
            &sources,
            &outputs,
            SourceRetirement::Remove,
        )?;
        Ok(CompactionResult {
            files_processed,
            files_created: outputs.len(),
            rows_written,
        })
    }

    /// Rewrite data files whose deleted fraction is at least
    /// `opts.delete_threshold`, dropping their deleted rows, in ONE snapshot.
    ///
    /// For each live file with a delete file masking at least that fraction of
    /// its rows, the file's LIVE rows are read (delete-aware) and written to a
    /// new file that preserves each row's original rowid; the old data file AND
    /// its delete file are retired and scheduled for deletion. A file whose rows
    /// are entirely deleted is retired with no replacement.
    ///
    /// Returns no-op metrics (and commits no snapshot) when no file exceeds the
    /// threshold. Errors if the table is read-only or `delete_threshold` is
    /// outside `[0.0, 1.0]`.
    pub async fn rewrite_data_files(
        &self,
        state: &dyn Session,
        opts: RewriteOptions,
    ) -> Result<CompactionResult> {
        if !(0.0..=1.0).contains(&opts.delete_threshold) {
            return Err(DuckLakeError::InvalidConfig(format!(
                "rewrite_data_files: delete_threshold must be in [0.0, 1.0], got {}",
                opts.delete_threshold
            )));
        }
        let writer = self.writer().ok_or_else(|| {
            DuckLakeError::InvalidConfig(
                "rewrite_data_files: table is read-only; open the catalog with a writer"
                    .to_string(),
            )
        })?;
        let schema_name = self.schema_name().ok_or_else(|| {
            DuckLakeError::Internal("writable table has no schema name".to_string())
        })?;

        let object_store = state
            .runtime_env()
            .object_store(self.object_store_url().as_ref())?;
        let table_writer = DuckLakeTableWriter::new(Arc::clone(writer), object_store)?;
        let column_ids = self.column_ids();
        let physical_schema = self.physical_schema();

        let mut sources: Vec<CompactionSourceFile> = Vec::new();
        let mut outputs: Vec<CompactionOutputFile> = Vec::new();
        let mut files_processed = 0usize;
        let mut rows_written = 0i64;

        let table_files = self.files()?;
        for tf in &table_files {
            let record_count = tf.max_row_count.unwrap_or(0);
            let delete_count = tf.delete_count.unwrap_or(0);
            // Only files with a live delete file masking >= threshold of the rows.
            if tf.delete_file_id.is_none() || record_count <= 0 {
                continue;
            }
            let ratio = delete_count as f64 / record_count as f64;
            if ratio < opts.delete_threshold {
                continue;
            }

            let scan = self.build_update_scan(state, tf).await?;
            let batches =
                datafusion::physical_plan::collect(Arc::clone(&scan.scan), state.task_ctx())
                    .await?;
            let out = self.apply_update_to_batches(&scan, &batches, None, &[])?;

            files_processed += 1;
            sources.push(CompactionSourceFile {
                data_file_id: tf.data_file_id,
                delete_file_id: tf.delete_file_id,
            });

            let live_rows = out.matched_count;
            if live_rows > 0 {
                let file = table_writer
                    .write_compacted_file(
                        schema_name,
                        self.table_name(),
                        physical_schema.as_ref(),
                        &column_ids,
                        &out.updated_batches,
                        false,
                    )
                    .await?;
                rows_written += live_rows as i64;
                // A rewrite output holds only currently-live rows and begins at
                // the compaction snapshot (begin_snapshot = None); its
                // pre-compaction history is served by the retained sources.
                outputs.push(CompactionOutputFile {
                    file,
                    partial_max: None,
                    begin_snapshot: None,
                });
            }
        }

        if sources.is_empty() {
            return Ok(CompactionResult::empty());
        }
        // Retire (do not remove) the sources: they still serve time travel to
        // pre-rewrite snapshots until their snapshots are expired.
        writer.commit_compaction(
            self.table_id(),
            self.base_snapshot(),
            &sources,
            &outputs,
            SourceRetirement::Retire,
        )?;
        Ok(CompactionResult {
            files_processed,
            files_created: outputs.len(),
            rows_written,
        })
    }
}
