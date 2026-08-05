//! High-level table writer for DuckLake catalogs.

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::sync::Arc;

use arrow::datatypes::{Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use datafusion::error::DataFusionError;
use datafusion::execution::SendableRecordBatchStream;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use futures::StreamExt;
use object_store::ObjectStore;
use object_store::buffered::BufWriter as ObjectBufWriter;
use object_store::path::Path as ObjectPath;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use tempfile::NamedTempFile;
use tokio::io::AsyncWriteExt;
use uuid::Uuid;

use crate::Result;
use crate::metadata_writer::{
    ColumnDef, DataFileInfo, DeleteFileEntry, DeleteFileInfo, MetadataWriter,
    SnapshotCommitMetadata, WriteMode, WriteResult, validate_delete_entries,
};
use crate::path_resolver::join_paths;
use crate::row_id::{embedded_rowid_field, embedded_snapshot_id_field};
use crate::table::delete_file_schema;

// The partition-group shape is shared with the split logic in `partition`.
pub use crate::partition::PartitionGroup;

/// Default cap on parquet files a partitioned streaming write keeps open at once,
/// matching DuckDB's `partition_write_max_open_files`. A streaming write cannot
/// know how many partitions its rows will touch, so it holds one open writer per
/// partition seen and, on reaching this cap, finalizes the least-recently-opened
/// file to make room (that partition simply gets another file if more of its rows
/// arrive). Without a cap, a high-cardinality partition key would exhaust file
/// descriptors and memory.
pub const DEFAULT_MAX_OPEN_PARTITIONS: usize = 100;

/// Floor on the target data file size, matching official DuckLake's
/// `MINIMUM_WRITE_FILE_SIZE` (`ducklake_insert.cpp`), which clamps with
/// `MaxValue<idx_t>(target_file_size, 4096)`. A smaller request would roll a new file
/// per batch and produce a file per row group.
pub const MINIMUM_TARGET_FILE_SIZE: usize = 4096;

/// Default target data file size: 512 MiB, matching official DuckLake's
/// `target_file_size` default (`1 << 29`). A write rolls over to a new file once
/// it reaches this size, so no single write can produce a file too large for
/// later compaction to reorganize (DuckLake compaction merges, never splits).
pub const DEFAULT_TARGET_FILE_SIZE: usize = 1 << 29;

/// Write-layout options carried from the catalog down to the insert path, so a
/// SQL `INSERT` builds its [`DuckLakeTableWriter`] with the same compression,
/// row-group caps, and file-rollover target the embedding engine configured.
/// A `None` field leaves the writer's default for that setting (uncompressed,
/// parquet-default row groups, [`DEFAULT_TARGET_FILE_SIZE`] rollover).
#[derive(Debug, Clone, Default)]
pub struct DuckLakeWriteOptions {
    /// Parquet compression codec; `None` = uncompressed.
    pub compression: Option<Compression>,
    /// Max rows per row group; `None` = parquet default.
    pub max_row_group_rows: Option<usize>,
    /// Max uncompressed bytes per row group; `None` = parquet default.
    pub max_row_group_bytes: Option<usize>,
    /// Target data file size for rollover (approx encoded bytes); `None` leaves
    /// the writer default ([`DEFAULT_TARGET_FILE_SIZE`]).
    pub target_file_size: Option<usize>,
    /// Max parquet files a partitioned streaming write keeps open at once; `None`
    /// leaves the writer default ([`DEFAULT_MAX_OPEN_PARTITIONS`]).
    pub max_open_partitions: Option<usize>,
}

/// Options shared by streaming and partitioned table writes.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TableWriteOptions {
    /// Metadata recorded in the snapshot change row.
    pub commit_metadata: SnapshotCommitMetadata,
    /// Catalog snapshot against which this write read its input.
    ///
    /// The commit fails if the target table's data-file generation changed
    /// after this snapshot. Commits to other tables do not cause a conflict.
    pub expected_base_snapshot_id: Option<i64>,
}

impl TableWriteOptions {
    /// Creates default write options.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            commit_metadata: SnapshotCommitMetadata::new(),
            expected_base_snapshot_id: None,
        }
    }

    /// Attaches metadata to the committed snapshot.
    #[must_use]
    pub fn with_commit_metadata(mut self, commit_metadata: SnapshotCommitMetadata) -> Self {
        self.commit_metadata = commit_metadata;
        self
    }

    /// Requires the write to commit against the target table's data-file
    /// generation visible at `snapshot_id`.
    #[must_use]
    pub const fn with_expected_base_snapshot_id(mut self, snapshot_id: i64) -> Self {
        self.expected_base_snapshot_id = Some(snapshot_id);
        self
    }
}

/// High-level writer for DuckLake tables.
#[derive(Debug)]
pub struct DuckLakeTableWriter {
    metadata: Arc<dyn MetadataWriter>,
    object_store: Arc<dyn ObjectStore>,
    /// The key path portion of the data_path (e.g., "/prefix/data/")
    base_key_path: String,
    /// Compression codec for written data files. Defaults to `UNCOMPRESSED`;
    /// override via [`DuckLakeTableWriter::with_compression`] to trade write
    /// CPU for ~2x smaller files (e.g. `LZ4`, `SNAPPY`, `ZSTD`).
    compression: Compression,
    /// Optional max rows per parquet row group. `None` leaves the parquet
    /// default. Set via [`DuckLakeTableWriter::with_max_row_group_rows`].
    max_row_group_rows: Option<usize>,
    /// Optional max *uncompressed* bytes per parquet row group. `None` leaves
    /// the parquet default (rows-only). A reader decodes a whole row group at
    /// once, so a byte cap bounds reader memory for wide schemas (e.g. large
    /// vector columns). Set via [`DuckLakeTableWriter::with_max_row_group_bytes`].
    max_row_group_bytes: Option<usize>,
    /// Target data file size in approximate encoded bytes. A write rolls over to a
    /// new file once the current file's estimated encoded size reaches this, so a
    /// large write produces several files instead of one. Paired with a sort order,
    /// each file then covers a contiguous, non-overlapping value range — which is
    /// what lets DuckLake skip whole files by their min/max at query time. Defaults
    /// to [`DEFAULT_TARGET_FILE_SIZE`] (matching official DuckLake); override via
    /// [`DuckLakeTableWriter::with_target_file_size`].
    target_file_size: usize,
    /// Max parquet files a partitioned streaming write keeps open concurrently.
    /// Defaults to [`DEFAULT_MAX_OPEN_PARTITIONS`]; override via
    /// [`DuckLakeTableWriter::with_max_open_partitions`].
    max_open_partitions: usize,
}

impl DuckLakeTableWriter {
    pub fn new(
        metadata: Arc<dyn MetadataWriter>,
        object_store: Arc<dyn ObjectStore>,
    ) -> Result<Self> {
        let data_path_str = metadata.get_data_path()?;
        let (_, key_path) = crate::path_resolver::parse_object_store_url(&data_path_str)?;

        Ok(Self {
            metadata,
            object_store,
            base_key_path: key_path,
            compression: Compression::UNCOMPRESSED,
            max_row_group_rows: None,
            max_row_group_bytes: None,
            target_file_size: DEFAULT_TARGET_FILE_SIZE,
            max_open_partitions: DEFAULT_MAX_OPEN_PARTITIONS,
        })
    }

    /// Override the parquet compression codec used for written data files.
    /// Defaults to [`Compression::UNCOMPRESSED`].
    pub fn with_compression(mut self, compression: Compression) -> Self {
        self.compression = compression;
        self
    }

    /// Cap the number of rows per parquet row group. Leaves the parquet
    /// default when unset.
    pub fn with_max_row_group_rows(mut self, rows: usize) -> Self {
        self.max_row_group_rows = Some(rows);
        self
    }

    /// Cap the *uncompressed* bytes per parquet row group, flushing the row
    /// group once it is reached. Because a parquet reader must decode an entire
    /// row group into memory at once, this bounds reader memory for wide
    /// schemas (e.g. large `List`/`FixedSizeList` vector columns) that would
    /// otherwise build multi-GiB row groups at the rows-only default. Leaves
    /// the parquet default when unset.
    pub fn with_max_row_group_bytes(mut self, bytes: usize) -> Self {
        self.max_row_group_bytes = Some(bytes);
        self
    }

    /// Override the target data file size (approx encoded bytes) at which a write
    /// rolls over to a new file, estimated from the writer's flushed + in-progress
    /// size and checked at batch boundaries. Combined with a sort order, each file
    /// holds a contiguous value range with a tight min/max, enabling file-level
    /// pruning. Defaults to [`DEFAULT_TARGET_FILE_SIZE`].
    pub fn with_target_file_size(mut self, bytes: usize) -> Self {
        self.target_file_size = bytes.max(MINIMUM_TARGET_FILE_SIZE);
        self
    }

    /// The target file size at which writes roll over (see
    /// [`with_target_file_size`](Self::with_target_file_size)).
    pub fn target_file_size(&self) -> usize {
        self.target_file_size
    }

    /// Cap the number of parquet files a partitioned streaming write keeps open at
    /// once. Defaults to [`DEFAULT_MAX_OPEN_PARTITIONS`]. Clamped to at least 1.
    pub fn with_max_open_partitions(mut self, files: usize) -> Self {
        self.max_open_partitions = files.max(1);
        self
    }

    /// Apply a [`DuckLakeWriteOptions`] set (compression, row-group caps, rollover
    /// target, open-partition cap). Each field overrides the corresponding setting
    /// only when present.
    pub fn with_options(mut self, options: &DuckLakeWriteOptions) -> Self {
        if let Some(compression) = options.compression {
            self.compression = compression;
        }
        if let Some(rows) = options.max_row_group_rows {
            self.max_row_group_rows = Some(rows);
        }
        if let Some(bytes) = options.max_row_group_bytes {
            self.max_row_group_bytes = Some(bytes);
        }
        if let Some(bytes) = options.target_file_size {
            self.target_file_size = bytes;
        }
        if let Some(files) = options.max_open_partitions {
            self.max_open_partitions = files.max(1);
        }
        self
    }

    /// Build the parquet [`WriterProperties`] shared by every write path from this
    /// writer's configured compression and row-group caps.
    fn build_writer_props(&self) -> WriterProperties {
        let mut builder = WriterProperties::builder()
            .set_writer_version(parquet::file::properties::WriterVersion::PARQUET_2_0)
            .set_compression(self.compression);
        if let Some(rows) = self.max_row_group_rows {
            builder = builder.set_max_row_group_row_count(Some(rows));
        }
        if let Some(bytes) = self.max_row_group_bytes {
            builder = builder.set_max_row_group_bytes(Some(bytes));
        }
        builder.build()
    }

    /// Begin a streaming write session.
    /// If mode is `WriteMode::Replace`, ends existing files.
    ///
    /// **Partition-aware**: when the target table is partitioned, the session splits
    /// each batch by the transformed partition key and keeps one open parquet per
    /// partition (up to
    /// [`max_open_partitions`](Self::with_max_open_partitions), finalizing the
    /// least-recently-opened file beyond that), rolling each over at
    /// [`target_file_size`](Self::with_target_file_size). Every file produced is
    /// committed in ONE snapshot by [`TableWriteSession::finish`], so a partitioned
    /// streaming write stays as atomic as an unpartitioned one.
    ///
    /// **Sort order is the caller's responsibility here** — rows are written in
    /// arrival order. Unlike [`Self::write_rows`], a streaming session cannot apply
    /// the table's sort order itself: the useful sort is a GLOBAL one (official
    /// DuckLake achieves it with a blocking `PhysicalOrder` above the insert plan),
    /// and doing that here would mean buffering the entire write — the very thing
    /// streaming exists to avoid, and unbounded across
    /// `max_open_partitions` open files. Sorting only *within* each file would buy
    /// nothing at the file level either, since a file's min/max is the min/max of its
    /// rows however they are ordered; only its row-group bounds would tighten.
    ///
    /// So to get the file-skipping benefit of a sort order from this path, feed
    /// batches already in sort order, or use [`Self::write_rows`] when the write fits
    /// in memory. Writing unsorted costs pruning quality only, never correctness.
    ///
    /// **Rolls by default.** A new data file is started once the current one exceeds
    /// [`target_file_size`](Self::with_target_file_size), and
    /// [`TableWriteSession::finish`] commits them all in one snapshot. Official
    /// DuckLake rotates on every write (`result.rotate = true` in
    /// `ducklake_insert.cpp`), and a single unbounded file could never be reorganized
    /// afterwards — DuckLake compaction merges but never splits.
    ///
    /// This is also the right default for a session finished with
    /// [`TableWriteSession::finish_with_deletes`]: that commit registers every
    /// appended file, however many the session rolled or partitioned into, in the
    /// same snapshot as the deletes.
    pub fn begin_write(
        &self,
        schema_name: &str,
        table_name: &str,
        arrow_schema: &Schema,
        mode: WriteMode,
    ) -> Result<TableWriteSession> {
        // Multicatalog backends share one physical `data_path`, so without a
        // per-catalog segment two catalogs writing the same (schema, table)
        // would dump files into the same directory. Prepend `cat_{id}` to keep
        // them physically isolated. Single-catalog backends report `None` and
        // skip the segment, preserving the historical `{schema}/{table}/…`
        // layout. `cat_` prefix + numeric id is rename-safe and needs no
        // sanitisation.
        let scoped_base = match self.metadata.catalog_id() {
            Some(id) => join_paths(&self.base_key_path, &format!("cat_{id}"))?,
            None => self.base_key_path.clone(),
        };
        let table_key = join_paths(&join_paths(&scoped_base, schema_name)?, table_name)?;
        let file_name = format!("{}.parquet", Uuid::new_v4());
        self.begin_write_internal(
            schema_name,
            table_name,
            arrow_schema,
            table_key,
            file_name.clone(),
            file_name,
            true,
            false,
            mode,
            StreamPartitionMode::Split,
            true,
        )
    }

    /// Begin a streaming write session that writes ONE data file, however large the
    /// input.
    ///
    /// A single file is not reorganizable later — DuckLake compaction merges but never
    /// splits — so prefer [`Self::begin_write`] unless the caller genuinely needs one
    /// output object (for example to address the whole write by path).
    pub fn begin_write_single_file(
        &self,
        schema_name: &str,
        table_name: &str,
        arrow_schema: &Schema,
        mode: WriteMode,
    ) -> Result<TableWriteSession> {
        let scoped_base = match self.metadata.catalog_id() {
            Some(id) => join_paths(&self.base_key_path, &format!("cat_{id}"))?,
            None => self.base_key_path.clone(),
        };
        let table_key = join_paths(&join_paths(&scoped_base, schema_name)?, table_name)?;
        let file_name = format!("{}.parquet", Uuid::new_v4());
        self.begin_write_internal(
            schema_name,
            table_name,
            arrow_schema,
            table_key,
            file_name.clone(),
            file_name,
            true,
            false,
            mode,
            StreamPartitionMode::Split,
            false,
        )
    }

    /// Begin a streaming write session whose parquet output carries an extra
    /// embedded row-id column (field-id [`ROW_ID_PARQUET_FIELD_ID`]) appended
    /// after the table's data columns, so rewritten rows preserve their DuckLake
    /// row lineage across the file rewrite (the commit behind `UPDATE` /
    /// compaction).
    ///
    /// `arrow_schema` describes ONLY the table's data columns (no rowid), exactly
    /// as for [`begin_write`](Self::begin_write); the embedded column is added to
    /// the parquet schema here and is NOT registered as a catalog column. Batches
    /// passed to [`TableWriteSession::write_batch`] must therefore have the data
    /// columns in order followed by a trailing `Int64` rowid column holding each
    /// row's original rowid. A later read detects the embedded column by its
    /// field-id and serves those rowids inline instead of synthesizing
    /// `row_id_start + position`.
    ///
    /// On a partitioned table, rows route through the standard partition sink: each
    /// rewritten row's partition is re-derived from its OWN (possibly updated) key
    /// values, so a row whose partition-key value changed lands in its NEW partition
    /// rather than inheriting the source file's. The embedded rowid column travels
    /// with the row into whichever partition file it lands in, so lineage survives a
    /// rewrite that spreads rows over several files, and
    /// [`TableWriteSession::finish_with_deletes`] commits all of them in the one
    /// snapshot that carries the deletes.
    ///
    /// [`ROW_ID_PARQUET_FIELD_ID`]: crate::row_id::ROW_ID_PARQUET_FIELD_ID
    pub fn begin_write_with_embedded_rowid(
        &self,
        schema_name: &str,
        table_name: &str,
        arrow_schema: &Schema,
        mode: WriteMode,
    ) -> Result<TableWriteSession> {
        let scoped_base = match self.metadata.catalog_id() {
            Some(id) => join_paths(&self.base_key_path, &format!("cat_{id}"))?,
            None => self.base_key_path.clone(),
        };
        let table_key = join_paths(&join_paths(&scoped_base, schema_name)?, table_name)?;
        let file_name = format!("{}.parquet", Uuid::new_v4());
        self.begin_write_internal(
            schema_name,
            table_name,
            arrow_schema,
            table_key,
            file_name.clone(),
            file_name,
            true,
            true,
            mode,
            StreamPartitionMode::Split,
            false,
        )
    }

    /// Begin a streaming write session with a custom file path (registered as absolute).
    pub fn begin_write_to_path(
        &self,
        schema_name: &str,
        table_name: &str,
        arrow_schema: &Schema,
        file_dir: &str,
        file_name: String,
        mode: WriteMode,
    ) -> Result<TableWriteSession> {
        let full_path = join_paths(file_dir, &file_name)?;
        self.begin_write_internal(
            schema_name,
            table_name,
            arrow_schema,
            file_dir.to_string(),
            file_name,
            full_path,
            false,
            false,
            mode,
            StreamPartitionMode::Reject {
                entry_point: "begin_write_to_path",
            },
            false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn begin_write_internal(
        &self,
        schema_name: &str,
        table_name: &str,
        arrow_schema: &Schema,
        file_dir: String,
        file_name: String,
        catalog_path: String,
        path_is_relative: bool,
        embed_rowid: bool,
        mode: WriteMode,
        partition_mode: StreamPartitionMode,
        roll: bool,
    ) -> Result<TableWriteSession> {
        let columns = arrow_schema_to_column_defs(arrow_schema)?;
        let setup =
            self.metadata
                .begin_write_transaction(schema_name, table_name, &columns, mode)?;
        // Data columns carry their catalog field-ids. When embedding row lineage,
        // append the reserved-field-id rowid column AFTER them; it is a parquet-only
        // column (not a catalog column), so it is absent from `columns`/`column_ids`
        // and the metadata commit never sees it.
        let schema_with_ids = {
            let mut schema = build_schema_with_field_ids(arrow_schema, &setup.column_ids);
            if embed_rowid {
                let mut fields: Vec<Field> =
                    schema.fields().iter().map(|f| f.as_ref().clone()).collect();
                fields.push(embedded_rowid_field());
                schema = Schema::new_with_metadata(fields, schema.metadata().clone());
            }
            Arc::new(schema)
        };

        let object_path_str = join_paths(&file_dir, &file_name)?;
        // Strip leading slash for object_store Path (it expects relative keys)
        let object_path = ObjectPath::from(object_path_str.trim_start_matches('/'));

        // Apply caller-configured row-group caps. The ArrowWriter enforces both
        // natively (flushing the row group when either is hit). The byte cap
        // matters for wide schemas: a parquet reader decodes a whole row group
        // at once, so an uncapped large vector column builds multi-GiB row
        // groups that OOM readers. Both default to the parquet default (unset).
        let mut props_builder = WriterProperties::builder()
            .set_writer_version(parquet::file::properties::WriterVersion::PARQUET_2_0)
            .set_compression(self.compression);
        if let Some(rows) = self.max_row_group_rows {
            props_builder = props_builder.set_max_row_group_row_count(Some(rows));
        }
        if let Some(bytes) = self.max_row_group_bytes {
            props_builder = props_builder.set_max_row_group_bytes(Some(bytes));
        }
        let props = props_builder.build();
        // Stream the parquet to a local staging file rather than an in-memory
        // buffer: a multi-GB table would otherwise be held whole in RAM and,
        // worse, uploaded as a single PUT (object stores cap a single PUT at
        // 5 GiB). `finish()` streams this file out via a multipart upload.
        let temp = NamedTempFile::new()?;
        let staging = std::io::BufWriter::new(temp.reopen()?);
        let writer = ArrowWriter::try_new(staging, schema_with_ids.clone(), Some(props))?;

        // A partitioned target routes rows through a per-partition sink. The
        // single-file writer above is still created: with zero rows the sink
        // produces no file, and a Replace then needs that 0-row marker to retire the
        // prior generation.
        let partition_sink =
            match self.resolve_partition(setup.table_id, &setup.column_ids, arrow_schema)? {
                None => None,
                Some(spec) => match partition_mode {
                    StreamPartitionMode::Reject {
                        entry_point,
                    } => {
                        return Err(crate::error::DuckLakeError::Unsupported(format!(
                            "{entry_point} does not support a partitioned table: it writes to one \
                         caller-determined file, but the table's partition spec requires rows \
                         to be split across one file per partition"
                        )));
                    },
                    StreamPartitionMode::Split => {
                        let scoped_base = match self.metadata.catalog_id() {
                            Some(id) => join_paths(&self.base_key_path, &format!("cat_{id}"))?,
                            None => self.base_key_path.clone(),
                        };
                        let table_key =
                            join_paths(&join_paths(&scoped_base, schema_name)?, table_name)?;
                        Some(PartitionSink {
                            key_names: spec.key_names(),
                            spec,
                            table_key,
                            schema_with_ids: schema_with_ids.clone(),
                            column_ids: setup.column_ids.clone(),
                            props: self.build_writer_props(),
                            target_file_size: self.target_file_size,
                            max_open: self.max_open_partitions,
                            open: Vec::new(),
                            staged: Vec::new(),
                        })
                    },
                },
            };

        // A partitioned target already rolls inside its per-partition sink, so a
        // second roller would be redundant (and would double-write).
        let roller = if roll && partition_sink.is_none() {
            let scoped_base = match self.metadata.catalog_id() {
                Some(id) => join_paths(&self.base_key_path, &format!("cat_{id}"))?,
                None => self.base_key_path.clone(),
            };
            let table_key = join_paths(&join_paths(&scoped_base, schema_name)?, table_name)?;
            Some(RollingFileWriter::new(
                table_key,
                None,
                schema_with_ids.clone(),
                setup.column_ids.len(),
                self.build_writer_props(),
                self.target_file_size,
                // Keep `TableWriteSession::file_path` accurate for the first file.
                Some(catalog_path.clone()),
            ))
        } else {
            None
        };

        Ok(TableWriteSession {
            metadata: Arc::clone(&self.metadata),
            object_store: Arc::clone(&self.object_store),
            object_path,
            schema_name: schema_name.to_string(),
            table_name: table_name.to_string(),
            snapshot_id: setup.snapshot_id,
            base_snapshot_id: setup.base_snapshot_id,
            expected_base_snapshot_id: None,
            table_id: setup.table_id,
            columns,
            column_ids: setup.column_ids,
            schema_with_ids,
            writer: Some(writer),
            temp: Some(temp),
            catalog_path,
            path_is_relative,
            mode,
            row_count: 0,
            nan_flags: Vec::new(),
            partition_sink,
            roller,
            rolled: Vec::new(),
            commit_metadata: SnapshotCommitMetadata::default(),
        })
    }

    /// Write batches to a table, replacing any existing data.
    ///
    /// Goes through [`Self::write_rows`], so a partitioned target is split into one
    /// file per partition and large inputs roll over by
    /// [`target_file_size`](Self::with_target_file_size).
    pub async fn write_table(
        &self,
        schema_name: &str,
        table_name: &str,
        batches: &[RecordBatch],
    ) -> Result<WriteResult> {
        self.write_all(schema_name, table_name, batches, WriteMode::Replace)
            .await
    }

    /// Write batches to a table, appending to existing data.
    ///
    /// Goes through [`Self::write_rows`], so a partitioned target is split into one
    /// file per partition and large inputs roll over by
    /// [`target_file_size`](Self::with_target_file_size).
    pub async fn append_table(
        &self,
        schema_name: &str,
        table_name: &str,
        batches: &[RecordBatch],
    ) -> Result<WriteResult> {
        self.write_all(schema_name, table_name, batches, WriteMode::Append)
            .await
    }

    /// Shared body of [`Self::write_table`] / [`Self::append_table`].
    ///
    /// A row-bearing input goes through the layout-aware [`Self::write_rows`]. An
    /// input of only empty batches keeps the single-file session path: `write_rows`
    /// produces no file at all, which for `Replace` would skip the truncation of the
    /// prior generation, whereas the session registers the 0-row file that carries it.
    async fn write_all(
        &self,
        schema_name: &str,
        table_name: &str,
        batches: &[RecordBatch],
        mode: WriteMode,
    ) -> Result<WriteResult> {
        if batches.is_empty() {
            return Err(crate::error::DuckLakeError::InvalidConfig(
                "No batches to write".to_string(),
            ));
        }

        let arrow_schema = batches[0].schema();
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        if total_rows > 0 {
            return self
                .write_rows(schema_name, table_name, &arrow_schema, mode, batches)
                .await;
        }

        let mut session = self.begin_write(schema_name, table_name, &arrow_schema, mode)?;
        for batch in batches {
            session.write_batch(batch)?;
        }
        session.finish().await
    }

    /// Write a positional `(file_path, pos)` delete parquet, upload it, and
    /// return the [`DeleteFileInfo`] to register via
    /// [`MetadataWriter::set_delete_file`].
    ///
    /// `positions` is the CUMULATIVE set of still-deleted physical row positions
    /// for `data_file_path`: the engine keeps at most one live delete file per
    /// data file, so each write carries the full set (the prior file is retired
    /// on commit). The delete file lands beside the data files it masks — the
    /// same `cat_{id}/{schema}/{table}/` layout as [`Self::begin_write`] — and is
    /// registered relative to the table, so the reader resolves it exactly like a
    /// data file. Readers key deletes off `pos`; `file_path` is recorded for
    /// provenance.
    pub async fn write_delete_file(
        &self,
        schema_name: &str,
        table_name: &str,
        data_file_path: &str,
        positions: &[i64],
    ) -> Result<DeleteFileInfo> {
        use arrow::array::{Int64Array, StringArray};

        let scoped_base = match self.metadata.catalog_id() {
            Some(id) => join_paths(&self.base_key_path, &format!("cat_{id}"))?,
            None => self.base_key_path.clone(),
        };
        let table_key = join_paths(&join_paths(&scoped_base, schema_name)?, table_name)?;
        let file_name = format!("{}.parquet", Uuid::new_v4());
        let object_path_str = join_paths(&table_key, &file_name)?;
        // Strip leading slash for object_store Path (it expects relative keys).
        let object_path = ObjectPath::from(object_path_str.trim_start_matches('/'));

        let schema = delete_file_schema();
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec![data_file_path; positions.len()])),
                Arc::new(Int64Array::from(positions.to_vec())),
            ],
        )?;

        // Stream to a local staging file, then multipart-upload it — the same
        // bounded-memory path `finish()` uses for data files.
        let props = WriterProperties::builder()
            .set_writer_version(parquet::file::properties::WriterVersion::PARQUET_2_0)
            .set_compression(self.compression)
            .build();
        let temp = NamedTempFile::new()?;
        let staging = std::io::BufWriter::new(temp.reopen()?);
        let mut writer = ArrowWriter::try_new(staging, schema, Some(props))?;
        writer.write(&batch)?;
        let staged = writer.into_inner()?;
        let mut file = staged
            .into_inner()
            .map_err(|e| crate::error::DuckLakeError::Io(e.into_error()))?;
        let file_size = file.metadata()?.len() as i64;
        let footer_size = read_footer_size(&mut file)?;

        let local = tokio::fs::File::open(temp.path()).await?;
        let mut reader = tokio::io::BufReader::new(local);
        let mut upload = ObjectBufWriter::new(Arc::clone(&self.object_store), object_path);
        if let Err(e) = stream_to_upload(&mut reader, &mut upload).await {
            let _ = upload.abort().await;
            return Err(e.into());
        }

        // Registered relative to the table path (like data files); the reader
        // resolves it against the same table data dir.
        Ok(
            DeleteFileInfo::new(file_name, file_size, positions.len() as i64)
                .with_footer_size(footer_size),
        )
    }

    /// Write ONE compacted parquet file to the table's data directory and return
    /// its [`DataFileInfo`], performing NO catalog work — the compaction commit
    /// ([`MetadataWriter::commit_compaction`]) registers the file and retires the
    /// sources atomically.
    ///
    /// The output embeds each row's original rowid (field-id
    /// [`ROW_ID_PARQUET_FIELD_ID`](crate::row_id::ROW_ID_PARQUET_FIELD_ID)) so
    /// row lineage survives the rewrite, exactly like the `UPDATE` writer; when
    /// `embed_snapshot_id` is set it ALSO embeds the per-row
    /// `_ducklake_internal_snapshot_id` column (field-id
    /// [`SNAPSHOT_ID_PARQUET_FIELD_ID`](crate::row_id::SNAPSHOT_ID_PARQUET_FIELD_ID))
    /// that marks a merged partial file.
    ///
    /// `data_schema` describes ONLY the table's data columns (catalog types, no
    /// rowid/snapshot); `data_column_ids` are their catalog `column_id`s (baked
    /// in as parquet field-ids so a read maps them back). Each batch in `batches`
    /// must have the data columns in order, then a trailing `Int64` rowid column,
    /// and — when `embed_snapshot_id` — a further trailing `Int64` snapshot-id
    /// column. Streams to a local staging file and multipart-uploads it, so peak
    /// memory stays bounded regardless of file size.
    #[allow(clippy::too_many_arguments)]
    pub async fn write_compacted_file(
        &self,
        schema_name: &str,
        table_name: &str,
        data_schema: &Schema,
        data_column_ids: &[i64],
        batches: &[RecordBatch],
        embed_snapshot_id: bool,
        partition_subpath: Option<&str>,
    ) -> Result<DataFileInfo> {
        let stream_schema = batches.first().map_or_else(
            || {
                let mut fields: Vec<Field> = data_schema
                    .fields()
                    .iter()
                    .map(|field| field.as_ref().clone())
                    .collect();
                fields.push(embedded_rowid_field());
                if embed_snapshot_id {
                    fields.push(embedded_snapshot_id_field());
                }
                Arc::new(Schema::new(fields))
            },
            RecordBatch::schema,
        );
        let batches = batches.to_vec();
        let stream =
            futures::stream::iter(batches.into_iter().map(Ok::<RecordBatch, DataFusionError>));
        let stream = Box::pin(RecordBatchStreamAdapter::new(stream_schema, stream));
        self.write_compacted_file_stream(
            schema_name,
            table_name,
            data_schema,
            data_column_ids,
            stream,
            embed_snapshot_id,
            partition_subpath,
        )
        .await
    }

    /// Write one compacted parquet file from a record-batch stream.
    ///
    /// This lets compaction feed a spilling DataFusion sort directly into
    /// Parquet without retaining the sorted output in memory.
    #[allow(clippy::too_many_arguments)]
    pub async fn write_compacted_file_stream(
        &self,
        schema_name: &str,
        table_name: &str,
        data_schema: &Schema,
        data_column_ids: &[i64],
        mut batches: SendableRecordBatchStream,
        embed_snapshot_id: bool,
        partition_subpath: Option<&str>,
    ) -> Result<DataFileInfo> {
        let scoped_base = match self.metadata.catalog_id() {
            Some(id) => join_paths(&self.base_key_path, &format!("cat_{id}"))?,
            None => self.base_key_path.clone(),
        };
        let table_key = join_paths(&join_paths(&scoped_base, schema_name)?, table_name)?;
        // A compacted file of a partitioned table lands in its partition's Hive
        // directory, like every other write of that partition; the returned
        // `DataFileInfo.path` is relative to the table dir either way.
        let file_name = match partition_subpath {
            Some(prefix) if !prefix.is_empty() => {
                format!("{prefix}/{}.parquet", Uuid::new_v4())
            },
            _ => format!("{}.parquet", Uuid::new_v4()),
        };
        let object_path_str = join_paths(&table_key, &file_name)?;
        let object_path = ObjectPath::from(object_path_str.trim_start_matches('/'));

        // Data columns carry their catalog field-ids; append the reserved-field-id
        // embedded rowid column, and for a merged partial file the snapshot-id
        // column. Neither embedded column is a catalog column.
        let schema_with_ids = {
            let base = build_schema_with_field_ids(data_schema, data_column_ids);
            let mut fields: Vec<Field> = base.fields().iter().map(|f| f.as_ref().clone()).collect();
            fields.push(embedded_rowid_field());
            if embed_snapshot_id {
                fields.push(embedded_snapshot_id_field());
            }
            Arc::new(Schema::new_with_metadata(fields, base.metadata().clone()))
        };

        let mut props_builder = WriterProperties::builder()
            .set_writer_version(parquet::file::properties::WriterVersion::PARQUET_2_0)
            .set_compression(self.compression);
        if let Some(rows) = self.max_row_group_rows {
            props_builder = props_builder.set_max_row_group_row_count(Some(rows));
        }
        if let Some(bytes) = self.max_row_group_bytes {
            props_builder = props_builder.set_max_row_group_bytes(Some(bytes));
        }
        let props = props_builder.build();

        let temp = NamedTempFile::new()?;
        let staging = std::io::BufWriter::new(temp.reopen()?);
        let mut writer = ArrowWriter::try_new(staging, schema_with_ids.clone(), Some(props))?;
        let mut row_count: i64 = 0;
        let mut nan_flags: Vec<Option<bool>> = Vec::new();
        while let Some(batch) = batches.next().await {
            let batch = batch?;
            if batch.num_columns() != schema_with_ids.fields().len() {
                return Err(crate::error::DuckLakeError::InvalidConfig(format!(
                    "write_compacted_file: batch has {} columns, expected {}",
                    batch.num_columns(),
                    schema_with_ids.fields().len()
                )));
            }
            let batch_with_ids =
                RecordBatch::try_new(schema_with_ids.clone(), batch.columns().to_vec())?;
            crate::stats_collect::accumulate_nan_flags(
                &mut nan_flags,
                &batch,
                data_column_ids.len(),
            );
            writer.write(&batch_with_ids)?;
            row_count += batch.num_rows() as i64;
        }
        let staged = writer.into_inner()?;
        let mut file = staged
            .into_inner()
            .map_err(|e| crate::error::DuckLakeError::Io(e.into_error()))?;
        let file_size = file.metadata()?.len() as i64;
        let footer_size = read_footer_size(&mut file)?;

        let local = tokio::fs::File::open(temp.path()).await?;
        let mut reader = tokio::io::BufReader::new(local);
        let mut upload = ObjectBufWriter::new(Arc::clone(&self.object_store), object_path);
        if let Err(e) = stream_to_upload(&mut reader, &mut upload).await {
            let _ = upload.abort().await;
            return Err(e.into());
        }

        // Collect stats for catalog columns only. Track NaN values while
        // consuming the stream because the Parquet footer omits that signal.
        let column_stats = crate::stats_collect::collect_column_stats(
            temp.path(),
            data_column_ids,
            row_count,
            &nan_flags,
        );

        Ok(DataFileInfo::new(file_name, file_size, row_count)
            .with_footer_size(footer_size)
            .with_column_stats(column_stats))
    }

    /// Write a partitioned dataset: each group is written to its own parquet file
    /// (Hive-style `col=value/…` subpath under the table dir), then ALL files are
    /// registered in ONE snapshot via
    /// [`MetadataWriter::register_data_files`].
    ///
    /// `arrow_schema` is the table's data columns (no rowid). `partition_id` is the
    /// active spec generation; `key_names` are the partition-key column names in key
    /// order (used only to build the readable Hive path — the catalog is
    /// authoritative). Each group is `(values, batches)` where `values[i]` is the
    /// DuckDB-canonical partition value (or `None` for NULL) for key `i`, shared by
    /// every row in `batches`. Groups must be non-empty.
    #[allow(clippy::too_many_arguments)]
    pub async fn write_partitioned(
        &self,
        schema_name: &str,
        table_name: &str,
        arrow_schema: &Schema,
        mode: WriteMode,
        partition_id: i64,
        key_names: &[String],
        groups: Vec<PartitionGroup>,
    ) -> Result<WriteResult> {
        self.write_partitioned_with_commit_metadata(
            schema_name,
            table_name,
            arrow_schema,
            mode,
            partition_id,
            key_names,
            groups,
            &SnapshotCommitMetadata::default(),
        )
        .await
    }

    /// Writes a partitioned dataset with metadata attached to its snapshot.
    ///
    /// Returns an error when the configured metadata writer does not support
    /// non-empty commit metadata.
    #[allow(clippy::too_many_arguments)]
    pub async fn write_partitioned_with_commit_metadata(
        &self,
        schema_name: &str,
        table_name: &str,
        arrow_schema: &Schema,
        mode: WriteMode,
        partition_id: i64,
        key_names: &[String],
        groups: Vec<PartitionGroup>,
        commit_metadata: &SnapshotCommitMetadata,
    ) -> Result<WriteResult> {
        let options = TableWriteOptions::new().with_commit_metadata(commit_metadata.clone());
        self.write_partitioned_with_options(
            schema_name,
            table_name,
            arrow_schema,
            mode,
            partition_id,
            key_names,
            groups,
            &options,
        )
        .await
    }

    /// Writes a partitioned dataset with snapshot metadata and an optional
    /// replacement precondition.
    #[allow(clippy::too_many_arguments)]
    pub async fn write_partitioned_with_options(
        &self,
        schema_name: &str,
        table_name: &str,
        arrow_schema: &Schema,
        mode: WriteMode,
        partition_id: i64,
        key_names: &[String],
        groups: Vec<PartitionGroup>,
        options: &TableWriteOptions,
    ) -> Result<WriteResult> {
        if groups.is_empty() {
            return Err(crate::error::DuckLakeError::InvalidConfig(
                "write_partitioned: no partition groups".to_string(),
            ));
        }
        let columns = arrow_schema_to_column_defs(arrow_schema)?;
        let setup =
            self.metadata
                .begin_write_transaction(schema_name, table_name, &columns, mode)?;
        let schema_with_ids =
            Arc::new(build_schema_with_field_ids(arrow_schema, &setup.column_ids));

        let scoped_base = match self.metadata.catalog_id() {
            Some(id) => join_paths(&self.base_key_path, &format!("cat_{id}"))?,
            None => self.base_key_path.clone(),
        };
        let table_key = join_paths(&join_paths(&scoped_base, schema_name)?, table_name)?;

        // Validate the caller's assignment against the live spec BEFORE writing
        // anything, so a bad one costs no uploads. A wrong arity or an unparseable
        // value would otherwise be persisted and then used as an exact pruning
        // bound, silently dropping rows from later reads. (Whether each group's rows
        // really carry its values is the caller's assertion — as in official
        // DuckLake's add_data_files, that cannot be checked without reading data.)
        if let Some(spec) =
            self.resolve_partition(setup.table_id, &setup.column_ids, arrow_schema)?
        {
            if spec.partition_id != partition_id {
                return Err(crate::error::DuckLakeError::Conflict(format!(
                    "write_partitioned targets partition spec {partition_id} but the table's live \
                     generation is {}; re-resolve the spec and retry",
                    spec.partition_id
                )));
            }
            for (values, _) in &groups {
                spec.validate_values(arrow_schema, values)?;
            }
        }

        let file_infos = self
            .write_partition_groups(
                &table_key,
                schema_with_ids,
                &setup.column_ids,
                partition_id,
                key_names,
                &groups,
            )
            .await?;
        let records_written: i64 = file_infos.iter().map(|f| f.record_count).sum();

        if file_infos.is_empty() {
            return Err(crate::error::DuckLakeError::InvalidConfig(
                "write_partitioned: partition groups produced no rows".to_string(),
            ));
        }

        let committed = self.metadata.register_data_files_with_commit_metadata(
            setup.table_id,
            schema_name,
            table_name,
            setup.snapshot_id,
            &file_infos,
            mode,
            options
                .expected_base_snapshot_id
                .unwrap_or(setup.base_snapshot_id),
            &columns,
            &setup.column_ids,
            &options.commit_metadata,
            options.expected_base_snapshot_id,
        )?;

        Ok(WriteResult {
            snapshot_id: committed.snapshot_id,
            table_id: committed.table_id,
            schema_id: committed.schema_id,
            files_written: file_infos.len(),
            records_written,
        })
    }

    /// Write `batches` to a table as ONE OR MORE data files (rolling over by
    /// [`target_file_size`](Self::with_target_file_size)) and commit them in one
    /// snapshot via [`MetadataWriter::register_data_files`].
    ///
    /// **Layout-aware**: the table's live partition AND sort specs are resolved here,
    /// so a partitioned target splits `batches` into one Hive directory per partition
    /// with each file's `partition_id` + values stamped, and a sorted target has its
    /// rows globally sorted before rolling — exactly as SQL `INSERT` does. This is
    /// what keeps a direct caller (no SQL, no
    /// [`crate::metadata_provider::MetadataProvider`]) from writing files the
    /// partition fence would reject, or files whose ranges overlap when the table
    /// declares a sort order.
    ///
    /// The sort is global across `batches`, mirroring official DuckLake's blocking
    /// `PhysicalOrder` above the insert plan, so successive rolled files cover
    /// contiguous, non-overlapping ranges. It therefore holds the whole write in
    /// memory; the streaming [`Self::begin_write`] path cannot do this and leaves sort
    /// order to the caller.
    ///
    /// The spec is read after the write transaction opens, which is the only view a
    /// caller without a `MetadataProvider` has. A spec change racing the commit is
    /// still caught by the fence.
    ///
    /// Callers that DID make a layout decision earlier (the SQL `INSERT` path, which
    /// resolves the spec at plan time and pre-splits) must use
    /// `write_rows_unpartitioned_as_planned` instead, so a spec that went
    /// live in between surfaces as a conflict rather than being silently applied to a
    /// write planned without it.
    ///
    /// `batches` must hold at least one row (the caller keeps the single-file session
    /// path for empty Replace truncation). `arrow_schema` is the table's data columns
    /// (no rowid).
    pub async fn write_rows(
        &self,
        schema_name: &str,
        table_name: &str,
        arrow_schema: &Schema,
        mode: WriteMode,
        batches: &[RecordBatch],
    ) -> Result<WriteResult> {
        self.write_rows_inner(schema_name, table_name, arrow_schema, mode, batches, true)
            .await
    }

    /// Write `batches` as UNPARTITIONED because the caller already established that
    /// the target has no partition spec.
    ///
    /// Used by the SQL `INSERT` path, which resolves the spec at plan time. If a
    /// `SET PARTITIONED BY` went live between planning and this commit, the partition
    /// fence rejects with a conflict and the caller retries against the new spec —
    /// deliberately, rather than re-laying-out the rows under a spec the plan never
    /// saw.
    pub(crate) async fn write_rows_unpartitioned_as_planned(
        &self,
        schema_name: &str,
        table_name: &str,
        arrow_schema: &Schema,
        mode: WriteMode,
        batches: &[RecordBatch],
    ) -> Result<WriteResult> {
        self.write_rows_inner(schema_name, table_name, arrow_schema, mode, batches, false)
            .await
    }

    /// Shared body of [`Self::write_rows`] and
    /// `write_rows_unpartitioned_as_planned`. `resolve_layout` selects
    /// whether the table's live partition spec drives the layout, or the caller's
    /// earlier "unpartitioned" determination stands (and the fence adjudicates).
    async fn write_rows_inner(
        &self,
        schema_name: &str,
        table_name: &str,
        arrow_schema: &Schema,
        mode: WriteMode,
        batches: &[RecordBatch],
        resolve_layout: bool,
    ) -> Result<WriteResult> {
        let columns = arrow_schema_to_column_defs(arrow_schema)?;
        let setup =
            self.metadata
                .begin_write_transaction(schema_name, table_name, &columns, mode)?;
        let schema_with_ids =
            Arc::new(build_schema_with_field_ids(arrow_schema, &setup.column_ids));

        let scoped_base = match self.metadata.catalog_id() {
            Some(id) => join_paths(&self.base_key_path, &format!("cat_{id}"))?,
            None => self.base_key_path.clone(),
        };
        let table_key = join_paths(&join_paths(&scoped_base, schema_name)?, table_name)?;

        let partition = if resolve_layout {
            self.resolve_partition(setup.table_id, &setup.column_ids, arrow_schema)?
        } else {
            None
        };

        // Lay the rows out in the table's sort order before splitting or rolling.
        // This is a GLOBAL sort over the whole write, matching official DuckLake's
        // blocking PhysicalOrder above the insert plan — that is what makes successive
        // rolled files cover contiguous, non-overlapping ranges, so a reader can skip
        // whole files. Splitting by partition afterwards preserves relative order
        // within each partition, so every file it produces stays sorted.
        //
        // Skipped when the caller already arranged the rows (`!resolve_layout` — the
        // SQL INSERT path, whose plan carries a SortExec for this same spec).
        let sorted_owned: Vec<RecordBatch> = if resolve_layout {
            let lengths: Vec<usize> = batches.iter().map(|b| b.num_rows()).collect();
            let sorted = crate::sort::sort_batches_by_spec(
                batches.to_vec(),
                arrow_schema,
                self.metadata.live_sort_spec(setup.table_id)?.as_ref(),
            )?;
            // The sort concatenates into ONE batch. Rollover is evaluated at batch
            // boundaries, so handing that single batch onward would emit one file of
            // unbounded size no matter how large the write — losing rollover exactly
            // when a sort order makes it most valuable. Re-slice back into the
            // caller's batch lengths (order-preserving, zero-copy) so rollover sees
            // the same boundaries it would have without the sort.
            reslice_to_lengths(sorted, &lengths)
        } else {
            Vec::new()
        };
        let batches: &[RecordBatch] = if resolve_layout {
            &sorted_owned
        } else {
            batches
        };

        let file_infos = match partition.as_ref() {
            Some(spec) => {
                let output_schema: SchemaRef = Arc::new(arrow_schema.clone());
                let groups =
                    crate::partition::split_batches_by_partition(&output_schema, batches, spec)?;
                self.write_partition_groups(
                    &table_key,
                    schema_with_ids,
                    &setup.column_ids,
                    spec.partition_id,
                    &spec.key_names(),
                    &groups,
                )
                .await?
            },
            None => {
                self.write_rolled_files(
                    &table_key,
                    None,
                    schema_with_ids,
                    &setup.column_ids,
                    batches,
                )
                .await?
            },
        };
        if file_infos.is_empty() {
            return Err(crate::error::DuckLakeError::InvalidConfig(
                "write_rows: input produced no rows".to_string(),
            ));
        }
        let records_written: i64 = file_infos.iter().map(|f| f.record_count).sum();

        let committed = self.metadata.register_data_files(
            setup.table_id,
            schema_name,
            table_name,
            setup.snapshot_id,
            &file_infos,
            mode,
            setup.base_snapshot_id,
            &columns,
            &setup.column_ids,
        )?;

        Ok(WriteResult {
            snapshot_id: committed.snapshot_id,
            table_id: committed.table_id,
            schema_id: committed.schema_id,
            files_written: file_infos.len(),
            records_written,
        })
    }

    /// Resolve the table's live partition spec against the columns this write is
    /// about to produce, or `None` when the table is unpartitioned.
    ///
    /// `column_ids[i]` is the catalog id of `arrow_schema` field `i` (the pairing
    /// `begin_write_transaction` returns). Errors on a spec this crate cannot
    /// produce (`bucket`/unknown) rather than writing files that violate it.
    fn resolve_partition(
        &self,
        table_id: i64,
        column_ids: &[i64],
        arrow_schema: &Schema,
    ) -> Result<Option<crate::partition::PartitionWriteSpec>> {
        match self.metadata.live_partition_spec(table_id)? {
            None => Ok(None),
            Some(spec) => Ok(Some(crate::partition::PartitionWriteSpec::resolve(
                &spec,
                column_ids,
                arrow_schema,
            )?)),
        }
    }

    /// Write each partition group to its own Hive directory under `table_key`,
    /// stamping `partition_id` and the group's values on every file produced.
    ///
    /// A group may roll over into several files (each a contiguous slice of the
    /// group's rows); all of them share that group's partition values, so the
    /// catalog records one partition per file as the spec requires.
    async fn write_partition_groups(
        &self,
        table_key: &str,
        schema_with_ids: SchemaRef,
        column_ids: &[i64],
        partition_id: i64,
        key_names: &[String],
        groups: &[PartitionGroup],
    ) -> Result<Vec<DataFileInfo>> {
        let mut file_infos: Vec<DataFileInfo> = Vec::with_capacity(groups.len());
        for (values, batches) in groups {
            // Readable Hive-style relative subpath; files land under the table dir
            // and are registered relative to it.
            let rel = crate::partition::hive_subpath(key_names, values);
            let rel_prefix = if rel.is_empty() {
                None
            } else {
                Some(rel.as_str())
            };
            let group_files = self
                .write_rolled_files(
                    table_key,
                    rel_prefix,
                    schema_with_ids.clone(),
                    column_ids,
                    batches,
                )
                .await?;
            let partition_values: Vec<(i32, Option<String>)> = values
                .iter()
                .enumerate()
                .map(|(i, v)| (i as i32, v.clone()))
                .collect();
            for info in group_files {
                file_infos.push(info.with_partition(partition_id, partition_values.clone()));
            }
        }
        Ok(file_infos)
    }

    /// Write `batches` into ONE OR MORE parquet files under `table_key` (data
    /// columns with catalog field-ids, no embedded rowid), rolling over to a new file
    /// whenever the current file reaches `target_file_size`. `rel_prefix`, when set, is
    /// a Hive-style subpath (a partition group) each file is placed under.
    ///
    /// Rollover mechanics live in [`RollingFileWriter`], shared with the streaming
    /// session's per-partition sink so the two cannot drift. This path differs only in
    /// its upload policy: having `await` available, it uploads each file as soon as it
    /// rolls, so at most one staged file occupies local disk at a time.
    ///
    /// Returns one [`DataFileInfo`] per file (relative catalog path set, stats and
    /// footer harvested); empty rows produce no file, and a write below
    /// `target_file_size` yields exactly one.
    async fn write_rolled_files(
        &self,
        table_key: &str,
        rel_prefix: Option<&str>,
        schema_with_ids: SchemaRef,
        column_ids: &[i64],
        batches: &[RecordBatch],
    ) -> Result<Vec<DataFileInfo>> {
        let mut roller = RollingFileWriter::new(
            table_key.to_string(),
            rel_prefix.map(str::to_string),
            schema_with_ids,
            column_ids.len(),
            self.build_writer_props(),
            self.target_file_size,
            None,
        );
        let mut files: Vec<DataFileInfo> = Vec::new();
        for batch in batches {
            if let Some(staged) = roller.write(batch)? {
                files.push(upload_staged_file(staged, &self.object_store, column_ids).await?);
            }
        }
        if let Some(staged) = roller.finish()? {
            files.push(upload_staged_file(staged, &self.object_store, column_ids).await?);
        }
        Ok(files)
    }
}

/// Split `batches` back into slices of `lengths` rows, in order.
///
/// A global sort returns one concatenated batch, but rollover is evaluated per batch,
/// so handing that single batch to a [`RollingFileWriter`] would emit one file of
/// unbounded size — losing rollover exactly where a sort order makes it most valuable.
/// `RecordBatch::slice` is a zero-copy view, so restoring the caller's boundaries costs
/// no data movement.
///
/// Returns the input untouched when it already matches `lengths` (no sort was applied)
/// or when the totals disagree, so a mismatch degrades to "write what we have" rather
/// than dropping or duplicating rows.
fn reslice_to_lengths(batches: Vec<RecordBatch>, lengths: &[usize]) -> Vec<RecordBatch> {
    let total: usize = lengths.iter().sum();
    if batches.len() != 1 || batches[0].num_rows() != total || lengths.len() <= 1 {
        return batches;
    }
    let combined = &batches[0];
    let mut out = Vec::with_capacity(lengths.len());
    let mut offset = 0usize;
    for len in lengths {
        if *len == 0 {
            continue;
        }
        out.push(combined.slice(offset, *len));
        offset += *len;
    }
    out
}

/// One parquet file being written to local staging.
#[derive(Debug)]
struct OpenFile {
    writer: ArrowWriter<std::io::BufWriter<std::fs::File>>,
    temp: NamedTempFile,
    /// Path relative to the table directory (includes any Hive subpath).
    catalog_path: String,
    object_path: ObjectPath,
    row_count: i64,
    nan_flags: Vec<Option<bool>>,
}

/// A finished parquet whose footer is written and whose staging file is complete on
/// disk, awaiting upload.
///
/// Holds a [`tempfile::TempPath`], not a `NamedTempFile`: nothing touches the file
/// again until upload, so keeping a descriptor open would make a writer's open-fd count
/// grow with the TOTAL number of files it produced rather than staying bounded.
/// `TempPath` keeps the file on disk (still deleted on drop) with no descriptor held.
#[derive(Debug)]
struct StagedFile {
    temp: tempfile::TempPath,
    catalog_path: String,
    object_path: ObjectPath,
    row_count: i64,
    nan_flags: Vec<Option<bool>>,
}

/// Writes batches into a sequence of parquet files, starting a new one whenever the
/// current file reaches `target_file_size`.
///
/// The single home for rollover, used by both write paths — the buffered
/// [`DuckLakeTableWriter::write_rolled_files`] and the streaming session's
/// per-partition sink — so the check cannot be applied in one place and forgotten in
/// the other. It was forgotten once already: a partitioned write left
/// `target_file_size` unenforceable within a partition.
///
/// Deliberately synchronous, and deliberately does NOT upload: `write` returns the
/// finished [`StagedFile`] whenever a roll happened and leaves the upload policy to the
/// caller. That is what lets the streaming session keep
/// [`TableWriteSession::write_batch`] synchronous (it defers uploads to `finish`) while
/// the buffered path uploads eagerly to bound local disk use.
#[derive(Debug)]
struct RollingFileWriter {
    table_key: String,
    rel_prefix: Option<String>,
    schema_with_ids: SchemaRef,
    /// Number of catalog data columns, for NaN-flag accumulation.
    data_column_count: usize,
    props: WriterProperties,
    target_file_size: usize,
    open: Option<OpenFile>,
    /// Catalog path to use for the FIRST file instead of minting a fresh name.
    ///
    /// A streaming session pre-computes its output path at `begin_write` and exposes it
    /// through [`TableWriteSession::file_path`]. Handing that path to the roller keeps
    /// that accessor accurate for the first (and, for a write below
    /// `target_file_size`, only) file, so rolling does not silently change what an
    /// existing caller observes. Taken on first use.
    first_catalog_path: Option<String>,
}

impl RollingFileWriter {
    fn new(
        table_key: String,
        rel_prefix: Option<String>,
        schema_with_ids: SchemaRef,
        data_column_count: usize,
        props: WriterProperties,
        target_file_size: usize,
        first_catalog_path: Option<String>,
    ) -> Self {
        Self {
            first_catalog_path,
            table_key,
            rel_prefix,
            schema_with_ids,
            data_column_count,
            props,
            target_file_size,
            open: None,
        }
    }

    /// Append `batch`, opening a file if none is in progress. Returns the finished
    /// [`StagedFile`] when this batch pushed the current file to `target_file_size`
    /// (rollover is evaluated at batch boundaries, so a file always holds a whole
    /// number of batches and any input ordering is preserved *across* files).
    ///
    /// `batch` must carry the table's data columns positionally; the field-id-tagged
    /// schema is re-imposed here.
    fn write(&mut self, batch: &RecordBatch) -> Result<Option<StagedFile>> {
        if batch.num_rows() == 0 {
            return Ok(None);
        }
        if self.open.is_none() {
            self.open = Some(self.open_file()?);
        }
        let batch_with_ids =
            RecordBatch::try_new(self.schema_with_ids.clone(), batch.columns().to_vec())?;
        let open = self.open.as_mut().expect("file opened above");
        crate::stats_collect::accumulate_nan_flags(
            &mut open.nan_flags,
            &batch_with_ids,
            self.data_column_count,
        );
        open.writer.write(&batch_with_ids)?;
        open.row_count += batch.num_rows() as i64;

        // Estimated encoded size = finished row groups + the in-progress one.
        // Strictly greater, matching official's parquet rotate predicate
        // (`FileSize() > file_size_bytes`), so a write landing exactly on the target
        // stays in one file.
        if open.writer.bytes_written() + open.writer.in_progress_size() > self.target_file_size {
            return Ok(Some(finalize_open_file(
                self.open.take().expect("file open"),
            )?));
        }
        Ok(None)
    }

    /// Finish the trailing (or only) file, if any rows were written.
    fn finish(&mut self) -> Result<Option<StagedFile>> {
        match self.open.take() {
            Some(open) => Ok(Some(finalize_open_file(open)?)),
            None => Ok(None),
        }
    }

    /// Whether a file is currently in progress (any rows written since the last roll).
    fn has_open_file(&self) -> bool {
        self.open.is_some()
    }

    fn open_file(&mut self) -> Result<OpenFile> {
        let catalog_path = match self.first_catalog_path.take() {
            Some(path) => path,
            None => {
                let file_name = format!("{}.parquet", Uuid::new_v4());
                match self.rel_prefix.as_deref() {
                    Some(prefix) if !prefix.is_empty() => format!("{prefix}/{file_name}"),
                    _ => file_name,
                }
            },
        };
        let object_path_str = join_paths(&self.table_key, &catalog_path)?;
        let object_path = ObjectPath::from(object_path_str.trim_start_matches('/'));
        let temp = NamedTempFile::new()?;
        let staging = std::io::BufWriter::new(temp.reopen()?);
        let writer = ArrowWriter::try_new(
            staging,
            self.schema_with_ids.clone(),
            Some(self.props.clone()),
        )?;
        Ok(OpenFile {
            writer,
            temp,
            catalog_path,
            object_path,
            row_count: 0,
            nan_flags: Vec::new(),
        })
    }
}

/// Write the parquet footer and flush the staging file to disk, releasing its
/// descriptor. Synchronous, so a streaming write needs no await to roll a file.
fn finalize_open_file(file: OpenFile) -> Result<StagedFile> {
    let staged = file.writer.into_inner()?;
    // `into_inner` flushes the buffered footer bytes to the OS file; dropping the
    // returned handle closes that descriptor.
    staged
        .into_inner()
        .map_err(|e| crate::error::DuckLakeError::Io(e.into_error()))?;
    Ok(StagedFile {
        temp: file.temp.into_temp_path(),
        catalog_path: file.catalog_path,
        object_path: file.object_path,
        row_count: file.row_count,
        nan_flags: file.nan_flags,
    })
}

/// Upload a finished staging file and harvest its per-column stats, returning the
/// [`DataFileInfo`] for the catalog commit (relative path; the caller stamps any
/// partition). On failure the multipart upload is aborted so no partial object is left.
async fn upload_staged_file(
    staged: StagedFile,
    object_store: &Arc<dyn ObjectStore>,
    column_ids: &[i64],
) -> Result<DataFileInfo> {
    // Reopen the staged file (its descriptor was released at finalize time).
    let mut file = std::fs::File::open(&staged.temp)?;
    let file_size = file.metadata()?.len() as i64;
    let footer_size = read_footer_size(&mut file)?;

    let local = tokio::fs::File::open(&staged.temp).await?;
    let mut reader = tokio::io::BufReader::new(local);
    let mut upload = ObjectBufWriter::new(Arc::clone(object_store), staged.object_path.clone());
    if let Err(e) = stream_to_upload(&mut reader, &mut upload).await {
        let _ = upload.abort().await;
        return Err(e.into());
    }

    let column_stats = crate::stats_collect::collect_column_stats(
        &staged.temp,
        column_ids,
        staged.row_count,
        &staged.nan_flags,
    );
    Ok(
        DataFileInfo::new(&staged.catalog_path, file_size, staged.row_count)
            .with_footer_size(footer_size)
            .with_column_stats(column_stats),
    )
}

/// How a streaming write session handles a partitioned target.
#[derive(Debug, Clone, Copy)]
enum StreamPartitionMode {
    /// Split rows across one file per partition (a [`PartitionSink`]).
    Split,
    /// Refuse: this entry point writes to a single file the caller chose (a custom
    /// path, or a file carrying embedded row lineage), which cannot also satisfy a
    /// partition spec. Errors with the entry point named, rather than letting the
    /// commit fail later with a partition-fence conflict that reads as a concurrent
    /// DDL change.
    Reject {
        entry_point: &'static str,
    },
}

/// Routes a streaming write's rows into one parquet file per partition.
///
/// Mirrors DuckDB's partitioned COPY sink (which is how official DuckLake writes a
/// partitioned table): keep a writer open per partition seen, roll at
/// `target_file_size`, and finalize the least-recently-opened one when the number of
/// open files would exceed `max_open`. All files produced are committed in ONE
/// snapshot, so a partitioned streaming write is as atomic as an unpartitioned one.
///
/// Each partition's file sequence is a [`RollingFileWriter`] — the same rollover
/// implementation the buffered path uses — so the two cannot drift. This sink differs
/// only in upload policy: `write_batch` must stay synchronous, so rolled and evicted
/// files are held as staged files on disk and uploaded together in `finish`.
#[derive(Debug)]
struct PartitionSink {
    spec: crate::partition::PartitionWriteSpec,
    key_names: Vec<String>,
    /// Object-store key prefix of the table directory; Hive subpaths hang off it.
    table_key: String,
    /// Field-id-tagged schema every written batch carries.
    schema_with_ids: SchemaRef,
    column_ids: Vec<i64>,
    props: WriterProperties,
    target_file_size: usize,
    max_open: usize,
    /// One roller per partition with a file in progress, oldest first (eviction takes
    /// from the front). Paired with the partition values its files carry.
    open: Vec<(Vec<Option<String>>, RollingFileWriter)>,
    /// Finished files awaiting upload at `finish`, with the partition each belongs to.
    staged: Vec<(Vec<Option<String>>, StagedFile)>,
}

impl PartitionSink {
    /// Split `batch` by partition and write each group to that partition's roller.
    fn write_batch(&mut self, batch: &RecordBatch) -> Result<()> {
        // The group batches are built with the field-id-tagged schema, so the roller
        // re-imposing that schema is a no-op rather than a mismatch.
        let groups = crate::partition::split_batches_by_partition(
            &self.schema_with_ids,
            std::slice::from_ref(batch),
            &self.spec,
        )?;
        for (values, batches) in groups {
            for group_batch in batches {
                if group_batch.num_rows() == 0 {
                    continue;
                }
                self.write_group_batch(&values, &group_batch)?;
            }
        }
        Ok(())
    }

    /// Append one partition's rows to its roller, opening one (and evicting another
    /// partition's file if already at `max_open`) when this partition has none.
    fn write_group_batch(&mut self, values: &[Option<String>], batch: &RecordBatch) -> Result<()> {
        let index = match self.open.iter().position(|(v, _)| v == values) {
            Some(index) => index,
            None => {
                if self.open.len() >= self.max_open {
                    // Evict the least-recently-opened partition: finish its file so it
                    // is complete on disk, and upload it at `finish`. That partition
                    // simply gets another file if more of its rows arrive.
                    let (evicted_values, mut evicted) = self.open.remove(0);
                    if let Some(staged) = evicted.finish()? {
                        self.staged.push((evicted_values, staged));
                    }
                }
                let rel = crate::partition::hive_subpath(&self.key_names, values);
                self.open.push((
                    values.to_vec(),
                    RollingFileWriter::new(
                        self.table_key.clone(),
                        if rel.is_empty() {
                            None
                        } else {
                            Some(rel)
                        },
                        self.schema_with_ids.clone(),
                        self.column_ids.len(),
                        self.props.clone(),
                        self.target_file_size,
                        None,
                    ),
                ));
                self.open.len() - 1
            },
        };

        let (partition_values, roller) = &mut self.open[index];
        if let Some(staged) = roller.write(batch)? {
            let partition_values = partition_values.clone();
            self.staged.push((partition_values, staged));
            // The roller rolled its file; drop it from `open` unless it already has a
            // fresh one in progress, so the open-file cap counts real open files.
            if !self.open[index].1.has_open_file() {
                self.open.remove(index);
            }
        }
        Ok(())
    }

    /// Finish every open file and upload all staged files, returning the
    /// [`DataFileInfo`]s to commit — each stamped with its partition.
    async fn into_file_infos(
        mut self,
        object_store: &Arc<dyn ObjectStore>,
    ) -> Result<Vec<DataFileInfo>> {
        for (values, mut roller) in std::mem::take(&mut self.open) {
            if let Some(staged) = roller.finish()? {
                self.staged.push((values, staged));
            }
        }
        let mut infos = Vec::with_capacity(self.staged.len());
        for (values, staged) in std::mem::take(&mut self.staged) {
            let partition_values: Vec<(i32, Option<String>)> = values
                .iter()
                .enumerate()
                .map(|(i, v)| (i as i32, v.clone()))
                .collect();
            let info = upload_staged_file(staged, object_store, &self.column_ids).await?;
            infos.push(info.with_partition(self.spec.partition_id, partition_values));
        }
        Ok(infos)
    }
}

/// Streaming write session./// Streaming write session. Batches stream to a local staging file; the
/// finished parquet is uploaded in `finish()`. If the session is dropped
/// without finishing, the staging file is removed and nothing is uploaded.
#[derive(Debug)]
pub struct TableWriteSession {
    metadata: Arc<dyn MetadataWriter>,
    object_store: Arc<dyn ObjectStore>,
    object_path: ObjectPath,
    /// Target identifiers threaded to `register_data_file`. Multicatalog Postgres
    /// writes the schema/table metadata at the commit (keyed by these names);
    /// single-catalog SQLite ignores them (it created them at begin).
    schema_name: String,
    table_name: String,
    snapshot_id: i64,
    /// Catalog head observed at `begin_write_transaction`; threaded to
    /// `register_data_file` so a `Replace` commit can abort if another writer
    /// published a newer generation of the table since this write began.
    base_snapshot_id: i64,
    /// Explicit table-state precondition supplied through [`TableWriteOptions`].
    expected_base_snapshot_id: Option<i64>,
    table_id: i64,
    /// Column generation for this write (in `column_order`). Threaded to the
    /// metadata writer at `finish()` so single-catalog backends, which defer the
    /// column generation out of `begin_write_transaction`, can insert the
    /// column rows with `column_ids` at the atomic commit.
    columns: Vec<ColumnDef>,
    column_ids: Vec<i64>,
    schema_with_ids: SchemaRef,
    /// Parquet writer streaming to the local staging file (`temp`). Batches are
    /// written to disk as they arrive rather than buffered in memory, so peak
    /// memory stays bounded by the parquet row-group size regardless of table
    /// size. The finished file is streamed to object storage in `finish()`.
    writer: Option<ArrowWriter<std::io::BufWriter<std::fs::File>>>,
    /// Local staging file backing `writer`. Kept alive for the session; the
    /// finished parquet is uploaded from it and the file is removed on drop.
    temp: Option<NamedTempFile>,
    /// Path to register in catalog (may be relative filename or absolute path)
    catalog_path: String,
    /// Whether the catalog_path is relative to table path
    path_is_relative: bool,
    /// Replace vs Append; passed to `register_data_file` so the head advance and
    /// (for Replace) prior-generation retirement commit atomically with the file.
    mode: WriteMode,
    row_count: i64,
    /// Per-data-column NaN presence, accumulated across written batches (the
    /// Parquet footer carries no NaN flag). One entry per catalog data column;
    /// `None` for non-float columns. Fed into `collect_column_stats` at finish.
    nan_flags: Vec<Option<bool>>,
    /// Set when the target table is partitioned: batches are routed here, one file
    /// per partition, and `writer`/`temp` above stay `None`. `finish` then commits
    /// every file the sink produced in a single snapshot.
    partition_sink: Option<PartitionSink>,
    /// Set unless the session is single-file (see
    /// [`DuckLakeTableWriter::begin_write_single_file`]): batches are routed
    /// through this instead of the single `writer` above, starting a new file each
    /// time one reaches `target_file_size`, and `finish` commits them all in one
    /// snapshot. `None` for a single-file session.
    roller: Option<RollingFileWriter>,
    /// Files the roller has finished, awaiting upload at `finish`.
    rolled: Vec<StagedFile>,
    commit_metadata: SnapshotCommitMetadata,
}

impl TableWriteSession {
    /// Applies snapshot metadata and an optional table-state precondition.
    #[must_use]
    pub fn with_options(mut self, options: &TableWriteOptions) -> Self {
        self.commit_metadata = options.commit_metadata.clone();
        if let Some(snapshot_id) = options.expected_base_snapshot_id {
            self.base_snapshot_id = snapshot_id;
        }
        self.expected_base_snapshot_id = options.expected_base_snapshot_id;
        self
    }

    /// Attaches metadata to the snapshot committed by this write.
    ///
    /// [`Self::finish`] returns an error when the configured metadata writer
    /// does not support non-empty commit metadata.
    #[must_use]
    pub fn with_commit_metadata(mut self, commit_metadata: SnapshotCommitMetadata) -> Self {
        self.commit_metadata = commit_metadata;
        self
    }

    pub fn write_batch(&mut self, batch: &RecordBatch) -> Result<()> {
        // Rolling or partitioned target: validate up front so the shared borrow of
        // `self` that `validate_batch_schema` takes is released before the roller or
        // sink is borrowed mutably.
        if self.roller.is_some() || self.partition_sink.is_some() {
            self.validate_batch_schema(batch)?;
        }
        if let Some(roller) = &mut self.roller {
            let rows = batch.num_rows() as i64;
            // `roller` and `rolled` are distinct fields, so both can be borrowed here.
            if let Some(staged) = roller.write(batch)? {
                self.rolled.push(staged);
            }
            self.row_count += rows;
            return Ok(());
        }
        if let Some(sink) = &mut self.partition_sink {
            let rows = batch.num_rows() as i64;
            sink.write_batch(batch)?;
            self.row_count += rows;
            return Ok(());
        }
        if self.writer.is_none() {
            return Err(crate::error::DuckLakeError::Internal(
                "Writer already closed".to_string(),
            ));
        }
        self.validate_batch_schema(batch)?;

        let batch_with_ids =
            RecordBatch::try_new(self.schema_with_ids.clone(), batch.columns().to_vec())?;
        // Note float-column NaN presence before the batch streams to disk (the
        // footer we later harvest has no NaN flag). Only the catalog data columns.
        crate::stats_collect::accumulate_nan_flags(
            &mut self.nan_flags,
            &batch_with_ids,
            self.column_ids.len(),
        );
        let writer = self.writer.as_mut().unwrap();
        writer.write(&batch_with_ids)?;
        self.row_count += batch.num_rows() as i64;
        Ok(())
    }

    fn validate_batch_schema(&self, batch: &RecordBatch) -> Result<()> {
        let batch_schema = batch.schema();
        let expected_schema = &self.schema_with_ids;

        if batch_schema.fields().len() != expected_schema.fields().len() {
            return Err(crate::error::DuckLakeError::InvalidConfig(format!(
                "Schema mismatch: batch has {} columns, expected {}",
                batch_schema.fields().len(),
                expected_schema.fields().len()
            )));
        }

        for (i, (batch_field, expected_field)) in batch_schema
            .fields()
            .iter()
            .zip(expected_schema.fields().iter())
            .enumerate()
        {
            if batch_field.data_type() != expected_field.data_type() {
                return Err(crate::error::DuckLakeError::InvalidConfig(format!(
                    "Schema mismatch at column {}: batch has type {:?}, expected {:?}",
                    i,
                    batch_field.data_type(),
                    expected_field.data_type()
                )));
            }
        }
        Ok(())
    }

    pub fn row_count(&self) -> i64 {
        self.row_count
    }

    pub fn snapshot_id(&self) -> i64 {
        self.snapshot_id
    }

    /// The object path this session writes to.
    ///
    /// For a partitioned session there is no single output file (one file per
    /// partition, plus rollovers), so this returns the table directory the partition
    /// subpaths hang off.
    pub fn file_path(&self) -> &str {
        self.object_path.as_ref()
    }

    pub async fn finish(mut self) -> Result<WriteResult> {
        // Rolling: finish the in-progress file, upload every file this session
        // produced, and commit them in ONE snapshot.
        if let Some(mut roller) = self.roller.take() {
            if let Some(staged) = roller.finish()? {
                self.rolled.push(staged);
            }
            let mut file_infos = Vec::with_capacity(self.rolled.len());
            for staged in std::mem::take(&mut self.rolled) {
                file_infos
                    .push(upload_staged_file(staged, &self.object_store, &self.column_ids).await?);
            }
            if file_infos.is_empty() {
                // No rows arrived. Fall through to the single-file path, which
                // registers the 0-row marker a Replace needs to retire the prior
                // generation.
                return self.finish_single_file().await;
            }
            let records_written: i64 = file_infos.iter().map(|f| f.record_count).sum();
            let committed = self.metadata.register_data_files_with_commit_metadata(
                self.table_id,
                &self.schema_name,
                &self.table_name,
                self.snapshot_id,
                &file_infos,
                self.mode,
                self.base_snapshot_id,
                &self.columns,
                &self.column_ids,
                &self.commit_metadata,
                self.expected_base_snapshot_id,
            )?;
            return Ok(WriteResult {
                snapshot_id: committed.snapshot_id,
                table_id: committed.table_id,
                schema_id: committed.schema_id,
                files_written: file_infos.len(),
                records_written,
            });
        }
        // Partitioned: commit every file the sink produced in ONE snapshot, so a
        // partitioned streaming write is as atomic as an unpartitioned one.
        if let Some(sink) = self.partition_sink.take() {
            let file_infos = sink.into_file_infos(&self.object_store).await?;
            if file_infos.is_empty() {
                // No rows reached any partition. Fall through to the single-file
                // path, which registers the 0-row marker that carries a Replace
                // truncation (and is exempt from the partition fence).
                return self.finish_single_file().await;
            }
            let records_written: i64 = file_infos.iter().map(|f| f.record_count).sum();
            let committed = self.metadata.register_data_files_with_commit_metadata(
                self.table_id,
                &self.schema_name,
                &self.table_name,
                self.snapshot_id,
                &file_infos,
                self.mode,
                self.base_snapshot_id,
                &self.columns,
                &self.column_ids,
                &self.commit_metadata,
                self.expected_base_snapshot_id,
            )?;
            return Ok(WriteResult {
                snapshot_id: committed.snapshot_id,
                table_id: committed.table_id,
                schema_id: committed.schema_id,
                files_written: file_infos.len(),
                records_written,
            });
        }
        self.finish_single_file().await
    }

    /// Commit this session's single staged file (the unpartitioned path, and the
    /// 0-row truncate marker of a partitioned Replace).
    async fn finish_single_file(mut self) -> Result<WriteResult> {
        let file_info = self.upload_staged().await?;
        // register_data_file returns the ids actually committed (snapshot id
        // assigned at commit; real schema/table ids, which may differ from the
        // begin-time reservations under a concurrent create). Report those.
        let committed = self.metadata.register_data_file_with_commit_metadata(
            self.table_id,
            &self.schema_name,
            &self.table_name,
            self.snapshot_id,
            &file_info,
            self.mode,
            self.base_snapshot_id,
            &self.columns,
            &self.column_ids,
            &self.commit_metadata,
            self.expected_base_snapshot_id,
        )?;

        Ok(WriteResult {
            snapshot_id: committed.snapshot_id,
            table_id: committed.table_id,
            schema_id: committed.schema_id,
            files_written: 1,
            records_written: self.row_count,
        })
    }

    /// Like [`finish`](Self::finish), but atomically applies positional
    /// `deletes` to existing data files in the SAME snapshot as this append —
    /// the commit behind an update/upsert (supersede rows and insert their new
    /// versions in one snapshot). The caller resolves the positions and writes
    /// each delete file (see [`DuckLakeTableWriter::write_delete_file`]) before
    /// calling this; `deletes` may be empty, which delegates to
    /// [`finish`](Self::finish).
    ///
    /// A rolling or partitioned session may have produced several appended files;
    /// all of them commit in the same snapshot as the deletes.
    pub async fn finish_with_deletes(mut self, deletes: &[DeleteFileEntry]) -> Result<WriteResult> {
        // No deletes means this IS a plain append, so take the ordinary commit path
        // and make the documented equivalence literal. This matters beyond
        // tidiness: the delete-carrying commit is implemented only by the backends
        // that support positional deletes, whereas `finish` reaches
        // `register_data_file`/`register_data_files`, which every backend
        // implements. Routing an empty-delete finish through the delete commit
        // would therefore fail as unsupported on a backend that can commit the
        // append perfectly well — and only after uploading, leaving orphaned
        // objects behind.
        if deletes.is_empty() {
            return self.finish().await;
        }
        // Reject an unsupported combination before uploading anything, so a misuse
        // leaves no orphan object in storage.
        validate_delete_entries(self.mode, deletes)?;
        let file_infos: Vec<DataFileInfo> = if let Some(sink) = self.partition_sink.take() {
            let file_infos = sink.into_file_infos(&self.object_store).await?;
            if file_infos.is_empty() {
                // No rows reached any partition. Fall through to the single-file
                // path, whose 0-row marker is what carries a Replace truncation
                // (and is exempt from the partition fence).
                vec![self.upload_staged().await?]
            } else {
                file_infos
            }
        } else if let Some(mut roller) = self.roller.take() {
            // `finish` writes the parquet footer locally; nothing is uploaded yet.
            if let Some(staged) = roller.finish()? {
                self.rolled.push(staged);
            }
            if self.rolled.is_empty() {
                // No rows arrived. Fall through to the single-file path, whose 0-row
                // marker is what carries a Replace truncation (and is exempt from the
                // partition fence) — same behaviour as a non-rolling session.
                vec![self.upload_staged().await?]
            } else {
                let mut file_infos = Vec::with_capacity(self.rolled.len());
                for staged in std::mem::take(&mut self.rolled) {
                    file_infos.push(
                        upload_staged_file(staged, &self.object_store, &self.column_ids).await?,
                    );
                }
                file_infos
            }
        } else {
            vec![self.upload_staged().await?]
        };
        let records_written: i64 = file_infos.iter().map(|f| f.record_count).sum();
        // One appended file goes through the single-file commit, so a backend that
        // implements only that form keeps working; N>1 needs the multi-file commit.
        let committed = match file_infos.as_slice() {
            [file_info] => self
                .metadata
                .register_data_file_with_deletes_and_commit_metadata(
                    self.table_id,
                    &self.schema_name,
                    &self.table_name,
                    self.snapshot_id,
                    file_info,
                    deletes,
                    self.mode,
                    self.base_snapshot_id,
                    &self.columns,
                    &self.column_ids,
                    &self.commit_metadata,
                    self.expected_base_snapshot_id,
                )?,
            file_infos => self
                .metadata
                .register_data_files_with_deletes_and_commit_metadata(
                    self.table_id,
                    &self.schema_name,
                    &self.table_name,
                    self.snapshot_id,
                    file_infos,
                    deletes,
                    self.mode,
                    self.base_snapshot_id,
                    &self.columns,
                    &self.column_ids,
                    &self.commit_metadata,
                    self.expected_base_snapshot_id,
                )?,
        };

        Ok(WriteResult {
            snapshot_id: committed.snapshot_id,
            table_id: committed.table_id,
            schema_id: committed.schema_id,
            files_written: file_infos.len(),
            records_written,
        })
    }

    /// Finalise + upload the staged parquet and return its [`DataFileInfo`],
    /// leaving the metadata commit to the caller. Shared by
    /// [`finish`](Self::finish) and [`finish_with_deletes`](Self::finish_with_deletes).
    async fn upload_staged(&mut self) -> Result<DataFileInfo> {
        let writer = self.writer.take().ok_or_else(|| {
            crate::error::DuckLakeError::Internal("Writer already closed".to_string())
        })?;
        let temp = self.temp.take().ok_or_else(|| {
            crate::error::DuckLakeError::Internal("Writer already closed".to_string())
        })?;

        // Finalise the parquet footer, then unwrap the `BufWriter` (its
        // `into_inner` flushes any buffered footer bytes to the OS file) so the
        // staging file on disk is the complete parquet.
        let staged = writer.into_inner()?;
        let mut file = staged
            .into_inner()
            .map_err(|e| crate::error::DuckLakeError::Io(e.into_error()))?;

        let file_size = file.metadata()?.len() as i64;
        let footer_size = read_footer_size(&mut file)?;

        // Stream the staged file to object storage. `BufWriter` chunks the
        // payload and switches to a multipart upload for large files, so there
        // is no 5 GiB single-PUT ceiling and memory stays bounded. On failure
        // we abort so no incomplete multipart parts are left behind.
        let local = tokio::fs::File::open(temp.path()).await?;
        let mut reader = tokio::io::BufReader::new(local);
        let mut upload =
            ObjectBufWriter::new(Arc::clone(&self.object_store), self.object_path.clone());
        if let Err(e) = stream_to_upload(&mut reader, &mut upload).await {
            let _ = upload.abort().await;
            return Err(e.into());
        }

        // Harvest per-column statistics from the parquet footer we just wrote
        // (mirrors DuckLake reading its writer's WRITTEN_FILE_STATISTICS) and
        // attach them for the catalog commit. Best-effort: on failure the file
        // is registered without stats, which is spec-safe.
        let column_stats = crate::stats_collect::collect_column_stats(
            temp.path(),
            &self.column_ids,
            self.row_count,
            &self.nan_flags,
        );

        let mut file_info = DataFileInfo::new(&self.catalog_path, file_size, self.row_count)
            .with_footer_size(footer_size)
            .with_column_stats(column_stats);
        if !self.path_is_relative {
            file_info = file_info.with_absolute_path();
        }
        Ok(file_info)
    }
}

// Drop deletes the staging `NamedTempFile`; a session abandoned before
// `finish()` uploads nothing and leaves no local file behind.

/// Stream a finished local parquet file to object storage and finalise the
/// upload. `BufWriter` switches to a multipart upload once the payload exceeds
/// its buffer, so files larger than the object store's single-PUT limit (5 GiB
/// on S3) upload fine and memory stays bounded.
async fn stream_to_upload<R>(reader: &mut R, upload: &mut ObjectBufWriter) -> std::io::Result<()>
where
    R: tokio::io::AsyncRead + Unpin + ?Sized,
{
    tokio::io::copy(reader, upload).await?;
    upload.shutdown().await?;
    Ok(())
}

/// Read the parquet footer length (thrift metadata + 8-byte trailer) from the
/// tail of a finished parquet file on disk. Stored as the nullable
/// `footer_size` hint in the catalog; readers fall back to a standard footer
/// read when it is absent.
fn read_footer_size(file: &mut std::fs::File) -> Result<i64> {
    let len = file.metadata()?.len();
    if len < 8 {
        return Err(crate::error::DuckLakeError::Internal(
            "Invalid Parquet file: too small".to_string(),
        ));
    }
    file.seek(SeekFrom::End(-8))?;
    let mut tail = [0u8; 8];
    file.read_exact(&mut tail)?;
    calculate_footer_size_from_bytes(&tail)
}

fn arrow_schema_to_column_defs(schema: &Schema) -> Result<Vec<ColumnDef>> {
    schema
        .fields()
        .iter()
        .map(|field| ColumnDef::from_arrow(field.name(), field.data_type(), field.is_nullable()))
        .collect()
}

fn build_schema_with_field_ids(schema: &Schema, column_ids: &[i64]) -> Schema {
    let fields: Vec<Field> = schema
        .fields()
        .iter()
        .zip(column_ids.iter())
        .map(|(field, &col_id)| {
            let mut metadata: HashMap<String, String> = field.metadata().clone();
            metadata.insert("PARQUET:field_id".to_string(), col_id.to_string());
            Field::new(field.name(), field.data_type().clone(), field.is_nullable())
                .with_metadata(metadata)
        })
        .collect();

    Schema::new_with_metadata(fields, schema.metadata().clone())
}

fn calculate_footer_size_from_bytes(buffer: &[u8]) -> Result<i64> {
    if buffer.len() < 8 {
        return Err(crate::error::DuckLakeError::Internal(
            "Invalid Parquet file: too small".to_string(),
        ));
    }

    let footer_bytes = &buffer[buffer.len() - 8..];

    if &footer_bytes[4..8] != b"PAR1" {
        return Err(crate::error::DuckLakeError::Internal(
            "Invalid Parquet file: missing PAR1 magic".to_string(),
        ));
    }

    let metadata_len =
        i32::from_le_bytes([footer_bytes[0], footer_bytes[1], footer_bytes[2], footer_bytes[3]])
            as i64;
    Ok(metadata_len + 8)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Int32Array, StringArray};
    use arrow::datatypes::DataType;

    #[test]
    fn test_arrow_schema_to_column_defs() {
        let schema = Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
        ]);

        let columns = arrow_schema_to_column_defs(&schema).unwrap();
        assert_eq!(columns.len(), 2);
        assert_eq!(columns[0].name, "id");
        assert_eq!(columns[0].ducklake_type, "int32");
        assert!(!columns[0].is_nullable);
        assert_eq!(columns[1].name, "name");
        assert_eq!(columns[1].ducklake_type, "varchar");
        assert!(columns[1].is_nullable);
    }

    #[test]
    fn test_build_schema_with_field_ids() {
        let schema = Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
        ]);

        let column_ids = vec![1, 2];
        let schema_with_ids = build_schema_with_field_ids(&schema, &column_ids);

        // Check that field_ids are embedded in metadata
        let field0_metadata = schema_with_ids.field(0).metadata();
        assert_eq!(
            field0_metadata.get("PARQUET:field_id"),
            Some(&"1".to_string())
        );

        let field1_metadata = schema_with_ids.field(1).metadata();
        assert_eq!(
            field1_metadata.get("PARQUET:field_id"),
            Some(&"2".to_string())
        );
    }

    #[test]
    fn test_write_parquet_to_buffer_with_field_ids() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec!["a", "b", "c"])),
            ],
        )
        .unwrap();

        let column_ids = vec![10, 20];
        let schema_with_ids = Arc::new(build_schema_with_field_ids(&schema, &column_ids));

        let props = WriterProperties::builder()
            .set_writer_version(parquet::file::properties::WriterVersion::PARQUET_2_0)
            .build();
        let mut writer =
            ArrowWriter::try_new(Vec::new(), schema_with_ids.clone(), Some(props)).unwrap();

        let batch_with_ids =
            RecordBatch::try_new(schema_with_ids, batch.columns().to_vec()).unwrap();
        writer.write(&batch_with_ids).unwrap();
        let buffer = writer.into_inner().unwrap();

        let file_size = buffer.len() as i64;
        let footer_size = calculate_footer_size_from_bytes(&buffer).unwrap();

        assert!(file_size > 0);
        assert!(footer_size > 0);
        assert!(footer_size < file_size);
    }

    #[test]
    fn test_calculate_footer_size_from_bytes() {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));

        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![1, 2, 3]))]).unwrap();

        let props = WriterProperties::builder()
            .set_writer_version(parquet::file::properties::WriterVersion::PARQUET_2_0)
            .build();
        let schema_with_ids = Arc::new(build_schema_with_field_ids(&batch.schema(), &[1]));
        let mut writer =
            ArrowWriter::try_new(Vec::new(), schema_with_ids.clone(), Some(props)).unwrap();

        let batch_with_ids =
            RecordBatch::try_new(schema_with_ids, batch.columns().to_vec()).unwrap();
        writer.write(&batch_with_ids).unwrap();
        let buffer = writer.into_inner().unwrap();

        let footer_size = calculate_footer_size_from_bytes(&buffer).unwrap();

        // Footer should be reasonable size (metadata + 8 bytes)
        assert!(footer_size >= 8);
        assert!(footer_size < 10000);
    }
}
