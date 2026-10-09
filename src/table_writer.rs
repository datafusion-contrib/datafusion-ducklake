//! High-level table writer for DuckLake catalogs.

use std::collections::{HashMap, VecDeque};
use std::io::{Read, Seek, SeekFrom};
use std::sync::Arc;

use arrow::array::Array;
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use datafusion::error::DataFusionError;
use datafusion::execution::SendableRecordBatchStream;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use futures::{FutureExt, StreamExt};
use object_store::ObjectStore;
use object_store::buffered::BufWriter as ObjectBufWriter;
use object_store::path::Path as ObjectPath;
use parquet::arrow::ArrowWriter;
use parquet::basic::{BrotliLevel, Compression, GzipLevel, ZstdLevel};
use parquet::file::properties::{WriterProperties, WriterVersion};
use tempfile::NamedTempFile;
use tokio::io::AsyncWriteExt;
use uuid::Uuid;

use crate::Result;
use crate::metadata_provider::DuckLakeInlinedData;
use crate::metadata_writer::{
    ColumnDef, DataFileInfo, DeleteFileEntry, DeleteFileInfo, InlinedRowRef, MetadataWriter,
    SnapshotCommitMetadata, StagedTableData, StagedTableWrite, WriteMode, WriteResult,
    validate_delete_entries,
};
use crate::path_resolver::join_paths;
use crate::row_id::{embedded_rowid_field, embedded_snapshot_id_field};
use crate::table::delete_file_schema;
use crate::write_encoding::RowGroupSampledWriter;

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

/// How many finished data files a write uploads to object storage concurrently.
///
/// A rolling write finishes each file to local disk and starts uploading it while
/// the write goes on, so the uploads are independent I/O with no ordering
/// requirement between them — only the *resulting* `DataFileInfo` order matters,
/// because `register_data_files` assigns `row_id_start` by walking that list in
/// order. Uploading them one at a time leaves the link idle for most of the write.
///
/// Kept modest rather than core-count-scaled, because the real cost is memory and
/// sockets rather than CPU — and it is larger than "one buffer per upload" suggests.
/// `object_store`'s `BufWriter` holds a 10 MiB buffer AND keeps up to 8 multipart
/// requests in flight on its own, so N concurrent uploads of large files peak near
/// `N * 9 * 10 MiB`: roughly 360 MiB at the default of 4, not 40 MiB. Raise this
/// only against a memory budget that accounts for that multiplier.
pub const DEFAULT_UPLOAD_CONCURRENCY: usize = 4;

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

/// Write and maintenance options carried from the catalog down to each table.
/// A SQL `INSERT`, update rewrite, or compaction output uses the same compression,
/// row-group caps, file-rollover target, sorting, and partition-path policy.
/// A `None` field leaves the receiving writer's configured value unchanged.
/// Catalog-derived options populate DuckDB's defaults: Snappy, 122,880 rows per
/// row group, and [`DEFAULT_TARGET_FILE_SIZE`] rollover.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct DuckLakeWriteOptions {
    /// Maximum rows stored in catalog-backed inlined storage. Zero disables
    /// inlining; `None` leaves the receiving writer's limit unchanged.
    pub data_inlining_row_limit: Option<usize>,
    /// Parquet compression codec; `None` leaves the writer's codec unchanged.
    pub compression: Option<Compression>,
    /// Parquet format version; `None` leaves the writer's version unchanged.
    pub parquet_version: Option<WriterVersion>,
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
    /// How many finished data files to upload concurrently; `None` leaves the
    /// writer default ([`DEFAULT_UPLOAD_CONCURRENCY`]). Clamped to at least 1.
    pub upload_concurrency: Option<usize>,
    /// Whether inserts honor the table's active sort order.
    pub sort_on_insert: Option<bool>,
    /// Whether partition values appear as Hive-style directories in file paths.
    pub hive_file_pattern: Option<bool>,
    /// Whether merge and rewrite maintenance include this table.
    pub auto_compact: Option<bool>,
    /// Minimum deleted-row fraction for automatic rewrite selection.
    pub rewrite_delete_threshold: Option<f64>,
    deferred_error: Option<String>,
}

impl DuckLakeWriteOptions {
    pub(crate) fn from_metadata_settings(settings: &HashMap<String, String>) -> Result<Self> {
        let compression_name = settings
            .get("parquet_compression")
            .map(String::as_str)
            .unwrap_or("snappy");
        let compression_level = compression_name
            .eq_ignore_ascii_case("zstd")
            .then(|| {
                setting_i32(settings, "parquet_compression_level").map(|level| level.or(Some(3)))
            })
            .transpose()?
            .flatten();

        Ok(Self {
            // Keep inlining opt-in until inline UPDATE, rowid, CDC, SQL flush, and
            // automatic maintenance are supported. Explicit settings still apply (#270).
            data_inlining_row_limit: setting_usize(settings, "data_inlining_row_limit", Some(0))?,
            compression: Some(setting_compression(compression_name, compression_level)?),
            parquet_version: setting_parquet_version(settings)?,
            max_row_group_rows: setting_usize(settings, "parquet_row_group_size", Some(122_880))?,
            max_row_group_bytes: setting_size(settings, "parquet_row_group_size_bytes", None)?,
            target_file_size: setting_size(
                settings,
                "target_file_size",
                Some(DEFAULT_TARGET_FILE_SIZE),
            )?,
            max_open_partitions: None,
            upload_concurrency: None,
            sort_on_insert: setting_bool(settings, "sort_on_insert", true)?,
            hive_file_pattern: setting_bool(settings, "hive_file_pattern", true)?,
            auto_compact: setting_bool(settings, "auto_compact", true)?,
            rewrite_delete_threshold: setting_f64(
                settings,
                "rewrite_delete_threshold",
                Some(0.95),
            )?,
            deferred_error: None,
        })
    }

    pub(crate) fn from_metadata_settings_deferred(settings: &HashMap<String, String>) -> Self {
        Self::from_metadata_settings(settings).unwrap_or_else(|e| Self {
            deferred_error: Some(e.to_string()),
            ..Self::default()
        })
    }

    /// Sets the maximum row count for catalog-backed inlining.
    #[must_use]
    pub fn with_data_inlining_row_limit(mut self, limit: usize) -> Self {
        self.data_inlining_row_limit = Some(limit);
        self
    }

    pub(crate) fn validate(&self) -> Result<()> {
        match &self.deferred_error {
            Some(error) => Err(crate::error::DuckLakeError::InvalidConfig(format!(
                "Invalid DuckLake write settings: {error}"
            ))),
            None => Ok(()),
        }
    }

    pub(crate) fn with_overrides(mut self, overrides: &Self) -> Self {
        if overrides.data_inlining_row_limit.is_some() {
            self.data_inlining_row_limit = overrides.data_inlining_row_limit;
        }
        if overrides.compression.is_some() {
            self.compression = overrides.compression;
        }
        if overrides.parquet_version.is_some() {
            self.parquet_version = overrides.parquet_version;
        }
        if overrides.max_row_group_rows.is_some() {
            self.max_row_group_rows = overrides.max_row_group_rows;
        }
        if overrides.max_row_group_bytes.is_some() {
            self.max_row_group_bytes = overrides.max_row_group_bytes;
        }
        if overrides.target_file_size.is_some() {
            self.target_file_size = overrides.target_file_size;
        }
        if overrides.max_open_partitions.is_some() {
            self.max_open_partitions = overrides.max_open_partitions;
        }
        if overrides.upload_concurrency.is_some() {
            self.upload_concurrency = overrides.upload_concurrency;
        }
        if overrides.sort_on_insert.is_some() {
            self.sort_on_insert = overrides.sort_on_insert;
        }
        if overrides.hive_file_pattern.is_some() {
            self.hive_file_pattern = overrides.hive_file_pattern;
        }
        if overrides.auto_compact.is_some() {
            self.auto_compact = overrides.auto_compact;
        }
        if overrides.rewrite_delete_threshold.is_some() {
            self.rewrite_delete_threshold = overrides.rewrite_delete_threshold;
        }
        self
    }
}

fn setting_usize(
    settings: &HashMap<String, String>,
    key: &str,
    default: Option<usize>,
) -> Result<Option<usize>> {
    settings
        .get(key)
        .map(|value| {
            value.parse::<usize>().map_err(|e| {
                crate::error::DuckLakeError::InvalidConfig(format!(
                    "Invalid ducklake_metadata {key} value '{value}': {e}"
                ))
            })
        })
        .transpose()
        .map(|value| value.or(default))
}

fn setting_i32(settings: &HashMap<String, String>, key: &str) -> Result<Option<i32>> {
    settings
        .get(key)
        .map(|value| {
            value.parse::<i32>().map_err(|e| {
                crate::error::DuckLakeError::InvalidConfig(format!(
                    "Invalid ducklake_metadata {key} value '{value}': {e}"
                ))
            })
        })
        .transpose()
}

fn setting_f64(
    settings: &HashMap<String, String>,
    key: &str,
    default: Option<f64>,
) -> Result<Option<f64>> {
    let value = settings
        .get(key)
        .map(|value| {
            value.parse::<f64>().map_err(|e| {
                crate::error::DuckLakeError::InvalidConfig(format!(
                    "Invalid ducklake_metadata {key} value '{value}': {e}"
                ))
            })
        })
        .transpose()?
        .or(default);
    if value.is_some_and(|value| !(0.0..=1.0).contains(&value)) {
        return Err(crate::error::DuckLakeError::InvalidConfig(format!(
            "ducklake_metadata {key} must be in [0.0, 1.0]"
        )));
    }
    Ok(value)
}

fn setting_parquet_version(settings: &HashMap<String, String>) -> Result<Option<WriterVersion>> {
    settings
        .get("parquet_version")
        .map(|value| match value.to_ascii_lowercase().as_str() {
            "1" | "v1" => Ok(WriterVersion::PARQUET_1_0),
            "2" | "v2" => Ok(WriterVersion::PARQUET_2_0),
            _ => Err(crate::error::DuckLakeError::InvalidConfig(format!(
                "Invalid ducklake_metadata parquet_version '{value}'; expected 1, 2, V1, or V2"
            ))),
        })
        .transpose()
}

fn setting_size(
    settings: &HashMap<String, String>,
    key: &str,
    default: Option<usize>,
) -> Result<Option<usize>> {
    settings
        .get(key)
        .map(|value| parse_size(key, value))
        .transpose()
        .map(|value| value.or(default))
}

fn parse_size(key: &str, value: &str) -> Result<usize> {
    let value = value.trim();
    let digits = value.bytes().take_while(u8::is_ascii_digit).count();
    let (number, unit) = value.split_at(digits);
    if number.is_empty() {
        return Err(crate::error::DuckLakeError::InvalidConfig(format!(
            "Invalid ducklake_metadata {key} size '{value}'"
        )));
    }
    let number = number.parse::<usize>().map_err(|e| {
        crate::error::DuckLakeError::InvalidConfig(format!(
            "Invalid ducklake_metadata {key} size '{value}': {e}"
        ))
    })?;
    let multiplier = match unit.trim().to_ascii_lowercase().as_str() {
        "" | "b" => 1,
        "kb" => 1_000,
        "mb" => 1_000_000,
        "gb" => 1_000_000_000,
        "tb" => 1_000_000_000_000,
        "kib" => 1 << 10,
        "mib" => 1 << 20,
        "gib" => 1 << 30,
        "tib" => 1usize << 40,
        _ => {
            return Err(crate::error::DuckLakeError::InvalidConfig(format!(
                "Invalid ducklake_metadata {key} size unit in '{value}'"
            )));
        },
    };
    number.checked_mul(multiplier).ok_or_else(|| {
        crate::error::DuckLakeError::InvalidConfig(format!(
            "ducklake_metadata {key} size '{value}' exceeds usize"
        ))
    })
}

fn setting_bool(
    settings: &HashMap<String, String>,
    key: &str,
    default: bool,
) -> Result<Option<bool>> {
    let value = match settings.get(key) {
        Some(value) => match value.to_ascii_lowercase().as_str() {
            "true" => true,
            "false" => false,
            _ => {
                return Err(crate::error::DuckLakeError::InvalidConfig(format!(
                    "Invalid ducklake_metadata {key} value '{value}'; expected true or false"
                )));
            },
        },
        None => default,
    };
    Ok(Some(value))
}

fn setting_compression(name: &str, level: Option<i32>) -> Result<Compression> {
    let invalid = |e: parquet::errors::ParquetError| {
        crate::error::DuckLakeError::InvalidConfig(format!(
            "Invalid ducklake_metadata parquet_compression_level {}: {e}",
            level.unwrap_or_default()
        ))
    };
    match name.to_ascii_lowercase().as_str() {
        "uncompressed" => Ok(Compression::UNCOMPRESSED),
        "snappy" => Ok(Compression::SNAPPY),
        "gzip" => Ok(Compression::GZIP(GzipLevel::default())),
        "brotli" => Ok(Compression::BROTLI(BrotliLevel::default())),
        "zstd" => Ok(Compression::ZSTD(match level {
            None | Some(0) => ZstdLevel::default(),
            Some(level) => ZstdLevel::try_new(level).map_err(invalid)?,
        })),
        "lz4" | "lz4_raw" => Ok(Compression::LZ4_RAW),
        _ => Err(crate::error::DuckLakeError::InvalidConfig(format!(
            "Unsupported ducklake_metadata parquet_compression '{name}'"
        ))),
    }
}

/// Options shared by streaming and partitioned table writes.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TableWriteOptions {
    /// Metadata recorded in the snapshot change row.
    pub commit_metadata: SnapshotCommitMetadata,
    /// Catalog snapshot against which this write read its input.
    ///
    /// The commit fails if the target table's data files, delete files or
    /// inlined rows changed after this snapshot, including a commit that only
    /// deleted rows. Commits to other tables do not cause a conflict.
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

#[derive(Debug)]
struct PreparedTableWrite {
    write: StagedTableWrite,
    object_paths: Vec<ObjectPath>,
    files_written: usize,
    records_written: i64,
}

/// A set of table changes committed in one DuckLake snapshot.
///
/// DuckDB, SQLite, MySQL, and both PostgreSQL writers finalize new or existing
/// tables in this transaction. Empty staging calls add no result entry, and an
/// empty transaction returns without publishing a snapshot. Inlined deletes
/// ignore missing row identities and rows inserted by the same transaction;
/// concurrent changes still conflict. An explicit expected-base snapshot adds
/// a table-state precondition, including for append stages.
#[derive(Debug)]
pub struct DuckLakeWriteTransaction<'a> {
    writer: &'a DuckLakeTableWriter,
    writes: Vec<PreparedTableWrite>,
    commit_metadata: SnapshotCommitMetadata,
    expected_base_snapshot_id: Option<i64>,
}

/// High-level writer for DuckLake tables.
#[derive(Debug, Clone)]
pub struct DuckLakeTableWriter {
    metadata: Arc<dyn MetadataWriter>,
    object_store: Arc<dyn ObjectStore>,
    /// The key path portion of the data_path (e.g., "/prefix/data/")
    base_key_path: String,
    /// Compression codec for written data files. Defaults to `UNCOMPRESSED`;
    /// override via [`DuckLakeTableWriter::with_compression`] to trade write
    /// CPU for ~2x smaller files (e.g. `LZ4`, `SNAPPY`, `ZSTD`).
    compression: Compression,
    writer_version: WriterVersion,
    /// Optional max rows per parquet row group. `None` leaves the parquet
    /// default. Set via [`DuckLakeTableWriter::with_max_row_group_rows`].
    max_row_group_rows: Option<usize>,
    /// Optional max *uncompressed* bytes per parquet row group. `None` leaves
    /// the parquet default (rows-only). A reader decodes a whole row group at
    /// once, so a byte cap bounds reader memory for wide schemas (e.g. large
    /// vector columns). Set via [`DuckLakeTableWriter::with_max_row_group_bytes`].
    max_row_group_bytes: Option<usize>,
    /// How many finished data files a write uploads concurrently. Defaults to
    /// [`DEFAULT_UPLOAD_CONCURRENCY`]; override via
    /// [`DuckLakeTableWriter::with_upload_concurrency`].
    upload_concurrency: usize,
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
    data_inlining_row_limit: Option<usize>,
    sort_on_insert: bool,
    hive_file_pattern: bool,
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
            writer_version: WriterVersion::PARQUET_2_0,
            max_row_group_rows: None,
            max_row_group_bytes: None,
            upload_concurrency: DEFAULT_UPLOAD_CONCURRENCY,
            target_file_size: DEFAULT_TARGET_FILE_SIZE,
            max_open_partitions: DEFAULT_MAX_OPEN_PARTITIONS,
            data_inlining_row_limit: None,
            sort_on_insert: true,
            hive_file_pattern: true,
        })
    }

    /// Starts a write transaction that can stage changes for multiple tables.
    #[must_use]
    pub fn transaction(&self) -> DuckLakeWriteTransaction<'_> {
        DuckLakeWriteTransaction {
            writer: self,
            writes: Vec::new(),
            commit_metadata: SnapshotCommitMetadata::default(),
            expected_base_snapshot_id: None,
        }
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
    /// size and checked at batch boundaries once the file's first row group is
    /// complete, so a file holds at least one row group. Combined with a sort order, each file
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

    /// Override how many finished data files a write uploads concurrently.
    /// Defaults to [`DEFAULT_UPLOAD_CONCURRENCY`]. Values below 1 are clamped to 1
    /// (a write must still upload its files).
    #[must_use]
    pub fn with_upload_concurrency(mut self, files: usize) -> Self {
        self.upload_concurrency = files.max(1);
        self
    }

    /// Apply a [`DuckLakeWriteOptions`] set (compression, row-group caps, rollover
    /// target, open-partition cap, upload concurrency). Each field overrides the
    /// corresponding setting only when present.
    pub fn with_options(mut self, options: &DuckLakeWriteOptions) -> Self {
        if let Some(limit) = options.data_inlining_row_limit {
            self.data_inlining_row_limit = Some(limit);
        }
        if let Some(compression) = options.compression {
            self.compression = compression;
        }
        if let Some(writer_version) = options.parquet_version {
            self.writer_version = writer_version;
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
        if let Some(files) = options.upload_concurrency {
            self.upload_concurrency = files.max(1);
        }
        if let Some(sort_on_insert) = options.sort_on_insert {
            self.sort_on_insert = sort_on_insert;
        }
        if let Some(hive_file_pattern) = options.hive_file_pattern {
            self.hive_file_pattern = hive_file_pattern;
        }
        self
    }

    /// Build the parquet [`WriterProperties`] shared by every write path from this
    /// writer's configured compression and row-group caps.
    fn build_writer_props(&self) -> WriterProperties {
        let mut builder = WriterProperties::builder()
            .set_writer_version(self.writer_version)
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
    /// DuckLake rotates an unpartitioned insert the same way (`result.rotate = true`
    /// in `ducklake_insert.cpp`), and a single unbounded file could never be
    /// reorganized afterwards — DuckLake compaction merges but never splits. A
    /// partitioned session rolls each partition's files too, which official does not.
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
        let validation_schema =
            Arc::new(self.validation_schema(schema_name, table_name, arrow_schema)?);
        let columns = arrow_schema_to_column_defs(&validation_schema)?;
        let setup =
            self.metadata
                .begin_write_transaction(schema_name, table_name, &columns, mode)?;
        // Data columns carry their catalog field-ids. When embedding row lineage,
        // append the reserved-field-id rowid column AFTER them; it is a parquet-only
        // column (not a catalog column), so it is absent from `columns`/`column_ids`
        // and the metadata commit never sees it.
        let schema_with_ids = {
            let mut schema = build_schema_with_field_ids(arrow_schema, &setup.field_ids)?;
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
            .set_writer_version(self.writer_version)
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
        let writer = RowGroupSampledWriter::new(staging, schema_with_ids.clone(), props);

        // A partitioned target routes rows through a per-partition sink. The
        // single-file writer above is still created: with zero rows the sink
        // produces no file, and a Replace then needs that 0-row marker to retire the
        // prior generation.
        let table_key = self.table_key(schema_name, table_name)?;
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
                    StreamPartitionMode::Split => Some(PartitionSink {
                        key_names: spec.key_names(),
                        spec,
                        table_key: table_key.clone(),
                        schema_with_ids: schema_with_ids.clone(),
                        props: self.build_writer_props(),
                        target_file_size: self.target_file_size,
                        max_open: self.max_open_partitions,
                        hive_file_pattern: self.hive_file_pattern,
                        open: Vec::new(),
                        uploads: StagedUploads::new(
                            Arc::clone(&self.object_store),
                            &setup.column_ids,
                            self.upload_concurrency,
                        ),
                        staged_values: Vec::new(),
                    }),
                },
            };

        // A partitioned target already rolls inside its per-partition sink, so a
        // second roller would be redundant (and would double-write).
        let roller = if roll && partition_sink.is_none() {
            Some(RollingFileWriter::new(
                table_key.clone(),
                None,
                schema_with_ids.clone(),
                arrow_schema.fields().len(),
                self.build_writer_props(),
                self.target_file_size,
                // Keep `TableWriteSession::file_path` accurate for the first file.
                Some(catalog_path.clone()),
            ))
        } else {
            None
        };

        let rolled = StagedUploads::new(
            Arc::clone(&self.object_store),
            &setup.column_ids,
            self.upload_concurrency,
        );
        Ok(TableWriteSession {
            metadata: Arc::clone(&self.metadata),
            object_store: Arc::clone(&self.object_store),
            object_path,
            table_key,
            schema_name: schema_name.to_string(),
            table_name: table_name.to_string(),
            snapshot_id: setup.snapshot_id,
            base_snapshot_id: setup.base_snapshot_id,
            expected_base_snapshot_id: None,
            table_id: setup.table_id,
            columns,
            column_ids: setup.column_ids,
            field_ids: setup.field_ids,
            schema_with_ids,
            validation_schema,
            writer: Some(writer),
            temp: Some(temp),
            catalog_path,
            path_is_relative,
            mode,
            row_count: 0,
            nan_flags: Vec::new(),
            partition_sink,
            roller,
            rolled,
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
    /// Materialize visible inlined rows as Parquet in one fenced snapshot.
    ///
    /// The caller reads `inlined_data` at `expected_base_snapshot_id`. The
    /// commit registers the resulting data files and ends those exact inlined
    /// row identities atomically. Empty input is a no-op.
    pub async fn flush_inlined_data(
        &self,
        schema_name: &str,
        table_name: &str,
        inlined_data: &[DuckLakeInlinedData],
        expected_base_snapshot_id: i64,
    ) -> Result<Option<WriteResult>> {
        if inlined_data.is_empty() {
            return Ok(None);
        }

        let mut batches = Vec::with_capacity(inlined_data.len());
        let mut rows = Vec::new();
        for inlined in inlined_data {
            if inlined.row_ids.len() != inlined.batch.num_rows()
                || inlined.begin_snapshots.len() != inlined.batch.num_rows()
            {
                return Err(crate::DuckLakeError::InvalidConfig(format!(
                    "inlined row identities for '{}' do not match its {} batch rows",
                    inlined.table_name,
                    inlined.batch.num_rows(),
                )));
            }
            batches.push(inlined.batch.clone());
            rows.extend(inlined.row_ids.iter().map(|row_id| InlinedRowRef {
                table_name: inlined.table_name.clone(),
                row_id: *row_id,
            }));
        }

        let schema = batches[0].schema();
        if batches
            .iter()
            .any(|batch| !batch.schema().contains(schema.as_ref()))
        {
            return Err(crate::DuckLakeError::InvalidConfig(
                "inlined flush batches do not share one compatible schema".to_string(),
            ));
        }

        let write_options = DuckLakeWriteOptions {
            data_inlining_row_limit: Some(0),
            target_file_size: Some(usize::MAX),
            ..DuckLakeWriteOptions::default()
        };
        let flush_writer = self.clone().with_options(&write_options);
        let transaction_options =
            TableWriteOptions::new().with_expected_base_snapshot_id(expected_base_snapshot_id);
        let mut transaction = flush_writer
            .transaction()
            .with_options(&transaction_options);
        transaction
            .stage_write_with_deletes(
                schema_name,
                table_name,
                schema.as_ref(),
                WriteMode::Append,
                &batches,
                &[],
                &rows,
            )
            .await?;
        transaction
            .writes
            .last_mut()
            .expect("flush stages one table")
            .write
            .inlined_flush = true;
        let mut results = transaction.commit().await?;
        if results.len() != 1 {
            return Err(crate::DuckLakeError::Internal(format!(
                "inlined flush committed {} table results, expected 1",
                results.len(),
            )));
        }
        Ok(results.pop())
    }

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

    fn validation_schema(
        &self,
        schema_name: &str,
        table_name: &str,
        incoming_schema: &Schema,
    ) -> Result<Schema> {
        let Some(columns) = self
            .metadata
            .get_table_column_nullability(schema_name, table_name)?
        else {
            return Ok(incoming_schema.clone());
        };
        let nullability: HashMap<String, bool> = columns
            .iter()
            .map(|(name, nullable)| (name.to_lowercase(), *nullable))
            .collect();
        let fields: Vec<Arc<Field>> = incoming_schema
            .fields()
            .iter()
            .map(|field| {
                nullability.get(&field.name().to_lowercase()).map_or_else(
                    || Arc::clone(field),
                    |nullable| Arc::new(field.as_ref().clone().with_nullable(*nullable)),
                )
            })
            .collect();
        Ok(Schema::new_with_metadata(
            fields,
            incoming_schema.metadata().clone(),
        ))
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
            .set_writer_version(self.writer_version)
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
        stream_to_upload(&mut reader, &mut upload).await?;

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
    /// in as parquet field-ids so a read maps them back), including nested field
    /// ids. `stats_column_ids` contains only the top-level catalog ids used to
    /// label per-column statistics. Each batch in `batches` must have the data
    /// columns in order, then a trailing `Int64` rowid column, and — when
    /// `embed_snapshot_id` — a further trailing `Int64` snapshot-id column.
    /// Streams to a local staging file and multipart-uploads it, so peak memory
    /// stays bounded regardless of file size.
    #[allow(clippy::too_many_arguments)]
    pub async fn write_compacted_file(
        &self,
        schema_name: &str,
        table_name: &str,
        data_schema: &Schema,
        data_column_ids: &[i64],
        stats_column_ids: &[i64],
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
            stats_column_ids,
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
        stats_column_ids: &[i64],
        batches: SendableRecordBatchStream,
        embed_snapshot_id: bool,
        partition_subpath: Option<&str>,
    ) -> Result<DataFileInfo> {
        self.write_compacted_file_stream_with_lineage(
            schema_name,
            table_name,
            data_schema,
            data_column_ids,
            stats_column_ids,
            batches,
            true,
            embed_snapshot_id,
            partition_subpath,
        )
        .await
    }

    /// [`Self::write_compacted_file_stream`], choosing whether the output
    /// embeds the rowid column.
    ///
    /// With `embed_rowid` false the batches carry no rowid column (the data
    /// columns, then the snapshot-id column when `embed_snapshot_id`), and the
    /// file's rowids are its catalog `row_id_start` plus each row's position:
    /// the caller must write the rows in rowid order and register the file
    /// with that `row_id_start`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn write_compacted_file_stream_with_lineage(
        &self,
        schema_name: &str,
        table_name: &str,
        data_schema: &Schema,
        data_column_ids: &[i64],
        stats_column_ids: &[i64],
        mut batches: SendableRecordBatchStream,
        embed_rowid: bool,
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
        // embedded rowid column unless the rows' position serves it, and for a
        // merged partial file the snapshot-id column. Neither embedded column is
        // a catalog column.
        let schema_with_ids = {
            let base = build_schema_with_field_ids(data_schema, data_column_ids)?;
            let mut fields: Vec<Field> = base.fields().iter().map(|f| f.as_ref().clone()).collect();
            if embed_rowid {
                fields.push(embedded_rowid_field());
            }
            if embed_snapshot_id {
                fields.push(embedded_snapshot_id_field());
            }
            Arc::new(Schema::new_with_metadata(fields, base.metadata().clone()))
        };

        let mut props_builder = WriterProperties::builder()
            .set_writer_version(self.writer_version)
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
        let mut writer = RowGroupSampledWriter::new(staging, schema_with_ids.clone(), props);
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
            let batch_with_ids = apply_field_ids(&batch, schema_with_ids.clone())?;
            crate::stats_collect::accumulate_nan_flags(
                &mut nan_flags,
                &batch,
                stats_column_ids.len(),
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
        stream_to_upload(&mut reader, &mut upload).await?;

        // Collect stats for catalog columns only. Track NaN values while
        // consuming the stream because the Parquet footer omits that signal.
        let column_stats = crate::stats_collect::collect_column_stats(
            temp.path(),
            stats_column_ids,
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
        let validation_schema = self.validation_schema(schema_name, table_name, arrow_schema)?;
        for (_, batches) in &groups {
            validate_not_null_batches(&validation_schema, batches)?;
        }
        let columns = arrow_schema_to_column_defs(&validation_schema)?;
        let setup =
            self.metadata
                .begin_write_transaction(schema_name, table_name, &columns, mode)?;

        let records_written: usize = groups
            .iter()
            .flat_map(|(_, batches)| batches)
            .map(RecordBatch::num_rows)
            .sum();

        // Validate the caller's assignment against the live spec BEFORE writing
        // anything (inline or Parquet), so a bad one costs no uploads. A wrong
        // arity or an unparseable value would otherwise be persisted and then
        // used as an exact pruning bound, silently dropping rows from later
        // reads. (Whether each group's rows really carry its values is the
        // caller's assertion — as in official DuckLake's add_data_files, that
        // cannot be checked without reading data.)
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
        } else {
            return Err(crate::DuckLakeError::Conflict(
                "partition spec changed before write; retry the write".to_string(),
            ));
        }

        let batches: Vec<RecordBatch> = groups
            .iter()
            .flat_map(|(_, batches)| batches.iter().cloned())
            .collect();
        if self.should_inline(records_written, arrow_schema, &batches) {
            let committed = self.metadata.register_inlined_data(
                setup.table_id,
                schema_name,
                table_name,
                setup.snapshot_id,
                &batches,
                mode,
                setup.base_snapshot_id,
                &columns,
                &setup.field_ids,
                &options.commit_metadata,
                options.expected_base_snapshot_id,
            )?;
            return Ok(WriteResult {
                snapshot_id: committed.snapshot_id,
                table_id: committed.table_id,
                schema_id: committed.schema_id,
                files_written: 0,
                records_written: records_written as i64,
            });
        }
        let schema_with_ids =
            Arc::new(build_schema_with_field_ids(arrow_schema, &setup.field_ids)?);

        let scoped_base = match self.metadata.catalog_id() {
            Some(id) => join_paths(&self.base_key_path, &format!("cat_{id}"))?,
            None => self.base_key_path.clone(),
        };
        let table_key = join_paths(&join_paths(&scoped_base, schema_name)?, table_name)?;

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
            &setup.field_ids,
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
        let validation_schema = self.validation_schema(schema_name, table_name, arrow_schema)?;
        validate_not_null_batches(&validation_schema, batches)?;
        let columns = arrow_schema_to_column_defs(&validation_schema)?;
        let setup =
            self.metadata
                .begin_write_transaction(schema_name, table_name, &columns, mode)?;

        let records_written: usize = batches.iter().map(RecordBatch::num_rows).sum();
        if !resolve_layout && self.metadata.live_partition_spec(setup.table_id)?.is_some() {
            return Err(crate::DuckLakeError::Conflict(
                "table gained a partition spec before write; retry the write".to_string(),
            ));
        }
        if self.should_inline(records_written, arrow_schema, batches) {
            let committed = self.metadata.register_inlined_data(
                setup.table_id,
                schema_name,
                table_name,
                setup.snapshot_id,
                batches,
                mode,
                setup.base_snapshot_id,
                &columns,
                &setup.field_ids,
                &SnapshotCommitMetadata::new(),
                None,
            )?;
            return Ok(WriteResult {
                snapshot_id: committed.snapshot_id,
                table_id: committed.table_id,
                schema_id: committed.schema_id,
                files_written: 0,
                records_written: records_written as i64,
            });
        }
        let schema_with_ids =
            Arc::new(build_schema_with_field_ids(arrow_schema, &setup.field_ids)?);

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
        let sorted_owned: Vec<RecordBatch> = if resolve_layout && self.sort_on_insert {
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
        let batches: &[RecordBatch] = if resolve_layout && self.sort_on_insert {
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
            &setup.field_ids,
        )?;

        Ok(WriteResult {
            snapshot_id: committed.snapshot_id,
            table_id: committed.table_id,
            schema_id: committed.schema_id,
            files_written: file_infos.len(),
            records_written,
        })
    }

    async fn prepare_rows_inner(
        &self,
        schema_name: &str,
        table_name: &str,
        arrow_schema: &Schema,
        mode: WriteMode,
        batches: &[RecordBatch],
        resolve_layout: bool,
    ) -> Result<PreparedTableWrite> {
        let validation_schema = self.validation_schema(schema_name, table_name, arrow_schema)?;
        validate_not_null_batches(&validation_schema, batches)?;
        let columns = arrow_schema_to_column_defs(&validation_schema)?;
        let setup =
            self.metadata
                .begin_write_transaction(schema_name, table_name, &columns, mode)?;
        let records_written: usize = batches.iter().map(RecordBatch::num_rows).sum();

        if records_written == 0 {
            return Err(crate::error::DuckLakeError::InvalidConfig(
                "multi-table row stage requires at least one row".to_string(),
            ));
        }

        if !resolve_layout && self.metadata.live_partition_spec(setup.table_id)?.is_some() {
            return Err(crate::DuckLakeError::Conflict(
                "table gained a partition spec before write; retry the write".to_string(),
            ));
        }
        if self.should_inline(records_written, arrow_schema, batches) {
            return Ok(PreparedTableWrite {
                write: StagedTableWrite {
                    table_id: setup.table_id,
                    schema_name: schema_name.to_string(),
                    table_name: table_name.to_string(),
                    base_snapshot_id: setup.base_snapshot_id,
                    mode,
                    columns,
                    column_ids: setup.field_ids,
                    data: StagedTableData::Inlined(batches.to_vec()),
                    snapshot_id_columns: Vec::new(),
                    positional_deletes: Vec::new(),
                    inlined_deletes: Vec::new(),
                    inlined_flush: false,
                },
                object_paths: Vec::new(),
                files_written: 0,
                records_written: records_written as i64,
            });
        }

        let schema_with_ids =
            Arc::new(build_schema_with_field_ids(arrow_schema, &setup.field_ids)?);
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
        let sorted_owned = if resolve_layout && self.sort_on_insert {
            let lengths = batches
                .iter()
                .map(RecordBatch::num_rows)
                .collect::<Vec<_>>();
            let sorted = crate::sort::sort_batches_by_spec(
                batches.to_vec(),
                arrow_schema,
                self.metadata.live_sort_spec(setup.table_id)?.as_ref(),
            )?;
            reslice_to_lengths(sorted, &lengths)
        } else {
            Vec::new()
        };
        let batches = if resolve_layout && self.sort_on_insert {
            &sorted_owned
        } else {
            batches
        };
        let file_infos = match partition.as_ref() {
            Some(spec) => {
                let output_schema = Arc::new(arrow_schema.clone());
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
        let object_paths = file_infos
            .iter()
            .map(|file| {
                self.staged_object_path(schema_name, table_name, &file.path, file.path_is_relative)
            })
            .collect::<Result<Vec<_>>>()?;
        let records_written = file_infos.iter().map(|file| file.record_count).sum();
        let files_written = file_infos.len();

        Ok(PreparedTableWrite {
            write: StagedTableWrite {
                table_id: setup.table_id,
                schema_name: schema_name.to_string(),
                table_name: table_name.to_string(),
                base_snapshot_id: setup.base_snapshot_id,
                mode,
                columns,
                column_ids: setup.field_ids,
                data: StagedTableData::Files(file_infos),
                snapshot_id_columns: Vec::new(),
                positional_deletes: Vec::new(),
                inlined_deletes: Vec::new(),
                inlined_flush: false,
            },
            object_paths,
            files_written,
            records_written,
        })
    }

    fn staged_object_path(
        &self,
        schema_name: &str,
        table_name: &str,
        path: &str,
        path_is_relative: bool,
    ) -> Result<ObjectPath> {
        object_key(
            &self.table_key(schema_name, table_name)?,
            path,
            path_is_relative,
        )
    }

    /// Object-store key of a table's directory: `{base}/[cat_{id}/]{schema}/{table}`.
    fn table_key(&self, schema_name: &str, table_name: &str) -> Result<String> {
        let scoped_base = match self.metadata.catalog_id() {
            Some(id) => join_paths(&self.base_key_path, &format!("cat_{id}"))?,
            None => self.base_key_path.clone(),
        };
        join_paths(&join_paths(&scoped_base, schema_name)?, table_name)
    }

    fn should_inline(&self, rows: usize, arrow_schema: &Schema, batches: &[RecordBatch]) -> bool {
        rows > 0
            && self
                .data_inlining_row_limit
                .is_some_and(|limit| rows <= limit)
            && self.metadata.supports_data_inlining(arrow_schema)
            && self.metadata.supports_data_inlining_values(batches)
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
            let rel = if self.hive_file_pattern {
                crate::partition::hive_subpath(key_names, values)
            } else {
                String::new()
            };
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
        let data_column_count = schema_with_ids.fields().len();
        let mut roller = RollingFileWriter::new(
            table_key.to_string(),
            rel_prefix.map(str::to_string),
            schema_with_ids,
            data_column_count,
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

impl DuckLakeWriteTransaction<'_> {
    /// Applies commit metadata and a shared table-state precondition.
    #[must_use]
    pub fn with_options(mut self, options: &TableWriteOptions) -> Self {
        self.commit_metadata = options.commit_metadata.clone();
        self.expected_base_snapshot_id = options.expected_base_snapshot_id;
        self
    }

    /// Stages one table write without committing catalog metadata.
    pub async fn stage_write(
        &mut self,
        schema_name: &str,
        table_name: &str,
        arrow_schema: &Schema,
        mode: WriteMode,
        batches: &[RecordBatch],
    ) -> Result<()> {
        if batches.iter().all(|batch| batch.num_rows() == 0) {
            return Ok(());
        }

        let prepared = self
            .writer
            .prepare_rows_inner(schema_name, table_name, arrow_schema, mode, batches, true)
            .await?;
        self.writes.push(prepared);
        Ok(())
    }

    /// Stages one table write with options that apply only to this stage.
    pub async fn stage_write_with_options(
        &mut self,
        schema_name: &str,
        table_name: &str,
        arrow_schema: &Schema,
        mode: WriteMode,
        batches: &[RecordBatch],
        options: &DuckLakeWriteOptions,
    ) -> Result<()> {
        if batches.iter().all(|batch| batch.num_rows() == 0) {
            return Ok(());
        }

        let writer = self.writer.clone().with_options(options);
        let prepared = writer
            .prepare_rows_inner(schema_name, table_name, arrow_schema, mode, batches, true)
            .await?;
        self.writes.push(prepared);
        Ok(())
    }

    /// Stages an inline write whose nullable snapshot columns use the commit snapshot.
    #[allow(clippy::too_many_arguments)]
    pub async fn stage_write_with_snapshot_columns(
        &mut self,
        schema_name: &str,
        table_name: &str,
        arrow_schema: &Schema,
        mode: WriteMode,
        batches: &[RecordBatch],
        options: &DuckLakeWriteOptions,
        snapshot_id_columns: &[&str],
    ) -> Result<()> {
        if batches.iter().all(|batch| batch.num_rows() == 0) {
            return Ok(());
        }

        let writer = self.writer.clone().with_options(options);
        if !writer.should_inline(
            batches.iter().map(RecordBatch::num_rows).sum(),
            arrow_schema,
            batches,
        ) {
            return Err(crate::DuckLakeError::InvalidConfig(
                "commit snapshot columns require an inlined table stage".to_string(),
            ));
        }
        for name in snapshot_id_columns {
            if !arrow_schema
                .fields()
                .iter()
                .any(|field| field.name() == name)
            {
                return Err(crate::DuckLakeError::InvalidConfig(format!(
                    "commit snapshot column '{name}' is not present in the staged schema"
                )));
            }
        }
        let mut prepared = writer
            .prepare_rows_inner(schema_name, table_name, arrow_schema, mode, batches, true)
            .await?;
        prepared.write.snapshot_id_columns = snapshot_id_columns
            .iter()
            .map(|name| (*name).to_string())
            .collect();
        self.writes.push(prepared);
        Ok(())
    }

    /// Stages an inline write with deletes and commit-snapshot columns.
    #[allow(clippy::too_many_arguments)]
    pub async fn stage_write_with_deletes_and_snapshot_columns(
        &mut self,
        schema_name: &str,
        table_name: &str,
        arrow_schema: &Schema,
        mode: WriteMode,
        batches: &[RecordBatch],
        positional_deletes: &[DeleteFileEntry],
        inlined_deletes: &[InlinedRowRef],
        options: &DuckLakeWriteOptions,
        snapshot_id_columns: &[&str],
    ) -> Result<()> {
        if batches.iter().all(|batch| batch.num_rows() == 0) {
            return self.stage_deletes(
                schema_name,
                table_name,
                arrow_schema,
                positional_deletes,
                inlined_deletes,
            );
        }

        if !positional_deletes.is_empty() {
            validate_delete_entries(mode, positional_deletes)?;
        }
        let delete_paths = positional_deletes
            .iter()
            .map(|entry| {
                self.writer.staged_object_path(
                    schema_name,
                    table_name,
                    &entry.delete.path,
                    entry.delete.path_is_relative,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        self.stage_write_with_snapshot_columns(
            schema_name,
            table_name,
            arrow_schema,
            mode,
            batches,
            options,
            snapshot_id_columns,
        )
        .await?;
        let prepared = self
            .writes
            .last_mut()
            .expect("snapshot-column stage appended one table");
        prepared.object_paths.extend(delete_paths);
        prepared.write.positional_deletes = positional_deletes.to_vec();
        prepared.write.inlined_deletes = inlined_deletes.to_vec();
        Ok(())
    }

    /// Stages inserted rows and deletes for one table.
    ///
    /// The transaction takes ownership of the positional delete objects and removes them if the
    /// transaction aborts or its metadata commit fails with nothing committed.
    #[allow(clippy::too_many_arguments)]
    pub async fn stage_write_with_deletes(
        &mut self,
        schema_name: &str,
        table_name: &str,
        arrow_schema: &Schema,
        mode: WriteMode,
        batches: &[RecordBatch],
        positional_deletes: &[DeleteFileEntry],
        inlined_deletes: &[InlinedRowRef],
    ) -> Result<()> {
        if batches.iter().all(|batch| batch.num_rows() == 0) {
            return self.stage_deletes(
                schema_name,
                table_name,
                arrow_schema,
                positional_deletes,
                inlined_deletes,
            );
        }

        if !positional_deletes.is_empty() {
            validate_delete_entries(mode, positional_deletes)?;
        }

        let delete_paths = positional_deletes
            .iter()
            .map(|entry| {
                self.writer.staged_object_path(
                    schema_name,
                    table_name,
                    &entry.delete.path,
                    entry.delete.path_is_relative,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        self.stage_write(schema_name, table_name, arrow_schema, mode, batches)
            .await?;
        let prepared = self
            .writes
            .last_mut()
            .expect("stage_write appended one table");
        prepared.object_paths.extend(delete_paths);
        prepared.write.positional_deletes = positional_deletes.to_vec();
        prepared.write.inlined_deletes = inlined_deletes.to_vec();
        Ok(())
    }

    /// Stages deletes for a table without inserting replacement rows.
    ///
    /// The transaction takes ownership of the positional delete objects and removes them if the
    /// transaction aborts or its metadata commit fails with nothing committed.
    pub fn stage_deletes(
        &mut self,
        schema_name: &str,
        table_name: &str,
        arrow_schema: &Schema,
        positional_deletes: &[DeleteFileEntry],
        inlined_deletes: &[InlinedRowRef],
    ) -> Result<()> {
        validate_delete_entries(WriteMode::Append, positional_deletes)?;
        if positional_deletes.is_empty() && inlined_deletes.is_empty() {
            return Ok(());
        }
        let validation_schema =
            self.writer
                .validation_schema(schema_name, table_name, arrow_schema)?;
        let columns = arrow_schema_to_column_defs(&validation_schema)?;
        let setup = self.writer.metadata.begin_write_transaction(
            schema_name,
            table_name,
            &columns,
            WriteMode::Append,
        )?;
        let object_paths = positional_deletes
            .iter()
            .map(|entry| {
                self.writer.staged_object_path(
                    schema_name,
                    table_name,
                    &entry.delete.path,
                    entry.delete.path_is_relative,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        self.writes.push(PreparedTableWrite {
            write: StagedTableWrite {
                table_id: setup.table_id,
                schema_name: schema_name.to_string(),
                table_name: table_name.to_string(),
                base_snapshot_id: setup.base_snapshot_id,
                mode: WriteMode::Append,
                columns,
                column_ids: setup.field_ids,
                data: StagedTableData::None,
                snapshot_id_columns: Vec::new(),
                positional_deletes: positional_deletes.to_vec(),
                inlined_deletes: inlined_deletes.to_vec(),
                inlined_flush: false,
            },
            object_paths,
            files_written: 0,
            records_written: 0,
        });
        Ok(())
    }

    /// Commits every staged table change in one metadata transaction.
    pub async fn commit(mut self) -> Result<Vec<WriteResult>> {
        if self.writes.is_empty() {
            return Ok(Vec::new());
        }
        let writes = self
            .writes
            .iter()
            .map(|prepared| prepared.write.clone())
            .collect::<Vec<_>>();
        let committed = self.writer.metadata.commit_multi_table(
            &writes,
            &self.commit_metadata,
            self.expected_base_snapshot_id,
        );
        let committed = match committed {
            Ok(committed) => committed,
            Err(e) => {
                // Remove the staged files unless the outcome is unknown: a
                // networked COMMIT that failed may still have applied, and
                // deleting the objects then would leave a committed snapshot
                // pointing at missing files. See `commit_definitely_rolled_back`.
                // A cleanup failure is logged rather than returned, so the caller still
                // sees the commit error it decides a retry on.
                if commit_definitely_rolled_back(&e)
                    && let Err(cleanup) = self.cleanup().await
                {
                    tracing::warn!(error = %cleanup, "failed to remove a file of a rejected commit");
                }
                return Err(e);
            },
        };
        if committed.tables.len() != self.writes.len() {
            return Err(crate::error::DuckLakeError::Internal(format!(
                "multi-table commit returned {} table ids for {} stages",
                committed.tables.len(),
                self.writes.len()
            )));
        }

        Ok(self
            .writes
            .iter()
            .zip(committed.tables)
            .map(|(prepared, table)| WriteResult {
                snapshot_id: committed.snapshot_id,
                table_id: table.table_id,
                schema_id: table.schema_id,
                files_written: prepared.files_written,
                records_written: prepared.records_written,
            })
            .collect())
    }

    /// Removes uploaded files without committing the staged metadata.
    pub async fn abort(mut self) -> Result<()> {
        self.cleanup().await
    }

    async fn cleanup(&mut self) -> Result<()> {
        let paths = self
            .writes
            .iter()
            .flat_map(|prepared| prepared.object_paths.iter().cloned())
            .collect();
        let failures = remove_objects(&self.writer.object_store, paths).await;
        if !failures.is_empty() {
            return Err(crate::error::DuckLakeError::Internal(format!(
                "failed to remove staged files: {}",
                failures.join("; ")
            )));
        }
        Ok(())
    }
}

/// Whether a failed commit is known to have registered nothing: every failure
/// except a `COMMIT` that failed on a networked catalog
/// (see [`MetadataWriter`] for the contract writers keep).
///
/// Official DuckLake removes a transaction's files on any failed commit
/// (`DuckLakeTransactionState::CleanupFiles`), including one whose `COMMIT` failed.
/// Here that one case keeps them: on PostgreSQL or MySQL the server may have
/// applied the `COMMIT` before the connection failed, and removing the files then
/// would leave a committed snapshot naming objects that no longer exist. The price
/// is an orphan when the commit did roll back, left for the orphan sweep, which
/// removes only files no snapshot references.
fn commit_definitely_rolled_back(error: &crate::error::DuckLakeError) -> bool {
    match error {
        #[cfg(any(feature = "metadata-postgres", feature = "metadata-mysql"))]
        crate::error::DuckLakeError::CommitOutcomeUnknown(_) => false,
        _ => true,
    }
}

/// Remove the data and delete files written for a commit that failed, when it
/// definitely registered nothing, and return the commit's error. See
/// [`commit_definitely_rolled_back`] for the one failure that keeps them.
async fn release_after_failed_commit(
    object_store: &Arc<dyn ObjectStore>,
    error: crate::error::DuckLakeError,
    objects: Vec<ObjectPath>,
) -> crate::error::DuckLakeError {
    if commit_definitely_rolled_back(&error) {
        for failure in remove_objects(object_store, objects).await {
            tracing::warn!(error = %failure, "failed to remove a file of a rejected commit");
        }
    }
    error
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
    writer: RowGroupSampledWriter<std::io::BufWriter<std::fs::File>>,
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
/// [`TableWriteSession::write_batch`] synchronous (it hands each file to a background
/// [`StagedUploads`]) while the buffered path awaits each upload in turn.
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
    /// number of batches and any input ordering is preserved *across* files). A
    /// file reports no size until its first row group is complete, so it never
    /// rolls before holding one; see [`crate::write_encoding`].
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
        let batch_with_ids = apply_field_ids(batch, self.schema_with_ids.clone())?;
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

    /// Open the next file. Its encoding is chosen from its first row group; see
    /// [`crate::write_encoding`].
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
        let writer =
            RowGroupSampledWriter::new(staging, self.schema_with_ids.clone(), self.props.clone());
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
#[tracing::instrument(name = "ducklake.finalize_open_file", level = "info", skip_all)]
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
/// partition).
///
/// On a COPY failure the multipart upload is aborted, so no partial object is left.
/// On a FLUSH failure it deliberately is not — `BufWriter::abort` panics once the
/// writer has been shut down — so a rejected or unacknowledged
/// `CompleteMultipartUpload` can leave the upload dangling for a bucket lifecycle
/// rule to reclaim. See [`stream_to_upload`].
#[tracing::instrument(name = "ducklake.upload_staged_file", level = "info", skip_all)]
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
    stream_to_upload(&mut reader, &mut upload).await?;

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
/// partitioned table): keep a writer open per partition seen, and finalize the
/// least-recently-opened one when the number of open files would exceed `max_open`.
/// Unlike it, each partition's file also rolls at `target_file_size`; official
/// DuckLake does not rotate a partitioned insert (`rotate = false` in
/// `ducklake_insert.cpp`). All files produced are committed in ONE
/// snapshot, so a partitioned streaming write is as atomic as an unpartitioned one.
///
/// Each partition's file sequence is a [`RollingFileWriter`] — the same rollover
/// implementation the buffered path uses — so the two cannot drift. This sink differs
/// only in upload policy: `write_batch` must stay synchronous, so rolled and evicted
/// files are handed to a [`StagedUploads`], which uploads them in the background.
#[derive(Debug)]
struct PartitionSink {
    spec: crate::partition::PartitionWriteSpec,
    key_names: Vec<String>,
    /// Object-store key prefix of the table directory; Hive subpaths hang off it.
    table_key: String,
    /// Field-id-tagged schema every written batch carries.
    schema_with_ids: SchemaRef,
    props: WriterProperties,
    target_file_size: usize,
    max_open: usize,
    hive_file_pattern: bool,
    /// One roller per partition with a file in progress, oldest first (eviction takes
    /// from the front). Paired with the partition values its files carry.
    open: Vec<(Vec<Option<String>>, RollingFileWriter)>,
    /// Uploads every finished partition file.
    uploads: StagedUploads,
    /// The partition values of each file handed to `uploads`, in the same order.
    staged_values: Vec<Vec<Option<String>>>,
}

impl PartitionSink {
    /// Split `batch` by partition and write each group to that partition's roller.
    fn write_batch(&mut self, batch: &RecordBatch) -> Result<()> {
        let batch_with_ids = apply_field_ids(batch, self.schema_with_ids.clone())?;
        let groups = crate::partition::split_batches_by_partition(
            &self.schema_with_ids,
            std::slice::from_ref(&batch_with_ids),
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
                    // is complete on disk, and start its upload. That partition
                    // simply gets another file if more of its rows arrive.
                    let (evicted_values, mut evicted) = self.open.remove(0);
                    if let Some(staged) = evicted.finish()? {
                        self.stage(evicted_values, staged);
                    }
                }
                let rel = if self.hive_file_pattern {
                    crate::partition::hive_subpath(&self.key_names, values)
                } else {
                    String::new()
                };
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
                        self.schema_with_ids.fields().len(),
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
            self.stage(partition_values, staged);
            // The roller rolled its file; drop it from `open` unless it already has a
            // fresh one in progress, so the open-file cap counts real open files.
            if !self.open[index].1.has_open_file() {
                self.open.remove(index);
            }
        }
        Ok(())
    }

    /// Hand a finished file of the partition `values` to the uploads.
    fn stage(&mut self, values: Vec<Option<String>>, staged: StagedFile) {
        self.staged_values.push(values);
        self.uploads.add(staged);
    }

    /// Finish every open file and wait for every upload, returning the files to
    /// commit — each [`DataFileInfo`] stamped with its partition.
    async fn finish_uploads(&mut self) -> Result<UploadedFiles> {
        for (values, mut roller) in std::mem::take(&mut self.open) {
            if let Some(staged) = roller.finish()? {
                self.stage(values, staged);
            }
        }
        let mut uploaded = self.uploads.finish().await?;
        // `zip` is positional and truncates silently; `finish` returns one info per
        // file handed in, in the same order `staged_values` was filled, which is
        // what makes this pairing sound. A mismatch would commit files under the
        // wrong partition or drop some, so it fails the write and removes the files.
        if uploaded.infos.len() != self.staged_values.len() {
            for failure in remove_objects(&self.uploads.object_store, uploaded.objects).await {
                tracing::warn!(error = %failure, "failed to remove a data file of a failed write");
            }
            return Err(crate::error::DuckLakeError::Internal(format!(
                "{} partition files were uploaded for {} staged",
                uploaded.infos.len(),
                self.staged_values.len()
            )));
        }
        uploaded.infos = std::mem::take(&mut self.staged_values)
            .into_iter()
            .zip(uploaded.infos)
            .map(|(values, info)| {
                let partition_values: Vec<(i32, Option<String>)> = values
                    .into_iter()
                    .enumerate()
                    .map(|(i, v)| (i as i32, v))
                    .collect();
                info.with_partition(self.spec.partition_id, partition_values)
            })
            .collect();
        Ok(uploaded)
    }
}

/// Uploads a streaming write's finished files while the write goes on, and hands
/// the commit their [`DataFileInfo`]s in write order.
///
/// A file starts uploading as soon as it is finished — rolled, evicted from the
/// open-partition set, or closed by `finish` — with at most `concurrency` uploads in
/// flight; later files wait on local disk in the order they finished. Its local copy
/// is removed when its upload ends, so a writer that waits for room
/// ([`TableWriteSession::write_batch_async`]) holds no more than `concurrency`
/// finished files on disk, rather than the whole output. Official DuckLake writes
/// the same way: its insert is a `COPY ... TO` straight into the table's data path,
/// so files land in storage while the statement runs and only the commit makes
/// them visible.
///
/// Uploads run as tasks on the tokio runtime that is current when the first one
/// starts, so they progress while the caller encodes the next file — including a
/// caller that drives the synchronous `write_batch` from a blocking thread. With no
/// runtime current, finished files wait and are uploaded by [`Self::finish`].
///
/// Order is load-bearing: `register_data_files` walks the list assigning
/// `row_id_start` from a running counter, so results are kept by write position,
/// never by completion order. Official DuckLake assigns row ids in collection order
/// too, which is what keeps the two equivalent.
///
/// No object uploaded here is referenced by any snapshot until the commit, and until
/// [`Self::finish`] hands them to it, this owns them: a failed upload, an abort, or a
/// drop removes every object it started. Removal is best effort; a survivor is an
/// unreferenced orphan, never a file a snapshot names. Once handed over, the session
/// removes them if the commit fails with nothing committed, as official DuckLake
/// does, but keeps them when a `COMMIT` on a networked catalog fails with its
/// outcome unknown, where official removes them; see
/// [`commit_definitely_rolled_back`].
#[derive(Debug)]
struct StagedUploads {
    object_store: Arc<dyn ObjectStore>,
    column_ids: Arc<[i64]>,
    /// How many uploads may be in flight at once.
    concurrency: usize,
    /// Finished files not yet started, oldest first, with their write position.
    waiting: VecDeque<(usize, StagedFile)>,
    /// Started uploads not yet collected, with their write position.
    running: Vec<(usize, tokio::task::JoinHandle<Result<DataFileInfo>>)>,
    /// One entry per file handed in, by write position; filled as uploads finish.
    uploaded: Vec<Option<DataFileInfo>>,
    /// The object key of every upload started: what a cleanup removes.
    started: Vec<ObjectPath>,
    /// The first upload failure, until it is returned to the caller.
    failure: Option<crate::error::DuckLakeError>,
    /// Set by the first upload failure. Nothing starts after it: each upload carries
    /// the object store's own retry budget (minutes per request), so continuing to
    /// start uploads against a failing store turns an outage into an hours-long hang.
    failed: bool,
    /// The runtime the uploads run on; a drop schedules its cleanup there.
    runtime: Option<tokio::runtime::Handle>,
    /// Set once the uploaded objects belong to the commit, or have been removed.
    settled: bool,
}

/// The files a [`StagedUploads`] uploaded, ready to commit.
#[derive(Debug)]
struct UploadedFiles {
    /// One per file, in write order.
    infos: Vec<DataFileInfo>,
    /// The object key of every file, for the caller to remove if the commit is
    /// rejected.
    objects: Vec<ObjectPath>,
}

impl StagedUploads {
    fn new(object_store: Arc<dyn ObjectStore>, column_ids: &[i64], concurrency: usize) -> Self {
        Self {
            object_store,
            column_ids: column_ids.into(),
            concurrency: concurrency.max(1),
            waiting: VecDeque::new(),
            running: Vec::new(),
            uploaded: Vec::new(),
            started: Vec::new(),
            failure: None,
            failed: false,
            runtime: None,
            settled: false,
        }
    }

    /// Queue a finished file and start whatever uploads now fit. Never blocks.
    fn add(&mut self, staged: StagedFile) {
        self.waiting.push_back((self.uploaded.len(), staged));
        self.uploaded.push(None);
        self.advance();
    }

    /// Collect the uploads that have ended and start waiting files in their place.
    /// Never blocks.
    fn advance(&mut self) {
        self.collect_finished();
        self.start_waiting();
    }

    /// Return the upload failure observed since the last call, if any; once one has
    /// been returned, every later call fails too.
    fn check(&mut self) -> Result<()> {
        if let Some(error) = self.failure.take() {
            return Err(error);
        }
        if self.failed {
            return Err(crate::error::DuckLakeError::Internal(
                "an earlier data file upload in this write failed".to_string(),
            ));
        }
        Ok(())
    }

    fn start_waiting(&mut self) {
        if self.failed {
            return;
        }
        if self.runtime.is_none() {
            self.runtime = tokio::runtime::Handle::try_current().ok();
        }
        let Some(runtime) = &self.runtime else {
            return;
        };
        while self.running.len() < self.concurrency {
            let Some((position, staged)) = self.waiting.pop_front() else {
                break;
            };
            self.started.push(staged.object_path.clone());
            let object_store = Arc::clone(&self.object_store);
            let column_ids = Arc::clone(&self.column_ids);
            let upload =
                async move { upload_staged_file(staged, &object_store, &column_ids).await };
            let task = runtime.spawn(tracing::Instrument::in_current_span(upload));
            self.running.push((position, task));
        }
    }

    /// Record every upload that has already ended, without waiting.
    fn collect_finished(&mut self) {
        let mut index = 0;
        while index < self.running.len() {
            // `is_finished` makes the poll below ready, except that tokio's
            // cooperative budget can defer it; the task is then collected later.
            if self.running[index].1.is_finished()
                && let Some(outcome) = (&mut self.running[index].1).now_or_never()
            {
                let (position, _) = self.running.swap_remove(index);
                self.record(position, outcome);
                continue;
            }
            index += 1;
        }
    }

    /// Wait for whichever running upload ends first, and record it.
    async fn collect_next(&mut self) {
        if self.running.is_empty() {
            return;
        }
        let (outcome, index, _) =
            futures::future::select_all(self.running.iter_mut().map(|(_, task)| task)).await;
        let (position, _) = self.running.swap_remove(index);
        self.record(position, outcome);
    }

    fn record(
        &mut self,
        position: usize,
        outcome: std::result::Result<Result<DataFileInfo>, tokio::task::JoinError>,
    ) {
        let outcome = outcome.unwrap_or_else(|join| {
            Err(crate::error::DuckLakeError::Internal(format!(
                "data file upload task failed: {join}"
            )))
        });
        match outcome {
            Ok(info) => self.uploaded[position] = Some(info),
            Err(error) if !self.failed => {
                self.failed = true;
                self.failure = Some(error);
            },
            // Logged rather than dropped: several uploads can fail before the stop
            // takes effect, and a systemic cause (an expired credential, a store
            // that went away) shows as several failures of which only one is
            // returned.
            Err(error) => tracing::warn!(
                error = %error,
                file_index = position,
                "additional upload failure in the same write"
            ),
        }
    }

    /// Wait until no finished file is waiting for an upload slot: on return, at most
    /// `concurrency` finished files remain on local disk.
    async fn wait_for_room(&mut self) -> Result<()> {
        self.advance();
        self.check()?;
        while !self.waiting.is_empty() && !self.running.is_empty() {
            self.collect_next().await;
            self.advance();
            self.check()?;
        }
        Ok(())
    }

    /// Upload every file still waiting, and hand all of them to the commit in write
    /// order.
    ///
    /// On failure, uploads already in flight are awaited — dropping one
    /// mid-multipart strands upload state only a bucket lifecycle rule could
    /// reclaim — files not yet started are dropped, and every object started is
    /// removed. Paths whose upload failed are included: an upload that fails at its
    /// final `CompleteMultipartUpload` can still have created its object.
    #[tracing::instrument(
        name = "ducklake.upload_staged_files",
        level = "info",
        skip_all,
        fields(files = self.uploaded.len(), concurrency = self.concurrency)
    )]
    async fn finish(&mut self) -> Result<UploadedFiles> {
        loop {
            self.advance();
            if self.failed {
                break;
            }
            if self.running.is_empty() {
                if self.waiting.is_empty() {
                    break;
                }
                self.failed = true;
                self.failure = Some(crate::error::DuckLakeError::Internal(
                    "uploading data files requires a tokio runtime".to_string(),
                ));
                break;
            }
            self.collect_next().await;
        }
        if self.failed {
            let error = self.check().err().unwrap_or_else(|| {
                crate::error::DuckLakeError::Internal("a data file upload failed".to_string())
            });
            for failure in self.remove_started().await {
                tracing::warn!(error = %failure, "failed to remove a data file after an aborted upload");
            }
            return Err(error);
        }
        let mut infos = Vec::with_capacity(self.uploaded.len());
        for (position, info) in self.uploaded.drain(..).enumerate() {
            // Unreachable while the loop above only exits with every file recorded
            // or a failure set; checked anyway, because registering fewer files
            // than were written would lose rows with no error.
            let info = info.ok_or_else(|| {
                crate::error::DuckLakeError::Internal(format!(
                    "data file {position} was neither uploaded nor reported failed"
                ))
            })?;
            infos.push(info);
        }
        self.settled = true;
        Ok(UploadedFiles {
            infos,
            objects: std::mem::take(&mut self.started),
        })
    }

    /// Abandon every upload: drop the files not yet started, await the uploads in
    /// flight, then remove every object started. Returns the removals that failed.
    ///
    /// Cancel-safe: an upload leaves `running` only once it has ended, and
    /// `started` is cleared and `settled` set only once the removal has finished,
    /// so a drop part-way through still awaits the rest and removes everything.
    async fn remove_started(&mut self) -> Vec<String> {
        self.waiting.clear();
        while let Some((_, task)) = self.running.last_mut() {
            let _ = task.await;
            self.running.pop();
        }
        let failures = remove_objects(&self.object_store, self.started.clone()).await;
        self.started.clear();
        self.settled = true;
        failures
    }
}

impl Drop for StagedUploads {
    /// A write dropped before its uploads were committed or removed — the caller
    /// stopped after an error, or its future was cancelled — leaves objects no
    /// snapshot will reference. Removing them needs `await`, so it is scheduled on
    /// the runtime the uploads ran on; if that runtime has shut down, the removal
    /// never runs and the objects stay as orphans. [`TableWriteSession::abort`] is
    /// the form that finishes the removal before it returns. Files not yet started
    /// need nothing: their local copies are removed as they drop.
    fn drop(&mut self) {
        if self.settled || self.started.is_empty() {
            return;
        }
        let Some(runtime) = self.runtime.take() else {
            return;
        };
        let running = std::mem::take(&mut self.running);
        let started = std::mem::take(&mut self.started);
        let object_store = Arc::clone(&self.object_store);
        runtime.spawn(async move {
            for (_, task) in running {
                let _ = task.await;
            }
            for failure in remove_objects(&object_store, started).await {
                tracing::warn!(
                    error = %failure,
                    "failed to remove a data file of an abandoned write"
                );
            }
        });
    }
}

/// The object-store key of a catalog file path: relative to `table_key`, or
/// absolute.
fn object_key(table_key: &str, path: &str, path_is_relative: bool) -> Result<ObjectPath> {
    let path = if path_is_relative {
        join_paths(table_key, path)?
    } else {
        path.to_string()
    };
    Ok(ObjectPath::from(path.trim_start_matches('/')))
}

/// Remove `paths` from `object_store`, best effort, returning the removals that
/// failed. A path that is already gone counts as removed: a file whose upload never
/// started was never created.
///
/// `delete_stream`, not a loop of `delete`: its S3 implementation batches into
/// `DeleteObjects` and other backends run deletes concurrently. A sequential loop
/// costs one full retry budget per file precisely when the store is unhealthy.
async fn remove_objects(
    object_store: &Arc<dyn ObjectStore>,
    paths: Vec<ObjectPath>,
) -> Vec<String> {
    let mut failures = Vec::new();
    if paths.is_empty() {
        return failures;
    }
    let to_delete = futures::stream::iter(paths.into_iter().map(Ok));
    let mut deletions = object_store.delete_stream(to_delete.boxed());
    while let Some(outcome) = deletions.next().await {
        match outcome {
            Ok(_)
            | Err(object_store::Error::NotFound {
                ..
            }) => {},
            Err(error) => failures.push(error.to_string()),
        }
    }
    failures
}

/// Streaming write session. Batches stream to local staging files. A rolling or
/// partitioned session uploads each file as soon as it is finished; a single-file
/// session uploads its one file in `finish()`. Nothing is committed before
/// `finish()`, which registers every file in one snapshot. A session dropped or
/// [aborted](Self::abort) without finishing removes its staging files and every
/// object it uploaded, and so does a `finish()` that fails with nothing committed.
/// A `COMMIT` on a networked catalog that fails with its outcome unknown keeps
/// them, where official DuckLake removes them, since the commit may have applied.
/// Top-level column IDs drive statistics and partitions, while recursive field
/// IDs drive catalog rows and Parquet metadata.
#[derive(Debug)]
pub struct TableWriteSession {
    metadata: Arc<dyn MetadataWriter>,
    object_store: Arc<dyn ObjectStore>,
    object_path: ObjectPath,
    /// Object-store key of the table directory, which a relative delete-file path
    /// passed to [`Self::finish_with_deletes`] resolves against.
    table_key: String,
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
    /// Top-level Arrow column generation for this write. Threaded to the metadata
    /// writer at `finish()` so single-catalog backends can flatten and insert the
    /// recursive column rows with `field_ids` at the atomic commit.
    columns: Vec<ColumnDef>,
    column_ids: Vec<i64>,
    field_ids: Vec<i64>,
    schema_with_ids: SchemaRef,
    validation_schema: SchemaRef,
    /// Parquet writer streaming to the local staging file (`temp`). Batches are
    /// written to disk as they arrive rather than buffered in memory, so peak
    /// memory stays bounded by the parquet row-group size regardless of table
    /// size. The finished file is streamed to object storage in `finish()`.
    writer: Option<RowGroupSampledWriter<std::io::BufWriter<std::fs::File>>>,
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
    /// Uploads the files the roller finishes, as it finishes them.
    rolled: StagedUploads,
    commit_metadata: SnapshotCommitMetadata,
}

impl TableWriteSession {
    // Keep the read plan's snapshot for source-scoped conflict checks without
    // enabling the broader table-generation precondition.
    pub(crate) const fn with_base_snapshot_id(mut self, snapshot_id: i64) -> Self {
        self.base_snapshot_id = snapshot_id;
        self
    }

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

    /// Write `batch` to the session's current file.
    ///
    /// A rolling or partitioned session starts uploading each file as soon as it is
    /// finished, in the background on the current tokio runtime; nothing is committed
    /// until [`Self::finish`]. This call never waits for an upload, so when batches
    /// arrive faster than files upload, finished files queue on local disk. Use
    /// [`Self::write_batch_async`] to wait for room instead.
    ///
    /// Returns an upload failure as soon as it is observed. The session must then be
    /// dropped or [aborted](Self::abort), which removes what it uploaded.
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
            match roller.write(batch)? {
                Some(staged) => self.rolled.add(staged),
                None => self.rolled.advance(),
            }
            self.row_count += rows;
            return self.rolled.check();
        }
        if let Some(sink) = &mut self.partition_sink {
            let rows = batch.num_rows() as i64;
            sink.write_batch(batch)?;
            sink.uploads.advance();
            self.row_count += rows;
            return sink.uploads.check();
        }
        if self.writer.is_none() {
            return Err(crate::error::DuckLakeError::Internal(
                "Writer already closed".to_string(),
            ));
        }
        self.validate_batch_schema(batch)?;

        let batch_with_ids = apply_field_ids(batch, self.schema_with_ids.clone())?;
        // Note float-column NaN presence before the batch streams to disk (the
        // footer we later harvest has no NaN flag). Only the catalog data columns.
        crate::stats_collect::accumulate_nan_flags(
            &mut self.nan_flags,
            &batch_with_ids,
            self.schema_with_ids.fields().len(),
        );
        let writer = self.writer.as_mut().unwrap();
        writer.write(&batch_with_ids)?;
        self.row_count += batch.num_rows() as i64;
        Ok(())
    }

    /// Write `batch` like [`Self::write_batch`], then wait until every finished file
    /// has started uploading.
    ///
    /// That bounds local disk: a rolling session then holds at most
    /// `upload_concurrency` finished files plus the one being written, and a
    /// partitioned one at most `upload_concurrency` plus its open partition files
    /// (`max_open_partitions`) — however large the write. A single-file session
    /// uploads only at `finish`, so for it this is `write_batch`.
    pub async fn write_batch_async(&mut self, batch: &RecordBatch) -> Result<()> {
        self.write_batch(batch)?;
        if self.roller.is_some() {
            self.rolled.wait_for_room().await?;
        } else if let Some(sink) = &mut self.partition_sink {
            sink.uploads.wait_for_room().await?;
        }
        Ok(())
    }

    /// Abandon the write without committing, and remove every file it already
    /// uploaded.
    ///
    /// Waits for the uploads in flight, so none lands after the removal. Dropping
    /// the session does the same in the background; this is the form to use when
    /// the caller needs the removal finished, or its failures reported.
    pub async fn abort(mut self) -> Result<()> {
        let mut failures = self.rolled.remove_started().await;
        if let Some(sink) = &mut self.partition_sink {
            failures.extend(sink.uploads.remove_started().await);
        }
        if failures.is_empty() {
            Ok(())
        } else {
            Err(crate::error::DuckLakeError::Internal(format!(
                "failed to remove uploaded data files: {}",
                failures.join("; ")
            )))
        }
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
            if !Self::data_type_contains_ignoring_nested_names(
                expected_field.data_type(),
                batch_field.data_type(),
            ) {
                return Err(crate::error::DuckLakeError::InvalidConfig(format!(
                    "Schema mismatch at column {}: batch has type {:?}, expected {:?}",
                    i,
                    batch_field.data_type(),
                    expected_field.data_type()
                )));
            }
        }
        validate_not_null_batches(&self.validation_schema, std::slice::from_ref(batch))?;
        Ok(())
    }

    fn data_type_contains_ignoring_nested_names(expected: &DataType, actual: &DataType) -> bool {
        match (expected, actual) {
            (DataType::List(expected), DataType::List(actual))
            | (DataType::LargeList(expected), DataType::LargeList(actual))
            | (DataType::ListView(expected), DataType::ListView(actual))
            | (DataType::LargeListView(expected), DataType::LargeListView(actual)) => {
                Self::field_contains_ignoring_name(expected, actual)
            },
            (
                DataType::FixedSizeList(expected, expected_size),
                DataType::FixedSizeList(actual, actual_size),
            ) => {
                expected_size == actual_size && Self::field_contains_ignoring_name(expected, actual)
            },
            (DataType::Map(expected, expected_sorted), DataType::Map(actual, actual_sorted)) => {
                expected_sorted == actual_sorted
                    && Self::field_contains_ignoring_name(expected, actual)
            },
            (DataType::Struct(expected), DataType::Struct(actual)) => {
                expected.len() == actual.len()
                    && expected
                        .iter()
                        .zip(actual.iter())
                        .all(|(expected, actual)| {
                            Self::field_contains_ignoring_name(expected, actual)
                        })
            },
            (
                DataType::Dictionary(expected_key, expected_value),
                DataType::Dictionary(actual_key, actual_value),
            ) => {
                Self::data_type_contains_ignoring_nested_names(expected_key, actual_key)
                    && Self::data_type_contains_ignoring_nested_names(expected_value, actual_value)
            },
            _ => expected.contains(actual),
        }
    }

    fn field_contains_ignoring_name(expected: &Field, actual: &Field) -> bool {
        Self::data_type_contains_ignoring_nested_names(expected.data_type(), actual.data_type())
            && expected.dict_is_ordered() == actual.dict_is_ordered()
            && (expected.is_nullable() || !actual.is_nullable())
            && actual.metadata().iter().all(|(key, value)| {
                expected
                    .metadata()
                    .get(key)
                    .is_some_and(|expected| expected == value)
            })
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

    #[tracing::instrument(name = "ducklake.write_session_finish", level = "info", skip_all)]
    pub async fn finish(mut self) -> Result<WriteResult> {
        // Rolling: finish the in-progress file, upload every file this session
        // produced, and commit them in ONE snapshot.
        if let Some(mut roller) = self.roller.take() {
            if let Some(staged) = roller.finish()? {
                self.rolled.add(staged);
            }
            let uploaded = self.rolled.finish().await?;
            if uploaded.infos.is_empty() {
                // No rows arrived. Fall through to the single-file path, which
                // registers the 0-row marker a Replace needs to retire the prior
                // generation.
                return self.finish_single_file().await;
            }
            let file_infos = uploaded.infos;
            let records_written: i64 = file_infos.iter().map(|f| f.record_count).sum();
            let committed = match self.metadata.register_data_files_with_commit_metadata(
                self.table_id,
                &self.schema_name,
                &self.table_name,
                self.snapshot_id,
                &file_infos,
                self.mode,
                self.base_snapshot_id,
                &self.columns,
                &self.field_ids,
                &self.commit_metadata,
                self.expected_base_snapshot_id,
            ) {
                Ok(committed) => committed,
                Err(e) => {
                    return Err(release_after_failed_commit(
                        &self.object_store,
                        e,
                        uploaded.objects,
                    )
                    .await);
                },
            };
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
        if let Some(mut sink) = self.partition_sink.take() {
            let uploaded = sink.finish_uploads().await?;
            if uploaded.infos.is_empty() {
                // No rows reached any partition. Fall through to the single-file
                // path, which registers the 0-row marker that carries a Replace
                // truncation (and is exempt from the partition fence).
                return self.finish_single_file().await;
            }
            let file_infos = uploaded.infos;
            let records_written: i64 = file_infos.iter().map(|f| f.record_count).sum();
            let committed = match self.metadata.register_data_files_with_commit_metadata(
                self.table_id,
                &self.schema_name,
                &self.table_name,
                self.snapshot_id,
                &file_infos,
                self.mode,
                self.base_snapshot_id,
                &self.columns,
                &self.field_ids,
                &self.commit_metadata,
                self.expected_base_snapshot_id,
            ) {
                Ok(committed) => committed,
                Err(e) => {
                    return Err(release_after_failed_commit(
                        &self.object_store,
                        e,
                        uploaded.objects,
                    )
                    .await);
                },
            };
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
        let (file_info, object) = self.upload_staged().await?;
        // register_data_file returns the ids actually committed (snapshot id
        // assigned at commit; real schema/table ids, which may differ from the
        // begin-time reservations under a concurrent create). Report those.
        let committed = match self.metadata.register_data_file_with_commit_metadata(
            self.table_id,
            &self.schema_name,
            &self.table_name,
            self.snapshot_id,
            &file_info,
            self.mode,
            self.base_snapshot_id,
            &self.columns,
            &self.field_ids,
            &self.commit_metadata,
            self.expected_base_snapshot_id,
        ) {
            Ok(committed) => committed,
            Err(e) => {
                return Err(release_after_failed_commit(&self.object_store, e, vec![object]).await);
            },
        };

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
    ///
    /// The session takes ownership of the delete files: if this call fails with
    /// nothing committed, they are removed with the appended files, as official
    /// DuckLake removes a failed transaction's delete files. Only a `COMMIT` whose
    /// outcome is unknown (`DuckLakeError::CommitOutcomeUnknown`) keeps them.
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
        let delete_objects = deletes
            .iter()
            .map(|entry| {
                object_key(
                    &self.table_key,
                    &entry.delete.path,
                    entry.delete.path_is_relative,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        // Reject an unsupported combination before uploading anything more. A
        // rolling or partitioned session has already uploaded the files it
        // finished, so remove those and the delete files rather than leave them as
        // orphans.
        if let Err(error) = validate_delete_entries(self.mode, deletes) {
            let object_store = Arc::clone(&self.object_store);
            if let Err(cleanup) = self.abort().await {
                tracing::warn!(error = %cleanup, "failed to remove the files of a rejected write");
            }
            for failure in remove_objects(&object_store, delete_objects).await {
                tracing::warn!(error = %failure, "failed to remove a delete file of a rejected write");
            }
            return Err(error);
        }
        // Every failure from here on commits nothing, except a `COMMIT` whose
        // outcome is unknown; the delete files are removed with the appended ones.
        let (file_infos, mut objects) = match self.finish_appended().await {
            Ok(appended) => appended,
            Err(error) => {
                for failure in remove_objects(&self.object_store, delete_objects).await {
                    tracing::warn!(error = %failure, "failed to remove a delete file of a failed write");
                }
                return Err(error);
            },
        };
        objects.extend(delete_objects);
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
                    &self.field_ids,
                    &self.commit_metadata,
                    self.expected_base_snapshot_id,
                ),
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
                    &self.field_ids,
                    &self.commit_metadata,
                    self.expected_base_snapshot_id,
                ),
        };
        let committed = match committed {
            Ok(committed) => committed,
            Err(e) => return Err(release_after_failed_commit(&self.object_store, e, objects).await),
        };
        Ok(WriteResult {
            snapshot_id: committed.snapshot_id,
            table_id: committed.table_id,
            schema_id: committed.schema_id,
            files_written: file_infos.len(),
            records_written,
        })
    }

    /// Finish and upload every appended file, returning each one's
    /// [`DataFileInfo`] and object key. A session with no rows yields the 0-row
    /// single file, whose marker is what carries a Replace truncation.
    async fn finish_appended(&mut self) -> Result<(Vec<DataFileInfo>, Vec<ObjectPath>)> {
        Ok(if let Some(mut sink) = self.partition_sink.take() {
            let uploaded = sink.finish_uploads().await?;
            if uploaded.infos.is_empty() {
                // No rows reached any partition. Fall through to the single-file
                // path, whose 0-row marker is what carries a Replace truncation
                // (and is exempt from the partition fence).
                let (info, object) = self.upload_staged().await?;
                (vec![info], vec![object])
            } else {
                (uploaded.infos, uploaded.objects)
            }
        } else if let Some(mut roller) = self.roller.take() {
            // `finish` writes the last file's parquet footer locally.
            if let Some(staged) = roller.finish()? {
                self.rolled.add(staged);
            }
            let uploaded = self.rolled.finish().await?;
            if uploaded.infos.is_empty() {
                // No rows arrived. Fall through to the single-file path, whose
                // 0-row marker is what carries a Replace truncation (and is exempt
                // from the partition fence) — same behaviour as a non-rolling
                // session.
                let (info, object) = self.upload_staged().await?;
                (vec![info], vec![object])
            } else {
                (uploaded.infos, uploaded.objects)
            }
        } else {
            let (info, object) = self.upload_staged().await?;
            (vec![info], vec![object])
        })
    }

    /// Finalise + upload the staged parquet and return its [`DataFileInfo`] and
    /// object key, leaving the metadata commit to the caller. Shared by
    /// [`finish`](Self::finish) and [`finish_with_deletes`](Self::finish_with_deletes).
    #[tracing::instrument(name = "ducklake.upload_staged", level = "info", skip_all)]
    async fn upload_staged(&mut self) -> Result<(DataFileInfo, ObjectPath)> {
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
        // is no 5 GiB single-PUT ceiling and memory stays bounded. A failure at
        // the final `CompleteMultipartUpload` can still have created the object,
        // so it is removed.
        let local = tokio::fs::File::open(temp.path()).await?;
        let mut reader = tokio::io::BufReader::new(local);
        let mut upload =
            ObjectBufWriter::new(Arc::clone(&self.object_store), self.object_path.clone());
        if let Err(error) = stream_to_upload(&mut reader, &mut upload).await {
            for failure in remove_objects(&self.object_store, vec![self.object_path.clone()]).await
            {
                tracing::warn!(error = %failure, "failed to remove a data file after a failed upload");
            }
            return Err(error.into());
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
        Ok((file_info, self.object_path.clone()))
    }
}

// Drop deletes the staging files, and each `StagedUploads` removes the objects
// it uploaded, so a session abandoned before `finish()` leaves nothing behind.

/// Stream a finished local parquet file to object storage and finalise the
/// upload. `BufWriter` switches to a multipart upload once the payload exceeds
/// its buffer, so files larger than the object store's single-PUT limit (5 GiB
/// on S3) upload fine and memory stays bounded.
async fn stream_to_upload<R>(reader: &mut R, upload: &mut ObjectBufWriter) -> std::io::Result<()>
where
    R: tokio::io::AsyncRead + Unpin + ?Sized,
{
    // The abort decision lives here rather than at each call site, because only ONE
    // of the two failure modes may be aborted and getting it wrong is a panic, not a
    // wrong result: `BufWriter::abort` panics if the writer has already been shut
    // down, and a failing `shutdown` leaves it in exactly that state. So aborting
    // after a flush failure turns a recoverable upload error into a panicking task.
    // Aborting after a copy failure is both safe and necessary — it releases the
    // multipart upload instead of stranding it.
    if let Err(e) = tokio::io::copy(reader, upload).await {
        let _ = upload.abort().await;
        return Err(e);
    }
    upload.shutdown().await
}

/// Read the Parquet Thrift metadata length from the tail of a finished file.
/// The 8-byte length-and-magic trailer is excluded. Stored as the nullable
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

pub(crate) fn validate_not_null_batches(
    target_schema: &Schema,
    batches: &[RecordBatch],
) -> Result<()> {
    for batch in batches {
        if batch.num_columns() < target_schema.fields().len() {
            return Err(crate::error::DuckLakeError::InvalidConfig(format!(
                "Schema mismatch: batch has {} columns, expected at least {}",
                batch.num_columns(),
                target_schema.fields().len()
            )));
        }

        for (field, array) in target_schema.fields().iter().zip(batch.columns()) {
            if !field.is_nullable() && array.null_count() > 0 {
                return Err(crate::error::DuckLakeError::InvalidConfig(format!(
                    "NOT NULL constraint failed: {}",
                    field.name()
                )));
            }
        }
    }
    Ok(())
}

fn build_schema_with_field_ids(schema: &Schema, column_ids: &[i64]) -> Result<Schema> {
    fn with_field_id(field: &Field, column_ids: &[i64], next_id: &mut usize) -> Result<Field> {
        let field_id = column_ids.get(*next_id).copied().ok_or_else(|| {
            crate::error::DuckLakeError::Internal(format!(
                "Missing field id for Arrow field '{}' at recursive position {}",
                field.name(),
                *next_id,
            ))
        })?;
        *next_id += 1;
        let data_type = match field.data_type() {
            DataType::List(child) => DataType::List(Arc::new(
                with_field_id(child, column_ids, next_id)?.with_name("element"),
            )),
            DataType::LargeList(child) => DataType::LargeList(Arc::new(
                with_field_id(child, column_ids, next_id)?.with_name("element"),
            )),
            DataType::FixedSizeList(child, size) => DataType::FixedSizeList(
                Arc::new(with_field_id(child, column_ids, next_id)?.with_name("element")),
                *size,
            ),
            DataType::Struct(children) => DataType::Struct(
                children
                    .iter()
                    .map(|child| with_field_id(child, column_ids, next_id).map(Arc::new))
                    .collect::<Result<Vec<_>>>()?
                    .into(),
            ),
            DataType::Map(entries, sorted) => {
                let DataType::Struct(children) = entries.data_type() else {
                    return Err(crate::error::DuckLakeError::InvalidConfig(
                        "Arrow map entries must be a struct".to_string(),
                    ));
                };
                let entries_type = DataType::Struct(
                    children
                        .iter()
                        .map(|child| with_field_id(child, column_ids, next_id).map(Arc::new))
                        .collect::<Result<Vec<_>>>()?
                        .into(),
                );
                DataType::Map(
                    Arc::new(
                        Field::new("key_value", entries_type, entries.is_nullable())
                            .with_metadata(entries.metadata().clone()),
                    ),
                    *sorted,
                )
            },
            data_type => data_type.clone(),
        };
        let mut metadata: HashMap<String, String> = field.metadata().clone();
        metadata.insert("PARQUET:field_id".to_string(), field_id.to_string());
        Ok(Field::new(field.name(), data_type, field.is_nullable()).with_metadata(metadata))
    }

    let mut next_id = 0;
    let fields = schema
        .fields()
        .iter()
        .map(|field| with_field_id(field, column_ids, &mut next_id))
        .collect::<Result<Vec<_>>>()?;
    if next_id != column_ids.len() {
        return Err(crate::error::DuckLakeError::Internal(format!(
            "Field id count {} exceeds Arrow schema node count {next_id}",
            column_ids.len(),
        )));
    }

    Ok(Schema::new_with_metadata(fields, schema.metadata().clone()))
}

fn apply_field_ids(batch: &RecordBatch, schema: SchemaRef) -> Result<RecordBatch> {
    let columns = batch
        .columns()
        .iter()
        .zip(schema.fields())
        .map(|(column, field)| {
            crate::column_rename::array_with_data_type(column, field.data_type())
                .map_err(crate::error::DuckLakeError::Arrow)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(crate::column_rename::record_batch_with_schema(
        schema, columns,
    )?)
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
        u32::from_le_bytes([footer_bytes[0], footer_bytes[1], footer_bytes[2], footer_bytes[3]]);
    Ok(i64::from(metadata_len))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Decimal128Array, Int32Array, StringArray, StringViewArray, StructArray};
    use arrow::datatypes::DataType;
    use rstest::rstest;

    #[rstest]
    #[case("gzip", "22", Compression::GZIP(GzipLevel::default()))]
    #[case("brotli", "12", Compression::BROTLI(BrotliLevel::default()))]
    #[case("snappy", "-1", Compression::SNAPPY)]
    fn compression_level_only_applies_to_zstd(
        #[case] codec: &str,
        #[case] level: &str,
        #[case] expected: Compression,
    ) {
        let settings = HashMap::from([
            ("parquet_compression".to_string(), codec.to_string()),
            ("parquet_compression_level".to_string(), level.to_string()),
        ]);

        let options = DuckLakeWriteOptions::from_metadata_settings(&settings).unwrap();

        assert_eq!(options.compression, Some(expected));
    }

    #[rstest]
    fn zstd_zero_uses_the_parquet_default_level() {
        let settings = HashMap::from([
            ("parquet_compression".to_string(), "zstd".to_string()),
            ("parquet_compression_level".to_string(), "0".to_string()),
        ]);

        let options = DuckLakeWriteOptions::from_metadata_settings(&settings).unwrap();

        assert_eq!(
            options.compression,
            Some(Compression::ZSTD(ZstdLevel::default()))
        );
    }

    #[rstest]
    fn zstd_without_a_level_uses_the_ducklake_default() {
        let settings = HashMap::from([("parquet_compression".to_string(), "zstd".to_string())]);

        let options = DuckLakeWriteOptions::from_metadata_settings(&settings).unwrap();

        assert_eq!(
            options.compression,
            Some(Compression::ZSTD(ZstdLevel::try_new(3).unwrap()))
        );
    }

    #[rstest]
    fn invalid_write_setting_is_deferred_until_validation() {
        let settings = HashMap::from([
            ("parquet_compression".to_string(), "zstd".to_string()),
            ("parquet_compression_level".to_string(), "23".to_string()),
        ]);

        let options = DuckLakeWriteOptions::from_metadata_settings_deferred(&settings);

        assert!(options.validate().is_err());
    }

    #[rstest]
    fn metadata_options_include_parquet_version_and_rewrite_threshold() {
        let settings = HashMap::from([
            ("parquet_version".to_string(), "V1".to_string()),
            ("rewrite_delete_threshold".to_string(), "0.75".to_string()),
        ]);

        let options = DuckLakeWriteOptions::from_metadata_settings(&settings).unwrap();

        assert_eq!(options.parquet_version, Some(WriterVersion::PARQUET_1_0));
        assert_eq!(options.rewrite_delete_threshold, Some(0.75));
    }

    #[test]
    fn metadata_write_options_apply_ducklake_defaults_and_units() {
        let settings = HashMap::from([
            ("parquet_compression".to_string(), "zstd".to_string()),
            ("parquet_compression_level".to_string(), "5".to_string()),
            (
                "parquet_row_group_size_bytes".to_string(),
                "2 MiB".to_string(),
            ),
            ("target_file_size".to_string(), "5MB".to_string()),
            ("sort_on_insert".to_string(), "false".to_string()),
            ("hive_file_pattern".to_string(), "false".to_string()),
            ("auto_compact".to_string(), "false".to_string()),
        ]);

        let options = DuckLakeWriteOptions::from_metadata_settings(&settings).unwrap();

        assert_eq!(
            options.compression,
            Some(Compression::ZSTD(ZstdLevel::try_new(5).unwrap()))
        );
        assert_eq!(options.data_inlining_row_limit, Some(0));
        assert_eq!(options.max_row_group_rows, Some(122_880));
        assert_eq!(options.max_row_group_bytes, Some(2 * 1_048_576));
        assert_eq!(options.target_file_size, Some(5_000_000));
        assert_eq!(options.max_open_partitions, None);
        assert_eq!(options.sort_on_insert, Some(false));
        assert_eq!(options.hive_file_pattern, Some(false));
        assert_eq!(options.auto_compact, Some(false));
    }

    #[test]
    fn explicit_write_options_override_catalog_settings_per_field() {
        let stored = DuckLakeWriteOptions::from_metadata_settings(&HashMap::from([
            ("parquet_compression".to_string(), "zstd".to_string()),
            ("target_file_size".to_string(), "5MB".to_string()),
        ]))
        .unwrap();
        let explicit = DuckLakeWriteOptions {
            compression: Some(Compression::LZ4_RAW),
            sort_on_insert: Some(false),
            ..Default::default()
        };

        let options = stored.with_overrides(&explicit);

        assert_eq!(options.compression, Some(Compression::LZ4_RAW));
        assert_eq!(options.data_inlining_row_limit, Some(0));
        assert_eq!(options.target_file_size, Some(5_000_000));
        assert_eq!(options.sort_on_insert, Some(false));
        assert_eq!(options.hive_file_pattern, Some(true));
    }

    #[test]
    fn metadata_lz4_uses_the_standard_parquet_codec() {
        let options = DuckLakeWriteOptions::from_metadata_settings(&HashMap::from([(
            "parquet_compression".to_string(),
            "lz4".to_string(),
        )]))
        .unwrap();

        assert_eq!(options.compression, Some(Compression::LZ4_RAW));
    }

    #[test]
    fn test_validate_not_null_batches_names_top_level_column() {
        let batch_schema = Arc::new(Schema::new(vec![Field::new(
            "required",
            DataType::Int32,
            true,
        )]));
        let batch = RecordBatch::try_new(
            batch_schema,
            vec![Arc::new(Int32Array::from(vec![Some(1), None]))],
        )
        .unwrap();
        let target_schema = Schema::new(vec![Field::new("required", DataType::Int32, false)]);

        let error = validate_not_null_batches(&target_schema, &[batch]).unwrap_err();

        assert_eq!(
            error.to_string(),
            "Invalid configuration: NOT NULL constraint failed: required"
        );
    }

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
        let schema_with_ids = build_schema_with_field_ids(&schema, &column_ids).unwrap();

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
    fn test_build_schema_with_nested_field_ids() {
        let map = DataType::Map(
            Arc::new(Field::new(
                "entries",
                DataType::Struct(
                    vec![
                        Arc::new(Field::new("key", DataType::Utf8, false)),
                        Arc::new(Field::new(
                            "value",
                            DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
                            true,
                        )),
                    ]
                    .into(),
                ),
                false,
            )),
            false,
        );
        let schema = Schema::new(vec![Field::new("attrs", map, true)]);

        let schema = build_schema_with_field_ids(&schema, &[10, 11, 12, 13]).unwrap();
        let root = schema.field(0);
        assert_eq!(root.metadata().get("PARQUET:field_id"), Some(&"10".into()));
        let DataType::Map(entries, false) = root.data_type() else {
            panic!("expected map");
        };
        assert!(!entries.metadata().contains_key("PARQUET:field_id"));
        let DataType::Struct(children) = entries.data_type() else {
            panic!("expected entries struct");
        };
        assert_eq!(
            children[0].metadata().get("PARQUET:field_id"),
            Some(&"11".into())
        );
        assert_eq!(
            children[1].metadata().get("PARQUET:field_id"),
            Some(&"12".into())
        );
        let DataType::List(element) = children[1].data_type() else {
            panic!("expected list value");
        };
        assert_eq!(
            element.metadata().get("PARQUET:field_id"),
            Some(&"13".into())
        );
    }

    #[test]
    fn test_apply_field_ids_rewrites_nested_field_metadata() {
        let fields = vec![
            Arc::new(Field::new("amount", DataType::Decimal128(38, 16), false)),
            Arc::new(Field::new("currency", DataType::Utf8View, false)),
        ];
        let values = StructArray::new(
            fields.clone().into(),
            vec![
                Arc::new(
                    Decimal128Array::from(vec![1, 2])
                        .with_precision_and_scale(38, 16)
                        .unwrap(),
                ),
                Arc::new(StringViewArray::from(vec!["USD", "EUR"])),
            ],
            None,
        );
        let schema = Arc::new(Schema::new(vec![Field::new(
            "money",
            DataType::Struct(fields.into()),
            false,
        )]));
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(values)]).unwrap();
        let schema_with_ids = Arc::new(build_schema_with_field_ids(&schema, &[1, 2, 3]).unwrap());

        let rewritten = apply_field_ids(&batch, schema_with_ids.clone()).unwrap();

        assert_eq!(rewritten.schema(), schema_with_ids);
        let DataType::Struct(fields) = rewritten.column(0).data_type() else {
            panic!("expected struct");
        };
        assert_eq!(
            fields[0].metadata().get("PARQUET:field_id"),
            Some(&"2".to_string())
        );
        assert_eq!(
            fields[1].metadata().get("PARQUET:field_id"),
            Some(&"3".to_string())
        );
        let mut writer = ArrowWriter::try_new(Vec::new(), schema_with_ids, None).unwrap();
        writer.write(&rewritten).unwrap();
    }

    #[test]
    fn build_schema_with_field_ids_rejects_missing_recursive_id() {
        let schema = Schema::new(vec![Field::new(
            "items",
            DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
            true,
        )]);

        let error = build_schema_with_field_ids(&schema, &[10]).unwrap_err();

        assert_eq!(
            error.to_string(),
            "Internal error: Missing field id for Arrow field 'item' at recursive position 1",
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
        let schema_with_ids = Arc::new(build_schema_with_field_ids(&schema, &column_ids).unwrap());

        let props = WriterProperties::builder()
            .set_writer_version(parquet::file::properties::WriterVersion::PARQUET_2_0)
            .build();
        let mut writer =
            ArrowWriter::try_new(Vec::new(), schema_with_ids.clone(), Some(props)).unwrap();

        let batch_with_ids = apply_field_ids(&batch, schema_with_ids).unwrap();
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
        let schema_with_ids = Arc::new(build_schema_with_field_ids(&batch.schema(), &[1]).unwrap());
        let mut writer =
            ArrowWriter::try_new(Vec::new(), schema_with_ids.clone(), Some(props)).unwrap();

        let batch_with_ids = apply_field_ids(&batch, schema_with_ids).unwrap();
        writer.write(&batch_with_ids).unwrap();
        let buffer = writer.into_inner().unwrap();

        let footer_size = calculate_footer_size_from_bytes(&buffer).unwrap();
        let tail = &buffer[buffer.len() - 8..];
        let metadata_len = i64::from(u32::from_le_bytes(tail[..4].try_into().unwrap()));

        assert_eq!(&tail[4..], b"PAR1");
        assert_eq!(footer_size, metadata_len);
    }

    /// Holds every write for a while, so uploads stay in flight long enough for a
    /// writer to outrun them.
    #[cfg(feature = "write-sqlite")]
    #[derive(Debug)]
    struct SlowStore(Arc<dyn ObjectStore>);

    #[cfg(feature = "write-sqlite")]
    impl std::fmt::Display for SlowStore {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "SlowStore")
        }
    }

    #[cfg(feature = "write-sqlite")]
    #[async_trait::async_trait]
    impl ObjectStore for SlowStore {
        async fn put_opts(
            &self,
            location: &ObjectPath,
            payload: object_store::PutPayload,
            opts: object_store::PutOptions,
        ) -> object_store::Result<object_store::PutResult> {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            self.0.put_opts(location, payload, opts).await
        }

        async fn put_multipart_opts(
            &self,
            location: &ObjectPath,
            opts: object_store::PutMultipartOptions,
        ) -> object_store::Result<Box<dyn object_store::MultipartUpload>> {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            self.0.put_multipart_opts(location, opts).await
        }

        async fn get_opts(
            &self,
            location: &ObjectPath,
            options: object_store::GetOptions,
        ) -> object_store::Result<object_store::GetResult> {
            self.0.get_opts(location, options).await
        }

        fn delete_stream(
            &self,
            locations: futures::stream::BoxStream<'static, object_store::Result<ObjectPath>>,
        ) -> futures::stream::BoxStream<'static, object_store::Result<ObjectPath>> {
            self.0.delete_stream(locations)
        }

        fn list(
            &self,
            prefix: Option<&ObjectPath>,
        ) -> futures::stream::BoxStream<'static, object_store::Result<object_store::ObjectMeta>>
        {
            self.0.list(prefix)
        }

        async fn list_with_delimiter(
            &self,
            prefix: Option<&ObjectPath>,
        ) -> object_store::Result<object_store::ListResult> {
            self.0.list_with_delimiter(prefix).await
        }

        async fn copy_opts(
            &self,
            from: &ObjectPath,
            to: &ObjectPath,
            options: object_store::CopyOptions,
        ) -> object_store::Result<()> {
            self.0.copy_opts(from, to, options).await
        }
    }

    /// `write_batch_async` keeps the finished files on local disk to the ones
    /// uploading — at most `upload_concurrency` — however far the writer outruns
    /// the store, while `write_batch` lets them queue. Every finished file is
    /// either waiting or uploading until its upload ends and its local copy is
    /// removed, so these two sets are exactly the finished files on disk; with the
    /// file being written, a rolling session holds at most `upload_concurrency + 1`.
    #[cfg(feature = "write-sqlite")]
    #[tokio::test(flavor = "multi_thread")]
    async fn waiting_for_room_bounds_the_finished_files_on_disk() {
        use crate::metadata_writer_sqlite::SqliteMetadataWriter;
        const CONCURRENCY: usize = 2;

        let dir = tempfile::tempdir().unwrap();
        let conn = format!("sqlite:{}?mode=rwc", dir.path().join("t.db").display());
        let data = dir.path().join("data");
        std::fs::create_dir_all(&data).unwrap();
        let writer = SqliteMetadataWriter::new_with_init(&conn).await.unwrap();
        writer.set_data_path(data.to_str().unwrap()).unwrap();
        let store: Arc<dyn ObjectStore> = Arc::new(SlowStore(Arc::new(
            object_store::local::LocalFileSystem::new(),
        )));
        let table_writer = DuckLakeTableWriter::new(Arc::new(writer), store)
            .unwrap()
            .with_target_file_size(MINIMUM_TARGET_FILE_SIZE)
            .with_max_row_group_rows(64)
            .with_upload_concurrency(CONCURRENCY);
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let batch = |b: i32| {
            RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(Int32Array::from_iter_values(b * 1000..(b + 1) * 1000))],
            )
            .unwrap()
        };

        // Waiting for room: nothing ever waits, and the uploads fill their slots.
        let mut session = table_writer
            .clone()
            .with_options(&v1_options())
            .begin_write("main", "bounded", schema.as_ref(), WriteMode::Append)
            .unwrap();
        let mut most_running = 0;
        for b in 0..60 {
            session.write_batch_async(&batch(b)).await.unwrap();
            assert!(
                session.rolled.waiting.is_empty(),
                "batch {b}: a finished file waits"
            );
            assert!(session.rolled.running.len() <= CONCURRENCY, "batch {b}");
            most_running = most_running.max(session.rolled.running.len());
        }
        assert_eq!(
            most_running, CONCURRENCY,
            "the writer must outrun the store"
        );
        let result = session.finish().await.unwrap();
        assert!(
            result.files_written > 2 * CONCURRENCY,
            "got {}",
            result.files_written
        );

        // Not waiting: the writer outruns the store and finished files queue.
        let mut session = table_writer
            .with_options(&v1_options())
            .begin_write("main", "queued", schema.as_ref(), WriteMode::Append)
            .unwrap();
        let mut most_waiting = 0;
        for b in 0..60 {
            session.write_batch(&batch(b)).unwrap();
            assert!(session.rolled.running.len() <= CONCURRENCY, "batch {b}");
            most_waiting = most_waiting.max(session.rolled.waiting.len());
        }
        assert!(most_waiting > 0, "write_batch must not wait for an upload");
        session.finish().await.unwrap();
    }

    /// Parquet V1, so a mostly-distinct column is not delta-encoded to almost
    /// nothing and a small write still rolls.
    #[cfg(feature = "write-sqlite")]
    fn v1_options() -> DuckLakeWriteOptions {
        DuckLakeWriteOptions {
            parquet_version: Some(WriterVersion::PARQUET_1_0),
            ..Default::default()
        }
    }
}
