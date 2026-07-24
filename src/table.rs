//! DuckLake table provider implementation

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::Result;
use crate::column_rename::ColumnRenameExec;
use crate::delete_filter::DeleteFilterExec;
use crate::metadata_provider::{
    DuckLakeFileColumnStatistics, DuckLakeFileData, DuckLakeFileMetadata, DuckLakeStatistics,
    DuckLakeTableColumn, DuckLakeTableColumnStatistics, DuckLakeTableFile,
    FILE_METADATA_BATCH_SIZE, MetadataProvider,
};
use crate::nan_pruning_barrier::NanPruningBarrierExec;
use crate::partition::PartitionSpec;
use crate::path_resolver::resolve_path;
use crate::positional_source::PositionalFileSource;
use crate::row_id::{
    FileRowNumberExec, ROW_ID_PARQUET_FIELD_ID, ROW_POS_COLUMN_NAME, ROWID_COLUMN_NAME, RowIdExec,
    SNAPSHOT_ID_PARQUET_FIELD_ID, rowid_field,
};
use crate::snapshot_filter::SnapshotFilterExec;
use crate::types::{
    build_arrow_schema, build_read_schema_with_field_id_mapping, extract_parquet_field_ids,
};

#[cfg(feature = "write")]
use crate::delete_exec::DuckLakeDeleteExec;
#[cfg(feature = "write")]
use crate::insert_exec::DuckLakeInsertExec;
#[cfg(feature = "write")]
use crate::metadata_writer::{MetadataWriter, WriteMode};
#[cfg(feature = "write")]
use crate::update_exec::DuckLakeUpdateExec;
#[cfg(feature = "write")]
use arrow::array::ArrayRef;
use datafusion::common::DFSchema;
use datafusion::common::pruning::PrunableStatistics;
#[cfg(feature = "write")]
use datafusion::logical_expr::Operator;
#[cfg(feature = "write")]
use datafusion::physical_expr::PhysicalExpr;
#[cfg(feature = "write")]
use datafusion::physical_expr::expressions::BinaryExpr;
use datafusion::physical_optimizer::pruning::PruningPredicate;

#[cfg(feature = "encryption")]
use crate::encryption::EncryptionFactoryBuilder;
use arrow::array::{Array, BooleanArray, Int64Array};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use async_trait::async_trait;
use datafusion::catalog::{Session, TableProvider};
use datafusion::common::stats::Precision;
use datafusion::common::{ColumnStatistics, ScalarValue, Statistics};
use datafusion::datasource::listing::PartitionedFile;
use datafusion::datasource::memory::MemorySourceConfig;
use datafusion::datasource::physical_plan::parquet::{ParquetAccessPlan, RowGroupAccess};
use datafusion::datasource::physical_plan::{FileGroup, FileScanConfigBuilder, ParquetSource};
use datafusion::datasource::source::DataSourceExec;
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::execution::object_store::ObjectStoreUrl;
#[cfg(feature = "write")]
use datafusion::logical_expr::dml::InsertOp;
use datafusion::logical_expr::{Expr, TableProviderFilterPushDown, TableType};
use datafusion::physical_plan::ExecutionPlan;
use futures::StreamExt;
use object_store::path::Path as ObjectPath;
use parquet::arrow::ParquetRecordBatchStreamBuilder;
use parquet::arrow::async_reader::ParquetObjectReader;

#[cfg(feature = "encryption")]
use datafusion::execution::parquet_encryption::EncryptionFactory;

// Delete file schema constants (public for testing)
pub const DELETE_FILE_PATH_COL: &str = "file_path";
pub const DELETE_POS_COL: &str = "pos";

/// Parquet field-id DuckLake's own `ducklake` extension assigns to a positional
/// delete file's `file_path` column (its `FILENAME` virtual column). We stamp it
/// on the delete files we WRITE so DuckDB can read our deletes back. This is the
/// DuckDB id (`i32::MAX - 1`), NOT Iceberg's positional-delete id `2147483546`.
pub const DELETE_FILE_PATH_FIELD_ID: i32 = 2_147_483_646;
/// Parquet field-id DuckLake assigns to a positional delete file's `pos` column
/// (its `FILE_ROW_NUMBER`/ordinal virtual column) — the DuckDB id (`i32::MAX -
/// 2`), NOT Iceberg's `2147483545`. See [`DELETE_FILE_PATH_FIELD_ID`].
pub const DELETE_POS_FIELD_ID: i32 = 2_147_483_645;

/// Build a `PARQUET:field_id` field-metadata map for the given reserved id.
fn parquet_field_id_metadata(field_id: i32) -> HashMap<String, String> {
    HashMap::from([("PARQUET:field_id".to_string(), field_id.to_string())])
}

/// Validate and convert file_size_bytes from i64 (as stored in DuckLake metadata) to u64.
///
/// DuckLake stores file sizes as signed integers in SQL. A negative value indicates
/// corrupt or invalid metadata. Without this check, a negative i64 cast to u64 would
/// wrap to a huge value (e.g., -1 becomes u64::MAX), causing confusing downstream errors.
pub(crate) fn validated_file_size(file_size_bytes: i64, file_path: &str) -> DataFusionResult<u64> {
    u64::try_from(file_size_bytes).map_err(|_| {
        DataFusionError::Execution(format!(
            "Invalid file_size_bytes ({}) for file '{}': value must be non-negative",
            file_size_bytes, file_path
        ))
    })
}

/// Validate and convert record_count from i64 (as stored in DuckLake metadata) to u64.
///
/// DuckLake stores record counts as signed integers in SQL. A negative value indicates
/// corrupt or invalid metadata. Without this check, a negative record_count would cause
/// incorrect behavior (e.g., empty ranges in full-file deletes, or incorrect row filtering).
pub(crate) fn validated_record_count(record_count: i64, file_path: &str) -> DataFusionResult<u64> {
    u64::try_from(record_count).map_err(|_| {
        DataFusionError::Execution(format!(
            "Invalid record_count ({}) for file '{}': value must be non-negative",
            record_count, file_path
        ))
    })
}

fn statistic_usize(value: i64, statistic: &str) -> Option<usize> {
    match usize::try_from(value) {
        Ok(value) => Some(value),
        Err(_) => {
            tracing::warn!(
                value,
                statistic,
                "Ignoring invalid negative DuckLake statistic"
            );
            None
        },
    }
}

fn decode_hex(value: &str) -> Option<Vec<u8>> {
    let compact: String = value.chars().filter(|c| *c != '-').collect();
    if !compact.len().is_multiple_of(2) {
        return None;
    }
    compact
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let pair = std::str::from_utf8(pair).ok()?;
            u8::from_str_radix(pair, 16).ok()
        })
        .collect()
}

/// Decode DuckLake's string representation for min/max statistics into a
/// scalar whose type exactly matches the Arrow field.
fn parse_statistic_scalar(
    value: &str,
    column: &DuckLakeTableColumn,
    data_type: &DataType,
) -> Option<ScalarValue> {
    let ducklake_type = column.column_type.trim().to_ascii_lowercase();

    // These types either have no scalar min/max in DuckLake or use
    // `extra_stats`, which DataFusion's ColumnStatistics cannot represent.
    if ducklake_type.starts_with("list")
        || ducklake_type.starts_with("array")
        || ducklake_type.starts_with("struct")
        || ducklake_type.starts_with("map")
        || matches!(
            ducklake_type.as_str(),
            "geometry"
                | "point"
                | "linestring"
                | "polygon"
                | "multipoint"
                | "multilinestring"
                | "multipolygon"
                | "geometrycollection"
                | "linestring z"
                | "timetz"
                | "time with time zone"
                | "interval"
        )
    {
        return None;
    }

    // Arrow has no representation for DuckDB's infinite date/timestamp
    // sentinels, so leave that bound unknown.
    if matches!(
        value.to_ascii_lowercase().as_str(),
        "infinity" | "-infinity"
    ) {
        return None;
    }

    let parsed = match data_type {
        DataType::Boolean => match value {
            "0" | "false" => Some(ScalarValue::Boolean(Some(false))),
            "1" | "true" => Some(ScalarValue::Boolean(Some(true))),
            _ => None,
        },
        DataType::Utf8 => Some(ScalarValue::Utf8(Some(value.to_string()))),
        DataType::LargeUtf8 => Some(ScalarValue::LargeUtf8(Some(value.to_string()))),
        DataType::Utf8View => Some(ScalarValue::Utf8View(Some(value.to_string()))),
        DataType::Binary => decode_hex(value).map(|value| ScalarValue::Binary(Some(value))),
        DataType::LargeBinary => {
            decode_hex(value).map(|value| ScalarValue::LargeBinary(Some(value)))
        },
        DataType::BinaryView => decode_hex(value).map(|value| ScalarValue::BinaryView(Some(value))),
        DataType::FixedSizeBinary(size) => decode_hex(value)
            .filter(|value| value.len() == *size as usize)
            .map(|value| ScalarValue::FixedSizeBinary(*size, Some(value))),
        DataType::List(_)
        | DataType::LargeList(_)
        | DataType::FixedSizeList(_, _)
        | DataType::Struct(_)
        | DataType::Map(_, _) => None,
        _ => ScalarValue::try_from_string(value.to_string(), data_type).ok(),
    };

    if parsed.is_none() {
        tracing::debug!(
            column = %column.column_name,
            ducklake_type = %column.column_type,
            value,
            "Ignoring DuckLake statistic that could not be decoded"
        );
    }
    parsed
}

/// Whether a stored float `max_value` is a usable upper bound.
///
/// Catalog min/max exclude NaN, and NaN sorts above every value in both DuckDB
/// and DataFusion (IEEE 754 totalOrder) — so a float column whose NaN state is
/// unknown (`None`, e.g. register-by-reference loads) or positive (`Some(true)`,
/// e.g. stats written by official DuckLake's INSERT) may hold values above its
/// recorded max, and pruning `x > C` on that max would wrongly drop rows. The
/// recorded min needs no such gate: NaN can never sit below it, so `min <= v`
/// holds for every value including NaN. Mirrors official DuckLake, which ORs
/// `contains_nan` into its greater-than filter SQL.
fn float_max_is_bound(data_type: &DataType, contains_nan: Option<bool>) -> bool {
    !matches!(
        data_type,
        DataType::Float16 | DataType::Float32 | DataType::Float64
    ) || contains_nan == Some(false)
}

fn scalar_precision(
    value: Option<&str>,
    column: &DuckLakeTableColumn,
    data_type: &DataType,
    exact: bool,
) -> Precision<ScalarValue> {
    match value.and_then(|value| parse_statistic_scalar(value, column, data_type)) {
        Some(value) if exact => Precision::Exact(value),
        Some(value) => Precision::Inexact(value),
        None => Precision::Absent,
    }
}

fn file_row_count(
    file: &DuckLakeTableFile,
    file_columns: Option<&HashMap<i64, DuckLakeFileColumnStatistics>>,
) -> Precision<usize> {
    let gross = file.max_row_count.or_else(|| {
        file_columns.and_then(|columns| columns.values().find_map(|stats| stats.value_count))
    });
    let Some(gross) = gross.and_then(|value| statistic_usize(value, "record_count")) else {
        return Precision::Absent;
    };

    if file.delete_file.is_some() {
        let Some(deleted) = file
            .delete_count
            .and_then(|value| statistic_usize(value, "delete_count"))
        else {
            return Precision::Absent;
        };
        gross
            .checked_sub(deleted)
            .map(Precision::Exact)
            .unwrap_or(Precision::Absent)
    } else {
        Precision::Exact(gross)
    }
}

fn build_datafusion_statistics(
    schema: &Schema,
    columns: &[DuckLakeTableColumn],
    table_files: &[DuckLakeTableFile],
    catalog: DuckLakeStatistics,
    use_current_table_statistics: bool,
    file_metadata_complete: bool,
) -> (Statistics, HashMap<i64, Arc<Statistics>>) {
    let table_column_rows: HashMap<i64, DuckLakeTableColumnStatistics> = catalog
        .columns
        .into_iter()
        .map(|stats| (stats.column_id, stats))
        .collect();
    let mut file_column_rows: HashMap<i64, HashMap<i64, DuckLakeFileColumnStatistics>> =
        HashMap::new();
    for stats in catalog.files {
        file_column_rows
            .entry(stats.data_file_id)
            .or_default()
            .insert(stats.column_id, stats);
    }

    let mut file_statistics = HashMap::with_capacity(table_files.len());
    for file in table_files {
        let raw_columns = file_column_rows.get(&file.data_file_id);
        let has_deletes = file.delete_file.is_some();
        let mut statistics = Statistics::new_unknown(schema);
        statistics.num_rows = file_row_count(file, raw_columns);
        statistics.total_byte_size =
            statistic_usize(file.file.file_size_bytes, "data_file.file_size_bytes")
                .map(Precision::Inexact)
                .unwrap_or(Precision::Absent);

        for (index, column) in columns.iter().enumerate() {
            let Some(raw) = raw_columns.and_then(|stats| stats.get(&column.column_id)) else {
                continue;
            };
            let field_type = schema.field(index).data_type();
            let exact = !has_deletes;
            let column_statistics = &mut statistics.column_statistics[index];
            column_statistics.null_count = raw
                .null_count
                .and_then(|value| statistic_usize(value, "file_column_stats.null_count"))
                .map(|value| {
                    if exact {
                        Precision::Exact(value)
                    } else {
                        Precision::Inexact(value)
                    }
                })
                .unwrap_or(Precision::Absent);
            column_statistics.min_value =
                scalar_precision(raw.min_value.as_deref(), column, field_type, exact);
            column_statistics.max_value = if float_max_is_bound(field_type, raw.contains_nan) {
                scalar_precision(raw.max_value.as_deref(), column, field_type, exact)
            } else {
                Precision::Absent
            };
            column_statistics.byte_size = raw
                .column_size_bytes
                .and_then(|value| statistic_usize(value, "file_column_stats.column_size_bytes"))
                .map(Precision::Inexact)
                .unwrap_or(Precision::Absent);
        }

        file_statistics.insert(file.data_file_id, Arc::new(statistics));
    }

    let mut table_statistics = Statistics::new_unknown(schema);

    // Per-file row counts are snapshot-aware and exact when all required
    // counts are present. Fall back to the approximate current-table counter.
    let mut row_total = Some(0usize);
    for file in table_files {
        let value = file_row_count(file, file_column_rows.get(&file.data_file_id));
        row_total = match (row_total, value.get_value()) {
            (Some(total), Some(value)) => total.checked_add(*value),
            _ => None,
        };
    }
    table_statistics.num_rows = if file_metadata_complete && let Some(rows) = row_total {
        Precision::Exact(rows)
    } else if use_current_table_statistics {
        catalog
            .table
            .as_ref()
            .and_then(|stats| stats.record_count)
            .and_then(|value| statistic_usize(value, "table_stats.record_count"))
            .map(Precision::Exact)
            .unwrap_or(Precision::Absent)
    } else {
        Precision::Absent
    };

    // DuckLake stores compressed file bytes while DataFusion describes Arrow
    // output bytes, so this value is necessarily an estimate.
    table_statistics.total_byte_size = if use_current_table_statistics {
        catalog
            .table
            .as_ref()
            .and_then(|stats| stats.file_size_bytes)
            .and_then(|value| statistic_usize(value, "table_stats.file_size_bytes"))
            .map(Precision::Inexact)
            .unwrap_or_else(|| fallback_table_byte_size(table_files))
    } else {
        fallback_table_byte_size(table_files)
    };

    let any_deletes = table_files.iter().any(|file| file.delete_file.is_some());
    for (index, column) in columns.iter().enumerate() {
        let field_type = schema.field(index).data_type();
        let output = &mut table_statistics.column_statistics[index];

        // Table-column rows are not snapshot-versioned. Only use them for the
        // current table generation, and mark bounds inexact because deletes can
        // leave conservative (wider) bounds behind.
        if use_current_table_statistics && let Some(raw) = table_column_rows.get(&column.column_id)
        {
            if raw.contains_null == Some(false) {
                output.null_count = Precision::Exact(0);
            }
            output.min_value = scalar_precision(
                raw.min_value.as_deref(),
                column,
                field_type,
                raw.bounds_are_exact,
            );
            output.max_value = if float_max_is_bound(field_type, raw.contains_nan) {
                scalar_precision(
                    raw.max_value.as_deref(),
                    column,
                    field_type,
                    raw.bounds_are_exact,
                )
            } else {
                Precision::Absent
            };
            output.byte_size = raw
                .column_size_bytes
                .and_then(|value| statistic_usize(value, "file_column_stats.column_size_bytes"))
                .map(Precision::Inexact)
                .unwrap_or(Precision::Absent);
        }

        if !file_metadata_complete {
            continue;
        }

        if table_files.is_empty() {
            output.null_count = Precision::Exact(0);
            output.byte_size = Precision::Exact(0);
            continue;
        }

        let mut null_total = Some(0usize);
        let mut byte_total = Some(0usize);
        let mut min_value: Option<ScalarValue> = None;
        let mut max_value: Option<ScalarValue> = None;
        let mut min_complete = true;
        let mut max_complete = true;

        for file in table_files {
            let Some(raw) = file_column_rows
                .get(&file.data_file_id)
                .and_then(|stats| stats.get(&column.column_id))
            else {
                null_total = None;
                byte_total = None;
                min_complete = false;
                max_complete = false;
                continue;
            };

            null_total = match (
                null_total,
                raw.null_count
                    .and_then(|value| statistic_usize(value, "file_column_stats.null_count")),
            ) {
                (Some(total), Some(value)) => total.checked_add(value),
                _ => None,
            };
            byte_total = match (
                byte_total,
                raw.column_size_bytes.and_then(|value| {
                    statistic_usize(value, "file_column_stats.column_size_bytes")
                }),
            ) {
                (Some(total), Some(value)) => total.checked_add(value),
                _ => None,
            };

            let all_null =
                matches!((raw.value_count, raw.null_count), (Some(v), Some(n)) if v == n);
            match raw
                .min_value
                .as_deref()
                .and_then(|value| parse_statistic_scalar(value, column, field_type))
            {
                Some(value) => {
                    min_value = match min_value {
                        Some(current) => current.partial_cmp(&value).map(|ordering| {
                            if ordering.is_le() {
                                current
                            } else {
                                value
                            }
                        }),
                        None => Some(value),
                    };
                    min_complete &= min_value.is_some();
                },
                None if all_null => {},
                None => min_complete = false,
            }
            // An unusable float max (NaN state unknown/positive) is treated as
            // absent: with `all_null` it contributes nothing, otherwise it
            // poisons `max_complete` so the aggregate max degrades to unknown.
            let usable_max = raw
                .max_value
                .as_deref()
                .filter(|_| float_max_is_bound(field_type, raw.contains_nan));
            match usable_max.and_then(|value| parse_statistic_scalar(value, column, field_type)) {
                Some(value) => {
                    max_value = match max_value {
                        Some(current) => current.partial_cmp(&value).map(|ordering| {
                            if ordering.is_ge() {
                                current
                            } else {
                                value
                            }
                        }),
                        None => Some(value),
                    };
                    max_complete &= max_value.is_some();
                },
                None if all_null => {},
                None => max_complete = false,
            }
        }

        if let Some(value) = null_total {
            output.null_count = if any_deletes {
                Precision::Inexact(value)
            } else {
                Precision::Exact(value)
            };
        }
        if let Some(value) = byte_total {
            output.byte_size = Precision::Inexact(value);
        }
        if min_complete && let Some(value) = min_value {
            output.min_value = if any_deletes {
                Precision::Inexact(value)
            } else {
                Precision::Exact(value)
            };
        }
        if max_complete && let Some(value) = max_value {
            output.max_value = if any_deletes {
                Precision::Inexact(value)
            } else {
                Precision::Exact(value)
            };
        }
    }

    (table_statistics, file_statistics)
}

fn fallback_table_byte_size(table_files: &[DuckLakeTableFile]) -> Precision<usize> {
    let data_bytes: i128 = table_files
        .iter()
        .map(|file| i128::from(file.file.file_size_bytes))
        .sum();
    let delete_bytes: i128 = table_files
        .iter()
        .filter_map(|file| file.delete_file.as_ref())
        .map(|file| i128::from(file.file_size_bytes))
        .sum();
    usize::try_from((data_bytes - delete_bytes).max(0))
        .map(Precision::Inexact)
        .unwrap_or(Precision::Absent)
}

/// Returns the expected schema for DuckLake delete files
///
/// Delete files have a standard schema: (file_path: VARCHAR, pos: INT64).
/// The file_path column records which data file the positions belong to (only
/// `pos` is consumed on read; the catalog already maps delete->data file). Both
/// fields carry DuckLake's reserved parquet field-ids
/// ([`DELETE_FILE_PATH_FIELD_ID`], [`DELETE_POS_FIELD_ID`]) so that delete files
/// WE write are readable by DuckDB's `ducklake` extension. Reads match by column
/// name, so the ids are inert on the read path (files without them still read).
pub fn delete_file_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new(DELETE_FILE_PATH_COL, DataType::Utf8, false)
            .with_metadata(parquet_field_id_metadata(DELETE_FILE_PATH_FIELD_ID)),
        Field::new(DELETE_POS_COL, DataType::Int64, false)
            .with_metadata(parquet_field_id_metadata(DELETE_POS_FIELD_ID)),
    ]))
}

/// Cached schema mapping for renamed columns
type SchemaMapping = (SchemaRef, HashMap<String, String>);

/// Per-file read configuration computed for the row-lineage scan path.
///
/// Encapsulates the decision made by `DuckLakeMultiFileReader::GetVirtualColumnExpression`
/// in the C++ extension: either the parquet file embeds a row-id column
/// (UPDATE/compaction case — surviving rowids preserved across file rewrite),
/// or it doesn't (INSERT-only case — synthesize from `row_id_start + position`).
#[derive(Debug, Clone)]
struct FileReadConfig {
    /// Schema we pass to `ParquetSource::new` for this file. When
    /// `embedded_rowid_parquet_name` is `Some`, this schema has the embedded
    /// rowid column appended at the end (under its parquet name).
    read_schema: SchemaRef,
    /// Parquet-name → user-facing-name renames. Includes the rowid rename
    /// (parquet column → `"rowid"`) when the file has an embedded column with
    /// a different name.
    name_mapping: HashMap<String, String>,
    /// `Some(parquet_column_name)` if the file embeds the rowid column
    /// (tagged with [`ROW_ID_PARQUET_FIELD_ID`]); `None` otherwise.
    embedded_rowid_parquet_name: Option<String>,
    /// `Some(parquet_column_name)` if the file embeds the per-row snapshot-id
    /// column (tagged with [`SNAPSHOT_ID_PARQUET_FIELD_ID`]) — i.e. it is a
    /// merged partial file; `None` otherwise. Not added to `read_schema` (that
    /// would shift the embedded-rowid column off the end, which several call
    /// sites rely on); the partial-file read path appends it explicitly.
    ///
    /// [`SNAPSHOT_ID_PARQUET_FIELD_ID`]: crate::row_id::SNAPSHOT_ID_PARQUET_FIELD_ID
    embedded_snapshot_parquet_name: Option<String>,
    /// True if the file carries a data column (parquet field-id) that is NOT in
    /// the table's CURRENT schema — i.e. a column dropped since the file was
    /// written. Reads null-drop it harmlessly, but compaction must NOT merge such
    /// a file: merged output is written at the current schema, so the dropped
    /// column's data would be lost (and its sources removed). `merge_adjacent_files`
    /// skips any group containing one.
    drops_current_columns: bool,
    /// Per-row-group starting physical row position (prefix sums of
    /// `row_groups[i].num_rows()`). `row_group_starts[i]` is the 0-based file
    /// position of the first row of row group `i`. Used to build row-group-
    /// aligned scan partitions whose starting position is known at plan time,
    /// so `FileRowNumberExec` can synthesize true physical positions instead of
    /// counting stream arrivals. The Parquet footer is the source of truth; the
    /// catalog does not store per-row-group counts.
    row_group_starts: Vec<i64>,
    /// Number of row groups in the file (`row_group_starts.len()`). Required to
    /// build a `ParquetAccessPlan` of the correct length.
    row_group_count: usize,
}

/// DuckLake table provider
///
/// Represents a table within a DuckLake schema and provides access to data via Parquet files.
/// Caches snapshot_id and uses it to load all metadata atomically.
///
/// `Clone` shares the `file_read_config_cache` (it is `Arc`-wrapped): a clone is
/// a cheap handle over the same cached parquet metadata. `delete_from` clones the
/// table into the returned `DuckLakeDeleteExec` so the delete work runs at
/// `execute` time (never at plan/EXPLAIN time).
#[derive(Clone)]
pub struct DuckLakeTable {
    #[allow(dead_code)]
    table_id: i64,
    table_name: String,
    #[allow(dead_code)]
    provider: Arc<dyn MetadataProvider>,
    /// Snapshot this table was opened at. Threaded to the delete-commit path as
    /// the `base_snapshot` (the generation the resolved positions were read
    /// against) for conflict diagnostics.
    #[cfg_attr(not(feature = "write"), allow(dead_code))]
    snapshot_id: i64,
    /// Object store URL for resolving file paths (e.g., s3://bucket/ or file:///)
    object_store_url: Arc<ObjectStoreUrl>,
    /// Table path for resolving relative file paths
    table_path: String,
    /// User-facing schema. Equals `physical_schema` when row lineage is off, or
    /// `physical_schema` with a `rowid` BIGINT appended at the end when on.
    schema: SchemaRef,
    /// Schema of the physical (parquet-backed) columns only — no rowid.
    physical_schema: SchemaRef,
    /// When true, `schema` includes a trailing `rowid` column and `scan()`
    /// injects it per-file via [`RowIdExec`].
    row_lineage: bool,
    /// Column metadata from DuckLake (needed for field_id mapping)
    columns: Vec<DuckLakeTableColumn>,
    /// Table-level statistics for the physical schema.
    table_statistics: Statistics,
    /// The table's active partition spec at `snapshot_id`, if any. Loaded once at
    /// construction and used by `scan()` to synthesize per-file min/max bounds
    /// (from each file's partition values) for partition pruning. `None` for an
    /// unpartitioned table, a catalog without partition support, or a table whose
    /// spec has changed (the provider returns `None` in that case to stay safe).
    partition_spec: Option<PartitionSpec>,
    /// Per-file row-lineage read config, populated lazily on the rowid scan
    /// path. Each file requires its own parquet metadata read to detect an
    /// embedded `_ducklake_internal_row_id` column; we memoize so repeated
    /// scans don't re-fetch. `Arc`-wrapped so a cloned table (see `delete_from`)
    /// shares the same memoized configs.
    file_read_config_cache: Arc<std::sync::Mutex<HashMap<String, Arc<FileReadConfig>>>>,
    /// Encryption factory for the metadata page currently being planned.
    #[cfg(feature = "encryption")]
    encryption_factory: Arc<std::sync::Mutex<Option<Arc<dyn EncryptionFactory>>>>,
    /// Schema name (needed for write operations)
    #[cfg(feature = "write")]
    schema_name: Option<String>,
    /// Metadata writer for write operations (when write feature is enabled)
    #[cfg(feature = "write")]
    writer: Option<Arc<dyn MetadataWriter>>,
}

impl std::fmt::Debug for DuckLakeTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DuckLakeTable")
            .field("table_id", &self.table_id)
            .field("table_name", &self.table_name)
            .field("table_path", &self.table_path)
            .field("schema", &self.schema)
            .field("columns", &self.columns)
            .finish_non_exhaustive()
    }
}

impl DuckLakeTable {
    /// Create a new DuckLake table
    pub fn new(
        table_id: i64,
        table_name: impl Into<String>,
        provider: Arc<dyn MetadataProvider>,
        snapshot_id: i64, // Received from schema
        object_store_url: Arc<ObjectStoreUrl>,
        table_path: String,
    ) -> Result<Self> {
        // File metadata is deliberately deferred until scan(), where it can be
        // consumed and pruned in bounded pages.
        let columns = provider.get_table_structure(table_id, snapshot_id)?;
        // Active partition spec (if any) for pruning. Loaded once at the bound
        // snapshot; `None` for unpartitioned tables or catalogs without partitions.
        let partition_spec = provider.get_partition_spec(table_id, snapshot_id)?;
        let physical_schema = Arc::new(build_arrow_schema(&columns)?);
        let schema = physical_schema.clone();
        let catalog_statistics = provider.get_table_summary_statistics(table_id, snapshot_id)?;
        // `ducklake_table_stats` and `ducklake_table_column_stats` describe the
        // current table generation. They must not be applied to an older
        // snapshot if a newer commit landed after the catalog was opened.
        let use_current_table_statistics = provider.get_current_snapshot()? == snapshot_id;
        let (table_statistics, _) = build_datafusion_statistics(
            physical_schema.as_ref(),
            &columns,
            &[],
            catalog_statistics,
            use_current_table_statistics,
            false,
        );

        // Build encryption factory from file encryption keys (when encryption feature is enabled)
        #[cfg(feature = "encryption")]
        let encryption_factory = Arc::new(std::sync::Mutex::new(None));

        Ok(Self {
            table_id,
            table_name: table_name.into(),
            provider,
            snapshot_id,
            object_store_url,
            table_path,
            schema,
            physical_schema,
            row_lineage: false,
            columns,
            table_statistics,
            partition_spec,
            #[cfg(feature = "encryption")]
            encryption_factory,
            file_read_config_cache: Arc::new(std::sync::Mutex::new(HashMap::new())),
            #[cfg(feature = "write")]
            schema_name: None,
            #[cfg(feature = "write")]
            writer: None,
        })
    }

    /// Enable / disable the row-lineage feature. When enabled, the table's
    /// public schema includes a trailing `rowid` BIGINT column synthesized
    /// from each row's catalog-recorded `row_id_start + position_in_file`.
    pub fn with_row_lineage(mut self, enabled: bool) -> Self {
        self.row_lineage = enabled;
        self.schema = if enabled {
            let mut fields: Vec<Arc<Field>> =
                self.physical_schema.fields().iter().cloned().collect();
            fields.push(Arc::new(rowid_field()));
            Arc::new(Schema::new(fields))
        } else {
            self.physical_schema.clone()
        };
        self
    }

    /// Index of the synthetic `rowid` column in `self.schema`, when enabled.
    fn rowid_index(&self) -> Option<usize> {
        self.row_lineage
            .then(|| self.physical_schema.fields().len())
    }

    /// The table's live data files (each with its catalog `data_file_id`, any
    /// live delete file, and that delete file's `delete_file_id`) at the snapshot
    /// this table was opened at. The positional-delete flow iterates these: for
    /// each, [`Self::resolve_positions`] finds the rows to delete,
    /// [`Self::read_delete_file_positions`] reads the already-deleted set, and
    /// the union is written back via `set_delete_file` (CAS on `delete_file_id`).
    pub fn files(&self) -> Result<Vec<DuckLakeTableFile>> {
        let files = self
            .provider
            .get_table_files_for_select(self.table_id, self.snapshot_id)?;
        #[cfg(feature = "encryption")]
        self.configure_encryption_factory(&files)?;
        Ok(files)
    }

    /// Resolve a file path (data or delete file) to its absolute path
    fn resolve_file_path(&self, file: &DuckLakeFileData) -> DataFusionResult<String> {
        resolve_path(&self.table_path, &file.path, file.path_is_relative)
            .map_err(|e| DataFusionError::External(Box::new(e)))
    }

    /// Build a DataFusion file descriptor and attach the catalog's file-level
    /// statistics. `include_rowid` adds an unknown trailing statistic for an
    /// embedded rowid column so the vector still matches the scan schema.
    fn partitioned_data_file(
        &self,
        table_file: &DuckLakeTableFile,
        include_rowid: bool,
        file_statistics: &HashMap<i64, Arc<Statistics>>,
    ) -> DataFusionResult<PartitionedFile> {
        let resolved_path = self.resolve_file_path(&table_file.file)?;
        let mut file = PartitionedFile::new(
            &resolved_path,
            validated_file_size(table_file.file.file_size_bytes, &resolved_path)?,
        );
        if let Some(footer_size) = table_file.file.footer_size
            && footer_size > 0
            && let Ok(hint) = usize::try_from(footer_size)
        {
            file = file.with_metadata_size_hint(hint);
        }
        if let Some(statistics) = file_statistics.get(&table_file.data_file_id) {
            let statistics = if include_rowid {
                let mut statistics = statistics.as_ref().clone();
                statistics
                    .column_statistics
                    .push(ColumnStatistics::new_unknown());
                Arc::new(statistics)
            } else {
                Arc::clone(statistics)
            };
            file = file.with_statistics(statistics);
        }
        Ok(file)
    }

    /// Plan-time file pruning: return the subset of `self.table_files` whose
    /// catalog column statistics (min/max/null counts) prove they *may* contain
    /// rows matching `filters`. Files with no recorded statistics are always
    /// kept.
    ///
    /// This complements the parquet opener's execution-time `FilePruner`, which
    /// applies the same statistics to skip *reading* non-matching files. Pruning
    /// here shrinks the physical plan itself (file count and the size/row
    /// estimates aggregated from the surviving files' statistics) so downstream
    /// join and aggregation planning sees only the relevant files.
    ///
    /// Purely an optimisation. Only static predicates are evaluated, so the
    /// result can only ever be a *superset* of the truly-matching files — it
    /// never drops a file that could match. On an empty filter set, or any error
    /// while building the predicate, every file is kept.
    ///
    /// Effectiveness note: a file with a live delete file has `Inexact`
    /// statistics (its recorded min/max may cover already-deleted rows). Because
    /// `PrunableStatistics` only prunes on `Exact` bounds, such a file is always
    /// kept — and, since the bounds are aggregated per column across all files,
    /// one `Inexact` file suppresses pruning by that column for the whole scan.
    /// Pruning is therefore most effective on append-only / delete-free files.
    /// Inject partition-derived min/max bounds into per-file statistics so the
    /// existing pruning path ([`Self::prune_table_files`]) can drop partition files
    /// that cannot match a predicate on a partition column.
    ///
    /// For each file's `partition_values`, map the value through the active spec to
    /// a source-column min/max envelope ([`crate::partition::PartitionTransform::source_bounds`]
    /// — `identity` exact, `year` a range; other transforms contribute nothing and
    /// the file is kept). Only fills a column bound the catalog left `Absent`, so
    /// real parquet-derived statistics (tighter, and already prunable) are always
    /// preserved. A NULL partition value or an unmappable column is skipped. A bound
    /// is marked `Exact` only when `min == max` (a genuine single-value extreme);
    /// a widened envelope stays `Inexact` so it can never corrupt MIN/MAX-from-
    /// statistics. Either way the envelope satisfies `min <= every row <= max`, so a
    /// file is never wrongly dropped.
    fn apply_partition_bounds(
        &self,
        table_files: &[DuckLakeTableFile],
        file_statistics: &mut HashMap<i64, Arc<Statistics>>,
    ) {
        let Some(spec) = self.partition_spec.as_ref() else {
            return;
        };
        // Only prune when the spec's key→column mapping is known to apply to every
        // live file (a single spec generation ever). After a re-partition a file's
        // values could belong to a retired generation with a different key order,
        // so mapping them through the current spec could mis-prune — skip pruning
        // then (the write path still uses the live spec via `insert_into`).
        if !spec.prune_safe {
            return;
        }
        for file in table_files {
            if file.partition_values.is_empty() {
                continue;
            }
            let mut updates: Vec<(usize, ScalarValue, ScalarValue)> = Vec::new();
            for (key_index, value) in &file.partition_values {
                let Some(value) = value.as_deref() else {
                    continue; // NULL partition value: cannot bound, keep the file.
                };
                let Some(column) = spec
                    .columns
                    .iter()
                    .find(|c| c.partition_key_index == *key_index)
                else {
                    continue;
                };
                let Some(index) = self
                    .columns
                    .iter()
                    .position(|c| c.column_id == column.column_id)
                else {
                    continue;
                };
                let data_type = self.physical_schema.field(index).data_type();
                if let Some((min, max)) = column.transform.source_bounds(value, data_type) {
                    updates.push((index, min, max));
                }
            }
            if updates.is_empty() {
                continue;
            }
            let Some(stats) = file_statistics.get_mut(&file.data_file_id) else {
                continue;
            };
            let stats = Arc::make_mut(stats);
            for (index, min, max) in updates {
                let Some(column_statistics) = stats.column_statistics.get_mut(index) else {
                    continue;
                };
                // Mark the bound `Exact` ONLY when the file holds a single value for
                // the column (`min == max`, e.g. `identity` or an integer `year`
                // column) — then min/max ARE the true extremes, safe both for pruning
                // (`PrunableStatistics` prunes only on `Exact`) and for MIN/MAX-from-
                // statistics. A widened envelope (e.g. `year` on a timestamp) is
                // `Inexact`: it must never be treated as the exact extreme, or
                // `SELECT max(ts)` could be answered from the year boundary. `Inexact`
                // does not prune, so widened partition bounds rely on real column
                // statistics (which DuckLake writers produce) for pruning.
                let exact = min == max;
                if matches!(column_statistics.min_value, Precision::Absent) {
                    column_statistics.min_value = if exact {
                        Precision::Exact(min)
                    } else {
                        Precision::Inexact(min)
                    };
                }
                if matches!(column_statistics.max_value, Precision::Absent) {
                    column_statistics.max_value = if exact {
                        Precision::Exact(max)
                    } else {
                        Precision::Inexact(max)
                    };
                }
            }
        }
    }

    fn prune_table_files<'a>(
        &self,
        pruning: Option<&PruningPredicate>,
        table_files: &'a [DuckLakeTableFile],
        file_statistics: &HashMap<i64, Arc<Statistics>>,
    ) -> Vec<&'a DuckLakeTableFile> {
        let Some(pruning) = pruning else {
            return table_files.iter().collect();
        };
        match self.file_pruning_mask(pruning, table_files, file_statistics) {
            Ok(mask) => {
                debug_assert_eq!(mask.len(), table_files.len());
                table_files
                    .iter()
                    .zip(mask)
                    .filter_map(|(tf, keep)| keep.then_some(tf))
                    .collect()
            },
            Err(e) => {
                tracing::debug!(error = %e, "skipping plan-time file pruning");
                table_files.iter().collect()
            },
        }
    }

    fn file_pruning_predicate(
        &self,
        state: &dyn Session,
        filters: &[Expr],
    ) -> DataFusionResult<Option<PruningPredicate>> {
        use datafusion::logical_expr::utils::conjunction;

        let Some(expr) = conjunction(filters.iter().cloned()) else {
            return Ok(None);
        };
        let df_schema = DFSchema::try_from(self.physical_schema.as_ref().clone())?;
        let predicate = state.create_physical_expr(expr, &df_schema)?;
        PruningPredicate::try_new(predicate, Arc::clone(&self.physical_schema)).map(Some)
    }

    /// Build a `PruningPredicate` from `filters` and evaluate it against every
    /// file's catalog statistics, returning a keep/drop mask 1:1 with
    /// `self.table_files` (`true` = keep). Filters and statistics are both keyed
    /// to `physical_schema` (the parquet-backed columns, excluding the synthetic
    /// rowid), matching how `file_statistics` is indexed.
    fn file_pruning_mask(
        &self,
        pruning: &PruningPredicate,
        table_files: &[DuckLakeTableFile],
        file_statistics: &HashMap<i64, Arc<Statistics>>,
    ) -> DataFusionResult<Vec<bool>> {
        // A file lacking recorded statistics (e.g. written before statistics
        // were produced) contributes `new_unknown`, which the predicate treats
        // as "cannot prune" — the file is kept.
        let per_file: Vec<Arc<Statistics>> = table_files
            .iter()
            .map(|tf| {
                file_statistics
                    .get(&tf.data_file_id)
                    .map(Arc::clone)
                    .unwrap_or_else(|| Arc::new(Statistics::new_unknown(&self.physical_schema)))
            })
            .collect();
        let stats = PrunableStatistics::new(per_file, Arc::clone(&self.physical_schema));
        pruning.prune(&stats)
    }

    /// Create a ParquetSource with encryption support if enabled and needed
    fn create_parquet_source(&self, schema: SchemaRef) -> ParquetSource {
        #[cfg(feature = "encryption")]
        if let Some(factory) = self.encryption_factory.lock().unwrap().as_ref().cloned() {
            return ParquetSource::new(schema).with_encryption_factory(factory);
        }
        ParquetSource::new(schema)
    }

    #[cfg(feature = "encryption")]
    fn configure_encryption_factory(&self, table_files: &[DuckLakeTableFile]) -> Result<()> {
        let mut builder = EncryptionFactoryBuilder::new();
        for table_file in table_files {
            let resolved_path = resolve_path(
                &self.table_path,
                &table_file.file.path,
                table_file.file.path_is_relative,
            )?;
            builder.add_file(&resolved_path, table_file.file.encryption_key.as_deref());
            if let Some(delete_file) = &table_file.delete_file {
                let path = resolve_path(
                    &self.table_path,
                    &delete_file.path,
                    delete_file.path_is_relative,
                )?;
                builder.add_file(&path, delete_file.encryption_key.as_deref());
            }
        }
        let factory = builder.build();
        *self.encryption_factory.lock().unwrap() = factory
            .has_encrypted_files()
            .then(|| Arc::new(factory) as Arc<dyn EncryptionFactory>);
        Ok(())
    }

    /// Compute the field_id -> physical-name read schema and rename mapping for a
    /// SINGLE file. Physical column names can differ across files (e.g. a column
    /// renamed after some files were written), so this is resolved per file.
    async fn file_schema_mapping(
        &self,
        state: &dyn Session,
        file: &DuckLakeFileData,
    ) -> DataFusionResult<SchemaMapping> {
        let resolved_path = self.resolve_file_path(file)?;
        let object_store = state
            .runtime_env()
            .object_store(self.object_store_url.as_ref())?;
        let object_path = ObjectPath::from(resolved_path.as_str());

        let reader = ParquetObjectReader::new(object_store, object_path);

        // Build the ParquetRecordBatchStreamBuilder with decryption if needed
        #[cfg(feature = "encryption")]
        let builder = {
            use parquet::arrow::arrow_reader::ArrowReaderOptions;

            // Check if file has encryption key
            let options = if let Some(ref key) = file.encryption_key {
                if !key.is_empty() {
                    let key_bytes = crate::encryption::DuckLakeEncryptionFactory::decode_key(key)?;
                    let decryption_props =
                        parquet::encryption::decrypt::FileDecryptionProperties::builder(key_bytes)
                            .build()
                            .map_err(|e| {
                                DataFusionError::Execution(format!(
                                    "Failed to create decryption properties: {}",
                                    e
                                ))
                            })?;
                    ArrowReaderOptions::new().with_file_decryption_properties(decryption_props)
                } else {
                    ArrowReaderOptions::new()
                }
            } else {
                ArrowReaderOptions::new()
            };

            ParquetRecordBatchStreamBuilder::new_with_options(reader, options)
                .await
                .map_err(|e| DataFusionError::External(Box::new(e)))?
        };

        #[cfg(not(feature = "encryption"))]
        let builder = ParquetRecordBatchStreamBuilder::new(reader)
            .await
            .map_err(|e| DataFusionError::External(Box::new(e)))?;

        let field_id_map = extract_parquet_field_ids(builder.metadata());

        // No field_ids means external file - use current schema directly
        if field_id_map.is_empty() {
            return Ok((self.schema.clone(), HashMap::new()));
        }

        let (read_schema, name_mapping) = build_read_schema_with_field_id_mapping(
            &self.columns,
            &field_id_map,
            Some(builder.schema().as_ref()),
        )
        .map_err(|e| DataFusionError::External(Box::new(e)))?;

        Ok((Arc::new(read_schema), name_mapping))
    }

    /// Scan `data_file` and return the physical positions of rows matching
    /// `predicate`, without applying delete files. These are the positions used
    /// by a delete file's `pos` column and
    /// [`crate::metadata_writer::MetadataWriter::set_delete_file`].
    ///
    /// Scans the whole file; pushing `predicate` down for row-group/bloom pruning
    /// is a possible optimization — but any such pushdown must exclude float
    /// predicates unless the file is known NaN-free (footer bounds exclude NaN;
    /// see `NanPruningBarrierExec`), or a DELETE/UPDATE could miss NaN rows.
    /// Only valid for insert-only files, where `position = rowid - row_id_start`.
    pub async fn resolve_positions(
        &self,
        state: &dyn Session,
        data_file: &DuckLakeFileData,
        predicate: Arc<dyn datafusion::physical_expr::PhysicalExpr>,
    ) -> DataFusionResult<HashSet<i64>> {
        // Positional scan of the data file: read the physical data columns and
        // materialize the true physical row position (`ROW_POS_COLUMN_NAME`) via
        // `FileRowNumberExec`, WITHOUT applying any delete files. Then evaluate
        // `predicate` per batch and collect the physical positions of matching
        // rows — exactly the `pos` values a positional delete file records.
        //
        // `predicate` is expressed against the table's logical column order
        // (column index i = the i-th logical/data field); `Column::evaluate` is
        // index-based, so it resolves against the read batch regardless of any
        // physical rename. `ROW_POS_COLUMN_NAME` is appended last and is never
        // referenced by the predicate. Valid for insert-only files, where the
        // physical position equals `rowid - row_id_start`.
        let file_cfg = self.build_file_read_config(state, data_file).await?;

        // Row-group-aligned partitions + a non-repartition, non-pruning source so
        // `FileRowNumberExec` yields true physical positions (mirrors the scan
        // paths in `build_exec_for_file_with_rowid`).
        let target_partitions = state.config().target_partitions();
        let (file_groups, partition_starts) =
            self.build_row_group_partitions(data_file, &file_cfg, target_partitions)?;

        let source = PositionalFileSource::wrap(Arc::new(
            self.create_parquet_source(file_cfg.read_schema.clone()),
        ));
        // Physical data columns only (logical order); embedded/rowid columns are
        // not needed to evaluate the predicate or read positions.
        let physical_proj: Vec<usize> = (0..self.physical_schema.fields().len()).collect();
        let scan = DataSourceExec::from_data_source(
            FileScanConfigBuilder::new(self.object_store_url.as_ref().clone(), source)
                .with_file_groups(file_groups)
                .with_partitioned_by_file_group(true)
                .with_projection_indices(Some(physical_proj))?
                .build(),
        );
        let plan: Arc<dyn ExecutionPlan> = Arc::new(FileRowNumberExec::new(scan, partition_starts));
        let pos_idx = plan.schema().index_of(ROW_POS_COLUMN_NAME)?;

        let batches = datafusion::physical_plan::collect(plan, state.task_ctx()).await?;

        let mut positions = HashSet::new();
        for batch in &batches {
            let mask = predicate.evaluate(batch)?.into_array(batch.num_rows())?;
            let mask = mask
                .as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(|| {
                    DataFusionError::Execution(
                        "resolve_positions: predicate did not evaluate to a boolean".to_string(),
                    )
                })?;
            let pos = batch
                .column(pos_idx)
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| {
                    DataFusionError::Internal(format!("{ROW_POS_COLUMN_NAME} column is not Int64"))
                })?;
            for i in 0..batch.num_rows() {
                // A NULL predicate result is treated as non-match (SQL semantics).
                if mask.is_valid(i) && mask.value(i) {
                    positions.insert(pos.value(i));
                }
            }
        }
        Ok(positions)
    }

    /// Read a delete file and return the set of physical row positions it marks
    /// deleted (the `pos` column). Callers use this to form the cumulative
    /// (prior ∪ new) position set when superseding a data file's live delete
    /// file via [`crate::metadata_writer::MetadataWriter::set_delete_file`].
    ///
    /// The delete file is already associated with a specific data file via
    /// metadata; only `pos` is read (the `file_path` column is documentation).
    pub async fn read_delete_file_positions(
        &self,
        state: &dyn Session,
        delete_file: &DuckLakeFileData,
    ) -> DataFusionResult<HashSet<i64>> {
        // Get the standard delete file schema
        let delete_schema = delete_file_schema();

        // Resolve the delete file path
        let resolved_delete_path = self.resolve_file_path(delete_file)?;

        // Create PartitionedFile with footer size hint if available
        let mut pf = PartitionedFile::new(
            &resolved_delete_path,
            validated_file_size(delete_file.file_size_bytes, &resolved_delete_path)?,
        );
        if let Some(footer_size) = delete_file.footer_size
            && footer_size > 0
            && let Ok(hint) = usize::try_from(footer_size)
        {
            pf = pf.with_metadata_size_hint(hint);
        }

        // Create file scan config for the delete file
        let file_scan_config = FileScanConfigBuilder::new(
            self.object_store_url.as_ref().clone(),
            Arc::new(self.create_parquet_source(delete_schema)),
        )
        .with_file_group(FileGroup::new(vec![pf]))
        .build();

        // Use DataSourceExec directly to preserve our ParquetSource with encryption factory
        let exec = DataSourceExec::from_data_source(file_scan_config);

        // Execute and collect all batches
        let task_ctx = state.task_ctx();
        let stream = exec.execute(0, task_ctx)?;

        let batches: Vec<RecordBatch> = stream
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<DataFusionResult<Vec<_>>>()
            .map_err(|e| {
                if is_object_store_not_found(&e) {
                    DataFusionError::Execution(format!(
                        "Delete file '{}' referenced in catalog metadata was not found. This may indicate catalog corruption or that the file was deleted outside of DuckLake.",
                        resolved_delete_path
                    ))
                } else {
                    e
                }
            })?;

        // Extract all positions from all batches
        let mut positions = HashSet::new();
        for batch in batches {
            extract_deleted_positions_from_batch(&batch, &mut positions)?;
        }

        Ok(positions)
    }

    /// Whether `file` embeds a `_ducklake_internal_row_id` column (tagged with
    /// [`ROW_ID_PARQUET_FIELD_ID`]) — i.e. it was rewritten by an UPDATE or
    /// compaction rather than being insert-only.
    ///
    /// [`Self::resolve_positions`] derives delete positions from the physical row
    /// index, which is only the DuckLake `pos` for insert-only files; a rewritten
    /// file's surviving rows carry embedded rowids whose physical order need not
    /// match, so the delete path must refuse such files rather than mis-delete.
    /// Memoized through the shared `file_read_config_cache`, so calling this right
    /// before `resolve_positions` costs at most one extra footer read per file.
    #[cfg(feature = "write")]
    pub(crate) async fn file_has_embedded_rowid(
        &self,
        state: &dyn Session,
        file: &DuckLakeFileData,
    ) -> DataFusionResult<bool> {
        let cfg = self.build_file_read_config(state, file).await?;
        Ok(cfg.embedded_rowid_parquet_name.is_some())
    }

    /// Build a single execution plan for all files without delete files
    ///
    /// Groups multiple files into a single efficient execution plan since they don't
    /// need delete filtering.
    async fn build_exec_for_files_without_deletes(
        &self,
        state: &dyn Session,
        files: &[&DuckLakeTableFile],
        file_statistics: &HashMap<i64, Arc<Statistics>>,
        projection: Option<&Vec<usize>>,
        limit: Option<usize>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        // Physical column names can differ across files (e.g. a column renamed
        // after some files were written), so the field_id -> physical-name read
        // schema must be resolved PER FILE. Group files that share the same
        // physical schema into one ParquetSource and union the groups; the common
        // case (no schema evolution) stays a single group / single scan.
        let mut groups: Vec<(SchemaMapping, Vec<PartitionedFile>)> = Vec::new();
        let mut group_index: HashMap<String, usize> = HashMap::new();

        for table_file in files {
            let mapping = self.file_schema_mapping(state, &table_file.file).await?;
            let pf = self.partitioned_data_file(table_file, false, file_statistics)?;

            // Group key: physical field names + types, then the rename mapping.
            let (read_schema, name_mapping) = &mapping;
            let mut key = String::new();
            for f in read_schema.fields() {
                key.push_str(f.name());
                key.push('\u{1}');
                key.push_str(&format!("{:?}", f.data_type()));
                key.push('\u{2}');
            }
            let mut pairs: Vec<(&String, &String)> = name_mapping.iter().collect();
            pairs.sort();
            for (k, v) in pairs {
                key.push_str(k);
                key.push('\u{3}');
                key.push_str(v);
                key.push('\u{4}');
            }

            match group_index.get(&key) {
                Some(&gi) => groups[gi].1.push(pf),
                None => {
                    group_index.insert(key, groups.len());
                    groups.push((mapping, vec![pf]));
                },
            }
        }

        let output_schema = match projection {
            Some(indices) => Arc::new(self.schema.project(indices)?),
            None => self.schema.clone(),
        };

        // Float columns whose NaN state isn't known false for every scanned
        // file: predicates on them must not reach the parquet reader's
        // row-group/page pruning (footer bounds exclude NaN).
        let nan_unsafe_columns = self.nan_unsafe_float_columns(files, file_statistics);

        // Build one scan per physical-schema group; ColumnRenameExec coerces each
        // group to the catalog schema (renamed columns or a differing Arrow type).
        let mut execs: Vec<Arc<dyn ExecutionPlan>> = Vec::with_capacity(groups.len());
        for ((read_schema, name_mapping), partitioned_files) in groups {
            let mut builder = FileScanConfigBuilder::new(
                self.object_store_url.as_ref().clone(),
                Arc::new(self.create_parquet_source(read_schema.clone())),
            )
            .with_limit(limit)
            .with_file_group(FileGroup::new(partitioned_files));

            if let Some(proj) = projection {
                builder = builder.with_projection_indices(Some(proj.clone()))?;
            }

            let parquet_exec: Arc<dyn ExecutionPlan> =
                DataSourceExec::from_data_source(builder.build());

            let mut exec = if !name_mapping.is_empty() || parquet_exec.schema() != output_schema {
                Arc::new(ColumnRenameExec::new(
                    parquet_exec,
                    output_schema.clone(),
                    name_mapping,
                )) as Arc<dyn ExecutionPlan>
            } else {
                parquet_exec
            };
            if !nan_unsafe_columns.is_empty() {
                exec = Arc::new(NanPruningBarrierExec::new(
                    exec,
                    Arc::clone(&nan_unsafe_columns),
                ));
            }
            execs.push(exec);
        }

        combine_execution_plans(execs)
    }

    /// Float columns whose stored max is unusable for at least one of `files`
    /// — the NaN state is unknown or positive, so parquet footer bounds (which
    /// exclude NaN) must not drive row-group/page pruning for predicates on
    /// them. Detected via the already-gated per-file statistics: a float
    /// column with an `Absent` max is exactly one whose `contains_nan` isn't
    /// known false (see `float_max_is_bound`); a file with no statistics entry
    /// is unknown across the board.
    fn nan_unsafe_float_columns(
        &self,
        files: &[&DuckLakeTableFile],
        file_statistics: &HashMap<i64, Arc<Statistics>>,
    ) -> Arc<HashSet<String>> {
        let mut unsafe_columns = HashSet::new();
        for (index, field) in self.physical_schema.fields().iter().enumerate() {
            if !matches!(
                field.data_type(),
                DataType::Float16 | DataType::Float32 | DataType::Float64
            ) {
                continue;
            }
            let any_unsafe = files.iter().any(|file| {
                file_statistics
                    .get(&file.data_file_id)
                    .is_none_or(|statistics| {
                        matches!(
                            statistics.column_statistics[index].max_value,
                            Precision::Absent
                        )
                    })
            });
            if any_unsafe {
                unsafe_columns.insert(field.name().clone());
            }
        }
        Arc::new(unsafe_columns)
    }

    /// Configure this table for write operations.
    ///
    /// This method enables write support by attaching a metadata writer and data path.
    /// Once configured, the table can handle INSERT INTO operations.
    ///
    /// # Arguments
    /// * `schema_name` - Name of the schema this table belongs to
    /// * `writer` - Metadata writer for catalog operations
    #[cfg(feature = "write")]
    pub fn with_writer(mut self, schema_name: String, writer: Arc<dyn MetadataWriter>) -> Self {
        self.schema_name = Some(schema_name);
        self.writer = Some(writer);
        self
    }

    /// Build an execution plan for a single file with delete filtering
    ///
    /// Creates a Parquet scan wrapped with a delete filter to exclude deleted rows.
    async fn build_exec_for_file_with_deletes(
        &self,
        state: &dyn Session,
        table_file: &DuckLakeTableFile,
        file_statistics: &HashMap<i64, Arc<Statistics>>,
        projection: Option<&Vec<usize>>,
        limit: Option<usize>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let file_cfg = self.build_file_read_config(state, &table_file.file).await?;

        // Deletes filter by physical row position, so this is a positional path:
        // it must read the file in row-group-aligned, non-repartitionable,
        // non-pruning partitions and synthesize positions before filtering.
        let deleted_positions = if let Some(ref delete_file) = table_file.delete_file {
            let p = self.read_delete_file_positions(state, delete_file).await?;
            (!p.is_empty()).then_some(p)
        } else {
            None
        };

        let output_schema = match projection {
            Some(indices) => Arc::new(self.schema.project(indices)?),
            None => self.schema.clone(),
        };

        // Explicit parquet projection over `read_schema`. rowid is never
        // projected on this path, so always read only the physical columns —
        // for an embedded-rowid file, `read_schema` has a trailing embedded
        // column we must NOT read here. With `projection = None` that means the
        // physical columns `0..physical_len` (not "all of read_schema").
        let proj_indices: Vec<usize> = match projection {
            Some(indices) => indices.clone(),
            None => (0..self.physical_schema.fields().len()).collect(),
        };

        let exec_after_delete: Arc<dyn ExecutionPlan> = if let Some(positions) = deleted_positions {
            // Positional path: no scan-level limit (would drop rows before the
            // delete filter); DataFusion enforces LIMIT above the table plan.
            let target_partitions = state.config().target_partitions();
            let (file_groups, partition_starts) =
                self.build_row_group_partitions(&table_file.file, &file_cfg, target_partitions)?;

            let source = PositionalFileSource::wrap(Arc::new(
                self.create_parquet_source(file_cfg.read_schema.clone()),
            ));
            let mut builder =
                FileScanConfigBuilder::new(self.object_store_url.as_ref().clone(), source)
                    .with_file_groups(file_groups)
                    // FileRowNumberExec seeds row positions from the scan
                    // partition index, so each partition must read exactly
                    // its configured row-group chunk. DF 54's shared work
                    // queue can otherwise let sibling partitions steal chunks.
                    .with_partitioned_by_file_group(true);
            builder = builder.with_projection_indices(Some(proj_indices.clone()))?;
            let scan = DataSourceExec::from_data_source(builder.build());

            let with_pos: Arc<dyn ExecutionPlan> =
                Arc::new(FileRowNumberExec::new(scan, partition_starts));
            Arc::new(DeleteFilterExec::try_new(
                with_pos,
                table_file.file.path.clone(),
                Arc::new(positions),
            )?)
        } else {
            // No actual deletes for this file: plain scan, scan-level limit OK.
            let pf = self.partitioned_data_file(
                table_file,
                file_cfg.embedded_rowid_parquet_name.is_some(),
                file_statistics,
            )?;
            let mut builder = FileScanConfigBuilder::new(
                self.object_store_url.as_ref().clone(),
                Arc::new(self.create_parquet_source(file_cfg.read_schema.clone())),
            )
            .with_limit(limit)
            .with_file_group(FileGroup::new(vec![pf]));
            builder = builder.with_projection_indices(Some(proj_indices.clone()))?;
            DataSourceExec::from_data_source(builder.build())
        };

        // ColumnRenameExec presents the catalog schema and, on the positional
        // path, drops the internal `__ducklake_row_pos` column (by name).
        if !file_cfg.name_mapping.is_empty() || exec_after_delete.schema() != output_schema {
            Ok(Arc::new(ColumnRenameExec::new(
                exec_after_delete,
                output_schema,
                file_cfg.name_mapping.clone(),
            )))
        } else {
            Ok(exec_after_delete)
        }
    }

    /// Inspect a single file's parquet metadata for the row-lineage scan
    /// path. Mirrors the per-file logic in `DuckLakeMultiFileReader::
    /// GetVirtualColumnExpression` (ducklake C++): if the file embeds a
    /// column tagged with [`ROW_ID_PARQUET_FIELD_ID`], project that column;
    /// otherwise synthesize rowid from `row_id_start + position`.
    async fn build_file_read_config(
        &self,
        state: &dyn Session,
        file: &DuckLakeFileData,
    ) -> DataFusionResult<Arc<FileReadConfig>> {
        let resolved_path = self.resolve_file_path(file)?;

        {
            let cache = self.file_read_config_cache.lock().unwrap();
            if let Some(cfg) = cache.get(&resolved_path) {
                return Ok(cfg.clone());
            }
        }

        let object_store = state
            .runtime_env()
            .object_store(self.object_store_url.as_ref())?;
        let object_path = ObjectPath::from(resolved_path.as_str());
        let reader = ParquetObjectReader::new(object_store, object_path);

        #[cfg(feature = "encryption")]
        let builder = {
            use parquet::arrow::arrow_reader::ArrowReaderOptions;
            let options = if let Some(ref key) = file.encryption_key {
                if !key.is_empty() {
                    let key_bytes = crate::encryption::DuckLakeEncryptionFactory::decode_key(key)?;
                    let decryption_props =
                        parquet::encryption::decrypt::FileDecryptionProperties::builder(key_bytes)
                            .build()
                            .map_err(|e| {
                                DataFusionError::Execution(format!(
                                    "Failed to create decryption properties: {}",
                                    e
                                ))
                            })?;
                    ArrowReaderOptions::new().with_file_decryption_properties(decryption_props)
                } else {
                    ArrowReaderOptions::new()
                }
            } else {
                ArrowReaderOptions::new()
            };
            ParquetRecordBatchStreamBuilder::new_with_options(reader, options)
                .await
                .map_err(|e| DataFusionError::External(Box::new(e)))?
        };

        #[cfg(not(feature = "encryption"))]
        let builder = ParquetRecordBatchStreamBuilder::new(reader)
            .await
            .map_err(|e| DataFusionError::External(Box::new(e)))?;

        let field_id_map = extract_parquet_field_ids(builder.metadata());

        // Per-row-group starting positions (prefix sums of num_rows), read from
        // the footer we already have open. Drives row-group-aligned scan
        // partitioning on positional paths.
        let row_groups = builder.metadata().row_groups();
        let row_group_count = row_groups.len();
        let mut row_group_starts = Vec::with_capacity(row_group_count);
        let mut row_acc: i64 = 0;
        for rg in row_groups {
            row_group_starts.push(row_acc);
            row_acc = row_acc.saturating_add(rg.num_rows());
        }

        // Standard read_schema + name_mapping for physical columns.
        let (physical_read_schema, mut name_mapping) = if field_id_map.is_empty() {
            (self.physical_schema.as_ref().clone(), HashMap::new())
        } else {
            let (s, m) = build_read_schema_with_field_id_mapping(
                &self.columns,
                &field_id_map,
                Some(builder.schema().as_ref()),
            )
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
            (s, m)
        };

        // Detect the embedded rowid column by reserved field-id.
        let embedded_rowid_parquet_name = field_id_map.get(&ROW_ID_PARQUET_FIELD_ID).cloned();
        // Detect the embedded per-row snapshot-id column (marks a partial file).
        let embedded_snapshot_parquet_name =
            field_id_map.get(&SNAPSHOT_ID_PARQUET_FIELD_ID).cloned();
        // Does the file carry a data column no longer in the current schema? Any
        // parquet field-id that is neither a reserved embedded column nor one of
        // the current catalog `column_id`s is a since-dropped column. Compaction
        // uses this to refuse merging a file whose data would be lost.
        let current_column_ids: std::collections::HashSet<i32> =
            self.columns.iter().map(|c| c.column_id as i32).collect();
        let drops_current_columns = field_id_map.keys().any(|fid| {
            *fid != ROW_ID_PARQUET_FIELD_ID
                && *fid != SNAPSHOT_ID_PARQUET_FIELD_ID
                && !current_column_ids.contains(fid)
        });

        let read_schema = if let Some(ref parquet_name) = embedded_rowid_parquet_name {
            // Append the embedded rowid column to read_schema under its
            // parquet name; ParquetExec will project it by name from the
            // file. We add a `parquet_name → "rowid"` rename so the user
            // sees the column as `rowid` (only needed if the names differ).
            let mut fields: Vec<Arc<Field>> =
                physical_read_schema.fields().iter().cloned().collect();
            fields.push(Arc::new(Field::new(
                parquet_name.clone(),
                DataType::Int64,
                true,
            )));
            if parquet_name != ROWID_COLUMN_NAME {
                name_mapping.insert(parquet_name.clone(), ROWID_COLUMN_NAME.to_string());
            }
            Arc::new(Schema::new(fields))
        } else {
            Arc::new(physical_read_schema)
        };

        let cfg = Arc::new(FileReadConfig {
            read_schema,
            name_mapping,
            embedded_rowid_parquet_name,
            embedded_snapshot_parquet_name,
            drops_current_columns,
            row_group_starts,
            row_group_count,
        });

        {
            let mut cache = self.file_read_config_cache.lock().unwrap();
            cache.entry(resolved_path).or_insert_with(|| cfg.clone());
        }

        Ok(cfg)
    }

    /// Build row-group-aligned scan partitions for a single file on a
    /// *positional* path (rowid synthesis and/or delete filtering).
    ///
    /// Returns one [`FileGroup`] per contiguous run of row groups (so each is a
    /// distinct DataFusion partition) together with a `partition_starts` vector
    /// whose `i`-th entry is the **true physical row position of the first row**
    /// of `file_groups[i]`. The two vectors are 1:1; `FileRowNumberExec` uses
    /// `partition_starts[partition]` to seed positions.
    ///
    /// Each chunk carries a whole-row-group `Scan`/`Skip` [`ParquetAccessPlan`]
    /// (never a `RowSelection`), so within a partition the reader emits a
    /// complete, contiguous, in-order run of physical rows. A single chunk
    /// (`target_partitions == 1`, or a file with ≤1 row group) carries no access
    /// plan and reads the whole file in order — identical to the legacy path.
    fn build_row_group_partitions(
        &self,
        file: &DuckLakeFileData,
        read_cfg: &FileReadConfig,
        target_partitions: usize,
    ) -> DataFusionResult<(Vec<FileGroup>, Vec<i64>)> {
        let resolved_path = self.resolve_file_path(file)?;
        let file_size = validated_file_size(file.file_size_bytes, &resolved_path)?;
        let footer_hint = file
            .footer_size
            .filter(|&s| s > 0)
            .and_then(|s| usize::try_from(s).ok());

        let make_pf = |access: Option<ParquetAccessPlan>| {
            let mut pf = PartitionedFile::new(&resolved_path, file_size);
            if let Some(hint) = footer_hint {
                pf = pf.with_metadata_size_hint(hint);
            }
            if let Some(plan) = access {
                pf = pf.with_extension(plan);
            }
            pf
        };

        let n = read_cfg.row_group_count;
        let k = target_partitions.max(1).min(n.max(1));

        // Single partition: whole file, in order, no access plan. Covers
        // target_partitions == 1 and files with 0 or 1 row groups.
        if k <= 1 {
            return Ok((vec![FileGroup::new(vec![make_pf(None)])], vec![0]));
        }

        // Split the n row groups into k contiguous chunks as evenly as possible
        // (row groups are written near-uniform, so group-count balancing closely
        // tracks row-count balancing). The first `rem` chunks get one extra group.
        let base = n / k;
        let rem = n % k;
        let mut file_groups = Vec::with_capacity(k);
        let mut partition_starts = Vec::with_capacity(k);
        let mut a = 0usize;
        for chunk in 0..k {
            let len = base + usize::from(chunk < rem);
            let b = a + len;
            debug_assert!(b <= n && len > 0);

            let row_groups: Vec<RowGroupAccess> = (0..n)
                .map(|rg| {
                    if rg >= a && rg < b {
                        RowGroupAccess::Scan
                    } else {
                        RowGroupAccess::Skip
                    }
                })
                .collect();

            file_groups.push(FileGroup::new(vec![make_pf(Some(ParquetAccessPlan::new(
                row_groups,
            )))]));
            partition_starts.push(read_cfg.row_group_starts[a]);
            a = b;
        }
        debug_assert_eq!(a, n);

        Ok((file_groups, partition_starts))
    }

    /// Build a plan for a single file when the synthetic `rowid` column is in
    /// the projection. Always uses per-file scans because each file may have a
    /// different layout (embedded rowid vs. synthesized) and a distinct
    /// `row_id_start`.
    ///
    /// Order on the positional path (non-embedded, or any file with deletes):
    ///   DataSourceExec → FileRowNumberExec → DeleteFilterExec(?) → RowIdExec(?)
    ///   → ColumnRenameExec. Embedded-rowid files with no deletes keep a plain
    ///   DataSourceExec → ColumnRenameExec (rowid read from the file).
    async fn build_exec_for_file_with_rowid(
        &self,
        state: &dyn Session,
        table_file: &DuckLakeTableFile,
        file_statistics: &HashMap<i64, Arc<Statistics>>,
        user_proj: &[usize],
        rowid_idx: usize,
        limit: Option<usize>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let file_cfg = self.build_file_read_config(state, &table_file.file).await?;
        let has_embedded = file_cfg.embedded_rowid_parquet_name.is_some();

        // Physical columns to read (everything the user asked for except rowid).
        let physical_proj: Vec<usize> = user_proj
            .iter()
            .filter(|&&i| i != rowid_idx)
            .copied()
            .collect();

        // Match the C++ extension: if the file embeds no rowid column AND the
        // catalog didn't record a `row_id_start`, lineage cannot be
        // reconstructed. Hard-error rather than silently emit NULL/garbage.
        if !has_embedded && table_file.row_id_start.is_none() {
            return Err(DataFusionError::Execution(format!(
                "File \"{}\" has no embedded `_ducklake_internal_row_id` column and no \
                 `row_id_start` set in the catalog — row lineage cannot be reconstructed",
                table_file.file.path
            )));
        }

        // Resolve deletes once.
        let deleted_positions = if let Some(ref delete_file) = table_file.delete_file {
            let p = self.read_delete_file_positions(state, delete_file).await?;
            (!p.is_empty()).then_some(p)
        } else {
            None
        };
        let has_deletes = deleted_positions.is_some();

        // We need synthesized physical positions when rowid must be synthesized
        // (non-embedded) or when positional deletes must be applied. Embedded-
        // rowid files with no deletes keep the legacy plain scan (rowid read from
        // the file; reader-side pruning and scan-level limit are safe there).
        let needs_position = !has_embedded || has_deletes;

        // Parquet read projection. For embedded files, also read the embedded
        // rowid column; `ColumnRenameExec` later maps it to `rowid` by name, so
        // its position in the read projection is irrelevant.
        let parquet_projection: Vec<usize> = if has_embedded {
            let rowid_col_in_read_schema = file_cfg.read_schema.fields().len() - 1;
            let mut p = physical_proj.clone();
            p.push(rowid_col_in_read_schema);
            p
        } else {
            physical_proj.clone()
        };

        let after_deletes: Arc<dyn ExecutionPlan> = if needs_position {
            // Positional path: row-group-aligned partitions + a non-repartition,
            // non-pruning source, so each partition emits a complete, contiguous,
            // in-order run of physical rows. No scan-level limit (it would drop
            // rows before delete filtering); DataFusion enforces LIMIT above.
            let target_partitions = state.config().target_partitions();
            let (file_groups, partition_starts) =
                self.build_row_group_partitions(&table_file.file, &file_cfg, target_partitions)?;

            let source = PositionalFileSource::wrap(Arc::new(
                self.create_parquet_source(file_cfg.read_schema.clone()),
            ));
            let mut builder =
                FileScanConfigBuilder::new(self.object_store_url.as_ref().clone(), source)
                    .with_file_groups(file_groups)
                    // FileRowNumberExec seeds row positions from the scan
                    // partition index, so each partition must read exactly
                    // its configured row-group chunk. DF 54's shared work
                    // queue can otherwise let sibling partitions steal chunks.
                    .with_partitioned_by_file_group(true);
            builder = builder.with_projection_indices(Some(parquet_projection))?;
            let scan = DataSourceExec::from_data_source(builder.build());

            // Materialize the physical position, then (optionally) filter deletes
            // by it, then (for non-embedded files) synthesize rowid from it.
            let mut plan: Arc<dyn ExecutionPlan> =
                Arc::new(FileRowNumberExec::new(scan, partition_starts));
            if let Some(p) = deleted_positions {
                plan = Arc::new(DeleteFilterExec::try_new(
                    plan,
                    table_file.file.path.clone(),
                    Arc::new(p),
                )?);
            }
            if !has_embedded {
                plan = Arc::new(RowIdExec::try_new(plan, table_file.row_id_start)?);
            }
            plan
        } else {
            // Embedded rowid, no deletes: legacy plain scan (cardinality-
            // preserving). Keep scan-level limit and reader pruning.
            let pf = self.partitioned_data_file(table_file, true, file_statistics)?;
            let mut builder = FileScanConfigBuilder::new(
                self.object_store_url.as_ref().clone(),
                Arc::new(self.create_parquet_source(file_cfg.read_schema.clone())),
            )
            .with_limit(limit)
            .with_file_group(FileGroup::new(vec![pf]));
            builder = builder.with_projection_indices(Some(parquet_projection))?;
            DataSourceExec::from_data_source(builder.build())
        };

        // Wrap with ColumnRenameExec to present the catalog schema. Required when
        // a physical column was renamed in the catalog, when the embedded rowid
        // column's parquet name differs from `"rowid"` (the common case — it's
        // `_ducklake_internal_row_id`), or when the file's physical Arrow type
        // differs from the catalog type (e.g. a DuckDB ARRAY read as
        // FixedSizeList vs the catalog's List). Coerces each column to
        // `output_schema`.
        let output_schema = self.output_schema_for_projection(user_proj, rowid_idx);
        let mut exec =
            if !file_cfg.name_mapping.is_empty() || after_deletes.schema() != output_schema {
                Arc::new(ColumnRenameExec::new(
                    after_deletes,
                    output_schema,
                    file_cfg.name_mapping.clone(),
                )) as Arc<dyn ExecutionPlan>
            } else {
                after_deletes
            };
        // The positional path already refuses all filter pushdown
        // (PositionalFileSource); only the legacy plain scan lets predicates
        // reach the parquet reader's pruning, so only it needs the NaN barrier.
        if !needs_position {
            let nan_unsafe_columns =
                self.nan_unsafe_float_columns(std::slice::from_ref(&table_file), file_statistics);
            if !nan_unsafe_columns.is_empty() {
                exec = Arc::new(NanPruningBarrierExec::new(exec, nan_unsafe_columns));
            }
        }
        Ok(exec)
    }

    /// Output schema for the rowid-projected per-file plan: physical fields
    /// (using their user-facing renamed names from `self.schema`) interleaved
    /// with the synthetic `rowid` field at `rowid_idx`.
    fn output_schema_for_projection(&self, user_proj: &[usize], rowid_idx: usize) -> SchemaRef {
        let mut fields: Vec<Arc<Field>> = Vec::with_capacity(user_proj.len());
        for &i in user_proj {
            if i == rowid_idx {
                fields.push(Arc::new(rowid_field()));
            } else {
                fields.push(self.schema.fields()[i].clone());
            }
        }
        Arc::new(Schema::new(fields))
    }

    /// Whether `table_file` is a merged partial file being read at a snapshot
    /// BELOW its `partial_max` — i.e. some of its rows originate from snapshots
    /// newer than the read snapshot and must be dropped per-row. When false (the
    /// common case: an ordinary file, or a partial file read at or after
    /// `partial_max`), the file is read by the existing paths with no filtering.
    fn needs_snapshot_filter(&self, table_file: &DuckLakeTableFile) -> bool {
        table_file
            .partial_max
            .is_some_and(|partial_max| self.snapshot_id < partial_max)
    }

    /// Build a plan for a single merged **partial file** read at a snapshot below
    /// its `partial_max`. Reads every column (data + embedded rowid + embedded
    /// snapshot-id), drops rows whose embedded origin snapshot exceeds the read
    /// snapshot via [`SnapshotFilterExec`], then presents `output_schema` (which
    /// also projects away the embedded snapshot-id and any unrequested embedded
    /// rowid column). Used only on the cold time-travel path, so it reads all
    /// columns and lets `LIMIT` apply above rather than pushing it into the scan.
    async fn build_exec_for_partial_file(
        &self,
        state: &dyn Session,
        table_file: &DuckLakeTableFile,
        output_schema: SchemaRef,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let file_cfg = self.build_file_read_config(state, &table_file.file).await?;
        let snap_name = file_cfg
            .embedded_snapshot_parquet_name
            .clone()
            .ok_or_else(|| {
                DataFusionError::Internal(format!(
                    "partial file \"{}\" is missing its embedded snapshot-id column",
                    table_file.file.path
                ))
            })?;

        // Append the embedded snapshot-id column to the file's read schema so the
        // scan materializes it. It is deliberately absent from the cached
        // `read_schema` (that would shift the embedded-rowid column off the end,
        // which other read paths rely on), so we append it here.
        let mut fields: Vec<Arc<Field>> = file_cfg.read_schema.fields().iter().cloned().collect();
        fields.push(Arc::new(Field::new(&snap_name, DataType::Int64, true)));
        let read_schema = Arc::new(Schema::new(fields));
        let projection: Vec<usize> = (0..read_schema.fields().len()).collect();

        let resolved_path = self.resolve_file_path(&table_file.file)?;
        let mut pf = PartitionedFile::new(
            &resolved_path,
            validated_file_size(table_file.file.file_size_bytes, &resolved_path)?,
        );
        if let Some(footer_size) = table_file.file.footer_size
            && footer_size > 0
            && let Ok(hint) = usize::try_from(footer_size)
        {
            pf = pf.with_metadata_size_hint(hint);
        }
        let builder = FileScanConfigBuilder::new(
            self.object_store_url.as_ref().clone(),
            Arc::new(self.create_parquet_source(read_schema.clone())),
        )
        .with_file_group(FileGroup::new(vec![pf]))
        .with_projection_indices(Some(projection))?;
        let scan = DataSourceExec::from_data_source(builder.build());

        // Drop rows newer than the read snapshot, then present the catalog schema.
        let filtered: Arc<dyn ExecutionPlan> = Arc::new(SnapshotFilterExec::try_new(
            scan,
            snap_name,
            self.snapshot_id,
        )?);
        Ok(Arc::new(ColumnRenameExec::new(
            filtered,
            output_schema,
            file_cfg.name_mapping.clone(),
        )))
    }

    /// A read-only clone of this table (no writer, no rowid projection, fresh
    /// per-file read-config cache) carrying exactly the metadata a scan needs.
    /// [`DuckLakeUpdateExec`] holds one so it can drive the per-file update
    /// scans ([`Self::compute_file_update`]) at execute time — `update()` only
    /// has `&self`, so it cannot hand the exec an `Arc<Self>` directly.
    #[cfg(feature = "write")]
    fn read_only_clone(&self) -> DuckLakeTable {
        DuckLakeTable {
            table_id: self.table_id,
            table_name: self.table_name.clone(),
            provider: Arc::clone(&self.provider),
            snapshot_id: self.snapshot_id,
            object_store_url: self.object_store_url.clone(),
            table_path: self.table_path.clone(),
            schema: self.physical_schema.clone(),
            physical_schema: self.physical_schema.clone(),
            row_lineage: false,
            columns: self.columns.clone(),
            table_statistics: self.table_statistics.clone(),
            partition_spec: self.partition_spec.clone(),
            // `snapshot_id`/cache match the post-#163 struct (Arc-wrapped cache,
            // pinned snapshot). A read-only clone starts with an empty cache.
            file_read_config_cache: Arc::new(std::sync::Mutex::new(HashMap::new())),
            #[cfg(feature = "encryption")]
            encryption_factory: self.encryption_factory.clone(),
            schema_name: None,
            writer: None,
        }
    }

    /// Physical (data-column) schema this table reads/writes, without the
    /// synthetic `rowid`. Used by [`DuckLakeUpdateExec`] to author the rewritten
    /// data file.
    #[cfg(feature = "write")]
    pub(crate) fn physical_schema(&self) -> SchemaRef {
        self.physical_schema.clone()
    }

    /// The metadata writer, when this table was opened writable
    /// (`DuckLakeCatalog::with_writer`). Used by the compaction ops.
    #[cfg(feature = "write")]
    pub(crate) fn writer(&self) -> Option<&Arc<dyn MetadataWriter>> {
        self.writer.as_ref()
    }

    /// The schema name, when this table was opened writable. Used by the
    /// compaction ops to author output file paths.
    #[cfg(feature = "write")]
    pub(crate) fn schema_name(&self) -> Option<&str> {
        self.schema_name.as_deref()
    }

    /// This table's name. Used by the compaction ops for output file paths.
    #[cfg(feature = "write")]
    pub(crate) fn table_name(&self) -> &str {
        &self.table_name
    }

    /// This table's catalog `table_id`. Used by the compaction commit.
    #[cfg(feature = "write")]
    pub(crate) fn table_id(&self) -> i64 {
        self.table_id
    }

    /// The snapshot this table was opened at — the base the compaction sources
    /// are read against, threaded to the commit as its conflict base.
    #[cfg(feature = "write")]
    pub(crate) fn base_snapshot(&self) -> i64 {
        self.snapshot_id
    }

    /// The object store URL for resolving this table's file paths.
    #[cfg(feature = "write")]
    pub(crate) fn object_store_url(&self) -> &Arc<ObjectStoreUrl> {
        &self.object_store_url
    }

    /// The live columns' catalog `column_id`s in `column_order` — the parquet
    /// field-ids a compaction output must bake in so its data columns map back
    /// to the catalog on read.
    #[cfg(feature = "write")]
    pub(crate) fn column_ids(&self) -> Vec<i64> {
        self.columns.iter().map(|c| c.column_id).collect()
    }

    /// Whether `file` carries a data column that is no longer in the table's
    /// current schema (dropped since it was written). `merge_adjacent_files`
    /// refuses to compact such a file — merged output is written at the current
    /// schema, which would drop that column's data. Reads the parquet footer
    /// (memoized in the per-file read-config cache).
    #[cfg(feature = "write")]
    pub(crate) async fn file_drops_current_columns(
        &self,
        state: &dyn Session,
        file: &DuckLakeFileData,
    ) -> DataFusionResult<bool> {
        Ok(self
            .build_file_read_config(state, file)
            .await?
            .drops_current_columns)
    }

    /// Build the positional read plan (and the metadata needed to interpret it)
    /// for one source file of an `UPDATE`. Runs at PLAN time: it reads the
    /// parquet footer (field-ids, row-group layout) and the file's live delete
    /// positions — the same plan-time reads `scan()` performs — but executes NO
    /// data scan and mutates nothing. The returned [`UpdateSourceScan::scan`]
    /// yields the physical data columns (logical order), the embedded rowid
    /// column when the file has one, and the internal physical-position column;
    /// [`Self::apply_update_to_batches`] turns its collected batches into the
    /// rewritten rows at execute time.
    ///
    /// Errors if the file has neither an embedded `_ducklake_internal_row_id`
    /// column nor a catalog `row_id_start`: its lineage cannot be reconstructed,
    /// so rewriting it would fabricate rowids.
    #[cfg(feature = "write")]
    pub(crate) async fn build_update_scan(
        &self,
        state: &dyn Session,
        table_file: &DuckLakeTableFile,
    ) -> DataFusionResult<UpdateSourceScan> {
        let file_cfg = self.build_file_read_config(state, &table_file.file).await?;
        let has_embedded = file_cfg.embedded_rowid_parquet_name.is_some();

        if !has_embedded && table_file.row_id_start.is_none() {
            return Err(DataFusionError::Execution(format!(
                "File \"{}\" has no embedded `_ducklake_internal_row_id` column and no \
                 `row_id_start` in the catalog — cannot preserve row lineage through UPDATE",
                table_file.file.path
            )));
        }

        // Rows already masked by a live delete file must not be re-updated, and
        // must remain masked in the file's new cumulative delete.
        let existing_deleted: HashSet<i64> = if let Some(ref delete_file) = table_file.delete_file {
            self.read_delete_file_positions(state, delete_file).await?
        } else {
            HashSet::new()
        };

        // Positional scan: row-group-aligned partitions + a non-repartition,
        // non-pruning source so `FileRowNumberExec` yields true physical
        // positions. Project the physical columns (logical order) and, for an
        // embedded file, the embedded rowid column too.
        let physical_len = self.physical_schema.fields().len();
        let target_partitions = state.config().target_partitions();
        let (file_groups, partition_starts) =
            self.build_row_group_partitions(&table_file.file, &file_cfg, target_partitions)?;
        let source = PositionalFileSource::wrap(Arc::new(
            self.create_parquet_source(file_cfg.read_schema.clone()),
        ));
        let mut proj: Vec<usize> = (0..physical_len).collect();
        let embedded_batch_idx = if has_embedded {
            proj.push(file_cfg.read_schema.fields().len() - 1);
            Some(physical_len)
        } else {
            None
        };
        let scan = DataSourceExec::from_data_source(
            FileScanConfigBuilder::new(self.object_store_url.as_ref().clone(), source)
                .with_file_groups(file_groups)
                .with_partitioned_by_file_group(true)
                .with_projection_indices(Some(proj))?
                .build(),
        );
        let mut plan: Arc<dyn ExecutionPlan> =
            Arc::new(FileRowNumberExec::new(scan, partition_starts));
        if !existing_deleted.is_empty() {
            plan = Arc::new(DeleteFilterExec::try_new(
                plan,
                table_file.file.path.clone(),
                Arc::new(existing_deleted.clone()),
            )?);
        }
        let pos_index = plan.schema().index_of(ROW_POS_COLUMN_NAME)?;

        Ok(UpdateSourceScan {
            scan: plan,
            physical_len,
            embedded_batch_idx,
            pos_index,
            row_id_start: table_file.row_id_start,
            existing_deleted,
            data_file_id: table_file.data_file_id,
            delete_file_id: table_file.delete_file_id,
            source_path: table_file.file.path.clone(),
        })
    }

    /// Turn the batches collected from an [`UpdateSourceScan`] into the rewritten
    /// row versions for one source file: select the rows matching `predicate`
    /// (or every live row when it is `None`), apply `assignments`, and RETAIN
    /// each row's original rowid so lineage survives the rewrite. Pure and
    /// synchronous — the exec runs it at execute time after `collect`ing the
    /// scan, so no [`Session`] is required.
    ///
    /// `assignments` are `(physical_column_index, new_value_expr)`; unlisted
    /// columns carry through unchanged. Returned batches are
    /// `[physical columns (catalog types)..., rowid]`, ready for
    /// [`DuckLakeTableWriter::begin_write_with_embedded_rowid`](crate::table_writer::DuckLakeTableWriter::begin_write_with_embedded_rowid).
    /// The original rowid is the embedded column when the file has one, else
    /// `row_id_start + physical_position`.
    #[cfg(feature = "write")]
    pub(crate) fn apply_update_to_batches(
        &self,
        scan: &UpdateSourceScan,
        batches: &[RecordBatch],
        predicate: Option<&Arc<dyn PhysicalExpr>>,
        assignments: &[(usize, Arc<dyn PhysicalExpr>)],
    ) -> DataFusionResult<FileUpdateOutput> {
        let physical_len = scan.physical_len;

        // Output schema for the rewritten rows: physical columns + rowid.
        let mut out_fields: Vec<Arc<Field>> =
            self.physical_schema.fields().iter().cloned().collect();
        out_fields.push(Arc::new(rowid_field()));
        let out_schema = Arc::new(Schema::new(out_fields));

        let mut updated_batches: Vec<RecordBatch> = Vec::new();
        let mut new_positions: Vec<i64> = Vec::new();

        for batch in batches {
            let n = batch.num_rows();
            if n == 0 {
                continue;
            }

            // Coerce physical columns to the catalog types the assignment /
            // predicate exprs (and the writer) expect.
            let mut phys_cols: Vec<ArrayRef> = Vec::with_capacity(physical_len);
            for i in 0..physical_len {
                phys_cols.push(crate::column_rename::coerce_column(
                    batch.column(i),
                    self.physical_schema.field(i).data_type(),
                )?);
            }
            let phys_batch = RecordBatch::try_new(self.physical_schema.clone(), phys_cols.clone())?;

            let row_pos = batch
                .column(scan.pos_index)
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| {
                    DataFusionError::Internal(format!("{ROW_POS_COLUMN_NAME} column is not Int64"))
                })?;

            // Predicate mask (all rows when there is no WHERE). A NULL predicate
            // result is a non-match (SQL semantics).
            let mask: BooleanArray = match predicate {
                Some(p) => {
                    let arr = p.evaluate(&phys_batch)?.into_array(n)?;
                    let b = arr.as_any().downcast_ref::<BooleanArray>().ok_or_else(|| {
                        DataFusionError::Execution(
                            "UPDATE predicate did not evaluate to a boolean".to_string(),
                        )
                    })?;
                    BooleanArray::from(
                        (0..n)
                            .map(|i| b.is_valid(i) && b.value(i))
                            .collect::<Vec<bool>>(),
                    )
                },
                None => BooleanArray::from(vec![true; n]),
            };
            if mask.true_count() == 0 {
                continue;
            }

            // Keep only matched rows, then apply the assignments to them.
            let matched_phys: Vec<ArrayRef> = phys_cols
                .iter()
                .map(|c| arrow::compute::filter(c.as_ref(), &mask))
                .collect::<std::result::Result<_, _>>()
                .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?;
            let matched_batch =
                RecordBatch::try_new(self.physical_schema.clone(), matched_phys.clone())?;
            let matched_rows = matched_batch.num_rows();

            let mut out_cols = matched_phys;
            for (col_idx, expr) in assignments {
                let val = expr.evaluate(&matched_batch)?.into_array(matched_rows)?;
                out_cols[*col_idx] = crate::column_rename::coerce_column(
                    &val,
                    self.physical_schema.field(*col_idx).data_type(),
                )?;
            }

            // Original rowids: embedded column when present, else synthesized.
            let matched_pos = arrow::compute::filter(row_pos, &mask)
                .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?;
            let matched_pos = matched_pos
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("filtered Int64Array");
            let rowid_col: ArrayRef = if let Some(idx) = scan.embedded_batch_idx {
                let embedded = batch
                    .column(idx)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .ok_or_else(|| {
                        DataFusionError::Internal("embedded rowid column is not Int64".to_string())
                    })?;
                let embedded: ArrayRef = Arc::new(embedded.clone());
                arrow::compute::filter(embedded.as_ref(), &mask)
                    .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?
            } else {
                let start = scan
                    .row_id_start
                    .expect("row_id_start checked in build_update_scan");
                Arc::new(Int64Array::from(
                    matched_pos
                        .values()
                        .iter()
                        .map(|p| start + p)
                        .collect::<Vec<i64>>(),
                ))
            };
            out_cols.push(rowid_col);
            updated_batches.push(RecordBatch::try_new(out_schema.clone(), out_cols)?);

            new_positions.extend(matched_pos.values().iter().copied());
        }

        let matched_count = new_positions.len();
        let mut cumulative = scan.existing_deleted.clone();
        cumulative.extend(new_positions);
        let mut cumulative_positions: Vec<i64> = cumulative.into_iter().collect();
        cumulative_positions.sort_unstable();

        Ok(FileUpdateOutput {
            updated_batches,
            matched_count,
            cumulative_positions,
        })
    }
}

/// Per-source-file read plan + metadata for an `UPDATE`, produced by
/// [`DuckLakeTable::build_update_scan`] at plan time and consumed by
/// [`DuckLakeUpdateExec`] at execute time.
#[cfg(feature = "write")]
#[derive(Clone)]
pub(crate) struct UpdateSourceScan {
    /// Positional read plan yielding `[physical columns..., (embedded rowid),
    /// __ducklake_row_pos]` for the source file, already masking rows removed by
    /// its live delete file.
    pub(crate) scan: Arc<dyn ExecutionPlan>,
    /// Number of physical (data) columns at the front of each scanned batch.
    pub(crate) physical_len: usize,
    /// Column index of the embedded rowid in each scanned batch, or `None` when
    /// the file has no embedded rowid (rowids are synthesized from
    /// `row_id_start + position`).
    pub(crate) embedded_batch_idx: Option<usize>,
    /// Column index of the internal physical-position column in each batch.
    pub(crate) pos_index: usize,
    /// The source file's catalog `row_id_start` (used to synthesize rowids for a
    /// non-embedded file).
    pub(crate) row_id_start: Option<i64>,
    /// Positions already masked by the file's live delete file, carried forward
    /// into the new cumulative delete.
    pub(crate) existing_deleted: HashSet<i64>,
    /// Catalog id of the source data file (the positional delete's target).
    pub(crate) data_file_id: i64,
    /// Catalog id of the file's currently-live delete file (compare-and-swap
    /// guard when superseding it), or `None`.
    pub(crate) delete_file_id: Option<i64>,
    /// The source data file's catalog path (records the delete's provenance).
    pub(crate) source_path: String,
}

/// The rewrite produced for one source data file by
/// [`DuckLakeTable::apply_update_to_batches`].
#[cfg(feature = "write")]
pub(crate) struct FileUpdateOutput {
    /// Rewritten row versions, `[physical columns..., rowid]`, carrying each
    /// row's original rowid. Empty when no rows matched.
    pub(crate) updated_batches: Vec<RecordBatch>,
    /// Number of rows this update rewrote in the source file.
    pub(crate) matched_count: usize,
    /// Physical positions to mask on the source file afterwards: the rows this
    /// update supersedes unioned with any already-deleted rows (sorted).
    pub(crate) cumulative_positions: Vec<i64>,
}

#[async_trait]
impl TableProvider for DuckLakeTable {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    fn statistics(&self) -> Option<Statistics> {
        let mut statistics = self.table_statistics.clone();
        if self.row_lineage {
            statistics
                .column_statistics
                .push(ColumnStatistics::new_unknown());
        }
        Some(statistics)
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> DataFusionResult<Vec<TableProviderFilterPushDown>> {
        // Mark all filters as Inexact because we apply delete filters after the scan.
        // DataFusion will reapply these filters after DeleteFilterExec to ensure
        // correctness, but Parquet can still use them for:
        // - Row group pruning via statistics
        // - Page-level filtering with late materialization
        // - Bloom filter lookups (if available)
        Ok(filters
            .iter()
            .map(|_| TableProviderFilterPushDown::Inexact)
            .collect())
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        // Filters drive plan-time file pruning below: `prune_table_files` drops
        // files whose catalog statistics prove they cannot match. They are also
        // pushed down to the parquet scanner by DataFusion's optimizer for row
        // group / page-level filtering. We declare them Inexact in
        // `supports_filters_pushdown`, so DataFusion reapplies them after our
        // scan — pruning here only ever removes provably non-matching files.
        filters: &[Expr],
        limit: Option<usize>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        // Row-lineage detour: when the synthetic `rowid` column is projected,
        // every file needs its own scan because each has a distinct
        // `row_id_start`. `projection == None` with row lineage on means "all
        // columns including rowid", which also routes through this path.
        let rowid_idx = self.rowid_index();
        let rowid_in_proj = match (rowid_idx, projection) {
            (Some(r), Some(p)) => p.contains(&r),
            (Some(_), None) => true,
            (None, _) => false,
        };

        let mut execs: Vec<Arc<dyn ExecutionPlan>> = Vec::new();
        let pruning = match self.file_pruning_predicate(state, filters) {
            Ok(pruning) => pruning,
            Err(error) => {
                tracing::debug!(%error, "skipping plan-time file pruning");
                None
            },
        };
        let mut after_data_file_id = None;
        loop {
            let metadata = self.provider.get_table_file_metadata_page(
                self.table_id,
                self.snapshot_id,
                after_data_file_id,
                FILE_METADATA_BATCH_SIZE,
            )?;
            if metadata.is_empty() {
                break;
            }
            if metadata.len() > FILE_METADATA_BATCH_SIZE {
                return Err(DataFusionError::External(Box::new(
                    crate::DuckLakeError::InvalidConfig(format!(
                        "metadata provider returned {} files for a {}-file planning page",
                        metadata.len(),
                        FILE_METADATA_BATCH_SIZE
                    )),
                )));
            }
            // A short page proves the file list is exhausted (pages are keyed
            // and ordered by data_file_id under a LIMIT), so stop after
            // processing it rather than paying one more metadata round trip
            // for the empty terminator page. For tables with fewer files than
            // the batch size — the common case — this halves the planning
            // metadata queries.
            let last_page = metadata.len() < FILE_METADATA_BATCH_SIZE;
            let next_after = metadata.last().unwrap().file.data_file_id;
            if after_data_file_id.is_some_and(|after| next_after <= after) {
                return Err(DataFusionError::External(Box::new(
                    crate::DuckLakeError::InvalidConfig(
                        "metadata provider returned a non-advancing file page".to_string(),
                    ),
                )));
            }
            after_data_file_id = Some(next_after);

            let mut catalog_file_statistics = Vec::new();
            let mut table_files = Vec::with_capacity(metadata.len());
            for DuckLakeFileMetadata {
                file,
                column_statistics,
            } in metadata
            {
                table_files.push(file);
                catalog_file_statistics.extend(column_statistics);
            }
            let (_, mut file_statistics) = build_datafusion_statistics(
                self.physical_schema.as_ref(),
                &self.columns,
                &table_files,
                DuckLakeStatistics {
                    files: catalog_file_statistics,
                    ..Default::default()
                },
                false,
                true,
            );
            // Synthesize per-file bounds from partition values so partition columns
            // prune even when a file carries no parquet-derived column statistics.
            self.apply_partition_bounds(&table_files, &mut file_statistics);
            #[cfg(feature = "encryption")]
            self.configure_encryption_factory(&table_files)?;

            let table_files =
                self.prune_table_files(pruning.as_ref(), &table_files, &file_statistics);

            if rowid_in_proj {
                let rowid_idx = rowid_idx.unwrap();
                let user_proj: Vec<usize> = projection
                    .cloned()
                    .unwrap_or_else(|| (0..self.schema.fields().len()).collect());
                for table_file in table_files {
                    let exec = if self.needs_snapshot_filter(table_file) {
                        let output_schema =
                            self.output_schema_for_projection(&user_proj, rowid_idx);
                        self.build_exec_for_partial_file(state, table_file, output_schema)
                            .await?
                    } else {
                        self.build_exec_for_file_with_rowid(
                            state,
                            table_file,
                            &file_statistics,
                            &user_proj,
                            rowid_idx,
                            limit,
                        )
                        .await?
                    };
                    execs.push(exec);
                }
                if last_page {
                    break;
                }
                continue;
            }

            let (needs_filter, rest): (Vec<_>, Vec<_>) = table_files
                .into_iter()
                .partition(|table_file| self.needs_snapshot_filter(table_file));
            let (files_with_deletes, files_without_deletes): (Vec<_>, Vec<_>) = rest
                .into_iter()
                .partition(|table_file| table_file.delete_file.is_some());

            if !files_without_deletes.is_empty() {
                execs.push(
                    self.build_exec_for_files_without_deletes(
                        state,
                        &files_without_deletes,
                        &file_statistics,
                        projection,
                        limit,
                    )
                    .await?,
                );
            }
            for table_file in files_with_deletes {
                execs.push(
                    self.build_exec_for_file_with_deletes(
                        state,
                        table_file,
                        &file_statistics,
                        projection,
                        limit,
                    )
                    .await?,
                );
            }
            for table_file in needs_filter {
                let output_schema = match projection {
                    Some(indices) => Arc::new(self.schema.project(indices)?),
                    None => self.schema.clone(),
                };
                execs.push(
                    self.build_exec_for_partial_file(state, table_file, output_schema)
                        .await?,
                );
            }
            if last_page {
                break;
            }
        }

        if rowid_in_proj {
            if execs.is_empty() {
                use datafusion::physical_plan::empty::EmptyExec;
                let rowid_idx = rowid_idx.unwrap();
                let user_proj: Vec<usize> = projection
                    .cloned()
                    .unwrap_or_else(|| (0..self.schema.fields().len()).collect());
                let schema = self.output_schema_for_projection(&user_proj, rowid_idx);
                return Ok(Arc::new(EmptyExec::new(schema)));
            }
            return combine_execution_plans(execs);
        }

        // Inlined data: rows DuckDB's data-inlining optimization stored directly
        // in the catalog (not in Parquet). Union them in so SELECT / COUNT(*)
        // include them. Providers without inlined data — or that don't implement
        // the read — return empty, so this is a no-op for ordinary catalogs.
        // (Phase 1: applies on this non-rowid read path; only the SQLite provider
        // surfaces inlined rows today.)
        let inlined =
            self.provider
                .get_inlined_data(self.table_id, self.snapshot_id, &self.columns)?;
        if inlined.iter().any(|b| b.num_rows() > 0) {
            let exec = MemorySourceConfig::try_new_exec(
                &[inlined],
                self.physical_schema.clone(),
                projection.cloned(),
            )?;
            execs.push(exec);
        }

        // Handle empty tables (no data files)
        if execs.is_empty() {
            use datafusion::physical_plan::empty::EmptyExec;
            let projected_schema = match projection {
                Some(indices) => Arc::new(self.schema.project(indices)?),
                None => self.schema.clone(),
            };
            return Ok(Arc::new(EmptyExec::new(projected_schema)));
        }

        // Combine execution plans
        combine_execution_plans(execs)
    }

    #[cfg(feature = "write")]
    async fn insert_into(
        &self,
        _state: &dyn Session,
        input: Arc<dyn ExecutionPlan>,
        insert_op: InsertOp,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let writer = self.writer.as_ref().ok_or_else(|| {
            DataFusionError::Plan(
                "Table is read-only. Use DuckLakeCatalog::with_writer() to enable writes."
                    .to_string(),
            )
        })?;

        let schema_name = self.schema_name.as_ref().ok_or_else(|| {
            DataFusionError::Internal("Schema name not set for writable table".to_string())
        })?;

        let write_mode = match insert_op {
            InsertOp::Append => WriteMode::Append,
            InsertOp::Overwrite | InsertOp::Replace => WriteMode::Replace,
        };

        // Resolve the partition spec at the CURRENT catalog head, NOT the snapshot
        // this table provider was pinned to when it was opened. A write always
        // commits at the head, so it must honor the spec live there; using the
        // pinned `self.partition_spec` would ignore a spec set/reset applied after
        // this provider was created (e.g. `execute_ducklake_sql(SET PARTITIONED BY)`
        // then `INSERT` in the same session) and could stamp a retired partition_id.
        // (`self.partition_spec`, pinned, is still used for read pruning, which is
        // snapshot-bound.)
        //
        // A transform we cannot PRODUCE (bucket/unknown) makes a partitioned INSERT
        // unsupported — reject rather than silently writing unpartitioned files that
        // would violate the spec.
        let head_snapshot = self
            .provider
            .get_current_snapshot()
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        let live_spec = self
            .provider
            .get_partition_spec(self.table_id, head_snapshot)
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        let partition = match live_spec.as_ref() {
            None => None,
            Some(spec) => {
                let mut keys = Vec::with_capacity(spec.columns.len());
                for column in &spec.columns {
                    if !column.transform.is_producible() {
                        return Err(DataFusionError::NotImplemented(format!(
                            "INSERT into a table partitioned by '{}' is not supported",
                            column.transform.to_catalog_string()
                        )));
                    }
                    let index = self
                        .columns
                        .iter()
                        .position(|c| c.column_id == column.column_id)
                        .ok_or_else(|| {
                            DataFusionError::Internal(format!(
                                "partition column_id {} not found in table schema",
                                column.column_id
                            ))
                        })?;
                    keys.push(crate::insert_exec::PartitionWriteKey {
                        input_index: index,
                        name: self.physical_schema.field(index).name().to_string(),
                        transform: column.transform.clone(),
                    });
                }
                Some(crate::insert_exec::PartitionWriteSpec {
                    partition_id: spec.partition_id,
                    keys,
                })
            },
        };

        Ok(Arc::new(DuckLakeInsertExec::new(
            input,
            Arc::clone(writer),
            schema_name.clone(),
            self.table_name.clone(),
            self.schema(),
            write_mode,
            self.object_store_url.clone(),
            partition,
        )))
    }

    /// Plan an `UPDATE t SET col = expr [, ...] [WHERE ...]`.
    ///
    /// `assignments` are `(column_name, new_value_expr)` for each SET (identity
    /// `c = c` assignments are already dropped by the planner). `filters` are the
    /// unqualified, AND-conjunctive WHERE predicates; an empty `filters` updates
    /// every live row. The returned [`DuckLakeUpdateExec`] performs the update at
    /// execute time and yields a single `count: UInt64` row — planning here is
    /// side-effect-free (no scans, no writes), so `EXPLAIN` never mutates data.
    #[cfg(feature = "write")]
    async fn update(
        &self,
        state: &dyn Session,
        assignments: Vec<(String, Expr)>,
        filters: Vec<Expr>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let writer = self.writer.as_ref().ok_or_else(|| {
            DataFusionError::Plan(
                "Table is read-only. Use DuckLakeCatalog::with_writer() to enable writes."
                    .to_string(),
            )
        })?;
        let schema_name = self.schema_name.as_ref().ok_or_else(|| {
            DataFusionError::Internal("Schema name not set for writable table".to_string())
        })?;

        // DuckDB / MySQL metadata writers do not implement the atomic
        // append-with-deletes commit UPDATE needs. Reject up front rather than
        // rewriting files and only failing at commit.
        if !writer.supports_update() {
            return Err(DataFusionError::NotImplemented(
                "UPDATE not supported on this metadata backend".to_string(),
            ));
        }

        // Assignment / filter expressions reference the table's DATA columns
        // (unqualified), never the synthetic `rowid`. Plan them against the
        // physical schema so column indices line up with the scanned batches.
        let df_schema = DFSchema::try_from(self.physical_schema.as_ref().clone())
            .map_err(|e| DataFusionError::External(Box::new(e)))?;

        let mut phys_assignments: Vec<(usize, Arc<dyn PhysicalExpr>)> =
            Vec::with_capacity(assignments.len());
        for (col_name, expr) in assignments {
            let idx = self.physical_schema.index_of(&col_name).map_err(|_| {
                DataFusionError::Plan(format!(
                    "UPDATE assignment targets unknown column '{col_name}'"
                ))
            })?;
            let pexpr = state.create_physical_expr(expr, &df_schema)?;
            phys_assignments.push((idx, pexpr));
        }

        // AND the WHERE predicates into one physical expression; empty => update
        // all rows (represented as `None`).
        let mut predicate: Option<Arc<dyn PhysicalExpr>> = None;
        for f in filters {
            let pe = state.create_physical_expr(f, &df_schema)?;
            predicate = Some(match predicate {
                None => pe,
                Some(prev) => Arc::new(BinaryExpr::new(prev, Operator::And, pe)),
            });
        }

        // Build the per-file positional read plans now (plan time). This reads
        // parquet footers + live delete positions — the same plan-time reads
        // `scan()` does — but no data scan and no mutation happen here; the exec
        // collects each scan and performs the rewrite + atomic commit at execute
        // time.
        let table_files = self
            .files()
            .map_err(|error| DataFusionError::External(Box::new(error)))?;
        let mut scans = Vec::with_capacity(table_files.len());
        for tf in &table_files {
            scans.push(self.build_update_scan(state, tf).await?);
        }

        Ok(Arc::new(DuckLakeUpdateExec::new(
            Arc::new(self.read_only_clone()),
            Arc::clone(writer),
            schema_name.clone(),
            self.table_name.clone(),
            scans,
            phys_assignments,
            predicate,
            self.object_store_url.clone(),
        )))
    }

    /// Plan a `DELETE FROM <table> [WHERE ...]`.
    ///
    /// `filters` are the already-analyzed, unqualified, AND-conjunctive
    /// predicates over this table's own columns (DataFusion strips qualifiers and
    /// dedups them). An empty `filters` means no `WHERE` => delete ALL rows.
    ///
    /// Returns a [`DuckLakeDeleteExec`] that performs the delete when executed
    /// (positional-delete files + one atomic metadata commit, or a metadata-only
    /// truncate for delete-all) and yields a single `count: UInt64` row. All
    /// mutation happens at execute time, so planning (e.g. `EXPLAIN`) is
    /// side-effect free.
    ///
    /// The catalog pins its snapshot at creation, so a session sees one
    /// generation for its lifetime: re-open the catalog between mutating
    /// statements. See the [`delete_exec`](crate::delete_exec) module docs
    /// ("Session lifecycle") for why a second in-session `DELETE` can conflict.
    #[cfg(feature = "write")]
    async fn delete_from(
        &self,
        state: &dyn Session,
        filters: Vec<Expr>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        use datafusion::logical_expr::utils::conjunction;

        let writer = self.writer.as_ref().ok_or_else(|| {
            DataFusionError::Plan(
                "Table is read-only. Use DuckLakeCatalog::with_writer() to enable writes."
                    .to_string(),
            )
        })?;
        let schema_name = self.schema_name.as_ref().ok_or_else(|| {
            DataFusionError::Internal("Schema name not set for writable table".to_string())
        })?;

        // Build the physical predicate. Empty `filters` (no WHERE) => delete ALL,
        // signalled by `None` and handled as a metadata-only truncate. We resolve
        // column references against the PHYSICAL schema (no synthetic `rowid`):
        // `resolve_positions` evaluates the predicate index-based against the
        // physically-read columns in logical order, so the physical expression's
        // column indices must line up with `physical_schema`. A predicate that
        // references a column absent from `physical_schema` (e.g. the synthetic
        // `rowid`) fails here rather than mis-deleting.
        let predicate = match conjunction(filters) {
            None => None,
            Some(expr) => {
                let df_schema =
                    datafusion::common::DFSchema::try_from(self.physical_schema.as_ref().clone())?;
                Some(state.create_physical_expr(expr, &df_schema)?)
            },
        };

        // The delete work (positional reads, delete-file writes, atomic commit)
        // MUST run at execute time — planning a DELETE (e.g. `EXPLAIN`) must not
        // mutate. `DuckLakeDeleteExec` captures the concrete `SessionState` to
        // drive the positional reads at execute time (a bare `TaskContext` cannot
        // build physical exprs / sub-plans), plus a clone of this table for its
        // reader methods.
        let session_state = state
            .as_any()
            .downcast_ref::<datafusion::execution::SessionState>()
            .ok_or_else(|| {
                DataFusionError::NotImplemented(
                    "DELETE on a DuckLake table requires a DataFusion SessionState session"
                        .to_string(),
                )
            })?
            .clone();

        Ok(Arc::new(DuckLakeDeleteExec::new(
            Arc::new(self.clone()),
            session_state,
            predicate,
            Arc::clone(writer),
            schema_name.clone(),
            self.table_name.clone(),
            self.table_id,
            self.snapshot_id,
            self.object_store_url.clone(),
        )))
    }
}

/// Combines multiple execution plans into a single plan
fn combine_execution_plans(
    execs: Vec<Arc<dyn ExecutionPlan>>,
) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
    if execs.len() == 1 {
        Ok(execs.into_iter().next().unwrap())
    } else {
        use datafusion::physical_plan::union::UnionExec;
        UnionExec::try_new(execs)
    }
}

/// Extract deleted row positions from a delete file RecordBatch
///
/// Delete files have schema: (file_path: VARCHAR, pos: INT64)
/// We only extract the "pos" column - the "file_path" column is metadata/documentation
/// only (for Iceberg compatibility). The metadata catalog already tells us which delete
/// file is associated with which data file.
fn extract_deleted_positions_from_batch(
    batch: &RecordBatch,
    positions: &mut HashSet<i64>,
) -> DataFusionResult<()> {
    // Get the pos column index by name (not magic number)
    let schema = batch.schema();
    let pos_idx = schema.index_of(DELETE_POS_COL)?;

    // Get the pos column
    let pos_array = batch
        .column(pos_idx)
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| {
            DataFusionError::Internal(format!("{} column not found or wrong type", DELETE_POS_COL))
        })?;

    // Extract all non-null positions
    for i in 0..batch.num_rows() {
        if !pos_array.is_null(i) {
            positions.insert(pos_array.value(i));
        }
    }

    Ok(())
}

/// Check if a DataFusion error is caused by an object store NotFound error.
fn is_object_store_not_found(err: &DataFusionError) -> bool {
    if let DataFusionError::ObjectStore(os_err) = err {
        return matches!(&**os_err, object_store::Error::NotFound { .. });
    }
    let mut source = std::error::Error::source(err);
    while let Some(e) = source {
        if let Some(os_err) = e.downcast_ref::<object_store::Error>() {
            return matches!(os_err, object_store::Error::NotFound { .. });
        }
        source = e.source();
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata_provider::{
        ColumnWithTable, DataFileChange, DeleteFileChange, FileWithTable, SchemaMetadata,
        SnapshotMetadata, TableMetadata, TableWithSchema,
    };
    use datafusion::prelude::{SessionContext, col, lit};
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Debug, Default)]
    struct LazyMillionFileProvider {
        eager_file_reads: AtomicUsize,
        max_page: AtomicUsize,
        page_calls: AtomicUsize,
    }

    impl MetadataProvider for LazyMillionFileProvider {
        fn get_current_snapshot(&self) -> Result<i64> {
            Ok(1)
        }

        fn get_data_path(&self) -> Result<String> {
            Ok("memory:///".to_string())
        }

        fn list_snapshots(&self) -> Result<Vec<SnapshotMetadata>> {
            unimplemented!()
        }

        fn list_schemas(&self, _snapshot_id: i64) -> Result<Vec<SchemaMetadata>> {
            unimplemented!()
        }

        fn list_tables(&self, _schema_id: i64, _snapshot_id: i64) -> Result<Vec<TableMetadata>> {
            unimplemented!()
        }

        fn get_table_structure(
            &self,
            _table_id: i64,
            _snapshot_id: i64,
        ) -> Result<Vec<DuckLakeTableColumn>> {
            Ok(vec![DuckLakeTableColumn::new(
                1,
                "id".to_string(),
                "bigint".to_string(),
                false,
            )])
        }

        fn get_table_files_for_select(
            &self,
            _table_id: i64,
            _snapshot_id: i64,
        ) -> Result<Vec<DuckLakeTableFile>> {
            self.eager_file_reads.fetch_add(1, Ordering::Relaxed);
            panic!("the eager file API must not be used while planning a scan")
        }

        fn get_table_summary_statistics(
            &self,
            _table_id: i64,
            _snapshot_id: i64,
        ) -> Result<DuckLakeStatistics> {
            Ok(DuckLakeStatistics {
                table: Some(crate::metadata_provider::DuckLakeTableStatistics {
                    record_count: Some(1_000_000),
                    file_size_bytes: Some(1_000_000),
                }),
                columns: vec![DuckLakeTableColumnStatistics {
                    column_id: 1,
                    contains_null: Some(false),
                    min_value: Some("1".to_string()),
                    max_value: Some("1000000".to_string()),
                    contains_nan: None,
                    column_size_bytes: Some(1_000_000),
                    bounds_are_exact: true,
                }],
                ..Default::default()
            })
        }

        fn get_table_file_metadata_page(
            &self,
            _table_id: i64,
            snapshot_id: i64,
            after_data_file_id: Option<i64>,
            limit: usize,
        ) -> Result<Vec<DuckLakeFileMetadata>> {
            self.page_calls.fetch_add(1, Ordering::Relaxed);
            let start = after_data_file_id.unwrap_or(0) + 1;
            let end = (start + i64::try_from(limit).unwrap()).min(1_000_001);
            let page: Vec<_> = (start..end)
                .map(|data_file_id| DuckLakeFileMetadata {
                    file: DuckLakeTableFile {
                        data_file_id,
                        file: DuckLakeFileData::new(
                            format!("file-{data_file_id}.parquet"),
                            true,
                            1,
                        ),
                        delete_file_id: None,
                        delete_file: None,
                        row_id_start: Some(data_file_id - 1),
                        snapshot_id: Some(snapshot_id),
                        begin_snapshot: Some(1),
                        schema_version: Some(0),
                        partial_max: None,
                        max_row_count: Some(1),
                        delete_count: None,
                        partition_id: None,
                        partition_values: Vec::new(),
                    },
                    column_statistics: vec![DuckLakeFileColumnStatistics {
                        data_file_id,
                        column_id: 1,
                        column_size_bytes: Some(1),
                        value_count: Some(1),
                        null_count: Some(0),
                        min_value: Some(data_file_id.to_string()),
                        max_value: Some(data_file_id.to_string()),
                        contains_nan: None,
                    }],
                })
                .collect();
            self.max_page.fetch_max(page.len(), Ordering::Relaxed);
            Ok(page)
        }

        fn get_schema_by_name(
            &self,
            _name: &str,
            _snapshot_id: i64,
        ) -> Result<Option<SchemaMetadata>> {
            unimplemented!()
        }

        fn get_table_by_name(
            &self,
            _schema_id: i64,
            _name: &str,
            _snapshot_id: i64,
        ) -> Result<Option<TableMetadata>> {
            unimplemented!()
        }

        fn table_exists(&self, _schema_id: i64, _name: &str, _snapshot_id: i64) -> Result<bool> {
            unimplemented!()
        }

        fn list_all_tables(&self, _snapshot_id: i64) -> Result<Vec<TableWithSchema>> {
            unimplemented!()
        }

        fn list_all_columns(&self, _snapshot_id: i64) -> Result<Vec<ColumnWithTable>> {
            unimplemented!()
        }

        fn list_all_files(&self, _snapshot_id: i64) -> Result<Vec<FileWithTable>> {
            unimplemented!()
        }

        fn get_data_files_added_between_snapshots(
            &self,
            _table_id: i64,
            _start_snapshot: i64,
            _end_snapshot: i64,
        ) -> Result<Vec<DataFileChange>> {
            unimplemented!()
        }

        fn get_delete_files_added_between_snapshots(
            &self,
            _table_id: i64,
            _start_snapshot: i64,
            _end_snapshot: i64,
        ) -> Result<Vec<DeleteFileChange>> {
            unimplemented!()
        }
    }

    #[test]
    fn planning_prunes_a_lazy_million_file_fixture_in_bounded_pages() -> Result<()> {
        let provider = Arc::new(LazyMillionFileProvider::default());
        let table = DuckLakeTable::new(
            1,
            "events",
            provider.clone(),
            1,
            Arc::new(ObjectStoreUrl::parse("memory://").unwrap()),
            String::new(),
        )?;
        assert_eq!(provider.eager_file_reads.load(Ordering::Relaxed), 0);
        let statistics = table.statistics().unwrap();
        assert_eq!(statistics.num_rows, Precision::Exact(1_000_000));
        assert_eq!(
            statistics.column_statistics[0].min_value,
            Precision::Exact(ScalarValue::Int64(Some(1)))
        );
        assert_eq!(
            statistics.column_statistics[0].max_value,
            Precision::Exact(ScalarValue::Int64(Some(1_000_000)))
        );
        assert_eq!(
            statistics.column_statistics[0].byte_size,
            Precision::Inexact(1_000_000)
        );

        let state = SessionContext::new().state();
        let filters = [col("id").eq(lit(999_999_i64))];
        let pruning = table
            .file_pruning_predicate(&state, &filters)
            .unwrap()
            .unwrap();
        let mut after = None;
        let mut retained = Vec::new();
        loop {
            let metadata =
                provider.get_table_file_metadata_page(1, 1, after, FILE_METADATA_BATCH_SIZE)?;
            if metadata.is_empty() {
                break;
            }
            after = metadata.last().map(|entry| entry.file.data_file_id);
            let files: Vec<_> = metadata.iter().map(|entry| entry.file.clone()).collect();
            let catalog_statistics = metadata
                .into_iter()
                .flat_map(|entry| entry.column_statistics)
                .collect();
            let (_, file_statistics) = build_datafusion_statistics(
                table.physical_schema.as_ref(),
                &table.columns,
                &files,
                DuckLakeStatistics {
                    files: catalog_statistics,
                    ..Default::default()
                },
                false,
                true,
            );
            retained.extend(
                table
                    .prune_table_files(Some(&pruning), &files, &file_statistics)
                    .into_iter()
                    .map(|file| file.data_file_id),
            );
        }

        assert_eq!(provider.max_page.load(Ordering::Relaxed), 4_096);
        assert_eq!(provider.page_calls.load(Ordering::Relaxed), 246);
        assert_eq!(retained, vec![999_999]);
        assert_eq!(provider.eager_file_reads.load(Ordering::Relaxed), 0);
        Ok(())
    }

    #[test]
    fn summary_bounds_with_live_deletes_remain_inexact() -> Result<()> {
        let columns =
            vec![DuckLakeTableColumn::new(1, "id".to_string(), "integer".to_string(), false)];
        let schema = build_arrow_schema(&columns)?;
        let (statistics, _) = build_datafusion_statistics(
            &schema,
            &columns,
            &[],
            DuckLakeStatistics {
                columns: vec![DuckLakeTableColumnStatistics {
                    column_id: 1,
                    contains_null: Some(false),
                    min_value: Some("0".to_string()),
                    max_value: Some("9".to_string()),
                    contains_nan: None,
                    column_size_bytes: Some(40),
                    bounds_are_exact: false,
                }],
                ..Default::default()
            },
            true,
            false,
        );

        let column = &statistics.column_statistics[0];
        assert_eq!(
            column.min_value,
            Precision::Inexact(ScalarValue::Int32(Some(0)))
        );
        assert_eq!(
            column.max_value,
            Precision::Inexact(ScalarValue::Int32(Some(9)))
        );
        assert_eq!(column.byte_size, Precision::Inexact(40));
        Ok(())
    }

    #[test]
    fn float_file_max_gated_by_contains_nan() -> Result<()> {
        let columns =
            vec![DuckLakeTableColumn::new(1, "x".to_string(), "double".to_string(), false)];
        let schema = build_arrow_schema(&columns)?;
        let mut file =
            DuckLakeTableFile::new(DuckLakeFileData::new("f.parquet".to_string(), true, 1));
        file.data_file_id = 7;

        for (contains_nan, max_usable) in [(None, false), (Some(true), false), (Some(false), true)]
        {
            let (table_stats, file_stats) = build_datafusion_statistics(
                &schema,
                &columns,
                std::slice::from_ref(&file),
                DuckLakeStatistics {
                    files: vec![DuckLakeFileColumnStatistics {
                        data_file_id: 7,
                        column_id: 1,
                        column_size_bytes: Some(16),
                        value_count: Some(2),
                        null_count: Some(0),
                        min_value: Some("1.0".to_string()),
                        max_value: Some("2.0".to_string()),
                        contains_nan,
                    }],
                    ..Default::default()
                },
                false,
                true,
            );

            // min is a valid lower bound regardless of NaN state — NaN sorts
            // above every value, so it can never undercut the recorded min.
            let per_file = &file_stats[&7].column_statistics[0];
            let table_col = &table_stats.column_statistics[0];
            assert_eq!(
                per_file.min_value,
                Precision::Exact(ScalarValue::Float64(Some(1.0))),
                "contains_nan={contains_nan:?}"
            );
            assert_eq!(
                table_col.min_value,
                Precision::Exact(ScalarValue::Float64(Some(1.0))),
                "contains_nan={contains_nan:?}"
            );

            let expected_max = if max_usable {
                Precision::Exact(ScalarValue::Float64(Some(2.0)))
            } else {
                Precision::Absent
            };
            assert_eq!(
                per_file.max_value, expected_max,
                "contains_nan={contains_nan:?}"
            );
            assert_eq!(
                table_col.max_value, expected_max,
                "contains_nan={contains_nan:?}"
            );
        }
        Ok(())
    }

    #[test]
    fn float_rollup_max_gated_by_contains_nan() -> Result<()> {
        let columns =
            vec![DuckLakeTableColumn::new(1, "x".to_string(), "double".to_string(), false)];
        let schema = build_arrow_schema(&columns)?;

        for (contains_nan, max_usable) in [(None, false), (Some(true), false), (Some(false), true)]
        {
            let (statistics, _) = build_datafusion_statistics(
                &schema,
                &columns,
                &[],
                DuckLakeStatistics {
                    columns: vec![DuckLakeTableColumnStatistics {
                        column_id: 1,
                        contains_null: Some(false),
                        min_value: Some("1.0".to_string()),
                        max_value: Some("2.0".to_string()),
                        contains_nan,
                        column_size_bytes: Some(16),
                        bounds_are_exact: true,
                    }],
                    ..Default::default()
                },
                true,
                false,
            );

            let column = &statistics.column_statistics[0];
            assert_eq!(
                column.min_value,
                Precision::Exact(ScalarValue::Float64(Some(1.0))),
                "contains_nan={contains_nan:?}"
            );
            let expected_max = if max_usable {
                Precision::Exact(ScalarValue::Float64(Some(2.0)))
            } else {
                Precision::Absent
            };
            assert_eq!(
                column.max_value, expected_max,
                "contains_nan={contains_nan:?}"
            );
        }
        Ok(())
    }

    #[test]
    fn float_table_aggregate_max_needs_all_files_nan_free() -> Result<()> {
        let columns =
            vec![DuckLakeTableColumn::new(1, "x".to_string(), "double".to_string(), false)];
        let schema = build_arrow_schema(&columns)?;
        let mut file_a =
            DuckLakeTableFile::new(DuckLakeFileData::new("a.parquet".to_string(), true, 1));
        file_a.data_file_id = 1;
        let mut file_b =
            DuckLakeTableFile::new(DuckLakeFileData::new("b.parquet".to_string(), true, 1));
        file_b.data_file_id = 2;
        let table_files = vec![file_a, file_b];

        let stat =
            |data_file_id, min: &str, max: &str, contains_nan| DuckLakeFileColumnStatistics {
                data_file_id,
                column_id: 1,
                column_size_bytes: Some(16),
                value_count: Some(2),
                null_count: Some(0),
                min_value: Some(min.to_string()),
                max_value: Some(max.to_string()),
                contains_nan,
            };
        let build = |nan_b| {
            build_datafusion_statistics(
                &schema,
                &columns,
                &table_files,
                DuckLakeStatistics {
                    files: vec![stat(1, "1.0", "2.0", Some(false)), stat(2, "0.5", "3.0", nan_b)],
                    ..Default::default()
                },
                false,
                true,
            )
            .0
        };

        // Both files known NaN-free: bounds fold across the files.
        let statistics = build(Some(false));
        assert_eq!(
            statistics.column_statistics[0].min_value,
            Precision::Exact(ScalarValue::Float64(Some(0.5)))
        );
        assert_eq!(
            statistics.column_statistics[0].max_value,
            Precision::Exact(ScalarValue::Float64(Some(3.0)))
        );

        // One file NaN-unknown: its max is untrusted, so the aggregate max
        // degrades to unknown; the min still folds from both files.
        let statistics = build(None);
        assert_eq!(
            statistics.column_statistics[0].min_value,
            Precision::Exact(ScalarValue::Float64(Some(0.5)))
        );
        assert_eq!(statistics.column_statistics[0].max_value, Precision::Absent);
        Ok(())
    }

    #[test]
    fn test_validated_file_size_positive() {
        assert_eq!(validated_file_size(0, "test.parquet").unwrap(), 0);
        assert_eq!(validated_file_size(1024, "test.parquet").unwrap(), 1024);
        assert_eq!(
            validated_file_size(i64::MAX, "test.parquet").unwrap(),
            i64::MAX as u64
        );
    }

    #[test]
    fn test_validated_file_size_negative() {
        let err = validated_file_size(-1, "data/test.parquet").unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("-1"),
            "Error should contain the negative value: {}",
            msg
        );
        assert!(
            msg.contains("data/test.parquet"),
            "Error should contain the file path: {}",
            msg
        );
    }

    #[test]
    fn test_validated_file_size_large_negative() {
        let err = validated_file_size(i64::MIN, "bad.parquet").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("bad.parquet"));
        assert!(msg.contains(&i64::MIN.to_string()));
    }

    #[test]
    fn test_validated_record_count_positive() {
        assert_eq!(validated_record_count(0, "test.parquet").unwrap(), 0);
        assert_eq!(validated_record_count(100, "test.parquet").unwrap(), 100);
        assert_eq!(
            validated_record_count(i64::MAX, "test.parquet").unwrap(),
            i64::MAX as u64
        );
    }

    #[test]
    fn test_validated_record_count_negative() {
        let err = validated_record_count(-1, "data/test.parquet").unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("-1"),
            "Error should contain the negative value: {}",
            msg
        );
        assert!(
            msg.contains("data/test.parquet"),
            "Error should contain the file path: {}",
            msg
        );
        assert!(
            msg.contains("record_count"),
            "Error should mention record_count: {}",
            msg
        );
    }

    #[test]
    fn test_validated_record_count_large_negative() {
        let err = validated_record_count(i64::MIN, "bad.parquet").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("bad.parquet"));
        assert!(msg.contains(&i64::MIN.to_string()));
    }

    #[test]
    fn test_parse_ducklake_statistic_encodings() {
        let boolean = DuckLakeTableColumn::new(1, "flag".to_string(), "boolean".to_string(), true);
        assert_eq!(
            parse_statistic_scalar("1", &boolean, &DataType::Boolean),
            Some(ScalarValue::Boolean(Some(true)))
        );

        let blob = DuckLakeTableColumn::new(2, "bytes".to_string(), "blob".to_string(), true);
        assert_eq!(
            parse_statistic_scalar("68656C6C6F", &blob, &DataType::BinaryView),
            Some(ScalarValue::BinaryView(Some(b"hello".to_vec())))
        );

        let uuid = DuckLakeTableColumn::new(3, "id".to_string(), "uuid".to_string(), true);
        assert_eq!(
            parse_statistic_scalar(
                "550e8400-e29b-41d4-a716-446655440000",
                &uuid,
                &DataType::FixedSizeBinary(16),
            ),
            Some(ScalarValue::FixedSizeBinary(
                16,
                Some(vec![
                    0x55, 0x0e, 0x84, 0x00, 0xe2, 0x9b, 0x41, 0xd4, 0xa7, 0x16, 0x44, 0x66, 0x55,
                    0x44, 0x00, 0x00,
                ]),
            ))
        );

        let decimal =
            DuckLakeTableColumn::new(4, "amount".to_string(), "decimal(10,2)".to_string(), true);
        assert_eq!(
            parse_statistic_scalar("123.45", &decimal, &DataType::Decimal128(10, 2)),
            Some(ScalarValue::Decimal128(Some(12_345), 10, 2))
        );
    }
}
