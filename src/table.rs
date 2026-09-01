//! DuckLake table provider implementation

use std::collections::{HashMap, HashSet};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use crate::Result;
use crate::column_rename::ColumnRenameExec;
use crate::delete_filter::DeleteFilterExec;
use crate::inlined_filter::translate_inlined_filters;
use crate::metadata_provider::{
    DuckLakeFileColumnStatistics, DuckLakeFileData, DuckLakeFileMetadata, DuckLakeNameMapping,
    DuckLakeStatistics, DuckLakeTableColumn, DuckLakeTableColumnStatistics, DuckLakeTableField,
    DuckLakeTableFile, FILE_METADATA_BATCH_SIZE, INLINED_DATA_REMEDIATION, MetadataProvider,
};
use crate::nan_pruning_barrier::NanPruningBarrierExec;
use crate::partition::PartitionSpec;
use crate::path_resolver::resolve_path;
use crate::row_id::{
    ROW_ID_PARQUET_FIELD_ID, ROWID_COLUMN_NAME, RowIdExec, SNAPSHOT_ID_PARQUET_FIELD_ID,
    positional_table_schema, positional_table_schema_reserving, rowid_field,
};
use crate::snapshot_filter::SnapshotFilterExec;
use crate::stats_filter::{self, StatsFilter};
use crate::types::{
    DuckLakeDefaultExprAdapterFactory, INITIAL_DEFAULT_METADATA_KEY,
    build_arrow_schema_from_fields, build_read_schema_with_field_id_mapping,
    build_read_schema_with_field_id_mapping_from_schema, build_read_schema_with_name_mapping,
    ducklake_to_arrow_type, extract_parquet_field_ids, parse_ducklake_default_scalar,
    parse_ducklake_scalar,
};

#[cfg(feature = "write")]
use crate::delete_exec::DuckLakeDeleteExec;
#[cfg(feature = "write")]
use crate::insert_exec::DuckLakeInsertExec;
#[cfg(feature = "write")]
use crate::metadata_writer::{MetadataWriter, WriteMode};
#[cfg(feature = "write")]
use crate::update_exec::DuckLakeUpdateExec;
use datafusion::common::DFSchema;
use datafusion::common::pruning::{PrunableStatistics, PruningStatistics};
#[cfg(feature = "write")]
use datafusion::logical_expr::Operator;
use datafusion::physical_expr::PhysicalExpr;
#[cfg(feature = "write")]
use datafusion::physical_expr::expressions::BinaryExpr;
use datafusion::physical_optimizer::pruning::{PruningPredicate, PruningPredicateBuilder};

#[cfg(feature = "encryption")]
use crate::encryption::DuckLakeEncryptionFactory;
use arrow::array::{Array, ArrayRef, BooleanArray, Int64Array};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use async_trait::async_trait;
use datafusion::catalog::{Session, TableProvider};
use datafusion::common::stats::Precision;
use datafusion::common::{Column, ColumnStatistics, ScalarValue, Statistics};
use datafusion::datasource::listing::PartitionedFile;
use datafusion::datasource::memory::MemorySourceConfig;
use datafusion::datasource::physical_plan::{
    FileGroup, FileScanConfigBuilder, FileSource, ParquetSource,
};
use datafusion::datasource::source::DataSourceExec;
use datafusion::datasource::table_schema::TableSchema;
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::execution::object_store::ObjectStoreUrl;
#[cfg(feature = "write")]
use datafusion::logical_expr::dml::InsertOp;
use datafusion::logical_expr::{Expr, TableProviderFilterPushDown, TableType};
use datafusion::physical_plan::ExecutionPlan;
use futures::StreamExt;
use object_store::ObjectStoreExt;
use object_store::path::Path as ObjectPath;
use parquet::arrow::ParquetRecordBatchStreamBuilder;
// Deprecated in parquet 59 in favour of a hand-rolled `AsyncFileReader`. The
// call sites below read a footer once each, so the replacement (a custom
// reader with its own I/O coalescing) buys nothing here.
// See https://github.com/apache/arrow-rs/issues/10308.
#[allow(deprecated)]
use parquet::arrow::async_reader::ParquetObjectReader;
use percent_encoding::percent_decode_str;

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

struct FileMetadataPages<'a> {
    provider: &'a dyn MetadataProvider,
    table_id: i64,
    snapshot_id: i64,
    after_data_file_id: Option<i64>,
    /// Predicate the catalog may use to skip files whose statistics prove they
    /// cannot match, lowered by [`crate::stats_filter`].
    ///
    /// It has to be applied inside the provider's query, before its `LIMIT`.
    /// Filtering a page here instead would break the keyset cursor below: a
    /// page whose candidates all failed the filter would come back empty, the
    /// iteration would stop, and every matching file beyond it would never be
    /// visited. That is why this is threaded to the provider rather than
    /// applied to what the provider returns.
    filter: Option<&'a StatsFilter>,
    page_name: &'static str,
    finished: bool,
}

impl Iterator for FileMetadataPages<'_> {
    type Item = Result<Vec<DuckLakeFileMetadata>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }
        let metadata = match self.provider.get_table_file_metadata_page_filtered(
            self.table_id,
            self.snapshot_id,
            self.after_data_file_id,
            FILE_METADATA_BATCH_SIZE,
            self.filter,
        ) {
            Ok(metadata) => metadata,
            Err(error) => {
                self.finished = true;
                return Some(Err(error));
            },
        };
        if metadata.is_empty() {
            self.finished = true;
            return None;
        }
        if metadata.len() > FILE_METADATA_BATCH_SIZE {
            self.finished = true;
            return Some(Err(crate::DuckLakeError::InvalidConfig(format!(
                "metadata provider returned {} files for a {}-file {} page",
                metadata.len(),
                FILE_METADATA_BATCH_SIZE,
                self.page_name,
            ))));
        }

        let next_after = metadata.last().unwrap().file.data_file_id;
        if self
            .after_data_file_id
            .is_some_and(|after| next_after <= after)
        {
            self.finished = true;
            return Some(Err(crate::DuckLakeError::InvalidConfig(
                "metadata provider returned a non-advancing file page".to_string(),
            )));
        }
        self.after_data_file_id = Some(next_after);
        // A short page means the provider ran out of rows, not that its filter
        // rejected some: `filter` narrows the query ahead of its `LIMIT`, so a
        // full page's worth of matches is always returned when one exists.
        self.finished = metadata.len() < FILE_METADATA_BATCH_SIZE;
        Some(Ok(metadata))
    }
}

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

    let parsed = parse_ducklake_scalar(value, data_type);

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

fn validate_column_defaults(columns: &[DuckLakeTableColumn]) -> Result<HashMap<String, Expr>> {
    let mut defaults = HashMap::new();
    for column in columns {
        let data_type = ducklake_to_arrow_type(&column.column_type)?;
        if let Some(value) = &column.initial_default
            && parse_ducklake_default_scalar(value, &data_type).is_none()
        {
            return Err(crate::DuckLakeError::InvalidConfig(format!(
                "Cannot decode initial_default '{value}' for column '{}' as {}",
                column.column_name, data_type
            )));
        }

        let Some(value) = &column.default_value else {
            continue;
        };
        match column.default_value_type.as_deref() {
            None | Some("literal") => {
                let scalar = parse_ducklake_default_scalar(value, &data_type).ok_or_else(|| {
                    crate::DuckLakeError::InvalidConfig(format!(
                        "Cannot decode default_value '{value}' for column '{}' as {}",
                        column.column_name, data_type
                    ))
                })?;
                defaults.insert(column.column_name.clone(), Expr::Literal(scalar, None));
            },
            Some("expression") => {
                tracing::debug!(
                    column = %column.column_name,
                    dialect = column.default_value_dialect.as_deref().unwrap_or("unknown"),
                    "Ignoring unevaluated DuckLake expression default"
                );
            },
            Some(value_type) => {
                return Err(crate::DuckLakeError::Unsupported(format!(
                    "Default value type '{value_type}' for column '{}' is not supported",
                    column.column_name
                )));
            },
        }
    }
    Ok(defaults)
}

pub(crate) fn apply_initial_default_metadata(
    schema: &Schema,
    columns: &[DuckLakeTableColumn],
    name_mapping: &HashMap<String, String>,
) -> Schema {
    let fields = schema
        .fields()
        .iter()
        .map(|field| {
            let logical_name = name_mapping
                .get(field.name())
                .map_or(field.name().as_str(), String::as_str);
            let Some(initial_default) = columns
                .iter()
                .find(|column| column.column_name == logical_name)
                .and_then(|column| column.initial_default.as_deref())
                .filter(|value| !value.eq_ignore_ascii_case("NULL"))
            else {
                return Arc::clone(field);
            };
            let mut metadata = field.metadata().clone();
            metadata.insert(
                INITIAL_DEFAULT_METADATA_KEY.to_string(),
                initial_default.to_string(),
            );
            Arc::new(field.as_ref().clone().with_metadata(metadata))
        })
        .collect::<Vec<_>>();
    Schema::new_with_metadata(fields, schema.metadata().clone())
}

/// Whether a stored float `min_value` / `max_value` is a usable bound.
///
/// Catalog min/max exclude NaN, so a float column whose NaN state is unknown
/// (`None`, e.g. register-by-reference loads) or positive (`Some(true)`, e.g.
/// stats written by official DuckLake's INSERT) may hold values outside its
/// recorded bounds, and pruning on them would wrongly drop rows.
///
/// **Both** bounds need the gate here, which is where this deliberately differs
/// from official DuckLake — an engine-layer difference, not a format one.
///
/// Official gates only the max, and that is correct for DuckDB, which
/// normalizes NaN: `-NaN = NaN` is true and either sign sorts above every
/// value, so NaN can only ever hide above a recorded max. DataFusion compares
/// floats with arrow's `total_cmp`, which is sign-sensitive:
/// `(-CAST('NaN' AS DOUBLE)) < CAST('-Infinity' AS DOUBLE)` is `true` here and
/// `false` in DuckDB. A negative NaN therefore sits *below* a recorded min in
/// this engine, and `contains_nan` does not record the sign — so neither bound
/// can be trusted.
///
/// Gating both is what keeps pruning consistent with the filter semantics this
/// engine actually has. The underlying `total_cmp`-vs-DuckDB ordering
/// difference is inherited from arrow and predates this gate; it is visible in
/// query *results* for `±NaN` comparisons, not only in pruning.
fn float_bound_is_usable(data_type: &DataType, contains_nan: Option<bool>) -> bool {
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

/// Whether the catalog *proves* this file holds no rows.
///
/// A file with no rows cannot hold a row matching any predicate, so it is safe to
/// drop from a pruning candidate set without consulting its statistics.
///
/// Only `Some(0)` is proof. `record_count` is optional in the catalog and some
/// providers do not surface it at all, so `None` means "unknown", never "zero":
/// treating it as zero would drop files full of live rows, silently losing them
/// from a scan and — through
/// [`files_matching`](DuckLakeTable::files_matching) — from a keyed mutation's
/// view of the table. Unknown keeps the file, in line with the fail-open contract
/// everywhere else in pruning.
fn file_is_known_empty(file: &DuckLakeTableFile) -> bool {
    file.max_row_count == Some(0)
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
            let bounds_usable = float_bound_is_usable(field_type, raw.contains_nan);
            column_statistics.min_value = if bounds_usable {
                scalar_precision(raw.min_value.as_deref(), column, field_type, exact)
            } else {
                Precision::Absent
            };
            column_statistics.max_value = if bounds_usable {
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
            let bounds_usable = float_bound_is_usable(field_type, raw.contains_nan);
            output.min_value = if bounds_usable {
                scalar_precision(
                    raw.min_value.as_deref(),
                    column,
                    field_type,
                    raw.bounds_are_exact,
                )
            } else {
                Precision::Absent
            };
            output.max_value = if bounds_usable {
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
            // An unusable float min (NaN state unknown/positive) is treated the
            // same way as an unusable max below: negative NaN sorts beneath
            // every value, so the recorded min is not a lower bound.
            let usable_min = raw
                .min_value
                .as_deref()
                .filter(|_| float_bound_is_usable(field_type, raw.contains_nan));
            match usable_min.and_then(|value| parse_statistic_scalar(value, column, field_type)) {
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
            // An unusable float bound is treated as absent: with `all_null` it
            // contributes nothing, otherwise it poisons `max_complete` so the
            // aggregate degrades to unknown.
            let usable_max = raw
                .max_value
                .as_deref()
                .filter(|_| float_bound_is_usable(field_type, raw.contains_nan));
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

/// Cached schema mapping for renamed and path-derived columns.
type SchemaMapping = (
    SchemaRef,
    HashMap<String, String>,
    HashMap<String, ScalarValue>,
);

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
    /// Number of row groups in the file. A scan cannot usefully be split into
    /// more partitions than this — see [`DuckLakeTable::split_across_partitions`].
    row_group_count: usize,
    /// Catalog column indices whose physical name in THIS file differs from the
    /// catalog name — a rename, or the placeholder given to a column the file
    /// predates.
    ///
    /// Indices rather than a flag, and derived positionally rather than from
    /// `name_mapping`, for two reasons. `name_mapping` also carries the
    /// synthetic `_ducklake_internal_row_id -> rowid` entry that every
    /// embedded-rowid file gets, so its emptiness says nothing about data
    /// columns. And a whole-file flag would refuse pushdown for every predicate
    /// on a file that has had any `ADD COLUMN`, however unrelated — see
    /// [`DuckLakeTable::predicate_is_prunable`], which rejects only a predicate
    /// that actually references one of these.
    renamed_column_indices: Arc<HashSet<usize>>,
    /// Logical column name -> value synthesized from this file's Hive path.
    constants: HashMap<String, ScalarValue>,
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
}

/// What one file's parquet footer says: the field-id → physical-name map, the
/// arrow schema the reader derives from it, and the row-group boundaries.
///
/// This is the ONLY place the crate parses a data file's footer for read
/// planning, so field-id resolution cannot drift between the table scan and the
/// change feeds.
pub(crate) struct ParquetFooterFacts {
    /// `field_id → physical column name`, at every nesting depth.
    pub(crate) field_ids: HashMap<i32, String>,
    /// The arrow schema the parquet reader derives for this file.
    pub(crate) arrow_schema: SchemaRef,
    /// Number of row groups in the file. Bounds how far a scan of it can
    /// usefully be split across partitions.
    pub(crate) row_group_count: usize,
}

/// A file's footer resolved against a table's CURRENT columns: the schema to
/// scan it with (each column under the physical name THIS file gives it, or a
/// guaranteed-absent name when the file predates the column), the renames back
/// to the catalog names, and the reserved embedded columns it carries.
#[derive(Debug, Clone)]
pub(crate) struct ParquetFileLayout {
    /// One field per current column, in catalog order, under its physical name.
    /// Carries no embedded columns — each read path appends the ones it wants.
    pub(crate) read_schema: SchemaRef,
    /// `physical name → catalog name`, for the columns whose names differ.
    pub(crate) name_mapping: HashMap<String, String>,
    /// Logical column values synthesized from Hive path segments.
    pub(crate) constants: HashMap<String, ScalarValue>,
    /// `Some(parquet_column_name)` if the file embeds the rowid column (tagged
    /// with [`ROW_ID_PARQUET_FIELD_ID`]).
    pub(crate) embedded_rowid_parquet_name: Option<String>,
    /// `Some(parquet_column_name)` if the file embeds the per-row snapshot-id
    /// column ([`SNAPSHOT_ID_PARQUET_FIELD_ID`]) — i.e. it is a merged partial
    /// file.
    pub(crate) embedded_snapshot_parquet_name: Option<String>,
    /// True if the file carries a data column that is NOT in the table's current
    /// schema — a column dropped since the file was written.
    pub(crate) drops_current_columns: bool,
    /// Number of row groups in the file.
    pub(crate) row_group_count: usize,
}

/// Read `resolved_path`'s parquet footer once. `encryption_key` is the file's
/// DuckLake encryption key when it has one; it is only usable with the
/// `encryption` feature, and this function cannot open an encrypted file
/// without it.
pub(crate) async fn read_parquet_footer_facts(
    state: &dyn Session,
    object_store_url: &ObjectStoreUrl,
    resolved_path: &str,
    encryption_key: Option<&str>,
) -> DataFusionResult<ParquetFooterFacts> {
    let object_store = state.runtime_env().object_store(object_store_url)?;
    let object_path = ObjectPath::from(resolved_path);
    #[allow(deprecated)]
    let reader = ParquetObjectReader::new(object_store, object_path);

    #[cfg(feature = "encryption")]
    let builder = {
        use parquet::arrow::arrow_reader::ArrowReaderOptions;
        let options = match encryption_key {
            Some(key) if !key.is_empty() => {
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
            },
            _ => ArrowReaderOptions::new(),
        };
        ParquetRecordBatchStreamBuilder::new_with_options(reader, options)
            .await
            .map_err(|e| DataFusionError::External(Box::new(e)))?
    };

    #[cfg(not(feature = "encryption"))]
    let builder = {
        // Without the feature there is no decryption to configure.
        let _ = encryption_key;
        ParquetRecordBatchStreamBuilder::new(reader)
            .await
            .map_err(|e| DataFusionError::External(Box::new(e)))?
    };

    let field_ids = extract_parquet_field_ids(builder.metadata());

    Ok(ParquetFooterFacts {
        field_ids,
        arrow_schema: builder.schema().clone(),
        row_group_count: builder.metadata().row_groups().len(),
    })
}

/// Resolve one file's columns against `columns` by field id.
///
/// `fallback_schema` is used verbatim for a file that carries no field ids at
/// all (an external or pre-DuckLake parquet file), where names are the only
/// thing to match on.
pub(crate) async fn read_parquet_file_layout(
    state: &dyn Session,
    object_store_url: &ObjectStoreUrl,
    resolved_path: &str,
    encryption_key: Option<&str>,
    columns: &[DuckLakeTableColumn],
    fallback_schema: &SchemaRef,
) -> DataFusionResult<Arc<ParquetFileLayout>> {
    let facts =
        read_parquet_footer_facts(state, object_store_url, resolved_path, encryption_key).await?;

    let (read_schema, name_mapping) = if facts.field_ids.is_empty() {
        (fallback_schema.clone(), HashMap::new())
    } else {
        let (schema, mapping) = build_read_schema_with_field_id_mapping(
            columns,
            &facts.field_ids,
            Some(facts.arrow_schema.as_ref()),
        )
        .map_err(|e| DataFusionError::External(Box::new(e)))?;
        (Arc::new(schema), mapping)
    };

    // Does the file carry a data column no longer in the current schema? Any
    // parquet field-id that is neither a reserved embedded column nor one of the
    // current catalog `column_id`s is a since-dropped column. Compaction uses
    // this to refuse merging a file whose data would be lost.
    let current_column_ids: HashSet<i32> = columns
        .iter()
        .flat_map(|column| {
            std::iter::once(column.column_id).chain(column.nested_column_ids.iter().copied())
        })
        .map(|column_id| column_id as i32)
        .collect();
    let drops_current_columns = facts.field_ids.keys().any(|fid| {
        *fid != ROW_ID_PARQUET_FIELD_ID
            && *fid != SNAPSHOT_ID_PARQUET_FIELD_ID
            && !current_column_ids.contains(fid)
    });

    Ok(Arc::new(ParquetFileLayout {
        read_schema,
        name_mapping,
        constants: HashMap::new(),
        embedded_rowid_parquet_name: facts.field_ids.get(&ROW_ID_PARQUET_FIELD_ID).cloned(),
        embedded_snapshot_parquet_name: facts.field_ids.get(&SNAPSHOT_ID_PARQUET_FIELD_ID).cloned(),
        drops_current_columns,
        row_group_count: facts.row_group_count,
    }))
}

pub(crate) fn apply_name_mapping_to_layout(
    provider: &dyn MetadataProvider,
    table_id: i64,
    snapshot_id: i64,
    columns: &[DuckLakeTableColumn],
    resolved_path: &str,
    mapping_id: Option<i64>,
    layout: Arc<ParquetFileLayout>,
) -> DataFusionResult<Arc<ParquetFileLayout>> {
    let Some(mapping_id) = mapping_id else {
        return Ok(layout);
    };
    let mapping = provider
        .get_name_mapping(mapping_id)
        .map_err(|e| DataFusionError::External(Box::new(e)))?;
    if mapping.table_id != table_id {
        return Err(DataFusionError::Execution(format!(
            "DuckLake name mapping {mapping_id} belongs to table {}, not table {table_id}",
            mapping.table_id
        )));
    }
    let fields = provider
        .get_table_fields(table_id, snapshot_id)
        .map_err(|e| DataFusionError::External(Box::new(e)))?;
    let mapped = build_read_schema_with_name_mapping(&fields, &mapping)
        .map_err(|e| DataFusionError::External(Box::new(e)))?;
    let constants = mapped
        .partitions
        .into_iter()
        .map(|partition| {
            DuckLakeTable::hive_partition_value(
                resolved_path,
                &partition.source_name,
                &partition.data_type,
            )
            .map(|value| (partition.logical_name, value))
        })
        .collect::<DataFusionResult<HashMap<_, _>>>()?;
    let mut layout = layout.as_ref().clone();
    layout.read_schema = Arc::new(apply_initial_default_metadata(
        &mapped.schema,
        columns,
        &mapped.names,
    ));
    layout.name_mapping = mapped.names;
    layout.constants = constants;
    Ok(Arc::new(layout))
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
    /// Literal defaults used by DataFusion when an INSERT omits a column
    column_defaults: HashMap<String, Expr>,
    /// Complete nested `ducklake_column` tree used by name mappings.
    fields: Vec<DuckLakeTableField>,
    /// Immutable name mappings cached by `mapping_id` across scans and clones.
    name_mapping_cache: Arc<std::sync::Mutex<HashMap<i64, Arc<DuckLakeNameMapping>>>>,
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
    /// Rows materialized by the most recent inlined scan before residual filters.
    inlined_materialized_row_count: Arc<AtomicUsize>,
    /// Decryption keys for the encrypted files this table has listed, and the
    /// factory built from them. Shared with every clone of the table, so a key
    /// one of them installs is available to all of them.
    #[cfg(feature = "encryption")]
    encryption_keys: Arc<InstalledEncryptionKeys>,
    /// Schema name (needed for write operations)
    #[cfg(feature = "write")]
    schema_name: Option<String>,
    /// Metadata writer for write operations (when write feature is enabled)
    #[cfg(feature = "write")]
    writer: Option<Arc<dyn MetadataWriter>>,
    /// Write-layout options (compression, row-group caps, file-rollover target)
    /// applied to the writer built for an INSERT into this table.
    #[cfg(feature = "write")]
    pub(crate) write_options: crate::table_writer::DuckLakeWriteOptions,
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

/// The decryption keys installed on a table, and the factory every reader
/// clones from them ([`DuckLakeTable::create_parquet_source`]).
///
/// Keys only ever accumulate here: [`DuckLakeTable::install_encryption_keys`]
/// merges a listing's keys into the set instead of replacing it. No single
/// listing sees every file — the catalog leaves files unlisted for
/// [`DuckLakeTable::files_matching`] (see [`crate::stats_filter`]) and a scan
/// installs page by page — so replacing would drop the key of a file the caller
/// is still holding from an earlier listing, with no way to get it back short
/// of listing again. The cell is shared with every clone of the table, so that
/// caller need not be the one that listed.
///
/// Accumulating cannot install a wrong key: a path maps to one key for the life
/// of the snapshot, because DuckLake data files are immutable and a rewritten or
/// compacted file is written under a fresh path. A key kept past its file's last
/// read is dead weight, never a wrong answer, and the set is bounded by the
/// encrypted files this table has listed.
///
/// The factory reads the very map that keys are installed into, so the two
/// halves cannot drift apart. Whatever key a path maps to now is the key a
/// reader gets for it, including after a second install of the same path —
/// which nothing here produces, every data and delete file being named by a
/// fresh UUID, but which the factory would otherwise be free to answer from a
/// stale copy.
///
/// Deliberately not `Debug`: the keys are secrets, which is why
/// [`DuckLakeEncryptionFactory`] prints only its file count.
#[cfg(feature = "encryption")]
#[derive(Default)]
struct InstalledEncryptionKeys {
    /// Resolved file path -> decryption key, unioned over every listing. Only
    /// encrypted files get an entry, so an unencrypted table keeps this empty.
    ///
    /// Shared with `factory` instead of copied into it, so installing a
    /// listing's keys costs one insert per path however many the table has
    /// already collected: a scan of P pages installs O(total keys), not
    /// O(keys per page x P).
    keys: Arc<std::sync::RwLock<HashMap<String, String>>>,
    /// The factory a reader clones, reading through `keys`. Unset until an
    /// encrypted file is listed, so an unencrypted table never attaches one;
    /// set once and never replaced, because a factory over the shared map
    /// already resolves the keys of every later listing. A reader holds the
    /// `Arc` it cloned, so nothing here disturbs a read already planned.
    factory: std::sync::OnceLock<Arc<dyn EncryptionFactory>>,
}

/// Record `path`'s decryption key, ignoring the absent-or-empty key an
/// unencrypted file carries.
///
/// These are the entries
/// [`EncryptionFactoryBuilder::add_file`](crate::encryption::EncryptionFactoryBuilder::add_file)
/// would store; they are collected into a plain map so that successive listings
/// can be unioned before a factory is built from the result.
#[cfg(feature = "encryption")]
fn record_encryption_key(keys: &mut HashMap<String, String>, path: &str, key: Option<&str>) {
    if let Some(key) = key.filter(|key| !key.is_empty()) {
        keys.insert(path.to_string(), key.to_string());
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
        let fields = provider.get_table_fields(table_id, snapshot_id)?;
        let columns = provider.get_table_structure(table_id, snapshot_id)?;
        let column_defaults = validate_column_defaults(&columns)?;
        // Active partition spec (if any) for pruning. Loaded once at the bound
        // snapshot; `None` for unpartitioned tables or catalogs without partitions.
        let partition_spec = provider.get_partition_spec(table_id, snapshot_id)?;
        let physical_schema = Arc::new(apply_initial_default_metadata(
            &build_arrow_schema_from_fields(&fields)?,
            &columns,
            &HashMap::new(),
        ));
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

        // File metadata — and with it every encryption key — is deferred to the
        // first listing, so the table opens with no keys installed.
        #[cfg(feature = "encryption")]
        let encryption_keys = Arc::new(InstalledEncryptionKeys::default());

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
            column_defaults,
            fields,
            name_mapping_cache: Arc::new(std::sync::Mutex::new(HashMap::new())),
            table_statistics,
            partition_spec,
            #[cfg(feature = "encryption")]
            encryption_keys,
            file_read_config_cache: Arc::new(std::sync::Mutex::new(HashMap::new())),
            inlined_materialized_row_count: Arc::new(AtomicUsize::new(0)),
            #[cfg(feature = "write")]
            schema_name: None,
            #[cfg(feature = "write")]
            writer: None,
            #[cfg(feature = "write")]
            write_options: crate::table_writer::DuckLakeWriteOptions::default(),
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

    /// Rows materialized by the most recent metadata-catalog inlined scan.
    #[must_use]
    pub fn inlined_materialized_row_count(&self) -> usize {
        self.inlined_materialized_row_count.load(Ordering::Relaxed)
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

    /// The subset of [`Self::files`] whose own catalog statistics permit a match
    /// for `predicate` — every file that could hold a matching row, and none that
    /// is provably empty of them.
    ///
    /// This is the file-level pruning a `SELECT` with the same filter performs,
    /// exposed for callers that drive their own per-file work instead of a scan:
    /// a keyed update, an upsert, or a delete that resolves row positions file by
    /// file with [`Self::resolve_positions`]. It prunes *further* than a scan in
    /// one respect. A scan applies delete files, so it treats a delete-bearing
    /// file's recorded bounds as approximate and keeps the file; these callers do
    /// not apply them, so for them the bounds still hold and the file can be
    /// dropped (see `restate_in_physical_row_space`). Every file a scan would
    /// keep for any other reason is kept here too. Without it such a
    /// caller must open every data file to discover which ones contain a key,
    /// which costs the whole table on every mutation. `predicate` is the same
    /// [`PhysicalExpr`] those callers pass to [`Self::resolve_positions`], so one
    /// expression drives both the file list and the row match; it is resolved
    /// against the table's physical (parquet-backed) columns, without the
    /// synthetic `rowid`.
    ///
    /// Pruning is **fail-open**, and deliberately so: a file is dropped only when
    /// its own statistics *prove* it cannot match. A file with no recorded
    /// statistics, an undecodable partition value, a NULL partition value, or a
    /// bound the catalog only knows approximately is always kept, and any error
    /// building or evaluating the pruning predicate discards the pruning done so
    /// far and returns every file in the page — including files an earlier conjunct
    /// had already excluded, since a failure part-way through leaves no basis for
    /// trusting the exclusions that preceded it. Callers may therefore treat the
    /// result as "these files may match" and must still evaluate `predicate`
    /// against the rows themselves; a mutation that skipped that step could
    /// silently miss a row.
    ///
    /// The one file dropped without consulting statistics is one the catalog
    /// records as holding **no rows** (`record_count` of exactly 0): having no
    /// rows, it cannot hold a matching one. That is a proof rather than an
    /// estimate, so it is the one exclusion the error path above does *not* give
    /// back — such a file is absent from the result whether or not pruning ran, and
    /// whether or not it failed. A file whose row count the catalog leaves unset is
    /// *not* treated as empty; unknown keeps the file, like every other unknown
    /// here.
    ///
    /// How much it prunes depends on how completely the catalog describes the
    /// files. A file that carries no usable bound is kept, while usable bounds on
    /// other files can still prune them.
    ///
    /// Partition values contribute bounds only on their own partition column, and
    /// only when the table has never been re-partitioned (after a re-partition a
    /// live file's values may belong to a retired spec generation whose key order
    /// differs, so they are ignored rather than risk mis-pruning). A predicate
    /// that does not reference a partition column is therefore never pruned by
    /// partition values.
    ///
    /// One limit on fail-open comes from the catalog rather than from here. A
    /// partition value whose recorded key index falls outside the range of a
    /// signed 32-bit integer is read as key index 0 by the SQLite, PostgreSQL
    /// and MySQL providers rather than skipped, so on such a catalog a bound can
    /// be attributed to the wrong partition column and a file that could match
    /// may be dropped. Nothing in this crate writes such a value; it would take a
    /// catalog produced by other means.
    ///
    /// Files are read from the catalog in bounded pages and pruned page by page,
    /// so peak memory tracks the size of the *result*, not the size of the table.
    ///
    /// One addition, on an encrypted table: the decryption key of every file
    /// this *lists* joins the table's installed set, one path/key pair each.
    /// That is a superset of what it returns — each page's keys are collected
    /// before the in-memory pruning runs over that page, so a file dropped
    /// there leaves its key behind, which is what keeps it openable for a
    /// caller holding it from an earlier listing. The set accumulates across
    /// listings rather than being replaced, so a file this call never listed
    /// keeps the key an earlier listing installed for it — see
    /// `InstalledEncryptionKeys` for why that matters and what bounds it.
    ///
    /// # Resolving positions on the returned files
    ///
    /// The returned list may include a file rewritten by an UPDATE or by
    /// compaction, and that is fine: [`Self::resolve_positions`] reads a file's
    /// true physical row positions and does not depend on a row's position
    /// matching `rowid - row_id_start`, so it is valid for rewritten files too.
    /// A rewritten file needs no special handling here.
    pub fn files_matching(
        &self,
        predicate: &Arc<dyn PhysicalExpr>,
    ) -> Result<Vec<DuckLakeTableFile>> {
        let conjuncts = datafusion::physical_expr::split_conjunction(predicate)
            .into_iter()
            .cloned();
        // Fail open: an un-prunable predicate means "keep everything", never an
        // error the caller could mistake for "no files match".
        let pruning = match self.pruning_predicates(conjuncts) {
            Ok(pruning) => pruning,
            Err(error) => {
                tracing::debug!(%error, "skipping predicate-based file pruning");
                Vec::new()
            },
        };
        // The same expression the pruning predicates above were built from,
        // lowered to catalog-statistics SQL so a provider that understands it
        // can stop listing files their statistics rule out — the cost of a
        // selective mutation then tracks the result rather than the table.
        // Fail open here too: `None` means "no SQL filter", and the in-memory
        // pruning still runs over every file that does come back.
        let stats_filter =
            stats_filter::lower_predicate(predicate, &self.physical_schema, &self.columns);

        // Decryption keys for the files these pages carry, which is not every
        // file at the snapshot: the catalog filter above leaves files unlisted,
        // and a file that never reaches this loop cannot have its key collected
        // here. The keys that are collected are therefore MERGED into the
        // table's installed set (`install_encryption_keys`), never installed in
        // place of it, so the key of a file an earlier listing did see stays
        // available to whoever holds that file. Readers clone the factory whole
        // (`create_parquet_source`) out of a cell shared with every clone of
        // this table, so replacing it here would strand exactly those keys — a
        // `resolve_positions` on a file `files` returned, or a scan on a clone.
        //
        // Keeping the retained files' keys still costs one path/key pair per
        // encrypted file returned, on top of whatever the table already holds.
        #[cfg(feature = "encryption")]
        let mut encryption_keys = HashMap::new();

        let mut matching = Vec::new();
        for metadata in self.file_metadata_pages("file matching", stats_filter.as_ref()) {
            let (table_files, mut file_statistics) = self.page_files_with_statistics(metadata?);
            restate_in_physical_row_space(&mut file_statistics, &table_files);
            #[cfg(feature = "encryption")]
            self.collect_encryption_keys(&mut encryption_keys, &table_files)?;
            matching.extend(
                self.prune_table_files_iteratively(&pruning, &table_files, &file_statistics)
                    .into_iter()
                    .cloned(),
            );
        }
        #[cfg(feature = "encryption")]
        self.install_encryption_keys(encryption_keys);
        Ok(matching)
    }

    pub(crate) fn inlined_data_with_row_ids(
        &self,
    ) -> Result<Vec<crate::metadata_provider::DuckLakeInlinedData>> {
        self.provider
            .get_inlined_data_with_row_ids(self.table_id, self.snapshot_id, &self.columns)
    }

    /// Page the catalog's file metadata, optionally handing the provider a
    /// lowered predicate so it can leave provably non-matching files unlisted.
    ///
    /// `filter` is only ever a pre-filter. Every caller still runs the
    /// in-memory [`PruningPredicate`]s over what comes back, because a provider
    /// may ignore the filter entirely (the trait method defaults to doing so)
    /// and because the in-memory pass prunes on things the catalog cannot see —
    /// partition-derived bounds among them.
    fn file_metadata_pages<'a>(
        &'a self,
        page_name: &'static str,
        filter: Option<&'a StatsFilter>,
    ) -> FileMetadataPages<'a> {
        FileMetadataPages {
            provider: self.provider.as_ref(),
            table_id: self.table_id,
            snapshot_id: self.snapshot_id,
            after_data_file_id: None,
            filter,
            page_name,
            finished: false,
        }
    }
    /// Resolve a file path (data or delete file) to its absolute path
    fn resolve_file_path(&self, file: &DuckLakeFileData) -> DataFusionResult<String> {
        resolve_path(&self.table_path, &file.path, file.path_is_relative)
            .map_err(|e| DataFusionError::External(Box::new(e)))
    }

    fn hive_partition_raw_value<'a>(resolved_path: &'a str, source_name: &str) -> Option<&'a str> {
        // Match DuckDB v1.5.5 HivePartitioning::Parse: scan the raw object key,
        // compare raw keys, and leave URL decoding to the value conversion
        // https://github.com/duckdb/duckdb/blob/v1.5.5/src/common/hive_partitioning.cpp
        let mut partition_start = 0;
        let mut equality_sign = 0;
        let mut candidate_partition = true;
        for (index, byte) in resolved_path.bytes().enumerate() {
            if matches!(byte, b'?' | b'\n') {
                candidate_partition = false;
            }
            if matches!(byte, b'/' | b'\\') {
                if candidate_partition && equality_sign > partition_start {
                    let key = &resolved_path[partition_start..equality_sign];
                    if key == source_name {
                        return Some(&resolved_path[equality_sign + 1..index]);
                    }
                }
                partition_start = index + 1;
                candidate_partition = true;
            } else if byte == b'=' {
                if equality_sign > partition_start {
                    candidate_partition = false;
                }
                equality_sign = index;
            }
        }
        None
    }

    fn normalize_hive_hex_integer(value: &str, data_type: &DataType) -> Option<String> {
        if !matches!(
            data_type,
            DataType::Int8
                | DataType::Int16
                | DataType::Int32
                | DataType::Int64
                | DataType::UInt8
                | DataType::UInt16
                | DataType::UInt32
                | DataType::UInt64
        ) {
            return None;
        }
        let digits = value
            .strip_prefix("0x")
            .or_else(|| value.strip_prefix("0X"))?;
        if digits.is_empty()
            || digits
                .split('_')
                .any(|part| part.is_empty() || !part.bytes().all(|byte| byte.is_ascii_hexdigit()))
        {
            return None;
        }
        let digits = digits.replace('_', "");
        u128::from_str_radix(&digits, 16)
            .ok()
            .map(|value| value.to_string())
    }

    fn normalize_hive_exponent_decimal(value: &str, data_type: &DataType) -> Option<String> {
        let (precision, scale) = match data_type {
            DataType::Decimal128(precision, scale) | DataType::Decimal256(precision, scale) => {
                (*precision, *scale)
            },
            _ => return None,
        };
        let exponent_index = value.find(['e', 'E'])?;
        if value[exponent_index + 1..].contains(['e', 'E']) {
            return None;
        }
        let exponent = value[exponent_index + 1..].parse::<i32>().ok()?;
        let mantissa = &value[..exponent_index];
        let (negative, mantissa) = if let Some(mantissa) = mantissa.strip_prefix('-') {
            (true, mantissa)
        } else if let Some(mantissa) = mantissa.strip_prefix('+') {
            (false, mantissa)
        } else {
            (false, mantissa)
        };
        let mut components = mantissa.split('.');
        let whole = components.next()?;
        let fraction = components.next().unwrap_or_default();
        if components.next().is_some()
            || (whole.is_empty() && fraction.is_empty())
            || !whole
                .bytes()
                .chain(fraction.bytes())
                .all(|byte| byte.is_ascii_digit())
        {
            return None;
        }
        let digits = format!("{whole}{fraction}");
        let decimal_index = i64::try_from(whole.len()).ok()? + i64::from(exponent);
        let max_expansion = usize::from(precision)
            .saturating_add(usize::from(scale.unsigned_abs()))
            .saturating_add(digits.len())
            .saturating_add(2);
        let leading_zeros = usize::try_from((-decimal_index).max(0)).ok()?;
        let trailing_zeros =
            usize::try_from((decimal_index - i64::try_from(digits.len()).ok()?).max(0)).ok()?;
        if leading_zeros > max_expansion || trailing_zeros > max_expansion {
            return None;
        }

        let mut normalized = String::with_capacity(digits.len() + max_expansion.min(32));
        if negative {
            normalized.push('-');
        }
        if decimal_index <= 0 {
            normalized.push_str("0.");
            normalized.extend(std::iter::repeat_n('0', leading_zeros));
            normalized.push_str(&digits);
        } else if usize::try_from(decimal_index).ok()? >= digits.len() {
            normalized.push_str(&digits);
            normalized.extend(std::iter::repeat_n('0', trailing_zeros));
        } else {
            let decimal_index = usize::try_from(decimal_index).ok()?;
            normalized.push_str(&digits[..decimal_index]);
            normalized.push('.');
            normalized.push_str(&digits[decimal_index..]);
        }
        Some(normalized)
    }

    pub(crate) fn hive_partition_value(
        resolved_path: &str,
        source_name: &str,
        data_type: &DataType,
    ) -> DataFusionResult<ScalarValue> {
        let raw_value = Self::hive_partition_raw_value(resolved_path, source_name).ok_or_else(|| {
            DataFusionError::Execution(format!(
                "column '{source_name}' uses a DuckLake Hive name mapping but is absent from file path '{resolved_path}'"
            ))
        })?;
        let value = percent_decode_str(raw_value).decode_utf8().map_err(|e| {
            DataFusionError::Execution(format!(
                "invalid percent encoding in Hive partition value '{raw_value}': {e}"
            ))
        })?;
        let value = value.into_owned();
        let is_varchar = matches!(
            data_type,
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
        );
        if value == "__HIVE_DEFAULT_PARTITION__"
            || value.eq_ignore_ascii_case("NULL")
            || (value.is_empty() && !is_varchar)
        {
            return ScalarValue::try_from(data_type);
        }
        let scalar_input = Self::normalize_hive_hex_integer(&value, data_type)
            .or_else(|| Self::normalize_hive_exponent_decimal(&value, data_type))
            .unwrap_or_else(|| value.clone());
        ScalarValue::try_from_string(scalar_input, data_type).map_err(|e| {
            DataFusionError::Execution(format!(
                "failed to parse Hive partition '{source_name}={value}' as {data_type}: {e}"
            ))
        })
    }

    fn mapped_schema(
        &self,
        mapping_id: i64,
        resolved_path: &str,
    ) -> DataFusionResult<SchemaMapping> {
        let mapping = {
            let cached = self.name_mapping_cache.lock().unwrap();
            cached.get(&mapping_id).cloned()
        };
        let mapping = match mapping {
            Some(mapping) => mapping,
            None => {
                let mapping = Arc::new(
                    self.provider
                        .get_name_mapping(mapping_id)
                        .map_err(|e| DataFusionError::External(Box::new(e)))?,
                );
                self.name_mapping_cache
                    .lock()
                    .unwrap()
                    .insert(mapping_id, Arc::clone(&mapping));
                mapping
            },
        };
        if mapping.table_id != self.table_id {
            return Err(DataFusionError::Execution(format!(
                "DuckLake name mapping {mapping_id} belongs to table {}, not table {}",
                mapping.table_id, self.table_id
            )));
        }
        let mapped = build_read_schema_with_name_mapping(&self.fields, mapping.as_ref())
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        let constants = mapped
            .partitions
            .into_iter()
            .map(|partition| {
                Self::hive_partition_value(
                    resolved_path,
                    &partition.source_name,
                    &partition.data_type,
                )
                .map(|value| (partition.logical_name, value))
            })
            .collect::<DataFusionResult<HashMap<_, _>>>()?;
        let schema = apply_initial_default_metadata(&mapped.schema, &self.columns, &mapped.names);
        Ok((Arc::new(schema), mapped.names, constants))
    }

    /// Build a DataFusion file descriptor and attach the catalog's file-level
    /// statistics. `include_rowid` adds an unknown trailing statistic for an
    /// embedded rowid column so the vector still matches the scan schema.
    /// A [`PartitionedFile`] for `file`: its resolved path, validated size, and
    /// the catalog's footer-size hint when it has one.
    fn partitioned_file(&self, file: &DuckLakeFileData) -> DataFusionResult<PartitionedFile> {
        let resolved_path = self.resolve_file_path(file)?;
        let mut pf = PartitionedFile::new(
            &resolved_path,
            validated_file_size(file.file_size_bytes, &resolved_path)?,
        );
        if let Some(footer_size) = file.footer_size
            && footer_size > 0
            && let Ok(hint) = usize::try_from(footer_size)
        {
            pf = pf.with_metadata_size_hint(hint);
        }
        Ok(pf)
    }

    fn partitioned_data_file(
        &self,
        table_file: &DuckLakeTableFile,
        include_rowid: bool,
        file_statistics: &HashMap<i64, Arc<Statistics>>,
    ) -> DataFusionResult<PartitionedFile> {
        let mut file = self.partitioned_file(&table_file.file)?;
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

    /// Split one catalog metadata page into its files and the per-file DataFusion
    /// statistics that drive pruning, with partition-derived bounds folded in.
    ///
    /// Every path that prunes files — planning a scan, and
    /// [`Self::files_matching`] — builds its statistics here, so they all prune
    /// against identical inputs.
    fn page_files_with_statistics(
        &self,
        metadata: Vec<DuckLakeFileMetadata>,
    ) -> (Vec<DuckLakeTableFile>, HashMap<i64, Arc<Statistics>>) {
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
        (table_files, file_statistics)
    }

    /// Inject partition-derived min/max bounds into per-file statistics so the
    /// existing pruning path ([`Self::prune_table_files_iteratively`]) can drop
    /// partition files that cannot match a predicate on a partition column.
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

    /// Return the files whose catalog column statistics prove they may contain
    /// rows matching every predicate.
    ///
    /// This complements the parquet opener's execution-time `FilePruner`, which
    /// uses the same statistics to skip reading non-matching files. Pruning here
    /// shrinks the physical plan itself, including its file count and aggregate
    /// estimates, so downstream planning sees only the relevant files.
    ///
    /// A file with a live delete file has `Inexact` statistics because its
    /// recorded min/max may cover deleted rows, and such a file is kept — unless
    /// the caller has already re-stated those bounds for a reader that does not
    /// apply deletes, which [`DuckLakeTable::files_matching`] does. A file is
    /// only removed when its own exact statistics prove it cannot match, and any
    /// pruning error abandons the whole attempt: it returns `candidates` — every
    /// input file bar the proven-empty ones — and so gives back files an earlier
    /// conjunct had already excluded, rather than trusting exclusions made before
    /// the failure.
    ///
    /// Files the catalog records as holding no rows are dropped up front by
    /// [`file_is_known_empty`], before any statistics are consulted. That
    /// exclusion rests on a proof rather than on statistics, which is why
    /// `candidates` (not `table_files`) is what both the no-predicates and the
    /// error path return: it is the one narrowing no return path here gives back.
    fn prune_table_files_iteratively<'a>(
        &self,
        predicates: &[PruningPredicate],
        table_files: &'a [DuckLakeTableFile],
        file_statistics: &HashMap<i64, Arc<Statistics>>,
    ) -> Vec<&'a DuckLakeTableFile> {
        // The candidate set every path below falls back to: the input minus the
        // files proven empty. Dropping those is not a pruning decision that can be
        // wrong, so it holds even when pruning is skipped entirely.
        let candidates: Vec<&'a DuckLakeTableFile> = table_files
            .iter()
            .filter(|file| !file_is_known_empty(file))
            .collect();
        let mut retained = candidates.clone();
        if predicates.is_empty() {
            return retained;
        }

        loop {
            let count_before = retained.len();
            for predicate in predicates {
                let mask = match self.file_pruning_mask_for(predicate, &retained, file_statistics) {
                    Ok(mask) => mask,
                    Err(e) => {
                        tracing::debug!(error = %e, "skipping plan-time file pruning");
                        return candidates;
                    },
                };
                debug_assert_eq!(mask.len(), retained.len());
                retained = retained
                    .into_iter()
                    .zip(mask)
                    .filter_map(|(file, keep)| keep.then_some(file))
                    .collect();
            }
            if retained.len() == count_before {
                return retained;
            }
        }
    }

    /// Convert scan filters into physical conjuncts against `physical_schema`.
    ///
    /// The single place `scan`'s logical filters become physical expressions.
    /// Both plan-time pruning paths read the result — the catalog-side
    /// statistics filter that narrows the listing query and the in-memory
    /// [`PruningPredicate`]s that run over what it returns — so building them
    /// once is what stops the two disagreeing about what the predicate is.
    fn physical_conjuncts(
        &self,
        state: &dyn Session,
        filters: &[Expr],
    ) -> DataFusionResult<Vec<Arc<dyn PhysicalExpr>>> {
        let df_schema = DFSchema::try_from(self.physical_schema.as_ref().clone())?;
        filters
            .iter()
            .flat_map(datafusion::logical_expr::utils::split_conjunction)
            .map(|expr| state.create_physical_expr(expr.clone(), &df_schema))
            .collect()
    }

    /// Build one [`PruningPredicate`] per conjunct against `physical_schema` (the
    /// parquet-backed columns, excluding the synthetic rowid) — the schema the
    /// per-file statistics are indexed by. The single place pruning predicates
    /// are constructed, whether the conjuncts came from scan filters or from a
    /// caller's own expression.
    fn pruning_predicates(
        &self,
        conjuncts: impl IntoIterator<Item = Arc<dyn PhysicalExpr>>,
    ) -> DataFusionResult<Vec<PruningPredicate>> {
        conjuncts
            .into_iter()
            .map(|conjunct| {
                PruningPredicateBuilder::new()
                    .with_file_schema(Arc::clone(&self.physical_schema))
                    .try_build(conjunct)
            })
            .collect()
    }

    /// Build a `PruningPredicate` from `filters` and evaluate it against every
    /// file's catalog statistics, returning a keep/drop mask 1:1 with
    /// `self.table_files` (`true` = keep). Filters and statistics are both keyed
    /// to `physical_schema` (the parquet-backed columns, excluding the synthetic
    /// rowid), matching how `file_statistics` is indexed.
    fn file_pruning_mask_for(
        &self,
        pruning: &PruningPredicate,
        table_files: &[&DuckLakeTableFile],
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
        let stats = FilePruningStatistics::new(per_file, Arc::clone(&self.physical_schema));
        pruning.prune(&stats)
    }

    /// Create a ParquetSource with encryption support if enabled and needed
    /// Build the parquet source for a scan, with the table's encryption factory
    /// attached when one is installed.
    ///
    /// Accepts either a plain file schema or a [`TableSchema`] — the latter is
    /// how a *positional* scan asks the reader for the physical-position virtual
    /// column (see [`positional_table_schema`]).
    fn create_parquet_source(&self, schema: impl Into<TableSchema>) -> ParquetSource {
        #[cfg(feature = "encryption")]
        if let Some(factory) = self.encryption_keys.factory.get() {
            return ParquetSource::new(schema).with_encryption_factory(Arc::clone(factory));
        }
        ParquetSource::new(schema)
    }

    fn scan_config_builder(&self, source: Arc<dyn FileSource>) -> FileScanConfigBuilder {
        FileScanConfigBuilder::new(self.object_store_url.as_ref().clone(), source)
            .with_expr_adapter(Some(Arc::new(DuckLakeDefaultExprAdapterFactory)))
    }

    /// Add each file's decryption key — and that of its live delete file — to
    /// `keys`, resolving paths the same way the readers do. Separate from
    /// installation so a caller reading the catalog in pages can accumulate keys
    /// across every page and install the whole set once.
    #[cfg(feature = "encryption")]
    fn collect_encryption_keys(
        &self,
        keys: &mut HashMap<String, String>,
        table_files: &[DuckLakeTableFile],
    ) -> Result<()> {
        for table_file in table_files {
            let resolved_path = resolve_path(
                &self.table_path,
                &table_file.file.path,
                table_file.file.path_is_relative,
            )?;
            record_encryption_key(
                keys,
                &resolved_path,
                table_file.file.encryption_key.as_deref(),
            );
            if let Some(delete_file) = &table_file.delete_file {
                let path = resolve_path(
                    &self.table_path,
                    &delete_file.path,
                    delete_file.path_is_relative,
                )?;
                record_encryption_key(keys, &path, delete_file.encryption_key.as_deref());
            }
        }
        Ok(())
    }

    /// Merge one listing's keys into the table's installed set.
    ///
    /// Merging, not replacing: a listing covers only the files it was given —
    /// one page of a scan, or the files the catalog filter left in a
    /// [`Self::files_matching`] page — and a reader may still be holding a file
    /// from an earlier one. See [`InstalledEncryptionKeys`].
    ///
    /// Cost is one insert per path. The factory looks keys up in the same map,
    /// so it is attached once — on the first encrypted file the table lists —
    /// and never rebuilt. A factory owning a copy would instead have every page
    /// of a scan clone every key collected so far, which over P pages of k keys
    /// is k*P*(P+1)/2 entry clones: a 245-page listing of a 1M-file encrypted
    /// table would copy 1.2x10^8 path/key pairs to install 1x10^6.
    ///
    /// A listing that carries no keys does not even take the lock, which is
    /// every page of the unencrypted tables that make up the common case.
    #[cfg(feature = "encryption")]
    fn install_encryption_keys(&self, keys: HashMap<String, String>) {
        if keys.is_empty() {
            return;
        }
        self.encryption_keys.keys.write().unwrap().extend(keys);
        // The first encrypted file this table has listed. Every listing after
        // it reuses this factory, which resolves paths through the map just
        // extended — so the keys it serves and the keys installed are one set,
        // not two that could disagree.
        self.encryption_keys.factory.get_or_init(|| {
            Arc::new(DuckLakeEncryptionFactory::shared(Arc::clone(
                &self.encryption_keys.keys,
            ))) as Arc<dyn EncryptionFactory>
        });
    }

    /// Collect and install the decryption keys of the files in one listing.
    #[cfg(feature = "encryption")]
    fn configure_encryption_factory(&self, table_files: &[DuckLakeTableFile]) -> Result<()> {
        let mut keys = HashMap::new();
        self.collect_encryption_keys(&mut keys, table_files)?;
        self.install_encryption_keys(keys);
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

        #[allow(deprecated)]
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

        if let Some(mapping_id) = file.mapping_id {
            return self.mapped_schema(mapping_id, &resolved_path);
        }

        // No field_ids means external file - use current schema directly
        if field_id_map.is_empty() {
            return Ok((self.schema.clone(), HashMap::new(), HashMap::new()));
        }

        let (read_schema, name_mapping) = build_read_schema_with_field_id_mapping_from_schema(
            &self.columns,
            self.physical_schema.as_ref(),
            &field_id_map,
            Some(builder.schema().as_ref()),
        )
        .map_err(|e| DataFusionError::External(Box::new(e)))?;

        let read_schema =
            apply_initial_default_metadata(&read_schema, &self.columns, &name_mapping);
        Ok((Arc::new(read_schema), name_mapping, HashMap::new()))
    }

    fn present_file_read_config(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        file_cfg: &FileReadConfig,
        physical_len: usize,
    ) -> Arc<dyn ExecutionPlan> {
        let mut output_fields: Vec<Arc<Field>> =
            self.physical_schema.fields().iter().cloned().collect();
        output_fields.extend(plan.schema().fields().iter().skip(physical_len).cloned());
        Arc::new(ColumnRenameExec::new_with_constants(
            plan,
            Arc::new(Schema::new(output_fields)),
            file_cfg.name_mapping.clone(),
            file_cfg.constants.clone(),
        ))
    }

    /// Scan `data_file` and return the physical positions of rows matching
    /// `predicate`, without applying delete files. These are the positions used
    /// by a delete file's `pos` column and
    /// [`crate::metadata_writer::MetadataWriter::set_delete_file`].
    ///
    /// `predicate` is pushed into the reader for row-group/bloom pruning only
    /// when `predicate_is_prunable` allows: never for a float predicate
    /// (footer bounds exclude NaN, and a hidden NaN row is one a DELETE fails to
    /// delete), and never for a file carrying a physical rename. Otherwise the
    /// whole file is scanned.
    ///
    /// Valid for every data file, including one rewritten by an UPDATE or by
    /// compaction. A delete file's `pos` is the row's **physical** index in the
    /// data file it targets, and that is the space this method returns and the
    /// space [`crate::delete_filter::DeleteFilterExec`] filters in — neither
    /// consults `row_id_start` nor a rowid. A rewritten file's rowids are
    /// therefore irrelevant here: they may be non-contiguous, or ordered
    /// differently from the rows they sit on, without affecting which physical
    /// row a resolved position names.
    pub async fn resolve_positions(
        &self,
        state: &dyn Session,
        data_file: &DuckLakeFileData,
        predicate: Arc<dyn datafusion::physical_expr::PhysicalExpr>,
    ) -> DataFusionResult<HashSet<i64>> {
        // Positional scan of the data file: read the physical data columns and the
        // reader-produced physical row position, WITHOUT applying any delete
        // files. Then evaluate `predicate` per batch and collect the physical
        // positions of matching rows — exactly the `pos` values a positional
        // delete file records.
        //
        // `predicate` is expressed against the table's logical column order
        // (column index i = the i-th logical/data field); `Column::evaluate` is
        // index-based, so it resolves against the read batch regardless of any
        // physical rename. `ROW_POS_COLUMN_NAME` is appended last and is never
        // referenced by the predicate.
        //
        // The projection is the physical data columns only. On a file that
        // embeds a rowid column that column sits last in `read_schema`, so
        // `0..physical_len` excludes it and the predicate's column indices still
        // line up with the table's logical order.
        let file_cfg = self.build_file_read_config(state, data_file).await?;

        // Physical data columns only (logical order), then the reader-produced
        // position column. Embedded/rowid columns are not needed to evaluate the
        // predicate or read positions.
        let (table_schema, pos_table_idx, _pos_name) = positional_table_schema_reserving(
            file_cfg.read_schema.clone(),
            self.physical_schema
                .fields()
                .iter()
                .map(|field| field.name().as_str()),
        );
        let mut proj: Vec<usize> = (0..self.physical_schema.fields().len()).collect();
        proj.push(pos_table_idx);
        let pos_idx = proj.len() - 1;

        // Hand the predicate to the reader when it is safe to prune on, so a
        // DELETE/UPDATE reads only the row groups that can contain matches
        // instead of every row of the file. Positions stay true under pruning
        // because the reader derives them from row-group offsets, and a row the
        // reader drops could not have matched anyway.
        let mut source = self.create_parquet_source(table_schema);
        if self.predicate_is_prunable(&predicate, &file_cfg) {
            source = source.with_predicate(Arc::clone(&predicate));
        }

        let plan: Arc<dyn ExecutionPlan> = DataSourceExec::from_data_source(
            self.scan_config_builder(Arc::new(source))
                .with_file_group(FileGroup::new(vec![self.partitioned_file(data_file)?]))
                .with_projection_indices(Some(proj))?
                .build(),
        );
        let plan = self.split_across_partitions(plan, state, &file_cfg)?;

        let plan =
            self.present_file_read_config(plan, &file_cfg, self.physical_schema.fields().len());

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
                    DataFusionError::Internal("physical-position column is not Int64".to_string())
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

    /// Turn a failed delete-file scan into a caller-facing error, replacing the
    /// raw parquet failure with a "delete file is missing" message when the file
    /// really is absent from the object store.
    ///
    /// The absence cannot be read off the error alone: DataFusion's parquet
    /// reader flattens the metadata-fetch failure into a `ParquetError::General`
    /// string, so the underlying `object_store::Error::NotFound` is no longer in
    /// the source chain. Probing the store is only done once a scan has already
    /// failed, so the happy path pays nothing.
    async fn classify_delete_file_read_error(
        &self,
        state: &dyn Session,
        resolved_delete_path: &str,
        err: DataFusionError,
    ) -> DataFusionError {
        let missing = if is_object_store_not_found(&err) {
            true
        } else {
            match state
                .runtime_env()
                .object_store(self.object_store_url.as_ref())
            {
                Ok(store) => matches!(
                    store.head(&ObjectPath::from(resolved_delete_path)).await,
                    Err(object_store::Error::NotFound { .. })
                ),
                Err(_) => false,
            }
        };
        if missing {
            DataFusionError::Execution(format!(
                "Delete file '{resolved_delete_path}' referenced in catalog metadata was not found. This may indicate catalog corruption or that the file was deleted outside of DuckLake."
            ))
        } else {
            err
        }
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

        let collected = stream
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<DataFusionResult<Vec<_>>>();
        let batches: Vec<RecordBatch> = match collected {
            Ok(batches) => batches,
            Err(e) => {
                return Err(self
                    .classify_delete_file_read_error(state, &resolved_delete_path, e)
                    .await);
            },
        };

        // Extract all positions from all batches
        let mut positions = HashSet::new();
        for batch in batches {
            extract_deleted_positions_from_batch(&batch, &mut positions)?;
        }

        Ok(positions)
    }

    pub(crate) fn inlined_deletes_by_file(&self) -> DataFusionResult<HashMap<i64, HashSet<i64>>> {
        let deletes = self
            .provider
            .get_inlined_deletes(self.table_id, self.snapshot_id)
            .map_err(|error| DataFusionError::External(Box::new(error)))?;
        let mut by_file: HashMap<i64, HashSet<i64>> = HashMap::new();
        for delete in deletes {
            if delete.data_file_id < 0 || delete.row_id < 0 {
                return Err(DataFusionError::Execution(format!(
                    "inlined delete has invalid file_id {} and row_id {}",
                    delete.data_file_id, delete.row_id
                )));
            }
            by_file
                .entry(delete.data_file_id)
                .or_default()
                .insert(delete.row_id);
        }
        Ok(by_file)
    }

    async fn deleted_positions_for_file(
        &self,
        state: &dyn Session,
        table_file: &DuckLakeTableFile,
        inlined_positions: Option<&HashSet<i64>>,
    ) -> DataFusionResult<HashSet<i64>> {
        let mut positions = inlined_positions.cloned().unwrap_or_default();
        if let Some(delete_file) = &table_file.delete_file {
            positions.extend(self.read_delete_file_positions(state, delete_file).await?);
        }
        Ok(positions)
    }

    /// Whether `file` was rewritten — by an UPDATE or by compaction — rather
    /// than only ever inserted. A rewritten file embeds its rows' original row
    /// ids as a reserved parquet column (tagged with
    /// [`ROW_ID_PARQUET_FIELD_ID`]); an insert-only file does not.
    ///
    /// This answers where a row's *rowid* comes from — the embedded column when
    /// there is one, `row_id_start + physical position` otherwise — and nothing
    /// more. It is **not** a precondition for [`Self::resolve_positions`] or for a
    /// keyed mutation. A delete file's `pos` is a row's physical index in the data
    /// file, and a rewrite leaves that meaningful: the rewritten file's rows sit
    /// at `0..n-1` exactly as any other file's do. What a rewrite does disturb is
    /// the rowid *sequence* — `rewrite_data_files` drops deleted rows, so the
    /// surviving rowids carry holes and no longer satisfy
    /// `rowid = row_id_start + position` (the catalog records no `row_id_start`
    /// for such a file at all). That is precisely why positions, not rowids, are
    /// what a delete file records.
    ///
    /// Reads the file's parquet footer once and memoizes the answer.
    pub async fn file_has_embedded_rowid(
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
            let (read_schema, name_mapping, constants) = &mapping;
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
            let mut constants: Vec<_> = constants.iter().collect();
            constants.sort_by_key(|(name, _)| *name);
            for (name, value) in constants {
                key.push_str(name);
                key.push('\u{5}');
                key.push_str(&format!("{value:?}"));
                key.push('\u{6}');
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
        for ((read_schema, name_mapping, constants), partitioned_files) in groups {
            let mut builder = self
                .scan_config_builder(Arc::new(self.create_parquet_source(read_schema.clone())))
                .with_limit(limit)
                .with_file_group(FileGroup::new(partitioned_files));

            if let Some(proj) = projection {
                builder = builder.with_projection_indices(Some(proj.clone()))?;
            }

            let parquet_exec: Arc<dyn ExecutionPlan> =
                DataSourceExec::from_data_source(builder.build());

            let mut exec = if !name_mapping.is_empty() || parquet_exec.schema() != output_schema {
                Arc::new(ColumnRenameExec::new_with_constants(
                    parquet_exec,
                    output_schema.clone(),
                    name_mapping,
                    constants,
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
    /// known false (see `float_bound_is_usable`); a file with no statistics entry
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

    /// Split a freshly built scan across `target_partitions` byte ranges.
    ///
    /// The read paths get this from the physical optimizer's
    /// `repartition_file_scans` rule. The paths that build a plan and run it
    /// themselves — `resolve_positions`, the UPDATE source scan, the compaction
    /// rewrite — never reach that rule (they go straight to
    /// `physical_plan::collect` / `execute`), so they ask for the split here.
    ///
    /// Safe because the physical row position comes from the reader, not from
    /// arrival order: every consumer on these paths reads positions out of the
    /// column, so a byte-range split cannot move a row's position. The scan this
    /// replaced did its own row-group-aligned splitting for the same reason, and
    /// dropping it without this would have made these paths single-threaded.
    ///
    /// Capped at the file's row-group count, because DataFusion splits purely by
    /// byte size and has no idea how many row groups a file has: asking for 8
    /// partitions of a 2-row-group file yields 6 that read nothing and 8 that
    /// each fetch the footer, since these scans install no shared metadata
    /// cache. The partitioning this replaced capped the same way.
    ///
    /// Skipped entirely when `optimizer.repartition_file_scans` is off, so these
    /// paths honour the setting the read paths get from the optimizer rule.
    ///
    /// Falls back to `plan` unchanged if the source declines to split.
    fn split_across_partitions(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        state: &dyn Session,
        file_cfg: &FileReadConfig,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let options = state.config_options();
        if !options.optimizer.repartition_file_scans {
            return Ok(plan);
        }
        let target = state
            .config()
            .target_partitions()
            .min(file_cfg.row_group_count.max(1));
        if target <= 1 {
            return Ok(plan);
        }
        Ok(plan.repartitioned(target, options)?.unwrap_or(plan))
    }

    /// Whether `predicate` may be pushed into the parquet reader for a
    /// [`Self::resolve_positions`] scan of a file described by `file_cfg`.
    ///
    /// Deliberately stricter than the read path's `NanPruningBarrierExec`, on
    /// two counts:
    ///
    /// - **Any float reference disqualifies the predicate**, not just one whose
    ///   file is not known NaN-free. Parquet footer float bounds exclude NaN
    ///   while `NaN > C` evaluates true, so pruning on them can hide a matching
    ///   NaN row — and here a hidden row is not a missing result but a row a
    ///   DELETE silently fails to delete. The read path can afford the precise
    ///   test because it has the catalog's per-file statistics; this path is
    ///   handed a bare [`DuckLakeFileData`] and has none. Plumbing them through
    ///   would let float predicates prune here too.
    ///
    /// - **A predicate referencing a column this file names differently is
    ///   disqualified.** The predicate carries catalog column *names*, and the
    ///   reader resolves a pushed predicate by name against the file's schema,
    ///   which uses the physical names. There is no `ColumnRenameExec` on this
    ///   path to rewrite them (`resolve_positions` evaluates positionally
    ///   instead), so such a column would fail to bind — and an unbindable
    ///   column folds the predicate to false, which prunes every row group and
    ///   deletes nothing.
    ///
    ///   Only the columns the predicate actually references matter, hence
    ///   [`FileReadConfig::renamed_column_indices`] rather than a per-file flag:
    ///   a file older than any `ADD COLUMN` names that column with a
    ///   placeholder, and refusing every predicate on such a file would give up
    ///   pruning on the whole history of a table that has ever gained a column.
    ///
    /// Both rejections fall back to scanning the whole file, which is what this
    /// path always did — never to a wrong answer.
    fn predicate_is_prunable(
        &self,
        predicate: &Arc<dyn datafusion::physical_expr::PhysicalExpr>,
        file_cfg: &FileReadConfig,
    ) -> bool {
        use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
        use datafusion::physical_expr::expressions::Column;

        let mut prunable = true;
        predicate
            .apply(|expr| {
                let Some(column) = expr.downcast_ref::<Column>() else {
                    return Ok(TreeNodeRecursion::Continue);
                };
                // An index outside the physical schema cannot be resolved, so
                // treat it as unsafe rather than assuming it is fine.
                let Some(field) = self.physical_schema.fields().get(column.index()) else {
                    prunable = false;
                    return Ok(TreeNodeRecursion::Stop);
                };
                let is_float = matches!(
                    field.data_type(),
                    DataType::Float16 | DataType::Float32 | DataType::Float64
                );
                if is_float || file_cfg.renamed_column_indices.contains(&column.index()) {
                    prunable = false;
                    return Ok(TreeNodeRecursion::Stop);
                }
                Ok(TreeNodeRecursion::Continue)
            })
            .expect("scanning a predicate for column references cannot fail");
        prunable
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

    /// Set the write-layout options applied to this table's INSERT path
    /// (compression, row-group caps, file-rollover target). Propagated from the
    /// catalog's [`with_write_options`](crate::DuckLakeCatalog::with_write_options).
    #[cfg(feature = "write")]
    pub fn with_write_options(
        mut self,
        options: crate::table_writer::DuckLakeWriteOptions,
    ) -> Self {
        self.write_options = options;
        self
    }

    /// Build an execution plan for a single file with delete filtering
    ///
    /// Creates a Parquet scan wrapped with a delete filter to exclude deleted rows.
    async fn build_exec_for_file_with_deletes(
        &self,
        state: &dyn Session,
        table_file: &DuckLakeTableFile,
        inlined_positions: Option<&HashSet<i64>>,
        file_statistics: &HashMap<i64, Arc<Statistics>>,
        projection: Option<&Vec<usize>>,
        limit: Option<usize>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let file_cfg = self.build_file_read_config(state, &table_file.file).await?;

        // Deletes filter by physical row position, so this is a positional path:
        // the scan must also read the reader-produced position column.
        let deleted_positions = self
            .deleted_positions_for_file(state, table_file, inlined_positions)
            .await?;
        let deleted_positions = (!deleted_positions.is_empty()).then_some(deleted_positions);

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
            // Positional path: read the physical position alongside the data so
            // deletes are matched by position. No scan-level limit (it would drop
            // rows before the delete filter); DataFusion enforces LIMIT above.
            let (table_schema, pos_table_idx, _pos_name) =
                positional_table_schema(file_cfg.read_schema.clone());
            let mut proj = proj_indices.clone();
            proj.push(pos_table_idx);
            let pos_index = proj.len() - 1;

            let pf = self.partitioned_data_file(
                table_file,
                file_cfg.embedded_rowid_parquet_name.is_some(),
                file_statistics,
            )?;
            let mut builder = self
                .scan_config_builder(Arc::new(self.create_parquet_source(table_schema)))
                .with_file_group(FileGroup::new(vec![pf]));
            builder = builder.with_projection_indices(Some(proj))?;
            let scan = DataSourceExec::from_data_source(builder.build());

            Arc::new(DeleteFilterExec::try_new(
                scan,
                table_file.file.path.clone(),
                Arc::new(positions),
                pos_index,
            )?)
        } else {
            // No actual deletes for this file: plain scan, scan-level limit OK.
            let pf = self.partitioned_data_file(
                table_file,
                file_cfg.embedded_rowid_parquet_name.is_some(),
                file_statistics,
            )?;
            let mut builder = self
                .scan_config_builder(Arc::new(
                    self.create_parquet_source(file_cfg.read_schema.clone()),
                ))
                .with_limit(limit)
                .with_file_group(FileGroup::new(vec![pf]));
            builder = builder.with_projection_indices(Some(proj_indices.clone()))?;
            DataSourceExec::from_data_source(builder.build())
        };

        // ColumnRenameExec presents the catalog schema and, on the positional
        // path, drops the internal physical-position column.
        let mut exec: Arc<dyn ExecutionPlan> = if !file_cfg.name_mapping.is_empty()
            || !file_cfg.constants.is_empty()
            || exec_after_delete.schema() != output_schema
        {
            Arc::new(ColumnRenameExec::new_with_constants(
                exec_after_delete,
                output_schema,
                file_cfg.name_mapping.clone(),
                file_cfg.constants.clone(),
            ))
        } else {
            exec_after_delete
        };

        // Both branches above can let a predicate reach the parquet reader's
        // pruning, so both need the NaN barrier: footer float bounds exclude
        // NaN, and pruning on them would silently drop NaN rows. The plain-scan
        // branch (a file whose delete file resolves to no positions) has always
        // been pushdown-capable and was missing this.
        let nan_unsafe_columns =
            self.nan_unsafe_float_columns(std::slice::from_ref(&table_file), file_statistics);
        if !nan_unsafe_columns.is_empty() {
            exec = Arc::new(NanPruningBarrierExec::new(exec, nan_unsafe_columns));
        }
        Ok(exec)
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

        #[cfg(feature = "encryption")]
        let encryption_key = file.encryption_key.as_deref();
        #[cfg(not(feature = "encryption"))]
        let encryption_key = None;

        let layout = read_parquet_file_layout(
            state,
            self.object_store_url.as_ref(),
            &resolved_path,
            encryption_key,
            &self.columns,
            &self.physical_schema,
        )
        .await?;

        let (physical_read_schema, mut name_mapping, constants) =
            if let Some(mapping_id) = file.mapping_id {
                self.mapped_schema(mapping_id, &resolved_path)?
            } else {
                (
                    layout.read_schema.clone(),
                    layout.name_mapping.clone(),
                    HashMap::new(),
                )
            };
        let read_schema = if let Some(ref parquet_name) = layout.embedded_rowid_parquet_name {
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
            physical_read_schema.clone()
        };

        // A catalog column reads under a different physical name when the file
        // renamed it or predates it. `physical_read_schema` is one field per
        // current column in catalog order, so the comparison is positional.
        let renamed_column_indices: HashSet<usize> = self
            .physical_schema
            .fields()
            .iter()
            .zip(physical_read_schema.fields())
            .enumerate()
            .filter(|(_, (catalog, physical))| catalog.name() != physical.name())
            .map(|(index, _)| index)
            .collect();

        let cfg = Arc::new(FileReadConfig {
            read_schema,
            row_group_count: layout.row_group_count,
            renamed_column_indices: Arc::new(renamed_column_indices),
            name_mapping,
            constants,
            embedded_rowid_parquet_name: layout.embedded_rowid_parquet_name.clone(),
            embedded_snapshot_parquet_name: layout.embedded_snapshot_parquet_name.clone(),
            drops_current_columns: layout.drops_current_columns,
        });

        {
            let mut cache = self.file_read_config_cache.lock().unwrap();
            cache.entry(resolved_path).or_insert_with(|| cfg.clone());
        }

        Ok(cfg)
    }

    /// Build a plan for a single file when the synthetic `rowid` column is in
    /// the projection. Always uses per-file scans because each file may have a
    /// different layout (embedded rowid vs. synthesized) and a distinct
    /// `row_id_start`.
    ///
    /// Order on the positional path (non-embedded, or any file with deletes):
    ///   DataSourceExec → DeleteFilterExec(?) → RowIdExec(?) → ColumnRenameExec,
    ///   with the position column produced by the scan itself. Embedded-rowid
    ///   files with no deletes keep a plain DataSourceExec → ColumnRenameExec
    ///   (rowid read from the file).
    #[allow(clippy::too_many_arguments, reason = "per-file row-lineage scan inputs stay explicit")]
    async fn build_exec_for_file_with_rowid(
        &self,
        state: &dyn Session,
        table_file: &DuckLakeTableFile,
        inlined_positions: Option<&HashSet<i64>>,
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
        let deleted_positions = self
            .deleted_positions_for_file(state, table_file, inlined_positions)
            .await?;
        let deleted_positions = (!deleted_positions.is_empty()).then_some(deleted_positions);
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
            // Positional path: the reader produces the physical position, which
            // feeds delete filtering and/or rowid synthesis. No scan-level limit
            // (it would drop rows before delete filtering); DataFusion enforces
            // LIMIT above.
            let (table_schema, pos_table_idx, _pos_name) =
                positional_table_schema(file_cfg.read_schema.clone());
            let mut proj = parquet_projection;
            proj.push(pos_table_idx);
            let pos_index = proj.len() - 1;

            let pf = self.partitioned_data_file(table_file, has_embedded, file_statistics)?;
            let mut builder = self
                .scan_config_builder(Arc::new(self.create_parquet_source(table_schema)))
                .with_file_group(FileGroup::new(vec![pf]));
            builder = builder.with_projection_indices(Some(proj))?;
            let mut plan: Arc<dyn ExecutionPlan> =
                DataSourceExec::from_data_source(builder.build());

            // Filter deletes by position, then (for non-embedded files)
            // synthesize rowid from it.
            if let Some(p) = deleted_positions {
                plan = Arc::new(DeleteFilterExec::try_new(
                    plan,
                    table_file.file.path.clone(),
                    Arc::new(p),
                    pos_index,
                )?);
            }
            if !has_embedded {
                plan = Arc::new(RowIdExec::try_new(
                    plan,
                    table_file.row_id_start,
                    pos_index,
                )?);
            }
            plan
        } else {
            // Embedded rowid, no deletes: legacy plain scan (cardinality-
            // preserving). Keep scan-level limit and reader pruning.
            let pf = self.partitioned_data_file(table_file, true, file_statistics)?;
            let mut builder = self
                .scan_config_builder(Arc::new(
                    self.create_parquet_source(file_cfg.read_schema.clone()),
                ))
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
        let mut exec = if !file_cfg.name_mapping.is_empty()
            || !file_cfg.constants.is_empty()
            || after_deletes.schema() != output_schema
        {
            Arc::new(ColumnRenameExec::new_with_constants(
                after_deletes,
                output_schema,
                file_cfg.name_mapping.clone(),
                file_cfg.constants.clone(),
            )) as Arc<dyn ExecutionPlan>
        } else {
            after_deletes
        };
        // Every branch here can now let a predicate reach the parquet reader's
        // row-group/page/bloom pruning: the positional branch no longer blocks
        // pushdown, and `DeleteFilterExec` / `RowIdExec` / `ColumnRenameExec`
        // forward it. So the NaN barrier is unconditional — footer float bounds
        // exclude NaN, and pruning on them would silently drop NaN rows.
        let nan_unsafe_columns =
            self.nan_unsafe_float_columns(std::slice::from_ref(&table_file), file_statistics);
        if !nan_unsafe_columns.is_empty() {
            exec = Arc::new(NanPruningBarrierExec::new(exec, nan_unsafe_columns));
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
        inlined_positions: Option<&HashSet<i64>>,
        output_schema: SchemaRef,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        // This path applies no delete filtering, and that has been safe only
        // because the combination cannot arise: a delete on a merged partial file
        // is authored in a snapshot later than the merge that produced it, which
        // is itself later than `partial_max`, while this path is chosen only when
        // reading BELOW `partial_max` (see `needs_snapshot_filter`) — and the
        // metadata provider attaches a delete file only from its own
        // `begin_snapshot` onward. So no delete file can be live at a snapshot
        // this path serves.
        //
        // Nothing has been observed to violate that; the check exists because the
        // assumption is newly load-bearing. Positional deletes on rewritten and
        // partial files used to be refused outright, so this path could not meet
        // one; now that it can in principle, an invariant that was safely
        // implicit is worth failing loudly on rather than silently returning rows
        // a delete file says are gone.
        if table_file.delete_file.is_some() {
            return Err(DataFusionError::Internal(format!(
                "partial file \"{}\" has a live delete file at read snapshot {}, which the \
                 snapshot-filtered read path cannot apply",
                table_file.file.path, self.snapshot_id
            )));
        }

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

        // Same contract as every other rowid path, and as the C++ extension: a
        // file with neither an embedded rowid nor a catalog `row_id_start` has no
        // reconstructable lineage. Gated on the output actually carrying `rowid`,
        // because this function serves two callers: the rowid-projected read and
        // an ordinary time-travel read, which projects the embedded column away
        // and needs no lineage at all. A merged partial file always carries the
        // embedded column, so this is a guard against a foreign catalog rather
        // than a live case — but without it the rowid caller's failure is a
        // confusing "no field named rowid" from the rename layer instead of a
        // statement of what is actually missing.
        if output_schema.field_with_name(ROWID_COLUMN_NAME).is_ok()
            && file_cfg.embedded_rowid_parquet_name.is_none()
            && table_file.row_id_start.is_none()
        {
            return Err(DataFusionError::Execution(format!(
                "File \"{}\" has no embedded `_ducklake_internal_row_id` column and no \
                 `row_id_start` set in the catalog — row lineage cannot be reconstructed",
                table_file.file.path
            )));
        }

        // Append the embedded snapshot-id column to the file's read schema so the
        // scan materializes it. It is deliberately absent from the cached
        // `read_schema` (that would shift the embedded-rowid column off the end,
        // which other read paths rely on), so we append it here.
        let mut fields: Vec<Arc<Field>> = file_cfg.read_schema.fields().iter().cloned().collect();
        fields.push(Arc::new(Field::new(&snap_name, DataType::Int64, true)));
        let read_schema = Arc::new(Schema::new(fields));
        let projection: Vec<usize> = (0..read_schema.fields().len()).collect();

        let deleted_positions = self
            .deleted_positions_for_file(state, table_file, inlined_positions)
            .await?;
        let scan: Arc<dyn ExecutionPlan> = if deleted_positions.is_empty() {
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
            let builder = self
                .scan_config_builder(Arc::new(self.create_parquet_source(read_schema.clone())))
                .with_file_group(FileGroup::new(vec![pf]))
                .with_projection_indices(Some(projection))?;
            DataSourceExec::from_data_source(builder.build())
        } else {
            // Positional: the reader appends the physical row position, which the
            // delete filter matches the delete set against.
            let (table_schema, pos_table_idx, _pos_name) =
                positional_table_schema(read_schema.clone());
            let mut proj = projection;
            proj.push(pos_table_idx);
            let pos_index = proj.len() - 1;
            let builder = self
                .scan_config_builder(Arc::new(self.create_parquet_source(table_schema)))
                .with_file_group(FileGroup::new(vec![
                    self.partitioned_file(&table_file.file)?,
                ]))
                .with_projection_indices(Some(proj))?;
            let scan = DataSourceExec::from_data_source(builder.build());
            Arc::new(DeleteFilterExec::try_new(
                scan,
                table_file.file.path.clone(),
                Arc::new(deleted_positions),
                pos_index,
            )?)
        };

        // No `NanPruningBarrierExec` here: `SnapshotFilterExec` does not implement
        // `gather_filters_for_pushdown`, so DataFusion's default bars every parent
        // filter and no predicate can reach the reader's float pruning. Give
        // `SnapshotFilterExec` filter pushdown and this path needs the barrier —
        // see `build_exec_for_file_with_deletes` for the shape.
        //
        // Drop rows newer than the read snapshot, then present the catalog schema.
        let filtered: Arc<dyn ExecutionPlan> = Arc::new(SnapshotFilterExec::try_new(
            scan,
            snap_name,
            self.snapshot_id,
        )?);
        Ok(Arc::new(ColumnRenameExec::new_with_constants(
            filtered,
            output_schema,
            file_cfg.name_mapping.clone(),
            file_cfg.constants.clone(),
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
            column_defaults: self.column_defaults.clone(),
            fields: self.fields.clone(),
            name_mapping_cache: Arc::clone(&self.name_mapping_cache),
            table_statistics: self.table_statistics.clone(),
            partition_spec: self.partition_spec.clone(),
            // `snapshot_id`/cache match the post-#163 struct (Arc-wrapped cache,
            // pinned snapshot). A read-only clone starts with an empty cache.
            file_read_config_cache: Arc::new(std::sync::Mutex::new(HashMap::new())),
            inlined_materialized_row_count: Arc::clone(&self.inlined_materialized_row_count),
            #[cfg(feature = "encryption")]
            encryption_keys: Arc::clone(&self.encryption_keys),
            schema_name: None,
            writer: None,
            write_options: self.write_options.clone(),
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

    /// Resolved catalog settings with explicit Rust overrides applied. Used by
    /// maintenance writes so their Parquet layout matches ordinary inserts.
    #[cfg(feature = "write")]
    pub(crate) fn write_options(&self) -> &crate::table_writer::DuckLakeWriteOptions {
        &self.write_options
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

    /// The table's live sort spec at the current catalog head, if any. The write
    /// and compaction paths use it to order rows before writing (tightening
    /// per-file min/max). Read at the head, not the pinned read snapshot, so a
    /// `SET SORTED BY` applied after this provider was opened is honored.
    #[cfg(feature = "write")]
    pub(crate) fn live_sort_spec(&self) -> crate::Result<Option<crate::sort::SortSpec>> {
        let head = self.provider.get_current_snapshot()?;
        self.provider.get_sort_spec(self.table_id, head)
    }

    /// The table's partition spec live at the CURRENT catalog head (not the
    /// snapshot this provider is pinned to) — the generation any write committing
    /// now must agree with. Mirrors [`Self::live_sort_spec`]; the pinned
    /// `self.partition_spec` remains the one used for read pruning, which is
    /// snapshot-bound.
    #[cfg(feature = "write")]
    pub(crate) fn live_partition_spec(&self) -> crate::Result<Option<PartitionSpec>> {
        let head = self.provider.get_current_snapshot()?;
        self.provider.get_partition_spec(self.table_id, head)
    }

    /// The live columns' catalog `column_id`s in `column_order` — the parquet
    /// field-ids a compaction output must bake in so its data columns map back
    /// to the catalog on read.
    #[cfg(feature = "write")]
    pub(crate) fn column_ids(&self) -> Vec<i64> {
        let mut ids = Vec::new();
        for column in &self.columns {
            ids.push(column.column_id);
            ids.extend_from_slice(&column.nested_column_ids);
        }
        ids
    }

    #[cfg(feature = "write")]
    pub(crate) fn top_level_column_ids(&self) -> Vec<i64> {
        self.columns.iter().map(|column| column.column_id).collect()
    }

    /// What `merge_adjacent_files` needs from one candidate's PARQUET FOOTER.
    /// Both facts come from the same footer read (memoized in the per-file
    /// read-config cache), so asking for them together costs one lookup.
    #[cfg(feature = "write")]
    pub(crate) async fn merge_source_facts(
        &self,
        state: &dyn Session,
        file: &DuckLakeFileData,
    ) -> DataFusionResult<MergeSourceFacts> {
        let cfg = self.build_file_read_config(state, file).await?;
        Ok(MergeSourceFacts {
            drops_current_columns: cfg.drops_current_columns,
            has_embedded_snapshot: cfg.embedded_snapshot_parquet_name.is_some(),
        })
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
        inlined_positions: Option<&HashSet<i64>>,
    ) -> DataFusionResult<UpdateSourceScan> {
        self.build_update_scan_with_snapshot(state, table_file, false, inlined_positions)
            .await
    }

    /// [`Self::build_update_scan`], additionally projecting the file's embedded
    /// per-row `_ducklake_internal_snapshot_id` column when `want_snapshot_id`.
    ///
    /// Only a merge re-reading a source that PHYSICALLY carries that column sets
    /// it: such a source's rows each hold their own origin snapshot, so the
    /// merged output must copy the column through rather than stamp one origin
    /// on every row. Errors when the file carries no such column — the caller
    /// decides from the footer, so asking for a column that is not there is a
    /// bug, and inventing origins would silently re-attribute history.
    ///
    /// The column is appended to a LOCAL read schema rather than to the cached
    /// [`FileReadConfig::read_schema`], which ends with the embedded rowid that
    /// the other read paths index from the end.
    #[cfg(feature = "write")]
    pub(crate) async fn build_update_scan_with_snapshot(
        &self,
        state: &dyn Session,
        table_file: &DuckLakeTableFile,
        want_snapshot_id: bool,
        inlined_positions: Option<&HashSet<i64>>,
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

        // Both delete sources mask the scan, but only prior Parquet positions are
        // carried into the replacement Parquet delete file. Inlined positions
        // remain owned by their metadata table until an explicit flush.
        let existing_parquet_deleted = match &table_file.delete_file {
            Some(delete_file) => self.read_delete_file_positions(state, delete_file).await?,
            None => HashSet::new(),
        };
        let mut deleted_positions = existing_parquet_deleted.clone();
        if let Some(inlined) = inlined_positions {
            deleted_positions.extend(inlined);
        }

        // The scan schema is the cached per-file one, plus the embedded
        // snapshot-id column when this caller asked for it. Appending here (and
        // not in `build_file_read_config`) keeps the cached schema's invariant
        // that the embedded rowid, when present, is its LAST field.
        let read_schema = if want_snapshot_id {
            let snap_name = file_cfg
                .embedded_snapshot_parquet_name
                .clone()
                .ok_or_else(|| {
                    DataFusionError::Internal(format!(
                        "File \"{}\" carries no embedded `_ducklake_internal_snapshot_id` \
                         column, so its rows' origin snapshots cannot be read",
                        table_file.file.path
                    ))
                })?;
            let mut fields: Vec<Arc<Field>> =
                file_cfg.read_schema.fields().iter().cloned().collect();
            fields.push(Arc::new(Field::new(&snap_name, DataType::Int64, true)));
            Arc::new(Schema::new(fields))
        } else {
            Arc::clone(&file_cfg.read_schema)
        };

        // Positional scan: the reader produces the physical position. Project the
        // physical columns (logical order), then for an embedded file the
        // embedded rowid column, then the position column.
        let physical_len = self.physical_schema.fields().len();
        let (table_schema, pos_table_idx, _pos_name) = positional_table_schema_reserving(
            read_schema.clone(),
            self.physical_schema
                .fields()
                .iter()
                .map(|field| field.name().as_str()),
        );
        // Built by pushing, so each appended column records the index it actually
        // landed at. Arithmetic off `physical_len` would silently misalign the
        // appended columns for a file that has one but not another, and all are
        // Int64 — nothing downstream would fail to downcast.
        let mut proj: Vec<usize> = (0..physical_len).collect();
        let embedded_batch_idx = has_embedded.then(|| {
            proj.push(file_cfg.read_schema.fields().len() - 1);
            proj.len() - 1
        });
        let embedded_snapshot_batch_idx = want_snapshot_id.then(|| {
            proj.push(read_schema.fields().len() - 1);
            proj.len() - 1
        });
        proj.push(pos_table_idx);
        let pos_index = proj.len() - 1;

        let scan: Arc<dyn ExecutionPlan> = DataSourceExec::from_data_source(
            self.scan_config_builder(Arc::new(self.create_parquet_source(table_schema)))
                .with_file_group(FileGroup::new(vec![
                    self.partitioned_file(&table_file.file)?,
                ]))
                .with_projection_indices(Some(proj))?
                .build(),
        );
        let mut plan = self.split_across_partitions(scan, state, &file_cfg)?;
        // Mask with BOTH delete sources (#262): inlined positions and prior
        // Parquet ones. Only `existing_parquet_deleted` is carried forward into
        // the replacement delete file.
        if !deleted_positions.is_empty() {
            plan = Arc::new(DeleteFilterExec::try_new(
                plan,
                table_file.file.path.clone(),
                Arc::new(deleted_positions),
                pos_index,
            )?);
        }

        plan = self.present_file_read_config(plan, &file_cfg, physical_len);

        Ok(UpdateSourceScan {
            scan: plan,
            physical_len,
            embedded_batch_idx,
            embedded_snapshot_batch_idx,
            pos_index,
            row_id_start: table_file.row_id_start,
            existing_parquet_deleted,
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
    ///
    /// The per-batch work lives in [`rewrite_scanned_batch`], which compaction
    /// applies inside its execution plan rather than after a `collect`, so both
    /// paths produce byte-identical rows.
    #[cfg(feature = "write")]
    pub(crate) fn apply_update_to_batches(
        &self,
        scan: &UpdateSourceScan,
        batches: &[RecordBatch],
        predicate: Option<&Arc<dyn PhysicalExpr>>,
        assignments: &[(usize, Arc<dyn PhysicalExpr>)],
    ) -> DataFusionResult<FileUpdateOutput> {
        let out_schema = rewrite_output_schema(&self.physical_schema);

        let mut updated_batches: Vec<RecordBatch> = Vec::new();
        let mut new_positions: Vec<i64> = Vec::new();

        for batch in batches {
            let Some(rewritten) = rewrite_scanned_batch(
                &self.physical_schema,
                &out_schema,
                scan,
                batch,
                predicate,
                assignments,
            )?
            else {
                continue;
            };
            updated_batches.push(rewritten.batch);
            new_positions.extend(rewritten.positions);
        }

        let matched_count = new_positions.len();
        let mut cumulative = scan.existing_parquet_deleted.clone();
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

/// What one merge candidate's parquet footer says about it, from
/// [`DuckLakeTable::merge_source_facts`].
#[cfg(feature = "write")]
#[derive(Clone, Copy)]
pub(crate) struct MergeSourceFacts {
    /// The file carries a data column that is no longer in the table's current
    /// schema (dropped since it was written). Merged output is written at the
    /// current schema, so merging such a file would lose that column's data —
    /// and remove the file that still holds it.
    pub(crate) drops_current_columns: bool,
    /// The file physically embeds the per-row `_ducklake_internal_snapshot_id`
    /// column, so its rows' origin snapshots are read from it rather than taken
    /// as the file's single catalog `begin_snapshot`. This is the ONLY thing
    /// that decides where a merge source's origins come from — the catalog's
    /// `partial_max` can be NULL on a file that carries the column (providers
    /// that do not read the field, catalogs predating it), and keying off it
    /// would re-stamp every row with one origin and drop the column.
    pub(crate) has_embedded_snapshot: bool,
}

/// Output schema of a rewritten source file: the table's physical columns (in
/// catalog types) followed by the preserved `rowid`. `UPDATE` and compaction
/// share it so both emit exactly the same shape.
#[cfg(feature = "write")]
pub(crate) fn rewrite_output_schema(physical_schema: &SchemaRef) -> SchemaRef {
    let mut fields: Vec<Arc<Field>> = physical_schema.fields().iter().cloned().collect();
    fields.push(Arc::new(rowid_field()));
    Arc::new(Schema::new(fields))
}

/// One batch of [`DuckLakeTable::apply_update_to_batches`] output.
#[cfg(feature = "write")]
pub(crate) struct RewrittenBatch {
    /// `[physical columns (catalog types)..., rowid]` for the matched rows.
    pub(crate) batch: RecordBatch,
    /// Physical positions of the matched rows within the source file, in batch
    /// order — the positions the source file's new cumulative delete must mask.
    pub(crate) positions: Vec<i64>,
    /// The matched rows' embedded per-row origin snapshots, present exactly when
    /// the scan projected them ([`UpdateSourceScan::embedded_snapshot_batch_idx`]).
    /// Filtered with the same mask as the data columns, so each value still
    /// travels with its own row.
    pub(crate) origin_snapshots: Option<ArrayRef>,
}

/// Rewrite ONE batch read from an [`UpdateSourceScan`] into its
/// `[physical columns..., rowid]` form, retaining each row's original rowid.
///
/// `Ok(None)` means the batch contributed nothing (it was empty, or `predicate`
/// matched no row); the caller skips it. `out_schema` must be
/// [`rewrite_output_schema`] of `physical_schema` — it is passed in so a
/// streaming caller builds it once rather than per batch.
///
/// Row-order independent: every value it reads (the physical columns, the
/// embedded rowid, the physical-position column) travels with its own row, so
/// the result is the same however the scan is partitioned or merged.
#[cfg(feature = "write")]
pub(crate) fn rewrite_scanned_batch(
    physical_schema: &SchemaRef,
    out_schema: &SchemaRef,
    scan: &UpdateSourceScan,
    batch: &RecordBatch,
    predicate: Option<&Arc<dyn PhysicalExpr>>,
    assignments: &[(usize, Arc<dyn PhysicalExpr>)],
) -> DataFusionResult<Option<RewrittenBatch>> {
    let physical_len = scan.physical_len;
    let n = batch.num_rows();
    if n == 0 {
        return Ok(None);
    }

    // Coerce physical columns to the catalog types the assignment /
    // predicate exprs (and the writer) expect.
    let mut phys_cols: Vec<ArrayRef> = Vec::with_capacity(physical_len);
    for i in 0..physical_len {
        phys_cols.push(crate::column_rename::coerce_column(
            batch.column(i),
            physical_schema.field(i).data_type(),
        )?);
    }
    let phys_batch = RecordBatch::try_new(Arc::clone(physical_schema), phys_cols.clone())?;

    let row_pos = batch
        .column(scan.pos_index)
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| {
            DataFusionError::Internal("physical-position column is not Int64".to_string())
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
        return Ok(None);
    }

    // Keep only matched rows, then apply the assignments to them.
    let matched_phys: Vec<ArrayRef> = phys_cols
        .iter()
        .enumerate()
        .map(|(i, column)| {
            let filtered = arrow::compute::filter(column.as_ref(), &mask)
                .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?;
            crate::column_rename::coerce_column(&filtered, physical_schema.field(i).data_type())
        })
        .collect::<DataFusionResult<_>>()?;
    let matched_batch = RecordBatch::try_new(Arc::clone(physical_schema), matched_phys.clone())?;
    let matched_rows = matched_batch.num_rows();

    let mut out_cols = matched_phys;
    for (col_idx, expr) in assignments {
        let val = expr.evaluate(&matched_batch)?.into_array(matched_rows)?;
        out_cols[*col_idx] =
            crate::column_rename::coerce_column(&val, physical_schema.field(*col_idx).data_type())?;
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

    // Carried through under the SAME mask as the data columns: taking it from
    // the unfiltered batch would hand each surviving row a different row's
    // origin whenever a predicate selected a subset.
    let origin_snapshots = match scan.embedded_snapshot_batch_idx {
        Some(idx) => Some(
            arrow::compute::filter(batch.column(idx).as_ref(), &mask)
                .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?,
        ),
        None => None,
    };

    Ok(Some(RewrittenBatch {
        batch: RecordBatch::try_new(Arc::clone(out_schema), out_cols)?,
        positions: matched_pos.values().to_vec(),
        origin_snapshots,
    }))
}

/// Per-source-file read plan + metadata for an `UPDATE`, produced by
/// [`DuckLakeTable::build_update_scan`] at plan time and consumed by
/// [`DuckLakeUpdateExec`] at execute time.
#[cfg(feature = "write")]
#[derive(Clone, Debug)]
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
    /// Column index of the embedded per-row `_ducklake_internal_snapshot_id` in
    /// each scanned batch, or `None` when the scan did not project it. Set only
    /// by [`DuckLakeTable::build_update_scan_with_snapshot`], for a merge source
    /// whose per-row origins must survive into the merged output.
    pub(crate) embedded_snapshot_batch_idx: Option<usize>,
    /// Column index of the internal physical-position column in each batch.
    pub(crate) pos_index: usize,
    /// The source file's catalog `row_id_start` (used to synthesize rowids for a
    /// non-embedded file).
    pub(crate) row_id_start: Option<i64>,
    /// Positions already masked by the live Parquet delete file, carried into
    /// the replacement Parquet delete file.
    pub(crate) existing_parquet_deleted: HashSet<i64>,
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

    fn get_column_default(&self, column: &str) -> Option<&Expr> {
        self.column_defaults.get(column)
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
        // Filters drive plan-time file pruning below: iterative conjunct pruning drops
        // files whose catalog statistics prove they cannot match. They are also
        // pushed down to the parquet scanner by DataFusion's optimizer for row
        // group / page-level filtering. We declare them Inexact in
        // `supports_filters_pushdown`, so DataFusion reapplies them after our
        // scan — pruning here only ever removes provably non-matching files.
        filters: &[Expr],
        limit: Option<usize>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let inlined_filter = translate_inlined_filters(filters);

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

        // The rowid plan below unions per-file scans only, so visible inlined
        // rows would silently vanish from a row-lineage read. Refuse loudly
        // instead until inlined rowid scans are supported.
        if rowid_in_proj {
            let inlined = self.provider.scan_inlined_data(
                self.table_id,
                self.snapshot_id,
                &self.columns,
                inlined_filter.as_ref(),
            )?;
            self.inlined_materialized_row_count
                .store(inlined.materialized_row_count, Ordering::Relaxed);
            if inlined.batches.iter().any(|batch| batch.num_rows() > 0) {
                return Err(crate::DuckLakeError::Unsupported(format!(
                    "row-lineage (rowid) scan on a table with inlined rows is not supported; \
                     {INLINED_DATA_REMEDIATION}"
                ))
                .into());
            }
        }

        let mut execs: Vec<Arc<dyn ExecutionPlan>> = Vec::new();
        let inlined_deletes = self.inlined_deletes_by_file()?;
        // One physical form of the filters feeds both pruning paths; see
        // `physical_conjuncts`. A failure to convert them means "prune
        // nothing", never an error — the filters are re-applied above this scan
        // regardless (`supports_filters_pushdown` declares them Inexact).
        let conjuncts = match self.physical_conjuncts(state, filters) {
            Ok(conjuncts) => conjuncts,
            Err(error) => {
                tracing::debug!(%error, "skipping plan-time file pruning");
                Vec::new()
            },
        };
        // Catalog-side pre-filter: the provider may leave files unlisted whose
        // statistics prove they cannot match, which is what stops planning cost
        // scaling with the table. Purely additive — `prune_table_files_iteratively`
        // still runs on every file that comes back, and a predicate that will
        // not lower simply yields no filter.
        let stats_filter = datafusion::physical_expr::conjunction_opt(conjuncts.iter().cloned())
            .and_then(|predicate| {
                stats_filter::lower_predicate(&predicate, &self.physical_schema, &self.columns)
            });
        let pruning = match self.pruning_predicates(conjuncts) {
            Ok(pruning) => pruning,
            Err(error) => {
                tracing::debug!(%error, "skipping plan-time file pruning");
                Vec::new()
            },
        };
        for metadata in self.file_metadata_pages("planning", stats_filter.as_ref()) {
            let (table_files, file_statistics) = self.page_files_with_statistics(metadata?);
            #[cfg(feature = "encryption")]
            self.configure_encryption_factory(&table_files)?;

            let table_files =
                self.prune_table_files_iteratively(&pruning, &table_files, &file_statistics);

            if rowid_in_proj {
                let rowid_idx = rowid_idx.unwrap();
                let user_proj: Vec<usize> = projection
                    .cloned()
                    .unwrap_or_else(|| (0..self.schema.fields().len()).collect());
                for table_file in table_files {
                    let exec = if self.needs_snapshot_filter(table_file) {
                        let output_schema =
                            self.output_schema_for_projection(&user_proj, rowid_idx);
                        self.build_exec_for_partial_file(
                            state,
                            table_file,
                            inlined_deletes.get(&table_file.data_file_id),
                            output_schema,
                        )
                        .await?
                    } else {
                        self.build_exec_for_file_with_rowid(
                            state,
                            table_file,
                            inlined_deletes.get(&table_file.data_file_id),
                            &file_statistics,
                            &user_proj,
                            rowid_idx,
                            limit,
                        )
                        .await?
                    };
                    execs.push(exec);
                }
                continue;
            }

            let (needs_filter, rest): (Vec<_>, Vec<_>) = table_files
                .into_iter()
                .partition(|table_file| self.needs_snapshot_filter(table_file));
            let (files_with_deletes, files_without_deletes): (Vec<_>, Vec<_>) =
                rest.into_iter().partition(|table_file| {
                    table_file.delete_file.is_some()
                        || inlined_deletes.contains_key(&table_file.data_file_id)
                });

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
                        inlined_deletes.get(&table_file.data_file_id),
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
                    self.build_exec_for_partial_file(
                        state,
                        table_file,
                        inlined_deletes.get(&table_file.data_file_id),
                        output_schema,
                    )
                    .await?,
                );
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
        let inlined = self.provider.scan_inlined_data(
            self.table_id,
            self.snapshot_id,
            &self.columns,
            inlined_filter.as_ref(),
        )?;
        self.inlined_materialized_row_count
            .store(inlined.materialized_row_count, Ordering::Relaxed);
        if inlined.batches.iter().any(|b| b.num_rows() > 0) {
            let exec = MemorySourceConfig::try_new_exec(
                &[inlined.batches],
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
        self.write_options
            .validate()
            .map_err(|e| DataFusionError::External(Box::new(e)))?;

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
                let column_ids: Vec<i64> = self.columns.iter().map(|c| c.column_id).collect();
                Some(
                    crate::partition::PartitionWriteSpec::resolve(
                        spec,
                        &column_ids,
                        self.physical_schema.as_ref(),
                    )
                    .map_err(|e| DataFusionError::External(Box::new(e)))?,
                )
            },
        };

        // Resolve the live sort spec (also at the head, for the same reason as the
        // partition spec). When it is producible — every key is a bare column
        // present in the write schema — wrap the input in a global SortExec so each
        // written file's rows are ordered, tightening per-file min/max statistics
        // for range pruning. Sorting is applied before the partition split, so each
        // per-partition file remains a sorted subsequence. An unsupported expression
        // or missing sort column is rejected before execution rather than silently
        // producing files that violate the active sort contract.
        let ordering = if self.write_options.sort_on_insert.unwrap_or(true) {
            let live_sort = self
                .provider
                .get_sort_spec(self.table_id, head_snapshot)
                .map_err(|e| DataFusionError::External(Box::new(e)))?;
            sort_ordering_for(&input.schema(), live_sort.as_ref())?
        } else {
            None
        };
        // Wrap the input in a SortExec now AND declare the same requirement on
        // DuckLakeInsertExec, so DataFusion's EnforceSorting keeps the ordering
        // (a plain SortExec with no downstream ordering requirement would be
        // optimized away, silently dropping the sort).
        let input = match ordering.clone() {
            Some(ordering) => Arc::new(datafusion::physical_plan::sorts::sort::SortExec::new(
                ordering, input,
            )) as Arc<dyn ExecutionPlan>,
            None => input,
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
            self.write_options.clone(),
            ordering,
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
        self.write_options
            .validate()
            .map_err(|e| DataFusionError::External(Box::new(e)))?;

        // DuckDB / MySQL metadata writers do not implement the atomic
        // append-with-deletes commit UPDATE needs. Reject up front rather than
        // rewriting files and only failing at commit.
        if !writer.supports_update() {
            return Err(DataFusionError::NotImplemented(
                "UPDATE not supported on this metadata backend".to_string(),
            ));
        }

        // UPDATE rewrites Parquet-resident rows only; visible inlined rows would
        // be silently skipped. Refuse loudly instead (same detection DELETE uses).
        let inlined =
            self.provider
                .get_inlined_data(self.table_id, self.snapshot_id, &self.columns)?;
        if inlined.iter().any(|batch| batch.num_rows() > 0) {
            return Err(crate::DuckLakeError::Unsupported(format!(
                "UPDATE on a table with inlined rows is not supported; \
                 {INLINED_DATA_REMEDIATION}"
            ))
            .into());
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
        let inlined_deletes = self.inlined_deletes_by_file()?;
        let mut scans = Vec::with_capacity(table_files.len());
        for tf in &table_files {
            scans.push(
                self.build_update_scan(state, tf, inlined_deletes.get(&tf.data_file_id))
                    .await?,
            );
        }

        Ok(Arc::new(DuckLakeUpdateExec::new(
            Arc::new(self.read_only_clone()),
            Arc::clone(writer),
            schema_name.clone(),
            self.table_name.clone(),
            self.snapshot_id,
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

        self.write_options
            .validate()
            .map_err(|e| DataFusionError::External(Box::new(e)))?;

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

/// The `LexOrdering` for a table's live sort spec against `schema`, or `None` when
/// the table has no sort order. Unsupported expressions and missing sort columns
/// fail the write before data is committed.
#[cfg(feature = "write")]
fn sort_ordering_for(
    schema: &arrow::datatypes::Schema,
    sort_spec: Option<&crate::sort::SortSpec>,
) -> datafusion::common::Result<Option<datafusion::physical_expr::LexOrdering>> {
    let Some(sort_spec) = sort_spec else {
        return Ok(None);
    };
    let keys = sort_spec.producible_columns().ok_or_else(|| {
        DataFusionError::NotImplemented(format!(
            "DuckLake sort order {} contains an unsupported expression; \
             datafusion-ducklake can write only bare-column sort keys",
            sort_spec.sort_id
        ))
    })?;
    if keys.is_empty() {
        return Ok(None);
    }
    let mut sort_exprs = Vec::with_capacity(keys.len());
    for (name, direction, null_order) in &keys {
        let index = schema.index_of(name).map_err(|_| {
            DataFusionError::Plan(format!(
                "DuckLake sort key '{name}' is not present in the write schema"
            ))
        })?;
        let column = Arc::new(datafusion::physical_expr::expressions::Column::new(
            name, index,
        ));
        sort_exprs.push(
            datafusion::physical_expr_common::sort_expr::PhysicalSortExpr::new(
                column,
                arrow::compute::SortOptions {
                    descending: matches!(direction, crate::sort::SortDirection::Desc),
                    nulls_first: null_order.nulls_first(),
                },
            ),
        );
    }
    Ok(datafusion::physical_expr::LexOrdering::new(sort_exprs))
}

/// Re-state a page's per-file statistics for a caller that reads a data file
/// **without applying its delete file**.
///
/// [`build_datafusion_statistics`] marks a delete-bearing file's recorded bounds
/// [`Precision::Inexact`], and [`FilePruningStatistics`] prunes only on `Exact`
/// ones. So the first mutation that writes a delete file also stops that file
/// contributing usable bounds, and every later call opens it whatever its key
/// range. Nothing restores them: compaction skips delete-bearing files.
///
/// Those bounds stay usable for this caller. [`DuckLakeTable::resolve_positions`]
/// scans physical rows and does not apply delete files — it exists to compute
/// the positions a delete file records — so a recorded bound still contains
/// every row it can see, and `null_count`, harvested from the parquet footer,
/// still counts them.
///
/// `Exact` here is DataFusion's word for "usable", not a claim of a true
/// extreme. The DuckLake spec requires `min_value`/`max_value` only to be a
/// lower and upper bound ("does not have to be exact"), and this promotes them
/// no further than that: a bound over the physical rows is still a bound over
/// any subset of them, which is what makes it valid for a delete-applying
/// reader too.
///
/// This promotes every `Inexact` bound on such a file, including the widened
/// envelopes [`DuckLakeTable::apply_partition_bounds`] synthesises, which it
/// leaves `Inexact` so they cannot be mistaken for real extrema. That is
/// deliberate and safe for pruning — an envelope is a true bound — and safe
/// only because the map this returns lives and dies inside
/// [`DuckLakeTable::files_matching`]. It must not reach scan or aggregate
/// statistics, where an envelope presented as an extreme could answer a
/// MIN/MAX-from-statistics query wrongly.
///
/// The live row count is deliberately left out. [`file_row_count`] subtracts
/// `delete_count`, making it the one figure deletes genuinely change. Promoting
/// it alongside a physical `null_count` would be unsound rather than imprecise:
/// DataFusion guards a comparison with `null_count != row_count`, so a file with
/// two physical rows — one NULL, one matching, the NULL deleted — would present
/// `null_count == row_count`, be judged entirely null, and be dropped while
/// still physically holding the row the caller must find. Withholding the count
/// leaves that rewrite inert.
fn restate_in_physical_row_space(
    statistics: &mut HashMap<i64, Arc<Statistics>>,
    files: &[DuckLakeTableFile],
) {
    fn physical<T: Clone + std::fmt::Debug + Eq + PartialOrd>(
        value: &Precision<T>,
    ) -> Precision<T> {
        match value {
            Precision::Inexact(value) => Precision::Exact(value.clone()),
            other => other.clone(),
        }
    }

    for file in files.iter().filter(|f| f.delete_file.is_some()) {
        let Some(current) = statistics.get(&file.data_file_id) else {
            continue;
        };
        let mut restated = current.as_ref().clone();
        // Not physical: this is live rows, deletes already subtracted.
        restated.num_rows = Precision::Absent;
        for column in &mut restated.column_statistics {
            column.min_value = physical(&column.min_value);
            column.max_value = physical(&column.max_value);
            column.null_count = physical(&column.null_count);
        }
        statistics.insert(file.data_file_id, Arc::new(restated));
    }
}

struct FilePruningStatistics {
    base: PrunableStatistics,
    statistics: Vec<Arc<Statistics>>,
    schema: SchemaRef,
}

impl FilePruningStatistics {
    fn new(statistics: Vec<Arc<Statistics>>, schema: SchemaRef) -> Self {
        let base = PrunableStatistics::new(statistics.clone(), Arc::clone(&schema));
        Self {
            base,
            statistics,
            schema,
        }
    }

    fn exact_values(
        &self,
        column: &Column,
        get_statistic: impl Fn(&ColumnStatistics) -> &Precision<ScalarValue>,
    ) -> Option<ArrayRef> {
        let index = self.schema.index_of(column.name()).ok()?;
        // DataFusion 54 substitutes an untyped null for an unavailable bound,
        // which cannot share an Arrow array with typed scalar bounds.
        let typed_null = ScalarValue::try_new_null(self.schema.field(index).data_type()).ok()?;
        let mut has_value = false;
        let values = self.statistics.iter().map(|statistics| {
            statistics
                .column_statistics
                .get(index)
                .and_then(|statistics| match get_statistic(statistics) {
                    Precision::Exact(value) => {
                        has_value = true;
                        Some(value.clone())
                    },
                    _ => None,
                })
                .unwrap_or_else(|| typed_null.clone())
        });
        ScalarValue::iter_to_array(values)
            .ok()
            .filter(|_| has_value)
    }
}

impl PruningStatistics for FilePruningStatistics {
    fn min_values(&self, column: &Column) -> Option<ArrayRef> {
        self.exact_values(column, |statistics| &statistics.min_value)
    }

    fn max_values(&self, column: &Column) -> Option<ArrayRef> {
        self.exact_values(column, |statistics| &statistics.max_value)
    }

    fn num_containers(&self) -> usize {
        self.base.num_containers()
    }

    fn null_counts(&self, column: &Column) -> Option<ArrayRef> {
        self.base.null_counts(column)
    }

    fn row_counts(&self) -> Option<ArrayRef> {
        self.base.row_counts()
    }

    fn contained(&self, column: &Column, values: &HashSet<ScalarValue>) -> Option<BooleanArray> {
        self.base.contained(column, values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata_provider::{
        ColumnWithTable, DataFileChange, DeleteFileChange, FileWithTable, SchemaMetadata,
        SnapshotMetadata, TableMetadata, TableWithSchema,
    };
    use crate::partition::{PartitionSpecColumn, PartitionTransform};
    use crate::types::build_arrow_schema;
    use datafusion::prelude::{SessionContext, col, lit};
    use rstest::rstest;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[rstest]
    #[case("NULL", DataType::Utf8, ScalarValue::Utf8(None))]
    #[case("null", DataType::Int32, ScalarValue::Int32(None))]
    #[case("", DataType::Int32, ScalarValue::Int32(None))]
    #[case("", DataType::Utf8, ScalarValue::Utf8(Some(String::new())))]
    #[case("__HIVE_DEFAULT_PARTITION__", DataType::Int32, ScalarValue::Int32(None))]
    #[case("0x10", DataType::Int32, ScalarValue::Int32(Some(16)))]
    #[case("1e2", DataType::Decimal128(10, 2), ScalarValue::Decimal128(Some(10_000), 10, 2))]
    #[case("1.25e-1", DataType::Decimal128(10, 3), ScalarValue::Decimal128(Some(125), 10, 3))]
    fn hive_partition_values_match_duckdb(
        #[case] encoded: &str,
        #[case] data_type: DataType,
        #[case] expected: ScalarValue,
    ) {
        let path = format!("file:///tmp/p={encoded}/data.parquet");

        let actual = DuckLakeTable::hive_partition_value(&path, "p", &data_type).unwrap();

        assert_eq!(actual, expected);
    }

    #[rstest]
    #[case(
        "s3://bucket/a=one/../p=7/file.parquet",
        "a",
        DataType::Utf8,
        ScalarValue::Utf8(Some("one".to_string()))
    )]
    #[case(
        "s3://bucket/p=a#b/file.parquet",
        "p",
        DataType::Utf8,
        ScalarValue::Utf8(Some("a#b".to_string()))
    )]
    #[case(
        "s3://bucket/not=a?b/p=7/file.parquet",
        "p",
        DataType::Int32,
        ScalarValue::Int32(Some(7))
    )]
    #[case(r"s3:\bucket\p=7\file.parquet", "p", DataType::Int32, ScalarValue::Int32(Some(7)))]
    #[case(
        "s3://bucket/p%20q=7/file.parquet",
        "p%20q",
        DataType::Int32,
        ScalarValue::Int32(Some(7))
    )]
    #[case(
        "s3://bucket/p=a%20b/file.parquet",
        "p",
        DataType::Utf8,
        ScalarValue::Utf8(Some("a b".to_string()))
    )]
    fn hive_partition_path_parsing_matches_duckdb(
        #[case] path: &str,
        #[case] source_name: &str,
        #[case] data_type: DataType,
        #[case] expected: ScalarValue,
    ) {
        let actual = DuckLakeTable::hive_partition_value(path, source_name, &data_type).unwrap();

        assert_eq!(actual, expected);
    }

    #[rstest]
    #[case("s3://bucket/p=7=8/file.parquet", "p")]
    #[case("s3://bucket/p%20q=7/file.parquet", "p q")]
    fn hive_partition_path_rejects_non_candidates(#[case] path: &str, #[case] source_name: &str) {
        let error =
            DuckLakeTable::hive_partition_value(path, source_name, &DataType::Int32).unwrap_err();

        assert!(error.to_string().contains("absent from file path"));
    }

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
            table_id: i64,
            _snapshot_id: i64,
        ) -> Result<Vec<DuckLakeTableColumn>> {
            if table_id == 2 {
                return Ok(vec![
                    DuckLakeTableColumn::new(
                        1,
                        "range_value".to_string(),
                        "bigint".to_string(),
                        false,
                    ),
                    DuckLakeTableColumn::new(
                        2,
                        "partition_key".to_string(),
                        "bigint".to_string(),
                        false,
                    ),
                ]);
            }
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
            table_id: i64,
            _snapshot_id: i64,
        ) -> Result<DuckLakeStatistics> {
            if table_id == 2 {
                return Ok(DuckLakeStatistics::default());
            }
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
        let conjuncts = table.physical_conjuncts(&state, &filters).unwrap();
        let pruning = table.pruning_predicates(conjuncts).unwrap();
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
                    .prune_table_files_iteratively(&pruning, &files, &file_statistics)
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
    fn iterative_pruning_revisits_range_conjunct_after_partition_conjunct() -> Result<()> {
        let table = DuckLakeTable::new(
            2,
            "events",
            Arc::new(LazyMillionFileProvider::default()),
            1,
            Arc::new(ObjectStoreUrl::parse("memory://").unwrap()),
            String::new(),
        )?;
        let state = SessionContext::new().state();
        let filters = [col("range_value").eq(lit(1_i64)), col("partition_key").eq(lit(1_i64))];
        let predicates = table.pruning_predicates(table.physical_conjuncts(&state, &filters)?)?;
        let file = |data_file_id| DuckLakeTableFile {
            data_file_id,
            file: DuckLakeFileData::new(format!("file-{data_file_id}.parquet"), true, 1),
            delete_file_id: None,
            delete_file: None,
            row_id_start: None,
            snapshot_id: Some(1),
            begin_snapshot: Some(1),
            schema_version: Some(0),
            partial_max: None,
            max_row_count: Some(1),
            delete_count: None,
            partition_id: None,
            partition_values: Vec::new(),
        };
        let files = vec![file(1), file(2), file(3)];
        let statistics = |range_value: Option<i64>, partition_key: i64| {
            let mut statistics = Statistics::new_unknown(table.physical_schema.as_ref());
            if let Some(value) = range_value {
                statistics.column_statistics[0].min_value =
                    Precision::Exact(ScalarValue::Int64(Some(value)));
                statistics.column_statistics[0].max_value =
                    Precision::Exact(ScalarValue::Int64(Some(value)));
            }
            statistics.column_statistics[1].min_value =
                Precision::Exact(ScalarValue::Int64(Some(partition_key)));
            statistics.column_statistics[1].max_value =
                Precision::Exact(ScalarValue::Int64(Some(partition_key)));
            Arc::new(statistics)
        };
        let file_statistics = HashMap::from([
            (1, statistics(Some(1), 1)),
            (2, statistics(Some(2), 1)),
            (3, statistics(None, 2)),
        ]);

        let retained = table
            .prune_table_files_iteratively(&predicates, &files, &file_statistics)
            .into_iter()
            .map(|file| file.data_file_id)
            .collect::<Vec<_>>();

        assert_eq!(retained, vec![1]);
        Ok(())
    }

    // ==================== `files_matching` ====================
    //
    // Fixture shape: a two-column table `(id BIGINT, region BIGINT)` optionally
    // partitioned by `region` with the `identity` transform. `id` stands in for a
    // mutation's key column and `region` for the partition column, so a predicate
    // can name one without the other.

    const ID_COLUMN_ID: i64 = 1;
    const REGION_COLUMN_ID: i64 = 2;

    /// A catalog of exactly the files a test hands it, served through the paged
    /// metadata API the pruning paths read.
    #[derive(Debug)]
    struct FixedFileProvider {
        files: Vec<DuckLakeFileMetadata>,
        partition_spec: Option<PartitionSpec>,
        /// Which files a page carries once a caller pushes a filter down, by
        /// `data_file_id`. `None` ignores the filter and lists everything, which
        /// is what the trait's default method does and what every test that does
        /// not name this field exercises.
        ///
        /// A real catalog decides this by evaluating the lowered filter against
        /// `ducklake_file_column_stats`; naming the surviving files outright says
        /// the same thing without a second implementation of those semantics.
        /// What matters here is that the other files are never listed at all, so
        /// nothing downstream — including key collection — can see them.
        visible_when_filtered: Option<Vec<i64>>,
    }

    impl MetadataProvider for FixedFileProvider {
        fn get_current_snapshot(&self) -> Result<i64> {
            Ok(1)
        }

        fn get_data_path(&self) -> Result<String> {
            Ok("memory:///".to_string())
        }

        fn get_table_structure(
            &self,
            _table_id: i64,
            _snapshot_id: i64,
        ) -> Result<Vec<DuckLakeTableColumn>> {
            Ok(vec![
                DuckLakeTableColumn::new(
                    ID_COLUMN_ID,
                    "id".to_string(),
                    "bigint".to_string(),
                    false,
                ),
                DuckLakeTableColumn::new(
                    REGION_COLUMN_ID,
                    "region".to_string(),
                    "bigint".to_string(),
                    false,
                ),
            ])
        }

        fn get_partition_spec(
            &self,
            _table_id: i64,
            _snapshot_id: i64,
        ) -> Result<Option<PartitionSpec>> {
            Ok(self.partition_spec.clone())
        }

        fn get_table_files_for_select(
            &self,
            _table_id: i64,
            _snapshot_id: i64,
        ) -> Result<Vec<DuckLakeTableFile>> {
            Ok(self.files.iter().map(|entry| entry.file.clone()).collect())
        }

        fn get_table_file_metadata_page(
            &self,
            _table_id: i64,
            _snapshot_id: i64,
            after_data_file_id: Option<i64>,
            limit: usize,
        ) -> Result<Vec<DuckLakeFileMetadata>> {
            Ok(self
                .files
                .iter()
                .filter(|entry| {
                    after_data_file_id.is_none_or(|after| entry.file.data_file_id > after)
                })
                .take(limit)
                .cloned()
                .collect())
        }

        fn get_table_file_metadata_page_filtered(
            &self,
            table_id: i64,
            snapshot_id: i64,
            after_data_file_id: Option<i64>,
            limit: usize,
            filter: Option<&crate::stats_filter::StatsFilter>,
        ) -> Result<Vec<DuckLakeFileMetadata>> {
            let Some(visible) = self
                .visible_when_filtered
                .as_ref()
                .filter(|_| filter.is_some())
            else {
                return self.get_table_file_metadata_page(
                    table_id,
                    snapshot_id,
                    after_data_file_id,
                    limit,
                );
            };
            // Narrowed inside the "query", ahead of the "limit", exactly as the
            // real providers must: the keyset cursor still advances over the
            // surviving files only.
            Ok(self
                .files
                .iter()
                .filter(|entry| visible.contains(&entry.file.data_file_id))
                .filter(|entry| {
                    after_data_file_id.is_none_or(|after| entry.file.data_file_id > after)
                })
                .take(limit)
                .cloned()
                .collect())
        }

        fn get_table_summary_statistics(
            &self,
            _table_id: i64,
            _snapshot_id: i64,
        ) -> Result<DuckLakeStatistics> {
            Ok(DuckLakeStatistics::default())
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

    /// The partition value a file records for `region`: `None` = the file records
    /// no partition value at all, `Some(None)` = it records SQL NULL.
    type RegionValue = Option<Option<&'static str>>;

    /// One data file. `id_bounds` become exact catalog min/max statistics for
    /// `id`; `None` leaves the file without any statistics, the shape a file
    /// written before statistics collection has. `region` never carries column
    /// statistics, so anything that prunes it must come from the partition value.
    fn fixture_file(
        data_file_id: i64,
        region: RegionValue,
        id_bounds: Option<(i64, i64)>,
    ) -> DuckLakeFileMetadata {
        let mut file = DuckLakeTableFile::new(DuckLakeFileData::new(
            format!("file-{data_file_id}.parquet"),
            true,
            1,
        ));
        file.data_file_id = data_file_id;
        file.snapshot_id = Some(1);
        file.begin_snapshot = Some(1);
        file.max_row_count = Some(1);
        if let Some(value) = region {
            file.partition_id = Some(1);
            file.partition_values = vec![(0, value.map(str::to_string))];
        }
        let column_statistics = id_bounds
            .map(|(min, max)| DuckLakeFileColumnStatistics {
                data_file_id,
                column_id: ID_COLUMN_ID,
                column_size_bytes: Some(8),
                value_count: Some(1),
                null_count: Some(0),
                min_value: Some(min.to_string()),
                max_value: Some(max.to_string()),
                contains_nan: None,
            })
            .into_iter()
            .collect();
        DuckLakeFileMetadata {
            file,
            column_statistics,
        }
    }

    /// A spec partitioning by `region` with the `identity` transform.
    ///
    /// The partition key index (0) deliberately differs from `region`'s position
    /// in the schema (1): a bound written at the key index instead of the column
    /// index would land on `id`, the very column a keyed mutation filters on.
    fn region_spec(prune_safe: bool) -> PartitionSpec {
        PartitionSpec {
            partition_id: 1,
            columns: vec![PartitionSpecColumn {
                partition_key_index: 0,
                column_id: REGION_COLUMN_ID,
                transform: PartitionTransform::Identity,
            }],
            prune_safe,
        }
    }

    fn fixed_table(
        files: Vec<DuckLakeFileMetadata>,
        partition_spec: Option<PartitionSpec>,
    ) -> Result<DuckLakeTable> {
        DuckLakeTable::new(
            1,
            "events",
            Arc::new(FixedFileProvider {
                files,
                partition_spec,
                visible_when_filtered: None,
            }),
            1,
            Arc::new(ObjectStoreUrl::parse("memory://").unwrap()),
            String::new(),
        )
    }

    /// A table whose catalog honours a pushed-down filter, listing only
    /// `visible_when_filtered` for a page that carries one. An unfiltered
    /// listing — [`DuckLakeTable::files`] among them — still sees every file.
    fn catalog_filtering_table(
        files: Vec<DuckLakeFileMetadata>,
        visible_when_filtered: Vec<i64>,
    ) -> Result<DuckLakeTable> {
        DuckLakeTable::new(
            1,
            "events",
            Arc::new(FixedFileProvider {
                files,
                partition_spec: None,
                visible_when_filtered: Some(visible_when_filtered),
            }),
            1,
            Arc::new(ObjectStoreUrl::parse("memory://").unwrap()),
            String::new(),
        )
    }

    /// The physical expression a caller would build for a scan, and hand to both
    /// `files_matching` and `resolve_positions`.
    /// `predicate_is_prunable` must distinguish a REAL physical rename from the
    /// synthetic `_ducklake_internal_row_id -> rowid` entry that every file
    /// written by `UPDATE` or compaction carries.
    ///
    /// This is asserted here rather than end-to-end because the guard is read
    /// only inside `resolve_positions`, whose plan is built and executed by
    /// `DuckLakeDeleteExec` and never surfaces in `EXPLAIN ANALYZE`. An
    /// integration test can observe that a `DELETE` is correct, but not that it
    /// pruned.
    #[test]
    fn predicate_is_prunable_ignores_the_synthetic_rowid_rename() -> Result<()> {
        let table = DuckLakeTable::new(
            1,
            "events",
            Arc::new(LazyMillionFileProvider::default()),
            1,
            Arc::new(ObjectStoreUrl::parse("memory://").unwrap()),
            String::new(),
        )?;
        let predicate = physical_predicate(&table, col("id").eq(lit(5_i64)));

        let cfg = |renamed: Vec<usize>, mapping: Vec<(&str, &str)>| FileReadConfig {
            read_schema: table.physical_schema.clone(),
            name_mapping: mapping
                .into_iter()
                .map(|(a, b)| (a.to_string(), b.to_string()))
                .collect(),
            constants: HashMap::new(),
            row_group_count: 4,
            renamed_column_indices: Arc::new(renamed.into_iter().collect()),
            embedded_rowid_parquet_name: None,
            embedded_snapshot_parquet_name: None,
            drops_current_columns: false,
        };
        let id_index = table.physical_schema.index_of("id").unwrap();

        // A file written by UPDATE / compaction: one mapping entry, but no data
        // column is renamed. Must still prune.
        assert!(
            table.predicate_is_prunable(
                &predicate,
                &cfg(
                    vec![],
                    vec![(
                        crate::row_id::EMBEDDED_ROW_ID_COLUMN_NAME,
                        ROWID_COLUMN_NAME
                    )]
                )
            ),
            "the synthetic rowid rename must not disable pruning"
        );

        // The predicate's own column reads under a different physical name, so
        // it cannot bind in the reader and must not be pushed.
        assert!(
            !table.predicate_is_prunable(&predicate, &cfg(vec![id_index], vec![])),
            "a predicate on a renamed column must disable pruning"
        );

        // A DIFFERENT column is renamed or absent — the shape a file older than
        // an `ADD COLUMN` has. The predicate never references it, so pruning
        // stands. (This fixture's table has one column, so the unrelated index
        // is synthetic; what matters is that it is not the one the predicate
        // references.)
        assert!(
            table.predicate_is_prunable(&predicate, &cfg(vec![id_index + 1], vec![])),
            "an unrelated renamed/absent column must not disable pruning"
        );

        // Nothing renamed: prunable.
        assert!(table.predicate_is_prunable(&predicate, &cfg(vec![], vec![])));
        Ok(())
    }

    fn physical_predicate(table: &DuckLakeTable, expr: Expr) -> Arc<dyn PhysicalExpr> {
        let df_schema = DFSchema::try_from(table.physical_schema.as_ref().clone()).unwrap();
        SessionContext::new()
            .state()
            .create_physical_expr(expr, &df_schema)
            .unwrap()
    }

    fn matching_ids(table: &DuckLakeTable, predicate: &Arc<dyn PhysicalExpr>) -> Result<Vec<i64>> {
        Ok(table
            .files_matching(predicate)?
            .into_iter()
            .map(|file| file.data_file_id)
            .collect())
    }

    #[test]
    fn files_matching_does_not_judge_a_delete_bearing_file_all_null() -> Result<()> {
        // The hazard the restatement is built around, and the one the simpler
        // test above cannot see because its files have no nulls.
        //
        // `null_count` is physical (parquet footer) while `file_row_count` is
        // live (deletes subtracted). Present both as usable and DataFusion's
        // `null_count != row_count` guard reads two physical rows -- one NULL,
        // one matching -- with the NULL deleted as "one null, one row, so
        // entirely null", and prunes a file that still physically holds the row
        // the caller has to find. Withholding the row count is what stops that,
        // so this fails if a future change promotes it.
        let mut entry = fixture_file(1, None, Some((5, 5)));
        entry.file.max_row_count = Some(2); // physical: [NULL, 5]
        entry.file.delete_count = Some(1); // the NULL is deleted -> 1 live row
        entry.file.delete_file_id = Some(1);
        entry.file.delete_file = Some(DuckLakeFileData::new(
            "delete-1.parquet".to_string(),
            true,
            1,
        ));
        for stats in &mut entry.column_statistics {
            stats.null_count = Some(1);
            stats.value_count = Some(2);
        }

        let table = fixed_table(vec![entry], None)?;
        let predicate = physical_predicate(&table, col("id").eq(lit(5_i64)));
        assert_eq!(
            matching_ids(&table, &predicate)?,
            vec![1],
            "a physical null_count equal to the LIVE row count must not make the \
             file look entirely null -- the matching physical row is still there"
        );
        Ok(())
    }

    #[test]
    fn files_matching_prunes_a_delete_bearing_file_by_its_partition_bound() -> Result<()> {
        // Same fixture and predicate as
        // `files_matching_prunes_a_partition_column_without_column_statistics`,
        // with one variable changed: the file that must be pruned carries a
        // delete file. Its only bound is the one `apply_partition_bounds`
        // synthesises, which the restatement also promotes -- so this pins that
        // a delete-bearing file is still prunable when partition values are all
        // it has, which is the shape a partitioned keyed mutation actually hits.
        let delete_bearing = |data_file_id: i64, value: &'static str| {
            let mut entry = fixture_file(data_file_id, Some(Some(value)), None);
            entry.file.max_row_count = Some(10);
            entry.file.delete_count = Some(1);
            entry.file.delete_file_id = Some(data_file_id);
            entry.file.delete_file = Some(DuckLakeFileData::new(
                format!("delete-{data_file_id}.parquet"),
                true,
                1,
            ));
            entry
        };

        let table = fixed_table(
            vec![fixture_file(1, Some(Some("10")), None), delete_bearing(2, "9999")],
            Some(region_spec(true)),
        )?;

        let predicate = physical_predicate(&table, col("region").eq(lit(10_i64)));
        assert_eq!(
            matching_ids(&table, &predicate)?,
            vec![1],
            "a delete-bearing file whose partition bound excludes the value must \
             still prune"
        );
        Ok(())
    }

    #[test]
    fn files_matching_still_prunes_a_file_that_carries_deletes() -> Result<()> {
        // `build_datafusion_statistics` marks a delete-bearing file's bounds
        // Inexact, and the pruning view surfaces only Exact ones. So without the
        // physical-space restatement, the first mutation that writes a delete
        // file makes that file unprunable for every mutation after it — and
        // permanently, because compaction skips delete-bearing files. Both
        // directions are asserted: the file must still be dropped when its
        // bounds exclude the value, and still kept when they do not.
        let carrying_deletes = |data_file_id: i64, id_bounds: (i64, i64)| {
            let mut entry = fixture_file(data_file_id, None, Some(id_bounds));
            // Leave live rows > 0, so the separate known-empty exclusion is not
            // what drops the file.
            entry.file.max_row_count = Some(10);
            entry.file.delete_count = Some(1);
            entry.file.delete_file_id = Some(data_file_id);
            entry.file.delete_file = Some(DuckLakeFileData::new(
                format!("delete-{data_file_id}.parquet"),
                true,
                1,
            ));
            entry
        };

        let table = fixed_table(
            vec![
                fixture_file(1, None, Some((1, 10))),
                carrying_deletes(2, (100, 110)),
                fixture_file(3, None, Some((200, 210))),
            ],
            None,
        )?;

        let inside = physical_predicate(&table, col("id").eq(lit(105_i64)));
        assert_eq!(
            matching_ids(&table, &inside)?,
            vec![2],
            "a delete-bearing file whose bounds admit the value must be kept"
        );

        let outside = physical_predicate(&table, col("id").eq(lit(5_i64)));
        assert_eq!(
            matching_ids(&table, &outside)?,
            vec![1],
            "a delete-bearing file whose bounds exclude the value must still prune"
        );
        Ok(())
    }

    #[test]
    fn files_matching_prunes_unpartitioned_files_by_column_statistics() -> Result<()> {
        let table = fixed_table(
            vec![
                fixture_file(1, None, Some((1, 10))),
                fixture_file(2, None, Some((100, 110))),
                fixture_file(3, None, Some((200, 210))),
            ],
            None,
        )?;

        let predicate = physical_predicate(&table, col("id").eq(lit(105_i64)));

        assert_eq!(matching_ids(&table, &predicate)?, vec![2]);
        Ok(())
    }

    /// Hex-encoded 16-byte AES key, standing in for a catalog-recorded one.
    #[cfg(feature = "encryption")]
    const FIXTURE_ENCRYPTION_KEY: &str = "0123456789abcdef0123456789abcdef";

    /// A file the catalog records an encryption key for.
    #[cfg(feature = "encryption")]
    fn encrypted_fixture_file(data_file_id: i64, id_bounds: (i64, i64)) -> DuckLakeFileMetadata {
        let mut entry = fixture_file(data_file_id, None, Some(id_bounds));
        entry.file.file.encryption_key = Some(FIXTURE_ENCRYPTION_KEY.to_string());
        entry
    }

    /// The key `path` resolves to through the table's installed factory, or
    /// `None` when the factory holds no key for it. `None` is also what an
    /// unencrypted file gets, so only ever ask this about an encrypted one.
    #[cfg(feature = "encryption")]
    async fn installed_decryption_properties(
        table: &DuckLakeTable,
        path: &str,
    ) -> Option<Arc<parquet::encryption::decrypt::FileDecryptionProperties>> {
        let factory = table
            .encryption_keys
            .factory
            .get()
            .cloned()
            .expect("an encrypted table installs a factory");
        factory
            .get_file_decryption_properties(&Default::default(), &ObjectPath::from(path))
            .await
            .expect("key lookup succeeds")
    }

    /// Pruning a file must not take its decryption key with it. This catalog
    /// lists every file — the filter reaches a provider that ignores it, as the
    /// trait's default method does — so the in-memory pass prunes files whose
    /// metadata, keys included, was collected on the way past.
    /// `files_matching_keeps_keys_the_catalog_filter_left_unlisted` covers the
    /// other half, where the file never reaches the loop at all.
    #[cfg(feature = "encryption")]
    #[tokio::test]
    async fn files_matching_keeps_decryption_keys_for_pruned_files() {
        let table = fixed_table(
            vec![encrypted_fixture_file(1, (1, 10)), encrypted_fixture_file(2, (100, 110))],
            None,
        )
        .expect("table opens");

        let predicate = physical_predicate(&table, col("id").eq(lit(105_i64)));
        let matching = table.files_matching(&predicate).expect("pruning succeeds");
        assert_eq!(
            matching.len(),
            1,
            "file 1's statistics exclude the key, so it is pruned"
        );

        assert!(
            installed_decryption_properties(&table, "file-1.parquet")
                .await
                .is_some(),
            "the pruned file's key must stay installed for other readers",
        );
    }

    /// A file the catalog filter leaves unlisted must keep the key an earlier
    /// listing installed for it.
    ///
    /// Catalog-side filtering is not the same as pruning: the file never reaches
    /// the page loop, so nothing there can collect its key. Installing the keys
    /// of the listed files *in place of* the table's set would therefore strand
    /// the key of every file the mutation did not touch — including the ones the
    /// caller is holding from its own `files()` listing and is about to read
    /// through `resolve_positions`, which opens them with whatever factory is
    /// installed. Merging is what keeps those readable.
    #[cfg(feature = "encryption")]
    #[tokio::test]
    async fn files_matching_keeps_keys_the_catalog_filter_left_unlisted() {
        let table = catalog_filtering_table(
            vec![encrypted_fixture_file(1, (1, 10)), encrypted_fixture_file(2, (100, 110))],
            // Only file 2 survives the filter this predicate lowers to.
            vec![2],
        )
        .expect("table opens");

        // How a keyed mutation gets its files: an unfiltered listing first, which
        // sees the whole snapshot and installs both keys.
        assert_eq!(table.files().expect("listing succeeds").len(), 2);

        let predicate = physical_predicate(&table, col("id").eq(lit(105_i64)));
        let matching = table.files_matching(&predicate).expect("pruning succeeds");
        assert_eq!(
            matching
                .iter()
                .map(|file| file.data_file_id)
                .collect::<Vec<_>>(),
            vec![2],
            "the catalog leaves file 1 out of the listing entirely"
        );

        assert!(
            installed_decryption_properties(&table, "file-1.parquet")
                .await
                .is_some(),
            "a file the catalog never listed must keep the key an earlier listing installed",
        );
        assert!(
            installed_decryption_properties(&table, "file-2.parquet")
                .await
                .is_some(),
            "the returned file's own key must be installed",
        );
    }

    /// A second install of the same path, carrying a different key, must be the
    /// key the factory then serves.
    ///
    /// Nothing in this crate produces one: every data and delete file is named
    /// by a fresh `Uuid::new_v4()`, compaction and partitioned and streaming
    /// writes included. The guarantee is still load-bearing, because the
    /// alternative is silent and permanent — a factory answering out of its own
    /// copy of the map would hand every reader the superseded key for the rest
    /// of the table's life, with the installed set and the served set
    /// disagreeing and no way to tell from either side.
    #[cfg(feature = "encryption")]
    #[tokio::test]
    async fn installing_a_second_key_for_a_path_supersedes_the_first() {
        /// A different valid key, so the two are told apart by what the factory
        /// returns rather than by one of them failing to decode.
        const SUPERSEDED_KEY: &str = "fedcba9876543210fedcba9876543210";

        let table = fixed_table(Vec::new(), None).expect("table opens");
        let install = |key: &str| {
            table.install_encryption_keys(HashMap::from([(
                "file-1.parquet".to_string(),
                key.to_string(),
            )]));
        };

        install(SUPERSEDED_KEY);
        // The same path with a new key: `HashMap::extend` overwrites the entry
        // and leaves the length alone, so "did the length change" is not a test
        // of whether anything was installed.
        install(FIXTURE_ENCRYPTION_KEY);

        let properties = installed_decryption_properties(&table, "file-1.parquet")
            .await
            .expect("the path has a key installed");
        assert_eq!(
            properties
                .footer_key(None)
                .expect("a footer key")
                .as_slice(),
            DuckLakeEncryptionFactory::decode_key(FIXTURE_ENCRYPTION_KEY)
                .expect("the fixture key decodes")
                .as_slice(),
            "the factory must serve the key installed last, not the superseded one",
        );
    }

    /// One factory serves every page of a listing, and serves keys installed
    /// after it was handed out.
    ///
    /// Both halves matter. A reader that cloned the factory while planning
    /// page 1 must be able to open a file listed on page 50 — that is the
    /// guarantee accumulation exists for. And because the factory reads the
    /// installed map rather than a copy of it, no page rebuilds it: rebuilding
    /// per page would clone every key collected so far and make a P-page
    /// listing quadratic in P.
    #[cfg(feature = "encryption")]
    #[tokio::test]
    async fn one_factory_serves_the_keys_of_every_page() {
        let table = fixed_table(Vec::new(), None).expect("table opens");
        let install = |path: &str| {
            table.install_encryption_keys(HashMap::from([(
                path.to_string(),
                FIXTURE_ENCRYPTION_KEY.to_string(),
            )]));
        };
        let installed_factory = || {
            table
                .encryption_keys
                .factory
                .get()
                .cloned()
                .expect("an encrypted table installs a factory")
        };

        install("file-1.parquet");
        let from_first_page = installed_factory();
        install("file-2.parquet");

        assert!(
            Arc::ptr_eq(&from_first_page, &installed_factory()),
            "a later page must reuse the installed factory, not rebuild it",
        );
        assert!(
            from_first_page
                .get_file_decryption_properties(
                    &Default::default(),
                    &ObjectPath::from("file-2.parquet"),
                )
                .await
                .expect("key lookup succeeds")
                .is_some(),
            "the factory handed out on page 1 must resolve a key installed on page 2",
        );
    }

    /// A file with no recorded statistics is always kept, while exact bounds can
    /// still prune the files beside it.
    #[test]
    fn files_matching_keeps_files_without_statistics() -> Result<()> {
        let table = fixed_table(
            vec![
                fixture_file(1, None, Some((1, 10))),
                fixture_file(2, None, None),
                fixture_file(3, None, Some((200, 210))),
            ],
            None,
        )?;

        let predicate = physical_predicate(&table, col("id").eq(lit(105_i64)));

        assert_eq!(matching_ids(&table, &predicate)?, vec![2]);
        Ok(())
    }

    /// A file the catalog records as holding no rows, and — the case that matters —
    /// carrying no per-column statistics row at all.
    ///
    /// A statistics harvest failure
    /// ([`crate::stats_collect::collect_column_stats`] yields no rows) and a column
    /// added by later DDL both produce this shape.
    fn empty_file_without_statistics(data_file_id: i64) -> DuckLakeFileMetadata {
        let mut entry = fixture_file(data_file_id, None, None);
        entry.file.max_row_count = Some(0);
        entry
    }

    /// A file whose `record_count` the catalog leaves unset — what a provider that
    /// does not surface one produces. Unknown, never empty.
    fn file_with_unknown_row_count(
        data_file_id: i64,
        id_bounds: (i64, i64),
    ) -> DuckLakeFileMetadata {
        let mut entry = fixture_file(data_file_id, None, Some(id_bounds));
        entry.file.max_row_count = None;
        entry
    }

    /// A 0-row file with no statistics row of its own, alongside files whose
    /// statistics are ideal.
    ///
    /// The control is
    /// `files_matching_prunes_unpartitioned_files_by_column_statistics`: the same
    /// three row-bearing files without the empty one, same predicate, same result.
    #[test]
    fn files_matching_prunes_around_an_empty_file_without_statistics() -> Result<()> {
        let table = fixed_table(
            vec![
                fixture_file(1, None, Some((1, 10))),
                fixture_file(2, None, Some((100, 110))),
                fixture_file(3, None, Some((200, 210))),
                empty_file_without_statistics(4),
            ],
            None,
        )?;

        let predicate = physical_predicate(&table, col("id").eq(lit(105_i64)));

        assert_eq!(matching_ids(&table, &predicate)?, vec![2]);
        Ok(())
    }

    /// Only a recorded row count of exactly 0 may drop a file. A catalog that
    /// leaves `record_count` unset must never be read as "empty": that would
    /// withhold a file holding live rows, and a keyed mutation that never sees the
    /// file holding its key inserts a duplicate instead of superseding it, with no
    /// error anywhere. Pinned because the mis-implementation is a one-token slip
    /// (`max_row_count.unwrap_or(0) == 0`) that nothing else here would catch.
    ///
    /// File 2 carries the unknown count and is the only file whose bounds admit the
    /// key, so reading unknown as empty collapses this result to nothing.
    #[test]
    fn files_matching_keeps_a_file_whose_row_count_is_unknown() -> Result<()> {
        let table = fixed_table(
            vec![
                fixture_file(1, None, Some((1, 10))),
                file_with_unknown_row_count(2, (100, 110)),
                empty_file_without_statistics(3),
            ],
            None,
        )?;

        let predicate = physical_predicate(&table, col("id").eq(lit(105_i64)));

        assert_eq!(matching_ids(&table, &predicate)?, vec![2]);
        Ok(())
    }

    /// A NULL partition value yields no bound, so that file must survive a
    /// predicate on its own partition column. Exact bounds can still prune the
    /// files beside it.
    ///
    /// The control is
    /// `files_matching_prunes_a_partition_column_without_column_statistics`:
    /// files 1 and 2 alone, same predicate, returns file 1.
    #[test]
    fn files_matching_keeps_a_file_with_a_null_partition_value() -> Result<()> {
        let table = fixed_table(
            vec![
                fixture_file(1, Some(Some("10")), None),
                fixture_file(2, Some(Some("9999")), None),
                fixture_file(3, Some(None), None),
            ],
            Some(region_spec(true)),
        )?;

        let predicate = physical_predicate(&table, col("region").eq(lit(10_i64)));

        assert_eq!(matching_ids(&table, &predicate)?, vec![1, 3]);
        Ok(())
    }

    /// A partition value that does not decode to the column type yields no
    /// bound, with the same consequences as a NULL one. Deliberately kept apart
    /// from the NULL case so either mistake fails independently.
    #[test]
    fn files_matching_keeps_a_file_with_an_undecodable_partition_value() -> Result<()> {
        let table = fixed_table(
            vec![
                fixture_file(1, Some(Some("10")), None),
                fixture_file(2, Some(Some("9999")), None),
                fixture_file(3, Some(Some("not-a-region")), None),
            ],
            Some(region_spec(true)),
        )?;

        let predicate = physical_predicate(&table, col("region").eq(lit(10_i64)));

        assert_eq!(matching_ids(&table, &predicate)?, vec![1, 3]);
        Ok(())
    }

    /// A predicate on a non-partition column must never be pruned by partition
    /// bounds. Getting this wrong would drop a file that holds the key, and a
    /// keyed mutation would then insert a second copy of that key instead of
    /// superseding it — silently, with no error anywhere.
    ///
    /// `files_matching_prunes_a_partition_column_without_column_statistics` is the
    /// counterpart that proves partition bounds really are live in this fixture,
    /// so this test cannot pass merely because nothing prunes.
    #[test]
    fn files_matching_never_prunes_a_non_partition_predicate_by_partition_bounds() -> Result<()> {
        let table = fixed_table(
            vec![
                // Partition value far outside the predicate's range and no `id`
                // statistics: only a partition bound leaking onto `id` could drop it.
                fixture_file(1, Some(Some("9999")), None),
                // Same distant partition value, but `id` statistics that admit the
                // key — the file genuinely holds a matching row.
                fixture_file(2, Some(Some("9999")), Some((100, 110))),
            ],
            Some(region_spec(true)),
        )?;

        let predicate = physical_predicate(&table, col("id").eq(lit(105_i64)));

        assert_eq!(matching_ids(&table, &predicate)?, vec![1, 2]);
        Ok(())
    }

    #[test]
    fn files_matching_prunes_a_partition_column_without_column_statistics() -> Result<()> {
        let table = fixed_table(
            vec![
                fixture_file(1, Some(Some("10")), None),
                fixture_file(2, Some(Some("9999")), None),
            ],
            Some(region_spec(true)),
        )?;

        let predicate = physical_predicate(&table, col("region").eq(lit(10_i64)));

        assert_eq!(matching_ids(&table, &predicate)?, vec![1]);
        Ok(())
    }

    /// Same fixture and predicate as
    /// `files_matching_prunes_a_partition_column_without_column_statistics`, with
    /// the single difference that the spec is not prune-safe — the table has been
    /// re-partitioned, so a file's values may belong to a retired generation whose
    /// key order differs. Nothing may be dropped.
    #[test]
    fn files_matching_ignores_partition_bounds_when_the_spec_is_not_prune_safe() -> Result<()> {
        let table = fixed_table(
            vec![
                fixture_file(1, Some(Some("10")), None),
                fixture_file(2, Some(Some("9999")), None),
            ],
            Some(region_spec(false)),
        )?;

        let predicate = physical_predicate(&table, col("region").eq(lit(10_i64)));

        assert_eq!(matching_ids(&table, &predicate)?, vec![1, 2]);
        Ok(())
    }

    #[test]
    fn files_matching_applies_every_conjunct() -> Result<()> {
        let table = fixed_table(
            vec![
                fixture_file(1, Some(Some("10")), Some((100, 110))),
                // Excluded by the partition conjunct alone.
                fixture_file(2, Some(Some("9999")), Some((100, 110))),
                // Excluded by the key conjunct alone.
                fixture_file(3, Some(Some("10")), Some((1, 10))),
            ],
            Some(region_spec(true)),
        )?;

        let predicate = physical_predicate(
            &table,
            col("id")
                .eq(lit(105_i64))
                .and(col("region").eq(lit(10_i64))),
        );

        assert_eq!(matching_ids(&table, &predicate)?, vec![1]);
        Ok(())
    }

    #[test]
    fn files_matching_returns_every_file_for_a_trivially_true_predicate() -> Result<()> {
        let table = fixed_table(
            vec![
                fixture_file(1, None, Some((1, 10))),
                fixture_file(2, None, Some((100, 110))),
                fixture_file(3, None, None),
            ],
            None,
        )?;

        let predicate = physical_predicate(&table, lit(true));

        assert_eq!(matching_ids(&table, &predicate)?, vec![1, 2, 3]);
        Ok(())
    }

    /// A million-file table read in bounded pages, with one match in the first
    /// page and one in the last: a result assembled from a single page, or from
    /// an eager whole-table file read, cannot produce both.
    #[test]
    fn files_matching_prunes_across_every_metadata_page() -> Result<()> {
        let provider = Arc::new(LazyMillionFileProvider::default());
        let table = DuckLakeTable::new(
            1,
            "events",
            provider.clone(),
            1,
            Arc::new(ObjectStoreUrl::parse("memory://").unwrap()),
            String::new(),
        )?;

        let predicate = physical_predicate(
            &table,
            col("id").eq(lit(5_i64)).or(col("id").eq(lit(999_999_i64))),
        );

        assert_eq!(matching_ids(&table, &predicate)?, vec![5, 999_999]);
        assert!(provider.page_calls.load(Ordering::Relaxed) > 1);
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

            // Neither bound survives an unknown/positive NaN state: DataFusion
            // orders floats with `total_cmp`, under which negative NaN sits
            // below every value (below -Infinity included), so a recorded min is
            // no more trustworthy than a recorded max. `contains_nan` does not
            // record the sign, so both must go.
            let per_file = &file_stats[&7].column_statistics[0];
            let table_col = &table_stats.column_statistics[0];
            let expected_min = if contains_nan == Some(false) {
                Precision::Exact(ScalarValue::Float64(Some(1.0)))
            } else {
                Precision::Absent
            };
            assert_eq!(
                per_file.min_value, expected_min,
                "contains_nan={contains_nan:?}"
            );
            assert_eq!(
                table_col.min_value, expected_min,
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
            // Both bounds are gated: negative NaN sits below a recorded min just
            // as positive NaN sits above a recorded max, and `contains_nan`
            // does not record the sign.
            let expected_min = if max_usable {
                Precision::Exact(ScalarValue::Float64(Some(1.0)))
            } else {
                Precision::Absent
            };
            assert_eq!(
                column.min_value, expected_min,
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

        // One file NaN-unknown: NEITHER of its bounds is trusted, so both
        // aggregates degrade to unknown. A negative NaN in that file would sit
        // below the folded min exactly as a positive one would sit above the
        // folded max.
        let statistics = build(None);
        assert_eq!(statistics.column_statistics[0].min_value, Precision::Absent);
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
        assert_eq!(
            parse_statistic_scalar("FALSE", &boolean, &DataType::Boolean),
            Some(ScalarValue::Boolean(Some(false)))
        );

        let blob = DuckLakeTableColumn::new(2, "bytes".to_string(), "blob".to_string(), true);
        assert_eq!(
            parse_statistic_scalar("68656C6C6F", &blob, &DataType::BinaryView),
            Some(ScalarValue::BinaryView(Some(b"hello".to_vec())))
        );
        assert_eq!(
            parse_statistic_scalar("\\x68656C6C6F", &blob, &DataType::BinaryView),
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

    #[test]
    fn expression_column_default_does_not_block_table_reads() {
        let column =
            DuckLakeTableColumn::new(1, "created_at".to_string(), "timestamp".to_string(), false)
                .with_defaults(
                    None,
                    Some("current_timestamp".to_string()),
                    Some("expression".to_string()),
                    Some("duckdb".to_string()),
                );

        assert!(validate_column_defaults(&[column]).unwrap().is_empty());
    }

    #[test]
    fn null_default_sentinel_does_not_create_an_operational_default() {
        let column = DuckLakeTableColumn::new(1, "id".to_string(), "int32".to_string(), false)
            .with_defaults(
                None,
                Some("NULL".to_string()),
                Some("literal".to_string()),
                Some("duckdb".to_string()),
            );

        assert!(column.default_value.is_none());
        assert!(validate_column_defaults(&[column]).unwrap().is_empty());
    }

    // `sort_ordering_for` is `#[cfg(feature = "write")]`, so the test that exercises
    // it needs the same gate — otherwise a read-only feature combination fails to
    // compile the test target.
    #[cfg(feature = "write")]
    #[test]
    fn sort_ordering_rejects_expression_sort_key() {
        let schema = Schema::new(vec![Field::new("id", DataType::Int64, false)]);
        let sort_spec = crate::sort::SortSpec {
            sort_id: 9,
            fields: vec![crate::sort::SortField {
                sort_key_index: 0,
                expression: "lower(id)".to_string(),
                dialect: crate::sort::DUCKDB_DIALECT.to_string(),
                direction: crate::sort::SortDirection::Asc,
                null_order: crate::sort::NullOrder::NullsLast,
            }],
        };

        let err = sort_ordering_for(&schema, Some(&sort_spec)).unwrap_err();

        assert_eq!(
            err.to_string(),
            "This feature is not implemented: DuckLake sort order 9 contains an unsupported \
             expression; datafusion-ducklake can write only bare-column sort keys",
        );
    }
}
