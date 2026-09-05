use crate::DuckLakeError;
use crate::metadata_provider::{
    ColumnWithTable, DataFileChange, DeleteFileChange, DuckLakeFileColumnStatistics,
    DuckLakeFileData, DuckLakeFileMetadata, DuckLakeInlinedData, DuckLakeInlinedDelete,
    DuckLakeNameMapping, DuckLakeNameMappingEntry, DuckLakeStatistics, DuckLakeTableColumn,
    DuckLakeTableColumnStatistics, DuckLakeTableField, DuckLakeTableFile, DuckLakeTableStatistics,
    FileWithTable, INLINED_DATA_REMEDIATION, MetadataProvider, MetadataSetting, SQL_GET_DATA_FILES,
    SQL_GET_DATA_FILES_ADDED_BETWEEN_SNAPSHOTS, SQL_GET_DELETE_FILES_ADDED_BETWEEN_SNAPSHOTS,
    SQL_GET_FILE_COLUMN_STATS, SQL_GET_FILE_PARTITION_VALUES, SQL_GET_LATEST_SNAPSHOT,
    SQL_GET_NAME_MAPPING, SQL_GET_PARTITION_SPEC, SQL_GET_SCHEMA_BY_NAME, SQL_GET_SORT_SPEC,
    SQL_GET_TABLE_BY_NAME, SQL_GET_TABLE_COLUMN_STATS, SQL_GET_TABLE_STATS, SQL_GET_VIEW_BY_NAME,
    SQL_LIST_ALL_FILES, SQL_LIST_ALL_TABLES, SQL_LIST_ALL_VIEWS, SQL_LIST_SCHEMAS,
    SQL_LIST_SNAPSHOTS, SQL_LIST_TABLES, SQL_LIST_VIEWS, SQL_TABLE_EXISTS, SchemaMetadata,
    SnapshotChangeMetadata, SnapshotMetadata, TableMetadata, TableWithSchema, ViewMetadata,
    ViewWithSchema, build_inlined_batch, inlined_delete_table_name, inlined_missing_scalar,
    is_inlined_data_table, reconstruct_columns, reconstruct_columns_with_table,
    resolve_metadata_settings,
};
use crate::partition::PartitionSpec;
use crate::sort::SortSpec;
use crate::stats_filter::{RenderedColumnFilter, StatsFilter, StatsLiteral, StatsSqlDialect};
use arrow::datatypes::{DataType, SchemaRef, TimeUnit};
use arrow::record_batch::RecordBatch;
use datafusion::common::ScalarValue;
use duckdb::AccessMode::ReadOnly;
use duckdb::types::{TimeUnit as DuckdbTimeUnit, Value, ValueRef};
use duckdb::{Config, Connection, OptionalExt, params};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

fn is_missing_statistics_table(error: &duckdb::Error) -> bool {
    let message = error.to_string().to_ascii_lowercase();
    message.contains("does not exist") || message.contains("not found")
}

fn quote_ident(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

/// DuckDB's spelling of the few SQL atoms statistics filtering needs.
///
/// DuckDB is the engine official DuckLake generates this SQL for, so every atom
/// here is what official emits verbatim: `TRY_CAST` is native, the default
/// `VARCHAR` collation is already byte-wise, and `contains_nan` is a real
/// `BOOLEAN`. The other backends have to work to reproduce these; this one is
/// the reference.
struct DuckdbStatsDialect;

impl StatsSqlDialect for DuckdbStatsDialect {
    /// The *stat* is gated on its shape before it is cast, by
    /// [`duckdb_castable_pattern`]. `TRY_CAST` on its own is not sufficient:
    /// DuckDB reads text this crate never writes — `nan`, `epoch`, `0x10` — into
    /// real values, and comparing against one of those prunes files that match.
    ///
    /// The constant is the other half of the comparison, and it is not cast. It
    /// is spliced as a bare quoted literal on the far side, where DuckDB
    /// converts it to the cast's target type eagerly and *raises* on one it
    /// cannot read rather than returning `NULL`:
    ///
    /// ```text
    /// TRY_CAST('2024-01-01' AS DATE) < '+12921-08-18'
    ///   Conversion Error: invalid date field format: "+12921-08-18"
    /// ```
    ///
    /// `stats_encode` writes that sign-prefixed year for any date past 9999 or
    /// before the common era, and [`crate::DuckLakeTable::files_matching`] takes
    /// an arbitrary `PhysicalExpr`, so nothing upstream keeps such a constant
    /// out. A temporal constant is therefore admitted only in the fixed-width
    /// four-digit-year encoding; declining one drops that comparison, which
    /// costs pruning and never rows.
    fn try_cast(&self, expr: &str, literal: &StatsLiteral, data_type: &DataType) -> Option<String> {
        if !duckdb_reads_constant(literal.text(), data_type) {
            return None;
        }
        let target = duckdb_cast_type(data_type)?;
        let pattern = duckdb_castable_pattern(data_type)?;
        Some(format!(
            "CASE WHEN regexp_full_match({expr}, '{pattern}') THEN TRY_CAST({expr} AS {target}) END"
        ))
    }

    /// Nothing to add. DuckDB compares `VARCHAR` byte-wise, which is the
    /// semantics DataFusion's `Utf8` comparison has, so a raw stat comparison
    /// already agrees with the in-memory pruning it pre-filters for.
    fn collate_binary(&self, expr: &str) -> String {
        expr.to_string()
    }

    fn boolean_is_not_false(&self, expr: &str) -> String {
        format!("{expr} IS NULL OR {expr} <> false")
    }
}

/// Whether DuckDB will read the constant `text` as a value of `data_type`
/// instead of raising a conversion error.
///
/// Only the temporal types need the test. A numeric constant renders unquoted,
/// so there is no string for DuckDB to parse at all, and a boolean renders as
/// `'true'` / `'false'`, both of which it always reads. A date or timestamp
/// renders quoted, and `chrono` signs any year outside `0000..=9999`:
/// `+12921-08-18` for one past 9999, `-0044-03-15` for one before the common
/// era. DuckDB raises `invalid date field format` on the `+` form, which is what
/// this keeps out. It does read the `-` form, so declining that one costs
/// pruning and nothing else, and the single fixed-width shape test is worth more
/// than the case it gives up. `0000-01-01` carries no sign and is admitted.
fn duckdb_reads_constant(text: &str, data_type: &DataType) -> bool {
    match data_type {
        DataType::Date32 | DataType::Date64 => is_canonical_date(text),
        DataType::Timestamp(_, None) => is_canonical_timestamp(text),
        DataType::Timestamp(_, Some(_)) => is_canonical_timestamptz(text),
        _ => true,
    }
}

/// Whether `text` is `YYYY-MM-DD` with a four-digit year.
fn is_canonical_date(text: &str) -> bool {
    matches_shape(text, "NNNN-NN-NN")
}

/// Whether `text` is `YYYY-MM-DD HH:MM:SS` with an optional fractional part,
/// all of it digits — what [`crate::stats_encode`] writes for a naive timestamp.
fn is_canonical_timestamp(text: &str) -> bool {
    let Some((base, fraction)) = text.split_at_checked(19) else {
        return false;
    };
    matches_shape(base, "NNNN-NN-NN NN:NN:NN")
        && match fraction.strip_prefix('.') {
            None => fraction.is_empty(),
            Some(digits) => {
                !digits.is_empty() && digits.bytes().all(|digit| digit.is_ascii_digit())
            },
        }
}

/// Whether `text` is a canonical timestamp carrying the `+00` suffix
/// [`crate::stats_encode`] appends after normalizing a zoned value to UTC.
fn is_canonical_timestamptz(text: &str) -> bool {
    text.strip_suffix("+00").is_some_and(is_canonical_timestamp)
}

/// Whether `text` matches `shape` character for character, where `N` stands for
/// any ASCII digit and every other character stands for itself.
fn matches_shape(text: &str, shape: &str) -> bool {
    text.len() == shape.len()
        && text
            .bytes()
            .zip(shape.bytes())
            .all(|(actual, expected)| match expected {
                b'N' => actual.is_ascii_digit(),
                _ => actual == expected,
            })
}

/// The DuckDB type name a statistic is cast to for comparison against a
/// constant of `data_type`, or `None` when DuckDB has no type that holds that
/// constant's values exactly.
///
/// Arrow's own `Display` is not SQL — `Timestamp(Microsecond, None)` and `Utf8`
/// are not names DuckDB knows — so the mapping is written out rather than
/// formatted from the type. Declining a type drops that one comparison, which
/// costs pruning and never rows, so anything that would not round-trip is
/// declined:
///
/// - `Float16` has no DuckDB type. Widening to `FLOAT` would compare a
///   half-precision bound against a single-precision literal.
/// - A nanosecond timestamp with a time zone has no DuckDB type either
///   (`TIMESTAMP_TZ_NS` does not exist) and `TIMESTAMPTZ` is microseconds.
///   Truncating a bound *toward* the constant would prune a file whose true
///   bound still admits a matching row. Second and millisecond precision widen
///   into `TIMESTAMPTZ` losslessly, so those are kept.
/// - A `DECIMAL` outside DuckDB's `precision <= 38`, `0 <= scale <= precision`
///   range is not a type DuckDB will cast to at all.
///
/// Everything else is absent because `stats_encode::encode_scalar` has no
/// canonical text for it (`TIME`, `INTERVAL`, `Decimal256`, `UUID`, blobs,
/// nested types), so no literal of that type ever reaches here.
/// A pattern admitting only the text [`crate::stats_encode`] writes for this
/// type, for use as a `regexp_full_match` gate on a stat value.
///
/// `TRY_CAST` alone is not enough. It answers "can DuckDB read this", and DuckDB
/// reads a good deal this crate never writes: `nan` and `NaN` become NaN, `epoch`
/// becomes 1970-01-01, `infinity` becomes an infinite date, `0x10` becomes 16.
/// Each casts to a *value*, which then prunes files — a `nan` bound makes
/// `x = 5.0` false for a file whose other rows match, and the in-memory path
/// would have kept that file. Official DuckLake carries the same shape and is
/// safe only because DuckDB wrote every stat it reads; a catalog this crate opens
/// may have been written by anything, including `ducklake_add_files` over a
/// pre-1.11 parquet-mr file whose float min/max hold NaN.
///
/// `inf` and `-inf` stay admitted: this crate writes them and DuckDB orders them
/// correctly.
fn duckdb_castable_pattern(data_type: &DataType) -> Option<&'static str> {
    Some(match data_type {
        DataType::Boolean => r"^(true|false)$",
        DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64 => r"^-?[0-9]{1,20}$",
        DataType::Decimal128(_, _) => r"^-?[0-9]{1,38}(\.[0-9]{1,38})?$",
        DataType::Float16 | DataType::Float32 | DataType::Float64 => {
            r"^-?(inf|[0-9]+(\.[0-9]+)?(e[-+][0-9]{1,3})?)$"
        },
        // Fixed-width four-digit year only, matching the constant guard. DuckDB
        // does read a sign-prefixed year, but admitting one on the stat side
        // while the constant side declines it would compare two encodings that
        // do not order together.
        DataType::Date32 | DataType::Date64 => r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$",
        DataType::Timestamp(_, None) => {
            r"^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}(\.[0-9]{1,9})?$"
        },
        DataType::Timestamp(_, Some(_)) => {
            r"^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}(\.[0-9]{1,9})?\+00$"
        },
        _ => return None,
    })
}

fn duckdb_cast_type(data_type: &DataType) -> Option<String> {
    let name = match data_type {
        DataType::Boolean => "BOOLEAN",
        DataType::Int8 => "TINYINT",
        DataType::Int16 => "SMALLINT",
        DataType::Int32 => "INTEGER",
        DataType::Int64 => "BIGINT",
        DataType::UInt8 => "UTINYINT",
        DataType::UInt16 => "USMALLINT",
        DataType::UInt32 => "UINTEGER",
        DataType::UInt64 => "UBIGINT",
        DataType::Float32 => "FLOAT",
        DataType::Float64 => "DOUBLE",
        DataType::Date32 | DataType::Date64 => "DATE",
        DataType::Timestamp(TimeUnit::Second, None) => "TIMESTAMP_S",
        DataType::Timestamp(TimeUnit::Millisecond, None) => "TIMESTAMP_MS",
        DataType::Timestamp(TimeUnit::Microsecond, None) => "TIMESTAMP",
        DataType::Timestamp(TimeUnit::Nanosecond, None) => "TIMESTAMP_NS",
        DataType::Timestamp(
            TimeUnit::Second | TimeUnit::Millisecond | TimeUnit::Microsecond,
            Some(_),
        ) => "TIMESTAMPTZ",
        DataType::Decimal128(precision, scale)
            if *precision <= 38 && *scale >= 0 && i16::from(*scale) <= i16::from(*precision) =>
        {
            return Some(format!("DECIMAL({precision}, {scale})"));
        },
        _ => return None,
    };
    Some(name.to_string())
}

/// The `WHERE` keyword that opens [`SQL_GET_DATA_FILES`]' own predicate.
///
/// Statistics joins have to sit between the `FROM` list and the `WHERE` clause,
/// the only place SQL accepts a join, so the shared const is split here rather
/// than duplicated for the filtered case.
const DATA_FILES_WHERE: &str = "WHERE data.table_id = ?";

/// Narrow [`SQL_GET_DATA_FILES`] with per-column catalog statistics, or `None`
/// when there is nothing to narrow it with.
///
/// Reproduces official DuckLake's three pieces —
/// `GenerateCTESectionFromRequirements`, `GenerateStatsJoinList` and the
/// conditions from `ConvertFilterPushdownToSQL`: one CTE per filtered column
/// selecting only the stats that column's condition reads, one `LEFT JOIN` per
/// CTE on `data.data_file_id`, and the conditions ANDed onto the existing
/// `WHERE`. Each CTE restricts to a single `(column_id, table_id)` pair, which
/// is what keeps the join at one row per file.
///
/// Values are inlined, never bound. The conditions arrive from
/// [`crate::stats_filter`] as finished SQL text, and the only values this adds
/// are `i64`s formatted as decimals, so the query gains no placeholder and the
/// caller's `params![]` list stays exactly the unfiltered query's.
///
/// `None` also covers a [`SQL_GET_DATA_FILES`] that no longer contains
/// [`DATA_FILES_WHERE`]: dropping the filter only costs pruning, whereas
/// splicing at the wrong place would produce a query that is silently wrong.
fn data_files_sql_filtered(table_id: i64, filters: &[RenderedColumnFilter]) -> Option<String> {
    if filters.is_empty() {
        return None;
    }
    let (from_section, where_section) = SQL_GET_DATA_FILES.split_once(DATA_FILES_WHERE)?;

    let ctes = filters
        .iter()
        .map(|filter| {
            format!(
                "{alias} AS (
        SELECT data_file_id, {stats}
        FROM ducklake_file_column_stats
        WHERE column_id = {column_id} AND table_id = {table_id}
    )",
                alias = filter.alias,
                stats = filter.stats.join(", "),
                column_id = filter.column_id,
            )
        })
        .collect::<Vec<_>>()
        .join(",\n    ");
    let joins = filters
        .iter()
        .map(|filter| {
            format!(
                "\n    LEFT JOIN {alias} ON {alias}.data_file_id = data.data_file_id",
                alias = filter.alias
            )
        })
        .collect::<String>();
    let conditions = filters
        .iter()
        .map(|filter| filter.condition.as_str())
        .collect::<Vec<_>>()
        .join("\n      AND ");

    // The stats joins land after the delete-file join rather than before it,
    // where official puts them. Both are LEFT JOINs of `data` and neither ON
    // clause mentions the other, so the two orders produce the same rows; this
    // one leaves the shared const's own text contiguous.
    Some(format!(
        "WITH {ctes}{from}{joins}\n    {DATA_FILES_WHERE}{where_section}\n      AND {conditions}",
        from = from_section.trim_end(),
    ))
}

/// Read one page of the data-file listing, optionally narrowed by catalog
/// statistics.
///
/// Returns the raw `duckdb::Error` so the caller can recognise a catalog that
/// predates `ducklake_file_column_stats` and retry unfiltered.
fn query_data_file_page(
    conn: &Connection,
    table_id: i64,
    snapshot_id: i64,
    after_data_file_id: i64,
    limit: i64,
    filters: &[RenderedColumnFilter],
) -> Result<Vec<DuckLakeTableFile>, duckdb::Error> {
    let base = data_files_sql_filtered(table_id, filters)
        .unwrap_or_else(|| SQL_GET_DATA_FILES.to_string());
    // The statistics conditions sit inside the query, ahead of the LIMIT, and
    // the keyset ordering is untouched. Filtering the page after fetching it
    // would break the cursor `FileMetadataPages` drives: a page whose
    // candidates all failed would come back empty, ending the iteration and
    // hiding every matching file beyond it.
    let sql = format!(
        "{base}
             AND data.data_file_id > ?
             ORDER BY data.data_file_id
             LIMIT ?"
    );
    let mut statement = conn.prepare(&sql)?;
    statement
        .query_map(
            params![
                table_id,
                snapshot_id,
                snapshot_id,
                table_id,
                snapshot_id,
                snapshot_id,
                after_data_file_id,
                limit
            ],
            |row| {
                let delete_file_id: Option<i64> = row.get(8)?;
                let (delete_file, delete_count) = if delete_file_id.is_some() {
                    (
                        Some(DuckLakeFileData {
                            path: row.get(9)?,
                            path_is_relative: row.get(10)?,
                            file_size_bytes: row.get(11)?,
                            footer_size: row.get(12)?,
                            encryption_key: row.get(13)?,
                            mapping_id: None,
                        }),
                        row.get(14)?,
                    )
                } else {
                    (None, None)
                };
                Ok(DuckLakeTableFile {
                    data_file_id: row.get(0)?,
                    file: DuckLakeFileData {
                        path: row.get(1)?,
                        path_is_relative: row.get(2)?,
                        file_size_bytes: row.get(3)?,
                        footer_size: row.get(4)?,
                        encryption_key: row.get(5)?,
                        mapping_id: row.get(15)?,
                    },
                    delete_file_id,
                    delete_file,
                    row_id_start: row.get(6)?,
                    snapshot_id: Some(snapshot_id),
                    begin_snapshot: None,
                    schema_version: None,
                    partial_max: None,
                    max_row_count: row.get(7)?,
                    delete_count,
                    partition_id: None,
                    partition_values: Vec::new(),
                })
            },
        )?
        .collect()
}

fn convert_time(value: i64, from: DuckdbTimeUnit, to: TimeUnit) -> Option<i64> {
    let from_nanos: i128 = match from {
        DuckdbTimeUnit::Second => 1_000_000_000,
        DuckdbTimeUnit::Millisecond => 1_000_000,
        DuckdbTimeUnit::Microsecond => 1_000,
        DuckdbTimeUnit::Nanosecond => 1,
    };
    let to_nanos: i128 = match to {
        TimeUnit::Second => 1_000_000_000,
        TimeUnit::Millisecond => 1_000_000,
        TimeUnit::Microsecond => 1_000,
        TimeUnit::Nanosecond => 1,
    };
    i64::try_from(i128::from(value) * from_nanos / to_nanos).ok()
}

fn duckdb_inlined_scalar(
    value: ValueRef<'_>,
    data_type: &DataType,
    column: &str,
) -> crate::Result<ScalarValue> {
    if matches!(value, ValueRef::Null) {
        return Ok(ScalarValue::try_from(data_type)?);
    }
    let scalar = match (data_type, value) {
        (DataType::Boolean, ValueRef::Boolean(value)) => ScalarValue::Boolean(Some(value)),
        (DataType::Int8, ValueRef::TinyInt(value)) => ScalarValue::Int8(Some(value)),
        (DataType::Int16, ValueRef::SmallInt(value)) => ScalarValue::Int16(Some(value)),
        (DataType::Int32, ValueRef::Int(value)) => ScalarValue::Int32(Some(value)),
        (DataType::Int64, ValueRef::BigInt(value)) => ScalarValue::Int64(Some(value)),
        (DataType::UInt8, ValueRef::UTinyInt(value)) => ScalarValue::UInt8(Some(value)),
        (DataType::UInt16, ValueRef::USmallInt(value)) => ScalarValue::UInt16(Some(value)),
        (DataType::UInt32, ValueRef::UInt(value)) => ScalarValue::UInt32(Some(value)),
        (DataType::UInt64, ValueRef::UBigInt(value)) => ScalarValue::UInt64(Some(value)),
        (DataType::Float32, ValueRef::Float(value)) => ScalarValue::Float32(Some(value)),
        (DataType::Float64, ValueRef::Double(value)) => ScalarValue::Float64(Some(value)),
        (DataType::Decimal128(_, _), ValueRef::Decimal(value)) => {
            crate::types::parse_ducklake_scalar(&value.to_string(), data_type).ok_or_else(|| {
                crate::DuckLakeError::Unsupported(format!(
                    "inlined data for column '{column}' cannot decode decimal '{value}' as {data_type}"
                ))
            })?
        },
        (DataType::Date32, ValueRef::Date32(value)) => ScalarValue::Date32(Some(value)),
        (DataType::Time64(to), ValueRef::Time64(from, value)) => {
            let value = convert_time(value, from, *to).ok_or_else(|| {
                crate::DuckLakeError::Unsupported(format!(
                    "inlined data for column '{column}' has an out-of-range time value"
                ))
            })?;
            match to {
                TimeUnit::Microsecond => ScalarValue::Time64Microsecond(Some(value)),
                TimeUnit::Nanosecond => ScalarValue::Time64Nanosecond(Some(value)),
                _ => {
                    return Err(crate::DuckLakeError::Unsupported(format!(
                        "inlined data column '{column}' has unsupported time unit {to:?}"
                    )));
                },
            }
        },
        (DataType::Timestamp(to, timezone), ValueRef::Timestamp(from, value)) => {
            let value = convert_time(value, from, *to).ok_or_else(|| {
                crate::DuckLakeError::Unsupported(format!(
                    "inlined data for column '{column}' has an out-of-range timestamp value"
                ))
            })?;
            match to {
                TimeUnit::Second => ScalarValue::TimestampSecond(Some(value), timezone.clone()),
                TimeUnit::Millisecond => {
                    ScalarValue::TimestampMillisecond(Some(value), timezone.clone())
                },
                TimeUnit::Microsecond => {
                    ScalarValue::TimestampMicrosecond(Some(value), timezone.clone())
                },
                TimeUnit::Nanosecond => {
                    ScalarValue::TimestampNanosecond(Some(value), timezone.clone())
                },
            }
        },
        (DataType::Interval(_), ValueRef::Interval { months, days, nanos }) => {
            ScalarValue::new_interval_mdn(months, days, nanos)
        },
        (DataType::Utf8, ValueRef::Text(value)) => {
            ScalarValue::Utf8(Some(decode_duckdb_text(value, column)?))
        },
        (DataType::LargeUtf8, ValueRef::Text(value)) => {
            ScalarValue::LargeUtf8(Some(decode_duckdb_text(value, column)?))
        },
        (DataType::Utf8View, ValueRef::Text(value)) => {
            ScalarValue::Utf8View(Some(decode_duckdb_text(value, column)?))
        },
        (DataType::Binary, ValueRef::Blob(value)) => {
            ScalarValue::Binary(Some(value.to_vec()))
        },
        (DataType::LargeBinary, ValueRef::Blob(value)) => {
            ScalarValue::LargeBinary(Some(value.to_vec()))
        },
        (DataType::BinaryView, ValueRef::Blob(value)) => {
            ScalarValue::BinaryView(Some(value.to_vec()))
        },
        (DataType::FixedSizeBinary(size), ValueRef::Text(value)) => {
            let value = decode_duckdb_text(value, column)?;
            crate::types::parse_ducklake_scalar(&value, data_type).ok_or_else(|| {
                crate::DuckLakeError::Unsupported(format!(
                    "inlined data for column '{column}' cannot decode '{value}' as fixed-size binary {size}"
                ))
                })?
        },
        (DataType::FixedSizeBinary(size), ValueRef::Blob(value))
            if value.len() == *size as usize =>
        {
            ScalarValue::FixedSizeBinary(*size, Some(value.to_vec()))
        },
        (DataType::List(_) | DataType::LargeList(_), value @ ValueRef::List(_, _))
        | (DataType::FixedSizeList(_, _), value @ ValueRef::Array(_, _))
        | (DataType::Struct(_), value @ ValueRef::Struct(_, _))
        | (DataType::Map(_, _), value @ ValueRef::Map(_, _)) => {
            duckdb_owned_scalar(Value::from(value), data_type, column)?
        }
        (data_type, value) => {
            return Err(crate::DuckLakeError::Unsupported(format!(
                "inlined data for column '{column}' has DuckDB type {:?}, which cannot be decoded \
                 as {data_type}; {INLINED_DATA_REMEDIATION}",
                value.data_type(),
            )));
        },
    };
    Ok(scalar)
}

fn duckdb_owned_scalar(
    value: Value,
    data_type: &DataType,
    column: &str,
) -> crate::Result<ScalarValue> {
    // duckdb-rs currently returns nested values through its Arrow 58 types, while this crate uses
    // Arrow 59. Remove this owned-value bridge once duckdb-rs upgrades to Arrow 59 and its nested
    // arrays can be passed directly to ScalarValue::try_from_array.
    if matches!(value, Value::Null) {
        return Ok(ScalarValue::try_from(data_type)?);
    }

    match (data_type, value) {
        (DataType::List(field) | DataType::LargeList(field), Value::List(values))
        | (DataType::FixedSizeList(field, _), Value::List(values))
        | (DataType::FixedSizeList(field, _), Value::Array(values)) => {
            let values = values
                .into_iter()
                .map(|value| duckdb_owned_scalar(value, field.data_type(), column))
                .collect::<crate::Result<Vec<_>>>()?;
            crate::nested_inline::build_list_scalar(data_type, values).ok_or_else(|| {
                crate::DuckLakeError::Unsupported(format!(
                    "inlined data column '{column}' cannot rebuild {data_type} from DuckDB list"
                ))
            })
        },
        (DataType::Struct(fields), Value::Struct(values)) => {
            let values = fields
                .iter()
                .map(|field| {
                    let value = values.get(field.name()).cloned().ok_or_else(|| {
                        crate::DuckLakeError::Unsupported(format!(
                            "inlined data column '{column}' DuckDB struct is missing field '{}'",
                            field.name()
                        ))
                    })?;
                    duckdb_owned_scalar(value, field.data_type(), column)
                })
                .collect::<crate::Result<Vec<_>>>()?;
            crate::nested_inline::build_struct_scalar(data_type, values).ok_or_else(|| {
                crate::DuckLakeError::Unsupported(format!(
                    "inlined data column '{column}' cannot rebuild {data_type} from DuckDB struct"
                ))
            })
        },
        (DataType::Map(entries, _), Value::Map(values)) => {
            let DataType::Struct(fields) = entries.data_type() else {
                return Err(crate::DuckLakeError::Unsupported(format!(
                    "inlined data column '{column}' has invalid Arrow map entries"
                )));
            };
            let length = values.iter().count();
            let mut keys = Vec::with_capacity(length);
            let mut mapped_values = Vec::with_capacity(length);
            for (key, value) in values.iter() {
                keys.push(duckdb_owned_scalar(
                    key.clone(),
                    fields[0].data_type(),
                    column,
                )?);
                mapped_values.push(duckdb_owned_scalar(
                    value.clone(),
                    fields[1].data_type(),
                    column,
                )?);
            }
            crate::nested_inline::build_map_scalar(data_type, keys, mapped_values).ok_or_else(
                || {
                    crate::DuckLakeError::Unsupported(format!(
                        "inlined data column '{column}' cannot rebuild {data_type} from DuckDB map"
                    ))
                },
            )
        },
        (DataType::Boolean, Value::Boolean(value)) => Ok(ScalarValue::Boolean(Some(value))),
        (DataType::Int8, Value::TinyInt(value)) => Ok(ScalarValue::Int8(Some(value))),
        (DataType::Int16, Value::SmallInt(value)) => Ok(ScalarValue::Int16(Some(value))),
        (DataType::Int32, Value::Int(value)) => Ok(ScalarValue::Int32(Some(value))),
        (DataType::Int64, Value::BigInt(value)) => Ok(ScalarValue::Int64(Some(value))),
        (DataType::UInt8, Value::UTinyInt(value)) => Ok(ScalarValue::UInt8(Some(value))),
        (DataType::UInt16, Value::USmallInt(value)) => Ok(ScalarValue::UInt16(Some(value))),
        (DataType::UInt32, Value::UInt(value)) => Ok(ScalarValue::UInt32(Some(value))),
        (DataType::UInt64, Value::UBigInt(value)) => Ok(ScalarValue::UInt64(Some(value))),
        (DataType::Float32, Value::Float(value)) => Ok(ScalarValue::Float32(Some(value))),
        (DataType::Float64, Value::Double(value)) => Ok(ScalarValue::Float64(Some(value))),
        (DataType::Decimal128(_, _), Value::Decimal(value)) => {
            crate::types::parse_ducklake_scalar_leaf(&value.to_string(), data_type).ok_or_else(
                || {
                    crate::DuckLakeError::Unsupported(format!(
                        "inlined data column '{column}' cannot decode nested decimal '{value}'"
                    ))
                },
            )
        },
        (DataType::Date32, Value::Date32(value)) => Ok(ScalarValue::Date32(Some(value))),
        (DataType::Time64(to), Value::Time64(from, value)) => {
            let value = convert_time(value, from, *to).ok_or_else(|| {
                crate::DuckLakeError::Unsupported(format!(
                    "inlined data column '{column}' has out-of-range nested time"
                ))
            })?;
            match to {
                TimeUnit::Microsecond => Ok(ScalarValue::Time64Microsecond(Some(value))),
                TimeUnit::Nanosecond => Ok(ScalarValue::Time64Nanosecond(Some(value))),
                _ => Err(crate::DuckLakeError::Unsupported(format!(
                    "inlined data column '{column}' has unsupported nested time unit {to:?}"
                ))),
            }
        },
        (DataType::Timestamp(to, timezone), Value::Timestamp(from, value)) => {
            let value = convert_time(value, from, *to).ok_or_else(|| {
                crate::DuckLakeError::Unsupported(format!(
                    "inlined data column '{column}' has out-of-range nested timestamp"
                ))
            })?;
            Ok(match to {
                TimeUnit::Second => ScalarValue::TimestampSecond(Some(value), timezone.clone()),
                TimeUnit::Millisecond => {
                    ScalarValue::TimestampMillisecond(Some(value), timezone.clone())
                },
                TimeUnit::Microsecond => {
                    ScalarValue::TimestampMicrosecond(Some(value), timezone.clone())
                },
                TimeUnit::Nanosecond => {
                    ScalarValue::TimestampNanosecond(Some(value), timezone.clone())
                },
            })
        },
        (
            DataType::Interval(_),
            Value::Interval {
                months,
                days,
                nanos,
            },
        ) => Ok(ScalarValue::new_interval_mdn(months, days, nanos)),
        (DataType::Utf8, Value::Text(value)) => Ok(ScalarValue::Utf8(Some(value))),
        (DataType::LargeUtf8, Value::Text(value)) => Ok(ScalarValue::LargeUtf8(Some(value))),
        (DataType::Utf8View, Value::Text(value)) => Ok(ScalarValue::Utf8View(Some(value))),
        (DataType::Binary, Value::Blob(value)) => Ok(ScalarValue::Binary(Some(value))),
        (DataType::LargeBinary, Value::Blob(value)) => Ok(ScalarValue::LargeBinary(Some(value))),
        (DataType::BinaryView, Value::Blob(value)) => Ok(ScalarValue::BinaryView(Some(value))),
        (DataType::FixedSizeBinary(size), Value::Blob(value)) if value.len() == *size as usize => {
            Ok(ScalarValue::FixedSizeBinary(*size, Some(value)))
        },
        (data_type, value) => Err(crate::DuckLakeError::Unsupported(format!(
            "inlined data column '{column}' DuckDB nested value {value:?} cannot decode as {data_type}"
        ))),
    }
}

fn decode_duckdb_text(value: &[u8], column: &str) -> crate::Result<String> {
    std::str::from_utf8(value).map(str::to_owned).map_err(|e| {
        crate::DuckLakeError::Unsupported(format!(
            "inlined data for column '{column}' contains invalid UTF-8: {e}"
        ))
    })
}

fn decode_view(row: &duckdb::Row<'_>) -> duckdb::Result<ViewMetadata> {
    Ok(ViewMetadata {
        view_id: row.get(0)?,
        schema_id: row.get(1)?,
        begin_snapshot: row.get(2)?,
        view_name: row.get(3)?,
        dialect: row.get(4)?,
        sql: row.get(5)?,
        column_aliases: row.get(6)?,
    })
}

/// Optional catalog-schema capabilities probed before version-dependent queries.
///
/// Older catalogs (spec 0.2) may lack the `partial_max` columns and the
/// inlined-data registry. CDC queries fall back to the old-spec
/// `partial_file_info` string (data files) or degrade the predicate to NULL
/// (delete files); inlined-data reads return empty when a capability is absent.
/// Older catalogs may also lack any or all of the four default-value columns.
#[derive(Debug, Clone, Copy)]
struct SchemaCapabilities {
    /// `ducklake_data_file.partial_max` exists.
    data_file_partial_max: bool,
    /// `ducklake_delete_file.partial_max` exists.
    delete_file_partial_max: bool,
    /// The `ducklake_inlined_data_tables` registry exists.
    inlined_data_tables: bool,
    /// The `ducklake_view` table exists.
    views: bool,
    /// `ducklake_column.initial_default` exists.
    column_initial_default: bool,
    /// `ducklake_column.default_value` exists.
    column_default_value: bool,
    /// `ducklake_column.default_value_type` exists.
    column_default_value_type: bool,
    /// `ducklake_column.default_value_dialect` exists.
    column_default_value_dialect: bool,
}

impl SchemaCapabilities {
    fn all(&self) -> bool {
        self.data_file_partial_max
            && self.delete_file_partial_max
            && self.inlined_data_tables
            && self.views
            && self.column_initial_default
            && self.column_default_value
            && self.column_default_value_type
            && self.column_default_value_dialect
    }
}

fn get_table_columns_sql(capabilities: SchemaCapabilities) -> String {
    let initial_default = if capabilities.column_initial_default {
        "initial_default"
    } else {
        "NULL AS initial_default"
    };
    let default_value = if capabilities.column_default_value {
        "default_value"
    } else {
        "NULL AS default_value"
    };
    let value_type = if capabilities.column_default_value_type {
        "default_value_type"
    } else {
        "NULL AS default_value_type"
    };
    let dialect = if capabilities.column_default_value_dialect {
        "default_value_dialect"
    } else {
        "NULL AS default_value_dialect"
    };
    format!(
        "SELECT column_id, column_name, column_type, nulls_allowed, parent_column,
                {initial_default}, {default_value}, {value_type}, {dialect}
         FROM ducklake_column
         WHERE table_id = ?
           AND ? >= begin_snapshot
           AND (? < end_snapshot OR end_snapshot IS NULL)
         ORDER BY column_order"
    )
}

fn list_all_columns_sql(capabilities: SchemaCapabilities) -> String {
    let initial_default = if capabilities.column_initial_default {
        "c.initial_default"
    } else {
        "NULL AS initial_default"
    };
    let default_value = if capabilities.column_default_value {
        "c.default_value"
    } else {
        "NULL AS default_value"
    };
    let value_type = if capabilities.column_default_value_type {
        "c.default_value_type"
    } else {
        "NULL AS default_value_type"
    };
    let dialect = if capabilities.column_default_value_dialect {
        "c.default_value_dialect"
    } else {
        "NULL AS default_value_dialect"
    };
    format!(
        "SELECT
            s.schema_name,
            t.table_name,
            c.column_id,
            c.column_name,
            c.column_type,
            c.nulls_allowed,
            c.parent_column,
            {initial_default},
            {default_value},
            {value_type},
            {dialect}
         FROM ducklake_schema s
         JOIN ducklake_table t ON s.schema_id = t.schema_id
         JOIN ducklake_column c ON t.table_id = c.table_id
         WHERE ? >= s.begin_snapshot
           AND (? < s.end_snapshot OR s.end_snapshot IS NULL)
           AND ? >= t.begin_snapshot
           AND (? < t.end_snapshot OR t.end_snapshot IS NULL)
           AND ? >= c.begin_snapshot
           AND (? < c.end_snapshot OR c.end_snapshot IS NULL)
         ORDER BY s.schema_name, t.table_name, c.column_order"
    )
}

/// DuckDB metadata provider
///
/// Uses a single shared connection protected by a Mutex to avoid
/// the overhead of creating a new connection for each metadata query.
/// This is safe for read-only operations.
#[derive(Debug, Clone)]
pub struct DuckdbMetadataProvider {
    conn: Arc<Mutex<Connection>>,
    /// Path to the catalog database, retained for logging/debugging
    #[allow(dead_code)]
    catalog_path: String,
    /// Positive-only memo of the optional-schema capability probes. `Arc` so
    /// derived `Clone` shares the cache across provider clones.
    schema_capabilities: Arc<OnceLock<SchemaCapabilities>>,
}

impl DuckdbMetadataProvider {
    /// Create a new DuckDB metadata provider
    pub fn new(catalog_path: impl Into<String>) -> crate::Result<Self> {
        let catalog_path = catalog_path.into();
        let conn = Self::create_connection(&catalog_path)?;

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            catalog_path,
            schema_capabilities: Arc::new(OnceLock::new()),
        })
    }

    pub(crate) fn from_shared_connection(
        conn: Arc<Mutex<Connection>>,
        catalog_path: String,
    ) -> Self {
        Self {
            conn,
            catalog_path,
            schema_capabilities: Arc::new(OnceLock::new()),
        }
    }

    /// Get a reference to the shared connection
    fn connection(&self) -> MutexGuard<'_, Connection> {
        self.conn.lock().expect("DuckDB connection mutex poisoned")
    }

    /// Whether the schema-capability memo is populated. Exposed for tests.
    #[doc(hidden)]
    pub fn schema_capabilities_cached(&self) -> bool {
        self.schema_capabilities.get().is_some()
    }

    /// Returns the catalog's optional-schema capabilities, probing at most
    /// once per provider lifetime on a fully-migrated catalog. Takes the
    /// caller's already-locked `&Connection` (the shared mutex is not
    /// reentrant).
    ///
    /// Cache-positive-only: capability existence is monotonic (migrations only
    /// add columns/tables, never drop them), so an all-`true` answer is an
    /// immutable fact and safe to memoize. A `false` answer is never cached —
    /// the next call re-probes, so a mid-flight catalog upgrade is picked up
    /// on the next call exactly like the previous per-call probing. Concurrent
    /// first calls may each probe once (one statement each) — harmless; a
    /// raced `set` is ignored.
    fn schema_capabilities(&self, conn: &Connection) -> crate::Result<SchemaCapabilities> {
        if let Some(caps) = self.schema_capabilities.get() {
            return Ok(*caps);
        }
        let (
            data_file_partial_max,
            delete_file_partial_max,
            inlined_data_tables,
            views,
            column_initial_default,
            column_default_value,
            column_default_value_type,
            column_default_value_dialect,
        ): (bool, bool, bool, bool, bool, bool, bool, bool) = conn.query_row(
            "SELECT
               (SELECT COUNT(*) FROM pragma_table_info('ducklake_data_file')
                WHERE name = 'partial_max') > 0,
               (SELECT COUNT(*) FROM pragma_table_info('ducklake_delete_file')
                WHERE name = 'partial_max') > 0,
               (SELECT COUNT(*) FROM information_schema.tables
                WHERE table_name = 'ducklake_inlined_data_tables') > 0,
               (SELECT COUNT(*) FROM information_schema.tables
                WHERE table_name = 'ducklake_view') > 0,
                (SELECT COUNT(*) FROM pragma_table_info('ducklake_column')
                 WHERE name = 'initial_default') > 0,
               (SELECT COUNT(*) FROM pragma_table_info('ducklake_column')
                WHERE name = 'default_value') > 0,
               (SELECT COUNT(*) FROM pragma_table_info('ducklake_column')
                WHERE name = 'default_value_type') > 0,
               (SELECT COUNT(*) FROM pragma_table_info('ducklake_column')
                WHERE name = 'default_value_dialect') > 0",
            [],
            |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                    row.get(5)?,
                    row.get(6)?,
                    row.get(7)?,
                ))
            },
        )?;
        let caps = SchemaCapabilities {
            data_file_partial_max,
            delete_file_partial_max,
            inlined_data_tables,
            views,
            column_initial_default,
            column_default_value,
            column_default_value_type,
            column_default_value_dialect,
        };
        if caps.all() {
            let _ = self.schema_capabilities.set(caps);
        }
        Ok(caps)
    }

    /// Create a new read-only connection to the catalog database
    fn create_connection(catalog_path: &str) -> crate::Result<Connection> {
        let config = Config::default().access_mode(ReadOnly)?;
        match Connection::open_with_flags(catalog_path, config) {
            Ok(con) => Ok(con),
            Err(msg)
                if msg
                    .to_string()
                    .starts_with("IO Error: Could not set lock on file") =>
            {
                tracing::warn!(
                    error = %msg,
                    "DuckDB file likely already open in write mode. Cannot connect"
                );
                Err(DuckLakeError::DuckDb(msg))
            },
            Err(msg) => {
                tracing::error!(error = %msg, "Failed to open DuckDB catalog");
                Err(DuckLakeError::DuckDb(msg))
            },
        }
    }
}

impl MetadataProvider for DuckdbMetadataProvider {
    fn get_current_snapshot(&self) -> crate::Result<i64> {
        let conn = self.connection();
        let snapshot_id: i64 = conn.query_row(SQL_GET_LATEST_SNAPSHOT, [], |row| row.get(0))?;
        Ok(snapshot_id)
    }

    fn get_data_path(&self) -> crate::Result<String> {
        self.get_metadata_settings(None, None)?
            .remove("data_path")
            .ok_or_else(|| {
                DuckLakeError::InvalidConfig(
                    "Missing required catalog metadata: 'data_path' not configured. \
                     The catalog may be uninitialized or corrupted."
                        .to_string(),
                )
            })
    }

    fn get_metadata_settings(
        &self,
        schema_id: Option<i64>,
        table_id: Option<i64>,
    ) -> crate::Result<HashMap<String, String>> {
        let conn = self.connection();
        let has_scope_columns: bool = conn.query_row(
            "SELECT COUNT(*) = 2 FROM pragma_table_info('ducklake_metadata') \
             WHERE name IN ('scope', 'scope_id')",
            [],
            |row| row.get(0),
        )?;
        let rows = if has_scope_columns {
            let mut stmt =
                conn.prepare("SELECT key, value, scope, scope_id FROM ducklake_metadata")?;
            stmt.query_map([], |row| {
                Ok(MetadataSetting {
                    key: row.get(0)?,
                    value: row.get(1)?,
                    scope: row.get(2)?,
                    scope_id: row.get(3)?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?
        } else {
            let mut stmt = conn.prepare("SELECT key, value FROM ducklake_metadata")?;
            stmt.query_map([], |row| {
                Ok(MetadataSetting {
                    key: row.get(0)?,
                    value: row.get(1)?,
                    scope: None,
                    scope_id: None,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?
        };
        resolve_metadata_settings(rows, schema_id, table_id)
    }

    fn list_snapshots(&self) -> crate::Result<Vec<SnapshotMetadata>> {
        let conn = self.connection();
        let mut stmt = conn.prepare(SQL_LIST_SNAPSHOTS)?;

        let snapshots = stmt
            .query_map([], |row| {
                let snapshot_id: i64 = row.get(0)?;
                let timestamp: Option<String> = row.get(1)?;
                Ok(SnapshotMetadata {
                    snapshot_id,
                    timestamp,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(snapshots)
    }

    fn list_snapshot_changes(&self) -> crate::Result<Vec<SnapshotChangeMetadata>> {
        let conn = self.connection();
        let mut statement = conn.prepare(
            "SELECT snapshot.snapshot_id,
                    CAST(snapshot.snapshot_time AS VARCHAR),
                    changes.changes_made,
                    changes.author,
                    changes.commit_message,
                    changes.commit_extra_info
             FROM ducklake_snapshot AS snapshot
             JOIN ducklake_snapshot_changes AS changes
               ON changes.snapshot_id = snapshot.snapshot_id
             ORDER BY snapshot.snapshot_id",
        )?;
        let changes = statement
            .query_map([], |row| {
                Ok(SnapshotChangeMetadata {
                    snapshot_id: row.get(0)?,
                    timestamp: row.get(1)?,
                    changes_made: row.get(2)?,
                    author: row.get(3)?,
                    commit_message: row.get(4)?,
                    commit_extra_info: row.get(5)?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(changes)
    }

    fn find_snapshot_by_commit_extra_info(&self, needle: &str) -> crate::Result<Option<i64>> {
        let conn = self.connection();
        let snapshot_id = conn
            .query_row(
                "SELECT changes.snapshot_id
                 FROM ducklake_snapshot_changes AS changes
                 WHERE (changes.commit_extra_info = ?
                        OR strpos(changes.commit_extra_info, ?) > 0)
                   AND EXISTS (
                       SELECT 1
                       FROM ducklake_data_file AS files
                       WHERE files.begin_snapshot = changes.snapshot_id
                         AND files.end_snapshot IS NULL
                   )
                 ORDER BY changes.snapshot_id
                 LIMIT 1",
                params![needle, needle],
                |row| row.get(0),
            )
            .optional()?;
        Ok(snapshot_id)
    }

    fn list_schemas(&self, snapshot_id: i64) -> crate::Result<Vec<SchemaMetadata>> {
        let conn = self.connection();
        let mut stmt = conn.prepare(SQL_LIST_SCHEMAS)?;

        let schemas = stmt
            .query_map([snapshot_id, snapshot_id], |row| {
                let schema_id: i64 = row.get(0)?;
                let schema_name: String = row.get(1)?;
                let path: String = row.get(2)?;
                let path_is_relative: bool = row.get(3)?;
                Ok(SchemaMetadata {
                    schema_id,
                    schema_name,
                    path,
                    path_is_relative,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(schemas)
    }

    fn list_tables(&self, schema_id: i64, snapshot_id: i64) -> crate::Result<Vec<TableMetadata>> {
        let conn = self.connection();
        let mut stmt = conn.prepare(SQL_LIST_TABLES)?;

        let tables = stmt
            .query_map([schema_id, snapshot_id, snapshot_id], |row| {
                let table_id: i64 = row.get(0)?;
                let table_name: String = row.get(1)?;
                let path: String = row.get(2)?;
                let path_is_relative: bool = row.get(3)?;
                Ok(TableMetadata {
                    table_id,
                    table_name,
                    path,
                    path_is_relative,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(tables)
    }

    fn list_views(&self, schema_id: i64, snapshot_id: i64) -> crate::Result<Vec<ViewMetadata>> {
        let conn = self.connection();
        if !self.schema_capabilities(&conn)?.views {
            return Ok(Vec::new());
        }
        let mut stmt = conn.prepare(SQL_LIST_VIEWS)?;
        let views = stmt
            .query_map([schema_id, snapshot_id, snapshot_id], decode_view)?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(views)
    }

    fn get_table_structure(
        &self,
        table_id: i64,
        snapshot_id: i64,
    ) -> crate::Result<Vec<DuckLakeTableColumn>> {
        let conn = self.connection();
        let sql = get_table_columns_sql(self.schema_capabilities(&conn)?);
        let mut stmt = conn.prepare(&sql)?;

        let raw_columns: Vec<(DuckLakeTableColumn, Option<i64>)> = stmt
            .query_map(duckdb::params![table_id, snapshot_id, snapshot_id], |row| {
                let column_id: i64 = row.get(0)?;
                let column_name: String = row.get(1)?;
                let column_type: String = row.get(2)?;
                let nulls_allowed: Option<bool> = row.get(3)?;
                let parent_column: Option<i64> = row.get(4)?;
                Ok((
                    DuckLakeTableColumn::new(
                        column_id,
                        column_name,
                        column_type,
                        nulls_allowed.unwrap_or(true),
                    )
                    .with_defaults(
                        row.get(5)?,
                        row.get(6)?,
                        row.get(7)?,
                        row.get(8)?,
                    ),
                    parent_column,
                ))
            })?
            .collect::<Result<Vec<_>, _>>()?;

        reconstruct_columns(raw_columns)
    }

    fn get_table_fields(
        &self,
        table_id: i64,
        snapshot_id: i64,
    ) -> crate::Result<Vec<DuckLakeTableField>> {
        let conn = self.connection();
        let sql = get_table_columns_sql(self.schema_capabilities(&conn)?);
        let mut stmt = conn.prepare(&sql)?;
        Ok(stmt
            .query_map(params![table_id, snapshot_id, snapshot_id], |row| {
                Ok(DuckLakeTableField {
                    column_id: row.get(0)?,
                    column_name: row.get(1)?,
                    column_type: row.get(2)?,
                    is_nullable: row.get::<_, Option<bool>>(3)?.unwrap_or(true),
                    parent_column: row.get(4)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?)
    }

    fn get_name_mapping(&self, mapping_id: i64) -> crate::Result<DuckLakeNameMapping> {
        let conn = self.connection();
        let mut stmt = conn.prepare(SQL_GET_NAME_MAPPING)?;
        let mut rows = stmt.query(params![mapping_id])?;
        let mut header = None;
        let mut entries = Vec::new();
        while let Some(row) = rows.next()? {
            header.get_or_insert((
                row.get::<_, i64>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, String>(2)?,
            ));
            if let Some(column_id) = row.get::<_, Option<i64>>(3)? {
                entries.push(DuckLakeNameMappingEntry {
                    column_id,
                    source_name: row.get(4)?,
                    target_field_id: row.get(5)?,
                    parent_column: row.get(6)?,
                    is_partition: row.get::<_, Option<bool>>(7)?.unwrap_or(false),
                });
            }
        }
        let (mapping_id, table_id, mapping_type) = header.ok_or_else(|| {
            crate::DuckLakeError::InvalidConfig(format!(
                "DuckLake name mapping {mapping_id} does not exist"
            ))
        })?;
        Ok(DuckLakeNameMapping {
            mapping_id,
            table_id,
            mapping_type,
            entries,
        })
    }

    fn get_table_files_for_select(
        &self,
        table_id: i64,
        snapshot_id: i64,
    ) -> crate::Result<Vec<DuckLakeTableFile>> {
        let conn = self.connection();
        let mut stmt = conn.prepare(SQL_GET_DATA_FILES)?;

        let files = stmt
            .query_map(
                [table_id, snapshot_id, snapshot_id, table_id, snapshot_id, snapshot_id],
                |row| {
                    // Parse data file (columns 0-7)
                    let data_file_id: i64 = row.get(0)?;
                    let data_file = DuckLakeFileData {
                        path: row.get(1)?,
                        path_is_relative: row.get(2)?,
                        file_size_bytes: row.get(3)?,
                        footer_size: row.get(4)?,
                        encryption_key: row.get(5)?,
                        mapping_id: row.get(15)?,
                    };
                    let row_id_start: Option<i64> = row.get(6)?;
                    let record_count: Option<i64> = row.get(7)?;

                    // Parse delete file (columns 8-14) if exists
                    let (delete_file, delete_count, delete_file_id) =
                        if let Ok(Some(dfid)) = row.get::<_, Option<i64>>(8) {
                            (
                                Some(DuckLakeFileData {
                                    path: row.get(9)?,
                                    path_is_relative: row.get(10)?,
                                    file_size_bytes: row.get(11)?,
                                    footer_size: row.get(12)?,
                                    encryption_key: row.get(13)?,
                                    mapping_id: None,
                                }),
                                row.get(14)?,
                                Some(dfid),
                            )
                        } else {
                            (None, None, None)
                        };

                    Ok(DuckLakeTableFile {
                        data_file_id,
                        file: data_file,
                        delete_file_id,
                        delete_file,
                        row_id_start,
                        snapshot_id: Some(snapshot_id),
                        begin_snapshot: None,
                        schema_version: None,
                        partial_max: None,
                        max_row_count: record_count,
                        delete_count,
                        partition_id: None,
                        partition_values: Vec::new(),
                    })
                },
            )?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(files)
    }

    fn get_partition_spec(
        &self,
        table_id: i64,
        snapshot_id: i64,
    ) -> crate::Result<Option<PartitionSpec>> {
        let conn = self.connection();
        // Pruning is only safe with exactly one spec generation ever (the common
        // "set once" case); after a re-partition a live file may carry values under
        // a retired generation whose key order differs (see PartitionSpec::prune_safe).
        // The live spec is returned regardless so the write path always targets the
        // current generation.
        let generation_count: i64 = match conn.query_row(
            "SELECT COUNT(*) FROM ducklake_partition_info WHERE table_id = ?",
            params![table_id],
            |row| row.get(0),
        ) {
            Ok(count) => count,
            Err(error) if is_missing_statistics_table(&error) => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        let prune_safe = generation_count == 1;
        let rows = match conn.prepare(SQL_GET_PARTITION_SPEC) {
            Ok(mut stmt) => stmt
                .query_map(params![table_id, snapshot_id, snapshot_id], |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        i32::try_from(row.get::<_, i64>(1)?).unwrap_or(0),
                        row.get::<_, i64>(2)?,
                        row.get::<_, String>(3)?,
                    ))
                })?
                .collect::<Result<Vec<_>, _>>()?,
            Err(error) if is_missing_statistics_table(&error) => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        Ok(PartitionSpec::from_rows(rows, prune_safe))
    }

    fn get_sort_spec(&self, table_id: i64, snapshot_id: i64) -> crate::Result<Option<SortSpec>> {
        let conn = self.connection();
        let rows = match conn.prepare(SQL_GET_SORT_SPEC) {
            Ok(mut stmt) => stmt
                .query_map(params![table_id, snapshot_id, snapshot_id], |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        i32::try_from(row.get::<_, i64>(1)?).unwrap_or(0),
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                        row.get::<_, String>(4)?,
                        row.get::<_, String>(5)?,
                    ))
                })?
                .collect::<Result<Vec<_>, _>>()?,
            Err(error) if is_missing_statistics_table(&error) => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        Ok(SortSpec::from_rows(rows))
    }

    fn get_table_summary_statistics(
        &self,
        table_id: i64,
        snapshot_id: i64,
    ) -> crate::Result<DuckLakeStatistics> {
        let conn = self.connection();
        let table = match conn.prepare(SQL_GET_TABLE_STATS) {
            Ok(mut stmt) => {
                let mut rows = stmt.query([table_id])?;
                rows.next()?
                    .map(|row| {
                        Ok::<_, duckdb::Error>(DuckLakeTableStatistics {
                            record_count: row.get(0)?,
                            file_size_bytes: row.get(1)?,
                        })
                    })
                    .transpose()?
            },
            Err(error) if is_missing_statistics_table(&error) => None,
            Err(error) => return Err(error.into()),
        };
        let column_sizes: HashMap<i64, i64> = match conn.prepare(
            "SELECT stats.column_id,
                    CASE
                      WHEN COUNT(*) = COUNT(stats.column_size_bytes)
                       AND COUNT(*) = (
                         SELECT COUNT(*) FROM ducklake_data_file visible
                         WHERE visible.table_id = ?
                           AND ? >= visible.begin_snapshot
                           AND (? < visible.end_snapshot OR visible.end_snapshot IS NULL)
                       )
                      THEN CAST(SUM(stats.column_size_bytes) AS BIGINT)
                    END
             FROM ducklake_file_column_stats stats
             INNER JOIN ducklake_data_file data
               ON data.data_file_id = stats.data_file_id
              AND data.table_id = stats.table_id
             WHERE stats.table_id = ?
               AND ? >= data.begin_snapshot
               AND (? < data.end_snapshot OR data.end_snapshot IS NULL)
             GROUP BY stats.column_id",
        ) {
            Ok(mut stmt) => stmt
                .query_map(
                    params![table_id, snapshot_id, snapshot_id, table_id, snapshot_id, snapshot_id],
                    |row| Ok((row.get::<_, i64>(0)?, row.get::<_, Option<i64>>(1)?)),
                )?
                .filter_map(|row| match row {
                    Ok((column_id, Some(size))) => Some(Ok((column_id, size))),
                    Ok((_, None)) => None,
                    Err(error) => Some(Err(error)),
                })
                .collect::<Result<_, _>>()?,
            Err(error) if is_missing_statistics_table(&error) => HashMap::new(),
            Err(error) => return Err(error.into()),
        };
        let bounds_are_exact: bool = conn.query_row(
            "SELECT NOT EXISTS (
                 SELECT 1 FROM ducklake_delete_file
                 WHERE table_id = ?
                   AND ? >= begin_snapshot
                   AND (? < end_snapshot OR end_snapshot IS NULL)
             )",
            params![table_id, snapshot_id, snapshot_id],
            |row| row.get(0),
        )?;
        let columns = match conn.prepare(SQL_GET_TABLE_COLUMN_STATS) {
            Ok(mut stmt) => stmt
                .query_map([table_id], |row| {
                    let column_id = row.get(0)?;
                    Ok(DuckLakeTableColumnStatistics {
                        column_id,
                        contains_null: row.get(1)?,
                        min_value: row.get(2)?,
                        max_value: row.get(3)?,
                        contains_nan: row.get(4)?,
                        column_size_bytes: column_sizes.get(&column_id).copied(),
                        bounds_are_exact,
                    })
                })?
                .collect::<Result<Vec<_>, _>>()?,
            Err(error) if is_missing_statistics_table(&error) => Vec::new(),
            Err(error) => return Err(error.into()),
        };
        Ok(DuckLakeStatistics {
            table,
            columns,
            files: Vec::new(),
        })
    }

    fn get_table_file_metadata_page(
        &self,
        table_id: i64,
        snapshot_id: i64,
        after_data_file_id: Option<i64>,
        limit: usize,
    ) -> crate::Result<Vec<DuckLakeFileMetadata>> {
        // The unfiltered listing is the filtered one with nothing to narrow it,
        // so there is only ever one page query to keep correct.
        self.get_table_file_metadata_page_filtered(
            table_id,
            snapshot_id,
            after_data_file_id,
            limit,
            None,
        )
    }

    fn get_table_file_metadata_page_filtered(
        &self,
        table_id: i64,
        snapshot_id: i64,
        after_data_file_id: Option<i64>,
        limit: usize,
        filter: Option<&StatsFilter>,
    ) -> crate::Result<Vec<DuckLakeFileMetadata>> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let limit = i64::try_from(limit).map_err(|_| {
            crate::DuckLakeError::InvalidConfig("file metadata page limit exceeds i64".to_string())
        })?;
        let conn = self.connection();
        // The keyset cursor's "start of table" sentinel: every `data_file_id`
        // is greater than it, so the first page is unbounded below.
        let after_data_file_id = after_data_file_id.unwrap_or(i64::MIN);

        // A filter the dialect cannot render contributes nothing, and this is
        // only ever a pre-filter: whatever comes back still goes through the
        // in-memory `PruningPredicate` in `table.rs`.
        let rendered = filter
            .and_then(|filter| filter.render(&DuckdbStatsDialect))
            .unwrap_or_default();
        let files = match query_data_file_page(
            &conn,
            table_id,
            snapshot_id,
            after_data_file_id,
            limit,
            &rendered,
        ) {
            Ok(files) => files,
            // The filter is advisory, so a catalog the narrowed query cannot run
            // still lists its files. A catalog written before
            // `ducklake_file_column_stats` existed is the case this was built
            // for — there is no table for the CTEs to read — but the retry
            // covers any failure the filter introduced, because the rendered
            // predicate is the one part of this query built from a caller's
            // arbitrary expression. Losing the pruning is slow; failing here
            // would fail a scan that planned fine without a filter. The error is
            // logged rather than swallowed, and a failure that is not the
            // filter's fault surfaces from the retry.
            Err(error) if !rendered.is_empty() => {
                tracing::debug!(
                    %error,
                    table_id,
                    "statistics-filtered file listing failed; listing every file"
                );
                query_data_file_page(&conn, table_id, snapshot_id, after_data_file_id, limit, &[])?
            },
            Err(error) => return Err(error.into()),
        };

        let Some(last_data_file_id) = files.last().map(|file| file.data_file_id) else {
            return Ok(Vec::new());
        };
        // A filtered page's ids are sparse within `(after, last]`, so the two
        // enrichment queries below are restricted to the ids actually returned
        // rather than to that whole range. Unfiltered the range holds at most
        // one page of files, but a selective filter can put a handful of
        // survivors at the far end of a million-file table, and the range would
        // then pull back the statistics of every file the filter just pruned —
        // the exact cost the pushdown exists to remove, and unbounded resident
        // memory besides. Ids are `i64`, so inlining them adds no placeholder
        // and the `params![]` lists stay as they are.
        let page_ids = (!rendered.is_empty()).then(|| {
            files
                .iter()
                .map(|file| file.data_file_id.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        });
        let page_id_filter = |column: &str| {
            page_ids.as_ref().map_or_else(String::new, |ids| {
                format!("\n               AND {column} IN ({ids})")
            })
        };
        let statistics = match conn.prepare(&format!(
            "SELECT stats.data_file_id, stats.column_id,
                    stats.column_size_bytes, stats.value_count, stats.null_count,
                    stats.min_value, stats.max_value, stats.contains_nan
             FROM ducklake_file_column_stats AS stats
             INNER JOIN ducklake_data_file AS data
               ON data.data_file_id = stats.data_file_id
              AND data.table_id = stats.table_id
             WHERE stats.table_id = ?
               AND ? >= data.begin_snapshot
               AND (? < data.end_snapshot OR data.end_snapshot IS NULL)
               AND stats.data_file_id > ?
               AND stats.data_file_id <= ?{}
             ORDER BY stats.data_file_id, stats.column_id",
            page_id_filter("stats.data_file_id")
        )) {
            Ok(mut statement) => statement
                .query_map(
                    params![
                        table_id,
                        snapshot_id,
                        snapshot_id,
                        after_data_file_id,
                        last_data_file_id
                    ],
                    |row| {
                        Ok(DuckLakeFileColumnStatistics {
                            data_file_id: row.get(0)?,
                            column_id: row.get(1)?,
                            column_size_bytes: row.get(2)?,
                            value_count: row.get(3)?,
                            null_count: row.get(4)?,
                            min_value: row.get(5)?,
                            max_value: row.get(6)?,
                            contains_nan: row.get(7)?,
                        })
                    },
                )?
                .collect::<Result<Vec<_>, _>>()?,
            Err(error) if is_missing_statistics_table(&error) => Vec::new(),
            Err(error) => return Err(error.into()),
        };
        let mut statistics_by_file: HashMap<i64, Vec<_>> = HashMap::new();
        for statistic in statistics {
            statistics_by_file
                .entry(statistic.data_file_id)
                .or_default()
                .push(statistic);
        }

        // Enrich with per-file partition values (for pruning), scoped to the page's
        // data_file_id range. Rows for files outside the page (e.g. retired at this
        // snapshot but in-range) are harmless — matched only to files in the page.
        let mut values_by_file: HashMap<i64, Vec<(i32, Option<String>)>> = HashMap::new();
        match conn.prepare(&format!(
            "{SQL_GET_FILE_PARTITION_VALUES}{}",
            page_id_filter("data_file_id")
        )) {
            Ok(mut stmt) => {
                let rows = stmt.query_map(
                    params![table_id, after_data_file_id, last_data_file_id],
                    |row| {
                        Ok((
                            row.get::<_, i64>(0)?,
                            i32::try_from(row.get::<_, i64>(1)?).unwrap_or(0),
                            row.get::<_, Option<String>>(2)?,
                        ))
                    },
                )?;
                for row in rows {
                    let (data_file_id, key_index, value) = row?;
                    values_by_file
                        .entry(data_file_id)
                        .or_default()
                        .push((key_index, value));
                }
            },
            Err(error) if is_missing_statistics_table(&error) => {},
            Err(error) => return Err(error.into()),
        }

        Ok(files
            .into_iter()
            .map(|mut file| {
                if let Some(values) = values_by_file.remove(&file.data_file_id) {
                    file.partition_values = values;
                }
                DuckLakeFileMetadata {
                    column_statistics: statistics_by_file
                        .remove(&file.data_file_id)
                        .unwrap_or_default(),
                    file,
                }
            })
            .collect())
    }

    fn get_table_statistics(
        &self,
        table_id: i64,
        snapshot_id: i64,
    ) -> crate::Result<DuckLakeStatistics> {
        let conn = self.connection();

        let table = match conn.prepare(SQL_GET_TABLE_STATS) {
            Ok(mut stmt) => {
                let mut rows = stmt.query([table_id])?;
                rows.next()?
                    .map(|row| {
                        Ok::<_, duckdb::Error>(DuckLakeTableStatistics {
                            record_count: row.get(0)?,
                            file_size_bytes: row.get(1)?,
                        })
                    })
                    .transpose()?
            },
            Err(error) if is_missing_statistics_table(&error) => None,
            Err(error) => return Err(error.into()),
        };

        let columns = match conn.prepare(SQL_GET_TABLE_COLUMN_STATS) {
            Ok(mut stmt) => stmt
                .query_map([table_id], |row| {
                    Ok(DuckLakeTableColumnStatistics {
                        column_id: row.get(0)?,
                        contains_null: row.get(1)?,
                        min_value: row.get(2)?,
                        max_value: row.get(3)?,
                        contains_nan: row.get(4)?,
                        column_size_bytes: None,
                        bounds_are_exact: false,
                    })
                })?
                .collect::<Result<Vec<_>, _>>()?,
            Err(error) if is_missing_statistics_table(&error) => Vec::new(),
            Err(error) => return Err(error.into()),
        };

        let files = match conn.prepare(SQL_GET_FILE_COLUMN_STATS) {
            Ok(mut stmt) => stmt
                .query_map([table_id, snapshot_id, snapshot_id], |row| {
                    Ok(DuckLakeFileColumnStatistics {
                        data_file_id: row.get(0)?,
                        column_id: row.get(1)?,
                        column_size_bytes: row.get(2)?,
                        value_count: row.get(3)?,
                        null_count: row.get(4)?,
                        min_value: row.get(5)?,
                        max_value: row.get(6)?,
                        contains_nan: row.get(7)?,
                    })
                })?
                .collect::<Result<Vec<_>, _>>()?,
            Err(error) if is_missing_statistics_table(&error) => Vec::new(),
            Err(error) => return Err(error.into()),
        };

        Ok(DuckLakeStatistics {
            table,
            columns,
            files,
        })
    }

    fn get_inlined_data(
        &self,
        table_id: i64,
        snapshot_id: i64,
        columns: &[DuckLakeTableColumn],
    ) -> crate::Result<Vec<RecordBatch>> {
        let conn = self.connection();
        if !self.schema_capabilities(&conn)?.inlined_data_tables {
            return Ok(Vec::new());
        }
        let mut registry =
            conn.prepare("SELECT table_name FROM ducklake_inlined_data_tables WHERE table_id = ?")?;
        let tables = registry
            .query_map([table_id], |row| row.get::<_, String>(0))?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let schema: SchemaRef = Arc::new(crate::types::build_arrow_schema(columns)?);
        let mut batches = Vec::new();

        for table in tables {
            if !is_inlined_data_table(&table) {
                continue;
            }
            let info_sql = format!("SELECT name FROM pragma_table_info('{table}')");
            let mut info = conn.prepare(&info_sql)?;
            let present = info
                .query_map([], |row| row.get::<_, String>(0))?
                .collect::<std::result::Result<HashSet<_>, _>>()?;
            let projected = columns
                .iter()
                .zip(schema.fields())
                .map(|(column, field)| {
                    if !present.contains(&column.column_name) {
                        return "NULL".to_string();
                    }
                    let ident = quote_ident(&column.column_name);
                    if matches!(
                        field.data_type(),
                        DataType::Utf8
                            | DataType::LargeUtf8
                            | DataType::Utf8View
                            | DataType::FixedSizeBinary(_)
                    ) {
                        format!("CAST({ident} AS VARCHAR)")
                    } else {
                        ident
                    }
                })
                .collect::<Vec<_>>()
                .join(", ");
            let sql = format!(
                "SELECT {projected} FROM {} \
                 WHERE ? >= begin_snapshot AND (? < end_snapshot OR end_snapshot IS NULL) \
                 ORDER BY row_id",
                quote_ident(&table)
            );
            let mut statement = conn.prepare(&sql)?;
            let mut query = statement.query(params![snapshot_id, snapshot_id])?;
            let mut rows = Vec::new();
            while let Some(row) = query.next()? {
                let values = schema
                    .fields()
                    .iter()
                    .enumerate()
                    .map(|(index, field)| {
                        if !present.contains(&columns[index].column_name) {
                            return inlined_missing_scalar(&columns[index], field.data_type());
                        }
                        duckdb_inlined_scalar(
                            row.get_ref(index)?,
                            field.data_type(),
                            &columns[index].column_name,
                        )
                    })
                    .collect::<crate::Result<Vec<_>>>()?;
                rows.push(values);
            }
            if !rows.is_empty() {
                batches.push(build_inlined_batch(schema.clone(), columns, &rows)?);
            }
        }
        Ok(batches)
    }

    fn get_inlined_deletes(
        &self,
        table_id: i64,
        snapshot_id: i64,
    ) -> crate::Result<Vec<DuckLakeInlinedDelete>> {
        let conn = self.connection();
        let table = inlined_delete_table_name(table_id)?;
        let sql = format!(
            "SELECT file_id, row_id FROM {} WHERE begin_snapshot <= ? ORDER BY file_id, row_id",
            quote_ident(&table)
        );
        let mut statement = match conn.prepare(&sql) {
            Ok(statement) => statement,
            Err(error) if is_missing_statistics_table(&error) => return Ok(Vec::new()),
            Err(error) => return Err(error.into()),
        };
        Ok(statement
            .query_map([snapshot_id], |row| {
                Ok(DuckLakeInlinedDelete {
                    data_file_id: row.get(0)?,
                    row_id: row.get(1)?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?)
    }

    fn get_inlined_data_with_row_ids(
        &self,
        table_id: i64,
        snapshot_id: i64,
        columns: &[DuckLakeTableColumn],
    ) -> crate::Result<Vec<DuckLakeInlinedData>> {
        let conn = self.connection();
        if !self.schema_capabilities(&conn)?.inlined_data_tables {
            return Ok(Vec::new());
        }

        let mut registry =
            conn.prepare("SELECT table_name FROM ducklake_inlined_data_tables WHERE table_id = ?")?;
        let tables = registry
            .query_map([table_id], |row| row.get::<_, String>(0))?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let schema: SchemaRef = Arc::new(crate::types::build_strict_arrow_schema(columns)?);
        let mut batches = Vec::new();
        for table in tables {
            if !is_inlined_data_table(&table) {
                continue;
            }

            let info_sql = format!("SELECT name FROM pragma_table_info('{table}')");
            let mut info = conn.prepare(&info_sql)?;
            let present = info
                .query_map([], |row| row.get::<_, String>(0))?
                .collect::<std::result::Result<HashSet<_>, _>>()?;
            let projected = columns
                .iter()
                .zip(schema.fields())
                .map(|(column, field)| {
                    if !present.contains(&column.column_name) {
                        return "NULL".to_string();
                    }
                    let ident = quote_ident(&column.column_name);
                    if matches!(
                        field.data_type(),
                        DataType::Utf8
                            | DataType::LargeUtf8
                            | DataType::Utf8View
                            | DataType::FixedSizeBinary(_)
                    ) {
                        format!("CAST({ident} AS VARCHAR)")
                    } else {
                        ident
                    }
                })
                .collect::<Vec<_>>()
                .join(", ");
            let sql = format!(
                "SELECT row_id, begin_snapshot, {projected} FROM {} \
                 WHERE ? >= begin_snapshot AND (? < end_snapshot OR end_snapshot IS NULL) \
                 ORDER BY begin_snapshot, row_id",
                quote_ident(&table),
            );
            let mut statement = conn.prepare(&sql)?;
            let mut query = statement.query(params![snapshot_id, snapshot_id])?;
            let mut row_ids = Vec::new();
            let mut begin_snapshots = Vec::new();
            let mut rows = Vec::new();
            while let Some(row) = query.next()? {
                row_ids.push(row.get(0)?);
                begin_snapshots.push(row.get(1)?);
                rows.push(
                    schema
                        .fields()
                        .iter()
                        .enumerate()
                        .map(|(index, field)| {
                            if !present.contains(&columns[index].column_name) {
                                return inlined_missing_scalar(&columns[index], field.data_type());
                            }
                            duckdb_inlined_scalar(
                                row.get_ref(index + 2)?,
                                field.data_type(),
                                &columns[index].column_name,
                            )
                        })
                        .collect::<crate::Result<Vec<_>>>()?,
                );
            }
            if rows.is_empty() {
                continue;
            }
            batches.push(DuckLakeInlinedData {
                table_name: table,
                row_ids,
                begin_snapshots,
                batch: build_inlined_batch(schema.clone(), columns, &rows)?,
            });
        }
        Ok(batches)
    }

    fn get_schema_by_name(
        &self,
        name: &str,
        snapshot_id: i64,
    ) -> crate::Result<Option<SchemaMetadata>> {
        let conn = self.connection();
        let mut stmt = conn.prepare(SQL_GET_SCHEMA_BY_NAME)?;

        let mut rows = stmt.query(params![name, snapshot_id, snapshot_id])?;

        if let Some(row) = rows.next()? {
            let schema_id: i64 = row.get(0)?;
            let schema_name: String = row.get(1)?;
            let path: String = row.get(2)?;
            let path_is_relative: bool = row.get(3)?;
            Ok(Some(SchemaMetadata {
                schema_id,
                schema_name,
                path,
                path_is_relative,
            }))
        } else {
            Ok(None)
        }
    }

    fn get_table_by_name(
        &self,
        schema_id: i64,
        name: &str,
        snapshot_id: i64,
    ) -> crate::Result<Option<TableMetadata>> {
        let conn = self.connection();
        let mut stmt = conn.prepare(SQL_GET_TABLE_BY_NAME)?;

        let mut rows = stmt.query(params![&schema_id, &name, &snapshot_id, &snapshot_id])?;

        if let Some(row) = rows.next()? {
            let table_id: i64 = row.get(0)?;
            let table_name: String = row.get(1)?;
            let path: String = row.get(2)?;
            let path_is_relative: bool = row.get(3)?;
            Ok(Some(TableMetadata {
                table_id,
                table_name,
                path,
                path_is_relative,
            }))
        } else {
            Ok(None)
        }
    }

    fn get_view_by_name(
        &self,
        schema_id: i64,
        name: &str,
        snapshot_id: i64,
    ) -> crate::Result<Option<ViewMetadata>> {
        let conn = self.connection();
        if !self.schema_capabilities(&conn)?.views {
            return Ok(None);
        }
        let mut stmt = conn.prepare(SQL_GET_VIEW_BY_NAME)?;
        let mut rows = stmt.query(params![schema_id, name, snapshot_id, snapshot_id])?;
        if let Some(row) = rows.next()? {
            Ok(Some(decode_view(row)?))
        } else {
            Ok(None)
        }
    }

    fn table_exists(&self, schema_id: i64, name: &str, snapshot_id: i64) -> crate::Result<bool> {
        let conn = self.connection();
        let exists: bool = conn.query_row(
            SQL_TABLE_EXISTS,
            params![schema_id, &name, &snapshot_id, &snapshot_id],
            |row| row.get(0),
        )?;
        Ok(exists)
    }

    fn list_all_tables(&self, snapshot_id: i64) -> crate::Result<Vec<TableWithSchema>> {
        let conn = self.connection();
        let mut stmt = conn.prepare(SQL_LIST_ALL_TABLES)?;

        let tables = stmt
            .query_map(
                params![snapshot_id, snapshot_id, snapshot_id, snapshot_id],
                |row| {
                    let schema_name: String = row.get(0)?;
                    let table = TableMetadata {
                        table_id: row.get(1)?,
                        table_name: row.get(2)?,
                        path: row.get(3)?,
                        path_is_relative: row.get(4)?,
                    };
                    Ok(TableWithSchema {
                        schema_name,
                        table,
                    })
                },
            )?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(tables)
    }

    fn list_all_views(&self, snapshot_id: i64) -> crate::Result<Vec<ViewWithSchema>> {
        let conn = self.connection();
        if !self.schema_capabilities(&conn)?.views {
            return Ok(Vec::new());
        }
        let mut stmt = conn.prepare(SQL_LIST_ALL_VIEWS)?;
        stmt.query_map(
            params![snapshot_id, snapshot_id, snapshot_id, snapshot_id],
            |row| {
                Ok(ViewWithSchema {
                    schema_name: row.get(0)?,
                    view: ViewMetadata {
                        view_id: row.get(1)?,
                        schema_id: row.get(2)?,
                        begin_snapshot: row.get(3)?,
                        view_name: row.get(4)?,
                        dialect: row.get(5)?,
                        sql: row.get(6)?,
                        column_aliases: row.get(7)?,
                    },
                })
            },
        )?
        .collect::<Result<Vec<_>, _>>()
        .map_err(Into::into)
    }

    fn list_all_columns(&self, snapshot_id: i64) -> crate::Result<Vec<ColumnWithTable>> {
        let conn = self.connection();
        let sql = list_all_columns_sql(self.schema_capabilities(&conn)?);
        let mut stmt = conn.prepare(&sql)?;

        let raw_columns: Vec<(ColumnWithTable, Option<i64>)> = stmt
            .query_map(
                params![
                    snapshot_id,
                    snapshot_id,
                    snapshot_id,
                    snapshot_id,
                    snapshot_id,
                    snapshot_id
                ],
                |row| {
                    let schema_name: String = row.get(0)?;
                    let table_name: String = row.get(1)?;
                    let nulls_allowed: Option<bool> = row.get(5)?;
                    let parent_column: Option<i64> = row.get(6)?;
                    let column = DuckLakeTableColumn::new(
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                        nulls_allowed.unwrap_or(true),
                    )
                    .with_defaults(
                        row.get(7)?,
                        row.get(8)?,
                        row.get(9)?,
                        row.get(10)?,
                    );
                    Ok((
                        ColumnWithTable {
                            schema_name,
                            table_name,
                            column,
                        },
                        parent_column,
                    ))
                },
            )?
            .collect::<Result<Vec<_>, _>>()?;

        reconstruct_columns_with_table(raw_columns)
    }

    fn list_all_files(&self, snapshot_id: i64) -> crate::Result<Vec<FileWithTable>> {
        let conn = self.connection();
        let mut stmt = conn.prepare(SQL_LIST_ALL_FILES)?;

        let files = stmt
            .query_map(
                params![
                    snapshot_id,
                    snapshot_id,
                    snapshot_id,
                    snapshot_id,
                    snapshot_id,
                    snapshot_id,
                    snapshot_id,
                    snapshot_id
                ],
                |row| {
                    let schema_name: String = row.get(0)?;
                    let table_name: String = row.get(1)?;

                    // Column 2 is data_file_id; columns 3-7 are the data file.
                    let data_file_id: i64 = row.get(2)?;
                    let data_file = DuckLakeFileData {
                        path: row.get(3)?,
                        path_is_relative: row.get(4)?,
                        file_size_bytes: row.get(5)?,
                        footer_size: row.get(6)?,
                        encryption_key: row.get(7)?,
                        mapping_id: None,
                    };

                    // Column 8 is delete_file_id (NULL when no live delete file).
                    let (delete_file, delete_file_id) =
                        if let Ok(Some(dfid)) = row.get::<_, Option<i64>>(8) {
                            (
                                Some(DuckLakeFileData {
                                    path: row.get(9)?,
                                    path_is_relative: row.get(10)?,
                                    file_size_bytes: row.get(11)?,
                                    footer_size: row.get(12)?,
                                    encryption_key: row.get(13)?,
                                    mapping_id: None,
                                }),
                                Some(dfid),
                            )
                        } else {
                            (None, None)
                        };

                    let max_row_count = row.get::<_, Option<i64>>(14)?;

                    Ok(FileWithTable {
                        schema_name,
                        table_name,
                        file: DuckLakeTableFile {
                            data_file_id,
                            file: data_file,
                            delete_file_id,
                            delete_file,
                            row_id_start: None,
                            snapshot_id: None,
                            begin_snapshot: None,
                            schema_version: None,
                            partial_max: None,
                            max_row_count,
                            delete_count: None,
                            partition_id: None,
                            partition_values: Vec::new(),
                        },
                    })
                },
            )?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(files)
    }

    fn get_data_files_added_between_snapshots(
        &self,
        table_id: i64,
        start_snapshot: i64,
        end_snapshot: i64,
    ) -> crate::Result<Vec<DataFileChange>> {
        let conn = self.connection();

        // DuckLake's catalog schema renamed the merged-partial-file marker:
        // older catalogs (spec 0.2, written by earlier ducklake extensions)
        // carry `partial_file_info` (a cumulative `snapshot:rowcount|...`
        // string); current ones carry `partial_max` (BIGINT). Detect which
        // column this catalog has and query accordingly.
        if self.schema_capabilities(&conn)?.data_file_partial_max {
            let mut stmt = conn.prepare(SQL_GET_DATA_FILES_ADDED_BETWEEN_SNAPSHOTS)?;
            let files = stmt
                .query_map(params![table_id, start_snapshot, end_snapshot], |row| {
                    Ok(DataFileChange {
                        begin_snapshot: row.get(0)?,
                        path: row.get(1)?,
                        path_is_relative: row.get(2)?,
                        file_size_bytes: row.get(3)?,
                        footer_size: row.get(4)?,
                        encryption_key: row.get(5)?,
                        row_id_start: row.get(6)?,
                        partial_max: row.get(7)?,
                        mapping_id: row.get(8)?,
                    })
                })?
                .collect::<Result<Vec<_>, _>>()?;
            return Ok(files);
        }

        // Old-spec catalog: fetch candidate partial files broadly and apply the
        // `partial_max >= start` bound in Rust after parsing the info string.
        let mut stmt = conn.prepare(
            "SELECT
                data.begin_snapshot,
                data.path,
                data.path_is_relative,
                data.file_size_bytes,
                data.footer_size,
                data.encryption_key,
                data.row_id_start,
                data.partial_file_info,
                data.mapping_id
            FROM ducklake_data_file AS data
            WHERE data.table_id = $1
              AND data.begin_snapshot <= $3
              AND (data.begin_snapshot >= $2 OR data.partial_file_info IS NOT NULL)
            ORDER BY data.begin_snapshot",
        )?;
        let files = stmt
            .query_map(params![table_id, start_snapshot, end_snapshot], |row| {
                let info: Option<String> = row.get(7)?;
                Ok(DataFileChange {
                    begin_snapshot: row.get(0)?,
                    path: row.get(1)?,
                    path_is_relative: row.get(2)?,
                    file_size_bytes: row.get(3)?,
                    footer_size: row.get(4)?,
                    encryption_key: row.get(5)?,
                    row_id_start: row.get(6)?,
                    partial_max: info.as_deref().and_then(parse_partial_file_info_max),
                    mapping_id: row.get(8)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .filter(|f: &DataFileChange| {
                f.begin_snapshot >= start_snapshot
                    || f.partial_max.is_some_and(|max| max >= start_snapshot)
            })
            .collect();

        Ok(files)
    }

    fn get_delete_files_added_between_snapshots(
        &self,
        table_id: i64,
        start_snapshot: i64,
        end_snapshot: i64,
    ) -> crate::Result<Vec<DeleteFileChange>> {
        let conn = self.connection();

        // Cumulative (current-spec) delete files can hold in-window deletions
        // even when their begin_snapshot predates the window; they are included
        // via `ducklake_delete_file.partial_max` (their max embedded snapshot).
        // Older catalogs have no such column — and no cumulative delete files —
        // so the predicate degrades to NULL there, keeping the plain
        // begin-snapshot window.
        let sql = if self.schema_capabilities(&conn)?.delete_file_partial_max {
            SQL_GET_DELETE_FILES_ADDED_BETWEEN_SNAPSHOTS.to_string()
        } else {
            SQL_GET_DELETE_FILES_ADDED_BETWEEN_SNAPSHOTS.replace("df.partial_max", "NULL")
        };
        let mut stmt = conn.prepare(&sql)?;

        let files = stmt
            .query_map(params![table_id, start_snapshot, end_snapshot], |row| {
                Ok(DeleteFileChange {
                    // data file
                    data_file_path: row.get(0)?,
                    data_file_path_is_relative: row.get(1)?,
                    data_file_size_bytes: row.get(2)?,
                    data_file_footer_size: row.get(3)?,
                    data_row_id_start: row.get(4)?,
                    data_record_count: row.get(5)?,
                    data_mapping_id: row.get(6)?,

                    // current delete
                    current_delete_path: row.get(7)?,
                    current_delete_path_is_relative: row.get(8)?,
                    current_delete_file_size_bytes: row.get(9)?,
                    current_delete_footer_size: row.get(10)?,

                    // previous delete
                    previous_delete_path: row.get(11)?,
                    previous_delete_path_is_relative: row.get(12)?,
                    previous_delete_file_size_bytes: row.get(13)?,
                    previous_delete_footer_size: row.get(14)?,

                    // snapshot
                    snapshot_id: row.get(15)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(files)
    }
}

/// Parse the maximum origin snapshot id out of an old-spec `partial_file_info`
/// string — a `|`-separated list of cumulative `snapshot:rowcount` pairs (e.g.
/// `"2:1|3:2|4:3"`), whose last pair carries the file's maximum snapshot.
fn parse_partial_file_info_max(info: &str) -> Option<i64> {
    info.rsplit('|')
        .next()
        .and_then(|pair| pair.split(':').next())
        .and_then(|snap| snap.trim().parse::<i64>().ok())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::datatypes::{DataType, Field};
    use duckdb::arrow::array::{Int32Builder, ListBuilder};
    use duckdb::types::{ListType, ValueRef};

    use super::{
        DuckdbMetadataProvider, DuckdbStatsDialect, SchemaCapabilities, data_files_sql_filtered,
        duckdb_inlined_scalar, get_table_columns_sql, is_missing_statistics_table,
        list_all_columns_sql, parse_partial_file_info_max, query_data_file_page,
    };
    use crate::metadata_provider::{DuckLakeTableColumn, MetadataProvider};
    use crate::stats_filter::{
        StatKind, StatsColumnFilter, StatsExpr, StatsFilter, lower_predicate,
    };
    use arrow::datatypes::Schema;
    use datafusion::common::ScalarValue;
    use datafusion::logical_expr::Operator;
    use datafusion::physical_expr::PhysicalExpr;
    use datafusion::physical_expr::expressions::{BinaryExpr, Column, lit};
    use duckdb::{Connection, params};
    use std::collections::BTreeSet;
    use tempfile::TempDir;

    /// Render `filters` for the DuckDB dialect and splice them into the
    /// data-file listing, for a single Int32 column with the given `column_id`.
    /// Count `?` outside single-quoted string literals, which is what a driver
    /// treats as a bind placeholder.
    fn placeholders_outside_literals(sql: &str) -> usize {
        let mut in_literal = false;
        let mut count = 0;
        for byte in sql.bytes() {
            match byte {
                b'\'' => in_literal = !in_literal,
                b'?' if !in_literal => count += 1,
                _ => {},
            }
        }
        count
    }

    fn filtered_listing_sql(
        predicate: Arc<dyn PhysicalExpr>,
        column_id: i64,
        table_id: i64,
    ) -> String {
        filtered_listing_sql_typed(predicate, column_id, table_id, DataType::Int32, "int32")
    }

    fn filtered_listing_sql_typed(
        predicate: Arc<dyn PhysicalExpr>,
        column_id: i64,
        table_id: i64,
        data_type: DataType,
        ducklake_type: &str,
    ) -> String {
        let schema = Schema::new(vec![Field::new("a", data_type, true)]);
        let column =
            DuckLakeTableColumn::new(column_id, "a".to_string(), ducklake_type.to_string(), true);
        let rendered = lower_predicate(&predicate, &schema, &[column])
            .expect("predicate lowers")
            .render(&DuckdbStatsDialect)
            .expect("filter renders for DuckDB");
        data_files_sql_filtered(table_id, &rendered).expect("filter splices into the listing")
    }

    /// `a > 5 AND a < 10` on an `INTEGER` column, spliced into the listing the
    /// same way official DuckLake assembles it: a CTE per column selecting only
    /// the stats the condition reads, one LEFT JOIN, conditions ANDed onto the
    /// existing WHERE, and no new bind placeholder.
    #[test]
    fn two_conjunct_integer_filter_renders_official_shape() {
        let a = Arc::new(Column::new("a", 0)) as Arc<dyn PhysicalExpr>;
        let predicate = Arc::new(BinaryExpr::new(
            Arc::new(BinaryExpr::new(Arc::clone(&a), Operator::Gt, lit(5i32))),
            Operator::And,
            Arc::new(BinaryExpr::new(a, Operator::Lt, lit(10i32))),
        )) as Arc<dyn PhysicalExpr>;

        let sql = filtered_listing_sql(predicate, 7, 3);
        assert!(
            sql.starts_with(
                "WITH col_7_stats AS (\n        SELECT data_file_id, min_value, max_value, value_count\n        \
                 FROM ducklake_file_column_stats\n        WHERE column_id = 7 AND table_id = 3\n    )"
            ),
            "unexpected CTE section:\n{sql}"
        );
        assert!(
            sql.contains(
                "\n    LEFT JOIN col_7_stats ON col_7_stats.data_file_id = data.data_file_id"
            ),
            "unexpected join:\n{sql}"
        );
        // `IS NOT FALSE` on the outside is what keeps a file whose condition is
        // merely unknown. A present-but-malformed stat is not NULL, so the
        // per-stat `IS NULL OR` disjuncts do not fire, but `TRY_CAST` of it is —
        // and a NULL condition under `WHERE ... AND` would prune the file.
        assert!(
            sql.ends_with(
                "\n      AND ((col_7_stats.data_file_id IS NULL OR \
                 ((col_7_stats.value_count IS NULL OR col_7_stats.value_count > 0) AND \
                 (col_7_stats.min_value IS NULL OR col_7_stats.max_value IS NULL OR \
                 (CASE WHEN regexp_full_match(col_7_stats.max_value, '^-?[0-9]{1,20}$') \
                 THEN TRY_CAST(col_7_stats.max_value AS INTEGER) END > 5) AND \
                 (CASE WHEN regexp_full_match(col_7_stats.min_value, '^-?[0-9]{1,20}$') \
                 THEN TRY_CAST(col_7_stats.min_value AS INTEGER) END < 10))))) IS NOT FALSE"
            ),
            "unexpected condition:\n{sql}"
        );
        // The six placeholders of the unfiltered listing, plus nothing: literals
        // are inlined so the caller's `params![]` list never has to change.
        //
        // Counted outside string literals, because a shape pattern contains `?`
        // as a regex quantifier. Every driver here tokenises those as string
        // content rather than as a placeholder — `filtered_listing_prunes_in_duckdb`
        // and `a_nan_float_bound_keeps_its_file` both prepare and run this SQL
        // with exactly eight bound parameters — but counting them would make this
        // assertion track the patterns instead of the parameter list.
        assert_eq!(
            placeholders_outside_literals(&sql),
            6,
            "placeholder count changed"
        );
    }

    /// The generated SQL is valid DuckDB and prunes on real stats rows: DuckDB
    /// binds and runs it, the file whose recorded range cannot match is gone,
    /// and the file with no stats row at all survives the LEFT JOIN.
    #[test]
    fn filtered_listing_prunes_in_duckdb() {
        let conn = Connection::open_in_memory().expect("in-memory DuckDB");
        conn.execute_batch(
            "CREATE TABLE ducklake_data_file (
                 data_file_id BIGINT, table_id BIGINT, begin_snapshot BIGINT,
                 end_snapshot BIGINT, path VARCHAR, path_is_relative BOOLEAN,
                 file_size_bytes BIGINT, footer_size BIGINT, encryption_key VARCHAR,
                 row_id_start BIGINT, record_count BIGINT, mapping_id BIGINT);
             CREATE TABLE ducklake_delete_file (
                 delete_file_id BIGINT, data_file_id BIGINT, table_id BIGINT,
                 begin_snapshot BIGINT, end_snapshot BIGINT, path VARCHAR,
                 path_is_relative BOOLEAN, file_size_bytes BIGINT, footer_size BIGINT,
                 encryption_key VARCHAR, delete_count BIGINT);
             CREATE TABLE ducklake_file_column_stats (
                 data_file_id BIGINT, table_id BIGINT, column_id BIGINT,
                 column_size_bytes BIGINT, value_count BIGINT, null_count BIGINT,
                 min_value VARCHAR, max_value VARCHAR, contains_nan BOOLEAN);
             INSERT INTO ducklake_data_file VALUES
                 (1, 3, 0, NULL, 'low.parquet', true, 10, 1, NULL, 0, 100, NULL),
                 (2, 3, 0, NULL, 'high.parquet', true, 10, 1, NULL, 100, 100, NULL),
                 (3, 3, 0, NULL, 'unknown.parquet', true, 10, 1, NULL, 200, 100, NULL);
             INSERT INTO ducklake_file_column_stats VALUES
                 (1, 3, 7, 8, 100, 0, '6', '9', NULL),
                 (2, 3, 7, 8, 100, 0, '100', '900', NULL);",
        )
        .expect("catalog fixture");

        let a = Arc::new(Column::new("a", 0)) as Arc<dyn PhysicalExpr>;
        let predicate = Arc::new(BinaryExpr::new(
            Arc::new(BinaryExpr::new(Arc::clone(&a), Operator::Gt, lit(5i32))),
            Operator::And,
            Arc::new(BinaryExpr::new(a, Operator::Lt, lit(10i32))),
        )) as Arc<dyn PhysicalExpr>;
        let sql = format!(
            "{}\n AND data.data_file_id > ?\n ORDER BY data.data_file_id\n LIMIT ?",
            filtered_listing_sql(predicate, 7, 3)
        );

        let mut statement = conn
            .prepare(&sql)
            .expect("DuckDB binds the filtered listing");
        let paths: Vec<String> = statement
            .query_map(
                params![3i64, 0i64, 0i64, 3i64, 0i64, 0i64, i64::MIN, 10i64],
                |row| row.get::<_, String>(1),
            )
            .expect("query runs")
            .collect::<Result<Vec<_>, _>>()
            .expect("rows decode");
        assert_eq!(paths, vec!["low.parquet", "unknown.parquet"]);
    }

    /// A float bound of `nan` keeps its file, and `inf` still prunes.
    ///
    /// `TRY_CAST('nan' AS DOUBLE)` is `nan`, not `NULL`, and `5.0 BETWEEN nan
    /// AND 9.0` is false — so before the shape gate this pruned a file whose
    /// other rows can match, while the in-memory path (`float_bound_is_usable`)
    /// kept it. `ducklake_add_files` over a pre-1.11 parquet-mr file writes
    /// exactly that bound.
    #[test]
    fn a_nan_float_bound_keeps_its_file() {
        let conn = Connection::open_in_memory().expect("in-memory DuckDB");
        conn.execute_batch(
            "CREATE TABLE ducklake_data_file (
                 data_file_id BIGINT, table_id BIGINT, begin_snapshot BIGINT,
                 end_snapshot BIGINT, path VARCHAR, path_is_relative BOOLEAN,
                 file_size_bytes BIGINT, footer_size BIGINT, encryption_key VARCHAR,
                 row_id_start BIGINT, record_count BIGINT, mapping_id BIGINT);
             CREATE TABLE ducklake_delete_file (
                 delete_file_id BIGINT, data_file_id BIGINT, table_id BIGINT,
                 begin_snapshot BIGINT, end_snapshot BIGINT, path VARCHAR,
                 path_is_relative BOOLEAN, file_size_bytes BIGINT, footer_size BIGINT,
                 encryption_key VARCHAR, delete_count BIGINT);
             CREATE TABLE ducklake_file_column_stats (
                 data_file_id BIGINT, table_id BIGINT, column_id BIGINT,
                 column_size_bytes BIGINT, value_count BIGINT, null_count BIGINT,
                 min_value VARCHAR, max_value VARCHAR, contains_nan BOOLEAN);
             INSERT INTO ducklake_data_file VALUES
                 (1, 3, 0, NULL, 'nan-bound.parquet', true, 10, 1, NULL, 0, 100, NULL),
                 (2, 3, 0, NULL, 'brackets.parquet', true, 10, 1, NULL, 100, 100, NULL),
                 (3, 3, 0, NULL, 'excludes.parquet', true, 10, 1, NULL, 200, 100, NULL),
                 (4, 3, 0, NULL, 'inf-bound.parquet', true, 10, 1, NULL, 300, 100, NULL);
             INSERT INTO ducklake_file_column_stats VALUES
                 (1, 3, 7, 8, 100, 0, 'nan', '9.0', NULL),
                 (2, 3, 7, 8, 100, 0, '1.0', '9.0', NULL),
                 (3, 3, 7, 8, 100, 0, '20.0', '90.0', NULL),
                 (4, 3, 7, 8, 100, 0, '-inf', '1.0', NULL);",
        )
        .expect("catalog fixture");

        let f = Arc::new(Column::new("a", 0)) as Arc<dyn PhysicalExpr>;
        let predicate =
            Arc::new(BinaryExpr::new(f, Operator::Eq, lit(5.0f64))) as Arc<dyn PhysicalExpr>;
        let sql = format!(
            "{}\n AND data.data_file_id > ?\n ORDER BY data.data_file_id\n LIMIT ?",
            filtered_listing_sql_typed(predicate, 7, 3, DataType::Float64, "double")
        );

        let mut statement = conn
            .prepare(&sql)
            .expect("DuckDB binds the filtered listing");
        let paths: Vec<String> = statement
            .query_map(
                params![3i64, 0i64, 0i64, 3i64, 0i64, 0i64, i64::MIN, 10i64],
                |row| row.get::<_, String>(1),
            )
            .expect("query runs")
            .collect::<Result<Vec<_>, _>>()
            .expect("rows decode");

        // `nan-bound` is kept because its bound is not a value this crate could
        // have written; `brackets` because 5.0 is inside [1.0, 9.0]; `excludes`
        // is pruned on real bounds; `inf-bound` is pruned on a real `-inf`
        // bound, which stays usable.
        assert_eq!(paths, vec!["nan-bound.parquet", "brackets.parquet"]);
    }

    /// A catalog predating `ducklake_file_column_stats` still lists its files.
    /// The filtered query cannot bind against it, and the error it raises is
    /// the one `get_table_file_metadata_page_filtered` recognises as "no
    /// statistics table" before retrying without the filter.
    #[test]
    fn missing_statistics_table_is_recognised_and_falls_back() {
        let conn = Connection::open_in_memory().expect("in-memory DuckDB");
        conn.execute_batch(
            "CREATE TABLE ducklake_data_file (
                 data_file_id BIGINT, table_id BIGINT, begin_snapshot BIGINT,
                 end_snapshot BIGINT, path VARCHAR, path_is_relative BOOLEAN,
                 file_size_bytes BIGINT, footer_size BIGINT, encryption_key VARCHAR,
                 row_id_start BIGINT, record_count BIGINT, mapping_id BIGINT);
             CREATE TABLE ducklake_delete_file (
                 delete_file_id BIGINT, data_file_id BIGINT, table_id BIGINT,
                 begin_snapshot BIGINT, end_snapshot BIGINT, path VARCHAR,
                 path_is_relative BOOLEAN, file_size_bytes BIGINT, footer_size BIGINT,
                 encryption_key VARCHAR, delete_count BIGINT);
             INSERT INTO ducklake_data_file VALUES
                 (1, 3, 0, NULL, 'low.parquet', true, 10, 1, NULL, 0, 100, NULL);",
        )
        .expect("legacy catalog fixture");

        let a = Arc::new(Column::new("a", 0)) as Arc<dyn PhysicalExpr>;
        let predicate =
            Arc::new(BinaryExpr::new(a, Operator::Gt, lit(5i32))) as Arc<dyn PhysicalExpr>;
        let schema = Schema::new(vec![Field::new("a", DataType::Int32, true)]);
        let column = DuckLakeTableColumn::new(7, "a".to_string(), "int32".to_string(), true);
        let filter = lower_predicate(&predicate, &schema, &[column]).expect("predicate lowers");
        let rendered = filter
            .render(&DuckdbStatsDialect)
            .expect("filter renders for DuckDB");

        let error = query_data_file_page(&conn, 3, 0, i64::MIN, 10, &rendered)
            .expect_err("the CTE cannot read a table that is not there");
        assert!(
            is_missing_statistics_table(&error),
            "unrecognised error: {error}"
        );
        // Naming the table is what makes this discriminate.
        // `is_missing_statistics_table` matches any "does not exist" / "not
        // found" message, so on its own it would equally accept a binder error
        // from a malformed filter — and the fallback would then quietly hide a
        // real bug behind "legacy catalog".
        assert!(
            error.to_string().contains("ducklake_file_column_stats"),
            "the filtered query failed for some reason other than the missing \
             statistics table: {error}"
        );

        let files = query_data_file_page(&conn, 3, 0, i64::MIN, 10, &[])
            .expect("the unfiltered retry still lists every file");
        assert_eq!(files.len(), 1);
    }

    #[test]
    fn nested_inlined_value_converts_between_arrow_versions() {
        let mut builder = ListBuilder::new(Int32Builder::new());
        builder.values().append_value(1);
        builder.values().append_value(2);
        builder.values().append_value(3);
        builder.append(true);
        let values = builder.finish();
        let target = DataType::List(Arc::new(Field::new("item", DataType::Int32, true)));
        let value = duckdb_inlined_scalar(
            ValueRef::List(ListType::Regular(&values), 0),
            &target,
            "tags",
        )
        .unwrap();
        assert_eq!(value.to_string(), "[1, 2, 3]");
    }

    #[test]
    fn legacy_columns_without_defaults_are_null_projected() -> duckdb::Result<()> {
        let conn = Connection::open_in_memory()?;
        conn.execute_batch(
            "CREATE TABLE ducklake_schema (
                schema_id BIGINT, schema_name VARCHAR, begin_snapshot BIGINT, end_snapshot BIGINT
            );
            CREATE TABLE ducklake_table (
                table_id BIGINT, schema_id BIGINT, table_name VARCHAR,
                begin_snapshot BIGINT, end_snapshot BIGINT
            );
            CREATE TABLE ducklake_column (
                column_id BIGINT, table_id BIGINT, column_order BIGINT, column_name VARCHAR,
                column_type VARCHAR, nulls_allowed BOOLEAN, parent_column BIGINT,
                begin_snapshot BIGINT, end_snapshot BIGINT
            );
            INSERT INTO ducklake_schema VALUES (1, 'main', 1, NULL);
            INSERT INTO ducklake_table VALUES (2, 1, 'events', 1, NULL);
            INSERT INTO ducklake_column VALUES (3, 2, 0, 'id', 'int64', false, NULL, 1, NULL);",
        )?;
        let capabilities = SchemaCapabilities {
            data_file_partial_max: false,
            delete_file_partial_max: false,
            inlined_data_tables: false,
            views: false,
            column_initial_default: false,
            column_default_value: false,
            column_default_value_type: false,
            column_default_value_dialect: false,
        };

        let table_defaults = conn.query_row(
            &get_table_columns_sql(capabilities),
            params![2, 1, 1],
            |row| {
                Ok((
                    row.get::<_, Option<String>>(5)?,
                    row.get::<_, Option<String>>(6)?,
                    row.get::<_, Option<String>>(7)?,
                    row.get::<_, Option<String>>(8)?,
                ))
            },
        )?;
        let listed_defaults = conn.query_row(
            &list_all_columns_sql(capabilities),
            params![1, 1, 1, 1, 1, 1],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, Option<String>>(7)?,
                    row.get::<_, Option<String>>(8)?,
                    row.get::<_, Option<String>>(9)?,
                    row.get::<_, Option<String>>(10)?,
                ))
            },
        )?;

        assert_eq!(table_defaults, (None, None, None, None));
        assert_eq!(
            listed_defaults,
            (
                "main".to_string(),
                "events".to_string(),
                None,
                None,
                None,
                None,
            )
        );
        Ok(())
    }

    #[test]
    fn parses_multi_pair_info() {
        assert_eq!(parse_partial_file_info_max("2:1|3:2|4:3"), Some(4));
    }

    #[test]
    fn parses_single_pair_info() {
        assert_eq!(parse_partial_file_info_max("7:100"), Some(7));
    }

    #[test]
    fn malformed_info_is_none() {
        assert_eq!(parse_partial_file_info_max(""), None);
        assert_eq!(parse_partial_file_info_max("nonsense"), None);
    }

    /// The DDL a paged listing reads, minus every table it does not touch.
    const PAGE_LISTING_SCHEMA: &str = "
        CREATE TABLE ducklake_data_file (
            data_file_id BIGINT, table_id BIGINT, begin_snapshot BIGINT,
            end_snapshot BIGINT, path VARCHAR, path_is_relative BOOLEAN,
            file_size_bytes BIGINT, footer_size BIGINT, encryption_key VARCHAR,
            row_id_start BIGINT, record_count BIGINT, mapping_id BIGINT);
        CREATE TABLE ducklake_delete_file (
            delete_file_id BIGINT, data_file_id BIGINT, table_id BIGINT,
            begin_snapshot BIGINT, end_snapshot BIGINT, path VARCHAR,
            path_is_relative BOOLEAN, file_size_bytes BIGINT, footer_size BIGINT,
            encryption_key VARCHAR, delete_count BIGINT);
        CREATE TABLE ducklake_file_partition_value (
            data_file_id BIGINT, table_id BIGINT, partition_key_index BIGINT,
            partition_value VARCHAR);
        CREATE TABLE ducklake_file_column_stats (
            data_file_id BIGINT, table_id BIGINT, column_id BIGINT,
            column_size_bytes BIGINT, value_count BIGINT, null_count BIGINT,
            min_value VARCHAR, max_value VARCHAR, contains_nan BOOLEAN);";

    /// A file-backed catalog built by `setup`, then opened by the provider.
    ///
    /// The provider opens read-only, so the writing connection is closed before
    /// it is constructed. The `TempDir` comes back with it because dropping it
    /// deletes the catalog.
    fn provider_over(setup: &str) -> (TempDir, DuckdbMetadataProvider) {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("catalog.duckdb");
        {
            let conn = Connection::open(&path).expect("writable catalog");
            conn.execute_batch(PAGE_LISTING_SCHEMA)
                .expect("catalog schema");
            conn.execute_batch(setup).expect("catalog fixture");
        }
        let provider = DuckdbMetadataProvider::new(path.to_string_lossy().to_string())
            .expect("read-only provider");
        (dir, provider)
    }

    /// Lower `a <op> value` on an `INT32` column whose `column_id` is 7.
    fn int32_filter(operator: Operator, value: i32) -> StatsFilter {
        let column = Arc::new(Column::new("a", 0)) as Arc<dyn PhysicalExpr>;
        let predicate =
            Arc::new(BinaryExpr::new(column, operator, lit(value))) as Arc<dyn PhysicalExpr>;
        let schema = Schema::new(vec![Field::new("a", DataType::Int32, true)]);
        let columns = vec![DuckLakeTableColumn::new(7, "a".to_string(), "int32".to_string(), true)];
        lower_predicate(&predicate, &schema, &columns).expect("predicate lowers")
    }

    /// A temporal constant outside the fixed-width four-digit-year encoding is
    /// declined rather than rendered.
    ///
    /// `TRY_CAST` protects the *stat*, not the constant. The constant is spliced
    /// bare on the far side of the comparison and DuckDB converts it eagerly, so
    /// `chrono`'s `+12921-08-18` — what `stats_encode` writes for a year past
    /// 9999 — raises a conversion error and takes the whole listing query with
    /// it. `is_missing_statistics_table` does not recognise that error, so
    /// before this guard the scan failed outright on a predicate that planned
    /// fine without pushdown, and `files_matching` takes an arbitrary
    /// `PhysicalExpr` with no SQL parser in the way.
    #[test]
    fn out_of_range_temporal_constants_are_declined() {
        let renders = |value: ScalarValue, ducklake_type: &str| {
            let data_type = value.data_type();
            let column = Arc::new(Column::new("t", 0)) as Arc<dyn PhysicalExpr>;
            let predicate = Arc::new(BinaryExpr::new(
                column,
                Operator::Lt,
                datafusion::physical_expr::expressions::lit(value),
            )) as Arc<dyn PhysicalExpr>;
            let schema = Schema::new(vec![Field::new("t", data_type, true)]);
            let columns =
                vec![DuckLakeTableColumn::new(7, "t".to_string(), ducklake_type.to_string(), true)];
            lower_predicate(&predicate, &schema, &columns)
                .expect("predicate lowers")
                .render(&DuckdbStatsDialect)
        };

        // 19_723 days after the epoch is 2024-01-01; 4_000_000 is +12921-08-18.
        let canonical = renders(ScalarValue::Date32(Some(19_723)), "date")
            .expect("a canonical date still pushes down");
        assert!(
            canonical[0].condition.contains(
                "CASE WHEN regexp_full_match(col_7_stats.min_value, '^[0-9]{4}-[0-9]{2}-[0-9]{2}$') \
                 THEN TRY_CAST(col_7_stats.min_value AS DATE) END < '2024-01-01'"
            ),
            "unexpected condition: {}",
            canonical[0].condition
        );
        assert!(renders(ScalarValue::Date32(Some(4_000_000)), "date").is_none());

        // The same for both timestamp spellings. 3.45e17 microseconds after the
        // epoch is in the year 12903, which `stats_encode` also signs.
        let far = 345_000_000_000_000_000;
        assert!(
            renders(
                ScalarValue::TimestampMicrosecond(Some(1_700_000_000_000_000), None),
                "timestamp"
            )
            .is_some()
        );
        assert!(
            renders(
                ScalarValue::TimestampMicrosecond(Some(far), None),
                "timestamp"
            )
            .is_none()
        );
        assert!(
            renders(
                ScalarValue::TimestampMicrosecond(Some(1_700_000_000_000_000), Some("UTC".into())),
                "timestamptz"
            )
            .is_some()
        );
        assert!(
            renders(
                ScalarValue::TimestampMicrosecond(Some(far), Some("UTC".into())),
                "timestamptz"
            )
            .is_none()
        );

        // Why it has to be declined, straight from DuckDB — and why the
        // missing-statistics-table fallback was never going to catch it.
        let conn = Connection::open_in_memory().expect("in-memory DuckDB");
        let error = conn
            .query_row(
                "SELECT TRY_CAST('2024-01-01' AS DATE) < '+12921-08-18'",
                [],
                |row| row.get::<_, bool>(0),
            )
            .expect_err("DuckDB converts the constant eagerly and raises");
        assert!(error.to_string().contains("+12921-08-18"), "{error}");
        assert!(!is_missing_statistics_table(&error), "{error}");
    }

    /// Any error from the narrowed listing retries unfiltered, not just the
    /// missing statistics table.
    ///
    /// The rendered predicate is the one part of the listing query built from a
    /// caller's arbitrary expression, so it is where an unanticipated shape
    /// reaches SQL — Finding 1 was exactly that, an error message no
    /// `is_missing_statistics_table` pattern would ever match. The filter only
    /// ever narrows the result, so dropping it costs planning time; failing
    /// here fails a scan that would have planned fine.
    ///
    /// The fault is injected rather than found: a `StatsColumnFilter` with a
    /// negative `column_id` renders the CTE alias `col_-1_stats`, which DuckDB
    /// refuses to parse. That is a parser error, so it stands in for the class
    /// of failure the narrow guard misses.
    #[test]
    fn filtered_listing_retries_unfiltered_after_any_error() {
        let (_dir, provider) = provider_over(
            "INSERT INTO ducklake_data_file VALUES
                 (1, 3, 0, NULL, 'low.parquet', true, 10, 1, NULL, 0, 100, NULL),
                 (2, 3, 0, NULL, 'high.parquet', true, 10, 1, NULL, 100, 100, NULL);",
        );
        let filter = StatsFilter {
            columns: vec![StatsColumnFilter {
                column_id: -1,
                referenced_stats: BTreeSet::from([StatKind::ValueCount]),
                needs_value_count_guard: false,
                condition: StatsExpr::CountPositive(StatKind::ValueCount),
            }],
        };
        let rendered = filter
            .render(&DuckdbStatsDialect)
            .expect("the broken filter still renders");

        let conn = Connection::open_in_memory().expect("in-memory DuckDB");
        conn.execute_batch(PAGE_LISTING_SCHEMA)
            .expect("catalog schema");
        let error = query_data_file_page(&conn, 3, 0, i64::MIN, 10, &rendered)
            .expect_err("DuckDB cannot parse the alias");
        // The point of the widening: the old guard would have re-raised this.
        assert!(
            !is_missing_statistics_table(&error),
            "this error is one the narrow guard already caught: {error}"
        );

        let files = provider
            .get_table_file_metadata_page_filtered(3, 0, None, 10, Some(&filter))
            .expect("the listing survives a filter it cannot run");
        assert_eq!(
            files
                .iter()
                .map(|file| file.file.data_file_id)
                .collect::<Vec<_>>(),
            vec![1, 2]
        );
    }

    /// A selective filter must not read the statistics of the files it pruned.
    ///
    /// The two enrichment queries used to be scoped `data_file_id > after AND <=
    /// last`, which is bounded by the page size only while nothing narrows the
    /// listing. With a filter the surviving ids are sparse: a single match at
    /// the far end of a large table puts `last` near the table maximum, and the
    /// first page's statistics query then returns every stats row below it.
    ///
    /// Row *counts* are not observable from the return value, so the pruned
    /// files carry a stats row that cannot be decoded: a NULL `column_id`, which
    /// `DuckLakeFileColumnStatistics` reads into an `i64`. Reading one is an
    /// error, so the filtered page succeeding is proof it read none of them —
    /// and the unfiltered page below, whose range does cover them, fails, which
    /// is what makes this discriminate rather than pass vacuously.
    #[test]
    fn filtered_page_reads_no_statistics_for_pruned_files() {
        let (_dir, provider) = provider_over(
            "INSERT INTO ducklake_data_file VALUES
                 (1, 3, 0, NULL, 'f1.parquet', true, 10, 1, NULL, 0, 100, NULL),
                 (2, 3, 0, NULL, 'f2.parquet', true, 10, 1, NULL, 100, 100, NULL),
                 (3, 3, 0, NULL, 'f3.parquet', true, 10, 1, NULL, 200, 100, NULL),
                 (4, 3, 0, NULL, 'f4.parquet', true, 10, 1, NULL, 300, 100, NULL),
                 (5, 3, 0, NULL, 'f5.parquet', true, 10, 1, NULL, 400, 100, NULL),
                 (6, 3, 0, NULL, 'f6.parquet', true, 10, 1, NULL, 500, 100, NULL);
             -- Files 1..=5 cannot hold a value above 50; file 6 can. The match
             -- is last, so the old range covered every pruned file.
             INSERT INTO ducklake_file_column_stats VALUES
                 (1, 3, 7, 8, 100, 0, '0', '10', NULL),
                 (2, 3, 7, 8, 100, 0, '0', '10', NULL),
                 (3, 3, 7, 8, 100, 0, '0', '10', NULL),
                 (4, 3, 7, 8, 100, 0, '0', '10', NULL),
                 (5, 3, 7, 8, 100, 0, '0', '10', NULL),
                 (6, 3, 7, 8, 100, 0, '100', '200', NULL);
             -- The undecodable rows. A NULL column_id also keeps them out of
             -- the filter's own CTE, which selects column_id = 7, so they
             -- change nothing about which files survive.
             INSERT INTO ducklake_file_column_stats VALUES
                 (1, 3, NULL, 8, 100, 0, '0', '10', NULL),
                 (2, 3, NULL, 8, 100, 0, '0', '10', NULL),
                 (3, 3, NULL, 8, 100, 0, '0', '10', NULL),
                 (4, 3, NULL, 8, 100, 0, '0', '10', NULL),
                 (5, 3, NULL, 8, 100, 0, '0', '10', NULL);",
        );

        let filter = int32_filter(Operator::Gt, 50);
        let files = provider
            .get_table_file_metadata_page_filtered(3, 0, None, 100, Some(&filter))
            .expect("the filtered page reads statistics only for the files it returned");
        assert_eq!(
            files
                .iter()
                .map(|file| file.file.data_file_id)
                .collect::<Vec<_>>(),
            vec![6]
        );

        // The same call with no filter returns every file, so its range does
        // reach the planted rows and it fails. That is the behaviour the
        // filtered call would have had while it scoped by range.
        provider
            .get_table_file_metadata_page(3, 0, None, 100)
            .expect_err("the unfiltered range reaches the undecodable rows");
    }
}
