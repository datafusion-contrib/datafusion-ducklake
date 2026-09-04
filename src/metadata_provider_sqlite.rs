//! SQLite metadata provider for DuckLake catalogs.

use crate::Result;
use crate::metadata_provider::{
    ColumnWithTable, DataFileChange, DeleteFileChange, DuckLakeFileColumnStatistics,
    DuckLakeFileData, DuckLakeFileMetadata, DuckLakeInlinedData, DuckLakeInlinedDelete,
    DuckLakeNameMapping, DuckLakeNameMappingEntry, DuckLakeStatistics, DuckLakeTableColumn,
    DuckLakeTableColumnStatistics, DuckLakeTableField, DuckLakeTableFile, DuckLakeTableStatistics,
    FileWithTable, MetadataProvider, MetadataSetting, SQL_GET_FILE_PARTITION_VALUES,
    SQL_GET_NAME_MAPPING, SQL_GET_PARTITION_SPEC, SQL_GET_SORT_SPEC, SQL_GET_TABLE_COLUMNS,
    SchemaMetadata, SnapshotChangeMetadata, SnapshotMetadata, TableMetadata, TableWithSchema,
    ViewMetadata, ViewWithSchema, block_on, inlined_delete_table_name, inlined_missing_scalar,
    reconstruct_columns, reconstruct_columns_with_table, resolve_metadata_settings,
};
use crate::partition::PartitionSpec;
use crate::sort::SortSpec;
use crate::stats_encode::{is_canonical_date, is_canonical_timestamp, is_canonical_timestamptz};
use crate::stats_filter::{StatsFilter, StatsLiteral, StatsSqlDialect};
use arrow::array::{
    ArrayRef, BinaryArray, BinaryViewArray, BooleanArray, Float32Array, Float64Array, Int8Array,
    Int16Array, Int32Array, Int64Array, LargeBinaryArray, RecordBatch, StringViewArray, UInt8Array,
    UInt16Array, UInt32Array, UInt64Array,
};
use arrow::datatypes::{DataType, SchemaRef};
use sqlx::AssertSqlSafe;
use sqlx::Row;
use sqlx::sqlite::{SqlitePool, SqlitePoolOptions, SqliteRow};
use sqlx::types::chrono::NaiveDateTime;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, OnceLock};

/// Quote a SQL identifier for SQLite (double-quote, doubling embedded quotes),
/// so catalog-supplied inlined-table / column names can't break the query.
fn quote_ident(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

fn decode_view(row: &SqliteRow) -> Result<ViewMetadata> {
    Ok(ViewMetadata {
        view_id: row.try_get(0)?,
        schema_id: row.try_get(1)?,
        begin_snapshot: row.try_get(2)?,
        view_name: row.try_get(3)?,
        dialect: row.try_get(4)?,
        sql: row.try_get(5)?,
        column_aliases: row.try_get(6)?,
    })
}

fn decode_table_file(row: &SqliteRow, snapshot_id: i64) -> Result<DuckLakeTableFile> {
    let data_file = DuckLakeFileData {
        path: row.try_get(1)?,
        path_is_relative: row.try_get(2)?,
        file_size_bytes: row.try_get(3)?,
        footer_size: row.try_get(4)?,
        encryption_key: row.try_get(5)?,
        mapping_id: row.try_get(19).unwrap_or(None),
    };
    let (delete_file, delete_count) = if row.try_get::<Option<i64>, _>(8)?.is_some() {
        (
            Some(DuckLakeFileData {
                path: row.try_get(9)?,
                path_is_relative: row.try_get(10)?,
                file_size_bytes: row.try_get(11)?,
                footer_size: row.try_get(12)?,
                encryption_key: row.try_get(13)?,
                mapping_id: None,
            }),
            row.try_get(14)?,
        )
    } else {
        (None, None)
    };
    Ok(DuckLakeTableFile {
        data_file_id: row.try_get(0)?,
        file: data_file,
        delete_file_id: row.try_get(8)?,
        delete_file,
        row_id_start: row.try_get(6)?,
        snapshot_id: Some(snapshot_id),
        begin_snapshot: row.try_get(15)?,
        schema_version: row.try_get(17)?,
        partial_max: row.try_get(16)?,
        max_row_count: row.try_get(7)?,
        delete_count,
        // Column 18 is present on the select-path query (which projects
        // `data.partition_id`) and absent on callers that share this decoder
        // without it; `try_get` failing there degrades to `None`, matching the
        // pre-partition behaviour. Per-key values are filled in by the caller.
        partition_id: row.try_get(18).unwrap_or(None),
        partition_values: Vec::new(),
    })
}

/// Build one Arrow [`RecordBatch`] (in `schema`, the table's physical schema)
/// from inlined rows fetched out of a `ducklake_inlined_data_*` table. `present`
/// is the set of the physical table's data-column names; a table column absent
/// from it (added after this inlined table's schema version) is null-filled.
/// Errors on a column type not yet supported for inlined reads (loud, never
/// silent) — inlined values for those types must be flushed to Parquet first.
fn build_inlined_batch(
    schema: &SchemaRef,
    columns: &[DuckLakeTableColumn],
    present: &HashSet<String>,
    rows: &[sqlx::sqlite::SqliteRow],
) -> Result<RecordBatch> {
    let n = rows.len();
    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(columns.len());
    for (i, col) in columns.iter().enumerate() {
        let dt = schema.field(i).data_type();
        let name = col.column_name.as_str();
        if !present.contains(name) {
            arrays.push(inlined_missing_scalar(col, dt)?.to_array_of_size(n)?);
            continue;
        }
        // SQLite stores INTEGER as i64 and REAL as f64; read at that width and
        // narrow/convert to the catalog's declared Arrow type.
        macro_rules! ints {
            ($arr:ty, $t:ty) => {{
                let mut b = Vec::with_capacity(n);
                for r in rows {
                    b.push(r.try_get::<Option<i64>, _>(name)?.map(|v| v as $t));
                }
                Arc::new(<$arr>::from(b)) as ArrayRef
            }};
        }
        let array: ArrayRef = match dt {
            DataType::Int8 => ints!(Int8Array, i8),
            DataType::Int16 => ints!(Int16Array, i16),
            DataType::Int32 => ints!(Int32Array, i32),
            DataType::Int64 => ints!(Int64Array, i64),
            DataType::UInt8 => ints!(UInt8Array, u8),
            DataType::UInt16 => ints!(UInt16Array, u16),
            DataType::UInt32 => ints!(UInt32Array, u32),
            DataType::UInt64 => {
                let mut values = Vec::with_capacity(n);
                for row in rows {
                    let value = row.try_get::<Option<String>, _>(name)?;
                    values.push(
                        value
                            .map(|value| value.parse::<u64>())
                            .transpose()
                            .map_err(|e| {
                                crate::DuckLakeError::InvalidConfig(format!(
                                    "invalid inlined UInt64 value for column '{name}': {e}"
                                ))
                            })?,
                    );
                }
                Arc::new(UInt64Array::from(values)) as ArrayRef
            },
            DataType::Float32 => {
                let mut b = Vec::with_capacity(n);
                for r in rows {
                    b.push(r.try_get::<Option<f64>, _>(name)?.map(|v| v as f32));
                }
                Arc::new(Float32Array::from(b)) as ArrayRef
            },
            DataType::Float64 => {
                let mut b = Vec::with_capacity(n);
                for r in rows {
                    b.push(r.try_get::<Option<f64>, _>(name)?);
                }
                Arc::new(Float64Array::from(b)) as ArrayRef
            },
            DataType::Utf8 => {
                let mut b: Vec<Option<String>> = Vec::with_capacity(n);
                for r in rows {
                    b.push(r.try_get::<Option<String>, _>(name)?);
                }
                Arc::new(arrow::array::StringArray::from(b)) as ArrayRef
            },
            DataType::Utf8View => {
                let mut values: Vec<Option<String>> = Vec::with_capacity(n);
                for row in rows {
                    values.push(row.try_get::<Option<String>, _>(name)?);
                }
                Arc::new(values.into_iter().collect::<StringViewArray>()) as ArrayRef
            },
            DataType::Boolean => {
                let mut b = Vec::with_capacity(n);
                for r in rows {
                    b.push(r.try_get::<Option<i64>, _>(name)?.map(|v| v != 0));
                }
                Arc::new(BooleanArray::from(b)) as ArrayRef
            },
            DataType::Binary => {
                let mut b: Vec<Option<Vec<u8>>> = Vec::with_capacity(n);
                for r in rows {
                    b.push(r.try_get::<Option<Vec<u8>>, _>(name)?);
                }
                Arc::new(BinaryArray::from(
                    b.iter().map(|o| o.as_deref()).collect::<Vec<_>>(),
                )) as ArrayRef
            },
            DataType::LargeBinary => {
                let mut values: Vec<Option<Vec<u8>>> = Vec::with_capacity(n);
                for row in rows {
                    values.push(row.try_get::<Option<Vec<u8>>, _>(name)?);
                }
                Arc::new(LargeBinaryArray::from(
                    values
                        .iter()
                        .map(|value| value.as_deref())
                        .collect::<Vec<_>>(),
                )) as ArrayRef
            },
            DataType::BinaryView => {
                let mut values: Vec<Option<Vec<u8>>> = Vec::with_capacity(n);
                for row in rows {
                    values.push(row.try_get::<Option<Vec<u8>>, _>(name)?);
                }
                Arc::new(values.into_iter().collect::<BinaryViewArray>()) as ArrayRef
            },
            DataType::FixedSizeBinary(size) if *size != 16 => {
                let values = rows
                    .iter()
                    .map(|row| -> Result<datafusion::common::ScalarValue> {
                        Ok(match row.try_get::<Option<Vec<u8>>, _>(name)? {
                            Some(value) if value.len() == *size as usize => {
                                datafusion::common::ScalarValue::FixedSizeBinary(*size, Some(value))
                            },
                            Some(value) => {
                                return Err(crate::DuckLakeError::InvalidConfig(format!(
                                    "inlined data column '{name}' expected {size} bytes, was {}",
                                    value.len(),
                                )));
                            },
                            None => datafusion::common::ScalarValue::FixedSizeBinary(*size, None),
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                datafusion::common::ScalarValue::iter_to_array(values.into_iter())?
            },
            other => {
                let values = rows
                    .iter()
                    .map(|row| -> Result<datafusion::common::ScalarValue> {
                        let value = row.try_get::<Option<String>, _>(name)?;
                        match value {
                            Some(value) => crate::types::parse_ducklake_scalar(&value, other)
                                .ok_or_else(|| {
                                    crate::DuckLakeError::Unsupported(format!(
                                        "inlined data column '{name}' type {other:?} cannot decode \
                                 '{value}'; {}",
                                        crate::metadata_provider::INLINED_DATA_REMEDIATION,
                                    ))
                                }),
                            None => Ok(datafusion::common::ScalarValue::try_from(other)?),
                        }
                    })
                    .collect::<Result<Vec<_>>>()?;
                datafusion::common::ScalarValue::iter_to_array(values.into_iter())?
            },
        };
        arrays.push(array);
    }
    Ok(RecordBatch::try_new(schema.clone(), arrays)?)
}

fn is_missing_statistics_table(error: &sqlx::Error) -> bool {
    let message = error.to_string().to_ascii_lowercase();
    message.contains("no such table") || message.contains("does not exist")
}

/// SQLite spelling of the statistics comparisons in [`crate::stats_filter`].
///
/// SQLite has no `TRY_CAST`, and its `CAST` never fails: `CAST('abc' AS REAL)`
/// is `0.0` and `CAST('abc' AS INTEGER)` is `0`. An unconditional cast would
/// therefore read a malformed bound as zero and could prune a file that matches,
/// so every cast below is guarded by a test that the text really is the number
/// it is about to be read as, and yields SQL `NULL` when it is not.
/// [`StatsSqlDialect::keep_when_unknown`] turns that `NULL` back into "keep the
/// file", so a bound this dialect will not read prunes nothing.
struct SqliteStatsDialect;

impl StatsSqlDialect for SqliteStatsDialect {
    /// A type not listed here is declined, which drops the comparison and
    /// prunes nothing. `DECIMAL` is the notable one: SQLite's only numeric with
    /// a fractional part is `REAL`, and a `DECIMAL(38, s)` constant can carry
    /// more significant digits than a double, so two decimals this engine
    /// orders can round to one value and compare equal.
    ///
    /// Temporal types are accepted only when *both* sides are canonically
    /// encoded, which is why the constant is inspected here. SQLite has no
    /// temporal type, so the comparison stays in the text domain, where order
    /// matches chronology only for the fixed-width four-digit-year encoding:
    /// `chrono` renders a year past 9999 as `+12345` and a negative one as
    /// `-0044`, and `+` and `-` both sort below every digit, so one of those on
    /// either side would invert the comparison. The stat is checked in SQL and
    /// the constant right here.
    fn try_cast(&self, expr: &str, literal: &StatsLiteral, data_type: &DataType) -> Option<String> {
        match data_type {
            DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64 => {
                // Round-tripping back to TEXT proves the cast consumed the
                // whole string and lost nothing: `'abc'` and `''` return as
                // `'0'`, `'1.5'` as `'1'`, and a value past `i64::MAX`
                // saturates, so all of them fail the comparison. Integer to
                // text is exact in SQLite, so a value that does round-trip is
                // compared exactly, including the full 64-bit range.
                Some(format!(
                    "CASE WHEN {expr} = CAST(CAST({expr} AS INTEGER) AS TEXT) \
                     THEN CAST({expr} AS INTEGER) END"
                ))
            },
            // That round trip cannot vet a REAL: SQLite renders a double back to
            // text with 15 significant digits, so an exact cast of
            // `0.30000000000000004` still returns as `0.3`. The text is checked
            // directly instead. `CAST(... AS REAL)` parses a well-formed decimal
            // to the nearest double, which is what `stats_encode` wrote and what
            // SQLite reads the constant as, so the comparison is exact.
            DataType::Float32 | DataType::Float64 => Some(format!(
                "CASE WHEN {} THEN CAST({expr} AS REAL) END",
                sqlite_is_float_text(expr)
            )),
            // A boolean stat is the text `true` / `false` and the constant is
            // rendered quoted, so comparing them as text is both well-defined
            // and correctly ordered ('false' < 'true'). Any other spelling is
            // not this encoding, and reading it as a boolean would be a guess.
            DataType::Boolean => Some(sqlite_guarded_text(
                expr,
                &format!("{} IN ('true', 'false')", self.collate_binary(expr)),
            )),
            // `YYYY-MM-DD`: fixed width, so text order is date order.
            DataType::Date32 if is_canonical_date(literal.text()) => {
                Some(sqlite_guarded_text(expr, &sqlite_is_canonical_date(expr)))
            },
            // `YYYY-MM-DD HH:MM:SS[.fff]`. The time unit does not matter — what
            // matters is the shape both sides are written in. A shorter string
            // is a prefix of a longer one and so sorts below it, which is right:
            // no fraction is a zero fraction, and `stats_encode` trims trailing
            // zeros so `.5` and `.50` cannot both occur.
            DataType::Timestamp(_, None) if is_canonical_timestamp(literal.text()) => Some(
                sqlite_guarded_text(expr, &sqlite_is_canonical_timestamp(expr)),
            ),
            // The same, plus the `+00` suffix `stats_encode` appends after
            // normalizing to UTC. The suffix is constant across everything the
            // guard admits, so it does not disturb the ordering; a catalog
            // written at another offset is declined rather than compared, since
            // `12:00:00+01` sorts above `12:00:00+00` and is earlier.
            DataType::Timestamp(_, Some(_)) if is_canonical_timestamptz(literal.text()) => Some(
                sqlite_guarded_text(expr, &sqlite_is_canonical_timestamptz(expr)),
            ),
            _ => None,
        }
    }

    /// SQLite's default collation is already byte-wise, but a catalog is free to
    /// declare `min_value` / `max_value` `COLLATE NOCASE`, and a comparison
    /// takes its collation from the column when the other side is a plain
    /// literal. Naming it keeps the comparison byte-wise like DataFusion's.
    fn collate_binary(&self, expr: &str) -> String {
        format!("{expr} COLLATE BINARY")
    }

    /// `contains_nan` is stored as SQLite's `0` / `1`, so this compares against
    /// `0` rather than a boolean keyword. Any other stored spelling compares
    /// unequal to `0`, which reads as "NaN state unknown" and keeps the file.
    fn boolean_is_not_false(&self, expr: &str) -> String {
        format!("{expr} IS NULL OR {expr} <> 0")
    }
}

/// SQL that is true only when `expr` holds text that `CAST(... AS REAL)` reads
/// as exactly the number it spells: an optional leading `-`, then digits around
/// a single `.`, with at least one digit present.
///
/// `CAST` stops at the first character it cannot use, so `'1.2.3'` becomes
/// `1.2`, `'1e'` becomes `1.0` and `'abc'` becomes `0.0`; only a whole-string
/// match rules those out. Exponent notation — which `stats_encode` emits outside
/// `[1e-4, 1e16)` — and `inf` deliberately do not match: those bounds prune
/// nothing rather than risk a partial parse.
/// Whether a stat is a float exactly as [`crate::stats_encode`] writes one, in
/// either notation it uses.
///
/// SQLite has no regular expressions in a stock build, so both shapes are
/// pinned with `GLOB`, `instr` and `substr`.
fn sqlite_is_float_text(expr: &str) -> String {
    format!(
        "(({}) OR ({}))",
        sqlite_is_fixed_decimal_text(expr),
        sqlite_is_scientific_text(expr)
    )
}

/// The scientific notation `stats_encode` writes for any magnitude outside
/// `[1e-4, 1e16)`: a mantissa, `e`, a sign, then digits.
///
/// The shape has to be checked because `CAST` stops at the first byte it cannot
/// use rather than failing — `CAST('1e' AS REAL)` is `1.0` and
/// `CAST('1e+2.5' AS REAL)` is `100.0`, both of which would compare as a value
/// the file does not contain. Everything this admits casts to exactly the double
/// `stats_encode` wrote, which is also how SQLite reads the constant, so the
/// comparison is exact.
///
/// `inf` and `-inf` are not admitted here and are not meant to be: SQLite reads
/// either as `0.0`, so a file carrying one contributes no usable bound and is
/// kept.
fn sqlite_is_scientific_text(expr: &str) -> String {
    let e = format!("instr({expr}, 'e')");
    let mantissa = format!("substr({expr}, 1, {e} - 1)");
    let exponent = format!("substr({expr}, {e} + 2)");
    format!(
        // A mantissa before the `e`, by the same reasoning as the fixed form
        // except that the dot is optional — `1e+20` has none — so stripping
        // digits from both ends must leave either nothing or exactly the dot.
        "{e} > 1 \
         AND {mantissa} GLOB '[-0-9]*' AND NOT ({mantissa} GLOB '?*-*') \
         AND ltrim({mantissa}, '-') GLOB '*[0-9]*' \
         AND rtrim(ltrim(ltrim({mantissa}, '-'), '0123456789'), '0123456789') IN ('', '.') \
         AND substr({expr}, {e} + 1, 1) GLOB '[-+]' \
         AND length({expr}) > {e} + 1 \
         AND {exponent} GLOB '[0-9]*' AND NOT ({exponent} GLOB '*[^0-9]*')"
    )
}

fn sqlite_is_fixed_decimal_text(expr: &str) -> String {
    // `ltrim(x, '-')` drops the sign, and the `?*-*` test rejects a `-` anywhere
    // but the front so a second one cannot survive that. Stripping leading and
    // then trailing digits from what is left must expose exactly the `.`, which
    // is what pins the shape to one dot with digits around it.
    format!(
        "{expr} GLOB '[-0-9]*' AND NOT ({expr} GLOB '?*-*') \
         AND ltrim({expr}, '-') GLOB '*[0-9]*' \
         AND rtrim(ltrim(ltrim({expr}, '-'), '0123456789'), '0123456789') = '.'"
    )
}

/// `expr` compared as text, but only where `guard` holds — SQL `NULL`
/// elsewhere, which keeps the file.
///
/// The collation is named on the `CASE` rather than inside it so it applies to
/// the value the comparison actually sees.
fn sqlite_guarded_text(expr: &str, guard: &str) -> String {
    format!("CASE WHEN {guard} THEN {expr} END COLLATE BINARY")
}

/// GLOB pattern for a canonical `YYYY-MM-DD`, four digits of year included.
const SQLITE_DATE_GLOB: &str = "[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]";

/// SQL that is true only when `expr` holds a canonical DuckLake date.
fn sqlite_is_canonical_date(expr: &str) -> String {
    format!("{expr} GLOB '{SQLITE_DATE_GLOB}'")
}

/// SQL that is true only when `expr` holds a canonical DuckLake timestamp:
/// `YYYY-MM-DD HH:MM:SS`, optionally followed by `.` and digits that do not end
/// in `0`.
///
/// The first 19 characters are pinned by the pattern, so every accepted value
/// has its fields in the same fixed-width positions. The fractional part is
/// checked separately because GLOB cannot say "one or more digits": position 20
/// must be a `.` followed by at least one digit, everything after it must be a
/// digit, and a trailing `0` is refused so that `.5` and `.50` — equal values
/// with different text order — cannot both be admitted.
fn sqlite_is_canonical_timestamp(expr: &str) -> String {
    format!(
        "{expr} GLOB '{SQLITE_DATE_GLOB} [0-9][0-9]:[0-9][0-9]:[0-9][0-9]*' \
         AND (substr({expr}, 20) = '' \
              OR (substr({expr}, 20) GLOB '.[0-9]*' \
                  AND ltrim(substr({expr}, 21), '0123456789') = '' \
                  AND NOT ({expr} GLOB '*0')))"
    )
}

/// SQL that is true only when `expr` holds a canonical DuckLake UTC timestamp:
/// a canonical timestamp with `stats_encode`'s `+00` suffix.
fn sqlite_is_canonical_timestamptz(expr: &str) -> String {
    // Check the timestamp on the value with the suffix removed. Truncating a
    // string shorter than the suffix yields '', which fails the shape test, and
    // the suffix test has already required the three characters anyway.
    let body = format!("substr({expr}, 1, length({expr}) - 3)");
    format!(
        "{expr} GLOB '*+00' AND {}",
        sqlite_is_canonical_timestamp(&body)
    )
}

/// Whether `text` is `YYYY-MM-DD` with a four-digit year.
/// The SQL a lowered statistics filter contributes to the file-listing query.
struct StatsFilterSql {
    /// `WITH col_<id>_stats AS (...), ...`, newline-terminated so it can be
    /// prefixed straight onto the `SELECT`.
    cte: String,
    /// One `LEFT JOIN` per column CTE.
    joins: String,
    /// The rendered per-column conditions, each already prefixed with `AND`.
    conditions: String,
}

/// Render `filter` for SQLite, or `None` when it contributes nothing.
///
/// Adds no bind parameters: [`crate::stats_filter`] inlines every literal, and
/// the only other values spliced in are `i64`s this process computed. The
/// caller's parameter list and its order are therefore untouched.
fn stats_filter_sql(filter: Option<&StatsFilter>, table_id: i64) -> Option<StatsFilterSql> {
    let rendered = filter?.render(&SqliteStatsDialect)?;
    let mut cte = String::from("WITH ");
    let mut joins = String::new();
    let mut conditions = String::new();
    for (index, column) in rendered.iter().enumerate() {
        if index > 0 {
            cte.push_str(",\n     ");
        }
        let alias = &column.alias;
        let stats = column.stats.join(", ");
        let column_id = column.column_id;
        cte.push_str(&format!(
            "{alias} AS (SELECT data_file_id, {stats}
                 FROM ducklake_file_column_stats
                 WHERE column_id = {column_id} AND table_id = {table_id})"
        ));
        joins.push_str(&format!(
            "
                 LEFT JOIN {alias} ON {alias}.data_file_id = data.data_file_id"
        ));
        // The condition arrives already wrapped in its no-stats, per-stat and
        // unknown guards, so it is spliced verbatim.
        conditions.push_str(&format!(
            "
                   AND {}",
            column.condition
        ));
    }
    cte.push('\n');
    Some(StatsFilterSql {
        cte,
        joins,
        conditions,
    })
}

/// Optional catalog-schema capabilities probed before scan / CDC / inlined-data
/// queries.
///
/// Minimal / pre-v1.0 catalogs may lack the `partial_max` / `partition_id`
/// columns, the `ducklake_schema_versions` ledger, and the inlined-data
/// registry; the queries degrade the corresponding projections to NULL (or
/// skip inlined-data reads) when a capability is absent.
#[derive(Debug, Clone, Copy)]
struct SchemaCapabilities {
    /// `ducklake_data_file.partial_max` exists.
    data_file_partial_max: bool,
    /// `ducklake_delete_file.partial_max` exists.
    delete_file_partial_max: bool,
    /// `ducklake_data_file.partition_id` exists.
    data_file_partition_id: bool,
    /// The `ducklake_schema_versions` table exists.
    schema_versions: bool,
    /// The `ducklake_inlined_data_tables` registry exists.
    inlined_data_tables: bool,
    /// The `ducklake_view` table exists.
    views: bool,
}

impl SchemaCapabilities {
    fn all(&self) -> bool {
        self.data_file_partial_max
            && self.delete_file_partial_max
            && self.data_file_partition_id
            && self.schema_versions
            && self.inlined_data_tables
            && self.views
    }
}

/// SQLite-based metadata provider for DuckLake catalogs.
#[derive(Debug, Clone)]
pub struct SqliteMetadataProvider {
    pub pool: SqlitePool,
    // Positive-only memo of the optional-schema capability probes. `Arc` so
    // derived `Clone` shares the cache across provider clones.
    schema_capabilities: Arc<OnceLock<SchemaCapabilities>>,
}

impl SqliteMetadataProvider {
    /// Creates a new provider for an existing DuckLake catalog.
    ///
    /// Connection string format: `sqlite:///path/to/catalog.db` or `sqlite::memory:`
    pub async fn new(connection_string: &str) -> Result<Self> {
        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect(connection_string)
            .await?;

        Ok(Self {
            pool,
            schema_capabilities: Arc::new(OnceLock::new()),
        })
    }

    /// Creates a provider over an existing connection pool. Replaces
    /// struct-literal construction, which stopped compiling when the
    /// schema-capability memo field was added.
    pub fn from_pool(pool: SqlitePool) -> Self {
        Self {
            pool,
            schema_capabilities: Arc::new(OnceLock::new()),
        }
    }

    /// Whether the schema-capability memo is populated. Exposed for tests.
    #[doc(hidden)]
    pub fn schema_capabilities_cached(&self) -> bool {
        self.schema_capabilities.get().is_some()
    }

    /// Returns the catalog's optional-schema capabilities, probing at most
    /// once per provider lifetime on a fully-migrated catalog.
    ///
    /// Cache-positive-only: capability existence is monotonic (migrations only
    /// add columns/tables, never drop them), so an all-`true` answer is an
    /// immutable fact and safe to memoize. A `false` answer is never cached —
    /// the next call re-probes, so a mid-flight catalog upgrade is picked up
    /// on the next call exactly like the previous per-call probing. Concurrent
    /// first calls may each probe once (one statement each) — harmless; a
    /// raced `set` is ignored.
    async fn schema_capabilities(&self) -> Result<SchemaCapabilities> {
        if let Some(caps) = self.schema_capabilities.get() {
            return Ok(*caps);
        }
        let row: (bool, bool, bool, bool, bool, bool) = sqlx::query_as(
            "SELECT
               (SELECT COUNT(*) FROM pragma_table_info('ducklake_data_file')
                WHERE name = 'partial_max') > 0,
               (SELECT COUNT(*) FROM pragma_table_info('ducklake_delete_file')
                WHERE name = 'partial_max') > 0,
               (SELECT COUNT(*) FROM pragma_table_info('ducklake_data_file')
                WHERE name = 'partition_id') > 0,
               (SELECT COUNT(*) FROM sqlite_master
                WHERE type = 'table' AND name = 'ducklake_schema_versions') > 0,
               (SELECT COUNT(*) FROM sqlite_master
                WHERE type = 'table' AND name = 'ducklake_inlined_data_tables') > 0,
               (SELECT COUNT(*) FROM sqlite_master
                WHERE type = 'table' AND name = 'ducklake_view') > 0",
        )
        .fetch_one(&self.pool)
        .await?;
        let caps = SchemaCapabilities {
            data_file_partial_max: row.0,
            delete_file_partial_max: row.1,
            data_file_partition_id: row.2,
            schema_versions: row.3,
            inlined_data_tables: row.4,
            views: row.5,
        };
        if caps.all() {
            let _ = self.schema_capabilities.set(caps);
        }
        Ok(caps)
    }

    /// Bind and run one page of the file-listing query.
    ///
    /// The parameter list is identical for the filtered and unfiltered
    /// spellings of that query — `stats_filter_sql` inlines everything it needs
    /// — which is what lets the caller retry with the filter dropped.
    async fn fetch_file_page(
        &self,
        sql: &str,
        table_id: i64,
        snapshot_id: i64,
        after_data_file_id: Option<i64>,
        limit: i64,
    ) -> std::result::Result<Vec<SqliteRow>, sqlx::Error> {
        sqlx::query(AssertSqlSafe(sql))
            .bind(table_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .bind(table_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .bind(after_data_file_id.unwrap_or(i64::MIN))
            .bind(limit)
            .fetch_all(&self.pool)
            .await
    }
}

impl MetadataProvider for SqliteMetadataProvider {
    fn get_current_snapshot(&self) -> Result<i64> {
        block_on(async {
            let row = sqlx::query("SELECT COALESCE(MAX(snapshot_id), 0) FROM ducklake_snapshot")
                .fetch_one(&self.pool)
                .await?;
            Ok(row.try_get(0)?)
        })
    }

    fn get_data_path(&self) -> Result<String> {
        self.get_metadata_settings(None, None)?
            .remove("data_path")
            .ok_or_else(|| {
                crate::error::DuckLakeError::InvalidConfig(
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
    ) -> Result<HashMap<String, String>> {
        block_on(async {
            let has_scope_columns: bool = sqlx::query_scalar(
                "SELECT COUNT(*) = 2 FROM pragma_table_info('ducklake_metadata') \
                 WHERE name IN ('scope', 'scope_id')",
            )
            .fetch_one(&self.pool)
            .await?;
            let rows = if has_scope_columns {
                sqlx::query("SELECT key, value, scope, scope_id FROM ducklake_metadata")
                    .fetch_all(&self.pool)
                    .await?
                    .into_iter()
                    .map(|row| {
                        Ok(MetadataSetting {
                            key: row.try_get(0)?,
                            value: row.try_get(1)?,
                            scope: row.try_get(2)?,
                            scope_id: row.try_get(3)?,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?
            } else {
                sqlx::query("SELECT key, value FROM ducklake_metadata")
                    .fetch_all(&self.pool)
                    .await?
                    .into_iter()
                    .map(|row| {
                        Ok(MetadataSetting {
                            key: row.try_get(0)?,
                            value: row.try_get(1)?,
                            scope: None,
                            scope_id: None,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?
            };
            resolve_metadata_settings(rows, schema_id, table_id)
        })
    }

    fn list_snapshots(&self) -> Result<Vec<SnapshotMetadata>> {
        block_on(async {
            let rows = sqlx::query(
                "SELECT snapshot_id, snapshot_time
                 FROM ducklake_snapshot ORDER BY snapshot_id",
            )
            .fetch_all(&self.pool)
            .await?;

            rows.into_iter()
                .map(|row| {
                    let snapshot_id: i64 = row.try_get(0)?;
                    let timestamp: Option<NaiveDateTime> = row.try_get(1)?;
                    let timestamp_str = timestamp
                        .map(|ts: NaiveDateTime| ts.format("%Y-%m-%d %H:%M:%S%.6f").to_string());

                    Ok(SnapshotMetadata {
                        snapshot_id,
                        timestamp: timestamp_str,
                    })
                })
                .collect()
        })
    }

    fn list_snapshot_changes(&self) -> Result<Vec<SnapshotChangeMetadata>> {
        block_on(async {
            let rows = sqlx::query(
                "SELECT snapshot.snapshot_id,
                        CAST(snapshot.snapshot_time AS TEXT) AS snapshot_time,
                        changes.changes_made,
                        changes.author,
                        changes.commit_message,
                        changes.commit_extra_info
                 FROM ducklake_snapshot AS snapshot
                 JOIN ducklake_snapshot_changes AS changes
                   ON changes.snapshot_id = snapshot.snapshot_id
                 ORDER BY snapshot.snapshot_id",
            )
            .fetch_all(&self.pool)
            .await?;

            rows.into_iter()
                .map(|row| {
                    Ok(SnapshotChangeMetadata {
                        snapshot_id: row.try_get("snapshot_id")?,
                        timestamp: row.try_get("snapshot_time")?,
                        changes_made: row.try_get("changes_made")?,
                        author: row.try_get("author")?,
                        commit_message: row.try_get("commit_message")?,
                        commit_extra_info: row.try_get("commit_extra_info")?,
                    })
                })
                .collect()
        })
    }

    fn find_snapshot_by_commit_extra_info(&self, needle: &str) -> Result<Option<i64>> {
        block_on(async {
            let row = sqlx::query(
                "SELECT changes.snapshot_id
                 FROM ducklake_snapshot_changes AS changes
                 WHERE (changes.commit_extra_info = ?
                        OR instr(changes.commit_extra_info, ?) > 0)
                   AND EXISTS (
                       SELECT 1 FROM ducklake_data_file AS files
                       WHERE files.begin_snapshot = changes.snapshot_id
                         AND files.end_snapshot IS NULL
                   )
                 ORDER BY changes.snapshot_id
                 LIMIT 1",
            )
            .bind(needle)
            .bind(needle)
            .fetch_optional(&self.pool)
            .await?;

            Ok(row.map(|row| row.try_get("snapshot_id")).transpose()?)
        })
    }

    fn list_schemas(&self, snapshot_id: i64) -> Result<Vec<SchemaMetadata>> {
        block_on(async {
            let rows = sqlx::query(
                "SELECT schema_id, schema_name, path, path_is_relative FROM ducklake_schema
                 WHERE ? >= begin_snapshot AND (? < end_snapshot OR end_snapshot IS NULL)",
            )
            .bind(snapshot_id)
            .bind(snapshot_id)
            .fetch_all(&self.pool)
            .await?;

            rows.into_iter()
                .map(|row| {
                    Ok(SchemaMetadata {
                        schema_id: row.try_get(0)?,
                        schema_name: row.try_get(1)?,
                        path: row.try_get(2)?,
                        path_is_relative: row.try_get(3)?,
                    })
                })
                .collect()
        })
    }

    fn list_tables(&self, schema_id: i64, snapshot_id: i64) -> Result<Vec<TableMetadata>> {
        block_on(async {
            let rows = sqlx::query(
                "SELECT table_id, table_name, path, path_is_relative FROM ducklake_table
                 WHERE schema_id = ?
                   AND ? >= begin_snapshot
                   AND (? < end_snapshot OR end_snapshot IS NULL)",
            )
            .bind(schema_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .fetch_all(&self.pool)
            .await?;

            rows.into_iter()
                .map(|row| {
                    Ok(TableMetadata {
                        table_id: row.try_get(0)?,
                        table_name: row.try_get(1)?,
                        path: row.try_get(2)?,
                        path_is_relative: row.try_get(3)?,
                    })
                })
                .collect()
        })
    }

    fn list_views(&self, schema_id: i64, snapshot_id: i64) -> Result<Vec<ViewMetadata>> {
        block_on(async {
            if !self.schema_capabilities().await?.views {
                return Ok(Vec::new());
            }
            let rows = sqlx::query(
                "SELECT view_id, schema_id, begin_snapshot, view_name, dialect, sql, column_aliases
                 FROM ducklake_view
                 WHERE schema_id = ?
                   AND ? >= begin_snapshot
                   AND (? < end_snapshot OR end_snapshot IS NULL)",
            )
            .bind(schema_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .fetch_all(&self.pool)
            .await?;

            rows.into_iter().map(|row| decode_view(&row)).collect()
        })
    }

    fn get_table_structure(
        &self,
        table_id: i64,
        snapshot_id: i64,
    ) -> Result<Vec<DuckLakeTableColumn>> {
        block_on(async {
            let rows = sqlx::query(SQL_GET_TABLE_COLUMNS)
                .bind(table_id)
                .bind(snapshot_id)
                .bind(snapshot_id)
                .fetch_all(&self.pool)
                .await?;

            let raw: Result<Vec<(DuckLakeTableColumn, Option<i64>)>> = rows
                .into_iter()
                .map(|row| {
                    let nulls_allowed: Option<bool> = row.try_get(3)?;
                    let parent_column: Option<i64> = row.try_get(4)?;
                    Ok((
                        DuckLakeTableColumn::new(
                            row.try_get(0)?,
                            row.try_get(1)?,
                            row.try_get(2)?,
                            nulls_allowed.unwrap_or(true),
                        )
                        .with_defaults(
                            row.try_get(5)?,
                            row.try_get(6)?,
                            row.try_get(7)?,
                            row.try_get(8)?,
                        ),
                        parent_column,
                    ))
                })
                .collect();
            reconstruct_columns(raw?)
        })
    }

    fn get_table_fields(&self, table_id: i64, snapshot_id: i64) -> Result<Vec<DuckLakeTableField>> {
        block_on(async {
            let rows = sqlx::query(
                "SELECT column_id, column_name, column_type, nulls_allowed, parent_column
                 FROM ducklake_column
                 WHERE table_id = ?
                   AND ? >= begin_snapshot
                   AND (? < end_snapshot OR end_snapshot IS NULL)
                 ORDER BY column_order",
            )
            .bind(table_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .fetch_all(&self.pool)
            .await?;
            rows.into_iter()
                .map(|row| {
                    Ok(DuckLakeTableField {
                        column_id: row.try_get(0)?,
                        column_name: row.try_get(1)?,
                        column_type: row.try_get(2)?,
                        is_nullable: row.try_get::<Option<bool>, _>(3)?.unwrap_or(true),
                        parent_column: row.try_get(4)?,
                    })
                })
                .collect()
        })
    }

    fn get_name_mapping(&self, mapping_id: i64) -> Result<DuckLakeNameMapping> {
        block_on(async {
            let rows = sqlx::query(SQL_GET_NAME_MAPPING)
                .bind(mapping_id)
                .fetch_all(&self.pool)
                .await?;
            let first = rows.first().ok_or_else(|| {
                crate::DuckLakeError::InvalidConfig(format!(
                    "DuckLake name mapping {mapping_id} does not exist"
                ))
            })?;
            let mut entries = Vec::new();
            for row in &rows {
                if let Some(column_id) = row.try_get::<Option<i64>, _>(3)? {
                    entries.push(DuckLakeNameMappingEntry {
                        column_id,
                        source_name: row.try_get(4)?,
                        target_field_id: row.try_get(5)?,
                        parent_column: row.try_get(6)?,
                        is_partition: row.try_get::<Option<bool>, _>(7)?.unwrap_or(false),
                    });
                }
            }
            Ok(DuckLakeNameMapping {
                mapping_id: first.try_get(0)?,
                table_id: first.try_get(1)?,
                mapping_type: first.try_get(2)?,
                entries,
            })
        })
    }

    fn get_table_files_for_select(
        &self,
        table_id: i64,
        snapshot_id: i64,
    ) -> Result<Vec<DuckLakeTableFile>> {
        block_on(async {
            // Backward compatibility: minimal / pre-v1.0 catalogs may lack the
            // `partial_max` and `partition_id` columns and the
            // `ducklake_schema_versions` ledger. Detect each and degrade those
            // projections to NULL so plain reads still work (all are consumed only
            // by compaction; `partial_max` also by time-travel reads of partial
            // files, which such catalogs never contain).
            let caps = self.schema_capabilities().await?;
            let partial_max_expr = if caps.data_file_partial_max {
                "data.partial_max"
            } else {
                "NULL"
            };
            // Such a catalog holds no partitioned files either, so NULL is exact.
            let partition_id_expr = if caps.data_file_partition_id {
                "data.partition_id"
            } else {
                "NULL"
            };
            let schema_version_expr = if caps.schema_versions {
                "(SELECT sv.schema_version
                  FROM ducklake_schema_versions sv
                  WHERE sv.table_id = data.table_id
                    AND sv.begin_snapshot <= data.begin_snapshot
                  ORDER BY sv.begin_snapshot DESC
                  LIMIT 1)"
            } else {
                "NULL"
            };
            let sql = format!(
                "SELECT
                    data.data_file_id,
                    data.path AS data_file_path,
                    data.path_is_relative AS data_path_is_relative,
                    data.file_size_bytes AS data_file_size,
                    data.footer_size AS data_footer_size,
                    data.encryption_key AS data_encryption_key,
                    data.row_id_start AS data_row_id_start,
                    data.record_count AS data_record_count,
                    del.delete_file_id,
                    del.path AS delete_file_path,
                    del.path_is_relative AS delete_path_is_relative,
                    del.file_size_bytes AS delete_file_size,
                    del.footer_size AS delete_footer_size,
                    del.encryption_key AS delete_encryption_key,
                    del.delete_count,
                    data.begin_snapshot AS data_begin_snapshot,
                    {partial_max_expr} AS data_partial_max,
                    {schema_version_expr} AS data_schema_version,
                    {partition_id_expr} AS data_partition_id,
                    data.mapping_id AS data_mapping_id
                FROM ducklake_data_file AS data
                LEFT JOIN ducklake_delete_file AS del
                    ON data.data_file_id = del.data_file_id
                    AND del.table_id = ?
                    AND ? >= del.begin_snapshot
                    AND (? < del.end_snapshot OR del.end_snapshot IS NULL)
                WHERE data.table_id = ?
                  AND ? >= data.begin_snapshot
                  AND (? < data.end_snapshot OR data.end_snapshot IS NULL)"
            );
            let rows = sqlx::query(AssertSqlSafe(sql.as_str()))
                .bind(table_id)
                .bind(snapshot_id)
                .bind(snapshot_id)
                .bind(table_id)
                .bind(snapshot_id)
                .bind(snapshot_id)
                .fetch_all(&self.pool)
                .await?;

            let mut files: Vec<DuckLakeTableFile> = rows
                .iter()
                .map(|row| decode_table_file(row, snapshot_id))
                .collect::<Result<Vec<_>>>()?;

            // Enrich with per-file partition values. Compaction reads its candidates
            // through this path and must group by, and preserve, each file's exact
            // partition — without the values it would merge across partitions and
            // strip the assignment. Scoped by the fetched id range; a catalog
            // predating partition support simply yields no rows.
            if let (Some(min), Some(max)) = (
                files.iter().map(|f| f.data_file_id).min(),
                files.iter().map(|f| f.data_file_id).max(),
            ) {
                let mut values_by_file: HashMap<i64, Vec<(i32, Option<String>)>> = HashMap::new();
                match sqlx::query(SQL_GET_FILE_PARTITION_VALUES)
                    .bind(table_id)
                    .bind(min - 1)
                    .bind(max)
                    .fetch_all(&self.pool)
                    .await
                {
                    Ok(rows) => {
                        for row in rows {
                            let data_file_id: i64 = row.try_get(0)?;
                            let key_index: i32 =
                                i32::try_from(row.try_get::<i64, _>(1)?).unwrap_or(0);
                            let value: Option<String> = row.try_get(2)?;
                            values_by_file
                                .entry(data_file_id)
                                .or_default()
                                .push((key_index, value));
                        }
                    },
                    Err(error) if is_missing_statistics_table(&error) => {},
                    Err(error) => return Err(error.into()),
                }
                for file in &mut files {
                    if let Some(values) = values_by_file.remove(&file.data_file_id) {
                        file.partition_values = values;
                    }
                }
            }
            Ok(files)
        })
    }

    fn get_partition_spec(&self, table_id: i64, snapshot_id: i64) -> Result<Option<PartitionSpec>> {
        block_on(async {
            // Pruning is only safe with exactly one spec generation ever (see
            // PartitionSpec::prune_safe); the live spec is returned regardless so
            // the write path always targets the current generation.
            let generation_count: i64 = match sqlx::query_scalar(
                "SELECT COUNT(*) FROM ducklake_partition_info WHERE table_id = ?",
            )
            .bind(table_id)
            .fetch_one(&self.pool)
            .await
            {
                Ok(count) => count,
                Err(error) if is_missing_statistics_table(&error) => return Ok(None),
                Err(error) => return Err(error.into()),
            };
            let prune_safe = generation_count == 1;
            let rows = match sqlx::query(SQL_GET_PARTITION_SPEC)
                .bind(table_id)
                .bind(snapshot_id)
                .bind(snapshot_id)
                .fetch_all(&self.pool)
                .await
            {
                Ok(rows) => rows,
                Err(error) if is_missing_statistics_table(&error) => return Ok(None),
                Err(error) => return Err(error.into()),
            };
            let parsed = rows
                .iter()
                .map(|row| {
                    Ok::<_, crate::DuckLakeError>((
                        row.try_get::<i64, _>(0)?,
                        i32::try_from(row.try_get::<i64, _>(1)?).unwrap_or(0),
                        row.try_get::<i64, _>(2)?,
                        row.try_get::<String, _>(3)?,
                    ))
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(PartitionSpec::from_rows(parsed, prune_safe))
        })
    }

    fn get_sort_spec(&self, table_id: i64, snapshot_id: i64) -> Result<Option<SortSpec>> {
        block_on(async {
            let rows = match sqlx::query(SQL_GET_SORT_SPEC)
                .bind(table_id)
                .bind(snapshot_id)
                .bind(snapshot_id)
                .fetch_all(&self.pool)
                .await
            {
                Ok(rows) => rows,
                Err(error) if is_missing_statistics_table(&error) => return Ok(None),
                Err(error) => return Err(error.into()),
            };
            let parsed = rows
                .iter()
                .map(|row| {
                    Ok::<_, crate::DuckLakeError>((
                        row.try_get::<i64, _>(0)?,
                        i32::try_from(row.try_get::<i64, _>(1)?).unwrap_or(0),
                        row.try_get::<String, _>(2)?,
                        row.try_get::<String, _>(3)?,
                        row.try_get::<String, _>(4)?,
                        row.try_get::<String, _>(5)?,
                    ))
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(SortSpec::from_rows(parsed))
        })
    }

    fn get_table_file_metadata_page(
        &self,
        table_id: i64,
        snapshot_id: i64,
        after_data_file_id: Option<i64>,
        limit: usize,
    ) -> Result<Vec<DuckLakeFileMetadata>> {
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
    ) -> Result<Vec<DuckLakeFileMetadata>> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let limit = i64::try_from(limit).map_err(|_| {
            crate::DuckLakeError::InvalidConfig("file metadata page limit exceeds i64".to_string())
        })?;
        block_on(async {
            let caps = self.schema_capabilities().await?;
            let partial_max_expr = if caps.data_file_partial_max {
                "data.partial_max"
            } else {
                "NULL"
            };
            let partition_id_expr = if caps.data_file_partition_id {
                "data.partition_id"
            } else {
                "NULL"
            };
            let schema_version_expr = if caps.schema_versions {
                "(SELECT sv.schema_version
                  FROM ducklake_schema_versions sv
                  WHERE sv.table_id = data.table_id
                    AND sv.begin_snapshot <= data.begin_snapshot
                  ORDER BY sv.begin_snapshot DESC
                  LIMIT 1)"
            } else {
                "NULL"
            };
            // The statistics filter narrows the list inside this query, before
            // LIMIT and with the `ORDER BY data.data_file_id` keyset ordering
            // intact. Filtering a fetched page instead would break the cursor in
            // `crate::table::FileMetadataPages`: a page whose candidates all
            // fail the filter comes back empty, which reads as "no files left",
            // and every matching file beyond it is never visited.
            let filter_sql = stats_filter_sql(filter, table_id);
            let build_sql = |filter_sql: Option<&StatsFilterSql>| {
                let (cte, joins, conditions) = filter_sql.map_or(("", "", ""), |sql| {
                    (
                        sql.cte.as_str(),
                        sql.joins.as_str(),
                        sql.conditions.as_str(),
                    )
                });
                format!(
                    "{cte}SELECT
                    data.data_file_id, data.path, data.path_is_relative,
                    data.file_size_bytes, data.footer_size, data.encryption_key,
                    data.row_id_start, data.record_count,
                    del.delete_file_id, del.path, del.path_is_relative,
                    del.file_size_bytes, del.footer_size, del.encryption_key,
                    del.delete_count, data.begin_snapshot,
                    {partial_max_expr}, {schema_version_expr}, {partition_id_expr},
                    data.mapping_id
                 FROM ducklake_data_file AS data
                 LEFT JOIN ducklake_delete_file AS del
                   ON data.data_file_id = del.data_file_id
                  AND del.table_id = ?
                  AND ? >= del.begin_snapshot
                  AND (? < del.end_snapshot OR del.end_snapshot IS NULL){joins}
                 WHERE data.table_id = ?
                   AND ? >= data.begin_snapshot
                   AND (? < data.end_snapshot OR data.end_snapshot IS NULL)
                   AND data.data_file_id > ?{conditions}
                 ORDER BY data.data_file_id
                 LIMIT ?"
                )
            };
            let rows = match self
                .fetch_file_page(
                    &build_sql(filter_sql.as_ref()),
                    table_id,
                    snapshot_id,
                    after_data_file_id,
                    limit,
                )
                .await
            {
                Ok(rows) => rows,
                // The filter is advisory, so a catalog the narrowed query cannot
                // run — most importantly one predating `ducklake_file_column_stats`,
                // where joining it is a hard error — still lists its files. The
                // retry uses the same parameters, and a failure that is not the
                // filter's fault surfaces from it.
                Err(error) if filter_sql.is_some() => {
                    tracing::debug!(
                        %error,
                        table_id,
                        "statistics-filtered file listing failed; listing every file"
                    );
                    self.fetch_file_page(
                        &build_sql(None),
                        table_id,
                        snapshot_id,
                        after_data_file_id,
                        limit,
                    )
                    .await?
                },
                Err(error) => return Err(error.into()),
            };
            let files = rows
                .iter()
                .map(|row| {
                    let mut file = decode_table_file(row, snapshot_id)?;
                    // partition_id projected as the trailing column (index 18); NULL
                    // (→ None) on unpartitioned files or pre-migration catalogs.
                    file.partition_id = row.try_get(18)?;
                    Ok(file)
                })
                .collect::<Result<Vec<_>>>()?;
            let Some(last_data_file_id) = files.last().map(|file| file.data_file_id) else {
                return Ok(Vec::new());
            };
            // A filtered page's ids are sparse within `(after, last]`, so the
            // two enrichment queries below are restricted to the ids actually
            // returned rather than to that whole range. Unfiltered the range
            // holds at most one page of files, but a selective filter can put a
            // handful of survivors at the far end of a million-file table, and
            // the range would then pull back the statistics of every file the
            // filter just pruned — the exact cost the pushdown exists to remove,
            // and unbounded resident memory besides.
            //
            // The ids are inlined rather than bound as one parameter, which is
            // what the PostgreSQL provider does. That is a dialect difference,
            // not a disagreement: `= ANY($n::bigint[])` needs an array type,
            // and PostgreSQL is the only one of these engines that has one (and
            // the only one sqlx 0.9 can bind a `Vec<i64>` to). SQLite's nearest
            // shape, `json_each` over a bound JSON document, would put the
            // JSON functions of whatever SQLite the build links between a
            // caller and its file listing. Ids are `i64`, so inlining them adds
            // no bind parameter and the parameter lists stay as they are.
            //
            // Inlining does give every page a distinct query string, and sqlx
            // keys its per-connection prepared-statement cache on that string.
            // Both queries below are therefore non-persistent whenever they
            // carry ids: sqlx then holds them in the driver's single scratch
            // slot rather than the cache, which holds 100 statements by default
            // and would otherwise fill with per-page strings that never repeat
            // and evict the listing query along with everything else the
            // connection had prepared. Preparing is local here, so what a fresh
            // string costs SQLite is a parse — the round trip PostgreSQL would
            // spend on Parse/Describe per page is the other half of why it
            // binds instead.
            let page_ids = filter_sql.as_ref().map(|_| {
                files
                    .iter()
                    .map(|file| file.data_file_id.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            });
            let page_id_filter = |column: &str| {
                page_ids.as_ref().map_or_else(String::new, |ids| {
                    format!("\n                   AND {column} IN ({ids})")
                })
            };

            let statistics = match sqlx::query(AssertSqlSafe(format!(
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
            )))
            .bind(table_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .bind(after_data_file_id.unwrap_or(i64::MIN))
            .bind(last_data_file_id)
            // Cacheable only while the text is stable; see the note above.
            .persistent(page_ids.is_none())
            .fetch_all(&self.pool)
            .await
            {
                Ok(rows) => rows
                    .into_iter()
                    .map(|row| {
                        Ok(DuckLakeFileColumnStatistics {
                            data_file_id: row.try_get(0)?,
                            column_id: row.try_get(1)?,
                            column_size_bytes: row.try_get(2)?,
                            value_count: row.try_get(3)?,
                            null_count: row.try_get(4)?,
                            min_value: row.try_get(5)?,
                            max_value: row.try_get(6)?,
                            contains_nan: row.try_get(7)?,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?,
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

            // Enrich with per-file partition values (for pruning), scoped to the
            // page's data_file_id range. Missing partition table => no enrichment.
            let mut values_by_file: HashMap<i64, Vec<(i32, Option<String>)>> = HashMap::new();
            match sqlx::query(AssertSqlSafe(format!(
                "{SQL_GET_FILE_PARTITION_VALUES}{}",
                page_id_filter("data_file_id")
            )))
            .bind(table_id)
            .bind(after_data_file_id.unwrap_or(i64::MIN))
            .bind(last_data_file_id)
            // Cacheable only while the text is stable; see the note above.
            .persistent(page_ids.is_none())
            .fetch_all(&self.pool)
            .await
            {
                Ok(rows) => {
                    for row in rows {
                        let data_file_id: i64 = row.try_get(0)?;
                        let key_index: i32 = i32::try_from(row.try_get::<i64, _>(1)?).unwrap_or(0);
                        let value: Option<String> = row.try_get(2)?;
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
        })
    }

    fn get_table_summary_statistics(
        &self,
        table_id: i64,
        snapshot_id: i64,
    ) -> Result<DuckLakeStatistics> {
        block_on(async {
            let table = match sqlx::query(
                "SELECT record_count, file_size_bytes
                 FROM ducklake_table_stats WHERE table_id = ?",
            )
            .bind(table_id)
            .fetch_optional(&self.pool)
            .await
            {
                Ok(row) => row
                    .map(|row| {
                        Ok::<_, sqlx::Error>(DuckLakeTableStatistics {
                            record_count: row.try_get(0)?,
                            file_size_bytes: row.try_get(1)?,
                        })
                    })
                    .transpose()?,
                Err(error) if is_missing_statistics_table(&error) => None,
                Err(error) => return Err(error.into()),
            };
            let column_sizes = match sqlx::query(
                "SELECT stats.column_id,
                        CASE
                          WHEN COUNT(*) = COUNT(stats.column_size_bytes)
                           AND COUNT(*) = (
                             SELECT COUNT(*) FROM ducklake_data_file visible
                             WHERE visible.table_id = ?
                               AND ? >= visible.begin_snapshot
                               AND (? < visible.end_snapshot OR visible.end_snapshot IS NULL)
                           )
                          THEN CAST(SUM(stats.column_size_bytes) AS INTEGER)
                        END
                 FROM ducklake_file_column_stats stats
                 INNER JOIN ducklake_data_file data
                   ON data.data_file_id = stats.data_file_id
                  AND data.table_id = stats.table_id
                 WHERE stats.table_id = ?
                   AND ? >= data.begin_snapshot
                   AND (? < data.end_snapshot OR data.end_snapshot IS NULL)
                 GROUP BY stats.column_id",
            )
            .bind(table_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .bind(table_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .fetch_all(&self.pool)
            .await
            {
                Ok(rows) => rows
                    .into_iter()
                    .filter_map(|row| match row.try_get::<Option<i64>, _>(1) {
                        Ok(Some(size)) => Some(row.try_get(0).map(|column_id| (column_id, size))),
                        Ok(None) => None,
                        Err(error) => Some(Err(error)),
                    })
                    .collect::<std::result::Result<HashMap<i64, i64>, _>>()?,
                Err(error) if is_missing_statistics_table(&error) => HashMap::new(),
                Err(error) => return Err(error.into()),
            };
            let bounds_are_exact: bool = sqlx::query_scalar(
                "SELECT NOT EXISTS (
                     SELECT 1 FROM ducklake_delete_file
                     WHERE table_id = ?
                       AND ? >= begin_snapshot
                       AND (? < end_snapshot OR end_snapshot IS NULL)
                 )",
            )
            .bind(table_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .fetch_one(&self.pool)
            .await?;
            let columns = match sqlx::query(
                "SELECT column_id, contains_null, min_value, max_value, contains_nan
                 FROM ducklake_table_column_stats WHERE table_id = ?",
            )
            .bind(table_id)
            .fetch_all(&self.pool)
            .await
            {
                Ok(rows) => rows
                    .into_iter()
                    .map(|row| {
                        let column_id = row.try_get(0)?;
                        Ok(DuckLakeTableColumnStatistics {
                            column_id,
                            contains_null: row.try_get(1)?,
                            min_value: row.try_get(2)?,
                            max_value: row.try_get(3)?,
                            contains_nan: row.try_get(4)?,
                            column_size_bytes: column_sizes.get(&column_id).copied(),
                            bounds_are_exact,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?,
                Err(error) if is_missing_statistics_table(&error) => Vec::new(),
                Err(error) => return Err(error.into()),
            };
            Ok(DuckLakeStatistics {
                table,
                columns,
                files: Vec::new(),
            })
        })
    }

    fn get_table_statistics(&self, table_id: i64, snapshot_id: i64) -> Result<DuckLakeStatistics> {
        block_on(async {
            let table = match sqlx::query(
                "SELECT record_count, file_size_bytes
                 FROM ducklake_table_stats WHERE table_id = ?",
            )
            .bind(table_id)
            .fetch_optional(&self.pool)
            .await
            {
                Ok(row) => row
                    .map(|row| {
                        Ok::<_, sqlx::Error>(DuckLakeTableStatistics {
                            record_count: row.try_get(0)?,
                            file_size_bytes: row.try_get(1)?,
                        })
                    })
                    .transpose()?,
                Err(error) if is_missing_statistics_table(&error) => None,
                Err(error) => return Err(error.into()),
            };

            let columns = match sqlx::query(
                "SELECT column_id, contains_null, min_value, max_value, contains_nan
                 FROM ducklake_table_column_stats WHERE table_id = ?",
            )
            .bind(table_id)
            .fetch_all(&self.pool)
            .await
            {
                Ok(rows) => rows
                    .into_iter()
                    .map(|row| {
                        Ok(DuckLakeTableColumnStatistics {
                            column_id: row.try_get(0)?,
                            contains_null: row.try_get(1)?,
                            min_value: row.try_get(2)?,
                            max_value: row.try_get(3)?,
                            contains_nan: row.try_get(4)?,
                            column_size_bytes: None,
                            bounds_are_exact: false,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?,
                Err(error) if is_missing_statistics_table(&error) => Vec::new(),
                Err(error) => return Err(error.into()),
            };

            let files = match sqlx::query(
                "SELECT
                    stats.data_file_id,
                    stats.column_id,
                    stats.column_size_bytes,
                    stats.value_count,
                    stats.null_count,
                    stats.min_value,
                    stats.max_value,
                    stats.contains_nan
                 FROM ducklake_file_column_stats AS stats
                 INNER JOIN ducklake_data_file AS data
                    ON data.data_file_id = stats.data_file_id
                    AND data.table_id = stats.table_id
                 WHERE stats.table_id = ?
                   AND ? >= data.begin_snapshot
                   AND (? < data.end_snapshot OR data.end_snapshot IS NULL)",
            )
            .bind(table_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .fetch_all(&self.pool)
            .await
            {
                Ok(rows) => rows
                    .into_iter()
                    .map(|row| {
                        Ok(DuckLakeFileColumnStatistics {
                            data_file_id: row.try_get(0)?,
                            column_id: row.try_get(1)?,
                            column_size_bytes: row.try_get(2)?,
                            value_count: row.try_get(3)?,
                            null_count: row.try_get(4)?,
                            min_value: row.try_get(5)?,
                            max_value: row.try_get(6)?,
                            contains_nan: row.try_get(7)?,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?,
                Err(error) if is_missing_statistics_table(&error) => Vec::new(),
                Err(error) => return Err(error.into()),
            };

            Ok(DuckLakeStatistics {
                table,
                columns,
                files,
            })
        })
    }

    fn get_inlined_data(
        &self,
        table_id: i64,
        snapshot_id: i64,
        columns: &[DuckLakeTableColumn],
    ) -> Result<Vec<RecordBatch>> {
        block_on(async {
            // Most catalogs have no inlined data — the registry table is absent.
            // Detect and return empty so they (and older catalogs) are unaffected.
            if !self.schema_capabilities().await?.inlined_data_tables {
                return Ok(Vec::new());
            }

            // Every physical inlined table for this table (one per schema version).
            let regs = sqlx::query(
                "SELECT table_name FROM ducklake_inlined_data_tables WHERE table_id = ?",
            )
            .bind(table_id)
            .fetch_all(&self.pool)
            .await?;
            if regs.is_empty() {
                return Ok(Vec::new());
            }

            let schema: SchemaRef = Arc::new(crate::types::build_arrow_schema(columns)?);
            let mut batches = Vec::new();
            for reg in regs {
                let phys: String = reg.try_get("table_name")?;
                // Defensive: only touch tables that look like DuckLake inline tables.
                if !phys.starts_with("ducklake_inlined_data_")
                    || !phys.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
                {
                    continue;
                }

                // Which of the table's columns this inline table physically has
                // (its layout matches the schema version it was created for).
                let info = sqlx::query(AssertSqlSafe(format!(
                    "SELECT name FROM pragma_table_info({})",
                    // pragma wants a string literal; single-quote-escape the name.
                    format_args!("'{}'", phys.replace('\'', "''"))
                )))
                .fetch_all(&self.pool)
                .await?;
                let present: HashSet<String> = info
                    .iter()
                    .filter_map(|r| r.try_get::<String, _>("name").ok())
                    .collect();

                // Project the table columns this inline table actually has; rows
                // visible at the snapshot (this predicate also hides inlined-row
                // deletes, which set end_snapshot). ORDER BY row_id for stability.
                let projected: Vec<String> = columns
                    .iter()
                    .filter(|c| present.contains(c.column_name.as_str()))
                    .map(|c| quote_ident(&c.column_name))
                    .collect();
                let select_list = if projected.is_empty() {
                    "1".to_string()
                } else {
                    projected.join(", ")
                };
                let sql = format!(
                    "SELECT {select_list} FROM {} \
                     WHERE ? >= begin_snapshot AND (? < end_snapshot OR end_snapshot IS NULL) \
                     ORDER BY row_id",
                    quote_ident(&phys)
                );
                let rows = sqlx::query(AssertSqlSafe(sql.as_str()))
                    .bind(snapshot_id)
                    .bind(snapshot_id)
                    .fetch_all(&self.pool)
                    .await?;
                if rows.is_empty() {
                    continue;
                }
                batches.push(build_inlined_batch(&schema, columns, &present, &rows)?);
            }
            Ok(batches)
        })
    }

    fn get_inlined_deletes(
        &self,
        table_id: i64,
        snapshot_id: i64,
    ) -> Result<Vec<DuckLakeInlinedDelete>> {
        block_on(async {
            let table = inlined_delete_table_name(table_id)?;
            let sql = format!(
                "SELECT file_id, row_id FROM {} WHERE begin_snapshot <= ? ORDER BY file_id, row_id",
                quote_ident(&table)
            );
            match sqlx::query(AssertSqlSafe(sql.as_str()))
                .bind(snapshot_id)
                .fetch_all(&self.pool)
                .await
            {
                Ok(rows) => rows
                    .into_iter()
                    .map(|row| {
                        Ok(DuckLakeInlinedDelete {
                            data_file_id: row.try_get(0)?,
                            row_id: row.try_get(1)?,
                        })
                    })
                    .collect(),
                Err(error) if is_missing_statistics_table(&error) => Ok(Vec::new()),
                Err(error) => Err(error.into()),
            }
        })
    }

    fn get_inlined_data_with_row_ids(
        &self,
        table_id: i64,
        snapshot_id: i64,
        columns: &[DuckLakeTableColumn],
    ) -> Result<Vec<DuckLakeInlinedData>> {
        block_on(async {
            if !self.schema_capabilities().await?.inlined_data_tables {
                return Ok(Vec::new());
            }
            let regs = sqlx::query(
                "SELECT table_name FROM ducklake_inlined_data_tables WHERE table_id = ?",
            )
            .bind(table_id)
            .fetch_all(&self.pool)
            .await?;
            let schema: SchemaRef = Arc::new(crate::types::build_arrow_schema(columns)?);
            let mut batches = Vec::new();
            for reg in regs {
                let physical_name: String = reg.try_get("table_name")?;
                if !physical_name.starts_with("ducklake_inlined_data_")
                    || !physical_name
                        .chars()
                        .all(|c| c.is_ascii_alphanumeric() || c == '_')
                {
                    continue;
                }
                let info = sqlx::query(AssertSqlSafe(format!(
                    "SELECT name FROM pragma_table_info({})",
                    format_args!("'{}'", physical_name.replace('\'', "''"))
                )))
                .fetch_all(&self.pool)
                .await?;
                let present: HashSet<String> = info
                    .iter()
                    .filter_map(|row| row.try_get::<String, _>("name").ok())
                    .collect();
                let projected = columns
                    .iter()
                    .filter(|column| present.contains(column.column_name.as_str()))
                    .map(|column| quote_ident(&column.column_name))
                    .collect::<Vec<_>>();
                let select_list = if projected.is_empty() {
                    "row_id, begin_snapshot".to_string()
                } else {
                    format!("row_id, begin_snapshot, {}", projected.join(", "))
                };
                let sql = format!(
                    "SELECT {select_list} FROM {} \
                     WHERE ? >= begin_snapshot AND (? < end_snapshot OR end_snapshot IS NULL) \
                 ORDER BY begin_snapshot, row_id",
                    quote_ident(&physical_name)
                );
                let rows = sqlx::query(AssertSqlSafe(sql))
                    .bind(snapshot_id)
                    .bind(snapshot_id)
                    .fetch_all(&self.pool)
                    .await?;
                if rows.is_empty() {
                    continue;
                }
                let row_ids = rows
                    .iter()
                    .map(|row| row.try_get("row_id"))
                    .collect::<std::result::Result<Vec<i64>, _>>()?;
                let begin_snapshots = rows
                    .iter()
                    .map(|row| row.try_get("begin_snapshot"))
                    .collect::<std::result::Result<Vec<i64>, _>>()?;
                batches.push(DuckLakeInlinedData {
                    table_name: physical_name,
                    row_ids,
                    begin_snapshots,
                    batch: build_inlined_batch(&schema, columns, &present, &rows)?,
                });
            }
            Ok(batches)
        })
    }

    fn get_schema_by_name(&self, name: &str, snapshot_id: i64) -> Result<Option<SchemaMetadata>> {
        block_on(async {
            let row = sqlx::query(
                "SELECT schema_id, schema_name, path, path_is_relative FROM ducklake_schema
                 WHERE schema_name = ?
                   AND ? >= begin_snapshot
                   AND (? < end_snapshot OR end_snapshot IS NULL)",
            )
            .bind(name)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .fetch_optional(&self.pool)
            .await?;

            match row {
                Some(r) => Ok(Some(SchemaMetadata {
                    schema_id: r.try_get(0)?,
                    schema_name: r.try_get(1)?,
                    path: r.try_get(2)?,
                    path_is_relative: r.try_get(3)?,
                })),
                None => Ok(None),
            }
        })
    }

    fn get_table_by_name(
        &self,
        schema_id: i64,
        name: &str,
        snapshot_id: i64,
    ) -> Result<Option<TableMetadata>> {
        block_on(async {
            let row = sqlx::query(
                "SELECT table_id, table_name, path, path_is_relative FROM ducklake_table
                 WHERE schema_id = ?
                   AND table_name = ?
                   AND ? >= begin_snapshot
                   AND (? < end_snapshot OR end_snapshot IS NULL)",
            )
            .bind(schema_id)
            .bind(name)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .fetch_optional(&self.pool)
            .await?;

            match row {
                Some(r) => Ok(Some(TableMetadata {
                    table_id: r.try_get(0)?,
                    table_name: r.try_get(1)?,
                    path: r.try_get(2)?,
                    path_is_relative: r.try_get(3)?,
                })),
                None => Ok(None),
            }
        })
    }

    fn get_view_by_name(
        &self,
        schema_id: i64,
        name: &str,
        snapshot_id: i64,
    ) -> Result<Option<ViewMetadata>> {
        block_on(async {
            if !self.schema_capabilities().await?.views {
                return Ok(None);
            }
            let row = sqlx::query(
                "SELECT view_id, schema_id, begin_snapshot, view_name, dialect, sql, column_aliases
                 FROM ducklake_view
                 WHERE schema_id = ?
                   AND view_name = ?
                   AND ? >= begin_snapshot
                   AND (? < end_snapshot OR end_snapshot IS NULL)",
            )
            .bind(schema_id)
            .bind(name)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .fetch_optional(&self.pool)
            .await?;

            row.map(|row| decode_view(&row)).transpose()
        })
    }

    fn table_exists(&self, schema_id: i64, name: &str, snapshot_id: i64) -> Result<bool> {
        block_on(async {
            let row = sqlx::query(
                "SELECT COUNT(*) FROM ducklake_table
                 WHERE schema_id = ?
                   AND table_name = ?
                   AND ? >= begin_snapshot
                   AND (? < end_snapshot OR end_snapshot IS NULL)",
            )
            .bind(schema_id)
            .bind(name)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .fetch_one(&self.pool)
            .await?;

            let count: i64 = row.try_get(0)?;
            Ok(count > 0)
        })
    }

    fn list_all_tables(&self, snapshot_id: i64) -> Result<Vec<TableWithSchema>> {
        block_on(async {
            let rows = sqlx::query(
                "SELECT s.schema_name, t.table_id, t.table_name, t.path, t.path_is_relative
                 FROM ducklake_schema s
                 JOIN ducklake_table t ON s.schema_id = t.schema_id
                 WHERE ? >= s.begin_snapshot
                   AND (? < s.end_snapshot OR s.end_snapshot IS NULL)
                   AND ? >= t.begin_snapshot
                   AND (? < t.end_snapshot OR t.end_snapshot IS NULL)
                 ORDER BY s.schema_name, t.table_name",
            )
            .bind(snapshot_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .fetch_all(&self.pool)
            .await?;

            rows.into_iter()
                .map(|row| {
                    let schema_name: String = row.try_get(0)?;
                    let table = TableMetadata {
                        table_id: row.try_get(1)?,
                        table_name: row.try_get(2)?,
                        path: row.try_get(3)?,
                        path_is_relative: row.try_get(4)?,
                    };
                    Ok(TableWithSchema {
                        schema_name,
                        table,
                    })
                })
                .collect()
        })
    }

    fn list_all_views(&self, snapshot_id: i64) -> Result<Vec<ViewWithSchema>> {
        block_on(async {
            if !self.schema_capabilities().await?.views {
                return Ok(Vec::new());
            }
            let rows = sqlx::query(
                "SELECT s.schema_name, v.view_id, v.schema_id, v.begin_snapshot, v.view_name,
                        v.dialect, v.sql, v.column_aliases
                 FROM ducklake_schema s
                 JOIN ducklake_view v ON s.schema_id = v.schema_id
                 WHERE ? >= s.begin_snapshot
                   AND (? < s.end_snapshot OR s.end_snapshot IS NULL)
                   AND ? >= v.begin_snapshot
                   AND (? < v.end_snapshot OR v.end_snapshot IS NULL)
                 ORDER BY s.schema_name, v.view_name",
            )
            .bind(snapshot_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .fetch_all(&self.pool)
            .await?;
            rows.into_iter()
                .map(|row| {
                    Ok(ViewWithSchema {
                        schema_name: row.try_get(0)?,
                        view: ViewMetadata {
                            view_id: row.try_get(1)?,
                            schema_id: row.try_get(2)?,
                            begin_snapshot: row.try_get(3)?,
                            view_name: row.try_get(4)?,
                            dialect: row.try_get(5)?,
                            sql: row.try_get(6)?,
                            column_aliases: row.try_get(7)?,
                        },
                    })
                })
                .collect()
        })
    }

    fn list_all_columns(&self, snapshot_id: i64) -> Result<Vec<ColumnWithTable>> {
        block_on(async {
            let rows = sqlx::query(
                "SELECT s.schema_name, t.table_name, c.column_id, c.column_name, c.column_type,
                        c.nulls_allowed, c.parent_column, c.initial_default, c.default_value,
                        c.default_value_type, c.default_value_dialect
                 FROM ducklake_schema s
                 JOIN ducklake_table t ON s.schema_id = t.schema_id
                 JOIN ducklake_column c ON t.table_id = c.table_id
                 WHERE ? >= s.begin_snapshot
                   AND (? < s.end_snapshot OR s.end_snapshot IS NULL)
                   AND ? >= t.begin_snapshot
                   AND (? < t.end_snapshot OR t.end_snapshot IS NULL)
                   AND ? >= c.begin_snapshot
                   AND (? < c.end_snapshot OR c.end_snapshot IS NULL)
                 ORDER BY s.schema_name, t.table_name, c.column_order",
            )
            .bind(snapshot_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .fetch_all(&self.pool)
            .await?;

            let raw: Result<Vec<(ColumnWithTable, Option<i64>)>> = rows
                .into_iter()
                .map(|row| {
                    let schema_name: String = row.try_get(0)?;
                    let table_name: String = row.try_get(1)?;
                    let nulls_allowed: Option<bool> = row.try_get(5)?;
                    let parent_column: Option<i64> = row.try_get(6)?;
                    let column = DuckLakeTableColumn::new(
                        row.try_get(2)?,
                        row.try_get(3)?,
                        row.try_get(4)?,
                        nulls_allowed.unwrap_or(true),
                    )
                    .with_defaults(
                        row.try_get(7)?,
                        row.try_get(8)?,
                        row.try_get(9)?,
                        row.try_get(10)?,
                    );
                    Ok((
                        ColumnWithTable {
                            schema_name,
                            table_name,
                            column,
                        },
                        parent_column,
                    ))
                })
                .collect();
            reconstruct_columns_with_table(raw?)
        })
    }

    fn list_all_files(&self, snapshot_id: i64) -> Result<Vec<FileWithTable>> {
        block_on(async {
            let rows = sqlx::query(
                "SELECT
                    s.schema_name,
                    t.table_name,
                    data.data_file_id,
                    data.path AS data_file_path,
                    data.path_is_relative AS data_path_is_relative,
                    data.file_size_bytes AS data_file_size,
                    data.footer_size AS data_footer_size,
                    data.encryption_key AS data_encryption_key,
                    del.delete_file_id,
                    del.path AS delete_file_path,
                    del.path_is_relative AS delete_path_is_relative,
                    del.file_size_bytes AS delete_file_size,
                    del.footer_size AS delete_footer_size,
                    del.encryption_key AS delete_encryption_key,
                    del.delete_count
                FROM ducklake_schema s
                JOIN ducklake_table t ON s.schema_id = t.schema_id
                JOIN ducklake_data_file data ON t.table_id = data.table_id
                LEFT JOIN ducklake_delete_file del
                    ON data.data_file_id = del.data_file_id
                    AND del.table_id = t.table_id
                    AND ? >= del.begin_snapshot
                    AND (? < del.end_snapshot OR del.end_snapshot IS NULL)
                WHERE ? >= s.begin_snapshot
                  AND (? < s.end_snapshot OR s.end_snapshot IS NULL)
                  AND ? >= t.begin_snapshot
                  AND (? < t.end_snapshot OR t.end_snapshot IS NULL)
                  AND ? >= data.begin_snapshot
                  AND (? < data.end_snapshot OR data.end_snapshot IS NULL)
                ORDER BY s.schema_name, t.table_name, data.path",
            )
            .bind(snapshot_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .fetch_all(&self.pool)
            .await?;

            rows.into_iter()
                .map(|row| {
                    let data_file = DuckLakeFileData {
                        path: row.try_get(3)?,
                        path_is_relative: row.try_get(4)?,
                        file_size_bytes: row.try_get(5)?,
                        footer_size: row.try_get(6)?,
                        encryption_key: row.try_get(7)?,
                        mapping_id: None,
                    };

                    let delete_file = if row.try_get::<Option<i64>, _>(8)?.is_some() {
                        Some(DuckLakeFileData {
                            path: row.try_get(9)?,
                            path_is_relative: row.try_get(10)?,
                            file_size_bytes: row.try_get(11)?,
                            footer_size: row.try_get(12)?,
                            encryption_key: row.try_get(13)?,
                            mapping_id: None,
                        })
                    } else {
                        None
                    };

                    Ok(FileWithTable {
                        schema_name: row.try_get(0)?,
                        table_name: row.try_get(1)?,
                        file: DuckLakeTableFile {
                            data_file_id: row.try_get(2)?,
                            file: data_file,
                            delete_file_id: row.try_get(8)?,
                            delete_file,
                            row_id_start: None,
                            snapshot_id: None,
                            begin_snapshot: None,
                            schema_version: None,
                            partial_max: None,
                            max_row_count: row.try_get(14)?,
                            delete_count: None,
                            partition_id: None,
                            partition_values: Vec::new(),
                        },
                    })
                })
                .collect()
        })
    }

    fn get_data_files_added_between_snapshots(
        &self,
        table_id: i64,
        start_snapshot: i64,
        end_snapshot: i64,
    ) -> Result<Vec<DataFileChange>> {
        block_on(async {
            // Older catalogs predate `partial_max`; degrade it to NULL there
            // (they cannot contain partial files), matching the probe pattern
            // used by the scan queries above.
            let pm = if self.schema_capabilities().await?.data_file_partial_max {
                "data.partial_max"
            } else {
                "NULL"
            };
            let rows = sqlx::query(AssertSqlSafe(format!(
                "SELECT
                    data.begin_snapshot,
                    data.path,
                    data.path_is_relative,
                    data.file_size_bytes,
                    data.footer_size,
                    data.encryption_key,
                    data.row_id_start,
                    {pm},
                    data.mapping_id
                FROM ducklake_data_file AS data
                WHERE data.table_id = ?1
                  AND data.begin_snapshot <= ?3
                  AND (data.begin_snapshot >= ?2
                       OR ({pm} IS NOT NULL AND {pm} >= ?2))
                ORDER BY data.begin_snapshot"
            )))
            .bind(table_id)
            .bind(start_snapshot)
            .bind(end_snapshot)
            .fetch_all(&self.pool)
            .await?;

            rows.into_iter()
                .map(|row| {
                    Ok(DataFileChange {
                        begin_snapshot: row.try_get(0)?,
                        path: row.try_get(1)?,
                        path_is_relative: row.try_get(2)?,
                        file_size_bytes: row.try_get(3)?,
                        footer_size: row.try_get(4)?,
                        encryption_key: row.try_get(5)?,
                        row_id_start: row.try_get(6)?,
                        partial_max: row.try_get(7)?,
                        mapping_id: row.try_get(8)?,
                    })
                })
                .collect()
        })
    }

    fn get_delete_files_added_between_snapshots(
        &self,
        table_id: i64,
        start_snapshot: i64,
        end_snapshot: i64,
    ) -> Result<Vec<DeleteFileChange>> {
        block_on(async {
            // SQLite doesn't support LATERAL JOIN, so we use correlated subqueries instead
            // This query has two parts:
            // 1. Incremental deletes: delete files added in the snapshot range
            // 2. Full file deletes: data files that were completely removed in the snapshot range
            //
            // Cumulative (current-spec) delete files can hold in-window deletions
            // even when their begin_snapshot predates the window; they are
            // included via `ducklake_delete_file.partial_max` (their max embedded
            // snapshot). Older catalogs lack the column — and cumulative delete
            // files — so it degrades to NULL there.
            let pm = if self.schema_capabilities().await?.delete_file_partial_max {
                "cd.partial_max"
            } else {
                "NULL"
            };
            let rows = sqlx::query(AssertSqlSafe(format!(
                r#"
-- Part 1: Incremental deletes (delete file added)
SELECT
    data.path AS data_path,
    data.path_is_relative AS data_path_is_relative,
    data.file_size_bytes AS data_file_size,
    data.footer_size AS data_footer_size,
    data.row_id_start,
    data.record_count,
    data.mapping_id,

    cd.path AS current_delete_path,
    cd.path_is_relative AS current_delete_path_is_relative,
    cd.file_size_bytes AS current_delete_file_size,
    cd.footer_size AS current_delete_footer_size,

    -- Previous delete file (correlated subquery instead of LATERAL)
    (SELECT path FROM ducklake_delete_file pd
     WHERE pd.table_id = ?
       AND pd.data_file_id = cd.data_file_id
       AND pd.begin_snapshot < cd.begin_snapshot
     ORDER BY pd.begin_snapshot DESC LIMIT 1) AS prev_delete_path,
    (SELECT path_is_relative FROM ducklake_delete_file pd
     WHERE pd.table_id = ?
       AND pd.data_file_id = cd.data_file_id
       AND pd.begin_snapshot < cd.begin_snapshot
     ORDER BY pd.begin_snapshot DESC LIMIT 1) AS prev_delete_path_is_relative,
    (SELECT file_size_bytes FROM ducklake_delete_file pd
     WHERE pd.table_id = ?
       AND pd.data_file_id = cd.data_file_id
       AND pd.begin_snapshot < cd.begin_snapshot
     ORDER BY pd.begin_snapshot DESC LIMIT 1) AS prev_delete_file_size,
    (SELECT footer_size FROM ducklake_delete_file pd
     WHERE pd.table_id = ?
       AND pd.data_file_id = cd.data_file_id
       AND pd.begin_snapshot < cd.begin_snapshot
     ORDER BY pd.begin_snapshot DESC LIMIT 1) AS prev_delete_footer_size,

    cd.begin_snapshot AS snapshot_id
FROM ducklake_delete_file cd
JOIN ducklake_data_file data ON data.data_file_id = cd.data_file_id
WHERE cd.table_id = ?
  AND cd.begin_snapshot <= ?
  AND (cd.begin_snapshot >= ?
       OR ({pm} IS NOT NULL AND {pm} >= ?))
  AND data.table_id = ?

UNION ALL

-- Part 2: Full file deletes (data file removed entirely)
SELECT
    data.path AS data_path,
    data.path_is_relative AS data_path_is_relative,
    data.file_size_bytes AS data_file_size,
    data.footer_size AS data_footer_size,
    data.row_id_start,
    data.record_count,
    data.mapping_id,

    NULL AS current_delete_path,
    NULL AS current_delete_path_is_relative,
    NULL AS current_delete_file_size,
    NULL AS current_delete_footer_size,

    -- Previous delete file
    (SELECT path FROM ducklake_delete_file pd
     WHERE pd.table_id = ?
       AND pd.data_file_id = data.data_file_id
       AND pd.begin_snapshot < data.end_snapshot
     ORDER BY pd.begin_snapshot DESC LIMIT 1) AS prev_delete_path,
    (SELECT path_is_relative FROM ducklake_delete_file pd
     WHERE pd.table_id = ?
       AND pd.data_file_id = data.data_file_id
       AND pd.begin_snapshot < data.end_snapshot
     ORDER BY pd.begin_snapshot DESC LIMIT 1) AS prev_delete_path_is_relative,
    (SELECT file_size_bytes FROM ducklake_delete_file pd
     WHERE pd.table_id = ?
       AND pd.data_file_id = data.data_file_id
       AND pd.begin_snapshot < data.end_snapshot
     ORDER BY pd.begin_snapshot DESC LIMIT 1) AS prev_delete_file_size,
    (SELECT footer_size FROM ducklake_delete_file pd
     WHERE pd.table_id = ?
       AND pd.data_file_id = data.data_file_id
       AND pd.begin_snapshot < data.end_snapshot
     ORDER BY pd.begin_snapshot DESC LIMIT 1) AS prev_delete_footer_size,

    data.end_snapshot AS snapshot_id
FROM ducklake_data_file data
WHERE data.table_id = ?
  AND data.end_snapshot >= ?
  AND data.end_snapshot <= ?
"#
            )))
            // Part 1 bindings: 4x table_id for prev subqueries, table_id for cd,
            // end, start (window), start (partial_max), table_id for data
            .bind(table_id)
            .bind(table_id)
            .bind(table_id)
            .bind(table_id)
            .bind(table_id)
            .bind(end_snapshot)
            .bind(start_snapshot)
            .bind(start_snapshot)
            .bind(table_id)
            // Part 2 bindings: 4x table_id for prev subqueries, table_id for data, start, end
            .bind(table_id)
            .bind(table_id)
            .bind(table_id)
            .bind(table_id)
            .bind(table_id)
            .bind(start_snapshot)
            .bind(end_snapshot)
            .fetch_all(&self.pool)
            .await?;

            rows.into_iter()
                .map(|row| {
                    Ok(DeleteFileChange {
                        // data file
                        data_file_path: row.try_get(0)?,
                        data_file_path_is_relative: row.try_get(1)?,
                        data_file_size_bytes: row.try_get(2)?,
                        data_file_footer_size: row.try_get(3)?,
                        data_row_id_start: row.try_get(4)?,
                        data_record_count: row.try_get(5)?,
                        data_mapping_id: row.try_get(6)?,

                        // current delete
                        current_delete_path: row.try_get(7)?,
                        current_delete_path_is_relative: row.try_get(8)?,
                        current_delete_file_size_bytes: row.try_get(9)?,
                        current_delete_footer_size: row.try_get(10)?,

                        // previous delete
                        previous_delete_path: row.try_get(11)?,
                        previous_delete_path_is_relative: row.try_get(12)?,
                        previous_delete_file_size_bytes: row.try_get(13)?,
                        previous_delete_footer_size: row.try_get(14)?,

                        // snapshot
                        snapshot_id: row.try_get(15)?,
                    })
                })
                .collect()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stats_filter::lower_predicate;
    use arrow::datatypes::{Field, Schema, TimeUnit};
    use datafusion::common::ScalarValue;
    use datafusion::logical_expr::Operator;
    use datafusion::physical_expr::PhysicalExpr;
    use datafusion::physical_expr::expressions::{BinaryExpr, Column, lit};

    /// Lower `a < <value>` on a column of `data_type` whose `column_id` is 1.
    fn lower_one(data_type: DataType, value: ScalarValue) -> crate::stats_filter::StatsFilter {
        let column = Arc::new(Column::new("a", 0)) as Arc<dyn PhysicalExpr>;
        let predicate =
            Arc::new(BinaryExpr::new(column, Operator::Lt, lit(value))) as Arc<dyn PhysicalExpr>;
        let schema = Schema::new(vec![Field::new("a", data_type, true)]);
        let columns = vec![DuckLakeTableColumn::new(1, "a".to_string(), "x".to_string(), true)];
        lower_predicate(&predicate, &schema, &columns).expect("lowered")
    }

    fn int32_column(column_id: i64) -> (Schema, Vec<DuckLakeTableColumn>) {
        (
            Schema::new(vec![Field::new("a", DataType::Int32, true)]),
            vec![DuckLakeTableColumn::new(column_id, "a".to_string(), "int32".to_string(), true)],
        )
    }

    /// The shape of `a > 5 AND a < 10` on an `INTEGER` column, as the listing
    /// query splices it. Both conjuncts read a bound and neither reads
    /// `null_count`, so the CTE also selects `value_count` for the guard.
    #[test]
    fn renders_a_two_conjunct_integer_filter() {
        let column = Arc::new(Column::new("a", 0)) as Arc<dyn PhysicalExpr>;
        let predicate = Arc::new(BinaryExpr::new(
            Arc::new(BinaryExpr::new(
                Arc::clone(&column),
                Operator::Gt,
                lit(5i32),
            )),
            Operator::And,
            Arc::new(BinaryExpr::new(column, Operator::Lt, lit(10i32))),
        )) as Arc<dyn PhysicalExpr>;
        let (schema, columns) = int32_column(7);
        let filter = lower_predicate(&predicate, &schema, &columns).expect("lowered");
        let sql = stats_filter_sql(Some(&filter), 42).expect("rendered");
        println!("CTE:\n{}", sql.cte);
        println!("JOINS:{}", sql.joins);
        println!("CONDITIONS:{}", sql.conditions);

        assert!(
            sql.cte
                .starts_with("WITH col_7_stats AS (SELECT data_file_id, ")
        );
        assert!(sql.cte.contains("min_value, max_value, value_count"));
        assert!(sql.cte.contains("WHERE column_id = 7 AND table_id = 42)"));
        assert!(
            sql.joins
                .contains("LEFT JOIN col_7_stats ON col_7_stats.data_file_id = data.data_file_id")
        );
        // Both bounds are read through the round-trip guard, and the whole
        // condition fails open on a malformed one.
        assert!(sql.conditions.contains(
            "CASE WHEN col_7_stats.max_value = CAST(CAST(col_7_stats.max_value AS INTEGER) AS TEXT) \
             THEN CAST(col_7_stats.max_value AS INTEGER) END > 5"
        ));
        assert!(sql.conditions.trim_end().ends_with(") IS NOT FALSE"));
    }

    /// A float bound is read through the text check, and the whole condition
    /// sits behind the `contains_nan` gate: catalog bounds exclude NaN, and
    /// DataFusion orders a negative NaN below every value, so neither bound can
    /// be trusted while the NaN state is unknown or positive.
    #[test]
    fn renders_a_float_filter_behind_the_nan_gate() {
        let column = Arc::new(Column::new("f", 0)) as Arc<dyn PhysicalExpr>;
        let predicate =
            Arc::new(BinaryExpr::new(column, Operator::Lt, lit(5.0f64))) as Arc<dyn PhysicalExpr>;
        let schema = Schema::new(vec![Field::new("f", DataType::Float64, true)]);
        let columns =
            vec![DuckLakeTableColumn::new(2, "f".to_string(), "double".to_string(), true)];
        let filter = lower_predicate(&predicate, &schema, &columns).expect("lowered");
        let sql = stats_filter_sql(Some(&filter), 1).expect("rendered");
        println!("CONDITIONS:{}", sql.conditions);
        assert!(sql.cte.contains("min_value, value_count, contains_nan"));
        assert!(
            sql.conditions
                .contains("(col_2_stats.contains_nan IS NULL OR col_2_stats.contains_nan <> 0)")
        );
        // The bound is only read as a number when the whole text is one.
        assert!(sql.conditions.contains(
            "rtrim(ltrim(ltrim(col_2_stats.min_value, '-'), '0123456789'), '0123456789') = '.'"
        ));
        assert!(
            sql.conditions
                .contains("THEN CAST(col_2_stats.min_value AS REAL) END < 5.0")
        );
    }

    /// A `DECIMAL` comparison is declined outright, so the column contributes
    /// no CTE, no join and no condition at all.
    #[test]
    fn declines_decimal_comparisons() {
        let filter = lower_one(
            DataType::Decimal128(10, 2),
            ScalarValue::Decimal128(Some(1234), 10, 2),
        );
        assert!(stats_filter_sql(Some(&filter), 1).is_none());
    }

    /// A date is compared as text, behind a shape test that pins the stat to the
    /// same fixed-width encoding the constant is in.
    #[test]
    fn renders_a_date_filter_as_a_guarded_text_comparison() {
        // 19_723 days after the epoch is 2024-01-01.
        let filter = lower_one(DataType::Date32, ScalarValue::Date32(Some(19_723)));
        let sql = stats_filter_sql(Some(&filter), 1).expect("rendered");
        println!("CONDITIONS:{}", sql.conditions);
        assert!(sql.conditions.contains(
            "CASE WHEN col_1_stats.min_value GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]' \
             THEN col_1_stats.min_value END COLLATE BINARY < '2024-01-01'"
        ));
    }

    /// A constant outside the canonical four-digit-year encoding is declined:
    /// `chrono` writes those years with a leading `+` or `-`, which sort below
    /// every digit and would invert the text comparison. `stats_encode` renders
    /// 4_000_000 days after the epoch as `+12921-08-18` and 800_000 before it as
    /// `-0221-09-04`.
    #[test]
    fn declines_a_date_constant_outside_the_canonical_encoding() {
        for days in [4_000_000, -800_000] {
            let filter = lower_one(DataType::Date32, ScalarValue::Date32(Some(days)));
            assert!(
                stats_filter_sql(Some(&filter), 1).is_none(),
                "date at {days} days must not render"
            );
        }
    }

    /// A timestamp is compared as text too, fraction and all, and the UTC suffix
    /// `stats_encode` appends is part of the shape the stat must match.
    #[test]
    fn renders_timestamp_filters_with_and_without_a_zone() {
        let filter = lower_one(
            DataType::Timestamp(TimeUnit::Microsecond, None),
            ScalarValue::TimestampMicrosecond(Some(1_700_000_000_500_000), None),
        );
        let sql = stats_filter_sql(Some(&filter), 1).expect("rendered");
        println!("NAIVE:{}", sql.conditions);
        assert!(sql.conditions.contains("< '2023-11-14 22:13:20.5'"));
        assert!(
            sql.conditions
                .contains("OR (substr(col_1_stats.min_value, 20) GLOB '.[0-9]*'")
        );

        let filter = lower_one(
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            ScalarValue::TimestampMicrosecond(Some(1_700_000_000_000_000), Some("UTC".into())),
        );
        let sql = stats_filter_sql(Some(&filter), 1).expect("rendered");
        println!("ZONED:{}", sql.conditions);
        assert!(sql.conditions.contains("< '2023-11-14 22:13:20+00'"));
        assert!(
            sql.conditions
                .contains("col_1_stats.min_value GLOB '*+00' AND substr(col_1_stats.min_value, 1, length(col_1_stats.min_value) - 3) GLOB")
        );
    }

    /// The canonical-encoding tests above are only meaningful if the checks
    /// agree with what `stats_encode` actually writes.
    #[test]
    fn the_canonical_checks_accept_what_the_encoder_writes() {
        assert!(is_canonical_date("2024-01-01"));
        assert!(!is_canonical_date("+12921-08-18"));
        assert!(!is_canonical_date("-0221-09-04"));
        assert!(!is_canonical_date("2024-01-01 00:00:00"));

        assert!(is_canonical_timestamp("2023-11-14 22:13:20"));
        assert!(is_canonical_timestamp("2023-11-14 22:13:20.5"));
        assert!(is_canonical_timestamp("2023-11-14 22:13:20.123456"));
        // Equal to `.5` but ordered above it as text, so never admitted.
        assert!(!is_canonical_timestamp("2023-11-14 22:13:20.50"));
        assert!(!is_canonical_timestamp("2023-11-14 22:13:20."));
        assert!(!is_canonical_timestamp("2023-11-14 22:13:20+00"));
        assert!(!is_canonical_timestamp("2024-01-01"));

        assert!(is_canonical_timestamptz("2023-11-14 22:13:20+00"));
        assert!(is_canonical_timestamptz("2023-11-14 22:13:20.5+00"));
        // Another offset orders wrongly against `+00`, so it is not canonical.
        assert!(!is_canonical_timestamptz("2023-11-14 22:13:20+01"));
        assert!(!is_canonical_timestamptz("2023-11-14 22:13:20"));
    }
}
