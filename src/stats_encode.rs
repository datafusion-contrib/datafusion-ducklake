//! DuckDB-canonical value → `VARCHAR` encoding for DuckLake column statistics.
//!
//! DuckLake stores per-file column `min_value` / `max_value` as `VARCHAR` in
//! `ducklake_file_column_stats`, using the exact text DuckDB's `Value::ToString()`
//! produces (see the official extension: `DuckLakeUtil::StatsToString` /
//! `ColumnStatsUnifier::StatsToString`). For a catalog written here to be prunable
//! by DuckDB — and for round-trip pruning to agree — these strings must be
//! **byte-identical** to what DuckDB writes for the same logical value.
//!
//! This module is the single source of truth for that encoding. Every rule below
//! was verified against a real `duckdb` CLI (`CAST(<value> AS VARCHAR)`); the
//! `tests` module pins those golden outputs.
//!
//! ## Coverage
//!
//! Encoded exactly (min/max emitted): signed/unsigned integers, `BOOLEAN`,
//! `FLOAT`/`DOUBLE`, `DECIMAL(p,s)` (128-bit), `DATE`, and `TIMESTAMP` with or
//! without a time zone (all Arrow time units). Timezone-bearing timestamps are
//! normalized to UTC so their encoding is independent of the DuckDB session
//! time zone.
//!
//! Deliberately **not** encoded yet — [`encode_scalar`] returns `None`, so the
//! caller stores `min_value`/`max_value` as `NULL` and the file is never pruned on
//! that column (spec-safe: DuckLake keeps files whose stats are NULL). Deferred:
//! - `TIME`, `BLOB`/`BINARY` (DuckLake never prunes on blobs anyway), `UUID`,
//!   `INTERVAL`, `DECIMAL256`, and all nested types (`STRUCT`/`LIST`/`MAP`).
//!   File statistics ([`encode_file_stat`]) and the bounds of inlined rows
//!   ([`inline_stat_text`]) write `TIME` and `BLOB` as official does.
//! - Over-long strings (see [`MAX_STRING_STAT_BYTES`]) in [`encode_scalar`].
//!   File statistics arrive already truncated as DuckDB's Parquet writer
//!   truncates them (see [`crate::stats_collect::StringBounds`]), and the
//!   bounds of inlined rows keep the whole value, as official's do.

use datafusion::common::ScalarValue;

/// Strings longer than this (in UTF-8 bytes) get no text from
/// [`encode_scalar`], which the read side uses for predicate constants: such a
/// constant is not compared against stored statistics.
pub const MAX_STRING_STAT_BYTES: usize = 2048;

/// Encode a single statistic value to DuckLake's canonical `VARCHAR` form.
///
/// Returns `None` when the value is NULL, of an unsupported type, or otherwise
/// should not be stored as a min/max bound (see the module docs). A `None` result
/// means "store SQL `NULL`", never an error.
pub fn encode_scalar(value: &ScalarValue) -> Option<String> {
    match value {
        // Integers — plain decimal, no separators. i128 covers every width.
        ScalarValue::Int8(Some(v)) => Some(i128::from(*v).to_string()),
        ScalarValue::Int16(Some(v)) => Some(i128::from(*v).to_string()),
        ScalarValue::Int32(Some(v)) => Some(i128::from(*v).to_string()),
        ScalarValue::Int64(Some(v)) => Some(i128::from(*v).to_string()),
        ScalarValue::UInt8(Some(v)) => Some(i128::from(*v).to_string()),
        ScalarValue::UInt16(Some(v)) => Some(i128::from(*v).to_string()),
        ScalarValue::UInt32(Some(v)) => Some(i128::from(*v).to_string()),
        ScalarValue::UInt64(Some(v)) => Some(i128::from(*v).to_string()),

        ScalarValue::Boolean(Some(v)) => Some(
            if *v {
                "true"
            } else {
                "false"
            }
            .to_string(),
        ),

        ScalarValue::Float32(Some(v)) => encode_f32(*v),
        ScalarValue::Float64(Some(v)) => encode_f64(*v),

        ScalarValue::Decimal128(Some(v), _precision, scale) => Some(encode_decimal128(*v, *scale)),

        ScalarValue::Date32(Some(days)) => encode_date32(*days),

        ScalarValue::TimestampSecond(Some(v), timezone) => {
            encode_timestamp(*v, 1, timezone.is_some())
        },
        ScalarValue::TimestampMillisecond(Some(v), timezone) => {
            encode_timestamp(*v, 1_000, timezone.is_some())
        },
        ScalarValue::TimestampMicrosecond(Some(v), timezone) => {
            encode_timestamp(*v, 1_000_000, timezone.is_some())
        },
        ScalarValue::TimestampNanosecond(Some(v), timezone) => {
            encode_timestamp(*v, 1_000_000_000, timezone.is_some())
        },

        // Strings — stored verbatim, subject to the length guard. A bound
        // holding a `\0` byte is kept, as official keeps it in a file's
        // statistics, so a merge sees the bound official's merge sees. A
        // table-wide bound drops it ([`ColumnStats::to_global`], as official's
        // `DuckLakeUtil::StatsToString`), PostgreSQL stores none (its text holds
        // no `0x00`), and a filter constant with one is refused.
        ScalarValue::Utf8(Some(s))
        | ScalarValue::LargeUtf8(Some(s))
        | ScalarValue::Utf8View(Some(s)) => {
            if s.len() <= MAX_STRING_STAT_BYTES {
                Some(s.clone())
            } else {
                None
            }
        },

        // NULL payloads and every not-yet-supported type ⇒ store NULL.
        _ => None,
    }
}

/// `true` if `value`'s type is one [`encode_scalar`] can encode (independent of
/// whether this particular value is NULL). Used to decide whether a column is
/// eligible for pruning stats at all.
pub fn is_encodable_type(value: &ScalarValue) -> bool {
    matches!(
        value,
        ScalarValue::Int8(_)
            | ScalarValue::Int16(_)
            | ScalarValue::Int32(_)
            | ScalarValue::Int64(_)
            | ScalarValue::UInt8(_)
            | ScalarValue::UInt16(_)
            | ScalarValue::UInt32(_)
            | ScalarValue::UInt64(_)
            | ScalarValue::Boolean(_)
            | ScalarValue::Float32(_)
            | ScalarValue::Float64(_)
            | ScalarValue::Decimal128(_, _, _)
            | ScalarValue::Date32(_)
            | ScalarValue::Utf8(_)
            | ScalarValue::LargeUtf8(_)
            | ScalarValue::Utf8View(_)
    ) || matches!(
        value,
        ScalarValue::TimestampSecond(_, _)
            | ScalarValue::TimestampMillisecond(_, _)
            | ScalarValue::TimestampMicrosecond(_, _)
            | ScalarValue::TimestampNanosecond(_, _)
    )
}

/// Order two already-encoded stat strings, for widening the global
/// `ducklake_table_column_stats` min/max across files.
///
/// DuckDB-canonical encodings of dates, timestamps and strings are already
/// lexically ordered (ISO-8601 / natural), so only numeric types need a parsed
/// comparison — `"10"` must sort after `"9"`, `"-5"` before `"3"`. Pass
/// `numeric = true` for integer/float/decimal columns (see
/// [`is_numeric_ducklake_type`]). Booleans need [`stat_cmp_kind`].
pub fn stat_cmp(a: &str, b: &str, numeric: bool) -> std::cmp::Ordering {
    stat_cmp_kind(
        a,
        b,
        if numeric {
            StatKind::Numeric
        } else {
            StatKind::Text
        },
    )
}

/// How the stat strings of a column compare, as official DuckLake's
/// `RequiresValueComparison` decides: numeric, temporal and boolean bounds by
/// the value they spell, everything else as bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum StatKind {
    /// Integers, decimals and floats.
    Numeric,
    /// `BOOLEAN`, whose bounds are spelled `true`/`false` or `1`/`0`.
    Boolean,
    /// `DATE`, `TIME` and the `TIMESTAMP` types, compared as the instant they
    /// spell.
    Temporal,
    /// Everything else.
    Text,
}

impl StatKind {
    /// The comparison a column of DuckLake type `ducklake_type` needs.
    pub fn of_ducklake_type(ducklake_type: &str) -> Self {
        let t = ducklake_type.trim().to_ascii_lowercase();
        if matches!(t.as_str(), "boolean" | "bool") {
            Self::Boolean
        } else if t == "interval" {
            Self::Text
        } else if is_numeric_ducklake_type(&t) {
            Self::Numeric
        } else if matches!(
            t.as_str(),
            "date"
                | "time"
                | "timetz"
                | "time_ns"
                | "timestamp"
                | "timestamptz"
                | "timestamp_s"
                | "timestamp_ms"
                | "timestamp_ns"
        ) {
            Self::Temporal
        } else {
            Self::Text
        }
    }
}

/// Order two already-encoded stat strings of a column compared as `kind`.
///
/// A numeric NaN (`nan`, which official writes for NaN values inlined into
/// the catalog) sorts above every number, as DuckDB orders it. A side that does
/// not parse as `kind` falls back to byte order.
pub fn stat_cmp_kind(a: &str, b: &str, kind: StatKind) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    match kind {
        StatKind::Numeric => {
            // Integers: exact.
            if let (Ok(x), Ok(y)) = (a.parse::<i128>(), b.parse::<i128>()) {
                return x.cmp(&y);
            }
            // Fixed-point (DECIMAL, and finite floats DuckDB renders without an
            // exponent): exact, matching official's typed decimal comparison —
            // avoids the f64 precision loss on high-scale decimals.
            if let Some(ord) = cmp_fixed_point(a, b) {
                return ord;
            }
            // Scientific / inf / nan: float ordering with NaN greatest.
            if let (Ok(x), Ok(y)) = (a.parse::<f64>(), b.parse::<f64>()) {
                return match (x.is_nan(), y.is_nan()) {
                    (true, true) => Ordering::Equal,
                    (true, false) => Ordering::Greater,
                    (false, true) => Ordering::Less,
                    (false, false) => x.partial_cmp(&y).unwrap_or(Ordering::Equal),
                };
            }
        },
        StatKind::Boolean => {
            if let (Some(x), Some(y)) = (parse_stat_bool(a), parse_stat_bool(b)) {
                return x.cmp(&y);
            }
        },
        StatKind::Temporal => {
            if let (Some(x), Some(y)) = (parse_stat_instant(a), parse_stat_instant(b)) {
                return x.cmp(&y);
            }
        },
        StatKind::Text => {},
    }
    a.cmp(b)
}

/// A temporal bound as DuckDB spells a `DATE`, `TIME` or `TIMESTAMP` value,
/// as a point on one ordered line: `infinity` and `-infinity` at the ends, a
/// date at its midnight, a ` (BC)` year counted backwards, a time of day
/// within its day, and a zone offset folded into UTC. DuckDB writes the era
/// after the date (`0044-03-15 (BC) 12:30:00.5`), and its years run past
/// any calendar library's range, so the day count is computed here. `None`
/// for text it does not spell.
fn parse_stat_instant(text: &str) -> Option<(i8, i128)> {
    match text {
        "infinity" => return Some((1, 0)),
        "-infinity" => return Some((-1, 0)),
        _ => {},
    }
    const NANOS_PER_DAY: i128 = 86_400_000_000_000;
    // A bare time of day: `HH:MM:SS[.f][+TZ]`.
    if text.len() >= 8 && text.as_bytes().get(2) == Some(&b':') {
        return parse_time_of_day(text).map(|nanos| (0, nanos));
    }
    let (date, rest) = match text.split_once(' ') {
        Some((date, rest)) => (date, Some(rest)),
        None => (text, None),
    };
    let (bc, time) = match rest {
        Some("(BC)") => (true, None),
        Some(rest) => match rest.strip_prefix("(BC) ") {
            Some(time) => (true, Some(time)),
            None => (false, Some(rest)),
        },
        None => (false, None),
    };
    let mut parts = date.splitn(3, '-');
    let year: i64 = parse_digits(parts.next()?)?;
    let month: i64 = parse_digits(parts.next()?)?;
    let day: i64 = parse_digits(parts.next()?)?;
    // DuckDB writes year 1 BC as `0001 (BC)`, the proleptic year 0.
    let year = if bc {
        1 - year
    } else {
        year
    };
    let days = i128::from(days_from_civil(year, month, day)?);
    let nanos = match time {
        Some(time) => parse_time_of_day(time)?,
        None => 0,
    };
    Some((0, days * NANOS_PER_DAY + nanos))
}

/// A run of ASCII digits, at most 18 of them, as a number.
fn parse_digits(text: &str) -> Option<i64> {
    if text.is_empty() || text.len() > 18 || !text.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    text.parse().ok()
}

/// Days from 1970-01-01 to the proleptic Gregorian `year`-`month`-`day`, for
/// any year an `i64` day count holds. `None` for a date that does not exist.
pub(crate) fn days_from_civil(year: i64, month: i64, day: i64) -> Option<i64> {
    if !(1..=12).contains(&month) {
        return None;
    }
    let leap = year.rem_euclid(4) == 0 && (year.rem_euclid(100) != 0 || year.rem_euclid(400) == 0);
    let month_days = match month {
        2 if leap => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    };
    if !(1..=month_days).contains(&day) {
        return None;
    }
    // Howard Hinnant's `days_from_civil`, with the year starting in March.
    let year = if month <= 2 {
        year - 1
    } else {
        year
    };
    let era = year.div_euclid(400);
    let year_of_era = year.rem_euclid(400);
    let month_from_march = (month + 9) % 12;
    let day_of_year = (153 * month_from_march + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era.checked_mul(146_097)?
        .checked_add(day_of_era)?
        .checked_sub(719_468)
}

/// Nanoseconds since midnight, UTC when an offset (`+HH[:MM[:SS]]`) follows.
fn parse_time_of_day(text: &str) -> Option<i128> {
    let (clock, offset) = match text.find(['+', '-']) {
        Some(at) => (&text[..at], Some(&text[at..])),
        None => (text, None),
    };
    let (hms, fraction) = clock.split_once('.').unwrap_or((clock, ""));
    let mut fields = hms.split(':');
    let hours = i128::from(parse_digits(fields.next()?)?);
    let minutes = i128::from(parse_digits(fields.next()?)?);
    let seconds = i128::from(parse_digits(fields.next()?)?);
    if fields.next().is_some()
        || fraction.len() > 9
        || !fraction.bytes().all(|b| b.is_ascii_digit())
    {
        return None;
    }
    let fraction: i128 = if fraction.is_empty() {
        0
    } else {
        format!("{fraction:0<9}").parse().ok()?
    };
    let seconds_of_day = |hours: i128, minutes: i128, seconds: i128| {
        hours
            .checked_mul(60)?
            .checked_add(minutes)?
            .checked_mul(60)?
            .checked_add(seconds)?
            .checked_mul(1_000_000_000)
    };
    let mut nanos = seconds_of_day(hours, minutes, seconds)?.checked_add(fraction)?;
    if let Some(offset) = offset {
        let negative = offset.starts_with('-');
        let mut fields = offset[1..].split(':');
        let hours = i128::from(parse_digits(fields.next()?)?);
        let minutes = match fields.next() {
            Some(m) => i128::from(parse_digits(m)?),
            None => 0,
        };
        let seconds = match fields.next() {
            Some(s) => i128::from(parse_digits(s)?),
            None => 0,
        };
        let shift = seconds_of_day(hours, minutes, seconds)?;
        nanos = if negative {
            nanos.checked_add(shift)?
        } else {
            nanos.checked_sub(shift)?
        };
    }
    Some(nanos)
}

/// A boolean bound as official or this crate spells it.
fn parse_stat_bool(text: &str) -> Option<bool> {
    match text {
        "true" | "1" => Some(true),
        "false" | "0" => Some(false),
        _ => None,
    }
}

/// Exact comparison of two fixed-point decimal strings (optional leading `-`,
/// digits, optional single `.`, digits). Aligns the fractional parts to a common
/// scale and compares the resulting integers, so it is exact regardless of
/// magnitude/scale. Returns `None` if either side is not plain fixed-point
/// (e.g. scientific notation, `inf`, `nan`), so the caller can fall back.
fn cmp_fixed_point(a: &str, b: &str) -> Option<std::cmp::Ordering> {
    fn split(s: &str) -> Option<(bool, &str, &str)> {
        let (neg, rest) = s.strip_prefix('-').map_or((false, s), |r| (true, r));
        let (int_part, frac_part) = rest.split_once('.').unwrap_or((rest, ""));
        if int_part.is_empty() && frac_part.is_empty() {
            return None;
        }
        if !int_part.bytes().all(|c| c.is_ascii_digit())
            || !frac_part.bytes().all(|c| c.is_ascii_digit())
        {
            return None;
        }
        Some((neg, int_part, frac_part))
    }
    let (neg_a, int_a, frac_a) = split(a)?;
    let (neg_b, int_b, frac_b) = split(b)?;
    let scale = frac_a.len().max(frac_b.len());
    // Concatenate integer + fractional digits, right-padding the fraction to the
    // common scale, then parse the scaled integer (leading zeros are fine).
    let scaled = |int_part: &str, frac_part: &str| -> Option<i128> {
        format!("{int_part}{frac_part:0<scale$}")
            .parse::<i128>()
            .ok()
    };
    let mut x = scaled(int_a, frac_a)?;
    let mut y = scaled(int_b, frac_b)?;
    if neg_a {
        x = -x;
    }
    if neg_b {
        y = -y;
    }
    Some(x.cmp(&y))
}

/// Whether a DuckLake type string denotes a numeric type whose stat strings need
/// parsed (not lexical) comparison. Conservative: a false positive only changes
/// how two *present* numeric-looking strings are ordered, never correctness of
/// non-numeric columns (whose bounds compare lexically either way).
pub fn is_numeric_ducklake_type(ducklake_type: &str) -> bool {
    let t = ducklake_type.trim().to_ascii_lowercase();
    // Covers int8/16/32/64, hugeint, all unsigned uint*/ubigint/uhugeint, and
    // the bigint/integer/tinyint/smallint spellings — all contain "int".
    t.contains("int")
        || t.starts_with("decimal")
        || t.starts_with("float")
        || matches!(t.as_str(), "double" | "real")
}

/// One `ducklake_file_column_stats` row's fields relevant to the global roll-up.
#[derive(Debug, Clone)]
pub struct FileColumnStat {
    pub column_id: i64,
    pub min_value: Option<String>,
    pub max_value: Option<String>,
    pub null_count: Option<i64>,
    pub contains_nan: Option<bool>,
    /// The number of non-NULL values, as official records it.
    pub value_count: Option<i64>,
}

impl FileColumnStat {
    /// Whether the file holds no non-NULL value in this column. Official's
    /// `AnyValid` is false for such a file, so it contributes no bounds.
    fn all_null(&self) -> bool {
        self.null_count.is_some() && self.value_count == Some(0)
    }
}

/// One `ducklake_table_column_stats` row: the table-wide roll-up for a column.
/// A `None` field is stored as SQL `NULL` (unknown).
#[derive(Debug, Clone, PartialEq)]
pub struct GlobalColumnStat {
    pub column_id: i64,
    pub min_value: Option<String>,
    pub max_value: Option<String>,
    pub contains_null: Option<bool>,
    pub contains_nan: Option<bool>,
}

/// Aggregate per-file column stats into the table-wide roll-up.
///
/// `per_file` is every `(live file, column)` stats row for the table;
/// `live_file_count` is the total number of live files. `kind(column_id)` says
/// how a column's bounds compare.
///
/// The completeness rule mirrors official DuckLake (`has_min`/`has_null_count`):
/// a global field is emitted only when **every** live file contributed it —
/// otherwise it degrades to `None` (SQL `NULL` = unknown). This is essential:
/// a live file with no stats row (harvest failed, written by another tool, or a
/// compacted file we didn't stat) must NOT let the roll-up claim a definite
/// `contains_null = false` or an under-covered min/max, which a reader could use
/// to wrongly drop rows. A file whose column is all NULL contributes no bounds
/// and does not degrade them, as official's `MergeStats` skips a source with no
/// valid value. min/max additionally widen (never tighten); bounds are widened
/// with a type-aware compare ([`stat_cmp_kind`]).
///
/// Output is sorted by `column_id` for determinism.
pub fn aggregate_global_column_stats(
    per_file: &[FileColumnStat],
    live_file_count: i64,
    kind: impl Fn(i64) -> StatKind,
) -> Vec<GlobalColumnStat> {
    use std::cmp::Ordering;
    use std::collections::HashMap;

    struct Agg {
        min: Option<String>,
        max: Option<String>,
        min_present: i64,
        max_present: i64,
        nullcount_present: i64,
        nan_present: i64,
        has_null: bool,
        has_nan: bool,
        kind: StatKind,
    }

    let mut aggs: HashMap<i64, Agg> = HashMap::new();
    for f in per_file {
        let agg = aggs.entry(f.column_id).or_insert_with(|| Agg {
            min: None,
            max: None,
            min_present: 0,
            max_present: 0,
            nullcount_present: 0,
            nan_present: 0,
            has_null: false,
            has_nan: false,
            kind: kind(f.column_id),
        });
        if f.all_null() {
            agg.min_present += 1;
            agg.max_present += 1;
        } else {
            if let Some(mn) = &f.min_value {
                agg.min_present += 1;
                agg.min = Some(match agg.min.take() {
                    Some(cur) if stat_cmp_kind(&cur, mn, agg.kind) != Ordering::Greater => cur,
                    _ => mn.clone(),
                });
            }
            if let Some(mx) = &f.max_value {
                agg.max_present += 1;
                agg.max = Some(match agg.max.take() {
                    Some(cur) if stat_cmp_kind(&cur, mx, agg.kind) != Ordering::Less => cur,
                    _ => mx.clone(),
                });
            }
        }
        if let Some(nc) = f.null_count {
            agg.nullcount_present += 1;
            if nc > 0 {
                agg.has_null = true;
            }
        }
        if let Some(nan) = f.contains_nan {
            agg.nan_present += 1;
            if nan {
                agg.has_nan = true;
            }
        }
    }

    let mut out: Vec<GlobalColumnStat> = aggs
        .into_iter()
        .map(|(column_id, a)| GlobalColumnStat {
            column_id,
            min_value: (a.min_present == live_file_count)
                .then_some(a.min)
                .flatten(),
            max_value: (a.max_present == live_file_count)
                .then_some(a.max)
                .flatten(),
            contains_null: (a.nullcount_present == live_file_count).then_some(a.has_null),
            // Tri-state, same coverage rule as the other fields and the same
            // semantics as official DuckLake's MergeStats: known (OR-reduced)
            // only when EVERY live file reported a NaN flag, else NULL/unknown.
            // A definite `false` is what lets readers trust a float max as an
            // upper bound (NaN would sort above it), so it must never be
            // claimed while any file's NaN state is unknown.
            contains_nan: (a.nan_present == live_file_count).then_some(a.has_nan),
        })
        .collect();
    out.sort_by_key(|g| g.column_id);
    out
}

/// One column's or nested leaf field's statistics while they are merged into a
/// table's roll-up, as official DuckLake's `DuckLakeColumnStats` holds them. A
/// `None` field is one the statistics do not know.
#[derive(Debug, Clone, PartialEq)]
pub struct ColumnStats {
    /// The smallest non-NULL value's text.
    pub min_value: Option<String>,
    /// The largest non-NULL value's text.
    pub max_value: Option<String>,
    /// The number of NULL values; a stored roll-up only says whether it is
    /// zero.
    pub null_count: Option<u64>,
    /// The number of values, NULL ones included.
    pub num_values: Option<u64>,
    pub contains_nan: Option<bool>,
    /// Whether some value is not NULL, when `num_values` and `null_count` do
    /// not say.
    pub any_valid: bool,
}

impl ColumnStats {
    /// A stored `ducklake_table_column_stats` row, read as official's
    /// `FromGlobalStats` reads it: a row with neither bound holds no valid
    /// value.
    pub fn from_global(stat: &GlobalColumnStat) -> Self {
        Self {
            min_value: stat.min_value.clone(),
            max_value: stat.max_value.clone(),
            null_count: stat.contains_null.map(u64::from),
            num_values: None,
            contains_nan: stat.contains_nan,
            any_valid: stat.min_value.is_some() || stat.max_value.is_some(),
        }
    }

    /// One data file's `ducklake_file_column_stats` row.
    pub fn from_file(stat: &FileColumnStat) -> Self {
        let null_count = stat.null_count.and_then(|n| u64::try_from(n).ok());
        let num_values = match (stat.value_count, null_count) {
            (Some(values), Some(nulls)) => u64::try_from(values).ok().map(|v| v + nulls),
            _ => None,
        };
        Self {
            min_value: stat.min_value.clone(),
            max_value: stat.max_value.clone(),
            null_count,
            num_values,
            contains_nan: stat.contains_nan,
            any_valid: true,
        }
    }

    /// The statistics of rows written inline, which official gathers with a
    /// NULL count but no value count or NaN flag.
    pub fn from_inlined(stat: &InlinedColumnStat) -> Self {
        Self {
            min_value: stat.min_value.clone(),
            max_value: stat.max_value.clone(),
            null_count: Some(u64::from(stat.contains_null)),
            num_values: None,
            contains_nan: None,
            any_valid: stat.any_valid,
        }
    }

    /// Whether some value is not NULL, as official's `AnyValid` decides.
    fn has_valid(&self) -> bool {
        match (self.num_values, self.null_count) {
            (Some(values), Some(nulls)) => values > nulls,
            _ => self.any_valid,
        }
    }

    /// Merge `new` in, as official's `DuckLakeColumnStats::MergeStats` does:
    /// counts add up and a flag ORs, each unknown once either side's is;
    /// values that are all NULL leave the bounds alone; bounds of statistics
    /// that held no valid value before this merge are replaced by `new`'s; otherwise each bound
    /// widens, compared as `kind`, and is unknown once either side's is.
    pub fn merge(&mut self, new: &Self, kind: StatKind) {
        use std::cmp::Ordering;
        // Whether every value so far was NULL, from the counts before this
        // merge adds `new`'s: upstream main (`had_only_nulls`) gives such
        // statistics `new`'s bounds. 1.5.5 asks after adding the counts, so a
        // commit whose first file holds only NULLs loses the bounds of the
        // files after it.
        let had_valid = self.has_valid();
        self.null_count = match (self.null_count, new.null_count) {
            (Some(a), Some(b)) => Some(a + b),
            _ => None,
        };
        self.num_values = match (self.num_values, new.num_values) {
            (Some(a), Some(b)) => Some(a + b),
            _ => None,
        };
        self.contains_nan = match (self.contains_nan, new.contains_nan) {
            (Some(a), Some(b)) => Some(a || b),
            _ => None,
        };
        if !new.has_valid() {
            return;
        }
        if !had_valid {
            self.min_value = new.min_value.clone();
            self.max_value = new.max_value.clone();
            self.any_valid = true;
            return;
        }
        let widen = |stored: &mut Option<String>, value: &Option<String>, keep: Ordering| {
            *stored = match (stored.take(), value) {
                (Some(current), Some(value)) => {
                    Some(if stat_cmp_kind(value, &current, kind) == keep {
                        value.clone()
                    } else {
                        current
                    })
                },
                _ => None,
            };
        };
        widen(&mut self.min_value, &new.min_value, Ordering::Less);
        widen(&mut self.max_value, &new.max_value, Ordering::Greater);
    }

    /// The `ducklake_table_column_stats` row official writes for these
    /// statistics: `contains_null` says whether the NULL count is above zero,
    /// and a bound holding a NUL byte is stored as NULL, as its
    /// `StatsToString` writes it.
    pub fn to_global(&self, column_id: i64) -> GlobalColumnStat {
        let stored = |bound: &Option<String>| bound.clone().filter(|text| !text.contains('\0'));
        GlobalColumnStat {
            column_id,
            min_value: stored(&self.min_value),
            max_value: stored(&self.max_value),
            contains_null: self.null_count.map(|nulls| nulls > 0),
            contains_nan: self.contains_nan,
        }
    }
}

/// Merge the statistics of newly written data into a table's stored
/// roll-up, as official DuckLake does when it commits new data files or
/// inlined rows, and return the rows to store.
///
/// `initialized` says whether the table had a `ducklake_table_stats` row
/// before this write. Without one, official starts from empty statistics and
/// inserts a row for every field `new` covers. With one, it starts from the
/// stored rows (see [`ColumnStats::from_global`]) and only updates them: a
/// field with no stored row — a column added since, or a nested column's
/// container — keeps none. A field some of `new` has values but no bounds
/// for, other than for NaN, gets no row: its bounds are unknown, which a row
/// cannot say. `new` is merged in order. Output is sorted by `column_id`.
pub fn merge_table_column_stats(
    existing: Vec<GlobalColumnStat>,
    initialized: bool,
    new: impl IntoIterator<Item = (i64, ColumnStats)>,
    kind: impl Fn(i64) -> StatKind,
) -> Vec<GlobalColumnStat> {
    use std::collections::BTreeMap;

    let mut merged: BTreeMap<i64, ColumnStats> = BTreeMap::new();
    if initialized {
        for stat in &existing {
            merged.insert(stat.column_id, ColumnStats::from_global(stat));
        }
    }
    let mut added = std::collections::BTreeSet::new();
    let mut touched = std::collections::BTreeSet::new();
    let mut unknown = std::collections::BTreeSet::new();
    for (column_id, stats) in new {
        touched.insert(column_id);
        // Values whose bounds are not known, which official's writers only
        // produce for NaN (whose flag keeps a float's bounds unused), would
        // leave a row with neither bound: one official reads as holding no
        // valid value, and a later write would then narrow it to its own
        // bounds. The field gets no row instead.
        if stats.has_valid()
            && (stats.min_value.is_none() || stats.max_value.is_none())
            && stats.contains_nan != Some(true)
        {
            unknown.insert(column_id);
        }
        match merged.get_mut(&column_id) {
            Some(current) => current.merge(&stats, kind(column_id)),
            None => {
                added.insert(column_id);
                merged.insert(column_id, stats);
            },
        }
    }
    // A bound holding a NUL byte cannot be stored (see `to_global`), so it
    // is unknown too, as a Parquet file's NUL bound is; official writes the
    // row without it, which official then reads as holding no valid value
    // beyond the other bound. Only a merged bound counts: a file's NUL bound
    // that another bound supersedes is not stored by official either.
    let nul = |bound: &Option<String>| bound.as_ref().is_some_and(|text| text.contains('\0'));
    merged
        .into_iter()
        .filter(|(column_id, _)| !initialized || !added.contains(column_id))
        .filter(|(column_id, _)| !unknown.contains(column_id))
        .filter(|(column_id, stats)| {
            !touched.contains(column_id) || !(nul(&stats.min_value) || nul(&stats.max_value))
        })
        // Nor may the merge itself leave a row with neither bound: a file
        // whose float values are NaN, or whose writer omitted the bounds for
        // one, has none, and merged in it leaves the column with none.
        // Official stores that row, reads it as holding no value, and narrows
        // it to the next write's bounds; here the field gets no row.
        .filter(|(column_id, stats)| {
            !(touched.contains(column_id)
                && stats.has_valid()
                && stats.min_value.is_none()
                && stats.max_value.is_none())
        })
        .map(|(column_id, stats)| stats.to_global(column_id))
        .collect()
}

/// Merge newly committed data files' statistics, `files` in commit order and
/// each holding its `ducklake_file_column_stats` rows, into a table's stored
/// roll-up; see [`merge_table_column_stats`].
///
/// `live_fields` are the field ids of the table's columns. A file that records
/// no statistics for one of them while another file or the stored roll-up
/// does, which official's writers never commit, leaves that field's bounds
/// unknown; a stored row cannot say so — official reads one with neither
/// bound as holding no valid value — so the field gets no row.
pub fn merge_file_column_stats(
    existing: Vec<GlobalColumnStat>,
    initialized: bool,
    files: &[Vec<FileColumnStat>],
    live_fields: &[i64],
    kind: impl Fn(i64) -> StatKind,
) -> Vec<GlobalColumnStat> {
    use std::collections::BTreeSet;

    let mut covered: BTreeSet<i64> = files.iter().flatten().map(|stat| stat.column_id).collect();
    if initialized {
        covered.extend(existing.iter().map(|stat| stat.column_id));
    }
    let unknown: BTreeSet<i64> = covered
        .into_iter()
        .filter(|id| live_fields.contains(id))
        .filter(|id| {
            files
                .iter()
                .any(|file| !file.iter().any(|stat| stat.column_id == *id))
        })
        .collect();
    let mut merged = merge_table_column_stats(
        existing,
        initialized,
        files
            .iter()
            .flatten()
            .map(|stat| (stat.column_id, ColumnStats::from_file(stat))),
        kind,
    );
    merged.retain(|stat| !unknown.contains(&stat.column_id));
    merged
}

/// The statistics of one column or nested leaf field over rows written inline,
/// as official DuckLake's `DuckLakeInlineData` gathers them before merging them
/// into the table-wide roll-up.
#[derive(Debug, Clone, PartialEq)]
pub struct InlinedColumnStat {
    /// The field id: a top-level column's, or a nested leaf field's.
    pub column_id: i64,
    /// Whether any written value is non-NULL.
    pub any_valid: bool,
    pub contains_null: bool,
    /// The bounds of the non-NULL values, as DuckDB casts them to `VARCHAR`;
    /// `None` when some non-NULL value has no text here (see
    /// [`inline_stat_text`]).
    pub min_value: Option<String>,
    pub max_value: Option<String>,
}

/// A value written inline as official's inlined statistics spell it: DuckDB's
/// `CAST(value AS VARCHAR)`. Beyond [`encode_scalar`], that is NaN as `nan`,
/// `TIME` as `HH:MM:SS[.fraction]`, a blob with every byte outside printable
/// ASCII (and `'`, `"`, `\`) escaped as `\xHH`, and strings of any length.
/// `ducklake_type` tells a `UUID` from a 16-byte blob. `None` for a type this
/// crate has no DuckDB text for (`INTERVAL`, `UUID` among them), which leaves
/// the bounds unknown.
pub fn inline_stat_text(value: &ScalarValue, ducklake_type: &str) -> Option<String> {
    let ducklake_type = ducklake_type.trim().to_ascii_lowercase();
    match value {
        ScalarValue::Float32(Some(v)) if v.is_nan() => Some(nan_text(v.is_sign_negative())),
        ScalarValue::Float64(Some(v)) if v.is_nan() => Some(nan_text(v.is_sign_negative())),
        ScalarValue::Utf8(Some(s))
        | ScalarValue::LargeUtf8(Some(s))
        | ScalarValue::Utf8View(Some(s)) => Some(s.clone()),
        ScalarValue::Time32Second(Some(v)) => encode_time(i64::from(*v), 1),
        ScalarValue::Time32Millisecond(Some(v)) => encode_time(i64::from(*v), 1_000),
        ScalarValue::Time64Microsecond(Some(v)) => encode_time(*v, 1_000_000),
        ScalarValue::Time64Nanosecond(Some(v)) => encode_time(*v, 1_000_000_000),
        ScalarValue::Binary(Some(bytes))
        | ScalarValue::LargeBinary(Some(bytes))
        | ScalarValue::BinaryView(Some(bytes)) => Some(blob_text(bytes)),
        ScalarValue::FixedSizeBinary(_, Some(bytes)) if ducklake_type == "blob" => {
            Some(blob_text(bytes))
        },
        _ => encode_scalar(value),
    }
}

fn nan_text(negative: bool) -> String {
    if negative {
        "-nan".to_string()
    } else {
        "nan".to_string()
    }
}

/// DuckDB's `TIME` text: `HH:MM:SS`, then the fraction with trailing zeros
/// trimmed, when it is not zero.
fn encode_time(value: i64, units_per_second: i64) -> Option<String> {
    // DuckDB's `TIME` runs from `00:00:00` to `24:00:00` inclusive.
    if value < 0 || i128::from(value) > 86_400 * i128::from(units_per_second) {
        return None;
    }
    let secs = value / units_per_second;
    let sub = value % units_per_second;
    let (hours, minutes, seconds) = (secs / 3600, secs / 60 % 60, secs % 60);
    let width = match units_per_second {
        1_000 => 3,
        1_000_000 => 6,
        1_000_000_000 => 9,
        _ => 0,
    };
    let mut text = format!("{hours:02}:{minutes:02}:{seconds:02}");
    if sub != 0 {
        let fraction = format!("{sub:0width$}");
        text.push('.');
        text.push_str(fraction.trim_end_matches('0'));
    }
    Some(text)
}

/// DuckDB's `BLOB` text: printable ASCII other than `'`, `"` and `\` as is,
/// every other byte as `\xHH`.
fn blob_text(bytes: &[u8]) -> String {
    let mut text = String::with_capacity(bytes.len());
    for byte in bytes {
        if (32..=126).contains(byte) && !matches!(byte, b'\'' | b'"' | b'\\') {
            text.push(char::from(*byte));
        } else {
            text.push_str(&format!("\\x{byte:02X}"));
        }
    }
    text
}

/// File-level stat text for `value`, as official's Parquet statistics spell it:
/// [`encode_scalar`], except that a `BOOLEAN` bound is `0` or `1`, a blob bound
/// is its bytes in upper-case hex, and a `TIME` bound is DuckDB's `TIME` text.
pub fn encode_file_stat(value: &ScalarValue) -> Option<String> {
    match value {
        ScalarValue::Boolean(Some(v)) => Some(
            if *v {
                "1"
            } else {
                "0"
            }
            .to_string(),
        ),
        ScalarValue::Binary(Some(bytes))
        | ScalarValue::LargeBinary(Some(bytes))
        | ScalarValue::BinaryView(Some(bytes)) => {
            Some(bytes.iter().map(|byte| format!("{byte:02X}")).collect())
        },
        ScalarValue::Time32Second(Some(v)) => encode_time(i64::from(*v), 1),
        ScalarValue::Time32Millisecond(Some(v)) => encode_time(i64::from(*v), 1_000),
        ScalarValue::Time64Microsecond(Some(v)) => encode_time(*v, 1_000_000),
        ScalarValue::Time64Nanosecond(Some(v)) => encode_time(*v, 1_000_000_000),
        _ => encode_scalar(value),
    }
}

/// Gather [`InlinedColumnStat`]s for rows written inline, for every top-level
/// column of a scalar type and every leaf field of a nested one, as official's
/// `UpdateStats` recurses into `STRUCT`, `LIST` and `MAP`.
///
/// `fields` are the table's columns flattened depth first — a column, then its
/// children — each with its field id and DuckLake type, as
/// `ducklake_column` lists them; column `i` of every batch is the `i`th
/// top-level column there. `fill(i)` gives the value stored in place of a NULL
/// in top-level column `i`, when the writer stores one.
pub fn inlined_column_stats(
    batches: &[arrow::record_batch::RecordBatch],
    fields: &[(i64, String)],
    fill: impl Fn(usize) -> Option<ScalarValue>,
) -> datafusion::common::Result<Vec<InlinedColumnStat>> {
    let mut stats: std::collections::BTreeMap<i64, LeafStat> = std::collections::BTreeMap::new();
    for batch in batches {
        let mut next = 0usize;
        for (index, array) in batch.columns().iter().enumerate() {
            let positions: Vec<(usize, bool)> = (0..array.len()).map(|row| (row, true)).collect();
            let fill = fill(index);
            gather_field(
                array.as_ref(),
                &positions,
                fields,
                &mut next,
                fill.as_ref(),
                &mut stats,
            )?;
        }
    }
    Ok(stats
        .into_iter()
        .map(|(column_id, leaf)| leaf.finish(column_id))
        .collect())
}

/// One field's statistics while rows are gathered.
struct LeafStat {
    kind: StatKind,
    any_valid: bool,
    contains_null: bool,
    min: Option<String>,
    max: Option<String>,
    bounds_known: bool,
}

impl LeafStat {
    fn add(&mut self, value: Option<String>) {
        use std::cmp::Ordering;
        self.any_valid = true;
        let Some(text) = value else {
            self.bounds_known = false;
            return;
        };
        if self
            .min
            .as_ref()
            .is_none_or(|min| stat_cmp_kind(&text, min, self.kind) == Ordering::Less)
        {
            self.min = Some(text.clone());
        }
        if self
            .max
            .as_ref()
            .is_none_or(|max| stat_cmp_kind(&text, max, self.kind) == Ordering::Greater)
        {
            self.max = Some(text);
        }
    }

    fn finish(self, column_id: i64) -> InlinedColumnStat {
        let (min_value, max_value) = if self.bounds_known {
            (self.min, self.max)
        } else {
            (None, None)
        };
        InlinedColumnStat {
            column_id,
            any_valid: self.any_valid,
            contains_null: self.contains_null,
            min_value,
            max_value,
        }
    }
}

/// Gather the statistics of `array`'s values at `positions` — `(index, valid)`
/// pairs, `valid` false where an enclosing struct is NULL — into the field at
/// `fields[*next]` and, for a nested type, its descendants, advancing `*next`
/// past them.
fn gather_field(
    array: &dyn arrow::array::Array,
    positions: &[(usize, bool)],
    fields: &[(i64, String)],
    next: &mut usize,
    fill: Option<&ScalarValue>,
    stats: &mut std::collections::BTreeMap<i64, LeafStat>,
) -> datafusion::common::Result<()> {
    use arrow::array::{Array, AsArray};
    use arrow::datatypes::DataType;

    let Some((field_id, ducklake_type)) = fields.get(*next) else {
        return Err(datafusion::common::DataFusionError::Internal(
            "inlined statistics: more fields than the table has".to_string(),
        ));
    };
    *next += 1;
    // The positions of a list's or map's elements: those of every valid row.
    let element_positions = |offsets: &[usize]| -> Vec<(usize, bool)> {
        let mut out = Vec::new();
        for (row, valid) in positions {
            if *valid && array.is_valid(*row) {
                out.extend((offsets[*row]..offsets[*row + 1]).map(|element| (element, true)));
            }
        }
        out
    };
    match array.data_type() {
        DataType::Struct(_) => {
            let structs = array.as_struct();
            let child_positions: Vec<(usize, bool)> = positions
                .iter()
                .map(|(row, valid)| (*row, *valid && structs.is_valid(*row)))
                .collect();
            for child in structs.columns() {
                gather_field(child.as_ref(), &child_positions, fields, next, None, stats)?;
            }
        },
        DataType::List(_) => {
            let list = array.as_list::<i32>();
            let offsets: Vec<usize> = list.offsets().iter().map(|o| *o as usize).collect();
            let elements = element_positions(&offsets);
            gather_field(list.values().as_ref(), &elements, fields, next, None, stats)?;
        },
        DataType::LargeList(_) => {
            let list = array.as_list::<i64>();
            let offsets: Vec<usize> = list.offsets().iter().map(|o| *o as usize).collect();
            let elements = element_positions(&offsets);
            gather_field(list.values().as_ref(), &elements, fields, next, None, stats)?;
        },
        DataType::FixedSizeList(_, size) => {
            let list = array.as_fixed_size_list();
            let size = *size as usize;
            let offsets: Vec<usize> = (0..=list.len()).map(|row| row * size).collect();
            let elements = element_positions(&offsets);
            gather_field(list.values().as_ref(), &elements, fields, next, None, stats)?;
        },
        DataType::Map(_, _) => {
            let map = array.as_map();
            let offsets: Vec<usize> = map.offsets().iter().map(|o| *o as usize).collect();
            let entries = element_positions(&offsets);
            gather_field(map.keys().as_ref(), &entries, fields, next, None, stats)?;
            gather_field(map.values().as_ref(), &entries, fields, next, None, stats)?;
        },
        _ => {
            let leaf = stats.entry(*field_id).or_insert_with(|| LeafStat {
                kind: StatKind::of_ducklake_type(ducklake_type),
                any_valid: false,
                contains_null: false,
                min: None,
                max: None,
                bounds_known: true,
            });
            for (row, valid) in positions {
                let mut value = if *valid {
                    ScalarValue::try_from_array(array, *row)?
                } else {
                    ScalarValue::Null
                };
                if value.is_null()
                    && let Some(stored) = fill
                {
                    value = stored.clone();
                }
                if value.is_null() {
                    leaf.contains_null = true;
                    continue;
                }
                leaf.add(inline_stat_text(&value, ducklake_type));
            }
        },
    }
    Ok(())
}

/// Merge the statistics of rows written inline into a table's stored roll-up,
/// as official DuckLake does on an inlined write; see
/// [`merge_table_column_stats`].
pub fn merge_inlined_column_stats(
    existing: Vec<GlobalColumnStat>,
    inlined: Vec<InlinedColumnStat>,
    initialized: bool,
    kind: impl Fn(i64) -> StatKind,
) -> Vec<GlobalColumnStat> {
    merge_table_column_stats(
        existing,
        initialized,
        inlined
            .iter()
            .map(|stat| (stat.column_id, ColumnStats::from_inlined(stat))),
        kind,
    )
}

// --------------------------------------------------------------------------
// Floating point — DuckDB's shortest-round-trip rendering (matches Python
// `repr`): fixed notation when the decimal exponent of the leading digit is in
// [-4, 16), else scientific with a signed, ≥2-digit zero-padded exponent. Whole
// values still carry a `.0`. NaN ⇒ None (DuckLake omits min/max when NaN is
// present); ±inf ⇒ "inf"/"-inf"; ±0.0 ⇒ "0.0"/"-0.0", as DuckDB spells them.
// --------------------------------------------------------------------------

fn encode_f64(v: f64) -> Option<String> {
    if v.is_nan() {
        return None;
    }
    if v.is_infinite() {
        return Some(
            if v < 0.0 {
                "-inf"
            } else {
                "inf"
            }
            .to_string(),
        );
    }
    if v == 0.0 {
        return Some(
            if v.is_sign_negative() {
                "-0.0"
            } else {
                "0.0"
            }
            .to_string(),
        );
    }
    // Rust's `{:e}` yields the shortest round-trip digits in normalized
    // `d[.ddd]e<exp>` form (no sign/pad on the exponent); reformat to DuckDB's.
    let sci = ties_to_even(
        format!("{:e}", v.abs()),
        &format!("{:.800e}", v.abs()),
        |text| text.parse::<f64>().ok() == Some(v.abs()),
    );
    Some(format_shortest(&sci, v.is_sign_negative()))
}

fn encode_f32(v: f32) -> Option<String> {
    if v.is_nan() {
        return None;
    }
    if v.is_infinite() {
        return Some(
            if v < 0.0 {
                "-inf"
            } else {
                "inf"
            }
            .to_string(),
        );
    }
    if v == 0.0 {
        return Some(
            if v.is_sign_negative() {
                "-0.0"
            } else {
                "0.0"
            }
            .to_string(),
        );
    }
    // Format from the f32 (not widened to f64) so the shortest digits are the
    // f32's own.
    Some(format_shortest(
        &format!("{:e}", v.abs()),
        v.is_sign_negative(),
    ))
}

/// The shortest digits of a value as DuckDB picks them: where two spellings of
/// the shortest length lie equally close to the value (it sits exactly
/// halfway between them), DuckDB takes the one with the even last digit, and
/// Rust's `{:e}` the larger (`768223024721377.25` is `…377.2` to DuckDB,
/// `…377.3` to Rust). `shortest` is Rust's `{:e}`, `exact` the value's exact
/// expansion in `{:.Ne}` form, and `reads_back` whether a spelling parses to
/// the value.
fn ties_to_even(shortest: String, exact: &str, reads_back: impl Fn(&str) -> bool) -> String {
    let split = |sci: &str| -> Option<(String, i32)> {
        let (mantissa, exp) = sci.split_once('e')?;
        let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
        Some((digits, exp.parse().ok()?))
    };
    let (Some((digits, exp)), Some((exact_digits, exact_exp))) = (split(&shortest), split(exact))
    else {
        return shortest;
    };
    let Ok(value) = digits.parse::<u64>() else {
        return shortest;
    };
    if value % 2 == 0 || digits.len() > 18 {
        return shortest;
    }
    let exact_digits = exact_digits.trim_end_matches('0');
    // A neighbour `other` of the shortest digits, and the midpoint between
    // them (`2·value ± 1` in units of half the last digit).
    for (other, midpoint) in [(value - 1, value * 10 - 5), (value + 1, value * 10 + 5)] {
        let midpoint_text = midpoint.to_string();
        let midpoint_exp = exp - digits.len() as i32 + midpoint_text.len() as i32 - 1;
        if midpoint_text != exact_digits || midpoint_exp != exact_exp {
            continue;
        }
        let other_text = other.to_string();
        let other_exp = exp - digits.len() as i32 + other_text.len() as i32;
        let trimmed = other_text.trim_end_matches('0');
        let candidate = match trimmed.len() {
            0 => continue,
            1 => format!("{trimmed}e{other_exp}"),
            _ => format!("{}.{}e{other_exp}", &trimmed[..1], &trimmed[1..]),
        };
        if reads_back(&candidate) {
            return candidate;
        }
    }
    shortest
}

/// Reformat Rust's `{:e}` output (e.g. `"3.14e0"`, `"1e16"`, `"5e-308"`) for a
/// non-zero, finite magnitude into DuckDB's canonical form, applying `neg`.
fn format_shortest(sci: &str, neg: bool) -> String {
    let (mantissa, exp_str) = sci.split_once('e').expect("`{:e}` always has 'e'");
    let exp10: i32 = exp_str.parse().expect("`{:e}` exponent is an integer");
    // Significant digits with no dot; `{:e}` never emits trailing zeros.
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();

    let body = if (-4..16).contains(&exp10) {
        format_fixed(&digits, exp10)
    } else {
        format_scientific(&digits, exp10)
    };
    if neg {
        format!("-{body}")
    } else {
        body
    }
}

/// Fixed-point layout of `digits` whose leading digit has power-of-ten `exp10`
/// (guaranteed `-4 <= exp10 < 16`).
fn format_fixed(digits: &str, exp10: i32) -> String {
    if exp10 >= 0 {
        let int_len = exp10 as usize + 1;
        if digits.len() <= int_len {
            // All digits are integral; pad and add the mandatory ".0".
            let zeros = int_len - digits.len();
            format!("{digits}{}.0", "0".repeat(zeros))
        } else {
            let (int_part, frac_part) = digits.split_at(int_len);
            format!("{int_part}.{frac_part}")
        }
    } else {
        // 0.00…<digits>, with (-exp10 - 1) leading zeros after the point.
        let zeros = (-exp10 - 1) as usize;
        format!("0.{}{digits}", "0".repeat(zeros))
    }
}

/// Scientific layout: `d[.ddd]e±EE` with the exponent signed and ≥2 digits.
fn format_scientific(digits: &str, exp10: i32) -> String {
    let mantissa = if digits.len() == 1 {
        digits.to_string()
    } else {
        format!("{}.{}", &digits[..1], &digits[1..])
    };
    let sign = if exp10 < 0 {
        '-'
    } else {
        '+'
    };
    format!("{mantissa}e{sign}{:02}", exp10.abs())
}

// --------------------------------------------------------------------------
// DECIMAL(p, s) — fixed-point text from the 128-bit unscaled value and scale.
// --------------------------------------------------------------------------

fn encode_decimal128(value: i128, scale: i8) -> String {
    if scale <= 0 {
        // Scale 0 (or the rare negative scale) renders as a plain integer.
        return value.to_string();
    }
    let scale = scale as usize;
    let neg = value < 0;
    let digits = value.unsigned_abs().to_string();
    let body = if digits.len() > scale {
        let (int_part, frac_part) = digits.split_at(digits.len() - scale);
        format!("{int_part}.{frac_part}")
    } else {
        // |value| < 1: "0." then left-pad the fraction to `scale` digits.
        format!("0.{digits:0>scale$}")
    };
    if neg {
        format!("-{body}")
    } else {
        body
    }
}

// --------------------------------------------------------------------------
// DATE / TIMESTAMP as DuckDB spells them.
// --------------------------------------------------------------------------

/// DuckDB's `date_t` sentinels: `infinity` is `i32::MAX`, `-infinity` its
/// negation, and every day between them names a date. `i32::MIN` is no date.
const DATE_INFINITY: i32 = i32::MAX;

/// DuckDB's `timestamp_t` sentinels, shared by every unit: `infinity` is
/// `i64::MAX` and `-infinity` its negation. `i64::MIN` is no timestamp.
const TIMESTAMP_INFINITY: i64 = i64::MAX;

/// The microseconds of DuckDB's earliest (`290309-12-22 (BC) 00:00:00`) and
/// latest (`294247-01-10 04:00:54.775806`) finite `TIMESTAMP`. A second,
/// millisecond or microsecond value outside them is no DuckDB timestamp.
const TIMESTAMP_MIN_MICROS: i64 = -9_223_372_022_400_000_000;
const TIMESTAMP_MAX_MICROS: i64 = 9_223_372_036_854_775_806;

/// The nanoseconds of DuckDB's earliest finite `TIMESTAMP_NS`,
/// `1677-09-22 00:00:00`; every larger value short of `infinity` is one.
const TIMESTAMP_NS_MIN: i64 = -9_223_286_400_000_000_000;

fn encode_date32(days: i32) -> Option<String> {
    match days {
        DATE_INFINITY => Some("infinity".to_string()),
        days if days == -DATE_INFINITY => Some("-infinity".to_string()),
        i32::MIN => None,
        days => Some(duckdb_date_text(i64::from(days))),
    }
}

/// The proleptic Gregorian `(year, month, day)` `days` after 1970-01-01, for
/// any day count (Howard Hinnant's `civil_from_days`).
pub(crate) fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let day_of_era = z.rem_euclid(146_097);
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_from_march = (5 * day_of_year + 2) / 153;
    let day = (day_of_year - (153 * month_from_march + 2) / 5 + 1) as u32;
    let month = if month_from_march < 10 {
        month_from_march + 3
    } else {
        month_from_march - 9
    } as u32;
    let year = year_of_era + era * 400 + i64::from(month <= 2);
    (year, month, day)
}

/// DuckDB's text of the date `days` after 1970-01-01: `YYYY-MM-DD` with the
/// year zero-padded to four digits (more when it has them, with no sign), and
/// ` (BC)` after a date before year 1, whose year counts back from 1 BC.
fn duckdb_date_text(days: i64) -> String {
    let (year, month, day) = civil_from_days(days);
    if year <= 0 {
        format!("{:04}-{month:02}-{day:02} (BC)", 1 - year)
    } else {
        format!("{year:04}-{month:02}-{day:02}")
    }
}

/// DuckDB's text of a `DATE`, `TIME` or `TIMESTAMP` value, as official stores
/// inlined temporal values and statistics; `None` for other types or a value
/// DuckDB's type does not hold.
pub(crate) fn duckdb_temporal_text(value: &ScalarValue) -> Option<String> {
    match value {
        ScalarValue::Date32(Some(_))
        | ScalarValue::TimestampSecond(Some(_), _)
        | ScalarValue::TimestampMillisecond(Some(_), _)
        | ScalarValue::TimestampMicrosecond(Some(_), _)
        | ScalarValue::TimestampNanosecond(Some(_), _) => encode_scalar(value),
        ScalarValue::Time32Second(Some(v)) => encode_time(i64::from(*v), 1),
        ScalarValue::Time32Millisecond(Some(v)) => encode_time(i64::from(*v), 1_000),
        ScalarValue::Time64Microsecond(Some(v)) => encode_time(*v, 1_000_000),
        ScalarValue::Time64Nanosecond(Some(v)) => encode_time(*v, 1_000_000_000),
        _ => None,
    }
}

/// Encode a timestamp given its value in `units_per_second` sub-second units
/// (1 = seconds, 1_000 = millis, 1_000_000 = micros, 1e9 = nanos), as DuckDB
/// spells it. `None` for a value DuckDB has no timestamp for.
fn encode_timestamp(value: i64, units_per_second: i64, timezone: bool) -> Option<String> {
    match value {
        TIMESTAMP_INFINITY => return Some("infinity".to_string()),
        value if value == -TIMESTAMP_INFINITY => return Some("-infinity".to_string()),
        i64::MIN => return None,
        _ => {},
    }
    if units_per_second <= 1_000_000 {
        let micros = i128::from(value) * i128::from(1_000_000 / units_per_second);
        if micros < i128::from(TIMESTAMP_MIN_MICROS) || micros > i128::from(TIMESTAMP_MAX_MICROS) {
            return None;
        }
    } else if value < TIMESTAMP_NS_MIN {
        return None;
    }
    let secs = value.div_euclid(units_per_second);
    let sub = value.rem_euclid(units_per_second); // 0..units_per_second
    let days = secs.div_euclid(86_400);
    let second_of_day = secs.rem_euclid(86_400);
    let base = format!(
        "{} {:02}:{:02}:{:02}",
        duckdb_date_text(days),
        second_of_day / 3_600,
        second_of_day / 60 % 60,
        second_of_day % 60
    );

    // Fractional seconds: render with the unit's native width, then trim
    // trailing zeros; omit entirely when zero (DuckDB: ".12", ".123456", "").
    let frac = match units_per_second {
        1 => String::new(),
        1_000 => format!("{sub:03}"),
        1_000_000 => format!("{sub:06}"),
        1_000_000_000 => format!("{sub:09}"),
        _ => String::new(),
    };
    let frac = frac.trim_end_matches('0');
    let mut encoded = if frac.is_empty() {
        base
    } else {
        format!("{base}.{frac}")
    };
    if timezone {
        encoded.push_str("+00");
    }
    Some(encoded)
}

/// Whether `text` matches `shape`, where `N` stands for any ASCII digit and
/// every other byte must match exactly.
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

/// Whether `text` is a date exactly as [`encode_scalar`] writes one.
///
/// `YYYY-MM-DD`, fixed width, so byte order is chronological order. A dialect
/// with no date type can compare two of these as text; one that casts still
/// wants the check, because a cast is only as good as what it is given.
///
/// The width is what matters, not the calendar. A year past 9999 has a fifth
/// digit (`12345-01-01` sorts below `2024-01-01`), a date before the common era
/// carries ` (BC)`, and other writers' `+12345` and `-0044` sort below every
/// digit — so admitting any of them would invert a comparison against an
/// ordinary date.
pub fn is_canonical_date(text: &str) -> bool {
    matches_shape(text, "NNNN-NN-NN")
}

/// Whether `text` is a naive timestamp exactly as [`encode_scalar`] writes one.
///
/// `YYYY-MM-DD HH:MM:SS` with an optional fraction. The time unit does not
/// matter — only the written shape does. A shorter string is a prefix of a
/// longer one and so sorts below it, which is correct: no fraction is a zero
/// fraction.
///
/// A fraction ending in `0` is refused. `encode_scalar` trims trailing zeros, so
/// `.5` and `.50` cannot both be written for the same instant — but a foreign
/// writer might emit `.50`, and as text that sorts *above* `.5` while naming an
/// earlier-or-equal instant.
pub fn is_canonical_timestamp(text: &str) -> bool {
    let Some((base, fraction)) = text.split_at_checked(19) else {
        return false;
    };
    if !matches_shape(base, "NNNN-NN-NN NN:NN:NN") {
        return false;
    }
    match fraction.strip_prefix('.') {
        None => fraction.is_empty(),
        Some(digits) => {
            !digits.is_empty()
                && digits.bytes().all(|digit| digit.is_ascii_digit())
                && !digits.ends_with('0')
        },
    }
}

/// Whether `text` is a zoned timestamp exactly as [`encode_scalar`] writes one.
///
/// [`is_canonical_timestamp`] plus the `+00` suffix `encode_scalar` appends
/// after normalizing to UTC. The suffix is constant across everything this
/// admits, so it does not disturb the ordering; a value at another offset is
/// refused rather than compared, since `12:00:00+01` sorts above `12:00:00+00`
/// and names an earlier instant.
pub fn is_canonical_timestamptz(text: &str) -> bool {
    text.strip_suffix("+00").is_some_and(is_canonical_timestamp)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// Every float spelling reads back as its value, ties included, and
    /// stays as short as Rust's.
    #[test]
    fn float_spellings_read_back() {
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        for _ in 0..200_000 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let double = f64::from_bits(state);
            if double.is_finite() && double != 0.0 {
                let text = encode_f64(double).unwrap();
                assert_eq!(text.parse::<f64>().unwrap(), double, "{text}");
            }
            let float = f32::from_bits(state as u32);
            if float.is_finite() && float != 0.0 {
                let text = encode_f32(float).unwrap();
                assert_eq!(text.parse::<f32>().unwrap(), float, "{text}");
            }
            // Doubles near 2^50 have quarter units, so halfway ties are common.
            let tie = (state % (1 << 52)) as f64 / 4.0 + 1e15;
            let text = encode_f64(tie).unwrap();
            assert_eq!(text.parse::<f64>().unwrap(), tie, "{text}");
            assert!(text.len() <= format!("{tie:?}").len() + 2, "{text}");
        }
    }

    /// Assert `encode_scalar` matches the golden string DuckDB's
    /// `CAST(<value> AS VARCHAR)` produced (captured from the `duckdb` CLI).
    fn golden(value: ScalarValue, expected: &str) {
        assert_eq!(
            encode_scalar(&value).as_deref(),
            Some(expected),
            "encoding of {value:?}"
        );
    }

    #[test]
    fn integers() {
        golden(ScalarValue::Int32(Some(-2147483648)), "-2147483648");
        golden(
            ScalarValue::Int64(Some(9223372036854775807)),
            "9223372036854775807",
        );
        golden(ScalarValue::Int8(Some(-1)), "-1");
        golden(
            ScalarValue::UInt64(Some(18446744073709551615)),
            "18446744073709551615",
        );
        golden(ScalarValue::UInt8(Some(0)), "0");
    }

    #[test]
    fn booleans() {
        golden(ScalarValue::Boolean(Some(true)), "true");
        golden(ScalarValue::Boolean(Some(false)), "false");
    }

    #[test]
    #[allow(clippy::approx_constant)] // 3.14 is deliberate test data, not π
    fn doubles() {
        // Golden values from `CAST(x::DOUBLE AS VARCHAR)`.
        golden(ScalarValue::Float64(Some(1.0)), "1.0");
        golden(ScalarValue::Float64(Some(3.14)), "3.14");
        golden(ScalarValue::Float64(Some(1e20)), "1e+20");
        golden(ScalarValue::Float64(Some(1e-20)), "1e-20");
        golden(ScalarValue::Float64(Some(-0.5)), "-0.5");
        golden(ScalarValue::Float64(Some(1.0 / 3.0)), "0.3333333333333333");
        golden(ScalarValue::Float64(Some(12345.0)), "12345.0");
        golden(ScalarValue::Float64(Some(0.001)), "0.001");
        golden(ScalarValue::Float64(Some(0.0001)), "0.0001");
        golden(ScalarValue::Float64(Some(0.00001)), "1e-05");
        golden(ScalarValue::Float64(Some(1e14)), "100000000000000.0");
        golden(ScalarValue::Float64(Some(1e15)), "1000000000000000.0");
        golden(ScalarValue::Float64(Some(1e16)), "1e+16");
        golden(
            ScalarValue::Float64(Some(12345678901234567.0)),
            "1.2345678901234568e+16",
        );
        golden(ScalarValue::Float64(Some(-1e20)), "-1e+20");
        golden(ScalarValue::Float64(Some(1e100)), "1e+100");
        golden(ScalarValue::Float64(Some(5e-308)), "5e-308");
        golden(ScalarValue::Float64(Some(0.1)), "0.1");
        golden(ScalarValue::Float64(Some(2.5)), "2.5");
        golden(ScalarValue::Float64(Some(-0.0)), "-0.0");
        golden(ScalarValue::Float64(Some(0.0)), "0.0");
        // Halfway between two shortest spellings: the even last digit, as
        // DuckDB spells it.
        golden(
            ScalarValue::Float64(Some(768_223_024_721_377.0 + 0.25)),
            "768223024721377.2",
        );
        golden(
            ScalarValue::Float64(Some(-(1_973_907_678_291_657.0 + 0.25))),
            "-1973907678291657.2",
        );
        golden(
            ScalarValue::Float64(Some(1_973_907_678_291_657.0 + 0.75)),
            "1973907678291657.8",
        );
        golden(ScalarValue::Float64(Some(f64::INFINITY)), "inf");
        golden(ScalarValue::Float64(Some(f64::NEG_INFINITY)), "-inf");
        assert_eq!(encode_scalar(&ScalarValue::Float64(Some(f64::NAN))), None);
    }

    #[test]
    #[allow(clippy::approx_constant)] // 3.14 is deliberate test data, not π
    fn floats() {
        golden(ScalarValue::Float32(Some(1.0)), "1.0");
        golden(ScalarValue::Float32(Some(3.14)), "3.14");
        golden(ScalarValue::Float32(Some(1e20)), "1e+20");
        golden(ScalarValue::Float32(Some(1.5e-10)), "1.5e-10");
    }

    #[test]
    fn decimals() {
        golden(ScalarValue::Decimal128(Some(12345), 10, 2), "123.45");
        golden(ScalarValue::Decimal128(Some(-5), 10, 2), "-0.05");
        golden(
            ScalarValue::Decimal128(Some(123456789), 18, 6),
            "123.456789",
        );
        golden(ScalarValue::Decimal128(Some(10000), 10, 2), "100.00");
        golden(ScalarValue::Decimal128(Some(42), 5, 0), "42");
        golden(
            ScalarValue::Decimal128(Some(-123456789), 18, 4),
            "-12345.6789",
        );
        golden(ScalarValue::Decimal128(Some(1), 10, 4), "0.0001");
    }

    #[test]
    fn dates() {
        golden(ScalarValue::Date32(Some(18266)), "2020-01-05");
        golden(ScalarValue::Date32(Some(0)), "1970-01-01");
        golden(ScalarValue::Date32(Some(-1)), "1969-12-31");
    }

    #[test]
    fn timestamps() {
        // 2020-01-05 12:34:56.123456 in micros since epoch.
        let micros = 1_578_227_696_123_456;
        golden(
            ScalarValue::TimestampMicrosecond(Some(micros), None),
            "2020-01-05 12:34:56.123456",
        );
        // Whole second — no fractional part.
        golden(
            ScalarValue::TimestampMicrosecond(Some(1_578_227_696_000_000), None),
            "2020-01-05 12:34:56",
        );
        // Millis with a trailing-zero fraction (.120 → .12).
        golden(
            ScalarValue::TimestampMillisecond(Some(1_578_227_696_120), None),
            "2020-01-05 12:34:56.12",
        );
        // Nanosecond precision.
        golden(
            ScalarValue::TimestampNanosecond(Some(1_578_227_696_123_456_789), None),
            "2020-01-05 12:34:56.123456789",
        );
        // Just before the epoch (negative value; div/rem must stay correct).
        golden(
            ScalarValue::TimestampSecond(Some(-1), None),
            "1969-12-31 23:59:59",
        );
        golden(
            ScalarValue::TimestampMicrosecond(Some(-876_544), Some(Arc::from("UTC"))),
            "1969-12-31 23:59:59.123456+00",
        );
        golden(
            ScalarValue::TimestampNanosecond(
                Some(1_578_227_696_123_456_789),
                Some(Arc::from("America/New_York")),
            ),
            "2020-01-05 12:34:56.123456789+00",
        );
    }

    #[test]
    fn strings_verbatim_and_length_guarded() {
        golden(ScalarValue::Utf8(Some("foo".to_string())), "foo");
        golden(ScalarValue::LargeUtf8(Some("".to_string())), "");
        let long = "x".repeat(MAX_STRING_STAT_BYTES + 1);
        assert_eq!(encode_scalar(&ScalarValue::Utf8(Some(long))), None);
        // An embedded NUL is kept, as official's file statistics keep it.
        golden(ScalarValue::Utf8(Some("ab\0cd".to_string())), "ab\0cd");
    }

    #[test]
    fn nulls_and_unsupported_are_none() {
        assert_eq!(encode_scalar(&ScalarValue::Int32(None)), None);
        assert_eq!(encode_scalar(&ScalarValue::Float64(None)), None);
        // Time and binary are deferred ⇒ None.
        assert_eq!(
            encode_scalar(&ScalarValue::Binary(Some(vec![0xAB, 0x01]))),
            None
        );
    }

    #[test]
    fn stat_ordering() {
        use std::cmp::Ordering;
        // Numeric: parsed, not lexical.
        assert_eq!(stat_cmp("9", "10", true), Ordering::Less);
        assert_eq!(stat_cmp("10", "9", true), Ordering::Greater);
        assert_eq!(stat_cmp("-5", "3", true), Ordering::Less);
        assert_eq!(stat_cmp("9.99", "123.45", true), Ordering::Less);
        // Exact fixed-point: high-magnitude decimals f64 cannot distinguish.
        assert_eq!(
            stat_cmp("100000000000000001.00", "100000000000000000.00", true),
            Ordering::Greater
        );
        assert_eq!(stat_cmp("0.05", "0.0001", true), Ordering::Greater);
        assert_eq!(stat_cmp("-12345.6789", "12345.6788", true), Ordering::Less);
        assert_eq!(stat_cmp("100.00", "100.00", true), Ordering::Equal);
        // Scientific/inf still fall back to float ordering.
        assert_eq!(stat_cmp("1e+20", "9.99", true), Ordering::Greater);
        // Lexical for non-numeric (ISO dates/timestamps sort chronologically).
        assert_eq!(stat_cmp("2020-01-05", "2020-02-01", false), Ordering::Less);
        assert_eq!(
            stat_cmp("2020-01-05 12:00:00", "2020-01-05 09:00:00", false),
            Ordering::Greater
        );
        assert_eq!(stat_cmp("apple", "banana", false), Ordering::Less);
    }

    #[test]
    fn numeric_type_classification() {
        for t in [
            "int32",
            "int64",
            "hugeint",
            "ubigint",
            "decimal(10,2)",
            "float",
            "double",
            "float32",
            "float64",
        ] {
            assert!(is_numeric_ducklake_type(t), "{t} should be numeric");
        }
        for t in ["varchar", "date", "timestamp", "timestamptz", "boolean", "uuid", "blob"] {
            assert!(!is_numeric_ducklake_type(t), "{t} should be non-numeric");
        }
    }

    fn fcs(
        column_id: i64,
        min: Option<&str>,
        max: Option<&str>,
        null_count: Option<i64>,
    ) -> FileColumnStat {
        FileColumnStat {
            column_id,
            min_value: min.map(str::to_string),
            max_value: max.map(str::to_string),
            null_count,
            contains_nan: None,
            value_count: None,
        }
    }

    fn fcs_nan(contains_nan: Option<bool>) -> FileColumnStat {
        FileColumnStat {
            contains_nan,
            ..fcs(1, Some("1.0"), Some("2.0"), Some(0))
        }
    }

    #[test]
    fn global_rollup_contains_nan_tristate() {
        // All live files known NaN-free ⇒ definite false (what lets readers
        // trust the float max as an upper bound).
        let out =
            aggregate_global_column_stats(&[fcs_nan(Some(false)), fcs_nan(Some(false))], 2, |_| {
                StatKind::Numeric
            });
        assert_eq!(out[0].contains_nan, Some(false));

        // Full coverage with any NaN ⇒ definite true.
        let out =
            aggregate_global_column_stats(&[fcs_nan(Some(false)), fcs_nan(Some(true))], 2, |_| {
                StatKind::Numeric
            });
        assert_eq!(out[0].contains_nan, Some(true));

        // One file's NaN state unknown ⇒ unknown, even though the other file
        // reported one — same coverage rule as official DuckLake's MergeStats.
        let out = aggregate_global_column_stats(&[fcs_nan(Some(true)), fcs_nan(None)], 2, |_| {
            StatKind::Numeric
        });
        assert_eq!(out[0].contains_nan, None);

        // A statless live file also degrades the flag to unknown.
        let out = aggregate_global_column_stats(&[fcs_nan(Some(false))], 2, |_| StatKind::Numeric);
        assert_eq!(out[0].contains_nan, None);
    }

    #[test]
    fn global_rollup_complete_coverage() {
        // Two live files, both with stats for column 1 (numeric): widen min/max
        // numerically and OR contains_null.
        let per_file =
            vec![fcs(1, Some("9"), Some("9"), Some(0)), fcs(1, Some("10"), Some("10"), Some(2))];
        let out = aggregate_global_column_stats(&per_file, 2, |_| StatKind::Numeric);
        assert_eq!(
            out,
            vec![GlobalColumnStat {
                column_id: 1,
                min_value: Some("9".to_string()),
                max_value: Some("10".to_string()),
                contains_null: Some(true),
                contains_nan: None, // per-file contains_nan always None ⇒ unknown
            }]
        );
    }

    #[test]
    fn global_rollup_incomplete_coverage_degrades_to_null() {
        // M1 regression: 2 live files but only ONE has a stats row (the other is
        // statless — harvest failed / external writer / uncomputed compaction).
        // Every field must degrade to None (SQL NULL), NEVER a definite
        // contains_null=false or an under-covered min/max.
        let per_file = vec![fcs(1, Some("5"), Some("5"), Some(0))];
        let out = aggregate_global_column_stats(&per_file, 2, |_| StatKind::Numeric);
        assert_eq!(
            out,
            vec![GlobalColumnStat {
                column_id: 1,
                min_value: None,
                max_value: None,
                contains_null: None,
                contains_nan: None,
            }]
        );
    }

    #[test]
    fn encodable_type_predicate() {
        assert!(is_encodable_type(&ScalarValue::Int32(None)));
        assert!(is_encodable_type(&ScalarValue::Float64(None)));
        assert!(is_encodable_type(&ScalarValue::TimestampMicrosecond(
            None, None
        )));
        assert!(is_encodable_type(&ScalarValue::TimestampMicrosecond(
            None,
            Some(Arc::from("UTC"))
        )));
        assert!(!is_encodable_type(&ScalarValue::Binary(None)));
    }

    fn stored(
        column_id: i64,
        null: Option<bool>,
        bounds: Option<(&str, &str)>,
    ) -> GlobalColumnStat {
        GlobalColumnStat {
            column_id,
            min_value: bounds.map(|b| b.0.to_string()),
            max_value: bounds.map(|b| b.1.to_string()),
            contains_null: null,
            contains_nan: Some(false),
        }
    }

    fn inlined(values: Vec<Option<i32>>) -> Vec<InlinedColumnStat> {
        let batch = arrow::record_batch::RecordBatch::try_new(
            Arc::new(arrow::datatypes::Schema::new(vec![
                arrow::datatypes::Field::new("a", arrow::datatypes::DataType::Int32, true),
            ])),
            vec![Arc::new(arrow::array::Int32Array::from(values))],
        )
        .unwrap();
        inlined_column_stats(&[batch], &[(1, "int32".to_string())], |_| None).unwrap()
    }

    /// Official's `MergeStats` on an inlined write: bounds widen numerically,
    /// `contains_nan` becomes unknown, an all-NULL write keeps the bounds, and
    /// a column without a stored row gets none once the table has statistics.
    #[test]
    fn inlined_stats_merge_as_official() {
        let merged = merge_inlined_column_stats(
            vec![stored(1, Some(false), Some(("5", "9")))],
            inlined(vec![Some(10), Some(-2)]),
            true,
            |_| StatKind::Numeric,
        );
        assert_eq!(
            merged,
            vec![GlobalColumnStat {
                column_id: 1,
                min_value: Some("-2".to_string()),
                max_value: Some("10".to_string()),
                contains_null: Some(false),
                contains_nan: None,
            }]
        );

        let merged = merge_inlined_column_stats(
            vec![stored(1, Some(false), Some(("5", "9")))],
            inlined(vec![None]),
            true,
            |_| StatKind::Numeric,
        );
        assert_eq!(merged[0].min_value.as_deref(), Some("5"));
        assert_eq!(merged[0].max_value.as_deref(), Some("9"));
        assert_eq!(merged[0].contains_null, Some(true));

        // A stored row with neither bound is official's "no valid value", so
        // it takes the inlined bounds whatever its NULL flag says.
        for null in [Some(true), Some(false), None] {
            let merged = merge_inlined_column_stats(
                vec![stored(1, null, None)],
                inlined(vec![Some(3)]),
                true,
                |_| StatKind::Numeric,
            );
            assert_eq!(merged[0].min_value.as_deref(), Some("3"));
            assert_eq!(merged[0].max_value.as_deref(), Some("3"));
        }
        // One unknown bound beside a known one stays unknown.
        let merged = merge_inlined_column_stats(
            vec![GlobalColumnStat {
                max_value: None,
                ..stored(1, Some(false), Some(("5", "9")))
            }],
            inlined(vec![Some(3)]),
            true,
            |_| StatKind::Numeric,
        );
        assert_eq!(merged[0].min_value.as_deref(), Some("3"));
        assert_eq!(merged[0].max_value, None);

        let merged = merge_inlined_column_stats(
            vec![stored(2, Some(false), Some(("a", "b")))],
            inlined(vec![Some(3)]),
            true,
            |_| StatKind::Numeric,
        );
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].column_id, 2);

        let merged =
            merge_inlined_column_stats(Vec::new(), inlined(vec![Some(3), None]), false, |_| {
                StatKind::Numeric
            });
        assert_eq!(
            merged,
            vec![GlobalColumnStat {
                column_id: 1,
                min_value: Some("3".to_string()),
                max_value: Some("3".to_string()),
                contains_null: Some(true),
                contains_nan: None,
            }]
        );
    }

    /// A file whose column is all NULL contributes no bounds and does not
    /// degrade the others', as official's `MergeStats` skips a source without
    /// a valid value; a file with unknown bounds still degrades them.
    #[test]
    fn global_rollup_skips_all_null_files() {
        let all_null = FileColumnStat {
            value_count: Some(0),
            ..fcs(1, None, None, Some(1))
        };
        let values = FileColumnStat {
            value_count: Some(12),
            ..fcs(1, Some("s05"), Some("s16"), Some(0))
        };
        let out = aggregate_global_column_stats(&[all_null, values.clone()], 2, |_| StatKind::Text);
        assert_eq!(out[0].min_value.as_deref(), Some("s05"));
        assert_eq!(out[0].max_value.as_deref(), Some("s16"));
        assert_eq!(out[0].contains_null, Some(true));

        let unknown = fcs(1, None, None, Some(0));
        let out = aggregate_global_column_stats(&[unknown, values], 2, |_| StatKind::Text);
        assert_eq!(out[0].min_value, None);
    }

    /// Booleans compare by value across both spellings, and NaN sorts above
    /// every number, as official's `RequiresValueComparison` compares them.
    #[test]
    fn value_comparisons() {
        use std::cmp::Ordering;
        assert_eq!(
            stat_cmp_kind("false", "1", StatKind::Boolean),
            Ordering::Less
        );
        assert_eq!(
            stat_cmp_kind("1", "false", StatKind::Boolean),
            Ordering::Greater
        );
        assert_eq!(
            stat_cmp_kind("true", "1", StatKind::Boolean),
            Ordering::Equal
        );
        assert_eq!(
            stat_cmp_kind("nan", "1.5", StatKind::Numeric),
            Ordering::Greater
        );
        assert_eq!(
            stat_cmp_kind("inf", "nan", StatKind::Numeric),
            Ordering::Less
        );
        assert_eq!(StatKind::of_ducklake_type("boolean"), StatKind::Boolean);
        assert_eq!(StatKind::of_ducklake_type("float64"), StatKind::Numeric);
        assert_eq!(StatKind::of_ducklake_type("time"), StatKind::Temporal);
        assert_eq!(StatKind::of_ducklake_type("interval"), StatKind::Text);
        assert_eq!(StatKind::of_ducklake_type("varchar"), StatKind::Text);
    }

    /// Temporal bounds compare as the instant they spell, as official's
    /// `RequiresValueComparison` casts them, not as text.
    #[test]
    fn temporal_value_comparisons() {
        use std::cmp::Ordering;
        let cmp = |a: &str, b: &str| stat_cmp_kind(a, b, StatKind::Temporal);
        // A year before the common era sorts below every year of it.
        assert_eq!(cmp("0044-03-15 (BC)", "0001-01-01"), Ordering::Less);
        assert_eq!(cmp("0002-01-01 (BC)", "0001-01-01 (BC)"), Ordering::Less);
        // Years past 9999 are wider than four digits.
        assert_eq!(cmp("10000-01-01", "9999-12-31"), Ordering::Greater);
        assert_eq!(cmp("infinity", "294247-01-01"), Ordering::Greater);
        assert_eq!(cmp("-infinity", "0001-01-01 (BC)"), Ordering::Less);
        // A shorter fraction is not a smaller instant.
        assert_eq!(
            cmp("2024-01-01 00:00:00.5", "2024-01-01 00:00:00.25"),
            Ordering::Greater
        );
        // Offsets fold into UTC.
        assert_eq!(
            cmp("2024-01-01 12:00:00+01", "2024-01-01 11:30:00+00"),
            Ordering::Less
        );
        assert_eq!(cmp("09:00:00", "10:00:00"), Ordering::Less);
        assert_eq!(
            cmp("2024-01-01 16:30:00+05:30", "2024-01-01 11:00:00+00"),
            Ordering::Equal
        );
        assert_eq!(cmp("23:59:59.999999", "24:00:00"), Ordering::Less);
        // DuckDB writes a timestamp's era between its date and its time.
        assert_eq!(
            cmp("0100-01-01 (BC) 00:00:00", "0044-03-15 (BC) 12:30:00.5"),
            Ordering::Less
        );
        assert_eq!(
            cmp("0044-03-15 (BC) 12:30:00.5", "0044-03-15 (BC) 12:30:00.25"),
            Ordering::Greater
        );
        assert_eq!(
            cmp("0001-12-31 (BC) 23:59:59", "0001-01-01 00:00:00"),
            Ordering::Less
        );
        // Years past chrono's range still order.
        assert_eq!(
            cmp("290309-12-22 (BC)", "262144-01-01 (BC)"),
            Ordering::Less
        );
        assert_eq!(cmp("5877642-06-25", "294247-01-10"), Ordering::Greater);
        assert_eq!(
            cmp("294247-01-10 04:00:54.775806", "262144-01-01 00:00:00"),
            Ordering::Greater
        );
        // A clock field too long for any instant is text, not a panic.
        assert_eq!(
            cmp(
                "2024-01-01 99999999999999999999999:00:00",
                "2024-01-01 00:00:00"
            ),
            "2024-01-01 99999999999999999999999:00:00".cmp("2024-01-01 00:00:00")
        );
        // Text that is no instant compares as text.
        assert_eq!(cmp("abc", "abd"), Ordering::Less);
    }

    /// Official's `MergeStats` for committed files: an all-NULL file leaves
    /// the bounds alone, a row with neither bound takes the next file's, and
    /// once the table has statistics only its stored rows are updated.
    #[test]
    fn file_stats_merge_as_official() {
        let file = |column_id: i64, bounds: Option<(&str, &str)>, values: i64, nulls: i64| {
            FileColumnStat {
                column_id,
                min_value: bounds.map(|b| b.0.to_string()),
                max_value: bounds.map(|b| b.1.to_string()),
                null_count: Some(nulls),
                contains_nan: None,
                value_count: Some(values),
            }
        };
        let numeric = |_| StatKind::Numeric;
        // Inlined 1..3 stored, then a file of 20 NULLs: bounds kept.
        let merged = merge_file_column_stats(
            vec![stored(1, Some(false), Some(("1", "3")))],
            true,
            &[vec![file(1, None, 0, 20)]],
            &[1],
            numeric,
        );
        assert_eq!(merged[0].min_value.as_deref(), Some("1"));
        assert_eq!(merged[0].max_value.as_deref(), Some("3"));
        assert_eq!(merged[0].contains_null, Some(true));
        // A stored row with neither bound takes the file's.
        let merged = merge_file_column_stats(
            vec![stored(1, Some(true), None)],
            true,
            &[vec![file(1, Some(("4", "9")), 2, 0)]],
            &[1],
            numeric,
        );
        assert_eq!(merged[0].min_value.as_deref(), Some("4"));
        assert_eq!(merged[0].max_value.as_deref(), Some("9"));
        // Fields without a stored row get none once the table has statistics.
        let merged = merge_file_column_stats(
            vec![stored(1, Some(false), Some(("1", "3")))],
            true,
            &[vec![file(1, Some(("2", "2")), 1, 0), file(2, Some(("7", "7")), 1, 0)]],
            &[1, 2],
            numeric,
        );
        assert_eq!(merged.len(), 1);
        // Without statistics every field the files cover gets a row.
        let merged = merge_file_column_stats(
            Vec::new(),
            false,
            &[vec![file(1, Some(("2", "2")), 1, 0), file(2, Some(("7", "7")), 1, 0)]],
            &[1, 2],
            numeric,
        );
        assert_eq!(merged.len(), 2);
        // A file without statistics for a live field leaves it with no row.
        let merged = merge_file_column_stats(
            vec![stored(1, Some(false), Some(("1", "3")))],
            true,
            &[vec![file(1, Some(("2", "2")), 1, 0)], Vec::new()],
            &[1],
            numeric,
        );
        assert!(merged.is_empty());
        // A NaN-bearing file's bounds still widen; its flag ORs in.
        let merged = merge_file_column_stats(
            vec![stored(1, Some(false), Some(("1.0", "3.0")))],
            true,
            &[vec![FileColumnStat {
                contains_nan: Some(true),
                ..file(1, Some(("7.0", "7.0")), 2, 0)
            }]],
            &[1],
            numeric,
        );
        assert_eq!(merged[0].max_value.as_deref(), Some("7.0"));
        assert_eq!(merged[0].contains_nan, Some(true));
    }

    /// Golden `CAST(value AS VARCHAR)` text from DuckDB 1.5.5, which official
    /// writes as the bounds of inlined rows.
    #[test]
    fn inline_stat_text_matches_duckdb() {
        let text =
            |value: ScalarValue, ducklake_type: &str| inline_stat_text(&value, ducklake_type);
        assert_eq!(
            text(ScalarValue::Float64(Some(f64::NAN)), "float64").as_deref(),
            Some("nan")
        );
        assert_eq!(
            text(ScalarValue::Float32(Some(f32::NAN)), "float32").as_deref(),
            Some("nan")
        );
        assert_eq!(
            text(ScalarValue::Time64Microsecond(Some(45_296_500_000)), "time").as_deref(),
            Some("12:34:56.5")
        );
        assert_eq!(
            text(ScalarValue::Time64Microsecond(Some(1)), "time").as_deref(),
            Some("00:00:00.000001")
        );
        assert_eq!(
            text(ScalarValue::Time64Microsecond(Some(3_600_000_000)), "time").as_deref(),
            Some("01:00:00")
        );
        assert_eq!(
            text(ScalarValue::Binary(Some(vec![0xAB, 0x00, b'z'])), "blob").as_deref(),
            Some("\\xAB\\x00z")
        );
        assert_eq!(
            text(
                ScalarValue::BinaryView(Some(vec![b'\'', b'"', b'~', 0x7F, b' ', b'A', b'\\'])),
                "blob"
            )
            .as_deref(),
            Some("\\x27\\x22~\\x7F A\\x5C")
        );
        let long = "y".repeat(3000);
        assert_eq!(
            text(ScalarValue::Utf8(Some(long.clone())), "varchar"),
            Some(long)
        );
        assert_eq!(
            text(ScalarValue::Boolean(Some(true)), "boolean").as_deref(),
            Some("true")
        );
        assert_eq!(
            text(ScalarValue::FixedSizeBinary(16, Some(vec![0; 16])), "uuid"),
            None
        );
        assert_eq!(
            encode_file_stat(&ScalarValue::Boolean(Some(true))).as_deref(),
            Some("1")
        );
        assert_eq!(
            encode_file_stat(&ScalarValue::Boolean(Some(false))).as_deref(),
            Some("0")
        );
    }

    /// Nested columns get statistics per leaf field, as official's
    /// `UpdateStats` recurses into `STRUCT`, `LIST` and `MAP`; a NULL struct
    /// makes its fields NULL, a NULL list contributes no element.
    #[test]
    fn inlined_stats_per_nested_leaf() {
        use arrow::array::{Array, ArrayRef, Int32Array, ListArray, StringArray, StructArray};
        use arrow::buffer::{NullBuffer, OffsetBuffer, ScalarBuffer};
        use arrow::datatypes::{DataType, Field, Fields, Schema};
        let struct_fields = Fields::from(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Utf8, true),
        ]);
        let structs = StructArray::new(
            struct_fields.clone(),
            vec![
                Arc::new(Int32Array::from(vec![999, 1, -50])) as ArrayRef,
                Arc::new(StringArray::from(vec![
                    Some("zzz"),
                    Some("b"),
                    Some("ignored"),
                ])),
            ],
            Some(NullBuffer::from(vec![true, true, false])),
        );
        let lists = ListArray::new(
            Arc::new(Field::new("element", DataType::Int32, true)),
            OffsetBuffer::new(ScalarBuffer::from(vec![0_i32, 2, 2, 3])),
            Arc::new(Int32Array::from(vec![Some(3), Some(1), Some(100)])),
            Some(NullBuffer::from(vec![true, true, false])),
        );
        let entry_fields = Fields::from(vec![
            Field::new("key", DataType::Utf8, false),
            Field::new("value", DataType::Int32, true),
        ]);
        let maps = arrow::array::MapArray::new(
            Arc::new(Field::new(
                "entries",
                DataType::Struct(entry_fields.clone()),
                false,
            )),
            OffsetBuffer::new(ScalarBuffer::from(vec![0_i32, 1, 2, 2])),
            StructArray::new(
                entry_fields,
                vec![
                    Arc::new(StringArray::from(vec!["k", "zz"])) as ArrayRef,
                    Arc::new(Int32Array::from(vec![Some(5), None])),
                ],
                None,
            ),
            None,
            false,
        );
        let batch = arrow::record_batch::RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("s", DataType::Struct(struct_fields), true),
                Field::new("l", lists.data_type().clone(), true),
                Field::new("m", maps.data_type().clone(), true),
            ])),
            vec![Arc::new(structs), Arc::new(lists), Arc::new(maps)],
        )
        .unwrap();
        let fields = [
            (10, "struct".to_string()),
            (11, "int32".to_string()),
            (12, "varchar".to_string()),
            (13, "list".to_string()),
            (14, "int32".to_string()),
            (15, "map".to_string()),
            (16, "varchar".to_string()),
            (17, "int32".to_string()),
        ];
        let stats = inlined_column_stats(&[batch], &fields, |_| None).unwrap();
        let by_id = |id| stats.iter().find(|s| s.column_id == id).unwrap();
        assert_eq!(stats.len(), 5);
        assert_eq!(by_id(16).min_value.as_deref(), Some("k"));
        assert_eq!(by_id(16).max_value.as_deref(), Some("zz"));
        assert_eq!(by_id(17).max_value.as_deref(), Some("5"));
        assert!(by_id(17).contains_null);
        assert_eq!(by_id(11).min_value.as_deref(), Some("1"));
        assert_eq!(by_id(11).max_value.as_deref(), Some("999"));
        assert!(by_id(11).contains_null);
        assert_eq!(by_id(12).min_value.as_deref(), Some("b"));
        assert_eq!(by_id(12).max_value.as_deref(), Some("zzz"));
        assert_eq!(by_id(14).min_value.as_deref(), Some("1"));
        assert_eq!(by_id(14).max_value.as_deref(), Some("3"));
        assert!(!by_id(14).contains_null);
    }
}

#[cfg(test)]
mod merge_regression_tests {
    use super::*;

    /// A commit whose first file holds only NULLs of a column keeps the
    /// bounds the later files bring, as upstream main's `MergeStats` does.
    #[test]
    fn a_null_file_before_values_keeps_their_bounds() {
        let file = |min: Option<&str>, nulls: i64, values: i64| {
            vec![FileColumnStat {
                column_id: 2,
                min_value: min.map(str::to_string),
                max_value: min.map(str::to_string),
                null_count: Some(nulls),
                contains_nan: None,
                value_count: Some(values),
            }]
        };
        let merged = merge_file_column_stats(
            Vec::new(),
            false,
            &[file(None, 1, 0), file(Some("4"), 0, 1)],
            &[2],
            |_| StatKind::of_ducklake_type("int64"),
        );
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].min_value.as_deref(), Some("4"));
        assert_eq!(merged[0].max_value.as_deref(), Some("4"));
        assert_eq!(merged[0].contains_null, Some(true));
    }

    /// A NaN-only file merged beside statistics without a NaN flag leaves no
    /// row with neither bound, which official would read as holding no value.
    #[test]
    fn a_nan_only_write_beside_unflagged_stats_leaves_no_row() {
        let nan_only = ColumnStats {
            min_value: None,
            max_value: None,
            null_count: Some(0),
            num_values: Some(1),
            contains_nan: Some(true),
            any_valid: true,
        };
        let inlined = ColumnStats {
            min_value: None,
            max_value: None,
            null_count: Some(1),
            num_values: None,
            contains_nan: None,
            any_valid: false,
        };
        let merged =
            merge_table_column_stats(Vec::new(), false, [(4, nan_only), (4, inlined)], |_| {
                StatKind::of_ducklake_type("float")
            });
        assert!(merged.is_empty(), "{merged:?}");
    }

    /// A file without float bounds (NaN only, or bounds its writer omitted
    /// for a NaN) beside files with values leaves the column no row, in
    /// either order: its values may lie outside the others' bounds.
    #[test]
    fn a_boundless_nan_file_leaves_no_row() {
        let file = |bounds: Option<(&str, &str)>, nan: bool| {
            vec![FileColumnStat {
                column_id: 4,
                min_value: bounds.map(|b| b.0.to_string()),
                max_value: bounds.map(|b| b.1.to_string()),
                null_count: Some(0),
                contains_nan: Some(nan),
                value_count: Some(1),
            }]
        };
        let float = |_| StatKind::of_ducklake_type("float");
        for files in [
            [file(Some(("1.0", "2.0")), false), file(None, true)],
            [file(None, true), file(Some(("1.0", "2.0")), false)],
        ] {
            let merged = merge_file_column_stats(Vec::new(), false, &files, &[4], float);
            assert!(merged.is_empty(), "{merged:?}");
        }
    }

    /// A file's bound holding a NUL byte that another file's bound
    /// supersedes leaves the merged bounds stored, as official stores them; a
    /// merged bound holding one leaves the field without a row.
    #[test]
    fn a_superseded_nul_bound_keeps_the_row() {
        let file = |min: &str, max: &str| {
            vec![FileColumnStat {
                column_id: 3,
                min_value: Some(min.to_string()),
                max_value: Some(max.to_string()),
                null_count: Some(0),
                contains_nan: None,
                value_count: Some(2),
            }]
        };
        let text = |_| StatKind::of_ducklake_type("varchar");
        let merged = merge_file_column_stats(
            Vec::new(),
            false,
            &[file("%_", "x"), file("a\0b", "c")],
            &[3],
            text,
        );
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].min_value.as_deref(), Some("%_"));
        assert_eq!(merged[0].max_value.as_deref(), Some("x"));
        let merged = merge_file_column_stats(
            Vec::new(),
            false,
            &[file("b", "x"), file("a\0b", "c")],
            &[3],
            text,
        );
        assert!(merged.is_empty(), "{merged:?}");
    }
}
