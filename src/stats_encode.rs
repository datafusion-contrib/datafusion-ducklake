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
//! `FLOAT`/`DOUBLE`, `DECIMAL(p,s)` (128-bit), `DATE`, and `TIMESTAMP` without a
//! time zone (all Arrow time units).
//!
//! Deliberately **not** encoded yet — [`encode_scalar`] returns `None`, so the
//! caller stores `min_value`/`max_value` as `NULL` and the file is never pruned on
//! that column (spec-safe: DuckLake keeps files whose stats are NULL). Deferred:
//! - `TIMESTAMP WITH TIME ZONE` — DuckDB renders it in the *session* time zone, so
//!   the stored string is not deterministic without pinning a zone; deferred until
//!   we settle on UTC rendering. See `[[timestamptz-stats]]`.
//! - `TIME`, `BLOB`/`BINARY` (DuckLake never prunes on blobs anyway), `UUID`,
//!   `INTERVAL`, `DECIMAL256`, and all nested types (`STRUCT`/`LIST`/`MAP`).
//! - Over-long strings (see [`MAX_STRING_STAT_BYTES`]) — DuckDB truncates and
//!   rounds min down / max up; until that is mirrored, long values store `NULL`.

use datafusion::common::ScalarValue;

/// Strings longer than this (in UTF-8 bytes) are not emitted as min/max stats.
///
/// DuckDB/Parquet truncate long string stats and adjust the bound (min rounds
/// down, max rounds up) to stay sound. Reproducing that faithfully is deferred;
/// until then an over-long value stores `NULL` (⇒ the file is kept), which is
/// always correct.
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

        // TIMESTAMP without a time zone. WITH time zone (`tz.is_some()`) is
        // deferred — see `[[timestamptz-stats]]` in the module docs.
        ScalarValue::TimestampSecond(Some(v), None) => encode_timestamp(*v, 1),
        ScalarValue::TimestampMillisecond(Some(v), None) => encode_timestamp(*v, 1_000),
        ScalarValue::TimestampMicrosecond(Some(v), None) => encode_timestamp(*v, 1_000_000),
        ScalarValue::TimestampNanosecond(Some(v), None) => encode_timestamp(*v, 1_000_000_000),

        // Strings — stored verbatim, subject to the length guard and the NUL
        // rule. A `\0` byte is dropped (→ SQL NULL) exactly as official DuckLake's
        // `DuckLakeUtil::StatsToString` does: Postgres text rejects `0x00` (which
        // would otherwise abort the whole commit), and a C-string consumer would
        // truncate at it. Dropping the bound just leaves the column unpruned.
        ScalarValue::Utf8(Some(s))
        | ScalarValue::LargeUtf8(Some(s))
        | ScalarValue::Utf8View(Some(s)) => {
            if s.len() <= MAX_STRING_STAT_BYTES && !s.contains('\0') {
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
        ScalarValue::TimestampSecond(_, None)
            | ScalarValue::TimestampMillisecond(_, None)
            | ScalarValue::TimestampMicrosecond(_, None)
            | ScalarValue::TimestampNanosecond(_, None)
    )
}

/// Order two already-encoded stat strings, for widening the global
/// `ducklake_table_column_stats` min/max across files.
///
/// DuckDB-canonical encodings of dates, timestamps, booleans and strings are
/// already lexically ordered (ISO-8601 / natural), so only numeric types need a
/// parsed comparison — `"10"` must sort after `"9"`, `"-5"` before `"3"`. Pass
/// `numeric = true` for integer/float/decimal columns (see
/// [`is_numeric_ducklake_type`]).
pub fn stat_cmp(a: &str, b: &str, numeric: bool) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    if numeric {
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
        // Scientific / inf / other numeric forms: fall back to float ordering.
        if let (Ok(x), Ok(y)) = (a.parse::<f64>(), b.parse::<f64>()) {
            return x.partial_cmp(&y).unwrap_or(Ordering::Equal);
        }
    }
    a.cmp(b)
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
        || matches!(t.as_str(), "float" | "double" | "real")
}

/// One `ducklake_file_column_stats` row's fields relevant to the global roll-up.
#[derive(Debug, Clone)]
pub struct FileColumnStat {
    pub column_id: i64,
    pub min_value: Option<String>,
    pub max_value: Option<String>,
    pub null_count: Option<i64>,
    pub contains_nan: Option<bool>,
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
/// `live_file_count` is the total number of live files. `numeric(column_id)`
/// says whether a column needs numeric (vs lexical) bound comparison.
///
/// The completeness rule mirrors official DuckLake (`has_min`/`has_null_count`):
/// a global field is emitted only when **every** live file contributed it —
/// otherwise it degrades to `None` (SQL `NULL` = unknown). This is essential:
/// a live file with no stats row (harvest failed, written by another tool, or a
/// compacted file we didn't stat) must NOT let the roll-up claim a definite
/// `contains_null = false` or an under-covered min/max, which a reader could use
/// to wrongly drop rows. min/max additionally widen (never tighten); bounds are
/// widened with a type-aware compare ([`stat_cmp`]).
///
/// Output is sorted by `column_id` for determinism.
pub fn aggregate_global_column_stats(
    per_file: &[FileColumnStat],
    live_file_count: i64,
    numeric: impl Fn(i64) -> bool,
) -> Vec<GlobalColumnStat> {
    use std::cmp::Ordering;
    use std::collections::HashMap;

    struct Agg {
        min: Option<String>,
        max: Option<String>,
        min_present: i64,
        max_present: i64,
        nullcount_present: i64,
        has_null: bool,
        has_nan: bool,
        numeric: bool,
    }

    let mut aggs: HashMap<i64, Agg> = HashMap::new();
    for f in per_file {
        let agg = aggs.entry(f.column_id).or_insert_with(|| Agg {
            min: None,
            max: None,
            min_present: 0,
            max_present: 0,
            nullcount_present: 0,
            has_null: false,
            has_nan: false,
            numeric: numeric(f.column_id),
        });
        if let Some(mn) = &f.min_value {
            agg.min_present += 1;
            agg.min = Some(match agg.min.take() {
                Some(cur) if stat_cmp(&cur, mn, agg.numeric) != Ordering::Greater => cur,
                _ => mn.clone(),
            });
        }
        if let Some(mx) = &f.max_value {
            agg.max_present += 1;
            agg.max = Some(match agg.max.take() {
                Some(cur) if stat_cmp(&cur, mx, agg.numeric) != Ordering::Less => cur,
                _ => mx.clone(),
            });
        }
        if let Some(nc) = f.null_count {
            agg.nullcount_present += 1;
            if nc > 0 {
                agg.has_null = true;
            }
        }
        if f.contains_nan == Some(true) {
            agg.has_nan = true;
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
            // Surface a table-level NaN flag only when NaN is actually present
            // (Some(true)); otherwise leave it NULL/unknown — matching official
            // DuckLake, which never records a definite table-level `false`.
            contains_nan: a.has_nan.then_some(true),
        })
        .collect();
    out.sort_by_key(|g| g.column_id);
    out
}

// --------------------------------------------------------------------------
// Floating point — DuckDB's shortest-round-trip rendering (matches Python
// `repr`): fixed notation when the decimal exponent of the leading digit is in
// [-4, 16), else scientific with a signed, ≥2-digit zero-padded exponent. Whole
// values still carry a `.0`. NaN ⇒ None (DuckLake omits min/max when NaN is
// present); ±inf ⇒ "inf"/"-inf"; ±0.0 ⇒ "0.0".
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
        return Some("0.0".to_string());
    }
    // Rust's `{:e}` yields the shortest round-trip digits in normalized
    // `d[.ddd]e<exp>` form (no sign/pad on the exponent); reformat to DuckDB's.
    Some(format_shortest(
        &format!("{:e}", v.abs()),
        v.is_sign_negative(),
    ))
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
        return Some("0.0".to_string());
    }
    // Format from the f32 (not widened to f64) so the shortest digits are the
    // f32's own, matching DuckDB's FLOAT rendering.
    Some(format_shortest(
        &format!("{:e}", v.abs()),
        v.is_sign_negative(),
    ))
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
// DATE / TIMESTAMP (no time zone) via chrono.
// --------------------------------------------------------------------------

fn epoch_date() -> chrono::NaiveDate {
    // 1970-01-01 is always valid; unwrap is infallible.
    chrono::NaiveDate::from_ymd_opt(1970, 1, 1).expect("epoch date is valid")
}

fn encode_date32(days: i32) -> Option<String> {
    let date = epoch_date().checked_add_signed(chrono::Duration::days(i64::from(days)))?;
    Some(date.format("%Y-%m-%d").to_string())
}

/// Encode a naive timestamp given its value in `units_per_second` sub-second
/// units (1 = seconds, 1_000 = millis, 1_000_000 = micros, 1e9 = nanos).
fn encode_timestamp(value: i64, units_per_second: i64) -> Option<String> {
    let secs = value.div_euclid(units_per_second);
    let sub = value.rem_euclid(units_per_second); // 0..units_per_second
    // Convert the sub-second remainder to nanoseconds for chrono.
    let nanos = (sub as i128 * 1_000_000_000 / units_per_second as i128) as u32;
    let dt = chrono::DateTime::from_timestamp(secs, nanos)?.naive_utc();
    let base = dt.format("%Y-%m-%d %H:%M:%S").to_string();

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
    if frac.is_empty() {
        Some(base)
    } else {
        Some(format!("{base}.{frac}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

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
        golden(ScalarValue::Float64(Some(-0.0)), "0.0");
        golden(ScalarValue::Float64(Some(0.0)), "0.0");
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
    }

    #[test]
    fn strings_verbatim_and_length_guarded() {
        golden(ScalarValue::Utf8(Some("foo".to_string())), "foo");
        golden(ScalarValue::LargeUtf8(Some("".to_string())), "");
        let long = "x".repeat(MAX_STRING_STAT_BYTES + 1);
        assert_eq!(encode_scalar(&ScalarValue::Utf8(Some(long))), None);
        // Embedded NUL ⇒ dropped (→ SQL NULL), mirroring official DuckLake.
        assert_eq!(
            encode_scalar(&ScalarValue::Utf8(Some("ab\0cd".to_string()))),
            None
        );
    }

    #[test]
    fn nulls_and_unsupported_are_none() {
        assert_eq!(encode_scalar(&ScalarValue::Int32(None)), None);
        assert_eq!(encode_scalar(&ScalarValue::Float64(None)), None);
        // Time / tz-timestamp / binary are deferred ⇒ None.
        assert_eq!(
            encode_scalar(&ScalarValue::TimestampMicrosecond(
                Some(0),
                Some(Arc::from("UTC"))
            )),
            None
        );
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
        for t in ["int32", "int64", "hugeint", "ubigint", "decimal(10,2)", "float", "double"] {
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
        }
    }

    #[test]
    fn global_rollup_complete_coverage() {
        // Two live files, both with stats for column 1 (numeric): widen min/max
        // numerically and OR contains_null.
        let per_file =
            vec![fcs(1, Some("9"), Some("9"), Some(0)), fcs(1, Some("10"), Some("10"), Some(2))];
        let out = aggregate_global_column_stats(&per_file, 2, |_| true);
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
        let out = aggregate_global_column_stats(&per_file, 2, |_| true);
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
        assert!(!is_encodable_type(&ScalarValue::TimestampMicrosecond(
            None,
            Some(Arc::from("UTC"))
        )));
        assert!(!is_encodable_type(&ScalarValue::Binary(None)));
    }
}
