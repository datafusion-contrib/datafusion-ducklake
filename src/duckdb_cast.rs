//! DuckDB's `VARCHAR` casts and option parsers, as official DuckLake's
//! `ducklake_set_option` applies them to the value of a stored setting.
//!
//! [`varchar_to_ubigint`] ports DuckDB's `TrySimpleIntegerCast`
//! (`integer_cast_operator.hpp`): leading and trailing whitespace, a sign,
//! `0x`/`0b` prefixes, single `_` separators between digits, a fraction
//! rounded half away from zero and an exponent are accepted, so `'1.5'` is
//! `2`, `'1e2'` is `100`, `'0x10'` is `16` and `'1_000'` is `1000`. A negative
//! sign is accepted only before zeros (`'-0'`). [`varchar_to_boolean`] ports
//! `TryCastStringBool`, [`varchar_to_double`] `TryDoubleCast` and
//! [`parse_memory_limit`] `DBConfig::ParseMemoryLimit`.

/// The range of an integer type, and DuckDB's digit count for it
/// (`NumericLimits<T>::Digits`).
#[derive(Debug, Clone, Copy)]
struct IntType {
    min: i128,
    max: i128,
    digits: u32,
}

/// `UBIGINT`, the type a value is cast to and accumulated in.
const UINT64: IntType = IntType {
    min: 0,
    max: u64::MAX as i128,
    digits: 20,
};
/// `TINYINT`, the type `hive_file_pattern` and `require_commit_message` are
/// cast to.
const INT8: IntType = IntType {
    min: i8::MIN as i128,
    max: i8::MAX as i128,
    digits: 3,
};
/// `BIGINT`, the type a cast to a smaller signed integer type accumulates a
/// fraction or an exponent in (`IntegerDecimalCastData<T>::StoreType`).
const INT64: IntType = IntType {
    min: i64::MIN as i128,
    max: i64::MAX as i128,
    digits: 19,
};
/// `SMALLINT`, the type an exponent is cast to.
const INT16: IntType = IntType {
    min: i16::MIN as i128,
    max: i16::MAX as i128,
    digits: 5,
};

impl IntType {
    fn signed(self) -> bool {
        self.min < 0
    }

    fn fits(self, value: i128) -> bool {
        (self.min..=self.max).contains(&value)
    }
}

fn is_space(byte: u8) -> bool {
    matches!(byte, b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r')
}

/// The state of one cast: the simple one (`IntegerCastData`) uses `result`
/// alone, the one with fractions and exponents
/// (`IntegerDecimalCastData`) all three.
#[derive(Default)]
struct State {
    result: i128,
    decimal: i128,
    decimal_digits: u16,
}

/// Which DuckDB operation drives the cast loop.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Op {
    /// `IntegerCastOperation`: digits only.
    Simple,
    /// `IntegerDecimalCastOperation`: digits, a fraction and an exponent.
    Decimal,
}

struct Cast {
    op: Op,
    /// The type values accumulate in.
    store: IntType,
    /// The type the result must fit, the store's but for a cast with a
    /// fraction or an exponent to an integer type narrower than `BIGINT`.
    result: IntType,
}

impl Cast {
    fn handle_digit(&self, state: &mut State, negative: bool, digit: u8) -> bool {
        let digit = i128::from(digit);
        if negative {
            if state.result < (self.store.min + digit) / 10 {
                return false;
            }
            state.result = state.result * 10 - digit;
        } else {
            if state.result > (self.store.max - digit) / 10 {
                return false;
            }
            state.result = state.result * 10 + digit;
        }
        true
    }

    fn handle_radix_digit(&self, state: &mut State, radix: i128, digit: u8) -> bool {
        let digit = i128::from(digit);
        if state.result > (self.store.max - digit) / radix {
            return false;
        }
        state.result = state.result * radix + digit;
        true
    }

    fn handle_decimal(&self, state: &mut State, digit: u8) -> bool {
        if self.op == Op::Simple {
            return false;
        }
        let digit = i128::from(digit);
        if state.decimal > (self.store.max - digit) / 10 {
            // DuckDB ignores digits past the store type's precision.
            return true;
        }
        state.decimal_digits = state.decimal_digits.wrapping_add(1);
        state.decimal = state.decimal * 10 + digit;
        true
    }

    fn checked(&self, value: i128) -> Option<i128> {
        self.store.fits(value).then_some(value)
    }

    /// `10^n` computed in the store type, wrapping as DuckDB's does.
    fn power_of_ten(&self, n: u32) -> i128 {
        if self.store.signed() {
            i128::from((0..n).fold(1i64, |power, _| power.wrapping_mul(10)))
        } else {
            i128::from((0..n).fold(1u64, |power, _| power.wrapping_mul(10)))
        }
    }

    fn handle_exponent(&self, state: &mut State, negative: bool, exponent: i16) -> bool {
        if self.op == Op::Simple {
            return false;
        }
        let mut e = exponent;
        if e < 0 {
            while e < 0 {
                e += 1;
                state.decimal = state.result % 10;
                state.result /= 10;
                if state.result == 0 && state.decimal == 0 {
                    break;
                }
            }
            if state.decimal < 0 {
                state.decimal = -state.decimal;
            }
            state.decimal_digits = 1;
            return self.finalize(state, negative);
        }
        while state.result != 0 && e > 0 {
            e -= 1;
            match self.checked(state.result * 10) {
                Some(value) => state.result = value,
                None => return false,
            }
        }
        if state.decimal == 0 {
            return self.finalize(state, negative);
        }
        let mut e = (i32::from(exponent) - i32::from(state.decimal_digits)) as i16;
        let mut remainder = 0;
        if e < 0 {
            if u32::from(e.unsigned_abs()) <= self.store.digits {
                let power = self.power_of_ten(u32::from(e.unsigned_abs()));
                remainder = state.decimal % power;
                state.decimal /= power;
            } else {
                state.decimal = 0;
            }
        } else {
            while e > 0 {
                e -= 1;
                match self.checked(state.decimal * 10) {
                    Some(value) => state.decimal = value,
                    None => return false,
                }
            }
        }
        state.decimal_digits = state.decimal_digits.wrapping_sub(exponent as u16);
        let combined = if negative {
            state.result - state.decimal
        } else {
            state.result + state.decimal
        };
        match self.checked(combined) {
            Some(value) => state.result = value,
            None => return false,
        }
        state.decimal = remainder;
        self.finalize(state, negative)
    }

    fn finalize(&self, state: &mut State, negative: bool) -> bool {
        if self.op == Op::Simple {
            return true;
        }
        if !self.result.fits(state.result) {
            return false;
        }
        while state.decimal > 10 {
            state.decimal /= 10;
            state.decimal_digits = state.decimal_digits.wrapping_sub(1);
        }
        if state.decimal_digits == 1 && state.decimal >= 5 {
            let rounded = if negative {
                state.result - 1
            } else {
                state.result + 1
            };
            if !self.result.fits(rounded) {
                return false;
            }
            state.result = rounded;
        }
        true
    }

    /// `IntegerCastLoop`: `buf` starts with the sign when `negative`.
    fn cast_loop(
        &self,
        buf: &[u8],
        state: &mut State,
        negative: bool,
        allow_exponent: bool,
    ) -> bool {
        let len = buf.len();
        let start_pos = if negative || buf.first() == Some(&b'+') {
            1
        } else {
            0
        };
        let mut pos = start_pos;
        while pos < len {
            if !buf[pos].is_ascii_digit() {
                if buf[pos] == b'.' {
                    let number_before_period = pos > start_pos;
                    pos += 1;
                    let start_digit = pos;
                    while pos < len {
                        if !buf[pos].is_ascii_digit() {
                            break;
                        }
                        if !self.handle_decimal(state, buf[pos] - b'0') {
                            return false;
                        }
                        pos += 1;
                        if pos != len && buf[pos] == b'_' {
                            pos += 1;
                            if pos == len || !buf[pos].is_ascii_digit() {
                                return false;
                            }
                        }
                    }
                    if !(number_before_period || pos > start_digit) {
                        return false;
                    }
                    if pos >= len {
                        break;
                    }
                }
                if is_space(buf[pos]) {
                    pos += 1;
                    while pos < len {
                        if !is_space(buf[pos]) {
                            return false;
                        }
                        pos += 1;
                    }
                    break;
                }
                if allow_exponent && (buf[pos] == b'e' || buf[pos] == b'E') {
                    if pos == start_pos {
                        return false;
                    }
                    pos += 1;
                    if pos >= len {
                        return false;
                    }
                    let exponent_cast = Cast {
                        op: Op::Simple,
                        store: INT16,
                        result: INT16,
                    };
                    let mut exponent = State::default();
                    let negative_exponent = buf[pos] == b'-';
                    if !exponent_cast.cast_loop(
                        &buf[pos..],
                        &mut exponent,
                        negative_exponent,
                        false,
                    ) {
                        return false;
                    }
                    return self.handle_exponent(state, negative, exponent.result as i16);
                }
                return false;
            }
            let digit = buf[pos] - b'0';
            pos += 1;
            if !self.handle_digit(state, negative, digit) {
                return false;
            }
            if pos != len && buf[pos] == b'_' {
                pos += 1;
                if pos == len || !buf[pos].is_ascii_digit() {
                    return false;
                }
            }
        }
        if !self.finalize(state, negative) {
            return false;
        }
        pos > start_pos
    }

    /// `IntegerHexCastLoop` and `IntegerBinaryCastLoop`: `buf` starts with
    /// the `x` or `b` of the prefix.
    fn radix_loop(&self, buf: &[u8], state: &mut State, radix: u32) -> bool {
        let len = buf.len();
        let mut pos = 1;
        let is_digit = |byte: u8| char::from(byte).is_digit(radix);
        while pos < len {
            let Some(digit) = char::from(buf[pos]).to_digit(radix) else {
                return false;
            };
            pos += 1;
            if pos != len && buf[pos] == b'_' {
                pos += 1;
                if pos == len || !is_digit(buf[pos]) {
                    return false;
                }
            }
            if !self.handle_radix_digit(state, i128::from(radix), digit as u8) {
                return false;
            }
        }
        if !self.finalize(state, false) {
            return false;
        }
        pos > 1
    }

    /// `TryIntegerCast`.
    fn try_cast(&self, input: &str, allow_exponent: bool) -> Option<i128> {
        let mut buf = input.as_bytes();
        while let [first, rest @ ..] = buf
            && is_space(*first)
        {
            buf = rest;
        }
        if buf.is_empty() {
            return None;
        }
        let mut state = State::default();
        let parsed = if buf[0] == b'-' {
            if !self.store.signed() && buf[1..].iter().any(|byte| *byte != b'0') {
                return None;
            }
            self.cast_loop(buf, &mut state, true, allow_exponent)
        } else if buf.len() > 1 && buf[0] == b'0' && matches!(buf[1], b'x' | b'X' | b'b' | b'B') {
            let radix = if matches!(buf[1], b'x' | b'X') {
                16
            } else {
                2
            };
            self.radix_loop(&buf[1..], &mut state, radix)
        } else {
            self.cast_loop(buf, &mut state, false, allow_exponent)
        };
        parsed.then_some(state.result)
    }
}

/// `input` cast to `UBIGINT` as DuckDB casts a `VARCHAR`, or `None` where
/// DuckDB's cast fails: first as a plain integer, then with a fraction or an
/// exponent, accumulated in `UBIGINT` (`IntegerDecimalCastData<uint64_t>`).
pub(crate) fn varchar_to_ubigint(input: &str) -> Option<u64> {
    Cast {
        op: Op::Simple,
        store: UINT64,
        result: UINT64,
    }
    .try_cast(input, false)
    .or_else(|| {
        Cast {
            op: Op::Decimal,
            store: UINT64,
            result: UINT64,
        }
        .try_cast(input, true)
    })
    .map(|value| value as u64)
}

/// `input` cast to `TINYINT` as DuckDB casts a `VARCHAR`, or `None` where
/// DuckDB's cast fails: first as a plain integer, then with a fraction or an
/// exponent accumulated in `BIGINT` (`IntegerDecimalCastData<int8_t>`), the
/// result rounded half away from zero and checked against `TINYINT`'s range
/// (`'127.4'` is `127`, `'127.5'` and `'128'` fail).
pub(crate) fn varchar_to_tinyint(input: &str) -> Option<i8> {
    Cast {
        op: Op::Simple,
        store: INT8,
        result: INT8,
    }
    .try_cast(input, false)
    .or_else(|| {
        Cast {
            op: Op::Decimal,
            store: INT64,
            result: INT8,
        }
        .try_cast(input, true)
    })
    .map(|value| value as i8)
}

/// `input` cast to `BOOLEAN` as DuckDB casts a `VARCHAR`
/// (`TryCastStringBool`, non-strict): `t`, `true`, `y`, `yes` and `1` are
/// `true`, `f`, `false`, `n`, `no` and `0` are `false`, in any case and with no
/// surrounding whitespace; anything else fails.
pub(crate) fn varchar_to_boolean(input: &str) -> Option<bool> {
    match input.to_ascii_lowercase().as_str() {
        "t" | "y" | "1" | "yes" | "true" => Some(true),
        "f" | "n" | "0" | "no" | "false" => Some(false),
        _ => None,
    }
}

/// `input` cast to `DOUBLE` as DuckDB casts a `VARCHAR` (`TryDoubleCast`,
/// non-strict, over `duckdb_fast_float`): surrounding whitespace, a leading
/// `+` or `-`, single `_` separators between digits, a missing integer or
/// fraction part (`'.5'`, `'5.'`), an exponent, and `inf`, `infinity` and
/// `nan` in any case are accepted.
pub(crate) fn varchar_to_double(input: &str) -> Option<f64> {
    let bytes = input.as_bytes();
    let mut start = 0;
    while start < bytes.len() && is_space(bytes[start]) {
        start += 1;
    }
    let mut end = bytes.len();
    while end > start && is_space(bytes[end - 1]) {
        end -= 1;
    }
    let mut buf = &bytes[start..end];
    if buf.first() == Some(&b'+') {
        buf = &buf[1..];
    }
    let (negative, body) = match buf.first() {
        Some(b'-') => (true, &buf[1..]),
        _ => (false, buf),
    };
    let special = std::str::from_utf8(body).ok()?.to_ascii_lowercase();
    let value = match special.as_str() {
        "inf" | "infinity" => Some(f64::INFINITY),
        "nan" => Some(f64::NAN),
        _ => None,
    };
    if let Some(value) = value {
        return Some(if negative {
            -value
        } else {
            value
        });
    }
    // Digits with single `_` separators between them, as `duckdb_fast_float`
    // reads each part.
    let mut digits = String::with_capacity(body.len() + 1);
    if negative {
        digits.push('-');
    }
    let mut pos = 0;
    let read_digits = |pos: &mut usize, digits: &mut String| -> Option<usize> {
        let begin = *pos;
        while *pos < body.len() && body[*pos].is_ascii_digit() {
            digits.push(char::from(body[*pos]));
            *pos += 1;
            if *pos < body.len() && body[*pos] == b'_' {
                *pos += 1;
                if *pos == body.len() || !body[*pos].is_ascii_digit() {
                    return None;
                }
            }
        }
        Some(*pos - begin)
    };
    let mut digit_count = read_digits(&mut pos, &mut digits)?;
    if pos < body.len() && body[pos] == b'.' {
        digits.push('.');
        pos += 1;
        digit_count += read_digits(&mut pos, &mut digits)?;
    }
    if digit_count == 0 {
        return None;
    }
    if pos < body.len() && matches!(body[pos], b'e' | b'E') {
        let mut exponent = String::from("e");
        let mut at = pos + 1;
        if at < body.len() && matches!(body[at], b'-' | b'+') {
            exponent.push(char::from(body[at]));
            at += 1;
        }
        if at < body.len() && body[at].is_ascii_digit() {
            read_digits(&mut at, &mut exponent)?;
            digits.push_str(&exponent);
            pos = at;
        }
    }
    if pos != body.len() {
        return None;
    }
    digits.parse::<f64>().ok()
}

/// A byte count as DuckDB's `DBConfig::ParseMemoryLimit` parses it, the
/// parser official's `ducklake_set_option` applies to `target_file_size`
/// and `parquet_row_group_size_bytes`: a number (read as a `DOUBLE`) and a
/// unit (`b`, `kb`/`k`, `mb`/`m`, `gb`/`g`, `tb`/`t` for powers of 1000,
/// `kib` to `tib` for powers of 1024, or their spelled-out names, in any case);
/// a leading `-`, `null` or `none` is the largest `UBIGINT`. The error is
/// DuckDB's message.
pub(crate) fn parse_memory_limit(input: &str) -> std::result::Result<u64, String> {
    if input.starts_with('-') || input == "null" || input == "none" {
        return Ok(u64::MAX);
    }
    let bytes = input.as_bytes();
    let mut idx = 0;
    while idx < bytes.len() && is_space(bytes[idx]) {
        idx += 1;
    }
    let num_start = idx;
    while idx < bytes.len()
        && (bytes[idx].is_ascii_digit() || matches!(bytes[idx], b'.' | b'e' | b'E' | b'-'))
    {
        idx += 1;
    }
    if idx == num_start {
        return Err("Memory must have a number (e.g. 1GB)".to_string());
    }
    let number = &input[num_start..idx];
    let limit =
        varchar_to_double(number).ok_or_else(|| format!("Invalid memory limit: '{number}'"))?;
    while idx < bytes.len() && is_space(bytes[idx]) {
        idx += 1;
    }
    let start = idx;
    while idx < bytes.len() && !is_space(bytes[idx]) {
        idx += 1;
    }
    if limit < 0.0 {
        // "Memory cannot be negative", which `ParseMemoryLimit` turns into
        // `-1` as an `idx_t`.
        return Ok(u64::MAX);
    }
    let unit = input[start..idx].to_ascii_lowercase();
    let multiplier: u64 = match unit.as_str() {
        "byte" | "bytes" | "b" => 1,
        "kilobyte" | "kilobytes" | "kb" | "k" => 1_000,
        "megabyte" | "megabytes" | "mb" | "m" => 1_000_000,
        "gigabyte" | "gigabytes" | "gb" | "g" => 1_000_000_000,
        "terabyte" | "terabytes" | "tb" | "t" => 1_000_000_000_000,
        "kib" => 1 << 10,
        "mib" => 1 << 20,
        "gib" => 1 << 30,
        "tib" => 1 << 40,
        _ => {
            return Err(format!(
                "Unknown unit for memory: '{unit}' (expected: KB, MB, GB, TB for 1000^i units or \
                 KiB, MiB, GiB, TiB for 1024^i units)"
            ));
        },
    };
    #[allow(clippy::cast_precision_loss)]
    let max_value = u64::MAX as f64;
    #[allow(clippy::cast_precision_loss)]
    let multiplier = multiplier as f64;
    if limit > max_value / multiplier {
        return Err("Memory value out of range: value is too large".to_string());
    }
    // `LossyNumericCast<idx_t>` of the product: truncation toward zero, NaN
    // as zero.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    Ok((multiplier * limit) as u64)
}

/// `value` as C++'s `std::to_string(double)` writes it (`"%f"`), the form
/// official's `ducklake_set_option` stores a `rewrite_delete_threshold` in:
/// six fractional digits, `nan`, `inf`.
pub(crate) fn double_to_std_string(value: f64) -> String {
    if value.is_nan() {
        if value.is_sign_negative() {
            "-nan".to_string()
        } else {
            "nan".to_string()
        }
    } else if value.is_infinite() {
        if value < 0.0 {
            "-inf"
        } else {
            "inf"
        }
        .to_string()
    } else {
        format!("{value:.6}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `(input, UBIGINT)` as DuckDB 1.5.5 casts it
    /// (`TRY_CAST('<input>' AS UBIGINT)`), `None` for NULL.
    const DUCKDB: &[(&str, Option<u64>)] = &[
        ("1.5", Some(2)),
        ("1e2", Some(100)),
        ("1_000", Some(1000)),
        ("0x10", Some(16)),
        ("-1", None),
        ("-0", Some(0)),
        (" 7 ", Some(7)),
        ("+07", Some(7)),
        ("", None),
        ("ten", None),
        ("1.4", Some(1)),
        ("2.5", Some(3)),
        ("0.5", Some(1)),
        ("1e-1", Some(0)),
        ("1.5e1", Some(15)),
        ("0b101", Some(5)),
        ("0o17", None),
        ("1__0", None),
        ("_1", None),
        ("1_", None),
        ("0X1F", Some(31)),
        ("18446744073709551615", Some(18_446_744_073_709_551_615)),
        ("18446744073709551616", None),
        ("1e20", None),
        ("-0.4", None),
        ("-0.5", None),
        ("0.49999", Some(0)),
        ("1.", Some(1)),
        (".5", Some(1)),
        ("1e", None),
        ("e1", None),
        ("0x", None),
        ("0xFFFFFFFFFFFFFFFF", Some(18_446_744_073_709_551_615)),
        ("0x1_0", Some(16)),
        ("1_0.5", Some(11)),
        ("1.2_3", Some(1)),
        ("1e1_0", Some(10_000_000_000)),
        ("\t8\n", Some(8)),
        ("+", None),
        ("-", None),
        ("1 2", None),
        ("00012", Some(12)),
        ("1E2", Some(100)),
        ("1.5E1", Some(15)),
        ("12.50e-1", Some(1)),
        ("0b", None),
        ("0B11", Some(3)),
        ("1e+2", Some(100)),
        ("+1e2", Some(100)),
        ("0x-1", None),
        ("-0x1", None),
        ("1.0000000000000000000001", Some(1)),
        ("99999999999999999999.5", None),
        ("18446744073709551615.4", Some(18_446_744_073_709_551_615)),
        ("18446744073709551615.5", None),
        ("1d", None),
        ("inf", None),
        ("nan", None),
        ("0.0", Some(0)),
        ("1,000", None),
    ];

    #[test]
    fn strings_cast_to_ubigint_as_duckdb_casts_them() {
        for (input, ubigint) in DUCKDB {
            assert_eq!(varchar_to_ubigint(input), *ubigint, "UBIGINT {input:?}");
        }
    }

    /// `(input, DOUBLE)` as DuckDB 1.5.5 casts it
    /// (`TRY_CAST('<input>' AS DOUBLE)`), `None` for NULL.
    const DUCKDB_DOUBLE: &[(&str, Option<f64>)] = &[
        (" 0.25 ", Some(0.25)),
        ("+.5", Some(0.5)),
        (".5", Some(0.5)),
        ("5.", Some(5.0)),
        ("1e-1", Some(0.1)),
        ("inf", Some(f64::INFINITY)),
        ("-inf", Some(f64::NEG_INFINITY)),
        ("Infinity", Some(f64::INFINITY)),
        ("-Infinity", Some(f64::NEG_INFINITY)),
        ("1_0", Some(10.0)),
        ("1__0", None),
        ("1_", None),
        ("0x1", None),
        ("1d", None),
        ("1,5", None),
        ("", None),
        ("1e", None),
        ("1e+", None),
        ("-0", Some(-0.0)),
        ("+-1", Some(-1.0)),
        ("1.5_5", Some(1.55)),
        ("1e1_0", Some(1e10)),
        (".", None),
        ("-.5", Some(-0.5)),
        ("1E2", Some(100.0)),
        ("++1", None),
        ("inff", None),
        ("1 2", None),
    ];

    /// `TRY_CAST(<input> AS TINYINT)` in the DuckDB 1.5.5 CLI.
    #[test]
    fn strings_cast_to_tinyint_as_duckdb_casts_them() {
        for (input, tinyint) in [
            ("0x10", Some(16)),
            (" 0x10 ", None),
            ("0xFF", None),
            ("-0.5", Some(-1)),
            ("0.4", Some(0)),
            ("127.4", Some(127)),
            ("127.5", None),
            ("-128.5", None),
            ("-128.4", Some(-128)),
            ("1e2", Some(100)),
            ("1e3", None),
            ("1.27e2", Some(127)),
            ("-1_0", Some(-10)),
            ("0b11", Some(3)),
            ("12.", Some(12)),
            (" -3 ", Some(-3)),
            ("128", None),
            ("true", None),
            ("", None),
        ] {
            assert_eq!(varchar_to_tinyint(input), tinyint, "TINYINT {input:?}");
        }
    }

    #[test]
    fn strings_cast_to_double_as_duckdb_casts_them() {
        for (input, double) in DUCKDB_DOUBLE {
            assert_eq!(varchar_to_double(input), *double, "DOUBLE {input:?}");
        }
        assert!(varchar_to_double("nan").is_some_and(f64::is_nan));
        assert!(varchar_to_double("NaN").is_some_and(f64::is_nan));
    }

    #[test]
    fn strings_cast_to_boolean_as_duckdb_casts_them() {
        for (input, boolean) in [
            ("t", Some(true)),
            ("T", Some(true)),
            ("TRUE", Some(true)),
            ("tRuE", Some(true)),
            ("y", Some(true)),
            ("YES", Some(true)),
            ("1", Some(true)),
            ("0", Some(false)),
            ("F", Some(false)),
            ("fAlSe", Some(false)),
            ("N", Some(false)),
            ("No", Some(false)),
            ("on", None),
            ("off", None),
            (" true", None),
            ("2", None),
            ("1.0", None),
            ("ye", None),
            ("tr", None),
            ("truee", None),
            ("", None),
        ] {
            assert_eq!(varchar_to_boolean(input), boolean, "BOOLEAN {input:?}");
        }
    }

    /// `(input, stored value or DuckDB's error)` as official's
    /// `ducklake_set_option('target_file_size', '<input>')` stores it on
    /// DuckDB 1.5.5.
    #[test]
    fn memory_limits_parse_as_duckdb_parses_them() {
        let unit_error = |unit: &str| {
            Err(format!(
                "Unknown unit for memory: '{unit}' (expected: KB, MB, GB, TB for 1000^i units or \
                 KiB, MiB, GiB, TiB for 1024^i units)"
            ))
        };
        for (input, parsed) in [
            ("1000", unit_error("")),
            ("1.5b", Ok(1)),
            ("1kb", Ok(1000)),
            ("1k", Ok(1000)),
            ("-1", Ok(u64::MAX)),
            ("none", Ok(u64::MAX)),
            (" 3 MB extra", Ok(3_000_000)),
            ("0", unit_error("")),
            ("0 b", Ok(0)),
            (
                "abc",
                Err("Memory must have a number (e.g. 1GB)".to_string()),
            ),
            ("1-2 kb", Err("Invalid memory limit: '1-2'".to_string())),
            ("1 zb", unit_error("zb")),
            ("", Err("Memory must have a number (e.g. 1GB)".to_string())),
            (
                "1e30 tb",
                Err("Memory value out of range: value is too large".to_string()),
            ),
            ("1e2KiB", Ok(102_400)),
            (" -1 kb", Ok(u64::MAX)),
            ("1.5MB", Ok(1_500_000)),
            ("2 GiB", Ok(2_147_483_648)),
        ] {
            assert_eq!(parse_memory_limit(input), parsed, "memory limit {input:?}");
        }
    }

    #[test]
    fn doubles_print_as_std_to_string_prints_them() {
        for (value, printed) in [
            (0.5, "0.500000"),
            (0.123_456_7, "0.123457"),
            (1e-7, "0.000000"),
            (-0.0, "-0.000000"),
            (1.0, "1.000000"),
            (f64::NAN, "nan"),
        ] {
            assert_eq!(double_to_std_string(value), printed);
        }
    }
}
