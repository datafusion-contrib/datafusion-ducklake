//! A seeded, randomized differential harness for data inlining and the writes
//! around it.
//!
//! Each seed generates a table and a history of writes and maintenance (small
//! and large inserts, updates, deletes, flushes, checkpoints, compaction,
//! column changes, inlining limits, partitioning, expiry), with values at the
//! edges of their types. The history runs twice: through official DuckLake
//! (the pinned DuckDB) on a catalog of its own, the oracle, and through this
//! crate on a catalog of the backend under test. After every step the harness
//! compares:
//!
//! - whether the step failed on each side, and how (the error's class);
//! - the rows with their row ids at the current snapshot, and at earlier
//!   steps' snapshots (time travel);
//! - `ducklake_table_changes` over windows of steps;
//! - this crate's answers to generated predicates and aggregates, at the head
//!   and at an earlier snapshot, against the *truth*: official's table read
//!   whole into a plain DuckDB table and queried there, where no statistics
//!   can prune;
//! - official's own pruned answers against the truth: they may differ only
//!   where a modelled upstream bug is;
//! - that each catalog's table-wide statistics cover the values it holds;
//! - where DuckDB can attach the crate's catalog (SQLite and DuckDB metadata),
//!   DuckDB's answers over it, the statistics rows, and the Hive directories
//!   and partition values of the live files, against official's.
//!
//! Official's `UPDATE` and `DELETE` run without filter pushdown, so its rows
//! stay right where its statistics are known to be wrong.
//!
//! Snapshot ids differ between the two catalogs (a crate checkpoint commits
//! each table's compaction on its own, for one), so snapshots are matched by
//! the step that made them. Known upstream data bugs, where this crate stays
//! correct, are modelled in [`known`]; each switches off only the cells it
//! affects (a column, a snapshot, a window, a constant), never the rest of a
//! history. [`coverage`] counts each comparison run and skipped, and why; the
//! fuzz and soak runs print the counts.
//!
//! On a mismatch the history is shrunk (steps, rows and columns dropped while
//! the same kind of mismatch remains) and printed as a regression test.
//!
//! `FUZZ_SEEDS` sets the number of seeds per backend (default 8 on SQLite and
//! DuckDB, 3 on the Docker backends, to keep the default suite's cost down;
//! `FUZZ_SEEDS=50` and `FUZZ_SEEDS=20` are the fuller runs) and `FUZZ_SEED`
//! runs one seed. An exhaustive (regression) run reads the rows at each earlier
//! snapshot and each change-feed window when it lands and all of them after the
//! last step; `FUZZ_ALL_WINDOWS` reads every one after every step. The
//! `#[ignore]`d `soak_*` tests run `FUZZ_SEEDS` seeds (default 2000; 300 on the
//! Docker backends).
//!
//! In CI the Linux `test-full` job runs this module (its features include
//! `write-duckdb`); the macOS job's feature set compiles it out.

#![cfg(all(feature = "write-sqlite", feature = "write-duckdb"))]

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Fields, TimeUnit};
use datafusion::prelude::SessionContext;
use datafusion_ducklake::metadata_writer::ColumnDef;
use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeTable, MergeOptions, MetadataProvider, MetadataWriter, RewriteOptions,
    WriteMode, execute_ducklake_sql, register_ducklake_functions,
};

use crate::inlined_mutation_differential_tests::Metadata;

/// Canonical text of `duckdb`'s Arrow arrays.
mod canon_duckdb {
    use duckdb::arrow as arrow_impl;
    include!("fuzz_canon.rs");
}

/// Canonical text of this crate's Arrow arrays.
mod canon_crate {
    use ::arrow as arrow_impl;
    include!("fuzz_canon.rs");
}

// ---------------------------------------------------------------------------
// Randomness
// ---------------------------------------------------------------------------

/// SplitMix64: small, seedable and the same on every platform.
#[derive(Clone)]
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn chance(&mut self, percent: u64) -> bool {
        self.next() % 100 < percent
    }

    fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        &items[self.below(items.len())]
    }
}

// ---------------------------------------------------------------------------
// Types and values
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Ty {
    Int,
    BigInt,
    Double,
    Float,
    Decimal,
    Varchar,
    Bool,
    Date,
    Time,
    Ts,
    TsTz,
    Blob,
    List,
    Struct,
    Map,
    SmallInt,
    UBigInt,
    /// `DECIMAL(38,10)`: wider than an `i64` holds.
    Decimal38,
    TsNs,
}

const TYPES: [Ty; 19] = [
    Ty::Int,
    Ty::BigInt,
    Ty::Double,
    Ty::Float,
    Ty::Decimal,
    Ty::Varchar,
    Ty::Bool,
    Ty::Date,
    Ty::Time,
    Ty::Ts,
    Ty::TsTz,
    Ty::Blob,
    Ty::List,
    Ty::Struct,
    Ty::Map,
    Ty::SmallInt,
    Ty::UBigInt,
    Ty::Decimal38,
    Ty::TsNs,
];

impl Ty {
    fn duckdb(self) -> &'static str {
        match self {
            Ty::Int => "INTEGER",
            Ty::BigInt => "BIGINT",
            Ty::Double => "DOUBLE",
            Ty::Float => "FLOAT",
            Ty::Decimal => "DECIMAL(18,3)",
            Ty::Varchar => "VARCHAR",
            Ty::Bool => "BOOLEAN",
            Ty::Date => "DATE",
            Ty::Time => "TIME",
            Ty::Ts => "TIMESTAMP",
            Ty::TsTz => "TIMESTAMPTZ",
            Ty::Blob => "BLOB",
            Ty::List => "INTEGER[]",
            Ty::Struct => "STRUCT(a INTEGER, b VARCHAR)",
            Ty::Map => "MAP(VARCHAR, INTEGER)",
            Ty::SmallInt => "SMALLINT",
            Ty::UBigInt => "UBIGINT",
            Ty::Decimal38 => "DECIMAL(38,10)",
            Ty::TsNs => "TIMESTAMP_NS",
        }
    }

    fn arrow(self) -> DataType {
        match self {
            Ty::Int => DataType::Int32,
            Ty::BigInt => DataType::Int64,
            Ty::Double => DataType::Float64,
            Ty::Float => DataType::Float32,
            Ty::Decimal => DataType::Decimal128(18, 3),
            Ty::Varchar => DataType::Utf8,
            Ty::Bool => DataType::Boolean,
            Ty::Date => DataType::Date32,
            Ty::Time => DataType::Time64(TimeUnit::Microsecond),
            Ty::Ts => DataType::Timestamp(TimeUnit::Microsecond, None),
            Ty::TsTz => DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            Ty::Blob => DataType::Binary,
            Ty::List => DataType::List(Arc::new(Field::new("element", DataType::Int32, true))),
            Ty::Struct => DataType::Struct(Fields::from(vec![
                Field::new("a", DataType::Int32, true),
                Field::new("b", DataType::Utf8, true),
            ])),
            Ty::Map => DataType::Map(
                Arc::new(Field::new(
                    "key_value",
                    DataType::Struct(Fields::from(vec![
                        Field::new("key", DataType::Utf8, false),
                        Field::new("value", DataType::Int32, true),
                    ])),
                    false,
                )),
                false,
            ),
            Ty::SmallInt => DataType::Int16,
            Ty::UBigInt => DataType::UInt64,
            Ty::Decimal38 => DataType::Decimal128(38, 10),
            Ty::TsNs => DataType::Timestamp(TimeUnit::Nanosecond, None),
        }
    }

    /// The type this crate's SQL names in `CAST(NULL AS …)`.
    fn datafusion_null(self) -> String {
        match self {
            Ty::Int => "CAST(NULL AS INT)".into(),
            Ty::BigInt => "CAST(NULL AS BIGINT)".into(),
            Ty::Double => "CAST(NULL AS DOUBLE)".into(),
            Ty::Float => "CAST(NULL AS REAL)".into(),
            Ty::Decimal => "CAST(NULL AS DECIMAL(18,3))".into(),
            Ty::Varchar => "CAST(NULL AS VARCHAR)".into(),
            Ty::Bool => "CAST(NULL AS BOOLEAN)".into(),
            Ty::Date => "arrow_cast(NULL, 'Date32')".into(),
            Ty::Time => "arrow_cast(NULL, 'Time64(Microsecond)')".into(),
            Ty::Ts => "arrow_cast(NULL, 'Timestamp(Microsecond, None)')".into(),
            Ty::TsTz => "arrow_cast(NULL, 'Timestamp(Microsecond, Some(\"UTC\"))')".into(),
            Ty::Blob => "CAST(NULL AS BYTEA)".into(),
            Ty::SmallInt => "CAST(NULL AS SMALLINT)".into(),
            Ty::UBigInt => "arrow_cast(NULL, 'UInt64')".into(),
            Ty::Decimal38 => "CAST(NULL AS DECIMAL(38,10))".into(),
            Ty::TsNs => "arrow_cast(NULL, 'Timestamp(Nanosecond, None)')".into(),
            // A nested NULL takes its type from a branch never taken.
            Ty::List | Ty::Struct | Ty::Map => format!(
                "CASE WHEN false THEN {} END",
                datafusion_literal(
                    self,
                    &match self {
                        Ty::List => V::List(vec![Some(0)]),
                        Ty::Struct => V::Struct(Some(0), Some(String::new())),
                        _ => V::Map(vec![(String::new(), Some(0))]),
                    }
                )
            ),
        }
    }

    fn is_float(self) -> bool {
        matches!(self, Ty::Double | Ty::Float)
    }

    fn is_temporal(self) -> bool {
        matches!(self, Ty::Date | Ty::Ts | Ty::TsTz)
    }

    /// Whether `=`, `<` and `>` compare this type in both engines alike.
    fn comparable(self) -> bool {
        !matches!(self, Ty::List | Ty::Struct | Ty::Map | Ty::Blob)
    }
}

/// A value of a column, in the representation both dialects render.
#[derive(Clone, Debug, PartialEq)]
enum V {
    Null,
    I(i64),
    F(f64),
    R(f32),
    /// A `DECIMAL(18,3)`, scaled by 1000.
    D(i64),
    S(String),
    B(bool),
    /// Days since 1970-01-01, DuckDB's `infinity` as `i32::MAX`.
    Date(i32),
    /// Microseconds since midnight.
    Time(i64),
    /// Microseconds since 1970-01-01 UTC, DuckDB's `infinity` as `i64::MAX`.
    Ts(i64),
    Bytes(Vec<u8>),
    List(Vec<Option<i32>>),
    Struct(Option<i32>, Option<String>),
    Map(Vec<(String, Option<i32>)>),
    U(u64),
    /// A `DECIMAL(38,10)`, scaled by 10^10.
    Big(i128),
    /// Nanoseconds since 1970-01-01.
    Tn(i64),
}

/// DuckDB's earliest and latest finite `TIMESTAMP` microseconds.
const TS_MIN: i64 = -9_223_372_022_400_000_000;
const TS_MAX: i64 = 9_223_372_036_854_775_806;

/// Strings past the 256 bytes DuckDB's Parquet writer keeps of a bound: one
/// it truncates at a byte, one with a character across byte 256, one whose
/// maximum it cannot truncate, and one past the 2048 bytes this crate's
/// filter pushdown takes a constant of.
fn long_strings() -> [String; 4] {
    [
        format!("{}z", "b".repeat(300)),
        format!("{}éyy", "a".repeat(255)),
        "é".repeat(140),
        format!("{}y", "c".repeat(3000)),
    ]
}

fn gen_value(rng: &mut Rng, ty: Ty) -> V {
    if rng.chance(15) {
        return V::Null;
    }
    let edge = rng.chance(35);
    match ty {
        Ty::Int => V::I(if edge {
            *rng.pick(&[i64::from(i32::MIN), i64::from(i32::MAX), 0, -1])
        } else {
            rng.below(20) as i64 - 10
        }),
        Ty::BigInt => V::I(if edge {
            *rng.pick(&[i64::MIN, i64::MAX, 0, -1])
        } else {
            rng.below(20) as i64 - 10
        }),
        Ty::Double => V::F(if edge {
            *rng.pick(&[
                f64::NAN,
                f64::INFINITY,
                f64::NEG_INFINITY,
                -0.0,
                0.0,
                f64::MAX,
                f64::MIN,
                f64::MIN_POSITIVE,
                1e300,
                1e20,
                -1e20,
                1e-7,
                5e-324,
                1e-310,
                0.48024702758086213,
            ])
        } else if rng.chance(30) {
            // A random double printed without an exponent, most with 16 or
            // 17 significant digits: DuckDB reads that text as a `DECIMAL`.
            let mantissa = rng.next() >> 12;
            let exponent = 1023 - 14 + rng.next() % 67;
            let sign = rng.next() & 1;
            f64::from_bits((sign << 63) | (exponent << 52) | mantissa)
        } else {
            (rng.below(40) as f64 - 20.0) / 4.0
        }),
        Ty::Float => V::R(if edge {
            *rng.pick(&[
                f32::NAN,
                f32::INFINITY,
                f32::NEG_INFINITY,
                -0.0,
                f32::MAX,
                f32::MIN,
                f32::MIN_POSITIVE,
                0.1,
                1e20,
                -1e20,
                1e-7,
                1.5e-10,
                1e-45,
                1e-40,
            ])
        } else {
            (rng.below(40) as f32 - 20.0) / 4.0
        }),
        Ty::Decimal => V::D(if edge {
            *rng.pick(&[999_999_999_999_999_999, -999_999_999_999_999_999, 0, 1, -1])
        } else {
            rng.below(20_000) as i64 - 10_000
        }),
        Ty::Varchar if edge && rng.chance(25) => V::S(rng.pick(&long_strings()).clone()),
        Ty::Varchar => V::S(if edge {
            rng.pick(&[
                "",
                "a\0b",
                "it's",
                "héllo ☃ 𝄞",
                "a very long string that is well past sixty four bytes in length, really",
                "%_\\",
            ])
            .to_string()
        } else {
            rng.pick(&["a", "b", "c", "x", "yy", "zzz"]).to_string()
        }),
        Ty::Bool => V::B(rng.chance(50)),
        Ty::Date => V::Date(if edge {
            *rng.pick(&[
                i32::MAX,
                -i32::MAX,
                2_147_483_646,
                -2_147_483_646,
                -755_687,
                2_932_896,
                -719_528,
                -719_162,
                108_853_222,
            ])
        } else {
            19_000 + rng.below(40) as i32
        }),
        Ty::Time => V::Time(if edge {
            *rng.pick(&[0, 86_400_000_000, 1, 45_296_500_000])
        } else {
            rng.below(86_400) as i64 * 1_000_000
        }),
        Ty::Ts | Ty::TsTz => V::Ts(if edge {
            *rng.pick(&[
                i64::MAX,
                -i64::MAX,
                TS_MIN,
                TS_MAX,
                -65_291_356_800_000_000,
                -63_517_778_999_500_000,
                8_458_224_206_400_123_456,
                253_402_300_800_000_000,
            ])
        } else {
            1_700_000_000_000_000 + rng.below(1000) as i64 * 3_600_000_000
        }),
        Ty::Blob => V::Bytes(if edge {
            rng.pick(&[
                vec![],
                vec![0, 1, 255],
                b"ab'\\".to_vec(),
                vec![0x62; 300],
                vec![0xFF; 300],
                vec![0x63; 3000],
            ])
            .clone()
        } else {
            vec![rng.below(256) as u8]
        }),
        Ty::List => V::List(if edge {
            rng.pick(&[vec![], vec![Some(1), None, Some(3)], vec![Some(i32::MAX)]])
                .clone()
        } else {
            vec![Some(rng.below(10) as i32)]
        }),
        Ty::Struct => {
            if edge {
                V::Struct(None, None)
            } else {
                V::Struct(
                    Some(rng.below(10) as i32),
                    Some(rng.pick(&["p", "q"]).to_string()),
                )
            }
        },
        Ty::Map => V::Map(if edge {
            rng.pick(&[vec![], vec![("z".to_string(), None)]]).clone()
        } else {
            vec![("k".to_string(), Some(rng.below(10) as i32))]
        }),
        Ty::SmallInt => V::I(if edge {
            *rng.pick(&[i64::from(i16::MIN), i64::from(i16::MAX), 0, -1])
        } else {
            rng.below(20) as i64 - 10
        }),
        Ty::UBigInt => V::U(if edge {
            *rng.pick(&[0, 1, u64::MAX, 1 << 63, (1 << 63) - 1])
        } else {
            rng.below(20) as u64
        }),
        Ty::Decimal38 => V::Big(if edge {
            let max = 10i128.pow(38) - 1;
            *rng.pick(&[max, -max, 0, 1, -1, 12_345_678_901_234_567_890_123])
        } else {
            (rng.below(20_000) as i128 - 10_000) * 1_000_000
        }),
        Ty::TsNs => V::Tn(if edge {
            *rng.pick(&[
                0,
                -1,
                1,
                9_000_000_000_000_000_000,
                -9_000_000_000_000_000_000,
                1_700_000_000_123_456_789,
            ])
        } else {
            1_700_000_000_000_000_000 + rng.below(1000) as i64 * 3_600_000_000_123
        }),
    }
}

fn decimal38_text(scaled: i128) -> String {
    let sign = if scaled < 0 {
        "-"
    } else {
        ""
    };
    let magnitude = scaled.unsigned_abs();
    format!(
        "{sign}{}.{:010}",
        magnitude / 10_000_000_000,
        magnitude % 10_000_000_000
    )
}

// --- DuckDB's text of dates and timestamps -----------------------------------

fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let month = if mp < 10 {
        mp + 3
    } else {
        mp - 9
    } as u32;
    (yoe + era * 400 + i64::from(month <= 2), month, day)
}

fn date_text(days: i64) -> String {
    let (year, month, day) = civil_from_days(days);
    if year <= 0 {
        format!("{:04}-{month:02}-{day:02} (BC)", 1 - year)
    } else {
        format!("{year:04}-{month:02}-{day:02}")
    }
}

fn clock_text(micros_of_day: i64) -> String {
    let secs = micros_of_day / 1_000_000;
    let sub = micros_of_day % 1_000_000;
    let mut text = format!("{:02}:{:02}:{:02}", secs / 3600, secs / 60 % 60, secs % 60);
    if sub != 0 {
        let fraction = format!("{sub:06}");
        text.push('.');
        text.push_str(fraction.trim_end_matches('0'));
    }
    text
}

fn timestamp_text(micros: i64) -> String {
    match micros {
        i64::MAX => "infinity".into(),
        v if v == -i64::MAX => "-infinity".into(),
        v => {
            let days = v.div_euclid(86_400_000_000);
            let rest = v.rem_euclid(86_400_000_000);
            let date = date_text(days);
            match date.strip_suffix(" (BC)") {
                Some(date) => format!("{date} (BC) {}", clock_text(rest)),
                None => format!("{date} {}", clock_text(rest)),
            }
        },
    }
}

fn float_text(value: f64) -> String {
    if value.is_nan() {
        "NaN".into()
    } else if value == f64::INFINITY {
        "inf".into()
    } else if value == f64::NEG_INFINITY {
        "-inf".into()
    } else {
        format!("{value:?}")
    }
}

fn float32_text(value: f32) -> String {
    if value.is_nan() {
        "NaN".into()
    } else if value == f32::INFINITY {
        "inf".into()
    } else if value == f32::NEG_INFINITY {
        "-inf".into()
    } else {
        format!("{value:?}")
    }
}

fn decimal_text(scaled: i64) -> String {
    let sign = if scaled < 0 {
        "-"
    } else {
        ""
    };
    let magnitude = scaled.unsigned_abs();
    format!("{sign}{}.{:03}", magnitude / 1000, magnitude % 1000)
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02X}")).collect()
}

fn quote(text: &str) -> String {
    format!("'{}'", text.replace('\'', "''"))
}

/// A string literal, with NUL bytes joined in by `chr(0)` in either dialect.
fn string_literal(text: &str, concat: &str) -> String {
    if !text.contains('\0') {
        return quote(text);
    }
    let parts = text.split('\0').map(quote).collect::<Vec<_>>();
    match concat {
        "||" => format!("({})", parts.join(" || chr(0) || ")),
        _ => format!("concat({})", parts.join(", chr(0), ")),
    }
}

/// `value` of type `ty` as a DuckDB SQL expression.
fn duckdb_literal(ty: Ty, value: &V) -> String {
    match (ty, value) {
        (_, V::Null) => format!("CAST(NULL AS {})", ty.duckdb()),
        (Ty::Int | Ty::BigInt, V::I(v)) => format!("CAST('{v}' AS {})", ty.duckdb()),
        (Ty::Double, V::F(v)) => format!("CAST('{}' AS DOUBLE)", float_text(*v)),
        (Ty::Float, V::R(v)) => format!("CAST('{}' AS FLOAT)", float32_text(*v)),
        (Ty::Decimal, V::D(v)) => format!("CAST('{}' AS DECIMAL(18,3))", decimal_text(*v)),
        (Ty::Varchar, V::S(s)) => string_literal(s, "||"),
        (Ty::Bool, V::B(b)) => b.to_string(),
        (Ty::Date, V::Date(d)) => {
            let text = match *d {
                i32::MAX => "infinity".to_string(),
                d if d == -i32::MAX => "-infinity".to_string(),
                d => date_text(i64::from(d)),
            };
            format!("CAST('{text}' AS DATE)")
        },
        (Ty::Time, V::Time(t)) => format!("CAST('{}' AS TIME)", clock_text(*t)),
        (Ty::Ts, V::Ts(t)) => format!("CAST('{}' AS TIMESTAMP)", timestamp_text(*t)),
        (Ty::TsTz, V::Ts(t)) => {
            let text = timestamp_text(*t);
            let text = if text.ends_with("infinity") {
                text
            } else {
                format!("{text}+00")
            };
            format!("CAST('{text}' AS TIMESTAMPTZ)")
        },
        (Ty::Blob, V::Bytes(b)) => format!("from_hex('{}')", hex(b)),
        (Ty::List, V::List(items)) => {
            let items = items
                .iter()
                .map(|item| item.map_or("NULL".to_string(), |v| v.to_string()))
                .collect::<Vec<_>>();
            format!("CAST([{}] AS INTEGER[])", items.join(", "))
        },
        (Ty::Struct, V::Struct(a, b)) => format!(
            "CAST({{'a': {}, 'b': {}}} AS STRUCT(a INTEGER, b VARCHAR))",
            a.map_or("NULL".to_string(), |v| v.to_string()),
            b.as_deref().map_or("NULL".to_string(), quote)
        ),
        (Ty::Map, V::Map(entries)) => {
            let entries = entries
                .iter()
                .map(|(k, v)| {
                    format!(
                        "{}: {}",
                        quote(k),
                        v.map_or("NULL".to_string(), |v| v.to_string())
                    )
                })
                .collect::<Vec<_>>();
            format!(
                "CAST(MAP {{{}}} AS MAP(VARCHAR, INTEGER))",
                entries.join(", ")
            )
        },
        (Ty::SmallInt, V::I(v)) => format!("CAST('{v}' AS SMALLINT)"),
        (Ty::UBigInt, V::U(v)) => format!("CAST('{v}' AS UBIGINT)"),
        (Ty::Decimal38, V::Big(v)) => format!("CAST('{}' AS DECIMAL(38,10))", decimal38_text(*v)),
        (Ty::TsNs, V::Tn(v)) => format!("make_timestamp_ns({v})"),
        (ty, value) => panic!("no {ty:?} value {value:?}"),
    }
}

/// `value` of type `ty` as an expression of this crate's (DataFusion's) SQL.
fn datafusion_literal(ty: Ty, value: &V) -> String {
    match (ty, value) {
        (_, V::Null) => ty.datafusion_null(),
        (Ty::Int, V::I(v)) => format!("CAST('{v}' AS INT)"),
        (Ty::BigInt, V::I(v)) => format!("CAST('{v}' AS BIGINT)"),
        (Ty::Double, V::F(v)) => format!("CAST('{}' AS DOUBLE)", float_text(*v)),
        (Ty::Float, V::R(v)) => format!("CAST('{}' AS REAL)", float32_text(*v)),
        (Ty::Decimal, V::D(v)) => format!("CAST('{}' AS DECIMAL(18,3))", decimal_text(*v)),
        (Ty::Varchar, V::S(s)) => string_literal(s, "concat"),
        (Ty::Bool, V::B(b)) => b.to_string(),
        (Ty::Date, V::Date(d)) => format!("arrow_cast({d}, 'Date32')"),
        (Ty::Time, V::Time(t)) => format!("arrow_cast({t}, 'Time64(Microsecond)')"),
        (Ty::Ts, V::Ts(t)) => format!("arrow_cast({t}, 'Timestamp(Microsecond, None)')"),
        (Ty::TsTz, V::Ts(t)) => {
            format!("arrow_cast({t}, 'Timestamp(Microsecond, Some(\"UTC\"))')")
        },
        (Ty::Blob, V::Bytes(b)) => format!("decode('{}', 'hex')", hex(b)),
        (Ty::List, V::List(items)) => {
            let items = items
                .iter()
                .map(|item| {
                    item.map_or("CAST(NULL AS INT)".to_string(), |v| {
                        format!("CAST({v} AS INT)")
                    })
                })
                .collect::<Vec<_>>();
            format!(
                "arrow_cast(make_array({}), 'List(Int32)')",
                items.join(", ")
            )
        },
        (Ty::Struct, V::Struct(a, b)) => format!(
            "named_struct('a', {}, 'b', {})",
            a.map_or("CAST(NULL AS INT)".to_string(), |v| format!(
                "CAST({v} AS INT)"
            )),
            b.as_deref()
                .map_or("CAST(NULL AS VARCHAR)".to_string(), quote)
        ),
        (Ty::Map, V::Map(entries)) => {
            let keys = entries.iter().map(|(k, _)| quote(k)).collect::<Vec<_>>();
            let values = entries
                .iter()
                .map(|(_, v)| {
                    v.map_or("CAST(NULL AS INT)".to_string(), |v| {
                        format!("CAST({v} AS INT)")
                    })
                })
                .collect::<Vec<_>>();
            format!(
                "map(arrow_cast(make_array({}), 'List(Utf8)'), arrow_cast(make_array({}), 'List(Int32)'))",
                keys.join(", "),
                values.join(", ")
            )
        },
        (Ty::SmallInt, V::I(v)) => format!("CAST('{v}' AS SMALLINT)"),
        (Ty::UBigInt, V::U(v)) => format!("arrow_cast('{v}', 'UInt64')"),
        (Ty::Decimal38, V::Big(v)) => format!("CAST('{}' AS DECIMAL(38,10))", decimal38_text(*v)),
        (Ty::TsNs, V::Tn(v)) => {
            format!("arrow_cast(CAST('{v}' AS BIGINT), 'Timestamp(Nanosecond, None)')")
        },
        (ty, value) => panic!("no {ty:?} value {value:?}"),
    }
}

// ---------------------------------------------------------------------------
// Histories
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
struct Col {
    name: String,
    ty: Ty,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Transform {
    Identity,
    Year,
    Month,
    Day,
}

#[derive(Clone, Debug, PartialEq)]
enum Pred {
    IdEq(i64),
    IdLt(i64),
    IdGt(i64),
    IdIn(Vec<i64>),
    IsNull(String),
    Eq(String, Ty, V),
}

#[derive(Clone, Debug, PartialEq)]
enum Step {
    /// Rows of `id` and a value per column, in column order.
    Insert(Vec<(i64, Vec<V>)>),
    Update {
        column: String,
        ty: Ty,
        value: V,
        pred: Pred,
    },
    Delete(Option<Pred>),
    Flush,
    Checkpoint,
    Merge,
    Rewrite,
    AddColumn(Col, Option<String>),
    Widen(String),
    Limit(usize),
    Partition(Option<(String, Transform)>),
    Expire,
}

#[derive(Clone, Debug, PartialEq)]
struct History {
    /// The columns the table is created with, after `id`.
    columns: Vec<Col>,
    /// The table's inlining limit when created; `None` keeps the default.
    limit: Option<usize>,
    steps: Vec<Step>,
}

impl History {
    /// The columns after `id` at `step`.
    fn columns_at(&self, step: usize) -> Vec<Col> {
        let mut columns = self.columns.clone();
        for s in &self.steps[..step] {
            match s {
                Step::AddColumn(column, _) => columns.push(column.clone()),
                Step::Widen(name) => {
                    for column in &mut columns {
                        if &column.name == name {
                            column.ty = Ty::BigInt;
                        }
                    }
                },
                _ => {},
            }
        }
        columns
    }
}

/// What the backend under test supports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Sqlite,
    Duckdb,
    Mysql,
    PostgresSingle,
    Multicatalog,
}

impl Kind {
    fn mutations(self) -> bool {
        self != Kind::PostgresSingle
    }

    fn maintenance(self) -> bool {
        self != Kind::PostgresSingle
    }

    /// Whether SQL writes inline and columns widen: the single-catalog
    /// PostgreSQL writer keeps SQL writes in Parquet and has no
    /// `promote_column_type` (see COMPATIBILITY.md), so its histories keep
    /// the limit at `0` on both sides.
    fn inlines(self) -> bool {
        self != Kind::PostgresSingle
    }
}

/// The history of seed `seed` on a backend of `kind`.
///
/// Not generated: `DROP COLUMN` and `RENAME COLUMN`, which this crate has no
/// write path for; `UUID` and `INTERVAL` columns, whose values the two engines'
/// Arrow output does not spell alike (and an `INTERVAL` cannot be flushed);
/// and histories over more than one table or with concurrent transactions.
fn generate(seed: u64, kind: Kind) -> History {
    let mut rng = Rng::new(seed);
    let mut columns = Vec::new();
    for index in 0..1 + rng.below(4) {
        columns.push(Col {
            name: format!("c{index}"),
            ty: *rng.pick(&TYPES),
        });
    }
    // A limit of `0` writes every insert to its own file, so the histories
    // hold multi-file tables the statistics filter can prune.
    let limit = rng.chance(70).then(|| *rng.pick(&[0usize, 0, 1, 10, 100]));
    let limit = if kind.inlines() {
        limit
    } else {
        Some(0)
    };
    let mut history = History {
        columns,
        limit,
        steps: Vec::new(),
    };
    let mut next_id = 1i64;
    let mut current_limit = limit.unwrap_or(10);
    let mut added = 0;
    // A string partition key never holds, and a column that held one never
    // becomes a partition key, a string past [`long_strings`]' length: its
    // percent-encoded Hive directory name is longer than a file name may be,
    // and official's write fails on the file system, not on DuckLake.
    let mut partition_key: Option<String> = None;
    let mut held_long: std::collections::BTreeSet<String> = Default::default();
    // A write that may inline gets no `DOUBLE` whose text official's inlined
    // write rounds (`known::inlined_double_text`): official's oracle stores
    // another value there. Such doubles still reach Parquet files, and every
    // filter probe.
    let value_for = |rng: &mut Rng,
                     column: &Col,
                     partition_key: &Option<String>,
                     held_long: &mut std::collections::BTreeSet<String>,
                     inline: bool| loop {
        let value = gen_value(rng, column.ty);
        let long = matches!(&value, V::S(s) if s.len() > 64);
        if long && partition_key.as_deref() == Some(column.name.as_str()) {
            continue;
        }
        if inline && known::double_fixed_point_filter(column.ty, &value) {
            continue;
        }
        if long {
            held_long.insert(column.name.clone());
        }
        return value;
    };
    for _ in 0..3 + rng.below(12) {
        let columns = history.columns_at(history.steps.len());
        let roll = rng.below(100);
        let step = match roll {
            0..=34 => {
                let rows = if rng.chance(60) {
                    1 + rng.below(current_limit.clamp(1, 4))
                } else {
                    current_limit + 1 + rng.below(20)
                };
                let all_null = rng.chance(10);
                let inline = rows <= current_limit;
                let rows = (0..rows)
                    .map(|_| {
                        let id = next_id;
                        next_id += 1;
                        let values = columns
                            .iter()
                            .map(|c| {
                                if all_null {
                                    V::Null
                                } else {
                                    value_for(&mut rng, c, &partition_key, &mut held_long, inline)
                                }
                            })
                            .collect();
                        (id, values)
                    })
                    .collect();
                Step::Insert(rows)
            },
            35..=46 if kind.mutations() => {
                let column = rng.pick(&columns).clone();
                Step::Update {
                    value: value_for(&mut rng, &column, &partition_key, &mut held_long, true),
                    column: column.name,
                    ty: column.ty,
                    pred: gen_pred(&mut rng, next_id, &columns),
                }
            },
            47..=56 if kind.mutations() => {
                Step::Delete((!rng.chance(10)).then(|| gen_pred(&mut rng, next_id, &columns)))
            },
            57..=63 => Step::Flush,
            64..=66 if kind.maintenance() => Step::Checkpoint,
            67..=70 if kind.maintenance() => Step::Merge,
            71..=74 if kind.maintenance() => Step::Rewrite,
            75..=79 if added < 2 => {
                added += 1;
                let ty = *rng.pick(&TYPES);
                let default = match ty {
                    Ty::Int | Ty::BigInt if rng.chance(50) => Some("7".to_string()),
                    Ty::Varchar if rng.chance(50) => Some("dflt".to_string()),
                    _ => None,
                };
                Step::AddColumn(
                    Col {
                        name: format!("a{added}"),
                        ty,
                    },
                    default,
                )
            },
            80..=89 if !kind.inlines() => Step::Flush,
            80..=82 => match columns.iter().find(|c| c.ty == Ty::Int) {
                Some(column) => Step::Widen(column.name.clone()),
                None => Step::Flush,
            },
            83..=89 => {
                current_limit = *rng.pick(&[0usize, 1, 10, 100]);
                Step::Limit(current_limit)
            },
            90..=95 => {
                let candidates = columns
                    .iter()
                    .filter(|c| {
                        (matches!(
                            c.ty,
                            Ty::Int | Ty::BigInt | Ty::SmallInt | Ty::Varchar | Ty::Bool
                        ) || c.ty.is_temporal())
                            && !held_long.contains(&c.name)
                    })
                    .collect::<Vec<_>>();
                if candidates.is_empty() || rng.chance(15) {
                    partition_key = None;
                    Step::Partition(None)
                } else {
                    let column = *rng.pick(&candidates);
                    let transform = if column.ty.is_temporal() {
                        *rng.pick(&[
                            Transform::Identity,
                            Transform::Year,
                            Transform::Month,
                            Transform::Day,
                        ])
                    } else {
                        Transform::Identity
                    };
                    partition_key = Some(column.name.clone());
                    Step::Partition(Some((column.name.clone(), transform)))
                }
            },
            96..=99 if kind.maintenance() => Step::Expire,
            _ => Step::Flush,
        };
        history.steps.push(step);
    }
    history
}

fn gen_pred(rng: &mut Rng, next_id: i64, columns: &[Col]) -> Pred {
    let id = 1 + rng.below(next_id.max(2) as usize) as i64;
    match rng.below(6) {
        0 => Pred::IdEq(id),
        1 => Pred::IdLt(id),
        2 => Pred::IdGt(id),
        3 => Pred::IdIn(
            (0..3)
                .map(|_| 1 + rng.below(next_id.max(2) as usize) as i64)
                .collect(),
        ),
        4 => Pred::IsNull(rng.pick(columns).name.clone()),
        _ => {
            let comparable = columns
                .iter()
                .filter(|c| c.ty.comparable() && !c.ty.is_float())
                .collect::<Vec<_>>();
            if comparable.is_empty() {
                Pred::IdEq(id)
            } else {
                let column = *rng.pick(&comparable);
                // The 1.5.5 oracle fails a filter on a string holding a NUL
                // byte, fixed on upstream main, not in 1.5.6
                // (upstream-bugs/nul-string-filter-parser-error.sql).
                let value = gen_value(rng, column.ty);
                if value == V::Null || matches!(&value, V::S(s) if s.contains('\0')) {
                    return Pred::IsNull(column.name.clone());
                }
                Pred::Eq(column.name.clone(), column.ty, value)
            }
        },
    }
}

fn render_pred(pred: &Pred, literal: fn(Ty, &V) -> String) -> String {
    match pred {
        Pred::IdEq(id) => format!("id = {id}"),
        Pred::IdLt(id) => format!("id < {id}"),
        Pred::IdGt(id) => format!("id > {id}"),
        Pred::IdIn(ids) => format!(
            "id IN ({})",
            ids.iter()
                .map(i64::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Pred::IsNull(column) => format!("{column} IS NULL"),
        Pred::Eq(column, ty, value) => format!("{column} = {}", literal(*ty, value)),
    }
}

fn transform_sql(column: &str, transform: Transform) -> String {
    match transform {
        Transform::Identity => column.to_string(),
        Transform::Year => format!("year({column})"),
        Transform::Month => format!("month({column})"),
        Transform::Day => format!("day({column})"),
    }
}

// ---------------------------------------------------------------------------
// The official side
// ---------------------------------------------------------------------------

/// A DuckDB connection with DuckLake and the SQLite scanner loaded, in UTC.
fn duckdb_session() -> duckdb::Connection {
    crate::common::ensure_ducklake_installed();
    crate::common::ensure_extension_installed("sqlite");
    // Every step opens a session; skipping the signature check of the
    // installed extensions keeps their load from hashing them each time.
    let config = duckdb::Config::default()
        .allow_unsigned_extensions()
        .unwrap();
    let connection = duckdb::Connection::open_in_memory_with_flags(config).unwrap();
    connection
        .execute_batch("LOAD ducklake; LOAD sqlite; SET TimeZone = 'UTC';")
        .unwrap();
    connection
}

/// Official DuckLake's catalog: a DuckDB metadata file and its data, attached
/// as `lake` on a connection that each step opens afresh. A session that read
/// a table caches its column types: one that read it before a column was
/// widened drops the column's bounds on its next insert in 1.5.5, fixed in
/// 1.5.6 (upstream-bugs/widened-column-stats-dropped-then-narrowed.sql), so
/// the oracle starts every step as a new client would.
struct Official {
    metadata: PathBuf,
    data: PathBuf,
    connection: duckdb::Connection,
}

impl Official {
    fn new(metadata: PathBuf, data: &Path) -> Self {
        let connection = Self::open(&metadata, data);
        Self {
            metadata,
            data: data.to_path_buf(),
            connection,
        }
    }

    fn open(metadata: &Path, data: &Path) -> duckdb::Connection {
        let connection = duckdb_session();
        connection
            .execute_batch(&format!(
                "ATTACH 'ducklake:{}' AS lake (DATA_PATH '{}/'); USE lake;",
                metadata.display(),
                data.display()
            ))
            .unwrap();
        connection
    }

    /// Start a new session on the catalog, closing the last one first: two
    /// DuckDB instances must not hold the file at once.
    fn reconnect(&mut self) {
        self.connection = duckdb::Connection::open_in_memory().unwrap();
        self.connection = Self::open(&self.metadata, &self.data);
    }

    fn run(&self, sql: &str) -> Result<(), String> {
        self.connection
            .execute_batch(sql)
            .map_err(|e| e.to_string())
    }

    /// The metadata file and its write-ahead log, and where [`Self::save`]
    /// keeps a copy of each.
    fn files(&self) -> [(PathBuf, PathBuf); 2] {
        let wal = PathBuf::from(format!("{}.wal", self.metadata.display()));
        let saved = |path: &Path| PathBuf::from(format!("{}.saved", path.display()));
        [(self.metadata.clone(), saved(&self.metadata)), (wal.clone(), saved(&wal))]
    }

    /// Keep a copy of the catalog's metadata as it is now, for
    /// [`Self::restore`]. Data files a later step writes stay behind
    /// unreferenced.
    fn save(&mut self) {
        self.connection = duckdb::Connection::open_in_memory().unwrap();
        for (file, copy) in self.files() {
            let _ = std::fs::remove_file(&copy);
            if file.exists() {
                std::fs::copy(&file, &copy).unwrap();
            }
        }
        self.connection = Self::open(&self.metadata, &self.data);
    }

    /// Put the catalog's metadata back as [`Self::save`] kept it.
    fn restore(&mut self) {
        self.connection = duckdb::Connection::open_in_memory().unwrap();
        for (file, copy) in self.files() {
            let _ = std::fs::remove_file(&file);
            if copy.exists() {
                std::fs::copy(&copy, &file).unwrap();
            }
        }
        self.connection = Self::open(&self.metadata, &self.data);
    }

    /// Run `sql` with DuckDB's filter pushdown off, so no statistics prune
    /// the files it reads.
    fn unpruned(&self, sql: &str) -> Result<(), String> {
        self.run("SET disabled_optimizers = 'filter_pushdown'")?;
        let result = self.run(sql);
        self.run("RESET disabled_optimizers")?;
        result
    }

    fn head(&self) -> i64 {
        let mut statement = self
            .connection
            .prepare("SELECT max(snapshot_id) FROM ducklake_snapshots('lake')")
            .unwrap();
        statement.query_row([], |row| row.get(0)).unwrap()
    }

    fn rows(
        connection: &duckdb::Connection,
        sql: &str,
    ) -> Result<(Vec<String>, Vec<Vec<String>>), String> {
        let mut statement = connection.prepare(sql).map_err(|e| e.to_string())?;
        let batches = statement
            .query_arrow([])
            .map_err(|e| e.to_string())?
            .collect::<Vec<_>>();
        Ok(canon_duckdb::rows(&batches))
    }

    fn apply(&self, history: &History, step: usize) -> Result<(), String> {
        let columns = history.columns_at(step);
        match &history.steps[step] {
            Step::Insert(rows) => {
                let values = rows
                    .iter()
                    .map(|(id, values)| {
                        let mut cells = vec![format!("CAST({id} AS BIGINT)")];
                        cells.extend(
                            columns
                                .iter()
                                .zip(values)
                                .map(|(c, v)| duckdb_literal(c.ty, v)),
                        );
                        format!("({})", cells.join(", "))
                    })
                    .collect::<Vec<_>>();
                self.run(&format!("INSERT INTO t VALUES {}", values.join(", ")))
            },
            // Official's `UPDATE` and `DELETE` run without filter pushdown:
            // where its statistics are known to be wrong, its file listing
            // would skip files holding rows that match, and the oracle's rows
            // would part from the right ones.
            Step::Update {
                column,
                ty,
                value,
                pred,
            } => self.unpruned(&format!(
                "UPDATE t SET {column} = {} WHERE {}",
                duckdb_literal(*ty, value),
                render_pred(pred, duckdb_literal)
            )),
            Step::Delete(None) => self.run("DELETE FROM t"),
            Step::Delete(Some(pred)) => self.unpruned(&format!(
                "DELETE FROM t WHERE {}",
                render_pred(pred, duckdb_literal)
            )),
            Step::Flush => self.run("CALL ducklake_flush_inlined_data('lake')"),
            Step::Checkpoint => self.run("CHECKPOINT lake"),
            Step::Merge => self.run("CALL ducklake_merge_adjacent_files('lake')"),
            Step::Rewrite => {
                self.run("CALL ducklake_rewrite_data_files('lake', 't', delete_threshold => 0.0)")
            },
            Step::AddColumn(column, default) => self.run(&format!(
                "ALTER TABLE t ADD COLUMN {} {}{}",
                column.name,
                column.ty.duckdb(),
                default
                    .as_ref()
                    .map(|d| format!(
                        " DEFAULT {}",
                        if column.ty == Ty::Varchar {
                            quote(d)
                        } else {
                            d.clone()
                        }
                    ))
                    .unwrap_or_default()
            )),
            Step::Widen(column) => self.run(&format!(
                "ALTER TABLE t ALTER COLUMN {column} SET DATA TYPE BIGINT"
            )),
            Step::Limit(limit) => self.run(&format!(
                "CALL lake.set_option('data_inlining_row_limit', {limit}, table_name => 't')"
            )),
            Step::Partition(None) => self.run("ALTER TABLE t RESET PARTITIONED BY"),
            Step::Partition(Some((column, transform))) => self.run(&format!(
                "ALTER TABLE t SET PARTITIONED BY ({})",
                transform_sql(column, *transform)
            )),
            Step::Expire => self.run("CALL ducklake_expire_snapshots('lake', older_than => now())"),
        }
    }
}

// ---------------------------------------------------------------------------
// The crate side
// ---------------------------------------------------------------------------

type Open = Arc<
    dyn Fn() -> futures::future::BoxFuture<
            'static,
            (Arc<dyn MetadataWriter>, Arc<dyn MetadataProvider>),
        > + Send
        + Sync,
>;

/// This crate's catalog on the backend under test.
struct Lake {
    kind: Kind,
    open: Open,
    data: PathBuf,
    /// The metadata database, where DuckDB can attach it.
    metadata: Option<PathBuf>,
    /// A metadata server DuckDB can read the catalog tables of, though not
    /// attach as a DuckLake: MySQL and single-catalog PostgreSQL.
    remote: Option<Metadata>,
}

fn sql_type(ty: Ty) -> &'static str {
    match ty {
        Ty::Int => "int32",
        Ty::BigInt => "int64",
        Ty::Varchar => "varchar",
        _ => "",
    }
}

impl Lake {
    /// A session over the catalog, at `snapshot` (else the latest), with the
    /// `rowid` column when `lineage` is set; writes take a session without it.
    async fn session(
        &self,
        snapshot: Option<i64>,
        lineage: bool,
    ) -> (
        SessionContext,
        DuckLakeCatalog,
        Arc<dyn MetadataWriter>,
        Arc<dyn MetadataProvider>,
    ) {
        let (writer, provider) = (self.open)().await;
        let ctx = SessionContext::new();
        let make = || {
            match snapshot {
                None => DuckLakeCatalog::with_writer(Arc::clone(&provider), Arc::clone(&writer))
                    .unwrap(),
                Some(snapshot) => {
                    DuckLakeCatalog::with_snapshot(Arc::clone(&provider), snapshot).unwrap()
                },
            }
            .with_row_lineage(lineage)
        };
        DuckLakeCatalog::register(&ctx, "lake", make());
        let catalog = make();
        register_ducklake_functions(&ctx, Arc::clone(&provider));
        (ctx, catalog, writer, provider)
    }

    async fn sql(&self, statement: &str) -> Result<(), String> {
        let (ctx, catalog, _, _) = self.session(None, false).await;
        execute_ducklake_sql(&ctx, &catalog, statement)
            .await
            .map_err(|e| e.to_string())?
            .collect()
            .await
            .map(|_| ())
            .map_err(|e| e.to_string())
    }

    async fn rows(
        &self,
        snapshot: Option<i64>,
        sql: &str,
    ) -> Result<(Vec<String>, Vec<Vec<String>>), String> {
        let (ctx, _, _, _) = self.session(snapshot, true).await;
        let batches = ctx
            .sql(sql)
            .await
            .map_err(|e| e.to_string())?
            .collect()
            .await
            .map_err(|e| e.to_string())?;
        Ok(canon_crate::rows(&batches))
    }

    async fn head(&self) -> i64 {
        let (_, provider) = (self.open)().await;
        provider.get_current_snapshot().unwrap()
    }

    async fn table(&self) -> (SessionContext, DuckLakeTable) {
        let (ctx, _, _, _) = self.session(None, false).await;
        let provider = ctx
            .catalog("lake")
            .unwrap()
            .schema("main")
            .unwrap()
            .table("t")
            .await
            .unwrap()
            .unwrap();
        let table = (provider.as_ref() as &dyn std::any::Any)
            .downcast_ref::<DuckLakeTable>()
            .expect("a DuckLakeTable")
            .clone();
        (ctx, table)
    }

    async fn table_id(&self) -> Option<i64> {
        let (_, provider) = (self.open)().await;
        let snapshot = provider.get_current_snapshot().unwrap();
        let schema = provider.get_schema_by_name("main", snapshot).unwrap()?;
        provider
            .get_table_by_name(schema.schema_id, "t", snapshot)
            .unwrap()
            .map(|t| t.table_id)
    }

    fn column_defs(columns: &[Col]) -> Vec<ColumnDef> {
        let mut defs = vec![ColumnDef::from_arrow("id", &DataType::Int64, true).unwrap()];
        defs.extend(
            columns
                .iter()
                .map(|c| ColumnDef::from_arrow(&c.name, &c.ty.arrow(), true).unwrap()),
        );
        defs
    }

    async fn create(&self, history: &History) -> Result<(), String> {
        let (writer, _) = (self.open)().await;
        let columns = Self::column_defs(&history.columns);
        let setup = writer
            .begin_write_transaction("main", "t", &columns, WriteMode::Replace)
            .map_err(|e| e.to_string())?;
        writer
            .publish_snapshot(
                setup.table_id,
                "main",
                "t",
                setup.snapshot_id,
                WriteMode::Replace,
                setup.base_snapshot_id,
                &columns,
                &setup.field_ids,
            )
            .map_err(|e| e.to_string())?;
        if let Some(limit) = history.limit {
            writer
                .set_table_setting(
                    setup.table_id,
                    "data_inlining_row_limit",
                    &limit.to_string(),
                )
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    async fn apply(&self, history: &History, step: usize) -> Result<(), String> {
        let columns = history.columns_at(step);
        match &history.steps[step] {
            Step::Insert(rows) => {
                let values = rows
                    .iter()
                    .map(|(id, values)| {
                        let mut cells = vec![format!("CAST({id} AS BIGINT)")];
                        cells.extend(
                            columns
                                .iter()
                                .zip(values)
                                .map(|(c, v)| datafusion_literal(c.ty, v)),
                        );
                        format!("({})", cells.join(", "))
                    })
                    .collect::<Vec<_>>();
                self.sql(&format!(
                    "INSERT INTO lake.main.t VALUES {}",
                    values.join(", ")
                ))
                .await
            },
            Step::Update {
                column,
                ty,
                value,
                pred,
            } => {
                self.sql(&format!(
                    "UPDATE lake.main.t SET {column} = {} WHERE {}",
                    datafusion_literal(*ty, value),
                    render_pred(pred, datafusion_literal)
                ))
                .await
            },
            Step::Delete(None) => self.sql("DELETE FROM lake.main.t").await,
            Step::Delete(Some(pred)) => {
                self.sql(&format!(
                    "DELETE FROM lake.main.t WHERE {}",
                    render_pred(pred, datafusion_literal)
                ))
                .await
            },
            Step::Flush => self.sql("CALL ducklake_flush_inlined_data('lake')").await,
            Step::Checkpoint => {
                let (ctx, catalog, _, _) = self.session(None, false).await;
                catalog
                    .checkpoint(&ctx.state())
                    .await
                    .map(|_| ())
                    .map_err(|e| e.to_string())
            },
            Step::Merge => {
                let (ctx, table) = self.table().await;
                // Official merges up to the table's `target_file_size`, which
                // the harness leaves at its default.
                let target = datafusion_ducklake::table_writer::DEFAULT_TARGET_FILE_SIZE as u64;
                table
                    .merge_adjacent_files(
                        &ctx.state(),
                        MergeOptions {
                            target_file_size: target,
                            max_merged_files: usize::MAX,
                            min_file_size: 0,
                        },
                    )
                    .await
                    .map(|_| ())
                    .map_err(|e| e.to_string())
            },
            Step::Rewrite => {
                let (ctx, table) = self.table().await;
                table
                    .rewrite_data_files(
                        &ctx.state(),
                        RewriteOptions {
                            delete_threshold: 0.0,
                            data_file_ids: None,
                        },
                    )
                    .await
                    .map(|_| ())
                    .map_err(|e| e.to_string())
            },
            Step::AddColumn(column, default) => {
                let (writer, _) = (self.open)().await;
                let mut defs = Self::column_defs(&columns);
                let mut def = ColumnDef::from_arrow(&column.name, &column.ty.arrow(), true)
                    .map_err(|e| e.to_string())?;
                if let Some(default) = default {
                    let _ = sql_type(column.ty);
                    def = def
                        .with_default(default.clone())
                        .map_err(|e| e.to_string())?;
                }
                defs.push(def);
                let setup = writer
                    .begin_write_transaction("main", "t", &defs, WriteMode::Append)
                    .map_err(|e| e.to_string())?;
                writer
                    .publish_snapshot(
                        setup.table_id,
                        "main",
                        "t",
                        setup.snapshot_id,
                        WriteMode::Append,
                        setup.base_snapshot_id,
                        &defs,
                        &setup.field_ids,
                    )
                    .map(|_| ())
                    .map_err(|e| e.to_string())
            },
            Step::Widen(column) => {
                let (writer, _) = (self.open)().await;
                let table_id = self.table_id().await.ok_or("no table")?;
                writer
                    .promote_column_type(table_id, column, "int64")
                    .map(|_| ())
                    .map_err(|e| e.to_string())
            },
            Step::Limit(limit) => {
                let (writer, _) = (self.open)().await;
                let table_id = self.table_id().await.ok_or("no table")?;
                writer
                    .set_table_setting(table_id, "data_inlining_row_limit", &limit.to_string())
                    .map_err(|e| e.to_string())
            },
            Step::Partition(None) => {
                self.sql("ALTER TABLE lake.main.t RESET PARTITIONED BY")
                    .await
            },
            Step::Partition(Some((column, transform))) => {
                self.sql(&format!(
                    "ALTER TABLE lake.main.t SET PARTITIONED BY ({})",
                    transform_sql(column, *transform)
                ))
                .await
            },
            Step::Expire => {
                let (writer, _) = (self.open)().await;
                writer
                    .maintenance()
                    .ok_or("no maintenance")?
                    .expire_snapshots(datafusion_ducklake::maintenance::ExpireCriteria::OlderThan(
                        chrono::Utc::now(),
                    ))
                    .await
                    .map(|_| ())
                    .map_err(|e| e.to_string())
            },
        }
    }

    /// A copy of the crate's metadata database DuckDB can attach.
    async fn duckdb_copy(&self, dir: &Path) -> Option<Metadata> {
        let source = self.metadata.as_ref()?;
        match self.kind {
            Kind::Sqlite => {
                let copy = dir.join("crate_copy.db");
                let _ = std::fs::remove_file(&copy);
                // Fold the write-ahead log into the copy.
                let pool = sqlx::SqlitePool::connect(&format!("sqlite:{}", source.display()))
                    .await
                    .unwrap();
                sqlx::query(sqlx::AssertSqlSafe(format!(
                    "VACUUM INTO '{}'",
                    copy.display()
                )))
                .execute(&pool)
                .await
                .unwrap();
                pool.close().await;
                crate::inlined_mutation_differential_tests::make_sqlite_catalog_duckdb_readable(
                    &copy,
                )
                .await;
                Some(Metadata::Sqlite(copy))
            },
            Kind::Duckdb => {
                let copy = dir.join("crate_copy.duckdb");
                let _ = std::fs::remove_file(&copy);
                let _ = std::fs::remove_file(dir.join("crate_copy.duckdb.wal"));
                let connection = duckdb::Connection::open_in_memory().unwrap();
                connection
                    .execute_batch(&format!(
                        "ATTACH '{}' AS src (READ_ONLY); ATTACH '{}' AS dst; \
                         COPY FROM DATABASE src TO dst;",
                        source.display(),
                        copy.display()
                    ))
                    .unwrap();
                Some(Metadata::Duckdb(copy))
            },
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Comparing
// ---------------------------------------------------------------------------

/// A difference between this crate and official DuckLake.
#[derive(Debug, Clone)]
struct Mismatch {
    /// The kind of comparison that differed; shrinking keeps it.
    kind: &'static str,
    step: usize,
    detail: String,
}

/// The snapshot each side's catalog was at after each step (and after the
/// table was created, at index 0).
#[derive(Default)]
struct Heads {
    official: Vec<i64>,
    ours: Vec<i64>,
}

impl Heads {
    /// The step (0 for the creation) whose commit made `snapshot` on a side.
    fn step_of(heads: &[i64], snapshot: i64) -> usize {
        heads
            .iter()
            .position(|head| *head >= snapshot)
            .unwrap_or(heads.len())
    }
}

/// Sort `rows` and map snapshot-id cells (the column named `snapshot_id`) to
/// the step that made them.
fn normalize(
    (names, mut rows): (Vec<String>, Vec<Vec<String>>),
    heads: &[i64],
) -> (Vec<String>, Vec<Vec<String>>) {
    if let Some(index) = names.iter().position(|name| name == "snapshot_id") {
        for row in &mut rows {
            if let Ok(snapshot) = row[index].parse::<i64>() {
                row[index] = format!("step{}", Heads::step_of(heads, snapshot));
            }
        }
    }
    rows.sort();
    (names, rows)
}

fn describe(rows: &(Vec<String>, Vec<Vec<String>>)) -> String {
    let mut text = rows.0.join(" | ");
    for row in &rows.1 {
        let _ = write!(text, "\n  {}", row.join(" | "));
    }
    text
}

/// How often each comparison ran and was skipped, and why, per backend; the
/// soak runs print it.
mod coverage {
    use std::collections::BTreeMap;
    use std::sync::Mutex;

    static COUNTS: Mutex<BTreeMap<String, u64>> = Mutex::new(BTreeMap::new());

    fn count(key: String) {
        *COUNTS.lock().unwrap().entry(key).or_default() += 1;
    }

    pub(super) fn ran(kind: super::Kind, comparison: &str) {
        count(format!("{kind:?}|{comparison}|ran"));
    }

    pub(super) fn skipped(kind: super::Kind, comparison: &str, reason: &str) {
        count(format!("{kind:?}|{comparison}|skipped: {reason}"));
    }

    /// One line per comparison: how often it ran, how often it was skipped
    /// and why.
    pub(super) fn summary(kind: super::Kind) -> String {
        let counts = COUNTS.lock().unwrap();
        let prefix = format!("{kind:?}|");
        // comparison -> (runs, [(skip reason, count)])
        type Counts<'a> = BTreeMap<&'a str, (u64, Vec<(&'a str, u64)>)>;
        let mut by_comparison: Counts<'_> = BTreeMap::new();
        for (key, n) in counts.iter() {
            let Some(rest) = key.strip_prefix(&prefix) else {
                continue;
            };
            let (comparison, what) = rest.split_once('|').unwrap();
            let entry = by_comparison.entry(comparison).or_default();
            if what == "ran" {
                entry.0 += n;
            } else {
                entry.1.push((what, *n));
            }
        }
        let mut text = format!("coverage on {kind:?}:");
        for (comparison, (ran, skips)) in by_comparison {
            let skipped: u64 = skips.iter().map(|(_, n)| n).sum();
            let total = ran + skipped;
            let _ = std::fmt::Write::write_fmt(
                &mut text,
                format_args!(
                    "\n  {comparison}: {ran} run, {skipped} skipped ({:.1}%)",
                    if total == 0 {
                        0.0
                    } else {
                        100.0 * skipped as f64 / total as f64
                    }
                ),
            );
            for (reason, n) in skips {
                let _ = std::fmt::Write::write_fmt(&mut text, format_args!("\n    {reason}: {n}"));
            }
        }
        text
    }
}

/// Upstream data bugs this crate does not reproduce, each recognized by the
/// exact cells it affects: a column, a snapshot, a change-feed window, or a
/// query's constant. None of them switches a comparison off for the rest of a
/// history.
mod known {
    use super::*;

    /// A string or blob bound official cannot store: one holding a NUL
    /// byte, or a Parquet maximum over 256 bytes whose first 256 have none
    /// DuckDB's writer can increment. Official stores a statistics row with
    /// neither bound, which a later write narrows to its own bounds in the
    /// 1.5.5 oracle (1.5.6 keeps it unbounded); this crate stores none (see
    /// COMPATIBILITY.md). Repros:
    /// `upstream-bugs/nul-string-filter-parser-error.sql`,
    /// `upstream-bugs/untruncatable-string-bound-narrowed.sql`.
    pub(super) fn unstored_value(v: &V) -> bool {
        match v {
            V::S(s) => {
                s.contains('\0')
                    || (s.len() > 256 && s.as_bytes()[..256].iter().all(|b| *b >= 0x7F))
            },
            V::Bytes(b) => b.len() > 256 && b[..256].iter().all(|b| *b == 0xFF),
            _ => false,
        }
    }

    /// [`unstored_value`] for the values written to `column` up to `step`.
    pub(super) fn unstored_column(history: &History, step: usize, column: &Col) -> bool {
        written(history, step, &column.name)
            .iter()
            .any(unstored_value)
    }

    /// The values of `column` written up to `step`.
    pub(super) fn written(history: &History, step: usize, column: &str) -> Vec<V> {
        let mut values = Vec::new();
        for (index, s) in history.steps[..=step].iter().enumerate() {
            match s {
                Step::Insert(rows) => {
                    let columns = history.columns_at(index);
                    for (_, row) in rows {
                        for (c, v) in columns.iter().zip(row) {
                            if c.name == column {
                                values.push(v.clone());
                            }
                        }
                    }
                },
                Step::Update {
                    column: c,
                    value,
                    ..
                } if c == column => values.push(value.clone()),
                _ => {},
            }
        }
        values
    }

    /// The values of `column` written before the state at heads index `at`
    /// (0 is the table's creation).
    pub(super) fn written_before(history: &History, at: usize, column: &str) -> Vec<V> {
        match at.checked_sub(1) {
            Some(step) => written(history, step, column),
            None => Vec::new(),
        }
    }

    /// The Parquet footers this crate writes leave NaN out of a float
    /// column's min/max, so DuckDB prunes NaN rows of them (a separate open
    /// issue), and record a largest value of `-0.0` as `0.0`, as the Parquet
    /// specification has writers do: a float column that ever held NaN or
    /// `-0.0`. Affects DuckDB's reads over this crate's catalog only.
    pub(super) fn float_special(history: &History, step: usize, column: &Col) -> bool {
        column.ty.is_float()
            && written(history, step, &column.name)
                .iter()
                .any(|v| match v {
                    V::F(f) => f.is_nan() || (*f == 0.0 && f.is_sign_negative()),
                    V::R(f) => f.is_nan() || (*f == 0.0 && f.is_sign_negative()),
                    _ => false,
                })
    }

    /// Official loses a column's bounds when one commit writes a file whose
    /// values of it are all NULL before one that has values: a partitioned
    /// write of several files (an insert past the limit, or a flush) of a
    /// column that ever held NULL. A 1.5.5 bug, fixed on upstream main (1.5.6
    /// still drops the bounds). Repro:
    /// `upstream-bugs/partitioned-insert-null-file-drops-bounds.sql`.
    pub(super) fn partitioned_null_file(history: &History, step: usize, column: &Col) -> bool {
        let mut partitioned = false;
        let mut limit = history.limit.unwrap_or(10);
        let mut multi_file = false;
        for s in &history.steps[..=step] {
            match s {
                Step::Partition(spec) => partitioned = spec.is_some(),
                Step::Limit(l) => limit = *l,
                Step::Insert(rows) if partitioned && rows.len() > limit => multi_file = true,
                Step::Flush
                | Step::Checkpoint
                | Step::Update {
                    ..
                } if partitioned => multi_file = true,
                _ => {},
            }
        }
        let values = written(history, step, &column.name);
        // A nested column's leaves can be NULL below a value that is not.
        let nested = matches!(column.ty, Ty::Struct | Ty::Map | Ty::List);
        multi_file && (nested || values.contains(&V::Null)) && values.iter().any(|v| *v != V::Null)
    }

    /// The 1.5.5 oracle drops a widened column's bounds when it next merges
    /// statistics into them (a flush after the widening, in any session) and
    /// narrows them on the write after; 1.5.6 keeps them. Repro:
    /// `upstream-bugs/widened-column-stats-dropped-then-narrowed.sql`.
    pub(super) fn widened(history: &History, step: usize, column: &Col) -> bool {
        history.steps[..=step]
            .iter()
            .any(|s| matches!(s, Step::Widen(name) if *name == column.name))
    }

    /// Which modelled upstream bug explains official's table-wide statistics
    /// of `column` no longer covering the values it holds after `step`.
    pub(super) fn lost_bounds(
        history: &History,
        step: usize,
        column: &Col,
    ) -> Option<&'static str> {
        if partitioned_null_file(history, step, column) {
            Some("official stats: partitioned NULL file")
        } else if unstored_column(history, step, column) {
            Some("official stats: unstored bound")
        } else if widened(history, step, column) {
            Some("official stats: widened column")
        } else if inlined_double_text(history, step + 1, column) {
            Some("official stats: rounded inlined DOUBLE")
        } else if column.ty.is_float()
            && written(history, step, &column.name)
                .iter()
                .any(|v| matches!(v, V::F(f) if f.is_nan()) || matches!(v, V::R(f) if f.is_nan()))
        {
            // A write whose values of a float are NaN or NULL has no bounds;
            // official stores the row with neither, which it reads as holding
            // no value. This crate stores none (see COMPATIBILITY.md).
            Some("official stats: NaN-only write")
        } else {
            None
        }
    }

    /// Whether `column` ever held `-0.0` up to heads index `at`. DuckDB's
    /// Parquet writer dictionary-encodes `-0.0` and `0.0` as one value, so
    /// official reads `-0.0` back as `0.0` from a file it wrote with both
    /// (duckdb/duckdb#25781), and DataFusion orders `-0.0` below `0.0` where
    /// DuckDB compares them equal: such a column's zeros compare without
    /// their sign. Repro: `upstream-bugs/parquet-dictionary-negative-zero.sql`.
    pub(super) fn negative_zero(history: &History, at: usize, column: &Col) -> bool {
        column.ty.is_float()
            && written_before(history, at, &column.name)
                .iter()
                .any(|v| match v {
                    V::F(f) => *f == 0.0 && f.is_sign_negative(),
                    V::R(f) => *f == 0.0 && f.is_sign_negative(),
                    _ => false,
                })
    }

    /// The sibling of duckdb/ducklake#1571 (see COMPATIBILITY.md): over a
    /// window that holds a rewrite, official's change feed reports each
    /// deletion and update row of an earlier step of the window twice. This
    /// crate reports each once: such a row repeated exactly twice is taken
    /// once. Repro: `upstream-bugs/m9-repro.sql`.
    pub(super) fn rewrite_repeats(
        history: &History,
        window: (usize, usize),
        (names, rows): (Vec<String>, Vec<Vec<String>>),
    ) -> ((Vec<String>, Vec<Vec<String>>), bool) {
        let Some(rewrite) = (window.0..window.1)
            .rev()
            .find(|step| matches!(history.steps[*step], Step::Rewrite | Step::Checkpoint))
        else {
            return ((names, rows), false);
        };
        let mut kept: Vec<Vec<String>> = Vec::with_capacity(rows.len());
        let mut index = 0;
        let mut fired = false;
        while index < rows.len() {
            let row = &rows[index];
            let earlier = row[0]
                .strip_prefix("step")
                .and_then(|n| n.parse::<usize>().ok())
                .is_some_and(|n| n <= rewrite);
            let mutation = row
                .get(2)
                .is_some_and(|kind| kind.contains("delete") || kind.contains("update"));
            if earlier
                && mutation
                && rows.get(index + 1) == Some(row)
                && rows.get(index + 2) != Some(row)
            {
                kept.push(row.clone());
                index += 2;
                fired = true;
            } else {
                kept.push(row.clone());
                index += 1;
            }
        }
        ((names, kept), fired)
    }

    /// Official on DuckDB metadata stores an inlined `DOUBLE` through its
    /// shortest text, which DuckDB reads back as a `DECIMAL` when it has no
    /// exponent; past 15 significant digits that lands on a neighbouring
    /// double (`0.48024702758086213` reads back `0.4802470275808622`). Repro:
    /// `upstream-bugs/inlined-double-rounded.sql`. Official's Parquet writes
    /// keep the value. The canonical cells ([`duckdb_decimal_double`]) of the
    /// values of `column` official may hold rounded at heads index `at`: those
    /// an insert of no more rows than the limit wrote, and, once an update
    /// ran under a limit above `0`, every value held before it (the update
    /// inlines its rows' new versions whole). Those cells compare as official
    /// rounds them.
    pub(super) fn inlined_double_keys(history: &History, at: usize, column: &Col) -> Vec<String> {
        if column.ty != Ty::Double {
            return Vec::new();
        }
        let mut limit = history.limit.unwrap_or(10);
        // Every value of the column written so far, and those official may
        // hold rounded.
        let mut held: Vec<&V> = Vec::new();
        let mut rounded: Vec<&V> = Vec::new();
        for (index, step) in history.steps[..at.min(history.steps.len())]
            .iter()
            .enumerate()
        {
            match step {
                Step::Limit(n) => limit = *n,
                Step::Insert(rows) => {
                    let columns = history.columns_at(index);
                    let values: Vec<&V> = rows
                        .iter()
                        .flat_map(|(_, row)| {
                            columns
                                .iter()
                                .zip(row)
                                .filter(|(c, _)| c.name == column.name)
                        })
                        .map(|(_, v)| v)
                        .collect();
                    if limit > 0 && rows.len() <= limit {
                        rounded.extend(&values);
                    }
                    held.extend(values);
                },
                // An update writes its rows' new versions, every column of
                // them, where the limit inlines them.
                Step::Update {
                    column: c,
                    value,
                    ..
                } => {
                    if c == &column.name {
                        held.push(value);
                    }
                    if limit > 0 {
                        rounded.extend(&held);
                    }
                },
                _ => {},
            }
        }
        let mut keys: Vec<String> = rounded
            .into_iter()
            .filter(|v| double_fixed_point_filter(Ty::Double, v))
            .filter_map(|v| match v {
                V::F(f) => Some(duckdb_decimal_double(&f.to_string())),
                _ => None,
            })
            .collect();
        keys.sort();
        keys.dedup();
        keys
    }

    /// Whether `column` holds a value [`inlined_double_keys`] names.
    pub(super) fn inlined_double_text(history: &History, at: usize, column: &Col) -> bool {
        !inlined_double_keys(history, at, column).is_empty()
    }

    /// A canonical cell of a column [`inlined_double_text`] holds, compared at
    /// 15 significant digits: official's inlined write may have rounded a
    /// value (DuckDB reads its text as a `DECIMAL` and casts it to a
    /// neighbouring double, which can print shorter), so the written value
    /// and the one official stored then read alike. Any other cell (NULL,
    /// zero, an infinity, NaN) is unchanged.
    pub(super) fn duckdb_decimal_double(cell: &str) -> String {
        match cell.parse::<f64>() {
            // Every finite non-zero double of the column, whatever its
            // digits: official's rounded value can print shorter.
            Ok(value) if value.is_finite() && value != 0.0 => format!("{value:.14e}"),
            _ => cell.to_string(),
        }
    }

    /// Official numbers the rows of rowid-adjacent files a merge combines by
    /// their position in the merged file (duckdb/ducklake#1605, followed by
    /// this crate), and the rows an `UPDATE` wrote to one file come in the
    /// order official's scan produced them, which DuckDB does not fix: after a
    /// merge of an update's rows, two catalogs can give the same rows each
    /// other's row ids. Whether a merge (or checkpoint) up to `step` followed
    /// an update; only the updated rows may then differ in their row ids
    /// ([`row_ids_traded`]), every other row still compares with its row id.
    pub(super) fn merged_update_order(history: &History, step: usize) -> bool {
        merged_update_ids(history, step).is_some()
    }

    /// The `id`s of the rows an update a merge (or checkpoint) followed up to
    /// `step` may have written: `None` without such an update, `Some(None)`
    /// when a predicate on a value column leaves them unknown (any row).
    pub(super) fn merged_update_ids(
        history: &History,
        step: usize,
    ) -> Option<Option<std::collections::BTreeSet<i64>>> {
        let mut inserted = std::collections::BTreeSet::new();
        let mut pending: Option<std::collections::BTreeSet<i64>> = Some(Default::default());
        let mut updated = false;
        let mut merged: Option<Option<std::collections::BTreeSet<i64>>> = None;
        for s in &history.steps[..=step] {
            match s {
                Step::Insert(rows) => inserted.extend(rows.iter().map(|(id, _)| *id)),
                Step::Update {
                    pred,
                    ..
                } => {
                    updated = true;
                    let ids: Option<Vec<i64>> = match pred {
                        Pred::IdEq(id) => Some(vec![*id]),
                        Pred::IdIn(ids) => Some(ids.clone()),
                        Pred::IdLt(n) => {
                            Some(inserted.iter().copied().filter(|id| id < n).collect())
                        },
                        Pred::IdGt(n) => {
                            Some(inserted.iter().copied().filter(|id| id > n).collect())
                        },
                        Pred::IsNull(_) | Pred::Eq(..) => None,
                    };
                    pending = match (pending, ids) {
                        (Some(mut set), Some(ids)) => {
                            set.extend(ids);
                            Some(set)
                        },
                        _ => None,
                    };
                },
                Step::Merge | Step::Checkpoint if updated => {
                    merged = Some(match (merged, &pending) {
                        (Some(None), _) | (_, None) => None,
                        (Some(Some(mut set)), Some(ids)) => {
                            set.extend(ids.iter().copied());
                            Some(set)
                        },
                        (None, Some(ids)) => Some(ids.clone()),
                    });
                },
                _ => {},
            }
        }
        merged
    }

    /// The rows of [`merged_update_order`]: whether `official` and `ours`
    /// differ only in the row ids of rows whose `id` is in `updated` (any
    /// row for `None`). Every other row must match with its row id, and the
    /// rows that differ must match without them.
    pub(super) fn row_ids_traded(
        official: &(Vec<String>, Vec<Vec<String>>),
        ours: &(Vec<String>, Vec<Vec<String>>),
        updated: &Option<std::collections::BTreeSet<i64>>,
    ) -> bool {
        if official.0 != ours.0 {
            return false;
        }
        let Some(rowid) = official.0.iter().position(|name| name == "rowid") else {
            return false;
        };
        let id = official.0.iter().position(|name| name == "id");
        let mut theirs: Vec<Vec<String>> = official.1.clone();
        let mut left: Vec<Vec<String>> = Vec::new();
        for row in &ours.1 {
            match theirs.iter().position(|other| other == row) {
                Some(index) => {
                    theirs.swap_remove(index);
                },
                None => left.push(row.clone()),
            }
        }
        if left.is_empty() || theirs.len() != left.len() {
            return false;
        }
        let affected = |row: &Vec<String>| match updated {
            None => true,
            Some(ids) => id
                .and_then(|i| row[i].parse::<i64>().ok())
                .is_some_and(|value| ids.contains(&value)),
        };
        if !theirs.iter().chain(&left).all(affected) {
            return false;
        }
        let strip = |rows: &[Vec<String>]| {
            let mut rest: Vec<Vec<String>> = rows
                .iter()
                .map(|r| {
                    let mut r = r.clone();
                    r.remove(rowid);
                    r
                })
                .collect();
            rest.sort();
            rest
        };
        strip(&theirs) == strip(&left)
    }

    /// The #1571 family (see COMPATIBILITY.md): official's change feed can
    /// report rows at the snapshot of a rewrite (or a merge, flush or
    /// checkpoint), which changes no row: earlier deletions and updates of a
    /// retired file again, or its other rows as identical update pairs. This
    /// crate reports some of them as official does, and none at others. At a
    /// maintenance step where this crate reports no row, official's rows are
    /// dropped; any row this crate reports at such a step still compares, so
    /// a row it reports there must be official's. Returns official's rows and
    /// the number dropped. Repro:
    /// `upstream-bugs/rewrite-after-inlined-update-identical-pairs.sql`.
    pub(super) fn maintenance_rows(
        history: &History,
        (names, official): (Vec<String>, Vec<Vec<String>>),
        ours: &[Vec<String>],
    ) -> ((Vec<String>, Vec<Vec<String>>), usize) {
        let maintenance = |cell: &str| {
            cell.strip_prefix("step")
                .and_then(|n| n.parse::<usize>().ok())
                .and_then(|n| n.checked_sub(1))
                .is_some_and(|s| {
                    matches!(
                        history.steps.get(s),
                        Some(Step::Rewrite | Step::Checkpoint | Step::Merge | Step::Flush)
                    )
                })
        };
        let before = official.len();
        let kept: Vec<Vec<String>> = official
            .into_iter()
            .filter(|row| !maintenance(&row[0]) || ours.iter().any(|c| c[0] == row[0]))
            .collect();
        let dropped = before - kept.len();
        ((names, kept), dropped)
    }

    /// duckdb/ducklake#1604: official's file listing reads a `FLOAT` bound
    /// as a `FLOAT` and the constant's text, when it carries an exponent
    /// (`1e+20`, `3.4028235e+38`), as a `DOUBLE`, and compares them as
    /// doubles, so it can drop the file holding the rows a filter at that
    /// constant matches. Whether official's answer to a filter on a column of
    /// `ty` at `probe` can be wrong that way.
    pub(super) fn float_exponent_filter(ty: Ty, probe: &V) -> bool {
        // DuckDB prints a FLOAT in fixed notation for magnitudes in
        // [1e-4, 1e16), and with an exponent otherwise.
        ty == Ty::Float
            && matches!(probe, V::R(f) if f.is_finite() && *f != 0.0
                && !(1e-4..1e16).contains(&f.abs()))
    }

    /// Official splices a `DOUBLE` constant into its file listing as its
    /// shortest text, which is a `DECIMAL` to DuckDB when it has no exponent;
    /// past 15 significant digits DuckDB's `DECIMAL` to `DOUBLE` conversion
    /// can land on a neighbouring double, and the file holding the match is
    /// dropped. Repro: `upstream-bugs/double-fixed-point-constant-prunes-match.sql`.
    /// Whether official's answer to a filter on a column of `ty` at `probe`
    /// can be wrong that way.
    pub(super) fn double_fixed_point_filter(ty: Ty, probe: &V) -> bool {
        let V::F(f) = probe else {
            return false;
        };
        if ty != Ty::Double || !f.is_finite() || *f == 0.0 {
            return false;
        }
        let text = format!("{f:?}");
        let digits = text
            .trim_start_matches('-')
            .chars()
            .filter(char::is_ascii_digit)
            .collect::<String>();
        !text.contains('e') && digits.trim_start_matches('0').len() > 15
    }

    /// Official's rewrite recomputes the table's statistics (#1377, which
    /// this crate does not do) by aggregating every current column over each
    /// inlined data table, and fails when an older one lacks a column added
    /// since: duckdb/ducklake#1490, fixed on upstream main by
    /// duckdb/ducklake#1577, whose rewrite commits as this crate's does. Repro:
    /// `upstream-bugs/rewrite-after-add-column-inlined-aggregate-error.sql`.
    pub(super) fn stats_recompute_error(history: &History, step: usize, error: &str) -> bool {
        matches!(history.steps[step], Step::Rewrite | Step::Checkpoint)
            && error.contains("Failed to read inlined-data aggregates")
    }

    /// The same recompute reads each inlined column's `MAX(..)::VARCHAR`,
    /// which DuckDB renders for the greatest `TIMESTAMPTZ` one millisecond
    /// past it (`294247-01-10 04:00:54.776806+00`); merging that text fails
    /// its cast, and the rewrite with it. This crate's rewrite reads no
    /// inlined aggregate. Repro:
    /// `upstream-bugs/rewrite-inlined-max-timestamptz.sql`.
    pub(super) fn max_timestamptz_recompute_error(
        history: &History,
        step: usize,
        error: &str,
    ) -> bool {
        matches!(history.steps[step], Step::Rewrite | Step::Checkpoint)
            && error.contains("timestamp field value out of range")
            && error.contains("294247-01-10")
            && history.columns_at(step).iter().any(|c| c.ty == Ty::TsTz)
    }

    /// Official inserts inlined rows into the table's latest inlined data
    /// table whatever its columns' types: after a column was widened while no
    /// inlined table was created for the new schema version (the limit was
    /// `0`), a value past the old type fails official's commit on its DuckDB
    /// catalog (on a SQLite catalog the insert commits and every later read
    /// fails). This crate reads the stale table at the widened type. Repro:
    /// `upstream-bugs/widened-column-inlined-into-stale-table.sql`.
    ///
    /// After a column was added that way, official's insert supplies more
    /// values than the older table has columns (duckdb/ducklake#1606). This
    /// crate refuses that insert too, on every backend, except where MySQL
    /// writes the rows to Parquet because its inlined columns cannot hold a
    /// value of them ([`mysql_fallback`]): that write inlines nothing, so the
    /// older table plays no part in it.
    pub(super) fn stale_inlined_table(
        history: &History,
        step: usize,
        error: &str,
        kind: Kind,
    ) -> bool {
        let widened = error.contains("can't be cast because the value is out of range")
            && history.steps[..step]
                .iter()
                .any(|s| matches!(s, Step::Widen(_)));
        let added = error.contains("columns but")
            && error.contains("values were supplied")
            && history.steps[..step]
                .iter()
                .any(|s| matches!(s, Step::AddColumn(..)))
            && kind == Kind::Mysql
            && matches!(&history.steps[step], Step::Insert(rows)
                if rows.iter().any(|(_, values)| values.iter().any(mysql_unheld)));
        matches!(history.steps[step], Step::Insert(_)) && (widened || added)
    }

    /// The 1.5.5 oracle splices a string holding a NUL byte into SQL and fails
    /// to parse it, fixed on upstream main, not in 1.5.6; besides a filter on
    /// such a string (`upstream-bugs/nul-string-filter-parser-error.sql`,
    /// which the generator does not produce), an insert into a table
    /// partitioned by a string column can fail this way. Whether `step` is
    /// such an insert failing so.
    pub(super) fn nul_partition_parser_error(history: &History, step: usize, error: &str) -> bool {
        let nul = |v: &V| matches!(v, V::S(s) if s.contains('\0'));
        let written = match &history.steps[step] {
            Step::Insert(rows) => rows.iter().any(|(_, values)| values.iter().any(nul)),
            // A flush or checkpoint writes the inlined rows.
            Step::Flush | Step::Checkpoint => history.steps[..step].iter().any(|s| match s {
                Step::Insert(rows) => rows.iter().any(|(_, values)| values.iter().any(nul)),
                Step::Update {
                    value,
                    ..
                } => nul(value),
                _ => false,
            }),
            _ => false,
        };
        error.contains("unterminated quoted string")
            && written
            && history.steps[..step]
                .iter()
                .any(|s| matches!(s, Step::Partition(Some(_))))
    }

    /// The partition side of [`stale_inlined_table`]: a partition spec set
    /// while the inlining limit is `0` gets no inlined table, so official's
    /// later inlined rows go to an older table and its flush writes them
    /// under that table's (older) layout, where this crate writes them under
    /// the live spec (see COMPATIBILITY.md). Whether a flush up to `step` can
    /// have done so: the Hive layout of live files then differs.
    pub(super) fn stale_inlined_partition(history: &History, step: usize) -> bool {
        let mut limit = history.limit.unwrap_or(10);
        let mut stale = false;
        let mut inlined = false;
        for s in &history.steps[..=step] {
            match s {
                Step::Limit(l) => limit = *l,
                Step::Partition(_) if limit == 0 => stale = true,
                Step::Insert(rows) if stale && rows.len() <= limit => inlined = true,
                Step::Update {
                    ..
                } if stale && limit > 0 => inlined = true,
                Step::Flush | Step::Checkpoint if inlined => return true,
                _ => {},
            }
        }
        false
    }

    /// PostgreSQL text holds no NUL byte, so a partition value with one
    /// fails the write on a PostgreSQL catalog (see COMPATIBILITY.md).
    pub(super) fn postgres_nul_partition(
        history: &History,
        step: usize,
        kind: Kind,
        error: &str,
    ) -> bool {
        matches!(kind, Kind::PostgresSingle | Kind::Multicatalog)
            && error.contains("0x00")
            && history.steps[..=step]
                .iter()
                .any(|s| matches!(s, Step::Partition(Some(_))))
            && history.steps[..=step].iter().any(|s| match s {
                Step::Insert(rows) => rows.iter().any(|(_, values)| {
                    values
                        .iter()
                        .any(|v| matches!(v, V::S(s) if s.contains('\0')))
                }),
                Step::Update {
                    value: V::S(s),
                    ..
                } => s.contains('\0'),
                _ => false,
            })
    }

    /// A value MySQL's inlined columns cannot hold (NaN, an infinity, a date
    /// or timestamp outside the years 1000 to 9999) sends this crate's MySQL
    /// write to Parquet where official inlines it (see COMPATIBILITY.md): the
    /// row ids of later inserts differ. Whether that happened up to `step`.
    pub(super) fn mysql_fallback(history: &History, step: usize, kind: Kind) -> bool {
        if kind == Kind::Mysql {
            return history.steps[..=step].iter().any(|s| match s {
                Step::Insert(rows) => rows
                    .iter()
                    .any(|(_, values)| values.iter().any(mysql_unheld)),
                Step::Update {
                    value,
                    ..
                } => mysql_unheld(value),
                _ => false,
            });
        }
        false
    }

    /// Whether `v` is a value MySQL's inlined columns cannot hold; see
    /// [`mysql_fallback`].
    pub(super) fn mysql_unheld(v: &V) -> bool {
        const DATE_MIN: i32 = -354_285;
        const DATE_MAX: i32 = 2_932_896;
        const TS_MIN: i64 = -30_610_224_000_000_000;
        const TS_MAX: i64 = 253_402_300_799_999_999;
        match v {
            V::F(f) => !f.is_finite(),
            V::R(f) => !f.is_finite(),
            V::Date(d) => !(DATE_MIN..=DATE_MAX).contains(d),
            V::Ts(t) => !(TS_MIN..=TS_MAX).contains(t),
            _ => false,
        }
    }

    /// duckdb/ducklake#1603: official's change feeds misdate deletions from a
    /// data file with `partial_max` (one `merge_adjacent_files` or a flush of
    /// inlined data wrote, so it carries each row's insert snapshot and row
    /// id). This crate dates each deletion at its own snapshot. Two shapes:
    ///
    /// - A delete file without per-row snapshots (an engine's first delete
    ///   from the file): official dates its deletions at the deleted rows'
    ///   insert snapshots, in every window holding the delete file's
    ///   `begin_snapshot`, and over a window also holding an insert reports
    ///   an update pair. Repro:
    ///   `upstream-bugs/merged-file-delete-misdated-in-change-feed.sql`.
    /// - A row id the file holds at two positions (a row updated while
    ///   inlined, then flushed): official keys its deletions by row id, so
    ///   over a window holding the deletions of two such positions it reports
    ///   both at one of their snapshots (a flushed update repeats its pairs,
    ///   and the later deletion goes missing). A row id two such files hold
    ///   (a partitioned flush puts the versions of an updated row in the
    ///   files of their partitions): official dates a deletion of it at
    ///   another snapshot in some runs, in any window holding the deletion.
    ///
    /// The groups of official snapshots a window must not hold all of to be
    /// unaffected: `[begin_snapshot]` per delete file of the first shape,
    /// each pair of deletion snapshots of positions sharing a row id, and
    /// each deletion snapshot of a row id two files hold. A
    /// delete file with per-row snapshots (a flush's, dated at the earliest of
    /// them) is otherwise read right, in every window. `None` when the files
    /// cannot be read.
    pub(super) fn misdated_delete_snapshots(
        connection: &duckdb::Connection,
    ) -> Option<Vec<Vec<i64>>> {
        let files = Official::rows(
            connection,
            "SELECT f.data_file_id, \
               CASE WHEN f.path_is_relative \
                 THEN m.value || s.path || tb.path || f.path ELSE f.path END, \
               d.begin_snapshot, \
               CASE WHEN d.path_is_relative \
                 THEN m.value || s.path || tb.path || d.path ELSE d.path END, \
               tb.table_id \
             FROM __ducklake_metadata_lake.ducklake_data_file f \
             JOIN __ducklake_metadata_lake.ducklake_table tb ON tb.table_id = f.table_id \
               AND tb.end_snapshot IS NULL \
             JOIN __ducklake_metadata_lake.ducklake_schema s ON s.schema_id = tb.schema_id \
               AND s.end_snapshot IS NULL \
             LEFT JOIN __ducklake_metadata_lake.ducklake_delete_file d USING (data_file_id) \
             CROSS JOIN (SELECT value FROM __ducklake_metadata_lake.ducklake_metadata \
               WHERE key = 'data_path') m \
             WHERE f.partial_max IS NOT NULL",
        )
        .ok()?
        .1;
        let unquote = |text: &str| text.trim_matches('"').replace('\'', "''");
        let null = |text: &str| text.is_empty() || text == "NULL";
        let mut groups = Vec::new();
        let mut by_file: std::collections::BTreeMap<String, (String, Vec<(i64, i64)>)> =
            Default::default();
        for file in files {
            let data = unquote(&file[1]);
            // Each deleted position's snapshot: the delete file's per-row
            // snapshots, or its begin_snapshot without them.
            let mut deleted: Vec<(i64, i64)> = Vec::new();
            if !null(&file[3]) {
                let begin: i64 = file[2].parse().ok()?;
                let delete = unquote(&file[3]);
                let columns = Official::rows(
                    connection,
                    &format!("SELECT name FROM parquet_schema('{delete}')"),
                )
                .ok()?
                .1;
                if columns
                    .iter()
                    .any(|c| c[0].trim_matches('"') == "_ducklake_internal_snapshot_id")
                {
                    for row in Official::rows(
                        connection,
                        &format!(
                            "SELECT pos, _ducklake_internal_snapshot_id \
                             FROM read_parquet('{delete}')"
                        ),
                    )
                    .ok()?
                    .1
                    {
                        deleted.push((row[0].parse().ok()?, row[1].parse().ok()?));
                    }
                } else {
                    groups.push(vec![begin]);
                    for row in Official::rows(
                        connection,
                        &format!("SELECT pos FROM read_parquet('{delete}')"),
                    )
                    .ok()?
                    .1
                    {
                        deleted.push((row[0].parse().ok()?, begin));
                    }
                }
            }
            // Inlined deletions of the file's rows, keyed by position.
            if let Ok((_, rows)) = Official::rows(
                connection,
                &format!(
                    "SELECT row_id, begin_snapshot \
                     FROM __ducklake_metadata_lake.ducklake_inlined_delete_{} \
                     WHERE file_id = {}",
                    file[4], file[0]
                ),
            ) {
                for row in rows {
                    deleted.push((row[0].parse().ok()?, row[1].parse().ok()?));
                }
            }
            let entry = by_file
                .entry(file[0].clone())
                .or_insert_with(|| (data, Vec::new()));
            for d in deleted {
                if !entry.1.contains(&d) {
                    entry.1.push(d);
                }
            }
        }
        // Each file's row ids by position, the deleted ones with their
        // snapshots, and the files each row id is held in.
        let mut files_of: std::collections::BTreeMap<String, std::collections::BTreeSet<String>> =
            Default::default();
        let mut deletions: Vec<(String, String, Vec<i64>)> = Vec::new();
        for (file, (data, deleted)) in &by_file {
            let rows = Official::rows(
                connection,
                &format!(
                    "SELECT file_row_number, _ducklake_internal_row_id \
                     FROM read_parquet('{data}', file_row_number = true)"
                ),
            )
            .map(|r| r.1)
            .unwrap_or_default();
            let mut by_row: std::collections::BTreeMap<String, Vec<i64>> = Default::default();
            for row in rows {
                files_of
                    .entry(row[1].clone())
                    .or_default()
                    .insert(file.clone());
                let position: i64 = row[0].parse().ok()?;
                for &(p, snapshot) in deleted {
                    if p == position {
                        by_row.entry(row[1].clone()).or_default().push(snapshot);
                    }
                }
            }
            for (row_id, snapshots) in by_row {
                deletions.push((file.clone(), row_id, snapshots));
            }
        }
        for (_, row_id, snapshots) in &deletions {
            if files_of.get(row_id).is_some_and(|files| files.len() > 1) {
                // A row id held in several files: official's deletion of it
                // can take another file's snapshot, or its insert snapshot,
                // in any window holding the deletion.
                groups.extend(snapshots.iter().map(|&s| vec![s]));
            }
            for (i, &a) in snapshots.iter().enumerate() {
                for &b in &snapshots[i + 1..] {
                    groups.push(vec![a, b]);
                }
            }
        }
        Some(groups)
    }

    /// duckdb/ducklake#1406: when `rewrite_data_files` retires a data file
    /// that has a delete file and inlined deletions, it ends the delete file,
    /// and a later flush writes the inlined deletions to a second delete file
    /// dated at their own snapshot that never ends. At the snapshots both
    /// cover, official reads the retired file once per delete file, each copy
    /// filtered by that file's deletions alone, and its change feeds report
    /// the ended delete file's deletions again at the rewrite's snapshot.
    /// This crate merges the deletions into the ended delete file (see
    /// COMPATIBILITY.md). Fixed on upstream main by duckdb/ducklake#1489, not
    /// in 1.5.6.
    /// Repro: `upstream-bugs/rewrite-flush-backdated-delete-file.sql`.
    /// Whether official's catalog holds a retired data file live at official
    /// snapshot `snapshot` with two delete files live there.
    pub(super) fn overlapping_delete_files(connection: &duckdb::Connection, snapshot: i64) -> bool {
        let live = |alias: &str| {
            format!(
                "{alias}.begin_snapshot <= {snapshot} \
                 AND ({alias}.end_snapshot IS NULL OR {alias}.end_snapshot > {snapshot})"
            )
        };
        retired_with_overlapping_deletes(
            connection,
            &format!("{} AND {} AND {}", live("f"), live("a"), live("b")),
        )
    }

    /// [`overlapping_delete_files`] for a change feed over official snapshots
    /// `first..=last`, on a retired data file with two delete files whose
    /// live ranges overlap: whether the file was retired at a snapshot in the
    /// window, or the window starts after the later delete file begins. In
    /// the latter official takes the later (backdated) file, which lacks the
    /// earlier one's deletions, for the deletions before the window, and so
    /// reports the earlier file's deletions, dated at its begin_snapshot, in
    /// every such window.
    pub(super) fn overlapping_delete_files_retired_in(
        connection: &duckdb::Connection,
        first: i64,
        last: i64,
    ) -> bool {
        retired_with_overlapping_deletes(
            connection,
            &format!(
                "(f.end_snapshot BETWEEN {first} AND {last} \
                  OR {first} > greatest(a.begin_snapshot, b.begin_snapshot)) \
                 AND a.begin_snapshot < coalesce(b.end_snapshot, 9223372036854775807) \
                 AND b.begin_snapshot < coalesce(a.end_snapshot, 9223372036854775807)"
            ),
        )
    }

    /// Whether official's catalog holds a retired data file `f` with two
    /// delete files `a` and `b` that meet `condition`.
    fn retired_with_overlapping_deletes(connection: &duckdb::Connection, condition: &str) -> bool {
        let rows = Official::rows(
            connection,
            &format!(
                "SELECT count(*) FROM __ducklake_metadata_lake.ducklake_data_file f \
                 JOIN __ducklake_metadata_lake.ducklake_delete_file a USING (data_file_id) \
                 JOIN __ducklake_metadata_lake.ducklake_delete_file b USING (data_file_id) \
                 WHERE a.delete_file_id < b.delete_file_id \
                 AND f.end_snapshot IS NOT NULL AND {condition}"
            ),
        )
        .unwrap();
        rows.1[0][0] != "0"
    }

    /// datafusion-contrib/datafusion-ducklake#377 (a difference of this
    /// crate, not an upstream bug): `rewrite_data_files` writes one file per
    /// rewritten file where official packs the surviving rows into fewer
    /// files. Whether a rewrite wrote the same rows here into more files than
    /// official did, from each side's `(files, rows)` of
    /// [`rewrite_outputs`]. From then on a later delete can be inlined
    /// against one of this crate's small files where official records it in
    /// a delete file that ends its larger file (or the reverse), so a side
    /// can keep a live file with no live row the other has ended. Only the
    /// layout of the files holding no live row may then differ
    /// ([`live_layout`]); rows, row ids, feeds and statistics compare as
    /// before.
    pub(super) fn rewrite_grouping(official: (i64, i64), ours: (i64, i64)) -> bool {
        official.0 >= 1 && ours.0 > official.0 && ours.1 == official.1
    }

    /// The data files of table `t`, and their rows, that a rewrite
    /// (`rewrite_delete` in the snapshot's changes) committed in a snapshot
    /// after `from` up to `to`, in the metadata attached raw on `connection`.
    pub(super) fn rewrite_outputs(
        connection: &duckdb::Connection,
        from: i64,
        to: i64,
    ) -> (i64, i64) {
        let rows = Official::rows(
            connection,
            &format!(
                "SELECT count(*)::BIGINT, coalesce(sum(f.record_count), 0)::BIGINT \
                 FROM ducklake_data_file f JOIN ducklake_table t USING (table_id) \
                 JOIN ducklake_snapshot_changes c ON c.snapshot_id = f.begin_snapshot \
                 WHERE t.table_name = 't' AND t.end_snapshot IS NULL \
                   AND f.begin_snapshot > {from} AND f.begin_snapshot <= {to} \
                   AND c.changes_made LIKE '%rewrite_delete%'"
            ),
        )
        .unwrap();
        (rows.1[0][0].parse().unwrap(), rows.1[0][1].parse().unwrap())
    }

    /// [`super::Run::layout_rows`] of only the live data files that hold a
    /// live row, as DuckDB reads the table `t` of the DuckLake catalog
    /// attached as `lake` on `connection` (its `filename` names the file each
    /// row came from; file names are unique UUIDs).
    pub(super) fn live_layout(connection: &duckdb::Connection) -> Rows {
        Official::rows(
            connection,
            "WITH live AS (SELECT DISTINCT regexp_extract(filename, '[^/]*$') AS name \
                FROM lake.main.t) \
             SELECT DISTINCT regexp_replace(f.path, '[^/]*$', '') AS dir, \
                pv.partition_key_index, pv.partition_value \
             FROM __ducklake_metadata_lake.ducklake_data_file f \
             JOIN __ducklake_metadata_lake.ducklake_table t USING (table_id) \
             LEFT JOIN __ducklake_metadata_lake.ducklake_file_partition_value pv \
               USING (data_file_id) \
             WHERE t.table_name = 't' AND t.end_snapshot IS NULL \
               AND f.end_snapshot IS NULL AND f.path_is_relative \
               AND regexp_extract(f.path, '[^/]*$') IN (SELECT name FROM live) \
             ORDER BY ALL",
        )
    }
}

/// The class of an error, so the two engines' failures of one step can be
/// compared by kind rather than by wording.
fn error_class(error: &str) -> &'static str {
    let lower = error.to_ascii_lowercase();
    if lower.contains("columns but") && lower.contains("values were supplied") {
        // An insert into a table's older inlined data table that lacks a
        // column added since (duckdb/ducklake#1606), which this crate refuses
        // with official's message.
        "#1606 inlined table shape"
    } else if lower.contains("conflict") {
        "conflict"
    } else if lower.contains("not null") || lower.contains("constraint") {
        "constraint"
    } else if lower.contains("not supported")
        || lower.contains("unsupported")
        || lower.contains("not implemented")
    {
        "unsupported"
    } else if lower.contains("conversion")
        || lower.contains("cast")
        || lower.contains("out of range")
        || lower.contains("overflow")
        || lower.contains("could not convert")
    {
        "conversion"
    } else if lower.contains("does not exist")
        || lower.contains("not found")
        || lower.contains("binder error")
        || lower.contains("catalog error")
    {
        "binding"
    } else {
        "other"
    }
}

/// One read the harness asks of both engines: `template` names the table
/// `{T}`, which is official's table, the materialized truth, or official's
/// table at a snapshot.
struct Query {
    template: String,
    ours: String,
    column: Option<Col>,
    probe: Option<V>,
}

impl Query {
    fn on(&self, table: &str) -> String {
        self.template.replace("{T}", table)
    }
}

struct Run<'a> {
    history: &'a History,
    official: Official,
    lake: Lake,
    dir: PathBuf,
    rng: Rng,
    exhaustive: bool,
    heads: Heads,
    /// The first step whose snapshot is still readable on both sides (after
    /// an expiry, only the latest).
    readable_from: usize,
    /// Whether to count this run's comparisons in [`coverage`].
    record: bool,
    /// Columns whose table-wide statistics official got wrong, and why: their
    /// statistics rows are not compared, and official's pruned answers about
    /// them may differ from the truth.
    stats_diverged: std::collections::BTreeMap<String, &'static str>,
    /// Why official's `ducklake_table_stats` row no longer compares.
    table_stats_diverged: Option<&'static str>,
    /// Steps that failed on official from a modelled upstream bug and
    /// committed on this crate without changing the table's rows (or with
    /// their rows deleted again).
    official_failed: std::collections::BTreeSet<usize>,
    /// Row ids no longer compare: this crate consumed ids official did not.
    blind: bool,
    /// Why the Hive layouts of the two catalogs' files no longer compare.
    layout_diverged: Option<&'static str>,
    /// The step of a rewrite that wrote the same rows into more files here
    /// than official ([`known::rewrite_grouping`], #377): the layouts of the
    /// files holding live rows still compare.
    rewrite_grouped: Option<usize>,
    /// Each comparison this run made and each it skipped, as
    /// `comparison|ran` or `comparison|skipped: reason`, whether or not it is
    /// [`Self::record`]ed in [`coverage`].
    tally: std::sync::Mutex<Vec<String>>,
}

/// The canonical rows of a read, or its error.
type Rows = Result<(Vec<String>, Vec<Vec<String>>), String>;

impl Run<'_> {
    fn ran(&self, comparison: &str) {
        self.tally.lock().unwrap().push(format!("{comparison}|ran"));
        if self.record {
            coverage::ran(self.lake.kind, comparison);
        }
    }

    fn skip(&self, comparison: &str, reason: &str) {
        self.tally
            .lock()
            .unwrap()
            .push(format!("{comparison}|skipped: {reason}"));
        if self.record {
            coverage::skipped(self.lake.kind, comparison, reason);
        }
    }

    /// Compare the Hive layouts of the live files of official's catalog and
    /// of this crate's (its metadata `ours`, read raw for `ours_layout`).
    /// After a rewrite official packed into fewer files
    /// ([`known::rewrite_grouping`]), layouts that differ compare again over
    /// only the files that hold a live row ([`known::live_layout`]).
    fn compare_layout(
        &self,
        step: usize,
        ours: &Metadata,
        ours_layout: Rows,
    ) -> Result<(), Mismatch> {
        let official_layout = self.official_layout();
        let full = compare("partition_layout", step, &official_layout, &ours_layout);
        if full.is_err()
            && let Some(rewrite) = self.rewrite_grouped
        {
            let official_live = Self::crate_spelling(known::live_layout(&self.official.connection));
            if let Metadata::Mysql {
                ..
            } = ours
            {
                // DuckDB cannot attach this crate's MySQL catalogs, so which of
                // its files hold live rows is unread: each directory where
                // official holds a live row must still hold a live file here.
                let (Ok((_, theirs)), Ok((_, all))) = (&official_live, &ours_layout) else {
                    return full;
                };
                if let Some(missing) = theirs.iter().find(|row| !all.contains(row)) {
                    return Err(Mismatch {
                        kind: "partition_layout",
                        step,
                        detail: format!(
                            "official holds live rows in {missing:?}, where this crate's MySQL \
                             catalog has no live file (after the rewrite at step {rewrite}, #377)"
                        ),
                    });
                }
                self.skip(
                    "partition_layout",
                    "#377 rewrite grouping: MySQL, official's live directories compared",
                );
                return Ok(());
            }
            let reader = duckdb_session();
            let load = match ours {
                Metadata::Postgres(_) => "postgres",
                _ => "sqlite",
            };
            crate::common::ensure_extension_installed(load);
            reader
                .execute_batch(&format!(
                    "LOAD {load}; ATTACH '{}' AS lake (DATA_PATH '{}/', READ_ONLY)",
                    ours.ducklake_uri(),
                    self.lake.data.display()
                ))
                .map_err(|e| Mismatch {
                    kind: "duckdb_attach",
                    step,
                    detail: e.to_string(),
                })?;
            let ours_live = known::live_layout(&reader);
            compare("partition_layout", step, &official_live, &ours_live).map_err(|m| {
                Mismatch {
                    detail: format!(
                        "{}\n(live files only, after the rewrite at step {rewrite} that \
                     wrote more files than official's, #377; all live files:\n{})",
                        m.detail,
                        full.as_ref().unwrap_err().detail
                    ),
                    ..m
                }
            })?;
            self.skip(
                "partition_layout",
                "#377 rewrite grouping: files with live rows compared",
            );
            return Ok(());
        }
        self.ran("partition_layout");
        full
    }

    /// How the cells of the columns at heads index `at` compare: the float
    /// columns whose zeros compare without their sign
    /// ([`known::negative_zero`]), and the `DOUBLE` columns whose values
    /// official may have inlined rounded ([`known::inlined_double_text`]).
    fn cells(&self, at: usize) -> Cells {
        let columns = self.history.columns_at(at);
        Cells {
            signless: columns
                .iter()
                .filter(|c| known::negative_zero(self.history, at, c))
                .map(|c| c.name.clone())
                .collect(),
            rounded: columns
                .iter()
                .filter(|c| known::inlined_double_text(self.history, at, c))
                .map(|c| c.name.clone())
                .collect(),
            rounded_keys: columns
                .iter()
                .flat_map(|c| known::inlined_double_keys(self.history, at, c))
                .collect(),
        }
    }

    /// [`compare`], and where only the row ids differ after a merge of an
    /// update's rows ([`known::merged_update_order`]), a skip.
    fn compare_rows(
        &self,
        kind: &'static str,
        step: usize,
        official: &Rows,
        ours: &Rows,
    ) -> Result<(), Mismatch> {
        let result = compare(kind, step, official, ours);
        if result.is_err()
            && let Some(updated) = known::merged_update_ids(self.history, step)
            && let (Ok(o), Ok(c)) = (official, ours)
            && known::row_ids_traded(o, c, &updated)
        {
            self.skip(kind, "#1605 row order of a merged update");
            return Ok(());
        }
        result
    }

    /// Whether an exhaustive run reads the rows at every earlier snapshot and
    /// every change-feed window after the step reaching `index`: after the
    /// history's last step, and after every step under `FUZZ_ALL_WINDOWS`.
    /// After the other steps it reads the snapshot just before and the windows
    /// ending at `index`, so each is still read when it lands and again at the
    /// end; reading every window after every step costs a cube of the
    /// history's length.
    fn every_earlier(&self, index: usize) -> bool {
        index == self.history.steps.len() || std::env::var("FUZZ_ALL_WINDOWS").is_ok()
    }

    async fn compare_step(&mut self, step: usize) -> Result<(), Mismatch> {
        let index = step + 1;
        let list = |at: usize| {
            std::iter::once("id".to_string())
                .chain(self.history.columns_at(at).into_iter().map(|c| c.name))
                .collect::<Vec<_>>()
                .join(", ")
        };
        let blind = self.blind || known::mysql_fallback(self.history, step, self.lake.kind);
        let signless = self.cells(index);
        let official = Official::rows(
            &self.official.connection,
            &format!("SELECT rowid, {} FROM t", list(index)),
        )
        .map(|r| {
            normalize_cells(
                without_row_ids(normalize(r, &self.heads.official), blind),
                &signless,
            )
        });
        let ours = self
            .lake
            .rows(
                None,
                &format!("SELECT rowid, {} FROM lake.main.t", list(index)),
            )
            .await
            .map(|r| {
                normalize_cells(
                    without_row_ids(normalize(r, &self.heads.ours), blind),
                    &signless,
                )
            });
        self.ran("rows");
        if blind {
            self.skip("row ids", "MySQL fallback to Parquet");
        } else {
            self.ran("row ids");
        }
        self.compare_rows("rows", step, &official, &ours)?;

        // Time travel to earlier steps: every one after the last step or under
        // `FUZZ_ALL_WINDOWS`, else the previous one in an exhaustive run and a
        // random one otherwise.
        let earlier: Vec<usize> = if self.exhaustive && self.every_earlier(index) {
            (self.readable_from..index).collect()
        } else if self.exhaustive && index > self.readable_from {
            vec![index - 1]
        } else if index > self.readable_from {
            vec![self.readable_from + self.rng.below(index - self.readable_from)]
        } else {
            Vec::new()
        };
        for at in earlier {
            if known::overlapping_delete_files(&self.official.connection, self.heads.official[at]) {
                self.skip("time_travel", "#1406 overlapping delete files");
                continue;
            }
            self.ran("time_travel");
            let signless = self.cells(at);
            let official = Official::rows(
                &self.official.connection,
                &format!(
                    "SELECT rowid, {} FROM t AT (VERSION => {})",
                    list(at),
                    self.heads.official[at]
                ),
            )
            .map(|r| {
                normalize_cells(
                    without_row_ids(normalize(r, &self.heads.official), blind),
                    &signless,
                )
            });
            let ours = self
                .lake
                .rows(
                    Some(self.heads.ours[at]),
                    &format!("SELECT rowid, {} FROM lake.main.t", list(at)),
                )
                .await
                .map(|r| {
                    normalize_cells(
                        without_row_ids(normalize(r, &self.heads.ours), blind),
                        &signless,
                    )
                });
            self.compare_rows("time_travel", step, &official, &ours)
                .map_err(|m| Mismatch {
                    detail: format!("at step {at}: {}", m.detail),
                    ..m
                })?;
        }

        // Change feeds over windows of steps: every one in a short history.
        // An exhaustive run reads each window when its last step lands and
        // every window again after the history's last step (see
        // [`Run::every_earlier`]).
        let windows: Vec<(usize, usize)> = if self.exhaustive && self.every_earlier(index) {
            (self.readable_from..index)
                .flat_map(|a| (a + 1..=index).map(move |b| (a, b)))
                .collect()
        } else if self.exhaustive {
            (self.readable_from..index).map(|a| (a, index)).collect()
        } else if index > self.readable_from {
            let a = self.readable_from + self.rng.below(index - self.readable_from);
            let b = a + 1 + self.rng.below(index - a);
            vec![(a, b)]
        } else {
            Vec::new()
        };
        let misdated = known::misdated_delete_snapshots(&self.official.connection);
        for (a, b) in windows {
            let (oa, ob) = (self.heads.official[a] + 1, self.heads.official[b]);
            let (ca, cb) = (self.heads.ours[a] + 1, self.heads.ours[b]);
            if (a..b).any(|s| self.official_failed.contains(&s)) {
                self.skip("changes", "window holds a step official failed");
                continue;
            }
            if oa > ob || ca > cb {
                // No snapshot on one side: compared as empty below.
                if oa > ob && ca > cb {
                    continue;
                }
            }
            match &misdated {
                Some(groups)
                    if groups
                        .iter()
                        .any(|g| g.iter().all(|s| (oa..=ob).contains(s))) =>
                {
                    self.skip("changes", "#1603 misdated partial-file deletion");
                    continue;
                },
                Some(_) => {},
                None => {
                    self.skip("changes", "#1603 deletions unreadable");
                    continue;
                },
            }
            if known::overlapping_delete_files_retired_in(&self.official.connection, oa, ob) {
                self.skip("changes", "#1406 overlapping delete files");
                continue;
            }
            // Both feeds, in one shape: official's deletions feed carries no
            // `change_type` column, this crate's a constant one.
            for (feed, official_sql, our_sql) in [
                (
                    "changes",
                    format!(
                        "SELECT * FROM ducklake_table_changes('lake', 'main', 't', {oa}, {ob})"
                    ),
                    format!("SELECT * FROM ducklake_table_changes('main.t', {ca}, {cb})"),
                ),
                (
                    "deletions",
                    format!(
                        "SELECT snapshot_id, rowid, 'delete' AS change_type, * \
                         FROM ducklake_table_deletions('lake', 'main', 't', {oa}, {ob})"
                    ),
                    format!("SELECT * FROM ducklake_table_deletions('main.t', {ca}, {cb})"),
                ),
            ] {
                let rows_kind = if feed == "changes" {
                    "changes rows"
                } else {
                    "deletions rows"
                };
                self.ran(feed);
                let empty = || Ok((Vec::new(), Vec::new()));
                let (official, repeated) = if oa > ob {
                    (empty(), false)
                } else {
                    match Official::rows(&self.official.connection, &official_sql) {
                        Ok(r) => {
                            let (r, repeated) = known::rewrite_repeats(
                                self.history,
                                (a, b),
                                normalize(r, &self.heads.official),
                            );
                            (Ok(r), repeated)
                        },
                        Err(e) => (Err(e), false),
                    }
                };
                if repeated {
                    self.skip(rows_kind, "#1571 sibling: rewrite repeats");
                }
                let ours = if ca > cb {
                    empty()
                } else {
                    self.lake
                        .rows(None, &our_sql)
                        .await
                        .map(|r| normalize(r, &self.heads.ours))
                };
                // Rows official dates at a maintenance step where this crate
                // reports none (#1571 family): a rewrite, merge, flush or
                // checkpoint changes no row.
                let official = match (official, &ours) {
                    (Ok(o), Ok((_, c))) => {
                        let (o, dropped) = known::maintenance_rows(self.history, o, c);
                        for _ in 0..dropped {
                            self.skip(
                                rows_kind,
                                "#1571 family: official's rows at a maintenance step",
                            );
                        }
                        Ok(o)
                    },
                    (official, _) => official,
                };
                let finish = |rows: Rows| {
                    rows.map(|r| normalize_cells(without_row_ids(r, blind), &signless))
                };
                let (official, ours) = (finish(official), finish(ours));
                self.compare_rows(feed, step, &official, &ours)
                    .map_err(|m| Mismatch {
                        detail: format!("window steps ({a}, {b}]: {}", m.detail),
                        ..m
                    })?;
            }
        }
        self.compare_reads(step).await?;
        // Predicates and aggregates at an earlier snapshot: every one after
        // each step of an exhaustive run, as which files a predicate prunes at
        // an earlier snapshot can change with the catalog's later statistics.
        let earlier: Vec<usize> = if self.exhaustive {
            (self.readable_from..index).collect()
        } else if index > self.readable_from {
            vec![self.readable_from + self.rng.below(index - self.readable_from)]
        } else {
            Vec::new()
        };
        for at in earlier {
            self.compare_reads_at(step, at).await?;
        }
        Ok(())
    }

    /// The predicates and aggregates read of the columns at heads index `at`:
    /// a count, per column a NULL count, and per comparable column its
    /// minimum and maximum and filters at a probe, half of the time a value
    /// the column holds.
    fn queries(&mut self, at: usize) -> Vec<Query> {
        let mut queries = vec![Query {
            template: "SELECT count(*) FROM {T}".to_string(),
            ours: "SELECT count(*) FROM lake.main.t".to_string(),
            column: None,
            probe: None,
        }];
        for column in self.history.columns_at(at) {
            let name = &column.name;
            let mut push = |template: String, ours: String, probe: Option<V>| {
                queries.push(Query {
                    template,
                    ours,
                    column: Some(column.clone()),
                    probe,
                });
            };
            push(
                format!("SELECT count(*) FROM {{T}} WHERE {name} IS NULL"),
                format!("SELECT count(*) FROM lake.main.t WHERE {name} IS NULL"),
                None,
            );
            if !column.ty.comparable() {
                continue;
            }
            push(
                format!("SELECT min({name}), max({name}), count({name}) FROM {{T}}"),
                format!("SELECT min({name}), max({name}), count({name}) FROM lake.main.t"),
                None,
            );
            let held = known::written_before(self.history, at, name)
                .into_iter()
                .filter(|v| *v != V::Null)
                .collect::<Vec<_>>();
            let probe = if !held.is_empty() && self.rng.chance(50) {
                self.rng.pick(&held).clone()
            } else {
                gen_value(&mut self.rng, column.ty)
            };
            let zero = matches!(probe, V::F(f) if f == 0.0) || matches!(probe, V::R(f) if f == 0.0);
            if probe != V::Null && !zero && !matches!(&probe, V::S(s) if s.contains('\0')) {
                let ops: &[&str] = if column.ty.is_float() {
                    &["=", "<", ">", "<=", ">="]
                } else {
                    &["=", "<", ">"]
                };
                for op in ops {
                    push(
                        format!(
                            "SELECT count(*) FROM {{T}} WHERE {name} {op} {}",
                            duckdb_literal(column.ty, &probe)
                        ),
                        format!(
                            "SELECT count(*) FROM lake.main.t WHERE {name} {op} {}",
                            datafusion_literal(column.ty, &probe)
                        ),
                        Some(probe.clone()),
                    );
                }
            }
            if column.ty.is_float() {
                let nan = if column.ty == Ty::Double {
                    V::F(f64::NAN)
                } else {
                    V::R(f32::NAN)
                };
                push(
                    format!(
                        "SELECT count(*) FROM {{T}} WHERE {name} = {}",
                        duckdb_literal(column.ty, &nan)
                    ),
                    format!(
                        "SELECT count(*) FROM lake.main.t WHERE {name} = {}",
                        datafusion_literal(column.ty, &nan)
                    ),
                    Some(nan),
                );
            }
        }
        queries
    }

    /// `rows` of a query on `column` with its zeros signless when the
    /// column's are at heads index `at`.
    fn zeros(&self, at: usize, column: Option<&Col>, rows: Rows) -> Rows {
        let all = || vec!["*".to_string()];
        let cells = Cells {
            signless: if column.is_some_and(|c| known::negative_zero(self.history, at, c)) {
                all()
            } else {
                Vec::new()
            },
            rounded: if column.is_some_and(|c| known::inlined_double_text(self.history, at, c)) {
                all()
            } else {
                Vec::new()
            },
            rounded_keys: column
                .map(|c| known::inlined_double_keys(self.history, at, c))
                .unwrap_or_default(),
        };
        rows.map(|r| normalize_cells(r, &cells))
    }

    /// Whether `query` filters a `DOUBLE` column whose values official may
    /// have inlined rounded ([`known::inlined_double_text`]): its count then
    /// differs by the rounded rows, which no normalization of the answer
    /// recovers.
    fn rounded_filter(&self, at: usize, query: &Query) -> bool {
        query.probe.is_some()
            && query
                .column
                .as_ref()
                .is_some_and(|c| known::inlined_double_text(self.history, at, c))
    }

    /// This crate's answers to the predicates and aggregates at heads index
    /// `at` (an earlier snapshot), against official's table at that snapshot
    /// read whole and filtered with nothing to prune by.
    async fn compare_reads_at(&mut self, step: usize, at: usize) -> Result<(), Mismatch> {
        if known::overlapping_delete_files(&self.official.connection, self.heads.official[at]) {
            self.skip("crate_reads_at", "#1406 overlapping delete files");
            return Ok(());
        }
        let queries = self.queries(at);
        self.official.reconnect();
        let official = &self.official.connection;
        official
            .execute_batch(&format!(
                "CREATE OR REPLACE TEMP TABLE truth AS SELECT * FROM t AT (VERSION => {})",
                self.heads.official[at]
            ))
            .map_err(|e| Mismatch {
                kind: "truth",
                step,
                detail: e.to_string(),
            })?;
        for query in &queries {
            let truth = self.zeros(
                at,
                query.column.as_ref(),
                Official::rows(official, &query.on("truth")),
            );
            let got = self.zeros(
                at,
                query.column.as_ref(),
                self.lake.rows(Some(self.heads.ours[at]), &query.ours).await,
            );
            let result = compare("crate_reads_at", step, &truth, &got);
            if result.is_err() && self.rounded_filter(at, query) {
                self.skip("crate_reads_at", "official's rounded inlined DOUBLE");
                continue;
            }
            self.ran("crate_reads_at");
            result.map_err(|m| Mismatch {
                detail: format!("at step {at}: `{}`: {}", query.ours, m.detail),
                ..m
            })?;
        }
        Ok(())
    }

    /// The table-wide statistics rows of `metadata`'s table `t`: each
    /// column's (or leaf field's) `(name, contains_null, contains_nan, min,
    /// max)`, and the table's `(record_count, next_row_id)`.
    fn stats_rows(metadata: &Metadata) -> (Rows, Rows) {
        let connection = duckdb_session();
        metadata.attach_raw(&connection);
        let column_stats = Official::rows(
            &connection,
            "SELECT (SELECT c.column_name FROM ducklake_column c \
                WHERE c.table_id = s.table_id AND c.column_id = s.column_id \
                ORDER BY c.begin_snapshot DESC LIMIT 1) AS name, \
                contains_null::BOOLEAN, contains_nan::BOOLEAN, min_value, max_value \
             FROM ducklake_table_column_stats s JOIN ducklake_table t USING (table_id) \
             WHERE t.table_name = 't' AND t.end_snapshot IS NULL",
        );
        let table_stats = Official::rows(
            &connection,
            "SELECT record_count, next_row_id FROM ducklake_table_stats s \
             JOIN ducklake_table t USING (table_id) \
             WHERE t.table_name = 't' AND t.end_snapshot IS NULL",
        );
        (column_stats, table_stats)
    }

    /// Mark the columns whose official table-wide statistics `step` changed
    /// without a write: official's rewrite recomputes them from the remaining
    /// files (duckdb/ducklake#1377), this crate keeps them widen-only.
    fn note_rewrite_recompute(&mut self, before: &(Rows, Rows), after: &(Rows, Rows)) {
        if let (Ok(b), Ok(a)) = (&before.1, &after.1)
            && b.1 != a.1
        {
            self.table_stats_diverged = Some("#1377 rewrite recomputes table stats");
        }
        if let (Ok(b), Ok(a)) = (&before.0, &after.0) {
            for row in &a.1 {
                if !b.1.contains(row) {
                    let name = row[0].trim_matches('"').to_string();
                    for owner in self.owners(&name) {
                        self.stats_diverged
                            .entry(owner)
                            .or_insert("#1377 rewrite recomputes column stats");
                    }
                }
            }
            for row in &b.1 {
                if !a.1.contains(row) {
                    let name = row[0].trim_matches('"').to_string();
                    for owner in self.owners(&name) {
                        self.stats_diverged
                            .entry(owner)
                            .or_insert("#1377 rewrite recomputes column stats");
                    }
                }
            }
        }
    }

    /// The live columns a statistics row named `name` belongs to: the column
    /// of that name, and the nested columns with a leaf field of it.
    fn owners(&self, name: &str) -> Vec<String> {
        if name == "id" {
            return vec!["id".to_string()];
        }
        let columns = self.history.columns_at(self.heads.official.len() - 1);
        columns
            .iter()
            .filter(|c| {
                let leaves: &[&str] = match c.ty {
                    Ty::Struct => &["a", "b"],
                    Ty::Map => &["key", "value"],
                    Ty::List => &["element"],
                    _ => &[],
                };
                c.name == name || leaves.contains(&name)
            })
            .map(|c| c.name.clone())
            .collect()
    }

    /// Whether the table-wide statistics of top-level `column` in metadata
    /// catalog `catalog` (attached to official's connection) cover every
    /// non-NaN value the truth table holds. A missing row claims nothing; a
    /// row with neither bound claims there is no value.
    fn covers(&self, column: &Col, catalog: &str) -> Result<bool, String> {
        let ty = match column.ty {
            Ty::List | Ty::Struct | Ty::Map | Ty::Blob => return Ok(true),
            ty => ty.duckdb(),
        };
        let name = &column.name;
        let finite = if column.ty.is_float() {
            format!(" WHERE NOT isnan({name})")
        } else {
            String::new()
        };
        let (_, rows) = Official::rows(
            &self.official.connection,
            &format!(
                "WITH s AS (SELECT s.min_value, s.max_value, s.contains_nan \
                   FROM {catalog}.ducklake_table_column_stats s \
                   JOIN {catalog}.ducklake_table tb ON tb.table_id = s.table_id \
                     AND tb.table_name = 't' AND tb.end_snapshot IS NULL \
                   JOIN {catalog}.ducklake_column c ON c.table_id = s.table_id \
                     AND c.column_id = s.column_id AND c.end_snapshot IS NULL \
                     AND c.parent_column IS NULL \
                   WHERE c.column_name = '{name}'), \
                 d AS (SELECT min({name}) AS lo, max({name}) AS hi, count({name}) AS n \
                   FROM truth{finite}) \
                 SELECT count(*) FROM s, d WHERE d.n > 0 AND \
                   ((s.min_value IS NULL AND s.max_value IS NULL) \
                   OR (NOT coalesce(TRY_CAST(s.contains_nan AS BOOLEAN), false) AND \
                   (s.min_value IS NULL OR s.max_value IS NULL \
                   OR TRY_CAST(s.min_value AS {ty}) > d.lo \
                   OR TRY_CAST(s.max_value AS {ty}) < d.hi)))"
            ),
        )?;
        Ok(rows[0][0] == "0")
    }

    /// DuckDB's answers over the crate's catalog, this crate's answers, and
    /// both catalogs' statistics rows, against official's.
    async fn compare_reads(&mut self, step: usize) -> Result<(), Mismatch> {
        let index = step + 1;
        let columns = self.history.columns_at(index);
        let queries = self.queries(index);
        // A fresh session: the 1.5.5 oracle fails a filter over inlined rows
        // in a session that read the table before a column was widened, fixed
        // in 1.5.6 (upstream-bugs/inlined-filter-after-time-travel-internal-error.sql).
        self.official.reconnect();
        self.official
            .connection
            .execute_batch("CREATE OR REPLACE TEMP TABLE truth AS SELECT * FROM t")
            .map_err(|e| Mismatch {
                kind: "truth",
                step,
                detail: e.to_string(),
            })?;
        let trace = std::env::var("FUZZ_TRACE").is_ok();

        // Official's statistics must cover its rows, but where a modelled
        // upstream bug lost them; then they are no longer compared, and
        // official's pruned answers about that column may be wrong.
        let official_stats = Self::stats_rows(&Metadata::Duckdb(self.official.metadata.clone()));
        for column in &columns {
            if self.stats_diverged.contains_key(&column.name) {
                continue;
            }
            let covered = self
                .covers(column, "__ducklake_metadata_lake")
                .map_err(|e| Mismatch {
                    kind: "truth",
                    step,
                    detail: e,
                })?;
            let nested_shape = matches!(column.ty, Ty::List | Ty::Struct | Ty::Map | Ty::Blob)
                && (known::partitioned_null_file(self.history, step, column)
                    || known::unstored_column(self.history, step, column));
            // A float holding only NaN: official's row with neither bound is
            // "covered" (no other value), and this crate stores none.
            let boundless = official_stats.0.as_ref().is_ok_and(|(_, rows)| {
                rows.iter().any(|row| {
                    row[0] == format!("{:?}", column.name) && row[3] == "NULL" && row[4] == "NULL"
                })
            });
            let nan_shape = boundless
                && known::written(self.history, step, &column.name)
                    .iter()
                    .any(|v| {
                        matches!(v, V::F(f) if f.is_nan()) || matches!(v, V::R(f) if f.is_nan())
                    });
            // A widened column whose bounds official dropped (see
            // `known::widened`) is "covered" when no row is left to hold.
            let widened_shape = boundless && known::widened(self.history, step, column);
            if !covered || nested_shape || nan_shape || widened_shape {
                match known::lost_bounds(self.history, step, column) {
                    Some(reason) => {
                        self.stats_diverged.insert(column.name.clone(), reason);
                    },
                    None => {
                        return Err(Mismatch {
                            kind: "official_model",
                            step,
                            detail: format!(
                                "official's statistics of {} no longer cover its values: {:?}",
                                column.name, official_stats.0
                            ),
                        });
                    },
                }
            }
        }

        let mut truth = Vec::with_capacity(queries.len());
        let mut pruned = Vec::with_capacity(queries.len());
        for query in &queries {
            if trace {
                eprintln!("official: {}", query.on("t"));
            }
            truth.push(self.zeros(
                index,
                query.column.as_ref(),
                Official::rows(&self.official.connection, &query.on("truth")),
            ));
            pruned.push(self.zeros(
                index,
                query.column.as_ref(),
                Official::rows(&self.official.connection, &query.on("t")),
            ));
        }

        // Official's own pruned answer may part from the truth only where a
        // modelled upstream bug drops a matching file.
        let explained = |query: &Query, run: &Self| -> Option<&'static str> {
            let column = query.column.as_ref()?;
            if let Some(probe) = &query.probe {
                if known::float_exponent_filter(column.ty, probe) {
                    return Some("#1604 float exponent");
                }
                if known::double_fixed_point_filter(column.ty, probe) {
                    return Some("double fixed-point constant");
                }
            }
            run.stats_diverged.get(&column.name).copied()
        };
        for ((query, truth), pruned) in queries.iter().zip(&truth).zip(&pruned) {
            if let (Ok(t), Ok(p)) = (truth, pruned)
                && t.1 != p.1
            {
                match explained(query, self) {
                    Some(reason) => self.skip("official_pruned", reason),
                    None => {
                        return Err(Mismatch {
                            kind: "official_model",
                            step,
                            detail: format!(
                                "`{}` is {:?} in official, but {:?} unpruned",
                                query.on("t"),
                                p.1,
                                t.1
                            ),
                        });
                    },
                }
            } else {
                self.ran("official_pruned");
            }
        }

        // This crate's answers, its own pruning and pushdown included,
        // against the truth.
        for (query, truth) in queries.iter().zip(&truth) {
            let got = self.zeros(
                index,
                query.column.as_ref(),
                self.lake.rows(None, &query.ours).await,
            );
            let result = compare("crate_reads", step, truth, &got);
            if result.is_err() && self.rounded_filter(index, query) {
                self.skip("crate_reads", "official's rounded inlined DOUBLE");
                continue;
            }
            self.ran("crate_reads");
            result.map_err(|m| Mismatch {
                detail: format!("`{}`: {}", query.ours, m.detail),
                ..m
            })?;
        }

        // DuckDB's answers over the crate's catalog.
        let Some(copy) = self.lake.duckdb_copy(&self.dir).await else {
            // A metadata server: only this crate's statistics, which must
            // cover its rows.
            if let Some(remote) = self.lake.remote.clone() {
                let attach = match &remote {
                    Metadata::Mysql {
                        host,
                        port,
                        database,
                    } => {
                        crate::common::ensure_extension_installed("mysql");
                        format!(
                            "LOAD mysql; ATTACH 'db={database} host={host} port={port} user=root' \
                             AS crate_meta (TYPE mysql, READ_ONLY)"
                        )
                    },
                    Metadata::Postgres(dsn) => {
                        crate::common::ensure_extension_installed("postgres");
                        format!(
                            "LOAD postgres; ATTACH '{dsn}' AS crate_meta (TYPE postgres, READ_ONLY)"
                        )
                    },
                    _ => unreachable!("remote metadata is MySQL or PostgreSQL"),
                };
                self.official
                    .connection
                    .execute_batch(&attach)
                    .map_err(|e| Mismatch {
                        kind: "truth",
                        step,
                        detail: e.to_string(),
                    })?;
                for column in &columns {
                    if known::inlined_double_text(self.history, index, column) {
                        self.skip("crate_stats_cover", "official's rounded inlined DOUBLE");
                        continue;
                    }
                    self.ran("crate_stats_cover");
                    let covered = self.covers(column, "crate_meta").map_err(|e| Mismatch {
                        kind: "truth",
                        step,
                        detail: e,
                    })?;
                    if !covered {
                        return Err(Mismatch {
                            kind: "crate_stats_cover",
                            step,
                            detail: format!(
                                "the crate's statistics of {} do not cover its values: {:?}",
                                column.name,
                                Self::stats_rows(&remote).0
                            ),
                        });
                    }
                }
                let _ = self.official.connection.execute_batch("DETACH crate_meta");
                // MySQL writes to Parquet what official inlines.
                let diverged = if known::mysql_fallback(self.history, step, self.lake.kind) {
                    Some("MySQL fallback to Parquet")
                } else if known::stale_inlined_partition(self.history, step) {
                    Some("official's flush of a stale inlined table")
                } else {
                    self.layout_diverged
                };
                if let Some(reason) = diverged {
                    self.skip("partition_layout", reason);
                    return Ok(());
                }
                self.compare_layout(step, &remote, Self::layout_rows(&remote))?;
            }
            return Ok(());
        };
        let reader = duckdb_session();
        reader
            .execute_batch(&format!(
                "ATTACH '{}' AS lake (DATA_PATH '{}/', READ_ONLY); USE lake;",
                copy.ducklake_uri(),
                self.lake.data.display()
            ))
            .map_err(|e| Mismatch {
                kind: "duckdb_attach",
                step,
                detail: e.to_string(),
            })?;
        for ((query, truth), pruned) in queries.iter().zip(&truth).zip(&pruned) {
            if trace {
                eprintln!("duckdb over the crate: {}", query.on("t"));
            }
            let got = self.zeros(
                index,
                query.column.as_ref(),
                Official::rows(&reader, &query.on("t")),
            );
            // DuckDB prunes the crate's catalog with official's bugs, or not,
            // depending on the files; either answer official could give is
            // accepted.
            let accepted = |answer: &Rows| match (answer, &got) {
                (Ok(a), Ok(g)) => a.1 == g.1,
                (Err(a), Err(g)) => error_class(a) == error_class(g),
                _ => false,
            };
            if accepted(pruned) {
                self.ran("duckdb_reads");
                continue;
            }
            if explained(query, self).is_some() && accepted(truth) {
                self.ran("duckdb_reads");
                continue;
            }
            let column = query.column.as_ref();
            if let Some(probe) = &query.probe
                && let Some(column) = column
                && (known::float_exponent_filter(column.ty, probe)
                    || known::double_fixed_point_filter(column.ty, probe))
            {
                self.skip("duckdb_reads", "DuckDB float constant pruning");
                continue;
            }
            if self.rounded_filter(index, query) {
                self.skip("duckdb_reads", "official's rounded inlined DOUBLE");
                continue;
            }
            if (query.template.contains("WHERE") || query.template.contains("min("))
                && let Some(column) = column
                && known::float_special(self.history, step, column)
            {
                self.skip("duckdb_reads", "crate footer NaN/-0.0");
                continue;
            }
            return Err(Mismatch {
                kind: "duckdb_reads",
                step,
                detail: format!(
                    "`{}`: \nofficial: {}\ntruth:    {}\ncrate:    {}",
                    query.on("t"),
                    pruned
                        .as_ref()
                        .map_or_else(|e| format!("error {e}"), describe),
                    truth
                        .as_ref()
                        .map_or_else(|e| format!("error {e}"), describe),
                    got.as_ref().map_or_else(|e| format!("error {e}"), describe),
                ),
            });
        }
        drop(reader);

        // This crate's statistics must cover its rows, always.
        let our_stats = Self::stats_rows(&copy);
        let attach = match &copy {
            Metadata::Sqlite(path) => {
                format!(
                    "ATTACH '{}' AS crate_meta (TYPE sqlite, READ_ONLY)",
                    path.display()
                )
            },
            Metadata::Duckdb(path) => {
                format!("ATTACH '{}' AS crate_meta (READ_ONLY)", path.display())
            },
            _ => unreachable!("only embedded catalogs are copied"),
        };
        self.official
            .connection
            .execute_batch(&attach)
            .map_err(|e| Mismatch {
                kind: "truth",
                step,
                detail: e.to_string(),
            })?;
        for column in &columns {
            // The truth holds official's rounded values, not this crate's.
            if known::inlined_double_text(self.history, index, column) {
                self.skip("crate_stats_cover", "official's rounded inlined DOUBLE");
                continue;
            }
            let covered = self.covers(column, "crate_meta").map_err(|e| Mismatch {
                kind: "truth",
                step,
                detail: e,
            })?;
            self.ran("crate_stats_cover");
            if !covered {
                return Err(Mismatch {
                    kind: "crate_stats_cover",
                    step,
                    detail: format!(
                        "the crate's statistics of {} do not cover its values: {:?}",
                        column.name, our_stats.0
                    ),
                });
            }
        }

        match self.table_stats_diverged {
            Some(reason) => self.skip("table_stats", reason),
            None => {
                self.ran("table_stats");
                compare("table_stats", step, &official_stats.1, &our_stats.1)?;
            },
        }
        let diverged = &self.stats_diverged;
        let mut skipped = Vec::new();
        let filter = |rows: &Rows, skipped: &mut Vec<&'static str>| {
            rows.clone().map(|(names, rows)| {
                let mut rows = rows
                    .into_iter()
                    .filter(|row| {
                        let name = row[0].trim_matches('"');
                        let rounded = columns.iter().any(|c| {
                            c.name == name && known::inlined_double_text(self.history, index, c)
                        });
                        let reason = self
                            .owners(name)
                            .iter()
                            .find_map(|owner| diverged.get(owner).copied())
                            .or(rounded.then_some("official's rounded inlined DOUBLE"));
                        if let Some(reason) = reason {
                            skipped.push(reason);
                        }
                        reason.is_none()
                    })
                    .collect::<Vec<_>>();
                rows.sort();
                (names, rows)
            })
        };
        let mut official_rows = filter(&official_stats.0, &mut skipped);
        let our_rows = filter(&our_stats.0, &mut Vec::new());
        // A float column that held NaN: a commit merging a file without
        // bounds leaves this crate no row (see `merge_table_column_stats`),
        // where official can keep one. No row claims nothing, so official's
        // row there is not compared.
        if let (Ok((_, theirs)), Ok((_, ours))) = (&mut official_rows, &our_rows) {
            let nan_columns: Vec<String> = columns
                .iter()
                .filter(|c| {
                    c.ty.is_float()
                        && known::written(self.history, step, &c.name).iter().any(|v| {
                            matches!(v, V::F(f) if f.is_nan()) || matches!(v, V::R(f) if f.is_nan())
                        })
                })
                .map(|c| format!("{:?}", c.name))
                .collect();
            theirs.retain(|row| {
                let dropped =
                    nan_columns.contains(&row[0]) && !ours.iter().any(|ours| ours[0] == row[0]);
                if dropped {
                    self.skip(
                        "column_stats rows",
                        "crate keeps no row after a boundless NaN file",
                    );
                }
                !dropped
            });
        }
        for reason in skipped {
            self.skip("column_stats rows", reason);
        }
        if let Ok((_, rows)) = &official_rows {
            for _ in rows {
                self.ran("column_stats rows");
            }
        }
        compare("column_stats", step, &official_rows, &our_rows)?;

        if let Some(reason) = self.layout_diverged {
            self.skip("partition_layout", reason);
            return Ok(());
        }
        if known::stale_inlined_partition(self.history, step) {
            self.skip(
                "partition_layout",
                "official's flush of a stale inlined table",
            );
            return Ok(());
        }
        // The Hive directories and partition values of the live files. A
        // directory's value spells each character outside
        // `[A-Za-z0-9-_.:]` as `_` where official percent-encodes it (see
        // COMPATIBILITY.md), so official's directories are compared in that
        // spelling.
        self.compare_layout(step, &copy, Self::layout_rows(&copy))
    }

    /// Note a rewrite at `step` (or a checkpoint's) that wrote the same rows
    /// into more files here than official ([`known::rewrite_grouping`]).
    async fn note_rewrite_grouping(&mut self, step: usize) {
        let outputs = |metadata: &Metadata, heads: &[i64]| {
            let connection = duckdb_session();
            metadata.attach_raw(&connection);
            known::rewrite_outputs(&connection, heads[step], heads[step + 1])
        };
        let official = outputs(
            &Metadata::Duckdb(self.official.metadata.clone()),
            &self.heads.official,
        );
        let ours = match self.lake.duckdb_copy(&self.dir).await {
            Some(copy) => copy,
            None => match &self.lake.remote {
                Some(remote) => remote.clone(),
                None => return,
            },
        };
        let ours = outputs(&ours, &self.heads.ours);
        if known::rewrite_grouping(official, ours) {
            self.rewrite_grouped = Some(step);
        }
    }

    /// Official's [`Self::layout_rows`], its directories in this crate's
    /// spelling ([`crate_hive_spelling`]).
    fn official_layout(&self) -> Rows {
        Self::crate_spelling(Self::layout_rows(&Metadata::Duckdb(
            self.official.metadata.clone(),
        )))
    }

    /// Official's layout `rows`, its directories in this crate's spelling
    /// ([`crate_hive_spelling`]).
    fn crate_spelling(rows: Rows) -> Rows {
        rows.map(|(names, rows)| {
            let mut rows: Vec<Vec<String>> = rows
                .into_iter()
                .map(|mut row| {
                    row[0] = format!("{:?}", crate_hive_spelling(row[0].trim_matches('"')));
                    row
                })
                .collect();
            rows.sort();
            rows.dedup();
            (names, rows)
        })
    }

    /// The distinct `(directory, partition key index, partition value)` of
    /// the live data files of `metadata`'s table `t`: a file's directory is
    /// its path relative to the table without the file name.
    fn layout_rows(metadata: &Metadata) -> Rows {
        let connection = duckdb_session();
        metadata.attach_raw(&connection);
        Official::rows(
            &connection,
            "SELECT DISTINCT regexp_replace(f.path, '[^/]*$', '') AS dir, \
                pv.partition_key_index, pv.partition_value \
             FROM ducklake_data_file f JOIN ducklake_table t USING (table_id) \
             LEFT JOIN ducklake_file_partition_value pv USING (data_file_id) \
             WHERE t.table_name = 't' AND t.end_snapshot IS NULL \
               AND f.end_snapshot IS NULL AND f.path_is_relative \
             ORDER BY ALL",
        )
    }
}

/// The columns whose cells compare normalized, by name (every column for
/// `*`).
#[derive(Default)]
struct Cells {
    /// `-0.0` reads as `0.0`.
    signless: Vec<String>,
    /// A double reads as official's inlined writes round it
    /// ([`known::duckdb_decimal_double`]).
    rounded: Vec<String>,
    /// The canonical cells of the values those writes rounded
    /// ([`known::inlined_double_keys`]): only such a cell is rounded.
    rounded_keys: Vec<String>,
}

/// `rows` with the cells of [`Cells`]' columns normalized, sorted again.
fn normalize_cells(
    rows: (Vec<String>, Vec<Vec<String>>),
    cells: &Cells,
) -> (Vec<String>, Vec<Vec<String>>) {
    let rows = signless_zeros(rows, &cells.signless);
    if cells.rounded.is_empty() {
        return rows;
    }
    let (names, rows) = rows;
    let all = cells.rounded.iter().any(|c| c == "*");
    let chosen: Vec<bool> = names
        .iter()
        .map(|name| all || cells.rounded.contains(name))
        .collect();
    let mut rows: Vec<Vec<String>> = rows
        .into_iter()
        .map(|row| {
            row.into_iter()
                .enumerate()
                .map(|(index, cell)| {
                    let key = known::duckdb_decimal_double(&cell);
                    if chosen.get(index).copied().unwrap_or(all)
                        && cells.rounded_keys.contains(&key)
                    {
                        key
                    } else {
                        cell
                    }
                })
                .collect()
        })
        .collect();
    rows.sort();
    (names, rows)
}

/// `rows` with each `-0.0` cell of the named columns (every column for
/// `*`) read as `0.0`, sorted again.
fn signless_zeros(
    (names, rows): (Vec<String>, Vec<Vec<String>>),
    columns: &[String],
) -> (Vec<String>, Vec<Vec<String>>) {
    if columns.is_empty() {
        return (names, rows);
    }
    let all = columns.iter().any(|c| c == "*");
    let chosen: Vec<bool> = names
        .iter()
        .map(|name| all || columns.contains(name))
        .collect();
    let mut rows: Vec<Vec<String>> = rows
        .into_iter()
        .map(|row| {
            row.into_iter()
                .enumerate()
                .map(|(index, cell)| {
                    if cell == "-0.0" && chosen.get(index).copied().unwrap_or(all) {
                        "0.0".to_string()
                    } else {
                        cell
                    }
                })
                .collect()
        })
        .collect();
    rows.sort();
    (names, rows)
}

/// `rows` without its `rowid` column, sorted again, when `blind`. A change
/// feed's update pairs are then a deletion and an insertion: official pairs
/// them by row id, which its merge renumbered.
fn without_row_ids(
    (names, rows): (Vec<String>, Vec<Vec<String>>),
    blind: bool,
) -> (Vec<String>, Vec<Vec<String>>) {
    let Some(index) = names
        .iter()
        .position(|name| name == "rowid")
        .filter(|_| blind)
    else {
        return (names, rows);
    };
    let kind = names.iter().position(|name| name == "change_type");
    let rows: Vec<Vec<String>> = rows
        .into_iter()
        .map(|mut row| {
            if let Some(kind) = kind {
                row[kind] = match row[kind].as_str() {
                    "\"update_preimage\"" => "\"delete\"".to_string(),
                    "\"update_postimage\"" => "\"insert\"".to_string(),
                    _ => row[kind].clone(),
                };
            }
            row
        })
        .collect();
    let mut names = names;
    names.remove(index);
    let mut rows: Vec<Vec<String>> = rows
        .into_iter()
        .map(|mut row| {
            row.remove(index);
            row
        })
        .collect();
    rows.sort();
    (names, rows)
}

fn compare(
    kind: &'static str,
    step: usize,
    official: &Result<(Vec<String>, Vec<Vec<String>>), String>,
    ours: &Result<(Vec<String>, Vec<Vec<String>>), String>,
) -> Result<(), Mismatch> {
    match (official, ours) {
        // Each engine names an aggregate its own way.
        (Ok(o), Ok(c))
            if o.1 == c.1 && (o.0 == c.0 || o.1.is_empty() || kind.starts_with("crate_reads")) =>
        {
            Ok(())
        },
        // Both failed: the same kind of failure.
        (Err(o), Err(c)) if error_class(o) == error_class(c) => Ok(()),
        _ => Err(Mismatch {
            kind,
            step,
            detail: format!(
                "\nofficial: {}\ncrate:    {}",
                official
                    .as_ref()
                    .map_or_else(|e| format!("error {e}"), describe),
                ours.as_ref()
                    .map_or_else(|e| format!("error {e}"), describe)
            ),
        }),
    }
}

// ---------------------------------------------------------------------------
// Running
// ---------------------------------------------------------------------------

/// Run `history` on a fresh pair of catalogs and return the first mismatch.
async fn run_history(
    history: &History,
    backend: &Backend,
    seed: u64,
    exhaustive: bool,
    record: bool,
) -> Result<(), Mismatch> {
    run_history_tallied(history, backend, seed, exhaustive, record)
        .await
        .map(|_| ())
}

/// [`run_history`], returning the comparisons the run made and skipped
/// ([`Run::tally`]).
async fn run_history_tallied(
    history: &History,
    backend: &Backend,
    seed: u64,
    exhaustive: bool,
    record: bool,
) -> Result<Vec<String>, Mismatch> {
    let temp = tempfile::TempDir::new().unwrap();
    let dir = temp.path().to_path_buf();
    // `FUZZ_KEEP` keeps the catalogs of the run for inspection.
    let _temp = if std::env::var("FUZZ_KEEP").is_ok() {
        eprintln!("keeping {}", dir.display());
        let _ = temp.keep();
        None
    } else {
        Some(temp)
    };
    let official_data = dir.join("official_data");
    std::fs::create_dir_all(&official_data).unwrap();
    let official = Official::new(dir.join("official.ducklake"), &official_data);
    let lake = backend.lake(&dir, seed).await;

    // The table, on both sides.
    let columns = std::iter::once("id BIGINT".to_string())
        .chain(
            history
                .columns
                .iter()
                .map(|c| format!("{} {}", c.name, c.ty.duckdb())),
        )
        .collect::<Vec<_>>();
    official
        .run(&format!("CREATE TABLE t ({})", columns.join(", ")))
        .map_err(|e| Mismatch {
            kind: "setup",
            step: 0,
            detail: e,
        })?;
    if let Some(limit) = history.limit {
        official
            .run(&format!(
                "CALL lake.set_option('data_inlining_row_limit', {limit}, table_name => 't')"
            ))
            .unwrap();
    }
    lake.create(history).await.map_err(|e| Mismatch {
        kind: "setup",
        step: 0,
        detail: e,
    })?;
    // A short history is cheap to check at every snapshot and window.
    let exhaustive = exhaustive || history.steps.len() <= 5;
    let mut run = Run {
        history,
        heads: Heads {
            official: vec![official.head()],
            ours: vec![lake.head().await],
        },
        official,
        lake,
        dir,
        rng: Rng::new(seed ^ 0xC0FF_EE00),
        exhaustive,
        readable_from: 0,
        record,
        stats_diverged: Default::default(),
        table_stats_diverged: None,
        official_failed: Default::default(),
        blind: false,
        layout_diverged: None,
        rewrite_grouped: None,
        tally: Default::default(),
    };
    for step in 0..history.steps.len() {
        if std::env::var("FUZZ_TRACE").is_ok() {
            eprintln!("step {step}: {}", step_source(&history.steps[step]));
        }
        run.official.reconnect();
        // A PostgreSQL catalog refuses a NUL byte in a partition value that
        // official stores; such a step is undone on official.
        let undoable = matches!(run.lake.kind, Kind::PostgresSingle | Kind::Multicatalog);
        if undoable {
            run.official.save();
        }
        let rewrite = matches!(history.steps[step], Step::Rewrite | Step::Checkpoint);
        let before =
            rewrite.then(|| Run::stats_rows(&Metadata::Duckdb(run.official.metadata.clone())));
        let official_result = run.official.apply(history, step);
        let ours = run.lake.apply(history, step).await;
        run.ran(&format!("step {}", step_kind(&history.steps[step])));
        match (&official_result, &ours) {
            (Ok(()), Ok(())) => {},
            (Err(official), Err(crate_error)) => {
                // A NUL byte in a partition value: official's flush fails to
                // parse its own SQL (`known::nul_partition_parser_error`) where
                // PostgreSQL refuses the value (`known::postgres_nul_partition`).
                // Both refuse the step.
                let nul_refusals = known::nul_partition_parser_error(history, step, official)
                    && known::postgres_nul_partition(history, step, run.lake.kind, crate_error);
                if nul_refusals {
                    run.skip("step outcome", "NUL partition value refused on both");
                } else if error_class(official) != error_class(crate_error) {
                    return Err(Mismatch {
                        kind: "outcome",
                        step,
                        detail: format!(
                            "both failed, differently ({} vs {})\nofficial: {official}\ncrate: {crate_error}",
                            error_class(official),
                            error_class(crate_error)
                        ),
                    });
                }
                run.ran(&format!("both failed: {}", error_class(official)));
            },
            (Ok(()), Err(error))
                if undoable
                    && known::postgres_nul_partition(history, step, run.lake.kind, error) =>
            {
                run.official.restore();
                run.skip(
                    "step outcome",
                    "PostgreSQL NUL partition value: undone on official",
                );
            },
            (Err(error), Ok(()))
                if matches!(history.steps[step], Step::Flush | Step::Checkpoint)
                    && known::nul_partition_parser_error(history, step, error) =>
            {
                // A flush changes no row: the catalogs hold the same rows,
                // laid out apart from here on.
                let reason = "official's flush refused a NUL partition value";
                run.official_failed.insert(step);
                run.layout_diverged = Some(reason);
                run.table_stats_diverged = Some(reason);
                for column in history.columns_at(step) {
                    run.stats_diverged.entry(column.name).or_insert(reason);
                }
                run.skip("step outcome", reason);
            },
            (Err(error), Ok(()))
                if known::stale_inlined_table(history, step, error, run.lake.kind)
                    || known::nul_partition_parser_error(history, step, error) =>
            {
                // Official refused the rows this crate committed: delete them
                // again, so both catalogs hold the same rows. Row ids,
                // statistics and the feeds over this step no longer compare.
                let Step::Insert(rows) = &history.steps[step] else {
                    unreachable!("the model matches inserts only");
                };
                let ids = rows
                    .iter()
                    .map(|(id, _)| id.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                run.lake
                    .sql(&format!("DELETE FROM lake.main.t WHERE id IN ({ids})"))
                    .await
                    .map_err(|e| Mismatch {
                        kind: "outcome",
                        step,
                        detail: format!("undoing the insert official refused: {e}"),
                    })?;
                run.official_failed.insert(step);
                run.blind = true;
                run.table_stats_diverged = Some("insert official refused, undone");
                for column in history.columns_at(step) {
                    run.stats_diverged
                        .entry(column.name)
                        .or_insert("insert official refused, undone");
                }
                run.stats_diverged
                    .entry("id".to_string())
                    .or_insert("insert official refused, undone");
                run.skip(
                    "step outcome",
                    "official refused the insert: undone on the crate",
                );
            },
            (Err(error), Ok(()))
                if known::stats_recompute_error(history, step, error)
                    || known::max_timestamptz_recompute_error(history, step, error) =>
            {
                // A rewrite changes no row: the catalogs still hold the same
                // rows, laid out apart from here on (official keeps the files
                // this crate rewrote or dropped), and the change feeds over
                // this step differ.
                run.official_failed.insert(step);
                let reason = if known::stats_recompute_error(history, step, error) {
                    "#1490 official rewrite fails"
                } else {
                    "official rewrite fails on the greatest TIMESTAMPTZ"
                };
                run.layout_diverged = Some(reason);
                run.skip("step outcome", reason);
            },
            _ => {
                return Err(Mismatch {
                    kind: "outcome",
                    step,
                    detail: format!("official: {:?}\ncrate: {:?}", official_result, ours),
                });
            },
        }
        if let Some(before) = before
            && official_result.is_ok()
        {
            let after = Run::stats_rows(&Metadata::Duckdb(run.official.metadata.clone()));
            run.note_rewrite_recompute(&before, &after);
        }
        run.heads.official.push(run.official.head());
        run.heads.ours.push(run.lake.head().await);
        if rewrite && official_result.is_ok() && ours.is_ok() {
            run.note_rewrite_grouping(step).await;
        }
        if matches!(history.steps[step], Step::Expire | Step::Checkpoint) {
            run.readable_from = step + 1;
        }
        run.compare_step(step).await?;
    }
    Ok(std::mem::take(&mut *run.tally.lock().unwrap()))
}

/// Official's percent-encoded Hive directory `dir` (`key=value/…/`) in this
/// crate's spelling: each value decoded, then each character outside
/// `[A-Za-z0-9-_.:]` spelled `_`.
fn crate_hive_spelling(dir: &str) -> String {
    let decode = |text: &str| {
        let bytes = text.as_bytes();
        let mut out = Vec::with_capacity(bytes.len());
        let mut index = 0;
        while index < bytes.len() {
            if bytes[index] == b'%'
                && index + 2 < bytes.len()
                && let Ok(byte) = u8::from_str_radix(&text[index + 1..index + 3], 16)
            {
                out.push(byte);
                index += 3;
            } else {
                out.push(bytes[index]);
                index += 1;
            }
        }
        String::from_utf8_lossy(&out).into_owned()
    };
    dir.split('/')
        .map(|segment| match segment.split_once('=') {
            Some((key, value)) if value != "__HIVE_DEFAULT_PARTITION__" => {
                let value: String = decode(value)
                    .chars()
                    .map(|c| {
                        if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | ':') {
                            c
                        } else {
                            '_'
                        }
                    })
                    .collect();
                format!("{key}={value}")
            },
            _ => segment.to_string(),
        })
        .collect::<Vec<_>>()
        .join("/")
}

/// The kind of `step`, for the coverage counts.
fn step_kind(step: &Step) -> &'static str {
    match step {
        Step::Insert(_) => "insert",
        Step::Update {
            ..
        } => "update",
        Step::Delete(None) => "delete all",
        Step::Delete(Some(_)) => "delete",
        Step::Flush => "flush",
        Step::Checkpoint => "checkpoint",
        Step::Merge => "merge",
        Step::Rewrite => "rewrite",
        Step::AddColumn(..) => "add column",
        Step::Widen(_) => "widen",
        Step::Limit(_) => "limit",
        Step::Partition(_) => "partition",
        Step::Expire => "expire",
    }
}

/// Smaller histories of the same seed, most aggressive first.
fn shrink_candidates(history: &History) -> Vec<History> {
    let mut candidates = Vec::new();
    // Drop one step. Dropping an added column drops its values from later
    // inserts, and is skipped while a later step names it.
    for index in 0..history.steps.len() {
        let mut candidate = history.clone();
        if let Step::AddColumn(column, _) = &history.steps[index] {
            if history.steps[index + 1..]
                .iter()
                .any(|step| step_names(step, &column.name))
            {
                continue;
            }
            let position = history.columns.len()
                + history.steps[..index]
                    .iter()
                    .filter(|step| matches!(step, Step::AddColumn(..)))
                    .count();
            for step in &mut candidate.steps[index + 1..] {
                if let Step::Insert(rows) = step {
                    for (_, values) in rows {
                        if position < values.len() {
                            values.remove(position);
                        }
                    }
                }
            }
        }
        candidate.steps.remove(index);
        candidates.push(candidate);
    }
    // Drop rows of an insert.
    for (index, step) in history.steps.iter().enumerate() {
        if let Step::Insert(rows) = step
            && rows.len() > 1
        {
            for keep in [rows.len() / 2, rows.len() - 1] {
                let mut candidate = history.clone();
                candidate.steps[index] = Step::Insert(rows[..keep].to_vec());
                candidates.push(candidate);
                let mut candidate = history.clone();
                candidate.steps[index] = Step::Insert(rows[rows.len() - keep..].to_vec());
                candidates.push(candidate);
            }
        }
    }
    // Drop a created column no step names.
    for index in 0..history.columns.len() {
        let name = &history.columns[index].name;
        let named = history.steps.iter().any(|step| step_names(step, name));
        if named || history.columns.len() == 1 {
            continue;
        }
        let mut candidate = history.clone();
        candidate.columns.remove(index);
        for step in &mut candidate.steps {
            if let Step::Insert(rows) = step {
                for (_, values) in rows {
                    values.remove(index);
                }
            }
        }
        candidates.push(candidate);
    }
    // Simplify values to NULL.
    for (index, step) in history.steps.iter().enumerate() {
        if let Step::Insert(rows) = step {
            for (row, (_, values)) in rows.iter().enumerate() {
                for (column, value) in values.iter().enumerate() {
                    if *value != V::Null {
                        let mut candidate = history.clone();
                        if let Step::Insert(rows) = &mut candidate.steps[index] {
                            rows[row].1[column] = V::Null;
                        }
                        candidates.push(candidate);
                    }
                }
            }
        }
    }
    candidates
}

/// Whether `step` names column `name`.
fn step_names(step: &Step, name: &str) -> bool {
    match step {
        Step::Update {
            column,
            pred,
            ..
        } => column == name || pred_names(pred, name),
        Step::Delete(Some(pred)) => pred_names(pred, name),
        Step::Widen(column) => column == name,
        Step::Partition(Some((column, _))) => column == name,
        _ => false,
    }
}

fn pred_names(pred: &Pred, name: &str) -> bool {
    matches!(pred, Pred::IsNull(c) | Pred::Eq(c, _, _) if c == name)
}

/// Shrink a failing history, comparing exhaustively; `None` when the
/// exhaustive run passes (the failure did not reproduce).
async fn shrink(
    history: History,
    backend: &Backend,
    seed: u64,
    kind: &'static str,
) -> Option<(History, Mismatch)> {
    // `FUZZ_SHRINK_SECS` (default 240) bounds the search; the smallest
    // history found by then is reported.
    let budget = std::time::Duration::from_secs(
        std::env::var("FUZZ_SHRINK_SECS")
            .ok()
            .and_then(|n| n.parse().ok())
            .unwrap_or(240),
    );
    let started = std::time::Instant::now();
    let mut best = history;
    let mut mismatch = run_history(&best, backend, seed, true, false).await.err()?;
    'outer: loop {
        for candidate in shrink_candidates(&best) {
            if started.elapsed() > budget {
                break 'outer;
            }
            if let Err(m) = run_history(&candidate, backend, seed, true, false).await
                && m.kind == kind
            {
                best = candidate;
                mismatch = m;
                continue 'outer;
            }
        }
        break;
    }
    Some((best, mismatch))
}

/// `history` as Rust source for a regression test.
fn as_test(history: &History, backend: Kind, mismatch: &Mismatch) -> String {
    let mut text = String::new();
    let _ = writeln!(
        text,
        "/// {} mismatch at step {} on {backend:?}.\n#[tokio::test(flavor = \"multi_thread\")]\n\
         async fn regression_() {{\n    let history = History {{\n        columns: vec![{}],\n        \
         limit: {:?},\n        steps: vec![",
        mismatch.kind,
        mismatch.step,
        history
            .columns
            .iter()
            .map(|c| format!("Col {{ name: {:?}.into(), ty: Ty::{:?} }}", c.name, c.ty))
            .collect::<Vec<_>>()
            .join(", "),
        history.limit
    );
    for step in &history.steps {
        let _ = writeln!(text, "            {},", step_source(step));
    }
    let _ = writeln!(
        text,
        "        ],\n    }};\n    check(&history, Kind::{backend:?}).await;\n}}"
    );
    text
}

fn value_source(value: &V) -> String {
    match value {
        V::S(s) => format!("V::S({s:?}.into())"),
        V::F(f) if f.is_nan() => "V::F(f64::NAN)".into(),
        V::F(f) if f.is_infinite() => format!(
            "V::F({}f64::INFINITY)",
            if *f < 0.0 {
                "-"
            } else {
                ""
            }
        ),
        V::F(f) => format!("V::F({f:?})"),
        V::R(f) if f.is_nan() => "V::R(f32::NAN)".into(),
        V::R(f) if f.is_infinite() => format!(
            "V::R({}f32::INFINITY)",
            if *f < 0.0 {
                "-"
            } else {
                ""
            }
        ),
        V::R(f) => format!("V::R({f:?})"),
        V::Map(entries) => format!(
            "V::Map(vec![{}])",
            entries
                .iter()
                .map(|(k, v)| format!("({k:?}.into(), {v:?})"))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        V::Struct(a, b) => format!(
            "V::Struct({a:?}, {})",
            b.as_ref()
                .map_or("None".into(), |b| format!("Some({b:?}.into())"))
        ),
        V::List(items) => format!("V::List(vec!{items:?})"),
        V::Bytes(bytes) => format!("V::Bytes(vec!{bytes:?})"),
        other => format!("V::{other:?}"),
    }
}

fn pred_source(pred: &Pred) -> String {
    match pred {
        Pred::IsNull(c) => format!("Pred::IsNull({c:?}.into())"),
        Pred::Eq(c, ty, v) => format!("Pred::Eq({c:?}.into(), Ty::{ty:?}, {})", value_source(v)),
        Pred::IdIn(ids) => format!("Pred::IdIn(vec!{ids:?})"),
        other => format!("Pred::{other:?}"),
    }
}

fn step_source(step: &Step) -> String {
    match step {
        Step::Insert(rows) => format!(
            "Step::Insert(vec![{}])",
            rows.iter()
                .map(|(id, values)| format!(
                    "({id}, vec![{}])",
                    values
                        .iter()
                        .map(value_source)
                        .collect::<Vec<_>>()
                        .join(", ")
                ))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Step::Update {
            column,
            ty,
            value,
            pred,
        } => format!(
            "Step::Update {{ column: {column:?}.into(), ty: Ty::{ty:?}, value: {}, pred: {} }}",
            value_source(value),
            pred_source(pred)
        ),
        Step::Delete(pred) => format!(
            "Step::Delete({})",
            pred.as_ref()
                .map_or("None".into(), |p| format!("Some({})", pred_source(p)))
        ),
        Step::AddColumn(c, default) => format!(
            "Step::AddColumn(Col {{ name: {:?}.into(), ty: Ty::{:?} }}, {})",
            c.name,
            c.ty,
            default
                .as_ref()
                .map_or("None".into(), |d| format!("Some({d:?}.into())"))
        ),
        Step::Widen(c) => format!("Step::Widen({c:?}.into())"),
        Step::Partition(p) => format!(
            "Step::Partition({})",
            p.as_ref().map_or("None".into(), |(c, t)| format!(
                "Some(({c:?}.into(), Transform::{t:?}))"
            ))
        ),
        other => format!("Step::{other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Backends
// ---------------------------------------------------------------------------

/// How to make a fresh catalog of the backend under test.
enum Backend {
    Sqlite,
    Duckdb,
    #[cfg(feature = "write-mysql")]
    Mysql {
        port: u16,
    },
    #[cfg(feature = "write-postgres")]
    PostgresSingle {
        port: u16,
    },
    #[cfg(feature = "multicatalog-postgres")]
    Multicatalog {
        port: u16,
    },
}

static DATABASES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

impl Backend {
    fn kind(&self) -> Kind {
        match self {
            Backend::Sqlite => Kind::Sqlite,
            Backend::Duckdb => Kind::Duckdb,
            #[cfg(feature = "write-mysql")]
            Backend::Mysql {
                ..
            } => Kind::Mysql,
            #[cfg(feature = "write-postgres")]
            Backend::PostgresSingle {
                ..
            } => Kind::PostgresSingle,
            #[cfg(feature = "multicatalog-postgres")]
            Backend::Multicatalog {
                ..
            } => Kind::Multicatalog,
        }
    }

    async fn lake(&self, dir: &Path, _seed: u64) -> Lake {
        let data = dir.join("crate_data");
        std::fs::create_dir_all(&data).unwrap();
        let data_text = data.to_str().unwrap().to_string();
        let kind = self.kind();
        let mut remote = None;
        let (open, metadata): (Open, Option<PathBuf>) = match self {
            Backend::Sqlite => {
                use datafusion_ducklake::{SqliteMetadataProvider, SqliteMetadataWriter};
                let path = dir.join("crate.db");
                let url = format!("sqlite:{}?mode=rwc", path.display());
                let writer = SqliteMetadataWriter::new_with_init(&url).await.unwrap();
                writer.set_data_path(&data_text).unwrap();
                let snapshot = writer.create_snapshot().unwrap();
                writer.get_or_create_schema("main", None, snapshot).unwrap();
                let writer: Arc<dyn MetadataWriter> = Arc::new(writer);
                let provider: Arc<dyn MetadataProvider> =
                    Arc::new(SqliteMetadataProvider::new(&url).await.unwrap());
                (
                    Arc::new(move || {
                        let pair = (Arc::clone(&writer), Arc::clone(&provider));
                        Box::pin(async move { pair })
                    }),
                    Some(path),
                )
            },
            Backend::Duckdb => {
                use datafusion_ducklake::DuckdbMetadataWriter;
                let path = dir.join("crate.duckdb");
                {
                    let writer =
                        DuckdbMetadataWriter::new_with_init(path.display().to_string()).unwrap();
                    writer.set_data_path(&data_text).unwrap();
                    let snapshot = writer.create_snapshot().unwrap();
                    writer.get_or_create_schema("main", None, snapshot).unwrap();
                }
                let opened = path.clone();
                (
                    Arc::new(move || {
                        let path = opened.clone();
                        Box::pin(async move {
                            let writer =
                                DuckdbMetadataWriter::new_with_init(path.display().to_string())
                                    .unwrap();
                            let provider: Arc<dyn MetadataProvider> =
                                Arc::new(writer.metadata_provider());
                            let writer: Arc<dyn MetadataWriter> = Arc::new(writer);
                            (writer, provider)
                        })
                    }),
                    Some(path),
                )
            },
            #[cfg(feature = "write-mysql")]
            Backend::Mysql {
                port,
            } => {
                use datafusion_ducklake::{MySqlMetadataProvider, MySqlMetadataWriter};
                let database = format!(
                    "fuzz_{}",
                    DATABASES.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                );
                let admin =
                    sqlx::MySqlPool::connect(&format!("mysql://root@127.0.0.1:{port}/test"))
                        .await
                        .unwrap();
                sqlx::query(sqlx::AssertSqlSafe(format!("CREATE DATABASE {database}")))
                    .execute(&admin)
                    .await
                    .unwrap();
                admin.close().await;
                let url = format!("mysql://root@127.0.0.1:{port}/{database}");
                remote = Some(Metadata::Mysql {
                    host: "127.0.0.1".to_string(),
                    port: *port,
                    database: database.clone(),
                });
                let writer = MySqlMetadataWriter::new_with_init(&url).await.unwrap();
                writer.set_data_path(&data_text).unwrap();
                let snapshot = writer.create_snapshot().unwrap();
                writer.get_or_create_schema("main", None, snapshot).unwrap();
                let writer: Arc<dyn MetadataWriter> = Arc::new(writer);
                let provider: Arc<dyn MetadataProvider> =
                    Arc::new(MySqlMetadataProvider::new(&url).await.unwrap());
                (
                    Arc::new(move || {
                        let pair = (Arc::clone(&writer), Arc::clone(&provider));
                        Box::pin(async move { pair })
                    }),
                    None,
                )
            },
            #[cfg(feature = "write-postgres")]
            Backend::PostgresSingle {
                port,
            } => {
                use datafusion_ducklake::{
                    PostgresMetadataProvider, PostgresSingleCatalogMetadataWriter,
                };
                let database = format!(
                    "fuzz_{}",
                    DATABASES.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                );
                let admin = sqlx::PgPool::connect(&format!(
                    "postgresql://postgres:postgres@127.0.0.1:{port}/postgres"
                ))
                .await
                .unwrap();
                sqlx::query(sqlx::AssertSqlSafe(format!("CREATE DATABASE {database}")))
                    .execute(&admin)
                    .await
                    .unwrap();
                admin.close().await;
                let url = format!("postgresql://postgres:postgres@127.0.0.1:{port}/{database}");
                remote = Some(Metadata::Postgres(format!(
                    "host=127.0.0.1 port={port} dbname={database} user=postgres password=postgres"
                )));
                let writer = PostgresSingleCatalogMetadataWriter::new_with_init(&url)
                    .await
                    .unwrap();
                writer.set_data_path(&data_text).unwrap();
                let snapshot = writer.create_snapshot().unwrap();
                writer.get_or_create_schema("main", None, snapshot).unwrap();
                let writer: Arc<dyn MetadataWriter> = Arc::new(writer);
                let provider: Arc<dyn MetadataProvider> =
                    Arc::new(PostgresMetadataProvider::new(&url).await.unwrap());
                (
                    Arc::new(move || {
                        let pair = (Arc::clone(&writer), Arc::clone(&provider));
                        Box::pin(async move { pair })
                    }),
                    None,
                )
            },
            #[cfg(feature = "multicatalog-postgres")]
            Backend::Multicatalog {
                port,
            } => {
                use datafusion_ducklake::{
                    MulticatalogManager, MulticatalogProvider, PostgresMetadataWriter,
                };
                let database = format!(
                    "fuzz_{}",
                    DATABASES.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                );
                let admin = sqlx::PgPool::connect(&format!(
                    "postgresql://postgres:postgres@127.0.0.1:{port}/postgres"
                ))
                .await
                .unwrap();
                sqlx::query(sqlx::AssertSqlSafe(format!("CREATE DATABASE {database}")))
                    .execute(&admin)
                    .await
                    .unwrap();
                admin.close().await;
                let url = format!("postgresql://postgres:postgres@127.0.0.1:{port}/{database}");
                let pool = sqlx::PgPool::connect(&url).await.unwrap();
                datafusion_ducklake::initialize_multicatalog_schema(&pool)
                    .await
                    .unwrap();
                let catalog = MulticatalogManager::new(pool.clone())
                    .create_catalog("lake")
                    .await
                    .unwrap();
                let writer = PostgresMetadataWriter::with_pool(pool.clone(), catalog)
                    .await
                    .unwrap();
                writer.set_data_path(&data_text).unwrap();
                let snapshot = writer.create_snapshot().unwrap();
                writer.get_or_create_schema("main", None, snapshot).unwrap();
                let writer: Arc<dyn MetadataWriter> = Arc::new(writer);
                let provider: Arc<dyn MetadataProvider> =
                    Arc::new(MulticatalogProvider::with_pool(pool, "lake").await.unwrap());
                // The store holds this one catalog, so its standard tables read
                // as the catalog's own for the statistics and layout checks.
                remote = Some(Metadata::Postgres(format!(
                    "host=127.0.0.1 port={port} dbname={database} user=postgres password=postgres"
                )));
                (
                    Arc::new(move || {
                        let pair = (Arc::clone(&writer), Arc::clone(&provider));
                        Box::pin(async move { pair })
                    }),
                    None,
                )
            },
        };
        Lake {
            kind,
            open,
            data,
            metadata,
            remote,
        }
    }
}

/// The seeds to run: `FUZZ_SEED` alone, else `FUZZ_SEEDS` (or `default`) of
/// them from `FUZZ_SEED_START` (default 0).
fn seeds(default: u64) -> std::ops::Range<u64> {
    if let Ok(seed) = std::env::var("FUZZ_SEED") {
        let seed = seed.parse().expect("FUZZ_SEED is a number");
        return seed..seed + 1;
    }
    let count = std::env::var("FUZZ_SEEDS")
        .ok()
        .and_then(|n| n.parse().ok())
        .unwrap_or(default);
    let start = std::env::var("FUZZ_SEED_START")
        .ok()
        .and_then(|n| n.parse().ok())
        .unwrap_or(0);
    start..start + count
}

/// Run every seed of `seeds` on `backend`, shrinking and reporting each
/// failure, and fail if any did.
async fn fuzz(backend: Backend, seeds: std::ops::Range<u64>) {
    let kind = backend.kind();
    let backend = Arc::new(backend);
    let workers = std::env::var("FUZZ_WORKERS")
        .ok()
        .and_then(|n| n.parse().ok())
        .unwrap_or(4usize);
    let next = Arc::new(std::sync::atomic::AtomicU64::new(seeds.start));
    let failures = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let mut handles = Vec::new();
    for _ in 0..workers {
        let backend = Arc::clone(&backend);
        let next = Arc::clone(&next);
        let failures = Arc::clone(&failures);
        let end = seeds.end;
        let seeds_start = seeds.start;
        // Each worker drives its seeds on a thread and runtime of its own: a
        // history holds DuckDB connections across awaits.
        handles.push(std::thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .build()
                .unwrap();
            runtime.block_on(async move {
                loop {
                    let seed = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if seed >= end {
                        break;
                    }
                    let history = generate(seed, kind);
                    if seed > seeds_start && (seed - seeds_start).is_multiple_of(50) {
                        eprintln!("{kind:?}: seed {seed} of {}..{end}", seeds_start);
                    }
                    if std::env::var("FUZZ_TRACE").is_ok() {
                        eprintln!("seed {seed} starts");
                    }
                    // A panic (a harness bug, or one in the crate) fails the
                    // seed, not the worker.
                    let outcome = futures::FutureExt::catch_unwind(std::panic::AssertUnwindSafe(
                        run_history(&history, &backend, seed, false, true),
                    ))
                    .await;
                    let first = match outcome {
                        Ok(Ok(())) => continue,
                        Ok(Err(first)) => first,
                        Err(panic) => {
                            let message = panic
                                .downcast_ref::<String>()
                                .cloned()
                                .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
                                .unwrap_or_default();
                            let report = format!(
                                "seed {seed} on {kind:?}: panicked: {message}\n{}",
                                as_test(
                                    &history,
                                    kind,
                                    &Mismatch {
                                        kind: "panic",
                                        step: 0,
                                        detail: message.clone(),
                                    }
                                )
                            );
                            eprintln!("{report}");
                            failures.lock().unwrap().push(report);
                            continue;
                        },
                    };
                    {
                        let report = match futures::FutureExt::catch_unwind(
                            std::panic::AssertUnwindSafe(shrink(
                                history.clone(),
                                &backend,
                                seed,
                                first.kind,
                            )),
                        )
                        .await
                        .ok()
                        .flatten()
                        {
                            Some((shrunk, mismatch)) => format!(
                                "seed {seed} on {kind:?}: {} at step {}: {}\n{}",
                                mismatch.kind,
                                mismatch.step,
                                mismatch.detail,
                                as_test(&shrunk, kind, &mismatch)
                            ),
                            None => format!(
                                "seed {seed} on {kind:?} (did not reproduce exhaustively): {} at \
                                 step {}: {}\n{}",
                                first.kind,
                                first.step,
                                first.detail,
                                as_test(&history, kind, &first)
                            ),
                        };
                        eprintln!("{report}");
                        failures.lock().unwrap().push(report);
                    }
                }
            });
        }));
    }
    for handle in handles {
        handle.join().unwrap();
    }
    eprintln!("{}", coverage::summary(kind));
    let failures = failures.lock().unwrap();
    assert!(
        failures.is_empty(),
        "{} of {} seeds failed on {kind:?}:\n{}",
        failures.len(),
        seeds.end - seeds.start,
        failures.join("\n\n")
    );
}

/// Run one history (a shrunk regression) and fail on any mismatch.
#[allow(dead_code)]
async fn check(history: &History, kind: Kind) {
    let backend = match kind {
        Kind::Sqlite => Backend::Sqlite,
        Kind::Duckdb => Backend::Duckdb,
        other => panic!("{other:?} needs a container"),
    };
    if let Err(mismatch) = run_history(history, &backend, 0, true, false).await {
        panic!(
            "{} at step {}: {}",
            mismatch.kind, mismatch.step, mismatch.detail
        );
    }
}

/// A sample of seeds for the default suite: `FUZZ_SEEDS` runs more, and
/// `soak_sqlite` runs 2000.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn fuzz_sqlite() {
    fuzz(Backend::Sqlite, seeds(8)).await;
}

/// A sample of seeds for the default suite: `FUZZ_SEEDS` runs more, and
/// `soak_duckdb` runs 2000.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn fuzz_duckdb() {
    fuzz(Backend::Duckdb, seeds(8)).await;
}

#[ignore = "soak run: FUZZ_SEEDS seeds (default 2000)"]
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn soak_sqlite() {
    fuzz(Backend::Sqlite, seeds(2000)).await;
}

#[ignore = "soak run: FUZZ_SEEDS seeds (default 2000)"]
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn soak_duckdb() {
    fuzz(Backend::Duckdb, seeds(2000)).await;
}

#[cfg(feature = "write-mysql")]
async fn mysql_container() -> (
    testcontainers::ContainerAsync<testcontainers_modules::mysql::Mysql>,
    u16,
) {
    use testcontainers::runners::AsyncRunner;
    let container = testcontainers_modules::mysql::Mysql::default()
        .start()
        .await
        .unwrap();
    let port = container.get_host_port_ipv4(3306).await.unwrap();
    (container, port)
}

#[cfg(feature = "write-postgres")]
async fn postgres_container() -> (
    testcontainers::ContainerAsync<testcontainers_modules::postgres::Postgres>,
    u16,
) {
    use testcontainers::runners::AsyncRunner;
    let container = testcontainers_modules::postgres::Postgres::default()
        .start()
        .await
        .unwrap();
    let port = container.get_host_port_ipv4(5432).await.unwrap();
    (container, port)
}

#[cfg(feature = "write-mysql")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn fuzz_mysql() {
    let (_container, port) = mysql_container().await;
    fuzz(
        Backend::Mysql {
            port,
        },
        // A sample for the default suite: `FUZZ_SEEDS` runs more, the soak 300.
        seeds(3),
    )
    .await;
}

#[cfg(feature = "write-postgres")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn fuzz_postgres_single_catalog() {
    let (_container, port) = postgres_container().await;
    fuzz(
        Backend::PostgresSingle {
            port,
        },
        // A sample for the default suite: `FUZZ_SEEDS` runs more, the soak 300.
        seeds(3),
    )
    .await;
}

#[cfg(feature = "multicatalog-postgres")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn fuzz_multicatalog() {
    let (_container, port) = postgres_container().await;
    fuzz(
        Backend::Multicatalog {
            port,
        },
        // A sample for the default suite: `FUZZ_SEEDS` runs more, the soak 300.
        seeds(3),
    )
    .await;
}

#[cfg(feature = "write-mysql")]
#[ignore = "soak run: FUZZ_SEEDS seeds (default 300)"]
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn soak_mysql() {
    let (_container, port) = mysql_container().await;
    fuzz(
        Backend::Mysql {
            port,
        },
        seeds(300),
    )
    .await;
}

#[cfg(feature = "write-postgres")]
#[ignore = "soak run: FUZZ_SEEDS seeds (default 300)"]
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn soak_postgres_single_catalog() {
    let (_container, port) = postgres_container().await;
    fuzz(
        Backend::PostgresSingle {
            port,
        },
        seeds(300),
    )
    .await;
}

#[cfg(feature = "multicatalog-postgres")]
#[ignore = "soak run: FUZZ_SEEDS seeds (default 300)"]
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn soak_multicatalog() {
    let (_container, port) = postgres_container().await;
    fuzz(
        Backend::Multicatalog {
            port,
        },
        seeds(300),
    )
    .await;
}

// ---------------------------------------------------------------------------
// Regressions found by the harness, shrunk
// ---------------------------------------------------------------------------

/// A partitioned insert's rows take row ids in the order official gives
/// them: an inlined insert in insert order, a Parquet one by partition in
/// first-seen order.
#[tokio::test(flavor = "multi_thread")]
async fn regression_partitioned_insert_row_ids() {
    let history = History {
        columns: vec![Col {
            name: "c0".into(),
            ty: Ty::TsTz,
        }],
        limit: Some(1),
        steps: vec![
            Step::Partition(Some(("c0".into(), Transform::Month))),
            Step::Insert(vec![
                (18, vec![V::Null]),
                (19, vec![V::Ts(1701011600000000)]),
                (20, vec![V::Ts(-9223372022400000000)]),
            ]),
        ],
    };
    check(&history, Kind::Sqlite).await;
    let history = History {
        columns: vec![Col {
            name: "c1".into(),
            ty: Ty::Date,
        }],
        limit: None,
        steps: vec![
            Step::Partition(Some(("c1".into(), Transform::Year))),
            Step::Insert(vec![
                (110, vec![V::Date(19020)]),
                (111, vec![V::Null]),
                (112, vec![V::Date(19027)]),
            ]),
        ],
    };
    check(&history, Kind::Duckdb).await;
}

/// String and blob statistics over 256 bytes: an inlined insert keeps the
/// whole value, a Parquet file DuckDB's truncated bounds, and a flush widens
/// the table's bounds by the file's.
#[tokio::test(flavor = "multi_thread")]
async fn regression_long_string_statistics() {
    let long = |fill: &str, count: usize, tail: &str| format!("{}{tail}", fill.repeat(count));
    let columns = vec![
        Col {
            name: "c0".into(),
            ty: Ty::Varchar,
        },
        Col {
            name: "c1".into(),
            ty: Ty::Blob,
        },
    ];
    let rows = vec![
        (1, vec![V::S(long("b", 70, "z")), V::Bytes(vec![b'b'; 70])]),
        (
            2,
            vec![V::S(long("b", 3000, "z")), V::Bytes(vec![b'b'; 3000])],
        ),
        (
            3,
            vec![V::S(long("a", 299, "y")), V::Bytes(vec![0xff; 300])],
        ),
    ];
    for limit in [Some(0), None] {
        let history = History {
            columns: columns.clone(),
            limit,
            steps: vec![Step::Insert(rows.clone()), Step::Flush],
        };
        check(&history, Kind::Sqlite).await;
    }
    // A maximum whose first 256 bytes cannot be incremented drops the file's
    // bounds for the column.
    let history = History {
        columns: vec![Col {
            name: "c0".into(),
            ty: Ty::Varchar,
        }],
        limit: Some(0),
        steps: vec![Step::Insert(vec![
            (1, vec![V::S("é".repeat(300))]),
            (2, vec![V::S("ê".into())]),
        ])],
    };
    check(&history, Kind::Sqlite).await;
}

/// Official's rewrite fails (#1490) after a column is added while rows are
/// inlined; this crate's rewrite drops the fully deleted files official
/// keeps, so the two layouts no longer compare.
#[tokio::test(flavor = "multi_thread")]
async fn regression_layout_after_official_rewrite_fails() {
    let history = History {
        columns: vec![Col { name: "c0".into(), ty: Ty::TsTz }, Col { name: "c1".into(), ty: Ty::Varchar }, Col { name: "c2".into(), ty: Ty::Ts }, Col { name: "c3".into(), ty: Ty::Double }],
        limit: Some(10),
        steps: vec![
            Step::Insert(vec![(4, vec![V::Ts(1702325600000000), V::S("a".into()), V::Ts(1702383200000000), V::F(0.0)]), (5, vec![V::Ts(1700522000000000), V::S("b".into()), V::Ts(1703564000000000), V::F(-3.0)]), (6, vec![V::Ts(1701926000000000), V::S("yy".into()), V::Ts(-65291356800000000), V::F(-3.5)]), (7, vec![V::Ts(1701429200000000), V::S("b".into()), V::Null, V::F(2.5)]), (8, vec![V::Ts(1701015200000000), V::S("x".into()), V::Ts(1701738800000000), V::F(1e-7)]), (9, vec![V::Ts(9223372036854775806), V::S("héllo ☃ 𝄞".into()), V::Ts(1703078000000000), V::F(2857966257206491.5)]), (10, vec![V::Ts(1702768400000000), V::S("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaéyy".into()), V::Ts(253402300800000000), V::F(3.5)]), (11, vec![V::Ts(-9223372036854775807), V::S("c".into()), V::Ts(-63517778999500000), V::F(0.25)]), (12, vec![V::Ts(1701094400000000), V::S("c".into()), V::Ts(1702808000000000), V::F(4.5)]), (13, vec![V::Ts(1702977200000000), V::S("".into()), V::Ts(1702804400000000), V::F(1.0)]), (14, vec![V::Null, V::S("".into()), V::Ts(1703423600000000), V::F(-0.5)]), (15, vec![V::Ts(-63517778999500000), V::S("x".into()), V::Ts(253402300800000000), V::Null]), (16, vec![V::Ts(1700846000000000), V::S("%_\\".into()), V::Null, V::F(-4.5)]), (17, vec![V::Ts(1700709200000000), V::S("c".into()), V::Ts(1701796400000000), V::F(4.25)]), (18, vec![V::Ts(9223372036854775806), V::S("c".into()), V::Ts(1703286800000000), V::F(0.25)]), (19, vec![V::Ts(1702836800000000), V::S("héllo ☃ 𝄞".into()), V::Null, V::F(1.7976931348623157e308)]), (20, vec![V::Ts(1702473200000000), V::S("zzz".into()), V::Ts(1702631600000000), V::F(2.25)]), (21, vec![V::Null, V::Null, V::Ts(1701357200000000), V::F(-0.0)]), (22, vec![V::Ts(1700475200000000), V::S("a\0b".into()), V::Ts(8458224206400123456), V::F(1.0)])]),
            Step::Flush,
            Step::Update { column: "c0".into(), ty: Ty::TsTz, value: V::Ts(1701188000000000), pred: Pred::IdIn(vec![21, 19, 10]) },
            Step::AddColumn(Col { name: "a1".into(), ty: Ty::Float }, None),
            Step::Insert(vec![(23, vec![V::Ts(-65291356800000000), V::S("zzz".into()), V::Ts(1701137600000000), V::F(-3.25), V::R(3.4028235e38)]), (24, vec![V::Ts(-63517778999500000), V::S("c".into()), V::Ts(1701522800000000), V::Null, V::R(1e-7)]), (25, vec![V::Ts(1703528000000000), V::S("zzz".into()), V::Ts(1701551600000000), V::F(-64844484111181.1), V::R(0.0)])]),
            Step::Insert(vec![(26, vec![V::Ts(-9223372022400000000), V::Null, V::Ts(1701026000000000), V::F(-2.0), V::Null]), (27, vec![V::Ts(1702300400000000), V::S("zzz".into()), V::Ts(1701065600000000), V::Null, V::R(-2.5)])]),
            Step::Delete(Some(Pred::IdLt(23))),
            Step::Rewrite,
            Step::Insert(vec![(28, vec![V::Ts(1700658800000000), V::S("x".into()), V::Ts(253402300800000000), V::F(0.25), V::R(-1.5)])]),
        ],
    };
    check(&history, Kind::Sqlite).await;
}

/// #377: this crate's rewrite writes a file per rewritten file where
/// official packs the survivors into one; the `DELETE` that follows is
/// inlined against each of this crate's small files, which stay live with no
/// live row, where official's delete file ends its one larger file. The live
/// layouts then differ only in files with no live row (shrunk from SQLite
/// soak seed 6100005). The model fires on that rewrite, and every row-level
/// comparison still runs; a rewrite of one file, which official also writes
/// to one file, does not fire it.
#[tokio::test(flavor = "multi_thread")]
async fn regression_rewrite_grouping_compares_live_files() {
    const GROUPING: &str =
        "partition_layout|skipped: #377 rewrite grouping: files with live rows compared";
    let grouped = History {
        columns: vec![Col {
            name: "c0".into(),
            ty: Ty::Time,
        }],
        limit: None,
        steps: vec![
            Step::Insert(vec![
                (1, vec![V::Null]),
                (2, vec![V::Null]),
                (3, vec![V::Null]),
                (4, vec![V::Null]),
                (5, vec![V::Null]),
                (6, vec![V::Null]),
                (7, vec![V::Null]),
                (8, vec![V::Time(14930000000)]),
                (9, vec![V::Time(75726000000)]),
                (10, vec![V::Time(12762000000)]),
                (11, vec![V::Time(60635000000)]),
                (12, vec![V::Time(57674000000)]),
                (13, vec![V::Time(26445000000)]),
                (14, vec![V::Null]),
            ]),
            Step::Update {
                column: "c0".into(),
                ty: Ty::Time,
                value: V::Time(0),
                pred: Pred::IdGt(3),
            },
            Step::Update {
                column: "c0".into(),
                ty: Ty::Time,
                value: V::Time(16935000000),
                pred: Pred::IdIn(vec![11, 7, 8]),
            },
            Step::Rewrite,
            Step::Delete(None),
        ],
    };
    let single = History {
        columns: vec![Col {
            name: "c0".into(),
            ty: Ty::BigInt,
        }],
        limit: Some(0),
        steps: vec![
            Step::Insert((1..=20).map(|id| (id, vec![V::I(id)])).collect()),
            Step::Delete(Some(Pred::IdLt(15))),
            Step::Rewrite,
            Step::Delete(None),
        ],
    };
    for backend in [Backend::Sqlite, Backend::Duckdb] {
        let kind = backend.kind();
        let tally = run_history_tallied(&grouped, &backend, 0, true, false)
            .await
            .unwrap_or_else(|m| panic!("{kind:?}: {} at step {}: {}", m.kind, m.step, m.detail));
        let count = |entry: &str| tally.iter().filter(|e| *e == entry).count();
        // Fires at the `DELETE` after the rewrite only.
        assert_eq!(count(GROUPING), 1, "{kind:?}: {tally:#?}");
        assert_eq!(
            count("partition_layout|ran"),
            grouped.steps.len() - 1,
            "{kind:?}"
        );
        // Rows, row ids, the change feeds, time travel and both engines'
        // reads still compare at every step.
        for comparison in
            ["rows", "row ids", "time_travel", "changes", "crate_reads", "crate_reads_at"]
        {
            assert!(
                count(&format!("{comparison}|ran")) >= grouped.steps.len(),
                "{kind:?}: {comparison} ran {} times",
                count(&format!("{comparison}|ran"))
            );
            assert!(
                !tally
                    .iter()
                    .any(|e| e.starts_with(&format!("{comparison}|skipped"))),
                "{kind:?}: {tally:#?}"
            );
        }
        let tally = run_history_tallied(&single, &backend, 0, true, false)
            .await
            .unwrap_or_else(|m| panic!("{kind:?}: {} at step {}: {}", m.kind, m.step, m.detail));
        assert!(
            !tally.iter().any(|e| e.contains("#377")),
            "{kind:?}: {tally:#?}"
        );
    }
    // The model's own shape: more files here than official's, the same rows.
    assert!(known::rewrite_grouping((1, 11), (2, 11)));
    assert!(!known::rewrite_grouping((2, 11), (2, 11)));
    assert!(!known::rewrite_grouping((1, 11), (2, 10)));
    assert!(!known::rewrite_grouping((0, 0), (1, 0)));
}

/// Every upstream divergence the harness models, each on its upstream shape,
/// checked at every snapshot and window: deletions misdated in official's
/// change feed (#1603), official's rewrite failing on an inlined data table
/// older than an added column (#1490), merge row numbering (#1605),
/// official's lost statistics (a partitioned NULL file, an unstorable string
/// bound, a widened column), the `FLOAT` exponent (#1604) and `DOUBLE`
/// fixed-point filter constants, signless zeros and NaN in Parquet files, and
/// official's statistics recompute at a rewrite (#1377). Each history asserts
/// the model it exercises fired, and one official is right on compares in
/// full. The Docker-only
/// models (MySQL's Parquet fallback, a NUL PostgreSQL partition value) are in
/// `regression_mysql` and `regression_multicatalog_created_schema_reads_its_files`.
#[tokio::test(flavor = "multi_thread")]
async fn regression_modelled_upstream_divergences() {
    let histories = [
        History {
            columns: vec![Col {
                name: "c0".into(),
                ty: Ty::Time,
            }],
            limit: None,
            steps: vec![
                Step::Insert(vec![(3, vec![V::Time(45296500000)])]),
                Step::Update {
                    column: "c0".into(),
                    ty: Ty::Time,
                    value: V::Time(67287000000),
                    pred: Pred::Eq("c0".into(), Ty::Time, V::Time(45296500000)),
                },
                Step::Update {
                    column: "c0".into(),
                    ty: Ty::Time,
                    value: V::Time(55613000000),
                    pred: Pred::IdIn(vec![4, 3, 3]),
                },
                Step::Flush,
            ],
        },
        History {
            columns: vec![Col {
                name: "c0".into(),
                ty: Ty::Blob,
            }],
            limit: None,
            steps: vec![
                Step::Insert(vec![(1, vec![V::Null]), (2, vec![V::Null])]),
                Step::Flush,
                Step::Limit(0),
                Step::Delete(Some(Pred::IdIn(vec![2, 16, 10]))),
            ],
        },
        History {
            columns: vec![Col {
                name: "c1".into(),
                ty: Ty::Time,
            }],
            limit: None,
            steps: vec![
                Step::Insert((1..=11).map(|id| (id, vec![V::Null])).collect()),
                Step::Delete(Some(Pred::IdLt(9))),
                Step::Insert(vec![(24, vec![V::Null])]),
                Step::AddColumn(
                    Col {
                        name: "a1".into(),
                        ty: Ty::Bool,
                    },
                    None,
                ),
                Step::Rewrite,
            ],
        },
        // A merge of rowid-adjacent files, one of them an update's new row
        // versions, numbers the merged rows from the first file's
        // `row_id_start`, on both sides.
        History {
            columns: vec![Col {
                name: "c0".into(),
                ty: Ty::Int,
            }],
            limit: Some(0),
            steps: vec![
                Step::Insert(vec![(12, vec![V::Null])]),
                Step::Update {
                    column: "c0".into(),
                    ty: Ty::BigInt,
                    value: V::I(-1),
                    pred: Pred::IdEq(12),
                },
                Step::Insert(vec![(36, vec![V::Null])]),
                Step::Checkpoint,
            ],
        },
        // Official's partitioned NULL-file statistics bug, on a map's leaves.
        History {
            columns: vec![
                Col {
                    name: "c0".into(),
                    ty: Ty::TsTz,
                },
                Col {
                    name: "c3".into(),
                    ty: Ty::Map,
                },
            ],
            limit: Some(0),
            steps: vec![
                Step::Partition(Some(("c0".into(), Transform::Identity))),
                Step::Insert(vec![
                    (16, vec![V::Null, V::Null]),
                    (
                        17,
                        vec![V::Ts(1702602800000000), V::Map(vec![("k".into(), Some(2))])],
                    ),
                ]),
                Step::Insert(vec![(35, vec![V::Null, V::Null])]),
            ],
        },
        // Both change feeds pair updates by the row ids the merge numbered.
        History {
            columns: vec![Col {
                name: "c1".into(),
                ty: Ty::Time,
            }],
            limit: Some(0),
            steps: vec![
                Step::Insert(vec![(19, vec![V::Time(12901000000)]), (20, vec![V::Null])]),
                Step::Update {
                    column: "c1".into(),
                    ty: Ty::Time,
                    value: V::Time(4360000000),
                    pred: Pred::IdGt(14),
                },
                Step::Insert(vec![(34, vec![V::Time(39697000000)])]),
                Step::Merge,
            ],
        },
        // Official drops the file holding a FLOAT value that prints with an
        // exponent from a filter at that value (duckdb/ducklake#1604); this
        // crate keeps it.
        History {
            columns: vec![Col {
                name: "c3".into(),
                ty: Ty::Float,
            }],
            limit: Some(0),
            steps: vec![
                Step::Insert(vec![(8, vec![V::R(f32::MAX)])]),
                Step::Insert(vec![(9, vec![V::R(1e20)])]),
                Step::Insert(vec![(10, vec![V::R(1e-7)]), (11, vec![V::R(1.0)])]),
                Step::Insert(vec![(12, vec![V::R(-1e20)])]),
            ],
        },
    ];
    let col = |name: &str, ty: Ty| Col {
        name: name.into(),
        ty,
    };
    let more = [
        // An inlined DOUBLE official stores rounded, read after a column change.
        History {
            columns: vec![col("c0", Ty::Double)],
            limit: None,
            steps: vec![
                Step::Insert(vec![(81, vec![V::F(0.48024702758086213)])]),
                Step::AddColumn(col("a1", Ty::Decimal38), None),
                Step::Flush,
            ],
        },
        // A DOUBLE constant DuckDB reads as a DECIMAL, over a Parquet file
        // official keeps the value in: compared in full.
        History {
            columns: vec![col("c0", Ty::Double)],
            limit: Some(0),
            steps: vec![
                Step::Insert(vec![(1, vec![V::F(0.48024702758086213)])]),
                Step::Insert(vec![(2, vec![V::F(5.0)])]),
            ],
        },
        // #1603: a deletion from a merged file, misdated in official's feed.
        History {
            columns: vec![col("c1", Ty::Decimal)],
            limit: Some(0),
            steps: vec![
                Step::Insert(vec![(2, vec![V::Null])]),
                Step::Insert(vec![(3, vec![V::D(6847)])]),
                Step::Merge,
                Step::Delete(Some(Pred::IsNull("c1".into()))),
            ],
        },
        // A string bound official cannot store.
        History {
            columns: vec![col("c0", Ty::Varchar)],
            limit: Some(0),
            steps: vec![
                Step::Insert(vec![(1, vec![V::S("a\0b".into())])]),
                Step::Insert(vec![(2, vec![V::S("x".into())])]),
            ],
        },
        // -0.0 beside 0.0, and NaN, in Parquet files.
        History {
            columns: vec![col("c0", Ty::Double), col("c1", Ty::Double)],
            limit: Some(0),
            steps: vec![
                Step::Insert(vec![(1, vec![V::F(-0.0), V::F(f64::NAN)])]),
                Step::Insert(vec![(2, vec![V::F(0.0), V::F(1.5)])]),
                Step::Merge,
            ],
        },
        // #1377: official's rewrite recomputes the table's statistics.
        History {
            columns: vec![col("c0", Ty::BigInt)],
            limit: Some(0),
            steps: vec![
                Step::Insert((1..=20).map(|id| (id, vec![V::I(id)])).collect()),
                Step::Delete(Some(Pred::IdLt(15))),
                Step::Rewrite,
            ],
        },
        // A widened column's bounds, dropped and narrowed by the 1.5.5 oracle.
        History {
            columns: vec![col("c2", Ty::Int)],
            limit: Some(10),
            steps: vec![
                Step::Insert(vec![(1, vec![V::I(0)])]),
                Step::Widen("c2".into()),
                Step::Flush,
                Step::Limit(0),
                Step::Insert(vec![(2, vec![V::I(5)])]),
            ],
        },
        // Official's rewrite fails on the greatest TIMESTAMPTZ held inline.
        History {
            columns: vec![col("c0", Ty::TsTz)],
            limit: Some(10),
            steps: vec![
                Step::Insert(
                    (1..=12)
                        .map(|id| (id, vec![V::Ts(1_700_000_000_000_000 + id * 3_600_000_000)]))
                        .collect(),
                ),
                Step::Insert(vec![(27, vec![V::Ts(9_223_372_036_854_775_806)])]),
                Step::Delete(Some(Pred::IdIn(vec![1, 2, 3]))),
                Step::Rewrite,
            ],
        },
    ];
    // The model each history exercises, by the reason its skip records;
    // none where the shape is one this crate follows official on.
    let fired: [&[&str]; 15] = [
        &["#1603 misdated partial-file deletion"],
        &["#1603 misdated partial-file deletion"],
        &["#1490 official rewrite fails"],
        &[],
        &["official stats: partitioned NULL file"],
        &[],
        &["#1604 float exponent"],
        &["official stats: rounded inlined DOUBLE", "double fixed-point constant"],
        &[],
        &["#1603 misdated partial-file deletion"],
        &["official stats: unstored bound"],
        &["official stats: NaN-only write", "crate footer NaN/-0.0"],
        &["#1377 rewrite recomputes"],
        &["official stats: widened column"],
        &["official rewrite fails on the greatest TIMESTAMPTZ"],
    ];
    for (index, (history, models)) in histories.iter().chain(&more).zip(fired).enumerate() {
        let tally = run_history_tallied(history, &Backend::Sqlite, 0, true, false)
            .await
            .unwrap_or_else(|m| {
                panic!(
                    "history {index}: {} at step {}: {}",
                    m.kind, m.step, m.detail
                )
            });
        let skipped: Vec<&String> = tally.iter().filter(|t| t.contains("|skipped: ")).collect();
        for model in models {
            assert!(
                skipped.iter().any(|t| t.contains(model)),
                "history {index}: `{model}` did not fire: {skipped:?}"
            );
        }
        if models.is_empty() {
            assert!(skipped.is_empty(), "history {index}: {skipped:?}");
        }
    }
}

/// A filter on a `DOUBLE` column at a constant whose shortest text is
/// fixed-point with 17 digits finds no row in official's catalog (DuckDB reads
/// the constant as a `DECIMAL`) and every row that matches in this crate's,
/// on both embedded backends.
#[tokio::test(flavor = "multi_thread")]
async fn double_fixed_point_filters_find_their_rows() {
    let history = History {
        columns: vec![Col {
            name: "c0".into(),
            ty: Ty::Double,
        }],
        limit: Some(0),
        steps: vec![
            Step::Insert(vec![(1, vec![V::F(0.48024702758086213)])]),
            Step::Insert(vec![(2, vec![V::F(5.0)])]),
        ],
    };
    let filter = "c0 = CAST('0.48024702758086213' AS DOUBLE)";
    for backend in [Backend::Sqlite, Backend::Duckdb] {
        let dir = tempfile::TempDir::new().unwrap();
        let data = dir.path().join("official_data");
        std::fs::create_dir_all(&data).unwrap();
        let official = Official::new(dir.path().join("official.ducklake"), &data);
        official
            .run("CREATE TABLE t (id BIGINT, c0 DOUBLE)")
            .unwrap();
        official
            .run("CALL lake.set_option('data_inlining_row_limit', 0, table_name => 't')")
            .unwrap();
        let lake = backend.lake(dir.path(), 0).await;
        lake.create(&history).await.unwrap();
        for step in 0..history.steps.len() {
            official.apply(&history, step).unwrap();
            lake.apply(&history, step).await.unwrap();
        }
        let (_, ours) = lake
            .rows(
                None,
                &format!("SELECT count(*) FROM lake.main.t WHERE {filter}"),
            )
            .await
            .unwrap();
        assert_eq!(ours, vec![vec!["1".to_string()]], "{:?}", backend.kind());
        let (_, theirs) = Official::rows(
            &official.connection,
            &format!("SELECT count(*) FROM t WHERE {filter}"),
        )
        .unwrap();
        // The upstream bug is still there to diverge from; when it is fixed,
        // drop `known::double_fixed_point_filter`.
        assert_eq!(
            theirs,
            vec![vec!["0".to_string()]],
            "official now finds the row"
        );
    }
}

/// duckdb/ducklake#1604: a filter on a `FLOAT` column at a constant that
/// prints with an exponent finds no row in official's catalog and every row
/// that matches in this crate's, on both embedded backends.
#[tokio::test(flavor = "multi_thread")]
async fn float_exponent_filters_find_their_rows() {
    let history = History {
        columns: vec![Col {
            name: "c0".into(),
            ty: Ty::Float,
        }],
        limit: Some(0),
        steps: vec![
            Step::Insert(vec![(1, vec![V::R(1.0)])]),
            Step::Insert(vec![(2, vec![V::R(1e20)])]),
            Step::Insert(vec![(3, vec![V::R(1e-7)])]),
            Step::Insert(vec![(4, vec![V::R(f32::MAX)])]),
        ],
    };
    // (filter, rows it matches)
    let filters = [
        ("c0 = CAST('1e20' AS {})", 1),
        ("c0 = CAST('1e-7' AS {})", 1),
        ("c0 <= CAST('1e-7' AS {})", 1),
        ("c0 = CAST('3.4028235e38' AS {})", 1),
        ("c0 >= CAST('3.4028235e38' AS {})", 1),
        ("c0 > CAST('1e20' AS {})", 1),
    ];
    for backend in [Backend::Sqlite, Backend::Duckdb] {
        let dir = tempfile::TempDir::new().unwrap();
        let data = dir.path().join("official_data");
        std::fs::create_dir_all(&data).unwrap();
        let official = Official::new(dir.path().join("official.ducklake"), &data);
        official
            .run("CREATE TABLE t (id BIGINT, c0 FLOAT)")
            .unwrap();
        official
            .run("CALL lake.set_option('data_inlining_row_limit', 0, table_name => 't')")
            .unwrap();
        let lake = backend.lake(dir.path(), 0).await;
        lake.create(&history).await.unwrap();
        for step in 0..history.steps.len() {
            official.apply(&history, step).unwrap();
            lake.apply(&history, step).await.unwrap();
        }
        let mut official_wrong = 0;
        for (filter, matches) in filters {
            let (_, rows) = lake
                .rows(
                    None,
                    &format!(
                        "SELECT count(*) FROM lake.main.t WHERE {}",
                        filter.replace("{}", "REAL")
                    ),
                )
                .await
                .unwrap();
            assert_eq!(
                rows,
                vec![vec![matches.to_string()]],
                "{:?}: {filter}",
                backend.kind()
            );
            let (_, rows) = Official::rows(
                &official.connection,
                &format!(
                    "SELECT count(*) FROM t WHERE {}",
                    filter.replace("{}", "FLOAT")
                ),
            )
            .unwrap();
            if rows != vec![vec![matches.to_string()]] {
                official_wrong += 1;
            }
        }
        // The upstream bug is still there to diverge from; when it is fixed,
        // drop `known::float_exponent_filter`.
        assert!(
            official_wrong > 0,
            "official now answers every filter right"
        );
    }
}

/// A merge of rowid-adjacent files numbers the merged rows from the first
/// file's `row_id_start`, as official's does, including the new row version an
/// `UPDATE` wrote with its row id embedded (duckdb/ducklake#1605).
#[tokio::test(flavor = "multi_thread")]
async fn merge_numbers_adjacent_updated_rows_as_official() {
    let dir = tempfile::TempDir::new().unwrap();
    let history = History {
        columns: vec![Col {
            name: "c0".into(),
            ty: Ty::Int,
        }],
        limit: Some(0),
        steps: vec![
            Step::Insert(vec![(12, vec![V::Null])]),
            Step::Update {
                column: "c0".into(),
                ty: Ty::BigInt,
                value: V::I(-1),
                pred: Pred::IdEq(12),
            },
            Step::Insert(vec![(36, vec![V::Null])]),
            Step::Merge,
        ],
    };
    let data = dir.path().join("official_data");
    std::fs::create_dir_all(&data).unwrap();
    let mut official = Official::new(dir.path().join("official.ducklake"), &data);
    official
        .run("CREATE TABLE t (id BIGINT, c0 INTEGER)")
        .unwrap();
    official
        .run("CALL lake.set_option('data_inlining_row_limit', 0, table_name => 't')")
        .unwrap();
    let lake = Backend::Sqlite.lake(dir.path(), 0).await;
    lake.create(&history).await.unwrap();
    for step in 0..history.steps.len() {
        official.reconnect();
        official.apply(&history, step).unwrap();
        lake.apply(&history, step).await.unwrap();
    }
    let (_, ours) = lake
        .rows(None, "SELECT rowid, id FROM lake.main.t ORDER BY id")
        .await
        .unwrap();
    let (_, theirs) =
        Official::rows(&official.connection, "SELECT rowid, id FROM t ORDER BY id").unwrap();
    assert_eq!(ours, vec![vec!["1", "12"], vec!["2", "36"]]);
    assert_eq!(ours, theirs);
}

/// duckdb/ducklake#1406: a flush after a rewrite that retired a file with a
/// delete file and an inlined deletion leaves official reading the retired
/// file twice at the inlined deletion's snapshot, and repeating an earlier
/// deletion in the change feed at the rewrite. This crate reads the rows that
/// snapshot held and reports no deletion at the rewrite, on both embedded
/// backends.
#[tokio::test(flavor = "multi_thread")]
async fn rewrite_then_flush_keeps_earlier_snapshots() {
    let history = History {
        columns: vec![Col {
            name: "c0".into(),
            ty: Ty::Int,
        }],
        limit: None,
        steps: vec![
            Step::Insert((1..=20).map(|id| (id, vec![V::I(id)])).collect()),
            Step::Delete(Some(Pred::IdEq(1))),
            Step::Flush,
            Step::Delete(Some(Pred::IdEq(2))),
            Step::Rewrite,
            Step::Flush,
        ],
    };
    // The ids after the second delete (index 4 of the heads).
    let expected: Vec<Vec<String>> = (3..=20).map(|id| vec![id.to_string()]).collect();
    for backend in [Backend::Sqlite, Backend::Duckdb] {
        let dir = tempfile::TempDir::new().unwrap();
        let data = dir.path().join("official_data");
        std::fs::create_dir_all(&data).unwrap();
        let mut official = Official::new(dir.path().join("official.ducklake"), &data);
        official
            .run("CREATE TABLE t (id BIGINT, c0 INTEGER)")
            .unwrap();
        let lake = backend.lake(dir.path(), 0).await;
        lake.create(&history).await.unwrap();
        let mut heads = Heads {
            official: vec![official.head()],
            ours: vec![lake.head().await],
        };
        for step in 0..history.steps.len() {
            official.reconnect();
            official.apply(&history, step).unwrap();
            lake.apply(&history, step).await.unwrap();
            heads.official.push(official.head());
            heads.ours.push(lake.head().await);
        }
        let (_, ours) = lake
            .rows(
                Some(heads.ours[4]),
                "SELECT id FROM lake.main.t ORDER BY id",
            )
            .await
            .unwrap();
        assert_eq!(ours, expected, "{:?}", backend.kind());
        // Unordered: the debug build of DuckDB asserts when an `ORDER BY`
        // meets official's resurrected rows outside the table's bounds.
        let (_, mut theirs) = Official::rows(
            &official.connection,
            &format!("SELECT id FROM t AT (VERSION => {})", heads.official[4]),
        )
        .unwrap();
        theirs.sort_by_key(|row| row[0].parse::<i64>().unwrap());
        // The upstream bug is still there to diverge from; when the oracle
        // has the fix, drop `known::overlapping_delete_files` and
        // `known::overlapping_delete_files_retired_in`.
        assert_ne!(theirs, expected, "official now reads the snapshot right");
        assert!(known::overlapping_delete_files(
            &official.connection,
            heads.official[4]
        ));
        // The rewrite (index 5 of the heads) deleted no row: this crate's
        // change feed over it reports none; official's repeats the deletion
        // its ended delete file records.
        let deletes = "change_type = 'delete'";
        let (_, ours) = lake
            .rows(
                None,
                &format!(
                    "SELECT count(*) FROM ducklake_table_changes('main.t', {0}, {0}) WHERE {deletes}",
                    heads.ours[5]
                ),
            )
            .await
            .unwrap();
        assert_eq!(ours, vec![vec!["0".to_string()]], "{:?}", backend.kind());
        let (_, theirs) = Official::rows(
            &official.connection,
            &format!(
                "SELECT count(*) FROM ducklake_table_changes('lake', 'main', 't', {0}, {0}) \
                 WHERE {deletes}",
                heads.official[5]
            ),
        )
        .unwrap();
        assert_ne!(theirs, vec![vec!["0".to_string()]]);
        assert!(known::overlapping_delete_files_retired_in(
            &official.connection,
            heads.official[5],
            heads.official[5]
        ));
        check(&history, backend.kind()).await;
    }
}

/// duckdb/ducklake#1406, seed 7400355 shrunk: a checkpoint flushes the
/// inlined deletions of a file a rewrite retired to a backdated delete file
/// that lacks the deletions of the file's earlier delete file. In every
/// window starting after it begins, official reports the earlier file's
/// deletions, dated at its begin_snapshot; this crate reports none. The model
/// fires there, and not in a window before the backdated file begins.
#[tokio::test(flavor = "multi_thread")]
async fn rewrite_checkpoint_backdated_delete_file_later_windows() {
    let history = History {
        columns: vec![Col {
            name: "c0".into(),
            ty: Ty::Varchar,
        }],
        limit: None,
        steps: vec![
            Step::Insert(
                (1..=14)
                    .map(|id| (id, vec![V::S(format!("v{id}"))]))
                    .collect(),
            ),
            Step::Update {
                column: "c0".into(),
                ty: Ty::Varchar,
                value: V::S("u".into()),
                pred: Pred::IdLt(13),
            },
            Step::Delete(None),
            Step::Rewrite,
            Step::Checkpoint,
            Step::Insert(
                (20..=33)
                    .map(|id| (id, vec![V::S(format!("w{id}"))]))
                    .collect(),
            ),
        ],
    };
    for backend in [Backend::Sqlite, Backend::Duckdb] {
        let dir = tempfile::TempDir::new().unwrap();
        let data = dir.path().join("official_data");
        std::fs::create_dir_all(&data).unwrap();
        let mut official = Official::new(dir.path().join("official.ducklake"), &data);
        official
            .run("CREATE TABLE t (id BIGINT, c0 VARCHAR)")
            .unwrap();
        let lake = backend.lake(dir.path(), 0).await;
        lake.create(&history).await.unwrap();
        let mut heads = Heads {
            official: vec![official.head()],
            ours: vec![lake.head().await],
        };
        for step in 0..history.steps.len() {
            official.reconnect();
            official.apply(&history, step).unwrap();
            lake.apply(&history, step).await.unwrap();
            heads.official.push(official.head());
            heads.ours.push(lake.head().await);
        }
        // The window of the last insert alone.
        let (oa, ob) = (heads.official[5] + 1, heads.official[6]);
        let (ca, cb) = (heads.ours[5] + 1, heads.ours[6]);
        let deletes = "change_type = 'delete'";
        let (_, ours) = lake
            .rows(
                None,
                &format!(
                    "SELECT count(*) FROM ducklake_table_changes('main.t', {ca}, {cb}) \
                     WHERE {deletes}"
                ),
            )
            .await
            .unwrap();
        assert_eq!(ours, vec![vec!["0".to_string()]], "{:?}", backend.kind());
        let (_, theirs) = Official::rows(
            &official.connection,
            &format!(
                "SELECT count(*) FROM ducklake_table_changes('lake', 'main', 't', {oa}, {ob}) \
                 WHERE {deletes}"
            ),
        )
        .unwrap();
        // The upstream bug is still there to diverge from.
        assert_eq!(theirs, vec![vec!["12".to_string()]], "official's feed");
        assert!(known::overlapping_delete_files_retired_in(
            &official.connection,
            oa,
            ob
        ));
        // The first insert's window precedes both delete files.
        assert!(!known::overlapping_delete_files_retired_in(
            &official.connection,
            heads.official[0] + 1,
            heads.official[1]
        ));
        check(&history, backend.kind()).await;
    }
}

/// A column added with a default and then widened reads its default in rows
/// written before it was added, inlined or in Parquet.
#[tokio::test(flavor = "multi_thread")]
async fn regression_widened_added_column_keeps_its_default() {
    for limit in [Some(10), Some(0)] {
        let history = History {
            columns: vec![Col {
                name: "c3".into(),
                ty: Ty::Float,
            }],
            limit,
            steps: vec![
                Step::Insert(vec![(1, vec![V::Null])]),
                Step::AddColumn(
                    Col {
                        name: "a1".into(),
                        ty: Ty::Int,
                    },
                    Some("7".into()),
                ),
                Step::Widen("a1".into()),
            ],
        };
        check(&history, Kind::Duckdb).await;
        check(&history, Kind::Sqlite).await;
    }
}

/// A Parquet file's float bounds keep the sign of the zero written first, as
/// DuckDB's writer keeps them (the footer's minimum zero is `-0.0`).
#[tokio::test(flavor = "multi_thread")]
async fn regression_parquet_zero_bounds_keep_their_sign() {
    for values in [vec![V::F(0.0)], vec![V::F(-0.0), V::F(0.0)], vec![V::F(0.0), V::F(-0.0)]] {
        let history = History {
            columns: vec![Col {
                name: "c3".into(),
                ty: Ty::Double,
            }],
            limit: Some(0),
            steps: vec![Step::Insert(
                values
                    .into_iter()
                    .enumerate()
                    .map(|(i, v)| (i as i64, vec![v]))
                    .collect(),
            )],
        };
        check(&history, Kind::Sqlite).await;
    }
}

/// An `UPDATE` of a partitioned table whose new row versions span several
/// partitions commits them with its deletions in one snapshot.
#[tokio::test(flavor = "multi_thread")]
async fn regression_partitioned_update_writes_a_file_per_partition() {
    let history = History {
        columns: vec![
            Col {
                name: "c0".into(),
                ty: Ty::Date,
            },
            Col {
                name: "c1".into(),
                ty: Ty::BigInt,
            },
        ],
        limit: Some(0),
        steps: vec![
            Step::Partition(Some(("c0".into(), Transform::Day))),
            Step::Insert(vec![
                (1, vec![V::Date(19001), V::I(1)]),
                (2, vec![V::Date(19002), V::I(2)]),
                (3, vec![V::Null, V::I(3)]),
            ]),
            Step::Update {
                column: "c1".into(),
                ty: Ty::BigInt,
                value: V::I(-7),
                pred: Pred::IdLt(10),
            },
        ],
    };
    check(&history, Kind::Duckdb).await;
    check(&history, Kind::Sqlite).await;
}

/// A rewrite retires a file whose row an inlined `UPDATE` deleted, and the
/// file holding that row's next version, which a later `UPDATE` deleted: the
/// retirement reports the row's deletion from both, and the rewritten row as
/// an insertion, all at the rewrite's snapshot. Official's feed left-joins
/// insertions and deletions on snapshot and row id, so it reports two
/// preimages and two postimages; this crate reported one postimage.
#[tokio::test(flavor = "multi_thread")]
async fn regression_retirement_pairs_each_join_match() {
    let cols = vec![
        Col {
            name: "c1".into(),
            ty: Ty::BigInt,
        },
        Col {
            name: "c2".into(),
            ty: Ty::BigInt,
        },
    ];
    let history = History {
        columns: cols,
        limit: None,
        steps: vec![
            Step::Insert((1..=23).map(|id| (id, vec![V::I(id), V::I(-id)])).collect()),
            Step::Update {
                column: "c1".into(),
                ty: Ty::BigInt,
                value: V::I(9),
                pred: Pred::IdIn(vec![19, 6, 23]),
            },
            Step::Update {
                column: "c2".into(),
                ty: Ty::BigInt,
                value: V::Null,
                pred: Pred::IdLt(21),
            },
            Step::Delete(Some(Pred::IdIn(vec![8]))),
            Step::Rewrite,
        ],
    };
    check(&history, Kind::Sqlite).await;
    check(&history, Kind::Duckdb).await;
}

/// The 1.5.5 oracle drops a widened column's table-wide bounds at a flush
/// (`known::widened`), also when no row is left for them to cover.
#[tokio::test(flavor = "multi_thread")]
async fn regression_widened_bounds_dropped_with_no_rows_left() {
    let history = History {
        columns: vec![Col {
            name: "c3".into(),
            ty: Ty::Int,
        }],
        limit: None,
        steps: vec![
            Step::Insert(vec![(7, vec![V::I(5)])]),
            Step::Widen("c3".into()),
            Step::Delete(None),
            Step::Flush,
        ],
    };
    check(&history, Kind::Sqlite).await;
    check(&history, Kind::Duckdb).await;
}

/// Inlined rows deleted at several snapshots and one updated, flushed, then
/// deleted from further: the flush's delete file begins at the earliest of
/// its deletions, and both change feeds report each deletion and update
/// preimage in every window holding its snapshot, also windows starting after
/// the delete file begins.
fn flushed_deletions_history() -> History {
    History {
        columns: vec![Col {
            name: "c1".into(),
            ty: Ty::Int,
        }],
        limit: Some(100),
        steps: vec![
            Step::Insert((1..=5).map(|id| (id, vec![V::I(id * 10)])).collect()),
            Step::Delete(Some(Pred::IdEq(1))),
            Step::Delete(Some(Pred::IdEq(2))),
            Step::Update {
                column: "c1".into(),
                ty: Ty::Int,
                value: V::I(33),
                pred: Pred::IdEq(3),
            },
            Step::Flush,
            Step::Delete(Some(Pred::IdEq(4))),
            Step::Insert(vec![(6, vec![V::I(60)])]),
            Step::Delete(Some(Pred::IdEq(3))),
        ],
    }
}

/// [`flushed_deletions_history`] on SQLite and DuckDB, every window.
#[tokio::test(flavor = "multi_thread")]
async fn regression_flushed_deletions_in_every_window() {
    check(&flushed_deletions_history(), Kind::Sqlite).await;
    check(&flushed_deletions_history(), Kind::Duckdb).await;
}

/// [`flushed_deletions_history`] on MySQL, every window.
#[cfg(feature = "write-mysql")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn regression_flushed_deletions_in_every_window_mysql() {
    let (_container, port) = mysql_container().await;
    let backend = Backend::Mysql {
        port,
    };
    if let Err(mismatch) = run_history(&flushed_deletions_history(), &backend, 0, true, false).await
    {
        panic!(
            "{} at step {}: {}",
            mismatch.kind, mismatch.step, mismatch.detail
        );
    }
}

/// [`flushed_deletions_history`] on a multicatalog PostgreSQL store, every
/// window. (The single-catalog PostgreSQL writer neither inlines nor
/// deletes through SQL.)
#[cfg(feature = "multicatalog-postgres")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn regression_flushed_deletions_in_every_window_multicatalog() {
    let (_container, port) = postgres_container().await;
    let backend = Backend::Multicatalog {
        port,
    };
    if let Err(mismatch) = run_history(&flushed_deletions_history(), &backend, 0, true, false).await
    {
        panic!(
            "{} at step {}: {}",
            mismatch.kind, mismatch.step, mismatch.detail
        );
    }
}

/// MySQL regressions the harness found: an inlined `TIMESTAMPTZ`, a widened
/// added column's default, and a partitioned `UPDATE` spanning files.
#[cfg(feature = "write-mysql")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn regression_mysql() {
    let (_container, port) = mysql_container().await;
    let backend = Backend::Mysql {
        port,
    };
    let histories = [
        History {
            columns: vec![Col {
                name: "c0".into(),
                ty: Ty::TsTz,
            }],
            limit: None,
            steps: vec![Step::Insert(vec![(1, vec![V::Ts(1701321200000000)])])],
        },
        History {
            columns: vec![Col {
                name: "c3".into(),
                ty: Ty::Float,
            }],
            limit: Some(10),
            steps: vec![
                Step::Insert(vec![(1, vec![V::Null])]),
                Step::AddColumn(
                    Col {
                        name: "a1".into(),
                        ty: Ty::Int,
                    },
                    Some("7".into()),
                ),
                Step::Widen("a1".into()),
            ],
        },
        History {
            columns: vec![
                Col {
                    name: "c0".into(),
                    ty: Ty::Date,
                },
                Col {
                    name: "c1".into(),
                    ty: Ty::BigInt,
                },
            ],
            limit: Some(0),
            steps: vec![
                Step::Partition(Some(("c0".into(), Transform::Day))),
                Step::Insert(vec![
                    (1, vec![V::Date(19001), V::I(1)]),
                    (2, vec![V::Date(19002), V::I(2)]),
                ]),
                Step::Update {
                    column: "c1".into(),
                    ty: Ty::BigInt,
                    value: V::I(-7),
                    pred: Pred::IdLt(10),
                },
            ],
        },
    ];
    // A merge keeps a partitioned table's files apart by partition.
    let partitioned_merge = History {
        columns: vec![Col {
            name: "c1".into(),
            ty: Ty::Int,
        }],
        limit: None,
        steps: vec![
            Step::Partition(Some(("c1".into(), Transform::Identity))),
            Step::Insert(vec![(1, vec![V::I(0)]), (2, vec![V::Null])]),
            Step::Flush,
            Step::Merge,
        ],
    };
    // `known::mysql_fallback`: a NaN goes to Parquet where official inlines it.
    let fallback = History {
        columns: vec![Col {
            name: "c0".into(),
            ty: Ty::Double,
        }],
        limit: None,
        steps: vec![
            Step::Insert(vec![(1, vec![V::F(f64::NAN)])]),
            Step::Insert(vec![(2, vec![V::F(1.5)])]),
            Step::Delete(Some(Pred::IdEq(2))),
        ],
    };
    // An insert after a column was added at limit 0 (duckdb/ducklake#1606):
    // refused as official refuses it, unless MySQL writes it to Parquet.
    let stale = |value: V| History {
        columns: vec![Col {
            name: "c0".into(),
            ty: Ty::Date,
        }],
        limit: Some(0),
        steps: vec![
            Step::AddColumn(
                Col {
                    name: "a1".into(),
                    ty: Ty::Bool,
                },
                None,
            ),
            Step::Limit(10),
            Step::Insert(vec![(36, vec![value, V::Null])]),
        ],
    };
    let (stale_inlined, stale_parquet) = (stale(V::Date(19000)), stale(V::Date(-2_147_483_647)));
    for (seed, history) in histories
        .iter()
        .chain([&fallback, &partitioned_merge, &stale_inlined, &stale_parquet])
        .enumerate()
    {
        if let Err(mismatch) = run_history(history, &backend, seed as u64, true, false).await {
            panic!(
                "{} at step {}: {}",
                mismatch.kind, mismatch.step, mismatch.detail
            );
        }
    }
}

/// A multicatalog schema made with `get_or_create_schema` lives under the
/// catalog's `cat_{id}` subtree, where the catalog's writes put its files, so
/// they read back.
#[cfg(feature = "multicatalog-postgres")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn regression_multicatalog_created_schema_reads_its_files() {
    let (_container, port) = postgres_container().await;
    let backend = Backend::Multicatalog {
        port,
    };
    let history = History {
        columns: vec![Col {
            name: "c2".into(),
            ty: Ty::BigInt,
        }],
        limit: Some(10),
        steps: vec![
            Step::Insert(vec![(7, vec![V::Null])]),
            Step::Flush,
            Step::Limit(0),
            Step::Insert((20..40).map(|i| (i, vec![V::I(i)])).collect()),
        ],
    };
    if let Err(mismatch) = run_history(&history, &backend, 0, true, false).await {
        panic!(
            "{} at step {}: {}",
            mismatch.kind, mismatch.step, mismatch.detail
        );
    }
    // A widened column's new inlined table holds the widened type.
    let history = History {
        columns: vec![Col {
            name: "c1".into(),
            ty: Ty::Int,
        }],
        limit: Some(1),
        steps: vec![Step::Widen("c1".into()), Step::Insert(vec![(1, vec![V::I(i64::MIN)])])],
    };
    if let Err(mismatch) = run_history(&history, &backend, 2, true, false).await {
        panic!(
            "{} at step {}: {}",
            mismatch.kind, mismatch.step, mismatch.detail
        );
    }
    // `known::postgres_nul_partition`: the write of a NUL partition value
    // fails and is undone on official; the history goes on.
    let history = History {
        columns: vec![Col {
            name: "c0".into(),
            ty: Ty::Varchar,
        }],
        limit: Some(0),
        steps: vec![
            Step::Partition(Some(("c0".into(), Transform::Identity))),
            Step::Insert(vec![(1, vec![V::S("a\0b".into())])]),
            Step::Insert(vec![(2, vec![V::S("x".into())])]),
        ],
    };
    if let Err(mismatch) = run_history(&history, &backend, 3, true, false).await {
        panic!(
            "{} at step {}: {}",
            mismatch.kind, mismatch.step, mismatch.detail
        );
    }
    // A flush of an inlined NUL partition value: official fails to parse its
    // own SQL, and PostgreSQL refuses the value.
    let history = History {
        columns: vec![Col {
            name: "c0".into(),
            ty: Ty::Varchar,
        }],
        limit: Some(10),
        steps: vec![
            Step::Partition(Some(("c0".into(), Transform::Identity))),
            Step::Insert(vec![(16, vec![V::S("a\0b".into())])]),
            Step::Flush,
        ],
    };
    if let Err(mismatch) = run_history(&history, &backend, 4, true, false).await {
        panic!(
            "{} at step {}: {}",
            mismatch.kind, mismatch.step, mismatch.detail
        );
    }
    // An inlined string holding a NUL byte reads back from its `BYTEA` column.
    let history = History {
        columns: vec![Col {
            name: "c0".into(),
            ty: Ty::Varchar,
        }],
        limit: None,
        steps: vec![Step::Insert(vec![(1, vec![V::S("a\0b".into())])])],
    };
    if let Err(mismatch) = run_history(&history, &backend, 1, true, false).await {
        panic!(
            "{} at step {}: {}",
            mismatch.kind, mismatch.step, mismatch.detail
        );
    }
}

/// `MetadataWriter::commit_truncate` ends every live row and records the
/// storage it ended as official's `DELETE` with no filter does:
/// `deleted_from_table` for Parquet files, `inlined_delete` for inlined rows.
/// A flush that read the inlined rows before the truncate then conflicts
/// instead of writing them back to Parquet.
async fn truncate_records_inlined_delete(backend: Backend) {
    let tokens = |changes: &str, table_id: i64| changes.replace(&table_id.to_string(), "T");
    let head_changes = |provider: &Arc<dyn MetadataProvider>| {
        let head = provider.get_current_snapshot().unwrap();
        provider
            .list_snapshot_changes()
            .unwrap()
            .into_iter()
            .find(|change| change.snapshot_id == head)
            .and_then(|change| change.changes_made)
            .unwrap()
    };
    // `commit_inlined_deletes` ends inlined rows only: `inlined_delete`.
    {
        let dir = tempfile::TempDir::new().unwrap();
        let lake = backend.lake(dir.path(), 0).await;
        let history = History {
            columns: vec![Col {
                name: "c0".into(),
                ty: Ty::BigInt,
            }],
            limit: Some(10),
            steps: vec![Step::Insert(vec![(1, vec![V::I(1)]), (2, vec![V::I(2)])])],
        };
        lake.create(&history).await.unwrap();
        lake.apply(&history, 0).await.unwrap();
        let table_id = lake.table_id().await.unwrap();
        let (writer, provider) = (lake.open)().await;
        let base = provider.get_current_snapshot().unwrap();
        let rows = provider.get_inlined_data_for_flush(table_id, base).unwrap();
        writer
            .commit_inlined_deletes(
                table_id,
                "main",
                "t",
                base,
                &[datafusion_ducklake::InlinedRowRef {
                    table_name: rows[0].rows.table_name.clone(),
                    row_id: rows[0].rows.row_ids[0],
                }],
            )
            .unwrap();
        drop((writer, provider));
        let (_, provider) = (lake.open)().await;
        assert_eq!(
            tokens(&head_changes(&provider), table_id),
            "inlined_delete:T",
            "{:?}",
            backend.kind()
        );
    }
    // (inserts written to Parquet first, then inlined; the expected tokens)
    let cases = [(false, "inlined_delete:T"), (true, "deleted_from_table:T,inlined_delete:T")];
    for (parquet_first, expected) in cases {
        let dir = tempfile::TempDir::new().unwrap();
        let data = dir.path().join("official_data");
        std::fs::create_dir_all(&data).unwrap();
        let official = Official::new(dir.path().join("official.ducklake"), &data);
        let lake = backend.lake(dir.path(), 0).await;
        let mut steps = Vec::new();
        if parquet_first {
            // More Parquet rows than the limit: official's `DELETE` writes
            // their deletions to a delete file rather than inlining them.
            steps.push(Step::Limit(0));
            steps.push(Step::Insert(
                (100..112).map(|id| (id, vec![V::I(id)])).collect(),
            ));
            steps.push(Step::Limit(10));
        }
        steps.push(Step::Insert(vec![(2, vec![V::I(2)]), (3, vec![V::I(3)])]));
        steps.push(Step::Delete(None));
        let history = History {
            columns: vec![Col {
                name: "c0".into(),
                ty: Ty::BigInt,
            }],
            limit: Some(10),
            steps,
        };
        official
            .run("CREATE TABLE t (id BIGINT, c0 BIGINT)")
            .unwrap();
        official
            .run("CALL lake.set_option('data_inlining_row_limit', 10, table_name => 't')")
            .unwrap();
        lake.create(&history).await.unwrap();
        let truncate = history.steps.len() - 1;
        for step in 0..truncate {
            official.apply(&history, step).unwrap();
            lake.apply(&history, step).await.unwrap();
        }
        let table_id = lake.table_id().await.unwrap();
        let (base, inlined_tables) = {
            let (_, provider) = (lake.open)().await;
            let base = provider.get_current_snapshot().unwrap();
            let inlined_tables = provider
                .get_inlined_data_for_flush(table_id, base)
                .unwrap()
                .into_iter()
                .map(|data| data.rows.table_name)
                .collect::<Vec<_>>();
            (base, inlined_tables)
        };
        assert!(!inlined_tables.is_empty());
        official.apply(&history, truncate).unwrap();
        let (writer, provider) = (lake.open)().await;
        writer.commit_truncate(table_id, "main", "t", base).unwrap();
        drop((writer, provider));
        let (writer, provider) = (lake.open)().await;

        let official_changes = Official::rows(
            &official.connection,
            "SELECT changes_made FROM __ducklake_metadata_lake.ducklake_snapshot_changes \
             ORDER BY snapshot_id DESC LIMIT 1",
        )
        .unwrap()
        .1[0][0]
            .trim_matches('"')
            .to_string();
        let official_table: i64 = Official::rows(
            &official.connection,
            "SELECT table_id FROM __ducklake_metadata_lake.ducklake_table WHERE table_name = 't'",
        )
        .unwrap()
        .1[0][0]
            .parse()
            .unwrap();
        assert_eq!(tokens(&official_changes, official_table), expected);
        let head = provider.get_current_snapshot().unwrap();
        assert_eq!(
            tokens(&head_changes(&provider), table_id),
            expected,
            "{:?}",
            backend.kind()
        );

        let error = writer
            .commit_inlined_flush(
                base,
                &[datafusion_ducklake::InlinedFlushTable {
                    table_id,
                    inlined_tables,
                    ..Default::default()
                }],
            )
            .unwrap_err();
        assert!(
            matches!(error, datafusion_ducklake::DuckLakeError::Conflict(_)),
            "{:?}: {error}",
            backend.kind()
        );
        assert_eq!(provider.get_current_snapshot().unwrap(), head);
        let (_, rows) = lake
            .rows(None, "SELECT count(*) FROM lake.main.t")
            .await
            .unwrap();
        assert_eq!(rows, vec![vec!["0".to_string()]], "{:?}", backend.kind());
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn truncate_records_inlined_delete_embedded() {
    truncate_records_inlined_delete(Backend::Sqlite).await;
    truncate_records_inlined_delete(Backend::Duckdb).await;
}

#[cfg(feature = "write-mysql")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
#[tokio::test(flavor = "multi_thread")]
async fn truncate_records_inlined_delete_mysql() {
    let (_container, port) = mysql_container().await;
    truncate_records_inlined_delete(Backend::Mysql {
        port,
    })
    .await;
}

#[cfg(feature = "multicatalog-postgres")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
#[tokio::test(flavor = "multi_thread")]
async fn truncate_records_inlined_delete_multicatalog() {
    let (_container, port) = postgres_container().await;
    truncate_records_inlined_delete(Backend::Multicatalog {
        port,
    })
    .await;
}

/// Each modelled upstream divergence recognizes its upstream shape and
/// nothing next to it.
#[test]
fn known_models_fire_only_on_their_shapes() {
    let col = |name: &str, ty: Ty| Col {
        name: name.into(),
        ty,
    };
    let history = |columns: Vec<Col>, steps: Vec<Step>| History {
        columns,
        limit: Some(0),
        steps,
    };

    // duckdb/ducklake#1604: a FLOAT constant printed with an exponent.
    assert!(known::float_exponent_filter(Ty::Float, &V::R(1e20)));
    assert!(known::float_exponent_filter(Ty::Float, &V::R(1e-7)));
    assert!(!known::float_exponent_filter(Ty::Float, &V::R(2.5)));
    assert!(!known::float_exponent_filter(Ty::Float, &V::R(0.0)));
    assert!(!known::float_exponent_filter(Ty::Double, &V::F(1e20)));

    // A DOUBLE constant whose fixed-point text has more than 15 digits.
    assert!(known::double_fixed_point_filter(
        Ty::Double,
        &V::F(0.48024702758086213)
    ));
    assert!(!known::double_fixed_point_filter(Ty::Double, &V::F(0.5)));
    assert!(!known::double_fixed_point_filter(Ty::Double, &V::F(1e20)));
    assert!(!known::double_fixed_point_filter(
        Ty::Double,
        &V::F(1.2345678901234567e-300)
    ));
    assert!(!known::double_fixed_point_filter(
        Ty::Float,
        &V::F(0.48024702758086213)
    ));

    // Official's inlined DOUBLE rounded through DuckDB's DECIMAL reading.
    assert_eq!(
        known::duckdb_decimal_double("0.48024702758086213"),
        known::duckdb_decimal_double("0.4802470275808622")
    );
    assert_eq!(
        known::duckdb_decimal_double("-0.0072002474755199644"),
        known::duckdb_decimal_double("-0.007200247475519964")
    );
    assert_eq!(
        known::duckdb_decimal_double("0.0030294917796618898"),
        known::duckdb_decimal_double("0.00302949177966189")
    );
    assert_eq!(known::duckdb_decimal_double("0.0"), "0.0");
    assert_eq!(known::duckdb_decimal_double("NULL"), "NULL");
    let (d0, d1) = (col("d0", Ty::Double), col("d1", Ty::Double));
    let written = vec![Step::Insert(vec![(1, vec![V::F(0.48024702758086213), V::F(0.5)])])];
    let inlined = History {
        limit: Some(10),
        ..history(vec![d0.clone(), d1.clone()], written.clone())
    };
    assert!(!known::inlined_double_text(&inlined, 0, &d0));
    assert!(known::inlined_double_text(&inlined, 1, &d0));
    assert!(!known::inlined_double_text(&inlined, 1, &d1));
    // Only the value official rounded is named: another cell of the column
    // compares exactly.
    let keys = known::inlined_double_keys(&inlined, 1, &d0);
    assert_eq!(
        keys,
        vec![known::duckdb_decimal_double("0.4802470275808622")]
    );
    assert!(!keys.contains(&known::duckdb_decimal_double("0.48024702758086")));
    // A Parquet write keeps the value: limit 0, or more rows than the limit.
    let parquet = history(vec![d0.clone(), d1.clone()], written);
    assert!(!known::inlined_double_text(&parquet, 1, &d0));
    let wide = History {
        limit: Some(1),
        ..history(
            vec![d0.clone()],
            vec![Step::Insert(vec![
                (1, vec![V::F(0.48024702758086213)]),
                (2, vec![V::F(0.5)]),
            ])],
        )
    };
    assert!(!known::inlined_double_text(&wide, 1, &d0));
    // An update under a limit raised since inlines.
    let updated = history(
        vec![d0.clone()],
        vec![
            Step::Insert(vec![(1, vec![V::F(0.5)])]),
            Step::Limit(10),
            Step::Update {
                column: "d0".into(),
                ty: Ty::Double,
                value: V::F(0.48024702758086213),
                pred: Pred::IdEq(1),
            },
        ],
    );
    assert!(!known::inlined_double_text(&updated, 2, &d0));
    assert!(known::inlined_double_text(&updated, 3, &d0));
    // A Parquet value an update of another column inlines again.
    let reinlined = history(
        vec![d0.clone(), d1.clone()],
        vec![
            Step::Insert(vec![(1, vec![V::F(0.48024702758086213), V::F(0.5)])]),
            Step::Limit(10),
            Step::Update {
                column: "d1".into(),
                ty: Ty::Double,
                value: V::F(1.5),
                pred: Pred::IdEq(1),
            },
        ],
    );
    assert!(!known::inlined_double_text(&reinlined, 2, &d0));
    assert!(known::inlined_double_text(&reinlined, 3, &d0));
    assert!(!known::inlined_double_text(&reinlined, 3, &d1));

    // Official's lost bounds: a partitioned multi-file write with a NULL file.
    let c0 = col("c0", Ty::BigInt);
    let partitioned = history(
        vec![c0.clone()],
        vec![
            Step::Partition(Some(("c0".into(), Transform::Identity))),
            Step::Insert(vec![(1, vec![V::Null]), (2, vec![V::I(4)])]),
        ],
    );
    assert_eq!(
        known::lost_bounds(&partitioned, 1, &c0),
        Some("official stats: partitioned NULL file")
    );
    let unpartitioned = history(
        vec![c0.clone()],
        vec![Step::Insert(vec![(1, vec![V::Null]), (2, vec![V::I(4)])])],
    );
    assert_eq!(known::lost_bounds(&unpartitioned, 0, &c0), None);
    let no_null = history(
        vec![c0.clone()],
        vec![
            Step::Partition(Some(("c0".into(), Transform::Identity))),
            Step::Insert(vec![(1, vec![V::I(3)]), (2, vec![V::I(4)])]),
        ],
    );
    assert_eq!(known::lost_bounds(&no_null, 1, &c0), None);
    // A float write holding NaN where official stores no bounds.
    let f0 = col("f0", Ty::Double);
    let nan_only = history(
        vec![f0.clone()],
        vec![Step::Insert(vec![(1, vec![V::F(f64::NAN)]), (2, vec![V::Null])])],
    );
    assert_eq!(
        known::lost_bounds(&nan_only, 0, &f0),
        Some("official stats: NaN-only write")
    );
    let finite = history(
        vec![f0.clone()],
        vec![Step::Insert(vec![(1, vec![V::F(1.5)]), (2, vec![V::Null])])],
    );
    assert_eq!(known::lost_bounds(&finite, 0, &f0), None);
    assert_eq!(known::lost_bounds(&nan_only, 0, &c0), None);

    // An unstorable string bound, only for the column that held it.
    let (s0, s1) = (col("s0", Ty::Varchar), col("s1", Ty::Varchar));
    let unstored = history(
        vec![s0.clone(), s1.clone()],
        vec![Step::Insert(vec![(1, vec![V::S("a\0b".into()), V::S("x".into())])])],
    );
    assert!(known::unstored_column(&unstored, 0, &s0));
    assert!(!known::unstored_column(&unstored, 0, &s1));
    assert!(known::unstored_value(&V::S("é".repeat(140))));
    assert!(!known::unstored_value(&V::S(format!(
        "{}z",
        "b".repeat(300)
    ))));

    // A widened column, only that one.
    let (w, other) = (col("w", Ty::Int), col("o", Ty::Int));
    let widened = history(
        vec![w.clone(), other.clone()],
        vec![Step::Widen("w".into())],
    );
    assert_eq!(
        known::lost_bounds(&widened, 0, &w),
        Some("official stats: widened column")
    );
    assert_eq!(known::lost_bounds(&widened, 0, &other), None);

    // -0.0 makes a float column's zeros signless from the step after it.
    let (d, e) = (col("d", Ty::Double), col("e", Ty::Double));
    let zeros = history(
        vec![d.clone(), e.clone()],
        vec![
            Step::Insert(vec![(1, vec![V::F(0.0), V::F(0.0)])]),
            Step::Insert(vec![(2, vec![V::F(-0.0), V::F(1.0)])]),
        ],
    );
    assert!(!known::negative_zero(&zeros, 1, &d));
    assert!(known::negative_zero(&zeros, 2, &d));
    assert!(!known::negative_zero(&zeros, 2, &e));
    assert!(known::float_special(&zeros, 1, &d));
    assert!(!known::float_special(&zeros, 1, &e));

    // The rewrite-repeats sibling of #1571: a deletion of an earlier step
    // repeated exactly twice in a window holding a rewrite.
    let rewrite = history(
        vec![c0.clone()],
        vec![Step::Delete(Some(Pred::IdEq(1))), Step::Rewrite],
    );
    let row = |s: &str, kind: &str| vec![s.to_string(), "0".to_string(), kind.to_string()];
    let repeated = (
        vec!["snapshot_id".into(), "rowid".into(), "change_type".into()],
        vec![row("step1", "\"delete\""), row("step1", "\"delete\"")],
    );
    let ((_, rows), fired) = known::rewrite_repeats(&rewrite, (0, 2), repeated.clone());
    assert!(fired);
    assert_eq!(rows.len(), 1);
    let ((_, rows), fired) = known::rewrite_repeats(&rewrite, (0, 1), repeated.clone());
    assert!(!fired, "no rewrite in the window");
    assert_eq!(rows.len(), 2);
    let inserts = (
        repeated.0.clone(),
        vec![row("step1", "\"insert\""), row("step1", "\"insert\"")],
    );
    assert!(!known::rewrite_repeats(&rewrite, (0, 2), inserts).1);

    // An inlined table older than the live spec: a widening's out-of-range
    // insert, and a partition spec's flush layout.
    let stale = history(
        vec![col("c2", Ty::Int)],
        vec![
            Step::Insert(vec![(1, vec![V::I(1)])]),
            Step::Widen("c2".into()),
            Step::Limit(10),
            Step::Insert(vec![(2, vec![V::I(i64::MAX)])]),
        ],
    );
    let refused = "can't be cast because the value is out of range for the destination type";
    assert!(known::stale_inlined_table(&stale, 3, refused, Kind::Sqlite));
    assert!(!known::stale_inlined_table(
        &stale,
        0,
        refused,
        Kind::Sqlite
    ));
    assert!(!known::stale_inlined_table(
        &stale,
        3,
        "Conflict",
        Kind::Sqlite
    ));
    // A column added at limit 0 (duckdb/ducklake#1606): refused on every
    // backend, except where MySQL writes the insert to Parquet.
    let added = |value: V| {
        history(
            vec![col("c0", Ty::Date)],
            vec![
                Step::AddColumn(col("a1", Ty::Bool), None),
                Step::Limit(10),
                Step::Insert(vec![(1, vec![value, V::Null])]),
            ],
        )
    };
    let supplied = "table ducklake_inlined_data_1_1 has 5 columns but 6 values were supplied";
    let unheld = added(V::Date(-2_147_483_647));
    assert!(known::stale_inlined_table(
        &unheld,
        2,
        supplied,
        Kind::Mysql
    ));
    assert!(!known::stale_inlined_table(
        &unheld,
        2,
        supplied,
        Kind::Sqlite
    ));
    assert!(!known::stale_inlined_table(
        &added(V::Date(19000)),
        2,
        supplied,
        Kind::Mysql
    ));
    let partitioned_late = history(
        vec![col("c2", Ty::Int)],
        vec![
            Step::Partition(Some(("c2".into(), Transform::Identity))),
            Step::Limit(10),
            Step::Insert(vec![(1, vec![V::I(1)])]),
            Step::Flush,
        ],
    );
    assert!(!known::stale_inlined_partition(&partitioned_late, 2));
    assert!(known::stale_inlined_partition(&partitioned_late, 3));
    let mut eager = partitioned_late.clone();
    eager.limit = Some(10);
    eager.steps.remove(1);
    assert!(
        !known::stale_inlined_partition(&eager, 2),
        "the spec got its table"
    );

    // A merge after an update may number the update's rows either way.
    let merged = history(
        vec![c0.clone()],
        vec![
            Step::Update {
                column: "c0".into(),
                ty: Ty::BigInt,
                value: V::I(1),
                pred: Pred::IdEq(1),
            },
            Step::Insert(vec![(2, vec![V::I(2)])]),
            Step::Merge,
        ],
    );
    assert!(!known::merged_update_order(&merged, 1));
    assert!(known::merged_update_order(&merged, 2));
    let merge_first = history(vec![c0.clone()], vec![Step::Merge, Step::Delete(None)]);
    assert!(!known::merged_update_order(&merge_first, 1));
    // Only the updated rows may differ in their row ids.
    let rows = |rows: &[[&str; 3]]| {
        (
            vec!["rowid".to_string(), "id".to_string(), "c0".to_string()],
            rows.iter()
                .map(|r| r.iter().map(|c| c.to_string()).collect())
                .collect::<Vec<Vec<String>>>(),
        )
    };
    let official = rows(&[["0", "1", "a"], ["1", "2", "b"], ["2", "3", "c"]]);
    let swapped = rows(&[["1", "1", "a"], ["0", "2", "b"], ["2", "3", "c"]]);
    let ids = |ids: &[i64]| Some(ids.iter().copied().collect());
    assert!(known::row_ids_traded(&official, &swapped, &ids(&[1, 2])));
    assert!(known::row_ids_traded(&official, &swapped, &None));
    // A row outside the update.
    assert!(!known::row_ids_traded(&official, &swapped, &ids(&[1])));
    // A cell that differs.
    assert!(!known::row_ids_traded(
        &official,
        &rows(&[["1", "1", "a"], ["0", "2", "x"], ["2", "3", "c"]]),
        &None
    ));
    assert!(!known::row_ids_traded(&official, &official, &None));
    // The ids an update a merge followed wrote: by id, or any row.
    let by_id = history(
        vec![c0.clone()],
        vec![
            Step::Insert(vec![(1, vec![V::Null]), (5, vec![V::Null])]),
            Step::Update {
                column: "c0".into(),
                ty: Ty::BigInt,
                value: V::I(1),
                pred: Pred::IdLt(3),
            },
            Step::Merge,
        ],
    );
    assert_eq!(known::merged_update_ids(&by_id, 1), None);
    assert_eq!(known::merged_update_ids(&by_id, 2), Some(ids(&[1])));

    // Official's rows at a maintenance step are dropped only where this
    // crate reports none there.
    let row = |s: &str, kind: &str| vec![s.to_string(), "7".to_string(), kind.to_string()];
    let names = vec!["snapshot_id".to_string(), "rowid".to_string(), "change_type".to_string()];
    let official = (
        names.clone(),
        vec![row("step2", "\"update_preimage\""), row("step1", "\"delete\"")],
    );
    let ((_, o), dropped) =
        known::maintenance_rows(&rewrite, official.clone(), &[row("step1", "\"delete\"")]);
    assert_eq!((o.len(), dropped), (1, 1));
    let ((_, o), dropped) = known::maintenance_rows(
        &rewrite,
        official,
        &[row("step2", "\"delete\""), row("step1", "\"delete\"")],
    );
    assert_eq!(
        (o.len(), dropped),
        (2, 0),
        "a step this crate reports rows at compares"
    );

    // #1490: official's rewrite fails on its inlined-aggregate error only.
    assert!(known::stats_recompute_error(
        &rewrite,
        1,
        "Failed to read inlined-data aggregates"
    ));
    assert!(!known::stats_recompute_error(
        &rewrite,
        0,
        "Failed to read inlined-data aggregates"
    ));
    assert!(!known::stats_recompute_error(&rewrite, 1, "Conflict"));

    // The greatest TIMESTAMPTZ, held inline, fails official's rewrite.
    let max_tstz = "Failed to cast value: timestamp field value out of range: \
                    \"294247-01-10 04:00:54.776806+00\"";
    let zoned = history(
        vec![col("c0", Ty::TsTz)],
        vec![Step::Delete(Some(Pred::IdEq(1))), Step::Rewrite],
    );
    assert!(known::max_timestamptz_recompute_error(&zoned, 1, max_tstz));
    assert!(!known::max_timestamptz_recompute_error(&zoned, 0, max_tstz));
    assert!(!known::max_timestamptz_recompute_error(
        &rewrite, 1, max_tstz
    ));
    assert!(!known::max_timestamptz_recompute_error(
        &zoned, 1, "Conflict"
    ));

    // A NUL byte in a PostgreSQL partition value.
    let nul = history(
        vec![s0.clone()],
        vec![
            Step::Partition(Some(("s0".into(), Transform::Identity))),
            Step::Insert(vec![(1, vec![V::S("a\0b".into())])]),
        ],
    );
    assert!(known::postgres_nul_partition(
        &nul,
        1,
        Kind::Multicatalog,
        "0x00"
    ));
    assert!(!known::postgres_nul_partition(
        &nul,
        1,
        Kind::Sqlite,
        "0x00"
    ));
    assert!(!known::postgres_nul_partition(
        &unstored,
        0,
        Kind::Multicatalog,
        "0x00"
    ));

    // The 1.5.5 oracle's SQL for a NUL partition value does not parse.
    let parse = "Parser Error: unterminated quoted string at or near \"'a\"";
    assert!(known::nul_partition_parser_error(&nul, 1, parse));
    assert!(!known::nul_partition_parser_error(&nul, 1, "Conflict"));
    assert!(
        !known::nul_partition_parser_error(&unstored, 0, parse),
        "unpartitioned"
    );
    let mut flushed = nul.clone();
    flushed.steps.push(Step::Flush);
    assert!(known::nul_partition_parser_error(&flushed, 2, parse));
    assert!(!known::nul_partition_parser_error(&flushed, 0, parse));

    // MySQL's fallback to Parquet for a value its inlined columns cannot hold.
    let nan = history(
        vec![d.clone()],
        vec![Step::Insert(vec![(1, vec![V::F(f64::NAN)])])],
    );
    assert!(known::mysql_fallback(&nan, 0, Kind::Mysql));
    assert!(!known::mysql_fallback(&nan, 0, Kind::Sqlite));
    assert!(!known::mysql_fallback(&zeros, 1, Kind::Mysql));
}

/// The catalog-read models fire on official's catalog where the upstream bug
/// is, and not on the same history without its trigger: #1603's misdated
/// deletions, and #1377's statistics recompute at a rewrite.
#[test]
fn known_catalog_models_fire_only_on_their_shapes() {
    let run = |steps: Vec<Step>| {
        let dir = tempfile::TempDir::new().unwrap();
        let data = dir.path().join("official_data");
        std::fs::create_dir_all(&data).unwrap();
        let mut official = Official::new(dir.path().join("official.ducklake"), &data);
        official
            .run("CREATE TABLE t (id BIGINT, c1 DECIMAL(18,3))")
            .unwrap();
        official
            .run("CALL lake.set_option('data_inlining_row_limit', 0, table_name => 't')")
            .unwrap();
        let history = History {
            columns: vec![Col {
                name: "c1".into(),
                ty: Ty::Decimal,
            }],
            limit: Some(0),
            steps,
        };
        let mut stats = vec![Run::stats_rows(&Metadata::Duckdb(official.metadata.clone()))];
        for step in 0..history.steps.len() {
            official.reconnect();
            official.apply(&history, step).unwrap();
            stats.push(Run::stats_rows(&Metadata::Duckdb(
                official.metadata.clone(),
            )));
        }
        let misdated = known::misdated_delete_snapshots(&official.connection).unwrap();
        (dir, misdated, stats)
    };
    let insert = |id: i64, value: V| Step::Insert(vec![(id, vec![value])]);
    // The repro: two files merged, then a row deleted to a delete file
    // without per-row snapshots. Only the deletion's snapshot is named: a
    // window holding the insert alone reads right.
    let (_dir, misdated, _) = run(vec![
        insert(2, V::Null),
        insert(3, V::D(6847)),
        Step::Merge,
        Step::Delete(Some(Pred::IsNull("c1".into()))),
    ]);
    assert_eq!(misdated.len(), 1, "{misdated:?}");
    assert!(misdated[0] == [5] || misdated[0] == [4], "{misdated:?}");
    // Without the merge the deleted file has no `partial_max`.
    let (_dir, misdated, _) = run(vec![
        insert(2, V::Null),
        insert(3, V::D(6847)),
        Step::Delete(Some(Pred::IsNull("c1".into()))),
    ]);
    assert!(misdated.is_empty(), "{misdated:?}");
    // A flush of inlined rows deleted at several snapshots writes a delete
    // file with per-row snapshots, which official reads right in every
    // window, also those after its begin_snapshot.
    let (_dir, misdated, _) = run(vec![
        Step::Limit(100),
        Step::Insert((1..=3).map(|id| (id, vec![V::D(id)])).collect()),
        Step::Delete(Some(Pred::IdEq(1))),
        Step::Delete(Some(Pred::IdEq(2))),
        Step::Flush,
        Step::Delete(Some(Pred::IdEq(3))),
    ]);
    assert!(misdated.is_empty(), "{misdated:?}");
    // A row updated while inlined is flushed as two versions under one row
    // id; deleting it later names the pair of the two deletions.
    let (_dir, misdated, _) = run(vec![
        Step::Limit(100),
        Step::Insert((1..=3).map(|id| (id, vec![V::D(id)])).collect()),
        Step::Update {
            column: "c1".into(),
            ty: Ty::Decimal,
            value: V::D(9),
            pred: Pred::IdEq(3),
        },
        Step::Flush,
        Step::Delete(Some(Pred::IdEq(3))),
    ]);
    assert_eq!(misdated.len(), 1, "{misdated:?}");
    assert_eq!(misdated[0].len(), 2, "{misdated:?}");

    // A rewrite after a delete recomputes official's statistics; a merge
    // does not.
    let rows = |stats: &(Rows, Rows)| stats.0.clone().unwrap().1;
    let (_dir, _, stats) = run(vec![
        Step::Insert((1..=20).map(|id| (id, vec![V::D(id * 1000)])).collect()),
        Step::Delete(Some(Pred::IdLt(15))),
        Step::Rewrite,
    ]);
    assert_ne!(rows(&stats[2]), rows(&stats[3]), "the rewrite recomputes");
    let (_dir, _, stats) = run(vec![
        insert(1, V::D(1000)),
        insert(2, V::D(2000)),
        Step::Merge,
    ]);
    assert_eq!(rows(&stats[2]), rows(&stats[3]), "a merge keeps them");
}

/// A `DOUBLE` halfway between two shortest spellings records its bounds in
/// the spelling DuckDB gives it (the even last digit), inlined or in Parquet.
#[tokio::test(flavor = "multi_thread")]
async fn regression_double_bound_ties_spelled_as_duckdb() {
    for limit in [Some(0), Some(1)] {
        let history = History {
            columns: vec![Col {
                name: "c1".into(),
                ty: Ty::Double,
            }],
            limit,
            steps: vec![
                Step::Insert(vec![(6, vec![V::F(1973907678291657.3)])]),
                Step::Insert(vec![
                    (9, vec![V::Null]),
                    (10, vec![V::F(-768223024721377.3)]),
                ]),
            ],
        };
        check(&history, Kind::Sqlite).await;
        check(&history, Kind::Duckdb).await;
    }
}

/// A partitioned insert whose NULL partition's file is written before a file
/// holding values keeps the values' table-wide bounds, as upstream main does.
#[tokio::test(flavor = "multi_thread")]
async fn regression_partitioned_null_file_keeps_bounds() {
    for ty in [Ty::BigInt, Ty::TsTz, Ty::Varchar] {
        let value = match ty {
            Ty::BigInt => V::I(4),
            Ty::TsTz => V::Ts(1_702_602_800_000_000),
            _ => V::S("x".into()),
        };
        let history = History {
            columns: vec![Col {
                name: "c0".into(),
                ty,
            }],
            limit: Some(0),
            steps: vec![
                Step::Partition(Some(("c0".into(), Transform::Identity))),
                Step::Insert(vec![(1, vec![V::Null]), (2, vec![value])]),
            ],
        };
        check(&history, Kind::Sqlite).await;
        check(&history, Kind::Duckdb).await;
    }
}

/// A Parquet file of strings, one holding a NUL byte inside the file's
/// bounds, records the bounds official records.
#[tokio::test(flavor = "multi_thread")]
async fn regression_nul_string_inside_file_bounds() {
    for limit in [Some(0), None] {
        let history = History {
            columns: vec![Col {
                name: "c2".into(),
                ty: Ty::Varchar,
            }],
            limit,
            steps: vec![Step::Insert(vec![
                (1, vec![V::S("%_\\".into())]),
                (2, vec![V::S("a\0b".into())]),
                (3, vec![V::S("x".into())]),
            ])],
        };
        check(&history, Kind::Sqlite).await;
    }
}

/// Inlined strings, one holding a NUL byte inside the bounds the others
/// set, keep the table-wide bounds official keeps.
#[tokio::test(flavor = "multi_thread")]
async fn regression_inlined_nul_string_inside_bounds() {
    let history = History {
        columns: vec![Col {
            name: "c1".into(),
            ty: Ty::Varchar,
        }],
        limit: Some(100),
        steps: vec![
            Step::Insert(vec![
                (1, vec![V::S("b".into())]),
                (3, vec![V::S("".into())]),
            ]),
            Step::Insert(vec![
                (4, vec![V::S("x".into())]),
                (5, vec![V::S("a\0b".into())]),
            ]),
        ],
    };
    check(&history, Kind::Sqlite).await;
    check(&history, Kind::Duckdb).await;
}

/// A checkpoint flushes a partitioned table's inlined rows to one file per
/// partition, as official's does, and its merge keeps them apart.
#[tokio::test(flavor = "multi_thread")]
async fn regression_checkpoint_keeps_partitions() {
    let history = History {
        columns: vec![
            Col {
                name: "c0".into(),
                ty: Ty::TsNs,
            },
            Col {
                name: "c1".into(),
                ty: Ty::Int,
            },
        ],
        limit: None,
        steps: vec![
            Step::Partition(Some(("c1".into(), Transform::Identity))),
            Step::Insert(vec![
                (1, vec![V::Tn(1_700_604_800_000_020_664), V::I(0)]),
                (2, vec![V::Null, V::Null]),
            ]),
            Step::Checkpoint,
        ],
    };
    for tail in [vec![Step::Flush], vec![Step::Flush, Step::Merge]] {
        let mut flushed = history.clone();
        flushed.steps.pop();
        flushed.steps.extend(tail);
        check(&flushed, Kind::Sqlite).await;
    }
    check(&history, Kind::Sqlite).await;
    check(&history, Kind::Duckdb).await;
}

/// A partition file whose only string holds a NUL byte keeps that bound in its
/// file statistics, as official's do, so the table-wide bounds the other
/// files set survive the merge.
#[tokio::test(flavor = "multi_thread")]
async fn regression_nul_partition_file_keeps_table_bounds() {
    let history = History {
        columns: vec![Col {
            name: "c1".into(),
            ty: Ty::Varchar,
        }],
        limit: Some(0),
        steps: vec![
            Step::Partition(Some(("c1".into(), Transform::Identity))),
            Step::Insert(vec![
                (1, vec![V::S("".into())]),
                (2, vec![V::S("a\0b".into())]),
                (3, vec![V::S("zzz".into())]),
            ]),
        ],
    };
    check(&history, Kind::Sqlite).await;
    check(&history, Kind::Duckdb).await;
}

/// Official reports identical update pairs at a rewrite of a flushed file
/// (the #1571 family), where this crate reports no row.
#[tokio::test(flavor = "multi_thread")]
async fn regression_rewrite_reports_identical_pairs() {
    let history = History {
        columns: vec![
            Col {
                name: "c0".into(),
                ty: Ty::Date,
            },
            Col {
                name: "c1".into(),
                ty: Ty::Time,
            },
            Col {
                name: "c2".into(),
                ty: Ty::Decimal,
            },
            Col {
                name: "c3".into(),
                ty: Ty::Date,
            },
        ],
        limit: None,
        steps: vec![
            Step::Insert(vec![
                (
                    4,
                    vec![V::Date(19030), V::Time(69434000000), V::D(4439), V::Date(19015)],
                ),
                (
                    5,
                    vec![V::Date(-755687), V::Time(23638000000), V::D(-3205), V::Date(2147483647)],
                ),
                (
                    6,
                    vec![V::Date(-2147483646), V::Time(37137000000), V::D(-1), V::Null],
                ),
                (7, vec![V::Null, V::Time(84058000000), V::D(6394), V::Null]),
                (
                    8,
                    vec![V::Date(19021), V::Time(37744000000), V::D(999999999999999999), V::Null],
                ),
                (
                    9,
                    vec![V::Date(19020), V::Time(76399000000), V::D(249), V::Date(19009)],
                ),
                (
                    10,
                    vec![V::Date(-2147483646), V::Time(0), V::D(1), V::Date(108853222)],
                ),
                (
                    11,
                    vec![V::Date(-2147483646), V::Time(1), V::D(999999999999999999), V::Null],
                ),
                (
                    12,
                    vec![V::Date(19015), V::Time(7730000000), V::D(-971), V::Date(19036)],
                ),
                (
                    13,
                    vec![V::Date(108853222), V::Null, V::D(-3473), V::Date(19018)],
                ),
                (
                    14,
                    vec![V::Date(19020), V::Time(68956000000), V::D(-1), V::Date(-719162)],
                ),
                (
                    15,
                    vec![V::Null, V::Time(19263000000), V::D(4221), V::Date(19022)],
                ),
                (
                    16,
                    vec![V::Date(19012), V::Time(60065000000), V::D(-9966), V::Date(19000)],
                ),
                (
                    17,
                    vec![V::Date(19014), V::Time(0), V::D(0), V::Date(19008)],
                ),
                (
                    18,
                    vec![V::Date(-755687), V::Time(69085000000), V::D(2258), V::Date(-2147483647)],
                ),
                (
                    19,
                    vec![V::Null, V::Time(45296500000), V::D(9447), V::Date(19034)],
                ),
                (
                    20,
                    vec![V::Date(-719162), V::Time(75271000000), V::D(1), V::Date(2932896)],
                ),
                (
                    21,
                    vec![V::Date(-2147483647), V::Time(86400000000), V::D(-1705), V::Date(19022)],
                ),
                (22, vec![V::Null, V::Time(0), V::D(-929), V::Date(19027)]),
                (
                    23,
                    vec![V::Date(19003), V::Time(30400000000), V::Null, V::Date(19020)],
                ),
            ]),
            Step::Update {
                column: "c2".into(),
                ty: Ty::Decimal,
                value: V::D(8464),
                pred: Pred::IdIn(vec![18, 9, 24]),
            },
            Step::AddColumn(
                Col {
                    name: "a1".into(),
                    ty: Ty::Decimal38,
                },
                None,
            ),
            Step::Rewrite,
            Step::Flush,
            Step::Rewrite,
            Step::Partition(Some(("c0".into(), Transform::Year))),
            Step::Insert(vec![
                (
                    24,
                    vec![V::Date(19010), V::Null, V::D(-267), V::Date(19014), V::Big(-8797000000)],
                ),
                (
                    25,
                    vec![
                        V::Date(2147483647),
                        V::Time(10971000000),
                        V::D(4957),
                        V::Date(19011),
                        V::Big(-1),
                    ],
                ),
            ]),
            Step::Partition(Some(("c3".into(), Transform::Year))),
            Step::Update {
                column: "c1".into(),
                ty: Ty::Time,
                value: V::Time(23390000000),
                pred: Pred::IdIn(vec![23, 13, 24]),
            },
            Step::Rewrite,
            Step::Update {
                column: "c1".into(),
                ty: Ty::Time,
                value: V::Time(2238000000),
                pred: Pred::IdIn(vec![3, 24, 26]),
            },
        ],
    };
    check(&history, Kind::Sqlite).await;
}

/// A merge numbers an update's rows by their order in its file, which
/// official's scan does not fix: the row ids may swap, the rows may not.
#[tokio::test(flavor = "multi_thread")]
async fn regression_merged_update_row_order() {
    let history = History {
        columns: vec![
            Col {
                name: "c0".into(),
                ty: Ty::Blob,
            },
            Col {
                name: "c1".into(),
                ty: Ty::SmallInt,
            },
            Col {
                name: "c2".into(),
                ty: Ty::Decimal38,
            },
            Col {
                name: "c3".into(),
                ty: Ty::SmallInt,
            },
        ],
        limit: None,
        steps: vec![
            Step::Limit(0),
            Step::Update {
                column: "c0".into(),
                ty: Ty::Blob,
                value: V::Bytes(vec![
                    98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98,
                    98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98,
                    98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98,
                    98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98,
                    98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98,
                    98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98,
                    98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98,
                    98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98,
                    98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98,
                    98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98,
                    98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98,
                    98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98,
                    98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98,
                    98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98,
                    98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98,
                ]),
                pred: Pred::Eq("c2".into(), Ty::Decimal38, V::Big(2501000000)),
            },
            Step::Insert(vec![(2, vec![V::Null, V::Null, V::Null, V::Null])]),
            Step::Merge,
            Step::Insert(vec![(
                3,
                vec![V::Null, V::I(32767), V::Big(4821000000), V::I(5)],
            )]),
            Step::Update {
                column: "c0".into(),
                ty: Ty::Blob,
                value: V::Null,
                pred: Pred::IdIn(vec![2, 3, 3]),
            },
            Step::Insert(vec![
                (4, vec![V::Bytes(vec![]), V::Null, V::Null, V::I(8)]),
                (
                    5,
                    vec![V::Bytes(vec![121]), V::I(-10), V::Big(12345678901234567890123), V::I(-1)],
                ),
                (
                    6,
                    vec![V::Bytes(vec![0, 1, 255]), V::I(-7), V::Big(-1887000000), V::I(0)],
                ),
                (
                    7,
                    vec![V::Bytes(vec![242]), V::I(32767), V::Big(-5038000000), V::Null],
                ),
                (
                    8,
                    vec![
                        V::Bytes(vec![
                            98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98,
                            98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98,
                            98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98,
                            98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98,
                            98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98,
                            98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98,
                            98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98,
                            98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98,
                            98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98,
                            98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98,
                            98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98,
                            98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98,
                            98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98,
                            98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98,
                            98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98,
                            98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98,
                            98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98,
                        ]),
                        V::Null,
                        V::Big(1517000000),
                        V::I(-2),
                    ],
                ),
                (
                    9,
                    vec![V::Bytes(vec![250]), V::I(-9), V::Big(-5871000000), V::I(-8)],
                ),
                (
                    10,
                    vec![V::Bytes(vec![0, 1, 255]), V::I(8), V::Big(-7124000000), V::I(0)],
                ),
                (11, vec![V::Bytes(vec![41]), V::I(6), V::Null, V::I(-3)]),
                (
                    12,
                    vec![
                        V::Bytes(vec![110]),
                        V::I(2),
                        V::Big(99999999999999999999999999999999999999),
                        V::I(-10),
                    ],
                ),
                (
                    13,
                    vec![V::Bytes(vec![184]), V::I(-8), V::Big(6758000000), V::I(1)],
                ),
                (
                    14,
                    vec![V::Bytes(vec![1]), V::I(-6), V::Big(6451000000), V::I(32767)],
                ),
                (
                    15,
                    vec![V::Bytes(vec![231]), V::I(9), V::Big(-7425000000), V::I(32767)],
                ),
                (
                    16,
                    vec![V::Bytes(vec![255]), V::I(-6), V::Big(-7376000000), V::Null],
                ),
                (
                    17,
                    vec![V::Bytes(vec![192]), V::I(-1), V::Big(4874000000), V::I(0)],
                ),
                (
                    18,
                    vec![V::Bytes(vec![97, 98, 39, 92]), V::I(6), V::Big(8036000000), V::I(-1)],
                ),
                (19, vec![V::Null, V::I(0), V::Big(-5457000000), V::I(7)]),
                (
                    20,
                    vec![V::Bytes(vec![36]), V::I(7), V::Big(-2225000000), V::Null],
                ),
                (
                    21,
                    vec![V::Bytes(vec![97, 98, 39, 92]), V::Null, V::Big(9344000000), V::I(9)],
                ),
                (
                    22,
                    vec![
                        V::Bytes(vec![70]),
                        V::I(-1),
                        V::Big(-99999999999999999999999999999999999999),
                        V::I(-5),
                    ],
                ),
            ]),
            Step::Flush,
            Step::Checkpoint,
        ],
    };
    check(&history, Kind::Sqlite).await;
}

/// A merge's output lands in the Hive directory official names for its
/// partition: the calendar key's name for the live generation, and its
/// sources' directory for a retired one.
#[tokio::test(flavor = "multi_thread")]
async fn regression_merge_names_hive_directories_as_official() {
    let insert = |id: i64, micros: i64| Step::Insert(vec![(id, vec![V::Ts(micros)])]);
    let column = vec![Col {
        name: "c1".into(),
        ty: Ty::Ts,
    }];
    let live = History {
        columns: column.clone(),
        limit: Some(0),
        steps: vec![
            Step::Partition(Some(("c1".into(), Transform::Month))),
            insert(1, 1_700_000_000_000_000),
            insert(2, 1_700_000_100_000_000),
            Step::Merge,
        ],
    };
    let retired = History {
        columns: column,
        limit: Some(0),
        steps: vec![
            Step::Partition(Some(("c1".into(), Transform::Month))),
            insert(1, 1_700_000_000_000_000),
            insert(2, 1_700_000_100_000_000),
            Step::Partition(None),
            Step::Merge,
        ],
    };
    for history in [live, retired] {
        check(&history, Kind::Sqlite).await;
        check(&history, Kind::Duckdb).await;
    }
}

/// Expiry removes a retired file's catalog row while an inlined deletion
/// still names its id; a file written later must not take that id, or the
/// deletion applies to it and its rows vanish.
#[tokio::test(flavor = "multi_thread")]
async fn regression_file_ids_are_never_reused() {
    let history = History {
        columns: vec![Col {
            name: "c2".into(),
            ty: Ty::BigInt,
        }],
        limit: Some(10),
        steps: vec![
            Step::Insert((1..=12).map(|id| (id, vec![V::I(id)])).collect()),
            Step::Update {
                column: "c2".into(),
                ty: Ty::BigInt,
                value: V::I(0),
                pred: Pred::IdGt(9),
            },
            Step::Rewrite,
            Step::Expire,
            Step::Checkpoint,
        ],
    };
    check(&history, Kind::Sqlite).await;
    check(&history, Kind::Duckdb).await;
}

/// A partitioned write whose files include one holding only NaN of a float
/// column, merged into statistics that keep no NaN flag, leaves no row with
/// neither bound (one DuckDB would read as holding no value).
#[tokio::test(flavor = "multi_thread")]
async fn regression_nan_only_file_leaves_no_boundless_row() {
    let history = History {
        columns: vec![
            Col {
                name: "c1".into(),
                ty: Ty::Float,
            },
            Col {
                name: "c2".into(),
                ty: Ty::Int,
            },
        ],
        limit: Some(10),
        steps: vec![
            Step::Insert(vec![(1, vec![V::R(1.0), V::I(1)])]),
            Step::Partition(Some(("c2".into(), Transform::Identity))),
            Step::Limit(0),
            Step::Insert(vec![
                (2, vec![V::R(2.0), V::I(1)]),
                (3, vec![V::R(f32::NAN), V::I(2)]),
            ]),
        ],
    };
    check(&history, Kind::Sqlite).await;
    check(&history, Kind::Duckdb).await;
}

/// A flush of a column's only (NaN) inlined value leaves the table-wide row
/// official leaves: the inlined `nan` bounds are no value's.
#[tokio::test(flavor = "multi_thread")]
async fn regression_flushed_nan_only_column() {
    let history = History {
        columns: vec![Col {
            name: "c2".into(),
            ty: Ty::Float,
        }],
        limit: Some(10),
        steps: vec![Step::Insert(vec![(1, vec![V::R(f32::NAN)])]), Step::Flush],
    };
    check(&history, Kind::Sqlite).await;
    check(&history, Kind::Duckdb).await;
}

/// A partitioned write whose NaN-only file comes with files holding values
/// leaves no table-wide row with neither bound, so a later write cannot
/// narrow it below the values.
#[tokio::test(flavor = "multi_thread")]
async fn regression_nan_only_file_leaves_no_narrowable_row() {
    let history = History {
        columns: vec![
            Col {
                name: "c1".into(),
                ty: Ty::Float,
            },
            Col {
                name: "c2".into(),
                ty: Ty::Int,
            },
        ],
        limit: Some(0),
        steps: vec![
            Step::Partition(Some(("c2".into(), Transform::Identity))),
            Step::Insert(vec![
                (1, vec![V::R(-3.0), V::I(1)]),
                (2, vec![V::R(f32::NAN), V::I(2)]),
                (3, vec![V::R(1e20), V::I(3)]),
            ]),
            Step::Insert(vec![(4, vec![V::R(2.75), V::I(1)])]),
        ],
    };
    check(&history, Kind::Sqlite).await;
    check(&history, Kind::Duckdb).await;
}

/// A `TIMESTAMPTZ` year partition records a year before 1 as DuckDB gives it,
/// counting the era (44 BC is `44`), where a `TIMESTAMP`'s is `-43`.
#[tokio::test(flavor = "multi_thread")]
async fn regression_zoned_bc_year_partition() {
    for ty in [Ty::TsTz, Ty::Ts] {
        let history = History {
            columns: vec![Col {
                name: "c0".into(),
                ty,
            }],
            limit: Some(0),
            steps: vec![
                Step::Partition(Some(("c0".into(), Transform::Year))),
                Step::Insert(vec![
                    (1, vec![V::Ts(-63_517_778_999_500_000)]),
                    (2, vec![V::Ts(1_702_685_600_000_000)]),
                ]),
            ],
        };
        check(&history, Kind::Sqlite).await;
    }
}

/// The 1.5.5 oracle's flush of a NUL partition value fails to parse its own
/// SQL; this crate's commits, and the history goes on with the two catalogs
/// holding the same rows.
#[tokio::test(flavor = "multi_thread")]
async fn regression_official_nul_partition_flush_fails() {
    let history = History {
        columns: vec![Col {
            name: "c0".into(),
            ty: Ty::Map,
        }],
        limit: Some(0),
        steps: vec![
            Step::AddColumn(
                Col {
                    name: "a1".into(),
                    ty: Ty::Varchar,
                },
                None,
            ),
            Step::Limit(10),
            Step::Partition(Some(("a1".into(), Transform::Identity))),
            Step::Insert(vec![(
                1,
                vec![V::Map(vec![("k".into(), Some(9))]), V::Null],
            )]),
            Step::Flush,
            Step::Insert(vec![(2, vec![V::Map(vec![]), V::S("a\0b".into())])]),
            Step::Flush,
            Step::Partition(None),
        ],
    };
    check(&history, Kind::Sqlite).await;
}

/// Official keeps a float row a NaN-only flush merges into; this crate keeps
/// none (bounds unknown), and that row is not compared.
#[tokio::test(flavor = "multi_thread")]
async fn regression_official_keeps_nan_float_row() {
    let history = History {
        columns: vec![
            Col {
                name: "c0".into(),
                ty: Ty::Varchar,
            },
            Col {
                name: "c1".into(),
                ty: Ty::Float,
            },
        ],
        limit: None,
        steps: vec![
            Step::Insert(vec![
                (1, vec![V::S("".into()), V::Null]),
                (2, vec![V::S("zzz".into()), V::R(-1.75)]),
                (3, vec![V::S("b".into()), V::R(0.25)]),
            ]),
            Step::Partition(Some(("c0".into(), Transform::Identity))),
            Step::Update {
                column: "c0".into(),
                ty: Ty::Varchar,
                value: V::S("zzz".into()),
                pred: Pred::IdEq(1),
            },
            Step::Flush,
            Step::Update {
                column: "c1".into(),
                ty: Ty::Float,
                value: V::R(f32::NAN),
                pred: Pred::IdIn(vec![3, 4, 1]),
            },
            Step::Merge,
            Step::Partition(Some(("c0".into(), Transform::Identity))),
            Step::Update {
                column: "c1".into(),
                ty: Ty::Float,
                value: V::R(1e-45),
                pred: Pred::IdIn(vec![1, 3, 1]),
            },
            Step::Flush,
            Step::Flush,
            Step::Insert(vec![
                (4, vec![V::Null, V::R(f32::INFINITY)]),
                (5, vec![V::S("x".into()), V::R(1.1754944e-38)]),
                (6, vec![V::S("a\0b".into()), V::R(f32::NAN)]),
                (7, vec![V::Null, V::R(0.0)]),
                (8, vec![V::S("%_\\".into()), V::Null]),
                (9, vec![V::S("%_\\".into()), V::R(0.0)]),
                (10, vec![V::S("héllo ☃ 𝄞".into()), V::R(1e-40)]),
                (11, vec![V::S("x".into()), V::R(-1e20)]),
                (12, vec![V::S("a\0b".into()), V::R(1.1754944e-38)]),
                (13, vec![V::S("c".into()), V::R(0.1)]),
                (14, vec![V::S("b".into()), V::R(-1.5)]),
                (15, vec![V::S("a".into()), V::R(-0.0)]),
            ]),
        ],
    };
    check(&history, Kind::Sqlite).await;
}
