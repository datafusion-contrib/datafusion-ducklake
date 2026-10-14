// Canonical text of Arrow values, shared by the differential fuzz harness for
// the two Arrow versions it reads: this crate's and the one the `duckdb` crate
// re-exports. The including module brings an `arrow_impl` path into scope.
//
// The text is the same for the same logical value whichever engine produced
// it: integers and decimals in decimal, floats in Rust's shortest form with
// every NaN as `NaN`, temporal values as their count of nanoseconds (days for
// a date), binary in hex, strings quoted, nested values recursively.

use self::arrow_impl::array::{Array, AsArray};
use self::arrow_impl::datatypes::{
    DataType, Date32Type, Decimal128Type, Float32Type, Float64Type, Int8Type, Int16Type,
    Int32Type, Int64Type, IntervalUnit, Time32MillisecondType, Time32SecondType,
    Time64MicrosecondType, Time64NanosecondType, TimeUnit, TimestampMicrosecondType,
    TimestampMillisecondType, TimestampNanosecondType, TimestampSecondType, UInt8Type,
    UInt16Type, UInt32Type, UInt64Type,
};
use self::arrow_impl::record_batch::RecordBatch;

/// The canonical text of the value at `row` of `array`.
pub(crate) fn cell(array: &dyn Array, row: usize) -> String {
    if array.is_null(row) {
        return "NULL".to_string();
    }
    match array.data_type() {
        DataType::Boolean => array.as_boolean().value(row).to_string(),
        DataType::Int8 => array.as_primitive::<Int8Type>().value(row).to_string(),
        DataType::Int16 => array.as_primitive::<Int16Type>().value(row).to_string(),
        DataType::Int32 => array.as_primitive::<Int32Type>().value(row).to_string(),
        DataType::Int64 => array.as_primitive::<Int64Type>().value(row).to_string(),
        DataType::UInt8 => array.as_primitive::<UInt8Type>().value(row).to_string(),
        DataType::UInt16 => array.as_primitive::<UInt16Type>().value(row).to_string(),
        DataType::UInt32 => array.as_primitive::<UInt32Type>().value(row).to_string(),
        DataType::UInt64 => array.as_primitive::<UInt64Type>().value(row).to_string(),
        DataType::Float32 => {
            let value = array.as_primitive::<Float32Type>().value(row);
            if value.is_nan() {
                "NaN".to_string()
            } else {
                format!("{value:?}")
            }
        },
        DataType::Float64 => {
            let value = array.as_primitive::<Float64Type>().value(row);
            if value.is_nan() {
                "NaN".to_string()
            } else {
                format!("{value:?}")
            }
        },
        DataType::Decimal128(_, scale) => {
            let value = array.as_primitive::<Decimal128Type>().value(row);
            format!("{value}e-{scale}")
        },
        DataType::Utf8 => format!("{:?}", array.as_string::<i32>().value(row)),
        DataType::LargeUtf8 => format!("{:?}", array.as_string::<i64>().value(row)),
        DataType::Utf8View => format!("{:?}", array.as_string_view().value(row)),
        DataType::Binary => hex(array.as_binary::<i32>().value(row)),
        DataType::LargeBinary => hex(array.as_binary::<i64>().value(row)),
        DataType::BinaryView => hex(array.as_binary_view().value(row)),
        DataType::FixedSizeBinary(_) => hex(array.as_fixed_size_binary().value(row)),
        DataType::Date32 => format!("d{}", array.as_primitive::<Date32Type>().value(row)),
        DataType::Time32(TimeUnit::Second) => format!(
            "t{}",
            i128::from(array.as_primitive::<Time32SecondType>().value(row)) * 1_000_000_000
        ),
        DataType::Time32(_) => format!(
            "t{}",
            i128::from(array.as_primitive::<Time32MillisecondType>().value(row)) * 1_000_000
        ),
        DataType::Time64(TimeUnit::Microsecond) => format!(
            "t{}",
            i128::from(array.as_primitive::<Time64MicrosecondType>().value(row)) * 1_000
        ),
        DataType::Time64(_) => format!(
            "t{}",
            array.as_primitive::<Time64NanosecondType>().value(row)
        ),
        DataType::Timestamp(unit, _) => {
            let (value, per) = match unit {
                TimeUnit::Second => (
                    array.as_primitive::<TimestampSecondType>().value(row),
                    1_000_000_000i128,
                ),
                TimeUnit::Millisecond => (
                    array.as_primitive::<TimestampMillisecondType>().value(row),
                    1_000_000,
                ),
                TimeUnit::Microsecond => (
                    array.as_primitive::<TimestampMicrosecondType>().value(row),
                    1_000,
                ),
                TimeUnit::Nanosecond => (
                    array.as_primitive::<TimestampNanosecondType>().value(row),
                    1,
                ),
            };
            // `infinity` and `-infinity` are the same sentinel in every unit.
            match value {
                i64::MAX => "ts+inf".to_string(),
                v if v == -i64::MAX => "ts-inf".to_string(),
                v => format!("ts{}", i128::from(v) * per),
            }
        },
        DataType::Interval(IntervalUnit::MonthDayNano) => {
            let value = array
                .as_primitive::<arrow_impl::datatypes::IntervalMonthDayNanoType>()
                .value(row);
            format!("i{}m{}d{}n", value.months, value.days, value.nanoseconds)
        },
        DataType::List(_) => {
            let list = array.as_list::<i32>();
            let values = list.value(row);
            let items = (0..values.len())
                .map(|index| cell(values.as_ref(), index))
                .collect::<Vec<_>>();
            format!("[{}]", items.join(", "))
        },
        DataType::LargeList(_) => {
            let list = array.as_list::<i64>();
            let values = list.value(row);
            let items = (0..values.len())
                .map(|index| cell(values.as_ref(), index))
                .collect::<Vec<_>>();
            format!("[{}]", items.join(", "))
        },
        DataType::Struct(fields) => {
            let structure = array.as_struct();
            let items = fields
                .iter()
                .enumerate()
                .map(|(index, field)| {
                    format!("{}: {}", field.name(), cell(structure.column(index).as_ref(), row))
                })
                .collect::<Vec<_>>();
            format!("{{{}}}", items.join(", "))
        },
        DataType::Map(_, _) => {
            let map = array.as_map();
            let entries = map.value(row);
            let items = (0..entries.len())
                .map(|index| {
                    format!(
                        "{}={}",
                        cell(entries.column(0).as_ref(), index),
                        cell(entries.column(1).as_ref(), index)
                    )
                })
                .collect::<Vec<_>>();
            format!("{{{}}}", items.join(", "))
        },
        other => format!("<unsupported {other}>"),
    }
}

fn hex(bytes: &[u8]) -> String {
    let mut text = String::from("x");
    for byte in bytes {
        text.push_str(&format!("{byte:02x}"));
    }
    text
}

/// The column names and canonical rows of `batches`.
pub(crate) fn rows(batches: &[RecordBatch]) -> (Vec<String>, Vec<Vec<String>>) {
    let mut names = Vec::new();
    let mut rows = Vec::new();
    for batch in batches {
        names = batch
            .schema()
            .fields()
            .iter()
            .map(|field| field.name().to_string())
            .collect();
        for row in 0..batch.num_rows() {
            rows.push(
                (0..batch.num_columns())
                    .map(|column| cell(batch.column(column).as_ref(), row))
                    .collect(),
            );
        }
    }
    (names, rows)
}
