//! Type mapping from DuckLake types to Arrow types

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::metadata_provider::{
    DuckLakeNameMapping, DuckLakeNameMappingEntry, DuckLakeTableColumn, DuckLakeTableField,
};
use crate::{DuckLakeError, Result};
use arrow::datatypes::{DataType, Field, IntervalUnit, Schema, TimeUnit};
use datafusion::common::tree_node::{Transformed, TransformedResult, TreeNode};
use datafusion::common::{Result as DataFusionResult, ScalarValue};
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_expr::expressions::{self, Column};
use datafusion::physical_expr_adapter::{
    DefaultPhysicalExprAdapter, PhysicalExprAdapter, PhysicalExprAdapterFactory,
};
use parquet::arrow::PARQUET_FIELD_ID_META_KEY;
use parquet::file::metadata::ParquetMetaData;

pub(crate) const MAX_NESTED_TYPE_DEPTH: usize = 128;

pub(crate) const INITIAL_DEFAULT_METADATA_KEY: &str = "ducklake.initial_default";

#[derive(Debug)]
pub(crate) struct DuckLakeDefaultExprAdapterFactory;

impl PhysicalExprAdapterFactory for DuckLakeDefaultExprAdapterFactory {
    fn create(
        &self,
        logical_file_schema: Arc<Schema>,
        physical_file_schema: Arc<Schema>,
    ) -> DataFusionResult<Arc<dyn PhysicalExprAdapter>> {
        Ok(Arc::new(DuckLakeDefaultExprAdapter {
            fallback: DefaultPhysicalExprAdapter::new(
                Arc::clone(&logical_file_schema),
                Arc::clone(&physical_file_schema),
            ),
            logical_file_schema,
            physical_file_schema,
        }))
    }
}

#[derive(Debug)]
struct DuckLakeDefaultExprAdapter {
    fallback: DefaultPhysicalExprAdapter,
    logical_file_schema: Arc<Schema>,
    physical_file_schema: Arc<Schema>,
}

impl PhysicalExprAdapter for DuckLakeDefaultExprAdapter {
    fn rewrite(&self, expr: Arc<dyn PhysicalExpr>) -> DataFusionResult<Arc<dyn PhysicalExpr>> {
        let rewritten = expr
            .transform(|expr| {
                let Some(column) = expr.downcast_ref::<Column>() else {
                    return Ok(Transformed::no(expr));
                };
                if self
                    .physical_file_schema
                    .field_with_name(column.name())
                    .is_ok()
                {
                    return Ok(Transformed::no(expr));
                }
                let Ok(field) = self.logical_file_schema.field_with_name(column.name()) else {
                    return Ok(Transformed::no(expr));
                };
                let Some(value) = field.metadata().get(INITIAL_DEFAULT_METADATA_KEY) else {
                    return Ok(Transformed::no(expr));
                };
                let scalar =
                    parse_ducklake_default_scalar(value, field.data_type()).ok_or_else(|| {
                        datafusion::error::DataFusionError::Execution(format!(
                            "Cannot decode initial_default '{value}' for column '{}' as {}",
                            column.name(),
                            field.data_type()
                        ))
                    })?;
                Ok(Transformed::yes(expressions::lit(scalar)))
            })
            .data()?;
        self.fallback.rewrite(rewritten)
    }
}

pub(crate) fn parse_ducklake_scalar(value: &str, data_type: &DataType) -> Option<ScalarValue> {
    match data_type {
        DataType::Boolean => match value.to_ascii_lowercase().as_str() {
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
    }
}

pub(crate) fn parse_ducklake_default_scalar(
    value: &str,
    data_type: &DataType,
) -> Option<ScalarValue> {
    match data_type {
        DataType::Binary => Some(ScalarValue::Binary(Some(parse_duckdb_blob(value)?))),
        DataType::LargeBinary => Some(ScalarValue::LargeBinary(Some(parse_duckdb_blob(value)?))),
        DataType::BinaryView => Some(ScalarValue::BinaryView(Some(parse_duckdb_blob(value)?))),
        _ => parse_ducklake_scalar(value, data_type),
    }
}

fn parse_duckdb_blob(value: &str) -> Option<Vec<u8>> {
    // DuckDB keeps printable bytes verbatim and escapes non-printable bytes as `\xNN`
    let bytes = value.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0;

    while index < bytes.len() {
        if bytes[index] == b'\\' && bytes.get(index + 1) == Some(&b'x') {
            let pair = bytes.get(index + 2..index + 4)?;
            let pair = std::str::from_utf8(pair).ok()?;
            decoded.push(u8::from_str_radix(pair, 16).ok()?);
            index += 4;
        } else {
            decoded.push(bytes[index]);
            index += 1;
        }
    }

    Some(decoded)
}

fn decode_hex(value: &str) -> Option<Vec<u8>> {
    let value = value
        .strip_prefix("\\x")
        .or_else(|| value.strip_prefix("0x"))
        .unwrap_or(value);
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

/// Convert a DuckLake type string to an Arrow DataType
pub fn ducklake_to_arrow_type(ducklake_type: &str) -> Result<DataType> {
    ducklake_to_arrow_type_inner(ducklake_type, 0)
}

fn ducklake_to_arrow_type_inner(ducklake_type: &str, depth: usize) -> Result<DataType> {
    if depth > MAX_NESTED_TYPE_DEPTH {
        return Err(DuckLakeError::UnsupportedType(format!(
            "Type exceeds maximum nesting depth {MAX_NESTED_TYPE_DEPTH}"
        )));
    }
    let type_str = ducklake_type.trim();

    // Handle parameterized types first
    if let Some(decimal_params) = parse_decimal(type_str)? {
        return Ok(decimal_params);
    }

    // Handle list/array types
    if let Some(list_type) = parse_list_type(type_str, depth)? {
        return Ok(list_type);
    }

    if let Some(struct_type) = parse_struct_type(type_str, depth)? {
        return Ok(struct_type);
    }

    if let Some(map_type) = parse_map_type(type_str, depth)? {
        return Ok(map_type);
    }

    // Handle basic types
    match type_str.to_ascii_lowercase().as_str() {
        // Boolean
        "boolean" | "bool" => Ok(DataType::Boolean),

        // Integers
        "int8" | "tinyint" => Ok(DataType::Int8),
        "int16" | "smallint" => Ok(DataType::Int16),
        "int32" | "int" | "integer" => Ok(DataType::Int32),
        "int64" | "bigint" | "long" => Ok(DataType::Int64),
        "uint8" | "utinyint" => Ok(DataType::UInt8),
        "uint16" | "usmallint" => Ok(DataType::UInt16),
        "uint32" | "uint" | "uinteger" => Ok(DataType::UInt32),
        "uint64" | "ubigint" => Ok(DataType::UInt64),

        // Floating point
        "float32" | "float" | "real" => Ok(DataType::Float32),
        "float64" | "double" => Ok(DataType::Float64),

        // Temporal types
        "time" => Ok(DataType::Time64(TimeUnit::Microsecond)),
        "date" => Ok(DataType::Date32),
        "timestamp" => Ok(DataType::Timestamp(TimeUnit::Microsecond, None)),
        "timestamptz" | "timestamp with time zone" => Ok(DataType::Timestamp(
            TimeUnit::Microsecond,
            Some("UTC".into()),
        )),
        "timestamptz_ns" => Ok(DataType::Timestamp(
            TimeUnit::Nanosecond,
            Some("UTC".into()),
        )),
        "timestamp_s" => Ok(DataType::Timestamp(TimeUnit::Second, None)),
        "timestamp_ms" => Ok(DataType::Timestamp(TimeUnit::Millisecond, None)),
        "timestamp_ns" => Ok(DataType::Timestamp(TimeUnit::Nanosecond, None)),
        "interval" => Ok(DataType::Interval(IntervalUnit::MonthDayNano)),

        // String types. Mapped to the "view" layout (Utf8View) rather than Utf8
        // to match DataFusion's default parquet read behaviour: its
        // `schema_force_view_types` option (on by default) rewrites Utf8/LargeUtf8
        // columns to Utf8View during schema inference. Building the scan from an
        // explicit catalog-derived schema bypasses that inference, so the view
        // layout is requested here instead. View arrays avoid the 2 GiB limit on a
        // single i32-offset value buffer and are cheaper to hash/compare for
        // group-by; DataFusion's parquet reader decodes the existing BYTE_ARRAY
        // columns straight into view arrays, so no cast and no data rewrite occurs.
        "varchar" | "text" | "string" => Ok(DataType::Utf8View),
        "json" => Ok(DataType::Utf8View), // JSON stored as UTF8 string

        // Binary types. BinaryView for the same reasons as the string types above.
        "blob" | "binary" | "bytea" => Ok(DataType::BinaryView),
        "uuid" => Ok(DataType::FixedSizeBinary(16)),

        // Geometry types (stored as binary WKB format). Kept as Binary (not
        // promoted to BinaryView): the WKB bytes are consumed by geometry
        // functions that expect a Binary layout.
        "point" | "linestring" | "polygon" | "multipoint" | "multilinestring" | "multipolygon"
        | "geometrycollection" | "linestring z" | "geometry" => Ok(DataType::Binary),

        // Time with timezone - not directly supported, use string
        "timetz" | "time with time zone" => Ok(DataType::Utf8View),

        _ => Err(DuckLakeError::UnsupportedType(ducklake_type.to_string())),
    }
}

/// Convert an Arrow DataType to a DuckLake type string
///
/// This is the reverse of `ducklake_to_arrow_type()`.
pub fn arrow_to_ducklake_type(arrow_type: &DataType) -> Result<String> {
    match arrow_type {
        // Boolean
        DataType::Boolean => Ok("boolean".to_string()),

        // Integers
        DataType::Int8 => Ok("int8".to_string()),
        DataType::Int16 => Ok("int16".to_string()),
        DataType::Int32 => Ok("int32".to_string()),
        DataType::Int64 => Ok("int64".to_string()),
        DataType::UInt8 => Ok("uint8".to_string()),
        DataType::UInt16 => Ok("uint16".to_string()),
        DataType::UInt32 => Ok("uint32".to_string()),
        DataType::UInt64 => Ok("uint64".to_string()),

        // Floating point
        DataType::Float32 => Ok("float32".to_string()),
        DataType::Float64 => Ok("float64".to_string()),

        // Temporal types
        DataType::Date32 | DataType::Date64 => Ok("date".to_string()),
        DataType::Time32(_) | DataType::Time64(_) => Ok("time".to_string()),
        DataType::Timestamp(TimeUnit::Second, None) => Ok("timestamp_s".to_string()),
        DataType::Timestamp(TimeUnit::Millisecond, None) => Ok("timestamp_ms".to_string()),
        DataType::Timestamp(TimeUnit::Microsecond, None) => Ok("timestamp".to_string()),
        DataType::Timestamp(TimeUnit::Nanosecond, None) => Ok("timestamp_ns".to_string()),
        // Tz-aware timestamps. DuckLake distinguishes nanosecond precision
        // (`timestamptz_ns` -> TIMESTAMP_TZ_NS) from microsecond (`timestamptz`
        // -> TIMESTAMP_TZ); collapsing ns into `timestamptz` truncates the
        // served value to µs on read while the physical parquet keeps ns. Second
        // and millisecond tz timestamps have no DuckLake type, so they widen
        // losslessly to µs `timestamptz`.
        DataType::Timestamp(TimeUnit::Nanosecond, Some(_)) => Ok("timestamptz_ns".to_string()),
        DataType::Timestamp(_, Some(_)) => Ok("timestamptz".to_string()),
        DataType::Interval(_) => Ok("interval".to_string()),

        // String types. Utf8View is the canonical read layout (see
        // `ducklake_to_arrow_type`); Utf8/LargeUtf8 map here as well so batches
        // produced by other code paths still round-trip to the same DuckLake type.
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => Ok("varchar".to_string()),

        // Binary types
        DataType::Binary | DataType::LargeBinary | DataType::BinaryView => Ok("blob".to_string()),
        DataType::FixedSizeBinary(16) => Ok("uuid".to_string()),
        DataType::FixedSizeBinary(_) => Ok("blob".to_string()),

        // Decimal types
        DataType::Decimal128(precision, scale) | DataType::Decimal256(precision, scale) => {
            Ok(format!("decimal({}, {})", precision, scale))
        },

        // Null type - map to varchar as there's no direct equivalent
        DataType::Null => Ok("varchar".to_string()),

        // Dictionary keys are an Arrow encoding detail. DuckLake records the logical value
        // type, while Parquet preserves dictionary encoding in the data file.
        DataType::Dictionary(_, value) => arrow_to_ducklake_type(value),

        // List types
        DataType::List(field) | DataType::LargeList(field) => {
            let inner = arrow_to_ducklake_type(field.data_type())?;
            Ok(format!("list<{}>", inner))
        },
        DataType::FixedSizeList(field, _) => {
            let inner = arrow_to_ducklake_type(field.data_type())?;
            Ok(format!("list<{}>", inner))
        },
        DataType::Struct(fields) => {
            let fields = fields
                .iter()
                .map(|field| {
                    Ok(format!(
                        "{}:{}",
                        field.name(),
                        arrow_to_ducklake_type(field.data_type())?
                    ))
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(format!("struct<{}>", fields.join(",")))
        },
        DataType::Map(entries, _) => {
            let DataType::Struct(fields) = entries.data_type() else {
                return Err(DuckLakeError::UnsupportedType(
                    "Arrow map entries must be a struct".to_string(),
                ));
            };
            let [key, value] = fields.as_ref() else {
                return Err(DuckLakeError::UnsupportedType(
                    "Arrow maps must have key and value fields".to_string(),
                ));
            };
            Ok(format!(
                "map<{},{}>",
                arrow_to_ducklake_type(key.data_type())?,
                arrow_to_ducklake_type(value.data_type())?
            ))
        },

        // Other unsupported types
        other => Err(DuckLakeError::UnsupportedType(format!(
            "Arrow type '{}' has no DuckLake equivalent",
            other
        ))),
    }
}

/// Maximum precision for Arrow Decimal256
const DECIMAL_MAX_PRECISION: u8 = 76;

/// Validate decimal precision and scale bounds
fn validate_decimal_precision_scale(precision: u8, scale: i8, type_str: &str) -> Result<()> {
    if precision == 0 {
        return Err(DuckLakeError::UnsupportedType(format!(
            "Decimal precision must be >= 1, got 0 in type '{}'",
            type_str
        )));
    }
    if precision > DECIMAL_MAX_PRECISION {
        return Err(DuckLakeError::UnsupportedType(format!(
            "Decimal precision must be <= {}, got {} in type '{}'",
            DECIMAL_MAX_PRECISION, precision, type_str
        )));
    }
    if scale >= 0 && scale as u8 > precision {
        return Err(DuckLakeError::UnsupportedType(format!(
            "Decimal scale ({}) must not exceed precision ({}) in type '{}'",
            scale, precision, type_str
        )));
    }
    Ok(())
}

/// Pick the Arrow decimal width for a validated `(precision, scale)`. Arrow
/// caps `Decimal128` at precision 38 and requires `Decimal256` above that, so
/// both the `decimal(P)` and `decimal(P, S)` paths must switch on `> 38` — a
/// single helper keeps them from diverging.
fn decimal_data_type(precision: u8, scale: i8) -> DataType {
    if precision > 38 {
        DataType::Decimal256(precision, scale)
    } else {
        DataType::Decimal128(precision, scale)
    }
}

/// Parse decimal type with precision and scale
/// Format: "decimal(precision, scale)" or "decimal(precision)"
///
/// Returns `Ok(None)` if the type string is not a decimal type.
/// Returns `Err` if it is a decimal type but has invalid precision/scale.
fn parse_decimal(type_str: &str) -> Result<Option<DataType>> {
    let normalized = type_str.to_ascii_lowercase();
    if !normalized.starts_with("decimal") && !normalized.starts_with("numeric") {
        return Ok(None);
    }

    // Extract parameters from parentheses
    let start = match type_str.find('(') {
        Some(s) => s,
        None => return Ok(None),
    };
    let end = match type_str.find(')') {
        Some(e) => e,
        None => return Ok(None),
    };
    let params = &type_str[start + 1..end];

    let parts: Vec<&str> = params.split(',').map(|s| s.trim()).collect();

    match parts.len() {
        1 => {
            let precision: u8 = parts[0].parse().map_err(|_| {
                DuckLakeError::UnsupportedType(format!(
                    "Invalid decimal precision '{}' in type '{}'",
                    parts[0], type_str
                ))
            })?;
            validate_decimal_precision_scale(precision, 0, type_str)?;
            Ok(Some(decimal_data_type(precision, 0)))
        },
        2 => {
            let precision: u8 = parts[0].parse().map_err(|_| {
                DuckLakeError::UnsupportedType(format!(
                    "Invalid decimal precision '{}' in type '{}'",
                    parts[0], type_str
                ))
            })?;
            let scale: i8 = parts[1].parse().map_err(|_| {
                DuckLakeError::UnsupportedType(format!(
                    "Invalid decimal scale '{}' in type '{}'",
                    parts[1], type_str
                ))
            })?;
            validate_decimal_precision_scale(precision, scale, type_str)?;
            Ok(Some(decimal_data_type(precision, scale)))
        },
        n => Err(DuckLakeError::UnsupportedType(format!(
            "Invalid decimal type: expected at most 2 parameters (precision, scale), got {} in type '{}'",
            n, type_str
        ))),
    }
}

/// Parse list/array type syntax and return `DataType::List` if matched.
///
/// Supported formats:
/// - `list<element_type>` / `array<element_type>` (DuckDB style)
/// - `element_type[]` (Postgres style, e.g. `varchar[]`, `float[]`)
///
fn parse_list_type(type_str: &str, depth: usize) -> Result<Option<DataType>> {
    let normalized = type_str.to_ascii_lowercase();
    let inner = if normalized.starts_with("list<") || normalized.starts_with("array<") {
        // list<type> or array<type>
        let start = type_str.find('<').unwrap();
        if !type_str.ends_with('>') {
            return Ok(None);
        }
        &type_str[start + 1..type_str.len() - 1]
    } else if let Some(stripped) = type_str.strip_suffix("[]") {
        // type[]
        stripped
    } else {
        return Ok(None);
    };

    let inner = inner.trim();
    if inner.is_empty() {
        return Err(DuckLakeError::UnsupportedType(format!(
            "List type '{}' has empty element type",
            type_str
        )));
    }

    let element_type = ducklake_to_arrow_type_inner(inner, depth + 1)?;
    Ok(Some(DataType::List(Arc::new(Field::new(
        "item",
        element_type,
        true,
    )))))
}

fn parse_struct_type(type_str: &str, depth: usize) -> Result<Option<DataType>> {
    if !type_str.to_ascii_lowercase().starts_with("struct<") {
        return Ok(None);
    }
    if !type_str.ends_with('>') {
        return Err(DuckLakeError::UnsupportedType(format!(
            "Struct type '{type_str}' is missing its closing '>'"
        )));
    }
    let inner = &type_str[7..type_str.len() - 1];
    if inner.is_empty() {
        return Ok(Some(DataType::Struct(Vec::<Arc<Field>>::new().into())));
    }
    let fields = split_top_level(inner, ',')?
        .into_iter()
        .map(|field| {
            let Some(separator) = top_level_separator(field, ':') else {
                return Err(DuckLakeError::UnsupportedType(format!(
                    "Struct field '{field}' must use name:type syntax"
                )));
            };
            let name = field[..separator].trim();
            let type_name = field[separator + 1..].trim();
            if name.is_empty() || type_name.is_empty() {
                return Err(DuckLakeError::UnsupportedType(format!(
                    "Struct field '{field}' must have a name and type"
                )));
            }
            Ok(Arc::new(Field::new(
                name,
                ducklake_to_arrow_type_inner(type_name, depth + 1)?,
                true,
            )))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Some(DataType::Struct(fields.into())))
}

fn parse_map_type(type_str: &str, depth: usize) -> Result<Option<DataType>> {
    if !type_str.to_ascii_lowercase().starts_with("map<") {
        return Ok(None);
    }
    if !type_str.ends_with('>') {
        return Err(DuckLakeError::UnsupportedType(format!(
            "Map type '{type_str}' is missing its closing '>'"
        )));
    }
    let parts = split_top_level(&type_str[4..type_str.len() - 1], ',')?;
    let [key_type, value_type] = parts.as_slice() else {
        return Err(DuckLakeError::UnsupportedType(format!(
            "Map type '{type_str}' must have key and value types"
        )));
    };
    let entries = DataType::Struct(
        vec![
            Arc::new(Field::new(
                "key",
                ducklake_to_arrow_type_inner(key_type, depth + 1)?,
                false,
            )),
            Arc::new(Field::new(
                "value",
                ducklake_to_arrow_type_inner(value_type, depth + 1)?,
                true,
            )),
        ]
        .into(),
    );
    Ok(Some(DataType::Map(
        Arc::new(Field::new("entries", entries, false)),
        false,
    )))
}

fn split_top_level(value: &str, separator: char) -> Result<Vec<&str>> {
    let mut parts = Vec::new();
    let mut start = 0;
    let mut angle_depth = 0_i32;
    let mut paren_depth = 0_i32;
    for (index, character) in value.char_indices() {
        match character {
            '<' => angle_depth += 1,
            '>' => angle_depth -= 1,
            '(' => paren_depth += 1,
            ')' => paren_depth -= 1,
            _ => {},
        }
        if angle_depth < 0 || paren_depth < 0 {
            return Err(DuckLakeError::UnsupportedType(format!(
                "Unbalanced nested type '{value}'"
            )));
        }
        if character == separator && angle_depth == 0 && paren_depth == 0 {
            parts.push(value[start..index].trim());
            start = index + character.len_utf8();
        }
    }
    if angle_depth != 0 || paren_depth != 0 {
        return Err(DuckLakeError::UnsupportedType(format!(
            "Unbalanced nested type '{value}'"
        )));
    }
    parts.push(value[start..].trim());
    Ok(parts)
}

fn top_level_separator(value: &str, separator: char) -> Option<usize> {
    let mut angle_depth = 0_i32;
    let mut paren_depth = 0_i32;
    for (index, character) in value.char_indices() {
        match character {
            '<' => angle_depth += 1,
            '>' => angle_depth -= 1,
            '(' => paren_depth += 1,
            ')' => paren_depth -= 1,
            _ => {},
        }
        if character == separator && angle_depth == 0 && paren_depth == 0 {
            return Some(index);
        }
    }
    None
}

/// Normalize a DuckLake type string to its canonical form.
///
/// Converts aliases and case variants to the canonical DuckLake type string.
/// For example: "int" -> "int32", "INTEGER" -> "int32", "text" -> "varchar".
///
/// Returns the canonical type string, or an error if the type is unrecognized.
pub fn normalize_ducklake_type(ducklake_type: &str) -> Result<String> {
    let arrow_type = ducklake_to_arrow_type(ducklake_type)?;
    arrow_to_ducklake_type(&arrow_type)
}

/// The allowlist of **lossless** type widenings that `promote_column_type` may
/// apply during schema evolution. Both type strings are normalized first.
///
/// This is a deliberately small, owned set (design §6) — the published DuckLake
/// stable-spec widenings, every entry provably lossless:
/// - Signed integer widening: int8 -> int16 -> int32 -> int64
/// - Unsigned integer widening: uint8 -> uint16 -> uint32 -> uint64
/// - Float widening: float32 -> float64
///
/// Deliberately **excluded** (each would need its own justified lossless entry +
/// cast-on-read coverage): integer -> float (`int64`/`uint64 -> float64` loses
/// precision past 2^53), `timestamp -> timestamptz`, and decimal precision/scale
/// widening. The read path is more permissive (it casts whatever a file holds);
/// this set only bounds what a *promote* may write. Same-type returns `true`
/// (a no-op); callers wanting strict change-detection use `types_equal_canonical`.
pub fn is_promotable(from: &str, to: &str) -> bool {
    let from_arrow = match ducklake_to_arrow_type(from) {
        Ok(t) => t,
        Err(_) => return false,
    };
    let to_arrow = match ducklake_to_arrow_type(to) {
        Ok(t) => t,
        Err(_) => return false,
    };

    is_arrow_promotable(&from_arrow, &to_arrow)
}

/// Check if one Arrow DataType can be safely promoted to another.
fn is_arrow_promotable(from: &DataType, to: &DataType) -> bool {
    use DataType::*;

    // Same type is trivially promotable
    if from == to {
        return true;
    }

    fn signed_int_rank(dt: &DataType) -> Option<u8> {
        match dt {
            Int8 => Some(0),
            Int16 => Some(1),
            Int32 => Some(2),
            Int64 => Some(3),
            _ => None,
        }
    }

    fn unsigned_int_rank(dt: &DataType) -> Option<u8> {
        match dt {
            UInt8 => Some(0),
            UInt16 => Some(1),
            UInt32 => Some(2),
            UInt64 => Some(3),
            _ => None,
        }
    }

    // Signed integer widening
    if let (Some(from_rank), Some(to_rank)) = (signed_int_rank(from), signed_int_rank(to)) {
        return from_rank < to_rank;
    }

    // Unsigned integer widening
    if let (Some(from_rank), Some(to_rank)) = (unsigned_int_rank(from), unsigned_int_rank(to)) {
        return from_rank < to_rank;
    }

    // Float widening
    if matches!(from, Float32) && matches!(to, Float64) {
        return true;
    }

    // DEFAULT allowlist ends here. Everything below is DELIBERATELY excluded
    // (design §6, review #4): the default promote set is the small, provably
    // LOSSLESS set from the published DuckLake stable-spec widenings — signed
    // integer widening, unsigned integer widening, and Float32 -> Float64 — and
    // nothing else. Notably:
    //   - `Int64`/`UInt64 -> Float64` is NOT lossless (precision loss past 2^53),
    //   - integer -> float in general, `Timestamp -> TimestampTZ` (a semantic
    //     reinterpretation, not a pure widen), and `Decimal` precision/scale
    //     widening each need their own individually justified lossless entry +
    //     cast-on-read coverage before being added here.
    // We own this set rather than tracking upstream's `TypePromotionIsAllowed`
    // (which delegates to a broad, DuckDB-version-dependent rule). The READ path
    // stays permissive (it casts whatever a file physically holds); this set only
    // governs what `promote_column_type` is allowed to WRITE.
    false
}

/// Check if two DuckLake type strings are compatible for schema evolution.
///
/// Types are compatible if they normalize to the same canonical type,
/// or if the existing type can be safely promoted to the new type.
pub fn types_compatible(existing_type: &str, new_type: &str) -> bool {
    // First try normalization: if both normalize to the same canonical form, they match
    let existing_normalized = match normalize_ducklake_type(existing_type) {
        Ok(t) => t,
        Err(_) => return false,
    };
    let new_normalized = match normalize_ducklake_type(new_type) {
        Ok(t) => t,
        Err(_) => return false,
    };

    if existing_normalized == new_normalized {
        return true;
    }

    // Then check if promotion is allowed
    is_promotable(existing_type, new_type)
}

/// Two DuckLake type strings denote the *same* type modulo aliases
/// (`int64` ≡ `bigint`, `int` ≡ `int32`), with **no** promotion/widening.
///
/// This is the comparison used by the data-write policy (§5 of the column
/// versioning design) and the commit-time type guard (§4.6): a data write
/// (`Replace`/`Append`) may not change a column's type, but an alias-only
/// restatement is a no-op. Unlike [`types_compatible`], a widening such as
/// `int32 -> int64` is **not** considered equal — that is schema evolution and
/// must go through an explicit promotion, never a data write.
pub fn types_equal_canonical(a: &str, b: &str) -> bool {
    match (normalize_ducklake_type(a), normalize_ducklake_type(b)) {
        (Ok(na), Ok(nb)) => na == nb,
        _ => false,
    }
}

/// Build an Arrow schema from a list of DuckLake table columns.
///
/// Struct fields are nullable on reads because a field added after older files were written is
/// absent from those files even when its current catalog row is non-nullable.
pub fn build_arrow_schema(columns: &[DuckLakeTableColumn]) -> Result<Schema> {
    let fields: Result<Vec<Field>> = columns
        .iter()
        .map(|col| {
            let data_type = read_compatible_data_type(&col.data_type()?, true);
            Ok(Field::new(&col.column_name, data_type, col.is_nullable))
        })
        .collect();

    Ok(Schema::new(fields?))
}

/// Prefix of the synthetic name a read schema gives a column the file does not
/// carry, so the scan null-fills it instead of matching a same-named column that
/// happens to be present.
pub(crate) const ABSENT_FIELD_PREFIX: &str = "__ducklake_absent_field_";

fn read_compatible_data_type(data_type: &DataType, nullable_struct_fields: bool) -> DataType {
    let rewrite_field = |field: &Arc<Field>, nullable: bool| {
        Arc::new(
            field
                .as_ref()
                .clone()
                .with_data_type(read_compatible_data_type(field.data_type(), true))
                .with_nullable(field.is_nullable() || nullable),
        )
    };
    match data_type {
        DataType::List(field) => DataType::List(rewrite_field(field, false)),
        DataType::LargeList(field) => DataType::LargeList(rewrite_field(field, false)),
        DataType::FixedSizeList(field, size) => {
            DataType::FixedSizeList(rewrite_field(field, false), *size)
        },
        DataType::Struct(fields) => DataType::Struct(
            fields
                .iter()
                .map(|field| rewrite_field(field, nullable_struct_fields))
                .collect::<Vec<_>>()
                .into(),
        ),
        DataType::Map(entries, sorted) => DataType::Map(
            Arc::new(
                entries
                    .as_ref()
                    .clone()
                    .with_data_type(read_compatible_data_type(entries.data_type(), false)),
            ),
            *sorted,
        ),
        _ => data_type.clone(),
    }
}

#[derive(Debug)]
struct TableFieldTree<'a> {
    fields: &'a [DuckLakeTableField],
    children: HashMap<Option<i64>, Vec<usize>>,
}

impl<'a> TableFieldTree<'a> {
    fn new(fields: &'a [DuckLakeTableField]) -> Result<Self> {
        let mut by_id = HashMap::with_capacity(fields.len());
        let mut children: HashMap<Option<i64>, Vec<usize>> = HashMap::new();
        for (index, field) in fields.iter().enumerate() {
            if by_id.insert(field.column_id, index).is_some() {
                return Err(DuckLakeError::InvalidConfig(format!(
                    "duplicate DuckLake column_id {}",
                    field.column_id
                )));
            }
            children.entry(field.parent_column).or_default().push(index);
        }
        for field in fields {
            if let Some(parent) = field.parent_column
                && !by_id.contains_key(&parent)
            {
                return Err(DuckLakeError::InvalidConfig(format!(
                    "DuckLake column {} references missing parent_column {}",
                    field.column_id, parent
                )));
            }
            let mut ancestors = HashSet::new();
            let mut ancestor = Some(field.column_id);
            while let Some(column_id) = ancestor {
                if !ancestors.insert(column_id) {
                    return Err(DuckLakeError::InvalidConfig(format!(
                        "cycle in DuckLake column tree at column_id {column_id}"
                    )));
                }
                ancestor = fields[by_id[&column_id]].parent_column;
            }
        }
        Ok(Self {
            fields,
            children,
        })
    }

    fn roots(&self) -> impl Iterator<Item = usize> + '_ {
        self.children.get(&None).into_iter().flatten().copied()
    }

    fn children(&self, column_id: i64) -> impl Iterator<Item = usize> + '_ {
        self.children
            .get(&Some(column_id))
            .into_iter()
            .flatten()
            .copied()
    }

    fn arrow_field(&self, index: usize, visiting: &mut HashSet<i64>) -> Result<Field> {
        let field = &self.fields[index];
        if !visiting.insert(field.column_id) {
            return Err(DuckLakeError::InvalidConfig(format!(
                "cycle in DuckLake column tree at column_id {}",
                field.column_id
            )));
        }
        let child_indices: Vec<_> = self.children(field.column_id).collect();
        let normalized = field.column_type.trim().to_ascii_lowercase();
        let data_type = match normalized.as_str() {
            "struct" => {
                let children = child_indices
                    .into_iter()
                    .map(|child| self.arrow_field(child, visiting).map(Arc::new))
                    .collect::<Result<Vec<_>>>()?;
                DataType::Struct(children.into())
            },
            "list" | "array" => {
                if child_indices.len() != 1 {
                    return Err(DuckLakeError::InvalidConfig(format!(
                        "DuckLake list column {} has {} children; expected 1",
                        field.column_id,
                        child_indices.len()
                    )));
                }
                let mut child = self.arrow_field(child_indices[0], visiting)?;
                child = child.with_name("item");
                DataType::List(Arc::new(child))
            },
            "map" => {
                if child_indices.len() != 2 {
                    return Err(DuckLakeError::InvalidConfig(format!(
                        "DuckLake map column {} has {} children; expected key and value",
                        field.column_id,
                        child_indices.len()
                    )));
                }
                let key = self
                    .arrow_field(child_indices[0], visiting)?
                    .with_nullable(false);
                let value = self.arrow_field(child_indices[1], visiting)?;
                if key.name() != "key" || value.name() != "value" {
                    return Err(DuckLakeError::InvalidConfig(format!(
                        "DuckLake map column {} children must be named key and value",
                        field.column_id
                    )));
                }
                DataType::Map(
                    Arc::new(Field::new(
                        "entries",
                        DataType::Struct(vec![Arc::new(key), Arc::new(value)].into()),
                        false,
                    )),
                    false,
                )
            },
            _ => {
                if !child_indices.is_empty() {
                    return Err(DuckLakeError::InvalidConfig(format!(
                        "DuckLake scalar column {} has nested children",
                        field.column_id
                    )));
                }
                ducklake_to_arrow_type(&field.column_type)?
            },
        };
        visiting.remove(&field.column_id);
        Ok(Field::new(
            &field.column_name,
            read_compatible_data_type(&data_type, true),
            field.is_nullable,
        ))
    }
}

/// Build an Arrow schema from the complete `ducklake_column` field tree.
pub fn build_arrow_schema_from_fields(fields: &[DuckLakeTableField]) -> Result<Schema> {
    build_arrow_schema(&top_level_columns(fields)?)
}

/// Return the top-level column records in catalog order.
pub fn top_level_columns(fields: &[DuckLakeTableField]) -> Result<Vec<DuckLakeTableColumn>> {
    crate::metadata_provider::reconstruct_columns(
        fields
            .iter()
            .map(|field| (field.column(), field.parent_column))
            .collect(),
    )
}

#[derive(Debug)]
struct NameMappingTree<'a> {
    entries: &'a [DuckLakeNameMappingEntry],
    children: HashMap<Option<i64>, Vec<usize>>,
}

impl<'a> NameMappingTree<'a> {
    fn new(mapping: &'a DuckLakeNameMapping) -> Result<Self> {
        if mapping.mapping_type != "map_by_name" {
            return Err(DuckLakeError::Unsupported(format!(
                "DuckLake column mapping type '{}'",
                mapping.mapping_type
            )));
        }
        let mut by_id = HashMap::with_capacity(mapping.entries.len());
        let mut children: HashMap<Option<i64>, Vec<usize>> = HashMap::new();
        for (index, entry) in mapping.entries.iter().enumerate() {
            if by_id.contains_key(&entry.column_id) {
                continue;
            }
            by_id.insert(entry.column_id, index);
            children.entry(entry.parent_column).or_default().push(index);
        }
        for index in by_id.values() {
            let entry = &mapping.entries[*index];
            if let Some(parent) = entry.parent_column
                && !by_id.contains_key(&parent)
            {
                return Err(DuckLakeError::InvalidConfig(format!(
                    "name-mapping column {} references missing parent_column {}",
                    entry.column_id, parent
                )));
            }
            if entry.is_partition && entry.parent_column.is_some() {
                return Err(DuckLakeError::InvalidConfig(format!(
                    "nested name-mapping column {} cannot be synthesized from a Hive path",
                    entry.column_id
                )));
            }
            let mut ancestors = HashSet::new();
            let mut ancestor = Some(entry.column_id);
            while let Some(column_id) = ancestor {
                if !ancestors.insert(column_id) {
                    return Err(DuckLakeError::InvalidConfig(format!(
                        "cycle in DuckLake name mapping at column_id {column_id}"
                    )));
                }
                ancestor = mapping.entries[by_id[&column_id]].parent_column;
            }
        }
        Ok(Self {
            entries: &mapping.entries,
            children,
        })
    }

    fn entry_for_target(
        &self,
        parent: Option<i64>,
        target_field_id: i64,
    ) -> Option<&DuckLakeNameMappingEntry> {
        self.children.get(&parent)?.iter().find_map(|index| {
            let entry = &self.entries[*index];
            (entry.target_field_id == target_field_id).then_some(entry)
        })
    }
}

fn physical_mapped_field(
    table: &TableFieldTree<'_>,
    mapping: &NameMappingTree<'_>,
    table_index: usize,
    mapping_entry: &DuckLakeNameMappingEntry,
) -> Result<Field> {
    let logical = table.arrow_field(table_index, &mut HashSet::new())?;
    let table_field = &table.fields[table_index];
    let data_type = match logical.data_type() {
        DataType::Struct(_) => {
            let children = table
                .children(table_field.column_id)
                .map(|child_index| {
                    let child = &table.fields[child_index];
                    match mapping.entry_for_target(Some(mapping_entry.column_id), child.column_id) {
                        Some(entry) => physical_mapped_field(table, mapping, child_index, entry),
                        None => {
                            let field = table.arrow_field(child_index, &mut HashSet::new())?;
                            Ok(Field::new(
                                format!("__ducklake_absent_field_{}", child.column_id),
                                field.data_type().clone(),
                                true,
                            ))
                        },
                    }
                })
                .collect::<Result<Vec<_>>>()?;
            DataType::Struct(children.into())
        },
        DataType::List(_) | DataType::LargeList(_) | DataType::FixedSizeList(_, _) => {
            let child_index = table
                .children(table_field.column_id)
                .next()
                .ok_or_else(|| {
                    DuckLakeError::InvalidConfig(format!(
                        "DuckLake list column {} has no element",
                        table_field.column_id
                    ))
                })?;
            let child = &table.fields[child_index];
            let mut element =
                match mapping.entry_for_target(Some(mapping_entry.column_id), child.column_id) {
                    Some(child_mapping) => {
                        physical_mapped_field(table, mapping, child_index, child_mapping)?
                    },
                    None => table
                        .arrow_field(child_index, &mut HashSet::new())?
                        .with_name(format!("__ducklake_absent_field_{}", child.column_id))
                        .with_nullable(true),
                };
            if matches!(element.name().as_str(), "array" | "element") {
                element = element.with_name("list");
            }
            DataType::List(Arc::new(element))
        },
        DataType::Map(_, sorted) => {
            let children = table
                .children(table_field.column_id)
                .enumerate()
                .map(|(index, child_index)| {
                    let child = &table.fields[child_index];
                    let field = match mapping
                        .entry_for_target(Some(mapping_entry.column_id), child.column_id)
                    {
                        Some(child_mapping) => {
                            physical_mapped_field(table, mapping, child_index, child_mapping)?
                        },
                        None => table
                            .arrow_field(child_index, &mut HashSet::new())?
                            .with_name(format!("__ducklake_absent_field_{}", child.column_id))
                            .with_nullable(true),
                    };
                    Ok(if index == 0 {
                        field.with_nullable(false)
                    } else {
                        field
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            DataType::Map(
                Arc::new(Field::new(
                    "entries",
                    DataType::Struct(children.into()),
                    false,
                )),
                *sorted,
            )
        },
        data_type => data_type.clone(),
    };
    Ok(Field::new(
        &mapping_entry.source_name,
        data_type,
        logical.is_nullable() || mapping_entry.parent_column.is_some(),
    ))
}

#[derive(Debug, PartialEq)]
pub struct HivePartitionField {
    pub logical_name: String,
    pub source_name: String,
    pub data_type: DataType,
}

#[derive(Debug, PartialEq)]
pub struct NameMappingSchema {
    pub schema: Schema,
    pub names: HashMap<String, String>,
    pub partitions: Vec<HivePartitionField>,
}

/// Build the per-file physical schema for a `map_by_name` mapping.
///
/// The returned partition fields identify Hive-derived values keyed by the
/// current logical column name. The caller extracts and parses the path values
/// because the resolved object-store path is available at scan planning time.
pub fn build_read_schema_with_name_mapping(
    fields: &[DuckLakeTableField],
    mapping: &DuckLakeNameMapping,
) -> Result<NameMappingSchema> {
    let table = TableFieldTree::new(fields)?;
    let mapping_tree = NameMappingTree::new(mapping)?;
    let mut name_mapping = HashMap::new();
    let mut partitions = Vec::new();
    let read_fields = table
        .roots()
        .map(|table_index| {
            let field = &table.fields[table_index];
            let logical = table.arrow_field(table_index, &mut HashSet::new())?;
            let Some(entry) = mapping_tree.entry_for_target(None, field.column_id) else {
                let read_name = format!("__ducklake_absent_field_{}", field.column_id);
                name_mapping.insert(read_name.clone(), field.column_name.clone());
                return Ok(Field::new(read_name, logical.data_type().clone(), true));
            };
            if entry.is_partition {
                let read_name = format!("__ducklake_partition_field_{}", field.column_id);
                partitions.push(HivePartitionField {
                    logical_name: field.column_name.clone(),
                    source_name: entry.source_name.clone(),
                    data_type: logical.data_type().clone(),
                });
                return Ok(Field::new(read_name, logical.data_type().clone(), true));
            }
            let physical = physical_mapped_field(&table, &mapping_tree, table_index, entry)?;
            if physical.name() != &field.column_name {
                name_mapping.insert(physical.name().clone(), field.column_name.clone());
            }
            Ok(physical)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(NameMappingSchema {
        schema: Schema::new(read_fields),
        names: name_mapping,
        partitions,
    })
}

/// Field-id mapping variant that takes the already-built logical schema. This
/// supports recursive types without re-parsing their catalog type rows.
pub fn build_read_schema_with_field_id_mapping_from_schema(
    current_columns: &[DuckLakeTableColumn],
    current_schema: &Schema,
    parquet_field_ids: &HashMap<i32, String>,
    file_schema: Option<&Schema>,
) -> Result<(Schema, HashMap<String, String>)> {
    if current_columns.len() != current_schema.fields().len() {
        return Err(DuckLakeError::Internal(
            "DuckLake columns and logical schema have different lengths".to_string(),
        ));
    }
    let columns = current_columns
        .iter()
        .zip(current_schema.fields())
        .map(|(column, logical)| {
            let mut column = column.clone();
            column.data_type = Some(logical.data_type().clone());
            column
        })
        .collect::<Vec<_>>();
    build_read_schema_with_field_id_mapping(&columns, parquet_field_ids, file_schema)
}

/// Extract field_id to column_name mapping from Parquet metadata.
/// DuckLake column_id == Parquet field_id, enabling column matching after renames.
pub fn extract_parquet_field_ids(metadata: &ParquetMetaData) -> HashMap<i32, String> {
    let schema_descr = metadata.file_metadata().schema_descr();
    fn collect(type_: &parquet::schema::types::Type, entries: &mut Vec<(i32, String)>) {
        let basic_info = type_.get_basic_info();
        if basic_info.has_id() {
            entries.push((basic_info.id(), type_.name().to_string()));
        }
        if type_.is_group() {
            for child in type_.get_fields() {
                collect(child, entries);
            }
        }
    }

    let mut entries = Vec::new();
    for field in schema_descr.root_schema().get_fields() {
        collect(field, &mut entries);
    }
    field_ids_dropping_duplicates(entries.into_iter())
}

/// Collect a `field_id -> name` map, dropping any `field_id` shared by more than
/// one field. DuckLake assigns exactly one field_id per catalog column node, so
/// a collision is malformed/adversarial parquet; binding the catalog
/// column with that id to either physical column risks reading the wrong data.
/// Dropping the id makes the reader null-fill that column (via its "field_id
/// absent" path) instead of silently substituting the wrong one.
fn field_ids_dropping_duplicates(
    entries: impl Iterator<Item = (i32, String)>,
) -> HashMap<i32, String> {
    let mut map: HashMap<i32, String> = HashMap::new();
    let mut duplicates: HashSet<i32> = HashSet::new();
    for (id, name) in entries {
        if map.insert(id, name).is_some() {
            duplicates.insert(id);
        }
    }
    for id in duplicates {
        map.remove(&id);
        tracing::warn!(
            field_id = id,
            "parquet file has multiple top-level columns sharing this field_id; \
             ignoring it — the affected column will read as NULL"
        );
    }
    map
}

fn arrow_field_id_names(schema: &Schema) -> HashMap<i32, String> {
    fn collect(field: &Field, names: &mut HashMap<i32, String>) {
        if let Some(id) = field
            .metadata()
            .get(PARQUET_FIELD_ID_META_KEY)
            .and_then(|value| value.parse().ok())
        {
            names.insert(id, field.name().clone());
        }
        match field.data_type() {
            DataType::List(child)
            | DataType::LargeList(child)
            | DataType::FixedSizeList(child, _)
            | DataType::Map(child, _) => collect(child, names),
            DataType::Struct(fields) => {
                for child in fields {
                    collect(child, names);
                }
            },
            _ => {},
        }
    }

    let mut names = HashMap::new();
    for field in schema.fields() {
        collect(field, &mut names);
    }
    names
}

fn data_type_has_field_ids(data_type: &DataType) -> bool {
    let fields: &[Arc<Field>] = match data_type {
        DataType::List(field)
        | DataType::LargeList(field)
        | DataType::FixedSizeList(field, _)
        | DataType::Map(field, _) => std::slice::from_ref(field),
        DataType::Struct(fields) => fields,
        _ => return false,
    };
    fields.iter().any(|field| {
        field.metadata().contains_key(PARQUET_FIELD_ID_META_KEY)
            || data_type_has_field_ids(field.data_type())
    })
}

fn read_data_type_with_field_id_mapping(
    data_type: &DataType,
    nested_column_ids: &[i64],
    parquet_field_ids: &HashMap<i32, String>,
    arrow_field_names: &HashMap<i32, String>,
    match_by_id: bool,
) -> Result<DataType> {
    fn rewrite_field(
        field: &Field,
        ids: &mut std::slice::Iter<'_, i64>,
        parquet_field_ids: &HashMap<i32, String>,
        arrow_field_names: &HashMap<i32, String>,
        match_by_id: bool,
    ) -> Result<Arc<Field>> {
        let column_id = ids.next().ok_or_else(|| {
            DuckLakeError::InvalidConfig(
                "Nested column metadata has fewer field ids than its Arrow type".to_string(),
            )
        })?;
        let field_id = i32::try_from(*column_id).map_err(|_| {
            DuckLakeError::Internal(format!(
                "column_id {column_id} for nested column '{}' exceeds i32 range for Parquet field_id",
                field.name()
            ))
        })?;
        let is_absent = match_by_id && !parquet_field_ids.contains_key(&field_id);
        let name = if let Some(parquet_name) = parquet_field_ids.get(&field_id) {
            arrow_field_names
                .get(&field_id)
                .unwrap_or(parquet_name)
                .clone()
        } else if match_by_id {
            format!("{ABSENT_FIELD_PREFIX}{field_id}")
        } else {
            field.name().clone()
        };
        let data_type = rewrite_type(
            field.data_type(),
            ids,
            parquet_field_ids,
            arrow_field_names,
            match_by_id,
        )?;
        // A nested node's field id is part of the Arrow *type* of its parent
        // (`DataType::List`/`Struct`/`Map` embed whole child `Field`s, metadata
        // included), so it takes part in every array/batch type check. The
        // physical file tags each nested node with its DuckLake field id, so the
        // read schema must declare the same id or batches the parquet reader
        // produces for the file do not match the schema the reader was handed.
        // Only stamp what the file actually carries: a node matched by name
        // (`match_by_id == false`, an external file with no ids) or one absent
        // from the file has no id to declare. Top-level fields are deliberately
        // left bare — their metadata is not part of any type, they are resolved
        // by name, and stamping them would make every read schema differ from
        // the catalog schema.
        let field_metadata_id = (match_by_id && !is_absent).then(|| field_id.to_string());
        let mut read_field = field
            .clone()
            .with_name(name)
            .with_data_type(data_type)
            .with_nullable(field.is_nullable() || is_absent);
        if let Some(id) = field_metadata_id {
            let mut metadata = read_field.metadata().clone();
            metadata.insert(PARQUET_FIELD_ID_META_KEY.to_string(), id);
            read_field = read_field.with_metadata(metadata);
        }
        Ok(Arc::new(read_field))
    }

    fn rewrite_type(
        data_type: &DataType,
        ids: &mut std::slice::Iter<'_, i64>,
        parquet_field_ids: &HashMap<i32, String>,
        arrow_field_names: &HashMap<i32, String>,
        match_by_id: bool,
    ) -> Result<DataType> {
        match data_type {
            DataType::List(field) => Ok(DataType::List(rewrite_field(
                field,
                ids,
                parquet_field_ids,
                arrow_field_names,
                match_by_id,
            )?)),
            DataType::LargeList(field) => Ok(DataType::LargeList(rewrite_field(
                field,
                ids,
                parquet_field_ids,
                arrow_field_names,
                match_by_id,
            )?)),
            DataType::FixedSizeList(field, size) => Ok(DataType::FixedSizeList(
                rewrite_field(
                    field,
                    ids,
                    parquet_field_ids,
                    arrow_field_names,
                    match_by_id,
                )?,
                *size,
            )),
            DataType::Struct(fields) => Ok(DataType::Struct(
                fields
                    .iter()
                    .map(|field| {
                        rewrite_field(
                            field,
                            ids,
                            parquet_field_ids,
                            arrow_field_names,
                            match_by_id,
                        )
                    })
                    .collect::<Result<Vec<_>>>()?
                    .into(),
            )),
            DataType::Map(entries, sorted) => Ok(DataType::Map(
                Arc::new(entries.as_ref().clone().with_data_type(rewrite_type(
                    entries.data_type(),
                    ids,
                    parquet_field_ids,
                    arrow_field_names,
                    match_by_id,
                )?)),
                *sorted,
            )),
            _ => Ok(data_type.clone()),
        }
    }

    let mut ids = nested_column_ids.iter();
    let result = rewrite_type(
        data_type,
        &mut ids,
        parquet_field_ids,
        arrow_field_names,
        match_by_id,
    )?;
    if ids.next().is_some() {
        return Err(DuckLakeError::InvalidConfig(
            "Nested column metadata has more field ids than its Arrow type".to_string(),
        ));
    }
    Ok(result)
}

/// Build a schema for reading Parquet files across schema evolution.
/// Returns (read_schema, name_mapping): read_schema uses each column's physical
/// name in the file, and name_mapping maps that physical name -> current name for
/// renamed columns. A current column whose field_id is absent from a file that
/// otherwise carries field_ids is read as an all-NULL column (the file predates
/// the column, or it was dropped then re-added under the same name).
///
/// Nested nullability is relaxed exactly as [`build_arrow_schema`] relaxes it: a
/// field added inside a struct, or renamed there, is recorded non-nullable in the
/// catalog while the physical node stays optional, and a file written before the
/// change does not carry it at all. Map keys stay non-nullable — a map's entries
/// are structurally non-null, and relaxing them makes a `MAP` column's arrays
/// disagree with the schema describing them.
pub fn build_read_schema_with_field_id_mapping(
    current_columns: &[DuckLakeTableColumn],
    parquet_field_ids: &HashMap<i32, String>,
    file_schema: Option<&Schema>,
) -> Result<(Schema, HashMap<String, String>)> {
    let mut name_mapping: HashMap<String, String> = HashMap::new();
    let arrow_field_names = file_schema.map(arrow_field_id_names).unwrap_or_default();

    let fields: Result<Vec<Field>> = current_columns
        .iter()
        .map(|col| {
            // Relax nested nullability BEFORE the field ids are stamped on:
            // the rewrite neither adds nor removes a node, so the
            // `nested_column_ids` iterator stays aligned with the type.
            let mut data_type = read_compatible_data_type(&col.data_type()?, true);
            let field_id = i32::try_from(col.column_id).map_err(|_| {
                DuckLakeError::Internal(format!(
                    "column_id {} for column '{}' exceeds i32 range for Parquet field_id",
                    col.column_id, col.column_name
                ))
            })?;

            // Resolve the physical name this column has in THIS file:
            //  - field_id present: that physical name (rename if it differs).
            //  - file has no field_ids: external/legacy parquet, match by name.
            //  - file has field_ids but not this one's: the column is absent from
            //    this file (added later, or DROPped + re-ADDed under the same name
            //    with a fresh field_id) and must read as NULL. Matching by name
            //    would alias a different same-named column (e.g. the still-present
            //    dropped column) and leak stale data, so use a name guaranteed
            //    absent so the scan null-fills it, then rename it back.
            let (read_name, needs_rename, is_absent) =
                if let Some(parquet_name) = parquet_field_ids.get(&field_id) {
                    if parquet_name != &col.column_name {
                        (parquet_name.clone(), true, false) // Column was renamed
                    } else {
                        (col.column_name.clone(), false, false)
                    }
                } else if parquet_field_ids.is_empty() {
                    (col.column_name.clone(), false, false) // external/legacy file
                } else {
                    (format!("{ABSENT_FIELD_PREFIX}{field_id}"), true, true)
                };

            if !is_absent && !col.nested_column_ids.is_empty() {
                let match_nested_by_id = file_schema
                    .and_then(|schema| schema.field_with_name(&read_name).ok())
                    .is_some_and(|field| data_type_has_field_ids(field.data_type()));
                data_type = read_data_type_with_field_id_mapping(
                    &data_type,
                    &col.nested_column_ids,
                    parquet_field_ids,
                    &arrow_field_names,
                    match_nested_by_id,
                )?;
            }

            if needs_rename {
                name_mapping.insert(read_name.clone(), col.column_name.clone());
            }

            // Without an initial default, an absent column is materialised as a
            // null array, so its read field must be nullable. ColumnRenameExec
            // still enforces the catalog nullability on output.
            let mut field = Field::new(
                read_name,
                data_type,
                col.is_nullable || (is_absent && col.initial_default.is_none()),
            );
            if is_absent && let Some(initial_default) = &col.initial_default {
                field = field.with_metadata(HashMap::from([(
                    INITIAL_DEFAULT_METADATA_KEY.to_string(),
                    initial_default.clone(),
                )]));
            }
            Ok(field)
        })
        .collect();

    Ok((Schema::new(fields?), name_mapping))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    fn mapped_fields() -> Vec<DuckLakeTableField> {
        vec![
            DuckLakeTableField {
                column_id: 1,
                column_name: "id".to_string(),
                column_type: "int32".to_string(),
                is_nullable: false,
                parent_column: None,
            },
            DuckLakeTableField {
                column_id: 2,
                column_name: "nested".to_string(),
                column_type: "struct".to_string(),
                is_nullable: true,
                parent_column: None,
            },
            DuckLakeTableField {
                column_id: 3,
                column_name: "a".to_string(),
                column_type: "int32".to_string(),
                is_nullable: true,
                parent_column: Some(2),
            },
            DuckLakeTableField {
                column_id: 4,
                column_name: "b".to_string(),
                column_type: "varchar".to_string(),
                is_nullable: true,
                parent_column: Some(2),
            },
            DuckLakeTableField {
                column_id: 5,
                column_name: "part".to_string(),
                column_type: "int32".to_string(),
                is_nullable: true,
                parent_column: None,
            },
        ]
    }

    fn name_mapping(mapping_type: &str) -> DuckLakeNameMapping {
        DuckLakeNameMapping {
            mapping_id: 7,
            table_id: 11,
            mapping_type: mapping_type.to_string(),
            entries: vec![
                DuckLakeNameMappingEntry {
                    column_id: 10,
                    source_name: "source_id".to_string(),
                    target_field_id: 1,
                    parent_column: None,
                    is_partition: false,
                },
                DuckLakeNameMappingEntry {
                    column_id: 11,
                    source_name: "source_nested".to_string(),
                    target_field_id: 2,
                    parent_column: None,
                    is_partition: false,
                },
                DuckLakeNameMappingEntry {
                    column_id: 12,
                    source_name: "source_b".to_string(),
                    target_field_id: 4,
                    parent_column: Some(11),
                    is_partition: false,
                },
                DuckLakeNameMappingEntry {
                    column_id: 13,
                    source_name: "source_a".to_string(),
                    target_field_id: 3,
                    parent_column: Some(11),
                    is_partition: false,
                },
                DuckLakeNameMappingEntry {
                    column_id: 14,
                    source_name: "part".to_string(),
                    target_field_id: 5,
                    parent_column: None,
                    is_partition: true,
                },
            ],
        }
    }

    #[test]
    fn builds_recursive_name_mapping_schema() {
        let fields = mapped_fields();
        let logical = build_arrow_schema_from_fields(&fields).unwrap();
        assert_eq!(logical.fields().len(), 3);
        assert_eq!(logical.field(0).name(), "id");
        let DataType::Struct(children) = logical.field(1).data_type() else {
            panic!("nested must be a struct");
        };
        assert_eq!(children[0].name(), "a");
        assert_eq!(children[1].name(), "b");

        let mapped =
            build_read_schema_with_name_mapping(&fields, &name_mapping("map_by_name")).unwrap();
        assert_eq!(mapped.schema.fields().len(), 3);
        assert_eq!(mapped.schema.field(0).name(), "source_id");
        assert_eq!(
            mapped.names.get("source_id").map(String::as_str),
            Some("id")
        );
        let DataType::Struct(children) = mapped.schema.field(1).data_type() else {
            panic!("mapped nested must be a struct");
        };
        assert_eq!(children[0].name(), "source_a");
        assert_eq!(children[1].name(), "source_b");
        assert_eq!(
            mapped.partitions,
            vec![HivePartitionField {
                logical_name: "part".to_string(),
                source_name: "part".to_string(),
                data_type: DataType::Int32,
            }]
        );
    }

    #[test]
    fn rejects_unknown_name_mapping_type() {
        let error =
            build_read_schema_with_name_mapping(&mapped_fields(), &name_mapping("map_by_position"))
                .unwrap_err();
        assert!(error.to_string().contains("map_by_position"));
    }

    #[test]
    fn rejects_missing_name_mapping_parent() {
        let mut mapping = name_mapping("map_by_name");
        mapping.entries[2].parent_column = Some(999);
        let error = build_read_schema_with_name_mapping(&mapped_fields(), &mapping).unwrap_err();
        assert!(error.to_string().contains("missing parent_column 999"));
    }

    #[rstest]
    fn name_mapping_uses_first_duplicate_entry() {
        let mut mapping = name_mapping("map_by_name");
        mapping.entries.push(DuckLakeNameMappingEntry {
            column_id: 10,
            source_name: "ignored_column_id_duplicate".to_string(),
            target_field_id: 5,
            parent_column: Some(999),
            is_partition: false,
        });
        mapping.entries.push(DuckLakeNameMappingEntry {
            column_id: 15,
            source_name: "ignored_target_duplicate".to_string(),
            target_field_id: 1,
            parent_column: None,
            is_partition: false,
        });

        let mapped = build_read_schema_with_name_mapping(&mapped_fields(), &mapping).unwrap();

        assert_eq!(mapped.schema.field(0).name(), "source_id");
        assert_eq!(mapped.partitions.len(), 1);
    }

    #[rstest]
    fn mapped_nested_fields_use_read_compatible_nullability() {
        let mut fields = mapped_fields();
        fields[2].is_nullable = false;

        let mapped =
            build_read_schema_with_name_mapping(&fields, &name_mapping("map_by_name")).unwrap();
        let DataType::Struct(children) = mapped.schema.field(1).data_type() else {
            panic!("mapped nested must be a struct");
        };

        assert!(children[0].is_nullable());
    }

    #[rstest]
    fn mapped_list_and_map_allow_omitted_children() {
        let fields = vec![
            DuckLakeTableField {
                column_id: 1,
                column_name: "items".to_string(),
                column_type: "list".to_string(),
                is_nullable: true,
                parent_column: None,
            },
            DuckLakeTableField {
                column_id: 2,
                column_name: "element".to_string(),
                column_type: "int32".to_string(),
                is_nullable: true,
                parent_column: Some(1),
            },
            DuckLakeTableField {
                column_id: 3,
                column_name: "properties".to_string(),
                column_type: "map".to_string(),
                is_nullable: true,
                parent_column: None,
            },
            DuckLakeTableField {
                column_id: 4,
                column_name: "key".to_string(),
                column_type: "varchar".to_string(),
                is_nullable: true,
                parent_column: Some(3),
            },
            DuckLakeTableField {
                column_id: 5,
                column_name: "value".to_string(),
                column_type: "int32".to_string(),
                is_nullable: true,
                parent_column: Some(3),
            },
        ];
        let mapping = DuckLakeNameMapping {
            mapping_id: 1,
            table_id: 1,
            mapping_type: "map_by_name".to_string(),
            entries: vec![
                DuckLakeNameMappingEntry {
                    column_id: 10,
                    source_name: "source_items".to_string(),
                    target_field_id: 1,
                    parent_column: None,
                    is_partition: false,
                },
                DuckLakeNameMappingEntry {
                    column_id: 11,
                    source_name: "source_properties".to_string(),
                    target_field_id: 3,
                    parent_column: None,
                    is_partition: false,
                },
            ],
        };

        let mapped = build_read_schema_with_name_mapping(&fields, &mapping).unwrap();
        let DataType::List(element) = mapped.schema.field(0).data_type() else {
            panic!("items must be a list");
        };
        assert_eq!(element.name(), "__ducklake_absent_field_2");
        assert!(element.is_nullable());
        let DataType::Map(entries, _) = mapped.schema.field(1).data_type() else {
            panic!("properties must be a map");
        };
        let DataType::Struct(children) = entries.data_type() else {
            panic!("map entries must be a struct");
        };
        assert_eq!(children[0].name(), "__ducklake_absent_field_4");
        assert!(!children[0].is_nullable());
        assert_eq!(children[1].name(), "__ducklake_absent_field_5");
        assert!(children[1].is_nullable());
    }

    #[test]
    fn test_build_read_schema_with_renamed_columns() {
        // Simulate: column was originally named "user_id", now renamed to "userId"
        let current_columns = vec![
            DuckLakeTableColumn::new(
                1,
                "userId".to_string(), // Current name (renamed)
                "int32".to_string(),
                true,
            ),
            DuckLakeTableColumn::new(
                2,
                "name".to_string(), // Not renamed
                "varchar".to_string(),
                true,
            ),
        ];

        // Parquet file has original names
        let mut parquet_field_ids = HashMap::new();
        parquet_field_ids.insert(1, "user_id".to_string()); // Original name
        parquet_field_ids.insert(2, "name".to_string()); // Same name

        let (read_schema, name_mapping) =
            build_read_schema_with_field_id_mapping(&current_columns, &parquet_field_ids, None)
                .unwrap();

        // Read schema should have original Parquet names
        assert_eq!(read_schema.field(0).name(), "user_id");
        assert_eq!(read_schema.field(1).name(), "name");

        // Name mapping should map old name to new name
        assert_eq!(name_mapping.len(), 1);
        assert_eq!(name_mapping.get("user_id"), Some(&"userId".to_string()));
    }

    #[test]
    fn test_build_read_schema_no_rename_needed() {
        let current_columns =
            vec![DuckLakeTableColumn::new(1, "id".to_string(), "int32".to_string(), true)];

        let mut parquet_field_ids = HashMap::new();
        parquet_field_ids.insert(1, "id".to_string()); // Same name

        let (read_schema, name_mapping) =
            build_read_schema_with_field_id_mapping(&current_columns, &parquet_field_ids, None)
                .unwrap();

        assert_eq!(read_schema.field(0).name(), "id");
        assert!(name_mapping.is_empty()); // No rename needed
    }

    #[test]
    fn test_build_read_schema_no_field_ids() {
        // External file without field_ids
        let current_columns =
            vec![DuckLakeTableColumn::new(1, "id".to_string(), "int32".to_string(), true)];

        let parquet_field_ids = HashMap::new(); // No field_ids in Parquet

        let (read_schema, name_mapping) =
            build_read_schema_with_field_id_mapping(&current_columns, &parquet_field_ids, None)
                .unwrap();

        // Falls back to current column name
        assert_eq!(read_schema.field(0).name(), "id");
        assert!(name_mapping.is_empty());
    }

    #[test]
    fn test_build_read_schema_absent_field_id_reads_null() {
        // `tag` (column_id 2) was DROPped then re-ADDed, so it has a fresh
        // field_id that is absent from a pre-drop file. The file still physically
        // carries a column literally named "tag" (the dropped one, different
        // field_id), which must NOT be aliased.
        let current_columns = vec![
            DuckLakeTableColumn::new(1, "id".to_string(), "int32".to_string(), true),
            DuckLakeTableColumn::new(2, "tag".to_string(), "varchar".to_string(), true),
        ];

        let mut parquet_field_ids = HashMap::new();
        parquet_field_ids.insert(1, "id".to_string()); // file has field_ids, but not 2

        let (read_schema, name_mapping) =
            build_read_schema_with_field_id_mapping(&current_columns, &parquet_field_ids, None)
                .unwrap();

        // `id` reads by name; `tag` gets a synthetic absent name (so the scan
        // null-fills it) mapped back to "tag", instead of binding to the
        // physically-present dropped "tag".
        assert_eq!(read_schema.field(0).name(), "id");
        assert_ne!(read_schema.field(1).name(), "tag");
        assert!(
            read_schema
                .field(1)
                .name()
                .starts_with("__ducklake_absent_field_")
        );
        assert!(read_schema.field(1).is_nullable());
        assert_eq!(
            name_mapping.get(read_schema.field(1).name()),
            Some(&"tag".to_string())
        );
    }

    #[test]
    fn test_build_read_schema_keeps_new_nested_field() {
        let current_columns = vec![DuckLakeTableColumn {
            column_id: 1,
            column_name: "payload".to_string(),
            column_type: "struct<a:int32,b:varchar,c:int32>".to_string(),
            is_nullable: false,
            data_type: Some(DataType::Struct(
                vec![
                    Arc::new(Field::new("a", DataType::Int32, false)),
                    Arc::new(Field::new("b", DataType::Utf8View, true)),
                    Arc::new(Field::new("c", DataType::Int32, false)),
                ]
                .into(),
            )),
            nested_column_ids: vec![2, 3, 4],
            initial_default: None,
            default_value: None,
            default_value_type: None,
            default_value_dialect: None,
        }];
        let parquet_field_ids = HashMap::from([
            (1, "payload".to_string()),
            (2, "a".to_string()),
            (3, "old_b".to_string()),
        ]);
        let id_metadata =
            |id: i32| HashMap::from([(PARQUET_FIELD_ID_META_KEY.to_string(), id.to_string())]);
        let file_schema = Schema::new(vec![
            Field::new(
                "payload",
                DataType::Struct(
                    vec![
                        Arc::new(
                            Field::new("a", DataType::Int32, false).with_metadata(id_metadata(2)),
                        ),
                        Arc::new(
                            Field::new("old_b", DataType::Utf8, true).with_metadata(id_metadata(3)),
                        ),
                    ]
                    .into(),
                ),
                false,
            )
            .with_metadata(id_metadata(1)),
        ]);

        let (read_schema, _) = build_read_schema_with_field_id_mapping(
            &current_columns,
            &parquet_field_ids,
            Some(&file_schema),
        )
        .unwrap();
        let DataType::Struct(fields) = read_schema.field(0).data_type() else {
            panic!("payload must remain a struct");
        };

        assert_eq!(fields.len(), 3);
        assert_eq!(fields[0].name(), "a");
        assert_eq!(fields[1].name(), "old_b");
        assert_eq!(fields[2].name(), "__ducklake_absent_field_4");
        assert_eq!(fields[2].data_type(), &DataType::Int32);
        assert!(fields[2].is_nullable());
    }

    #[test]
    fn test_build_read_schema_does_not_reuse_dropped_nested_name() {
        let current_columns = vec![DuckLakeTableColumn {
            column_id: 1,
            column_name: "payload".to_string(),
            column_type: "struct<c:int32>".to_string(),
            is_nullable: false,
            data_type: Some(DataType::Struct(
                vec![Arc::new(Field::new("c", DataType::Int32, true))].into(),
            )),
            nested_column_ids: vec![4],
            initial_default: None,
            default_value: None,
            default_value_type: None,
            default_value_dialect: None,
        }];
        let parquet_field_ids = HashMap::from([(1, "payload".to_string()), (3, "c".to_string())]);
        let id_metadata =
            |id: i32| HashMap::from([(PARQUET_FIELD_ID_META_KEY.to_string(), id.to_string())]);
        let file_schema = Schema::new(vec![
            Field::new(
                "payload",
                DataType::Struct(
                    vec![Arc::new(
                        Field::new("c", DataType::Int32, true).with_metadata(id_metadata(3)),
                    )]
                    .into(),
                ),
                false,
            )
            .with_metadata(id_metadata(1)),
        ]);

        let (read_schema, _) = build_read_schema_with_field_id_mapping(
            &current_columns,
            &parquet_field_ids,
            Some(&file_schema),
        )
        .unwrap();
        let DataType::Struct(fields) = read_schema.field(0).data_type() else {
            panic!("payload must remain a struct");
        };

        assert_eq!(fields[0].name(), "__ducklake_absent_field_4");
    }

    /// Collect `(dotted path, field id)` for every nested node of `data_type`,
    /// reading the id from the node's `PARQUET:field_id` metadata. Nodes without
    /// the metadata are reported with `None` so a missing id is visible in the
    /// assertion rather than silently skipped.
    fn nested_field_ids(data_type: &DataType) -> Vec<(String, Option<String>)> {
        fn walk(prefix: &str, field: &Field, out: &mut Vec<(String, Option<String>)>) {
            let path = format!("{prefix}.{}", field.name());
            out.push((
                path.clone(),
                field.metadata().get(PARQUET_FIELD_ID_META_KEY).cloned(),
            ));
            collect(&path, field.data_type(), out);
        }
        fn collect(prefix: &str, data_type: &DataType, out: &mut Vec<(String, Option<String>)>) {
            match data_type {
                DataType::List(child)
                | DataType::LargeList(child)
                | DataType::FixedSizeList(child, _) => walk(prefix, child, out),
                DataType::Struct(children) => {
                    for child in children {
                        walk(prefix, child, out);
                    }
                },
                // The Map wrapper ("entries"/"key_value") is a synthetic parquet
                // group with no DuckLake column and no field id; descend through
                // it without recording it.
                DataType::Map(entries, _) => collect(prefix, entries.data_type(), out),
                _ => {},
            }
        }

        let mut out = Vec::new();
        collect("", data_type, &mut out);
        out
    }

    /// Columns for a table shaped
    /// `v LIST<FLOAT>, s STRUCT<label VARCHAR, score INT>, m MAP<VARCHAR, INT>, nn LIST<LIST<INT>>`,
    /// with catalog ids assigned depth-first (1..12) the way DuckLake assigns them.
    fn nested_test_columns() -> Vec<DuckLakeTableColumn> {
        vec![
            DuckLakeTableColumn {
                column_id: 1,
                column_name: "v".to_string(),
                column_type: "list".to_string(),
                is_nullable: true,
                data_type: Some(DataType::List(Arc::new(Field::new(
                    "item",
                    DataType::Float32,
                    true,
                )))),
                nested_column_ids: vec![2],
                initial_default: None,
                default_value: None,
                default_value_type: None,
                default_value_dialect: None,
            },
            DuckLakeTableColumn {
                column_id: 3,
                column_name: "s".to_string(),
                column_type: "struct".to_string(),
                is_nullable: true,
                data_type: Some(DataType::Struct(
                    vec![
                        Arc::new(Field::new("label", DataType::Utf8View, true)),
                        Arc::new(Field::new("score", DataType::Int32, true)),
                    ]
                    .into(),
                )),
                nested_column_ids: vec![4, 5],
                initial_default: None,
                default_value: None,
                default_value_type: None,
                default_value_dialect: None,
            },
            DuckLakeTableColumn {
                column_id: 6,
                column_name: "m".to_string(),
                column_type: "map".to_string(),
                is_nullable: true,
                data_type: Some(DataType::Map(
                    Arc::new(Field::new(
                        "entries",
                        DataType::Struct(
                            vec![
                                Arc::new(Field::new("key", DataType::Utf8View, false)),
                                Arc::new(Field::new("value", DataType::Int32, true)),
                            ]
                            .into(),
                        ),
                        false,
                    )),
                    false,
                )),
                nested_column_ids: vec![7, 8],
                initial_default: None,
                default_value: None,
                default_value_type: None,
                default_value_dialect: None,
            },
            DuckLakeTableColumn {
                column_id: 9,
                column_name: "nn".to_string(),
                column_type: "list".to_string(),
                is_nullable: true,
                data_type: Some(DataType::List(Arc::new(Field::new(
                    "item",
                    DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
                    true,
                )))),
                nested_column_ids: vec![10, 11],
                initial_default: None,
                default_value: None,
                default_value_type: None,
                default_value_dialect: None,
            },
        ]
    }

    /// The file schema a DuckLake writer produces for [`nested_test_columns`]:
    /// every semantic node tagged with its field id, list children named
    /// `element`, and no id on the Map wrapper group.
    fn nested_test_file_schema() -> Schema {
        let id = |id: i32| HashMap::from([(PARQUET_FIELD_ID_META_KEY.to_string(), id.to_string())]);
        Schema::new(vec![
            Field::new(
                "v",
                DataType::List(Arc::new(
                    Field::new("element", DataType::Float32, true).with_metadata(id(2)),
                )),
                true,
            )
            .with_metadata(id(1)),
            Field::new(
                "s",
                DataType::Struct(
                    vec![
                        Arc::new(Field::new("label", DataType::Utf8, true).with_metadata(id(4))),
                        Arc::new(Field::new("score", DataType::Int32, true).with_metadata(id(5))),
                    ]
                    .into(),
                ),
                true,
            )
            .with_metadata(id(3)),
            Field::new(
                "m",
                DataType::Map(
                    Arc::new(Field::new(
                        "key_value",
                        DataType::Struct(
                            vec![
                                Arc::new(
                                    Field::new("key", DataType::Utf8, false).with_metadata(id(7)),
                                ),
                                Arc::new(
                                    Field::new("value", DataType::Int32, true).with_metadata(id(8)),
                                ),
                            ]
                            .into(),
                        ),
                        false,
                    )),
                    false,
                ),
                true,
            )
            .with_metadata(id(6)),
            Field::new(
                "nn",
                DataType::List(Arc::new(
                    Field::new(
                        "element",
                        DataType::List(Arc::new(
                            Field::new("element", DataType::Int32, true).with_metadata(id(11)),
                        )),
                        true,
                    )
                    .with_metadata(id(10)),
                )),
                true,
            )
            .with_metadata(id(9)),
        ])
    }

    fn nested_test_parquet_field_ids() -> HashMap<i32, String> {
        HashMap::from([
            (1, "v".to_string()),
            (2, "element".to_string()),
            (3, "s".to_string()),
            (4, "label".to_string()),
            (5, "score".to_string()),
            (6, "m".to_string()),
            (7, "key".to_string()),
            (8, "value".to_string()),
            (9, "nn".to_string()),
            (10, "element".to_string()),
            (11, "element".to_string()),
        ])
    }

    /// A nested node's `PARQUET:field_id` is part of its parent's Arrow *type*,
    /// so it takes part in every batch type check. The read schema describes the
    /// physical file to the parquet reader, and DuckLake tags every nested node
    /// in the file with its field id — so the read schema must declare the same
    /// ids for List elements, Struct children and Map key/value at every depth.
    #[test]
    fn test_read_schema_declares_nested_field_ids() {
        let (read_schema, _) = build_read_schema_with_field_id_mapping(
            &nested_test_columns(),
            &nested_test_parquet_field_ids(),
            Some(&nested_test_file_schema()),
        )
        .unwrap();

        let ids: Vec<(String, Option<String>)> = read_schema
            .fields()
            .iter()
            .flat_map(|field| nested_field_ids(field.data_type()))
            .collect();
        assert_eq!(
            ids,
            vec![
                (".element".to_string(), Some("2".to_string())),
                (".label".to_string(), Some("4".to_string())),
                (".score".to_string(), Some("5".to_string())),
                (".key".to_string(), Some("7".to_string())),
                (".value".to_string(), Some("8".to_string())),
                (".element".to_string(), Some("10".to_string())),
                (".element.element".to_string(), Some("11".to_string())),
            ],
            "every nested node must declare the field id the file records for it"
        );

        // The Map wrapper is a synthetic parquet group, not a DuckLake column: it
        // must stay untagged, matching what the writer emits.
        let DataType::Map(entries, _) = read_schema.field(2).data_type() else {
            panic!("m must remain a map");
        };
        assert!(
            !entries.metadata().contains_key(PARQUET_FIELD_ID_META_KEY),
            "the map wrapper group has no DuckLake column and must carry no field id"
        );

        // Top-level fields are matched by name and their metadata is not part of
        // any Arrow type, so they stay bare.
        for field in read_schema.fields() {
            assert!(
                !field.metadata().contains_key(PARQUET_FIELD_ID_META_KEY),
                "top-level field '{}' must not be tagged",
                field.name()
            );
        }
    }

    #[test]
    fn ducklake_primitive_literals_decode_exactly() {
        let cases = vec![
            ("boolean", "1", ScalarValue::Boolean(Some(true))),
            ("int8", "-8", ScalarValue::Int8(Some(-8))),
            ("int16", "-1600", ScalarValue::Int16(Some(-1600))),
            ("int32", "-3200", ScalarValue::Int32(Some(-3200))),
            ("int64", "-6400", ScalarValue::Int64(Some(-6400))),
            ("uint8", "8", ScalarValue::UInt8(Some(8))),
            ("uint16", "1600", ScalarValue::UInt16(Some(1600))),
            ("uint32", "3200", ScalarValue::UInt32(Some(3200))),
            ("uint64", "6400", ScalarValue::UInt64(Some(6400))),
            ("float32", "3.5", ScalarValue::Float32(Some(3.5))),
            ("float64", "7.25", ScalarValue::Float64(Some(7.25))),
            (
                "decimal(10,2)",
                "123.45",
                ScalarValue::Decimal128(Some(12_345), 10, 2),
            ),
            (
                "time",
                "00:00:01.000002",
                ScalarValue::Time64Microsecond(Some(1_000_002)),
            ),
            ("date", "1970-01-02", ScalarValue::Date32(Some(1))),
            (
                "timestamp",
                "1970-01-01 00:00:01.000002",
                ScalarValue::TimestampMicrosecond(Some(1_000_002), None),
            ),
            (
                "timestamptz",
                "1970-01-01 00:00:01.000002+00",
                ScalarValue::TimestampMicrosecond(Some(1_000_002), Some("UTC".into())),
            ),
            (
                "timestamp_s",
                "1970-01-01 00:00:01",
                ScalarValue::TimestampSecond(Some(1), None),
            ),
            (
                "timestamp_ms",
                "1970-01-01 00:00:00.001",
                ScalarValue::TimestampMillisecond(Some(1), None),
            ),
            (
                "timestamp_ns",
                "1970-01-01 00:00:00.000000001",
                ScalarValue::TimestampNanosecond(Some(1), None),
            ),
            (
                "interval",
                "1 month 2 days 3 seconds",
                ScalarValue::new_interval_mdn(1, 2, 3_000_000_000),
            ),
            (
                "varchar",
                "hello",
                ScalarValue::Utf8View(Some("hello".to_string())),
            ),
            (
                "json",
                r#"{"key":"value"}"#,
                ScalarValue::Utf8View(Some(r#"{"key":"value"}"#.to_string())),
            ),
            (
                "blob",
                r"\x68656C6C6F",
                ScalarValue::BinaryView(Some(b"hello".to_vec())),
            ),
            (
                "uuid",
                "550e8400-e29b-41d4-a716-446655440000",
                ScalarValue::FixedSizeBinary(
                    16,
                    Some(vec![
                        0x55, 0x0e, 0x84, 0x00, 0xe2, 0x9b, 0x41, 0xd4, 0xa7, 0x16, 0x44, 0x66,
                        0x55, 0x44, 0x00, 0x00,
                    ]),
                ),
            ),
            (
                "timetz",
                "12:30:00+02",
                ScalarValue::Utf8View(Some("12:30:00+02".to_string())),
            ),
        ];

        for (ducklake_type, encoded, expected) in cases {
            let data_type = ducklake_to_arrow_type(ducklake_type).unwrap();
            assert_eq!(
                parse_ducklake_scalar(encoded, &data_type),
                Some(expected),
                "DuckLake {ducklake_type} literal {encoded}"
            );
        }
    }

    /// An external / legacy file whose nested nodes carry no field ids is matched
    /// structurally. Declaring ids the file does not have would be the same
    /// divergence in the opposite direction, so nothing is stamped.
    #[test]
    fn test_read_schema_omits_nested_field_ids_for_file_without_them() {
        let file_schema = Schema::new(vec![
            Field::new(
                "v",
                DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
                true,
            ),
            Field::new(
                "s",
                DataType::Struct(
                    vec![
                        Arc::new(Field::new("label", DataType::Utf8, true)),
                        Arc::new(Field::new("score", DataType::Int32, true)),
                    ]
                    .into(),
                ),
                true,
            ),
        ]);
        let columns = nested_test_columns()[..2].to_vec();
        // A file written before nested nodes were tagged: top-level ids only.
        let parquet_field_ids = HashMap::from([(1, "v".to_string()), (3, "s".to_string())]);

        let (read_schema, _) = build_read_schema_with_field_id_mapping(
            &columns,
            &parquet_field_ids,
            Some(&file_schema),
        )
        .unwrap();

        let ids: Vec<(String, Option<String>)> = read_schema
            .fields()
            .iter()
            .flat_map(|field| nested_field_ids(field.data_type()))
            .collect();
        assert_eq!(
            ids,
            vec![
                (".item".to_string(), None),
                (".label".to_string(), None),
                (".score".to_string(), None),
            ],
            "a file without nested field ids must not be described as having them"
        );
    }

    /// A nested field the file predates reads as NULL under a synthetic name.
    /// There is no physical node to describe, so it gets no field id either.
    #[test]
    fn test_read_schema_omits_field_id_for_absent_nested_field() {
        let mut parquet_field_ids = nested_test_parquet_field_ids();
        parquet_field_ids.remove(&5); // `score` was added after this file was written

        let (read_schema, _) = build_read_schema_with_field_id_mapping(
            &nested_test_columns()[1..2],
            &parquet_field_ids,
            Some(&nested_test_file_schema()),
        )
        .unwrap();

        let DataType::Struct(fields) = read_schema.field(0).data_type() else {
            panic!("s must remain a struct");
        };
        assert_eq!(
            fields[0].metadata().get(PARQUET_FIELD_ID_META_KEY),
            Some(&"4".to_string()),
            "a present nested field keeps its id"
        );
        assert_eq!(fields[1].name(), "__ducklake_absent_field_5");
        assert!(
            !fields[1].metadata().contains_key(PARQUET_FIELD_ID_META_KEY),
            "a nested field absent from the file has no physical node to tag"
        );
    }

    #[test]
    fn test_nested_type_depth_is_bounded() {
        let nested = format!(
            "{}int32{}",
            "list<".repeat(MAX_NESTED_TYPE_DEPTH + 1),
            ">".repeat(MAX_NESTED_TYPE_DEPTH + 1)
        );

        let error = ducklake_to_arrow_type(&nested).unwrap_err();

        assert!(error.to_string().contains("maximum nesting depth"));
    }

    #[test]
    fn ducklake_blob_defaults_preserve_text_and_decode_escapes() {
        assert_eq!(
            parse_ducklake_default_scalar("abcd", &DataType::BinaryView),
            Some(ScalarValue::BinaryView(Some(b"abcd".to_vec())))
        );
        assert_eq!(
            parse_ducklake_default_scalar(r"\xAB\xCD", &DataType::BinaryView),
            Some(ScalarValue::BinaryView(Some(vec![0xab, 0xcd])))
        );
        assert_eq!(
            parse_ducklake_default_scalar("abc", &DataType::BinaryView),
            Some(ScalarValue::BinaryView(Some(b"abc".to_vec())))
        );
    }

    #[test]
    fn ducklake_primitive_literals_reject_invalid_encodings() {
        for (ducklake_type, encoded) in [
            ("boolean", "yes"),
            ("int32", "3.5"),
            ("uint8", "256"),
            ("date", "not-a-date"),
            ("blob", "ABC"),
            ("uuid", "550e8400-e29b-41d4-a716"),
        ] {
            let data_type = ducklake_to_arrow_type(ducklake_type).unwrap();
            assert_eq!(
                parse_ducklake_scalar(encoded, &data_type),
                None,
                "DuckLake {ducklake_type} literal {encoded} must be rejected"
            );
        }
    }

    #[test]
    fn test_basic_types() {
        assert_eq!(
            ducklake_to_arrow_type("boolean").unwrap(),
            DataType::Boolean
        );
        assert_eq!(ducklake_to_arrow_type("int32").unwrap(), DataType::Int32);
        assert_eq!(ducklake_to_arrow_type("int64").unwrap(), DataType::Int64);
        assert_eq!(
            ducklake_to_arrow_type("float64").unwrap(),
            DataType::Float64
        );
        assert_eq!(
            ducklake_to_arrow_type("varchar").unwrap(),
            DataType::Utf8View
        );
        assert_eq!(
            ducklake_to_arrow_type("blob").unwrap(),
            DataType::BinaryView
        );
    }

    #[test]
    fn test_string_types_map_to_utf8view() {
        // String columns use the Utf8View layout so wide, high-cardinality
        // group-by does not hit the 2 GiB i32-offset buffer limit, matching
        // DataFusion's default parquet read behaviour (schema_force_view_types).
        for t in ["varchar", "text", "string", "json", "timetz", "time with time zone"] {
            assert_eq!(
                ducklake_to_arrow_type(t).unwrap(),
                DataType::Utf8View,
                "{t} should map to Utf8View"
            );
        }
    }

    #[test]
    fn test_binary_types_map_to_binaryview() {
        for t in ["blob", "binary", "bytea"] {
            assert_eq!(
                ducklake_to_arrow_type(t).unwrap(),
                DataType::BinaryView,
                "{t} should map to BinaryView"
            );
        }
    }

    #[test]
    fn test_geometry_stays_binary() {
        // Geometry WKB is consumed by geometry functions that expect the Binary
        // layout, so it is deliberately not promoted to BinaryView.
        for t in [
            "geometry",
            "point",
            "linestring",
            "polygon",
            "multipoint",
            "multilinestring",
            "multipolygon",
            "geometrycollection",
        ] {
            assert_eq!(
                ducklake_to_arrow_type(t).unwrap(),
                DataType::Binary,
                "{t} should stay Binary"
            );
        }
    }

    #[test]
    fn test_uuid_stays_fixed_size_binary() {
        assert_eq!(
            ducklake_to_arrow_type("uuid").unwrap(),
            DataType::FixedSizeBinary(16)
        );
    }

    #[test]
    fn test_view_types_write_back_to_string_and_blob() {
        // The write direction accepts every string/binary Arrow layout, including
        // the view layouts now produced on read, so a read/write round-trip keeps
        // the DuckLake catalog type stable.
        assert_eq!(
            arrow_to_ducklake_type(&DataType::Utf8View).unwrap(),
            "varchar"
        );
        assert_eq!(arrow_to_ducklake_type(&DataType::Utf8).unwrap(), "varchar");
        assert_eq!(
            arrow_to_ducklake_type(&DataType::LargeUtf8).unwrap(),
            "varchar"
        );
        assert_eq!(
            arrow_to_ducklake_type(&DataType::BinaryView).unwrap(),
            "blob"
        );
        assert_eq!(arrow_to_ducklake_type(&DataType::Binary).unwrap(), "blob");
        assert_eq!(
            arrow_to_ducklake_type(&DataType::LargeBinary).unwrap(),
            "blob"
        );
    }

    #[test]
    fn test_dictionary_uses_logical_value_type() {
        let dictionary = DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8));
        assert_eq!(arrow_to_ducklake_type(&dictionary).unwrap(), "varchar");

        let list = DataType::List(Arc::new(Field::new("item", dictionary, true)));
        assert_eq!(arrow_to_ducklake_type(&list).unwrap(), "list<varchar>");
    }

    #[test]
    fn test_string_binary_normalize_is_stable() {
        // normalize = arrow_to_ducklake_type(ducklake_to_arrow_type(t)); it must
        // still terminate at the canonical DuckLake type now that the read layout
        // is a view type, otherwise schema-evolution comparisons would error.
        assert_eq!(normalize_ducklake_type("varchar").unwrap(), "varchar");
        assert_eq!(normalize_ducklake_type("text").unwrap(), "varchar");
        assert_eq!(normalize_ducklake_type("json").unwrap(), "varchar");
        assert_eq!(normalize_ducklake_type("blob").unwrap(), "blob");
        assert_eq!(normalize_ducklake_type("binary").unwrap(), "blob");
    }

    #[test]
    fn test_view_type_list_children() {
        // list<varchar> recurses through the same mapping, so its element is
        // Utf8View.
        assert_eq!(
            ducklake_to_arrow_type("list<varchar>").unwrap(),
            DataType::List(Arc::new(Field::new("item", DataType::Utf8View, true)))
        );
    }

    #[test]
    fn test_decimal_types() {
        assert_eq!(
            ducklake_to_arrow_type("decimal(10, 2)").unwrap(),
            DataType::Decimal128(10, 2)
        );
        assert_eq!(
            ducklake_to_arrow_type("decimal(38, 10)").unwrap(),
            DataType::Decimal128(38, 10)
        );
    }

    #[test]
    fn test_decimal_single_param_over_38_uses_decimal256() {
        // A single-parameter `decimal(P)` with P > 38 must widen to Decimal256,
        // like the two-parameter path — not build an invalid Decimal128 (Arrow
        // caps Decimal128 precision at 38).
        assert_eq!(
            ducklake_to_arrow_type("decimal(50)").unwrap(),
            DataType::Decimal256(50, 0)
        );
        // At/under 38 stays Decimal128 on the single-parameter path.
        assert_eq!(
            ducklake_to_arrow_type("decimal(38)").unwrap(),
            DataType::Decimal128(38, 0)
        );
        // The two-parameter path keeps switching on > 38 too.
        assert_eq!(
            ducklake_to_arrow_type("decimal(50, 10)").unwrap(),
            DataType::Decimal256(50, 10)
        );
    }

    #[test]
    fn test_field_ids_dropping_duplicates() {
        // Unique field_ids are kept as-is.
        let unique =
            field_ids_dropping_duplicates([(1, "a".to_string()), (2, "b".to_string())].into_iter());
        assert_eq!(unique.get(&1), Some(&"a".to_string()));
        assert_eq!(unique.get(&2), Some(&"b".to_string()));

        // A field_id shared by two columns is dropped entirely (neither name
        // wins), so the reader null-fills that column instead of binding the
        // wrong one. Other ids are unaffected.
        let with_dup = field_ids_dropping_duplicates(
            [(3, "a".to_string()), (3, "b".to_string()), (4, "c".to_string())].into_iter(),
        );
        assert_eq!(
            with_dup.get(&3),
            None,
            "a duplicated field_id must not map to either column"
        );
        assert_eq!(with_dup.get(&4), Some(&"c".to_string()));
    }

    #[test]
    fn test_temporal_types() {
        assert_eq!(ducklake_to_arrow_type("date").unwrap(), DataType::Date32);
        assert_eq!(
            ducklake_to_arrow_type("timestamp").unwrap(),
            DataType::Timestamp(TimeUnit::Microsecond, None)
        );
        assert_eq!(
            ducklake_to_arrow_type("timestamptz").unwrap(),
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into()))
        );
        // Nanosecond tz-aware timestamps map to DuckLake's TIMESTAMP_TZ_NS.
        assert_eq!(
            ducklake_to_arrow_type("timestamptz_ns").unwrap(),
            DataType::Timestamp(TimeUnit::Nanosecond, Some("UTC".into()))
        );
    }

    #[test]
    fn test_list_type_angle_bracket() {
        let result = ducklake_to_arrow_type("list<int32>").unwrap();
        let expected = DataType::List(Arc::new(Field::new("item", DataType::Int32, true)));
        assert_eq!(result, expected);
    }

    #[test]
    fn test_list_type_various_elements() {
        let cases = vec![
            ("list<varchar>", DataType::Utf8View),
            ("list<float64>", DataType::Float64),
            ("list<boolean>", DataType::Boolean),
            ("list<date>", DataType::Date32),
        ];
        for (type_str, expected_inner) in cases {
            let result = ducklake_to_arrow_type(type_str).unwrap();
            let expected =
                DataType::List(Arc::new(Field::new("item", expected_inner.clone(), true)));
            assert_eq!(result, expected, "Failed for {}", type_str);
        }
    }

    #[test]
    fn test_array_type_angle_bracket() {
        let result = ducklake_to_arrow_type("array<varchar>").unwrap();
        let expected = DataType::List(Arc::new(Field::new("item", DataType::Utf8View, true)));
        assert_eq!(result, expected);
    }

    #[test]
    fn test_list_type_postgres_bracket_syntax() {
        let cases = vec![
            ("varchar[]", DataType::Utf8View),
            ("float64[]", DataType::Float64),
            ("int32[]", DataType::Int32),
            ("boolean[]", DataType::Boolean),
            ("bigint[]", DataType::Int64),
            ("text[]", DataType::Utf8View),
            ("float[]", DataType::Float32),
            ("integer[]", DataType::Int32),
        ];
        for (type_str, expected_inner) in cases {
            let result = ducklake_to_arrow_type(type_str).unwrap();
            let expected =
                DataType::List(Arc::new(Field::new("item", expected_inner.clone(), true)));
            assert_eq!(result, expected, "Failed for {}", type_str);
        }
    }

    #[test]
    fn test_list_type_empty_element_errors() {
        assert!(ducklake_to_arrow_type("list<>").is_err());
        assert!(ducklake_to_arrow_type("[]").is_err());
    }

    #[test]
    fn test_struct_type() {
        let result = ducklake_to_arrow_type("STRUCT<Price:DECIMAL(38,16),Label:VARCHAR>").unwrap();
        assert_eq!(
            result,
            DataType::Struct(
                vec![
                    Arc::new(Field::new("Price", DataType::Decimal128(38, 16), true,)),
                    Arc::new(Field::new("Label", DataType::Utf8View, true)),
                ]
                .into(),
            )
        );
    }

    #[test]
    fn test_map_type() {
        let result = ducklake_to_arrow_type("map<varchar,list<int32>>").unwrap();
        let DataType::Map(entries, false) = result else {
            panic!("expected map");
        };
        let DataType::Struct(fields) = entries.data_type() else {
            panic!("expected map entries struct");
        };
        assert_eq!(fields[0].name(), "key");
        assert_eq!(fields[0].data_type(), &DataType::Utf8View);
        assert_eq!(
            fields[1].data_type(),
            &DataType::List(Arc::new(Field::new("item", DataType::Int32, true)))
        );
    }

    #[test]
    fn test_arbitrarily_composed_nested_types() {
        let type_name = "list<struct<levels:list<struct<price:decimal(38, 16)>>,attrs:map<varchar,list<int32>>>>";
        let arrow = ducklake_to_arrow_type(type_name).unwrap();
        assert_eq!(arrow_to_ducklake_type(&arrow).unwrap(), type_name);

        let nested_lists = ducklake_to_arrow_type("int32[][]").unwrap();
        assert_eq!(
            arrow_to_ducklake_type(&nested_lists).unwrap(),
            "list<list<int32>>"
        );
    }

    #[test]
    fn test_unknown_type_error() {
        // Test completely unknown types also return error
        let result = ducklake_to_arrow_type("completely_unknown_type");
        assert!(result.is_err());
        match result {
            Err(DuckLakeError::UnsupportedType(msg)) => {
                assert_eq!(msg, "completely_unknown_type");
            },
            _ => panic!("Expected UnsupportedType error for unknown type"),
        }
    }

    #[test]
    fn test_arrow_to_ducklake_basic_types() {
        assert_eq!(
            arrow_to_ducklake_type(&DataType::Boolean).unwrap(),
            "boolean"
        );
        assert_eq!(arrow_to_ducklake_type(&DataType::Int8).unwrap(), "int8");
        assert_eq!(arrow_to_ducklake_type(&DataType::Int16).unwrap(), "int16");
        assert_eq!(arrow_to_ducklake_type(&DataType::Int32).unwrap(), "int32");
        assert_eq!(arrow_to_ducklake_type(&DataType::Int64).unwrap(), "int64");
        assert_eq!(arrow_to_ducklake_type(&DataType::UInt8).unwrap(), "uint8");
        assert_eq!(arrow_to_ducklake_type(&DataType::UInt16).unwrap(), "uint16");
        assert_eq!(arrow_to_ducklake_type(&DataType::UInt32).unwrap(), "uint32");
        assert_eq!(arrow_to_ducklake_type(&DataType::UInt64).unwrap(), "uint64");
        assert_eq!(
            arrow_to_ducklake_type(&DataType::Float32).unwrap(),
            "float32"
        );
        assert_eq!(
            arrow_to_ducklake_type(&DataType::Float64).unwrap(),
            "float64"
        );
        assert_eq!(arrow_to_ducklake_type(&DataType::Utf8).unwrap(), "varchar");
        assert_eq!(arrow_to_ducklake_type(&DataType::Binary).unwrap(), "blob");
    }

    #[test]
    fn test_arrow_to_ducklake_temporal_types() {
        assert_eq!(arrow_to_ducklake_type(&DataType::Date32).unwrap(), "date");
        assert_eq!(arrow_to_ducklake_type(&DataType::Date64).unwrap(), "date");
        assert_eq!(
            arrow_to_ducklake_type(&DataType::Time64(TimeUnit::Microsecond)).unwrap(),
            "time"
        );
        assert_eq!(
            arrow_to_ducklake_type(&DataType::Timestamp(TimeUnit::Microsecond, None)).unwrap(),
            "timestamp"
        );
        assert_eq!(
            arrow_to_ducklake_type(&DataType::Timestamp(
                TimeUnit::Microsecond,
                Some("UTC".into())
            ))
            .unwrap(),
            "timestamptz"
        );
        // Nanosecond tz-aware timestamps get their own DuckLake type rather than
        // collapsing to µs `timestamptz` (which would silently truncate on read).
        assert_eq!(
            arrow_to_ducklake_type(&DataType::Timestamp(
                TimeUnit::Nanosecond,
                Some("UTC".into())
            ))
            .unwrap(),
            "timestamptz_ns"
        );
        // A non-UTC zone label still selects the ns type by unit; the instant is
        // UTC-normalised and the zone relabels to UTC on read (DuckLake stores an
        // instant, not the zone name).
        assert_eq!(
            arrow_to_ducklake_type(&DataType::Timestamp(
                TimeUnit::Nanosecond,
                Some("America/New_York".into())
            ))
            .unwrap(),
            "timestamptz_ns"
        );
        // Second/millisecond tz timestamps have no DuckLake type; they widen
        // losslessly to µs `timestamptz`.
        assert_eq!(
            arrow_to_ducklake_type(&DataType::Timestamp(
                TimeUnit::Millisecond,
                Some("UTC".into())
            ))
            .unwrap(),
            "timestamptz"
        );
    }

    #[test]
    fn test_arrow_to_ducklake_decimal() {
        assert_eq!(
            arrow_to_ducklake_type(&DataType::Decimal128(10, 2)).unwrap(),
            "decimal(10, 2)"
        );
        assert_eq!(
            arrow_to_ducklake_type(&DataType::Decimal256(40, 5)).unwrap(),
            "decimal(40, 5)"
        );
    }

    #[test]
    fn test_arrow_to_ducklake_uuid() {
        assert_eq!(
            arrow_to_ducklake_type(&DataType::FixedSizeBinary(16)).unwrap(),
            "uuid"
        );
        // Non-16 byte fixed size binary becomes blob
        assert_eq!(
            arrow_to_ducklake_type(&DataType::FixedSizeBinary(32)).unwrap(),
            "blob"
        );
    }

    #[test]
    fn test_arrow_to_ducklake_roundtrip() {
        // Verify roundtrip: arrow -> ducklake -> arrow for common types. Strings
        // and binary use the view layouts (Utf8View/BinaryView) here because that
        // is the canonical Arrow type `ducklake_to_arrow_type` produces; the
        // non-view layouts collapse to the same DuckLake type and are covered by
        // `test_view_types_write_back_to_string_and_blob`.
        let test_types = vec![
            DataType::Boolean,
            DataType::Int32,
            DataType::Int64,
            DataType::Float64,
            DataType::Utf8View,
            DataType::BinaryView,
            DataType::Date32,
            DataType::Timestamp(TimeUnit::Microsecond, None),
            DataType::Timestamp(TimeUnit::Nanosecond, None),
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            DataType::Timestamp(TimeUnit::Nanosecond, Some("UTC".into())),
            DataType::Decimal128(10, 2),
            DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
            DataType::List(Arc::new(Field::new("item", DataType::Utf8View, true))),
        ];

        for original in test_types {
            let ducklake = arrow_to_ducklake_type(&original).unwrap();
            let back = ducklake_to_arrow_type(&ducklake).unwrap();
            assert_eq!(original, back, "Roundtrip failed for {:?}", original);
        }
    }

    /// Regression: a nanosecond tz-aware timestamp (the pandas/PyArrow default
    /// for tz-aware datetimes) must not be cataloged as µs `timestamptz`. Doing
    /// so left the physical parquet at ns while the catalog claimed µs, so the
    /// read path silently truncated sub-microsecond precision on every scan.
    #[test]
    fn test_nanosecond_timestamptz_preserves_precision() {
        let ns_tz = DataType::Timestamp(TimeUnit::Nanosecond, Some("UTC".into()));

        // Catalog type must encode nanosecond precision, not collapse to µs.
        let ducklake = arrow_to_ducklake_type(&ns_tz).unwrap();
        assert_eq!(ducklake, "timestamptz_ns");
        assert_ne!(
            ducklake, "timestamptz",
            "ns tz-aware timestamp must not collapse to µs timestamptz"
        );

        // And the catalog type round-trips back to nanosecond precision, so the
        // served schema matches the physical parquet and no ns->µs cast occurs.
        let back = ducklake_to_arrow_type(&ducklake).unwrap();
        assert_eq!(back, ns_tz);

        // The two tz precisions stay distinct (not mutually promotable): changing
        // a column's precision is not a safe widening.
        assert!(!is_promotable("timestamptz", "timestamptz_ns"));
        assert!(!is_promotable("timestamptz_ns", "timestamptz"));
        assert!(!types_compatible("timestamptz", "timestamptz_ns"));
    }

    #[test]
    fn test_arrow_to_ducklake_list() {
        let list_type = DataType::List(Arc::new(Field::new("item", DataType::Int32, true)));
        assert_eq!(arrow_to_ducklake_type(&list_type).unwrap(), "list<int32>");

        let list_type = DataType::List(Arc::new(Field::new("item", DataType::Utf8, true)));
        assert_eq!(arrow_to_ducklake_type(&list_type).unwrap(), "list<varchar>");

        let large_list = DataType::LargeList(Arc::new(Field::new("item", DataType::Float64, true)));
        assert_eq!(
            arrow_to_ducklake_type(&large_list).unwrap(),
            "list<float64>"
        );
    }

    #[test]
    fn test_arrow_to_ducklake_struct() {
        let struct_type = DataType::Struct(vec![Field::new("a", DataType::Int32, true)].into());
        assert_eq!(
            arrow_to_ducklake_type(&struct_type).unwrap(),
            "struct<a:int32>"
        );
    }

    #[test]
    fn test_decimal_precision_zero_rejected() {
        let result = ducklake_to_arrow_type("decimal(0, 0)");
        assert!(result.is_err());
        match result {
            Err(DuckLakeError::UnsupportedType(msg)) => {
                assert!(msg.contains("precision must be >= 1"));
            },
            _ => panic!("Expected UnsupportedType error for precision=0"),
        }
    }

    #[test]
    fn test_decimal_precision_too_large_rejected() {
        let result = ducklake_to_arrow_type("decimal(77, 0)");
        assert!(result.is_err());
        match result {
            Err(DuckLakeError::UnsupportedType(msg)) => {
                assert!(msg.contains("precision must be <= 76"));
            },
            _ => panic!("Expected UnsupportedType error for precision=77"),
        }
    }

    #[test]
    fn test_decimal_precision_255_rejected() {
        let result = ducklake_to_arrow_type("decimal(255, 0)");
        assert!(result.is_err());
        match result {
            Err(DuckLakeError::UnsupportedType(msg)) => {
                assert!(msg.contains("precision must be <= 76"));
            },
            _ => panic!("Expected UnsupportedType error for precision=255"),
        }
    }

    #[test]
    fn test_decimal_scale_exceeds_precision_rejected() {
        let result = ducklake_to_arrow_type("decimal(10, 11)");
        assert!(result.is_err());
        match result {
            Err(DuckLakeError::UnsupportedType(msg)) => {
                assert!(msg.contains("scale (11) must not exceed precision (10)"));
            },
            _ => panic!("Expected UnsupportedType error for scale > precision"),
        }
    }

    #[test]
    fn test_decimal_valid_edge_cases() {
        assert_eq!(
            ducklake_to_arrow_type("decimal(1, 0)").unwrap(),
            DataType::Decimal128(1, 0)
        );
        assert_eq!(
            ducklake_to_arrow_type("decimal(38, 0)").unwrap(),
            DataType::Decimal128(38, 0)
        );
        assert_eq!(
            ducklake_to_arrow_type("decimal(39, 0)").unwrap(),
            DataType::Decimal256(39, 0)
        );
        assert_eq!(
            ducklake_to_arrow_type("decimal(76, 0)").unwrap(),
            DataType::Decimal256(76, 0)
        );
        assert_eq!(
            ducklake_to_arrow_type("decimal(10, 10)").unwrap(),
            DataType::Decimal128(10, 10)
        );
    }

    #[test]
    fn test_decimal_negative_precision_rejected() {
        let result = ducklake_to_arrow_type("decimal(-1, 0)");
        assert!(result.is_err());
    }

    #[test]
    fn test_decimal_too_many_parameters_rejected() {
        let result = ducklake_to_arrow_type("decimal(1,2,3)");
        assert!(result.is_err());
        match result {
            Err(DuckLakeError::UnsupportedType(msg)) => {
                assert!(msg.contains("expected at most 2 parameters"));
                assert!(msg.contains("got 3"));
            },
            _ => panic!("Expected UnsupportedType error for 3 parameters"),
        }

        let result = ducklake_to_arrow_type("decimal(10,2,5,3)");
        assert!(result.is_err());
        match result {
            Err(DuckLakeError::UnsupportedType(msg)) => {
                assert!(msg.contains("expected at most 2 parameters"));
                assert!(msg.contains("got 4"));
            },
            _ => panic!("Expected UnsupportedType error for 4 parameters"),
        }
    }

    #[test]
    fn test_decimal_negative_scale_valid() {
        assert_eq!(
            ducklake_to_arrow_type("decimal(10, -2)").unwrap(),
            DataType::Decimal128(10, -2)
        );
    }

    #[test]
    fn test_build_schema_with_list_type() {
        let columns = vec![
            DuckLakeTableColumn::new(1, "id".to_string(), "int32".to_string(), true),
            DuckLakeTableColumn::new(2, "tags".to_string(), "list<varchar>".to_string(), true),
        ];

        let schema = build_arrow_schema(&columns).unwrap();
        assert_eq!(schema.fields().len(), 2);
        assert_eq!(
            *schema.field(1).data_type(),
            DataType::List(Arc::new(Field::new("item", DataType::Utf8View, true)))
        );
    }

    #[test]
    fn test_build_schema_with_struct_type() {
        let columns = vec![DuckLakeTableColumn::new(
            1,
            "data".to_string(),
            "struct<a:int32>".to_string(),
            true,
        )];

        let schema = build_arrow_schema(&columns).unwrap();
        assert_eq!(
            schema.field(0).data_type(),
            &DataType::Struct(vec![Field::new("a", DataType::Int32, true)].into())
        );
    }

    #[test]
    fn test_column_id_i32_max_succeeds() {
        let columns = vec![DuckLakeTableColumn::new(
            i32::MAX as i64,
            "id".to_string(),
            "int32".to_string(),
            true,
        )];

        let mut parquet_field_ids = HashMap::new();
        parquet_field_ids.insert(i32::MAX, "id".to_string());

        let result = build_read_schema_with_field_id_mapping(&columns, &parquet_field_ids, None);
        assert!(result.is_ok(), "column_id = i32::MAX should succeed");
    }

    #[test]
    fn test_column_id_overflow_returns_error() {
        let columns = vec![DuckLakeTableColumn::new(
            i32::MAX as i64 + 1, // 2147483648, exceeds i32 range
            "id".to_string(),
            "int32".to_string(),
            true,
        )];

        let parquet_field_ids = HashMap::new();

        let result = build_read_schema_with_field_id_mapping(&columns, &parquet_field_ids, None);
        assert!(result.is_err(), "column_id > i32::MAX should fail");
        match result {
            Err(DuckLakeError::Internal(msg)) => {
                assert!(
                    msg.contains("2147483648"),
                    "Error should contain the overflowing value: {}",
                    msg
                );
                assert!(
                    msg.contains("exceeds i32 range"),
                    "Error should explain the issue: {}",
                    msg
                );
            },
            _ => panic!("Expected Internal error for column_id overflow"),
        }
    }

    #[test]
    fn test_column_id_negative_within_i32_range_succeeds() {
        let columns =
            vec![DuckLakeTableColumn::new(-1, "id".to_string(), "int32".to_string(), true)];

        let mut parquet_field_ids = HashMap::new();
        parquet_field_ids.insert(-1_i32, "id".to_string());

        let result = build_read_schema_with_field_id_mapping(&columns, &parquet_field_ids, None);
        assert!(
            result.is_ok(),
            "Negative column_id within i32 range should succeed"
        );
    }

    // ── normalize_ducklake_type tests ──

    #[test]
    fn test_normalize_int_aliases() {
        assert_eq!(normalize_ducklake_type("int").unwrap(), "int32");
        assert_eq!(normalize_ducklake_type("integer").unwrap(), "int32");
        assert_eq!(normalize_ducklake_type("INT").unwrap(), "int32");
        assert_eq!(normalize_ducklake_type("Integer").unwrap(), "int32");
        assert_eq!(normalize_ducklake_type("int32").unwrap(), "int32");
    }

    #[test]
    fn test_normalize_bigint_aliases() {
        assert_eq!(normalize_ducklake_type("bigint").unwrap(), "int64");
        assert_eq!(normalize_ducklake_type("long").unwrap(), "int64");
        assert_eq!(normalize_ducklake_type("BIGINT").unwrap(), "int64");
        assert_eq!(normalize_ducklake_type("int64").unwrap(), "int64");
    }

    #[test]
    fn test_normalize_string_aliases() {
        assert_eq!(normalize_ducklake_type("text").unwrap(), "varchar");
        assert_eq!(normalize_ducklake_type("string").unwrap(), "varchar");
        assert_eq!(normalize_ducklake_type("varchar").unwrap(), "varchar");
        assert_eq!(normalize_ducklake_type("TEXT").unwrap(), "varchar");
        assert_eq!(normalize_ducklake_type("STRING").unwrap(), "varchar");
    }

    #[test]
    fn test_normalize_float_aliases() {
        assert_eq!(normalize_ducklake_type("float").unwrap(), "float32");
        assert_eq!(normalize_ducklake_type("real").unwrap(), "float32");
        assert_eq!(normalize_ducklake_type("FLOAT").unwrap(), "float32");
        assert_eq!(normalize_ducklake_type("float32").unwrap(), "float32");
    }

    #[test]
    fn test_normalize_double_aliases() {
        assert_eq!(normalize_ducklake_type("double").unwrap(), "float64");
        assert_eq!(normalize_ducklake_type("DOUBLE").unwrap(), "float64");
        assert_eq!(normalize_ducklake_type("float64").unwrap(), "float64");
    }

    #[test]
    fn test_normalize_bool_aliases() {
        assert_eq!(normalize_ducklake_type("bool").unwrap(), "boolean");
        assert_eq!(normalize_ducklake_type("boolean").unwrap(), "boolean");
        assert_eq!(normalize_ducklake_type("BOOLEAN").unwrap(), "boolean");
    }

    #[test]
    fn test_normalize_smallint_aliases() {
        assert_eq!(normalize_ducklake_type("smallint").unwrap(), "int16");
        assert_eq!(normalize_ducklake_type("SMALLINT").unwrap(), "int16");
        assert_eq!(normalize_ducklake_type("int16").unwrap(), "int16");
    }

    #[test]
    fn test_normalize_tinyint_aliases() {
        assert_eq!(normalize_ducklake_type("tinyint").unwrap(), "int8");
        assert_eq!(normalize_ducklake_type("TINYINT").unwrap(), "int8");
        assert_eq!(normalize_ducklake_type("int8").unwrap(), "int8");
    }

    #[test]
    fn test_normalize_unknown_type_errors() {
        assert!(normalize_ducklake_type("foobar").is_err());
    }

    // ── is_promotable tests ──

    #[test]
    fn test_promotable_same_type() {
        assert!(is_promotable("int32", "int32"));
        assert!(is_promotable("varchar", "varchar"));
        assert!(is_promotable("float64", "float64"));
    }

    #[test]
    fn test_promotable_signed_int_widening() {
        assert!(is_promotable("int8", "int16"));
        assert!(is_promotable("int8", "int32"));
        assert!(is_promotable("int8", "int64"));
        assert!(is_promotable("int16", "int32"));
        assert!(is_promotable("int16", "int64"));
        assert!(is_promotable("int32", "int64"));
    }

    #[test]
    fn test_promotable_signed_int_narrowing_rejected() {
        assert!(!is_promotable("int64", "int32"));
        assert!(!is_promotable("int32", "int16"));
        assert!(!is_promotable("int16", "int8"));
    }

    #[test]
    fn test_promotable_unsigned_int_widening() {
        assert!(is_promotable("uint8", "uint16"));
        assert!(is_promotable("uint8", "uint32"));
        assert!(is_promotable("uint8", "uint64"));
        assert!(is_promotable("uint16", "uint32"));
        assert!(is_promotable("uint32", "uint64"));
    }

    #[test]
    fn test_promotable_unsigned_narrowing_rejected() {
        assert!(!is_promotable("uint64", "uint32"));
        assert!(!is_promotable("uint32", "uint16"));
    }

    #[test]
    fn test_promotable_float_widening() {
        assert!(is_promotable("float32", "float64"));
    }

    #[test]
    fn test_promotable_float_narrowing_rejected() {
        assert!(!is_promotable("float64", "float32"));
    }

    #[test]
    fn test_promotable_int_to_float_excluded() {
        // int -> float is NOT in the conservative default set (design §6, review
        // #4): int64/uint64 -> float64 loses precision past 2^53, so the whole
        // int->float family is excluded until added as a justified per-width entry.
        assert!(!is_promotable("int8", "float64"));
        assert!(!is_promotable("int16", "float64"));
        assert!(!is_promotable("int32", "float64"));
        assert!(!is_promotable("int64", "float64"));
        assert!(!is_promotable("int32", "float32"));
    }

    #[test]
    fn test_promotable_timestamp_to_timestamptz_excluded() {
        // timestamp -> timestamptz is a semantic reinterpretation, not a pure
        // widen; excluded from the default set (both directions rejected).
        assert!(!is_promotable("timestamp", "timestamptz"));
        assert!(!is_promotable("timestamptz", "timestamp"));
    }

    #[test]
    fn test_promotable_decimal_excluded() {
        // Decimal precision/scale widening is excluded from the conservative
        // default set; it needs its own justified lossless entry + cast-on-read.
        assert!(!is_promotable("decimal(10, 2)", "decimal(18, 4)"));
        assert!(!is_promotable("decimal(10, 2)", "decimal(20, 2)"));
        assert!(!is_promotable("decimal(18, 4)", "decimal(10, 2)")); // narrowing also rejected
        // Same decimal type is still trivially "promotable" (a no-op).
        assert!(is_promotable("decimal(10, 2)", "decimal(10, 2)"));
    }

    #[test]
    fn test_promotable_incompatible_types() {
        assert!(!is_promotable("int32", "varchar"));
        assert!(!is_promotable("varchar", "int32"));
        assert!(!is_promotable("boolean", "int32"));
        assert!(!is_promotable("date", "timestamp"));
    }

    #[test]
    fn test_promotable_unknown_types() {
        assert!(!is_promotable("foobar", "int32"));
        assert!(!is_promotable("int32", "foobar"));
    }

    #[test]
    fn test_promotable_with_aliases() {
        // Uses normalized forms internally
        assert!(is_promotable("int", "bigint")); // int32 -> int64
        assert!(is_promotable("tinyint", "integer")); // int8 -> int32
        assert!(is_promotable("float", "double")); // float32 -> float64
    }

    // ── types_compatible tests ──

    #[test]
    fn test_types_compatible_same_canonical() {
        assert!(types_compatible("int", "int32"));
        assert!(types_compatible("int32", "int"));
        assert!(types_compatible("integer", "int"));
        assert!(types_compatible("text", "varchar"));
        assert!(types_compatible("string", "text"));
        assert!(types_compatible("bigint", "int64"));
        assert!(types_compatible("float", "real"));
        assert!(types_compatible("double", "float64"));
        assert!(types_compatible("bool", "boolean"));
    }

    #[test]
    fn test_types_compatible_case_insensitive() {
        assert!(types_compatible("INT", "int32"));
        assert!(types_compatible("VARCHAR", "text"));
        assert!(types_compatible("BIGINT", "int64"));
    }

    #[test]
    fn test_types_compatible_with_promotion() {
        assert!(types_compatible("int32", "int64"));
        assert!(types_compatible("float32", "float64"));
        // timestamp -> timestamptz is no longer in the conservative promote set
        // (design §6, review #4) — a semantic reinterpretation, not a pure widen.
        assert!(!types_compatible("timestamp", "timestamptz"));
    }

    #[test]
    fn test_types_equal_canonical() {
        // Alias-only differences are EQUAL — the §5 data-write "no-op" case
        // (a Replace/Append restating bigint as int64 must NOT be rejected).
        assert!(types_equal_canonical("int64", "bigint"));
        assert!(types_equal_canonical("bigint", "int64"));
        assert!(types_equal_canonical("int", "int32"));
        assert!(types_equal_canonical("text", "varchar"));
        assert!(types_equal_canonical("INT64", "int64")); // case-insensitive
        // A genuine widening is NOT canonical-equal — it must go through
        // promote_column_type, not a data write (unlike `types_compatible`).
        assert!(!types_equal_canonical("int32", "int64"));
        assert!(!types_equal_canonical("float32", "float64"));
        // Unrelated / unknown types differ.
        assert!(!types_equal_canonical("int32", "varchar"));
        assert!(!types_equal_canonical("foobar", "int32"));
    }

    #[test]
    fn test_types_compatible_narrowing_rejected() {
        assert!(!types_compatible("int64", "int32"));
        assert!(!types_compatible("float64", "float32"));
    }

    #[test]
    fn test_types_compatible_incompatible() {
        assert!(!types_compatible("int32", "varchar"));
        assert!(!types_compatible("varchar", "int32"));
        assert!(!types_compatible("boolean", "float64"));
    }

    #[test]
    fn test_types_compatible_unknown() {
        assert!(!types_compatible("foobar", "int32"));
        assert!(!types_compatible("int32", "foobar"));
        assert!(!types_compatible("foobar", "bazqux"));
    }

    /// A struct child the catalog records NON-nullable — what DDL that adds or
    /// renames a nested field writes — must read as nullable, so the read schema
    /// stays type-identical to the catalog schema the provider advertises. When
    /// they differ the scan has to cast, and casting a nullable physical child to
    /// a non-nullable logical one is refused outright.
    #[test]
    fn test_read_schema_relaxes_nested_nullability_like_catalog_schema() {
        let columns = vec![DuckLakeTableColumn {
            column_id: 1,
            column_name: "s".to_string(),
            column_type: "struct<a:int32,b:int32>".to_string(),
            is_nullable: true,
            data_type: Some(DataType::Struct(
                vec![
                    Arc::new(Field::new("a", DataType::Int32, true)),
                    Arc::new(Field::new("b", DataType::Int32, false)),
                ]
                .into(),
            )),
            nested_column_ids: vec![2, 3],
            initial_default: None,
            default_value: None,
            default_value_type: None,
            default_value_dialect: None,
        }];
        let parquet_field_ids =
            HashMap::from([(1, "s".to_string()), (2, "a".to_string()), (3, "b".to_string())]);

        let (read_schema, _) =
            build_read_schema_with_field_id_mapping(&columns, &parquet_field_ids, None).unwrap();
        let DataType::Struct(fields) = read_schema.field(0).data_type() else {
            panic!("s must remain a struct");
        };
        assert!(fields[1].is_nullable(), "child `b` must read as nullable");

        let catalog_schema = build_arrow_schema(&columns).unwrap();
        assert!(
            crate::column_rename::types_equal_ignoring_field_metadata(
                read_schema.field(0).data_type(),
                catalog_schema.field(0).data_type(),
            ),
            "read {:?} != catalog {:?}",
            read_schema.field(0).data_type(),
            catalog_schema.field(0).data_type(),
        );
    }

    /// A MAP's key is structurally non-null. The relaxation must not reach it:
    /// a nullable key makes the arrays a `MAP` column produces disagree with the
    /// schema they are read under.
    #[test]
    fn test_read_schema_keeps_map_key_non_nullable() {
        let entries = Field::new(
            "entries",
            DataType::Struct(
                vec![
                    Arc::new(Field::new("key", DataType::Utf8, false)),
                    Arc::new(Field::new("value", DataType::Int32, true)),
                ]
                .into(),
            ),
            false,
        );
        let columns = vec![DuckLakeTableColumn {
            column_id: 1,
            column_name: "m".to_string(),
            column_type: "map<varchar,int32>".to_string(),
            is_nullable: true,
            data_type: Some(DataType::Map(Arc::new(entries), false)),
            nested_column_ids: vec![2, 3],
            initial_default: None,
            default_value: None,
            default_value_type: None,
            default_value_dialect: None,
        }];
        let parquet_field_ids =
            HashMap::from([(1, "m".to_string()), (2, "key".to_string()), (3, "value".to_string())]);

        let (read_schema, _) =
            build_read_schema_with_field_id_mapping(&columns, &parquet_field_ids, None).unwrap();
        let DataType::Map(entries, _) = read_schema.field(0).data_type() else {
            panic!("m must remain a map");
        };
        let DataType::Struct(fields) = entries.data_type() else {
            panic!("map entries must be a struct");
        };
        assert!(!fields[0].is_nullable(), "map key must stay non-nullable");
    }
}
