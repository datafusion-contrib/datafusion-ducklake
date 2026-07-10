//! Type mapping from DuckLake types to Arrow types

use std::collections::HashMap;
use std::sync::Arc;

use crate::metadata_provider::DuckLakeTableColumn;
use crate::{DuckLakeError, Result};
use arrow::datatypes::{DataType, Field, IntervalUnit, Schema, TimeUnit};
use parquet::file::metadata::ParquetMetaData;

/// Convert a DuckLake type string to an Arrow DataType
pub fn ducklake_to_arrow_type(ducklake_type: &str) -> Result<DataType> {
    // Normalize type string (lowercase, remove whitespace)
    let normalized = ducklake_type.trim().to_lowercase();

    // Handle parameterized types first
    if let Some(decimal_params) = parse_decimal(&normalized)? {
        return Ok(decimal_params);
    }

    // Handle list/array types
    if let Some(list_type) = parse_list_type(&normalized)? {
        return Ok(list_type);
    }

    // Handle basic types
    match normalized.as_str() {
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

        _ => {
            // Check for complex types (struct, map)
            if normalized.starts_with("struct") {
                Err(DuckLakeError::UnsupportedType(format!(
                    "Struct type '{}' not yet supported. Please open an issue at https://github.com/hotdata-dev/datafusion-ducklake if you need this feature.",
                    ducklake_type
                )))
            } else if normalized.starts_with("map") {
                Err(DuckLakeError::UnsupportedType(format!(
                    "Map type '{}' not yet supported. Please open an issue at https://github.com/hotdata-dev/datafusion-ducklake if you need this feature.",
                    ducklake_type
                )))
            } else {
                Err(DuckLakeError::UnsupportedType(ducklake_type.to_string()))
            }
        },
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

        // List types
        DataType::List(field) | DataType::LargeList(field) => {
            let inner = arrow_to_ducklake_type(field.data_type())?;
            Ok(format!("list<{}>", inner))
        },
        DataType::FixedSizeList(field, _) => {
            let inner = arrow_to_ducklake_type(field.data_type())?;
            Ok(format!("list<{}>", inner))
        },
        DataType::Struct(_) => Err(DuckLakeError::UnsupportedType(format!(
            "Struct type '{}' not yet supported for writing",
            arrow_type
        ))),
        DataType::Map(_, _) => Err(DuckLakeError::UnsupportedType(format!(
            "Map type '{}' not yet supported for writing",
            arrow_type
        ))),

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

/// Parse decimal type with precision and scale
/// Format: "decimal(precision, scale)" or "decimal(precision)"
///
/// Returns `Ok(None)` if the type string is not a decimal type.
/// Returns `Err` if it is a decimal type but has invalid precision/scale.
fn parse_decimal(type_str: &str) -> Result<Option<DataType>> {
    if !type_str.starts_with("decimal") && !type_str.starts_with("numeric") {
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
            Ok(Some(DataType::Decimal128(precision, 0)))
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
            if precision > 38 {
                Ok(Some(DataType::Decimal256(precision, scale)))
            } else {
                Ok(Some(DataType::Decimal128(precision, scale)))
            }
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
/// Only simple (non-nested) element types are supported.
fn parse_list_type(type_str: &str) -> Result<Option<DataType>> {
    let inner = if type_str.starts_with("list<") || type_str.starts_with("array<") {
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

    // Only support simple (non-nested) element types
    if inner.contains('<') || inner.contains('[') || inner.contains('{') {
        return Err(DuckLakeError::UnsupportedType(format!(
            "Nested complex type '{}' not yet supported",
            type_str
        )));
    }

    let element_type = ducklake_to_arrow_type(inner)?;
    Ok(Some(DataType::List(Arc::new(Field::new(
        "item",
        element_type,
        true,
    )))))
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

/// Build an Arrow schema from a list of DuckLake table columns
pub fn build_arrow_schema(columns: &[DuckLakeTableColumn]) -> Result<Schema> {
    let fields: Result<Vec<Field>> = columns
        .iter()
        .map(|col| {
            let data_type = ducklake_to_arrow_type(&col.column_type)?;
            Ok(Field::new(&col.column_name, data_type, col.is_nullable))
        })
        .collect();

    Ok(Schema::new(fields?))
}

/// Extract field_id to column_name mapping from Parquet metadata.
/// DuckLake column_id == Parquet field_id, enabling column matching after renames.
pub fn extract_parquet_field_ids(metadata: &ParquetMetaData) -> HashMap<i32, String> {
    let schema_descr = metadata.file_metadata().schema_descr();
    let mut field_id_map = HashMap::new();

    // DuckLake assigns one field_id per *top-level* column (`column_id` == the
    // top-level field's field_id), so read ids off the top-level fields — the
    // root group's direct children — NOT the Parquet leaf columns.
    //
    // For a scalar the top-level field *is* the leaf, so both are equivalent. But
    // for a nested column (List / struct / map) the field_id lives on the group
    // node while the leaves carry none — e.g. a `List<Float32>` column `v` is
    // `v (group, field_id=2) -> list -> element (leaf, no id)`. Walking the leaves
    // (`num_columns()`) misses `v`'s id entirely, so the matcher treats the column
    // as absent and null-fills it. Walking top-level fields finds the id where it
    // actually sits.
    for field in schema_descr.root_schema().get_fields() {
        let basic_info = field.get_basic_info();
        if basic_info.has_id() {
            field_id_map.insert(basic_info.id(), field.name().to_string());
        }
    }

    field_id_map
}

/// Build a schema for reading Parquet files across schema evolution.
/// Returns (read_schema, name_mapping): read_schema uses each column's physical
/// name in the file, and name_mapping maps that physical name -> current name for
/// renamed columns. A current column whose field_id is absent from a file that
/// otherwise carries field_ids is read as an all-NULL column (the file predates
/// the column, or it was dropped then re-added under the same name).
pub fn build_read_schema_with_field_id_mapping(
    current_columns: &[DuckLakeTableColumn],
    parquet_field_ids: &HashMap<i32, String>,
    file_schema: Option<&Schema>,
) -> Result<(Schema, HashMap<String, String>)> {
    let mut name_mapping: HashMap<String, String> = HashMap::new();

    let fields: Result<Vec<Field>> = current_columns
        .iter()
        .map(|col| {
            let mut data_type = ducklake_to_arrow_type(&col.column_type)?;
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
                    (format!("__ducklake_absent_field_{}", field_id), true, true)
                };

            // For list/nested columns the Parquet reader reproduces the *file's*
            // child field name (e.g. "" for files our streaming writer produces
            // vs "item" for files written through the Arrow writer), and
            // DataFusion's scan validates each batch against the read schema. The
            // reconstructed name (always "item") may not match, so adopt the
            // file's actual type for these columns when the file schema is known.
            // Scalars keep the reconstructed type (precise per the catalog). An
            // absent column has a synthetic name not in the file, so skip it.
            if !is_absent
                && matches!(
                    data_type,
                    DataType::List(_) | DataType::LargeList(_) | DataType::FixedSizeList(_, _)
                )
                && let Some(fs) = file_schema
                && let Ok(file_field) = fs.field_with_name(&read_name)
            {
                data_type = file_field.data_type().clone();
            }

            if needs_rename {
                name_mapping.insert(read_name.clone(), col.column_name.clone());
            }

            // An absent column is materialised as a null array by the scan, so its
            // read field must be nullable; the catalog nullability is still
            // enforced on output by ColumnRenameExec.
            Ok(Field::new(
                read_name,
                data_type,
                col.is_nullable || is_absent,
            ))
        })
        .collect();

    Ok((Schema::new(fields?), name_mapping))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_read_schema_with_renamed_columns() {
        // Simulate: column was originally named "user_id", now renamed to "userId"
        let current_columns = vec![
            DuckLakeTableColumn {
                column_id: 1,
                column_name: "userId".to_string(), // Current name (renamed)
                column_type: "int32".to_string(),
                is_nullable: true,
            },
            DuckLakeTableColumn {
                column_id: 2,
                column_name: "name".to_string(), // Not renamed
                column_type: "varchar".to_string(),
                is_nullable: true,
            },
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
        let current_columns = vec![DuckLakeTableColumn {
            column_id: 1,
            column_name: "id".to_string(),
            column_type: "int32".to_string(),
            is_nullable: true,
        }];

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
        let current_columns = vec![DuckLakeTableColumn {
            column_id: 1,
            column_name: "id".to_string(),
            column_type: "int32".to_string(),
            is_nullable: true,
        }];

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
            DuckLakeTableColumn {
                column_id: 1,
                column_name: "id".to_string(),
                column_type: "int32".to_string(),
                is_nullable: true,
            },
            DuckLakeTableColumn {
                column_id: 2,
                column_name: "tag".to_string(),
                column_type: "varchar".to_string(),
                is_nullable: true,
            },
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
    fn test_unsupported_struct_type_errors() {
        // Test struct type returns error
        let result = ducklake_to_arrow_type("struct<a:int32,b:varchar>");
        assert!(result.is_err());
        match result {
            Err(DuckLakeError::UnsupportedType(msg)) => {
                assert!(msg.contains("struct<a:int32,b:varchar>"));
                assert!(msg.contains("not yet supported"));
                assert!(msg.contains("open an issue"));
            },
            _ => panic!("Expected UnsupportedType error for struct type"),
        }
    }

    #[test]
    fn test_unsupported_map_type_errors() {
        // Test map type returns error
        let result = ducklake_to_arrow_type("map<varchar,int32>");
        assert!(result.is_err());
        match result {
            Err(DuckLakeError::UnsupportedType(msg)) => {
                assert!(msg.contains("map<varchar,int32>"));
                assert!(msg.contains("not yet supported"));
                assert!(msg.contains("open an issue"));
            },
            _ => panic!("Expected UnsupportedType error for map type"),
        }
    }

    #[test]
    fn test_nested_complex_types_error() {
        // Nested complex types return error
        let result = ducklake_to_arrow_type("list<struct<a:int32,b:varchar>>");
        assert!(result.is_err());
        match result {
            Err(DuckLakeError::UnsupportedType(msg)) => {
                assert!(msg.contains("Nested complex type"));
                assert!(msg.contains("not yet supported"));
            },
            _ => panic!("Expected UnsupportedType error for nested complex type"),
        }

        // Nested list also errors
        assert!(ducklake_to_arrow_type("list<list<int32>>").is_err());
        assert!(ducklake_to_arrow_type("int32[][]").is_err());
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
    fn test_arrow_to_ducklake_unsupported_struct() {
        let struct_type = DataType::Struct(vec![Field::new("a", DataType::Int32, true)].into());
        let result = arrow_to_ducklake_type(&struct_type);
        assert!(result.is_err());
        match result {
            Err(DuckLakeError::UnsupportedType(msg)) => {
                assert!(msg.contains("Struct type"));
                assert!(msg.contains("not yet supported"));
            },
            _ => panic!("Expected UnsupportedType error"),
        }
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
            DuckLakeTableColumn {
                column_id: 1,
                column_name: "id".to_string(),
                column_type: "int32".to_string(),
                is_nullable: true,
            },
            DuckLakeTableColumn {
                column_id: 2,
                column_name: "tags".to_string(),
                column_type: "list<varchar>".to_string(),
                is_nullable: true,
            },
        ];

        let schema = build_arrow_schema(&columns).unwrap();
        assert_eq!(schema.fields().len(), 2);
        assert_eq!(
            *schema.field(1).data_type(),
            DataType::List(Arc::new(Field::new("item", DataType::Utf8View, true)))
        );
    }

    #[test]
    fn test_build_schema_with_unsupported_type() {
        // Test that build_arrow_schema propagates complex type errors
        let columns = vec![DuckLakeTableColumn {
            column_id: 1,
            column_name: "data".to_string(),
            column_type: "struct<a:int32>".to_string(),
            is_nullable: true,
        }];

        let result = build_arrow_schema(&columns);
        assert!(result.is_err());
    }

    #[test]
    fn test_column_id_i32_max_succeeds() {
        let columns = vec![DuckLakeTableColumn {
            column_id: i32::MAX as i64,
            column_name: "id".to_string(),
            column_type: "int32".to_string(),
            is_nullable: true,
        }];

        let mut parquet_field_ids = HashMap::new();
        parquet_field_ids.insert(i32::MAX, "id".to_string());

        let result = build_read_schema_with_field_id_mapping(&columns, &parquet_field_ids, None);
        assert!(result.is_ok(), "column_id = i32::MAX should succeed");
    }

    #[test]
    fn test_column_id_overflow_returns_error() {
        let columns = vec![DuckLakeTableColumn {
            column_id: i32::MAX as i64 + 1, // 2147483648, exceeds i32 range
            column_name: "id".to_string(),
            column_type: "int32".to_string(),
            is_nullable: true,
        }];

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
        let columns = vec![DuckLakeTableColumn {
            column_id: -1,
            column_name: "id".to_string(),
            column_type: "int32".to_string(),
            is_nullable: true,
        }];

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
}
