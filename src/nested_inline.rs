use std::sync::Arc;

#[cfg(feature = "write")]
use arrow::array::FixedSizeBinaryArray;
use arrow::array::{
    Array, ArrayRef, FixedSizeListArray, LargeListArray, ListArray, MapArray, StructArray,
    new_empty_array,
};
use arrow::buffer::{OffsetBuffer, ScalarBuffer};
use arrow::datatypes::{DataType, Field, Fields, IntervalUnit, TimeUnit};
use datafusion::common::ScalarValue;

#[cfg(feature = "write")]
use crate::{DuckLakeError, Result};

pub(crate) fn scalar_type_supports_inlining(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Null
            | DataType::Boolean
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float32
            | DataType::Float64
            | DataType::Decimal128(_, _)
            | DataType::Date32
            | DataType::Time64(TimeUnit::Microsecond)
            | DataType::Timestamp(_, _)
            | DataType::Interval(IntervalUnit::MonthDayNano)
            | DataType::Utf8
            | DataType::LargeUtf8
            | DataType::Utf8View
            | DataType::Binary
            | DataType::LargeBinary
            | DataType::BinaryView
            | DataType::FixedSizeBinary(_)
    )
}

pub(crate) fn type_supports_inlining(data_type: &DataType) -> bool {
    type_supports_inlining_at_depth(data_type, 0)
}

fn type_supports_inlining_at_depth(data_type: &DataType, depth: usize) -> bool {
    if depth > crate::types::MAX_NESTED_TYPE_DEPTH {
        return false;
    }

    match data_type {
        DataType::List(field) | DataType::LargeList(field) | DataType::FixedSizeList(field, _) => {
            type_supports_inlining_at_depth(field.data_type(), depth + 1)
        },
        DataType::Struct(fields) => fields
            .iter()
            .all(|field| type_supports_inlining_at_depth(field.data_type(), depth + 1)),
        DataType::Map(entries, _) => match entries.data_type() {
            DataType::Struct(fields) if fields.len() == 2 => fields
                .iter()
                .all(|field| type_supports_inlining_at_depth(field.data_type(), depth + 1)),
            _ => false,
        },
        _ => scalar_type_supports_inlining(data_type),
    }
}

pub(crate) fn duckdb_type_name(data_type: &DataType) -> Option<String> {
    duckdb_type_name_at_depth(data_type, 0)
}

fn duckdb_type_name_at_depth(data_type: &DataType, depth: usize) -> Option<String> {
    if depth > crate::types::MAX_NESTED_TYPE_DEPTH {
        return None;
    }

    let scalar = match data_type {
        DataType::Null => "VARCHAR",
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
        DataType::Date32 => "DATE",
        DataType::Time64(TimeUnit::Microsecond) => "TIME",
        DataType::Timestamp(TimeUnit::Second, _) => "TIMESTAMP_S",
        DataType::Timestamp(TimeUnit::Millisecond, _) => "TIMESTAMP_MS",
        DataType::Timestamp(TimeUnit::Microsecond, _) => "TIMESTAMP",
        DataType::Timestamp(TimeUnit::Nanosecond, _) => "TIMESTAMP_NS",
        DataType::Interval(IntervalUnit::MonthDayNano) => "INTERVAL",
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => "VARCHAR",
        DataType::Binary | DataType::LargeBinary | DataType::BinaryView => "BLOB",
        DataType::FixedSizeBinary(_) => "BLOB",
        DataType::Decimal128(precision, scale) => {
            return Some(format!("DECIMAL({precision}, {scale})"));
        },
        DataType::List(field) | DataType::LargeList(field) | DataType::FixedSizeList(field, _) => {
            return Some(format!(
                "{}[]",
                duckdb_type_name_at_depth(field.data_type(), depth + 1)?
            ));
        },
        DataType::Struct(fields) => {
            let children = fields
                .iter()
                .map(|field| {
                    Some(format!(
                        "{} {}",
                        quote_ident(field.name()),
                        duckdb_type_name_at_depth(field.data_type(), depth + 1)?
                    ))
                })
                .collect::<Option<Vec<_>>>()?;
            return Some(format!("STRUCT({})", children.join(", ")));
        },
        DataType::Map(entries, _) => {
            let DataType::Struct(fields) = entries.data_type() else {
                return None;
            };
            if fields.len() != 2 {
                return None;
            }
            return Some(format!(
                "MAP({}, {})",
                duckdb_type_name_at_depth(fields[0].data_type(), depth + 1)?,
                duckdb_type_name_at_depth(fields[1].data_type(), depth + 1)?
            ));
        },
        _ => return None,
    };
    Some(scalar.to_string())
}

fn quote_ident(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

#[cfg(feature = "write")]
pub(crate) fn values_support_inlining(array: &dyn Array) -> bool {
    if let Some(intervals) = array
        .as_any()
        .downcast_ref::<arrow::array::IntervalMonthDayNanoArray>()
    {
        return intervals
            .iter()
            .flatten()
            .all(|value| value.nanoseconds % 1000 == 0);
    }
    array
        .to_data()
        .child_data()
        .iter()
        .all(|child| values_support_inlining(arrow::array::make_array(child.clone()).as_ref()))
}

#[cfg(feature = "write")]
pub(crate) fn render_text(array: &dyn Array, row: usize) -> Result<String> {
    render_text_at_depth(array, row, 0, false)
}

#[cfg(feature = "write")]
fn render_text_at_depth(
    array: &dyn Array,
    row: usize,
    depth: usize,
    map_key: bool,
) -> Result<String> {
    if depth > crate::types::MAX_NESTED_TYPE_DEPTH {
        return Err(DuckLakeError::UnsupportedType(format!(
            "Nested inline value exceeds maximum depth {}",
            crate::types::MAX_NESTED_TYPE_DEPTH
        )));
    }
    if array.is_null(row) {
        return Ok("NULL".to_string());
    }

    match array.data_type() {
        DataType::List(_) => {
            let array = downcast::<ListArray>(array)?;
            let offsets = array.value_offsets();
            render_list(
                array.values().as_ref(),
                offsets[row] as usize,
                offsets[row + 1] as usize,
                depth,
            )
        },
        DataType::LargeList(_) => {
            let array = downcast::<LargeListArray>(array)?;
            let offsets = array.value_offsets();
            render_list(
                array.values().as_ref(),
                offsets[row] as usize,
                offsets[row + 1] as usize,
                depth,
            )
        },
        DataType::FixedSizeList(_, size) => {
            let array = downcast::<FixedSizeListArray>(array)?;
            let start = row
                * usize::try_from(*size).map_err(|e| {
                    DuckLakeError::UnsupportedType(format!(
                        "Invalid fixed-size list length {size}: {e}"
                    ))
                })?;
            render_list(
                array.values().as_ref(),
                start,
                start + *size as usize,
                depth,
            )
        },
        DataType::Struct(fields) => {
            let array = downcast::<StructArray>(array)?;
            let values = fields
                .iter()
                .enumerate()
                .map(|(index, field)| {
                    Ok(format!(
                        "{}: {}",
                        quote_text(field.name()),
                        render_text_at_depth(array.column(index).as_ref(), row, depth + 1, false)?
                    ))
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(format!("{{{}}}", values.join(", ")))
        },
        DataType::Map(_, _) => {
            let array = downcast::<MapArray>(array)?;
            let offsets = array.value_offsets();
            let start = offsets[row] as usize;
            let end = offsets[row + 1] as usize;
            let values = (start..end)
                .map(|index| {
                    Ok(format!(
                        "{}={}",
                        render_text_at_depth(array.keys().as_ref(), index, depth + 1, true)?,
                        render_text_at_depth(array.values().as_ref(), index, depth + 1, false)?
                    ))
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(format!("{{{}}}", values.join(", ")))
        },
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => {
            let value = crate::metadata_writer::display_value(array, row)?;
            if map_key && is_bare_map_key(&value) {
                Ok(value)
            } else {
                Ok(quote_text(&value))
            }
        },
        DataType::Binary
        | DataType::LargeBinary
        | DataType::BinaryView
        | DataType::FixedSizeBinary(_) => Ok(quote_text(&encode_blob(binary_value(array, row)?))),
        DataType::Date32
        | DataType::Timestamp(_, _)
        | DataType::Time32(_)
        | DataType::Time64(_) => crate::metadata_writer::inlined_text_value(array, row),
        _ => crate::metadata_writer::display_value(array, row),
    }
}

#[cfg(feature = "write")]
fn render_list(values: &dyn Array, start: usize, end: usize, depth: usize) -> Result<String> {
    let values = (start..end)
        .map(|index| render_text_at_depth(values, index, depth + 1, false))
        .collect::<Result<Vec<_>>>()?;
    Ok(format!("[{}]", values.join(", ")))
}

#[cfg(feature = "write")]
fn downcast<T: 'static>(array: &dyn Array) -> Result<&T> {
    array.as_any().downcast_ref::<T>().ok_or_else(|| {
        DuckLakeError::Internal(format!(
            "Arrow data type {:?} and array implementation disagree",
            array.data_type()
        ))
    })
}

#[cfg(feature = "write")]
fn binary_value(array: &dyn Array, row: usize) -> Result<&[u8]> {
    match array.data_type() {
        DataType::Binary => Ok(downcast::<arrow::array::BinaryArray>(array)?.value(row)),
        DataType::LargeBinary => Ok(downcast::<arrow::array::LargeBinaryArray>(array)?.value(row)),
        DataType::BinaryView => Ok(downcast::<arrow::array::BinaryViewArray>(array)?.value(row)),
        DataType::FixedSizeBinary(_) => Ok(downcast::<FixedSizeBinaryArray>(array)?.value(row)),
        data_type => Err(DuckLakeError::Internal(format!(
            "Expected binary nested inline leaf, found {data_type:?}"
        ))),
    }
}

#[cfg(feature = "write")]
fn encode_blob(value: &[u8]) -> String {
    let mut encoded = String::new();
    for byte in value {
        // Escape every byte to protect nested delimiters and the NULL token
        encoded.push_str(&format!("\\x{byte:02X}"));
    }
    encoded
}

fn quote_text(value: &str) -> String {
    let mut quoted = String::with_capacity(value.len() + 2);
    quoted.push('\'');
    for character in value.chars() {
        if matches!(character, '\\' | '\'') {
            quoted.push('\\');
        }
        quoted.push(character);
    }
    quoted.push('\'');
    quoted
}

fn is_bare_map_key(value: &str) -> bool {
    !value.is_empty()
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
}

pub(crate) fn parse_text(value: &str, data_type: &DataType) -> Option<ScalarValue> {
    let mut parser = Parser::new(value);
    let scalar = parser.parse_value(data_type, 0, &[])?;
    parser.skip_whitespace();
    (parser.position == parser.input.len()).then_some(scalar)
}

struct Parser<'a> {
    input: &'a [u8],
    position: usize,
}

impl<'a> Parser<'a> {
    fn new(input: &'a str) -> Self {
        Self {
            input: input.as_bytes(),
            position: 0,
        }
    }

    fn parse_value(
        &mut self,
        data_type: &DataType,
        depth: usize,
        terminators: &[u8],
    ) -> Option<ScalarValue> {
        if depth > crate::types::MAX_NESTED_TYPE_DEPTH {
            return None;
        }
        self.skip_whitespace();
        if self.consume_null() {
            return ScalarValue::try_from(data_type).ok();
        }

        match data_type {
            data_type @ (DataType::List(_)
            | DataType::LargeList(_)
            | DataType::FixedSizeList(_, _)) => self.parse_list(data_type, depth),
            DataType::Struct(fields) => self.parse_struct(fields, depth),
            DataType::Map(entries, sorted) => self.parse_map(entries, *sorted, depth),
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
                if self.peek() == Some(b'\'') =>
            {
                let value = self.parse_quoted()?;
                match data_type {
                    DataType::Utf8 => Some(ScalarValue::Utf8(Some(value))),
                    DataType::LargeUtf8 => Some(ScalarValue::LargeUtf8(Some(value))),
                    DataType::Utf8View => Some(ScalarValue::Utf8View(Some(value))),
                    _ => unreachable!(),
                }
            },
            DataType::Binary
            | DataType::LargeBinary
            | DataType::BinaryView
            | DataType::FixedSizeBinary(_) => {
                let value = if self.peek() == Some(b'\'') {
                    self.parse_quoted()?
                } else {
                    self.take_until(terminators).trim().to_string()
                };
                let scalar =
                    crate::types::parse_ducklake_default_scalar(&value, &DataType::Binary)?;
                scalar.cast_to(data_type).ok()
            },
            _ => {
                let token = self.take_until(terminators).trim();
                crate::types::parse_ducklake_scalar_leaf(token, data_type)
            },
        }
    }

    fn parse_list(&mut self, data_type: &DataType, depth: usize) -> Option<ScalarValue> {
        let field = match data_type {
            DataType::List(field)
            | DataType::LargeList(field)
            | DataType::FixedSizeList(field, _) => field,
            _ => return None,
        };
        self.expect(b'[')?;
        let mut values = Vec::new();
        self.skip_whitespace();
        if !self.consume(b']') {
            loop {
                values.push(self.parse_value(field.data_type(), depth + 1, b",]")?);
                self.skip_whitespace();
                if self.consume(b']') {
                    break;
                }
                self.expect(b',')?;
            }
        }
        build_list_scalar(data_type, values)
    }

    /// A struct's text, `{'name': value, ...}`, holding each of `fields` at
    /// most once. Fields are matched by name, in whatever order the text lists
    /// them, and one the text leaves out is NULL, as DuckDB's cast from text to
    /// a struct reads it: a catalog whose struct children an earlier writer
    /// renumbered lists them in another order than the text was written in.
    fn parse_struct(&mut self, fields: &Fields, depth: usize) -> Option<ScalarValue> {
        self.expect(b'{')?;
        let mut values: Vec<Option<ScalarValue>> = vec![None; fields.len()];
        self.skip_whitespace();
        if !self.consume(b'}') {
            loop {
                self.skip_whitespace();
                let name = self.parse_quoted()?;
                let index = fields.iter().position(|field| field.name() == &name)?;
                // A field named twice takes its last value, as DuckDB's cast
                // from text takes it.
                self.expect(b':')?;
                values[index] =
                    Some(self.parse_value(fields[index].data_type(), depth + 1, b",}")?);
                self.skip_whitespace();
                if self.consume(b'}') {
                    break;
                }
                self.expect(b',')?;
            }
        }
        let values = values
            .into_iter()
            .zip(fields.iter())
            .map(|(value, field)| match value {
                Some(value) => Some(value),
                None => ScalarValue::try_from(field.data_type()).ok(),
            })
            .collect::<Option<Vec<_>>>()?;
        build_struct_scalar(&DataType::Struct(fields.clone()), values)
    }

    fn parse_map(
        &mut self,
        entries: &Arc<Field>,
        sorted: bool,
        depth: usize,
    ) -> Option<ScalarValue> {
        let DataType::Struct(fields) = entries.data_type() else {
            return None;
        };
        if fields.len() != 2 {
            return None;
        }
        self.expect(b'{')?;
        let mut keys = Vec::new();
        let mut values = Vec::new();
        self.skip_whitespace();
        if !self.consume(b'}') {
            loop {
                let key = self.parse_value(fields[0].data_type(), depth + 1, b"=")?;
                if key.is_null() {
                    return None;
                }
                self.expect(b'=')?;
                keys.push(key);
                values.push(self.parse_value(fields[1].data_type(), depth + 1, b",}")?);
                self.skip_whitespace();
                if self.consume(b'}') {
                    break;
                }
                self.expect(b',')?;
            }
        }
        build_map_scalar(&DataType::Map(Arc::clone(entries), sorted), keys, values)
    }

    fn parse_quoted(&mut self) -> Option<String> {
        self.expect(b'\'')?;
        let mut value = Vec::new();
        while let Some(byte) = self.peek() {
            self.position += 1;
            match byte {
                b'\\' => {
                    let escaped = self.peek()?;
                    self.position += 1;
                    value.push(escaped);
                },
                b'\'' if self.peek() == Some(b'\'') => {
                    self.position += 1;
                    value.push(b'\'');
                },
                b'\'' => return String::from_utf8(value).ok(),
                byte => value.push(byte),
            }
        }
        None
    }

    fn consume_null(&mut self) -> bool {
        let remaining = &self.input[self.position..];
        if remaining.len() < 4 || !remaining[..4].eq_ignore_ascii_case(b"NULL") {
            return false;
        }
        if remaining
            .get(4)
            .is_some_and(|byte| byte.is_ascii_alphanumeric() || *byte == b'_')
        {
            return false;
        }
        self.position += 4;
        true
    }

    fn take_until(&mut self, terminators: &[u8]) -> &str {
        let start = self.position;
        while self.peek().is_some_and(|byte| !terminators.contains(&byte)) {
            self.position += 1;
        }
        std::str::from_utf8(&self.input[start..self.position]).unwrap_or_default()
    }

    fn skip_whitespace(&mut self) {
        while self.peek().is_some_and(|byte| byte.is_ascii_whitespace()) {
            self.position += 1;
        }
    }

    fn expect(&mut self, expected: u8) -> Option<()> {
        self.skip_whitespace();
        self.consume(expected).then_some(())
    }

    fn consume(&mut self, expected: u8) -> bool {
        if self.peek() == Some(expected) {
            self.position += 1;
            true
        } else {
            false
        }
    }

    fn peek(&self) -> Option<u8> {
        self.input.get(self.position).copied()
    }
}

fn scalars_to_array(values: Vec<ScalarValue>, data_type: &DataType) -> Option<ArrayRef> {
    if values.is_empty() {
        Some(new_empty_array(data_type))
    } else {
        ScalarValue::iter_to_array(values).ok()
    }
}

pub(crate) fn build_list_scalar(
    data_type: &DataType,
    values: Vec<ScalarValue>,
) -> Option<ScalarValue> {
    match data_type {
        DataType::List(field) => {
            let values = scalars_to_array(values, field.data_type())?;
            let length = i32::try_from(values.len()).ok()?;
            Some(ScalarValue::List(Arc::new(
                ListArray::try_new(
                    Arc::clone(field),
                    OffsetBuffer::new(ScalarBuffer::from(vec![0_i32, length])),
                    values,
                    None,
                )
                .ok()?,
            )))
        },
        DataType::LargeList(field) => {
            let values = scalars_to_array(values, field.data_type())?;
            Some(ScalarValue::LargeList(Arc::new(
                LargeListArray::try_new(
                    Arc::clone(field),
                    OffsetBuffer::new(ScalarBuffer::from(vec![0_i64, values.len() as i64])),
                    values,
                    None,
                )
                .ok()?,
            )))
        },
        DataType::FixedSizeList(field, size) => {
            if values.len() != usize::try_from(*size).ok()? {
                return None;
            }
            let values = scalars_to_array(values, field.data_type())?;
            Some(ScalarValue::FixedSizeList(Arc::new(
                FixedSizeListArray::try_new(Arc::clone(field), *size, values, None).ok()?,
            )))
        },
        _ => None,
    }
}

pub(crate) fn build_struct_scalar(
    data_type: &DataType,
    values: Vec<ScalarValue>,
) -> Option<ScalarValue> {
    let DataType::Struct(fields) = data_type else {
        return None;
    };
    if fields.len() != values.len() {
        return None;
    }
    let arrays = values
        .into_iter()
        .map(|value| value.to_array().ok())
        .collect::<Option<Vec<ArrayRef>>>()?;
    Some(ScalarValue::Struct(Arc::new(
        StructArray::try_new(fields.clone(), arrays, None).ok()?,
    )))
}

pub(crate) fn build_map_scalar(
    data_type: &DataType,
    keys: Vec<ScalarValue>,
    values: Vec<ScalarValue>,
) -> Option<ScalarValue> {
    let DataType::Map(entries, sorted) = data_type else {
        return None;
    };
    let DataType::Struct(fields) = entries.data_type() else {
        return None;
    };
    if fields.len() != 2 || keys.len() != values.len() || keys.iter().any(ScalarValue::is_null) {
        return None;
    }
    let keys = scalars_to_array(keys, fields[0].data_type())?;
    let values = scalars_to_array(values, fields[1].data_type())?;
    let length = i32::try_from(keys.len()).ok()?;
    let entries_array = StructArray::try_new(fields.clone(), vec![keys, values], None).ok()?;
    Some(ScalarValue::Map(Arc::new(
        MapArray::try_new(
            Arc::clone(entries),
            OffsetBuffer::new(ScalarBuffer::from(vec![0_i32, length])),
            entries_array,
            None,
            *sorted,
        )
        .ok()?,
    )))
}

/// One node below a nested column: its catalog column id and its own child
/// nodes, in the order its Arrow type lists them.
#[derive(Debug, PartialEq, Eq)]
struct NestedNode {
    id: i64,
    children: Vec<NestedNode>,
}

/// The child nodes of `data_type`, taking their ids from `ids` in the
/// pre-order `DuckLakeTableColumn::nested_column_ids` lists them: a struct's
/// fields, a list's element, a map's key then value. `None` when `ids` runs out.
fn nested_nodes(
    data_type: &DataType,
    ids: &mut std::slice::Iter<'_, i64>,
) -> Option<Vec<NestedNode>> {
    let child_types: Vec<&DataType> = match data_type {
        DataType::List(field) | DataType::LargeList(field) | DataType::FixedSizeList(field, _) => {
            vec![field.data_type()]
        },
        DataType::Struct(fields) => fields.iter().map(|field| field.data_type()).collect(),
        DataType::Map(entries, _) => match entries.data_type() {
            DataType::Struct(fields) => fields.iter().map(|field| field.data_type()).collect(),
            _ => return None,
        },
        _ => Vec::new(),
    };
    child_types
        .into_iter()
        .map(|child| {
            let id = *ids.next()?;
            Some(NestedNode {
                id,
                children: nested_nodes(child, ids)?,
            })
        })
        .collect()
}

/// The nodes of a nested column of type `data_type` whose pre-order node ids
/// are `ids`; an error when the two disagree.
fn column_nodes(data_type: &DataType, ids: &[i64]) -> crate::Result<Vec<NestedNode>> {
    let mut iter = ids.iter();
    match nested_nodes(data_type, &mut iter) {
        Some(nodes) if iter.next().is_none() => Ok(nodes),
        _ => Err(crate::DuckLakeError::InvalidConfig(format!(
            "nested column ids {ids:?} do not match the shape of {data_type}"
        ))),
    }
}

/// Whether a nested value of type `read` with node ids `read_ids` already has
/// the shape of `target` with `target_ids`: the same nodes, names and leaf types,
/// whatever each field's nullability.
pub(crate) fn same_nested_shape(
    read: &DataType,
    read_ids: &[i64],
    target: &DataType,
    target_ids: &[i64],
) -> bool {
    fn nullable(data_type: &DataType) -> DataType {
        let field = |field: &Arc<Field>| {
            Arc::new(
                field
                    .as_ref()
                    .clone()
                    .with_data_type(nullable(field.data_type()))
                    .with_nullable(true),
            )
        };
        match data_type {
            DataType::List(child) => DataType::List(field(child)),
            DataType::LargeList(child) => DataType::LargeList(field(child)),
            DataType::FixedSizeList(child, size) => DataType::FixedSizeList(field(child), *size),
            DataType::Struct(fields) => DataType::Struct(fields.iter().map(field).collect()),
            DataType::Map(entries, sorted) => DataType::Map(
                Arc::new(
                    entries
                        .as_ref()
                        .clone()
                        .with_data_type(nullable(entries.data_type())),
                ),
                *sorted,
            ),
            other => other.clone(),
        }
    }
    read_ids == target_ids && nullable(read) == nullable(target)
}

/// `array`, a nested column of type `read` whose nodes carry the column ids
/// `read_ids`, rebuilt as `target` with node ids `target_ids`.
///
/// Every node is matched by column id, never by name or position, so a field
/// renamed since `array` was written keeps its values, one dropped is left out,
/// and one `array` predates takes its `initial_default` from
/// `initial_defaults` (by column id), or NULL without one. A leaf whose type
/// was widened is cast.
pub(crate) fn adapt_nested_array(
    array: &ArrayRef,
    read: &DataType,
    read_ids: &[i64],
    target: &DataType,
    target_ids: &[i64],
    initial_defaults: &std::collections::HashMap<i64, String>,
) -> crate::Result<ArrayRef> {
    let read_nodes = column_nodes(read, read_ids)?;
    let target_nodes = column_nodes(target, target_ids)?;
    adapt_nodes(
        array,
        read,
        &read_nodes,
        target,
        &target_nodes,
        initial_defaults,
    )
}

fn adapt_nodes(
    array: &ArrayRef,
    read: &DataType,
    read_nodes: &[NestedNode],
    target: &DataType,
    target_nodes: &[NestedNode],
    initial_defaults: &std::collections::HashMap<i64, String>,
) -> crate::Result<ArrayRef> {
    use arrow::array::AsArray;

    if read == target && read_nodes == target_nodes {
        return Ok(Arc::clone(array));
    }
    Ok(match (read, target) {
        (DataType::Struct(read_fields), DataType::Struct(target_fields)) => {
            let read_array = array.as_struct();
            let children = target_fields
                .iter()
                .zip(target_nodes)
                .map(|(target_field, target_node)| {
                    match read_nodes.iter().position(|node| node.id == target_node.id) {
                        Some(index) => adapt_nodes(
                            read_array.column(index),
                            read_fields[index].data_type(),
                            &read_nodes[index].children,
                            target_field.data_type(),
                            &target_node.children,
                            initial_defaults,
                        ),
                        None => absent_field(
                            target_field.data_type(),
                            initial_defaults.get(&target_node.id),
                            array.len(),
                        ),
                    }
                })
                .collect::<crate::Result<Vec<_>>>()?;
            Arc::new(StructArray::try_new(
                target_fields.clone(),
                children,
                read_array.nulls().cloned(),
            )?)
        },
        (DataType::List(read_field), DataType::List(target_field)) => {
            let read_array = array.as_list::<i32>();
            let values = adapt_nodes(
                read_array.values(),
                read_field.data_type(),
                only(read_nodes)?,
                target_field.data_type(),
                only(target_nodes)?,
                initial_defaults,
            )?;
            Arc::new(ListArray::try_new(
                Arc::clone(target_field),
                read_array.offsets().clone(),
                values,
                read_array.nulls().cloned(),
            )?)
        },
        (DataType::LargeList(read_field), DataType::LargeList(target_field)) => {
            let read_array = array.as_list::<i64>();
            let values = adapt_nodes(
                read_array.values(),
                read_field.data_type(),
                only(read_nodes)?,
                target_field.data_type(),
                only(target_nodes)?,
                initial_defaults,
            )?;
            Arc::new(LargeListArray::try_new(
                Arc::clone(target_field),
                read_array.offsets().clone(),
                values,
                read_array.nulls().cloned(),
            )?)
        },
        (
            DataType::FixedSizeList(read_field, read_size),
            DataType::FixedSizeList(target_field, target_size),
        ) if read_size == target_size => {
            let read_array = array.as_fixed_size_list();
            let values = adapt_nodes(
                read_array.values(),
                read_field.data_type(),
                only(read_nodes)?,
                target_field.data_type(),
                only(target_nodes)?,
                initial_defaults,
            )?;
            Arc::new(FixedSizeListArray::try_new(
                Arc::clone(target_field),
                *target_size,
                values,
                read_array.nulls().cloned(),
            )?)
        },
        (DataType::Map(read_entries, _), DataType::Map(target_entries, sorted)) => {
            let read_array = array.as_map();
            let entries: ArrayRef = Arc::new(read_array.entries().clone());
            let entries = adapt_nodes(
                &entries,
                read_entries.data_type(),
                read_nodes,
                target_entries.data_type(),
                target_nodes,
                initial_defaults,
            )?;
            Arc::new(MapArray::try_new(
                Arc::clone(target_entries),
                read_array.offsets().clone(),
                entries.as_struct().clone(),
                read_array.nulls().cloned(),
                *sorted,
            )?)
        },
        (read, target) if read.is_nested() || target.is_nested() => {
            return Err(crate::DuckLakeError::Unsupported(format!(
                "inlined nested value of type {read} cannot be read as {target}"
            )));
        },
        (_, target) => arrow::compute::cast(array, target)?,
    })
}

/// The child nodes of a list's single element node.
fn only(nodes: &[NestedNode]) -> crate::Result<&[NestedNode]> {
    match nodes {
        [node] => Ok(node.children.as_slice()),
        _ => Err(crate::DuckLakeError::Internal(
            "a list has exactly one element node".to_string(),
        )),
    }
}

/// `len` values of a field the value predates: its `initial_default`, or NULL.
fn absent_field(
    data_type: &DataType,
    initial_default: Option<&String>,
    len: usize,
) -> crate::Result<ArrayRef> {
    match initial_default.filter(|value| !value.eq_ignore_ascii_case("NULL")) {
        Some(value) => crate::types::parse_ducklake_default_scalar(value, data_type)
            .ok_or_else(|| {
                crate::DuckLakeError::InvalidConfig(format!(
                    "Cannot decode initial_default '{value}' of a nested field as {data_type}"
                ))
            })?
            .to_array_of_size(len)
            .map_err(Into::into),
        None => Ok(arrow::array::new_null_array(data_type, len)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(all(feature = "write", feature = "metadata-duckdb"))]
    use arrow::array::BinaryArray;
    #[cfg(feature = "write")]
    use arrow::array::{Int32Array, StringArray};

    #[test]
    fn reference_nested_literals_parse() {
        let list_type = DataType::List(Arc::new(Field::new("item", DataType::Int32, true)));
        assert_eq!(
            parse_text("[1, NULL, 3]", &list_type).unwrap().to_string(),
            "[1, , 3]"
        );
        assert_eq!(parse_text("[]", &list_type).unwrap().to_string(), "[]");

        let fields = Fields::from(vec![
            Field::new("plain", DataType::Int32, true),
            Field::new("q\"uote", DataType::Utf8, true),
        ]);
        let struct_type = DataType::Struct(fields);
        let scalar = parse_text("{'plain': 7, 'q\"uote': 'a,b\\'c'}", &struct_type).unwrap();
        assert!(!scalar.is_null());
    }

    /// A struct's text names its fields in any order and may leave one out,
    /// which reads as NULL, or name one twice, taking the last value, as
    /// DuckDB's cast from text to a struct reads it.
    #[test]
    fn struct_text_matches_fields_by_name() {
        let struct_type = DataType::Struct(Fields::from(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Int32, true),
        ]));
        assert_eq!(
            parse_text("{'b': 1, 'a': 2}", &struct_type)
                .unwrap()
                .to_string(),
            "{a:2,b:1}"
        );
        assert_eq!(
            parse_text("{'a': 1}", &struct_type).unwrap().to_string(),
            "{a:1,b:}"
        );
        assert!(parse_text("{'z': 1}", &struct_type).is_none());
        // `'{''a'': 1, ''b'': 3, ''a'': 2}'::STRUCT(a INT, b INT)` is
        // `{'a': 2, 'b': 3}` in DuckDB 1.5.5.
        assert_eq!(
            parse_text("{'a': 1, 'b': 3, 'a': 2}", &struct_type)
                .unwrap()
                .to_string(),
            "{a:2,b:3}"
        );
        assert_eq!(
            parse_text("{'a': 1, 'a': NULL}", &struct_type)
                .unwrap()
                .to_string(),
            "{a:,b:}"
        );
    }

    #[cfg(feature = "write")]
    #[test]
    fn renderer_matches_reference_list_and_struct_literals() {
        let values = Int32Array::from(vec![Some(1), None, Some(3)]);
        let offsets = OffsetBuffer::new(ScalarBuffer::from(vec![0_i32, 3]));
        let list = ListArray::new(
            Arc::new(Field::new("item", DataType::Int32, true)),
            offsets,
            Arc::new(values),
            None,
        );
        assert_eq!(render_text(&list, 0).unwrap(), "[1, NULL, 3]");

        let fields = Fields::from(vec![
            Field::new("plain", DataType::Int32, true),
            Field::new("q\"uote", DataType::Utf8, true),
        ]);
        let array = StructArray::new(
            fields,
            vec![
                Arc::new(Int32Array::from(vec![7])) as ArrayRef,
                Arc::new(StringArray::from(vec!["café,雪'c"])) as ArrayRef,
            ],
            None,
        );
        let encoded = render_text(&array, 0).unwrap();
        assert_eq!(encoded, "{'plain': 7, 'q\"uote': 'café,雪\\'c'}");
        assert_eq!(
            parse_text(&encoded, array.data_type()).unwrap(),
            ScalarValue::try_from_array(&array, 0).unwrap()
        );
    }

    #[cfg(feature = "write")]
    #[test]
    fn map_reference_literals_round_trip() {
        let fields = Fields::from(vec![
            Field::new("key", DataType::Utf8, false),
            Field::new("value", DataType::Int32, true),
        ]);
        let entries = Arc::new(Field::new(
            "entries",
            DataType::Struct(fields.clone()),
            false,
        ));
        let array = MapArray::new(
            Arc::clone(&entries),
            OffsetBuffer::new(ScalarBuffer::from(vec![0_i32, 2, 2, 2])),
            StructArray::new(
                fields,
                vec![
                    Arc::new(StringArray::from(vec!["a,b", "q'x"])) as ArrayRef,
                    Arc::new(Int32Array::from(vec![Some(10), None])) as ArrayRef,
                ],
                None,
            ),
            Some(arrow::buffer::NullBuffer::from(vec![true, true, false])),
            false,
        );
        let data_type = array.data_type().clone();
        let expected = ["{'a,b'=10, 'q\\'x'=NULL}", "{}", "NULL"];
        for (row, expected) in expected.into_iter().enumerate() {
            let encoded = render_text(&array, row).unwrap();
            assert_eq!(encoded, expected);
            assert_eq!(
                parse_text(&encoded, &data_type).unwrap(),
                ScalarValue::try_from_array(&array, row).unwrap()
            );
        }
    }
    #[cfg(all(feature = "write", feature = "metadata-duckdb"))]
    #[test]
    fn binary_literals_round_trip_through_duckdb() {
        let values = BinaryArray::from(vec![Some(&b"A\0,]\\'NULL"[..]), None]);
        let list = ListArray::new(
            Arc::new(Field::new("item", DataType::Binary, true)),
            OffsetBuffer::new(ScalarBuffer::from(vec![0_i32, 2])),
            Arc::new(values),
            None,
        );
        let encoded = render_text(&list, 0).unwrap();
        let conn = duckdb::Connection::open_in_memory().unwrap();
        let bytes: Vec<u8> = conn
            .query_row("SELECT CAST(? AS BLOB[])[1]", [&encoded], |row| row.get(0))
            .unwrap();
        assert_eq!(bytes, b"A\0,]\\'NULL");
        assert_eq!(
            parse_text(&encoded, list.data_type()).unwrap(),
            ScalarValue::try_from_array(&list, 0).unwrap()
        );
        for scalar in [
            ScalarValue::LargeBinary(Some(bytes.clone())),
            ScalarValue::BinaryView(Some(bytes.clone())),
            ScalarValue::FixedSizeBinary(10, Some(bytes.clone())),
        ] {
            let item_type = scalar.data_type();
            let null = ScalarValue::try_from(&item_type).unwrap();
            let data_type = DataType::List(Arc::new(Field::new("item", item_type, true)));
            let expected = build_list_scalar(&data_type, vec![scalar, null]).unwrap();
            assert_eq!(parse_text(&encoded, &data_type).unwrap(), expected);
        }
        let reference: String = conn
            .query_row("SELECT [CAST(? AS BLOB)]::VARCHAR", [&bytes], |row| {
                row.get(0)
            })
            .unwrap();
        let expected =
            build_list_scalar(list.data_type(), vec![ScalarValue::Binary(Some(bytes))]).unwrap();
        assert_eq!(parse_text(&reference, list.data_type()).unwrap(), expected);
    }
}
