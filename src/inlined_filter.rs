//! Conservative filter translation for metadata-catalog inlined scans.

use std::collections::HashMap;

use arrow::datatypes::{DataType, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::common::ScalarValue;
use datafusion::logical_expr::{Expr, Operator};

/// Result of a metadata-provider inlined scan.
#[derive(Debug, Default)]
pub struct InlinedDataScan {
    /// Materialized Arrow batches.
    pub batches: Vec<RecordBatch>,
    /// Rows materialized by the metadata backend before residual filtering.
    pub materialized_row_count: usize,
}

impl InlinedDataScan {
    pub(crate) fn from_batches(batches: Vec<RecordBatch>) -> Self {
        let materialized_row_count = batches.iter().map(RecordBatch::num_rows).sum();
        Self {
            batches,
            materialized_row_count,
        }
    }
}

/// Backend-neutral predicate accepted by an inlined metadata scan.
#[derive(Clone, Debug, PartialEq)]
pub enum InlinedFilter {
    Comparison {
        column: String,
        op: InlinedComparison,
        value: InlinedValue,
    },
    IsNull(String),
    IsNotNull(String),
    Prefix {
        column: String,
        prefix: String,
    },
    And(Vec<InlinedFilter>),
    Or(Vec<InlinedFilter>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InlinedComparison {
    Eq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
}

impl InlinedComparison {
    fn reversed(self) -> Self {
        match self {
            Self::Eq => Self::Eq,
            Self::NotEq => Self::NotEq,
            Self::Lt => Self::Gt,
            Self::LtEq => Self::GtEq,
            Self::Gt => Self::Lt,
            Self::GtEq => Self::LtEq,
        }
    }

    fn sql(self) -> &'static str {
        match self {
            Self::Eq => "=",
            Self::NotEq => "<>",
            Self::Lt => "<",
            Self::LtEq => "<=",
            Self::Gt => ">",
            Self::GtEq => ">=",
        }
    }

    fn is_range(self) -> bool {
        matches!(self, Self::Lt | Self::LtEq | Self::Gt | Self::GtEq)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum InlinedValue {
    Bool(bool),
    I64(i64),
    U64(u64),
    F64(f64),
    Utf8(String),
    Binary(Vec<u8>),
}

/// SQL bind value after backend physical-encoding validation.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum InlinedSqlBind {
    Bool(bool),
    I64(i64),
    U64(u64),
    F64(f64),
    Text(String),
    Bytes(Vec<u8>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum InlinedSqlDialect {
    Sqlite,
    Postgres,
    DuckDb,
    MySql,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RenderedInlinedFilter {
    pub sql: String,
    pub binds: Vec<InlinedSqlBind>,
}

/// Translate DataFusion filters once. Unsupported top-level filters are omitted.
pub(crate) fn translate_inlined_filters(filters: &[Expr]) -> Option<InlinedFilter> {
    let translated = filters
        .iter()
        .filter_map(translate_expr)
        .collect::<Vec<_>>();
    match translated.len() {
        0 => None,
        1 => translated.into_iter().next(),
        _ => Some(InlinedFilter::And(translated)),
    }
}

fn translate_expr(expr: &Expr) -> Option<InlinedFilter> {
    match expr {
        Expr::BinaryExpr(binary) if binary.op == Operator::And => {
            let children = [translate_expr(&binary.left), translate_expr(&binary.right)]
                .into_iter()
                .flatten()
                .collect::<Vec<_>>();
            match children.len() {
                0 => None,
                1 => children.into_iter().next(),
                _ => Some(InlinedFilter::And(children)),
            }
        },
        Expr::BinaryExpr(binary) if binary.op == Operator::Or => Some(InlinedFilter::Or(vec![
            translate_expr(&binary.left)?,
            translate_expr(&binary.right)?,
        ])),
        Expr::BinaryExpr(binary) => {
            let op = comparison(binary.op)?;
            if let (Some(column), Some(value)) = (column(&binary.left), literal(&binary.right)) {
                Some(InlinedFilter::Comparison {
                    column,
                    op,
                    value,
                })
            } else {
                Some(InlinedFilter::Comparison {
                    column: column(&binary.right)?,
                    op: op.reversed(),
                    value: literal(&binary.left)?,
                })
            }
        },
        Expr::IsNull(expr) => Some(InlinedFilter::IsNull(column(expr)?)),
        Expr::IsNotNull(expr) => Some(InlinedFilter::IsNotNull(column(expr)?)),
        Expr::Like(like) if !like.negated && !like.case_insensitive => {
            let pattern = match like.pattern.as_ref() {
                Expr::Literal(value, _) => scalar_string(value)?,
                _ => return None,
            };
            Some(InlinedFilter::Prefix {
                column: column(&like.expr)?,
                prefix: parse_prefix_pattern(&pattern, like.escape_char)?,
            })
        },
        Expr::ScalarFunction(function)
            if function.name() == "starts_with" && function.args.len() == 2 =>
        {
            let prefix = match &function.args[1] {
                Expr::Literal(value, _) => scalar_string(value)?,
                _ => return None,
            };
            Some(InlinedFilter::Prefix {
                column: column(&function.args[0])?,
                prefix,
            })
        },
        _ => None,
    }
}

fn comparison(op: Operator) -> Option<InlinedComparison> {
    match op {
        Operator::Eq => Some(InlinedComparison::Eq),
        Operator::NotEq => Some(InlinedComparison::NotEq),
        Operator::Lt => Some(InlinedComparison::Lt),
        Operator::LtEq => Some(InlinedComparison::LtEq),
        Operator::Gt => Some(InlinedComparison::Gt),
        Operator::GtEq => Some(InlinedComparison::GtEq),
        _ => None,
    }
}

fn column(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Column(column) => Some(column.name.clone()),
        _ => None,
    }
}

fn literal(expr: &Expr) -> Option<InlinedValue> {
    match expr {
        Expr::Literal(value, _) => scalar(value),
        _ => None,
    }
}

fn scalar(value: &ScalarValue) -> Option<InlinedValue> {
    macro_rules! signed {
        ($value:expr) => {
            $value.map(|value| InlinedValue::I64(i64::from(value)))
        };
    }
    macro_rules! unsigned {
        ($value:expr) => {
            $value.map(|value| InlinedValue::U64(u64::from(value)))
        };
    }
    match value {
        ScalarValue::Boolean(value) => value.map(InlinedValue::Bool),
        ScalarValue::Int8(value) => signed!(*value),
        ScalarValue::Int16(value) => signed!(*value),
        ScalarValue::Int32(value) => signed!(*value),
        ScalarValue::Int64(value) => value.map(InlinedValue::I64),
        ScalarValue::UInt8(value) => unsigned!(*value),
        ScalarValue::UInt16(value) => unsigned!(*value),
        ScalarValue::UInt32(value) => unsigned!(*value),
        ScalarValue::UInt64(value) => value.map(InlinedValue::U64),
        ScalarValue::Float32(Some(value)) if value.is_finite() => {
            Some(InlinedValue::F64(f64::from(*value)))
        },
        ScalarValue::Float64(Some(value)) if value.is_finite() => Some(InlinedValue::F64(*value)),
        ScalarValue::Utf8(value) | ScalarValue::LargeUtf8(value) | ScalarValue::Utf8View(value) => {
            value.clone().map(InlinedValue::Utf8)
        },
        ScalarValue::Binary(value)
        | ScalarValue::LargeBinary(value)
        | ScalarValue::BinaryView(value) => value.clone().map(InlinedValue::Binary),
        ScalarValue::FixedSizeBinary(_, value) => value.clone().map(InlinedValue::Binary),
        _ => None,
    }
}

fn scalar_string(value: &ScalarValue) -> Option<String> {
    match scalar(value)? {
        InlinedValue::Utf8(value) => Some(value),
        _ => None,
    }
}

fn parse_prefix_pattern(pattern: &str, escape: Option<char>) -> Option<String> {
    let mut prefix = String::new();
    let mut chars = pattern.chars().peekable();
    let mut wildcard = false;
    while let Some(ch) = chars.next() {
        if Some(ch) == escape {
            prefix.push(chars.next()?);
            continue;
        }
        match ch {
            '_' => return None,
            '%' => {
                if wildcard || chars.peek().is_some() {
                    return None;
                }
                wildcard = true;
            },
            _ if wildcard => return None,
            _ => prefix.push(ch),
        }
    }
    wildcard.then_some(prefix)
}

pub(crate) fn render_inlined_filter(
    filter: &InlinedFilter,
    dialect: InlinedSqlDialect,
    schema: &Schema,
    physical_types: &HashMap<String, String>,
    first_placeholder: usize,
) -> Option<RenderedInlinedFilter> {
    let fields = schema
        .fields()
        .iter()
        .map(|field| (field.name().as_str(), field.data_type()))
        .collect::<HashMap<_, _>>();
    let mut binds = Vec::new();
    let sql = render_node(
        filter,
        dialect,
        &fields,
        physical_types,
        first_placeholder,
        &mut binds,
    )?;
    Some(RenderedInlinedFilter {
        sql,
        binds,
    })
}

fn render_node(
    filter: &InlinedFilter,
    dialect: InlinedSqlDialect,
    fields: &HashMap<&str, &DataType>,
    physical_types: &HashMap<String, String>,
    first_placeholder: usize,
    binds: &mut Vec<InlinedSqlBind>,
) -> Option<String> {
    match filter {
        InlinedFilter::And(children) => {
            let rendered = children
                .iter()
                .filter_map(|child| {
                    render_node(
                        child,
                        dialect,
                        fields,
                        physical_types,
                        first_placeholder,
                        binds,
                    )
                })
                .collect::<Vec<_>>();
            match rendered.len() {
                0 => None,
                1 => rendered.into_iter().next(),
                _ => Some(format!("({})", rendered.join(" AND "))),
            }
        },
        InlinedFilter::Or(children) => {
            let checkpoint = binds.len();
            let rendered = children
                .iter()
                .map(|child| {
                    render_node(
                        child,
                        dialect,
                        fields,
                        physical_types,
                        first_placeholder,
                        binds,
                    )
                })
                .collect::<Option<Vec<_>>>();
            match rendered {
                Some(rendered) => Some(format!("({})", rendered.join(" OR "))),
                None => {
                    binds.truncate(checkpoint);
                    None
                },
            }
        },
        InlinedFilter::IsNull(column) | InlinedFilter::IsNotNull(column) => {
            physical_types.contains_key(column).then(|| {
                format!(
                    "{} IS {}NULL",
                    quote_ident(column, dialect),
                    if matches!(filter, InlinedFilter::IsNotNull(_)) {
                        "NOT "
                    } else {
                        ""
                    }
                )
            })
        },
        InlinedFilter::Comparison {
            column,
            op,
            value,
        } => {
            let data_type = *fields.get(column.as_str())?;
            let physical_type = physical_types.get(column)?;
            if !physical_type_matches(dialect, data_type, physical_type) {
                return None;
            }
            let bind = if *data_type == DataType::UInt64 {
                u64_sql_bind(value, dialect)?
            } else {
                sql_bind(value, data_type, dialect, op.is_range())?
            };
            let placeholder = placeholder(dialect, first_placeholder + binds.len());
            binds.push(bind);
            if *data_type == DataType::UInt64 {
                // SQLite legacy tables mix unpadded rows with padded rows written
                // after the storage change, so every operator needs the padded
                // normalization; the other text backends only store canonical
                // digits and need it for ordering alone.
                if (op.is_range() || dialect == InlinedSqlDialect::Sqlite)
                    && u64_uses_legacy_text(dialect, physical_type)
                    && let Some((ident, value)) =
                        text_u64_range_operands(column, &placeholder, dialect)
                {
                    return Some(format!("{ident} {} {value}", op.sql()));
                }
                let (ident, value) =
                    native_u64_operands(column, &placeholder, dialect, physical_type);
                return Some(format!("{ident} {} {value}", op.sql()));
            }
            let ident = comparison_ident(column, data_type, dialect);
            Some(format!("{ident} {} {placeholder}", op.sql()))
        },
        InlinedFilter::Prefix {
            column,
            prefix,
        } => {
            let data_type = *fields.get(column.as_str())?;
            if !is_utf8(data_type)
                || !physical_type_matches(dialect, data_type, physical_types.get(column)?)
            {
                return None;
            }
            let bind = sql_bind(
                &InlinedValue::Utf8(prefix.clone()),
                data_type,
                dialect,
                false,
            )?;
            let ident = quote_ident(column, dialect);
            let first = placeholder(dialect, first_placeholder + binds.len());
            binds.push(bind.clone());
            match dialect {
                InlinedSqlDialect::DuckDb => Some(format!("starts_with({ident}, {first})")),
                InlinedSqlDialect::Sqlite => {
                    let second = placeholder(dialect, first_placeholder + binds.len());
                    binds.push(bind);
                    Some(format!(
                        "substr({ident}, 1, length({first})) = {second} COLLATE BINARY"
                    ))
                },
                InlinedSqlDialect::Postgres => {
                    let second = placeholder(dialect, first_placeholder + binds.len());
                    binds.push(bind);
                    Some(format!(
                        "substring({ident} from 1 for octet_length({first})) = {second}"
                    ))
                },
                InlinedSqlDialect::MySql => {
                    let second = placeholder(dialect, first_placeholder + binds.len());
                    binds.push(bind);
                    Some(format!(
                        "LEFT(BINARY {ident}, OCTET_LENGTH({first})) = BINARY {second}"
                    ))
                },
            }
        },
    }
}

fn sql_bind(
    value: &InlinedValue,
    data_type: &DataType,
    dialect: InlinedSqlDialect,
    range: bool,
) -> Option<InlinedSqlBind> {
    match (data_type, value) {
        (DataType::Boolean, InlinedValue::Bool(value)) if !range => {
            Some(InlinedSqlBind::Bool(*value))
        },
        (
            DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64,
            InlinedValue::I64(value),
        ) => Some(InlinedSqlBind::I64(*value)),
        (DataType::UInt8 | DataType::UInt16 | DataType::UInt32, InlinedValue::U64(value)) => {
            i64::try_from(*value).ok().map(InlinedSqlBind::I64)
        },
        (DataType::UInt64, InlinedValue::U64(value)) => match dialect {
            InlinedSqlDialect::Sqlite if range => Some(InlinedSqlBind::Text(value.to_string())),
            InlinedSqlDialect::Sqlite => i64::try_from(*value).ok().map(InlinedSqlBind::I64),
            InlinedSqlDialect::DuckDb => Some(InlinedSqlBind::U64(*value)),
            InlinedSqlDialect::Postgres | InlinedSqlDialect::MySql => {
                Some(InlinedSqlBind::Text(value.to_string()))
            },
        },
        (DataType::Float32 | DataType::Float64, InlinedValue::F64(value))
            if value.is_finite() && !(dialect == InlinedSqlDialect::MySql && range) =>
        {
            Some(if dialect == InlinedSqlDialect::MySql {
                InlinedSqlBind::Text(value.to_string())
            } else {
                InlinedSqlBind::F64(*value)
            })
        },
        (data_type, InlinedValue::Utf8(value)) if is_utf8(data_type) => {
            Some(if dialect == InlinedSqlDialect::Postgres {
                InlinedSqlBind::Bytes(value.as_bytes().to_vec())
            } else {
                InlinedSqlBind::Text(value.clone())
            })
        },
        (data_type, InlinedValue::Binary(value)) if is_binary(data_type) => {
            Some(InlinedSqlBind::Bytes(value.clone()))
        },
        _ => None,
    }
}

fn u64_sql_bind(value: &InlinedValue, dialect: InlinedSqlDialect) -> Option<InlinedSqlBind> {
    let InlinedValue::U64(value) = value else {
        return None;
    };
    match dialect {
        InlinedSqlDialect::DuckDb => Some(InlinedSqlBind::U64(*value)),
        InlinedSqlDialect::Sqlite | InlinedSqlDialect::Postgres | InlinedSqlDialect::MySql => {
            Some(InlinedSqlBind::Text(value.to_string()))
        },
    }
}

fn u64_uses_legacy_text(dialect: InlinedSqlDialect, physical_type: &str) -> bool {
    let physical = physical_type.trim().to_ascii_uppercase();
    match dialect {
        InlinedSqlDialect::Sqlite => physical != "TEXT",
        InlinedSqlDialect::Postgres => {
            !physical.starts_with("NUMERIC") && !physical.starts_with("DECIMAL")
        },
        InlinedSqlDialect::MySql => !(physical.contains("BIGINT") && physical.contains("UNSIGNED")),
        InlinedSqlDialect::DuckDb => false,
    }
}

fn native_u64_operands(
    column: &str,
    placeholder: &str,
    dialect: InlinedSqlDialect,
    physical_type: &str,
) -> (String, String) {
    let ident = quote_ident(column, dialect);
    let physical = physical_type.trim().to_ascii_uppercase();
    match dialect {
        InlinedSqlDialect::Sqlite if physical == "TEXT" => (
            format!("{ident} COLLATE BINARY"),
            format!("substr('00000000000000000000' || {placeholder}, -20, 20)"),
        ),
        InlinedSqlDialect::Postgres
            if physical.starts_with("NUMERIC") || physical.starts_with("DECIMAL") =>
        {
            (ident, format!("CAST({placeholder} AS NUMERIC(20,0))"))
        },
        InlinedSqlDialect::MySql
            if physical.contains("BIGINT") && physical.contains("UNSIGNED") =>
        {
            (
                ident,
                format!("CAST(CAST({placeholder} AS DECIMAL(20,0)) AS UNSIGNED)"),
            )
        },
        _ => (ident, placeholder.to_string()),
    }
}

fn text_u64_range_operands(
    column: &str,
    placeholder: &str,
    dialect: InlinedSqlDialect,
) -> Option<(String, String)> {
    // These backends store UInt64 as canonical decimal text. Fixed-width
    // zero padding makes byte ordering match the complete UInt64 numeric
    // domain. SQLite must NOT use printf('%020s', ...): its 0 flag pads
    // strings with spaces, which sort before digits and misorder rows the
    // padded writer stored in a legacy table.
    let ident = quote_ident(column, dialect);
    match dialect {
        InlinedSqlDialect::Sqlite => Some((
            format!("substr('00000000000000000000' || {ident}, -20, 20) COLLATE BINARY"),
            format!("substr('00000000000000000000' || {placeholder}, -20, 20)"),
        )),
        InlinedSqlDialect::Postgres => Some((
            format!("convert_to(lpad({ident}, 20, '0'), 'UTF8')"),
            format!("convert_to(lpad({placeholder}, 20, '0'), 'UTF8')"),
        )),
        InlinedSqlDialect::MySql => Some((
            format!("BINARY LPAD({ident}, 20, '0')"),
            format!("BINARY LPAD({placeholder}, 20, '0')"),
        )),
        InlinedSqlDialect::DuckDb => None,
    }
}

fn comparison_ident(column: &str, data_type: &DataType, dialect: InlinedSqlDialect) -> String {
    let ident = quote_ident(column, dialect);
    match (dialect, data_type) {
        (InlinedSqlDialect::Sqlite, data_type) if is_utf8(data_type) => {
            format!("{ident} COLLATE BINARY")
        },
        (InlinedSqlDialect::MySql, data_type) if is_utf8(data_type) => {
            format!("BINARY {ident}")
        },
        _ => ident,
    }
}

fn quote_ident(name: &str, dialect: InlinedSqlDialect) -> String {
    if dialect == InlinedSqlDialect::MySql {
        format!("`{}`", name.replace('`', "``"))
    } else {
        format!("\"{}\"", name.replace('"', "\"\""))
    }
}

fn placeholder(dialect: InlinedSqlDialect, index: usize) -> String {
    if dialect == InlinedSqlDialect::Postgres {
        format!("${}", index + 1)
    } else {
        "?".to_string()
    }
}

fn physical_type_matches(
    dialect: InlinedSqlDialect,
    data_type: &DataType,
    physical_type: &str,
) -> bool {
    let physical = physical_type.trim().to_ascii_uppercase();
    match dialect {
        InlinedSqlDialect::Sqlite => match data_type {
            DataType::Boolean
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32 => physical.contains("INT") || physical == "BOOLEAN",
            DataType::UInt64 => physical.contains("CHAR") || physical.contains("TEXT"),
            DataType::Float32 | DataType::Float64 => {
                physical.contains("DOUBLE") || physical.contains("REAL")
            },
            data_type if is_utf8(data_type) => {
                physical.contains("CHAR") || physical.contains("TEXT")
            },
            data_type if is_binary(data_type) => physical.contains("BLOB"),
            _ => false,
        },
        InlinedSqlDialect::Postgres => match data_type {
            DataType::Boolean => physical == "BOOLEAN" || physical == "BOOL",
            DataType::Int8 | DataType::Int16 => physical == "SMALLINT" || physical == "INT2",
            DataType::Int32 => physical == "INTEGER" || physical == "INT4",
            DataType::Int64 => physical == "BIGINT" || physical == "INT8",
            DataType::UInt8 | DataType::UInt16 => {
                matches!(physical.as_str(), "INTEGER" | "INT4")
            },
            DataType::UInt32 => matches!(physical.as_str(), "BIGINT" | "INT8"),
            DataType::UInt64 => {
                physical.contains("CHAR")
                    || physical == "TEXT"
                    || physical.starts_with("NUMERIC")
                    || physical.starts_with("DECIMAL")
            },
            DataType::Float32 => matches!(physical.as_str(), "REAL" | "FLOAT4"),
            DataType::Float64 => {
                matches!(physical.as_str(), "DOUBLE PRECISION" | "FLOAT8")
            },
            DataType::Decimal32(_, _)
            | DataType::Decimal64(_, _)
            | DataType::Decimal128(_, _)
            | DataType::Decimal256(_, _) => {
                physical.starts_with("NUMERIC") || physical.starts_with("DECIMAL")
            },
            data_type if is_utf8(data_type) || is_binary(data_type) => physical == "BYTEA",
            _ => false,
        },
        InlinedSqlDialect::DuckDb => match data_type {
            DataType::Boolean => physical == "BOOLEAN",
            DataType::Int8 => matches!(physical.as_str(), "TINYINT" | "INT8"),
            DataType::Int16 => matches!(physical.as_str(), "SMALLINT" | "INT16"),
            DataType::Int32 => matches!(physical.as_str(), "INTEGER" | "INT32"),
            DataType::Int64 => matches!(physical.as_str(), "BIGINT" | "INT64"),
            DataType::UInt8 => matches!(physical.as_str(), "UTINYINT" | "UINT8"),
            DataType::UInt16 => matches!(physical.as_str(), "USMALLINT" | "UINT16"),
            DataType::UInt32 => matches!(physical.as_str(), "UINTEGER" | "UINT32"),
            DataType::UInt64 => matches!(physical.as_str(), "UBIGINT" | "UINT64"),
            DataType::Float32 => matches!(physical.as_str(), "FLOAT" | "REAL"),
            DataType::Float64 => physical == "DOUBLE",
            data_type if is_utf8(data_type) => physical.contains("VARCHAR"),
            data_type if is_binary(data_type) => physical == "BLOB",
            _ => false,
        },
        InlinedSqlDialect::MySql => match data_type {
            DataType::Boolean
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32 => physical.contains("BIGINT") || physical.contains("INT"),
            DataType::UInt64 => {
                physical.contains("TEXT")
                    || physical.contains("CHAR")
                    || (physical.contains("BIGINT") && physical.contains("UNSIGNED"))
            },
            DataType::Float32 => {
                physical.contains("TEXT") || physical.contains("CHAR") || physical == "FLOAT"
            },
            DataType::Float64 => {
                physical.contains("TEXT")
                    || physical.contains("CHAR")
                    || physical == "DOUBLE"
                    || physical == "DOUBLE PRECISION"
            },
            DataType::Decimal32(_, _)
            | DataType::Decimal64(_, _)
            | DataType::Decimal128(_, _)
            | DataType::Decimal256(_, _) => {
                physical.contains("TEXT")
                    || physical.contains("CHAR")
                    || physical.starts_with("DECIMAL")
                    || physical.starts_with("NUMERIC")
            },
            data_type if is_utf8(data_type) => {
                physical.contains("TEXT") || physical.contains("CHAR")
            },
            data_type if is_binary(data_type) => {
                physical.contains("BLOB") || physical.contains("BINARY")
            },
            _ => false,
        },
    }
}

fn is_utf8(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
    )
}

fn is_binary(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Binary | DataType::LargeBinary | DataType::BinaryView
    )
}

#[cfg(test)]
mod tests {
    use arrow::datatypes::Field;
    use datafusion::logical_expr::{col, lit};

    use super::*;

    #[test]
    fn and_pushes_supported_children_independently() {
        let expr = col("id").eq(lit("A")).and(col("x") + lit(1).eq(lit(2)));
        assert_eq!(
            translate_inlined_filters(&[expr]),
            Some(InlinedFilter::Comparison {
                column: "id".to_string(),
                op: InlinedComparison::Eq,
                value: InlinedValue::Utf8("A".to_string()),
            })
        );
    }

    #[test]
    fn or_requires_every_branch() {
        let expr = col("id").eq(lit("A")).or(col("x") + lit(1).eq(lit(2)));
        assert_eq!(translate_inlined_filters(&[expr]), None);
    }

    #[test]
    fn prefix_pattern_decodes_escaped_wildcards() {
        assert_eq!(
            parse_prefix_pattern(r"ES\%\_%", Some('\\')),
            Some("ES%_".to_string())
        );
        assert_eq!(parse_prefix_pattern("ES_%", None), None);
    }

    fn render_schema() -> Schema {
        Schema::new(vec![
            Field::new("id", DataType::Int32, true),
            Field::new("note", DataType::Utf8, true),
            Field::new("u64", DataType::UInt64, true),
        ])
    }

    #[test]
    fn mysql_string_filters_force_binary_semantics_without_like_wildcards() {
        let present = HashMap::from([("note".to_string(), "LONGTEXT".to_string())]);
        let rendered = render_inlined_filter(
            &InlinedFilter::Prefix {
                column: "note".to_string(),
                prefix: "a%_".to_string(),
            },
            InlinedSqlDialect::MySql,
            &render_schema(),
            &present,
            2,
        )
        .unwrap();
        assert_eq!(
            rendered.sql,
            "LEFT(BINARY `note`, OCTET_LENGTH(?)) = BINARY ?"
        );
        assert_eq!(
            rendered.binds,
            vec![InlinedSqlBind::Text("a%_".to_string()), InlinedSqlBind::Text("a%_".to_string()),]
        );
    }

    #[test]
    fn postgres_prefix_uses_bytea_and_numbered_parameters() {
        let present = HashMap::from([("note".to_string(), "BYTEA".to_string())]);
        let rendered = render_inlined_filter(
            &InlinedFilter::Prefix {
                column: "note".to_string(),
                prefix: "Ab".to_string(),
            },
            InlinedSqlDialect::Postgres,
            &render_schema(),
            &present,
            2,
        )
        .unwrap();
        assert_eq!(
            rendered.sql,
            "substring(\"note\" from 1 for octet_length($3)) = $4"
        );
        assert_eq!(
            rendered.binds,
            vec![InlinedSqlBind::Bytes(b"Ab".to_vec()), InlinedSqlBind::Bytes(b"Ab".to_vec()),]
        );
    }

    #[test]
    fn missing_column_falls_back_for_or_but_not_supported_and_sibling() {
        let present = HashMap::from([("id".to_string(), "BIGINT".to_string())]);
        let supported = InlinedFilter::Comparison {
            column: "id".to_string(),
            op: InlinedComparison::Eq,
            value: InlinedValue::I64(7),
        };
        let missing = InlinedFilter::Comparison {
            column: "missing".to_string(),
            op: InlinedComparison::Eq,
            value: InlinedValue::I64(7),
        };
        assert!(
            render_inlined_filter(
                &InlinedFilter::Or(vec![supported.clone(), missing.clone()]),
                InlinedSqlDialect::Sqlite,
                &render_schema(),
                &present,
                2,
            )
            .is_none()
        );
        let rendered = render_inlined_filter(
            &InlinedFilter::And(vec![supported, missing]),
            InlinedSqlDialect::Sqlite,
            &render_schema(),
            &present,
            2,
        )
        .unwrap();
        assert_eq!(rendered.sql, "\"id\" = ?");
        assert_eq!(rendered.binds, vec![InlinedSqlBind::I64(7)]);
    }

    #[test]
    fn uint64_ranges_preserve_order_for_text_and_numeric_encodings() {
        let filter = InlinedFilter::Comparison {
            column: "u64".to_string(),
            op: InlinedComparison::Gt,
            value: InlinedValue::U64(u64::MAX),
        };
        let sqlite = render_inlined_filter(
            &filter,
            InlinedSqlDialect::Sqlite,
            &render_schema(),
            &HashMap::from([("u64".to_string(), "VARCHAR".to_string())]),
            2,
        )
        .unwrap();
        assert_eq!(
            sqlite.sql,
            "substr('00000000000000000000' || \"u64\", -20, 20) COLLATE BINARY \
             > substr('00000000000000000000' || ?, -20, 20)"
        );
        let postgres = render_inlined_filter(
            &filter,
            InlinedSqlDialect::Postgres,
            &render_schema(),
            &HashMap::from([("u64".to_string(), "character varying".to_string())]),
            2,
        )
        .unwrap();
        assert_eq!(
            postgres.sql,
            "convert_to(lpad(\"u64\", 20, '0'), 'UTF8') > \
             convert_to(lpad($3, 20, '0'), 'UTF8')"
        );
        let mysql = render_inlined_filter(
            &filter,
            InlinedSqlDialect::MySql,
            &render_schema(),
            &HashMap::from([("u64".to_string(), "LONGTEXT".to_string())]),
            2,
        );
        assert_eq!(
            mysql.unwrap().sql,
            "BINARY LPAD(`u64`, 20, '0') > BINARY LPAD(?, 20, '0')"
        );
        assert_eq!(
            render_inlined_filter(
                &filter,
                InlinedSqlDialect::DuckDb,
                &render_schema(),
                &HashMap::from([("u64".to_string(), "UBIGINT".to_string())]),
                2,
            )
            .unwrap()
            .binds,
            vec![InlinedSqlBind::U64(u64::MAX)]
        );
    }

    #[test]
    fn range_pushdown_respects_each_backend_physical_encoding() {
        let cases = [
            (InlinedSqlDialect::Sqlite, "BIGINT", DataType::Int64, true),
            (InlinedSqlDialect::Sqlite, "DOUBLE", DataType::Float64, true),
            (InlinedSqlDialect::Sqlite, "VARCHAR", DataType::UInt64, true),
            (InlinedSqlDialect::Postgres, "bigint", DataType::Int64, true),
            (
                InlinedSqlDialect::Postgres,
                "character varying",
                DataType::UInt64,
                true,
            ),
            (InlinedSqlDialect::DuckDb, "UBIGINT", DataType::UInt64, true),
            (InlinedSqlDialect::MySql, "bigint", DataType::Int64, true),
            (
                InlinedSqlDialect::MySql,
                "longtext",
                DataType::Float64,
                false,
            ),
        ];
        for (dialect, physical, data_type, expected) in cases {
            let schema = Schema::new(vec![Field::new("value", data_type.clone(), true)]);
            let physical_types = HashMap::from([("value".to_string(), physical.to_string())]);
            let value = match data_type {
                DataType::UInt64 => InlinedValue::U64(7),
                DataType::Float64 => InlinedValue::F64(7.5),
                _ => InlinedValue::I64(7),
            };
            let rendered = render_inlined_filter(
                &InlinedFilter::Comparison {
                    column: "value".to_string(),
                    op: InlinedComparison::GtEq,
                    value,
                },
                dialect,
                &schema,
                &physical_types,
                2,
            );
            assert_eq!(
                rendered.is_some(),
                expected,
                "dialect={dialect:?} physical={physical} type={data_type:?}"
            );
        }
    }

    #[test]
    fn native_uint64_comparisons_leave_columns_bare() {
        let filter = InlinedFilter::Comparison {
            column: "u64".to_string(),
            op: InlinedComparison::Gt,
            value: InlinedValue::U64(u64::MAX),
        };
        let cases = [
            (
                InlinedSqlDialect::Sqlite,
                "TEXT",
                "\"u64\" COLLATE BINARY > substr('00000000000000000000' || ?, -20, 20)",
            ),
            (
                InlinedSqlDialect::Postgres,
                "NUMERIC",
                "\"u64\" > CAST($3 AS NUMERIC(20,0))",
            ),
            (
                InlinedSqlDialect::MySql,
                "bigint unsigned",
                "`u64` > CAST(CAST(? AS DECIMAL(20,0)) AS UNSIGNED)",
            ),
        ];
        for (dialect, physical_type, expected) in cases {
            let rendered = render_inlined_filter(
                &filter,
                dialect,
                &render_schema(),
                &HashMap::from([("u64".to_string(), physical_type.to_string())]),
                2,
            )
            .unwrap();
            assert_eq!(rendered.sql, expected);
            assert_eq!(
                rendered.binds,
                vec![InlinedSqlBind::Text(u64::MAX.to_string())]
            );
        }
    }

    #[test]
    fn sqlite_legacy_uint64_equality_normalizes_padding() {
        // Legacy VARCHAR tables can hold both unpadded rows and padded rows
        // written after the storage change; a bare equality would miss the
        // padded ones, so every operator normalizes through zero padding.
        let filter = InlinedFilter::Comparison {
            column: "u64".to_string(),
            op: InlinedComparison::Eq,
            value: InlinedValue::U64(5),
        };
        let rendered = render_inlined_filter(
            &filter,
            InlinedSqlDialect::Sqlite,
            &render_schema(),
            &HashMap::from([("u64".to_string(), "VARCHAR".to_string())]),
            0,
        )
        .unwrap();
        assert_eq!(
            rendered.sql,
            "substr('00000000000000000000' || \"u64\", -20, 20) COLLATE BINARY \
             = substr('00000000000000000000' || ?, -20, 20)"
        );
        assert_eq!(rendered.binds, vec![InlinedSqlBind::Text("5".to_string())]);
    }

    #[test]
    fn changed_physical_encoding_falls_back() {
        let schema = Schema::new(vec![Field::new("id", DataType::Int32, true)]);
        let physical_types = HashMap::from([("id".to_string(), "BYTEA".to_string())]);
        assert!(
            render_inlined_filter(
                &InlinedFilter::Comparison {
                    column: "id".to_string(),
                    op: InlinedComparison::Eq,
                    value: InlinedValue::I64(7),
                },
                InlinedSqlDialect::Postgres,
                &schema,
                &physical_types,
                2,
            )
            .is_none()
        );
    }
}
