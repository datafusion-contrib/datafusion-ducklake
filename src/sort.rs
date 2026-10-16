//! DuckLake table sort order: sort spec model.
//!
//! A sorted DuckLake table records, in the catalog, a **sort spec**
//! (`ducklake_sort_info` + `ducklake_sort_expression`). Unlike a partition spec, a
//! sort spec is *not* a pruning mechanism and carries no per-file catalog rows: its
//! sole job is to order rows *within* each data file on write, which tightens the
//! per-file min/max statistics so the existing statistics-based file pruner skips
//! more files at query time. It is the DuckLake analogue of an Iceberg sort order;
//! there is no Z-order / multi-dimensional clustering.
//!
//! Following the DuckLake spec, each sort key is stored as an **expression** string
//! (with a `dialect`, always `"duckdb"`), plus a sort direction (`ASC`/`DESC`) and a
//! null ordering (`NULLS_FIRST`/`NULLS_LAST`). Storing an expression (rather than a
//! `column_id`) is what lets DuckDB sort by arbitrary expressions/macros.
//!
//! Scope note: `SET SORTED BY` through this crate records only **bare column
//! references** (`SORTED BY (device_id, ts DESC)`). A spec with any other
//! expression (written by DuckDB) is round-tripped verbatim; inserts, the
//! inlined-data flush and compaction sort by it, evaluating it with
//! **DataFusion's** semantics.
//! Where those differ from DuckDB's (integer division, overflow, type
//! promotion, function behaviour), rows may be ordered differently inside files
//! than official orders them; query results are unaffected. When DataFusion
//! cannot plan an expression, or evaluating it fails over the rows being
//! written, the rows keep their arrival order (a flush: row id order; a
//! compaction: the order an unsorted compaction writes): the order is a layout
//! preference and never fails a write.
//!
//! A key names a column as DuckDB's binder resolves it: the column spelled
//! exactly as the key, else the one equal to it ignoring ASCII case. A key
//! naming a column the table lacks fails the write, as official's does.

/// A sort key's direction. Serializes to the catalog `sort_direction` string
/// (`"ASC"` / `"DESC"`), matching DuckLake's on-disk form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortDirection {
    Asc,
    Desc,
}

impl SortDirection {
    /// Parse a catalog `sort_direction` string. Case-insensitive; matches DuckLake,
    /// which treats `"DESC"` as descending and everything else as ascending.
    pub fn parse(value: &str) -> Self {
        if value.trim().eq_ignore_ascii_case("DESC") {
            SortDirection::Desc
        } else {
            SortDirection::Asc
        }
    }

    /// The catalog `sort_direction` string this direction serializes to.
    pub fn to_catalog_string(self) -> &'static str {
        match self {
            SortDirection::Asc => "ASC",
            SortDirection::Desc => "DESC",
        }
    }
}

/// A sort key's null ordering. Serializes to the catalog `null_order` string
/// (`"NULLS_FIRST"` / `"NULLS_LAST"`, underscore form), matching DuckLake.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NullOrder {
    NullsFirst,
    NullsLast,
}

impl NullOrder {
    /// Parse a catalog `null_order` string. Case-insensitive; matches DuckLake,
    /// which treats `"NULLS_FIRST"` as nulls-first and everything else as nulls-last.
    pub fn parse(value: &str) -> Self {
        if value.trim().eq_ignore_ascii_case("NULLS_FIRST") {
            NullOrder::NullsFirst
        } else {
            NullOrder::NullsLast
        }
    }

    /// The catalog `null_order` string this ordering serializes to.
    pub fn to_catalog_string(self) -> &'static str {
        match self {
            NullOrder::NullsFirst => "NULLS_FIRST",
            NullOrder::NullsLast => "NULLS_LAST",
        }
    }

    /// Whether nulls sort first, as the boolean Arrow / DataFusion sort options use.
    pub fn nulls_first(self) -> bool {
        matches!(self, NullOrder::NullsFirst)
    }
}

/// The catalog `dialect` value this crate writes for every sort expression it
/// produces. DuckLake round-trips sort expressions through the DuckDB parser, so a
/// bare column name written under this dialect is read back identically.
pub const DUCKDB_DIALECT: &str = "duckdb";

/// One key of a sort spec: an expression to sort by, plus direction and null order.
///
/// The key is an expression *string* (`ducklake_sort_expression.expression`), not a
/// `column_id` — DuckLake sort keys are expression-based. For a spec this crate
/// produced, `expression` is a bare column name; for a spec written by DuckDB it may
/// be any expression, in which case [`SortField::column_candidate`] returns `None`
/// and writes evaluate it with DataFusion ([`SortSpec`]'s module notes how).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SortField {
    /// 0-based position of this key within the sort order.
    pub sort_key_index: i32,
    /// The sort expression (`ducklake_sort_expression.expression`).
    pub expression: String,
    /// The expression dialect (`ducklake_sort_expression.dialect`), e.g. `"duckdb"`.
    pub dialect: String,
    /// Ascending or descending.
    pub direction: SortDirection,
    /// Where nulls sort.
    pub null_order: NullOrder,
}

impl SortField {
    /// Build a bare-column sort field this crate can produce. `expression` is the
    /// column name as DuckDB writes a column reference: bare when it needs no
    /// quotes, else double-quoted (`"My Col"`, `"year"`), as official stores
    /// `SET SORTED BY` keys. Dialect is set to [`DUCKDB_DIALECT`].
    pub fn column(
        sort_key_index: i32,
        column: impl Into<String>,
        direction: SortDirection,
        null_order: NullOrder,
    ) -> Self {
        SortField {
            sort_key_index,
            expression: crate::snapshot_changes::display_identifier(&column.into()),
            dialect: DUCKDB_DIALECT.to_string(),
            direction,
            null_order,
        }
    }

    /// Interpret this key's expression as a bare column reference, returning the
    /// column name if it is one. A bare column is either an unquoted simple
    /// identifier (`ts`, `device_id`) or a double-quoted identifier (`"My Col"`,
    /// unquoted here, `""` read as `"`). Anything else — function calls,
    /// arithmetic, qualified names, multiple tokens — yields `None`.
    pub fn column_candidate(&self) -> Option<String> {
        parse_bare_column(&self.expression)
    }

    /// Whether official evaluates this key: its dialect is exactly `duckdb`
    /// (`field.dialect != "duckdb"` skips it in `ducklake_sort_data.cpp` and
    /// `ducklake_compaction_functions.cpp`).
    pub(crate) fn is_duckdb_dialect(&self) -> bool {
        self.dialect == DUCKDB_DIALECT
    }
}

/// The index among `names` of the column a sort key naming `key` binds to, as
/// DuckDB's binder binds it: the name spelled exactly as `key`, else the one
/// name equal to it ignoring ASCII case (DuckDB's column lookup is
/// case-insensitive). `None` when no name matches.
///
/// Several names equal to `key` ignoring case, none exactly, fail with
/// [`DuckLakeError::InvalidConfig`](crate::DuckLakeError::InvalidConfig):
/// DuckDB's binder refuses a table with such columns ("duplicate column
/// name"), so no column is the one official would sort by.
pub(crate) fn bind_sort_column<'a>(
    names: impl IntoIterator<Item = &'a str>,
    key: &str,
) -> crate::Result<Option<usize>> {
    let mut folded = None;
    let mut ambiguous = false;
    for (index, name) in names.into_iter().enumerate() {
        if name == key {
            return Ok(Some(index));
        }
        if name.eq_ignore_ascii_case(key) {
            ambiguous |= folded.is_some();
            folded = Some(index);
        }
    }
    if ambiguous {
        return Err(crate::DuckLakeError::InvalidConfig(format!(
            "DuckLake sort key '{key}' is ambiguous: several columns match it ignoring case"
        )));
    }
    Ok(folded)
}

/// Check that every key of `fields` is a bare column binding to one of
/// `names` ([`bind_sort_column`]), as `set_sort_spec` requires. Keys naming no
/// column fail with official's binder error ([`missing_sort_columns`]),
/// naming every such column.
#[cfg(feature = "write")]
pub(crate) fn check_sort_key_columns(fields: &[SortField], names: &[String]) -> crate::Result<()> {
    let mut missing = Vec::new();
    for field in fields {
        let column = field.column_candidate().ok_or_else(|| {
            crate::DuckLakeError::InvalidConfig(format!(
                "set_sort_spec: sort key '{}' is not a bare column; only column sort keys are \
                 supported",
                field.expression
            ))
        })?;
        if bind_sort_column(names.iter().map(String::as_str), &column)?.is_none() {
            note_missing(&mut missing, &column);
        }
    }
    if !missing.is_empty() {
        return Err(missing_sort_columns(&missing));
    }
    Ok(())
}

/// The columns sort expression `sql` references, as official's
/// `ParsedExpressionIterator` visits its column references: a qualified
/// reference (`t.b`, or a struct field `s.x`, which DuckDB parses as a column
/// reference too) by its last part, as `ColumnRefExpression::GetColumnName`
/// names it, with whether it was qualified. `None` when it does not parse.
#[cfg(feature = "write")]
fn referenced_columns(sql: &str) -> Option<Vec<(String, bool)>> {
    use datafusion::sql::sqlparser::ast::{Expr as SqlExpr, visit_expressions};
    use datafusion::sql::sqlparser::dialect::GenericDialect;
    use datafusion::sql::sqlparser::parser::Parser;
    use std::ops::ControlFlow;

    if let Some(column) = parse_bare_column(sql) {
        return Some(vec![(column, false)]);
    }
    let expr = Parser::new(&GenericDialect {})
        .try_with_sql(sql)
        .ok()?
        .parse_expr()
        .ok()?;
    let mut columns = Vec::new();
    let _ = visit_expressions(&expr, |node| {
        match node {
            SqlExpr::Identifier(ident) => columns.push((ident.value.clone(), false)),
            SqlExpr::CompoundIdentifier(parts) => {
                if let Some(last) = parts.last() {
                    columns.push((last.value.clone(), true));
                }
            },
            _ => {},
        }
        ControlFlow::<()>::Continue(())
    });
    Some(columns)
}

/// Refuse to drop `removed` columns of a table whose live sort keys are
/// `keys` (`(expression, dialect)`), as official's `ALTER TABLE ... DROP
/// COLUMN` refuses a column a `duckdb` key references, compared ignoring
/// ASCII case ("Cannot drop column "b" - the table is sorted by this column.
/// ..."). A key this crate cannot parse might reference any column, so it
/// refuses every drop, where official parses it and may allow the drop.
#[cfg(feature = "write")]
pub(crate) fn refuse_dropping_sort_columns(
    keys: &[(String, String)],
    removed: &[&str],
) -> crate::Result<()> {
    let Some(first) = removed.first() else {
        return Ok(());
    };
    for (expression, dialect) in keys {
        if dialect != DUCKDB_DIALECT {
            continue;
        }
        let Some(columns) = referenced_columns(expression) else {
            return Err(crate::DuckLakeError::InvalidConfig(format!(
                "Cannot drop column \"{first}\" - the table is sorted by the expression \
                 \"{expression}\", which could not be parsed to check the columns it uses. \
                 Reset or change the sort order on this table in order to drop this column"
            )));
        };
        for (column, _) in columns {
            if let Some(name) = removed
                .iter()
                .find(|name| name.eq_ignore_ascii_case(&column))
            {
                return Err(crate::DuckLakeError::InvalidConfig(format!(
                    "Cannot drop column \"{name}\" - the table is sorted by this column. Reset \
                     or change the sort order on this table in order to drop this column"
                )));
            }
        }
    }
    Ok(())
}

/// The SQL reading a table's live sort keys as `(expression, dialect)`, for
/// [`refuse_dropping_sort_columns`], its table id bound as `?`.
#[cfg(feature = "write")]
pub(crate) const LIVE_SORT_KEYS_SQL: &str = "SELECT se.expression, se.dialect
     FROM ducklake_sort_info AS si
     JOIN ducklake_sort_expression AS se
       ON se.sort_id = si.sort_id AND se.table_id = si.table_id
     WHERE si.table_id = ? AND si.end_snapshot IS NULL";

/// [`LIVE_SORT_KEYS_SQL`], its table id bound as `$1`.
#[cfg(feature = "write")]
pub(crate) const LIVE_SORT_KEYS_SQL_PG: &str = "SELECT se.expression, se.dialect
     FROM ducklake_sort_info AS si
     JOIN ducklake_sort_expression AS se
       ON se.sort_id = si.sort_id AND se.table_id = si.table_id
     WHERE si.table_id = $1 AND si.end_snapshot IS NULL";

/// Check that every column the `duckdb` keys of `spec` reference binds to a
/// column of `schema`, as official's `ValidateSortExpressionColumns` checks
/// whenever it plans a sort, before any row is read. Keys naming a missing
/// column fail with official's binder error ([`missing_sort_columns`]);
/// several columns matching a key by case alone fail with the ambiguity error
/// of [`bind_sort_column`], and a qualified column reference (`t.id`) with
/// official's error. A key DataFusion cannot parse is not checked.
#[cfg(feature = "write")]
pub(crate) fn validate_sort_columns(
    spec: &SortSpec,
    schema: &arrow::datatypes::Schema,
) -> crate::Result<()> {
    let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    let mut missing = Vec::new();
    for field in spec.fields.iter().filter(|field| field.is_duckdb_dialect()) {
        for (column, qualified) in referenced_columns(&field.expression).unwrap_or_default() {
            if qualified {
                return Err(crate::DuckLakeError::InvalidConfig(
                    "Unexpected qualified column reference - only unqualified columns are \
                     supported"
                        .to_string(),
                ));
            }
            if bind_sort_column(names.iter().copied(), &column)?.is_none() {
                note_missing(&mut missing, &column);
            }
        }
    }
    if !missing.is_empty() {
        return Err(missing_sort_columns(&missing));
    }
    Ok(())
}

/// The error for sort keys naming columns the table lacks, worded as
/// official's binder words it (`ValidateSortExpressionColumns`).
pub(crate) fn missing_sort_columns(missing: &[String]) -> crate::DuckLakeError {
    crate::DuckLakeError::InvalidConfig(format!(
        "Columns in the SET SORTED BY statement were not found in the DuckLake table. Unmatched \
         columns were: {}",
        missing.join(", ")
    ))
}

/// Note `name` among the sort keys' `missing` columns, once.
pub(crate) fn note_missing(missing: &mut Vec<String>, name: &str) {
    if !missing.iter().any(|known| known == name) {
        missing.push(name.to_string());
    }
}

/// Parse an expression string as a single bare column reference. Returns the column
/// name (with surrounding double quotes stripped) or `None` if it is not a lone
/// identifier. Deliberately conservative: only what we can safely map to one Arrow
/// column.
fn parse_bare_column(expr: &str) -> Option<String> {
    let trimmed = expr.trim();
    if trimmed.is_empty() {
        return None;
    }
    // Double-quoted identifier: "..." with doubled "" escapes inside.
    if let Some(inner) = trimmed.strip_prefix('"').and_then(|s| s.strip_suffix('"'))
        && !inner.is_empty()
    {
        if inner.replace("\"\"", "").contains('"') {
            return None;
        }
        return Some(inner.replace("\"\"", "\""));
    }
    // Unquoted simple identifier: [A-Za-z_][A-Za-z0-9_]*
    let mut chars = trimmed.chars();
    let first = chars.next()?;
    if !(first.is_ascii_alphabetic() || first == '_') {
        return None;
    }
    if trimmed
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_')
    {
        Some(trimmed.to_string())
    } else {
        None
    }
}

/// A table's active sort spec (one generation of `ducklake_sort_info`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SortSpec {
    /// `ducklake_sort_info.sort_id` for this spec generation.
    pub sort_id: i64,
    /// Sort keys, ordered by `sort_key_index` (primary key first).
    pub fields: Vec<SortField>,
}

impl SortSpec {
    /// Whether every key is a bare column reference, the keys `SET SORTED BY`
    /// through this crate records. A spec with any other expression is still
    /// applied on write, evaluated with DataFusion.
    pub fn is_producible(&self) -> bool {
        !self.fields.is_empty() && self.fields.iter().all(|f| f.column_candidate().is_some())
    }

    /// The sort keys of the `duckdb` dialect as `(column_name, direction,
    /// null_order)`, in order, or `None` if any of them is not a bare column
    /// (see [`SortSpec::is_producible`]).
    pub fn producible_columns(&self) -> Option<Vec<(String, SortDirection, NullOrder)>> {
        // Skip fields whose dialect is not `duckdb`, matching official DuckLake, which
        // `continue`s past them when building the sort expression
        // (`ducklake_sort_data.cpp`). Their expression is written in some other
        // engine's dialect, so evaluating it here would sort by something official
        // ignores entirely.
        self.fields
            .iter()
            .filter(|f| f.is_duckdb_dialect())
            .map(|f| f.column_candidate().map(|c| (c, f.direction, f.null_order)))
            .collect()
    }

    /// Build a spec from catalog rows `(sort_id, sort_key_index, expression, dialect,
    /// sort_direction, null_order)` — the join of `ducklake_sort_info` and
    /// `ducklake_sort_expression` for the single LIVE generation, ordered by
    /// `sort_key_index`. Returns `None` when there are no rows (unsorted). Every row
    /// is expected to carry the same `sort_id`; the first row's id is used.
    pub fn from_rows(rows: Vec<(i64, i32, String, String, String, String)>) -> Option<SortSpec> {
        let sort_id = rows.first()?.0;
        let fields = rows
            .into_iter()
            .map(
                |(_, sort_key_index, expression, dialect, sort_direction, null_order)| SortField {
                    sort_key_index,
                    expression,
                    dialect,
                    direction: SortDirection::parse(&sort_direction),
                    null_order: NullOrder::parse(&null_order),
                },
            )
            .collect();
        Some(SortSpec {
            sort_id,
            fields,
        })
    }
}

/// The session sort-key expressions are planned with. Identifiers keep their
/// case, so a key names its column exactly as the catalog does.
#[cfg(feature = "write")]
pub(crate) fn sort_key_context() -> datafusion::prelude::SessionContext {
    let mut config = datafusion::prelude::SessionConfig::new();
    config.options_mut().sql_parser.enable_ident_normalization = false;
    datafusion::prelude::SessionContext::new_with_config(config)
}

/// Plan sort expression `sql` over columns `names` (`df_schema`) with
/// `context` ([`sort_key_context`]), each identifier bound to the column
/// DuckDB's binder binds it to ([`bind_sort_column`]). An identifier naming no
/// column is noted in `missing`. `None` when DataFusion cannot parse or plan
/// the expression, or a column is missing.
///
/// The expression is evaluated with DataFusion's semantics, not DuckDB's:
/// where they differ (integer division, overflow, type promotion, function
/// behaviour) the rows may be ordered differently than official orders them.
/// The order is layout only, so query results are unaffected.
///
/// DuckDB writes some function calls qualified by its default schema
/// (`SET SORTED BY ([a, b])` stores `main.list_value(a, b)`, `extract(day
/// FROM ts)` stores `main.date_part('day', ts)`); the `main.` qualifier is
/// dropped so the function resolves by name in DataFusion. A function
/// DataFusion lacks (`list_value`, `struct_pack`, `row`) still fails to plan.
#[cfg(feature = "write")]
pub(crate) fn plan_sort_expression(
    context: &datafusion::prelude::SessionContext,
    sql: &str,
    names: &[&str],
    df_schema: &datafusion::common::DFSchema,
    missing: &mut Vec<String>,
) -> crate::Result<Option<datafusion::logical_expr::Expr>> {
    use datafusion::sql::sqlparser::ast::{
        Expr as SqlExpr, Ident, ObjectNamePart, visit_expressions_mut,
    };
    use std::ops::ControlFlow;

    let state = context.state();
    let dialect = state.config().options().sql_parser.dialect;
    let Ok(mut parsed) = state.sql_to_expr_with_alias(sql, &dialect) else {
        return Ok(None);
    };
    let before = missing.len();
    let mut failed = None;
    let _ = visit_expressions_mut(&mut parsed.expr, |node| {
        if let SqlExpr::Function(function) = node
            && let [ObjectNamePart::Identifier(schema), _] = function.name.0.as_slice()
            && schema.value.eq_ignore_ascii_case("main")
        {
            function.name.0.remove(0);
        }
        if let SqlExpr::Identifier(ident) = node {
            match bind_sort_column(names.iter().copied(), &ident.value) {
                // Quoted, so it names the column exactly as spelled.
                Ok(Some(index)) => *ident = Ident::with_quote('"', names[index]),
                Ok(None) => note_missing(missing, &ident.value),
                Err(error) => {
                    failed = Some(error);
                    return ControlFlow::Break(());
                },
            }
        }
        ControlFlow::Continue(())
    });
    if let Some(error) = failed {
        return Err(error);
    }
    if missing.len() > before {
        return Ok(None);
    }
    Ok(state
        .create_logical_expr_from_sql_expr(parsed, df_schema)
        .ok())
}

/// `values` as DuckDB orders them, for a sort key: every float `nan` is the
/// positive one (DuckDB sorts every `nan` after `inf`, whatever its sign
/// bit) and `-0.0` is `0.0` (DuckDB sorts them as equal). Any other array is
/// returned as is.
#[cfg(feature = "write")]
pub(crate) fn duckdb_sort_values(values: &arrow::array::ArrayRef) -> arrow::array::ArrayRef {
    use arrow::array::{ArrayRef, AsArray};
    use arrow::datatypes::{ArrowPrimitiveType, DataType, Float16Type, Float32Type, Float64Type};
    use std::sync::Arc;
    type F16 = <Float16Type as ArrowPrimitiveType>::Native;
    let canonical: ArrayRef = match values.data_type() {
        DataType::Float64 => Arc::new(
            values
                .as_primitive::<Float64Type>()
                .unary::<_, Float64Type>(|value| {
                    if value.is_nan() {
                        f64::NAN
                    } else if value == 0.0 {
                        0.0
                    } else {
                        value
                    }
                }),
        ),
        DataType::Float32 => Arc::new(
            values
                .as_primitive::<Float32Type>()
                .unary::<_, Float32Type>(|value| {
                    if value.is_nan() {
                        f32::NAN
                    } else if value == 0.0 {
                        0.0
                    } else {
                        value
                    }
                }),
        ),
        DataType::Float16 => Arc::new(
            values
                .as_primitive::<Float16Type>()
                .unary::<_, Float16Type>(|value| {
                    if value.is_nan() {
                        F16::NAN
                    } else if value == F16::ZERO || value == F16::NEG_ZERO {
                        F16::ZERO
                    } else {
                        value
                    }
                }),
        ),
        _ => Arc::clone(values),
    };
    canonical
}

/// `expr`, a float, as a sort key ordering as DuckDB orders a float column
/// ([`duckdb_sort_values`]).
#[cfg(feature = "write")]
fn float_sort_key(expr: datafusion::logical_expr::Expr) -> datafusion::logical_expr::Expr {
    datafusion::logical_expr::ScalarUDF::new_from_impl(FloatSortKey::new()).call(vec![expr])
}

/// A float sort key's values as DuckDB orders them ([`duckdb_sort_values`]).
#[cfg(feature = "write")]
#[derive(Debug, PartialEq, Eq, Hash)]
struct FloatSortKey {
    signature: datafusion::logical_expr::Signature,
}

#[cfg(feature = "write")]
impl FloatSortKey {
    fn new() -> Self {
        Self {
            signature: datafusion::logical_expr::Signature::any(
                1,
                datafusion::logical_expr::Volatility::Immutable,
            ),
        }
    }
}

#[cfg(feature = "write")]
impl datafusion::logical_expr::ScalarUDFImpl for FloatSortKey {
    fn name(&self) -> &str {
        "ducklake_float_sort_key"
    }

    fn signature(&self) -> &datafusion::logical_expr::Signature {
        &self.signature
    }

    fn return_type(
        &self,
        arg_types: &[arrow::datatypes::DataType],
    ) -> datafusion::common::Result<arrow::datatypes::DataType> {
        Ok(arg_types[0].clone())
    }

    fn invoke_with_args(
        &self,
        args: datafusion::logical_expr::ScalarFunctionArgs,
    ) -> datafusion::common::Result<datafusion::logical_expr::ColumnarValue> {
        use datafusion::logical_expr::ColumnarValue;
        Ok(match &args.args[0] {
            ColumnarValue::Array(values) => ColumnarValue::Array(duckdb_sort_values(values)),
            ColumnarValue::Scalar(value) => {
                ColumnarValue::Scalar(datafusion::common::ScalarValue::try_from_array(
                    &duckdb_sort_values(&value.to_array()?),
                    0,
                )?)
            },
        })
    }
}

/// The physical sort expressions of `spec` over rows of `schema`: one per key
/// of the `duckdb` dialect, as official skips the others
/// (`ducklake_sort_data.cpp`); a bare column is read by position, and any
/// other expression is planned with DataFusion ([`plan_sort_expression`]).
/// Columns bind as DuckDB's binder binds them ([`bind_sort_column`]). Empty
/// when no key is left, or when DataFusion cannot plan one of them: the rows
/// then keep the caller's order, as the order is layout only. A bare float
/// column sorts as DuckDB sorts it (`nan` last, `-0.0` equal to `0.0`).
///
/// A key naming a column `schema` lacks fails with official's binder error
/// ([`missing_sort_columns`]); several columns matching a key by case alone
/// fail with the ambiguity error of [`bind_sort_column`].
#[cfg(feature = "write")]
pub(crate) fn sort_exprs(
    spec: &SortSpec,
    schema: &arrow::datatypes::Schema,
) -> crate::Result<Vec<datafusion::physical_expr::PhysicalSortExpr>> {
    use datafusion::common::DFSchema;
    use datafusion::logical_expr::Expr;
    use datafusion::physical_expr::{PhysicalExpr, PhysicalSortExpr};
    use std::sync::Arc;

    enum Key {
        Column(usize),
        Expression(Expr),
    }

    let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    let mut context = None;
    let mut keys = Vec::with_capacity(spec.fields.len());
    let mut missing = Vec::new();
    let mut plannable = true;
    for field in spec.fields.iter().filter(|field| field.is_duckdb_dialect()) {
        let options = arrow::compute::SortOptions {
            descending: matches!(field.direction, SortDirection::Desc),
            nulls_first: field.null_order.nulls_first(),
        };
        if let Some(name) = field.column_candidate() {
            match bind_sort_column(names.iter().copied(), &name)? {
                // A float column sorts as DuckDB sorts it: `nan` last, `-0.0`
                // equal to `0.0`.
                Some(index) if schema.field(index).data_type().is_floating() => {
                    if context.is_none() {
                        context = Some((sort_key_context(), DFSchema::try_from(schema.clone())?));
                    }
                    let column =
                        Expr::Column(datafusion::common::Column::new_unqualified(names[index]));
                    keys.push((Key::Expression(float_sort_key(column)), options));
                },
                Some(index) => keys.push((Key::Column(index), options)),
                None => note_missing(&mut missing, &name),
            }
            continue;
        }
        let (context, df_schema) = match &context {
            Some(planned) => planned,
            None => context.insert((sort_key_context(), DFSchema::try_from(schema.clone())?)),
        };
        match plan_sort_expression(context, &field.expression, &names, df_schema, &mut missing)? {
            Some(expr) => keys.push((Key::Expression(expr), options)),
            None => plannable = false,
        }
    }
    if !missing.is_empty() {
        return Err(missing_sort_columns(&missing));
    }
    if !plannable {
        return Ok(Vec::new());
    }
    let mut out = Vec::with_capacity(keys.len());
    for (key, options) in keys {
        let expr: Arc<dyn PhysicalExpr> = match key {
            Key::Column(index) => Arc::new(datafusion::physical_expr::expressions::Column::new(
                names[index],
                index,
            )),
            Key::Expression(expr) => {
                let (context, df_schema) = context.as_ref().expect("planned with its key");
                match context.create_physical_expr(expr, df_schema) {
                    Ok(physical) => physical,
                    Err(_) => return Ok(Vec::new()),
                }
            },
        };
        out.push(PhysicalSortExpr::new(expr, options));
    }
    Ok(out)
}

/// Reorder rows by a table's sort order, returning ONE sorted batch.
///
/// `batches` may carry trailing columns beyond `data_schema` (a compaction output
/// embeds the rowid, and for a partial file the per-row snapshot id); sort keys
/// ([`sort_exprs`]) resolve to `data_schema` positions — the leading columns — so
/// those trailing columns travel with their rows.
///
/// This is a **global** sort across all of `batches`, matching official DuckLake,
/// which sorts a write by planting a blocking `PhysicalOrder` above the insert plan
/// (`ducklake_insert.cpp`). That is what makes successive rolled files cover
/// contiguous, non-overlapping value ranges — the property that lets a reader skip
/// whole files. Sorting *within* a file would not achieve it: a file's min/max is
/// the min/max of its rows regardless of their order, so only its row-group bounds
/// would tighten. Rows with equal keys keep their order.
///
/// Returns `batches` unchanged when there is no sort order, no key of the
/// `duckdb` dialect, no rows, or a key DataFusion cannot plan or fails to
/// evaluate over the rows (a division by zero, a failing cast): the order is a
/// layout preference, so the rows keep their arrival order rather than fail
/// the write.
#[cfg(feature = "write")]
pub(crate) fn sort_batches_by_spec(
    batches: Vec<arrow::record_batch::RecordBatch>,
    data_schema: &arrow::datatypes::Schema,
    sort_spec: Option<&SortSpec>,
) -> crate::Result<Vec<arrow::record_batch::RecordBatch>> {
    use arrow::array::{ArrayRef, RecordBatch};
    use std::sync::Arc;

    let Some(sort_spec) = sort_spec else {
        return Ok(batches);
    };
    if batches.iter().all(|b| b.num_rows() == 0) {
        return Ok(batches);
    }
    let keys = sort_exprs(sort_spec, data_schema)?;
    if keys.is_empty() {
        return Ok(batches);
    }
    let full_schema = batches[0].schema();
    let combined = arrow::compute::concat_batches(&full_schema, &batches)?;
    let rows = combined.num_rows();
    let mut sort_columns = Vec::with_capacity(keys.len() + 1);
    for key in &keys {
        let Ok(values) = key
            .expr
            .evaluate(&combined)
            .and_then(|value| value.into_array(rows))
        else {
            return Ok(batches);
        };
        sort_columns.push(arrow::compute::SortColumn {
            values,
            options: Some(key.options),
        });
    }
    // The row's position breaks ties, so the sort is stable.
    sort_columns.push(arrow::compute::SortColumn {
        values: Arc::new(arrow::array::UInt64Array::from_iter_values(0..rows as u64)),
        options: None,
    });
    let indices = arrow::compute::lexsort_to_indices(&sort_columns, None)?;
    let sorted_columns = combined
        .columns()
        .iter()
        .map(|c| arrow::compute::take(c, &indices, None))
        .collect::<std::result::Result<Vec<ArrayRef>, _>>()?;
    let sorted_columns = sorted_columns
        .iter()
        .zip(full_schema.fields())
        .map(|(column, field)| crate::column_rename::coerce_column(column, field.data_type()))
        .collect::<datafusion::common::Result<Vec<_>>>()?;
    Ok(vec![RecordBatch::try_new(full_schema, sorted_columns)?])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direction_roundtrip_and_case_insensitive() {
        assert_eq!(SortDirection::parse("ASC"), SortDirection::Asc);
        assert_eq!(SortDirection::parse("desc"), SortDirection::Desc);
        assert_eq!(SortDirection::parse("DESC"), SortDirection::Desc);
        // DuckLake: anything that isn't DESC is ASC.
        assert_eq!(SortDirection::parse("whatever"), SortDirection::Asc);
        assert_eq!(SortDirection::Asc.to_catalog_string(), "ASC");
        assert_eq!(SortDirection::Desc.to_catalog_string(), "DESC");
    }

    #[test]
    fn null_order_roundtrip_and_case_insensitive() {
        assert_eq!(NullOrder::parse("NULLS_FIRST"), NullOrder::NullsFirst);
        assert_eq!(NullOrder::parse("nulls_first"), NullOrder::NullsFirst);
        assert_eq!(NullOrder::parse("NULLS_LAST"), NullOrder::NullsLast);
        // DuckLake: anything that isn't NULLS_FIRST is NULLS_LAST.
        assert_eq!(NullOrder::parse("anything"), NullOrder::NullsLast);
        assert_eq!(NullOrder::NullsFirst.to_catalog_string(), "NULLS_FIRST");
        assert_eq!(NullOrder::NullsLast.to_catalog_string(), "NULLS_LAST");
        assert!(NullOrder::NullsFirst.nulls_first());
        assert!(!NullOrder::NullsLast.nulls_first());
    }

    #[test]
    fn bare_column_expressions_are_producible() {
        assert_eq!(parse_bare_column("ts"), Some("ts".to_string()));
        assert_eq!(
            parse_bare_column("  device_id "),
            Some("device_id".to_string())
        );
        assert_eq!(parse_bare_column("_x1"), Some("_x1".to_string()));
        // double-quoted identifier with spaces
        assert_eq!(parse_bare_column("\"My Col\""), Some("My Col".to_string()));
        // a doubled quote inside is one quote
        assert_eq!(parse_bare_column("\"a\"\"b\""), Some("a\"b".to_string()));
        assert_eq!(parse_bare_column("\"a\"b\""), None);
    }

    #[test]
    fn column_keys_are_written_as_duckdb_writes_column_references() {
        let key = |name: &str| {
            SortField::column(0, name, SortDirection::Asc, NullOrder::NullsLast).expression
        };
        // As the DuckDB 1.5.5 CLI stores `SET SORTED BY (...)`.
        assert_eq!(key("id"), "id");
        assert_eq!(key("ID"), "ID");
        assert_eq!(key("x1"), "x1");
        assert_eq!(key("My Col"), "\"My Col\"");
        assert_eq!(key("name"), "\"name\"");
        assert_eq!(key("Select"), "\"Select\"");
        assert_eq!(key("1x"), "\"1x\"");
        assert_eq!(key("a\"b"), "\"a\"\"b\"");
        for name in ["id", "My Col", "name", "a\"b"] {
            let field = SortField::column(0, name, SortDirection::Asc, NullOrder::NullsLast);
            assert_eq!(field.column_candidate().as_deref(), Some(name));
        }
    }

    #[cfg(feature = "write")]
    #[test]
    fn set_sort_spec_names_every_missing_column() {
        let names = vec!["id".to_string(), "v".to_string()];
        let fields = |keys: &[&str]| -> Vec<SortField> {
            keys.iter()
                .enumerate()
                .map(|(index, key)| {
                    SortField::column(index as i32, *key, SortDirection::Asc, NullOrder::NullsLast)
                })
                .collect()
        };
        assert!(check_sort_key_columns(&fields(&["ID", "v"]), &names).is_ok());
        let error = check_sort_key_columns(&fields(&["a", "id", "b", "a"]), &names)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains(
                "Columns in the SET SORTED BY statement were not found in the DuckLake table. \
                 Unmatched columns were: a, b"
            ),
            "{error}"
        );
    }

    #[cfg(feature = "write")]
    #[test]
    fn dropping_a_referenced_column_is_refused() {
        let keys =
            |expression: &str, dialect: &str| vec![(expression.to_string(), dialect.to_string())];
        for expression in ["v", "V", "\"v\"", "upper(V)", "main.lower((v || id))"] {
            let error = refuse_dropping_sort_columns(&keys(expression, "duckdb"), &["id", "v"])
                .unwrap_err()
                .to_string();
            assert!(
                error.contains("Cannot drop column"),
                "{expression}: {error}"
            );
        }
        // A qualified reference names its last part, as DuckDB's
        // `ColumnRefExpression::GetColumnName` does.
        assert!(refuse_dropping_sort_columns(&keys("(t.v + 1)", "duckdb"), &["V"]).is_err());
        assert!(refuse_dropping_sort_columns(&keys("(t.v + 1)", "duckdb"), &["t"]).is_ok());
        // A key that does not parse might reference any column.
        let error = refuse_dropping_sort_columns(&keys("v +* (", "duckdb"), &["w"])
            .unwrap_err()
            .to_string();
        assert!(error.contains("could not be parsed"), "{error}");
        assert!(refuse_dropping_sort_columns(&keys("v +* (", "duckdb"), &[]).is_ok());
        assert!(refuse_dropping_sort_columns(&keys("upper(V)", "duckdb"), &["w"]).is_ok());
        assert!(refuse_dropping_sort_columns(&keys("v", "spark"), &["v"]).is_ok());
        assert!(refuse_dropping_sort_columns(&[], &["v"]).is_ok());
    }

    #[cfg(feature = "write")]
    #[test]
    fn validation_reads_the_columns_expressions_reference() {
        use arrow::datatypes::{DataType, Field, Schema};
        let schema = Schema::new(vec![
            Field::new("id", DataType::Int64, true),
            Field::new("v", DataType::Utf8, true),
        ]);
        let spec = |expression: &str, dialect: &str| SortSpec {
            sort_id: 1,
            fields: vec![SortField {
                sort_key_index: 0,
                expression: expression.to_string(),
                dialect: dialect.to_string(),
                direction: SortDirection::Asc,
                null_order: NullOrder::NullsLast,
            }],
        };
        for ok in ["id", "ID", "(id % 2)", "main.abs((id - 2))", "no_such_macro(v)"] {
            assert!(
                validate_sort_columns(&spec(ok, "duckdb"), &schema).is_ok(),
                "{ok}"
            );
        }
        let error = validate_sort_columns(&spec("(gone + other + gone)", "duckdb"), &schema)
            .unwrap_err()
            .to_string();
        assert!(
            error.ends_with("Unmatched columns were: gone, other"),
            "{error}"
        );
        assert!(validate_sort_columns(&spec("gone", "spark"), &schema).is_ok());
        // Official refuses a qualified reference (checked with the DuckDB CLI:
        // `SET SORTED BY (t.v + 1)`).
        let error = validate_sort_columns(&spec("(t.v + 1)", "duckdb"), &schema)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains(
                "Unexpected qualified column reference - only unqualified columns are supported"
            ),
            "{error}"
        );
    }

    #[cfg(feature = "write")]
    #[test]
    fn main_qualified_functions_plan_by_their_name() {
        use arrow::datatypes::{DataType, Field, Schema};
        let schema = Schema::new(vec![Field::new("id", DataType::Int64, true)]);
        let key = |expression: &str| SortSpec {
            sort_id: 1,
            fields: vec![SortField {
                sort_key_index: 0,
                expression: expression.to_string(),
                dialect: DUCKDB_DIALECT.to_string(),
                direction: SortDirection::Asc,
                null_order: NullOrder::NullsLast,
            }],
        };
        assert_eq!(
            sort_exprs(&key("main.abs((id - 2))"), &schema)
                .unwrap()
                .len(),
            1
        );
        assert_eq!(sort_exprs(&key("MAIN.abs(id)"), &schema).unwrap().len(), 1);
        // Functions DataFusion lacks keep the caller's order.
        assert!(
            sort_exprs(&key("main.list_value(id)"), &schema)
                .unwrap()
                .is_empty()
        );
        assert!(
            sort_exprs(&key("other.abs(id)"), &schema)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn non_column_expressions_are_not_producible() {
        for expr in ["", "date_trunc('day', ts)", "a + b", "t.ts", "1", "ts, device_id", "\"\""] {
            assert_eq!(
                parse_bare_column(expr),
                None,
                "expr {expr:?} should not be a bare column"
            );
        }
    }

    #[test]
    fn producible_only_when_all_keys_are_columns() {
        let ok = SortSpec {
            sort_id: 1,
            fields: vec![
                SortField::column(0, "device_id", SortDirection::Asc, NullOrder::NullsLast),
                SortField::column(1, "ts", SortDirection::Desc, NullOrder::NullsFirst),
            ],
        };
        assert!(ok.is_producible());
        assert_eq!(
            ok.producible_columns().unwrap(),
            vec![
                (
                    "device_id".to_string(),
                    SortDirection::Asc,
                    NullOrder::NullsLast
                ),
                ("ts".to_string(), SortDirection::Desc, NullOrder::NullsFirst),
            ]
        );

        let mixed = SortSpec {
            sort_id: 2,
            fields: vec![
                SortField::column(0, "device_id", SortDirection::Asc, NullOrder::NullsLast),
                SortField {
                    sort_key_index: 1,
                    expression: "date_trunc('day', ts)".to_string(),
                    dialect: DUCKDB_DIALECT.to_string(),
                    direction: SortDirection::Asc,
                    null_order: NullOrder::NullsLast,
                },
            ],
        };
        assert!(!mixed.is_producible());
        assert_eq!(mixed.producible_columns(), None);
    }

    #[test]
    fn from_rows_orders_and_parses() {
        let rows = vec![
            (
                7,
                0,
                "device_id".to_string(),
                "duckdb".to_string(),
                "ASC".to_string(),
                "NULLS_LAST".to_string(),
            ),
            (
                7,
                1,
                "ts".to_string(),
                "duckdb".to_string(),
                "DESC".to_string(),
                "NULLS_FIRST".to_string(),
            ),
        ];
        let spec = SortSpec::from_rows(rows).unwrap();
        assert_eq!(spec.sort_id, 7);
        assert_eq!(spec.fields.len(), 2);
        assert_eq!(spec.fields[0].direction, SortDirection::Asc);
        assert_eq!(spec.fields[0].null_order, NullOrder::NullsLast);
        assert_eq!(spec.fields[1].direction, SortDirection::Desc);
        assert_eq!(spec.fields[1].null_order, NullOrder::NullsFirst);
        assert!(spec.is_producible());
    }

    #[test]
    fn sort_keys_bind_as_duckdb_binds_columns() {
        let bind = |names: &[&str], key: &str| bind_sort_column(names.iter().copied(), key);
        assert_eq!(bind(&["id", "v"], "id").unwrap(), Some(0));
        assert_eq!(bind(&["id", "v"], "ID").unwrap(), Some(0));
        assert_eq!(bind(&["ID", "v"], "id").unwrap(), Some(0));
        assert_eq!(bind(&["id", "v"], "gone").unwrap(), None);
        // Exact spelling first, among columns equal ignoring case.
        assert_eq!(bind(&["id", "ID"], "ID").unwrap(), Some(1));
        assert_eq!(bind(&["ID", "id"], "id").unwrap(), Some(1));
        assert!(bind(&["id", "ID"], "Id").is_err());
        // ASCII case only, as DuckDB folds identifiers.
        assert_eq!(bind(&["é"], "É").unwrap(), None);
    }

    #[test]
    fn dialect_matches_exactly() {
        let mut field = SortField::column(0, "id", SortDirection::Asc, NullOrder::NullsLast);
        assert!(field.is_duckdb_dialect());
        field.dialect = "DuckDB".to_string();
        assert!(!field.is_duckdb_dialect());
    }

    #[test]
    fn from_rows_empty_is_none() {
        assert_eq!(SortSpec::from_rows(vec![]), None);
    }

    #[cfg(feature = "write")]
    #[test]
    fn sort_batches_preserves_nested_field_metadata() {
        use std::{collections::HashMap, sync::Arc};

        use arrow::{
            array::{ArrayRef, Int32Array, Int64Array, StringArray, StructArray},
            datatypes::{DataType, Field, Schema},
            record_batch::RecordBatch,
        };

        let field_id =
            |value: &str| HashMap::from([("PARQUET:field_id".to_string(), value.to_string())]);
        let money_fields = vec![
            Arc::new(Field::new("amount", DataType::Int32, false).with_metadata(field_id("2"))),
            Arc::new(Field::new("currency", DataType::Utf8, false).with_metadata(field_id("3"))),
        ];
        let money: ArrayRef = Arc::new(StructArray::new(
            money_fields.clone().into(),
            vec![
                Arc::new(Int32Array::from(vec![20, 10])),
                Arc::new(StringArray::from(vec!["EUR", "USD"])),
            ],
            None,
        ));
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("money", DataType::Struct(money_fields.into()), false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from(vec![2, 1])), money],
        )
        .unwrap();
        let sort_spec = SortSpec {
            sort_id: 1,
            fields: vec![SortField::column(0, "id", SortDirection::Asc, NullOrder::NullsLast)],
        };

        let sorted = sort_batches_by_spec(vec![batch], &schema, Some(&sort_spec)).unwrap();

        assert_eq!(
            sorted[0]
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .values(),
            &[1, 2]
        );
        assert_eq!(
            sorted[0].schema().field(1).data_type(),
            sorted[0].column(1).data_type()
        );
        let DataType::Struct(fields) = sorted[0].column(1).data_type() else {
            panic!("money must remain a struct");
        };
        assert_eq!(
            fields[0].metadata().get("PARQUET:field_id"),
            Some(&"2".into())
        );
        assert_eq!(
            fields[1].metadata().get("PARQUET:field_id"),
            Some(&"3".into())
        );
    }

    /// One key over column `x` of `schema`, ascending.
    #[cfg(feature = "write")]
    fn sort_by(
        expression: &str,
        schema: &arrow::datatypes::SchemaRef,
        column: arrow::array::ArrayRef,
    ) -> crate::Result<Vec<arrow::record_batch::RecordBatch>> {
        let rows = column.len();
        let batch = arrow::record_batch::RecordBatch::try_new(
            schema.clone(),
            vec![
                column,
                std::sync::Arc::new(arrow::array::Int64Array::from_iter_values(0..rows as i64)),
            ],
        )
        .unwrap();
        let spec = SortSpec {
            sort_id: 1,
            fields: vec![SortField {
                sort_key_index: 0,
                expression: expression.to_string(),
                dialect: DUCKDB_DIALECT.to_string(),
                direction: SortDirection::Asc,
                null_order: NullOrder::NullsLast,
            }],
        };
        sort_batches_by_spec(vec![batch], schema, Some(&spec))
    }

    /// The `pos` column of sorted rows: each row's input position.
    #[cfg(feature = "write")]
    fn positions(sorted: &[arrow::record_batch::RecordBatch]) -> Vec<i64> {
        sorted
            .iter()
            .flat_map(|batch| {
                batch
                    .column(1)
                    .as_any()
                    .downcast_ref::<arrow::array::Int64Array>()
                    .unwrap()
                    .values()
                    .to_vec()
            })
            .collect()
    }

    #[cfg(feature = "write")]
    fn schema_of(data_type: arrow::datatypes::DataType) -> arrow::datatypes::SchemaRef {
        use arrow::datatypes::{DataType, Field, Schema};
        std::sync::Arc::new(Schema::new(vec![
            Field::new("x", data_type, true),
            Field::new("pos", DataType::Int64, false),
        ]))
    }

    /// A bare float column sorts as DuckDB sorts it: every `nan` after `inf`
    /// whatever its sign bit, and `-0.0` equal to `0.0`, so ties keep their
    /// order (checked with the DuckDB CLI on `ORDER BY x` over `nan`, `-nan`,
    /// `-0.0`, `0.0`, `inf`, `-inf`, `NULL`). Guards against Arrow's total
    /// order, which puts a negative `nan` first and `-0.0` before `0.0`.
    #[cfg(feature = "write")]
    #[test]
    fn bare_float_columns_sort_as_duckdb() {
        use arrow::array::{Float32Array, Float64Array};
        use arrow::datatypes::DataType;
        use std::sync::Arc;
        let values = [
            Some(f64::NAN),
            Some(-f64::NAN),
            Some(0.0),
            Some(-0.0),
            Some(f64::INFINITY),
            Some(f64::NEG_INFINITY),
            None,
            Some(1.0),
            Some(-0.0),
        ];
        let expected = vec![5, 2, 3, 8, 7, 4, 0, 1, 6];
        let schema = schema_of(DataType::Float64);
        let sorted = sort_by("x", &schema, Arc::new(Float64Array::from(values.to_vec()))).unwrap();
        assert_eq!(positions(&sorted), expected);
        let schema = schema_of(DataType::Float32);
        let column = Float32Array::from(
            values
                .iter()
                .map(|value| value.map(|value| value as f32))
                .collect::<Vec<_>>(),
        );
        let sorted = sort_by("x", &schema, Arc::new(column)).unwrap();
        assert_eq!(positions(&sorted), expected);
    }

    /// An expression key is evaluated with DataFusion's semantics: `x / 2`
    /// of integers is integer division (DuckDB's would divide as `DOUBLE`),
    /// so `3` and `2` tie, as do `1` and `0`, and ties keep their order.
    #[cfg(feature = "write")]
    #[test]
    fn expression_keys_evaluate_with_datafusion() {
        use arrow::array::Int64Array;
        use arrow::datatypes::DataType;
        use std::sync::Arc;
        let schema = schema_of(DataType::Int64);
        let column = Arc::new(Int64Array::from(vec![3, 1, 2, 0]));
        let sorted = sort_by("x / 2", &schema, column.clone()).unwrap();
        assert_eq!(positions(&sorted), vec![1, 3, 0, 2]);
        // DuckDB's `main.` function qualifier is dropped.
        let sorted = sort_by("main.abs((x - 2))", &schema, column).unwrap();
        assert_eq!(positions(&sorted), vec![2, 0, 1, 3]);
    }

    /// A key that fails over the rows (a division by zero, a failing cast) or that
    /// DataFusion cannot plan leaves the rows in their arrival order instead
    /// of failing the write.
    #[cfg(feature = "write")]
    #[test]
    fn failing_expression_keys_keep_arrival_order() {
        use arrow::array::{Int64Array, StringArray};
        use arrow::datatypes::DataType;
        use std::sync::Arc;
        let schema = schema_of(DataType::Int64);
        let column = Arc::new(Int64Array::from(vec![3, 0, 1]));
        for key in ["x / 0", "x % (x - 1)", "main.list_value(x)", "x ** 2", "x // 2"] {
            let sorted = sort_by(key, &schema, column.clone()).unwrap();
            assert_eq!(positions(&sorted), vec![0, 1, 2], "{key}");
        }
        let schema = schema_of(DataType::Utf8);
        let column = Arc::new(StringArray::from(vec!["3", "abc", "1"]));
        let sorted = sort_by("CAST(x AS BIGINT)", &schema, column).unwrap();
        assert_eq!(positions(&sorted), vec![0, 1, 2]);
        // A missing column still fails, as official's binder does.
        let schema = schema_of(DataType::Int64);
        let error = sort_by("gone + 1", &schema, Arc::new(Int64Array::from(vec![1])))
            .unwrap_err()
            .to_string();
        assert!(error.contains("Unmatched columns were: gone"), "{error}");
    }
}
