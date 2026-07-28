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
//! Scope note: this crate *produces* sort orders only for **bare column references**
//! (`SORTED BY (device_id, ts DESC)`). A spec whose expression is anything more
//! complex is *tolerated on read* (round-tripped verbatim) but never applied on our
//! writes — we simply write unsorted, which is always correct (only the min/max
//! locality, i.e. pruning effectiveness, is affected). This mirrors how
//! [`crate::partition`] tolerates `bucket`/unknown transforms on read but never
//! produces them.

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
/// and the write path skips sorting.
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
    /// column name; dialect is set to [`DUCKDB_DIALECT`].
    pub fn column(
        sort_key_index: i32,
        column: impl Into<String>,
        direction: SortDirection,
        null_order: NullOrder,
    ) -> Self {
        SortField {
            sort_key_index,
            expression: column.into(),
            dialect: DUCKDB_DIALECT.to_string(),
            direction,
            null_order,
        }
    }

    /// Interpret this key's expression as a bare column reference, returning the
    /// column name if it is one. A bare column is either an unquoted simple
    /// identifier (`ts`, `device_id`) or a double-quoted identifier (`"My Col"`,
    /// unquoted here). Anything else — function calls, arithmetic, qualified names,
    /// multiple tokens — yields `None`, so the write path leaves such files unsorted
    /// rather than guessing.
    pub fn column_candidate(&self) -> Option<String> {
        parse_bare_column(&self.expression)
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
        && !inner.contains('"')
    {
        return Some(inner.to_string());
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
    /// Whether this crate can *apply* this sort on write: true only when every key
    /// is a bare column reference. A spec containing any non-column expression is
    /// not producible (we tolerate it on read but write such files unsorted).
    pub fn is_producible(&self) -> bool {
        !self.fields.is_empty() && self.fields.iter().all(|f| f.column_candidate().is_some())
    }

    /// The producible sort keys as `(column_name, direction, null_order)`, in order,
    /// or `None` if any key is not a bare column (see [`SortSpec::is_producible`]).
    pub fn producible_columns(&self) -> Option<Vec<(String, SortDirection, NullOrder)>> {
        self.fields
            .iter()
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

/// Reorder rows by a table's sort order, returning ONE sorted batch.
///
/// `batches` may carry trailing columns beyond `data_schema` (a compaction output
/// embeds the rowid, and for a partial file the per-row snapshot id); sort keys
/// resolve to `data_schema` positions — the leading columns — so those trailing
/// columns travel with their rows.
///
/// This is a **global** sort across all of `batches`, matching official DuckLake,
/// which sorts a write by planting a blocking `PhysicalOrder` above the insert plan
/// (`ducklake_insert.cpp`). That is what makes successive rolled files cover
/// contiguous, non-overlapping value ranges — the property that lets a reader skip
/// whole files. Sorting *within* a file would not achieve it: a file's min/max is
/// the min/max of its rows regardless of their order, so only its row-group bounds
/// would tighten.
///
/// Returns `batches` unchanged when there is no producible sort order, a key is not
/// in the schema, or there are no rows — sort order affects statistics locality
/// only, never correctness.
#[cfg(feature = "write")]
pub(crate) fn sort_batches_by_spec(
    batches: Vec<arrow::record_batch::RecordBatch>,
    data_schema: &arrow::datatypes::Schema,
    sort_spec: Option<&SortSpec>,
) -> crate::Result<Vec<arrow::record_batch::RecordBatch>> {
    use arrow::array::{ArrayRef, RecordBatch};
    use std::sync::Arc;

    let Some(keys) = sort_spec.and_then(|spec| spec.producible_columns()) else {
        return Ok(batches);
    };
    if keys.is_empty() || batches.iter().all(|b| b.num_rows() == 0) {
        return Ok(batches);
    }
    let mut resolved = Vec::with_capacity(keys.len());
    for (name, direction, null_order) in &keys {
        let Ok(index) = data_schema.index_of(name) else {
            return Ok(batches);
        };
        resolved.push((
            index,
            arrow::compute::SortOptions {
                descending: matches!(direction, SortDirection::Desc),
                nulls_first: null_order.nulls_first(),
            },
        ));
    }
    let full_schema = batches[0].schema();
    let combined = arrow::compute::concat_batches(&full_schema, &batches)?;
    let sort_columns: Vec<arrow::compute::SortColumn> = resolved
        .iter()
        .map(|(index, options)| arrow::compute::SortColumn {
            values: Arc::clone(combined.column(*index)),
            options: Some(*options),
        })
        .collect();
    let indices = arrow::compute::lexsort_to_indices(&sort_columns, None)?;
    let sorted_columns = combined
        .columns()
        .iter()
        .map(|c| arrow::compute::take(c, &indices, None))
        .collect::<std::result::Result<Vec<ArrayRef>, _>>()?;
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
    fn from_rows_empty_is_none() {
        assert_eq!(SortSpec::from_rows(vec![]), None);
    }
}
