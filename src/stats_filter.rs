//! Lower a pushed-down predicate to DuckLake catalog-statistics SQL.
//!
//! DuckLake records per-file `min_value` / `max_value` / `null_count` /
//! `value_count` / `contains_nan` in `ducklake_file_column_stats`. Official
//! DuckLake narrows the data-file list inside the metadata query itself: one CTE
//! per filtered column, `LEFT JOIN`ed to `ducklake_data_file` on `data_file_id`,
//! with the pushed-down filter rewritten against those stat columns
//! (`ducklake_metadata_manager.cpp`, `ConvertFilterPushdownToSQL`). Files whose
//! statistics prove they cannot contain a matching row are never listed, so the
//! cost of planning a selective query stops being proportional to the table.
//!
//! This module is the whole of that rewrite. It takes a physical predicate and
//! produces a backend-agnostic [`StatsFilter`]; [`StatsFilter::render`] turns
//! that into SQL for one dialect. Nothing here talks to a database, and no
//! backend re-implements any of the semantics below.
//!
//! # Why the input is a physical expression
//!
//! Both pruning call sites can supply one. [`crate::DuckLakeTable::files_matching`]
//! receives an `Arc<dyn PhysicalExpr>` directly, and `scan` already converts its
//! logical filters with `create_physical_expr` to build the in-memory pruning
//! predicates. Lowering from `PhysicalExpr` therefore covers both paths with one
//! implementation, and guarantees the SQL and the in-memory pruning are reading
//! the same expression.
//!
//! # Fail open, always
//!
//! Every function here returns `None` rather than an error when it cannot
//! faithfully express a predicate. `None` means "no pruning" — list every file
//! and let the in-memory [`datafusion::physical_optimizer::pruning::PruningPredicate`]
//! do what it can. A filter that prunes too little is slow; one that prunes too
//! much silently loses rows, and on the mutation path (`files_matching`) it makes
//! a keyed mutation insert a duplicate instead of superseding a row. When in
//! doubt this module does nothing.
//!
//! # Deliberate deviations from official DuckLake
//!
//! Every one of these is a divergence, and the list is meant to be exhaustive —
//! a shorter list than the code has is worse than no list. Each either follows
//! from where the SQL runs, or from how this engine orders values. None of them
//! prunes a file official would keep except where noted.
//!
//! **1. `TRY_CAST` is not portable, so each dialect supplies its own.** Official
//! always evaluates its filter SQL inside DuckDB, even against a foreign
//! catalog: its Postgres manager overrides only the CTE *body* and wraps it in
//! `postgres_query(...)`, so the join, the casts and the predicate all run in
//! DuckDB. This crate sends SQL natively to each catalog engine, where
//! `TRY_CAST` does not exist and a plain `CAST` of a malformed stat either
//! aborts the query (PostgreSQL) or silently returns zero (SQLite, MySQL) — the
//! second of which would prune a matching file. Each dialect supplies a
//! NULL-on-unparseable construct through [`StatsSqlDialect::try_cast`], and may
//! decline a type it cannot handle safely, which drops that comparison.
//!
//! **2. Raw string comparisons force a binary collation.** Official inherits
//! DuckDB's byte-wise collation. A native MySQL catalog defaults to
//! `utf8mb4_0900_ai_ci`, which is case- and accent-insensitive and would drop
//! files that match. DataFusion compares `Utf8` byte-wise, so raw comparisons go
//! through [`StatsSqlDialect::collate_binary`].
//!
//! **3. Float bounds are gated on `contains_nan`, except for equality.**
//! Catalog bounds for a float column exclude NaN, so a file whose NaN state is
//! unknown or positive can hold values outside them. DuckDB normalizes NaN —
//! `-NaN = NaN` is true and either sign sorts above every value — so there NaN
//! can only hide *above* a recorded max, and official guards only the max
//! (`GenerateConstantFilterDouble` appends `OR contains_nan` for `>`, `>=` and
//! `<>`). DataFusion compares floats with arrow's `total_cmp`, which is
//! sign-sensitive: `-NaN < -Infinity` is true here and false in DuckDB. A
//! negative NaN therefore sits *below* a recorded min, and `contains_nan` does
//! not record the sign, so neither bound can be trusted.
//! `float_bound_is_usable` in `table.rs` gates both bounds for that reason, and
//! `nan_pruning_barrier` keeps such predicates out of the parquet scan
//! for the same one. So a condition reading a bound on a float column is
//! rendered as `(<bounds not usable>) OR (<official condition>)`. That matches
//! official exactly for `>`, `>=` and `<>`, where its own `OR contains_nan`
//! already keeps every NaN-bearing file, and is strictly more conservative for
//! `<` and `<=`. Equality is left ungated, as official leaves it: no NaN
//! compares equal to a finite value under `total_cmp`, and
//! `StatsLiteral::new` refuses a NaN constant, so every row satisfying
//! `x = C` is a non-NaN row and the recorded bounds already decide it.
//!
//! **4. A stat is validated against its encoding before it is used.** Official
//! hands the stored text straight to `TRY_CAST`. That is safe for official
//! because DuckDB wrote every stat it reads; a catalog this crate opens may have
//! been written by anything, including `ducklake_add_files` over a pre-1.11
//! parquet-mr file whose float min/max hold NaN. An input function is
//! permissive by design: DuckDB reads `nan`, `epoch` and `0x10` into real
//! values, and PostgreSQL's `pg_input_is_valid` additionally accepts `today`,
//! `now`, `infinity` and `NaN`. Each casts to something that then prunes files —
//! a `nan` bound makes `x = 5.0` false for a file whose other rows match, and a
//! `today` bound would make pruning depend on the wall clock. So every dialect
//! admits only the shapes [`crate::stats_encode`] writes. This is strictly more
//! conservative than official and declines a few stats official would accept.
//!
//! **5. A condition that evaluates to SQL `NULL` keeps its file.** Official has
//! no counterpart; [`StatsSqlDialect::keep_when_unknown`] wraps every per-column
//! condition. The per-stat `IS NULL` disjuncts cannot cover this on their own,
//! because a stat that is present but malformed is not NULL while the cast of it
//! is, and the condition sits under `WHERE ... AND`, where `NULL` excludes the
//! row. Pruning happens only on a definite `false`.
//!
//! **6. Temporal comparisons happen in the text domain on some dialects.**
//! Official casts them. PostgreSQL and MySQL store microseconds and *round* a
//! longer fraction, which is monotonic but not injective — two distinct
//! nanosecond instants can land on one microsecond, so a strict comparison that
//! holds of the stored values comes back false. SQLite has no temporal type at
//! all. The canonical encoding is chronologically ordered byte-wise, so
//! comparing the two strings answers the same question exactly, at full
//! precision, and parses nothing — an impossible date cannot raise. It is sound
//! only for that one encoding: `chrono` renders a year past 9999 as `+12345` and
//! one before the common era as `-0044`, both of which sort below every digit,
//! and `.50` sorts above `.5` while naming the same instant. Anything else is
//! declined.
//!
//! **7. A NaN or non-finite constant pushes down nothing.** Official prunes on
//! `contains_nan` for `x = NaN`, and emits a real comparison for `x < inf`.
//! `StatsLiteral::new` refuses both: a non-finite constant renders as a quoted
//! literal, which a dialect that coerces text to numbers reads as zero, and
//! `x < inf` would then prune every file with a non-negative minimum. This is
//! less pruning than official, never more.
//!
//! **8. The float gate keys off the column's type, official off the
//! constant's.** Official's NaN handling is selected by the constant
//! (`type ? *type : constant_expr->GetValue().type()`, with `type` always null
//! from its entry point). It is the *column's* stored bounds that exclude NaN,
//! and a `PhysicalExpr` handed to [`crate::DuckLakeTable::files_matching`]
//! carries no guarantee the constant was coerced to the column's type, so the
//! gate follows the column while the cast still follows the constant. The
//! difference shows only for a non-float column compared against a float
//! constant, where this prunes and official cannot; a non-float column holds no
//! NaN, so its bounds are sound.

use std::collections::BTreeSet;
use std::sync::Arc;

use arrow::datatypes::{DataType, Schema};
use datafusion::common::ScalarValue;
use datafusion::logical_expr::Operator;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_expr::expressions::{
    BinaryExpr, Column, InListExpr, IsNotNullExpr, IsNullExpr, Literal,
};

use crate::metadata_provider::DuckLakeTableColumn;
use crate::stats_encode;

/// One statistics column in `ducklake_file_column_stats`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum StatKind {
    MinValue,
    MaxValue,
    NullCount,
    ValueCount,
    ContainsNan,
}

impl StatKind {
    /// Column name as spelled in `ducklake_file_column_stats`.
    pub fn column_name(self) -> &'static str {
        match self {
            Self::MinValue => "min_value",
            Self::MaxValue => "max_value",
            Self::NullCount => "null_count",
            Self::ValueCount => "value_count",
            Self::ContainsNan => "contains_nan",
        }
    }

    /// Whether this stat is one of the two order-bearing bounds. Only these need
    /// a cast, a collation, or the float gate.
    fn is_bound(self) -> bool {
        matches!(self, Self::MinValue | Self::MaxValue)
    }
}

/// A constant rendered into statistics SQL.
///
/// `text` is produced by [`stats_encode::encode_scalar`], so it is byte-identical
/// to what the write path stored in `min_value` / `max_value`. That shared
/// encoding is what makes comparing them meaningful at all.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StatsLiteral {
    text: String,
    /// Cast target for the *stat* column, or `None` to compare raw text.
    ///
    /// Official takes this from the constant's type and never the column's
    /// (`const auto &target_type = type ? *type : constant_expr->GetValue().type()`),
    /// so a mixed-type comparison casts the way the constant asks.
    cast: Option<DataType>,
    /// Whether the literal renders unquoted. Official emits a bare number only
    /// for finite numerics and a quoted literal for everything else, including
    /// non-finite floats.
    unquoted: bool,
}

/// Comparison against one order-bearing bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundOp {
    Lt,
    LtEq,
    Gt,
    GtEq,
}

impl BoundOp {
    fn sql(self) -> &'static str {
        match self {
            Self::Lt => "<",
            Self::LtEq => "<=",
            Self::Gt => ">",
            Self::GtEq => ">=",
        }
    }
}

/// A lowered condition over one column's statistics.
#[derive(Debug, Clone, PartialEq)]
pub enum StatsExpr {
    And(Vec<StatsExpr>),
    Or(Vec<StatsExpr>),
    /// `<literal> BETWEEN min_value AND max_value` — equality.
    LiteralWithinBounds(StatsLiteral),
    /// `NOT (min_value = <literal> AND max_value = <literal>)` — inequality.
    /// Prunes only a file proven to hold that one value in every row.
    NotEveryRowEqual(StatsLiteral),
    /// `<bound> <op> <literal>` — a range comparison.
    BoundCompare {
        stat: StatKind,
        op: BoundOp,
        literal: StatsLiteral,
    },
    /// `<count> > 0`, for `null_count` (`IS NULL`) or `value_count`
    /// (`IS NOT NULL`).
    CountPositive(StatKind),
    /// `contains_nan IS NULL OR contains_nan <> false`. True whenever a float
    /// column's recorded bounds cannot be trusted; see the module docs.
    FloatBoundsUnusable,
}

/// Everything one column contributes to the listing query.
#[derive(Debug, Clone, PartialEq)]
pub struct StatsColumnFilter {
    /// `ducklake_column.column_id`, which is what `ducklake_file_column_stats`
    /// keys on. Not the Arrow field index.
    pub column_id: i64,
    /// Stats the condition itself reads. Each one gets an `IS NULL OR` disjunct
    /// so a present-but-incomplete stats row fails open.
    pub referenced_stats: BTreeSet<StatKind>,
    /// Whether `(value_count IS NULL OR value_count > 0)` guards the condition.
    ///
    /// Official adds it when the filter reads a bound but does not read
    /// `null_count`. A file of nothing but NULLs has no min/max, and a filter
    /// that cannot be satisfied by a NULL row must not prune it on absent
    /// bounds. When the filter *does* mention `null_count` the guard is dropped,
    /// because such a filter can be satisfied by a NULL row.
    pub needs_value_count_guard: bool,
    pub condition: StatsExpr,
}

impl StatsColumnFilter {
    /// Stats the CTE must select: everything the condition reads, plus
    /// `value_count` when it is only the guard.
    ///
    /// This is deliberately *not* the same set as [`Self::referenced_stats`].
    /// Official builds its `IS NULL` disjuncts before inserting the guard stat,
    /// so a guard-only `value_count` appears in the select list and in the guard
    /// but never as an `IS NULL` disjunct. Keeping the two sets distinct here is
    /// what stops five backends from each re-deriving that subtlety.
    pub fn cte_stats(&self) -> BTreeSet<StatKind> {
        let mut stats = self.referenced_stats.clone();
        if self.needs_value_count_guard {
            stats.insert(StatKind::ValueCount);
        }
        stats
    }
}

/// A whole predicate lowered to per-column statistics conditions.
#[derive(Debug, Clone, PartialEq)]
pub struct StatsFilter {
    /// One entry per filtered column, ordered by `column_id`.
    ///
    /// Official keys these off an `unordered_map`, so its condition order is
    /// hash-dependent and explicitly not normative. Sorting makes the generated
    /// SQL deterministic and diffable, which the tests rely on.
    pub columns: Vec<StatsColumnFilter>,
}

/// SQL a particular catalog engine needs spelled its own way.
///
/// Only these few atoms differ between backends; [`StatsFilter::render`] owns
/// the structure, the guards and the ordering.
pub trait StatsSqlDialect {
    /// `TRY_CAST(<expr> AS <data_type>)`: the value cast to `data_type`, or
    /// SQL `NULL` when the text will not parse. Returning `None` declines the
    /// type, which drops the comparison and prunes nothing.
    ///
    /// Returning something that *errors* on malformed input, or that coerces it
    /// to a number, is a correctness bug — see the module docs.
    ///
    /// `literal` is the constant this stat will be compared against, whose
    /// encoded text is [`StatsLiteral::text`]. A dialect that reproduces a
    /// comparison in the *text* domain — SQLite has no temporal type, so its
    /// only option for a date is comparing the encoded strings — needs it: text
    /// order equals value order only while both sides are canonically encoded,
    /// and the dialect can test the stat for that in SQL but the constant only
    /// here. A dialect whose engine converts the constant can need it too, to
    /// decline one it would refuse to convert.
    fn try_cast(&self, expr: &str, literal: &StatsLiteral, data_type: &DataType) -> Option<String>;

    /// Force byte-wise comparison of a raw (uncast) string stat.
    fn collate_binary(&self, expr: &str) -> String;

    /// `<expr> IS NULL OR <expr> <> false` for a possibly-NULL boolean stat.
    fn boolean_is_not_false(&self, expr: &str) -> String;

    /// Render a canonical stats encoding as a SQL string literal.
    ///
    /// The default doubles embedded single quotes, which is standard SQL and
    /// what official emits (`DuckLakeUtil::SQLLiteralToString`). A dialect that
    /// gives another character meaning inside a quoted string must override
    /// this — MySQL treats backslash as an escape unless `NO_BACKSLASH_ESCAPES`
    /// is set, and `stats_encode` passes `Utf8` through verbatim, so a value
    /// containing one reaches the SQL text.
    fn quote_literal(&self, text: &str) -> String {
        format!("'{}'", text.replace('\'', "''"))
    }

    /// Keep a file whose condition evaluates to SQL `NULL`.
    ///
    /// The condition sits under `WHERE ... AND`, where `NULL` excludes the row —
    /// so "unknown" would prune. The per-stat `IS NULL` disjuncts cannot cover
    /// this on their own: they test the stored column, but a stat that is
    /// *present and malformed* is not NULL, while the cast of it is. Official
    /// carries the same shape and is safe only because DuckDB wrote every stat
    /// it reads; a catalog this crate opens may have been written by anything.
    ///
    /// Overriding this is for spelling, not policy. Pruning must happen only on
    /// a definite `false`.
    fn keep_when_unknown(&self, condition: &str) -> String {
        format!("({condition}) IS NOT FALSE")
    }
}

/// Whether official casts the stat for comparison against this type.
///
/// Mirrors `RequiresValueComparison` in `ducklake_stats.hpp`: numeric, temporal
/// or boolean. Everything else — `VARCHAR`, `ENUM`, `UUID` — is compared as
/// text, matching how the write path merged those bounds in the first place.
fn requires_value_comparison(data_type: &DataType) -> bool {
    data_type.is_numeric() || data_type.is_temporal() || matches!(data_type, DataType::Boolean)
}

/// Whether a value renders as a bare number. Non-finite floats do not: official
/// routes them through the quoted-literal path.
fn renders_unquoted(value: &ScalarValue, data_type: &DataType) -> bool {
    if !data_type.is_numeric() {
        return false;
    }
    match value {
        ScalarValue::Float16(Some(v)) => v.is_finite(),
        ScalarValue::Float32(Some(v)) => v.is_finite(),
        ScalarValue::Float64(Some(v)) => v.is_finite(),
        _ => true,
    }
}

impl StatsLiteral {
    /// Build a literal, or `None` when this constant cannot be compared against
    /// stored statistics.
    ///
    /// Refuses, matching official and this crate's own encoder: NULL constants;
    /// binary types, which official never pushes down (`LogicalTypeId::BLOB`);
    /// and anything [`stats_encode::encode_scalar`] has no canonical text for —
    /// `TIME`, `UUID`, `INTERVAL`, `Decimal256`, nested types, NaN, strings over
    /// [`stats_encode::MAX_STRING_STAT_BYTES`], and strings containing a NUL
    /// byte. A NUL is what official's own `'\0'` check rejects, and what the
    /// write path stores as NULL, so there is nothing to compare against.
    fn new(value: &ScalarValue) -> Option<Self> {
        if value.is_null() {
            return None;
        }
        let data_type = value.data_type();
        if matches!(
            data_type,
            DataType::Binary
                | DataType::LargeBinary
                | DataType::BinaryView
                | DataType::FixedSizeBinary(_)
        ) {
            return None;
        }
        // A non-finite float has a canonical encoding (`inf`, `-inf`) but no
        // usable comparison. It renders as a quoted literal, so a dialect that
        // coerces text to a number reads `'inf'` as 0 and `x < inf` then prunes
        // every file whose minimum is non-negative. The read path already
        // refuses these bounds (`parse_statistic_scalar` in `table.rs` has no
        // Arrow representation for them), so there is nothing on the other side
        // of the comparison either.
        if matches!(
            value,
            ScalarValue::Float16(Some(_))
                | ScalarValue::Float32(Some(_))
                | ScalarValue::Float64(Some(_))
        ) && !renders_unquoted(value, &data_type)
        {
            return None;
        }
        let text = stats_encode::encode_scalar(value)?;
        // `encode_scalar` already drops NUL-bearing strings; assert it rather
        // than trusting it, because a NUL reaching the SQL text is a hazard in
        // every dialect.
        if text.contains('\0') {
            return None;
        }
        let cast = requires_value_comparison(&data_type).then_some(data_type.clone());
        Some(Self {
            text,
            cast: cast.clone(),
            unquoted: renders_unquoted(value, &data_type),
        })
    }

    /// The constant's encoded text, exactly as [`stats_encode::encode_scalar`]
    /// produced it and byte-identical to what the write path stored in
    /// `min_value` / `max_value`. Unquoted and unescaped: this is for a dialect
    /// to *inspect* in [`StatsSqlDialect::try_cast`], never to splice into SQL.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// The literal as SQL.
    fn render(&self, ctx: &RenderContext<'_>) -> String {
        if self.unquoted {
            self.text.clone()
        } else {
            ctx.dialect.quote_literal(&self.text)
        }
    }

    /// A stat column cast for comparison against this literal, or `None` if the
    /// dialect declined the type.
    fn render_stat(&self, stat: StatKind, ctx: &RenderContext<'_>) -> Option<String> {
        let raw = ctx.qualify(stat);
        match &self.cast {
            Some(data_type) => ctx.dialect.try_cast(&raw, self, data_type),
            None => Some(ctx.dialect.collate_binary(&raw)),
        }
    }
}

/// Everything rendering needs for one column's CTE.
struct RenderContext<'a> {
    dialect: &'a dyn StatsSqlDialect,
    alias: String,
}

impl RenderContext<'_> {
    /// `<alias>.<stat>`
    fn qualify(&self, stat: StatKind) -> String {
        format!("{}.{}", self.alias, stat.column_name())
    }
}

/// SQL for one column: its CTE alias, the stats it selects, and its condition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderedColumnFilter {
    /// CTE alias, `col_<column_id>_stats`.
    pub alias: String,
    /// `ducklake_file_column_stats.column_id` the CTE restricts to.
    pub column_id: i64,
    /// Stat columns the CTE must select, in a stable order. `data_file_id` is
    /// always required in addition to these and is not listed.
    pub stats: Vec<&'static str>,
    /// The `WHERE` condition, already wrapped in its no-stats and NULL guards.
    pub condition: String,
}

impl StatsFilter {
    /// Render for one dialect, or `None` when nothing survives.
    ///
    /// A column whose condition cannot be rendered — because the dialect
    /// declined a cast, say — contributes nothing at all: no condition, no CTE,
    /// no join. Official does the same (`if (filter_condition.empty()) continue`),
    /// and emitting an unused CTE would be a pointless join.
    pub fn render(&self, dialect: &dyn StatsSqlDialect) -> Option<Vec<RenderedColumnFilter>> {
        let rendered: Vec<_> = self
            .columns
            .iter()
            .filter_map(|column| Self::render_column(column, dialect))
            .collect();
        (!rendered.is_empty()).then_some(rendered)
    }

    fn render_column(
        column: &StatsColumnFilter,
        dialect: &dyn StatsSqlDialect,
    ) -> Option<RenderedColumnFilter> {
        let context = RenderContext {
            dialect,
            alias: format!("col_{}_stats", column.column_id),
        };
        let condition = render_expr(&column.condition, &context)?;

        // One `IS NULL OR` disjunct per stat the condition reads. A stats row can
        // exist with individual stats NULL — the write path stores NULL when the
        // parquet footer carried no bound, no NaN signal, or an inconsistent
        // null/value count — and the whole condition sits under `WHERE ... AND`,
        // so a NULL condition would prune the file. These disjuncts are what make
        // an incomplete row fail open, and dropping one to "simplify" the SQL
        // loses rows.
        let null_checks: String = column
            .referenced_stats
            .iter()
            .map(|stat| format!("{} IS NULL OR ", context.qualify(*stat)))
            .collect();

        let body = if column.needs_value_count_guard {
            let value_count = context.qualify(StatKind::ValueCount);
            format!("({value_count} IS NULL OR {value_count} > 0) AND ({null_checks}{condition})")
        } else {
            format!("{null_checks}{condition}")
        };

        // A file written before this column existed has no stats row at all and
        // LEFT JOINs to all-NULL. It must always be kept.
        let data_file_id = format!("{}.data_file_id", context.alias);
        let condition = format!("({data_file_id} IS NULL OR ({body}))");
        Some(RenderedColumnFilter {
            alias: context.alias.clone(),
            column_id: column.column_id,
            stats: column
                .cte_stats()
                .iter()
                .map(|stat| stat.column_name())
                .collect(),
            condition: dialect.keep_when_unknown(&condition),
        })
    }
}

/// Render one condition, or `None` if it cannot be expressed.
///
/// The `And` / `Or` asymmetry is load-bearing and matches official. Dropping a
/// conjunct from an `AND` only weakens pruning, so an unrenderable child is
/// skipped. Dropping a branch from an `OR` would prune files that branch admits,
/// so a single unrenderable child abandons the whole disjunction. Reversing
/// these silently loses rows.
fn render_expr(expr: &StatsExpr, ctx: &RenderContext<'_>) -> Option<String> {
    match expr {
        StatsExpr::And(children) => {
            let parts: Vec<_> = children
                .iter()
                .filter_map(|child| render_expr(child, ctx))
                .map(|part| format!("({part})"))
                .collect();
            (!parts.is_empty()).then(|| parts.join(" AND "))
        },
        StatsExpr::Or(children) => {
            let mut parts = Vec::with_capacity(children.len());
            for child in children {
                parts.push(format!("({})", render_expr(child, ctx)?));
            }
            (!parts.is_empty()).then(|| parts.join(" OR "))
        },
        StatsExpr::LiteralWithinBounds(literal) => {
            let min = literal.render_stat(StatKind::MinValue, ctx)?;
            let max = literal.render_stat(StatKind::MaxValue, ctx)?;
            Some(format!("{} BETWEEN {min} AND {max}", literal.render(ctx)))
        },
        StatsExpr::NotEveryRowEqual(literal) => {
            let min = literal.render_stat(StatKind::MinValue, ctx)?;
            let max = literal.render_stat(StatKind::MaxValue, ctx)?;
            let value = literal.render(ctx);
            Some(format!("NOT ({min} = {value} AND {max} = {value})"))
        },
        StatsExpr::BoundCompare {
            stat,
            op,
            literal,
        } => {
            let bound = literal.render_stat(*stat, ctx)?;
            Some(format!("{bound} {} {}", op.sql(), literal.render(ctx)))
        },
        StatsExpr::CountPositive(stat) => Some(format!("{} > 0", ctx.qualify(*stat))),
        StatsExpr::FloatBoundsUnusable => Some(
            ctx.dialect
                .boolean_is_not_false(&ctx.qualify(StatKind::ContainsNan)),
        ),
    }
}

/// One column's contribution while lowering is still in progress.
struct Lowered {
    condition: StatsExpr,
    referenced_stats: BTreeSet<StatKind>,
}

/// Lower a physical predicate to per-column statistics conditions.
///
/// `schema` is the table's physical schema and `columns` its DuckLake columns in
/// the same order, which is how an Arrow field index becomes the `column_id`
/// that `ducklake_file_column_stats` keys on.
///
/// Returns `None` when nothing at all could be lowered.
pub fn lower_predicate(
    predicate: &Arc<dyn PhysicalExpr>,
    schema: &Schema,
    columns: &[DuckLakeTableColumn],
) -> Option<StatsFilter> {
    let mut by_column: Vec<(i64, Lowered)> = Vec::new();

    // Top-level conjuncts are independent: each may prune on its own, and one
    // that cannot be lowered simply contributes nothing.
    for conjunct in datafusion::physical_expr::split_conjunction(predicate) {
        let Some((column_id, lowered)) = lower_conjunct(conjunct, schema, columns) else {
            continue;
        };
        match by_column.iter_mut().find(|(id, _)| *id == column_id) {
            // Two conjuncts on one column intersect, exactly as official sees
            // them when its optimizer hands over a single `CONJUNCTION_AND`.
            Some((_, existing)) => {
                let previous =
                    std::mem::replace(&mut existing.condition, StatsExpr::And(Vec::new()));
                existing.condition = match previous {
                    StatsExpr::And(mut parts) => {
                        parts.push(lowered.condition);
                        StatsExpr::And(parts)
                    },
                    other => StatsExpr::And(vec![other, lowered.condition]),
                };
                existing.referenced_stats.extend(lowered.referenced_stats);
            },
            None => by_column.push((column_id, lowered)),
        }
    }

    if by_column.is_empty() {
        return None;
    }
    by_column.sort_by_key(|(column_id, _)| *column_id);

    let columns = by_column
        .into_iter()
        .map(|(column_id, lowered)| {
            // Official: guard when the filter reads a bound but not
            // `null_count`. A filter mentioning `null_count` can be satisfied by
            // a NULL row, and an all-NULL file has no bounds, so guarding it
            // would prune a file that matches.
            let reads_null_count = lowered.referenced_stats.contains(&StatKind::NullCount);
            let reads_bound = lowered.referenced_stats.iter().any(|stat| stat.is_bound());
            StatsColumnFilter {
                column_id,
                needs_value_count_guard: !reads_null_count && reads_bound,
                referenced_stats: lowered.referenced_stats,
                condition: lowered.condition,
            }
        })
        .collect();

    Some(StatsFilter {
        columns,
    })
}

/// Lower one conjunct, which must concern exactly one column.
///
/// A conjunct spanning several columns — `a > 5 OR b < 3` — cannot become a
/// per-column zone-map condition and is dropped. Official never encounters one:
/// DuckDB's optimizer hands it a map already keyed by column.
fn lower_conjunct(
    expr: &Arc<dyn PhysicalExpr>,
    schema: &Schema,
    columns: &[DuckLakeTableColumn],
) -> Option<(i64, Lowered)> {
    let index = sole_column_index(expr)?;
    let field = schema.fields().get(index)?;
    let column = columns.get(index)?;

    let mut referenced_stats = BTreeSet::new();
    let mut condition = lower_node(expr, index, &mut referenced_stats)?;

    // Float gate (see module docs). Applied once around the whole condition
    // rather than around each comparison: `gate OR (a AND b)` and
    // `(gate OR a) AND (gate OR b)` are equivalent, and the former is the SQL a
    // reader can actually follow. Only conditions that read a bound need it —
    // `IS NULL` on a float column prunes on `null_count` alone and is unaffected.
    if needs_float_gate(&condition) && is_float(field.data_type()) {
        referenced_stats.insert(StatKind::ContainsNan);
        condition = StatsExpr::Or(vec![StatsExpr::FloatBoundsUnusable, condition]);
    }

    Some((
        column.column_id,
        Lowered {
            condition,
            referenced_stats,
        },
    ))
}

/// Whether a condition can be satisfied by a NaN row whose value falls outside
/// the recorded bounds, and so needs the float gate.
///
/// Equality cannot. Arrow orders floats with `total_cmp`, under which a NaN of
/// either sign compares equal to no finite value, and [`StatsLiteral::new`]
/// refuses a NaN constant — so every row satisfying `x = C` is a non-NaN row,
/// and non-NaN rows *are* bounded by the recorded min/max. Official reaches the
/// same conclusion for its own engine and leaves `COMPARE_EQUAL` ungated, so
/// skipping the gate here is convergence as well as recovered pruning: gating it
/// would keep every file whose `contains_nan` is unknown — every
/// register-by-reference load — against the most selective predicate there is.
///
/// Every other comparison can. `>` and `>=` admit `+NaN`, `<` and `<=` admit
/// `-NaN` under `total_cmp`, and `<>` admits both; the recorded bounds exclude
/// NaN in all three cases, so a file holding one must be kept.
fn needs_float_gate(expr: &StatsExpr) -> bool {
    match expr {
        StatsExpr::And(children) | StatsExpr::Or(children) => children.iter().any(needs_float_gate),
        StatsExpr::BoundCompare {
            ..
        }
        | StatsExpr::NotEveryRowEqual(_) => true,
        // `LiteralWithinBounds` is the equality form; the counts and the gate
        // itself read no bound.
        StatsExpr::LiteralWithinBounds(_)
        | StatsExpr::CountPositive(_)
        | StatsExpr::FloatBoundsUnusable => false,
    }
}

/// Whether catalog bounds for this type exclude NaN and so need the float gate.
fn is_float(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Float16 | DataType::Float32 | DataType::Float64
    )
}

/// The single column index an expression references, or `None` if it references
/// none or more than one.
fn sole_column_index(expr: &Arc<dyn PhysicalExpr>) -> Option<usize> {
    use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};

    let mut found: Option<usize> = None;
    let mut conflicting = false;
    expr.apply(|node| {
        if let Some(column) = node.downcast_ref::<Column>() {
            match found {
                Some(index) if index != column.index() => {
                    conflicting = true;
                    return Ok(TreeNodeRecursion::Stop);
                },
                Some(_) => {},
                None => found = Some(column.index()),
            }
        }
        Ok(TreeNodeRecursion::Continue)
    })
    .ok()?;

    (!conflicting).then_some(found).flatten()
}

/// Lower one node of a single-column expression.
///
/// The float gate is not decided here — [`lower_conjunct`] applies it once around
/// the finished condition — so this only ever needs the constant's type, which it
/// reads from each literal.
///
/// Children are lowered into their own stat sets and merged only on success, so
/// a branch that fails to lower never leaves a stat behind. A stray stat would
/// only add an `IS NULL OR` disjunct and weaken pruning, but the SQL should say
/// exactly what it reads.
fn lower_node(
    expr: &Arc<dyn PhysicalExpr>,
    column_index: usize,
    referenced_stats: &mut BTreeSet<StatKind>,
) -> Option<StatsExpr> {
    if let Some(binary) = expr.downcast_ref::<BinaryExpr>() {
        return match binary.op() {
            Operator::And => {
                // An unlowerable conjunct only costs pruning, so skip it.
                let mut parts = Vec::new();
                let mut stats = BTreeSet::new();
                for child in [binary.left(), binary.right()] {
                    let mut child_stats = BTreeSet::new();
                    if let Some(part) = lower_node(child, column_index, &mut child_stats) {
                        parts.push(part);
                        stats.extend(child_stats);
                    }
                }
                if parts.is_empty() {
                    return None;
                }
                referenced_stats.extend(stats);
                Some(if parts.len() == 1 {
                    parts.remove(0)
                } else {
                    StatsExpr::And(parts)
                })
            },
            Operator::Or => {
                // A missing branch would prune files that branch admits, so the
                // whole disjunction is abandoned.
                let mut parts = Vec::new();
                let mut stats = BTreeSet::new();
                for child in [binary.left(), binary.right()] {
                    let mut child_stats = BTreeSet::new();
                    parts.push(lower_node(child, column_index, &mut child_stats)?);
                    stats.extend(child_stats);
                }
                referenced_stats.extend(stats);
                Some(StatsExpr::Or(parts))
            },
            operator => lower_comparison(binary, *operator, column_index, referenced_stats),
        };
    }

    if let Some(is_null) = expr.downcast_ref::<IsNullExpr>() {
        subject_column(is_null.arg(), column_index)?;
        referenced_stats.insert(StatKind::NullCount);
        return Some(StatsExpr::CountPositive(StatKind::NullCount));
    }

    if let Some(is_not_null) = expr.downcast_ref::<IsNotNullExpr>() {
        subject_column(is_not_null.arg(), column_index)?;
        referenced_stats.insert(StatKind::ValueCount);
        return Some(StatsExpr::CountPositive(StatKind::ValueCount));
    }

    if let Some(in_list) = expr.downcast_ref::<InListExpr>() {
        // `NOT IN` proves nothing about a range: official pushes down only the
        // positive form.
        if in_list.negated() {
            return None;
        }
        subject_column(in_list.expr(), column_index)?;
        if in_list.list().is_empty() {
            return None;
        }
        let mut parts = Vec::with_capacity(in_list.list().len());
        for element in in_list.list() {
            // A single unusable element — a non-constant, a NULL, or a value with
            // no canonical encoding — abandons the whole list, for the same
            // reason as `OR`.
            let literal = element.downcast_ref::<Literal>()?;
            parts.push(StatsExpr::LiteralWithinBounds(StatsLiteral::new(
                literal.value(),
            )?));
        }
        referenced_stats.insert(StatKind::MinValue);
        referenced_stats.insert(StatKind::MaxValue);
        return Some(if parts.len() == 1 {
            parts.remove(0)
        } else {
            StatsExpr::Or(parts)
        });
    }

    None
}

/// Lower a comparison of a bare column against a constant.
fn lower_comparison(
    binary: &BinaryExpr,
    operator: Operator,
    column_index: usize,
    referenced_stats: &mut BTreeSet<StatKind>,
) -> Option<StatsExpr> {
    // The subject must be the bare column. Official accepts only a column
    // reference (`IsSimpleFilterSubject`), so `CAST(a AS BIGINT) > 5` and
    // `a > b` push down nothing.
    let (literal, operator) = if subject_column(binary.left(), column_index).is_some() {
        (binary.right().downcast_ref::<Literal>()?, operator)
    } else if subject_column(binary.right(), column_index).is_some() {
        // Constant on the left: `5 < a` means `a > 5`.
        (
            binary.left().downcast_ref::<Literal>()?,
            flip_operator(operator)?,
        )
    } else {
        return None;
    };

    let literal = StatsLiteral::new(literal.value())?;

    let (stats, condition): (&[StatKind], StatsExpr) = match operator {
        // A file can hold the value only if it lies within the recorded range.
        Operator::Eq => (
            &[StatKind::MinValue, StatKind::MaxValue],
            StatsExpr::LiteralWithinBounds(literal),
        ),
        // Prunes only a file proven to hold that single value throughout.
        Operator::NotEq => (
            &[StatKind::MinValue, StatKind::MaxValue],
            StatsExpr::NotEveryRowEqual(literal),
        ),
        Operator::Lt => (
            &[StatKind::MinValue],
            StatsExpr::BoundCompare {
                stat: StatKind::MinValue,
                op: BoundOp::Lt,
                literal,
            },
        ),
        Operator::LtEq => (
            &[StatKind::MinValue],
            StatsExpr::BoundCompare {
                stat: StatKind::MinValue,
                op: BoundOp::LtEq,
                literal,
            },
        ),
        Operator::Gt => (
            &[StatKind::MaxValue],
            StatsExpr::BoundCompare {
                stat: StatKind::MaxValue,
                op: BoundOp::Gt,
                literal,
            },
        ),
        Operator::GtEq => (
            &[StatKind::MaxValue],
            StatsExpr::BoundCompare {
                stat: StatKind::MaxValue,
                op: BoundOp::GtEq,
                literal,
            },
        ),
        // `IS DISTINCT FROM` and friends pass DuckDB's comparison check but hit
        // official's `default` case and push down nothing.
        _ => return None,
    };

    referenced_stats.extend(stats.iter().copied());
    Some(condition)
}

/// The column an expression is, if it is a bare reference to `column_index`.
fn subject_column(expr: &Arc<dyn PhysicalExpr>, column_index: usize) -> Option<()> {
    let column = expr.downcast_ref::<Column>()?;
    (column.index() == column_index).then_some(())
}

/// Mirror a comparison so the column sits on the left.
fn flip_operator(operator: Operator) -> Option<Operator> {
    Some(match operator {
        Operator::Eq => Operator::Eq,
        Operator::NotEq => Operator::NotEq,
        Operator::Lt => Operator::Gt,
        Operator::LtEq => Operator::GtEq,
        Operator::Gt => Operator::Lt,
        Operator::GtEq => Operator::LtEq,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::{ArrowPrimitiveType, Field, TimeUnit};
    use datafusion::physical_expr::expressions::{CastExpr, NotExpr, in_list, lit};

    // ---------------------------------------------------------------------
    // Dialects
    // ---------------------------------------------------------------------

    /// DuckDB type name for a cast target, mirroring `LogicalType::ToString()`.
    ///
    /// This is the dialect official DuckLake actually renders into, so the SQL
    /// these tests assert can be diffed line-for-line against
    /// `ConvertFilterPushdownToSQL`'s output.
    fn duckdb_type_name(data_type: &DataType) -> Option<String> {
        Some(match data_type {
            DataType::Boolean => "BOOLEAN".to_string(),
            DataType::Int8 => "TINYINT".to_string(),
            DataType::Int16 => "SMALLINT".to_string(),
            DataType::Int32 => "INTEGER".to_string(),
            DataType::Int64 => "BIGINT".to_string(),
            DataType::UInt8 => "UTINYINT".to_string(),
            DataType::UInt16 => "USMALLINT".to_string(),
            DataType::UInt32 => "UINTEGER".to_string(),
            DataType::UInt64 => "UBIGINT".to_string(),
            DataType::Float32 => "FLOAT".to_string(),
            DataType::Float64 => "DOUBLE".to_string(),
            DataType::Date32 => "DATE".to_string(),
            DataType::Decimal128(precision, scale) => {
                format!("DECIMAL({precision},{scale})")
            },
            DataType::Timestamp(TimeUnit::Microsecond, None) => "TIMESTAMP".to_string(),
            _ => return None,
        })
    }

    /// Reference dialect: real `TRY_CAST`, byte-wise collation for free, real
    /// booleans. What official DuckLake emits.
    struct Duck;

    impl StatsSqlDialect for Duck {
        fn try_cast(
            &self,
            expr: &str,
            _literal: &StatsLiteral,
            data_type: &DataType,
        ) -> Option<String> {
            Some(format!(
                "TRY_CAST({expr} AS {})",
                duckdb_type_name(data_type)?
            ))
        }

        fn collate_binary(&self, expr: &str) -> String {
            expr.to_string()
        }

        fn boolean_is_not_false(&self, expr: &str) -> String {
            format!("{expr} IS NULL OR {expr} <> false")
        }
    }

    /// A dialect that cannot safely reproduce `TRY_CAST` for `BIGINT`, so it
    /// declines the type. Exercises the seam that lets a backend drop one
    /// comparison instead of risking a wrong answer.
    struct NoBigint;

    impl StatsSqlDialect for NoBigint {
        fn try_cast(
            &self,
            expr: &str,
            literal: &StatsLiteral,
            data_type: &DataType,
        ) -> Option<String> {
            if matches!(data_type, DataType::Int64) {
                return None;
            }
            Duck.try_cast(expr, literal, data_type)
        }

        fn collate_binary(&self, expr: &str) -> String {
            Duck.collate_binary(expr)
        }

        fn boolean_is_not_false(&self, expr: &str) -> String {
            Duck.boolean_is_not_false(expr)
        }
    }

    // ---------------------------------------------------------------------
    // Fixture
    // ---------------------------------------------------------------------

    /// A table: an Arrow schema plus the DuckLake columns in the same order,
    /// which is how a field index becomes a `column_id`.
    struct Fixture {
        schema: Schema,
        columns: Vec<DuckLakeTableColumn>,
    }

    impl Fixture {
        /// `(column name, Arrow type, column_id)` per column, in field order.
        fn new(spec: &[(&str, DataType, i64)]) -> Self {
            let schema = Schema::new(
                spec.iter()
                    .map(|(name, data_type, _)| Field::new(*name, data_type.clone(), true))
                    .collect::<Vec<_>>(),
            );
            let columns = spec
                .iter()
                .map(|(name, data_type, column_id)| {
                    DuckLakeTableColumn::new(
                        *column_id,
                        (*name).to_string(),
                        duckdb_type_name(data_type).unwrap_or_else(|| "VARCHAR".to_string()),
                        true,
                    )
                })
                .collect();
            Self {
                schema,
                columns,
            }
        }

        /// A bare reference to a column, by name.
        fn column(&self, name: &str) -> Arc<dyn PhysicalExpr> {
            let index = self
                .schema
                .fields()
                .iter()
                .position(|field| field.name() == name)
                .expect("column in fixture");
            Arc::new(Column::new(name, index))
        }

        fn lower(&self, predicate: &Arc<dyn PhysicalExpr>) -> Option<StatsFilter> {
            lower_predicate(predicate, &self.schema, &self.columns)
        }

        fn render_with(
            &self,
            predicate: &Arc<dyn PhysicalExpr>,
            dialect: &dyn StatsSqlDialect,
        ) -> Option<Vec<RenderedColumnFilter>> {
            self.lower(predicate)?.render(dialect)
        }

        fn render(&self, predicate: &Arc<dyn PhysicalExpr>) -> Vec<RenderedColumnFilter> {
            self.render_with(predicate, &Duck).expect("rendered")
        }

        /// The single rendered column filter, when there should be exactly one.
        fn only(&self, predicate: &Arc<dyn PhysicalExpr>) -> RenderedColumnFilter {
            let mut rendered = self.render(predicate);
            assert_eq!(rendered.len(), 1, "expected exactly one column filter");
            rendered.remove(0)
        }

        /// The rendered `WHERE` condition for the single filtered column.
        fn sql(&self, predicate: &Arc<dyn PhysicalExpr>) -> String {
            self.only(predicate).condition
        }

        /// The stat columns the single filtered column's CTE must select.
        fn cte_stats(&self, predicate: &Arc<dyn PhysicalExpr>) -> Vec<&'static str> {
            self.only(predicate).stats
        }

        /// `true` when nothing at all pushes down.
        fn lowers_to_nothing(&self, predicate: &Arc<dyn PhysicalExpr>) -> bool {
            self.lower(predicate).is_none()
        }
    }

    /// An `Int32` column `a` with `column_id` 1, plus an `Int32` column `b` with
    /// `column_id` 2. `column_id` deliberately differs from the field index.
    fn ints() -> Fixture {
        Fixture::new(&[("a", DataType::Int32, 1), ("b", DataType::Int32, 2)])
    }

    fn bin(
        left: Arc<dyn PhysicalExpr>,
        op: Operator,
        right: Arc<dyn PhysicalExpr>,
    ) -> Arc<dyn PhysicalExpr> {
        Arc::new(BinaryExpr::new(left, op, right))
    }

    // ---------------------------------------------------------------------
    // Operator map (official `GenerateConstantFilter`)
    // ---------------------------------------------------------------------

    #[test]
    fn eq_renders_literal_between_bounds() {
        let table = ints();
        let predicate = bin(table.column("a"), Operator::Eq, lit(5i32));
        assert_eq!(
            table.sql(&predicate),
            "((col_1_stats.data_file_id IS NULL OR \
             ((col_1_stats.value_count IS NULL OR col_1_stats.value_count > 0) AND \
             (col_1_stats.min_value IS NULL OR col_1_stats.max_value IS NULL OR \
             5 BETWEEN TRY_CAST(col_1_stats.min_value AS INTEGER) AND \
             TRY_CAST(col_1_stats.max_value AS INTEGER))))) IS NOT FALSE"
        );
        assert_eq!(
            table.cte_stats(&predicate),
            vec!["min_value", "max_value", "value_count"]
        );
    }

    #[test]
    fn not_eq_renders_not_every_row_equal() {
        let table = ints();
        let predicate = bin(table.column("a"), Operator::NotEq, lit(5i32));
        assert_eq!(
            table.sql(&predicate),
            "((col_1_stats.data_file_id IS NULL OR \
             ((col_1_stats.value_count IS NULL OR col_1_stats.value_count > 0) AND \
             (col_1_stats.min_value IS NULL OR col_1_stats.max_value IS NULL OR \
             NOT (TRY_CAST(col_1_stats.min_value AS INTEGER) = 5 AND \
             TRY_CAST(col_1_stats.max_value AS INTEGER) = 5))))) IS NOT FALSE"
        );
    }

    #[test]
    fn gt_uses_max_only() {
        let table = ints();
        let predicate = bin(table.column("a"), Operator::Gt, lit(5i32));
        assert_eq!(
            table.sql(&predicate),
            "((col_1_stats.data_file_id IS NULL OR \
             ((col_1_stats.value_count IS NULL OR col_1_stats.value_count > 0) AND \
             (col_1_stats.max_value IS NULL OR \
             TRY_CAST(col_1_stats.max_value AS INTEGER) > 5)))) IS NOT FALSE"
        );
        assert_eq!(
            table.cte_stats(&predicate),
            vec!["max_value", "value_count"]
        );
    }

    #[test]
    fn gt_eq_uses_max_only() {
        let table = ints();
        let predicate = bin(table.column("a"), Operator::GtEq, lit(5i32));
        assert_eq!(
            table.sql(&predicate),
            "((col_1_stats.data_file_id IS NULL OR \
             ((col_1_stats.value_count IS NULL OR col_1_stats.value_count > 0) AND \
             (col_1_stats.max_value IS NULL OR \
             TRY_CAST(col_1_stats.max_value AS INTEGER) >= 5)))) IS NOT FALSE"
        );
    }

    #[test]
    fn lt_uses_min_only() {
        let table = ints();
        let predicate = bin(table.column("a"), Operator::Lt, lit(5i32));
        assert_eq!(
            table.sql(&predicate),
            "((col_1_stats.data_file_id IS NULL OR \
             ((col_1_stats.value_count IS NULL OR col_1_stats.value_count > 0) AND \
             (col_1_stats.min_value IS NULL OR \
             TRY_CAST(col_1_stats.min_value AS INTEGER) < 5)))) IS NOT FALSE"
        );
        assert_eq!(
            table.cte_stats(&predicate),
            vec!["min_value", "value_count"]
        );
    }

    #[test]
    fn lt_eq_uses_min_only() {
        let table = ints();
        let predicate = bin(table.column("a"), Operator::LtEq, lit(5i32));
        assert_eq!(
            table.sql(&predicate),
            "((col_1_stats.data_file_id IS NULL OR \
             ((col_1_stats.value_count IS NULL OR col_1_stats.value_count > 0) AND \
             (col_1_stats.min_value IS NULL OR \
             TRY_CAST(col_1_stats.min_value AS INTEGER) <= 5)))) IS NOT FALSE"
        );
    }

    #[test]
    fn is_null_uses_null_count() {
        let table = ints();
        let predicate = Arc::new(IsNullExpr::new(table.column("a"))) as Arc<dyn PhysicalExpr>;
        assert_eq!(
            table.sql(&predicate),
            "((col_1_stats.data_file_id IS NULL OR \
             (col_1_stats.null_count IS NULL OR col_1_stats.null_count > 0))) IS NOT FALSE"
        );
        assert_eq!(table.cte_stats(&predicate), vec!["null_count"]);
    }

    #[test]
    fn is_not_null_uses_value_count() {
        let table = ints();
        let predicate = Arc::new(IsNotNullExpr::new(table.column("a"))) as Arc<dyn PhysicalExpr>;
        assert_eq!(
            table.sql(&predicate),
            "((col_1_stats.data_file_id IS NULL OR \
             (col_1_stats.value_count IS NULL OR col_1_stats.value_count > 0))) IS NOT FALSE"
        );
        assert_eq!(table.cte_stats(&predicate), vec!["value_count"]);
    }

    #[test]
    fn in_list_is_an_or_of_equalities() {
        let table = ints();
        let predicate = in_list(
            table.column("a"),
            vec![lit(1i32), lit(2i32)],
            &false,
            &table.schema,
        )
        .expect("in list");
        assert_eq!(
            table.sql(&predicate),
            "((col_1_stats.data_file_id IS NULL OR \
             ((col_1_stats.value_count IS NULL OR col_1_stats.value_count > 0) AND \
             (col_1_stats.min_value IS NULL OR col_1_stats.max_value IS NULL OR \
             (1 BETWEEN TRY_CAST(col_1_stats.min_value AS INTEGER) AND \
             TRY_CAST(col_1_stats.max_value AS INTEGER)) OR \
             (2 BETWEEN TRY_CAST(col_1_stats.min_value AS INTEGER) AND \
             TRY_CAST(col_1_stats.max_value AS INTEGER)))))) IS NOT FALSE"
        );
        assert_eq!(
            table.cte_stats(&predicate),
            vec!["min_value", "max_value", "value_count"]
        );
    }

    #[test]
    fn string_comparison_is_raw_and_collated() {
        // Official does not cast VARCHAR (`RequiresValueComparison` is false), so
        // the bound is compared as stored text. Deviation 2 routes it through
        // `collate_binary`, which is the identity in DuckDB.
        let table = Fixture::new(&[("s", DataType::Utf8, 4)]);
        let predicate = bin(table.column("s"), Operator::Eq, lit("x'y"));
        assert_eq!(
            table.sql(&predicate),
            "((col_4_stats.data_file_id IS NULL OR \
             ((col_4_stats.value_count IS NULL OR col_4_stats.value_count > 0) AND \
             (col_4_stats.min_value IS NULL OR col_4_stats.max_value IS NULL OR \
             'x''y' BETWEEN col_4_stats.min_value AND col_4_stats.max_value)))) IS NOT FALSE"
        );
    }

    // ---------------------------------------------------------------------
    // G1 / G2 / G3 — the guards
    // ---------------------------------------------------------------------

    #[test]
    fn g1_every_condition_is_wrapped_in_the_no_stats_row_guard() {
        // A file written before the column existed LEFT JOINs to all-NULL and
        // must survive. Every shape gets the wrapper, guard or no guard — with
        // the unknown-keeps wrapper outside it.
        let table = ints();
        let a = table.column("a");
        let predicates: Vec<Arc<dyn PhysicalExpr>> = vec![
            bin(Arc::clone(&a), Operator::Gt, lit(5i32)),
            bin(Arc::clone(&a), Operator::Eq, lit(5i32)),
            Arc::new(IsNullExpr::new(Arc::clone(&a))),
            Arc::new(IsNotNullExpr::new(Arc::clone(&a))),
        ];
        for predicate in &predicates {
            let sql = table.sql(predicate);
            assert!(
                sql.starts_with("((col_1_stats.data_file_id IS NULL OR (")
                    && sql.ends_with(")) IS NOT FALSE"),
                "missing G1 wrapper: {sql}"
            );
        }
    }

    #[test]
    fn g2_one_is_null_disjunct_per_referenced_stat() {
        // `a > 5 OR a IS NULL` reads max_value and null_count, so both get a
        // fail-open disjunct — and nothing else does.
        let table = ints();
        let a = table.column("a");
        let predicate = bin(
            bin(Arc::clone(&a), Operator::Gt, lit(5i32)),
            Operator::Or,
            Arc::new(IsNullExpr::new(a)),
        );
        assert_eq!(
            table.sql(&predicate),
            "((col_1_stats.data_file_id IS NULL OR \
             (col_1_stats.max_value IS NULL OR col_1_stats.null_count IS NULL OR \
             (TRY_CAST(col_1_stats.max_value AS INTEGER) > 5) OR \
             (col_1_stats.null_count > 0)))) IS NOT FALSE"
        );
        assert_eq!(table.cte_stats(&predicate), vec!["max_value", "null_count"]);
    }

    #[test]
    fn g2_guard_only_value_count_is_selected_but_never_null_checked() {
        // The subtle half. `a > 5` does not read value_count; the G3 guard does.
        // Official builds its `IS NULL` disjuncts BEFORE inserting the guard
        // stat, so value_count appears in the CTE select list and inside the
        // guard, but never as an `IS NULL OR` disjunct. An `IS NULL` disjunct
        // here would make the guard vacuous.
        let table = ints();
        let predicate = bin(table.column("a"), Operator::Gt, lit(5i32));
        let rendered = table.only(&predicate);

        assert!(
            rendered.stats.contains(&"value_count"),
            "{:?}",
            rendered.stats
        );
        assert_eq!(
            rendered.condition,
            "((col_1_stats.data_file_id IS NULL OR \
             ((col_1_stats.value_count IS NULL OR col_1_stats.value_count > 0) AND \
             (col_1_stats.max_value IS NULL OR \
             TRY_CAST(col_1_stats.max_value AS INTEGER) > 5)))) IS NOT FALSE"
        );
        // Exactly one `value_count IS NULL` in the whole condition: the guard's.
        assert_eq!(
            rendered.condition.matches("value_count IS NULL").count(),
            1,
            "{}",
            rendered.condition
        );
        // And it is not one of the fail-open disjuncts.
        assert!(
            !rendered.condition.contains(
                "col_1_stats.value_count IS NULL OR \
                 col_1_stats.max_value"
            ),
            "{}",
            rendered.condition
        );

        let lowered = table.lower(&predicate).expect("lowered");
        let column = &lowered.columns[0];
        assert!(column.needs_value_count_guard);
        assert!(!column.referenced_stats.contains(&StatKind::ValueCount));
        assert!(column.cte_stats().contains(&StatKind::ValueCount));
    }

    #[test]
    fn g2_is_not_null_value_count_is_null_checked() {
        // The other half of the asymmetry: here value_count IS the filter, so it
        // does get a fail-open disjunct, and there is no G3 guard to add a
        // second mention of it.
        let table = ints();
        let predicate = Arc::new(IsNotNullExpr::new(table.column("a"))) as Arc<dyn PhysicalExpr>;
        let rendered = table.only(&predicate);
        assert_eq!(
            rendered.condition,
            "((col_1_stats.data_file_id IS NULL OR \
             (col_1_stats.value_count IS NULL OR col_1_stats.value_count > 0))) IS NOT FALSE"
        );
        assert_eq!(
            rendered.condition.matches("value_count IS NULL").count(),
            1,
            "{}",
            rendered.condition
        );

        let lowered = table.lower(&predicate).expect("lowered");
        let column = &lowered.columns[0];
        assert!(!column.needs_value_count_guard);
        assert!(column.referenced_stats.contains(&StatKind::ValueCount));
    }

    #[test]
    fn g3_value_count_guard_present_for_every_value_based_comparison() {
        let table = ints();
        let a = table.column("a");
        for op in [
            Operator::Eq,
            Operator::NotEq,
            Operator::Lt,
            Operator::LtEq,
            Operator::Gt,
            Operator::GtEq,
        ] {
            let predicate = bin(Arc::clone(&a), op, lit(5i32));
            let lowered = table.lower(&predicate).expect("lowered");
            assert!(
                lowered.columns[0].needs_value_count_guard,
                "missing guard for {op:?}"
            );
            assert!(
                table.sql(&predicate).contains(
                    "(col_1_stats.value_count IS NULL OR col_1_stats.value_count > 0) AND"
                ),
                "missing guard SQL for {op:?}"
            );
        }

        let in_predicate =
            in_list(Arc::clone(&a), vec![lit(1i32)], &false, &table.schema).expect("in list");
        assert!(table.lower(&in_predicate).expect("lowered").columns[0].needs_value_count_guard);
    }

    #[test]
    fn g3_value_count_guard_absent_for_is_null() {
        let table = ints();
        let predicate = Arc::new(IsNullExpr::new(table.column("a"))) as Arc<dyn PhysicalExpr>;
        let lowered = table.lower(&predicate).expect("lowered");
        assert!(!lowered.columns[0].needs_value_count_guard);
        assert!(!table.sql(&predicate).contains("value_count > 0"));
    }

    #[test]
    fn g3_value_count_guard_absent_for_is_not_null() {
        // Not because null_count is read, but because no bound is.
        let table = ints();
        let predicate = Arc::new(IsNotNullExpr::new(table.column("a"))) as Arc<dyn PhysicalExpr>;
        let lowered = table.lower(&predicate).expect("lowered");
        assert!(!lowered.columns[0].needs_value_count_guard);
    }

    #[test]
    fn g3_value_count_guard_absent_when_is_null_appears_anywhere() {
        // `a > 5 OR a IS NULL` can be satisfied by a NULL row, and an all-NULL
        // file has no bounds. Guarding it on value_count > 0 would prune a file
        // that matches.
        let table = ints();
        let a = table.column("a");
        let predicate = bin(
            bin(Arc::clone(&a), Operator::Gt, lit(5i32)),
            Operator::Or,
            Arc::new(IsNullExpr::new(a)),
        );
        let lowered = table.lower(&predicate).expect("lowered");
        assert!(!lowered.columns[0].needs_value_count_guard);
        let sql = table.sql(&predicate);
        assert!(
            !sql.contains("col_1_stats.value_count"),
            "value_count must not appear at all: {sql}"
        );
        assert!(!table.cte_stats(&predicate).contains(&"value_count"));
    }

    #[test]
    fn g3_value_count_guard_absent_when_is_null_is_a_separate_conjunct() {
        // Same rule after two conjuncts on one column intersect.
        let table = ints();
        let a = table.column("a");
        let predicate = bin(
            bin(Arc::clone(&a), Operator::Gt, lit(5i32)),
            Operator::And,
            Arc::new(IsNullExpr::new(a)),
        );
        let lowered = table.lower(&predicate).expect("lowered");
        assert!(!lowered.columns[0].needs_value_count_guard);
        assert_eq!(
            table.sql(&predicate),
            "((col_1_stats.data_file_id IS NULL OR \
             (col_1_stats.max_value IS NULL OR col_1_stats.null_count IS NULL OR \
             (TRY_CAST(col_1_stats.max_value AS INTEGER) > 5) AND \
             (col_1_stats.null_count > 0)))) IS NOT FALSE"
        );
    }

    // ---------------------------------------------------------------------
    // G4 / G8 / G9 / G10 / G11
    // ---------------------------------------------------------------------

    #[test]
    fn g4_column_lowering_to_nothing_contributes_no_entry() {
        // `b` is compared through a cast, so it pushes down nothing. It must
        // produce no condition, no CTE and no join — not an always-true one.
        let table = ints();
        let cast_b = Arc::new(CastExpr::new(table.column("b"), DataType::Int64, None));
        let predicate = bin(
            bin(table.column("a"), Operator::Gt, lit(5i32)),
            Operator::And,
            bin(cast_b, Operator::Gt, lit(5i64)),
        );
        let rendered = table.render(&predicate);
        assert_eq!(rendered.len(), 1);
        assert_eq!(rendered[0].column_id, 1);
        assert_eq!(rendered[0].alias, "col_1_stats");
    }

    #[test]
    fn g8_cast_subject_pushes_down_nothing() {
        let table = ints();
        let cast_a = Arc::new(CastExpr::new(table.column("a"), DataType::Int64, None));
        let predicate = bin(cast_a, Operator::Gt, lit(5i64));
        assert!(table.lowers_to_nothing(&predicate));
    }

    #[test]
    fn g8_column_to_column_comparison_pushes_down_nothing() {
        let table = ints();
        let predicate = bin(table.column("a"), Operator::Gt, table.column("b"));
        assert!(table.lowers_to_nothing(&predicate));
    }

    #[test]
    fn g9_constant_on_the_left_flips_the_operator() {
        let table = ints();
        let flipped = bin(lit(5i32), Operator::Lt, table.column("a"));
        let direct = bin(table.column("a"), Operator::Gt, lit(5i32));
        assert_eq!(table.sql(&flipped), table.sql(&direct));
        assert_eq!(
            table.sql(&flipped),
            "((col_1_stats.data_file_id IS NULL OR \
             ((col_1_stats.value_count IS NULL OR col_1_stats.value_count > 0) AND \
             (col_1_stats.max_value IS NULL OR \
             TRY_CAST(col_1_stats.max_value AS INTEGER) > 5)))) IS NOT FALSE"
        );

        // And the whole flip table, against the direct form.
        for (constant_op, column_op) in [
            (Operator::Eq, Operator::Eq),
            (Operator::NotEq, Operator::NotEq),
            (Operator::Lt, Operator::Gt),
            (Operator::LtEq, Operator::GtEq),
            (Operator::Gt, Operator::Lt),
            (Operator::GtEq, Operator::LtEq),
        ] {
            let flipped = bin(lit(5i32), constant_op, table.column("a"));
            let direct = bin(table.column("a"), column_op, lit(5i32));
            assert_eq!(
                table.sql(&flipped),
                table.sql(&direct),
                "{constant_op:?} should flip to {column_op:?}"
            );
        }
    }

    #[test]
    fn g10_and_drops_an_unlowerable_child_and_keeps_going() {
        // `(a > 5 AND CAST(a AS BIGINT) > 5) OR a < 2`. The nested AND loses its
        // second child; dropping a conjunct only weakens pruning, so the AND
        // keeps `a > 5` and the OR still lowers.
        let table = ints();
        let a = table.column("a");
        let cast_a = Arc::new(CastExpr::new(Arc::clone(&a), DataType::Int64, None));
        let predicate = bin(
            bin(
                bin(Arc::clone(&a), Operator::Gt, lit(5i32)),
                Operator::And,
                bin(cast_a, Operator::Gt, lit(5i64)),
            ),
            Operator::Or,
            bin(Arc::clone(&a), Operator::Lt, lit(2i32)),
        );
        assert_eq!(
            table.sql(&predicate),
            "((col_1_stats.data_file_id IS NULL OR \
             ((col_1_stats.value_count IS NULL OR col_1_stats.value_count > 0) AND \
             (col_1_stats.min_value IS NULL OR col_1_stats.max_value IS NULL OR \
             (TRY_CAST(col_1_stats.max_value AS INTEGER) > 5) OR \
             (TRY_CAST(col_1_stats.min_value AS INTEGER) < 2))))) IS NOT FALSE"
        );
    }

    #[test]
    fn g10_or_with_any_unlowerable_child_lowers_to_nothing() {
        // `a > 5 OR CAST(a AS BIGINT) > 5`. The right branch admits files the
        // left one does not, so keeping only the left branch would prune files
        // the predicate matches. The whole disjunction must be abandoned.
        let table = ints();
        let a = table.column("a");
        let cast_a = Arc::new(CastExpr::new(Arc::clone(&a), DataType::Int64, None));
        let predicate = bin(
            bin(Arc::clone(&a), Operator::Gt, lit(5i32)),
            Operator::Or,
            bin(cast_a, Operator::Gt, lit(5i64)),
        );
        assert!(
            table.lowers_to_nothing(&predicate),
            "an OR with an unlowerable branch must not prune: {:?}",
            table.lower(&predicate)
        );
    }

    #[test]
    fn g10_top_level_conjunct_that_cannot_lower_is_skipped() {
        // The outer AND is split into independent conjuncts; the unlowerable one
        // simply contributes nothing.
        let table = ints();
        let a = table.column("a");
        let cast_a = Arc::new(CastExpr::new(Arc::clone(&a), DataType::Int64, None));
        let predicate = bin(
            bin(cast_a, Operator::Gt, lit(5i64)),
            Operator::And,
            bin(a, Operator::Lt, lit(2i32)),
        );
        assert_eq!(
            table.sql(&predicate),
            "((col_1_stats.data_file_id IS NULL OR \
             ((col_1_stats.value_count IS NULL OR col_1_stats.value_count > 0) AND \
             (col_1_stats.min_value IS NULL OR \
             TRY_CAST(col_1_stats.min_value AS INTEGER) < 2)))) IS NOT FALSE"
        );
    }

    #[test]
    fn g11_cast_target_type_comes_from_the_constant_not_the_column() {
        // Column is INTEGER, constant is BIGINT. Official takes the target type
        // from the constant (`type ? *type : constant_expr->GetValue().type()`),
        // so the stat is cast to BIGINT.
        let table = ints();
        let predicate = bin(table.column("a"), Operator::Gt, lit(5i64));
        let sql = table.sql(&predicate);
        assert!(
            sql.contains("TRY_CAST(col_1_stats.max_value AS BIGINT)"),
            "{sql}"
        );
        assert!(!sql.contains("INTEGER"), "{sql}");
        assert_eq!(
            sql,
            "((col_1_stats.data_file_id IS NULL OR \
             ((col_1_stats.value_count IS NULL OR col_1_stats.value_count > 0) AND \
             (col_1_stats.max_value IS NULL OR \
             TRY_CAST(col_1_stats.max_value AS BIGINT) > 5)))) IS NOT FALSE"
        );
    }

    // ---------------------------------------------------------------------
    // Type refusals
    // ---------------------------------------------------------------------

    #[test]
    fn null_constant_pushes_down_nothing() {
        let table = ints();
        let predicate = bin(
            table.column("a"),
            Operator::Eq,
            lit(ScalarValue::Int32(None)),
        );
        assert!(table.lowers_to_nothing(&predicate));
    }

    #[test]
    fn binary_constants_push_down_nothing() {
        let table = Fixture::new(&[("blob", DataType::Binary, 1)]);
        let values = [
            ScalarValue::Binary(Some(vec![1, 2, 3])),
            ScalarValue::LargeBinary(Some(vec![1, 2, 3])),
            ScalarValue::BinaryView(Some(vec![1, 2, 3])),
            ScalarValue::FixedSizeBinary(3, Some(vec![1, 2, 3])),
        ];
        for value in values {
            let predicate = bin(table.column("blob"), Operator::Eq, lit(value.clone()));
            assert!(
                table.lowers_to_nothing(&predicate),
                "BLOB must never push down: {value:?}"
            );
        }
    }

    #[test]
    fn over_long_string_constant_pushes_down_nothing() {
        let table = Fixture::new(&[("s", DataType::Utf8, 1)]);
        let long = "x".repeat(stats_encode::MAX_STRING_STAT_BYTES + 1);
        let predicate = bin(table.column("s"), Operator::Eq, lit(long));
        assert!(table.lowers_to_nothing(&predicate));

        // The boundary length still pushes down, so the refusal is the length
        // rule and not a blanket string refusal.
        let exact = "x".repeat(stats_encode::MAX_STRING_STAT_BYTES);
        let predicate = bin(table.column("s"), Operator::Eq, lit(exact));
        assert!(!table.lowers_to_nothing(&predicate));
    }

    #[test]
    fn string_constant_with_nul_byte_pushes_down_nothing() {
        let table = Fixture::new(&[("s", DataType::Utf8, 1)]);
        let predicate = bin(table.column("s"), Operator::Eq, lit("ab\0cd"));
        assert!(table.lowers_to_nothing(&predicate));
    }

    #[test]
    fn nan_constant_pushes_down_nothing() {
        let table = Fixture::new(&[("f", DataType::Float64, 1)]);
        for op in [
            Operator::Eq,
            Operator::NotEq,
            Operator::Lt,
            Operator::LtEq,
            Operator::Gt,
            Operator::GtEq,
        ] {
            let predicate = bin(table.column("f"), op, lit(f64::NAN));
            assert!(
                table.lowers_to_nothing(&predicate),
                "NaN constant must not prune for {op:?}"
            );
        }
        let predicate = bin(table.column("f"), Operator::Gt, lit(f32::NAN));
        assert!(table.lowers_to_nothing(&predicate));
    }

    #[test]
    fn infinite_constant_pushes_down_nothing() {
        // Official routes non-finite floats through the quoted-literal path
        // (`ValueIsFinite` is false), so it would emit `'inf'` against a numeric
        // cast. This crate refuses them instead: the literal is quoted, and a
        // dialect that coerces text to a number reads `'inf'` as 0, so
        // `f < inf` would prune every file whose minimum is non-negative. The
        // read path has no Arrow representation for these bounds either, so
        // there is nothing on the other side of the comparison.
        let table = Fixture::new(&[("f", DataType::Float64, 1)]);
        for value in [f64::INFINITY, f64::NEG_INFINITY] {
            for op in [
                Operator::Eq,
                Operator::NotEq,
                Operator::Lt,
                Operator::LtEq,
                Operator::Gt,
                Operator::GtEq,
            ] {
                let predicate = bin(table.column("f"), op, lit(value));
                assert!(
                    table.lowers_to_nothing(&predicate),
                    "{value} must not prune for {op:?}"
                );
            }
        }

        // Float32 too, and inside an IN list it kills the whole list.
        let floats = Fixture::new(&[("g", DataType::Float32, 1)]);
        let predicate = bin(floats.column("g"), Operator::Lt, lit(f32::NEG_INFINITY));
        assert!(floats.lowers_to_nothing(&predicate));

        let predicate = in_list(
            table.column("f"),
            vec![lit(1.0f64), lit(f64::INFINITY)],
            &false,
            &table.schema,
        )
        .expect("in list");
        assert!(table.lowers_to_nothing(&predicate));
    }

    #[test]
    fn in_list_with_a_null_element_kills_the_whole_list() {
        let table = ints();
        let predicate = in_list(
            table.column("a"),
            vec![lit(1i32), lit(ScalarValue::Int32(None)), lit(3i32)],
            &false,
            &table.schema,
        )
        .expect("in list");
        assert!(table.lowers_to_nothing(&predicate));
    }

    #[test]
    fn in_list_with_a_non_literal_element_kills_the_whole_list() {
        let table = ints();
        let computed = bin(lit(2i32), Operator::Plus, lit(3i32));
        let predicate = in_list(
            table.column("a"),
            vec![lit(1i32), computed],
            &false,
            &table.schema,
        )
        .expect("in list");
        assert!(table.lowers_to_nothing(&predicate));
    }

    #[test]
    fn in_list_with_an_unencodable_element_kills_the_whole_list() {
        let table = Fixture::new(&[("f", DataType::Float64, 1)]);
        let predicate = in_list(
            table.column("f"),
            vec![lit(1.0f64), lit(f64::NAN)],
            &false,
            &table.schema,
        )
        .expect("in list");
        assert!(table.lowers_to_nothing(&predicate));
    }

    #[test]
    fn not_in_pushes_down_nothing() {
        let table = ints();
        let predicate = in_list(
            table.column("a"),
            vec![lit(1i32), lit(2i32)],
            &true,
            &table.schema,
        )
        .expect("not in list");
        assert!(table.lowers_to_nothing(&predicate));
    }

    #[test]
    fn empty_in_list_pushes_down_nothing() {
        let table = ints();
        let predicate =
            in_list(table.column("a"), vec![], &false, &table.schema).expect("empty in list");
        assert!(table.lowers_to_nothing(&predicate));
    }

    // ---------------------------------------------------------------------
    // Multi-column
    // ---------------------------------------------------------------------

    #[test]
    fn conjunct_spanning_two_columns_pushes_down_nothing() {
        // `a > 5 OR b < 3` is not a per-column zone-map condition. Turning it
        // into either half alone would prune files the other half admits.
        let table = ints();
        let predicate = bin(
            bin(table.column("a"), Operator::Gt, lit(5i32)),
            Operator::Or,
            bin(table.column("b"), Operator::Lt, lit(3i32)),
        );
        assert!(table.lowers_to_nothing(&predicate));
    }

    #[test]
    fn two_conjuncts_on_two_columns_render_in_column_id_order() {
        // Field order is (a, b) but `column_id` order is (2, 9), so the output
        // must be b then a.
        let table = Fixture::new(&[("a", DataType::Int32, 9), ("b", DataType::Int32, 2)]);
        let predicate = bin(
            bin(table.column("a"), Operator::Gt, lit(5i32)),
            Operator::And,
            bin(table.column("b"), Operator::Lt, lit(3i32)),
        );
        let rendered = table.render(&predicate);
        assert_eq!(
            rendered
                .iter()
                .map(|filter| filter.column_id)
                .collect::<Vec<_>>(),
            vec![2, 9]
        );
        assert_eq!(
            rendered
                .iter()
                .map(|filter| filter.alias.as_str())
                .collect::<Vec<_>>(),
            vec!["col_2_stats", "col_9_stats"]
        );
        assert_eq!(
            rendered[0].condition,
            "((col_2_stats.data_file_id IS NULL OR \
             ((col_2_stats.value_count IS NULL OR col_2_stats.value_count > 0) AND \
             (col_2_stats.min_value IS NULL OR \
             TRY_CAST(col_2_stats.min_value AS INTEGER) < 3)))) IS NOT FALSE"
        );
        assert_eq!(
            rendered[1].condition,
            "((col_9_stats.data_file_id IS NULL OR \
             ((col_9_stats.value_count IS NULL OR col_9_stats.value_count > 0) AND \
             (col_9_stats.max_value IS NULL OR \
             TRY_CAST(col_9_stats.max_value AS INTEGER) > 5)))) IS NOT FALSE"
        );
    }

    #[test]
    fn two_conjuncts_on_one_column_intersect_into_one_entry() {
        let table = ints();
        let a = table.column("a");
        let predicate = bin(
            bin(Arc::clone(&a), Operator::Gt, lit(5i32)),
            Operator::And,
            bin(a, Operator::Lt, lit(10i32)),
        );
        assert_eq!(
            table.sql(&predicate),
            "((col_1_stats.data_file_id IS NULL OR \
             ((col_1_stats.value_count IS NULL OR col_1_stats.value_count > 0) AND \
             (col_1_stats.min_value IS NULL OR col_1_stats.max_value IS NULL OR \
             (TRY_CAST(col_1_stats.max_value AS INTEGER) > 5) AND \
             (TRY_CAST(col_1_stats.min_value AS INTEGER) < 10))))) IS NOT FALSE"
        );
        assert_eq!(
            table.cte_stats(&predicate),
            vec!["min_value", "max_value", "value_count"]
        );
    }

    #[test]
    fn three_conjuncts_on_one_column_intersect_flat() {
        let table = ints();
        let a = table.column("a");
        let predicate = bin(
            bin(
                bin(Arc::clone(&a), Operator::Gt, lit(5i32)),
                Operator::And,
                bin(Arc::clone(&a), Operator::Lt, lit(10i32)),
            ),
            Operator::And,
            bin(a, Operator::NotEq, lit(7i32)),
        );
        assert_eq!(
            table.sql(&predicate),
            "((col_1_stats.data_file_id IS NULL OR \
             ((col_1_stats.value_count IS NULL OR col_1_stats.value_count > 0) AND \
             (col_1_stats.min_value IS NULL OR col_1_stats.max_value IS NULL OR \
             (TRY_CAST(col_1_stats.max_value AS INTEGER) > 5) AND \
             (TRY_CAST(col_1_stats.min_value AS INTEGER) < 10) AND \
             (NOT (TRY_CAST(col_1_stats.min_value AS INTEGER) = 7 AND \
             TRY_CAST(col_1_stats.max_value AS INTEGER) = 7)))))) IS NOT FALSE"
        );
    }

    // ---------------------------------------------------------------------
    // Authorized deviation 3 — the float gate
    // ---------------------------------------------------------------------

    #[test]
    fn float_bound_condition_is_gated_on_contains_nan() {
        let table = Fixture::new(&[("f", DataType::Float64, 3)]);
        let predicate = bin(table.column("f"), Operator::Lt, lit(5.0f64));
        assert_eq!(
            table.sql(&predicate),
            "((col_3_stats.data_file_id IS NULL OR \
             ((col_3_stats.value_count IS NULL OR col_3_stats.value_count > 0) AND \
             (col_3_stats.min_value IS NULL OR col_3_stats.contains_nan IS NULL OR \
             (col_3_stats.contains_nan IS NULL OR col_3_stats.contains_nan <> false) OR \
             (TRY_CAST(col_3_stats.min_value AS DOUBLE) < 5.0))))) IS NOT FALSE"
        );
        assert_eq!(
            table.cte_stats(&predicate),
            vec!["min_value", "value_count", "contains_nan"]
        );
    }

    #[test]
    fn float32_bound_condition_is_gated_too() {
        let table = Fixture::new(&[("f", DataType::Float32, 3)]);
        let predicate = bin(table.column("f"), Operator::Gt, lit(5.0f32));
        assert_eq!(
            table.sql(&predicate),
            "((col_3_stats.data_file_id IS NULL OR \
             ((col_3_stats.value_count IS NULL OR col_3_stats.value_count > 0) AND \
             (col_3_stats.max_value IS NULL OR col_3_stats.contains_nan IS NULL OR \
             (col_3_stats.contains_nan IS NULL OR col_3_stats.contains_nan <> false) OR \
             (TRY_CAST(col_3_stats.max_value AS FLOAT) > 5.0))))) IS NOT FALSE"
        );
    }

    #[test]
    fn float_is_null_is_not_gated() {
        // `IS NULL` reads only null_count, which NaN cannot corrupt.
        let table = Fixture::new(&[("f", DataType::Float64, 3)]);
        let predicate = Arc::new(IsNullExpr::new(table.column("f"))) as Arc<dyn PhysicalExpr>;
        assert_eq!(
            table.sql(&predicate),
            "((col_3_stats.data_file_id IS NULL OR \
             (col_3_stats.null_count IS NULL OR col_3_stats.null_count > 0))) IS NOT FALSE"
        );
        assert_eq!(table.cte_stats(&predicate), vec!["null_count"]);
    }

    #[test]
    fn float_is_not_null_is_not_gated() {
        let table = Fixture::new(&[("f", DataType::Float64, 3)]);
        let predicate = Arc::new(IsNotNullExpr::new(table.column("f"))) as Arc<dyn PhysicalExpr>;
        assert_eq!(
            table.sql(&predicate),
            "((col_3_stats.data_file_id IS NULL OR \
             (col_3_stats.value_count IS NULL OR col_3_stats.value_count > 0))) IS NOT FALSE"
        );
        assert!(!table.cte_stats(&predicate).contains(&"contains_nan"));
    }

    #[test]
    fn non_float_column_is_never_gated() {
        let table = ints();
        for predicate in [
            bin(table.column("a"), Operator::Gt, lit(5i32)),
            bin(table.column("a"), Operator::Eq, lit(5i32)),
            bin(table.column("a"), Operator::NotEq, lit(5i32)),
        ] {
            let sql = table.sql(&predicate);
            assert!(!sql.contains("contains_nan"), "{sql}");
            assert!(!table.cte_stats(&predicate).contains(&"contains_nan"));
        }

        let dates = Fixture::new(&[("d", DataType::Date32, 1)]);
        let predicate = bin(
            dates.column("d"),
            Operator::Gt,
            lit(ScalarValue::Date32(Some(1))),
        );
        assert!(!dates.sql(&predicate).contains("contains_nan"));
    }

    #[test]
    fn float_equals_nan_lowers_to_nothing() {
        // Official answers `x = NaN` with `contains_nan` alone. This crate has no
        // canonical text for NaN, so the comparison is refused outright, which
        // keeps every file — strictly less pruning, never a lost row.
        let table = Fixture::new(&[("f", DataType::Float64, 3)]);
        let predicate = bin(table.column("f"), Operator::Eq, lit(f64::NAN));
        assert!(table.lowers_to_nothing(&predicate));
    }

    /// Three-valued OR, as SQL evaluates it.
    fn or3(left: Option<bool>, right: Option<bool>) -> Option<bool> {
        match (left, right) {
            (Some(true), _) | (_, Some(true)) => Some(true),
            (Some(false), Some(false)) => Some(false),
            _ => None,
        }
    }

    /// Three-valued AND, as SQL evaluates it.
    fn and3(left: Option<bool>, right: Option<bool>) -> Option<bool> {
        match (left, right) {
            (Some(false), _) | (_, Some(false)) => Some(false),
            (Some(true), Some(true)) => Some(true),
            _ => None,
        }
    }

    #[test]
    fn float_gate_admits_exactly_the_same_files_as_official_for_gt_and_gt_eq() {
        // The equivalence claim in the module docs, for the two operators where
        // official also guards on `contains_nan`.
        //
        // Official renders, for `f > C` on a DOUBLE column:
        //     max_value IS NULL OR contains_nan IS NULL
        //       OR (TRY_CAST(max_value AS DOUBLE) > C OR contains_nan)
        // This crate renders:
        //     max_value IS NULL OR contains_nan IS NULL
        //       OR ((contains_nan IS NULL OR contains_nan <> false)
        //           OR TRY_CAST(max_value AS DOUBLE) > C)
        //
        // Both sit under the same G1 wrapper and the same G3 value_count guard,
        // so the only question is the bracketed tail, over the three states of
        // `contains_nan`:
        //
        //   contains_nan IS NULL — the shared `contains_nan IS NULL` disjunct is
        //     TRUE, so both admit the file regardless of the bound. (In official
        //     the tail's `OR contains_nan` would be NULL, not TRUE, which is
        //     precisely why that disjunct has to be there.)
        //   contains_nan = true — official's `OR contains_nan` is TRUE; this
        //     crate's `contains_nan <> false` is TRUE. Both admit.
        //   contains_nan = false — official's `OR contains_nan` is FALSE and this
        //     crate's gate is FALSE, so both reduce to `max > C` (with the
        //     `max_value IS NULL` disjunct still failing open). Identical.
        //
        // Hence: same admitted set, not merely a superset. The deviation costs
        // pruning only for `<`, `<=`, `=` and `<>`, where official does not look
        // at `contains_nan` at all.
        let constant = 5.0_f64;
        for contains_nan in [None, Some(true), Some(false)] {
            for max_value in [None, Some(1.0_f64), Some(9.0_f64)] {
                for gt_eq in [false, true] {
                    let bound_matches = max_value.map(|max| {
                        if gt_eq {
                            max >= constant
                        } else {
                            max > constant
                        }
                    });

                    let official = or3(
                        or3(Some(max_value.is_none()), Some(contains_nan.is_none())),
                        or3(bound_matches, contains_nan),
                    );
                    let gate = or3(Some(contains_nan.is_none()), contains_nan);
                    let ours = or3(
                        or3(Some(max_value.is_none()), Some(contains_nan.is_none())),
                        or3(gate, bound_matches),
                    );

                    assert_eq!(
                        official == Some(true),
                        ours == Some(true),
                        "diverged for contains_nan={contains_nan:?} \
                         max_value={max_value:?} gt_eq={gt_eq}"
                    );
                    // The unknown-keeps wrapper is inert across this state
                    // space: with every stat either SQL NULL or well formed, an
                    // outer `IS NULL` disjunct always rescues an unknown tail,
                    // so the condition is never itself NULL and `IS NOT FALSE`
                    // agrees with `= TRUE`. The wrapper changes the answer only
                    // for a stat that is present but will not cast, which is
                    // `keep_when_unknown_rescues_a_malformed_stat` below.
                    assert_ne!(ours, None, "condition should not be unknown here");
                    assert_ne!(official, None, "condition should not be unknown here");
                }
            }
        }

        // The value_count guard is shared, so ANDing it in cannot separate them.
        for value_count in [None, Some(0_i64), Some(3_i64)] {
            let guard = or3(
                Some(value_count.is_none()),
                value_count.map(|count| count > 0),
            );
            assert_eq!(
                and3(guard, Some(true)) == Some(true),
                value_count != Some(0),
                "value_count guard model wrong for {value_count:?}"
            );
        }
    }

    // ---------------------------------------------------------------------
    // Determinism
    // ---------------------------------------------------------------------

    #[test]
    fn rendering_is_byte_identical_across_runs() {
        let table = Fixture::new(&[
            ("a", DataType::Int32, 9),
            ("b", DataType::Utf8, 2),
            ("f", DataType::Float64, 5),
        ]);
        let build = || {
            bin(
                bin(
                    bin(table.column("a"), Operator::Gt, lit(5i32)),
                    Operator::And,
                    bin(table.column("b"), Operator::Eq, lit("x")),
                ),
                Operator::And,
                bin(table.column("f"), Operator::Lt, lit(2.5f64)),
            )
        };
        let first = table.render(&build());
        for _ in 0..8 {
            assert_eq!(table.render(&build()), first);
        }
        assert_eq!(
            first
                .iter()
                .map(|filter| filter.column_id)
                .collect::<Vec<_>>(),
            vec![2, 5, 9],
            "columns must come out sorted by column_id"
        );
    }

    // ---------------------------------------------------------------------
    // Dialect seam
    // ---------------------------------------------------------------------

    #[test]
    fn declined_cast_drops_only_that_comparison_inside_an_and() {
        let table = ints();
        let a = table.column("a");
        let predicate = bin(
            bin(Arc::clone(&a), Operator::Gt, lit(5i32)),
            Operator::And,
            bin(a, Operator::Gt, lit(6i64)),
        );
        // The reference dialect keeps both.
        assert_eq!(
            table.sql(&predicate),
            "((col_1_stats.data_file_id IS NULL OR \
             ((col_1_stats.value_count IS NULL OR col_1_stats.value_count > 0) AND \
             (col_1_stats.max_value IS NULL OR \
             (TRY_CAST(col_1_stats.max_value AS INTEGER) > 5) AND \
             (TRY_CAST(col_1_stats.max_value AS BIGINT) > 6))))) IS NOT FALSE"
        );

        // One that cannot spell TRY_CAST for BIGINT keeps the other half.
        let rendered = table
            .render_with(&predicate, &NoBigint)
            .expect("the INTEGER comparison still renders");
        assert_eq!(rendered.len(), 1);
        assert_eq!(
            rendered[0].condition,
            "((col_1_stats.data_file_id IS NULL OR \
             ((col_1_stats.value_count IS NULL OR col_1_stats.value_count > 0) AND \
             (col_1_stats.max_value IS NULL OR \
             (TRY_CAST(col_1_stats.max_value AS INTEGER) > 5))))) IS NOT FALSE"
        );
    }

    #[test]
    fn declined_cast_collapses_a_whole_or_but_leaves_other_columns() {
        let table = ints();
        let a = table.column("a");
        let predicate = bin(
            bin(
                bin(Arc::clone(&a), Operator::Gt, lit(5i32)),
                Operator::Or,
                bin(a, Operator::Gt, lit(6i64)),
            ),
            Operator::And,
            bin(table.column("b"), Operator::Lt, lit(3i32)),
        );
        // Reference dialect: both columns survive.
        assert_eq!(
            table
                .render(&predicate)
                .iter()
                .map(|filter| filter.column_id)
                .collect::<Vec<_>>(),
            vec![1, 2]
        );

        // Declining BIGINT kills the OR outright, so column 1 contributes
        // nothing — no condition, no CTE, no join — while column 2 is untouched.
        let rendered = table
            .render_with(&predicate, &NoBigint)
            .expect("column b still renders");
        assert_eq!(rendered.len(), 1);
        assert_eq!(rendered[0].column_id, 2);
        assert_eq!(
            rendered[0].condition,
            "((col_2_stats.data_file_id IS NULL OR \
             ((col_2_stats.value_count IS NULL OR col_2_stats.value_count > 0) AND \
             (col_2_stats.min_value IS NULL OR \
             TRY_CAST(col_2_stats.min_value AS INTEGER) < 3)))) IS NOT FALSE"
        );
    }

    #[test]
    fn declining_every_cast_renders_nothing_at_all() {
        let table = ints();
        let predicate = bin(table.column("a"), Operator::Gt, lit(6i64));
        assert!(table.render_with(&predicate, &NoBigint).is_none());
    }

    // ---------------------------------------------------------------------
    // Shapes official's `default:` arms refuse
    // ---------------------------------------------------------------------

    #[test]
    fn not_expression_pushes_down_nothing() {
        // Official's BOUND_OPERATOR switch has no OPERATOR_NOT arm.
        let table = ints();
        let inner = bin(table.column("a"), Operator::Gt, lit(5i32));
        let predicate = Arc::new(NotExpr::new(inner)) as Arc<dyn PhysicalExpr>;
        assert!(table.lowers_to_nothing(&predicate));
    }

    #[test]
    fn non_comparison_operators_push_down_nothing() {
        let table = ints();
        for op in
            [Operator::IsDistinctFrom, Operator::IsNotDistinctFrom, Operator::Plus, Operator::Minus]
        {
            let predicate = bin(table.column("a"), op, lit(5i32));
            assert!(table.lowers_to_nothing(&predicate), "{op:?} must not prune");
        }
    }

    #[test]
    fn is_null_on_a_cast_subject_pushes_down_nothing() {
        // Official requires `IsSimpleFilterSubject` on the IS NULL child too.
        let table = ints();
        let cast_a = Arc::new(CastExpr::new(table.column("a"), DataType::Int64, None));
        let predicate = Arc::new(IsNullExpr::new(cast_a)) as Arc<dyn PhysicalExpr>;
        assert!(table.lowers_to_nothing(&predicate));
    }

    #[test]
    fn is_not_null_on_a_cast_subject_pushes_down_nothing() {
        let table = ints();
        let cast_a = Arc::new(CastExpr::new(table.column("a"), DataType::Int64, None));
        let predicate = Arc::new(IsNotNullExpr::new(cast_a)) as Arc<dyn PhysicalExpr>;
        assert!(table.lowers_to_nothing(&predicate));
    }

    #[test]
    fn in_list_with_a_cast_subject_pushes_down_nothing() {
        let table = ints();
        let cast_a = Arc::new(CastExpr::new(table.column("a"), DataType::Int64, None));
        let predicate =
            in_list(cast_a, vec![lit(1i64), lit(2i64)], &false, &table.schema).expect("in list");
        assert!(table.lowers_to_nothing(&predicate));
    }

    #[test]
    fn predicate_with_no_column_reference_pushes_down_nothing() {
        let table = ints();
        let predicate = bin(lit(1i32), Operator::Gt, lit(0i32));
        assert!(table.lowers_to_nothing(&predicate));
    }

    #[test]
    fn and_with_both_children_unlowerable_collapses_its_or_parent() {
        // The AND contributes nothing, and an OR with a nothing branch is
        // abandoned wholesale.
        let table = ints();
        let a = table.column("a");
        let cast_a = || Arc::new(CastExpr::new(table.column("a"), DataType::Int64, None));
        let predicate = bin(
            bin(
                bin(cast_a(), Operator::Gt, lit(5i64)),
                Operator::And,
                bin(cast_a(), Operator::Lt, lit(9i64)),
            ),
            Operator::Or,
            bin(a, Operator::Lt, lit(2i32)),
        );
        assert!(table.lowers_to_nothing(&predicate));
    }

    // ---------------------------------------------------------------------
    // Constant rendering: cast target and quoting, per `RequiresValueComparison`
    // ---------------------------------------------------------------------

    #[test]
    fn boolean_constant_is_cast_but_quoted() {
        // BOOLEAN requires value comparison (so the stat is cast) but is not
        // numeric (so the constant is a quoted literal). Official does exactly
        // this pair.
        let table = Fixture::new(&[("flag", DataType::Boolean, 1)]);
        let predicate = bin(table.column("flag"), Operator::Eq, lit(true));
        assert_eq!(
            table.sql(&predicate),
            "((col_1_stats.data_file_id IS NULL OR \
             ((col_1_stats.value_count IS NULL OR col_1_stats.value_count > 0) AND \
             (col_1_stats.min_value IS NULL OR col_1_stats.max_value IS NULL OR \
             'true' BETWEEN TRY_CAST(col_1_stats.min_value AS BOOLEAN) AND \
             TRY_CAST(col_1_stats.max_value AS BOOLEAN))))) IS NOT FALSE"
        );
    }

    #[test]
    fn decimal_constant_is_cast_and_unquoted() {
        let table = Fixture::new(&[("d", DataType::Decimal128(5, 2), 1)]);
        let predicate = bin(
            table.column("d"),
            Operator::Gt,
            lit(ScalarValue::Decimal128(Some(525), 5, 2)),
        );
        assert_eq!(
            table.sql(&predicate),
            "((col_1_stats.data_file_id IS NULL OR \
             ((col_1_stats.value_count IS NULL OR col_1_stats.value_count > 0) AND \
             (col_1_stats.max_value IS NULL OR \
             TRY_CAST(col_1_stats.max_value AS DECIMAL(5,2)) > 5.25)))) IS NOT FALSE"
        );
    }

    #[test]
    fn date_constant_is_cast_and_quoted() {
        let table = Fixture::new(&[("d", DataType::Date32, 1)]);
        let predicate = bin(
            table.column("d"),
            Operator::GtEq,
            lit(ScalarValue::Date32(Some(0))),
        );
        assert_eq!(
            table.sql(&predicate),
            "((col_1_stats.data_file_id IS NULL OR \
             ((col_1_stats.value_count IS NULL OR col_1_stats.value_count > 0) AND \
             (col_1_stats.max_value IS NULL OR \
             TRY_CAST(col_1_stats.max_value AS DATE) >= '1970-01-01')))) IS NOT FALSE"
        );
    }

    #[test]
    fn timestamp_constant_is_cast_and_quoted() {
        let table = Fixture::new(&[("t", DataType::Timestamp(TimeUnit::Microsecond, None), 1)]);
        let value = ScalarValue::TimestampMicrosecond(Some(0), None);
        let encoded = stats_encode::encode_scalar(&value).expect("encodable");
        let predicate = bin(table.column("t"), Operator::Lt, lit(value));
        assert_eq!(
            table.sql(&predicate),
            format!(
                "((col_1_stats.data_file_id IS NULL OR \
                 ((col_1_stats.value_count IS NULL OR col_1_stats.value_count > 0) AND \
                 (col_1_stats.min_value IS NULL OR \
                 TRY_CAST(col_1_stats.min_value AS TIMESTAMP) < '{encoded}')))) IS NOT FALSE"
            )
        );
    }

    #[test]
    fn negative_constant_renders_with_its_sign() {
        let table = ints();
        let predicate = bin(table.column("a"), Operator::Gt, lit(-5i32));
        assert!(
            table
                .sql(&predicate)
                .contains("TRY_CAST(col_1_stats.max_value AS INTEGER) > -5"),
            "{}",
            table.sql(&predicate)
        );
    }

    // ---------------------------------------------------------------------
    // Parenthesisation under nesting
    // ---------------------------------------------------------------------

    #[test]
    fn nested_or_inside_and_inside_or_keeps_its_brackets() {
        // `((a = 1 OR a = 2) AND (a = 3 OR a = 4)) OR a = 5`. The inner AND must
        // stay bracketed: SQL binds AND tighter than OR, so losing one pair of
        // brackets here would silently change which files are admitted.
        let table = ints();
        let a = table.column("a");
        let eq = |value: i32| bin(Arc::clone(&a), Operator::Eq, lit(value));
        let predicate = bin(
            bin(
                bin(eq(1), Operator::Or, eq(2)),
                Operator::And,
                bin(eq(3), Operator::Or, eq(4)),
            ),
            Operator::Or,
            eq(5),
        );
        let between = |value: i32| {
            format!(
                "{value} BETWEEN TRY_CAST(col_1_stats.min_value AS INTEGER) AND \
                 TRY_CAST(col_1_stats.max_value AS INTEGER)"
            )
        };
        assert_eq!(
            table.sql(&predicate),
            format!(
                "((col_1_stats.data_file_id IS NULL OR \
                 ((col_1_stats.value_count IS NULL OR col_1_stats.value_count > 0) AND \
                 (col_1_stats.min_value IS NULL OR col_1_stats.max_value IS NULL OR \
                 ((({}) OR ({})) AND (({}) OR ({}))) OR ({}))))) IS NOT FALSE",
                between(1),
                between(2),
                between(3),
                between(4),
                between(5),
            )
        );
    }

    // ---------------------------------------------------------------------
    // More float-gate coverage
    // ---------------------------------------------------------------------

    #[test]
    fn float_eq_is_not_gated() {
        // Deliberately a separate test from `<>`. Under `total_cmp` a NaN of
        // either sign is equal to no finite value, and a NaN constant is refused
        // outright, so every row satisfying `f = C` is a non-NaN row — and
        // non-NaN rows are bounded by the recorded min/max. Official leaves
        // COMPARE_EQUAL ungated for the same reason.
        let table = Fixture::new(&[("f", DataType::Float64, 3)]);
        let predicate = bin(table.column("f"), Operator::Eq, lit(1.5f64));
        assert_eq!(
            table.sql(&predicate),
            "((col_3_stats.data_file_id IS NULL OR \
             ((col_3_stats.value_count IS NULL OR col_3_stats.value_count > 0) AND \
             (col_3_stats.min_value IS NULL OR col_3_stats.max_value IS NULL OR \
             1.5 BETWEEN TRY_CAST(col_3_stats.min_value AS DOUBLE) AND \
             TRY_CAST(col_3_stats.max_value AS DOUBLE))))) IS NOT FALSE"
        );
        assert_eq!(
            table.cte_stats(&predicate),
            vec!["min_value", "max_value", "value_count"],
            "an ungated equality must not drag contains_nan into the CTE"
        );
    }

    #[test]
    fn float_not_eq_is_still_gated() {
        // The other half, and the reason the two are separate tests: `NaN <> C`
        // is TRUE for every C, so a file holding a NaN satisfies this predicate
        // — and the recorded bounds exclude NaN, so they cannot prove otherwise.
        // A change that ungates `<>` alongside `=` would lose rows.
        let table = Fixture::new(&[("f", DataType::Float64, 3)]);
        let predicate = bin(table.column("f"), Operator::NotEq, lit(1.5f64));
        assert_eq!(
            table.sql(&predicate),
            "((col_3_stats.data_file_id IS NULL OR \
             ((col_3_stats.value_count IS NULL OR col_3_stats.value_count > 0) AND \
             (col_3_stats.min_value IS NULL OR col_3_stats.max_value IS NULL OR \
             col_3_stats.contains_nan IS NULL OR \
             (col_3_stats.contains_nan IS NULL OR col_3_stats.contains_nan <> false) OR \
             (NOT (TRY_CAST(col_3_stats.min_value AS DOUBLE) = 1.5 AND \
             TRY_CAST(col_3_stats.max_value AS DOUBLE) = 1.5)))))) IS NOT FALSE"
        );
        assert_eq!(
            table.cte_stats(&predicate),
            vec!["min_value", "max_value", "value_count", "contains_nan"]
        );
    }

    #[test]
    fn float_in_list_is_not_gated() {
        // `IN` lowers to an OR of equality forms, and `needs_float_gate`
        // recurses through the OR finding only `LiteralWithinBounds`, so the
        // whole disjunction is ungated for the same reason a single `=` is.
        let table = Fixture::new(&[("f", DataType::Float64, 3)]);
        let predicate = in_list(
            table.column("f"),
            vec![lit(1.0f64), lit(2.0f64)],
            &false,
            &table.schema,
        )
        .expect("in list");
        let sql = table.sql(&predicate);
        assert!(!sql.contains("contains_nan"), "{sql}");
        assert_eq!(
            sql,
            "((col_3_stats.data_file_id IS NULL OR \
             ((col_3_stats.value_count IS NULL OR col_3_stats.value_count > 0) AND \
             (col_3_stats.min_value IS NULL OR col_3_stats.max_value IS NULL OR \
             (1.0 BETWEEN TRY_CAST(col_3_stats.min_value AS DOUBLE) AND \
             TRY_CAST(col_3_stats.max_value AS DOUBLE)) OR \
             (2.0 BETWEEN TRY_CAST(col_3_stats.min_value AS DOUBLE) AND \
             TRY_CAST(col_3_stats.max_value AS DOUBLE)))))) IS NOT FALSE"
        );
        assert_eq!(
            table.cte_stats(&predicate),
            vec!["min_value", "max_value", "value_count"]
        );
    }

    #[test]
    fn float_column_casts_to_the_constant_type_and_is_still_gated() {
        // G11 on a float column: the cast follows the INTEGER constant, but the
        // gate follows the column, because it is the column's stored bounds that
        // exclude NaN.
        let table = Fixture::new(&[("f", DataType::Float64, 3)]);
        let predicate = bin(table.column("f"), Operator::Gt, lit(5i32));
        let sql = table.sql(&predicate);
        assert!(
            sql.contains("TRY_CAST(col_3_stats.max_value AS INTEGER) > 5"),
            "{sql}"
        );
        assert!(sql.contains("contains_nan <> false"), "{sql}");
    }

    #[test]
    fn integer_column_with_a_float_constant_is_not_gated() {
        // Documents a divergence. Official keys its NaN handling off the
        // CONSTANT's type, so `a > 5.0` on an INTEGER column would get official's
        // `OR contains_nan`. This crate keys the gate off the COLUMN's type and
        // omits it. An integer column's `contains_nan` can never be true, so the
        // two admit the same files; it is only the SQL text that differs.
        let table = ints();
        let predicate = bin(table.column("a"), Operator::Gt, lit(5.0f64));
        let sql = table.sql(&predicate);
        assert!(!sql.contains("contains_nan"), "{sql}");
        assert!(
            sql.contains("TRY_CAST(col_1_stats.max_value AS DOUBLE) > 5.0"),
            "{sql}"
        );
    }

    #[test]
    fn a_dialect_declining_the_float_cast_drops_the_column_gate_and_all() {
        // The gate alone proves nothing, so it must never survive on its own.
        struct NoDouble;
        impl StatsSqlDialect for NoDouble {
            fn try_cast(
                &self,
                expr: &str,
                literal: &StatsLiteral,
                data_type: &DataType,
            ) -> Option<String> {
                if matches!(data_type, DataType::Float64) {
                    return None;
                }
                Duck.try_cast(expr, literal, data_type)
            }
            fn collate_binary(&self, expr: &str) -> String {
                Duck.collate_binary(expr)
            }
            fn boolean_is_not_false(&self, expr: &str) -> String {
                Duck.boolean_is_not_false(expr)
            }
        }

        let table = Fixture::new(&[("f", DataType::Float64, 3)]);
        let predicate = bin(table.column("f"), Operator::Gt, lit(5.0f64));
        assert!(table.render_with(&predicate, &Duck).is_some());
        assert!(table.render_with(&predicate, &NoDouble).is_none());
    }

    // ---------------------------------------------------------------------
    // Where the exact stat set matters
    // ---------------------------------------------------------------------

    #[test]
    fn a_dropped_branch_leaves_no_stat_behind() {
        // Official threads ONE `referenced_stats` set through the whole
        // recursion and mutates it in place, so a branch that inserts a stat and
        // then fails leaves that stat in the set. Here `a IS NULL` would insert
        // `null_count` before the sibling `CAST(a AS BIGINT) > 5` kills the OR;
        // official would then see `null_count` referenced and drop the G3
        // value_count guard.
        //
        // This crate lowers each branch into its own set and merges only on
        // success, so the surviving condition (`a > 5`) is described exactly and
        // keeps its guard. That prunes MORE than official — soundly: a file of
        // nothing but NULLs cannot satisfy `a > 5`, which is the only condition
        // that survived.
        let table = ints();
        let a = table.column("a");
        let cast_a = Arc::new(CastExpr::new(table.column("a"), DataType::Int64, None));
        let predicate = bin(
            bin(
                Arc::new(IsNullExpr::new(Arc::clone(&a))) as Arc<dyn PhysicalExpr>,
                Operator::Or,
                bin(cast_a, Operator::Gt, lit(5i64)),
            ),
            Operator::And,
            bin(a, Operator::Gt, lit(5i32)),
        );
        let lowered = table.lower(&predicate).expect("lowered");
        assert_eq!(
            lowered.columns[0]
                .referenced_stats
                .iter()
                .copied()
                .collect::<Vec<_>>(),
            vec![StatKind::MaxValue]
        );
        assert!(lowered.columns[0].needs_value_count_guard);
        assert_eq!(
            table.sql(&predicate),
            "((col_1_stats.data_file_id IS NULL OR \
             ((col_1_stats.value_count IS NULL OR col_1_stats.value_count > 0) AND \
             (col_1_stats.max_value IS NULL OR \
             TRY_CAST(col_1_stats.max_value AS INTEGER) > 5)))) IS NOT FALSE"
        );
    }

    #[test]
    fn a_render_time_drop_keeps_the_stat_it_no_longer_reads() {
        // The mirror image, and deliberately fail-open. Lowering records
        // min_value and max_value; the dialect then declines BIGINT so the
        // min_value comparison disappears at render time. The stale
        // `min_value IS NULL OR` disjunct and the stale CTE column stay, which
        // can only keep more files, never fewer. Pinned so that "tidying" the
        // stat set later has to be a deliberate decision.
        let table = ints();
        let a = table.column("a");
        let predicate = bin(
            bin(Arc::clone(&a), Operator::Gt, lit(5i32)),
            Operator::And,
            bin(a, Operator::Lt, lit(10i64)),
        );
        let rendered = table
            .render_with(&predicate, &NoBigint)
            .expect("the INTEGER comparison survives");
        assert_eq!(rendered.len(), 1);
        assert_eq!(
            rendered[0].stats,
            vec!["min_value", "max_value", "value_count"]
        );
        assert_eq!(
            rendered[0].condition,
            "((col_1_stats.data_file_id IS NULL OR \
             ((col_1_stats.value_count IS NULL OR col_1_stats.value_count > 0) AND \
             (col_1_stats.min_value IS NULL OR col_1_stats.max_value IS NULL OR \
             (TRY_CAST(col_1_stats.max_value AS INTEGER) > 5))))) IS NOT FALSE"
        );
    }

    #[test]
    fn float16_constant_pushes_down_nothing() {
        // `encode_scalar` has no Float16 arm, so there is no canonical text to
        // compare a stored bound against.
        type F16 = <arrow::datatypes::Float16Type as ArrowPrimitiveType>::Native;
        let table = Fixture::new(&[("h", DataType::Float16, 1)]);
        let predicate = bin(
            table.column("h"),
            Operator::Gt,
            lit(ScalarValue::Float16(Some(F16::from_f32(1.5)))),
        );
        assert!(table.lowers_to_nothing(&predicate));
    }

    // ---------------------------------------------------------------------
    // The float gate against official, operator by operator
    // ---------------------------------------------------------------------

    /// Three-valued NOT.
    fn not3(value: Option<bool>) -> Option<bool> {
        value.map(|inner| !inner)
    }

    /// Every stats state a float column's row can be in, for a `5.0` constant.
    fn float_stats_states() -> Vec<(Option<bool>, Option<f64>, Option<f64>)> {
        let mut states = Vec::new();
        for contains_nan in [None, Some(true), Some(false)] {
            for min_value in [None, Some(1.0_f64), Some(5.0_f64)] {
                for max_value in [None, Some(5.0_f64), Some(9.0_f64)] {
                    states.push((contains_nan, min_value, max_value));
                }
            }
        }
        states
    }

    /// This crate's gate: `contains_nan IS NULL OR contains_nan <> false`.
    fn gate(contains_nan: Option<bool>) -> Option<bool> {
        or3(Some(contains_nan.is_none()), contains_nan)
    }

    #[test]
    fn float_gate_admits_exactly_official_set_for_not_eq_too() {
        // The module docs list `<>` among the operators where this crate is
        // "strictly more conservative" than official. It is not: official's
        // `GenerateConstantFilterDouble` appends `OR contains_nan` for
        // COMPARE_NOTEQUAL as well as for `>` and `>=`, so `<>` belongs in the
        // same exact-equivalence class. Asserted here so the claim is checked
        // rather than assumed.
        let constant = 5.0_f64;
        for (contains_nan, min_value, max_value) in float_stats_states() {
            // NOT (min = C AND max = C), three-valued.
            let body = not3(and3(
                min_value.map(|min| min == constant),
                max_value.map(|max| max == constant),
            ));
            // Both forms carry the same fail-open disjuncts: official references
            // min_value, max_value and contains_nan, and so does this crate.
            let null_checks = or3(
                or3(Some(min_value.is_none()), Some(max_value.is_none())),
                Some(contains_nan.is_none()),
            );
            let official = or3(null_checks, or3(body, contains_nan));
            let ours = or3(null_checks, or3(gate(contains_nan), body));
            assert_eq!(
                official == Some(true),
                ours == Some(true),
                "diverged for contains_nan={contains_nan:?} min={min_value:?} max={max_value:?}"
            );
            // As for `>`: no well-formed state leaves the condition unknown, so
            // the `IS NOT FALSE` wrapper does not disturb the equivalence.
            assert_ne!(ours, None);
            assert_ne!(official, None);
        }
    }

    #[test]
    fn float_gate_is_strictly_more_conservative_for_lt_and_lt_eq() {
        // The two operators where official looks at no NaN signal at all but
        // this crate still gates, because `-NaN` sorts BELOW every value under
        // `total_cmp` and so hides beneath a recorded min. The gate must only
        // ever keep files official would prune, and must actually keep one.
        //
        // `=` used to be modelled here too. It no longer is: the gate does not
        // apply to equality any more, and the exact-equivalence claim for `=`
        // and `IN` lives in
        // `ungated_equality_admits_exactly_official_set_and_recovers_pruning`.
        let constant = 5.0_f64;
        let mut strictly_more = false;
        for (contains_nan, min_value, _) in float_stats_states() {
            for strict in [true, false] {
                // `f < 5.0` / `f <= 5.0`: official reads min_value only.
                let body = min_value.map(|min| {
                    if strict {
                        min < constant
                    } else {
                        min <= constant
                    }
                });
                let official = or3(Some(min_value.is_none()), body);
                let ours = or3(
                    or3(Some(min_value.is_none()), Some(contains_nan.is_none())),
                    or3(gate(contains_nan), body),
                );
                assert!(
                    official != Some(true) || ours == Some(true),
                    "the gate must never prune what official keeps: \
                     contains_nan={contains_nan:?} min={min_value:?} strict={strict}"
                );
                if ours == Some(true) && official != Some(true) {
                    strictly_more = true;
                }
            }
        }
        assert!(
            strictly_more,
            "the gate should keep at least one file official prunes"
        );
    }

    #[test]
    fn float_gate_applies_to_exactly_the_bound_reading_operators() {
        // The operator-by-operator rule, structurally. `needs_float_gate` is
        // about which STAT SHAPE a condition lowers to, so this table is the
        // readable form of that mapping.
        let table = Fixture::new(&[("f", DataType::Float64, 3)]);
        for (op, gated) in [
            // `=` admits only non-NaN rows, which the bounds do cover.
            (Operator::Eq, false),
            // `NaN <> C` is true, and the bounds exclude NaN.
            (Operator::NotEq, true),
            // `-NaN` sorts below every value under `total_cmp`.
            (Operator::Lt, true),
            (Operator::LtEq, true),
            // `+NaN` sorts above every value.
            (Operator::Gt, true),
            (Operator::GtEq, true),
        ] {
            let predicate = bin(table.column("f"), op, lit(1.5f64));
            let sql = table.sql(&predicate);
            assert_eq!(
                sql.contains("contains_nan"),
                gated,
                "wrong gate decision for {op:?}: {sql}"
            );
            assert_eq!(
                table.cte_stats(&predicate).contains(&"contains_nan"),
                gated,
                "wrong CTE stats for {op:?}"
            );
        }

        // `IN` is an OR of equality forms: ungated.
        let predicate = in_list(
            table.column("f"),
            vec![lit(1.0f64), lit(2.0f64)],
            &false,
            &table.schema,
        )
        .expect("in list");
        assert!(!table.sql(&predicate).contains("contains_nan"));

        // The null-count shapes read no bound at all, so they never gated.
        let predicate = Arc::new(IsNullExpr::new(table.column("f"))) as Arc<dyn PhysicalExpr>;
        assert!(!table.sql(&predicate).contains("contains_nan"));
        let predicate = Arc::new(IsNotNullExpr::new(table.column("f"))) as Arc<dyn PhysicalExpr>;
        assert!(!table.sql(&predicate).contains("contains_nan"));
    }

    #[test]
    fn float_eq_or_lt_in_one_conjunct_is_still_gated() {
        // The mixed case. `needs_float_gate` recurses through the OR, and the
        // `<` branch needs the gate even though the `=` branch does not — so the
        // whole conjunct is gated. Ungating it because one branch is an equality
        // would lose every file whose `-NaN` sits below the recorded min.
        let table = Fixture::new(&[("f", DataType::Float64, 3)]);
        let f = table.column("f");
        let predicate = bin(
            bin(Arc::clone(&f), Operator::Eq, lit(1.5f64)),
            Operator::Or,
            bin(f, Operator::Lt, lit(3.0f64)),
        );
        assert_eq!(
            table.sql(&predicate),
            "((col_3_stats.data_file_id IS NULL OR \
             ((col_3_stats.value_count IS NULL OR col_3_stats.value_count > 0) AND \
             (col_3_stats.min_value IS NULL OR col_3_stats.max_value IS NULL OR \
             col_3_stats.contains_nan IS NULL OR \
             (col_3_stats.contains_nan IS NULL OR col_3_stats.contains_nan <> false) OR \
             ((1.5 BETWEEN TRY_CAST(col_3_stats.min_value AS DOUBLE) AND \
             TRY_CAST(col_3_stats.max_value AS DOUBLE)) OR \
             (TRY_CAST(col_3_stats.min_value AS DOUBLE) < 3.0)))))) IS NOT FALSE"
        );
        assert_eq!(
            table.cte_stats(&predicate),
            vec!["min_value", "max_value", "value_count", "contains_nan"]
        );

        // And the AND form too, since `needs_float_gate` recurses both ways.
        let f = table.column("f");
        let predicate = bin(
            bin(Arc::clone(&f), Operator::Eq, lit(1.5f64)),
            Operator::And,
            bin(f, Operator::Lt, lit(3.0f64)),
        );
        assert!(table.sql(&predicate).contains("contains_nan <> false"));
    }

    #[test]
    fn ungated_equality_admits_exactly_official_set_and_recovers_pruning() {
        // What the ungating actually buys, proved rather than asserted.
        //
        // Official's `COMPARE_EQUAL` on a DOUBLE column is
        //     min_value IS NULL OR max_value IS NULL OR (C BETWEEN min AND max)
        // with no `contains_nan` anywhere. This crate now renders the same
        // thing. The old gated form was
        //     min_value IS NULL OR max_value IS NULL OR contains_nan IS NULL
        //       OR ((contains_nan IS NULL OR contains_nan <> false)
        //           OR (C BETWEEN min AND max))
        // whose leading `contains_nan IS NULL` disjunct kept EVERY file with an
        // unknown NaN state — every register-by-reference load — against the
        // most selective predicate there is.
        let constant = 5.0_f64;
        // Recovery has two flavours and both matter: a file whose NaN state is
        // UNKNOWN (the register-by-reference case the reviewer flagged) and one
        // that is known to CONTAIN a NaN. The second is sound for the same
        // reason as the first: a NaN row cannot satisfy `f = C`, so the file is
        // prunable on its non-NaN bounds regardless.
        let mut recovered_unknown_nan = 0;
        let mut recovered_known_nan = 0;
        for contains_nan in [None, Some(true), Some(false)] {
            for (min_value, max_value) in [
                (None, None),
                (None, Some(9.0_f64)),
                (Some(1.0_f64), None),
                (Some(1.0_f64), Some(9.0_f64)), // brackets the constant
                (Some(1.0_f64), Some(3.0_f64)), // entirely below it
                (Some(7.0_f64), Some(9.0_f64)), // entirely above it
                (Some(5.0_f64), Some(5.0_f64)), // exactly it
            ] {
                let between = and3(
                    min_value.map(|min| min <= constant),
                    max_value.map(|max| max >= constant),
                );
                let null_checks = or3(Some(min_value.is_none()), Some(max_value.is_none()));

                let official = or3(null_checks, between);
                let ours = or3(null_checks, between);
                let old_gated = or3(
                    or3(null_checks, Some(contains_nan.is_none())),
                    or3(gate(contains_nan), between),
                );

                // Never unknown, so the `IS NOT FALSE` wrapper is inert and
                // "admits" means the same thing on both sides.
                assert_ne!(ours, None);
                assert_eq!(
                    official == Some(true),
                    ours != Some(false),
                    "diverged from official for contains_nan={contains_nan:?} \
                     min={min_value:?} max={max_value:?}"
                );

                // The gate could only ever have kept more, never fewer.
                assert!(
                    ours != Some(true) || old_gated == Some(true),
                    "ungating must not admit anything the gate rejected"
                );

                if old_gated == Some(true) && ours == Some(false) {
                    // The recovered pruning: the NaN signal no longer rescues a
                    // file whose bounds exclude the constant.
                    match contains_nan {
                        None => recovered_unknown_nan += 1,
                        Some(true) => recovered_known_nan += 1,
                        Some(false) => {
                            panic!("contains_nan = false never gated, so nothing to recover")
                        },
                    }
                }
            }
        }
        // Two bound states exclude the constant — entirely below and entirely
        // above — and each is recovered for both NaN states that used to gate.
        assert_eq!(
            recovered_unknown_nan, 2,
            "a file with contains_nan NULL whose bounds exclude the constant \
             must now be pruned"
        );
        assert_eq!(
            recovered_known_nan, 2,
            "a file known to contain a NaN is prunable too: a NaN row cannot \
             satisfy an equality"
        );

        // `IN` is the OR of two such forms and inherits the property.
        for contains_nan in [None, Some(true), Some(false)] {
            for (min_value, max_value) in
                [(Some(1.0_f64), Some(3.0_f64)), (Some(1.0_f64), Some(9.0_f64))]
            {
                let null_checks = or3(Some(min_value.is_none()), Some(max_value.is_none()));
                let between =
                    |c: f64| and3(min_value.map(|min| min <= c), max_value.map(|max| max >= c));
                let body = or3(between(5.0), between(8.0));
                let official = or3(null_checks, body);
                let ours = or3(null_checks, body);
                let old_gated = or3(
                    or3(null_checks, Some(contains_nan.is_none())),
                    or3(gate(contains_nan), body),
                );
                assert_eq!(official == Some(true), ours != Some(false));
                assert!(ours != Some(true) || old_gated == Some(true));
            }
        }
        // The concrete recovered case for IN: bounds [1, 3] hold neither 5 nor
        // 8, and an unknown NaN state used to keep the file anyway.
        let null_checks = or3(Some(false), Some(false));
        let body = or3(
            and3(Some(1.0 <= 5.0), Some(3.0 >= 5.0)),
            and3(Some(1.0 <= 8.0), Some(3.0 >= 8.0)),
        );
        assert_eq!(or3(null_checks, body), Some(false), "IN now prunes it");
        assert_eq!(
            or3(or3(null_checks, Some(true)), or3(gate(None), body)),
            Some(true),
            "the gated form used to keep it"
        );
    }

    #[test]
    fn bare_boolean_column_predicate_pushes_down_nothing() {
        // `WHERE flag` is a BOUND_COLUMN_REF, which falls through official's
        // outer switch to `default:`. DataFusion hands this shape over as-is, so
        // it is worth pinning.
        let table = Fixture::new(&[("flag", DataType::Boolean, 1)]);
        let predicate = table.column("flag");
        assert!(table.lowers_to_nothing(&predicate));
    }

    #[test]
    fn each_float_conjunct_carries_its_own_gate() {
        // The gate is applied per conjunct, so intersecting two conjuncts on one
        // float column yields `(gate OR A) AND (gate OR B)` rather than
        // `gate OR (A AND B)`. Those are equivalent; pinned because the extra
        // repetition is easy to mistake for a bug.
        let table = Fixture::new(&[("f", DataType::Float64, 3)]);
        let f = table.column("f");
        let predicate = bin(
            bin(Arc::clone(&f), Operator::Gt, lit(5.0f64)),
            Operator::And,
            bin(f, Operator::Lt, lit(10.0f64)),
        );
        assert_eq!(
            table.sql(&predicate),
            "((col_3_stats.data_file_id IS NULL OR \
             ((col_3_stats.value_count IS NULL OR col_3_stats.value_count > 0) AND \
             (col_3_stats.min_value IS NULL OR col_3_stats.max_value IS NULL OR \
             col_3_stats.contains_nan IS NULL OR \
             ((col_3_stats.contains_nan IS NULL OR col_3_stats.contains_nan <> false) OR \
             (TRY_CAST(col_3_stats.max_value AS DOUBLE) > 5.0)) AND \
             ((col_3_stats.contains_nan IS NULL OR col_3_stats.contains_nan <> false) OR \
             (TRY_CAST(col_3_stats.min_value AS DOUBLE) < 10.0)))))) IS NOT FALSE"
        );
        assert_eq!(
            table.cte_stats(&predicate),
            vec!["min_value", "max_value", "value_count", "contains_nan"]
        );
    }

    // ---------------------------------------------------------------------
    // keep_when_unknown — pruning only on a definite `false`
    // ---------------------------------------------------------------------

    #[test]
    fn keep_when_unknown_wraps_outside_the_no_stats_row_guard() {
        // Order matters. Inside the `data_file_id IS NULL OR ...` guard the
        // wrapper would only cover part of the expression and an unknown from
        // the guard itself would still prune; outside, nothing the column
        // contributes can evaluate to NULL under `WHERE ... AND`.
        let table = ints();
        let predicate = bin(table.column("a"), Operator::Gt, lit(5i32));
        let sql = table.sql(&predicate);
        assert_eq!(
            sql,
            "((col_1_stats.data_file_id IS NULL OR \
             ((col_1_stats.value_count IS NULL OR col_1_stats.value_count > 0) AND \
             (col_1_stats.max_value IS NULL OR \
             TRY_CAST(col_1_stats.max_value AS INTEGER) > 5)))) IS NOT FALSE"
        );
        // The G1 guard is nested within the wrapper, not the other way round.
        assert!(
            sql.starts_with("((col_1_stats.data_file_id IS NULL OR "),
            "{sql}"
        );
        assert!(sql.ends_with(") IS NOT FALSE"), "{sql}");
        assert!(
            !sql.contains("IS NOT FALSE)"),
            "the wrapper must be outermost: {sql}"
        );
        // Exactly one wrapper per column, not one per stat or per comparison.
        assert_eq!(sql.matches("IS NOT FALSE").count(), 1, "{sql}");
    }

    #[test]
    fn keep_when_unknown_wraps_every_column_exactly_once() {
        let table = ints();
        let predicate = bin(
            bin(table.column("a"), Operator::Gt, lit(5i32)),
            Operator::And,
            bin(table.column("b"), Operator::Lt, lit(3i32)),
        );
        let rendered = table.render(&predicate);
        assert_eq!(rendered.len(), 2);
        for filter in &rendered {
            assert_eq!(
                filter.condition.matches("IS NOT FALSE").count(),
                1,
                "{}",
                filter.condition
            );
            assert!(filter.condition.ends_with(") IS NOT FALSE"));
        }
    }

    #[test]
    fn keep_when_unknown_is_honoured_when_a_dialect_overrides_it() {
        // Overriding is for spelling, not policy: a dialect without
        // `IS NOT FALSE` can say the same thing another way, and rendering must
        // use what the dialect returned rather than the default.
        struct Coalescing;
        impl StatsSqlDialect for Coalescing {
            fn try_cast(
                &self,
                expr: &str,
                literal: &StatsLiteral,
                data_type: &DataType,
            ) -> Option<String> {
                Duck.try_cast(expr, literal, data_type)
            }
            fn collate_binary(&self, expr: &str) -> String {
                Duck.collate_binary(expr)
            }
            fn boolean_is_not_false(&self, expr: &str) -> String {
                Duck.boolean_is_not_false(expr)
            }
            fn keep_when_unknown(&self, condition: &str) -> String {
                format!("COALESCE({condition}, TRUE)")
            }
        }

        let table = ints();
        let predicate = bin(table.column("a"), Operator::Gt, lit(5i32));
        let rendered = table
            .render_with(&predicate, &Coalescing)
            .expect("rendered");
        assert_eq!(rendered.len(), 1);
        assert_eq!(
            rendered[0].condition,
            "COALESCE((col_1_stats.data_file_id IS NULL OR \
             ((col_1_stats.value_count IS NULL OR col_1_stats.value_count > 0) AND \
             (col_1_stats.max_value IS NULL OR \
             TRY_CAST(col_1_stats.max_value AS INTEGER) > 5))), TRUE)"
        );
        assert!(!rendered[0].condition.contains("IS NOT FALSE"));
    }

    /// A stored stat, as SQL sees it: absent (`None`), present but not castable
    /// to the comparison type (`Some(None)`), or present and well formed.
    type StoredBound = Option<Option<f64>>;

    #[test]
    fn keep_when_unknown_rescues_a_malformed_stat() {
        // The bug the wrapper fixes. The per-stat `IS NULL` disjuncts test the
        // STORED column, but a stat that is present and malformed is not NULL
        // while `TRY_CAST` of it is — so the whole condition is NULL, and
        // `WHERE ... AND NULL` drops the file. Official carries the identical
        // shape and is safe only because DuckDB wrote every stat it reads.
        //
        // Modelled for `a > 5` on an INTEGER column:
        //     max_value IS NULL OR TRY_CAST(max_value AS INTEGER) > 5
        let constant = 5.0_f64;
        let mut rescued = 0;
        for stored in [
            None,                // SQL NULL       -> the IS NULL disjunct fires
            Some(None),          // 'not-a-number' -> TRY_CAST yields NULL
            Some(Some(1.0_f64)), // below the constant
            Some(Some(9.0_f64)), // above it
        ] as [StoredBound; 4]
        {
            let is_null = Some(stored.is_none());
            let compared = stored.flatten().map(|value| value > constant);
            let inner = or3(is_null, compared);

            // Official prunes on unknown, because the clause sits under
            // `WHERE ... AND`.
            let official_admits = inner == Some(true);
            // This crate keeps it: `IS NOT FALSE` prunes only a definite false.
            let ours_admits = inner != Some(false);

            assert!(
                !official_admits || ours_admits,
                "the wrapper must never prune what official keeps: {stored:?}"
            );
            if ours_admits && !official_admits {
                assert_eq!(stored, Some(None), "only a malformed stat should differ");
                rescued += 1;
            }
        }
        assert_eq!(
            rescued, 1,
            "the malformed stat must be the one file rescued"
        );

        // And a well-formed stat below the constant is still a DEFINITE false,
        // which `IS NOT FALSE` prunes. The wrapper is not a blanket
        // "keep everything".
        let stored: StoredBound = Some(Some(1.0_f64));
        let inner = or3(
            Some(stored.is_none()),
            stored.flatten().map(|value| value > constant),
        );
        assert_eq!(inner, Some(false), "a well-formed miss must still prune");
    }

    #[test]
    fn try_cast_receives_the_literal_the_stat_is_compared_against() {
        // `try_cast` is handed the constant as well as the stat, so a dialect
        // that reproduces a comparison in the text domain can check the
        // constant's encoding, and one whose engine converts the constant can
        // decline one it would refuse. Both halves are exercised: the literal
        // that arrives is this comparison's own constant, and declining on it
        // drops that comparison alone.
        struct LiteralAware;
        impl StatsSqlDialect for LiteralAware {
            fn try_cast(
                &self,
                expr: &str,
                literal: &StatsLiteral,
                data_type: &DataType,
            ) -> Option<String> {
                // Stand-in for "this dialect cannot convert that constant".
                if literal.text().starts_with('-') {
                    return None;
                }
                Some(format!(
                    "TRY_CAST({expr} AS {}) /* {} */",
                    duckdb_type_name(data_type)?,
                    literal.text()
                ))
            }

            fn collate_binary(&self, expr: &str) -> String {
                Duck.collate_binary(expr)
            }

            fn boolean_is_not_false(&self, expr: &str) -> String {
                Duck.boolean_is_not_false(expr)
            }
        }

        let table = ints();

        // The literal handed over is this comparison's own constant.
        let predicate = bin(table.column("a"), Operator::Gt, lit(5i32));
        let rendered = table
            .render_with(&predicate, &LiteralAware)
            .expect("rendered");
        assert_eq!(
            rendered[0].condition,
            "((col_1_stats.data_file_id IS NULL OR \
             ((col_1_stats.value_count IS NULL OR col_1_stats.value_count > 0) AND \
             (col_1_stats.max_value IS NULL OR \
             TRY_CAST(col_1_stats.max_value AS INTEGER) /* 5 */ > 5)))) IS NOT FALSE"
        );

        // Both bounds of an equality see the same literal.
        let predicate = bin(table.column("a"), Operator::Eq, lit(7i32));
        let rendered = table
            .render_with(&predicate, &LiteralAware)
            .expect("rendered");
        assert_eq!(rendered[0].condition.matches("/* 7 */").count(), 2);

        // Declining on the literal drops only that comparison: the AND keeps
        // the other half, so the decision really is per-comparison and not
        // per-column.
        let a = table.column("a");
        let predicate = bin(
            bin(Arc::clone(&a), Operator::Gt, lit(5i32)),
            Operator::And,
            bin(Arc::clone(&a), Operator::Lt, lit(-3i32)),
        );
        let rendered = table
            .render_with(&predicate, &LiteralAware)
            .expect("the positive constant still renders");
        assert_eq!(
            rendered[0].condition,
            "((col_1_stats.data_file_id IS NULL OR \
             ((col_1_stats.value_count IS NULL OR col_1_stats.value_count > 0) AND \
             (col_1_stats.min_value IS NULL OR col_1_stats.max_value IS NULL OR \
             (TRY_CAST(col_1_stats.max_value AS INTEGER) /* 5 */ > 5))))) IS NOT FALSE"
        );

        // And a lone declined comparison renders nothing at all.
        let predicate = bin(a, Operator::Lt, lit(-3i32));
        assert!(table.render_with(&predicate, &LiteralAware).is_none());
    }

    // ---------------------------------------------------------------------
    // quote_literal — the dialect owns string escaping
    // ---------------------------------------------------------------------

    #[test]
    fn quote_literal_default_doubles_embedded_quotes() {
        // Standard SQL, and what official emits
        // (`DuckLakeUtil::SQLLiteralToString`). `stats_encode` stores `Utf8`
        // verbatim, so a constant carrying quotes reaches the SQL text and the
        // doubling is the only thing keeping the statement well formed.
        let table = Fixture::new(&[("s", DataType::Utf8, 1)]);
        let predicate = bin(table.column("s"), Operator::Eq, lit("it's 'quoted'"));
        assert_eq!(
            table.sql(&predicate),
            "((col_1_stats.data_file_id IS NULL OR \
             ((col_1_stats.value_count IS NULL OR col_1_stats.value_count > 0) AND \
             (col_1_stats.min_value IS NULL OR col_1_stats.max_value IS NULL OR \
             'it''s ''quoted''' BETWEEN col_1_stats.min_value AND \
             col_1_stats.max_value)))) IS NOT FALSE"
        );
    }

    #[test]
    fn quote_literal_override_is_honoured() {
        // MySQL gives backslash a meaning inside a quoted string unless
        // `NO_BACKSLASH_ESCAPES` is set, so a constant containing one would
        // reach the server mangled and compare against the wrong bound. The
        // dialect, not this module, decides how to spell the literal.
        struct BackslashEscaping;
        impl StatsSqlDialect for BackslashEscaping {
            fn try_cast(
                &self,
                expr: &str,
                literal: &StatsLiteral,
                data_type: &DataType,
            ) -> Option<String> {
                Duck.try_cast(expr, literal, data_type)
            }

            fn collate_binary(&self, expr: &str) -> String {
                Duck.collate_binary(expr)
            }

            fn boolean_is_not_false(&self, expr: &str) -> String {
                Duck.boolean_is_not_false(expr)
            }

            fn quote_literal(&self, text: &str) -> String {
                format!("'{}'", text.replace('\\', "\\\\").replace('\'', "''"))
            }
        }

        let table = Fixture::new(&[("s", DataType::Utf8, 1)]);
        let predicate = bin(table.column("s"), Operator::Eq, lit("a\\b'c"));

        // Default: quotes doubled, backslash passed through untouched.
        assert_eq!(
            table.sql(&predicate),
            "((col_1_stats.data_file_id IS NULL OR \
             ((col_1_stats.value_count IS NULL OR col_1_stats.value_count > 0) AND \
             (col_1_stats.min_value IS NULL OR col_1_stats.max_value IS NULL OR \
             'a\\b''c' BETWEEN col_1_stats.min_value AND \
             col_1_stats.max_value)))) IS NOT FALSE"
        );

        // Override: backslash doubled as well.
        let rendered = table
            .render_with(&predicate, &BackslashEscaping)
            .expect("rendered");
        assert_eq!(
            rendered[0].condition,
            "((col_1_stats.data_file_id IS NULL OR \
             ((col_1_stats.value_count IS NULL OR col_1_stats.value_count > 0) AND \
             (col_1_stats.min_value IS NULL OR col_1_stats.max_value IS NULL OR \
             'a\\\\b''c' BETWEEN col_1_stats.min_value AND \
             col_1_stats.max_value)))) IS NOT FALSE"
        );
    }

    #[test]
    fn quote_literal_is_not_used_for_unquoted_numerics() {
        // Only the quoted path goes through the dialect. A finite numeric is
        // emitted bare, exactly as official's `CastValueToTarget` does, so an
        // override must not be able to disturb it.
        struct Marking;
        impl StatsSqlDialect for Marking {
            fn try_cast(
                &self,
                expr: &str,
                literal: &StatsLiteral,
                data_type: &DataType,
            ) -> Option<String> {
                Duck.try_cast(expr, literal, data_type)
            }

            fn collate_binary(&self, expr: &str) -> String {
                Duck.collate_binary(expr)
            }

            fn boolean_is_not_false(&self, expr: &str) -> String {
                Duck.boolean_is_not_false(expr)
            }

            fn quote_literal(&self, text: &str) -> String {
                format!("Q<{text}>")
            }
        }

        // A string constant is quoted, so it is marked.
        let strings = Fixture::new(&[("s", DataType::Utf8, 1)]);
        let predicate = bin(strings.column("s"), Operator::Eq, lit("x"));
        let rendered = strings.render_with(&predicate, &Marking).expect("rendered");
        assert!(
            rendered[0].condition.contains("Q<x> BETWEEN"),
            "{}",
            rendered[0].condition
        );

        // A finite numeric is not.
        let ints = ints();
        let predicate = bin(ints.column("a"), Operator::Gt, lit(5i32));
        let rendered = ints.render_with(&predicate, &Marking).expect("rendered");
        assert_eq!(
            rendered[0].condition,
            "((col_1_stats.data_file_id IS NULL OR \
             ((col_1_stats.value_count IS NULL OR col_1_stats.value_count > 0) AND \
             (col_1_stats.max_value IS NULL OR \
             TRY_CAST(col_1_stats.max_value AS INTEGER) > 5)))) IS NOT FALSE"
        );

        // But a BOOLEAN constant is quoted — it requires a value comparison yet
        // is not numeric — so it does go through the dialect.
        let flags = Fixture::new(&[("flag", DataType::Boolean, 1)]);
        let predicate = bin(flags.column("flag"), Operator::Eq, lit(true));
        let rendered = flags.render_with(&predicate, &Marking).expect("rendered");
        assert!(
            rendered[0].condition.contains("Q<true> BETWEEN"),
            "{}",
            rendered[0].condition
        );
    }
}
