//! PostgreSQL metadata provider for DuckLake catalogs.

use crate::Result;
use crate::inlined_filter::{
    InlinedDataScan, InlinedFilter, InlinedSqlBind, InlinedSqlDialect, render_inlined_filter,
};
use crate::metadata_provider::{
    ColumnWithTable, DataFileChange, DeleteFileChange, DuckLakeFileColumnStatistics,
    DuckLakeFileData, DuckLakeFileMetadata, DuckLakeInlinedData, DuckLakeInlinedDelete,
    DuckLakeNameMapping, DuckLakeNameMappingEntry, DuckLakeStatistics, DuckLakeTableColumn,
    DuckLakeTableColumnStatistics, DuckLakeTableField, DuckLakeTableFile, DuckLakeTableStatistics,
    FileWithTable, InlinedDataBackend, MetadataProvider, MetadataSetting, SchemaMetadata,
    SnapshotChangeMetadata, SnapshotMetadata, TableMetadata, TableWithSchema, ViewMetadata,
    ViewWithSchema, block_on, inlined_delete_table_name, inlined_text_projection,
    is_inlined_data_table, parse_inlined_rows_with_present, reconstruct_columns,
    reconstruct_columns_with_table, resolve_metadata_settings,
};
use crate::partition::PartitionSpec;
use crate::sort::SortSpec;
use crate::stats_encode::{is_canonical_date, is_canonical_timestamp, is_canonical_timestamptz};
use crate::stats_filter::{RenderedColumnFilter, StatsFilter, StatsLiteral, StatsSqlDialect};
use arrow::datatypes::{DataType, SchemaRef};
use arrow::record_batch::RecordBatch;
use sqlx::AssertSqlSafe;
use sqlx::Row;
use sqlx::postgres::{PgPool, PgPoolOptions, PgRow};
use sqlx::types::chrono::NaiveDateTime;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, OnceLock};

fn is_missing_statistics_table(error: &sqlx::Error) -> bool {
    let message = error.to_string().to_ascii_lowercase();
    message.contains("does not exist") || message.contains("undefined table")
}

fn quote_ident(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

fn decode_view(row: &PgRow) -> Result<ViewMetadata> {
    Ok(ViewMetadata {
        view_id: row.try_get(0)?,
        schema_id: row.try_get(1)?,
        begin_snapshot: row.try_get(2)?,
        view_name: row.try_get(3)?,
        dialect: row.try_get(4)?,
        sql: row.try_get(5)?,
        column_aliases: row.try_get(6)?,
    })
}

fn decode_table_file(row: &PgRow, snapshot_id: i64) -> Result<DuckLakeTableFile> {
    let delete_file_id: Option<i64> = row.try_get(8)?;
    let (delete_file, delete_count) = if delete_file_id.is_some() {
        (
            Some(DuckLakeFileData {
                path: row.try_get(9)?,
                path_is_relative: row.try_get(10)?,
                file_size_bytes: row.try_get(11)?,
                footer_size: row.try_get(12)?,
                encryption_key: row.try_get(13)?,
                mapping_id: None,
            }),
            row.try_get(14)?,
        )
    } else {
        (None, None)
    };
    Ok(DuckLakeTableFile {
        data_file_id: row.try_get(0)?,
        file: DuckLakeFileData {
            path: row.try_get(1)?,
            path_is_relative: row.try_get(2)?,
            file_size_bytes: row.try_get(3)?,
            footer_size: row.try_get(4)?,
            encryption_key: row.try_get(5)?,
            mapping_id: row.try_get(19).unwrap_or(None),
        },
        delete_file_id,
        delete_file,
        row_id_start: row.try_get(6)?,
        snapshot_id: Some(snapshot_id),
        begin_snapshot: row.try_get(15)?,
        schema_version: row.try_get(17)?,
        partial_max: row.try_get(16)?,
        max_row_count: row.try_get(7)?,
        delete_count,
        // Column 18 is present on the select-path query (which projects
        // `data.partition_id`) and absent on callers that share this decoder
        // without it; `try_get` failing there degrades to `None`, matching the
        // pre-partition behaviour. Per-key values are filled in by the caller.
        partition_id: row.try_get(18).unwrap_or(None),
        partition_values: Vec::new(),
    })
}

fn decode_name_mapping_rows(
    requested_mapping_id: i64,
    rows: &[PgRow],
) -> Result<DuckLakeNameMapping> {
    let first = rows.first().ok_or_else(|| {
        crate::DuckLakeError::InvalidConfig(format!(
            "DuckLake name mapping {requested_mapping_id} does not exist"
        ))
    })?;
    let mut entries = Vec::new();
    for row in rows {
        if let Some(column_id) = row.try_get::<Option<i64>, _>(3)? {
            entries.push(DuckLakeNameMappingEntry {
                column_id,
                source_name: row.try_get(4)?,
                target_field_id: row.try_get(5)?,
                parent_column: row.try_get(6)?,
                is_partition: row.try_get::<Option<bool>, _>(7)?.unwrap_or(false),
            });
        }
    }
    Ok(DuckLakeNameMapping {
        mapping_id: first.try_get(0)?,
        table_id: first.try_get(1)?,
        mapping_type: first.try_get(2)?,
        entries,
    })
}

macro_rules! bind_repeat {
    ($query:expr, $value:expr, 1) => {
        $query.bind($value)
    };
    ($query:expr, $value:expr, 2) => {
        $query.bind($value).bind($value)
    };
    ($query:expr, $value:expr, 3) => {
        $query.bind($value).bind($value).bind($value)
    };
    ($query:expr, $value:expr, 4) => {
        $query.bind($value).bind($value).bind($value).bind($value)
    };
    ($query:expr, $value:expr, 6) => {
        $query
            .bind($value)
            .bind($value)
            .bind($value)
            .bind($value)
            .bind($value)
            .bind($value)
    };
    ($query:expr, $value:expr, 8) => {
        $query
            .bind($value)
            .bind($value)
            .bind($value)
            .bind($value)
            .bind($value)
            .bind($value)
            .bind($value)
            .bind($value)
    };
}

/// Optional catalog-schema capabilities probed before scan / CDC / inlined-data
/// queries.
///
/// Minimal / pre-v1.0 catalogs may lack the `partial_max` columns, the
/// `ducklake_schema_versions` ledger, and the inlined-data registry. Queries
/// degrade the corresponding projections to NULL or skip inlined-data reads
/// when a capability is absent.
#[derive(Debug, Clone, Copy)]
struct SchemaCapabilities {
    /// `ducklake_data_file.partial_max` exists.
    data_file_partial_max: bool,
    /// `ducklake_delete_file.partial_max` exists.
    delete_file_partial_max: bool,
    /// The `ducklake_schema_versions` table exists.
    schema_versions: bool,
    /// `ducklake_data_file.partition_id` exists.
    data_file_partition_id: bool,
    /// The `ducklake_inlined_data_tables` registry exists.
    inlined_data_tables: bool,
    /// The `ducklake_view` table exists.
    views: bool,
    /// The server has `pg_input_is_valid` (PostgreSQL 16+), which
    /// [`PostgresStatsDialect`] needs for its exact `TRY_CAST` stand-in.
    ///
    /// A server capability rather than a catalog one, so it is deliberately not
    /// part of [`Self::all`]: gating the memo on it would make an otherwise
    /// fully-migrated catalog on an older server re-probe on every call, and a
    /// stale `false` only costs pruning.
    soft_input_validation: bool,
}

impl SchemaCapabilities {
    fn all(&self) -> bool {
        self.data_file_partial_max
            && self.delete_file_partial_max
            && self.schema_versions
            && self.data_file_partition_id
            && self.inlined_data_tables
            && self.views
    }
}

/// Run one page of the data-file listing.
///
/// Returns the raw `sqlx::Error` so the caller can retry unfiltered when the
/// narrowed form of the query is what the catalog could not run. The bind order
/// is the listing query's, filtered or not: statistics literals are inlined by
/// [`crate::stats_filter`], so narrowing the query adds no parameter.
pub(crate) async fn fetch_data_file_page(
    pool: &PgPool,
    sql: &str,
    table_id: i64,
    snapshot_id: i64,
    after_data_file_id: i64,
    limit: i64,
) -> std::result::Result<Vec<PgRow>, sqlx::Error> {
    sqlx::query(AssertSqlSafe(sql))
        .bind(table_id)
        .bind(snapshot_id)
        .bind(snapshot_id)
        .bind(table_id)
        .bind(snapshot_id)
        .bind(snapshot_id)
        .bind(after_data_file_id)
        .bind(limit)
        .fetch_all(pool)
        .await
}

/// Statistics SQL for a catalog queried natively as PostgreSQL.
///
/// Shared with [`crate::multicatalog_provider`], which speaks the same dialect
/// to the same server; a divergence between two copies of the constructs below
/// would be a silent correctness difference between the two readers.
///
/// PostgreSQL has no `TRY_CAST`, and a plain `CAST` of a stat string it cannot
/// parse raises an error that aborts the whole listing query — one malformed or
/// foreign-written `min_value` would turn a scan into a hard failure. The two
/// constructs below are what stand in for `TRY_CAST`, and which one is used
/// depends on the server:
///
/// - PostgreSQL 16 and later have `pg_input_is_valid(text, type)`, which runs a
///   type's input function in soft-error mode. It is exact: it rejects garbage,
///   out-of-range integers, overflowing and underflowing floats, and impossible
///   calendar dates like `2020-02-31`, all of which a plain `CAST` would raise
///   on.
/// - Older servers get a regular-expression validity test instead. A regex
///   cannot decide a calendar date, so temporal types are declined outright
///   there rather than risking an abort; see [`postgres_castable_pattern`].
///
/// Both are wrapped in a `CASE`, whose arms PostgreSQL evaluates lazily, so the
/// `CAST` never runs on a value the test rejected.
pub(crate) struct PostgresStatsDialect {
    /// `pg_input_is_valid` exists (PostgreSQL 16+).
    pub(crate) soft_input_validation: bool,
}

impl StatsSqlDialect for PostgresStatsDialect {
    /// Both sides are inspected, for different reasons.
    ///
    /// The validity test built below covers only the *stat*, which is a value
    /// read from a row. The constant is a different matter: it is spliced into
    /// the statement as a bare literal on the other side of the comparison, and
    /// PostgreSQL coerces that unknown-type literal to the cast target while
    /// *parsing*, before a single row is read. A constant the target's input
    /// function refuses is therefore a syntax-time error that aborts the entire
    /// listing query — it fires on `EXPLAIN` against an empty table. Only a
    /// temporal constant can do that, and a temporal comparison never reaches a
    /// cast: it is compared as text, gated on both sides by
    /// [`postgres_canonical_temporal_pattern`].
    ///
    /// Declining drops that one comparison and costs pruning. Emitting a
    /// constant PostgreSQL cannot read costs the scan.
    fn try_cast(&self, expr: &str, literal: &StatsLiteral, data_type: &DataType) -> Option<String> {
        if let Some(pattern) = postgres_canonical_temporal_pattern(data_type, literal.text()) {
            // Compared as text, not cast. PostgreSQL's temporal types hold
            // microseconds and *round* a longer fraction, which is monotonic but
            // not injective — two distinct nanosecond instants can land on one
            // microsecond, making a strict comparison that holds of the stored
            // values come back false. Casting also needs the input function,
            // which decides a calendar no pattern can, and which is only
            // interrogable through `pg_input_is_valid` on PostgreSQL 16 and
            // later.
            //
            // Text sidesteps all of it. The canonical encoding is
            // chronologically ordered byte-wise, so comparing the two strings
            // answers the same question a cast would, exactly, at full
            // precision, on every server version — and nothing is parsed, so an
            // impossible date cannot raise.
            let text = self.collate_binary(expr);
            return Some(format!("CASE WHEN {text} ~ '{pattern}' THEN {text} END"));
        }
        // Past the temporal branch every constant is a bare number or quoted
        // `true` / `false`, and no encoding of those aborts the statement while
        // PostgreSQL parses it — so from here only the stat needs vetting.
        let _ = literal;
        let target = postgres_cast_type(data_type)?;
        // The shape test is required on every server version. `pg_input_is_valid`
        // is not a substitute for it: it answers "can the input function read
        // this", and that function is deliberately permissive. It accepts
        // `today`, `now`, `epoch` and `infinity` for a date, `NaN` and
        // `Infinity` for a numeric, `nan` for a float, and `0x10` and `+5` for
        // an integer — none of which this crate or official DuckLake writes, and
        // each of which casts to a *value* that then prunes files. A stat of
        // `today` on a date column would make pruning depend on the wall clock.
        //
        // `COLLATE "C"` so the pattern's character ranges are ASCII and nothing
        // else, whatever the database's collation. Without it `[0-9]` is a
        // locale-dependent range, and a locale that widened it would feed the
        // `CAST` a digit its input function rejects.
        let pattern = postgres_castable_pattern(target)?;
        let shape = format!("({expr} COLLATE \"C\") ~ '{pattern}'");
        let guard = if self.soft_input_validation {
            // Shape proves the text is one this crate could have written;
            // `pg_input_is_valid` proves the value is in range. Neither is
            // redundant: the shape rejects `NaN` and `0x10`, which the input
            // function reads, and the input function rejects an overflowing
            // 300-digit numeric, which the shape admits.
            format!("{shape} AND pg_input_is_valid({expr}, '{target}')")
        } else {
            shape
        };
        Some(format!(
            "CASE WHEN {guard} THEN CAST({expr} AS {target}) END"
        ))
    }

    /// PostgreSQL compares `text` in the database's collation, which for any
    /// ICU or libc locale is neither byte-wise nor even a fixed order across
    /// servers. DataFusion compares `Utf8` byte-wise, so a raw stat comparison
    /// is forced into `C` — the one collation defined to be byte-wise.
    fn collate_binary(&self, expr: &str) -> String {
        format!("({expr} COLLATE \"C\")")
    }

    fn boolean_is_not_false(&self, expr: &str) -> String {
        format!("{expr} IS NULL OR {expr} <> false")
    }
}

/// The PostgreSQL type a statistic is cast to for comparison against a constant
/// of `data_type`, or `None` when there is none that orders the constant's
/// values the way this engine does.
///
/// Arrow's `Display` is not SQL, so the mapping is written out. Declining a type
/// drops that one comparison, which costs pruning and never rows.
///
/// Every integer width and `DECIMAL` maps to `numeric` rather than to
/// `smallint` / `integer` / `bigint`. `numeric` holds all of them exactly, it
/// compares exactly against the integer or fixed-point literal on the other
/// side, and it has no range a stat written by another engine could overflow —
/// a narrower target would decline such a stat and lose the pruning, and the
/// stats table is small enough that the arithmetic is not what this query
/// costs. Floats map to `double precision` for the same reason: both sides are
/// the shortest round-trip *text* of the value, so parsing them at higher
/// precision than the column preserves both equality and order.
///
/// Absent types are ones [`crate::stats_encode::encode_scalar`] has no
/// canonical text for (`TIME`, `INTERVAL`, `Decimal256`, `UUID`, blobs, nested
/// types), so no literal of that type ever reaches here.
pub(crate) fn postgres_cast_type(data_type: &DataType) -> Option<&'static str> {
    Some(match data_type {
        DataType::Boolean => "boolean",
        DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64
        | DataType::Decimal128(_, _) => "numeric",
        DataType::Float16 | DataType::Float32 | DataType::Float64 => "double precision",
        _ => return None,
    })
}

/// A pattern that admits only text `CAST(... AS <target>)` is guaranteed to
/// accept, for servers without `pg_input_is_valid`. `None` declines the type.
///
/// Each pattern is deliberately narrower than the type's input function. It
/// admits exactly the shapes [`crate::stats_encode`] writes — and DuckDB with
/// it — and rejects everything else, including forms PostgreSQL would in fact
/// accept (`+5`, `nan`, hexadecimal `numeric`, leading whitespace). A rejected
/// stat yields SQL `NULL` and the file is kept, so being strict costs pruning
/// and nothing else, whereas being one case too loose aborts the query.
///
/// The digit-count bounds are what make that guarantee. 255 is also the largest
/// repetition count PostgreSQL's regex engine accepts:
///
/// - `numeric`: at most 255 digits either side of the point, far short of the
///   131072 at which `numeric` input overflows.
/// - `double precision`: fixed-point is bounded to ±1e255 and, away from zero,
///   1e-255, well inside the ±1.8e308 range and nowhere near underflow;
///   scientific notation is bounded to a single leading digit and a two-digit
///   exponent, so at most ~1e100. A magnitude beyond that is declined rather
///   than risking the overflow error a plain `CAST` raises. `inf` and `-inf`
///   are admitted because they are how a stored bound spells an infinity, and
///   `float8` parses both on every server; no *constant* reaches here as one,
///   since [`crate::stats_filter`] refuses a non-finite literal outright.
///
/// Temporal types have no pattern. `2020-02-31` matches every plausible date
/// regex and `CAST`ing it raises, and no regular expression decides a calendar,
/// so on these servers a date or timestamp comparison pushes down nothing.
pub(crate) fn postgres_castable_pattern(target: &str) -> Option<&'static str> {
    Some(match target {
        "boolean" => r"^(true|false)$",
        "numeric" => r"^-?[0-9]{1,255}(\.[0-9]{1,255})?$",
        // `inf` is a value this crate writes and PostgreSQL orders correctly.
        // `nan` is not: it parses, and comparing against it prunes a file whose
        // other rows may well match.
        "double precision" => {
            r"^-?(inf|[0-9]{1,255}(\.[0-9]{1,255})?|[0-9](\.[0-9]{1,255})?e[-+][0-9]{1,2})$"
        },
        _ => return None,
    })
}

/// The pattern gating a temporal stat compared as text, or `None` when this is
/// not a temporal type or the constant is not canonically encoded.
///
/// Both sides have to be in the one encoding whose byte order is chronological.
/// The constant is checked here in Rust; the returned pattern checks the stat in
/// SQL. A fraction ending in `0` is refused on both sides: as text `.50` sorts
/// above `.5` while naming the same instant.
fn postgres_canonical_temporal_pattern(
    data_type: &DataType,
    constant: &str,
) -> Option<&'static str> {
    const DATE: &str = r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$";
    const TIMESTAMP: &str =
        r"^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}([.][0-9]*[1-9])?$";
    const TIMESTAMPTZ: &str =
        r"^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}([.][0-9]*[1-9])?[+]00$";

    match data_type {
        DataType::Date32 | DataType::Date64 if is_canonical_date(constant) => Some(DATE),
        DataType::Timestamp(_, None) if is_canonical_timestamp(constant) => Some(TIMESTAMP),
        DataType::Timestamp(_, Some(_)) if is_canonical_timestamptz(constant) => Some(TIMESTAMPTZ),
        _ => None,
    }
}

/// The three fragments that narrow a file listing by catalog statistics: the
/// `WITH` prefix, the joins that bring each column's stats in, and the extra
/// `WHERE` conjuncts. `None` when there is nothing to narrow by.
pub(crate) struct StatsFilterSql {
    pub(crate) with_prefix: String,
    pub(crate) joins: String,
    pub(crate) conditions: String,
}

/// Build the statistics fragments for `filters`, exactly as official DuckLake
/// assembles them: one CTE per column selecting only the stats its condition
/// reads, one `LEFT JOIN` on `data_file_id`, and the conditions ANDed onto the
/// listing's existing `WHERE`.
///
/// Every literal is already inlined by [`crate::stats_filter`], so the listing
/// query's bind placeholders are untouched and its `.bind()` chain does not move.
pub(crate) fn stats_filter_sql(
    table_id: i64,
    filters: &[RenderedColumnFilter],
) -> Option<StatsFilterSql> {
    if filters.is_empty() {
        return None;
    }
    let ctes = filters
        .iter()
        .map(|filter| {
            format!(
                "{alias} AS (
                     SELECT data_file_id, {stats}
                     FROM ducklake_file_column_stats
                     WHERE column_id = {column_id} AND table_id = {table_id}
                 )",
                alias = filter.alias,
                stats = filter.stats.join(", "),
                column_id = filter.column_id,
            )
        })
        .collect::<Vec<_>>()
        .join(",\n                 ");
    let joins = filters
        .iter()
        .map(|filter| {
            format!(
                "\n                 LEFT JOIN {alias} ON {alias}.data_file_id = data.data_file_id",
                alias = filter.alias
            )
        })
        .collect::<String>();
    // Each condition already carries its own fail-open guards — the no-stats-row
    // `data_file_id IS NULL`, the per-stat `IS NULL` disjuncts, and
    // `StatsSqlDialect::keep_when_unknown` for a comparison that lands on NULL
    // because a present stat would not parse. They are spliced verbatim.
    let conditions = filters
        .iter()
        .map(|filter| {
            format!(
                "\n                   AND {condition}",
                condition = filter.condition
            )
        })
        .collect::<String>();
    Some(StatsFilterSql {
        with_prefix: format!("WITH {ctes}\n                 "),
        joins,
        conditions,
    })
}

/// PostgreSQL-based metadata provider for DuckLake catalogs.
#[derive(Debug, Clone)]
pub struct PostgresMetadataProvider {
    pub pool: PgPool,
    // Positive-only memo of the optional-schema capability probes. `Arc` so
    // derived `Clone` shares the cache across provider clones.
    schema_capabilities: Arc<OnceLock<SchemaCapabilities>>,
}

impl PostgresMetadataProvider {
    /// Creates a new provider for an existing DuckLake catalog.
    pub async fn new(connection_string: &str) -> Result<Self> {
        let pool = PgPoolOptions::new()
            .max_connections(5)
            .connect(connection_string)
            .await?;

        Ok(Self {
            pool,
            schema_capabilities: Arc::new(OnceLock::new()),
        })
    }

    /// Creates a provider over an existing connection pool. Replaces
    /// struct-literal construction, which stopped compiling when the
    /// schema-capability memo field was added.
    pub fn from_pool(pool: PgPool) -> Self {
        Self {
            pool,
            schema_capabilities: Arc::new(OnceLock::new()),
        }
    }

    /// Whether the schema-capability memo is populated. Exposed for tests.
    #[doc(hidden)]
    pub fn schema_capabilities_cached(&self) -> bool {
        self.schema_capabilities.get().is_some()
    }

    /// Returns the catalog's optional-schema capabilities, probing at most
    /// once per provider lifetime on a fully-migrated catalog.
    ///
    /// Cache-positive-only: capability existence is monotonic (migrations only
    /// add columns/tables, never drop them), so an all-`true` answer is an
    /// immutable fact and safe to memoize. A `false` answer is never cached —
    /// the next call re-probes, so a mid-flight catalog upgrade is picked up
    /// on the next call exactly like the previous per-call probing. Concurrent
    /// first calls may each probe once (one statement each) — harmless; a
    /// raced `set` is ignored.
    async fn schema_capabilities(&self) -> Result<SchemaCapabilities> {
        if let Some(caps) = self.schema_capabilities.get() {
            return Ok(*caps);
        }
        let row: (bool, bool, bool, bool, bool, bool, bool) = sqlx::query_as(
            "SELECT
               EXISTS (SELECT 1 FROM information_schema.columns
                       WHERE table_name = 'ducklake_data_file' AND column_name = 'partial_max'),
               EXISTS (SELECT 1 FROM information_schema.columns
                       WHERE table_name = 'ducklake_delete_file' AND column_name = 'partial_max'),
               to_regclass('ducklake_schema_versions') IS NOT NULL,
               EXISTS (SELECT 1 FROM information_schema.columns
                       WHERE table_name = 'ducklake_data_file' AND column_name = 'partition_id'),
               to_regclass('ducklake_inlined_data_tables') IS NOT NULL,
               to_regclass('ducklake_view') IS NOT NULL,
               to_regprocedure('pg_input_is_valid(text,text)') IS NOT NULL",
        )
        .fetch_one(&self.pool)
        .await?;
        let caps = SchemaCapabilities {
            data_file_partial_max: row.0,
            delete_file_partial_max: row.1,
            schema_versions: row.2,
            data_file_partition_id: row.3,
            inlined_data_tables: row.4,
            views: row.5,
            soft_input_validation: row.6,
        };
        if caps.all() {
            let _ = self.schema_capabilities.set(caps);
        }
        Ok(caps)
    }

    /// One page of the visible file listing, optionally narrowed inside SQL by
    /// catalog statistics.
    ///
    /// Backs both [`MetadataProvider::get_table_file_metadata_page`] (`filter`
    /// `None`) and [`MetadataProvider::get_table_file_metadata_page_filtered`],
    /// so the paging contract is written once.
    fn file_metadata_page(
        &self,
        table_id: i64,
        snapshot_id: i64,
        after_data_file_id: Option<i64>,
        limit: usize,
        filter: Option<&StatsFilter>,
    ) -> Result<Vec<DuckLakeFileMetadata>> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let limit = i64::try_from(limit).map_err(|_| {
            crate::DuckLakeError::InvalidConfig("file metadata page limit exceeds i64".to_string())
        })?;
        block_on(async {
            let caps = self.schema_capabilities().await?;
            let partial_max_expr = if caps.data_file_partial_max {
                "data.partial_max::bigint"
            } else {
                "NULL::bigint"
            };
            let schema_version_expr = if caps.schema_versions {
                "(SELECT sv.schema_version::bigint
                  FROM ducklake_schema_versions sv
                  WHERE sv.table_id = data.table_id
                    AND sv.begin_snapshot <= data.begin_snapshot
                  ORDER BY sv.begin_snapshot DESC LIMIT 1)"
            } else {
                "NULL::bigint"
            };
            let dialect = PostgresStatsDialect {
                soft_input_validation: caps.soft_input_validation,
            };
            let rendered = filter.and_then(|filter| filter.render(&dialect));
            let stats_sql = rendered
                .as_deref()
                .and_then(|filters| stats_filter_sql(table_id, filters));

            // The statistics conditions go inside the query, ahead of the
            // LIMIT, with the keyset ordering untouched. Filtering a page after
            // fetching it would break the cursor `FileMetadataPages` drives: a
            // page whose candidates all failed the filter would come back
            // empty, which ends the iteration and hides every matching file
            // beyond it.
            let listing_sql = |stats_sql: Option<&StatsFilterSql>| {
                let (with_prefix, joins, conditions) = stats_sql
                    .map(|sql| {
                        (
                            sql.with_prefix.as_str(),
                            sql.joins.as_str(),
                            sql.conditions.as_str(),
                        )
                    })
                    .unwrap_or_default();
                format!(
                    "{with_prefix}SELECT data.data_file_id, data.path, data.path_is_relative,
                        data.file_size_bytes, data.footer_size, data.encryption_key,
                        data.row_id_start, data.record_count,
                        del.delete_file_id, del.path, del.path_is_relative,
                        del.file_size_bytes, del.footer_size, del.encryption_key,
                        del.delete_count, data.begin_snapshot::bigint,
                        {partial_max_expr}, {schema_version_expr},
                        NULL::bigint AS data_partition_id,
                        data.mapping_id::bigint
                 FROM ducklake_data_file AS data
                 LEFT JOIN ducklake_delete_file AS del
                   ON data.data_file_id = del.data_file_id
                  AND del.table_id = $1
                  AND $2 >= del.begin_snapshot
                  AND ($3 < del.end_snapshot OR del.end_snapshot IS NULL){joins}
                 WHERE data.table_id = $4
                   AND $5 >= data.begin_snapshot
                   AND ($6 < data.end_snapshot OR data.end_snapshot IS NULL)
                   AND data.data_file_id > $7{conditions}
                 ORDER BY data.data_file_id
                 LIMIT $8"
                )
            };

            let after = after_data_file_id.unwrap_or(i64::MIN);
            let rows = match fetch_data_file_page(
                &self.pool,
                &listing_sql(stats_sql.as_ref()),
                table_id,
                snapshot_id,
                after,
                limit,
            )
            .await
            {
                Ok(rows) => rows,
                // The filter is advisory, so a catalog the narrowed query
                // cannot run — one predating `ducklake_file_column_stats`,
                // where joining it is a hard error, or one whose stats provoke
                // an error the dialect did not anticipate — still lists its
                // files. Any error retries, not just the missing table: the
                // narrowed query is the only thing this arm adds, and listing
                // every live file is always correct. The retry uses the same
                // parameters, and a failure that is not the filter's fault
                // surfaces from it.
                Err(error) if stats_sql.is_some() => {
                    tracing::debug!(
                        %error,
                        table_id,
                        "statistics-filtered file listing failed; listing every file"
                    );
                    fetch_data_file_page(
                        &self.pool,
                        &listing_sql(None),
                        table_id,
                        snapshot_id,
                        after,
                        limit,
                    )
                    .await?
                },
                Err(error) => return Err(error.into()),
            };
            let files = rows
                .iter()
                .map(|row| decode_table_file(row, snapshot_id))
                .collect::<Result<Vec<_>>>()?;
            let Some(last_data_file_id) = files.last().map(|file| file.data_file_id) else {
                return Ok(Vec::new());
            };
            // A filtered page's ids are sparse within `(after, last]`, so the
            // two enrichment queries below are restricted to the ids actually
            // returned instead of to that whole range — otherwise a selective
            // filter would read the per-column stats of every file it just
            // pruned, which is the cost the filter exists to avoid.
            //
            // The ids are bound as one array parameter rather than spelled out
            // in the SQL. Inlining a page's worth of them gives every page a
            // distinct query string, and sqlx keys its per-connection
            // prepared-statement cache on that string: each page would pay a
            // Parse/Describe round trip and evict the other statements the
            // connection had cached. The listing query's own `$1..$8` are fixed
            // by `fetch_data_file_page` and untouched by this; the two
            // enrichment queries number their parameters independently.
            let page_ids: Option<Vec<i64>> = stats_sql
                .as_ref()
                .map(|_| files.iter().map(|file| file.data_file_id).collect());
            let page_id_filter = |column: &str, parameter: usize| {
                page_ids.as_ref().map_or_else(String::new, |_| {
                    format!("\n                   AND {column} = ANY(${parameter}::bigint[])")
                })
            };
            let statistics_sql = format!(
                "SELECT stats.data_file_id, stats.column_id,
                        stats.column_size_bytes, stats.value_count, stats.null_count,
                        stats.min_value, stats.max_value, stats.contains_nan
                 FROM ducklake_file_column_stats AS stats
                 INNER JOIN ducklake_data_file AS data
                   ON data.data_file_id = stats.data_file_id
                  AND data.table_id = stats.table_id
                 WHERE stats.table_id = $1
                   AND $2 >= data.begin_snapshot
                   AND ($3 < data.end_snapshot OR data.end_snapshot IS NULL)
                   AND stats.data_file_id > $4
                   AND stats.data_file_id <= $5{}
                 ORDER BY stats.data_file_id, stats.column_id",
                page_id_filter("stats.data_file_id", 6)
            );
            let mut statistics_query = sqlx::query(AssertSqlSafe(statistics_sql))
                .bind(table_id)
                .bind(snapshot_id)
                .bind(snapshot_id)
                .bind(after_data_file_id.unwrap_or(i64::MIN))
                .bind(last_data_file_id);
            if let Some(ids) = page_ids.as_deref() {
                statistics_query = statistics_query.bind(ids);
            }
            let statistics = match statistics_query.fetch_all(&self.pool).await {
                Ok(rows) => rows
                    .into_iter()
                    .map(|row| {
                        Ok(DuckLakeFileColumnStatistics {
                            data_file_id: row.try_get(0)?,
                            column_id: row.try_get(1)?,
                            column_size_bytes: row.try_get(2)?,
                            value_count: row.try_get(3)?,
                            null_count: row.try_get(4)?,
                            min_value: row.try_get(5)?,
                            max_value: row.try_get(6)?,
                            contains_nan: row.try_get(7)?,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?,
                Err(error) if is_missing_statistics_table(&error) => Vec::new(),
                Err(error) => return Err(error.into()),
            };
            let mut statistics_by_file: HashMap<i64, Vec<_>> = HashMap::new();
            for statistic in statistics {
                statistics_by_file
                    .entry(statistic.data_file_id)
                    .or_default()
                    .push(statistic);
            }

            // Enrich with per-file partition values (for pruning), scoped to the
            // page's data_file_id range. Missing partition table => no enrichment.
            let mut values_by_file: HashMap<i64, Vec<(i32, Option<String>)>> = HashMap::new();
            let partition_values_sql = format!(
                "SELECT data_file_id, partition_key_index, partition_value
                 FROM ducklake_file_partition_value
                 WHERE table_id = $1 AND data_file_id > $2 AND data_file_id <= $3{}",
                page_id_filter("data_file_id", 4)
            );
            let mut partition_values_query = sqlx::query(AssertSqlSafe(partition_values_sql))
                .bind(table_id)
                .bind(after_data_file_id.unwrap_or(i64::MIN))
                .bind(last_data_file_id);
            if let Some(ids) = page_ids.as_deref() {
                partition_values_query = partition_values_query.bind(ids);
            }
            match partition_values_query.fetch_all(&self.pool).await {
                Ok(rows) => {
                    for row in rows {
                        let data_file_id: i64 = row.try_get(0)?;
                        let key_index: i32 = i32::try_from(row.try_get::<i64, _>(1)?).unwrap_or(0);
                        let value: Option<String> = row.try_get(2)?;
                        values_by_file
                            .entry(data_file_id)
                            .or_default()
                            .push((key_index, value));
                    }
                },
                Err(error) if is_missing_statistics_table(&error) => {},
                Err(error) => return Err(error.into()),
            }

            Ok(files
                .into_iter()
                .map(|mut file| {
                    if let Some(values) = values_by_file.remove(&file.data_file_id) {
                        file.partition_values = values;
                    }
                    DuckLakeFileMetadata {
                        column_statistics: statistics_by_file
                            .remove(&file.data_file_id)
                            .unwrap_or_default(),
                        file,
                    }
                })
                .collect())
        })
    }
}

impl MetadataProvider for PostgresMetadataProvider {
    fn get_current_snapshot(&self) -> Result<i64> {
        block_on(async {
            let row = sqlx::query("SELECT COALESCE(MAX(snapshot_id), 0) FROM ducklake_snapshot")
                .fetch_one(&self.pool)
                .await?;
            Ok(row.try_get(0)?)
        })
    }

    fn get_data_path(&self) -> Result<String> {
        self.get_metadata_settings(None, None)?
            .remove("data_path")
            .ok_or_else(|| {
                crate::error::DuckLakeError::InvalidConfig(
                    "Missing required catalog metadata: 'data_path' not configured. \
                     The catalog may be uninitialized or corrupted."
                        .to_string(),
                )
            })
    }

    fn get_metadata_settings(
        &self,
        schema_id: Option<i64>,
        table_id: Option<i64>,
    ) -> Result<HashMap<String, String>> {
        block_on(async {
            let has_scope_columns: bool = sqlx::query_scalar(
                "SELECT COUNT(*) = 2 FROM information_schema.columns \
                 WHERE table_schema = current_schema() \
                 AND table_name = 'ducklake_metadata' \
                 AND column_name IN ('scope', 'scope_id')",
            )
            .fetch_one(&self.pool)
            .await?;
            let rows = if has_scope_columns {
                sqlx::query("SELECT key, value, scope, scope_id FROM ducklake_metadata")
                    .fetch_all(&self.pool)
                    .await?
                    .into_iter()
                    .map(|row| {
                        Ok(MetadataSetting {
                            key: row.try_get(0)?,
                            value: row.try_get(1)?,
                            scope: row.try_get(2)?,
                            scope_id: row.try_get(3)?,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?
            } else {
                sqlx::query("SELECT key, value FROM ducklake_metadata")
                    .fetch_all(&self.pool)
                    .await?
                    .into_iter()
                    .map(|row| {
                        Ok(MetadataSetting {
                            key: row.try_get(0)?,
                            value: row.try_get(1)?,
                            scope: None,
                            scope_id: None,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?
            };
            resolve_metadata_settings(rows, schema_id, table_id)
        })
    }

    fn list_snapshots(&self) -> Result<Vec<SnapshotMetadata>> {
        block_on(async {
            let rows = sqlx::query(
                "SELECT snapshot_id, snapshot_time
                 FROM ducklake_snapshot ORDER BY snapshot_id",
            )
            .fetch_all(&self.pool)
            .await?;

            rows.into_iter()
                .map(|row| {
                    let snapshot_id: i64 = row.try_get(0)?;
                    let timestamp: Option<NaiveDateTime> = row.try_get(1)?;
                    let timestamp_str = timestamp
                        .map(|ts: NaiveDateTime| ts.format("%Y-%m-%d %H:%M:%S%.6f").to_string());

                    Ok(SnapshotMetadata {
                        snapshot_id,
                        timestamp: timestamp_str,
                    })
                })
                .collect()
        })
    }

    fn list_snapshot_changes(&self) -> Result<Vec<SnapshotChangeMetadata>> {
        block_on(async {
            let rows = sqlx::query(
                "SELECT snapshot.snapshot_id,
                        snapshot.snapshot_time::text AS snapshot_time,
                        changes.changes_made,
                        changes.author,
                        changes.commit_message,
                        changes.commit_extra_info
                 FROM ducklake_snapshot AS snapshot
                 JOIN ducklake_snapshot_changes AS changes
                   ON changes.snapshot_id = snapshot.snapshot_id
                 ORDER BY snapshot.snapshot_id",
            )
            .fetch_all(&self.pool)
            .await?;

            rows.into_iter()
                .map(|row| {
                    Ok(SnapshotChangeMetadata {
                        snapshot_id: row.try_get("snapshot_id")?,
                        timestamp: row.try_get("snapshot_time")?,
                        changes_made: row.try_get("changes_made")?,
                        author: row.try_get("author")?,
                        commit_message: row.try_get("commit_message")?,
                        commit_extra_info: row.try_get("commit_extra_info")?,
                    })
                })
                .collect()
        })
    }

    fn find_snapshot_by_commit_extra_info(&self, needle: &str) -> Result<Option<i64>> {
        block_on(async {
            let row = sqlx::query(
                "SELECT changes.snapshot_id
                 FROM ducklake_snapshot_changes AS changes
                 WHERE (changes.commit_extra_info = $1
                        OR strpos(changes.commit_extra_info, $2) > 0)
                   AND EXISTS (
                       SELECT 1 FROM ducklake_data_file AS files
                       WHERE files.begin_snapshot = changes.snapshot_id
                         AND files.end_snapshot IS NULL
                   )
                 ORDER BY changes.snapshot_id
                 LIMIT 1",
            )
            .bind(needle)
            .bind(needle)
            .fetch_optional(&self.pool)
            .await?;

            Ok(row.map(|row| row.try_get("snapshot_id")).transpose()?)
        })
    }

    fn list_schemas(&self, snapshot_id: i64) -> Result<Vec<SchemaMetadata>> {
        block_on(async {
            let rows = sqlx::query(
                "SELECT schema_id, schema_name, path, path_is_relative FROM ducklake_schema
                 WHERE $1 >= begin_snapshot AND ($2 < end_snapshot OR end_snapshot IS NULL)",
            )
            .bind(snapshot_id)
            .bind(snapshot_id)
            .fetch_all(&self.pool)
            .await?;

            rows.into_iter()
                .map(|row| {
                    Ok(SchemaMetadata {
                        schema_id: row.try_get(0)?,
                        schema_name: row.try_get(1)?,
                        path: row.try_get(2)?,
                        path_is_relative: row.try_get(3)?,
                    })
                })
                .collect()
        })
    }

    fn list_tables(&self, schema_id: i64, snapshot_id: i64) -> Result<Vec<TableMetadata>> {
        block_on(async {
            let rows = sqlx::query(
                "SELECT table_id, table_name, path, path_is_relative FROM ducklake_table
                 WHERE schema_id = $1
                   AND $2 >= begin_snapshot
                   AND ($3 < end_snapshot OR end_snapshot IS NULL)",
            )
            .bind(schema_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .fetch_all(&self.pool)
            .await?;

            rows.into_iter()
                .map(|row| {
                    Ok(TableMetadata {
                        table_id: row.try_get(0)?,
                        table_name: row.try_get(1)?,
                        path: row.try_get(2)?,
                        path_is_relative: row.try_get(3)?,
                    })
                })
                .collect()
        })
    }

    fn list_views(&self, schema_id: i64, snapshot_id: i64) -> Result<Vec<ViewMetadata>> {
        block_on(async {
            if !self.schema_capabilities().await?.views {
                return Ok(Vec::new());
            }
            let rows = sqlx::query(
                "SELECT view_id, schema_id, begin_snapshot, view_name, dialect, sql, column_aliases
                 FROM ducklake_view
                 WHERE schema_id = $1
                   AND $2 >= begin_snapshot
                   AND ($3 < end_snapshot OR end_snapshot IS NULL)",
            )
            .bind(schema_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .fetch_all(&self.pool)
            .await?;

            rows.into_iter().map(|row| decode_view(&row)).collect()
        })
    }

    fn get_table_structure(
        &self,
        table_id: i64,
        snapshot_id: i64,
    ) -> Result<Vec<DuckLakeTableColumn>> {
        block_on(async {
            let rows = sqlx::query(
                "SELECT column_id, column_name, column_type, nulls_allowed, parent_column,
                        initial_default, default_value, default_value_type, default_value_dialect
                 FROM ducklake_column
                 WHERE table_id = $1
                   AND $2 >= begin_snapshot
                   AND ($3 < end_snapshot OR end_snapshot IS NULL)
                 ORDER BY column_order",
            )
            .bind(table_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .fetch_all(&self.pool)
            .await?;

            let raw: Result<Vec<(DuckLakeTableColumn, Option<i64>)>> = rows
                .into_iter()
                .map(|row| {
                    let nulls_allowed: Option<bool> = row.try_get(3)?;
                    let parent_column: Option<i64> = row.try_get(4)?;
                    Ok((
                        DuckLakeTableColumn::new(
                            row.try_get(0)?,
                            row.try_get(1)?,
                            row.try_get(2)?,
                            nulls_allowed.unwrap_or(true),
                        )
                        .with_defaults(
                            row.try_get(5)?,
                            row.try_get(6)?,
                            row.try_get(7)?,
                            row.try_get(8)?,
                        ),
                        parent_column,
                    ))
                })
                .collect();
            reconstruct_columns(raw?)
        })
    }

    fn get_table_fields(&self, table_id: i64, snapshot_id: i64) -> Result<Vec<DuckLakeTableField>> {
        block_on(async {
            let rows = sqlx::query(
                "SELECT column_id, column_name, column_type, nulls_allowed, parent_column
                 FROM ducklake_column
                 WHERE table_id = $1
                   AND $2 >= begin_snapshot
                   AND ($3 < end_snapshot OR end_snapshot IS NULL)
                 ORDER BY column_order",
            )
            .bind(table_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .fetch_all(&self.pool)
            .await?;
            rows.into_iter()
                .map(|row| {
                    Ok(DuckLakeTableField {
                        column_id: row.try_get(0)?,
                        column_name: row.try_get(1)?,
                        column_type: row.try_get(2)?,
                        is_nullable: row.try_get::<Option<bool>, _>(3)?.unwrap_or(true),
                        parent_column: row.try_get(4)?,
                    })
                })
                .collect()
        })
    }

    fn get_name_mapping(&self, mapping_id: i64) -> Result<DuckLakeNameMapping> {
        block_on(async {
            let rows = sqlx::query(
                "SELECT mapping.mapping_id, mapping.table_id, mapping.type,
                        name.column_id, name.source_name, name.target_field_id,
                        name.parent_column, name.is_partition
                 FROM ducklake_column_mapping AS mapping
                 JOIN ducklake_name_mapping AS name
                   ON name.mapping_id = mapping.mapping_id
                 WHERE mapping.mapping_id = $1
                 ORDER BY name.parent_column NULLS FIRST, name.column_id",
            )
            .bind(mapping_id)
            .fetch_all(&self.pool)
            .await?;
            decode_name_mapping_rows(mapping_id, &rows)
        })
    }

    fn get_table_files_for_select(
        &self,
        table_id: i64,
        snapshot_id: i64,
    ) -> Result<Vec<DuckLakeTableFile>> {
        block_on(async {
            // Backward compatibility: minimal / pre-v1.0 catalogs may lack the
            // `partial_max` column and the `ducklake_schema_versions` ledger.
            // Detect both and degrade those projections to NULL so plain reads
            // still work (both are consumed only by compaction; `partial_max`
            // also by time-travel reads of partial files, which such catalogs
            // never contain).
            let caps = self.schema_capabilities().await?;
            let partial_max_expr = if caps.data_file_partial_max {
                "data.partial_max::bigint"
            } else {
                "NULL::bigint"
            };
            // A catalog predating partition support has no `partition_id` column;
            // such a catalog holds no partitioned files either, so NULL is exact.
            let partition_id_expr = if caps.data_file_partition_id {
                "data.partition_id::bigint"
            } else {
                "NULL::bigint"
            };
            let schema_version_expr = if caps.schema_versions {
                "(SELECT sv.schema_version::bigint
                  FROM ducklake_schema_versions sv
                  WHERE sv.table_id = data.table_id
                    AND sv.begin_snapshot <= data.begin_snapshot
                  ORDER BY sv.begin_snapshot DESC
                  LIMIT 1)"
            } else {
                "NULL::bigint"
            };
            let sql = format!(
                "SELECT
                    data.data_file_id,
                    data.path AS data_file_path,
                    data.path_is_relative AS data_path_is_relative,
                    data.file_size_bytes AS data_file_size,
                    data.footer_size AS data_footer_size,
                    data.encryption_key AS data_encryption_key,
                    data.row_id_start AS data_row_id_start,
                    data.record_count AS data_record_count,
                    del.delete_file_id,
                    del.path AS delete_file_path,
                    del.path_is_relative AS delete_path_is_relative,
                    del.file_size_bytes AS delete_file_size,
                    del.footer_size AS delete_footer_size,
                    del.encryption_key AS delete_encryption_key,
                    del.delete_count,
                    data.begin_snapshot::bigint AS data_begin_snapshot,
                    {partial_max_expr} AS data_partial_max,
                    {schema_version_expr} AS data_schema_version,
                    {partition_id_expr} AS data_partition_id,
                    data.mapping_id::bigint AS data_mapping_id
                FROM ducklake_data_file AS data
                LEFT JOIN ducklake_delete_file AS del
                    ON data.data_file_id = del.data_file_id
                    AND del.table_id = $1
                    AND $2 >= del.begin_snapshot
                    AND ($3 < del.end_snapshot OR del.end_snapshot IS NULL)
                WHERE data.table_id = $4
                  AND $5 >= data.begin_snapshot
                  AND ($6 < data.end_snapshot OR data.end_snapshot IS NULL)"
            );
            let rows = sqlx::query(AssertSqlSafe(sql.as_str()))
                .bind(table_id)
                .bind(snapshot_id)
                .bind(snapshot_id)
                .bind(table_id)
                .bind(snapshot_id)
                .bind(snapshot_id)
                .fetch_all(&self.pool)
                .await?;

            let mut files: Vec<DuckLakeTableFile> = rows
                .iter()
                .map(|row| decode_table_file(row, snapshot_id))
                .collect::<Result<Vec<_>>>()?;

            // Enrich with per-file partition values. Compaction reads its candidates
            // through this path and must group by, and preserve, each file's exact
            // partition — without the values it would merge across partitions and
            // strip the assignment. Scoped by the fetched id range; a catalog
            // predating partition support simply yields no rows.
            if let (Some(min), Some(max)) = (
                files.iter().map(|f| f.data_file_id).min(),
                files.iter().map(|f| f.data_file_id).max(),
            ) {
                let mut values_by_file: HashMap<i64, Vec<(i32, Option<String>)>> = HashMap::new();
                match sqlx::query(
                    "SELECT data_file_id, partition_key_index, partition_value
                     FROM ducklake_file_partition_value
                     WHERE table_id = $1 AND data_file_id >= $2 AND data_file_id <= $3",
                )
                .bind(table_id)
                .bind(min)
                .bind(max)
                .fetch_all(&self.pool)
                .await
                {
                    Ok(rows) => {
                        for row in rows {
                            let data_file_id: i64 = row.try_get(0)?;
                            let key_index: i32 =
                                i32::try_from(row.try_get::<i64, _>(1)?).unwrap_or(0);
                            let value: Option<String> = row.try_get(2)?;
                            values_by_file
                                .entry(data_file_id)
                                .or_default()
                                .push((key_index, value));
                        }
                    },
                    Err(error) if is_missing_statistics_table(&error) => {},
                    Err(error) => return Err(error.into()),
                }
                for file in &mut files {
                    if let Some(values) = values_by_file.remove(&file.data_file_id) {
                        file.partition_values = values;
                    }
                }
            }
            Ok(files)
        })
    }

    fn get_partition_spec(&self, table_id: i64, snapshot_id: i64) -> Result<Option<PartitionSpec>> {
        block_on(async {
            // Pruning is only safe with exactly one spec generation ever (see
            // PartitionSpec::prune_safe); the live spec is returned regardless so
            // the write path always targets the current generation.
            let generation_count: i64 = match sqlx::query_scalar(
                "SELECT COUNT(*) FROM ducklake_partition_info WHERE table_id = $1",
            )
            .bind(table_id)
            .fetch_one(&self.pool)
            .await
            {
                Ok(count) => count,
                Err(error) if is_missing_statistics_table(&error) => return Ok(None),
                Err(error) => return Err(error.into()),
            };
            let prune_safe = generation_count == 1;
            let rows = match sqlx::query(
                "SELECT pi.partition_id, pc.partition_key_index, pc.column_id, pc.transform
                 FROM ducklake_partition_info AS pi
                 JOIN ducklake_partition_column AS pc
                   ON pc.partition_id = pi.partition_id AND pc.table_id = pi.table_id
                 WHERE pi.table_id = $1
                   AND $2 >= pi.begin_snapshot
                   AND ($3 < pi.end_snapshot OR pi.end_snapshot IS NULL)
                 ORDER BY pc.partition_key_index",
            )
            .bind(table_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .fetch_all(&self.pool)
            .await
            {
                Ok(rows) => rows,
                Err(error) if is_missing_statistics_table(&error) => return Ok(None),
                Err(error) => return Err(error.into()),
            };
            let parsed = rows
                .iter()
                .map(|row| {
                    Ok::<_, crate::DuckLakeError>((
                        row.try_get::<i64, _>(0)?,
                        i32::try_from(row.try_get::<i64, _>(1)?).unwrap_or(0),
                        row.try_get::<i64, _>(2)?,
                        row.try_get::<String, _>(3)?,
                    ))
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(PartitionSpec::from_rows(parsed, prune_safe))
        })
    }

    fn get_sort_spec(&self, table_id: i64, snapshot_id: i64) -> Result<Option<SortSpec>> {
        block_on(async {
            let rows = match sqlx::query(
                "SELECT si.sort_id, se.sort_key_index, se.expression, se.dialect,
                        se.sort_direction, se.null_order
                 FROM ducklake_sort_info AS si
                 JOIN ducklake_sort_expression AS se
                   ON se.sort_id = si.sort_id AND se.table_id = si.table_id
                 WHERE si.table_id = $1
                   AND $2 >= si.begin_snapshot
                   AND ($3 < si.end_snapshot OR si.end_snapshot IS NULL)
                 ORDER BY se.sort_key_index",
            )
            .bind(table_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .fetch_all(&self.pool)
            .await
            {
                Ok(rows) => rows,
                Err(error) if is_missing_statistics_table(&error) => return Ok(None),
                Err(error) => return Err(error.into()),
            };
            let parsed = rows
                .iter()
                .map(|row| {
                    Ok::<_, crate::DuckLakeError>((
                        row.try_get::<i64, _>(0)?,
                        i32::try_from(row.try_get::<i64, _>(1)?).unwrap_or(0),
                        row.try_get::<String, _>(2)?,
                        row.try_get::<String, _>(3)?,
                        row.try_get::<String, _>(4)?,
                        row.try_get::<String, _>(5)?,
                    ))
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(SortSpec::from_rows(parsed))
        })
    }

    fn get_table_file_metadata_page(
        &self,
        table_id: i64,
        snapshot_id: i64,
        after_data_file_id: Option<i64>,
        limit: usize,
    ) -> Result<Vec<DuckLakeFileMetadata>> {
        self.file_metadata_page(table_id, snapshot_id, after_data_file_id, limit, None)
    }

    fn get_table_file_metadata_page_filtered(
        &self,
        table_id: i64,
        snapshot_id: i64,
        after_data_file_id: Option<i64>,
        limit: usize,
        filter: Option<&StatsFilter>,
    ) -> Result<Vec<DuckLakeFileMetadata>> {
        self.file_metadata_page(table_id, snapshot_id, after_data_file_id, limit, filter)
    }

    fn get_table_summary_statistics(
        &self,
        table_id: i64,
        snapshot_id: i64,
    ) -> Result<DuckLakeStatistics> {
        block_on(async {
            let table = match sqlx::query(
                "SELECT record_count, file_size_bytes
                 FROM ducklake_table_stats WHERE table_id = $1",
            )
            .bind(table_id)
            .fetch_optional(&self.pool)
            .await
            {
                Ok(row) => row
                    .map(|row| {
                        Ok::<_, sqlx::Error>(DuckLakeTableStatistics {
                            record_count: row.try_get(0)?,
                            file_size_bytes: row.try_get(1)?,
                        })
                    })
                    .transpose()?,
                Err(error) if is_missing_statistics_table(&error) => None,
                Err(error) => return Err(error.into()),
            };
            let column_sizes = match sqlx::query(
                "SELECT stats.column_id,
                        CASE
                          WHEN COUNT(*) = COUNT(stats.column_size_bytes)
                           AND COUNT(*) = (
                             SELECT COUNT(*) FROM ducklake_data_file visible
                             WHERE visible.table_id = $1
                               AND $2 >= visible.begin_snapshot
                               AND ($3 < visible.end_snapshot OR visible.end_snapshot IS NULL)
                           )
                          THEN CAST(SUM(stats.column_size_bytes) AS BIGINT)
                        END
                 FROM ducklake_file_column_stats stats
                 INNER JOIN ducklake_data_file data
                   ON data.data_file_id = stats.data_file_id
                  AND data.table_id = stats.table_id
                 WHERE stats.table_id = $4
                   AND $5 >= data.begin_snapshot
                   AND ($6 < data.end_snapshot OR data.end_snapshot IS NULL)
                 GROUP BY stats.column_id",
            )
            .bind(table_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .bind(table_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .fetch_all(&self.pool)
            .await
            {
                Ok(rows) => rows
                    .into_iter()
                    .filter_map(|row| match row.try_get::<Option<i64>, _>(1) {
                        Ok(Some(size)) => Some(row.try_get(0).map(|column_id| (column_id, size))),
                        Ok(None) => None,
                        Err(error) => Some(Err(error)),
                    })
                    .collect::<std::result::Result<HashMap<i64, i64>, _>>()?,
                Err(error) if is_missing_statistics_table(&error) => HashMap::new(),
                Err(error) => return Err(error.into()),
            };
            let bounds_are_exact: bool = sqlx::query_scalar(
                "SELECT NOT EXISTS (
                     SELECT 1 FROM ducklake_delete_file
                     WHERE table_id = $1
                       AND $2 >= begin_snapshot
                       AND ($3 < end_snapshot OR end_snapshot IS NULL)
                 )",
            )
            .bind(table_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .fetch_one(&self.pool)
            .await?;
            let columns = match sqlx::query(
                "SELECT column_id, contains_null, min_value, max_value, contains_nan
                 FROM ducklake_table_column_stats WHERE table_id = $1",
            )
            .bind(table_id)
            .fetch_all(&self.pool)
            .await
            {
                Ok(rows) => rows
                    .into_iter()
                    .map(|row| {
                        let column_id = row.try_get(0)?;
                        Ok(DuckLakeTableColumnStatistics {
                            column_id,
                            contains_null: row.try_get(1)?,
                            min_value: row.try_get(2)?,
                            max_value: row.try_get(3)?,
                            contains_nan: row.try_get(4)?,
                            column_size_bytes: column_sizes.get(&column_id).copied(),
                            bounds_are_exact,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?,
                Err(error) if is_missing_statistics_table(&error) => Vec::new(),
                Err(error) => return Err(error.into()),
            };
            Ok(DuckLakeStatistics {
                table,
                columns,
                files: Vec::new(),
            })
        })
    }

    fn get_table_statistics(&self, table_id: i64, snapshot_id: i64) -> Result<DuckLakeStatistics> {
        block_on(async {
            let table = match sqlx::query(
                "SELECT record_count, file_size_bytes
                 FROM ducklake_table_stats WHERE table_id = $1",
            )
            .bind(table_id)
            .fetch_optional(&self.pool)
            .await
            {
                Ok(row) => row
                    .map(|row| {
                        Ok::<_, sqlx::Error>(DuckLakeTableStatistics {
                            record_count: row.try_get(0)?,
                            file_size_bytes: row.try_get(1)?,
                        })
                    })
                    .transpose()?,
                Err(error) if is_missing_statistics_table(&error) => None,
                Err(error) => return Err(error.into()),
            };

            let columns = match sqlx::query(
                "SELECT column_id, contains_null, min_value, max_value, contains_nan
                 FROM ducklake_table_column_stats WHERE table_id = $1",
            )
            .bind(table_id)
            .fetch_all(&self.pool)
            .await
            {
                Ok(rows) => rows
                    .into_iter()
                    .map(|row| {
                        Ok(DuckLakeTableColumnStatistics {
                            column_id: row.try_get(0)?,
                            contains_null: row.try_get(1)?,
                            min_value: row.try_get(2)?,
                            max_value: row.try_get(3)?,
                            contains_nan: row.try_get(4)?,
                            column_size_bytes: None,
                            bounds_are_exact: false,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?,
                Err(error) if is_missing_statistics_table(&error) => Vec::new(),
                Err(error) => return Err(error.into()),
            };

            let files = match sqlx::query(
                "SELECT
                    stats.data_file_id,
                    stats.column_id,
                    stats.column_size_bytes,
                    stats.value_count,
                    stats.null_count,
                    stats.min_value,
                    stats.max_value,
                    stats.contains_nan
                 FROM ducklake_file_column_stats AS stats
                 INNER JOIN ducklake_data_file AS data
                    ON data.data_file_id = stats.data_file_id
                    AND data.table_id = stats.table_id
                 WHERE stats.table_id = $1
                   AND $2 >= data.begin_snapshot
                   AND ($3 < data.end_snapshot OR data.end_snapshot IS NULL)",
            )
            .bind(table_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .fetch_all(&self.pool)
            .await
            {
                Ok(rows) => rows
                    .into_iter()
                    .map(|row| {
                        Ok(DuckLakeFileColumnStatistics {
                            data_file_id: row.try_get(0)?,
                            column_id: row.try_get(1)?,
                            column_size_bytes: row.try_get(2)?,
                            value_count: row.try_get(3)?,
                            null_count: row.try_get(4)?,
                            min_value: row.try_get(5)?,
                            max_value: row.try_get(6)?,
                            contains_nan: row.try_get(7)?,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?,
                Err(error) if is_missing_statistics_table(&error) => Vec::new(),
                Err(error) => return Err(error.into()),
            };

            Ok(DuckLakeStatistics {
                table,
                columns,
                files,
            })
        })
    }

    fn get_inlined_data(
        &self,
        table_id: i64,
        snapshot_id: i64,
        columns: &[DuckLakeTableColumn],
    ) -> Result<Vec<RecordBatch>> {
        Ok(self
            .scan_inlined_data(table_id, snapshot_id, columns, None)?
            .batches)
    }

    fn scan_inlined_data(
        &self,
        table_id: i64,
        snapshot_id: i64,
        columns: &[DuckLakeTableColumn],
        filter: Option<&InlinedFilter>,
    ) -> Result<InlinedDataScan> {
        block_on(async {
            if !self.schema_capabilities().await?.inlined_data_tables {
                return Ok(InlinedDataScan::default());
            }
            let registry = sqlx::query(
                "SELECT table_name FROM ducklake_inlined_data_tables WHERE table_id = $1",
            )
            .bind(table_id)
            .fetch_all(&self.pool)
            .await?;
            let schema: SchemaRef = Arc::new(crate::types::build_arrow_schema(columns)?);
            let mut batches = Vec::new();

            for entry in registry {
                let table: String = entry.try_get("table_name")?;
                if !is_inlined_data_table(&table) {
                    continue;
                }
                let physical_columns = sqlx::query(
                    "SELECT column_name, data_type FROM information_schema.columns
                 WHERE table_schema = current_schema() AND table_name = $1",
                )
                .bind(&table)
                .fetch_all(&self.pool)
                .await?;
                let present = physical_columns
                    .iter()
                    .map(|row| row.try_get::<String, _>(0))
                    .collect::<std::result::Result<HashSet<_>, _>>()?;
                let physical_types = physical_columns
                    .iter()
                    .map(|row| Ok((row.try_get::<String, _>(0)?, row.try_get::<String, _>(1)?)))
                    .collect::<std::result::Result<HashMap<_, _>, sqlx::Error>>()?;
                let projected = columns
                    .iter()
                    .zip(schema.fields())
                    .map(|(column, field)| {
                        if !present.contains(&column.column_name) {
                            "NULL::text".to_string()
                        } else {
                            let ident = quote_ident(&column.column_name);
                            inlined_text_projection(
                                InlinedDataBackend::Postgres,
                                column,
                                field.data_type(),
                                &ident,
                            )
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                let rendered = filter.and_then(|filter| {
                    render_inlined_filter(
                        filter,
                        InlinedSqlDialect::Postgres,
                        schema.as_ref(),
                        &physical_types,
                        2,
                    )
                });
                let pushed = rendered
                    .as_ref()
                    .map(|rendered| format!(" AND ({})", rendered.sql))
                    .unwrap_or_default();
                let sql = format!(
                    "SELECT {projected} FROM {} \
                 WHERE $1 >= begin_snapshot AND ($2 < end_snapshot OR end_snapshot IS NULL){pushed} \
                     ORDER BY row_id",
                    quote_ident(&table)
                );
                let mut query = sqlx::query(AssertSqlSafe(sql.as_str()))
                    .bind(snapshot_id)
                    .bind(snapshot_id);
                if let Some(rendered) = rendered {
                    for bind in rendered.binds {
                        query = match bind {
                            InlinedSqlBind::Bool(value) => query.bind(value),
                            InlinedSqlBind::I64(value) => query.bind(value),
                            InlinedSqlBind::U64(value) => query.bind(value.to_string()),
                            InlinedSqlBind::F64(value) => query.bind(value),
                            InlinedSqlBind::Text(value) => query.bind(value),
                            InlinedSqlBind::Bytes(value) => query.bind(value),
                        };
                    }
                }
                let rows = query.fetch_all(&self.pool).await?;
                if rows.is_empty() {
                    continue;
                }
                let rows = rows
                    .into_iter()
                    .map(|row| {
                        (0..columns.len())
                            .map(|index| row.try_get::<Option<String>, _>(index))
                            .collect::<std::result::Result<Vec<_>, _>>()
                    })
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                batches.push(parse_inlined_rows_with_present(
                    schema.clone(),
                    columns,
                    rows,
                    Some(&present),
                )?);
            }
            Ok(InlinedDataScan::from_batches(batches))
        })
    }

    fn get_inlined_data_with_row_ids(
        &self,
        table_id: i64,
        snapshot_id: i64,
        columns: &[DuckLakeTableColumn],
    ) -> Result<Vec<DuckLakeInlinedData>> {
        block_on(async {
            if !self.schema_capabilities().await?.inlined_data_tables {
                return Ok(Vec::new());
            }
            let entries = sqlx::query(
                "SELECT table_name FROM ducklake_inlined_data_tables WHERE table_id = $1",
            )
            .bind(table_id)
            .fetch_all(&self.pool)
            .await?;
            let schema: SchemaRef = Arc::new(crate::types::build_strict_arrow_schema(columns)?);
            let mut batches = Vec::new();

            for entry in entries {
                let table: String = entry.try_get("table_name")?;
                if !is_inlined_data_table(&table) {
                    continue;
                }
                let present = sqlx::query(
                    "SELECT column_name FROM information_schema.columns
                     WHERE table_schema = current_schema() AND table_name = $1",
                )
                .bind(&table)
                .fetch_all(&self.pool)
                .await?
                .into_iter()
                .map(|row| row.try_get::<String, _>(0))
                .collect::<std::result::Result<HashSet<_>, _>>()?;
                let projected = columns
                    .iter()
                    .zip(schema.fields())
                    .map(|(column, field)| {
                        if !present.contains(&column.column_name) {
                            "NULL::text".to_string()
                        } else {
                            let ident = quote_ident(&column.column_name);
                            inlined_text_projection(
                                InlinedDataBackend::Postgres,
                                column,
                                field.data_type(),
                                &ident,
                            )
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                let sql = format!(
                    "SELECT row_id, begin_snapshot, {projected} FROM {}
                     WHERE $1 >= begin_snapshot AND ($2 < end_snapshot OR end_snapshot IS NULL)
                     ORDER BY begin_snapshot, row_id",
                    quote_ident(&table)
                );
                let rows = sqlx::query(AssertSqlSafe(sql.as_str()))
                    .bind(snapshot_id)
                    .bind(snapshot_id)
                    .fetch_all(&self.pool)
                    .await?;
                if rows.is_empty() {
                    continue;
                }
                let row_ids = rows
                    .iter()
                    .map(|row| row.try_get("row_id"))
                    .collect::<std::result::Result<Vec<i64>, _>>()?;
                let begin_snapshots = rows
                    .iter()
                    .map(|row| row.try_get("begin_snapshot"))
                    .collect::<std::result::Result<Vec<i64>, _>>()?;
                let decoded_rows = rows
                    .into_iter()
                    .map(|row| {
                        (0..columns.len())
                            .map(|index| row.try_get::<Option<String>, _>(index + 2))
                            .collect::<std::result::Result<Vec<_>, _>>()
                    })
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                batches.push(DuckLakeInlinedData {
                    table_name: table,
                    row_ids,
                    begin_snapshots,
                    batch: parse_inlined_rows_with_present(
                        schema.clone(),
                        columns,
                        decoded_rows,
                        Some(&present),
                    )?,
                });
            }
            Ok(batches)
        })
    }

    fn get_inlined_deletes(
        &self,
        table_id: i64,
        snapshot_id: i64,
    ) -> Result<Vec<DuckLakeInlinedDelete>> {
        block_on(async {
            let table = inlined_delete_table_name(table_id)?;
            let sql = format!(
                "SELECT file_id, row_id FROM {} WHERE begin_snapshot <= $1 ORDER BY file_id, row_id",
                quote_ident(&table)
            );
            match sqlx::query(AssertSqlSafe(sql.as_str()))
                .bind(snapshot_id)
                .fetch_all(&self.pool)
                .await
            {
                Ok(rows) => rows
                    .into_iter()
                    .map(|row| {
                        Ok(DuckLakeInlinedDelete {
                            data_file_id: row.try_get(0)?,
                            row_id: row.try_get(1)?,
                        })
                    })
                    .collect(),
                Err(error) if is_missing_statistics_table(&error) => Ok(Vec::new()),
                Err(error) => Err(error.into()),
            }
        })
    }

    fn get_schema_by_name(&self, name: &str, snapshot_id: i64) -> Result<Option<SchemaMetadata>> {
        block_on(async {
            let row = sqlx::query(
                "SELECT schema_id, schema_name, path, path_is_relative FROM ducklake_schema
                 WHERE schema_name = $1
                   AND $2 >= begin_snapshot
                   AND ($3 < end_snapshot OR end_snapshot IS NULL)",
            )
            .bind(name)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .fetch_optional(&self.pool)
            .await?;

            match row {
                Some(r) => Ok(Some(SchemaMetadata {
                    schema_id: r.try_get(0)?,
                    schema_name: r.try_get(1)?,
                    path: r.try_get(2)?,
                    path_is_relative: r.try_get(3)?,
                })),
                None => Ok(None),
            }
        })
    }

    fn get_table_by_name(
        &self,
        schema_id: i64,
        name: &str,
        snapshot_id: i64,
    ) -> Result<Option<TableMetadata>> {
        block_on(async {
            let row = sqlx::query(
                "SELECT table_id, table_name, path, path_is_relative FROM ducklake_table
                 WHERE schema_id = $1
                   AND table_name = $2
                   AND $3 >= begin_snapshot
                   AND ($4 < end_snapshot OR end_snapshot IS NULL)",
            )
            .bind(schema_id)
            .bind(name)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .fetch_optional(&self.pool)
            .await?;

            match row {
                Some(r) => Ok(Some(TableMetadata {
                    table_id: r.try_get(0)?,
                    table_name: r.try_get(1)?,
                    path: r.try_get(2)?,
                    path_is_relative: r.try_get(3)?,
                })),
                None => Ok(None),
            }
        })
    }

    fn get_view_by_name(
        &self,
        schema_id: i64,
        name: &str,
        snapshot_id: i64,
    ) -> Result<Option<ViewMetadata>> {
        block_on(async {
            if !self.schema_capabilities().await?.views {
                return Ok(None);
            }
            let row = sqlx::query(
                "SELECT view_id, schema_id, begin_snapshot, view_name, dialect, sql, column_aliases
                 FROM ducklake_view
                 WHERE schema_id = $1
                   AND view_name = $2
                   AND $3 >= begin_snapshot
                   AND ($4 < end_snapshot OR end_snapshot IS NULL)",
            )
            .bind(schema_id)
            .bind(name)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .fetch_optional(&self.pool)
            .await?;

            row.map(|row| decode_view(&row)).transpose()
        })
    }

    fn table_exists(&self, schema_id: i64, name: &str, snapshot_id: i64) -> Result<bool> {
        block_on(async {
            let row = sqlx::query(
                "SELECT EXISTS(
                    SELECT 1 FROM ducklake_table
                    WHERE schema_id = $1
                      AND table_name = $2
                      AND $3 >= begin_snapshot
                      AND ($4 < end_snapshot OR end_snapshot IS NULL)
                )",
            )
            .bind(schema_id)
            .bind(name)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .fetch_one(&self.pool)
            .await?;

            Ok(row.try_get(0)?)
        })
    }

    fn list_all_tables(&self, snapshot_id: i64) -> Result<Vec<TableWithSchema>> {
        block_on(async {
            let rows = bind_repeat!(
                sqlx::query(
                    "SELECT s.schema_name, t.table_id, t.table_name, t.path, t.path_is_relative
                     FROM ducklake_schema s
                     JOIN ducklake_table t ON s.schema_id = t.schema_id
                     WHERE $1 >= s.begin_snapshot
                       AND ($2 < s.end_snapshot OR s.end_snapshot IS NULL)
                       AND $3 >= t.begin_snapshot
                       AND ($4 < t.end_snapshot OR t.end_snapshot IS NULL)
                     ORDER BY s.schema_name, t.table_name"
                ),
                snapshot_id,
                4
            )
            .fetch_all(&self.pool)
            .await?;

            rows.into_iter()
                .map(|row| {
                    let schema_name: String = row.try_get(0)?;
                    let table = TableMetadata {
                        table_id: row.try_get(1)?,
                        table_name: row.try_get(2)?,
                        path: row.try_get(3)?,
                        path_is_relative: row.try_get(4)?,
                    };
                    Ok(TableWithSchema {
                        schema_name,
                        table,
                    })
                })
                .collect()
        })
    }

    fn list_all_views(&self, snapshot_id: i64) -> Result<Vec<ViewWithSchema>> {
        block_on(async {
            if !self.schema_capabilities().await?.views {
                return Ok(Vec::new());
            }
            let rows = sqlx::query(
                "SELECT s.schema_name, v.view_id, v.schema_id, v.begin_snapshot, v.view_name,
                        v.dialect, v.sql, v.column_aliases
                 FROM ducklake_schema s
                 JOIN ducklake_view v ON s.schema_id = v.schema_id
                 WHERE $1 >= s.begin_snapshot
                   AND ($1 < s.end_snapshot OR s.end_snapshot IS NULL)
                   AND $1 >= v.begin_snapshot
                   AND ($1 < v.end_snapshot OR v.end_snapshot IS NULL)
                 ORDER BY s.schema_name, v.view_name",
            )
            .bind(snapshot_id)
            .fetch_all(&self.pool)
            .await?;
            rows.into_iter()
                .map(|row| {
                    Ok(ViewWithSchema {
                        schema_name: row.try_get(0)?,
                        view: ViewMetadata {
                            view_id: row.try_get(1)?,
                            schema_id: row.try_get(2)?,
                            begin_snapshot: row.try_get(3)?,
                            view_name: row.try_get(4)?,
                            dialect: row.try_get(5)?,
                            sql: row.try_get(6)?,
                            column_aliases: row.try_get(7)?,
                        },
                    })
                })
                .collect()
        })
    }

    fn list_all_columns(&self, snapshot_id: i64) -> Result<Vec<ColumnWithTable>> {
        block_on(async {
            let rows = sqlx::query(
                "SELECT s.schema_name, t.table_name, c.column_id, c.column_name, c.column_type,
                        c.nulls_allowed, c.parent_column, c.initial_default, c.default_value,
                        c.default_value_type, c.default_value_dialect
                 FROM ducklake_schema s
                 JOIN ducklake_table t ON s.schema_id = t.schema_id
                 JOIN ducklake_column c ON t.table_id = c.table_id
                 WHERE $1 >= s.begin_snapshot
                   AND ($2 < s.end_snapshot OR s.end_snapshot IS NULL)
                   AND $3 >= t.begin_snapshot
                   AND ($4 < t.end_snapshot OR t.end_snapshot IS NULL)
                   AND $5 >= c.begin_snapshot
                   AND ($6 < c.end_snapshot OR c.end_snapshot IS NULL)
                 ORDER BY s.schema_name, t.table_name, c.column_order",
            )
            .bind(snapshot_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .fetch_all(&self.pool)
            .await?;

            let raw: Result<Vec<(ColumnWithTable, Option<i64>)>> = rows
                .into_iter()
                .map(|row| {
                    let schema_name: String = row.try_get(0)?;
                    let table_name: String = row.try_get(1)?;
                    let nulls_allowed: Option<bool> = row.try_get(5)?;
                    let parent_column: Option<i64> = row.try_get(6)?;
                    let column = DuckLakeTableColumn::new(
                        row.try_get(2)?,
                        row.try_get(3)?,
                        row.try_get(4)?,
                        nulls_allowed.unwrap_or(true),
                    )
                    .with_defaults(
                        row.try_get(7)?,
                        row.try_get(8)?,
                        row.try_get(9)?,
                        row.try_get(10)?,
                    );
                    Ok((
                        ColumnWithTable {
                            schema_name,
                            table_name,
                            column,
                        },
                        parent_column,
                    ))
                })
                .collect();
            reconstruct_columns_with_table(raw?)
        })
    }

    fn list_all_files(&self, snapshot_id: i64) -> Result<Vec<FileWithTable>> {
        block_on(async {
            let rows = sqlx::query(
                "SELECT
                    s.schema_name,
                    t.table_name,
                    data.data_file_id,
                    data.path AS data_file_path,
                    data.path_is_relative AS data_path_is_relative,
                    data.file_size_bytes AS data_file_size,
                    data.footer_size AS data_footer_size,
                    data.encryption_key AS data_encryption_key,
                    del.delete_file_id,
                    del.path AS delete_file_path,
                    del.path_is_relative AS delete_path_is_relative,
                    del.file_size_bytes AS delete_file_size,
                    del.footer_size AS delete_footer_size,
                    del.encryption_key AS delete_encryption_key,
                    del.delete_count
                FROM ducklake_schema s
                JOIN ducklake_table t ON s.schema_id = t.schema_id
                JOIN ducklake_data_file data ON t.table_id = data.table_id
                LEFT JOIN ducklake_delete_file del
                    ON data.data_file_id = del.data_file_id
                    AND del.table_id = t.table_id
                    AND $1 >= del.begin_snapshot
                    AND ($2 < del.end_snapshot OR del.end_snapshot IS NULL)
                WHERE $3 >= s.begin_snapshot
                  AND ($4 < s.end_snapshot OR s.end_snapshot IS NULL)
                  AND $5 >= t.begin_snapshot
                  AND ($6 < t.end_snapshot OR t.end_snapshot IS NULL)
                  AND $7 >= data.begin_snapshot
                  AND ($8 < data.end_snapshot OR data.end_snapshot IS NULL)
                ORDER BY s.schema_name, t.table_name, data.path",
            )
            .bind(snapshot_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .fetch_all(&self.pool)
            .await?;

            rows.into_iter()
                .map(|row| {
                    let data_file = DuckLakeFileData {
                        path: row.try_get(3)?,
                        path_is_relative: row.try_get(4)?,
                        file_size_bytes: row.try_get(5)?,
                        footer_size: row.try_get(6)?,
                        encryption_key: row.try_get(7)?,
                        mapping_id: None,
                    };

                    let delete_file = if row.try_get::<Option<i64>, _>(8)?.is_some() {
                        Some(DuckLakeFileData {
                            path: row.try_get(9)?,
                            path_is_relative: row.try_get(10)?,
                            file_size_bytes: row.try_get(11)?,
                            footer_size: row.try_get(12)?,
                            encryption_key: row.try_get(13)?,
                            mapping_id: None,
                        })
                    } else {
                        None
                    };

                    Ok(FileWithTable {
                        schema_name: row.try_get(0)?,
                        table_name: row.try_get(1)?,
                        file: DuckLakeTableFile {
                            data_file_id: row.try_get(2)?,
                            file: data_file,
                            delete_file_id: row.try_get(8)?,
                            delete_file,
                            row_id_start: None,
                            snapshot_id: None,
                            begin_snapshot: None,
                            schema_version: None,
                            partial_max: None,
                            max_row_count: row.try_get(14)?,
                            delete_count: None,
                            partition_id: None,
                            partition_values: Vec::new(),
                        },
                    })
                })
                .collect()
        })
    }

    fn get_data_files_added_between_snapshots(
        &self,
        table_id: i64,
        start_snapshot: i64,
        end_snapshot: i64,
    ) -> Result<Vec<DataFileChange>> {
        block_on(async {
            // Older catalogs predate `partial_max`; degrade it to NULL there
            // (they cannot contain partial files), matching the probe pattern
            // used by the scan queries above.
            let pm = if self.schema_capabilities().await?.data_file_partial_max {
                "data.partial_max::bigint"
            } else {
                "NULL::bigint"
            };
            let rows = sqlx::query(AssertSqlSafe(format!(
                "SELECT
                    data.begin_snapshot,
                    data.path,
                    data.path_is_relative,
                    data.file_size_bytes,
                    data.footer_size,
                    data.encryption_key,
                    data.row_id_start,
                    {pm},
                    data.mapping_id
                FROM ducklake_data_file AS data
                WHERE data.table_id = $1
                  AND data.begin_snapshot <= $3
                  AND (data.begin_snapshot >= $2
                       OR ({pm} IS NOT NULL AND {pm} >= $2))
                ORDER BY data.begin_snapshot"
            )))
            .bind(table_id)
            .bind(start_snapshot)
            .bind(end_snapshot)
            .fetch_all(&self.pool)
            .await?;

            rows.into_iter()
                .map(|row| {
                    Ok(DataFileChange {
                        begin_snapshot: row.try_get(0)?,
                        path: row.try_get(1)?,
                        path_is_relative: row.try_get(2)?,
                        file_size_bytes: row.try_get(3)?,
                        footer_size: row.try_get(4)?,
                        encryption_key: row.try_get(5)?,
                        row_id_start: row.try_get(6)?,
                        partial_max: row.try_get(7)?,
                        mapping_id: row.try_get(8)?,
                    })
                })
                .collect()
        })
    }

    fn get_delete_files_added_between_snapshots(
        &self,
        table_id: i64,
        start_snapshot: i64,
        end_snapshot: i64,
    ) -> Result<Vec<DeleteFileChange>> {
        block_on(async {
            // PostgreSQL equivalent of DuckDB's SQL_GET_DELETE_FILES_ADDED_BETWEEN_SNAPSHOTS
            // Uses LATERAL joins instead of MAX_BY/COLUMNS
            // Cumulative (current-spec) delete files can hold in-window deletions
            // even when their begin_snapshot predates the window; included via
            // `ducklake_delete_file.partial_max`. Older catalogs lack the column
            // (and cumulative delete files); degrade it to NULL there.
            let pm = if self.schema_capabilities().await?.delete_file_partial_max {
                "ddf.partial_max::bigint"
            } else {
                "NULL::bigint"
            };
            let rows = sqlx::query(AssertSqlSafe(format!(
                r#"
WITH current_delete AS (
    SELECT
        ddf.data_file_id,
        ddf.begin_snapshot,
        ddf.path,
        ddf.path_is_relative,
        ddf.file_size_bytes,
        ddf.footer_size,
        ddf.encryption_key
    FROM ducklake_delete_file ddf
    WHERE ddf.table_id = $1
      AND ddf.begin_snapshot <= $3
      AND (ddf.begin_snapshot >= $2
           OR ({pm} IS NOT NULL AND {pm} >= $2))
),

data_files AS (
    SELECT df.*
    FROM ducklake_data_file df
    WHERE df.table_id = $1
)

-- Part 1: Incremental deletes
SELECT
    data.path,
    data.path_is_relative,
    data.file_size_bytes,
    data.footer_size,
    data.row_id_start,
    data.record_count,
    data.mapping_id,
    current_delete.path,
    current_delete.path_is_relative,
    current_delete.file_size_bytes,
    current_delete.footer_size,
    prev.path,
    prev.path_is_relative,
    prev.file_size_bytes,
    prev.footer_size,
    current_delete.begin_snapshot
FROM current_delete
JOIN data_files data USING (data_file_id)
LEFT JOIN LATERAL (
    SELECT
        ddf.path,
        ddf.path_is_relative,
        ddf.file_size_bytes,
        ddf.footer_size
    FROM ducklake_delete_file ddf
    WHERE ddf.table_id = $1
      AND ddf.data_file_id = current_delete.data_file_id
      AND ddf.begin_snapshot < current_delete.begin_snapshot
    ORDER BY ddf.begin_snapshot DESC
    LIMIT 1
) prev ON true

UNION ALL

-- Part 2: Full file deletes
SELECT
    data.path,
    data.path_is_relative,
    data.file_size_bytes,
    data.footer_size,
    data.row_id_start,
    data.record_count,
    data.mapping_id,
    NULL::VARCHAR,
    NULL::BOOLEAN,
    NULL::BIGINT,
    NULL::BIGINT,
    prev.path,
    prev.path_is_relative,
    prev.file_size_bytes,
    prev.footer_size,
    data.end_snapshot
FROM ducklake_data_file data
LEFT JOIN LATERAL (
    SELECT
        ddf.path,
        ddf.path_is_relative,
        ddf.file_size_bytes,
        ddf.footer_size
    FROM ducklake_delete_file ddf
    WHERE ddf.table_id = $1
      AND ddf.data_file_id = data.data_file_id
      AND ddf.begin_snapshot < data.end_snapshot
    ORDER BY ddf.begin_snapshot DESC
    LIMIT 1
) prev ON true
WHERE data.table_id = $1
  AND data.end_snapshot >= $2
  AND data.end_snapshot <= $3
"#
            )))
            .bind(table_id)
            .bind(start_snapshot)
            .bind(end_snapshot)
            .fetch_all(&self.pool)
            .await?;

            rows.into_iter()
                .map(|row| {
                    Ok(DeleteFileChange {
                        // data file
                        data_file_path: row.try_get(0)?,
                        data_file_path_is_relative: row.try_get(1)?,
                        data_file_size_bytes: row.try_get(2)?,
                        data_file_footer_size: row.try_get(3)?,
                        data_row_id_start: row.try_get(4)?,
                        data_record_count: row.try_get(5)?,
                        data_mapping_id: row.try_get(6)?,

                        // current delete
                        current_delete_path: row.try_get(7)?,
                        current_delete_path_is_relative: row.try_get(8)?,
                        current_delete_file_size_bytes: row.try_get(9)?,
                        current_delete_footer_size: row.try_get(10)?,

                        // previous delete
                        previous_delete_path: row.try_get(11)?,
                        previous_delete_path_is_relative: row.try_get(12)?,
                        previous_delete_file_size_bytes: row.try_get(13)?,
                        previous_delete_footer_size: row.try_get(14)?,

                        // snapshot
                        snapshot_id: row.try_get(15)?,
                    })
                })
                .collect()
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
    use datafusion::common::ScalarValue;
    use datafusion::logical_expr::Operator;
    use datafusion::physical_expr::PhysicalExpr;
    use datafusion::physical_expr::expressions::{BinaryExpr, Column, lit};

    use super::{PostgresStatsDialect, StatsFilterSql, stats_filter_sql};
    use crate::metadata_provider::DuckLakeTableColumn;
    use crate::stats_filter::lower_predicate;

    fn column(name: &str, data_type: DataType, column_id: i64) -> (Field, DuckLakeTableColumn) {
        (
            Field::new(name, data_type, true),
            DuckLakeTableColumn {
                column_id,
                column_name: name.to_string(),
                column_type: "int32".to_string(),
                is_nullable: true,
                data_type: None,
                nested_column_ids: Vec::new(),
                initial_default: None,
                default_value: None,
                default_value_type: None,
                default_value_dialect: None,
            },
        )
    }

    /// Lower `predicate` over one column and splice it, or `None` when nothing
    /// pushes down for this server.
    fn splice(
        predicate: Arc<dyn PhysicalExpr>,
        field: (Field, DuckLakeTableColumn),
        table_id: i64,
        soft_input_validation: bool,
    ) -> Option<StatsFilterSql> {
        let schema = Schema::new(vec![field.0]);
        let rendered = lower_predicate(&predicate, &schema, &[field.1])
            .expect("predicate lowers")
            .render(&PostgresStatsDialect {
                soft_input_validation,
            })?;
        stats_filter_sql(table_id, &rendered)
    }

    fn int_range_predicate() -> Arc<dyn PhysicalExpr> {
        // a > 5 AND a < 10
        let a = Arc::new(Column::new("a", 0)) as Arc<dyn PhysicalExpr>;
        Arc::new(BinaryExpr::new(
            Arc::new(BinaryExpr::new(Arc::clone(&a), Operator::Gt, lit(5i32))),
            Operator::And,
            Arc::new(BinaryExpr::new(a, Operator::Lt, lit(10i32))),
        ))
    }

    /// The shape official assembles — one CTE selecting only the stats the
    /// condition reads, one LEFT JOIN, the condition ANDed onto the existing
    /// WHERE — with `pg_input_is_valid` standing in for `TRY_CAST`.
    #[test]
    fn two_conjunct_integer_filter_renders_the_official_shape() {
        let spliced = splice(
            int_range_predicate(),
            column("a", DataType::Int32, 7),
            3,
            true,
        )
        .expect("filter splices");
        assert_eq!(
            spliced.with_prefix,
            "WITH col_7_stats AS (
                     SELECT data_file_id, min_value, max_value, value_count
                     FROM ducklake_file_column_stats
                     WHERE column_id = 7 AND table_id = 3
                 )
                 "
        );
        assert_eq!(
            spliced.joins,
            "\n                 LEFT JOIN col_7_stats ON col_7_stats.data_file_id = data.data_file_id"
        );
        assert_eq!(
            spliced.conditions,
            "\n                   AND ((col_7_stats.data_file_id IS NULL OR \
             ((col_7_stats.value_count IS NULL OR col_7_stats.value_count > 0) AND \
             (col_7_stats.min_value IS NULL OR col_7_stats.max_value IS NULL OR \
             (CASE WHEN (col_7_stats.max_value COLLATE \"C\") ~ \
             '^-?[0-9]{1,255}(\\.[0-9]{1,255})?$' AND \
             pg_input_is_valid(col_7_stats.max_value, 'numeric') \
             THEN CAST(col_7_stats.max_value AS numeric) END > 5) AND \
             (CASE WHEN (col_7_stats.min_value COLLATE \"C\") ~ \
             '^-?[0-9]{1,255}(\\.[0-9]{1,255})?$' AND \
             pg_input_is_valid(col_7_stats.min_value, 'numeric') \
             THEN CAST(col_7_stats.min_value AS numeric) END < 10))))) IS NOT FALSE"
        );
    }

    /// A server without `pg_input_is_valid` validates by pattern instead, and
    /// still pushes the same comparison down.
    #[test]
    fn integer_filter_falls_back_to_a_pattern_before_postgresql_16() {
        let spliced = splice(
            int_range_predicate(),
            column("a", DataType::Int32, 7),
            3,
            false,
        )
        .expect("filter splices");
        assert!(
            spliced.conditions.contains(
                "CASE WHEN (col_7_stats.max_value COLLATE \"C\") ~ \
                 '^-?[0-9]{1,255}(\\.[0-9]{1,255})?$' \
                 THEN CAST(col_7_stats.max_value AS numeric) END > 5"
            ),
            "unexpected condition:\n{}",
            spliced.conditions
        );
        assert!(!spliced.conditions.contains("pg_input_is_valid"));
    }

    /// Soft validation never stands alone: the stat's shape is required too.
    ///
    /// `pg_input_is_valid` answers "can the input function read this", and that
    /// function is permissive by design — it accepts `today`, `now`, `epoch` and
    /// `infinity` for a date, `NaN` for a numeric, and `nan` for a float. Each
    /// casts to a *value* that then prunes files, and a stat of `today` would
    /// make pruning depend on the wall clock. So a shape pattern gates every
    /// comparison on every server version, and soft validation only adds the
    /// calendar check on top.
    #[test]
    fn soft_validation_still_requires_the_stat_shape() {
        let float_filter = |soft| {
            let f = Arc::new(Column::new("a", 0)) as Arc<dyn PhysicalExpr>;
            let predicate =
                Arc::new(BinaryExpr::new(f, Operator::Eq, lit(5.0f64))) as Arc<dyn PhysicalExpr>;
            splice(predicate, column("a", DataType::Float64, 2), 3, soft)
                .expect("a float equality pushes down")
                .conditions
        };

        for soft in [false, true] {
            let sql = float_filter(soft);
            // The shape pattern is present whether or not the server can soft-validate.
            assert!(
                sql.contains("COLLATE \"C\") ~ '^-?(inf|"),
                "no shape gate (soft_input_validation = {soft}): {sql}"
            );
            // `inf` is admitted, `nan` is not: `nan` parses on PostgreSQL 16 and
            // comparing against it prunes a file whose other rows can match.
            assert!(!sql.contains("nan"), "nan admitted: {sql}");
        }
    }

    /// A temporal comparison happens in the text domain, on every server
    /// version and at every precision.
    ///
    /// This is where PostgreSQL stops needing `pg_input_is_valid`: nothing is
    /// parsed, so no calendar has to be decided and no fraction is rounded away.
    /// Both are real gains — a pre-16 server pruned nothing at all on a date
    /// column, and a nanosecond column pruned nothing on any version because
    /// `timestamp` holds microseconds and rounds.
    #[test]
    fn temporal_is_compared_as_text_on_every_server_version() {
        let render = |value: ScalarValue, data_type: DataType, soft| {
            let t = Arc::new(Column::new("a", 0)) as Arc<dyn PhysicalExpr>;
            let predicate =
                Arc::new(BinaryExpr::new(t, Operator::Lt, lit(value))) as Arc<dyn PhysicalExpr>;
            splice(predicate, column("a", data_type, 2), 3, soft).map(|sql| sql.conditions)
        };

        for soft in [false, true] {
            let date = render(ScalarValue::Date32(Some(19_723)), DataType::Date32, soft)
                .expect("a date pushes down");
            assert!(
                date.contains(
                    "(col_2_stats.min_value COLLATE \"C\") ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}$'"
                ),
                "date not compared as text (soft = {soft}): {date}"
            );
            // No cast, and therefore no dependence on the input function.
            assert!(!date.contains("CAST("), "date was cast: {date}");
            assert!(!date.contains("pg_input_is_valid"), "date parsed: {date}");

            // Nanosecond precision, which a microsecond cast could not order.
            let nanos = render(
                ScalarValue::TimestampNanosecond(Some(1_577_836_800_123_456_700), None),
                DataType::Timestamp(TimeUnit::Nanosecond, None),
                soft,
            )
            .expect("a nanosecond timestamp pushes down");
            assert!(nanos.contains("'2020-01-01 00:00:00.1234567'"), "{nanos}");
            assert!(!nanos.contains("CAST("), "nanosecond was cast: {nanos}");

            // Zoned, carrying the `+00` suffix on both sides.
            let zoned = render(
                ScalarValue::TimestampMicrosecond(
                    Some(1_577_836_800_000_000),
                    Some("+00:00".into()),
                ),
                DataType::Timestamp(TimeUnit::Microsecond, Some("+00:00".into())),
                soft,
            )
            .expect("a zoned timestamp pushes down");
            assert!(zoned.contains("[+]00$"), "{zoned}");
        }
    }

    /// An encoding whose bytes do not order chronologically is declined.
    ///
    /// Text comparison is only sound for the one shape `stats_encode` writes.
    /// `chrono` renders a year past 9999 as `+12345` and one before the common
    /// era as `-0044`, and both sort below every digit; `.50` sorts above `.5`
    /// while naming the same instant; and `12:00:00+01` sorts above
    /// `12:00:00+00` while being earlier.
    #[test]
    fn temporal_declines_encodings_that_do_not_order_as_text() {
        let declines = |value: ScalarValue, data_type: DataType| {
            let t = Arc::new(Column::new("a", 0)) as Arc<dyn PhysicalExpr>;
            let predicate =
                Arc::new(BinaryExpr::new(t, Operator::Lt, lit(value))) as Arc<dyn PhysicalExpr>;
            splice(predicate, column("a", data_type, 2), 3, true).is_none()
        };

        // 4_000_000 days after the epoch renders `+12921-08-18`; -800_000 gives
        // `-0221-09-04`.
        assert!(declines(
            ScalarValue::Date32(Some(4_000_000)),
            DataType::Date32
        ));
        assert!(declines(
            ScalarValue::Date32(Some(-800_000)),
            DataType::Date32
        ));

        // Year zero orders fine as text, so unlike the cast path this admits it.
        assert!(!declines(
            ScalarValue::Date32(Some(-719_528)),
            DataType::Date32
        ));

        // A constant at another offset cannot be built: `encode_scalar`
        // normalizes to UTC and appends `+00` whatever the zone says. The
        // offset guard therefore protects the stat side, where a foreign
        // catalog may hold `+01` — and that is the emitted pattern's job, which
        // `temporal_is_compared_as_text_on_every_server_version` pins.

        // The stat side is pinned by the emitted pattern, which admits no
        // fraction ending in `0`.
        let t = Arc::new(Column::new("a", 0)) as Arc<dyn PhysicalExpr>;
        let predicate = Arc::new(BinaryExpr::new(
            t,
            Operator::Lt,
            lit(ScalarValue::TimestampMicrosecond(
                Some(1_577_836_800_500_000),
                None,
            )),
        )) as Arc<dyn PhysicalExpr>;
        let sql = splice(
            predicate,
            column("a", DataType::Timestamp(TimeUnit::Microsecond, None), 2),
            3,
            true,
        )
        .expect("a canonical timestamp pushes down")
        .conditions;
        assert!(sql.contains("([.][0-9]*[1-9])?$"), "{sql}");
    }

    /// A string bound is compared raw, forced to the one collation PostgreSQL
    /// defines as byte-wise so it agrees with DataFusion's `Utf8` ordering.
    #[test]
    fn string_bounds_are_compared_under_the_c_collation() {
        let s = Arc::new(Column::new("s", 0)) as Arc<dyn PhysicalExpr>;
        let predicate =
            Arc::new(BinaryExpr::new(s, Operator::Eq, lit("abc"))) as Arc<dyn PhysicalExpr>;
        let spliced =
            splice(predicate, column("s", DataType::Utf8, 1), 3, true).expect("filter splices");
        assert!(
            spliced.conditions.contains(
                "'abc' BETWEEN (col_1_stats.min_value COLLATE \"C\") \
                 AND (col_1_stats.max_value COLLATE \"C\")"
            ),
            "unexpected condition:\n{}",
            spliced.conditions
        );
        assert!(!spliced.conditions.contains("CAST"));
    }
}
