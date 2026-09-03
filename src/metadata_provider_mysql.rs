//! MySQL metadata provider for DuckLake catalogs.

use crate::Result;
use crate::metadata_provider::{
    ColumnWithTable, DataFileChange, DeleteFileChange, DuckLakeFileColumnStatistics,
    DuckLakeFileData, DuckLakeFileMetadata, DuckLakeInlinedDelete, DuckLakeNameMapping,
    DuckLakeNameMappingEntry, DuckLakeStatistics, DuckLakeTableColumn,
    DuckLakeTableColumnStatistics, DuckLakeTableField, DuckLakeTableFile, DuckLakeTableStatistics,
    FileWithTable, InlinedDataBackend, MetadataProvider, SQL_GET_FILE_PARTITION_VALUES,
    SQL_GET_PARTITION_SPEC, SQL_GET_SORT_SPEC, SQL_GET_TABLE_COLUMNS, SchemaMetadata,
    SnapshotMetadata, TableMetadata, TableWithSchema, ViewMetadata, ViewWithSchema, block_on,
    inlined_delete_table_name, inlined_text_projection, is_inlined_data_table,
    parse_inlined_rows_with_present, reconstruct_columns, reconstruct_columns_with_table,
};
use crate::partition::PartitionSpec;
use crate::sort::SortSpec;
use crate::stats_encode::{is_canonical_date, is_canonical_timestamp, is_canonical_timestamptz};
use crate::stats_filter::{StatsFilter, StatsLiteral, StatsSqlDialect};
use arrow::datatypes::{DataType, SchemaRef};
use arrow::record_batch::RecordBatch;
use sqlx::AssertSqlSafe;
use sqlx::Row;
use sqlx::mysql::{MySqlPool, MySqlPoolOptions, MySqlRow};
use sqlx::types::chrono::NaiveDateTime;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, OnceLock};

fn is_missing_statistics_table(error: &sqlx::Error) -> bool {
    let message = error.to_string().to_ascii_lowercase();
    message.contains("doesn't exist")
        || message.contains("does not exist")
        || message.contains("unknown table")
}

fn quote_ident(name: &str) -> String {
    format!("`{}`", name.replace('`', "``"))
}

/// MySQL spelling of the statistics comparisons in [`crate::stats_filter`].
///
/// Two engine facts shape everything here.
///
/// MySQL has no `TRY_CAST`, and `CAST` to a number does not fail: `CAST('abc' AS
/// DOUBLE)` and `CAST('abc' AS DECIMAL)` are `0` with a warning, so casting a
/// malformed bound unconditionally would compare it as zero and could prune a
/// file that matches. Every numeric cast below is therefore gated on a `REGEXP`
/// that the text really is a number of that shape, and yields SQL `NULL`
/// otherwise; [`StatsSqlDialect::keep_when_unknown`] turns that `NULL` back into
/// "keep the file". The temporal casts need no gate — MySQL already returns
/// `NULL` for a datetime it cannot parse.
///
/// MySQL's default collation is `utf8mb4_0900_ai_ci`, which is case- *and*
/// accent-insensitive: with it, `'Apple' = 'apple'` and `'cafe' = 'café'` are
/// both true. DataFusion compares `Utf8` byte-wise, so a raw string bound
/// compared under that collation can place a value inside a range the engine
/// puts outside it, and drop a file that matches. [`Self::collate_binary`]
/// forces byte-wise comparison on every uncast string comparison.
struct MySqlStatsDialect;

impl StatsSqlDialect for MySqlStatsDialect {
    /// A type not listed here is declined, which drops the comparison and
    /// prunes nothing: `TIMESTAMP` in nanoseconds, because `DATETIME` holds
    /// microseconds and MySQL *rounds* the extra digits (`.123456789` becomes
    /// `.123457`), which can move a bound outward past the constant; a
    /// timezone-bearing `TIMESTAMP`, because `stats_encode` writes it with a
    /// `+00` suffix that MySQL's datetime parser rejects; and `DECIMAL` with a
    /// scale past 30, MySQL's maximum.
    ///
    /// A temporal constant is inspected as well as its type. MySQL converts the
    /// constant to the stat's type to compare them, and one it cannot convert is
    /// an *error* — `CAST('2024-01-01' AS DATE) <= '12345-01-01'` raises 1525 —
    /// not a NULL. That would cost the whole listing its filter (the unfiltered
    /// retry catches it), so a constant outside the canonical four-digit-year
    /// encoding is declined here instead.
    fn try_cast(&self, expr: &str, literal: &StatsLiteral, data_type: &DataType) -> Option<String> {
        match data_type {
            // Every integer bound is read as DECIMAL(65, 0) rather than SIGNED:
            // `CAST(... AS SIGNED)` saturates at `i64::MAX`, which would read a
            // `u64` bound above it as a smaller number, while DECIMAL(65, 0)
            // holds the whole unsigned range exactly and compares exactly
            // against the constant. The 20-digit cap is what keeps a longer
            // string from saturating DECIMAL(65, 0) in turn.
            DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64 => Some(mysql_guarded_cast(
                expr,
                "^-?[0-9]{1,20}$",
                "DECIMAL(65, 0)",
            )),
            // `stats_encode` writes a finite double as its shortest round-trip
            // decimal, in fixed or exponent form, so parsing it back as DOUBLE
            // recovers the exact value the writer had. The pattern also rejects
            // the `inf` / `-inf` bounds the encoder writes for infinities, which
            // MySQL would otherwise read as `0`.
            //
            // Both digit runs are bounded, and that bound is load-bearing. A
            // magnitude MySQL cannot represent does not fail the cast — it
            // *saturates*, silently: on 8.x `CAST('1e+400' AS DOUBLE)` is
            // `1.7976931348623157e308`, `CAST('1e-400' AS DOUBLE)` is `0`, and
            // 400 nines is DBL_MAX again. Saturation moves a bound inward or
            // outward without a warning the guard can see, which is the
            // "coerces it to a number" failure this dialect exists to prevent.
            // The bounds mirror the PostgreSQL dialect's: fixed point to ±1e255,
            // and scientific to one leading digit with a two-digit exponent, so
            // roughly 1e100 — both far inside the ±1.8e308 a DOUBLE holds, and
            // far outside anything a real bound of a float column is. A larger
            // magnitude is declined rather than read, which costs pruning only.
            DataType::Float32 | DataType::Float64 => Some(mysql_guarded_cast(
                expr,
                "^-?([0-9]{1,255}([.][0-9]{1,255})?|[0-9]([.][0-9]{1,255})?e[-+][0-9]{1,2})$",
                "DOUBLE",
            )),
            DataType::Decimal128(precision, scale) => {
                let scale = usize::try_from(*scale).ok().filter(|scale| *scale <= 30)?;
                // A DECIMAL(p, s) value has at most p - s integer digits, and
                // `encode_decimal128` always writes at least one ("0.5"). More
                // than that is not a value of the constant's type, and more
                // fractional digits than the scale would be *rounded* by the
                // cast, which can move a bound past the constant.
                let integer_digits = usize::from(*precision).saturating_sub(scale).max(1);
                let pattern = if scale == 0 {
                    format!("^-?[0-9]{{1,{integer_digits}}}$")
                } else {
                    format!("^-?[0-9]{{1,{integer_digits}}}([.][0-9]{{1,{scale}}})?$")
                };
                Some(mysql_guarded_cast(
                    expr,
                    &pattern,
                    &format!("DECIMAL(65, {scale})"),
                ))
            },
            // Shape first, then the cast — both are load-bearing, and neither
            // covers the other. MySQL's `DATE` parser is lenient in a way that
            // turns a stat no writer of ours produces into a definite bound:
            // `' 2020-01-01 '`, `2020/01/01`, `20200101`, `2020.01.01` and
            // `2020-1-1` all convert to 2020-01-01, so a catalog carrying one
            // would prune on a bound this crate never wrote. The shape test
            // refuses them. The cast then refuses what a shape cannot judge —
            // `2020-02-31` is `NNNN-NN-NN` and no calendar has it, and MySQL
            // reads it as NULL.
            //
            // The constant is checked in Rust for the same reason it is
            // everywhere else: one outside MySQL's year range makes the
            // comparison's implicit conversion an *error* rather than a NULL,
            // and only the unfiltered retry in
            // `get_table_file_metadata_page_filtered` would keep that from
            // failing the listing.
            DataType::Date32 if is_canonical_date(literal.text()) => {
                let guarded = self.collate_binary(expr);
                Some(format!(
                    "CAST(CASE WHEN {} THEN {guarded} END AS DATE)",
                    mysql_matches_whole(&guarded, MYSQL_CANONICAL_DATE)
                ))
            },
            // Every timestamp is compared as text rather than cast, at every
            // precision. `DATETIME(6)` holds microseconds and *rounds* a longer
            // fraction, which is monotonic but not injective: two distinct
            // nanosecond instants can land on one microsecond, so a strict
            // comparison that holds of the stored values comes back false.
            // Guarding only the constant does not help, because it is the
            // *stat* that gets rounded — `CAST('…00.1234566' AS DATETIME(6))`
            // is `…123457`, which is not less than `…123457`.
            //
            // The canonical encoding is chronologically ordered byte-wise, so
            // comparing the two strings answers the same question exactly and
            // at full precision. Byte-collated so the server's case-insensitive
            // default cannot equate two different instants.
            DataType::Timestamp(_, None) if is_canonical_timestamp(literal.text()) => {
                let text = self.collate_binary(expr);
                Some(mysql_guarded_text(&text, MYSQL_CANONICAL_TIMESTAMP))
            },
            // The same, plus the `+00` suffix `stats_encode` appends after
            // normalizing to UTC. No temporal type parses it — `CAST('… +00' AS
            // DATETIME)` is NULL — and it is constant across everything the
            // guard admits, so it does not disturb the ordering.
            DataType::Timestamp(_, Some(_)) if is_canonical_timestamptz(literal.text()) => {
                let text = self.collate_binary(expr);
                Some(mysql_guarded_text(&text, MYSQL_CANONICAL_TIMESTAMPTZ))
            },
            // A boolean bound is the text `true` / `false` and the constant is
            // rendered quoted, so comparing them as text is well-defined and
            // correctly ordered ('false' < 'true'). The guard is applied to the
            // binary-collated form so that a catalog holding some other
            // spelling — `TRUE`, which the default collation would accept here —
            // is declined rather than mis-compared.
            DataType::Boolean => {
                let text = self.collate_binary(expr);
                Some(format!(
                    "CASE WHEN {text} IN ('true', 'false') THEN {text} END"
                ))
            },
            _ => None,
        }
    }

    /// `utf8mb4_0900_bin`, not `utf8mb4_bin`: the latter is a PAD SPACE
    /// collation, so it compares `'a'` and `'a '` as equal and orders neither
    /// below the other. DataFusion compares `Utf8` byte-wise, where `'a'` is
    /// less than `'a '` — so under the padded collation a file whose bound is
    /// `'a'` is pruned from `WHERE s < 'a '` though it holds a matching row.
    /// `utf8mb4_0900_bin` is NO PAD and byte-wise. A server without it raises,
    /// and the unfiltered retry lists every file rather than pruning wrongly.
    ///
    /// `CONVERT ... USING utf8mb4` before the collation, not a bare
    /// `COLLATE`: naming a collation from another character set is
    /// error 1253, which would fail the whole listing query on a catalog whose
    /// `min_value` / `max_value` are, say, `latin1`. Transcoding first also
    /// makes the comparison byte-wise over *UTF-8* bytes, which is the order
    /// DataFusion compares `Utf8` in.
    fn collate_binary(&self, expr: &str) -> String {
        format!("CONVERT({expr} USING utf8mb4) COLLATE utf8mb4_0900_bin")
    }

    /// `contains_nan` is a `BOOLEAN` column, which MySQL stores as `TINYINT(1)`,
    /// so the `FALSE` keyword compares against it correctly.
    fn boolean_is_not_false(&self, expr: &str) -> String {
        format!("{expr} IS NULL OR {expr} <> FALSE")
    }

    /// MySQL gives `\` a second meaning inside a quoted string: unless the
    /// server's `sql_mode` carries `NO_BACKSLASH_ESCAPES` it opens an escape
    /// sequence. `stats_encode` passes `Utf8` through verbatim, so a value
    /// holding a backslash reaches this text, and the standard rendering of a
    /// constant `a\` is `'a\'` — whose closing quote the server reads as
    /// escaped, taking the rest of the statement with it (error 1064).
    ///
    /// Doubling the backslash repairs that under the default mode only. Under
    /// `NO_BACKSLASH_ESCAPES`, which the `ORACLE` mode sets, `'a\\'` is the
    /// two-character string `a\\`; comparing it against a file whose bounds are
    /// both `a\` is false, and the file is pruned though it matches. That is the
    /// one outcome this module never accepts, so backslash-bearing text is
    /// rendered as a hexadecimal literal instead, which has a single meaning in
    /// both modes. The `_utf8mb4` introducer keeps it a character string rather
    /// than a binary one; the explicit `COLLATE` on the stat side of every raw
    /// string comparison has the stronger coercibility and still decides the
    /// collation.
    ///
    /// Text with no backslash — every temporal and boolean constant, and nearly
    /// every string one — takes the ordinary quoted form, quotes doubled.
    fn quote_literal(&self, text: &str) -> String {
        if !text.contains('\\') {
            return format!("'{}'", text.replace('\'', "''"));
        }
        let hex: String = text.bytes().map(|byte| format!("{byte:02X}")).collect();
        format!("_utf8mb4 X'{hex}'")
    }
}

/// The date shape [`crate::stats_encode`] writes, as a MySQL regular expression
/// — unanchored at the end; see [`mysql_matches_whole`].
const MYSQL_CANONICAL_DATE: &str = r"^[0-9]{4}-[0-9]{2}-[0-9]{2}";

/// The naive timestamp shape [`crate::stats_encode`] writes, as a MySQL regular
/// expression — deliberately *unanchored at the end*; see
/// [`mysql_matches_whole`].
///
/// A fraction must not end in `0`: as text `.50` sorts above `.5` while naming
/// the same instant, so admitting both would misorder them.
const MYSQL_CANONICAL_TIMESTAMP: &str =
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}([.][0-9]*[1-9])?";

/// The same, plus the `+00` suffix. Another offset is refused rather than
/// compared: `12:00:00+01` sorts above `12:00:00+00` and names an earlier
/// instant.
const MYSQL_CANONICAL_TIMESTAMPTZ: &str =
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}([.][0-9]*[1-9])?[+]00";

/// A collated stat, or SQL `NULL` when it is not the encoding this comparison
/// expects.
fn mysql_guarded_text(collated: &str, pattern: &str) -> String {
    format!(
        "CASE WHEN {} THEN {collated} END",
        mysql_matches_whole(collated, pattern)
    )
}

/// Whether `expr` matches `pattern` over its *whole* value.
///
/// `$` will not do it. MySQL 8's `REGEXP` is ICU, where `$` matches before a
/// final line terminator, so a pattern ending in `$` also admits a stat with a
/// trailing newline, carriage return, U+0085, U+2028 or U+2029. That is not
/// cosmetic: U+2028's UTF-8 lead byte is `0xE2` and U+0085's is `0xC2`, both of
/// which sort *above* `.` (0x2E), so such a stat compares as though it were
/// later than any fractional timestamp and a file holding matching rows is
/// pruned.
///
/// `\z` is the usual answer and is unusable here — under `NO_BACKSLASH_ESCAPES`
/// the server reads the pattern literally, `\z` matches nothing, and every
/// temporal comparison would silently stop pruning. Comparing the value against
/// its own leading match needs no escape and means the same thing in both modes.
fn mysql_matches_whole(expr: &str, pattern: &str) -> String {
    format!("{expr} = REGEXP_SUBSTR({expr}, '{pattern}')")
}

/// `CAST(<expr> AS <sql_type>)` for text matching `pattern`, SQL `NULL`
/// otherwise.
///
/// `pattern` is a MySQL regular expression anchored at both ends. It is written
/// with `[.]` rather than `\.` deliberately: a backslash in MySQL's `REGEXP`
/// argument is a string escape as well as a regex one, and doubling it once more
/// through `format!` is a well-known way to end up matching any character.
fn mysql_guarded_cast(expr: &str, pattern: &str, sql_type: &str) -> String {
    format!("CASE WHEN {expr} REGEXP '{pattern}' THEN CAST({expr} AS {sql_type}) END")
}

/// The SQL a lowered statistics filter contributes to the file-listing query.
struct StatsFilterSql {
    /// `WITH col_<id>_stats AS (...), ...`, newline-terminated so it can be
    /// prefixed straight onto the `SELECT`.
    cte: String,
    /// One `LEFT JOIN` per column CTE.
    joins: String,
    /// The rendered per-column conditions, each already prefixed with `AND`.
    conditions: String,
}

/// Render `filter` for MySQL, or `None` when it contributes nothing.
///
/// Adds no bind parameters: [`crate::stats_filter`] inlines every literal, and
/// the only other values spliced in are `i64`s this process computed. The
/// caller's parameter list and its order are therefore untouched.
fn stats_filter_sql(filter: Option<&StatsFilter>, table_id: i64) -> Option<StatsFilterSql> {
    let rendered = filter?.render(&MySqlStatsDialect)?;
    let mut cte = String::from("WITH ");
    let mut joins = String::new();
    let mut conditions = String::new();
    for (index, column) in rendered.iter().enumerate() {
        if index > 0 {
            cte.push_str(",\n     ");
        }
        let alias = &column.alias;
        let stats = column.stats.join(", ");
        let column_id = column.column_id;
        cte.push_str(&format!(
            "{alias} AS (SELECT data_file_id, {stats}
                        FROM ducklake_file_column_stats
                        WHERE column_id = {column_id} AND table_id = {table_id})"
        ));
        joins.push_str(&format!(
            "
                 LEFT JOIN {alias} ON {alias}.data_file_id = data.data_file_id"
        ));
        // The condition arrives already wrapped in its no-stats, per-stat and
        // unknown guards, so it is spliced verbatim.
        conditions.push_str(&format!(
            "
                   AND {}",
            column.condition
        ));
    }
    cte.push('\n');
    Some(StatsFilterSql {
        cte,
        joins,
        conditions,
    })
}

fn decode_view(row: &MySqlRow) -> Result<ViewMetadata> {
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

fn decode_table_file(row: &MySqlRow, snapshot_id: i64) -> Result<DuckLakeTableFile> {
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
            mapping_id: row.try_get(15).unwrap_or(None),
        },
        delete_file_id,
        delete_file,
        row_id_start: row.try_get(6)?,
        snapshot_id: Some(snapshot_id),
        begin_snapshot: None,
        schema_version: None,
        partial_max: None,
        max_row_count: row.try_get(7)?,
        delete_count,
        partition_id: None,
        partition_values: Vec::new(),
    })
}

/// Optional catalog-schema capabilities probed before CDC / inlined-data queries.
///
/// Older catalogs may lack the `partial_max` columns and the inlined-data
/// registry. CDC queries degrade the corresponding projections/predicates to
/// NULL, and inlined-data reads return empty when a capability is absent.
#[derive(Debug, Clone, Copy)]
struct SchemaCapabilities {
    /// `ducklake_data_file.partial_max` exists.
    data_file_partial_max: bool,
    /// `ducklake_delete_file.partial_max` exists.
    delete_file_partial_max: bool,
    /// The `ducklake_inlined_data_tables` registry exists.
    inlined_data_tables: bool,
    /// The `ducklake_view` table exists.
    views: bool,
}

impl SchemaCapabilities {
    fn all(&self) -> bool {
        self.data_file_partial_max
            && self.delete_file_partial_max
            && self.inlined_data_tables
            && self.views
    }
}

/// MySQL-based metadata provider for DuckLake catalogs.
#[derive(Debug, Clone)]
pub struct MySqlMetadataProvider {
    pub pool: MySqlPool,
    // Positive-only memo of the optional-schema capability probes. `Arc` so
    // derived `Clone` shares the cache across provider clones.
    schema_capabilities: Arc<OnceLock<SchemaCapabilities>>,
}

impl MySqlMetadataProvider {
    /// Creates a new provider for an existing DuckLake catalog.
    pub async fn new(connection_string: &str) -> Result<Self> {
        let pool = MySqlPoolOptions::new()
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
    pub fn from_pool(pool: MySqlPool) -> Self {
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
        let row: (i64, i64, i64, i64) = sqlx::query_as(
            "SELECT
               (SELECT COUNT(*) FROM information_schema.columns
                WHERE table_schema = DATABASE()
                  AND table_name = 'ducklake_data_file'
                  AND column_name = 'partial_max'),
               (SELECT COUNT(*) FROM information_schema.columns
                WHERE table_schema = DATABASE()
                  AND table_name = 'ducklake_delete_file'
                  AND column_name = 'partial_max'),
               (SELECT COUNT(*) FROM information_schema.tables
                WHERE table_schema = DATABASE()
                  AND table_name = 'ducklake_inlined_data_tables'),
               (SELECT COUNT(*) FROM information_schema.tables
                WHERE table_schema = DATABASE()
                  AND table_name = 'ducklake_view')",
        )
        .fetch_one(&self.pool)
        .await?;
        let caps = SchemaCapabilities {
            data_file_partial_max: row.0 > 0,
            delete_file_partial_max: row.1 > 0,
            inlined_data_tables: row.2 > 0,
            views: row.3 > 0,
        };
        if caps.all() {
            let _ = self.schema_capabilities.set(caps);
        }
        Ok(caps)
    }

    /// Bind and run one page of the file-listing query.
    ///
    /// The parameter list is identical for the filtered and unfiltered
    /// spellings of that query — `stats_filter_sql` inlines everything it needs
    /// — which is what lets the caller retry with the filter dropped.
    async fn fetch_file_page(
        &self,
        sql: &str,
        table_id: i64,
        snapshot_id: i64,
        after_data_file_id: Option<i64>,
        limit: i64,
    ) -> std::result::Result<Vec<MySqlRow>, sqlx::Error> {
        sqlx::query(AssertSqlSafe(sql))
            .bind(table_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .bind(table_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .bind(after_data_file_id.unwrap_or(i64::MIN))
            .bind(limit)
            .fetch_all(&self.pool)
            .await
    }
}

impl MetadataProvider for MySqlMetadataProvider {
    fn get_current_snapshot(&self) -> Result<i64> {
        block_on(async {
            let row = sqlx::query("SELECT COALESCE(MAX(snapshot_id), 0) FROM ducklake_snapshot")
                .fetch_one(&self.pool)
                .await?;
            Ok(row.try_get(0)?)
        })
    }

    fn get_data_path(&self) -> Result<String> {
        block_on(async {
            let row = sqlx::query(
                "SELECT value FROM ducklake_metadata WHERE `key` = ? AND scope IS NULL",
            )
            .bind("data_path")
            .fetch_optional(&self.pool)
            .await?;

            match row {
                Some(r) => Ok(r.try_get(0)?),
                None => Err(crate::error::DuckLakeError::InvalidConfig(
                    "Missing required catalog metadata: 'data_path' not configured. \
                     The catalog may be uninitialized or corrupted."
                        .to_string(),
                )),
            }
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

    fn list_schemas(&self, snapshot_id: i64) -> Result<Vec<SchemaMetadata>> {
        block_on(async {
            let rows = sqlx::query(
                "SELECT schema_id, schema_name, path, path_is_relative FROM ducklake_schema
                 WHERE ? >= begin_snapshot AND (? < end_snapshot OR end_snapshot IS NULL)",
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
                 WHERE schema_id = ?
                   AND ? >= begin_snapshot
                   AND (? < end_snapshot OR end_snapshot IS NULL)",
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
                "SELECT view_id, schema_id, begin_snapshot, view_name, dialect, `sql`, column_aliases
                 FROM ducklake_view
                 WHERE schema_id = ?
                   AND ? >= begin_snapshot
                   AND (? < end_snapshot OR end_snapshot IS NULL)",
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
            let rows = sqlx::query(SQL_GET_TABLE_COLUMNS)
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
                 WHERE table_id = ?
                   AND ? >= begin_snapshot
                   AND (? < end_snapshot OR end_snapshot IS NULL)
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
                 WHERE mapping.mapping_id = ?
                 ORDER BY name.parent_column IS NOT NULL, name.parent_column, name.column_id",
            )
            .bind(mapping_id)
            .fetch_all(&self.pool)
            .await?;
            let first = rows.first().ok_or_else(|| {
                crate::DuckLakeError::InvalidConfig(format!(
                    "DuckLake name mapping {mapping_id} does not exist"
                ))
            })?;
            let mut entries = Vec::new();
            for row in &rows {
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
        })
    }

    fn get_table_files_for_select(
        &self,
        table_id: i64,
        snapshot_id: i64,
    ) -> Result<Vec<DuckLakeTableFile>> {
        block_on(async {
            let rows = sqlx::query(
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
                    data.mapping_id
                FROM ducklake_data_file AS data
                LEFT JOIN ducklake_delete_file AS del
                    ON data.data_file_id = del.data_file_id
                    AND del.table_id = ?
                    AND ? >= del.begin_snapshot
                    AND (? < del.end_snapshot OR del.end_snapshot IS NULL)
                WHERE data.table_id = ?
                  AND ? >= data.begin_snapshot
                  AND (? < data.end_snapshot OR data.end_snapshot IS NULL)",
            )
            .bind(table_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .bind(table_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .fetch_all(&self.pool)
            .await?;

            rows.iter()
                .map(|row| decode_table_file(row, snapshot_id))
                .collect()
        })
    }

    fn get_partition_spec(&self, table_id: i64, snapshot_id: i64) -> Result<Option<PartitionSpec>> {
        block_on(async {
            // Pruning is only safe with exactly one spec generation ever (see
            // PartitionSpec::prune_safe); the live spec is returned regardless so
            // the write path always targets the current generation.
            let generation_count: i64 = match sqlx::query_scalar(
                "SELECT COUNT(*) FROM ducklake_partition_info WHERE table_id = ?",
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
            let rows = match sqlx::query(SQL_GET_PARTITION_SPEC)
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
            let rows = match sqlx::query(SQL_GET_SORT_SPEC)
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
        self.get_table_file_metadata_page_filtered(
            table_id,
            snapshot_id,
            after_data_file_id,
            limit,
            None,
        )
    }

    fn get_table_file_metadata_page_filtered(
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
            // The statistics filter narrows the list inside this query, before
            // LIMIT and with the `ORDER BY data.data_file_id` keyset ordering
            // intact. Filtering a fetched page instead would break the cursor in
            // `crate::table::FileMetadataPages`: a page whose candidates all
            // fail the filter comes back empty, which reads as "no files left",
            // and every matching file beyond it is never visited.
            let filter_sql = stats_filter_sql(filter, table_id);
            let build_sql = |filter_sql: Option<&StatsFilterSql>| {
                let (cte, joins, conditions) = filter_sql.map_or(("", "", ""), |sql| {
                    (
                        sql.cte.as_str(),
                        sql.joins.as_str(),
                        sql.conditions.as_str(),
                    )
                });
                format!(
                    "{cte}SELECT data.data_file_id, data.path, data.path_is_relative,
                        data.file_size_bytes, data.footer_size, data.encryption_key,
                        data.row_id_start, data.record_count,
                        del.delete_file_id, del.path, del.path_is_relative,
                        del.file_size_bytes, del.footer_size, del.encryption_key,
                        del.delete_count,
                        data.mapping_id
                 FROM ducklake_data_file AS data
                 LEFT JOIN ducklake_delete_file AS del
                   ON data.data_file_id = del.data_file_id
                  AND del.table_id = ?
                  AND ? >= del.begin_snapshot
                  AND (? < del.end_snapshot OR del.end_snapshot IS NULL){joins}
                 WHERE data.table_id = ?
                   AND ? >= data.begin_snapshot
                   AND (? < data.end_snapshot OR data.end_snapshot IS NULL)
                   AND data.data_file_id > ?{conditions}
                 ORDER BY data.data_file_id
                 LIMIT ?"
                )
            };
            let rows = match self
                .fetch_file_page(
                    &build_sql(filter_sql.as_ref()),
                    table_id,
                    snapshot_id,
                    after_data_file_id,
                    limit,
                )
                .await
            {
                Ok(rows) => rows,
                // The filter is advisory, so a catalog the narrowed query cannot
                // run — most importantly one predating
                // `ducklake_file_column_stats`, where joining it is a hard error
                // — still lists its files. The retry uses the same parameters,
                // and a failure that is not the filter's fault surfaces from it.
                Err(error) if filter_sql.is_some() => {
                    tracing::debug!(
                        %error,
                        table_id,
                        "statistics-filtered file listing failed; listing every file"
                    );
                    self.fetch_file_page(
                        &build_sql(None),
                        table_id,
                        snapshot_id,
                        after_data_file_id,
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
            // returned rather than to that whole range. Unfiltered the range
            // holds at most one page of files, but a selective filter can put a
            // handful of survivors at the far end of a million-file table, and
            // the range would then pull back the statistics of every file the
            // filter just pruned — the exact cost the pushdown exists to remove,
            // and unbounded resident memory besides.
            //
            // The ids are inlined rather than bound as one parameter, which is
            // what the PostgreSQL provider does. That is a dialect difference,
            // not a disagreement: `= ANY($n::bigint[])` needs an array type,
            // and PostgreSQL is the only one of these engines that has one (and
            // the only one sqlx 0.9 can bind a `Vec<i64>` to). MySQL's
            // single-parameter shapes are worse than the inlining: `FIND_IN_SET`
            // over a comma-separated string is a per-row scan that no index on
            // `data_file_id` can serve, and a `JSON_TABLE` join would raise the
            // server floor to MySQL 8.0.4. Ids are `i64`, so inlining them adds
            // no bind parameter and the parameter lists stay as they are.
            //
            // Inlining does give every page a distinct query string, and sqlx
            // keys its per-connection prepared-statement cache on that string.
            // A fresh string costs a `COM_STMT_PREPARE` round trip that nothing
            // can amortize, since it will never be reused — so both queries
            // below are non-persistent whenever they carry ids, which has sqlx
            // close the statement after executing it instead of caching it. The
            // cache holds 100 statements by default and would otherwise fill
            // with per-page strings, evicting the listing query and everything
            // else the connection had prepared, adding a second prepare per
            // page to the one already paid.
            let page_ids = filter_sql.as_ref().map(|_| {
                files
                    .iter()
                    .map(|file| file.data_file_id.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            });
            let page_id_filter = |column: &str| {
                page_ids.as_ref().map_or_else(String::new, |ids| {
                    format!("\n                   AND {column} IN ({ids})")
                })
            };
            let statistics = match sqlx::query(AssertSqlSafe(format!(
                "SELECT stats.data_file_id, stats.column_id,
                        stats.column_size_bytes, stats.value_count, stats.null_count,
                        stats.min_value, stats.max_value, stats.contains_nan
                 FROM ducklake_file_column_stats AS stats
                 INNER JOIN ducklake_data_file AS data
                   ON data.data_file_id = stats.data_file_id
                  AND data.table_id = stats.table_id
                 WHERE stats.table_id = ?
                   AND ? >= data.begin_snapshot
                   AND (? < data.end_snapshot OR data.end_snapshot IS NULL)
                   AND stats.data_file_id > ?
                   AND stats.data_file_id <= ?{}
                 ORDER BY stats.data_file_id, stats.column_id",
                page_id_filter("stats.data_file_id")
            )))
            .bind(table_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .bind(after_data_file_id.unwrap_or(i64::MIN))
            .bind(last_data_file_id)
            // Cacheable only while the text is stable; see the note above.
            .persistent(page_ids.is_none())
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
            match sqlx::query(AssertSqlSafe(format!(
                "{SQL_GET_FILE_PARTITION_VALUES}{}",
                page_id_filter("data_file_id")
            )))
            .bind(table_id)
            .bind(after_data_file_id.unwrap_or(i64::MIN))
            .bind(last_data_file_id)
            // Cacheable only while the text is stable; see the note above.
            .persistent(page_ids.is_none())
            .fetch_all(&self.pool)
            .await
            {
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

    fn get_table_summary_statistics(
        &self,
        table_id: i64,
        snapshot_id: i64,
    ) -> Result<DuckLakeStatistics> {
        block_on(async {
            let table = match sqlx::query(
                "SELECT record_count, file_size_bytes
                 FROM ducklake_table_stats WHERE table_id = ?",
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
                             WHERE visible.table_id = ?
                               AND ? >= visible.begin_snapshot
                               AND (? < visible.end_snapshot OR visible.end_snapshot IS NULL)
                           )
                          THEN CAST(SUM(stats.column_size_bytes) AS SIGNED)
                        END
                 FROM ducklake_file_column_stats stats
                 INNER JOIN ducklake_data_file data
                   ON data.data_file_id = stats.data_file_id
                  AND data.table_id = stats.table_id
                 WHERE stats.table_id = ?
                   AND ? >= data.begin_snapshot
                   AND (? < data.end_snapshot OR data.end_snapshot IS NULL)
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
                     WHERE table_id = ?
                       AND ? >= begin_snapshot
                       AND (? < end_snapshot OR end_snapshot IS NULL)
                 )",
            )
            .bind(table_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .fetch_one(&self.pool)
            .await?;
            let columns = match sqlx::query(
                "SELECT column_id, contains_null, min_value, max_value, contains_nan
                 FROM ducklake_table_column_stats WHERE table_id = ?",
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
                 FROM ducklake_table_stats WHERE table_id = ?",
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
                 FROM ducklake_table_column_stats WHERE table_id = ?",
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
                 WHERE stats.table_id = ?
                   AND ? >= data.begin_snapshot
                   AND (? < data.end_snapshot OR data.end_snapshot IS NULL)",
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
        block_on(async {
            if !self.schema_capabilities().await?.inlined_data_tables {
                return Ok(Vec::new());
            }
            let registry = sqlx::query(
                "SELECT table_name FROM ducklake_inlined_data_tables WHERE table_id = ?",
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
                let present = sqlx::query(
                    "SELECT column_name FROM information_schema.columns
                     WHERE table_schema = DATABASE() AND table_name = ?",
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
                            "NULL".to_string()
                        } else {
                            let ident = quote_ident(&column.column_name);
                            inlined_text_projection(
                                InlinedDataBackend::MySql,
                                column,
                                field.data_type(),
                                &ident,
                            )
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                let sql = format!(
                    "SELECT {projected} FROM {} \
                     WHERE ? >= begin_snapshot AND (? < end_snapshot OR end_snapshot IS NULL) \
                     ORDER BY row_id",
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
                "SELECT file_id, row_id FROM {} WHERE begin_snapshot <= ? ORDER BY file_id, row_id",
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
                 WHERE schema_name = ?
                   AND ? >= begin_snapshot
                   AND (? < end_snapshot OR end_snapshot IS NULL)",
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
                 WHERE schema_id = ?
                   AND table_name = ?
                   AND ? >= begin_snapshot
                   AND (? < end_snapshot OR end_snapshot IS NULL)",
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
                "SELECT view_id, schema_id, begin_snapshot, view_name, dialect, `sql`, column_aliases
                 FROM ducklake_view
                 WHERE schema_id = ?
                   AND view_name = ?
                   AND ? >= begin_snapshot
                   AND (? < end_snapshot OR end_snapshot IS NULL)",
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
            // MySQL doesn't support SELECT EXISTS(...) the same way PostgreSQL does
            // Use COUNT instead
            let row = sqlx::query(
                "SELECT COUNT(*) FROM ducklake_table
                 WHERE schema_id = ?
                   AND table_name = ?
                   AND ? >= begin_snapshot
                   AND (? < end_snapshot OR end_snapshot IS NULL)",
            )
            .bind(schema_id)
            .bind(name)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .fetch_one(&self.pool)
            .await?;

            let count: i64 = row.try_get(0)?;
            Ok(count > 0)
        })
    }

    fn list_all_tables(&self, snapshot_id: i64) -> Result<Vec<TableWithSchema>> {
        block_on(async {
            let rows = sqlx::query(
                "SELECT s.schema_name, t.table_id, t.table_name, t.path, t.path_is_relative
                 FROM ducklake_schema s
                 JOIN ducklake_table t ON s.schema_id = t.schema_id
                 WHERE ? >= s.begin_snapshot
                   AND (? < s.end_snapshot OR s.end_snapshot IS NULL)
                   AND ? >= t.begin_snapshot
                   AND (? < t.end_snapshot OR t.end_snapshot IS NULL)
                 ORDER BY s.schema_name, t.table_name",
            )
            .bind(snapshot_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
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
                        v.dialect, v.`sql`, v.column_aliases
                 FROM ducklake_schema s
                 JOIN ducklake_view v ON s.schema_id = v.schema_id
                 WHERE ? >= s.begin_snapshot
                   AND (? < s.end_snapshot OR s.end_snapshot IS NULL)
                   AND ? >= v.begin_snapshot
                   AND (? < v.end_snapshot OR v.end_snapshot IS NULL)
                 ORDER BY s.schema_name, v.view_name",
            )
            .bind(snapshot_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
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
                 WHERE ? >= s.begin_snapshot
                   AND (? < s.end_snapshot OR s.end_snapshot IS NULL)
                   AND ? >= t.begin_snapshot
                   AND (? < t.end_snapshot OR t.end_snapshot IS NULL)
                   AND ? >= c.begin_snapshot
                   AND (? < c.end_snapshot OR c.end_snapshot IS NULL)
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
                    AND ? >= del.begin_snapshot
                    AND (? < del.end_snapshot OR del.end_snapshot IS NULL)
                WHERE ? >= s.begin_snapshot
                  AND (? < s.end_snapshot OR s.end_snapshot IS NULL)
                  AND ? >= t.begin_snapshot
                  AND (? < t.end_snapshot OR t.end_snapshot IS NULL)
                  AND ? >= data.begin_snapshot
                  AND (? < data.end_snapshot OR data.end_snapshot IS NULL)
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
            // (they cannot contain partial files).
            let pm = if self.schema_capabilities().await?.data_file_partial_max {
                "data.partial_max"
            } else {
                "NULL"
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
                WHERE data.table_id = ?
                  AND data.begin_snapshot <= ?
                  AND (data.begin_snapshot >= ?
                       OR ({pm} IS NOT NULL AND {pm} >= ?))
                ORDER BY data.begin_snapshot"
            )))
            .bind(table_id)
            .bind(end_snapshot)
            .bind(start_snapshot)
            .bind(start_snapshot)
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
            // MySQL equivalent of DuckDB's SQL_GET_DELETE_FILES_ADDED_BETWEEN_SNAPSHOTS
            // Uses LATERAL (supported in MySQL 8.0.14+) for previous delete file lookup
            //
            // Cumulative (current-spec) delete files can hold in-window deletions
            // even when their begin_snapshot predates the window; included via
            // `ducklake_delete_file.partial_max`. Older catalogs lack the column
            // (and cumulative delete files); degrade it to NULL there.
            let pm = if self.schema_capabilities().await?.delete_file_partial_max {
                "ddf.partial_max"
            } else {
                "NULL"
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
    WHERE ddf.table_id = ?
      AND ddf.begin_snapshot <= ?
      AND (ddf.begin_snapshot >= ?
           OR ({pm} IS NOT NULL AND {pm} >= ?))
),

data_files AS (
    SELECT df.*
    FROM ducklake_data_file df
    WHERE df.table_id = ?
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
    WHERE ddf.table_id = ?
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
    NULL,
    NULL,
    NULL,
    NULL,
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
    WHERE ddf.table_id = ?
      AND ddf.data_file_id = data.data_file_id
      AND ddf.begin_snapshot < data.end_snapshot
    ORDER BY ddf.begin_snapshot DESC
    LIMIT 1
) prev ON true
WHERE data.table_id = ?
  AND data.end_snapshot >= ?
  AND data.end_snapshot <= ?
"#
            )))
            // Part 1 bindings: table_id (current_delete), end, start (window),
            // start (partial_max), table_id (data_files), table_id (prev lateral)
            .bind(table_id)
            .bind(end_snapshot)
            .bind(start_snapshot)
            .bind(start_snapshot)
            .bind(table_id)
            .bind(table_id)
            // Part 2 bindings: table_id (prev lateral), table_id (data), start_snapshot, end_snapshot
            .bind(table_id)
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
    use super::*;
    use crate::stats_filter::lower_predicate;
    use arrow::datatypes::{Field, Schema, TimeUnit};
    use datafusion::logical_expr::Operator;
    use datafusion::physical_expr::PhysicalExpr;
    use datafusion::physical_expr::expressions::{BinaryExpr, Column, lit};

    /// The shape of `a > 5 AND a < 10` on an `INTEGER` column, as the listing
    /// query splices it. Both conjuncts read a bound and neither reads
    /// `null_count`, so the CTE also selects `value_count` for the guard.
    #[test]
    fn renders_a_two_conjunct_integer_filter() {
        let column = Arc::new(Column::new("a", 0)) as Arc<dyn PhysicalExpr>;
        let predicate = Arc::new(BinaryExpr::new(
            Arc::new(BinaryExpr::new(
                Arc::clone(&column),
                Operator::Gt,
                lit(5i32),
            )),
            Operator::And,
            Arc::new(BinaryExpr::new(column, Operator::Lt, lit(10i32))),
        )) as Arc<dyn PhysicalExpr>;
        let schema = Schema::new(vec![Field::new("a", DataType::Int32, true)]);
        let columns = vec![DuckLakeTableColumn::new(7, "a".to_string(), "int32".to_string(), true)];
        let filter = lower_predicate(&predicate, &schema, &columns).expect("lowered");
        let sql = stats_filter_sql(Some(&filter), 42).expect("rendered");
        println!("CTE:\n{}", sql.cte);
        println!("JOINS:{}", sql.joins);
        println!("CONDITIONS:{}", sql.conditions);

        assert!(
            sql.cte
                .starts_with("WITH col_7_stats AS (SELECT data_file_id, ")
        );
        assert!(sql.cte.contains("min_value, max_value, value_count"));
        assert!(sql.cte.contains("WHERE column_id = 7 AND table_id = 42)"));
        assert!(
            sql.joins
                .contains("LEFT JOIN col_7_stats ON col_7_stats.data_file_id = data.data_file_id")
        );
        assert!(sql.conditions.contains(
            "CASE WHEN col_7_stats.max_value REGEXP '^-?[0-9]{1,20}$' \
             THEN CAST(col_7_stats.max_value AS DECIMAL(65, 0)) END > 5"
        ));
        assert!(sql.conditions.trim_end().ends_with(") IS NOT FALSE"));
    }

    /// A raw string bound is compared byte-wise, never under the connection's
    /// default case- and accent-insensitive collation.
    #[test]
    fn compares_string_bounds_byte_wise() {
        let column = Arc::new(Column::new("s", 0)) as Arc<dyn PhysicalExpr>;
        let predicate =
            Arc::new(BinaryExpr::new(column, Operator::Eq, lit("apple"))) as Arc<dyn PhysicalExpr>;
        let schema = Schema::new(vec![Field::new("s", DataType::Utf8, true)]);
        let columns =
            vec![DuckLakeTableColumn::new(3, "s".to_string(), "varchar".to_string(), true)];
        let filter = lower_predicate(&predicate, &schema, &columns).expect("lowered");
        let sql = stats_filter_sql(Some(&filter), 1).expect("rendered");
        assert!(sql.conditions.contains(
            "'apple' BETWEEN CONVERT(col_3_stats.min_value USING utf8mb4) COLLATE utf8mb4_0900_bin \
             AND CONVERT(col_3_stats.max_value USING utf8mb4) COLLATE utf8mb4_0900_bin"
        ));
    }

    /// A date constant MySQL would refuse to convert is declined here instead:
    /// comparing a `DATE` against `'+12921-08-18'` is error 1525, which would
    /// cost the listing its filter entirely.
    #[test]
    fn declines_a_date_constant_outside_the_canonical_encoding() {
        let column = Arc::new(Column::new("d", 0)) as Arc<dyn PhysicalExpr>;
        let schema = Schema::new(vec![Field::new("d", DataType::Date32, true)]);
        let columns = vec![DuckLakeTableColumn::new(1, "d".to_string(), "date".to_string(), true)];
        let lower = |days: i32| {
            let predicate = Arc::new(BinaryExpr::new(
                Arc::clone(&column),
                Operator::Lt,
                datafusion::physical_expr::expressions::lit(
                    datafusion::common::ScalarValue::Date32(Some(days)),
                ),
            )) as Arc<dyn PhysicalExpr>;
            lower_predicate(&predicate, &schema, &columns).expect("lowered")
        };

        // 19_723 days after the epoch is 2024-01-01; 4_000_000 is +12921-08-18.
        let sql = stats_filter_sql(Some(&lower(19_723)), 1).expect("rendered");
        // Shape-guarded before the cast: MySQL's DATE parser normalises
        // `' 2020-01-01 '`, `2020/01/01` and `20200101`, so the stat has to be
        // pinned to the encoder's spelling before it is converted.
        assert!(
            sql.conditions.contains("REGEXP_SUBSTR(")
                && sql.conditions.contains("AS DATE) < '2024-01-01'"),
            "unexpected date condition: {}",
            sql.conditions
        );
        assert!(stats_filter_sql(Some(&lower(4_000_000)), 1).is_none());
    }

    /// A nanosecond timestamp is compared as text, not cast.
    ///
    /// `DATETIME(6)` holds microseconds and rounds a longer fraction, which is
    /// monotonic but not injective — two distinct nanosecond instants can land
    /// on one microsecond, so a strict comparison that holds of the stored
    /// values comes back false. The encoded text is chronologically ordered at
    /// full precision, so comparing the strings answers the same question
    /// exactly. `stats_encode` trims trailing zeros, and the pattern refuses a
    /// fraction ending in `0`, because as text `.50` sorts above `.5` while
    /// naming the same instant.
    #[test]
    fn nanosecond_timestamps_are_compared_as_text() {
        let column = Arc::new(Column::new("t", 0)) as Arc<dyn PhysicalExpr>;
        let predicate = Arc::new(BinaryExpr::new(
            column,
            Operator::Lt,
            datafusion::physical_expr::expressions::lit(
                datafusion::common::ScalarValue::TimestampNanosecond(Some(1), None),
            ),
        )) as Arc<dyn PhysicalExpr>;
        let schema = Schema::new(vec![Field::new(
            "t",
            DataType::Timestamp(TimeUnit::Nanosecond, None),
            true,
        )]);
        let columns =
            vec![DuckLakeTableColumn::new(1, "t".to_string(), "timestamp_ns".to_string(), true)];
        let filter = lower_predicate(&predicate, &schema, &columns).expect("lowered");
        let sql = stats_filter_sql(Some(&filter), 1)
            .expect("a nanosecond timestamp pushes down")
            .conditions;

        // The constant keeps all nine digits, and nothing is cast.
        assert!(sql.contains("'1970-01-01 00:00:00.000000001'"), "{sql}");
        assert!(!sql.contains("CAST("), "nanosecond was cast: {sql}");
        // The stat side is pinned to the same encoding, byte-collated.
        assert!(
            sql.contains("= REGEXP_SUBSTR(") && sql.contains("([.][0-9]*[1-9])?'"),
            "the stat must be matched over its whole value, not to a `$` anchor: {sql}"
        );
        assert!(sql.contains("COLLATE utf8mb4_0900_bin"), "{sql}");
    }

    /// A constant holding a backslash never reaches the SQL as an escape.
    ///
    /// `'a\'` — the standard rendering — makes MySQL read the closing quote as
    /// escaped and the statement dies with error 1064. Doubling the backslash
    /// repairs that only under the default `sql_mode`: with
    /// `NO_BACKSLASH_ESCAPES` set, `'a\\'` is the two-character string `a\\`,
    /// which does not equal the value the writer stored and would prune a file
    /// that matches. The hexadecimal form has one meaning under both.
    #[test]
    fn backslash_bearing_constants_render_as_hex_literals() {
        let dialect = MySqlStatsDialect;
        assert_eq!(dialect.quote_literal("apple"), "'apple'");
        assert_eq!(dialect.quote_literal("a'b"), "'a''b'");
        assert_eq!(dialect.quote_literal("a\\"), "_utf8mb4 X'615C'");
        assert_eq!(dialect.quote_literal("a\\'"), "_utf8mb4 X'615C27'");
        // UTF-8 bytes, not code points: `é` is 0xC3 0xA9.
        assert_eq!(dialect.quote_literal("\u{e9}\\"), "_utf8mb4 X'C3A95C'");

        let column = Arc::new(Column::new("s", 0)) as Arc<dyn PhysicalExpr>;
        let predicate =
            Arc::new(BinaryExpr::new(column, Operator::Eq, lit("a\\"))) as Arc<dyn PhysicalExpr>;
        let schema = Schema::new(vec![Field::new("s", DataType::Utf8, true)]);
        let columns =
            vec![DuckLakeTableColumn::new(3, "s".to_string(), "varchar".to_string(), true)];
        let filter = lower_predicate(&predicate, &schema, &columns).expect("lowered");
        let sql = stats_filter_sql(Some(&filter), 1).expect("rendered");
        assert!(
            sql.conditions.contains(
                "_utf8mb4 X'615C' BETWEEN CONVERT(col_3_stats.min_value USING utf8mb4) \
                 COLLATE utf8mb4_0900_bin"
            ),
            "unexpected condition: {}",
            sql.conditions
        );
        assert!(
            !sql.conditions.contains('\\'),
            "a backslash reached the SQL text: {}",
            sql.conditions
        );
    }

    /// The float guard bounds both digit runs, because `CAST` does not fail on a
    /// magnitude no `DOUBLE` holds — it saturates. Measured on MySQL 8:
    /// `CAST('1e+400' AS DOUBLE)` is `1.7976931348623157e308`,
    /// `CAST('1e-400' AS DOUBLE)` is `0`, and 400 nines is DBL_MAX again, all
    /// without an error the guard could see. Reading a bound that way is the
    /// "coerces it to a number" failure this dialect exists to prevent.
    #[test]
    fn the_float_guard_declines_magnitudes_mysql_would_saturate() {
        let column = Arc::new(Column::new("f", 0)) as Arc<dyn PhysicalExpr>;
        let predicate =
            Arc::new(BinaryExpr::new(column, Operator::Gt, lit(1.0f64))) as Arc<dyn PhysicalExpr>;
        let schema = Schema::new(vec![Field::new("f", DataType::Float64, true)]);
        let columns =
            vec![DuckLakeTableColumn::new(9, "f".to_string(), "double".to_string(), true)];
        let filter = lower_predicate(&predicate, &schema, &columns).expect("lowered");
        let sql = stats_filter_sql(Some(&filter), 1).expect("rendered");

        // Read the pattern out of the SQL the dialect actually emitted, so this
        // cannot drift from it.
        let (_, after) = sql
            .conditions
            .split_once("REGEXP '")
            .expect("the float cast is guarded by a REGEXP");
        let (pattern, _) = after.split_once("' THEN").expect("guarded cast");
        let guard = regex::Regex::new(pattern).expect("a valid pattern");

        for admitted in
            ["0.0", "-1.5", "1e+16", "1e-20", "1.2345678901234568e+16", &"9".repeat(255)]
        {
            assert!(guard.is_match(admitted), "{admitted} should be admitted");
        }
        for declined in [
            // Saturating magnitudes: the reason the bounds exist.
            "1e+400",
            "1e-400",
            &"9".repeat(400),
            // Representable but past the margin, so declined rather than read.
            "1e+100",
            // The encoder's infinities, which MySQL would read as 0.
            "inf",
            "-inf",
            // Not this encoding at all.
            "nan",
            "0x10",
            " 1.0",
        ] {
            assert!(!guard.is_match(declined), "{declined} should be declined");
        }
    }
}
