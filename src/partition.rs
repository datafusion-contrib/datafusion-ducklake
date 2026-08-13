//! DuckLake table partitioning: transform model + partition spec.
//!
//! A partitioned DuckLake table records, in the catalog, a **partition spec**
//! (`ducklake_partition_info` + `ducklake_partition_column`) and, per data file, the
//! single **partition value** every row in that file shares for each partition key
//! (`ducklake_file_partition_value`). This module is the shared vocabulary for both the
//! read path (spec + values drive file pruning) and the write path (spec drives how
//! rows are split into per-partition files).
//!
//! Following the DuckLake spec, a partition key column is combined with a **transform**:
//! `identity` (the raw value), or the temporal parts `year` / `month` / `day` / `hour`,
//! or `bucket(N)` (Murmur3 hashing). DuckLake stores the transformed value as a literal
//! calendar value (e.g. `month → "6"` in 1..12, `year → "2023"`), *not* an
//! order-preserving epoch offset — which is why only `identity` and `year` yield a
//! contiguous range on the source column (see [`PartitionTransform::source_bounds`]).
//!
//! Scope note: this crate actively prunes/produces `identity` + temporal transforms.
//! `bucket(N)` is *tolerated on read* (parsed, but never pruned or produced) and any
//! unrecognized transform is preserved as [`PartitionTransform::Unknown`] and treated
//! as "cannot prune / cannot produce" — always safe (a file is kept, never mis-dropped).

use arrow::datatypes::DataType;
use datafusion::common::ScalarValue;

/// A DuckLake partition transform applied to a partition-key column.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PartitionTransform {
    /// The column value itself (no transform).
    Identity,
    /// Calendar year, e.g. `2023` (order-preserving on the source column).
    Year,
    /// Calendar month `1..=12` (NOT order-preserving: a `month=6` file holds every June).
    Month,
    /// Calendar day-of-month `1..=31` (not order-preserving).
    Day,
    /// Hour-of-day `0..=23` (not order-preserving).
    Hour,
    /// Murmur3 hash into `N` buckets. Tolerated on read (never pruned or produced here).
    Bucket(u32),
    /// A transform string this crate does not recognize. Preserved verbatim so it can be
    /// round-tripped, but treated as non-prunable / non-producible.
    Unknown(String),
}

impl PartitionTransform {
    /// Parse a `ducklake_partition_column.transform` string (also the form accepted by
    /// the SQL DDL hook). Recognizes `identity`, `year`, `month`, `day`, `hour`, and
    /// `bucket(N)`; anything else becomes [`PartitionTransform::Unknown`].
    pub fn parse(transform: &str) -> Self {
        let trimmed = transform.trim();
        match trimmed.to_ascii_lowercase().as_str() {
            "identity" => return PartitionTransform::Identity,
            "year" => return PartitionTransform::Year,
            "month" => return PartitionTransform::Month,
            "day" => return PartitionTransform::Day,
            "hour" => return PartitionTransform::Hour,
            _ => {},
        }
        // bucket(N)
        if let Some(rest) = trimmed
            .strip_prefix("bucket(")
            .or_else(|| trimmed.strip_prefix("BUCKET("))
            && let Some(inner) = rest.strip_suffix(')')
            && let Ok(n) = inner.trim().parse::<u32>()
        {
            return PartitionTransform::Bucket(n);
        }
        PartitionTransform::Unknown(trimmed.to_string())
    }

    /// The catalog `transform` string this transform serializes to.
    pub fn to_catalog_string(&self) -> String {
        match self {
            PartitionTransform::Identity => "identity".to_string(),
            PartitionTransform::Year => "year".to_string(),
            PartitionTransform::Month => "month".to_string(),
            PartitionTransform::Day => "day".to_string(),
            PartitionTransform::Hour => "hour".to_string(),
            PartitionTransform::Bucket(n) => format!("bucket({n})"),
            PartitionTransform::Unknown(s) => s.clone(),
        }
    }

    /// Whether this crate can *produce* files for this transform on write.
    /// `bucket` and `unknown` are read-only (tolerated but not produced).
    pub fn is_producible(&self) -> bool {
        matches!(
            self,
            PartitionTransform::Identity
                | PartitionTransform::Year
                | PartitionTransform::Month
                | PartitionTransform::Day
                | PartitionTransform::Hour
        )
    }

    /// Whether `value` is a well-formed partition value for this transform on a
    /// column of `column_type`. A NULL value (`None`) is always well-formed — it is a
    /// partition in its own right.
    ///
    /// Deliberately matches official DuckLake's checks exactly, no more:
    ///
    /// - `identity` — must cast to the column's type. Official casts the Hive value
    ///   to the field type and errors when it will not (`MapHiveColumn`).
    /// - `year`/`month`/`day`/`hour` — must parse as an integer, and nothing further.
    ///   Official types these keys as `BIGINT` (not the source column type) and only
    ///   casts (`MapPartitionColumns` via `GetPartitionKeyType`); it does NOT check
    ///   that a month is 1..12 or an hour 0..23. Adding such a range check here would
    ///   reject values official accepts, so it is left out on purpose.
    /// - `bucket(N)` — must be an integer in `0..N`, the one range check official
    ///   does make (`IsValidTransformedHivePartitionValue`).
    ///
    /// None of this can tell whether the FILE's rows actually share the value — only
    /// the caller knows that, in official too.
    pub(crate) fn value_is_well_formed(&self, value: Option<&str>, column_type: &DataType) -> bool {
        let Some(value) = value else {
            return true;
        };
        match self {
            PartitionTransform::Identity => {
                ScalarValue::try_from_string(value.to_string(), column_type).is_ok()
            },
            PartitionTransform::Year
            | PartitionTransform::Month
            | PartitionTransform::Day
            | PartitionTransform::Hour => value.trim().parse::<i64>().is_ok(),
            PartitionTransform::Bucket(buckets) => {
                matches!(value.trim().parse::<i64>(), Ok(b) if b >= 0 && b < i64::from(*buckets))
            },
            // An unrecognized transform's value domain is unknown; accept it rather
            // than reject a value some other writer legitimately produced.
            PartitionTransform::Unknown(_) => true,
        }
    }

    /// Derive a `(min, max)` **envelope** on the *source column* for a file whose
    /// partition value for this transform is `value`, as `ScalarValue`s of the source
    /// column's `data_type`. The envelope is guaranteed to satisfy `min <= every row
    /// value <= max`, so it is always safe to use for pruning (it may be loose, never
    /// too tight — a file is never wrongly dropped).
    ///
    /// - `Identity` → `(v, v)` (exact: every row equals `v`).
    /// - `Year` → `[Y-01-01, (Y+1)-01-01]` for date/timestamp columns (a valid, slightly
    ///   loose envelope — the true max is `< (Y+1)-01-01`).
    /// - `Month` / `Day` / `Hour` → `None` (calendar components are not contiguous on the
    ///   source column, so no single range bounds the file).
    /// - `Bucket` / `Unknown` → `None`.
    ///
    /// Returns `None` when the value cannot be decoded to the column type (fail open).
    pub fn source_bounds(
        &self,
        value: &str,
        data_type: &DataType,
    ) -> Option<(ScalarValue, ScalarValue)> {
        match self {
            PartitionTransform::Identity => {
                let scalar = ScalarValue::try_from_string(value.to_string(), data_type).ok()?;
                Some((scalar.clone(), scalar))
            },
            PartitionTransform::Year => {
                let year: i64 = value.trim().parse().ok()?;
                year_bounds(year, data_type)
            },
            PartitionTransform::Month
            | PartitionTransform::Day
            | PartitionTransform::Hour
            | PartitionTransform::Bucket(_)
            | PartitionTransform::Unknown(_) => None,
        }
    }
}

/// Build the `[Y-01-01, (Y+1)-01-01]` source-column envelope for the `year` transform.
/// Uses Arrow's string→scalar cast (no chrono dependency). The upper bound is the start
/// of the next year — a valid over-estimate of the true max (which is `< (Y+1)-01-01`).
fn year_bounds(year: i64, data_type: &DataType) -> Option<(ScalarValue, ScalarValue)> {
    let (min_str, max_str) = match data_type {
        DataType::Date32 | DataType::Date64 => {
            (format!("{year}-01-01"), format!("{}-01-01", year + 1))
        },
        // Only time-zone-NAIVE timestamps: DuckDB computes `year(timestamptz)` in
        // the session time zone, so a UTC-anchored envelope could exclude real
        // rows near the year boundary. For tz-aware timestamps we derive no
        // envelope (temporal pruning then relies on real column zone-maps).
        DataType::Timestamp(_, None) => (
            format!("{year}-01-01 00:00:00"),
            format!("{}-01-01 00:00:00", year + 1),
        ),
        // Some catalogs store a bare integer year column partitioned "by year"; then the
        // partition value IS the column value, so identity-style exact bounds apply.
        DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64 => {
            let scalar = ScalarValue::try_from_string(year.to_string(), data_type).ok()?;
            return Some((scalar.clone(), scalar));
        },
        _ => return None,
    };
    let min = ScalarValue::try_from_string(min_str, data_type).ok()?;
    let max = ScalarValue::try_from_string(max_str, data_type).ok()?;
    Some((min, max))
}

/// A partition spec resolved against a concrete write schema: how a write path
/// splits incoming rows into per-partition files.
///
/// Built by [`PartitionWriteSpec::resolve`] from the table's live
/// [`PartitionSpec`]. Every write path that produces NEW rows for a partitioned
/// table goes through this — SQL `INSERT` ([`crate::insert_exec::DuckLakeInsertExec`]),
/// the low-level writer entry points, and the UPDATE rewrite — so all of them lay
/// files out the same way and stamp the same `partition_id`.
#[derive(Debug, Clone)]
pub struct PartitionWriteSpec {
    /// The active spec generation (`ducklake_partition_info.partition_id`).
    pub partition_id: i64,
    /// Partition keys, in key order.
    pub keys: Vec<PartitionWriteKey>,
}

/// One partition key resolved for the write path.
#[derive(Debug, Clone)]
pub struct PartitionWriteKey {
    /// Column index in the write input schema.
    pub input_index: usize,
    /// Column name (used only for the readable Hive-style path).
    pub name: String,
    /// Transform applied to the column value to form the partition value.
    pub transform: PartitionTransform,
}

impl PartitionWriteSpec {
    /// Resolve `spec` against the columns a write is about to produce:
    /// `column_ids[i]` is the catalog `column_id` of `schema` field `i` (the 1:1
    /// pairing every write path already has).
    ///
    /// Errors with [`crate::DuckLakeError::Unsupported`] on a transform this crate
    /// cannot PRODUCE (`bucket`/unknown) — writing unpartitioned files into a table
    /// whose spec demands them would violate the spec, so a partitioned write must
    /// fail loudly rather than silently degrade. Errors with
    /// [`crate::DuckLakeError::Internal`] if a partition key names a column absent
    /// from the write schema (the catalog and the write disagree).
    pub fn resolve(
        spec: &PartitionSpec,
        column_ids: &[i64],
        schema: &arrow::datatypes::Schema,
    ) -> crate::Result<PartitionWriteSpec> {
        let mut keys = Vec::with_capacity(spec.columns.len());
        for column in &spec.columns {
            if !column.transform.is_producible() {
                return Err(crate::DuckLakeError::Unsupported(format!(
                    "writing to a table partitioned by '{}' is not supported",
                    column.transform.to_catalog_string()
                )));
            }
            let index = column_ids
                .iter()
                .position(|id| *id == column.column_id)
                .ok_or_else(|| {
                    crate::DuckLakeError::Internal(format!(
                        "partition column_id {} not found in table schema",
                        column.column_id
                    ))
                })?;
            let field = schema.fields().get(index).ok_or_else(|| {
                crate::DuckLakeError::Internal(format!(
                    "partition column index {index} out of range for write schema"
                ))
            })?;
            let name = field.name().to_string();

            // Caveat, deliberately NOT an error: a temporal transform on a
            // time-zone-aware timestamp is computed here in UTC, whereas DuckDB
            // evaluates `year(timestamptz)` in the session time zone, so near a
            // boundary the two produce different partition values for the same row.
            // Official DuckLake permits this combination (it emits `year(col)` and
            // lets the session decide), so rejecting it would refuse a table official
            // accepts. Our read path is unaffected: `year_bounds` declines to derive
            // an envelope for tz-aware timestamps, so we never prune on such a key —
            // only a DuckDB reader applying its own session rule could mis-prune.
            keys.push(PartitionWriteKey {
                input_index: index,
                name,
                transform: column.transform.clone(),
            });
        }
        Ok(PartitionWriteSpec {
            partition_id: spec.partition_id,
            keys,
        })
    }

    /// The partition-key column names in key order — the input to
    /// [`hive_subpath`].
    pub(crate) fn key_names(&self) -> Vec<String> {
        self.keys.iter().map(|k| k.name.clone()).collect()
    }

    /// Validate one file's partition `values` (in key order) against this spec and
    /// the write schema: one value per key, each well-formed for its transform and
    /// column type (see [`PartitionTransform::value_is_well_formed`]).
    ///
    /// Values this crate derived itself are well-formed by construction; this guards
    /// the paths that accept them from a caller
    /// ([`crate::table_writer::DuckLakeTableWriter::write_partitioned`]), where a
    /// wrong arity or an unparseable value would otherwise be persisted and then
    /// used as an EXACT pruning bound — silently dropping rows from later reads.
    pub(crate) fn validate_values(
        &self,
        schema: &arrow::datatypes::Schema,
        values: &[Option<String>],
    ) -> crate::Result<()> {
        if values.len() != self.keys.len() {
            return Err(crate::DuckLakeError::InvalidConfig(format!(
                "partitioned write supplied {} value(s) for a spec with {} key(s)",
                values.len(),
                self.keys.len()
            )));
        }
        for (key, value) in self.keys.iter().zip(values.iter()) {
            let column_type = schema
                .fields()
                .get(key.input_index)
                .map(|f| f.data_type())
                .ok_or_else(|| {
                    crate::DuckLakeError::Internal(format!(
                        "partition key column index {} out of range for write schema",
                        key.input_index
                    ))
                })?;
            if !key
                .transform
                .value_is_well_formed(value.as_deref(), column_type)
            {
                return Err(crate::DuckLakeError::InvalidConfig(format!(
                    "partition value {value:?} is not valid for key '{}' with transform '{}' on a \
                     {column_type} column",
                    key.name,
                    key.transform.to_catalog_string()
                )));
            }
        }
        Ok(())
    }
}

/// The Hive-style relative subpath (`key=value/…`) a file carrying `values` is
/// placed under, given the partition-key column names in key order. Returns an
/// empty string when there are no keys (an unpartitioned file lives directly
/// under the table directory).
///
/// A `None` value (SQL NULL) uses DuckDB's `__HIVE_DEFAULT_PARTITION__` sentinel.
/// Key names and values are percent-encoded by [`escape_partition_path`], which
/// reproduces official DuckLake's on-disk layout byte for byte. Official escapes
/// both halves through `HivePartitioning::Escape` in
/// `src/storage/ducklake_partition_data.cpp`, and its insert path delegates
/// directory naming to DuckDB core's partitioned COPY, which applies that same
/// escape.
///
/// The catalog (`ducklake_file_partition_value`) remains the authoritative source
/// for pruning — nothing reads values back out of the path — but the encoding is
/// reversible, so the directory name alone identifies the partition value the way
/// official's `HivePartitioning::Unescape` expects.
pub(crate) fn hive_subpath(key_names: &[String], values: &[Option<String>]) -> String {
    let mut rel = String::new();
    for (i, value) in values.iter().enumerate() {
        let name = escape_partition_path(key_names.get(i).map(String::as_str).unwrap_or("key"));
        let encoded = match value {
            Some(v) => escape_partition_path(v),
            None => escape_partition_path("__HIVE_DEFAULT_PARTITION__"),
        };
        if rel.is_empty() {
            rel = format!("{name}={encoded}");
        } else {
            rel = format!("{rel}/{name}={encoded}");
        }
    }
    rel
}

/// Percent-encode one Hive directory-name component, byte for byte identical to
/// DuckDB's `StringUtil::URLEncode` with `encode_slash = true` — the escape
/// official DuckLake applies to every partition key name and value.
///
/// `A-Z`, `a-z`, `0-9`, `_`, `-`, `~` and `.` pass through; every other byte
/// becomes `%` plus two **uppercase** hex digits. Encoding per byte rather than
/// per `char` is what keeps multi-byte UTF-8 identical to DuckDB, which walks the
/// string a byte at a time. `/` is escaped rather than preserved, so a value can
/// never introduce a directory level or escape the partition directory.
fn escape_partition_path(value: &str) -> String {
    const HEX_DIGIT: &[u8; 16] = b"0123456789ABCDEF";
    let mut out = String::with_capacity(value.len());
    for byte in value.as_bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'_' | b'-' | b'~' | b'.' => {
                out.push(char::from(*byte))
            },
            _ => {
                out.push('%');
                out.push(char::from(HEX_DIGIT[usize::from(byte >> 4)]));
                out.push(char::from(HEX_DIGIT[usize::from(byte & 15)]));
            },
        }
    }
    out
}

/// One partition group of a partitioned write: the per-key partition values every
/// row in the group shares (`values[i]` for partition key `i`, `None` == SQL NULL),
/// and the row batches for that partition.
#[cfg(feature = "write")]
pub type PartitionGroup = (Vec<Option<String>>, Vec<arrow::record_batch::RecordBatch>);

/// Apply a partition transform to a whole column array: identity returns the
/// column unchanged; the temporal transforms return an `Int32` calendar component
/// (year/month/day/hour) via Arrow's `date_part`. Only producible transforms are
/// valid here — [`PartitionWriteSpec::resolve`] rejects `bucket`/unknown up front.
#[cfg(feature = "write")]
fn transform_array(
    transform: &PartitionTransform,
    array: &arrow::array::ArrayRef,
) -> crate::Result<arrow::array::ArrayRef> {
    use arrow::compute::{DatePart, date_part};
    let part = match transform {
        PartitionTransform::Identity => return Ok(std::sync::Arc::clone(array)),
        PartitionTransform::Year => DatePart::Year,
        PartitionTransform::Month => DatePart::Month,
        PartitionTransform::Day => DatePart::Day,
        PartitionTransform::Hour => DatePart::Hour,
        other => {
            return Err(crate::DuckLakeError::Unsupported(format!(
                "partitioned write with transform '{}' is not supported",
                other.to_catalog_string()
            )));
        },
    };
    Ok(date_part(array, part)?)
}

/// Split rows into groups keyed by the tuple of transformed, DuckDB-canonical
/// partition values — one group per distinct key. Returns `(values, batches)` per
/// group, where `values[i]` is the encoded value for partition key `i` (`None` for
/// SQL NULL). Rows sharing a key land in the same group regardless of which input
/// batch they came from, and keep their relative order.
///
/// Each group holds ONE OUTPUT BATCH PER INPUT BATCH that contributed rows to it,
/// rather than a single concatenated batch. This matters because the writer evaluates
/// file rollover at batch boundaries: collapsing a group into one batch would leave
/// `target_file_size` unenforceable within a partition, emitting one file of unbounded
/// size however large the write. It also keeps peak memory to one input batch's worth
/// of `take` output instead of a full copy of the input.
///
/// `output_schema` is the schema the returned batches carry (the table's clean data
/// columns); `batches` must already match it positionally.
///
/// Write-only: the partition-value encoding lives in [`crate::stats_encode`], which
/// is itself gated behind `write`.
#[cfg(feature = "write")]
pub(crate) fn split_batches_by_partition(
    output_schema: &arrow::datatypes::SchemaRef,
    batches: &[arrow::record_batch::RecordBatch],
    spec: &PartitionWriteSpec,
) -> crate::Result<Vec<PartitionGroup>> {
    use arrow::array::{ArrayRef, RecordBatch, UInt32Array};
    use arrow::compute::take;
    use std::collections::HashMap;

    // Group index by partition values, so groups keep first-seen order across
    // batches and every batch appends into the same group.
    let mut order: Vec<Vec<Option<String>>> = Vec::new();
    let mut groups: HashMap<Vec<Option<String>>, Vec<RecordBatch>> = HashMap::new();

    for batch in batches {
        let num_rows = batch.num_rows();
        if num_rows == 0 {
            continue;
        }
        // Transform each partition-key column once per batch.
        let mut transformed: Vec<ArrayRef> = Vec::with_capacity(spec.keys.len());
        for key in &spec.keys {
            transformed.push(transform_array(
                &key.transform,
                batch.column(key.input_index),
            )?);
        }

        // Row indices of this batch, bucketed by encoded partition-value tuple.
        // Ascending within a bucket, so relative order is preserved.
        let mut per_batch: HashMap<Vec<Option<String>>, Vec<u32>> = HashMap::new();
        let mut per_batch_order: Vec<Vec<Option<String>>> = Vec::new();
        for row in 0..num_rows {
            let mut values: Vec<Option<String>> = Vec::with_capacity(spec.keys.len());
            for array in &transformed {
                let scalar = ScalarValue::try_from_array(array, row)?;
                // `encode_scalar` returns `None` for BOTH a genuine SQL NULL and a
                // non-null value of a type it cannot encode. Those must not be
                // conflated: silently mapping an unencodable non-null value to `None`
                // would group every distinct such value into one file with a NULL
                // partition value (data corruption). A NULL is a legitimate partition
                // value; an unencodable non-null value is a hard error.
                let encoded = if scalar.is_null() {
                    None
                } else {
                    match crate::stats_encode::encode_scalar(&scalar) {
                        Some(encoded) => Some(encoded),
                        None => {
                            return Err(crate::DuckLakeError::Unsupported(format!(
                                "partitioned write: partition-key value of type {} cannot be \
                                 encoded; partitioning by this column type is not supported",
                                array.data_type()
                            )));
                        },
                    }
                };
                values.push(encoded);
            }
            if !per_batch.contains_key(&values) {
                per_batch_order.push(values.clone());
            }
            per_batch.entry(values).or_default().push(row as u32);
        }

        // Materialize this batch's contribution to each group it touched.
        for values in per_batch_order {
            let indices = per_batch.remove(&values).unwrap_or_default();
            if indices.is_empty() {
                continue;
            }
            let index_array = UInt32Array::from(indices);
            let columns = batch
                .columns()
                .iter()
                .map(|c| take(c, &index_array, None))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            let out = RecordBatch::try_new(output_schema.clone(), columns)?;
            if !groups.contains_key(&values) {
                order.push(values.clone());
            }
            groups.entry(values).or_default().push(out);
        }
    }

    Ok(order
        .into_iter()
        .filter_map(|values| groups.remove(&values).map(|batches| (values, batches)))
        .collect())
}

/// One column of a partition spec: which table column, and how it is transformed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartitionSpecColumn {
    /// 0-based position of this column within the partition key.
    pub partition_key_index: i32,
    /// The `ducklake_column.column_id` this partition key transforms.
    pub column_id: i64,
    /// The transform applied to the column value.
    pub transform: PartitionTransform,
}

/// A table's active partition spec (one generation of `ducklake_partition_info`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartitionSpec {
    /// `ducklake_partition_info.partition_id` for this spec generation.
    pub partition_id: i64,
    /// Partition-key columns, ordered by `partition_key_index`.
    pub columns: Vec<PartitionSpecColumn>,
    /// Whether this spec's `partition_key_index → column` mapping may safely be
    /// used to PRUNE arbitrary live files by their stored partition values.
    ///
    /// True only when the table has exactly one partition-spec generation ever, so
    /// every live file's values were written under this same mapping. After a
    /// re-partition (`SET`→`SET`, or `SET`→`RESET`→`SET`) a live file could carry
    /// values from a RETIRED generation whose key order differs, so mapping them
    /// through this spec could mis-prune — pruning is therefore disabled
    /// (`false`). It does NOT affect the write path: a write always targets the
    /// single live generation, which is unambiguous regardless of history.
    pub prune_safe: bool,
}

impl PartitionSpec {
    /// Look up the transform for a given `column_id`, if it is a partition key.
    pub fn transform_for_column(&self, column_id: i64) -> Option<&PartitionTransform> {
        self.columns
            .iter()
            .find(|c| c.column_id == column_id)
            .map(|c| &c.transform)
    }

    /// Build a spec from catalog rows `(partition_id, partition_key_index,
    /// column_id, transform)` (the join of `ducklake_partition_info` and
    /// `ducklake_partition_column`, ordered by `partition_key_index`) for the
    /// single LIVE generation. Returns `None` when there are no rows
    /// (unpartitioned). `prune_safe` records whether pruning may use this mapping
    /// (see [`PartitionSpec::prune_safe`]). Every row is expected to carry the same
    /// `partition_id`; the first row's id is used.
    pub fn from_rows(
        rows: Vec<(i64, i32, i64, String)>,
        prune_safe: bool,
    ) -> Option<PartitionSpec> {
        let partition_id = rows.first()?.0;
        let columns = rows
            .into_iter()
            .map(
                |(_, partition_key_index, column_id, transform)| PartitionSpecColumn {
                    partition_key_index,
                    column_id,
                    transform: PartitionTransform::parse(&transform),
                },
            )
            .collect();
        Some(PartitionSpec {
            partition_id,
            columns,
            prune_safe,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // Only the `write`-gated split tests below build record batches.
    #[cfg(feature = "write")]
    use arrow::array::{ArrayRef, RecordBatch, StringArray};
    #[cfg(feature = "write")]
    use arrow::datatypes::SchemaRef;
    use arrow::datatypes::{Field, Schema};
    #[cfg(feature = "write")]
    use std::sync::Arc;

    #[cfg(feature = "write")]
    fn identity_region_spec() -> PartitionWriteSpec {
        PartitionWriteSpec {
            partition_id: 1,
            keys: vec![PartitionWriteKey {
                input_index: 0,
                name: "region".to_string(),
                transform: PartitionTransform::Identity,
            }],
        }
    }

    #[cfg(feature = "write")]
    #[test]
    fn split_groups_by_identity_and_keeps_null_partition() {
        let schema: SchemaRef = Arc::new(Schema::new(vec![Field::new(
            "region",
            DataType::Utf8,
            true,
        )]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(StringArray::from(vec![Some("us"), None, Some("us")])) as ArrayRef],
        )
        .unwrap();
        let groups = split_batches_by_partition(
            &schema,
            std::slice::from_ref(&batch),
            &identity_region_spec(),
        )
        .unwrap();
        // "us" (2 rows) and a legitimate NULL partition (1 row).
        assert_eq!(groups.len(), 2);
        let total: usize = groups
            .iter()
            .flat_map(|(_, b)| b)
            .map(|b| b.num_rows())
            .sum();
        assert_eq!(total, 3);
        let mut values: Vec<Option<String>> = groups.iter().map(|(v, _)| v[0].clone()).collect();
        values.sort();
        assert_eq!(values, vec![None, Some("us".to_string())]);
    }

    #[cfg(feature = "write")]
    #[test]
    fn split_errors_on_unencodable_non_null_value_instead_of_corrupting() {
        let schema: SchemaRef = Arc::new(Schema::new(vec![Field::new(
            "region",
            DataType::Utf8,
            true,
        )]));
        // A NUL byte makes the value unencodable (encode_scalar returns None) but it
        // is NOT null — it must error, not silently collapse into a NULL partition
        // and commingle with genuinely-null rows.
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(StringArray::from(vec![Some("a\u{0}b")])) as ArrayRef],
        )
        .unwrap();
        let err = split_batches_by_partition(
            &schema,
            std::slice::from_ref(&batch),
            &identity_region_spec(),
        )
        .unwrap_err();
        assert!(
            err.to_string().to_lowercase().contains("encode"),
            "expected an encode error, got: {err}"
        );
    }

    #[test]
    fn hive_subpath_encodes_keys_values_and_nulls() {
        let keys = vec!["region".to_string(), "day".to_string()];
        assert_eq!(
            hive_subpath(&keys, &[Some("us".into()), Some("3".into())]),
            "region=us/day=3"
        );
        // A NULL partition value uses DuckDB's sentinel directory name. Every
        // byte of it is unreserved, so escaping leaves it verbatim.
        assert_eq!(
            hive_subpath(&keys, &[None, Some("3".into())]),
            "region=__HIVE_DEFAULT_PARTITION__/day=3"
        );
        // Path separators in a value can never introduce a directory level or
        // escape the partition directory: `/` encodes to %2F.
        assert_eq!(
            hive_subpath(&keys[..1], &[Some("a/../b".into())]),
            "region=a%2F..%2Fb"
        );
        // A key name is escaped on the same terms as a value.
        assert_eq!(
            hive_subpath(&["odd name".to_string()], &[Some("x".into())]),
            "odd%20name=x"
        );
        // No keys: the file lives directly under the table directory.
        assert_eq!(hive_subpath(&[], &[]), "");
    }

    /// Golden directory names captured from the official DuckLake extension
    /// (DuckDB v1.5.5): a `timestamptz`-partitioned table written by
    /// `ALTER TABLE … SET PARTITIONED BY (ts)` produced exactly these paths in
    /// `ducklake_data_file.path`. The stored `partition_value` was the
    /// unescaped `2024-01-15 12:30:00+00`, so this pins only the path encoding.
    #[test]
    fn hive_subpath_matches_official_ducklake_paths() {
        let keys = vec!["ts".to_string()];
        assert_eq!(
            hive_subpath(&keys, &[Some("2024-01-15 12:30:00+00".into())]),
            "ts=2024-01-15%2012%3A30%3A00%2B00"
        );
        assert_eq!(
            hive_subpath(&keys, &[Some("2024-06-02 04:00:00.5+00".into())]),
            "ts=2024-06-02%2004%3A00%3A00.5%2B00"
        );
    }

    #[test]
    fn escape_partition_path_matches_duckdb_url_encode() {
        // Unreserved set is exactly A-Za-z0-9 and `_ - ~ .`.
        assert_eq!(
            escape_partition_path("aZ09_-~."),
            "aZ09_-~.",
            "unreserved bytes must pass through"
        );
        // Uppercase hex, and `+` is escaped rather than treated as a space.
        assert_eq!(escape_partition_path(" :+/=%"), "%20%3A%2B%2F%3D%25");
        // Encoded per byte, not per char: 'é' is U+00E9 => C3 A9 in UTF-8.
        assert_eq!(escape_partition_path("é"), "%C3%A9");
        assert_eq!(escape_partition_path(""), "");
    }

    #[test]
    fn resolve_maps_column_ids_to_write_schema_indices() {
        let schema = Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("region", DataType::Utf8, true),
        ]);
        let spec = PartitionSpec {
            partition_id: 7,
            columns: vec![PartitionSpecColumn {
                partition_key_index: 0,
                column_id: 20,
                transform: PartitionTransform::Identity,
            }],
            prune_safe: true,
        };
        // column_ids[i] is the catalog id of schema field i, so column_id 20 is
        // field 1 ("region") — not key index 0.
        let resolved = PartitionWriteSpec::resolve(&spec, &[10, 20], &schema).unwrap();
        assert_eq!(resolved.partition_id, 7);
        assert_eq!(resolved.keys.len(), 1);
        assert_eq!(resolved.keys[0].input_index, 1);
        assert_eq!(resolved.keys[0].name, "region");
        assert_eq!(resolved.key_names(), vec!["region".to_string()]);
    }

    #[test]
    fn resolve_rejects_non_producible_transform() {
        let schema = Schema::new(vec![Field::new("id", DataType::Int64, false)]);
        let spec = PartitionSpec {
            partition_id: 1,
            columns: vec![PartitionSpecColumn {
                partition_key_index: 0,
                column_id: 10,
                transform: PartitionTransform::Bucket(8),
            }],
            prune_safe: true,
        };
        // `bucket` is readable but not producible: a partitioned write must fail
        // rather than silently emit unpartitioned files the spec forbids.
        let err = PartitionWriteSpec::resolve(&spec, &[10], &schema).unwrap_err();
        assert!(
            err.to_string().contains("bucket(8)"),
            "expected the transform in the error, got: {err}"
        );
    }

    #[test]
    fn resolve_allows_temporal_transform_on_tz_aware_timestamp() {
        // Official DuckLake permits `year(timestamptz)` — it emits `year(col)` and
        // lets the session time zone decide the value — so we must not refuse it.
        // The value we compute is UTC-based, which is a documented cross-engine
        // caveat on `resolve`, not an error.
        let tz = DataType::Timestamp(arrow::datatypes::TimeUnit::Microsecond, Some("UTC".into()));
        let schema = Schema::new(vec![Field::new("ts", tz, true)]);
        let spec = PartitionSpec {
            partition_id: 1,
            columns: vec![PartitionSpecColumn {
                partition_key_index: 0,
                column_id: 10,
                transform: PartitionTransform::Year,
            }],
            prune_safe: true,
        };
        assert!(PartitionWriteSpec::resolve(&spec, &[10], &schema).is_ok());
    }

    #[test]
    fn resolve_errors_when_partition_column_absent_from_write_schema() {
        let schema = Schema::new(vec![Field::new("id", DataType::Int64, false)]);
        let spec = PartitionSpec {
            partition_id: 1,
            columns: vec![PartitionSpecColumn {
                partition_key_index: 0,
                column_id: 99,
                transform: PartitionTransform::Identity,
            }],
            prune_safe: true,
        };
        assert!(PartitionWriteSpec::resolve(&spec, &[10], &schema).is_err());
    }

    #[test]
    fn parse_and_roundtrip() {
        for (s, expected) in [
            ("identity", PartitionTransform::Identity),
            ("year", PartitionTransform::Year),
            ("MONTH", PartitionTransform::Month),
            ("day", PartitionTransform::Day),
            ("hour", PartitionTransform::Hour),
            ("bucket(8)", PartitionTransform::Bucket(8)),
        ] {
            assert_eq!(PartitionTransform::parse(s), expected);
        }
        // roundtrip catalog strings
        for t in [
            PartitionTransform::Identity,
            PartitionTransform::Year,
            PartitionTransform::Month,
            PartitionTransform::Day,
            PartitionTransform::Hour,
            PartitionTransform::Bucket(4),
        ] {
            assert_eq!(PartitionTransform::parse(&t.to_catalog_string()), t);
        }
    }

    #[test]
    fn unknown_transform_preserved() {
        let t = PartitionTransform::parse("truncate(10)");
        assert_eq!(t, PartitionTransform::Unknown("truncate(10)".to_string()));
        assert_eq!(t.to_catalog_string(), "truncate(10)");
        assert!(!t.is_producible());
        assert_eq!(t.source_bounds("x", &DataType::Utf8), None);
    }

    #[test]
    fn identity_bounds_are_exact() {
        let (min, max) = PartitionTransform::Identity
            .source_bounds("42", &DataType::Int32)
            .unwrap();
        assert_eq!(min, ScalarValue::Int32(Some(42)));
        assert_eq!(max, ScalarValue::Int32(Some(42)));

        let (min, max) = PartitionTransform::Identity
            .source_bounds("us", &DataType::Utf8)
            .unwrap();
        assert_eq!(min, ScalarValue::Utf8(Some("us".to_string())));
        assert_eq!(max, ScalarValue::Utf8(Some("us".to_string())));
    }

    #[test]
    fn year_bounds_span_the_year_for_dates() {
        let (min, max) = PartitionTransform::Year
            .source_bounds("2023", &DataType::Date32)
            .unwrap();
        let expected_min =
            ScalarValue::try_from_string("2023-01-01".to_string(), &DataType::Date32).unwrap();
        let expected_max =
            ScalarValue::try_from_string("2024-01-01".to_string(), &DataType::Date32).unwrap();
        assert_eq!(min, expected_min);
        assert_eq!(max, expected_max);
    }

    #[test]
    fn year_on_tz_aware_timestamp_has_no_bounds() {
        // year(timestamptz) is session-tz-dependent; deriving a UTC envelope could
        // wrongly drop rows near the year boundary, so we produce no bounds.
        let tz = DataType::Timestamp(arrow::datatypes::TimeUnit::Microsecond, Some("UTC".into()));
        assert_eq!(PartitionTransform::Year.source_bounds("2023", &tz), None);
        // Naive timestamps and dates still get an envelope.
        let naive = DataType::Timestamp(arrow::datatypes::TimeUnit::Microsecond, None);
        assert!(
            PartitionTransform::Year
                .source_bounds("2023", &naive)
                .is_some()
        );
    }

    #[test]
    fn non_order_preserving_transforms_have_no_bounds() {
        for t in [
            PartitionTransform::Month,
            PartitionTransform::Day,
            PartitionTransform::Hour,
            PartitionTransform::Bucket(4),
        ] {
            assert_eq!(t.source_bounds("6", &DataType::Date32), None);
        }
    }
}
