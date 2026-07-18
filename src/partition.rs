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
