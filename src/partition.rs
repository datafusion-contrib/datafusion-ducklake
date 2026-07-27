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
            let name = schema
                .fields()
                .get(index)
                .map(|f| f.name().to_string())
                .ok_or_else(|| {
                    crate::DuckLakeError::Internal(format!(
                        "partition column index {index} out of range for write schema"
                    ))
                })?;
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
}

/// The Hive-style relative subpath (`key=value/…`) a file carrying `values` is
/// placed under, given the partition-key column names in key order. Returns an
/// empty string when there are no keys (an unpartitioned file lives directly
/// under the table directory).
///
/// A `None` value (SQL NULL) uses DuckDB's `__HIVE_DEFAULT_PARTITION__` sentinel,
/// matching official DuckLake's on-disk layout. The catalog
/// (`ducklake_file_partition_value`) is the authoritative source for pruning, so
/// this path is for human readability and interop only — values are sanitized for
/// the filesystem without affecting correctness, and a sanitization collision
/// between two distinct values is harmless (files carry distinct UUID names and
/// distinct catalog values).
pub(crate) fn hive_subpath(key_names: &[String], values: &[Option<String>]) -> String {
    let mut rel = String::new();
    for (i, value) in values.iter().enumerate() {
        let name = key_names.get(i).map(String::as_str).unwrap_or("key");
        let encoded = match value {
            Some(v) => sanitize_partition_path(v),
            None => "__HIVE_DEFAULT_PARTITION__".to_string(),
        };
        if rel.is_empty() {
            rel = format!("{name}={encoded}");
        } else {
            rel = format!("{rel}/{name}={encoded}");
        }
    }
    rel
}

/// Sanitize a partition value for use in a Hive-style directory name — see
/// [`hive_subpath`] for why a lossy mapping is safe here.
fn sanitize_partition_path(value: &str) -> String {
    value
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | ':') {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// One partition group of a partitioned write: the per-key partition values every
/// row in the group shares (`values[i]` for partition key `i`, `None` == SQL NULL),
/// and the row batches for that partition.
pub type PartitionGroup = (Vec<Option<String>>, Vec<arrow::record_batch::RecordBatch>);

/// Apply a partition transform to a whole column array: identity returns the
/// column unchanged; the temporal transforms return an `Int32` calendar component
/// (year/month/day/hour) via Arrow's `date_part`. Only producible transforms are
/// valid here — [`PartitionWriteSpec::resolve`] rejects `bucket`/unknown up front.
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
/// batch they came from.
///
/// `output_schema` is the schema the returned batches carry (the table's clean data
/// columns); `batches` must already match it positionally.
pub(crate) fn split_batches_by_partition(
    output_schema: &arrow::datatypes::SchemaRef,
    batches: &[arrow::record_batch::RecordBatch],
    spec: &PartitionWriteSpec,
) -> crate::Result<Vec<PartitionGroup>> {
    use arrow::array::{ArrayRef, RecordBatch, UInt32Array};
    use arrow::compute::{concat_batches, take};
    use std::collections::HashMap;

    if batches.is_empty() {
        return Ok(Vec::new());
    }
    let input_schema = batches[0].schema();
    let combined = concat_batches(&input_schema, batches)?;
    let num_rows = combined.num_rows();
    if num_rows == 0 {
        return Ok(Vec::new());
    }

    // Transform each partition-key column once for the whole dataset.
    let mut transformed: Vec<ArrayRef> = Vec::with_capacity(spec.keys.len());
    for key in &spec.keys {
        transformed.push(transform_array(
            &key.transform,
            combined.column(key.input_index),
        )?);
    }

    // Group row indices by the encoded partition-value tuple.
    let mut groups: HashMap<Vec<Option<String>>, Vec<u32>> = HashMap::new();
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
        groups.entry(values).or_default().push(row as u32);
    }

    // Materialize one batch per group via `take` (output uses the clean schema).
    let mut result = Vec::with_capacity(groups.len());
    for (values, indices) in groups {
        let index_array = UInt32Array::from(indices);
        let columns = combined
            .columns()
            .iter()
            .map(|c| take(c, &index_array, None))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let batch = RecordBatch::try_new(output_schema.clone(), columns)?;
        result.push((values, vec![batch]));
    }
    Ok(result)
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
    use arrow::array::{ArrayRef, RecordBatch, StringArray};
    use arrow::datatypes::{Field, Schema, SchemaRef};
    use std::sync::Arc;

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
        // A NULL partition value uses DuckDB's sentinel directory name.
        assert_eq!(
            hive_subpath(&keys, &[None, Some("3".into())]),
            "region=__HIVE_DEFAULT_PARTITION__/day=3"
        );
        // Path separators in a value can never escape the partition directory.
        assert_eq!(
            hive_subpath(&keys[..1], &[Some("a/../b".into())]),
            "region=a_.._b"
        );
        // No keys: the file lives directly under the table directory.
        assert_eq!(hive_subpath(&[], &[]), "");
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
