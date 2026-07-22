//! Harvest per-file column statistics from a finished Parquet file's metadata.
//!
//! This mirrors how official DuckLake produces `ducklake_file_column_stats`: it
//! does **not** re-scan the data, it reads the statistics the Parquet writer
//! already computed while writing (row-group metadata). DuckLake's C++ path asks
//! its COPY for `WRITTEN_FILE_STATISTICS` and parses the returned min/max/null
//! counts (`ducklake_insert.cpp`); we do the equivalent by reading the parquet
//! file footer we just wrote and aggregating the per-row-group statistics into
//! one bound per column.
//!
//! Values are converted to DuckDB-canonical `VARCHAR` via [`crate::stats_encode`]
//! so the rows are byte-identical to what DuckDB writes (and thus prunable by
//! DuckDB and by the read side in #161).

use std::cmp::Ordering;
use std::path::Path;

use arrow::array::{Array, ArrayRef, Float32Array, Float64Array};
use arrow::datatypes::{DataType, SchemaRef};
use arrow::record_batch::RecordBatch;
use datafusion::common::ScalarValue;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::arrow_reader::statistics::StatisticsConverter;
use parquet::file::metadata::ParquetMetaData;

use crate::metadata_writer::ColumnStat;
use crate::stats_encode;

/// Whether a `FLOAT`/`DOUBLE` array contains a NaN among its non-null values.
/// Returns `None` for non-floating arrays (NaN is not applicable), so a column's
/// `contains_nan` stays `NULL` = unknown for those.
///
/// The standard Parquet footer records no NaN flag, so — unlike min/max/null
/// which we harvest from the footer — this is computed directly from the Arrow
/// data at write time (the batch is already in memory, so it is a cheap CPU pass
/// with no extra I/O). Lets a reader (e.g. DuckDB) prune float columns, which it
/// otherwise skips when `contains_nan` is unknown.
pub fn array_contains_nan(array: &dyn Array) -> Option<bool> {
    match array.data_type() {
        DataType::Float32 => {
            let a = array.as_any().downcast_ref::<Float32Array>()?;
            Some((0..a.len()).any(|i| a.is_valid(i) && a.value(i).is_nan()))
        },
        DataType::Float64 => {
            let a = array.as_any().downcast_ref::<Float64Array>()?;
            Some((0..a.len()).any(|i| a.is_valid(i) && a.value(i).is_nan()))
        },
        _ => None,
    }
}

/// Fold a batch's NaN presence into a running per-column accumulator over the
/// first `n` columns (the catalog data columns; trailing embedded rowid/snapshot
/// columns are excluded by choosing `n = column_ids.len()`). `acc` grows to `n`
/// entries; each is `Some(true)` if any batch's column had a NaN, `Some(false)`
/// if it's a float column seen without NaN, or `None` for non-float columns.
pub fn accumulate_nan_flags(acc: &mut Vec<Option<bool>>, batch: &RecordBatch, n: usize) {
    if acc.len() < n {
        acc.resize(n, None);
    }
    for (slot, col) in acc.iter_mut().zip(batch.columns()).take(n) {
        if let Some(has) = array_contains_nan(col.as_ref()) {
            *slot = Some(slot.unwrap_or(false) || has);
        }
    }
}

/// Harvest per-column statistics for the Parquet file at `path`.
///
/// `column_ids` are the catalog `column_id`s for the file's columns, in physical
/// (written) order; `row_count` is the total rows in the file. Returns one
/// [`ColumnStat`] per column whose stats could be read.
///
/// Best-effort and never fatal: any failure to open the file or read a column's
/// statistics is logged and that column is simply omitted (or its bound left
/// `None`). A missing/`None` bound is spec-safe — DuckLake keeps (never prunes)
/// a file whose stat is `NULL` — so a degraded harvest can only cost pruning,
/// never correctness.
pub fn collect_column_stats(
    path: &Path,
    column_ids: &[i64],
    row_count: i64,
    contains_nan_flags: &[Option<bool>],
) -> Vec<ColumnStat> {
    match try_collect(path, column_ids, row_count, contains_nan_flags) {
        Ok(stats) => stats,
        Err(e) => {
            tracing::warn!(
                error = %e,
                path = %path.display(),
                "failed to harvest parquet statistics; writing file without column stats"
            );
            Vec::new()
        },
    }
}

fn try_collect(
    path: &Path,
    column_ids: &[i64],
    row_count: i64,
    contains_nan_flags: &[Option<bool>],
) -> crate::Result<Vec<ColumnStat>> {
    let file = std::fs::File::open(path)?;
    // Parses only the footer (no data pages are read).
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| crate::error::DuckLakeError::Internal(format!("parquet metadata: {e}")))?;
    // Use the reader's own reconstructed Arrow schema + parquet schema so
    // `StatisticsConverter`'s name/type match check always agrees (avoids the
    // field-id metadata mismatch a hand-built write schema would trigger).
    let arrow_schema = builder.schema().clone();
    let metadata = builder.metadata().clone();
    Ok(compute_column_stats(
        &arrow_schema,
        &metadata,
        column_ids,
        row_count,
        contains_nan_flags,
    ))
}

/// Compute per-column statistics from an already-parsed parquet footer.
///
/// The shared core behind both harvest paths: the write-through path opens a
/// local staging file for the footer (see [`collect_column_stats`]), while a
/// register-by-reference path (e.g. a server-side-copy load) reads the footer
/// from the object store. Both feed the exact same computation here so the
/// stats are identical regardless of how the file arrived.
///
/// `arrow_schema` must be the schema reconstructed from `metadata` (as a parquet
/// reader builder produces), so `StatisticsConverter`'s name/type match agrees.
/// `column_ids[i]` labels file column `i`. `contains_nan_flags` is empty when the
/// caller could not scan the data for NaN (footer-only paths): min/max are then
/// kept, and only NaN-suppression is skipped.
pub fn compute_column_stats(
    arrow_schema: &SchemaRef,
    metadata: &ParquetMetaData,
    column_ids: &[i64],
    row_count: i64,
    contains_nan_flags: &[Option<bool>],
) -> Vec<ColumnStat> {
    let parquet_schema = metadata.file_metadata().schema_descr();
    let row_groups = metadata.row_groups();

    let fields = arrow_schema.fields();
    let n = fields.len().min(column_ids.len());
    let mut out = Vec::with_capacity(n);

    for (idx, column_id) in column_ids.iter().copied().enumerate().take(n) {
        let field = &fields[idx];
        let name = field.name();

        let converter = match StatisticsConverter::try_new(name, arrow_schema, parquet_schema) {
            Ok(c) => c,
            Err(e) => {
                tracing::debug!(column = name, error = %e, "no statistics converter for column");
                continue;
            },
        };

        // Null counts: sum across row groups. Present for files we write
        // (statistics on by default). If unknown, both counts stay None
        // (DuckLake stores value_count/null_count NULL together).
        let (null_count, value_count) = match converter.row_group_null_counts(row_groups.iter()) {
            Ok(counts) => {
                let total: u64 = (0..counts.len())
                    .filter(|i| !counts.is_null(*i))
                    .map(|i| counts.value(i))
                    .sum();
                let null_count = i64::try_from(total).ok();
                let value_count = null_count.map(|nc| (row_count - nc).max(0));
                (null_count, value_count)
            },
            Err(_) => (None, None),
        };

        // Computed from the Arrow data at write time (the footer has no NaN
        // flag); None for non-float columns.
        let contains_nan = contains_nan_flags.get(idx).copied().flatten();

        // When NaN is present, suppress min/max entirely (store NULL), matching
        // official DuckLake. The Parquet footer's min/max exclude NaN, but under
        // DuckDB's NaN ordering (NaN sorts above every value) a reader that
        // pruned on the NaN-excluded max could wrongly drop the NaN row for a
        // predicate like `x > 100`. NULL bounds => the file is never pruned on
        // this column, which is always sound.
        let (min_scalar, max_scalar) = if contains_nan == Some(true) {
            (None, None)
        } else {
            let min_scalar = converter
                .row_group_mins(row_groups.iter())
                .ok()
                .and_then(|arr| reduce(&arr, Ordering::Less));
            let max_scalar = converter
                .row_group_maxes(row_groups.iter())
                .ok()
                .and_then(|arr| reduce(&arr, Ordering::Greater));
            (min_scalar, max_scalar)
        };

        // Compressed on-disk size of this column: sum the compressed size of
        // every parquet *leaf* column belonging to this top-level field, across
        // all row groups. Mirrors official DuckLake, which records the parquet
        // `total_compressed_size` per column (`ducklake_add_data_files.cpp`).
        // Read from the footer we already parsed, no extra I/O.
        //
        // Parquet column chunks are leaf columns, so a nested field (struct/map)
        // spans several — map each leaf back to its root field via the schema
        // descriptor rather than assuming leaf index == field index. Uses
        // checked addition and rejects a negative chunk size, so a corrupt or
        // adversarial footer yields `None` (no stat) instead of a panic or a
        // wrapped/bogus value. `None` also when the field has no leaf chunks.
        let column_size_bytes: Option<i64> = {
            let mut sum: i64 = 0;
            let mut seen = false;
            let mut ok = true;
            'rows: for rg in row_groups.iter() {
                for (leaf, chunk) in rg.columns().iter().enumerate() {
                    if parquet_schema.get_column_root_idx(leaf) != idx {
                        continue;
                    }
                    seen = true;
                    let cs = chunk.compressed_size();
                    match (cs >= 0).then(|| sum.checked_add(cs)).flatten() {
                        Some(v) => sum = v,
                        None => {
                            ok = false;
                            break 'rows;
                        },
                    }
                }
            }
            (ok && seen).then_some(sum)
        };

        out.push(ColumnStat {
            column_id,
            min_value: min_scalar.as_ref().and_then(stats_encode::encode_scalar),
            max_value: max_scalar.as_ref().and_then(stats_encode::encode_scalar),
            null_count,
            value_count,
            contains_nan,
            column_size_bytes,
        });
    }

    out
}

/// Reduce a per-row-group statistics array to a single bound: the smallest
/// element for `Ordering::Less` (min), the largest for `Ordering::Greater`
/// (max). Null entries mean "statistic unknown for that row group" and are
/// skipped. Returns `None` if every entry is null/empty.
fn reduce(array: &ArrayRef, keep: Ordering) -> Option<ScalarValue> {
    let mut acc: Option<ScalarValue> = None;
    for i in 0..array.len() {
        if array.is_null(i) {
            continue;
        }
        let Ok(value) = ScalarValue::try_from_array(array.as_ref(), i) else {
            continue;
        };
        acc = Some(match acc {
            None => value,
            Some(current) => {
                if value.partial_cmp(&current) == Some(keep) {
                    value
                } else {
                    current
                }
            },
        });
    }
    acc
}
