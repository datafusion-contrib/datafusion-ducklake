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
use arrow::datatypes::{DataType, Field, SchemaRef};
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

/// Fold a batch's NaN presence into a running accumulator over the Parquet leaf
/// columns of its first `n` columns (the catalog data columns; trailing
/// embedded rowid/snapshot columns are excluded by choosing
/// `n = column_ids.len()`), in the file's leaf order: a scalar column is one
/// leaf, a nested one has a leaf per scalar field below it. Each entry is
/// `Some(true)` if any batch had a NaN in that leaf, `Some(false)` for a float
/// leaf seen without one, or `None` for a leaf that is not a float.
pub fn accumulate_nan_flags(acc: &mut Vec<Option<bool>>, batch: &RecordBatch, n: usize) {
    let mut flags = Vec::new();
    for column in batch.columns().iter().take(n) {
        let rows: Vec<usize> = (0..column.len()).collect();
        visit_leaves(column.as_ref(), &rows, &mut |leaf, valid| {
            flags.push(leaf_has_nan(leaf, valid));
        });
    }
    if acc.len() < flags.len() {
        acc.resize(flags.len(), None);
    }
    for (slot, flag) in acc.iter_mut().zip(flags) {
        if let Some(has) = flag {
            *slot = Some(slot.unwrap_or(false) || has);
        }
    }
}

/// Whether a float leaf holds a NaN among the values at `valid`; `None` for a
/// leaf that is not a float.
fn leaf_has_nan(leaf: &dyn Array, valid: &[usize]) -> Option<bool> {
    match leaf.data_type() {
        DataType::Float32 => {
            let a = leaf.as_any().downcast_ref::<Float32Array>();
            a.map(|a| valid.iter().any(|r| a.value(*r).is_nan()))
        },
        DataType::Float64 => {
            let a = leaf.as_any().downcast_ref::<Float64Array>();
            a.map(|a| valid.iter().any(|r| a.value(*r).is_nan()))
        },
        _ => None,
    }
}

/// Call `visit` with each leaf of `array` and the indexes of its non-NULL
/// values among `rows`, in Parquet leaf order. A value below a NULL struct,
/// list or map is no value.
fn visit_leaves(array: &dyn Array, rows: &[usize], visit: &mut dyn FnMut(&dyn Array, &[usize])) {
    use arrow::array::AsArray;
    let valid: Vec<usize> = rows
        .iter()
        .copied()
        .filter(|r| array.is_valid(*r))
        .collect();
    let elements = |offsets: &[usize]| -> Vec<usize> {
        valid
            .iter()
            .flat_map(|r| offsets[*r]..offsets[*r + 1])
            .collect()
    };
    match array.data_type() {
        DataType::Struct(_) => {
            for child in array.as_struct().columns() {
                visit_leaves(child.as_ref(), &valid, visit);
            }
        },
        DataType::List(_) => {
            let list = array.as_list::<i32>();
            let offsets: Vec<usize> = list.offsets().iter().map(|o| *o as usize).collect();
            visit_leaves(list.values().as_ref(), &elements(&offsets), visit);
        },
        DataType::LargeList(_) => {
            let list = array.as_list::<i64>();
            let offsets: Vec<usize> = list.offsets().iter().map(|o| *o as usize).collect();
            visit_leaves(list.values().as_ref(), &elements(&offsets), visit);
        },
        DataType::FixedSizeList(_, size) => {
            let list = array.as_fixed_size_list();
            let size = *size as usize;
            let offsets: Vec<usize> = (0..=list.len()).map(|row| row * size).collect();
            visit_leaves(list.values().as_ref(), &elements(&offsets), visit);
        },
        DataType::Map(_, _) => {
            let map = array.as_map();
            let offsets: Vec<usize> = map.offsets().iter().map(|o| *o as usize).collect();
            let entries = elements(&offsets);
            visit_leaves(map.keys().as_ref(), &entries, visit);
            visit_leaves(map.values().as_ref(), &entries, visit);
        },
        _ => visit(array, &valid),
    }
}

/// The length past which DuckDB's Parquet writer truncates a string or blob
/// bound (`MAX_STRING_STATISTICS_SIZE`).
const MAX_STRING_STATISTICS_SIZE: usize = 256;

/// The bounds of one `VARCHAR` or `BLOB` leaf as DuckDB's Parquet writer
/// keeps them while writing a file (`StringStatisticsState`), so a file
/// written here records the statistics official records for the same rows.
///
/// A value over 256 bytes is truncated: a minimum to its first 256 bytes (for
/// `VARCHAR`, back to a character boundary), a maximum to the last byte in
/// its first 256 that can be incremented (below `0x7F` for `VARCHAR`, `0xFF`
/// for `BLOB`), incremented. A maximum with no such byte leaves the leaf
/// without bounds for the rest of the file. Values are compared with the
/// bounds kept so far, truncated ones included, as DuckDB compares them.
#[derive(Clone, Debug)]
pub struct StringBounds {
    blob: bool,
    has_stats: bool,
    failed: bool,
    min: Vec<u8>,
    max: Vec<u8>,
}

impl StringBounds {
    fn new(blob: bool) -> Self {
        Self {
            blob,
            has_stats: false,
            failed: false,
            min: Vec::new(),
            max: Vec::new(),
        }
    }

    fn update(&mut self, value: &[u8]) {
        if self.failed {
            return;
        }
        if !self.has_stats || value < self.min.as_slice() {
            self.min = if value.len() > MAX_STRING_STATISTICS_SIZE {
                truncate_min(value, self.blob)
            } else {
                value.to_vec()
            };
        }
        if !self.has_stats || value > self.max.as_slice() {
            if value.len() > MAX_STRING_STATISTICS_SIZE {
                let Some(max) = truncate_max(value, self.blob) else {
                    self.failed = true;
                    self.has_stats = false;
                    self.min.clear();
                    self.max.clear();
                    return;
                };
                self.max = max;
            } else {
                self.max = value.to_vec();
            }
        }
        self.has_stats = true;
    }

    /// The `(min, max)` the file records, as statistic values.
    fn bounds(&self) -> Option<(ScalarValue, ScalarValue)> {
        if !self.has_stats {
            return None;
        }
        if self.blob {
            return Some((
                ScalarValue::Binary(Some(self.min.clone())),
                ScalarValue::Binary(Some(self.max.clone())),
            ));
        }
        Some((
            ScalarValue::Utf8(Some(String::from_utf8(self.min.clone()).ok()?)),
            ScalarValue::Utf8(Some(String::from_utf8(self.max.clone()).ok()?)),
        ))
    }
}

/// DuckDB's `TruncateMin`: a prefix of `value` no longer than 256 bytes,
/// ending on a character boundary for `VARCHAR`.
fn truncate_min(value: &[u8], blob: bool) -> Vec<u8> {
    let mut size = MAX_STRING_STATISTICS_SIZE.min(value.len());
    if !blob {
        while size > 0 && size < value.len() && value[size] & 0xc0 == 0x80 {
            size -= 1;
        }
    }
    value[..size].to_vec()
}

/// DuckDB's `TryTruncateMax`: the first 256 bytes of `value` up to the last
/// one below the limit, that byte incremented; `None` when there is none.
fn truncate_max(value: &[u8], blob: bool) -> Option<Vec<u8>> {
    let limit = if blob {
        0xFF
    } else {
        0x7F
    };
    let end = MAX_STRING_STATISTICS_SIZE.min(value.len());
    let position = value[..end].iter().rposition(|byte| *byte < limit)?;
    let mut max = value[..=position].to_vec();
    max[position] += 1;
    Some(max)
}

/// [`StringBounds`] applied to an already reduced pair of footer bounds, for
/// a file whose values were not seen as it was written.
fn truncate_bound(value: ScalarValue, keep: Ordering) -> Option<ScalarValue> {
    let (bytes, blob) = match &value {
        ScalarValue::Utf8(Some(s))
        | ScalarValue::LargeUtf8(Some(s))
        | ScalarValue::Utf8View(Some(s)) => (s.as_bytes(), false),
        ScalarValue::Binary(Some(b))
        | ScalarValue::LargeBinary(Some(b))
        | ScalarValue::BinaryView(Some(b)) => (b.as_slice(), true),
        _ => return Some(value),
    };
    if bytes.len() <= MAX_STRING_STATISTICS_SIZE {
        return Some(value);
    }
    let truncated = if keep == Ordering::Less {
        truncate_min(bytes, blob)
    } else {
        truncate_max(bytes, blob)?
    };
    Some(if blob {
        ScalarValue::Binary(Some(truncated))
    } else {
        ScalarValue::Utf8(Some(String::from_utf8(truncated).ok()?))
    })
}

/// The bounds of one `FLOAT` or `DOUBLE` leaf as DuckDB's Parquet writer
/// keeps them: NaN left out, and a value replacing a bound only when it is
/// strictly beyond it, so among equal values (`-0.0` and `0.0`) the first
/// written stays.
#[derive(Clone, Debug)]
struct FloatBounds {
    single: bool,
    bounds: Option<(f64, f64)>,
}

impl FloatBounds {
    fn update(&mut self, value: f64) {
        if value.is_nan() {
            return;
        }
        match &mut self.bounds {
            None => self.bounds = Some((value, value)),
            Some((min, max)) => {
                if value < *min {
                    *min = value;
                }
                if value > *max {
                    *max = value;
                }
            },
        }
    }

    fn scalars(&self) -> Option<(ScalarValue, ScalarValue)> {
        let (min, max) = self.bounds?;
        Some(if self.single {
            (
                ScalarValue::Float32(Some(min as f32)),
                ScalarValue::Float32(Some(max as f32)),
            )
        } else {
            (
                ScalarValue::Float64(Some(min)),
                ScalarValue::Float64(Some(max)),
            )
        })
    }
}

/// The bounds a writer keeps for one leaf from the values it writes.
#[derive(Clone, Debug)]
enum LeafBounds {
    Text(StringBounds),
    Float(FloatBounds),
}

impl LeafBounds {
    fn bounds(&self) -> Option<(ScalarValue, ScalarValue)> {
        match self {
            Self::Text(bounds) => bounds.bounds(),
            Self::Float(bounds) => bounds.scalars(),
        }
    }
}

/// What a writer learns about each Parquet leaf from the values it writes,
/// beyond what the footer records as official does: NaN presence
/// ([`accumulate_nan_flags`]), DuckDB's string and blob bounds
/// ([`StringBounds`]), and float bounds that keep the sign of the zero
/// written first (the footer's minimum zero is always `-0.0`).
#[derive(Clone, Debug, Default)]
pub struct WrittenLeafStats {
    nan_flags: Vec<Option<bool>>,
    bounds: Vec<Option<LeafBounds>>,
}

impl WrittenLeafStats {
    /// Fold in a batch's first `n` columns, as [`accumulate_nan_flags`] does.
    pub fn accumulate(&mut self, batch: &RecordBatch, n: usize) {
        use arrow::array::AsArray;
        use arrow::datatypes::{Float32Type, Float64Type};
        accumulate_nan_flags(&mut self.nan_flags, batch, n);
        let mut leaf = 0usize;
        for column in batch.columns().iter().take(n) {
            let rows: Vec<usize> = (0..column.len()).collect();
            visit_leaves(column.as_ref(), &rows, &mut |array, valid| {
                if self.bounds.len() <= leaf {
                    self.bounds.resize(leaf + 1, None);
                }
                let slot = &mut self.bounds[leaf];
                leaf += 1;
                match array.data_type() {
                    DataType::Float32 | DataType::Float64 => {
                        let single = array.data_type() == &DataType::Float32;
                        let LeafBounds::Float(state) = slot.get_or_insert_with(|| {
                            LeafBounds::Float(FloatBounds {
                                single,
                                bounds: None,
                            })
                        }) else {
                            return;
                        };
                        for row in valid {
                            state.update(if single {
                                f64::from(array.as_primitive::<Float32Type>().value(*row))
                            } else {
                                array.as_primitive::<Float64Type>().value(*row)
                            });
                        }
                    },
                    DataType::Utf8
                    | DataType::LargeUtf8
                    | DataType::Utf8View
                    | DataType::Binary
                    | DataType::LargeBinary
                    | DataType::BinaryView => {
                        let blob = matches!(
                            array.data_type(),
                            DataType::Binary | DataType::LargeBinary | DataType::BinaryView
                        );
                        let LeafBounds::Text(state) =
                            slot.get_or_insert_with(|| LeafBounds::Text(StringBounds::new(blob)))
                        else {
                            return;
                        };
                        for row in valid {
                            let bytes: &[u8] = match array.data_type() {
                                DataType::Utf8 => array.as_string::<i32>().value(*row).as_bytes(),
                                DataType::LargeUtf8 => {
                                    array.as_string::<i64>().value(*row).as_bytes()
                                },
                                DataType::Utf8View => array.as_string_view().value(*row).as_bytes(),
                                DataType::Binary => array.as_binary::<i32>().value(*row),
                                DataType::LargeBinary => array.as_binary::<i64>().value(*row),
                                _ => array.as_binary_view().value(*row),
                            };
                            state.update(bytes);
                        }
                    },
                    _ => {},
                }
            });
        }
    }
}

/// The scalar fields below `field`, in Parquet leaf order: `field` itself when
/// it is not nested.
fn arrow_leaves(field: &Field) -> Vec<&Field> {
    match field.data_type() {
        DataType::Struct(children) => children.iter().flat_map(|c| arrow_leaves(c)).collect(),
        DataType::List(child) | DataType::LargeList(child) | DataType::FixedSizeList(child, _) => {
            arrow_leaves(child)
        },
        DataType::Map(entries, _) => arrow_leaves(entries),
        _ => vec![field],
    }
}

/// How a nested column's leaves are labelled with catalog column ids.
#[derive(Debug, Clone, Copy)]
enum NestedLeafIds<'a> {
    /// A nested column's leaves get no statistics.
    Skip,
    /// Each leaf takes the field id the file tags it with: a file this crate
    /// writes tags every field with its catalog column id.
    FileFieldIds,
    /// `ids[leaf]` is the catalog column id of Parquet leaf `leaf`; a leaf
    /// without one gets no statistics.
    Mapped(&'a [Option<i64>]),
}

/// Harvest per-column statistics for the Parquet file at `path`.
///
/// `column_ids` are the catalog `column_id`s for the file's columns, in physical
/// (written) order; `row_count` is the total rows in the file. Returns one
/// [`ColumnStat`] per scalar column whose stats could be read; a nested column
/// gets none, as the file's field ids need not be catalog column ids. String
/// and blob bounds come from the footer, truncated as [`StringBounds`]
/// truncates them; a writer that saw the values passes them through
/// [`collect_written_column_stats`] instead.
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
    collect(
        path,
        column_ids,
        row_count,
        contains_nan_flags,
        None,
        NestedLeafIds::Skip,
    )
}

/// [`collect_column_stats`] for a file whose values `leaves` saw as they were
/// written: its NaN flags, and string and blob bounds exactly as DuckDB's
/// writer keeps them.
pub fn collect_written_column_stats(
    path: &Path,
    column_ids: &[i64],
    row_count: i64,
    leaves: &WrittenLeafStats,
) -> Vec<ColumnStat> {
    collect(
        path,
        column_ids,
        row_count,
        &leaves.nan_flags,
        Some(&leaves.bounds),
        NestedLeafIds::Skip,
    )
}

/// [`collect_written_column_stats`] for a file this crate wrote, whose field
/// ids are the catalog column ids: a nested column also gets one
/// [`ColumnStat`] per scalar field below it, keyed by that field's id, as
/// official records them.
pub(crate) fn collect_own_written_column_stats(
    path: &Path,
    column_ids: &[i64],
    row_count: i64,
    leaves: &WrittenLeafStats,
) -> Vec<ColumnStat> {
    collect(
        path,
        column_ids,
        row_count,
        &leaves.nan_flags,
        Some(&leaves.bounds),
        NestedLeafIds::FileFieldIds,
    )
}

fn collect(
    path: &Path,
    column_ids: &[i64],
    row_count: i64,
    contains_nan_flags: &[Option<bool>],
    bounds: Option<&[Option<LeafBounds>]>,
    nested: NestedLeafIds<'_>,
) -> Vec<ColumnStat> {
    match try_collect(
        path,
        column_ids,
        row_count,
        contains_nan_flags,
        bounds,
        nested,
    ) {
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
    bounds: Option<&[Option<LeafBounds>]>,
    nested: NestedLeafIds<'_>,
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
    Ok(compute(
        &arrow_schema,
        &metadata,
        column_ids,
        row_count,
        contains_nan_flags,
        bounds,
        nested,
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
/// `column_ids[i]` labels file column `i`, and a nested column gets no
/// statistics: the file's field ids need not be catalog column ids
/// ([`compute_column_stats_with_leaf_ids`] labels its leaves).
/// `contains_nan_flags`, indexed by Parquet leaf, is empty when the caller could
/// not scan the data for NaN (footer-only paths): `contains_nan` is then
/// unknown. String and blob bounds are truncated as [`StringBounds`] truncates
/// them.
pub fn compute_column_stats(
    arrow_schema: &SchemaRef,
    metadata: &ParquetMetaData,
    column_ids: &[i64],
    row_count: i64,
    contains_nan_flags: &[Option<bool>],
) -> Vec<ColumnStat> {
    compute(
        arrow_schema,
        metadata,
        column_ids,
        row_count,
        contains_nan_flags,
        None,
        NestedLeafIds::Skip,
    )
}

/// [`compute_column_stats`] that also records one [`ColumnStat`] per scalar
/// field of a nested column, as official does, keyed by `leaf_column_ids`:
/// `leaf_column_ids[leaf]` is the catalog column id of Parquet leaf `leaf`
/// (the file's leaf order, which [`ParquetMetaData`]'s schema descriptor
/// numbers). A leaf of a nested column with no entry, or `None`, gets no
/// statistics; a scalar column is still labelled by `column_ids`.
pub fn compute_column_stats_with_leaf_ids(
    arrow_schema: &SchemaRef,
    metadata: &ParquetMetaData,
    column_ids: &[i64],
    leaf_column_ids: &[Option<i64>],
    row_count: i64,
    contains_nan_flags: &[Option<bool>],
) -> Vec<ColumnStat> {
    compute(
        arrow_schema,
        metadata,
        column_ids,
        row_count,
        contains_nan_flags,
        None,
        NestedLeafIds::Mapped(leaf_column_ids),
    )
}

fn compute(
    arrow_schema: &SchemaRef,
    metadata: &ParquetMetaData,
    column_ids: &[i64],
    row_count: i64,
    contains_nan_flags: &[Option<bool>],
    bounds: Option<&[Option<LeafBounds>]>,
    nested_ids: NestedLeafIds<'_>,
) -> Vec<ColumnStat> {
    let parquet_schema = metadata.file_metadata().schema_descr();
    let row_groups = metadata.row_groups();

    let fields = arrow_schema.fields();
    let n = fields.len().min(column_ids.len());
    let mut out = Vec::with_capacity(n);

    // Official records a row per Parquet leaf column: a scalar column's own,
    // and one per scalar field of a nested column, keyed by that field's
    // catalog id when `nested_ids` supplies one.
    let mut first_leaf_of_root: Vec<Option<usize>> = vec![None; fields.len()];
    for leaf in 0..parquet_schema.num_columns() {
        let root = parquet_schema.get_column_root_idx(leaf);
        if root >= n {
            continue;
        }
        let first_leaf = *first_leaf_of_root[root].get_or_insert(leaf);
        let root_field = &fields[root];
        let nested = matches!(
            root_field.data_type(),
            DataType::Struct(_)
                | DataType::List(_)
                | DataType::LargeList(_)
                | DataType::FixedSizeList(_, _)
                | DataType::Map(_, _)
        );
        let (column_id, converter) = if nested {
            let column_id = match nested_ids {
                NestedLeafIds::Skip => None,
                NestedLeafIds::FileFieldIds => {
                    let descriptor = parquet_schema.column(leaf);
                    let info = descriptor.self_type().get_basic_info();
                    info.has_id().then(|| i64::from(info.id()))
                },
                NestedLeafIds::Mapped(ids) => ids.get(leaf).copied().flatten(),
            };
            let Some(column_id) = column_id else {
                continue;
            };
            let leaves = arrow_leaves(root_field);
            let Some(leaf_field) = leaves.get(leaf - first_leaf) else {
                continue;
            };
            match StatisticsConverter::from_column_index(leaf, leaf_field, parquet_schema) {
                Ok(converter) => (column_id, converter),
                Err(e) => {
                    tracing::debug!(leaf, error = %e, "no statistics converter for leaf column");
                    continue;
                },
            }
        } else {
            match StatisticsConverter::try_new(root_field.name(), arrow_schema, parquet_schema) {
                Ok(converter) => (column_ids[root], converter),
                Err(e) => {
                    tracing::debug!(
                        column = root_field.name(),
                        error = %e,
                        "no statistics converter for column"
                    );
                    continue;
                },
            }
        };

        // Null counts: sum across row groups. Present for files we write
        // (statistics on by default). If unknown, both counts stay None
        // (DuckLake stores value_count/null_count NULL together). A leaf's
        // values are its column chunks' levels, NULL ones included.
        let (null_count, value_count) = match converter.row_group_null_counts(row_groups.iter()) {
            Ok(counts) => {
                let total: u64 = (0..counts.len())
                    .filter(|i| !counts.is_null(*i))
                    .map(|i| counts.value(i))
                    .sum();
                let null_count = i64::try_from(total).ok();
                let values = if nested {
                    row_groups
                        .iter()
                        .map(|rg| rg.column(leaf).num_values())
                        .sum::<i64>()
                } else {
                    row_count
                };
                let value_count = null_count.map(|nc| (values - nc).max(0));
                (null_count, value_count)
            },
            Err(_) => (None, None),
        };

        // Computed from the Arrow data at write time (the footer has no NaN
        // flag); None for non-float leaves.
        let contains_nan = contains_nan_flags.get(leaf).copied().flatten();

        // The footer's float bounds leave NaN out, as official's do; the file
        // records the NaN in `contains_nan`, which a reader must check before
        // trusting a float bound.
        //
        // A string, blob or float leaf whose values the writer saw takes the
        // bounds DuckDB's writer would have kept for them; otherwise the
        // footer's, strings truncated alike.
        let (min_scalar, max_scalar) = match bounds
            .and_then(|bounds| bounds.get(leaf))
            .and_then(Option::as_ref)
        {
            Some(bounds) => bounds.bounds().unzip(),
            None => {
                let min = converter
                    .row_group_mins(row_groups.iter())
                    .ok()
                    .and_then(|arr| reduce(&arr, Ordering::Less))
                    .and_then(|value| truncate_bound(value, Ordering::Less));
                let max = converter
                    .row_group_maxes(row_groups.iter())
                    .ok()
                    .and_then(|arr| reduce(&arr, Ordering::Greater));
                // A maximum that cannot be truncated leaves no bounds at all.
                match max {
                    Some(max) => match truncate_bound(max, Ordering::Greater) {
                        Some(max) => (min, Some(max)),
                        None => (None, None),
                    },
                    None => (min, None),
                }
            },
        };

        // Compressed on-disk size: official records the parquet
        // `total_compressed_size` of the column (`ducklake_add_data_files.cpp`)
        // — for a scalar column its chunk, for a nested leaf the leaf's chunk.
        // Read from the footer we already parsed, no extra I/O. Uses checked
        // addition and rejects a negative chunk size, so a corrupt or
        // adversarial footer yields `None` (no stat) instead of a panic or a
        // wrapped/bogus value.
        let column_size_bytes: Option<i64> = {
            let mut sum: i64 = 0;
            let mut ok = true;
            for rg in row_groups.iter() {
                let cs = rg.column(leaf).compressed_size();
                match (cs >= 0).then(|| sum.checked_add(cs)).flatten() {
                    Some(v) => sum = v,
                    None => {
                        ok = false;
                        break;
                    },
                }
            }
            (ok && !row_groups.is_empty()).then_some(sum)
        };

        out.push(ColumnStat {
            column_id,
            min_value: min_scalar.as_ref().and_then(stats_encode::encode_file_stat),
            max_value: max_scalar.as_ref().and_then(stats_encode::encode_file_stat),
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

#[cfg(test)]
mod tests {
    use super::*;

    fn bounds(values: &[&[u8]], blob: bool) -> Option<(Vec<u8>, Vec<u8>)> {
        let mut state = StringBounds::new(blob);
        for value in values {
            state.update(value);
        }
        state.has_stats.then_some((state.min, state.max))
    }

    #[test]
    fn short_values_are_kept_whole() {
        let long = vec![b'b'; 256];
        assert_eq!(
            bounds(&[b"a", &long, b"0"], false),
            Some((b"0".to_vec(), long.clone()))
        );
    }

    #[test]
    fn long_values_truncate_as_duckdb_does() {
        // 254 ASCII bytes then a two-byte character straddling byte 256: the
        // minimum ends before the character.
        let mut varchar = vec![b'a'; 255];
        varchar.extend("é".as_bytes());
        varchar.extend(b"zz");
        let (min, max) = bounds(&[&varchar], false).unwrap();
        assert_eq!(min, vec![b'a'; 255]);
        // The maximum ends at the last ASCII byte in the first 256, incremented.
        let mut expected = vec![b'a'; 254];
        expected.push(b'b');
        assert_eq!(max, expected);

        // A blob increments any byte below 0xFF and cuts at 256 bytes.
        let mut blob = vec![0x10; 300];
        blob[255] = 0xFF;
        let (min, max) = bounds(&[&blob], true).unwrap();
        assert_eq!(min, blob[..256].to_vec());
        let mut expected = vec![0x10; 254];
        expected.push(0x11);
        assert_eq!(max, expected);
    }

    #[test]
    fn a_maximum_that_cannot_be_truncated_drops_the_bounds_for_good() {
        let high = "é".repeat(200);
        assert_eq!(bounds(&[b"a", high.as_bytes()], false), None);
        // Later values do not bring bounds back.
        assert_eq!(
            bounds(&[high.as_bytes(), "ê".as_bytes(), b"a"], false),
            None
        );
        assert_eq!(bounds(&[&[0xFF; 300]], true), None);
        // The same value after a larger short one is never a maximum.
        assert_eq!(
            bounds(&["ê".as_bytes(), high.as_bytes()], false),
            Some((high.as_bytes()[..256].to_vec(), "ê".as_bytes().to_vec()))
        );
    }

    #[test]
    fn float_bounds_keep_the_first_zero_and_skip_nan() {
        let bounds = |values: &[f64]| {
            let mut state = FloatBounds {
                single: false,
                bounds: None,
            };
            for value in values {
                state.update(*value);
            }
            state
                .bounds
                .map(|(min, max)| (min.to_bits(), max.to_bits()))
        };
        assert_eq!(bounds(&[0.0]), Some((0.0f64.to_bits(), 0.0f64.to_bits())));
        assert_eq!(
            bounds(&[-0.0, 0.0, f64::NAN]),
            Some(((-0.0f64).to_bits(), (-0.0f64).to_bits()))
        );
        assert_eq!(bounds(&[f64::NAN]), None);
        assert_eq!(
            bounds(&[1.0, -2.0, f64::NAN]),
            Some(((-2.0f64).to_bits(), 1.0f64.to_bits()))
        );
    }

    #[test]
    fn footer_bounds_truncate_alike() {
        let long = "b".repeat(300);
        assert_eq!(
            truncate_bound(ScalarValue::Utf8(Some(long.clone())), Ordering::Less),
            Some(ScalarValue::Utf8(Some("b".repeat(256))))
        );
        assert_eq!(
            truncate_bound(ScalarValue::Utf8(Some(long)), Ordering::Greater),
            Some(ScalarValue::Utf8(Some(format!("{}c", "b".repeat(255)))))
        );
        assert_eq!(
            truncate_bound(
                ScalarValue::Binary(Some(vec![0xFF; 300])),
                Ordering::Greater
            ),
            None
        );
    }
}
