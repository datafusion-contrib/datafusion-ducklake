//! Per-file parquet encoding choices for data files, following official
//! DuckLake's writer (DuckDB's parquet writer).
//!
//! DuckDB decides dictionary encoding per column chunk after buffering the row
//! group: it counts the chunk's distinct values and keeps a dictionary only when
//! they number at most a fifth of the row group's rows, and never for a column
//! with no values. A chunk that keeps its dictionary also gets a bloom filter
//! built from it (false-positive rate 0.01); a chunk without one gets neither.
//!
//! arrow-rs applies one set of writer properties to a whole file. So a data file
//! buffers its first row group — up to the row-group row cap, or the byte cap
//! when one is set, or the end of the write — counts each column exactly as
//! DuckDB does, and fixes the file's properties from that count:
//!
//! - a column over the limit, or with no values, is written without a dictionary
//!   and without a bloom filter;
//! - any other column keeps its dictionary and gets a bloom filter at DuckDB's
//!   rate. Its dictionary byte limit is sized for the limit of a full row group,
//!   so a later row group that outgrows it falls back as DuckDB's would drop its
//!   dictionary; arrow-rs keeps the pages written before the fallback
//!   dictionary-encoded, and the file's bloom filter stays on.
//!
//! The file's first row group is decided exactly as DuckDB decides it. A later
//! row group follows the first one's choice, apart from that fallback.
//!
//! Only top-level columns are tuned. Nested columns, booleans, intervals and
//! dictionary-typed arrays keep arrow-rs's defaults, as do fixed-length byte
//! array columns under Parquet V1, where arrow-rs writes no dictionary for them.

use std::collections::HashSet;
use std::io::Write;

use arrow::array::{Array, ArrayRef, AsArray};
use arrow::datatypes::{DataType, SchemaRef};
use arrow::record_batch::RecordBatch;
use arrow::row::{RowConverter, Rows, SortField};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::{DEFAULT_MAX_ROW_GROUP_ROW_COUNT, WriterProperties, WriterVersion};
use parquet::schema::types::ColumnPath;

use crate::Result;

/// DuckDB keeps a column chunk's dictionary only while its distinct values number
/// at most `row group rows / DICTIONARY_ROWS_PER_ENTRY`.
const DICTIONARY_ROWS_PER_ENTRY: usize = 5;

/// DuckDB's default bloom filter false-positive rate.
const BLOOM_FILTER_FPP: f64 = 0.01;

/// Bytes a dictionary entry of `data_type` occupies once written, taking the
/// longest value in `arrays` for a variable-length type, or `None` when the type
/// is not tuned.
fn dictionary_entry_width(
    data_type: &DataType,
    arrays: &[&ArrayRef],
    writer_version: WriterVersion,
) -> Option<usize> {
    let fixed_len_v1 = writer_version == WriterVersion::PARQUET_1_0;
    match data_type {
        DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::Float32
        | DataType::Date32
        | DataType::Time32(_) => Some(4),
        DataType::Int64
        | DataType::UInt64
        | DataType::Float64
        | DataType::Date64
        | DataType::Time64(_)
        | DataType::Timestamp(_, _)
        | DataType::Duration(_) => Some(8),
        DataType::Decimal32(precision, _)
        | DataType::Decimal64(precision, _)
        | DataType::Decimal128(precision, _)
        | DataType::Decimal256(precision, _) => match precision {
            // arrow-rs writes precision 2..=9 as INT32 and 1 or 10..=18 as INT64.
            2..=9 => Some(4),
            0..=18 => Some(8),
            // Written as a fixed-length byte array.
            _ if fixed_len_v1 => None,
            _ => Some(usize::from(*precision).div_ceil(2)),
        },
        DataType::FixedSizeBinary(width) if !fixed_len_v1 => usize::try_from(*width).ok(),
        DataType::Utf8
        | DataType::LargeUtf8
        | DataType::Utf8View
        | DataType::Binary
        | DataType::LargeBinary
        | DataType::BinaryView => Some(arrays.iter().filter_map(|a| max_len(a)).max()? + 4),
        _ => None,
    }
}

/// Longest non-null value, in bytes, of a string or binary array.
fn max_len(array: &ArrayRef) -> Option<usize> {
    let longest = match array.data_type() {
        DataType::Utf8 => array
            .as_string::<i32>()
            .iter()
            .flatten()
            .map(str::len)
            .max(),
        DataType::LargeUtf8 => array
            .as_string::<i64>()
            .iter()
            .flatten()
            .map(str::len)
            .max(),
        DataType::Utf8View => array.as_string_view().iter().flatten().map(str::len).max(),
        DataType::Binary => array
            .as_binary::<i32>()
            .iter()
            .flatten()
            .map(<[u8]>::len)
            .max(),
        DataType::LargeBinary => array
            .as_binary::<i64>()
            .iter()
            .flatten()
            .map(<[u8]>::len)
            .max(),
        DataType::BinaryView => array
            .as_binary_view()
            .iter()
            .flatten()
            .map(<[u8]>::len)
            .max(),
        _ => return None,
    };
    Some(longest.unwrap_or(0))
}

/// Distinct non-null values across `arrays`, counted up to `limit + 1`: the
/// count is exact at or below `limit`, and `limit + 1` means "over the limit".
fn distinct_up_to(arrays: &[&ArrayRef], limit: usize) -> Result<usize> {
    let Some(first) = arrays.first() else {
        return Ok(0);
    };
    let converter = RowConverter::new(vec![SortField::new(first.data_type().clone())])?;
    let rows: Vec<Rows> = arrays
        .iter()
        .map(|array| converter.convert_columns(std::slice::from_ref(*array)))
        .collect::<std::result::Result<_, _>>()?;
    let mut distinct = HashSet::new();
    for (array, rows) in arrays.iter().zip(&rows) {
        for index in 0..array.len() {
            if array.is_valid(index) && distinct.insert(rows.row(index)) && distinct.len() > limit {
                return Ok(limit + 1);
            }
        }
    }
    Ok(distinct.len())
}

/// The rows `base` puts in a full row group.
fn row_group_rows(base: &WriterProperties) -> usize {
    base.max_row_group_row_count()
        .unwrap_or(DEFAULT_MAX_ROW_GROUP_ROW_COUNT)
}

/// `base` with each tuned top-level column's dictionary and bloom filter set from
/// the file's first row group: the first [`row_group_rows`] rows of `buffered`, or
/// all of them when there are fewer. Columns are matched by position; the batches
/// carry the file's full parquet schema.
pub(crate) fn props_for_row_group(
    base: &WriterProperties,
    buffered: &[RecordBatch],
) -> Result<WriterProperties> {
    let full_rows = row_group_rows(base);
    let mut row_group = Vec::with_capacity(buffered.len());
    let mut rows = 0;
    for batch in buffered {
        if rows == full_rows {
            break;
        }
        let take = batch.num_rows().min(full_rows - rows);
        row_group.push(batch.slice(0, take));
        rows += take;
    }
    let Some(first) = row_group.first() else {
        return Ok(base.clone());
    };
    // The limit this row group is judged by, and the one a full row group of the
    // file would be judged by.
    let limit = rows / DICTIONARY_ROWS_PER_ENTRY;
    let full_limit = full_rows / DICTIONARY_ROWS_PER_ENTRY;
    let writer_version = base.writer_version();
    let schema = first.schema();
    let mut builder = base.clone().into_builder();
    for (index, field) in schema.fields().iter().enumerate() {
        let columns: Vec<&ArrayRef> = row_group.iter().map(|batch| batch.column(index)).collect();
        let Some(width) = dictionary_entry_width(field.data_type(), &columns, writer_version)
        else {
            continue;
        };
        let path = ColumnPath::new(vec![field.name().clone()]);
        let distinct = distinct_up_to(&columns, limit)?;
        if distinct == 0 || distinct > limit {
            builder = builder
                .set_column_dictionary_enabled(path.clone(), false)
                .set_column_bloom_filter_enabled(path, false);
        } else {
            let entries = full_limit.max(limit).max(1);
            builder = builder
                .set_column_dictionary_enabled(path.clone(), true)
                .set_column_dictionary_page_size_limit(
                    path.clone(),
                    entries.saturating_mul(width.max(1)),
                )
                .set_column_bloom_filter_enabled(path.clone(), true)
                .set_column_bloom_filter_fpp(path.clone(), BLOOM_FILTER_FPP)
                .set_column_bloom_filter_max_ndv(path, entries as u64);
        }
    }
    Ok(builder.build())
}

/// The error for a writer used after a failed open: its sink went with the failure.
fn closed() -> crate::error::DuckLakeError {
    crate::error::DuckLakeError::Internal(
        "parquet writer is closed: opening its file failed earlier".to_string(),
    )
}

/// Bytes `batch`'s buffers hold for its own rows. A slice of a larger batch counts
/// only its share, where [`RecordBatch::get_array_memory_size`] would count the
/// whole parent allocation.
fn sliced_memory_size(batch: &RecordBatch) -> usize {
    batch
        .columns()
        .iter()
        .map(|column| {
            column
                .to_data()
                .get_slice_memory_size()
                .unwrap_or_else(|_| column.get_array_memory_size())
        })
        .sum()
}

/// A parquet writer that buffers its file's first row group before opening, so
/// the file's encoding can be chosen from it with [`props_for_row_group`].
///
/// It reports no bytes while buffering, so a caller that rolls files by size
/// cannot roll before the first row group is complete — as DuckDB, which rotates
/// only after writing a row group. A writer finished while still buffering opens
/// with the buffered rows (or, with none, the base properties, so a zero-row file
/// is still a valid parquet file).
#[derive(Debug)]
pub(crate) struct RowGroupSampledWriter<W: Write + Send> {
    pending: Option<(W, SchemaRef, WriterProperties)>,
    buffered: Vec<RecordBatch>,
    buffered_rows: usize,
    buffered_bytes: usize,
    writer: Option<ArrowWriter<W>>,
}

impl<W: Write + Send> RowGroupSampledWriter<W> {
    pub(crate) fn new(sink: W, schema: SchemaRef, base: WriterProperties) -> Self {
        Self {
            pending: Some((sink, schema, base)),
            buffered: Vec::new(),
            buffered_rows: 0,
            buffered_bytes: 0,
            writer: None,
        }
    }

    /// Write `batch`, which must carry the file's parquet schema. Buffers it while
    /// the first row group is incomplete, and opens the file once it is.
    pub(crate) fn write(&mut self, batch: &RecordBatch) -> Result<()> {
        if let Some(writer) = &mut self.writer {
            writer.write(batch)?;
            return Ok(());
        }
        if batch.num_rows() == 0 {
            return Ok(());
        }
        let Some((_, _, base)) = &self.pending else {
            return Err(closed());
        };
        let full_rows = row_group_rows(base);
        let full_bytes = base.max_row_group_bytes();
        self.buffered_rows += batch.num_rows();
        self.buffered_bytes += sliced_memory_size(batch);
        self.buffered.push(batch.clone());
        if self.buffered_rows >= full_rows || full_bytes.is_some_and(|b| self.buffered_bytes >= b) {
            self.open()?;
        }
        Ok(())
    }

    /// Open the file with properties chosen from the buffered rows, and write them.
    fn open(&mut self) -> Result<()> {
        let (sink, schema, base) = self.pending.take().ok_or_else(closed)?;
        let props = props_for_row_group(&base, &self.buffered)?;
        let mut writer = ArrowWriter::try_new(sink, schema, Some(props))?;
        for batch in std::mem::take(&mut self.buffered) {
            writer.write(&batch)?;
        }
        self.writer = Some(writer);
        Ok(())
    }

    /// Bytes flushed to the sink so far; zero while buffering.
    pub(crate) fn bytes_written(&self) -> usize {
        self.writer.as_ref().map_or(0, ArrowWriter::bytes_written)
    }

    /// Estimated encoded size of the in-progress row group; zero while buffering.
    pub(crate) fn in_progress_size(&self) -> usize {
        self.writer
            .as_ref()
            .map_or(0, ArrowWriter::in_progress_size)
    }

    /// Write the footer and return the sink.
    pub(crate) fn into_inner(mut self) -> Result<W> {
        if self.writer.is_none() {
            self.open()?;
        }
        let writer = self.writer.take().expect("writer opened above");
        Ok(writer.into_inner()?)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{Float64Array, Int64Array, StringArray};
    use arrow::datatypes::{Field, Schema};
    use bytes::Bytes;
    use parquet::basic::Encoding;
    use parquet::file::reader::{FileReader, SerializedFileReader};

    use super::*;

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("v", DataType::Float64, false),
            Field::new("s", DataType::Utf8, false),
            Field::new("status", DataType::Utf8, true),
            Field::new("mid", DataType::Int64, false),
            Field::new("empty", DataType::Int64, true),
        ]))
    }

    /// Rows `lo..hi`: `id`, `v`, `s` unique; `status` four values with NULLs;
    /// `mid` 2,000 values; `empty` all NULL.
    fn batch(lo: i64, hi: i64) -> RecordBatch {
        RecordBatch::try_new(
            schema(),
            vec![
                Arc::new(Int64Array::from_iter_values(lo..hi)),
                Arc::new(Float64Array::from_iter_values(
                    (lo..hi).map(|i| i as f64 * 0.5),
                )),
                Arc::new(StringArray::from_iter_values(
                    (lo..hi).map(|i| format!("user_{i}")),
                )),
                Arc::new(StringArray::from_iter(
                    (lo..hi).map(|i| (i % 7 != 0).then(|| format!("status_{}", i % 4))),
                )),
                Arc::new(Int64Array::from_iter_values((lo..hi).map(|i| i % 2_000))),
                Arc::new(Int64Array::from(vec![None::<i64>; (hi - lo) as usize])),
            ],
        )
        .unwrap()
    }

    /// Write `batches` through [`RowGroupSampledWriter`] and report, per column of
    /// the first row group, whether it carries a dictionary and a bloom filter.
    fn layout(batches: &[RecordBatch], row_group_rows: usize) -> Vec<(String, bool, bool)> {
        let base = WriterProperties::builder()
            .set_max_row_group_row_count(Some(row_group_rows))
            .build();
        let mut writer = RowGroupSampledWriter::new(Vec::new(), schema(), base);
        for batch in batches {
            writer.write(batch).unwrap();
        }
        let bytes = Bytes::from(writer.into_inner().unwrap());
        let reader = SerializedFileReader::new(bytes).unwrap();
        reader
            .metadata()
            .row_group(0)
            .columns()
            .iter()
            .map(|column| {
                let dictionary = column.dictionary_page_offset().is_some()
                    || column
                        .encodings()
                        .any(|encoding| encoding == Encoding::RLE_DICTIONARY);
                (
                    column.column_path().string(),
                    dictionary,
                    column.bloom_filter_offset().is_some(),
                )
            })
            .collect()
    }

    fn assert_repeating(layout: &[(String, bool, bool)], repeating: &[&str]) {
        for (name, dictionary, bloom) in layout {
            let expected = repeating.contains(&name.as_str());
            assert_eq!(*dictionary, expected, "{name}: dictionary");
            assert_eq!(*bloom, expected, "{name}: bloom filter");
        }
    }

    #[test]
    fn a_row_group_is_counted_the_way_duckdb_counts_it() {
        // 20,000 rows, limit 4,000: `mid`'s 2,000 values stay under it.
        let batches: Vec<_> = (0..20).map(|b| batch(b * 1_000, (b + 1) * 1_000)).collect();
        assert_repeating(&layout(&batches, 20_000), &["status", "mid"]);
    }

    #[test]
    fn a_small_first_batch_does_not_decide_the_file() {
        // A first batch of one row is buffered with the rest of the row group, so
        // the repeating columns still keep their dictionaries.
        let mut batches = vec![batch(0, 1)];
        batches.extend((0..20).map(|b| batch(1 + b * 1_000, 1 + (b + 1) * 1_000)));
        assert_repeating(&layout(&batches, 20_001), &["status", "mid"]);
    }

    #[test]
    fn a_file_smaller_than_a_row_group_is_judged_by_its_own_rows() {
        // 5,000 rows, limit 1,000: `mid`'s 2,000 values are now over it.
        let batches: Vec<_> = (0..5).map(|b| batch(b * 1_000, (b + 1) * 1_000)).collect();
        assert_repeating(&layout(&batches, 122_880), &["status"]);
    }

    #[test]
    fn one_oversized_batch_is_judged_by_its_first_row_group_only() {
        // One 20,000-row batch against 2,000-row row groups: the first row group
        // holds all 2,000 `mid` values, over its limit of 400, although the whole
        // batch (2,000 values against a limit of 4,000) would stay under.
        assert_repeating(&layout(&[batch(0, 20_000)], 2_000), &["status"]);
    }

    #[test]
    fn a_slice_counts_only_its_own_rows_toward_the_byte_cap() {
        let parent = batch(0, 20_000);
        let slice = parent.slice(0, 1_000);
        assert!(sliced_memory_size(&slice) * 10 < parent.get_array_memory_size());
    }

    #[test]
    fn a_zero_row_writer_still_writes_a_valid_file() {
        let writer =
            RowGroupSampledWriter::new(Vec::new(), schema(), WriterProperties::builder().build());
        let bytes = Bytes::from(writer.into_inner().unwrap());
        let reader = SerializedFileReader::new(bytes).unwrap();
        assert_eq!(reader.metadata().file_metadata().num_rows(), 0);
        assert_eq!(
            reader
                .metadata()
                .file_metadata()
                .schema_descr()
                .num_columns(),
            6
        );
    }

    #[test]
    fn no_bytes_are_reported_before_the_first_row_group_is_complete() {
        let base = WriterProperties::builder()
            .set_max_row_group_row_count(Some(2_000))
            .build();
        let mut writer = RowGroupSampledWriter::new(Vec::new(), schema(), base);
        writer.write(&batch(0, 1_000)).unwrap();
        assert_eq!(writer.bytes_written() + writer.in_progress_size(), 0);
        writer.write(&batch(1_000, 2_000)).unwrap();
        assert!(writer.bytes_written() + writer.in_progress_size() > 0);
    }
}
