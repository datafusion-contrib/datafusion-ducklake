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
//! A file whose row groups are encoded on several threads
//! ([`crate::parallel_encode`]) instead decides every row group from its own
//! rows, as DuckDB decides each column chunk: its row groups are encoded
//! before it is known which file they land in.
//!
//! Only top-level columns are tuned. Nested columns, booleans, intervals and
//! dictionary-typed arrays keep arrow-rs's defaults, as do fixed-length byte
//! array columns under Parquet V1, where arrow-rs writes no dictionary for them.
//! A column whose dictionary the caller set ([`DataFileProperties::dictionary_set`])
//! is not tuned either: it keeps the caller's setting and gets no bloom filter.
//!
//! A column whose bloom filter the caller set
//! ([`DataFileProperties::bloom_filter_set`]) keeps that setting, whatever its
//! dictionary: its dictionary is still decided as above, but the decision no
//! longer turns its filter on or off. The caller's filter is sized for the rows
//! of the row group being decided, not for a full row group: arrow-rs holds a
//! filter's whole starting size in memory until the file closes, and folding it
//! down to the values written frees none of it. So a write of a few rows over
//! many such columns allocates a few bytes per column, not a full row group's
//! filter each.

use std::collections::HashSet;
use std::io::Write;
use std::sync::Arc;

use arrow::array::{Array, ArrayData, ArrayRef, AsArray, BinaryViewArray, StringViewArray};
use arrow::datatypes::{DataType, SchemaRef};
use arrow::record_batch::RecordBatch;
use arrow::row::{RowConverter, SortField};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::{DEFAULT_MAX_ROW_GROUP_ROW_COUNT, WriterProperties, WriterVersion};
use parquet::schema::types::ColumnPath;

use crate::Result;
use crate::parallel_encode::{EncodePool, ParallelFileWriter, RowGroupQueue};
use crate::stats_collect::WrittenLeafStats;

/// DuckDB keeps a column chunk's dictionary only while its distinct values number
/// at most `row group rows / DICTIONARY_ROWS_PER_ENTRY`.
const DICTIONARY_ROWS_PER_ENTRY: usize = 5;

/// DuckDB's default bloom filter false-positive rate.
pub(crate) const BLOOM_FILTER_FPP: f64 = 0.01;

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
    // Each array is converted when its turn comes, so a column that goes over
    // the limit early converts none of the rest of the row group. What is kept
    // across arrays is a copy of each distinct value's row, at most `limit + 1`.
    let mut distinct: HashSet<Box<[u8]>> = HashSet::new();
    for array in arrays {
        let rows = converter.convert_columns(std::slice::from_ref(*array))?;
        for index in (0..array.len()).filter(|index| array.is_valid(*index)) {
            let row = rows.row(index);
            if !distinct.contains(row.as_ref()) {
                distinct.insert(row.as_ref().into());
                if distinct.len() > limit {
                    return Ok(limit + 1);
                }
            }
        }
    }
    Ok(distinct.len())
}

/// The writer properties a data file starts from, the top-level columns whose
/// dictionary or bloom filter the caller set, which [`props_for_row_group`]
/// leaves alone, and the pool that encodes the file's row groups.
#[derive(Clone, Debug)]
pub(crate) struct DataFileProperties {
    pub(crate) props: WriterProperties,
    pub(crate) dictionary_set: Arc<HashSet<String>>,
    /// The leaf columns whose bloom filter the caller turned on or off in
    /// [`Self::props`].
    pub(crate) bloom_filter_set: Arc<HashSet<ColumnPath>>,
    /// The threads that encode the file's row groups, shared by every file of
    /// the write ([`crate::parallel_encode`]); `None` encodes on the writing
    /// thread.
    pub(crate) pool: Option<Arc<EncodePool>>,
}

impl From<WriterProperties> for DataFileProperties {
    fn from(props: WriterProperties) -> Self {
        Self {
            props,
            dictionary_set: Arc::default(),
            bloom_filter_set: Arc::default(),
            pool: None,
        }
    }
}

/// The rows `base` puts in a full row group.
fn row_group_rows(base: &WriterProperties) -> usize {
    base.max_row_group_row_count()
        .unwrap_or(DEFAULT_MAX_ROW_GROUP_ROW_COUNT)
}

/// `base` with each tuned top-level column's dictionary and bloom filter set from
/// the file's first row group: the first [`row_group_rows`] rows of `buffered`, or
/// all of them when there are fewer. A column whose bloom filter the caller set
/// gets its dictionary only, and its filter, when on, is sized for those rows.
/// Columns are matched by position; the batches carry the file's full parquet
/// schema.
pub(crate) fn props_for_row_group(
    base: &DataFileProperties,
    buffered: &[RecordBatch],
) -> Result<WriterProperties> {
    let dictionary_set = &base.dictionary_set;
    let bloom_filter_set = &base.bloom_filter_set;
    let base = &base.props;
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
    // A caller's filter holds at most this row group's values, so it is sized
    // for them: left to arrow-rs, it would start at a full row group's size.
    for path in bloom_filter_set.iter() {
        if base.bloom_filter_properties(path).is_some() {
            builder = builder.set_column_bloom_filter_max_ndv(path.clone(), rows.max(1) as u64);
        }
    }
    for (index, field) in schema.fields().iter().enumerate() {
        if dictionary_set.contains(field.name()) {
            continue;
        }
        let columns: Vec<&ArrayRef> = row_group.iter().map(|batch| batch.column(index)).collect();
        let Some(width) = dictionary_entry_width(field.data_type(), &columns, writer_version)
        else {
            continue;
        };
        let path = ColumnPath::new(vec![field.name().clone()]);
        let caller_bloom_filter = bloom_filter_set.contains(&path);
        let distinct = distinct_up_to(&columns, limit)?;
        if distinct == 0 || distinct > limit {
            builder = builder.set_column_dictionary_enabled(path.clone(), false);
            if !caller_bloom_filter {
                builder = builder.set_column_bloom_filter_enabled(path, false);
            }
        } else {
            let entries = full_limit.max(limit).max(1);
            builder = builder
                .set_column_dictionary_enabled(path.clone(), true)
                .set_column_dictionary_page_size_limit(
                    path.clone(),
                    entries.saturating_mul(width.max(1)),
                );
            if !caller_bloom_filter {
                builder = builder
                    .set_column_bloom_filter_enabled(path.clone(), true)
                    .set_column_bloom_filter_fpp(path.clone(), BLOOM_FILTER_FPP)
                    .set_column_bloom_filter_max_ndv(path, entries as u64);
            }
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
pub(crate) fn sliced_memory_size(batch: &RecordBatch) -> usize {
    batch
        .columns()
        .iter()
        .map(|column| {
            let data = column.to_data();
            match data.get_slice_memory_size() {
                Ok(bytes) => bytes + view_value_bytes(&data),
                // The whole allocation, data buffers included.
                Err(_) => column.get_array_memory_size(),
            }
        })
        .sum()
}

/// Bytes the values of `data`'s own rows hold in the data buffers of every view
/// array in it, at any depth.
///
/// `get_slice_memory_size` counts a `Utf8View` or `BinaryView` array as its
/// 16-byte views alone. A value longer than 12 bytes lives in the array's data
/// buffers instead, so text that DataFusion decodes as views would otherwise
/// count as 16 bytes a row, however wide it is.
fn view_value_bytes(data: &ArrayData) -> usize {
    let own = match data.data_type() {
        DataType::Utf8View => StringViewArray::from(data.clone()).total_buffer_bytes_used(),
        DataType::BinaryView => BinaryViewArray::from(data.clone()).total_buffer_bytes_used(),
        _ => 0,
    };
    own + data
        .child_data()
        .iter()
        .map(view_value_bytes)
        .sum::<usize>()
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
    pending: Option<(W, SchemaRef, DataFileProperties)>,
    buffered: Vec<RecordBatch>,
    buffered_rows: usize,
    buffered_bytes: usize,
    writer: Option<ArrowWriter<W>>,
}

impl<W: Write + Send> RowGroupSampledWriter<W> {
    pub(crate) fn new(sink: W, schema: SchemaRef, base: impl Into<DataFileProperties>) -> Self {
        Self {
            pending: Some((sink, schema, base.into())),
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
        let full_rows = row_group_rows(&base.props);
        let full_bytes = base.props.max_row_group_bytes();
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

/// One data file and the [`WrittenLeafStats`] of every row written to it:
/// encoded on the writing thread by a [`RowGroupSampledWriter`], or row group
/// by row group on the pool of its properties ([`crate::parallel_encode`]),
/// which notes the stats on the pool's threads too.
#[derive(Debug)]
pub(crate) enum DataFileWriter<W: Write + Send> {
    Sampled {
        writer: Box<RowGroupSampledWriter<W>>,
        leaf_stats: WrittenLeafStats,
        stats_columns: usize,
    },
    RowGroups(Box<ParallelFileWriter<W>>),
}

impl<W: Write + Send> DataFileWriter<W> {
    /// A file of `schema` on `sink`, whose leaf stats cover the first
    /// `stats_columns` columns.
    pub(crate) fn try_new(
        sink: W,
        schema: SchemaRef,
        props: DataFileProperties,
        stats_columns: usize,
    ) -> Result<Self> {
        if let Some(queue) = RowGroupQueue::try_new(schema.clone(), &props, stats_columns)? {
            return Ok(Self::RowGroups(Box::new(ParallelFileWriter::try_new(
                sink,
                schema,
                &props.props,
                queue,
            )?)));
        }
        Ok(Self::Sampled {
            writer: Box::new(RowGroupSampledWriter::new(sink, schema, props)),
            leaf_stats: WrittenLeafStats::default(),
            stats_columns,
        })
    }

    /// Write `batch`, which must carry the file's parquet schema.
    pub(crate) fn write(&mut self, batch: &RecordBatch) -> Result<()> {
        match self {
            Self::Sampled {
                writer,
                leaf_stats,
                stats_columns,
            } => {
                leaf_stats.accumulate(batch, *stats_columns);
                writer.write(batch)
            },
            Self::RowGroups(writer) => writer.write(batch),
        }
    }

    /// Write the footer and return the sink and the file's leaf stats.
    pub(crate) fn into_inner(self) -> Result<(W, WrittenLeafStats)> {
        match self {
            Self::Sampled {
                writer,
                leaf_stats,
                ..
            } => Ok((writer.into_inner()?, leaf_stats)),
            Self::RowGroups(writer) => writer.into_inner(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{Float64Array, Int64Array, MapArray, StringArray, StructArray};
    use arrow::buffer::OffsetBuffer;
    use arrow::datatypes::{Field, Fields, Schema};
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

    /// Write `batches` through [`RowGroupSampledWriter`] from `base`.
    fn write_with(batches: &[RecordBatch], base: impl Into<DataFileProperties>) -> Bytes {
        let mut writer = RowGroupSampledWriter::new(Vec::new(), schema(), base);
        for batch in batches {
            writer.write(batch).unwrap();
        }
        Bytes::from(writer.into_inner().unwrap())
    }

    /// Write `batches` through [`RowGroupSampledWriter`] and report, per column of
    /// the first row group, whether it carries a dictionary and a bloom filter.
    fn layout(batches: &[RecordBatch], row_group_rows: usize) -> Vec<(String, bool, bool)> {
        let base = WriterProperties::builder()
            .set_max_row_group_row_count(Some(row_group_rows))
            .build();
        layout_of(write_with(batches, base))
    }

    /// Per column of `bytes`' first row group, whether it carries a dictionary
    /// and a bloom filter.
    fn layout_of(bytes: Bytes) -> Vec<(String, bool, bool)> {
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

    /// Properties with full row groups of `row_group_rows`, and a bloom filter
    /// the caller turned on for each of `on` and off for each of `off`, as
    /// `build_writer_props` sets them.
    fn caller_bloom_filters(
        row_group_rows: usize,
        on: &[&str],
        off: &[&str],
    ) -> DataFileProperties {
        let mut builder =
            WriterProperties::builder().set_max_row_group_row_count(Some(row_group_rows));
        for column in on {
            builder = builder
                .set_column_bloom_filter_enabled(ColumnPath::from(*column), true)
                .set_column_bloom_filter_fpp(ColumnPath::from(*column), BLOOM_FILTER_FPP);
        }
        for column in off {
            builder = builder.set_column_bloom_filter_enabled(ColumnPath::from(*column), false);
        }
        DataFileProperties {
            props: builder.build(),
            dictionary_set: Arc::default(),
            bloom_filter_set: Arc::new(
                on.iter().chain(off).map(|c| ColumnPath::from(*c)).collect(),
            ),
            pool: None,
        }
    }

    #[test]
    fn a_bloom_filter_the_caller_set_replaces_the_dictionary_rule() {
        // 20,000 rows, limit 4,000: `status` and `mid` keep their dictionaries,
        // `id` and `s` are unique and keep none.
        let batches: Vec<_> = (0..20).map(|b| batch(b * 1_000, (b + 1) * 1_000)).collect();
        let base = caller_bloom_filters(20_000, &["id", "s"], &["status"]);
        let expected = [
            ("id", false, true),
            ("v", false, false),
            ("s", false, true),
            ("status", true, false),
            // Not set by the caller: the rule gives it a filter with its dictionary.
            ("mid", true, true),
            ("empty", false, false),
        ]
        .map(|(column, dictionary, bloom)| (column.to_string(), dictionary, bloom));
        assert_eq!(layout_of(write_with(&batches, base)), expected);
    }

    #[test]
    fn a_bloom_filter_the_caller_set_is_folded_to_the_values_written() {
        use parquet::file::properties::ReaderProperties;
        use parquet::file::serialized_reader::ReadOptionsBuilder;

        // 1,000 unique values in a file whose full row group is 122,880 rows.
        let bytes = write_with(
            &[batch(0, 1_000)],
            caller_bloom_filters(122_880, &["s"], &[]),
        );
        let options = ReadOptionsBuilder::new()
            .with_reader_properties(
                ReaderProperties::builder()
                    .set_read_bloom_filter(true)
                    .build(),
            )
            .build();
        let reader = SerializedFileReader::new_with_options(bytes, options).unwrap();
        let s = 2;
        let length = reader
            .metadata()
            .row_group(0)
            .column(s)
            .bloom_filter_length()
            .expect("the filter's length is recorded");
        // Sized for 122,880 values at 1%, the filter would take 256 KiB.
        assert!(length < 16 * 1024, "a filter of {length} bytes");
        let row_group = reader.get_row_group(0).unwrap();
        let filter = row_group
            .get_column_bloom_filter(s)
            .expect("s has a bloom filter");
        for i in 0..1_000 {
            let value = format!("user_{i}");
            assert!(filter.check(value.as_str()), "{value} is missing");
        }
    }

    /// The distinct count a written file's properties give `path`'s filter.
    fn bloom_ndv(props: &WriterProperties, path: &ColumnPath) -> u64 {
        props
            .bloom_filter_properties(path)
            .unwrap_or_else(|| panic!("{path} has no bloom filter"))
            .ndv()
    }

    /// A write of one row over 400 columns with a caller's filter sizes each
    /// filter for that row, not for the 1,000,000-row cap: arrow-rs holds a
    /// filter's starting size until the file closes, so cap-sized filters
    /// would hold 2 MiB each, 800 MiB here, for one row.
    #[test]
    fn a_bloom_filter_the_caller_set_on_a_few_rows_is_sized_for_them() {
        const COLUMNS: usize = 400;
        const CAP: usize = 1_000_000;
        let schema = Arc::new(Schema::new(
            (0..COLUMNS)
                .map(|c| Field::new(format!("c{c}"), DataType::Utf8, true))
                .collect::<Vec<_>>(),
        ));
        let row = RecordBatch::try_new(
            Arc::clone(&schema),
            (0..COLUMNS)
                .map(|c| Arc::new(StringArray::from(vec![format!("v{c}")])) as ArrayRef)
                .collect(),
        )
        .unwrap();
        let paths: Vec<ColumnPath> = (0..COLUMNS)
            .map(|c| ColumnPath::from(format!("c{c}").as_str()))
            .collect();
        let mut builder = WriterProperties::builder().set_max_row_group_row_count(Some(CAP));
        for path in &paths {
            builder = builder
                .set_column_bloom_filter_enabled(path.clone(), true)
                .set_column_bloom_filter_fpp(path.clone(), BLOOM_FILTER_FPP);
        }
        let base = DataFileProperties {
            props: builder.build(),
            dictionary_set: Arc::default(),
            bloom_filter_set: Arc::new(paths.iter().cloned().collect()),
            pool: None,
        };
        assert_eq!(bloom_ndv(&base.props, &paths[0]), CAP as u64);

        let props = props_for_row_group(&base, std::slice::from_ref(&row)).unwrap();
        for path in &paths {
            assert_eq!(bloom_ndv(&props, path), 1, "{path}");
        }

        let mut writer = RowGroupSampledWriter::new(Vec::new(), Arc::clone(&schema), base);
        writer.write(&row).unwrap();
        let reader = SerializedFileReader::new(Bytes::from(writer.into_inner().unwrap())).unwrap();
        for column in reader.metadata().row_group(0).columns() {
            let length = column
                .bloom_filter_length()
                .unwrap_or_else(|| panic!("{} has no filter", column.column_path()));
            assert!(length <= 64, "{}: {length} bytes", column.column_path());
        }
    }

    /// A full row group's filter keeps a full row group's size, so the rate
    /// holds however many distinct values it gets.
    #[test]
    fn a_bloom_filter_the_caller_set_on_a_full_row_group_is_sized_for_it() {
        let batches: Vec<_> = (0..4).map(|b| batch(b * 1_000, (b + 1) * 1_000)).collect();
        let base = caller_bloom_filters(2_000, &["s"], &[]);
        let props = props_for_row_group(&base, &batches).unwrap();
        assert_eq!(bloom_ndv(&props, &ColumnPath::from("s")), 2_000);
    }

    /// A caller's filter on a nested column is set on its leaves, which the
    /// per-file choice never tunes, and is sized from the rows all the same.
    #[test]
    fn a_bloom_filter_the_caller_set_on_a_nested_column_is_sized_for_the_rows() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "tags",
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
            true,
        )]));
        let leaf = parquet::arrow::ArrowSchemaConverter::new()
            .convert(&schema)
            .unwrap()
            .column(0)
            .path()
            .clone();
        let mut values = arrow::array::ListBuilder::new(arrow::array::StringBuilder::new());
        for i in 0..10 {
            values.values().append_value(format!("t{i}"));
            values.append(true);
        }
        let batch =
            RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(values.finish())]).unwrap();
        let base = DataFileProperties {
            props: WriterProperties::builder()
                .set_max_row_group_row_count(Some(1_000_000))
                .set_column_bloom_filter_enabled(leaf.clone(), true)
                .set_column_bloom_filter_fpp(leaf.clone(), BLOOM_FILTER_FPP)
                .build(),
            dictionary_set: Arc::default(),
            bloom_filter_set: Arc::new(HashSet::from([leaf.clone()])),
            pool: None,
        };
        let props = props_for_row_group(&base, &[batch]).unwrap();
        assert_eq!(bloom_ndv(&props, &leaf), 10, "{leaf}");
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

    /// 16 batches of 32 rows, each row carrying a 40 KiB value built by
    /// `payload` from the row's id: 20 MiB of values, five times
    /// [`VIEW_TEST_CAP`].
    fn write_wide_rows(
        payload_type: DataType,
        payload: impl Fn(Vec<String>) -> ArrayRef,
    ) -> RowGroupSampledWriter<Vec<u8>> {
        const ROW_BYTES: usize = 40 * 1024;
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("payload", payload_type, false),
        ]));
        let base = WriterProperties::builder()
            .set_max_row_group_row_count(Some(1_000_000))
            .set_max_row_group_bytes(Some(VIEW_TEST_CAP))
            .build();
        let mut writer = RowGroupSampledWriter::new(Vec::new(), schema.clone(), base);
        for b in 0..16i64 {
            let ids: Vec<i64> = (b * 32..(b + 1) * 32).collect();
            let values = ids
                .iter()
                .map(|i| {
                    let mut text = format!("row-{i:08}-");
                    text.extend(std::iter::repeat_n('x', ROW_BYTES - text.len()));
                    text
                })
                .collect();
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(Int64Array::from(ids)), payload(values)],
            )
            .unwrap();
            writer.write(&batch).unwrap();
        }
        writer
    }

    const VIEW_TEST_CAP: usize = 4 * 1024 * 1024;

    fn assert_sample_ended(writer: &RowGroupSampledWriter<Vec<u8>>) {
        assert!(
            writer.writer.is_some(),
            "20 MiB of rows went past a {VIEW_TEST_CAP}-byte row-group cap and the writer \
             is still buffering {} batches, counted as {} bytes",
            writer.buffered.len(),
            writer.buffered_bytes,
        );
    }

    /// The byte cap ends the sampled row group for string-view columns too.
    ///
    /// A compaction merge reads its sources through DataFusion, which decodes
    /// strings as `Utf8View`. A view array keeps its values longer than 12
    /// bytes in separate data buffers, which `get_slice_memory_size` does not
    /// count: it counts only the 16-byte views. Measured that way, 40 KiB rows
    /// look like 16-byte rows, the cap is never reached, and the writer buffers
    /// every batch of the file until `into_inner`.
    #[test]
    fn the_byte_cap_ends_the_sample_for_string_view_columns() {
        let writer = write_wide_rows(DataType::Utf8View, |values| {
            Arc::new(StringViewArray::from_iter_values(values))
        });
        assert_sample_ended(&writer);
    }

    #[test]
    fn the_byte_cap_ends_the_sample_for_binary_view_columns() {
        let writer = write_wide_rows(DataType::BinaryView, |values| {
            Arc::new(BinaryViewArray::from_iter_values(values))
        });
        assert_sample_ended(&writer);
    }

    /// A view nested in a struct is counted as a top-level one is.
    #[test]
    fn the_byte_cap_ends_the_sample_for_a_view_in_a_struct() {
        let fields = Fields::from(vec![Field::new("text", DataType::Utf8View, false)]);
        let writer = write_wide_rows(DataType::Struct(fields.clone()), |values| {
            Arc::new(StructArray::new(
                fields.clone(),
                vec![Arc::new(StringViewArray::from_iter_values(values))],
                None,
            ))
        });
        assert_sample_ended(&writer);
    }

    /// A view as a map's values, one entry per row, is counted too.
    #[test]
    fn the_byte_cap_ends_the_sample_for_views_in_a_map() {
        let entries = Fields::from(vec![
            Field::new("keys", DataType::Utf8View, false),
            Field::new("values", DataType::Utf8View, true),
        ]);
        let entries_field = Arc::new(Field::new(
            "entries",
            DataType::Struct(entries.clone()),
            false,
        ));
        let writer = write_wide_rows(DataType::Map(Arc::clone(&entries_field), false), |values| {
            let rows = values.len();
            let keys = StringViewArray::from_iter_values((0..rows).map(|_| "k"));
            Arc::new(MapArray::new(
                Arc::clone(&entries_field),
                OffsetBuffer::from_lengths(std::iter::repeat_n(1, rows)),
                StructArray::new(
                    entries.clone(),
                    vec![Arc::new(keys), Arc::new(StringViewArray::from_iter_values(values))],
                    None,
                ),
                None,
                false,
            ))
        });
        assert_sample_ended(&writer);
    }

    #[test]
    fn distinct_values_are_counted_across_arrays_up_to_the_limit() {
        let a: ArrayRef = Arc::new(StringArray::from(vec![Some("a"), None, Some("b")]));
        let b: ArrayRef = Arc::new(StringArray::from(vec![Some("b"), Some("c"), None]));
        let c: ArrayRef = Arc::new(StringArray::from(vec![Some("d")]));
        // Exact at or below the limit; NULLs are not values; a value repeated in
        // a later array is counted once.
        assert_eq!(distinct_up_to(&[&a, &b], 3).unwrap(), 3);
        assert_eq!(distinct_up_to(&[&a, &b, &c], 4).unwrap(), 4);
        // Over the limit reads as `limit + 1`, however many more there are.
        assert_eq!(distinct_up_to(&[&a, &b, &c], 2).unwrap(), 3);
        assert_eq!(distinct_up_to(&[&a, &b, &c], 0).unwrap(), 1);
        assert_eq!(distinct_up_to(&[], 5).unwrap(), 0);
    }

    /// A slice of a view array counts only the values its own rows point at,
    /// not every value in the data buffers it shares with its parent.
    #[test]
    fn a_view_slice_counts_only_its_own_values() {
        let values = StringViewArray::from_iter_values((0..1_000).map(|i| format!("{i:0>1024}")));
        let schema = Arc::new(Schema::new(vec![Field::new(
            "s",
            DataType::Utf8View,
            false,
        )]));
        let parent = RecordBatch::try_new(schema, vec![Arc::new(values)]).unwrap();
        let whole = sliced_memory_size(&parent);
        let slice = sliced_memory_size(&parent.slice(0, 100));
        assert!(whole >= 1_000 * 1024, "the parent counts {whole} bytes");
        assert!(
            (100 * 1024..2 * 100 * 1024).contains(&slice),
            "a 100-row slice counts {slice} bytes"
        );
    }
}
