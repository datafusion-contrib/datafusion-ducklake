//! Parquet data page settings on `DuckLakeWriteOptions`
//! (`data_page_row_count_limit`, `data_page_size_limit`, `write_batch_size`).
//!
//! Each test reads the written files' page index with the plain parquet reader,
//! so a setting that never reached the writer shows up as pages of the parquet
//! default size. The paths covered are every one that writes a data file: the
//! table writer, SQL `INSERT` and `UPDATE`, the inlined-data flush, merge and
//! rewrite. A
//! small-page file must keep its page index (offset index and page-level
//! statistics), read normally in official DuckDB, and let
//! `read_columns_by_rowid` fetch less of it than a default-page file.

#![cfg(feature = "write-sqlite")]

use std::collections::BTreeSet;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{BinaryArray, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_ducklake::{
    ColumnWriteOptions, DuckLakeCatalog, DuckLakeTable, DuckLakeTableWriter, DuckLakeWriteOptions,
    MergeOptions, MetadataWriter, RewriteOptions, SqliteMetadataProvider, SqliteMetadataWriter,
};
use object_store::local::LocalFileSystem;
use parquet::file::page_index::column_index::ColumnIndexMetaData;
use parquet::file::properties::{DEFAULT_DATA_PAGE_ROW_COUNT_LIMIT, DEFAULT_PAGE_SIZE};
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::file::serialized_reader::ReadOptionsBuilder;
use tempfile::TempDir;

use crate::resolve_positions_projection_tests::ByteCountingStore;

/// Rows per page the small-page options ask for.
const PAGE_ROWS: usize = 50;

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

/// `rows` rows per page, the write batch no larger, so the limit holds per page.
fn small_page_options(rows: usize) -> DuckLakeWriteOptions {
    let mut options = DuckLakeWriteOptions::default();
    options.data_page_row_count_limit = Some(rows);
    options.write_batch_size = Some(rows);
    options
}

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, true),
        Field::new("v", DataType::Utf8, true),
    ]))
}

fn value(id: i64) -> String {
    format!("value-{id:010}-padding-padding")
}

fn batch(ids: Range<i64>) -> RecordBatch {
    RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(Int64Array::from_iter_values(ids.clone())),
            Arc::new(StringArray::from_iter_values(ids.map(value))),
        ],
    )
    .unwrap()
}

/// A fresh SQLite catalog in `temp`, returning its connection string.
async fn new_lake(temp: &TempDir) -> String {
    let data = temp.path().join("data");
    std::fs::create_dir_all(&data).unwrap();
    let url = format!("sqlite:{}?mode=rwc", temp.path().join("lake.db").display());
    let writer = SqliteMetadataWriter::new_with_init(&url).await.unwrap();
    writer.set_data_path(data.to_str().unwrap()).unwrap();
    url
}

async fn table_writer(url: &str, options: &DuckLakeWriteOptions) -> DuckLakeTableWriter {
    let writer = SqliteMetadataWriter::new(url).await.unwrap();
    DuckLakeTableWriter::new(Arc::new(writer), Arc::new(LocalFileSystem::new()))
        .unwrap()
        .with_options(options)
}

/// A session with the catalog at `url` registered as `lake`, writing with
/// `options`.
async fn session(url: &str, options: DuckLakeWriteOptions) -> SessionContext {
    let ctx = SessionContext::new();
    let writer = SqliteMetadataWriter::new(url).await.unwrap();
    let provider = SqliteMetadataProvider::new(url).await.unwrap();
    let catalog = DuckLakeCatalog::with_writer(Arc::new(provider), Arc::new(writer))
        .unwrap()
        .with_write_options(options);
    DuckLakeCatalog::register(&ctx, "lake", catalog);
    ctx
}

async fn table(ctx: &SessionContext, name: &str) -> DuckLakeTable {
    let provider = ctx
        .catalog("lake")
        .unwrap()
        .schema("main")
        .unwrap()
        .table(name)
        .await
        .unwrap()
        .unwrap();
    (provider.as_ref() as &dyn std::any::Any)
        .downcast_ref::<DuckLakeTable>()
        .expect("a DuckLakeTable")
        .clone()
}

async fn sql(ctx: &SessionContext, statement: &str) {
    ctx.sql(statement)
        .await
        .unwrap_or_else(|e| panic!("`{statement}`: {e}"))
        .collect()
        .await
        .unwrap_or_else(|e| panic!("`{statement}`: {e}"));
}

/// The pages of one column chunk, from the file's offset index.
#[derive(Debug)]
struct ChunkPages {
    column: String,
    /// Rows in each page.
    rows: Vec<i64>,
    /// Bytes of each page, header included.
    bytes: Vec<i64>,
    /// Whether the column index holds this chunk's page statistics.
    has_column_index: bool,
}

/// Every column chunk's pages in `path`, read with the plain parquet reader.
/// Panics when the file has no offset index or no column index.
fn pages(path: &Path) -> Vec<ChunkPages> {
    let reader = SerializedFileReader::new_with_options(
        std::fs::File::open(path).unwrap(),
        ReadOptionsBuilder::new().with_page_index().build(),
    )
    .unwrap();
    let metadata = reader.metadata();
    let offsets = metadata
        .offset_index()
        .unwrap_or_else(|| panic!("{}: no offset index", path.display()));
    let column_index = metadata
        .column_index()
        .unwrap_or_else(|| panic!("{}: no column index", path.display()));
    let mut chunks = Vec::new();
    for (group, row_group) in metadata.row_groups().iter().enumerate() {
        for (column, chunk) in row_group.columns().iter().enumerate() {
            let locations = offsets[group][column].page_locations();
            let starts: Vec<i64> = locations.iter().map(|page| page.first_row_index).collect();
            let rows = starts
                .iter()
                .zip(starts.iter().skip(1).chain([&row_group.num_rows()]))
                .map(|(start, end)| end - start)
                .collect();
            chunks.push(ChunkPages {
                column: chunk.column_path().string(),
                rows,
                bytes: locations
                    .iter()
                    .map(|page| i64::from(page.compressed_page_size))
                    .collect(),
                has_column_index: !matches!(column_index[group][column], ColumnIndexMetaData::NONE),
            });
        }
    }
    chunks
}

/// The data files under `dir`: every parquet file with an `id` column (delete
/// files hold `file_path` and `pos`).
fn data_files(dir: &Path) -> BTreeSet<PathBuf> {
    let mut found = BTreeSet::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|ext| ext == "parquet") {
                let reader =
                    SerializedFileReader::new(std::fs::File::open(&path).unwrap()).unwrap();
                let schema = reader.metadata().file_metadata().schema_descr_ptr();
                if schema.columns().iter().any(|column| column.name() == "id") {
                    found.insert(path);
                }
            }
        }
    }
    found
}

/// Every page of every column of `path` holds at most `limit` rows, every
/// chunk keeps its page statistics, and the file has the pages its row count
/// needs at that limit.
fn assert_small_pages(path: &Path, limit: usize, what: &str) {
    let chunks = pages(path);
    assert!(!chunks.is_empty(), "{what}: no column chunk");
    let limit = i64::try_from(limit).unwrap();
    for chunk in &chunks {
        let rows: i64 = chunk.rows.iter().sum();
        assert!(
            chunk.rows.iter().all(|&rows| rows <= limit),
            "{what}: {} has a page over {limit} rows: {:?}",
            chunk.column,
            chunk.rows
        );
        assert!(
            i64::try_from(chunk.rows.len()).unwrap() >= (rows + limit - 1) / limit,
            "{what}: {} has {} pages for {rows} rows",
            chunk.column,
            chunk.rows.len()
        );
        assert!(
            chunk.has_column_index,
            "{what}: {} has no page statistics",
            chunk.column
        );
    }
}

/// The data files `dir` holds that `before` does not, which must be some.
fn new_files(dir: &Path, before: &BTreeSet<PathBuf>, what: &str) -> BTreeSet<PathBuf> {
    let now = data_files(dir);
    let added: BTreeSet<PathBuf> = now.difference(before).cloned().collect();
    assert!(!added.is_empty(), "{what} wrote no data file");
    added
}

// ---------------------------------------------------------------------------
// Page layout of the written files
// ---------------------------------------------------------------------------

/// `data_page_row_count_limit` (with a `write_batch_size` no larger) caps the
/// rows of every page of every column, and the file keeps its offset index and
/// page-level statistics. Guards against the limit not reaching the parquet
/// writer, and against a page index dropped from the files.
#[tokio::test(flavor = "multi_thread")]
async fn page_row_count_limit_caps_every_page_and_keeps_the_page_index() {
    let temp = TempDir::new().unwrap();
    let url = new_lake(&temp).await;
    table_writer(&url, &small_page_options(PAGE_ROWS))
        .await
        .write_table("main", "t", &[batch(0..10_000)])
        .await
        .unwrap();
    let files = data_files(&temp.path().join("data"));
    assert_eq!(files.len(), 1);
    let file = files.first().unwrap();
    assert_small_pages(file, PAGE_ROWS, "write_table");
    for chunk in pages(file) {
        assert_eq!(chunk.rows.len(), 10_000 / PAGE_ROWS, "{}", chunk.column);
    }
}

/// `data_page_size_limit` bounds the bytes of every page, best effort. Guards against the byte limit not reaching the writer.
#[tokio::test(flavor = "multi_thread")]
async fn page_size_limit_caps_every_page() {
    const LIMIT: usize = 2_048;
    const BATCH: usize = 8;
    let temp = TempDir::new().unwrap();
    let url = new_lake(&temp).await;
    let mut options = DuckLakeWriteOptions::default();
    options.data_page_size_limit = Some(LIMIT);
    options.write_batch_size = Some(BATCH);
    table_writer(&url, &options)
        .await
        .write_table("main", "t", &[batch(0..5_000)])
        .await
        .unwrap();
    let files = data_files(&temp.path().join("data"));
    let chunks = pages(files.first().unwrap());
    let v = chunks.iter().find(|chunk| chunk.column == "v").unwrap();
    let total: i64 = v.bytes.iter().sum();
    // Best effort: the writer checks its estimate of a page's encoded size
    // between batches, so a page can run past the limit, though not double it.
    let cap = 2 * i64::try_from(LIMIT).unwrap();
    assert!(
        v.bytes.iter().all(|&bytes| bytes <= cap),
        "a page of v over twice {LIMIT} bytes: {:?}",
        v.bytes
    );
    assert!(
        i64::try_from(v.bytes.len()).unwrap() >= total / cap,
        "{} pages for {total} bytes of v",
        v.bytes.len()
    );
    assert!(v.has_column_index);
}

/// Without page settings a file has the parquet writer's default pages: a
/// narrow column's pages run to the default row count limit, a wide one's to
/// the default byte size. Guards against a default the crate set in place of
/// the parquet writer's.
#[tokio::test(flavor = "multi_thread")]
async fn unset_page_options_keep_the_parquet_defaults() {
    let options = DuckLakeWriteOptions::default();
    assert_eq!(options.data_page_row_count_limit, None);
    assert_eq!(options.data_page_size_limit, None);
    assert_eq!(options.write_batch_size, None);

    const ROWS: i64 = 50_000;
    let temp = TempDir::new().unwrap();
    let url = new_lake(&temp).await;
    table_writer(&url, &options)
        .await
        .write_table("main", "t", &[batch(0..ROWS)])
        .await
        .unwrap();
    let files = data_files(&temp.path().join("data"));
    let chunks = pages(files.first().unwrap());
    let id = chunks.iter().find(|chunk| chunk.column == "id").unwrap();
    let default_rows = i64::try_from(DEFAULT_DATA_PAGE_ROW_COUNT_LIMIT).unwrap();
    assert_eq!(
        id.rows,
        vec![default_rows, default_rows, ROWS - 2 * default_rows],
        "an 8-byte column's pages hold the default row count"
    );
    let v = chunks.iter().find(|chunk| chunk.column == "v").unwrap();
    let total: i64 = v.bytes.iter().sum();
    let default_bytes = i64::try_from(DEFAULT_PAGE_SIZE).unwrap();
    assert!(
        i64::try_from(v.rows.len()).unwrap() <= total / default_bytes + 3,
        "v in {} pages for {total} bytes",
        v.rows.len()
    );
    assert!(id.has_column_index && v.has_column_index);
}

/// Per-column settings reach the file: a column's compression, dictionary and
/// encoding are read back from its chunk metadata and its page size from the
/// page index, on every leaf of a list column too, while the other columns keep
/// the file-wide settings. A set dictionary outranks the one the first row group
/// would choose, both ways. Rows per page uncapped leave a narrow column one page
/// per row group. Guards against column settings set on the wrong path, lost to
/// the row-group dictionary choice, or applied to every column.
#[tokio::test(flavor = "multi_thread")]
async fn column_options_reach_the_written_file() {
    use parquet::basic::{Compression, Encoding};
    use parquet::file::properties::DEFAULT_DATA_PAGE_ROW_COUNT_LIMIT;

    const ROWS: i64 = 50_000;
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, true),
        // Distinct values: the first row group alone would write no dictionary.
        Field::new("v", DataType::Utf8, true),
        // Four values: the first row group alone would write a dictionary.
        Field::new("s", DataType::Utf8, true),
        Field::new(
            "emb",
            DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
            true,
        ),
    ]));
    let emb = arrow::array::ListArray::from_iter_primitive::<arrow::datatypes::Float32Type, _, _>(
        (0..ROWS).map(|id| Some((0..4).map(move |i| Some((id * 4 + i) as f32)))),
    );
    let rows = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from_iter_values(0..ROWS)),
            Arc::new(StringArray::from_iter_values((0..ROWS).map(value))),
            Arc::new(StringArray::from_iter_values(
                (0..ROWS).map(|id| format!("status-{}", id % 4)),
            )),
            Arc::new(emb),
        ],
    )
    .unwrap();

    let mut options = DuckLakeWriteOptions::default()
        .with_column_options(
            "v",
            ColumnWriteOptions::new()
                .with_dictionary_enabled(true)
                .with_compression(Compression::SNAPPY),
        )
        .with_column_options(
            "s",
            ColumnWriteOptions::new()
                .with_dictionary_enabled(false)
                .with_page_size_limit(4_096),
        )
        .with_column_options(
            "emb",
            ColumnWriteOptions::new()
                .with_dictionary_enabled(false)
                .with_encoding(Encoding::BYTE_STREAM_SPLIT),
        );
    options.data_page_row_count_limit = Some(usize::MAX);
    options.max_row_group_rows = Some(usize::try_from(ROWS).unwrap());

    let temp = TempDir::new().unwrap();
    let url = new_lake(&temp).await;
    table_writer(&url, &options)
        .await
        .write_table("main", "t", std::slice::from_ref(&rows))
        .await
        .unwrap();
    let files = data_files(&temp.path().join("data"));
    let file = files.first().unwrap();
    let reader = SerializedFileReader::new(std::fs::File::open(file).unwrap()).unwrap();
    let row_group = reader.metadata().row_group(0);
    let chunk = |path: &str| {
        row_group
            .columns()
            .iter()
            .find(|chunk| chunk.column_path().string() == path)
            .unwrap_or_else(|| {
                panic!(
                    "no chunk {path} in {:?}",
                    row_group
                        .columns()
                        .iter()
                        .map(|chunk| chunk.column_path().string())
                        .collect::<Vec<_>>()
                )
            })
    };
    let dictionary = |path: &str| chunk(path).dictionary_page_offset().is_some();
    let encodings = |path: &str| chunk(path).encodings().collect::<Vec<_>>();

    assert!(dictionary("v"), "v keeps the dictionary it was given");
    assert_eq!(chunk("v").compression(), Compression::SNAPPY);
    assert!(
        !dictionary("s"),
        "s drops the dictionary it would have chosen"
    );
    assert_eq!(chunk("s").compression(), Compression::UNCOMPRESSED);
    assert!(!dictionary("emb.list.element"));
    assert!(
        encodings("emb.list.element").contains(&Encoding::BYTE_STREAM_SPLIT),
        "{:?}",
        encodings("emb.list.element")
    );
    assert_eq!(chunk("id").compression(), Compression::UNCOMPRESSED);
    assert!(!encodings("id").contains(&Encoding::BYTE_STREAM_SPLIT));

    let chunks = pages(file);
    let pages_of = |path: &str| chunks.iter().find(|chunk| chunk.column == path).unwrap();
    // `s` pages stay near its own 4 KiB limit; `v`'s run to the file's 1 MiB.
    let s = pages_of("s");
    assert!(
        s.bytes.iter().all(|&bytes| bytes <= 2 * 4_096),
        "{:?}",
        s.bytes
    );
    assert!(s.bytes.len() > 8, "{} pages of s", s.bytes.len());
    assert!(pages_of("v").bytes.len() < s.bytes.len() / 4);
    // No row cap: 50,000 8-byte ids fit one 1 MiB page, past the default
    // 20,000-row cap.
    let id = pages_of("id");
    assert_eq!(id.rows, vec![ROWS], "id pages");
    assert!(ROWS > i64::try_from(DEFAULT_DATA_PAGE_ROW_COUNT_LIMIT).unwrap());

    // The values read back.
    let ctx = session(&url, DuckLakeWriteOptions::default()).await;
    let check = ctx
        .sql("SELECT count(*), count(DISTINCT v), count(DISTINCT s), sum(emb[2]) FROM lake.main.t")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let check = arrow::util::pretty::pretty_format_batches(&check)
        .unwrap()
        .to_string();
    // sum over id of (4 * id + 1)
    let emb_sum: i64 = (0..ROWS).map(|id| 4 * id + 1).sum();
    assert!(check.contains("| 50000 "), "{check}");
    assert!(check.contains("| 4 "), "{check}");
    assert!(check.contains(&emb_sum.to_string()), "{check}");
}

/// A column encoding its type cannot take fails every kind of write with an
/// error, not the panic the Parquet writer raises for it, and the failed write
/// commits nothing and leaves no file: a new table, an append, a streaming
/// session and a SQL `INSERT`. Guards against an encoding handed to the
/// writer unchecked, or checked only after a write has committed its setup.
#[tokio::test(flavor = "multi_thread")]
async fn an_encoding_the_column_type_cannot_take_fails_the_write() {
    use datafusion_ducklake::{MetadataProvider, WriteMode};

    let temp = TempDir::new().unwrap();
    let data = temp.path().join("data");
    let url = new_lake(&temp).await;
    let bad = DuckLakeWriteOptions::default()
        .with_column_options(
            "v",
            ColumnWriteOptions::new()
                .with_dictionary_enabled(false)
                .with_encoding(parquet::basic::Encoding::DELTA_BINARY_PACKED),
        )
        .with_data_inlining_row_limit(0);
    let refused = |error: &datafusion_ducklake::DuckLakeError| {
        matches!(error, datafusion_ducklake::DuckLakeError::InvalidConfig(message)
            if message.contains("DELTA_BINARY_PACKED") && message.contains("column v"))
    };
    let provider = SqliteMetadataProvider::new(&url).await.unwrap();

    // A new table.
    let error = table_writer(&url, &bad)
        .await
        .write_table("main", "new", &[batch(0..100)])
        .await
        .unwrap_err();
    assert!(refused(&error), "{error}");
    assert!(data_files(&data).is_empty());

    // An existing table, written without the bad option first.
    table_writer(&url, &DuckLakeWriteOptions::default())
        .await
        .write_table("main", "t", &[batch(0..100)])
        .await
        .unwrap();
    let head = provider.get_current_snapshot().unwrap();
    let files = data_files(&data);
    let unchanged = |what: &str| {
        assert_eq!(
            provider.get_current_snapshot().unwrap(),
            head,
            "{what}: committed"
        );
        assert_eq!(data_files(&data), files, "{what}: left a file");
    };

    let error = table_writer(&url, &bad)
        .await
        .append_table("main", "t", &[batch(100..200)])
        .await
        .unwrap_err();
    assert!(refused(&error), "{error}");
    unchanged("append");

    let writer = table_writer(&url, &bad).await;
    match writer.begin_write("main", "t", schema().as_ref(), WriteMode::Append) {
        Err(error) => assert!(refused(&error), "{error}"),
        Ok(mut session) => {
            let error = match session.write_batch(&batch(100..200)) {
                Err(error) => error,
                Ok(()) => session.finish().await.unwrap_err(),
            };
            assert!(refused(&error), "{error}");
        },
    }
    unchanged("streaming session");

    let ctx = session(&url, bad).await;
    let error = ctx
        .sql("INSERT INTO lake.main.t SELECT value, 'x' FROM generate_series(0, 99)")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap_err();
    assert!(error.to_string().contains("DELTA_BINARY_PACKED"), "{error}");
    unchanged("INSERT");
}

/// A catalog's write options reach every path that writes a data file: SQL
/// `INSERT`, the inlined-data flush, merge, rewrite and SQL `UPDATE`, each output with pages
/// of at most the configured rows. Guards against a maintenance path that
/// builds its writer without the catalog's options, or copies the row-group
/// settings but not the page ones.
#[tokio::test(flavor = "multi_thread")]
async fn insert_flush_merge_rewrite_and_update_write_small_pages() {
    let temp = TempDir::new().unwrap();
    let data = temp.path().join("data");
    let url = new_lake(&temp).await;
    let mut options = small_page_options(PAGE_ROWS).with_data_inlining_row_limit(500);
    options.max_row_group_rows = Some(1_000);

    table_writer(&url, &options)
        .await
        .write_table("main", "t", &[batch(0..3_000)])
        .await
        .unwrap();
    let mut seen = data_files(&data);
    for file in &seen {
        assert_small_pages(file, PAGE_ROWS, "table writer");
    }

    let ctx = session(&url, options).await;
    sql(
        &ctx,
        "INSERT INTO lake.main.t SELECT value AS id, concat('value-', value) AS v \
         FROM generate_series(3000, 5999)",
    )
    .await;
    for file in new_files(&data, &seen, "INSERT") {
        assert_small_pages(&file, PAGE_ROWS, "INSERT");
    }
    seen = data_files(&data);

    // Inlined under the 500-row limit, then flushed to a file.
    sql(
        &ctx,
        "INSERT INTO lake.main.t SELECT value AS id, concat('value-', value) AS v \
         FROM generate_series(6000, 6299)",
    )
    .await;
    assert_eq!(data_files(&data), seen, "300 rows are inlined");
    let catalog = ctx.catalog("lake").unwrap();
    let catalog = (catalog.as_ref() as &dyn std::any::Any)
        .downcast_ref::<DuckLakeCatalog>()
        .unwrap();
    let flushed = catalog
        .flush_inlined_data(&ctx.state(), Some("main"), Some("t"))
        .await
        .unwrap();
    assert_eq!(flushed.len(), 1);
    for file in new_files(&data, &seen, "flush") {
        assert_small_pages(&file, PAGE_ROWS, "flush");
    }
    seen = data_files(&data);

    let merged = table(&ctx, "t")
        .await
        .merge_adjacent_files(&ctx.state(), MergeOptions::default())
        .await
        .unwrap();
    assert!(merged.files_created > 0, "{merged:?}");
    let merged_files = new_files(&data, &seen, "merge");
    for file in &merged_files {
        assert_small_pages(file, PAGE_ROWS, "merge");
    }
    seen = data_files(&data);

    sql(&ctx, "DELETE FROM lake.main.t WHERE id % 10 <> 0").await;
    let rewritten = table(&ctx, "t")
        .await
        .rewrite_data_files(
            &ctx.state(),
            RewriteOptions {
                delete_threshold: 0.5,
                ..RewriteOptions::default()
            },
        )
        .await
        .unwrap();
    assert!(rewritten.files_created > 0, "{rewritten:?}");
    for file in new_files(&data, &seen, "rewrite") {
        assert_small_pages(&file, PAGE_ROWS, "rewrite");
    }
    seen = data_files(&data);

    // Every live row, over the inlining limit, so the new versions go to a file.
    sql(&ctx, "UPDATE lake.main.t SET v = concat('u-', v)").await;
    for file in new_files(&data, &seen, "UPDATE") {
        assert_small_pages(&file, PAGE_ROWS, "UPDATE");
    }

    let count = ctx
        .sql("SELECT count(*), sum(id) FROM lake.main.t")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let count = arrow::util::pretty::pretty_format_batches(&count)
        .unwrap()
        .to_string();
    // ids 0..6300 that are multiples of 10.
    assert!(count.contains("| 630 "), "{count}");
    assert!(count.contains("| 1981350 "), "{count}");
}

// ---------------------------------------------------------------------------
// Readers
// ---------------------------------------------------------------------------

/// Official DuckDB reads a table this crate wrote with small pages, deletes
/// applied, and agrees with this crate's own scan row for row, through a
/// filter its page index can prune with. Guards against a page layout another
/// Parquet reader cannot read.
#[cfg(feature = "metadata-duckdb")]
#[tokio::test(flavor = "multi_thread")]
async fn official_duckdb_reads_a_small_page_table() {
    use crate::inlined_mutation_differential_tests::{Metadata, duckdb_query};

    let temp = TempDir::new().unwrap();
    let data = temp.path().join("data");
    let url = new_lake(&temp).await;
    let mut options = small_page_options(10).with_data_inlining_row_limit(0);
    options.max_row_group_rows = Some(2_000);
    table_writer(&url, &options)
        .await
        .write_table("main", "t", &[batch(0..5_000)])
        .await
        .unwrap();
    let ctx = session(&url, options).await;
    sql(&ctx, "DELETE FROM lake.main.t WHERE id % 7 = 3").await;
    for file in data_files(&data) {
        assert_small_pages(&file, 10, "written");
    }
    drop(ctx);

    let crate_ctx = session(&url, DuckLakeWriteOptions::default()).await;
    for filter in ["true", "id BETWEEN 1234 AND 1301", "v = 'value-0000004321-padding-padding'"] {
        let query = format!("SELECT id, v FROM lake.main.t WHERE {filter} ORDER BY id");
        let crate_rows: Vec<String> = {
            let batches = crate_ctx
                .sql(&query)
                .await
                .unwrap()
                .collect()
                .await
                .unwrap();
            let mut rows = Vec::new();
            for batch in &batches {
                let ids = batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap();
                let values = arrow::compute::cast(batch.column(1), &DataType::Utf8).unwrap();
                let values = values.as_any().downcast_ref::<StringArray>().unwrap();
                for row in 0..batch.num_rows() {
                    rows.push(format!("{},{}", ids.value(row), values.value(row)));
                }
            }
            rows
        };
        let expected: Vec<String> = (0..5_000)
            .filter(|id| id % 7 != 3)
            .map(|id| format!("{id},{}", value(id)))
            .filter(|row| match filter {
                "true" => true,
                "id BETWEEN 1234 AND 1301" => {
                    let id: i64 = row.split(',').next().unwrap().parse().unwrap();
                    (1234..=1301).contains(&id)
                },
                _ => row.ends_with(",value-0000004321-padding-padding"),
            })
            .collect();
        assert_eq!(crate_rows, expected, "this crate, {filter}");
        let official = duckdb_query(
            &Metadata::Sqlite(temp.path().join("lake.db")),
            &data,
            &query,
        );
        assert_eq!(official, expected, "official DuckDB, {filter}");
    }
}

/// Official DuckDB reads a table written with per-column settings a vector
/// table uses: floats and a float list in `BYTE_STREAM_SPLIT` without a
/// dictionary, strings in `DELTA_LENGTH_BYTE_ARRAY` with small pages and
/// their own codec, integers in `DELTA_BINARY_PACKED`, rows per page
/// uncapped. Guards against column settings that produce a file another
/// Parquet reader cannot read.
#[cfg(feature = "metadata-duckdb")]
#[tokio::test(flavor = "multi_thread")]
async fn official_duckdb_reads_a_table_with_column_options() {
    use crate::inlined_mutation_differential_tests::{Metadata, duckdb_query};
    use parquet::basic::{Compression, Encoding};

    const ROWS: i64 = 5_000;
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, true),
        Field::new("v", DataType::Utf8, true),
        Field::new("f", DataType::Float32, true),
        Field::new(
            "emb",
            DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
            true,
        ),
    ]));
    let emb = arrow::array::ListArray::from_iter_primitive::<arrow::datatypes::Float32Type, _, _>(
        (0..ROWS).map(|id| Some([Some(id as f32), Some((2 * id) as f32)])),
    );
    let rows = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from_iter_values(0..ROWS)),
            Arc::new(StringArray::from_iter_values((0..ROWS).map(value))),
            Arc::new(arrow::array::Float32Array::from_iter_values(
                (0..ROWS).map(|id| id as f32),
            )),
            Arc::new(emb),
        ],
    )
    .unwrap();
    let split = ColumnWriteOptions::new()
        .with_dictionary_enabled(false)
        .with_encoding(Encoding::BYTE_STREAM_SPLIT);
    let mut options = DuckLakeWriteOptions::default()
        .with_column_options(
            "id",
            ColumnWriteOptions::new()
                .with_dictionary_enabled(false)
                .with_encoding(Encoding::DELTA_BINARY_PACKED),
        )
        .with_column_options(
            "v",
            ColumnWriteOptions::new()
                .with_dictionary_enabled(false)
                .with_encoding(Encoding::DELTA_LENGTH_BYTE_ARRAY)
                .with_compression(Compression::SNAPPY)
                .with_page_size_limit(1_024),
        )
        .with_column_options("f", split.clone())
        .with_column_options("emb", split.with_page_size_limit(2_048));
    options.data_page_row_count_limit = Some(usize::MAX);

    let temp = TempDir::new().unwrap();
    let data = temp.path().join("data");
    let url = new_lake(&temp).await;
    table_writer(&url, &options)
        .await
        .write_table("main", "t", std::slice::from_ref(&rows))
        .await
        .unwrap();
    let files = data_files(&data);
    let reader =
        SerializedFileReader::new(std::fs::File::open(files.first().unwrap()).unwrap()).unwrap();
    let encodings: Vec<(String, Vec<Encoding>)> = reader
        .metadata()
        .row_group(0)
        .columns()
        .iter()
        .map(|chunk| (chunk.column_path().string(), chunk.encodings().collect()))
        .collect();
    for (column, encoding) in [
        ("id", Encoding::DELTA_BINARY_PACKED),
        ("v", Encoding::DELTA_LENGTH_BYTE_ARRAY),
        ("f", Encoding::BYTE_STREAM_SPLIT),
        ("emb.list.element", Encoding::BYTE_STREAM_SPLIT),
    ] {
        let (_, used) = encodings.iter().find(|(path, _)| path == column).unwrap();
        assert!(used.contains(&encoding), "{column}: {used:?}");
    }

    let metadata = Metadata::Sqlite(temp.path().join("lake.db"));
    let summary = duckdb_query(
        &metadata,
        &data,
        "SELECT count(*), count(DISTINCT v), sum(id), sum(f)::BIGINT, \
         sum(list_sum(emb))::BIGINT FROM lake.main.t",
    );
    let ids: i64 = (0..ROWS).sum();
    assert_eq!(
        summary,
        vec![format!("{ROWS},{ROWS},{ids},{ids},{}", 3 * ids)]
    );
    let rows = duckdb_query(
        &metadata,
        &data,
        "SELECT id, v, f::BIGINT, emb[2]::BIGINT FROM lake.main.t \
         WHERE id IN (0, 1234, 4999) ORDER BY id",
    );
    let expected: Vec<String> = [0, 1234, 4999]
        .iter()
        .map(|id| format!("{id},{},{id},{}", value(*id), 2 * id))
        .collect();
    assert_eq!(rows, expected);
}

/// `read_columns_by_rowid` of a few rows reads far less of a file written with
/// small pages than of the same rows in a default-page file, the footer
/// already read. Guards against small pages that the read never sees: the
/// setting not applied, or the page index not used to select them.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn read_columns_by_rowid_reads_less_of_a_small_page_file() {
    const ROWS: i64 = 16_000;
    const PAYLOAD: usize = 512;
    let payload_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, true),
        Field::new("payload", DataType::Binary, true),
    ]));
    // Incompressible, distinct bytes per row, so no encoding shrinks a page.
    let payload = |id: i64| -> Vec<u8> {
        let mut state = (id as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..PAYLOAD)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state as u8
            })
            .collect()
    };
    let rows = RecordBatch::try_new(
        Arc::clone(&payload_schema),
        vec![
            Arc::new(Int64Array::from_iter_values(0..ROWS)),
            Arc::new(BinaryArray::from_iter_values((0..ROWS).map(payload))),
        ],
    )
    .unwrap();

    let wanted = [7, 8_003, ROWS - 2];
    let mut read = Vec::new();
    for page_rows in [None, Some(32)] {
        let temp = TempDir::new().unwrap();
        let url = new_lake(&temp).await;
        let mut options = match page_rows {
            Some(rows) => small_page_options(rows),
            None => DuckLakeWriteOptions::default(),
        };
        options.max_row_group_rows = Some(usize::try_from(ROWS).unwrap());
        table_writer(&url, &options)
            .await
            .write_table("main", "t", std::slice::from_ref(&rows))
            .await
            .unwrap();

        let store = Arc::new(ByteCountingStore::new());
        let ctx = SessionContext::new_with_config(SessionConfig::new().with_target_partitions(4));
        ctx.runtime_env().register_object_store(
            &url::Url::parse("file:///").unwrap(),
            Arc::clone(&store) as Arc<dyn object_store::ObjectStore>,
        );
        let provider = SqliteMetadataProvider::new(&url).await.unwrap();
        let catalog = DuckLakeCatalog::new(provider)
            .unwrap()
            .with_row_lineage(true);
        ctx.register_catalog("lake", Arc::new(catalog));
        let t = table(&ctx, "t").await;
        let total = u64::try_from(t.files().unwrap()[0].file.file_size_bytes).unwrap();
        let state = ctx.state();
        // The first read takes the footer, which the table then keeps.
        t.read_columns_by_rowid(&state, &wanted, None)
            .await
            .unwrap();
        store.take();
        let got = t
            .read_columns_by_rowid(&state, &wanted, None)
            .await
            .unwrap();
        let bytes = store.take();
        assert_eq!(got.row_ids, wanted.to_vec());
        let payloads = arrow::compute::cast(got.batch.column(1), &DataType::Binary).unwrap();
        let payloads = payloads.as_any().downcast_ref::<BinaryArray>().unwrap();
        for (row, id) in wanted.iter().enumerate() {
            assert_eq!(payloads.value(row), payload(*id).as_slice(), "rowid {id}");
        }
        eprintln!("pages of {page_rows:?} rows: read {bytes} of {total} bytes for 3 rows");
        assert!(bytes > 0, "the counting store saw no reads");
        read.push(bytes);
    }
    let (default_pages, small_pages) = (read[0], read[1]);
    assert!(
        small_pages * 10 < default_pages,
        "small pages read {small_pages} bytes, default pages {default_pages}"
    );
}
