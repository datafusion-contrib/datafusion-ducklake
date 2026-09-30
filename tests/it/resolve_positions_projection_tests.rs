//! `DuckLakeTable::resolve_positions` reads only the data columns its predicate
//! references.
//!
//! A keyed DELETE names one or two narrow key columns, while the rows it
//! targets can carry wide payloads beside them. Every test here asserts two
//! things on a table built that way — a 1536-byte binary column and a long text
//! column ahead of the key columns:
//!
//! - **Positions**: the resolved positions equal an oracle computed from the
//!   data file itself, read column-by-column with the plain parquet reader and
//!   filtered in Rust. The oracle never goes through this crate's scan, so a
//!   predicate rebound to the wrong column cannot agree with it.
//! - **Bytes**: the scan fetched a small fraction of the file, counted by an
//!   object store that sees every read. A narrow projection is only visible
//!   there; a positions-only assertion passes either way.
//!
//! The key columns sit AFTER the wide ones on purpose, so each referenced
//! column's index in the narrowed batch differs from its catalog index.
//!
//! The shapes are the ones where a column's physical identity differs from its
//! catalog one: a renamed key, a widened key, a key the file predates, a
//! name-mapped file with a Hive constant, a file that embeds row ids, a
//! partitioned table, and files already carrying a delete file.

#![cfg(all(feature = "write-sqlite", feature = "metadata-sqlite"))]

use std::collections::HashSet;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use arrow::array::{Array, ArrayRef, BinaryArray, Int32Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use datafusion::catalog::TableProvider;
use datafusion::common::DFSchema;
use datafusion::config::ConfigOptions;
use datafusion::datasource::MemTable;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::prelude::*;
use object_store::local::LocalFileSystem;
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use tempfile::TempDir;

use datafusion_ducklake::partition::PartitionTransform;
use datafusion_ducklake::{
    ColumnDef, DuckLakeCatalog, DuckLakeTable, DuckLakeTableFile, DuckLakeTableWriter,
    DuckLakeWriteOptions, MergeOptions, MetadataWriter, SqliteMetadataProvider,
    SqliteMetadataWriter, WriteMode,
};

/// Rows per fixture file.
const ROWS: i64 = 8_000;
/// Rows per parquet row group: several per file, so the scan splits across
/// partitions and row-group pruning has something to act on.
const ROWS_PER_ROW_GROUP: usize = 2_000;
/// Width of the binary payload column, in bytes per row.
const PAYLOAD_BYTES: usize = 1_536;
/// Width of the text column, in characters per row.
const NOTE_CHARS: usize = 256;
/// A narrow scan must read less than `1 / NARROW_FRACTION` of the file. The
/// key columns are well under 1% of these fixtures' bytes, and a scan that
/// decodes the wide columns reads nearly all of them, so the bound separates
/// the two by more than an order of magnitude either way.
const NARROW_FRACTION: u64 = 20;

// ---------------------------------------------------------------------------
// Byte-counting object store
// ---------------------------------------------------------------------------

/// An `ObjectStore` that counts the bytes every read asks for, delegating all
/// of it to `inner`. Covers both ways the parquet reader fetches: `get_ranges`
/// (column chunks) and `get_opts` (single ranges, and `get_range`, whose
/// default implementation goes through it).
#[derive(Debug)]
struct ByteCountingStore {
    inner: Arc<dyn object_store::ObjectStore>,
    bytes: AtomicU64,
}

impl ByteCountingStore {
    fn new() -> Self {
        Self {
            inner: Arc::new(LocalFileSystem::new()),
            bytes: AtomicU64::new(0),
        }
    }

    fn take(&self) -> u64 {
        self.bytes.swap(0, Ordering::SeqCst)
    }
}

impl std::fmt::Display for ByteCountingStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ByteCountingStore({})", self.inner)
    }
}

#[async_trait::async_trait]
impl object_store::ObjectStore for ByteCountingStore {
    async fn put_opts(
        &self,
        location: &object_store::path::Path,
        payload: object_store::PutPayload,
        opts: object_store::PutOptions,
    ) -> object_store::Result<object_store::PutResult> {
        self.inner.put_opts(location, payload, opts).await
    }

    async fn put_multipart_opts(
        &self,
        location: &object_store::path::Path,
        opts: object_store::PutMultipartOptions,
    ) -> object_store::Result<Box<dyn object_store::MultipartUpload>> {
        self.inner.put_multipart_opts(location, opts).await
    }

    async fn get_opts(
        &self,
        location: &object_store::path::Path,
        options: object_store::GetOptions,
    ) -> object_store::Result<object_store::GetResult> {
        let result = self.inner.get_opts(location, options).await?;
        self.bytes
            .fetch_add(result.range.end - result.range.start, Ordering::SeqCst);
        Ok(result)
    }

    async fn get_ranges(
        &self,
        location: &object_store::path::Path,
        ranges: &[Range<u64>],
    ) -> object_store::Result<Vec<bytes::Bytes>> {
        let requested: u64 = ranges.iter().map(|range| range.end - range.start).sum();
        self.bytes.fetch_add(requested, Ordering::SeqCst);
        self.inner.get_ranges(location, ranges).await
    }

    fn delete_stream(
        &self,
        locations: futures::stream::BoxStream<
            'static,
            object_store::Result<object_store::path::Path>,
        >,
    ) -> futures::stream::BoxStream<'static, object_store::Result<object_store::path::Path>> {
        self.inner.delete_stream(locations)
    }

    fn list(
        &self,
        prefix: Option<&object_store::path::Path>,
    ) -> futures::stream::BoxStream<'static, object_store::Result<object_store::ObjectMeta>> {
        self.inner.list(prefix)
    }

    async fn list_with_delimiter(
        &self,
        prefix: Option<&object_store::path::Path>,
    ) -> object_store::Result<object_store::ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }

    async fn copy_opts(
        &self,
        from: &object_store::path::Path,
        to: &object_store::path::Path,
        options: object_store::CopyOptions,
    ) -> object_store::Result<()> {
        self.inner.copy_opts(from, to, options).await
    }
}

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

/// Deterministic pseudo-random bytes for row `id`, so the wide columns do not
/// compress away and a scan that reads them really does pay for them.
fn noise(id: i64, salt: u64, len: usize) -> Vec<u8> {
    let mut state = (id as u64)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(salt)
        | 1;
    let mut out = Vec::with_capacity(len + 8);
    while out.len() < len {
        // xorshift64
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        out.extend_from_slice(&state.to_le_bytes());
    }
    out.truncate(len);
    out
}

fn payload(id: i64) -> Vec<u8> {
    noise(id, 1, PAYLOAD_BYTES)
}

fn note(id: i64) -> String {
    noise(id, 2, NOTE_CHARS / 2)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// `(payload, note, id, k2[, tag])`: the wide columns first, the keys after.
fn wide_schema(k2_type: &DataType, with_tag: bool) -> SchemaRef {
    let mut fields = vec![
        Field::new("payload", DataType::Binary, false),
        Field::new("note", DataType::Utf8, false),
        Field::new("id", DataType::Int64, false),
        Field::new("k2", k2_type.clone(), false),
    ];
    if with_tag {
        fields.push(Field::new("tag", DataType::Int32, true));
    }
    Arc::new(Schema::new(fields))
}

/// Default `k2`: ten values, each present in every row group, so a predicate
/// on it prunes nothing and the comparison is about projection alone.
fn k2_of(id: i64) -> i64 {
    id % 10
}

/// `tag` on a file written after the column was added.
fn tag_of(id: i64) -> Option<i32> {
    (id % 4 != 0).then_some((id % 3) as i32)
}

/// Rows `ids`, in `ROWS_PER_ROW_GROUP`-row batches.
fn wide_batches(
    ids: Range<i64>,
    k2_type: &DataType,
    k2: impl Fn(i64) -> i64,
    with_tag: bool,
) -> Vec<RecordBatch> {
    let schema = wide_schema(k2_type, with_tag);
    let ids: Vec<i64> = ids.collect();
    ids.chunks(ROWS_PER_ROW_GROUP)
        .map(|chunk| {
            let payloads: Vec<Vec<u8>> = chunk.iter().map(|&id| payload(id)).collect();
            let mut columns: Vec<ArrayRef> = vec![
                Arc::new(BinaryArray::from_iter_values(payloads.iter())),
                Arc::new(StringArray::from_iter_values(
                    chunk.iter().map(|&id| note(id)),
                )),
                Arc::new(Int64Array::from(chunk.to_vec())),
            ];
            columns.push(match k2_type {
                DataType::Int32 => Arc::new(Int32Array::from_iter_values(
                    chunk.iter().map(|&id| i32::try_from(k2(id)).unwrap()),
                )),
                DataType::Int64 => {
                    Arc::new(Int64Array::from_iter_values(chunk.iter().map(|&id| k2(id))))
                },
                other => panic!("unsupported k2 type {other}"),
            });
            if with_tag {
                columns.push(Arc::new(Int32Array::from_iter(
                    chunk.iter().map(|&id| tag_of(id)),
                )));
            }
            RecordBatch::try_new(Arc::clone(&schema), columns).unwrap()
        })
        .collect()
}

fn db_url(temp: &TempDir) -> String {
    format!("sqlite:{}?mode=rwc", temp.path().join("test.db").display())
}

async fn new_writer(temp: &TempDir) -> Arc<SqliteMetadataWriter> {
    let data_path = temp.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();
    let writer = SqliteMetadataWriter::new_with_init(&db_url(temp))
        .await
        .unwrap();
    writer.set_data_path(data_path.to_str().unwrap()).unwrap();
    Arc::new(writer)
}

fn table_writer(writer: Arc<SqliteMetadataWriter>) -> DuckLakeTableWriter {
    DuckLakeTableWriter::new(writer, Arc::new(LocalFileSystem::new()))
        .unwrap()
        .with_max_row_group_rows(ROWS_PER_ROW_GROUP)
}

/// Replace `main.t` with rows `ids` as one data file; returns the table id.
async fn seed(temp: &TempDir, ids: Range<i64>) -> i64 {
    let writer = new_writer(temp).await;
    table_writer(writer)
        .write_table(
            "main",
            "t",
            &wide_batches(ids, &DataType::Int32, k2_of, false),
        )
        .await
        .unwrap()
        .table_id
}

/// Append one more data file of `batches` to `main.t`.
async fn append(temp: &TempDir, batches: &[RecordBatch]) {
    let writer = Arc::new(SqliteMetadataWriter::new(&db_url(temp)).await.unwrap());
    table_writer(writer)
        .append_table("main", "t", batches)
        .await
        .unwrap();
}

/// A session over the catalog's current head, reading through a fresh
/// [`ByteCountingStore`], split across partitions so every scan runs the
/// parallel case.
struct Harness {
    ctx: SessionContext,
    store: Arc<ByteCountingStore>,
}

async fn open(temp: &TempDir) -> Harness {
    let mut config = ConfigOptions::new();
    config.execution.target_partitions = 8;
    config.optimizer.repartition_file_scans = true;
    config.optimizer.repartition_file_min_size = 1;
    let ctx = SessionContext::new_with_config(SessionConfig::from(config));

    let store = Arc::new(ByteCountingStore::new());
    ctx.runtime_env().register_object_store(
        &url::Url::parse("file:///").unwrap(),
        Arc::clone(&store) as Arc<dyn object_store::ObjectStore>,
    );

    let writer = SqliteMetadataWriter::new(&db_url(temp)).await.unwrap();
    let provider = SqliteMetadataProvider::new(&db_url(temp)).await.unwrap();
    let catalog = DuckLakeCatalog::with_writer(Arc::new(provider), Arc::new(writer))
        .unwrap()
        .with_write_options({
            let mut options = DuckLakeWriteOptions::default().with_data_inlining_row_limit(0);
            options.max_row_group_rows = Some(ROWS_PER_ROW_GROUP);
            options
        });
    ctx.register_catalog("ducklake", Arc::new(catalog));
    Harness {
        ctx,
        store,
    }
}

impl Harness {
    async fn table(&self) -> DuckLakeTable {
        let provider = self
            .ctx
            .catalog("ducklake")
            .unwrap()
            .schema("main")
            .unwrap()
            .table("t")
            .await
            .unwrap()
            .unwrap();
        (provider.as_ref() as &dyn std::any::Any)
            .downcast_ref::<DuckLakeTable>()
            .expect("provider is a DuckLakeTable")
            .clone()
    }

    /// `expr` planned against the table's catalog schema — how a keyed
    /// mutation builds the predicate it hands `resolve_positions`.
    fn predicate(&self, table: &DuckLakeTable, expr: Expr) -> Arc<dyn PhysicalExpr> {
        let schema = DFSchema::try_from(table.schema().as_ref().clone()).unwrap();
        self.ctx
            .state()
            .create_physical_expr(expr, &schema)
            .unwrap()
    }

    /// Resolve `expr` in `file`, returning the sorted positions and the bytes
    /// the resolution read.
    async fn resolve(
        &self,
        table: &DuckLakeTable,
        file: &DuckLakeTableFile,
        expr: Expr,
    ) -> (Vec<i64>, u64) {
        let predicate = self.predicate(table, expr);
        self.store.take();
        let positions = table
            .resolve_positions(&self.ctx.state(), &file.file, predicate)
            .await
            .unwrap();
        let bytes = self.store.take();
        let mut positions: Vec<i64> = positions.into_iter().collect();
        positions.sort_unstable();
        (positions, bytes)
    }

    /// Run a DML statement, returning its row count and the bytes it read.
    async fn dml(&self, sql: &str) -> (u64, u64) {
        self.store.take();
        let batches = self.ctx.sql(sql).await.unwrap().collect().await.unwrap();
        let bytes = self.store.take();
        let count = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<arrow::array::UInt64Array>()
            .expect("DML yields a UInt64 count")
            .value(0);
        (count, bytes)
    }
}

/// Rename a column the way a catalog does: close its current generation and
/// open another under the SAME column id, so the file keeps the old name and
/// the id is what ties the two together.
async fn rename_column(temp: &TempDir, table_id: i64, from: &str, to: &str, column_type: &str) {
    let pool = sqlx::sqlite::SqlitePool::connect(&db_url(temp))
        .await
        .unwrap();
    let next: i64 = sqlx::query_scalar("SELECT MAX(snapshot_id) + 1 FROM ducklake_snapshot")
        .fetch_one(&pool)
        .await
        .unwrap();
    let (column_id, column_order): (i64, i64) = sqlx::query_as(
        "SELECT column_id, column_order FROM ducklake_column
         WHERE table_id = ? AND column_name = ? AND end_snapshot IS NULL",
    )
    .bind(table_id)
    .bind(from)
    .fetch_one(&pool)
    .await
    .unwrap();
    sqlx::query(
        "INSERT INTO ducklake_snapshot (snapshot_id, snapshot_time, schema_version)
         SELECT ?, snapshot_time, schema_version + 1 FROM ducklake_snapshot
         ORDER BY snapshot_id DESC LIMIT 1",
    )
    .bind(next)
    .execute(&pool)
    .await
    .unwrap();
    sqlx::query(
        "UPDATE ducklake_column SET end_snapshot = ?
         WHERE table_id = ? AND column_id = ? AND end_snapshot IS NULL",
    )
    .bind(next)
    .bind(table_id)
    .bind(column_id)
    .execute(&pool)
    .await
    .unwrap();
    sqlx::query(
        "INSERT INTO ducklake_column
           (column_id, table_id, column_name, column_type, column_order, begin_snapshot,
            nulls_allowed)
         VALUES (?, ?, ?, ?, ?, ?, false)",
    )
    .bind(column_id)
    .bind(table_id)
    .bind(to)
    .bind(column_type)
    .bind(column_order)
    .bind(next)
    .execute(&pool)
    .await
    .unwrap();
    pool.close().await;
}

/// The on-disk path of a data file of `main.t`.
fn file_path(temp: &TempDir, file: &DuckLakeTableFile) -> PathBuf {
    if file.file.path_is_relative {
        temp.path()
            .join("data")
            .join("main")
            .join("t")
            .join(&file.file.path)
    } else {
        PathBuf::from(&file.file.path)
    }
}

/// One column of a parquet file in physical order, read by its PHYSICAL name
/// with the plain parquet reader and cast to Int64. This is the oracle's view
/// of the file: nothing in it goes through this crate.
fn physical_i64(path: &Path, column: &str) -> Vec<Option<i64>> {
    let builder =
        ParquetRecordBatchReaderBuilder::try_new(std::fs::File::open(path).unwrap()).unwrap();
    let index = builder
        .parquet_schema()
        .columns()
        .iter()
        .position(|c| c.name() == column)
        .unwrap_or_else(|| panic!("{} has no physical column {column}", path.display()));
    let mask = ProjectionMask::leaves(builder.parquet_schema(), [index]);
    let mut out = Vec::new();
    for batch in builder.with_projection(mask).build().unwrap() {
        let column = arrow::compute::cast(batch.unwrap().column(0), &DataType::Int64).unwrap();
        let column = column.as_any().downcast_ref::<Int64Array>().unwrap();
        out.extend(column.iter());
    }
    out
}

/// The physical column names of a parquet file.
fn physical_columns(path: &Path) -> Vec<String> {
    ParquetRecordBatchReaderBuilder::try_new(std::fs::File::open(path).unwrap())
        .unwrap()
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect()
}

/// The positions at which `keep` holds, over physically ordered rows.
fn oracle<T>(rows: &[T], keep: impl Fn(&T) -> bool) -> Vec<i64> {
    rows.iter()
        .enumerate()
        .filter(|(_, row)| keep(row))
        .map(|(position, _)| position as i64)
        .collect()
}

fn zip2(a: Vec<Option<i64>>, b: Vec<Option<i64>>) -> Vec<(Option<i64>, Option<i64>)> {
    assert_eq!(a.len(), b.len());
    a.into_iter().zip(b).collect()
}

fn assert_narrow(bytes: u64, file_bytes: u64, what: &str) {
    assert!(
        bytes > 0,
        "{what}: the counting store saw no reads, so the bound below proves nothing"
    );
    assert!(
        bytes * NARROW_FRACTION < file_bytes,
        "{what}: read {bytes} bytes of a {file_bytes}-byte file; a scan of the key \
         columns alone reads under 1/{NARROW_FRACTION} of it"
    );
}

fn file_bytes(file: &DuckLakeTableFile) -> u64 {
    u64::try_from(file.file.file_size_bytes).unwrap()
}

/// Live `(id, k2)` rows, ascending by id, and a check that every live row's
/// wide columns still hold what was written for its id.
async fn live_rows(temp: &TempDir) -> Vec<(i64, i64)> {
    let h = open(temp).await;
    let batches = h
        .ctx
        .sql("SELECT id, CAST(k2 AS BIGINT), payload, note FROM ducklake.main.t ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let mut out = Vec::new();
    for batch in &batches {
        let ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let k2 = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        // The read side may present these as view types; compare values.
        let payloads = arrow::compute::cast(batch.column(2), &DataType::Binary).unwrap();
        let payloads = payloads.as_any().downcast_ref::<BinaryArray>().unwrap();
        let notes = arrow::compute::cast(batch.column(3), &DataType::Utf8).unwrap();
        let notes = notes.as_any().downcast_ref::<StringArray>().unwrap();
        for row in 0..batch.num_rows() {
            let id = ids.value(row);
            assert_eq!(
                payloads.value(row),
                payload(id).as_slice(),
                "payload of {id}"
            );
            assert_eq!(notes.value(row), note(id), "note of {id}");
            out.push((id, k2.value(row)));
        }
    }
    out
}

fn expected_rows(ids: impl Iterator<Item = i64>, deleted: impl Fn(i64) -> bool) -> Vec<(i64, i64)> {
    ids.filter(|&id| !deleted(id))
        .map(|id| (id, k2_of(id)))
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// The base case, and the one the revert check runs against: a predicate on
/// narrow key columns reads a small fraction of a file whose rows are wide.
/// Every predicate shape is checked against the oracle: one key, a composite
/// key, a key on a non-leading column, a literal, and one that references a wide
/// column — the last proving the counter observes the reads it bounds.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn resolve_positions_reads_only_the_columns_the_predicate_names() {
    let temp = TempDir::new().unwrap();
    seed(&temp, 0..ROWS).await;
    let h = open(&temp).await;
    let table = h.table().await;
    let files = table.files().unwrap();
    assert_eq!(files.len(), 1);
    let file = &files[0];
    let path = file_path(&temp, file);
    let rows = zip2(physical_i64(&path, "id"), physical_i64(&path, "k2"));

    // One key column, matching in every row group.
    let (positions, bytes) = h.resolve(&table, file, col("k2").eq(lit(3i32))).await;
    assert_eq!(positions, oracle(&rows, |(_, k2)| *k2 == Some(3)));
    assert_eq!(positions.len(), ROWS as usize / 10);
    eprintln!(
        "resolve_positions(k2 = 3): read {bytes} of {} bytes",
        file_bytes(file)
    );
    assert_narrow(bytes, file_bytes(file), "k2 = 3");

    // A composite key, listed in the opposite order to the columns.
    let (positions, bytes) = h
        .resolve(
            &table,
            file,
            col("k2").eq(lit(7i32)).and(col("id").gt_eq(lit(5_000i64))),
        )
        .await;
    assert_eq!(
        positions,
        oracle(&rows, |(id, k2)| *k2 == Some(7) && id.unwrap() >= 5_000)
    );
    assert!(!positions.is_empty());
    assert_narrow(bytes, file_bytes(file), "k2 = 7 AND id >= 5000");

    // An IN list on the key, as a batched keyed delete issues.
    let wanted = [11i64, 2_222, 4_445, 7_999];
    let (positions, bytes) = h
        .resolve(
            &table,
            file,
            col("id").in_list(wanted.iter().map(|&v| lit(v)).collect(), false),
        )
        .await;
    assert_eq!(
        positions,
        oracle(&rows, |(id, _)| wanted.contains(&id.unwrap()))
    );
    assert_eq!(positions.len(), wanted.len());
    assert_narrow(bytes, file_bytes(file), "id IN (...)");

    // No column at all: every row matches, or none does. Positions come from
    // the reader's row numbering and the footer is already cached, so this
    // reads no data at all.
    let (positions, bytes) = h.resolve(&table, file, lit(true)).await;
    assert_eq!(positions, (0..ROWS).collect::<Vec<_>>());
    assert!(
        bytes * NARROW_FRACTION < file_bytes(file),
        "TRUE read {bytes} of {} bytes",
        file_bytes(file)
    );
    let (positions, _) = h.resolve(&table, file, lit(false)).await;
    assert!(positions.is_empty());

    // A predicate that does reference the wide column must read it — and the
    // counter must see it, or every bound above is vacuous.
    let (positions, bytes) = h
        .resolve(
            &table,
            file,
            col("payload").is_not_null().and(col("k2").eq(lit(3i32))),
        )
        .await;
    assert_eq!(positions, oracle(&rows, |(_, k2)| *k2 == Some(3)));
    assert!(
        bytes * 2 > file_bytes(file),
        "a predicate on the payload column read only {bytes} of {} bytes",
        file_bytes(file)
    );
}

/// The SQL `DELETE` path end to end: a keyed delete over a wide table reads
/// only the key columns, removes exactly its rows, and a second delete against
/// the file — which now carries a delete file — does the same.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn keyed_sql_delete_reads_only_key_columns_across_delete_files() {
    let temp = TempDir::new().unwrap();
    seed(&temp, 0..ROWS).await;
    let h = open(&temp).await;
    let total = file_bytes(&h.table().await.files().unwrap()[0]);

    let (deleted, bytes) = h
        .dml("DELETE FROM ducklake.main.t WHERE k2 = 3 AND id < 4000")
        .await;
    assert_eq!(deleted, 400);
    assert_narrow(bytes, total, "first DELETE");

    let h = open(&temp).await;
    let files = h.table().await.files().unwrap();
    assert!(
        files[0].delete_file.is_some(),
        "the first DELETE must leave a delete file on the data file"
    );
    let (deleted, bytes) = h
        .dml("DELETE FROM ducklake.main.t WHERE id = 5 OR (id = 4013 AND k2 = 3)")
        .await;
    assert_eq!(deleted, 2);
    assert_narrow(bytes, total, "DELETE on a file carrying a delete file");

    assert_eq!(
        live_rows(&temp).await,
        expected_rows(0..ROWS, |id| (k2_of(id) == 3 && id < 4000)
            || id == 5
            || id == 4013),
    );
}

/// A renamed key column: the file still names it `id`, the catalog calls it
/// `key`. The predicate is never pushed into the reader for such a column, so
/// this is the case where projection is the whole of the saving.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn renamed_key_column() {
    let temp = TempDir::new().unwrap();
    let table_id = seed(&temp, 0..ROWS).await;
    rename_column(&temp, table_id, "id", "key", "int64").await;

    let h = open(&temp).await;
    let table = h.table().await;
    assert_eq!(table.schema().field(2).name(), "key");
    let file = &table.files().unwrap()[0];
    let path = file_path(&temp, file);
    assert!(physical_columns(&path).contains(&"id".to_string()));
    let rows = zip2(physical_i64(&path, "id"), physical_i64(&path, "k2"));

    let (positions, bytes) = h
        .resolve(
            &table,
            file,
            col("key").lt(lit(300i64)).and(col("k2").eq(lit(4i32))),
        )
        .await;
    assert_eq!(
        positions,
        oracle(&rows, |(id, k2)| id.unwrap() < 300 && *k2 == Some(4))
    );
    assert_eq!(positions.len(), 30);
    assert_narrow(bytes, file_bytes(file), "renamed key");

    let (deleted, bytes) = h.dml("DELETE FROM ducklake.main.t WHERE key = 4321").await;
    assert_eq!(deleted, 1);
    assert_narrow(bytes, file_bytes(file), "DELETE on a renamed key");
    let h = open(&temp).await;
    let remaining = h
        .ctx
        .sql("SELECT count(*) FROM ducklake.main.t WHERE key = 4321 OR key = 4320")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(
        remaining[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0),
        1
    );
}

/// A widened key column: the first file stores `k2` as Int32, the catalog now
/// declares Int64, and a later file holds values beyond the Int32 range. The
/// narrow scan must still cast the old file's column on read.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn widened_key_column() {
    let temp = TempDir::new().unwrap();
    let table_id = seed(&temp, 0..ROWS).await;
    let writer = SqliteMetadataWriter::new(&db_url(&temp)).await.unwrap();
    writer.promote_column_type(table_id, "k2", "int64").unwrap();
    let beyond_i32 = 5_000_000_000i64;
    append(
        &temp,
        &wide_batches(
            ROWS..ROWS + 2 * ROWS_PER_ROW_GROUP as i64,
            &DataType::Int64,
            |id| {
                if id % 2 == 0 {
                    beyond_i32
                } else {
                    k2_of(id)
                }
            },
            false,
        ),
    )
    .await;

    let h = open(&temp).await;
    let table = h.table().await;
    assert_eq!(table.schema().field(3).data_type(), &DataType::Int64);
    let mut files = table.files().unwrap();
    files.sort_by_key(|f| f.data_file_id);
    let (old, new) = (&files[0], &files[1]);

    let old_rows = zip2(
        physical_i64(&file_path(&temp, old), "id"),
        physical_i64(&file_path(&temp, old), "k2"),
    );
    let (positions, bytes) = h.resolve(&table, old, col("k2").eq(lit(3i64))).await;
    assert_eq!(positions, oracle(&old_rows, |(_, k2)| *k2 == Some(3)));
    assert_eq!(positions.len(), ROWS as usize / 10);
    assert_narrow(bytes, file_bytes(old), "widened key, Int32 file");

    let new_rows = zip2(
        physical_i64(&file_path(&temp, new), "id"),
        physical_i64(&file_path(&temp, new), "k2"),
    );
    let (positions, bytes) = h.resolve(&table, new, col("k2").eq(lit(beyond_i32))).await;
    assert_eq!(
        positions,
        oracle(&new_rows, |(_, k2)| *k2 == Some(beyond_i32))
    );
    assert_eq!(positions.len(), ROWS_PER_ROW_GROUP);
    assert_narrow(bytes, file_bytes(new), "widened key, Int64 file");
}

/// A key column added after the first file was written. That file predates
/// it, so the column reads as NULL there; the later file carries real values.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn key_column_the_file_predates() {
    let temp = TempDir::new().unwrap();
    seed(&temp, 0..ROWS).await;
    append(
        &temp,
        &wide_batches(
            ROWS..ROWS + 2 * ROWS_PER_ROW_GROUP as i64,
            &DataType::Int32,
            k2_of,
            true,
        ),
    )
    .await;

    let h = open(&temp).await;
    let table = h.table().await;
    assert_eq!(table.schema().field(4).name(), "tag");
    let mut files = table.files().unwrap();
    files.sort_by_key(|f| f.data_file_id);
    let (old, new) = (&files[0], &files[1]);
    assert!(!physical_columns(&file_path(&temp, old)).contains(&"tag".to_string()));

    let old_rows = zip2(
        physical_i64(&file_path(&temp, old), "id"),
        physical_i64(&file_path(&temp, old), "k2"),
    );
    let (positions, bytes) = h
        .resolve(
            &table,
            old,
            col("tag").is_null().and(col("k2").eq(lit(6i32))),
        )
        .await;
    assert_eq!(positions, oracle(&old_rows, |(_, k2)| *k2 == Some(6)));
    assert_eq!(positions.len(), ROWS as usize / 10);
    assert_narrow(bytes, file_bytes(old), "added column, file predates it");
    let (positions, _) = h.resolve(&table, old, col("tag").eq(lit(1i32))).await;
    assert!(positions.is_empty(), "the old file holds no tag values");

    let new_rows = zip2(
        physical_i64(&file_path(&temp, new), "id"),
        physical_i64(&file_path(&temp, new), "tag"),
    );
    let (positions, bytes) = h
        .resolve(
            &table,
            new,
            col("tag")
                .eq(lit(1i32))
                .and(col("id").lt(lit(ROWS + 1_000))),
        )
        .await;
    assert_eq!(
        positions,
        oracle(&new_rows, |(id, tag)| *tag == Some(1)
            && id.unwrap() < ROWS + 1_000)
    );
    assert!(!positions.is_empty());
    assert_narrow(bytes, file_bytes(new), "added column, file carries it");
}

/// A partitioned table: one file per `k2` value. A keyed delete resolves in
/// each candidate file with a narrow scan and removes exactly its rows.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn partitioned_table() {
    let temp = TempDir::new().unwrap();
    let writer = new_writer(&temp).await;
    let schema = wide_schema(&DataType::Int32, false);
    let columns: Vec<ColumnDef> = schema
        .fields()
        .iter()
        .map(|f| ColumnDef::from_arrow(f.name(), f.data_type(), f.is_nullable()).unwrap())
        .collect();
    let txn = writer
        .begin_write_transaction("main", "t", &columns, WriteMode::Replace)
        .unwrap();
    writer
        .publish_snapshot(
            txn.table_id,
            "main",
            "t",
            txn.snapshot_id,
            WriteMode::Replace,
            txn.base_snapshot_id,
            &columns,
            &txn.column_ids,
        )
        .unwrap();
    writer
        .set_partition_spec(
            txn.table_id,
            &[("k2".to_string(), PartitionTransform::Identity)],
        )
        .unwrap();

    let h = open(&temp).await;
    let source = MemTable::try_new(
        Arc::clone(&schema),
        vec![wide_batches(0..ROWS, &DataType::Int32, k2_of, false)],
    )
    .unwrap();
    h.ctx.register_table("source", Arc::new(source)).unwrap();
    h.ctx
        .sql("INSERT INTO ducklake.main.t SELECT * FROM source")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    let h = open(&temp).await;
    let table = h.table().await;
    let files = table.files().unwrap();
    assert_eq!(files.len(), 10, "one file per k2 value");
    let mut resolved = 0;
    for file in &files {
        let path = file_path(&temp, file);
        let rows = zip2(physical_i64(&path, "id"), physical_i64(&path, "k2"));
        let (positions, bytes) = h
            .resolve(
                &table,
                file,
                col("k2").eq(lit(3i32)).and(col("id").lt(lit(500i64))),
            )
            .await;
        assert_eq!(
            positions,
            oracle(&rows, |(id, k2)| *k2 == Some(3) && id.unwrap() < 500)
        );
        resolved += positions.len();
        assert_narrow(bytes, file_bytes(file), "partitioned file");
    }
    assert_eq!(resolved, 50);

    let (deleted, _) = h
        .dml("DELETE FROM ducklake.main.t WHERE k2 = 3 AND id < 500")
        .await;
    assert_eq!(deleted, 50);
    assert_eq!(
        live_rows(&temp).await,
        expected_rows(0..ROWS, |id| k2_of(id) == 3 && id < 500),
    );
}

/// A merged file embeds each row's rowid (and origin snapshot) as extra
/// physical columns after the data columns. The narrow scan must neither read
/// them nor let them shift a key column's index.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn file_that_embeds_row_ids() {
    let temp = TempDir::new().unwrap();
    let third = ROWS / 3;
    seed(&temp, 0..third).await;
    for start in [third, 2 * third] {
        append(
            &temp,
            &wide_batches(start..start + third, &DataType::Int32, k2_of, false),
        )
        .await;
    }
    let h = open(&temp).await;
    let result = h
        .table()
        .await
        .merge_adjacent_files(&h.ctx.state(), MergeOptions::default())
        .await
        .unwrap();
    assert_eq!(result.files_created, 1);

    let h = open(&temp).await;
    let table = h.table().await;
    let files = table.files().unwrap();
    assert_eq!(files.len(), 1);
    let file = &files[0];
    let path = file_path(&temp, file);
    assert!(
        physical_columns(&path).contains(&"_ducklake_internal_row_id".to_string()),
        "the merged file must embed row ids, or this test covers nothing new"
    );
    let rows = zip2(physical_i64(&path, "id"), physical_i64(&path, "k2"));

    let (positions, bytes) = h
        .resolve(
            &table,
            file,
            col("id")
                .in_list(vec![lit(1i64), lit(third + 1), lit(2 * third + 1)], false)
                .or(col("k2").eq(lit(9i32)).and(col("id").lt(lit(100i64)))),
        )
        .await;
    assert_eq!(
        positions,
        oracle(&rows, |(id, k2)| {
            let id = id.unwrap();
            [1, third + 1, 2 * third + 1].contains(&id) || (*k2 == Some(9) && id < 100)
        })
    );
    assert_eq!(positions.len(), 13);
    assert_narrow(bytes, file_bytes(file), "merged file");

    let (deleted, _) = h
        .dml("DELETE FROM ducklake.main.t WHERE id = 1 OR id = 5001")
        .await;
    assert_eq!(deleted, 2);
    assert_eq!(
        live_rows(&temp).await,
        expected_rows(0..3 * third, |id| id == 1 || id == 5001),
    );
}

/// A file registered through `ducklake_add_data_files`: it has no field ids,
/// so its columns are found through a name mapping, the key is renamed in the
/// catalog after registration, and the partition column is a Hive path
/// constant that is not in the file at all.
#[cfg(feature = "metadata-duckdb")]
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn name_mapped_file() {
    use datafusion_ducklake::DuckdbMetadataProvider;

    let temp = TempDir::new().unwrap();
    let catalog_path = temp.path().join("mapped.ducklake");
    let data_path = temp.path().join("data");
    let hive = data_path.join("part=10");
    std::fs::create_dir_all(&hive).unwrap();
    let parquet_path = hive.join("mapped.parquet");

    crate::common::ensure_extension_installed("ducklake");
    crate::common::ensure_extension_installed("parquet");
    let conn = duckdb::Connection::open_in_memory().unwrap();
    conn.execute_batch("LOAD ducklake; LOAD parquet;").unwrap();
    conn.execute_batch(&format!(
        "ATTACH 'ducklake:{}' AS lake (DATA_PATH '{}', DATA_INLINING_ROW_LIMIT 0);
         CREATE TABLE lake.mapped(wide VARCHAR, source_id BIGINT, k2 INTEGER, part INTEGER);
         COPY (
             SELECT (SELECT string_agg(md5(i::VARCHAR || '-' || j::VARCHAR), '')
                     FROM range(48) r(j)) AS wide,
                    i AS source_id,
                    (i % 10)::INTEGER AS k2
             FROM range({ROWS}) t(i)
         ) TO '{}' (FORMAT PARQUET, ROW_GROUP_SIZE {ROWS_PER_ROW_GROUP});
         CALL ducklake_add_data_files('lake', 'mapped', '{}/**/*.parquet',
                                      hive_partitioning => true);
         ALTER TABLE lake.mapped RENAME COLUMN source_id TO id;",
        catalog_path.display(),
        data_path.display(),
        parquet_path.display(),
        data_path.display(),
    ))
    .unwrap();
    drop(conn);

    let mut config = ConfigOptions::new();
    config.execution.target_partitions = 8;
    config.optimizer.repartition_file_scans = true;
    config.optimizer.repartition_file_min_size = 1;
    let ctx = SessionContext::new_with_config(SessionConfig::from(config));
    let store = Arc::new(ByteCountingStore::new());
    ctx.runtime_env().register_object_store(
        &url::Url::parse("file:///").unwrap(),
        Arc::clone(&store) as Arc<dyn object_store::ObjectStore>,
    );
    let provider = DuckdbMetadataProvider::new(catalog_path.to_string_lossy()).unwrap();
    ctx.register_catalog(
        "ducklake",
        Arc::new(DuckLakeCatalog::new(provider).unwrap()),
    );
    let h = Harness {
        ctx,
        store,
    };

    let provider = h
        .ctx
        .catalog("ducklake")
        .unwrap()
        .schema("main")
        .unwrap()
        .table("mapped")
        .await
        .unwrap()
        .unwrap();
    let table = (provider.as_ref() as &dyn std::any::Any)
        .downcast_ref::<DuckLakeTable>()
        .unwrap()
        .clone();
    let files = table.files().unwrap();
    assert_eq!(files.len(), 1);
    let file = &files[0];
    assert!(
        file.file.mapping_id.is_some(),
        "the file must be read through a name mapping"
    );
    let columns = physical_columns(&parquet_path);
    assert!(columns.contains(&"source_id".to_string()));
    assert!(!columns.contains(&"part".to_string()));
    let rows = zip2(
        physical_i64(&parquet_path, "source_id"),
        physical_i64(&parquet_path, "k2"),
    );
    let size = std::fs::metadata(&parquet_path).unwrap().len();

    let (positions, bytes) = h
        .resolve(
            &table,
            file,
            col("id").lt(lit(700i64)).and(col("k2").eq(lit(2i32))),
        )
        .await;
    assert_eq!(
        positions,
        oracle(&rows, |(id, k2)| id.unwrap() < 700 && *k2 == Some(2))
    );
    assert_eq!(positions.len(), 70);
    assert_narrow(bytes, size, "name-mapped file");

    // On the Hive constant alone, and combined with a mapped key.
    let (positions, bytes) = h.resolve(&table, file, col("part").eq(lit(10i32))).await;
    assert_eq!(positions, (0..ROWS).collect::<Vec<_>>());
    // The constant is not in the file, so this reads no column data at all.
    assert!(
        bytes * NARROW_FRACTION < size,
        "Hive constant read {bytes} of {size} bytes"
    );
    let (positions, _) = h
        .resolve(
            &table,
            file,
            col("part").eq(lit(10i32)).and(col("id").eq(lit(4_242i64))),
        )
        .await;
    assert_eq!(positions, oracle(&rows, |(id, _)| *id == Some(4_242)));
    let (positions, _) = h.resolve(&table, file, col("part").eq(lit(9i32))).await;
    assert!(positions.is_empty());

    // The wide column itself resolves too, and is read.
    let (positions, bytes) = h
        .resolve(
            &table,
            file,
            col("wide").is_not_null().and(col("id").eq(lit(17i64))),
        )
        .await;
    assert_eq!(positions, oracle(&rows, |(id, _)| *id == Some(17)));
    assert!(
        bytes * 2 > size,
        "a predicate on `wide` read only {bytes} of {size}"
    );

    // Guard against an oracle that could agree by accident.
    let unique: HashSet<_> = rows.iter().map(|(id, _)| *id).collect();
    assert_eq!(unique.len(), ROWS as usize);
}
