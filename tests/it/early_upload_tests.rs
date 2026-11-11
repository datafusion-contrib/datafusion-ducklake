//! A streaming write uploads each data file as soon as it is finished, and still
//! commits all of them in one snapshot at `finish`. Official DuckLake also writes
//! its insert straight into the table's data path and registers the files only at
//! commit.
//!
//! What these tests pin is what makes early upload safe rather than the upload
//! timing itself: every file is uploaded once and committed once, in write order,
//! and a write that does not commit leaves no object behind and no file in the
//! catalog. That covers an upload failure, an abort and a dropped session, which
//! this crate cleans up and official does not (it learns of a statement's files
//! only once the statement finishes), and a commit that fails with nothing
//! committed, whose files both remove. A `COMMIT` whose outcome is unknown keeps
//! its files, where official removes them.

#![cfg(all(feature = "write-sqlite", feature = "metadata-sqlite"))]

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use arrow::array::{Array, Int32Array, Int64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::prelude::SessionContext;
use futures::StreamExt;
use object_store::ObjectStore;
use object_store::local::LocalFileSystem;
use object_store::path::Path as ObjectPath;
use sqlx::sqlite::SqlitePool;
use tempfile::TempDir;

use datafusion_ducklake::partition::PartitionTransform;
use datafusion_ducklake::{
    ColumnDef, CommitIds, DataFileInfo, DeleteFileEntry, DuckLakeCatalog, DuckLakeError,
    DuckLakeTableWriter, DuckLakeWriteOptions, EncodePool, MetadataWriter, SnapshotCommitMetadata,
    SqliteMetadataProvider, SqliteMetadataWriter, TableWriteOptions, TableWriteSession, WriteMode,
    WriteSetupResult,
};

/// Small row groups written as Parquet V1, so a write of a few thousand rows
/// rolls into many files at a 4 KiB target (a file rolls only once its first row
/// group is complete).
fn rolling_write_options() -> DuckLakeWriteOptions {
    let mut options = DuckLakeWriteOptions::default();
    options.max_row_group_rows = Some(64);
    options.parquet_version = Some(parquet::file::properties::WriterVersion::PARQUET_1_0);
    options
}

/// A store that records every write it is asked to start, by object key, and can
/// fail the Nth one.
#[derive(Debug)]
struct RecordingStore {
    inner: Arc<dyn ObjectStore>,
    writes: Mutex<HashMap<String, usize>>,
    attempts: AtomicUsize,
    fail_on: Option<usize>,
}

impl RecordingStore {
    fn new(inner: Arc<dyn ObjectStore>, fail_on: Option<usize>) -> Self {
        Self {
            inner,
            writes: Mutex::new(HashMap::new()),
            attempts: AtomicUsize::new(0),
            fail_on,
        }
    }

    /// Count a write to `location`, and fail it if it is the configured one.
    fn start_write(&self, location: &ObjectPath) -> object_store::Result<()> {
        *self
            .writes
            .lock()
            .unwrap()
            .entry(location.to_string())
            .or_default() += 1;
        let attempt = self.attempts.fetch_add(1, Ordering::SeqCst) + 1;
        if self.fail_on == Some(attempt) {
            return Err(object_store::Error::Generic {
                store: "RecordingStore",
                source: "injected upload failure".into(),
            });
        }
        Ok(())
    }

    fn writes(&self) -> HashMap<String, usize> {
        self.writes.lock().unwrap().clone()
    }
}

impl std::fmt::Display for RecordingStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RecordingStore({})", self.inner)
    }
}

#[async_trait::async_trait]
impl ObjectStore for RecordingStore {
    async fn put_opts(
        &self,
        location: &ObjectPath,
        payload: object_store::PutPayload,
        opts: object_store::PutOptions,
    ) -> object_store::Result<object_store::PutResult> {
        self.start_write(location)?;
        self.inner.put_opts(location, payload, opts).await
    }

    async fn put_multipart_opts(
        &self,
        location: &ObjectPath,
        opts: object_store::PutMultipartOptions,
    ) -> object_store::Result<Box<dyn object_store::MultipartUpload>> {
        self.start_write(location)?;
        self.inner.put_multipart_opts(location, opts).await
    }

    async fn get_opts(
        &self,
        location: &ObjectPath,
        options: object_store::GetOptions,
    ) -> object_store::Result<object_store::GetResult> {
        self.inner.get_opts(location, options).await
    }

    fn delete_stream(
        &self,
        locations: futures::stream::BoxStream<'static, object_store::Result<ObjectPath>>,
    ) -> futures::stream::BoxStream<'static, object_store::Result<ObjectPath>> {
        self.inner.delete_stream(locations)
    }

    fn list(
        &self,
        prefix: Option<&ObjectPath>,
    ) -> futures::stream::BoxStream<'static, object_store::Result<object_store::ObjectMeta>> {
        self.inner.list(prefix)
    }

    async fn list_with_delimiter(
        &self,
        prefix: Option<&ObjectPath>,
    ) -> object_store::Result<object_store::ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }

    async fn copy_opts(
        &self,
        from: &ObjectPath,
        to: &ObjectPath,
        options: object_store::CopyOptions,
    ) -> object_store::Result<()> {
        self.inner.copy_opts(from, to, options).await
    }
}

/// A SQLite catalog whose data path is `<temp>/data`, and a recording store over
/// the local filesystem.
struct Env {
    temp: TempDir,
    conn_str: String,
    writer: Arc<SqliteMetadataWriter>,
    store: Arc<RecordingStore>,
}

impl Env {
    async fn new(fail_on: Option<usize>) -> Self {
        let temp = TempDir::new().unwrap();
        let data_path = temp.path().join("data");
        std::fs::create_dir_all(&data_path).unwrap();
        let conn_str = format!("sqlite:{}?mode=rwc", temp.path().join("test.db").display());
        let writer = SqliteMetadataWriter::new_with_init(&conn_str)
            .await
            .unwrap();
        writer.set_data_path(data_path.to_str().unwrap()).unwrap();
        let store = Arc::new(RecordingStore::new(
            Arc::new(LocalFileSystem::new()),
            fail_on,
        ));
        Self {
            temp,
            conn_str,
            writer: Arc::new(writer),
            store,
        }
    }

    fn table_writer(&self, concurrency: usize) -> DuckLakeTableWriter {
        let store: Arc<dyn ObjectStore> = self.store.clone();
        DuckLakeTableWriter::new(self.writer.clone(), store)
            .unwrap()
            .with_target_file_size(4 * 1024)
            .with_options(&rolling_write_options())
            .with_upload_concurrency(concurrency)
    }

    /// Store the catalog-wide `per_thread_output` option, as official's
    /// `set_option` does.
    fn set_per_thread_output(&self, value: &str) {
        self.writer
            .set_global_setting("per_thread_output", value)
            .unwrap();
    }

    /// The parquet files present under the data path, by file name.
    fn parquet_on_disk(&self) -> Vec<String> {
        fn walk(dir: &std::path::Path, out: &mut Vec<String>) {
            for entry in std::fs::read_dir(dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    walk(&path, out);
                } else if path.extension().is_some_and(|e| e == "parquet") {
                    out.push(path.file_name().unwrap().to_string_lossy().into_owned());
                }
            }
        }
        let mut out = Vec::new();
        walk(&self.temp.path().join("data"), &mut out);
        out.sort();
        out
    }

    /// The live data files the catalog records, by file name.
    async fn committed_files(&self) -> Vec<String> {
        let pool = SqlitePool::connect(&self.conn_str).await.unwrap();
        let paths: Vec<(String,)> =
            sqlx::query_as("SELECT path FROM ducklake_data_file WHERE end_snapshot IS NULL")
                .fetch_all(&pool)
                .await
                .unwrap();
        let mut names: Vec<String> = paths
            .into_iter()
            .map(|(p,)| p.rsplit('/').next().unwrap().to_string())
            .collect();
        names.sort();
        names
    }

    /// Wait until the number of parquet files under the data path satisfies `done`.
    async fn wait_for_parquet_count(&self, what: &str, done: impl Fn(usize) -> bool) {
        let deadline = Instant::now() + Duration::from_secs(60);
        loop {
            let count = self.parquet_on_disk().len();
            if done(count) {
                return;
            }
            assert!(
                Instant::now() < deadline,
                "{what}: {count} parquet files on disk"
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }
}

fn table_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("val", DataType::Int32, false),
    ]))
}

/// `batches` batches of ascending ids, with uneven sizes so files differ in row
/// count.
fn batches(schema: &Arc<Schema>, batches: i32) -> Vec<RecordBatch> {
    let mut next_id = 0i32;
    (0..batches)
        .map(|b| {
            let count = 40 + (b % 7) * 37;
            let ids: Vec<i32> = (next_id..next_id + count).collect();
            next_id += count;
            let vals: Vec<i32> = ids.iter().map(|id| id * 10).collect();
            RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(Int32Array::from(ids)), Arc::new(Int32Array::from(vals))],
            )
            .unwrap()
        })
        .collect()
}

#[derive(Clone, Copy, Debug)]
enum Feed {
    /// `write_batch`: never waits for an upload.
    Sync,
    /// `write_batch_async`: waits for an upload slot after each batch.
    Async,
}

async fn feed(
    session: &mut TableWriteSession,
    batch: &RecordBatch,
    how: Feed,
) -> datafusion_ducklake::Result<()> {
    match how {
        Feed::Sync => session.write_batch(batch),
        Feed::Async => session.write_batch_async(batch).await,
    }
}

/// Every file is uploaded exactly once, and the snapshot registers exactly the
/// files that were uploaded: nothing missing, nothing extra, no stray object.
#[tokio::test(flavor = "multi_thread")]
async fn every_file_is_uploaded_once_and_committed_once() {
    for how in [Feed::Sync, Feed::Async] {
        let env = Env::new(None).await;
        let schema = table_schema();
        let input = batches(&schema, 60);
        let mut session = env
            .table_writer(3)
            .begin_write("main", "t", schema.as_ref(), WriteMode::Append)
            .unwrap();
        for batch in &input {
            feed(&mut session, batch, how).await.unwrap();
        }
        let result = session.finish().await.unwrap();
        assert!(
            result.files_written > 5,
            "{how:?}: the write must roll, got {}",
            result.files_written
        );

        let writes = env.store.writes();
        assert!(
            writes.values().all(|n| *n == 1),
            "{how:?}: every object must be written exactly once: {writes:?}"
        );
        let mut written: Vec<String> = writes
            .keys()
            .map(|k| k.rsplit('/').next().unwrap().to_string())
            .collect();
        written.sort();
        let committed = env.committed_files().await;
        assert_eq!(committed.len(), result.files_written, "{how:?}");
        assert_eq!(
            written, committed,
            "{how:?}: uploaded and committed files must match"
        );
        assert_eq!(env.parquet_on_disk(), committed, "{how:?}: no stray object");
    }
}

/// A rolling write uploads its files while it is still being written, rather than
/// all of them at `finish`. The sync `write_batch` starts them in the background.
#[tokio::test(flavor = "multi_thread")]
async fn files_upload_before_finish() {
    let env = Env::new(None).await;
    let schema = table_schema();
    let mut session = env
        .table_writer(4)
        .begin_write("main", "t", schema.as_ref(), WriteMode::Append)
        .unwrap();
    for batch in &batches(&schema, 60) {
        session.write_batch(batch).unwrap();
    }
    env.wait_for_parquet_count("files must upload before finish", |n| n > 0)
        .await;
    assert!(
        env.committed_files().await.is_empty(),
        "an uploaded file must not be visible before the commit"
    );
    let result = session.finish().await.unwrap();
    assert_eq!(env.committed_files().await.len(), result.files_written);
}

/// An upload failure fails the write, commits nothing, and removes every object
/// that already landed — whether the failure surfaces from a write or from
/// `finish`.
#[tokio::test(flavor = "multi_thread")]
async fn a_failed_upload_commits_nothing_and_leaves_nothing() {
    for how in [Feed::Sync, Feed::Async] {
        let env = Env::new(Some(4)).await;
        let schema = table_schema();
        let mut session = env
            .table_writer(2)
            .begin_write("main", "t", schema.as_ref(), WriteMode::Append)
            .unwrap();
        let mut early = None;
        for batch in &batches(&schema, 120) {
            if let Err(e) = feed(&mut session, batch, how).await {
                early = Some(e);
                break;
            }
        }
        let err = match early {
            Some(err) => {
                session.abort().await.unwrap();
                err
            },
            None => session
                .finish()
                .await
                .expect_err("the upload failure must fail the write"),
        };
        assert!(
            err.to_string().contains("injected upload failure"),
            "{how:?}: the upload error must surface, got: {err}"
        );
        assert!(
            env.store.writes().len() >= 4,
            "{how:?}: some files must have landed first"
        );
        assert_eq!(
            env.parquet_on_disk(),
            Vec::<String>::new(),
            "{how:?}: nothing may be left"
        );
        assert!(
            env.committed_files().await.is_empty(),
            "{how:?}: nothing may be committed"
        );
    }
}

/// `abort` removes every file the write already uploaded, and commits nothing.
#[tokio::test(flavor = "multi_thread")]
async fn abort_removes_the_uploaded_files() {
    let env = Env::new(None).await;
    let schema = table_schema();
    let mut session = env
        .table_writer(3)
        .begin_write("main", "t", schema.as_ref(), WriteMode::Append)
        .unwrap();
    for batch in &batches(&schema, 60) {
        session.write_batch_async(batch).await.unwrap();
    }
    env.wait_for_parquet_count("files must upload before the abort", |n| n > 0)
        .await;
    session.abort().await.unwrap();
    assert_eq!(env.parquet_on_disk(), Vec::<String>::new());
    assert!(env.committed_files().await.is_empty());
}

/// A session dropped without `finish` — the caller stopped on an error, or its
/// future was cancelled — removes what it uploaded in the background.
#[tokio::test(flavor = "multi_thread")]
async fn a_dropped_session_removes_the_uploaded_files() {
    let env = Env::new(None).await;
    let schema = table_schema();
    let mut session = env
        .table_writer(3)
        .begin_write("main", "t", schema.as_ref(), WriteMode::Append)
        .unwrap();
    for batch in &batches(&schema, 60) {
        session.write_batch(batch).unwrap();
    }
    env.wait_for_parquet_count("files must upload before the drop", |n| n > 0)
        .await;
    drop(session);
    env.wait_for_parquet_count("a dropped session must remove its uploads", |n| n == 0)
        .await;
    assert!(env.committed_files().await.is_empty());
}

/// A commit rejected before it applied — here a fence conflict — registered
/// nothing, so the files uploaded for it are removed, as official DuckLake removes a
/// failed transaction's files. The table's committed files are untouched.
#[tokio::test(flavor = "multi_thread")]
async fn a_rejected_commit_removes_the_files_it_uploaded() {
    let env = Env::new(None).await;
    let schema = table_schema();
    let table_writer = env.table_writer(3);
    let initial = table_writer
        .write_table("main", "t", &batches(&schema, 2))
        .await
        .unwrap();
    table_writer
        .append_table("main", "t", &batches(&schema, 2))
        .await
        .unwrap();
    let committed_before = env.committed_files().await;
    assert_eq!(env.parquet_on_disk(), committed_before);

    let mut session = table_writer
        .begin_write("main", "t", schema.as_ref(), WriteMode::Replace)
        .unwrap()
        .with_options(
            &TableWriteOptions::new().with_expected_base_snapshot_id(initial.snapshot_id),
        );
    for batch in &batches(&schema, 60) {
        session.write_batch_async(batch).await.unwrap();
    }
    let err = session
        .finish()
        .await
        .expect_err("a stale base must conflict");
    assert!(
        matches!(err, datafusion_ducklake::DuckLakeError::Conflict(_)),
        "got: {err}"
    );
    assert!(
        env.store.writes().len() > committed_before.len() + 5,
        "the rejected write must have uploaded several files"
    );
    assert_eq!(env.committed_files().await, committed_before);
    assert_eq!(
        env.parquet_on_disk(),
        committed_before,
        "the rejected write's files must be removed, and only those"
    );
}

/// Create `events(id, region)` partitioned by `region`.
fn create_partitioned_table(writer: &SqliteMetadataWriter) {
    let cols = vec![
        ColumnDef::from_arrow("id", &DataType::Int32, false).unwrap(),
        ColumnDef::from_arrow("region", &DataType::Utf8, false).unwrap(),
    ];
    let s = writer
        .begin_write_transaction("main", "events", &cols, WriteMode::Replace)
        .unwrap();
    writer
        .publish_snapshot(
            s.table_id,
            "main",
            "events",
            s.snapshot_id,
            WriteMode::Replace,
            s.base_snapshot_id,
            &cols,
            &s.column_ids,
        )
        .unwrap();
    writer
        .set_partition_spec(
            s.table_id,
            &[("region".to_string(), PartitionTransform::Identity)],
        )
        .unwrap();
}

fn events_batches(count: i32) -> Vec<RecordBatch> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("region", DataType::Utf8, false),
    ]));
    (0..count)
        .map(|b| {
            let ids: Vec<i32> = (b * 200..(b + 1) * 200).collect();
            let regions: Vec<&str> = ids
                .iter()
                .map(|id| ["us", "eu", "ap"][(*id as usize) % 3])
                .collect();
            RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int32Array::from(ids)),
                    Arc::new(arrow::array::StringArray::from(regions)),
                ],
            )
            .unwrap()
        })
        .collect()
}

/// A partitioned write rolls and evicts files per partition; each is uploaded
/// once, committed once with its partition, and an abort removes them all.
#[tokio::test(flavor = "multi_thread")]
async fn a_partitioned_write_uploads_each_file_once_and_abort_removes_them() {
    // Commit: every file once, all registered.
    let env = Env::new(None).await;
    create_partitioned_table(&env.writer);
    let input = events_batches(30);
    let mut session = env
        .table_writer(3)
        .with_max_open_partitions(2)
        .begin_write(
            "main",
            "events",
            input[0].schema().as_ref(),
            WriteMode::Append,
        )
        .unwrap();
    for batch in &input {
        session.write_batch_async(batch).await.unwrap();
    }
    let result = session.finish().await.unwrap();
    assert!(result.files_written > 3, "got {}", result.files_written);
    let writes = env.store.writes();
    assert!(writes.values().all(|n| *n == 1), "{writes:?}");
    let committed = env.committed_files().await;
    assert_eq!(committed.len(), result.files_written);
    assert_eq!(env.parquet_on_disk(), committed);

    // Abort: nothing left, nothing committed.
    let env = Env::new(None).await;
    create_partitioned_table(&env.writer);
    let mut session = env
        .table_writer(3)
        .with_max_open_partitions(2)
        .begin_write(
            "main",
            "events",
            input[0].schema().as_ref(),
            WriteMode::Append,
        )
        .unwrap();
    for batch in &input {
        session.write_batch_async(batch).await.unwrap();
    }
    env.wait_for_parquet_count("partition files must upload before the abort", |n| n > 0)
        .await;
    session.abort().await.unwrap();
    assert_eq!(env.parquet_on_disk(), Vec::<String>::new());
    assert!(env.committed_files().await.is_empty());
}

/// Read `t` back through DataFusion as `(rowid, id, val)` ordered by `rowid`.
async fn read_back(conn_str: &str) -> Vec<(i64, i32, i32)> {
    let provider = SqliteMetadataProvider::new(conn_str).await.unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog(
        "ducklake",
        // `rowid` is a lineage column, present only when the catalog opts in.
        Arc::new(
            DuckLakeCatalog::new(provider)
                .unwrap()
                .with_row_lineage(true),
        ),
    );
    let mut stream = ctx
        .sql("SELECT rowid, id, val FROM ducklake.main.t ORDER BY rowid")
        .await
        .unwrap()
        .execute_stream()
        .await
        .unwrap();
    let mut rows = Vec::new();
    while let Some(batch) = stream.next().await {
        let batch = batch.unwrap();
        let rowid = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let id = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let val = batch
            .column(2)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        for i in 0..batch.num_rows() {
            assert!(!rowid.is_null(i));
            rows.push((rowid.value(i), id.value(i), val.value(i)));
        }
    }
    rows
}

/// A large multi-file write reads back identically however it was uploaded: one
/// upload at a time from `write_batch`, or many in flight from
/// `write_batch_async` — rows, values and row ids alike.
#[tokio::test(flavor = "multi_thread")]
async fn a_multi_file_write_reads_back_identically_at_any_concurrency() {
    let schema = table_schema();
    let input = batches(&schema, 200);
    let expected: Vec<(i64, i32, i32)> = input
        .iter()
        .flat_map(|b| {
            let id = b
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .clone();
            let val = b
                .column(1)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .clone();
            (0..b.num_rows()).map(move |i| (id.value(i), val.value(i)))
        })
        .enumerate()
        .map(|(rowid, (id, val))| (rowid as i64, id, val))
        .collect();

    let mut file_counts = Vec::new();
    for (concurrency, how) in [(1, Feed::Sync), (8, Feed::Async), (8, Feed::Sync)] {
        let env = Env::new(None).await;
        let mut session = env
            .table_writer(concurrency)
            .begin_write("main", "t", schema.as_ref(), WriteMode::Append)
            .unwrap();
        for batch in &input {
            feed(&mut session, batch, how).await.unwrap();
        }
        let result = session.finish().await.unwrap();
        file_counts.push(result.files_written);
        assert_eq!(
            read_back(&env.conn_str).await,
            expected,
            "concurrency {concurrency}, {how:?}: read-back must match the input in write order"
        );
    }
    assert!(
        file_counts[0] > 20,
        "the write must span many files: {file_counts:?}"
    );
    assert!(
        file_counts.windows(2).all(|w| w[0] == w[1]),
        "{file_counts:?}"
    );
}

// ---------------------------------------------------------------------------
// Upload order, cancellation, and the stores that drive them
// ---------------------------------------------------------------------------

/// How a [`ControlledStore`] breaks a multipart upload.
#[derive(Debug, Clone, Copy)]
enum MultipartFault {
    /// Every part fails.
    Part,
    /// The upload completes, creating the object, and then reports failure: an
    /// unacknowledged `CompleteMultipartUpload`.
    Complete,
}

fn injected(what: &'static str) -> object_store::Error {
    object_store::Error::Generic {
        store: "ControlledStore",
        source: what.into(),
    }
}

/// A store over the local filesystem that can hold writes back, to make upload
/// completion order and in-flight uploads deterministic, and can break multipart
/// uploads.
#[derive(Debug)]
struct ControlledStore {
    inner: Arc<dyn ObjectStore>,
    /// Hold the first write until this many other writes have finished.
    hold_first_until: Option<usize>,
    /// Hold every write until [`Self::release`].
    gate: Option<tokio::sync::Semaphore>,
    multipart_fault: Option<MultipartFault>,
    attempts: AtomicUsize,
    entered: tokio::sync::watch::Sender<usize>,
    finished: tokio::sync::watch::Sender<usize>,
}

impl ControlledStore {
    fn new(inner: Arc<dyn ObjectStore>) -> Self {
        Self {
            inner,
            hold_first_until: None,
            gate: None,
            multipart_fault: None,
            attempts: AtomicUsize::new(0),
            entered: tokio::sync::watch::channel(0).0,
            finished: tokio::sync::watch::channel(0).0,
        }
    }

    /// Let every held write proceed.
    fn release(&self) {
        if let Some(gate) = &self.gate {
            gate.add_permits(1 << 20);
        }
    }

    /// Wait until `n` writes have started.
    async fn wait_entered(&self, n: usize) {
        let mut entered = self.entered.subscribe();
        tokio::time::timeout(Duration::from_secs(60), entered.wait_for(|c| *c >= n))
            .await
            .expect("writes never started")
            .unwrap();
    }

    /// Wait until every write that started has finished.
    async fn wait_settled(&self) {
        let deadline = Instant::now() + Duration::from_secs(60);
        while *self.finished.borrow() < *self.entered.borrow() {
            assert!(Instant::now() < deadline, "held writes never finished");
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    async fn hold(&self) {
        let attempt = self.attempts.fetch_add(1, Ordering::SeqCst);
        self.entered.send_modify(|c| *c += 1);
        if let Some(gate) = &self.gate {
            gate.acquire().await.unwrap().forget();
        }
        if attempt == 0
            && let Some(n) = self.hold_first_until
        {
            let mut finished = self.finished.subscribe();
            finished.wait_for(|c| *c >= n).await.unwrap();
        }
    }
}

impl std::fmt::Display for ControlledStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ControlledStore({})", self.inner)
    }
}

#[derive(Debug)]
struct FaultyUpload {
    inner: Box<dyn object_store::MultipartUpload>,
    fault: MultipartFault,
}

#[async_trait::async_trait]
impl object_store::MultipartUpload for FaultyUpload {
    fn put_part(&mut self, data: object_store::PutPayload) -> object_store::UploadPart {
        match self.fault {
            MultipartFault::Part => Box::pin(async { Err(injected("injected part failure")) }),
            MultipartFault::Complete => self.inner.put_part(data),
        }
    }

    async fn complete(&mut self) -> object_store::Result<object_store::PutResult> {
        self.inner.complete().await?;
        Err(injected("injected complete failure"))
    }

    async fn abort(&mut self) -> object_store::Result<()> {
        self.inner.abort().await
    }
}

#[async_trait::async_trait]
impl ObjectStore for ControlledStore {
    async fn put_opts(
        &self,
        location: &ObjectPath,
        payload: object_store::PutPayload,
        opts: object_store::PutOptions,
    ) -> object_store::Result<object_store::PutResult> {
        self.hold().await;
        let result = self.inner.put_opts(location, payload, opts).await;
        self.finished.send_modify(|c| *c += 1);
        result
    }

    async fn put_multipart_opts(
        &self,
        location: &ObjectPath,
        opts: object_store::PutMultipartOptions,
    ) -> object_store::Result<Box<dyn object_store::MultipartUpload>> {
        let inner = self.inner.put_multipart_opts(location, opts).await?;
        Ok(match self.multipart_fault {
            Some(fault) => Box::new(FaultyUpload {
                inner,
                fault,
            }),
            None => inner,
        })
    }

    async fn get_opts(
        &self,
        location: &ObjectPath,
        options: object_store::GetOptions,
    ) -> object_store::Result<object_store::GetResult> {
        self.inner.get_opts(location, options).await
    }

    fn delete_stream(
        &self,
        locations: futures::stream::BoxStream<'static, object_store::Result<ObjectPath>>,
    ) -> futures::stream::BoxStream<'static, object_store::Result<ObjectPath>> {
        self.inner.delete_stream(locations)
    }

    fn list(
        &self,
        prefix: Option<&ObjectPath>,
    ) -> futures::stream::BoxStream<'static, object_store::Result<object_store::ObjectMeta>> {
        self.inner.list(prefix)
    }

    async fn list_with_delimiter(
        &self,
        prefix: Option<&ObjectPath>,
    ) -> object_store::Result<object_store::ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }

    async fn copy_opts(
        &self,
        from: &ObjectPath,
        to: &ObjectPath,
        options: object_store::CopyOptions,
    ) -> object_store::Result<()> {
        self.inner.copy_opts(from, to, options).await
    }
}

impl Env {
    /// A rolling table writer over `store` instead of the recording one.
    fn controlled_writer(
        &self,
        store: &Arc<ControlledStore>,
        concurrency: usize,
    ) -> DuckLakeTableWriter {
        let store: Arc<dyn ObjectStore> = store.clone();
        DuckLakeTableWriter::new(self.writer.clone(), store)
            .unwrap()
            .with_target_file_size(4 * 1024)
            .with_options(&rolling_write_options())
            .with_upload_concurrency(concurrency)
    }

    /// Every file under the data path, by file name, whatever its extension.
    fn files_on_disk(&self) -> Vec<String> {
        fn walk(dir: &std::path::Path, out: &mut Vec<String>) {
            for entry in std::fs::read_dir(dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    walk(&path, out);
                } else {
                    out.push(path.file_name().unwrap().to_string_lossy().into_owned());
                }
            }
        }
        let mut out = Vec::new();
        walk(&self.temp.path().join("data"), &mut out);
        out.sort();
        out
    }
}

/// The `(rowid, id, val)` rows `batches` should read back as, in write order.
fn expected_rows(input: &[RecordBatch]) -> Vec<(i64, i32, i32)> {
    input
        .iter()
        .flat_map(|b| {
            let id = b
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .clone();
            let val = b
                .column(1)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .clone();
            (0..b.num_rows()).map(move |i| (id.value(i), val.value(i)))
        })
        .enumerate()
        .map(|(rowid, (id, val))| (rowid as i64, id, val))
        .collect()
}

/// Uploads that finish in the reverse of their write order still commit in write
/// order: the first write is held until three later ones have landed, so a commit
/// that took completion order would hand the first file's rows later row ids.
#[tokio::test(flavor = "multi_thread")]
async fn uploads_finishing_out_of_order_commit_in_write_order() {
    let schema = table_schema();
    let input = batches(&schema, 60);
    for how in [Feed::Sync, Feed::Async] {
        let env = Env::new(None).await;
        let mut store = ControlledStore::new(Arc::new(LocalFileSystem::new()));
        store.hold_first_until = Some(3);
        let store = Arc::new(store);
        let mut session = env
            .controlled_writer(&store, 4)
            .begin_write("main", "t", schema.as_ref(), WriteMode::Append)
            .unwrap();
        for batch in &input {
            feed(&mut session, batch, how).await.unwrap();
        }
        let result = session.finish().await.unwrap();
        assert!(
            result.files_written > 4,
            "{how:?}: got {}",
            result.files_written
        );
        assert_eq!(
            read_back(&env.conn_str).await,
            expected_rows(&input),
            "{how:?}: rows and row ids must follow write order"
        );
    }
}

/// Write until the store holds at least one upload, with one upload in flight at
/// a time.
async fn session_with_a_held_upload(env: &Env, store: &Arc<ControlledStore>) -> TableWriteSession {
    let schema = table_schema();
    let mut session = env
        .controlled_writer(store, 1)
        .begin_write("main", "t", schema.as_ref(), WriteMode::Append)
        .unwrap();
    for batch in &batches(&schema, 60) {
        session.write_batch(batch).unwrap();
    }
    store.wait_entered(1).await;
    session
}

fn gated_store() -> Arc<ControlledStore> {
    let mut store = ControlledStore::new(Arc::new(LocalFileSystem::new()));
    store.gate = Some(tokio::sync::Semaphore::new(0));
    Arc::new(store)
}

/// `abort` waits for an upload in flight before removing what was uploaded, so
/// an upload that lands late is removed too.
#[tokio::test(flavor = "multi_thread")]
async fn abort_removes_an_upload_that_lands_late() {
    let env = Env::new(None).await;
    let store = gated_store();
    let session = session_with_a_held_upload(&env, &store).await;
    let abort = tokio::spawn(session.abort());
    // The abort cannot finish while the upload it waits for is held.
    let deadline = Instant::now() + Duration::from_millis(200);
    while Instant::now() < deadline {
        assert!(
            !abort.is_finished(),
            "abort returned with an upload in flight"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    store.release();
    abort.await.unwrap().unwrap();
    store.wait_settled().await;
    assert_eq!(env.files_on_disk(), Vec::<String>::new());
    assert!(env.committed_files().await.is_empty());
}

/// A dropped session waits, in the background, for an upload in flight before
/// removing what was uploaded.
#[tokio::test(flavor = "multi_thread")]
async fn a_dropped_session_removes_an_upload_that_lands_late() {
    let env = Env::new(None).await;
    let store = gated_store();
    let session = session_with_a_held_upload(&env, &store).await;
    drop(session);
    store.release();
    store.wait_settled().await;
    env.wait_for_parquet_count("the late upload must be removed", |n| n == 0)
        .await;
    assert!(env.committed_files().await.is_empty());
}

/// An `abort` cancelled while it waits for an upload in flight leaves the session
/// to its drop, which still waits for that upload before removing it.
#[tokio::test(flavor = "multi_thread")]
async fn a_cancelled_abort_still_removes_an_upload_that_lands_late() {
    let env = Env::new(None).await;
    let store = gated_store();
    let session = session_with_a_held_upload(&env, &store).await;
    // `timeout` polls the abort once, so it is parked on the held upload when the
    // timeout drops it.
    let cancelled = tokio::time::timeout(Duration::from_millis(50), session.abort()).await;
    assert!(cancelled.is_err(), "the abort must still be waiting");
    store.release();
    store.wait_settled().await;
    env.wait_for_parquet_count("the late upload must be removed", |n| n == 0)
        .await;
    assert!(env.committed_files().await.is_empty());
}

/// A multipart upload (a file over the 10 MiB buffer) that fails at a part, or
/// whose completion is not acknowledged after the object was created, leaves no
/// object, from a rolling session and from a single-file one.
#[tokio::test(flavor = "multi_thread")]
async fn a_failed_multipart_upload_leaves_no_object() {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "payload",
        DataType::Binary,
        false,
    )]));
    // Incompressible, so the file outgrows the multipart threshold.
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let input: Vec<RecordBatch> = (0..12)
        .map(|_| {
            let rows: Vec<Vec<u8>> = (0..1024)
                .map(|_| {
                    (0..1024)
                        .map(|_| {
                            state ^= state << 13;
                            state ^= state >> 7;
                            state ^= state << 17;
                            state as u8
                        })
                        .collect()
                })
                .collect();
            RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(arrow::array::BinaryArray::from_iter_values(rows))],
            )
            .unwrap()
        })
        .collect();
    for fault in [MultipartFault::Part, MultipartFault::Complete] {
        for single_file in [false, true] {
            let env = Env::new(None).await;
            let mut store = ControlledStore::new(Arc::new(LocalFileSystem::new()));
            store.multipart_fault = Some(fault);
            let store: Arc<ControlledStore> = Arc::new(store);
            let dyn_store: Arc<dyn ObjectStore> = store.clone();
            let writer = DuckLakeTableWriter::new(env.writer.clone(), dyn_store)
                .unwrap()
                .with_target_file_size(64 * 1024 * 1024);
            let mut session = if single_file {
                writer.begin_write_single_file("main", "blobs", schema.as_ref(), WriteMode::Append)
            } else {
                writer.begin_write("main", "blobs", schema.as_ref(), WriteMode::Append)
            }
            .unwrap();
            for batch in &input {
                session.write_batch_async(batch).await.unwrap();
            }
            let err = session
                .finish()
                .await
                .expect_err("the multipart failure must fail the write");
            // A failed completion is followed by an abort of the same upload, whose
            // error can be the one returned, so only a part failure names its cause.
            if matches!(fault, MultipartFault::Part) {
                assert!(
                    err.to_string().contains("injected"),
                    "single file {single_file}: got {err}"
                );
            }
            assert_eq!(
                env.files_on_disk(),
                Vec::<String>::new(),
                "{fault:?}, single file {single_file}: nothing may be left"
            );
            assert!(env.committed_files().await.is_empty());
        }
    }
}

// ---------------------------------------------------------------------------
// Commit failures: what is removed, and what is kept
// ---------------------------------------------------------------------------

/// How a [`CommitFaultWriter`] fails the commit of a rolling write.
#[derive(Debug, Clone, Copy)]
enum CommitFault {
    /// Fail before anything is committed, with an error that is not a conflict.
    BeforeCommit,
    /// Commit, then report the `COMMIT` as failed with its outcome unknown — a
    /// lost acknowledgement.
    #[cfg(any(feature = "metadata-postgres", feature = "metadata-mysql"))]
    OutcomeUnknown,
}

/// A SQLite metadata writer whose multi-file commit fails as `fault` says.
#[derive(Debug)]
struct CommitFaultWriter {
    inner: SqliteMetadataWriter,
    fault: CommitFault,
}

impl MetadataWriter for CommitFaultWriter {
    fn create_snapshot(&self) -> datafusion_ducklake::Result<i64> {
        self.inner.create_snapshot()
    }

    fn get_or_create_schema(
        &self,
        name: &str,
        path: Option<&str>,
        snapshot_id: i64,
    ) -> datafusion_ducklake::Result<(i64, bool)> {
        self.inner.get_or_create_schema(name, path, snapshot_id)
    }

    fn get_or_create_table(
        &self,
        schema_id: i64,
        name: &str,
        path: Option<&str>,
        snapshot_id: i64,
    ) -> datafusion_ducklake::Result<(i64, bool)> {
        self.inner
            .get_or_create_table(schema_id, name, path, snapshot_id)
    }

    fn set_columns(
        &self,
        table_id: i64,
        columns: &[ColumnDef],
        snapshot_id: i64,
    ) -> datafusion_ducklake::Result<Vec<i64>> {
        self.inner.set_columns(table_id, columns, snapshot_id)
    }

    #[allow(clippy::too_many_arguments)]
    fn register_data_file(
        &self,
        table_id: i64,
        schema_name: &str,
        table_name: &str,
        snapshot_id: i64,
        file: &DataFileInfo,
        mode: WriteMode,
        base_snapshot: i64,
        columns: &[ColumnDef],
        column_ids: &[i64],
    ) -> datafusion_ducklake::Result<CommitIds> {
        self.inner.register_data_file(
            table_id,
            schema_name,
            table_name,
            snapshot_id,
            file,
            mode,
            base_snapshot,
            columns,
            column_ids,
        )
    }

    #[allow(clippy::too_many_arguments)]
    #[cfg_attr(
        not(any(feature = "metadata-postgres", feature = "metadata-mysql")),
        allow(unused_variables)
    )]
    fn register_data_files_with_commit_metadata(
        &self,
        table_id: i64,
        schema_name: &str,
        table_name: &str,
        snapshot_id: i64,
        files: &[DataFileInfo],
        mode: WriteMode,
        base_snapshot: i64,
        columns: &[ColumnDef],
        column_ids: &[i64],
        commit_metadata: &SnapshotCommitMetadata,
        expected_base_snapshot_id: Option<i64>,
    ) -> datafusion_ducklake::Result<CommitIds> {
        match self.fault {
            CommitFault::BeforeCommit => Err(DuckLakeError::Internal(
                "injected failure before commit".to_string(),
            )),
            #[cfg(any(feature = "metadata-postgres", feature = "metadata-mysql"))]
            CommitFault::OutcomeUnknown => {
                self.inner.register_data_files_with_commit_metadata(
                    table_id,
                    schema_name,
                    table_name,
                    snapshot_id,
                    files,
                    mode,
                    base_snapshot,
                    columns,
                    column_ids,
                    commit_metadata,
                    expected_base_snapshot_id,
                )?;
                Err(DuckLakeError::CommitOutcomeUnknown(sqlx::Error::Io(
                    std::io::Error::new(std::io::ErrorKind::ConnectionReset, "lost ack"),
                )))
            },
        }
    }

    fn end_table_files(&self, table_id: i64, snapshot_id: i64) -> datafusion_ducklake::Result<u64> {
        self.inner.end_table_files(table_id, snapshot_id)
    }

    fn get_data_path(&self) -> datafusion_ducklake::Result<String> {
        self.inner.get_data_path()
    }

    fn get_table_column_nullability(
        &self,
        schema_name: &str,
        table_name: &str,
    ) -> datafusion_ducklake::Result<Option<Vec<(String, bool)>>> {
        self.inner
            .get_table_column_nullability(schema_name, table_name)
    }

    fn set_data_path(&self, path: &str) -> datafusion_ducklake::Result<()> {
        self.inner.set_data_path(path)
    }

    fn initialize_schema(&self) -> datafusion_ducklake::Result<()> {
        self.inner.initialize_schema()
    }

    fn begin_write_transaction(
        &self,
        schema_name: &str,
        table_name: &str,
        columns: &[ColumnDef],
        mode: WriteMode,
    ) -> datafusion_ducklake::Result<WriteSetupResult> {
        self.inner
            .begin_write_transaction(schema_name, table_name, columns, mode)
    }
}

/// Run a rolling write of many files through a commit that fails as `fault` says,
/// returning the error.
async fn rolling_write_failing_with(env: &Env, fault: CommitFault) -> DuckLakeError {
    let metadata: Arc<dyn MetadataWriter> = Arc::new(CommitFaultWriter {
        inner: SqliteMetadataWriter::new(&env.conn_str).await.unwrap(),
        fault,
    });
    let store: Arc<dyn ObjectStore> = env.store.clone();
    let schema = table_schema();
    let mut session = DuckLakeTableWriter::new(metadata, store)
        .unwrap()
        .with_target_file_size(4 * 1024)
        .with_options(&rolling_write_options())
        .with_upload_concurrency(3)
        .begin_write("main", "t", schema.as_ref(), WriteMode::Append)
        .unwrap();
    for batch in &batches(&schema, 60) {
        session.write_batch_async(batch).await.unwrap();
    }
    let err = session.finish().await.expect_err("the commit must fail");
    assert!(
        env.store.writes().len() > 5,
        "the write must have uploaded several files"
    );
    err
}

/// A commit that fails before `COMMIT` registered nothing, whatever the error, so
/// its files are removed — not only on a conflict.
#[tokio::test(flavor = "multi_thread")]
async fn a_commit_failing_before_commit_removes_its_files() {
    let env = Env::new(None).await;
    let err = rolling_write_failing_with(&env, CommitFault::BeforeCommit).await;
    assert!(matches!(err, DuckLakeError::Internal(_)), "got: {err}");
    assert_eq!(env.parquet_on_disk(), Vec::<String>::new());
    assert!(env.committed_files().await.is_empty());
}

/// A `COMMIT` that failed with its outcome unknown may have applied — here it did —
/// so every uploaded file is kept, and the committed snapshot reads back whole.
#[cfg(any(feature = "metadata-postgres", feature = "metadata-mysql"))]
#[tokio::test(flavor = "multi_thread")]
async fn a_commit_with_an_unknown_outcome_keeps_every_file() {
    let env = Env::new(None).await;
    let err = rolling_write_failing_with(&env, CommitFault::OutcomeUnknown).await;
    assert!(
        matches!(err, DuckLakeError::CommitOutcomeUnknown(_)),
        "got: {err}"
    );
    let committed = env.committed_files().await;
    assert_eq!(committed.len(), env.store.writes().len());
    assert_eq!(env.parquet_on_disk(), committed, "every file must be kept");
    let schema = table_schema();
    assert_eq!(
        read_back(&env.conn_str).await,
        expected_rows(&batches(&schema, 60))
    );
}

/// A single-file session whose commit a stale base snapshot rejects removes the
/// file it uploaded.
#[tokio::test(flavor = "multi_thread")]
async fn a_rejected_single_file_commit_removes_its_file() {
    let env = Env::new(None).await;
    let schema = table_schema();
    let table_writer = env.table_writer(3);
    let initial = table_writer
        .write_table("main", "t", &batches(&schema, 2))
        .await
        .unwrap();
    table_writer
        .append_table("main", "t", &batches(&schema, 2))
        .await
        .unwrap();
    let committed_before = env.committed_files().await;

    let mut session = table_writer
        .begin_write_single_file("main", "t", schema.as_ref(), WriteMode::Replace)
        .unwrap()
        .with_options(
            &TableWriteOptions::new().with_expected_base_snapshot_id(initial.snapshot_id),
        );
    for batch in &batches(&schema, 10) {
        session.write_batch(batch).unwrap();
    }
    let err = session
        .finish()
        .await
        .expect_err("a stale base must conflict");
    assert!(matches!(err, DuckLakeError::Conflict(_)), "got: {err}");
    assert_eq!(env.store.writes().len(), committed_before.len() + 1);
    assert_eq!(env.committed_files().await, committed_before);
    assert_eq!(env.parquet_on_disk(), committed_before);
}

/// A `finish_with_deletes` whose commit is rejected — the delete file's
/// compare-and-swap guard names a delete file that is not live — removes the
/// appended files and the delete file alike.
#[tokio::test(flavor = "multi_thread")]
async fn a_rejected_finish_with_deletes_removes_data_and_delete_files() {
    let env = Env::new(None).await;
    let schema = table_schema();
    let table_writer = env.table_writer(3);
    table_writer
        .write_table("main", "t", &batches(&schema, 1))
        .await
        .unwrap();
    let committed_before = env.committed_files().await;
    assert_eq!(committed_before.len(), 1);
    let pool = SqlitePool::connect(&env.conn_str).await.unwrap();
    let (data_file_id, data_path): (i64, String) = sqlx::query_as(
        "SELECT data_file_id, path FROM ducklake_data_file WHERE end_snapshot IS NULL",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    let delete = table_writer
        .write_delete_file("main", "t", &data_path, &[0, 1])
        .await
        .unwrap();
    assert_eq!(env.parquet_on_disk().len(), 2, "the delete file must exist");

    let mut session = table_writer
        .begin_write("main", "t", schema.as_ref(), WriteMode::Append)
        .unwrap();
    for batch in &batches(&schema, 60) {
        session.write_batch_async(batch).await.unwrap();
    }
    let err = session
        .finish_with_deletes(&[DeleteFileEntry {
            data_file_id,
            expected_prev_delete_file: Some(i64::MAX),
            delete,
        }])
        .await
        .expect_err("a stale delete guard must conflict");
    assert!(matches!(err, DuckLakeError::Conflict(_)), "got: {err}");
    assert!(
        env.store.writes().len() > committed_before.len() + 5,
        "the rejected write must have uploaded several files"
    );
    assert_eq!(env.committed_files().await, committed_before);
    assert_eq!(
        env.parquet_on_disk(),
        committed_before,
        "the appended files and the delete file must be removed, and only those"
    );
}

/// A partitioned write reads back through the provider with a filter on the
/// partition key: the listing prunes files by their recorded partition values, so
/// a file recorded under another partition's values would lose or add rows.
#[tokio::test(flavor = "multi_thread")]
async fn a_partitioned_write_reads_back_under_a_partition_filter() {
    let env = Env::new(None).await;
    create_partitioned_table(&env.writer);
    let input = events_batches(30);
    let mut session = env
        .table_writer(3)
        .with_max_open_partitions(2)
        .begin_write(
            "main",
            "events",
            input[0].schema().as_ref(),
            WriteMode::Append,
        )
        .unwrap();
    for batch in &input {
        session.write_batch_async(batch).await.unwrap();
    }
    let result = session.finish().await.unwrap();
    assert!(result.files_written > 3, "got {}", result.files_written);

    let provider = SqliteMetadataProvider::new(&env.conn_str).await.unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog(
        "ducklake",
        Arc::new(DuckLakeCatalog::new(provider).unwrap()),
    );
    for region in ["us", "eu", "ap"] {
        let batches = ctx
            .sql(&format!(
                "SELECT id, region FROM ducklake.main.events WHERE region = '{region}' ORDER BY id"
            ))
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let mut ids = Vec::new();
        for batch in &batches {
            let id = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            let got = arrow::compute::cast(batch.column(1), &DataType::Utf8).unwrap();
            let got = got
                .as_any()
                .downcast_ref::<arrow::array::StringArray>()
                .unwrap();
            for i in 0..batch.num_rows() {
                assert_eq!(got.value(i), region);
                ids.push(id.value(i));
            }
        }
        let expected: Vec<i32> = (0..30 * 200)
            .filter(|id| ["us", "eu", "ap"][(*id as usize) % 3] == region)
            .collect();
        assert_eq!(ids, expected, "region {region}");
    }
}

// ---------------------------------------------------------------------------
// A write split into lanes
// ---------------------------------------------------------------------------

/// The expected `(rowid, id, val)` of `input`, in write order.
fn rows_in_order(input: &[RecordBatch]) -> Vec<(i64, i32, i32)> {
    input
        .iter()
        .flat_map(|b| {
            let id = b
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .clone();
            let val = b
                .column(1)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .clone();
            (0..b.num_rows()).map(move |i| (id.value(i), val.value(i)))
        })
        .enumerate()
        .map(|(rowid, (id, val))| (rowid as i64, id, val))
        .collect()
}

/// Split `session` into `parts.len()` lanes, write lane `i` the batches of
/// `parts[i]` on a blocking thread of its own, all at once, and join the
/// lanes back in order.
async fn write_in_lanes(session: &mut TableWriteSession, parts: Vec<Vec<RecordBatch>>) {
    let lanes = session.lanes(parts.len()).unwrap();
    assert_eq!(lanes.len(), parts.len(), "one lane per part");
    let tasks: Vec<_> = lanes
        .into_iter()
        .zip(parts)
        .map(|(mut lane, part)| {
            tokio::task::spawn_blocking(move || {
                for batch in &part {
                    lane.write_batch(batch).unwrap();
                }
                lane
            })
        })
        .collect();
    let mut lanes = Vec::new();
    for task in tasks {
        lanes.push(task.await.unwrap());
    }
    session.join_lanes(lanes).unwrap();
}

/// `input` cut into `n` consecutive parts.
fn consecutive_parts(input: &[RecordBatch], n: usize) -> Vec<Vec<RecordBatch>> {
    let per = input.len().div_ceil(n);
    input.chunks(per).map(<[RecordBatch]>::to_vec).collect()
}

#[test]
fn a_lane_can_move_to_another_thread() {
    fn is_send<T: Send>() {}
    is_send::<datafusion_ducklake::WriteLane>();
}

/// Lanes written at once commit in one snapshot, in lane order: the rows read
/// back with the row ids one writer would have given them, every file
/// uploaded once and registered once.
#[tokio::test(flavor = "multi_thread")]
async fn a_write_split_into_lanes_reads_back_in_lane_order() {
    let schema = table_schema();
    let input = batches(&schema, 200);
    let expected = rows_in_order(&input);
    for lanes in [1, 3, 8] {
        let env = Env::new(None).await;
        env.set_per_thread_output("true");
        let mut session = env
            .table_writer(4)
            .begin_write("main", "t", schema.as_ref(), WriteMode::Append)
            .unwrap();
        write_in_lanes(&mut session, consecutive_parts(&input, lanes)).await;
        let result = session.finish().await.unwrap();
        assert!(
            result.files_written >= lanes,
            "{lanes} lanes: {} files",
            result.files_written
        );
        assert_eq!(result.records_written as usize, expected.len());
        assert_eq!(read_back(&env.conn_str).await, expected, "{lanes} lanes");
        let writes = env.store.writes();
        assert!(
            writes.values().all(|n| *n == 1),
            "{lanes} lanes: {writes:?}"
        );
        assert_eq!(
            env.parquet_on_disk(),
            env.committed_files().await,
            "{lanes} lanes"
        );
    }
}

/// Rows the session wrote before it split come first.
#[tokio::test(flavor = "multi_thread")]
async fn rows_written_before_the_split_come_first() {
    let schema = table_schema();
    let input = batches(&schema, 60);
    let env = Env::new(None).await;
    env.set_per_thread_output("true");
    let mut session = env
        .table_writer(4)
        .begin_write("main", "t", schema.as_ref(), WriteMode::Append)
        .unwrap();
    for batch in &input[..10] {
        session.write_batch(batch).unwrap();
    }
    write_in_lanes(&mut session, consecutive_parts(&input[10..], 3)).await;
    assert!(
        session.write_batch(&input[0]).is_err(),
        "a split session must refuse rows of its own"
    );
    session.finish().await.unwrap();
    assert_eq!(read_back(&env.conn_str).await, rows_in_order(&input));
}

/// `events` read back through DataFusion as `(id, region)`, ordered by id.
async fn read_back_events(conn_str: &str) -> Vec<(i32, String)> {
    let provider = SqliteMetadataProvider::new(conn_str).await.unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog(
        "ducklake",
        Arc::new(DuckLakeCatalog::new(provider).unwrap()),
    );
    let batches = ctx
        .sql("SELECT id, region FROM ducklake.main.events ORDER BY id")
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
            .downcast_ref::<Int32Array>()
            .unwrap();
        let regions = arrow::compute::cast(batch.column(1), &DataType::Utf8).unwrap();
        let regions = regions
            .as_any()
            .downcast_ref::<arrow::array::StringArray>()
            .unwrap();
        for i in 0..batch.num_rows() {
            rows.push((ids.value(i), regions.value(i).to_string()));
        }
    }
    rows
}

/// The `(id, region)` rows of [`events_batches`]`(count)`, ordered by id.
fn events_rows(count: i32) -> Vec<(i32, String)> {
    (0..count * 200)
        .map(|id| (id, ["us", "eu", "ap"][(id as usize) % 3].to_string()))
        .collect()
}

/// The live files' record counts of `events`, summed by partition value.
async fn rows_per_region(conn_str: &str) -> Vec<(String, i64)> {
    let pool = SqlitePool::connect(conn_str).await.unwrap();
    sqlx::query_as(
        "SELECT v.partition_value, SUM(f.record_count) FROM ducklake_data_file f
         JOIN ducklake_file_partition_value v ON v.data_file_id = f.data_file_id
         WHERE f.end_snapshot IS NULL GROUP BY 1 ORDER BY 1",
    )
    .fetch_all(&pool)
    .await
    .unwrap()
}

/// A partitioned table is never split, as official never writes a
/// partitioned insert per thread: with `per_thread_output` on, a write that
/// asks for several lanes gets one, and every row lands once, in its
/// partition.
#[tokio::test(flavor = "multi_thread")]
async fn a_partitioned_write_is_not_split_even_with_per_thread_output_on() {
    let env = Env::new(None).await;
    env.set_per_thread_output("true");
    create_partitioned_table(&env.writer);
    let input = events_batches(30);
    let mut session = env
        .table_writer(3)
        .with_max_open_partitions(2)
        .begin_write(
            "main",
            "events",
            input[0].schema().as_ref(),
            WriteMode::Append,
        )
        .unwrap();
    let mut lanes = session.lanes(4).unwrap();
    assert_eq!(lanes.len(), 1, "a partitioned write splits into one lane");
    for batch in &input {
        lanes[0].write_batch(batch).unwrap();
    }
    session.join_lanes(lanes).unwrap();
    let result = session.finish().await.unwrap();
    assert_eq!(result.records_written, 30 * 200);
    assert_eq!(env.parquet_on_disk(), env.committed_files().await);
    assert_eq!(
        rows_per_region(&env.conn_str).await,
        vec![("ap".to_string(), 2_000), ("eu".to_string(), 2_000), ("us".to_string(), 2_000)]
    );
    assert_eq!(read_back_events(&env.conn_str).await, events_rows(30));
}

/// Lanes of an unpartitioned write whose row groups are encoded on a pool
/// they share: the rows read back in lane order, with the row ids one writer
/// gives them, and every file is uploaded once and registered once.
#[tokio::test(flavor = "multi_thread")]
async fn lanes_sharing_an_encode_pool_read_back_in_lane_order() {
    let schema = table_schema();
    let input = batches(&schema, 200);
    let expected = rows_in_order(&input);
    let env = Env::new(None).await;
    env.set_per_thread_output("true");
    let mut session = env
        .table_writer(4)
        .with_encode_parallelism(3)
        .begin_write("main", "t", schema.as_ref(), WriteMode::Append)
        .unwrap();
    write_in_lanes(&mut session, consecutive_parts(&input, 4)).await;
    let result = session.finish().await.unwrap();
    assert!(result.files_written >= 4, "{} files", result.files_written);
    assert_eq!(result.records_written as usize, expected.len());
    assert_eq!(read_back(&env.conn_str).await, expected);
    let writes = env.store.writes();
    assert!(writes.values().all(|n| *n == 1), "{writes:?}");
    assert_eq!(env.parquet_on_disk(), env.committed_files().await);
}

/// Writes at once on one shared encode pool whose budget holds about two of
/// their row groups, so many row groups are encoded on the threads that
/// write instead: four lanes of an unpartitioned write, and a partitioned
/// write with one partition file open at a time. Every row of each reads back
/// once, in order, with its values.
#[tokio::test(flavor = "multi_thread")]
async fn writes_sharing_a_small_encode_pool_read_back_as_written() {
    let env = Env::new(None).await;
    env.set_per_thread_output("true");
    create_partitioned_table(&env.writer);
    // A 64-row group of `t` holds 512 bytes, charged twice that.
    let pool = EncodePool::try_new(2, 2 * 1024).unwrap();

    let events = events_batches(30);
    let mut options = rolling_write_options();
    options.max_row_group_rows = Some(16);
    let store: Arc<dyn ObjectStore> = env.store.clone();
    let mut partitioned = DuckLakeTableWriter::new(env.writer.clone(), store)
        .unwrap()
        .with_options(&options.with_encode_pool(Arc::clone(&pool)))
        .with_target_file_size(4 * 1024)
        .with_max_open_partitions(1)
        .begin_write(
            "main",
            "events",
            events[0].schema().as_ref(),
            WriteMode::Append,
        )
        .unwrap();
    let partitioned = tokio::task::spawn_blocking(move || {
        for batch in &events {
            partitioned.write_batch(batch).unwrap();
        }
        partitioned
    });

    let schema = table_schema();
    let input = batches(&schema, 200);
    let mut session = env
        .table_writer(4)
        .with_encode_pool(Arc::clone(&pool))
        .begin_write("main", "t", schema.as_ref(), WriteMode::Append)
        .unwrap();
    write_in_lanes(&mut session, consecutive_parts(&input, 4)).await;

    let partitioned = partitioned.await.unwrap();
    let result = session.finish().await.unwrap();
    assert_eq!(result.records_written as usize, rows_in_order(&input).len());
    assert_eq!(
        partitioned.finish().await.unwrap().records_written,
        30 * 200
    );
    assert_eq!(read_back(&env.conn_str).await, rows_in_order(&input));
    assert_eq!(read_back_events(&env.conn_str).await, events_rows(30));
    assert_eq!(env.parquet_on_disk(), env.committed_files().await);
}

/// A partitioned write whose row groups are encoded on a pool, with one
/// partition file open at a time, so every partition switch evicts a file
/// whose row groups may still be encoding: every row lands once, in its
/// partition, with its values.
#[tokio::test(flavor = "multi_thread")]
async fn a_partitioned_write_encoded_on_a_pool_lands_every_row_once_across_evictions() {
    let env = Env::new(None).await;
    create_partitioned_table(&env.writer);
    let input = events_batches(30);
    let mut options = rolling_write_options();
    options.max_row_group_rows = Some(16);
    let store: Arc<dyn ObjectStore> = env.store.clone();
    let mut session = DuckLakeTableWriter::new(env.writer.clone(), store)
        .unwrap()
        .with_options(&options)
        .with_target_file_size(4 * 1024)
        .with_max_open_partitions(1)
        .with_encode_parallelism(4)
        .begin_write(
            "main",
            "events",
            input[0].schema().as_ref(),
            WriteMode::Append,
        )
        .unwrap();
    for batch in &input {
        session.write_batch(batch).unwrap();
    }
    let result = session.finish().await.unwrap();
    assert_eq!(result.records_written, 30 * 200);
    // Each batch holds every region, so one open file evicts at least twice
    // a batch.
    assert!(result.files_written >= 60, "{} files", result.files_written);
    let writes = env.store.writes();
    assert!(writes.values().all(|n| *n == 1), "{writes:?}");
    assert_eq!(env.parquet_on_disk(), env.committed_files().await);
    assert_eq!(
        rows_per_region(&env.conn_str).await,
        vec![("ap".to_string(), 2_000), ("eu".to_string(), 2_000), ("us".to_string(), 2_000)]
    );
    assert_eq!(read_back_events(&env.conn_str).await, events_rows(30));
}

/// A split write that does not commit leaves nothing: abort removes every
/// lane's files, and so does dropping the session.
#[tokio::test(flavor = "multi_thread")]
async fn a_split_write_that_does_not_commit_leaves_nothing() {
    let schema = table_schema();
    let input = batches(&schema, 120);

    let env = Env::new(None).await;
    env.set_per_thread_output("true");
    let mut session = env
        .table_writer(4)
        .begin_write("main", "t", schema.as_ref(), WriteMode::Append)
        .unwrap();
    write_in_lanes(&mut session, consecutive_parts(&input, 3)).await;
    env.wait_for_parquet_count("lane files must upload before the abort", |n| n > 0)
        .await;
    session.abort().await.unwrap();
    assert_eq!(env.parquet_on_disk(), Vec::<String>::new());
    assert!(env.committed_files().await.is_empty());

    let env = Env::new(None).await;
    env.set_per_thread_output("true");
    let mut session = env
        .table_writer(4)
        .begin_write("main", "t", schema.as_ref(), WriteMode::Append)
        .unwrap();
    write_in_lanes(&mut session, consecutive_parts(&input, 3)).await;
    env.wait_for_parquet_count("lane files must upload before the drop", |n| n > 0)
        .await;
    drop(session);
    env.wait_for_parquet_count("a dropped split write must remove its files", |n| n == 0)
        .await;
    assert!(env.committed_files().await.is_empty());
}

/// An upload failing in one lane fails the write, commits nothing, and leaves
/// no file of any lane behind.
#[tokio::test(flavor = "multi_thread")]
async fn a_failed_upload_in_a_lane_commits_nothing_and_leaves_nothing() {
    let schema = table_schema();
    let input = batches(&schema, 120);
    let env = Env::new(Some(5)).await;
    env.set_per_thread_output("true");
    let mut session = env
        .table_writer(2)
        .begin_write("main", "t", schema.as_ref(), WriteMode::Append)
        .unwrap();
    let lanes = session.lanes(3).unwrap();
    let tasks: Vec<_> = lanes
        .into_iter()
        .zip(consecutive_parts(&input, 3))
        .map(|(mut lane, part)| {
            tokio::task::spawn_blocking(move || {
                for batch in &part {
                    // The failure surfaces on whichever lane sees it first.
                    if lane.write_batch(batch).is_err() {
                        break;
                    }
                }
                lane
            })
        })
        .collect();
    let mut lanes = Vec::new();
    for task in tasks {
        lanes.push(task.await.unwrap());
    }
    let finished = match session.join_lanes(lanes) {
        Ok(()) => session.finish().await.map(|_| ()),
        Err(e) => {
            session.abort().await.unwrap();
            Err(e)
        },
    };
    assert!(
        finished.is_err(),
        "an upload failed, so the write must fail"
    );
    env.wait_for_parquet_count("a failed split write must remove its files", |n| n == 0)
        .await;
    assert!(env.committed_files().await.is_empty());
}

/// Batches of ascending ids, of `sizes` rows each.
fn sized_batches(schema: &Arc<Schema>, sizes: &[i32]) -> Vec<RecordBatch> {
    let mut next_id = 0i32;
    sizes
        .iter()
        .map(|&count| {
            let ids: Vec<i32> = (next_id..next_id + count).collect();
            next_id += count;
            let vals: Vec<i32> = ids.iter().map(|id| id * 10).collect();
            RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(Int32Array::from(ids)), Arc::new(Int32Array::from(vals))],
            )
            .unwrap()
        })
        .collect()
}

/// Lanes are official's per-thread output: with the table's
/// `per_thread_output` off, the default, a write splits into one lane
/// however many it asks for, and still writes every row; with it on, into as
/// many as asked.
#[tokio::test(flavor = "multi_thread")]
async fn per_thread_output_decides_how_many_lanes_a_write_splits_into() {
    let schema = table_schema();
    let input = batches(&schema, 60);
    for (setting, expected_lanes) in [(None, 1), (Some("false"), 1), (Some("true"), 4)] {
        let env = Env::new(None).await;
        if let Some(value) = setting {
            env.set_per_thread_output(value);
        }
        let mut session = env
            .table_writer(2)
            .begin_write("main", "t", schema.as_ref(), WriteMode::Append)
            .unwrap();
        let mut lanes = session.lanes(4).unwrap();
        assert_eq!(lanes.len(), expected_lanes, "per_thread_output {setting:?}");
        let count = lanes.len();
        for (i, batch) in input.iter().enumerate() {
            // Consecutive runs of batches per lane keep the rows in order.
            let lane = i * count / input.len();
            lanes[lane].write_batch(batch).unwrap();
        }
        session.join_lanes(lanes).unwrap();
        let result = session.finish().await.unwrap();
        assert_eq!(result.records_written as usize, rows_in_order(&input).len());
        assert_eq!(
            read_back(&env.conn_str).await,
            rows_in_order(&input),
            "per_thread_output {setting:?}"
        );
    }
}

/// `per_thread_output` is read table over schema over catalog-wide, as
/// official's `TryGetConfigOption` searches them, and only `true` splits: a
/// table's own value outranks its schema's and the catalog's either way, and
/// a schema's outranks the catalog's.
#[tokio::test(flavor = "multi_thread")]
async fn per_thread_output_is_read_table_over_schema_over_catalog_wide() {
    let schema = table_schema();
    let cases = [
        // (catalog-wide, schema, table, lanes)
        (Some("true"), None, Some("false"), 1),
        (Some("false"), None, Some("true"), 4),
        (Some("true"), Some("false"), None, 1),
        (Some("false"), Some("true"), None, 4),
        (None, Some("true"), Some("false"), 1),
    ];
    for (global, schema_value, table_value, expected_lanes) in cases {
        let env = Env::new(None).await;
        let cols = vec![
            ColumnDef::from_arrow("id", &DataType::Int32, false).unwrap(),
            ColumnDef::from_arrow("val", &DataType::Int32, false).unwrap(),
        ];
        let created = env
            .writer
            .begin_write_transaction("main", "t", &cols, WriteMode::Replace)
            .unwrap();
        env.writer
            .publish_snapshot(
                created.table_id,
                "main",
                "t",
                created.snapshot_id,
                WriteMode::Replace,
                created.base_snapshot_id,
                &cols,
                &created.column_ids,
            )
            .unwrap();
        if let Some(value) = global {
            env.set_per_thread_output(value);
        }
        if let Some(value) = schema_value {
            let pool = SqlitePool::connect(&env.conn_str).await.unwrap();
            let (schema_id,): (i64,) = sqlx::query_as(
                "SELECT schema_id FROM ducklake_schema WHERE schema_name = 'main' \
                 AND end_snapshot IS NULL",
            )
            .fetch_one(&pool)
            .await
            .unwrap();
            env.writer
                .set_schema_setting(schema_id, "per_thread_output", value)
                .unwrap();
        }
        if let Some(value) = table_value {
            env.writer
                .set_table_setting(created.table_id, "per_thread_output", value)
                .unwrap();
        }
        let mut session = env
            .table_writer(2)
            .begin_write("main", "t", schema.as_ref(), WriteMode::Append)
            .unwrap();
        let lanes = session.lanes(4).unwrap();
        assert_eq!(
            lanes.len(),
            expected_lanes,
            "catalog-wide {global:?}, schema {schema_value:?}, table {table_value:?}"
        );
        session.join_lanes(lanes).unwrap();
        session.abort().await.unwrap();
    }
}

/// A split write is inlined as one write is: official counts the rows of all
/// its threads against the limit. Rows the session held and rows every lane
/// held commit inlined, in lane order, and no data file is written.
#[tokio::test(flavor = "multi_thread")]
async fn a_split_write_within_the_inlining_limit_is_inlined() {
    let schema = table_schema();
    let input = sized_batches(&schema, &[3, 2, 4, 1, 5]);
    let env = Env::new(None).await;
    env.set_per_thread_output("true");
    let mut options = DuckLakeWriteOptions::default();
    options.data_inlining_row_limit = Some(1_000);
    let mut session = env
        .table_writer(2)
        .with_options(&options)
        .begin_write("main", "t", schema.as_ref(), WriteMode::Append)
        .unwrap();
    session.write_batch(&input[0]).unwrap();
    let mut lanes = session.lanes(3).unwrap();
    lanes[0].write_batch(&input[1]).unwrap();
    lanes[1].write_batch(&input[2]).unwrap();
    lanes[1].write_batch(&input[3]).unwrap();
    lanes[2].write_batch(&input[4]).unwrap();
    session.join_lanes(lanes).unwrap();
    assert!(session.is_inlining(), "15 rows are within the limit");
    let result = session.finish().await.unwrap();
    assert_eq!(result.files_written, 0, "the write must be inlined");
    assert_eq!(result.records_written, 15);
    assert_eq!(env.parquet_on_disk(), Vec::<String>::new());
    assert!(env.committed_files().await.is_empty());
    assert_eq!(read_back(&env.conn_str).await, rows_in_order(&input));
}

/// A split write that passes the limit only across its lanes goes to
/// Parquet whole: the lane that takes the total past it writes its rows out
/// at once, and the rows the session and the other lanes still hold are
/// written out at finish, each to its own files, in write order.
#[tokio::test(flavor = "multi_thread")]
async fn a_split_write_past_the_inlining_limit_writes_every_held_row() {
    let schema = table_schema();
    let input = sized_batches(&schema, &[30, 40, 40, 10]);
    let env = Env::new(None).await;
    env.set_per_thread_output("true");
    let mut options = DuckLakeWriteOptions::default();
    options.data_inlining_row_limit = Some(100);
    let mut session = env
        .table_writer(2)
        .with_options(&options)
        .begin_write("main", "t", schema.as_ref(), WriteMode::Append)
        .unwrap();
    session.write_batch(&input[0]).unwrap();
    let mut lanes = session.lanes(3).unwrap();
    lanes[0].write_batch(&input[1]).unwrap();
    assert!(session.is_inlining(), "70 rows are within the limit");
    lanes[1].write_batch(&input[2]).unwrap();
    assert!(!session.is_inlining(), "110 rows are past the limit");
    lanes[2].write_batch(&input[3]).unwrap();
    session.join_lanes(lanes).unwrap();
    let result = session.finish().await.unwrap();
    assert!(result.files_written >= 4, "{} files", result.files_written);
    assert_eq!(result.records_written, 120);
    assert_eq!(env.parquet_on_disk(), env.committed_files().await);
    assert_eq!(read_back(&env.conn_str).await, rows_in_order(&input));
}

/// A Replace split into lanes that all stay empty still commits the 0-row
/// marker that retires the table's earlier rows.
#[tokio::test(flavor = "multi_thread")]
async fn a_replace_whose_lanes_are_all_empty_empties_the_table() {
    let schema = table_schema();
    let env = Env::new(None).await;
    env.set_per_thread_output("true");
    let mut session = env
        .table_writer(2)
        .begin_write("main", "t", schema.as_ref(), WriteMode::Append)
        .unwrap();
    for batch in &batches(&schema, 20) {
        session.write_batch(batch).unwrap();
    }
    session.finish().await.unwrap();
    assert!(!read_back(&env.conn_str).await.is_empty());

    let mut session = env
        .table_writer(2)
        .begin_write("main", "t", schema.as_ref(), WriteMode::Replace)
        .unwrap();
    let lanes = session.lanes(3).unwrap();
    assert_eq!(lanes.len(), 3);
    session.join_lanes(lanes).unwrap();
    let result = session.finish().await.unwrap();
    assert_eq!(result.records_written, 0);
    assert_eq!(read_back(&env.conn_str).await, Vec::new());
}

/// Splitting twice, joining to a write that never split, and writing to the
/// session of a split write are misuse, refused as such.
#[tokio::test(flavor = "multi_thread")]
async fn misusing_lanes_is_refused_as_invalid() {
    let schema = table_schema();
    let input = batches(&schema, 1);
    let env = Env::new(None).await;
    env.set_per_thread_output("true");
    let writer = env.table_writer(2);

    let mut unsplit = writer
        .begin_write("main", "t", schema.as_ref(), WriteMode::Append)
        .unwrap();
    let mut split = writer
        .begin_write("main", "t", schema.as_ref(), WriteMode::Append)
        .unwrap();
    let lanes = split.lanes(2).unwrap();
    assert!(matches!(
        split.lanes(2),
        Err(DuckLakeError::InvalidConfig(_))
    ));
    assert!(matches!(
        split.write_batch(&input[0]),
        Err(DuckLakeError::InvalidConfig(_))
    ));
    assert!(matches!(
        unsplit.join_lanes(lanes),
        Err(DuckLakeError::InvalidConfig(_))
    ));
    unsplit.abort().await.unwrap();
    split.abort().await.unwrap();
}

/// A lane goes back only to the session it came from: another session
/// refuses it, and a refused join leaves that session unable to commit.
#[tokio::test(flavor = "multi_thread")]
async fn a_lane_of_another_write_is_refused_and_fails_the_join() {
    let schema = table_schema();
    let input = batches(&schema, 30);
    let env = Env::new(None).await;
    env.set_per_thread_output("true");
    let writer = env.table_writer(2);
    let mut first = writer
        .begin_write("main", "t", schema.as_ref(), WriteMode::Append)
        .unwrap();
    let mut second = writer
        .begin_write("main", "t", schema.as_ref(), WriteMode::Append)
        .unwrap();
    let mut own = first.lanes(1).unwrap();
    let mut foreign = second.lanes(1).unwrap();
    for batch in &input {
        own[0].write_batch(batch).unwrap();
        foreign[0].write_batch(batch).unwrap();
    }
    first.join_lanes(own).unwrap();
    let error = first.join_lanes(foreign).unwrap_err();
    assert!(
        matches!(error, DuckLakeError::InvalidConfig(_)),
        "a foreign lane must be refused as invalid: {error}"
    );
    // Every lane `first` issued was joined, yet the failed join fails it.
    let error = first.finish().await.unwrap_err();
    assert!(
        matches!(error, DuckLakeError::InvalidConfig(_)),
        "a write whose join failed must not commit: {error}"
    );
    drop(second);
    env.wait_for_parquet_count("a refused write must remove its files", |n| n == 0)
        .await;
    assert!(env.committed_files().await.is_empty());
}

/// A write with a lane not joined back cannot commit: finish refuses it, and
/// once the lane is dropped nothing of the write is left.
#[tokio::test(flavor = "multi_thread")]
async fn an_unjoined_lane_fails_the_finish_and_commits_nothing() {
    let schema = table_schema();
    let input = batches(&schema, 60);
    let env = Env::new(None).await;
    env.set_per_thread_output("true");
    let mut session = env
        .table_writer(2)
        .begin_write("main", "t", schema.as_ref(), WriteMode::Append)
        .unwrap();
    let mut lanes = session.lanes(2).unwrap();
    for (i, batch) in input.iter().enumerate() {
        lanes[i % 2].write_batch(batch).unwrap();
    }
    let unjoined = lanes.pop().unwrap();
    session.join_lanes(lanes).unwrap();
    let error = session.finish().await.unwrap_err();
    assert!(
        matches!(error, DuckLakeError::InvalidConfig(_)),
        "a write with an unjoined lane must not commit: {error}"
    );
    drop(unjoined);
    env.wait_for_parquet_count("an uncommitted split write must remove its files", |n| {
        n == 0
    })
    .await;
    assert!(env.committed_files().await.is_empty());
}

/// A schema with a leaf of every kind the encoders and the statistics treat
/// apart: an integer, text short and over 256 bytes, a double with both
/// zeros, NaN and NULL, a map, a list, a struct, and text over 256 bytes
/// whose 256th byte falls inside a character (`note`).
fn nested_schema() -> Arc<Schema> {
    use arrow::datatypes::Fields;
    let entries = Fields::from(vec![
        Field::new("key", DataType::Utf8, false),
        Field::new("value", DataType::Utf8, true),
    ]);
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, true),
        Field::new("score", DataType::Float64, true),
        Field::new(
            "attrs",
            DataType::Map(
                Arc::new(Field::new("entries", DataType::Struct(entries), false)),
                false,
            ),
            true,
        ),
        Field::new(
            "tags",
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
            true,
        ),
        Field::new(
            "point",
            DataType::Struct(Fields::from(vec![
                Field::new("x", DataType::Float64, true),
                Field::new("ok", DataType::Boolean, true),
            ])),
            true,
        ),
        Field::new("note", DataType::Utf8, true),
    ]))
}

/// The greatest `note` of [`nested_batch`]: its first row's.
const GREATEST_NOTE: &str = "a😀";

/// Rows `lo..hi` of [`nested_schema`], NULLs in every nullable column.
fn nested_batch(lo: i64, hi: i64) -> RecordBatch {
    use arrow::array::{BooleanArray, Float64Array, ListArray, MapArray, StringArray, StructArray};
    use arrow::buffer::{NullBuffer, OffsetBuffer};
    let schema = nested_schema();
    let rows = lo..hi;
    let names = StringArray::from_iter(rows.clone().map(|i| match i % 6 {
        0 => None,
        1 => Some(format!(
            "{}{}",
            "é".repeat((i % 5) as usize),
            "n".repeat(300)
        )),
        2 => Some(format!("日本-{}", i % 40)),
        _ => Some(format!("name-{i}")),
    }));
    let scores = Float64Array::from_iter(rows.clone().map(|i| match i % 9 {
        0 => None,
        1 => Some(-0.0),
        2 => Some(0.0),
        3 => Some(f64::NAN),
        _ => Some((i % 1000) as f64 * 0.5 - 100.0),
    }));
    let lengths: Vec<usize> = rows.clone().map(|i| (i % 3) as usize).collect();
    let entries: usize = lengths.iter().sum();
    let DataType::Map(entries_field, _) = schema.field(3).data_type().clone() else {
        unreachable!("attrs is a map");
    };
    let DataType::Struct(entry_fields) = entries_field.data_type().clone() else {
        unreachable!("a map's entries are a struct");
    };
    let attrs = MapArray::new(
        entries_field,
        OffsetBuffer::from_lengths(lengths),
        StructArray::new(
            entry_fields,
            vec![
                Arc::new(StringArray::from_iter_values(
                    (0..entries).map(|k| format!("key-{}", k % 4)),
                )),
                Arc::new(StringArray::from_iter(
                    (0..entries).map(|k| (k % 5 != 0).then(|| format!("value-{}", k % 11))),
                )),
            ],
            None,
        ),
        Some(NullBuffer::from_iter(rows.clone().map(|i| i % 7 != 0))),
        false,
    );
    let tags = ListArray::new(
        Arc::new(Field::new("item", DataType::Utf8, true)),
        OffsetBuffer::from_lengths(rows.clone().map(|i| usize::from(i % 4 != 0))),
        Arc::new(StringArray::from_iter_values(
            rows.clone()
                .filter(|i| i % 4 != 0)
                .map(|i| format!("tag-{}", i % 3)),
        )),
        Some(NullBuffer::from_iter(rows.clone().map(|i| i % 4 != 0))),
    );
    let DataType::Struct(point_fields) = schema.field(5).data_type().clone() else {
        unreachable!("point is a struct");
    };
    let point = StructArray::new(
        point_fields,
        vec![
            Arc::new(Float64Array::from_iter(rows.clone().map(|i| {
                (i % 5 != 0).then_some(if i % 8 == 0 {
                    -0.0
                } else {
                    i as f64
                })
            }))),
            Arc::new(BooleanArray::from_iter(
                rows.clone().map(|i| (i % 3 != 0).then_some(i % 2 == 0)),
            )),
        ],
        Some(NullBuffer::from_iter(rows.clone().map(|i| i % 10 != 0))),
    );
    // Row 0 holds the greatest note. Every eleventh row from row 7 holds a
    // smaller one of 301 bytes whose 256th byte is inside an `é`, so its
    // maximum truncates to "b", past every note.
    let notes = StringArray::from_iter(rows.clone().map(|i| match i {
        0 => Some(GREATEST_NOTE.to_string()),
        i if i % 11 == 7 => Some(format!("a{}", "é".repeat(150))),
        i if i % 11 == 0 => None,
        i => Some(format!("a-{i}")),
    }));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from_iter_values(rows)),
            Arc::new(names),
            Arc::new(scores),
            Arc::new(attrs),
            Arc::new(tags),
            Arc::new(point),
            Arc::new(notes),
        ],
    )
    .unwrap()
}

/// What a write left: every row read back through the provider in row-id
/// order, the live files' row counts in row-id order, each file's column
/// stats (without sizes, which depend on the encoding), and the table's.
struct WrittenTable {
    rows: String,
    ids: Vec<i64>,
    rowids: Vec<i64>,
    file_rows: Vec<i64>,
    file_stats: Vec<Vec<Option<String>>>,
    table_stats: Vec<Vec<Option<String>>>,
    /// The column id of `note`, as the stats rows print it.
    note: String,
}

/// `stats` rows whose column id, at `column`, is `note`, with the maximum,
/// at `max`, taken out and returned apart.
fn split_note_max(
    stats: &[Vec<Option<String>>],
    note: &str,
    column: usize,
    max: usize,
) -> (Vec<Vec<Option<String>>>, Vec<Option<String>>) {
    let mut maxima = Vec::new();
    let rest = stats
        .iter()
        .map(|row| {
            let mut row = row.clone();
            if row[column].as_deref() == Some(note) {
                maxima.push(row[max].take());
            }
            row
        })
        .collect();
    (rest, maxima)
}

async fn written_table(conn_str: &str) -> WrittenTable {
    let provider = SqliteMetadataProvider::new(conn_str).await.unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog(
        "ducklake",
        Arc::new(
            DuckLakeCatalog::new(provider)
                .unwrap()
                .with_row_lineage(true),
        ),
    );
    let batches = ctx
        .sql(
            "SELECT rowid, id, name, score, attrs, tags, point, note FROM ducklake.main.t \
             ORDER BY rowid",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let column = |i: usize| -> Vec<i64> {
        batches
            .iter()
            .flat_map(|b| {
                b.column(i)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .values()
                    .to_vec()
            })
            .collect()
    };
    let rows = datafusion::arrow::util::pretty::pretty_format_batches(&batches)
        .unwrap()
        .to_string();
    let pool = SqlitePool::connect(conn_str).await.unwrap();
    let file_rows: Vec<(i64,)> = sqlx::query_as(
        "SELECT record_count FROM ducklake_data_file WHERE end_snapshot IS NULL \
         ORDER BY row_id_start",
    )
    .fetch_all(&pool)
    .await
    .unwrap();
    let text_rows = |rows: Vec<sqlx::sqlite::SqliteRow>| -> Vec<Vec<Option<String>>> {
        use sqlx::Row;
        rows.iter()
            .map(|row| (0..row.len()).map(|i| row.get(i)).collect())
            .collect()
    };
    let file_stats = text_rows(
        sqlx::query(
            "SELECT CAST(f.row_id_start AS TEXT), CAST(s.column_id AS TEXT), \
             CAST(s.value_count AS TEXT), CAST(s.null_count AS TEXT), s.min_value, \
             s.max_value, CAST(s.contains_nan AS TEXT) \
             FROM ducklake_file_column_stats s \
             JOIN ducklake_data_file f ON f.data_file_id = s.data_file_id \
             ORDER BY f.row_id_start, s.column_id",
        )
        .fetch_all(&pool)
        .await
        .unwrap(),
    );
    let table_stats = text_rows(
        sqlx::query(
            "SELECT CAST(column_id AS TEXT), CAST(contains_null AS TEXT), \
             CAST(contains_nan AS TEXT), min_value, max_value \
             FROM ducklake_table_column_stats ORDER BY column_id",
        )
        .fetch_all(&pool)
        .await
        .unwrap(),
    );
    let (note,): (i64,) = sqlx::query_as(
        "SELECT column_id FROM ducklake_column WHERE column_name = 'note' AND end_snapshot IS NULL",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    WrittenTable {
        note: note.to_string(),
        rows,
        ids: column(1),
        rowids: column(0),
        file_rows: file_rows.into_iter().map(|(n,)| n).collect(),
        file_stats,
        table_stats,
    }
}

/// A write whose row groups are encoded on several threads holds what one
/// thread writes: the same rows, values and order read back, row ids alike,
/// and the same table statistics. Into one file, with row groups cut by the
/// row cap alone, the file's statistics are the same too, but for the one
/// known difference: a string maximum truncated inside a character, which
/// one thread compares with the next values and several threads take per row
/// group, so the file and the table record the looser `b` where one thread
/// keeps the greatest note. With a tiny target size, many row groups are
/// encoding at once while files roll: every file holds rows, and no row is
/// lost or written twice.
#[tokio::test(flavor = "multi_thread")]
async fn row_groups_encoded_on_several_threads_hold_what_one_thread_writes() {
    let mut input = Vec::new();
    let mut lo = 0;
    // Two batches larger than a row group, and some smaller.
    for size in [1, 1_200, 7, 900, 1_500, 333] {
        input.push(nested_batch(lo, lo + size));
        lo += size;
    }
    let total = lo;
    let mut options = DuckLakeWriteOptions::default().with_data_inlining_row_limit(0);
    options.max_row_group_rows = Some(250);
    for target in [1 << 30, 4 * 1024] {
        let mut written = Vec::new();
        for threads in [1, 4] {
            let env = Env::new(None).await;
            let store: Arc<dyn ObjectStore> = env.store.clone();
            let mut session = DuckLakeTableWriter::new(env.writer.clone(), store)
                .unwrap()
                .with_options(&options)
                .with_target_file_size(target)
                .with_encode_parallelism(threads)
                .begin_write("main", "t", nested_schema().as_ref(), WriteMode::Append)
                .unwrap();
            for batch in &input {
                session.write_batch(batch).unwrap();
            }
            session.finish().await.unwrap();
            let table = written_table(&env.conn_str).await;
            assert_eq!(
                table.ids,
                (0..total).collect::<Vec<_>>(),
                "{threads} threads"
            );
            assert_eq!(table.rowids, (0..total).collect::<Vec<_>>());
            assert!(
                table.file_rows.iter().all(|rows| *rows > 0),
                "{threads} threads: {:?}",
                table.file_rows
            );
            assert_eq!(table.file_rows.iter().sum::<i64>(), total);
            written.push(table);
        }
        let (one, several) = (&written[0], &written[1]);
        assert_eq!(several.rows, one.rows, "target {target}");
        assert!(!one.table_stats.is_empty());
        assert_eq!(several.note, one.note);
        if target == 1 << 30 {
            assert_eq!(several.file_rows, vec![total]);
            assert_eq!(one.file_rows, vec![total]);
            // File stats: (row_id_start, column_id, .., min, max, ..).
            let (several_files, several_max) = split_note_max(&several.file_stats, &one.note, 1, 5);
            let (one_files, one_max) = split_note_max(&one.file_stats, &one.note, 1, 5);
            assert_eq!(several_files, one_files);
            assert_eq!(one_max, vec![Some(GREATEST_NOTE.to_string())]);
            assert_eq!(several_max, vec![Some("b".to_string())]);
            // Table stats: (column_id, .., min, max).
            let (several_table, several_max) =
                split_note_max(&several.table_stats, &one.note, 0, 4);
            let (one_table, one_max) = split_note_max(&one.table_stats, &one.note, 0, 4);
            assert_eq!(several_table, one_table);
            assert_eq!(one_max, vec![Some(GREATEST_NOTE.to_string())]);
            assert_eq!(several_max, vec![Some("b".to_string())]);
        } else {
            // Several of one thread's files hold a long note and not the
            // greatest, so its table maximum is the looser one as well.
            assert_eq!(several.table_stats, one.table_stats, "target {target}");
            assert!(
                several.file_rows.len() > 4,
                "files: {:?}",
                several.file_rows
            );
        }
    }
}
