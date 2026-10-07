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
    DuckLakeTableWriter, DuckLakeWriteOptions, MetadataWriter, SnapshotCommitMetadata,
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
