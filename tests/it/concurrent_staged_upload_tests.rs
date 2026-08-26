//! Ordered-concurrent staged-file upload.
//!
//! `TableWriteSession::finish` uploads the files a rolling write produced with
//! several in flight, but must register them in WRITE ORDER: the commit assigns
//! each file's `row_id_start` by walking the list and advancing a running counter
//! (`register_data_files_with_commit_metadata`). Reordering the uploads would
//! renumber rows silently — no error, no failed commit, just different lineage ids
//! and different per-file value ranges.
//!
//! So these tests pin the ORDERING PROPERTY, not the concurrency: the same write
//! must produce byte-identical catalog rows at `upload_concurrency` 1 and at 4.
//! Official DuckLake assigns row ids in collection order too, which is why
//! preserving it here is what keeps the two equivalent.

#![cfg(all(feature = "write-sqlite", feature = "metadata-sqlite"))]

use std::sync::Arc;

use arrow::array::Int32Array;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use object_store::local::LocalFileSystem;
use tempfile::TempDir;

use datafusion_ducklake::{DuckLakeTableWriter, MetadataWriter, SqliteMetadataWriter, WriteMode};
use sqlx::sqlite::SqlitePool;

/// One committed data file as the catalog records it: `(row_id_start, id min,
/// id max, record_count)`. Everything an out-of-order registration could corrupt.
type CommittedFile = (Option<i64>, Option<String>, Option<String>, i64);

async fn create_writer(temp_dir: &TempDir) -> SqliteMetadataWriter {
    let db_path = temp_dir.path().join("test.db");
    let data_path = temp_dir.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();
    let conn_str = format!("sqlite:{}?mode=rwc", db_path.display());
    let writer = SqliteMetadataWriter::new_with_init(&conn_str)
        .await
        .unwrap();
    writer.set_data_path(data_path.to_str().unwrap()).unwrap();
    writer
}

fn table_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("val", DataType::Int32, false),
    ]))
}

/// One rolling append of `batches` at the given upload concurrency. Returns
/// `(files_written, per-file (row_id_start, min_id, max_id) ordered by data_file_id)`.
async fn rolling_append(
    upload_concurrency: usize,
) -> (
    usize,
    Vec<(Option<i64>, Option<String>, Option<String>, i64)>,
) {
    let temp_dir = TempDir::new().unwrap();
    let writer = Arc::new(create_writer(&temp_dir).await);
    let object_store: Arc<dyn object_store::ObjectStore> = Arc::new(LocalFileSystem::new());
    let schema = table_schema();

    // Ascending, disjoint id ranges per batch, and a small target file size, so the
    // write rolls into several files whose value ranges are checkable and whose
    // correct order is unambiguous.
    // Deliberately UNEVEN batch sizes. With uniform batches every file holds the
    // same row count, so `row_id_start` comes out 0, N, 2N... under every possible
    // permutation and cannot discriminate order at all.
    let mut next_id = 0i32;
    let batches: Vec<RecordBatch> = (0..24)
        .map(|b: i32| {
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
        .collect();

    let mut session = DuckLakeTableWriter::new(writer.clone(), object_store.clone())
        .unwrap()
        .with_target_file_size(4 * 1024)
        .with_upload_concurrency(upload_concurrency)
        .begin_write("main", "t", schema.as_ref(), WriteMode::Append)
        .unwrap();
    for batch in &batches {
        session.write_batch(batch).unwrap();
    }
    let result = session.finish().await.unwrap();
    assert!(
        result.files_written > 1,
        "the write must span several files or concurrency is untested, got {}",
        result.files_written
    );
    assert_eq!(result.records_written, i64::from(next_id));

    let db_path = temp_dir.path().join("test.db");
    let pool = SqlitePool::connect(&format!("sqlite:{}", db_path.display()))
        .await
        .unwrap();
    // `row_id_start` plus the id column's recorded bounds, in data_file_id order —
    // which is the order the commit walked the uploaded list in.
    let rows: Vec<CommittedFile> = sqlx::query_as(
        "SELECT d.row_id_start, s.min_value, s.max_value, d.record_count
           FROM ducklake_data_file d
           JOIN ducklake_file_column_stats s ON s.data_file_id = d.data_file_id
          WHERE d.end_snapshot IS NULL AND s.column_id = 1
          ORDER BY d.data_file_id",
    )
    .fetch_all(&pool)
    .await
    .unwrap();
    (result.files_written, rows)
}

/// The property: concurrency changes nothing the catalog records.
#[tokio::test(flavor = "multi_thread")]
async fn staged_uploads_register_in_write_order_regardless_of_concurrency() {
    let (serial_files, serial_rows) = rolling_append(1).await;
    let (concurrent_files, concurrent_rows) = rolling_append(4).await;

    assert_eq!(
        serial_files, concurrent_files,
        "same input must produce the same file count"
    );
    assert_eq!(
        serial_rows, concurrent_rows,
        "row_id_start and per-file id bounds must be identical at concurrency 1 and 4 — \
         a difference means the uploads were registered out of write order"
    );
}

/// `row_id_start` must be strictly ascending in data_file_id order and contiguous:
/// each file starts exactly where the previous one ended. This is the invariant an
/// out-of-order registration breaks, stated directly rather than only by comparison.
#[tokio::test(flavor = "multi_thread")]
async fn concurrent_staged_uploads_keep_row_ids_contiguous_and_ascending() {
    let (files_written, rows) = rolling_append(4).await;
    assert_eq!(rows.len(), files_written);

    let starts: Vec<i64> = rows.iter().filter_map(|(s, _, _, _)| *s).collect();
    assert_eq!(
        starts.len(),
        rows.len(),
        "every file must carry a row_id_start"
    );
    // THIS test is the ordering guard, via the ascending per-file minima below. The
    // sibling test that compares concurrency 1 against 4 is only a secondary
    // consistency check: a reordering bug that is not concurrency-dependent applies
    // to both of its runs and cancels out, so a total list inversion passes it.
    //
    // Contiguity itself is a commit invariant rather than an ordering guard — row ids
    // stay contiguous however the list was ordered, since the counter walks whatever
    // it is given. It is asserted because a gap or overlap corrupts lineage on its own.
    let counts: Vec<i64> = rows.iter().map(|(_, _, _, c)| *c).collect();
    let mut expected = starts[0];
    for (i, (start, count)) in starts.iter().zip(&counts).enumerate() {
        assert_eq!(
            *start, expected,
            "file {i}: row_id_start must continue the previous file's range; \
             starts={starts:?} counts={counts:?}"
        );
        expected += count;
    }

    // Per-file id ranges must also ascend and not overlap: the source ids were
    // written ascending, so a file covering a lower range than its predecessor is
    // exactly the reordering symptom.
    let mins: Vec<i32> = rows
        .iter()
        .filter_map(|(_, min, _, _)| min.as_ref().and_then(|m| m.parse::<i32>().ok()))
        .collect();
    assert_eq!(
        mins.len(),
        rows.len(),
        "every file must record an id minimum"
    );
    assert!(
        mins.windows(2).all(|w| w[0] < w[1]),
        "per-file id minima must ascend with data_file_id, got {mins:?}"
    );
    // Ascending minima alone do not prove the ranges do not OVERLAP, and `max` is
    // already selected — so assert the real property: each file's range ends before
    // the next one begins.
    let maxes: Vec<i32> = rows
        .iter()
        .filter_map(|(_, _, max, _)| max.as_ref().and_then(|m| m.parse::<i32>().ok()))
        .collect();
    assert_eq!(
        maxes.len(),
        rows.len(),
        "every file must record an id maximum"
    );
    for i in 0..rows.len() - 1 {
        assert!(
            maxes[i] < mins[i + 1],
            "file {i} must end before file {} begins: maxes={maxes:?} mins={mins:?}",
            i + 1
        );
    }
    // And the sequence must start at 0 and cover every row, which pins the ends that
    // a purely relative check leaves free.
    assert_eq!(starts[0], 0, "row ids must start at 0, got {starts:?}");
    // The fixture's premise: uniform file sizes make `row_id_start` identical under
    // every permutation, so the checks above would prove nothing. Pin it, or a future
    // change to rollover silently degrades this test to a tautology.
    assert!(
        counts
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len()
            > 1,
        "files must differ in row count or ordering cannot be detected: {counts:?}"
    );
    assert_eq!(
        starts[starts.len() - 1] + counts[counts.len() - 1],
        counts.iter().sum::<i64>(),
        "the last file must end at the total row count"
    );
}

/// A store that delegates to a real one but fails the Nth write.
///
/// Exists because the cleanup branch — delete what landed when a sibling upload
/// fails — is the one part of this change that DELETES objects, and it is
/// unreachable from a success-path test. An untested delete loop is exactly the
/// thing that should not ship.
#[derive(Debug)]
struct FailNthWrite {
    inner: Arc<dyn object_store::ObjectStore>,
    writes: std::sync::atomic::AtomicUsize,
    fail_on: usize,
}

impl std::fmt::Display for FailNthWrite {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FailNthWrite({})", self.fail_on)
    }
}

impl FailNthWrite {
    fn should_fail(&self) -> bool {
        use std::sync::atomic::Ordering;
        self.writes.fetch_add(1, Ordering::SeqCst) + 1 == self.fail_on
    }
    fn injected() -> object_store::Error {
        object_store::Error::Generic {
            store: "FailNthWrite",
            source: "injected upload failure".into(),
        }
    }
}

#[async_trait::async_trait]
impl object_store::ObjectStore for FailNthWrite {
    async fn put_opts(
        &self,
        location: &object_store::path::Path,
        payload: object_store::PutPayload,
        opts: object_store::PutOptions,
    ) -> object_store::Result<object_store::PutResult> {
        if self.should_fail() {
            return Err(Self::injected());
        }
        self.inner.put_opts(location, payload, opts).await
    }

    async fn put_multipart_opts(
        &self,
        location: &object_store::path::Path,
        opts: object_store::PutMultipartOptions,
    ) -> object_store::Result<Box<dyn object_store::MultipartUpload>> {
        if self.should_fail() {
            return Err(Self::injected());
        }
        self.inner.put_multipart_opts(location, opts).await
    }

    async fn get_opts(
        &self,
        location: &object_store::path::Path,
        options: object_store::GetOptions,
    ) -> object_store::Result<object_store::GetResult> {
        self.inner.get_opts(location, options).await
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

/// When one upload in a concurrent batch fails, the files that DID land must be
/// removed rather than left as orphans, and the original error must surface — not a
/// cleanup error masking it.
///
/// This is the failure path official DuckLake covers by tracking the files a write
/// created and removing them when it does not commit.
#[tokio::test(flavor = "multi_thread")]
async fn a_failed_upload_removes_the_files_that_already_landed() {
    use futures::StreamExt;

    const FAIL_ON: usize = 3;
    const CONCURRENCY: usize = 4;
    // Enough batches to roll into MANY files: the skip is only observable if the
    // write would otherwise attempt far more uploads than the in-flight window.
    const TOTAL_BATCHES: usize = 400;

    let temp_dir = TempDir::new().unwrap();
    let writer = Arc::new(create_writer(&temp_dir).await);
    let data_dir = temp_dir.path().join("data");
    let real: Arc<dyn object_store::ObjectStore> =
        Arc::new(LocalFileSystem::new_with_prefix(&data_dir).unwrap());
    // Fail the 3rd write. Which FILE that is depends on scheduling, but the outcome
    // does not: at least one file lands before it, and the assertions below are
    // indifferent to which.
    let failing = Arc::new(FailNthWrite {
        inner: Arc::clone(&real),
        writes: std::sync::atomic::AtomicUsize::new(0),
        fail_on: FAIL_ON,
    });
    let store_writes = Arc::clone(&failing);
    let store: Arc<dyn object_store::ObjectStore> = failing;
    let schema = table_schema();

    let mut session = DuckLakeTableWriter::new(writer.clone(), Arc::clone(&store))
        .unwrap()
        .with_target_file_size(4 * 1024)
        .with_upload_concurrency(CONCURRENCY)
        .begin_write("main", "t", schema.as_ref(), WriteMode::Append)
        .unwrap();
    for b in 0..TOTAL_BATCHES as i32 {
        let ids: Vec<i32> = (b * 100..(b + 1) * 100).collect();
        let vals: Vec<i32> = ids.iter().map(|id| id * 10).collect();
        session
            .write_batch(
                &RecordBatch::try_new(
                    schema.clone(),
                    vec![Arc::new(Int32Array::from(ids)), Arc::new(Int32Array::from(vals))],
                )
                .unwrap(),
            )
            .unwrap();
    }
    let err = session
        .finish()
        .await
        .expect_err("the injected upload failure must fail the write");
    let msg = err.to_string();
    assert!(
        msg.contains("injected upload failure"),
        "the original upload error must surface; got: {msg}"
    );

    // Nothing may be left behind: every object this batch landed before the failure
    // must have been removed. A leak here is billable storage no snapshot references.
    let leftovers: Vec<String> = real
        .list(None)
        .filter_map(|m| async move { m.ok().map(|m| m.location.to_string()) })
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .filter(|p| p.ends_with(".parquet"))
        .collect();
    assert!(
        leftovers.is_empty(),
        "a failed concurrent upload batch must leave no data files behind, found: {leftovers:?}"
    );

    // The skip itself, which nothing else pins: once a batch has failed, uploads that
    // have not started are abandoned rather than drained. Without it, EVERY file in
    // the batch is uploaded before the error surfaces — which on a real store means
    // one full retry budget each, turning an outage into an hours-long hang instead
    // of a prompt failure. Removing the skip flag and reverting to an unconditional
    // drain leaves every other assertion in this file passing, so assert the write
    // count directly.
    let attempted = store_writes
        .writes
        .load(std::sync::atomic::Ordering::SeqCst);
    // The write rolls into far more files than this, so an unconditional drain would
    // attempt all of them. Only the in-flight window may run past the first failure.
    assert!(
        attempted <= FAIL_ON + CONCURRENCY,
        "a failed batch must abandon uploads it had not started: attempted \
         {attempted}, expected <= {} (fail_on {FAIL_ON} + concurrency {CONCURRENCY})",
        FAIL_ON + CONCURRENCY
    );
}
