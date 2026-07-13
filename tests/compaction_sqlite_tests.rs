//! Integration tests for explicit DuckLake compaction on the SQLite backend:
//! `DuckLakeTable::merge_adjacent_files` and `rewrite_data_files`.
//!
//! Compaction rewrites data files, so these assert the load-bearing invariants
//! end-to-end: fewer live files with identical query results, exactly one new
//! snapshot, source files retired + scheduled for deletion, rowid lineage
//! preserved, time travel to a pre-compaction snapshot still returning the
//! original rows, and the same-schema-version merge boundary.

#![cfg(all(feature = "write-sqlite", feature = "metadata-sqlite"))]

use std::sync::Arc;

use arrow::array::{Array, Int32Array, Int64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::prelude::*;
use object_store::local::LocalFileSystem;
use sqlx::Row;
use sqlx::sqlite::SqlitePool;
use tempfile::TempDir;

use datafusion_ducklake::maintenance::{CleanupCriteria, cleanup_old_files_sqlite};
use datafusion_ducklake::{
    CompactionResult, DuckLakeCatalog, DuckLakeTable, DuckLakeTableWriter, MergeOptions,
    MetadataWriter, RewriteOptions, SqliteMetadataProvider, SqliteMetadataWriter,
};

fn two_col_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("val", DataType::Int32, false),
    ]))
}

fn object_store() -> Arc<dyn object_store::ObjectStore> {
    Arc::new(LocalFileSystem::new())
}

fn db_url(temp: &TempDir) -> String {
    format!("sqlite:{}?mode=rwc", temp.path().join("test.db").display())
}

fn ro_url(temp: &TempDir) -> String {
    format!("sqlite:{}", temp.path().join("test.db").display())
}

async fn make_writer(temp: &TempDir) -> SqliteMetadataWriter {
    let data_path = temp.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();
    let writer = SqliteMetadataWriter::new_with_init(&db_url(temp))
        .await
        .unwrap();
    writer.set_data_path(data_path.to_str().unwrap()).unwrap();
    writer
}

fn batch(schema: Arc<Schema>, cols: Vec<Arc<dyn Array>>) -> RecordBatch {
    RecordBatch::try_new(schema, cols).unwrap()
}

/// Seed a fresh `main.t(id, val)` as one data file (Replace on a new table).
async fn seed(temp: &TempDir, ids: Vec<i32>, vals: Vec<i32>) {
    let writer = Arc::new(make_writer(temp).await);
    let b = batch(
        two_col_schema(),
        vec![Arc::new(Int32Array::from(ids)), Arc::new(Int32Array::from(vals))],
    );
    DuckLakeTableWriter::new(writer, object_store())
        .unwrap()
        .write_table("main", "t", &[b])
        .await
        .unwrap();
}

/// Append one more `(id, val)` data file to `main.t`.
async fn append(temp: &TempDir, ids: Vec<i32>, vals: Vec<i32>) {
    let writer = Arc::new(SqliteMetadataWriter::new(&db_url(temp)).await.unwrap());
    let b = batch(
        two_col_schema(),
        vec![Arc::new(Int32Array::from(ids)), Arc::new(Int32Array::from(vals))],
    );
    DuckLakeTableWriter::new(writer, object_store())
        .unwrap()
        .append_table("main", "t", &[b])
        .await
        .unwrap();
}

async fn pool(temp: &TempDir) -> SqlitePool {
    SqlitePool::connect(&ro_url(temp)).await.unwrap()
}

async fn scalar_i64(p: &SqlitePool, sql: &str) -> i64 {
    sqlx::query(sql)
        .fetch_one(p)
        .await
        .unwrap()
        .try_get::<i64, _>(0)
        .unwrap()
}

async fn opt_i64(p: &SqlitePool, sql: &str) -> Option<i64> {
    sqlx::query(sql)
        .fetch_one(p)
        .await
        .unwrap()
        .try_get::<Option<i64>, _>(0)
        .unwrap()
}

/// Current live `(id, val)` rows of `main.t`, ascending, through the full read
/// path (which applies any live delete file / embedded-rowid file).
async fn read_rows(temp: &TempDir) -> Vec<(i32, i32)> {
    let provider = SqliteMetadataProvider::new(&ro_url(temp)).await.unwrap();
    rows_via(DuckLakeCatalog::new(provider).unwrap()).await
}

/// `(id, val)` rows of `main.t` as of `snapshot` (time travel).
async fn read_rows_at(temp: &TempDir, snapshot: i64) -> Vec<(i32, i32)> {
    let provider = Arc::new(SqliteMetadataProvider::new(&ro_url(temp)).await.unwrap());
    rows_via(DuckLakeCatalog::with_snapshot(provider, snapshot).unwrap()).await
}

async fn rows_via(catalog: DuckLakeCatalog) -> Vec<(i32, i32)> {
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));
    let batches = ctx
        .sql("SELECT id, val FROM ducklake.main.t ORDER BY id, val")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let mut out = Vec::new();
    for b in &batches {
        let ids = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        let vals = b.column(1).as_any().downcast_ref::<Int32Array>().unwrap();
        for i in 0..b.num_rows() {
            out.push((ids.value(i), vals.value(i)));
        }
    }
    out
}

/// Current live `(id, rowid)` of `main.t`, ascending by id, via a row-lineage
/// catalog — the rowid is each row's DuckLake row-lineage id.
async fn read_id_rowid(temp: &TempDir) -> Vec<(i32, i64)> {
    let provider = SqliteMetadataProvider::new(&ro_url(temp)).await.unwrap();
    let catalog = DuckLakeCatalog::new(provider)
        .unwrap()
        .with_row_lineage(true);
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));
    let batches = ctx
        .sql("SELECT id, rowid FROM ducklake.main.t ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let mut out = Vec::new();
    for b in &batches {
        let ids = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        let rids = b.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
        for i in 0..b.num_rows() {
            out.push((ids.value(i), rids.value(i)));
        }
    }
    out
}

/// Downcast the writable `main.t` provider to a `DuckLakeTable` and run `op` on
/// it (the compaction ops are `DuckLakeTable` methods). A fresh writable catalog
/// is opened so the table binds to the latest snapshot.
async fn with_writable_table<F, Fut>(temp: &TempDir, op: F) -> CompactionResult
where
    F: FnOnce(DuckLakeTable, datafusion::execution::SessionState) -> Fut,
    Fut: std::future::Future<Output = CompactionResult>,
{
    let writer = SqliteMetadataWriter::new(&db_url(temp)).await.unwrap();
    let provider = SqliteMetadataProvider::new(&db_url(temp)).await.unwrap();
    let catalog = DuckLakeCatalog::with_writer(Arc::new(provider), Arc::new(writer)).unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));
    let provider = ctx
        .catalog("ducklake")
        .unwrap()
        .schema("main")
        .unwrap()
        .table("t")
        .await
        .unwrap()
        .unwrap();
    let table = (provider.as_ref() as &dyn std::any::Any)
        .downcast_ref::<DuckLakeTable>()
        .expect("provider is a DuckLakeTable")
        .clone();
    op(table, ctx.state()).await
}

async fn run_merge(temp: &TempDir, opts: MergeOptions) -> CompactionResult {
    with_writable_table(temp, |table, state| async move {
        table.merge_adjacent_files(&state, opts).await.unwrap()
    })
    .await
}

async fn run_rewrite(temp: &TempDir, opts: RewriteOptions) -> CompactionResult {
    with_writable_table(temp, |table, state| async move {
        table.rewrite_data_files(&state, opts).await.unwrap()
    })
    .await
}

#[tokio::test(flavor = "multi_thread")]
async fn merge_coalesces_small_files_preserving_results_rowids_and_time_travel() {
    let temp = TempDir::new().unwrap();
    // Three inserts -> three small data files, all at schema version 1.
    seed(&temp, vec![1, 2], vec![10, 20]).await;
    append(&temp, vec![3, 4], vec![30, 40]).await;
    append(&temp, vec![5, 6], vec![50, 60]).await;

    let p = pool(&temp).await;
    let tid = scalar_i64(&p, "SELECT table_id FROM ducklake_table LIMIT 1").await;
    let pre_snapshot = scalar_i64(&p, "SELECT MAX(snapshot_id) FROM ducklake_snapshot").await;
    let snapshots_before = scalar_i64(&p, "SELECT COUNT(*) FROM ducklake_snapshot").await;
    let live_before = scalar_i64(
        &p,
        "SELECT COUNT(*) FROM ducklake_data_file WHERE end_snapshot IS NULL",
    )
    .await;
    assert_eq!(live_before, 3, "three small files before merge");
    // The oldest source's origin snapshot — the merged partial file must begin
    // here so historical reads back to this point still see it.
    let min_origin = scalar_i64(&p, "SELECT MIN(begin_snapshot) FROM ducklake_data_file").await;

    let rows_before = read_rows(&temp).await;
    let id_rowid_before = read_id_rowid(&temp).await;
    assert_eq!(
        rows_before,
        vec![(1, 10), (2, 20), (3, 30), (4, 40), (5, 50), (6, 60)]
    );

    // Default options: a huge target coalesces all three tiny files into one.
    let result = run_merge(&temp, MergeOptions::default()).await;
    assert_eq!(result.files_processed, 3, "all three sources merged");
    assert_eq!(result.files_created, 1, "into one file");
    assert_eq!(result.rows_written, 6);

    // Exactly one new snapshot.
    let snapshots_after = scalar_i64(&p, "SELECT COUNT(*) FROM ducklake_snapshot").await;
    assert_eq!(
        snapshots_after,
        snapshots_before + 1,
        "exactly one new snapshot"
    );
    let new_snapshot = scalar_i64(&p, "SELECT MAX(snapshot_id) FROM ducklake_snapshot").await;
    assert_eq!(new_snapshot, pre_snapshot + 1);

    // Fewer live files: exactly one, and it is the partial merged file.
    let live_after = scalar_i64(
        &p,
        "SELECT COUNT(*) FROM ducklake_data_file WHERE end_snapshot IS NULL",
    )
    .await;
    assert_eq!(live_after, 1, "one live file after merge");
    let partial_max = opt_i64(
        &p,
        "SELECT partial_max FROM ducklake_data_file WHERE end_snapshot IS NULL",
    )
    .await;
    assert_eq!(
        partial_max,
        Some(pre_snapshot),
        "partial_max = max origin snapshot among merged rows"
    );
    let merged_row_id_start = opt_i64(
        &p,
        "SELECT row_id_start FROM ducklake_data_file WHERE end_snapshot IS NULL",
    )
    .await;
    assert_eq!(
        merged_row_id_start, None,
        "merged file serves rowids inline"
    );
    let merged_begin = scalar_i64(
        &p,
        "SELECT begin_snapshot FROM ducklake_data_file WHERE end_snapshot IS NULL",
    )
    .await;
    assert_eq!(
        merged_begin, min_origin,
        "merged partial file begins at the MIN origin snapshot (visible to history)"
    );

    // The three source rows are REMOVED from the catalog (not just retired) — the
    // partial file now represents them for every snapshot — so only the one
    // merged row remains in ducklake_data_file.
    let total_rows = scalar_i64(&p, "SELECT COUNT(*) FROM ducklake_data_file").await;
    assert_eq!(
        total_rows, 1,
        "source rows removed; only the merged file remains"
    );
    // Their physical files are scheduled for deletion (safe: unreachable now).
    let scheduled = scalar_i64(
        &p,
        "SELECT COUNT(*) FROM ducklake_files_scheduled_for_deletion",
    )
    .await;
    assert_eq!(scheduled, 3, "three source files scheduled for deletion");

    // changes_made records the compaction.
    let changes: String = sqlx::query(&format!(
        "SELECT changes_made FROM ducklake_snapshot_changes WHERE snapshot_id = {new_snapshot}"
    ))
    .fetch_one(&p)
    .await
    .unwrap()
    .try_get(0)
    .unwrap();
    assert_eq!(changes, format!("compacted_table:{tid}"));

    // Identical query results, and rowid lineage preserved across the rewrite.
    assert_eq!(
        read_rows(&temp).await,
        rows_before,
        "results unchanged by merge"
    );
    assert_eq!(
        read_id_rowid(&temp).await,
        id_rowid_before,
        "rowids preserved across merge"
    );

    // Time travel to the pre-merge snapshot still returns the original rows
    // (the retired source files are only scheduled, not yet deleted).
    assert_eq!(
        read_rows_at(&temp, pre_snapshot).await,
        rows_before,
        "time travel to pre-merge snapshot returns the original rows"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn rewrite_drops_deleted_rows_and_retires_data_and_delete_files() {
    let temp = TempDir::new().unwrap();
    // One file of ten rows.
    seed(
        &temp,
        (1..=10).collect(),
        (1..=10).map(|v| v * 10).collect(),
    )
    .await;
    let p = pool(&temp).await;
    let tid = scalar_i64(&p, "SELECT table_id FROM ducklake_table LIMIT 1").await;
    let create_snapshot = scalar_i64(&p, "SELECT MAX(snapshot_id) FROM ducklake_snapshot").await;

    // Delete eight of the ten rows via SQL (a positional delete file).
    {
        let writer = SqliteMetadataWriter::new(&db_url(&temp)).await.unwrap();
        let provider = SqliteMetadataProvider::new(&db_url(&temp)).await.unwrap();
        let catalog = DuckLakeCatalog::with_writer(Arc::new(provider), Arc::new(writer)).unwrap();
        let ctx = SessionContext::new();
        ctx.register_catalog("ducklake", Arc::new(catalog));
        ctx.sql("DELETE FROM ducklake.main.t WHERE id <= 8")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
    }
    let after_delete_snapshot =
        scalar_i64(&p, "SELECT MAX(snapshot_id) FROM ducklake_snapshot").await;
    assert_eq!(
        read_rows(&temp).await,
        vec![(9, 90), (10, 100)],
        "8 of 10 deleted"
    );
    // Sanity: one live data file with a live delete file masking 8 rows.
    assert_eq!(
        scalar_i64(
            &p,
            "SELECT COUNT(*) FROM ducklake_delete_file WHERE end_snapshot IS NULL"
        )
        .await,
        1
    );

    // 8/10 = 0.8 deleted; rewrite with a 0.5 threshold.
    let result = run_rewrite(
        &temp,
        RewriteOptions {
            delete_threshold: 0.5,
        },
    )
    .await;
    assert_eq!(result.files_processed, 1);
    assert_eq!(result.files_created, 1);
    assert_eq!(result.rows_written, 2, "only the two live rows rewritten");

    let new_snapshot = scalar_i64(&p, "SELECT MAX(snapshot_id) FROM ducklake_snapshot").await;
    assert_eq!(
        new_snapshot,
        after_delete_snapshot + 1,
        "exactly one new snapshot"
    );

    // Results unchanged.
    assert_eq!(read_rows(&temp).await, vec![(9, 90), (10, 100)]);

    // Exactly one live data file, with record_count = live rows and no live delete file.
    assert_eq!(
        scalar_i64(
            &p,
            "SELECT COUNT(*) FROM ducklake_data_file WHERE end_snapshot IS NULL"
        )
        .await,
        1
    );
    assert_eq!(
        scalar_i64(
            &p,
            "SELECT record_count FROM ducklake_data_file WHERE end_snapshot IS NULL"
        )
        .await,
        2,
        "new file contains only the live rows"
    );
    assert_eq!(
        opt_i64(
            &p,
            "SELECT partial_max FROM ducklake_data_file WHERE end_snapshot IS NULL"
        )
        .await,
        None,
        "a rewrite output is not a partial file"
    );
    assert_eq!(
        scalar_i64(
            &p,
            "SELECT COUNT(*) FROM ducklake_delete_file WHERE end_snapshot IS NULL"
        )
        .await,
        0,
        "no live delete file after rewrite"
    );

    // BOTH the old data file AND its delete file retired at the new snapshot ...
    assert_eq!(
        scalar_i64(
            &p,
            &format!("SELECT COUNT(*) FROM ducklake_data_file WHERE end_snapshot = {new_snapshot}")
        )
        .await,
        1
    );
    assert_eq!(
        scalar_i64(
            &p,
            &format!(
                "SELECT COUNT(*) FROM ducklake_delete_file WHERE end_snapshot = {new_snapshot}"
            )
        )
        .await,
        1
    );
    // ... but NOT scheduled for deletion: a rewrite output holds only the
    // currently-live rows, so the retired source (all ten rows + its delete
    // file) still serves time travel to the pre-rewrite snapshot. It is
    // reclaimed later by expire_snapshots, not at the rewrite.
    assert_eq!(
        scalar_i64(
            &p,
            "SELECT COUNT(*) FROM ducklake_files_scheduled_for_deletion"
        )
        .await,
        0,
        "rewrite sources are retained (not scheduled) for time travel"
    );

    // changes_made records the compaction.
    let changes: String = sqlx::query(&format!(
        "SELECT changes_made FROM ducklake_snapshot_changes WHERE snapshot_id = {new_snapshot}"
    ))
    .fetch_one(&p)
    .await
    .unwrap()
    .try_get(0)
    .unwrap();
    assert_eq!(changes, format!("compacted_table:{tid}"));

    // Rowid lineage of the surviving rows preserved (row 9 was position 8, row 10 position 9).
    assert_eq!(read_id_rowid(&temp).await, vec![(9, 8), (10, 9)]);

    // Time travel to before the DELETE still returns all ten rows (the original
    // data file is retained, and — unscheduled — survives even a cleanup).
    assert_eq!(read_rows_at(&temp, create_snapshot).await.len(), 10);
}

#[tokio::test(flavor = "multi_thread")]
async fn merge_respects_schema_version_boundary() {
    let temp = TempDir::new().unwrap();
    // Two files at schema version 1.
    seed(&temp, vec![1, 2], vec![10, 20]).await;
    append(&temp, vec![3, 4], vec![30, 40]).await;

    // A DDL (append a batch with an extra column) bumps schema_version to 2 and
    // adds a third file under the new version, without retiring the first two.
    {
        let three_col = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("val", DataType::Int32, false),
            Field::new("note", DataType::Int32, true),
        ]));
        let writer = Arc::new(SqliteMetadataWriter::new(&db_url(&temp)).await.unwrap());
        let b = batch(
            three_col,
            vec![
                Arc::new(Int32Array::from(vec![5, 6])),
                Arc::new(Int32Array::from(vec![50, 60])),
                Arc::new(Int32Array::from(vec![500, 600])),
            ],
        );
        DuckLakeTableWriter::new(writer, object_store())
            .unwrap()
            .append_table("main", "t", &[b])
            .await
            .unwrap();
    }

    let p = pool(&temp).await;
    // Confirm the setup: three live files spanning two schema versions.
    assert_eq!(
        scalar_i64(
            &p,
            "SELECT COUNT(*) FROM ducklake_data_file WHERE end_snapshot IS NULL"
        )
        .await,
        3
    );
    assert_eq!(
        scalar_i64(
            &p,
            "SELECT COUNT(DISTINCT schema_version) FROM ducklake_schema_versions"
        )
        .await,
        2,
        "a DDL bumped the schema version"
    );
    let v2_file = scalar_i64(
        &p,
        "SELECT MAX(data_file_id) FROM ducklake_data_file WHERE end_snapshot IS NULL",
    )
    .await;

    // Merge: only the two same-version files may combine; the newer-version file
    // must be left alone (never merged across the DDL boundary).
    let result = run_merge(&temp, MergeOptions::default()).await;
    assert_eq!(result.files_processed, 2, "only the two v1 files merged");
    assert_eq!(result.files_created, 1);

    // The v2 file is untouched (still live, never scheduled).
    assert_eq!(
        scalar_i64(
            &p,
            &format!(
                "SELECT COUNT(*) FROM ducklake_data_file \
                 WHERE data_file_id = {v2_file} AND end_snapshot IS NULL"
            )
        )
        .await,
        1,
        "the newer-schema-version file was not merged"
    );
    assert_eq!(
        scalar_i64(
            &p,
            "SELECT COUNT(*) FROM ducklake_files_scheduled_for_deletion"
        )
        .await,
        2,
        "only the two v1 source files scheduled"
    );
    // Two live files remain: the merged v1 file and the untouched v2 file.
    assert_eq!(
        scalar_i64(
            &p,
            "SELECT COUNT(*) FROM ducklake_data_file WHERE end_snapshot IS NULL"
        )
        .await,
        2
    );

    // Results are unchanged by the (partial) merge.
    assert_eq!(
        read_rows(&temp).await,
        vec![(1, 10), (2, 20), (3, 30), (4, 40), (5, 50), (6, 60)]
    );
}

/// A partial file must NEVER be re-merged: the read path that reconstructs a
/// source's rows does not surface the embedded per-row origin column, so
/// re-merging would collapse every row onto the file's single begin_snapshot and
/// corrupt time travel. `merge_adjacent_files` therefore excludes partial files
/// from its candidates.
#[tokio::test(flavor = "multi_thread")]
async fn merge_never_remerges_a_partial_file() {
    let temp = TempDir::new().unwrap();
    seed(&temp, vec![1], vec![10]).await; // snapshot 1
    append(&temp, vec![2], vec![20]).await; // snapshot 2
    append(&temp, vec![3], vec![30]).await; // snapshot 3

    // Merge #1 -> partial file P (origins {1,2,3}, begin=1, partial_max=3).
    let r1 = run_merge(&temp, MergeOptions::default()).await;
    assert_eq!(r1.files_created, 1);
    let p = pool(&temp).await;
    assert_eq!(
        opt_i64(
            &p,
            "SELECT partial_max FROM ducklake_data_file WHERE end_snapshot IS NULL"
        )
        .await,
        Some(3),
        "merge produced a partial file"
    );

    append(&temp, vec![4], vec![40]).await; // snapshot 5 (one more small file, D)

    // Merge #2: P is excluded (partial), leaving only D — a single file, so no
    // group of >= 2 forms and nothing is merged. Crucially, P is not re-merged.
    let r2 = run_merge(&temp, MergeOptions::default()).await;
    assert_eq!(
        r2.files_processed, 0,
        "the partial file P is not a candidate; D alone cannot merge"
    );

    // Time travel to snapshot 2 must still return exactly rows from origins <= 2.
    // If P had been re-merged, its rows would all carry origin 1 and (3,30) would
    // wrongly reappear here.
    assert_eq!(
        read_rows_at(&temp, 2).await,
        vec![(1, 10), (2, 20)],
        "time travel intact: no origins collapsed by a re-merge"
    );
    // Current snapshot sees everything.
    assert_eq!(
        read_rows(&temp).await,
        vec![(1, 10), (2, 20), (3, 30), (4, 40)]
    );
}

/// Merge must not silently drop a column's data. When a column has been dropped
/// since a file was written, merging that file (output is written at the CURRENT
/// schema) would lose the column — so `merge_adjacent_files` skips such files.
#[tokio::test(flavor = "multi_thread")]
async fn merge_skips_files_whose_columns_were_dropped() {
    let temp = TempDir::new().unwrap();
    // Two (id, val) files at schema version 1.
    seed(&temp, vec![1, 2], vec![10, 20]).await; // snapshot 1 (file A)
    append(&temp, vec![3, 4], vec![30, 40]).await; // snapshot 2 (file B)

    // A DDL that DROPS `val` (append a batch with only `id`) bumps the schema and
    // ends the `val` column; A and B stay live (Append), now at an older version
    // whose schema includes a column absent from the current one.
    {
        let id_only = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let writer = Arc::new(SqliteMetadataWriter::new(&db_url(&temp)).await.unwrap());
        let b = batch(id_only, vec![Arc::new(Int32Array::from(vec![5]))]);
        DuckLakeTableWriter::new(writer, object_store())
            .unwrap()
            .append_table("main", "t", &[b])
            .await
            .unwrap();
    }

    let p = pool(&temp).await;
    assert_eq!(
        scalar_i64(
            &p,
            "SELECT COUNT(*) FROM ducklake_data_file WHERE end_snapshot IS NULL"
        )
        .await,
        3,
        "three live files (A, B at v1; C at v2)"
    );

    // Merge: the v1 group {A, B} carries `val`, which the current schema dropped,
    // so it is skipped (merging would lose `val`); the v2 file is a singleton.
    let result = run_merge(&temp, MergeOptions::default()).await;
    assert_eq!(
        result.files_processed, 0,
        "column-dropping files are not merged"
    );
    assert_eq!(
        scalar_i64(
            &p,
            "SELECT COUNT(*) FROM ducklake_files_scheduled_for_deletion"
        )
        .await,
        0,
        "nothing merged, nothing scheduled"
    );
    assert_eq!(
        scalar_i64(
            &p,
            "SELECT COUNT(*) FROM ducklake_data_file WHERE end_snapshot IS NULL"
        )
        .await,
        3,
        "all three files remain live (A, B not removed)"
    );

    // Time travel to snapshot 1 still returns A's rows WITH `val` intact — proof
    // that A was not merged into a current-schema (val-less) file and removed.
    assert_eq!(read_rows_at(&temp, 1).await, vec![(1, 10), (2, 20)]);
}

/// The durability property: after a merge, physically deleting the retired
/// source files (via `cleanup_old_files`) must NOT break time travel — the
/// merged partial file serves every historical snapshot on its own, via per-row
/// `_ducklake_internal_snapshot_id` filtering. This is the case the pre-fix
/// implementation got wrong (it scheduled sources while the merged file was
/// invisible to historical reads).
#[tokio::test(flavor = "multi_thread")]
async fn merge_partial_file_serves_time_travel_after_sources_are_deleted() {
    let temp = TempDir::new().unwrap();
    seed(&temp, vec![1, 2], vec![10, 20]).await; // snapshot 1
    append(&temp, vec![3, 4], vec![30, 40]).await; // snapshot 2
    append(&temp, vec![5, 6], vec![50, 60]).await; // snapshot 3

    let p = pool(&temp).await;
    let result = run_merge(&temp, MergeOptions::default()).await;
    assert_eq!(result.files_processed, 3);
    assert_eq!(result.files_created, 1);

    // Physically delete the scheduled source parquet files. Afterwards the three
    // ORIGINAL files are gone from disk; only the merged partial file remains.
    let deleted = {
        let writer = SqliteMetadataWriter::new(&db_url(&temp)).await.unwrap();
        cleanup_old_files_sqlite(&writer, object_store(), CleanupCriteria::All, false)
            .await
            .unwrap()
    };
    assert_eq!(
        deleted.len(),
        3,
        "three source parquet files physically deleted"
    );
    assert_eq!(
        scalar_i64(
            &p,
            "SELECT COUNT(*) FROM ducklake_files_scheduled_for_deletion"
        )
        .await,
        0,
        "scheduled rows cleared after cleanup"
    );

    // Time travel is now served ENTIRELY by the merged partial file (the sources
    // no longer exist) via per-row origin-snapshot filtering.
    assert_eq!(
        read_rows_at(&temp, 1).await,
        vec![(1, 10), (2, 20)],
        "as of snapshot 1: only the first insert"
    );
    assert_eq!(
        read_rows_at(&temp, 2).await,
        vec![(1, 10), (2, 20), (3, 30), (4, 40)],
        "as of snapshot 2: first two inserts"
    );
    assert_eq!(
        read_rows_at(&temp, 3).await,
        vec![(1, 10), (2, 20), (3, 30), (4, 40), (5, 50), (6, 60)],
        "as of snapshot 3: all three inserts"
    );
    // The current snapshot still returns everything.
    assert_eq!(read_rows(&temp).await.len(), 6);
}
