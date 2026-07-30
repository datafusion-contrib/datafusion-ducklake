//! Integration tests for SQL `DELETE FROM t [WHERE ...]` on DuckLake tables
//! (`TableProvider::delete_from` -> `DuckLakeDeleteExec`).
//!
//! Correctness is the point: a positional-delete bug silently deletes the wrong
//! rows, so every test asserts the SURVIVING ids after a fresh read (the read
//! path applies the committed delete files) plus the reported rows-affected
//! count. Run against the SQLite backend, the one the crate can exercise without
//! a container.

#![cfg(all(feature = "write-sqlite", feature = "metadata-sqlite"))]

use std::sync::Arc;

use arrow::array::{Array, Int32Array, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::prelude::*;
use object_store::local::LocalFileSystem;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use tempfile::TempDir;

use datafusion_ducklake::types::extract_parquet_field_ids;
use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeTableWriter, MetadataWriter, SqliteMetadataProvider,
    SqliteMetadataWriter,
};
use sqlx::sqlite::SqlitePool;

fn object_store() -> Arc<dyn object_store::ObjectStore> {
    Arc::new(LocalFileSystem::new())
}

fn id_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]))
}

fn id_batch(ids: &[i32]) -> RecordBatch {
    RecordBatch::try_new(id_schema(), vec![Arc::new(Int32Array::from(ids.to_vec()))]).unwrap()
}

/// A freshly-initialized, writable SQLite catalog + data dir in a temp dir.
async fn new_writer(temp: &TempDir) -> SqliteMetadataWriter {
    let db_path = temp.path().join("test.db");
    let data_path = temp.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();
    let conn = format!("sqlite:{}?mode=rwc", db_path.display());
    let writer = SqliteMetadataWriter::new_with_init(&conn).await.unwrap();
    writer.set_data_path(data_path.to_str().unwrap()).unwrap();
    writer
}

/// A writable `SessionContext` bound to the catalog's CURRENT snapshot. Create
/// this AFTER seeding data so the table provider sees the seeded files, and
/// create a fresh one after each committing statement to observe the new head.
async fn writable_ctx(temp: &TempDir) -> SessionContext {
    let conn = format!("sqlite:{}?mode=rwc", temp.path().join("test.db").display());
    let writer = SqliteMetadataWriter::new(&conn).await.unwrap();
    let provider = SqliteMetadataProvider::new(&conn).await.unwrap();
    let catalog = DuckLakeCatalog::with_writer(Arc::new(provider), Arc::new(writer)).unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));
    ctx
}

/// Read `id`s from `ducklake.main.t`, ascending, through the full read path
/// (which applies any live delete file), using a fresh read-only context.
async fn read_ids(temp: &TempDir) -> Vec<i32> {
    let conn = format!("sqlite:{}", temp.path().join("test.db").display());
    let provider = SqliteMetadataProvider::new(&conn).await.unwrap();
    let catalog = DuckLakeCatalog::new(provider).unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));
    let batches = ctx
        .sql("SELECT id FROM ducklake.main.t ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let mut ids = Vec::new();
    for b in &batches {
        let col = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        for i in 0..b.num_rows() {
            ids.push(col.value(i));
        }
    }
    ids
}

/// Number of snapshots in the catalog (to assert a multi-file DELETE commits in
/// exactly ONE new snapshot).
async fn snapshot_count(temp: &TempDir) -> i64 {
    let conn = format!("sqlite:{}", temp.path().join("test.db").display());
    let pool = SqlitePool::connect(&conn).await.unwrap();
    sqlx::query_scalar("SELECT COUNT(*) FROM ducklake_snapshot")
        .fetch_one(&pool)
        .await
        .unwrap()
}

/// Run a DELETE statement and return the reported rows-affected count.
async fn run_delete(ctx: &SessionContext, sql: &str) -> u64 {
    let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
    assert_eq!(batches.len(), 1, "DELETE returns a single count batch");
    assert_eq!(batches[0].num_rows(), 1, "count batch has one row");
    batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .expect("count column is UInt64")
        .value(0)
}

/// Single data file: `DELETE ... WHERE id = N` removes exactly that row.
#[tokio::test(flavor = "multi_thread")]
async fn delete_single_file_predicate() {
    let temp = TempDir::new().unwrap();
    let writer = Arc::new(new_writer(&temp).await);
    DuckLakeTableWriter::new(writer, object_store())
        .unwrap()
        .write_table("main", "t", &[id_batch(&[1, 2, 3, 4])])
        .await
        .unwrap();
    assert_eq!(read_ids(&temp).await, vec![1, 2, 3, 4], "baseline");

    let ctx = writable_ctx(&temp).await;
    let count = run_delete(&ctx, "DELETE FROM ducklake.main.t WHERE id = 2").await;
    assert_eq!(count, 1, "one row deleted");
    assert_eq!(read_ids(&temp).await, vec![1, 3, 4], "id 2 gone");
}

/// Rows spread across TWO data files; a predicate hitting both must delete from
/// both and commit in exactly ONE new snapshot (atomic multi-file DELETE).
#[tokio::test(flavor = "multi_thread")]
async fn delete_multi_file_atomic() {
    let temp = TempDir::new().unwrap();
    let writer = Arc::new(new_writer(&temp).await);
    let tw = DuckLakeTableWriter::new(writer, object_store()).unwrap();
    // File 1: ids [1,2,3]; File 2 (append): ids [4,5,6].
    tw.write_table("main", "t", &[id_batch(&[1, 2, 3])])
        .await
        .unwrap();
    tw.append_table("main", "t", &[id_batch(&[4, 5, 6])])
        .await
        .unwrap();
    assert_eq!(read_ids(&temp).await, vec![1, 2, 3, 4, 5, 6], "baseline");

    let before = snapshot_count(&temp).await;
    let ctx = writable_ctx(&temp).await;
    // id 2 lives in file 1; ids 4 and 6 live in file 2.
    let count = run_delete(
        &ctx,
        "DELETE FROM ducklake.main.t WHERE id = 2 OR id = 4 OR id = 6",
    )
    .await;
    assert_eq!(count, 3, "three rows deleted across two files");

    let after = snapshot_count(&temp).await;
    assert_eq!(
        after,
        before + 1,
        "multi-file DELETE must commit in exactly one new snapshot"
    );
    assert_eq!(
        read_ids(&temp).await,
        vec![1, 3, 5],
        "survivors across files"
    );
}

/// `DELETE FROM t` with no WHERE removes ALL rows (metadata-only truncate).
#[tokio::test(flavor = "multi_thread")]
async fn delete_all_rows_no_where() {
    let temp = TempDir::new().unwrap();
    let writer = Arc::new(new_writer(&temp).await);
    DuckLakeTableWriter::new(writer, object_store())
        .unwrap()
        .write_table("main", "t", &[id_batch(&[1, 2, 3, 4])])
        .await
        .unwrap();

    let ctx = writable_ctx(&temp).await;
    let count = run_delete(&ctx, "DELETE FROM ducklake.main.t").await;
    assert_eq!(count, 4, "all four rows deleted");
    assert_eq!(read_ids(&temp).await, Vec::<i32>::new(), "table empty");
}

/// Two DELETEs against the SAME data file. The second must UNION with the first
/// (cumulative positions): the row deleted first stays deleted (no
/// resurrection), and the second row is also removed.
#[tokio::test(flavor = "multi_thread")]
async fn delete_after_delete_is_cumulative() {
    let temp = TempDir::new().unwrap();
    let writer = Arc::new(new_writer(&temp).await);
    DuckLakeTableWriter::new(writer, object_store())
        .unwrap()
        .write_table("main", "t", &[id_batch(&[1, 2, 3, 4])])
        .await
        .unwrap();

    // First delete: id 2 (physical position 1).
    let ctx1 = writable_ctx(&temp).await;
    let c1 = run_delete(&ctx1, "DELETE FROM ducklake.main.t WHERE id = 2").await;
    assert_eq!(c1, 1);
    assert_eq!(read_ids(&temp).await, vec![1, 3, 4], "after first delete");

    // Second delete (fresh ctx, sees the live delete file): id 4 (position 3).
    let ctx2 = writable_ctx(&temp).await;
    let c2 = run_delete(&ctx2, "DELETE FROM ducklake.main.t WHERE id = 4").await;
    assert_eq!(c2, 1, "only the newly-matched row counts");
    assert_eq!(
        read_ids(&temp).await,
        vec![1, 3],
        "cumulative: id 2 stayed deleted, id 4 now deleted (no resurrection)"
    );
}

/// A predicate that matches only already-deleted rows is a no-op: it deletes
/// nothing new and does NOT create a snapshot.
#[tokio::test(flavor = "multi_thread")]
async fn delete_already_deleted_is_noop() {
    let temp = TempDir::new().unwrap();
    let writer = Arc::new(new_writer(&temp).await);
    DuckLakeTableWriter::new(writer, object_store())
        .unwrap()
        .write_table("main", "t", &[id_batch(&[1, 2, 3, 4])])
        .await
        .unwrap();

    let ctx1 = writable_ctx(&temp).await;
    assert_eq!(
        run_delete(&ctx1, "DELETE FROM ducklake.main.t WHERE id = 2").await,
        1
    );

    let snaps = snapshot_count(&temp).await;
    // Re-delete the same (now already-deleted) row.
    let ctx2 = writable_ctx(&temp).await;
    let c = run_delete(&ctx2, "DELETE FROM ducklake.main.t WHERE id = 2").await;
    assert_eq!(c, 0, "nothing new to delete");
    assert_eq!(
        snapshot_count(&temp).await,
        snaps,
        "a no-op DELETE creates no snapshot"
    );
    assert_eq!(read_ids(&temp).await, vec![1, 3, 4]);
}

/// A DELETE matching no rows deletes nothing and reports 0.
#[tokio::test(flavor = "multi_thread")]
async fn delete_no_match() {
    let temp = TempDir::new().unwrap();
    let writer = Arc::new(new_writer(&temp).await);
    DuckLakeTableWriter::new(writer, object_store())
        .unwrap()
        .write_table("main", "t", &[id_batch(&[1, 2, 3])])
        .await
        .unwrap();

    let ctx = writable_ctx(&temp).await;
    let count = run_delete(&ctx, "DELETE FROM ducklake.main.t WHERE id = 999").await;
    assert_eq!(count, 0);
    assert_eq!(read_ids(&temp).await, vec![1, 2, 3], "unchanged");
}

/// Interop: the positional delete parquet we write must carry DuckDB's reserved
/// field-ids (2147483646 for `file_path`, 2147483645 for `pos`) so DuckDB's own
/// `ducklake` extension can read our deletes — NOT Iceberg's 2147483546/…545.
#[tokio::test(flavor = "multi_thread")]
async fn delete_file_uses_duckdb_field_ids() {
    let temp = TempDir::new().unwrap();
    let writer = Arc::new(new_writer(&temp).await);
    let tw = DuckLakeTableWriter::new(writer, object_store()).unwrap();
    // Seed the table so the `main/t` data directory exists.
    tw.write_table("main", "t", &[id_batch(&[1, 2, 3, 4])])
        .await
        .unwrap();

    // Author a delete file directly and inspect the parquet we wrote.
    let info = tw
        .write_delete_file("main", "t", "data-file.parquet", &[1, 3])
        .await
        .unwrap();
    let del_path = temp
        .path()
        .join("data")
        .join("main")
        .join("t")
        .join(&info.path);

    let file = std::fs::File::open(&del_path).expect("delete parquet exists on disk");
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
    let field_ids = extract_parquet_field_ids(builder.metadata().as_ref());

    assert_eq!(
        field_ids.get(&2_147_483_646).map(String::as_str),
        Some("file_path"),
        "file_path must use DuckDB's FILENAME field-id 2147483646, got {field_ids:?}"
    );
    assert_eq!(
        field_ids.get(&2_147_483_645).map(String::as_str),
        Some("pos"),
        "pos must use DuckDB's ORDINAL field-id 2147483645, got {field_ids:?}"
    );
    // Explicitly assert the Iceberg ids are NOT used.
    assert!(
        !field_ids.contains_key(&2_147_483_546) && !field_ids.contains_key(&2_147_483_545),
        "must not use Iceberg positional-delete field-ids: {field_ids:?}"
    );
}

/// A no-op truncate must NOT create a snapshot. The catalog pins its snapshot, so
/// a second `DELETE FROM t` in the same session still sees the already-ended files
/// as live (bypassing the caller's emptiness guard); the DB-level guard in
/// `commit_truncate` must then decline to allocate a spurious empty snapshot.
#[tokio::test(flavor = "multi_thread")]
async fn delete_truncate_repeat_same_ctx_no_spurious_snapshot() {
    let temp = TempDir::new().unwrap();
    let writer = Arc::new(new_writer(&temp).await);
    DuckLakeTableWriter::new(writer, object_store())
        .unwrap()
        .write_table("main", "t", &[id_batch(&[1, 2, 3, 4])])
        .await
        .unwrap();

    let before = snapshot_count(&temp).await;
    let ctx = writable_ctx(&temp).await;

    let c1 = run_delete(&ctx, "DELETE FROM ducklake.main.t").await;
    assert_eq!(c1, 4, "first truncate removes all rows");
    let after_first = snapshot_count(&temp).await;
    assert_eq!(
        after_first,
        before + 1,
        "first truncate commits exactly one snapshot"
    );

    // Second truncate in the SAME (pinned) session: a DB-level no-op.
    let c2 = run_delete(&ctx, "DELETE FROM ducklake.main.t").await;
    assert_eq!(c2, 0, "nothing left to truncate");
    assert_eq!(
        snapshot_count(&temp).await,
        after_first,
        "a no-op truncate must not create a snapshot"
    );
    assert_eq!(
        read_ids(&temp).await,
        Vec::<i32>::new(),
        "table stays empty"
    );
}

/// A second filtered DELETE in the SAME session that re-touches a file modified by
/// the first must abort with a clear conflict (the catalog is pinned to the
/// pre-delete snapshot) — and, crucially, must NOT resurrect the first delete.
#[tokio::test(flavor = "multi_thread")]
async fn delete_second_in_session_conflicts_without_resurrection() {
    let temp = TempDir::new().unwrap();
    let writer = Arc::new(new_writer(&temp).await);
    DuckLakeTableWriter::new(writer, object_store())
        .unwrap()
        .write_table("main", "t", &[id_batch(&[1, 2, 3, 4])])
        .await
        .unwrap();

    // One session, catalog pinned at the pre-delete snapshot.
    let ctx = writable_ctx(&temp).await;
    assert_eq!(
        run_delete(&ctx, "DELETE FROM ducklake.main.t WHERE id = 2").await,
        1
    );
    assert_eq!(read_ids(&temp).await, vec![1, 3, 4], "id 2 deleted");

    // Second DELETE on the SAME session re-touches the same data file. Its
    // compare-and-swap disagrees with the now-live delete file, so it aborts.
    let err = ctx
        .sql("DELETE FROM ducklake.main.t WHERE id = 4")
        .await
        .unwrap()
        .collect()
        .await
        .expect_err("second in-session DELETE must conflict, not silently corrupt");
    let msg = err.to_string();
    assert!(
        msg.contains("Re-open the catalog") && msg.contains("THIS session"),
        "conflict message must explain the pinned-snapshot cause, got: {msg}"
    );

    // The abort must be clean: id 2 stayed deleted, id 4 was NOT deleted, and no
    // row was resurrected.
    assert_eq!(
        read_ids(&temp).await,
        vec![1, 3, 4],
        "aborted DELETE must not resurrect id 2 nor delete id 4"
    );
}
