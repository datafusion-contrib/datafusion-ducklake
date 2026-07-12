//! Integration tests for SQL `UPDATE` support (SQLite metadata backend).
//!
//! Exercises `UPDATE t SET col = expr [, ...] [WHERE ...]` end-to-end through
//! DataFusion's SQL interface against a writable DuckLake catalog: affected-row
//! count, the resulting values, atomicity (one snapshot), rowid-lineage
//! preservation across the file rewrite, the change feed (preimage/postimage),
//! and update-all.

#![cfg(all(feature = "write-sqlite", feature = "metadata-sqlite"))]

use std::sync::Arc;

use arrow::array::{Array, Int32Array, Int64Array, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::prelude::*;
use object_store::local::LocalFileSystem;
use sqlx::sqlite::SqlitePool;
use tempfile::TempDir;

use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeTableWriter, MetadataProvider, MetadataWriter, SqliteMetadataProvider,
    SqliteMetadataWriter, register_ducklake_functions,
};

/// The `(id, val)` schema used throughout.
fn table_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("val", DataType::Int32, false),
    ]))
}

fn object_store() -> Arc<dyn object_store::ObjectStore> {
    Arc::new(LocalFileSystem::new())
}

/// A writable SQLite-backed catalog + data dir in `temp_dir`.
async fn make_writer(temp_dir: &TempDir) -> SqliteMetadataWriter {
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

/// Seed a single data file `t(id, val)` from the given rows via the low-level
/// writer (deterministic file layout: `row_id_start = 0`, positions `0..n`).
async fn seed_table(temp_dir: &TempDir, ids: Vec<i32>, vals: Vec<i32>) {
    let writer = Arc::new(make_writer(temp_dir).await);
    let batch = RecordBatch::try_new(
        table_schema(),
        vec![Arc::new(Int32Array::from(ids)), Arc::new(Int32Array::from(vals))],
    )
    .unwrap();
    DuckLakeTableWriter::new(writer, object_store())
        .unwrap()
        .write_table("main", "t", &[batch])
        .await
        .unwrap();
}

/// Append a second data file to `t`.
async fn append_file(temp_dir: &TempDir, ids: Vec<i32>, vals: Vec<i32>) {
    let db_path = temp_dir.path().join("test.db");
    let conn_str = format!("sqlite:{}?mode=rwc", db_path.display());
    let writer = Arc::new(SqliteMetadataWriter::new(&conn_str).await.unwrap());
    let batch = RecordBatch::try_new(
        table_schema(),
        vec![Arc::new(Int32Array::from(ids)), Arc::new(Int32Array::from(vals))],
    )
    .unwrap();
    DuckLakeTableWriter::new(writer, object_store())
        .unwrap()
        .append_table("main", "t", &[batch])
        .await
        .unwrap();
}

/// A writable SessionContext bound to the seeded catalog.
async fn writable_ctx(temp_dir: &TempDir) -> SessionContext {
    let db_path = temp_dir.path().join("test.db");
    let conn_str = format!("sqlite:{}?mode=rwc", db_path.display());
    let writer = SqliteMetadataWriter::new(&conn_str).await.unwrap();
    let provider = SqliteMetadataProvider::new(&conn_str).await.unwrap();
    let catalog = DuckLakeCatalog::with_writer(Arc::new(provider), Arc::new(writer)).unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));
    ctx
}

/// A read-only SessionContext; when `row_lineage`, tables expose the `rowid`
/// column.
async fn read_ctx(temp_dir: &TempDir, row_lineage: bool) -> SessionContext {
    let db_path = temp_dir.path().join("test.db");
    let conn_str = format!("sqlite:{}", db_path.display());
    let provider = SqliteMetadataProvider::new(&conn_str).await.unwrap();
    let catalog = DuckLakeCatalog::new(provider)
        .unwrap()
        .with_row_lineage(row_lineage);
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));
    ctx
}

/// A SessionContext with the `ducklake_*()` table functions registered.
async fn functions_ctx(temp_dir: &TempDir) -> SessionContext {
    let db_path = temp_dir.path().join("test.db");
    let conn_str = format!("sqlite:{}", db_path.display());
    let provider = SqliteMetadataProvider::new(&conn_str).await.unwrap();
    let provider_arc: Arc<dyn MetadataProvider> =
        Arc::new(SqliteMetadataProvider::new(&conn_str).await.unwrap());
    let catalog = DuckLakeCatalog::new(provider).unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));
    register_ducklake_functions(&ctx, provider_arc);
    ctx
}

/// Run `sql` and return the single `count` (UInt64) it yields (INSERT/UPDATE).
async fn run_dml_count(ctx: &SessionContext, sql: &str) -> u64 {
    let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
    assert_eq!(batches.len(), 1, "DML should yield exactly one batch");
    assert_eq!(batches[0].num_rows(), 1, "DML count batch has one row");
    batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .expect("count column is UInt64")
        .value(0)
}

/// `(id, val)` from `t`, ascending by id, through the full read path.
async fn read_pairs(temp_dir: &TempDir) -> Vec<(i32, i32)> {
    let ctx = read_ctx(temp_dir, false).await;
    let batches = ctx
        .sql("SELECT id, val FROM ducklake.main.t ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let mut rows = Vec::new();
    for b in &batches {
        let ids = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        let vals = b.column(1).as_any().downcast_ref::<Int32Array>().unwrap();
        for i in 0..b.num_rows() {
            rows.push((ids.value(i), vals.value(i)));
        }
    }
    rows
}

/// `(rowid, id, val)` from `t`, ascending by id, via the row-lineage read path.
async fn read_rowid_rows(temp_dir: &TempDir) -> Vec<(i64, i32, i32)> {
    let ctx = read_ctx(temp_dir, true).await;
    let batches = ctx
        .sql("SELECT rowid, id, val FROM ducklake.main.t ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let mut rows = Vec::new();
    for b in &batches {
        let rowids = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        let ids = b.column(1).as_any().downcast_ref::<Int32Array>().unwrap();
        let vals = b.column(2).as_any().downcast_ref::<Int32Array>().unwrap();
        for i in 0..b.num_rows() {
            assert!(!rowids.is_null(i), "rowid must not be NULL after UPDATE");
            rows.push((rowids.value(i), ids.value(i), vals.value(i)));
        }
    }
    rows
}

async fn snapshot_count(temp_dir: &TempDir) -> i64 {
    let db_path = temp_dir.path().join("test.db");
    let conn_str = format!("sqlite:{}", db_path.display());
    let pool = SqlitePool::connect(&conn_str).await.unwrap();
    sqlx::query_scalar("SELECT COUNT(*) FROM ducklake_snapshot")
        .fetch_one(&pool)
        .await
        .unwrap()
}

async fn max_snapshot(temp_dir: &TempDir) -> i64 {
    let db_path = temp_dir.path().join("test.db");
    let conn_str = format!("sqlite:{}", db_path.display());
    let pool = SqlitePool::connect(&conn_str).await.unwrap();
    sqlx::query_scalar("SELECT COALESCE(MAX(snapshot_id), 0) FROM ducklake_snapshot")
        .fetch_one(&pool)
        .await
        .unwrap()
}

async fn live_data_file_count(temp_dir: &TempDir) -> i64 {
    let db_path = temp_dir.path().join("test.db");
    let conn_str = format!("sqlite:{}", db_path.display());
    let pool = SqlitePool::connect(&conn_str).await.unwrap();
    sqlx::query_scalar("SELECT COUNT(*) FROM ducklake_data_file WHERE end_snapshot IS NULL")
        .fetch_one(&pool)
        .await
        .unwrap()
}

// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn update_sets_value_where_id() {
    let temp_dir = TempDir::new().unwrap();
    seed_table(&temp_dir, vec![1, 2, 3, 4], vec![10, 20, 30, 40]).await;
    assert_eq!(
        read_pairs(&temp_dir).await,
        vec![(1, 10), (2, 20), (3, 30), (4, 40)],
        "baseline"
    );

    let ctx = writable_ctx(&temp_dir).await;
    let count = run_dml_count(&ctx, "UPDATE ducklake.main.t SET val = 200 WHERE id = 2").await;
    assert_eq!(count, 1, "one row matched id = 2");

    let rows = read_pairs(&temp_dir).await;
    assert_eq!(
        rows,
        vec![(1, 10), (2, 200), (3, 30), (4, 40)],
        "id=2 gets the new value; others unchanged"
    );
    assert_eq!(rows.len(), 4, "row count is unchanged by UPDATE");
}

#[tokio::test(flavor = "multi_thread")]
async fn update_expression_referencing_column() {
    let temp_dir = TempDir::new().unwrap();
    seed_table(&temp_dir, vec![1, 2, 3], vec![10, 20, 30]).await;

    let ctx = writable_ctx(&temp_dir).await;
    // Assignment expression references the column being updated.
    let count = run_dml_count(
        &ctx,
        "UPDATE ducklake.main.t SET val = val + 5 WHERE id >= 2",
    )
    .await;
    assert_eq!(count, 2, "ids 2 and 3 match");

    assert_eq!(
        read_pairs(&temp_dir).await,
        vec![(1, 10), (2, 25), (3, 35)],
        "matched rows get val + 5"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn update_multi_row_multi_file_is_one_snapshot() {
    let temp_dir = TempDir::new().unwrap();
    // Two data files: A=(1,10),(2,20); B=(3,30),(4,40).
    seed_table(&temp_dir, vec![1, 2], vec![10, 20]).await;
    append_file(&temp_dir, vec![3, 4], vec![30, 40]).await;
    assert_eq!(
        read_pairs(&temp_dir).await,
        vec![(1, 10), (2, 20), (3, 30), (4, 40)],
        "baseline across two files"
    );
    assert_eq!(
        live_data_file_count(&temp_dir).await,
        2,
        "two live data files"
    );

    let before = snapshot_count(&temp_dir).await;

    // Update one row from each file in a single statement.
    let ctx = writable_ctx(&temp_dir).await;
    let count = run_dml_count(
        &ctx,
        "UPDATE ducklake.main.t SET val = val + 1 WHERE id IN (2, 3)",
    )
    .await;
    assert_eq!(count, 2, "one row from each file matched");

    assert_eq!(
        read_pairs(&temp_dir).await,
        vec![(1, 10), (2, 21), (3, 31), (4, 40)],
        "one row updated in each file; the rest unchanged"
    );

    let after = snapshot_count(&temp_dir).await;
    assert_eq!(
        after - before,
        1,
        "the whole multi-file update is exactly ONE new snapshot (atomic)"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn update_preserves_rowid_lineage() {
    let temp_dir = TempDir::new().unwrap();
    // One file: rowids 0,1,2,3 for ids 1,2,3,4.
    seed_table(&temp_dir, vec![1, 2, 3, 4], vec![10, 20, 30, 40]).await;
    assert_eq!(
        read_rowid_rows(&temp_dir).await,
        vec![(0, 1, 10), (1, 2, 20), (2, 3, 30), (3, 4, 40)],
        "baseline rowids"
    );

    let ctx = writable_ctx(&temp_dir).await;
    let count = run_dml_count(
        &ctx,
        "UPDATE ducklake.main.t SET val = val * 10 WHERE id IN (2, 4)",
    )
    .await;
    assert_eq!(count, 2);

    // The updated rows keep their ORIGINAL rowids (1 and 3), proving lineage
    // survives the file rewrite via the embedded row-id column.
    assert_eq!(
        read_rowid_rows(&temp_dir).await,
        vec![(0, 1, 10), (1, 2, 200), (2, 3, 30), (3, 4, 400)],
        "rowids 1 and 3 are retained by their updated rows"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn update_all_rows_without_where() {
    let temp_dir = TempDir::new().unwrap();
    seed_table(&temp_dir, vec![1, 2, 3], vec![10, 20, 30]).await;

    let ctx = writable_ctx(&temp_dir).await;
    let count = run_dml_count(&ctx, "UPDATE ducklake.main.t SET val = 99").await;
    assert_eq!(count, 3, "no WHERE updates every row");

    assert_eq!(
        read_pairs(&temp_dir).await,
        vec![(1, 99), (2, 99), (3, 99)],
        "all rows set to 99"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn update_change_feed_emits_preimage_and_postimage() {
    let temp_dir = TempDir::new().unwrap();
    seed_table(&temp_dir, vec![1, 2, 3], vec![10, 20, 30]).await;

    let before = max_snapshot(&temp_dir).await;
    let ctx = writable_ctx(&temp_dir).await;
    let count = run_dml_count(&ctx, "UPDATE ducklake.main.t SET val = 200 WHERE id = 2").await;
    assert_eq!(count, 1);
    let after = max_snapshot(&temp_dir).await;
    assert_eq!(after - before, 1, "one update snapshot");

    // The change feed over the update snapshot pairs the delete + insert that
    // share rowid 1 into update_preimage (old) + update_postimage (new).
    let fctx = functions_ctx(&temp_dir).await;
    let sql = format!(
        "SELECT id, val, change_type \
         FROM ducklake_table_changes('main.t', {before}, {after}) \
         ORDER BY change_type, id"
    );
    let batches = fctx.sql(&sql).await.unwrap().collect().await.unwrap();

    let mut got: Vec<(i32, i32, String)> = Vec::new();
    for b in &batches {
        let ids = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        let vals = b.column(1).as_any().downcast_ref::<Int32Array>().unwrap();
        let ct = b.column(2);
        let ct = ct
            .as_any()
            .downcast_ref::<arrow::array::StringArray>()
            .map(|a| {
                (0..a.len())
                    .map(|i| a.value(i).to_string())
                    .collect::<Vec<_>>()
            })
            .or_else(|| {
                ct.as_any()
                    .downcast_ref::<arrow::array::StringViewArray>()
                    .map(|a| {
                        (0..a.len())
                            .map(|i| a.value(i).to_string())
                            .collect::<Vec<_>>()
                    })
            })
            .expect("change_type is a string column");
        for (i, ct_val) in ct.iter().enumerate() {
            got.push((ids.value(i), vals.value(i), ct_val.clone()));
        }
    }

    assert_eq!(
        got,
        vec![(2, 200, "update_postimage".to_string()), (2, 20, "update_preimage".to_string()),],
        "the update surfaces as a preimage (old) + postimage (new) pair"
    );
}

/// A pure delete (no matching insert) must NOT surface in `ducklake_table_changes`
/// as an update: the correlation only pairs a delete + insert sharing a rowid.
#[tokio::test(flavor = "multi_thread")]
async fn update_change_feed_ignores_unrelated_inserts() {
    let temp_dir = TempDir::new().unwrap();
    seed_table(&temp_dir, vec![1, 2], vec![10, 20]).await;

    let before = max_snapshot(&temp_dir).await;
    let ctx = writable_ctx(&temp_dir).await;
    run_dml_count(&ctx, "UPDATE ducklake.main.t SET val = 99 WHERE id = 1").await;
    let after = max_snapshot(&temp_dir).await;

    // Exactly one preimage + one postimage for the single updated row; no
    // spurious plain insert/delete rows.
    let fctx = functions_ctx(&temp_dir).await;
    let sql = format!(
        "SELECT change_type, COUNT(*) AS n \
         FROM ducklake_table_changes('main.t', {before}, {after}) \
         GROUP BY change_type ORDER BY change_type"
    );
    let batches = fctx.sql(&sql).await.unwrap().collect().await.unwrap();
    let mut counts: Vec<(String, i64)> = Vec::new();
    for b in &batches {
        let ct = b.column(0);
        let ct = ct
            .as_any()
            .downcast_ref::<arrow::array::StringArray>()
            .map(|a| {
                (0..a.len())
                    .map(|i| a.value(i).to_string())
                    .collect::<Vec<_>>()
            })
            .or_else(|| {
                ct.as_any()
                    .downcast_ref::<arrow::array::StringViewArray>()
                    .map(|a| {
                        (0..a.len())
                            .map(|i| a.value(i).to_string())
                            .collect::<Vec<_>>()
                    })
            })
            .unwrap();
        let n = b.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
        for (i, ct_val) in ct.iter().enumerate() {
            counts.push((ct_val.clone(), n.value(i)));
        }
    }
    assert_eq!(
        counts,
        vec![("update_postimage".to_string(), 1), ("update_preimage".to_string(), 1),],
        "only the correlated pair is surfaced"
    );
}

/// A second UPDATE in the SAME session that re-touches a file the first UPDATE
/// modified must abort with a clear conflict (the catalog pins its snapshot to
/// the pre-update generation) — and must NOT corrupt: the first update's result
/// is preserved and the second row is left unchanged.
#[tokio::test(flavor = "multi_thread")]
async fn update_second_in_session_conflicts_without_corruption() {
    let temp_dir = TempDir::new().unwrap();
    seed_table(&temp_dir, vec![1, 2, 3, 4], vec![10, 20, 30, 40]).await;

    let ctx = writable_ctx(&temp_dir).await;
    assert_eq!(
        run_dml_count(&ctx, "UPDATE ducklake.main.t SET val = 200 WHERE id = 2").await,
        1
    );
    assert_eq!(
        read_pairs(&temp_dir).await,
        vec![(1, 10), (2, 200), (3, 30), (4, 40)]
    );

    // Second UPDATE (same session, same file) — aborts on the commit CAS.
    let err = ctx
        .sql("UPDATE ducklake.main.t SET val = 300 WHERE id = 3")
        .await
        .unwrap()
        .collect()
        .await
        .expect_err("second in-session UPDATE must conflict, not silently corrupt");
    let msg = err.to_string();
    assert!(
        msg.contains("Re-open the catalog") && msg.contains("THIS session"),
        "conflict message must explain the pinned-snapshot cause, got: {msg}"
    );

    // Clean abort: id=2 stays updated, id=3 unchanged, no row lost/duplicated.
    assert_eq!(
        read_pairs(&temp_dir).await,
        vec![(1, 10), (2, 200), (3, 30), (4, 40)],
        "aborted UPDATE leaves the first update intact and id=3 unchanged"
    );
}

/// CDC over a range that mixes an unrelated plain INSERT with an UPDATE: the
/// insert must surface as `insert`, and the update as a preimage/postimage pair.
/// Exercises the correlated path together with a non-embedded insert file (the
/// `update_change_feed_ignores_unrelated_inserts` test does not actually insert).
#[tokio::test(flavor = "multi_thread")]
async fn update_change_feed_mixed_insert_and_update() {
    let temp_dir = TempDir::new().unwrap();
    seed_table(&temp_dir, vec![1, 2], vec![10, 20]).await;
    let before = max_snapshot(&temp_dir).await;

    // Snapshot +1: a plain INSERT (no embedded rowid).
    run_dml_count(
        &writable_ctx(&temp_dir).await,
        "INSERT INTO ducklake.main.t VALUES (9, 90)",
    )
    .await;
    // Snapshot +2: an UPDATE.
    run_dml_count(
        &writable_ctx(&temp_dir).await,
        "UPDATE ducklake.main.t SET val = 100 WHERE id = 1",
    )
    .await;
    let after = max_snapshot(&temp_dir).await;

    let fctx = functions_ctx(&temp_dir).await;
    let sql = format!(
        "SELECT id, val, change_type FROM ducklake_table_changes('main.t', {before}, {after}) \
         ORDER BY change_type, id"
    );
    let batches = fctx.sql(&sql).await.unwrap().collect().await.unwrap();
    let mut got: Vec<(i32, i32, String)> = Vec::new();
    for b in &batches {
        let ids = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        let vals = b.column(1).as_any().downcast_ref::<Int32Array>().unwrap();
        let ct = b.column(2);
        let cts: Vec<String> = ct
            .as_any()
            .downcast_ref::<arrow::array::StringArray>()
            .map(|a| (0..a.len()).map(|i| a.value(i).to_string()).collect())
            .or_else(|| {
                ct.as_any()
                    .downcast_ref::<arrow::array::StringViewArray>()
                    .map(|a| (0..a.len()).map(|i| a.value(i).to_string()).collect())
            })
            .expect("change_type is a string column");
        for (i, c) in cts.iter().enumerate() {
            got.push((ids.value(i), vals.value(i), c.clone()));
        }
    }
    assert_eq!(
        got,
        vec![
            (9, 90, "insert".to_string()),
            (1, 100, "update_postimage".to_string()),
            (1, 10, "update_preimage".to_string()),
        ],
        "unrelated insert stays an insert; the update is a preimage/postimage pair"
    );
}

/// CDC over an INSERT-only range (no UPDATE/DELETE): every added row is an
/// `insert` and nothing is reclassified. Guards the fast path that must NOT do
/// the correlated delete+insert probing when the range applied no deletes.
#[tokio::test(flavor = "multi_thread")]
async fn change_feed_insert_only_range_is_all_inserts() {
    let temp_dir = TempDir::new().unwrap();
    seed_table(&temp_dir, vec![1, 2], vec![10, 20]).await;
    let before = max_snapshot(&temp_dir).await;
    run_dml_count(
        &writable_ctx(&temp_dir).await,
        "INSERT INTO ducklake.main.t VALUES (3, 30)",
    )
    .await;
    let after = max_snapshot(&temp_dir).await;

    let fctx = functions_ctx(&temp_dir).await;
    let sql = format!(
        "SELECT change_type, COUNT(*) AS n FROM ducklake_table_changes('main.t', {before}, {after}) \
         GROUP BY change_type ORDER BY change_type"
    );
    let batches = fctx.sql(&sql).await.unwrap().collect().await.unwrap();
    let mut counts: Vec<(String, i64)> = Vec::new();
    for b in &batches {
        let ct = b.column(0);
        let cts: Vec<String> = ct
            .as_any()
            .downcast_ref::<arrow::array::StringArray>()
            .map(|a| (0..a.len()).map(|i| a.value(i).to_string()).collect())
            .or_else(|| {
                ct.as_any()
                    .downcast_ref::<arrow::array::StringViewArray>()
                    .map(|a| (0..a.len()).map(|i| a.value(i).to_string()).collect())
            })
            .unwrap();
        let n = b.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
        for (i, c) in cts.iter().enumerate() {
            counts.push((c.clone(), n.value(i)));
        }
    }
    assert_eq!(
        counts,
        vec![("insert".to_string(), 1)],
        "insert-only range yields only inserts"
    );
}
