//! Integration tests for SQL `UPDATE` support (SQLite metadata backend).
//!
//! Exercises `UPDATE t SET col = expr [, ...] [WHERE ...]` end-to-end through
//! DataFusion's SQL interface against a writable DuckLake catalog: affected-row
//! count, the resulting values, atomicity (one snapshot), rowid-lineage
//! preservation across the file rewrite, the change feed (preimage/postimage),
//! and update-all.
//!
//! The final group covers the same path on a PARTITIONED table, where the rewrite
//! re-derives each row's partition from its post-assignment values: a row whose
//! partition key changed moves to its new partition, a rewrite touching several
//! partitions produces one file per partition, and all of it — every appended file
//! plus every positional delete — still commits in ONE snapshot with lineage intact.

#![cfg(all(feature = "write-sqlite", feature = "metadata-sqlite"))]

use std::sync::Arc;

use arrow::array::{Array, Int32Array, Int64Array, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::prelude::*;
use object_store::local::LocalFileSystem;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use sqlx::sqlite::SqlitePool;
use tempfile::TempDir;

use datafusion_ducklake::sort::{NullOrder, SortDirection, SortField};
use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeTableWriter, MetadataProvider, MetadataWriter, SqliteMetadataProvider,
    SqliteMetadataWriter, WriteMode, register_ducklake_functions,
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
async fn update_applies_sort_order() {
    let temp_dir = TempDir::new().unwrap();
    seed_table(&temp_dir, vec![1, 1, 1], vec![30, 10, 20]).await;
    let writer = Arc::new(make_writer(&temp_dir).await);
    let provider = SqliteMetadataProvider::new(&format!(
        "sqlite:{}",
        temp_dir.path().join("test.db").display()
    ))
    .await
    .unwrap();
    let snapshot = provider.get_current_snapshot().unwrap();
    let schema = provider
        .get_schema_by_name("main", snapshot)
        .unwrap()
        .unwrap();
    let table = provider
        .get_table_by_name(schema.schema_id, "t", snapshot)
        .unwrap()
        .unwrap();
    writer
        .set_sort_spec(
            table.table_id,
            &[SortField::column(0, "val", SortDirection::Asc, NullOrder::NullsLast)],
        )
        .unwrap();

    let ctx = writable_ctx(&temp_dir).await;
    assert_eq!(
        run_dml_count(
            &ctx,
            "UPDATE ducklake.main.t SET val = val + 1 WHERE id = 1"
        )
        .await,
        3,
    );

    let provider = SqliteMetadataProvider::new(&format!(
        "sqlite:{}",
        temp_dir.path().join("test.db").display()
    ))
    .await
    .unwrap();
    let snapshot = provider.get_current_snapshot().unwrap();
    let mut files = provider
        .get_table_file_metadata_page(table.table_id, snapshot, None, 16)
        .unwrap();
    files.sort_by_key(|metadata| metadata.file.data_file_id);
    let output = files.last().unwrap();
    let path = temp_dir
        .path()
        .join("data/main/t")
        .join(&output.file.file.path);
    let file = std::fs::File::open(path).unwrap();
    let batch = ParquetRecordBatchReaderBuilder::try_new(file)
        .unwrap()
        .build()
        .unwrap()
        .next()
        .unwrap()
        .unwrap();
    let values = batch
        .column(1)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();

    assert_eq!(files.len(), 2);
    assert_eq!(
        values.values().iter().copied().collect::<Vec<_>>(),
        vec![11, 21, 31],
    );
}

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
    // Bounds are inclusive: start at the first post-`before` snapshot.
    let start = before + 1;
    let fctx = functions_ctx(&temp_dir).await;
    let sql = format!(
        "SELECT id, val, change_type \
         FROM ducklake_table_changes('main.t', {start}, {after}) \
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
    // Bounds are inclusive: start at the first post-`before` snapshot.
    let start = before + 1;
    let fctx = functions_ctx(&temp_dir).await;
    let sql = format!(
        "SELECT change_type, COUNT(*) AS n \
         FROM ducklake_table_changes('main.t', {start}, {after}) \
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

    // Bounds are inclusive: start at the first post-`before` snapshot.
    let start = before + 1;
    let fctx = functions_ctx(&temp_dir).await;
    let sql = format!(
        "SELECT id, val, change_type FROM ducklake_table_changes('main.t', {start}, {after}) \
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

    // Bounds are inclusive: start at the first post-`before` snapshot.
    let start = before + 1;
    let fctx = functions_ctx(&temp_dir).await;
    let sql = format!(
        "SELECT change_type, COUNT(*) AS n FROM ducklake_table_changes('main.t', {start}, {after}) \
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

// ---------------------------------------------------------------------------
// Partitioned UPDATE (row-rewrite session on a partitioned table)
// ---------------------------------------------------------------------------

/// Create `p(region, id, val)` partitioned by `region`, then seed it through SQL
/// `INSERT` so the rows land one file per partition, exactly as a real ingest does.
async fn seed_partitioned_table(temp_dir: &TempDir) {
    use datafusion_ducklake::partition::PartitionTransform;

    let writer = make_writer(temp_dir).await;
    let cols = vec![
        datafusion_ducklake::ColumnDef::from_arrow("region", &DataType::Utf8, true).unwrap(),
        datafusion_ducklake::ColumnDef::from_arrow("id", &DataType::Int32, false).unwrap(),
        datafusion_ducklake::ColumnDef::from_arrow("val", &DataType::Int32, false).unwrap(),
    ];
    let s = writer
        .begin_write_transaction("main", "p", &cols, WriteMode::Replace)
        .unwrap();
    writer
        .publish_snapshot(
            s.table_id,
            "main",
            "p",
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

    let ctx = writable_ctx(temp_dir).await;
    ctx.sql(
        "INSERT INTO ducklake.main.p SELECT * FROM (VALUES \
            ('us', 1, 10), ('us', 2, 20), ('eu', 3, 30), ('eu', 4, 40)) AS t(region, id, val)",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();
}

/// `(region, id, val)` from `p`, ascending by id.
async fn read_partitioned_rows(temp_dir: &TempDir) -> Vec<(String, i32, i32)> {
    let ctx = read_ctx(temp_dir, false).await;
    let batches = ctx
        .sql("SELECT region, id, val FROM ducklake.main.p ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let mut rows = Vec::new();
    for b in &batches {
        let region = b
            .column(0)
            .as_any()
            .downcast_ref::<arrow::array::StringViewArray>()
            .unwrap();
        let ids = b.column(1).as_any().downcast_ref::<Int32Array>().unwrap();
        let vals = b.column(2).as_any().downcast_ref::<Int32Array>().unwrap();
        for i in 0..b.num_rows() {
            rows.push((region.value(i).to_string(), ids.value(i), vals.value(i)));
        }
    }
    rows
}

/// `(rowid, region, id)` from `p`, ascending by id, via the row-lineage read path.
async fn read_partitioned_rowids(temp_dir: &TempDir) -> Vec<(i64, String, i32)> {
    let ctx = read_ctx(temp_dir, true).await;
    let batches = ctx
        .sql("SELECT rowid, region, id FROM ducklake.main.p ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let mut rows = Vec::new();
    for b in &batches {
        let rowids = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        let region = b
            .column(1)
            .as_any()
            .downcast_ref::<arrow::array::StringViewArray>()
            .unwrap();
        let ids = b.column(2).as_any().downcast_ref::<Int32Array>().unwrap();
        for i in 0..b.num_rows() {
            assert!(!rowids.is_null(i), "rowid must not be NULL after UPDATE");
            rows.push((rowids.value(i), region.value(i).to_string(), ids.value(i)));
        }
    }
    rows
}

/// Each live data file of `p` as `(data_file_id, begin_snapshot, partition_value)`.
async fn partitioned_files(temp_dir: &TempDir) -> Vec<(i64, i64, Option<String>)> {
    let db_path = temp_dir.path().join("test.db");
    let conn_str = format!("sqlite:{}", db_path.display());
    let pool = SqlitePool::connect(&conn_str).await.unwrap();
    sqlx::query_as(
        "SELECT f.data_file_id, f.begin_snapshot,
                (SELECT v.partition_value FROM ducklake_file_partition_value v
                 WHERE v.data_file_id = f.data_file_id AND v.partition_key_index = 0)
         FROM ducklake_data_file f
         WHERE f.end_snapshot IS NULL
         ORDER BY f.data_file_id",
    )
    .fetch_all(&pool)
    .await
    .unwrap()
}

/// An `UPDATE` that CHANGES a row's partition-key value must re-derive that row's
/// partition from its NEW values, so the rewritten version lands in its new
/// partition rather than inheriting the source file's — and must do so in ONE
/// snapshot, with the row's lineage rowid intact.
#[tokio::test(flavor = "multi_thread")]
async fn partitioned_update_moves_row_to_its_new_partition() {
    let temp_dir = TempDir::new().unwrap();
    seed_partitioned_table(&temp_dir).await;

    let before = read_partitioned_rowids(&temp_dir).await;
    let rowid_of_1 = before.iter().find(|(_, _, id)| *id == 1).unwrap().0;
    let snapshots_before = snapshot_count(&temp_dir).await;

    let ctx = writable_ctx(&temp_dir).await;
    let updated = run_dml_count(
        &ctx,
        "UPDATE ducklake.main.p SET region = 'apac' WHERE id = 1",
    )
    .await;
    assert_eq!(updated, 1);

    assert_eq!(
        read_partitioned_rows(&temp_dir).await,
        vec![
            ("apac".to_string(), 1, 10),
            ("us".to_string(), 2, 20),
            ("eu".to_string(), 3, 30),
            ("eu".to_string(), 4, 40),
        ],
        "row 1 moved to 'apac'; the rest are untouched"
    );

    // ONE snapshot for the whole mutation.
    assert_eq!(
        snapshot_count(&temp_dir).await,
        snapshots_before + 1,
        "a partitioned UPDATE is one snapshot"
    );

    // The rewritten row kept its ORIGINAL rowid.
    let after = read_partitioned_rowids(&temp_dir).await;
    assert_eq!(
        after.iter().find(|(_, _, id)| *id == 1).unwrap().0,
        rowid_of_1,
        "the moved row keeps its lineage rowid"
    );

    // Its new file is stamped with the NEW partition value, not the source file's.
    let head = max_snapshot(&temp_dir).await;
    let appended: Vec<_> = partitioned_files(&temp_dir)
        .await
        .into_iter()
        .filter(|(_, begin, _)| *begin == head)
        .collect();
    assert_eq!(appended.len(), 1, "one rewritten file: {appended:?}");
    assert_eq!(
        appended[0].2,
        Some("apac".to_string()),
        "the rewritten row's file carries its NEW partition value"
    );
}

/// An `UPDATE` whose rewritten rows span SEVERAL partitions commits one file per
/// partition, all in ONE snapshot, with every row's lineage preserved.
#[tokio::test(flavor = "multi_thread")]
async fn partitioned_update_spanning_partitions_is_one_snapshot() {
    let temp_dir = TempDir::new().unwrap();
    seed_partitioned_table(&temp_dir).await;

    let before = read_partitioned_rowids(&temp_dir).await;
    let snapshots_before = snapshot_count(&temp_dir).await;

    // Touches one row in 'us' and one in 'eu': the rewritten rows stay in their own
    // (different) partitions, so the appended side is two files.
    let ctx = writable_ctx(&temp_dir).await;
    let updated = run_dml_count(
        &ctx,
        "UPDATE ducklake.main.p SET val = val * 100 WHERE id = 1 OR id = 3",
    )
    .await;
    assert_eq!(updated, 2);

    assert_eq!(
        read_partitioned_rows(&temp_dir).await,
        vec![
            ("us".to_string(), 1, 1000),
            ("us".to_string(), 2, 20),
            ("eu".to_string(), 3, 3000),
            ("eu".to_string(), 4, 40),
        ],
        "one row updated in each partition"
    );
    assert_eq!(
        snapshot_count(&temp_dir).await,
        snapshots_before + 1,
        "a multi-partition rewrite is still one snapshot"
    );

    // Two appended files, one per touched partition, both on the committed snapshot.
    let head = max_snapshot(&temp_dir).await;
    let mut appended: Vec<Option<String>> = partitioned_files(&temp_dir)
        .await
        .into_iter()
        .filter(|(_, begin, _)| *begin == head)
        .map(|(_, _, value)| value)
        .collect();
    appended.sort();
    assert_eq!(
        appended,
        vec![Some("eu".to_string()), Some("us".to_string())],
        "one rewritten file per touched partition, each with its own value"
    );

    // The delete side lands on that SAME snapshot: one delete file per source
    // partition file whose rows were superseded. N appends + M deletes, one commit.
    let db_path = temp_dir.path().join("test.db");
    let conn_str = format!("sqlite:{}", db_path.display());
    let pool = SqlitePool::connect(&conn_str).await.unwrap();
    let delete_snapshots: Vec<i64> = sqlx::query_scalar(
        "SELECT begin_snapshot FROM ducklake_delete_file WHERE end_snapshot IS NULL",
    )
    .fetch_all(&pool)
    .await
    .unwrap();
    assert_eq!(
        delete_snapshots.len(),
        2,
        "one delete file per superseded source file"
    );
    assert!(
        delete_snapshots.iter().all(|snap| *snap == head),
        "the deletes share the appended files' snapshot: {delete_snapshots:?}"
    );

    // Lineage: every row's rowid is unchanged by the rewrite.
    assert_eq!(
        read_partitioned_rowids(&temp_dir).await,
        before,
        "rowids map back across a multi-partition rewrite"
    );
}

/// A partition spec built from a CALENDAR TRANSFORM (not identity) must behave the
/// same: an `UPDATE` that changes the underlying timestamp re-derives the row's
/// partition through the transform and moves the row, in one snapshot, with lineage
/// intact.
///
/// This case is called out because the reference implementation, at the DuckDB
/// version this crate links, aborts an equivalent partition-moving `UPDATE` with an
/// internal error when the spec contains a transform (its update sink hands the
/// transformed key — an integer — to a code path expecting the raw string). This crate
/// derives partition values from the rewritten batch itself, so it is unaffected; the
/// test exists to keep it that way.
#[tokio::test(flavor = "multi_thread")]
async fn partitioned_update_with_calendar_transform_moves_row() {
    use arrow::array::TimestampMicrosecondArray;
    use arrow::datatypes::TimeUnit;
    use datafusion_ducklake::partition::PartitionTransform;

    let temp_dir = TempDir::new().unwrap();
    let ts_type = DataType::Timestamp(TimeUnit::Microsecond, None);
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("ts", ts_type.clone(), true),
    ]));

    // Create `e(id, ts)` partitioned by `day(ts)`.
    let writer = make_writer(&temp_dir).await;
    let cols = vec![
        datafusion_ducklake::ColumnDef::from_arrow("id", &DataType::Int32, false).unwrap(),
        datafusion_ducklake::ColumnDef::from_arrow("ts", &ts_type, true).unwrap(),
    ];
    let s = writer
        .begin_write_transaction("main", "e", &cols, WriteMode::Replace)
        .unwrap();
    writer
        .publish_snapshot(
            s.table_id,
            "main",
            "e",
            s.snapshot_id,
            WriteMode::Replace,
            s.base_snapshot_id,
            &cols,
            &s.column_ids,
        )
        .unwrap();
    writer
        .set_partition_spec(s.table_id, &[("ts".to_string(), PartitionTransform::Day)])
        .unwrap();

    // Two rows on day 5, one on day 6 -> two partitions.
    let day5: i64 = 1_770_249_600_000_000; // 2026-02-05
    let day6: i64 = 1_770_336_000_000_000; // 2026-02-06
    let seed = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(TimestampMicrosecondArray::from(vec![day5, day5, day6])),
        ],
    )
    .unwrap();
    let seed_writer = Arc::new(
        SqliteMetadataWriter::new(&format!(
            "sqlite:{}?mode=rwc",
            temp_dir.path().join("test.db").display()
        ))
        .await
        .unwrap(),
    );
    DuckLakeTableWriter::new(seed_writer, object_store())
        .unwrap()
        .append_table("main", "e", &[seed])
        .await
        .unwrap();

    let db_path = temp_dir.path().join("test.db");
    let conn_str = format!("sqlite:{}", db_path.display());
    let pool = SqlitePool::connect(&conn_str).await.unwrap();
    let seeded_partitions: Vec<Option<String>> = sqlx::query_scalar(
        "SELECT v.partition_value FROM ducklake_file_partition_value v
         JOIN ducklake_data_file f ON f.data_file_id = v.data_file_id
         WHERE f.end_snapshot IS NULL ORDER BY v.partition_value",
    )
    .fetch_all(&pool)
    .await
    .unwrap();
    assert_eq!(
        seeded_partitions,
        vec![Some("5".to_string()), Some("6".to_string())],
        "the seed splits across day-5 and day-6 partitions"
    );

    // Lineage baseline.
    let lineage_ctx = read_ctx(&temp_dir, true).await;
    let before = lineage_ctx
        .sql("SELECT rowid, id FROM ducklake.main.e ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let rowid_of_1 = before[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    let snapshots_before = snapshot_count(&temp_dir).await;

    // Move row 1 from day 5 to day 9 (a different month, so month/day both change).
    let ctx = writable_ctx(&temp_dir).await;
    let updated = run_dml_count(
        &ctx,
        "UPDATE ducklake.main.e SET ts = arrow_cast(1773014400000000, 'Timestamp(Microsecond, None)') WHERE id = 1",
    )
    .await;
    assert_eq!(updated, 1);

    assert_eq!(
        snapshot_count(&temp_dir).await,
        snapshots_before + 1,
        "one snapshot"
    );

    // The rewritten row's file is stamped with the TRANSFORMED new value (day 9).
    let head = max_snapshot(&temp_dir).await;
    let appended: Vec<Option<String>> = sqlx::query_scalar(
        "SELECT v.partition_value FROM ducklake_file_partition_value v
         JOIN ducklake_data_file f ON f.data_file_id = v.data_file_id
         WHERE f.begin_snapshot = ?",
    )
    .bind(head)
    .fetch_all(&pool)
    .await
    .unwrap();
    assert_eq!(
        appended,
        vec![Some("9".to_string())],
        "the moved row's file carries the transformed NEW partition value"
    );

    // Values and lineage survived.
    let read = read_ctx(&temp_dir, true).await;
    let after = read
        .sql("SELECT rowid, id, date_part('day', ts) FROM ducklake.main.e ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let rowids = after[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(
        rowids.value(0),
        rowid_of_1,
        "the moved row keeps its lineage rowid"
    );
    let days: Vec<i32> = {
        let col = after[0].column(2);
        let a = col.as_any().downcast_ref::<Int32Array>().unwrap();
        (0..a.len()).map(|i| a.value(i)).collect()
    };
    assert_eq!(
        days,
        vec![9, 5, 6],
        "row 1 moved to day 9; rows 2,3 untouched"
    );
}

/// A table that gained its partition spec AFTER data was already written: the source
/// file is unpartitioned, but the rewritten rows must be written under the table's
/// LIVE spec (partition-split, stamped with the live generation) or the commit's
/// partition fence rejects them.
///
/// This is the "adopt partitioning on an existing table, then update it" path, where
/// the source and output partition state legitimately differ.
#[tokio::test(flavor = "multi_thread")]
async fn update_partitions_rewritten_rows_when_spec_was_set_after_data() {
    use datafusion_ducklake::partition::PartitionTransform;

    let temp_dir = TempDir::new().unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("region", DataType::Utf8, true),
        Field::new("id", DataType::Int32, false),
        Field::new("val", DataType::Int32, false),
    ]));

    // Seed ONE unpartitioned file spanning two regions.
    let writer = Arc::new(make_writer(&temp_dir).await);
    let seed = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(arrow::array::StringArray::from(vec!["us", "us", "eu"])),
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(Int32Array::from(vec![10, 20, 30])),
        ],
    )
    .unwrap();
    let seeded = DuckLakeTableWriter::new(writer.clone(), object_store())
        .unwrap()
        .write_table("main", "q", &[seed])
        .await
        .unwrap();

    let db_path = temp_dir.path().join("test.db");
    let conn_str = format!("sqlite:{}", db_path.display());
    let pool = SqlitePool::connect(&conn_str).await.unwrap();
    let seed_partition_id: Option<i64> =
        sqlx::query_scalar("SELECT partition_id FROM ducklake_data_file WHERE begin_snapshot = ?")
            .bind(seeded.snapshot_id)
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(seed_partition_id, None, "the seed file is unpartitioned");

    // NOW adopt a partition spec.
    writer
        .set_partition_spec(
            seeded.table_id,
            &[("region".to_string(), PartitionTransform::Identity)],
        )
        .unwrap();
    let live_partition_id: i64 = sqlx::query_scalar(
        "SELECT partition_id FROM ducklake_partition_info
         WHERE table_id = ? AND end_snapshot IS NULL",
    )
    .bind(seeded.table_id)
    .fetch_one(&pool)
    .await
    .unwrap();

    // Update rows in BOTH regions, so the rewrite spans two output partitions.
    let ctx = writable_ctx(&temp_dir).await;
    let updated = run_dml_count(
        &ctx,
        "UPDATE ducklake.main.q SET val = val + 1 WHERE id = 1 OR id = 3",
    )
    .await;
    assert_eq!(updated, 2);

    // Both rewritten files carry the LIVE partition generation and their own value.
    let head: i64 = sqlx::query_scalar("SELECT MAX(snapshot_id) FROM ducklake_snapshot")
        .fetch_one(&pool)
        .await
        .unwrap();
    let appended: Vec<(Option<i64>, Option<String>)> = sqlx::query_as(
        "SELECT f.partition_id,
                (SELECT v.partition_value FROM ducklake_file_partition_value v
                 WHERE v.data_file_id = f.data_file_id AND v.partition_key_index = 0)
         FROM ducklake_data_file f WHERE f.begin_snapshot = ?
         ORDER BY f.data_file_id",
    )
    .bind(head)
    .fetch_all(&pool)
    .await
    .unwrap();
    assert_eq!(
        appended.len(),
        2,
        "one file per output partition: {appended:?}"
    );
    assert!(
        appended
            .iter()
            .all(|(id, _)| *id == Some(live_partition_id)),
        "rewritten files carry the live partition generation: {appended:?}"
    );
    let mut values: Vec<Option<String>> = appended.into_iter().map(|(_, v)| v).collect();
    values.sort();
    assert_eq!(values, vec![Some("eu".to_string()), Some("us".to_string())],);

    // Values correct, and the untouched row still reads from the unpartitioned file.
    let read = read_ctx(&temp_dir, false).await;
    let rows = read
        .sql("SELECT id, val FROM ducklake.main.q ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let pairs: Vec<(i32, i32)> = rows
        .iter()
        .flat_map(|b| {
            let ids = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
            let vals = b.column(1).as_any().downcast_ref::<Int32Array>().unwrap();
            (0..b.num_rows())
                .map(|i| (ids.value(i), vals.value(i)))
                .collect::<Vec<_>>()
        })
        .collect();
    assert_eq!(pairs, vec![(1, 11), (2, 20), (3, 31)]);
}
