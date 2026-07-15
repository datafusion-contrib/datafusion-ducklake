//! Integration tests for reading DuckLake **data inlining** on the SQLite
//! backend.
//!
//! DuckDB's ducklake extension stores small INSERTs directly in the catalog
//! database (in `ducklake_inlined_data_<tid>_<sv>` tables registered in
//! `ducklake_inlined_data_tables`) instead of Parquet. A reader that only scans
//! `ducklake_data_file` silently undercounts. These tests hand-craft inlined
//! tables exactly as DuckDB would and assert that `SELECT` / `COUNT(*)` include
//! the inlined rows, that inlined-row deletes (`end_snapshot`) are respected, and
//! that time travel is correct — while catalogs with no inlined data are
//! unaffected.

#![cfg(all(feature = "write-sqlite", feature = "metadata-sqlite"))]

use std::sync::Arc;

use arrow::array::Int32Array;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::prelude::*;
use object_store::local::LocalFileSystem;
use sqlx::sqlite::SqlitePool;
use tempfile::TempDir;

use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeTableWriter, MetadataWriter, SqliteMetadataProvider,
    SqliteMetadataWriter,
};

fn table_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("val", DataType::Int32, false),
    ]))
}

fn object_store() -> Arc<dyn object_store::ObjectStore> {
    Arc::new(LocalFileSystem::new())
}

fn rw_url(t: &TempDir) -> String {
    format!("sqlite:{}?mode=rwc", t.path().join("test.db").display())
}
fn ro_url(t: &TempDir) -> String {
    format!("sqlite:{}", t.path().join("test.db").display())
}

fn batch(ids: Vec<i32>, vals: Vec<i32>) -> RecordBatch {
    RecordBatch::try_new(
        table_schema(),
        vec![Arc::new(Int32Array::from(ids)), Arc::new(Int32Array::from(vals))],
    )
    .unwrap()
}

async fn make_writer(t: &TempDir) -> SqliteMetadataWriter {
    let data = t.path().join("data");
    std::fs::create_dir_all(&data).unwrap();
    let w = SqliteMetadataWriter::new_with_init(&rw_url(t))
        .await
        .unwrap();
    w.set_data_path(data.to_str().unwrap()).unwrap();
    w
}

/// `(id, val)` from `main.t`, ascending, as of `snapshot` (or latest).
async fn read_rows(t: &TempDir, snapshot: Option<i64>) -> Vec<(i32, i32)> {
    let provider = SqliteMetadataProvider::new(&ro_url(t)).await.unwrap();
    let catalog = match snapshot {
        Some(s) => DuckLakeCatalog::with_snapshot(Arc::new(provider), s).unwrap(),
        None => DuckLakeCatalog::new(provider).unwrap(),
    };
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));
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

/// Create the inlining registry + a physical inlined-insert table for `t`, laid
/// out exactly as DuckDB's extension would: `ducklake_inlined_data_<tid>_1(
/// row_id, begin_snapshot, end_snapshot, id, val)`.
async fn seed_inlined(
    pool: &SqlitePool,
    table_id: i64,
    rows: &[(i64, i64, Option<i64>, i32, i32)],
) {
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS ducklake_inlined_data_tables
             (table_id BIGINT, table_name VARCHAR, schema_version BIGINT)",
    )
    .execute(pool)
    .await
    .unwrap();
    let phys = format!("ducklake_inlined_data_{table_id}_1");
    sqlx::query(&format!(
        "CREATE TABLE IF NOT EXISTS {phys}
             (row_id BIGINT, begin_snapshot BIGINT, end_snapshot BIGINT, id INTEGER, val INTEGER)"
    ))
    .execute(pool)
    .await
    .unwrap();
    sqlx::query(
        "INSERT INTO ducklake_inlined_data_tables (table_id, table_name, schema_version)
         VALUES (?, ?, 1)",
    )
    .bind(table_id)
    .bind(&phys)
    .execute(pool)
    .await
    .unwrap();
    for (row_id, begin, end, id, val) in rows {
        sqlx::query(&format!(
            "INSERT INTO {phys} (row_id, begin_snapshot, end_snapshot, id, val) VALUES (?,?,?,?,?)"
        ))
        .bind(row_id)
        .bind(begin)
        .bind(*end)
        .bind(id)
        .bind(val)
        .execute(pool)
        .await
        .unwrap();
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn inlined_rows_are_unioned_into_reads_with_visibility_and_time_travel() {
    let t = TempDir::new().unwrap();
    // Parquet-backed rows: file1 at snapshot 1, file2 at snapshot 2.
    let w = Arc::new(make_writer(&t).await);
    DuckLakeTableWriter::new(w, object_store())
        .unwrap()
        .write_table("main", "t", &[batch(vec![1, 2], vec![10, 20])])
        .await
        .unwrap();
    let w2 = Arc::new(SqliteMetadataWriter::new(&rw_url(&t)).await.unwrap());
    DuckLakeTableWriter::new(w2, object_store())
        .unwrap()
        .append_table("main", "t", &[batch(vec![7, 8], vec![70, 80])])
        .await
        .unwrap();

    let pool = SqlitePool::connect(&rw_url(&t)).await.unwrap();
    let table_id: i64 = sqlx::query_scalar("SELECT table_id FROM ducklake_table LIMIT 1")
        .fetch_one(&pool)
        .await
        .unwrap();

    // Baseline (no inlined data yet): only the Parquet rows.
    assert_eq!(
        read_rows(&t, None).await,
        vec![(1, 10), (2, 20), (7, 70), (8, 80)]
    );

    // Inlined rows (as DuckDB would store them):
    //  - (3,30): live from snapshot 1 (end_snapshot NULL)
    //  - (5,50): inserted at snapshot 1, DELETED at snapshot 2 (end_snapshot = 2)
    seed_inlined(
        &pool,
        table_id,
        &[(100, 1, None, 3, 30), (101, 1, Some(2), 5, 50)],
    )
    .await;

    // At the latest snapshot (2): Parquet rows + the live inlined (3,30); the
    // inlined (5,50) is excluded because it was deleted at snapshot 2.
    assert_eq!(
        read_rows(&t, None).await,
        vec![(1, 10), (2, 20), (3, 30), (7, 70), (8, 80)],
        "inlined live row included; deleted inlined row excluded"
    );

    // Time travel to snapshot 1: only file1's Parquet rows, plus BOTH inlined
    // rows (neither deleted yet at snapshot 1; file2 not yet visible).
    assert_eq!(
        read_rows(&t, Some(1)).await,
        vec![(1, 10), (2, 20), (3, 30), (5, 50)],
        "time travel sees the inlined rows as of that snapshot"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn catalog_without_inlining_is_unaffected() {
    let t = TempDir::new().unwrap();
    let w = Arc::new(make_writer(&t).await);
    DuckLakeTableWriter::new(w, object_store())
        .unwrap()
        .write_table("main", "t", &[batch(vec![1, 2, 3], vec![10, 20, 30])])
        .await
        .unwrap();
    // No ducklake_inlined_data_tables exists -> get_inlined_data returns empty,
    // reads are exactly the Parquet rows.
    assert_eq!(read_rows(&t, None).await, vec![(1, 10), (2, 20), (3, 30)]);
}
