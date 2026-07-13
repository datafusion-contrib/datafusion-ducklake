//! Postgres multicatalog counterpart of `compaction_sqlite_tests.rs`.
//!
//! The multicatalog Postgres write path is a *separate implementation* from the
//! SQLite one (per-catalog head via `ducklake_catalog_snapshot_map`, catalog-scoped
//! lookups, `MulticatalogProvider` as the reader), so compaction is re-validated
//! here end to end: `merge_adjacent_files` produces a partial file with correct
//! results + time travel, and `rewrite_data_files` drops a file's deleted rows.
//! This exercises the multicatalog reader surfacing `begin_snapshot` /
//! `schema_version` / `partial_max` — without which merge would silently no-op.
//! Docker-gated (testcontainers Postgres).

#![cfg(feature = "write-postgres")]

use std::sync::Arc;

use arrow::array::{Array, Int32Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::prelude::*;
use datafusion_ducklake::{
    CompactionResult, DuckLakeCatalog, DuckLakeTable, DuckLakeTableWriter, MergeOptions,
    MetadataProvider, MetadataWriter, MulticatalogManager, MulticatalogProvider,
    PostgresMetadataWriter, RewriteOptions,
};
use object_store::local::LocalFileSystem;
use sqlx::Row;
use sqlx::postgres::{PgPool, PgPoolOptions};
use tempfile::TempDir;
use testcontainers::ContainerAsync;
use testcontainers::runners::AsyncRunner;
use testcontainers_modules::postgres::Postgres;

type ObjStore = Arc<dyn object_store::ObjectStore>;

async fn spin_up_postgres() -> anyhow::Result<(PgPool, ContainerAsync<Postgres>)> {
    let container = Postgres::default().start().await?;
    let port = container.get_host_port_ipv4(5432).await?;
    let conn_str = format!("postgresql://postgres:postgres@127.0.0.1:{}/postgres", port);
    let pool = PgPoolOptions::new()
        .max_connections(5)
        .connect(&conn_str)
        .await?;
    datafusion_ducklake::initialize_multicatalog_schema(&pool).await?;
    Ok((pool, container))
}

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("val", DataType::Int32, false),
    ]))
}

fn batch(ids: Vec<i32>, vals: Vec<i32>) -> RecordBatch {
    RecordBatch::try_new(
        schema(),
        vec![Arc::new(Int32Array::from(ids)), Arc::new(Int32Array::from(vals))],
    )
    .unwrap()
}

async fn writer_for(
    pool: &PgPool,
    cat: i64,
    data_path: &std::path::Path,
) -> Arc<PostgresMetadataWriter> {
    let w = PostgresMetadataWriter::with_pool(pool.clone(), cat)
        .await
        .unwrap();
    w.set_data_path(data_path.to_str().unwrap()).unwrap();
    Arc::new(w)
}

/// Read `(id, val)` from `<cat>.public.t`, optionally as of `snapshot`.
async fn read_rows(pool: &PgPool, cat_name: &str, snapshot: Option<i64>) -> Vec<(i32, i32)> {
    let provider = MulticatalogProvider::with_pool(pool.clone(), cat_name)
        .await
        .unwrap();
    let catalog = match snapshot {
        Some(s) => DuckLakeCatalog::with_snapshot(Arc::new(provider), s).unwrap(),
        None => DuckLakeCatalog::new(provider).unwrap(),
    };
    let ctx = SessionContext::new();
    ctx.register_catalog(cat_name, Arc::new(catalog));
    let batches = ctx
        .sql(&format!(
            "SELECT id, val FROM {cat_name}.public.t ORDER BY id"
        ))
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

/// Live data-file metadata for `<cat>.public.t` at the catalog head, via the
/// multicatalog provider (also verifies it surfaces the compaction fields).
async fn live_files(pool: &PgPool, cat_name: &str) -> Vec<datafusion_ducklake::DuckLakeTableFile> {
    let provider = MulticatalogProvider::with_pool(pool.clone(), cat_name)
        .await
        .unwrap();
    let head = provider.get_current_snapshot().unwrap();
    let sch = provider
        .get_schema_by_name("public", head)
        .unwrap()
        .unwrap();
    let tbl = provider
        .get_table_by_name(sch.schema_id, "t", head)
        .unwrap()
        .unwrap();
    provider
        .get_table_files_for_select(tbl.table_id, head)
        .unwrap()
}

async fn scalar_i64(pool: &PgPool, sql: &str, cat: i64) -> i64 {
    sqlx::query(sql)
        .bind(cat)
        .fetch_one(pool)
        .await
        .unwrap()
        .try_get::<i64, _>(0)
        .unwrap()
}

/// Downcast the writable `<cat>.public.t` provider to a `DuckLakeTable` and run `op`.
async fn with_writable_table<F, Fut>(
    pool: &PgPool,
    cat: i64,
    cat_name: &str,
    data: &std::path::Path,
    op: F,
) -> CompactionResult
where
    F: FnOnce(DuckLakeTable, datafusion::execution::SessionState) -> Fut,
    Fut: std::future::Future<Output = datafusion_ducklake::Result<CompactionResult>>,
{
    let provider = MulticatalogProvider::with_pool(pool.clone(), cat_name)
        .await
        .unwrap();
    let writer = writer_for(pool, cat, data).await;
    let catalog = DuckLakeCatalog::with_writer(Arc::new(provider), writer).unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog(cat_name, Arc::new(catalog));
    let provider = ctx
        .catalog(cat_name)
        .unwrap()
        .schema("public")
        .unwrap()
        .table("t")
        .await
        .unwrap()
        .unwrap();
    let table = (provider.as_ref() as &dyn std::any::Any)
        .downcast_ref::<DuckLakeTable>()
        .expect("provider is a DuckLakeTable")
        .clone();
    op(table, ctx.state()).await.unwrap()
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn merge_adjacent_files_postgres() {
    let (pool, _c) = spin_up_postgres().await.unwrap();
    let tmp = TempDir::new().unwrap();
    let data = tmp.path().join("data");
    std::fs::create_dir_all(&data).unwrap();
    let os: ObjStore = Arc::new(LocalFileSystem::new());
    let cat_name = "cat";
    let cat = MulticatalogManager::new(pool.clone())
        .create_catalog(cat_name)
        .await
        .unwrap();

    // Three INSERTs -> three small data files at three origin snapshots.
    DuckLakeTableWriter::new(writer_for(&pool, cat, &data).await, os.clone())
        .unwrap()
        .write_table("public", "t", &[batch(vec![1, 2], vec![10, 20])])
        .await
        .unwrap();
    let first = MulticatalogProvider::with_pool(pool.clone(), cat_name)
        .await
        .unwrap()
        .get_current_snapshot()
        .unwrap();
    for (ids, vals) in [(vec![3, 4], vec![30, 40]), (vec![5, 6], vec![50, 60])] {
        DuckLakeTableWriter::new(writer_for(&pool, cat, &data).await, os.clone())
            .unwrap()
            .append_table("public", "t", &[batch(ids, vals)])
            .await
            .unwrap();
    }
    let pre_merge = MulticatalogProvider::with_pool(pool.clone(), cat_name)
        .await
        .unwrap()
        .get_current_snapshot()
        .unwrap();
    assert_eq!(live_files(&pool, cat_name).await.len(), 3, "three files");
    let rows_before = vec![(1, 10), (2, 20), (3, 30), (4, 40), (5, 50), (6, 60)];
    assert_eq!(read_rows(&pool, cat_name, None).await, rows_before);

    // Merge: the multicatalog reader must surface begin_snapshot/schema_version
    // (else this silently no-ops), then commit_compaction removes the sources.
    let result = with_writable_table(&pool, cat, cat_name, &data, |t, s| async move {
        t.merge_adjacent_files(&s, MergeOptions::default()).await
    })
    .await;
    assert_eq!(result.files_processed, 3, "all three merged");
    assert_eq!(result.files_created, 1);

    let files = live_files(&pool, cat_name).await;
    assert_eq!(files.len(), 1, "one merged file remains");
    assert_eq!(
        files[0].partial_max,
        Some(pre_merge),
        "partial_max = max origin snapshot"
    );
    // Sources removed from the catalog and scheduled for deletion (catalog-scoped).
    assert_eq!(
        scalar_i64(
            &pool,
            "SELECT COUNT(*) FROM ducklake_files_scheduled_for_deletion WHERE catalog_id = $1",
            cat,
        )
        .await,
        3,
    );
    // Results unchanged, and time travel to the pre-merge snapshot still works
    // (served by the partial file's per-row origin filtering).
    assert_eq!(read_rows(&pool, cat_name, None).await, rows_before);
    assert_eq!(
        read_rows(&pool, cat_name, Some(pre_merge)).await,
        rows_before
    );
    // Time travel to the first snapshot returns only the first insert's rows —
    // served entirely by the merged partial file's per-row origin filtering.
    assert_eq!(
        read_rows(&pool, cat_name, Some(first)).await,
        vec![(1, 10), (2, 20)]
    );
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn rewrite_data_files_postgres() {
    let (pool, _c) = spin_up_postgres().await.unwrap();
    let tmp = TempDir::new().unwrap();
    let data = tmp.path().join("data");
    std::fs::create_dir_all(&data).unwrap();
    let os: ObjStore = Arc::new(LocalFileSystem::new());
    let cat_name = "cat";
    let cat = MulticatalogManager::new(pool.clone())
        .create_catalog(cat_name)
        .await
        .unwrap();

    // One file of ten rows.
    DuckLakeTableWriter::new(writer_for(&pool, cat, &data).await, os.clone())
        .unwrap()
        .write_table(
            "public",
            "t",
            &[batch((1..=10).collect(), (1..=10).map(|v| v * 10).collect())],
        )
        .await
        .unwrap();

    // Delete eight of ten rows via SQL (a positional delete file).
    {
        let provider = MulticatalogProvider::with_pool(pool.clone(), cat_name)
            .await
            .unwrap();
        let writer = writer_for(&pool, cat, &data).await;
        let catalog = DuckLakeCatalog::with_writer(Arc::new(provider), writer).unwrap();
        let ctx = SessionContext::new();
        ctx.register_catalog(cat_name, Arc::new(catalog));
        ctx.sql(&format!("DELETE FROM {cat_name}.public.t WHERE id <= 8"))
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
    }
    assert_eq!(
        read_rows(&pool, cat_name, None).await,
        vec![(9, 90), (10, 100)]
    );

    // 8/10 deleted; rewrite with a 0.5 threshold.
    let result = with_writable_table(&pool, cat, cat_name, &data, |t, s| async move {
        t.rewrite_data_files(
            &s,
            RewriteOptions {
                delete_threshold: 0.5,
            },
        )
        .await
    })
    .await;
    assert_eq!(result.files_processed, 1);
    assert_eq!(result.files_created, 1);
    assert_eq!(result.rows_written, 2);

    // One live data file (the rewrite output), no live delete file, same results.
    let files = live_files(&pool, cat_name).await;
    assert_eq!(files.len(), 1);
    assert_eq!(
        files[0].partial_max, None,
        "a rewrite output is not partial"
    );
    assert_eq!(files[0].delete_file_id, None, "no live delete file");
    assert_eq!(
        read_rows(&pool, cat_name, None).await,
        vec![(9, 90), (10, 100)]
    );
}
