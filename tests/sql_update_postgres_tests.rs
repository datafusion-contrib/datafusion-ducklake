//! Postgres (multicatalog) coverage for SQL `UPDATE` end to end.
//!
//! The UPDATE commit reuses `register_data_file_with_deletes` (already validated
//! on Postgres by `append_with_deletes_postgres_tests.rs`); this PR only flips
//! `PostgresMetadataWriter::supports_update()` to true. What was untested is the
//! full `TableProvider::update` path driven by SQL against a writable multicatalog
//! catalog, which this test exercises: affected-row count, the resulting values,
//! and rowid-lineage preservation across the embedded-rowid file rewrite.
//! Docker-gated (testcontainers Postgres).

#![cfg(feature = "write-postgres")]

use std::sync::Arc;

use arrow::array::{Array, Int32Array, Int64Array, RecordBatch, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::prelude::*;
use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeTableWriter, MetadataWriter, MulticatalogManager,
    MulticatalogProvider, PostgresMetadataWriter,
};
use object_store::local::LocalFileSystem;
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

/// A writable SessionContext over the multicatalog catalog (provider + writer).
async fn writable_ctx(
    pool: &PgPool,
    cat_name: &str,
    cat: i64,
    data: &std::path::Path,
) -> SessionContext {
    let provider = MulticatalogProvider::with_pool(pool.clone(), cat_name)
        .await
        .unwrap();
    let writer = PostgresMetadataWriter::with_pool(pool.clone(), cat)
        .await
        .unwrap();
    writer.set_data_path(data.to_str().unwrap()).unwrap();
    let catalog = DuckLakeCatalog::with_writer(Arc::new(provider), Arc::new(writer)).unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog(cat_name, Arc::new(catalog));
    ctx
}

async fn read_rowid_rows(pool: &PgPool, cat_name: &str) -> Vec<(i64, i32, i32)> {
    let provider = MulticatalogProvider::with_pool(pool.clone(), cat_name)
        .await
        .unwrap();
    let catalog = DuckLakeCatalog::new(provider)
        .unwrap()
        .with_row_lineage(true);
    let ctx = SessionContext::new();
    ctx.register_catalog(cat_name, Arc::new(catalog));
    let batches = ctx
        .sql(&format!(
            "SELECT rowid, id, val FROM {cat_name}.public.t ORDER BY id"
        ))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let mut rows = Vec::new();
    for b in &batches {
        let ri = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        let i = b.column(1).as_any().downcast_ref::<Int32Array>().unwrap();
        let v = b.column(2).as_any().downcast_ref::<Int32Array>().unwrap();
        for r in 0..b.num_rows() {
            assert!(!ri.is_null(r), "rowid must not be NULL after UPDATE");
            rows.push((ri.value(r), i.value(r), v.value(r)));
        }
    }
    rows
}

/// SQL `UPDATE ... WHERE` end to end on multicatalog Postgres: correct count,
/// correct new/old values, row count unchanged, and rowid lineage preserved.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn update_where_end_to_end_postgres() {
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

    // Seed (1,10),(2,20),(3,40),(4,40) as one data file (rowids 0..3).
    let seed = RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4])),
            Arc::new(Int32Array::from(vec![10, 20, 30, 40])),
        ],
    )
    .unwrap();
    DuckLakeTableWriter::new(writer_for(&pool, cat, &data).await, os.clone())
        .unwrap()
        .write_table("public", "t", &[seed])
        .await
        .unwrap();
    assert_eq!(
        read_rowid_rows(&pool, cat_name).await,
        vec![(0, 1, 10), (1, 2, 20), (2, 3, 30), (3, 4, 40)],
        "baseline rowids"
    );

    // UPDATE via SQL through a writable catalog.
    let ctx = writable_ctx(&pool, cat_name, cat, &data).await;
    let batches = ctx
        .sql(&format!(
            "UPDATE {cat_name}.public.t SET val = val * 10 WHERE id IN (2, 4)"
        ))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let count = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .expect("count is UInt64")
        .value(0);
    assert_eq!(count, 2, "ids 2 and 4 matched");

    // Updated rows keep their ORIGINAL rowids (1 and 3); others unchanged.
    assert_eq!(
        read_rowid_rows(&pool, cat_name).await,
        vec![(0, 1, 10), (1, 2, 200), (2, 3, 30), (3, 4, 400)],
        "values updated in place; rowids 1 and 3 preserved across the rewrite"
    );
}
