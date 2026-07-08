//! Postgres multicatalog counterpart of `append_with_deletes_tests.rs`.
//!
//! The multicatalog Postgres write path is a *separate implementation* from the
//! SQLite one (per-catalog head, catalog-scoped lookups), so the atomic
//! append-with-deletes primitive (`register_data_file_with_deletes`, driven via
//! `TableWriteSession::finish_with_deletes`) is re-validated here end to end:
//! an update (delete + insert by key) lands as ONE snapshot, with the resulting
//! VALUES asserted. Docker-gated (testcontainers Postgres).

#![cfg(feature = "write-postgres")]

use std::sync::Arc;

use arrow::array::{Array, Int32Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::logical_expr::Operator;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_expr::expressions::{BinaryExpr, col, lit};
use datafusion::prelude::*;
use datafusion_ducklake::{
    DeleteFileEntry, DuckLakeCatalog, DuckLakeTable, DuckLakeTableWriter, MetadataProvider,
    MetadataWriter, MulticatalogManager, MulticatalogProvider, PostgresMetadataWriter, WriteMode,
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

/// The `(id, val)` table schema used throughout.
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

async fn read_pairs(pool: &PgPool, cat_name: &str) -> Vec<(i32, i32)> {
    let provider = MulticatalogProvider::with_pool(pool.clone(), cat_name)
        .await
        .unwrap();
    let catalog = DuckLakeCatalog::new(provider).unwrap();
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

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn update_via_finish_with_deletes_is_one_snapshot_postgres() {
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
    let sch = schema();

    // Seed (id, val): (1,10),(2,20),(3,30),(4,40) as one data file.
    let seed = RecordBatch::try_new(
        sch.clone(),
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
        read_pairs(&pool, cat_name).await,
        vec![(1, 10), (2, 20), (3, 30), (4, 40)],
        "baseline"
    );

    // Catalog-scoped metadata: head, table, the single live data file.
    let meta = MulticatalogProvider::with_pool(pool.clone(), cat_name)
        .await
        .unwrap();
    let head = meta.get_current_snapshot().unwrap();
    let schema_meta = meta.get_schema_by_name("public", head).unwrap().unwrap();
    let table_meta = meta
        .get_table_by_name(schema_meta.schema_id, "t", head)
        .unwrap()
        .unwrap();
    let files = meta
        .get_table_files_for_select(table_meta.table_id, head)
        .unwrap();
    assert_eq!(files.len(), 1, "one seed data file");
    let tf = files[0].clone();

    // Resolve positions of ids {2,4} on the seed file (physical positions 1,3).
    let read = MulticatalogProvider::with_pool(pool.clone(), cat_name)
        .await
        .unwrap();
    let catalog = DuckLakeCatalog::new(read).unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog(cat_name, Arc::new(catalog));
    let table_provider = ctx
        .catalog(cat_name)
        .unwrap()
        .schema("public")
        .unwrap()
        .table("t")
        .await
        .unwrap()
        .unwrap();
    let table = (table_provider.as_ref() as &dyn std::any::Any)
        .downcast_ref::<DuckLakeTable>()
        .expect("provider is a DuckLakeTable");
    let data_schema = schema();
    let id: Arc<dyn PhysicalExpr> = col("id", data_schema.as_ref()).unwrap();
    let eq2: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(id.clone(), Operator::Eq, lit(2i32)));
    let eq4: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(id, Operator::Eq, lit(4i32)));
    let predicate: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(eq2, Operator::Or, eq4));
    let state = ctx.state();
    let mut positions: Vec<i64> = table
        .resolve_positions(&state, &tf.file, predicate)
        .await
        .unwrap()
        .into_iter()
        .collect();
    positions.sort_unstable();
    assert_eq!(positions, vec![1, 3], "ids 2,4 sit at positions 1,3");

    // Author the delete file, then append the NEW versions and commit them
    // together with the delete in ONE snapshot.
    let writer = writer_for(&pool, cat, &data).await;
    let del_info = DuckLakeTableWriter::new(writer.clone(), os.clone())
        .unwrap()
        .write_delete_file("public", "t", &tf.file.path, &positions)
        .await
        .unwrap();
    let new_versions = RecordBatch::try_new(
        sch.clone(),
        vec![Arc::new(Int32Array::from(vec![2, 4])), Arc::new(Int32Array::from(vec![200, 400]))],
    )
    .unwrap();
    let mut session = DuckLakeTableWriter::new(writer.clone(), os.clone())
        .unwrap()
        .begin_write("public", "t", sch.as_ref(), WriteMode::Append)
        .unwrap();
    session.write_batch(&new_versions).unwrap();
    let entries = vec![DeleteFileEntry {
        data_file_id: tf.data_file_id,
        expected_prev_delete_file: tf.delete_file_id,
        delete: del_info,
    }];
    let result = session.finish_with_deletes(&entries).await.unwrap();

    assert_eq!(
        read_pairs(&pool, cat_name).await,
        vec![(1, 10), (2, 200), (3, 30), (4, 400)],
        "rows 2,4 updated in place; 1,3 unchanged"
    );

    // Atomicity: the delete file and the appended data file carry the SAME
    // begin_snapshot — the committed head — so they became visible together.
    let delete_snap: i64 = sqlx::query_scalar(
        "SELECT begin_snapshot FROM ducklake_delete_file
         WHERE data_file_id = $1 AND end_snapshot IS NULL",
    )
    .bind(tf.data_file_id)
    .fetch_one(&pool)
    .await
    .unwrap();
    let appended_snap: i64 = sqlx::query_scalar(
        "SELECT begin_snapshot FROM ducklake_data_file
         WHERE table_id = $1 AND data_file_id <> $2 AND end_snapshot IS NULL",
    )
    .bind(table_meta.table_id)
    .bind(tf.data_file_id)
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(
        delete_snap, appended_snap,
        "delete file and appended data file share one snapshot"
    );
    assert_eq!(
        delete_snap, result.snapshot_id,
        "that shared snapshot is the committed head"
    );
}
