//! Postgres coverage for the SQL-`DELETE` commit primitives added in the
//! `delete_from` PR: [`MetadataWriter::commit_positional_deletes`] and
//! [`MetadataWriter::commit_truncate`]. The multicatalog Postgres write path is a
//! separate implementation from SQLite (catalog-scoped head, `NOW()`,
//! `schema_version` carry-forward, `::BIGINT` casts, `lock_catalog` /
//! `advance_catalog_head`), so those ~220 lines of Postgres-specific SQL are
//! re-validated here end to end rather than trusted from the SQLite tests.
//!
//! Driven at the `MetadataWriter` level (the same way
//! `append_with_deletes_postgres_tests.rs` drives `register_data_file_with_deletes`)
//! so the test targets the new SQL directly. Docker-gated (testcontainers).

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
    MetadataWriter, MulticatalogManager, MulticatalogProvider, PostgresMetadataWriter,
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

/// Number of snapshots mapped to this catalog (advance_catalog_head inserts one
/// row per committed snapshot; the no-op truncate guard must insert none).
async fn catalog_snapshot_count(pool: &PgPool, cat: i64) -> i64 {
    sqlx::query_scalar("SELECT COUNT(*) FROM ducklake_catalog_snapshot_map WHERE catalog_id = $1")
        .bind(cat)
        .fetch_one(pool)
        .await
        .unwrap()
}

/// Drives the two new commit primitives directly on the Postgres writer:
/// (1) `commit_positional_deletes` for a WHERE delete, asserting the read
/// reflects it and it lands as exactly one catalog snapshot; (2) `commit_truncate`
/// removing the survivors with a correct rows-affected count; (3) a repeated
/// `commit_truncate` proving the no-op guard commits no spurious snapshot.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn positional_delete_and_truncate_commit_postgres() {
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

    // Seed (1,10),(2,20),(3,30),(4,40) as one data file.
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

    // Resolve physical positions of ids {2,4} (positions 1,3) on the seed file.
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

    // --- commit_positional_deletes: a WHERE delete, no appended data file. ---
    let snaps_before_delete = catalog_snapshot_count(&pool, cat).await;
    let writer = writer_for(&pool, cat, &data).await;
    let del_info = DuckLakeTableWriter::new(writer.clone(), os.clone())
        .unwrap()
        .write_delete_file("public", "t", &tf.file.path, &positions)
        .await
        .unwrap();
    let entries = vec![DeleteFileEntry {
        data_file_id: tf.data_file_id,
        expected_prev_delete_file: tf.delete_file_id,
        delete: del_info,
    }];
    let commit = writer
        .commit_positional_deletes(table_meta.table_id, "public", "t", head, &entries)
        .unwrap();

    assert_eq!(
        read_pairs(&pool, cat_name).await,
        vec![(1, 10), (3, 30)],
        "ids 2,4 deleted; 1,3 survive"
    );
    assert_eq!(
        catalog_snapshot_count(&pool, cat).await,
        snaps_before_delete + 1,
        "positional delete commits exactly one catalog snapshot"
    );
    let delete_snap: i64 = sqlx::query_scalar(
        "SELECT begin_snapshot FROM ducklake_delete_file
         WHERE data_file_id = $1 AND end_snapshot IS NULL",
    )
    .bind(tf.data_file_id)
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(
        delete_snap, commit.snapshot_id,
        "delete file's begin_snapshot is the committed head"
    );

    // --- commit_truncate: remove the two survivors. ---
    let head2 = MulticatalogProvider::with_pool(pool.clone(), cat_name)
        .await
        .unwrap()
        .get_current_snapshot()
        .unwrap();
    let removed = writer
        .commit_truncate(table_meta.table_id, "public", "t", head2)
        .unwrap();
    assert_eq!(
        removed, 2,
        "gross 4 minus 2 already-deleted = 2 live rows removed"
    );
    assert_eq!(
        read_pairs(&pool, cat_name).await,
        Vec::<(i32, i32)>::new(),
        "table empty"
    );
    let snaps_after_truncate = catalog_snapshot_count(&pool, cat).await;

    // --- no-op guard: a second truncate must NOT commit a snapshot. ---
    let removed_again = writer
        .commit_truncate(table_meta.table_id, "public", "t", head2)
        .unwrap();
    assert_eq!(removed_again, 0, "nothing left to truncate");
    assert_eq!(
        catalog_snapshot_count(&pool, cat).await,
        snaps_after_truncate,
        "a no-op truncate must not create a snapshot"
    );
}
