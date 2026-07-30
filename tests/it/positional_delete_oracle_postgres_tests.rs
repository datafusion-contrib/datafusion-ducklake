//! Postgres multicatalog counterpart of the positional-delete differential
//! oracle (the SQLite version lives in `positional_delete_oracle_tests.rs`).
//!
//! runtimedb runs on the **multicatalog Postgres** write path, and its
//! `set_delete_file` / snapshot handling is a *separate implementation* from the
//! SQLite single-catalog path — the catalog head is per-catalog (not a global
//! `MAX(snapshot_id)`), and table/file lookups are catalog-scoped. So the same
//! differential (delete via the real path → read → compare surviving VALUES
//! against a position-math-free reference) is re-run here against the production
//! backend, end to end with real parquet.
//!
//! The read + `resolve_positions` side is provider-agnostic (`DuckLakeCatalog`
//! works over any `MetadataProvider`, `MulticatalogProvider` included), and all
//! catalog-scoped lookups go through the provider API rather than raw SQL, so
//! nothing here reaches across catalogs.
//!
//! Docker-gated (testcontainers Postgres), matching the crate's other Postgres
//! tests; ignored under `skip-tests-with-docker` on macOS.

#![cfg(feature = "write-postgres")]

use std::sync::Arc;

use arrow::array::{Array, Int32Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::logical_expr::Operator;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_expr::expressions::{BinaryExpr, col, lit};
use datafusion::prelude::*;
use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeTable, DuckLakeTableWriter, MetadataProvider, MetadataWriter,
    MulticatalogManager, MulticatalogProvider, PostgresMetadataWriter,
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

fn object_store() -> ObjStore {
    Arc::new(LocalFileSystem::new())
}

/// The ORACLE: which ids survive deleting `del`, with zero position math.
fn survivors(ids: &[i32], del: &[i32]) -> Vec<i32> {
    let mut s: Vec<i32> = ids.iter().copied().filter(|x| !del.contains(x)).collect();
    s.sort_unstable();
    s
}

async fn create_catalog(pool: &PgPool, name: &str) -> i64 {
    MulticatalogManager::new(pool.clone())
        .create_catalog(name)
        .await
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

async fn write_ids(
    pool: &PgPool,
    cat: i64,
    os: ObjStore,
    data_path: &std::path::Path,
    ids: &[i32],
    rg: usize,
) {
    let w = writer_for(pool, cat, data_path).await;
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)])),
        vec![Arc::new(Int32Array::from(ids.to_vec()))],
    )
    .unwrap();
    DuckLakeTableWriter::new(w, os)
        .unwrap()
        .with_max_row_group_rows(rg)
        .write_table("public", "t", &[batch])
        .await
        .unwrap();
}

async fn append_ids(
    pool: &PgPool,
    cat: i64,
    os: ObjStore,
    data_path: &std::path::Path,
    ids: &[i32],
) {
    let w = writer_for(pool, cat, data_path).await;
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)])),
        vec![Arc::new(Int32Array::from(ids.to_vec()))],
    )
    .unwrap();
    DuckLakeTableWriter::new(w, os)
        .unwrap()
        .append_table("public", "t", &[batch])
        .await
        .unwrap();
}

/// Positional delete for `del` across every live data file in the catalog,
/// cumulative-aware — all lookups catalog-scoped via the provider API.
async fn apply_delete(
    pool: &PgPool,
    cat: i64,
    cat_name: &str,
    os: ObjStore,
    data_path: &std::path::Path,
    del: &[i32],
) {
    if del.is_empty() {
        return;
    }

    // Catalog-scoped metadata: head, schema, table, live files (with their
    // data_file_id and any live delete file = the cumulative prev).
    let meta = MulticatalogProvider::with_pool(pool.clone(), cat_name)
        .await
        .unwrap();
    let head = meta.get_current_snapshot().unwrap();
    let schema = meta.get_schema_by_name("public", head).unwrap().unwrap();
    let table_meta = meta
        .get_table_by_name(schema.schema_id, "t", head)
        .unwrap()
        .unwrap();
    let files = meta
        .get_table_files_for_select(table_meta.table_id, head)
        .unwrap();

    // A DuckLakeTable (over the same catalog) to resolve key -> physical positions.
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
    let state = ctx.state();

    let data_schema = Schema::new(vec![Field::new("id", DataType::Int32, false)]);
    let id = col("id", &data_schema).unwrap();
    let predicate = del
        .iter()
        .map(|d| -> Arc<dyn PhysicalExpr> {
            Arc::new(BinaryExpr::new(id.clone(), Operator::Eq, lit(*d)))
        })
        .reduce(|acc, e| Arc::new(BinaryExpr::new(acc, Operator::Or, e)))
        .expect("del is non-empty");

    let writer = writer_for(pool, cat, data_path).await;

    for tf in &files {
        let positions: Vec<i64> = table
            .resolve_positions(&state, &tf.file, predicate.clone())
            .await
            .unwrap()
            .into_iter()
            .collect();
        if positions.is_empty() {
            continue;
        }
        let del_info = DuckLakeTableWriter::new(writer.clone(), os.clone())
            .unwrap()
            .write_delete_file("public", "t", &tf.file.path, &positions)
            .await
            .unwrap();
        writer
            .set_delete_file(
                table_meta.table_id,
                "public",
                "t",
                head,
                tf.data_file_id,
                tf.delete_file_id, // cumulative prev (None on first generation)
                head,
                &del_info,
            )
            .unwrap();
    }
}

async fn read_ids(pool: &PgPool, cat_name: &str) -> Vec<i32> {
    let provider = MulticatalogProvider::with_pool(pool.clone(), cat_name)
        .await
        .unwrap();
    let catalog = DuckLakeCatalog::new(provider).unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog(cat_name, Arc::new(catalog));
    let batches = ctx
        .sql(&format!("SELECT id FROM {cat_name}.public.t ORDER BY id"))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let mut ids = Vec::new();
    for b in &batches {
        let c = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        for i in 0..b.num_rows() {
            ids.push(c.value(i));
        }
    }
    ids
}

async fn read_id_extra(pool: &PgPool, cat_name: &str) -> Vec<(i32, Option<i32>)> {
    let provider = MulticatalogProvider::with_pool(pool.clone(), cat_name)
        .await
        .unwrap();
    let catalog = DuckLakeCatalog::new(provider).unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog(cat_name, Arc::new(catalog));
    let batches = ctx
        .sql(&format!(
            "SELECT id, extra FROM {cat_name}.public.t ORDER BY id"
        ))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let mut rows = Vec::new();
    for b in &batches {
        let ids = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        let extra = b.column(1).as_any().downcast_ref::<Int32Array>().unwrap();
        for i in 0..b.num_rows() {
            let e = if extra.is_null(i) {
                None
            } else {
                Some(extra.value(i))
            };
            rows.push((ids.value(i), e));
        }
    }
    rows
}

/// Curated id-only shapes on multicatalog Postgres: multi-row-group deletes,
/// deletes across appended files, update (delete + re-insert), and cumulative
/// multi-generation deletes — each in its own catalog, one container.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn positional_delete_oracle_postgres() {
    let (pool, _c) = spin_up_postgres().await.unwrap();
    let tmp = TempDir::new().unwrap();
    let data = tmp.path().join("data");
    std::fs::create_dir_all(&data).unwrap();
    let os = object_store();

    // --- Curated single-file, multi-row-group shapes. ---
    let shapes: Vec<(&str, Vec<i32>, usize, Vec<i32>)> = vec![
        ("mrg_diff_groups", (1..=6).collect(), 2, vec![3, 5]),
        ("mrg_whole_group", (1..=6).collect(), 2, vec![1, 2]),
        ("mrg_boundary", (1..=6).collect(), 4, vec![4, 5]),
        ("ragged_last", (1..=5).collect(), 2, vec![5]),
        ("delete_all", vec![1, 2, 3], 2, vec![1, 2, 3]),
        ("no_match", (1..=4).collect(), 2, vec![99]),
    ];
    for (name, ids, rg, del) in shapes {
        let cat = create_catalog(&pool, name).await;
        write_ids(&pool, cat, os.clone(), &data, &ids, rg).await;
        assert_eq!(
            read_ids(&pool, name).await,
            survivors(&ids, &[]),
            "baseline [{name}]"
        );
        apply_delete(&pool, cat, name, os.clone(), &data, &del).await;
        assert_eq!(
            read_ids(&pool, name).await,
            survivors(&ids, &del),
            "survivors [{name}]"
        );
    }

    // --- Deletes across appended files. ---
    {
        let name = "appended";
        let cat = create_catalog(&pool, name).await;
        write_ids(&pool, cat, os.clone(), &data, &[1, 2, 3, 4], 2).await;
        append_ids(&pool, cat, os.clone(), &data, &[5, 6, 7, 8]).await;
        let del = vec![2, 3, 6, 8];
        apply_delete(&pool, cat, name, os.clone(), &data, &del).await;
        assert_eq!(
            read_ids(&pool, name).await,
            survivors(&[1, 2, 3, 4, 5, 6, 7, 8], &del),
            "survivors across two files"
        );
    }

    // --- Update = delete old + re-insert (re-inserted deleted key survives). ---
    {
        let name = "update";
        let cat = create_catalog(&pool, name).await;
        let orig = vec![1, 2, 3, 4, 5, 6];
        write_ids(&pool, cat, os.clone(), &data, &orig, 2).await;
        let del = vec![2, 4];
        apply_delete(&pool, cat, name, os.clone(), &data, &del).await;
        append_ids(&pool, cat, os.clone(), &data, &[2, 7]).await;
        let mut want = survivors(&orig, &del);
        want.extend_from_slice(&[2, 7]);
        want.sort_unstable();
        assert_eq!(
            read_ids(&pool, name).await,
            want,
            "update: delete old + insert new"
        );
    }

    // --- Cumulative multi-generation deletes. ---
    {
        let name = "cumulative";
        let cat = create_catalog(&pool, name).await;
        let ids: Vec<i32> = (1..=8).collect();
        write_ids(&pool, cat, os.clone(), &data, &ids, 3).await;
        apply_delete(&pool, cat, name, os.clone(), &data, &[2, 4]).await;
        assert_eq!(
            read_ids(&pool, name).await,
            survivors(&ids, &[2, 4]),
            "after gen 1"
        );
        apply_delete(&pool, cat, name, os.clone(), &data, &[2, 4, 6, 8]).await;
        assert_eq!(
            read_ids(&pool, name).await,
            survivors(&ids, &[2, 4, 6, 8]),
            "after gen 2"
        );
    }
}

/// Schema-evolved table on multicatalog Postgres: an appended wider file adds a
/// nullable column; the pre-evolution file null-fills it, and a delete across
/// both files keeps the correct `(id, extra)` rows.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn positional_delete_oracle_postgres_schema_evolution() {
    let (pool, _c) = spin_up_postgres().await.unwrap();
    let tmp = TempDir::new().unwrap();
    let data = tmp.path().join("data");
    std::fs::create_dir_all(&data).unwrap();
    let os = object_store();

    let name = "schema_evo";
    let cat = create_catalog(&pool, name).await;

    // File 1: {id}.
    write_ids(&pool, cat, os.clone(), &data, &[1, 2, 3], 2).await;

    // File 2: append under a wider schema — adds nullable `extra` (DDL).
    let w = writer_for(&pool, cat, &data).await;
    let wider = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("extra", DataType::Int32, true),
    ]));
    let b2 = RecordBatch::try_new(
        wider,
        vec![
            Arc::new(Int32Array::from(vec![4, 5])),
            Arc::new(Int32Array::from(vec![Some(40), Some(50)])),
        ],
    )
    .unwrap();
    DuckLakeTableWriter::new(w, os.clone())
        .unwrap()
        .append_table("public", "t", &[b2])
        .await
        .unwrap();

    assert_eq!(
        read_id_extra(&pool, name).await,
        vec![(1, None), (2, None), (3, None), (4, Some(40)), (5, Some(50))],
        "baseline: pre-evolution rows null-fill extra"
    );

    apply_delete(&pool, cat, name, os.clone(), &data, &[2, 5]).await;

    assert_eq!(
        read_id_extra(&pool, name).await,
        vec![(1, None), (3, None), (4, Some(40))],
        "survivors keep correct (id, extra) across the schema boundary"
    );
}
