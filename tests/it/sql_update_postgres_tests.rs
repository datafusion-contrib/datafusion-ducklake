//! Postgres (multicatalog) coverage for SQL `UPDATE` end to end.
//!
//! The UPDATE commit reuses the append-with-deletes primitives (validated on
//! Postgres by `append_with_deletes_postgres_tests.rs`); what these cover is the
//! full `TableProvider::update` path driven by SQL against a writable multicatalog
//! catalog: affected-row count, the resulting values, rowid-lineage preservation
//! across the embedded-rowid file rewrite, the table's sort order, and — on a
//! partitioned table — rows moving to their new partition with the whole
//! multi-file rewrite plus its deletes in one snapshot.
//! Docker-gated (testcontainers Postgres).

#![cfg(feature = "write-postgres")]

use std::sync::Arc;

use arrow::array::{Array, Int32Array, Int64Array, RecordBatch, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::prelude::*;
use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeTableWriter, MetadataProvider, MetadataWriter, MulticatalogManager,
    MulticatalogProvider, NullOrder, PostgresMetadataWriter, SortDirection, SortField,
};
use object_store::local::LocalFileSystem;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
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

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn update_applies_postgres_sort_order() {
    let (pool, _container) = spin_up_postgres().await.unwrap();
    let temp = TempDir::new().unwrap();
    let data = temp.path().join("data");
    std::fs::create_dir_all(&data).unwrap();
    let catalog_name = "cat";
    let catalog_id = MulticatalogManager::new(pool.clone())
        .create_catalog(catalog_name)
        .await
        .unwrap();
    let writer = writer_for(&pool, catalog_id, &data).await;
    let metadata: Arc<dyn MetadataWriter> = writer.clone();
    let batch = RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(Int32Array::from(vec![1, 1, 1])),
            Arc::new(Int32Array::from(vec![30, 10, 20])),
        ],
    )
    .unwrap();
    let created = DuckLakeTableWriter::new(metadata, Arc::new(LocalFileSystem::new()))
        .unwrap()
        .write_table("public", "t", &[batch])
        .await
        .unwrap();
    writer
        .set_sort_spec(
            created.table_id,
            &[SortField::column(0, "val", SortDirection::Asc, NullOrder::NullsLast)],
        )
        .unwrap();

    let ctx = writable_ctx(&pool, catalog_name, catalog_id, &data).await;
    let batches = ctx
        .sql(&format!(
            "UPDATE {catalog_name}.public.t SET val = val + 1 WHERE id = 1"
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
        .unwrap()
        .value(0);
    let provider = MulticatalogProvider::with_pool(pool.clone(), catalog_name)
        .await
        .unwrap();
    let snapshot = provider.get_current_snapshot().unwrap();
    let mut files = provider
        .get_table_file_metadata_page(created.table_id, snapshot, None, 16)
        .unwrap();
    files.sort_by_key(|metadata| metadata.file.data_file_id);
    let output = files.last().unwrap();
    let path = data
        .join(format!("cat_{catalog_id}"))
        .join("public/t")
        .join(&output.file.file.path);
    let file = std::fs::File::open(path).unwrap();
    let output_batch = ParquetRecordBatchReaderBuilder::try_new(file)
        .unwrap()
        .build()
        .unwrap()
        .next()
        .unwrap()
        .unwrap();
    let values = output_batch
        .column(1)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();

    assert_eq!(count, 3);
    assert_eq!(files.len(), 2);
    assert_eq!(
        values.values().iter().copied().collect::<Vec<_>>(),
        vec![11, 21, 31],
    );
}

/// Partitioned `UPDATE` on the multicatalog Postgres writer, end to end through SQL:
/// an assignment that changes a row's partition-key value MOVES the row to its new
/// partition, the rewrite spans one file per output partition, all of it commits in
/// ONE snapshot together with the positional deletes, and every row keeps its `rowid`
/// lineage across the multi-partition rewrite.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn partitioned_update_moves_rows_and_is_one_snapshot_postgres() {
    use datafusion_ducklake::partition::PartitionTransform;

    let (pool, _c) = spin_up_postgres().await.unwrap();
    let tmp = TempDir::new().unwrap();
    let data = tmp.path().join("data");
    std::fs::create_dir_all(&data).unwrap();
    let os: ObjStore = Arc::new(LocalFileSystem::new());
    let cat_name = "pg_part_update";
    let cat = MulticatalogManager::new(pool.clone())
        .create_catalog(cat_name)
        .await
        .unwrap();

    // `val` is the partition key, so `SET val = ...` moves rows across partitions.
    let seed = RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(Int32Array::from(vec![10, 10, 20])),
        ],
    )
    .unwrap();
    let seeded = DuckLakeTableWriter::new(writer_for(&pool, cat, &data).await, os.clone())
        .unwrap()
        .write_table("public", "t", &[seed])
        .await
        .unwrap();
    writer_for(&pool, cat, &data)
        .await
        .set_partition_spec(
            seeded.table_id,
            &[("val".to_string(), PartitionTransform::Identity)],
        )
        .unwrap();
    let live_partition_id: i64 = sqlx::query_scalar(
        "SELECT partition_id FROM ducklake_partition_info
         WHERE table_id = $1 AND end_snapshot IS NULL",
    )
    .bind(seeded.table_id)
    .fetch_one(&pool)
    .await
    .unwrap();

    let before = read_rowid_rows(&pool, cat_name).await;
    assert_eq!(before, vec![(0, 1, 10), (1, 2, 10), (2, 3, 20)]);
    let snapshots_before: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM ducklake_snapshot")
        .fetch_one(&pool)
        .await
        .unwrap();

    // Rows 1 and 3 move to two DIFFERENT new partitions (11 and 21).
    let ctx = writable_ctx(&pool, cat_name, cat, &data).await;
    let batches = ctx
        .sql(&format!(
            "UPDATE {cat_name}.public.t SET val = val + 1 WHERE id IN (1, 3)"
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
    assert_eq!(count, 2);

    // One new snapshot for the whole mutation.
    let snapshots_after: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM ducklake_snapshot")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(
        snapshots_after,
        snapshots_before + 1,
        "a partitioned UPDATE is one snapshot"
    );

    // Two appended files, one per NEW partition, both carrying the live generation.
    let head: i64 = sqlx::query_scalar("SELECT MAX(snapshot_id) FROM ducklake_snapshot")
        .fetch_one(&pool)
        .await
        .unwrap();
    let appended: Vec<(Option<i64>, Option<String>)> = sqlx::query_as(
        "SELECT f.partition_id,
                (SELECT v.partition_value FROM ducklake_file_partition_value v
                 WHERE v.data_file_id = f.data_file_id AND v.partition_key_index = 0)
         FROM ducklake_data_file f
         WHERE f.table_id = $1 AND f.begin_snapshot = $2
         ORDER BY f.data_file_id",
    )
    .bind(seeded.table_id)
    .bind(head)
    .fetch_all(&pool)
    .await
    .unwrap();
    assert_eq!(
        appended.len(),
        2,
        "one rewritten file per output partition: {appended:?}"
    );
    assert!(
        appended
            .iter()
            .all(|(id, _)| *id == Some(live_partition_id)),
        "rewritten files carry the live partition generation: {appended:?}"
    );
    let mut values: Vec<Option<String>> = appended.into_iter().map(|(_, v)| v).collect();
    values.sort();
    assert_eq!(
        values,
        vec![Some("11".to_string()), Some("21".to_string())],
        "each moved row landed in its NEW partition"
    );

    // The deletes that supersede the old versions share that snapshot.
    let delete_snaps: Vec<i64> = sqlx::query_scalar(
        "SELECT begin_snapshot FROM ducklake_delete_file
         WHERE table_id = $1 AND end_snapshot IS NULL",
    )
    .bind(seeded.table_id)
    .fetch_all(&pool)
    .await
    .unwrap();
    assert!(!delete_snaps.is_empty(), "the old versions are superseded");
    assert!(
        delete_snaps.iter().all(|snap| *snap == head),
        "deletes share the appended files' snapshot: {delete_snaps:?}"
    );

    // Values moved and lineage survived the multi-partition rewrite.
    assert_eq!(
        read_rowid_rows(&pool, cat_name).await,
        vec![(0, 1, 11), (1, 2, 10), (2, 3, 21)],
        "rows 1,3 moved partition and kept rowids 0 and 2"
    );
}
