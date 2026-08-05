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

/// An append+delete commit on a PARTITIONED table must persist the appended file's
/// partition assignment, like every other commit path does.
///
/// Without it the commit succeeds and the rows read back correctly — the only symptom
/// is that the new file carries no `partition_id` and no
/// `ducklake_file_partition_value` rows, so it can never be pruned again: an island in
/// an otherwise partitioned table. The partition fence does not catch this, because the
/// `DataFileInfo` it validates DOES carry the assignment; only the persistence was
/// missing. So this asserts the CATALOG ROWS, not the query result.
///
/// The commit carries a REAL delete entry: an empty `deletes` slice is a plain append
/// and delegates to `finish`, which would exercise a different commit method and leave
/// this path uncovered.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn append_with_deletes_persists_partition_metadata_postgres() {
    use datafusion_ducklake::partition::PartitionTransform;

    let (pool, _c) = spin_up_postgres().await.unwrap();
    let mgr = MulticatalogManager::new(pool.clone());
    let cat = mgr.create_catalog("pg_part_deletes").await.unwrap();
    let temp = TempDir::new().unwrap();
    let data_path = temp.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();
    let writer = writer_for(&pool, cat, &data_path).await;
    let object_store: ObjStore = Arc::new(LocalFileSystem::new());
    let table_writer =
        DuckLakeTableWriter::new(writer.clone() as Arc<dyn MetadataWriter>, object_store).unwrap();

    // Seed one row, then partition the table by `val`.
    let seed = RecordBatch::try_new(
        schema(),
        vec![Arc::new(Int32Array::from(vec![1])), Arc::new(Int32Array::from(vec![7]))],
    )
    .unwrap();
    let seeded = table_writer
        .write_table("public", "t", &[seed])
        .await
        .unwrap();
    writer
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

    // Supersede the seed row (its only row, at position 0) so the commit genuinely
    // carries a delete — the code path a partitioned update takes.
    let (seed_data_file_id, seed_path) = sqlx::query_as::<_, (i64, String)>(
        "SELECT data_file_id, path FROM ducklake_data_file
         WHERE table_id = $1 AND end_snapshot IS NULL",
    )
    .bind(seeded.table_id)
    .fetch_one(&pool)
    .await
    .unwrap();
    let del_info = table_writer
        .write_delete_file("public", "t", &seed_path, &[0])
        .await
        .unwrap();

    // Append one row alongside that delete.
    let appended = RecordBatch::try_new(
        schema(),
        vec![Arc::new(Int32Array::from(vec![2])), Arc::new(Int32Array::from(vec![9]))],
    )
    .unwrap();
    let mut session = table_writer
        .begin_write("public", "t", schema().as_ref(), WriteMode::Append)
        .unwrap();
    session.write_batch(&appended).unwrap();
    session
        .finish_with_deletes(&[DeleteFileEntry {
            data_file_id: seed_data_file_id,
            expected_prev_delete_file: None,
            delete: del_info,
        }])
        .await
        .unwrap();

    // The appended file must carry the live generation AND its partition value.
    let rows: Vec<(Option<i64>, Option<String>)> = sqlx::query_as(
        "SELECT df.partition_id, fpv.partition_value
         FROM ducklake_data_file AS df
         LEFT JOIN ducklake_file_partition_value AS fpv
           ON fpv.data_file_id = df.data_file_id
         WHERE df.table_id = $1 AND df.record_count = 1 AND df.end_snapshot IS NULL
           AND df.begin_snapshot = (SELECT MAX(snapshot_id) FROM ducklake_snapshot)",
    )
    .bind(seeded.table_id)
    .fetch_all(&pool)
    .await
    .unwrap();
    assert_eq!(rows.len(), 1, "exactly one appended file: {rows:?}");
    assert_eq!(
        rows[0].0,
        Some(live_partition_id),
        "the appended file must carry the live partition generation"
    );
    assert_eq!(
        rows[0].1,
        Some("9".to_string()),
        "the appended file must carry its partition value"
    );
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

/// Multicatalog Postgres counterpart of the multi-file append+delete commit: N appended
/// data files AND M delete files land in ONE snapshot, with each appended file carrying
/// its own partition assignment and its own per-column statistics.
///
/// Partitioned so the appended side genuinely spans several files: the new row versions
/// MOVE partition, which is the shape a partitioned keyed mutation produces.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn multi_file_append_with_deletes_is_one_snapshot_postgres() {
    use datafusion_ducklake::partition::PartitionTransform;

    let (pool, _c) = spin_up_postgres().await.unwrap();
    let tmp = TempDir::new().unwrap();
    let data = tmp.path().join("data");
    std::fs::create_dir_all(&data).unwrap();
    let os: ObjStore = Arc::new(LocalFileSystem::new());
    let cat_name = "pg_multi_deletes";
    let cat = MulticatalogManager::new(pool.clone())
        .create_catalog(cat_name)
        .await
        .unwrap();
    let sch = schema();
    let writer = writer_for(&pool, cat, &data).await;
    let table_writer =
        DuckLakeTableWriter::new(writer.clone() as Arc<dyn MetadataWriter>, os.clone()).unwrap();

    // Seed one row per partition of `val`, then partition the table by `val` so the
    // seed files each hold exactly one row at position 0.
    let seed = RecordBatch::try_new(
        sch.clone(),
        vec![Arc::new(Int32Array::from(vec![1])), Arc::new(Int32Array::from(vec![10]))],
    )
    .unwrap();
    let seeded = table_writer
        .write_table("public", "t", &[seed])
        .await
        .unwrap();
    writer
        .set_partition_spec(
            seeded.table_id,
            &[("val".to_string(), PartitionTransform::Identity)],
        )
        .unwrap();
    let extra = RecordBatch::try_new(
        sch.clone(),
        vec![Arc::new(Int32Array::from(vec![2])), Arc::new(Int32Array::from(vec![20]))],
    )
    .unwrap();
    let seeded_partitioned = DuckLakeTableWriter::new(
        writer_for(&pool, cat, &data).await as Arc<dyn MetadataWriter>,
        os.clone(),
    )
    .unwrap()
    .append_table("public", "t", &[extra])
    .await
    .unwrap();

    // The two live data files: the pre-partition seed and the partitioned append.
    let seed_files: Vec<(i64, String, bool, i64)> = sqlx::query_as(
        "SELECT data_file_id, path, path_is_relative, file_size_bytes
         FROM ducklake_data_file
         WHERE table_id = $1 AND end_snapshot IS NULL
         ORDER BY data_file_id",
    )
    .bind(seeded.table_id)
    .fetch_all(&pool)
    .await
    .unwrap();
    assert_eq!(seed_files.len(), 2, "two seed data files");

    // Supersede the row in each seed file (each holds one row, at position 0).
    let mut entries = Vec::new();
    for (data_file_id, path, _, _) in &seed_files {
        let del = DuckLakeTableWriter::new(
            writer_for(&pool, cat, &data).await as Arc<dyn MetadataWriter>,
            os.clone(),
        )
        .unwrap()
        .write_delete_file("public", "t", path, &[0])
        .await
        .unwrap();
        entries.push(DeleteFileEntry {
            data_file_id: *data_file_id,
            expected_prev_delete_file: None,
            delete: del,
        });
    }

    // New versions in two DIFFERENT partitions -> the partitioned session produces two
    // appended files, which the one-file cap used to refuse.
    let new_versions = RecordBatch::try_new(
        sch.clone(),
        vec![Arc::new(Int32Array::from(vec![1, 2])), Arc::new(Int32Array::from(vec![100, 200]))],
    )
    .unwrap();
    let mut session = DuckLakeTableWriter::new(
        writer_for(&pool, cat, &data).await as Arc<dyn MetadataWriter>,
        os.clone(),
    )
    .unwrap()
    .begin_write("public", "t", sch.as_ref(), WriteMode::Append)
    .unwrap();
    session.write_batch(&new_versions).unwrap();
    let committed = session.finish_with_deletes(&entries).await.unwrap();
    assert_eq!(
        committed.files_written, 2,
        "one appended file per partition"
    );

    assert_eq!(
        read_pairs(&pool, cat_name).await,
        vec![(1, 100), (2, 200)],
        "both rows superseded by their new versions"
    );

    // ONE snapshot for both appended files AND both delete files.
    let appended: Vec<(i64, i64, Option<i64>, Option<String>)> = sqlx::query_as(
        "SELECT df.data_file_id, df.begin_snapshot, df.partition_id, fpv.partition_value
         FROM ducklake_data_file AS df
         LEFT JOIN ducklake_file_partition_value AS fpv
           ON fpv.data_file_id = df.data_file_id
         WHERE df.table_id = $1 AND df.begin_snapshot > $2
         ORDER BY df.data_file_id",
    )
    .bind(seeded.table_id)
    .bind(seeded_partitioned.snapshot_id)
    .fetch_all(&pool)
    .await
    .unwrap();
    assert_eq!(appended.len(), 2, "two appended files: {appended:?}");
    assert!(
        appended
            .iter()
            .all(|(_, snap, _, _)| *snap == committed.snapshot_id),
        "every appended file carries the one committed snapshot: {appended:?}"
    );
    let mut partition_values: Vec<Option<String>> =
        appended.iter().map(|(_, _, _, v)| v.clone()).collect();
    partition_values.sort();
    assert_eq!(
        partition_values,
        vec![Some("100".to_string()), Some("200".to_string())],
        "each appended file carries its own partition value"
    );
    let delete_snaps: Vec<i64> = sqlx::query_scalar(
        "SELECT begin_snapshot FROM ducklake_delete_file
         WHERE table_id = $1 AND end_snapshot IS NULL",
    )
    .bind(seeded.table_id)
    .fetch_all(&pool)
    .await
    .unwrap();
    assert_eq!(delete_snaps.len(), 2, "one delete file per seed file");
    assert!(
        delete_snaps
            .iter()
            .all(|snap| *snap == committed.snapshot_id),
        "both delete files share that same snapshot: {delete_snaps:?}"
    );

    // Per-column statistics for EVERY appended file, not just the first.
    for (data_file_id, _, _, _) in &appended {
        let stats: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM ducklake_file_column_stats WHERE data_file_id = $1",
        )
        .bind(data_file_id)
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(
            stats, 2,
            "file {data_file_id} must carry stats for both columns"
        );
    }

    // Row lineage: the appended files draw distinct, non-overlapping rowid ranges.
    let row_id_starts: Vec<i64> = sqlx::query_scalar(
        "SELECT row_id_start FROM ducklake_data_file
         WHERE table_id = $1 AND begin_snapshot = $2 ORDER BY row_id_start",
    )
    .bind(seeded.table_id)
    .bind(committed.snapshot_id)
    .fetch_all(&pool)
    .await
    .unwrap();
    assert_eq!(row_id_starts.len(), 2);
    assert!(
        row_id_starts[0] < row_id_starts[1],
        "each appended file gets its own rowid range: {row_id_starts:?}"
    );
}
