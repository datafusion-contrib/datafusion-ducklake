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
    MetadataProvider, MetadataWriter, MulticatalogManager, MulticatalogProvider, NullOrder,
    PostgresMetadataWriter, RewriteOptions, SortDirection, SortField,
};
use object_store::local::LocalFileSystem;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use sqlx::AssertSqlSafe;
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

fn file_values(data: &std::path::Path, catalog_id: i64, path: &str) -> Vec<i32> {
    let file = std::fs::File::open(
        data.join(format!("cat_{catalog_id}"))
            .join("public")
            .join("t")
            .join(path),
    )
    .unwrap();
    ParquetRecordBatchReaderBuilder::try_new(file)
        .unwrap()
        .build()
        .unwrap()
        .map(|batch| {
            let batch = batch.unwrap();
            batch
                .column_by_name("val")
                .unwrap()
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .values()
                .to_vec()
        })
        .collect::<Vec<_>>()
        .concat()
}

async fn scalar_i64(pool: &PgPool, sql: &str, cat: i64) -> i64 {
    sqlx::query(AssertSqlSafe(sql))
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
    let created = DuckLakeTableWriter::new(writer_for(&pool, cat, &data).await, os.clone())
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
    writer_for(&pool, cat, &data)
        .await
        .set_sort_spec(
            created.table_id,
            &[SortField::column(0, "val", SortDirection::Desc, NullOrder::NullsLast)],
        )
        .unwrap();

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
    assert_eq!(
        file_values(&data, cat, &files[0].file.path),
        vec![60, 50, 40, 30, 20, 10],
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
    let created = DuckLakeTableWriter::new(writer_for(&pool, cat, &data).await, os.clone())
        .unwrap()
        .write_table(
            "public",
            "t",
            &[batch((1..=10).collect(), (1..=10).map(|v| v * 10).collect())],
        )
        .await
        .unwrap();
    writer_for(&pool, cat, &data)
        .await
        .set_sort_spec(
            created.table_id,
            &[SortField::column(0, "val", SortDirection::Desc, NullOrder::NullsLast)],
        )
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
                ..RewriteOptions::default()
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
    assert_eq!(file_values(&data, cat, &files[0].file.path), vec![100, 90]);
    assert_eq!(
        read_rows(&pool, cat_name, None).await,
        vec![(9, 90), (10, 100)]
    );
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn rewrite_targets_explicit_postgres_data_files() {
    let (pool, _container) = spin_up_postgres().await.unwrap();
    let temp = TempDir::new().unwrap();
    let data = temp.path().join("data");
    std::fs::create_dir_all(&data).unwrap();
    let store: ObjStore = Arc::new(LocalFileSystem::new());
    let catalog_name = "cat";
    let catalog_id = MulticatalogManager::new(pool.clone())
        .create_catalog(catalog_name)
        .await
        .unwrap();
    let writer = writer_for(&pool, catalog_id, &data).await;
    let metadata: Arc<dyn MetadataWriter> = writer.clone();
    let created = DuckLakeTableWriter::new(Arc::clone(&metadata), Arc::clone(&store))
        .unwrap()
        .write_table("public", "t", &[batch(vec![1, 2], vec![10, 20])])
        .await
        .unwrap();
    DuckLakeTableWriter::new(metadata, store)
        .unwrap()
        .append_table("public", "t", &[batch(vec![3, 4], vec![30, 40])])
        .await
        .unwrap();
    writer
        .set_sort_spec(
            created.table_id,
            &[SortField::column(0, "val", SortDirection::Desc, NullOrder::NullsLast)],
        )
        .unwrap();
    let mut before = live_files(&pool, catalog_name).await;
    before.sort_by_key(|file| file.data_file_id);
    let selected_id = before[0].data_file_id;
    let unaffected_path = before[1].file.path.clone();

    let result = with_writable_table(
        &pool,
        catalog_id,
        catalog_name,
        &data,
        |table, state| async move {
            table
                .rewrite_data_files(
                    &state,
                    RewriteOptions {
                        data_file_ids: Some(vec![selected_id]),
                        ..RewriteOptions::default()
                    },
                )
                .await
        },
    )
    .await;

    let after = live_files(&pool, catalog_name).await;
    let rewritten = after
        .iter()
        .find(|file| file.file.path != unaffected_path)
        .unwrap();
    assert_eq!(
        result,
        CompactionResult {
            files_processed: 1,
            files_created: 1,
            rows_written: 2,
        },
    );
    assert_eq!(after.len(), 2);
    assert!(after.iter().any(|file| file.file.path == unaffected_path),);
    assert_eq!(
        file_values(&data, catalog_id, &rewritten.file.path),
        vec![20, 10],
    );
    assert_eq!(
        read_rows(&pool, catalog_name, None).await,
        vec![(1, 10), (2, 20), (3, 30), (4, 40)],
    );
}

/// The Postgres counterpart of the SQLite
/// `merge_only_within_a_partition_and_preserves_assignment` test.
///
/// Compaction of a PARTITIONED table must merge only within a partition AND carry
/// each output's partition assignment over from its sources, on every metadata
/// backend. Official DuckLake takes the merged file's partition straight from
/// `source_files[0]` and writes the resulting `ducklake_file_partition_value` rows
/// through its shared file-registration path, so an output that kept its rows but
/// lost its partition row is not a representable state there.
///
/// This asserts on CATALOG state, not query results: a merged file that dropped its
/// partition assignment still returns every row (zone maps keep pruning), so reads
/// look healthy while partition-value pruning is silently and permanently gone.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn merge_preserves_partition_assignment_postgres() {
    use datafusion_ducklake::partition::PartitionTransform;
    use datafusion_ducklake::{ColumnDef, WriteMode};

    let (pool, _container) = spin_up_postgres().await.unwrap();
    let temp = TempDir::new().unwrap();
    let data = temp.path().join("data");
    std::fs::create_dir_all(&data).unwrap();
    let store: ObjStore = Arc::new(LocalFileSystem::new());
    let cat_name = "cat";
    let cat = MulticatalogManager::new(pool.clone())
        .create_catalog(cat_name)
        .await
        .unwrap();

    // Create `public.t(id, val)` with no data, then partition it by `val`, so every
    // data file that follows is written into a partition.
    let writer = writer_for(&pool, cat, &data).await;
    let cols = vec![
        ColumnDef::from_arrow("id", &DataType::Int32, false).unwrap(),
        ColumnDef::from_arrow("val", &DataType::Int32, false).unwrap(),
    ];
    let setup = writer
        .begin_write_transaction("public", "t", &cols, WriteMode::Replace)
        .unwrap();
    writer
        .publish_snapshot(
            setup.table_id,
            "public",
            "t",
            setup.snapshot_id,
            WriteMode::Replace,
            setup.base_snapshot_id,
            &cols,
            &setup.column_ids,
        )
        .unwrap();
    writer
        .set_partition_spec(
            setup.table_id,
            &[("val".to_string(), PartitionTransform::Identity)],
        )
        .unwrap();
    let table_id = setup.table_id;

    // Two appends, each landing rows in BOTH partitions (val=1 and val=2), so each
    // append writes one small file per partition: four files over two origin
    // snapshots. Spanning origin snapshots also drives the partial-data-file path,
    // so each merged output carries a `partial_max` as well as its partition.
    for id in [1, 2] {
        DuckLakeTableWriter::new(writer_for(&pool, cat, &data).await, store.clone())
            .unwrap()
            .append_table("public", "t", &[batch(vec![id, id + 10], vec![1, 2])])
            .await
            .unwrap();
    }
    assert_eq!(
        live_files(&pool, cat_name).await.len(),
        4,
        "two appends x two partitions = four files"
    );

    // Merge with a target large enough to bin everything that is legal to bin.
    let result = with_writable_table(&pool, cat, cat_name, &data, |t, s| async move {
        t.merge_adjacent_files(
            &s,
            MergeOptions {
                target_file_size: 1 << 30,
                max_merged_files: 1024,
                min_file_size: 0,
            },
        )
        .await
    })
    .await;
    assert!(result.did_work(), "the small files must be compacted");
    assert_eq!(result.files_processed, 4, "all four sources merged");
    assert_eq!(
        result.files_created, 2,
        "one output per partition, never one across partitions"
    );

    // The load-bearing assertion: read the merged files' partition assignment back
    // out of the catalog. One live file per partition, each with a non-NULL
    // `partition_id` and its own `ducklake_file_partition_value` row.
    let live_after: Vec<(Option<i64>, Option<String>, Option<i64>)> = sqlx::query(
        "SELECT df.partition_id, fpv.partition_value, df.partial_max
         FROM ducklake_data_file AS df
         LEFT JOIN ducklake_file_partition_value AS fpv
           ON fpv.data_file_id = df.data_file_id
         WHERE df.table_id = $1 AND df.end_snapshot IS NULL
         ORDER BY fpv.partition_value",
    )
    .bind(table_id)
    .fetch_all(&pool)
    .await
    .unwrap()
    .into_iter()
    .map(|r| {
        (
            r.try_get::<Option<i64>, _>(0).unwrap(),
            r.try_get::<Option<String>, _>(1).unwrap(),
            r.try_get::<Option<i64>, _>(2).unwrap(),
        )
    })
    .collect();

    assert_eq!(
        live_after.len(),
        2,
        "exactly one merged file per partition, each with exactly one partition \
         value row: {live_after:?}"
    );
    let values: Vec<Option<String>> = live_after.iter().map(|(_, v, _)| v.clone()).collect();
    assert_eq!(
        values,
        vec![Some("1".to_string()), Some("2".to_string())],
        "each merged file keeps its own partition value: {live_after:?}"
    );
    for (partition_id, value, partial_max) in &live_after {
        assert!(
            partition_id.is_some(),
            "a merged file of a partitioned table must keep its partition_id \
             (value={value:?})"
        );
        assert!(
            partial_max.is_some(),
            "an output merged across origin snapshots is a partial file, and must \
             carry its partition alongside partial_max (value={value:?})"
        );
    }

    // Every partitioned live file agrees with the table's live partition spec, so a
    // later merge pass still sees well-formed partition groups to bin within.
    let spec_id = writer
        .live_partition_spec(table_id)
        .unwrap()
        .expect("table is partitioned")
        .partition_id;
    for (partition_id, _, _) in &live_after {
        assert_eq!(
            *partition_id,
            Some(spec_id),
            "merged files must stay on the live partition spec"
        );
    }

    // And the rows survive intact — this passes both before and after the fix,
    // which is exactly why the catalog assertions above are the real test.
    assert_eq!(
        read_rows(&pool, cat_name, None).await,
        vec![(1, 1), (2, 1), (11, 2), (12, 2)]
    );
}

/// The Postgres counterpart of the SQLite `rewrite_preserves_partition_assignment`
/// test. A delete-driven rewrite commits through the same `commit_compaction`
/// output-registration path as a merge, so it lost the partition assignment in
/// exactly the same way. The output holds a subset of one source file's rows, so it
/// belongs to precisely that file's partition — official DuckLake takes the
/// partition from `source_files[0]` on this path too.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn rewrite_preserves_partition_assignment_postgres() {
    use datafusion_ducklake::partition::PartitionTransform;
    use datafusion_ducklake::{ColumnDef, WriteMode};

    let (pool, _container) = spin_up_postgres().await.unwrap();
    let temp = TempDir::new().unwrap();
    let data = temp.path().join("data");
    std::fs::create_dir_all(&data).unwrap();
    let store: ObjStore = Arc::new(LocalFileSystem::new());
    let cat_name = "cat";
    let cat = MulticatalogManager::new(pool.clone())
        .create_catalog(cat_name)
        .await
        .unwrap();

    let writer = writer_for(&pool, cat, &data).await;
    let cols = vec![
        ColumnDef::from_arrow("id", &DataType::Int32, false).unwrap(),
        ColumnDef::from_arrow("val", &DataType::Int32, false).unwrap(),
    ];
    let setup = writer
        .begin_write_transaction("public", "t", &cols, WriteMode::Replace)
        .unwrap();
    writer
        .publish_snapshot(
            setup.table_id,
            "public",
            "t",
            setup.snapshot_id,
            WriteMode::Replace,
            setup.base_snapshot_id,
            &cols,
            &setup.column_ids,
        )
        .unwrap();
    writer
        .set_partition_spec(
            setup.table_id,
            &[("val".to_string(), PartitionTransform::Identity)],
        )
        .unwrap();
    let table_id = setup.table_id;
    let live_spec_id = writer
        .live_partition_spec(table_id)
        .unwrap()
        .expect("table is partitioned")
        .partition_id;

    // One partition (val=1) holding ten rows, so the rewrite has a single source.
    DuckLakeTableWriter::new(writer_for(&pool, cat, &data).await, store.clone())
        .unwrap()
        .append_table("public", "t", &[batch((1..=10).collect(), vec![1; 10])])
        .await
        .unwrap();

    // Delete eight of ten rows, then rewrite past the 0.5 threshold.
    {
        let provider = MulticatalogProvider::with_pool(pool.clone(), cat_name)
            .await
            .unwrap();
        let catalog =
            DuckLakeCatalog::with_writer(Arc::new(provider), writer_for(&pool, cat, &data).await)
                .unwrap();
        let ctx = SessionContext::new();
        ctx.register_catalog(cat_name, Arc::new(catalog));
        ctx.sql(&format!("DELETE FROM {cat_name}.public.t WHERE id <= 8"))
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
    }
    let result = with_writable_table(&pool, cat, cat_name, &data, |t, s| async move {
        t.rewrite_data_files(
            &s,
            RewriteOptions {
                delete_threshold: 0.5,
                ..RewriteOptions::default()
            },
        )
        .await
    })
    .await;
    assert_eq!(result.files_created, 1, "the live rows are rewritten");

    // The rewritten file keeps the partition it came from.
    let (partition_id, value): (Option<i64>, Option<String>) = sqlx::query(
        "SELECT df.partition_id, fpv.partition_value
         FROM ducklake_data_file AS df
         LEFT JOIN ducklake_file_partition_value AS fpv
           ON fpv.data_file_id = df.data_file_id
         WHERE df.table_id = $1 AND df.end_snapshot IS NULL",
    )
    .bind(table_id)
    .fetch_one(&pool)
    .await
    .map(|r| {
        (
            r.try_get::<Option<i64>, _>(0).unwrap(),
            r.try_get::<Option<String>, _>(1).unwrap(),
        )
    })
    .unwrap();
    assert_eq!(
        partition_id,
        Some(live_spec_id),
        "the rewritten file must keep its partition_id"
    );
    assert_eq!(
        value,
        Some("1".to_string()),
        "the rewritten file must keep its partition value"
    );
    // And the surviving rows read back.
    assert_eq!(
        read_rows(&pool, cat_name, None).await,
        vec![(9, 1), (10, 1)]
    );
}
