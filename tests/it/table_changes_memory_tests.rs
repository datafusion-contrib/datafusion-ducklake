//! The change feeds run within the session's memory pool.
//!
//! `ducklake_table_changes` streams its insertions and deletions and pairs
//! them through hash joins built on `(snapshot_id, rowid)` keys alone, as
//! official DuckLake's macro joins them; the deletion side holds one data
//! file's delete positions at a time. All of that is charged to the session's
//! memory pool: a window whose rows far exceed the pool completes, and one
//! whose pairing keys exceed it fails with a resources error instead of
//! growing past it.

#![cfg(all(feature = "write-sqlite", feature = "metadata-sqlite"))]

use std::sync::Arc;

use arrow::array::{Array, Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::error::DataFusionError;
use datafusion::execution::memory_pool::GreedyMemoryPool;
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::prelude::{SessionConfig, SessionContext};
use futures::StreamExt;
use tempfile::TempDir;

use crate::test_alloc::measure;

use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeTableWriter, DuckLakeWriteOptions, MetadataProvider, MetadataWriter,
    SqliteMetadataProvider, SqliteMetadataWriter, execute_ducklake_sql,
    register_ducklake_functions,
};

const FILES: i64 = 20;
const ROWS_PER_FILE: i64 = 10_000;

async fn open(dir: &TempDir) -> (Arc<dyn MetadataWriter>, Arc<dyn MetadataProvider>) {
    let url = format!("sqlite:{}?mode=rwc", dir.path().join("c.db").display());
    let writer = SqliteMetadataWriter::new_with_init(&url).await.unwrap();
    let provider = SqliteMetadataProvider::new(&url).await.unwrap();
    (Arc::new(writer), Arc::new(provider))
}

/// A session over the catalog in `dir`, its memory pool capped at
/// `pool_bytes` when given. Writes are not inlined.
async fn session(
    dir: &TempDir,
    pool_bytes: Option<usize>,
) -> (SessionContext, DuckLakeCatalog, Arc<dyn MetadataProvider>) {
    session_inlining(dir, pool_bytes, 0).await
}

/// [`session`], inlining writes of at most `inlining_row_limit` rows.
async fn session_inlining(
    dir: &TempDir,
    pool_bytes: Option<usize>,
    inlining_row_limit: usize,
) -> (SessionContext, DuckLakeCatalog, Arc<dyn MetadataProvider>) {
    let (writer, provider) = open(dir).await;
    let mut runtime = RuntimeEnvBuilder::new();
    if let Some(bytes) = pool_bytes {
        runtime = runtime.with_memory_pool(Arc::new(GreedyMemoryPool::new(bytes)));
    }
    let ctx = SessionContext::new_with_config_rt(
        SessionConfig::new().with_target_partitions(4),
        runtime.build_arc().unwrap(),
    );
    let make = || {
        DuckLakeCatalog::with_writer(Arc::clone(&provider), Arc::clone(&writer))
            .unwrap()
            .with_write_options(
                DuckLakeWriteOptions::default().with_data_inlining_row_limit(inlining_row_limit),
            )
    };
    DuckLakeCatalog::register(&ctx, "lake", make());
    register_ducklake_functions(&ctx, Arc::clone(&provider));
    (ctx, make(), provider)
}

async fn sql(dir: &TempDir, statement: &str) {
    let (ctx, catalog, _) = session(dir, None).await;
    execute_ducklake_sql(&ctx, &catalog, statement)
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
}

/// `t(id, val, s)` written as `files` files of `rows` rows, ids `0..`.
async fn seed(dir: &TempDir, files: i64, rows: i64) {
    let data = dir.path().join("data");
    std::fs::create_dir_all(&data).unwrap();
    open(dir)
        .await
        .0
        .set_data_path(data.to_str().unwrap())
        .unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("val", DataType::Int64, false),
        Field::new("s", DataType::Utf8, false),
    ]));
    for f in 0..files {
        let ids: Vec<i64> = (0..rows).map(|i| f * rows + i).collect();
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int64Array::from(ids.clone())),
                Arc::new(Int64Array::from(ids.clone())),
                Arc::new(StringArray::from(
                    ids.iter()
                        .map(|i| format!("payload-{i:012}-abcdefghijklmnopqrstuvwxyz"))
                        .collect::<Vec<_>>(),
                )),
            ],
        )
        .unwrap();
        let writer = DuckLakeTableWriter::new(
            open(dir).await.0,
            Arc::new(object_store::local::LocalFileSystem::new()),
        )
        .unwrap();
        if f == 0 {
            writer.write_table("main", "t", &[batch]).await.unwrap();
        } else {
            writer.append_table("main", "t", &[batch]).await.unwrap();
        }
    }
}

/// `change_type → rows` of the window's change feed, streamed under a pool of
/// `pool_bytes`.
async fn change_counts(
    dir: &TempDir,
    pool_bytes: usize,
) -> Result<Vec<(String, i64)>, DataFusionError> {
    let (ctx, _, provider) = session(dir, Some(pool_bytes)).await;
    stream_change_counts(&ctx, provider.as_ref()).await
}

async fn stream_change_counts(
    ctx: &SessionContext,
    provider: &dyn MetadataProvider,
) -> Result<Vec<(String, i64)>, DataFusionError> {
    let head = provider.get_current_snapshot().unwrap();
    let mut stream = ctx
        .sql(&format!(
            "SELECT change_type, id FROM ducklake_table_changes('main.t', 0, {head})"
        ))
        .await?
        .execute_stream()
        .await?;
    let mut counts = std::collections::BTreeMap::<String, i64>::new();
    while let Some(batch) = stream.next().await {
        let batch = batch?;
        let types = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        for t in types.iter() {
            *counts.entry(t.unwrap().to_string()).or_default() += 1;
        }
    }
    Ok(counts.into_iter().collect())
}

fn is_resources_exhausted(error: &DataFusionError) -> bool {
    matches!(error.find_root(), DataFusionError::ResourcesExhausted(_))
}

/// 200k inserted rows, an UPDATE of 20k of them and a DELETE of 10k others:
/// about 12 MB of rows in the window, read whole by the feed.
async fn updated_catalog() -> TempDir {
    let dir = TempDir::new().unwrap();
    seed(&dir, FILES, ROWS_PER_FILE).await;
    sql(
        &dir,
        "UPDATE lake.main.t SET val = val + 1 WHERE id % 10 = 0",
    )
    .await;
    sql(&dir, "DELETE FROM lake.main.t WHERE id % 20 = 1").await;
    dir
}

/// Less than the bytes of the window's rows: [`updated_catalog`]'s inserted
/// rows alone take about 13 MB in Arrow.
const WINDOW_BOUND: usize = 12 << 20;

/// The plan the change feed's table provider returns runs as it stands,
/// without the physical optimizer, as change detection runs the feeds: its
/// pairing joins build on one partition however many the key sides have.
#[tokio::test(flavor = "multi_thread")]
async fn the_change_feed_plan_runs_without_the_physical_optimizer() {
    use datafusion::logical_expr::lit;
    use datafusion::physical_plan::ExecutionPlanProperties;
    let dir = updated_catalog().await;
    let (ctx, _, provider) = session(&dir, None).await;
    let head = provider.get_current_snapshot().unwrap();
    let state = ctx.state();
    let args = [lit("main.t"), lit(0i64), lit(head)];
    let table = ctx
        .table_function("ducklake_table_changes")
        .unwrap()
        .create_table_provider_with_args(datafusion::catalog::TableFunctionArgs::new(&args, &state))
        .unwrap();
    let plan = table.scan(&state, None, &[], None).await.unwrap();
    assert!(
        plan.output_partitioning().partition_count() > 1
            || ctx.state().config().target_partitions() > 1,
        "the scan must run with several partitions to exercise the join's build side"
    );
    let mut stream = datafusion::physical_plan::execute_stream(plan, ctx.task_ctx()).unwrap();
    let mut rows = 0;
    while let Some(batch) = stream.next().await {
        rows += batch.unwrap().num_rows();
    }
    // 200,000 inserts, 20,000 update pre- and postimages each, 10,000 deletes.
    assert_eq!(rows, 250_000);
}

/// A single-threaded runtime: [`measure`] counts the allocations of the
/// thread it runs on, so the plan must run on that thread alone.
fn current_thread() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
}

/// The feed streams: under a pool of a third of the window's rows it
/// completes, and the most it allocates at once stays under
/// [`WINDOW_BOUND`]. A feed that collected the window before returning any of
/// it would hold every row of it at once.
#[test]
fn a_change_feed_larger_than_the_pool_streams_through_it() {
    let rt = current_thread();
    let dir = rt.block_on(updated_catalog());
    let total = FILES * ROWS_PER_FILE;
    // 4 MiB: a third of the window's rows, and far more than its pairing keys.
    let (ctx, _, provider) = rt.block_on(session(&dir, Some(4 << 20)));
    let (counts, measured) = measure(|| rt.block_on(stream_change_counts(&ctx, provider.as_ref())));
    assert_eq!(
        counts.unwrap(),
        vec![
            ("delete".to_string(), total / 20),
            ("insert".to_string(), total),
            ("update_postimage".to_string(), total / 10),
            ("update_preimage".to_string(), total / 10),
        ]
    );
    assert!(
        measured.peak < WINDOW_BOUND,
        "the feed allocated {} bytes at once: the window is not streamed",
        measured.peak
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn pairing_keys_beyond_the_pool_fail_with_a_resources_error() {
    let dir = updated_catalog().await;
    // 20k update keys of 16 bytes each, plus the hash table, do not fit.
    let error = change_counts(&dir, 256 << 10).await.unwrap_err();
    assert!(
        is_resources_exhausted(&error),
        "expected a resources error, got {error}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn delete_positions_are_charged_to_the_pool() {
    let dir = TempDir::new().unwrap();
    seed(&dir, 1, 200_000).await;
    sql(&dir, "DELETE FROM lake.main.t WHERE id % 2 = 0").await;
    let deletions = |pool: usize| {
        let dir = &dir;
        async move {
            let (ctx, _, provider) = session(dir, Some(pool)).await;
            let head = provider.get_current_snapshot().unwrap();
            ctx.sql(&format!(
                "SELECT count(*) FROM ducklake_table_deletions('main.t', 0, {head})"
            ))
            .await?
            .collect()
            .await
        }
    };
    // 100k positions of one file are held while its rows are read.
    let error = deletions(64 << 10).await.unwrap_err();
    assert!(
        is_resources_exhausted(&error),
        "expected a resources error, got {error}"
    );
    let batches = deletions(16 << 20).await.unwrap();
    let count = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!(count, 100_000);
}

/// A data file deleted whole reports its rows without building the set of
/// its positions: under a pool far smaller than that set the feed completes,
/// and the most it allocates at once stays under 2 MiB, less than a hash set
/// of the file's 200k positions takes on its own.
#[test]
fn a_whole_file_deletion_holds_no_position_set() {
    let rt = current_thread();
    let dir = TempDir::new().unwrap();
    rt.block_on(seed(&dir, 1, 200_000));
    rt.block_on(sql(&dir, "DELETE FROM lake.main.t"));
    let (ctx, _, provider) = rt.block_on(session(&dir, Some(64 << 10)));
    let head = provider.get_current_snapshot().unwrap();
    let (batches, measured) = measure(|| {
        rt.block_on(async {
            ctx.sql(&format!(
                "SELECT count(*) FROM ducklake_table_deletions('main.t', 0, {head})"
            ))
            .await?
            .collect()
            .await
        })
    });
    let batches = batches.unwrap();
    let count = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(count.len(), 1);
    assert_eq!(count.value(0), 200_000);
    assert!(
        measured.peak < 2 << 20,
        "the feed allocated {} bytes at once: the deletion holds a position set",
        measured.peak
    );
}

const INLINED_COMMITS: i64 = 200;
const INLINED_ROWS: i64 = 15;
const PARQUET_ROWS: i64 = 20_000;
const INLINED_FILE_DELETE_COMMITS: i64 = 200;

/// A long history of small inlined writes: `t` holds one Parquet file of
/// [`PARQUET_ROWS`] rows, then [`INLINED_COMMITS`] inlined inserts of
/// [`INLINED_ROWS`] rows of about 1 KiB each (about 3 MB in the catalog), a
/// DELETE of every fourth inlined row, and [`INLINED_FILE_DELETE_COMMITS`]
/// inlined deletions of ten Parquet rows each.
async fn inlined_catalog() -> TempDir {
    let dir = TempDir::new().unwrap();
    seed(&dir, 1, PARQUET_ROWS).await;
    let (ctx, catalog, _) = session_inlining(&dir, None, 1000).await;
    let run = |statement: String| {
        let (ctx, catalog) = (&ctx, &catalog);
        async move {
            execute_ducklake_sql(ctx, catalog, &statement)
                .await
                .unwrap()
                .collect()
                .await
                .unwrap();
        }
    };
    let payload = "x".repeat(1000);
    for commit in 0..INLINED_COMMITS {
        let values: Vec<String> = (0..INLINED_ROWS)
            .map(|i| {
                let id = PARQUET_ROWS + commit * INLINED_ROWS + i;
                format!("({id}, {id}, '{payload}{id}')")
            })
            .collect();
        run(format!(
            "INSERT INTO lake.main.t VALUES {}",
            values.join(", ")
        ))
        .await;
    }
    run(format!(
        "DELETE FROM lake.main.t WHERE id >= {PARQUET_ROWS} AND id % 4 = 0"
    ))
    .await;
    for commit in 0..INLINED_FILE_DELETE_COMMITS {
        run(format!(
            "DELETE FROM lake.main.t WHERE id >= {} AND id < {}",
            commit * 10,
            commit * 10 + 10
        ))
        .await;
    }
    dir
}

const INLINED_TOTAL: i64 = INLINED_COMMITS * INLINED_ROWS;
const INLINED_DELETED: i64 = INLINED_TOTAL / 4 + INLINED_FILE_DELETE_COMMITS * 10;

/// The count of rows of `function` over the whole history, under a pool of
/// `pool_bytes`.
async fn feed_count(
    dir: &TempDir,
    function: &str,
    pool_bytes: usize,
) -> Result<i64, DataFusionError> {
    let (ctx, _, provider) = session_inlining(dir, Some(pool_bytes), 1000).await;
    let head = provider.get_current_snapshot().unwrap();
    let batches = ctx
        .sql(&format!(
            "SELECT count(*) FROM {function}('main.t', 0, {head})"
        ))
        .await?
        .collect()
        .await?;
    Ok(batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0))
}

/// The inlined rows and deletions a feed reads from the catalog at planning
/// are charged to the session's pool for as long as the plan holds them: a
/// history of them larger than the pool fails with a resources error, and
/// within a large one the feeds are exact.
#[tokio::test(flavor = "multi_thread")]
async fn inlined_history_is_charged_to_the_pool() {
    let dir = inlined_catalog().await;
    for function in
        ["ducklake_table_changes", "ducklake_table_insertions", "ducklake_table_deletions"]
    {
        let error = feed_count(&dir, function, 1 << 20).await.unwrap_err();
        assert!(
            is_resources_exhausted(&error),
            "{function}: expected a resources error, got {error}"
        );
    }

    let large = 64 << 20;
    assert_eq!(
        feed_count(&dir, "ducklake_table_insertions", large)
            .await
            .unwrap(),
        PARQUET_ROWS + INLINED_TOTAL
    );
    assert_eq!(
        feed_count(&dir, "ducklake_table_deletions", large)
            .await
            .unwrap(),
        INLINED_DELETED
    );
    let (ctx, _, provider) = session_inlining(&dir, Some(large), 1000).await;
    assert_eq!(
        stream_change_counts(&ctx, provider.as_ref()).await.unwrap(),
        vec![
            ("delete".to_string(), INLINED_DELETED),
            ("insert".to_string(), PARQUET_ROWS + INLINED_TOTAL),
        ]
    );
}

/// The inlined deletions of Parquet rows alone, grouped by data file at
/// planning, are charged too.
#[tokio::test(flavor = "multi_thread")]
async fn inlined_file_deletions_are_charged_to_the_pool() {
    let dir = TempDir::new().unwrap();
    seed(&dir, 1, PARQUET_ROWS).await;
    let (ctx, catalog, _) = session_inlining(&dir, None, 1000).await;
    // One inlined deletion per commit: each is a separate change of the
    // window, carrying its data file's path.
    for id in 0..1000 {
        execute_ducklake_sql(
            &ctx,
            &catalog,
            &format!("DELETE FROM lake.main.t WHERE id = {id}"),
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    }
    let error = feed_count(&dir, "ducklake_table_deletions", 128 << 10)
        .await
        .unwrap_err();
    assert!(
        is_resources_exhausted(&error),
        "expected a resources error, got {error}"
    );
    assert_eq!(
        feed_count(&dir, "ducklake_table_deletions", 64 << 20)
            .await
            .unwrap(),
        1000
    );
}
