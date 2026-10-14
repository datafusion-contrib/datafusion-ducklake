//! The insert side of official DuckLake's conflict check
//! (`CheckForConflicts` in `ducklake_transaction_state.cpp`): a commit that
//! inserts into a table, as Parquet files or inlined rows, refuses when another
//! transaction deleted from the table, from its files or its inlined rows,
//! since the insert's base snapshot. A concurrent insert does not conflict.
//! A positional delete file (`set_delete_file`) refuses a concurrent insert, as
//! official's delete does.

#![cfg(feature = "write")]

use std::sync::Arc;

use arrow::array::{Int32Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::prelude::SessionContext;
use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeError, DuckLakeTableWriter, DuckLakeWriteOptions, MetadataProvider,
    MetadataWriter, WriteMode, execute_ducklake_sql,
};
use object_store::local::LocalFileSystem;

struct Side {
    writer: Arc<dyn MetadataWriter>,
    provider: Arc<dyn MetadataProvider>,
    /// The lake's data directory.
    data: std::path::PathBuf,
}

/// The Parquet objects under `dir`.
fn parquet_objects(dir: &std::path::Path) -> usize {
    let mut count = 0;
    let mut pending = vec![dir.to_path_buf()];
    while let Some(dir) = pending.pop() {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                pending.push(path);
            } else if path.extension().is_some_and(|ext| ext == "parquet") {
                count += 1;
            }
        }
    }
    count
}

fn schema() -> Schema {
    Schema::new(vec![Field::new("a", DataType::Int32, true)])
}

fn batch(values: &[i32]) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(schema()),
        vec![Arc::new(Int32Array::from(values.to_vec()))],
    )
    .unwrap()
}

fn table_writer(side: &Side, limit: usize) -> DuckLakeTableWriter {
    DuckLakeTableWriter::new(Arc::clone(&side.writer), Arc::new(LocalFileSystem::new()))
        .unwrap()
        .with_options(&DuckLakeWriteOptions::default().with_data_inlining_row_limit(limit))
}

async fn sql(side: &Side, statement: &str) {
    let ctx = SessionContext::new();
    let catalog =
        DuckLakeCatalog::with_writer(Arc::clone(&side.provider), Arc::clone(&side.writer)).unwrap();
    DuckLakeCatalog::register(&ctx, "lake", catalog);
    let handle =
        DuckLakeCatalog::with_writer(Arc::clone(&side.provider), Arc::clone(&side.writer)).unwrap();
    execute_ducklake_sql(&ctx, &handle, statement)
        .await
        .unwrap_or_else(|e| panic!("`{statement}`: {e}"))
        .collect()
        .await
        .unwrap_or_else(|e| panic!("`{statement}`: {e}"));
}

async fn rows(side: &Side) -> Vec<i32> {
    let ctx = SessionContext::new();
    let snapshot = side.provider.get_current_snapshot().unwrap();
    let catalog = DuckLakeCatalog::with_snapshot(Arc::clone(&side.provider), snapshot).unwrap();
    ctx.register_catalog("lake", Arc::new(catalog));
    let batches = ctx
        .sql("SELECT a FROM lake.main.t ORDER BY a")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    batches
        .iter()
        .flat_map(|b| {
            b.column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .iter()
                .flatten()
                .collect::<Vec<_>>()
        })
        .collect()
}

fn table_id(side: &Side) -> i64 {
    let snapshot = side.provider.get_current_snapshot().unwrap();
    let schema = side
        .provider
        .get_schema_by_name("main", snapshot)
        .unwrap()
        .unwrap();
    side.provider
        .get_table_by_name(schema.schema_id, "t", snapshot)
        .unwrap()
        .unwrap()
        .table_id
}

fn expect_conflict(result: Result<impl std::fmt::Debug, DuckLakeError>, described: &str) {
    match result {
        Err(DuckLakeError::Conflict(message)) => assert!(
            message.contains(described),
            "expected a conflict saying `{described}`, got `{message}`"
        ),
        other => panic!("expected a conflict saying `{described}`, got {other:?}"),
    }
}

/// A Parquet append whose base predates a concurrent `DELETE` refuses.
async fn append_after_concurrent_delete(side: &Side, limit: usize) {
    table_writer(side, limit)
        .append_table("main", "t", &[batch(&[1, 2, 3])])
        .await
        .unwrap();
    let mut session = table_writer(side, 0)
        .begin_write("main", "t", &schema(), WriteMode::Append)
        .unwrap();
    session.write_batch(&batch(&[10])).unwrap();
    sql(side, "DELETE FROM lake.main.t WHERE a = 1").await;
    let objects = parquet_objects(&side.data);
    // A small `DELETE` is recorded as inlined deletions under the catalog's
    // default inlining limit, a larger one as a delete file: either refuses.
    expect_conflict(session.finish().await, "attempting to insert into table");
    assert_eq!(
        parquet_objects(&side.data),
        objects,
        "the refused append removed the file it uploaded"
    );
    assert_eq!(
        rows(side).await,
        vec![2, 3],
        "the refused append left nothing"
    );
}

/// A staged insert, inlined or Parquet, whose base predates a concurrent
/// `DELETE` refuses.
async fn staged_insert_after_concurrent_delete(side: &Side, limit: usize) {
    table_writer(side, limit)
        .append_table("main", "t", &[batch(&[1, 2, 3])])
        .await
        .unwrap();
    let writer = table_writer(side, limit);
    let mut transaction = writer.transaction();
    transaction
        .stage_write("main", "t", &schema(), WriteMode::Append, &[batch(&[10])])
        .await
        .unwrap();
    sql(side, "DELETE FROM lake.main.t WHERE a = 1").await;
    expect_conflict(
        transaction.commit().await,
        "attempting to insert into table",
    );
    assert_eq!(
        rows(side).await,
        vec![2, 3],
        "the refused insert left nothing"
    );
}

/// A SQL `INSERT` planned before a concurrent `DELETE` of its table commits
/// refuses, though its write begins after the `DELETE`: official's insert is
/// checked from the snapshot its statement read.
async fn sql_insert_planned_before_concurrent_delete(side: &Side, limit: usize) {
    use datafusion::physical_plan::collect;
    table_writer(side, limit)
        .append_table("main", "t", &[batch(&[1, 2, 3])])
        .await
        .unwrap();
    let ctx = SessionContext::new();
    let catalog =
        DuckLakeCatalog::with_writer(Arc::clone(&side.provider), Arc::clone(&side.writer)).unwrap();
    DuckLakeCatalog::register(&ctx, "lake", catalog);
    let plan = ctx
        .sql("INSERT INTO lake.main.t VALUES (10)")
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    sql(side, "DELETE FROM lake.main.t WHERE a = 1").await;
    let error = collect(plan, ctx.task_ctx()).await.unwrap_err().to_string();
    assert!(
        error.contains("attempting to insert into table"),
        "limit {limit}: {error}"
    );
    assert_eq!(
        rows(side).await,
        vec![2, 3],
        "the refused insert left nothing"
    );
}

/// Concurrent appends commute, inlined or not.
async fn concurrent_appends_commute(side: &Side, limit: usize) {
    table_writer(side, limit)
        .append_table("main", "t", &[batch(&[1])])
        .await
        .unwrap();
    let mut session = table_writer(side, 0)
        .begin_write("main", "t", &schema(), WriteMode::Append)
        .unwrap();
    session.write_batch(&batch(&[3])).unwrap();
    let writer = table_writer(side, limit);
    let mut transaction = writer.transaction();
    transaction
        .stage_write("main", "t", &schema(), WriteMode::Append, &[batch(&[4])])
        .await
        .unwrap();
    table_writer(side, limit)
        .append_table("main", "t", &[batch(&[2])])
        .await
        .unwrap();
    session.finish().await.unwrap();
    transaction.commit().await.unwrap();
    assert_eq!(rows(side).await, vec![1, 2, 3, 4]);
}

/// `set_delete_file` refuses an insert into its table since its base
/// snapshot, as official's delete refuses a concurrent insert.
async fn delete_file_after_concurrent_append(side: &Side) {
    table_writer(side, 0)
        .append_table("main", "t", &[batch(&[1, 2, 3])])
        .await
        .unwrap();
    let base = side.provider.get_current_snapshot().unwrap();
    let table = table_id(side);
    let files = side
        .provider
        .get_table_files_for_select(table, base)
        .unwrap();
    let data_file_id = files[0].data_file_id;
    table_writer(side, 0)
        .append_table("main", "t", &[batch(&[4])])
        .await
        .unwrap();
    let delete = table_writer(side, 0)
        .write_delete_file("main", "t", "ignored", &[0])
        .await
        .unwrap();
    expect_conflict(
        side.writer
            .set_delete_file(table, "main", "t", base, data_file_id, None, base, &delete),
        "inserted into it",
    );
    assert_eq!(rows(side).await, vec![1, 2, 3, 4]);
}

type Open = Box<dyn Fn() -> std::pin::Pin<Box<dyn std::future::Future<Output = Side>>>>;

async fn run_all(open: Open, inlines: bool) {
    let limits: &[usize] = if inlines {
        &[0, 10]
    } else {
        &[0]
    };
    for &limit in limits {
        append_after_concurrent_delete(&open().await, limit).await;
        staged_insert_after_concurrent_delete(&open().await, limit).await;
        sql_insert_planned_before_concurrent_delete(&open().await, limit).await;
        concurrent_appends_commute(&open().await, limit).await;
    }
    delete_file_after_concurrent_append(&open().await).await;
}

#[cfg(feature = "write-sqlite")]
#[tokio::test(flavor = "multi_thread")]
async fn sqlite() {
    use datafusion_ducklake::{SqliteMetadataProvider, SqliteMetadataWriter};
    let temp = Arc::new(tempfile::TempDir::new().unwrap());
    let counter = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    run_all(
        Box::new(move || {
            let temp = Arc::clone(&temp);
            let index = counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Box::pin(async move {
                let dir = temp.path().join(format!("lake{index}"));
                std::fs::create_dir_all(&dir).unwrap();
                let url = format!("sqlite:{}?mode=rwc", dir.join("c.db").display());
                let writer = SqliteMetadataWriter::new_with_init(&url).await.unwrap();
                writer.set_data_path(dir.to_str().unwrap()).unwrap();
                Side {
                    data: dir.clone(),
                    writer: Arc::new(writer),
                    provider: Arc::new(SqliteMetadataProvider::new(&url).await.unwrap()),
                }
            })
        }),
        true,
    )
    .await;
}

#[cfg(feature = "write-duckdb")]
#[tokio::test(flavor = "multi_thread")]
async fn duckdb() {
    use datafusion_ducklake::DuckdbMetadataWriter;
    let temp = Arc::new(tempfile::TempDir::new().unwrap());
    let counter = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    run_all(
        Box::new(move || {
            let temp = Arc::clone(&temp);
            let index = counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Box::pin(async move {
                let dir = temp.path().join(format!("lake{index}"));
                std::fs::create_dir_all(&dir).unwrap();
                let writer =
                    DuckdbMetadataWriter::new_with_init(dir.join("c.duckdb").display().to_string())
                        .unwrap();
                writer.set_data_path(dir.to_str().unwrap()).unwrap();
                let provider = writer.metadata_provider();
                Side {
                    data: dir.clone(),
                    writer: Arc::new(writer),
                    provider: Arc::new(provider),
                }
            })
        }),
        true,
    )
    .await;
}

#[cfg(feature = "write-mysql")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
#[tokio::test(flavor = "multi_thread")]
async fn mysql() {
    use datafusion_ducklake::{MySqlMetadataProvider, MySqlMetadataWriter};
    use testcontainers::runners::AsyncRunner;
    use testcontainers_modules::mysql::Mysql;
    let container = Mysql::default().start().await.unwrap();
    let port = container.get_host_port_ipv4(3306).await.unwrap();
    let temp = Arc::new(tempfile::TempDir::new().unwrap());
    let counter = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    run_all(
        Box::new(move || {
            let temp = Arc::clone(&temp);
            let index = counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Box::pin(async move {
                let admin =
                    sqlx::MySqlPool::connect(&format!("mysql://root@127.0.0.1:{port}/test"))
                        .await
                        .unwrap();
                sqlx::query(sqlx::AssertSqlSafe(format!("CREATE DATABASE lake{index}")))
                    .execute(&admin)
                    .await
                    .unwrap();
                let dir = temp.path().join(format!("lake{index}"));
                std::fs::create_dir_all(&dir).unwrap();
                let url = format!("mysql://root@127.0.0.1:{port}/lake{index}");
                let writer = MySqlMetadataWriter::new_with_init(&url).await.unwrap();
                writer.set_data_path(dir.to_str().unwrap()).unwrap();
                // A writer without global settings takes the limit per table;
                // the tests pass it on each write.
                Side {
                    data: dir.clone(),
                    writer: Arc::new(writer),
                    provider: Arc::new(MySqlMetadataProvider::new(&url).await.unwrap()),
                }
            })
        }),
        true,
    )
    .await;
}

#[cfg(feature = "multicatalog-postgres")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
#[tokio::test(flavor = "multi_thread")]
async fn multicatalog() {
    use datafusion_ducklake::{MulticatalogManager, MulticatalogProvider, PostgresMetadataWriter};
    use testcontainers::runners::AsyncRunner;
    use testcontainers_modules::postgres::Postgres;
    let container = Postgres::default().start().await.unwrap();
    let port = container.get_host_port_ipv4(5432).await.unwrap();
    let temp = Arc::new(tempfile::TempDir::new().unwrap());
    let counter = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    run_all(
        Box::new(move || {
            let temp = Arc::clone(&temp);
            let index = counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Box::pin(async move {
                let admin = sqlx::PgPool::connect(&format!(
                    "postgresql://postgres:postgres@127.0.0.1:{port}/postgres"
                ))
                .await
                .unwrap();
                sqlx::query(sqlx::AssertSqlSafe(format!("CREATE DATABASE lake{index}")))
                    .execute(&admin)
                    .await
                    .unwrap();
                let dir = temp.path().join(format!("lake{index}"));
                std::fs::create_dir_all(&dir).unwrap();
                let url = format!("postgresql://postgres:postgres@127.0.0.1:{port}/lake{index}");
                let pool = sqlx::PgPool::connect(&url).await.unwrap();
                datafusion_ducklake::initialize_multicatalog_schema(&pool)
                    .await
                    .unwrap();
                let catalog = MulticatalogManager::new(pool.clone())
                    .create_catalog("lake")
                    .await
                    .unwrap();
                let writer = PostgresMetadataWriter::with_pool(pool.clone(), catalog)
                    .await
                    .unwrap();
                writer.set_data_path(dir.to_str().unwrap()).unwrap();
                Side {
                    data: dir.clone(),
                    writer: Arc::new(writer),
                    provider: Arc::new(
                        MulticatalogProvider::with_pool(pool, "lake").await.unwrap(),
                    ),
                }
            })
        }),
        true,
    )
    .await;
}
