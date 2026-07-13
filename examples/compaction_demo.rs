//! Demo of explicit DuckLake compaction on the single-catalog (SQLite + LocalFS)
//! write path: `DuckLakeTable::merge_adjacent_files` and `rewrite_data_files`.
//!
//! Scenario:
//!   Table `t`  — three INSERTs -> three small files; `merge_adjacent_files`
//!                coalesces them into one merged (partial) file. Query results are
//!                unchanged and time travel to the pre-merge snapshot still works.
//!   Table `t2` — one INSERT of ten rows, then DELETE most of them;
//!                `rewrite_data_files` rewrites the file to hold only the live
//!                rows, retiring the old data + delete files.
//!
//! Run with:
//!     cargo run --no-default-features --features write-sqlite,metadata-sqlite \
//!         --example compaction_demo

use std::sync::Arc;

use arrow::array::Int32Array;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::prelude::*;
use object_store::local::LocalFileSystem;
use sqlx::Row;
use sqlx::sqlite::SqlitePool;

use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeTable, DuckLakeTableWriter, MergeOptions, MetadataWriter,
    RewriteOptions, SqliteMetadataProvider, SqliteMetadataWriter,
};

type ObjStore = Arc<dyn object_store::ObjectStore>;

fn table_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("val", DataType::Int32, false),
    ]))
}

fn batch(ids: Vec<i32>, vals: Vec<i32>) -> RecordBatch {
    RecordBatch::try_new(
        table_schema(),
        vec![Arc::new(Int32Array::from(ids)), Arc::new(Int32Array::from(vals))],
    )
    .unwrap()
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> anyhow::Result<()> {
    let temp = tempfile::TempDir::new()?;
    let db_path = temp.path().join("catalog.db");
    let data_path = temp.path().join("data");
    std::fs::create_dir_all(&data_path)?;
    let conn = format!("sqlite:{}?mode=rwc", db_path.display());
    let ro_conn = format!("sqlite:{}", db_path.display());
    let os: ObjStore = Arc::new(LocalFileSystem::new());

    let writer = SqliteMetadataWriter::new_with_init(&conn).await?;
    writer.set_data_path(data_path.to_str().unwrap())?;
    let pool = SqlitePool::connect(&ro_conn).await?;

    // ===================================================================
    // merge_adjacent_files: three small files -> one merged partial file
    // ===================================================================
    DuckLakeTableWriter::new(
        Arc::new(SqliteMetadataWriter::new(&conn).await?),
        os.clone(),
    )?
    .write_table("main", "t", &[batch(vec![1, 2], vec![10, 20])])
    .await?;
    for (ids, vals) in [(vec![3, 4], vec![30, 40]), (vec![5, 6], vec![50, 60])] {
        DuckLakeTableWriter::new(
            Arc::new(SqliteMetadataWriter::new(&conn).await?),
            os.clone(),
        )?
        .append_table("main", "t", &[batch(ids, vals)])
        .await?;
    }
    let pre_merge_snapshot = max_snapshot(&pool).await?;
    println!("== merge_adjacent_files ==");
    println!("after 3 inserts:");
    println!("  live files : {}", live_data_files(&pool, "t").await?);
    println!("  rows       : {:?}", read_rows(&ro_conn, "t", None).await?);

    let result = with_writable_table(&conn, "t", |table, state| async move {
        table
            .merge_adjacent_files(&state, MergeOptions::default())
            .await
    })
    .await?;
    println!("merge_adjacent_files -> {result:?}");
    println!("  live files : {}", live_data_files(&pool, "t").await?);
    println!(
        "  partial_max: {:?}",
        partial_max_of_live(&pool, "t").await?
    );
    println!("  rows       : {:?}", read_rows(&ro_conn, "t", None).await?);
    println!(
        "  time travel @{pre_merge_snapshot}: {:?}",
        read_rows(&ro_conn, "t", Some(pre_merge_snapshot)).await?
    );

    // ===================================================================
    // rewrite_data_files: drop a file's deleted rows (insert-only source)
    // ===================================================================
    DuckLakeTableWriter::new(
        Arc::new(SqliteMetadataWriter::new(&conn).await?),
        os.clone(),
    )?
    .write_table(
        "main",
        "t2",
        &[batch((1..=10).collect(), (1..=10).map(|v| v * 10).collect())],
    )
    .await?;
    {
        // Delete 8 of the 10 rows, producing a data file that is ~80% tombstoned.
        let ctx = writable_ctx(&conn).await?;
        ctx.sql("DELETE FROM ducklake.main.t2 WHERE id <= 8")
            .await?
            .collect()
            .await?;
    }
    println!("\n== rewrite_data_files ==");
    println!("after DELETE WHERE id <= 8:");
    println!("  live files : {}", live_data_files(&pool, "t2").await?);
    println!(
        "  rows       : {:?}",
        read_rows(&ro_conn, "t2", None).await?
    );

    let result = with_writable_table(&conn, "t2", |table, state| async move {
        table
            .rewrite_data_files(
                &state,
                RewriteOptions {
                    delete_threshold: 0.5,
                },
            )
            .await
    })
    .await?;
    println!("rewrite_data_files(threshold=0.5) -> {result:?}");
    println!(
        "  live files       : {}",
        live_data_files(&pool, "t2").await?
    );
    println!(
        "  live delete files: {}",
        live_delete_files(&pool, "t2").await?
    );
    println!(
        "  rows             : {:?}",
        read_rows(&ro_conn, "t2", None).await?
    );

    Ok(())
}

/// Downcast the writable `main.<table>` provider to a `DuckLakeTable` and run `op`.
async fn with_writable_table<T, F, Fut>(conn: &str, table: &str, op: F) -> anyhow::Result<T>
where
    F: FnOnce(DuckLakeTable, datafusion::execution::SessionState) -> Fut,
    Fut: std::future::Future<Output = datafusion_ducklake::Result<T>>,
{
    let ctx = writable_ctx(conn).await?;
    let provider = ctx
        .catalog("ducklake")
        .unwrap()
        .schema("main")
        .unwrap()
        .table(table)
        .await?
        .unwrap();
    let dl = (provider.as_ref() as &dyn std::any::Any)
        .downcast_ref::<DuckLakeTable>()
        .expect("provider is a DuckLakeTable")
        .clone();
    Ok(op(dl, ctx.state()).await?)
}

async fn writable_ctx(conn: &str) -> anyhow::Result<SessionContext> {
    let writer = SqliteMetadataWriter::new(conn).await?;
    let provider = SqliteMetadataProvider::new(conn).await?;
    let catalog = DuckLakeCatalog::with_writer(Arc::new(provider), Arc::new(writer))?;
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));
    Ok(ctx)
}

/// Read `(id, val)` from `main.<table>`, optionally as of `snapshot` (time travel).
async fn read_rows(
    ro_conn: &str,
    table: &str,
    snapshot: Option<i64>,
) -> anyhow::Result<Vec<(i32, i32)>> {
    let provider = SqliteMetadataProvider::new(ro_conn).await?;
    let catalog = match snapshot {
        Some(s) => DuckLakeCatalog::with_snapshot(Arc::new(provider), s)?,
        None => DuckLakeCatalog::new(provider)?,
    };
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));
    let batches = ctx
        .sql(&format!(
            "SELECT id, val FROM ducklake.main.{table} ORDER BY id"
        ))
        .await?
        .collect()
        .await?;
    let mut rows = Vec::new();
    for b in &batches {
        let ids = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        let vals = b.column(1).as_any().downcast_ref::<Int32Array>().unwrap();
        for i in 0..b.num_rows() {
            rows.push((ids.value(i), vals.value(i)));
        }
    }
    Ok(rows)
}

async fn scalar(pool: &SqlitePool, sql: &str, table: &str) -> anyhow::Result<i64> {
    Ok(sqlx::query(sql)
        .bind(table)
        .fetch_one(pool)
        .await?
        .try_get::<i64, _>(0)?)
}

async fn live_data_files(pool: &SqlitePool, table: &str) -> anyhow::Result<i64> {
    scalar(
        pool,
        "SELECT COUNT(*) FROM ducklake_data_file df
         JOIN ducklake_table t ON t.table_id = df.table_id
         WHERE t.table_name = ? AND df.end_snapshot IS NULL",
        table,
    )
    .await
}

async fn live_delete_files(pool: &SqlitePool, table: &str) -> anyhow::Result<i64> {
    scalar(
        pool,
        "SELECT COUNT(*) FROM ducklake_delete_file df
         JOIN ducklake_table t ON t.table_id = df.table_id
         WHERE t.table_name = ? AND df.end_snapshot IS NULL",
        table,
    )
    .await
}

async fn partial_max_of_live(pool: &SqlitePool, table: &str) -> anyhow::Result<Option<i64>> {
    Ok(sqlx::query(
        "SELECT df.partial_max FROM ducklake_data_file df
         JOIN ducklake_table t ON t.table_id = df.table_id
         WHERE t.table_name = ? AND df.end_snapshot IS NULL LIMIT 1",
    )
    .bind(table)
    .fetch_one(pool)
    .await?
    .try_get::<Option<i64>, _>(0)?)
}

async fn max_snapshot(pool: &SqlitePool) -> anyhow::Result<i64> {
    Ok(
        sqlx::query("SELECT MAX(snapshot_id) FROM ducklake_snapshot")
            .fetch_one(pool)
            .await?
            .try_get::<i64, _>(0)?,
    )
}
