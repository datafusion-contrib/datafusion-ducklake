//! Deletion inlining on the SQLite backend: small SQL `DELETE`s and
//! `UPDATE`s of Parquet rows go to `ducklake_inlined_delete_<table_id>`, a
//! flush turns them into a delete file with per-row snapshots, and every
//! snapshot reads the same rows throughout.

#![cfg(all(feature = "write-sqlite", feature = "metadata-sqlite"))]

use std::sync::Arc;

use arrow::array::{Array, Int32Array, Int64Array, RecordBatch, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::prelude::*;
use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeTable, DuckLakeTableWriter, DuckLakeWriteOptions, MetadataProvider,
    MetadataWriter, SqliteMetadataProvider, SqliteMetadataWriter, is_conflict,
};
use object_store::local::LocalFileSystem;
use sqlx::Row;
use sqlx::sqlite::SqlitePool;
use tempfile::TempDir;

type Rows = Vec<(i32, i32)>;

struct Lake {
    tmp: TempDir,
}

impl Lake {
    async fn new() -> Self {
        let tmp = TempDir::new().unwrap();
        std::fs::create_dir_all(tmp.path().join("data")).unwrap();
        let writer = SqliteMetadataWriter::new_with_init(&Self::url(&tmp))
            .await
            .unwrap();
        writer
            .set_data_path(tmp.path().join("data").to_str().unwrap())
            .unwrap();
        Self {
            tmp,
        }
    }

    fn url(tmp: &TempDir) -> String {
        format!("sqlite:{}?mode=rwc", tmp.path().join("test.db").display())
    }

    async fn writer(&self) -> Arc<SqliteMetadataWriter> {
        Arc::new(
            SqliteMetadataWriter::new(&Self::url(&self.tmp))
                .await
                .unwrap(),
        )
    }

    async fn provider(&self) -> Arc<SqliteMetadataProvider> {
        Arc::new(
            SqliteMetadataProvider::new(&Self::url(&self.tmp))
                .await
                .unwrap(),
        )
    }

    async fn seed(&self, ids: Vec<i32>) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("val", DataType::Int32, false),
        ]));
        let vals: Vec<i32> = ids.iter().map(|id| id * 10).collect();
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int32Array::from(ids)), Arc::new(Int32Array::from(vals))],
        )
        .unwrap();
        DuckLakeTableWriter::new(self.writer().await, Arc::new(LocalFileSystem::new()))
            .unwrap()
            .append_table("main", "t", &[batch])
            .await
            .unwrap();
    }

    async fn ctx(&self, limit: usize, snapshot: Option<i64>) -> SessionContext {
        let provider = self.provider().await;
        let catalog = match snapshot {
            Some(snapshot) => DuckLakeCatalog::with_snapshot(provider, snapshot).unwrap(),
            None => DuckLakeCatalog::with_writer(provider, self.writer().await)
                .unwrap()
                .with_write_options(
                    DuckLakeWriteOptions::default().with_data_inlining_row_limit(limit),
                ),
        };
        let ctx = SessionContext::new();
        ctx.register_catalog("lake", Arc::new(catalog));
        ctx
    }

    async fn exec(&self, sql: &str, limit: usize) -> u64 {
        let batches = self
            .ctx(limit, None)
            .await
            .sql(sql)
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap()
            .value(0)
    }

    async fn head(&self) -> i64 {
        self.provider().await.get_current_snapshot().unwrap()
    }

    async fn rows(&self, snapshot: i64) -> Rows {
        let batches = self
            .ctx(0, Some(snapshot))
            .await
            .sql("SELECT id, val FROM lake.main.t ORDER BY id")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let mut out = Vec::new();
        for b in &batches {
            let ids = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
            let vals = b.column(1).as_any().downcast_ref::<Int32Array>().unwrap();
            for r in 0..b.num_rows() {
                out.push((ids.value(r), vals.value(r)));
            }
        }
        out
    }

    async fn count(&self, snapshot: i64) -> i64 {
        let batches = self
            .ctx(0, Some(snapshot))
            .await
            .sql("SELECT COUNT(*) FROM lake.main.t")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0)
    }

    async fn record(&self, history: &mut Vec<(i64, Rows, i64)>) {
        let head = self.head().await;
        history.push((head, self.rows(head).await, self.count(head).await));
    }

    async fn check(&self, history: &[(i64, Rows, i64)]) {
        for (snapshot, rows, count) in history {
            assert_eq!(&self.rows(*snapshot).await, rows, "snapshot {snapshot}");
            assert_eq!(self.count(*snapshot).await, *count, "snapshot {snapshot}");
        }
    }

    async fn pool(&self) -> SqlitePool {
        SqlitePool::connect(&Self::url(&self.tmp)).await.unwrap()
    }

    /// `(file_id, row_id, begin_snapshot)` of every inlined deletion.
    async fn inline_file_deletes(&self) -> Vec<(i64, i64, i64)> {
        let pool = self.pool().await;
        let table_id: i64 = sqlx::query_scalar(
            "SELECT table_id FROM ducklake_table WHERE table_name = 't' AND end_snapshot IS NULL",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        let table = format!("ducklake_inlined_delete_{table_id}");
        let exists: Option<i64> =
            sqlx::query_scalar("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?")
                .bind(&table)
                .fetch_optional(&pool)
                .await
                .unwrap();
        if exists.is_none() {
            return Vec::new();
        }
        sqlx::query(sqlx::AssertSqlSafe(format!(
            "SELECT file_id, row_id, begin_snapshot FROM \"{table}\" ORDER BY 1, 2"
        )))
        .fetch_all(&pool)
        .await
        .unwrap()
        .into_iter()
        .map(|row| (row.get(0), row.get(1), row.get(2)))
        .collect()
    }

    async fn live_delete_files(&self) -> i64 {
        sqlx::query_scalar("SELECT COUNT(*) FROM ducklake_delete_file WHERE end_snapshot IS NULL")
            .fetch_one(&self.pool().await)
            .await
            .unwrap()
    }

    fn objects(&self) -> usize {
        fn walk(dir: &std::path::Path) -> usize {
            std::fs::read_dir(dir)
                .unwrap()
                .map(|entry| {
                    let path = entry.unwrap().path();
                    if path.is_dir() {
                        walk(&path)
                    } else {
                        1
                    }
                })
                .sum()
        }
        walk(&self.tmp.path().join("data"))
    }

    /// Plan `sql` now against a pinned snapshot; run it later.
    async fn plan(
        &self,
        sql: &str,
    ) -> (
        Arc<dyn datafusion::physical_plan::ExecutionPlan>,
        SessionContext,
    ) {
        let ctx = self.ctx(5, None).await;
        let plan = ctx
            .sql(sql)
            .await
            .unwrap()
            .create_physical_plan()
            .await
            .unwrap();
        (plan, ctx)
    }

    async fn table(&self) -> (DuckLakeTable, SessionContext) {
        let ctx = self.ctx(0, None).await;
        let provider = ctx
            .catalog("lake")
            .unwrap()
            .schema("main")
            .unwrap()
            .table("t")
            .await
            .unwrap()
            .unwrap();
        let table = (provider.as_ref() as &dyn std::any::Any)
            .downcast_ref::<DuckLakeTable>()
            .unwrap()
            .clone();
        (table, ctx)
    }
}

fn ids(rows: &[i32]) -> Rows {
    rows.iter().map(|id| (*id, id * 10)).collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn small_delete_and_update_inline_then_flush() {
    let lake = Lake::new().await;
    lake.seed((1..=8).collect()).await;
    let mut history = Vec::new();
    lake.record(&mut history).await;
    let objects = lake.objects();

    assert_eq!(
        lake.exec("DELETE FROM lake.main.t WHERE id IN (1, 2)", 2)
            .await,
        2
    );
    lake.record(&mut history).await;
    assert_eq!(
        lake.exec("UPDATE lake.main.t SET val = 0 WHERE id = 3", 2)
            .await,
        1
    );
    lake.record(&mut history).await;
    assert_eq!(lake.objects(), objects, "no object written");
    assert_eq!(lake.live_delete_files().await, 0);
    assert_eq!(lake.inline_file_deletes().await.len(), 3);

    // Above the limit: a delete file, while the inlined deletions stay.
    lake.exec("DELETE FROM lake.main.t WHERE id >= 6", 2).await;
    lake.record(&mut history).await;
    assert_eq!(lake.live_delete_files().await, 1);
    assert_eq!(lake.inline_file_deletes().await.len(), 3);
    lake.exec("DELETE FROM lake.main.t WHERE id = 4", 2).await;
    lake.record(&mut history).await;
    let mut expected = vec![(3, 0)];
    expected.extend(ids(&[5]));
    assert_eq!(history.last().unwrap().1, expected);
    lake.check(&history).await;

    let backlog = lake
        .writer()
        .await
        .tables_with_inlined_file_deletes()
        .unwrap();
    assert_eq!(backlog.len(), 1);
    assert_eq!(backlog[0].rows, 4);
    let (table, ctx) = lake.table().await;
    let flushed = table.flush_inlined_deletes(&ctx.state()).await.unwrap();
    assert_eq!(flushed.files_flushed, 1);
    assert_eq!(flushed.rows_flushed, 4);
    assert!(lake.inline_file_deletes().await.is_empty());
    assert!(
        lake.writer()
            .await
            .tables_with_inlined_file_deletes()
            .unwrap()
            .is_empty()
    );
    assert_eq!(lake.live_delete_files().await, 1);
    lake.record(&mut history).await;
    lake.check(&history).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn stale_flush_and_stale_delete_abort() {
    let lake = Lake::new().await;
    lake.seed((1..=4).collect()).await;
    lake.exec("DELETE FROM lake.main.t WHERE id = 1", 5).await;

    // The flush reads, a DELETE commits first: the flush aborts.
    let (stale, ctx) = lake.table().await;
    lake.exec("DELETE FROM lake.main.t WHERE id = 2", 5).await;
    let head = lake.head().await;
    let error = stale
        .flush_inlined_deletes(&ctx.state())
        .await
        .expect_err("stale flush must abort");
    assert!(error.is_conflict(), "{error}");
    assert_eq!(lake.head().await, head);

    // A DELETE planned, the flush commits first. A DELETE lists its files
    // when it runs; its pinned snapshot then sees the flushed delete file,
    // filtered to that snapshot, which is exactly what it would have seen
    // before: it commits correctly.
    let (delete, planned) = lake.plan("DELETE FROM lake.main.t WHERE id = 3").await;
    let (table, ctx) = lake.table().await;
    table.flush_inlined_deletes(&ctx.state()).await.unwrap();
    datafusion::physical_plan::collect(delete, planned.task_ctx())
        .await
        .unwrap();
    let head = lake.head().await;
    assert_eq!(lake.rows(head).await, ids(&[4]));
    assert_eq!(lake.count(head).await, 1);

    // A DELETE planned, then another DELETE and a flush commit: the flushed
    // file holds a deletion the planned DELETE's snapshot does not see, so it
    // aborts rather than write a delete file without it.
    lake.seed(vec![5, 6, 7, 8]).await;
    let (delete, planned) = lake.plan("DELETE FROM lake.main.t WHERE id >= 5").await;
    lake.exec("DELETE FROM lake.main.t WHERE id = 5", 5).await;
    let (table, ctx) = lake.table().await;
    table.flush_inlined_deletes(&ctx.state()).await.unwrap();
    let head = lake.head().await;
    let error = datafusion::physical_plan::collect(delete, planned.task_ctx())
        .await
        .expect_err("stale DELETE must abort");
    assert!(is_conflict(&error), "{error}");
    assert_eq!(lake.head().await, head);
    assert_eq!(lake.rows(head).await, ids(&[4, 6, 7, 8]));
}
