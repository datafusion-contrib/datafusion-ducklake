//! Maintenance (flush, merge, rewrite) racing DML on the multicatalog
//! PostgreSQL backend, with no lock held by the caller.
//!
//! Each test plans one side against a pinned snapshot, lets the other side
//! commit, and then commits the stale side. The stale side must either commit
//! a correct result or abort with a conflict that [`is_conflict`] recognizes
//! and that leaves the catalog unchanged. Either way no row may be lost,
//! duplicated or brought back. Docker-gated (testcontainers Postgres).

#![cfg(feature = "write-postgres")]

use std::sync::Arc;

use arrow::array::{Array, Int32Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::execution::SessionState;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::prelude::*;
use datafusion_ducklake::{
    CompactionResult, DuckLakeCatalog, DuckLakeTable, MergeOptions, MetadataProvider,
    RewriteOptions, is_conflict,
};

use super::sql_update_inline_postgres_tests::{CAT, Lake, batch, some};

impl Lake {
    /// A writable handle on `public.t`, pinned at the current head.
    pub(crate) async fn table_handle(&self, limit: usize) -> (DuckLakeTable, SessionState) {
        let ctx = self.ctx(limit).await;
        let provider = ctx
            .catalog(CAT)
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
        (table, ctx.state())
    }

    async fn merge(&self) -> CompactionResult {
        let (table, state) = self.table_handle(0).await;
        table
            .merge_adjacent_files(&state, MergeOptions::default())
            .await
            .unwrap()
    }

    /// Plan `sql` now, through physical planning (which lists the files and
    /// reads the footers the statement will touch); run it later.
    async fn plan(&self, sql: &str, limit: usize) -> Planned {
        let ctx = self.ctx(limit).await;
        let plan = ctx
            .sql(sql)
            .await
            .unwrap()
            .create_physical_plan()
            .await
            .unwrap();
        Planned {
            plan,
            ctx,
        }
    }
}

/// A physical plan built against a pinned snapshot, executed later.
struct Planned {
    plan: Arc<dyn ExecutionPlan>,
    ctx: SessionContext,
}

impl Planned {
    async fn collect(self) -> datafusion::error::Result<Vec<RecordBatch>> {
        datafusion::physical_plan::collect(self.plan, self.ctx.task_ctx()).await
    }
}

fn rewrite_all() -> RewriteOptions {
    RewriteOptions {
        delete_threshold: 0.0,
        data_file_ids: None,
    }
}

/// Two small Parquet files, rows 1..=4, so a merge has a bin to merge.
async fn two_parquet_files(lake: &Lake) {
    lake.seed(batch(vec![1, 2], vec![10, 20]), 0).await;
    lake.seed(batch(vec![3, 4], vec![30, 40]), 0).await;
    assert_eq!(lake.live_files().await, (2, 0));
}

// (a) inline UPDATE vs flush_inlined_data, both commit orders.

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn inline_update_vs_flush() {
    let lake = Lake::new().await;
    lake.seed(batch(vec![1, 2, 3], vec![10, 20, 30]), 10).await;

    // UPDATE planned, flush commits first: the UPDATE aborts.
    let update = lake
        .plan(
            &format!("UPDATE {CAT}.public.t SET val = 0 WHERE id = 1"),
            10,
        )
        .await;
    lake.flush().await;
    let head = lake.head().await;
    let error = update.collect().await.expect_err("stale UPDATE must abort");
    assert!(is_conflict(&error), "{error}");
    assert_eq!(lake.head().await, head, "an abort commits nothing");
    assert_eq!(lake.rows(None).await, some(&[(1, 10), (2, 20), (3, 30)]));
    // A retry succeeds (the row is now in Parquet; the new version is inline).
    lake.exec(
        &format!("UPDATE {CAT}.public.t SET val = 0 WHERE id = 1"),
        10,
    )
    .await;
    assert_eq!(lake.rows(None).await, some(&[(1, 0), (2, 20), (3, 30)]));

    // Flush reads, UPDATE commits first: the flush aborts. The UPDATE ended
    // the version the flush read and wrote a new one with the SAME row id, so a
    // flush that went through would end the new version and write the old one.
    let (inputs, base) = lake.flush_inputs().await;
    assert_eq!(inputs.iter().map(|d| d.batch.num_rows()).sum::<usize>(), 1);
    lake.exec(
        &format!("UPDATE {CAT}.public.t SET val = 5 WHERE id IN (1, 2)"),
        10,
    )
    .await;
    let head = lake.head().await;
    let error = lake
        .try_flush(&inputs, base)
        .await
        .expect_err("stale flush must abort");
    assert!(error.is_conflict(), "{error}");
    assert_eq!(lake.head().await, head);
    assert_eq!(lake.rows(None).await, some(&[(1, 5), (2, 5), (3, 30)]));
    // A retried flush writes exactly the live rows.
    lake.flush().await;
    assert_eq!(lake.rows(None).await, some(&[(1, 5), (2, 5), (3, 30)]));
    assert!(lake.inline_versions().await.iter().all(|v| v.2.is_some()));
}

// (b) UPDATE (Parquet rows; new versions inline or Parquet) vs merge.

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn update_vs_merge_adjacent_files() {
    // limit 10: new versions inline; limit 0: new versions in Parquet.
    for limit in [10, 0] {
        let lake = Lake::new().await;
        two_parquet_files(&lake).await;

        // UPDATE planned, merge commits first: the UPDATE's target file is gone.
        let update = lake
            .plan(
                &format!("UPDATE {CAT}.public.t SET val = -1 WHERE id = 3"),
                limit,
            )
            .await;
        assert_eq!(lake.merge().await.files_processed, 2);
        let head = lake.head().await;
        let error = update.collect().await.expect_err("stale UPDATE must abort");
        assert!(is_conflict(&error), "limit {limit}: {error}");
        assert_eq!(lake.head().await, head);
        let merged = some(&[(1, 10), (2, 20), (3, 30), (4, 40)]);
        assert_eq!(lake.rows(None).await, merged);
        // A statement only parsed before the merge (`ctx.sql`, no physical
        // plan yet) lists files when it is executed. Its pinned snapshot then
        // sees the merged file (a merge output is visible from its sources'
        // first snapshot), so it commits, correctly.
        let update = lake
            .ctx(limit)
            .await
            .sql(&format!("UPDATE {CAT}.public.t SET val = -1 WHERE id = 3"))
            .await
            .unwrap();
        lake.seed(batch(vec![5], vec![50]), 0).await;
        lake.seed(batch(vec![6], vec![60]), 0).await;
        assert!(lake.merge().await.did_work());
        update.collect().await.unwrap();
        assert_eq!(
            lake.rows(None).await,
            some(&[(1, 10), (2, 20), (3, -1), (4, 40), (5, 50), (6, 60)])
        );

        // Merge reads, UPDATE commits first (a delete lands on a merge source):
        // the merge aborts.
        let lake = Lake::new().await;
        two_parquet_files(&lake).await;
        let (stale, state) = lake.table_handle(limit).await;
        lake.exec(
            &format!("UPDATE {CAT}.public.t SET val = -2 WHERE id = 2"),
            limit,
        )
        .await;
        let head = lake.head().await;
        let error = stale
            .merge_adjacent_files(&state, MergeOptions::default())
            .await
            .expect_err("stale merge must abort");
        assert!(error.is_conflict(), "limit {limit}: {error}");
        assert_eq!(lake.head().await, head);
        assert_eq!(
            lake.rows(None).await,
            some(&[(1, 10), (2, -2), (3, 30), (4, 40)])
        );
    }
}

/// An UPDATE of inlined rows only and a merge of Parquet files touch disjoint
/// storage: both commit, in either order, and nothing is lost.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn inline_only_update_and_merge_both_commit() {
    let lake = Lake::new().await;
    two_parquet_files(&lake).await;
    lake.exec(
        &format!("INSERT INTO {CAT}.public.t VALUES (5, 50), (6, 60)"),
        10,
    )
    .await;

    let update = lake
        .plan(
            &format!("UPDATE {CAT}.public.t SET val = 0 WHERE id = 5"),
            10,
        )
        .await;
    assert_eq!(lake.merge().await.files_processed, 2);
    update.collect().await.unwrap();
    assert_eq!(
        lake.rows(None).await,
        some(&[(1, 10), (2, 20), (3, 30), (4, 40), (5, 0), (6, 60)])
    );

    lake.seed(batch(vec![7], vec![70]), 0).await;
    lake.seed(batch(vec![8], vec![80]), 0).await;
    let (stale, state) = lake.table_handle(10).await;
    lake.exec(
        &format!("UPDATE {CAT}.public.t SET val = 1 WHERE id = 6"),
        10,
    )
    .await;
    let merged = stale
        .merge_adjacent_files(&state, MergeOptions::default())
        .await
        .unwrap();
    assert!(merged.did_work());
    assert_eq!(
        lake.rows(None).await,
        some(&[(1, 10), (2, 20), (3, 30), (4, 40), (5, 0), (6, 1), (7, 70), (8, 80)])
    );
}

// (b) UPDATE vs rewrite_data_files.

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn update_vs_rewrite_data_files() {
    for limit in [10, 0] {
        // A file with a delete, so a rewrite has work.
        let lake = Lake::new().await;
        lake.seed(batch(vec![1, 2, 3, 4], vec![10, 20, 30, 40]), 0)
            .await;
        lake.exec(&format!("DELETE FROM {CAT}.public.t WHERE id = 4"), limit)
            .await;

        // UPDATE planned, rewrite commits first.
        let update = lake
            .plan(
                &format!("UPDATE {CAT}.public.t SET val = -1 WHERE id = 1"),
                limit,
            )
            .await;
        let (table, state) = lake.table_handle(limit).await;
        assert!(
            table
                .rewrite_data_files(&state, rewrite_all())
                .await
                .unwrap()
                .did_work()
        );
        let head = lake.head().await;
        let error = update.collect().await.expect_err("stale UPDATE must abort");
        assert!(is_conflict(&error), "limit {limit}: {error}");
        assert_eq!(lake.head().await, head);
        assert_eq!(lake.rows(None).await, some(&[(1, 10), (2, 20), (3, 30)]));

        // Rewrite reads, UPDATE commits first.
        lake.exec(&format!("DELETE FROM {CAT}.public.t WHERE id = 3"), limit)
            .await;
        let (stale, state) = lake.table_handle(limit).await;
        lake.exec(
            &format!("UPDATE {CAT}.public.t SET val = -2 WHERE id = 2"),
            limit,
        )
        .await;
        let head = lake.head().await;
        let error = stale
            .rewrite_data_files(&state, rewrite_all())
            .await
            .expect_err("stale rewrite must abort");
        assert!(error.is_conflict(), "limit {limit}: {error}");
        assert_eq!(lake.head().await, head);
        assert_eq!(lake.rows(None).await, some(&[(1, 10), (2, -2)]));
    }
}

// (c) DELETE vs flush_inlined_data, both commit orders.

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn delete_vs_flush() {
    let lake = Lake::new().await;
    lake.seed(batch(vec![1, 2, 3], vec![10, 20, 30]), 10).await;

    // DELETE planned, flush commits first.
    let delete = lake
        .plan(&format!("DELETE FROM {CAT}.public.t WHERE id = 1"), 10)
        .await;
    lake.flush().await;
    let head = lake.head().await;
    let error = delete.collect().await.expect_err("stale DELETE must abort");
    assert!(is_conflict(&error), "{error}");
    assert_eq!(lake.head().await, head);
    assert_eq!(lake.rows(None).await, some(&[(1, 10), (2, 20), (3, 30)]));

    // Flush reads, DELETE commits first.
    lake.exec(
        &format!("INSERT INTO {CAT}.public.t VALUES (4, 40), (5, 50)"),
        10,
    )
    .await;
    let (inputs, base) = lake.flush_inputs().await;
    lake.exec(&format!("DELETE FROM {CAT}.public.t WHERE id = 4"), 10)
        .await;
    let head = lake.head().await;
    let error = lake
        .try_flush(&inputs, base)
        .await
        .expect_err("stale flush must abort");
    assert!(error.is_conflict(), "{error}");
    assert_eq!(lake.head().await, head);
    assert_eq!(
        lake.rows(None).await,
        some(&[(1, 10), (2, 20), (3, 30), (5, 50)])
    );
    lake.flush().await;
    assert_eq!(
        lake.rows(None).await,
        some(&[(1, 10), (2, 20), (3, 30), (5, 50)])
    );
}

/// A DELETE planned before an UPDATE of the same inlined row must not end the
/// new version, which carries the same row id as the one it read.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn stale_delete_after_inline_update_aborts() {
    let lake = Lake::new().await;
    lake.seed(batch(vec![1, 2], vec![10, 20]), 10).await;
    let delete = lake
        .plan(&format!("DELETE FROM {CAT}.public.t WHERE id = 1"), 10)
        .await;
    lake.exec(
        &format!("UPDATE {CAT}.public.t SET val = 11 WHERE id = 1"),
        10,
    )
    .await;
    let error = delete.collect().await.expect_err("stale DELETE must abort");
    assert!(is_conflict(&error), "{error}");
    assert_eq!(lake.rows(None).await, some(&[(1, 11), (2, 20)]));
}

// Appends vs maintenance.

/// An append never aborts because of a flush or a compaction, and a
/// compaction never aborts because of an append. A flush that read before an
/// append committed aborts (its fence covers every change to the table).
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn appends_vs_maintenance() {
    let lake = Lake::new().await;
    two_parquet_files(&lake).await;
    let mut expected = some(&[(1, 10), (2, 20), (3, 30), (4, 40)]);

    // INSERT planned (inline and Parquet), merge commits first: both commit.
    for (id, limit) in [(5, 10), (6, 0)] {
        let insert = lake
            .plan(
                &format!("INSERT INTO {CAT}.public.t VALUES ({id}, {id}0)"),
                limit,
            )
            .await;
        lake.seed(batch(vec![100 + id], vec![0]), 0).await;
        assert!(lake.merge().await.did_work());
        insert.collect().await.unwrap();
        expected.push((id, Some(id * 10)));
        expected.push((100 + id, Some(0)));
        expected.sort();
        assert_eq!(lake.rows(None).await, expected);
    }

    // Merge reads, INSERT commits first: both commit.
    lake.seed(batch(vec![7], vec![70]), 0).await;
    lake.seed(batch(vec![8], vec![80]), 0).await;
    let (stale, state) = lake.table_handle(0).await;
    lake.exec(&format!("INSERT INTO {CAT}.public.t VALUES (9, 90)"), 0)
        .await;
    assert!(
        stale
            .merge_adjacent_files(&state, MergeOptions::default())
            .await
            .unwrap()
            .did_work()
    );
    expected.extend(some(&[(7, 70), (8, 80), (9, 90)]));
    expected.sort();
    assert_eq!(lake.rows(None).await, expected);

    // INSERT planned, flush commits first: the INSERT commits.
    let insert = lake
        .plan(&format!("INSERT INTO {CAT}.public.t VALUES (10, 100)"), 10)
        .await;
    lake.flush().await;
    insert.collect().await.unwrap();
    expected.push((10, Some(100)));
    expected.sort();
    assert_eq!(lake.rows(None).await, expected);

    // Flush reads, INSERT commits first: the flush aborts (retryable).
    for limit in [10, 0] {
        let (inputs, base) = lake.flush_inputs().await;
        assert!(!inputs.is_empty());
        let id = 20 + limit as i32;
        lake.exec(
            &format!("INSERT INTO {CAT}.public.t VALUES ({id}, 1)"),
            limit,
        )
        .await;
        let error = lake
            .try_flush(&inputs, base)
            .await
            .expect_err("stale flush");
        assert!(error.is_conflict(), "{error}");
        expected.push((id, Some(1)));
        expected.sort();
        assert_eq!(lake.rows(None).await, expected);
    }
    lake.flush().await;
    assert_eq!(lake.rows(None).await, expected);
}

// Schema changes vs maintenance.

fn wide_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("val", DataType::Int32, true),
        Field::new("extra", DataType::Int32, true),
    ]))
}

fn ids_only(ids: Vec<i32>) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)])),
        vec![Arc::new(Int32Array::from(ids))],
    )
    .unwrap()
}

/// `(id, extra)` at the head of a table that gained `extra`.
async fn extra_rows(lake: &Lake) -> Vec<(i32, Option<i32>)> {
    let provider = lake.provider().await;
    let head = provider.get_current_snapshot().unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog(
        CAT,
        Arc::new(DuckLakeCatalog::with_snapshot(provider, head).unwrap()),
    );
    let batches = ctx
        .sql(&format!("SELECT id, extra FROM {CAT}.public.t ORDER BY id"))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let mut out = Vec::new();
    for b in &batches {
        let ids = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        let extra = b.column(1).as_any().downcast_ref::<Int32Array>().unwrap();
        for r in 0..b.num_rows() {
            out.push((ids.value(r), extra.is_valid(r).then(|| extra.value(r))));
        }
    }
    out
}

/// A flush that read before a column was added aborts: its stage names the
/// old column set.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn flush_vs_add_column_aborts() {
    let lake = Lake::new().await;
    lake.seed(batch(vec![1, 2], vec![10, 20]), 10).await;
    let (inputs, base) = lake.flush_inputs().await;
    let wide = RecordBatch::try_new(
        wide_schema(),
        vec![
            Arc::new(Int32Array::from(vec![3])),
            Arc::new(Int32Array::from(vec![30])),
            Arc::new(Int32Array::from(vec![300])),
        ],
    )
    .unwrap();
    lake.table_writer(0)
        .await
        .append_table("public", "t", &[wide])
        .await
        .unwrap();
    let error = lake
        .try_flush(&inputs, base)
        .await
        .expect_err("stale flush");
    assert!(error.is_conflict(), "{error}");
    assert_eq!(
        extra_rows(&lake).await,
        vec![(1, None), (2, None), (3, Some(300))]
    );
    // Retried against the new schema, the flush commits.
    lake.flush().await;
    assert_eq!(
        extra_rows(&lake).await,
        vec![(1, None), (2, None), (3, Some(300))]
    );
}

/// A merge that read before a column was added or dropped commits its output
/// with the column set it read. That is correct: the output holds exactly the
/// columns its sources held, readers fill a column a file lacks with NULL,
/// and ignore a column the current schema no longer has, and time travel
/// still reads the dropped column.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn merge_vs_schema_change_commits_correctly() {
    // ADD COLUMN.
    let lake = Lake::new().await;
    two_parquet_files(&lake).await;
    let (stale, state) = lake.table_handle(0).await;
    let wide = RecordBatch::try_new(
        wide_schema(),
        vec![
            Arc::new(Int32Array::from(vec![5])),
            Arc::new(Int32Array::from(vec![50])),
            Arc::new(Int32Array::from(vec![500])),
        ],
    )
    .unwrap();
    lake.table_writer(0)
        .await
        .append_table("public", "t", &[wide])
        .await
        .unwrap();
    let merged = stale
        .merge_adjacent_files(&state, MergeOptions::default())
        .await
        .unwrap();
    assert_eq!(merged.files_processed, 2);
    assert_eq!(
        extra_rows(&lake).await,
        vec![(1, None), (2, None), (3, None), (4, None), (5, Some(500))]
    );
    assert_eq!(
        lake.rows(None).await,
        some(&[(1, 10), (2, 20), (3, 30), (4, 40), (5, 50)])
    );

    // DROP COLUMN (an append without `val` retires it).
    let lake = Lake::new().await;
    two_parquet_files(&lake).await;
    let before_drop = lake.head().await;
    let (stale, state) = lake.table_handle(0).await;
    lake.table_writer(0)
        .await
        .append_table("public", "t", &[ids_only(vec![5])])
        .await
        .unwrap();
    let merged = stale
        .merge_adjacent_files(&state, MergeOptions::default())
        .await
        .unwrap();
    assert_eq!(merged.files_processed, 2);
    assert_eq!(
        lake.rows(Some(before_drop)).await,
        some(&[(1, 10), (2, 20), (3, 30), (4, 40)]),
        "time travel still reads the dropped column from the merged file"
    );
    let provider = lake.provider().await;
    let head = provider.get_current_snapshot().unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog(
        CAT,
        Arc::new(DuckLakeCatalog::with_snapshot(provider, head).unwrap()),
    );
    let batches = ctx
        .sql(&format!("SELECT * FROM {CAT}.public.t ORDER BY id"))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(batches[0].num_columns(), 1);
    let ids: Vec<i32> = batches
        .iter()
        .flat_map(|b| {
            b.column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .values()
                .to_vec()
        })
        .collect();
    assert_eq!(ids, vec![1, 2, 3, 4, 5]);
}

// Deletion inlining: DML that records inlined deletions of Parquet rows vs
// the flush of those deletions and vs compaction, in both commit orders.

/// One Parquet file (rows 1..=6) with one inlined deletion (row 6), so a
/// flush of inlined deletions has work.
async fn parquet_file_with_an_inlined_delete(lake: &Lake) {
    lake.seed(
        batch(vec![1, 2, 3, 4, 5, 6], vec![10, 20, 30, 40, 50, 60]),
        0,
    )
    .await;
    lake.exec(&format!("DELETE FROM {CAT}.public.t WHERE id = 6"), 10)
        .await;
    assert_eq!(lake.live_files().await, (1, 0));
    assert_eq!(lake.inline_file_deletes().await.len(), 1);
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn inlined_delete_update_vs_flush_of_deletes() {
    let lake = Lake::new().await;
    parquet_file_with_an_inlined_delete(&lake).await;

    // UPDATE planned, the flush of deletions commits first: the UPDATE aborts
    // (the file's live delete file changed).
    let update = lake
        .plan(
            &format!("UPDATE {CAT}.public.t SET val = 0 WHERE id = 1"),
            10,
        )
        .await;
    assert_eq!(lake.flush_deletes().await.rows_flushed, 1);
    let head = lake.head().await;
    let error = update.collect().await.expect_err("stale UPDATE must abort");
    assert!(is_conflict(&error), "{error}");
    assert_eq!(lake.head().await, head, "an abort commits nothing");
    assert!(lake.inline_file_deletes().await.is_empty());
    let expected = some(&[(1, 10), (2, 20), (3, 30), (4, 40), (5, 50)]);
    assert_eq!(lake.rows(None).await, expected);

    // The flush reads, an UPDATE and a DELETE commit first: the flush aborts.
    lake.exec(
        &format!("UPDATE {CAT}.public.t SET val = 0 WHERE id = 1"),
        10,
    )
    .await;
    let (stale, state) = lake.table_handle(0).await;
    lake.exec(&format!("DELETE FROM {CAT}.public.t WHERE id = 2"), 10)
        .await;
    let head = lake.head().await;
    let error = stale
        .flush_inlined_deletes(&state)
        .await
        .expect_err("stale flush must abort");
    assert!(error.is_conflict(), "{error}");
    assert_eq!(lake.head().await, head);
    let expected = some(&[(1, 0), (3, 30), (4, 40), (5, 50)]);
    assert_eq!(lake.rows(None).await, expected);
    assert_eq!(lake.count(None).await, 4);
    assert_eq!(lake.flush_deletes().await.rows_flushed, 2);
    assert_eq!(lake.rows(None).await, expected);
    assert_eq!(lake.count(None).await, 4);
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn inlined_delete_vs_rewrite_data_files() {
    let lake = Lake::new().await;
    parquet_file_with_an_inlined_delete(&lake).await;

    // The rewrite reads, an inlined DELETE commits first: the rewrite aborts.
    let (stale, state) = lake.table_handle(0).await;
    lake.exec(&format!("DELETE FROM {CAT}.public.t WHERE id = 5"), 10)
        .await;
    let head = lake.head().await;
    let error = stale
        .rewrite_data_files(&state, rewrite_all())
        .await
        .expect_err("stale rewrite must abort");
    assert!(error.is_conflict(), "{error}");
    assert_eq!(lake.head().await, head);
    let expected = some(&[(1, 10), (2, 20), (3, 30), (4, 40)]);
    assert_eq!(lake.rows(None).await, expected);

    // DELETE planned, the rewrite commits first: the DELETE aborts.
    let delete = lake
        .plan(&format!("DELETE FROM {CAT}.public.t WHERE id = 4"), 10)
        .await;
    let (table, state) = lake.table_handle(0).await;
    assert!(
        table
            .rewrite_data_files(&state, rewrite_all())
            .await
            .unwrap()
            .did_work()
    );
    let head = lake.head().await;
    let error = delete.collect().await.expect_err("stale DELETE must abort");
    assert!(is_conflict(&error), "{error}");
    assert_eq!(lake.head().await, head);
    assert_eq!(lake.rows(None).await, expected);
    assert_eq!(lake.count(None).await, 4);
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn inlined_delete_vs_merge_adjacent_files() {
    let lake = Lake::new().await;
    two_parquet_files(&lake).await;

    // The merge reads, an inlined DELETE commits first: the merge aborts.
    let (stale, state) = lake.table_handle(0).await;
    lake.exec(&format!("DELETE FROM {CAT}.public.t WHERE id = 1"), 10)
        .await;
    let head = lake.head().await;
    let error = stale
        .merge_adjacent_files(&state, MergeOptions::default())
        .await
        .expect_err("stale merge must abort");
    assert!(error.is_conflict(), "{error}");
    assert_eq!(lake.head().await, head);
    let expected = some(&[(2, 20), (3, 30), (4, 40)]);
    assert_eq!(lake.rows(None).await, expected);
    // A fresh merge skips the file with an inlined deletion.
    assert!(!lake.merge().await.did_work());

    // After a flush of the deletions the file has a delete file, which merge
    // also skips. DELETE planned, then a rewrite and a merge commit first. A
    // DELETE lists its files when it runs, and its pinned snapshot sees the
    // merged file (visible from its sources' first snapshot), so it commits
    // correctly.
    lake.flush_deletes().await;
    let delete = lake
        .plan(&format!("DELETE FROM {CAT}.public.t WHERE id = 3"), 10)
        .await;
    let (table, state) = lake.table_handle(0).await;
    assert!(
        table
            .rewrite_data_files(&state, rewrite_all())
            .await
            .unwrap()
            .did_work()
    );
    assert!(lake.merge().await.did_work());
    delete.collect().await.unwrap();
    assert_eq!(lake.rows(None).await, some(&[(2, 20), (4, 40)]));
    assert_eq!(lake.count(None).await, 2);
    assert_eq!(lake.inline_file_deletes().await.len(), 1);

    // UPDATE planned (it lists its files at plan time), merge commits first:
    // the UPDATE aborts.
    let lake = Lake::new().await;
    two_parquet_files(&lake).await;
    let update = lake
        .plan(
            &format!("UPDATE {CAT}.public.t SET val = 0 WHERE id = 3"),
            10,
        )
        .await;
    assert!(lake.merge().await.did_work());
    let head = lake.head().await;
    let error = update.collect().await.expect_err("stale UPDATE must abort");
    assert!(is_conflict(&error), "{error}");
    assert_eq!(lake.head().await, head);
    assert_eq!(
        lake.rows(None).await,
        some(&[(1, 10), (2, 20), (3, 30), (4, 40)])
    );
}

/// Two DML statements that inline deletions of rows in one file, planned at
/// the same snapshot: the second to commit aborts, even for the same row. On
/// different files both commit.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn concurrent_inlined_deletes() {
    // The same row, and another row of the same file.
    for sql in [
        format!("DELETE FROM {CAT}.public.t WHERE id = 1"),
        format!("UPDATE {CAT}.public.t SET val = 0 WHERE id = 2"),
    ] {
        let lake = Lake::new().await;
        two_parquet_files(&lake).await;
        let first = lake
            .plan(&format!("DELETE FROM {CAT}.public.t WHERE id = 1"), 10)
            .await;
        let second = lake.plan(&sql, 10).await;
        first.collect().await.unwrap();
        let head = lake.head().await;
        let error = second.collect().await.expect_err("second must abort");
        assert!(is_conflict(&error), "{sql}: {error}");
        assert_eq!(lake.head().await, head);
        assert_eq!(lake.rows(None).await, some(&[(2, 20), (3, 30), (4, 40)]));
    }

    // Different files commute.
    let lake = Lake::new().await;
    two_parquet_files(&lake).await;
    let first = lake
        .plan(&format!("DELETE FROM {CAT}.public.t WHERE id = 2"), 10)
        .await;
    let second = lake
        .plan(&format!("DELETE FROM {CAT}.public.t WHERE id = 3"), 10)
        .await;
    first.collect().await.unwrap();
    second.collect().await.unwrap();
    assert_eq!(lake.rows(None).await, some(&[(1, 10), (4, 40)]));
    assert_eq!(lake.count(None).await, 2);
}

/// A DELETE that writes a delete file, planned before an inlined deletion of
/// the same file commits, aborts rather than lose the inlined one; in the
/// other order the inlined deletion aborts.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn delete_file_vs_inlined_delete() {
    let lake = Lake::new().await;
    lake.seed(batch(vec![1, 2, 3, 4], vec![10, 20, 30, 40]), 0)
        .await;

    let big = lake
        .plan(&format!("DELETE FROM {CAT}.public.t WHERE id <= 2"), 1)
        .await;
    lake.exec(&format!("DELETE FROM {CAT}.public.t WHERE id = 3"), 1)
        .await;
    let head = lake.head().await;
    let error = big.collect().await.expect_err("stale DELETE must abort");
    assert!(is_conflict(&error), "{error}");
    assert_eq!(lake.head().await, head);

    let small = lake
        .plan(&format!("DELETE FROM {CAT}.public.t WHERE id = 4"), 1)
        .await;
    lake.exec(&format!("DELETE FROM {CAT}.public.t WHERE id <= 2"), 1)
        .await;
    let head = lake.head().await;
    let error = small.collect().await.expect_err("stale DELETE must abort");
    assert!(is_conflict(&error), "{error}");
    assert_eq!(lake.head().await, head);
    assert_eq!(lake.rows(None).await, some(&[(4, 40)]));
    assert_eq!(lake.count(None).await, 1);
}

/// A DELETE lists its files when it runs. After a flush of inlined deletions
/// its pinned snapshot sees the flushed delete file filtered to that
/// snapshot: when that is all the file holds, the DELETE commits correctly;
/// when the file also holds a later deletion, the DELETE aborts (in both the
/// inlined and the delete-file form) rather than bring the row back.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn stale_delete_vs_flush_of_deletes() {
    let lake = Lake::new().await;
    parquet_file_with_an_inlined_delete(&lake).await;

    let delete = lake
        .plan(&format!("DELETE FROM {CAT}.public.t WHERE id = 1"), 10)
        .await;
    lake.flush_deletes().await;
    delete.collect().await.unwrap();
    assert_eq!(
        lake.rows(None).await,
        some(&[(2, 20), (3, 30), (4, 40), (5, 50)])
    );
    assert_eq!(lake.count(None).await, 4);

    for (sql, limit) in [
        (format!("DELETE FROM {CAT}.public.t WHERE id >= 4"), 10),
        (format!("DELETE FROM {CAT}.public.t WHERE id >= 4"), 1),
    ] {
        let delete = lake.plan(&sql, limit).await;
        lake.exec(&format!("DELETE FROM {CAT}.public.t WHERE id = 5"), 10)
            .await;
        lake.flush_deletes().await;
        let head = lake.head().await;
        let error = delete.collect().await.expect_err("stale DELETE must abort");
        assert!(is_conflict(&error), "limit {limit}: {error}");
        assert_eq!(lake.head().await, head);
        assert_eq!(lake.rows(None).await, some(&[(2, 20), (3, 30), (4, 40)]));
        assert_eq!(lake.count(None).await, 3);
        // Put row 5 back for the next round.
        lake.exec(&format!("INSERT INTO {CAT}.public.t VALUES (5, 50)"), 0)
            .await;
    }
}
