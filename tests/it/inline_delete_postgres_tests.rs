//! Deletion inlining on the multicatalog PostgreSQL backend: a SQL `DELETE`
//! or `UPDATE` that removes at most `data_inlining_row_limit` rows held in
//! Parquet files records them in `ducklake_inlined_delete_<table_id>` instead
//! of writing a delete file, and `DuckLakeTable::flush_inlined_deletes` later
//! materializes them into delete files. These tests check the catalog tables
//! and the object store directly, and read every snapshot back through time
//! travel. Docker-gated (testcontainers Postgres).

#![cfg(feature = "write-postgres")]

use datafusion_ducklake::{InlinedDeleteFlushResult, MetadataWriter};

use super::sql_update_inline_postgres_tests::{CAT, Lake, batch, some};

type Rows = Vec<(i32, Option<i32>)>;

impl Lake {
    /// Flush the inlined deletions of `public.t` at the current head.
    pub(crate) async fn flush_deletes(&self) -> InlinedDeleteFlushResult {
        let (table, state) = self.table_handle(0).await;
        table.flush_inlined_deletes(&state).await.unwrap()
    }

    /// Record the rows and `COUNT(*)` of the current head.
    async fn record(&self, history: &mut Vec<(i64, Rows, i64)>) {
        history.push((
            self.head().await,
            self.rows(None).await,
            self.count(None).await,
        ));
    }

    /// Every recorded snapshot still reads its own rows and count.
    async fn check_history(&self, history: &[(i64, Rows, i64)]) {
        for (snapshot, rows, count) in history {
            assert_eq!(
                &self.rows(Some(*snapshot)).await,
                rows,
                "rows at snapshot {snapshot}"
            );
            assert_eq!(
                self.count(Some(*snapshot)).await,
                *count,
                "count at snapshot {snapshot}"
            );
        }
    }

    /// `(delete_file_id, begin_snapshot, end_snapshot, delete_count,
    /// partial_max)` of every delete file row, ordered.
    async fn delete_files(&self) -> Vec<(i64, i64, Option<i64>, i64, Option<i64>)> {
        use sqlx::Row;
        sqlx::query(
            "SELECT delete_file_id, begin_snapshot, end_snapshot, delete_count, partial_max
             FROM ducklake_delete_file ORDER BY delete_file_id",
        )
        .fetch_all(&self.pool)
        .await
        .unwrap()
        .into_iter()
        .map(|row| (row.get(0), row.get(1), row.get(2), row.get(3), row.get(4)))
        .collect()
    }
}

fn ids(rows: &[i32]) -> Rows {
    rows.iter().map(|id| (*id, Some(id * 10))).collect()
}

/// An UPDATE of Parquet rows within the limit touches only the catalog: no
/// new object, no delete file.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn update_within_the_limit_writes_no_object() {
    let lake = Lake::new().await;
    lake.seed(batch(vec![1, 2, 3, 4], vec![10, 20, 30, 40]), 0)
        .await;
    let mut history = Vec::new();
    lake.record(&mut history).await;
    let objects = lake.objects();
    assert_eq!(objects.len(), 1);

    let n = lake
        .exec(
            &format!("UPDATE {CAT}.public.t SET val = -val WHERE id IN (1, 3)"),
            2,
        )
        .await;
    assert_eq!(n, 2);
    let after = lake.head().await;
    assert_eq!(lake.objects(), objects, "no object written");
    assert_eq!(lake.live_files().await, (1, 0));
    let file_id = lake.inline_file_deletes().await[0].0;
    assert_eq!(
        lake.inline_file_deletes().await,
        vec![(file_id, 0, after), (file_id, 2, after)]
    );
    lake.record(&mut history).await;
    assert_eq!(history[1].1, some(&[(1, -10), (2, 20), (3, -30), (4, 40)]));
    assert_eq!(history[1].2, 4);
    lake.check_history(&history).await;
}

/// A DELETE of Parquet rows within the limit touches only the catalog.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn delete_within_the_limit_writes_no_object() {
    let lake = Lake::new().await;
    lake.seed(batch(vec![1, 2, 3], vec![10, 20, 30]), 0).await;
    lake.seed(batch(vec![4, 5, 6], vec![40, 50, 60]), 0).await;
    let mut history = Vec::new();
    lake.record(&mut history).await;
    let objects = lake.objects();

    // Two rows in two files: one inlined deletion per file.
    let n = lake
        .exec(&format!("DELETE FROM {CAT}.public.t WHERE id IN (2, 6)"), 2)
        .await;
    assert_eq!(n, 2);
    let after = lake.head().await;
    assert_eq!(lake.objects(), objects, "no object written");
    assert_eq!(lake.live_files().await, (2, 0));
    let deletes = lake.inline_file_deletes().await;
    assert_eq!(deletes.len(), 2);
    assert_ne!(deletes[0].0, deletes[1].0);
    assert!(deletes.iter().all(|d| d.1 == 1 || d.1 == 2));
    assert!(deletes.iter().all(|d| d.2 == after));
    lake.record(&mut history).await;
    assert_eq!(history[1].1, ids(&[1, 3, 4, 5]));

    // Deleting an already deleted row changes nothing.
    let head = lake.head().await;
    assert_eq!(
        lake.exec(&format!("DELETE FROM {CAT}.public.t WHERE id = 2"), 2)
            .await,
        0
    );
    assert_eq!(lake.head().await, head);
    lake.check_history(&history).await;
}

/// Above the limit, a DELETE and an UPDATE write delete files as before.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn above_the_limit_writes_a_delete_file() {
    let lake = Lake::new().await;
    lake.seed(
        batch(vec![1, 2, 3, 4, 5, 6], vec![10, 20, 30, 40, 50, 60]),
        0,
    )
    .await;
    let mut history = Vec::new();
    lake.record(&mut history).await;

    lake.exec(&format!("DELETE FROM {CAT}.public.t WHERE id <= 3"), 2)
        .await;
    assert_eq!(lake.live_files().await, (1, 1));
    assert!(lake.inline_file_deletes().await.is_empty());
    lake.record(&mut history).await;

    // Three new versions go to Parquet, the old ones to a delete file.
    lake.exec(&format!("UPDATE {CAT}.public.t SET val = 0"), 2)
        .await;
    assert_eq!(lake.live_files().await, (2, 1));
    assert!(lake.inline_file_deletes().await.is_empty());
    lake.record(&mut history).await;
    assert_eq!(history[2].1, some(&[(4, 0), (5, 0), (6, 0)]));

    // With inlining off, a single-row DELETE writes a delete file.
    lake.exec(&format!("DELETE FROM {CAT}.public.t WHERE id = 4"), 0)
        .await;
    assert!(lake.inline_file_deletes().await.is_empty());
    lake.record(&mut history).await;
    lake.check_history(&history).await;
}

/// One data file with both a delete file and inlined deletions, in every
/// order, then a flush: reads, COUNT(*) and time travel stay correct.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn delete_file_and_inlined_deletes_on_one_file() {
    let lake = Lake::new().await;
    let all: Vec<i32> = (1..=10).collect();
    lake.seed(batch(all.clone(), all.iter().map(|v| v * 10).collect()), 0)
        .await;
    let mut history = Vec::new();
    lake.record(&mut history).await;

    // Inlined, then a delete file, then inlined again, then an UPDATE.
    lake.exec(&format!("DELETE FROM {CAT}.public.t WHERE id = 1"), 2)
        .await;
    lake.record(&mut history).await;
    lake.exec(
        &format!("DELETE FROM {CAT}.public.t WHERE id IN (2, 3, 4)"),
        2,
    )
    .await;
    assert_eq!(lake.live_files().await, (1, 1));
    lake.record(&mut history).await;
    lake.exec(&format!("DELETE FROM {CAT}.public.t WHERE id IN (1, 5)"), 2)
        .await;
    lake.record(&mut history).await;
    lake.exec(
        &format!("UPDATE {CAT}.public.t SET val = 0 WHERE id IN (6, 7)"),
        2,
    )
    .await;
    lake.record(&mut history).await;
    assert_eq!(lake.live_files().await, (1, 1));
    assert_eq!(lake.inline_file_deletes().await.len(), 4);
    assert_eq!(
        history.last().unwrap().1,
        some(&[(6, 0), (7, 0), (8, 80), (9, 90), (10, 100)])
    );
    assert_eq!(history.last().unwrap().2, 5);
    lake.check_history(&history).await;

    // A DELETE above the limit on the same file carries only the delete file's
    // positions forward; the inlined ones stay inlined.
    lake.exec(&format!("DELETE FROM {CAT}.public.t WHERE id >= 9"), 1)
        .await;
    lake.record(&mut history).await;
    assert_eq!(lake.inline_file_deletes().await.len(), 4);
    lake.check_history(&history).await;

    // The flush moves every inlined deletion into one delete file and keeps
    // every snapshot's rows.
    let objects = lake.objects().len();
    let flushed = lake.flush_deletes().await;
    assert_eq!(flushed.files_flushed, 1);
    assert_eq!(flushed.rows_flushed, 4);
    assert!(lake.inline_file_deletes().await.is_empty());
    assert_eq!(lake.live_files().await, (1, 1));
    assert_eq!(lake.objects().len(), objects + 1);
    lake.record(&mut history).await;
    let head = history.last().unwrap();
    assert_eq!(head.1, history[history.len() - 2].1);
    lake.check_history(&history).await;

    // Only one delete file is visible at any snapshot.
    let files = lake.delete_files().await;
    let live = files.iter().filter(|f| f.2.is_none()).collect::<Vec<_>>();
    assert_eq!(live.len(), 1);
    assert_eq!(live[0].3, 9, "all nine deleted positions");
    assert_eq!(
        live[0].1, history[1].0,
        "visible from the first inlined deletion"
    );
    for (snapshot, _, _) in &history {
        let visible = files
            .iter()
            .filter(|f| f.1 <= *snapshot && f.2.is_none_or(|end| *snapshot < end))
            .count();
        assert!(visible <= 1, "{visible} delete files visible at {snapshot}");
    }

    // A second flush has nothing to do and commits nothing.
    let head = lake.head().await;
    let again = lake.flush_deletes().await;
    assert_eq!(again.snapshot_id, None);
    assert_eq!(lake.head().await, head);

    // Deletions after a flush go inline again, and a second flush folds the
    // first flush's delete file forward.
    lake.exec(&format!("DELETE FROM {CAT}.public.t WHERE id = 8"), 2)
        .await;
    lake.record(&mut history).await;
    lake.flush_deletes().await;
    lake.record(&mut history).await;
    assert!(lake.inline_file_deletes().await.is_empty());
    assert_eq!(history.last().unwrap().1, some(&[(6, 0), (7, 0)]));
    lake.check_history(&history).await;
}

/// Repeated small UPDATEs of one row: the Parquet version gets one inlined
/// deletion, later versions are inline rows; every snapshot reads its value,
/// before and after a flush of the deletions.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn repeated_update_of_a_parquet_row() {
    let lake = Lake::new().await;
    lake.seed(batch(vec![1, 2], vec![10, 20]), 0).await;
    let objects = lake.objects();
    let mut history = Vec::new();
    lake.record(&mut history).await;
    for value in [11, 12, 13] {
        lake.exec(
            &format!("UPDATE {CAT}.public.t SET val = {value} WHERE id = 1"),
            10,
        )
        .await;
        lake.record(&mut history).await;
    }
    assert_eq!(lake.objects(), objects);
    assert_eq!(lake.inline_file_deletes().await.len(), 1);
    lake.flush_deletes().await;
    lake.record(&mut history).await;
    assert_eq!(history.last().unwrap().1, some(&[(1, 13), (2, 20)]));
    lake.check_history(&history).await;
}

/// The flushed delete file carries each row's deletion snapshot, and the
/// backlog listing names the table until the flush.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn flush_writes_per_row_snapshots_and_empties_the_backlog() {
    let lake = Lake::new().await;
    lake.seed(batch(vec![1, 2, 3, 4], vec![10, 20, 30, 40]), 0)
        .await;
    let writer = lake.writer().await;
    assert!(writer.supports_inlined_file_deletes());
    assert!(
        writer
            .tables_with_inlined_file_deletes()
            .unwrap()
            .is_empty()
    );

    lake.exec(&format!("DELETE FROM {CAT}.public.t WHERE id = 2"), 5)
        .await;
    let first = lake.head().await;
    lake.exec(&format!("DELETE FROM {CAT}.public.t WHERE id = 4"), 5)
        .await;
    let second = lake.head().await;
    let backlog = writer.tables_with_inlined_file_deletes().unwrap();
    assert_eq!(backlog.len(), 1);
    assert_eq!(
        (
            backlog[0].schema_name.as_str(),
            backlog[0].table_name.as_str()
        ),
        ("public", "t")
    );
    assert_eq!(backlog[0].rows, 2);

    let before = lake.objects();
    let flushed = lake.flush_deletes().await;
    assert_eq!(flushed.rows_flushed, 2);
    assert!(
        writer
            .tables_with_inlined_file_deletes()
            .unwrap()
            .is_empty()
    );
    let files = lake.delete_files().await;
    assert_eq!(files.len(), 1);
    assert_eq!(files[0].1, first);
    assert_eq!(files[0].2, None);
    assert_eq!(files[0].3, 2);
    assert_eq!(files[0].4, Some(second));

    // Read the new delete file: positions 1 and 3, deleted at `first` and
    // `second`.
    let new_object = lake
        .objects()
        .into_iter()
        .find(|object| !before.contains(object))
        .unwrap();
    let file = std::fs::File::open(lake.data.join(&new_object)).unwrap();
    let reader = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file)
        .unwrap()
        .build()
        .unwrap();
    let mut rows = Vec::new();
    for batch in reader {
        let batch = batch.unwrap();
        let pos = batch
            .column_by_name("pos")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow::array::Int64Array>()
            .unwrap()
            .clone();
        let snapshot = batch
            .column_by_name("_ducklake_internal_snapshot_id")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow::array::Int64Array>()
            .unwrap()
            .clone();
        for i in 0..batch.num_rows() {
            rows.push((pos.value(i), snapshot.value(i)));
        }
    }
    assert_eq!(rows, vec![(1, first), (3, second)]);
    assert_eq!(lake.rows(Some(first)).await, ids(&[1, 3, 4]));
    assert_eq!(lake.rows(None).await, ids(&[1, 3]));
}

/// Compaction absorbs inlined deletions; expiring the snapshots that still
/// need the rewritten file removes its inlined deletions.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn rewrite_absorbs_and_expire_removes_inlined_deletes() {
    let lake = Lake::new().await;
    lake.seed(batch(vec![1, 2, 3, 4], vec![10, 20, 30, 40]), 0)
        .await;
    lake.exec(&format!("DELETE FROM {CAT}.public.t WHERE id IN (1, 2)"), 5)
        .await;
    let mut history = Vec::new();
    lake.record(&mut history).await;
    let (table, state) = lake.table_handle(0).await;
    let rewritten = table
        .rewrite_data_files(
            &state,
            datafusion_ducklake::RewriteOptions {
                delete_threshold: 0.0,
                data_file_ids: None,
            },
        )
        .await
        .unwrap();
    assert!(rewritten.did_work());
    lake.record(&mut history).await;
    assert_eq!(history[1].1, ids(&[3, 4]));
    lake.check_history(&history).await;
    // The retired file's inlined deletions serve time travel: a flush leaves
    // them alone.
    assert_eq!(lake.flush_deletes().await.snapshot_id, None);
    assert_eq!(lake.inline_file_deletes().await.len(), 2);

    datafusion_ducklake::MulticatalogManager::new(lake.pool.clone())
        .expire_snapshots_in_catalog(
            CAT,
            datafusion_ducklake::maintenance::ExpireCriteria::OlderThan(
                chrono::Utc::now() + chrono::Duration::days(1),
            ),
        )
        .await
        .unwrap();
    assert!(lake.inline_file_deletes().await.is_empty());
    assert_eq!(lake.rows(None).await, ids(&[3, 4]));
}
