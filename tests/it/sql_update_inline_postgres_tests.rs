//! SQL `UPDATE` with DuckLake data inlining on the multicatalog PostgreSQL
//! backend.
//!
//! An UPDATE ends the old version of every matching row and writes a new
//! version, in one snapshot. Old versions held inline are ended in their
//! inlined data table; old versions held in Parquet get a positional delete.
//! New versions are stored inline when the writer's `data_inlining_row_limit`
//! admits them, otherwise in a Parquet file, and keep the row id of the version
//! they replace. These tests read the shared inlined tables directly to check
//! that, and read the table through time travel and after a flush.
//! Docker-gated (testcontainers Postgres).

#![cfg(feature = "write-postgres")]

use std::sync::Arc;

use arrow::array::{Array, Int32Array, RecordBatch, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::prelude::*;
use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeTableWriter, DuckLakeWriteOptions, MetadataProvider, MetadataWriter,
    MulticatalogManager, MulticatalogProvider, PostgresMetadataWriter,
};
use object_store::local::LocalFileSystem;
use sqlx::Row;
use sqlx::postgres::{PgPool, PgPoolOptions};
use tempfile::TempDir;
use testcontainers::ContainerAsync;
use testcontainers::runners::AsyncRunner;
use testcontainers_modules::postgres::Postgres;

pub(crate) const CAT: &str = "cat";

pub(crate) struct Lake {
    pub(crate) pool: PgPool,
    pub(crate) cat: i64,
    pub(crate) data: std::path::PathBuf,
    _tmp: TempDir,
    _container: ContainerAsync<Postgres>,
}

pub(crate) fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("val", DataType::Int32, true),
    ]))
}

pub(crate) fn batch(ids: Vec<i32>, vals: Vec<i32>) -> RecordBatch {
    RecordBatch::try_new(
        schema(),
        vec![Arc::new(Int32Array::from(ids)), Arc::new(Int32Array::from(vals))],
    )
    .unwrap()
}

impl Lake {
    pub(crate) async fn new() -> Self {
        let container = Postgres::default().start().await.unwrap();
        let port = container.get_host_port_ipv4(5432).await.unwrap();
        let pool = PgPoolOptions::new()
            .max_connections(5)
            .connect(&format!(
                "postgresql://postgres:postgres@127.0.0.1:{port}/postgres"
            ))
            .await
            .unwrap();
        datafusion_ducklake::initialize_multicatalog_schema(&pool)
            .await
            .unwrap();
        let cat = MulticatalogManager::new(pool.clone())
            .create_catalog(CAT)
            .await
            .unwrap();
        let tmp = TempDir::new().unwrap();
        let data = tmp.path().join("data");
        std::fs::create_dir_all(&data).unwrap();
        Self {
            pool,
            cat,
            data,
            _tmp: tmp,
            _container: container,
        }
    }

    pub(crate) async fn writer(&self) -> Arc<PostgresMetadataWriter> {
        let writer = PostgresMetadataWriter::with_pool(self.pool.clone(), self.cat)
            .await
            .unwrap();
        writer.set_data_path(self.data.to_str().unwrap()).unwrap();
        Arc::new(writer)
    }

    pub(crate) async fn table_writer(&self, limit: usize) -> DuckLakeTableWriter {
        DuckLakeTableWriter::new(self.writer().await, Arc::new(LocalFileSystem::new()))
            .unwrap()
            .with_options(&DuckLakeWriteOptions::default().with_data_inlining_row_limit(limit))
    }

    /// Write `rows` to `public.t` (creating it), inline when within `limit`.
    pub(crate) async fn seed(&self, rows: RecordBatch, limit: usize) {
        self.table_writer(limit)
            .await
            .append_table("public", "t", &[rows])
            .await
            .unwrap();
    }

    pub(crate) async fn provider(&self) -> Arc<MulticatalogProvider> {
        Arc::new(
            MulticatalogProvider::with_pool(self.pool.clone(), CAT)
                .await
                .unwrap(),
        )
    }

    /// A fresh writable session bound to the latest snapshot.
    pub(crate) async fn ctx(&self, limit: usize) -> SessionContext {
        let catalog = DuckLakeCatalog::with_writer(self.provider().await, self.writer().await)
            .unwrap()
            .with_write_options(
                DuckLakeWriteOptions::default().with_data_inlining_row_limit(limit),
            );
        let ctx = SessionContext::new();
        ctx.register_catalog(CAT, Arc::new(catalog));
        ctx
    }

    /// Run one DML statement in a fresh session and return its row count.
    pub(crate) async fn exec(&self, sql: &str, limit: usize) -> u64 {
        let batches = self
            .ctx(limit)
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

    pub(crate) async fn head(&self) -> i64 {
        self.provider().await.get_current_snapshot().unwrap()
    }

    /// `(id, val)` rows of `public.t`, at `snapshot` or at the head.
    pub(crate) async fn rows(&self, snapshot: Option<i64>) -> Vec<(i32, Option<i32>)> {
        let provider = self.provider().await;
        let snapshot = match snapshot {
            Some(snapshot) => snapshot,
            None => provider.get_current_snapshot().unwrap(),
        };
        let catalog = DuckLakeCatalog::with_snapshot(provider, snapshot).unwrap();
        let ctx = SessionContext::new();
        ctx.register_catalog(CAT, Arc::new(catalog));
        let batches = ctx
            .sql(&format!("SELECT id, val FROM {CAT}.public.t ORDER BY id"))
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
                out.push((ids.value(r), vals.is_valid(r).then(|| vals.value(r))));
            }
        }
        out
    }

    /// Every stored inlined row version of `public.t`:
    /// `(row_id, begin_snapshot, end_snapshot, id, val)`, ordered.
    pub(crate) async fn inline_versions(&self) -> Vec<(i64, i64, Option<i64>, i32, Option<i32>)> {
        let table_id = self.table_id("t").await;
        let column = |name: &'static str| {
            let pool = self.pool.clone();
            async move {
                sqlx::query_scalar::<_, i64>(
                    "SELECT column_id FROM ducklake_column
                     WHERE table_id = $1 AND column_name = $2 AND end_snapshot IS NULL",
                )
                .bind(table_id)
                .bind(name)
                .fetch_one(&pool)
                .await
                .unwrap()
            }
        };
        let (id_column, val_column) = (column("id").await, column("val").await);
        let rows = sqlx::query(
            "SELECT row_id, begin_snapshot, end_snapshot, data FROM ducklake_inlined_row
             WHERE table_id = $1",
        )
        .bind(table_id)
        .fetch_all(&self.pool)
        .await
        .unwrap();
        let mut out = Vec::new();
        for row in rows {
            let cells = decode_inline_cells(&row.get::<Vec<u8>, _>(3));
            let int = |column: i64| {
                cells
                    .get(&column)
                    .cloned()
                    .flatten()
                    .map(|text| text.parse::<i32>().unwrap())
            };
            out.push((
                row.get::<i64, _>(0),
                row.get::<i64, _>(1),
                row.get::<Option<i64>, _>(2),
                int(id_column).unwrap(),
                int(val_column),
            ));
        }
        out.sort();
        out
    }

    /// The live table id of `public.<name>`.
    pub(crate) async fn table_id(&self, name: &str) -> i64 {
        sqlx::query_scalar(
            "SELECT table_id FROM ducklake_table WHERE table_name = $1 AND end_snapshot IS NULL",
        )
        .bind(name)
        .fetch_one(&self.pool)
        .await
        .unwrap()
    }

    /// `(live data files, live delete files)`.
    pub(crate) async fn live_files(&self) -> (i64, i64) {
        let data: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM ducklake_data_file WHERE end_snapshot IS NULL",
        )
        .fetch_one(&self.pool)
        .await
        .unwrap();
        let deletes: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM ducklake_delete_file WHERE end_snapshot IS NULL",
        )
        .fetch_one(&self.pool)
        .await
        .unwrap();
        (data, deletes)
    }

    /// Every inlined deletion of a Parquet row of `public.t`:
    /// `(file_id, row_id, begin_snapshot)`, ordered.
    pub(crate) async fn inline_file_deletes(&self) -> Vec<(i64, i64, i64)> {
        let table_id = self.table_id("t").await;
        sqlx::query(
            "SELECT file_id, row_id, begin_snapshot FROM ducklake_inlined_file_delete
             WHERE table_id = $1 ORDER BY 1, 2, 3",
        )
        .bind(table_id)
        .fetch_all(&self.pool)
        .await
        .unwrap()
        .into_iter()
        .map(|row| (row.get(0), row.get(1), row.get(2)))
        .collect()
    }

    /// Every object under the data directory (relative paths, sorted).
    pub(crate) fn objects(&self) -> Vec<String> {
        fn walk(dir: &std::path::Path, root: &std::path::Path, out: &mut Vec<String>) {
            for entry in std::fs::read_dir(dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    walk(&path, root, out);
                } else {
                    out.push(path.strip_prefix(root).unwrap().display().to_string());
                }
            }
        }
        let mut out = Vec::new();
        walk(&self.data, &self.data, &mut out);
        out.sort();
        out
    }

    /// `SELECT COUNT(*)` of `public.t`, at `snapshot` or at the head.
    pub(crate) async fn count(&self, snapshot: Option<i64>) -> i64 {
        let provider = self.provider().await;
        let snapshot = match snapshot {
            Some(snapshot) => snapshot,
            None => provider.get_current_snapshot().unwrap(),
        };
        let catalog = DuckLakeCatalog::with_snapshot(provider, snapshot).unwrap();
        let ctx = SessionContext::new();
        ctx.register_catalog(CAT, Arc::new(catalog));
        let batches = ctx
            .sql(&format!("SELECT COUNT(*) FROM {CAT}.public.t"))
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<arrow::array::Int64Array>()
            .unwrap()
            .value(0)
    }

    /// Read what a flush of `public.t` at the current head would write: the
    /// visible inlined rows and the head they were read at.
    pub(crate) async fn flush_inputs(
        &self,
    ) -> (
        Vec<datafusion_ducklake::metadata_provider::DuckLakeInlinedData>,
        i64,
    ) {
        let provider = self.provider().await;
        let head = provider.get_current_snapshot().unwrap();
        let schema = provider
            .get_schema_by_name("public", head)
            .unwrap()
            .unwrap();
        let table = provider
            .get_table_by_name(schema.schema_id, "t", head)
            .unwrap()
            .unwrap();
        let columns = provider.get_table_structure(table.table_id, head).unwrap();
        let live = provider
            .get_inlined_data_with_row_ids(table.table_id, head, &columns)
            .unwrap();
        (live, head)
    }

    /// Commit a flush of rows read by [`Self::flush_inputs`].
    pub(crate) async fn try_flush(
        &self,
        inputs: &[datafusion_ducklake::metadata_provider::DuckLakeInlinedData],
        base: i64,
    ) -> datafusion_ducklake::Result<Option<datafusion_ducklake::WriteResult>> {
        self.table_writer(0)
            .await
            .flush_inlined_data("public", "t", inputs, base)
            .await
    }

    /// Flush every visible inlined row of `public.t` to Parquet.
    pub(crate) async fn flush(&self) -> i64 {
        let (inputs, head) = self.flush_inputs().await;
        self.try_flush(&inputs, head)
            .await
            .unwrap()
            .expect("there were inlined rows to flush")
            .snapshot_id
    }
}

/// Decodes the `data` cell of a `ducklake_inlined_row` row into
/// `column_id -> value` (NULL as `None`; strings and text as UTF-8). Written
/// from the format's documentation, independently of the library's decoder.
pub(crate) fn decode_inline_cells(data: &[u8]) -> std::collections::HashMap<i64, Option<String>> {
    assert_eq!(data[0], 1, "format version");
    let count = u32::from_le_bytes(data[1..5].try_into().unwrap()) as usize;
    let mut at = 5;
    let mut cells = std::collections::HashMap::new();
    for _ in 0..count {
        let column_id = i64::from_le_bytes(data[at..at + 8].try_into().unwrap());
        let tag = data[at + 8];
        at += 9;
        let value = match tag {
            0 => None,
            1 | 2 => {
                let len = u32::from_le_bytes(data[at..at + 4].try_into().unwrap()) as usize;
                at += 4;
                let value = String::from_utf8(data[at..at + len].to_vec()).unwrap();
                at += len;
                Some(value)
            },
            other => panic!("unknown tag {other}"),
        };
        cells.insert(column_id, value);
    }
    assert_eq!(at, data.len(), "trailing bytes");
    cells
}

pub(crate) fn some(rows: &[(i32, i32)]) -> Vec<(i32, Option<i32>)> {
    rows.iter().map(|(id, val)| (*id, Some(*val))).collect()
}

/// Updating inlined rows ends their old versions and writes new versions
/// inline with the same row ids, in one snapshot, with no object-store write.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn update_inlined_rows_stays_inline() {
    let lake = Lake::new().await;
    lake.seed(batch(vec![1, 2, 3], vec![10, 20, 30]), 10).await;
    assert_eq!(lake.live_files().await, (0, 0));
    let before = lake.head().await;
    let before_versions = lake.inline_versions().await;

    let n = lake
        .exec(
            &format!(
                "UPDATE {CAT}.public.t SET val = CASE id WHEN 1 THEN 100 WHEN 3 THEN 300 END \
                 WHERE id IN (1, 3)"
            ),
            10,
        )
        .await;
    assert_eq!(n, 2);
    let after = lake.head().await;
    assert_eq!(after, before + 1, "one snapshot");

    assert_eq!(lake.rows(None).await, some(&[(1, 100), (2, 20), (3, 300)]));
    assert_eq!(
        lake.rows(Some(before)).await,
        some(&[(1, 10), (2, 20), (3, 30)])
    );
    assert_eq!(lake.live_files().await, (0, 0), "no Parquet written");

    let rid = |id: i32| {
        before_versions
            .iter()
            .find(|v| v.3 == id)
            .map(|v| v.0)
            .unwrap()
    };
    let begin = before_versions[0].1;
    let mut expected = vec![
        (rid(1), begin, Some(after), 1, Some(10)),
        (rid(1), after, None, 1, Some(100)),
        (rid(2), begin, None, 2, Some(20)),
        (rid(3), begin, Some(after), 3, Some(30)),
        (rid(3), after, None, 3, Some(300)),
    ];
    expected.sort();
    assert_eq!(lake.inline_versions().await, expected);

    let count = lake
        .ctx(10)
        .await
        .sql(&format!("SELECT COUNT(*) FROM {CAT}.public.t"))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let count = count[0]
        .column(0)
        .as_any()
        .downcast_ref::<arrow::array::Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!(count, 3);
}

/// Updating Parquet rows with inlining on: the old versions get a positional
/// delete, the new versions go inline with the Parquet rows' row ids.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn update_parquet_rows_writes_inline_versions() {
    let lake = Lake::new().await;
    lake.seed(batch(vec![1, 2, 3, 4], vec![10, 20, 30, 40]), 0)
        .await;
    assert_eq!(lake.live_files().await, (1, 0));
    let before = lake.head().await;

    let n = lake
        .exec(
            &format!("UPDATE {CAT}.public.t SET val = val + 1 WHERE id IN (2, 4)"),
            10,
        )
        .await;
    assert_eq!(n, 2);
    let after = lake.head().await;
    assert_eq!(after, before + 1);

    assert_eq!(
        lake.rows(None).await,
        some(&[(1, 10), (2, 21), (3, 30), (4, 41)])
    );
    assert_eq!(
        lake.rows(Some(before)).await,
        some(&[(1, 10), (2, 20), (3, 30), (4, 40)])
    );
    // No new data file and no delete file: the old versions' ends are
    // inlined deletions of positions 1 and 3.
    assert_eq!(
        lake.live_files().await,
        (1, 0),
        "no new data or delete file"
    );
    let file_id = lake.inline_file_deletes().await[0].0;
    assert_eq!(
        lake.inline_file_deletes().await,
        vec![(file_id, 1, after), (file_id, 3, after)]
    );
    // Parquet row ids are row_id_start (0) + position.
    assert_eq!(
        lake.inline_versions().await,
        vec![(1, after, None, 2, Some(21)), (3, after, None, 4, Some(41))]
    );
}

/// One UPDATE matching inlined and Parquet rows commits both kinds of end and
/// the new versions in one snapshot.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn update_mixed_inlined_and_parquet_rows() {
    let lake = Lake::new().await;
    lake.seed(batch(vec![1, 2, 3], vec![10, 20, 30]), 0).await;
    let n = lake
        .exec(
            &format!("INSERT INTO {CAT}.public.t VALUES (4, 40), (5, 50)"),
            10,
        )
        .await;
    assert_eq!(n, 2);
    assert_eq!(lake.live_files().await, (1, 0));
    let before = lake.head().await;

    let n = lake
        .exec(
            &format!("UPDATE {CAT}.public.t SET val = -val WHERE id IN (2, 4)"),
            10,
        )
        .await;
    assert_eq!(n, 2);
    let after = lake.head().await;
    assert_eq!(after, before + 1);
    assert_eq!(
        lake.rows(None).await,
        some(&[(1, 10), (2, -20), (3, 30), (4, -40), (5, 50)])
    );
    assert_eq!(
        lake.rows(Some(before)).await,
        some(&[(1, 10), (2, 20), (3, 30), (4, 40), (5, 50)])
    );
    assert_eq!(lake.live_files().await, (1, 0));
    assert_eq!(lake.inline_file_deletes().await.len(), 1);
    let live: Vec<_> = lake
        .inline_versions()
        .await
        .into_iter()
        .filter(|v| v.2.is_none())
        .map(|v| (v.0, v.3, v.4))
        .collect();
    // Row ids: Parquet rows 0..3, the inlined insert 3..5.
    assert_eq!(
        live,
        vec![(1, 2, Some(-20)), (3, 4, Some(-40)), (4, 5, Some(50))]
    );
}

/// Repeated updates of the same row each end the previous version; every
/// snapshot reads its own value.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn repeated_updates_of_the_same_row() {
    let lake = Lake::new().await;
    lake.seed(batch(vec![1, 2], vec![10, 20]), 0).await;
    let mut history = vec![(lake.head().await, 10)];
    for value in [11, 12, 13] {
        let n = lake
            .exec(
                &format!("UPDATE {CAT}.public.t SET val = {value} WHERE id = 1"),
                10,
            )
            .await;
        assert_eq!(n, 1);
        history.push((lake.head().await, value));
    }
    for (snapshot, value) in &history {
        assert_eq!(
            lake.rows(Some(*snapshot)).await,
            some(&[(1, *value), (2, 20)]),
            "snapshot {snapshot}"
        );
    }
    let versions = lake.inline_versions().await;
    assert_eq!(versions.len(), 3);
    assert!(
        versions.iter().all(|v| v.0 == 0),
        "row id 0 kept: {versions:?}"
    );
    assert_eq!(versions.iter().filter(|v| v.2.is_none()).count(), 1);
    // The Parquet row got one inlined deletion; later updates only touch
    // inlined versions.
    assert_eq!(lake.live_files().await, (1, 0));
    assert_eq!(lake.inline_file_deletes().await.len(), 1);
}

/// A DELETE after an UPDATE ends the live new version only.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn update_then_delete() {
    let lake = Lake::new().await;
    lake.seed(batch(vec![1, 2, 3], vec![10, 20, 30]), 10).await;
    let seeded = lake.head().await;
    lake.exec(
        &format!("UPDATE {CAT}.public.t SET val = 99 WHERE id < 3"),
        10,
    )
    .await;
    let updated = lake.head().await;
    let n = lake
        .exec(&format!("DELETE FROM {CAT}.public.t WHERE id = 1"), 10)
        .await;
    assert_eq!(n, 1);
    let deleted = lake.head().await;

    assert_eq!(lake.rows(None).await, some(&[(2, 99), (3, 30)]));
    assert_eq!(
        lake.rows(Some(updated)).await,
        some(&[(1, 99), (2, 99), (3, 30)])
    );
    assert_eq!(
        lake.rows(Some(seeded)).await,
        some(&[(1, 10), (2, 20), (3, 30)])
    );
    let row1: Vec<_> = lake
        .inline_versions()
        .await
        .into_iter()
        .filter(|v| v.3 == 1)
        .map(|v| (v.1, v.2, v.4))
        .collect();
    assert_eq!(
        row1,
        vec![(seeded, Some(updated), Some(10)), (updated, Some(deleted), Some(99))]
    );
}

/// An UPDATE that matches nothing publishes no snapshot.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn update_with_zero_matches_is_a_no_op() {
    let lake = Lake::new().await;
    lake.seed(batch(vec![1, 2], vec![10, 20]), 0).await;
    lake.exec(&format!("INSERT INTO {CAT}.public.t VALUES (3, 30)"), 10)
        .await;
    let before = lake.head().await;
    let versions = lake.inline_versions().await;
    let n = lake
        .exec(
            &format!("UPDATE {CAT}.public.t SET val = 0 WHERE id = 99"),
            10,
        )
        .await;
    assert_eq!(n, 0);
    assert_eq!(lake.head().await, before);
    assert_eq!(lake.inline_versions().await, versions);
    assert_eq!(lake.live_files().await, (1, 0));
}

/// More updated rows than the inline limit: the new versions go to a Parquet
/// file that embeds their row ids, and the inlined old versions still end in
/// the same snapshot.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn update_above_the_inline_limit_writes_parquet() {
    let lake = Lake::new().await;
    lake.seed(batch(vec![1, 2, 3], vec![10, 20, 30]), 10).await;
    let before = lake.head().await;

    let n = lake
        .exec(&format!("UPDATE {CAT}.public.t SET val = val * 2"), 2)
        .await;
    assert_eq!(n, 3);
    let after = lake.head().await;
    assert_eq!(after, before + 1);
    assert_eq!(lake.rows(None).await, some(&[(1, 20), (2, 40), (3, 60)]));
    assert_eq!(
        lake.rows(Some(before)).await,
        some(&[(1, 10), (2, 20), (3, 30)])
    );
    assert_eq!(lake.live_files().await, (1, 0));
    let versions = lake.inline_versions().await;
    assert_eq!(versions.len(), 3);
    assert!(versions.iter().all(|v| v.2 == Some(after)));

    // The Parquet file's rows keep their row ids: a row-lineage read now works
    // (no inlined rows remain) and returns the original ids.
    let catalog = DuckLakeCatalog::with_snapshot(lake.provider().await, lake.head().await)
        .unwrap()
        .with_row_lineage(true);
    let ctx = SessionContext::new();
    ctx.register_catalog(CAT, Arc::new(catalog));
    let batches = ctx
        .sql(&format!("SELECT rowid, id FROM {CAT}.public.t ORDER BY id"))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let rowids = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<arrow::array::Int64Array>()
        .unwrap();
    let mut expected: Vec<i64> = versions.iter().map(|v| v.0).collect();
    expected.sort();
    assert_eq!(rowids.values().to_vec(), expected);

    // A following small UPDATE of a row now in Parquet goes inline again.
    lake.exec(
        &format!("UPDATE {CAT}.public.t SET val = 1 WHERE id = 2"),
        2,
    )
    .await;
    assert_eq!(lake.rows(None).await, some(&[(1, 20), (2, 1), (3, 60)]));
    assert_eq!(lake.live_files().await, (1, 0));
    assert_eq!(lake.inline_file_deletes().await.len(), 1);
}

/// Flushing after a mix of inline inserts, updates and deletes writes exactly
/// the live rows, and every earlier snapshot still reads its own rows.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn flush_after_inline_updates() {
    let lake = Lake::new().await;
    lake.seed(batch(vec![1, 2, 3], vec![10, 20, 30]), 0).await;
    let mut history = vec![(lake.head().await, lake.rows(None).await)];
    for sql in [
        format!("INSERT INTO {CAT}.public.t VALUES (4, 40), (5, 50), (6, 60)"),
        format!("UPDATE {CAT}.public.t SET val = val + 1 WHERE id IN (1, 4)"),
        format!("UPDATE {CAT}.public.t SET val = val + 1 WHERE id IN (1, 5)"),
        format!("DELETE FROM {CAT}.public.t WHERE id IN (2, 5)"),
        format!("UPDATE {CAT}.public.t SET val = NULL WHERE id = 6"),
        format!("DELETE FROM {CAT}.public.t WHERE id = 4"),
    ] {
        lake.exec(&sql, 10).await;
        history.push((lake.head().await, lake.rows(None).await));
    }
    let live = vec![(1, Some(12)), (3, Some(30)), (6, None)];
    assert_eq!(history.last().unwrap().1, live);

    let flushed = lake.flush().await;
    assert_eq!(lake.rows(None).await, live);
    assert!(
        lake.inline_versions().await.iter().all(|v| v.2.is_some()),
        "no live inlined row after the flush"
    );
    for (snapshot, rows) in &history {
        assert_eq!(
            &lake.rows(Some(*snapshot)).await,
            rows,
            "snapshot {snapshot}"
        );
    }
    assert!(flushed > history.last().unwrap().0);

    // UPDATE and DELETE keep working on the flushed rows.
    lake.exec(
        &format!("UPDATE {CAT}.public.t SET val = 7 WHERE id = 6"),
        10,
    )
    .await;
    lake.exec(&format!("DELETE FROM {CAT}.public.t WHERE id = 3"), 10)
        .await;
    assert_eq!(lake.rows(None).await, vec![(1, Some(12)), (6, Some(7))]);
    assert_eq!(lake.rows(Some(flushed)).await, live);

    // A second flush picks up the new inline version only.
    let second = lake.flush().await;
    assert_eq!(lake.rows(None).await, vec![(1, Some(12)), (6, Some(7))]);
    assert_eq!(lake.rows(Some(flushed)).await, live);
    assert!(second > flushed);
}

/// The UPDATE's predicate is still pushed into the Parquet reader while
/// inlined rows are matched in memory.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn keyed_update_on_a_table_with_row_groups_and_inlined_rows() {
    let lake = Lake::new().await;
    let ids: Vec<i32> = (0..40).collect();
    let vals: Vec<i32> = ids.iter().map(|id| id * 10).collect();
    DuckLakeTableWriter::new(lake.writer().await, Arc::new(LocalFileSystem::new()))
        .unwrap()
        .with_max_row_group_rows(4)
        .append_table("public", "t", &[batch(ids, vals)])
        .await
        .unwrap();
    lake.exec(
        &format!("INSERT INTO {CAT}.public.t VALUES (100, 1000)"),
        10,
    )
    .await;

    let n = lake
        .exec(
            &format!(
                "UPDATE {CAT}.public.t SET val = CASE id WHEN 22 THEN -1 WHEN 100 THEN -2 END \
                 WHERE id IN (22, 100)"
            ),
            10,
        )
        .await;
    assert_eq!(n, 2);
    let rows = lake.rows(None).await;
    assert_eq!(rows.len(), 41);
    assert!(rows.contains(&(22, Some(-1))));
    assert!(rows.contains(&(100, Some(-2))));
    assert!(rows.contains(&(21, Some(210))));
    assert!(rows.contains(&(23, Some(230))));

    // Only the inlined row matches: nothing new for the Parquet file.
    let n = lake
        .exec(
            &format!("UPDATE {CAT}.public.t SET val = -3 WHERE id = 100"),
            10,
        )
        .await;
    assert_eq!(n, 1);
    assert_eq!(lake.live_files().await, (1, 0));
    assert_eq!(lake.inline_file_deletes().await.len(), 1);
    assert!(lake.rows(None).await.contains(&(100, Some(-3))));
}

/// An UPDATE planned before a flush that ended its inlined rows conflicts and
/// changes nothing; so does one planned before a concurrent inlined DELETE.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn stale_inline_update_conflicts() {
    let lake = Lake::new().await;
    lake.seed(batch(vec![1, 2], vec![10, 20]), 10).await;

    // Concurrent flush.
    let stale = lake.ctx(10).await;
    let plan = stale
        .sql(&format!("UPDATE {CAT}.public.t SET val = 0 WHERE id = 1"))
        .await
        .unwrap();
    lake.flush().await;
    let head = lake.head().await;
    let error = plan
        .collect()
        .await
        .expect_err("stale UPDATE must conflict");
    assert!(datafusion_ducklake::is_conflict(&error), "{error}");
    assert_eq!(lake.head().await, head);
    assert_eq!(lake.rows(None).await, some(&[(1, 10), (2, 20)]));

    // Concurrent inlined DELETE.
    lake.exec(&format!("INSERT INTO {CAT}.public.t VALUES (3, 30)"), 10)
        .await;
    let stale = lake.ctx(10).await;
    let plan = stale
        .sql(&format!("UPDATE {CAT}.public.t SET val = 0 WHERE id = 3"))
        .await
        .unwrap();
    lake.exec(&format!("DELETE FROM {CAT}.public.t WHERE id = 3"), 10)
        .await;
    let head = lake.head().await;
    let error = plan
        .collect()
        .await
        .expect_err("stale UPDATE must conflict");
    assert!(datafusion_ducklake::is_conflict(&error), "{error}");
    assert_eq!(lake.head().await, head);
    assert_eq!(lake.rows(None).await, some(&[(1, 10), (2, 20)]));

    // Concurrent UPDATE of the same Parquet row (positional compare-and-swap).
    let stale = lake.ctx(10).await;
    let plan = stale
        .sql(&format!("UPDATE {CAT}.public.t SET val = 0 WHERE id = 2"))
        .await
        .unwrap();
    lake.exec(
        &format!("UPDATE {CAT}.public.t SET val = 21 WHERE id = 2"),
        10,
    )
    .await;
    let error = plan
        .collect()
        .await
        .expect_err("stale UPDATE must conflict");
    assert!(datafusion_ducklake::is_conflict(&error), "{error}");
    assert_eq!(lake.rows(None).await, some(&[(1, 10), (2, 21)]));
}

/// The writer the multicatalog backend exposes advertises inline UPDATE.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn postgres_writer_supports_inline_update() {
    let lake = Lake::new().await;
    let writer: Arc<dyn MetadataWriter> = lake.writer().await;
    assert!(writer.supports_inline_update());
}
