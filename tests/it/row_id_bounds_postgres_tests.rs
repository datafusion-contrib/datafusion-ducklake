//! Per-file `rowid` bounds on a multicatalog Postgres store
//! (`datafusion_ducklake::row_id_bounds`): what writes record, that a lookup
//! by `rowid` finds the same rows with them as without, that it opens no file
//! whose bounds hold no id it asked for, and that compaction keeps them.
//! Docker-gated (testcontainers Postgres).

#![cfg(feature = "write-postgres")]

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow::array::{Int64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::prelude::SessionContext;
use datafusion_ducklake::rowid_lookup::RowLocation;
use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeTable, DuckLakeTableWriter, MergeOptions, MetadataProvider,
    MetadataWriter, MulticatalogManager, MulticatalogProvider, PostgresMetadataWriter,
    execute_ducklake_sql,
};
use object_store::local::LocalFileSystem;
use sqlx::postgres::{PgPool, PgPoolOptions};
use tempfile::TempDir;
use testcontainers::ContainerAsync;
use testcontainers::runners::AsyncRunner;
use testcontainers_modules::postgres::Postgres;

const FILE_ROWS: i64 = 1_000;
const FILES: i64 = 4;

struct Store {
    pool: PgPool,
    _container: ContainerAsync<Postgres>,
    _temp: TempDir,
    data: std::path::PathBuf,
    catalog_id: i64,
}

impl Store {
    /// A store with bounds enabled and one catalog, `cat`, holding
    /// `public.t (id BIGINT, val BIGINT)` loaded by `FILES` appends of
    /// `FILE_ROWS` rows each, so that row `id` has `rowid` `id`.
    async fn loaded() -> Self {
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
        let manager = MulticatalogManager::new(pool.clone()).with_row_id_bounds(true);
        manager.create_row_id_bounds_table().await.unwrap();
        let catalog_id = manager.create_catalog("cat").await.unwrap();
        let temp = TempDir::new().unwrap();
        let data = temp.path().join("data");
        std::fs::create_dir_all(&data).unwrap();
        let store = Self {
            pool,
            _container: container,
            _temp: temp,
            data,
            catalog_id,
        };
        for file in 0..FILES {
            let from = file * FILE_ROWS;
            let ids: Vec<i64> = (from..from + FILE_ROWS).collect();
            let table_writer = DuckLakeTableWriter::new(
                store.writer(true).await,
                Arc::new(LocalFileSystem::new()),
            )
            .unwrap();
            if file == 0 {
                table_writer
                    .write_table("public", "t", &[batch(&ids)])
                    .await
                    .unwrap();
            } else {
                table_writer
                    .append_table("public", "t", &[batch(&ids)])
                    .await
                    .unwrap();
            }
        }
        store
    }

    async fn writer(&self, bounds: bool) -> Arc<PostgresMetadataWriter> {
        let writer = PostgresMetadataWriter::with_pool(self.pool.clone(), self.catalog_id)
            .await
            .unwrap()
            .with_data_inlining_row_limit(0)
            .with_row_id_bounds(bounds);
        writer.set_data_path(self.data.to_str().unwrap()).unwrap();
        Arc::new(writer)
    }

    async fn catalog(&self, bounds: bool) -> DuckLakeCatalog {
        let provider = MulticatalogProvider::with_pool_and_id(self.pool.clone(), self.catalog_id)
            .await
            .unwrap()
            .with_row_id_bounds(bounds);
        DuckLakeCatalog::with_writer(Arc::new(provider), self.writer(bounds).await).unwrap()
    }

    /// `public.t` at the head, read with bounds when `bounds`.
    async fn table(&self, bounds: bool) -> (DuckLakeTable, SessionContext) {
        let ctx = SessionContext::new();
        ctx.register_catalog("cat", Arc::new(self.catalog(bounds).await));
        let provider = ctx
            .catalog("cat")
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
        (table, ctx)
    }

    async fn sql(&self, statement: &str) {
        let ctx = SessionContext::new();
        let catalog = self.catalog(true).await;
        ctx.register_catalog("cat", Arc::new(self.catalog(true).await));
        execute_ducklake_sql(&ctx, &catalog, statement)
            .await
            .unwrap_or_else(|e| panic!("`{statement}`: {e}"))
            .collect()
            .await
            .unwrap_or_else(|e| panic!("`{statement}`: {e}"));
    }

    /// Each live file's path and recorded bounds (`None` when it has none).
    async fn live_bounds(&self) -> BTreeMap<String, Option<(i64, i64)>> {
        let provider = MulticatalogProvider::with_pool_and_id(self.pool.clone(), self.catalog_id)
            .await
            .unwrap();
        let head = provider.get_current_snapshot().unwrap();
        let schema = provider
            .get_schema_by_name("public", head)
            .unwrap()
            .unwrap();
        let table = provider
            .get_table_by_name(schema.schema_id, "t", head)
            .unwrap()
            .unwrap();
        let files = provider
            .get_table_files_for_select(table.table_id, head)
            .unwrap();
        let ids: Vec<i64> = files.iter().map(|file| file.data_file_id).collect();
        let recorded = provider
            .with_row_id_bounds(true)
            .get_file_row_id_bounds(&ids)
            .unwrap();
        files
            .iter()
            .map(|file| {
                (
                    file.file.path.clone(),
                    recorded.get(&file.data_file_id).copied(),
                )
            })
            .collect()
    }

    async fn bounds_rows(&self) -> i64 {
        sqlx::query_scalar("SELECT count(*) FROM ducklake_catalog_data_file_row_id_bounds")
            .fetch_one(&self.pool)
            .await
            .unwrap()
    }

    fn object(&self, path: &str) -> std::path::PathBuf {
        self.data
            .join(format!("cat_{}", self.catalog_id))
            .join("public")
            .join("t")
            .join(path)
    }
}

fn batch(ids: &[i64]) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("val", DataType::Int64, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(ids.to_vec())),
            Arc::new(Int64Array::from(
                ids.iter().map(|id| id * 10).collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap()
}

/// `(rowid, data_file_id, position)` of each located row, sorted.
async fn located(table: &DuckLakeTable, ctx: &SessionContext, ids: &[i64]) -> Vec<(i64, i64, i64)> {
    let mut rows: Vec<(i64, i64, i64)> = table
        .locate_rowids(&ctx.state(), ids)
        .await
        .unwrap()
        .into_iter()
        .map(|row| match row.location {
            RowLocation::File {
                data_file_id,
                position,
                ..
            } => (row.row_id, data_file_id, position),
            other => panic!("rowid {} is not in a file: {other:?}", row.row_id),
        })
        .collect();
    rows.sort_unstable();
    rows
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn inserts_record_their_positional_ranges() {
    let store = Store::loaded().await;
    let mut bounds: Vec<(i64, i64)> = store
        .live_bounds()
        .await
        .into_values()
        .map(|bounds| bounds.expect("every inserted file has bounds"))
        .collect();
    bounds.sort_unstable();
    let expected: Vec<(i64, i64)> = (0..FILES)
        .map(|file| (file * FILE_ROWS, (file + 1) * FILE_ROWS - 1))
        .collect();
    assert_eq!(bounds, expected);
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn an_update_records_the_embedded_rowids_it_wrote() {
    let store = Store::loaded().await;
    let before = store.live_bounds().await;
    store
        .sql("UPDATE cat.public.t SET val = -val WHERE id BETWEEN 2100 AND 2109")
        .await;
    let added: Vec<Option<(i64, i64)>> = store
        .live_bounds()
        .await
        .into_iter()
        .filter(|(path, _)| !before.contains_key(path))
        .map(|(_, bounds)| bounds)
        .collect();
    // The new row versions keep their rows' rowids, not the range the
    // file's `row_id_start` would imply.
    assert_eq!(added, vec![Some((2100, 2109))]);
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn a_lookup_finds_the_same_rows_with_bounds_as_without() {
    let store = Store::loaded().await;
    store
        .sql("UPDATE cat.public.t SET val = -val WHERE id BETWEEN 2100 AND 2109")
        .await;
    store.sql("DELETE FROM cat.public.t WHERE id = 3500").await;
    // Positional rows, updated rows (now in the update's file), a deleted
    // row, and ids no row has.
    let ids = [0, 999, 1000, 2099, 2100, 2105, 2109, 2110, 3500, 3999, 4000, 9_999_999];
    let (with, with_ctx) = store.table(true).await;
    let (without, without_ctx) = store.table(false).await;
    let found = located(&with, &with_ctx, &ids).await;
    assert_eq!(found, located(&without, &without_ctx, &ids).await);
    let found_ids: Vec<i64> = found.iter().map(|(id, ..)| *id).collect();
    assert_eq!(
        found_ids,
        vec![0, 999, 1000, 2099, 2100, 2105, 2109, 2110, 3999]
    );
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn a_lookup_opens_no_file_its_bounds_rule_out() {
    let store = Store::loaded().await;
    store
        .sql("UPDATE cat.public.t SET val = -val WHERE id BETWEEN 2100 AND 2109")
        .await;
    // Remove every object that holds none of the ids looked up: a lookup that
    // opened one would fail.
    let wanted = [10, 3990];
    for (path, bounds) in store.live_bounds().await {
        let (min, max) = bounds.expect("every file has bounds");
        if !wanted.iter().any(|id| (min..=max).contains(id)) {
            std::fs::remove_file(store.object(&path)).unwrap();
        }
    }
    let (with, ctx) = store.table(true).await;
    let found: Vec<i64> = located(&with, &ctx, &wanted)
        .await
        .into_iter()
        .map(|(id, ..)| id)
        .collect();
    assert_eq!(found, wanted);
    // Without bounds, the lookup opens them, and fails.
    let (without, ctx) = store.table(false).await;
    assert!(without.locate_rowids(&ctx.state(), &wanted).await.is_err());
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn compaction_replaces_its_sources_bounds_with_its_outputs() {
    let store = Store::loaded().await;
    store
        .sql("UPDATE cat.public.t SET val = -val WHERE id BETWEEN 2100 AND 2109")
        .await;
    let (table, ctx) = store.table(true).await;
    let result = table
        .merge_adjacent_files(&ctx.state(), MergeOptions::default())
        .await
        .unwrap();
    assert!(result.files_created > 0, "the merge wrote something");
    let live = store.live_bounds().await;
    assert!(
        live.values().all(Option::is_some),
        "every live file has bounds after the merge: {live:?}"
    );
    // The retired sources' rows went with them.
    assert_eq!(store.bounds_rows().await, live.len() as i64);
    // The merged files still find every row.
    let ids: Vec<i64> = (0..FILES * FILE_ROWS).step_by(97).collect();
    let (with, with_ctx) = store.table(true).await;
    let (without, without_ctx) = store.table(false).await;
    assert_eq!(
        located(&with, &with_ctx, &ids).await,
        located(&without, &without_ctx, &ids).await
    );
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn a_writer_without_bounds_records_none() {
    let store = Store::loaded().await;
    let recorded = store.bounds_rows().await;
    DuckLakeTableWriter::new(store.writer(false).await, Arc::new(LocalFileSystem::new()))
        .unwrap()
        .append_table("public", "t", &[batch(&[5000, 5001])])
        .await
        .unwrap();
    assert_eq!(store.bounds_rows().await, recorded);
    // Its file has no bounds, so a lookup still reads it: its rows took the
    // next rowids, `FILES * FILE_ROWS` on.
    let next = FILES * FILE_ROWS;
    let (with, ctx) = store.table(true).await;
    let found: Vec<i64> = located(&with, &ctx, &[next + 1])
        .await
        .into_iter()
        .map(|(id, ..)| id)
        .collect();
    assert_eq!(found, vec![next + 1]);
}
