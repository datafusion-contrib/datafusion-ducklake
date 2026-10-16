//! Postgres multicatalog counterpart of `file_ordering_tests.rs`: the
//! experimental `ext_file_order` sidecar through `PostgresMetadataWriter` and
//! `MulticatalogProvider`, including a writer role without DDL rights.
//! Docker-gated (testcontainers Postgres).

#![cfg(all(feature = "write-postgres", feature = "ext-file-order"))]

use std::sync::Arc;

use arrow::array::{Int64Array, RecordBatch, TimestampMicrosecondArray};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit};
use datafusion::datasource::MemTable;
use datafusion::execution::context::SessionConfig;
use datafusion::physical_plan::displayable;
use datafusion::prelude::*;
use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeTable, DuckLakeTableWriter, DuckLakeWriteOptions, MetadataWriter,
    MulticatalogManager, MulticatalogProvider, NullOrder, PostgresMetadataWriter, SortDirection,
    SortField, VerifyOrderOptions,
};
use object_store::local::LocalFileSystem;
use sqlx::postgres::{PgPool, PgPoolOptions};
use tempfile::TempDir;
use testcontainers::ContainerAsync;
use testcontainers::runners::AsyncRunner;
use testcontainers_modules::postgres::Postgres;

const TOP_N: &str = "SELECT ts, id FROM cat.public.logs ORDER BY ts DESC LIMIT 7";

struct Env {
    pool: PgPool,
    port: u16,
    catalog_id: i64,
    table_id: i64,
    data: TempDir,
    _container: ContainerAsync<Postgres>,
}

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            false,
        ),
        Field::new("id", DataType::Int64, false),
    ]))
}

/// One row per `ts`, in the order given; `id` = `ts`.
fn batch(ts: impl IntoIterator<Item = i64>) -> RecordBatch {
    let ts: Vec<i64> = ts.into_iter().collect();
    RecordBatch::try_new(
        schema(),
        vec![Arc::new(TimestampMicrosecondArray::from(ts.clone())), Arc::new(Int64Array::from(ts))],
    )
    .unwrap()
}

fn ext_options(enabled: bool) -> DuckLakeWriteOptions {
    let mut options = DuckLakeWriteOptions::default().with_data_inlining_row_limit(0);
    options.ext_file_order = Some(enabled);
    // Rows land in the order given, so a test decides which files are sorted.
    options.sort_on_insert = Some(false);
    options
}

impl Env {
    /// A catalog `cat` with `public.logs` sorted by `ts DESC`, no files yet.
    async fn new() -> Self {
        let container = Postgres::default().start().await.unwrap();
        let port = container.get_host_port_ipv4(5432).await.unwrap();
        let pool = pool_for(port, "postgres", "postgres").await;
        datafusion_ducklake::initialize_multicatalog_schema(&pool)
            .await
            .unwrap();
        let catalog_id = MulticatalogManager::new(pool.clone())
            .create_catalog("cat")
            .await
            .unwrap();
        let data = TempDir::new().unwrap();
        let mut env = Self {
            pool,
            port,
            catalog_id,
            table_id: 0,
            data,
            _container: container,
        };
        let writer = env.writer(&env.pool).await;
        writer
            .set_data_path(env.data.path().to_str().unwrap())
            .unwrap();
        let created = DuckLakeTableWriter::new(writer.clone(), Arc::new(LocalFileSystem::new()))
            .unwrap()
            .with_options(&ext_options(false))
            .write_table("public", "logs", &[batch([0])])
            .await
            .unwrap();
        env.table_id = created.table_id;
        writer
            .set_sort_spec(
                env.table_id,
                &[SortField::column(0, "ts", SortDirection::Desc, NullOrder::NullsFirst)],
            )
            .unwrap();
        env
    }

    async fn writer(&self, pool: &PgPool) -> Arc<PostgresMetadataWriter> {
        Arc::new(
            PostgresMetadataWriter::with_pool(pool.clone(), self.catalog_id)
                .await
                .unwrap()
                .with_data_inlining_row_limit(0),
        )
    }

    async fn append(&self, pool: &PgPool, rows: RecordBatch, ext_file_order: bool) {
        DuckLakeTableWriter::new(self.writer(pool).await, Arc::new(LocalFileSystem::new()))
            .unwrap()
            .with_options(&ext_options(ext_file_order))
            .append_table("public", "logs", &[rows])
            .await
            .unwrap();
    }

    async fn ctx(&self, partitions: usize) -> SessionContext {
        let provider = MulticatalogProvider::with_pool(self.pool.clone(), "cat")
            .await
            .unwrap();
        let catalog =
            DuckLakeCatalog::with_writer(Arc::new(provider), self.writer(&self.pool).await)
                .unwrap()
                .with_ext_file_order(true);
        let ctx = SessionContext::new_with_config(
            SessionConfig::new()
                .with_target_partitions(partitions)
                .with_batch_size(64),
        );
        ctx.register_catalog("cat", Arc::new(catalog));
        ctx
    }

    async fn sidecar_rows(&self) -> i64 {
        let exists: bool =
            sqlx::query_scalar("SELECT to_regclass('ext_ducklake_file_order') IS NOT NULL")
                .fetch_one(&self.pool)
                .await
                .unwrap();
        if !exists {
            return -1;
        }
        sqlx::query_scalar("SELECT count(*) FROM ext_ducklake_file_order")
            .fetch_one(&self.pool)
            .await
            .unwrap()
    }
}

async fn pool_for(port: u16, user: &str, password: &str) -> PgPool {
    PgPoolOptions::new()
        .max_connections(5)
        .connect(&format!(
            "postgresql://{user}:{password}@127.0.0.1:{port}/postgres"
        ))
        .await
        .unwrap()
}

async fn plan(ctx: &SessionContext, sql: &str) -> String {
    let plan = ctx
        .sql(sql)
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    displayable(plan.as_ref()).indent(true).to_string()
}

/// `sql` over the DuckLake table vs over an in-memory copy of its rows.
async fn assert_matches_reference(ctx: &SessionContext, sql: &str) {
    let all = ctx
        .sql("SELECT ts, id FROM cat.public.logs")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let reference = SessionContext::new();
    reference
        .register_table(
            "logs",
            Arc::new(MemTable::try_new(all[0].schema(), vec![all]).unwrap()),
        )
        .unwrap();
    let got = ctx.sql(sql).await.unwrap().collect().await.unwrap();
    let want = reference
        .sql(&sql.replace("cat.public.logs", "logs"))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(
        arrow::util::pretty::pretty_format_batches(&got)
            .unwrap()
            .to_string(),
        arrow::util::pretty::pretty_format_batches(&want)
            .unwrap()
            .to_string(),
        "{sql}"
    );
}

async fn table(ctx: &SessionContext) -> DuckLakeTable {
    let provider = ctx
        .catalog("cat")
        .unwrap()
        .schema("public")
        .unwrap()
        .table("logs")
        .await
        .unwrap()
        .unwrap();
    (provider.as_ref() as &dyn std::any::Any)
        .downcast_ref::<DuckLakeTable>()
        .unwrap()
        .clone()
}

/// The writer records sorted files, not unsorted ones; the verify job records
/// a file written with the option off; the scan declares the order only over
/// recorded files; prune keeps rows of listed files.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn multicatalog_records_verifies_and_declares() {
    let env = Env::new().await;
    // The table-creating file predates the sort spec: the verify job records
    // it without rewriting it.
    let ctx = env.ctx(4).await;
    let verified = table(&ctx)
        .await
        .verify_file_orders(&ctx.state(), VerifyOrderOptions::default())
        .await
        .unwrap();
    assert_eq!(
        (verified.files_verified, verified.files_recorded),
        (1, 1),
        "{verified:?}"
    );
    for a in 0..3i64 {
        env.append(
            &env.pool,
            batch((a * 1_000 + 1..(a + 1) * 1_000).rev()),
            true,
        )
        .await;
    }
    assert_eq!(env.sidecar_rows().await, 4);
    for partitions in [1, 4] {
        let ctx = env.ctx(partitions).await;
        let p = plan(&ctx, TOP_N).await;
        assert!(
            !p.contains("SortExec") && p.contains("output_ordering=[ts@0 DESC]"),
            "{p}"
        );
        assert_matches_reference(&ctx, TOP_N).await;
    }

    // Ascending under a DESC spec: not recorded, and it holds the newest
    // rows, so nothing is declared.
    env.append(&env.pool, batch(3_000..4_000), true).await;
    assert_eq!(env.sidecar_rows().await, 4);
    for partitions in [1, 4] {
        let ctx = env.ctx(partitions).await;
        assert!(plan(&ctx, TOP_N).await.contains("SortExec"));
        assert_matches_reference(&ctx, TOP_N).await;
    }

    // Written with the option off, then recorded by the verify job, which
    // skips the recorded files and leaves the ascending one unrecorded.
    env.append(&env.pool, batch((4_000..5_000).rev()), false)
        .await;
    let ctx = env.ctx(4).await;
    let verified = table(&ctx)
        .await
        .verify_file_orders(&ctx.state(), VerifyOrderOptions::default())
        .await
        .unwrap();
    assert_eq!(
        (
            verified.files_verified,
            verified.files_recorded,
            verified.files_unsorted,
            verified.files_skipped
        ),
        (2, 1, 1, 4),
        "{verified:?}"
    );
    assert_eq!(env.sidecar_rows().await, 5);
    for partitions in [1, 4] {
        assert_matches_reference(&env.ctx(partitions).await, TOP_N).await;
    }
    // Every row's file is listed: nothing to prune.
    assert_eq!(env.writer(&env.pool).await.prune_file_orders().unwrap(), 0);
}

/// The sidecar never fails a data write: a writer that may not write it (here:
/// an administrator's table the writer role holds no privilege on) commits its
/// files unrecorded, and records once it may.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn sidecar_it_cannot_write_never_fails_a_write() {
    let env = Env::new().await;
    // An administrator records first, so the sidecar exists and is theirs.
    env.append(&env.pool, batch((1..1_000).rev()), true).await;
    assert_eq!(env.sidecar_rows().await, 1);
    for sql in [
        "CREATE ROLE w LOGIN PASSWORD 'w'",
        "GRANT USAGE, CREATE ON SCHEMA public TO w",
        "GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO w",
        "GRANT USAGE, SELECT, UPDATE ON ALL SEQUENCES IN SCHEMA public TO w",
        "REVOKE ALL ON ext_ducklake_file_order FROM w",
    ] {
        sqlx::query(sql).execute(&env.pool).await.unwrap();
    }
    let w = pool_for(env.port, "w", "w").await;

    env.append(&w, batch((1_000..2_000).rev()), true).await;
    assert_eq!(env.sidecar_rows().await, 1);
    for partitions in [1, 4] {
        assert_matches_reference(&env.ctx(partitions).await, TOP_N).await;
    }

    sqlx::query("GRANT SELECT, INSERT, UPDATE ON ext_ducklake_file_order TO w")
        .execute(&env.pool)
        .await
        .unwrap();
    env.append(&w, batch((2_000..3_000).rev()), true).await;
    assert_eq!(env.sidecar_rows().await, 2);
    for partitions in [1, 4] {
        assert_matches_reference(&env.ctx(partitions).await, TOP_N).await;
    }
}
