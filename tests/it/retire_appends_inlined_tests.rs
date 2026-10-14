//! `MetadataWriter::retire_appends_since` with data inlining: an append whose
//! rows were inlined is retired like a Parquet append, and a delta holding
//! deletions it cannot revert — inlined deletions, or a delete file rewritten
//! with the deletions dated before the base — or a live file holding rows of
//! both sides of the base is refused.

#![cfg(feature = "write")]
#![allow(dead_code)]

use std::sync::Arc;

use arrow::array::{Int32Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::prelude::SessionContext;
use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeError, DuckLakeTableWriter, MetadataProvider, MetadataWriter,
    execute_ducklake_sql,
};
use object_store::local::LocalFileSystem;

struct Backend {
    writer: Arc<dyn MetadataWriter>,
    provider: Arc<dyn MetadataProvider>,
}

fn rows(from: i32, to: i32) -> RecordBatch {
    let ids = (from..to).collect::<Vec<_>>();
    let names = ids.iter().map(|id| format!("x{id}")).collect::<Vec<_>>();
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, true),
            Field::new("name", DataType::Utf8, true),
        ])),
        vec![Arc::new(Int32Array::from(ids)), Arc::new(StringArray::from(names))],
    )
    .unwrap()
}

async fn sql(backend: &Backend, statement: &str) {
    let ctx = SessionContext::new();
    let catalog =
        DuckLakeCatalog::with_writer(Arc::clone(&backend.provider), Arc::clone(&backend.writer))
            .unwrap();
    DuckLakeCatalog::register(&ctx, "lake", catalog);
    let handle =
        DuckLakeCatalog::with_writer(Arc::clone(&backend.provider), Arc::clone(&backend.writer))
            .unwrap();
    execute_ducklake_sql(&ctx, &handle, statement)
        .await
        .unwrap_or_else(|e| panic!("`{statement}`: {e}"))
        .collect()
        .await
        .unwrap_or_else(|e| panic!("`{statement}`: {e}"));
}

/// The ids `main.t` holds at `snapshot`.
async fn ids_at(backend: &Backend, snapshot: i64) -> Vec<i32> {
    let ctx = SessionContext::new();
    let catalog = DuckLakeCatalog::with_snapshot(Arc::clone(&backend.provider), snapshot).unwrap();
    ctx.register_catalog("tt", Arc::new(catalog));
    let batches = ctx
        .sql("SELECT id FROM tt.main.t ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .values()
                .to_vec()
        })
        .collect()
}

fn table_id(backend: &Backend) -> i64 {
    let snapshot = backend.provider.get_current_snapshot().unwrap();
    let schema = backend
        .provider
        .get_schema_by_name("main", snapshot)
        .unwrap()
        .unwrap();
    backend
        .provider
        .get_table_by_name(schema.schema_id, "t", snapshot)
        .unwrap()
        .unwrap()
        .table_id
}

/// `main.t` with ids `0..40` in one Parquet file; returns the table id.
async fn setup(backend: &Backend) -> i64 {
    DuckLakeTableWriter::new(
        Arc::clone(&backend.writer),
        Arc::new(LocalFileSystem::new()),
    )
    .unwrap()
    .append_table("main", "t", &[rows(0, 40)])
    .await
    .unwrap();
    table_id(backend)
}

/// An `INSERT` inlined under the default limit is retired: its rows end at
/// the retiring snapshot and stay readable before it.
async fn retires_an_inlined_append(backend: &Backend) {
    let table = setup(backend).await;
    let base = backend.provider.get_current_snapshot().unwrap();
    sql(
        backend,
        "INSERT INTO lake.main.t VALUES (100, 'a'), (101, 'b')",
    )
    .await;
    let appended = backend.provider.get_current_snapshot().unwrap();
    let retired = backend
        .writer
        .retire_appends_since(table, base)
        .unwrap()
        .expect("the inlined append is retired");
    assert_eq!(ids_at(backend, retired).await, (0..40).collect::<Vec<_>>());
    let mut before = (0..40).collect::<Vec<_>>();
    before.extend([100, 101]);
    assert_eq!(ids_at(backend, appended).await, before);
    // A retry appends exactly once.
    sql(
        backend,
        "INSERT INTO lake.main.t VALUES (100, 'a'), (101, 'b')",
    )
    .await;
    let head = backend.provider.get_current_snapshot().unwrap();
    assert_eq!(ids_at(backend, head).await, before);
}

/// An inlined deletion of a Parquet row after the base cannot be reverted by
/// retiring files, so the retirement is refused and changes nothing.
async fn refuses_an_inlined_file_deletion(backend: &Backend) {
    let table = setup(backend).await;
    let base = backend.provider.get_current_snapshot().unwrap();
    sql(backend, "DELETE FROM lake.main.t WHERE id = 3").await;
    assert_refused(backend, table, base).await;
}

/// Deleting a row inlined before the base is refused the same way.
async fn refuses_an_inlined_row_deletion(backend: &Backend) {
    let table = setup(backend).await;
    sql(backend, "INSERT INTO lake.main.t VALUES (100, 'a')").await;
    let base = backend.provider.get_current_snapshot().unwrap();
    sql(backend, "DELETE FROM lake.main.t WHERE id = 100").await;
    assert_refused(backend, table, base).await;
}

/// A `DELETE` that rewrites a delete file dates the file at its earliest
/// deletion, before the base; it is still a post-base deletion and refused.
async fn refuses_a_rewritten_delete_file(backend: &Backend) {
    let table = setup(backend).await;
    sql(backend, "DELETE FROM lake.main.t WHERE id < 12").await;
    let base = backend.provider.get_current_snapshot().unwrap();
    sql(
        backend,
        "DELETE FROM lake.main.t WHERE id BETWEEN 12 AND 24",
    )
    .await;
    assert_refused(backend, table, base).await;
}

/// A flush writes inlined rows from both sides of the base into one file that
/// begins before the base: that live file holds rows retiring cannot keep, so
/// the retirement is refused.
async fn refuses_a_file_flushed_across_the_base(backend: &Backend) {
    let table = setup(backend).await;
    sql(backend, "INSERT INTO lake.main.t VALUES (100, 'a')").await;
    let base = backend.provider.get_current_snapshot().unwrap();
    sql(backend, "INSERT INTO lake.main.t VALUES (101, 'b')").await;
    sql(backend, "CALL ducklake_flush_inlined_data('lake')").await;
    assert_refused(backend, table, base).await;
}

/// A catalog whose `ducklake_delete_file` predates `partial_max` (written by
/// v0.5.0 and earlier), opened through its schema initialization: a flush moving an
/// inlined deletion into a delete file, a `DELETE` rewriting that delete file,
/// an `UPDATE`, and `retire_appends_since` all work.
async fn writes_to_a_catalog_without_delete_file_partial_max(backend: &Backend) {
    let table = setup(backend).await;
    sql(backend, "DELETE FROM lake.main.t WHERE id = 3").await;
    sql(backend, "CALL ducklake_flush_inlined_data('lake')").await;
    backend
        .writer
        .set_table_setting(table, "data_inlining_row_limit", "0")
        .unwrap();
    sql(backend, "DELETE FROM lake.main.t WHERE id = 5").await;
    sql(backend, "UPDATE lake.main.t SET name = 'u' WHERE id = 7").await;
    let base = backend.provider.get_current_snapshot().unwrap();
    sql(backend, "INSERT INTO lake.main.t VALUES (100, 'a')").await;
    let retired = backend
        .writer
        .retire_appends_since(table, base)
        .unwrap()
        .expect("the append is retired");
    let expected: Vec<i32> = (0..40).filter(|id| *id != 3 && *id != 5).collect();
    assert_eq!(ids_at(backend, retired).await, expected);
    assert_eq!(ids_at(backend, base).await, expected);
}

async fn assert_refused(backend: &Backend, table: i64, base: i64) {
    let head = backend.provider.get_current_snapshot().unwrap();
    let before = ids_at(backend, head).await;
    let error = backend
        .writer
        .retire_appends_since(table, base)
        .unwrap_err();
    assert!(matches!(error, DuckLakeError::Conflict(_)), "{error}");
    assert_eq!(backend.provider.get_current_snapshot().unwrap(), head);
    assert_eq!(ids_at(backend, head).await, before);
}

#[cfg(feature = "write-sqlite")]
mod sqlite {
    use super::*;
    use datafusion_ducklake::{SqliteMetadataProvider, SqliteMetadataWriter};
    use tempfile::TempDir;

    async fn backend(dir: &TempDir) -> Backend {
        let url = format!(
            "sqlite:{}?mode=rwc",
            dir.path().join("catalog.db").display()
        );
        let data = dir.path().join("data");
        std::fs::create_dir_all(&data).unwrap();
        let writer = SqliteMetadataWriter::new_with_init(&url).await.unwrap();
        writer.set_data_path(data.to_str().unwrap()).unwrap();
        Backend {
            writer: Arc::new(writer),
            provider: Arc::new(SqliteMetadataProvider::new(&url).await.unwrap()),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn inlined_append() {
        let dir = TempDir::new().unwrap();
        retires_an_inlined_append(&backend(&dir).await).await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn inlined_file_deletion() {
        let dir = TempDir::new().unwrap();
        refuses_an_inlined_file_deletion(&backend(&dir).await).await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn inlined_row_deletion() {
        let dir = TempDir::new().unwrap();
        refuses_an_inlined_row_deletion(&backend(&dir).await).await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn rewritten_delete_file() {
        let dir = TempDir::new().unwrap();
        refuses_a_rewritten_delete_file(&backend(&dir).await).await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn file_flushed_across_the_base() {
        let dir = TempDir::new().unwrap();
        refuses_a_file_flushed_across_the_base(&backend(&dir).await).await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn delete_file_without_partial_max() {
        let dir = TempDir::new().unwrap();
        drop(backend(&dir).await);
        let url = format!("sqlite:{}", dir.path().join("catalog.db").display());
        let pool = sqlx::SqlitePool::connect(&url).await.unwrap();
        sqlx::query("ALTER TABLE ducklake_delete_file DROP COLUMN partial_max")
            .execute(&pool)
            .await
            .unwrap();
        pool.close().await;
        writes_to_a_catalog_without_delete_file_partial_max(&backend(&dir).await).await;
    }
}

#[cfg(feature = "write-duckdb")]
mod duckdb_backend {
    use super::*;
    use datafusion_ducklake::DuckdbMetadataWriter;
    use tempfile::TempDir;

    fn backend(dir: &TempDir) -> Backend {
        let path = dir.path().join("catalog.duckdb");
        let data = dir.path().join("data");
        std::fs::create_dir_all(&data).unwrap();
        let writer =
            DuckdbMetadataWriter::new_with_init(path.to_string_lossy().into_owned()).unwrap();
        writer.set_data_path(data.to_str().unwrap()).unwrap();
        let provider = writer.metadata_provider();
        Backend {
            writer: Arc::new(writer),
            provider: Arc::new(provider),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn inlined_append() {
        let dir = TempDir::new().unwrap();
        retires_an_inlined_append(&backend(&dir)).await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn inlined_file_deletion() {
        let dir = TempDir::new().unwrap();
        refuses_an_inlined_file_deletion(&backend(&dir)).await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn inlined_row_deletion() {
        let dir = TempDir::new().unwrap();
        refuses_an_inlined_row_deletion(&backend(&dir)).await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn rewritten_delete_file() {
        let dir = TempDir::new().unwrap();
        refuses_a_rewritten_delete_file(&backend(&dir)).await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn file_flushed_across_the_base() {
        let dir = TempDir::new().unwrap();
        refuses_a_file_flushed_across_the_base(&backend(&dir)).await;
    }
}

#[cfg(feature = "write-mysql")]
mod mysql_backend {
    use super::*;
    use datafusion_ducklake::{MySqlMetadataProvider, MySqlMetadataWriter};
    use tempfile::TempDir;
    use testcontainers::runners::AsyncRunner;
    use testcontainers_modules::mysql::Mysql;

    #[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
    #[tokio::test(flavor = "multi_thread")]
    async fn scenarios() {
        let container = Mysql::default().start().await.unwrap();
        let port = container.get_host_port_ipv4(3306).await.unwrap();
        let admin = format!("mysql://root@127.0.0.1:{port}/test");
        let pool = sqlx::MySqlPool::connect(&admin).await.unwrap();
        for (index, scenario) in
            ["append", "file deletion", "row deletion", "rewritten delete file", "flush"]
                .into_iter()
                .enumerate()
        {
            let database = format!("retire_{index}");
            sqlx::query(sqlx::AssertSqlSafe(format!("CREATE DATABASE {database}")))
                .execute(&pool)
                .await
                .unwrap();
            let url = format!("mysql://root@127.0.0.1:{port}/{database}");
            let dir = TempDir::new().unwrap();
            let data = dir.path().join("data");
            std::fs::create_dir_all(&data).unwrap();
            let writer = MySqlMetadataWriter::new_with_init(&url).await.unwrap();
            writer.set_data_path(data.to_str().unwrap()).unwrap();
            let backend = Backend {
                writer: Arc::new(writer),
                provider: Arc::new(MySqlMetadataProvider::new(&url).await.unwrap()),
            };
            match scenario {
                "append" => retires_an_inlined_append(&backend).await,
                "file deletion" => refuses_an_inlined_file_deletion(&backend).await,
                "row deletion" => refuses_an_inlined_row_deletion(&backend).await,
                "flush" => refuses_a_file_flushed_across_the_base(&backend).await,
                _ => refuses_a_rewritten_delete_file(&backend).await,
            }
        }
    }
}

#[cfg(feature = "multicatalog-postgres")]
mod multicatalog_backend {
    use super::*;
    use datafusion_ducklake::{MulticatalogManager, MulticatalogProvider, PostgresMetadataWriter};
    use sqlx::postgres::PgPoolOptions;
    use tempfile::TempDir;
    use testcontainers::runners::AsyncRunner;
    use testcontainers_modules::postgres::Postgres;

    #[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
    #[tokio::test(flavor = "multi_thread")]
    async fn scenarios() {
        let container = Postgres::default().start().await.unwrap();
        let port = container.get_host_port_ipv4(5432).await.unwrap();
        let url = format!("postgresql://postgres:postgres@127.0.0.1:{port}/postgres");
        let pool = PgPoolOptions::new()
            .max_connections(5)
            .connect(&url)
            .await
            .unwrap();
        datafusion_ducklake::initialize_multicatalog_schema(&pool)
            .await
            .unwrap();
        for (index, scenario) in
            ["append", "file deletion", "row deletion", "rewritten delete file", "flush"]
                .into_iter()
                .enumerate()
        {
            let name = format!("lake{index}");
            let catalog = MulticatalogManager::new(pool.clone())
                .create_catalog(&name)
                .await
                .unwrap();
            let dir = TempDir::new().unwrap();
            let data = dir.path().join("data");
            std::fs::create_dir_all(&data).unwrap();
            let writer = PostgresMetadataWriter::with_pool(pool.clone(), catalog)
                .await
                .unwrap();
            writer.set_data_path(data.to_str().unwrap()).unwrap();
            let backend = Backend {
                writer: Arc::new(writer),
                provider: Arc::new(
                    MulticatalogProvider::with_pool(pool.clone(), &name)
                        .await
                        .unwrap(),
                ),
            };
            match scenario {
                "append" => retires_an_inlined_append(&backend).await,
                "file deletion" => refuses_an_inlined_file_deletion(&backend).await,
                "row deletion" => refuses_an_inlined_row_deletion(&backend).await,
                "flush" => refuses_a_file_flushed_across_the_base(&backend).await,
                _ => refuses_a_rewritten_delete_file(&backend).await,
            }
        }

        // A store from v0.6.0 and earlier: no `ducklake_delete_file.partial_max`
        // (v0.5.0 and earlier) and no `ducklake_metadata.scope_id`.
        sqlx::query("CREATE DATABASE legacy")
            .execute(&pool)
            .await
            .unwrap();
        let legacy_url = format!("postgresql://postgres:postgres@127.0.0.1:{port}/legacy");
        let pool = PgPoolOptions::new()
            .max_connections(5)
            .connect(&legacy_url)
            .await
            .unwrap();
        datafusion_ducklake::initialize_multicatalog_schema(&pool)
            .await
            .unwrap();
        for statement in [
            "ALTER TABLE ducklake_delete_file DROP COLUMN partial_max",
            "ALTER TABLE ducklake_metadata DROP COLUMN scope_id",
        ] {
            sqlx::query(statement).execute(&pool).await.unwrap();
        }
        datafusion_ducklake::initialize_multicatalog_schema(&pool)
            .await
            .unwrap();
        let catalog = MulticatalogManager::new(pool.clone())
            .create_catalog("legacy")
            .await
            .unwrap();
        let dir = TempDir::new().unwrap();
        let data = dir.path().join("data");
        std::fs::create_dir_all(&data).unwrap();
        let writer = PostgresMetadataWriter::with_pool(pool.clone(), catalog)
            .await
            .unwrap();
        writer.set_data_path(data.to_str().unwrap()).unwrap();
        let backend = Backend {
            writer: Arc::new(writer),
            provider: Arc::new(
                MulticatalogProvider::with_pool(pool.clone(), "legacy")
                    .await
                    .unwrap(),
            ),
        };
        writes_to_a_catalog_without_delete_file_partial_max(&backend).await;
    }
}
