#![cfg(all(feature = "write-sqlite", feature = "metadata-duckdb"))]
//! A consistent read (`MetadataProvider::begin_consistent_read`) on each
//! backend: its reads see one state of the catalog while a writer commits, and
//! they run on the read's own connection.
//!
//! The providers of the sqlx backends read through a pool of one connection,
//! which the read holds, and which gives up on a wait after two seconds: a read
//! that went to the pool instead would fail.

use std::ops::Range;
use std::sync::Arc;
use std::time::Duration;

use arrow::array::{ArrayRef, Int64Array};
use arrow::record_batch::RecordBatch;
use datafusion_ducklake::metadata_provider::{FILE_METADATA_BATCH_SIZE, InlinedRowSelection};
use datafusion_ducklake::{
    DuckLakeTableWriter, DuckLakeWriteOptions, MetadataProvider, MetadataWriter,
    SqliteMetadataProvider, SqliteMetadataWriter,
};
use object_store::local::LocalFileSystem;
use tempfile::TempDir;

/// How long a test's one-connection pool waits for its connection.
const POOL_WAIT: Duration = Duration::from_secs(2);

/// The rows `ids` of table `t`.
fn batch(ids: Range<i64>) -> RecordBatch {
    RecordBatch::try_from_iter(vec![(
        "id",
        Arc::new(Int64Array::from_iter_values(ids)) as ArrayRef,
    )])
    .unwrap()
}

/// What a read of table `t` sees: its files and its inlined rows, at every
/// snapshot the catalog will ever have.
fn seen(provider: &dyn MetadataProvider, table_id: i64) -> (usize, usize) {
    let head = provider.get_current_snapshot().unwrap();
    let columns = provider.get_table_structure(table_id, head).unwrap();
    let files = provider
        .get_table_file_metadata_page(table_id, i64::MAX, None, FILE_METADATA_BATCH_SIZE)
        .unwrap()
        .len();
    let inlined = provider
        .read_inlined_rows(
            table_id,
            &columns,
            InlinedRowSelection::VisibleAt(i64::MAX),
            None,
        )
        .unwrap()
        .iter()
        .map(|rows| rows.batch.num_rows())
        .sum();
    (files, inlined)
}

/// Write table `t` of `schema` through `writer`: a file and two inlined rows.
/// Then begin a consistent read of `provider`, and commit another file and
/// two more inlined rows while it is open. The read sees neither, nor a new
/// head, and every read a scan, a mutation or a change feed makes answers on
/// it. Once it ends, the provider sees both commits.
async fn a_read_sees_one_state(
    writer: Arc<dyn MetadataWriter>,
    provider: Arc<dyn MetadataProvider>,
    data_path: &std::path::Path,
    schema: &str,
) {
    writer.set_data_path(data_path.to_str().unwrap()).unwrap();
    let options = DuckLakeWriteOptions::default().with_data_inlining_row_limit(10);
    let tables = DuckLakeTableWriter::new(Arc::clone(&writer), Arc::new(LocalFileSystem::new()))
        .unwrap()
        .with_options(&options);
    tables
        .write_table(schema, "t", &[batch(0..100)])
        .await
        .unwrap();
    tables
        .append_table(schema, "t", &[batch(100..102)])
        .await
        .unwrap();

    let read = provider
        .begin_consistent_read()
        .unwrap()
        .expect("the provider begins a consistent read");
    let bound = Arc::clone(read.provider());
    let head = bound.get_current_snapshot().unwrap();
    let schema_id = bound
        .get_schema_by_name(schema, head)
        .unwrap()
        .expect("the schema")
        .schema_id;
    let table_id = bound
        .get_table_by_name(schema_id, "t", head)
        .unwrap()
        .expect("table t")
        .table_id;
    // A backend whose writer does not inline writes the rows to a file.
    let before = seen(bound.as_ref(), table_id);
    assert_eq!(before.0 + before.1 / 2, 2, "{before:?}");

    tables
        .append_table(schema, "t", &[batch(200..300)])
        .await
        .unwrap();
    tables
        .append_table(schema, "t", &[batch(300..302)])
        .await
        .unwrap();
    // Read through the writer: the provider's one connection is the read's.
    assert!(writer.head_snapshot().unwrap() > Some(head));

    // Inside the read, nothing committed since it began.
    assert_eq!(bound.get_current_snapshot().unwrap(), head);
    assert_eq!(seen(bound.as_ref(), table_id), before);
    let columns = bound.get_table_structure(table_id, head).unwrap();
    bound.get_inlined_deletes(table_id, head).unwrap();
    bound
        .scan_inlined_data_noting_absence(table_id, head, &columns, None)
        .unwrap();
    assert_eq!(
        bound
            .get_table_file_metadata_page_filtered(
                table_id,
                i64::MAX,
                None,
                FILE_METADATA_BATCH_SIZE,
                None
            )
            .unwrap()
            .len(),
        before.0
    );
    assert_eq!(
        bound
            .get_table_files_for_select(table_id, i64::MAX)
            .unwrap()
            .len(),
        before.0
    );
    bound
        .get_inlined_data_with_row_ids(table_id, head, &columns)
        .unwrap();
    assert_eq!(
        bound
            .get_data_files_added_between_snapshots(table_id, 0, i64::MAX)
            .unwrap()
            .len(),
        before.0
    );
    bound
        .get_delete_files_added_between_snapshots(table_id, 0, i64::MAX)
        .unwrap();
    bound
        .get_inlined_file_deletes_between(table_id, 0, i64::MAX)
        .unwrap();
    bound
        .read_inlined_rows(
            table_id,
            &columns,
            InlinedRowSelection::InsertedBetween {
                start: 0,
                end: i64::MAX,
            },
            None,
        )
        .unwrap();
    bound.get_inlined_data_for_flush(table_id, head).unwrap();
    bound
        .get_inlined_file_deletes_for_flush(table_id, head)
        .unwrap();

    // A read begun inside it shares it, and ends nothing.
    let nested = bound
        .begin_consistent_read()
        .unwrap()
        .expect("a bound provider begins a read within its own");
    assert_eq!(nested.provider().get_current_snapshot().unwrap(), head);
    nested.end().unwrap();
    assert_eq!(seen(bound.as_ref(), table_id), before);

    read.end().unwrap();
    assert!(provider.get_current_snapshot().unwrap() > head);
    let after = seen(provider.as_ref(), table_id);
    assert_eq!(
        (after.0 - before.0) + (after.1 - before.1) / 2,
        2,
        "{before:?} then {after:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_sqlite_read_sees_one_state() {
    use sqlx::sqlite::SqlitePoolOptions;

    let temp = TempDir::new().unwrap();
    let path = temp.path().join("catalog.db");
    let url = format!("sqlite:{}?mode=rwc", path.display());
    let writer = SqliteMetadataWriter::new_with_init(&url).await.unwrap();
    // In rollback-journal mode the writer's commits would wait for the read.
    crate::catalog_read_memo_tests::use_write_ahead_log(&path).unwrap();
    let pool = SqlitePoolOptions::new()
        .max_connections(1)
        .acquire_timeout(POOL_WAIT)
        .connect(&url)
        .await
        .unwrap();
    let provider = SqliteMetadataProvider::from_pool(pool);
    a_read_sees_one_state(
        Arc::new(writer),
        Arc::new(provider),
        &temp.path().join("data"),
        "main",
    )
    .await;
}

/// In rollback-journal mode, a commit cannot land while a read holds the
/// catalog: the writer waits for the read to end, and the read keeps seeing
/// one state.
#[tokio::test(flavor = "multi_thread")]
async fn a_sqlite_commit_waits_for_a_read_in_rollback_journal_mode() {
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("catalog.db");
    let url = format!("sqlite:{}?mode=rwc", path.display());
    let writer = Arc::new(SqliteMetadataWriter::new_with_init(&url).await.unwrap());
    let data_path = temp.path().join("data");
    writer.set_data_path(data_path.to_str().unwrap()).unwrap();
    let tables = Arc::new(
        DuckLakeTableWriter::new(
            Arc::clone(&writer) as Arc<dyn MetadataWriter>,
            Arc::new(LocalFileSystem::new()),
        )
        .unwrap(),
    );
    tables
        .write_table("main", "t", &[batch(0..100)])
        .await
        .unwrap();
    let provider = SqliteMetadataProvider::new(&url).await.unwrap();
    let head = provider.get_current_snapshot().unwrap();

    let read = provider
        .begin_consistent_read()
        .unwrap()
        .expect("a consistent read");
    // The read takes its shared lock at its first statement.
    assert_eq!(read.provider().get_current_snapshot().unwrap(), head);
    let commit = tokio::spawn({
        let tables = Arc::clone(&tables);
        async move { tables.append_table("main", "t", &[batch(100..200)]).await }
    });
    tokio::time::sleep(Duration::from_millis(500)).await;
    assert!(!commit.is_finished(), "the commit waits for the read");
    assert_eq!(read.provider().get_current_snapshot().unwrap(), head);
    read.end().unwrap();
    commit.await.unwrap().unwrap();
    assert!(provider.get_current_snapshot().unwrap() > head);
}

#[cfg(feature = "write-duckdb")]
#[tokio::test(flavor = "multi_thread")]
async fn a_duckdb_read_sees_one_state() {
    use datafusion_ducklake::DuckdbMetadataWriter;

    let temp = TempDir::new().unwrap();
    let path = temp.path().join("catalog.ducklake");
    let writer = DuckdbMetadataWriter::new_with_init(path.to_str().unwrap()).unwrap();
    // The provider shares the writer's connection; the read takes one of its
    // own.
    let provider = writer.metadata_provider();
    a_read_sees_one_state(
        Arc::new(writer),
        Arc::new(provider),
        &temp.path().join("data"),
        "main",
    )
    .await;
}

#[cfg(feature = "write-postgres")]
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn a_postgres_read_sees_one_state() {
    use datafusion_ducklake::{PostgresMetadataProvider, PostgresSingleCatalogMetadataWriter};
    use sqlx::postgres::PgPoolOptions;
    use testcontainers::runners::AsyncRunner;
    use testcontainers_modules::postgres::Postgres;

    let container = Postgres::default().start().await.unwrap();
    let port = container.get_host_port_ipv4(5432).await.unwrap();
    let url = format!("postgresql://postgres:postgres@127.0.0.1:{port}/postgres");
    let writer = PostgresSingleCatalogMetadataWriter::new_with_init(&url)
        .await
        .unwrap();
    let pool = PgPoolOptions::new()
        .max_connections(1)
        .acquire_timeout(POOL_WAIT)
        .connect(&url)
        .await
        .unwrap();
    let provider = PostgresMetadataProvider::from_pool(pool);
    let temp = TempDir::new().unwrap();
    a_read_sees_one_state(
        Arc::new(writer),
        Arc::new(provider),
        &temp.path().join("data"),
        "main",
    )
    .await;
}

#[cfg(feature = "write-mysql")]
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn a_mysql_read_sees_one_state() {
    use datafusion_ducklake::{MySqlMetadataProvider, MySqlMetadataWriter};
    use sqlx::mysql::MySqlPoolOptions;
    use testcontainers::runners::AsyncRunner;
    use testcontainers_modules::mysql::Mysql;

    let container = Mysql::default().start().await.unwrap();
    let port = container.get_host_port_ipv4(3306).await.unwrap();
    let url = format!("mysql://root@127.0.0.1:{port}/test");
    let writer = MySqlMetadataWriter::new_with_init(&url).await.unwrap();
    let pool = MySqlPoolOptions::new()
        .max_connections(1)
        .acquire_timeout(POOL_WAIT)
        .connect(&url)
        .await
        .unwrap();
    let provider = MySqlMetadataProvider::from_pool(pool);
    let temp = TempDir::new().unwrap();
    a_read_sees_one_state(
        Arc::new(writer),
        Arc::new(provider),
        &temp.path().join("data"),
        "main",
    )
    .await;
}

#[cfg(feature = "write-postgres")]
mod multicatalog {
    use datafusion_ducklake::{MulticatalogManager, MulticatalogProvider, PostgresMetadataWriter};
    use sqlx::PgPool;
    use sqlx::postgres::PgPoolOptions;
    use testcontainers::ContainerAsync;
    use testcontainers::runners::AsyncRunner;
    use testcontainers_modules::postgres::Postgres;

    use super::*;

    /// A multicatalog store on a new server, one catalog in it, a writer of
    /// that catalog, and a provider of it over a pool of `connections`.
    async fn store(
        connections: u32,
    ) -> (
        ContainerAsync<Postgres>,
        Arc<dyn MetadataWriter>,
        MulticatalogProvider,
    ) {
        let container = Postgres::default().start().await.unwrap();
        let port = container.get_host_port_ipv4(5432).await.unwrap();
        let url = format!("postgresql://postgres:postgres@127.0.0.1:{port}/postgres");
        let pool = PgPool::connect(&url).await.unwrap();
        datafusion_ducklake::initialize_multicatalog_schema(&pool)
            .await
            .unwrap();
        let catalog_id = MulticatalogManager::new(pool.clone())
            .create_catalog("lake")
            .await
            .unwrap();
        let writer = PostgresMetadataWriter::with_pool(pool, catalog_id)
            .await
            .unwrap();
        let small = PgPoolOptions::new()
            .max_connections(connections)
            .acquire_timeout(POOL_WAIT)
            .connect(&url)
            .await
            .unwrap();
        let provider = MulticatalogProvider::with_pool_and_id(small, catalog_id)
            .await
            .unwrap();
        (container, Arc::new(writer), provider)
    }

    #[tokio::test(flavor = "multi_thread")]
    #[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
    async fn a_multicatalog_read_sees_one_state() {
        let (_container, writer, provider) = store(1).await;
        let temp = TempDir::new().unwrap();
        a_read_sees_one_state(
            writer,
            Arc::new(provider),
            &temp.path().join("data"),
            "public",
        )
        .await;
    }

    /// A provider bound to a read session reads on the session's connection.
    /// The reads open on it at once share one transaction: the first begins it
    /// and the last to end commits it. So a read begun while another is open
    /// sees that one's state, and a read begun after both ended sees the
    /// commits since.
    #[tokio::test(flavor = "multi_thread")]
    #[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
    async fn reads_on_a_read_session_share_one_transaction() {
        let (_container, writer, provider) = store(1).await;
        let temp = TempDir::new().unwrap();
        let data_path = temp.path().join("data");
        writer.set_data_path(data_path.to_str().unwrap()).unwrap();
        let tables =
            DuckLakeTableWriter::new(Arc::clone(&writer), Arc::new(LocalFileSystem::new()))
                .unwrap();
        tables
            .write_table("public", "t", &[batch(0..100)])
            .await
            .unwrap();

        // The session holds the pool's only connection.
        let session = provider.begin_read_session().await.unwrap();
        let bound = provider.with_read_session(&session).unwrap();
        let head = bound.get_current_snapshot().unwrap();

        let first = bound.begin_consistent_read().unwrap().expect("a read");
        assert_eq!(first.provider().get_current_snapshot().unwrap(), head);
        tables
            .append_table("public", "t", &[batch(100..200)])
            .await
            .unwrap();
        let second = bound.begin_consistent_read().unwrap().expect("a read");
        assert_eq!(second.provider().get_current_snapshot().unwrap(), head);
        first.end().unwrap();
        // The second read is still open, and so is the transaction.
        assert_eq!(second.provider().get_current_snapshot().unwrap(), head);
        assert_eq!(bound.get_current_snapshot().unwrap(), head);
        second.end().unwrap();
        let moved = bound.get_current_snapshot().unwrap();
        assert!(moved > head);

        let third = bound.begin_consistent_read().unwrap().expect("a read");
        assert_eq!(third.provider().get_current_snapshot().unwrap(), moved);
        // Ending the session commits the read open on it.
        session.end().await.unwrap();
        third.end().unwrap();
        assert_eq!(provider.get_current_snapshot().unwrap(), moved);
    }
}
