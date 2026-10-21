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

    // The provider of the ended read begins a new read, not one within it.
    a_read_after_its_end_begins_anew(&bound, &tables, schema).await;
}

/// `bound` is the provider of a read that has ended. A read it begins sees one
/// state of the catalog while `tables` commits to table `t` of `schema`.
async fn a_read_after_its_end_begins_anew(
    bound: &Arc<dyn MetadataProvider>,
    tables: &DuckLakeTableWriter,
    schema: &str,
) {
    let again = bound
        .begin_consistent_read()
        .unwrap()
        .expect("a provider of an ended read begins a new one");
    let head = again.provider().get_current_snapshot().unwrap();
    tables
        .append_table(schema, "t", &[batch(400..500)])
        .await
        .unwrap();
    assert_eq!(again.provider().get_current_snapshot().unwrap(), head);
    again.end().unwrap();
    assert!(bound.get_current_snapshot().unwrap() > head);
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

/// `SELECT count(id)` of table `t` of `schema`, through a catalog over
/// `provider` and a scan of the table's files.
async fn count_scanned(provider: impl MetadataProvider + 'static, schema: &str) -> i64 {
    use datafusion::prelude::SessionContext;
    use datafusion_ducklake::DuckLakeCatalog;

    let ctx = SessionContext::new();
    ctx.register_catalog("lake", Arc::new(DuckLakeCatalog::new(provider).unwrap()));
    let batches = ctx
        .sql(&format!(
            "SELECT count(id) FROM lake.{schema}.t WHERE id >= 0"
        ))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("count is Int64")
        .value(0)
}

/// Write table `t` of `schema`, 100 rows in a file, through `writer`. Then
/// query it through a `CatalogReadMemo` over `provider`, whose pool holds one
/// connection: a fill inside a scan's consistent read reads on the read's
/// connection alone, so each query answers before the pool's wait runs out,
/// and the memo keeps the fill. Then four queries at once, each with a memo of
/// its own, over `two`, whose pool holds two connections: each holds at most
/// one, so none waits out the pool.
async fn memo_queries_need_one_connection(
    writer: Arc<dyn MetadataWriter>,
    provider: Arc<dyn MetadataProvider>,
    two: Arc<dyn MetadataProvider>,
    data_path: &std::path::Path,
    schema: &str,
) {
    use datafusion_ducklake::CatalogReadMemo;

    writer.set_data_path(data_path.to_str().unwrap()).unwrap();
    DuckLakeTableWriter::new(Arc::clone(&writer), Arc::new(LocalFileSystem::new()))
        .unwrap()
        .write_table(schema, "t", &[batch(0..100)])
        .await
        .unwrap();

    let memo = CatalogReadMemo::new(CatalogReadMemo::DEFAULT_MAX_BYTES);
    for query in 0..2 {
        let started = std::time::Instant::now();
        let bound = memo.bind(Arc::clone(&provider)).unwrap();
        assert_eq!(count_scanned(bound, schema).await, 100, "query {query}");
        assert!(started.elapsed() < POOL_WAIT, "query {query}");
        let kept = format!("{memo:?}");
        assert!(kept.contains("tables: 1"), "query {query}: {kept}");
    }

    let queries = (0..4).map(|_| {
        let two = Arc::clone(&two);
        let schema = schema.to_string();
        tokio::spawn(async move {
            let memo = CatalogReadMemo::new(CatalogReadMemo::DEFAULT_MAX_BYTES);
            let count = count_scanned(memo.bind(two).unwrap(), &schema).await;
            (count, format!("{memo:?}"))
        })
    });
    for query in queries.collect::<Vec<_>>() {
        let (count, kept) = query.await.expect("the query answers");
        assert_eq!(count, 100);
        assert!(kept.contains("tables: 1"), "{kept}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn sqlite_memo_queries_need_one_connection() {
    use sqlx::sqlite::SqlitePoolOptions;

    let temp = TempDir::new().unwrap();
    let path = temp.path().join("catalog.db");
    let url = format!("sqlite:{}?mode=rwc", path.display());
    let writer = SqliteMetadataWriter::new_with_init(&url).await.unwrap();
    let pool = |connections| {
        SqlitePoolOptions::new()
            .max_connections(connections)
            .acquire_timeout(POOL_WAIT)
            .connect_lazy(&url)
            .unwrap()
    };
    memo_queries_need_one_connection(
        Arc::new(writer),
        Arc::new(SqliteMetadataProvider::from_pool(pool(1))),
        Arc::new(SqliteMetadataProvider::from_pool(pool(2))),
        &temp.path().join("data"),
        "main",
    )
    .await;
}

#[cfg(feature = "write-postgres")]
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn postgres_memo_queries_need_one_connection() {
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
    let pool = |connections| {
        PgPoolOptions::new()
            .max_connections(connections)
            .acquire_timeout(POOL_WAIT)
            .connect_lazy(&url)
            .unwrap()
    };
    let temp = TempDir::new().unwrap();
    memo_queries_need_one_connection(
        Arc::new(writer),
        Arc::new(PostgresMetadataProvider::from_pool(pool(1))),
        Arc::new(PostgresMetadataProvider::from_pool(pool(2))),
        &temp.path().join("data"),
        "main",
    )
    .await;
}

/// The provider of a memo's read that has ended begins a new read of the
/// wrapped provider, not one within the ended read.
#[tokio::test(flavor = "multi_thread")]
async fn a_memo_read_after_its_end_begins_anew() {
    use datafusion_ducklake::CatalogReadMemo;

    let temp = TempDir::new().unwrap();
    let path = temp.path().join("catalog.db");
    let url = format!("sqlite:{}?mode=rwc", path.display());
    let writer: Arc<dyn MetadataWriter> =
        Arc::new(SqliteMetadataWriter::new_with_init(&url).await.unwrap());
    crate::catalog_read_memo_tests::use_write_ahead_log(&path).unwrap();
    writer
        .set_data_path(temp.path().join("data").to_str().unwrap())
        .unwrap();
    let tables =
        DuckLakeTableWriter::new(Arc::clone(&writer), Arc::new(LocalFileSystem::new())).unwrap();
    tables
        .write_table("main", "t", &[batch(0..100)])
        .await
        .unwrap();
    let provider: Arc<dyn MetadataProvider> =
        Arc::new(SqliteMetadataProvider::new(&url).await.unwrap());
    let memo = CatalogReadMemo::new(CatalogReadMemo::DEFAULT_MAX_BYTES);
    let bound = memo.bind(provider).unwrap();

    let read = bound.begin_consistent_read().unwrap().expect("a read");
    let ended = Arc::clone(read.provider());
    ended.get_current_snapshot().unwrap();
    read.end().unwrap();
    a_read_after_its_end_begins_anew(&ended, &tables, "main").await;
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

/// The optional tables of a single-catalog PostgreSQL catalog, which a catalog
/// can lack and every read treats as empty.
#[cfg(feature = "write-postgres")]
const OPTIONAL_TABLES: &[&str] = &[
    "ducklake_partition_info",
    "ducklake_partition_column",
    "ducklake_sort_info",
    "ducklake_sort_expression",
    "ducklake_table_stats",
    "ducklake_table_column_stats",
    "ducklake_file_column_stats",
    "ducklake_file_partition_value",
    "ducklake_view",
];

/// `SELECT count(id)` of table `t` of `schema`, through a catalog over
/// `provider` at its head and a scan of the table's files.
#[cfg(feature = "write-postgres")]
async fn count_at_head(provider: Arc<dyn MetadataProvider>, schema: &str) -> i64 {
    use datafusion::prelude::SessionContext;
    use datafusion_ducklake::DuckLakeCatalog;

    let head = provider.get_current_snapshot().unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog(
        "lake",
        Arc::new(DuckLakeCatalog::with_snapshot(provider, head).unwrap()),
    );
    let batches = ctx
        .sql(&format!(
            "SELECT count(id) FROM lake.{schema}.t WHERE id >= 0"
        ))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("count is Int64")
        .value(0)
}

/// `provider` reads a catalog whose table `t` of `schema` holds 100 rows in
/// one file, and which lacks its optional tables. Inside one consistent read,
/// every read of an optional table reads as empty, and the read's transaction
/// stays usable to its end: a statement that failed would end it, and fail
/// every later read. Then a query scans the table.
#[cfg(feature = "write-postgres")]
async fn reads_without_optional_tables(provider: Arc<dyn MetadataProvider>, schema: &str) {
    use datafusion_ducklake::{TagObjectType, TagTarget};

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
    let columns = bound.get_table_structure(table_id, head).unwrap();
    assert!(bound.get_partition_spec(table_id, head).unwrap().is_none());
    assert!(bound.get_sort_spec(table_id, head).unwrap().is_none());
    let statistics = bound.get_table_statistics(table_id, head).unwrap();
    assert!(statistics.table.is_none(), "{statistics:?}");
    assert!(statistics.columns.is_empty(), "{statistics:?}");
    assert!(statistics.files.is_empty(), "{statistics:?}");
    let summary = bound.get_table_summary_statistics(table_id, head).unwrap();
    assert!(summary.table.is_none(), "{summary:?}");
    assert!(summary.columns.is_empty(), "{summary:?}");
    for target in [
        TagTarget::Object {
            object_type: TagObjectType::Table,
            object_id: table_id,
        },
        TagTarget::Column {
            table_id,
            column_id: columns[0].column_id,
        },
    ] {
        assert!(bound.get_tags(target, head).unwrap().is_empty());
    }
    assert!(bound.list_all_object_tags(head).unwrap().is_empty());
    assert!(bound.list_all_column_tags(head).unwrap().is_empty());
    assert_eq!(
        bound.get_view_id_by_name(schema_id, "v", head).unwrap(),
        None
    );
    assert!(bound.list_views(schema_id, head).unwrap().is_empty());
    let files = bound.get_table_files_for_select(table_id, head).unwrap();
    assert_eq!(files.len(), 1);
    assert!(files[0].partition_values.is_empty());
    let page = bound
        .get_table_file_metadata_page(table_id, head, None, FILE_METADATA_BATCH_SIZE)
        .unwrap();
    assert_eq!(page.len(), 1);
    assert!(page[0].column_statistics.is_empty());
    assert!(
        bound
            .get_inlined_deletes(table_id, head)
            .unwrap()
            .is_empty()
    );
    assert_eq!(bound.get_current_snapshot().unwrap(), head);
    read.end().unwrap();

    assert_eq!(count_at_head(provider, schema).await, 100);
}

/// A consistent read of a PostgreSQL catalog without its optional tables
/// reads each as empty, as a read outside one does, and sends no statement
/// that fails. Every read after the first succeeding in the same transaction
/// shows that no statement failed, as a failed one ends the transaction. That
/// the server ran no `SAVEPOINT` is the guard against a read that hides a
/// failure under one (`pg_stat_statements` records completed statements, so it
/// counts the savepoints but not a failed statement).
#[cfg(feature = "write-postgres")]
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn a_postgres_read_skips_missing_optional_tables() {
    use datafusion_ducklake::{PostgresMetadataProvider, PostgresSingleCatalogMetadataWriter};
    use sqlx::postgres::PgPoolOptions;
    use testcontainers::ImageExt;
    use testcontainers::runners::AsyncRunner;
    use testcontainers_modules::postgres::Postgres;

    let container = Postgres::default()
        .with_cmd(["postgres", "-c", "shared_preload_libraries=pg_stat_statements"])
        .start()
        .await
        .unwrap();
    let port = container.get_host_port_ipv4(5432).await.unwrap();
    let url = format!("postgresql://postgres:postgres@127.0.0.1:{port}/postgres");
    let writer = PostgresSingleCatalogMetadataWriter::new_with_init(&url)
        .await
        .unwrap();
    let temp = TempDir::new().unwrap();
    writer
        .set_data_path(temp.path().join("data").to_str().unwrap())
        .unwrap();
    DuckLakeTableWriter::new(Arc::new(writer), Arc::new(LocalFileSystem::new()))
        .unwrap()
        .write_table("main", "t", &[batch(0..100)])
        .await
        .unwrap();
    let admin = sqlx::PgPool::connect(&url).await.unwrap();
    for table in OPTIONAL_TABLES
        .iter()
        .chain(&["ducklake_tag", "ducklake_column_tag"])
    {
        sqlx::query(sqlx::AssertSqlSafe(format!("DROP TABLE {table}")))
            .execute(&admin)
            .await
            .unwrap();
    }
    sqlx::query("CREATE EXTENSION pg_stat_statements")
        .execute(&admin)
        .await
        .unwrap();
    sqlx::query("SELECT pg_stat_statements_reset()")
        .execute(&admin)
        .await
        .unwrap();

    let pool = PgPoolOptions::new()
        .max_connections(1)
        .acquire_timeout(POOL_WAIT)
        .connect(&url)
        .await
        .unwrap();
    reads_without_optional_tables(Arc::new(PostgresMetadataProvider::from_pool(pool)), "main")
        .await;

    // The reads completed and are counted, so a savepoint would be too.
    let (savepoints, snapshot_reads): (i64, i64) = sqlx::query_as(
        "SELECT count(*) FILTER (WHERE query ILIKE '%savepoint%'),
                count(*) FILTER (WHERE query ILIKE '%ducklake_snapshot%')
         FROM pg_stat_statements",
    )
    .fetch_one(&admin)
    .await
    .unwrap();
    assert!(snapshot_reads > 0);
    assert_eq!(savepoints, 0);
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

    /// A flush that commits between a consistent read's read of the inlined
    /// rows and its file listing moves no row out of the read: inside it the
    /// rows are still inlined and the flush's file is not listed. A read after
    /// it sees the file and no inlined row.
    #[tokio::test(flavor = "multi_thread")]
    #[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
    async fn a_flush_between_a_multicatalog_reads_moves_no_row_out_of_it() {
        use datafusion::prelude::SessionContext;
        use datafusion_ducklake::DuckLakeCatalog;

        let (container, writer, provider) = store(1).await;
        let temp = TempDir::new().unwrap();
        writer
            .set_data_path(temp.path().join("data").to_str().unwrap())
            .unwrap();
        let tables =
            DuckLakeTableWriter::new(Arc::clone(&writer), Arc::new(LocalFileSystem::new()))
                .unwrap()
                .with_options(&DuckLakeWriteOptions::default().with_data_inlining_row_limit(10));
        tables
            .write_table("public", "t", &[batch(0..100)])
            .await
            .unwrap();
        tables
            .append_table("public", "t", &[batch(100..102)])
            .await
            .unwrap();
        // The flush reads through a pool of its own: the provider's one
        // connection is the read's.
        let port = container.get_host_port_ipv4(5432).await.unwrap();
        let pool = PgPool::connect(&format!(
            "postgresql://postgres:postgres@127.0.0.1:{port}/postgres"
        ))
        .await
        .unwrap();
        let flusher = DuckLakeCatalog::with_writer(
            Arc::new(
                MulticatalogProvider::with_pool_and_id(pool, provider.catalog_id())
                    .await
                    .unwrap(),
            ),
            Arc::clone(&writer),
        )
        .unwrap();

        let read = provider
            .begin_consistent_read()
            .unwrap()
            .expect("the provider begins a consistent read");
        let bound = Arc::clone(read.provider());
        let head = bound.get_current_snapshot().unwrap();
        let schema_id = bound
            .get_schema_by_name("public", head)
            .unwrap()
            .expect("the schema")
            .schema_id;
        let table_id = bound
            .get_table_by_name(schema_id, "t", head)
            .unwrap()
            .expect("table t")
            .table_id;
        let columns = bound.get_table_structure(table_id, head).unwrap();
        let inlined: usize = bound
            .read_inlined_rows(
                table_id,
                &columns,
                InlinedRowSelection::VisibleAt(head),
                None,
            )
            .unwrap()
            .iter()
            .map(|rows| rows.batch.num_rows())
            .sum();
        assert_eq!(inlined, 2);

        let flushed = flusher
            .flush_inlined_data(&SessionContext::new().state(), Some("public"), Some("t"))
            .await
            .unwrap();
        assert_eq!(flushed.len(), 1);
        assert!(writer.head_snapshot().unwrap() > Some(head));

        assert_eq!(seen(bound.as_ref(), table_id), (1, 2));
        drop(bound);
        read.end().unwrap();
        assert_eq!(seen(&provider, table_id), (2, 0));
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
        let ended = Arc::clone(third.provider());
        // Ending the session commits the read open on it.
        session.end().await.unwrap();
        third.end().unwrap();
        assert_eq!(provider.get_current_snapshot().unwrap(), moved);
        // A read through the ended read's provider begins a new one, on a
        // pooled connection of its own.
        let fourth = ended.begin_consistent_read().unwrap().expect("a read");
        assert_eq!(fourth.provider().get_current_snapshot().unwrap(), moved);
        fourth.end().unwrap();
    }

    /// A consistent read on a read session of a store without its optional
    /// tables reads each as empty, and its transaction on the session's
    /// connection stays usable to its end.
    #[tokio::test(flavor = "multi_thread")]
    #[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
    async fn a_read_session_skips_missing_optional_tables() {
        let (container, writer, provider) = store(1).await;
        let temp = TempDir::new().unwrap();
        writer
            .set_data_path(temp.path().join("data").to_str().unwrap())
            .unwrap();
        DuckLakeTableWriter::new(writer, Arc::new(LocalFileSystem::new()))
            .unwrap()
            .write_table("public", "t", &[batch(0..100)])
            .await
            .unwrap();
        let port = container.get_host_port_ipv4(5432).await.unwrap();
        let admin = PgPool::connect(&format!(
            "postgresql://postgres:postgres@127.0.0.1:{port}/postgres"
        ))
        .await
        .unwrap();
        for table in OPTIONAL_TABLES
            .iter()
            .chain(&["ducklake_catalog_tag", "ducklake_catalog_column_tag"])
        {
            sqlx::query(sqlx::AssertSqlSafe(format!("DROP TABLE {table}")))
                .execute(&admin)
                .await
                .unwrap();
        }

        let session = provider.begin_read_session().await.unwrap();
        reads_without_optional_tables(
            Arc::new(provider.with_read_session(&session).unwrap()),
            "public",
        )
        .await;
        session.end().await.unwrap();
    }
}

/// A consistent read of a MySQL catalog whose snapshot predates the table's
/// inlined-deletion table reads no inlined deletions: MySQL refuses a read of a
/// table created after the snapshot (`ER_TABLE_DEF_CHANGED`, 1412), and the
/// table holds nothing at the snapshot. The same error on another read still
/// fails it.
#[cfg(feature = "write-mysql")]
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn a_mysql_read_before_the_inlined_deletion_table_reads_none() {
    use datafusion::prelude::SessionContext;
    use datafusion_ducklake::{DuckLakeCatalog, MySqlMetadataProvider, MySqlMetadataWriter};
    use sqlx::mysql::MySqlPoolOptions;
    use testcontainers::runners::AsyncRunner;
    use testcontainers_modules::mysql::Mysql;

    let container = Mysql::default().start().await.unwrap();
    let port = container.get_host_port_ipv4(3306).await.unwrap();
    let url = format!("mysql://root@127.0.0.1:{port}/test");
    let writer = Arc::new(
        MySqlMetadataWriter::new_with_init(&url)
            .await
            .unwrap()
            .with_data_inlining_row_limit(10),
    );
    let temp = TempDir::new().unwrap();
    writer
        .set_data_path(temp.path().join("data").to_str().unwrap())
        .unwrap();
    DuckLakeTableWriter::new(writer.clone(), Arc::new(LocalFileSystem::new()))
        .unwrap()
        .with_options(&DuckLakeWriteOptions::default().with_data_inlining_row_limit(0))
        .write_table("main", "t", &[batch(0..100)])
        .await
        .unwrap();
    let admin = sqlx::MySqlPool::connect(&url).await.unwrap();
    let deletion_tables = async || -> i64 {
        sqlx::query_scalar(
            "SELECT count(*) FROM information_schema.tables
             WHERE table_schema = 'test' AND table_name LIKE 'ducklake_inlined_delete%'",
        )
        .fetch_one(&admin)
        .await
        .unwrap()
    };
    assert_eq!(deletion_tables().await, 0);

    let pool = MySqlPoolOptions::new()
        .max_connections(1)
        .acquire_timeout(POOL_WAIT)
        .connect(&url)
        .await
        .unwrap();
    let provider = MySqlMetadataProvider::from_pool(pool);
    let read = provider
        .begin_consistent_read()
        .unwrap()
        .expect("the provider begins a consistent read");
    let bound = Arc::clone(read.provider());
    // The first read fixes the read's snapshot.
    let head = bound.get_current_snapshot().unwrap();
    let schema_id = bound
        .get_schema_by_name("main", head)
        .unwrap()
        .expect("the schema")
        .schema_id;
    let table_id = bound
        .get_table_by_name(schema_id, "t", head)
        .unwrap()
        .expect("table t")
        .table_id;

    // The table's first inlined deletion creates its inlined-deletion table.
    let ctx = SessionContext::new();
    DuckLakeCatalog::register(
        &ctx,
        "lake",
        DuckLakeCatalog::with_writer(
            Arc::new(MySqlMetadataProvider::new(&url).await.unwrap()),
            writer.clone(),
        )
        .unwrap(),
    );
    ctx.sql("DELETE FROM lake.main.t WHERE id < 5")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(deletion_tables().await, 1);
    let moved = writer.head_snapshot().unwrap().expect("a head");
    assert!(moved > head);

    // Inside the read, the table did not exist yet.
    assert!(
        bound
            .get_inlined_deletes(table_id, head)
            .unwrap()
            .is_empty()
    );
    assert!(
        bound
            .get_inlined_file_deletes_for_flush(table_id, head)
            .unwrap()
            .is_empty()
    );
    assert!(
        bound
            .get_inlined_file_deletes_between(table_id, 0, head)
            .unwrap()
            .is_empty()
    );
    assert_eq!(bound.get_current_snapshot().unwrap(), head);

    // A table rebuilt since the snapshot raises the same error, which fails
    // any other read.
    sqlx::query("ALTER TABLE ducklake_view ENGINE = InnoDB")
        .execute(&admin)
        .await
        .unwrap();
    let error = bound.list_views(schema_id, head).unwrap_err().to_string();
    assert!(error.contains("1412"), "{error}");
    // A read that tolerates a missing table does not read that error as one.
    sqlx::query("ALTER TABLE ducklake_partition_info ENGINE = InnoDB")
        .execute(&admin)
        .await
        .unwrap();
    let error = bound
        .get_partition_spec(table_id, head)
        .unwrap_err()
        .to_string();
    assert!(error.contains("1412"), "{error}");
    drop(bound);
    let _ = read.end();

    assert_eq!(
        provider.get_inlined_deletes(table_id, moved).unwrap().len(),
        5
    );
}
