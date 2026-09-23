#![cfg(any(feature = "write-sqlite", feature = "write-postgres", feature = "write-mysql"))]
//! The synchronous `MetadataProvider` bridge, under runtimes it does not own.
//!
//! `CatalogProvider::schema_names`, `CatalogProvider::schema`,
//! `SchemaProvider::table_names` and `SchemaProvider::table_exist` are plain
//! `fn` in DataFusion, so every catalog round trip is awaited from a blocking
//! context inside this crate. These tests pin the shapes that context has to
//! survive: a multi-threaded runtime with fewer spare threads than there are
//! concurrent scans, a single-threaded runtime, a thread with no runtime at
//! all, and a catalog whose connections would otherwise belong to the runtime
//! the call blocks.
//!
//! And the one shape it does not survive: a pool the caller opened and handed
//! over, whose connections this crate never gets to move.

use std::sync::Arc;
use std::time::Duration;

use arrow::array::{Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::prelude::*;
use object_store::local::LocalFileSystem;
use tempfile::TempDir;

use datafusion_ducklake::{DuckLakeCatalog, DuckLakeTableWriter, MetadataProvider, MetadataWriter};

/// Rows the fixture table holds, and so the row count every scan must return.
const FIXTURE_ROWS: usize = 3;

fn fixture_batch() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, true),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(vec![1_i64, 2, 3])),
            Arc::new(StringArray::from(vec![Some("a"), Some("b"), Some("c")])),
        ],
    )
    .unwrap()
}

/// Publishes the fixture table through `writer`, which must already carry a
/// data path.
async fn write_fixture(writer: Arc<dyn MetadataWriter>) {
    DuckLakeTableWriter::new(writer, Arc::new(LocalFileSystem::new()))
        .unwrap()
        .write_table("main", "t", &[fixture_batch()])
        .await
        .unwrap();
}

// ---------------------------------------------------------------------------
// A multi-threaded runtime with fewer spare threads than concurrent scans
// ---------------------------------------------------------------------------

/// Concurrent scans, against a runtime deliberately too narrow to lend a thread
/// per in-flight catalog call.
///
/// A blocking bridge that drives its future on the embedder's runtime needs one
/// of that runtime's threads per in-flight call — `block_in_place` hands the
/// worker's queue to a thread from the blocking pool, so the thread it needs
/// just moves pools. Once in-flight calls outnumber the threads available, the
/// future nobody is left to poll is the one every blocked thread is waiting
/// for, and the runtime stops completely.
///
/// The numbers below are the smallest that reach that threshold on any machine:
/// with the bridge driving futures on this runtime, sixteen concurrent scans
/// against four blocking threads never returns a row.
#[cfg(feature = "write-postgres")]
#[test]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
fn concurrent_scans_survive_a_runtime_with_no_spare_threads() {
    const WORKER_THREADS: usize = 2;
    const BLOCKING_THREADS: usize = 4;
    const CONCURRENT_SCANS: usize = 16;

    // A runtime of its own for the fixture, so the container outlives the
    // runtime the scans run on and is torn down however the test ends.
    let fixture_runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let temp_dir = TempDir::new().unwrap();
    let (container, conn_str) = fixture_runtime.block_on(postgres_fixture(&temp_dir));

    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    std::thread::Builder::new()
        .name("narrow-runtime".to_string())
        .spawn(move || {
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(WORKER_THREADS)
                .max_blocking_threads(BLOCKING_THREADS)
                .enable_all()
                .build()
                .unwrap();
            let scanned = runtime.block_on(async move {
                let provider = datafusion_ducklake::PostgresMetadataProvider::new(&conn_str)
                    .await
                    .unwrap();
                let ctx = SessionContext::new();
                ctx.register_catalog("lake", Arc::new(DuckLakeCatalog::new(provider).unwrap()));

                let scans = (0..CONCURRENT_SCANS)
                    .map(|_| {
                        let ctx = ctx.clone();
                        tokio::spawn(async move {
                            ctx.sql("SELECT id, name FROM lake.main.t ORDER BY id")
                                .await?
                                .collect()
                                .await
                        })
                    })
                    .collect::<Vec<_>>();

                let mut rows = 0;
                for scan in scans {
                    let batches = scan.await.unwrap()?;
                    rows += batches.iter().map(|b| b.num_rows()).sum::<usize>();
                }
                Ok::<_, datafusion::error::DataFusionError>(rows)
            });
            let _ = tx.send(scanned);
        })
        .unwrap();

    // The deadline is enforced from this thread rather than with
    // `tokio::time::timeout` inside that runtime: a runtime whose every thread
    // is blocked does not fire its own timers either, so the timeout would be
    // one more thing waiting for the deadlock to clear.
    let outcome = rx.recv_timeout(Duration::from_secs(120));
    fixture_runtime.block_on(async move { drop(container) });

    let rows = outcome
        .expect("the concurrent scans never finished: the query runtime deadlocked")
        .expect("a scan failed");
    assert_eq!(rows, CONCURRENT_SCANS * FIXTURE_ROWS);
}

// ---------------------------------------------------------------------------
// Runtimes the embedder picks, which this crate does not get to choose
// ---------------------------------------------------------------------------

/// A whole write-then-read cycle on a single-threaded runtime.
///
/// `tokio::task::block_in_place` panics outside a multi-threaded runtime, so a
/// bridge that reaches for it unconditionally makes every catalog call under
/// `#[tokio::main(flavor = "current_thread")]` a panic rather than a query.
#[cfg(feature = "write-sqlite")]
#[test]
fn a_scan_runs_under_a_current_thread_runtime() {
    use datafusion_ducklake::SqliteMetadataProvider;

    let temp_dir = TempDir::new().unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let rows = runtime.block_on(async {
        let conn_str = sqlite_fixture(&temp_dir).await;
        let provider = SqliteMetadataProvider::new(&conn_str).await.unwrap();
        let ctx = SessionContext::new();
        ctx.register_catalog("lake", Arc::new(DuckLakeCatalog::new(provider).unwrap()));

        let batches = ctx
            .sql("SELECT id, name FROM lake.main.t ORDER BY id")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        batches.iter().map(|b| b.num_rows()).sum::<usize>()
    });

    assert_eq!(rows, FIXTURE_ROWS);
}

/// A PostgreSQL catalog opened, and then read, on a `current_thread` runtime.
///
/// Opening the pool is what this pins. A sqlx connection is only ever reported
/// readable by the I/O driver of the runtime that opened it, so a pool opened
/// on the single-threaded runtime below takes its readiness from the very
/// runtime the first catalog call then blocks, and waits on a driver nobody is
/// left to drive.
/// The SQLite cases cannot reach this: sqlx runs a SQLite connection on a
/// thread of its own rather than through an I/O driver, so its readiness does
/// not depend on a runtime at all.
#[cfg(feature = "write-postgres")]
#[test]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
fn a_postgres_scan_runs_under_a_current_thread_runtime() {
    // A runtime of its own for the fixture, so the container outlives the
    // runtime the scan runs on and is torn down however the test ends.
    let fixture_runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let temp_dir = TempDir::new().unwrap();
    let (container, conn_str) = fixture_runtime.block_on(postgres_fixture(&temp_dir));

    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    std::thread::Builder::new()
        .name("current-thread-runtime".to_string())
        .spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            let scanned = runtime.block_on(async move {
                let provider = datafusion_ducklake::PostgresMetadataProvider::new(&conn_str)
                    .await
                    .unwrap();
                let ctx = SessionContext::new();
                ctx.register_catalog("lake", Arc::new(DuckLakeCatalog::new(provider).unwrap()));

                let batches = ctx
                    .sql("SELECT id, name FROM lake.main.t ORDER BY id")
                    .await?
                    .collect()
                    .await?;
                Ok::<_, datafusion::error::DataFusionError>(
                    batches.iter().map(|b| b.num_rows()).sum::<usize>(),
                )
            });
            let _ = tx.send(scanned);
        })
        .unwrap();

    // Enforced from this thread rather than inside that runtime, for the reason
    // the concurrency test gives: a runtime waiting on its own driver has no
    // thread left to fire a timer with.
    let outcome = rx.recv_timeout(Duration::from_secs(120));
    fixture_runtime.block_on(async move { drop(container) });

    let rows = outcome
        .expect("the scan never finished: the catalog call waited on the runtime it had blocked")
        .expect("the scan failed");
    assert_eq!(rows, FIXTURE_ROWS);
}

/// A MySQL catalog opened, and then read, on a `current_thread` runtime.
///
/// The PostgreSQL case above pins the class; this pins that MySQL is in it.
/// sqlx registers a MySQL socket with the I/O driver of the runtime that opened
/// it exactly as it does a PostgreSQL one, so `MySqlMetadataProvider::new` has
/// to open its pool on the catalog runtime for the same reason, and would fail
/// the same way if it stopped.
#[cfg(feature = "write-mysql")]
#[test]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
fn a_mysql_scan_runs_under_a_current_thread_runtime() {
    // A runtime of its own for the fixture, so the container outlives the
    // runtime the scan runs on and is torn down however the test ends.
    let fixture_runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let temp_dir = TempDir::new().unwrap();
    let (container, conn_str) = fixture_runtime.block_on(mysql_fixture(&temp_dir));

    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    std::thread::Builder::new()
        .name("mysql-current-thread-runtime".to_string())
        .spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            let scanned = runtime.block_on(async move {
                let provider = datafusion_ducklake::MySqlMetadataProvider::new(&conn_str)
                    .await
                    .unwrap();
                let ctx = SessionContext::new();
                ctx.register_catalog("lake", Arc::new(DuckLakeCatalog::new(provider).unwrap()));

                let batches = ctx
                    .sql("SELECT id, name FROM lake.main.t ORDER BY id")
                    .await?
                    .collect()
                    .await?;
                Ok::<_, datafusion::error::DataFusionError>(
                    batches.iter().map(|b| b.num_rows()).sum::<usize>(),
                )
            });
            let _ = tx.send(scanned);
        })
        .unwrap();

    // Enforced from this thread rather than inside that runtime, for the reason
    // the concurrency test gives: a runtime waiting on its own driver has no
    // thread left to fire a timer with.
    let outcome = rx.recv_timeout(Duration::from_secs(120));
    fixture_runtime.block_on(async move { drop(container) });

    let rows = outcome
        .expect("the scan never finished: the catalog call waited on the runtime it had blocked")
        .expect("the scan failed");
    assert_eq!(rows, FIXTURE_ROWS);
}

// ---------------------------------------------------------------------------
// The pool the caller opened, which this crate does not get to move
// ---------------------------------------------------------------------------

/// A PostgreSQL catalog over an adopted pool, on a `current_thread` runtime,
/// asserted as it behaves rather than as one would want it to.
///
/// `from_pool` adopts a pool; it does not move it. `PoolOptions::connect` opens
/// a connection before it returns, so a pool built the ordinary way on the
/// caller's runtime already holds a socket registered with that runtime's I/O
/// driver, and only that driver ever reports it readable. The first catalog
/// call is what stops that driver being driven, so the connection never comes
/// back and the call ends at the pool's acquire timeout.
///
/// Owning a driver cannot rescue this — those connections predate the call —
/// and that is the line the crate docs and the `from_pool` rustdoc draw: the
/// guarantee covers the pools this crate opens. Every other test here builds
/// its catalog with `new`, so this is the only one that would notice an adopted
/// pool moving further from it. The timeout is not a regression either: a
/// bridge reaching for `block_in_place` panicked outright under this runtime,
/// so a typed error after a wait is what the change bought this shape.
#[cfg(feature = "write-postgres")]
#[test]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
fn an_adopted_pool_still_stalls_under_a_current_thread_runtime() {
    use datafusion_ducklake::{DuckLakeError, PostgresMetadataProvider};
    use sqlx::postgres::PgPoolOptions;

    // A runtime of its own for the fixture, so the container outlives the
    // runtime the call runs on and is torn down however the test ends.
    let fixture_runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let temp_dir = TempDir::new().unwrap();
    let (container, conn_str) = fixture_runtime.block_on(postgres_fixture(&temp_dir));

    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    std::thread::Builder::new()
        .name("adopted-pool-current-thread".to_string())
        .spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            let outcome = runtime.block_on(async move {
                // What an embedder would write, with the acquire timeout
                // shortened from its default of thirty seconds: this call can
                // only ever reach that timeout, so the default would be thirty
                // seconds of suite time spent waiting for a known answer.
                let pool = PgPoolOptions::new()
                    .max_connections(5)
                    .acquire_timeout(Duration::from_secs(10))
                    .connect(&conn_str)
                    .await
                    .unwrap();
                PostgresMetadataProvider::from_pool(pool).get_current_snapshot()
            });
            let _ = tx.send(outcome);
        })
        .unwrap();

    // Enforced from this thread for the reason the tests above give, and set
    // well clear of the acquire timeout so that a late answer is still an
    // answer rather than a missing one.
    let outcome = rx.recv_timeout(Duration::from_secs(120));
    fixture_runtime.block_on(async move { drop(container) });

    let error = outcome
        .expect("the catalog call neither answered nor gave up")
        .expect_err("an adopted pool takes its readiness from the runtime the call blocks");
    assert!(
        matches!(error, DuckLakeError::Sqlx(sqlx::Error::PoolTimedOut)),
        "expected the pool's acquire timeout, got {error:?}"
    );
}

/// A catalog call from a thread with no runtime at all.
///
/// `MetadataProvider` is a public synchronous trait, so an embedder may call it
/// from anywhere — including a plain `std::thread`, where `Handle::current()`
/// panics.
#[cfg(feature = "write-sqlite")]
#[test]
fn a_provider_call_runs_off_the_runtime() {
    use datafusion_ducklake::SqliteMetadataProvider;

    let temp_dir = TempDir::new().unwrap();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let provider = runtime.block_on(async {
        let conn_str = sqlite_fixture(&temp_dir).await;
        SqliteMetadataProvider::new(&conn_str).await.unwrap()
    });

    let snapshot = std::thread::spawn(move || provider.get_current_snapshot())
        .join()
        .unwrap()
        .expect("a catalog call off the runtime");
    assert!(snapshot >= 1, "the fixture publishes at least one snapshot");
}

/// Starts a PostgreSQL container and publishes the fixture into a catalog on
/// it, returning the container — which owns it for as long as it is held — and
/// its connection string.
#[cfg(feature = "write-postgres")]
async fn postgres_fixture(
    temp_dir: &TempDir,
) -> (
    testcontainers::ContainerAsync<testcontainers_modules::postgres::Postgres>,
    String,
) {
    use testcontainers::runners::AsyncRunner;
    use testcontainers_modules::postgres::Postgres;

    use datafusion_ducklake::PostgresSingleCatalogMetadataWriter;

    let container = Postgres::default().start().await.unwrap();
    let port = container.get_host_port_ipv4(5432).await.unwrap();
    let conn_str = format!("postgresql://postgres:postgres@127.0.0.1:{port}/postgres");

    let writer = PostgresSingleCatalogMetadataWriter::new_with_init(&conn_str)
        .await
        .unwrap();
    let data_path = temp_dir.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();
    writer.set_data_path(data_path.to_str().unwrap()).unwrap();
    write_fixture(Arc::new(writer)).await;

    (container, conn_str)
}

/// Starts a MySQL container and publishes the fixture into a catalog on it,
/// returning the container — which owns it for as long as it is held — and its
/// connection string.
#[cfg(feature = "write-mysql")]
async fn mysql_fixture(
    temp_dir: &TempDir,
) -> (
    testcontainers::ContainerAsync<testcontainers_modules::mysql::Mysql>,
    String,
) {
    use testcontainers::runners::AsyncRunner;
    use testcontainers_modules::mysql::Mysql;

    use datafusion_ducklake::MySqlMetadataWriter;

    let container = Mysql::default().start().await.unwrap();
    let port = container.get_host_port_ipv4(3306).await.unwrap();
    let conn_str = format!("mysql://root@127.0.0.1:{port}/test");

    let writer = MySqlMetadataWriter::new_with_init(&conn_str).await.unwrap();
    let data_path = temp_dir.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();
    writer.set_data_path(data_path.to_str().unwrap()).unwrap();
    write_fixture(Arc::new(writer)).await;

    (container, conn_str)
}

/// Publishes the fixture into a SQLite catalog under `temp_dir`, returning its
/// connection string.
#[cfg(feature = "write-sqlite")]
async fn sqlite_fixture(temp_dir: &TempDir) -> String {
    use datafusion_ducklake::SqliteMetadataWriter;

    let db_path = temp_dir.path().join("catalog.db");
    let data_path = temp_dir.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();

    let conn_str = format!("sqlite:{}?mode=rwc", db_path.display());
    let writer = SqliteMetadataWriter::new_with_init(&conn_str)
        .await
        .unwrap();
    writer.set_data_path(data_path.to_str().unwrap()).unwrap();
    write_fixture(Arc::new(writer)).await;

    conn_str
}
