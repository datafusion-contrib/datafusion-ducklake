//! Building a table reads no file's column statistics on any built-in
//! provider (#404). A roll-up of `ducklake_file_column_stats` grows with every
//! file the table holds, and a caller that builds the table per query paid it
//! on every query.
//!
//! On PostgreSQL the tests count statements (see also
//! `postgres_single_catalog_write_tests::building_a_table_reads_no_file_column_stats`).
//! Elsewhere they rename `column_size_bytes` once the files are written: a read
//! that names it fails the build.

#![cfg(feature = "write")]

use std::sync::Arc;

use arrow::array::Int64Array;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::catalog::CatalogProvider;
use datafusion::common::stats::Precision;
use object_store::local::LocalFileSystem;

use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeTableWriter, DuckLakeWriteOptions, MetadataProvider, MetadataWriter,
};

const HIDE_COLUMN_SIZES: &str = "ALTER TABLE ducklake_file_column_stats \
     RENAME COLUMN column_size_bytes TO column_size_bytes_hidden";

fn batch(ids: &[i64]) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
    RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(ids.to_vec()))]).unwrap()
}

/// Write `main.t` as three data files.
async fn write_files(writer: Arc<dyn MetadataWriter>) {
    let tw = DuckLakeTableWriter::new(writer, Arc::new(LocalFileSystem::new()))
        .unwrap()
        .with_options(&DuckLakeWriteOptions::default().with_data_inlining_row_limit(0));
    tw.write_table("main", "t", &[batch(&[1, 2])])
        .await
        .unwrap();
    for id in 3..5 {
        tw.append_table("main", "t", &[batch(&[id])]).await.unwrap();
    }
}

/// Build `main.t` through `provider` and check it reports no column size.
async fn assert_table_builds(provider: impl MetadataProvider + 'static) {
    let catalog = DuckLakeCatalog::new(provider).unwrap();
    let table = catalog
        .schema("main")
        .expect("schema main")
        .table("t")
        .await
        .expect("building the table must not read file column sizes")
        .expect("table t");
    let statistics = table.statistics().expect("table statistics");
    assert!(matches!(statistics.num_rows, Precision::Inexact(4)));
    assert!(
        statistics
            .column_statistics
            .iter()
            .all(|column| column.byte_size == Precision::Absent)
    );
}

#[cfg(feature = "write-sqlite")]
#[tokio::test(flavor = "multi_thread")]
async fn sqlite_table_build_reads_no_file_column_sizes() {
    use datafusion_ducklake::{SqliteMetadataProvider, SqliteMetadataWriter};

    let temp = tempfile::TempDir::new().unwrap();
    let url = format!("sqlite:{}?mode=rwc", temp.path().join("c.db").display());
    let writer = SqliteMetadataWriter::new_with_init(&url).await.unwrap();
    writer
        .set_data_path(temp.path().join("data").to_str().unwrap())
        .unwrap();
    write_files(Arc::new(writer)).await;

    let pool = sqlx::SqlitePool::connect(&url).await.unwrap();
    sqlx::query(HIDE_COLUMN_SIZES).execute(&pool).await.unwrap();
    pool.close().await;

    assert_table_builds(SqliteMetadataProvider::new(&url).await.unwrap()).await;
}

#[cfg(feature = "write-duckdb")]
#[tokio::test(flavor = "multi_thread")]
async fn duckdb_table_build_reads_no_file_column_sizes() {
    use datafusion_ducklake::{DuckdbMetadataProvider, DuckdbMetadataWriter};

    let temp = tempfile::TempDir::new().unwrap();
    let path = temp.path().join("c.ducklake");
    let path = path.to_str().unwrap();
    let writer = DuckdbMetadataWriter::new_with_init(path).unwrap();
    writer
        .set_data_path(temp.path().join("data").to_str().unwrap())
        .unwrap();
    write_files(Arc::new(writer)).await;

    duckdb::Connection::open(path)
        .unwrap()
        .execute_batch(HIDE_COLUMN_SIZES)
        .unwrap();

    assert_table_builds(DuckdbMetadataProvider::new(path).unwrap()).await;
}

#[cfg(feature = "write-mysql")]
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn mysql_table_build_reads_no_file_column_sizes() {
    use datafusion_ducklake::{MySqlMetadataProvider, MySqlMetadataWriter};
    use testcontainers::runners::AsyncRunner;
    use testcontainers_modules::mysql::Mysql;

    let container = Mysql::default().start().await.unwrap();
    let port = container.get_host_port_ipv4(3306).await.unwrap();
    let url = format!("mysql://root@127.0.0.1:{port}/test");
    let temp = tempfile::TempDir::new().unwrap();
    let writer = MySqlMetadataWriter::new_with_init(&url).await.unwrap();
    writer
        .set_data_path(temp.path().join("data").to_str().unwrap())
        .unwrap();
    write_files(Arc::new(writer)).await;

    let pool = sqlx::MySqlPool::connect(&url).await.unwrap();
    sqlx::query(HIDE_COLUMN_SIZES).execute(&pool).await.unwrap();

    assert_table_builds(MySqlMetadataProvider::new(&url).await.unwrap()).await;
}

#[cfg(all(feature = "write-postgres", feature = "multicatalog-postgres"))]
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn multicatalog_table_build_reads_no_file_column_sizes() {
    use datafusion_ducklake::{
        MulticatalogManager, MulticatalogProvider, PostgresMetadataWriter,
        initialize_multicatalog_schema,
    };
    use testcontainers::ImageExt;
    use testcontainers::runners::AsyncRunner;
    use testcontainers_modules::postgres::Postgres;

    // This provider reads "column does not exist" as a catalog without
    // statistics, so a renamed column would hide a read: count statements.
    let container = Postgres::default()
        .with_cmd(["-c", "fsync=off", "-c", "shared_preload_libraries=pg_stat_statements"])
        .start()
        .await
        .unwrap();
    let port = container.get_host_port_ipv4(5432).await.unwrap();
    let url = format!("postgresql://postgres:postgres@127.0.0.1:{port}/postgres");
    let pool = sqlx::PgPool::connect(&url).await.unwrap();
    initialize_multicatalog_schema(&pool).await.unwrap();
    let catalog_id = MulticatalogManager::new(pool.clone())
        .create_catalog("lake")
        .await
        .unwrap();
    let writer = PostgresMetadataWriter::with_pool(pool.clone(), catalog_id)
        .await
        .unwrap();
    let temp = tempfile::TempDir::new().unwrap();
    writer
        .set_data_path(temp.path().join("data").to_str().unwrap())
        .unwrap();
    write_files(Arc::new(writer)).await;

    sqlx::query("CREATE EXTENSION pg_stat_statements")
        .execute(&pool)
        .await
        .unwrap();
    let provider = MulticatalogProvider::with_pool_and_id(pool.clone(), catalog_id)
        .await
        .unwrap();
    sqlx::query("SELECT pg_stat_statements_reset()")
        .execute(&pool)
        .await
        .unwrap();
    assert_table_builds(provider).await;
    let file_column_stats_reads: i64 = sqlx::query_scalar(
        "SELECT COALESCE(SUM(calls), 0)::BIGINT FROM pg_stat_statements
         WHERE query LIKE '%ducklake_file_column_stats%'
           AND query NOT LIKE '%pg_stat_statements%'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(file_column_stats_reads, 0);
}
