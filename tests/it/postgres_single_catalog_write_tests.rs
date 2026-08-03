#![cfg(feature = "write-postgres")]
//! Integration tests for the **single-catalog** (standard DuckLake) Postgres writer.
//!
//! The point of `PostgresSingleCatalogMetadataWriter` is that it produces the
//! same catalog shape as the SQLite/MySQL writers and DuckDB's `ducklake`
//! extension — unlike `PostgresMetadataWriter`, which writes this crate's
//! library-specific multicatalog layout. So alongside the usual write/read
//! coverage these tests assert the *shape*:
//!
//! - no `ducklake_catalog*` map tables exist
//! - no `catalog_id` column on any DuckLake table
//! - `catalog_id()` is `None`, so file paths stay unscoped
//! - `ducklake_schema.path` / `ducklake_table.path` are bare relative names
//!   (no `cat_{id}/` prefix)
//!
//! Plus the behaviour the multicatalog path cannot offer: SQL `CREATE TABLE AS
//! SELECT`.

use std::sync::Arc;

use arrow::array::{Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::prelude::*;
use object_store::local::LocalFileSystem;
use sqlx::postgres::{PgPool, PgPoolOptions};
use tempfile::TempDir;
use testcontainers::ContainerAsync;
use testcontainers::runners::AsyncRunner;
use testcontainers_modules::postgres::Postgres;

use datafusion_ducklake::metadata_writer::{ColumnDef, MetadataWriter, WriteMode};
use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeTableWriter, PostgresMetadataProvider,
    PostgresSingleCatalogMetadataWriter,
};

/// Spin up Postgres, open a single-catalog writer against it, and point the
/// catalog at a temp data directory. Returns everything the caller must keep
/// alive (dropping the container tears the database down).
async fn setup() -> anyhow::Result<(
    PostgresSingleCatalogMetadataWriter,
    PgPool,
    String,
    TempDir,
    ContainerAsync<Postgres>,
)> {
    let container = Postgres::default().start().await?;
    let port = container.get_host_port_ipv4(5432).await?;
    let conn_str = format!("postgresql://postgres:postgres@127.0.0.1:{}/postgres", port);

    let writer = PostgresSingleCatalogMetadataWriter::new_with_init(&conn_str).await?;

    let temp_dir = TempDir::new()?;
    let data_path = temp_dir.path().join("data");
    std::fs::create_dir_all(&data_path)?;
    writer.set_data_path(data_path.to_str().unwrap())?;

    let pool = PgPoolOptions::new()
        .max_connections(5)
        .connect(&conn_str)
        .await?;

    Ok((writer, pool, conn_str, temp_dir, container))
}

fn object_store() -> Arc<dyn object_store::ObjectStore> {
    Arc::new(LocalFileSystem::new())
}

fn batch(ids: Vec<i64>, names: Vec<Option<&str>>) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, true),
    ]));
    RecordBatch::try_new(
        schema,
        vec![Arc::new(Int64Array::from(ids)), Arc::new(StringArray::from(names))],
    )
    .unwrap()
}

fn cols() -> Vec<ColumnDef> {
    vec![
        ColumnDef::new("id", "int64", false).unwrap(),
        ColumnDef::new("name", "varchar", true).unwrap(),
    ]
}

async fn read_context(conn_str: &str) -> SessionContext {
    let provider = PostgresMetadataProvider::new(conn_str).await.unwrap();
    let catalog = DuckLakeCatalog::new(provider).unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("lake", Arc::new(catalog));
    ctx
}

async fn table_exists(pool: &PgPool, name: &str) -> bool {
    let count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM information_schema.tables
         WHERE table_schema = 'public' AND table_name = $1",
    )
    .bind(name)
    .fetch_one(pool)
    .await
    .unwrap();
    count > 0
}

// ---------------------------------------------------------------------------
// Catalog shape: the actual point of #165
// ---------------------------------------------------------------------------

/// The multicatalog map tables must NOT be created by this writer. Their
/// presence is what makes a catalog unreadable by other DuckLake tools.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn creates_no_multicatalog_tables() {
    let (_writer, pool, _conn, _tmp, _container) = setup().await.unwrap();

    for t in ["ducklake_catalog", "ducklake_catalog_snapshot_map", "ducklake_catalog_schema_map"] {
        assert!(
            !table_exists(&pool, t).await,
            "single-catalog writer must not create the multicatalog table `{t}`"
        );
    }

    // ...and the standard tables must be there.
    for t in [
        "ducklake_metadata",
        "ducklake_snapshot",
        "ducklake_snapshot_changes",
        "ducklake_schema",
        "ducklake_table",
        "ducklake_column",
        "ducklake_data_file",
        "ducklake_delete_file",
        "ducklake_table_stats",
        "ducklake_file_column_stats",
        "ducklake_table_column_stats",
        "ducklake_schema_versions",
        "ducklake_partition_info",
        "ducklake_partition_column",
        "ducklake_file_partition_value",
        "ducklake_sort_info",
        "ducklake_sort_expression",
        "ducklake_files_scheduled_for_deletion",
    ] {
        assert!(table_exists(&pool, t).await, "missing standard table `{t}`");
    }
}

/// No DuckLake table may carry a `catalog_id` column — that column is the
/// multicatalog layout's scoping mechanism and is not in the spec.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn no_table_carries_a_catalog_id_column() {
    let (_writer, pool, _conn, _tmp, _container) = setup().await.unwrap();

    let offenders: Vec<String> = sqlx::query_scalar(
        "SELECT table_name FROM information_schema.columns
         WHERE table_schema = 'public'
           AND table_name LIKE 'ducklake\\_%'
           AND column_name = 'catalog_id'
         ORDER BY table_name",
    )
    .fetch_all(&pool)
    .await
    .unwrap();

    assert!(
        offenders.is_empty(),
        "catalog_id must not appear on any DuckLake table; found on {offenders:?}"
    );
}

/// `catalog_id()` returning `None` is what keeps file placement unscoped —
/// `{data_path}/{schema}/{table}/…` rather than `{data_path}/cat_{id}/…`.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn catalog_id_is_none() {
    let (writer, _pool, _conn, _tmp, _container) = setup().await.unwrap();
    assert_eq!(writer.catalog_id(), None);
}

/// Schema and table paths are stored as bare relative names, matching the other
/// single-catalog backends.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn paths_are_unscoped_and_relative() {
    let (writer, pool, _conn, _tmp, _container) = setup().await.unwrap();

    DuckLakeTableWriter::new(Arc::new(writer), object_store())
        .unwrap()
        .write_table("main", "users", &[batch(vec![1, 2], vec![Some("a"), None])])
        .await
        .unwrap();

    let (schema_path, schema_rel): (String, bool) = sqlx::query_as(
        "SELECT path, path_is_relative FROM ducklake_schema WHERE schema_name = 'main'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(schema_path, "main", "schema path must not be cat_-scoped");
    assert!(schema_rel);

    let (table_path, table_rel): (String, bool) = sqlx::query_as(
        "SELECT path, path_is_relative FROM ducklake_table WHERE table_name = 'users'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(table_path, "users");
    assert!(table_rel);

    // The data file lands relative to the resolved table path.
    let file_rel: bool = sqlx::query_scalar("SELECT path_is_relative FROM ducklake_data_file")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert!(file_rel);
}

/// `ducklake_column` must be the bare upstream shape so a versioned column can
/// hold several rows sharing one `column_id`. The multicatalog writer uses a
/// composite PK instead; upstream and the other single-catalog backends use none.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn ducklake_column_has_no_primary_key() {
    let (_writer, pool, _conn, _tmp, _container) = setup().await.unwrap();

    let pk_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM information_schema.table_constraints
         WHERE table_schema = 'public'
           AND table_name = 'ducklake_column'
           AND constraint_type = 'PRIMARY KEY'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(pk_count, 0, "ducklake_column must be a bare table");
}

// ---------------------------------------------------------------------------
// Write / read round-trips
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn write_and_read_back() {
    let (writer, _pool, conn_str, _tmp, _container) = setup().await.unwrap();

    let result = DuckLakeTableWriter::new(Arc::new(writer), object_store())
        .unwrap()
        .write_table(
            "main",
            "users",
            &[batch(vec![1, 2, 3], vec![Some("a"), Some("b"), None])],
        )
        .await
        .unwrap();
    assert_eq!(result.records_written, 3);
    assert_eq!(result.files_written, 1);
    assert!(result.snapshot_id > 0);

    let ctx = read_context(&conn_str).await;
    let batches = ctx
        .sql("SELECT id, name FROM lake.main.users ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total, 3);
    let ids = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(ids.values(), &[1, 2, 3]);
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn append_accumulates_rows_and_snapshots() {
    let (writer, pool, conn_str, _tmp, _container) = setup().await.unwrap();
    let tw = DuckLakeTableWriter::new(Arc::new(writer), object_store()).unwrap();

    tw.write_table("main", "t", &[batch(vec![1], vec![Some("a")])])
        .await
        .unwrap();
    tw.append_table("main", "t", &[batch(vec![2], vec![Some("b")])])
        .await
        .unwrap();

    let snapshots: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM ducklake_snapshot")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(snapshots, 2, "each write publishes exactly one snapshot");

    // An unchanged column keeps its column_id across writes (it is the parquet
    // field_id; churning it would make already-written files read back NULL).
    let live_ids: Vec<i64> = sqlx::query_scalar(
        "SELECT column_id FROM ducklake_column WHERE end_snapshot IS NULL ORDER BY column_order",
    )
    .fetch_all(&pool)
    .await
    .unwrap();
    assert_eq!(live_ids.len(), 2);

    let ctx = read_context(&conn_str).await;
    let batches = ctx
        .sql("SELECT count(*) AS n FROM lake.main.t")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let n = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!(n, 2);
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn replace_retires_the_prior_generation() {
    let (writer, pool, conn_str, _tmp, _container) = setup().await.unwrap();
    let tw = DuckLakeTableWriter::new(Arc::new(writer), object_store()).unwrap();

    tw.write_table(
        "main",
        "t",
        &[batch(vec![1, 2], vec![Some("a"), Some("b")])],
    )
    .await
    .unwrap();
    tw.write_table("main", "t", &[batch(vec![9], vec![Some("z")])])
        .await
        .unwrap();

    let live: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM ducklake_data_file WHERE end_snapshot IS NULL")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(live, 1, "Replace leaves exactly the new file live");

    let retired: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM ducklake_data_file WHERE end_snapshot IS NOT NULL",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(retired, 1);

    let ctx = read_context(&conn_str).await;
    let batches = ctx
        .sql("SELECT id FROM lake.main.t")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let ids = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(ids.values(), &[9]);
}

// ---------------------------------------------------------------------------
// SQL surface — CTAS is the capability the multicatalog path cannot offer
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn sql_ctas_then_insert_then_select() {
    let (writer, _pool, conn_str, _tmp, _container) = setup().await.unwrap();

    let provider = PostgresMetadataProvider::new(&conn_str).await.unwrap();
    let catalog = DuckLakeCatalog::with_writer(Arc::new(provider), Arc::new(writer)).unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("lake", Arc::new(catalog));

    ctx.sql("CREATE TABLE lake.main.nums AS SELECT 1 AS id, 'one' AS name")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    // A catalog pins its snapshot at construction, so re-open to observe the
    // committed CTAS before appending to it.
    let provider = PostgresMetadataProvider::new(&conn_str).await.unwrap();
    let writer2 = PostgresSingleCatalogMetadataWriter::new(&conn_str)
        .await
        .unwrap();
    let catalog = DuckLakeCatalog::with_writer(Arc::new(provider), Arc::new(writer2)).unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("lake", Arc::new(catalog));

    ctx.sql("INSERT INTO lake.main.nums VALUES (2, 'two')")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    let ctx = read_context(&conn_str).await;
    let batches = ctx
        .sql("SELECT count(*) AS n FROM lake.main.nums")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let n = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!(n, 2);
}

// ---------------------------------------------------------------------------
// Metadata bookkeeping
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn initialize_schema_is_idempotent() {
    let (writer, pool, _conn, _tmp, _container) = setup().await.unwrap();

    // Re-running must neither error nor re-seed the counters.
    writer.initialize_schema().unwrap();
    writer.initialize_schema().unwrap();

    let counters: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM ducklake_metadata
         WHERE key IN ('next_column_id','next_snapshot_id','next_partition_id','next_sort_id')
           AND scope IS NULL",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(counters, 4, "each id counter must be seeded exactly once");
}

/// A DDL commit bumps `schema_version` and writes a ledger row; a pure data
/// write carries the version forward and writes none. Mirrors upstream's
/// `if (SchemaChangesMade()) schema_version++`.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn schema_version_bumps_on_ddl_and_carries_on_data_write() {
    let (writer, pool, _conn, _tmp, _container) = setup().await.unwrap();
    let tw = DuckLakeTableWriter::new(Arc::new(writer), object_store()).unwrap();

    // Creating write == DDL.
    tw.write_table("main", "t", &[batch(vec![1], vec![Some("a")])])
        .await
        .unwrap();
    let v1: i64 = sqlx::query_scalar("SELECT MAX(schema_version) FROM ducklake_snapshot")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(v1, 1);

    // Same-schema append == pure data write.
    tw.append_table("main", "t", &[batch(vec![2], vec![Some("b")])])
        .await
        .unwrap();
    let v2: i64 = sqlx::query_scalar("SELECT MAX(schema_version) FROM ducklake_snapshot")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(v2, 1, "a pure data write must not bump schema_version");

    let ledger: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM ducklake_schema_versions")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(ledger, 1, "only the DDL commit writes a ledger row");
}

/// Snapshot ids are counter-allocated, so they are dense and ordered by commit —
/// the property the `Replace` conflict test relies on.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn snapshot_ids_are_dense_and_commit_ordered() {
    let (writer, pool, _conn, _tmp, _container) = setup().await.unwrap();
    let tw = DuckLakeTableWriter::new(Arc::new(writer), object_store()).unwrap();

    for i in 1..=3 {
        tw.append_table("main", "t", &[batch(vec![i], vec![Some("x")])])
            .await
            .unwrap();
    }

    let ids: Vec<i64> =
        sqlx::query_scalar("SELECT snapshot_id FROM ducklake_snapshot ORDER BY snapshot_id")
            .fetch_all(&pool)
            .await
            .unwrap();
    assert_eq!(ids, vec![1, 2, 3]);
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn row_id_start_advances_monotonically() {
    let (writer, pool, _conn, _tmp, _container) = setup().await.unwrap();
    let tw = DuckLakeTableWriter::new(Arc::new(writer), object_store()).unwrap();

    tw.write_table("main", "t", &[batch(vec![1, 2], vec![None, None])])
        .await
        .unwrap();
    tw.append_table("main", "t", &[batch(vec![3], vec![None])])
        .await
        .unwrap();

    let starts: Vec<i64> =
        sqlx::query_scalar("SELECT row_id_start FROM ducklake_data_file ORDER BY data_file_id")
            .fetch_all(&pool)
            .await
            .unwrap();
    assert_eq!(starts, vec![0, 2]);
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn column_stats_are_persisted() {
    let (writer, pool, _conn, _tmp, _container) = setup().await.unwrap();

    DuckLakeTableWriter::new(Arc::new(writer), object_store())
        .unwrap()
        .write_table(
            "main",
            "t",
            &[batch(vec![5, 1, 9], vec![Some("a"), Some("b"), None])],
        )
        .await
        .unwrap();

    let (min_v, max_v): (Option<String>, Option<String>) = sqlx::query_as(
        "SELECT s.min_value, s.max_value
         FROM ducklake_file_column_stats s
         JOIN ducklake_column c
           ON c.column_id = s.column_id AND c.end_snapshot IS NULL
         WHERE c.column_name = 'id'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(min_v.as_deref(), Some("1"));
    assert_eq!(max_v.as_deref(), Some("9"));

    let rollup: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM ducklake_table_column_stats")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(rollup, 2, "one roll-up row per live column");
}

// ---------------------------------------------------------------------------
// Partition / sort specs
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn partition_spec_set_then_reset() {
    use datafusion_ducklake::PartitionTransform;

    let (writer, pool, _conn, _tmp, _container) = setup().await.unwrap();
    let snapshot = writer.create_snapshot().unwrap();
    let (schema_id, _) = writer.get_or_create_schema("main", None, snapshot).unwrap();
    let (table_id, _) = writer
        .get_or_create_table(schema_id, "t", None, snapshot)
        .unwrap();
    writer.set_columns(table_id, &cols(), snapshot).unwrap();

    writer
        .set_partition_spec(
            table_id,
            &[("name".to_string(), PartitionTransform::Identity)],
        )
        .unwrap();

    let spec = writer.live_partition_spec(table_id).unwrap();
    assert!(spec.is_some(), "spec must be live after SET");

    let live_specs: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM ducklake_partition_info WHERE end_snapshot IS NULL",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(live_specs, 1);

    writer.reset_partition_spec(table_id).unwrap();
    assert!(writer.live_partition_spec(table_id).unwrap().is_none());
}

/// Resetting when nothing is set must not publish an empty snapshot.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn reset_partition_spec_on_unpartitioned_table_is_a_noop() {
    let (writer, pool, _conn, _tmp, _container) = setup().await.unwrap();
    let snapshot = writer.create_snapshot().unwrap();
    let (schema_id, _) = writer.get_or_create_schema("main", None, snapshot).unwrap();
    let (table_id, _) = writer
        .get_or_create_table(schema_id, "t", None, snapshot)
        .unwrap();

    let before: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM ducklake_snapshot")
        .fetch_one(&pool)
        .await
        .unwrap();
    let head = writer.reset_partition_spec(table_id).unwrap();
    let after: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM ducklake_snapshot")
        .fetch_one(&pool)
        .await
        .unwrap();

    assert_eq!(before, after, "no-op reset must not publish a snapshot");
    assert_eq!(head, snapshot);
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn sort_spec_set_then_reset() {
    use datafusion_ducklake::{NullOrder, SortDirection, SortField};

    let (writer, _pool, _conn, _tmp, _container) = setup().await.unwrap();
    let snapshot = writer.create_snapshot().unwrap();
    let (schema_id, _) = writer.get_or_create_schema("main", None, snapshot).unwrap();
    let (table_id, _) = writer
        .get_or_create_table(schema_id, "t", None, snapshot)
        .unwrap();
    writer.set_columns(table_id, &cols(), snapshot).unwrap();

    let field = SortField {
        sort_key_index: 0,
        expression: "id".to_string(),
        dialect: "duckdb".to_string(),
        direction: SortDirection::Asc,
        null_order: NullOrder::NullsLast,
    };
    writer.set_sort_spec(table_id, &[field]).unwrap();
    assert!(writer.live_sort_spec(table_id).unwrap().is_some());

    writer.reset_sort_spec(table_id).unwrap();
    assert!(writer.live_sort_spec(table_id).unwrap().is_none());
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn set_partition_spec_rejects_unknown_column() {
    use datafusion_ducklake::PartitionTransform;

    let (writer, _pool, _conn, _tmp, _container) = setup().await.unwrap();
    let snapshot = writer.create_snapshot().unwrap();
    let (schema_id, _) = writer.get_or_create_schema("main", None, snapshot).unwrap();
    let (table_id, _) = writer
        .get_or_create_table(schema_id, "t", None, snapshot)
        .unwrap();
    writer.set_columns(table_id, &cols(), snapshot).unwrap();

    let err = writer
        .set_partition_spec(
            table_id,
            &[("nope".to_string(), PartitionTransform::Identity)],
        )
        .unwrap_err();
    assert!(
        err.to_string().contains("no live column 'nope'"),
        "unexpected error: {err}"
    );
}

// ---------------------------------------------------------------------------
// Error paths
// ---------------------------------------------------------------------------

/// A data write must never silently change a column's type — that is schema
/// evolution and belongs to `promote_column_type`, which this writer does not
/// implement.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn append_rejects_a_column_type_change() {
    let (writer, _pool, _conn, _tmp, _container) = setup().await.unwrap();

    writer
        .begin_write_transaction("main", "t", &cols(), WriteMode::Replace)
        .unwrap();
    let snapshot = writer.create_snapshot().unwrap();
    let (schema_id, _) = writer.get_or_create_schema("main", None, snapshot).unwrap();
    let (table_id, _) = writer
        .get_or_create_table(schema_id, "t", None, snapshot)
        .unwrap();
    writer.set_columns(table_id, &cols(), snapshot).unwrap();

    let widened = vec![
        ColumnDef::new("id", "varchar", false).unwrap(),
        ColumnDef::new("name", "varchar", true).unwrap(),
    ];
    let err = writer
        .begin_write_transaction("main", "t", &widened, WriteMode::Append)
        .unwrap_err();
    assert!(
        matches!(
            err,
            datafusion_ducklake::DuckLakeError::UnsupportedTypeChange { .. }
        ),
        "unexpected error: {err}"
    );
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn append_rejects_a_new_non_nullable_column() {
    let (writer, _pool, _conn, _tmp, _container) = setup().await.unwrap();
    let snapshot = writer.create_snapshot().unwrap();
    let (schema_id, _) = writer.get_or_create_schema("main", None, snapshot).unwrap();
    let (table_id, _) = writer
        .get_or_create_table(schema_id, "t", None, snapshot)
        .unwrap();
    writer.set_columns(table_id, &cols(), snapshot).unwrap();

    let mut extended = cols();
    extended.push(ColumnDef::new("extra", "int64", false).unwrap());
    let err = writer
        .begin_write_transaction("main", "t", &extended, WriteMode::Append)
        .unwrap_err();
    assert!(
        err.to_string().contains("must be nullable"),
        "unexpected error: {err}"
    );
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn set_columns_rejects_an_empty_schema() {
    let (writer, _pool, _conn, _tmp, _container) = setup().await.unwrap();
    let snapshot = writer.create_snapshot().unwrap();
    let (schema_id, _) = writer.get_or_create_schema("main", None, snapshot).unwrap();
    let (table_id, _) = writer
        .get_or_create_table(schema_id, "t", None, snapshot)
        .unwrap();

    let err = writer.set_columns(table_id, &[], snapshot).unwrap_err();
    assert!(
        err.to_string().contains("at least one column"),
        "unexpected error: {err}"
    );
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn get_data_path_errors_when_unset() {
    let container = Postgres::default().start().await.unwrap();
    let port = container.get_host_port_ipv4(5432).await.unwrap();
    let conn_str = format!("postgresql://postgres:postgres@127.0.0.1:{}/postgres", port);
    let writer = PostgresSingleCatalogMetadataWriter::new_with_init(&conn_str)
        .await
        .unwrap();

    let err = writer.get_data_path().unwrap_err();
    assert!(
        err.to_string().contains("data_path"),
        "unexpected error: {err}"
    );
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn set_data_path_replaces_rather_than_duplicates() {
    let (writer, pool, _conn, _tmp, _container) = setup().await.unwrap();

    writer.set_data_path("/tmp/one").unwrap();
    writer.set_data_path("/tmp/two").unwrap();

    let rows: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM ducklake_metadata WHERE key = 'data_path' AND scope IS NULL",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(rows, 1);
    assert_eq!(writer.get_data_path().unwrap(), "/tmp/two");
}

/// Deletes, upserts, compaction and type promotion are out of scope for this
/// writer and must surface the trait's erroring defaults rather than silently
/// doing nothing.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn unsupported_operations_error() {
    let (writer, _pool, _conn, _tmp, _container) = setup().await.unwrap();
    let snapshot = writer.create_snapshot().unwrap();
    let (schema_id, _) = writer.get_or_create_schema("main", None, snapshot).unwrap();
    let (table_id, _) = writer
        .get_or_create_table(schema_id, "t", None, snapshot)
        .unwrap();
    writer.set_columns(table_id, &cols(), snapshot).unwrap();

    assert!(
        writer
            .promote_column_type(table_id, "id", "varchar")
            .is_err(),
        "type promotion must be rejected"
    );
    assert!(!writer.supports_update(), "UPDATE is not supported");
}

// ---------------------------------------------------------------------------
// Snapshot change records + commit metadata (#209 surface)
// ---------------------------------------------------------------------------

async fn changes_for(pool: &PgPool, snapshot_id: i64) -> Option<String> {
    sqlx::query_scalar("SELECT changes_made FROM ducklake_snapshot_changes WHERE snapshot_id = $1")
        .bind(snapshot_id)
        .fetch_one(pool)
        .await
        .unwrap()
}

/// The creating write records the schema, the table, and the insert — all three
/// accumulate onto the one snapshot's `changes_made`.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn snapshot_changes_record_create_and_insert() {
    let (writer, pool, _conn, _tmp, _container) = setup().await.unwrap();

    let result = DuckLakeTableWriter::new(Arc::new(writer), object_store())
        .unwrap()
        .write_table("main", "t", &[batch(vec![1], vec![Some("a")])])
        .await
        .unwrap();

    let changes = changes_for(&pool, result.snapshot_id).await.unwrap();
    assert!(
        changes.contains(r#"created_schema:"main""#),
        "missing created_schema in {changes:?}"
    );
    assert!(
        changes.contains(r#"created_table:"main"."t""#),
        "missing created_table in {changes:?}"
    );
    assert!(
        changes.contains(&format!("inserted_into_table:{}", result.table_id)),
        "missing inserted_into_table in {changes:?}"
    );
}

/// A `Replace` that supersedes existing files records both the delete and the
/// insert, per `table_write_changes`.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn replace_records_delete_and_insert() {
    let (writer, pool, _conn, _tmp, _container) = setup().await.unwrap();
    let tw = DuckLakeTableWriter::new(Arc::new(writer), object_store()).unwrap();

    tw.write_table("main", "t", &[batch(vec![1], vec![Some("a")])])
        .await
        .unwrap();
    let second = tw
        .write_table("main", "t", &[batch(vec![2], vec![Some("b")])])
        .await
        .unwrap();

    let changes = changes_for(&pool, second.snapshot_id).await.unwrap();
    assert!(
        changes.contains(&format!("deleted_from_table:{}", second.table_id)),
        "missing deleted_from_table in {changes:?}"
    );
    assert!(
        changes.contains(&format!("inserted_into_table:{}", second.table_id)),
        "missing inserted_into_table in {changes:?}"
    );
}

/// Every snapshot must carry exactly one change row — `insert_snapshot` seeds it.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn every_snapshot_has_exactly_one_change_row() {
    let (writer, pool, _conn, _tmp, _container) = setup().await.unwrap();
    let tw = DuckLakeTableWriter::new(Arc::new(writer), object_store()).unwrap();

    tw.write_table("main", "t", &[batch(vec![1], vec![None])])
        .await
        .unwrap();
    tw.append_table("main", "t", &[batch(vec![2], vec![None])])
        .await
        .unwrap();

    let orphans: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM ducklake_snapshot s
         LEFT JOIN ducklake_snapshot_changes c ON c.snapshot_id = s.snapshot_id
         WHERE c.snapshot_id IS NULL",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(orphans, 0, "every snapshot needs a change row");

    let snapshots: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM ducklake_snapshot")
        .fetch_one(&pool)
        .await
        .unwrap();
    let change_rows: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM ducklake_snapshot_changes")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(snapshots, change_rows);
}

/// Author / message / extra-info round-trip through
/// `register_data_file_with_commit_metadata`.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn commit_metadata_is_persisted() {
    use datafusion_ducklake::SnapshotCommitMetadata;
    use datafusion_ducklake::metadata_writer::DataFileInfo;

    let (writer, pool, _conn, _tmp, _container) = setup().await.unwrap();
    let setup_result = writer
        .begin_write_transaction("main", "t", &cols(), WriteMode::Append)
        .unwrap();

    let meta = SnapshotCommitMetadata::new()
        .with_author("alice")
        .with_message("nightly load")
        .with_extra_info("job=42");

    let ids = writer
        .register_data_file_with_commit_metadata(
            setup_result.table_id,
            "main",
            "t",
            setup_result.snapshot_id,
            &DataFileInfo::new("t/data.parquet", 1024, 1),
            WriteMode::Append,
            setup_result.base_snapshot_id,
            &cols(),
            &setup_result.column_ids,
            &meta,
            None,
        )
        .unwrap();

    let (author, message, extra): (Option<String>, Option<String>, Option<String>) =
        sqlx::query_as(
            "SELECT author, commit_message, commit_extra_info
             FROM ducklake_snapshot_changes WHERE snapshot_id = $1",
        )
        .bind(ids.snapshot_id)
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(author.as_deref(), Some("alice"));
    assert_eq!(message.as_deref(), Some("nightly load"));
    assert_eq!(extra.as_deref(), Some("job=42"));
}

/// Conditional (compare-and-swap) writes are not implemented on this backend and
/// must fail closed rather than commit unconditionally.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn conditional_writes_are_rejected() {
    use datafusion_ducklake::SnapshotCommitMetadata;
    use datafusion_ducklake::metadata_writer::DataFileInfo;

    let (writer, _pool, _conn, _tmp, _container) = setup().await.unwrap();
    let setup_result = writer
        .begin_write_transaction("main", "t", &cols(), WriteMode::Append)
        .unwrap();

    let err = writer
        .register_data_file_with_commit_metadata(
            setup_result.table_id,
            "main",
            "t",
            setup_result.snapshot_id,
            &DataFileInfo::new("t/data.parquet", 1024, 1),
            WriteMode::Append,
            setup_result.base_snapshot_id,
            &cols(),
            &setup_result.column_ids,
            &SnapshotCommitMetadata::default(),
            Some(setup_result.base_snapshot_id),
        )
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("conditional writes are not supported"),
        "unexpected error: {err}"
    );

    let files_err = writer
        .register_data_files_with_commit_metadata(
            setup_result.table_id,
            "main",
            "t",
            setup_result.snapshot_id,
            &[DataFileInfo::new("t/data.parquet", 1024, 1)],
            WriteMode::Append,
            setup_result.base_snapshot_id,
            &cols(),
            &setup_result.column_ids,
            &SnapshotCommitMetadata::default(),
            Some(setup_result.base_snapshot_id),
        )
        .unwrap_err();
    assert!(
        files_err
            .to_string()
            .contains("conditional multi-file writes are not supported"),
        "unexpected error: {files_err}"
    );
}

/// Partition DDL is an ALTER and must say so in the change record.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn partition_ddl_records_altered_table() {
    use datafusion_ducklake::PartitionTransform;

    let (writer, pool, _conn, _tmp, _container) = setup().await.unwrap();
    let snapshot = writer.create_snapshot().unwrap();
    let (schema_id, _) = writer.get_or_create_schema("main", None, snapshot).unwrap();
    let (table_id, _) = writer
        .get_or_create_table(schema_id, "t", None, snapshot)
        .unwrap();
    writer.set_columns(table_id, &cols(), snapshot).unwrap();

    let ddl_snapshot = writer
        .set_partition_spec(
            table_id,
            &[("name".to_string(), PartitionTransform::Identity)],
        )
        .unwrap();

    let changes = changes_for(&pool, ddl_snapshot).await.unwrap();
    assert_eq!(changes, format!("altered_table:{table_id}"));
}

/// Opening a writer over an existing pool must not re-run DDL or disturb state.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn from_pool_shares_an_existing_connection_pool() {
    let (writer, pool, _conn, _tmp, _container) = setup().await.unwrap();
    DuckLakeTableWriter::new(Arc::new(writer), object_store())
        .unwrap()
        .write_table("main", "t", &[batch(vec![1], vec![Some("a")])])
        .await
        .unwrap();

    let adopted = PostgresSingleCatalogMetadataWriter::from_pool(pool.clone());
    assert_eq!(adopted.catalog_id(), None);
    let head: i64 = sqlx::query_scalar("SELECT MAX(snapshot_id) FROM ducklake_snapshot")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(head, 1);
}
