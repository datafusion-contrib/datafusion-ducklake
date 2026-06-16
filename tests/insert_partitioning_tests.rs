//! Regression tests for `INSERT INTO ... SELECT` with a multi-partition input.
//!
//! `DuckLakeInsertExec::execute` only drives `input.execute(0)`. Unless the
//! operator declares `required_input_distribution() == SinglePartition`,
//! DataFusion may hand it a multi-partition input (any parallelized scan,
//! aggregation, join, or `UNION`) and partitions `1..N` would be silently
//! dropped — losing rows with no error. These tests feed a two-partition
//! source and assert every row is written.

#![cfg(all(feature = "write-sqlite", feature = "metadata-sqlite"))]

use std::sync::Arc;

use arrow::array::{Int64Array, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::datasource::MemTable;
use datafusion::physical_plan::ExecutionPlanProperties;
use datafusion::prelude::*;
use tempfile::TempDir;

use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeTableWriter, MetadataWriter, SqliteMetadataProvider,
    SqliteMetadataWriter,
};

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]))
}

fn batch(ids: Vec<i64>) -> RecordBatch {
    RecordBatch::try_new(schema(), vec![Arc::new(Int64Array::from(ids))]).unwrap()
}

/// A `MemTable` source with two partitions: [1..=5] and [6..=10] (10 rows).
fn two_partition_source() -> Arc<MemTable> {
    Arc::new(
        MemTable::try_new(
            schema(),
            vec![vec![batch(vec![1, 2, 3, 4, 5])], vec![batch(vec![6, 7, 8, 9, 10])]],
        )
        .unwrap(),
    )
}

struct Harness {
    _temp_dir: TempDir,
    conn_str: String,
    data_path: String,
}

impl Harness {
    async fn new() -> Self {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let data_path = temp_dir.path().join("data");
        std::fs::create_dir_all(&data_path).unwrap();
        let conn_str = format!("sqlite:{}?mode=rwc", db_path.display());

        let writer = SqliteMetadataWriter::new_with_init(&conn_str)
            .await
            .unwrap();
        writer.set_data_path(data_path.to_str().unwrap()).unwrap();
        writer.create_snapshot().unwrap();

        Self {
            _temp_dir: temp_dir,
            conn_str,
            data_path: data_path.to_str().unwrap().to_string(),
        }
    }

    /// Create `main.<table>` seeded with the given ids via the direct writer.
    async fn seed_table(&self, table: &str, ids: Vec<i64>) {
        let writer = SqliteMetadataWriter::new(&self.conn_str).await.unwrap();
        let store: Arc<dyn object_store::ObjectStore> =
            Arc::new(object_store::local::LocalFileSystem::new());
        DuckLakeTableWriter::new(Arc::new(writer), store)
            .unwrap()
            .write_table("main", table, &[batch(ids)])
            .await
            .unwrap();
    }

    async fn writable_ctx(&self) -> SessionContext {
        let writer = SqliteMetadataWriter::new(&self.conn_str).await.unwrap();
        writer.set_data_path(&self.data_path).unwrap();
        let provider = SqliteMetadataProvider::new(&self.conn_str).await.unwrap();
        let catalog = DuckLakeCatalog::with_writer(Arc::new(provider), Arc::new(writer)).unwrap();
        let ctx = SessionContext::new();
        ctx.register_catalog("ducklake", Arc::new(catalog));
        ctx
    }

    /// Fresh read context so we observe the post-write snapshot rather than one
    /// cached at table registration.
    async fn count(&self, table: &str) -> i64 {
        let provider = SqliteMetadataProvider::new(&self.conn_str).await.unwrap();
        let catalog = DuckLakeCatalog::new(provider).unwrap();
        let ctx = SessionContext::new();
        ctx.register_catalog("ducklake", Arc::new(catalog));
        let b = ctx
            .sql(&format!("SELECT count(*) FROM ducklake.main.{table}"))
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        b[0].column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0)
    }
}

async fn run_dml_collect(ctx: &SessionContext, sql: &str) -> u64 {
    let out = ctx.sql(sql).await.unwrap().collect().await.unwrap();
    out[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap()
        .value(0)
}

#[tokio::test(flavor = "multi_thread")]
async fn insert_into_from_multipartition_source_keeps_all_rows() {
    let h = Harness::new().await;
    h.seed_table("t", vec![0]).await; // 1 seed row

    let ctx = h.writable_ctx().await;
    ctx.register_table("src", two_partition_source()).unwrap();

    // Sanity: the source really is multi-partition (otherwise the test is moot).
    let parts = ctx
        .sql("SELECT * FROM src")
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap()
        .output_partitioning()
        .partition_count();
    assert!(
        parts > 1,
        "source must have >1 partition to exercise the bug, got {parts}"
    );

    let reported = run_dml_collect(&ctx, "INSERT INTO ducklake.main.t SELECT * FROM src").await;
    assert_eq!(reported, 10, "INSERT must report all 10 input rows written");
    assert_eq!(
        h.count("t").await,
        11,
        "table must hold seed + all inserted rows"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn insert_overwrite_from_multipartition_source_keeps_all_rows() {
    let h = Harness::new().await;
    h.seed_table("t", vec![0]).await; // seed to be replaced

    let ctx = h.writable_ctx().await;
    ctx.register_table("src", two_partition_source()).unwrap();

    let reported =
        run_dml_collect(&ctx, "INSERT OVERWRITE ducklake.main.t SELECT * FROM src").await;
    assert_eq!(
        reported, 10,
        "INSERT OVERWRITE must report all 10 input rows written"
    );
    assert_eq!(
        h.count("t").await,
        10,
        "overwrite must replace seed with all 10 rows"
    );
}
