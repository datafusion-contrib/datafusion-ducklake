//! Data paths on object stores other than S3.
//!
//! A catalog whose `data_path` is `scheme://bucket/prefix/` must write, read,
//! update and delete through the object store registered in DataFusion's
//! `RuntimeEnv` for `scheme://bucket/`, for any scheme (`gs://`, `az://`, ...).
//! Each case registers an in-memory store under the scheme, so no cloud
//! service is needed.

#![cfg(all(feature = "write-sqlite", feature = "metadata-sqlite"))]

use std::sync::Arc;

use arrow::array::{Array, Int32Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::prelude::*;
use futures::TryStreamExt;
use object_store::ObjectStore;
use object_store::memory::InMemory;
use rstest::rstest;
use tempfile::TempDir;

use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeTableWriter, MetadataWriter, SqliteMetadataProvider,
    SqliteMetadataWriter,
};

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("val", DataType::Int32, false),
    ]))
}

async fn ids_and_vals(ctx: &SessionContext) -> Vec<(i32, i32)> {
    let batches = ctx
        .sql("SELECT id, val FROM ducklake.main.t ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let mut out = Vec::new();
    for b in &batches {
        let ids = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        let vals = b.column(1).as_any().downcast_ref::<Int32Array>().unwrap();
        for i in 0..b.num_rows() {
            out.push((ids.value(i), vals.value(i)));
        }
    }
    out
}

#[rstest]
#[case::gcs("gs")]
#[case::gcs_long("gcs")]
#[case::azure("az")]
#[case::azure_dfs("abfss")]
#[case::upper_case_scheme("GS")]
#[tokio::test]
async fn data_path_on_any_object_store_scheme(#[case] scheme: &str) {
    let temp = TempDir::new().unwrap();
    let conn = format!("sqlite:{}?mode=rwc", temp.path().join("c.db").display());
    let data_path = format!("{scheme}://lake/warehouse/");
    let store = Arc::new(InMemory::new());

    let writer = Arc::new(SqliteMetadataWriter::new_with_init(&conn).await.unwrap());
    writer.set_data_path(&data_path).unwrap();
    let batch = RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(Int32Array::from(vec![10, 20, 30])),
        ],
    )
    .unwrap();
    DuckLakeTableWriter::new(writer.clone(), store.clone())
        .unwrap()
        .write_table("main", "t", &[batch])
        .await
        .unwrap();

    // Reads and DML find the files through the runtime's store for the
    // scheme.
    let runtime = Arc::new(RuntimeEnvBuilder::new().build().unwrap());
    let bucket = url::Url::parse(&format!("{}://lake", scheme.to_ascii_lowercase())).unwrap();
    runtime.register_object_store(&bucket, store.clone());
    let ctx = async || {
        let provider = SqliteMetadataProvider::new(&conn).await.unwrap();
        let writer = SqliteMetadataWriter::new(&conn).await.unwrap();
        let catalog = DuckLakeCatalog::with_writer(Arc::new(provider), Arc::new(writer)).unwrap();
        let ctx = SessionContext::new_with_config_rt(SessionConfig::new(), runtime.clone());
        ctx.register_catalog("ducklake", Arc::new(catalog));
        ctx
    };

    assert_eq!(
        ids_and_vals(&ctx().await).await,
        vec![(1, 10), (2, 20), (3, 30)]
    );
    for sql in [
        "UPDATE ducklake.main.t SET val = 21 WHERE id = 2",
        "DELETE FROM ducklake.main.t WHERE id = 1",
        "INSERT INTO ducklake.main.t VALUES (4, 40)",
    ] {
        ctx().await.sql(sql).await.unwrap().collect().await.unwrap();
    }
    assert_eq!(
        ids_and_vals(&ctx().await).await,
        vec![(2, 21), (3, 30), (4, 40)]
    );

    // Every file (data and delete) is in the store, under the prefix.
    let paths: Vec<String> = store
        .list(None)
        .map_ok(|m| m.location.to_string())
        .try_collect()
        .await
        .unwrap();
    assert!(paths.len() >= 4, "{paths:?}");
    assert!(
        paths.iter().all(|p| p.starts_with("warehouse/")),
        "{paths:?}"
    );
}
