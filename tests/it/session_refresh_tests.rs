//! Writable catalog snapshot refresh behavior.

#![cfg(all(feature = "write-sqlite", feature = "metadata-sqlite"))]

use std::sync::Arc;

use arrow::array::{Int32Array, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::prelude::SessionContext;
use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeTableWriter, MetadataWriter, SqliteMetadataProvider,
    SqliteMetadataWriter,
};
use object_store::local::LocalFileSystem;
use tempfile::TempDir;

fn batch(rows: &[(i32, i32)]) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("value", DataType::Int32, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(
                rows.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            )),
            Arc::new(Int32Array::from(
                rows.iter().map(|(_, value)| *value).collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap()
}

async fn setup() -> (TempDir, String, i64) {
    let temp = TempDir::new().unwrap();
    let database = temp.path().join("catalog.db");
    let data = temp.path().join("data");
    std::fs::create_dir(&data).unwrap();
    let connection = format!("sqlite:{}?mode=rwc", database.display());
    let writer = Arc::new(
        SqliteMetadataWriter::new_with_init(&connection)
            .await
            .unwrap(),
    );
    writer.set_data_path(data.to_str().unwrap()).unwrap();
    let result = DuckLakeTableWriter::new(writer, Arc::new(LocalFileSystem::new()))
        .unwrap()
        .write_table("main", "t", &[batch(&[(1, 10)])])
        .await
        .unwrap();
    (temp, connection, result.snapshot_id)
}

async fn writable_context(connection: &str) -> SessionContext {
    let provider = SqliteMetadataProvider::new(connection).await.unwrap();
    let writer = SqliteMetadataWriter::new(connection).await.unwrap();
    let catalog = DuckLakeCatalog::with_writer(Arc::new(provider), Arc::new(writer)).unwrap();
    let context = SessionContext::new();
    context.register_catalog("lake", Arc::new(catalog));
    context
}

async fn fixed_context(connection: &str, snapshot_id: i64) -> SessionContext {
    let provider = Arc::new(SqliteMetadataProvider::new(connection).await.unwrap());
    let catalog = DuckLakeCatalog::with_snapshot(provider, snapshot_id).unwrap();
    let context = SessionContext::new();
    context.register_catalog("lake", Arc::new(catalog));
    context
}

async fn rows(context: &SessionContext) -> Vec<(i32, i32)> {
    let batches = context
        .sql("SELECT id, value FROM lake.main.t ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let mut rows = Vec::new();
    for batch in batches {
        let ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let values = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        rows.extend((0..batch.num_rows()).map(|index| (ids.value(index), values.value(index))));
    }
    rows
}

async fn affected(context: &SessionContext, sql: &str) -> u64 {
    let batches = context.sql(sql).await.unwrap().collect().await.unwrap();
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 1);
    batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap()
        .value(0)
}

#[tokio::test(flavor = "multi_thread")]
async fn own_commits_refresh_insert_update_and_delete_reads() {
    let (_temp, connection, _initial_snapshot) = setup().await;
    let context = writable_context(&connection).await;

    assert_eq!(rows(&context).await, vec![(1, 10)]);
    assert_eq!(
        affected(
            &context,
            "INSERT INTO lake.main.t (id, value) VALUES (2, 20)"
        )
        .await,
        1
    );
    assert_eq!(rows(&context).await, vec![(1, 10), (2, 20)]);

    assert_eq!(
        affected(&context, "DELETE FROM lake.main.t WHERE id = 1").await,
        1
    );
    assert_eq!(rows(&context).await, vec![(2, 20)]);

    assert_eq!(
        affected(&context, "UPDATE lake.main.t SET value = 21 WHERE id = 2").await,
        1
    );
    assert_eq!(rows(&context).await, vec![(2, 21)]);

    assert_eq!(
        affected(&context, "UPDATE lake.main.t SET value = 22 WHERE id = 2").await,
        1
    );
    assert_eq!(rows(&context).await, vec![(2, 22)]);

    assert_eq!(affected(&context, "DELETE FROM lake.main.t").await, 1);
    assert_eq!(rows(&context).await, Vec::<(i32, i32)>::new());
}

#[tokio::test(flavor = "multi_thread")]
async fn external_commit_does_not_refresh_writable_or_explicit_pin() {
    let (_temp, connection, initial_snapshot) = setup().await;
    let writable = writable_context(&connection).await;
    let fixed = fixed_context(&connection, initial_snapshot).await;

    let external_writer = Arc::new(SqliteMetadataWriter::new(&connection).await.unwrap());
    DuckLakeTableWriter::new(external_writer, Arc::new(LocalFileSystem::new()))
        .unwrap()
        .append_table("main", "t", &[batch(&[(2, 20)])])
        .await
        .unwrap();

    assert_eq!(rows(&writable).await, vec![(1, 10)]);
    assert_eq!(rows(&fixed).await, vec![(1, 10)]);

    assert_eq!(
        affected(
            &writable,
            "INSERT INTO lake.main.t (id, value) VALUES (3, 30)"
        )
        .await,
        1
    );
    assert_eq!(rows(&writable).await, vec![(1, 10), (2, 20), (3, 30)]);
    assert_eq!(rows(&fixed).await, vec![(1, 10)]);
}
