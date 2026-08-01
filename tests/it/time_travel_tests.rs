//! Snapshot selection by version and timestamp.

#![cfg(all(feature = "write-sqlite", feature = "metadata-sqlite"))]

use std::sync::Arc;

use arrow::array::Int32Array;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use chrono::{TimeZone, Utc};
use datafusion::prelude::SessionContext;
use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeTableWriter, SqliteMetadataProvider, SqliteMetadataWriter,
    register_ducklake_functions,
};
use object_store::local::LocalFileSystem;
use sqlx::SqlitePool;
use tempfile::TempDir;

fn id_batch(ids: &[i32]) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(ids.to_vec()))]).unwrap()
}

async fn ids(context: &SessionContext, sql: &str) -> Vec<i32> {
    let batches = context.sql(sql).await.unwrap().collect().await.unwrap();
    let mut values = Vec::new();
    for batch in batches {
        let array = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        values.extend((0..array.len()).map(|index| array.value(index)));
    }
    values
}

#[tokio::test(flavor = "multi_thread")]
async fn version_timestamp_and_same_second_cdc_select_snapshots() {
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
    datafusion_ducklake::MetadataWriter::set_data_path(writer.as_ref(), data.to_str().unwrap())
        .unwrap();
    let initial = DuckLakeTableWriter::new(
        writer.clone(),
        Arc::new(LocalFileSystem::new()) as Arc<dyn object_store::ObjectStore>,
    )
    .unwrap()
    .write_table("main", "t", &[id_batch(&[1, 2, 3])])
    .await
    .unwrap();

    let provider = SqliteMetadataProvider::new(&connection).await.unwrap();
    let writable = DuckLakeCatalog::with_writer(Arc::new(provider), writer).unwrap();
    let write_context = SessionContext::new();
    write_context.register_catalog("lake", Arc::new(writable));
    write_context
        .sql("DELETE FROM lake.main.t WHERE id = 2")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    let pool = SqlitePool::connect(&connection).await.unwrap();
    let delete_snapshot: i64 = sqlx::query_scalar("SELECT MAX(snapshot_id) FROM ducklake_snapshot")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert!(delete_snapshot > initial.snapshot_id);
    sqlx::query("UPDATE ducklake_snapshot SET snapshot_time = ? WHERE snapshot_id = ?")
        .bind("2026-08-01 12:00:00.000000")
        .bind(initial.snapshot_id)
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query("UPDATE ducklake_snapshot SET snapshot_time = ? WHERE snapshot_id = ?")
        .bind("2026-08-01 13:00:00.000000")
        .bind(delete_snapshot)
        .execute(&pool)
        .await
        .unwrap();
    pool.close().await;

    let current_provider = SqliteMetadataProvider::new(&connection).await.unwrap();
    let current = SessionContext::new();
    current.register_catalog(
        "lake",
        Arc::new(DuckLakeCatalog::new(current_provider).unwrap()),
    );
    assert_eq!(
        ids(&current, "SELECT id FROM lake.main.t ORDER BY id").await,
        vec![1, 3]
    );

    let timestamp = Utc
        .with_ymd_and_hms(2026, 8, 1, 12, 30, 0)
        .single()
        .unwrap();
    let timestamp_provider = Arc::new(SqliteMetadataProvider::new(&connection).await.unwrap());
    let timestamp_catalog =
        DuckLakeCatalog::with_snapshot_at(timestamp_provider, timestamp).unwrap();
    let timestamp_context = SessionContext::new();
    timestamp_context.register_catalog("past", Arc::new(timestamp_catalog));
    assert_eq!(
        ids(&timestamp_context, "SELECT id FROM past.main.t ORDER BY id").await,
        vec![1, 2, 3]
    );
    let invalid_catalog_provider =
        Arc::new(SqliteMetadataProvider::new(&connection).await.unwrap());
    let before_first = Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).single().unwrap();
    assert!(
        DuckLakeCatalog::with_snapshot_at(invalid_catalog_provider, before_first)
            .unwrap_err()
            .to_string()
            .contains("No snapshot found at or before timestamp")
    );

    let pool = SqlitePool::connect(&connection).await.unwrap();
    sqlx::query("UPDATE ducklake_snapshot SET snapshot_time = ? WHERE snapshot_id = ?")
        .bind("2026-08-01 12:00:00.000000")
        .bind(delete_snapshot)
        .execute(&pool)
        .await
        .unwrap();
    pool.close().await;

    let function_provider = Arc::new(SqliteMetadataProvider::new(&connection).await.unwrap());
    let function_context = SessionContext::new();
    register_ducklake_functions(&function_context, function_provider);
    let by_version = ids(
        &function_context,
        &format!(
            "SELECT id FROM ducklake_table_at('main', 't', {}) ORDER BY id",
            initial.snapshot_id
        ),
    )
    .await;
    let by_timestamp = ids(
        &function_context,
        "SELECT id FROM ducklake_table_at(\
         'main', 't', TIMESTAMP '2026-08-01 12:00:00') ORDER BY id",
    )
    .await;
    let deleted_through_timestamp = ids(
        &function_context,
        &format!(
            "SELECT id FROM ducklake_table_deletions(\
             'main.t', {}, TIMESTAMP '2026-08-01 12:00:00') ORDER BY id",
            initial.snapshot_id
        ),
    )
    .await;
    assert_eq!(by_version, vec![1, 2, 3]);
    assert_eq!(by_timestamp, vec![1, 3]);
    assert_eq!(deleted_through_timestamp, vec![2]);

    let invalid_version = function_context
        .sql("SELECT * FROM ducklake_table_at('main', 't', 999999)")
        .await
        .unwrap_err()
        .to_string();
    assert!(invalid_version.contains("Snapshot 999999 does not exist"));
    let invalid_timestamp = function_context
        .sql(
            "SELECT * FROM ducklake_table_at(\
             'main', 't', TIMESTAMP '2020-01-01 00:00:00')",
        )
        .await
        .unwrap_err()
        .to_string();
    assert!(invalid_timestamp.contains("No snapshot found at or before timestamp"));
}
