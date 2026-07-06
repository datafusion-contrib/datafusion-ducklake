//! Round-trip tests for the positional-delete write path (#864 / #862):
//! `MetadataWriter::set_delete_file` registers a positional `(file_path, pos)`
//! delete file, and a subsequent read applies it via `DeleteFilterExec`. These
//! validate the fenced, cumulative, ≤1-live-per-data-file write end-to-end
//! through the SQLite backend (the one the crate's tests can run without a
//! container), asserting surviving VALUES — a positional bug silently deletes
//! the wrong rows, so value assertions are the point.

#![cfg(all(feature = "write-sqlite", feature = "metadata-sqlite"))]

use std::sync::Arc;

use arrow::array::{Array, Int32Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::prelude::*;
use object_store::local::LocalFileSystem;
use parquet::arrow::ArrowWriter;
use tempfile::TempDir;

use datafusion_ducklake::{
    DeleteFileInfo, DuckLakeCatalog, DuckLakeTableWriter, MetadataWriter, SqliteMetadataProvider,
    SqliteMetadataWriter,
};
use sqlx::Row;
use sqlx::sqlite::SqlitePool;

/// A writable SQLite-backed catalog + a data dir, in a temp dir.
async fn create_writer(temp_dir: &TempDir) -> SqliteMetadataWriter {
    let db_path = temp_dir.path().join("test.db");
    let data_path = temp_dir.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();
    let conn_str = format!("sqlite:{}?mode=rwc", db_path.display());
    let writer = SqliteMetadataWriter::new_with_init(&conn_str)
        .await
        .unwrap();
    writer.set_data_path(data_path.to_str().unwrap()).unwrap();
    writer
}

/// Read `id`s from `test.main.t`, ascending, through the full read path (which
/// applies any live delete file).
async fn read_ids(temp_dir: &TempDir) -> Vec<i32> {
    let db_path = temp_dir.path().join("test.db");
    let conn_str = format!("sqlite:{}", db_path.display());
    let provider = SqliteMetadataProvider::new(&conn_str).await.unwrap();
    let catalog = DuckLakeCatalog::new(provider).unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("test", Arc::new(catalog));
    let batches = ctx
        .sql("SELECT id FROM test.main.t ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let mut ids = Vec::new();
    for b in &batches {
        let col = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        for i in 0..b.num_rows() {
            ids.push(col.value(i));
        }
    }
    ids
}

/// Write a positional delete parquet `(file_path VARCHAR, pos BIGINT)` — the
/// DuckLake standard delete-file schema — and return its byte size. Only `pos`
/// is read back; `file_path` is documentation.
fn write_delete_parquet(path: &std::path::Path, data_file_path: &str, positions: &[i64]) -> i64 {
    let schema = Arc::new(Schema::new(vec![
        Field::new("file_path", DataType::Utf8, false),
        Field::new("pos", DataType::Int64, false),
    ]));
    let file_paths = StringArray::from(vec![data_file_path; positions.len()]);
    let pos = Int64Array::from(positions.to_vec());
    let batch =
        RecordBatch::try_new(schema.clone(), vec![Arc::new(file_paths), Arc::new(pos)]).unwrap();
    let file = std::fs::File::create(path).unwrap();
    let mut w = ArrowWriter::try_new(file, schema, None).unwrap();
    w.write(&batch).unwrap();
    w.close().unwrap();
    std::fs::metadata(path).unwrap().len() as i64
}

#[tokio::test(flavor = "multi_thread")]
async fn set_delete_file_positional_delete_roundtrip() {
    let temp_dir = TempDir::new().unwrap();
    let writer = Arc::new(create_writer(&temp_dir).await);
    let object_store: Arc<dyn object_store::ObjectStore> = Arc::new(LocalFileSystem::new());

    // Write ids [1,2,3,4] as one insert-only data file (physical positions 0..3).
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let batch =
        RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![1, 2, 3, 4]))]).unwrap();
    DuckLakeTableWriter::new(writer.clone(), object_store.clone())
        .unwrap()
        .write_table("main", "t", &[batch])
        .await
        .unwrap();
    assert_eq!(read_ids(&temp_dir).await, vec![1, 2, 3, 4], "baseline");

    // Resolve the catalog ids for the freshly-written data file.
    let db_path = temp_dir.path().join("test.db");
    let pool = SqlitePool::connect(&format!("sqlite:{}", db_path.display()))
        .await
        .unwrap();
    let table_id: i64 =
        sqlx::query_scalar("SELECT table_id FROM ducklake_table WHERE end_snapshot IS NULL")
            .fetch_one(&pool)
            .await
            .unwrap();
    let df_row = sqlx::query(
        "SELECT data_file_id, path FROM ducklake_data_file
         WHERE table_id = ? AND end_snapshot IS NULL",
    )
    .bind(table_id)
    .fetch_one(&pool)
    .await
    .unwrap();
    let data_file_id: i64 = df_row.try_get(0).unwrap();
    let data_file_path: String = df_row.try_get(1).unwrap();
    let base1: i64 = sqlx::query_scalar("SELECT MAX(snapshot_id) FROM ducklake_snapshot")
        .fetch_one(&pool)
        .await
        .unwrap();

    // Delete physical positions {1, 3} → ids 2 and 4. First delete: no prior.
    let del1 = temp_dir.path().join("delete1.parquet");
    let size1 = write_delete_parquet(&del1, &data_file_path, &[1, 3]);
    let info1 =
        DeleteFileInfo::new(del1.to_string_lossy().to_string(), size1, 2).with_absolute_path();
    writer
        .set_delete_file(
            table_id,
            "main",
            "t",
            base1,
            data_file_id,
            None,
            base1,
            &info1,
        )
        .unwrap();
    assert_eq!(
        read_ids(&temp_dir).await,
        vec![1, 3],
        "positions 1,3 deleted (ids 2,4)"
    );

    // Cumulative second delete: supersede the first, deleting {1, 2, 3} → ids
    // 2, 3, 4. The CAS must see the first delete file as the live prior.
    let prev: i64 = sqlx::query_scalar(
        "SELECT delete_file_id FROM ducklake_delete_file
         WHERE data_file_id = ? AND end_snapshot IS NULL",
    )
    .bind(data_file_id)
    .fetch_one(&pool)
    .await
    .unwrap();
    let base2: i64 = sqlx::query_scalar("SELECT MAX(snapshot_id) FROM ducklake_snapshot")
        .fetch_one(&pool)
        .await
        .unwrap();
    let del2 = temp_dir.path().join("delete2.parquet");
    let size2 = write_delete_parquet(&del2, &data_file_path, &[1, 2, 3]);
    let info2 =
        DeleteFileInfo::new(del2.to_string_lossy().to_string(), size2, 3).with_absolute_path();
    writer
        .set_delete_file(
            table_id,
            "main",
            "t",
            base2,
            data_file_id,
            Some(prev),
            base2,
            &info2,
        )
        .unwrap();
    assert_eq!(
        read_ids(&temp_dir).await,
        vec![1],
        "cumulative delete of positions 1,2,3 (ids 2,3,4)"
    );

    // Exactly one delete file is live for the data file (the prior was retired).
    let live: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM ducklake_delete_file
         WHERE data_file_id = ? AND end_snapshot IS NULL",
    )
    .bind(data_file_id)
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(live, 1, "at most one live delete file per data file");
}

#[tokio::test(flavor = "multi_thread")]
async fn set_delete_file_rejects_stale_prior() {
    // The compare-and-swap must reject a write whose `expected_prev_delete_file`
    // doesn't match the live delete file (a concurrent delete won).
    let temp_dir = TempDir::new().unwrap();
    let writer = Arc::new(create_writer(&temp_dir).await);
    let object_store: Arc<dyn object_store::ObjectStore> = Arc::new(LocalFileSystem::new());

    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let batch =
        RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![1, 2, 3, 4]))]).unwrap();
    DuckLakeTableWriter::new(writer.clone(), object_store.clone())
        .unwrap()
        .write_table("main", "t", &[batch])
        .await
        .unwrap();

    let db_path = temp_dir.path().join("test.db");
    let pool = SqlitePool::connect(&format!("sqlite:{}", db_path.display()))
        .await
        .unwrap();
    let table_id: i64 =
        sqlx::query_scalar("SELECT table_id FROM ducklake_table WHERE end_snapshot IS NULL")
            .fetch_one(&pool)
            .await
            .unwrap();
    let df_row = sqlx::query(
        "SELECT data_file_id, path FROM ducklake_data_file
         WHERE table_id = ? AND end_snapshot IS NULL",
    )
    .bind(table_id)
    .fetch_one(&pool)
    .await
    .unwrap();
    let data_file_id: i64 = df_row.try_get(0).unwrap();
    let data_file_path: String = df_row.try_get(1).unwrap();
    let base: i64 = sqlx::query_scalar("SELECT MAX(snapshot_id) FROM ducklake_snapshot")
        .fetch_one(&pool)
        .await
        .unwrap();

    // Establish a live delete file (prior = None).
    let del1 = temp_dir.path().join("delete1.parquet");
    let size1 = write_delete_parquet(&del1, &data_file_path, &[1]);
    let info1 =
        DeleteFileInfo::new(del1.to_string_lossy().to_string(), size1, 1).with_absolute_path();
    writer
        .set_delete_file(
            table_id,
            "main",
            "t",
            base,
            data_file_id,
            None,
            base,
            &info1,
        )
        .unwrap();

    // A write that still thinks there's no prior delete file must be rejected.
    let del2 = temp_dir.path().join("delete2.parquet");
    let size2 = write_delete_parquet(&del2, &data_file_path, &[1, 2]);
    let info2 =
        DeleteFileInfo::new(del2.to_string_lossy().to_string(), size2, 2).with_absolute_path();
    let base2: i64 = sqlx::query_scalar("SELECT MAX(snapshot_id) FROM ducklake_snapshot")
        .fetch_one(&pool)
        .await
        .unwrap();
    let err = writer
        .set_delete_file(
            table_id,
            "main",
            "t",
            base2,
            data_file_id,
            None,
            base2,
            &info2,
        )
        .expect_err("stale expected_prev_delete_file must be rejected");
    assert!(
        matches!(err, datafusion_ducklake::DuckLakeError::Conflict(_)),
        "expected a Conflict, got {err:?}"
    );
}
