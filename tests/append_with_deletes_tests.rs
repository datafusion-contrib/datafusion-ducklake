//! Round-trip tests for the combined append-with-deletes write path:
//! `MetadataWriter::register_data_file_with_deletes` (driven via
//! `TableWriteSession::finish_with_deletes`) registers a new data file AND
//! positional delete files for existing data files in ONE snapshot — the commit
//! primitive behind an update/upsert (supersede rows, insert their new versions,
//! atomically). These validate the atomic single-snapshot behaviour and the
//! resulting VALUES end-to-end through the SQLite backend, since a bug here
//! either half-applies the mutation or updates the wrong rows.

#![cfg(all(feature = "write-sqlite", feature = "metadata-sqlite"))]

use std::sync::Arc;

use arrow::array::{Array, Int32Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::prelude::*;
use object_store::local::LocalFileSystem;
use tempfile::TempDir;

use datafusion::logical_expr::Operator;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_expr::expressions::{BinaryExpr, col, lit};
use datafusion_ducklake::{
    DataFileInfo, DeleteFileEntry, DeleteFileInfo, DuckLakeCatalog, DuckLakeError,
    DuckLakeFileData, DuckLakeTable, DuckLakeTableWriter, MetadataWriter, SqliteMetadataProvider,
    SqliteMetadataWriter, WriteMode,
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

/// Read `(id, val)` from `test.main.t`, ascending by `id`, through the full read
/// path (which applies any live delete file).
async fn read_pairs(temp_dir: &TempDir) -> Vec<(i32, i32)> {
    let db_path = temp_dir.path().join("test.db");
    let conn_str = format!("sqlite:{}", db_path.display());
    let provider = SqliteMetadataProvider::new(&conn_str).await.unwrap();
    let catalog = DuckLakeCatalog::new(provider).unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("test", Arc::new(catalog));
    let batches = ctx
        .sql("SELECT id, val FROM test.main.t ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let mut rows = Vec::new();
    for b in &batches {
        let ids = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        let vals = b.column(1).as_any().downcast_ref::<Int32Array>().unwrap();
        for i in 0..b.num_rows() {
            rows.push((ids.value(i), vals.value(i)));
        }
    }
    rows
}

/// The `(id, val)` table schema used throughout.
fn table_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("val", DataType::Int32, false),
    ]))
}

/// Resolve the physical positions of rows matching `id == wanted` within
/// `data_file`, via the crate's `resolve_positions`.
async fn positions_for_id(conn_str: &str, data_file: &DuckLakeFileData, wanted: i32) -> Vec<i64> {
    let provider = SqliteMetadataProvider::new(conn_str).await.unwrap();
    let catalog = DuckLakeCatalog::new(provider).unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("test", Arc::new(catalog));
    let table_provider = ctx
        .catalog("test")
        .unwrap()
        .schema("main")
        .unwrap()
        .table("t")
        .await
        .unwrap()
        .unwrap();
    let table = (table_provider.as_ref() as &dyn std::any::Any)
        .downcast_ref::<DuckLakeTable>()
        .expect("provider is a DuckLakeTable");
    let data_schema = table_schema();
    let id: Arc<dyn PhysicalExpr> = col("id", data_schema.as_ref()).unwrap();
    let predicate: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(id, Operator::Eq, lit(wanted)));
    let state = ctx.state();
    let mut positions: Vec<i64> = table
        .resolve_positions(&state, data_file, predicate)
        .await
        .unwrap()
        .into_iter()
        .collect();
    positions.sort_unstable();
    positions
}

/// The live data files for `table_id`, in insertion order (ascending
/// `data_file_id`), each as `(data_file_id, DuckLakeFileData)` ready to scan.
async fn live_data_files(pool: &SqlitePool, table_id: i64) -> Vec<(i64, DuckLakeFileData)> {
    let rows = sqlx::query(
        "SELECT data_file_id, path, path_is_relative, file_size_bytes
         FROM ducklake_data_file
         WHERE table_id = ? AND end_snapshot IS NULL
         ORDER BY data_file_id",
    )
    .bind(table_id)
    .fetch_all(pool)
    .await
    .unwrap();
    rows.into_iter()
        .map(|r| {
            let id: i64 = r.try_get(0).unwrap();
            let path: String = r.try_get(1).unwrap();
            let rel: bool = r.try_get::<i64, _>(2).unwrap() != 0;
            let size: i64 = r.try_get(3).unwrap();
            (id, DuckLakeFileData::new(path, rel, size))
        })
        .collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn update_via_finish_with_deletes_is_one_atomic_snapshot() {
    let temp_dir = TempDir::new().unwrap();
    let writer = Arc::new(create_writer(&temp_dir).await);
    let object_store: Arc<dyn object_store::ObjectStore> = Arc::new(LocalFileSystem::new());
    let schema = table_schema();

    // Seed (id, val): (1,10),(2,20),(3,30),(4,40) as one data file.
    let seed = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4])),
            Arc::new(Int32Array::from(vec![10, 20, 30, 40])),
        ],
    )
    .unwrap();
    DuckLakeTableWriter::new(writer.clone(), object_store.clone())
        .unwrap()
        .write_table("main", "t", &[seed])
        .await
        .unwrap();
    assert_eq!(
        read_pairs(&temp_dir).await,
        vec![(1, 10), (2, 20), (3, 30), (4, 40)],
        "baseline"
    );

    let db_path = temp_dir.path().join("test.db");
    let conn_str = format!("sqlite:{}", db_path.display());
    let pool = SqlitePool::connect(&conn_str).await.unwrap();
    let table_id: i64 =
        sqlx::query_scalar("SELECT table_id FROM ducklake_table WHERE end_snapshot IS NULL")
            .fetch_one(&pool)
            .await
            .unwrap();
    let files = live_data_files(&pool, table_id).await;
    assert_eq!(files.len(), 1);
    let (data_file_id, data_file) = files.into_iter().next().unwrap();

    // Update ids {2, 4}: resolve their positions (1 and 3) and author one
    // cumulative delete file for the seed data file.
    let mut positions = positions_for_id(&conn_str, &data_file, 2).await;
    positions.extend(positions_for_id(&conn_str, &data_file, 4).await);
    positions.sort_unstable();
    assert_eq!(positions, vec![1, 3], "ids 2,4 sit at positions 1,3");
    let del_info = DuckLakeTableWriter::new(writer.clone(), object_store.clone())
        .unwrap()
        .write_delete_file("main", "t", &data_file.path, &positions)
        .await
        .unwrap();

    // Append the NEW versions (2,200),(4,400) and commit them together with the
    // delete in ONE snapshot.
    let new_versions = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(vec![2, 4])), Arc::new(Int32Array::from(vec![200, 400]))],
    )
    .unwrap();
    let mut session = DuckLakeTableWriter::new(writer.clone(), object_store.clone())
        .unwrap()
        .begin_write("main", "t", schema.as_ref(), WriteMode::Append)
        .unwrap();
    session.write_batch(&new_versions).unwrap();
    let entries = vec![DeleteFileEntry {
        data_file_id,
        expected_prev_delete_file: None,
        delete: del_info,
    }];
    let result = session.finish_with_deletes(&entries).await.unwrap();

    // Old versions of 2,4 are gone; the new versions are present; 1,3 untouched.
    assert_eq!(
        read_pairs(&temp_dir).await,
        vec![(1, 10), (2, 200), (3, 30), (4, 400)],
        "rows 2,4 updated in place; 1,3 unchanged"
    );

    // Atomicity: the delete file and the appended data file carry the SAME
    // begin_snapshot — the committed head — so they became visible together.
    let delete_snap: i64 = sqlx::query_scalar(
        "SELECT begin_snapshot FROM ducklake_delete_file
         WHERE data_file_id = ? AND end_snapshot IS NULL",
    )
    .bind(data_file_id)
    .fetch_one(&pool)
    .await
    .unwrap();
    let appended_snap: i64 = sqlx::query_scalar(
        "SELECT begin_snapshot FROM ducklake_data_file
         WHERE table_id = ? AND data_file_id <> ? AND end_snapshot IS NULL",
    )
    .bind(table_id)
    .bind(data_file_id)
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(
        delete_snap, appended_snap,
        "delete file and appended data file share one snapshot"
    );
    assert_eq!(
        delete_snap, result.snapshot_id,
        "that shared snapshot is the committed head"
    );

    // Exactly one delete file is live for the seed file.
    let live: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM ducklake_delete_file
         WHERE data_file_id = ? AND end_snapshot IS NULL",
    )
    .bind(data_file_id)
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(live, 1, "one live delete file for the seed data file");
}

#[tokio::test(flavor = "multi_thread")]
async fn update_spanning_two_data_files_commits_one_snapshot() {
    let temp_dir = TempDir::new().unwrap();
    let writer = Arc::new(create_writer(&temp_dir).await);
    let object_store: Arc<dyn object_store::ObjectStore> = Arc::new(LocalFileSystem::new());
    let schema = table_schema();

    // Two data files: A = (1,10),(2,20); B = (3,30),(4,40).
    let file_a = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(vec![1, 2])), Arc::new(Int32Array::from(vec![10, 20]))],
    )
    .unwrap();
    DuckLakeTableWriter::new(writer.clone(), object_store.clone())
        .unwrap()
        .write_table("main", "t", &[file_a])
        .await
        .unwrap();
    let file_b = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(vec![3, 4])), Arc::new(Int32Array::from(vec![30, 40]))],
    )
    .unwrap();
    DuckLakeTableWriter::new(writer.clone(), object_store.clone())
        .unwrap()
        .append_table("main", "t", &[file_b])
        .await
        .unwrap();
    assert_eq!(
        read_pairs(&temp_dir).await,
        vec![(1, 10), (2, 20), (3, 30), (4, 40)],
        "baseline across two files"
    );

    let db_path = temp_dir.path().join("test.db");
    let conn_str = format!("sqlite:{}", db_path.display());
    let pool = SqlitePool::connect(&conn_str).await.unwrap();
    let table_id: i64 =
        sqlx::query_scalar("SELECT table_id FROM ducklake_table WHERE end_snapshot IS NULL")
            .fetch_one(&pool)
            .await
            .unwrap();
    let files = live_data_files(&pool, table_id).await;
    assert_eq!(files.len(), 2, "two live data files");
    let (file_a_id, file_a_data) = files[0].clone();
    let (file_b_id, file_b_data) = files[1].clone();

    // Update id 2 (in file A) and id 3 (in file B): one delete entry per file,
    // one appended data file with both new versions — all in one commit.
    let pos_a = positions_for_id(&conn_str, &file_a_data, 2).await;
    assert_eq!(pos_a, vec![1], "id 2 is at position 1 in file A");
    let pos_b = positions_for_id(&conn_str, &file_b_data, 3).await;
    assert_eq!(pos_b, vec![0], "id 3 is at position 0 in file B");
    let del_a = DuckLakeTableWriter::new(writer.clone(), object_store.clone())
        .unwrap()
        .write_delete_file("main", "t", &file_a_data.path, &pos_a)
        .await
        .unwrap();
    let del_b = DuckLakeTableWriter::new(writer.clone(), object_store.clone())
        .unwrap()
        .write_delete_file("main", "t", &file_b_data.path, &pos_b)
        .await
        .unwrap();

    let new_versions = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(vec![2, 3])), Arc::new(Int32Array::from(vec![200, 300]))],
    )
    .unwrap();
    let mut session = DuckLakeTableWriter::new(writer.clone(), object_store.clone())
        .unwrap()
        .begin_write("main", "t", schema.as_ref(), WriteMode::Append)
        .unwrap();
    session.write_batch(&new_versions).unwrap();
    let entries = vec![
        DeleteFileEntry {
            data_file_id: file_a_id,
            expected_prev_delete_file: None,
            delete: del_a,
        },
        DeleteFileEntry {
            data_file_id: file_b_id,
            expected_prev_delete_file: None,
            delete: del_b,
        },
    ];
    let result = session.finish_with_deletes(&entries).await.unwrap();

    assert_eq!(
        read_pairs(&temp_dir).await,
        vec![(1, 10), (2, 200), (3, 300), (4, 40)],
        "one row updated from each file; the others unchanged"
    );

    // Both delete files and the appended file share the one committed snapshot.
    let snaps: Vec<i64> = sqlx::query_scalar(
        "SELECT begin_snapshot FROM ducklake_delete_file WHERE end_snapshot IS NULL
         UNION
         SELECT begin_snapshot FROM ducklake_data_file
         WHERE table_id = ? AND data_file_id NOT IN (?, ?) AND end_snapshot IS NULL",
    )
    .bind(table_id)
    .bind(file_a_id)
    .bind(file_b_id)
    .fetch_all(&pool)
    .await
    .unwrap();
    assert_eq!(
        snaps,
        vec![result.snapshot_id],
        "both deletes and the append committed in exactly one snapshot"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn register_data_file_with_deletes_rejects_invalid_entries() {
    let temp_dir = TempDir::new().unwrap();
    let writer = create_writer(&temp_dir).await;

    // The entries are validated before any database work, so no table need exist;
    // the file/delete infos are placeholders.
    let file = DataFileInfo::new("new.parquet", 1, 1);
    let entry = |data_file_id: i64| DeleteFileEntry {
        data_file_id,
        expected_prev_delete_file: None,
        delete: DeleteFileInfo::new("del.parquet", 1, 1),
    };

    // Replace + deletes is rejected up front: Replace retires the very files the
    // deletes target, so the combination can never succeed.
    let err = writer
        .register_data_file_with_deletes(
            1,
            "main",
            "t",
            0,
            &file,
            &[entry(1)],
            WriteMode::Replace,
            0,
            &[],
            &[],
        )
        .expect_err("Replace + deletes must be rejected");
    assert!(
        matches!(err, DuckLakeError::InvalidConfig(_)),
        "got {err:?}"
    );

    // Two entries for the same data file are rejected (positions must be unioned
    // into one entry per file).
    let err = writer
        .register_data_file_with_deletes(
            1,
            "main",
            "t",
            0,
            &file,
            &[entry(7), entry(7)],
            WriteMode::Append,
            0,
            &[],
            &[],
        )
        .expect_err("duplicate data_file_id must be rejected");
    assert!(
        matches!(err, DuckLakeError::InvalidConfig(_)),
        "got {err:?}"
    );
}
