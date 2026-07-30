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

use datafusion::logical_expr::Operator;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_expr::expressions::{BinaryExpr, col, lit};
use datafusion_ducklake::{
    DeleteFileInfo, DuckLakeCatalog, DuckLakeFileData, DuckLakeTable, DuckLakeTableWriter,
    MetadataWriter, SqliteMetadataProvider, SqliteMetadataWriter,
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

/// The full crate-side delete pipeline, end to end: `resolve_positions` finds
/// the matching rows' physical positions, `write_delete_file` writes and uploads
/// the `(file_path, pos)` delete parquet, `set_delete_file` registers it, and the
/// read path applies it. This is the only test that drives `resolve_positions`
/// (position discovery) and `write_delete_file` (delete-file authoring); the
/// other tests hand-build the delete parquet and positions.
#[tokio::test(flavor = "multi_thread")]
async fn resolve_write_and_apply_positional_delete() {
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

    // Catalog ids + the data file's stored (path, path_is_relative, size), which
    // are what `DuckLakeFileData` needs to be scanned.
    let db_path = temp_dir.path().join("test.db");
    let conn_str = format!("sqlite:{}", db_path.display());
    let pool = SqlitePool::connect(&conn_str).await.unwrap();
    let table_id: i64 =
        sqlx::query_scalar("SELECT table_id FROM ducklake_table WHERE end_snapshot IS NULL")
            .fetch_one(&pool)
            .await
            .unwrap();
    let df_row = sqlx::query(
        "SELECT data_file_id, path, path_is_relative, file_size_bytes
         FROM ducklake_data_file WHERE table_id = ? AND end_snapshot IS NULL",
    )
    .bind(table_id)
    .fetch_one(&pool)
    .await
    .unwrap();
    let data_file_id: i64 = df_row.try_get(0).unwrap();
    let data_file_path: String = df_row.try_get(1).unwrap();
    let path_is_relative: bool = df_row.try_get::<i64, _>(2).unwrap() != 0;
    let file_size_bytes: i64 = df_row.try_get(3).unwrap();
    let data_file =
        DuckLakeFileData::new(data_file_path.clone(), path_is_relative, file_size_bytes);

    // Resolve positions for `id = 2 OR id = 4` — ids at physical positions 1 and
    // 3. The predicate is index-based against the table's logical column order
    // (`id` is column 0).
    let provider = SqliteMetadataProvider::new(&conn_str).await.unwrap();
    let catalog = DuckLakeCatalog::new(provider).unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("test", Arc::new(catalog));
    let schema_provider = ctx.catalog("test").unwrap().schema("main").unwrap();
    let table_provider = schema_provider.table("t").await.unwrap().unwrap();
    // `TableProvider: Any` in DataFusion 54 (no `as_any()` method); upcast to
    // `dyn Any` to reach the concrete `DuckLakeTable`.
    let table = (table_provider.as_ref() as &dyn std::any::Any)
        .downcast_ref::<DuckLakeTable>()
        .expect("provider is a DuckLakeTable");

    let data_schema = Schema::new(vec![Field::new("id", DataType::Int32, false)]);
    let id: Arc<dyn PhysicalExpr> = col("id", &data_schema).unwrap();
    let eq2: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(id.clone(), Operator::Eq, lit(2i32)));
    let eq4: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(id, Operator::Eq, lit(4i32)));
    let predicate: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(eq2, Operator::Or, eq4));

    let state = ctx.state();
    let positions = table
        .resolve_positions(&state, &data_file, predicate)
        .await
        .unwrap();
    let mut positions: Vec<i64> = positions.into_iter().collect();
    positions.sort_unstable();
    assert_eq!(
        positions,
        vec![1, 3],
        "ids 2 and 4 sit at positions 1 and 3"
    );

    // Author the delete file from those positions and register it (no prior).
    let del_info = DuckLakeTableWriter::new(writer.clone(), object_store.clone())
        .unwrap()
        .write_delete_file("main", "t", &data_file_path, &positions)
        .await
        .unwrap();
    let base: i64 = sqlx::query_scalar("SELECT MAX(snapshot_id) FROM ducklake_snapshot")
        .fetch_one(&pool)
        .await
        .unwrap();
    writer
        .set_delete_file(
            table_id,
            "main",
            "t",
            base,
            data_file_id,
            None,
            base,
            &del_info,
        )
        .unwrap();

    assert_eq!(
        read_ids(&temp_dir).await,
        vec![1, 3],
        "resolve -> write_delete_file -> set_delete_file deletes ids 2,4"
    );
}

/// #864 fence: a concurrent APPEND that adds an unrelated data file must NOT
/// block a positional delete on a still-live data file. The resolved positions
/// are physical row indices in the target file, which an append never moves, so
/// the delete commits even against a pre-append `base_snapshot` and both files
/// coexist. (The old table-wide fence rejected this; the target-file fence does
/// not.)
#[tokio::test(flavor = "multi_thread")]
async fn set_delete_file_allows_concurrent_append_to_other_file() {
    let temp_dir = TempDir::new().unwrap();
    let writer = Arc::new(create_writer(&temp_dir).await);
    let object_store: Arc<dyn object_store::ObjectStore> = Arc::new(LocalFileSystem::new());

    // Data file D: ids [1,2,3,4] at physical positions 0..3.
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3, 4]))],
    )
    .unwrap();
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
    // Base snapshot captured BEFORE the concurrent append.
    let base: i64 = sqlx::query_scalar("SELECT MAX(snapshot_id) FROM ducklake_snapshot")
        .fetch_one(&pool)
        .await
        .unwrap();

    // Concurrent append: adds a NEW data file (ids [5,6]) and advances the head,
    // so the table's data generation moved past `base`.
    let batch2 = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)])),
        vec![Arc::new(Int32Array::from(vec![5, 6]))],
    )
    .unwrap();
    DuckLakeTableWriter::new(writer.clone(), object_store.clone())
        .unwrap()
        .append_table("main", "t", &[batch2])
        .await
        .unwrap();
    assert_eq!(
        read_ids(&temp_dir).await,
        vec![1, 2, 3, 4, 5, 6],
        "after append"
    );

    // Delete positions {1,3} (ids 2,4) on the ORIGINAL file, committing against
    // the pre-append `base`. Must succeed despite the intervening append.
    let del = temp_dir.path().join("delete_append.parquet");
    let size = write_delete_parquet(&del, &data_file_path, &[1, 3]);
    let info = DeleteFileInfo::new(del.to_string_lossy().to_string(), size, 2).with_absolute_path();
    writer
        .set_delete_file(table_id, "main", "t", base, data_file_id, None, base, &info)
        .expect("append to an unrelated file must not block the delete");

    assert_eq!(
        read_ids(&temp_dir).await,
        vec![1, 3, 5, 6],
        "positions 1,3 deleted from the original file; appended rows untouched"
    );
}

/// #864 fence: a positional delete on a data file that a concurrent Replace has
/// RETIRED must be rejected — the resolved positions refer to a file that is no
/// longer live, so committing them would mask the wrong generation.
#[tokio::test(flavor = "multi_thread")]
async fn set_delete_file_rejects_delete_on_retired_file() {
    let temp_dir = TempDir::new().unwrap();
    let writer = Arc::new(create_writer(&temp_dir).await);
    let object_store: Arc<dyn object_store::ObjectStore> = Arc::new(LocalFileSystem::new());

    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3, 4]))],
    )
    .unwrap();
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

    // Concurrent Replace: retires the original data file and writes a new one.
    let batch2 = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)])),
        vec![Arc::new(Int32Array::from(vec![7, 8]))],
    )
    .unwrap();
    DuckLakeTableWriter::new(writer.clone(), object_store.clone())
        .unwrap()
        .write_table("main", "t", &[batch2])
        .await
        .unwrap();

    // Deleting from the now-retired file must Conflict.
    let del = temp_dir.path().join("delete_retired.parquet");
    let size = write_delete_parquet(&del, &data_file_path, &[1]);
    let info = DeleteFileInfo::new(del.to_string_lossy().to_string(), size, 1).with_absolute_path();
    let err = writer
        .set_delete_file(table_id, "main", "t", base, data_file_id, None, base, &info)
        .expect_err("delete on a retired data file must be rejected");
    assert!(
        matches!(err, datafusion_ducklake::DuckLakeError::Conflict(_)),
        "expected a Conflict, got {err:?}"
    );
}
