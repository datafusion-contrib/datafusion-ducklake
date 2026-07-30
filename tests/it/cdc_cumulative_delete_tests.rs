//! Cumulative (current-spec, 3-column) delete files: per-row snapshot
//! windowing in the CDC feeds.
//!
//! Current official DuckLake writes ONE cumulative delete file per data file,
//! whose rows carry the snapshot at which each position was deleted
//! (`(file_path, pos, _ducklake_internal_snapshot_id)`, field ids
//! 2147483646 / 2147483645 / 2147483539), registered with
//! `begin_snapshot` = MIN embedded snapshot and `partial_max` = MAX. Neither
//! this crate's writer nor the installed (1.4-era) extension produces such
//! files, so the fixture hand-writes one exactly as current official does and
//! asserts the feeds window its rows PER ROW — each deletion reported at its
//! own delete snapshot, including windows that start past `begin_snapshot`.

#![cfg(all(feature = "write-sqlite", feature = "metadata-sqlite"))]

use std::sync::Arc;

use arrow::array::{Array, Int32Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::prelude::*;
use object_store::local::LocalFileSystem;
use sqlx::Row;
use sqlx::sqlite::SqlitePool;
use tempfile::TempDir;

use datafusion_ducklake::row_id::SNAPSHOT_ID_PARQUET_FIELD_ID;
use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeTableWriter, MetadataWriter, SqliteMetadataProvider,
    SqliteMetadataWriter, register_ducklake_functions,
};

fn db_url(temp: &TempDir) -> String {
    format!("sqlite:{}?mode=rwc", temp.path().join("test.db").display())
}

fn ro_url(temp: &TempDir) -> String {
    format!("sqlite:{}", temp.path().join("test.db").display())
}

/// A parquet field tagged with a `PARQUET:field_id`.
fn field_with_id(name: &str, data_type: DataType, nullable: bool, field_id: i32) -> Field {
    let mut metadata = std::collections::HashMap::new();
    metadata.insert("PARQUET:field_id".to_string(), field_id.to_string());
    Field::new(name, data_type, nullable).with_metadata(metadata)
}

/// Write a cumulative delete parquet next to the data file: positions `pos`
/// deleted at snapshots `snaps` (parallel arrays).
fn write_cumulative_delete_file(
    path: &std::path::Path,
    data_file_name: &str,
    pos: Vec<i64>,
    snaps: Vec<i64>,
) -> i64 {
    let schema = Arc::new(Schema::new(vec![
        field_with_id("file_path", DataType::Utf8, false, 2_147_483_646),
        field_with_id("pos", DataType::Int64, false, 2_147_483_645),
        field_with_id(
            "_ducklake_internal_snapshot_id",
            DataType::Int64,
            true,
            SNAPSHOT_ID_PARQUET_FIELD_ID,
        ),
    ]));
    let n = pos.len();
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(vec![data_file_name; n])),
            Arc::new(Int64Array::from(pos)),
            Arc::new(Int64Array::from(snaps)),
        ],
    )
    .unwrap();
    let file = std::fs::File::create(path).unwrap();
    let mut writer = parquet::arrow::ArrowWriter::try_new(file, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    std::fs::metadata(path).unwrap().len() as i64
}

async fn feed_rows(ctx: &SessionContext, sql: &str) -> Vec<(i64, i64, String, i32)> {
    let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
    let mut rows = Vec::new();
    for b in &batches {
        let snaps = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        let rowids = b.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
        let cts = b.column(2).as_any().downcast_ref::<StringArray>().unwrap();
        let ids = b.column(3).as_any().downcast_ref::<Int32Array>().unwrap();
        for r in 0..b.num_rows() {
            rows.push((
                snaps.value(r),
                rowids.value(r),
                cts.value(r).to_string(),
                ids.value(r),
            ));
        }
    }
    rows.sort();
    rows
}

/// Positions 1 (id=2) and 3 (id=4) deleted at snapshots 2 and 3 respectively,
/// recorded in ONE cumulative delete file with begin_snapshot=2 /
/// partial_max=3. Every window must attribute each deletion to its own
/// snapshot; windows starting past begin_snapshot must still see in-window
/// rows (via ducklake_delete_file.partial_max).
#[tokio::test(flavor = "multi_thread")]
async fn cumulative_delete_file_windows_per_row() {
    let temp = TempDir::new().unwrap();

    // Seed main.t(id, val) with ids 1..4 in one data file (row_id_start = 0).
    let data_path = temp.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();
    let writer = SqliteMetadataWriter::new_with_init(&db_url(&temp))
        .await
        .unwrap();
    writer.set_data_path(data_path.to_str().unwrap()).unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("val", DataType::Int32, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4])),
            Arc::new(Int32Array::from(vec![10, 20, 30, 40])),
        ],
    )
    .unwrap();
    DuckLakeTableWriter::new(Arc::new(writer), Arc::new(LocalFileSystem::new()) as _)
        .unwrap()
        .write_table("main", "t", &[batch])
        .await
        .unwrap();

    let pool = SqlitePool::connect(&ro_url(&temp)).await.unwrap();
    let wpool = SqlitePool::connect(&db_url(&temp)).await.unwrap();

    let row = sqlx::query(
        "SELECT data_file_id, table_id, path FROM ducklake_data_file WHERE end_snapshot IS NULL",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    let data_file_id: i64 = row.try_get(0).unwrap();
    let table_id: i64 = row.try_get(1).unwrap();
    let data_file_rel: String = row.try_get(2).unwrap();

    // Locate the (single) data parquet on disk; the delete file goes next to
    // it, registered relative to the table path like the data file is.
    fn find_parquet(dir: &std::path::Path) -> Option<std::path::PathBuf> {
        for entry in std::fs::read_dir(dir).ok()? {
            let p = entry.ok()?.path();
            if p.is_dir() {
                if let Some(found) = find_parquet(&p) {
                    return Some(found);
                }
            } else if p.extension().is_some_and(|e| e == "parquet") {
                return Some(p);
            }
        }
        None
    }
    let data_file_on_disk = find_parquet(&data_path).expect("seeded data parquet exists");
    let table_dir = data_file_on_disk.parent().unwrap().to_path_buf();
    let base: i64 = sqlx::query("SELECT MAX(snapshot_id) FROM ducklake_snapshot")
        .fetch_one(&pool)
        .await
        .unwrap()
        .try_get(0)
        .unwrap();
    let (s_del1, s_del2) = (base + 1, base + 2);

    // Two delete snapshots, recorded as plain snapshot rows.
    for s in [s_del1, s_del2] {
        sqlx::query("INSERT INTO ducklake_snapshot (snapshot_id, schema_version) VALUES (?, 0)")
            .bind(s)
            .execute(&wpool)
            .await
            .unwrap();
    }

    let delete_name = "cumulative-delete.parquet";
    let delete_size = write_cumulative_delete_file(
        &table_dir.join(delete_name),
        &data_file_rel,
        vec![1, 3],
        vec![s_del1, s_del2],
    );
    sqlx::query(
        "INSERT INTO ducklake_delete_file
           (data_file_id, table_id, path, path_is_relative, file_size_bytes,
            delete_count, begin_snapshot, partial_max)
         VALUES (?, ?, ?, 1, ?, 2, ?, ?)",
    )
    .bind(data_file_id)
    .bind(table_id)
    .bind(delete_name)
    .bind(delete_size)
    .bind(s_del1)
    .bind(s_del2)
    .execute(&wpool)
    .await
    .unwrap();

    // Read the feeds through the sqlite provider.
    let provider = Arc::new(SqliteMetadataProvider::new(&ro_url(&temp)).await.unwrap());
    let catalog =
        DuckLakeCatalog::new(SqliteMetadataProvider::new(&ro_url(&temp)).await.unwrap()).unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));
    register_ducklake_functions(&ctx, provider);

    let del = |a: i64, b: i64| {
        format!(
            "SELECT snapshot_id, rowid, change_type, id FROM \
             ducklake_table_deletions('main.t', {a}, {b})"
        )
    };
    let chg = |a: i64, b: i64| {
        format!(
            "SELECT snapshot_id, rowid, change_type, id FROM \
             ducklake_table_changes('main.t', {a}, {b}) WHERE change_type = 'delete'"
        )
    };

    // Each deletion at its own snapshot; positions 1/3 are ids 2/4, rowids 1/3.
    let d1 = (s_del1, 1i64, "delete".to_string(), 2i32);
    let d2 = (s_del2, 3i64, "delete".to_string(), 4i32);

    assert_eq!(
        feed_rows(&ctx, &del(s_del1, s_del1)).await,
        vec![d1.clone()]
    );
    assert_eq!(
        feed_rows(&ctx, &del(s_del2, s_del2)).await,
        vec![d2.clone()]
    );
    assert_eq!(
        feed_rows(&ctx, &del(s_del1, s_del2)).await,
        vec![d1.clone(), d2.clone()]
    );
    // Window starting past the delete file's begin_snapshot: reached only via
    // ducklake_delete_file.partial_max, and must NOT re-report the earlier
    // deletion.
    assert_eq!(feed_rows(&ctx, &del(s_del2, 1000)).await, vec![d2.clone()]);

    // ducklake_table_changes windows the same way (pure deletes).
    assert_eq!(feed_rows(&ctx, &chg(s_del1, s_del1)).await, vec![d1]);
    assert_eq!(feed_rows(&ctx, &chg(s_del2, 1000)).await, vec![d2]);
}
