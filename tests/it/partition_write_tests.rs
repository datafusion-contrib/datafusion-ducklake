//! Partitioned write tests (SQLite single-catalog).
//!
//! Exercises the full partitioned-INSERT path: set a spec, INSERT via SQL, and
//! verify the crate writes one data file per partition with the correct
//! `partition_id` + `ducklake_file_partition_value` rows, reads them back, and
//! prunes on the partition column.

#![cfg(all(feature = "write-sqlite", feature = "metadata-sqlite"))]

use std::sync::Arc;

use arrow::datatypes::{DataType, TimeUnit};
use datafusion::prelude::*;
use tempfile::TempDir;

use datafusion_ducklake::metadata_provider::MetadataProvider;
use datafusion_ducklake::partition::PartitionTransform;
use datafusion_ducklake::{
    ColumnDef, DuckLakeCatalog, MetadataWriter, SqliteMetadataProvider, SqliteMetadataWriter,
    WriteMode, execute_ducklake_sql,
};

struct Env {
    conn_str: String,
    table_id: i64,
    _temp: TempDir,
}

/// Create a writable SQLite catalog with an `events(id, region, ts)` table that is
/// partitioned by `(region, year(ts))` BEFORE any data is written.
async fn setup() -> Env {
    let temp = TempDir::new().unwrap();
    let db_path = temp.path().join("test.db");
    let data_path = temp.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();
    let conn_str = format!("sqlite:{}?mode=rwc", db_path.display());

    let writer = SqliteMetadataWriter::new_with_init(&conn_str)
        .await
        .unwrap();
    writer.set_data_path(data_path.to_str().unwrap()).unwrap();

    let ts_type = DataType::Timestamp(TimeUnit::Microsecond, None);
    let cols = vec![
        ColumnDef::from_arrow("id", &DataType::Int32, false).unwrap(),
        ColumnDef::from_arrow("region", &DataType::Utf8, true).unwrap(),
        ColumnDef::from_arrow("ts", &ts_type, true).unwrap(),
    ];
    // Create the (empty) table, then set the partition spec — so a catalog opened
    // afterwards pins a snapshot that already has the spec.
    let s = writer
        .begin_write_transaction("main", "events", &cols, WriteMode::Replace)
        .unwrap();
    writer
        .publish_snapshot(
            s.table_id,
            "main",
            "events",
            s.snapshot_id,
            WriteMode::Replace,
            s.base_snapshot_id,
            &cols,
            &s.column_ids,
        )
        .unwrap();
    writer
        .set_partition_spec(
            s.table_id,
            &[
                ("region".to_string(), PartitionTransform::Identity),
                ("ts".to_string(), PartitionTransform::Year),
            ],
        )
        .unwrap();

    Env {
        conn_str,
        table_id: s.table_id,
        _temp: temp,
    }
}

/// Open a fresh writable context (pins the current head).
async fn write_ctx(conn_str: &str) -> SessionContext {
    let writer = SqliteMetadataWriter::new_with_init(conn_str).await.unwrap();
    let provider = SqliteMetadataProvider::new(conn_str).await.unwrap();
    let catalog = DuckLakeCatalog::with_writer(Arc::new(provider), Arc::new(writer)).unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));
    ctx
}

/// Open a fresh read-only context (pins the current head to see prior writes).
async fn read_ctx(conn_str: &str) -> SessionContext {
    let provider = SqliteMetadataProvider::new(conn_str).await.unwrap();
    let catalog = DuckLakeCatalog::new(provider).unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));
    ctx
}

const INSERT_SQL: &str = "INSERT INTO ducklake.main.events \
     SELECT * FROM (VALUES \
        (1, 'us', TIMESTAMP '2023-01-15 10:00:00'), \
        (2, 'us', TIMESTAMP '2024-06-20 12:00:00'), \
        (3, 'eu', TIMESTAMP '2023-03-10 08:00:00'), \
        (4, 'eu', TIMESTAMP '2024-11-05 18:00:00')) AS t(id, region, ts)";

/// Four `events` rows spanning 4 partitions — (us,2023), (us,2024), (eu,2023),
/// (eu,2024) — as separate batches, so a streaming session sees each partition
/// across more than one `write_batch` call.
fn events_batches() -> Vec<arrow::record_batch::RecordBatch> {
    use arrow::array::{ArrayRef, Int32Array, RecordBatch, TimestampMicrosecondArray};
    use arrow::datatypes::{Field, Schema};

    let ts_type = DataType::Timestamp(TimeUnit::Microsecond, None);
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("region", DataType::Utf8, true),
        Field::new("ts", ts_type, true),
    ]));
    // 2023-01-15T10:00:00Z and 2024-06-20T12:00:00Z in micros.
    let y2023: i64 = 1_673_776_800_000_000;
    let y2024: i64 = 1_718_884_800_000_000;
    let rows: [(&[i32], &[&str], &[i64]); 2] =
        [(&[1, 2], &["us", "us"], &[y2023, y2024]), (&[3, 4], &["eu", "eu"], &[y2023, y2024])];
    rows.iter()
        .map(|(ids, regions, times)| {
            RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int32Array::from(ids.to_vec())) as ArrayRef,
                    Arc::new(arrow::array::StringArray::from(regions.to_vec())) as ArrayRef,
                    Arc::new(TimestampMicrosecondArray::from(times.to_vec())) as ArrayRef,
                ],
            )
            .unwrap()
        })
        .collect()
}

/// The streaming session (`begin_write` + `write_batch` + `finish`) — the entry
/// point an embedding engine uses for ingest, with no SQL and no MetadataProvider —
/// must split rows across one file per partition and commit them in ONE snapshot.
#[tokio::test(flavor = "multi_thread")]
async fn streaming_session_splits_rows_across_partitions() {
    use datafusion_ducklake::table_writer::DuckLakeTableWriter;

    let env = setup().await;
    let writer = SqliteMetadataWriter::new_with_init(&env.conn_str)
        .await
        .unwrap();
    let object_store: Arc<dyn object_store::ObjectStore> =
        Arc::new(object_store::local::LocalFileSystem::new());
    let table_writer = DuckLakeTableWriter::new(Arc::new(writer), object_store).unwrap();

    let batches = events_batches();
    let arrow_schema = batches[0].schema();
    let mut session = table_writer
        .begin_write("main", "events", arrow_schema.as_ref(), WriteMode::Append)
        .unwrap();
    for batch in &batches {
        session.write_batch(batch).unwrap();
    }
    let result = session.finish().await.unwrap();

    // One file per (region, year) partition, all in a single snapshot.
    assert_eq!(result.records_written, 4);
    assert_eq!(
        result.files_written, 4,
        "a streaming write must produce one file per partition"
    );

    let provider = SqliteMetadataProvider::new(&env.conn_str).await.unwrap();
    let snap = provider.get_current_snapshot().unwrap();
    let live = provider
        .get_partition_spec(env.table_id, snap)
        .unwrap()
        .unwrap();
    let page = provider
        .get_table_file_metadata_page(env.table_id, snap, None, 4096)
        .unwrap();
    assert_eq!(page.len(), 4);
    let mut seen: Vec<Vec<Option<String>>> = Vec::new();
    for meta in &page {
        assert_eq!(
            meta.file.partition_id,
            Some(live.partition_id),
            "every streamed file must carry the live partition generation"
        );
        let mut values = meta.file.partition_values.clone();
        values.sort_by_key(|(index, _)| *index);
        seen.push(values.into_iter().map(|(_, v)| v).collect());
        // The Hive directory mirrors the partition values.
        assert!(
            meta.file.file.path.contains("region="),
            "partitioned file must live under a Hive path, got {}",
            meta.file.file.path
        );
    }
    seen.sort();
    assert_eq!(
        seen,
        vec![
            vec![Some("eu".to_string()), Some("2023".to_string())],
            vec![Some("eu".to_string()), Some("2024".to_string())],
            vec![Some("us".to_string()), Some("2023".to_string())],
            vec![Some("us".to_string()), Some("2024".to_string())],
        ]
    );

    // The rows read back intact through the partitioned layout.
    let ctx = read_ctx(&env.conn_str).await;
    let rows = ctx
        .sql("SELECT count(*) FROM ducklake.main.events WHERE region = 'us'")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let count = rows[0]
        .column(0)
        .as_any()
        .downcast_ref::<arrow::array::Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!(count, 2);
}

/// With the open-file cap at 1, a streaming write touching 4 partitions must still
/// land every row: the sink finalizes the open file to make room and re-opens a
/// partition later if more of its rows arrive.
#[tokio::test(flavor = "multi_thread")]
async fn streaming_session_respects_open_partition_cap() {
    use datafusion_ducklake::table_writer::DuckLakeTableWriter;

    let env = setup().await;
    let writer = SqliteMetadataWriter::new_with_init(&env.conn_str)
        .await
        .unwrap();
    let object_store: Arc<dyn object_store::ObjectStore> =
        Arc::new(object_store::local::LocalFileSystem::new());
    let table_writer = DuckLakeTableWriter::new(Arc::new(writer), object_store)
        .unwrap()
        .with_max_open_partitions(1);

    let batches = events_batches();
    let arrow_schema = batches[0].schema();
    let mut session = table_writer
        .begin_write("main", "events", arrow_schema.as_ref(), WriteMode::Append)
        .unwrap();
    for batch in &batches {
        session.write_batch(batch).unwrap();
    }
    let result = session.finish().await.unwrap();
    assert_eq!(result.records_written, 4, "eviction must not drop rows");

    let ctx = read_ctx(&env.conn_str).await;
    let rows = ctx
        .sql("SELECT count(*) FROM ducklake.main.events")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let count = rows[0]
        .column(0)
        .as_any()
        .downcast_ref::<arrow::array::Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!(count, 4, "every row must be readable after eviction");
}

/// A partitioned write must still respect `target_file_size` WITHIN each partition.
///
/// Rollover is evaluated at batch boundaries, so if the partition splitter collapsed
/// each group into one batch the writer would emit a single file per partition of
/// unbounded size — `target_file_size` silently unenforceable exactly on the tables
/// most likely to be large. DuckLake merges but never splits, so such a file could
/// never be broken up afterwards either.
#[tokio::test(flavor = "multi_thread")]
async fn partitioned_write_rolls_within_each_partition() {
    use arrow::array::{ArrayRef, Int32Array, RecordBatch, TimestampMicrosecondArray};
    use arrow::datatypes::{Field, Schema};
    use datafusion_ducklake::table_writer::DuckLakeTableWriter;

    let env = setup().await; // partitioned by (region, year(ts))
    let ts_type = DataType::Timestamp(TimeUnit::Microsecond, None);
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("region", DataType::Utf8, true),
        Field::new("ts", ts_type, true),
    ]));
    // Many batches, ALL in the same (us, 2023) partition, so the only way to end up
    // with more than one file is rollover firing inside that partition.
    let y2023: i64 = 1_673_776_800_000_000;
    let batches: Vec<RecordBatch> = (0..40)
        .map(|b| {
            let ids: Vec<i32> = (b * 100..(b + 1) * 100).collect();
            let n = ids.len();
            RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int32Array::from(ids)) as ArrayRef,
                    Arc::new(arrow::array::StringArray::from(vec!["us"; n])) as ArrayRef,
                    Arc::new(TimestampMicrosecondArray::from(vec![y2023; n])) as ArrayRef,
                ],
            )
            .unwrap()
        })
        .collect();

    let writer = SqliteMetadataWriter::new_with_init(&env.conn_str)
        .await
        .unwrap();
    let object_store: Arc<dyn object_store::ObjectStore> =
        Arc::new(object_store::local::LocalFileSystem::new());
    let table_writer = DuckLakeTableWriter::new(Arc::new(writer), object_store)
        .unwrap()
        .with_target_file_size(8 * 1024);

    let result = table_writer
        .append_table("main", "events", &batches)
        .await
        .unwrap();
    assert_eq!(result.records_written, 4000);
    assert!(
        result.files_written > 1,
        "a single partition exceeding target_file_size must roll into several files, got {}",
        result.files_written
    );

    // Every file still carries that one partition, and the rows all read back.
    let provider = SqliteMetadataProvider::new(&env.conn_str).await.unwrap();
    let snap = provider.get_current_snapshot().unwrap();
    let page = provider
        .get_table_file_metadata_page(env.table_id, snap, None, 4096)
        .unwrap();
    assert_eq!(page.len(), result.files_written);
    for meta in &page {
        let mut values = meta.file.partition_values.clone();
        values.sort_by_key(|(index, _)| *index);
        let values: Vec<Option<String>> = values.into_iter().map(|(_, v)| v).collect();
        assert_eq!(
            values,
            vec![Some("us".to_string()), Some("2023".to_string())],
            "every rolled file belongs to the same partition"
        );
    }
    let ctx = read_ctx(&env.conn_str).await;
    let rows = ctx
        .sql("SELECT count(*) FROM ducklake.main.events")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(
        rows[0]
            .column(0)
            .as_any()
            .downcast_ref::<arrow::array::Int64Array>()
            .unwrap()
            .value(0),
        4000
    );
}

/// `begin_write_to_path` targets ONE caller-named file, so it cannot satisfy a
/// partition spec that needs one file per partition. It must refuse an incompatible
/// write before upload rather than commit a file without partition metadata.
///
/// It is the only entry point that refuses: a rolling or partitioned session now
/// commits every file it produced, so neither `begin_write` nor
/// `begin_write_single_file` rejects a partitioned table.
#[tokio::test(flavor = "multi_thread")]
async fn custom_path_write_refuses_a_partitioned_table() {
    use datafusion_ducklake::table_writer::DuckLakeTableWriter;

    let env = setup().await; // partitioned by (region, year(ts))
    let writer = SqliteMetadataWriter::new_with_init(&env.conn_str)
        .await
        .unwrap();
    let object_store: Arc<dyn object_store::ObjectStore> =
        Arc::new(object_store::local::LocalFileSystem::new());
    let table_writer = DuckLakeTableWriter::new(Arc::new(writer), object_store).unwrap();
    let batches = events_batches();
    let arrow_schema = batches[0].schema();

    // Custom-path session.
    let err = table_writer
        .begin_write_to_path(
            "main",
            "events",
            arrow_schema.as_ref(),
            "/tmp/some-dir",
            "f.parquet".to_string(),
            WriteMode::Append,
        )
        .expect_err("custom-path write must refuse a partitioned table");
    assert!(
        err.to_string().contains("begin_write_to_path"),
        "the error must name the entry point, got: {err}"
    );

    // Nothing was committed.
    let provider = SqliteMetadataProvider::new(&env.conn_str).await.unwrap();
    let snap = provider.get_current_snapshot().unwrap();
    assert!(
        provider
            .get_table_file_metadata_page(env.table_id, snap, None, 4096)
            .unwrap()
            .is_empty(),
        "a refused write must commit no data files"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn partitioned_update_commits_one_partition_file_atomically() {
    use arrow::array::{Int32Array, Int64Array, StringViewArray, UInt64Array};
    use sqlx::sqlite::SqlitePool;

    let env = setup().await;
    let ctx = write_ctx(&env.conn_str).await;
    let inserted = ctx
        .sql(
            "INSERT INTO ducklake.main.events VALUES \
             (1, 'us', TIMESTAMP '2024-06-20 12:00:00')",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(
        inserted[0]
            .column(0)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap()
            .value(0),
        1,
    );

    let provider = SqliteMetadataProvider::new(&env.conn_str).await.unwrap();
    let before_snapshot = provider.get_current_snapshot().unwrap();
    let ctx = write_ctx(&env.conn_str).await;
    let updated = ctx
        .sql("UPDATE ducklake.main.events SET region = 'eu' WHERE id = 1")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(
        updated[0]
            .column(0)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap()
            .value(0),
        1,
    );

    let provider = SqliteMetadataProvider::new(&env.conn_str).await.unwrap();
    let snapshot = provider.get_current_snapshot().unwrap();
    let page = provider
        .get_table_file_metadata_page(env.table_id, snapshot, None, 4096)
        .unwrap();
    let replacement = page
        .iter()
        .find(|metadata| metadata.file.begin_snapshot == Some(snapshot))
        .unwrap();
    let mut partition_values = replacement.file.partition_values.clone();
    partition_values.sort_by_key(|(index, _)| *index);

    let writer = SqliteMetadataWriter::new_with_init(&env.conn_str)
        .await
        .unwrap();
    let catalog = DuckLakeCatalog::with_writer(
        Arc::new(SqliteMetadataProvider::new(&env.conn_str).await.unwrap()),
        Arc::new(writer),
    )
    .unwrap()
    .with_row_lineage(true);
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));
    let rows = ctx
        .sql("SELECT rowid, id, region FROM ducklake.main.events")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let rows = arrow::compute::concat_batches(&rows[0].schema(), &rows).unwrap();
    let pool = SqlitePool::connect(&env.conn_str).await.unwrap();
    let delete_count: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM ducklake_delete_file WHERE begin_snapshot = ?")
            .bind(snapshot)
            .fetch_one(&pool)
            .await
            .unwrap();

    assert_eq!(snapshot, before_snapshot + 1);
    assert_eq!(
        partition_values,
        vec![(0, Some("eu".to_string())), (1, Some("2024".to_string()))],
    );
    assert_eq!(delete_count, 1);
    assert_eq!(rows.num_rows(), 1);
    assert_eq!(
        rows.column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0),
        0,
    );
    assert_eq!(
        rows.column(1)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap()
            .value(0),
        1,
    );
    assert_eq!(
        rows.column(2)
            .as_any()
            .downcast_ref::<StringViewArray>()
            .unwrap()
            .value(0),
        "eu",
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn partitioned_append_with_deletes_commits_one_partition_file_atomically() {
    use datafusion_ducklake::metadata_writer::DeleteFileEntry;
    use datafusion_ducklake::table_writer::DuckLakeTableWriter;
    use sqlx::sqlite::SqlitePool;

    let env = setup().await;
    let writer: Arc<dyn MetadataWriter> = Arc::new(
        SqliteMetadataWriter::new_with_init(&env.conn_str)
            .await
            .unwrap(),
    );
    let object_store: Arc<dyn object_store::ObjectStore> =
        Arc::new(object_store::local::LocalFileSystem::new());
    let table_writer =
        DuckLakeTableWriter::new(Arc::clone(&writer), Arc::clone(&object_store)).unwrap();
    let batches = events_batches();
    let schema = batches[0].schema();

    let mut seed = table_writer
        .begin_write("main", "events", schema.as_ref(), WriteMode::Append)
        .unwrap();
    seed.write_batch(&batches[0].slice(0, 1)).unwrap();
    let seeded = seed.finish().await.unwrap();

    let pool = SqlitePool::connect(&env.conn_str).await.unwrap();
    let (source_data_file_id, source_path) = sqlx::query_as::<_, (i64, String)>(
        "SELECT data_file_id, path FROM ducklake_data_file
             WHERE table_id = ? AND begin_snapshot = ?",
    )
    .bind(env.table_id)
    .bind(seeded.snapshot_id)
    .fetch_one(&pool)
    .await
    .unwrap();
    let delete = table_writer
        .write_delete_file("main", "events", &source_path, &[0])
        .await
        .unwrap();

    let mut replacement = table_writer
        .begin_write("main", "events", schema.as_ref(), WriteMode::Append)
        .unwrap();
    replacement.write_batch(&batches[0].slice(1, 1)).unwrap();
    let committed = replacement
        .finish_with_deletes(&[DeleteFileEntry {
            data_file_id: source_data_file_id,
            expected_prev_delete_file: None,
            delete,
        }])
        .await
        .unwrap();

    let delete_snapshot: i64 = sqlx::query_scalar(
        "SELECT begin_snapshot FROM ducklake_delete_file
         WHERE data_file_id = ? AND end_snapshot IS NULL",
    )
    .bind(source_data_file_id)
    .fetch_one(&pool)
    .await
    .unwrap();
    let provider = SqliteMetadataProvider::new(&env.conn_str).await.unwrap();
    let page = provider
        .get_table_file_metadata_page(env.table_id, committed.snapshot_id, None, 4096)
        .unwrap();
    let appended = page
        .iter()
        .find(|metadata| metadata.file.data_file_id != source_data_file_id)
        .unwrap();

    assert_eq!(committed.files_written, 1);
    assert_eq!(committed.records_written, 1);
    assert_eq!(delete_snapshot, committed.snapshot_id);
    assert_eq!(appended.file.begin_snapshot, Some(committed.snapshot_id));
    assert_eq!(
        appended.file.partition_values,
        vec![(0, Some("us".to_string())), (1, Some("2024".to_string()))],
    );
}

/// A keyed mutation on a PARTITIONED table: the new row versions land in several
/// partitions, so the appended side is several files. All of them must commit in the
/// SAME snapshot as the delete files that supersede the old versions.
///
/// This is the shape official DuckLake produces for `UPDATE` on a partitioned table
/// (a partition-moving update writes one data file per touched output partition and
/// one delete file per touched input file, all stamped with one snapshot).
///
/// Also asserts each appended file carries its OWN partition values and its OWN
/// per-column statistics — a commit that only recorded the first file's would leave
/// the rest unprunable.
#[tokio::test(flavor = "multi_thread")]
async fn partitioned_append_with_deletes_commits_every_partition_file_atomically() {
    use datafusion_ducklake::metadata_writer::DeleteFileEntry;
    use datafusion_ducklake::table_writer::DuckLakeTableWriter;
    use sqlx::sqlite::SqlitePool;

    let env = setup().await; // partitioned by (region, year(ts))
    let writer: Arc<dyn MetadataWriter> = Arc::new(
        SqliteMetadataWriter::new_with_init(&env.conn_str)
            .await
            .unwrap(),
    );
    let object_store: Arc<dyn object_store::ObjectStore> =
        Arc::new(object_store::local::LocalFileSystem::new());
    let table_writer =
        DuckLakeTableWriter::new(Arc::clone(&writer), Arc::clone(&object_store)).unwrap();
    let batches = events_batches();
    let schema = batches[0].schema();

    // Seed the 4 rows across the 4 (region, year) partitions -> 4 data files.
    let mut seed = table_writer
        .begin_write("main", "events", schema.as_ref(), WriteMode::Append)
        .unwrap();
    for batch in &batches {
        seed.write_batch(batch).unwrap();
    }
    let seeded = seed.finish().await.unwrap();
    assert_eq!(seeded.files_written, 4, "one seed file per partition");

    // Supersede ids 1 and 3 — one row each, so position 0 in their own file.
    let pool = SqlitePool::connect(&env.conn_str).await.unwrap();
    let seed_files = sqlx::query_as::<_, (i64, String, String)>(
        "SELECT f.data_file_id, f.path,
                (SELECT group_concat(v.partition_value, '/') FROM ducklake_file_partition_value v
                 WHERE v.data_file_id = f.data_file_id ORDER BY v.partition_key_index)
         FROM ducklake_data_file f
         WHERE f.table_id = ? AND f.begin_snapshot = ?
         ORDER BY f.data_file_id",
    )
    .bind(env.table_id)
    .bind(seeded.snapshot_id)
    .fetch_all(&pool)
    .await
    .unwrap();
    // (us,2023) holds id 1 and (eu,2023) holds id 3.
    let mut entries = Vec::new();
    for (data_file_id, path, partition) in &seed_files {
        if partition != "us/2023" && partition != "eu/2023" {
            continue;
        }
        let delete = table_writer
            .write_delete_file("main", "events", path, &[0])
            .await
            .unwrap();
        entries.push(DeleteFileEntry {
            data_file_id: *data_file_id,
            expected_prev_delete_file: None,
            delete,
        });
    }
    assert_eq!(entries.len(), 2, "two superseded partition files");

    // The new versions MOVE partition: id 1 -> (apac,2023), id 3 -> (apac,2024). The
    // partitioned session therefore produces TWO appended files, which the old
    // one-file cap refused outright.
    let new_versions = {
        use arrow::array::{ArrayRef, Int32Array, RecordBatch, TimestampMicrosecondArray};
        RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 3])) as ArrayRef,
                Arc::new(arrow::array::StringArray::from(vec!["apac", "apac"])) as ArrayRef,
                Arc::new(TimestampMicrosecondArray::from(vec![
                    1_673_776_800_000_000i64,
                    1_718_884_800_000_000i64,
                ])) as ArrayRef,
            ],
        )
        .unwrap()
    };
    let mut session = table_writer
        .begin_write("main", "events", schema.as_ref(), WriteMode::Append)
        .unwrap();
    session.write_batch(&new_versions).unwrap();
    let committed = session.finish_with_deletes(&entries).await.unwrap();
    assert_eq!(
        committed.files_written, 2,
        "one appended file per partition"
    );
    assert_eq!(committed.records_written, 2);

    // ONE snapshot carries both appended files AND both delete files.
    let appended_snapshots: Vec<i64> = sqlx::query_scalar(
        "SELECT DISTINCT begin_snapshot FROM ducklake_data_file
         WHERE table_id = ? AND begin_snapshot > ?",
    )
    .bind(env.table_id)
    .bind(seeded.snapshot_id)
    .fetch_all(&pool)
    .await
    .unwrap();
    assert_eq!(
        appended_snapshots,
        vec![committed.snapshot_id],
        "both appended files share the one committed snapshot"
    );
    let delete_snapshots: Vec<i64> = sqlx::query_scalar(
        "SELECT DISTINCT begin_snapshot FROM ducklake_delete_file
         WHERE table_id = ? AND end_snapshot IS NULL",
    )
    .bind(env.table_id)
    .fetch_all(&pool)
    .await
    .unwrap();
    assert_eq!(
        delete_snapshots,
        vec![committed.snapshot_id],
        "both delete files share that same snapshot"
    );

    // Each appended file carries its OWN partition values and its OWN column stats.
    let provider = SqliteMetadataProvider::new(&env.conn_str).await.unwrap();
    let page = provider
        .get_table_file_metadata_page(env.table_id, committed.snapshot_id, None, 4096)
        .unwrap();
    let mut appended: Vec<_> = page
        .iter()
        .filter(|m| m.file.begin_snapshot == Some(committed.snapshot_id))
        .collect();
    appended.sort_by_key(|m| m.file.data_file_id);
    assert_eq!(appended.len(), 2);
    let partitions: Vec<Vec<(i32, Option<String>)>> = {
        let mut p: Vec<_> = appended
            .iter()
            .map(|m| m.file.partition_values.clone())
            .collect();
        p.sort();
        p
    };
    assert_eq!(
        partitions,
        vec![
            vec![(0, Some("apac".to_string())), (1, Some("2023".to_string()))],
            vec![(0, Some("apac".to_string())), (1, Some("2024".to_string()))],
        ],
        "each appended file is stamped with its own partition"
    );
    for metadata in &appended {
        let ids: Vec<i64> = {
            let mut ids: Vec<i64> = metadata
                .column_statistics
                .iter()
                .map(|s| s.column_id)
                .collect();
            ids.sort_unstable();
            ids.dedup();
            ids
        };
        assert_eq!(
            ids.len(),
            3,
            "file {} must carry stats for all three columns, got {:?}",
            metadata.file.data_file_id,
            metadata.column_statistics
        );
    }

    // The mutation reads back as an in-place update.
    let ctx = read_ctx(&env.conn_str).await;
    let rows = ctx
        .sql("SELECT id, region FROM ducklake.main.events ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let regions: Vec<String> = rows
        .iter()
        .flat_map(|b| {
            let col = b
                .column(1)
                .as_any()
                .downcast_ref::<arrow::array::StringViewArray>()
                .unwrap();
            (0..b.num_rows())
                .map(|i| col.value(i).to_string())
                .collect::<Vec<_>>()
        })
        .collect();
    assert_eq!(
        regions,
        vec!["apac", "us", "apac", "eu"],
        "ids 1,3 moved partition; ids 2,4 untouched"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn partitioned_insert_writes_one_file_per_partition() {
    let env = setup().await;

    let ctx = write_ctx(&env.conn_str).await;
    let inserted = ctx.sql(INSERT_SQL).await.unwrap().collect().await.unwrap();
    // INSERT reports the number of rows written.
    let count = inserted[0]
        .column(0)
        .as_any()
        .downcast_ref::<arrow::array::UInt64Array>()
        .unwrap()
        .value(0);
    assert_eq!(count, 4, "4 rows inserted");

    // The write should produce four files — one per (region, year) partition —
    // each carrying two partition values.
    let provider = SqliteMetadataProvider::new(&env.conn_str).await.unwrap();
    let snapshot = provider.get_current_snapshot().unwrap();
    let page = provider
        .get_table_file_metadata_page(env.table_id, snapshot, None, 4096)
        .unwrap();
    assert_eq!(page.len(), 4, "one file per (region, year) partition");
    for meta in &page {
        assert!(
            meta.file.partition_id.is_none() || meta.file.partition_id.is_some(),
            "partition_id column readable"
        );
        assert_eq!(
            meta.file.partition_values.len(),
            2,
            "each file has (region, year) values"
        );
    }
    // The distinct (region, year) tuples cover the four expected partitions.
    let mut tuples: Vec<(Option<String>, Option<String>)> = page
        .iter()
        .map(|m| {
            let region = m
                .file
                .partition_values
                .iter()
                .find(|(k, _)| *k == 0)
                .and_then(|(_, v)| v.clone());
            let year = m
                .file
                .partition_values
                .iter()
                .find(|(k, _)| *k == 1)
                .and_then(|(_, v)| v.clone());
            (region, year)
        })
        .collect();
    tuples.sort();
    assert_eq!(
        tuples,
        vec![
            (Some("eu".to_string()), Some("2023".to_string())),
            (Some("eu".to_string()), Some("2024".to_string())),
            (Some("us".to_string()), Some("2023".to_string())),
            (Some("us".to_string()), Some("2024".to_string())),
        ]
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn partitioned_insert_reads_back_correctly_and_prunes() {
    let env = setup().await;
    write_ctx(&env.conn_str)
        .await
        .sql(INSERT_SQL)
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    // Read back all rows.
    let rctx = read_ctx(&env.conn_str).await;
    let all = rctx
        .sql("SELECT id FROM ducklake.main.events ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let total: usize = all.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total, 4);

    // Filter on the partition column: correct rows + pruned plan.
    let us = rctx
        .sql("SELECT id FROM ducklake.main.events WHERE region = 'us'")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let us_total: usize = us.iter().map(|b| b.num_rows()).sum();
    assert_eq!(us_total, 2, "two 'us' rows");

    let plan = rctx
        .sql("SELECT id FROM ducklake.main.events WHERE region = 'us'")
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    let display = datafusion::physical_plan::displayable(plan.as_ref())
        .indent(true)
        .to_string();
    let files = display.matches(".parquet").count();
    // The table is partitioned by (region, year(ts)) → 4 files (us/eu × 2023/2024).
    // Filtering region='us' must prune the two 'eu' files via the identity bound,
    // keeping exactly the two 'us' files (not all 4, and never fewer than 2).
    assert_eq!(
        files, 2,
        "region='us' must prune the two 'eu' partition files, keeping exactly the two 'us' files; got {files}:\n{display}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn partition_pruning_enables_range_pruning_on_remaining_files() {
    let env = setup().await;
    let ctx = write_ctx(&env.conn_str).await;
    ctx.sql(
        "INSERT INTO ducklake.main.events VALUES
            (1, 'data', TIMESTAMP '2023-01-15 10:00:00'),
            (2, 'data', TIMESTAMP '2024-06-20 12:00:00'),
            (3, 'metadata', NULL)",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();
    let rctx = read_ctx(&env.conn_str).await;
    let query = "SELECT id FROM ducklake.main.events
        WHERE region = 'data'
          AND ts >= TIMESTAMP '2023-01-01 00:00:00'
          AND ts < TIMESTAMP '2024-01-01 00:00:00'";
    let plan = rctx
        .sql(query)
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    let display = datafusion::physical_plan::displayable(plan.as_ref())
        .indent(true)
        .to_string();
    let rows = rctx.sql(query).await.unwrap().collect().await.unwrap();

    assert_eq!(display.matches(".parquet").count(), 1, "{display}");
    assert_eq!(rows.iter().map(|batch| batch.num_rows()).sum::<usize>(), 1);
}

/// Create the `events` table (no partition spec) and return `(conn_str, table_id, temp)`.
async fn create_events_table_no_spec() -> (String, i64, TempDir) {
    let temp = TempDir::new().unwrap();
    let db_path = temp.path().join("test.db");
    let data_path = temp.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();
    let conn_str = format!("sqlite:{}?mode=rwc", db_path.display());

    let writer = SqliteMetadataWriter::new_with_init(&conn_str)
        .await
        .unwrap();
    writer.set_data_path(data_path.to_str().unwrap()).unwrap();
    let ts_type = DataType::Timestamp(TimeUnit::Microsecond, None);
    let cols = vec![
        ColumnDef::from_arrow("id", &DataType::Int32, false).unwrap(),
        ColumnDef::from_arrow("region", &DataType::Utf8, true).unwrap(),
        ColumnDef::from_arrow("ts", &ts_type, true).unwrap(),
    ];
    let s = writer
        .begin_write_transaction("main", "events", &cols, WriteMode::Replace)
        .unwrap();
    writer
        .publish_snapshot(
            s.table_id,
            "main",
            "events",
            s.snapshot_id,
            WriteMode::Replace,
            s.base_snapshot_id,
            &cols,
            &s.column_ids,
        )
        .unwrap();
    (conn_str, s.table_id, temp)
}

async fn writable_catalog(conn_str: &str) -> (SessionContext, Arc<DuckLakeCatalog>) {
    let writer = SqliteMetadataWriter::new_with_init(conn_str).await.unwrap();
    let provider = SqliteMetadataProvider::new(conn_str).await.unwrap();
    let catalog =
        Arc::new(DuckLakeCatalog::with_writer(Arc::new(provider), Arc::new(writer)).unwrap());
    let ctx = SessionContext::new();
    ctx.register_catalog(
        "ducklake",
        Arc::clone(&catalog) as Arc<dyn datafusion::catalog::CatalogProvider>,
    );
    (ctx, catalog)
}

#[tokio::test(flavor = "multi_thread")]
async fn sql_hook_set_and_reset_partitioned_by() {
    let (conn_str, table_id, _temp) = create_events_table_no_spec().await;
    let (ctx, catalog) = writable_catalog(&conn_str).await;

    // SET PARTITIONED BY via the SQL hook.
    execute_ducklake_sql(
        &ctx,
        &catalog,
        "ALTER TABLE ducklake.main.events SET PARTITIONED BY (region, year(ts))",
    )
    .await
    .unwrap();

    let p = SqliteMetadataProvider::new(&conn_str).await.unwrap();
    let snap = p.get_current_snapshot().unwrap();
    let spec = p
        .get_partition_spec(table_id, snap)
        .unwrap()
        .expect("spec set via SQL hook");
    assert_eq!(spec.columns.len(), 2);
    assert_eq!(spec.columns[0].transform, PartitionTransform::Identity);
    assert_eq!(spec.columns[1].transform, PartitionTransform::Year);

    // RESET PARTITIONED BY removes it.
    execute_ducklake_sql(
        &ctx,
        &catalog,
        "ALTER TABLE ducklake.main.events RESET PARTITIONED BY",
    )
    .await
    .unwrap();
    let p2 = SqliteMetadataProvider::new(&conn_str).await.unwrap();
    let snap2 = p2.get_current_snapshot().unwrap();
    assert!(
        p2.get_partition_spec(table_id, snap2).unwrap().is_none(),
        "spec removed after RESET"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn sql_hook_rejects_unknown_transform() {
    let (conn_str, _table_id, _temp) = create_events_table_no_spec().await;
    let (ctx, catalog) = writable_catalog(&conn_str).await;
    let err = execute_ducklake_sql(
        &ctx,
        &catalog,
        "ALTER TABLE ducklake.main.events SET PARTITIONED BY (bucket(4, region))",
    )
    .await
    .unwrap_err();
    assert!(
        err.to_string().to_lowercase().contains("transform"),
        "expected an unsupported-transform error, got: {err}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn sql_hook_delegates_non_partition_sql() {
    let (conn_str, _table_id, _temp) = create_events_table_no_spec().await;
    let (ctx, catalog) = writable_catalog(&conn_str).await;
    // A plain query flows through to ctx.sql unchanged.
    let batches = execute_ducklake_sql(&ctx, &catalog, "SELECT 1 AS x")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total, 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn insert_stays_partitioned_after_repartition() {
    // Regression: a SECOND partition-spec change (re-partition) must not silently
    // make subsequent INSERTs write unpartitioned files under the live spec.
    let (conn_str, table_id, _temp) = create_events_table_no_spec().await;
    {
        let w = SqliteMetadataWriter::new_with_init(&conn_str)
            .await
            .unwrap();
        // Two generations: region (identity) then year(ts).
        w.set_partition_spec(
            table_id,
            &[("region".to_string(), PartitionTransform::Identity)],
        )
        .unwrap();
        w.set_partition_spec(table_id, &[("ts".to_string(), PartitionTransform::Year)])
            .unwrap();
    }

    write_ctx(&conn_str)
        .await
        .sql(INSERT_SQL)
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    // The INSERT must partition by the LIVE spec (year(ts)) → one file per year,
    // not a single unpartitioned file.
    let provider = SqliteMetadataProvider::new(&conn_str).await.unwrap();
    let snapshot = provider.get_current_snapshot().unwrap();
    let page = provider
        .get_table_file_metadata_page(table_id, snapshot, None, 4096)
        .unwrap();
    assert_eq!(
        page.len(),
        2,
        "re-partitioned INSERT must produce one file per year, not one unpartitioned file"
    );
    let mut years: Vec<String> = page
        .iter()
        .filter_map(|m| m.file.partition_values.first().and_then(|(_, v)| v.clone()))
        .collect();
    years.sort();
    assert_eq!(years, vec!["2023".to_string(), "2024".to_string()]);
}

#[tokio::test(flavor = "multi_thread")]
async fn set_partitioned_then_insert_same_session_partitions() {
    // P1 regression: SET PARTITIONED BY via the SQL hook, then INSERT in the SAME
    // session (the catalog was pinned BEFORE the spec existed) must still partition
    // — the write path resolves the spec at the current head, not the pinned snapshot.
    let (conn_str, table_id, _temp) = create_events_table_no_spec().await;
    let (ctx, catalog) = writable_catalog(&conn_str).await; // pins the pre-spec snapshot
    execute_ducklake_sql(
        &ctx,
        &catalog,
        "ALTER TABLE ducklake.main.events SET PARTITIONED BY (region)",
    )
    .await
    .unwrap();
    ctx.sql(INSERT_SQL).await.unwrap().collect().await.unwrap();

    let provider = SqliteMetadataProvider::new(&conn_str).await.unwrap();
    let snap = provider.get_current_snapshot().unwrap();
    let page = provider
        .get_table_file_metadata_page(table_id, snap, None, 4096)
        .unwrap();
    assert_eq!(
        page.len(),
        2,
        "same-session SET then INSERT must partition by region (one file per region)"
    );
    let live_pid = provider
        .get_partition_spec(table_id, snap)
        .unwrap()
        .expect("live spec present")
        .partition_id;
    for m in &page {
        assert_eq!(
            m.file.partition_id,
            Some(live_pid),
            "file must be stamped with the LIVE partition_id, never a retired one"
        );
        assert_eq!(m.file.partition_values.len(), 1);
    }
    let mut regions: Vec<String> = page
        .iter()
        .filter_map(|m| m.file.partition_values.first().and_then(|(_, v)| v.clone()))
        .collect();
    regions.sort();
    assert_eq!(regions, vec!["eu".to_string(), "us".to_string()]);
}

#[tokio::test(flavor = "multi_thread")]
async fn reset_partitioned_then_insert_same_session_is_unpartitioned() {
    // P1 regression: after RESET, a same-session INSERT must write ONE unpartitioned
    // file with NO partition_id — never a retired partition id from the pinned spec.
    let (conn_str, table_id, _temp) = create_events_table_no_spec().await;
    {
        let w = SqliteMetadataWriter::new_with_init(&conn_str)
            .await
            .unwrap();
        w.set_partition_spec(
            table_id,
            &[("region".to_string(), PartitionTransform::Identity)],
        )
        .unwrap();
    }
    let (ctx, catalog) = writable_catalog(&conn_str).await; // pinned where region-spec is live
    execute_ducklake_sql(
        &ctx,
        &catalog,
        "ALTER TABLE ducklake.main.events RESET PARTITIONED BY",
    )
    .await
    .unwrap();
    ctx.sql(INSERT_SQL).await.unwrap().collect().await.unwrap();

    let provider = SqliteMetadataProvider::new(&conn_str).await.unwrap();
    let snap = provider.get_current_snapshot().unwrap();
    let page = provider
        .get_table_file_metadata_page(table_id, snap, None, 4096)
        .unwrap();
    assert_eq!(
        page.len(),
        1,
        "after RESET, a same-session INSERT is unpartitioned"
    );
    assert!(
        page[0].file.partition_id.is_none(),
        "must not stamp a retired partition_id after RESET"
    );
    assert!(page[0].file.partition_values.is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn concurrent_reset_during_insert_conflicts() {
    // P1 (concurrency): a partition spec retired by a concurrent RESET/SET *after*
    // the insert plan captured it but *before* the insert commits must abort at the
    // commit-time fence — never stamp a retired partition_id into a committed file.
    let (conn_str, table_id, _temp) = create_events_table_no_spec().await;
    {
        let w = SqliteMetadataWriter::new_with_init(&conn_str)
            .await
            .unwrap();
        w.set_partition_spec(
            table_id,
            &[("region".to_string(), PartitionTransform::Identity)],
        )
        .unwrap();
    }
    let (ctx, _catalog) = writable_catalog(&conn_str).await;
    // Build the physical plan now: insert_into captures the LIVE spec (region).
    let plan = ctx
        .sql(INSERT_SQL)
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    // Concurrently retire that spec before the captured plan executes.
    {
        let w = SqliteMetadataWriter::new_with_init(&conn_str)
            .await
            .unwrap();
        w.reset_partition_spec(table_id).unwrap();
    }
    // Executing the captured plan must hit the fence and abort with a conflict.
    let result = datafusion::physical_plan::collect(plan, ctx.task_ctx()).await;
    let err = result.expect_err("insert against a concurrently-retired spec must conflict");
    let msg = err.to_string().to_lowercase();
    assert!(
        msg.contains("partition spec") || msg.contains("concurrent"),
        "expected a partition-spec conflict, got: {err}"
    );
    // Nothing committed: the tx rolled back, so no data files exist.
    let provider = SqliteMetadataProvider::new(&conn_str).await.unwrap();
    let snap = provider.get_current_snapshot().unwrap();
    let page = provider
        .get_table_file_metadata_page(table_id, snap, None, 4096)
        .unwrap();
    assert!(
        page.is_empty(),
        "a conflicting insert must not commit any data files"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn concurrent_set_during_unpartitioned_insert_conflicts() {
    // Inverse P1: an unpartitioned INSERT plan captured while the table had NO spec,
    // then a concurrent SET PARTITIONED BY makes it partitioned before the plan
    // commits. The commit must abort — never leave a partition_id-less file in a
    // now-partitioned table, and never silently re-lay-out the rows under a spec the
    // plan never saw. The caller re-plans against the new spec and retries.
    let (conn_str, table_id, _temp) = create_events_table_no_spec().await;
    let (ctx, _catalog) = writable_catalog(&conn_str).await;
    // Build the plan now: table is unpartitioned -> partition = None.
    let plan = ctx
        .sql(INSERT_SQL)
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    // Concurrently make the table partitioned.
    {
        let w = SqliteMetadataWriter::new_with_init(&conn_str)
            .await
            .unwrap();
        w.set_partition_spec(
            table_id,
            &[("region".to_string(), PartitionTransform::Identity)],
        )
        .unwrap();
    }
    // Executing the stale unpartitioned plan must hit the singular-commit fence.
    let result = datafusion::physical_plan::collect(plan, ctx.task_ctx()).await;
    let err = result.expect_err("unpartitioned insert into a now-partitioned table must conflict");
    let msg = err.to_string().to_lowercase();
    assert!(
        msg.contains("partition spec") || msg.contains("concurrent"),
        "expected a partition-spec conflict, got: {err}"
    );
    // Nothing committed.
    let provider = SqliteMetadataProvider::new(&conn_str).await.unwrap();
    let snap = provider.get_current_snapshot().unwrap();
    let page = provider
        .get_table_file_metadata_page(table_id, snap, None, 4096)
        .unwrap();
    assert!(
        page.is_empty(),
        "a conflicting unpartitioned insert must not commit any data files"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn empty_overwrite_truncates_partitioned_table() {
    // Regression: an empty INSERT OVERWRITE (0 rows) on a PARTITIONED table must
    // truncate (retire the prior generation) via the single-file path — it must NOT
    // trip the inverse fence. The 0-row truncate marker carries no partition_id, but
    // it also carries no data, so it cannot violate the live-spec invariant.
    let env = setup().await; // partitioned by (region, year(ts))
    write_ctx(&env.conn_str)
        .await
        .sql(INSERT_SQL)
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    // Sanity: 4 rows live across the partitions.
    {
        let p = SqliteMetadataProvider::new(&env.conn_str).await.unwrap();
        let s = p.get_current_snapshot().unwrap();
        assert_eq!(p.get_table_row_count(env.table_id, s).unwrap(), 4);
    }
    // Empty INSERT OVERWRITE (WHERE 1=2 → 0 rows) must truncate, not conflict.
    let ctx = write_ctx(&env.conn_str).await;
    ctx.sql(
        "INSERT OVERWRITE ducklake.main.events \
         SELECT * FROM (VALUES (1, 'us', TIMESTAMP '2023-01-15 10:00:00')) AS t(id, region, ts) \
         WHERE 1 = 2",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();
    // After truncate: 0 live rows.
    let p = SqliteMetadataProvider::new(&env.conn_str).await.unwrap();
    let s = p.get_current_snapshot().unwrap();
    assert_eq!(
        p.get_table_row_count(env.table_id, s).unwrap(),
        0,
        "empty INSERT OVERWRITE must truncate the partitioned table, not conflict"
    );
}

/// `PartitionSink::into_file_infos` zips each file's partition values onto the
/// concurrent upload results positionally, so a desynchronisation mislabels which
/// partition a file belongs to — no error, no change in row count.
///
/// SCOPE, stated precisely, because the previous version of this comment overstated
/// it. This is a SMOKE TEST for the partitioned concurrent path, not a proven
/// regression guard: flipping the shared helper to `buffer_unordered` does not
/// reliably fail it, because on a local store these uploads complete in submission
/// order however they are polled. The proven ordering guards are the two tests in
/// `concurrent_staged_upload_tests`, which do fail under that mutation and exercise
/// the same helper this path calls.
///
/// The single-run assertion below is a genuine but PARTIAL check: the fixture gives
/// the `us` partition 3000-row files and the others 600-row files, so it catches a
/// label that crossed between `us` and any other partition. A mislabel confined to
/// `{eu, apac, latam}` is invisible to it, since those three are indistinguishable
/// by row count.
///
#[tokio::test(flavor = "multi_thread")]
async fn partitioned_staged_uploads_are_order_stable_across_concurrency() {
    use arrow::array::{ArrayRef, Int32Array, RecordBatch, TimestampMicrosecondArray};
    use arrow::datatypes::{Field, Schema};
    use datafusion_ducklake::table_writer::DuckLakeTableWriter;

    /// One streaming partitioned write at the given upload concurrency. Returns the
    /// committed `(partition key index, partition value, row_id_start,
    /// record_count)` in data_file_id order — everything the ordering could corrupt.
    async fn write_at(concurrency: usize) -> Vec<(Option<i32>, Option<String>, Option<i64>, i64)> {
        let env = setup().await; // partitioned by (region, year(ts))
        let ts_type = DataType::Timestamp(TimeUnit::Microsecond, None);
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("region", DataType::Utf8, true),
            Field::new("ts", ts_type, true),
        ]));
        let y2023: i64 = 1_673_776_800_000_000;
        // Several partitions, each large enough to roll, so the sink finishes many
        // staged files and the uploads genuinely overlap.
        // Partition sizes are skewed so the uploads differ in duration, and so that
        // `us` files are distinguishable from the rest by row count — which is what
        // the assertion at the end relies on. The skew was NOT enough to make
        // completion order diverge on a local store; see the scope note on this test.
        let regions = ["us", "eu", "apac", "latam"];
        let mut next_id = 0i32;
        let batches: Vec<RecordBatch> = (0..40)
            .map(|b: i32| {
                let region = regions[(b as usize) % regions.len()];
                let count = if region == "us" {
                    3000
                } else {
                    60
                };
                let ids: Vec<i32> = (next_id..next_id + count).collect();
                next_id += count;
                let n = ids.len();
                RecordBatch::try_new(
                    schema.clone(),
                    vec![
                        Arc::new(Int32Array::from(ids)) as ArrayRef,
                        Arc::new(arrow::array::StringArray::from(vec![region; n])) as ArrayRef,
                        Arc::new(TimestampMicrosecondArray::from(vec![y2023; n])) as ArrayRef,
                    ],
                )
                .unwrap()
            })
            .collect();

        let writer = SqliteMetadataWriter::new_with_init(&env.conn_str)
            .await
            .unwrap();
        let object_store: Arc<dyn object_store::ObjectStore> =
            Arc::new(object_store::local::LocalFileSystem::new());
        let mut session = DuckLakeTableWriter::new(Arc::new(writer), object_store)
            .unwrap()
            .with_target_file_size(8 * 1024)
            .with_upload_concurrency(concurrency)
            .begin_write("main", "events", schema.as_ref(), WriteMode::Append)
            .unwrap();
        for batch in &batches {
            session.write_batch(batch).unwrap();
        }
        let result = session.finish().await.unwrap();
        assert_eq!(result.records_written, i64::from(next_id));
        assert!(
            result.files_written > 4,
            "the write must span several files per partition or concurrency is \
             untested, got {}",
            result.files_written
        );

        let pool = sqlx::sqlite::SqlitePool::connect(&env.conn_str)
            .await
            .unwrap();
        sqlx::query_as(
            // Both partition keys: the table is partitioned by (region, year(ts)),
            // and a positional mix-up could corrupt either. Checking only index 0
            // would miss a file whose year landed on the wrong row.
            "SELECT p.partition_key_index, p.partition_value, d.row_id_start,
                    d.record_count
               FROM ducklake_data_file d
               LEFT JOIN ducklake_file_partition_value p
                      ON p.data_file_id = d.data_file_id
              WHERE d.end_snapshot IS NULL
              ORDER BY d.data_file_id, p.partition_key_index",
        )
        .fetch_all(&pool)
        .await
        .unwrap()
    }

    let serial = write_at(1).await;
    let concurrent = write_at(4).await;

    // Absolute check within ONE run: a `us` file must hold 3000 rows and any other
    // partition's file 600. This catches a label that crossed between `us` and the
    // rest — which a 1-vs-4 comparison cannot, since a reordering applying to both
    // runs cancels out. It does NOT distinguish eu/apac/latam from each other.
    for rows in [&serial, &concurrent] {
        for (key_index, value, _start, count) in rows.iter() {
            if *key_index != Some(0) {
                continue;
            }
            let expected = if value.as_deref() == Some("us") {
                3000
            } else {
                600
            };
            assert_eq!(
                *count, expected,
                "partition {value:?} holds {expected}-row files by construction; a \
                 different count means this file's partition label came from another \
                 file's slot"
            );
        }
    }

    assert_eq!(
        serial, concurrent,
        "partition values, row_id_start and record_count must be identical at \
         concurrency 1 and 4 — a difference means the sink registered its files out \
         of order, so partition labels or row ids landed on the wrong file"
    );
}
