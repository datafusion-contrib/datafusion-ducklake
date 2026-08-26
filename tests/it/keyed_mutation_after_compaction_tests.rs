//! Keyed DELETE / UPDATE against data files that compaction has **rewritten**.
//!
//! A delete file's `pos` is a row's PHYSICAL index in the data file it targets,
//! and a rewrite leaves that meaningful: the rewritten file's rows sit at
//! `0..n-1` exactly as any other file's do. What a rewrite disturbs is the rowid
//! *sequence* — `rewrite_data_files` drops deleted rows, so the surviving rowids
//! carry holes, and the catalog records no `row_id_start` for such a file at all.
//! `rowid = row_id_start + position` is therefore not merely unreliable on a
//! rewritten file, it is not computable; positions are what a delete file
//! records, and they remain exact.
//!
//! Every test here pins that distinction down in the two ways it can break
//! silently: it asserts the exact surviving `(id, rowid)` rows, AND that the
//! resolved/written `pos` values are physical indices rather than rowids.
//!
//! How the second assertion is made differs, because a merge reads its sources
//! concurrently and does not fix which slot a row lands in. Where the physical
//! order IS determined — `delete_on_rewritten_file_whose_physical_order_is_reversed`,
//! under a sort order that runs position and rowid in opposite directions, and
//! `keyed_mutation_after_rewrite_with_rowid_holes`, where the rowid run has
//! gaps — the positions are asserted exactly, and those two are the decisive
//! coverage. The merge tests instead choose a row whose rowid cannot be a valid
//! slot of the file, so a rowid-keyed resolver cannot return a plausible answer
//! by chance, and check that the DELETE records the position the resolver
//! found.

#![cfg(all(feature = "write-sqlite", feature = "metadata-sqlite"))]

use std::sync::Arc;

use arrow::array::{Array, Int32Array, Int64Array, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::catalog::TableProvider;
use datafusion::logical_expr::Operator;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_expr::expressions::{BinaryExpr, col, lit};
use datafusion::prelude::*;
use object_store::local::LocalFileSystem;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use sqlx::sqlite::SqlitePool;
use sqlx::{AssertSqlSafe, Row};
use tempfile::TempDir;

use datafusion_ducklake::{
    CompactionResult, DuckLakeCatalog, DuckLakeTable, DuckLakeTableWriter, MergeOptions,
    MetadataWriter, NullOrder, RewriteOptions, SortDirection, SortField, SqliteMetadataProvider,
    SqliteMetadataWriter,
};

// ---------------------------------------------------------------------------
// Harness (mirrors `compaction_sqlite_tests` / `files_matching_tests`)
// ---------------------------------------------------------------------------

fn two_col_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("val", DataType::Int32, false),
    ]))
}

fn object_store() -> Arc<dyn object_store::ObjectStore> {
    Arc::new(LocalFileSystem::new())
}

fn db_url(temp: &TempDir) -> String {
    format!("sqlite:{}?mode=rwc", temp.path().join("test.db").display())
}

fn ro_url(temp: &TempDir) -> String {
    format!("sqlite:{}", temp.path().join("test.db").display())
}

async fn make_writer(temp: &TempDir) -> SqliteMetadataWriter {
    let data_path = temp.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();
    let writer = SqliteMetadataWriter::new_with_init(&db_url(temp))
        .await
        .unwrap();
    writer.set_data_path(data_path.to_str().unwrap()).unwrap();
    writer
}

fn batch(ids: Vec<i32>, vals: Vec<i32>) -> RecordBatch {
    RecordBatch::try_new(
        two_col_schema(),
        vec![Arc::new(Int32Array::from(ids)), Arc::new(Int32Array::from(vals))],
    )
    .unwrap()
}

/// Seed a fresh `main.t(id, val)` as one data file.
async fn seed(temp: &TempDir, ids: Vec<i32>, vals: Vec<i32>) {
    let writer = Arc::new(make_writer(temp).await);
    DuckLakeTableWriter::new(writer, object_store())
        .unwrap()
        .write_table("main", "t", &[batch(ids, vals)])
        .await
        .unwrap();
}

/// Append one more data file to `main.t`.
async fn append(temp: &TempDir, ids: Vec<i32>, vals: Vec<i32>) {
    let writer = Arc::new(SqliteMetadataWriter::new(&db_url(temp)).await.unwrap());
    DuckLakeTableWriter::new(writer, object_store())
        .unwrap()
        .append_table("main", "t", &[batch(ids, vals)])
        .await
        .unwrap();
}

async fn pool(temp: &TempDir) -> SqlitePool {
    SqlitePool::connect(&ro_url(temp)).await.unwrap()
}

async fn scalar_i64(p: &SqlitePool, sql: &str) -> i64 {
    sqlx::query(AssertSqlSafe(sql))
        .fetch_one(p)
        .await
        .unwrap()
        .try_get::<i64, _>(0)
        .unwrap()
}

async fn opt_i64(p: &SqlitePool, sql: &str) -> Option<i64> {
    sqlx::query(AssertSqlSafe(sql))
        .fetch_one(p)
        .await
        .unwrap()
        .try_get::<Option<i64>, _>(0)
        .unwrap()
}

/// Run a DML statement against a writable catalog and return its reported count.
/// A fresh catalog is opened per call so each statement sees the latest head.
async fn run_dml(temp: &TempDir, sql: &str) -> u64 {
    let writer = SqliteMetadataWriter::new(&db_url(temp)).await.unwrap();
    let provider = SqliteMetadataProvider::new(&db_url(temp)).await.unwrap();
    let catalog = DuckLakeCatalog::with_writer(Arc::new(provider), Arc::new(writer)).unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));
    let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
    batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .expect("DML yields a UInt64 count")
        .value(0)
}

/// Current live `(id, val)` rows, ascending.
async fn read_rows(temp: &TempDir) -> Vec<(i32, i32)> {
    let provider = SqliteMetadataProvider::new(&ro_url(temp)).await.unwrap();
    let catalog = DuckLakeCatalog::new(provider).unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));
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

/// Current live `(id, rowid)`, ascending by id — each row's DuckLake row-lineage
/// id. This is where a rewrite's rowid holes show up.
async fn read_id_rowid(temp: &TempDir) -> Vec<(i32, i64)> {
    let provider = SqliteMetadataProvider::new(&ro_url(temp)).await.unwrap();
    let catalog = DuckLakeCatalog::new(provider)
        .unwrap()
        .with_row_lineage(true);
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));
    let batches = ctx
        .sql("SELECT id, rowid FROM ducklake.main.t ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let mut out = Vec::new();
    for b in &batches {
        let ids = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        let rids = b.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
        for i in 0..b.num_rows() {
            out.push((ids.value(i), rids.value(i)));
        }
    }
    out
}

/// The `val` column of one data file, in PHYSICAL order — what compaction
/// actually laid down, as opposed to what a query returns.
fn file_values(temp: &TempDir, path: &str) -> Vec<i32> {
    let file =
        std::fs::File::open(temp.path().join("data").join("main").join("t").join(path)).unwrap();
    ParquetRecordBatchReaderBuilder::try_new(file)
        .unwrap()
        .build()
        .unwrap()
        .map(|batch| {
            let batch = batch.unwrap();
            batch
                .column_by_name("val")
                .unwrap()
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .values()
                .to_vec()
        })
        .collect::<Vec<_>>()
        .concat()
}

/// A read-only session plus the `DuckLakeTable` behind `ducklake.main.t`.
async fn open_table(temp: &TempDir) -> (SessionContext, Arc<dyn TableProvider>) {
    let provider = SqliteMetadataProvider::new(&ro_url(temp)).await.unwrap();
    let catalog = DuckLakeCatalog::new(provider).unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));
    let table = ctx
        .catalog("ducklake")
        .unwrap()
        .schema("main")
        .unwrap()
        .table("t")
        .await
        .unwrap()
        .unwrap();
    (ctx, table)
}

fn as_ducklake(table: &Arc<dyn TableProvider>) -> &DuckLakeTable {
    (table.as_ref() as &dyn std::any::Any)
        .downcast_ref::<DuckLakeTable>()
        .expect("provider is a DuckLakeTable")
}

/// `id = wanted`, against the table's logical column order — the expression a
/// keyed mutation passes to `files_matching` and `resolve_positions`.
fn id_equals(wanted: i32) -> Arc<dyn PhysicalExpr> {
    let schema = two_col_schema();
    Arc::new(BinaryExpr::new(
        col("id", schema.as_ref()).unwrap(),
        Operator::Eq,
        lit(wanted),
    ))
}

/// Every position recorded by every LIVE delete file of `main.t`, sorted — read
/// back through the crate's own delete-file reader, so this is what the read path
/// will actually mask.
async fn live_delete_positions(temp: &TempDir) -> Vec<i64> {
    let (ctx, provider) = open_table(temp).await;
    let table = as_ducklake(&provider);
    let state = ctx.state();
    let mut out: Vec<i64> = Vec::new();
    for tf in table.files().unwrap() {
        if let Some(delete_file) = &tf.delete_file {
            out.extend(
                table
                    .read_delete_file_positions(&state, delete_file)
                    .await
                    .unwrap(),
            );
        }
    }
    out.sort_unstable();
    out
}

/// Resolve the physical positions `predicate` matches in the single live data
/// file, asserting there is exactly one.
async fn resolve_in_only_live_file(temp: &TempDir, predicate: Arc<dyn PhysicalExpr>) -> Vec<i64> {
    let (ctx, provider) = open_table(temp).await;
    let table = as_ducklake(&provider);
    let files = table.files().unwrap();
    assert_eq!(files.len(), 1, "expected exactly one live data file");
    let mut positions: Vec<i64> = table
        .resolve_positions(&ctx.state(), &files[0].file, predicate)
        .await
        .unwrap()
        .into_iter()
        .collect();
    positions.sort_unstable();
    positions
}

async fn with_writable_table<F, Fut>(temp: &TempDir, op: F) -> CompactionResult
where
    F: FnOnce(DuckLakeTable, datafusion::execution::SessionState) -> Fut,
    Fut: std::future::Future<Output = CompactionResult>,
{
    let writer = SqliteMetadataWriter::new(&db_url(temp)).await.unwrap();
    let provider = SqliteMetadataProvider::new(&db_url(temp)).await.unwrap();
    let catalog = DuckLakeCatalog::with_writer(Arc::new(provider), Arc::new(writer)).unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));
    let provider = ctx
        .catalog("ducklake")
        .unwrap()
        .schema("main")
        .unwrap()
        .table("t")
        .await
        .unwrap()
        .unwrap();
    let table = (provider.as_ref() as &dyn std::any::Any)
        .downcast_ref::<DuckLakeTable>()
        .expect("provider is a DuckLakeTable")
        .clone();
    op(table, ctx.state()).await
}

async fn run_merge(temp: &TempDir, opts: MergeOptions) -> CompactionResult {
    with_writable_table(temp, |table, state| async move {
        table.merge_adjacent_files(&state, opts).await.unwrap()
    })
    .await
}

async fn run_rewrite(temp: &TempDir, opts: RewriteOptions) -> CompactionResult {
    with_writable_table(temp, |table, state| async move {
        table.rewrite_data_files(&state, opts).await.unwrap()
    })
    .await
}

// ---------------------------------------------------------------------------

/// A merge of files from several origin snapshots produces a **partial** file
/// (embedded rowid + embedded origin-snapshot column). A keyed DELETE against it
/// must resolve and record the row's physical position.
#[tokio::test(flavor = "multi_thread")]
async fn delete_after_merge_of_multi_origin_files() {
    let temp = TempDir::new().unwrap();
    seed(&temp, vec![1], vec![10]).await;
    append(&temp, vec![2], vec![20]).await;
    append(&temp, vec![3], vec![30]).await;

    let result = run_merge(&temp, MergeOptions::default()).await;
    assert_eq!(
        result,
        CompactionResult {
            files_processed: 3,
            files_created: 1,
            rows_written: 3,
        },
    );

    let p = pool(&temp).await;
    assert!(
        opt_i64(
            &p,
            "SELECT partial_max FROM ducklake_data_file WHERE end_snapshot IS NULL"
        )
        .await
        .is_some(),
        "a merge spanning three origin snapshots writes a partial file",
    );

    // Resolved rather than hardcoded: a merge reads its sources concurrently and
    // hands them on as they arrive, so which physical slot id = 2 lands in is not
    // fixed. What must hold is that the DELETE records the slot the resolver
    // found — that the delete is keyed on physical position at all. Whether that
    // position can be told apart from the rowid is settled by the sorted case
    // below, where the two deliberately run opposite.
    let position = resolve_in_only_live_file(&temp, id_equals(2)).await;
    assert_eq!(position.len(), 1, "id = 2 occurs once");

    assert_eq!(
        run_dml(&temp, "DELETE FROM ducklake.main.t WHERE id = 2").await,
        1,
    );
    assert_eq!(live_delete_positions(&temp).await, position);
    assert_eq!(read_rows(&temp).await, vec![(1, 10), (3, 30)]);
    assert_eq!(read_id_rowid(&temp).await, vec![(1, 0), (3, 2)]);
}

/// The decisive ordering case: a table sort order makes the rewritten file's
/// physical order the REVERSE of its rowid order, so position and rowid move in
/// opposite directions. Resolving by rowid would delete a row from the far side
/// of the file.
#[tokio::test(flavor = "multi_thread")]
async fn delete_on_rewritten_file_whose_physical_order_is_reversed() {
    let temp = TempDir::new().unwrap();
    seed(
        &temp,
        (1..=10).collect(),
        (1..=10).map(|v| v * 10).collect(),
    )
    .await;

    let p = pool(&temp).await;
    let table_id = scalar_i64(&p, "SELECT table_id FROM ducklake_table LIMIT 1").await;
    SqliteMetadataWriter::new(&db_url(&temp))
        .await
        .unwrap()
        .set_sort_spec(
            table_id,
            &[SortField::column(0, "val", SortDirection::Desc, NullOrder::NullsLast)],
        )
        .unwrap();

    // Delete half the rows, then rewrite: the output carries only the survivors,
    // laid out val-descending.
    assert_eq!(
        run_dml(&temp, "DELETE FROM ducklake.main.t WHERE id <= 5").await,
        5,
    );
    let result = run_rewrite(
        &temp,
        RewriteOptions {
            delete_threshold: 0.5,
            ..RewriteOptions::default()
        },
    )
    .await;
    assert_eq!(result.files_created, 1);
    assert_eq!(result.rows_written, 5);

    let live_path: String =
        sqlx::query_scalar("SELECT path FROM ducklake_data_file WHERE end_snapshot IS NULL")
            .fetch_one(&p)
            .await
            .unwrap();
    assert_eq!(
        file_values(&temp, &live_path),
        vec![100, 90, 80, 70, 60],
        "physical order is val-descending: ids 10,9,8,7,6 at positions 0..4",
    );
    assert_eq!(
        read_id_rowid(&temp).await,
        vec![(6, 5), (7, 6), (8, 7), (9, 8), (10, 9)],
        "rowid lineage survives the rewrite",
    );

    // id = 8 is at physical position 2 but carries rowid 7. Position and rowid
    // run in OPPOSITE directions here, so a rowid-based resolution would name
    // position 7 — past the end of a five-row file, or (with a longer file) a
    // completely different row.
    assert_eq!(
        resolve_in_only_live_file(&temp, id_equals(8)).await,
        vec![2]
    );

    assert_eq!(
        run_dml(&temp, "DELETE FROM ducklake.main.t WHERE id = 8").await,
        1,
    );
    assert_eq!(
        live_delete_positions(&temp).await,
        vec![2],
        "the written pos is the physical index (2), not the rowid (7)",
    );
    assert_eq!(
        read_id_rowid(&temp).await,
        vec![(6, 5), (7, 6), (9, 8), (10, 9)],
    );
}

/// Merging a PARTITIONED table merges within each partition, so a merged file
/// holds a non-contiguous rowid run and the catalog records no `row_id_start` for
/// it. This is the layout a partitioned, frequently appended table reaches in
/// practice.
#[tokio::test(flavor = "multi_thread")]
async fn delete_after_partitioned_merge() {
    use datafusion_ducklake::partition::PartitionTransform;
    use datafusion_ducklake::{ColumnDef, WriteMode};

    let temp = TempDir::new().unwrap();
    let writer = Arc::new(make_writer(&temp).await);
    let cols = vec![
        ColumnDef::from_arrow("id", &DataType::Int32, false).unwrap(),
        ColumnDef::from_arrow("val", &DataType::Int32, false).unwrap(),
    ];
    {
        let s = writer
            .begin_write_transaction("main", "t", &cols, WriteMode::Replace)
            .unwrap();
        writer
            .publish_snapshot(
                s.table_id,
                "main",
                "t",
                s.snapshot_id,
                WriteMode::Replace,
                s.base_snapshot_id,
                &cols,
                &s.field_ids,
            )
            .unwrap();
        writer
            .set_partition_spec(
                s.table_id,
                &[("val".to_string(), PartitionTransform::Identity)],
            )
            .unwrap();
    }

    // Three appends, each touching both partitions: one row into val=1 and one
    // into val=2. Rowids therefore alternate between the two partitions.
    for id in [1, 2, 3] {
        append(&temp, vec![id, id + 10], vec![1, 2]).await;
    }

    let p = pool(&temp).await;
    assert_eq!(
        scalar_i64(
            &p,
            "SELECT COUNT(*) FROM ducklake_data_file WHERE end_snapshot IS NULL"
        )
        .await,
        6,
        "three appends x two partitions",
    );

    let result = run_merge(&temp, MergeOptions::default()).await;
    assert_eq!(result.files_created, 2, "one merged file per partition");

    assert_eq!(
        scalar_i64(
            &p,
            "SELECT COUNT(*) FROM ducklake_data_file \
             WHERE end_snapshot IS NULL AND row_id_start IS NULL"
        )
        .await,
        2,
        "a merged partition file's rowids are non-contiguous, so it embeds them \
         and records no row_id_start",
    );

    assert_eq!(
        read_id_rowid(&temp).await,
        vec![(1, 0), (2, 2), (3, 4), (11, 1), (12, 3), (13, 5)],
        "partition val=1 holds rowids 0,2,4 and val=2 holds 1,3,5",
    );

    // id = 3 on purpose, not id = 2: its rowid is 4, which is OUTSIDE the three
    // slots (0..=2) of the val=1 partition file, so no position a correct
    // resolver can return is ever equal to it. id = 2 would not do -- its rowid
    // is 2, a valid slot, so a rowid-keyed resolver could return it and pass by
    // coincidence once the merge stopped fixing which slot a row lands in.
    //
    // Which slot id = 3 actually occupies does depend on the order its source
    // file's read completed, so the position is resolved rather than asserted.
    let (ctx, provider) = open_table(&temp).await;
    let table = as_ducklake(&provider);
    let matching = table.files_matching(&id_equals(3)).unwrap();
    let mut positions: Vec<i64> = table
        .resolve_positions(&ctx.state(), &matching[0].file, id_equals(3))
        .await
        .unwrap()
        .into_iter()
        .collect();
    positions.sort_unstable();
    assert_eq!(positions.len(), 1, "id = 3 occurs once");
    assert!(
        (0..3).contains(&positions[0]),
        "a physical position in the three-row partition file, never the rowid 4 \
         a rowid-keyed resolver would have returned: {positions:?}",
    );

    assert_eq!(
        run_dml(&temp, "DELETE FROM ducklake.main.t WHERE id = 3").await,
        1,
    );
    assert_eq!(live_delete_positions(&temp).await, positions);
    assert_eq!(
        read_id_rowid(&temp).await,
        vec![(1, 0), (2, 2), (11, 1), (12, 3), (13, 5)],
    );
}

/// The production state: a keyed-idempotent ingest always has deletes in flight
/// when compaction runs, so the file a later mutation lands on is a
/// `rewrite_data_files` output whose rowid run has HOLES where the applied
/// deletes were.
///
/// `merge_adjacent_files` cannot produce this — it declines files carrying live
/// deletes — so the rewrite path is the one that applies deletes, retires the
/// delete file, and materialises an embedded rowid column with gaps. Both a
/// DELETE and an UPDATE are exercised, landing on opposite sides of a hole.
#[tokio::test(flavor = "multi_thread")]
async fn keyed_mutation_after_rewrite_with_rowid_holes() {
    let temp = TempDir::new().unwrap();
    seed(&temp, (1..=7).collect(), (1..=7).map(|v| v * 10).collect()).await;
    let p = pool(&temp).await;

    // Two rows out of the middle: rowids 1 and 3 become holes.
    assert_eq!(
        run_dml(&temp, "DELETE FROM ducklake.main.t WHERE id IN (2, 4)").await,
        2,
    );
    assert_eq!(live_delete_positions(&temp).await, vec![1, 3]);

    // A low threshold so the 2/7 deleted fraction qualifies: the rewrite applies
    // the deletes and drops the delete file.
    let result = run_rewrite(
        &temp,
        RewriteOptions {
            delete_threshold: 0.01,
            ..RewriteOptions::default()
        },
    )
    .await;
    assert_eq!(
        result,
        CompactionResult {
            files_processed: 1,
            files_created: 1,
            rows_written: 5,
        },
    );
    assert_eq!(
        opt_i64(
            &p,
            "SELECT row_id_start FROM ducklake_data_file WHERE end_snapshot IS NULL"
        )
        .await,
        None,
        "the rewrite output records no row_id_start, so `rowid - row_id_start` \
         is not computable for it",
    );
    assert_eq!(
        scalar_i64(
            &p,
            "SELECT COUNT(*) FROM ducklake_delete_file WHERE end_snapshot IS NULL"
        )
        .await,
        0,
        "the rewrite applied and retired the delete file",
    );

    // Physical positions 0..4 map to rowids 0,2,4,5,6 — holes at 1 and 3.
    assert_eq!(
        read_id_rowid(&temp).await,
        vec![(1, 0), (3, 2), (5, 4), (6, 5), (7, 6)],
    );

    // THE assertion. id = 5 carries rowid 4 but sits at physical position 2,
    // because both holes precede it. Resolving by rowid would name position 4 —
    // which in this file is id 7.
    assert_eq!(
        resolve_in_only_live_file(&temp, id_equals(5)).await,
        vec![2],
        "physical position 2, not rowid 4 (position 4 here is id 7)",
    );

    assert_eq!(
        run_dml(&temp, "DELETE FROM ducklake.main.t WHERE id = 5").await,
        1,
    );
    assert_eq!(
        live_delete_positions(&temp).await,
        vec![2],
        "the written pos is the physical index, not the rowid",
    );
    assert_eq!(
        read_rows(&temp).await,
        vec![(1, 10), (3, 30), (6, 60), (7, 70)],
        "exactly id = 5 was removed",
    );
    assert_eq!(
        read_id_rowid(&temp).await,
        vec![(1, 0), (3, 2), (6, 5), (7, 6)],
    );

    // Now the other side of the holes: id = 1 is at physical position 0. Its
    // rowid must survive the update.
    assert_eq!(
        run_dml(&temp, "UPDATE ducklake.main.t SET val = 999 WHERE id = 1").await,
        1,
    );
    assert_eq!(
        read_rows(&temp).await,
        vec![(1, 999), (3, 30), (6, 60), (7, 70)],
        "exactly id = 1 was updated",
    );
    assert_eq!(
        read_id_rowid(&temp).await,
        vec![(1, 0), (3, 2), (6, 5), (7, 6)],
        "rowid lineage preserved through an update on a hole-bearing file",
    );
}
