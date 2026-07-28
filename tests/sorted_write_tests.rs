//! Sorted-table write tests (SQLite single-catalog).
//!
//! Exercises the full sort path: `SET/RESET SORTED BY` DDL + catalog persistence,
//! sort-on-insert via `SortExec`, and size-based file rollover producing several
//! contiguous, non-overlapping sorted files so a range filter prunes whole files.

#![cfg(all(feature = "write-sqlite", feature = "metadata-sqlite"))]

use std::sync::Arc;

use arrow::array::{Int32Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::datasource::MemTable;
use datafusion::execution::context::SessionConfig;
use datafusion::prelude::*;
use tempfile::TempDir;

use datafusion_ducklake::metadata_provider::MetadataProvider;
use datafusion_ducklake::sort::{NullOrder, SortDirection};
use datafusion_ducklake::{
    ColumnDef, DuckLakeCatalog, DuckLakeWriteOptions, MetadataWriter, SqliteMetadataProvider,
    SqliteMetadataWriter, WriteMode, execute_ducklake_sql,
};

struct Env {
    conn_str: String,
    table_id: i64,
    _temp: TempDir,
}

/// Create a writable SQLite catalog with an empty `events(id, val)` table.
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

    let cols = vec![
        ColumnDef::from_arrow("id", &DataType::Int32, false).unwrap(),
        ColumnDef::from_arrow("val", &DataType::Int32, true).unwrap(),
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

    Env {
        conn_str,
        table_id: s.table_id,
        _temp: temp,
    }
}

/// A writable context whose writer rolls over at `target_file_size` and whose
/// scans emit small batches (so a single INSERT yields several batches → several
/// rolled files).
async fn write_ctx(conn_str: &str, target_file_size: usize, batch_size: usize) -> SessionContext {
    let writer = SqliteMetadataWriter::new_with_init(conn_str).await.unwrap();
    let provider = SqliteMetadataProvider::new(conn_str).await.unwrap();
    let options = DuckLakeWriteOptions {
        target_file_size: Some(target_file_size),
        ..Default::default()
    };
    let catalog = DuckLakeCatalog::with_writer(Arc::new(provider), Arc::new(writer))
        .unwrap()
        .with_write_options(options);
    let ctx = SessionContext::new_with_config(SessionConfig::new().with_batch_size(batch_size));
    ctx.register_catalog("ducklake", Arc::new(catalog));
    ctx
}

/// A concrete writable catalog handle for running `SET/RESET SORTED BY` DDL via
/// `execute_ducklake_sql` (same underlying catalog DB as the query context).
async fn ddl_catalog(conn_str: &str) -> DuckLakeCatalog {
    let writer = SqliteMetadataWriter::new_with_init(conn_str).await.unwrap();
    let provider = SqliteMetadataProvider::new(conn_str).await.unwrap();
    DuckLakeCatalog::with_writer(Arc::new(provider), Arc::new(writer)).unwrap()
}

async fn read_ctx(conn_str: &str) -> SessionContext {
    let provider = SqliteMetadataProvider::new(conn_str).await.unwrap();
    let catalog = DuckLakeCatalog::new(provider).unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));
    ctx
}

/// Register `src(id, val)` with `n` rows whose `val` is a shuffled permutation of
/// `0..n` (so an insert must actually sort to become ordered).
fn register_shuffled_source(ctx: &SessionContext, n: i32) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("val", DataType::Int32, true),
    ]));
    let ids: Vec<i32> = (0..n).collect();
    // A coprime stride gives a deterministic permutation of 0..n (n and 7919 are
    // coprime for our n), so vals are unique and out of order without randomness.
    let vals: Vec<i32> = (0..n)
        .map(|i| ((i as i64 * 7919) % n as i64) as i32)
        .collect();
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(ids)), Arc::new(Int32Array::from(vals))],
    )
    .unwrap();
    let table = MemTable::try_new(schema, vec![vec![batch]]).unwrap();
    ctx.register_table("src", Arc::new(table)).unwrap();
}

/// Number of distinct `.parquet` files a query's physical plan scans.
async fn files_scanned(ctx: &SessionContext, sql: &str) -> usize {
    let plan = ctx
        .sql(sql)
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    let display = datafusion::physical_plan::displayable(plan.as_ref())
        .indent(true)
        .to_string();
    display.matches(".parquet").count()
}

async fn live_file_count(conn_str: &str, table_id: i64) -> usize {
    let provider = SqliteMetadataProvider::new(conn_str).await.unwrap();
    let snapshot = provider.get_current_snapshot().unwrap();
    provider
        .get_table_file_metadata_page(table_id, snapshot, None, 4096)
        .unwrap()
        .len()
}

/// The low-level bulk write (`write_rows`, and so `write_table`/`append_table`) must
/// apply the table's sort order itself. It has no plan to carry a `SortExec`, so
/// without this a caller outside SQL gets rolled files whose ranges OVERLAP — the
/// files exist but no range filter can skip any of them, silently losing the entire
/// point of `SET SORTED BY`.
///
/// Asserts the file-level property that matters: each rolled file's `val` range is
/// disjoint from every other's. A per-file sort could not achieve this (a file's
/// min/max does not depend on row order) — only a global sort before rolling can.
#[tokio::test(flavor = "multi_thread")]
async fn low_level_bulk_write_sorts_and_yields_non_overlapping_files() {
    use datafusion_ducklake::table_writer::DuckLakeTableWriter;

    let env = setup().await;
    let dl = ddl_catalog(&env.conn_str).await;
    let ddl_ctx = read_ctx(&env.conn_str).await;
    execute_ducklake_sql(
        &ddl_ctx,
        &dl,
        "ALTER TABLE ducklake.main.events SET SORTED BY (val)",
    )
    .await
    .unwrap();

    // Shuffled input, fed as many small batches so rollover produces several files.
    let n = 4000i32;
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("val", DataType::Int32, true),
    ]));
    let batches: Vec<RecordBatch> = (0..n)
        .step_by(200)
        .map(|start| {
            let ids: Vec<i32> = (start..(start + 200).min(n)).collect();
            let vals: Vec<i32> = ids
                .iter()
                .map(|i| ((*i as i64 * 7919) % n as i64) as i32)
                .collect();
            RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(Int32Array::from(ids)), Arc::new(Int32Array::from(vals))],
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
        // Small target so this write rolls into several files.
        .with_target_file_size(8 * 1024);
    let result = table_writer
        .append_table("main", "events", &batches)
        .await
        .unwrap();
    assert_eq!(result.records_written, n as i64);
    assert!(
        result.files_written > 1,
        "the write must roll into several files for this to be meaningful, got {}",
        result.files_written
    );

    // Every file's [min, max] on the sort key must be disjoint from every other's.
    let provider = SqliteMetadataProvider::new(&env.conn_str).await.unwrap();
    let snapshot = provider.get_current_snapshot().unwrap();
    let page = provider
        .get_table_file_metadata_page(env.table_id, snapshot, None, 4096)
        .unwrap();
    let val_column_id = 2i64; // events(id, val) -> column_ids 1, 2
    let mut ranges: Vec<(i64, i64)> = page
        .iter()
        .map(|meta| {
            let stat = meta
                .column_statistics
                .iter()
                .find(|s| s.column_id == val_column_id)
                .expect("val stats present");
            (
                stat.min_value.as_deref().unwrap().parse::<i64>().unwrap(),
                stat.max_value.as_deref().unwrap().parse::<i64>().unwrap(),
            )
        })
        .collect();
    ranges.sort();
    for pair in ranges.windows(2) {
        assert!(
            pair[0].1 < pair[1].0,
            "file ranges must not overlap after a sorted bulk write: {:?} then {:?}",
            pair[0],
            pair[1]
        );
    }

    // And a range filter therefore skips files.
    let ctx = read_ctx(&env.conn_str).await;
    let scanned = files_scanned(&ctx, "SELECT id FROM ducklake.main.events WHERE val < 100").await;
    assert!(
        scanned < ranges.len(),
        "a range filter must prune files: scanned {scanned} of {}",
        ranges.len()
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn set_and_reset_sorted_by_persists_in_catalog() {
    let env = setup().await;
    let ctx = write_ctx(&env.conn_str, 1 << 20, 8192).await;
    let dl = ddl_catalog(&env.conn_str).await;

    execute_ducklake_sql(
        &ctx,
        &dl,
        "ALTER TABLE ducklake.main.events SET SORTED BY (val DESC NULLS FIRST, id)",
    )
    .await
    .unwrap();

    let provider = SqliteMetadataProvider::new(&env.conn_str).await.unwrap();
    let snapshot = provider.get_current_snapshot().unwrap();
    let spec = provider
        .get_sort_spec(env.table_id, snapshot)
        .unwrap()
        .expect("sort spec present after SET");
    assert_eq!(spec.fields.len(), 2);
    assert_eq!(spec.fields[0].expression, "val");
    assert_eq!(spec.fields[0].direction, SortDirection::Desc);
    assert_eq!(spec.fields[0].null_order, NullOrder::NullsFirst);
    assert_eq!(spec.fields[1].expression, "id");
    assert_eq!(spec.fields[1].direction, SortDirection::Asc);
    // Default null order for ASC is NULLS LAST (matching DuckDB).
    assert_eq!(spec.fields[1].null_order, NullOrder::NullsLast);

    execute_ducklake_sql(
        &ctx,
        &dl,
        "ALTER TABLE ducklake.main.events RESET SORTED BY",
    )
    .await
    .unwrap();
    let provider = SqliteMetadataProvider::new(&env.conn_str).await.unwrap();
    let snapshot = provider.get_current_snapshot().unwrap();
    assert!(
        provider
            .get_sort_spec(env.table_id, snapshot)
            .unwrap()
            .is_none(),
        "sort spec cleared after RESET"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn set_sorted_by_rejects_non_column_and_unknown_column() {
    let env = setup().await;
    let ctx = write_ctx(&env.conn_str, 1 << 20, 8192).await;
    let dl = ddl_catalog(&env.conn_str).await;

    // A non-bare-column sort key is rejected at parse time (v1 scope).
    let err = execute_ducklake_sql(
        &ctx,
        &dl,
        "ALTER TABLE ducklake.main.events SET SORTED BY (date_trunc('day', val))",
    )
    .await
    .unwrap_err();
    assert!(
        err.to_string().to_lowercase().contains("sort key"),
        "expected non-column rejection, got: {err}"
    );

    // An unknown column is rejected at SET time by column validation.
    let err = execute_ducklake_sql(
        &ctx,
        &dl,
        "ALTER TABLE ducklake.main.events SET SORTED BY (nope)",
    )
    .await
    .unwrap_err();
    assert!(
        err.to_string().contains("nope") || err.to_string().contains("no live column"),
        "expected unknown-column rejection, got: {err}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn sort_plus_rollover_creates_multiple_files_and_prunes_range() {
    let env = setup().await;

    // Small target + small batches so one INSERT of 2000 rows rolls into many files.
    let ctx = write_ctx(&env.conn_str, 2048, 200).await;
    let dl = ddl_catalog(&env.conn_str).await;
    execute_ducklake_sql(
        &ctx,
        &dl,
        "ALTER TABLE ducklake.main.events SET SORTED BY (val)",
    )
    .await
    .unwrap();

    register_shuffled_source(&ctx, 2000);
    let inserted = ctx
        .sql("INSERT INTO ducklake.main.events SELECT id, val FROM src")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let count = inserted[0]
        .column(0)
        .as_any()
        .downcast_ref::<arrow::array::UInt64Array>()
        .unwrap()
        .value(0);
    assert_eq!(count, 2000);

    // Rollover produced several files.
    let files = live_file_count(&env.conn_str, env.table_id).await;
    assert!(files > 1, "expected multiple rolled files, got {files}");

    // Correctness: all rows read back.
    let rctx = read_ctx(&env.conn_str).await;
    let total: usize = rctx
        .sql("SELECT val FROM ducklake.main.events")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap()
        .iter()
        .map(|b| b.num_rows())
        .sum();
    assert_eq!(total, 2000);

    // A narrow range on the sort column must prune most files: because the data
    // is globally sorted by `val` and rolled into contiguous files, only the few
    // files covering [1990, 2000) survive.
    let scanned = files_scanned(
        &rctx,
        "SELECT id FROM ducklake.main.events WHERE val >= 1990",
    )
    .await;
    assert!(
        scanned < files,
        "range filter should prune sorted files: scanned {scanned} of {files}"
    );

    // And the surviving rows are correct.
    let hi: usize = rctx
        .sql("SELECT id FROM ducklake.main.events WHERE val >= 1990")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap()
        .iter()
        .map(|b| b.num_rows())
        .sum();
    assert_eq!(hi, 10, "vals 1990..=1999");
}

#[tokio::test(flavor = "multi_thread")]
async fn per_file_ranges_are_ordered_and_non_overlapping_when_sorted() {
    let env = setup().await;
    let ctx = write_ctx(&env.conn_str, 2048, 200).await;
    let dl = ddl_catalog(&env.conn_str).await;
    execute_ducklake_sql(
        &ctx,
        &dl,
        "ALTER TABLE ducklake.main.events SET SORTED BY (val)",
    )
    .await
    .unwrap();
    register_shuffled_source(&ctx, 2000);
    ctx.sql("INSERT INTO ducklake.main.events SELECT id, val FROM src")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    // Find the catalog column_id of `val`.
    let provider = SqliteMetadataProvider::new(&env.conn_str).await.unwrap();
    let snapshot = provider.get_current_snapshot().unwrap();
    let columns = provider
        .get_table_structure(env.table_id, snapshot)
        .unwrap();
    let val_col_id = columns
        .iter()
        .find(|c| c.column_name == "val")
        .map(|c| c.column_id)
        .expect("val column");

    // Each file's [min,max] on `val`, sorted by min; a globally sorted+rolled
    // layout has strictly increasing, non-overlapping per-file ranges.
    let page = provider
        .get_table_file_metadata_page(env.table_id, snapshot, None, 4096)
        .unwrap();
    let mut ranges: Vec<(i64, i64)> = page
        .iter()
        .map(|m| {
            let s = m
                .column_statistics
                .iter()
                .find(|s| s.column_id == val_col_id)
                .expect("val stats");
            (
                s.min_value.as_ref().unwrap().parse::<i64>().unwrap(),
                s.max_value.as_ref().unwrap().parse::<i64>().unwrap(),
            )
        })
        .collect();
    ranges.sort();
    for w in ranges.windows(2) {
        assert!(
            w[0].1 < w[1].0,
            "file ranges must not overlap: {:?} then {:?}",
            w[0],
            w[1]
        );
    }
    assert_eq!(ranges.first().unwrap().0, 0);
    assert_eq!(ranges.last().unwrap().1, 1999);
}
