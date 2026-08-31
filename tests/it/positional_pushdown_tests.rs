//! Correctness tests for *positional* scan paths now that physical row positions
//! come from the parquet reader rather than being synthesized above the scan.
//!
//! Reader-derived positions are absolute, so DataFusion is free to prune row
//! groups, select rows, and split one file across partitions on these paths —
//! all of which the previous design had to refuse. That freedom is the point of
//! the change, and it is also what these tests guard: every one of them asserts
//! query RESULTS under a configuration that exercises pruning and parallelism
//! together, because a plan-shape assertion alone would pass whether or not the
//! positions survived it.
//!
//! The NaN cases are the sharp edge. Parquet footer bounds for float columns
//! exclude NaN while DataFusion evaluates `NaN > C` as true (IEEE totalOrder),
//! so a predicate reaching the reader can prune a row group that still holds
//! matching NaN rows. `NanPruningBarrierExec` prevents that, and these paths did
//! not all have it.

#![cfg(all(feature = "write-sqlite", feature = "metadata-sqlite"))]

use std::sync::Arc;

use arrow::array::{Array, Float64Array, Int32Array, Int64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::config::ConfigOptions;
use datafusion::prelude::*;
use object_store::local::LocalFileSystem;
use tempfile::TempDir;

use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeTableWriter, MetadataWriter, SqliteMetadataProvider,
    SqliteMetadataWriter, register_ducklake_functions,
};

/// Rows per file: enough to span several parquet row groups once the writer's
/// row-group size is small, so pruning and byte-range splitting both have
/// something to work with.
const ROWS: i32 = 60_000;

/// Rows per parquet row group in the fixtures. Small enough that `ROWS` spans
/// several, so pruning and splitting are real rather than nominal.
const ROWS_PER_ROW_GROUP: usize = 5_000;

/// Session config that splits single files across partitions, so every test
/// runs the parallel case rather than the incidental single-partition one.
fn split_ctx() -> SessionContext {
    let mut cfg = ConfigOptions::new();
    cfg.execution.target_partitions = 8;
    cfg.optimizer.repartition_file_scans = true;
    cfg.optimizer.repartition_file_min_size = 1;
    SessionContext::new_with_config(SessionConfig::from(cfg))
}

async fn new_writer(temp: &TempDir) -> SqliteMetadataWriter {
    let db_path = temp.path().join("test.db");
    let data_path = temp.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();
    let conn = format!("sqlite:{}?mode=rwc", db_path.display());
    let writer = SqliteMetadataWriter::new_with_init(&conn).await.unwrap();
    writer.set_data_path(data_path.to_str().unwrap()).unwrap();
    writer
}

/// A context over the catalog's current snapshot. `lineage` opts into the
/// synthetic `rowid` column, which is what puts a scan on the positional path
/// even with no delete files present.
async fn ctx_for(temp: &TempDir, lineage: bool, writable: bool) -> SessionContext {
    let conn = format!("sqlite:{}", temp.path().join("test.db").display());
    let catalog = if writable {
        let conn_rw = format!("sqlite:{}?mode=rwc", temp.path().join("test.db").display());
        let writer = SqliteMetadataWriter::new(&conn_rw).await.unwrap();
        let provider = Arc::new(SqliteMetadataProvider::new(&conn).await.unwrap());
        DuckLakeCatalog::with_writer(provider, Arc::new(writer)).unwrap()
    } else {
        DuckLakeCatalog::new(SqliteMetadataProvider::new(&conn).await.unwrap()).unwrap()
    }
    .with_row_lineage(lineage);
    let ctx = split_ctx();
    ctx.register_catalog("ducklake", Arc::new(catalog));
    // The table functions need their own handle on the catalog metadata.
    register_ducklake_functions(
        &ctx,
        Arc::new(SqliteMetadataProvider::new(&conn).await.unwrap()),
    );
    ctx
}

/// Record a column rename the way a real catalog does: close the existing
/// generation and open another under the SAME column id, so the id is the stable
/// key and the name is not. Parquet files keep the old physical name, which is
/// exactly what populates `name_mapping` on read.
async fn rename_column(temp: &TempDir, from: &str, to: &str, column_type: &str) {
    let url = format!("sqlite:{}", temp.path().join("test.db").display());
    let pool = sqlx::sqlite::SqlitePool::connect(&url).await.unwrap();
    let next: i64 = sqlx::query_scalar("SELECT MAX(snapshot_id) + 1 FROM ducklake_snapshot")
        .fetch_one(&pool)
        .await
        .unwrap();
    let (table_id, column_id, column_order): (i64, i64, i64) = sqlx::query_as(
        "SELECT table_id, column_id, column_order FROM ducklake_column
         WHERE column_name = ? AND end_snapshot IS NULL",
    )
    .bind(from)
    .fetch_one(&pool)
    .await
    .unwrap();
    sqlx::query(
        "INSERT INTO ducklake_snapshot (snapshot_id, snapshot_time, schema_version)
         SELECT ?, snapshot_time, schema_version FROM ducklake_snapshot
         ORDER BY snapshot_id DESC LIMIT 1",
    )
    .bind(next)
    .execute(&pool)
    .await
    .unwrap();
    sqlx::query(
        "UPDATE ducklake_column SET end_snapshot = ? WHERE column_id = ? AND end_snapshot IS NULL",
    )
    .bind(next)
    .bind(column_id)
    .execute(&pool)
    .await
    .unwrap();
    sqlx::query(
        "INSERT INTO ducklake_column
           (column_id, table_id, column_name, column_type, column_order, begin_snapshot, nulls_allowed)
         VALUES (?, ?, ?, ?, ?, ?, false)",
    )
    .bind(column_id)
    .bind(table_id)
    .bind(to)
    .bind(column_type)
    .bind(column_order)
    .bind(next)
    .execute(&pool)
    .await
    .unwrap();
    pool.close().await;
}

fn float_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("x", DataType::Float64, true),
    ]))
}

/// Seed `t(id, x)` with `ROWS` rows where `x == id`, and a single NaN row at
/// `id = nan_at`. Written in several batches so the file spans row groups.
async fn seed_float_table(temp: &TempDir, nan_at: Option<i32>) {
    let writer = Arc::new(new_writer(temp).await);
    // Small row groups on purpose. The parquet default is 1Mi rows, which would
    // put this whole fixture in ONE row group — and then neither row-group
    // pruning nor byte-range splitting has anything to bite on, so tests that
    // claim to exercise them would pass vacuously.
    let table_writer = DuckLakeTableWriter::new(writer, Arc::new(LocalFileSystem::new()))
        .unwrap()
        .with_max_row_group_rows(ROWS_PER_ROW_GROUP);

    let mut batches = Vec::new();
    let chunk = 10_000;
    let mut start = 0;
    while start < ROWS {
        let end = (start + chunk).min(ROWS);
        let ids: Vec<i32> = (start..end).collect();
        let xs: Vec<f64> = ids
            .iter()
            .map(|&i| {
                if Some(i) == nan_at {
                    f64::NAN
                } else {
                    f64::from(i)
                }
            })
            .collect();
        batches.push(
            RecordBatch::try_new(
                float_schema(),
                vec![Arc::new(Int32Array::from(ids)) as _, Arc::new(Float64Array::from(xs)) as _],
            )
            .unwrap(),
        );
        start = end;
    }
    table_writer
        .write_table("main", "t", &batches)
        .await
        .unwrap();
}

async fn rows(ctx: &SessionContext, sql: &str) -> Vec<RecordBatch> {
    ctx.sql(sql).await.unwrap().collect().await.unwrap()
}

async fn row_count(ctx: &SessionContext, sql: &str) -> usize {
    rows(ctx, sql).await.iter().map(|b| b.num_rows()).sum()
}

async fn plan_of(ctx: &SessionContext, sql: &str) -> String {
    let batches = rows(ctx, &format!("EXPLAIN {sql}")).await;
    datafusion::arrow::util::pretty::pretty_format_batches(&batches)
        .unwrap()
        .to_string()
}

// ---------------------------------------------------------------------------
// NaN safety on the paths that gained filter pushdown
// ---------------------------------------------------------------------------

/// `build_exec_for_file_with_rowid`: a rowid scan used to refuse all pushdown,
/// so it skipped the NaN barrier. Now that predicates reach the reader, the
/// barrier must be there — otherwise the footer max (a finite value, since
/// parquet bounds exclude NaN) prunes away the row group holding the NaN row.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn nan_rows_survive_a_filtered_rowid_scan() {
    let temp = TempDir::new().unwrap();
    seed_float_table(&temp, Some(12_345)).await;
    let ctx = ctx_for(&temp, true, false).await;

    // NaN sorts above every finite value, so it matches a bound every real
    // value fails. Exactly one row must come back: the NaN row.
    let batches = rows(
        &ctx,
        "SELECT rowid, id FROM ducklake.main.t WHERE x > 1e300",
    )
    .await;
    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total, 1, "the NaN row must not be pruned away");

    let ids: Vec<i32> = batches
        .iter()
        .flat_map(|b| {
            let c = b.column(1).as_any().downcast_ref::<Int32Array>().unwrap();
            (0..b.num_rows()).map(|i| c.value(i)).collect::<Vec<_>>()
        })
        .collect();
    assert_eq!(ids, vec![12_345]);

    // And its rowid is still the true physical position, not a count of the
    // rows that survived pruning.
    let rowids: Vec<i64> = batches
        .iter()
        .flat_map(|b| {
            let c = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
            (0..b.num_rows()).map(|i| c.value(i)).collect::<Vec<_>>()
        })
        .collect();
    assert_eq!(
        rowids,
        vec![12_345],
        "rowid must be row_id_start + physical position, unaffected by pruning"
    );

    assert!(
        plan_of(
            &ctx,
            "SELECT rowid, id FROM ducklake.main.t WHERE x > 1e300"
        )
        .await
        .contains("NanPruningBarrierExec"),
        "the rowid path must install the NaN barrier"
    );
}

/// `build_exec_for_file_with_deletes` had no NaN barrier on either branch. Its
/// plain-scan branch has always accepted pushed predicates, so this was a live
/// gap; its positional branch gains pushdown with this change.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn nan_rows_survive_a_filtered_scan_with_deletes() {
    let temp = TempDir::new().unwrap();
    seed_float_table(&temp, Some(12_345)).await;

    // Delete a row far from the NaN row, so a delete file exists and the scan
    // takes the positional branch, but the NaN row itself survives.
    let write_ctx = ctx_for(&temp, false, true).await;
    write_ctx
        .sql("DELETE FROM ducklake.main.t WHERE id = 7")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    let ctx = ctx_for(&temp, false, false).await;
    assert_eq!(
        row_count(&ctx, "SELECT id FROM ducklake.main.t WHERE x > 1e300").await,
        1,
        "the NaN row must not be pruned away on a scan with deletes"
    );
    assert_eq!(
        row_count(&ctx, "SELECT id FROM ducklake.main.t WHERE id = 7").await,
        0,
        "the deleted row must still be gone"
    );
    assert_eq!(
        row_count(&ctx, "SELECT id FROM ducklake.main.t").await,
        ROWS as usize - 1
    );

    assert!(
        plan_of(&ctx, "SELECT id FROM ducklake.main.t WHERE x > 1e300")
            .await
            .contains("NanPruningBarrierExec"),
        "the delete path must install the NaN barrier"
    );
}

/// A NaN-free float column keeps full pruning: the barrier is installed only
/// when a scanned file's NaN state is unknown or positive, so this must NOT
/// suppress pushdown.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn nan_free_float_column_keeps_reader_pruning() {
    let temp = TempDir::new().unwrap();
    seed_float_table(&temp, None).await;
    let ctx = ctx_for(&temp, true, false).await;

    let sql = "SELECT rowid, id FROM ducklake.main.t WHERE x > 59000";
    assert_eq!(row_count(&ctx, sql).await, (ROWS - 59_001) as usize);

    let plan = plan_of(&ctx, sql).await;
    assert!(
        !plan.contains("NanPruningBarrierExec"),
        "a known NaN-free column must not be barriered:\n{plan}"
    );
    assert!(
        plan.contains("predicate="),
        "the predicate must reach the reader:\n{plan}"
    );
}

// ---------------------------------------------------------------------------
// Pruning and parallelism actually happen, and positions survive both
// ---------------------------------------------------------------------------

/// The combination the previous design could not express: a positional scan
/// that is split across partitions AND prunes with a pushed predicate. Every
/// surviving rowid must still equal its physical position.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn rowids_survive_pruning_and_splitting_together() {
    let temp = TempDir::new().unwrap();
    seed_float_table(&temp, None).await;
    let ctx = ctx_for(&temp, true, false).await;

    let sql = "SELECT rowid, id FROM ducklake.main.t WHERE id >= 55000 ORDER BY id";
    let batches = rows(&ctx, sql).await;
    let mut pairs: Vec<(i64, i32)> = Vec::new();
    for b in &batches {
        let r = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        let i = b.column(1).as_any().downcast_ref::<Int32Array>().unwrap();
        for k in 0..b.num_rows() {
            pairs.push((r.value(k), i.value(k)));
        }
    }
    assert_eq!(pairs.len(), (ROWS - 55_000) as usize);
    // `x == id` and rows were written in id order, so physical position == id.
    for (rowid, id) in pairs {
        assert_eq!(
            rowid,
            i64::from(id),
            "rowid must equal physical position even after pruning + splitting"
        );
    }

    let plan = plan_of(&ctx, sql).await;
    assert!(plan.contains("predicate="), "expected pushdown:\n{plan}");
}

/// Deletes are matched by absolute position, so pruning rows in the reader
/// cannot shift which rows the delete file names.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn deletes_stay_correct_under_reader_pruning() {
    let temp = TempDir::new().unwrap();
    seed_float_table(&temp, None).await;

    let write_ctx = ctx_for(&temp, false, true).await;
    write_ctx
        .sql("DELETE FROM ducklake.main.t WHERE id IN (0, 1, 55555, 59999)")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    let ctx = ctx_for(&temp, false, false).await;
    // A predicate that prunes most of the file, spanning two deleted rows.
    let ids: Vec<i32> = rows(
        &ctx,
        "SELECT id FROM ducklake.main.t WHERE id >= 55555 AND id <= 55557 ORDER BY id",
    )
    .await
    .iter()
    .flat_map(|b| {
        let c = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        (0..b.num_rows()).map(|i| c.value(i)).collect::<Vec<_>>()
    })
    .collect();
    assert_eq!(
        ids,
        vec![55_556, 55_557],
        "55555 was deleted; pruning must not shift which positions the delete names"
    );

    assert_eq!(
        row_count(&ctx, "SELECT id FROM ducklake.main.t").await,
        ROWS as usize - 4
    );
}

/// A `LIMIT` must never sink below the delete filter, where it would count rows
/// the delete file removes. `DeleteFilterExec` opts out of limit pushdown; this
/// pins the end-to-end consequence.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn limit_is_applied_after_deletes() {
    let temp = TempDir::new().unwrap();
    seed_float_table(&temp, None).await;

    let write_ctx = ctx_for(&temp, false, true).await;
    write_ctx
        .sql("DELETE FROM ducklake.main.t WHERE id < 100")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    let ctx = ctx_for(&temp, false, false).await;
    assert_eq!(
        row_count(&ctx, "SELECT id FROM ducklake.main.t LIMIT 10").await,
        10,
        "LIMIT must see post-delete rows, not the first 10 physical rows"
    );
}

// ---------------------------------------------------------------------------
// The internal position column is not a reserved name
// ---------------------------------------------------------------------------

/// DuckLake reserves no column names — official DuckLake validates none — so a
/// table may legitimately have a column called `__ducklake_row_pos`. The
/// internal position column must not shadow it, and neither must be read in
/// place of the other.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn a_user_column_named_like_the_internal_position_column_is_safe() {
    let temp = TempDir::new().unwrap();
    let writer = Arc::new(new_writer(&temp).await);
    let table_writer = DuckLakeTableWriter::new(writer, Arc::new(LocalFileSystem::new())).unwrap();

    // The user's column holds values deliberately unrelated to physical
    // position, so binding the wrong one is detectable.
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("__ducklake_row_pos", DataType::Int64, true),
    ]));
    let ids: Vec<i32> = (0..1_000).collect();
    let mine: Vec<i64> = ids.iter().map(|&i| i64::from(i) * -7).collect();
    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(Int32Array::from(ids)) as _, Arc::new(Int64Array::from(mine)) as _],
    )
    .unwrap();
    table_writer
        .write_table("main", "t", &[batch])
        .await
        .unwrap();

    let ctx = ctx_for(&temp, true, false).await;
    let batches = rows(
        &ctx,
        "SELECT rowid, id, __ducklake_row_pos FROM ducklake.main.t ORDER BY id LIMIT 5",
    )
    .await;
    let mut seen = Vec::new();
    for b in &batches {
        let r = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        let i = b.column(1).as_any().downcast_ref::<Int32Array>().unwrap();
        let u = b.column(2).as_any().downcast_ref::<Int64Array>().unwrap();
        for k in 0..b.num_rows() {
            seen.push((r.value(k), i.value(k), u.value(k)));
        }
    }
    assert_eq!(
        seen,
        vec![(0, 0, 0), (1, 1, -7), (2, 2, -14), (3, 3, -21), (4, 4, -28)],
        "rowid comes from the reader's position column; the user's same-named \
         column must come back untouched"
    );
}

// ---------------------------------------------------------------------------
// DELETE/UPDATE position resolution
// ---------------------------------------------------------------------------

/// `resolve_positions` hands its predicate to the reader so a DELETE prunes row
/// groups instead of reading every row. Positions stay true under that pruning,
/// so the right rows must still be the ones that go.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn delete_with_a_prunable_predicate_removes_exactly_the_right_rows() {
    let temp = TempDir::new().unwrap();
    seed_float_table(&temp, None).await;

    let write_ctx = ctx_for(&temp, false, true).await;
    write_ctx
        .sql("DELETE FROM ducklake.main.t WHERE id >= 40000 AND id < 40010")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    let ctx = ctx_for(&temp, false, false).await;
    assert_eq!(
        row_count(&ctx, "SELECT id FROM ducklake.main.t").await,
        ROWS as usize - 10
    );
    assert_eq!(
        row_count(
            &ctx,
            "SELECT id FROM ducklake.main.t WHERE id >= 39999 AND id < 40011"
        )
        .await,
        2,
        "only 39999 and 40010 should remain in that window"
    );
}

/// The data-loss guard. Parquet footer float bounds exclude NaN, so pushing a
/// float predicate into the reader could hide a matching NaN row — and a row the
/// reader hides is a row the DELETE silently fails to delete, which is worse
/// than a missing query result. `predicate_is_prunable` refuses float
/// predicates, so the NaN row must still be deleted.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn delete_on_a_float_predicate_still_removes_nan_rows() {
    let temp = TempDir::new().unwrap();
    seed_float_table(&temp, Some(12_345)).await;

    let write_ctx = ctx_for(&temp, false, true).await;
    let affected = write_ctx
        .sql("DELETE FROM ducklake.main.t WHERE x > 1e300")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let reported: u64 = affected
        .iter()
        .map(|b| {
            b.column(0)
                .as_any()
                .downcast_ref::<arrow::array::UInt64Array>()
                .expect("rows-affected is UInt64")
                .value(0)
        })
        .sum();
    assert_eq!(reported, 1, "the NaN row must be found and deleted");

    let ctx = ctx_for(&temp, false, false).await;
    assert_eq!(
        row_count(&ctx, "SELECT id FROM ducklake.main.t WHERE id = 12345").await,
        0,
        "the NaN row must actually be gone"
    );
    assert_eq!(
        row_count(&ctx, "SELECT id FROM ducklake.main.t").await,
        ROWS as usize - 1
    );
}

/// A rename means the file's physical column names differ from the catalog names
/// the predicate carries, and this path has no `ColumnRenameExec` to rewrite
/// them. Pushing anyway would not merely be slow: the reader resolves a pushed
/// predicate by name, an unresolvable column becomes a null literal, the
/// predicate folds to false, every row group prunes — and the DELETE silently
/// removes nothing. `predicate_is_prunable` refuses any file with a rename.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn delete_after_a_column_rename_removes_the_right_rows() {
    let temp = TempDir::new().unwrap();
    seed_float_table(&temp, None).await;

    rename_column(&temp, "id", "ident", "int32").await;

    let read_ctx = ctx_for(&temp, false, false).await;
    assert_eq!(
        row_count(&read_ctx, "SELECT ident FROM ducklake.main.t").await,
        ROWS as usize,
        "the renamed column must read through the old physical name"
    );

    let write_ctx = ctx_for(&temp, false, true).await;
    write_ctx
        .sql("DELETE FROM ducklake.main.t WHERE ident >= 40000 AND ident < 40010")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    let ctx = ctx_for(&temp, false, false).await;
    assert_eq!(
        row_count(&ctx, "SELECT ident FROM ducklake.main.t").await,
        ROWS as usize - 10,
        "the DELETE must remove exactly 10 rows, not zero"
    );
    assert_eq!(
        row_count(
            &ctx,
            "SELECT ident FROM ducklake.main.t WHERE ident >= 40000 AND ident < 40010"
        )
        .await,
        0
    );
}

/// The internal position column must never reach a user-visible schema — not
/// through `SELECT *`, not through the CDC feeds, not through
/// `information_schema`. It is an implementation detail of one scan, and
/// official DuckLake has no such column in any of its outputs.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn the_internal_position_column_never_reaches_a_user_visible_schema() {
    let temp = TempDir::new().unwrap();
    seed_float_table(&temp, None).await;

    let write_ctx = ctx_for(&temp, false, true).await;
    write_ctx
        .sql("DELETE FROM ducklake.main.t WHERE id < 5")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    let ctx = ctx_for(&temp, true, false).await;
    let mut checked = 0;
    for sql in [
        "SELECT * FROM ducklake.main.t LIMIT 1",
        "SELECT * FROM ducklake_table_changes('main.t', 1, 3) LIMIT 1",
        "SELECT * FROM ducklake_table_deletions('main.t', 1, 3) LIMIT 1",
        "SELECT * FROM ducklake_table_insertions('main.t', 1, 3) LIMIT 1",
        "SELECT * FROM ducklake.information_schema.columns",
    ] {
        let df = ctx
            .sql(sql)
            .await
            .unwrap_or_else(|e| panic!("`{sql}` must be queryable: {e}"));
        checked += 1;
        let names: Vec<String> = df
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();
        assert!(
            !names.iter().any(|n| n.starts_with("__ducklake_row_pos")),
            "internal position column leaked into `{sql}`: {names:?}"
        );

        // And it must not appear as a *value* either, e.g. via a column listing.
        for batch in df.collect().await.unwrap() {
            let rendered =
                datafusion::arrow::util::pretty::pretty_format_batches(&[batch]).unwrap();
            assert!(
                !rendered.to_string().contains("__ducklake_row_pos"),
                "internal position column leaked into the rows of `{sql}`"
            );
        }
    }
    assert_eq!(checked, 5, "every surface must actually have been queried");
}

/// The collision guard must consider the CATALOG names, not only the file's
/// physical ones.
///
/// `present_catalog_schema` puts catalog names in front of the scan's trailing
/// internal columns and lets `ColumnRenameExec` bind every output field **by
/// name**. So a catalog column called `__ducklake_row_pos` whose physical name
/// in this file differs — because it was renamed after the file was written —
/// slips past a guard that only inspects the file's schema: no clash is seen,
/// the internal column keeps the plain name, and both output fields then resolve
/// to the user's data column. Every internal column is Int64, so nothing fails
/// to downcast; the CDC feeds just report the wrong rows.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn cdc_is_safe_when_a_catalog_column_is_renamed_onto_the_internal_name() {
    let temp = TempDir::new().unwrap();
    let writer = Arc::new(new_writer(&temp).await);
    let table_writer = DuckLakeTableWriter::new(writer, Arc::new(LocalFileSystem::new())).unwrap();

    // Physical names are `x`, `v`. Values of `x` are deliberately far from any
    // physical position, so binding the wrong column is unmistakable.
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Int64, false),
        Field::new("v", DataType::Int64, false),
    ]));
    let n = 200i64;
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(
                (0..n).map(|i| i * -1000).collect::<Vec<_>>(),
            )) as _,
            Arc::new(Int64Array::from((0..n).collect::<Vec<_>>())) as _,
        ],
    )
    .unwrap();
    table_writer
        .write_table("main", "t", &[batch])
        .await
        .unwrap();

    // Catalog name becomes `__ducklake_row_pos`; the file still calls it `x`.
    rename_column(&temp, "x", "__ducklake_row_pos", "int64").await;

    let write_ctx = ctx_for(&temp, false, true).await;
    write_ctx
        .sql("DELETE FROM ducklake.main.t WHERE v IN (3, 5, 11)")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    let ctx = ctx_for(&temp, true, false).await;

    // The plain read must still be right.
    let remaining = row_count(&ctx, "SELECT v FROM ducklake.main.t").await;
    assert_eq!(remaining, n as usize - 3);

    // And the deletions feed must name exactly the rows that were deleted.
    let mut deleted: Vec<i64> = rows(
        &ctx,
        "SELECT v FROM ducklake_table_deletions('main.t', 1, 100)",
    )
    .await
    .iter()
    .flat_map(|b| {
        let idx = b.schema().index_of("v").expect("v column");
        let c = b
            .column(idx)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("v is Int64");
        (0..b.num_rows()).map(|i| c.value(i)).collect::<Vec<_>>()
    })
    .collect();
    deleted.sort_unstable();
    assert_eq!(
        deleted,
        vec![3, 5, 11],
        "the deletions feed must match rows by physical position, not by a \
         catalog column that happens to share the internal column's name"
    );
}

// ---------------------------------------------------------------------------
// The optimisations actually happen (not just "the plan mentions them")
// ---------------------------------------------------------------------------

/// `EXPLAIN ANALYZE` output for `sql`, which carries the parquet reader's
/// execution metrics.
async fn analyze(ctx: &SessionContext, sql: &str) -> String {
    let batches = rows(ctx, &format!("EXPLAIN ANALYZE {sql}")).await;
    datafusion::arrow::util::pretty::pretty_format_batches(&batches)
        .unwrap()
        .to_string()
}

/// Parse a `name=<total> total → <matched> matched` metric out of an
/// EXPLAIN ANALYZE blob.
fn pruning_metric(plan: &str, name: &str) -> Option<(usize, usize)> {
    let at = plan.find(&format!("{name}="))? + name.len() + 1;
    let rest = &plan[at..];
    let total: usize = rest
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect::<String>()
        .parse()
        .ok()?;
    let arrow = rest.find('→')? + '→'.len_utf8();
    let matched: usize = rest[arrow..]
        .trim_start()
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect::<String>()
        .parse()
        .ok()?;
    Some((total, matched))
}

/// Parse `output_rows_skew=N%`.
fn skew_percent(plan: &str) -> Option<usize> {
    let at = plan.find("output_rows_skew=")? + "output_rows_skew=".len();
    plan[at..]
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect::<String>()
        .parse()
        .ok()
}

/// Row groups are really skipped on a positional path. Every other pushdown test
/// asserts on the plan *string*, which would still say `predicate=` if pruning
/// silently stopped working; this asserts the reader's own counter.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn a_positional_scan_really_prunes_row_groups() {
    let temp = TempDir::new().unwrap();
    seed_float_table(&temp, None).await;
    let ctx = ctx_for(&temp, true, false).await;

    let plan = analyze(
        &ctx,
        "SELECT rowid, id FROM ducklake.main.t WHERE id >= 55000",
    )
    .await;
    let (total, matched) = pruning_metric(&plan, "row_groups_pruned_statistics")
        .unwrap_or_else(|| panic!("no row-group metric in:\n{plan}"));
    assert!(
        total > 1,
        "the fixture must span several row groups, or pruning proves nothing \
         (got {total}):\n{plan}"
    );
    // `id >= 55000` selects the last 5000 of 60000 rows, i.e. one 5000-row group.
    assert!(
        matched < total,
        "a selective predicate must actually prune row groups, got \
         {matched}/{total}:\n{plan}"
    );
    assert_eq!(
        matched, 1,
        "only the last row group can hold id >= 55000:\n{plan}"
    );
}

/// One file really is read by more than one partition. `" groups:"` in the plan
/// says the file was *split*; it does not say rows reached more than one
/// partition, which is the property that makes reader-derived positions
/// load-bearing.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn a_positional_scan_really_uses_more_than_one_partition() {
    let temp = TempDir::new().unwrap();
    seed_float_table(&temp, None).await;
    let ctx = ctx_for(&temp, true, false).await;

    let plan = analyze(&ctx, "SELECT rowid, id FROM ducklake.main.t").await;
    let skew =
        skew_percent(&plan).unwrap_or_else(|| panic!("no output_rows_skew metric in:\n{plan}"));
    assert!(
        skew < 100,
        "100% skew means a single partition read the whole file, so byte-range \
         splitting is not actually being exercised:\n{plan}"
    );
    // And the answer is still right under that split.
    assert_eq!(
        row_count(&ctx, "SELECT rowid FROM ducklake.main.t").await,
        ROWS as usize
    );
}

// ---------------------------------------------------------------------------
// Predicates over renamed and synthetic columns
// ---------------------------------------------------------------------------

/// The read side of a rename, which is new code: `ColumnRenameExec` rewrites a
/// pushed predicate's column *names* to the file's physical ones. Get that wrong
/// and the reader cannot bind the column, the predicate folds to false, and
/// every row group prunes — returning nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn a_predicate_on_a_renamed_column_is_pushed_correctly() {
    let temp = TempDir::new().unwrap();
    seed_float_table(&temp, None).await;
    rename_column(&temp, "id", "ident", "int32").await;

    let write_ctx = ctx_for(&temp, false, true).await;
    write_ctx
        .sql("DELETE FROM ducklake.main.t WHERE ident = 55000")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    let ctx = ctx_for(&temp, true, false).await;
    // With a delete file (positional path) and rowid projected.
    assert_eq!(
        row_count(
            &ctx,
            "SELECT rowid, ident FROM ducklake.main.t WHERE ident >= 55000"
        )
        .await,
        (ROWS - 55_000) as usize - 1,
        "a predicate on a renamed column must bind to its physical name"
    );
    // And without rowid, still on the delete path.
    assert_eq!(
        row_count(
            &ctx,
            "SELECT ident FROM ducklake.main.t WHERE ident >= 55000"
        )
        .await,
        (ROWS - 55_000) as usize - 1
    );
}

/// A predicate on the synthetic `rowid` column must never be pushed: `rowid` is
/// derived from a virtual column, and DataFusion refuses a pushed predicate that
/// references one. `RowIdExec` rejects it, and the answer must still be right.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn a_predicate_on_rowid_is_answered_without_being_pushed() {
    let temp = TempDir::new().unwrap();
    seed_float_table(&temp, None).await;
    let ctx = ctx_for(&temp, true, false).await;

    assert_eq!(
        row_count(&ctx, "SELECT id FROM ducklake.main.t WHERE rowid >= 59000").await,
        1_000
    );
    // Mixed conjunct: the data-column half may be pushed, the rowid half not.
    assert_eq!(
        row_count(
            &ctx,
            "SELECT id FROM ducklake.main.t WHERE rowid = 12345 AND id = 12345"
        )
        .await,
        1
    );
    assert_eq!(
        row_count(
            &ctx,
            "SELECT id FROM ducklake.main.t WHERE rowid = 12345 AND id = 999"
        )
        .await,
        0
    );
}
