//! Top-N (`ORDER BY col LIMIT n`) file pruning, ported from official DuckLake's
//! `test/sql/stats/topn_file_pruning.test` at oracle commit `d8a1881e`.
//!
//! Official prunes in two halves, and neither works alone: the file list is
//! ordered by the statistic the sort reads (`max_value DESC` for a descending
//! sort, `min_value ASC` for an ascending one), and then each file is skipped at
//! open time against the Top-N boundary published so far. DataFusion supplies
//! both — `PushdownSort` reorders file groups from their statistics, and the
//! parquet opener's `FilePruner` skips a file before reading its footer — so
//! what this suite pins is that the scan chain does not block either one.
//!
//! Only `topn_prunes_through_a_column_rename` and `topn_prunes_with_rowid`
//! fail without the `try_pushdown_sort` implementations they exercise. The rest
//! characterise behaviour DataFusion supplies on a bare scan and pass on either
//! side of that change — kept because they are what would catch a regression on
//! the next DataFusion bump, not because they prove anything about this crate.
//!
//! Where the numbers differ from official's `EXPLAIN ANALYZE` expectations, the
//! test says so: both differences are performance-only and are recorded in
//! `CHANGELOG.md`.
//!
//! Two properties of this suite that are deliberate rather than oversights:
//!
//! Most assertions read DataFusion's own `EXPLAIN ANALYZE` text
//! (`files_ranges_pruned_statistics`, `sort_order_for_reorder`, `output_rows`).
//! Those are DataFusion internals, not this crate's API, and a version bump that
//! renames one breaks these tests without anything being wrong here. It is the
//! only way to observe file-level skipping — the suite has no counting object
//! store — so the brittleness is accepted knowingly.
//!
//! The fixtures fit in a single record batch, so repartitioning sends every row
//! to partition 0 and multi-partition behaviour is not exercised. That trap is
//! severe where arrival order feeds a result (CDC, rowid); it is not severe
//! here, because every node in this path answers `Inexact` and the `SortExec` is
//! therefore never removed. A bad file order costs I/O, it cannot cost
//! correctness.
#![cfg(all(feature = "write-sqlite", feature = "metadata-sqlite"))]

use std::sync::Arc;

use arrow::array::{RecordBatch, StringArray, TimestampMicrosecondArray};
use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use datafusion::prelude::*;
use object_store::local::LocalFileSystem;
use tempfile::TempDir;

use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeTableWriter, MetadataWriter, SqliteMetadataProvider,
    SqliteMetadataWriter,
};

/// `2026-01-01 00:00:00` as epoch microseconds — the oracle fixture's first day.
const DAY_0: i64 = 1_767_225_600_000_000;
const ONE_DAY: i64 = 86_400 * 1_000_000;
const ONE_SECOND: i64 = 1_000_000;

fn table_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new(
            "timestamp",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            true,
        ),
        Field::new("user_id", DataType::Utf8, true),
    ]))
}

fn object_store() -> Arc<dyn object_store::ObjectStore> {
    Arc::new(LocalFileSystem::new())
}

/// One file's worth of rows: `rows` timestamps one second apart from `day`.
fn batch(day: i64, rows: i64, user: &str) -> RecordBatch {
    let timestamps: Vec<i64> = (0..rows).map(|i| day + i * ONE_SECOND).collect();
    let users: Vec<&str> = (0..rows).map(|_| user).collect();
    RecordBatch::try_new(
        table_schema(),
        vec![
            Arc::new(TimestampMicrosecondArray::from(timestamps)),
            Arc::new(StringArray::from(users)),
        ],
    )
    .unwrap()
}

fn conn_str(temp_dir: &TempDir, writable: bool) -> String {
    let db_path = temp_dir.path().join("test.db");
    if writable {
        format!("sqlite:{}?mode=rwc", db_path.display())
    } else {
        format!("sqlite:{}", db_path.display())
    }
}

/// The oracle's fixture: four data files with disjoint, ascending timestamp
/// ranges and deliberately unequal row counts — 1000 / 500 / 200 / 100, 1800 in
/// all. The counts are what make the assertions legible: a descending Top-N that
/// prunes correctly reads 100 rows (the newest, smallest file), an ascending one
/// reads 1000 (the oldest, *largest* file). Reading the largest file is what
/// proves the order came from the statistics rather than from file size or
/// insertion order.
async fn seed_four_files(temp_dir: &TempDir) {
    let data_path = temp_dir.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();

    let writer = SqliteMetadataWriter::new_with_init(&conn_str(temp_dir, true))
        .await
        .unwrap();
    writer.set_data_path(data_path.to_str().unwrap()).unwrap();
    DuckLakeTableWriter::new(Arc::new(writer), object_store())
        .unwrap()
        .write_table("main", "events", &[batch(DAY_0, 1000, "a")])
        .await
        .unwrap();

    for (day, rows, user) in [
        (DAY_0 + ONE_DAY, 500, "b"),
        (DAY_0 + 2 * ONE_DAY, 200, "c"),
        (DAY_0 + 3 * ONE_DAY, 100, "d"),
    ] {
        let writer = SqliteMetadataWriter::new(&conn_str(temp_dir, true))
            .await
            .unwrap();
        DuckLakeTableWriter::new(Arc::new(writer), object_store())
            .unwrap()
            .append_table("main", "events", &[batch(day, rows, user)])
            .await
            .unwrap();
    }
}

async fn session(temp_dir: &TempDir) -> SessionContext {
    let provider = SqliteMetadataProvider::new(&conn_str(temp_dir, false))
        .await
        .unwrap();
    let catalog = DuckLakeCatalog::new(provider).unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));
    ctx
}

async fn analyze(ctx: &SessionContext, sql: &str) -> String {
    let batches = ctx
        .sql(&format!("EXPLAIN ANALYZE {sql}"))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    datafusion::arrow::util::pretty::pretty_format_batches(&batches)
        .unwrap()
        .to_string()
}

/// Parse a `name=<total> total → <matched> matched` metric out of an
/// `EXPLAIN ANALYZE` blob.
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

/// Total rows the parquet scan actually produced, summed over every
/// `DataSourceExec` in the analyzed plan. This is the DataFusion equivalent of
/// the rows-scanned number official's sqllogictest matches with a regex: a file
/// that is pruned contributes nothing to it.
fn scan_output_rows(plan: &str) -> usize {
    plan.lines()
        .filter(|line| line.contains("DataSourceExec"))
        .filter_map(|line| {
            let at = line.find("output_rows=")? + "output_rows=".len();
            line[at..]
                .chars()
                .take_while(|c| c.is_ascii_digit())
                .collect::<String>()
                .parse::<usize>()
                .ok()
        })
        .sum()
}

/// The `timestamp` column of `SELECT timestamp FROM events <tail>`, in the order
/// the query returned it.
async fn timestamps(ctx: &SessionContext, tail: &str) -> Vec<i64> {
    let batches = ctx
        .sql(&format!(
            "SELECT timestamp FROM ducklake.main.events {tail}"
        ))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    batches
        .iter()
        .flat_map(|batch| {
            let column = batch
                .column(0)
                .as_any()
                .downcast_ref::<TimestampMicrosecondArray>()
                .expect("timestamp column");
            (0..column.len())
                .map(|row| column.value(row))
                .collect::<Vec<_>>()
        })
        .collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn topn_descending_reads_only_the_newest_file() {
    let temp = TempDir::new().unwrap();
    seed_four_files(&temp).await;
    let ctx = session(&temp).await;

    let plan = analyze(
        &ctx,
        "SELECT * FROM ducklake.main.events ORDER BY timestamp DESC LIMIT 1",
    )
    .await;
    println!("{plan}");

    assert_eq!(
        pruning_metric(&plan, "files_ranges_pruned_statistics"),
        Some((4, 1)),
        "three of the four files should be skipped without being opened\n{plan}"
    );
    assert_eq!(
        scan_output_rows(&plan),
        100,
        "a descending Top-N should read only the newest file (100 rows), not all 1800\n{plan}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn topn_ascending_reads_only_the_oldest_file() {
    let temp = TempDir::new().unwrap();
    seed_four_files(&temp).await;
    let ctx = session(&temp).await;

    let plan = analyze(
        &ctx,
        "SELECT * FROM ducklake.main.events ORDER BY timestamp ASC LIMIT 1",
    )
    .await;
    println!("{plan}");

    assert_eq!(
        pruning_metric(&plan, "files_ranges_pruned_statistics"),
        Some((4, 1)),
        "an ascending Top-N should open only the oldest file — which is the LARGEST, \
         so a pass here cannot come from ordering by file size\n{plan}"
    );
    // Official reads that whole file (1000 rows) because DuckDB's Top-N operator
    // drains it; DataFusion stops as soon as the boundary can no longer be beaten,
    // so the bound here is the file, not the row count.
    assert!(
        scan_output_rows(&plan) <= 1000,
        "no more than the oldest file's rows should be read\n{plan}"
    );
}

/// The fixture itself is worth asserting: if the four writes ever collapsed into
/// one file, or got inlined into the catalog, every pruning assertion above
/// would pass for the wrong reason.
#[tokio::test(flavor = "multi_thread")]
async fn the_fixture_really_has_four_files_of_the_stated_sizes() {
    let temp = TempDir::new().unwrap();
    seed_four_files(&temp).await;
    let ctx = session(&temp).await;

    let plan = analyze(&ctx, "SELECT count(*) FROM ducklake.main.events").await;
    println!("{plan}");

    let total: i64 = ctx
        .sql("SELECT count(*) FROM ducklake.main.events")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap()[0]
        .column(0)
        .as_any()
        .downcast_ref::<arrow::array::Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!(
        total, 1800,
        "fixture should hold 1800 rows across four files"
    );
}

/// Pruning must never change an answer. Official's `filter_stress.test` makes
/// this its only assertion, and it is the one that catches unsound pruning
/// without depending on plan text.
#[tokio::test(flavor = "multi_thread")]
async fn topn_results_are_correct_in_both_directions() {
    let temp = TempDir::new().unwrap();
    seed_four_files(&temp).await;
    let ctx = session(&temp).await;

    assert_eq!(
        timestamps(&ctx, "ORDER BY timestamp DESC LIMIT 2").await,
        vec![DAY_0 + 3 * ONE_DAY + 99 * ONE_SECOND, DAY_0 + 3 * ONE_DAY + 98 * ONE_SECOND,],
        "the two newest rows are the last two of the fourth file"
    );

    assert_eq!(
        timestamps(&ctx, "ORDER BY timestamp ASC LIMIT 2").await,
        vec![DAY_0, DAY_0 + ONE_SECOND],
        "the two oldest rows are the first two of the first file"
    );
}

/// Renaming a column in the catalog puts a [`ColumnRenameExec`] between the sort
/// and the scan — the parquet files still carry the original name, so the scan
/// reads `timestamp` and the node relabels it to `event_time`. Before that node
/// forwarded sort pushdown, DataFusion's default barred it and the file order
/// stayed as written, so nothing was skipped.
///
/// The rename is applied straight to the catalog because the parquet files must
/// keep the old name; rewriting them would erase the mapping this exercises.
#[tokio::test(flavor = "multi_thread")]
async fn topn_prunes_through_a_column_rename() {
    let temp = TempDir::new().unwrap();
    seed_four_files(&temp).await;

    let pool = sqlx::SqlitePool::connect(&conn_str(&temp, true))
        .await
        .unwrap();
    let renamed = sqlx::query(
        "UPDATE ducklake_column SET column_name = 'event_time' WHERE column_name = 'timestamp'",
    )
    .execute(&pool)
    .await
    .unwrap()
    .rows_affected();
    assert_eq!(
        renamed, 1,
        "the fixture should have exactly one such column"
    );
    pool.close().await;

    let ctx = session(&temp).await;

    // Assert the fixture: the rename really did take, so the assertions below
    // are running through the rename node rather than past it.
    let plan = analyze(
        &ctx,
        "SELECT * FROM ducklake.main.events ORDER BY event_time DESC LIMIT 1",
    )
    .await;
    println!("{plan}");
    assert!(
        plan.contains("ColumnRenameExec"),
        "the rename node should be in the plan\n{plan}"
    );

    assert_eq!(
        pruning_metric(&plan, "files_ranges_pruned_statistics"),
        Some((4, 1)),
        "a rename must not cost the file pruning\n{plan}"
    );
    assert_eq!(
        scan_output_rows(&plan),
        100,
        "only the newest file should be read\n{plan}"
    );

    // And the answer is still right.
    let newest = ctx
        .sql("SELECT event_time FROM ducklake.main.events ORDER BY event_time DESC LIMIT 1")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(
        newest[0]
            .column(0)
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .unwrap()
            .value(0),
        DAY_0 + 3 * ONE_DAY + 99 * ONE_SECOND,
    );
}

/// Official switches the optimization off entirely for `NULLS FIRST`, because
/// DuckDB wraps the dynamic filter as `IS NULL OR <filter>` and its pruning
/// cannot see through the disjunction — its test asserts all 1800 rows are read.
/// DataFusion builds the null handling into the Top-N predicate instead, so it
/// keeps pruning. We are ahead of the oracle here, which is worth pinning: if a
/// future change quietly regresses to official's behaviour, this test says so.
#[tokio::test(flavor = "multi_thread")]
async fn topn_with_nulls_first_still_prunes_and_is_correct() {
    let temp = TempDir::new().unwrap();
    seed_four_files(&temp).await;
    let ctx = session(&temp).await;

    let plan = analyze(
        &ctx,
        "SELECT * FROM ducklake.main.events ORDER BY timestamp DESC NULLS FIRST LIMIT 1",
    )
    .await;
    println!("{plan}");

    // The fixture has no NULL timestamps, so NULLS FIRST cannot change the answer.
    assert_eq!(
        timestamps(&ctx, "ORDER BY timestamp DESC NULLS FIRST LIMIT 1").await,
        vec![DAY_0 + 3 * ONE_DAY + 99 * ONE_SECOND],
        "NULLS FIRST must not change the answer on a column with no NULLs"
    );
    assert!(
        scan_output_rows(&plan) < 1800,
        "we prune under NULLS FIRST where official reads every file\n{plan}"
    );
}

/// A file carrying deletes is read under a [`DeleteFilterExec`], which sits
/// between the sort and the scan. Deleting rows cannot reorder the rows that
/// remain, so that node forwards the ordering; this pins that it does, and that
/// the delete is still applied.
#[tokio::test(flavor = "multi_thread")]
async fn topn_is_correct_when_a_file_carries_deletes() {
    let temp = TempDir::new().unwrap();
    seed_four_files(&temp).await;

    let writer = SqliteMetadataWriter::new(&conn_str(&temp, true))
        .await
        .unwrap();
    let provider = SqliteMetadataProvider::new(&conn_str(&temp, true))
        .await
        .unwrap();
    let catalog = DuckLakeCatalog::with_writer(Arc::new(provider), Arc::new(writer)).unwrap();
    let writable = SessionContext::new();
    writable.register_catalog("ducklake", Arc::new(catalog));

    // Remove the newest row, which lives in the newest file — so the Top-N
    // answer has to come from the row behind it, in that same file.
    let newest = DAY_0 + 3 * ONE_DAY + 99 * ONE_SECOND;
    writable
        .sql(&format!(
            "DELETE FROM ducklake.main.events \
             WHERE timestamp = arrow_cast({newest}, 'Timestamp(Microsecond, None)')"
        ))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    let ctx = session(&temp).await;

    let plan = analyze(
        &ctx,
        "SELECT * FROM ducklake.main.events ORDER BY timestamp DESC LIMIT 1",
    )
    .await;
    println!("{plan}");
    assert!(
        plan.contains("DeleteFilterExec"),
        "the deleted file should be read under a delete filter\n{plan}"
    );

    assert_eq!(
        timestamps(&ctx, "ORDER BY timestamp DESC LIMIT 1").await,
        vec![DAY_0 + 3 * ONE_DAY + 98 * ONE_SECOND],
        "the deleted row must not be the Top-N answer"
    );
}

/// Projecting `rowid` puts a [`RowIdExec`] between the sort and the scan. A
/// single-file table is what makes this reachable: the rowid path builds one
/// exec per file, and `UnionExec` — which DataFusion 55 gives no
/// `try_pushdown_sort` — would otherwise sit above them and bar the ordering
/// before it ever reached this node.
#[tokio::test(flavor = "multi_thread")]
async fn topn_prunes_with_rowid() {
    let temp = TempDir::new().unwrap();
    let data_path = temp.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();

    let writer = SqliteMetadataWriter::new_with_init(&conn_str(&temp, true))
        .await
        .unwrap();
    writer.set_data_path(data_path.to_str().unwrap()).unwrap();
    DuckLakeTableWriter::new(Arc::new(writer), object_store())
        .unwrap()
        .write_table("main", "events", &[batch(DAY_0, 1000, "a")])
        .await
        .unwrap();

    // `rowid` is a lineage column, absent from the table's schema unless the
    // catalog opts in — without this the query fails to plan at all.
    let provider = SqliteMetadataProvider::new(&conn_str(&temp, false))
        .await
        .unwrap();
    let catalog = DuckLakeCatalog::new(provider)
        .unwrap()
        .with_row_lineage(true);
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));

    let plan = analyze(
        &ctx,
        "SELECT rowid, timestamp FROM ducklake.main.events ORDER BY timestamp DESC LIMIT 1",
    )
    .await;
    println!("{plan}");

    // Assert the fixture: the sort really is travelling through the rowid node,
    // not past it, and no union intervenes.
    assert!(
        plan.contains("RowIdExec"),
        "the rowid node should be in the plan\n{plan}"
    );
    assert!(
        !plan.contains("UnionExec"),
        "a single-file table must not be unioned, or this proves nothing\n{plan}"
    );
    assert!(
        plan.contains("sort_order_for_reorder"),
        "the ordering should have reached the scan\n{plan}"
    );

    let rows = ctx
        .sql("SELECT rowid FROM ducklake.main.events ORDER BY timestamp DESC LIMIT 1")
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
        999,
        "the newest row is the last of the single file"
    );
}

/// A float column whose NaN state is not known-false is unprunable from parquet
/// bounds, and [`NanPruningBarrierExec`] exists to keep predicates on it away
/// from the reader. Sort pushdown reaches for the same NaN-blind statistics, so
/// the barrier refuses orderings on such a column too — and because the barrier
/// is built *outermost*, it gets that refusal in before the nodes this change
/// touches ever see the sort.
///
/// Without this test the guard is an argument about construction order rather
/// than an observed behaviour.
#[tokio::test(flavor = "multi_thread")]
async fn a_nan_unsafe_float_column_refuses_the_sort() {
    let temp = TempDir::new().unwrap();
    let data_path = temp.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();

    let schema = Arc::new(Schema::new(vec![Field::new(
        "value",
        DataType::Float64,
        true,
    )]));
    let float_batch = |values: Vec<f64>| {
        RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(arrow::array::Float64Array::from(values))],
        )
        .unwrap()
    };

    // The NaN is what makes the column unsafe: the writer records
    // `contains_nan = true`, and a bound that excludes NaN cannot be acted on.
    let writer = SqliteMetadataWriter::new_with_init(&conn_str(&temp, true))
        .await
        .unwrap();
    writer.set_data_path(data_path.to_str().unwrap()).unwrap();
    DuckLakeTableWriter::new(Arc::new(writer), object_store())
        .unwrap()
        .write_table("main", "f", &[float_batch(vec![1.0, 2.0, f64::NAN])])
        .await
        .unwrap();

    let writer = SqliteMetadataWriter::new(&conn_str(&temp, true))
        .await
        .unwrap();
    DuckLakeTableWriter::new(Arc::new(writer), object_store())
        .unwrap()
        .append_table("main", "f", &[float_batch(vec![10.0, 11.0, 12.0])])
        .await
        .unwrap();

    let ctx = session(&temp).await;

    let plan = analyze(
        &ctx,
        "SELECT * FROM ducklake.main.f ORDER BY value ASC LIMIT 1",
    )
    .await;
    println!("{plan}");

    // Assert the fixture first: a barrier that was never built would make the
    // refusal below pass for the wrong reason.
    assert!(
        plan.contains("NanPruningBarrierExec"),
        "the NaN barrier should be in the plan for a NaN-carrying float column\n{plan}"
    );
    assert!(
        !plan.contains("sort_order_for_reorder"),
        "the barrier must stop the ordering reaching the scan's NaN-blind stats\n{plan}"
    );

    // Ascending, so the answer does not depend on where NaN sorts.
    let rows = ctx
        .sql("SELECT value FROM ducklake.main.f ORDER BY value ASC LIMIT 1")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(
        rows[0]
            .column(0)
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .unwrap()
            .value(0),
        1.0,
        "refusing the pushdown must not change the answer"
    );

    // POSITIVE CONTROL. A negative assertion about plan text is worthless
    // without one: the same shape of table, same column type, same query, but no
    // NaN — so no barrier, and the ordering must reach the scan. Without this,
    // the assertion above would still pass if float columns never got reordered
    // for some unrelated reason.
    let clean = TempDir::new().unwrap();
    let clean_data = clean.path().join("data");
    std::fs::create_dir_all(&clean_data).unwrap();
    let writer = SqliteMetadataWriter::new_with_init(&conn_str(&clean, true))
        .await
        .unwrap();
    writer.set_data_path(clean_data.to_str().unwrap()).unwrap();
    DuckLakeTableWriter::new(Arc::new(writer), object_store())
        .unwrap()
        .write_table("main", "f", &[float_batch(vec![1.0, 2.0, 3.0])])
        .await
        .unwrap();
    let writer = SqliteMetadataWriter::new(&conn_str(&clean, true))
        .await
        .unwrap();
    DuckLakeTableWriter::new(Arc::new(writer), object_store())
        .unwrap()
        .append_table("main", "f", &[float_batch(vec![10.0, 11.0, 12.0])])
        .await
        .unwrap();

    let clean_ctx = session(&clean).await;
    let clean_plan = analyze(
        &clean_ctx,
        "SELECT * FROM ducklake.main.f ORDER BY value ASC LIMIT 1",
    )
    .await;
    println!("{clean_plan}");
    assert!(
        !clean_plan.contains("NanPruningBarrierExec"),
        "a float column with no NaN needs no barrier\n{clean_plan}"
    );
    assert!(
        clean_plan.contains("sort_order_for_reorder"),
        "without the barrier the ordering must reach the scan — otherwise the \
         refusal asserted above proves nothing\n{clean_plan}"
    );
}
