//! `DuckLakeTable::restore_rows_to`: a table's rows restored to an earlier
//! snapshot in one forward commit, each restored row keeping its `rowid`.
//!
//! Each scenario runs on every backend whose writer stages row mutations:
//! SQLite and DuckDB per test, MySQL and multi-catalog PostgreSQL as one test
//! per backend (one container, a fresh catalog per scenario). Single-catalog
//! PostgreSQL refuses. Every restore is checked the same way
//! ([`restore_and_check`]): `rowid, *` at the head equals `rowid, *` at the
//! mark, the snapshot before the restore still reads what it read, and the
//! crate's change feed of each snapshot from the mark through the restore turns
//! the rows before it into the rows at it. A last test has official DuckLake
//! read a restored DuckDB catalog, diff its change feed against this crate's,
//! and go on writing it.

#![cfg(feature = "restore")]
#![cfg(any(feature = "write-sqlite", feature = "write-duckdb"))]

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

use arrow::array::{Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::prelude::SessionContext;
use datafusion_ducklake::maintenance::ExpireCriteria;
use datafusion_ducklake::{
    CatalogReadMemo, DuckLakeCatalog, DuckLakeError, DuckLakeTable, DuckLakeWriteOptions,
    InlinedFlushPolicy, MergeOptions, MetadataProvider, RestoreResult, RewriteOptions,
};

#[cfg(feature = "write-duckdb")]
use crate::inlining_abstraction_tests::duckdb;
#[cfg(feature = "write-sqlite")]
use crate::inlining_abstraction_tests::sqlite;
use crate::inlining_abstraction_tests::{Lake, MakeLake, batch, create, create_with, lines};

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

/// `main.<table>` of `lake` at the head, opened through a catalog with
/// `options`.
async fn table_with(
    lake: &Lake,
    table: &str,
    options: DuckLakeWriteOptions,
) -> (SessionContext, DuckLakeTable) {
    let ctx = SessionContext::new();
    DuckLakeCatalog::register(&ctx, "lake", lake.catalog().with_write_options(options));
    let provider = ctx
        .catalog("lake")
        .unwrap()
        .schema("main")
        .unwrap()
        .table(table)
        .await
        .unwrap()
        .unwrap();
    let table = (provider.as_ref() as &dyn std::any::Any)
        .downcast_ref::<DuckLakeTable>()
        .unwrap()
        .clone();
    (ctx, table)
}

async fn restore(lake: &Lake, table: &str, mark: i64) -> Result<RestoreResult, DuckLakeError> {
    let (ctx, handle) = lake.table(table).await;
    Box::pin(handle.restore_rows_to(&ctx.state(), mark)).await
}

/// The crate's change feed of `main.<table>` from `from` through `through`,
/// as `(snapshot, change_type, "rowid,<columns>")`.
async fn changes(lake: &Lake, table: &str, from: i64, through: i64) -> Vec<(i64, String, String)> {
    let ctx = SessionContext::new();
    datafusion_ducklake::register_ducklake_functions(&ctx, Arc::clone(&lake.provider));
    let batches = ctx
        .sql(&format!(
            "SELECT * FROM ducklake_table_changes('main.{table}', {from}, {through}) \
             ORDER BY snapshot_id, rowid, change_type"
        ))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    lines(&batches, false)
        .into_iter()
        .map(|line| {
            let mut cells = line.splitn(4, ',');
            let snapshot = cells.next().unwrap().parse().unwrap();
            let rowid = cells.next().unwrap().to_string();
            let change = cells.next().unwrap().to_string();
            (
                snapshot,
                change,
                format!("{rowid},{}", cells.next().unwrap()),
            )
        })
        .collect()
}

/// `rowid -> line` of rows as [`Lake::rows_at`] lists them.
fn by_rowid(rows: &[String]) -> BTreeMap<i64, String> {
    rows.iter().map(|row| (rowid(row), row.clone())).collect()
}

fn rowid(row: &str) -> i64 {
    row.split(',').next().unwrap().parse().unwrap()
}

/// `rows` with `changes` applied snapshot by snapshot: each snapshot's deletes
/// and pre-images first, then its inserts and post-images.
fn replay(rows: &[String], changes: &[(i64, String, String)]) -> Vec<String> {
    let mut live = by_rowid(rows);
    let snapshots: Vec<i64> = changes
        .iter()
        .map(|(snapshot, _, _)| *snapshot)
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    for snapshot in snapshots {
        let at = changes.iter().filter(|(s, _, _)| *s == snapshot);
        for (_, change, row) in at.clone() {
            if matches!(change.as_str(), "delete" | "update_preimage") {
                assert_eq!(
                    live.remove(&rowid(row)).as_deref(),
                    Some(row.as_str()),
                    "snapshot {snapshot} deletes a row that is not live: {row}"
                );
            }
        }
        for (_, change, row) in at {
            if matches!(change.as_str(), "insert" | "update_postimage") {
                assert!(
                    live.insert(rowid(row), row.clone()).is_none(),
                    "snapshot {snapshot} inserts a live rowid: {row}"
                );
            }
        }
    }
    live.into_values().collect()
}

/// Restore `main.<table>` to `mark` and check it: `rowid, *` at the head
/// equals `rowid, *` at the mark, the snapshot before the restore still reads
/// what it read, the counts cover the rows that differ (a row moved without
/// change is removed and restored), and the crate's change feed of each
/// snapshot from the mark through the restore turns the rows before it into
/// the rows at it.
async fn restore_and_check(lake: &Lake, table: &str, mark: i64) -> RestoreResult {
    Box::pin(restore_and_check_with(
        lake,
        table,
        mark,
        DuckLakeWriteOptions::default(),
    ))
    .await
}

/// [`restore_and_check`] on the table opened through a catalog with `options`.
async fn restore_and_check_with(
    lake: &Lake,
    table: &str,
    mark: i64,
    options: DuckLakeWriteOptions,
) -> RestoreResult {
    let before = lake.head();
    let pre = lake.rows(table).await;
    let expected = lake.rows_at(table, Some(mark)).await;
    let (ctx, handle) = table_with(lake, table, options).await;
    let result = Box::pin(handle.restore_rows_to(&ctx.state(), mark))
        .await
        .unwrap();
    assert_eq!(
        lake.rows(table).await,
        expected,
        "rowid, * at the head equals the mark's"
    );
    assert_eq!(lake.rows_at(table, Some(before)).await, pre, "history kept");
    let pre_set: HashSet<&String> = pre.iter().collect();
    let expected_set: HashSet<&String> = expected.iter().collect();
    let removed = pre_set.difference(&expected_set).count() as u64;
    let restored = expected_set.difference(&pre_set).count() as u64;
    assert!(result.rows_removed >= removed, "{result:?}");
    assert!(result.rows_restored >= restored, "{result:?}");
    assert_eq!(
        result.rows_removed as i64 - result.rows_restored as i64,
        pre.len() as i64 - expected.len() as i64,
        "{result:?}"
    );
    match result.snapshot_id {
        None => {
            assert_eq!(lake.head(), before, "nothing to undo commits nothing");
            assert_eq!(pre, expected);
        },
        Some(snapshot) => {
            assert_eq!(snapshot, lake.head(), "one snapshot");
            // Snapshot by snapshot, the feed turns the rows at the mark into
            // the rows at each later snapshot, the restore's included.
            let mut rows = expected.clone();
            for at in lake
                .provider
                .snapshot_ids_between(mark + 1, snapshot)
                .unwrap()
            {
                let feed = changes(lake, table, at, at).await;
                rows = replay(&rows, &feed);
                assert_eq!(
                    rows,
                    lake.rows_at(table, Some(at)).await,
                    "the feed of {at}"
                );
            }
        },
    }
    result
}

/// `rowid, *` of `main.<table>` at `snapshot` where `filter` holds, one line
/// per row, ordered by `rowid`.
async fn rows_where(lake: &Lake, table: &str, snapshot: i64, filter: &str) -> Vec<String> {
    let ctx = SessionContext::new();
    let catalog = DuckLakeCatalog::with_snapshot(Arc::clone(&lake.provider), snapshot)
        .unwrap()
        .with_row_lineage(true);
    ctx.register_catalog("lake", Arc::new(catalog));
    let batches = ctx
        .sql(&format!(
            "SELECT * FROM lake.main.{table} WHERE {filter} ORDER BY rowid"
        ))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    lines(&batches, true)
}

/// The files under `dir`, recursively.
fn files_under(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut files = Vec::new();
    let mut pending = vec![dir.to_path_buf()];
    while let Some(dir) = pending.pop() {
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                pending.push(path);
            } else {
                files.push(path);
            }
        }
    }
    files.sort();
    files
}

/// `(rows_removed, rows_restored, files_ended)` of `result`.
fn counts(result: &RestoreResult) -> (u64, u64, usize) {
    (
        result.rows_removed,
        result.rows_restored,
        result.files_ended,
    )
}

/// Data file paths of `main.<table>` live at the head.
fn file_paths(lake: &Lake, table: &str) -> Vec<String> {
    lake.provider
        .get_table_files_for_select(lake.table_id(table), lake.head())
        .unwrap()
        .into_iter()
        .map(|file| file.file.path)
        .collect()
}

fn ids(values: impl IntoIterator<Item = i64>) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, true)])),
        vec![Arc::new(Int64Array::from(values.into_iter().collect::<Vec<_>>()))],
    )
    .unwrap()
}

/// Rows `(region, id, v)`.
fn regional(rows: &[(&str, i64)], prefix: &str) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("region", DataType::Utf8, true),
            Field::new("id", DataType::Int64, true),
            Field::new("v", DataType::Utf8, true),
        ])),
        vec![
            Arc::new(StringArray::from(
                rows.iter().map(|(region, _)| *region).collect::<Vec<_>>(),
            )),
            Arc::new(Int64Array::from(
                rows.iter().map(|(_, id)| *id).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                rows.iter()
                    .map(|(_, id)| format!("{prefix}{id}"))
                    .collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap()
}

// ---------------------------------------------------------------------------
// Scenarios
// ---------------------------------------------------------------------------

/// Appends since the mark, to Parquet and inlined, are removed: the added file
/// ends whole and the inlined rows are ended. Restoring again commits nothing.
pub(crate) async fn restores_appends(make: MakeLake) {
    let lake = make(Some(5)).await;
    create(&lake, "t").await;
    lake.append_parquet("t", &batch(1..=10, "a")).await;
    lake.append("t", &batch(100..=101, "i")).await;
    let mark = lake.head();
    lake.append_parquet("t", &batch(11..=20, "b")).await;
    lake.append("t", &batch(102..=104, "j")).await;
    let result = restore_and_check(&lake, "t", mark).await;
    assert_eq!(counts(&result), (13, 0, 1));
    assert_eq!(lake.inlined_rows("t"), 2);
    assert_eq!(lake.data_files("t"), 1);

    let head = lake.head();
    let again = restore(&lake, "t", mark).await.unwrap();
    assert_eq!(again, RestoreResult::default());
    assert_eq!(lake.head(), head);

    // A file added since the mark with at most the limit's live rows stays
    // live, its rows deleted inlined, as official's `DELETE` deletes them;
    // so does one with a deletion inlined already, and one whose rows are
    // all deleted is left alone.
    let mark = lake.head();
    let deletes = lake.inlined_deletes("t");
    lake.append_parquet("t", &batch(30..=32, "s")).await;
    lake.append_parquet("t", &batch(40..=49, "d")).await;
    lake.sql("DELETE FROM lake.main.t WHERE id = 40")
        .await
        .unwrap();
    lake.append_parquet("t", &batch(50..=52, "g")).await;
    lake.sql("DELETE FROM lake.main.t WHERE id >= 50 AND id <= 52")
        .await
        .unwrap();
    let files = lake.data_files("t");
    let result = restore_and_check(&lake, "t", mark).await;
    assert_eq!(counts(&result), (3 + 9, 0, 0));
    assert_eq!(lake.data_files("t"), files, "no file ended");
    assert_eq!(lake.inlined_deletes("t"), deletes + 1 + 3 + 3);
    assert_eq!(lake.delete_files("t"), 1, "the file with 9 live rows");
}

/// Deletes since the mark — inlined deletions of Parquet rows, a delete file,
/// ended inlined rows — are undone by inserting the rows again with their row
/// ids: inlined, as there are few.
pub(crate) async fn restores_deletes(make: MakeLake) {
    let lake = make(Some(40)).await;
    create(&lake, "t").await;
    lake.append_parquet("t", &batch(1..=40, "a")).await;
    lake.append("t", &batch(100..=103, "i")).await;
    let mark = lake.head();
    for statement in [
        "DELETE FROM lake.main.t WHERE id IN (3, 4)",
        "DELETE FROM lake.main.t WHERE id BETWEEN 10 AND 35",
        "DELETE FROM lake.main.t WHERE id = 101",
        "DELETE FROM lake.main.t WHERE id = 38",
    ] {
        lake.sql(statement).await.unwrap();
    }
    assert!(lake.delete_files("t") > 0 || lake.inlined_deletes("t") > 0);
    let inlined = lake.inlined_rows("t");
    let result = restore_and_check(&lake, "t", mark).await;
    assert_eq!(counts(&result), (0, 30, 0));
    assert_eq!(lake.inlined_rows("t"), inlined + 30, "restored rows inline");
}

/// A delete file written before the mark and still the file's live one at
/// the head: rows deleted since, inlined, are restored, and the rows it
/// deletes stay deleted.
pub(crate) async fn restores_beside_a_kept_delete_file(make: MakeLake) {
    let lake = make(Some(5)).await;
    create(&lake, "t").await;
    lake.append_parquet("t", &batch(1..=20, "a")).await;
    lake.sql("DELETE FROM lake.main.t WHERE id <= 8")
        .await
        .unwrap();
    assert_eq!(lake.delete_files("t"), 1);
    let mark = lake.head();
    lake.sql("DELETE FROM lake.main.t WHERE id IN (15, 16)")
        .await
        .unwrap();
    assert_eq!(lake.delete_files("t"), 1);
    lake.append("t", &batch([90], "n")).await;
    let result = restore_and_check(&lake, "t", mark).await;
    assert_eq!(counts(&result), (1, 2, 0));
}

/// Updates since the mark, of Parquet and inlined rows, are undone: the new
/// versions are removed and the old ones restored, with their row ids. More
/// rows than the limit are restored to a Parquet file that embeds them.
pub(crate) async fn restores_updates(make: MakeLake) {
    let lake = make(Some(5)).await;
    create(&lake, "t").await;
    lake.append_parquet("t", &batch(1..=20, "a")).await;
    lake.append("t", &batch(100..=102, "i")).await;
    let mark = lake.head();
    for statement in [
        "UPDATE lake.main.t SET v = 'u' WHERE id <= 3",
        "UPDATE lake.main.t SET v = 'w' WHERE id >= 8 AND id <= 20",
        "UPDATE lake.main.t SET v = 'x' WHERE id = 100",
        "UPDATE lake.main.t SET v = 'y' WHERE id = 2",
    ] {
        lake.sql(statement).await.unwrap();
    }
    let files = lake.data_files("t");
    let result = restore_and_check(&lake, "t", mark).await;
    assert_eq!((result.rows_removed, result.rows_restored), (17, 17));
    assert!(
        lake.data_files("t") <= files,
        "the update's file ended, one restored file at most added"
    );
}

/// Keyed upserts, updates and deletes since the mark are undone.
pub(crate) async fn restores_keyed_mutations(make: MakeLake) {
    let lake = make(Some(5)).await;
    create(&lake, "t").await;
    lake.append_parquet("t", &batch(1..=15, "a")).await;
    lake.append_parquet("t", &batch(16..=30, "a")).await;
    lake.append("t", &batch(100..=103, "i")).await;
    let mark = lake.head();
    let (ctx, table) = lake.table("t").await;
    table
        .keyed_mutation(&["id"])
        .upsert_by_keys(&ctx.state(), vec![batch([102, 7, 200], "new")])
        .await
        .unwrap();
    let (ctx, table) = lake.table("t").await;
    table
        .keyed_mutation(&["id"])
        .update_by_keys(&ctx.state(), vec![batch([6, 103], "upd")])
        .await
        .unwrap();
    let (ctx, table) = lake.table("t").await;
    table
        .keyed_mutation(&["id"])
        .delete_by_keys(&ctx.state(), vec![ids(1..=3)])
        .await
        .unwrap();
    let (ctx, table) = lake.table("t").await;
    table
        .keyed_mutation(&["id"])
        .upsert_by_keys(&ctx.state(), vec![batch(25..=40, "big")])
        .await
        .unwrap();
    let result = restore_and_check(&lake, "t", mark).await;
    assert!(result.files_ended >= 1, "{result:?}");
}

/// A replace since the mark is undone: the replacement's files end and the
/// replaced rows are restored.
pub(crate) async fn restores_replace(make: MakeLake) {
    let lake = make(Some(5)).await;
    create(&lake, "t").await;
    lake.append_parquet("t", &batch(1..=12, "a")).await;
    lake.append("t", &batch(100..=101, "i")).await;
    let mark = lake.head();
    lake.table_writer()
        .with_options(&DuckLakeWriteOptions::default().with_data_inlining_row_limit(0))
        .write_table("main", "t", &[batch(500..=520, "r")])
        .await
        .unwrap();
    lake.table_writer()
        .write_table("main", "t", &[batch(600..=601, "s")])
        .await
        .unwrap();
    let result = restore_and_check(&lake, "t", mark).await;
    assert_eq!((result.rows_removed, result.rows_restored), (2, 14));
}

/// A history mixing every mutation, restored to one mark after another,
/// including to a mark between two restores and to the head before the first
/// restore: each restore is itself undoable.
pub(crate) async fn restores_mixes(make: MakeLake) {
    let lake = make(Some(5)).await;
    create(&lake, "t").await;
    lake.append_parquet("t", &batch(1..=10, "a")).await;
    let m1 = lake.head();
    lake.append("t", &batch(11..=12, "i")).await;
    lake.sql("UPDATE lake.main.t SET v = 'u1' WHERE id = 1")
        .await
        .unwrap();
    let m2 = lake.head();
    lake.sql("DELETE FROM lake.main.t WHERE id BETWEEN 2 AND 5")
        .await
        .unwrap();
    lake.append_parquet("t", &batch(30..=40, "b")).await;
    lake.sql("UPDATE lake.main.t SET v = 'u2' WHERE id >= 35 OR id = 11")
        .await
        .unwrap();
    let m3 = lake.head();
    let (ctx, table) = lake.table("t").await;
    table
        .keyed_mutation(&["id"])
        .upsert_by_keys(&ctx.state(), vec![batch([1, 30, 50], "up")])
        .await
        .unwrap();
    lake.sql("DELETE FROM lake.main.t WHERE id = 12")
        .await
        .unwrap();
    let before_restores = lake.head();

    let first_restore = restore_and_check(&lake, "t", m2).await.snapshot_id.unwrap();
    restore_and_check(&lake, "t", m1).await;
    restore_and_check(&lake, "t", m3).await;
    restore_and_check(&lake, "t", before_restores).await;
    restore_and_check(&lake, "t", first_restore).await;
}

/// A partitioned table's restored rows land in their own partitions, those
/// moved by an update since the mark included; few rows restore inlined.
pub(crate) async fn restores_partitioned(make: MakeLake) {
    let lake = make(Some(0)).await;
    create_with(
        &lake,
        "p",
        &[("region", DataType::Utf8), ("id", DataType::Int64), ("v", DataType::Utf8)],
    )
    .unwrap();
    lake.sql("ALTER TABLE lake.main.p SET PARTITIONED BY (region)")
        .await
        .unwrap();
    let rows: Vec<(&str, i64)> = (1..=12)
        .map(|id| (["a", "b", "c"][(id % 3) as usize], id))
        .collect();
    lake.table_writer()
        .append_table("main", "p", &[regional(&rows, "r")])
        .await
        .unwrap();
    let mark = lake.head();
    for statement in [
        "DELETE FROM lake.main.p WHERE region = 'a'",
        "UPDATE lake.main.p SET region = 'z' WHERE id IN (1, 2)",
    ] {
        lake.sql(statement).await.unwrap();
    }
    lake.table_writer()
        .append_table("main", "p", &[regional(&[("d", 90), ("a", 91)], "n")])
        .await
        .unwrap();
    let result = restore_and_check(&lake, "p", mark).await;
    assert!(result.rows_restored >= 6, "{result:?}");
    // Pruned by partition, the head reads each partition as the mark does.
    for region in ["a", "b", "c", "d", "z"] {
        let filter = format!("region = '{region}'");
        assert_eq!(
            rows_where(&lake, "p", lake.head(), &filter).await,
            rows_where(&lake, "p", mark, &filter).await,
            "{filter}"
        );
    }
    for path in file_paths(&lake, "p") {
        assert!(
            ["region=a/", "region=b/", "region=c/"]
                .iter()
                .any(|part| path.contains(part)),
            "every live file sits in a partition of the mark: {path}"
        );
    }

    // Under a limit, a small restore of a partitioned table inlines.
    let lake = make(Some(5)).await;
    create_with(
        &lake,
        "p",
        &[("region", DataType::Utf8), ("id", DataType::Int64), ("v", DataType::Utf8)],
    )
    .unwrap();
    lake.sql("ALTER TABLE lake.main.p SET PARTITIONED BY (region)")
        .await
        .unwrap();
    lake.table_writer()
        .with_options(&DuckLakeWriteOptions::default().with_data_inlining_row_limit(0))
        .append_table("main", "p", &[regional(&rows, "r")])
        .await
        .unwrap();
    let mark = lake.head();
    lake.sql("DELETE FROM lake.main.p WHERE id <= 3")
        .await
        .unwrap();
    let result = restore_and_check(&lake, "p", mark).await;
    assert_eq!(counts(&result), (0, 3, 0));
    assert_eq!(lake.inlined_rows("p"), 3);
}

/// A restore of more rows than one output file holds streams them into
/// several files, each embedding its rows' ids.
pub(crate) async fn restores_into_several_files(make: MakeLake) {
    let lake = make(Some(5)).await;
    create(&lake, "t").await;
    let wide = |range: std::ops::RangeInclusive<i64>| {
        let ids: Vec<i64> = range.collect();
        let values: Vec<String> = ids.iter().map(|id| format!("{id:0>64}")).collect();
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("id", DataType::Int64, true),
                Field::new("v", DataType::Utf8, true),
            ])),
            vec![Arc::new(Int64Array::from(ids)), Arc::new(StringArray::from(values))],
        )
        .unwrap()
    };
    lake.append_parquet("t", &wide(1..=3000)).await;
    lake.append_parquet("t", &wide(3001..=6000)).await;
    // A delete file before the mark, rewritten after it with each position's
    // snapshot: the mark reads the rewritten file, dated, and so does the head.
    lake.sql("DELETE FROM lake.main.t WHERE id % 3 = 0")
        .await
        .unwrap();
    let mark = lake.head();
    lake.sql("DELETE FROM lake.main.t WHERE id % 2 = 0")
        .await
        .unwrap();
    lake.sql("UPDATE lake.main.t SET v = 'u' WHERE id > 5000")
        .await
        .unwrap();
    let before = lake.data_files("t");
    let pre: HashSet<String> = lake.rows("t").await.into_iter().collect();
    let wanted: HashSet<String> = lake.rows_at("t", Some(mark)).await.into_iter().collect();

    let mut options = DuckLakeWriteOptions::default();
    options.target_file_size = Some(4096);
    options.max_row_group_rows = Some(64);
    let result = restore_and_check_with(&lake, "t", mark, options).await;
    assert_eq!(
        counts(&result),
        (
            pre.difference(&wanted).count() as u64,
            wanted.difference(&pre).count() as u64,
            1
        ),
        "{result:?}"
    );
    let written = lake.data_files("t") + result.files_ended - before;
    assert!(written > 1, "the restore wrote {written} files");
}

/// A flush, merge and rewrite inside the window: rows a partial file moved
/// stay put, a rewrite's files end and the files it ended are restored.
pub(crate) async fn restores_across_flush_and_compaction(make: MakeLake) {
    let lake = make(Some(5)).await;
    create(&lake, "t").await;
    for start in [1, 4, 7, 10] {
        lake.append("t", &batch(start..start + 3, "i")).await;
    }
    lake.append_parquet("t", &batch(50..=59, "p")).await;
    lake.append_parquet("t", &batch(60..=69, "p")).await;
    let m1 = lake.head();

    let always =
        DuckLakeWriteOptions::default().with_inlined_flush_policy(InlinedFlushPolicy::always());
    let (ctx, table) = table_with(&lake, "t", always.clone()).await;
    let merged = table
        .merge_adjacent_files(&ctx.state(), MergeOptions::default())
        .await
        .unwrap();
    assert!(merged.did_work(), "{merged:?}");
    assert_eq!(lake.inlined_rows("t"), 0);
    lake.sql("DELETE FROM lake.main.t WHERE id IN (2, 51)")
        .await
        .unwrap();
    lake.append("t", &batch(80..=82, "j")).await;
    let m2 = lake.head();
    let (ctx, table) = table_with(&lake, "t", always.clone()).await;
    table
        .merge_adjacent_files(&ctx.state(), MergeOptions::default())
        .await
        .unwrap();
    lake.sql("DELETE FROM lake.main.t WHERE id BETWEEN 60 AND 66")
        .await
        .unwrap();
    let (ctx, table) = lake.table("t").await;
    let rewritten = table
        .rewrite_data_files(
            &ctx.state(),
            RewriteOptions {
                delete_threshold: 0.0,
                data_file_ids: None,
            },
        )
        .await
        .unwrap();
    assert!(rewritten.did_work(), "{rewritten:?}");
    lake.sql("UPDATE lake.main.t SET v = 'late' WHERE id = 80")
        .await
        .unwrap();

    restore_and_check(&lake, "t", m2).await;
    restore_and_check(&lake, "t", m1).await;
}

/// Refusals: a snapshot not before the table's, one before the table
/// existed, a read-only table, a column or partition-spec change since the
/// mark, an expired mark, and a commit since the table was read. None of them
/// commits anything.
pub(crate) async fn refusals(make: MakeLake) {
    let lake = make(Some(5)).await;
    Box::pin(refusals_of_reads(&lake)).await;
    Box::pin(refusals_of_changes(&lake)).await;
    Box::pin(refusals_of_conflicts(&lake)).await;
}

/// A snapshot not before the table's, one the table did not exist at, and a
/// read-only table.
async fn refusals_of_reads(lake: &Lake) {
    create(lake, "t").await;
    lake.append_parquet("t", &batch(1..=5, "a")).await;
    let mark = lake.head();
    lake.append_parquet("t", &batch(6..=9, "b")).await;
    let head = lake.head();

    for snapshot in [head, head + 5] {
        let error = restore(lake, "t", snapshot).await.unwrap_err();
        assert!(matches!(error, DuckLakeError::InvalidConfig(_)), "{error}");
    }
    create(lake, "later").await;
    lake.append_parquet("later", &batch(1..=3, "l")).await;
    let error = restore(lake, "later", head).await.unwrap_err();
    assert!(
        matches!(&error, DuckLakeError::InvalidConfig(m) if m.contains("does not exist")),
        "{error}"
    );
    let ctx = SessionContext::new();
    ctx.register_catalog(
        "ro",
        Arc::new(DuckLakeCatalog::with_snapshot(Arc::clone(&lake.provider), head).unwrap()),
    );
    let read_only = ctx
        .catalog("ro")
        .unwrap()
        .schema("main")
        .unwrap()
        .table("t")
        .await
        .unwrap()
        .unwrap();
    let read_only = (read_only.as_ref() as &dyn std::any::Any)
        .downcast_ref::<DuckLakeTable>()
        .unwrap();
    let error = Box::pin(read_only.restore_rows_to(&ctx.state(), mark))
        .await
        .unwrap_err();
    assert!(matches!(error, DuckLakeError::InvalidConfig(_)), "{error}");
}

/// Column changes, a partition-spec change and an expired mark.
async fn refusals_of_changes(lake: &Lake) {
    // A column change since the mark.
    create(lake, "c").await;
    lake.append_parquet("c", &batch(1..=5, "a")).await;
    let mark_c = lake.head();
    create_with(
        lake,
        "c",
        &[("id", DataType::Int64), ("v", DataType::Utf8), ("w", DataType::Int64)],
    )
    .unwrap();
    let head = lake.head();
    let error = restore(lake, "c", mark_c).await.unwrap_err();
    assert!(
        matches!(&error, DuckLakeError::Unsupported(m) if m.contains("columns changed")),
        "{error}"
    );
    assert_eq!(lake.head(), head);

    // A column dropped since the mark.
    create(lake, "dr").await;
    lake.append_parquet("dr", &batch(1..=5, "a")).await;
    let mark_d = lake.head();
    create_with(lake, "dr", &[("id", DataType::Int64)]).unwrap();
    let head = lake.head();
    let error = restore(lake, "dr", mark_d).await.unwrap_err();
    assert!(
        matches!(&error, DuckLakeError::Unsupported(m) if m.contains("columns changed")),
        "{error}"
    );
    assert_eq!(lake.head(), head);

    // A column widened since the mark.
    create_with(
        lake,
        "wi",
        &[("id", DataType::Int64), ("n", DataType::Int32)],
    )
    .unwrap();
    lake.append_parquet(
        "wi",
        &RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("id", DataType::Int64, true),
                Field::new("n", DataType::Int32, true),
            ])),
            vec![
                Arc::new(Int64Array::from(vec![1, 2])),
                Arc::new(arrow::array::Int32Array::from(vec![7, 8])),
            ],
        )
        .unwrap(),
    )
    .await;
    let mark_w = lake.head();
    lake.writer
        .promote_column_type(lake.table_id("wi"), "n", "int64")
        .unwrap();
    let head = lake.head();
    let error = restore(lake, "wi", mark_w).await.unwrap_err();
    assert!(
        matches!(&error, DuckLakeError::Unsupported(m) if m.contains("columns changed")),
        "{error}"
    );
    assert_eq!(lake.head(), head);

    // A partition-spec change since the mark.
    let mark = lake.head();
    lake.sql("ALTER TABLE lake.main.t SET PARTITIONED BY (v)")
        .await
        .unwrap();
    let head = lake.head();
    let error = restore(lake, "t", mark).await.unwrap_err();
    assert!(
        matches!(&error, DuckLakeError::Unsupported(m) if m.contains("partition spec")),
        "{error}"
    );
    assert_eq!(lake.head(), head);

    // An expired mark.
    create(lake, "e").await;
    lake.append_parquet("e", &batch(1..=5, "a")).await;
    let expired = lake.head();
    lake.append_parquet("e", &batch(6..=9, "b")).await;
    lake.writer
        .maintenance()
        .unwrap()
        .expire_snapshots(ExpireCriteria::Versions(vec![expired]))
        .await
        .unwrap();
    let head = lake.head();
    let error = restore(lake, "e", expired).await.unwrap_err();
    assert!(
        matches!(error, DuckLakeError::InvalidSnapshot(_)),
        "{error}"
    );
    assert_eq!(lake.head(), head);
}

/// Commits since the table was read.
async fn refusals_of_conflicts(lake: &Lake) {
    // A commit since the table was read conflicts.
    create(lake, "s").await;
    lake.append_parquet("s", &batch(1..=5, "a")).await;
    let mark = lake.head();
    lake.append_parquet("s", &batch(6..=9, "b")).await;
    let (ctx, stale) = lake.table("s").await;
    lake.sql("DELETE FROM lake.main.s WHERE id = 7")
        .await
        .unwrap();
    let head = lake.head();
    let rows = lake.rows("s").await;
    let error = Box::pin(stale.restore_rows_to(&ctx.state(), mark))
        .await
        .unwrap_err();
    assert!(matches!(error, DuckLakeError::Conflict(_)), "{error}");
    assert_eq!(lake.head(), head);
    assert_eq!(lake.rows("s").await, rows);
    restore_and_check(lake, "s", mark).await;

    // A restore that only inserts rows again conflicts with any commit to
    // the table since it was read: another restore of the same rows (whose
    // row ids would be live twice), or an insert (whose rows the mark never
    // had).
    create(lake, "r").await;
    lake.append_parquet("r", &batch(1..=10, "a")).await;
    let mark = lake.head();
    lake.sql("DELETE FROM lake.main.r WHERE id = 3")
        .await
        .unwrap();
    let (ctx, first) = lake.table("r").await;
    let (ctx2, second) = lake.table("r").await;
    Box::pin(first.restore_rows_to(&ctx.state(), mark))
        .await
        .unwrap();
    let head = lake.head();
    let error = Box::pin(second.restore_rows_to(&ctx2.state(), mark))
        .await
        .unwrap_err();
    assert!(matches!(error, DuckLakeError::Conflict(_)), "{error}");
    assert_eq!(lake.head(), head);
    assert_eq!(lake.rows("r").await, lake.rows_at("r", Some(mark)).await);
    lake.sql("DELETE FROM lake.main.r WHERE id = 4")
        .await
        .unwrap();
    let (ctx, stale) = lake.table("r").await;
    lake.append("r", &batch([70], "x")).await;
    let head = lake.head();
    let error = Box::pin(stale.restore_rows_to(&ctx.state(), mark))
        .await
        .unwrap_err();
    assert!(matches!(error, DuckLakeError::Conflict(_)), "{error}");
    assert_eq!(lake.head(), head);
    // A restore that wrote files and then lost to a concurrent commit leaves
    // none of them behind.
    create(lake, "x").await;
    lake.append_parquet("x", &batch(1..=30, "a")).await;
    let mark = lake.head();
    lake.sql("DELETE FROM lake.main.x WHERE id <= 20")
        .await
        .unwrap();
    lake.append_parquet("x", &batch(40..=49, "b")).await;
    lake.sql("DELETE FROM lake.main.x WHERE id IN (40, 41)")
        .await
        .unwrap();
    let (ctx, stale) = lake.table("x").await;
    lake.append("x", &batch([90], "c")).await;
    let head = lake.head();
    let objects = files_under(&lake.data);
    let error = Box::pin(stale.restore_rows_to(&ctx.state(), mark))
        .await
        .unwrap_err();
    assert!(matches!(error, DuckLakeError::Conflict(_)), "{error}");
    assert_eq!(lake.head(), head);
    assert_eq!(
        files_under(&lake.data),
        objects,
        "the refused restore leaves none of the files it wrote"
    );

    // A concurrent `DELETE` that rewrites the delete file of a file the
    // restore does not touch writes a delete file beginning before the
    // restore's snapshot; it still conflicts.
    create(lake, "w").await;
    lake.append_parquet("w", &batch(1..=20, "a")).await;
    lake.sql("DELETE FROM lake.main.w WHERE id <= 6")
        .await
        .unwrap();
    let mark = lake.head();
    lake.append_parquet("w", &batch(50..=59, "b")).await;
    let (ctx, stale) = lake.table("w").await;
    lake.sql("DELETE FROM lake.main.w WHERE id BETWEEN 7 AND 12")
        .await
        .unwrap();
    let head = lake.head();
    let error = Box::pin(stale.restore_rows_to(&ctx.state(), mark))
        .await
        .unwrap_err();
    assert!(matches!(error, DuckLakeError::Conflict(_)), "{error}");
    assert_eq!(lake.head(), head);
}

/// A table read through a [`CatalogReadMemo`]: a restore through a provider
/// bound to it, with the memo filled at the head and at the mark, leaves the
/// mark's rows, and the next bind serves them; a commit since the bound
/// table was read aborts the restore.
pub(crate) async fn restores_through_a_catalog_read_memo(make: MakeLake) {
    let lake = make(Some(5)).await;
    create(&lake, "t").await;
    lake.append_parquet("t", &batch(1..=20, "a")).await;
    let mark = lake.head();
    for statement in [
        "DELETE FROM lake.main.t WHERE id IN (3, 4)",
        "UPDATE lake.main.t SET v = 'u' WHERE id = 7",
    ] {
        lake.sql(statement).await.unwrap();
    }
    lake.append_parquet("t", &batch(30..=39, "b")).await;
    let expected = lake.rows_at("t", Some(mark)).await;
    let memo = CatalogReadMemo::new(CatalogReadMemo::DEFAULT_MAX_BYTES);
    let bound = |memo: &Arc<CatalogReadMemo>| -> Arc<dyn MetadataProvider> {
        Arc::new(memo.bind(Arc::clone(&lake.provider)).unwrap())
    };
    let read = async |provider: Arc<dyn MetadataProvider>, snapshot: i64| {
        let ctx = SessionContext::new();
        let catalog = DuckLakeCatalog::with_snapshot(provider, snapshot)
            .unwrap()
            .with_row_lineage(true);
        ctx.register_catalog("lake", Arc::new(catalog));
        let batches = ctx
            .sql("SELECT * FROM lake.main.t ORDER BY rowid")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        lines(&batches, true)
    };
    let open = async |provider: Arc<dyn MetadataProvider>| {
        let ctx = SessionContext::new();
        DuckLakeCatalog::register(
            &ctx,
            "lake",
            DuckLakeCatalog::with_writer(provider, Arc::clone(&lake.writer)).unwrap(),
        );
        let table = ctx
            .catalog("lake")
            .unwrap()
            .schema("main")
            .unwrap()
            .table("t")
            .await
            .unwrap()
            .unwrap();
        let table = (table.as_ref() as &dyn std::any::Any)
            .downcast_ref::<DuckLakeTable>()
            .unwrap()
            .clone();
        (ctx, table)
    };

    // Fill the memo at the head and at the mark.
    let provider = bound(&memo);
    let head = lake.head();
    let pre = read(Arc::clone(&provider), head).await;
    assert_eq!(read(provider, mark).await, expected);

    let (ctx, table) = open(bound(&memo)).await;
    let result = Box::pin(table.restore_rows_to(&ctx.state(), mark))
        .await
        .unwrap();
    assert!(result.snapshot_id.is_some(), "{result:?}");
    assert_eq!(lake.rows("t").await, expected);
    assert_eq!(read(bound(&memo), lake.head()).await, expected);
    assert_eq!(read(bound(&memo), head).await, pre, "history kept");

    // A commit after the bound table was read aborts its restore.
    lake.sql("DELETE FROM lake.main.t WHERE id = 1")
        .await
        .unwrap();
    let (ctx, table) = open(bound(&memo)).await;
    lake.append("t", &batch([90], "x")).await;
    let head = lake.head();
    let error = Box::pin(table.restore_rows_to(&ctx.state(), mark))
        .await
        .unwrap_err();
    assert!(matches!(error, DuckLakeError::Conflict(_)), "{error}");
    assert_eq!(lake.head(), head);
}

/// The table `(id, d DATE, ts TIMESTAMP, tstz TIMESTAMPTZ, dec DECIMAL(18,3),
/// f DOUBLE, b BOOLEAN, blob BLOB)`.
fn typed_columns() -> Vec<(&'static str, DataType)> {
    vec![
        ("id", DataType::Int64),
        ("d", DataType::Date32),
        (
            "ts",
            DataType::Timestamp(arrow::datatypes::TimeUnit::Microsecond, None),
        ),
        (
            "tstz",
            DataType::Timestamp(arrow::datatypes::TimeUnit::Microsecond, Some("UTC".into())),
        ),
        ("dec", DataType::Decimal128(18, 3)),
        ("f", DataType::Float64),
        ("b", DataType::Boolean),
        ("blob", DataType::Binary),
    ]
}

/// Rows of [`typed_columns`] for `ids`: every value NULL on a multiple of 5,
/// NaN and infinities among the doubles, dates and times either side of
/// 1970.
fn typed(ids: impl IntoIterator<Item = i64>) -> RecordBatch {
    use arrow::array::{
        BinaryArray, BooleanArray, Date32Array, Decimal128Array, Float64Array,
        TimestampMicrosecondArray,
    };
    let ids: Vec<i64> = ids.into_iter().collect();
    let null = |id: &i64| id % 5 == 0;
    let schema = Arc::new(Schema::new(
        typed_columns()
            .into_iter()
            .map(|(name, data_type)| Field::new(name, data_type, true))
            .collect::<Vec<_>>(),
    ));
    let micros = |id: &i64| (!null(id)).then(|| (id - 10) * 86_400_000_123);
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(ids.clone())),
            Arc::new(Date32Array::from(
                ids.iter()
                    .map(|id| (!null(id)).then(|| (*id as i32 - 10) * 400))
                    .collect::<Vec<_>>(),
            )),
            Arc::new(TimestampMicrosecondArray::from(
                ids.iter().map(micros).collect::<Vec<_>>(),
            )),
            Arc::new(
                TimestampMicrosecondArray::from(ids.iter().map(micros).collect::<Vec<_>>())
                    .with_timezone("UTC"),
            ),
            Arc::new(
                Decimal128Array::from(
                    ids.iter()
                        .map(|id| (!null(id)).then(|| i128::from(*id) * 1_234_567 - 9_000_000))
                        .collect::<Vec<_>>(),
                )
                .with_precision_and_scale(18, 3)
                .unwrap(),
            ),
            Arc::new(Float64Array::from(
                ids.iter()
                    .map(|id| {
                        (!null(id)).then(|| match id % 4 {
                            0 => f64::NAN,
                            1 => f64::INFINITY,
                            2 => -1.5 * *id as f64,
                            _ => f64::NEG_INFINITY,
                        })
                    })
                    .collect::<Vec<_>>(),
            )),
            Arc::new(BooleanArray::from(
                ids.iter()
                    .map(|id| (!null(id) && id % 3 != 0).then_some(id % 2 == 0))
                    .collect::<Vec<_>>(),
            )),
            Arc::new(BinaryArray::from_iter(
                ids.iter().map(|id| (!null(id)).then(|| id.to_le_bytes())),
            )),
        ],
    )
    .unwrap()
}

/// Dates, timestamps with and without a time zone, decimals, doubles with
/// NaN and infinities, booleans, blobs and rows of NULLs, in Parquet and
/// inlined, restored inlined and in Parquet, values and row ids as at the
/// mark.
pub(crate) async fn restores_typed_columns(make: MakeLake) {
    let lake = make(Some(5)).await;
    create_with(&lake, "ty", &typed_columns()).unwrap();
    lake.append_parquet("ty", &typed(1..=20)).await;
    lake.append("ty", &typed(21..=24)).await;
    // MySQL stores some of these types in Parquet where the others inline.
    let inlines = lake.inlined_rows("ty") > 0;
    let mark = lake.head();
    for statement in [
        "DELETE FROM lake.main.ty WHERE id IN (2, 5, 22)",
        "UPDATE lake.main.ty SET f = 'NaN'::DOUBLE, b = NULL WHERE id >= 10 AND id <= 18",
        "UPDATE lake.main.ty SET blob = NULL, d = NULL WHERE id = 23",
    ] {
        lake.sql(statement).await.unwrap();
    }
    lake.append("ty", &typed(30..=31)).await;
    let result = restore_and_check(&lake, "ty", mark).await;
    assert!(result.rows_restored >= 12, "{result:?}");

    // A few rows only: restored inlined.
    let mark = lake.head();
    let inlined = lake.inlined_rows("ty");
    lake.sql("DELETE FROM lake.main.ty WHERE id IN (1, 3, 4)")
        .await
        .unwrap();
    let result = restore_and_check(&lake, "ty", mark).await;
    assert_eq!(counts(&result), (0, 3, 0));
    if inlines {
        assert_eq!(lake.inlined_rows("ty"), inlined + 3);
    }
}

/// Snapshots between the mark and the head expired, and their files cleaned
/// up: the rows at the mark are still restored whole.
pub(crate) async fn restores_across_expired_snapshots(make: MakeLake) {
    let lake = make(Some(5)).await;
    create(&lake, "t").await;
    lake.append_parquet("t", &batch(1..=20, "a")).await;
    lake.append("t", &batch(100..=101, "i")).await;
    let mark = lake.head();
    lake.sql("DELETE FROM lake.main.t WHERE id <= 8")
        .await
        .unwrap();
    let first = lake.head();
    lake.sql("UPDATE lake.main.t SET v = 'u' WHERE id >= 15")
        .await
        .unwrap();
    let second = lake.head();
    lake.append_parquet("t", &batch(30..=39, "b")).await;
    lake.sql("DELETE FROM lake.main.t WHERE id = 100")
        .await
        .unwrap();
    let expected = lake.rows_at("t", Some(mark)).await;
    let objects = files_under(&lake.data).len();
    let maintenance = lake.writer.maintenance().unwrap();
    maintenance
        .expire_snapshots(ExpireCriteria::Versions(vec![first, second]))
        .await
        .unwrap();
    maintenance
        .cleanup_old_files(
            Arc::new(object_store::local::LocalFileSystem::new()),
            datafusion_ducklake::maintenance::CleanupCriteria::All,
            false,
        )
        .await
        .unwrap();
    assert!(
        files_under(&lake.data).len() < objects,
        "the cleanup removed a file only the expired snapshots read"
    );
    let before = lake.head();
    let pre = lake.rows("t").await;
    let result = restore(&lake, "t", mark).await.unwrap();
    assert!(result.snapshot_id.is_some(), "{result:?}");
    assert_eq!(lake.rows("t").await, expected);
    assert_eq!(lake.rows_at("t", Some(before)).await, pre, "history kept");
    assert_eq!(lake.rows_at("t", Some(mark)).await, expected);
}

/// Nothing to undo commits nothing: no change to the table since the mark,
/// or changes that cancel out (rows inserted and deleted again).
pub(crate) async fn nothing_to_undo(make: MakeLake) {
    let lake = make(Some(5)).await;
    create(&lake, "t").await;
    create(&lake, "u").await;
    lake.append_parquet("t", &batch(1..=10, "a")).await;
    lake.append("t", &batch(100..=101, "i")).await;
    let mark = lake.head();
    lake.append_parquet("u", &batch(1..=10, "a")).await;
    let result = restore_and_check(&lake, "t", mark).await;
    assert_eq!(result, RestoreResult::default());

    lake.append_parquet("t", &batch(20..=29, "b")).await;
    lake.append("t", &batch(200..=201, "j")).await;
    lake.sql("DELETE FROM lake.main.t WHERE id BETWEEN 20 AND 29 OR id >= 200")
        .await
        .unwrap();
    let result = restore_and_check(&lake, "t", mark).await;
    assert_eq!(result, RestoreResult::default());
}

pub(crate) async fn all_scenarios(make: MakeLake) {
    Box::pin(restores_appends(Arc::clone(&make))).await;
    Box::pin(restores_deletes(Arc::clone(&make))).await;
    Box::pin(restores_beside_a_kept_delete_file(Arc::clone(&make))).await;
    Box::pin(restores_updates(Arc::clone(&make))).await;
    Box::pin(restores_keyed_mutations(Arc::clone(&make))).await;
    Box::pin(restores_replace(Arc::clone(&make))).await;
    Box::pin(restores_mixes(Arc::clone(&make))).await;
    Box::pin(restores_partitioned(Arc::clone(&make))).await;
    Box::pin(restores_into_several_files(Arc::clone(&make))).await;
    Box::pin(restores_across_flush_and_compaction(Arc::clone(&make))).await;
    Box::pin(refusals(Arc::clone(&make))).await;
    Box::pin(restores_through_a_catalog_read_memo(Arc::clone(&make))).await;
    Box::pin(restores_typed_columns(Arc::clone(&make))).await;
    Box::pin(restores_across_expired_snapshots(Arc::clone(&make))).await;
    Box::pin(nothing_to_undo(make)).await;
}

macro_rules! per_backend {
    ($($name:ident),* $(,)?) => {
        $(
            #[cfg(feature = "write-sqlite")]
            mod $name {
                #[tokio::test(flavor = "multi_thread")]
                async fn sqlite() {
                    Box::pin(super::$name(super::sqlite())).await;
                }

                #[cfg(feature = "write-duckdb")]
                #[tokio::test(flavor = "multi_thread")]
                async fn duckdb() {
                    Box::pin(super::$name(super::duckdb())).await;
                }
            }
        )*
    };
}

per_backend!(
    restores_appends,
    restores_deletes,
    restores_beside_a_kept_delete_file,
    restores_updates,
    restores_keyed_mutations,
    restores_replace,
    restores_mixes,
    restores_partitioned,
    restores_into_several_files,
    restores_across_flush_and_compaction,
    refusals,
    restores_through_a_catalog_read_memo,
    restores_typed_columns,
    restores_across_expired_snapshots,
    nothing_to_undo,
);

#[cfg(feature = "write-mysql")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
#[tokio::test(flavor = "multi_thread")]
async fn mysql_scenarios() {
    use testcontainers::runners::AsyncRunner;
    let container = testcontainers_modules::mysql::Mysql::default()
        .start()
        .await
        .unwrap();
    let port = container.get_host_port_ipv4(3306).await.unwrap();
    Box::pin(all_scenarios(crate::inlining_abstraction_tests::mysql(
        port,
    )))
    .await;
}

/// Every scenario on multi-catalog PostgreSQL; then a commit guard that
/// refuses fails the restore with its own error and commits nothing, and one
/// that approves lets it commit.
#[cfg(feature = "write-postgres")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
#[tokio::test(flavor = "multi_thread")]
async fn multicatalog_scenarios() {
    use datafusion_ducklake::MetadataWriter;
    use std::sync::atomic::{AtomicBool, Ordering};
    use testcontainers::runners::AsyncRunner;

    #[derive(Debug)]
    struct Refused;
    impl std::fmt::Display for Refused {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("lease lost")
        }
    }
    impl std::error::Error for Refused {}

    #[derive(Debug, Default)]
    struct Guard {
        refuse: AtomicBool,
    }
    #[async_trait::async_trait]
    impl datafusion_ducklake::PostgresCommitGuard for Guard {
        async fn before_commit(
            &self,
            _conn: &mut sqlx::PgConnection,
        ) -> Result<(), datafusion_ducklake::CommitGuardError> {
            if self.refuse.load(Ordering::SeqCst) {
                Err(Box::new(Refused))
            } else {
                Ok(())
            }
        }
    }

    let container = testcontainers_modules::postgres::Postgres::default()
        .start()
        .await
        .unwrap();
    let port = container.get_host_port_ipv4(5432).await.unwrap();
    let pool = sqlx::PgPool::connect(&format!(
        "postgresql://postgres:postgres@127.0.0.1:{port}/postgres"
    ))
    .await
    .unwrap();
    datafusion_ducklake::initialize_multicatalog_schema(&pool)
        .await
        .unwrap();
    Box::pin(all_scenarios(
        crate::inlining_abstraction_tests::multicatalog(pool.clone()),
    ))
    .await;

    // On the heap, apart from the scenarios: one test body awaiting both
    // outgrows a test thread's stack in a debug build.
    Box::pin(guarded(pool)).await;

    async fn guarded(pool: sqlx::PgPool) {
        let dir = tempfile::TempDir::new().unwrap();
        let data = dir.path().join("data");
        std::fs::create_dir_all(&data).unwrap();
        let catalog_id = datafusion_ducklake::MulticatalogManager::new(pool.clone())
            .create_catalog("guarded")
            .await
            .unwrap();
        let guard = Arc::new(Guard::default());
        let writer =
            datafusion_ducklake::PostgresMetadataWriter::with_pool(pool.clone(), catalog_id)
                .await
                .unwrap()
                .with_data_inlining_row_limit(5)
                .with_commit_guard(guard.clone());
        writer.set_data_path(data.to_str().unwrap()).unwrap();
        let provider = datafusion_ducklake::MulticatalogProvider::with_pool(pool, "guarded")
            .await
            .unwrap();
        let lake = Lake {
            writer: Arc::new(writer),
            provider: Arc::new(provider),
            data,
            inlines: true,
            _dir: dir,
        };
        create(&lake, "t").await;
        lake.append_parquet("t", &batch(1..=20, "a")).await;
        lake.append("t", &batch(100..=101, "i")).await;
        let mark = lake.head();
        lake.sql("DELETE FROM lake.main.t WHERE id <= 12")
            .await
            .unwrap();
        lake.append_parquet("t", &batch(30..=39, "b")).await;
        lake.sql("DELETE FROM lake.main.t WHERE id >= 30 AND id <= 31")
            .await
            .unwrap();
        let head = lake.head();
        let rows = lake.rows("t").await;
        let objects = files_under(&lake.data);
        guard.refuse.store(true, Ordering::SeqCst);
        let error = restore(&lake, "t", mark).await.unwrap_err();
        assert!(
            matches!(&error, DuckLakeError::CommitRefused(source) if source.is::<Refused>()),
            "{error}"
        );
        assert_eq!(lake.head(), head);
        assert_eq!(lake.rows("t").await, rows);
        assert_eq!(
            files_under(&lake.data),
            objects,
            "the refused restore leaves none of the files it wrote"
        );
        guard.refuse.store(false, Ordering::SeqCst);
        restore_and_check(&lake, "t", mark).await;
    }
}

/// Single-catalog PostgreSQL stages no row mutation, so it refuses a
/// restore as it refuses `UPDATE` and `DELETE`, committing nothing.
#[cfg(feature = "write-postgres")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
#[tokio::test(flavor = "multi_thread")]
async fn postgres_single_catalog_refuses() {
    use testcontainers::runners::AsyncRunner;
    let container = testcontainers_modules::postgres::Postgres::default()
        .start()
        .await
        .unwrap();
    let port = container.get_host_port_ipv4(5432).await.unwrap();
    let lake = crate::inlining_abstraction_tests::postgres_single(port)(Some(5)).await;
    create(&lake, "t").await;
    lake.append_parquet("t", &batch(1..=10, "a")).await;
    let mark = lake.head();
    lake.append_parquet("t", &batch(11..=20, "b")).await;
    let head = lake.head();
    let error = restore(&lake, "t", mark).await.unwrap_err();
    assert!(
        matches!(&error, DuckLakeError::Unsupported(m) if m.contains("stages row mutations")),
        "{error}"
    );
    assert_eq!(lake.head(), head);
}

/// A column renamed since the mark (by official DuckLake, as this crate has
/// no rename) refuses the restore.
#[cfg(feature = "write-duckdb")]
#[tokio::test(flavor = "multi_thread")]
async fn rename_since_the_mark_refuses() {
    use crate::inlined_mutation_differential_tests as differential;
    let lake = duckdb()(Some(5)).await;
    create(&lake, "t").await;
    lake.append_parquet("t", &batch(1..=10, "a")).await;
    let mark = lake.head();
    let Lake {
        writer,
        provider,
        data,
        _dir: dir,
        ..
    } = lake;
    drop((writer, provider));
    let path = dir.path().join("lake.ducklake");
    differential::official(
        &differential::Metadata::Duckdb(path.clone()),
        &data,
        &["ALTER TABLE lake.main.t RENAME COLUMN v TO w".to_string()],
    );
    let writer =
        datafusion_ducklake::DuckdbMetadataWriter::new(path.display().to_string()).unwrap();
    let provider = writer.metadata_provider();
    let lake = Lake {
        writer: Arc::new(writer),
        provider: Arc::new(provider),
        data,
        inlines: true,
        _dir: dir,
    };
    let head = lake.head();
    assert!(head > mark);
    let error = restore(&lake, "t", mark).await.unwrap_err();
    assert!(
        matches!(&error, DuckLakeError::Unsupported(m) if m.contains("columns changed")),
        "{error}"
    );
    assert_eq!(lake.head(), head);
}

// ---------------------------------------------------------------------------
// Differential against official DuckLake
// ---------------------------------------------------------------------------

/// A history written by this crate on a DuckDB catalog — Parquet and inlined
/// appends, inlined and file deletes, updates of both, a keyed upsert, a flush
/// and a merge, more deletes after them — then restored by this crate. Official
/// DuckLake then reads `rowid, *` at the head equal to `AT (VERSION => mark)`
/// and every snapshot from the mark through the restore as this crate reads
/// it, reports this crate's change feed over each of those snapshots and over
/// the restore with the write before it, and goes on writing the table: an
/// update, an insert, an expiry and the removal of old and orphaned files,
/// each read back the same by official and this crate. A second table's
/// restore of few rows, inlined with their row ids, reads the same way.
#[cfg(feature = "write-duckdb")]
#[tokio::test(flavor = "multi_thread")]
async fn official_reads_a_restore() {
    use crate::inlined_mutation_differential_tests as differential;

    let lake = duckdb()(Some(5)).await;
    create(&lake, "t").await;
    lake.append_parquet("t", &batch(1..=30, "a")).await;
    lake.append_parquet("t", &batch(31..=60, "b")).await;
    lake.append("t", &batch(100..=102, "i")).await;
    let mark = lake.head();
    for statement in [
        "DELETE FROM lake.main.t WHERE id IN (3, 4)",
        "DELETE FROM lake.main.t WHERE id BETWEEN 35 AND 50",
        "UPDATE lake.main.t SET v = 'u' WHERE id <= 2",
        "UPDATE lake.main.t SET v = 'w' WHERE id = 100",
        "DELETE FROM lake.main.t WHERE id = 101",
    ] {
        lake.sql(statement).await.unwrap();
    }
    lake.append_parquet("t", &batch(200..=239, "c")).await;
    lake.append("t", &batch([300], "x")).await;
    let (ctx, table) = lake.table("t").await;
    table
        .keyed_mutation(&["id"])
        .upsert_by_keys(&ctx.state(), vec![batch([5, 6, 400], "k")])
        .await
        .unwrap();
    let always =
        DuckLakeWriteOptions::default().with_inlined_flush_policy(InlinedFlushPolicy::always());
    let (ctx, table) = table_with(&lake, "t", always).await;
    assert!(
        table
            .merge_adjacent_files(&ctx.state(), MergeOptions::default())
            .await
            .unwrap()
            .did_work()
    );
    for statement in [
        "DELETE FROM lake.main.t WHERE id = 205",
        "UPDATE lake.main.t SET v = 'z' WHERE id BETWEEN 10 AND 20",
    ] {
        lake.sql(statement).await.unwrap();
    }
    let before = lake.head();
    let result = restore_and_check(&lake, "t", mark).await;
    let restored = result.snapshot_id.unwrap();

    // What this crate reads and reports, before official opens the catalog.
    let snapshots = lake.provider.snapshot_ids_between(mark, restored).unwrap();
    let mut ours_at = Vec::new();
    for snapshot in &snapshots {
        ours_at.push((*snapshot, lake.rows_at("t", Some(*snapshot)).await));
    }
    // Each snapshot alone, and the restore with the write before it. A longer
    // window is left out: over an inlined row updated twice in it, official's
    // feed repeats the row and reports the later update as an insert.
    let mut windows: Vec<(i64, i64)> = snapshots[1..].iter().map(|s| (*s, *s)).collect();
    windows.push((before, restored));
    let feeds = SessionContext::new();
    datafusion_ducklake::register_ducklake_functions(&feeds, Arc::clone(&lake.provider));
    let mut ours = Vec::new();
    for (from, through) in windows {
        let batches = feeds
            .sql(&format!(
                "SELECT * FROM ducklake_table_changes('main.t', {from}, {through}) \
                 ORDER BY snapshot_id, rowid, change_type"
            ))
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        ours.push(((from, through), differential::arrow_lines(&batches)));
    }
    drop(feeds);
    let expected = lake.rows_at("t", Some(mark)).await;

    // A restore of few rows, inlined with their row ids.
    create(&lake, "s").await;
    lake.append_parquet("s", &batch(1..=10, "a")).await;
    let mark_s = lake.head();
    lake.sql("DELETE FROM lake.main.s WHERE id IN (2, 3)")
        .await
        .unwrap();
    let inlined = lake.inlined_rows("s");
    restore_and_check(&lake, "s", mark_s).await;
    assert_eq!(lake.inlined_rows("s"), inlined + 2);
    let expected_s = lake.rows_at("s", Some(mark_s)).await;
    let Lake {
        writer,
        provider,
        data,
        _dir: dir,
        ..
    } = lake;
    drop((writer, provider));
    let path = dir.path().join("lake.ducklake");
    let metadata = differential::Metadata::Duckdb(path.clone());

    let connection = differential::duckdb_connection();
    differential::attach_lake(&connection, &metadata, &data, false);
    let rows_sql = "SELECT rowid, id, v FROM lake.t ORDER BY rowid";
    let rows_at = |snapshot: i64| {
        format!("SELECT rowid, id, v FROM lake.t AT (VERSION => {snapshot}) ORDER BY rowid")
    };
    assert_eq!(differential::query_lines(&connection, rows_sql), expected);
    assert_eq!(
        differential::query_lines(&connection, &rows_at(mark)),
        expected
    );
    for (snapshot, rows) in &ours_at {
        assert_eq!(
            &differential::query_lines(&connection, &rows_at(*snapshot)),
            rows,
            "official reads snapshot {snapshot} as this crate does"
        );
    }
    for ((from, through), ours) in &ours {
        let official = differential::query_lines(
            &connection,
            &format!(
                "SELECT * FROM ducklake_table_changes('lake', 'main', 't', {from}, {through}) \
                 ORDER BY snapshot_id, rowid, change_type"
            ),
        );
        assert_eq!(&official, ours, "window {from}..={through}");
    }
    let listed = differential::query_lines(
        &connection,
        "SELECT snapshot_id FROM ducklake_snapshots('lake') ORDER BY snapshot_id",
    );
    assert!(listed.contains(&restored.to_string()));
    let s_sql = "SELECT rowid, id, v FROM lake.s ORDER BY rowid";
    assert_eq!(differential::query_lines(&connection, s_sql), expected_s);
    assert_eq!(
        differential::query_lines(
            &connection,
            &format!("SELECT rowid, id, v FROM lake.s AT (VERSION => {mark_s}) ORDER BY rowid")
        ),
        expected_s
    );

    // Official goes on writing the restored table. Only writes that add no
    // data file: official cannot register one in a catalog this crate created
    // (it inserts `ducklake_data_file` rows by position, and this crate's
    // columns are ordered differently), restore or not, so its flush, merge
    // and rewrite are not run here.
    let mut current = expected.clone();
    for (statement, change) in [
        (
            "UPDATE lake.t SET v = 'after' WHERE id = 1",
            Some((1, "after")),
        ),
        ("INSERT INTO lake.t VALUES (900, 'o')", None),
        (
            "CALL ducklake_expire_snapshots('lake', older_than => now())",
            None,
        ),
        (
            "CALL ducklake_cleanup_old_files('lake', cleanup_all => true)",
            None,
        ),
        (
            "CALL ducklake_delete_orphaned_files('lake', cleanup_all => true)",
            None,
        ),
    ] {
        connection
            .execute_batch(statement)
            .unwrap_or_else(|e| panic!("official `{statement}`: {e}"));
        let read = differential::query_lines(&connection, rows_sql);
        if let Some((id, value)) = change {
            current = current
                .iter()
                .map(|row| {
                    let cells: Vec<&str> = row.split(',').collect();
                    if cells[1] == id.to_string() {
                        format!("{},{id},{value}", cells[0])
                    } else {
                        row.clone()
                    }
                })
                .collect();
        } else if statement.starts_with("INSERT") {
            let inserted = read
                .iter()
                .find(|row| row.ends_with(",900,o"))
                .expect("the inserted row")
                .clone();
            current.push(inserted);
        }
        assert_eq!(read, current, "after `{statement}`");
    }
    assert_eq!(differential::query_lines(&connection, s_sql), expected_s);
    drop(connection);

    // And this crate reads what official left.
    let writer =
        datafusion_ducklake::DuckdbMetadataWriter::new(path.display().to_string()).unwrap();
    let provider: Arc<dyn datafusion_ducklake::MetadataProvider> =
        Arc::new(writer.metadata_provider());
    let head = provider.get_current_snapshot().unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog(
        "lake",
        Arc::new(
            DuckLakeCatalog::with_snapshot(provider, head)
                .unwrap()
                .with_row_lineage(true),
        ),
    );
    let batches = ctx
        .sql("SELECT rowid, id, v FROM lake.main.t ORDER BY rowid")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(differential::arrow_lines(&batches), current);
}
