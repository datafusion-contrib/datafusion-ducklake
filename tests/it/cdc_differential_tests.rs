#![cfg(feature = "metadata-duckdb")]
//! Differential CDC conformance tests.
//!
//! Each test builds a DuckLake catalog, runs the OFFICIAL DuckDB ducklake
//! extension's change feeds (`ducklake_table_changes` / `ducklake_table_deletions`)
//! and this crate's implementations over the identical catalog, canonicalizes
//! both outputs, and diffs them — so conformance of rows, rowids, snapshot_ids
//! and change_types is proven by execution, not asserted by hand.
//!
//! Known surface differences are bridged by explicit NORMALIZERS. Each is a
//! ratchet: when the crate converges on the official behavior, delete the
//! normalizer and the diff tightens automatically.
//!
//! * NORMALIZER-DELETIONS-CHANGE-TYPE — the crate's `ducklake_table_deletions`
//!   materializes a constant `change_type='delete'` column that official's
//!   function does not have (official also exposes rowid/snapshot_id as
//!   virtual columns rather than in `SELECT *` — inherent to DataFusion's
//!   lack of virtual columns, documented in COMPATIBILITY.md). The
//!   `table_changes` column list is asserted positionally identical to
//!   official's; the deletions list must be official's with `change_type`
//!   inserted after `(snapshot_id, rowid)`.
//!
//! Schema evolution between snapshots IS covered, at the bottom of this file:
//! renames (top-level, nested, and a cycle), a column dropped and re-added under
//! the same name, and a column added between the queried snapshots — over both
//! plan shapes and all three feeds. Encrypted (PME) catalogs cannot be diffed
//! here at all (the extension writing these fixtures does not encrypt), and
//! compaction REWRITES — as opposed to the adjacent-file merges two scenarios
//! below do exercise — are still uncovered.

use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;

use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::prelude::*;
use datafusion_ducklake::{
    DuckLakeCatalog, DuckdbMetadataProvider, MetadataProvider, register_ducklake_functions,
};
use duckdb::types::Value;
use tempfile::TempDir;

fn box_err<E: std::error::Error + Send + Sync + 'static>(e: E) -> DataFusionError {
    DataFusionError::External(Box::new(e))
}

/// Write a DuckLake catalog at `path` by running `statements` through the
/// official extension. The connection drops at return, releasing all locks.
fn write_catalog(path: &Path, statements: &[&str]) -> DataFusionResult<()> {
    let conn = official_connection(path)?;
    for s in statements {
        conn.execute(s, []).map_err(box_err)?;
    }
    Ok(())
}

/// Open an in-memory DuckDB connection with the official ducklake extension
/// loaded and the catalog at `path` attached as `c`.
fn official_connection(path: &Path) -> DataFusionResult<duckdb::Connection> {
    let conn = duckdb::Connection::open_in_memory().map_err(box_err)?;
    crate::common::ensure_ducklake_installed();
    conn.execute("LOAD ducklake;", []).map_err(box_err)?;
    conn.execute(&format!("ATTACH 'ducklake:{}' AS c;", path.display()), [])
        .map_err(box_err)?;
    Ok(conn)
}

/// A row canonicalized for cross-engine comparison: the CDC metadata columns
/// extracted by name, plus the table cells rendered to strings.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct CanonRow {
    snapshot_id: i64,
    rowid: Option<i64>,
    /// `None` for feeds that have no change_type column (official deletions).
    change_type: Option<String>,
    cells: Vec<String>,
}

/// One engine's canonicalized feed output: sorted rows, the full column-name
/// list in output order, and the residual (non-CDC) column names.
#[derive(Debug, Clone, PartialEq, Eq)]
struct CanonFeed {
    all_columns: Vec<String>,
    table_columns: Vec<String>,
    rows: Vec<CanonRow>,
}

impl CanonFeed {
    fn new(all_columns: Vec<String>, table_columns: Vec<String>, mut rows: Vec<CanonRow>) -> Self {
        rows.sort();
        Self {
            all_columns,
            table_columns,
            rows,
        }
    }
}

/// Render a duckdb value to the shared canonical string form.
fn duckdb_cell(v: &Value) -> String {
    match v {
        Value::Null => "NULL".to_string(),
        Value::Boolean(b) => b.to_string(),
        Value::TinyInt(i) => i.to_string(),
        Value::SmallInt(i) => i.to_string(),
        Value::Int(i) => i.to_string(),
        Value::BigInt(i) => i.to_string(),
        Value::Float(f) => format!("{:?}", f),
        Value::Double(f) => format!("{:?}", f),
        Value::Text(s) => s.clone(),
        other => panic!(
            "unsupported duckdb value in differential scenario (keep scenario column \
             types within the canonicalizer's set): {other:?}"
        ),
    }
}

/// Render an arrow cell to the shared canonical string form.
fn arrow_cell(batch: &RecordBatch, col: usize, row: usize) -> String {
    use arrow::array::*;
    let a = batch.column(col);
    if a.is_null(row) {
        return "NULL".to_string();
    }
    match a.data_type() {
        DataType::Boolean => a
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap()
            .value(row)
            .to_string(),
        DataType::Int8 => a
            .as_any()
            .downcast_ref::<Int8Array>()
            .unwrap()
            .value(row)
            .to_string(),
        DataType::Int16 => a
            .as_any()
            .downcast_ref::<Int16Array>()
            .unwrap()
            .value(row)
            .to_string(),
        DataType::Int32 => a
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap()
            .value(row)
            .to_string(),
        DataType::Int64 => a
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(row)
            .to_string(),
        DataType::Float32 => {
            format!(
                "{:?}",
                a.as_any()
                    .downcast_ref::<Float32Array>()
                    .unwrap()
                    .value(row)
            )
        },
        DataType::Float64 => {
            format!(
                "{:?}",
                a.as_any()
                    .downcast_ref::<Float64Array>()
                    .unwrap()
                    .value(row)
            )
        },
        DataType::Utf8 => a
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .value(row)
            .to_string(),
        DataType::Utf8View => a
            .as_any()
            .downcast_ref::<StringViewArray>()
            .unwrap()
            .value(row)
            .to_string(),
        other => panic!(
            "unsupported arrow type in differential scenario (keep scenario column \
             types within the canonicalizer's set): {other:?}"
        ),
    }
}

/// Split raw named cells into a CanonRow, pulling the CDC columns out by name.
/// `require_change_type` distinguishes feeds that must carry one from feeds
/// that must not (official deletions).
fn canon_row(
    names: &[String],
    raw: Vec<String>,
    require_change_type: bool,
) -> (CanonRow, Vec<String>) {
    let mut snapshot_id = None;
    let mut rowid = None;
    let mut change_type = None;
    let mut cells = Vec::new();
    let mut table_columns = Vec::new();
    for (name, value) in names.iter().zip(raw) {
        match name.as_str() {
            "snapshot_id" => snapshot_id = Some(value.parse::<i64>().expect("snapshot_id i64")),
            "rowid" => {
                rowid = Some(if value == "NULL" {
                    None
                } else {
                    Some(value.parse::<i64>().expect("rowid i64"))
                })
            },
            "change_type" => change_type = Some(value),
            _ => {
                table_columns.push(name.clone());
                cells.push(value);
            },
        }
    }
    assert_eq!(
        change_type.is_some(),
        require_change_type,
        "change_type presence mismatch (columns: {names:?})"
    );
    (
        CanonRow {
            snapshot_id: snapshot_id.expect("snapshot_id column present"),
            rowid: rowid.expect("rowid column present"),
            change_type,
            cells,
        },
        table_columns,
    )
}

/// Run `sql` on the official connection and canonicalize.
fn official_feed(
    conn: &duckdb::Connection,
    sql: &str,
    require_change_type: bool,
) -> DataFusionResult<CanonFeed> {
    let mut stmt = conn.prepare(sql).map_err(box_err)?;
    let raw_rows: Vec<Vec<Value>> = stmt
        .query_map([], |row| {
            let mut out = Vec::new();
            let mut i = 0;
            while let Ok(v) = row.get::<usize, Value>(i) {
                out.push(v);
                i += 1;
            }
            Ok(out)
        })
        .map_err(box_err)?
        .collect::<Result<_, _>>()
        .map_err(box_err)?;
    let names: Vec<String> = stmt.column_names().into_iter().collect();

    let mut rows = Vec::new();
    let mut table_columns = Vec::new();
    for raw in raw_rows {
        let rendered: Vec<String> = raw.iter().map(duckdb_cell).collect();
        let (row, cols) = canon_row(&names, rendered, require_change_type);
        table_columns = cols;
        rows.push(row);
    }
    if rows.is_empty() {
        // No rows to derive residual names from; leave empty (callers skip the
        // name assertions for empty feeds).
        table_columns.clear();
        return Ok(CanonFeed::new(Vec::new(), table_columns, rows));
    }
    Ok(CanonFeed::new(names, table_columns, rows))
}

/// Run `sql` through the crate (DataFusion) and canonicalize.
async fn crate_feed(
    ctx: &SessionContext,
    sql: &str,
    require_change_type: bool,
) -> DataFusionResult<CanonFeed> {
    let batches = ctx.sql(sql).await?.collect().await?;
    let mut rows = Vec::new();
    let mut table_columns = Vec::new();
    let mut all_columns = Vec::new();
    for batch in &batches {
        let names: Vec<String> = batch
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();
        for r in 0..batch.num_rows() {
            let rendered: Vec<String> = (0..batch.num_columns())
                .map(|c| arrow_cell(batch, c, r))
                .collect();
            let (row, cols) = canon_row(&names, rendered, require_change_type);
            table_columns = cols;
            rows.push(row);
        }
        all_columns = names;
    }
    if rows.is_empty() {
        all_columns.clear();
    }
    Ok(CanonFeed::new(all_columns, table_columns, rows))
}

async fn crate_context(path: &Path) -> DataFusionResult<SessionContext> {
    let path = path.to_str().expect("utf8 path");
    let provider = DuckdbMetadataProvider::new(path)?;
    let provider_arc: Arc<dyn MetadataProvider> = Arc::new(DuckdbMetadataProvider::new(path)?);
    let catalog = DuckLakeCatalog::new(provider)?;
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));
    register_ducklake_functions(&ctx, provider_arc);
    Ok(ctx)
}

/// The snapshot windows to diff for a catalog whose snapshots are `ids`
/// (official inclusive-both-ends convention): every single snapshot from the
/// table's creation on, every adjacent pair, every suffix, and the full range.
fn windows(ids: &[i64]) -> Vec<(i64, i64)> {
    // Snapshot 0 is the catalog-initialization snapshot; the scenario table
    // exists from snapshot 1 onward. Official table_changes resolves the table
    // at `end_snapshot`, so keep end >= 1.
    let usable: Vec<i64> = ids.iter().copied().filter(|&s| s >= 1).collect();
    let &max = usable.last().expect("at least one snapshot");
    let mut out = HashSet::new();
    out.insert((0, max));
    for &s in &usable {
        out.insert((s, s));
        out.insert((s, max));
    }
    for pair in usable.windows(2) {
        out.insert((pair[0], pair[1]));
    }
    let mut out: Vec<_> = out.into_iter().collect();
    out.sort();
    out
}

fn snapshot_ids(conn: &duckdb::Connection) -> DataFusionResult<Vec<i64>> {
    let mut stmt = conn
        .prepare("SELECT snapshot_id FROM ducklake_snapshots('c') ORDER BY snapshot_id")
        .map_err(box_err)?;
    let ids: Vec<i64> = stmt
        .query_map([], |row| row.get(0))
        .map_err(box_err)?
        .collect::<Result<_, _>>()
        .map_err(box_err)?;
    Ok(ids)
}

fn pretty(feed: &CanonFeed) -> String {
    let mut s = format!("  table columns: {:?}\n", feed.table_columns);
    for r in &feed.rows {
        s.push_str(&format!(
            "  snap={} rowid={:?} type={:?} cells={:?}\n",
            r.snapshot_id, r.rowid, r.change_type, r.cells
        ));
    }
    s
}

fn assert_feeds_match(context: &str, official: &CanonFeed, ours: &CanonFeed) {
    // Empty feeds carry no residual column names to compare.
    if !official.rows.is_empty() && !ours.rows.is_empty() {
        assert_eq!(
            official.table_columns, ours.table_columns,
            "{context}: table-column names/order diverge"
        );
    }
    assert_eq!(
        official.rows,
        ours.rows,
        "{context}: rows diverge\n--- official ---\n{}--- crate ---\n{}",
        pretty(official),
        pretty(ours)
    );
}

/// Build the catalog from `statements`, then diff both CDC feeds between the
/// two engines over every derived snapshot window.
async fn assert_cdc_conformance(table: &str, statements: &[&str]) -> DataFusionResult<()> {
    let tmp = TempDir::new().map_err(box_err)?;
    let path = tmp.path().join("diff.ducklake");
    write_catalog(&path, statements)?;

    // Official side first, then drop the connection before the crate's
    // provider opens the metadata database.
    let mut official: Vec<((i64, i64), CanonFeed, CanonFeed, CanonFeed)> = Vec::new();
    {
        let conn = official_connection(&path)?;
        for (a, b) in windows(&snapshot_ids(&conn)?) {
            let changes = official_feed(
                &conn,
                &format!("SELECT * FROM ducklake_table_changes('c', 'main', '{table}', {a}, {b})"),
                true,
            )?;
            // rowid/snapshot_id are virtual on the official deletions and
            // insertions scans: project them explicitly; neither feed has a
            // change_type column.
            let deletions = official_feed(
                &conn,
                &format!(
                    "SELECT snapshot_id, rowid, * FROM \
                     ducklake_table_deletions('c', 'main', '{table}', {a}, {b})"
                ),
                false,
            )?;
            let insertions = official_feed(
                &conn,
                &format!(
                    "SELECT snapshot_id, rowid, * FROM \
                     ducklake_table_insertions('c', 'main', '{table}', {a}, {b})"
                ),
                false,
            )?;
            official.push(((a, b), changes, deletions, insertions));
        }
    }

    let ctx = crate_context(&path).await?;
    for ((a, b), official_changes, official_deletions, official_insertions) in official {
        // The insertions feed has no crate-side surface difference at all:
        // `SELECT *` must match official's explicit projection VERBATIM.
        let crate_insertions = crate_feed(
            &ctx,
            &format!("SELECT * FROM ducklake_table_insertions('main.{table}', {a}, {b})"),
            false,
        )
        .await?;
        if !official_insertions.all_columns.is_empty() && !crate_insertions.all_columns.is_empty() {
            assert_eq!(
                official_insertions.all_columns, crate_insertions.all_columns,
                "table_insertions window [{a},{b}]: column list diverges from official"
            );
        }
        assert_feeds_match(
            &format!("table_insertions window [{a},{b}]"),
            &official_insertions,
            &crate_insertions,
        );

        // Bounds are inclusive on both ends, matching official DuckLake.
        let crate_changes = crate_feed(
            &ctx,
            &format!("SELECT * FROM ducklake_table_changes('main.{table}', {a}, {b})"),
            true,
        )
        .await?;
        let crate_deletions = crate_feed(
            &ctx,
            &format!("SELECT * FROM ducklake_table_deletions('main.{table}', {a}, {b})"),
            true,
        )
        .await?;

        // Column placement is converged for table_changes: the crate's
        // `SELECT *` column list must be positionally IDENTICAL to official's
        // (snapshot_id, rowid, change_type, table columns).
        if !official_changes.all_columns.is_empty() && !crate_changes.all_columns.is_empty() {
            assert_eq!(
                official_changes.all_columns, crate_changes.all_columns,
                "table_changes window [{a},{b}]: full column list diverges from official"
            );
        }

        // The crate's table_changes must match official's VERBATIM — inserts,
        // update pre/postimages, and pure deletes included.
        assert_feeds_match(
            &format!("table_changes window [{a},{b}]"),
            &official_changes,
            &crate_changes,
        );

        // NORMALIZER-DELETIONS-CHANGE-TYPE: the crate's deletions column list
        // must be official's `(snapshot_id, rowid, <table cols>)` with the
        // crate's constant change_type column inserted after rowid.
        if !official_deletions.all_columns.is_empty() && !crate_deletions.all_columns.is_empty() {
            let mut expected = official_deletions.all_columns.clone();
            expected.insert(2, "change_type".to_string());
            assert_eq!(
                expected, crate_deletions.all_columns,
                "table_deletions window [{a},{b}]: column list diverges from official + change_type"
            );
        }

        // NORMALIZER-DELETE-ROUTING (half 2): the crate's table_deletions must
        // match official's ducklake_table_deletions (all deleted rows, update
        // preimages included). The crate adds a constant change_type='delete'
        // column official lacks; strip it after asserting the constant.
        for r in &crate_deletions.rows {
            assert_eq!(
                r.change_type.as_deref(),
                Some("delete"),
                "crate table_deletions must tag every row 'delete'"
            );
        }
        let crate_deletions_stripped = CanonFeed::new(
            official_deletions.all_columns.clone(),
            crate_deletions.table_columns.clone(),
            crate_deletions
                .rows
                .into_iter()
                .map(|mut r| {
                    r.change_type = None;
                    r
                })
                .collect(),
        );
        assert_feeds_match(
            &format!("table_deletions window [{a},{b}]"),
            &official_deletions,
            &crate_deletions_stripped,
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Scenarios
// ---------------------------------------------------------------------------

/// Multiple insert batches across snapshots, including NULL cells: rowids must
/// continue across files and every row surfaces as `insert`.
#[tokio::test]
async fn diff_plain_inserts_multi_snapshot() -> DataFusionResult<()> {
    assert_cdc_conformance(
        "t",
        &[
            "CREATE TABLE c.t(id INTEGER, name VARCHAR);",
            "INSERT INTO c.t VALUES (1, 'a'), (2, NULL), (3, 'c');",
            "INSERT INTO c.t VALUES (4, 'd');",
            "INSERT INTO c.t VALUES (5, 'e'), (6, 'f');",
        ],
    )
    .await
}

/// An UPDATE must pair into update_preimage/update_postimage with a preserved
/// rowid on both engines.
#[tokio::test]
async fn diff_update_pairing() -> DataFusionResult<()> {
    assert_cdc_conformance(
        "t",
        &[
            "CREATE TABLE c.t(id INTEGER, val VARCHAR);",
            "INSERT INTO c.t VALUES (1, 'one'), (2, 'two'), (3, 'three');",
            "UPDATE c.t SET val = 'TWO' WHERE id = 2;",
        ],
    )
    .await
}

/// Two successive UPDATEs of the same row: the second update's preimage reads
/// from a rewritten file whose rowid is embedded, not synthesized.
#[tokio::test]
async fn diff_update_of_update() -> DataFusionResult<()> {
    assert_cdc_conformance(
        "t",
        &[
            "CREATE TABLE c.t(id INTEGER, val VARCHAR);",
            "INSERT INTO c.t VALUES (1, 'one'), (2, 'two');",
            "UPDATE c.t SET val = 'TWO' WHERE id = 2;",
            "UPDATE c.t SET val = 'TWO-AGAIN' WHERE id = 2;",
        ],
    )
    .await
}

/// A partial DELETE: deleted rows carry their original rowids and old values.
#[tokio::test]
async fn diff_partial_delete() -> DataFusionResult<()> {
    assert_cdc_conformance(
        "t",
        &[
            "CREATE TABLE c.t(id INTEGER, name VARCHAR);",
            "INSERT INTO c.t VALUES (1, 'a'), (2, 'b'), (3, 'c'), (4, 'd');",
            "DELETE FROM c.t WHERE id IN (2, 4);",
        ],
    )
    .await
}

/// Deleting every row of a file (full-file delete has no delete file — the
/// data file is simply retired).
#[tokio::test]
async fn diff_full_file_delete() -> DataFusionResult<()> {
    assert_cdc_conformance(
        "t",
        &[
            "CREATE TABLE c.t(id INTEGER, name VARCHAR);",
            "INSERT INTO c.t VALUES (1, 'a'), (2, 'b');",
            "DELETE FROM c.t;",
        ],
    )
    .await
}

/// Insert → delete-all → re-insert → partial delete: delete files spanning
/// several files and generations.
#[tokio::test]
async fn diff_delete_then_reinsert() -> DataFusionResult<()> {
    assert_cdc_conformance(
        "t",
        &[
            "CREATE TABLE c.t(id INTEGER);",
            "INSERT INTO c.t VALUES (1), (2), (3);",
            "DELETE FROM c.t;",
            "INSERT INTO c.t VALUES (4), (5), (6), (7);",
            "DELETE FROM c.t WHERE id IN (5, 6);",
        ],
    )
    .await
}

/// A mixed lifecycle across many snapshots: multi-file inserts, an update, a
/// delete, and a trailing insert.
#[tokio::test]
async fn diff_mixed_lifecycle() -> DataFusionResult<()> {
    assert_cdc_conformance(
        "t",
        &[
            "CREATE TABLE c.t(id INTEGER, qty INTEGER, tag VARCHAR);",
            "INSERT INTO c.t VALUES (1, 10, 'x'), (2, 20, 'y'), (3, 30, 'z');",
            "INSERT INTO c.t VALUES (4, 40, NULL);",
            "UPDATE c.t SET qty = 25 WHERE id = 2;",
            "DELETE FROM c.t WHERE id = 3;",
            "INSERT INTO c.t VALUES (5, 50, 'w');",
        ],
    )
    .await
}

/// Wider scalar types (BOOLEAN, BIGINT, DOUBLE) with NULLs through insert,
/// update and delete.
#[tokio::test]
async fn diff_wide_scalar_types() -> DataFusionResult<()> {
    assert_cdc_conformance(
        "t",
        &[
            "CREATE TABLE c.t(id INTEGER, big BIGINT, score DOUBLE, ok BOOLEAN, name VARCHAR);",
            "INSERT INTO c.t VALUES \
                (1, 9007199254740993, 1.5, true, 'a'), \
                (2, NULL, NULL, false, NULL), \
                (3, -1, 0.25, NULL, 'c');",
            "UPDATE c.t SET score = 2.75, ok = true WHERE id = 2;",
            "DELETE FROM c.t WHERE id = 1;",
        ],
    )
    .await
}

/// An UPDATE that rewrites every row of the table in one snapshot.
#[tokio::test]
async fn diff_update_all_rows() -> DataFusionResult<()> {
    assert_cdc_conformance(
        "t",
        &[
            "CREATE TABLE c.t(id INTEGER, val INTEGER);",
            "INSERT INTO c.t VALUES (1, 100), (2, 200), (3, 300);",
            "UPDATE c.t SET val = val + 1;",
        ],
    )
    .await
}

/// Compaction: several small inserts merged into one partial file, then a
/// post-compaction insert. Official attributes each merged row to its ORIGIN
/// snapshot in the change feed (via the embedded
/// `_ducklake_internal_snapshot_id` column); windows starting after the
/// merged file's begin_snapshot must still see the in-window origin rows, and
/// the merge itself emits no CDC events.
///
/// The INSTALLED (1.4-era) extension has two since-fixed gaps here, so this
/// scenario cannot live-diff everything against it:
///  * windows starting past the merged file's begin_snapshot exclude the file
///    entirely (its CDC predicate predates partial-file inclusion) — those
///    windows are asserted against CURRENT official semantics (per
///    `test/sql/compaction/small_insert_compaction.test` upstream);
///  * deletes targeting a merged file are mis-attributed to origin snapshots
///    (its delete files lack the per-row snapshot column) — not exercised
///    here, tracked as a #179 follow-up.
///
/// Snapshots: 1 = CREATE TABLE, 2/3/4 = single-row inserts (rowids 0/1/2),
/// 5 = merge (no CDC events), 6 = insert of (4,'d') (rowid 3).
#[tokio::test]
async fn diff_compacted_inserts() -> DataFusionResult<()> {
    let tmp = TempDir::new().map_err(box_err)?;
    let path = tmp.path().join("compact.ducklake");
    write_catalog(
        &path,
        &[
            "CREATE TABLE c.t(id INTEGER, name VARCHAR);",
            "INSERT INTO c.t VALUES (1, 'a');",
            "INSERT INTO c.t VALUES (2, 'b');",
            "INSERT INTO c.t VALUES (3, 'c');",
            "CALL ducklake_merge_adjacent_files('c');",
            "INSERT INTO c.t VALUES (4, 'd');",
        ],
    )?;

    // Windows whose start does not exceed the merged file's begin_snapshot
    // (2): the installed extension includes the file and resolves per-row
    // origin snapshots correctly, so these live-diff against it.
    let live_windows = [(0, 6), (1, 6), (2, 6), (2, 2), (0, 2), (2, 4)];
    let mut official: Vec<((i64, i64), CanonFeed)> = Vec::new();
    {
        let conn = official_connection(&path)?;
        for (a, b) in live_windows {
            let changes = official_feed(
                &conn,
                &format!("SELECT * FROM ducklake_table_changes('c', 'main', 't', {a}, {b})"),
                true,
            )?;
            official.push(((a, b), changes));
        }
    }
    let ctx = crate_context(&path).await?;
    for ((a, b), official_changes) in official {
        let crate_changes = crate_feed(
            &ctx,
            &format!("SELECT * FROM ducklake_table_changes('main.t', {a}, {b})"),
            true,
        )
        .await?;
        assert_feeds_match(
            &format!("compacted table_changes window [{a},{b}]"),
            &official_changes,
            &crate_changes,
        );
    }

    // Windows reachable only through partial_max: asserted against CURRENT
    // official semantics (each row at its origin snapshot; the merge emits
    // nothing).
    let insert_row = |snapshot: i64, rowid: i64, id: &str, name: &str| CanonRow {
        snapshot_id: snapshot,
        rowid: Some(rowid),
        change_type: Some("insert".to_string()),
        cells: vec![id.to_string(), name.to_string()],
    };
    let expectations: [((i64, i64), Vec<CanonRow>); 5] = [
        ((3, 3), vec![insert_row(3, 1, "2", "b")]),
        ((4, 4), vec![insert_row(4, 2, "3", "c")]),
        ((5, 5), vec![]),
        (
            (3, 6),
            vec![
                insert_row(3, 1, "2", "b"),
                insert_row(4, 2, "3", "c"),
                insert_row(6, 3, "4", "d"),
            ],
        ),
        ((4, 5), vec![insert_row(4, 2, "3", "c")]),
    ];
    for ((a, b), expected) in expectations {
        let got = crate_feed(
            &ctx,
            &format!("SELECT * FROM ducklake_table_changes('main.t', {a}, {b})"),
            true,
        )
        .await?;
        assert_eq!(
            got.rows,
            expected,
            "partial-only window [{a},{b}] diverges from current official semantics\n{}",
            pretty(&got)
        );
    }
    Ok(())
}

/// A DELETE targeting a compaction-merged file must be reported at its COMMIT
/// snapshot with the row's preserved rowid and old values — current official
/// DuckLake semantics (its delete files carry per-row delete snapshots).
///
/// This cannot live-diff: the installed (1.4-era) extension writes delete
/// files without the per-row snapshot column and resolves the deleted row's
/// snapshot through the DATA file's embedded origin column instead, so it
/// mis-reports the delete as an update pair at the row's ORIGIN snapshot —
/// since fixed upstream. Asserted against current official semantics.
#[tokio::test]
async fn diff_delete_targeting_merged_file() -> DataFusionResult<()> {
    let tmp = TempDir::new().map_err(box_err)?;
    let path = tmp.path().join("merged_del.ducklake");
    write_catalog(
        &path,
        &[
            "CREATE TABLE c.t(id INTEGER, name VARCHAR);",
            "INSERT INTO c.t VALUES (1, 'a');",
            "INSERT INTO c.t VALUES (2, 'b');",
            "INSERT INTO c.t VALUES (3, 'c');",
            "CALL ducklake_merge_adjacent_files('c');",
            "INSERT INTO c.t VALUES (4, 'd');",
            "DELETE FROM c.t WHERE id = 2;",
        ],
    )?;
    let ctx = crate_context(&path).await?;
    // Snapshots: 2/3/4 = inserts (rowids 0/1/2), 5 = merge, 6 = insert, 7 = delete.
    let deleted = CanonRow {
        snapshot_id: 7,
        rowid: Some(1),
        change_type: Some("delete".to_string()),
        cells: vec!["2".to_string(), "b".to_string()],
    };
    for (a, b) in [(7, 7), (0, 1000), (6, 1000)] {
        let got = crate_feed(
            &ctx,
            &format!(
                "SELECT * FROM ducklake_table_changes('main.t', {a}, {b}) \
                 WHERE change_type = 'delete'"
            ),
            true,
        )
        .await?;
        assert_eq!(
            got.rows,
            vec![deleted.clone()],
            "window [{a},{b}]: the delete must surface at its commit snapshot\n{}",
            pretty(&got)
        );
    }
    // And the deletions feed agrees.
    let got = crate_feed(
        &ctx,
        "SELECT * FROM ducklake_table_deletions('main.t', 7, 7)",
        true,
    )
    .await?;
    assert_eq!(got.rows, vec![deleted], "deletions feed\n{}", pretty(&got));
    Ok(())
}

/// Timestamp bounds: a full-range timestamp window must equal the full-range
/// integer window, and live-diff against official's TIMESTAMPTZ overloads.
/// (The crate accepts timestamp STRINGS; snapshot times are UTC.)
#[tokio::test]
async fn diff_timestamp_bounds() -> DataFusionResult<()> {
    let tmp = TempDir::new().map_err(box_err)?;
    let path = tmp.path().join("ts.ducklake");
    write_catalog(
        &path,
        &[
            "CREATE TABLE c.t(id INTEGER, val VARCHAR);",
            "INSERT INTO c.t VALUES (1, 'one'), (2, 'two');",
            "UPDATE c.t SET val = 'TWO' WHERE id = 2;",
            "DELETE FROM c.t WHERE id = 1;",
        ],
    )?;
    let official = {
        let conn = official_connection(&path)?;
        official_feed(
            &conn,
            "SELECT * FROM ducklake_table_changes('c', 'main', 't', \
             TIMESTAMP '1970-01-01 00:00:00+00', TIMESTAMP '2100-01-01 00:00:00+00')",
            true,
        )?
    };
    let ctx = crate_context(&path).await?;
    let by_timestamp = crate_feed(
        &ctx,
        "SELECT * FROM ducklake_table_changes('main.t', '1970-01-01', '2100-01-01')",
        true,
    )
    .await?;
    let by_id = crate_feed(
        &ctx,
        "SELECT * FROM ducklake_table_changes('main.t', 0, 1000)",
        true,
    )
    .await?;
    assert_eq!(
        by_timestamp.rows, by_id.rows,
        "timestamp window must equal the integer window"
    );
    assert_feeds_match("timestamp-bounded table_changes", &official, &by_timestamp);
    Ok(())
}

// ---------------------------------------------------------------------------
// Schema evolution between snapshots
// ---------------------------------------------------------------------------
//
// A data file records each column under the physical name it had when the file
// was written, tagged with the column's field id. Official DuckLake resolves a
// change feed's columns BY FIELD ID against the schema as of the window's END
// snapshot, so a column renamed after a file was written still reads that
// file's values, and a column dropped and re-added under the same name reads as
// NULL for files that predate the re-add. Resolving by name instead returns
// NULL, another column's values, or — in a rename cycle — silently swapped
// ones.
//
// These are all COLUMN-level: every scenario keeps the table itself alive
// throughout. Resolving the table and schema at the right snapshot is a separate
// concern, and one these tests do not reach.

/// Rendered cells of one query result, sorted row-wise.
type ProjectedRows = Vec<Vec<String>>;

/// The rows of an arbitrary projection, rendered to the canonical strings and
/// sorted. Used where [`CanonRow`] cannot represent the projection: a feed
/// queried WITHOUT `rowid` — the projection that selects the insert-only plan
/// shape — and nested values extracted with engine-specific SQL.
fn official_projection(conn: &duckdb::Connection, sql: &str) -> DataFusionResult<ProjectedRows> {
    let mut stmt = conn.prepare(sql).map_err(box_err)?;
    let mut rows: ProjectedRows = stmt
        .query_map([], |row| {
            let mut out = Vec::new();
            let mut i = 0;
            while let Ok(v) = row.get::<usize, Value>(i) {
                out.push(duckdb_cell(&v));
                i += 1;
            }
            Ok(out)
        })
        .map_err(box_err)?
        .collect::<Result<_, _>>()
        .map_err(box_err)?;
    rows.sort();
    Ok(rows)
}

/// The crate-side counterpart of [`official_projection`].
async fn crate_projection(ctx: &SessionContext, sql: &str) -> DataFusionResult<ProjectedRows> {
    let batches = ctx.sql(sql).await?.collect().await?;
    let mut rows: ProjectedRows = Vec::new();
    for batch in &batches {
        for r in 0..batch.num_rows() {
            rows.push(
                (0..batch.num_columns())
                    .map(|c| arrow_cell(batch, c, r))
                    .collect(),
            );
        }
    }
    rows.sort();
    Ok(rows)
}

/// Diff a projection over every snapshot window of the catalog built by
/// `statements` whose END snapshot is at least `min_end`. `official_sql` and
/// `crate_sql` are templates with `{a}` / `{b}` window placeholders; they differ
/// only where the two dialects do (the table function's argument form).
///
/// Both engines resolve a feed's columns as of the window's end snapshot, so a
/// projection naming a column that some DDL statement introduced is unbindable
/// in windows that end before it: `min_end` is the snapshot from which every
/// projected name exists. Windows below it are covered by the whole-row
/// comparison in [`assert_cdc_conformance`], which projects no names of its own.
async fn assert_projection_conformance(
    statements: &[&str],
    min_end: i64,
    official_sql: &str,
    crate_sql: &str,
) -> DataFusionResult<()> {
    let tmp = TempDir::new().map_err(box_err)?;
    let path = tmp.path().join("diff.ducklake");
    write_catalog(&path, statements)?;

    let fill = |template: &str, a: i64, b: i64| {
        template
            .replace("{a}", &a.to_string())
            .replace("{b}", &b.to_string())
    };

    let mut expected: Vec<((i64, i64), ProjectedRows)> = Vec::new();
    {
        let conn = official_connection(&path)?;
        for (a, b) in windows(&snapshot_ids(&conn)?) {
            if b < min_end {
                continue;
            }
            let rows = official_projection(&conn, &fill(official_sql, a, b))?;
            expected.push(((a, b), rows));
        }
    }
    assert!(
        !expected.is_empty(),
        "min_end {min_end} excluded every snapshot window"
    );

    let ctx = crate_context(&path).await?;
    for ((a, b), want) in expected {
        let sql = fill(crate_sql, a, b);
        let got = crate_projection(&ctx, &sql).await?;
        assert_eq!(got, want, "window [{a},{b}] diverges from official: {sql}");
    }
    Ok(())
}

/// One file written before a top-level rename, one after.
const RENAME: &[&str] = &[
    "CREATE TABLE c.t(id INTEGER, nm VARCHAR);",
    "INSERT INTO c.t VALUES (1, 'a'), (2, 'b');",
    "ALTER TABLE c.t RENAME COLUMN nm TO name;",
    "INSERT INTO c.t VALUES (3, 'c');",
];

/// A rename with deletes on both sides of it: the second DELETE's source data
/// file predates the rename, so the deleted rows' old values are only reachable
/// by field id.
const RENAME_WITH_DELETES: &[&str] = &[
    "CREATE TABLE c.t(id INTEGER, nm VARCHAR);",
    "INSERT INTO c.t VALUES (1, 'a'), (2, 'b');",
    "DELETE FROM c.t WHERE id = 1;",
    "ALTER TABLE c.t RENAME COLUMN nm TO name;",
    "INSERT INTO c.t VALUES (3, 'c');",
    "DELETE FROM c.t WHERE id = 2;",
];

/// `note` is dropped and re-added under the same name, so the old and the new
/// column share a name and differ only by field id.
const DROP_THEN_READD: &[&str] = &[
    "CREATE TABLE c.t(id INTEGER, note VARCHAR);",
    "INSERT INTO c.t VALUES (1, 'old-1'), (2, 'old-2');",
    "ALTER TABLE c.t DROP COLUMN note;",
    "ALTER TABLE c.t ADD COLUMN note VARCHAR;",
    "INSERT INTO c.t VALUES (3, 'new-3');",
];

/// A top-level rename: rows written before it must still surface their values.
#[tokio::test]
async fn diff_renamed_column() -> DataFusionResult<()> {
    assert_cdc_conformance("t", RENAME).await
}

/// The same rename with deletes around it, so `delete` rows and the deletions
/// feed read a pre-rename source file.
#[tokio::test]
async fn diff_renamed_column_with_deletes() -> DataFusionResult<()> {
    assert_cdc_conformance("t", RENAME_WITH_DELETES).await
}

/// Projecting `rowid` away selects a different plan for `ducklake_table_changes`
/// and `ducklake_table_insertions` (one scan per file, unioned, with the CDC
/// columns prepended) and skips the rowid resolution in
/// `ducklake_table_deletions`. Each builds its own read schema, so one plan
/// shape proves nothing about the other.
#[tokio::test]
async fn diff_renamed_column_without_rowid_projection() -> DataFusionResult<()> {
    // `name` exists from the rename (snapshot 3 of RENAME, 4 of
    // RENAME_WITH_DELETES) on.
    assert_projection_conformance(
        RENAME,
        3,
        "SELECT snapshot_id, change_type, name \
         FROM ducklake_table_changes('c', 'main', 't', {a}, {b})",
        "SELECT snapshot_id, change_type, name FROM ducklake_table_changes('main.t', {a}, {b})",
    )
    .await?;
    assert_projection_conformance(
        RENAME,
        3,
        "SELECT snapshot_id, name FROM ducklake_table_insertions('c', 'main', 't', {a}, {b})",
        "SELECT snapshot_id, name FROM ducklake_table_insertions('main.t', {a}, {b})",
    )
    .await?;
    assert_projection_conformance(
        RENAME_WITH_DELETES,
        4,
        "SELECT snapshot_id, name FROM ducklake_table_deletions('c', 'main', 't', {a}, {b})",
        "SELECT snapshot_id, name FROM ducklake_table_deletions('main.t', {a}, {b})",
    )
    .await
}

/// A dropped, then re-added, column name. Resolving by name reads the dropped
/// column's data out of the old files; the re-added column must read as NULL
/// there. Which of the two the window even sees depends on its END snapshot, so
/// this is asserted over every window rather than against one hard-coded shape.
#[tokio::test]
async fn diff_dropped_then_readded_column() -> DataFusionResult<()> {
    assert_cdc_conformance("t", DROP_THEN_READD).await
}

/// The same, on the insert-only plan shape.
#[tokio::test]
async fn diff_dropped_then_readded_column_without_rowid_projection() -> DataFusionResult<()> {
    assert_projection_conformance(
        DROP_THEN_READD,
        1,
        "SELECT snapshot_id, change_type, id \
         FROM ducklake_table_changes('c', 'main', 't', {a}, {b})",
        "SELECT snapshot_id, change_type, id FROM ducklake_table_changes('main.t', {a}, {b})",
    )
    .await?;
    // Between the DROP (snapshot 3) and the re-ADD (4) there is no `note`
    // column at all, so project it only from the re-add on.
    assert_projection_conformance(
        DROP_THEN_READD,
        4,
        "SELECT snapshot_id, change_type, id, note \
         FROM ducklake_table_changes('c', 'main', 't', {a}, {b})",
        "SELECT snapshot_id, change_type, id, note \
         FROM ducklake_table_changes('main.t', {a}, {b})",
    )
    .await
}

/// A rename CYCLE: `x` and `y` swap names. Resolving by name does not lose a
/// value here, it returns the OTHER column's — the same row shape, silently
/// wrong.
///
/// Only windows ending at or after the last rename (snapshot 5) can be diffed:
/// the extension writing this catalog leaves the intermediate generations
/// overlapping (`x` spans snapshots 1-5 while `tmp` spans 3-5 under the SAME
/// column id), so its own `ducklake_table_changes` fails with "Column with name
/// x already exists" for a window that ends mid-cycle.
#[tokio::test]
async fn diff_renamed_column_cycle() -> DataFusionResult<()> {
    assert_projection_conformance(
        &[
            "CREATE TABLE c.t(x VARCHAR, y VARCHAR);",
            "INSERT INTO c.t VALUES ('in-x', 'in-y');",
            "ALTER TABLE c.t RENAME COLUMN x TO tmp;",
            "ALTER TABLE c.t RENAME COLUMN y TO x;",
            "ALTER TABLE c.t RENAME COLUMN tmp TO y;",
            "INSERT INTO c.t VALUES ('post-x', 'post-y');",
        ],
        5,
        "SELECT snapshot_id, rowid, change_type, x, y \
         FROM ducklake_table_changes('c', 'main', 't', {a}, {b})",
        "SELECT snapshot_id, rowid, change_type, x, y \
         FROM ducklake_table_changes('main.t', {a}, {b})",
    )
    .await
}

/// A column added between the queried snapshots and renamed after that: files
/// written before the ADD carry no such field id at all, files between ADD and
/// RENAME carry it under the old name.
#[tokio::test]
async fn diff_column_added_then_renamed() -> DataFusionResult<()> {
    assert_cdc_conformance(
        "t",
        &[
            "CREATE TABLE c.t(id INTEGER);",
            "INSERT INTO c.t VALUES (1);",
            "ALTER TABLE c.t ADD COLUMN c INTEGER;",
            "INSERT INTO c.t VALUES (2, 20);",
            "ALTER TABLE c.t RENAME COLUMN c TO cc;",
            "INSERT INTO c.t VALUES (3, 30);",
        ],
    )
    .await
}

/// An UPDATE before the rename. Its rewritten file embeds the row ids, which
/// takes a different scan branch from a plain insert — the `update_postimage`
/// rows come from that branch.
#[tokio::test]
async fn diff_renamed_column_after_update() -> DataFusionResult<()> {
    assert_cdc_conformance(
        "t",
        &[
            "CREATE TABLE c.t(id INTEGER, nm VARCHAR);",
            "INSERT INTO c.t VALUES (1, 'a'), (2, 'b');",
            "UPDATE c.t SET nm = 'B' WHERE id = 2;",
            "ALTER TABLE c.t RENAME COLUMN nm TO name;",
            "INSERT INTO c.t VALUES (3, 'c');",
        ],
    )
    .await
}

/// A rename of a field INSIDE a struct. Nested nodes carry their own field ids,
/// so a fix that resolves only top-level columns still loses this one.
#[tokio::test]
async fn diff_renamed_nested_field() -> DataFusionResult<()> {
    let statements = &[
        "CREATE TABLE c.t(id INTEGER, s STRUCT(a INTEGER, b INTEGER));",
        "INSERT INTO c.t VALUES (1, {'a': 10, 'b': 20});",
        "ALTER TABLE c.t RENAME COLUMN s.b TO bb;",
        "INSERT INTO c.t VALUES (2, {'a': 30, 'bb': 40});",
    ];
    // `s.bb` exists from the rename (snapshot 3) on.
    assert_projection_conformance(
        statements,
        3,
        "SELECT snapshot_id, change_type, id, s['a'], s['bb'] \
         FROM ducklake_table_changes('c', 'main', 't', {a}, {b})",
        "SELECT snapshot_id, change_type, id, s['a'], s['bb'] \
         FROM ducklake_table_changes('main.t', {a}, {b})",
    )
    .await?;
    assert_projection_conformance(
        statements,
        3,
        "SELECT snapshot_id, rowid, id, s['a'], s['bb'] \
         FROM ducklake_table_insertions('c', 'main', 't', {a}, {b})",
        "SELECT snapshot_id, rowid, id, s['a'], s['bb'] \
         FROM ducklake_table_insertions('main.t', {a}, {b})",
    )
    .await
}

/// A rename over a COMPACTION-MERGED file: its rows span several snapshots
/// (attributed per row by an embedded snapshot column) and it predates the
/// rename.
///
/// Only windows whose start does not exceed the merged file's `begin_snapshot`
/// live-diff against the installed extension — the same 1.4-era gap
/// `diff_compacted_inserts` documents (windows starting past it exclude the
/// merged file entirely).
#[tokio::test]
async fn diff_renamed_column_over_compacted_file() -> DataFusionResult<()> {
    let tmp = TempDir::new().map_err(box_err)?;
    let path = tmp.path().join("compact_rename.ducklake");
    write_catalog(
        &path,
        &[
            "CREATE TABLE c.t(id INTEGER, nm VARCHAR);",
            "INSERT INTO c.t VALUES (1, 'a');",
            "INSERT INTO c.t VALUES (2, 'b');",
            "CALL ducklake_merge_adjacent_files('c');",
            "ALTER TABLE c.t RENAME COLUMN nm TO name;",
            "INSERT INTO c.t VALUES (3, 'c');",
        ],
    )?;
    // Snapshots: 1 = CREATE, 2/3 = inserts, 4 = merge, 5 = rename, 6 = insert.
    let live_windows = [(0, 6), (1, 6), (2, 6), (2, 3)];
    let mut official: Vec<((i64, i64), CanonFeed)> = Vec::new();
    {
        let conn = official_connection(&path)?;
        for (a, b) in live_windows {
            official.push((
                (a, b),
                official_feed(
                    &conn,
                    &format!("SELECT * FROM ducklake_table_changes('c', 'main', 't', {a}, {b})"),
                    true,
                )?,
            ));
        }
    }
    let ctx = crate_context(&path).await?;
    for ((a, b), official_changes) in official {
        let crate_changes = crate_feed(
            &ctx,
            &format!("SELECT * FROM ducklake_table_changes('main.t', {a}, {b})"),
            true,
        )
        .await?;
        assert_feeds_match(
            &format!("compacted + renamed table_changes window [{a},{b}]"),
            &official_changes,
            &crate_changes,
        );
    }
    Ok(())
}

/// GUARD (passes before the field-id fix). Every field a CDC feed advertises —
/// at every nesting depth — must carry EMPTY metadata.
///
/// A read schema describes a file's nested nodes with the `PARQUET:field_id`
/// the file tags them with. That metadata is part of the parent's Arrow type, so
/// leaking it into a feed's output makes the feed's own batches disagree with
/// the schema it advertises ("expected List(Float32) but found
/// List(Float32, field: 'element', metadata: {\"PARQUET:field_id\": \"3\"})").
#[tokio::test]
async fn cdc_output_fields_carry_no_parquet_metadata() -> DataFusionResult<()> {
    use arrow::datatypes::{DataType, Field};

    fn assert_bare(field: &Field, path: &str, feed: &str) {
        assert!(
            field.metadata().is_empty(),
            "{feed}: field {path} carries metadata {:?}",
            field.metadata()
        );
        match field.data_type() {
            DataType::List(child)
            | DataType::LargeList(child)
            | DataType::FixedSizeList(child, _)
            | DataType::Map(child, _) => {
                assert_bare(child, &format!("{path}.{}", child.name()), feed)
            },
            DataType::Struct(children) => {
                for child in children {
                    assert_bare(child, &format!("{path}.{}", child.name()), feed);
                }
            },
            _ => {},
        }
    }

    let tmp = TempDir::new().map_err(box_err)?;
    let path = tmp.path().join("bare.ducklake");
    write_catalog(
        &path,
        &[
            "CREATE TABLE c.t(id INTEGER, v FLOAT[], s STRUCT(a INTEGER), m MAP(VARCHAR, INTEGER));",
            "INSERT INTO c.t VALUES (1, [1.5, 2.5], {'a': 10}, MAP(['k'], [7]));",
            "ALTER TABLE c.t RENAME COLUMN v TO vals;",
            "INSERT INTO c.t VALUES (2, [3.5], {'a': 20}, MAP(['j'], [8]));",
            "DELETE FROM c.t WHERE id = 1;",
        ],
    )?;
    let ctx = crate_context(&path).await?;
    for feed in ["ducklake_table_changes", "ducklake_table_insertions", "ducklake_table_deletions"]
    {
        let batches = ctx
            .sql(&format!("SELECT * FROM {feed}('main.t', 0, 1000)"))
            .await?
            .collect()
            .await?;
        assert!(!batches.is_empty(), "{feed} produced no batches");
        for batch in &batches {
            for field in batch.schema().fields() {
                assert_bare(field, field.name(), feed);
            }
            // The batch's own arrays must agree with that schema, which is what
            // the metadata check is really about.
            RecordBatch::try_new(batch.schema(), batch.columns().to_vec())
                .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?;
        }
    }
    Ok(())
}

/// GUARD (passes before the field-id fix). List / struct / map values must
/// round-trip through all three feeds unchanged: the read schema now describes
/// nested nodes, and a nested value re-wrapped under the wrong type is exactly
/// how the reported arrow error is produced. No DDL here — the point is that the
/// ordinary case keeps working.
#[tokio::test]
async fn diff_nested_values_round_trip() -> DataFusionResult<()> {
    let statements = &[
        "CREATE TABLE c.t(id INTEGER, v FLOAT[], s STRUCT(a INTEGER), m MAP(VARCHAR, INTEGER));",
        "INSERT INTO c.t VALUES (1, [1.5, 2.5], {'a': 10}, MAP(['k'], [7]));",
        "INSERT INTO c.t VALUES (2, [3.5], {'a': 20}, MAP(['k'], [8]));",
        "UPDATE c.t SET id = 12 WHERE id = 2;",
        "DELETE FROM c.t WHERE id = 1;",
    ];
    // `m['k']` is not usable on either side of the diff: a MAP column's key is
    // served as `Utf8` while a string literal is `Utf8View`, and the lookup will
    // not compare the two (an ordinary `SELECT` hits this too, so it is not a CDC
    // question). `map_extract` reaches the value on both engines.
    let projection = "snapshot_id, rowid, id, v[1], s['a'], map_extract(m, 'k')[1]";
    assert_projection_conformance(
        statements,
        1,
        &format!(
            "SELECT {projection}, change_type \
             FROM ducklake_table_changes('c', 'main', 't', {{a}}, {{b}})"
        ),
        &format!(
            "SELECT {projection}, change_type \
             FROM ducklake_table_changes('main.t', {{a}}, {{b}})"
        ),
    )
    .await?;
    for feed in ["ducklake_table_insertions", "ducklake_table_deletions"] {
        assert_projection_conformance(
            statements,
            1,
            &format!("SELECT {projection} FROM {feed}('c', 'main', 't', {{a}}, {{b}})"),
            &format!("SELECT {projection} FROM {feed}('main.t', {{a}}, {{b}})"),
        )
        .await?;
    }
    Ok(())
}

/// GUARD (passes before the field-id fix). A struct child added by DDL is
/// recorded NON-nullable in the catalog while the parquet node stays optional;
/// the feed must read it as NULL for files that predate it rather than failing
/// the nullability check.
#[tokio::test]
async fn diff_struct_child_added_by_ddl() -> DataFusionResult<()> {
    assert_projection_conformance(
        &[
            "CREATE TABLE c.t(id INTEGER, s STRUCT(a INTEGER));",
            "INSERT INTO c.t VALUES (1, {'a': 10});",
            "ALTER TABLE c.t ADD COLUMN s.b INTEGER;",
            "INSERT INTO c.t VALUES (2, {'a': 30, 'b': 40});",
        ],
        // `s.b` exists from the ADD (snapshot 3) on.
        3,
        "SELECT snapshot_id, change_type, id, s['a'], s['b'] \
         FROM ducklake_table_changes('c', 'main', 't', {a}, {b})",
        "SELECT snapshot_id, change_type, id, s['a'], s['b'] \
         FROM ducklake_table_changes('main.t', {a}, {b})",
    )
    .await
}

// ---------------------------------------------------------------------------
// The window's end snapshot resolves the TABLE, not only its columns
// ---------------------------------------------------------------------------

/// A change feed reports against the schema as of the window's END snapshot, and
/// that includes the table itself: a window over a table that was dropped LATER
/// must still be served, in a catalog that has since moved on. Past the drop the
/// table is gone and the query is refused — official DuckLake says "does not
/// exist at version N".
#[tokio::test]
async fn diff_window_over_since_dropped_table() -> DataFusionResult<()> {
    let tmp = TempDir::new().map_err(box_err)?;
    let path = tmp.path().join("dropped.ducklake");
    write_catalog(
        &path,
        &[
            // 1 = CREATE t, 2/3 = inserts, 4 = DROP t, 5 = CREATE other,
            // 6 = insert into other.
            "CREATE TABLE c.t(id INTEGER);",
            "INSERT INTO c.t VALUES (1);",
            "INSERT INTO c.t VALUES (2);",
            "DROP TABLE c.t;",
            "CREATE TABLE c.other(x INTEGER);",
            "INSERT INTO c.other VALUES (9);",
        ],
    )?;

    let live_windows = [(0, 3), (1, 3), (2, 2), (3, 3), (1, 1)];
    let mut official: Vec<((i64, i64), CanonFeed)> = Vec::new();
    {
        let conn = official_connection(&path)?;
        for (a, b) in live_windows {
            official.push((
                (a, b),
                official_feed(
                    &conn,
                    &format!("SELECT * FROM ducklake_table_changes('c', 'main', 't', {a}, {b})"),
                    true,
                )?,
            ));
        }
        // Past the drop the table does not exist at the window's end snapshot.
        for (a, b) in [(4, 4), (1, 6)] {
            assert!(
                official_feed(
                    &conn,
                    &format!("SELECT * FROM ducklake_table_changes('c', 'main', 't', {a}, {b})"),
                    true,
                )
                .is_err(),
                "official must refuse window [{a},{b}] over a dropped table"
            );
        }
    }

    let ctx = crate_context(&path).await?;
    for ((a, b), official_changes) in official {
        let crate_changes = crate_feed(
            &ctx,
            &format!("SELECT * FROM ducklake_table_changes('main.t', {a}, {b})"),
            true,
        )
        .await?;
        assert_feeds_match(
            &format!("since-dropped table_changes window [{a},{b}]"),
            &official_changes,
            &crate_changes,
        );
    }
    for (a, b) in [(4, 4), (1, 6)] {
        let error = match ctx
            .sql(&format!(
                "SELECT * FROM ducklake_table_changes('main.t', {a}, {b})"
            ))
            .await
        {
            Ok(df) => df.collect().await.err(),
            Err(e) => Some(e),
        };
        let message = error
            .map(|e| e.to_string())
            .unwrap_or_else(|| String::from("<no error>"));
        assert!(
            message.contains("does not exist at snapshot"),
            "window [{a},{b}] over a dropped table: unexpected outcome: {message}"
        );
    }
    Ok(())
}

/// GUARD (passes before the field-id fix). A column widened by `ALTER … TYPE`
/// after a file was written. The per-file read schema declares the column's
/// CURRENT type, so the narrower values the older file physically holds have to be
/// widened on the way out — the one evolution case that already worked by name,
/// and which must keep working now that the schema is built per file.
#[tokio::test]
async fn diff_promoted_column_type() -> DataFusionResult<()> {
    assert_cdc_conformance(
        "t",
        &[
            "CREATE TABLE c.t(id INTEGER, n INTEGER);",
            "INSERT INTO c.t VALUES (1, 10);",
            "ALTER TABLE c.t ALTER COLUMN n TYPE BIGINT;",
            "INSERT INTO c.t VALUES (2, 20);",
            "DELETE FROM c.t WHERE id = 1;",
        ],
    )
    .await
}
