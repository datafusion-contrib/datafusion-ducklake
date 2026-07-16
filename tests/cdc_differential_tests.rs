#![cfg(feature = "metadata-duckdb")]
//! Differential CDC conformance tests (#179).
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
//! * NORMALIZER-BOUNDS — official snapshot bounds are inclusive on both ends;
//!   the crate's are exclusive-start / inclusive-end. Official `[a, b]` is
//!   queried on the crate as `(a-1, b]`.
//! * NORMALIZER-DELETE-ROUTING — official `table_changes` emits pure deletes
//!   as `change_type='delete'`; the crate routes them to
//!   `ducklake_table_deletions`. We therefore assert the full change-set in
//!   two halves: crate `table_changes` == official `table_changes` minus its
//!   'delete' rows, and crate `table_deletions` == official
//!   `ducklake_table_deletions` (all deleted rows, update preimages included).
//! * NORMALIZER-COLUMN-PLACEMENT — official leads with
//!   `(snapshot_id, rowid, change_type)`; the crate appends
//!   `(rowid, snapshot_id, change_type)` after the table columns. Rows are
//!   canonicalized by column NAME; the residual table-column name order is
//!   still asserted equal.
//!
//! Not yet covered (tracked in #179): TIMESTAMPTZ bounds, encrypted (PME)
//! catalogs, compaction rewrites, schema evolution between snapshots.

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
    conn.execute("INSTALL ducklake;", []).map_err(box_err)?;
    conn.execute("INSTALL parquet;", []).map_err(box_err)?;
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

/// One engine's canonicalized feed output: sorted rows + the residual
/// (non-CDC) column names in their original order.
#[derive(Debug, Clone, PartialEq, Eq)]
struct CanonFeed {
    table_columns: Vec<String>,
    rows: Vec<CanonRow>,
}

impl CanonFeed {
    fn new(table_columns: Vec<String>, mut rows: Vec<CanonRow>) -> Self {
        rows.sort();
        Self {
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
        // name assertion for empty feeds).
        table_columns.clear();
    }
    Ok(CanonFeed::new(table_columns, rows))
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
    }
    Ok(CanonFeed::new(table_columns, rows))
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
    let mut official: Vec<((i64, i64), CanonFeed, CanonFeed)> = Vec::new();
    {
        let conn = official_connection(&path)?;
        for (a, b) in windows(&snapshot_ids(&conn)?) {
            let changes = official_feed(
                &conn,
                &format!("SELECT * FROM ducklake_table_changes('c', 'main', '{table}', {a}, {b})"),
                true,
            )?;
            let deletions = official_feed(
                &conn,
                // rowid/snapshot_id are virtual on the official deletions scan:
                // project them explicitly; there is no change_type column.
                &format!(
                    "SELECT snapshot_id, rowid, * FROM \
                     ducklake_table_deletions('c', 'main', '{table}', {a}, {b})"
                ),
                false,
            )?;
            official.push(((a, b), changes, deletions));
        }
    }

    let ctx = crate_context(&path).await?;
    for ((a, b), official_changes, official_deletions) in official {
        // NORMALIZER-BOUNDS: official [a, b] == crate (a-1, b].
        let (ca, cb) = (a - 1, b);
        let crate_changes = crate_feed(
            &ctx,
            &format!("SELECT * FROM ducklake_table_changes('main.{table}', {ca}, {cb})"),
            true,
        )
        .await?;
        let crate_deletions = crate_feed(
            &ctx,
            &format!("SELECT * FROM ducklake_table_deletions('main.{table}', {ca}, {cb})"),
            true,
        )
        .await?;

        // NORMALIZER-DELETE-ROUTING (half 1): the crate's table_changes must
        // match official's minus its pure-delete rows.
        let official_nondelete = CanonFeed::new(
            official_changes.table_columns.clone(),
            official_changes
                .rows
                .iter()
                .filter(|r| r.change_type.as_deref() != Some("delete"))
                .cloned()
                .collect(),
        );
        assert_feeds_match(
            &format!("table_changes window [{a},{b}]"),
            &official_nondelete,
            &crate_changes,
        );

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
