#![cfg(feature = "metadata-duckdb")]
//! Rowid correctness for the CDC feeds (`ducklake_table_changes` /
//! `ducklake_table_deletions`).
//!
//! A row's rowid must be stable and consistent across its lifetime:
//!   * a plain insert's rowid is `row_id_start + physical_position`;
//!   * a second insert continues from the first file's `row_id_start`;
//!   * a delete reports the same rowid the row was inserted with;
//!   * an UPDATE's preimage and postimage share the (preserved) rowid.
//!
//! These are the guarantees an incremental, rowid-keyed index relies on.

mod common;

use std::sync::Arc;

use arrow::array::{Array, Int32Array, Int64Array, StringArray};
use arrow::record_batch::RecordBatch;
use datafusion::error::Result as DataFusionResult;
use datafusion::prelude::*;
use datafusion_ducklake::{
    DuckLakeCatalog, DuckdbMetadataProvider, MetadataProvider, register_ducklake_functions,
};
use tempfile::TempDir;

/// Open an in-memory DuckDB connection attached to `path` (as catalog `c`) and
/// run `statements` in order, writing a DuckLake catalog + parquet data.
fn write_catalog(path: &std::path::Path, statements: &[&str]) -> DataFusionResult<()> {
    let conn = duckdb::Connection::open_in_memory().map_err(box_err)?;
    conn.execute("INSTALL ducklake;", []).map_err(box_err)?;
    conn.execute("INSTALL parquet;", []).map_err(box_err)?;
    conn.execute("LOAD ducklake;", []).map_err(box_err)?;
    conn.execute(&format!("ATTACH 'ducklake:{}' AS c;", path.display()), [])
        .map_err(box_err)?;
    for s in statements {
        conn.execute(s, []).map_err(box_err)?;
    }
    Ok(())
}

fn box_err<E: std::error::Error + Send + Sync + 'static>(
    e: E,
) -> datafusion::error::DataFusionError {
    datafusion::error::DataFusionError::External(Box::new(e))
}

async fn ctx_for(path: &str) -> DataFusionResult<SessionContext> {
    let provider = DuckdbMetadataProvider::new(path)?;
    let provider_arc: Arc<dyn MetadataProvider> = Arc::new(DuckdbMetadataProvider::new(path)?);
    let catalog = DuckLakeCatalog::new(provider)?;
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));
    register_ducklake_functions(&ctx, provider_arc);
    Ok(ctx)
}

fn i32_col(b: &RecordBatch, i: usize) -> &Int32Array {
    b.column(i).as_any().downcast_ref::<Int32Array>().unwrap()
}
fn i64_col(b: &RecordBatch, i: usize) -> &Int64Array {
    b.column(i).as_any().downcast_ref::<Int64Array>().unwrap()
}
fn str_at(b: &RecordBatch, i: usize, r: usize) -> String {
    if let Some(a) = b.column(i).as_any().downcast_ref::<StringArray>() {
        return a.value(r).to_string();
    }
    if let Some(a) = b
        .column(i)
        .as_any()
        .downcast_ref::<arrow::array::StringViewArray>()
    {
        return a.value(r).to_string();
    }
    panic!(
        "column {i} is not a string, got {:?}",
        b.column(i).data_type()
    );
}

/// Collect `(id, rowid, change_type)` rows, locating each column by name (robust
/// to whether the feed honors projection).
async fn rows(ctx: &SessionContext, sql: &str) -> DataFusionResult<Vec<(i32, i64, String)>> {
    let batches = ctx.sql(sql).await?.collect().await?;
    let mut out = Vec::new();
    for b in &batches {
        let id_i = b.schema().index_of("id").unwrap();
        let rid_i = b.schema().index_of("rowid").unwrap();
        let ct_i = b.schema().index_of("change_type").unwrap();
        let id = i32_col(b, id_i);
        let rid = i64_col(b, rid_i);
        for r in 0..b.num_rows() {
            out.push((id.value(r), rid.value(r), str_at(b, ct_i, r)));
        }
    }
    Ok(out)
}

/// A single INSERT of three rows → one data file with `row_id_start = 0`, so
/// rowids are exactly the physical positions 0, 1, 2.
#[tokio::test]
async fn plain_insert_rowid_is_physical_position() -> DataFusionResult<()> {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("ins.ducklake");
    write_catalog(
        &path,
        &[
            "CREATE TABLE c.t(id INTEGER, name VARCHAR);",
            "INSERT INTO c.t VALUES (10,'a'),(20,'b'),(30,'c');",
        ],
    )?;
    let ctx = ctx_for(path.to_str().unwrap()).await?;
    let mut r = rows(
        &ctx,
        "SELECT id, rowid, change_type FROM ducklake_table_changes('main.t', 0, 1000) ORDER BY rowid",
    )
    .await?;
    r.sort_by_key(|x| x.1);
    assert_eq!(
        r,
        vec![(10, 0, "insert".into()), (20, 1, "insert".into()), (30, 2, "insert".into()),]
    );
    Ok(())
}

/// A second INSERT starts a new file whose `row_id_start` continues from the
/// first, so rowids run 0..5 across the two files.
#[tokio::test]
async fn second_insert_continues_row_id_start() -> DataFusionResult<()> {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("ins2.ducklake");
    write_catalog(
        &path,
        &[
            "CREATE TABLE c.t(id INTEGER, name VARCHAR);",
            "INSERT INTO c.t VALUES (10,'a'),(20,'b'),(30,'c');",
            "INSERT INTO c.t VALUES (40,'d'),(50,'e');",
        ],
    )?;
    let ctx = ctx_for(path.to_str().unwrap()).await?;
    let mut r = rows(
        &ctx,
        "SELECT id, rowid, change_type FROM ducklake_table_changes('main.t', 0, 1000) ORDER BY rowid",
    )
    .await?;
    r.sort_by_key(|x| x.1);
    assert_eq!(
        r,
        vec![
            (10, 0, "insert".into()),
            (20, 1, "insert".into()),
            (30, 2, "insert".into()),
            (40, 3, "insert".into()),
            (50, 4, "insert".into()),
        ]
    );
    Ok(())
}

/// A DELETE reports the deleted row through `ducklake_table_deletions` with the
/// same rowid it was inserted with (`row_id_start + position`).
#[tokio::test]
async fn delete_reports_original_rowid() -> DataFusionResult<()> {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("del.ducklake");
    write_catalog(
        &path,
        &[
            "CREATE TABLE c.t(id INTEGER, name VARCHAR);",
            "INSERT INTO c.t VALUES (10,'a'),(20,'b'),(30,'c'),(40,'d'),(50,'e');",
            "DELETE FROM c.t WHERE id = 30;", // physical position 2 => rowid 2
        ],
    )?;
    let ctx = ctx_for(path.to_str().unwrap()).await?;

    // The projected query must return exactly the requested columns, in order.
    let batches = ctx
        .sql("SELECT id, rowid, change_type FROM ducklake_table_deletions('main.t', 0, 1000)")
        .await?
        .collect()
        .await?;
    let schema = batches[0].schema();
    let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    assert_eq!(
        names,
        vec!["id", "rowid", "change_type"],
        "projection honored"
    );

    let r = rows(
        &ctx,
        "SELECT id, rowid, change_type FROM ducklake_table_deletions('main.t', 0, 1000)",
    )
    .await?;
    assert_eq!(r, vec![(30, 2, "delete".into())]);
    Ok(())
}

/// An UPDATE surfaces as a preimage (old values) + postimage (new values) that
/// share the row's preserved rowid.
#[tokio::test]
async fn update_preimage_and_postimage_share_rowid() -> DataFusionResult<()> {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("upd.ducklake");
    write_catalog(
        &path,
        &[
            "CREATE TABLE c.t(id INTEGER, name VARCHAR);",
            "INSERT INTO c.t VALUES (10,'a'),(20,'b'),(30,'c');", // rowids 0,1,2
            "UPDATE c.t SET name = 'B' WHERE id = 20;",           // rowid 1 preserved
        ],
    )?;
    let ctx = ctx_for(path.to_str().unwrap()).await?;
    let batches = ctx
        .sql("SELECT id, rowid, change_type, name FROM ducklake_table_changes('main.t', 0, 1000)")
        .await?
        .collect()
        .await?;

    let mut pre: Vec<(i64, String)> = Vec::new(); // (rowid, name)
    let mut post: Vec<(i64, String)> = Vec::new();
    for b in &batches {
        let rid = i64_col(b, 1);
        for r in 0..b.num_rows() {
            let ct = str_at(b, 2, r);
            let name = str_at(b, 3, r);
            match ct.as_str() {
                "update_preimage" => pre.push((rid.value(r), name)),
                "update_postimage" => post.push((rid.value(r), name)),
                _ => {},
            }
        }
    }
    assert_eq!(pre, vec![(1, "b".into())], "preimage: old value at rowid 1");
    assert_eq!(
        post,
        vec![(1, "B".into())],
        "postimage: new value at rowid 1"
    );
    Ok(())
}

/// A row deleted AFTER an UPDATE lives in an UPDATE-output file whose logical
/// rowid is the embedded ROW_ID, not `row_id_start + position`. table_deletions
/// must report that preserved rowid, so a delete keys the same as the row's
/// insert / update_postimage. Regression guard for the delete-after-update case.
#[tokio::test]
async fn delete_after_update_reports_embedded_rowid() -> DataFusionResult<()> {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("del_upd.ducklake");
    write_catalog(
        &path,
        &[
            "CREATE TABLE c.t(id INTEGER, name VARCHAR);",
            "INSERT INTO c.t VALUES (10,'a'),(20,'b'),(30,'c');", // rowids 0,1,2
            "UPDATE c.t SET name = 'B' WHERE id = 20;",           // rowid 1 preserved (embedded)
            "DELETE FROM c.t WHERE id = 20;",                     // delete the updated row
        ],
    )?;
    let ctx = ctx_for(path.to_str().unwrap()).await?;
    let r = rows(
        &ctx,
        "SELECT * FROM ducklake_table_deletions('main.t', 0, 1000)",
    )
    .await?;
    // Every delete event for id=20 carries its stable rowid 1: the row was
    // inserted at rowid 1, the UPDATE preserved it (embedded), and the DELETE of
    // the update-output row must report the embedded rowid — not that file's
    // row_id_start + position.
    assert!(!r.is_empty(), "expected delete rows for id=20");
    for (id, rowid, ct) in &r {
        assert_eq!(*id, 20, "only id=20 was deleted");
        assert_eq!(
            *rowid, 1,
            "delete must report the preserved rowid 1, not a rewrite-file position"
        );
        assert_eq!(ct, "delete");
    }
    Ok(())
}

/// Projecting `ducklake_table_deletions` WITHOUT rowid must not require (or
/// synthesize) one: `SELECT id, change_type` returns exactly those columns and
/// succeeds even for source files that could not produce a rowid.
#[tokio::test]
async fn deletions_without_rowid_projection_omits_rowid() -> DataFusionResult<()> {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("del_norowid.ducklake");
    write_catalog(
        &path,
        &[
            "CREATE TABLE c.t(id INTEGER, name VARCHAR);",
            "INSERT INTO c.t VALUES (10,'a'),(20,'b'),(30,'c');",
            "DELETE FROM c.t WHERE id = 20;",
        ],
    )?;
    let ctx = ctx_for(path.to_str().unwrap()).await?;
    let batches = ctx
        .sql("SELECT id, change_type FROM ducklake_table_deletions('main.t', 0, 1000)")
        .await?
        .collect()
        .await?;
    let schema = batches[0].schema();
    let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    assert_eq!(
        names,
        vec!["id", "change_type"],
        "rowid must not appear when it is not projected"
    );
    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total, 1, "one row (id=20) deleted");
    Ok(())
}

/// Projecting `ducklake_table_changes` WITHOUT rowid, over a range that contains
/// an UPDATE (which forces the correlated path), must still succeed and label the
/// update — plain inserts in the range must not require a rowid they don't emit.
#[tokio::test]
async fn changes_without_rowid_projection_still_correlates() -> DataFusionResult<()> {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("chg_norowid.ducklake");
    write_catalog(
        &path,
        &[
            "CREATE TABLE c.t(id INTEGER, name VARCHAR);",
            "INSERT INTO c.t VALUES (10,'a'),(20,'b'),(30,'c');",
            "UPDATE c.t SET name = 'B' WHERE id = 20;",
        ],
    )?;
    let ctx = ctx_for(path.to_str().unwrap()).await?;
    let batches = ctx
        .sql("SELECT id, change_type FROM ducklake_table_changes('main.t', 0, 1000)")
        .await?
        .collect()
        .await?;
    let schema = batches[0].schema();
    let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    assert_eq!(
        names,
        vec!["id", "change_type"],
        "rowid must not appear when it is not projected"
    );
    let mut change_types: Vec<String> = Vec::new();
    for b in &batches {
        for r in 0..b.num_rows() {
            change_types.push(str_at(b, 1, r));
        }
    }
    assert!(
        change_types.iter().any(|c| c == "update_postimage"),
        "UPDATE correlated to a postimage without projecting rowid"
    );
    assert!(
        change_types.iter().any(|c| c == "update_preimage"),
        "UPDATE correlated to a preimage without projecting rowid"
    );
    Ok(())
}

/// With rowid projected over a range that has plain inserts AND a pure delete
/// (no UPDATE / embedded rowid), the changes feed returns the inserts with
/// correct rowids and the pure delete as a `delete` row that preserves the
/// rowid the row was inserted with (matching official DuckLake).
#[tokio::test]
async fn changes_with_rowid_plain_inserts_and_pure_delete() -> DataFusionResult<()> {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("chg_puredel.ducklake");
    write_catalog(
        &path,
        &[
            "CREATE TABLE c.t(id INTEGER, name VARCHAR);",
            "INSERT INTO c.t VALUES (10,'a'),(20,'b'),(30,'c');",
            "DELETE FROM c.t WHERE id = 20;",
        ],
    )?;
    let ctx = ctx_for(path.to_str().unwrap()).await?;
    let mut r = rows(
        &ctx,
        "SELECT id, rowid, change_type FROM ducklake_table_changes('main.t', 0, 1000) \
         ORDER BY rowid, change_type",
    )
    .await?;
    r.sort();
    // The three inserted rows with rowids 0,1,2; the pure delete surfaces with
    // the same rowid (1) its row was inserted with.
    assert_eq!(
        r,
        vec![
            (10, 0, "insert".into()),
            (20, 1, "delete".into()),
            (20, 1, "insert".into()),
            (30, 2, "insert".into()),
        ]
    );
    Ok(())
}

/// A non-rowid projection over a delete-only window must succeed even when the
/// delete's source file has a NULL `row_id_start` (older/foreign catalogs):
/// with rowid neither output nor pairable (no postimage in range), no rowid is
/// synthesized — mirroring `ducklake_table_deletions`. Projecting rowid over
/// the same window still fails, since a real value would be required.
#[tokio::test]
async fn pure_delete_without_row_id_start_non_rowid_projection() -> DataFusionResult<()> {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("chg_norowidstart.ducklake");
    write_catalog(
        &path,
        &[
            "CREATE TABLE c.t(id INTEGER, name VARCHAR);",
            "INSERT INTO c.t VALUES (10,'a'),(20,'b');",
            "DELETE FROM c.t WHERE id = 20;",
        ],
    )?;
    // Simulate an older/foreign catalog that never recorded row_id_start.
    {
        let conn = duckdb::Connection::open(&path).map_err(box_err)?;
        conn.execute("UPDATE ducklake_data_file SET row_id_start = NULL;", [])
            .map_err(box_err)?;
    }
    let ctx = ctx_for(path.to_str().unwrap()).await?;

    // Without rowid: the delete surfaces, no synthesis needed.
    let batches = ctx
        .sql(
            "SELECT id, change_type FROM ducklake_table_changes('main.t', 0, 1000) \
             WHERE change_type = 'delete'",
        )
        .await?
        .collect()
        .await?;
    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total, 1, "the pure delete surfaces without rowid synthesis");

    // With rowid projected: synthesis is required and must fail loudly rather
    // than emitting wrong ids.
    let err = ctx
        .sql("SELECT rowid, change_type FROM ducklake_table_changes('main.t', 0, 1000)")
        .await?
        .collect()
        .await;
    assert!(
        err.is_err(),
        "rowid projection must error when it cannot be synthesized"
    );
    Ok(())
}
