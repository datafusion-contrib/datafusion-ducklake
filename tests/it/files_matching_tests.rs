//! Integration tests for `DuckLakeTable::files_matching` against a real
//! (SQLite-backed) catalog, covering the two things the unit tests in
//! `src/table.rs` cannot: that real catalog statistics actually prune, and that
//! the safety contract a caller needs when it resolves positions on the returned
//! files is reachable from outside the crate.
//!
//! The second is the important one. `resolve_positions` derives a row's delete
//! position from its physical index in the file, which is only the position
//! DuckLake records for a file that has never been rewritten. `files_matching`
//! answers from catalog metadata alone, so it cannot know which files those are —
//! and it must not guess and silently withhold one, because a keyed mutation that
//! never sees a file holding its key inserts a duplicate instead of superseding.
//! Both failure directions are silent, so the file is returned and the caller is
//! given a public way to detect it and refuse.

#![cfg(all(feature = "write-sqlite", feature = "metadata-sqlite"))]

use std::sync::Arc;

use arrow::array::{Int32Array, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::catalog::TableProvider;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_expr::expressions::{BinaryExpr, col, lit};
use datafusion::prelude::*;
use object_store::local::LocalFileSystem;
use tempfile::TempDir;

use datafusion::logical_expr::Operator;
use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeTable, DuckLakeTableWriter, MetadataWriter, SqliteMetadataProvider,
    SqliteMetadataWriter,
};

/// The `(id, val)` table used throughout.
fn table_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("val", DataType::Int32, false),
    ]))
}

fn object_store() -> Arc<dyn object_store::ObjectStore> {
    Arc::new(LocalFileSystem::new())
}

fn batch(ids: Vec<i32>, vals: Vec<i32>) -> RecordBatch {
    RecordBatch::try_new(
        table_schema(),
        vec![Arc::new(Int32Array::from(ids)), Arc::new(Int32Array::from(vals))],
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

/// Create the catalog and write `t`'s first data file.
async fn seed_table(temp_dir: &TempDir, ids: Vec<i32>, vals: Vec<i32>) {
    let data_path = temp_dir.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();
    let writer = SqliteMetadataWriter::new_with_init(&conn_str(temp_dir, true))
        .await
        .unwrap();
    writer.set_data_path(data_path.to_str().unwrap()).unwrap();
    DuckLakeTableWriter::new(Arc::new(writer), object_store())
        .unwrap()
        .write_table("main", "t", &[batch(ids, vals)])
        .await
        .unwrap();
}

/// Append a second data file to `t`.
async fn append_file(temp_dir: &TempDir, ids: Vec<i32>, vals: Vec<i32>) {
    let writer = SqliteMetadataWriter::new(&conn_str(temp_dir, true))
        .await
        .unwrap();
    DuckLakeTableWriter::new(Arc::new(writer), object_store())
        .unwrap()
        .append_table("main", "t", &[batch(ids, vals)])
        .await
        .unwrap();
}

/// Run a DML statement and return the row count it reports.
async fn run_dml(temp_dir: &TempDir, sql: &str) -> u64 {
    let writer = SqliteMetadataWriter::new(&conn_str(temp_dir, true))
        .await
        .unwrap();
    let provider = SqliteMetadataProvider::new(&conn_str(temp_dir, true))
        .await
        .unwrap();
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

/// A read-only session plus the `DuckLakeTable` behind `ducklake.main.t`.
async fn open_table(temp_dir: &TempDir) -> (SessionContext, Arc<dyn TableProvider>) {
    let provider = SqliteMetadataProvider::new(&conn_str(temp_dir, false))
        .await
        .unwrap();
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

/// `id = wanted`, as the physical expression a caller passes to both
/// `files_matching` and `resolve_positions`.
fn id_equals(wanted: i32) -> Arc<dyn PhysicalExpr> {
    let schema = table_schema();
    Arc::new(BinaryExpr::new(
        col("id", schema.as_ref()).unwrap(),
        Operator::Eq,
        lit(wanted),
    ))
}

// ---------------------------------------------------------------------------

/// Two insert-only files with disjoint `id` ranges: the catalog's own per-file
/// statistics must leave only the one that can hold the key, and that file must
/// be usable — `resolve_positions` finds the row in it.
#[tokio::test(flavor = "multi_thread")]
async fn files_matching_returns_only_the_file_whose_statistics_admit_the_key() {
    let temp_dir = TempDir::new().unwrap();
    seed_table(&temp_dir, vec![1, 2, 3], vec![10, 20, 30]).await;
    append_file(&temp_dir, vec![101, 102, 103], vec![40, 50, 60]).await;

    let (ctx, provider) = open_table(&temp_dir).await;
    let table = as_ducklake(&provider);
    let predicate = id_equals(102);

    let matching = table.files_matching(&predicate).unwrap();

    assert_eq!(
        matching.len(),
        1,
        "only the second file's statistics admit id = 102, got {:?}",
        matching.iter().map(|f| &f.file.path).collect::<Vec<_>>(),
    );
    let positions = table
        .resolve_positions(&ctx.state(), &matching[0].file, predicate)
        .await
        .unwrap();
    assert_eq!(
        positions.into_iter().collect::<Vec<_>>(),
        vec![1],
        "the retained file holds the key at physical position 1",
    );
}

/// After an UPDATE rewrites a file, `files_matching` must still return it — and
/// the caller must be able to tell, through the public API alone, that resolving
/// positions on it is unsafe.
///
/// The two assertions guard opposite failure directions. Dropping the rewritten
/// file would make a keyed mutation insert a duplicate key; failing to flag it
/// would let the caller resolve positions on it and delete the wrong rows. A
/// blanket "everything is unsafe" answer is ruled out by requiring the
/// insert-only file in the same result to report `false`.
#[tokio::test(flavor = "multi_thread")]
async fn a_rewritten_file_is_returned_and_reports_an_embedded_rowid() {
    let temp_dir = TempDir::new().unwrap();
    seed_table(&temp_dir, vec![1, 2, 3], vec![10, 20, 30]).await;
    // Rewrites the seeded file: the new file carries the surviving rows with
    // their original row ids embedded, and a positional delete supersedes the
    // old copies.
    assert_eq!(
        run_dml(
            &temp_dir,
            "UPDATE ducklake.main.t SET val = 99 WHERE id = 2"
        )
        .await,
        1,
    );

    let (ctx, provider) = open_table(&temp_dir).await;
    let table = as_ducklake(&provider);
    let matching = table.files_matching(&id_equals(2)).unwrap();

    assert_eq!(
        matching.len(),
        2,
        "both the original file and the rewritten one can hold id = 2, got {:?}",
        matching.iter().map(|f| &f.file.path).collect::<Vec<_>>(),
    );

    let mut rewritten = 0;
    let mut insert_only = 0;
    for file in &matching {
        if table
            .file_has_embedded_rowid(&ctx.state(), &file.file)
            .await
            .unwrap()
        {
            rewritten += 1;
        } else {
            insert_only += 1;
        }
    }
    assert_eq!(
        (rewritten, insert_only),
        (1, 1),
        "exactly the UPDATE's output must be flagged as rewritten",
    );
}
