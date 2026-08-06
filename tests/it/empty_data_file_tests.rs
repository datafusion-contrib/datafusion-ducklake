//! Reads over a table carrying a live data file with no rows.
//!
//! Such a file is ordinary: a write of only empty batches registers one, and the
//! reference implementation emits one for partitioned output, so a catalog written
//! elsewhere can contain them too. Pruning drops these files before it consults
//! statistics — having no rows, they cannot hold a matching one — which means the
//! scan planner never sees them either.
//!
//! These are no-regression guards for that exclusion, not reproductions of a bug:
//! every assertion here holds both before and after it. What they pin is that
//! removing a file from the candidate set cannot change what a read *returns* —
//! neither the rows of a table that also holds real files, nor the shape of a table
//! whose only live file is the empty one, where the exclusion empties the candidate
//! set outright and the read must still answer with zero rows rather than fail, on
//! the plain and the row-lineage planning paths alike.

#![cfg(all(feature = "write-sqlite", feature = "metadata-sqlite"))]

use std::sync::Arc;

use arrow::array::Int32Array;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::prelude::*;
use object_store::local::LocalFileSystem;
use tempfile::TempDir;

use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeTableWriter, MetadataWriter, SqliteMetadataProvider,
    SqliteMetadataWriter,
};

fn table_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("val", DataType::Int32, false),
    ]))
}

fn batch(ids: Vec<i32>, vals: Vec<i32>) -> RecordBatch {
    RecordBatch::try_new(
        table_schema(),
        vec![Arc::new(Int32Array::from(ids)), Arc::new(Int32Array::from(vals))],
    )
    .unwrap()
}

fn conn_str(temp_dir: &TempDir) -> String {
    format!(
        "sqlite:{}?mode=rwc",
        temp_dir.path().join("test.db").display()
    )
}

fn object_store() -> Arc<dyn object_store::ObjectStore> {
    Arc::new(LocalFileSystem::new())
}

/// Create the catalog and write `t`'s first data file from `batches`, which may be
/// entirely empty — an all-empty write still registers a data file, with a
/// `record_count` of 0.
async fn seed_table(temp_dir: &TempDir, batches: &[RecordBatch]) {
    let data_path = temp_dir.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();
    let writer = SqliteMetadataWriter::new_with_init(&conn_str(temp_dir))
        .await
        .unwrap();
    writer.set_data_path(data_path.to_str().unwrap()).unwrap();
    DuckLakeTableWriter::new(Arc::new(writer), object_store())
        .unwrap()
        .write_table("main", "t", batches)
        .await
        .unwrap();
}

async fn append(temp_dir: &TempDir, batches: &[RecordBatch]) {
    let writer = SqliteMetadataWriter::new(&conn_str(temp_dir))
        .await
        .unwrap();
    DuckLakeTableWriter::new(Arc::new(writer), object_store())
        .unwrap()
        .append_table("main", "t", batches)
        .await
        .unwrap();
}

async fn read_only_session(temp_dir: &TempDir) -> SessionContext {
    session(temp_dir, false).await
}

async fn session(temp_dir: &TempDir, row_lineage: bool) -> SessionContext {
    let provider = SqliteMetadataProvider::new(&conn_str(temp_dir))
        .await
        .unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog(
        "ducklake",
        Arc::new(
            DuckLakeCatalog::new(provider)
                .unwrap()
                .with_row_lineage(row_lineage),
        ),
    );
    ctx
}

/// Every `id` the query returns, ascending.
async fn query_ids(ctx: &SessionContext, sql: &str) -> Vec<i32> {
    let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
    let mut ids: Vec<i32> = batches
        .iter()
        .flat_map(|b| {
            let column = b
                .column_by_name("id")
                .expect("projection includes id")
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("id is Int32")
                .clone();
            (0..column.len()).map(move |i| column.value(i))
        })
        .collect();
    ids.sort_unstable();
    ids
}

/// A table holding real files *and* an empty one reads exactly as it would
/// without the empty file — unfiltered, filtered, and counted.
#[tokio::test(flavor = "multi_thread")]
async fn a_scan_over_a_table_with_an_empty_file_returns_every_row() {
    let temp_dir = TempDir::new().unwrap();
    seed_table(&temp_dir, &[batch(vec![1, 2, 3], vec![10, 20, 30])]).await;
    append(&temp_dir, &[batch(vec![101, 102, 103], vec![40, 50, 60])]).await;
    // A live data file with no rows, sitting alongside the two above.
    append(&temp_dir, &[RecordBatch::new_empty(table_schema())]).await;

    let ctx = read_only_session(&temp_dir).await;

    assert_eq!(
        query_ids(&ctx, "SELECT id FROM ducklake.main.t").await,
        vec![1, 2, 3, 101, 102, 103],
        "the empty file must not cost the table any rows",
    );
    // A predicate that only one file's statistics admit: the row still comes back.
    assert_eq!(
        query_ids(&ctx, "SELECT id FROM ducklake.main.t WHERE id = 102").await,
        vec![102],
    );
    // Answered from statistics rather than by reading files, so it is worth
    // asserting separately: the empty file contributes 0 and must not shift it.
    let counted = ctx
        .sql("SELECT count(*) FROM ducklake.main.t")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(
        counted[0]
            .column(0)
            .as_any()
            .downcast_ref::<arrow::array::Int64Array>()
            .expect("count(*) is Int64")
            .value(0),
        6,
    );
}

/// A table whose only live file is the empty one. Excluding it leaves nothing to
/// scan at all, so this is the case where the exclusion could plausibly turn a
/// legitimate zero-row read into a failure or a wrong-shaped result.
#[tokio::test(flavor = "multi_thread")]
async fn a_table_whose_only_live_file_is_empty_reads_as_zero_rows() {
    let temp_dir = TempDir::new().unwrap();
    seed_table(&temp_dir, &[RecordBatch::new_empty(table_schema())]).await;

    let ctx = read_only_session(&temp_dir).await;

    let batches = ctx
        .sql("SELECT id, val FROM ducklake.main.t")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(
        batches.iter().map(RecordBatch::num_rows).sum::<usize>(),
        0,
        "an empty table reads as zero rows",
    );
    // The schema must still be the table's, not an artefact of having no file to
    // derive one from.
    let schema = ctx
        .sql("SELECT id, val FROM ducklake.main.t")
        .await
        .unwrap()
        .schema()
        .as_arrow()
        .clone();
    assert_eq!(
        schema
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .collect::<Vec<_>>(),
        vec!["id", "val"],
    );
    assert_eq!(
        query_ids(&ctx, "SELECT id FROM ducklake.main.t WHERE id = 1").await,
        Vec::<i32>::new(),
    );
}

/// The same shape read with row lineage on. Projecting the synthetic `rowid`
/// takes a separate planning path — one scan per file, since each carries its own
/// starting row id — with its own empty-plan branch. Excluding empty files is what
/// makes that branch reachable for a table that does have a live file, so it is
/// worth reading through rather than assuming it behaves like the path above.
#[tokio::test(flavor = "multi_thread")]
async fn a_row_lineage_read_of_an_only_empty_table_reads_as_zero_rows() {
    let temp_dir = TempDir::new().unwrap();
    seed_table(&temp_dir, &[RecordBatch::new_empty(table_schema())]).await;

    let ctx = session(&temp_dir, true).await;

    let frame = ctx
        .sql("SELECT id, val, rowid FROM ducklake.main.t")
        .await
        .unwrap();
    let schema = frame.schema().as_arrow().clone();
    assert_eq!(
        schema
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .collect::<Vec<_>>(),
        vec!["id", "val", "rowid"],
        "the rowid projection must survive an empty candidate set",
    );
    let batches = frame.collect().await.unwrap();
    assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 0,);
}
