//! Integration tests for `DuckLakeTable::files_matching` against a real
//! (SQLite-backed) catalog, covering the two things the unit tests in
//! `src/table.rs` cannot: that real catalog statistics actually prune, and that a
//! rewritten file is returned like any other and reports itself as rewritten.
//!
//! `files_matching` answers from catalog metadata alone, so it cannot know which
//! of the files it returns were rewritten by an UPDATE or by compaction — and it
//! must not guess and silently withhold one, because a keyed mutation that never
//! sees a file holding its key inserts a duplicate instead of superseding it.
//! Nor does it need to: `resolve_positions` reads a file's true physical row
//! positions, which a rewrite leaves meaningful, so a rewritten file needs no
//! special handling. `file_has_embedded_rowid` still distinguishes the two,
//! because it answers where a row's *rowid* comes from — see
//! `keyed_mutation_after_compaction_tests` for the mutations themselves.

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
/// the caller must be able to tell, through the public API alone, which of the
/// files it got are rewritten.
///
/// Dropping the rewritten file would make a keyed mutation insert a duplicate
/// key, so it has to come back. The flag is not a safety gate — positions resolve
/// correctly on a rewritten file — but it does report where a row's rowid comes
/// from, and a blanket "everything is rewritten" answer is ruled out by requiring
/// the insert-only file in the same result to report `false`.
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

// ---------------------------------------------------------------------------
// Identity-partitioned tables
// ---------------------------------------------------------------------------

/// Seed a partitioned `t(id, val)` whose partition key is `id` itself, so each
/// file genuinely holds one value for it and the partition values the writer
/// stores are true of their files.
async fn seed_partitioned_table(temp_dir: &TempDir, ids: &[i32]) {
    use datafusion_ducklake::partition::PartitionTransform;
    use datafusion_ducklake::{ColumnDef, DuckLakeWriteOptions, WriteMode};

    let data_path = temp_dir.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();
    let conn = conn_str(temp_dir, true);
    let writer = SqliteMetadataWriter::new_with_init(&conn).await.unwrap();
    writer.set_data_path(data_path.to_str().unwrap()).unwrap();

    let cols = vec![
        ColumnDef::from_arrow("id", &DataType::Int32, false).unwrap(),
        ColumnDef::from_arrow("val", &DataType::Int32, false).unwrap(),
    ];
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
            &s.column_ids,
        )
        .unwrap();
    writer
        .set_partition_spec(
            s.table_id,
            &[("id".to_string(), PartitionTransform::Identity)],
        )
        .unwrap();

    let values = ids
        .iter()
        .map(|id| format!("({id}, {})", id * 10))
        .collect::<Vec<_>>()
        .join(", ");
    let writer = SqliteMetadataWriter::new(&conn).await.unwrap();
    let provider = SqliteMetadataProvider::new(&conn).await.unwrap();
    let catalog = DuckLakeCatalog::with_writer(Arc::new(provider), Arc::new(writer))
        .unwrap()
        .with_write_options(DuckLakeWriteOptions::default().with_data_inlining_row_limit(0));
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));
    ctx.sql(&format!(
        "INSERT INTO ducklake.main.t SELECT * FROM (VALUES {values}) AS v(id, val)"
    ))
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();
}

/// `files_matching` is the keyed-mutation entry point: a file it fails to return
/// makes the mutation insert a duplicate key instead of superseding the row. The
/// partition pre-filter narrows this listing too, so it has to be exactly as
/// conservative here as when planning a scan — and it must still be doing
/// something, which the second assertion pins.
#[tokio::test(flavor = "multi_thread")]
async fn files_matching_narrows_an_identity_partitioned_table() {
    let temp_dir = TempDir::new().unwrap();
    seed_partitioned_table(&temp_dir, &[1, 2, 102]).await;

    let (ctx, provider) = open_table(&temp_dir).await;
    let table = as_ducklake(&provider);
    let predicate = id_equals(102);
    let matching = table.files_matching(&predicate).unwrap();

    assert_eq!(
        matching.len(),
        1,
        "only the id=102 partition can hold the key, got {:?}",
        matching.iter().map(|f| &f.file.path).collect::<Vec<_>>(),
    );
    let positions = table
        .resolve_positions(&ctx.state(), &matching[0].file, predicate)
        .await
        .unwrap();
    assert_eq!(positions.into_iter().collect::<Vec<_>>(), vec![0]);
}

/// The same path, over a partition value spelled in a way the encoder never
/// writes but an integer parser still reads. Dropping this file would make a
/// keyed mutation insert a duplicate, which is the worst failure this mechanism
/// can have — so the pre-filter must decline to refute and hand the file back.
#[tokio::test(flavor = "multi_thread")]
async fn files_matching_keeps_a_non_canonically_spelled_partition_value() {
    for hostile in ["0102", "+102", "102.0", "0x102", " 102", "102\t"] {
        let temp_dir = TempDir::new().unwrap();
        seed_partitioned_table(&temp_dir, &[1, 2, 102]).await;

        let pool = sqlx::SqlitePool::connect(&conn_str(&temp_dir, true))
            .await
            .unwrap();
        let affected = sqlx::query(
            "UPDATE ducklake_file_partition_value SET partition_value = ?
             WHERE partition_key_index = 0 AND partition_value = '102'",
        )
        .bind(hostile)
        .execute(&pool)
        .await
        .unwrap()
        .rows_affected();
        assert_eq!(affected, 1);

        let (_ctx, provider) = open_table(&temp_dir).await;
        let table = as_ducklake(&provider);
        let matching = table.files_matching(&id_equals(102)).unwrap();
        assert_eq!(
            matching.len(),
            1,
            "partition value {hostile:?} must not withhold the file a keyed \
             mutation needs, got {:?}",
            matching.iter().map(|f| &f.file.path).collect::<Vec<_>>(),
        );
    }
}
