//! Replace-on-survivors oracle + differential harness for positional deletes.
//!
//! The example tests in `positional_delete_tests.rs` hardcode the expected
//! survivors on a single 4-row, single-row-group file. This harness
//! instead *computes* the reference survivors independently of the position math
//! and drives many shapes through the real positional path
//! (`resolve_positions -> write_delete_file -> set_delete_file`), asserting the
//! surviving VALUES match — a positional bug silently deletes the wrong rows.
//!
//! Why this catches what the example tests can't: the oracle never computes a
//! physical position. Survivors are derived by *predicate* (which rows are not
//! deleted), so if the SUT's `row_group_start (prefix-sum) + within-group
//! offset` math is wrong, the two disagree. The generators deliberately exercise
//! the regions the single-file example tests never touch: **multi-row-group**
//! files, **multiple data files** (append), **schema-evolved** files, **update**
//! (delete + re-insert), and **multi-generation** (cumulative) deletes.
//!
//! Deterministic on purpose: fixed shapes + a bounded exhaustive sweep, no
//! randomness — reproducible and parallel-safe on CI (each case gets its own
//! temp dir + SQLite catalog). Split into several `#[test]` fns so `cargo test`
//! parallelizes across them.

#![cfg(all(feature = "write-sqlite", feature = "metadata-sqlite"))]

use std::sync::Arc;

use arrow::array::{Array, Int32Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::logical_expr::Operator;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_expr::expressions::{BinaryExpr, col, lit};
use datafusion::prelude::*;
use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeFileData, DuckLakeTable, DuckLakeTableWriter, MetadataWriter,
    SqliteMetadataProvider, SqliteMetadataWriter,
};
use object_store::local::LocalFileSystem;
use sqlx::Row;
use sqlx::sqlite::SqlitePool;
use tempfile::TempDir;

type ObjStore = Arc<dyn object_store::ObjectStore>;

fn object_store() -> ObjStore {
    Arc::new(LocalFileSystem::new())
}

/// A writable SQLite-backed catalog + a data dir, in a temp dir.
async fn create_writer(temp_dir: &TempDir) -> Arc<SqliteMetadataWriter> {
    let db_path = temp_dir.path().join("test.db");
    let data_path = temp_dir.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();
    let conn_str = format!("sqlite:{}?mode=rwc", db_path.display());
    let writer = SqliteMetadataWriter::new_with_init(&conn_str)
        .await
        .unwrap();
    writer.set_data_path(data_path.to_str().unwrap()).unwrap();
    Arc::new(writer)
}

async fn pool_for(temp_dir: &TempDir) -> SqlitePool {
    let db_path = temp_dir.path().join("test.db");
    SqlitePool::connect(&format!("sqlite:{}", db_path.display()))
        .await
        .unwrap()
}

fn int32_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]))
}

/// Read `id`s from `test.main.t`, ascending, through the full read path (which
/// applies any live delete files).
async fn read_ids(temp_dir: &TempDir) -> Vec<i32> {
    let conn_str = format!("sqlite:{}", temp_dir.path().join("test.db").display());
    let provider = SqliteMetadataProvider::new(&conn_str).await.unwrap();
    let catalog = DuckLakeCatalog::new(provider).unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("test", Arc::new(catalog));
    let batches = ctx
        .sql("SELECT id FROM test.main.t ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let mut ids = Vec::new();
    for b in &batches {
        let c = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        for i in 0..b.num_rows() {
            ids.push(c.value(i));
        }
    }
    ids
}

/// Read `(id, extra)` rows, ascending by id — for the schema-evolution case
/// where rows from the pre-evolution file must null-fill `extra`.
async fn read_id_extra(temp_dir: &TempDir) -> Vec<(i32, Option<i32>)> {
    let conn_str = format!("sqlite:{}", temp_dir.path().join("test.db").display());
    let provider = SqliteMetadataProvider::new(&conn_str).await.unwrap();
    let catalog = DuckLakeCatalog::new(provider).unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("test", Arc::new(catalog));
    let batches = ctx
        .sql("SELECT id, extra FROM test.main.t ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let mut rows = Vec::new();
    for b in &batches {
        let ids = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        let extra = b.column(1).as_any().downcast_ref::<Int32Array>().unwrap();
        for i in 0..b.num_rows() {
            let e = if extra.is_null(i) {
                None
            } else {
                Some(extra.value(i))
            };
            rows.push((ids.value(i), e));
        }
    }
    rows
}

/// The ORACLE, single column: which ids survive deleting `del`, with zero
/// position math — delete-by-key removes *every* matching row. Sorted to match
/// the `ORDER BY id` read. (Replace-on-survivors in spirit: exactly the set a
/// full rewrite-keeping-survivors retains.)
fn survivors(ids: &[i32], del: &[i32]) -> Vec<i32> {
    let mut s: Vec<i32> = ids.iter().copied().filter(|x| !del.contains(x)).collect();
    s.sort_unstable();
    s
}

/// Write `ids` as one data file with `max_row_group_rows = rg` (so `rg < ids.len()`
/// yields a genuinely multi-row-group file).
async fn write_initial(writer: Arc<SqliteMetadataWriter>, os: ObjStore, ids: &[i32], rg: usize) {
    let batch = RecordBatch::try_new(
        int32_schema(),
        vec![Arc::new(Int32Array::from(ids.to_vec()))],
    )
    .unwrap();
    DuckLakeTableWriter::new(writer, os)
        .unwrap()
        .with_max_row_group_rows(rg)
        .write_table("main", "t", &[batch])
        .await
        .unwrap();
}

/// Append `ids` as an additional same-schema data file.
async fn append_ids(writer: Arc<SqliteMetadataWriter>, os: ObjStore, ids: &[i32]) {
    let batch = RecordBatch::try_new(
        int32_schema(),
        vec![Arc::new(Int32Array::from(ids.to_vec()))],
    )
    .unwrap();
    DuckLakeTableWriter::new(writer, os)
        .unwrap()
        .append_table("main", "t", &[batch])
        .await
        .unwrap();
}

/// The real positional-delete path for the key set `del` (an `id IN (del)`
/// predicate), applied across **every** live data file and cumulative-aware:
/// resolve matching physical positions per file, author a `(file_path, pos)`
/// delete file, and register it superseding any prior live delete file for that
/// data file (the ≤1-live CAS). Pass the *full accumulated* key set to build a
/// cumulative generation. No-op for files with no match.
async fn apply_delete(
    temp_dir: &TempDir,
    writer: Arc<SqliteMetadataWriter>,
    os: ObjStore,
    del: &[i32],
) {
    if del.is_empty() {
        return;
    }
    let conn_str = format!("sqlite:{}", temp_dir.path().join("test.db").display());
    let pool = pool_for(temp_dir).await;

    let table_id: i64 =
        sqlx::query_scalar("SELECT table_id FROM ducklake_table WHERE end_snapshot IS NULL")
            .fetch_one(&pool)
            .await
            .unwrap();
    let files = sqlx::query(
        "SELECT data_file_id, path, path_is_relative, file_size_bytes
         FROM ducklake_data_file WHERE table_id = ? AND end_snapshot IS NULL ORDER BY data_file_id",
    )
    .bind(table_id)
    .fetch_all(&pool)
    .await
    .unwrap();

    // `id = d0 OR id = d1 OR ...`, index-based on the table's logical column 0.
    let data_schema = Schema::new(vec![Field::new("id", DataType::Int32, false)]);
    let id = col("id", &data_schema).unwrap();
    let predicate = del
        .iter()
        .map(|d| -> Arc<dyn PhysicalExpr> {
            Arc::new(BinaryExpr::new(id.clone(), Operator::Eq, lit(*d)))
        })
        .reduce(|acc, e| Arc::new(BinaryExpr::new(acc, Operator::Or, e)))
        .expect("del is non-empty");

    let provider = SqliteMetadataProvider::new(&conn_str).await.unwrap();
    let catalog = DuckLakeCatalog::new(provider).unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("test", Arc::new(catalog));
    let table_provider = ctx
        .catalog("test")
        .unwrap()
        .schema("main")
        .unwrap()
        .table("t")
        .await
        .unwrap()
        .unwrap();
    let table = (table_provider.as_ref() as &dyn std::any::Any)
        .downcast_ref::<DuckLakeTable>()
        .expect("provider is a DuckLakeTable");
    let state = ctx.state();

    for f in &files {
        let data_file_id: i64 = f.try_get(0).unwrap();
        let path: String = f.try_get(1).unwrap();
        let path_is_relative: bool = f.try_get::<i64, _>(2).unwrap() != 0;
        let size: i64 = f.try_get(3).unwrap();
        let data_file = DuckLakeFileData::new(path.clone(), path_is_relative, size);

        let positions: Vec<i64> = table
            .resolve_positions(&state, &data_file, predicate.clone())
            .await
            .unwrap()
            .into_iter()
            .collect();
        if positions.is_empty() {
            continue;
        }

        let prev: Option<i64> = sqlx::query_scalar(
            "SELECT delete_file_id FROM ducklake_delete_file
             WHERE data_file_id = ? AND end_snapshot IS NULL",
        )
        .bind(data_file_id)
        .fetch_optional(&pool)
        .await
        .unwrap();

        let del_info = DuckLakeTableWriter::new(writer.clone(), os.clone())
            .unwrap()
            .write_delete_file("main", "t", &path, &positions)
            .await
            .unwrap();
        let base: i64 = sqlx::query_scalar("SELECT MAX(snapshot_id) FROM ducklake_snapshot")
            .fetch_one(&pool)
            .await
            .unwrap();
        writer
            .set_delete_file(
                table_id,
                "main",
                "t",
                base,
                data_file_id,
                prev,
                base,
                &del_info,
            )
            .unwrap();
    }
}

// ---------------------------------------------------------------------------
// 1. Curated multi-row-group shapes.
// ---------------------------------------------------------------------------

struct Shape {
    ids: Vec<i32>,
    rg: usize,
    del: Vec<i32>,
    note: &'static str,
}

#[tokio::test(flavor = "multi_thread")]
async fn curated_multi_row_group_shapes() {
    let shapes = vec![
        Shape {
            ids: (1..=4).collect(),
            rg: 100,
            del: vec![2, 4],
            note: "single row group (control)",
        },
        Shape {
            ids: (1..=6).collect(),
            rg: 2,
            del: vec![3, 5],
            note: "3 row groups, deletes in different groups",
        },
        Shape {
            ids: (1..=6).collect(),
            rg: 2,
            del: vec![1, 2],
            note: "delete an entire row group",
        },
        Shape {
            ids: (1..=6).collect(),
            rg: 4,
            del: vec![4, 5],
            note: "delete spans the row-group boundary",
        },
        Shape {
            ids: vec![10, 20, 30, 40, 50, 60],
            rg: 2,
            del: vec![10, 30, 50],
            note: "first row of each group",
        },
        Shape {
            ids: (1..=5).collect(),
            rg: 2,
            del: vec![5],
            note: "last row, ragged final group",
        },
        Shape {
            ids: vec![1, 2, 3],
            rg: 2,
            del: vec![1, 2, 3],
            note: "delete every row",
        },
        Shape {
            ids: (1..=4).collect(),
            rg: 2,
            del: vec![99],
            note: "no match — nothing deleted",
        },
    ];

    for shape in shapes {
        let temp_dir = TempDir::new().unwrap();
        let writer = create_writer(&temp_dir).await;
        let os = object_store();

        write_initial(writer.clone(), os.clone(), &shape.ids, shape.rg).await;
        assert_eq!(
            read_ids(&temp_dir).await,
            survivors(&shape.ids, &[]),
            "baseline [{}]",
            shape.note
        );

        apply_delete(&temp_dir, writer.clone(), os.clone(), &shape.del).await;

        assert_eq!(
            read_ids(&temp_dir).await,
            survivors(&shape.ids, &shape.del),
            "survivors mismatch [{}] rg={} del={:?}",
            shape.note,
            shape.rg,
            shape.del
        );
    }
}

// ---------------------------------------------------------------------------
// 2. Bounded exhaustive sweep — every delete subset × row-group size, for a
//    few small file sizes. This is the position-math safety net: it is complete
//    over the small region where boundary bugs live, and fully deterministic.
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn exhaustive_small_files_all_delete_subsets() {
    for n in [4usize, 5, 6] {
        let ids: Vec<i32> = (1..=n as i32).collect();
        for rg in [1usize, 2, 3] {
            for mask in 0u32..(1u32 << n) {
                let del: Vec<i32> = (0..n)
                    .filter(|i| mask & (1 << i) != 0)
                    .map(|i| ids[i])
                    .collect();

                let temp_dir = TempDir::new().unwrap();
                let writer = create_writer(&temp_dir).await;
                let os = object_store();

                write_initial(writer.clone(), os.clone(), &ids, rg).await;
                apply_delete(&temp_dir, writer.clone(), os.clone(), &del).await;

                assert_eq!(
                    read_ids(&temp_dir).await,
                    survivors(&ids, &del),
                    "n={n} rg={rg} del={del:?}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 3. Deletes across multiple data files (append), each with its own positions.
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn deletes_across_appended_files() {
    let temp_dir = TempDir::new().unwrap();
    let writer = create_writer(&temp_dir).await;
    let os = object_store();

    // Two data files, each multi-row-group.
    write_initial(writer.clone(), os.clone(), &[1, 2, 3, 4], 2).await;
    append_ids(writer.clone(), os.clone(), &[5, 6, 7, 8]).await;
    assert_eq!(
        read_ids(&temp_dir).await,
        vec![1, 2, 3, 4, 5, 6, 7, 8],
        "baseline"
    );

    // Delete keys living in *both* files: 2,3 (file A) and 6,8 (file B).
    let del = vec![2, 3, 6, 8];
    apply_delete(&temp_dir, writer.clone(), os.clone(), &del).await;

    assert_eq!(
        read_ids(&temp_dir).await,
        survivors(&[1, 2, 3, 4, 5, 6, 7, 8], &del),
        "survivors across two files"
    );
}

// ---------------------------------------------------------------------------
// 4. Deletes across a schema-evolved table (old file lacks the added column).
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn deletes_across_schema_evolution() {
    let temp_dir = TempDir::new().unwrap();
    let writer = create_writer(&temp_dir).await;
    let os = object_store();

    // File 1: schema {id}.
    write_initial(writer.clone(), os.clone(), &[1, 2, 3], 2).await;

    // File 2: append under a WIDER schema — adds nullable `extra` (DDL). Old
    // file's rows must null-fill `extra` on read.
    let wider = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("extra", DataType::Int32, true),
    ]));
    let b2 = RecordBatch::try_new(
        wider,
        vec![
            Arc::new(Int32Array::from(vec![4, 5])),
            Arc::new(Int32Array::from(vec![Some(40), Some(50)])),
        ],
    )
    .unwrap();
    DuckLakeTableWriter::new(writer.clone(), os.clone())
        .unwrap()
        .append_table("main", "t", &[b2])
        .await
        .unwrap();

    assert_eq!(
        read_id_extra(&temp_dir).await,
        vec![(1, None), (2, None), (3, None), (4, Some(40)), (5, Some(50))],
        "baseline: pre-evolution rows null-fill extra"
    );

    // Delete one row from each file: 2 (old-schema file) and 5 (new-schema file).
    apply_delete(&temp_dir, writer.clone(), os.clone(), &[2, 5]).await;

    assert_eq!(
        read_id_extra(&temp_dir).await,
        vec![(1, None), (3, None), (4, Some(40))],
        "survivors keep correct (id, extra) across the schema boundary"
    );
}

// ---------------------------------------------------------------------------
// 5. Update = positional delete of old versions + append of new versions
//    (models the update flow at the crate primitive level). A re-inserted deleted key
//    survives (it lands in a new file the delete doesn't cover).
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn update_as_delete_then_insert() {
    let temp_dir = TempDir::new().unwrap();
    let writer = create_writer(&temp_dir).await;
    let os = object_store();

    let orig = vec![1, 2, 3, 4, 5, 6];
    write_initial(writer.clone(), os.clone(), &orig, 2).await;

    // Delete 2 and 4 from the original file, then append new rows [2, 7] — id 2
    // is re-inserted (must survive), id 7 is new.
    let del = vec![2, 4];
    apply_delete(&temp_dir, writer.clone(), os.clone(), &del).await;
    append_ids(writer.clone(), os.clone(), &[2, 7]).await;

    let mut want = survivors(&orig, &del); // [1,3,5,6]
    want.extend_from_slice(&[2, 7]);
    want.sort_unstable(); // [1,2,3,5,6,7]

    assert_eq!(
        read_ids(&temp_dir).await,
        want,
        "update: delete old + insert new"
    );
}

// ---------------------------------------------------------------------------
// 6. Multi-generation (cumulative) deletes: a second delete supersedes the
//    first with the accumulated positions; ≤1 live delete file remains.
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn cumulative_deletes_across_generations() {
    let temp_dir = TempDir::new().unwrap();
    let writer = create_writer(&temp_dir).await;
    let os = object_store();

    let ids: Vec<i32> = (1..=8).collect();
    write_initial(writer.clone(), os.clone(), &ids, 3).await;

    // Generation 1: delete {2,4}.
    apply_delete(&temp_dir, writer.clone(), os.clone(), &[2, 4]).await;
    assert_eq!(
        read_ids(&temp_dir).await,
        survivors(&ids, &[2, 4]),
        "after gen 1"
    );

    // Generation 2: the accumulated set {2,4,6,8} — supersedes gen 1.
    apply_delete(&temp_dir, writer.clone(), os.clone(), &[2, 4, 6, 8]).await;
    assert_eq!(
        read_ids(&temp_dir).await,
        survivors(&ids, &[2, 4, 6, 8]),
        "after gen 2 (cumulative)"
    );

    // Invariant: exactly one live delete file remains for the single data file.
    let pool = pool_for(&temp_dir).await;
    let live: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM ducklake_delete_file WHERE end_snapshot IS NULL")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(
        live, 1,
        "≤1 live delete file per data file after superseding"
    );
}
