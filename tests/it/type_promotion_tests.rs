//! Library-level write→read VALUE round-trip + schema-evolution tests
//! (§14 of `docs/column-id-versioning-design.md`).
//!
//! These exist so datafusion-ducklake catches its own type-evolution
//! regressions instead of a downstream consumer — the gap that let bug C
//! (silent type-drop on Replace) and #148 (List read-back all-NULL) reach
//! runtimedb's tests first. Every test writes real batches and reads VALUES
//! back through the full provider.

#![cfg(all(feature = "write-sqlite", feature = "metadata-sqlite"))]

use std::sync::Arc;

use arrow::array::{Array, Float32Array, Float64Array, Int32Array, Int64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::prelude::*;
use object_store::local::LocalFileSystem;
use tempfile::TempDir;

use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeError, DuckLakeTableWriter, MetadataWriter, SqliteMetadataProvider,
    SqliteMetadataWriter, TypeChangeOperation, TypeChangeWriteMode, WriteMode,
};
use sqlx::Row;
use sqlx::sqlite::SqlitePool;

/// Open a read pool over the catalog's SQLite file.
async fn open_pool(temp_dir: &TempDir) -> SqlitePool {
    let db_path = temp_dir.path().join("test.db");
    SqlitePool::connect(&format!("sqlite:{}", db_path.display()))
        .await
        .unwrap()
}

/// The catalog's current (max) `schema_version`.
async fn max_schema_version(pool: &SqlitePool) -> i64 {
    sqlx::query_scalar("SELECT COALESCE(MAX(schema_version), 0) FROM ducklake_snapshot")
        .fetch_one(pool)
        .await
        .unwrap()
}

fn create_object_store() -> Arc<dyn object_store::ObjectStore> {
    Arc::new(LocalFileSystem::new())
}

/// A writable SQLite-backed test catalog in a temp dir.
async fn create_test_env() -> (SqliteMetadataWriter, TempDir) {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("test.db");
    let data_path = temp_dir.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();

    let conn_str = format!("sqlite:{}?mode=rwc", db_path.display());
    let writer = SqliteMetadataWriter::new_with_init(&conn_str)
        .await
        .unwrap();
    writer.set_data_path(data_path.to_str().unwrap()).unwrap();

    (writer, temp_dir)
}

fn assert_unsupported_type_change<T>(
    res: datafusion_ducklake::Result<T>,
    expected_operation: TypeChangeOperation,
    expected_column: &str,
    expected_from: &str,
    expected_to: &str,
    context: &str,
) {
    match res {
        Err(DuckLakeError::UnsupportedTypeChange {
            operation,
            column,
            from,
            to,
        }) => {
            assert_eq!(operation, expected_operation, "{context}: operation");
            assert_eq!(column, expected_column, "{context}: column");
            assert_eq!(from, expected_from, "{context}: from");
            assert_eq!(to, expected_to, "{context}: to");
        },
        Err(other) => panic!("{context}: expected UnsupportedTypeChange, got {other:?}"),
        Ok(_) => panic!("{context}: expected UnsupportedTypeChange, got Ok"),
    }
}

fn assert_invalid_config<T>(res: datafusion_ducklake::Result<T>, context: &str) {
    match res {
        Err(DuckLakeError::InvalidConfig(_)) => {},
        Err(DuckLakeError::UnsupportedTypeChange {
            ..
        }) => {
            panic!("{context}: expected InvalidConfig, got UnsupportedTypeChange")
        },
        Err(other) => panic!("{context}: expected InvalidConfig, got {other:?}"),
        Ok(_) => panic!("{context}: expected InvalidConfig, got Ok"),
    }
}

/// A read-only `SessionContext` over the same catalog (registered as `test`).
async fn create_read_context(temp_dir: &TempDir) -> SessionContext {
    let db_path = temp_dir.path().join("test.db");
    let conn_str = format!("sqlite:{}", db_path.display());

    let provider = SqliteMetadataProvider::new(&conn_str).await.unwrap();
    let catalog = DuckLakeCatalog::new(provider).unwrap();

    let ctx = SessionContext::new();
    ctx.register_catalog("test", Arc::new(catalog));
    ctx
}

fn int32_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]))
}

fn int64_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]))
}

/// Positive control: a same-schema `Replace` round-trips values. Validates the
/// harness independent of any schema-evolution behavior.
#[tokio::test(flavor = "multi_thread")]
async fn replace_same_schema_roundtrips_values() {
    let (writer, temp_dir) = create_test_env().await;
    let object_store = create_object_store();

    let batch = RecordBatch::try_new(
        int32_schema(),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
    )
    .unwrap();
    DuckLakeTableWriter::new(Arc::new(writer), Arc::clone(&object_store))
        .unwrap()
        .write_table("main", "t", &[batch])
        .await
        .unwrap();

    let ctx = create_read_context(&temp_dir).await;
    let batches = ctx
        .sql("SELECT id FROM test.main.t ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let ids = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    assert_eq!(ids.values(), &[1, 2, 3]);
}

/// C1 (the bug this PR fixes): a `Replace` that CHANGES a column's type must be
/// REJECTED, not silently dropped. Before §5, the writer keeps the old catalog
/// type and the type change is lost (reads serve the stale type). A widen is a
/// schema change and must go through `promote_column_type`, never a data write.
#[tokio::test(flavor = "multi_thread")]
async fn replace_with_type_change_is_rejected() {
    let (writer, _temp_dir) = create_test_env().await;
    let object_store = create_object_store();

    // t(id int32)
    let b32 = RecordBatch::try_new(
        int32_schema(),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
    )
    .unwrap();
    DuckLakeTableWriter::new(Arc::new(writer.clone()), Arc::clone(&object_store))
        .unwrap()
        .write_table("main", "t", &[b32])
        .await
        .unwrap();

    // Replace with id as int64, value beyond the i32 range.
    let b64 = RecordBatch::try_new(
        int64_schema(),
        vec![Arc::new(Int64Array::from(vec![5_000_000_000_i64]))],
    )
    .unwrap();
    let res = DuckLakeTableWriter::new(Arc::new(writer), Arc::clone(&object_store))
        .unwrap()
        .write_table("main", "t", &[b64])
        .await;

    assert_unsupported_type_change(
        res,
        TypeChangeOperation::DataWrite {
            mode: TypeChangeWriteMode::Replace,
        },
        "id",
        "int32",
        "int64",
        "Replace with a column type change must be rejected",
    );
}

/// C2: an `Append` that WIDENS a column's type must be REJECTED too. Before §5,
/// `types_compatible` accepts widenings and the append silently proceeds under
/// the old catalog type.
#[tokio::test(flavor = "multi_thread")]
async fn append_with_type_change_is_rejected() {
    let (writer, _temp_dir) = create_test_env().await;
    let object_store = create_object_store();

    let b32 = RecordBatch::try_new(
        int32_schema(),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
    )
    .unwrap();
    DuckLakeTableWriter::new(Arc::new(writer.clone()), Arc::clone(&object_store))
        .unwrap()
        .write_table("main", "t", &[b32])
        .await
        .unwrap();

    let b64 =
        RecordBatch::try_new(int64_schema(), vec![Arc::new(Int64Array::from(vec![4, 5]))]).unwrap();
    let res = DuckLakeTableWriter::new(Arc::new(writer), Arc::clone(&object_store))
        .unwrap()
        .append_table("main", "t", &[b64])
        .await;

    assert_unsupported_type_change(
        res,
        TypeChangeOperation::DataWrite {
            mode: TypeChangeWriteMode::Append,
        },
        "id",
        "int32",
        "int64",
        "Append with a column type change must be rejected",
    );
}

/// §14 A1 — the heart of the feature: after an explicit `promote_column_type`
/// widens `id` from int32 to int64, a read sees the column as Int64 AND the values
/// in the OLD (physically int32) file come back intact. This also empirically
/// answers the cast-on-read open question: if DataFusion up-casts the old narrow
/// file for the wide read schema we hand it, this passes as-is.
#[tokio::test(flavor = "multi_thread")]
async fn promote_widens_column_and_old_values_read_back() {
    let (writer, temp_dir) = create_test_env().await;
    let object_store = create_object_store();

    // t(id int32) = [1, 2, 3], written to a physically-int32 Parquet file.
    let b32 = RecordBatch::try_new(
        int32_schema(),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
    )
    .unwrap();
    let res = DuckLakeTableWriter::new(Arc::new(writer.clone()), Arc::clone(&object_store))
        .unwrap()
        .write_table("main", "t", &[b32])
        .await
        .unwrap();

    // Explicit schema evolution: widen id int32 -> int64 (no data rewritten).
    let new_snapshot = writer
        .promote_column_type(res.table_id, "id", "int64")
        .unwrap();
    assert!(
        new_snapshot > res.snapshot_id,
        "promote creates a newer snapshot"
    );

    // Now append a value BEYOND the int32 range under the widened type — exactly
    // the value bug C used to silently lose. This is a normal data write (the
    // catalog type already matches int64, so §5 does not reject it).
    let beyond_i32 = 5_000_000_000_i64; // > i32::MAX
    let b64 = RecordBatch::try_new(
        int64_schema(),
        vec![Arc::new(Int64Array::from(vec![beyond_i32]))],
    )
    .unwrap();
    DuckLakeTableWriter::new(Arc::new(writer), Arc::clone(&object_store))
        .unwrap()
        .append_table("main", "t", &[b64])
        .await
        .unwrap();

    // A single read now spans the OLD int32 file and the NEW int64 file, all under
    // the widened type: old values up-cast, and the beyond-range value is intact.
    let ctx = create_read_context(&temp_dir).await;
    let batches = ctx
        .sql("SELECT id FROM test.main.t ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(
        batches[0].schema().field(0).data_type(),
        &DataType::Int64,
        "promoted column must read as Int64"
    );
    let mut got: Vec<i64> = Vec::new();
    for b in &batches {
        let ids = b
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("id column should be Int64 after promote");
        got.extend(ids.values().iter().copied());
    }
    got.sort();
    assert_eq!(
        got,
        vec![1, 2, 3, beyond_i32],
        "old int32 file up-casts AND the beyond-i32 value (bug C) survives"
    );
}

/// `promote_column_type` only allows lossless widenings — a narrowing (or any
/// non-promotable change) is a hard error, never a silent data-corrupting cast.
#[tokio::test(flavor = "multi_thread")]
async fn promote_rejects_non_widening() {
    let (writer, _temp_dir) = create_test_env().await;
    let object_store = create_object_store();

    let b64 = RecordBatch::try_new(
        int64_schema(),
        vec![Arc::new(Int64Array::from(vec![1_i64, 2, 3]))],
    )
    .unwrap();
    let res = DuckLakeTableWriter::new(Arc::new(writer.clone()), Arc::clone(&object_store))
        .unwrap()
        .write_table("main", "t", &[b64])
        .await
        .unwrap();

    // int64 -> int32 is a narrowing; must be rejected.
    let narrow = writer.promote_column_type(res.table_id, "id", "int32");
    assert_unsupported_type_change(
        narrow,
        TypeChangeOperation::PromoteColumnType,
        "id",
        "int64",
        "int32",
        "narrowing promote must be rejected",
    );

    // int64 -> int64 is a no-op; reported as an error (no change), not applied.
    let noop = writer.promote_column_type(res.table_id, "id", "bigint");
    assert_invalid_config(
        noop,
        "no-op (same canonical type) promote must be rejected without UnsupportedTypeChange",
    );

    let missing = writer.promote_column_type(res.table_id, "missing", "int32");
    assert_invalid_config(
        missing,
        "missing-column promote must be rejected without UnsupportedTypeChange",
    );
}

/// §4.6 race — settled empirically. An Append SESSION begins under the old (int32)
/// schema and stages its file; THEN a promote (int32 -> int64) commits; THEN the
/// session finishes. The question: does this corrupt reads (the review's High #1),
/// or does cast-on-read make a racing narrow file benign for a *widening* promote?
///
/// The assertion is the invariant that actually matters: whatever the outcome
/// (the append commits, or it cleanly aborts with an error), a subsequent read
/// must be correct — no NULL-filled rows, no wrong values. The printed outcome
/// tells us whether an explicit commit-time abort-guard is still needed.
#[tokio::test(flavor = "multi_thread")]
async fn append_session_racing_a_promote_never_corrupts() {
    let (writer, temp_dir) = create_test_env().await;
    let object_store = create_object_store();

    // t(id int32) = [1, 2, 3].
    let b32 = RecordBatch::try_new(
        int32_schema(),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
    )
    .unwrap();
    let res = DuckLakeTableWriter::new(Arc::new(writer.clone()), Arc::clone(&object_store))
        .unwrap()
        .write_table("main", "t", &[b32])
        .await
        .unwrap();

    // Begin an Append session under the CURRENT (int32) schema — stages before promote.
    let tw = DuckLakeTableWriter::new(Arc::new(writer.clone()), Arc::clone(&object_store)).unwrap();
    let mut session = tw
        .begin_write("main", "t", &int32_schema(), WriteMode::Append)
        .unwrap();
    let b32b =
        RecordBatch::try_new(int32_schema(), vec![Arc::new(Int32Array::from(vec![4, 5]))]).unwrap();
    session.write_batch(&b32b).unwrap();

    // Meanwhile a promote int32 -> int64 commits.
    writer
        .promote_column_type(res.table_id, "id", "int64")
        .unwrap();

    // The append session finishes AFTER the promote.
    let finish_result = session.finish().await;

    // Invariant: the read is clean regardless of which way the race resolved.
    let ctx = create_read_context(&temp_dir).await;
    let batches = ctx
        .sql("SELECT id FROM test.main.t ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(
        batches[0].schema().field(0).data_type(),
        &DataType::Int64,
        "column reads as Int64 after the promote"
    );
    let mut got: Vec<i64> = Vec::new();
    for b in &batches {
        let ids = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        for i in 0..b.num_rows() {
            assert!(!ids.is_null(i), "race must never NULL-fill a row");
            got.push(ids.value(i));
        }
    }
    got.sort();
    // On SQLite the write lock serializes the two, so the Append commits (its
    // int32 file casts up under the now-int64 column — benign), and all rows are
    // present and correct. No NULL-fill, no lost rows, no reverted type.
    assert!(
        finish_result.is_ok(),
        "the racing Append should commit cleanly: {finish_result:?}"
    );
    assert_eq!(
        got,
        vec![1, 2, 3, 4, 5],
        "racing Append commits with all values intact under the widened type"
    );
}

fn f32_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![Field::new("v", DataType::Float32, false)]))
}

fn f64_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![Field::new("v", DataType::Float64, false)]))
}

/// §14 A4 — cast-on-read is not int-specific: a Float32 -> Float64 promote reads the
/// old f32 file back as f64 with values intact (1.5/2.5 are exact in both widths).
#[tokio::test(flavor = "multi_thread")]
async fn promote_float32_to_float64_roundtrip() {
    let (writer, temp_dir) = create_test_env().await;
    let object_store = create_object_store();

    let bf32 = RecordBatch::try_new(
        f32_schema(),
        vec![Arc::new(Float32Array::from(vec![1.5_f32, 2.5]))],
    )
    .unwrap();
    let res = DuckLakeTableWriter::new(Arc::new(writer.clone()), Arc::clone(&object_store))
        .unwrap()
        .write_table("main", "f", &[bf32])
        .await
        .unwrap();

    writer
        .promote_column_type(res.table_id, "v", "float64")
        .unwrap();

    let bf64 = RecordBatch::try_new(
        f64_schema(),
        vec![Arc::new(Float64Array::from(vec![1.0e10_f64]))],
    )
    .unwrap();
    DuckLakeTableWriter::new(Arc::new(writer), Arc::clone(&object_store))
        .unwrap()
        .append_table("main", "f", &[bf64])
        .await
        .unwrap();

    let ctx = create_read_context(&temp_dir).await;
    let batches = ctx
        .sql("SELECT v FROM test.main.f ORDER BY v")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(batches[0].schema().field(0).data_type(), &DataType::Float64);
    let mut got: Vec<f64> = Vec::new();
    for b in &batches {
        let vs = b.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        got.extend(vs.values().iter().copied());
    }
    assert_eq!(
        got,
        vec![1.5, 2.5, 1.0e10],
        "f32 file up-casts to f64, values intact"
    );
}

/// §14 A1/G3 — a FILTER on the promoted column spans the old (int32) and new
/// (int64) files: the predicate must apply correctly after cast-on-read. `id >
/// 4e9` keeps only the beyond-i32 row from the new file, drops the old [1,2,3].
#[tokio::test(flavor = "multi_thread")]
async fn filter_on_promoted_column_spans_old_and_new_files() {
    let (writer, temp_dir) = create_test_env().await;
    let object_store = create_object_store();

    let b32 = RecordBatch::try_new(
        int32_schema(),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
    )
    .unwrap();
    let res = DuckLakeTableWriter::new(Arc::new(writer.clone()), Arc::clone(&object_store))
        .unwrap()
        .write_table("main", "t", &[b32])
        .await
        .unwrap();
    writer
        .promote_column_type(res.table_id, "id", "int64")
        .unwrap();
    let b64 = RecordBatch::try_new(
        int64_schema(),
        vec![Arc::new(Int64Array::from(vec![5_000_000_000_i64]))],
    )
    .unwrap();
    DuckLakeTableWriter::new(Arc::new(writer), Arc::clone(&object_store))
        .unwrap()
        .append_table("main", "t", &[b64])
        .await
        .unwrap();

    let ctx = create_read_context(&temp_dir).await;
    let batches = ctx
        .sql("SELECT id FROM test.main.t WHERE id > 4000000000 ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let mut got: Vec<i64> = Vec::new();
    for b in &batches {
        let ids = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        got.extend(ids.values().iter().copied());
    }
    assert_eq!(
        got,
        vec![5_000_000_000],
        "filter on the promoted column must drop old-file rows below the bound and keep the beyond-i32 row"
    );
}

/// §14 G3: COUNT(*) across a promote boundary — the zero-column COUNT scan must
/// count rows from the old (int32) file AND the new (int64) file correctly.
#[tokio::test(flavor = "multi_thread")]
async fn count_star_across_promote_boundary() {
    let (writer, temp_dir) = create_test_env().await;
    let object_store = create_object_store();

    let b32 = RecordBatch::try_new(
        int32_schema(),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
    )
    .unwrap();
    let res = DuckLakeTableWriter::new(Arc::new(writer.clone()), Arc::clone(&object_store))
        .unwrap()
        .write_table("main", "t", &[b32])
        .await
        .unwrap();
    writer
        .promote_column_type(res.table_id, "id", "int64")
        .unwrap();
    let b64 = RecordBatch::try_new(
        int64_schema(),
        vec![Arc::new(Int64Array::from(vec![5_000_000_000_i64, 6_000_000_000]))],
    )
    .unwrap();
    DuckLakeTableWriter::new(Arc::new(writer), Arc::clone(&object_store))
        .unwrap()
        .append_table("main", "t", &[b64])
        .await
        .unwrap();

    let ctx = create_read_context(&temp_dir).await;
    let batches = ctx
        .sql("SELECT COUNT(*) FROM test.main.t")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let cnt = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!(
        cnt, 5,
        "COUNT(*) must span the old int32 file + new int64 file (3 + 2)"
    );
}

// ── schema_version tracking (issue #151: SQLite snapshot DDL model) ──

/// A fresh catalog is shaped to track schema_version: `ducklake_snapshot` carries
/// the `schema_version` column and the `ducklake_schema_versions` ledger table
/// exists. Matches the Postgres writer (and upstream, modulo the deliberately
/// omitted `next_catalog_id`/`next_file_id` allocator columns).
#[tokio::test(flavor = "multi_thread")]
async fn new_catalog_has_schema_version_shape() {
    let (_writer, temp_dir) = create_test_env().await;
    let pool = open_pool(&temp_dir).await;

    let has_col: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM pragma_table_info('ducklake_snapshot') WHERE name = 'schema_version'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(
        has_col, 1,
        "ducklake_snapshot must carry a schema_version column"
    );

    let has_table: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'ducklake_schema_versions'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(
        has_table, 1,
        "ducklake_schema_versions ledger table must exist"
    );
}

/// A type promotion is DDL: it bumps the per-catalog `schema_version` and writes
/// one `ducklake_schema_versions` ledger row at the promote snapshot. Resolves the
/// `TODO(schema_version)` left in `promote_column_type` by #149.
#[tokio::test(flavor = "multi_thread")]
async fn promote_bumps_schema_version_and_records_ledger_row() {
    let (writer, temp_dir) = create_test_env().await;
    let object_store = create_object_store();

    // Create t(id int32) — DDL, so schema_version becomes 1.
    let batch = RecordBatch::try_new(
        int32_schema(),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
    )
    .unwrap();
    DuckLakeTableWriter::new(Arc::new(writer.clone()), Arc::clone(&object_store))
        .unwrap()
        .write_table("main", "t", &[batch])
        .await
        .unwrap();

    let pool = open_pool(&temp_dir).await;
    let v_before = max_schema_version(&pool).await;
    assert_eq!(v_before, 1, "table creation is DDL → schema_version 1");

    // Promote id int32 -> int64.
    let table_id: i64 = sqlx::query_scalar(
        "SELECT table_id FROM ducklake_table WHERE table_name = 't' AND end_snapshot IS NULL",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    let promote_snapshot = writer.promote_column_type(table_id, "id", "int64").unwrap();

    let v_after = max_schema_version(&pool).await;
    assert_eq!(
        v_after,
        v_before + 1,
        "promote is DDL → schema_version bumps by 1"
    );

    // Exactly one ledger row, at the promote snapshot, with the bumped version.
    let row = sqlx::query(
        "SELECT begin_snapshot, schema_version, table_id
         FROM ducklake_schema_versions WHERE table_id = ? AND begin_snapshot = ?",
    )
    .bind(table_id)
    .bind(promote_snapshot)
    .fetch_one(&pool)
    .await
    .unwrap();
    let ledger_version: i64 = row.try_get("schema_version").unwrap();
    assert_eq!(
        ledger_version, v_after,
        "ledger row carries the bumped schema_version"
    );
}

/// A pure data write (an Append with an unchanged schema) must NOT bump
/// `schema_version` — only schema changes do. This is the trap the shared
/// `columns_differ` classifier guards against; mirrors the Postgres E2 test.
#[tokio::test(flavor = "multi_thread")]
async fn data_write_does_not_bump_schema_version() {
    let (writer, temp_dir) = create_test_env().await;
    let object_store = create_object_store();

    // Create t(id int32) — DDL → schema_version 1.
    let b1 = RecordBatch::try_new(
        int32_schema(),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
    )
    .unwrap();
    DuckLakeTableWriter::new(Arc::new(writer.clone()), Arc::clone(&object_store))
        .unwrap()
        .write_table("main", "t", &[b1])
        .await
        .unwrap();

    let pool = open_pool(&temp_dir).await;
    let v_after_create = max_schema_version(&pool).await;
    assert_eq!(v_after_create, 1);

    // Append more rows under the SAME schema — a pure data write.
    let b2 =
        RecordBatch::try_new(int32_schema(), vec![Arc::new(Int32Array::from(vec![4, 5]))]).unwrap();
    DuckLakeTableWriter::new(Arc::new(writer), Arc::clone(&object_store))
        .unwrap()
        .append_table("main", "t", &[b2])
        .await
        .unwrap();

    let v_after_append = max_schema_version(&pool).await;
    assert_eq!(
        v_after_append, v_after_create,
        "a same-schema Append is a data write and must NOT bump schema_version"
    );

    // And it wrote no ledger row for the append snapshot.
    let ledger_rows: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM ducklake_schema_versions")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(
        ledger_rows, 1,
        "only the table-creation DDL writes a ledger row; the Append writes none"
    );
}

/// Resolve the live `table_id` for `main.t` in the catalog.
async fn live_table_id(pool: &SqlitePool) -> i64 {
    sqlx::query_scalar(
        "SELECT table_id FROM ducklake_table WHERE table_name = 't' AND end_snapshot IS NULL",
    )
    .fetch_one(pool)
    .await
    .unwrap()
}

/// A `drop_table` is DDL: it bumps `schema_version` but writes NO
/// `ducklake_schema_versions` row (the table has no live schema afterward). Drop
/// is the one DDL where the bump and the ledger diverge — the easiest place for
/// the two halves to drift — so it gets its own test (mirrors the Postgres path).
#[tokio::test(flavor = "multi_thread")]
async fn drop_table_bumps_schema_version_without_ledger_row() {
    let (writer, temp_dir) = create_test_env().await;
    let object_store = create_object_store();

    let batch = RecordBatch::try_new(
        int32_schema(),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
    )
    .unwrap();
    DuckLakeTableWriter::new(Arc::new(writer.clone()), Arc::clone(&object_store))
        .unwrap()
        .write_table("main", "t", &[batch])
        .await
        .unwrap();

    let pool = open_pool(&temp_dir).await;
    let v_before = max_schema_version(&pool).await;
    assert_eq!(v_before, 1);

    assert!(
        writer.drop_table("main", "t").unwrap(),
        "table existed → dropped"
    );

    let v_after = max_schema_version(&pool).await;
    assert_eq!(v_after, v_before + 1, "drop is DDL → schema_version bumps");

    // The drop snapshot is the latest; it must carry NO ledger row, and the total
    // ledger count stays at 1 (just the table-creation row).
    let drop_snap: i64 = sqlx::query_scalar("SELECT MAX(snapshot_id) FROM ducklake_snapshot")
        .fetch_one(&pool)
        .await
        .unwrap();
    let at_drop: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM ducklake_schema_versions WHERE begin_snapshot = ?",
    )
    .bind(drop_snap)
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(at_drop, 0, "a drop writes no ducklake_schema_versions row");
    let total: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM ducklake_schema_versions")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(total, 1, "only the create DDL left a ledger row");
}

/// An Append that ADDS a (nullable) column is a schema change, classified as DDL
/// through the shared `columns_differ` — the exact path that helper was extracted
/// for. It must bump `schema_version` and write a ledger row. (The promote tests
/// never exercise `columns_differ`; this is its only SQLite write-path coverage.)
#[tokio::test(flavor = "multi_thread")]
async fn append_adding_column_is_ddl_and_bumps_schema_version() {
    let (writer, temp_dir) = create_test_env().await;
    let object_store = create_object_store();

    // t(id int32) — DDL → schema_version 1.
    let b1 = RecordBatch::try_new(
        int32_schema(),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
    )
    .unwrap();
    DuckLakeTableWriter::new(Arc::new(writer.clone()), Arc::clone(&object_store))
        .unwrap()
        .write_table("main", "t", &[b1])
        .await
        .unwrap();

    let pool = open_pool(&temp_dir).await;
    assert_eq!(max_schema_version(&pool).await, 1);
    let table_id = live_table_id(&pool).await;

    // Append rows under a WIDER schema: add a nullable `extra` column.
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
    DuckLakeTableWriter::new(Arc::new(writer), Arc::clone(&object_store))
        .unwrap()
        .append_table("main", "t", &[b2])
        .await
        .unwrap();

    let v_after = max_schema_version(&pool).await;
    assert_eq!(
        v_after, 2,
        "adding a column is DDL → schema_version bumps to 2"
    );
    let total: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM ducklake_schema_versions WHERE table_id = ?")
            .bind(table_id)
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(total, 2, "create + add-column each wrote a ledger row");
}

/// A same-schema `Replace` (CTAS over an existing table) is a pure data write and
/// must NOT bump `schema_version` — Replace takes a distinct sub-path from Append
/// (retire + re-insert column rows), so it gets its own no-bump test.
#[tokio::test(flavor = "multi_thread")]
async fn same_schema_replace_does_not_bump_schema_version() {
    let (writer, temp_dir) = create_test_env().await;
    let object_store = create_object_store();

    let b1 = RecordBatch::try_new(
        int32_schema(),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
    )
    .unwrap();
    DuckLakeTableWriter::new(Arc::new(writer.clone()), Arc::clone(&object_store))
        .unwrap()
        .write_table("main", "t", &[b1])
        .await
        .unwrap();

    let pool = open_pool(&temp_dir).await;
    assert_eq!(max_schema_version(&pool).await, 1);

    // Replace with the SAME schema, different rows.
    let b2 = RecordBatch::try_new(
        int32_schema(),
        vec![Arc::new(Int32Array::from(vec![7, 8, 9, 10]))],
    )
    .unwrap();
    DuckLakeTableWriter::new(Arc::new(writer), Arc::clone(&object_store))
        .unwrap()
        .write_table("main", "t", &[b2])
        .await
        .unwrap();

    assert_eq!(
        max_schema_version(&pool).await,
        1,
        "a same-schema Replace is a data write and must NOT bump schema_version"
    );
}

/// Sequential schema changes produce a dense, monotonic `schema_version`
/// (1 → 2 → 3), each with its own ledger row. Guards `bump_schema_version`'s
/// MAX-over-other-snapshots formulation against an off-by-one on the second bump.
#[tokio::test(flavor = "multi_thread")]
async fn sequential_promotes_are_monotonic() {
    let (writer, temp_dir) = create_test_env().await;
    let object_store = create_object_store();

    // t(a int32, b int32) — DDL → schema_version 1.
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int32, false),
        Field::new("b", DataType::Int32, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(Int32Array::from(vec![1, 2])), Arc::new(Int32Array::from(vec![3, 4]))],
    )
    .unwrap();
    DuckLakeTableWriter::new(Arc::new(writer.clone()), Arc::clone(&object_store))
        .unwrap()
        .write_table("main", "t", &[batch])
        .await
        .unwrap();

    let pool = open_pool(&temp_dir).await;
    let table_id = live_table_id(&pool).await;
    assert_eq!(max_schema_version(&pool).await, 1);

    writer.promote_column_type(table_id, "a", "int64").unwrap();
    assert_eq!(max_schema_version(&pool).await, 2, "first promote → 2");

    writer.promote_column_type(table_id, "b", "int64").unwrap();
    assert_eq!(max_schema_version(&pool).await, 3, "second promote → 3");

    // Three distinct ledger rows (create, promote a, promote b), versions 1/2/3.
    let versions: Vec<i64> = sqlx::query_scalar(
        "SELECT schema_version FROM ducklake_schema_versions WHERE table_id = ? ORDER BY schema_version",
    )
    .bind(table_id)
    .fetch_all(&pool)
    .await
    .unwrap();
    assert_eq!(versions, vec![1, 2, 3], "dense monotonic ledger versions");
}

/// End-to-end across the migration seam: a pre-existing catalog whose
/// `ducklake_snapshot` predates `schema_version` (with historical snapshots) is
/// migrated on open, then a write + promote must produce a correct, monotonic
/// `schema_version` that continues past the legacy snapshots. Covers the
/// migrate-then-write path the unit migration test does not exercise.
#[tokio::test(flavor = "multi_thread")]
async fn migrated_legacy_catalog_writes_and_promotes() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("test.db");
    let data_path = temp_dir.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();
    let conn_str = format!("sqlite:{}?mode=rwc", db_path.display());

    // Pre-create a LEGACY ducklake_snapshot (no schema_version) with history.
    {
        let pool = SqlitePool::connect(&conn_str).await.unwrap();
        sqlx::query(
            "CREATE TABLE ducklake_snapshot (
                snapshot_id INTEGER PRIMARY KEY,
                snapshot_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )",
        )
        .execute(&pool)
        .await
        .unwrap();
        sqlx::query("INSERT INTO ducklake_snapshot (snapshot_id) VALUES (1), (2)")
            .execute(&pool)
            .await
            .unwrap();
        pool.close().await;
    }

    // Open via the writer: runs SQL_CREATE_SCHEMA (creates the remaining tables) +
    // migrate_add_schema_version (adds the column, backfilling history to 0).
    let writer = SqliteMetadataWriter::new_with_init(&conn_str)
        .await
        .unwrap();
    writer.set_data_path(data_path.to_str().unwrap()).unwrap();
    let object_store = create_object_store();

    let batch = RecordBatch::try_new(
        int32_schema(),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
    )
    .unwrap();
    DuckLakeTableWriter::new(Arc::new(writer.clone()), Arc::clone(&object_store))
        .unwrap()
        .write_table("main", "t", &[batch])
        .await
        .unwrap();

    let pool = open_pool(&temp_dir).await;
    // First write is DDL; schema_version starts from the backfilled-0 history → 1.
    assert_eq!(
        max_schema_version(&pool).await,
        1,
        "first write after migration → version 1"
    );
    let table_id = live_table_id(&pool).await;

    writer.promote_column_type(table_id, "id", "int64").unwrap();
    assert_eq!(
        max_schema_version(&pool).await,
        2,
        "promote after migration → version 2"
    );

    // New snapshots continue past the legacy ids (no reuse/collision).
    let max_snap: i64 = sqlx::query_scalar("SELECT MAX(snapshot_id) FROM ducklake_snapshot")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert!(
        max_snap >= 4,
        "snapshot ids continue past the 2 legacy snapshots"
    );
}
