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
    DuckLakeCatalog, DuckLakeTableWriter, MetadataWriter, SqliteMetadataProvider,
    SqliteMetadataWriter, WriteMode,
};

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

    let batch =
        RecordBatch::try_new(int32_schema(), vec![Arc::new(Int32Array::from(vec![1, 2, 3]))])
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
    let b32 =
        RecordBatch::try_new(int32_schema(), vec![Arc::new(Int32Array::from(vec![1, 2, 3]))])
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

    assert!(
        res.is_err(),
        "Replace with a column type change must be rejected (got Ok — the silent type-drop bug C)"
    );
}

/// C2: an `Append` that WIDENS a column's type must be REJECTED too. Before §5,
/// `types_compatible` accepts widenings and the append silently proceeds under
/// the old catalog type.
#[tokio::test(flavor = "multi_thread")]
async fn append_with_type_change_is_rejected() {
    let (writer, _temp_dir) = create_test_env().await;
    let object_store = create_object_store();

    let b32 =
        RecordBatch::try_new(int32_schema(), vec![Arc::new(Int32Array::from(vec![1, 2, 3]))])
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

    assert!(
        res.is_err(),
        "Append with a column type change (even a widening) must be rejected (got Ok — silent acceptance)"
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
    let b32 =
        RecordBatch::try_new(int32_schema(), vec![Arc::new(Int32Array::from(vec![1, 2, 3]))])
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
    assert!(new_snapshot > res.snapshot_id, "promote creates a newer snapshot");

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
    assert!(narrow.is_err(), "narrowing promote must be rejected");

    // int64 -> int64 is a no-op; reported as an error (no change), not applied.
    let noop = writer.promote_column_type(res.table_id, "id", "bigint");
    assert!(noop.is_err(), "no-op (same canonical type) promote must be rejected");
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
    let b32 =
        RecordBatch::try_new(int32_schema(), vec![Arc::new(Int32Array::from(vec![1, 2, 3]))])
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
    assert!(finish_result.is_ok(), "the racing Append should commit cleanly: {finish_result:?}");
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
    assert_eq!(got, vec![1.5, 2.5, 1.0e10], "f32 file up-casts to f64, values intact");
}

/// §14 A1/G3 — a FILTER on the promoted column spans the old (int32) and new
/// (int64) files: the predicate must apply correctly after cast-on-read. `id >
/// 4e9` keeps only the beyond-i32 row from the new file, drops the old [1,2,3].
#[tokio::test(flavor = "multi_thread")]
async fn filter_on_promoted_column_spans_old_and_new_files() {
    let (writer, temp_dir) = create_test_env().await;
    let object_store = create_object_store();

    let b32 =
        RecordBatch::try_new(int32_schema(), vec![Arc::new(Int32Array::from(vec![1, 2, 3]))])
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

    let b32 =
        RecordBatch::try_new(int32_schema(), vec![Arc::new(Int32Array::from(vec![1, 2, 3]))])
            .unwrap();
    let res = DuckLakeTableWriter::new(Arc::new(writer.clone()), Arc::clone(&object_store))
        .unwrap()
        .write_table("main", "t", &[b32])
        .await
        .unwrap();
    writer.promote_column_type(res.table_id, "id", "int64").unwrap();
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
    assert_eq!(cnt, 5, "COUNT(*) must span the old int32 file + new int64 file (3 + 2)");
}
