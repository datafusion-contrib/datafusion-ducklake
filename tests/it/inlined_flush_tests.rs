//! `ducklake_flush_inlined_data`: catalog-inlined rows moved to Parquet, on
//! every writable backend.
//!
//! Each scenario records what the table returns at every snapshot, flushes, and
//! then requires the same values at every snapshot again, plus no inlined rows
//! left and the catalog bookkeeping official DuckLake's flush produces.

#![cfg(feature = "write")]
#![allow(dead_code)]

use std::sync::Arc;

use arrow::array::{Int32Array, Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::util::pretty::pretty_format_batches;
use datafusion::prelude::SessionContext;
use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeTableWriter, DuckLakeWriteOptions, MetadataProvider, MetadataWriter,
    execute_ducklake_sql, register_ducklake_functions, register_ducklake_maintenance_functions,
};
use object_store::ObjectStore;
use object_store::local::LocalFileSystem;

/// The writer and reader of one backend's catalog.
pub(crate) struct Backend {
    pub(crate) writer: Arc<dyn MetadataWriter>,
    pub(crate) provider: Arc<dyn MetadataProvider>,
}

pub(crate) fn store() -> Arc<dyn ObjectStore> {
    Arc::new(LocalFileSystem::new())
}

fn inline_options(limit: usize) -> DuckLakeWriteOptions {
    DuckLakeWriteOptions::default().with_data_inlining_row_limit(limit)
}

fn table_writer(backend: &Backend, limit: usize) -> DuckLakeTableWriter {
    DuckLakeTableWriter::new(Arc::clone(&backend.writer), store())
        .unwrap()
        .with_options(&inline_options(limit))
}

fn two_columns(ids: &[i32], names: &[&str]) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, true),
            Field::new("name", DataType::Utf8, true),
        ])),
        vec![Arc::new(Int32Array::from(ids.to_vec())), Arc::new(StringArray::from(names.to_vec()))],
    )
    .unwrap()
}

fn three_columns(ids: &[i32], names: &[&str], extra: &[Option<i64>]) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, true),
            Field::new("name", DataType::Utf8, true),
            Field::new("extra", DataType::Int64, true),
        ])),
        vec![
            Arc::new(Int32Array::from(ids.to_vec())),
            Arc::new(StringArray::from(names.to_vec())),
            Arc::new(Int64Array::from(extra.to_vec())),
        ],
    )
    .unwrap()
}

/// A writable session over `backend` with the catalog registered as `lake`.
fn session(backend: &Backend) -> (SessionContext, DuckLakeCatalog) {
    let ctx = SessionContext::new();
    let catalog =
        DuckLakeCatalog::with_writer(Arc::clone(&backend.provider), Arc::clone(&backend.writer))
            .unwrap();
    DuckLakeCatalog::register(&ctx, "lake", catalog);
    register_ducklake_functions(&ctx, Arc::clone(&backend.provider));
    register_ducklake_maintenance_functions(&ctx);
    let handle =
        DuckLakeCatalog::with_writer(Arc::clone(&backend.provider), Arc::clone(&backend.writer))
            .unwrap();
    (ctx, handle)
}

pub(crate) async fn sql(ctx: &SessionContext, handle: &DuckLakeCatalog, sql: &str) -> String {
    let batches = execute_ducklake_sql(ctx, handle, sql)
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    pretty_format_batches(&batches).unwrap().to_string()
}

/// `query` over `main.<table>` at `snapshot`, rendered.
pub(crate) async fn rows_at(backend: &Backend, snapshot: i64, query: &str) -> String {
    let ctx = SessionContext::new();
    let catalog = DuckLakeCatalog::with_snapshot(Arc::clone(&backend.provider), snapshot).unwrap();
    ctx.register_catalog("tt", Arc::new(catalog));
    let batches = ctx.sql(query).await.unwrap().collect().await.unwrap();
    pretty_format_batches(&batches).unwrap().to_string()
}

/// Every snapshot from `first` to the head with what `query` returns at it.
pub(crate) async fn history(backend: &Backend, first: i64, query: &str) -> Vec<(i64, String)> {
    let head = backend.provider.get_current_snapshot().unwrap();
    let mut out = Vec::new();
    for snapshot in first..=head {
        out.push((snapshot, rows_at(backend, snapshot, query).await));
    }
    out
}

pub(crate) fn table_id(backend: &Backend, table: &str) -> i64 {
    let snapshot = backend.provider.get_current_snapshot().unwrap();
    let schema = backend
        .provider
        .get_schema_by_name("main", snapshot)
        .unwrap()
        .unwrap();
    backend
        .provider
        .get_table_by_name(schema.schema_id, table, snapshot)
        .unwrap()
        .unwrap()
        .table_id
}

pub(crate) fn inlined_row_count(backend: &Backend, table: &str) -> usize {
    let id = table_id(backend, table);
    let snapshot = backend.provider.get_current_snapshot().unwrap();
    backend
        .provider
        .get_inlined_data_for_flush(id, snapshot)
        .unwrap()
        .iter()
        .map(|data| data.rows.batch.num_rows())
        .sum()
}

fn last_changes(backend: &Backend) -> Option<String> {
    let snapshot = backend.provider.get_current_snapshot().unwrap();
    backend
        .provider
        .list_snapshot_changes()
        .unwrap()
        .into_iter()
        .find(|change| change.snapshot_id == snapshot)
        .and_then(|change| change.changes_made)
}

/// Every Parquet object under `dir`, sorted.
pub(crate) fn parquet_objects(dir: &std::path::Path) -> Vec<String> {
    let mut files = Vec::new();
    let mut pending = vec![dir.to_path_buf()];
    while let Some(dir) = pending.pop() {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                pending.push(path);
            } else if path.extension().is_some_and(|ext| ext == "parquet") {
                files.push(path.display().to_string());
            }
        }
    }
    files.sort();
    files
}

/// `count` rows `(id, name)` with ids from `first`.
fn many_rows(first: i32, count: i32) -> RecordBatch {
    let ids = (first..first + count).collect::<Vec<_>>();
    let names = ids
        .iter()
        .map(|id| format!("name-{id:06}"))
        .collect::<Vec<_>>();
    let names = names.iter().map(String::as_str).collect::<Vec<_>>();
    two_columns(&ids, &names)
}

/// Inlined inserts across two schema versions with an inlined delete in between;
/// flushed through `CALL`. Values must match at every snapshot afterwards.
pub(crate) async fn scenario_schema_evolution_and_deletes(backend: &Backend) {
    let writer = table_writer(backend, 10);
    let first = writer
        .write_table("main", "t", &[two_columns(&[1, 2, 3], &["a", "b", "c"])])
        .await
        .unwrap();
    assert_eq!(first.files_written, 0, "rows must be inlined");
    writer
        .append_table("main", "t", &[two_columns(&[4], &["d"])])
        .await
        .unwrap();
    let (ctx, handle) = session(backend);
    sql(&ctx, &handle, "DELETE FROM lake.main.t WHERE id = 2").await;
    writer
        .append_table(
            "main",
            "t",
            &[three_columns(&[5, 6], &["e", "f"], &[Some(50), None])],
        )
        .await
        .unwrap();
    sql(&ctx, &handle, "DELETE FROM lake.main.t WHERE id = 6").await;

    let query = "SELECT * FROM tt.main.t ORDER BY id";
    let before = history(backend, first.snapshot_id, query).await;
    let flushed_from = backend.provider.get_current_snapshot().unwrap();

    let result = sql(&ctx, &handle, "CALL ducklake_flush_inlined_data('lake')").await;
    assert!(
        result.contains("| main        | t          | 6            |"),
        "{result}"
    );
    let head = backend.provider.get_current_snapshot().unwrap();
    assert_eq!(head, flushed_from + 1, "one snapshot for the flush");
    assert_eq!(
        last_changes(backend).as_deref(),
        Some(format!("inline_flush:{}", table_id(backend, "t")).as_str())
    );
    assert_eq!(inlined_row_count(backend, "t"), 0);

    // Every earlier snapshot reads the same values, now from Parquet.
    for (snapshot, rows) in &before {
        assert_eq!(
            &rows_at(backend, *snapshot, query).await,
            rows,
            "snapshot {snapshot}"
        );
    }
    assert_eq!(
        &rows_at(backend, head, query).await,
        &before.last().unwrap().1
    );

    // Two partial files (one per schema version), each with a delete file.
    let files = backend
        .provider
        .get_table_files_for_select(table_id(backend, "t"), head)
        .unwrap();
    assert_eq!(files.len(), 2);
    for file in &files {
        assert!(file.partial_max.is_some());
        assert!(file.delete_file.is_some());
        assert!(file.row_id_start.is_some());
    }
    let mut begins = files
        .iter()
        .map(|file| (file.begin_snapshot.unwrap(), file.partial_max.unwrap()))
        .collect::<Vec<_>>();
    begins.sort_unstable();
    assert_eq!(begins[0].0, first.snapshot_id);

    // COUNT(*) from catalog counts agrees with the rows.
    let counted = rows_at(backend, head, "SELECT count(*) AS n FROM tt.main.t").await;
    assert!(counted.contains("| 4 |"), "{counted}");

    // Nothing left: a second flush commits nothing.
    let again = sql(
        &ctx,
        &handle,
        "SELECT * FROM ducklake_flush_inlined_data('lake')",
    )
    .await;
    assert!(!again.contains("| t "), "{again}");
    assert_eq!(backend.provider.get_current_snapshot().unwrap(), head);
}

/// More than one DataFusion batch of inlined rows, flushed through the table
/// function form, filtered to the table.
pub(crate) async fn scenario_many_rows(backend: &Backend) {
    let writer = table_writer(backend, 20_000);
    let ids: Vec<i32> = (0..10_000).collect();
    let names: Vec<String> = ids.iter().map(|id| format!("n{id}")).collect();
    let names: Vec<&str> = names.iter().map(String::as_str).collect();
    let written = writer
        .write_table("main", "big", &[two_columns(&ids, &names)])
        .await
        .unwrap();
    assert_eq!(written.files_written, 0);
    let (ctx, handle) = session(backend);
    sql(
        &ctx,
        &handle,
        "DELETE FROM lake.main.big WHERE id % 1000 = 7",
    )
    .await;
    let query =
        "SELECT count(*) AS n, sum(id) AS s, min(name) AS lo, max(name) AS hi FROM tt.main.big";
    let before = history(backend, written.snapshot_id, query).await;
    let result = sql(
        &ctx,
        &handle,
        "CALL ducklake_flush_inlined_data('lake', schema_name => 'main', table_name => 'big')",
    )
    .await;
    assert!(result.contains("| 10000 "), "{result}");
    assert_eq!(inlined_row_count(backend, "big"), 0);
    for (snapshot, rows) in &before {
        assert_eq!(
            &rows_at(backend, *snapshot, query).await,
            rows,
            "snapshot {snapshot}"
        );
    }
    let head = backend.provider.get_current_snapshot().unwrap();
    assert_eq!(
        &rows_at(backend, head, query).await,
        &before.last().unwrap().1
    );
}

/// A partitioned table flushes one file per partition, with partition values.
pub(crate) async fn scenario_partitioned(backend: &Backend) {
    let writer = table_writer(backend, 10);
    let created = writer
        .write_table("main", "p", &[two_columns(&[1], &["a"])])
        .await
        .unwrap();
    backend
        .writer
        .set_partition_spec(
            created.table_id,
            &[(
                "name".to_string(),
                datafusion_ducklake::PartitionTransform::Identity,
            )],
        )
        .unwrap();
    writer
        .append_table("main", "p", &[two_columns(&[2, 3, 4], &["x", "y", "x"])])
        .await
        .unwrap();
    let (ctx, handle) = session(backend);
    sql(&ctx, &handle, "DELETE FROM lake.main.p WHERE id = 3").await;
    let query = "SELECT * FROM tt.main.p ORDER BY id";
    let before = history(backend, created.snapshot_id, query).await;
    sql(
        &ctx,
        &handle,
        "CALL ducklake_flush_inlined_data('lake', table_name => 'p')",
    )
    .await;
    assert_eq!(inlined_row_count(backend, "p"), 0);
    for (snapshot, rows) in &before {
        assert_eq!(
            &rows_at(backend, *snapshot, query).await,
            rows,
            "snapshot {snapshot}"
        );
    }
    let head = backend.provider.get_current_snapshot().unwrap();
    let files = backend
        .provider
        .get_table_files_for_select(created.table_id, head)
        .unwrap();
    // One unpartitioned file for the rows written before the spec, and one per
    // partition (x, y) for the rest.
    assert_eq!(files.len(), 3, "{files:?}");
    let mut directories = files
        .iter()
        .map(|file| file.file.path.split('/').next().unwrap().to_string())
        .filter(|dir| dir.starts_with("name="))
        .collect::<Vec<_>>();
    directories.sort();
    assert_eq!(directories, vec!["name=x", "name=y"], "{files:?}");
    let pruned = rows_at(
        backend,
        head,
        "SELECT id FROM tt.main.p WHERE name = 'x' ORDER BY id",
    )
    .await;
    assert!(
        pruned.contains("| 2  |") && pruned.contains("| 4  |"),
        "{pruned}"
    );
}

/// The flush rolls a partition's rows over into a new file at the table's
/// `target_file_size`, and each file's delete positions count from its own
/// first row.
pub(crate) async fn scenario_rollover(backend: &Backend) {
    let created = table_writer(backend, 20_000)
        .write_table("main", "r", &[many_rows(0, 10_000)])
        .await
        .unwrap();
    assert_eq!(created.files_written, 0);
    for (key, value) in [("target_file_size", "4KB"), ("parquet_row_group_size", "500")] {
        backend
            .writer
            .set_table_setting(created.table_id, key, value)
            .unwrap();
    }
    let (ctx, handle) = session(backend);
    // Deletions across the whole range land in several files.
    sql(
        &ctx,
        &handle,
        "DELETE FROM lake.main.r WHERE id % 997 = 0 OR id IN (2047, 2048, 9999)",
    )
    .await;
    let query = "SELECT count(*) AS n, sum(id) AS s, min(name) AS lo, max(name) AS hi \
                 FROM tt.main.r";
    let before = history(backend, created.snapshot_id, query).await;
    sql(&ctx, &handle, "CALL ducklake_flush_inlined_data('lake')").await;
    assert_eq!(inlined_row_count(backend, "r"), 0);
    let head = backend.provider.get_current_snapshot().unwrap();
    let files = backend
        .provider
        .get_table_files_for_select(created.table_id, head)
        .unwrap();
    assert!(files.len() >= 3, "{} files", files.len());
    assert_eq!(
        files
            .iter()
            .map(|file| file.max_row_count.unwrap())
            .sum::<i64>(),
        10_000
    );
    assert!(
        files
            .iter()
            .filter(|file| file.delete_file_id.is_some())
            .count()
            >= 3,
        "{files:?}"
    );
    for (snapshot, rows) in &before {
        assert_eq!(
            &rows_at(backend, *snapshot, query).await,
            rows,
            "snapshot {snapshot}"
        );
    }
    let rows = rows_at(
        backend,
        head,
        "SELECT id FROM tt.main.r WHERE id IN (2046, 2047, 2048, 2049, 9998, 9999) ORDER BY id",
    )
    .await;
    assert!(
        rows.contains("| 2046 |") && rows.contains("| 2049 |"),
        "{rows}"
    );
    assert!(
        !rows.contains("| 2047 |") && !rows.contains("| 2048 |"),
        "{rows}"
    );
    assert!(
        rows.contains("| 9998 |") && !rows.contains("| 9999 |"),
        "{rows}"
    );
}

/// A partitioned flush writes each partition's rows to one file whatever
/// the table's `target_file_size`, as official's flush `COPY` writes a
/// partitioned table without rotating its files.
pub(crate) async fn scenario_partitioned_past_target_file_size(backend: &Backend) {
    let writer = table_writer(backend, 50_000);
    let created = writer
        .write_table("main", "q", &[two_columns(&[-1], &["a"])])
        .await
        .unwrap();
    backend
        .writer
        .set_partition_spec(
            created.table_id,
            &[(
                "name".to_string(),
                datafusion_ducklake::PartitionTransform::Identity,
            )],
        )
        .unwrap();
    for (key, value) in [("target_file_size", "4KB"), ("parquet_row_group_size", "500")] {
        backend
            .writer
            .set_table_setting(created.table_id, key, value)
            .unwrap();
    }
    // Values that do not compress, so each partition's rows are far past
    // the target size.
    let ids = (0..40_000u32)
        .map(|i| i.wrapping_mul(2_654_435_761) as i32)
        .collect::<Vec<_>>();
    let names = (0..40_000)
        .map(|i| {
            if i % 2 == 0 {
                "a"
            } else {
                "b"
            }
        })
        .collect::<Vec<_>>();
    let appended = writer
        .append_table("main", "q", &[two_columns(&ids, &names)])
        .await
        .unwrap();
    assert_eq!(appended.files_written, 0, "rows must be inlined");
    let (ctx, handle) = session(backend);
    sql(&ctx, &handle, "CALL ducklake_flush_inlined_data('lake')").await;
    let head = backend.provider.get_current_snapshot().unwrap();
    let files = backend
        .provider
        .get_table_files_for_select(created.table_id, head)
        .unwrap();
    let partitioned = files
        .iter()
        .filter(|file| file.partition_id.is_some())
        .collect::<Vec<_>>();
    assert_eq!(partitioned.len(), 2, "{files:?}");
    assert_eq!(
        partitioned
            .iter()
            .map(|file| file.max_row_count.unwrap())
            .sum::<i64>(),
        40_000
    );
}

#[cfg(feature = "write-sqlite")]
mod sqlite {
    use super::*;
    use datafusion_ducklake::{SqliteMetadataProvider, SqliteMetadataWriter};
    use tempfile::TempDir;

    pub(crate) async fn backend(dir: &TempDir) -> Backend {
        let url = format!(
            "sqlite:{}?mode=rwc",
            dir.path().join("catalog.db").display()
        );
        let data = dir.path().join("data");
        std::fs::create_dir_all(&data).unwrap();
        let writer = SqliteMetadataWriter::new_with_init(&url).await.unwrap();
        writer.set_data_path(data.to_str().unwrap()).unwrap();
        let provider = SqliteMetadataProvider::new(&url).await.unwrap();
        Backend {
            writer: Arc::new(writer),
            provider: Arc::new(provider),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn flush_preserves_values_across_schema_versions_and_deletes() {
        let dir = TempDir::new().unwrap();
        let backend = backend(&dir).await;
        scenario_schema_evolution_and_deletes(&backend).await;
    }

    /// A row a delete file deletes and a later inlined deletion records again
    /// (official commits two concurrent `DELETE`s of one row so; the inlined
    /// deletion is written here directly): the flush's merged delete file
    /// keeps the earlier snapshot, so the row stays deleted at the delete
    /// file's snapshot.
    #[tokio::test(flavor = "multi_thread")]
    async fn flush_keeps_the_earlier_snapshot_of_a_position_deleted_twice() {
        let dir = TempDir::new().unwrap();
        let backend = backend(&dir).await;
        let ids: Vec<i32> = (0..100).collect();
        let names: Vec<&str> = ids.iter().map(|_| "x").collect();
        table_writer(&backend, 0)
            .write_table("main", "t", &[two_columns(&ids, &names)])
            .await
            .unwrap();
        backend
            .writer
            .set_global_setting("data_inlining_row_limit", "5")
            .unwrap();
        let (ctx, handle) = session(&backend);
        sql(&ctx, &handle, "DELETE FROM lake.main.t WHERE id < 20").await;
        let deleted_at = backend.provider.get_current_snapshot().unwrap();

        let pool = sqlx::SqlitePool::connect(&format!(
            "sqlite:{}",
            dir.path().join("catalog.db").display()
        ))
        .await
        .unwrap();
        let (table_id, data_file_id): (i64, i64) = sqlx::query_as(
            "SELECT table_id, data_file_id FROM ducklake_data_file WHERE end_snapshot IS NULL",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        let again = deleted_at + 1;
        for statement in [
            format!(
                "INSERT INTO ducklake_snapshot (snapshot_id, snapshot_time, schema_version) \
                 SELECT {again}, CURRENT_TIMESTAMP, schema_version FROM ducklake_snapshot \
                 WHERE snapshot_id = {deleted_at}"
            ),
            format!(
                "INSERT INTO ducklake_snapshot_changes (snapshot_id, changes_made) \
                 VALUES ({again}, 'inlined_delete:{table_id}')"
            ),
            format!(
                "CREATE TABLE IF NOT EXISTS ducklake_inlined_delete_{table_id} \
                 (file_id BIGINT, row_id BIGINT, begin_snapshot BIGINT)"
            ),
            format!(
                "INSERT INTO ducklake_inlined_delete_{table_id} VALUES ({data_file_id}, 5, {again})"
            ),
        ] {
            sqlx::query(sqlx::AssertSqlSafe(statement))
                .execute(&pool)
                .await
                .unwrap();
        }
        pool.close().await;

        let query = "SELECT count(*), count(*) FILTER (WHERE id = 5) FROM tt.main.t";
        let expected = rows_at(&backend, deleted_at, query).await;
        assert!(expected.contains("| 80 "), "{expected}");
        sql(&ctx, &handle, "CALL ducklake_flush_inlined_data('lake')").await;
        assert_eq!(rows_at(&backend, deleted_at, query).await, expected);
        assert_eq!(rows_at(&backend, again, query).await, expected);
    }

    /// The catalog `ducklake_flush_inlined_data` names must be the one the
    /// statement runs against: official errors on a catalog it does not know.
    #[tokio::test(flavor = "multi_thread")]
    async fn flush_rejects_another_catalog_name() {
        let dir = TempDir::new().unwrap();
        let lake = backend(&dir).await;
        let (ctx, handle) = session(&lake);
        let error =
            execute_ducklake_sql(&ctx, &handle, "CALL ducklake_flush_inlined_data('other')")
                .await
                .unwrap_err()
                .to_string();
        assert!(
            error.contains("Failed to find attached database \"other\""),
            "{error}"
        );

        // `lake` names a catalog, but not the one the statement runs against.
        let unrelated_dir = TempDir::new().unwrap();
        let unrelated = backend(&unrelated_dir).await;
        let (_, unrelated_handle) = session(&unrelated);
        let error = execute_ducklake_sql(
            &ctx,
            &unrelated_handle,
            "CALL ducklake_flush_inlined_data('lake')",
        )
        .await
        .unwrap_err()
        .to_string();
        assert!(error.contains("is not the DuckLake catalog"), "{error}");
        sql(&ctx, &handle, "CALL ducklake_flush_inlined_data('lake')").await;
    }

    /// `CALL` resolves as DuckDB does: the catalog name in any case, the
    /// function under `main`, `system` or `system.main` only, and a missing
    /// schema or table reported in official's words.
    #[tokio::test(flavor = "multi_thread")]
    async fn call_resolves_names_as_duckdb_does() {
        let dir = TempDir::new().unwrap();
        let lake = backend(&dir).await;
        table_writer(&lake, 10)
            .append_table("main", "t", &[two_columns(&[1], &["a"])])
            .await
            .unwrap();
        let (ctx, handle) = session(&lake);
        for call in [
            "CALL ducklake_flush_inlined_data('LAKE')",
            "CALL main.ducklake_flush_inlined_data('lake')",
            "CALL SYSTEM.ducklake_flush_inlined_data('lake')",
            "CALL system.main.ducklake_flush_inlined_data('lake')",
        ] {
            sql(&ctx, &handle, call).await;
        }
        for (call, expected) in [
            (
                "CALL lake.ducklake_flush_inlined_data('lake')",
                "Table Function with name \"ducklake_flush_inlined_data\" is not in the catalog, \
                 but it exists in the ducklake extension.",
            ),
            (
                "CALL lake.main.ducklake_flush_inlined_data('lake')",
                "Table Function with name \"ducklake_flush_inlined_data\" is not in the catalog",
            ),
            (
                "CALL ducklake_flush_inlined_data('lake', schema_name => 'nope')",
                "Schema \"nope\" not found in DuckLakeCatalog \"lake\"",
            ),
            (
                "CALL ducklake_flush_inlined_data('lake', table_name => 'nope')",
                "Table with name nope does not exist!",
            ),
        ] {
            let error = execute_ducklake_sql(&ctx, &handle, call)
                .await
                .map(|_| ())
                .unwrap_err()
                .to_string();
            assert!(error.contains(expected), "{call}: {error}");
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn flush_writes_many_rows() {
        let dir = TempDir::new().unwrap();
        let backend = backend(&dir).await;
        scenario_many_rows(&backend).await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn flush_writes_one_file_per_partition() {
        let dir = TempDir::new().unwrap();
        let backend = backend(&dir).await;
        scenario_partitioned(&backend).await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn flush_rolls_files_over_at_target_file_size() {
        let dir = TempDir::new().unwrap();
        scenario_rollover(&backend(&dir).await).await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn partitioned_flush_writes_one_file_per_partition_past_target_file_size() {
        let dir = TempDir::new().unwrap();
        scenario_partitioned_past_target_file_size(&backend(&dir).await).await;
    }

    async fn pool(dir: &TempDir) -> sqlx::SqlitePool {
        sqlx::SqlitePool::connect(&format!(
            "sqlite:{}",
            dir.path().join("catalog.db").display()
        ))
        .await
        .unwrap()
    }

    /// A change-feed window spanning flushed history reports what it reported
    /// before the flush: each update as its pre- and postimage, once. Official
    /// DuckLake (DuckDB 1.5.5) diverges here — after the flush its window repeats
    /// the first update's rows and reports the second update as an insert.
    #[tokio::test(flavor = "multi_thread")]
    async fn change_feed_across_a_flush_is_unchanged() {
        let dir = TempDir::new().unwrap();
        let backend = backend(&dir).await;
        backend
            .writer
            .set_global_setting("data_inlining_row_limit", "5")
            .unwrap();
        table_writer(&backend, 5)
            .append_table("main", "t", &[two_columns(&[1], &["a"])])
            .await
            .unwrap();
        let (ctx, handle) = session(&backend);
        sql(&ctx, &handle, "UPDATE lake.main.t SET name = 'b'").await;
        let first = backend.provider.get_current_snapshot().unwrap();
        sql(&ctx, &handle, "UPDATE lake.main.t SET name = 'c'").await;
        let second = backend.provider.get_current_snapshot().unwrap();
        assert_eq!(inlined_row_count(&backend, "t"), 3);
        let changes = format!(
            "SELECT snapshot_id - {first} AS snapshot, rowid, change_type, id, name \
             FROM ducklake_table_changes('main.t', {first}, {second}) \
             ORDER BY snapshot_id, change_type"
        );
        let feed = |ctx: &SessionContext| {
            let ctx = ctx.clone();
            let changes = changes.clone();
            async move {
                let batches = ctx.sql(&changes).await.unwrap().collect().await.unwrap();
                pretty_format_batches(&batches).unwrap().to_string()
            }
        };
        let before = feed(&ctx).await;
        let expected = "\
+----------+-------+------------------+----+------+
| snapshot | rowid | change_type      | id | name |
+----------+-------+------------------+----+------+
| 0        | 0     | update_postimage | 1  | b    |
| 0        | 0     | update_preimage  | 1  | a    |
| 1        | 0     | update_postimage | 1  | c    |
| 1        | 0     | update_preimage  | 1  | b    |
+----------+-------+------------------+----+------+";
        assert_eq!(before, expected);
        sql(&ctx, &handle, "CALL ducklake_flush_inlined_data('lake')").await;
        assert_eq!(inlined_row_count(&backend, "t"), 0);
        assert_eq!(feed(&ctx).await, expected);
    }

    /// Official drops the emptied tables of superseded schema versions, keeps
    /// the latest one, and records `inline_flush` only.
    #[tokio::test(flavor = "multi_thread")]
    async fn flush_drops_emptied_superseded_inlined_tables() {
        let dir = TempDir::new().unwrap();
        let backend = backend(&dir).await;
        scenario_schema_evolution_and_deletes(&backend).await;
        let pool = pool(&dir).await;
        let registered: Vec<(String, i64)> =
            sqlx::query_as("SELECT table_name, schema_version FROM ducklake_inlined_data_tables")
                .fetch_all(&pool)
                .await
                .unwrap();
        assert_eq!(registered.len(), 1, "{registered:?}");
        let physical: Vec<String> = sqlx::query_scalar(
            "SELECT name FROM sqlite_master
             WHERE type = 'table' AND name LIKE 'ducklake_inlined_data_%'",
        )
        .fetch_all(&pool)
        .await
        .unwrap();
        assert!(
            physical.contains(&registered[0].0) && physical.len() == 2,
            "{physical:?}"
        );
        let (record_count, next_row_id): (i64, i64) =
            sqlx::query_as("SELECT record_count, next_row_id FROM ducklake_table_stats")
                .fetch_one(&pool)
                .await
                .unwrap();
        // The flush counts no rows twice and draws no row ids.
        assert_eq!((record_count, next_row_id), (6, 6));
    }

    /// The table's sort order applies to the flushed file, then row id.
    #[tokio::test(flavor = "multi_thread")]
    async fn flush_writes_rows_in_sort_order() {
        use datafusion_ducklake::{NullOrder, SortDirection, SortField};
        let dir = TempDir::new().unwrap();
        let backend = backend(&dir).await;
        let writer = table_writer(&backend, 10);
        let created = writer
            .write_table("main", "s", &[two_columns(&[1, 2, 3], &["b", "c", "a"])])
            .await
            .unwrap();
        backend
            .writer
            .set_sort_spec(
                created.table_id,
                &[SortField {
                    sort_key_index: 0,
                    expression: "name".to_string(),
                    dialect: "duckdb".to_string(),
                    direction: SortDirection::Desc,
                    null_order: NullOrder::NullsLast,
                }],
            )
            .unwrap();
        let (ctx, handle) = session(&backend);
        sql(&ctx, &handle, "CALL ducklake_flush_inlined_data('lake')").await;
        let head = backend.provider.get_current_snapshot().unwrap();
        let files = backend
            .provider
            .get_table_files_for_select(created.table_id, head)
            .unwrap();
        assert_eq!(files.len(), 1);
        let path = dir
            .path()
            .join("data")
            .join("main")
            .join("s")
            .join(&files[0].file.path);
        let reader = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(
            std::fs::File::open(path).unwrap(),
        )
        .unwrap()
        .build()
        .unwrap();
        let batch = reader.map(Result::unwrap).next().unwrap();
        let names =
            arrow::compute::cast(batch.column_by_name("name").unwrap(), &DataType::Utf8).unwrap();
        let names = names
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|name| name.unwrap().to_string())
            .collect::<Vec<_>>();
        let row_ids = batch
            .column_by_name("_ducklake_internal_row_id")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .values()
            .to_vec();
        assert_eq!(names, vec!["c", "b", "a"]);
        assert_eq!(row_ids, vec![1, 0, 2]);
    }

    /// Inlined deletions of Parquet rows move into one delete file per data file,
    /// merged with the file's live delete file, which is removed and scheduled.
    #[tokio::test(flavor = "multi_thread")]
    async fn flush_moves_inlined_file_deletions_into_delete_files() {
        let dir = TempDir::new().unwrap();
        let backend = backend(&dir).await;
        let parquet_writer = table_writer(&backend, 0);
        let first = parquet_writer
            .write_table(
                "main",
                "f",
                &[two_columns(&[1, 2, 3, 4], &["a", "b", "c", "d"])],
            )
            .await
            .unwrap();
        let second = parquet_writer
            .append_table("main", "f", &[two_columns(&[5, 6], &["e", "f"])])
            .await
            .unwrap();
        let (ctx, handle) = session(&backend);
        // A positional delete file on the first file.
        backend
            .writer
            .set_global_setting("data_inlining_row_limit", "0")
            .unwrap();
        sql(&ctx, &handle, "DELETE FROM lake.main.f WHERE id = 1").await;
        let s_deleted = backend.provider.get_current_snapshot().unwrap();
        let files = backend
            .provider
            .get_table_files_for_select(first.table_id, s_deleted)
            .unwrap();
        let file_of = |count: i64| {
            files
                .iter()
                .find(|file| file.max_row_count == Some(count))
                .unwrap()
                .data_file_id
        };
        let (big, small) = (file_of(4), file_of(2));
        let s_a = backend.writer.create_snapshot().unwrap();
        let s_b = backend.writer.create_snapshot().unwrap();
        // DuckDB's `ducklake_inlined_delete_<table_id>` layout.
        let pool = pool(&dir).await;
        let table = format!("ducklake_inlined_delete_{}", first.table_id);
        for statement in [
            format!("CREATE TABLE {table} (file_id BIGINT, row_id BIGINT, begin_snapshot BIGINT)"),
            format!(
                "INSERT INTO {table} VALUES ({big}, 2, {s_a}), ({small}, 0, {s_a}), ({small}, 1, {s_b})"
            ),
        ] {
            sqlx::query(sqlx::AssertSqlSafe(statement))
                .execute(&pool)
                .await
                .unwrap();
        }
        let query = "SELECT * FROM tt.main.f ORDER BY id";
        let before = history(&backend, first.snapshot_id, query).await;
        assert!(!before.last().unwrap().1.contains("| 5 "), "{before:?}");

        let result = sql(&ctx, &handle, "CALL ducklake_flush_inlined_data('lake')").await;
        // No inlined rows were flushed, so the result has no row.
        assert!(!result.contains("| f "), "{result}");
        assert_eq!(
            last_changes(&backend).as_deref(),
            Some(format!("deleted_from_table:{}", first.table_id).as_str())
        );
        for (snapshot, rows) in &before {
            assert_eq!(
                &rows_at(&backend, *snapshot, query).await,
                rows,
                "snapshot {snapshot}"
            );
        }
        let left: i64 =
            sqlx::query_scalar(sqlx::AssertSqlSafe(format!("SELECT COUNT(*) FROM {table}")))
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(left, 0);
        let deletes: Vec<(i64, i64, i64, Option<i64>)> = sqlx::query_as(
            "SELECT data_file_id, delete_count, begin_snapshot, partial_max
             FROM ducklake_delete_file ORDER BY data_file_id",
        )
        .fetch_all(&pool)
        .await
        .unwrap();
        assert_eq!(
            deletes,
            vec![(big, 2, s_deleted, Some(s_a)), (small, 2, s_a, Some(s_b))]
        );
        let scheduled: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM ducklake_files_scheduled_for_deletion")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(scheduled, 1, "the replaced delete file is scheduled");
        let _ = second;
    }

    /// A flush whose base snapshot predates an inlined delete aborts and commits
    /// nothing.
    #[tokio::test(flavor = "multi_thread")]
    async fn stale_flush_conflicts() {
        let dir = TempDir::new().unwrap();
        let backend = backend(&dir).await;
        let written = table_writer(&backend, 10)
            .write_table("main", "t", &[two_columns(&[1, 2], &["a", "b"])])
            .await
            .unwrap();
        let (ctx, handle) = session(&backend);
        sql(&ctx, &handle, "DELETE FROM lake.main.t WHERE id = 1").await;
        let head = backend.provider.get_current_snapshot().unwrap();
        let error = backend
            .writer
            .commit_inlined_flush(
                written.snapshot_id,
                &[datafusion_ducklake::InlinedFlushTable {
                    table_id: written.table_id,
                    inlined_tables: vec![format!(
                        "ducklake_inlined_data_{}_{}",
                        written.table_id, 1
                    )],
                    ..Default::default()
                }],
            )
            .unwrap_err();
        assert!(
            matches!(error, datafusion_ducklake::DuckLakeError::Conflict(_)),
            "{error}"
        );
        assert_eq!(backend.provider.get_current_snapshot().unwrap(), head);
        assert_eq!(inlined_row_count(&backend, "t"), 2);
    }

    /// Arrow cannot write `Interval(MonthDayNano)` to Parquet: the flush fails
    /// with a typed error and the rows stay inlined and readable.
    #[tokio::test(flavor = "multi_thread")]
    async fn interval_rows_cannot_be_flushed() {
        use arrow::array::IntervalMonthDayNanoArray;
        use arrow::datatypes::{IntervalMonthDayNano, IntervalUnit};
        let dir = TempDir::new().unwrap();
        let backend = backend(&dir).await;
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("id", DataType::Int32, true),
                Field::new("span", DataType::Interval(IntervalUnit::MonthDayNano), true),
            ])),
            vec![
                Arc::new(Int32Array::from(vec![1])),
                Arc::new(IntervalMonthDayNanoArray::from(vec![
                    IntervalMonthDayNano::new(1, 2, 3_000),
                ])),
            ],
        )
        .unwrap();
        let written = table_writer(&backend, 10)
            .write_table("main", "iv", &[batch])
            .await
            .unwrap();
        assert_eq!(written.files_written, 0);
        let (ctx, handle) = session(&backend);
        let error = execute_ducklake_sql(&ctx, &handle, "CALL ducklake_flush_inlined_data('lake')")
            .await
            .unwrap_err();
        assert!(error.to_string().contains("INTERVAL"), "{error}");
        assert_eq!(inlined_row_count(&backend, "iv"), 1);
    }

    /// DuckDB's DuckLake extension reads a catalog this crate flushed: the same
    /// values at every snapshot, through the partial files and their delete files.
    #[tokio::test(flavor = "multi_thread")]
    async fn duckdb_reads_a_flushed_catalog_at_every_snapshot() {
        let dir = TempDir::new().unwrap();
        let backend = backend(&dir).await;
        scenario_schema_evolution_and_deletes(&backend).await;
        let head = backend.provider.get_current_snapshot().unwrap();
        let mut expected = String::new();
        for snapshot in 1..=head {
            let ctx = SessionContext::new();
            let catalog =
                DuckLakeCatalog::with_snapshot(Arc::clone(&backend.provider), snapshot).unwrap();
            ctx.register_catalog("tt", Arc::new(catalog));
            let Ok(frame) = ctx.sql("SELECT id, name FROM tt.main.t ORDER BY id").await else {
                continue;
            };
            for batch in frame.collect().await.unwrap() {
                let ids = batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .unwrap();
                let names = arrow::compute::cast(batch.column(1), &DataType::Utf8).unwrap();
                let names = names.as_any().downcast_ref::<StringArray>().unwrap();
                for row in 0..batch.num_rows() {
                    expected.push_str(&format!(
                        "{snapshot},{},{}\n",
                        ids.value(row),
                        names.value(row)
                    ));
                }
            }
        }

        // The columns and tables DuckDB requires that this crate's catalogs omit.
        let pool = pool(&dir).await;
        for statement in [
            "ALTER TABLE ducklake_snapshot ADD COLUMN next_catalog_id BIGINT NOT NULL DEFAULT 1000",
            "ALTER TABLE ducklake_snapshot ADD COLUMN next_file_id BIGINT NOT NULL DEFAULT 1000",
            "ALTER TABLE ducklake_schema ADD COLUMN schema_uuid VARCHAR",
            "UPDATE ducklake_schema SET schema_uuid = '00000000-0000-0000-0000-000000000001'",
            "ALTER TABLE ducklake_table ADD COLUMN table_uuid VARCHAR",
            "UPDATE ducklake_table SET table_uuid = '00000000-0000-0000-0000-000000000002'",
            "ALTER TABLE ducklake_data_file ADD COLUMN file_order BIGINT",
            "ALTER TABLE ducklake_data_file ADD COLUMN file_format VARCHAR",
            "ALTER TABLE ducklake_delete_file ADD COLUMN format VARCHAR",
            "CREATE TABLE ducklake_column_mapping (mapping_id BIGINT, table_id BIGINT, type VARCHAR)",
            "CREATE TABLE ducklake_file_variant_stats (data_file_id BIGINT, table_id BIGINT, column_id BIGINT, variant_path VARCHAR, shredded_type VARCHAR, column_size_bytes BIGINT, value_count BIGINT, null_count BIGINT, min_value VARCHAR, max_value VARCHAR, contains_nan BOOLEAN, extra_stats VARCHAR)",
            "CREATE TABLE ducklake_macro (schema_id BIGINT, macro_id BIGINT, macro_name VARCHAR, begin_snapshot BIGINT, end_snapshot BIGINT)",
            "CREATE TABLE ducklake_macro_impl (macro_id BIGINT, impl_id BIGINT, dialect VARCHAR, sql VARCHAR, type VARCHAR)",
            "CREATE TABLE ducklake_macro_parameters (macro_id BIGINT, impl_id BIGINT, column_id BIGINT, parameter_name VARCHAR, parameter_type VARCHAR, default_value VARCHAR, default_value_type VARCHAR)",
            "CREATE TABLE ducklake_name_mapping (mapping_id BIGINT, column_id BIGINT, source_name VARCHAR, target_field_id BIGINT, parent_column BIGINT, is_partition BOOLEAN)",
            "CREATE TABLE IF NOT EXISTS ducklake_view (view_id BIGINT, view_uuid VARCHAR, begin_snapshot BIGINT, end_snapshot BIGINT, schema_id BIGINT, view_name VARCHAR, dialect VARCHAR, sql VARCHAR, column_aliases VARCHAR)",
        ] {
            sqlx::query(statement).execute(&pool).await.unwrap();
        }
        pool.close().await;

        let mut script = format!(
            "LOAD ducklake; LOAD sqlite; ATTACH 'ducklake:sqlite:{}' AS lake;",
            dir.path().join("catalog.db").display()
        );
        for snapshot in 1..=head {
            script.push_str(&format!(
                " SELECT {snapshot}, id, name FROM lake.main.t AT (VERSION => {snapshot}) ORDER BY id;"
            ));
        }
        let output = std::process::Command::new("duckdb")
            .args(["-csv", "-noheader", ":memory:", "-c", &script])
            .output()
            .unwrap();
        let stdout = String::from_utf8(output.stdout).unwrap();
        assert!(
            output.status.success(),
            "DuckDB read failed: {}\n{stdout}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(expected.lines().count() > 20, "{expected}");
        assert_eq!(stdout, expected);
    }

    /// `auto_compact = false` exempts a table, as in official.
    #[tokio::test(flavor = "multi_thread")]
    async fn auto_compact_false_skips_the_table() {
        let dir = TempDir::new().unwrap();
        let backend = backend(&dir).await;
        let written = table_writer(&backend, 10)
            .write_table("main", "t", &[two_columns(&[1], &["a"])])
            .await
            .unwrap();
        backend
            .writer
            .set_table_setting(written.table_id, "auto_compact", "false")
            .unwrap();
        let (ctx, handle) = session(&backend);
        sql(&ctx, &handle, "CALL ducklake_flush_inlined_data('lake')").await;
        assert_eq!(inlined_row_count(&backend, "t"), 1);
    }

    /// `checkpoint` flushes, then compacts what the flush and earlier writes
    /// left, each in its own snapshot.
    #[tokio::test(flavor = "multi_thread")]
    async fn checkpoint_flushes_then_merges() {
        let dir = TempDir::new().unwrap();
        let backend = backend(&dir).await;
        let parquet = table_writer(&backend, 0);
        let first = parquet
            .write_table("main", "t", &[two_columns(&[1], &["a"])])
            .await
            .unwrap();
        parquet
            .append_table("main", "t", &[two_columns(&[2], &["b"])])
            .await
            .unwrap();
        table_writer(&backend, 10)
            .append_table("main", "t", &[two_columns(&[3, 4], &["c", "d"])])
            .await
            .unwrap();
        let query = "SELECT * FROM tt.main.t ORDER BY id";
        let before = history(&backend, first.snapshot_id, query).await;
        let (ctx, catalog) = session(&backend);
        let result = catalog.checkpoint(&ctx.state()).await.unwrap();
        assert_eq!(result.flushed.len(), 1);
        assert_eq!(result.flushed[0].rows_flushed, 2);
        assert_eq!(result.merged.len(), 1, "{result:?}");
        assert_eq!(result.merged[0].2.files_processed, 3);
        assert_eq!(inlined_row_count(&backend, "t"), 0);
        for (snapshot, rows) in &before {
            assert_eq!(
                &rows_at(&backend, *snapshot, query).await,
                rows,
                "snapshot {snapshot}"
            );
        }
        let head = backend.provider.get_current_snapshot().unwrap();
        assert_eq!(
            backend
                .provider
                .get_table_files_for_select(first.table_id, head)
                .unwrap()
                .len(),
            1
        );
    }

    /// The flush table function is registered only on request: a session with
    /// the read functions alone does not find it. It names the catalog in any
    /// case.
    #[tokio::test(flavor = "multi_thread")]
    async fn flush_table_function_is_opt_in() {
        let dir = TempDir::new().unwrap();
        let backend = backend(&dir).await;
        table_writer(&backend, 10)
            .write_table("main", "t", &[two_columns(&[1], &["a"])])
            .await
            .unwrap();
        let ctx = SessionContext::new();
        let catalog = DuckLakeCatalog::with_writer(
            Arc::clone(&backend.provider),
            Arc::clone(&backend.writer),
        )
        .unwrap();
        DuckLakeCatalog::register(&ctx, "lake", catalog);
        register_ducklake_functions(&ctx, Arc::clone(&backend.provider));
        let error = ctx
            .sql("SELECT * FROM ducklake_flush_inlined_data('lake')")
            .await
            .unwrap_err();
        assert!(error.to_string().contains("not found"), "{error}");
        assert_eq!(inlined_row_count(&backend, "t"), 1);

        register_ducklake_maintenance_functions(&ctx);
        // The catalog name matches in any case, as DuckDB matches it.
        let batches = ctx
            .sql("SELECT * FROM ducklake_flush_inlined_data('LAKE')")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 1);
        assert_eq!(inlined_row_count(&backend, "t"), 0);
    }

    /// A catalog whose `encrypted` option is `true` gets encrypted Parquet from
    /// official's flush; this crate writes none, so it refuses to flush there,
    /// before writing anything, and `checkpoint` skips the flush.
    #[tokio::test(flavor = "multi_thread")]
    async fn encrypted_catalog_refuses_flush_and_checkpoint() {
        let dir = TempDir::new().unwrap();
        let backend = backend(&dir).await;
        table_writer(&backend, 10)
            .write_table("main", "t", &[two_columns(&[1, 2], &["a", "b"])])
            .await
            .unwrap();
        let pool = pool(&dir).await;
        sqlx::query(
            "INSERT INTO ducklake_metadata (key, value, scope) VALUES ('encrypted', 'true', NULL)",
        )
        .execute(&pool)
        .await
        .unwrap();
        pool.close().await;
        let head = backend.provider.get_current_snapshot().unwrap();
        let (ctx, handle) = session(&backend);
        let error = handle
            .flush_inlined_data(&ctx.state(), None, None)
            .await
            .unwrap_err();
        assert!(
            matches!(error, datafusion_ducklake::DuckLakeError::Unsupported(_)),
            "{error}"
        );
        // Checkpoint skips the phases that write Parquet.
        let result = handle.checkpoint(&ctx.state()).await.unwrap();
        assert!(result.flushed.is_empty(), "{result:?}");
        let error = execute_ducklake_sql(&ctx, &handle, "CALL ducklake_flush_inlined_data('lake')")
            .await
            .unwrap_err();
        assert!(error.to_string().contains("encrypted"), "{error}");
        assert_eq!(backend.provider.get_current_snapshot().unwrap(), head);
        assert_eq!(inlined_row_count(&backend, "t"), 2);
        assert!(parquet_objects(&dir.path().join("data")).is_empty());
    }

    /// `CALL ducklake_flush_inlined_data` through `execute_ducklake_sql` flushes
    /// the catalog it is given when the statement runs, once, and returns
    /// official's result columns with `rows_flushed` as `HUGEINT`'s Arrow type.
    #[tokio::test(flavor = "multi_thread")]
    async fn call_flushes_once_on_the_given_catalog() {
        let dir = TempDir::new().unwrap();
        let backend = backend(&dir).await;
        table_writer(&backend, 10)
            .write_table("main", "t", &[two_columns(&[1, 2], &["a", "b"])])
            .await
            .unwrap();
        // The call runs on the handle, which the name it gives must denote.
        let ctx = SessionContext::new();
        let handle = DuckLakeCatalog::with_writer(
            Arc::clone(&backend.provider),
            Arc::clone(&backend.writer),
        )
        .unwrap();
        DuckLakeCatalog::register(
            &ctx,
            "lake",
            DuckLakeCatalog::with_writer(
                Arc::clone(&backend.provider),
                Arc::clone(&backend.writer),
            )
            .unwrap(),
        );
        let head = backend.provider.get_current_snapshot().unwrap();
        let frame = execute_ducklake_sql(&ctx, &handle, "CALL ducklake_flush_inlined_data('lake')")
            .await
            .unwrap();
        assert_eq!(inlined_row_count(&backend, "t"), 0);
        assert_eq!(backend.provider.get_current_snapshot().unwrap(), head + 1);
        let schema = frame.schema().as_arrow().clone();
        assert_eq!(
            schema.field_with_name("rows_flushed").unwrap().data_type(),
            &DataType::Decimal128(38, 0)
        );
        for _ in 0..2 {
            let batches = frame.clone().collect().await.unwrap();
            let rendered = pretty_format_batches(&batches).unwrap().to_string();
            assert!(
                rendered.contains("| main        | t          | 2 "),
                "{rendered}"
            );
        }
        assert_eq!(backend.provider.get_current_snapshot().unwrap(), head + 1);
    }

    /// `schema_name =>` alone flushes every table of that schema and no other.
    #[tokio::test(flavor = "multi_thread")]
    async fn schema_name_alone_flushes_that_schema() {
        let dir = TempDir::new().unwrap();
        let backend = backend(&dir).await;
        let writer = table_writer(&backend, 10);
        for (schema, table) in [("main", "a"), ("main", "b"), ("other", "c")] {
            writer
                .write_table(schema, table, &[two_columns(&[1], &["a"])])
                .await
                .unwrap();
        }
        let (ctx, handle) = session(&backend);
        let result = sql(
            &ctx,
            &handle,
            "/* main only */ CALL ducklake_flush_inlined_data('lake', schema_name => 'main')",
        )
        .await;
        assert!(
            result.contains("| main ") && result.contains("| a "),
            "{result}"
        );
        assert!(result.contains("| b "), "{result}");
        assert!(!result.contains("| other "), "{result}");
        assert_eq!(inlined_row_count(&backend, "a"), 0);
        assert_eq!(inlined_row_count(&backend, "b"), 0);
        let snapshot = backend.provider.get_current_snapshot().unwrap();
        let other = backend
            .provider
            .get_schema_by_name("other", snapshot)
            .unwrap()
            .unwrap();
        let c = backend
            .provider
            .get_table_by_name(other.schema_id, "c", snapshot)
            .unwrap()
            .unwrap();
        assert_eq!(
            backend
                .provider
                .get_inlined_data_for_flush(c.table_id, snapshot)
                .unwrap()
                .len(),
            1
        );
    }

    /// A catalog-wide flush is all or nothing: one table that cannot be flushed
    /// fails it, nothing commits, and the files written for the other tables are
    /// removed.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_failing_table_fails_the_whole_flush_and_leaves_no_files() {
        use arrow::array::IntervalMonthDayNanoArray;
        use arrow::datatypes::{IntervalMonthDayNano, IntervalUnit};
        let dir = TempDir::new().unwrap();
        let backend = backend(&dir).await;
        let writer = table_writer(&backend, 10);
        // Tables flush in schema and table order: `a` is written before `z` fails.
        writer
            .write_table("main", "a", &[two_columns(&[1, 2], &["a", "b"])])
            .await
            .unwrap();
        let interval = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("id", DataType::Int32, true),
                Field::new("span", DataType::Interval(IntervalUnit::MonthDayNano), true),
            ])),
            vec![
                Arc::new(Int32Array::from(vec![1])),
                Arc::new(IntervalMonthDayNanoArray::from(vec![
                    IntervalMonthDayNano::new(1, 2, 3_000),
                ])),
            ],
        )
        .unwrap();
        writer.write_table("main", "z", &[interval]).await.unwrap();
        let head = backend.provider.get_current_snapshot().unwrap();
        let files = parquet_objects(dir.path());
        let (ctx, handle) = session(&backend);
        let error = execute_ducklake_sql(&ctx, &handle, "CALL ducklake_flush_inlined_data('lake')")
            .await
            .unwrap_err();
        assert!(error.to_string().contains("INTERVAL"), "{error}");
        assert_eq!(backend.provider.get_current_snapshot().unwrap(), head);
        assert_eq!(inlined_row_count(&backend, "a"), 2);
        assert_eq!(inlined_row_count(&backend, "z"), 1);
        assert_eq!(parquet_objects(dir.path()), files);
    }

    /// The commit's guards: a concurrent flush of the same table, a replaced
    /// delete file, and a retired data file each abort the commit.
    #[tokio::test(flavor = "multi_thread")]
    async fn commit_guards_abort_on_concurrent_changes() {
        use datafusion_ducklake::{
            DeleteFileInfo, DuckLakeError, InlinedFlushDeleteFile, InlinedFlushFileDelete,
            InlinedFlushTable,
        };
        let dir = TempDir::new().unwrap();
        let backend = backend(&dir).await;
        let written = table_writer(&backend, 10)
            .write_table("main", "t", &[two_columns(&[1, 2], &["a", "b"])])
            .await
            .unwrap();
        let inlined = backend
            .provider
            .get_inlined_data_for_flush(written.table_id, written.snapshot_id)
            .unwrap()
            .into_iter()
            .map(|data| data.rows.table_name)
            .collect::<Vec<_>>();
        let (ctx, handle) = session(&backend);
        sql(&ctx, &handle, "CALL ducklake_flush_inlined_data('lake')").await;

        // A flush that read before the one above committed.
        let head = backend.provider.get_current_snapshot().unwrap();
        let error = backend
            .writer
            .commit_inlined_flush(
                written.snapshot_id,
                &[InlinedFlushTable {
                    table_id: written.table_id,
                    inlined_tables: inlined,
                    ..Default::default()
                }],
            )
            .unwrap_err();
        assert!(matches!(error, DuckLakeError::Conflict(_)), "{error}");
        assert_eq!(backend.provider.get_current_snapshot().unwrap(), head);

        let file = backend
            .provider
            .get_table_files_for_select(written.table_id, head)
            .unwrap()[0]
            .data_file_id;
        let file_delete = |expected_prev_delete_file| InlinedFlushFileDelete {
            data_file_id: file,
            data_file_end: None,
            expected_prev_delete_file,
            delete: InlinedFlushDeleteFile {
                delete: DeleteFileInfo::new("unused.parquet", 1, 1),
                begin_snapshot: head,
                partial_max: None,
                end_snapshot: None,
            },
        };
        // The file has no delete file; one that expects one is stale.
        let error = backend
            .writer
            .commit_inlined_flush(
                head,
                &[InlinedFlushTable {
                    table_id: written.table_id,
                    file_deletes: vec![file_delete(Some(12345))],
                    ..Default::default()
                }],
            )
            .unwrap_err();
        assert!(matches!(error, DuckLakeError::Conflict(_)), "{error}");

        // Retire the file (its rows are not deleted inline), then move a
        // deletion into it.
        backend
            .writer
            .set_global_setting("data_inlining_row_limit", "0")
            .unwrap();
        sql(&ctx, &handle, "DELETE FROM lake.main.t").await;
        let retired_head = backend.provider.get_current_snapshot().unwrap();
        let error = backend
            .writer
            .commit_inlined_flush(
                retired_head,
                &[InlinedFlushTable {
                    table_id: written.table_id,
                    file_deletes: vec![file_delete(None)],
                    ..Default::default()
                }],
            )
            .unwrap_err();
        assert!(matches!(error, DuckLakeError::Conflict(_)), "{error}");
        assert_eq!(
            backend.provider.get_current_snapshot().unwrap(),
            retired_head
        );
    }

    /// With `hive_file_pattern` off, a partitioned flush writes flat files that
    /// still carry their partition values (official DuckDB 1.5.6's fixed flush).
    #[tokio::test(flavor = "multi_thread")]
    async fn flush_honours_hive_file_pattern() {
        let dir = TempDir::new().unwrap();
        let backend = backend(&dir).await;
        let writer = table_writer(&backend, 10);
        let created = writer
            .write_table("main", "h", &[two_columns(&[], &[])])
            .await
            .unwrap();
        backend
            .writer
            .set_table_setting(created.table_id, "hive_file_pattern", "false")
            .unwrap();
        backend
            .writer
            .set_partition_spec(
                created.table_id,
                &[(
                    "name".to_string(),
                    datafusion_ducklake::PartitionTransform::Identity,
                )],
            )
            .unwrap();
        writer
            .append_table(
                "main",
                "h",
                &[two_columns(&[0, 1, 2, 3, 4], &["0", "1", "0", "1", "0"])],
            )
            .await
            .unwrap();
        let (ctx, handle) = session(&backend);
        sql(&ctx, &handle, "CALL ducklake_flush_inlined_data('lake')").await;
        let head = backend.provider.get_current_snapshot().unwrap();
        let files = backend
            .provider
            .get_table_files_for_select(created.table_id, head)
            .unwrap();
        let flushed = files
            .iter()
            .filter(|file| file.partition_id.is_some())
            .collect::<Vec<_>>();
        assert_eq!(flushed.len(), 2, "{files:?}");
        assert!(
            flushed.iter().all(|file| !file.file.path.contains('/')),
            "{files:?}"
        );
        let rows = rows_at(
            &backend,
            head,
            "SELECT id FROM tt.main.h WHERE name = '0' ORDER BY id",
        )
        .await;
        assert!(rows.contains("| 0  |") && rows.contains("| 4  |"), "{rows}");
    }

    /// A sort key that is an expression applies to the flushed file, as in
    /// official; a key naming a column an older schema version lacks fails the
    /// flush, as official's binder does, and leaves nothing behind.
    #[tokio::test(flavor = "multi_thread")]
    async fn flush_sorts_by_expression_keys() {
        use datafusion_ducklake::{NullOrder, SortDirection, SortField};
        let dir = TempDir::new().unwrap();
        let backend = backend(&dir).await;
        let writer = table_writer(&backend, 10);
        let created = writer
            .write_table("main", "e", &[two_columns(&[1, 2, 3], &["bb", "a", "ccc"])])
            .await
            .unwrap();
        backend
            .writer
            .set_sort_spec(
                created.table_id,
                &[SortField {
                    sort_key_index: 0,
                    expression: "name".to_string(),
                    dialect: "duckdb".to_string(),
                    direction: SortDirection::Desc,
                    null_order: NullOrder::NullsLast,
                }],
            )
            .unwrap();
        // This crate writes bare-column keys only; DuckDB writes any expression.
        let pool = pool(&dir).await;
        sqlx::query("UPDATE ducklake_sort_expression SET expression = 'length(name)'")
            .execute(&pool)
            .await
            .unwrap();
        let (ctx, handle) = session(&backend);
        sql(&ctx, &handle, "CALL ducklake_flush_inlined_data('lake')").await;
        let head = backend.provider.get_current_snapshot().unwrap();
        let files = backend
            .provider
            .get_table_files_for_select(created.table_id, head)
            .unwrap();
        assert_eq!(files.len(), 1);
        let path = dir
            .path()
            .join("data")
            .join("main")
            .join("e")
            .join(&files[0].file.path);
        let batch = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(
            std::fs::File::open(path).unwrap(),
        )
        .unwrap()
        .build()
        .unwrap()
        .map(Result::unwrap)
        .next()
        .unwrap();
        let ids = batch
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap()
            .values()
            .to_vec();
        assert_eq!(ids, vec![3, 1, 2]);

        // A key on a column added after the first schema version: the flush
        // fails, commits nothing and removes the files it wrote.
        let created = writer
            .write_table("main", "v", &[two_columns(&[1, 2], &["a", "b"])])
            .await
            .unwrap();
        writer
            .append_table(
                "main",
                "v",
                &[three_columns(&[3, 4], &["c", "d"], &[Some(1), Some(2)])],
            )
            .await
            .unwrap();
        backend
            .writer
            .set_sort_spec(
                created.table_id,
                &[SortField {
                    sort_key_index: 0,
                    expression: "extra".to_string(),
                    dialect: "duckdb".to_string(),
                    direction: SortDirection::Desc,
                    null_order: NullOrder::NullsLast,
                }],
            )
            .unwrap();
        let head = backend.provider.get_current_snapshot().unwrap();
        let files_before = parquet_objects(&dir.path().join("data"));
        let error = handle
            .flush_inlined_data(&ctx.state(), None, Some("v"))
            .await
            .unwrap_err();
        assert!(
            matches!(error, datafusion_ducklake::DuckLakeError::InvalidConfig(_))
                && error.to_string().contains("\"extra\""),
            "{error}"
        );
        assert_eq!(backend.provider.get_current_snapshot().unwrap(), head);
        assert_eq!(inlined_row_count(&backend, "v"), 4);
        assert_eq!(parquet_objects(&dir.path().join("data")), files_before);
    }

    /// How a data file retired by `rewrite_data_files` came by its delete
    /// files before a flush moves its inlined deletions.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum RetiredDeletes {
        /// None: official's flush is right here.
        NoDeleteFile,
        /// One, ended when the file was retired: official writes a second.
        EndedDeleteFile,
        /// Two in a chain, the earlier ended by the later.
        Chain,
    }

    /// `(delete_file_id, delete_count, begin, end, partial_max)` of `file`'s
    /// delete files.
    async fn delete_files_of(
        pool: &sqlx::SqlitePool,
        file: i64,
    ) -> Vec<(i64, i64, i64, Option<i64>, Option<i64>)> {
        sqlx::query_as(
            "SELECT delete_file_id, delete_count, begin_snapshot, end_snapshot, partial_max
             FROM ducklake_delete_file WHERE data_file_id = ? ORDER BY begin_snapshot",
        )
        .bind(file)
        .fetch_all(pool)
        .await
        .unwrap()
    }

    /// Row 5 of a six-row file deleted inline at `s_a`, then `case`'s delete
    /// files, then the file retired; returns the flush's effect on the file's
    /// delete files and inlined deletions, with time travel checked unchanged.
    async fn flush_retired_file(case: RetiredDeletes) {
        let dir = TempDir::new().unwrap();
        let backend = backend(&dir).await;
        let created = table_writer(&backend, 0)
            .write_table(
                "main",
                "f",
                &[two_columns(&[1, 2, 3, 4, 5, 6], &["a", "b", "c", "d", "e", "f"])],
            )
            .await
            .unwrap();
        let data = backend
            .provider
            .get_table_files_for_select(created.table_id, created.snapshot_id)
            .unwrap()[0]
            .clone();
        let file = data.data_file_id;
        let s_a = backend.writer.create_snapshot().unwrap();
        let pool = pool(&dir).await;
        let table = format!("ducklake_inlined_delete_{}", created.table_id);
        for statement in [
            format!("CREATE TABLE {table} (file_id BIGINT, row_id BIGINT, begin_snapshot BIGINT)"),
            format!("INSERT INTO {table} VALUES ({file}, 4, {s_a})"),
        ] {
            sqlx::query(sqlx::AssertSqlSafe(statement))
                .execute(&pool)
                .await
                .unwrap();
        }
        let (ctx, handle) = session(&backend);
        match case {
            RetiredDeletes::NoDeleteFile => {},
            RetiredDeletes::EndedDeleteFile => {
                backend
                    .writer
                    .set_global_setting("data_inlining_row_limit", "0")
                    .unwrap();
                sql(&ctx, &handle, "DELETE FROM lake.main.f WHERE id = 6").await;
            },
            RetiredDeletes::Chain => {
                let writer = table_writer(&backend, 0);
                let data_path = dir
                    .path()
                    .join("data")
                    .join("main")
                    .join("f")
                    .join(&data.file.path);
                let data_path = data_path.to_str().unwrap();
                let mut previous = None;
                for positions in [vec![5], vec![0, 5]] {
                    let delete = writer
                        .write_delete_file("main", "f", data_path, &positions)
                        .await
                        .unwrap();
                    let head = backend.provider.get_current_snapshot().unwrap();
                    backend
                        .writer
                        .commit_positional_deletes(
                            created.table_id,
                            "main",
                            "f",
                            head,
                            &[datafusion_ducklake::DeleteFileEntry {
                                data_file_id: file,
                                expected_prev_delete_file: previous,
                                delete,
                            }],
                        )
                        .unwrap();
                    previous = delete_files_of(&pool, file).await.last().map(|row| row.0);
                }
            },
        }
        let table_provider = datafusion::catalog::CatalogProvider::schema(&handle, "main")
            .unwrap()
            .table("f")
            .await
            .unwrap()
            .unwrap();
        let rewritten = table_provider
            .downcast_ref::<datafusion_ducklake::DuckLakeTable>()
            .unwrap()
            .rewrite_data_files(
                &ctx.state(),
                datafusion_ducklake::RewriteOptions {
                    delete_threshold: 0.0,
                    data_file_ids: None,
                },
            )
            .await
            .unwrap();
        assert_eq!(rewritten.files_processed, 1);
        let head = backend.provider.get_current_snapshot().unwrap();
        let before_deletes = delete_files_of(&pool, file).await;
        let query = "SELECT id FROM tt.main.f ORDER BY id";
        let before = history(&backend, created.snapshot_id, query).await;
        let at_a = before.iter().find(|(s, _)| *s == s_a).unwrap();
        assert!(!at_a.1.contains("| 5  |") && at_a.1.contains("| 6  |"));

        let (ctx, handle) = session(&backend);
        sql(&ctx, &handle, "CALL ducklake_flush_inlined_data('lake')").await;
        let left: i64 =
            sqlx::query_scalar(sqlx::AssertSqlSafe(format!("SELECT COUNT(*) FROM {table}")))
                .fetch_one(&pool)
                .await
                .unwrap();
        let after_deletes = delete_files_of(&pool, file).await;
        match case {
            // As official: a delete file from the deletion's snapshot that
            // never ends, as the retired file needs no end.
            RetiredDeletes::NoDeleteFile => {
                assert_eq!(left, 0);
                assert_eq!(backend.provider.get_current_snapshot().unwrap(), head + 1);
                assert_eq!(before_deletes, vec![]);
                assert_eq!(after_deletes.len(), 1);
                let (_, count, begin, end, max) = after_deletes[0];
                assert_eq!((count, begin, end, max), (1, s_a, None, Some(s_a)));
                assert_eq!(
                    last_changes(&backend).as_deref(),
                    Some(format!("deleted_from_table:{}", created.table_id).as_str())
                );
            },
            // The ended delete file is replaced by one carrying both deletions
            // that ends where it ended; official writes a second, live one.
            RetiredDeletes::EndedDeleteFile => {
                assert_eq!(left, 0);
                assert_eq!(before_deletes.len(), 1);
                let (_, _, old_begin, old_end, _) = before_deletes[0];
                assert_eq!(after_deletes.len(), 1);
                let (_, count, begin, end, max) = after_deletes[0];
                assert_eq!((count, begin, end, max), (2, s_a, old_end, Some(old_begin)));
            },
            // The merged file would overlap the earlier delete file: the
            // deletion stays inlined and nothing commits.
            RetiredDeletes::Chain => {
                assert_eq!(left, 1);
                assert_eq!(backend.provider.get_current_snapshot().unwrap(), head);
                assert_eq!(after_deletes, before_deletes);
            },
        }
        for (snapshot, rows) in &before {
            assert_eq!(
                &rows_at(&backend, *snapshot, query).await,
                rows,
                "{case:?}: snapshot {snapshot}"
            );
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn inlined_deletions_of_a_retired_file_move_to_a_delete_file() {
        flush_retired_file(RetiredDeletes::NoDeleteFile).await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn inlined_deletions_of_a_retired_file_merge_into_its_ended_delete_file() {
        flush_retired_file(RetiredDeletes::EndedDeleteFile).await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn inlined_deletions_that_cannot_merge_into_a_delete_file_chain_stay_inlined() {
        flush_retired_file(RetiredDeletes::Chain).await;
    }
}

/// Every shared scenario, by name.
const SCENARIOS: [&str; 5] = [
    "schema evolution and deletes",
    "many rows",
    "partitioned",
    "rollover",
    "partitioned past target file size",
];

async fn run_scenario(name: &str, backend: &Backend) {
    match name {
        "schema evolution and deletes" => scenario_schema_evolution_and_deletes(backend).await,
        "many rows" => scenario_many_rows(backend).await,
        "partitioned" => scenario_partitioned(backend).await,
        "rollover" => scenario_rollover(backend).await,
        "partitioned past target file size" => {
            scenario_partitioned_past_target_file_size(backend).await
        },
        other => panic!("unknown scenario {other}"),
    }
}

#[cfg(feature = "write-duckdb")]
mod duckdb_backend {
    use super::*;
    use datafusion_ducklake::DuckdbMetadataWriter;
    use tempfile::TempDir;

    fn backend(dir: &TempDir) -> Backend {
        let data = dir.path().join("data");
        std::fs::create_dir_all(&data).unwrap();
        let writer =
            DuckdbMetadataWriter::new_with_init(dir.path().join("c.duckdb").display().to_string())
                .unwrap();
        writer.set_data_path(data.to_str().unwrap()).unwrap();
        let provider = writer.metadata_provider();
        Backend {
            writer: Arc::new(writer),
            provider: Arc::new(provider),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn scenarios() {
        for name in SCENARIOS {
            let dir = TempDir::new().unwrap();
            run_scenario(name, &backend(&dir)).await;
        }
    }
}

#[cfg(feature = "write-mysql")]
mod mysql_backend {
    use super::*;
    use datafusion_ducklake::{MySqlMetadataProvider, MySqlMetadataWriter};
    use tempfile::TempDir;
    use testcontainers::runners::AsyncRunner;
    use testcontainers_modules::mysql::Mysql;

    #[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
    #[tokio::test(flavor = "multi_thread")]
    async fn scenarios() {
        let container = Mysql::default().start().await.unwrap();
        let port = container.get_host_port_ipv4(3306).await.unwrap();
        let admin = format!("mysql://root@127.0.0.1:{port}/test");
        let pool = sqlx::MySqlPool::connect(&admin).await.unwrap();
        for (index, name) in SCENARIOS.into_iter().enumerate() {
            let database = format!("flush_{index}");
            sqlx::query(sqlx::AssertSqlSafe(format!("CREATE DATABASE {database}")))
                .execute(&pool)
                .await
                .unwrap();
            let url = format!("mysql://root@127.0.0.1:{port}/{database}");
            let dir = TempDir::new().unwrap();
            let data = dir.path().join("data");
            std::fs::create_dir_all(&data).unwrap();
            let writer = MySqlMetadataWriter::new_with_init(&url).await.unwrap();
            writer.set_data_path(data.to_str().unwrap()).unwrap();
            let backend = Backend {
                writer: Arc::new(writer),
                provider: Arc::new(MySqlMetadataProvider::new(&url).await.unwrap()),
            };
            run_scenario(name, &backend).await;
        }
    }
}

#[cfg(feature = "multicatalog-postgres")]
mod multicatalog_backend {
    use super::*;
    use datafusion_ducklake::{MulticatalogManager, MulticatalogProvider, PostgresMetadataWriter};
    use tempfile::TempDir;
    use testcontainers::runners::AsyncRunner;
    use testcontainers_modules::postgres::Postgres;

    #[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
    #[tokio::test(flavor = "multi_thread")]
    async fn scenarios() {
        let container = Postgres::default().start().await.unwrap();
        let port = container.get_host_port_ipv4(5432).await.unwrap();
        let admin = format!("postgresql://postgres:postgres@127.0.0.1:{port}/postgres");
        let admin_pool = sqlx::PgPool::connect(&admin).await.unwrap();
        for (index, name) in SCENARIOS.into_iter().enumerate() {
            let database = format!("flush_{index}");
            sqlx::query(sqlx::AssertSqlSafe(format!("CREATE DATABASE {database}")))
                .execute(&admin_pool)
                .await
                .unwrap();
            let url = format!("postgresql://postgres:postgres@127.0.0.1:{port}/{database}");
            let pool = sqlx::PgPool::connect(&url).await.unwrap();
            datafusion_ducklake::initialize_multicatalog_schema(&pool)
                .await
                .unwrap();
            let catalog = MulticatalogManager::new(pool.clone())
                .create_catalog("lake")
                .await
                .unwrap();
            let dir = TempDir::new().unwrap();
            let data = dir.path().join("data");
            std::fs::create_dir_all(&data).unwrap();
            let writer = PostgresMetadataWriter::with_pool(pool.clone(), catalog)
                .await
                .unwrap();
            writer.set_data_path(data.to_str().unwrap()).unwrap();
            let backend = Backend {
                writer: Arc::new(writer),
                provider: Arc::new(MulticatalogProvider::with_pool(pool, "lake").await.unwrap()),
            };
            run_scenario(name, &backend).await;
        }
    }
}
