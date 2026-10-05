//! Integration tests for DuckLake `MERGE INTO` on SQLite metadata.

#![cfg(all(feature = "write-sqlite", feature = "metadata-sqlite"))]

use std::path::Path;
use std::sync::Arc;

use arrow::array::{Array, Float64Array, Int32Array, Int64Array, StringArray, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use arrow::util::display::array_value_to_string;
use datafusion::catalog::CatalogProvider;
use datafusion::prelude::{SessionContext, col};
use object_store::local::LocalFileSystem;
use sqlx::sqlite::SqlitePool;
use tempfile::TempDir;

use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeTableWriter, MetadataWriter, SqliteMetadataProvider,
    SqliteMetadataWriter, execute_ducklake_sql, partition::PartitionTransform,
};

fn table_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("salary", DataType::Float64, false),
    ]))
}

fn seed_batch() -> RecordBatch {
    RecordBatch::try_new(
        table_schema(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2])),
            Arc::new(StringArray::from(vec!["John", "Anna"])),
            Arc::new(Float64Array::from(vec![92_000.0, 100_000.0])),
        ],
    )
    .unwrap()
}

fn paths(temp_dir: &TempDir) -> (String, String) {
    let db_path = temp_dir.path().join("test.db");
    let data_path = temp_dir.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();
    (
        format!("sqlite:{}?mode=rwc", db_path.display()),
        data_path.to_str().unwrap().to_string(),
    )
}

async fn seed_table(temp_dir: &TempDir) -> String {
    let (conn_str, data_path) = paths(temp_dir);
    let writer = Arc::new(
        SqliteMetadataWriter::new_with_init(&conn_str)
            .await
            .unwrap(),
    );
    writer.set_data_path(&data_path).unwrap();
    DuckLakeTableWriter::new(writer, Arc::new(LocalFileSystem::new()))
        .unwrap()
        .write_table("main", "people", &[seed_batch()])
        .await
        .unwrap();
    conn_str
}

async fn writable_catalog(conn_str: &str) -> (SessionContext, Arc<DuckLakeCatalog>) {
    let writer = SqliteMetadataWriter::new(conn_str).await.unwrap();
    let provider = SqliteMetadataProvider::new(conn_str).await.unwrap();
    let catalog =
        Arc::new(DuckLakeCatalog::with_writer(Arc::new(provider), Arc::new(writer)).unwrap());
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::clone(&catalog) as Arc<dyn CatalogProvider>);
    (ctx, catalog)
}

async fn read_rows(conn_str: &str) -> Vec<(i64, i32, String, f64)> {
    let provider = SqliteMetadataProvider::new(conn_str).await.unwrap();
    let catalog = DuckLakeCatalog::new(provider)
        .unwrap()
        .with_row_lineage(true);
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));
    let batches = ctx
        .table("ducklake.main.people")
        .await
        .unwrap()
        .select(vec![col("rowid"), col("id"), col("name"), col("salary")])
        .unwrap()
        .sort(vec![col("id").sort(true, true)])
        .unwrap()
        .collect()
        .await
        .unwrap();
    let mut rows = Vec::new();
    for batch in batches {
        let rowids = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let ids = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let salaries = batch
            .column(3)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        for row in 0..batch.num_rows() {
            rows.push((
                rowids.value(row),
                ids.value(row),
                array_value_to_string(batch.column(2), row).unwrap(),
                salaries.value(row),
            ));
        }
    }
    rows
}

async fn snapshot_id(pool: &SqlitePool) -> i64 {
    sqlx::query_scalar("SELECT COALESCE(MAX(snapshot_id), 0) FROM ducklake_snapshot")
        .fetch_one(pool)
        .await
        .unwrap()
}

async fn run_count(ctx: &SessionContext, catalog: &DuckLakeCatalog, sql: &str) -> u64 {
    let batches = execute_ducklake_sql(ctx, catalog, sql)
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 1);
    batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap()
        .value(0)
}

#[tokio::test(flavor = "multi_thread")]
async fn merge_upsert_preserves_rowids_in_one_snapshot() {
    let temp_dir = TempDir::new().unwrap();
    let conn_str = seed_table(&temp_dir).await;
    let pool = SqlitePool::connect(&conn_str).await.unwrap();
    let before = snapshot_id(&pool).await;
    let (ctx, catalog) = writable_catalog(&conn_str).await;

    let count = run_count(
        &ctx,
        &catalog,
        "MERGE INTO ducklake.main.people AS p
         USING (VALUES (1, 'John', 105000.0), (3, 'Sarah', 95000.0)) AS s(id, name, salary)
         ON p.id = s.id
         WHEN MATCHED THEN UPDATE
         WHEN NOT MATCHED THEN INSERT",
    )
    .await;

    assert_eq!(count, 2);
    assert_eq!(snapshot_id(&pool).await, before + 1);
    assert_eq!(
        read_rows(&conn_str).await,
        vec![
            (0, 1, "John".to_string(), 105_000.0),
            (1, 2, "Anna".to_string(), 100_000.0),
            (2, 3, "Sarah".to_string(), 95_000.0),
        ]
    );

    let next_row_id: i64 = sqlx::query_scalar("SELECT next_row_id FROM ducklake_table_stats")
        .fetch_one(&pool)
        .await
        .unwrap();
    let output_row_id_starts: Vec<Option<i64>> = sqlx::query_scalar(
        "SELECT row_id_start FROM ducklake_data_file
         WHERE begin_snapshot = ? ORDER BY data_file_id",
    )
    .bind(before + 1)
    .fetch_all(&pool)
    .await
    .unwrap();
    // Two rows seeded, one id reserved for the inserted row, then a fresh range of
    // two for the rewrite output as for UPDATE; lineage stays in the embedded column.
    assert_eq!(next_row_id, 5);
    assert_eq!(output_row_id_starts, vec![Some(3)]);
}

#[tokio::test(flavor = "multi_thread")]
async fn merge_delete_applies_matched_predicate() {
    let temp_dir = TempDir::new().unwrap();
    let conn_str = seed_table(&temp_dir).await;
    let (ctx, catalog) = writable_catalog(&conn_str).await;

    let count = run_count(
        &ctx,
        &catalog,
        "MERGE INTO ducklake.main.people AS p
         USING (VALUES (1), (2)) AS s(id)
         ON p.id = s.id
         WHEN MATCHED AND p.salary >= 100000.0 THEN DELETE",
    )
    .await;

    assert_eq!(count, 1);
    assert_eq!(
        read_rows(&conn_str).await,
        vec![(0, 1, "John".to_string(), 92_000.0)]
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn merge_update_set_changes_only_assigned_columns() {
    let temp_dir = TempDir::new().unwrap();
    let conn_str = seed_table(&temp_dir).await;
    let (ctx, catalog) = writable_catalog(&conn_str).await;

    let count = run_count(
        &ctx,
        &catalog,
        "MERGE INTO ducklake.main.people AS p
         USING (VALUES (1, 98000.0)) AS s(id, salary)
         ON p.id = s.id
         WHEN MATCHED THEN UPDATE SET salary = s.salary",
    )
    .await;

    assert_eq!(count, 1);
    assert_eq!(
        read_rows(&conn_str).await,
        vec![(0, 1, "John".to_string(), 98_000.0), (1, 2, "Anna".to_string(), 100_000.0),]
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn merge_rejects_multiple_matched_actions_before_writing() {
    let temp_dir = TempDir::new().unwrap();
    let conn_str = seed_table(&temp_dir).await;
    let pool = SqlitePool::connect(&conn_str).await.unwrap();
    let before = snapshot_id(&pool).await;
    let (ctx, catalog) = writable_catalog(&conn_str).await;

    let err = execute_ducklake_sql(
        &ctx,
        &catalog,
        "MERGE INTO ducklake.main.people AS p
         USING (VALUES (1, 'John', 105000.0)) AS s(id, name, salary)
         ON p.id = s.id
         WHEN MATCHED AND p.salary < 100000.0 THEN UPDATE
         WHEN MATCHED AND p.salary >= 100000.0 THEN DELETE",
    )
    .await
    .unwrap_err();

    assert_eq!(
        err.to_string(),
        "This feature is not implemented: MERGE INTO with DuckLake only supports a single UPDATE/DELETE action currently; the DuckLake specification permits more"
    );
    assert_eq!(snapshot_id(&pool).await, before);
    assert_eq!(
        read_rows(&conn_str).await,
        vec![(0, 1, "John".to_string(), 92_000.0), (1, 2, "Anna".to_string(), 100_000.0),]
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn merge_rejects_partitioned_target_before_writing() {
    let temp_dir = TempDir::new().unwrap();
    let conn_str = seed_table(&temp_dir).await;
    let pool = SqlitePool::connect(&conn_str).await.unwrap();
    let writer = SqliteMetadataWriter::new(&conn_str).await.unwrap();
    let table_id: i64 = sqlx::query_scalar("SELECT table_id FROM ducklake_table")
        .fetch_one(&pool)
        .await
        .unwrap();
    writer
        .set_partition_spec(
            table_id,
            &[("id".to_string(), PartitionTransform::Identity)],
        )
        .unwrap();
    let before = snapshot_id(&pool).await;
    let (ctx, catalog) = writable_catalog(&conn_str).await;

    let err = execute_ducklake_sql(
        &ctx,
        &catalog,
        "MERGE INTO ducklake.main.people AS p
         USING (VALUES (1, 'John', 105000.0)) AS s(id, name, salary)
         ON p.id = s.id
         WHEN MATCHED THEN UPDATE",
    )
    .await
    .unwrap_err();

    assert_eq!(
        err.to_string(),
        "This feature is not implemented: MERGE INTO does not support partitioned target tables currently"
    );
    assert_eq!(snapshot_id(&pool).await, before);
}

fn official_rows(path: &Path) -> anyhow::Result<Vec<(i32, String, f64)>> {
    let conn = duckdb::Connection::open_in_memory()?;
    conn.execute("LOAD ducklake", [])?;
    conn.execute(&format!("ATTACH 'ducklake:{}' AS c", path.display()), [])?;
    conn.execute(
        "CREATE TABLE c.people (id INTEGER, name VARCHAR, salary DOUBLE)",
        [],
    )?;
    conn.execute(
        "INSERT INTO c.people VALUES (1, 'John', 92000.0), (2, 'Anna', 100000.0)",
        [],
    )?;
    conn.execute(
        "MERGE INTO c.people AS p
         USING (VALUES (1, 'John', 105000.0), (3, 'Sarah', 95000.0)) AS s(id, name, salary)
         ON p.id = s.id
         WHEN MATCHED THEN UPDATE
         WHEN NOT MATCHED THEN INSERT",
        [],
    )?;
    let mut statement = conn.prepare("SELECT id, name, salary FROM c.people ORDER BY id")?;
    let rows = statement
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(rows)
}

#[tokio::test(flavor = "multi_thread")]
async fn merge_matches_official_ducklake_rows() {
    let oracle_dir = TempDir::new().unwrap();
    let expected = official_rows(&oracle_dir.path().join("oracle.ducklake")).unwrap();
    let temp_dir = TempDir::new().unwrap();
    let conn_str = seed_table(&temp_dir).await;
    let (ctx, catalog) = writable_catalog(&conn_str).await;
    run_count(
        &ctx,
        &catalog,
        "MERGE INTO ducklake.main.people AS p
         USING (VALUES (1, 'John', 105000.0), (3, 'Sarah', 95000.0)) AS s(id, name, salary)
         ON p.id = s.id
         WHEN MATCHED THEN UPDATE
         WHEN NOT MATCHED THEN INSERT",
    )
    .await;
    let actual = read_rows(&conn_str)
        .await
        .into_iter()
        .map(|(_, id, name, salary)| (id, name, salary))
        .collect::<Vec<_>>();

    assert_eq!(actual, expected);
}
