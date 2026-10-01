//! Latest-snapshot reads of catalogs that are not bound to a snapshot.

#![cfg(all(feature = "write-sqlite", feature = "metadata-sqlite"))]

use std::sync::Arc;

use arrow::array::{Int32Array, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::catalog::CatalogProvider;
use datafusion::prelude::SessionContext;
use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeTableWriter, MetadataWriter, SqliteMetadataProvider,
    SqliteMetadataWriter, register_snapshot_consistency,
};
use object_store::local::LocalFileSystem;
use tempfile::TempDir;

fn batch(rows: &[(i32, i32)]) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("value", DataType::Int32, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(
                rows.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            )),
            Arc::new(Int32Array::from(
                rows.iter().map(|(_, value)| *value).collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap()
}

async fn setup() -> (TempDir, String, i64) {
    let temp = TempDir::new().unwrap();
    let database = temp.path().join("catalog.db");
    let data = temp.path().join("data");
    std::fs::create_dir(&data).unwrap();
    let connection = format!("sqlite:{}?mode=rwc", database.display());
    let writer = Arc::new(
        SqliteMetadataWriter::new_with_init(&connection)
            .await
            .unwrap(),
    );
    writer.set_data_path(data.to_str().unwrap()).unwrap();
    let result = DuckLakeTableWriter::new(writer, Arc::new(LocalFileSystem::new()))
        .unwrap()
        .write_table("main", "t", &[batch(&[(1, 10)])])
        .await
        .unwrap();
    (temp, connection, result.snapshot_id)
}

async fn writable_context(connection: &str) -> SessionContext {
    let provider = SqliteMetadataProvider::new(connection).await.unwrap();
    let writer = SqliteMetadataWriter::new(connection).await.unwrap();
    let catalog = DuckLakeCatalog::with_writer(Arc::new(provider), Arc::new(writer)).unwrap();
    let context = SessionContext::new();
    context.register_catalog("lake", Arc::new(catalog));
    context
}

async fn read_only_context(connection: &str) -> SessionContext {
    let provider = SqliteMetadataProvider::new(connection).await.unwrap();
    let catalog = DuckLakeCatalog::new(provider).unwrap();
    let context = SessionContext::new();
    context.register_catalog("lake", Arc::new(catalog));
    context
}

async fn fixed_context(connection: &str, snapshot_id: i64) -> SessionContext {
    let provider = Arc::new(SqliteMetadataProvider::new(connection).await.unwrap());
    let catalog = DuckLakeCatalog::with_snapshot(provider, snapshot_id).unwrap();
    let context = SessionContext::new();
    context.register_catalog("lake", Arc::new(catalog));
    context
}

async fn rows(context: &SessionContext) -> Vec<(i32, i32)> {
    let batches = context
        .sql("SELECT id, value FROM lake.main.t ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let mut rows = Vec::new();
    for batch in batches {
        let ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let values = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        rows.extend((0..batch.num_rows()).map(|index| (ids.value(index), values.value(index))));
    }
    rows
}

async fn affected(context: &SessionContext, sql: &str) -> u64 {
    let batches = context.sql(sql).await.unwrap().collect().await.unwrap();
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
async fn own_commits_refresh_insert_update_and_delete_reads() {
    let (_temp, connection, _initial_snapshot) = setup().await;
    let context = writable_context(&connection).await;

    assert_eq!(rows(&context).await, vec![(1, 10)]);
    assert_eq!(
        affected(
            &context,
            "INSERT INTO lake.main.t (id, value) VALUES (2, 20)"
        )
        .await,
        1
    );
    assert_eq!(rows(&context).await, vec![(1, 10), (2, 20)]);

    assert_eq!(
        affected(&context, "DELETE FROM lake.main.t WHERE id = 1").await,
        1
    );
    assert_eq!(rows(&context).await, vec![(2, 20)]);

    assert_eq!(
        affected(&context, "UPDATE lake.main.t SET value = 21 WHERE id = 2").await,
        1
    );
    assert_eq!(rows(&context).await, vec![(2, 21)]);

    assert_eq!(
        affected(&context, "UPDATE lake.main.t SET value = 22 WHERE id = 2").await,
        1
    );
    assert_eq!(rows(&context).await, vec![(2, 22)]);

    assert_eq!(affected(&context, "DELETE FROM lake.main.t").await, 1);
    assert_eq!(rows(&context).await, Vec::<(i32, i32)>::new());
}

#[tokio::test(flavor = "multi_thread")]
async fn external_commit_is_visible_except_to_explicit_snapshot() {
    let (_temp, connection, initial_snapshot) = setup().await;
    let writable = writable_context(&connection).await;
    let read_only = read_only_context(&connection).await;
    let fixed = fixed_context(&connection, initial_snapshot).await;

    assert_eq!(rows(&writable).await, vec![(1, 10)]);
    assert_eq!(rows(&read_only).await, vec![(1, 10)]);

    let external_writer = Arc::new(SqliteMetadataWriter::new(&connection).await.unwrap());
    DuckLakeTableWriter::new(external_writer, Arc::new(LocalFileSystem::new()))
        .unwrap()
        .append_table("main", "t", &[batch(&[(2, 20)])])
        .await
        .unwrap();

    assert_eq!(rows(&writable).await, vec![(1, 10), (2, 20)]);
    assert_eq!(rows(&read_only).await, vec![(1, 10), (2, 20)]);
    assert_eq!(rows(&fixed).await, vec![(1, 10)]);

    assert_eq!(
        affected(
            &writable,
            "INSERT INTO lake.main.t (id, value) VALUES (3, 30)"
        )
        .await,
        1
    );
    assert_eq!(rows(&writable).await, vec![(1, 10), (2, 20), (3, 30)]);
    assert_eq!(rows(&read_only).await, vec![(1, 10), (2, 20), (3, 30)]);
    assert_eq!(rows(&fixed).await, vec![(1, 10)]);
}

async fn setup_two_tables() -> (TempDir, String) {
    let (temp, connection, _) = setup().await;
    let writer = Arc::new(SqliteMetadataWriter::new(&connection).await.unwrap());
    DuckLakeTableWriter::new(writer, Arc::new(LocalFileSystem::new()))
        .unwrap()
        .write_table("main", "u", &[batch(&[(1, 10)])])
        .await
        .unwrap();
    (temp, connection)
}

async fn append_row(connection: &str, table: &str, row: (i32, i32)) {
    let writer = Arc::new(SqliteMetadataWriter::new(connection).await.unwrap());
    DuckLakeTableWriter::new(writer, Arc::new(LocalFileSystem::new()))
        .unwrap()
        .append_table("main", table, &[batch(&[row])])
        .await
        .unwrap();
}

async fn count_pair(context: &SessionContext) -> (i64, i64) {
    let batches = context
        .sql("SELECT (SELECT count(*) FROM t) AS t_rows, (SELECT count(*) FROM u) AS u_rows")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let count = |index: usize| {
        batches[0]
            .column(index)
            .as_any()
            .downcast_ref::<arrow::array::Int64Array>()
            .unwrap()
            .value(0)
    };
    (count(0), count(1))
}

#[tokio::test(flavor = "multi_thread")]
async fn statement_reads_one_snapshot_across_table_lookups() {
    let (_temp, connection) = setup_two_tables().await;
    let provider = SqliteMetadataProvider::new(&connection).await.unwrap();
    let schema = DuckLakeCatalog::new(provider)
        .unwrap()
        .schema("main")
        .unwrap();

    // Looked up before the commits below, so it resolves an older snapshot than `u`.
    let table_t = schema.table("t").await.unwrap().unwrap();
    append_row(&connection, "t", (2, 20)).await;
    append_row(&connection, "u", (2, 20)).await;
    let table_u = schema.table("u").await.unwrap().unwrap();

    let plain = SessionContext::new();
    plain.register_table("t", table_t.clone()).unwrap();
    plain.register_table("u", table_u.clone()).unwrap();
    let consistent = SessionContext::new();
    register_snapshot_consistency(&consistent);
    consistent.register_table("t", table_t).unwrap();
    consistent.register_table("u", table_u).unwrap();

    assert_eq!(count_pair(&plain).await, (1, 2));
    assert_eq!(count_pair(&consistent).await, (2, 2));
}

#[tokio::test(flavor = "multi_thread")]
async fn insert_select_reads_one_snapshot_across_source_and_target() {
    let (_temp, connection) = setup_two_tables().await;
    let provider = SqliteMetadataProvider::new(&connection).await.unwrap();
    let writer = SqliteMetadataWriter::new(&connection).await.unwrap();
    let schema = DuckLakeCatalog::with_writer(Arc::new(provider), Arc::new(writer))
        .unwrap()
        .schema("main")
        .unwrap();

    let table_t = schema.table("t").await.unwrap().unwrap();
    append_row(&connection, "t", (2, 20)).await;
    append_row(&connection, "u", (2, 20)).await;
    let table_u = schema.table("u").await.unwrap().unwrap();

    let context = SessionContext::new();
    register_snapshot_consistency(&context);
    context.register_table("t", table_t).unwrap();
    context.register_table("u", table_u).unwrap();

    assert_eq!(
        affected(&context, "INSERT INTO u SELECT id + 10, value FROM t").await,
        2
    );
}

async fn insert_view(connection: &str, view_id: i64, sql: &str, begin: i64, end: Option<i64>) {
    let pool = sqlx::SqlitePool::connect(connection).await.unwrap();
    let schema_id: i64 = sqlx::query_scalar(
        "SELECT schema_id FROM ducklake_schema WHERE schema_name = 'main' AND end_snapshot IS NULL",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    sqlx::query(
        "INSERT INTO ducklake_view
         (view_id, schema_id, view_name, dialect, sql, column_aliases, begin_snapshot, end_snapshot)
         VALUES (?, ?, 'vt', 'duckdb', ?, '', ?, ?)",
    )
    .bind(view_id)
    .bind(schema_id)
    .bind(sql)
    .bind(begin)
    .bind(end)
    .execute(&pool)
    .await
    .unwrap();
}

async fn latest_snapshot(connection: &str) -> i64 {
    let pool = sqlx::SqlitePool::connect(connection).await.unwrap();
    sqlx::query_scalar("SELECT MAX(snapshot_id) FROM ducklake_snapshot")
        .fetch_one(&pool)
        .await
        .unwrap()
}

async fn view_count_pair(context: &SessionContext) -> datafusion::error::Result<(i64, i64)> {
    let batches = context
        .sql("SELECT (SELECT count(*) FROM vt) AS v_rows, (SELECT count(*) FROM u) AS u_rows")
        .await?
        .collect()
        .await?;
    let count = |index: usize| {
        batches[0]
            .column(index)
            .as_any()
            .downcast_ref::<arrow::array::Int64Array>()
            .unwrap()
            .value(0)
    };
    Ok((count(0), count(1)))
}

#[tokio::test(flavor = "multi_thread")]
async fn statement_moves_view_tables_to_the_common_snapshot() {
    let (_temp, connection) = setup_two_tables().await;
    insert_view(&connection, 100, "SELECT id FROM t", 0, None).await;
    let provider = SqliteMetadataProvider::new(&connection).await.unwrap();
    let schema = DuckLakeCatalog::new(provider)
        .unwrap()
        .schema("main")
        .unwrap();

    let view = schema.table("vt").await.unwrap().unwrap();
    append_row(&connection, "t", (2, 20)).await;
    append_row(&connection, "u", (2, 20)).await;
    let table_u = schema.table("u").await.unwrap().unwrap();

    let plain = SessionContext::new();
    plain.register_table("vt", view.clone()).unwrap();
    plain.register_table("u", table_u.clone()).unwrap();
    let consistent = SessionContext::new();
    register_snapshot_consistency(&consistent);
    consistent.register_table("vt", view).unwrap();
    consistent.register_table("u", table_u).unwrap();

    assert_eq!(view_count_pair(&plain).await.unwrap(), (1, 2));
    assert_eq!(view_count_pair(&consistent).await.unwrap(), (2, 2));
}

#[tokio::test(flavor = "multi_thread")]
async fn statement_fails_when_its_view_changed_between_lookups() {
    let (_temp, connection) = setup_two_tables().await;
    insert_view(&connection, 100, "SELECT id FROM t", 0, None).await;
    let provider = SqliteMetadataProvider::new(&connection).await.unwrap();
    let schema = DuckLakeCatalog::new(provider)
        .unwrap()
        .schema("main")
        .unwrap();

    let view = schema.table("vt").await.unwrap().unwrap();
    append_row(&connection, "u", (2, 20)).await;
    let redefined_at = latest_snapshot(&connection).await;
    let pool = sqlx::SqlitePool::connect(&connection).await.unwrap();
    sqlx::query("UPDATE ducklake_view SET end_snapshot = ? WHERE view_id = 100")
        .bind(redefined_at)
        .execute(&pool)
        .await
        .unwrap();
    insert_view(
        &connection,
        101,
        "SELECT id + 1 AS id FROM t",
        redefined_at,
        None,
    )
    .await;
    let table_u = schema.table("u").await.unwrap().unwrap();

    let context = SessionContext::new();
    register_snapshot_consistency(&context);
    context.register_table("vt", view).unwrap();
    context.register_table("u", table_u).unwrap();

    let error = view_count_pair(&context).await.unwrap_err();
    assert!(
        error
            .to_string()
            .contains("changed between lookups of one statement; retry the statement"),
        "{error}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn snapshots_of_different_catalogs_are_not_compared() {
    let (_temp_1, connection_1, _) = setup().await;
    let (_temp_2, connection_2, _) = setup().await;
    // The second catalog has the higher snapshot IDs.
    for id in 2..5 {
        append_row(&connection_2, "t", (id, id * 10)).await;
    }

    let schema_1 = DuckLakeCatalog::new(SqliteMetadataProvider::new(&connection_1).await.unwrap())
        .unwrap()
        .schema("main")
        .unwrap();
    let schema_2 = DuckLakeCatalog::new(SqliteMetadataProvider::new(&connection_2).await.unwrap())
        .unwrap()
        .schema("main")
        .unwrap();
    let table_1 = schema_1.table("t").await.unwrap().unwrap();
    append_row(&connection_1, "t", (2, 20)).await;
    let table_2 = schema_2.table("t").await.unwrap().unwrap();

    let context = SessionContext::new();
    register_snapshot_consistency(&context);
    context.register_table("t1", table_1).unwrap();
    context.register_table("t2", table_2).unwrap();
    let batches = context
        .sql("SELECT (SELECT count(*) FROM t1) AS rows_1, (SELECT count(*) FROM t2) AS rows_2")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let count = |index: usize| {
        batches[0]
            .column(index)
            .as_any()
            .downcast_ref::<arrow::array::Int64Array>()
            .unwrap()
            .value(0)
    };

    assert_eq!((count(0), count(1)), (1, 4));
}
