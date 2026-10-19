//! Integration tests for `CREATE TABLE AS SELECT` through `execute_ducklake_sql`.

#![cfg(all(feature = "write-sqlite", feature = "metadata-sqlite"))]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use arrow::array::{
    Array, BooleanArray, Float64Array, Int32Array, Int64Array, ListArray, MapArray, StringArray,
    StructArray,
};
use arrow::buffer::OffsetBuffer;
use arrow::datatypes::{DataType, Field, Fields, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use datafusion::datasource::MemTable;
use datafusion::prelude::*;
use object_store::ObjectStore;
use object_store::local::LocalFileSystem;
use rstest::rstest;
use tempfile::TempDir;

use datafusion::logical_expr::{ColumnarValue, ScalarUDF, Volatility, create_udf};
use datafusion_ducklake::{
    ColumnDef, DuckLakeCatalog, DuckLakeWriteOptions, MetadataProvider, MetadataWriter,
    SqliteMetadataProvider, SqliteMetadataWriter, WriteMode, execute_ducklake_sql,
};

struct Lake {
    temp: TempDir,
    connection: String,
}

impl Lake {
    async fn provider(&self) -> SqliteMetadataProvider {
        SqliteMetadataProvider::new(&self.connection).await.unwrap()
    }

    async fn head(&self) -> i64 {
        self.provider().await.get_current_snapshot().unwrap()
    }

    async fn table_id(&self, name: &str) -> Option<i64> {
        let provider = self.provider().await;
        let head = provider.get_current_snapshot().unwrap();
        let schema = provider.get_schema_by_name("main", head).unwrap().unwrap();
        provider
            .get_table_by_name(schema.schema_id, name, head)
            .unwrap()
            .map(|table| table.table_id)
    }

    /// `(column_name, column_type, is_nullable)` of the live table.
    async fn columns(&self, name: &str) -> Vec<(String, String, bool)> {
        let provider = self.provider().await;
        let table_id = self.table_id(name).await.unwrap();
        provider
            .get_table_structure(table_id, provider.get_current_snapshot().unwrap())
            .unwrap()
            .into_iter()
            .map(|column| (column.column_name, column.column_type, column.is_nullable))
            .collect()
    }

    /// `ducklake_table` rows named `name`, whatever their snapshots.
    async fn table_rows(&self, name: &str) -> i64 {
        let pool = sqlx::SqlitePool::connect(&self.connection).await.unwrap();
        sqlx::query_scalar("SELECT COUNT(*) FROM ducklake_table WHERE table_name = ?")
            .bind(name)
            .fetch_one(&pool)
            .await
            .unwrap()
    }

    async fn data_file_count(&self, name: &str) -> usize {
        let provider = self.provider().await;
        let table_id = self.table_id(name).await.unwrap();
        provider
            .get_table_files_for_select(table_id, provider.get_current_snapshot().unwrap())
            .unwrap()
            .len()
    }

    /// A writable catalog registered as `ducklake` next to the in-memory `source` table.
    async fn writable(
        &self,
        options: DuckLakeWriteOptions,
    ) -> (SessionContext, Arc<DuckLakeCatalog>) {
        let provider = self.provider().await;
        let writer = SqliteMetadataWriter::new(&self.connection).await.unwrap();
        let catalog = Arc::new(
            DuckLakeCatalog::with_writer(Arc::new(provider), Arc::new(writer))
                .unwrap()
                .with_write_options(options),
        );
        let ctx = SessionContext::new();
        ctx.register_catalog(
            "ducklake",
            Arc::clone(&catalog) as Arc<dyn datafusion::catalog::CatalogProvider>,
        );
        ctx.register_table("source", source_table()).unwrap();
        (ctx, catalog)
    }

    /// Rows from a freshly opened read-only catalog, so they reflect the committed head.
    async fn query(&self, sql: &str) -> Vec<RecordBatch> {
        let catalog = DuckLakeCatalog::new(self.provider().await).unwrap();
        let ctx = SessionContext::new();
        ctx.register_catalog("ducklake", Arc::new(catalog));
        ctx.sql(sql).await.unwrap().collect().await.unwrap()
    }
}

async fn lake() -> Lake {
    let temp = TempDir::new().unwrap();
    let data_path = temp.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();
    let connection = format!("sqlite:{}?mode=rwc", temp.path().join("test.db").display());
    let writer = SqliteMetadataWriter::new_with_init(&connection)
        .await
        .unwrap();
    writer.set_data_path(data_path.to_str().unwrap()).unwrap();
    let snapshot = writer.create_snapshot().unwrap();
    writer.get_or_create_schema("main", None, snapshot).unwrap();
    Lake {
        temp,
        connection,
    }
}

fn source_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true),
        Field::new("score", DataType::Float64, true),
    ]))
}

fn source_batch(ids: &[i32], names: &[Option<&str>], scores: &[Option<f64>]) -> RecordBatch {
    RecordBatch::try_new(
        source_schema(),
        vec![
            Arc::new(Int32Array::from(ids.to_vec())),
            Arc::new(StringArray::from(names.to_vec())),
            Arc::new(Float64Array::from(scores.to_vec())),
        ],
    )
    .unwrap()
}

/// Two partitions of three rows each, so a CTAS input spans partitions.
fn source_table() -> Arc<MemTable> {
    let first = source_batch(
        &[1, 2, 3],
        &[Some("Alice"), Some("Bob"), None],
        &[Some(1.5), Some(2.5), Some(3.5)],
    );
    let second = source_batch(
        &[4, 5, 6],
        &[Some("Alice"), Some("Bob"), Some("Alice")],
        &[Some(4.5), None, Some(6.5)],
    );
    Arc::new(MemTable::try_new(source_schema(), vec![vec![first], vec![second]]).unwrap())
}

fn int32_column(batches: &[RecordBatch], index: usize) -> Vec<Option<i32>> {
    batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(index)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .iter()
                .collect::<Vec<_>>()
        })
        .collect()
}

fn int64_column(batches: &[RecordBatch], index: usize) -> Vec<Option<i64>> {
    batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(index)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .iter()
                .collect::<Vec<_>>()
        })
        .collect()
}

/// String values cast to `Utf8`, since the DuckDB provider reads text as `Utf8View`.
fn string_column(batches: &[RecordBatch], index: usize) -> Vec<Option<String>> {
    batches
        .iter()
        .flat_map(|batch| {
            arrow::compute::cast(batch.column(index), &DataType::Utf8)
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .iter()
                .map(|value| value.map(str::to_string))
                .collect::<Vec<_>>()
        })
        .collect()
}

fn float64_column(batches: &[RecordBatch], index: usize) -> Vec<Option<f64>> {
    batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(index)
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap()
                .iter()
                .collect::<Vec<_>>()
        })
        .collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn ctas_commits_filtered_rows_and_nullable_columns_in_one_snapshot() {
    let lake = lake().await;
    let (ctx, catalog) = lake.writable(DuckLakeWriteOptions::default()).await;
    let head = lake.head().await;

    let result = execute_ducklake_sql(
        &ctx,
        &catalog,
        "CREATE TABLE ducklake.main.picked AS \
         SELECT id, name, score FROM source WHERE id <> 2 ORDER BY id",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();

    assert_eq!(result.iter().map(RecordBatch::num_rows).sum::<usize>(), 0);
    assert_eq!(lake.head().await, head + 1);
    let table_id = lake.table_id("picked").await.unwrap();
    let changes = lake.provider().await.list_snapshot_changes().unwrap();
    let last = changes.last().unwrap();
    assert_eq!(last.snapshot_id, head + 1);
    assert_eq!(
        last.changes_made.as_deref(),
        Some(format!("created_table:\"main\".\"picked\",inlined_insert:{table_id}").as_str())
    );
    assert_eq!(
        lake.columns("picked").await,
        vec![
            ("id".to_string(), "int32".to_string(), true),
            ("name".to_string(), "varchar".to_string(), true),
            ("score".to_string(), "float64".to_string(), true),
        ]
    );
    assert_eq!(lake.data_file_count("picked").await, 0);

    let rows = lake
        .query("SELECT id, name, score FROM ducklake.main.picked ORDER BY id")
        .await;
    assert_eq!(
        int32_column(&rows, 0),
        vec![Some(1), Some(3), Some(4), Some(5), Some(6)]
    );
    assert_eq!(
        string_column(&rows, 1),
        vec![
            Some("Alice".to_string()),
            None,
            Some("Alice".to_string()),
            Some("Bob".to_string()),
            Some("Alice".to_string()),
        ]
    );
    assert_eq!(
        float64_column(&rows, 2),
        vec![Some(1.5), Some(3.5), Some(4.5), None, Some(6.5)]
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn ctas_commits_aggregated_rows_across_input_partitions() {
    let lake = lake().await;
    let (ctx, catalog) = lake.writable(DuckLakeWriteOptions::default()).await;

    execute_ducklake_sql(
        &ctx,
        &catalog,
        "CREATE TABLE ducklake.main.totals AS \
         SELECT name, count(*) AS n, sum(score) AS total FROM source GROUP BY name",
    )
    .await
    .unwrap();

    assert_eq!(
        lake.columns("totals").await,
        vec![
            ("name".to_string(), "varchar".to_string(), true),
            ("n".to_string(), "int64".to_string(), true),
            ("total".to_string(), "float64".to_string(), true),
        ]
    );
    let rows = lake
        .query("SELECT name, n, total FROM ducklake.main.totals ORDER BY name NULLS FIRST")
        .await;
    assert_eq!(
        string_column(&rows, 0),
        vec![None, Some("Alice".to_string()), Some("Bob".to_string())]
    );
    assert_eq!(int64_column(&rows, 1), vec![Some(1), Some(3), Some(2)]);
    assert_eq!(
        float64_column(&rows, 2),
        vec![Some(3.5), Some(12.5), Some(2.5)]
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn ctas_with_empty_result_publishes_columns_and_no_data_file() {
    let lake = lake().await;
    let (ctx, catalog) = lake.writable(DuckLakeWriteOptions::default()).await;
    let head = lake.head().await;

    execute_ducklake_sql(
        &ctx,
        &catalog,
        "CREATE TABLE ducklake.main.none AS SELECT id, name FROM source WHERE id > 100",
    )
    .await
    .unwrap();

    assert_eq!(lake.head().await, head + 1);
    assert_eq!(
        lake.columns("none").await,
        vec![
            ("id".to_string(), "int32".to_string(), true),
            ("name".to_string(), "varchar".to_string(), true),
        ]
    );
    assert_eq!(lake.data_file_count("none").await, 0);
    let rows = lake
        .query("SELECT count(*) AS n FROM ducklake.main.none")
        .await;
    assert_eq!(int64_column(&rows, 0), vec![Some(0)]);
}

#[tokio::test(flavor = "multi_thread")]
async fn ctas_applies_catalog_write_options() {
    let lake = lake().await;
    let (ctx, catalog) = lake
        .writable(DuckLakeWriteOptions::default().with_data_inlining_row_limit(0))
        .await;

    execute_ducklake_sql(
        &ctx,
        &catalog,
        "CREATE TABLE ducklake.main.parquet AS SELECT id FROM source WHERE id <= 2",
    )
    .await
    .unwrap();

    assert_eq!(lake.data_file_count("parquet").await, 1);
    let rows = lake
        .query("SELECT id FROM ducklake.main.parquet ORDER BY id")
        .await;
    assert_eq!(int32_column(&rows, 0), vec![Some(1), Some(2)]);
}

#[tokio::test(flavor = "multi_thread")]
async fn ctas_rejects_existing_table_and_if_not_exists_keeps_it() {
    let lake = lake().await;
    let (ctx, catalog) = lake.writable(DuckLakeWriteOptions::default()).await;
    execute_ducklake_sql(
        &ctx,
        &catalog,
        "CREATE TABLE ducklake.main.kept AS SELECT id FROM source WHERE id = 1",
    )
    .await
    .unwrap();
    let head = lake.head().await;

    let error = execute_ducklake_sql(
        &ctx,
        &catalog,
        "CREATE TABLE ducklake.main.kept AS SELECT id FROM source WHERE id = 2",
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        error.contains("Cannot create table 'kept': a table with that name already exists"),
        "{error}"
    );

    execute_ducklake_sql(
        &ctx,
        &catalog,
        "CREATE TABLE IF NOT EXISTS ducklake.main.kept AS SELECT id FROM source WHERE id = 2",
    )
    .await
    .unwrap();

    assert_eq!(lake.head().await, head);
    let rows = lake.query("SELECT id FROM ducklake.main.kept").await;
    assert_eq!(int32_column(&rows, 0), vec![Some(1)]);
}

#[tokio::test(flavor = "multi_thread")]
async fn ctas_rejects_unsupported_forms_without_touching_the_catalog() {
    let lake = lake().await;
    let (ctx, catalog) = lake.writable(DuckLakeWriteOptions::default()).await;
    let head = lake.head().await;

    for (sql, expected) in [
        (
            "CREATE OR REPLACE TABLE ducklake.main.t AS SELECT id FROM source",
            "CREATE OR REPLACE TABLE AS SELECT is not supported; DROP TABLE first",
        ),
        (
            "CREATE OR REPLACE TABLE ducklake.main.t (p) AS SELECT id FROM source",
            "CREATE OR REPLACE TABLE AS SELECT is not supported; DROP TABLE first",
        ),
        (
            "CREATE TABLE ducklake.main.t (a INT) AS SELECT id FROM source",
            "CREATE TABLE AS SELECT does not accept column types",
        ),
        (
            "CREATE TABLE ducklake.main.t (p, q, r, s) AS SELECT id, name, score FROM source",
            "CREATE TABLE AS SELECT lists more column names than the query returns",
        ),
        (
            "CREATE TABLE ducklake.main.t (p, \"P\") AS SELECT id, name FROM source",
            "CREATE TABLE AS SELECT produced duplicate column name 'P'",
        ),
        (
            "CREATE TABLE ducklake.main.t (name) AS SELECT id, name FROM source",
            "CREATE TABLE AS SELECT produced duplicate column name 'name'",
        ),
        (
            "CREATE TABLE ducklake.missing.t AS SELECT id FROM source",
            "schema 'missing' not found",
        ),
        (
            "CREATE TABLE ducklake.main.\"../t\" AS SELECT id FROM source",
            "must not contain path separators",
        ),
    ] {
        let error = execute_ducklake_sql(&ctx, &catalog, sql)
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains(expected), "{sql}: {error}");
    }

    let read_only = Arc::new(DuckLakeCatalog::new(lake.provider().await).unwrap());
    let read_only_ctx = SessionContext::new();
    read_only_ctx.register_catalog(
        "ducklake",
        Arc::clone(&read_only) as Arc<dyn datafusion::catalog::CatalogProvider>,
    );
    read_only_ctx
        .register_table("source", source_table())
        .unwrap();
    let error = execute_ducklake_sql(
        &read_only_ctx,
        &read_only,
        "CREATE TABLE ducklake.main.t AS SELECT id FROM source",
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(error.contains("catalog is read-only"), "{error}");

    assert_eq!(lake.head().await, head);
    assert_eq!(lake.table_id("t").await, None);
}

#[tokio::test(flavor = "multi_thread")]
async fn ctas_write_failure_leaves_the_catalog_head_and_table_list_unchanged() {
    let lake = lake().await;
    // Inlined rows never reach the data directory, so write Parquet.
    let (ctx, catalog) = lake
        .writable(DuckLakeWriteOptions::default().with_data_inlining_row_limit(0))
        .await;
    let head = lake.head().await;

    let file_url = url::Url::parse("file:///").unwrap();
    ctx.register_object_store(&file_url, unwritable_store(lake.temp.path()));
    let outcome = execute_ducklake_sql(
        &ctx,
        &catalog,
        "CREATE TABLE ducklake.main.failed AS SELECT id FROM source",
    )
    .await;
    ctx.register_object_store(&file_url, Arc::new(LocalFileSystem::new()));

    assert!(outcome.is_err(), "the upload fails");
    assert_eq!(lake.head().await, head);
    assert_eq!(lake.table_rows("failed").await, 0);

    // The next commit takes the snapshot id the failed write had reserved.
    execute_ducklake_sql(
        &ctx,
        &catalog,
        "CREATE TABLE ducklake.main.later AS SELECT id FROM source WHERE id = 6",
    )
    .await
    .unwrap();
    assert_eq!(lake.head().await, head + 1);
    assert_eq!(lake.table_id("failed").await, None);
    let names = lake
        .query("SELECT table_name FROM ducklake.information_schema.tables ORDER BY table_name")
        .await;
    assert_eq!(string_column(&names, 0), vec![Some("later".to_string())]);

    // Retrying the same statement creates the table.
    execute_ducklake_sql(
        &ctx,
        &catalog,
        "CREATE TABLE ducklake.main.failed AS SELECT id FROM source WHERE id <= 2",
    )
    .await
    .unwrap();
    assert_eq!(lake.head().await, head + 2);
    let rows = lake
        .query("SELECT id FROM ducklake.main.failed ORDER BY id")
        .await;
    assert_eq!(int32_column(&rows, 0), vec![Some(1), Some(2)]);
    let names = lake
        .query("SELECT table_name FROM ducklake.information_schema.tables ORDER BY table_name")
        .await;
    assert_eq!(
        string_column(&names, 0),
        vec![Some("failed".to_string()), Some("later".to_string())]
    );
    assert_eq!(lake.table_rows("failed").await, 1);
}

/// Column names follow DuckDB's CTAS: a repeated name, compared ignoring case,
/// takes the first free `_1`, `_2`, … suffix, and a column list renames the
/// leading columns.
#[tokio::test(flavor = "multi_thread")]
async fn ctas_names_columns_as_duckdb_does() {
    let lake = lake().await;
    let (ctx, catalog) = lake.writable(DuckLakeWriteOptions::default()).await;

    for (table, sql, names) in [
        (
            "joined",
            "CREATE TABLE ducklake.main.joined AS \
             SELECT * FROM (SELECT 1 AS id) x, (SELECT 2 AS id) y",
            vec!["id", "id_1"],
        ),
        (
            "cased",
            "CREATE TABLE ducklake.main.cased AS SELECT 1 AS a, 2 AS \"A\"",
            vec!["a", "A_1"],
        ),
        (
            "suffixed",
            "CREATE TABLE ducklake.main.suffixed AS \
             SELECT * FROM (SELECT 1 AS a) x, (SELECT 2 AS a) y, (SELECT 3 AS a_1) z",
            vec!["a", "a_1", "a_1_1"],
        ),
        (
            "listed",
            "CREATE TABLE ducklake.main.listed (p, \"Q\") AS SELECT 1 AS x, 2 AS y",
            vec!["p", "Q"],
        ),
        (
            "partial",
            "CREATE TABLE ducklake.main.partial (p) AS SELECT 1 AS x, 2 AS y",
            vec!["p", "y"],
        ),
    ] {
        execute_ducklake_sql(&ctx, &catalog, sql).await.unwrap();
        let columns = lake.columns(table).await;
        assert_eq!(
            columns
                .iter()
                .map(|(name, _, _)| name.as_str())
                .collect::<Vec<_>>(),
            names,
            "{sql}"
        );
    }
    let rows = lake
        .query("SELECT id, id_1 FROM ducklake.main.joined")
        .await;
    assert_eq!(int64_column(&rows, 0), vec![Some(1)]);
    assert_eq!(int64_column(&rows, 1), vec![Some(2)]);
}

/// `IF NOT EXISTS` over a view's name is a no-op, as in official DuckLake.
#[tokio::test(flavor = "multi_thread")]
async fn ctas_if_not_exists_on_a_view_name_is_a_no_op() {
    let lake = lake().await;
    let (ctx, catalog) = lake.writable(DuckLakeWriteOptions::default()).await;
    let head = lake.head().await;
    let provider = lake.provider().await;
    let schema_id = provider
        .get_schema_by_name("main", head)
        .unwrap()
        .unwrap()
        .schema_id;
    let pool = sqlx::SqlitePool::connect(&lake.connection).await.unwrap();
    sqlx::query(
        "INSERT INTO ducklake_view
             (view_id, schema_id, view_name, dialect, sql, column_aliases, begin_snapshot)
         VALUES (100, ?, 'v', 'duckdb', 'SELECT 1', '', ?)",
    )
    .bind(schema_id)
    .bind(head)
    .execute(&pool)
    .await
    .unwrap();

    execute_ducklake_sql(
        &ctx,
        &catalog,
        "CREATE TABLE IF NOT EXISTS ducklake.main.v AS SELECT id FROM source",
    )
    .await
    .unwrap();
    let error = execute_ducklake_sql(
        &ctx,
        &catalog,
        "CREATE TABLE ducklake.main.v AS SELECT id FROM source",
    )
    .await
    .unwrap_err()
    .to_string();

    assert!(
        error.contains("a view with that name already exists"),
        "{error}"
    );
    assert_eq!(lake.head().await, head);
    assert_eq!(lake.table_rows("v").await, 0);
}

/// A CTAS resolves its name with the session's default catalog and schema, and
/// writes only when that names the DuckLake catalog it runs against; any other
/// goes to DataFusion.
#[tokio::test(flavor = "multi_thread")]
async fn ctas_writes_only_to_the_catalog_its_name_resolves_to() {
    let lake = lake().await;
    let (ctx, catalog) = lake.writable(DuckLakeWriteOptions::default()).await;
    let other = self::lake().await;
    let (_, other_catalog) = other.writable(DuckLakeWriteOptions::default()).await;
    ctx.register_catalog(
        "other",
        Arc::clone(&other_catalog) as Arc<dyn datafusion::catalog::CatalogProvider>,
    );
    let (head, other_head) = (lake.head().await, other.head().await);

    execute_ducklake_sql(
        &ctx,
        &catalog,
        "CREATE TABLE scratch AS SELECT id FROM source WHERE id <= 2",
    )
    .await
    .unwrap();
    let scratch = ctx
        .sql("SELECT id FROM datafusion.public.scratch ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let error = execute_ducklake_sql(
        &ctx,
        &catalog,
        "CREATE TABLE other.main.t AS SELECT id FROM source",
    )
    .await
    .unwrap_err()
    .to_string();
    let unchanged = (lake.head().await, other.head().await);
    execute_ducklake_sql(
        &ctx,
        &other_catalog,
        "CREATE TABLE other.main.t AS SELECT id FROM source WHERE id = 6",
    )
    .await
    .unwrap();

    let config = SessionConfig::new().with_default_catalog_and_schema("ducklake", "main");
    let defaulted_ctx = SessionContext::new_with_config(config);
    defaulted_ctx.register_catalog(
        "ducklake",
        Arc::clone(&catalog) as Arc<dyn datafusion::catalog::CatalogProvider>,
    );
    execute_ducklake_sql(
        &defaulted_ctx,
        &catalog,
        "CREATE TABLE defaulted AS SELECT column1 AS id FROM (VALUES (1), (2), (3))",
    )
    .await
    .unwrap();

    assert_eq!(int32_column(&scratch, 0), vec![Some(1), Some(2)]);
    assert!(
        error.contains("run it through execute_ducklake_sql"),
        "{error}"
    );
    assert_eq!(unchanged, (head, other_head));
    assert_eq!(other.head().await, other_head + 1);
    assert_eq!(lake.head().await, head + 1);
    assert_eq!(lake.table_id("scratch").await, None);
    assert_eq!(lake.table_id("t").await, None);
    let rows = other.query("SELECT id FROM ducklake.main.t").await;
    assert_eq!(int32_column(&rows, 0), vec![Some(6)]);
    let rows = lake
        .query("SELECT id FROM ducklake.main.defaulted ORDER BY id")
        .await;
    assert_eq!(int64_column(&rows, 0), vec![Some(1), Some(2), Some(3)]);
}

/// With no stored or catalog-configured limit, a CTAS takes the metadata
/// writer's own `data_inlining_row_limit`, as `INSERT` does.
#[tokio::test(flavor = "multi_thread")]
async fn ctas_applies_the_writers_inlining_limit() {
    let lake = lake().await;
    let writer = SqliteMetadataWriter::new(&lake.connection)
        .await
        .unwrap()
        .with_data_inlining_row_limit(0);
    let catalog = Arc::new(
        DuckLakeCatalog::with_writer(Arc::new(lake.provider().await), Arc::new(writer)).unwrap(),
    );
    let ctx = SessionContext::new();
    ctx.register_catalog(
        "ducklake",
        Arc::clone(&catalog) as Arc<dyn datafusion::catalog::CatalogProvider>,
    );
    ctx.register_table("source", source_table()).unwrap();

    execute_ducklake_sql(
        &ctx,
        &catalog,
        "CREATE TABLE ducklake.main.small AS SELECT id FROM source WHERE id <= 2",
    )
    .await
    .unwrap();

    assert_eq!(lake.data_file_count("small").await, 1);
    let rows = lake
        .query("SELECT id FROM ducklake.main.small ORDER BY id")
        .await;
    assert_eq!(int32_column(&rows, 0), vec![Some(1), Some(2)]);
}

/// Every batch of a multi-partition input reaches the Parquet file.
#[tokio::test(flavor = "multi_thread")]
async fn ctas_writes_every_input_batch_to_parquet() {
    let lake = lake().await;
    let (ctx, catalog) = lake
        .writable(DuckLakeWriteOptions::default().with_data_inlining_row_limit(0))
        .await;

    execute_ducklake_sql(
        &ctx,
        &catalog,
        "CREATE TABLE ducklake.main.everything AS SELECT * FROM source",
    )
    .await
    .unwrap();

    assert_eq!(lake.data_file_count("everything").await, 1);
    let rows = lake
        .query("SELECT id, name, score FROM ducklake.main.everything ORDER BY id")
        .await;
    assert_eq!(
        int32_column(&rows, 0),
        vec![Some(1), Some(2), Some(3), Some(4), Some(5), Some(6)]
    );
    assert_eq!(
        string_column(&rows, 1),
        vec![
            Some("Alice".to_string()),
            Some("Bob".to_string()),
            None,
            Some("Alice".to_string()),
            Some("Bob".to_string()),
            Some("Alice".to_string()),
        ]
    );
    assert_eq!(
        float64_column(&rows, 2),
        vec![Some(1.5), Some(2.5), Some(3.5), Some(4.5), None, Some(6.5)]
    );
}

/// A source whose nested children are all non-null: a struct field, a list
/// element, and a map's key and value.
fn nested_source() -> Arc<MemTable> {
    let point = Fields::from(vec![Field::new("x", DataType::Int32, false)]);
    let entries = Fields::from(vec![
        Field::new("key", DataType::Utf8, false),
        Field::new("value", DataType::Int32, false),
    ]);
    let schema = Arc::new(Schema::new(vec![
        Field::new("point", DataType::Struct(point.clone()), false),
        Field::new(
            "items",
            DataType::List(Arc::new(Field::new("item", DataType::Int32, false))),
            false,
        ),
        Field::new(
            "tags",
            DataType::Map(
                Arc::new(Field::new(
                    "entries",
                    DataType::Struct(entries.clone()),
                    false,
                )),
                false,
            ),
            false,
        ),
    ]));
    let points = StructArray::new(point, vec![Arc::new(Int32Array::from(vec![1, 2]))], None);
    let items = ListArray::new(
        Arc::new(Field::new("item", DataType::Int32, false)),
        OffsetBuffer::from_lengths([2, 1]),
        Arc::new(Int32Array::from(vec![3, 4, 5])),
        None,
    );
    let tags = MapArray::new(
        Arc::new(Field::new(
            "entries",
            DataType::Struct(entries.clone()),
            false,
        )),
        OffsetBuffer::from_lengths([1, 0]),
        StructArray::new(
            entries,
            vec![Arc::new(StringArray::from(vec!["a"])), Arc::new(Int32Array::from(vec![6]))],
            None,
        ),
        None,
        false,
    );
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(points), Arc::new(items), Arc::new(tags)],
    )
    .unwrap();
    Arc::new(MemTable::try_new(schema, vec![vec![batch]]).unwrap())
}

/// Every column and nested child of a CTAS table is nullable, as official
/// records them, whether its rows go to Parquet or are inlined.
#[rstest]
#[case::parquet(0)]
#[case::inlined(10)]
#[tokio::test(flavor = "multi_thread")]
async fn ctas_makes_every_nested_child_nullable(#[case] inlining_limit: usize) {
    let lake = lake().await;
    let (ctx, catalog) = lake
        .writable(DuckLakeWriteOptions::default().with_data_inlining_row_limit(inlining_limit))
        .await;
    ctx.register_table("nested", nested_source()).unwrap();

    execute_ducklake_sql(
        &ctx,
        &catalog,
        "CREATE TABLE ducklake.main.nested AS SELECT * FROM nested",
    )
    .await
    .unwrap();

    let table_id = lake.table_id("nested").await.unwrap();
    let pool = sqlx::SqlitePool::connect(&lake.connection).await.unwrap();
    let columns: Vec<(String, bool)> = sqlx::query_as(
        "SELECT column_name, nulls_allowed FROM ducklake_column
         WHERE table_id = ? ORDER BY column_id",
    )
    .bind(table_id)
    .fetch_all(&pool)
    .await
    .unwrap();
    let rows = lake
        .query(
            "SELECT point['x'], items[1], items[2], CAST(cardinality(tags) AS INT) \
             FROM ducklake.main.nested ORDER BY point['x']",
        )
        .await;

    assert_eq!(
        columns,
        ["point", "x", "items", "element", "tags", "key", "value"]
            .into_iter()
            .map(|name| (name.to_string(), true))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        lake.data_file_count("nested").await,
        usize::from(inlining_limit == 0)
    );
    assert_eq!(int32_column(&rows, 0), vec![Some(1), Some(2)]);
    assert_eq!(int32_column(&rows, 1), vec![Some(3), Some(5)]);
    assert_eq!(int32_column(&rows, 2), vec![Some(4), None]);
    assert_eq!(int32_column(&rows, 3), vec![Some(1), Some(0)]);
}

/// A predicate over `source.id` that, the first time it runs, creates an empty
/// `raced` with the CTAS's `id` column through `writer`, as another session
/// would, and then keeps every row or none.
fn rival_udf(writer: Arc<dyn MetadataWriter>, keep: bool) -> ScalarUDF {
    let created = Arc::new(AtomicBool::new(false));
    create_udf(
        "rival",
        vec![DataType::Int32],
        DataType::Boolean,
        Volatility::Volatile,
        Arc::new(move |args: &[ColumnarValue]| {
            if !created.swap(true, Ordering::SeqCst) {
                let columns = vec![ColumnDef::new("id", "int32", true).unwrap()];
                let setup = writer
                    .begin_write_transaction("main", "raced", &columns, WriteMode::Replace)
                    .unwrap();
                writer
                    .publish_snapshot(
                        setup.table_id,
                        "main",
                        "raced",
                        setup.snapshot_id,
                        WriteMode::Replace,
                        setup.base_snapshot_id,
                        &columns,
                        &setup.field_ids,
                    )
                    .unwrap();
            }
            let rows = match &args[0] {
                ColumnarValue::Array(array) => array.len(),
                ColumnarValue::Scalar(_) => 1,
            };
            Ok(ColumnarValue::Array(Arc::new(BooleanArray::from(vec![
                keep;
                rows
            ]))))
        }),
    )
}

/// A table another session creates under the name while the query runs makes
/// the CTAS conflict, with an empty result and with rows, instead of replacing
/// it, as official DuckLake refuses the create.
#[tokio::test(flavor = "multi_thread")]
async fn ctas_conflicts_with_a_table_created_while_its_query_runs() {
    for keep in [false, true] {
        let lake = lake().await;
        let (ctx, catalog) = lake
            .writable(DuckLakeWriteOptions::default().with_data_inlining_row_limit(0))
            .await;
        let writer = SqliteMetadataWriter::new(&lake.connection).await.unwrap();
        ctx.register_udf(rival_udf(Arc::new(writer), keep));
        let head = lake.head().await;

        let error = execute_ducklake_sql(
            &ctx,
            &catalog,
            "CREATE TABLE ducklake.main.raced AS SELECT id FROM source WHERE rival(id)",
        )
        .await
        .unwrap_err()
        .to_string();

        let rival_id = lake.table_id("raced").await.unwrap();
        let expected = if keep {
            format!(
                "table {rival_id} was created by another transaction after snapshot {head}; \
                 retry the write"
            )
        } else {
            "table \"main\".\"raced\" was created by another transaction; retry the statement"
                .to_string()
        };
        assert!(error.ends_with(&expected), "keep {keep}: {error}");
        assert_eq!(lake.head().await, head + 1, "keep {keep}");
        assert_eq!(lake.data_file_count("raced").await, 0, "keep {keep}");
        assert_eq!(lake.table_rows("raced").await, 1, "keep {keep}");
    }
}

/// A store whose every write fails: its root is a regular file, so no object
/// can be created under it, whatever the user's permissions.
fn unwritable_store(dir: &std::path::Path) -> Arc<dyn ObjectStore> {
    let root = dir.join("not-a-directory");
    std::fs::write(&root, b"").unwrap();
    Arc::new(LocalFileSystem::new_with_prefix(root).unwrap())
}

#[cfg(all(feature = "write-duckdb", feature = "metadata-duckdb"))]
mod duckdb_backend {
    use super::*;
    use datafusion_ducklake::DuckdbMetadataWriter;

    /// A writable DuckDB-backend catalog registered as `ducklake` next to the
    /// in-memory `source` table.
    fn duckdb_lake(temp: &TempDir) -> (SessionContext, Arc<DuckLakeCatalog>, String) {
        let database = temp.path().join("catalog.ducklake");
        let database = database.to_str().unwrap().to_string();
        let data_path = temp.path().join("data");
        std::fs::create_dir_all(&data_path).unwrap();
        let writer = DuckdbMetadataWriter::new_with_init(&database).unwrap();
        writer.set_data_path(data_path.to_str().unwrap()).unwrap();
        let snapshot = writer.create_snapshot().unwrap();
        writer.get_or_create_schema("main", None, snapshot).unwrap();
        let provider = writer.metadata_provider();
        let catalog = Arc::new(
            DuckLakeCatalog::with_writer(Arc::new(provider), Arc::new(writer))
                .unwrap()
                .with_write_options(
                    DuckLakeWriteOptions::default().with_data_inlining_row_limit(0),
                ),
        );
        let ctx = SessionContext::new();
        ctx.register_catalog(
            "ducklake",
            Arc::clone(&catalog) as Arc<dyn datafusion::catalog::CatalogProvider>,
        );
        ctx.register_table("source", source_table()).unwrap();
        (ctx, catalog, database)
    }

    /// The DuckDB writer commits a Parquet-backed CTAS, and DuckDB itself reads
    /// the rows and the nullable columns back from the catalog.
    #[tokio::test(flavor = "multi_thread")]
    async fn ctas_on_the_duckdb_writer_commits_rows_duckdb_reads_back() {
        let temp = TempDir::new().unwrap();
        let (ctx, catalog, database) = duckdb_lake(&temp);
        let head = catalog.provider().get_current_snapshot().unwrap();

        execute_ducklake_sql(
            &ctx,
            &catalog,
            "CREATE TABLE ducklake.main.picked AS SELECT id, name FROM source WHERE id > 4",
        )
        .await
        .unwrap();

        let provider = catalog.provider();
        assert_eq!(provider.get_current_snapshot().unwrap(), head + 1);
        // The writable catalog pins the snapshot it was opened at, so read through
        // one opened at the new head.
        let reader = DuckLakeCatalog::with_snapshot(Arc::clone(&provider), head + 1).unwrap();
        ctx.register_catalog("ducklake", Arc::new(reader));
        let rows = ctx
            .sql("SELECT id, name FROM ducklake.main.picked ORDER BY id")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        assert_eq!(int32_column(&rows, 0), vec![Some(5), Some(6)]);
        assert_eq!(
            string_column(&rows, 1),
            vec![Some("Bob".to_string()), Some("Alice".to_string())]
        );

        // DuckDB allows one client per catalog file, so release the writer first.
        drop(rows);
        drop(ctx);
        drop(provider);
        drop(catalog);
        crate::common::ensure_ducklake_installed();
        let oracle = duckdb::Connection::open_in_memory().unwrap();
        oracle
            .execute_batch(&format!(
                "LOAD ducklake; ATTACH 'ducklake:{database}' AS lake (READ_ONLY);"
            ))
            .unwrap();
        let mut statement = oracle
            .prepare("SELECT id, name FROM lake.main.picked ORDER BY id")
            .unwrap();
        let picked = statement
            .query_map([], |row| {
                Ok((row.get::<_, i32>(0)?, row.get::<_, String>(1)?))
            })
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(
            picked,
            vec![(5, "Bob".to_string()), (6, "Alice".to_string())]
        );
        let mut statement = oracle
            .prepare(
                "SELECT column_name, column_type, nulls_allowed \
                 FROM __ducklake_metadata_lake.ducklake_column \
                 WHERE end_snapshot IS NULL ORDER BY column_order",
            )
            .unwrap();
        let columns = statement
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, bool>(2)?,
                ))
            })
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(
            columns,
            vec![
                ("id".to_string(), "int32".to_string(), true),
                ("name".to_string(), "varchar".to_string(), true),
            ]
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn ctas_on_the_duckdb_writer_leaves_no_row_for_a_failed_table() {
        let temp = TempDir::new().unwrap();
        let (ctx, catalog, database) = duckdb_lake(&temp);
        let provider = catalog.provider();
        let head = provider.get_current_snapshot().unwrap();
        let schema_id = provider
            .get_schema_by_name("main", head)
            .unwrap()
            .unwrap()
            .schema_id;

        let file_url = url::Url::parse("file:///").unwrap();
        ctx.register_object_store(&file_url, unwritable_store(temp.path()));
        let outcome = execute_ducklake_sql(
            &ctx,
            &catalog,
            "CREATE TABLE ducklake.main.failed AS SELECT id FROM source",
        )
        .await;
        assert!(outcome.is_err(), "the upload fails");
        assert_eq!(provider.get_current_snapshot().unwrap(), head);
        assert!(
            provider
                .get_table_by_name(schema_id, "failed", head)
                .unwrap()
                .is_none()
        );

        // Official DuckDB then creates the same name over the catalog: the
        // failed write left no row for it to collide with.
        drop(ctx);
        drop(provider);
        drop(catalog);
        crate::common::ensure_ducklake_installed();
        let oracle = duckdb::Connection::open_in_memory().unwrap();
        oracle
            .execute_batch(&format!(
                "LOAD ducklake; ATTACH 'ducklake:{database}' AS lake;
                 CREATE TABLE lake.main.failed AS SELECT 7 AS x;"
            ))
            .unwrap();
        let rows: i64 = oracle
            .query_row(
                "SELECT COUNT(*) FROM __ducklake_metadata_lake.ducklake_table
                 WHERE table_name = 'failed'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        let x: i32 = oracle
            .query_row("SELECT x FROM lake.main.failed", [], |row| row.get(0))
            .unwrap();
        assert_eq!(rows, 1);
        assert_eq!(x, 7);
    }

    /// Of two writes creating one table on the DuckDB writer, the second to
    /// commit conflicts instead of writing into the first's table.
    #[tokio::test(flavor = "multi_thread")]
    async fn second_create_of_a_table_conflicts_on_the_duckdb_writer() {
        let temp = TempDir::new().unwrap();
        let (_ctx, catalog, _database) = duckdb_lake(&temp);
        let writer = catalog.writer().unwrap();
        let columns = vec![ColumnDef::new("id", "int32", true).unwrap()];
        let first = writer
            .begin_write_transaction("main", "t", &columns, WriteMode::Replace)
            .unwrap();
        let second = writer
            .begin_write_transaction("main", "t", &columns, WriteMode::Replace)
            .unwrap();
        let publish = |setup: &datafusion_ducklake::WriteSetupResult| {
            writer.publish_snapshot(
                setup.table_id,
                "main",
                "t",
                setup.snapshot_id,
                WriteMode::Replace,
                setup.base_snapshot_id,
                &columns,
                &setup.field_ids,
            )
        };

        let committed = publish(&first).unwrap();
        let error = publish(&second).unwrap_err();

        assert_ne!(first.table_id, second.table_id);
        assert_eq!(committed.table_id, first.table_id);
        assert!(
            matches!(&error, datafusion_ducklake::DuckLakeError::Conflict(message)
                if message == "table \"main\".\"t\" was created by another transaction; retry the write"),
            "{error}"
        );
    }
}
