//! Integration tests for `CREATE TABLE AS SELECT` through `execute_ducklake_sql`.

#![cfg(all(feature = "write-sqlite", feature = "metadata-sqlite"))]

use std::sync::Arc;

use arrow::array::{Array, Float64Array, Int32Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use datafusion::datasource::MemTable;
use datafusion::prelude::*;
use tempfile::TempDir;

use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeWriteOptions, MetadataProvider, MetadataWriter,
    SqliteMetadataProvider, SqliteMetadataWriter, execute_ducklake_sql,
};

struct Lake {
    temp: TempDir,
    connection: String,
}

impl Lake {
    fn data_path(&self) -> std::path::PathBuf {
        self.temp.path().join("data")
    }

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
        Some(format!("created_table:\"main\".\"picked\",inserted_into_table:{table_id}").as_str())
    );
    assert_eq!(
        lake.columns("picked").await,
        vec![
            ("id".to_string(), "int32".to_string(), true),
            ("name".to_string(), "varchar".to_string(), true),
            ("score".to_string(), "float64".to_string(), true),
        ]
    );
    assert_eq!(lake.data_file_count("picked").await, 1);

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
        .writable(DuckLakeWriteOptions::default().with_data_inlining_row_limit(10))
        .await;

    execute_ducklake_sql(
        &ctx,
        &catalog,
        "CREATE TABLE ducklake.main.inlined AS SELECT id FROM source WHERE id <= 2",
    )
    .await
    .unwrap();

    assert_eq!(lake.data_file_count("inlined").await, 0);
    let rows = lake
        .query("SELECT id FROM ducklake.main.inlined ORDER BY id")
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
            "CREATE TABLE ducklake.main.t (a INT) AS SELECT id FROM source",
            "CREATE TABLE AS SELECT does not accept a column list",
        ),
        (
            "CREATE TABLE ducklake.main.t AS SELECT * FROM source a JOIN source b ON a.id = b.id",
            "CREATE TABLE AS SELECT produced duplicate column name 'id'",
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

    let read_only = DuckLakeCatalog::new(lake.provider().await).unwrap();
    let error = execute_ducklake_sql(
        &ctx,
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
    let (ctx, catalog) = lake.writable(DuckLakeWriteOptions::default()).await;
    let head = lake.head().await;

    let data_path = lake.data_path();
    let writable = std::fs::metadata(&data_path).unwrap().permissions();
    let mut read_only = writable.clone();
    read_only.set_readonly(true);
    std::fs::set_permissions(&data_path, read_only).unwrap();
    let outcome = execute_ducklake_sql(
        &ctx,
        &catalog,
        "CREATE TABLE ducklake.main.failed AS SELECT id FROM source",
    )
    .await;
    std::fs::set_permissions(&data_path, writable).unwrap();

    assert!(outcome.is_err(), "the data directory is not writable");
    assert_eq!(lake.head().await, head);
    assert_eq!(lake.table_id("failed").await, None);

    // The next commit takes the snapshot id the failed write had reserved; the
    // pending table row must not surface with it.
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

    // Retrying the same statement reuses the pending row and publishes it.
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
        let catalog =
            Arc::new(DuckLakeCatalog::with_writer(Arc::new(provider), Arc::new(writer)).unwrap());
        let ctx = SessionContext::new();
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
    async fn ctas_on_the_duckdb_writer_hides_a_failed_table_from_later_snapshots() {
        let temp = TempDir::new().unwrap();
        let (ctx, catalog, _database) = duckdb_lake(&temp);
        let provider = catalog.provider();
        let head = provider.get_current_snapshot().unwrap();
        let schema_id = provider
            .get_schema_by_name("main", head)
            .unwrap()
            .unwrap()
            .schema_id;

        let data_path = temp.path().join("data");
        let writable = std::fs::metadata(&data_path).unwrap().permissions();
        let mut read_only = writable.clone();
        read_only.set_readonly(true);
        std::fs::set_permissions(&data_path, read_only).unwrap();
        let outcome = execute_ducklake_sql(
            &ctx,
            &catalog,
            "CREATE TABLE ducklake.main.failed AS SELECT id FROM source",
        )
        .await;
        std::fs::set_permissions(&data_path, writable).unwrap();
        assert!(outcome.is_err(), "the data directory is not writable");
        assert_eq!(provider.get_current_snapshot().unwrap(), head);

        execute_ducklake_sql(
            &ctx,
            &catalog,
            "CREATE TABLE ducklake.main.later AS SELECT id FROM source WHERE id = 6",
        )
        .await
        .unwrap();
        let published = provider.get_current_snapshot().unwrap();
        assert_eq!(published, head + 1);
        assert!(
            provider
                .get_table_by_name(schema_id, "failed", published)
                .unwrap()
                .is_none()
        );
        assert!(
            provider
                .get_table_by_name(schema_id, "later", published)
                .unwrap()
                .is_some()
        );
    }
}
