//! Catalogs that share one PostgreSQL database's DuckLake tables, each the
//! rows whose `catalog_id` is its connection's `ducklake.catalog_id`, which
//! official DuckDB and this crate both read and write.

#![cfg(all(
    feature = "write-postgres",
    feature = "metadata-duckdb",
    feature = "metadata-postgres"
))]

use std::sync::Arc;

use arrow::array::Int32Array;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion_ducklake::{
    DuckLakeTableWriter, DuckLakeWriteOptions, MetadataProvider, MetadataWriter,
    PostgresMetadataProvider, PostgresSingleCatalogMetadataWriter,
};
use object_store::local::LocalFileSystem;
use tempfile::TempDir;
use testcontainers::ContainerAsync;
use testcontainers::runners::AsyncRunner;
use testcontainers_modules::postgres::Postgres;

use crate::common;

/// A PostgreSQL server with `lake`, an ordinary role, which row-level security
/// applies to, owning its `public` schema.
struct Store {
    _container: ContainerAsync<Postgres>,
    port: u16,
    temp: TempDir,
}

impl Store {
    async fn new() -> Self {
        let container = Postgres::default().start().await.unwrap();
        let port = container.get_host_port_ipv4(5432).await.unwrap();
        let admin = sqlx::PgPool::connect(&format!(
            "postgresql://postgres:postgres@127.0.0.1:{port}/postgres"
        ))
        .await
        .unwrap();
        for statement in
            ["CREATE ROLE lake LOGIN PASSWORD 'lake'", "ALTER SCHEMA public OWNER TO lake"]
        {
            sqlx::query(statement).execute(&admin).await.unwrap();
        }
        Self {
            _container: container,
            port,
            temp: TempDir::new().unwrap(),
        }
    }

    /// The crate's connection string for `catalog`.
    fn url(&self, catalog: i64) -> String {
        format!(
            "postgresql://lake:lake@127.0.0.1:{}/postgres?options=-c%20ducklake.catalog_id%3D{catalog}",
            self.port
        )
    }

    /// The crate's writer of `catalog`, creating it with its data path.
    async fn writer(&self, catalog: i64) -> Arc<dyn MetadataWriter> {
        let writer = PostgresSingleCatalogMetadataWriter::new_with_init(&self.url(catalog))
            .await
            .unwrap();
        let data_path = self.temp.path().join(format!("catalog{catalog}"));
        std::fs::create_dir_all(&data_path).unwrap();
        writer.set_data_path(data_path.to_str().unwrap()).unwrap();
        Arc::new(writer)
    }

    /// Official DuckDB attached read-write to `catalog`.
    fn official(&self, catalog: i64) -> duckdb::Connection {
        common::ensure_ducklake_installed();
        common::ensure_extension_installed("postgres");
        let oracle = duckdb::Connection::open_in_memory().unwrap();
        oracle
            .execute_batch(&format!(
                "LOAD ducklake; LOAD postgres;
                 ATTACH 'ducklake:postgres:host=127.0.0.1 port={} user=lake password=lake \
                 dbname=postgres options=-cducklake.catalog_id={catalog}' AS lake
                     (DATA_INLINING_ROW_LIMIT 0);",
                self.port
            ))
            .unwrap();
        oracle
    }
}

/// Append `ids` to `main.<table>` of `writer`'s catalog, creating it, as
/// Parquet.
async fn append_ids(writer: Arc<dyn MetadataWriter>, table: &str, ids: Vec<i32>) {
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, true)]));
    let batch = RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(ids))]).unwrap();
    DuckLakeTableWriter::new(writer, Arc::new(LocalFileSystem::new()))
        .unwrap()
        .with_options(&DuckLakeWriteOptions::default().with_data_inlining_row_limit(0))
        .append_table("main", table, &[batch])
        .await
        .unwrap();
}

/// `(table, ids)` of every live table official reads in its attached
/// catalog's `ducklake_table`, in order.
fn official_rows(oracle: &duckdb::Connection) -> Vec<(String, Vec<i32>)> {
    let tables: Vec<String> = {
        let mut statement = oracle
            .prepare(
                "SELECT table_name FROM __ducklake_metadata_lake.ducklake_table
                 WHERE end_snapshot IS NULL ORDER BY table_name",
            )
            .unwrap();
        statement
            .query_map([], |row| row.get(0))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap()
    };
    tables
        .into_iter()
        .map(|table| {
            let mut statement = oracle
                .prepare(&format!("SELECT id FROM lake.main.{table} ORDER BY id"))
                .unwrap();
            let ids = statement
                .query_map([], |row| row.get(0))
                .unwrap()
                .collect::<Result<_, _>>()
                .unwrap();
            (table, ids)
        })
        .collect()
}

/// Two catalogs share the store's tables: official DuckDB and the crate each
/// write both in turn, a rename included, and every reader sees only its own
/// catalog. Each catalog's ids lie in its own range.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn catalogs_share_one_store_and_official_duckdb_writes_each() {
    let store = Store::new().await;
    for catalog in [1, 2] {
        append_ids(store.writer(catalog).await, "crate", vec![catalog as i32]).await;
    }
    for catalog in [1, 2] {
        store
            .official(catalog)
            .execute_batch(&format!(
                "CREATE TABLE lake.main.named AS SELECT {} AS id;
                 ALTER TABLE lake.main.named RENAME TO official;
                 DELETE FROM lake.main.crate WHERE id = {catalog};
                 INSERT INTO lake.main.crate VALUES ({});",
                catalog * 10,
                catalog * 100
            ))
            .unwrap();
    }
    for catalog in [1, 2] {
        append_ids(
            store.writer(catalog).await,
            "crate",
            vec![catalog as i32 * 1000],
        )
        .await;
    }
    let rows: Vec<_> = [1, 2]
        .into_iter()
        .map(|catalog| official_rows(&store.official(catalog)))
        .collect();
    let table_ids: Vec<Vec<i64>> = {
        let mut table_ids = Vec::new();
        for catalog in [1, 2] {
            let provider = PostgresMetadataProvider::new(&store.url(catalog))
                .await
                .unwrap();
            let head = provider.get_current_snapshot().unwrap();
            let schema = provider.get_schema_by_name("main", head).unwrap().unwrap();
            table_ids.push(
                provider
                    .list_tables(schema.schema_id, head)
                    .unwrap()
                    .into_iter()
                    .map(|table| table.table_id)
                    .collect(),
            );
        }
        table_ids
    };

    assert_eq!(
        rows,
        vec![
            vec![("crate".to_string(), vec![100, 1000]), ("official".to_string(), vec![10]),],
            vec![("crate".to_string(), vec![200, 2000]), ("official".to_string(), vec![20]),],
        ]
    );
    for (catalog, ids) in [1_i64, 2].into_iter().zip(&table_ids) {
        assert_eq!(ids.len(), 2);
        assert!(
            ids.iter()
                .all(|id| (catalog << 40..(catalog + 1) << 40).contains(id)),
            "catalog {catalog}: {ids:?}"
        );
    }
}

/// A role that bypasses row-level security would see every catalog of a
/// shared store, so initialization refuses it.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn a_shared_store_refuses_a_role_that_bypasses_row_level_security() {
    let store = Store::new().await;
    let url = format!(
        "postgresql://postgres:postgres@127.0.0.1:{}/postgres?options=-c%20ducklake.catalog_id%3D1",
        store.port
    );

    let error = PostgresSingleCatalogMetadataWriter::new_with_init(&url)
        .await
        .unwrap_err();

    assert!(
        error.to_string().contains("without SUPERUSER or BYPASSRLS"),
        "{error}"
    );
}
