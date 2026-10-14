//! Column names reserved while data inlining is enabled, as official DuckLake
//! (`DuckLakeUtil::ValidateNoInlinedSystemColumns`) refuses them: creating or
//! adding such a column while inlining is on, and enabling inlining over a
//! table that has one.

#![cfg(feature = "write-sqlite")]

use std::sync::Arc;

use arrow::array::{Int32Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::prelude::SessionContext;
use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeTableWriter, DuckLakeWriteOptions, MetadataProvider, MetadataWriter,
    SqliteMetadataProvider, SqliteMetadataWriter, execute_ducklake_sql,
};
use object_store::local::LocalFileSystem;
use tempfile::TempDir;

struct Lake {
    _temp: TempDir,
    url: String,
    writer: Arc<SqliteMetadataWriter>,
    provider: Arc<dyn MetadataProvider>,
}

async fn lake() -> Lake {
    let temp = TempDir::new().unwrap();
    let url = format!("sqlite:{}?mode=rwc", temp.path().join("c.db").display());
    let data = temp.path().join("data");
    std::fs::create_dir_all(&data).unwrap();
    let writer = SqliteMetadataWriter::new_with_init(&url).await.unwrap();
    writer.set_data_path(data.to_str().unwrap()).unwrap();
    let snapshot = writer.create_snapshot().unwrap();
    writer.get_or_create_schema("main", None, snapshot).unwrap();
    let provider = Arc::new(SqliteMetadataProvider::new(&url).await.unwrap());
    Lake {
        _temp: temp,
        url,
        writer: Arc::new(writer),
        provider,
    }
}

impl Lake {
    async fn sql(&self, statement: &str) -> datafusion::error::Result<()> {
        let ctx = SessionContext::new();
        let writer: Arc<dyn MetadataWriter> = self.writer.clone();
        let catalog =
            DuckLakeCatalog::with_writer(Arc::clone(&self.provider), Arc::clone(&writer)).unwrap();
        DuckLakeCatalog::register(&ctx, "lake", catalog);
        let handle = DuckLakeCatalog::with_writer(Arc::clone(&self.provider), writer).unwrap();
        execute_ducklake_sql(&ctx, &handle, statement)
            .await?
            .collect()
            .await
            .map(|_| ())
    }

    fn table_id(&self, name: &str) -> i64 {
        let snapshot = self.provider.get_current_snapshot().unwrap();
        let schema = self
            .provider
            .get_schema_by_name("main", snapshot)
            .unwrap()
            .unwrap();
        self.provider
            .get_table_by_name(schema.schema_id, name, snapshot)
            .unwrap()
            .unwrap()
            .table_id
    }

    async fn append(
        &self,
        table: &str,
        columns: &[&str],
        limit: usize,
    ) -> datafusion_ducklake::Result<()> {
        let schema = Arc::new(Schema::new(
            columns
                .iter()
                .map(|name| Field::new(*name, DataType::Int32, true))
                .collect::<Vec<_>>(),
        ));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            columns
                .iter()
                .map(|_| Arc::new(Int32Array::from(vec![1])) as _)
                .collect(),
        )
        .unwrap();
        let writer: Arc<dyn MetadataWriter> = self.writer.clone();
        DuckLakeTableWriter::new(writer, Arc::new(LocalFileSystem::new()))?
            .with_options(&DuckLakeWriteOptions::default().with_data_inlining_row_limit(limit))
            .append_table("main", table, &[batch])
            .await
            .map(|_| ())
    }

    async fn set_version(&self, version: &str) {
        let pool = sqlx::SqlitePool::connect(&self.url).await.unwrap();
        sqlx::query(
            "UPDATE ducklake_metadata SET value = ? WHERE key = 'version' AND scope IS NULL",
        )
        .bind(version)
        .execute(&pool)
        .await
        .unwrap();
        let updated: i64 = sqlx::query_scalar(
            "SELECT count(*) FROM ducklake_metadata WHERE key = 'version' AND value = ?",
        )
        .bind(version)
        .fetch_one(&pool)
        .await
        .unwrap();
        if updated == 0 {
            sqlx::query(
                "INSERT INTO ducklake_metadata (key, value, scope, scope_id) \
                 VALUES ('version', ?, NULL, NULL)",
            )
            .bind(version)
            .execute(&pool)
            .await
            .unwrap();
        }
    }
}

const RESERVED: &str = "is reserved by DuckLake for internal use when data inlining is enabled";

/// `CREATE TABLE` with a reserved column name fails while inlining is on (the
/// default limit is 10), as official's does, and succeeds with inlining off.
#[tokio::test(flavor = "multi_thread")]
async fn create_table_refuses_reserved_names_while_inlining() {
    let lake = lake().await;
    for column in [
        "row_id",
        "begin_snapshot",
        "END_SNAPSHOT",
        "_ducklake_internal_row_id",
        "_ducklake_internal_snapshot_id",
    ] {
        let error = lake
            .sql(&format!(
                "CREATE TABLE lake.main.t (id INT, \"{column}\" INT)"
            ))
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains(RESERVED), "{column}: {error}");
    }
    // Not reserved in a DuckLake 1.0 catalog.
    lake.sql("CREATE TABLE lake.main.ok (id INT, _ducklake_row_id INT)")
        .await
        .unwrap();

    lake.writer
        .set_global_setting("data_inlining_row_limit", "0")
        .unwrap();
    lake.sql("CREATE TABLE lake.main.t (id INT, row_id INT)")
        .await
        .unwrap();
}

/// Enabling inlining over a table with a reserved column name fails, at
/// table scope and catalog-wide, unless that table opted out; inserting into
/// it with inlining off keeps working.
#[tokio::test(flavor = "multi_thread")]
async fn enabling_inlining_refuses_tables_with_reserved_names() {
    let lake = lake().await;
    lake.writer
        .set_global_setting("data_inlining_row_limit", "0")
        .unwrap();
    lake.sql("CREATE TABLE lake.main.t (id INT, row_id INT)")
        .await
        .unwrap();
    let table = lake.table_id("t");

    let error = lake
        .writer
        .set_table_setting(table, "data_inlining_row_limit", "10")
        .unwrap_err()
        .to_string();
    assert!(
        error.contains(
            "Cannot enable data inlining for table \"t\". Column \"row_id\" conflicts with a \
             reserved DuckLake internal column name used for inlining"
        ),
        "{error}"
    );
    let error = lake
        .writer
        .set_global_setting("data_inlining_row_limit", "10")
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("Cannot enable data inlining for table \"t\""),
        "{error}"
    );

    // Disabling stays allowed; a table opted out at table scope is skipped.
    lake.writer
        .set_table_setting(table, "data_inlining_row_limit", "0")
        .unwrap();
    lake.writer
        .set_global_setting("data_inlining_row_limit", "10")
        .unwrap();
    lake.sql("INSERT INTO lake.main.t VALUES (1, 2)")
        .await
        .unwrap();
}

/// A write that adds a reserved column while the table's inlining limit is
/// above `0` fails, as official's `ADD COLUMN` does: the stored limit when the
/// catalog has one, else the writer's own; writing an existing one does not.
#[tokio::test(flavor = "multi_thread")]
async fn adding_a_reserved_column_while_inlining_fails() {
    let lake = lake().await;
    lake.append("t", &["id"], 10).await.unwrap();
    let error = lake
        .append("t", &["id", "begin_snapshot"], 10)
        .await
        .unwrap_err()
        .to_string();
    assert!(error.contains(RESERVED), "{error}");
    lake.writer
        .set_table_setting(lake.table_id("t"), "data_inlining_row_limit", "10")
        .unwrap();
    let error = lake
        .append("t", &["id", "begin_snapshot"], 0)
        .await
        .unwrap_err()
        .to_string();
    assert!(error.contains(RESERVED), "stored limit: {error}");
    lake.writer
        .set_table_setting(lake.table_id("t"), "data_inlining_row_limit", "0")
        .unwrap();
    lake.append("t", &["id", "begin_snapshot"], 10)
        .await
        .unwrap();
    lake.append("t", &["id", "begin_snapshot"], 10)
        .await
        .unwrap();
}

/// A DuckLake 1.1 catalog prefixes the system columns, so the prefixed names
/// are reserved and the bare ones are not.
#[tokio::test(flavor = "multi_thread")]
async fn a_1_1_catalog_reserves_the_prefixed_names() {
    let lake = lake().await;
    lake.set_version("1.1").await;
    lake.sql("CREATE TABLE lake.main.t (id INT, row_id INT, end_snapshot INT)")
        .await
        .unwrap();
    for column in ["_ducklake_row_id", "_ducklake_begin_snapshot", "_ducklake_internal_row_id"] {
        let error = lake
            .sql(&format!(
                "CREATE TABLE lake.main.u (id INT, \"{column}\" INT)"
            ))
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains(RESERVED), "{column}: {error}");
    }
}

/// The single-catalog PostgreSQL writer refuses a reserved column on `CREATE
/// TABLE` and on a write that adds one, as the other writers do.
#[cfg(feature = "write-postgres")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
#[tokio::test(flavor = "multi_thread")]
async fn postgres_single_catalog_refuses_reserved_columns() {
    use datafusion_ducklake::{PostgresMetadataProvider, PostgresSingleCatalogMetadataWriter};
    use testcontainers::runners::AsyncRunner;
    use testcontainers_modules::postgres::Postgres;
    let container = Postgres::default().start().await.unwrap();
    let port = container.get_host_port_ipv4(5432).await.unwrap();
    let url = format!("postgresql://postgres:postgres@127.0.0.1:{port}/postgres");
    let temp = TempDir::new().unwrap();
    let writer: Arc<dyn MetadataWriter> = Arc::new(
        PostgresSingleCatalogMetadataWriter::new_with_init(&url)
            .await
            .unwrap(),
    );
    writer.set_data_path(temp.path().to_str().unwrap()).unwrap();
    let snapshot = writer.create_snapshot().unwrap();
    writer.get_or_create_schema("main", None, snapshot).unwrap();
    let provider: Arc<dyn MetadataProvider> =
        Arc::new(PostgresMetadataProvider::new(&url).await.unwrap());
    let ctx = SessionContext::new();
    DuckLakeCatalog::register(
        &ctx,
        "lake",
        DuckLakeCatalog::with_writer(Arc::clone(&provider), Arc::clone(&writer)).unwrap(),
    );
    let handle = DuckLakeCatalog::with_writer(Arc::clone(&provider), Arc::clone(&writer)).unwrap();
    let error = execute_ducklake_sql(
        &ctx,
        &handle,
        "CREATE TABLE lake.main.u (id INT, row_id INT)",
    )
    .await
    .map(|_| ())
    .unwrap_err()
    .to_string();
    assert!(error.contains(RESERVED), "{error}");

    let batch = |columns: &[&str]| {
        RecordBatch::try_new(
            Arc::new(Schema::new(
                columns
                    .iter()
                    .map(|name| Field::new(*name, DataType::Int32, true))
                    .collect::<Vec<_>>(),
            )),
            columns
                .iter()
                .map(|_| Arc::new(Int32Array::from(vec![1])) as _)
                .collect(),
        )
        .unwrap()
    };
    let table_writer =
        DuckLakeTableWriter::new(Arc::clone(&writer), Arc::new(LocalFileSystem::new()))
            .unwrap()
            .with_options(&DuckLakeWriteOptions::default().with_data_inlining_row_limit(10));
    table_writer
        .append_table("main", "t", &[batch(&["id"])])
        .await
        .unwrap();
    let error = table_writer
        .append_table("main", "t", &[batch(&["id", "end_snapshot"])])
        .await
        .unwrap_err()
        .to_string();
    assert!(error.contains(RESERVED), "{error}");
}

impl Lake {
    /// An append through a `DuckLakeTableWriter` built without options, whose own
    /// inlining limit is `0`.
    async fn append_default(
        &self,
        table: &str,
        columns: &[&str],
    ) -> datafusion_ducklake::Result<()> {
        let schema = Arc::new(Schema::new(
            columns
                .iter()
                .map(|name| Field::new(*name, DataType::Int32, true))
                .collect::<Vec<_>>(),
        ));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            columns
                .iter()
                .map(|_| Arc::new(Int32Array::from(vec![1])) as _)
                .collect(),
        )
        .unwrap();
        let writer: Arc<dyn MetadataWriter> = self.writer.clone();
        DuckLakeTableWriter::new(writer, Arc::new(LocalFileSystem::new()))?
            .append_table("main", table, &[batch])
            .await
            .map(|_| ())
    }

    /// Registered inlined data tables of `table`.
    async fn inlined_tables(&self, table: &str) -> i64 {
        let pool = sqlx::SqlitePool::connect(&self.url).await.unwrap();
        sqlx::query_scalar("SELECT count(*) FROM ducklake_inlined_data_tables WHERE table_id = ?")
            .bind(self.table_id(table))
            .fetch_one(&pool)
            .await
            .unwrap()
    }

    /// Data files of `table`.
    async fn data_files(&self, table: &str) -> i64 {
        let pool = sqlx::SqlitePool::connect(&self.url).await.unwrap();
        sqlx::query_scalar("SELECT count(*) FROM ducklake_data_file WHERE table_id = ?")
            .bind(self.table_id(table))
            .fetch_one(&pool)
            .await
            .unwrap()
    }

    /// Store `data_inlining_row_limit` at the scope of schema `main`.
    async fn set_schema_limit(&self, limit: &str) {
        let snapshot = self.provider.get_current_snapshot().unwrap();
        let schema_id = self
            .provider
            .get_schema_by_name("main", snapshot)
            .unwrap()
            .unwrap()
            .schema_id;
        let pool = sqlx::SqlitePool::connect(&self.url).await.unwrap();
        sqlx::query(
            "INSERT INTO ducklake_metadata (key, value, scope, scope_id) \
             VALUES ('data_inlining_row_limit', ?, 'schema', ?)",
        )
        .bind(limit)
        .bind(schema_id)
        .execute(&pool)
        .await
        .unwrap();
    }
}

/// A `DuckLakeTableWriter` whose own limit is the default `0` applies it where
/// the catalog stores none, as official's `DATA_INLINING_ROW_LIMIT 0` attach
/// option does: on a catalog storing no limit it writes Parquet, accepts a
/// reserved name, and creates no inlined data table.
#[tokio::test(flavor = "multi_thread")]
async fn default_table_writer_does_not_inline_on_an_unconfigured_catalog() {
    let lake = lake().await;
    lake.append_default("t", &["id", "row_id"]).await.unwrap();
    assert_eq!(lake.inlined_tables("t").await, 0);
    assert_eq!(lake.data_files("t").await, 1);
    lake.append_default("t", &["id", "row_id", "end_snapshot"])
        .await
        .unwrap();
    assert_eq!(lake.inlined_tables("t").await, 0);
    assert_eq!(lake.data_files("t").await, 2);
}

/// A positive limit the catalog stores, catalog-wide or at schema scope,
/// outranks the writer's own `0`: a reserved name is refused and a new table
/// gets its inlined data table.
#[tokio::test(flavor = "multi_thread")]
async fn a_stored_limit_outranks_the_default_table_writer() {
    for scope in ["catalog", "schema"] {
        let lake = lake().await;
        match scope {
            "catalog" => lake
                .writer
                .set_global_setting("data_inlining_row_limit", "5")
                .unwrap(),
            _ => lake.set_schema_limit("5").await,
        }
        let error = lake
            .append_default("u", &["id", "row_id"])
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains(RESERVED), "{scope}: {error}");
        lake.append_default("t", &["id"]).await.unwrap();
        assert_eq!(lake.inlined_tables("t").await, 1, "{scope}");
        // The writer's own limit still decides whether its rows are inlined.
        assert_eq!(lake.data_files("t").await, 1, "{scope}");
    }
}

/// A table-scoped positive limit applies to a later write by a default
/// writer: it refuses a new reserved column and creates the inlined data
/// table of the new schema version.
#[tokio::test(flavor = "multi_thread")]
async fn a_table_limit_outranks_the_default_table_writer() {
    let lake = lake().await;
    lake.append_default("t", &["id"]).await.unwrap();
    assert_eq!(lake.inlined_tables("t").await, 0);
    lake.writer
        .set_table_setting(lake.table_id("t"), "data_inlining_row_limit", "5")
        .unwrap();
    let error = lake
        .append_default("t", &["id", "begin_snapshot"])
        .await
        .unwrap_err()
        .to_string();
    assert!(error.contains(RESERVED), "{error}");
    lake.append_default("t", &["id", "v"]).await.unwrap();
    assert!(lake.inlined_tables("t").await >= 1);
}

/// A stored `0` outranks a writer's positive limit: the reserved name is
/// accepted and no inlined data table is created.
#[tokio::test(flavor = "multi_thread")]
async fn a_stored_zero_outranks_a_positive_writer_limit() {
    let lake = lake().await;
    lake.writer
        .set_global_setting("data_inlining_row_limit", "0")
        .unwrap();
    lake.append("t", &["id", "row_id"], 10).await.unwrap();
    assert_eq!(lake.inlined_tables("t").await, 0);
    assert_eq!(lake.data_files("t").await, 1);
}

/// SQL through `DuckLakeCatalog` keeps official's default of `10` where the
/// catalog stores no limit: `CREATE TABLE` creates the inlined data table.
#[tokio::test(flavor = "multi_thread")]
async fn sql_create_table_keeps_the_default_limit() {
    let lake = lake().await;
    lake.sql("CREATE TABLE lake.main.t (id INT)").await.unwrap();
    assert_eq!(lake.inlined_tables("t").await, 1);
    lake.sql("INSERT INTO lake.main.t VALUES (1)")
        .await
        .unwrap();
    assert_eq!(lake.data_files("t").await, 0);
}
