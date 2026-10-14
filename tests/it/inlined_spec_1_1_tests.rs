//! Inlined data tables in the DuckLake 1.1 layout, on every metadata backend.
//!
//! DuckLake 1.1 names the system columns of `ducklake_inlined_data_*` tables
//! `_ducklake_row_id`, `_ducklake_begin_snapshot` and `_ducklake_end_snapshot`.
//! Each test writes inlined rows with this crate (which creates the 1.0 layout),
//! then renames the columns exactly as DuckDB's `v1.1-dev1` migration does, and
//! checks that reads, deletes, appends and flushes see the same values.

#![cfg(feature = "write")]

use std::sync::Arc;

use arrow::array::{Array, Int32Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use arrow::util::display::array_value_to_string;
use datafusion::prelude::SessionContext;
use datafusion_ducklake::inlined_filter::{InlinedComparison, InlinedFilter, InlinedValue};
use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeTableWriter, DuckLakeWriteOptions, MetadataProvider, MetadataWriter,
    WriteMode,
};
use object_store::local::LocalFileSystem;
use tempfile::TempDir;

const PREFIXED: [&str; 3] =
    ["_ducklake_row_id", "_ducklake_begin_snapshot", "_ducklake_end_snapshot"];

/// A metadata backend under test.
trait Backend {
    /// Open a provider and a writer over the catalog.
    async fn open(&self) -> (Arc<dyn MetadataProvider>, Arc<dyn MetadataWriter>);
    /// Run `sql` against the metadata database while no provider or writer is open.
    async fn execute(&self, sql: &str);
    /// The first column of every row `sql` returns, as text.
    async fn query_strings(&self, sql: &str) -> Vec<String>;
    /// Column names of `table`, in table order.
    async fn columns(&self, table: &str) -> Vec<String>;
    /// The `key` column of `ducklake_metadata`, quoted where it is reserved.
    fn key_column(&self) -> &'static str {
        "key"
    }
}

/// Record `version` as the catalog's DuckLake version.
async fn set_catalog_version(backend: &impl Backend, version: &str) {
    let key = backend.key_column();
    backend
        .execute(&format!(
            "DELETE FROM ducklake_metadata WHERE {key} = 'version'"
        ))
        .await;
    backend
        .execute(&format!(
            "INSERT INTO ducklake_metadata ({key}, value, scope, scope_id) \
             VALUES ('version', '{version}', NULL, NULL)"
        ))
        .await;
}

fn inlined_table_names(backend: &impl Backend) -> impl std::future::Future<Output = Vec<String>> {
    backend.query_strings("SELECT table_name FROM ducklake_inlined_data_tables ORDER BY table_name")
}

/// Rename the inlined data system columns and record the catalog version, as
/// DuckDB's `v1.1-dev1` migration does.
async fn migrate_to_spec_1_1(backend: &impl Backend) {
    for table in backend
        .query_strings("SELECT table_name FROM ducklake_inlined_data_tables")
        .await
    {
        for (old, new) in ["row_id", "begin_snapshot", "end_snapshot"]
            .into_iter()
            .zip(PREFIXED)
        {
            backend
                .execute(&format!("ALTER TABLE {table} RENAME COLUMN {old} TO {new}"))
                .await;
        }
    }
    set_catalog_version(backend, "1.1-dev1").await;
}

fn object_store() -> Arc<dyn object_store::ObjectStore> {
    Arc::new(LocalFileSystem::new())
}

fn batch(ids: &[i32], notes: &[Option<&str>]) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("note", DataType::Utf8, true),
    ]));
    RecordBatch::try_new(
        schema,
        vec![Arc::new(Int32Array::from(ids.to_vec())), Arc::new(StringArray::from(notes.to_vec()))],
    )
    .unwrap()
}

fn inlining_writer(writer: Arc<dyn MetadataWriter>) -> DuckLakeTableWriter {
    DuckLakeTableWriter::new(writer, object_store())
        .unwrap()
        .with_options(&DuckLakeWriteOptions::default().with_data_inlining_row_limit(10))
}

/// Every row as `value|value|...`, with `NULL` for nulls.
fn render(batches: &[RecordBatch]) -> Vec<String> {
    batches
        .iter()
        .flat_map(|batch| {
            (0..batch.num_rows()).map(move |row| {
                batch
                    .columns()
                    .iter()
                    .map(|column| {
                        if column.is_null(row) {
                            "NULL".to_string()
                        } else {
                            array_value_to_string(column, row).unwrap()
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("|")
            })
        })
        .collect()
}

async fn sql(ctx: &SessionContext, query: &str) -> Vec<String> {
    render(&ctx.sql(query).await.unwrap().collect().await.unwrap())
}

fn context(catalog: DuckLakeCatalog) -> SessionContext {
    let ctx = SessionContext::new();
    ctx.register_catalog("lake", Arc::new(catalog));
    ctx
}

/// Table id and columns of `main.t` at `snapshot_id`.
fn table_structure(
    provider: &dyn MetadataProvider,
    snapshot_id: i64,
) -> (
    i64,
    Vec<datafusion_ducklake::metadata_provider::DuckLakeTableColumn>,
) {
    let schema = provider
        .get_schema_by_name("main", snapshot_id)
        .unwrap()
        .unwrap();
    let table = provider
        .get_table_by_name(schema.schema_id, "t", snapshot_id)
        .unwrap()
        .unwrap();
    let columns = provider
        .get_table_structure(table.table_id, snapshot_id)
        .unwrap();
    (table.table_id, columns)
}

const ALL: &str = "SELECT id, note FROM lake.main.t ORDER BY id";

/// Check reads of `main.t`, whose inlined rows are `(1, 'a')`, `(3, 'c')` and
/// `(4, NULL)` with row ids 0, 2 and 3, and `(2, 'b')` ended after `inserted`.
/// Returns the table id and columns at the current snapshot.
async fn assert_spec_1_1_reads(
    ctx: &SessionContext,
    provider: &Arc<dyn MetadataProvider>,
    inserted: i64,
) -> (
    i64,
    Vec<datafusion_ducklake::metadata_provider::DuckLakeTableColumn>,
) {
    assert_eq!(sql(ctx, ALL).await, vec!["1|a", "3|c", "4|NULL"]);
    assert_eq!(
        sql(ctx, "SELECT COUNT(*) FROM lake.main.t").await,
        vec!["3"]
    );
    assert_eq!(
        sql(ctx, "SELECT id FROM lake.main.t WHERE id >= 3 ORDER BY id").await,
        vec!["3", "4"]
    );
    assert_eq!(
        sql(ctx, "SELECT id FROM lake.main.t WHERE note = 'c'").await,
        vec!["3"]
    );

    // The deleted row is still visible before its delete.
    let pinned = context(DuckLakeCatalog::with_snapshot(provider.clone(), inserted).unwrap());
    assert_eq!(sql(&pinned, ALL).await, vec!["1|a", "2|b", "3|c", "4|NULL"]);

    // The pushed filter runs in the metadata database.
    let current = provider.get_current_snapshot().unwrap();
    let (table_id, columns) = table_structure(provider.as_ref(), current);
    let pushed = provider
        .scan_inlined_data(
            table_id,
            current,
            &columns,
            Some(&InlinedFilter::Comparison {
                column: "note".to_string(),
                op: InlinedComparison::Eq,
                value: InlinedValue::Utf8("c".to_string()),
            }),
        )
        .unwrap();
    assert_eq!(pushed.materialized_row_count, 1);
    assert_eq!(render(&pushed.batches), vec!["3|c"]);

    let with_row_ids = provider
        .get_inlined_data_with_row_ids(table_id, current, &columns)
        .unwrap();
    assert_eq!(
        with_row_ids
            .iter()
            .flat_map(|data| data.row_ids.clone())
            .collect::<Vec<_>>(),
        vec![0, 2, 3]
    );
    assert_eq!(
        with_row_ids
            .iter()
            .flat_map(|data| render(std::slice::from_ref(&data.batch)))
            .collect::<Vec<_>>(),
        vec!["1|a", "3|c", "4|NULL"]
    );
    (table_id, columns)
}

/// Assert every registered inlined data table has the 1.1 system columns.
async fn assert_spec_1_1_layout(backend: &impl Backend) {
    let tables = backend
        .query_strings("SELECT table_name FROM ducklake_inlined_data_tables")
        .await;
    assert!(!tables.is_empty());
    for table in tables {
        assert_eq!(backend.columns(&table).await[..3], PREFIXED);
    }
}

/// Read, delete, append and flush inlined rows of a catalog migrated to the 1.1 layout.
async fn spec_1_1_inlined_round_trip(backend: &impl Backend) {
    // Inlined rows in the 1.0 layout; `id = 2` is ended by a delete.
    let inserted = {
        let (provider, writer) = backend.open().await;
        let inserted = inlining_writer(writer.clone())
            .write_table(
                "main",
                "t",
                &[batch(&[1, 2, 3, 4], &[Some("a"), Some("b"), Some("c"), None])],
            )
            .await
            .unwrap();
        assert_eq!(inserted.files_written, 0, "the rows must be inlined");
        let ctx = context(DuckLakeCatalog::with_writer(provider, writer).unwrap());
        assert_eq!(
            sql(&ctx, "DELETE FROM lake.main.t WHERE id = 2").await,
            vec!["1"]
        );
        inserted.snapshot_id
    };

    migrate_to_spec_1_1(backend).await;

    {
        let (provider, writer) = backend.open().await;
        let ctx = context(DuckLakeCatalog::with_writer(provider.clone(), writer.clone()).unwrap());
        let (table_id, columns) = assert_spec_1_1_reads(&ctx, &provider, inserted).await;
        let pinned = context(DuckLakeCatalog::with_snapshot(provider.clone(), inserted).unwrap());

        // Writes go to the existing 1.1 table.
        inlining_writer(writer.clone())
            .append_table("main", "t", &[batch(&[5], &[Some("e")])])
            .await
            .unwrap();
        assert_eq!(
            sql(&ctx, "DELETE FROM lake.main.t WHERE id = 3").await,
            vec!["1"]
        );
        assert_eq!(sql(&ctx, ALL).await, vec!["1|a", "4|NULL", "5|e"]);

        let before_flush = provider.get_current_snapshot().unwrap();
        let flushed = DuckLakeTableWriter::new(writer.clone(), object_store())
            .unwrap()
            .flush_table_inlined_data(provider.clone(), "main", "t", before_flush)
            .await
            .unwrap()
            .unwrap();
        // Every inlined row is written, deleted ones masked by a delete file.
        assert_eq!(flushed.records_written, 5);
        assert_eq!(sql(&ctx, ALL).await, vec!["1|a", "4|NULL", "5|e"]);
        assert!(
            provider
                .get_inlined_data_with_row_ids(table_id, flushed.snapshot_id, &columns)
                .unwrap()
                .is_empty()
        );
        assert_eq!(sql(&pinned, ALL).await, vec!["1|a", "2|b", "3|c", "4|NULL"]);
    }

    assert_spec_1_1_layout(backend).await;
}

fn options() -> DuckLakeWriteOptions {
    DuckLakeWriteOptions::default().with_data_inlining_row_limit(10)
}

fn extended_batch(ids: &[i32], notes: &[&str], extra: &[i32]) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("note", DataType::Utf8, true),
        Field::new("extra", DataType::Int32, true),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(ids.to_vec())),
            Arc::new(StringArray::from(notes.to_vec())),
            Arc::new(Int32Array::from(extra.to_vec())),
        ],
    )
    .unwrap()
}

/// One table whose inlined rows span both layouts: a 1.0 table left unrenamed in
/// a `1.1-dev1` catalog, and the 1.1 tables this crate creates there, through
/// both the direct and the staged commit paths.
async fn spec_1_1_mixed_layouts(backend: &impl Backend) {
    {
        let (_, writer) = backend.open().await;
        let written = inlining_writer(writer)
            .write_table("main", "t", &[batch(&[1, 2], &[Some("a"), Some("b")])])
            .await
            .unwrap();
        assert_eq!(written.files_written, 0);
    }
    let legacy = inlined_table_names(backend).await;
    assert_eq!(legacy.len(), 1);
    set_catalog_version(backend, "1.1-dev1").await;

    {
        let (provider, writer) = backend.open().await;
        // A new column starts a schema version, so a new inlined table.
        let appended = inlining_writer(writer.clone())
            .append_table("main", "t", &[extended_batch(&[3], &["c"], &[30])])
            .await
            .unwrap();
        assert_eq!(appended.files_written, 0);

        // The staged path appends to that table and creates one for a new table.
        let table_writer = inlining_writer(writer.clone());
        let mut transaction = table_writer.transaction();
        let staged = extended_batch(&[4], &["d"], &[40]);
        transaction
            .stage_write_with_options(
                "main",
                "t",
                staged.schema().as_ref(),
                WriteMode::Append,
                std::slice::from_ref(&staged),
                &options(),
            )
            .await
            .unwrap();
        let committed = transaction.commit().await.unwrap();
        assert_eq!(committed[0].files_written, 0);
        // Creating a table moves the schema version, so it gets a commit of its own.
        let mut transaction = table_writer.transaction();
        let other = batch(&[9], &[Some("z")]);
        transaction
            .stage_write_with_options(
                "main",
                "u",
                other.schema().as_ref(),
                WriteMode::Append,
                std::slice::from_ref(&other),
                &options(),
            )
            .await
            .unwrap();
        let committed = transaction.commit().await.unwrap();
        assert!(committed.iter().all(|result| result.files_written == 0));

        let ctx = context(DuckLakeCatalog::with_writer(provider.clone(), writer.clone()).unwrap());
        let all = "SELECT id, note, extra FROM lake.main.t ORDER BY id";
        assert_eq!(
            sql(&ctx, all).await,
            vec!["1|a|NULL", "2|b|NULL", "3|c|30", "4|d|40"]
        );
        assert_eq!(
            sql(&ctx, "SELECT id, note FROM lake.main.u").await,
            vec!["9|z"]
        );
        assert_eq!(
            sql(
                &ctx,
                "SELECT id FROM lake.main.t WHERE note >= 'b' ORDER BY id"
            )
            .await,
            vec!["2", "3", "4"]
        );

        // One delete ends a row in each layout.
        assert_eq!(
            sql(&ctx, "DELETE FROM lake.main.t WHERE id IN (1, 3)").await,
            vec!["2"]
        );
        assert_eq!(sql(&ctx, all).await, vec!["2|b|NULL", "4|d|40"]);

        // The legacy table keeps its own layout beside the new ones.
        assert_eq!(
            backend.columns(&legacy[0]).await[..3].to_vec(),
            ["row_id", "begin_snapshot", "end_snapshot"]
        );
        let current = provider.get_current_snapshot().unwrap();
        let (table_id, columns) = table_structure(provider.as_ref(), current);
        let inlined = provider
            .get_inlined_data_with_row_ids(table_id, current, &columns)
            .unwrap();
        assert_eq!(
            inlined
                .iter()
                .flat_map(|data| data.row_ids.clone())
                .collect::<Vec<_>>(),
            vec![1, 3]
        );
        let flushed = DuckLakeTableWriter::new(writer.clone(), object_store())
            .unwrap()
            .flush_table_inlined_data(provider.clone(), "main", "t", current)
            .await
            .unwrap()
            .unwrap();
        // Every inlined row is written, deleted ones masked by a delete file.
        assert_eq!(flushed.records_written, 4);
        assert_eq!(sql(&ctx, all).await, vec!["2|b|NULL", "4|d|40"]);
    }

    // The flush emptied the superseded legacy table and dropped it.
    let tables = inlined_table_names(backend).await;
    assert_eq!(tables.len(), 2, "{tables:?}");
    assert!(!tables.contains(&legacy[0]), "{tables:?}");
    for table in tables {
        assert_eq!(
            backend.columns(&table).await[..3].to_vec(),
            PREFIXED,
            "{table}"
        );
    }
}

/// In a 1.1 catalog data columns may use the bare system column names, and rows
/// with them are inlined.
async fn spec_1_1_bare_data_column_names(backend: &impl Backend) {
    set_catalog_version(backend, "1.1-dev1").await;
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("row_id", DataType::Int32, true),
        Field::new("begin_snapshot", DataType::Int32, true),
    ]));
    let rows = |ids: Vec<i32>, row_ids: Vec<i32>, begins: Vec<i32>| {
        RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(ids)),
                Arc::new(Int32Array::from(row_ids)),
                Arc::new(Int32Array::from(begins)),
            ],
        )
        .unwrap()
    };
    let (provider, writer) = backend.open().await;
    let written = inlining_writer(writer.clone())
        .write_table(
            "main",
            "t",
            &[rows(vec![1, 2, 3], vec![10, 20, 30], vec![100, 200, 300])],
        )
        .await
        .unwrap();
    assert_eq!(written.files_written, 0, "the rows must be inlined");
    let table_writer = inlining_writer(writer.clone());
    let mut transaction = table_writer.transaction();
    let staged = rows(vec![4], vec![40], vec![400]);
    transaction
        .stage_write_with_options(
            "main",
            "t",
            &schema,
            WriteMode::Append,
            std::slice::from_ref(&staged),
            &options(),
        )
        .await
        .unwrap();
    assert_eq!(transaction.commit().await.unwrap()[0].files_written, 0);

    let ctx = context(DuckLakeCatalog::with_writer(provider.clone(), writer).unwrap());
    let all = "SELECT id, row_id, begin_snapshot FROM lake.main.t ORDER BY id";
    assert_eq!(
        sql(&ctx, all).await,
        vec!["1|10|100", "2|20|200", "3|30|300", "4|40|400"]
    );
    assert_eq!(
        sql(
            &ctx,
            "SELECT id FROM lake.main.t WHERE begin_snapshot > 150 ORDER BY id"
        )
        .await,
        vec!["2", "3", "4"]
    );
    // The delete ends the row by its system row id, not by the data column.
    assert_eq!(
        sql(&ctx, "DELETE FROM lake.main.t WHERE id = 1").await,
        vec!["1"]
    );
    assert_eq!(
        sql(&ctx, all).await,
        vec!["2|20|200", "3|30|300", "4|40|400"]
    );
    let current = provider.get_current_snapshot().unwrap();
    let (table_id, columns) = table_structure(provider.as_ref(), current);
    let with_row_ids = provider
        .get_inlined_data_with_row_ids(table_id, current, &columns)
        .unwrap();
    assert_eq!(
        with_row_ids
            .iter()
            .flat_map(|data| data.row_ids.clone())
            .collect::<Vec<_>>(),
        vec![1, 2, 3]
    );
    drop((ctx, provider));

    let table = inlined_table_names(backend).await.remove(0);
    assert_eq!(
        backend.columns(&table).await,
        [PREFIXED.as_slice(), &["id", "row_id", "begin_snapshot"]].concat()
    );
}

/// An inlined table whose leading columns are neither layout is an error.
async fn rejects_an_inlined_table_without_system_columns(backend: &impl Backend) {
    {
        let (_, writer) = backend.open().await;
        inlining_writer(writer)
            .write_table("main", "t", &[batch(&[1], &[Some("a")])])
            .await
            .unwrap();
    }
    let table = inlined_table_names(backend).await.remove(0);
    backend
        .execute(&format!(
            "ALTER TABLE {table} RENAME COLUMN row_id TO something_else"
        ))
        .await;
    let (provider, writer) = backend.open().await;
    let ctx = context(DuckLakeCatalog::with_writer(provider, writer).unwrap());
    let error = ctx
        .sql("SELECT id FROM lake.main.t")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("does not start with the DuckLake system columns"),
        "{error}"
    );
}

#[cfg(feature = "write-sqlite")]
mod sqlite {
    use super::*;
    use datafusion_ducklake::{SqliteMetadataProvider, SqliteMetadataWriter};
    use sqlx::sqlite::SqlitePool;
    use sqlx::{AssertSqlSafe, Row};

    pub(super) struct Sqlite {
        url: String,
    }

    impl Sqlite {
        pub(super) async fn new(temp: &TempDir) -> Self {
            let data = temp.path().join("data");
            std::fs::create_dir_all(&data).unwrap();
            let url = format!("sqlite:{}?mode=rwc", temp.path().join("meta.db").display());
            let writer = SqliteMetadataWriter::new_with_init(&url).await.unwrap();
            writer.set_data_path(data.to_str().unwrap()).unwrap();
            Self {
                url,
            }
        }

        async fn pool(&self) -> SqlitePool {
            SqlitePool::connect(&self.url).await.unwrap()
        }
    }

    impl Backend for Sqlite {
        async fn open(&self) -> (Arc<dyn MetadataProvider>, Arc<dyn MetadataWriter>) {
            (
                Arc::new(SqliteMetadataProvider::new(&self.url).await.unwrap()),
                Arc::new(SqliteMetadataWriter::new(&self.url).await.unwrap()),
            )
        }

        async fn execute(&self, sql: &str) {
            sqlx::query(AssertSqlSafe(sql.to_string()))
                .execute(&self.pool().await)
                .await
                .unwrap();
        }

        async fn query_strings(&self, sql: &str) -> Vec<String> {
            sqlx::query(AssertSqlSafe(sql.to_string()))
                .fetch_all(&self.pool().await)
                .await
                .unwrap()
                .iter()
                .map(|row| row.try_get(0).unwrap())
                .collect()
        }

        async fn columns(&self, table: &str) -> Vec<String> {
            sqlx::query_scalar("SELECT name FROM pragma_table_info(?) ORDER BY cid")
                .bind(table)
                .fetch_all(&self.pool().await)
                .await
                .unwrap()
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn sqlite_reads_and_writes_spec_1_1_inlined_tables() {
        let temp = TempDir::new().unwrap();
        spec_1_1_inlined_round_trip(&Sqlite::new(&temp).await).await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn sqlite_spec_1_1_mixed_layouts() {
        let temp = TempDir::new().unwrap();
        spec_1_1_mixed_layouts(&Sqlite::new(&temp).await).await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn sqlite_spec_1_1_bare_data_column_names() {
        let temp = TempDir::new().unwrap();
        spec_1_1_bare_data_column_names(&Sqlite::new(&temp).await).await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn sqlite_rejects_an_inlined_table_without_system_columns() {
        let temp = TempDir::new().unwrap();
        rejects_an_inlined_table_without_system_columns(&Sqlite::new(&temp).await).await;
    }
}

#[cfg(feature = "write-duckdb")]
mod duckdb_backend {
    use super::*;
    use datafusion_ducklake::DuckdbMetadataWriter;

    pub(super) struct Duckdb {
        path: String,
    }

    impl Duckdb {
        fn connection(&self) -> duckdb::Connection {
            duckdb::Connection::open(&self.path).unwrap()
        }
    }

    impl Backend for Duckdb {
        async fn open(&self) -> (Arc<dyn MetadataProvider>, Arc<dyn MetadataWriter>) {
            let writer = DuckdbMetadataWriter::new(&self.path).unwrap();
            (Arc::new(writer.metadata_provider()), Arc::new(writer))
        }

        async fn execute(&self, sql: &str) {
            self.connection().execute_batch(sql).unwrap();
        }

        async fn query_strings(&self, sql: &str) -> Vec<String> {
            let connection = self.connection();
            let mut statement = connection.prepare(sql).unwrap();
            statement
                .query_map([], |row| row.get::<_, String>(0))
                .unwrap()
                .collect::<Result<_, _>>()
                .unwrap()
        }

        async fn columns(&self, table: &str) -> Vec<String> {
            let connection = self.connection();
            let mut statement = connection
                .prepare("SELECT name FROM pragma_table_info(?) ORDER BY cid")
                .unwrap();
            statement
                .query_map([table], |row| row.get::<_, String>(0))
                .unwrap()
                .collect::<Result<_, _>>()
                .unwrap()
        }
    }

    impl Duckdb {
        fn new(temp: &TempDir) -> Self {
            let data = temp.path().join("data");
            std::fs::create_dir_all(&data).unwrap();
            let path = temp.path().join("meta.duckdb").display().to_string();
            let writer = DuckdbMetadataWriter::new_with_init(&path).unwrap();
            writer.set_data_path(data.to_str().unwrap()).unwrap();
            Self {
                path,
            }
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn duckdb_reads_and_writes_spec_1_1_inlined_tables() {
        let temp = TempDir::new().unwrap();
        spec_1_1_inlined_round_trip(&Duckdb::new(&temp)).await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn duckdb_spec_1_1_mixed_layouts() {
        let temp = TempDir::new().unwrap();
        spec_1_1_mixed_layouts(&Duckdb::new(&temp)).await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn duckdb_spec_1_1_bare_data_column_names() {
        let temp = TempDir::new().unwrap();
        spec_1_1_bare_data_column_names(&Duckdb::new(&temp)).await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn duckdb_rejects_an_inlined_table_without_system_columns() {
        let temp = TempDir::new().unwrap();
        rejects_an_inlined_table_without_system_columns(&Duckdb::new(&temp)).await;
    }
}

#[cfg(feature = "write-postgres")]
mod postgres {
    use super::*;
    use datafusion_ducklake::{MulticatalogManager, PostgresMetadataProvider};
    use sqlx::postgres::PgPool;
    use sqlx::{AssertSqlSafe, Row};
    use testcontainers::ContainerAsync;
    use testcontainers::runners::AsyncRunner;
    use testcontainers_modules::postgres::Postgres as PostgresImage;

    pub(super) struct Postgres {
        pool: PgPool,
        /// The multi-catalog store and catalog the writer and provider open.
        multicatalog: Option<(MulticatalogManager, i64)>,
    }

    async fn start() -> (ContainerAsync<PostgresImage>, String, u16) {
        let container = PostgresImage::default().start().await.unwrap();
        let port = container.get_host_port_ipv4(5432).await.unwrap();
        let url = format!("postgresql://postgres:postgres@127.0.0.1:{port}/postgres");
        (container, url, port)
    }

    impl Backend for Postgres {
        async fn open(&self) -> (Arc<dyn MetadataProvider>, Arc<dyn MetadataWriter>) {
            let (manager, catalog_id) = self
                .multicatalog
                .as_ref()
                .expect("only the multi-catalog layout has a writer for inlined rows");
            (
                Arc::new(manager.provider(*catalog_id).await.unwrap()),
                Arc::new(manager.writer(*catalog_id).await.unwrap()),
            )
        }

        async fn execute(&self, sql: &str) {
            sqlx::query(AssertSqlSafe(sql.to_string()))
                .execute(&self.pool)
                .await
                .unwrap();
        }

        async fn query_strings(&self, sql: &str) -> Vec<String> {
            sqlx::query(AssertSqlSafe(sql.to_string()))
                .fetch_all(&self.pool)
                .await
                .unwrap()
                .iter()
                .map(|row| row.try_get(0).unwrap())
                .collect()
        }

        async fn columns(&self, table: &str) -> Vec<String> {
            sqlx::query_scalar(
                "SELECT column_name::text FROM information_schema.columns
                 WHERE table_schema = current_schema() AND table_name = $1
                 ORDER BY ordinal_position",
            )
            .bind(table)
            .fetch_all(&self.pool)
            .await
            .unwrap()
        }
    }

    /// Inlined rows written by the DuckDB extension in the spec layout.
    #[cfg(feature = "metadata-duckdb")]
    #[tokio::test(flavor = "multi_thread")]
    #[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
    async fn postgres_reads_spec_1_1_inlined_tables() {
        let (_container, url, port) = start().await;
        let temp = TempDir::new().unwrap();
        let data = temp.path().join("data");
        std::fs::create_dir_all(&data).unwrap();
        let inserted = {
            crate::common::ensure_ducklake_installed();
            let conn = duckdb::Connection::open_in_memory().unwrap();
            conn.execute_batch("INSTALL postgres; LOAD postgres; LOAD ducklake;")
                .unwrap();
            conn.execute_batch(&format!(
                "ATTACH 'ducklake:postgres:host=127.0.0.1 port={port} dbname=postgres \
                 user=postgres password=postgres' AS lake \
                 (DATA_PATH '{}', DATA_INLINING_ROW_LIMIT 10);
                 CREATE TABLE lake.t (id INTEGER, note VARCHAR);
                 INSERT INTO lake.t VALUES (1, 'a'), (2, 'b'), (3, 'c'), (4, NULL);",
                data.display().to_string().replace('\'', "''")
            ))
            .unwrap();
            let inserted: i64 = conn
                .query_row(
                    "SELECT max(snapshot_id) FROM ducklake_snapshots('lake')",
                    [],
                    |row| row.get(0),
                )
                .unwrap();
            conn.execute_batch("DELETE FROM lake.t WHERE id = 2; DETACH lake;")
                .unwrap();
            inserted
        };
        let backend = Postgres {
            pool: PgPool::connect(&url).await.unwrap(),
            multicatalog: None,
        };
        assert!(
            backend
                .query_strings("SELECT count(*)::text FROM ducklake_data_file")
                .await
                == vec!["0"],
            "the rows must be inlined"
        );
        migrate_to_spec_1_1(&backend).await;

        let provider: Arc<dyn MetadataProvider> =
            Arc::new(PostgresMetadataProvider::new(&url).await.unwrap());
        let ctx = context(
            DuckLakeCatalog::new(PostgresMetadataProvider::new(&url).await.unwrap()).unwrap(),
        );
        assert_spec_1_1_reads(&ctx, &provider, inserted).await;
        assert_spec_1_1_layout(&backend).await;
    }

    async fn multicatalog(url: &str, temp: &TempDir) -> Postgres {
        let data = temp.path().join("data");
        std::fs::create_dir_all(&data).unwrap();
        let manager = MulticatalogManager::connect(url, 5).await.unwrap();
        let catalog_id = manager.create_catalog("spec_1_1").await.unwrap();
        manager
            .writer(catalog_id)
            .await
            .unwrap()
            .set_data_path(data.to_str().unwrap())
            .unwrap();
        Postgres {
            pool: PgPool::connect(url).await.unwrap(),
            multicatalog: Some((manager, catalog_id)),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    #[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
    async fn postgres_multicatalog_reads_and_writes_spec_1_1_inlined_tables() {
        let (_container, url, _) = start().await;
        let temp = TempDir::new().unwrap();
        spec_1_1_inlined_round_trip(&multicatalog(&url, &temp).await).await;
    }

    #[tokio::test(flavor = "multi_thread")]
    #[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
    async fn postgres_multicatalog_spec_1_1_mixed_layouts() {
        let (_container, url, _) = start().await;
        let temp = TempDir::new().unwrap();
        spec_1_1_mixed_layouts(&multicatalog(&url, &temp).await).await;
    }

    #[tokio::test(flavor = "multi_thread")]
    #[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
    async fn postgres_multicatalog_spec_1_1_bare_data_column_names() {
        let (_container, url, _) = start().await;
        let temp = TempDir::new().unwrap();
        spec_1_1_bare_data_column_names(&multicatalog(&url, &temp).await).await;
    }

    #[tokio::test(flavor = "multi_thread")]
    #[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
    async fn postgres_multicatalog_rejects_an_inlined_table_without_system_columns() {
        let (_container, url, _) = start().await;
        let temp = TempDir::new().unwrap();
        rejects_an_inlined_table_without_system_columns(&multicatalog(&url, &temp).await).await;
    }
}

#[cfg(feature = "write-mysql")]
mod mysql {
    use super::*;
    use datafusion_ducklake::{MySqlMetadataProvider, MySqlMetadataWriter};
    use sqlx::MySqlPool;
    use sqlx::{AssertSqlSafe, Row};
    use testcontainers::ContainerAsync;
    use testcontainers::runners::AsyncRunner;
    use testcontainers_modules::mysql::Mysql;

    pub(super) struct MySql {
        url: String,
        pool: MySqlPool,
    }

    impl Backend for MySql {
        async fn open(&self) -> (Arc<dyn MetadataProvider>, Arc<dyn MetadataWriter>) {
            (
                Arc::new(MySqlMetadataProvider::new(&self.url).await.unwrap()),
                Arc::new(MySqlMetadataWriter::new(&self.url).await.unwrap()),
            )
        }

        async fn execute(&self, sql: &str) {
            sqlx::query(AssertSqlSafe(sql.to_string()))
                .execute(&self.pool)
                .await
                .unwrap();
        }

        async fn query_strings(&self, sql: &str) -> Vec<String> {
            sqlx::query(AssertSqlSafe(sql.to_string()))
                .fetch_all(&self.pool)
                .await
                .unwrap()
                .iter()
                .map(|row| row.try_get(0).unwrap())
                .collect()
        }

        async fn columns(&self, table: &str) -> Vec<String> {
            sqlx::query_scalar(
                "SELECT column_name FROM information_schema.columns
                 WHERE table_schema = DATABASE() AND table_name = ?
                 ORDER BY ordinal_position",
            )
            .bind(table)
            .fetch_all(&self.pool)
            .await
            .unwrap()
        }

        fn key_column(&self) -> &'static str {
            "`key`"
        }
    }

    async fn start() -> (ContainerAsync<Mysql>, MySql, TempDir) {
        let container = Mysql::default().start().await.unwrap();
        let port = container.get_host_port_ipv4(3306).await.unwrap();
        let url = format!("mysql://root@127.0.0.1:{port}/test");
        let temp = TempDir::new().unwrap();
        let data = temp.path().join("data");
        std::fs::create_dir_all(&data).unwrap();
        let writer = MySqlMetadataWriter::new_with_init(&url).await.unwrap();
        writer.set_data_path(data.to_str().unwrap()).unwrap();
        let pool = MySqlPool::connect(&url).await.unwrap();
        (
            container,
            MySql {
                url,
                pool,
            },
            temp,
        )
    }

    #[tokio::test(flavor = "multi_thread")]
    #[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
    async fn mysql_reads_and_writes_spec_1_1_inlined_tables() {
        let (_container, backend, _temp) = start().await;
        spec_1_1_inlined_round_trip(&backend).await;
    }

    #[tokio::test(flavor = "multi_thread")]
    #[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
    async fn mysql_spec_1_1_mixed_layouts() {
        let (_container, backend, _temp) = start().await;
        spec_1_1_mixed_layouts(&backend).await;
    }

    #[tokio::test(flavor = "multi_thread")]
    #[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
    async fn mysql_spec_1_1_bare_data_column_names() {
        let (_container, backend, _temp) = start().await;
        spec_1_1_bare_data_column_names(&backend).await;
    }

    #[tokio::test(flavor = "multi_thread")]
    #[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
    async fn mysql_rejects_an_inlined_table_without_system_columns() {
        let (_container, backend, _temp) = start().await;
        rejects_an_inlined_table_without_system_columns(&backend).await;
    }
}
