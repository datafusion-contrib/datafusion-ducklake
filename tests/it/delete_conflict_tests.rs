//! A table-level write conflict check sees rows deleted since its base snapshot.
//!
//! An `expected_base_snapshot_id` precondition, and a `Replace` checked against
//! the snapshot it began at, fail with `Conflict` when another commit since that
//! snapshot changed the table's deletions only: it wrote a delete file, ended
//! one, or added an inlined delete, and registered no data file. A deletion on
//! another table, or one committed at or before the base, is no conflict.
//!
//! Each scenario runs on every backend with a writer: SQLite and DuckDB per
//! test, the container backends (PostgreSQL single-catalog and multicatalog,
//! MySQL) as one test per backend over distinct tables.

#![cfg(feature = "write")]
#![cfg(any(
    feature = "write-sqlite",
    feature = "write-duckdb",
    feature = "write-postgres",
    feature = "write-mysql"
))]

use std::sync::Arc;

use arrow::array::Int32Array;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::prelude::SessionContext;
use datafusion_ducklake::metadata_provider::{DuckLakeInlinedData, DuckLakeTableFile};
use datafusion_ducklake::metadata_writer::InlinedRowRef;
use datafusion_ducklake::{
    DeleteFileEntry, DuckLakeCatalog, DuckLakeError, DuckLakeTableWriter, DuckLakeWriteOptions,
    MetadataProvider, MetadataWriter, TableWriteOptions, WriteMode, WriteResult,
};
use object_store::local::LocalFileSystem;

/// One backend's catalog: a writer, a provider over the same catalog, and a
/// way to change its metadata tables directly.
#[allow(clippy::large_enum_variant)]
enum Harness {
    #[cfg(feature = "write-sqlite")]
    Sqlite {
        writer: Arc<datafusion_ducklake::SqliteMetadataWriter>,
        provider: Arc<datafusion_ducklake::SqliteMetadataProvider>,
        pool: sqlx::SqlitePool,
        _dir: tempfile::TempDir,
    },
    #[cfg(feature = "write-duckdb")]
    Duckdb {
        // DuckDB allows one client per catalog file, so a direct metadata change
        // closes the writer, edits the file and opens it again.
        writer: std::sync::Mutex<Option<Arc<datafusion_ducklake::DuckdbMetadataWriter>>>,
        path: String,
        _dir: tempfile::TempDir,
    },
    #[cfg(feature = "write-postgres")]
    PostgresSingle {
        writer: Arc<datafusion_ducklake::PostgresSingleCatalogMetadataWriter>,
        provider: Arc<datafusion_ducklake::PostgresMetadataProvider>,
        pool: sqlx::PgPool,
        _dir: tempfile::TempDir,
        _container: testcontainers::ContainerAsync<testcontainers_modules::postgres::Postgres>,
    },
    #[cfg(feature = "write-postgres")]
    PostgresMulticatalog {
        writer: Arc<datafusion_ducklake::PostgresMetadataWriter>,
        provider: Arc<datafusion_ducklake::MulticatalogProvider>,
        pool: sqlx::PgPool,
        _dir: tempfile::TempDir,
        _container: testcontainers::ContainerAsync<testcontainers_modules::postgres::Postgres>,
    },
    #[cfg(feature = "write-mysql")]
    MySql {
        writer: Arc<datafusion_ducklake::MySqlMetadataWriter>,
        provider: Arc<datafusion_ducklake::MySqlMetadataProvider>,
        pool: sqlx::MySqlPool,
        _dir: tempfile::TempDir,
        _container: testcontainers::ContainerAsync<testcontainers_modules::mysql::Mysql>,
    },
}

fn data_dir(dir: &tempfile::TempDir) -> String {
    let data = dir.path().join("data");
    std::fs::create_dir_all(&data).unwrap();
    data.to_str().unwrap().to_string()
}

impl Harness {
    #[cfg(feature = "write-sqlite")]
    async fn sqlite() -> Self {
        let dir = tempfile::TempDir::new().unwrap();
        let conn = format!("sqlite:{}?mode=rwc", dir.path().join("test.db").display());
        let writer = datafusion_ducklake::SqliteMetadataWriter::new_with_init(&conn)
            .await
            .unwrap();
        writer.set_data_path(&data_dir(&dir)).unwrap();
        let provider = datafusion_ducklake::SqliteMetadataProvider::new(&conn)
            .await
            .unwrap();
        let pool = sqlx::SqlitePool::connect(&conn).await.unwrap();
        Self::Sqlite {
            writer: Arc::new(writer),
            provider: Arc::new(provider),
            pool,
            _dir: dir,
        }
    }

    #[cfg(feature = "write-duckdb")]
    fn duckdb() -> Self {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir
            .path()
            .join("catalog.ducklake")
            .to_str()
            .unwrap()
            .to_string();
        let writer = datafusion_ducklake::DuckdbMetadataWriter::new_with_init(&path).unwrap();
        writer.set_data_path(&data_dir(&dir)).unwrap();
        Self::Duckdb {
            writer: std::sync::Mutex::new(Some(Arc::new(writer))),
            path,
            _dir: dir,
        }
    }

    #[cfg(feature = "write-postgres")]
    async fn postgres_single() -> Self {
        use testcontainers::runners::AsyncRunner;
        let container = testcontainers_modules::postgres::Postgres::default()
            .start()
            .await
            .unwrap();
        let port = container.get_host_port_ipv4(5432).await.unwrap();
        let conn = format!("postgresql://postgres:postgres@127.0.0.1:{port}/postgres");
        let dir = tempfile::TempDir::new().unwrap();
        let writer = datafusion_ducklake::PostgresSingleCatalogMetadataWriter::new_with_init(&conn)
            .await
            .unwrap();
        writer.set_data_path(&data_dir(&dir)).unwrap();
        let provider = datafusion_ducklake::PostgresMetadataProvider::new(&conn)
            .await
            .unwrap();
        let pool = sqlx::PgPool::connect(&conn).await.unwrap();
        Self::PostgresSingle {
            writer: Arc::new(writer),
            provider: Arc::new(provider),
            pool,
            _dir: dir,
            _container: container,
        }
    }

    #[cfg(feature = "write-postgres")]
    async fn postgres_multicatalog() -> Self {
        use testcontainers::runners::AsyncRunner;
        let container = testcontainers_modules::postgres::Postgres::default()
            .start()
            .await
            .unwrap();
        let port = container.get_host_port_ipv4(5432).await.unwrap();
        let conn = format!("postgresql://postgres:postgres@127.0.0.1:{port}/postgres");
        let pool = sqlx::PgPool::connect(&conn).await.unwrap();
        datafusion_ducklake::initialize_multicatalog_schema(&pool)
            .await
            .unwrap();
        let catalog_id = datafusion_ducklake::MulticatalogManager::new(pool.clone())
            .create_catalog("lake")
            .await
            .unwrap();
        let dir = tempfile::TempDir::new().unwrap();
        let writer =
            datafusion_ducklake::PostgresMetadataWriter::with_pool(pool.clone(), catalog_id)
                .await
                .unwrap();
        writer.set_data_path(&data_dir(&dir)).unwrap();
        let provider = datafusion_ducklake::MulticatalogProvider::with_pool(pool.clone(), "lake")
            .await
            .unwrap();
        Self::PostgresMulticatalog {
            writer: Arc::new(writer),
            provider: Arc::new(provider),
            pool,
            _dir: dir,
            _container: container,
        }
    }

    #[cfg(feature = "write-mysql")]
    async fn mysql() -> Self {
        use testcontainers::runners::AsyncRunner;
        let container = testcontainers_modules::mysql::Mysql::default()
            .start()
            .await
            .unwrap();
        let port = container.get_host_port_ipv4(3306).await.unwrap();
        let conn = format!("mysql://root@127.0.0.1:{port}/test");
        let dir = tempfile::TempDir::new().unwrap();
        let writer = datafusion_ducklake::MySqlMetadataWriter::new_with_init(&conn)
            .await
            .unwrap();
        writer.set_data_path(&data_dir(&dir)).unwrap();
        let provider = datafusion_ducklake::MySqlMetadataProvider::new(&conn)
            .await
            .unwrap();
        let pool = sqlx::MySqlPool::connect(&conn).await.unwrap();
        Self::MySql {
            writer: Arc::new(writer),
            provider: Arc::new(provider),
            pool,
            _dir: dir,
            _container: container,
        }
    }

    fn writer(&self) -> Arc<dyn MetadataWriter> {
        match self {
            #[cfg(feature = "write-sqlite")]
            Self::Sqlite {
                writer,
                ..
            } => writer.clone(),
            #[cfg(feature = "write-duckdb")]
            Self::Duckdb {
                writer,
                ..
            } => writer.lock().unwrap().clone().unwrap(),
            #[cfg(feature = "write-postgres")]
            Self::PostgresSingle {
                writer,
                ..
            } => writer.clone(),
            #[cfg(feature = "write-postgres")]
            Self::PostgresMulticatalog {
                writer,
                ..
            } => writer.clone(),
            #[cfg(feature = "write-mysql")]
            Self::MySql {
                writer,
                ..
            } => writer.clone(),
        }
    }

    fn provider(&self) -> Arc<dyn MetadataProvider> {
        match self {
            #[cfg(feature = "write-sqlite")]
            Self::Sqlite {
                provider,
                ..
            } => provider.clone(),
            #[cfg(feature = "write-duckdb")]
            Self::Duckdb {
                writer,
                ..
            } => Arc::new(writer.lock().unwrap().as_ref().unwrap().metadata_provider()),
            #[cfg(feature = "write-postgres")]
            Self::PostgresSingle {
                provider,
                ..
            } => provider.clone(),
            #[cfg(feature = "write-postgres")]
            Self::PostgresMulticatalog {
                provider,
                ..
            } => provider.clone(),
            #[cfg(feature = "write-mysql")]
            Self::MySql {
                provider,
                ..
            } => provider.clone(),
        }
    }

    /// Run one statement directly against the catalog's metadata tables.
    async fn exec(&self, statement: &str) {
        let sql = sqlx::AssertSqlSafe(statement.to_string());
        match self {
            #[cfg(feature = "write-sqlite")]
            Self::Sqlite {
                pool,
                ..
            } => {
                sqlx::query(sql).execute(pool).await.unwrap();
            },
            #[cfg(feature = "write-duckdb")]
            Self::Duckdb {
                writer,
                path,
                ..
            } => {
                let mut slot = writer.lock().unwrap();
                let open = slot.take().unwrap();
                assert_eq!(Arc::strong_count(&open), 1, "the writer is still in use");
                drop(open);
                let connection = duckdb::Connection::open(path).unwrap();
                connection.execute_batch(statement).unwrap();
                drop(connection);
                *slot = Some(Arc::new(
                    datafusion_ducklake::DuckdbMetadataWriter::new(path.clone()).unwrap(),
                ));
            },
            #[cfg(feature = "write-postgres")]
            Self::PostgresSingle {
                pool,
                ..
            }
            | Self::PostgresMulticatalog {
                pool,
                ..
            } => {
                sqlx::query(sql).execute(pool).await.unwrap();
            },
            #[cfg(feature = "write-mysql")]
            Self::MySql {
                pool,
                ..
            } => {
                sqlx::query(sql).execute(pool).await.unwrap();
            },
        }
    }

    /// Whether the backend takes an expected base snapshot on a streamed write
    /// session, not only on a transaction.
    fn preconditions_on_sessions(&self) -> bool {
        match self {
            #[cfg(feature = "write-duckdb")]
            Self::Duckdb {
                ..
            } => false,
            #[cfg(feature = "write-mysql")]
            Self::MySql {
                ..
            } => false,
            #[allow(unreachable_patterns)]
            _ => true,
        }
    }

    /// Whether the backend writes inlined rows, which the flush scenarios need.
    /// The single-catalog PostgreSQL writer writes every row to Parquet.
    #[cfg_attr(not(any(feature = "write-postgres", feature = "write-mysql")), allow(dead_code))]
    fn writes_inlined_data(&self) -> bool {
        match self {
            #[cfg(feature = "write-postgres")]
            Self::PostgresSingle {
                ..
            } => false,
            #[allow(unreachable_patterns)]
            _ => true,
        }
    }

    fn table_writer(&self) -> DuckLakeTableWriter {
        DuckLakeTableWriter::new(self.writer(), Arc::new(LocalFileSystem::new())).unwrap()
    }
}

const SCHEMA: &str = "main";

fn id_schema() -> Schema {
    Schema::new(vec![Field::new("id", DataType::Int32, false)])
}

fn ids(values: &[i32]) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(id_schema()),
        vec![Arc::new(Int32Array::from(values.to_vec()))],
    )
    .unwrap()
}

async fn create(h: &Harness, table: &str, values: &[i32]) -> WriteResult {
    h.table_writer()
        .write_table(SCHEMA, table, &[ids(values)])
        .await
        .unwrap()
}

/// Append `values` to `table`, requiring that it is unchanged since `base`.
/// A write session carries the precondition where the backend supports it on
/// a streamed write; a transaction carries it on every backend.
async fn append_if_unchanged(
    h: &Harness,
    table: &str,
    values: &[i32],
    base: i64,
) -> datafusion_ducklake::Result<WriteResult> {
    let options = TableWriteOptions::new().with_expected_base_snapshot_id(base);
    let table_writer = h.table_writer();
    if h.preconditions_on_sessions() {
        let mut session = table_writer
            .begin_write(SCHEMA, table, &id_schema(), WriteMode::Append)?
            .with_options(&options);
        session.write_batch(&ids(values))?;
        return session.finish().await;
    }
    let mut transaction = table_writer.transaction().with_options(&options);
    transaction
        .stage_write(
            SCHEMA,
            table,
            &id_schema(),
            WriteMode::Append,
            &[ids(values)],
        )
        .await?;
    Ok(transaction.commit().await?.remove(0))
}

/// Commit a delete of the first row of the live data file `pick` selects, and
/// nothing else, returning its snapshot.
async fn delete_first_row(
    h: &Harness,
    table: &str,
    table_id: i64,
    pick: impl Fn(&DuckLakeTableFile) -> bool,
) -> i64 {
    let provider = h.provider();
    let head = provider.get_current_snapshot().unwrap();
    let files = provider.get_table_files_for_select(table_id, head).unwrap();
    drop(provider);
    let target = files.iter().find(|file| pick(file)).unwrap();
    let table_writer = h.table_writer();
    let delete = table_writer
        .write_delete_file(SCHEMA, table, &target.file.path, &[0])
        .await
        .unwrap();
    let entry = DeleteFileEntry {
        data_file_id: target.data_file_id,
        expected_prev_delete_file: target.delete_file_id,
        delete,
    };
    let mut transaction = table_writer.transaction();
    transaction
        .stage_write_with_deletes(
            SCHEMA,
            table,
            &id_schema(),
            WriteMode::Append,
            &[],
            &[entry],
            &[],
        )
        .await
        .unwrap();
    let committed = transaction.commit().await.unwrap();
    assert_eq!(committed.len(), 1);
    assert_eq!(
        committed[0].files_written, 0,
        "the commit wrote no data file"
    );
    committed[0].snapshot_id
}

/// The table's live `id`s, ascending, read through the provider.
async fn read_ids(h: &Harness, table: &str) -> Vec<i32> {
    let catalog = DuckLakeCatalog::with_writer(h.provider(), h.writer()).unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("lake", Arc::new(catalog));
    let batches = ctx
        .sql(&format!("SELECT id FROM lake.{SCHEMA}.{table} ORDER BY id"))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .values()
                .to_vec()
        })
        .collect()
}

/// A table with one Parquet file holding `1, 2` and inlined rows `3, 4`, and
/// its inlined rows with their identities as of the head.
async fn parquet_and_inlined(h: &Harness, table: &str) -> (i64, i64, Vec<DuckLakeInlinedData>) {
    let table_id = create(h, table, &[1, 2]).await.table_id;
    let inlined = h
        .table_writer()
        .with_options(&DuckLakeWriteOptions::default().with_data_inlining_row_limit(10))
        .append_table(SCHEMA, table, &[ids(&[3, 4])])
        .await
        .unwrap();
    assert_eq!(inlined.files_written, 0, "the appended rows are inlined");
    let provider = h.provider();
    let base = provider.get_current_snapshot().unwrap();
    let columns = provider.get_table_structure(table_id, base).unwrap();
    let rows = provider
        .get_inlined_data_with_row_ids(table_id, base, &columns)
        .unwrap();
    assert_eq!(
        rows.iter().map(|data| data.batch.num_rows()).sum::<usize>(),
        2
    );
    (table_id, base, rows)
}

fn any_file(_: &DuckLakeTableFile) -> bool {
    true
}

fn assert_conflict<T: std::fmt::Debug>(result: datafusion_ducklake::Result<T>) {
    match result {
        Err(DuckLakeError::Conflict(_)) => {},
        other => panic!("expected a write conflict, got {other:?}"),
    }
}

/// A delete-only commit writes a delete file after the base: a conditional
/// append fails. A delete at the base, or on another table, does not.
async fn new_delete_file_conflicts(h: &Harness, a: &str, b: &str) {
    let first = create(h, a, &[1, 2]).await;
    let other = create(h, b, &[1, 2]).await;
    let base = delete_first_row(h, a, first.table_id, any_file).await;
    delete_first_row(h, b, other.table_id, any_file).await;

    let appended = append_if_unchanged(h, a, &[3, 4], base).await.unwrap();

    // The appended file has no delete file yet, so this commit adds one and
    // ends none.
    delete_first_row(h, a, first.table_id, |file| file.delete_file_id.is_none()).await;
    assert_conflict(append_if_unchanged(h, a, &[5], appended.snapshot_id).await);
    // The losing append left the table as the two deletes did.
    assert_eq!(read_ids(h, a).await, vec![2, 4]);
}

/// A commit after the base ended a delete file and wrote nothing else: a
/// conditional append against the base fails, one against that commit does not.
async fn ended_delete_file_conflicts(h: &Harness, a: &str, b: &str) {
    let first = create(h, a, &[1, 2]).await;
    let base = delete_first_row(h, a, first.table_id, any_file).await;
    let later = create(h, b, &[1]).await.snapshot_id;
    // No write path ends a delete file without also writing a data or delete
    // file, so the row is stamped directly, as a commit at `later` would.
    h.exec(&format!(
        "UPDATE ducklake_delete_file SET end_snapshot = {later}
         WHERE table_id = {} AND end_snapshot IS NULL",
        first.table_id
    ))
    .await;

    assert_conflict(append_if_unchanged(h, a, &[3], base).await);
    // The losing append added nothing; with its delete file ended, both rows
    // are live.
    assert_eq!(read_ids(h, a).await, vec![1, 2]);
    append_if_unchanged(h, a, &[3], later).await.unwrap();
}

/// A `Replace` begun before a delete-only commit on its table fails; one begun
/// before a delete on another table commits.
async fn replace_conflicts_with_concurrent_delete(h: &Harness, a: &str, b: &str) {
    let first = create(h, a, &[1, 2]).await;
    let other = create(h, b, &[1, 2]).await;

    let mut replace = h
        .table_writer()
        .begin_write(SCHEMA, a, &id_schema(), WriteMode::Replace)
        .unwrap();
    replace.write_batch(&ids(&[10, 11])).unwrap();
    delete_first_row(h, b, other.table_id, any_file).await;
    replace.finish().await.unwrap();

    let mut replace = h
        .table_writer()
        .begin_write(SCHEMA, a, &id_schema(), WriteMode::Replace)
        .unwrap();
    replace.write_batch(&ids(&[20, 21])).unwrap();
    delete_first_row(h, a, first.table_id, any_file).await;
    assert_conflict(replace.finish().await);
    // The losing replacement left the table as the delete did.
    assert_eq!(read_ids(h, a).await, vec![11]);
}

/// An inlined delete committed after the base fails a conditional append; one
/// committed at the base does not.
async fn inlined_delete_conflicts(h: &Harness, a: &str, b: &str) {
    let first = create(h, a, &[1, 2]).await;
    let at_base = create(h, b, &[1]).await.snapshot_id;
    let provider = h.provider();
    let file_id = provider
        .get_table_files_for_select(first.table_id, at_base)
        .unwrap()[0]
        .data_file_id;
    drop(provider);
    let inlined = format!("ducklake_inlined_delete_{}", first.table_id);
    h.exec(&format!(
        "CREATE TABLE {inlined} (file_id BIGINT, row_id BIGINT, begin_snapshot BIGINT)"
    ))
    .await;
    h.exec(&format!(
        "INSERT INTO {inlined} VALUES ({file_id}, 0, {at_base})"
    ))
    .await;
    let appended = append_if_unchanged(h, a, &[3], at_base).await.unwrap();

    let later = h
        .table_writer()
        .append_table(SCHEMA, b, &[ids(&[2])])
        .await
        .unwrap()
        .snapshot_id;
    h.exec(&format!(
        "INSERT INTO {inlined} VALUES ({file_id}, 1, {later})"
    ))
    .await;
    assert_conflict(append_if_unchanged(h, a, &[4], appended.snapshot_id).await);
    // The losing append left the table as the two inlined deletes did.
    assert_eq!(read_ids(h, a).await, vec![3]);
}

/// A flush of inlined rows commits over a concurrent commit that only deleted
/// a Parquet row, as in official DuckLake, and the delete survives it.
async fn flush_commits_over_concurrent_delete_file(h: &Harness, a: &str) {
    let (table_id, base, rows) = parquet_and_inlined(h, a).await;
    delete_first_row(h, a, table_id, |file| file.delete_file_id.is_none()).await;

    let flushed = h
        .table_writer()
        .flush_inlined_data(SCHEMA, a, &rows, base)
        .await
        .unwrap()
        .unwrap();

    assert_eq!(flushed.files_written, 1);
    assert_eq!(read_ids(h, a).await, vec![2, 3, 4]);
}

/// A flush fails over a concurrent inlined delete of a Parquet row, as in
/// official DuckLake, and registers nothing.
async fn flush_conflicts_with_concurrent_inlined_delete(h: &Harness, a: &str, b: &str) {
    let (table_id, base, rows) = parquet_and_inlined(h, a).await;
    let provider = h.provider();
    let file_id = provider.get_table_files_for_select(table_id, base).unwrap()[0].data_file_id;
    drop(provider);
    let later = create(h, b, &[1]).await.snapshot_id;
    let inlined = format!("ducklake_inlined_delete_{table_id}");
    h.exec(&format!(
        "CREATE TABLE {inlined} (file_id BIGINT, row_id BIGINT, begin_snapshot BIGINT)"
    ))
    .await;
    h.exec(&format!(
        "INSERT INTO {inlined} VALUES ({file_id}, 0, {later})"
    ))
    .await;

    assert_conflict(
        h.table_writer()
            .flush_inlined_data(SCHEMA, a, &rows, base)
            .await,
    );
    let provider = h.provider();
    let head = provider.get_current_snapshot().unwrap();
    assert_eq!(
        provider
            .get_table_files_for_select(table_id, head)
            .unwrap()
            .len(),
        1,
        "the losing flush registered no data file"
    );
}

/// A flush fails over a concurrent delete of one of the inlined rows it moves,
/// and the table keeps the delete and its other rows.
async fn flush_conflicts_with_concurrent_inlined_row_delete(h: &Harness, a: &str) {
    let (_, base, rows) = parquet_and_inlined(h, a).await;
    let row = InlinedRowRef {
        table_name: rows[0].table_name.clone(),
        row_id: rows[0].row_ids[0],
    };
    let table_writer = h.table_writer();
    let mut transaction = table_writer.transaction();
    transaction
        .stage_write_with_deletes(SCHEMA, a, &id_schema(), WriteMode::Append, &[], &[], &[row])
        .await
        .unwrap();
    transaction.commit().await.unwrap();
    drop(table_writer);

    assert_conflict(
        h.table_writer()
            .flush_inlined_data(SCHEMA, a, &rows, base)
            .await,
    );
    assert_eq!(read_ids(h, a).await, vec![1, 2, 4]);
}

#[cfg_attr(not(any(feature = "write-postgres", feature = "write-mysql")), allow(dead_code))]
async fn all_scenarios(h: &Harness) {
    new_delete_file_conflicts(h, "a1", "b1").await;
    ended_delete_file_conflicts(h, "a2", "b2").await;
    replace_conflicts_with_concurrent_delete(h, "a3", "b3").await;
    inlined_delete_conflicts(h, "a4", "b4").await;
    if h.writes_inlined_data() {
        flush_commits_over_concurrent_delete_file(h, "a5").await;
        flush_conflicts_with_concurrent_inlined_delete(h, "a6", "b6").await;
        flush_conflicts_with_concurrent_inlined_row_delete(h, "a7").await;
    }
}

#[cfg(feature = "write-sqlite")]
mod sqlite {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    async fn new_delete_file() {
        new_delete_file_conflicts(&Harness::sqlite().await, "a", "b").await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn ended_delete_file() {
        ended_delete_file_conflicts(&Harness::sqlite().await, "a", "b").await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn replace_and_concurrent_delete() {
        replace_conflicts_with_concurrent_delete(&Harness::sqlite().await, "a", "b").await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn inlined_delete() {
        inlined_delete_conflicts(&Harness::sqlite().await, "a", "b").await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn flush_and_concurrent_delete_file() {
        flush_commits_over_concurrent_delete_file(&Harness::sqlite().await, "a").await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn flush_and_concurrent_inlined_delete() {
        flush_conflicts_with_concurrent_inlined_delete(&Harness::sqlite().await, "a", "b").await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn flush_and_concurrent_inlined_row_delete() {
        flush_conflicts_with_concurrent_inlined_row_delete(&Harness::sqlite().await, "a").await;
    }
}

#[cfg(feature = "write-duckdb")]
mod duckdb_backend {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    async fn new_delete_file() {
        new_delete_file_conflicts(&Harness::duckdb(), "a", "b").await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn ended_delete_file() {
        ended_delete_file_conflicts(&Harness::duckdb(), "a", "b").await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn replace_and_concurrent_delete() {
        replace_conflicts_with_concurrent_delete(&Harness::duckdb(), "a", "b").await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn inlined_delete() {
        inlined_delete_conflicts(&Harness::duckdb(), "a", "b").await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn flush_and_concurrent_delete_file() {
        flush_commits_over_concurrent_delete_file(&Harness::duckdb(), "a").await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn flush_and_concurrent_inlined_delete() {
        flush_conflicts_with_concurrent_inlined_delete(&Harness::duckdb(), "a", "b").await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn flush_and_concurrent_inlined_row_delete() {
        flush_conflicts_with_concurrent_inlined_row_delete(&Harness::duckdb(), "a").await;
    }
}

#[cfg(feature = "write-postgres")]
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn postgres_single_catalog() {
    all_scenarios(&Harness::postgres_single().await).await;
}

#[cfg(feature = "write-postgres")]
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn postgres_multicatalog() {
    all_scenarios(&Harness::postgres_multicatalog().await).await;
}

#[cfg(feature = "write-mysql")]
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn mysql() {
    all_scenarios(&Harness::mysql().await).await;
}
