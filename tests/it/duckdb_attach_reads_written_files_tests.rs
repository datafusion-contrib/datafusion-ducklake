//! Official DuckDB reads the Parquet files this crate writes, and writes the
//! catalogs it creates.
//!
//! DuckDB resolves a data file as `data_path`, schema path, table path, and file
//! name concatenated without separators, and validates the stored `footer_size`
//! against the file. A catalog that stores slash-less paths or a footer size that
//! includes the 8-byte trailer is unreadable to it, so only a full read through
//! an attach catches either defect.

//!
//! Each server backend runs with its write feature; DuckDB, which attaches
//! every catalog, is required.

#![cfg(all(feature = "write-duckdb", feature = "metadata-duckdb"))]

use std::sync::Arc;

use arrow::array::{Int32Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use std::ops::Range;

use datafusion::prelude::SessionContext;
#[cfg(feature = "write-mysql")]
use datafusion_ducklake::MySqlMetadataWriter;
#[cfg(feature = "write-postgres")]
use datafusion_ducklake::PostgresSingleCatalogMetadataWriter;
#[cfg(feature = "write-sqlite")]
use datafusion_ducklake::SqliteMetadataWriter;
use datafusion_ducklake::{
    ColumnDef, DuckLakeCatalog, DuckLakeError, DuckLakeTableWriter, DuckLakeWriteOptions,
    DuckdbMetadataProvider, DuckdbMetadataWriter, MetadataWriter, WriteMode, WriteSetupResult,
};
#[cfg(any(feature = "write-postgres", feature = "write-mysql"))]
use testcontainers::ContainerAsync;
#[cfg(any(feature = "write-postgres", feature = "write-mysql"))]
use testcontainers::runners::AsyncRunner;
#[cfg(feature = "write-mysql")]
use testcontainers_modules::mysql::Mysql;
#[cfg(feature = "write-postgres")]
use testcontainers_modules::postgres::Postgres;

use crate::common;
use object_store::local::LocalFileSystem;
use tempfile::TempDir;

#[tokio::test(flavor = "multi_thread")]
async fn duckdb_attach_reads_rows_the_crate_wrote() {
    let temp = TempDir::new().unwrap();
    let database = temp.path().join("catalog.ducklake");
    let database = database.to_str().unwrap().to_string();
    let data_path = temp.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();

    let writer = DuckdbMetadataWriter::new_with_init(&database).unwrap();
    writer.set_data_path(data_path.to_str().unwrap()).unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![7, 11, 13])),
            Arc::new(StringArray::from(vec![Some("a"), None, Some("c")])),
        ],
    )
    .unwrap();
    DuckLakeTableWriter::new(Arc::new(writer), Arc::new(LocalFileSystem::new()))
        .unwrap()
        .write_table("main", "readable", &[batch])
        .await
        .unwrap();

    // DuckDB allows one client per catalog file, and the writer is dropped above.
    let oracle = duckdb::Connection::open_in_memory().unwrap();
    oracle
        .execute_batch(&format!(
            "LOAD ducklake; ATTACH 'ducklake:{database}' AS lake (READ_ONLY);"
        ))
        .unwrap();
    let mut statement = oracle
        .prepare("SELECT id, name FROM lake.main.readable ORDER BY id")
        .unwrap();
    let rows: Vec<(i32, Option<String>)> = statement
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();

    assert_eq!(
        rows,
        vec![(7, Some("a".to_string())), (11, None), (13, Some("c".to_string())),]
    );
}

/// A catalog this crate creates, on one metadata backend, that official
/// DuckDB attaches.
struct Lake {
    temp: TempDir,
    backend: Backend,
}

/// A metadata backend; a server one keeps its container alive.
enum Backend {
    DuckDb,
    #[cfg(feature = "write-sqlite")]
    Sqlite,
    #[cfg(feature = "write-postgres")]
    Postgres {
        _container: ContainerAsync<Postgres>,
        port: u16,
    },
    #[cfg(feature = "write-mysql")]
    Mysql {
        _container: ContainerAsync<Mysql>,
        port: u16,
    },
}

/// Which metadata backend [`Lake::new`] creates the catalog on.
#[derive(Clone, Copy)]
enum Kind {
    DuckDb,
    #[cfg(feature = "write-sqlite")]
    Sqlite,
    #[cfg(feature = "write-postgres")]
    Postgres,
    #[cfg(feature = "write-mysql")]
    Mysql,
}

impl Kind {
    /// Whether official DuckDB 1.5.5 appends to an existing table on this
    /// backend: its MySQL connector refuses the `UPDATE` an append runs.
    fn official_appends(self) -> bool {
        match self {
            #[cfg(feature = "write-mysql")]
            Kind::Mysql => false,
            _ => true,
        }
    }
}

impl Lake {
    async fn new(kind: Kind) -> Self {
        let lake = Self::start(kind).await;
        lake.writer()
            .await
            .set_data_path(&lake.data_path())
            .unwrap();
        lake
    }

    /// A catalog the 0.9.0 release initialized on `kind`'s backend, loaded from
    /// its dump, with the keys, defaults and id allocators that release creates.
    async fn released(kind: Kind) -> Self {
        let lake = Self::start(kind).await;
        lake.execute_metadata_sql(match kind {
            Kind::DuckDb => include_str!("fixtures/release_0_9_0/duckdb.sql"),
            #[cfg(feature = "write-sqlite")]
            Kind::Sqlite => include_str!("fixtures/release_0_9_0/sqlite.sql"),
            #[cfg(feature = "write-postgres")]
            Kind::Postgres => include_str!("fixtures/release_0_9_0/postgres.sql"),
            #[cfg(feature = "write-mysql")]
            Kind::Mysql => include_str!("fixtures/release_0_9_0/mysql.sql"),
        })
        .await;
        lake.migrate_layout().await;
        lake.uninitialized_writer()
            .await
            .set_data_path(&lake.data_path())
            .unwrap();
        lake
    }

    /// The metadata backend `kind` names, holding no catalog yet.
    async fn start(kind: Kind) -> Self {
        let backend = match kind {
            Kind::DuckDb => Backend::DuckDb,
            #[cfg(feature = "write-sqlite")]
            Kind::Sqlite => Backend::Sqlite,
            #[cfg(feature = "write-postgres")]
            Kind::Postgres => {
                let container = Postgres::default().start().await.unwrap();
                let port = container.get_host_port_ipv4(5432).await.unwrap();
                Backend::Postgres {
                    _container: container,
                    port,
                }
            },
            #[cfg(feature = "write-mysql")]
            Kind::Mysql => {
                let container = Mysql::default().start().await.unwrap();
                let port = container.get_host_port_ipv4(3306).await.unwrap();
                Backend::Mysql {
                    _container: container,
                    port,
                }
            },
        };
        let lake = Self {
            temp: TempDir::new().unwrap(),
            backend,
        };
        std::fs::create_dir_all(lake.data_path()).unwrap();
        lake
    }

    fn data_path(&self) -> String {
        self.temp.path().join("data").to_str().unwrap().to_string()
    }

    fn metadata_path(&self) -> String {
        let file = match self.backend {
            Backend::DuckDb => "catalog.ducklake",
            _ => "catalog.sqlite",
        };
        self.temp.path().join(file).to_str().unwrap().to_string()
    }

    /// The URL the crate's writer and `sqlx` connect to.
    fn url(&self) -> String {
        match &self.backend {
            Backend::DuckDb => self.metadata_path(),
            #[cfg(feature = "write-sqlite")]
            Backend::Sqlite => format!("sqlite:{}?mode=rwc", self.metadata_path()),
            #[cfg(feature = "write-postgres")]
            Backend::Postgres {
                port,
                ..
            } => format!("postgresql://postgres:postgres@127.0.0.1:{port}/postgres"),
            #[cfg(feature = "write-mysql")]
            Backend::Mysql {
                port,
                ..
            } => format!("mysql://root@127.0.0.1:{port}/test"),
        }
    }

    /// The crate's writer, initializing (or migrating) the catalog.
    async fn writer(&self) -> Arc<dyn MetadataWriter> {
        let url = self.url();
        match self.backend {
            Backend::DuckDb => Arc::new(DuckdbMetadataWriter::new_with_init(url).unwrap()),
            #[cfg(feature = "write-sqlite")]
            Backend::Sqlite => Arc::new(SqliteMetadataWriter::new_with_init(&url).await.unwrap()),
            #[cfg(feature = "write-postgres")]
            Backend::Postgres {
                ..
            } => Arc::new(
                PostgresSingleCatalogMetadataWriter::new_with_init(&url)
                    .await
                    .unwrap(),
            ),
            #[cfg(feature = "write-mysql")]
            Backend::Mysql {
                ..
            } => Arc::new(MySqlMetadataWriter::new_with_init(&url).await.unwrap()),
        }
    }

    /// The crate's writer, opened without initialization.
    async fn uninitialized_writer(&self) -> Arc<dyn MetadataWriter> {
        let url = self.url();
        match self.backend {
            Backend::DuckDb => Arc::new(DuckdbMetadataWriter::new(url).unwrap()),
            #[cfg(feature = "write-sqlite")]
            Backend::Sqlite => Arc::new(SqliteMetadataWriter::new(&url).await.unwrap()),
            #[cfg(feature = "write-postgres")]
            Backend::Postgres {
                ..
            } => Arc::new(
                PostgresSingleCatalogMetadataWriter::new(&url)
                    .await
                    .unwrap(),
            ),
            #[cfg(feature = "write-mysql")]
            Backend::Mysql {
                ..
            } => Arc::new(MySqlMetadataWriter::new(&url).await.unwrap()),
        }
    }

    /// Run the crate's opt-in migration, which an earlier release's catalog
    /// needs before a writer opens it.
    async fn migrate_layout(&self) {
        let url = self.url();
        match self.backend {
            Backend::DuckDb => DuckdbMetadataWriter::migrate(url).unwrap(),
            #[cfg(feature = "write-sqlite")]
            Backend::Sqlite => SqliteMetadataWriter::migrate(&url).await.unwrap(),
            #[cfg(feature = "write-postgres")]
            Backend::Postgres {
                ..
            } => PostgresSingleCatalogMetadataWriter::migrate(&url)
                .await
                .unwrap(),
            #[cfg(feature = "write-mysql")]
            Backend::Mysql {
                ..
            } => MySqlMetadataWriter::migrate(&url).await.unwrap(),
        }
    }

    /// Migrate the catalog to official's layout.
    async fn migrate(&self) {
        let url = self.url();
        match self.backend {
            Backend::DuckDb => DuckdbMetadataWriter::new(url)
                .unwrap()
                .migrate_to_official_layout()
                .unwrap(),
            #[cfg(feature = "write-sqlite")]
            Backend::Sqlite => SqliteMetadataWriter::new(&url)
                .await
                .unwrap()
                .migrate_to_official_layout()
                .unwrap(),
            #[cfg(feature = "write-postgres")]
            Backend::Postgres {
                ..
            } => PostgresSingleCatalogMetadataWriter::new(&url)
                .await
                .unwrap()
                .migrate_to_official_layout()
                .unwrap(),
            #[cfg(feature = "write-mysql")]
            Backend::Mysql {
                ..
            } => MySqlMetadataWriter::new(&url)
                .await
                .unwrap()
                .migrate_to_official_layout()
                .unwrap(),
        }
    }

    /// Migrate the catalog while another writer appends each of `ids` to
    /// `main.<table>`, one commit each, returning them. The migration starts
    /// once the first append committed. A DuckDB catalog file takes one client
    /// at a time, so there the appends commit before the migration.
    async fn migrate_while_appending(&self, table: &str, ids: Range<i32>) -> Vec<i32> {
        let ids = ids.collect::<Vec<_>>();
        if matches!(self.backend, Backend::DuckDb) {
            for id in &ids {
                append_retrying(self.writer().await, table, *id).await;
            }
            self.migrate().await;
            return ids;
        }
        let writer = self.writer().await;
        let (first, started) = tokio::sync::oneshot::channel();
        let appends = {
            let (table, ids) = (table.to_string(), ids.clone());
            tokio::spawn(async move {
                let mut first = Some(first);
                for id in ids {
                    append_retrying(Arc::clone(&writer), &table, id).await;
                    if let Some(first) = first.take() {
                        first.send(()).unwrap();
                    }
                }
            })
        };
        started.await.unwrap();
        self.migrate().await;
        appends.await.unwrap();
        ids
    }

    /// Official DuckDB attached read-write to the catalog.
    fn official(&self) -> duckdb::Connection {
        let (extension, uri) = match &self.backend {
            Backend::DuckDb => ("ducklake", format!("ducklake:{}", self.metadata_path())),
            #[cfg(feature = "write-sqlite")]
            Backend::Sqlite => (
                "sqlite",
                format!("ducklake:sqlite:{}", self.metadata_path()),
            ),
            #[cfg(feature = "write-postgres")]
            Backend::Postgres {
                port,
                ..
            } => (
                "postgres",
                format!(
                    "ducklake:postgres:host=127.0.0.1 port={port} user=postgres \
                     password=postgres dbname=postgres"
                ),
            ),
            #[cfg(feature = "write-mysql")]
            Backend::Mysql {
                port,
                ..
            } => (
                "mysql",
                format!("ducklake:mysql:host=127.0.0.1 port={port} user=root database=test"),
            ),
        };
        common::ensure_ducklake_installed();
        common::ensure_extension_installed(extension);
        let oracle = duckdb::Connection::open_in_memory().unwrap();
        oracle
            .execute_batch(&format!(
                "LOAD ducklake; LOAD {extension};
                 ATTACH '{uri}' AS lake (DATA_INLINING_ROW_LIMIT 0);"
            ))
            .unwrap();
        oracle
    }

    /// Run `sql`, `;`-separated statements, on the catalog's metadata database.
    async fn execute_metadata_sql(&self, sql: &str) {
        #[cfg_attr(
            not(any(feature = "write-postgres", feature = "write-mysql")),
            allow(unused_variables)
        )]
        let statements = sql
            .split(';')
            .map(str::trim)
            .filter(|statement| !statement.is_empty());
        match self.backend {
            Backend::DuckDb => duckdb::Connection::open(self.metadata_path())
                .unwrap()
                .execute_batch(sql)
                .unwrap(),
            #[cfg(feature = "write-sqlite")]
            Backend::Sqlite => {
                let pool = sqlx::SqlitePool::connect(&self.url()).await.unwrap();
                sqlx::raw_sql(sqlx::AssertSqlSafe(sql.to_string()))
                    .execute(&pool)
                    .await
                    .unwrap();
                pool.close().await;
            },
            #[cfg(feature = "write-postgres")]
            Backend::Postgres {
                ..
            } => {
                let pool = sqlx::PgPool::connect(&self.url()).await.unwrap();
                for statement in statements {
                    sqlx::query(sqlx::AssertSqlSafe(statement.to_string()))
                        .execute(&pool)
                        .await
                        .unwrap();
                }
                pool.close().await;
            },
            #[cfg(feature = "write-mysql")]
            Backend::Mysql {
                ..
            } => {
                let pool = sqlx::MySqlPool::connect(&self.url()).await.unwrap();
                for statement in statements {
                    sqlx::query(sqlx::AssertSqlSafe(statement.to_string()))
                        .execute(&pool)
                        .await
                        .unwrap();
                }
                pool.close().await;
            },
        }
    }

    /// The column names of `table` in the metadata database, in order.
    async fn column_order(&self, table: &str) -> Vec<String> {
        let pragma = "SELECT name FROM pragma_table_info(?) ORDER BY cid";
        match self.backend {
            Backend::DuckDb => {
                let connection = duckdb::Connection::open(self.metadata_path()).unwrap();
                let mut statement = connection.prepare(pragma).unwrap();
                statement
                    .query_map([table], |row| row.get(0))
                    .unwrap()
                    .collect::<Result<_, _>>()
                    .unwrap()
            },
            #[cfg(feature = "write-sqlite")]
            Backend::Sqlite => {
                let pool = sqlx::SqlitePool::connect(&self.url()).await.unwrap();
                let names = sqlx::query_scalar(pragma)
                    .bind(table)
                    .fetch_all(&pool)
                    .await
                    .unwrap();
                pool.close().await;
                names
            },
            #[cfg(feature = "write-postgres")]
            Backend::Postgres {
                ..
            } => {
                let pool = sqlx::PgPool::connect(&self.url()).await.unwrap();
                let names = sqlx::query_scalar(
                    // PostgreSQL 11's `information_schema.columns` repeats an
                    // identity column.
                    "SELECT attname::text FROM pg_attribute
                     WHERE attrelid = to_regclass($1) AND attnum > 0 AND NOT attisdropped
                     ORDER BY attnum",
                )
                .bind(table)
                .fetch_all(&pool)
                .await
                .unwrap();
                pool.close().await;
                names
            },
            #[cfg(feature = "write-mysql")]
            Backend::Mysql {
                ..
            } => {
                let pool = sqlx::MySqlPool::connect(&self.url()).await.unwrap();
                let names = sqlx::query_scalar(
                    "SELECT CAST(column_name AS CHAR) FROM information_schema.columns
                     WHERE table_schema = DATABASE() AND table_name = ?
                     ORDER BY ordinal_position",
                )
                .bind(table)
                .fetch_all(&pool)
                .await
                .unwrap();
                pool.close().await;
                names
            },
        }
    }

    /// The `id` values of `main.<table>` as the crate reads them, in order.
    async fn crate_ids(&self, table: &str) -> Vec<i32> {
        let catalog = match self.backend {
            Backend::DuckDb => {
                DuckLakeCatalog::new(DuckdbMetadataProvider::new(self.metadata_path()).unwrap())
            },
            #[cfg(feature = "write-sqlite")]
            Backend::Sqlite => DuckLakeCatalog::new(
                datafusion_ducklake::SqliteMetadataProvider::new(&self.url())
                    .await
                    .unwrap(),
            ),
            #[cfg(feature = "write-postgres")]
            Backend::Postgres {
                ..
            } => DuckLakeCatalog::new(
                datafusion_ducklake::PostgresMetadataProvider::new(&self.url())
                    .await
                    .unwrap(),
            ),
            #[cfg(feature = "write-mysql")]
            Backend::Mysql {
                ..
            } => DuckLakeCatalog::new(
                datafusion_ducklake::MySqlMetadataProvider::new(&self.url())
                    .await
                    .unwrap(),
            ),
        }
        .unwrap();
        let context = SessionContext::new();
        context.register_catalog("lake", Arc::new(catalog));
        let batches = context
            .sql(&format!("SELECT id FROM lake.main.{table} ORDER BY id"))
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
}

/// Write `ids` as the only column of `main.<table>` through `writer`, to a
/// Parquet file: as a new table with [`WriteMode::Replace`], else appended.
async fn write_ids(writer: Arc<dyn MetadataWriter>, table: &str, ids: Vec<i32>, mode: WriteMode) {
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, true)]));
    let batch = RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(ids))]).unwrap();
    let writer = DuckLakeTableWriter::new(writer, Arc::new(LocalFileSystem::new()))
        .unwrap()
        .with_options(&DuckLakeWriteOptions::default().with_data_inlining_row_limit(0));
    match mode {
        WriteMode::Replace => writer.write_table("main", table, &[batch]).await,
        WriteMode::Append => writer.append_table("main", table, &[batch]).await,
    }
    .unwrap();
}

/// Append `id` to `main.<table>` through `writer`, to a Parquet file, retrying
/// a write another commit conflicted with.
async fn append_retrying(writer: Arc<dyn MetadataWriter>, table: &str, id: i32) {
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, true)]));
    let batch = RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![id]))]).unwrap();
    let writer = DuckLakeTableWriter::new(writer, Arc::new(LocalFileSystem::new()))
        .unwrap()
        .with_options(&DuckLakeWriteOptions::default().with_data_inlining_row_limit(0));
    for _ in 0..10 {
        match writer
            .append_table("main", table, std::slice::from_ref(&batch))
            .await
        {
            Ok(_) => return,
            Err(DuckLakeError::Conflict(_)) => continue,
            Err(e) => panic!("append of {id} failed: {e}"),
        }
    }
    panic!("append of {id} kept conflicting");
}

/// The `id` values of `lake.main.<table>`, in order.
fn official_ids(oracle: &duckdb::Connection, table: &str) -> Vec<i32> {
    let mut statement = oracle
        .prepare(&format!("SELECT id FROM lake.main.{table} ORDER BY id"))
        .unwrap();
    statement
        .query_map([], |row| row.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap()
}

/// Every `(table_id, table_name)` row of `lake`'s `ducklake_table`, by id and
/// version.
fn official_tables(oracle: &duckdb::Connection) -> Vec<(i64, String)> {
    let mut statement = oracle
        .prepare(
            "SELECT table_id, table_name FROM __ducklake_metadata_lake.ducklake_table
             ORDER BY table_id, begin_snapshot",
        )
        .unwrap();
    statement
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap()
}

/// Official DuckDB writes into a catalog the crate created, renaming a table
/// among others, and the crate writes after it, each allocating ids past the
/// other's: the catalog tables official inserts into by position follow its
/// layout, and every snapshot carries the `next_catalog_id` and `next_file_id`
/// it allocates from. Each side appends to a table the other created, except
/// official on MySQL, whose DuckDB 1.5.5 connector refuses the `UPDATE` official
/// runs to append to a table, and each side reads back every row.
async fn duckdb_and_the_crate_write_one_catalog_in_turn(kind: Kind) {
    let lake = Lake::new(kind).await;
    write_ids(
        lake.writer().await,
        "readable",
        vec![7, 11, 13],
        WriteMode::Replace,
    )
    .await;

    let oracle = lake.official();
    oracle
        .execute_batch(
            "CREATE TABLE lake.main.official AS SELECT * FROM (VALUES (1), (2)) t(id);
             DELETE FROM lake.main.readable WHERE id = 11;
             CREATE TABLE lake.main.named AS SELECT 3 AS id;
             ALTER TABLE lake.main.named RENAME TO renamed;",
        )
        .unwrap();
    drop(oracle);
    write_ids(lake.writer().await, "after", vec![23], WriteMode::Replace).await;
    write_ids(lake.writer().await, "official", vec![5], WriteMode::Append).await;
    let oracle = lake.official();
    oracle
        .execute_batch("CREATE TABLE lake.main.official_after AS SELECT 29 AS id;")
        .unwrap();
    if kind.official_appends() {
        oracle
            .execute_batch("INSERT INTO lake.main.after VALUES (31);")
            .unwrap();
    }
    let readable = official_ids(&oracle, "readable");
    let official_rows = official_ids(&oracle, "official");
    let after = official_ids(&oracle, "after");
    let renamed = official_ids(&oracle, "renamed");
    let official_after = official_ids(&oracle, "official_after");
    let tables = official_tables(&oracle);
    let files: (i64, i64) = oracle
        .query_row(
            "SELECT COUNT(*), COUNT(DISTINCT file_id) FROM (
                 SELECT data_file_id AS file_id FROM __ducklake_metadata_lake.ducklake_data_file
                 UNION ALL
                 SELECT delete_file_id FROM __ducklake_metadata_lake.ducklake_delete_file)",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .unwrap();
    // DuckDB holds one client per catalog file.
    drop(oracle);
    let mut crate_rows = Vec::new();
    for table in ["readable", "official", "after", "renamed", "official_after"] {
        crate_rows.push(lake.crate_ids(table).await);
    }
    let appended = if kind.official_appends() {
        vec![23, 31]
    } else {
        vec![23]
    };

    assert_eq!(readable, vec![7, 13]);
    assert_eq!(official_rows, vec![1, 2, 5]);
    assert_eq!(after, appended);
    assert_eq!(
        crate_rows,
        [
            readable.clone(),
            official_rows.clone(),
            after.clone(),
            renamed.clone(),
            official_after.clone()
        ]
    );
    assert_eq!(renamed, vec![3]);
    assert_eq!(official_after, vec![29]);
    assert_eq!(
        tables
            .iter()
            .map(|(_, name)| name.as_str())
            .collect::<Vec<_>>(),
        vec!["readable", "official", "named", "renamed", "after", "official_after"]
    );
    // The rename keeps the table's id; every other table has its own.
    assert_eq!(tables[2].0, tables[3].0);
    let ids = [0, 1, 2, 4, 5].map(|index| tables[index].0);
    assert!(ids.windows(2).all(|pair| pair[0] < pair[1]), "{tables:?}");
    // Three data files from official and three from the crate, official's
    // append, and official's delete file.
    let files_written = if kind.official_appends() {
        8
    } else {
        7
    };
    assert_eq!(files, (files_written, files_written));
}

/// Each table whose rows a commit inlines advances `next_file_id` by one, as
/// official DuckDB does, so an official session sees that its cached table
/// statistics are stale.
async fn inlined_inserts_advance_next_file_id(kind: Kind) {
    let lake = Lake::new(kind).await;
    write_ids(lake.writer().await, "files", vec![1], WriteMode::Replace).await;
    let head_file_id = || {
        lake.official()
            .query_row(
                "SELECT next_file_id FROM __ducklake_metadata_lake.ducklake_snapshot
                 ORDER BY snapshot_id DESC LIMIT 1",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap()
    };
    let before = head_file_id();
    inline_ids(lake.writer().await, WriteMode::Replace).await;
    let created = head_file_id();
    inline_ids(lake.writer().await, WriteMode::Append).await;
    let appended = head_file_id();

    assert_eq!(created, before + 1);
    assert_eq!(appended, before + 2);
}

/// Write two ids to `main.inlined` through `writer`, inlined.
async fn inline_ids(writer: Arc<dyn MetadataWriter>, mode: WriteMode) {
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, true)]));
    let batch = RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![2, 3]))]).unwrap();
    let inlining = DuckLakeTableWriter::new(writer, Arc::new(LocalFileSystem::new()))
        .unwrap()
        .with_options(&DuckLakeWriteOptions::default().with_data_inlining_row_limit(10));
    match mode {
        WriteMode::Replace => inlining.write_table("main", "inlined", &[batch]).await,
        WriteMode::Append => inlining.append_table("main", "inlined", &[batch]).await,
    }
    .unwrap();
}

/// The crate appends to a table official DuckDB renamed in the directory the
/// table keeps, where official reads the table whole.
async fn the_crate_appends_to_a_table_official_renamed(kind: Kind) {
    let lake = Lake::new(kind).await;
    write_ids(lake.writer().await, "named", vec![1, 2], WriteMode::Replace).await;
    lake.official()
        .execute_batch("ALTER TABLE lake.main.named RENAME TO renamed;")
        .unwrap();
    write_ids(lake.writer().await, "renamed", vec![3], WriteMode::Append).await;
    let data = lake.temp.path().join("data").join("main");
    let files = std::fs::read_dir(data.join("named")).unwrap().count();
    let renamed = official_ids(&lake.official(), "renamed");

    assert_eq!(renamed, vec![1, 2, 3]);
    assert_eq!(files, 2);
    assert!(!data.join("renamed").exists());
}

/// A commit after one that left the head's id watermarks unset, as a release
/// before 0.10 does, raises them past the ids official DuckDB handed out, also
/// from a writer that skipped the repair initialization makes, so official
/// never takes an id again.
async fn a_commit_after_unset_watermarks_keeps_official_ids_unique(kind: Kind) {
    let lake = Lake::new(kind).await;
    write_ids(lake.writer().await, "crate", vec![1], WriteMode::Replace).await;
    lake.official()
        .execute_batch(
            "CREATE TABLE lake.main.first AS SELECT 2 AS id;
             CREATE TABLE lake.main.second AS SELECT 3 AS id;",
        )
        .unwrap();
    lake.execute_metadata_sql(match lake.backend {
        // The DuckDB writer's watermark columns default to these.
        Backend::DuckDb => "UPDATE ducklake_snapshot SET next_catalog_id = 1, next_file_id = 0",
        _ => "UPDATE ducklake_snapshot SET next_catalog_id = NULL, next_file_id = NULL",
    })
    .await;
    write_ids(
        lake.uninitialized_writer().await,
        "crate",
        vec![4],
        WriteMode::Append,
    )
    .await;
    let oracle = lake.official();
    oracle
        .execute_batch("CREATE TABLE lake.main.third AS SELECT 5 AS id;")
        .unwrap();
    let tables = official_tables(&oracle);
    let third = official_ids(&oracle, "third");

    assert_eq!(
        tables
            .iter()
            .map(|(_, name)| name.as_str())
            .collect::<Vec<_>>(),
        vec!["crate", "first", "second", "third"]
    );
    assert!(
        tables.windows(2).all(|pair| pair[0].0 < pair[1].0),
        "{tables:?}"
    );
    assert_eq!(third, vec![5]);
}

/// The crate adds a column to a table official DuckDB added one to, numbering
/// it past the table's columns as official does, so official reads the table.
async fn the_crate_adds_a_column_after_official_added_one(kind: Kind) {
    let lake = Lake::new(kind).await;
    write_ids(lake.writer().await, "evolving", vec![1], WriteMode::Replace).await;
    lake.official()
        .execute_batch("ALTER TABLE lake.main.evolving ADD COLUMN official_column INTEGER;")
        .unwrap();
    // The crate column comes first, so it takes the first id the write
    // reserves, the one official took when the counter lags official's.
    let schema = Arc::new(Schema::new(vec![
        Field::new("crate_column", DataType::Int32, true),
        Field::new("id", DataType::Int32, true),
        Field::new("official_column", DataType::Int32, true),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![4])),
            Arc::new(Int32Array::from(vec![2])),
            Arc::new(Int32Array::from(vec![3])),
        ],
    )
    .unwrap();
    DuckLakeTableWriter::new(lake.writer().await, Arc::new(LocalFileSystem::new()))
        .unwrap()
        .with_options(&DuckLakeWriteOptions::default().with_data_inlining_row_limit(0))
        .append_table("main", "evolving", &[batch])
        .await
        .unwrap();
    let oracle = lake.official();
    let rows: Vec<(i32, Option<i32>, Option<i32>)> = {
        let mut statement = oracle
            .prepare("SELECT id, official_column, crate_column FROM lake.main.evolving ORDER BY id")
            .unwrap();
        statement
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap()
    };
    let columns: Vec<(i64, String)> = {
        let mut statement = oracle
            .prepare(
                "SELECT c.column_id, c.column_name
                 FROM __ducklake_metadata_lake.ducklake_column c
                 JOIN __ducklake_metadata_lake.ducklake_table t ON t.table_id = c.table_id
                 WHERE t.table_name = 'evolving' AND c.end_snapshot IS NULL
                 ORDER BY c.column_id",
            )
            .unwrap();
        statement
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap()
    };

    assert_eq!(rows, vec![(1, None, None), (2, Some(3), Some(4))]);
    assert_eq!(
        columns,
        vec![
            (1, "id".to_string()),
            (2, "official_column".to_string()),
            (3, "crate_column".to_string())
        ]
    );
}

/// The crate takes schema and table ids, a schema it creates alone included,
/// from the one `next_catalog_id` namespace official DuckDB allocates both
/// from, so no schema shares an id with a table.
async fn schemas_and_tables_share_one_id_namespace(kind: Kind) {
    let lake = Lake::new(kind).await;
    write_ids(lake.writer().await, "zeroth", vec![0], WriteMode::Replace).await;
    lake.official()
        .execute_batch(
            "CREATE SCHEMA lake.official;
             CREATE TABLE lake.official.first AS SELECT 1 AS id;",
        )
        .unwrap();
    let writer = lake.writer().await;
    let snapshot = writer.create_snapshot().unwrap();
    writer
        .get_or_create_schema("crate", None, snapshot)
        .unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, true)]));
    let batch = RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![2]))]).unwrap();
    DuckLakeTableWriter::new(writer, Arc::new(LocalFileSystem::new()))
        .unwrap()
        .write_table("crate", "second", &[batch])
        .await
        .unwrap();
    let oracle = lake.official();
    oracle
        .execute_batch("CREATE TABLE lake.official.third AS SELECT 3 AS id;")
        .unwrap();
    let ids: Vec<(i64, String)> = {
        let mut statement = oracle
            .prepare(
                "SELECT id, name FROM (
                     SELECT schema_id AS id, schema_name AS name
                     FROM __ducklake_metadata_lake.ducklake_schema
                     UNION ALL
                     SELECT DISTINCT table_id, table_name
                     FROM __ducklake_metadata_lake.ducklake_table)
                 ORDER BY id",
            )
            .unwrap();
        statement
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap()
    };
    let second: i32 = oracle
        .query_row("SELECT id FROM lake.crate.second", [], |row| row.get(0))
        .unwrap();

    assert_eq!(
        ids.iter()
            .map(|(_, name)| name.as_str())
            .collect::<Vec<_>>(),
        vec!["zeroth", "main", "official", "first", "crate", "second", "third"]
    );
    assert!(ids.windows(2).all(|pair| pair[0].0 < pair[1].0), "{ids:?}");
    assert_eq!(second, 2);
}

/// Publish the table `setup` began, as the write path's commit does.
fn publish(
    writer: &dyn MetadataWriter,
    table: &str,
    setup: &WriteSetupResult,
    mode: WriteMode,
) -> datafusion_ducklake::Result<i64> {
    writer
        .publish_snapshot(
            setup.table_id,
            "main",
            table,
            setup.snapshot_id,
            mode,
            setup.base_snapshot_id,
            &[ColumnDef::new("id", "int32", true).unwrap()],
            &setup.field_ids,
        )
        .map(|ids| ids.table_id)
}

/// Begin a write of `main.<table>`, with one `id` column, through `writer`.
fn begin(writer: &dyn MetadataWriter, table: &str, mode: WriteMode) -> WriteSetupResult {
    writer
        .begin_write_transaction(
            "main",
            table,
            &[ColumnDef::new("id", "int32", true).unwrap()],
            mode,
        )
        .unwrap()
}

/// Two writes that create the same table: the one that commits second
/// conflicts instead of writing into the table the first created, as official
/// DuckLake refuses a create of a table another transaction created.
async fn a_second_create_of_a_table_conflicts(kind: Kind) {
    let lake = Lake::new(kind).await;
    let writer = lake.writer().await;
    let first = begin(writer.as_ref(), "twice", WriteMode::Replace);
    let second = begin(writer.as_ref(), "twice", WriteMode::Replace);
    let committed = publish(writer.as_ref(), "twice", &first, WriteMode::Replace).unwrap();
    let error = publish(writer.as_ref(), "twice", &second, WriteMode::Replace).unwrap_err();

    assert_ne!(first.table_id, second.table_id);
    assert_eq!(committed, first.table_id);
    assert!(
        matches!(&error, DuckLakeError::Conflict(message)
            if message == "table \"main\".\"twice\" was created by another transaction; retry the write"),
        "{error}"
    );
}

/// A create whose table id official DuckDB allocated between the setup that
/// reserved it and the commit conflicts, and its retry commits: the setup
/// leaves the committed head snapshot as it is, as official does.
async fn a_create_whose_reserved_id_official_took_conflicts(kind: Kind) {
    let lake = Lake::new(kind).await;
    write_ids(lake.writer().await, "first", vec![1], WriteMode::Replace).await;
    let setup = begin(lake.writer().await.as_ref(), "reserved", WriteMode::Replace);
    lake.official()
        .execute_batch("CREATE TABLE lake.main.official AS SELECT 2 AS id;")
        .unwrap();
    let official_id = official_tables(&lake.official())
        .into_iter()
        .find(|(_, name)| name == "official")
        .unwrap()
        .0;
    let writer = lake.writer().await;
    let error = publish(writer.as_ref(), "reserved", &setup, WriteMode::Replace).unwrap_err();
    let retry = begin(writer.as_ref(), "reserved", WriteMode::Replace);
    let retried = publish(writer.as_ref(), "reserved", &retry, WriteMode::Replace).unwrap();
    drop(writer);
    let official = official_ids(&lake.official(), "official");

    assert_eq!(official_id, setup.table_id);
    assert!(
        matches!(&error, DuckLakeError::Conflict(message)
            if message == "the id reserved for table \"main\".\"reserved\" was taken by another \
                           transaction; retry the write"),
        "{error}"
    );
    assert_eq!(retried, retry.table_id);
    assert!(retry.table_id > official_id, "{retry:?}");
    assert_eq!(official, vec![2]);
}

/// A write whose base snapshot was expired before its commit: one that creates
/// a table commits, as an `Append` too, since its writer reserved the table's
/// id and knows the table did not exist. An `Append` to a table that existed
/// at its setup and was dropped and expired meanwhile conflicts, as it would
/// bring the table back. Not on DuckDB metadata, whose catalog file takes no
/// second client while the writer holds it; see
/// `an_append_creating_a_duckdb_table_past_an_expired_base_commits`.
async fn a_write_past_an_expired_base_commits_unless_its_table_was_dropped(kind: Kind) {
    let lake = Lake::new(kind).await;
    write_ids(lake.writer().await, "doomed", vec![1], WriteMode::Replace).await;
    let writer = lake.writer().await;
    let replace = begin(writer.as_ref(), "replaced", WriteMode::Replace);
    let append = begin(writer.as_ref(), "appended", WriteMode::Append);
    let existing = begin(writer.as_ref(), "doomed", WriteMode::Append);
    write_ids(writer.clone(), "later", vec![2], WriteMode::Replace).await;
    // As a drop of `doomed` and an expiry of every snapshot up to the base
    // leave the catalog: no row of either.
    lake.execute_metadata_sql(&format!(
        "DELETE FROM ducklake_snapshot WHERE snapshot_id <= {base};
         DELETE FROM ducklake_table WHERE table_name = 'doomed'",
        base = replace.base_snapshot_id
    ))
    .await;
    let replaced = publish(writer.as_ref(), "replaced", &replace, WriteMode::Replace).unwrap();
    let appended = publish(writer.as_ref(), "appended", &append, WriteMode::Append).unwrap();
    let error = publish(writer.as_ref(), "doomed", &existing, WriteMode::Append).unwrap_err();
    drop(writer);
    let tables = official_tables(&lake.official());

    assert_eq!(replace.base_snapshot_id, append.base_snapshot_id);
    assert_eq!(replaced, replace.table_id);
    assert_eq!(appended, append.table_id);
    assert!(
        matches!(&error, DuckLakeError::Conflict(message)
            if message == "table \"main\".\"doomed\" may have been dropped by another \
                           transaction: the snapshot this write began on was expired; retry the \
                           write"),
        "{error}"
    );
    assert_eq!(
        tables
            .iter()
            .map(|(_, name)| name.as_str())
            .collect::<Vec<_>>(),
        ["replaced", "appended", "later"]
    );
}

/// A create whose table id official DuckDB took for a table of the same name
/// between the setup that reserved it and the commit conflicts instead of
/// merging into official's table. Not on DuckDB metadata, whose catalog file
/// official cannot attach while the crate's writer holds it.
async fn a_create_whose_reserved_id_official_took_for_the_same_name_conflicts(kind: Kind) {
    let lake = Lake::new(kind).await;
    write_ids(lake.writer().await, "first", vec![1], WriteMode::Replace).await;
    let writer = lake.writer().await;
    let setup = begin(writer.as_ref(), "same", WriteMode::Replace);
    lake.official()
        .execute_batch("CREATE TABLE lake.main.same AS SELECT 2 AS id;")
        .unwrap();
    let official_id = official_tables(&lake.official())
        .into_iter()
        .find(|(_, name)| name == "same")
        .unwrap()
        .0;
    let error = publish(writer.as_ref(), "same", &setup, WriteMode::Replace).unwrap_err();
    drop(writer);
    let same = official_ids(&lake.official(), "same");

    assert_eq!(official_id, setup.table_id);
    assert!(
        matches!(&error, DuckLakeError::Conflict(message)
            if message == "the id reserved for table \"main\".\"same\" was taken by another \
                           transaction; retry the write"),
        "{error}"
    );
    assert_eq!(same, vec![2]);
}

/// A first write to a table of a schema official DuckDB stored under a path
/// other than its name puts the table's files under that path, where official
/// reads them.
async fn a_new_table_goes_under_its_schema_stored_path(kind: Kind) {
    let lake = Lake::new(kind).await;
    write_ids(lake.writer().await, "first", vec![1], WriteMode::Replace).await;
    lake.official()
        .execute_batch("CREATE SCHEMA lake.stored;")
        .unwrap();
    lake.execute_metadata_sql(
        "UPDATE ducklake_schema SET path = 'elsewhere/' WHERE schema_name = 'stored'",
    )
    .await;
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, true)]));
    let batch = RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![2, 3]))]).unwrap();
    DuckLakeTableWriter::new(lake.writer().await, Arc::new(LocalFileSystem::new()))
        .unwrap()
        .with_options(&DuckLakeWriteOptions::default().with_data_inlining_row_limit(0))
        .write_table("stored", "placed", &[batch])
        .await
        .unwrap();
    let data = lake.temp.path().join("data");
    let files = std::fs::read_dir(data.join("elsewhere").join("placed"))
        .unwrap()
        .count();
    let rows: Vec<i32> = {
        let oracle = lake.official();
        let mut statement = oracle
            .prepare("SELECT id FROM lake.stored.placed ORDER BY id")
            .unwrap();
        statement
            .query_map([], |row| row.get(0))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap()
    };

    assert_eq!(files, 1);
    assert!(!data.join("stored").exists());
    assert_eq!(rows, vec![2, 3]);
}

#[tokio::test(flavor = "multi_thread")]
async fn inlined_inserts_advance_next_file_id_on_duckdb() {
    inlined_inserts_advance_next_file_id(Kind::DuckDb).await;
}

#[cfg(feature = "write-sqlite")]
#[tokio::test(flavor = "multi_thread")]
async fn inlined_inserts_advance_next_file_id_on_sqlite() {
    inlined_inserts_advance_next_file_id(Kind::Sqlite).await;
}

#[cfg(feature = "write-postgres")]
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn inlined_inserts_advance_next_file_id_on_postgres() {
    inlined_inserts_advance_next_file_id(Kind::Postgres).await;
}

#[cfg(feature = "write-mysql")]
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn inlined_inserts_advance_next_file_id_on_mysql() {
    inlined_inserts_advance_next_file_id(Kind::Mysql).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn duckdb_and_the_crate_write_one_duckdb_catalog_in_turn() {
    duckdb_and_the_crate_write_one_catalog_in_turn(Kind::DuckDb).await;
}

#[cfg(feature = "write-sqlite")]
#[tokio::test(flavor = "multi_thread")]
async fn duckdb_and_the_crate_write_one_sqlite_catalog_in_turn() {
    duckdb_and_the_crate_write_one_catalog_in_turn(Kind::Sqlite).await;
}

/// A catalog the 0.9.0 release initialized keeps its column order on
/// initialization, so that release still writes it, and the crate writes it
/// too, also through a writer that skips initialization; the migration then
/// brings it to official's layout, keeping its rows and the rows writers commit
/// while it runs, and both official DuckDB and the crate write it afterwards.
async fn migration_brings_a_release_0_9_0_catalog_to_duckdb_writes(kind: Kind) {
    let lake = Lake::released(kind).await;
    // A schema the release created, which has no `schema_uuid`.
    let overriding = match kind {
        #[cfg(feature = "write-postgres")]
        Kind::Postgres => "OVERRIDING SYSTEM VALUE",
        _ => "",
    };
    lake.execute_metadata_sql(&format!(
        "INSERT INTO ducklake_snapshot (snapshot_id, schema_version) VALUES (1, 1);
         INSERT INTO ducklake_schema
             (schema_id, schema_name, path, path_is_relative, begin_snapshot)
             {overriding} VALUES (1, 'released', 'released/', TRUE, 1)"
    ))
    .await;
    let released_columns = lake.column_order("ducklake_table").await;
    write_ids(
        lake.uninitialized_writer().await,
        "uninitialized",
        vec![19],
        WriteMode::Replace,
    )
    .await;
    let writer = lake.writer().await;
    let initialized_columns = lake.column_order("ducklake_table").await;
    write_ids(writer, "unmigrated", vec![17], WriteMode::Replace).await;
    let concurrent = lake.migrate_while_appending("unmigrated", 18..24).await;
    write_ids(
        lake.writer().await,
        "migrated",
        vec![23],
        WriteMode::Replace,
    )
    .await;
    let schema_columns = lake.column_order("ducklake_schema").await;
    let table_columns = lake.column_order("ducklake_table").await;
    let oracle = lake.official();
    oracle
        .execute_batch(
            "CREATE TABLE lake.main.official AS SELECT 31 AS id;
             ALTER TABLE lake.main.official RENAME TO renamed;",
        )
        .unwrap();
    let official_rows: Vec<Vec<i32>> = ["uninitialized", "unmigrated", "migrated", "renamed"]
        .into_iter()
        .map(|table| official_ids(&oracle, table))
        .collect();
    let uuid_versions: Vec<(String, String)> = {
        let mut statement = oracle
            .prepare(
                "SELECT schema_name, substr(schema_uuid::VARCHAR, 15, 1)
                 FROM __ducklake_metadata_lake.ducklake_schema
                 UNION ALL
                 SELECT DISTINCT table_name, substr(table_uuid::VARCHAR, 15, 1)
                 FROM __ducklake_metadata_lake.ducklake_table
                 ORDER BY 1",
            )
            .unwrap();
        statement
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap()
    };
    drop(oracle);
    let mut crate_rows = Vec::new();
    for table in ["uninitialized", "unmigrated", "migrated", "renamed"] {
        crate_rows.push(lake.crate_ids(table).await);
    }

    // Official's columns, the release's first, in its order, then those it
    // lacks, appended.
    let official_columns = [
        "table_id",
        "table_uuid",
        "begin_snapshot",
        "end_snapshot",
        "schema_id",
        "table_name",
        "path",
        "path_is_relative",
    ];
    let mut kept_columns = released_columns.clone();
    kept_columns.extend(
        official_columns
            .iter()
            .filter(|column| !released_columns.iter().any(|name| name == *column))
            .map(|column| column.to_string()),
    );
    assert_eq!(initialized_columns, kept_columns);
    assert_eq!(
        schema_columns,
        [
            "schema_id",
            "schema_uuid",
            "begin_snapshot",
            "end_snapshot",
            "schema_name",
            "path",
            "path_is_relative"
        ]
    );
    assert_eq!(table_columns, official_columns);
    let mut unmigrated = vec![17];
    unmigrated.extend(concurrent);
    assert_eq!(official_rows, [vec![19], unmigrated, vec![23], vec![31]]);
    assert_eq!(crate_rows, official_rows);
    // Every schema and table has a v7 UUID, the release's schema one the
    // migration gave it, except on DuckDB, where that release gave it a v4.
    let released = match kind {
        Kind::DuckDb => "4",
        _ => "7",
    };
    assert_eq!(
        uuid_versions,
        ["main", "migrated", "official", "released", "renamed", "uninitialized", "unmigrated"].map(
            |name| {
                let version = if name == "released" {
                    released
                } else {
                    "7"
                };
                (name.to_string(), version.to_string())
            }
        )
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn migration_brings_a_release_0_9_0_duckdb_catalog_to_duckdb_writes() {
    migration_brings_a_release_0_9_0_catalog_to_duckdb_writes(Kind::DuckDb).await;
}

#[cfg(feature = "write-sqlite")]
#[tokio::test(flavor = "multi_thread")]
async fn migration_brings_a_release_0_9_0_sqlite_catalog_to_duckdb_writes() {
    migration_brings_a_release_0_9_0_catalog_to_duckdb_writes(Kind::Sqlite).await;
}

#[cfg(feature = "write-postgres")]
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn duckdb_and_the_crate_write_one_postgres_catalog_in_turn() {
    duckdb_and_the_crate_write_one_catalog_in_turn(Kind::Postgres).await;
}

#[cfg(feature = "write-postgres")]
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn migration_brings_a_release_0_9_0_postgres_catalog_to_duckdb_writes() {
    migration_brings_a_release_0_9_0_catalog_to_duckdb_writes(Kind::Postgres).await;
}

#[cfg(feature = "write-mysql")]
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn duckdb_and_the_crate_write_one_mysql_catalog_in_turn() {
    duckdb_and_the_crate_write_one_catalog_in_turn(Kind::Mysql).await;
}

#[cfg(feature = "write-mysql")]
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn migration_brings_a_release_0_9_0_mysql_catalog_to_duckdb_writes() {
    migration_brings_a_release_0_9_0_catalog_to_duckdb_writes(Kind::Mysql).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn the_crate_appends_to_a_table_official_renamed_on_duckdb() {
    the_crate_appends_to_a_table_official_renamed(Kind::DuckDb).await;
}

#[cfg(feature = "write-sqlite")]
#[tokio::test(flavor = "multi_thread")]
async fn the_crate_appends_to_a_table_official_renamed_on_sqlite() {
    the_crate_appends_to_a_table_official_renamed(Kind::Sqlite).await;
}

#[cfg(feature = "write-postgres")]
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn the_crate_appends_to_a_table_official_renamed_on_postgres() {
    the_crate_appends_to_a_table_official_renamed(Kind::Postgres).await;
}

#[cfg(feature = "write-mysql")]
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn the_crate_appends_to_a_table_official_renamed_on_mysql() {
    the_crate_appends_to_a_table_official_renamed(Kind::Mysql).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_commit_after_unset_watermarks_keeps_official_ids_unique_on_duckdb() {
    a_commit_after_unset_watermarks_keeps_official_ids_unique(Kind::DuckDb).await;
}

#[cfg(feature = "write-sqlite")]
#[tokio::test(flavor = "multi_thread")]
async fn a_commit_after_unset_watermarks_keeps_official_ids_unique_on_sqlite() {
    a_commit_after_unset_watermarks_keeps_official_ids_unique(Kind::Sqlite).await;
}

#[cfg(feature = "write-postgres")]
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn a_commit_after_unset_watermarks_keeps_official_ids_unique_on_postgres() {
    a_commit_after_unset_watermarks_keeps_official_ids_unique(Kind::Postgres).await;
}

#[cfg(feature = "write-mysql")]
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn a_commit_after_unset_watermarks_keeps_official_ids_unique_on_mysql() {
    a_commit_after_unset_watermarks_keeps_official_ids_unique(Kind::Mysql).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn the_crate_adds_a_column_after_official_added_one_on_duckdb() {
    the_crate_adds_a_column_after_official_added_one(Kind::DuckDb).await;
}

#[cfg(feature = "write-sqlite")]
#[tokio::test(flavor = "multi_thread")]
async fn the_crate_adds_a_column_after_official_added_one_on_sqlite() {
    the_crate_adds_a_column_after_official_added_one(Kind::Sqlite).await;
}

#[cfg(feature = "write-postgres")]
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn the_crate_adds_a_column_after_official_added_one_on_postgres() {
    the_crate_adds_a_column_after_official_added_one(Kind::Postgres).await;
}

#[cfg(feature = "write-mysql")]
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn the_crate_adds_a_column_after_official_added_one_on_mysql() {
    the_crate_adds_a_column_after_official_added_one(Kind::Mysql).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn schemas_and_tables_share_one_id_namespace_on_duckdb() {
    schemas_and_tables_share_one_id_namespace(Kind::DuckDb).await;
}

#[cfg(feature = "write-sqlite")]
#[tokio::test(flavor = "multi_thread")]
async fn schemas_and_tables_share_one_id_namespace_on_sqlite() {
    schemas_and_tables_share_one_id_namespace(Kind::Sqlite).await;
}

#[cfg(feature = "write-postgres")]
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn schemas_and_tables_share_one_id_namespace_on_postgres() {
    schemas_and_tables_share_one_id_namespace(Kind::Postgres).await;
}

#[cfg(feature = "write-mysql")]
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn schemas_and_tables_share_one_id_namespace_on_mysql() {
    schemas_and_tables_share_one_id_namespace(Kind::Mysql).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_second_create_of_a_table_conflicts_on_duckdb() {
    a_second_create_of_a_table_conflicts(Kind::DuckDb).await;
}

#[cfg(feature = "write-sqlite")]
#[tokio::test(flavor = "multi_thread")]
async fn a_second_create_of_a_table_conflicts_on_sqlite() {
    a_second_create_of_a_table_conflicts(Kind::Sqlite).await;
}

#[cfg(feature = "write-postgres")]
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn a_second_create_of_a_table_conflicts_on_postgres() {
    a_second_create_of_a_table_conflicts(Kind::Postgres).await;
}

#[cfg(feature = "write-mysql")]
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn a_second_create_of_a_table_conflicts_on_mysql() {
    a_second_create_of_a_table_conflicts(Kind::Mysql).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_create_whose_reserved_id_official_took_conflicts_on_duckdb() {
    a_create_whose_reserved_id_official_took_conflicts(Kind::DuckDb).await;
}

#[cfg(feature = "write-sqlite")]
#[tokio::test(flavor = "multi_thread")]
async fn a_create_whose_reserved_id_official_took_conflicts_on_sqlite() {
    a_create_whose_reserved_id_official_took_conflicts(Kind::Sqlite).await;
}

#[cfg(feature = "write-postgres")]
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn a_create_whose_reserved_id_official_took_conflicts_on_postgres() {
    a_create_whose_reserved_id_official_took_conflicts(Kind::Postgres).await;
}

#[cfg(feature = "write-mysql")]
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn a_create_whose_reserved_id_official_took_conflicts_on_mysql() {
    a_create_whose_reserved_id_official_took_conflicts(Kind::Mysql).await;
}

#[cfg(feature = "write-sqlite")]
#[tokio::test(flavor = "multi_thread")]
async fn a_write_past_an_expired_base_commits_unless_its_table_was_dropped_on_sqlite() {
    a_write_past_an_expired_base_commits_unless_its_table_was_dropped(Kind::Sqlite).await;
}

#[cfg(feature = "write-postgres")]
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn a_write_past_an_expired_base_commits_unless_its_table_was_dropped_on_postgres() {
    a_write_past_an_expired_base_commits_unless_its_table_was_dropped(Kind::Postgres).await;
}

#[cfg(feature = "write-mysql")]
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn a_write_past_an_expired_base_commits_unless_its_table_was_dropped_on_mysql() {
    a_write_past_an_expired_base_commits_unless_its_table_was_dropped(Kind::Mysql).await;
}

/// The PostgreSQL migration, which rebuilds a table to reorder its columns,
/// keeps the table's owner, grants and indexes and its identity sequence's
/// grants, to a role whose name needs quoting too, and refuses a table a view depends on before changing anything.
#[cfg(feature = "write-postgres")]
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn postgres_migration_keeps_grants_and_indexes_and_refuses_dependent_views() {
    let lake = Lake::released(Kind::Postgres).await;
    lake.execute_metadata_sql(
        "CREATE ROLE \"read'er\";
         GRANT SELECT ON ducklake_table TO \"read'er\";
         GRANT USAGE ON SEQUENCE ducklake_table_table_id_seq TO \"read'er\";
         CREATE INDEX ducklake_table_by_name ON ducklake_table (table_name)",
    )
    .await;
    lake.migrate().await;
    let pool = sqlx::PgPool::connect(&lake.url()).await.unwrap();
    let (select, usage, index, owner): (bool, bool, Option<String>, String) = sqlx::query_as(
        "SELECT has_table_privilege('read''er', 'ducklake_table', 'SELECT'),
                has_sequence_privilege('read''er', pg_get_serial_sequence('ducklake_table', 'table_id'),
                                       'USAGE'),
                (SELECT indexdef FROM pg_indexes WHERE indexname = 'ducklake_table_by_name'),
                (SELECT pg_get_userbyid(relowner)::text FROM pg_class
                 WHERE oid = 'ducklake_table'::regclass)",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    pool.close().await;

    let viewed = Lake::released(Kind::Postgres).await;
    viewed
        .execute_metadata_sql("CREATE VIEW live_schemas AS SELECT * FROM ducklake_schema")
        .await;
    let refused = PostgresSingleCatalogMetadataWriter::new(&viewed.url())
        .await
        .unwrap()
        .migrate_to_official_layout()
        .unwrap_err();
    let refused_columns = viewed.column_order("ducklake_schema").await;

    assert!(select);
    assert!(usage);
    assert_eq!(
        index.as_deref(),
        Some(
            "CREATE INDEX ducklake_table_by_name ON public.ducklake_table USING btree (table_name)"
        )
    );
    assert_eq!(owner, "postgres");
    assert_eq!(
        refused.to_string(),
        "Invalid configuration: migrate_to_official_layout rebuilds ducklake_schema, and views \
         live_schemas depend on it; drop them, migrate, then create them again"
    );
    assert_eq!(
        refused_columns,
        [
            "schema_id",
            "schema_name",
            "path",
            "path_is_relative",
            "begin_snapshot",
            "end_snapshot",
            "schema_uuid"
        ]
    );
}

#[cfg(feature = "write-sqlite")]
#[tokio::test(flavor = "multi_thread")]
async fn a_create_whose_reserved_id_official_took_for_the_same_name_conflicts_on_sqlite() {
    a_create_whose_reserved_id_official_took_for_the_same_name_conflicts(Kind::Sqlite).await;
}

#[cfg(feature = "write-postgres")]
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn a_create_whose_reserved_id_official_took_for_the_same_name_conflicts_on_postgres() {
    a_create_whose_reserved_id_official_took_for_the_same_name_conflicts(Kind::Postgres).await;
}

#[cfg(feature = "write-mysql")]
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn a_create_whose_reserved_id_official_took_for_the_same_name_conflicts_on_mysql() {
    a_create_whose_reserved_id_official_took_for_the_same_name_conflicts(Kind::Mysql).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_new_table_goes_under_its_schema_stored_path_on_duckdb() {
    a_new_table_goes_under_its_schema_stored_path(Kind::DuckDb).await;
}

#[cfg(feature = "write-sqlite")]
#[tokio::test(flavor = "multi_thread")]
async fn a_new_table_goes_under_its_schema_stored_path_on_sqlite() {
    a_new_table_goes_under_its_schema_stored_path(Kind::Sqlite).await;
}

#[cfg(feature = "write-postgres")]
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn a_new_table_goes_under_its_schema_stored_path_on_postgres() {
    a_new_table_goes_under_its_schema_stored_path(Kind::Postgres).await;
}

#[cfg(feature = "write-mysql")]
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn a_new_table_goes_under_its_schema_stored_path_on_mysql() {
    a_new_table_goes_under_its_schema_stored_path(Kind::Mysql).await;
}

/// A role that reads and writes the catalog tables of a current catalog, with
/// `USAGE` on its sequences but neither ownership, `CREATE` nor `UPDATE` on a
/// sequence, opens a writer through `new` and `from_pool` and appends, also
/// after official DuckDB added files: file ids come from the catalog's counter,
/// not from a sequence the writer would have to advance past official's.
#[cfg(feature = "write-postgres")]
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn a_postgres_role_without_ddl_privileges_writes_a_current_catalog() {
    let lake = Lake::new(Kind::Postgres).await;
    write_ids(lake.writer().await, "shared", vec![1], WriteMode::Replace).await;
    lake.execute_metadata_sql(
        "CREATE ROLE writer LOGIN PASSWORD 'writer';
         REVOKE CREATE ON SCHEMA public FROM PUBLIC;
         GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO writer;
         GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO writer",
    )
    .await;
    let url = lake.url().replace("postgres:postgres@", "writer:writer@");
    let opened = PostgresSingleCatalogMetadataWriter::new(&url)
        .await
        .unwrap();
    write_ids(Arc::new(opened), "shared", vec![2], WriteMode::Append).await;
    let pool = sqlx::PgPool::connect(&url).await.unwrap();
    let pooled = PostgresSingleCatalogMetadataWriter::from_pool(pool)
        .await
        .unwrap();
    write_ids(Arc::new(pooled), "shared", vec![3], WriteMode::Append).await;
    lake.official()
        .execute_batch(
            "INSERT INTO lake.main.shared VALUES (4);
             DELETE FROM lake.main.shared WHERE id = 1;",
        )
        .unwrap();
    let after_official = PostgresSingleCatalogMetadataWriter::new(&url)
        .await
        .unwrap();
    write_ids(
        Arc::new(after_official),
        "shared",
        vec![5],
        WriteMode::Append,
    )
    .await;
    let shared = lake.crate_ids("shared").await;
    let official = official_ids(&lake.official(), "shared");

    assert_eq!(shared, vec![2, 3, 4, 5]);
    assert_eq!(official, shared);
}

/// A MySQL user that reads and writes the catalog's tables without `CREATE`
/// opens a writer on a current catalog and appends.
#[cfg(feature = "write-mysql")]
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn a_mysql_user_without_ddl_privileges_writes_a_current_catalog() {
    let lake = Lake::new(Kind::Mysql).await;
    write_ids(lake.writer().await, "shared", vec![1], WriteMode::Replace).await;
    lake.execute_metadata_sql(
        "CREATE USER 'writer'@'%' IDENTIFIED BY 'writer';
         GRANT SELECT, INSERT, UPDATE, DELETE ON test.* TO 'writer'@'%'",
    )
    .await;
    let url = lake.url().replace("root@", "writer:writer@");
    let opened = MySqlMetadataWriter::new(&url).await.unwrap();
    write_ids(Arc::new(opened), "shared", vec![2], WriteMode::Append).await;
    let shared = lake.crate_ids("shared").await;

    assert_eq!(shared, vec![1, 2]);
}

/// An `Append` that creates a table on DuckDB metadata commits although its
/// base snapshot was expired meanwhile, as its writer reserved the table's id.
#[tokio::test(flavor = "multi_thread")]
async fn an_append_creating_a_duckdb_table_past_an_expired_base_commits() {
    let lake = Lake::new(Kind::DuckDb).await;
    write_ids(lake.writer().await, "first", vec![1], WriteMode::Replace).await;
    let writer = Arc::new(DuckdbMetadataWriter::new(lake.metadata_path()).unwrap());
    let setup = begin(writer.as_ref(), "appended", WriteMode::Append);
    write_ids(writer.clone(), "second", vec![2], WriteMode::Replace).await;
    let expired = writer
        .expire_snapshots(datafusion_ducklake::maintenance::ExpireCriteria::OlderThan(
            chrono::Utc::now() + chrono::Duration::minutes(1),
        ))
        .unwrap();
    let appended = publish(writer.as_ref(), "appended", &setup, WriteMode::Append).unwrap();
    drop(writer);
    let tables = official_tables(&lake.official());

    assert!(
        expired
            .iter()
            .any(|snapshot| snapshot.snapshot_id == setup.base_snapshot_id),
        "{expired:?}"
    );
    assert_eq!(appended, setup.table_id);
    assert_eq!(
        tables
            .iter()
            .map(|(_, name)| name.as_str())
            .collect::<Vec<_>>(),
        ["first", "appended", "second"]
    );
}
