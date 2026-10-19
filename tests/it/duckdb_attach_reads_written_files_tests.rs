//! Official DuckDB reads the Parquet files this crate writes, and writes the
//! catalogs it creates.
//!
//! DuckDB resolves a data file as `data_path`, schema path, table path, and file
//! name concatenated without separators, and validates the stored `footer_size`
//! against the file. A catalog that stores slash-less paths or a footer size that
//! includes the 8-byte trailer is unreadable to it, so only a full read through
//! an attach catches either defect.

#![cfg(all(
    feature = "write-duckdb",
    feature = "metadata-duckdb",
    feature = "write-sqlite",
    feature = "write-postgres",
    feature = "write-mysql"
))]

use std::sync::Arc;

use arrow::array::{Int32Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion_ducklake::{
    DuckLakeTableWriter, DuckLakeWriteOptions, DuckdbMetadataWriter, MetadataWriter,
    MySqlMetadataWriter, PostgresSingleCatalogMetadataWriter, SqliteMetadataWriter, WriteMode,
};
use testcontainers::ContainerAsync;
use testcontainers::runners::AsyncRunner;
use testcontainers_modules::mysql::Mysql;
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
    Sqlite,
    Postgres {
        _container: ContainerAsync<Postgres>,
        port: u16,
    },
    Mysql {
        _container: ContainerAsync<Mysql>,
        port: u16,
    },
}

/// Which metadata backend [`Lake::new`] creates the catalog on.
#[derive(Clone, Copy)]
enum Kind {
    DuckDb,
    Sqlite,
    Postgres,
    Mysql,
}

impl Lake {
    async fn new(kind: Kind) -> Self {
        let backend = match kind {
            Kind::DuckDb => Backend::DuckDb,
            Kind::Sqlite => Backend::Sqlite,
            Kind::Postgres => {
                let container = Postgres::default().start().await.unwrap();
                let port = container.get_host_port_ipv4(5432).await.unwrap();
                Backend::Postgres {
                    _container: container,
                    port,
                }
            },
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
        let data_path = lake.temp.path().join("data");
        std::fs::create_dir_all(&data_path).unwrap();
        lake.writer()
            .await
            .set_data_path(data_path.to_str().unwrap())
            .unwrap();
        lake
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
            Backend::Sqlite => format!("sqlite:{}?mode=rwc", self.metadata_path()),
            Backend::Postgres {
                port,
                ..
            } => format!("postgresql://postgres:postgres@127.0.0.1:{port}/postgres"),
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
            Backend::Sqlite => Arc::new(SqliteMetadataWriter::new_with_init(&url).await.unwrap()),
            Backend::Postgres {
                ..
            } => Arc::new(
                PostgresSingleCatalogMetadataWriter::new_with_init(&url)
                    .await
                    .unwrap(),
            ),
            Backend::Mysql {
                ..
            } => Arc::new(MySqlMetadataWriter::new_with_init(&url).await.unwrap()),
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
            Backend::Sqlite => SqliteMetadataWriter::new(&url)
                .await
                .unwrap()
                .migrate_to_official_layout()
                .unwrap(),
            Backend::Postgres {
                ..
            } => PostgresSingleCatalogMetadataWriter::new(&url)
                .await
                .unwrap()
                .migrate_to_official_layout()
                .unwrap(),
            Backend::Mysql {
                ..
            } => MySqlMetadataWriter::new(&url)
                .await
                .unwrap()
                .migrate_to_official_layout()
                .unwrap(),
        }
    }

    /// Official DuckDB attached read-write to the catalog.
    fn official(&self) -> duckdb::Connection {
        let (extension, uri) = match &self.backend {
            Backend::DuckDb => ("ducklake", format!("ducklake:{}", self.metadata_path())),
            Backend::Sqlite => (
                "sqlite",
                format!("ducklake:sqlite:{}", self.metadata_path()),
            ),
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
        let statements = sql
            .split(';')
            .map(str::trim)
            .filter(|statement| !statement.is_empty());
        match self.backend {
            Backend::DuckDb => duckdb::Connection::open(self.metadata_path())
                .unwrap()
                .execute_batch(sql)
                .unwrap(),
            Backend::Sqlite => {
                let pool = sqlx::SqlitePool::connect(&self.url()).await.unwrap();
                sqlx::raw_sql(sqlx::AssertSqlSafe(sql.to_string()))
                    .execute(&pool)
                    .await
                    .unwrap();
                pool.close().await;
            },
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
}

/// Write `ids` as the only column of a new `main.<table>` through `writer`, to
/// a Parquet file.
async fn write_ids(writer: Arc<dyn MetadataWriter>, table: &str, ids: Vec<i32>) {
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, true)]));
    let batch = RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(ids))]).unwrap();
    DuckLakeTableWriter::new(writer, Arc::new(LocalFileSystem::new()))
        .unwrap()
        .with_options(&DuckLakeWriteOptions::default().with_data_inlining_row_limit(0))
        .write_table("main", table, &[batch])
        .await
        .unwrap();
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

/// Official DuckDB writes into a catalog the crate created, renaming a table
/// among others, and the crate writes after it, each allocating ids past the
/// other's: the catalog tables official inserts into by position follow its
/// layout, and every snapshot carries the `next_catalog_id` and `next_file_id`
/// it allocates from. Official only creates, renames and deletes here: on
/// MySQL, DuckDB 1.5.5's connector refuses the `UPDATE` official runs to
/// append to a table.
async fn duckdb_and_the_crate_write_one_catalog_in_turn(kind: Kind) {
    let lake = Lake::new(kind).await;
    write_ids(lake.writer().await, "readable", vec![7, 11, 13]).await;

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
    write_ids(lake.writer().await, "after", vec![23]).await;
    let oracle = lake.official();
    oracle
        .execute_batch("CREATE TABLE lake.main.official_after AS SELECT 29 AS id;")
        .unwrap();
    let readable = official_ids(&oracle, "readable");
    let official_rows = official_ids(&oracle, "official");
    let after = official_ids(&oracle, "after");
    let renamed = official_ids(&oracle, "renamed");
    let official_after = official_ids(&oracle, "official_after");
    let tables: Vec<(i64, String)> = {
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
    };
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

    assert_eq!(readable, vec![7, 13]);
    assert_eq!(official_rows, vec![1, 2]);
    assert_eq!(after, vec![23]);
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
    // Five data files, three from official and two from the crate, and
    // official's delete file.
    assert_eq!(files, (6, 6));
}

/// Each table whose rows a commit inlines advances `next_file_id` by one, as
/// official DuckDB does, so an official session sees that its cached table
/// statistics are stale.
async fn inlined_inserts_advance_next_file_id(kind: Kind) {
    let lake = Lake::new(kind).await;
    write_ids(lake.writer().await, "files", vec![1]).await;
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

#[tokio::test(flavor = "multi_thread")]
async fn inlined_inserts_advance_next_file_id_on_duckdb() {
    inlined_inserts_advance_next_file_id(Kind::DuckDb).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn inlined_inserts_advance_next_file_id_on_sqlite() {
    inlined_inserts_advance_next_file_id(Kind::Sqlite).await;
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn inlined_inserts_advance_next_file_id_on_postgres() {
    inlined_inserts_advance_next_file_id(Kind::Postgres).await;
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn inlined_inserts_advance_next_file_id_on_mysql() {
    inlined_inserts_advance_next_file_id(Kind::Mysql).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn duckdb_and_the_crate_write_one_duckdb_catalog_in_turn() {
    duckdb_and_the_crate_write_one_catalog_in_turn(Kind::DuckDb).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn duckdb_and_the_crate_write_one_sqlite_catalog_in_turn() {
    duckdb_and_the_crate_write_one_catalog_in_turn(Kind::Sqlite).await;
}

/// The schema, table and file layout of this crate's catalogs before they
/// followed official's, rebuilt from a current catalog.
const SQL_EARLIER_LAYOUT: &str = "
    CREATE TABLE legacy_schema AS SELECT schema_id, schema_name, path, path_is_relative,
        begin_snapshot, end_snapshot FROM ducklake_schema;
    DROP TABLE ducklake_schema;
    ALTER TABLE legacy_schema RENAME TO ducklake_schema;
    CREATE TABLE legacy_table AS SELECT table_id, schema_id, table_name, path,
        path_is_relative, begin_snapshot, end_snapshot FROM ducklake_table;
    DROP TABLE ducklake_table;
    ALTER TABLE legacy_table RENAME TO ducklake_table;
    CREATE TABLE legacy_data_file AS SELECT data_file_id, table_id, path, path_is_relative,
        file_size_bytes, footer_size, encryption_key, record_count, row_id_start, mapping_id,
        begin_snapshot, end_snapshot, partial_max, partition_id FROM ducklake_data_file;
    DROP TABLE ducklake_data_file;
    ALTER TABLE legacy_data_file RENAME TO ducklake_data_file;
    CREATE TABLE legacy_delete_file AS SELECT delete_file_id, data_file_id, table_id, path,
        path_is_relative, file_size_bytes, footer_size, encryption_key, delete_count,
        begin_snapshot, end_snapshot, partial_max FROM ducklake_delete_file;
    DROP TABLE ducklake_delete_file;
    ALTER TABLE legacy_delete_file RENAME TO ducklake_delete_file;";

/// A catalog written in this crate's earlier layout (columns out of
/// official's order or missing, id watermarks unset) keeps its column order on
/// initialization, so a release before 0.10 still writes it, and the crate
/// writes it too; the migration then brings it to official's layout, keeping
/// its rows, and official DuckDB writes it, a rename included.
async fn migration_brings_an_earlier_layout_to_duckdb_writes(kind: Kind) {
    let lake = Lake::new(kind).await;
    write_ids(lake.writer().await, "readable", vec![7, 11, 13]).await;
    lake.execute_metadata_sql(SQL_EARLIER_LAYOUT).await;
    if let Backend::Postgres {
        ..
    } = lake.backend
    {
        // Releases before 0.10 keyed these ids `GENERATED ALWAYS AS IDENTITY`.
        for (table, id) in [
            ("ducklake_schema", "schema_id"),
            ("ducklake_table", "table_id"),
            ("ducklake_data_file", "data_file_id"),
            ("ducklake_delete_file", "delete_file_id"),
        ] {
            lake.execute_metadata_sql(&format!(
                "ALTER TABLE {table} ALTER COLUMN {id} SET NOT NULL;
                 ALTER TABLE {table} ALTER COLUMN {id} ADD GENERATED ALWAYS AS IDENTITY;
                 ALTER TABLE {table} ADD PRIMARY KEY ({id});
                 SELECT setval(pg_get_serial_sequence('{table}', '{id}'),
                               GREATEST(1, COALESCE((SELECT MAX({id}) FROM {table}), 0)))"
            ))
            .await;
        }
    }
    // The DuckDB writer left the watermarks at their column defaults, the other
    // writers had no watermark columns, and none kept a `next_catalog_id`
    // counter.
    lake.execute_metadata_sql(match lake.backend {
        Backend::DuckDb => {
            "UPDATE ducklake_snapshot SET next_catalog_id = 1, next_file_id = 0;
             DELETE FROM ducklake_metadata WHERE key IN ('next_catalog_id', 'next_file_id');"
        },
        Backend::Mysql {
            ..
        } => {
            "ALTER TABLE ducklake_snapshot DROP COLUMN next_catalog_id;
             ALTER TABLE ducklake_snapshot DROP COLUMN next_file_id;
             DELETE FROM ducklake_metadata WHERE `key` = 'next_catalog_id';"
        },
        _ => {
            "ALTER TABLE ducklake_snapshot DROP COLUMN next_catalog_id;
             ALTER TABLE ducklake_snapshot DROP COLUMN next_file_id;
             DELETE FROM ducklake_metadata WHERE key = 'next_catalog_id';"
        },
    })
    .await;

    let writer = lake.writer().await;
    let initialized_columns = lake.column_order("ducklake_table").await;
    write_ids(writer, "unmigrated", vec![17]).await;
    lake.migrate().await;
    let schema_columns = lake.column_order("ducklake_schema").await;
    let table_columns = lake.column_order("ducklake_table").await;
    let oracle = lake.official();
    oracle
        .execute_batch(
            "CREATE TABLE lake.main.official AS SELECT 31 AS id;
             ALTER TABLE lake.main.official RENAME TO renamed;",
        )
        .unwrap();

    assert_eq!(
        initialized_columns,
        [
            "table_id",
            "schema_id",
            "table_name",
            "path",
            "path_is_relative",
            "begin_snapshot",
            "end_snapshot",
            "table_uuid"
        ]
    );

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
    assert_eq!(
        table_columns,
        [
            "table_id",
            "table_uuid",
            "begin_snapshot",
            "end_snapshot",
            "schema_id",
            "table_name",
            "path",
            "path_is_relative"
        ]
    );
    assert_eq!(official_ids(&oracle, "readable"), vec![7, 11, 13]);
    assert_eq!(official_ids(&oracle, "unmigrated"), vec![17]);
    assert_eq!(official_ids(&oracle, "renamed"), vec![31]);
}

#[tokio::test(flavor = "multi_thread")]
async fn migration_brings_an_earlier_duckdb_catalog_to_duckdb_writes() {
    migration_brings_an_earlier_layout_to_duckdb_writes(Kind::DuckDb).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn migration_brings_an_earlier_sqlite_catalog_to_duckdb_writes() {
    migration_brings_an_earlier_layout_to_duckdb_writes(Kind::Sqlite).await;
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn duckdb_and_the_crate_write_one_postgres_catalog_in_turn() {
    duckdb_and_the_crate_write_one_catalog_in_turn(Kind::Postgres).await;
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn migration_brings_an_earlier_postgres_catalog_to_duckdb_writes() {
    migration_brings_an_earlier_layout_to_duckdb_writes(Kind::Postgres).await;
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn duckdb_and_the_crate_write_one_mysql_catalog_in_turn() {
    duckdb_and_the_crate_write_one_catalog_in_turn(Kind::Mysql).await;
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn migration_brings_an_earlier_mysql_catalog_to_duckdb_writes() {
    migration_brings_an_earlier_layout_to_duckdb_writes(Kind::Mysql).await;
}
