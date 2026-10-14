//! Maintenance reclaims catalog-inlined physical storage, as official DuckLake does.
//!
//! Ports the scenarios of official's `test/sql/data_inlining/` reclaim tests
//! (`inlined_data_table_leak`, `inlined_delete_table_drop_cleanup`,
//! `superseded_inlined_table_flush_drop`, `empty_inlined_tables_alter_cleanup`) to every
//! backend with a maintenance path: SQLite, DuckDB, MySQL, and both PostgreSQL layouts.
//! Each asserts on the physical tables in the metadata database, on
//! `ducklake_inlined_data_tables`, and on the values a still-live table reads back.
//!
//! A flush deletes the rows it moves, as official's does, so the flush that empties a
//! superseded table drops it.

#![cfg(feature = "write-sqlite")]

use std::sync::Arc;

use arrow::array::{Array, Int32Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use arrow::util::display::array_value_to_string;
use chrono::{Duration, Utc};
use datafusion::prelude::SessionContext;
use datafusion_ducklake::maintenance::ExpireCriteria;
use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeTableWriter, DuckLakeWriteOptions, MetadataProvider, MetadataWriter,
    SqliteMetadataProvider, SqliteMetadataWriter, WriteResult,
};
use object_store::local::LocalFileSystem;
use sqlx::AssertSqlSafe;
use tempfile::TempDir;

/// The backend under test, with what its maintenance calls need.
enum Kind {
    Sqlite {
        writer: Arc<SqliteMetadataWriter>,
        url: String,
    },
    #[cfg(feature = "write-mysql")]
    Mysql {
        writer: Arc<datafusion_ducklake::MySqlMetadataWriter>,
        url: String,
    },
    #[cfg(feature = "write-postgres")]
    Multicatalog {
        manager: datafusion_ducklake::multicatalog::MulticatalogManager,
        catalog: String,
        catalog_id: i64,
    },
    #[cfg(feature = "write-postgres")]
    PostgresSingle {
        url: String,
    },
}

/// A raw connection to the metadata database.
enum Raw {
    Sqlite(sqlx::SqlitePool),
    #[cfg(feature = "write-mysql")]
    Mysql(sqlx::MySqlPool),
    #[cfg(feature = "write-postgres")]
    Postgres(sqlx::PgPool),
}

struct Lake {
    kind: Kind,
    writer: Arc<dyn MetadataWriter>,
    raw: Raw,
    _data: Arc<TempDir>,
}

async fn sqlite_lake() -> (Lake, TempDir) {
    let temp = TempDir::new().unwrap();
    let url = format!("sqlite:{}?mode=rwc", temp.path().join("meta.db").display());
    let writer = Arc::new(SqliteMetadataWriter::new_with_init(&url).await.unwrap());
    let data = Arc::new(TempDir::new().unwrap());
    writer.set_data_path(data.path().to_str().unwrap()).unwrap();
    let raw = Raw::Sqlite(sqlx::SqlitePool::connect(&url).await.unwrap());
    let lake = Lake {
        writer: writer.clone(),
        kind: Kind::Sqlite {
            writer,
            url,
        },
        raw,
        _data: data,
    };
    (lake, temp)
}

impl Lake {
    async fn execute(&self, sql: &str) {
        match &self.raw {
            Raw::Sqlite(pool) => {
                sqlx::query(AssertSqlSafe(sql)).execute(pool).await.unwrap();
            },
            #[cfg(feature = "write-mysql")]
            Raw::Mysql(pool) => {
                sqlx::query(AssertSqlSafe(sql)).execute(pool).await.unwrap();
            },
            #[cfg(feature = "write-postgres")]
            Raw::Postgres(pool) => {
                sqlx::query(AssertSqlSafe(sql)).execute(pool).await.unwrap();
            },
        }
    }

    async fn scalar(&self, sql: &str) -> i64 {
        match &self.raw {
            Raw::Sqlite(pool) => sqlx::query_scalar(AssertSqlSafe(sql))
                .fetch_one(pool)
                .await
                .unwrap(),
            #[cfg(feature = "write-mysql")]
            Raw::Mysql(pool) => sqlx::query_scalar(AssertSqlSafe(sql))
                .fetch_one(pool)
                .await
                .unwrap(),
            #[cfg(feature = "write-postgres")]
            Raw::Postgres(pool) => sqlx::query_scalar(AssertSqlSafe(sql))
                .fetch_one(pool)
                .await
                .unwrap(),
        }
    }

    async fn strings(&self, sql: &str) -> Vec<String> {
        let mut names: Vec<String> = match &self.raw {
            Raw::Sqlite(pool) => sqlx::query_scalar(AssertSqlSafe(sql))
                .fetch_all(pool)
                .await
                .unwrap(),
            #[cfg(feature = "write-mysql")]
            Raw::Mysql(pool) => sqlx::query_scalar(AssertSqlSafe(sql))
                .fetch_all(pool)
                .await
                .unwrap(),
            #[cfg(feature = "write-postgres")]
            Raw::Postgres(pool) => sqlx::query_scalar(AssertSqlSafe(sql))
                .fetch_all(pool)
                .await
                .unwrap(),
        };
        names.sort();
        names
    }

    /// Physical inlined data and inlined delete tables in the metadata database.
    async fn physical_inlined_tables(&self) -> Vec<String> {
        let sql = match &self.raw {
            Raw::Sqlite(_) => "SELECT name FROM sqlite_master WHERE type = 'table'",
            #[cfg(feature = "write-mysql")]
            Raw::Mysql(_) => {
                "SELECT CAST(table_name AS CHAR) FROM information_schema.tables
                 WHERE table_schema = DATABASE()"
            },
            #[cfg(feature = "write-postgres")]
            Raw::Postgres(_) => {
                "SELECT tablename::text FROM pg_tables WHERE schemaname = current_schema()"
            },
        };
        self.strings(sql)
            .await
            .into_iter()
            .filter(|name| {
                name.starts_with("ducklake_inlined_") && name != "ducklake_inlined_data_tables"
            })
            .collect()
    }

    /// Physical inlined data tables no registry row names. Only the MySQL writer
    /// leaves these: it creates an inlined table before its commit settles the schema
    /// version the name carries, and leaves the one it mispredicted. No reader lists
    /// them, and official reclaims nothing it has not registered, so neither does
    /// maintenance here.
    async fn strays(&self) -> Vec<String> {
        let registry = self.registry().await;
        self.physical_inlined_tables()
            .await
            .into_iter()
            .filter(|name| name.starts_with("ducklake_inlined_data_") && !registry.contains(name))
            .collect()
    }

    /// [`Self::physical_inlined_tables`] without `strays`.
    async fn physical_without(&self, strays: &[String]) -> Vec<String> {
        self.physical_inlined_tables()
            .await
            .into_iter()
            .filter(|name| !strays.contains(name))
            .collect()
    }

    /// `ducklake_inlined_data_tables`, as `table_name` strings.
    async fn registry(&self) -> Vec<String> {
        self.strings("SELECT table_name FROM ducklake_inlined_data_tables")
            .await
    }

    async fn provider(&self) -> Arc<dyn MetadataProvider> {
        match &self.kind {
            Kind::Sqlite {
                url,
                ..
            } => Arc::new(SqliteMetadataProvider::new(url).await.unwrap()),
            #[cfg(feature = "write-mysql")]
            Kind::Mysql {
                url,
                ..
            } => Arc::new(
                datafusion_ducklake::MySqlMetadataProvider::new(url)
                    .await
                    .unwrap(),
            ),
            #[cfg(feature = "write-postgres")]
            Kind::Multicatalog {
                manager,
                catalog_id,
                ..
            } => Arc::new(manager.provider(*catalog_id).await.unwrap()),
            #[cfg(feature = "write-postgres")]
            Kind::PostgresSingle {
                url,
            } => Arc::new(
                datafusion_ducklake::PostgresMetadataProvider::new(url)
                    .await
                    .unwrap(),
            ),
        }
    }

    /// Expire every snapshot but the latest.
    async fn expire_all(&self) -> usize {
        let criteria = ExpireCriteria::OlderThan(Utc::now() + Duration::minutes(1));
        match &self.kind {
            Kind::Sqlite {
                writer,
                ..
            } => writer.expire_snapshots(criteria).unwrap().len(),
            #[cfg(feature = "write-mysql")]
            Kind::Mysql {
                writer,
                ..
            } => writer.expire_snapshots(criteria).unwrap().len(),
            #[cfg(feature = "write-postgres")]
            Kind::Multicatalog {
                manager,
                catalog,
                ..
            } => manager
                .expire_snapshots_in_catalog(catalog, criteria)
                .await
                .unwrap()
                .len(),
            #[cfg(feature = "write-postgres")]
            Kind::PostgresSingle {
                ..
            } => unreachable!("the single-catalog PostgreSQL writer has no expire"),
        }
    }

    /// Drop a table at a new snapshot.
    async fn drop_table(&self, table: &str) {
        match &self.kind {
            Kind::Sqlite {
                writer,
                ..
            } => assert!(writer.drop_table("main", table).unwrap()),
            // The MySQL writer has no DROP TABLE: end the table's rows at a new
            // snapshot, as a drop does.
            #[cfg(feature = "write-mysql")]
            Kind::Mysql {
                ..
            } => {
                let table_id = self.table_id(table).await;
                let snapshot = self
                    .scalar("SELECT MAX(snapshot_id) FROM ducklake_snapshot")
                    .await;
                let version = self
                    .scalar("SELECT MAX(schema_version) FROM ducklake_snapshot")
                    .await;
                let next = snapshot + 1;
                self.execute(&format!(
                    "INSERT INTO ducklake_snapshot (snapshot_id, schema_version)
                     VALUES ({next}, {})",
                    version + 1
                ))
                .await;
                for child in [
                    "ducklake_table",
                    "ducklake_column",
                    "ducklake_data_file",
                    "ducklake_delete_file",
                ] {
                    self.execute(&format!(
                        "UPDATE {child} SET end_snapshot = {next}
                         WHERE table_id = {table_id} AND end_snapshot IS NULL"
                    ))
                    .await;
                }
            },
            #[cfg(feature = "write-postgres")]
            Kind::Multicatalog {
                manager,
                catalog,
                ..
            } => assert!(
                manager
                    .drop_table_in_catalog(catalog, "main", table)
                    .await
                    .unwrap()
            ),
            #[cfg(feature = "write-postgres")]
            Kind::PostgresSingle {
                ..
            } => unreachable!("not exercised on the single-catalog layout"),
        }
    }

    async fn table_id(&self, table: &str) -> i64 {
        let provider = self.provider().await;
        let snapshot = provider.get_current_snapshot().unwrap();
        let schema = provider
            .get_schema_by_name("main", snapshot)
            .unwrap()
            .unwrap();
        provider
            .get_table_by_name(schema.schema_id, table, snapshot)
            .unwrap()
            .unwrap()
            .table_id
    }

    fn table_writer(&self, inlining_row_limit: usize) -> DuckLakeTableWriter {
        let options =
            DuckLakeWriteOptions::default().with_data_inlining_row_limit(inlining_row_limit);
        DuckLakeTableWriter::new(self.writer.clone(), Arc::new(LocalFileSystem::new()))
            .unwrap()
            .with_options(&options)
    }

    /// `table`'s rows as `i|j` strings (`j` only when `with_j`), at `snapshot` or
    /// the latest one.
    async fn rows(&self, table: &str, with_j: bool, snapshot: Option<i64>) -> Vec<String> {
        let provider = self.provider().await;
        let catalog = match snapshot {
            Some(snapshot) => DuckLakeCatalog::with_snapshot(provider, snapshot).unwrap(),
            None => DuckLakeCatalog::with_writer(provider, self.writer.clone()).unwrap(),
        };
        let ctx = SessionContext::new();
        ctx.register_catalog("lake", Arc::new(catalog));
        let columns = if with_j {
            "i, j"
        } else {
            "i"
        };
        let batches = ctx
            .sql(&format!(
                "SELECT {columns} FROM lake.main.{table} ORDER BY i"
            ))
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let mut rows = Vec::new();
        for batch in batches {
            for row in 0..batch.num_rows() {
                rows.push(
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
                        .join("|"),
                );
            }
        }
        rows
    }

    #[cfg(feature = "write-postgres")]
    async fn sql(&self, statement: &str) {
        let catalog =
            DuckLakeCatalog::with_writer(self.provider().await, self.writer.clone()).unwrap();
        let ctx = SessionContext::new();
        ctx.register_catalog("lake", Arc::new(catalog));
        ctx.sql(statement).await.unwrap().collect().await.unwrap();
    }

    /// Flush `table`'s inlined rows to Parquet.
    async fn flush(&self, table: &str) -> Option<WriteResult> {
        let provider = self.provider().await;
        let snapshot = provider.get_current_snapshot().unwrap();
        self.table_writer(0)
            .flush_table_inlined_data(provider, "main", table, snapshot)
            .await
            .unwrap()
    }

    /// Store an inlined deletion of `row_ids` of `table`'s live Parquet file in
    /// `ducklake_inlined_delete_<table_id>`, with official's layout.
    async fn inline_delete_parquet_rows(&self, table: &str, row_ids: &[i64]) -> String {
        let table_id = self.table_id(table).await;
        let snapshot = self.provider().await.get_current_snapshot().unwrap();
        let file_id = self
            .scalar(&format!(
                "SELECT MIN(data_file_id) FROM ducklake_data_file
                 WHERE table_id = {table_id} AND end_snapshot IS NULL"
            ))
            .await;
        let name = format!("ducklake_inlined_delete_{table_id}");
        self.execute(&format!(
            "CREATE TABLE {name}(file_id BIGINT, row_id BIGINT, begin_snapshot BIGINT)"
        ))
        .await;
        for row_id in row_ids {
            self.execute(&format!(
                "INSERT INTO {name} VALUES ({file_id}, {row_id}, {snapshot})"
            ))
            .await;
        }
        name
    }
}

fn batch_i(values: &[i32]) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![Field::new("i", DataType::Int32, true)]));
    RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(values.to_vec()))]).unwrap()
}

fn batch_ij(values: &[(i32, i32)]) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("i", DataType::Int32, true),
        Field::new("j", DataType::Int32, true),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(
                values.iter().map(|(i, _)| *i).collect::<Vec<_>>(),
            )),
            Arc::new(Int32Array::from(
                values.iter().map(|(_, j)| *j).collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap()
}

fn strings(values: &[&str]) -> Vec<String> {
    values.iter().map(|value| value.to_string()).collect()
}

/// `inlined_data_table_leak` + `inlined_delete_table_drop_cleanup`: a dropped table's
/// inlined data tables (one per schema version) and its inlined delete table go once
/// expire reclaims the table; another table's inlined storage stays and reads back.
async fn dropped_table_storage_is_reclaimed(lake: &Lake) {
    let inline = lake.table_writer(10);
    let t = inline
        .write_table("main", "t", &[batch_i(&[1, 2, 3])])
        .await
        .unwrap();
    inline
        .append_table("main", "t", &[batch_ij(&[(4, 40)])])
        .await
        .unwrap();
    let parquet_rows = (100..120).map(|i| (i, i)).collect::<Vec<_>>();
    lake.table_writer(0)
        .append_table("main", "t", &[batch_ij(&parquet_rows)])
        .await
        .unwrap();
    let delete_table = lake.inline_delete_parquet_rows("t", &[0, 1]).await;
    let keep = inline
        .write_table("main", "keep", &[batch_i(&[7, 8])])
        .await
        .unwrap();

    let t_rows = lake.rows("t", true, None).await;
    assert_eq!(t_rows.len(), 4 + 18, "{t_rows:?}");
    assert_eq!(
        t_rows[..5],
        strings(&["1|NULL", "2|NULL", "3|NULL", "4|40", "102|102"])
    );
    let strays = lake.strays().await;
    let t_prefix = format!("ducklake_inlined_data_{}_", t.table_id);
    let t_storage = lake
        .physical_without(&strays)
        .await
        .into_iter()
        .filter(|name| name.starts_with(&t_prefix) || *name == delete_table)
        .collect::<Vec<_>>();
    assert_eq!(t_storage.len(), 3, "{t_storage:?}");

    lake.drop_table("t").await;
    assert!(lake.expire_all().await > 0);

    let physical = lake.physical_without(&strays).await;
    let keep_prefix = format!("ducklake_inlined_data_{}_", keep.table_id);
    assert!(
        physical.iter().all(|name| !t_storage.contains(name)),
        "{physical:?}"
    );
    assert_eq!(physical.len(), 1, "{physical:?}");
    assert!(physical[0].starts_with(&keep_prefix), "{physical:?}");
    assert_eq!(lake.registry().await, physical);
    assert_eq!(lake.rows("keep", false, None).await, strings(&["7", "8"]));
}

/// `superseded_inlined_table_flush_drop`: the flush that empties a superseded inlined
/// table drops it and keeps the current version's table; time travel reads the flushed
/// rows from Parquet.
async fn superseded_table_is_dropped_by_the_flush_that_empties_it(lake: &Lake) {
    let inline = lake.table_writer(10);
    let created = inline
        .write_table("main", "t1", &[batch_i(&[1, 2, 3])])
        .await
        .unwrap();
    inline
        .append_table("main", "t1", &[batch_ij(&[(4, 40)])])
        .await
        .unwrap();
    let registry = lake.registry().await;
    assert_eq!(registry.len(), 2, "{registry:?}");
    let all = strings(&["1|NULL", "2|NULL", "3|NULL", "4|40"]);
    let strays = lake.strays().await;

    let flushed = lake.flush("t1").await.unwrap();
    assert_eq!(flushed.records_written, 4);
    let current = registry.last().unwrap().clone();
    assert_eq!(lake.registry().await, vec![current.clone()]);
    assert_eq!(lake.physical_without(&strays).await, vec![current.clone()]);
    assert_eq!(
        lake.scalar(&format!("SELECT COUNT(*) FROM {current}"))
            .await,
        0
    );
    assert_eq!(lake.rows("t1", true, None).await, all);
    assert_eq!(
        lake.rows("t1", false, Some(created.snapshot_id)).await,
        strings(&["1", "2", "3"])
    );
    assert!(lake.flush("t1").await.is_none());

    inline
        .append_table("main", "t1", &[batch_ij(&[(5, 50)])])
        .await
        .unwrap();
    assert_eq!(lake.registry().await, vec![current]);
    assert_eq!(
        lake.rows("t1", true, None).await,
        strings(&["1|NULL", "2|NULL", "3|NULL", "4|40", "5|50"])
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn sqlite_dropped_table_storage_is_reclaimed() {
    let (lake, _temp) = sqlite_lake().await;
    dropped_table_storage_is_reclaimed(&lake).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn sqlite_superseded_table_is_dropped_by_the_flush_that_empties_it() {
    let (lake, _temp) = sqlite_lake().await;
    superseded_table_is_dropped_by_the_flush_that_empties_it(&lake).await;
}

/// `empty_inlined_tables_alter_cleanup`, on tables an official writer left behind: a
/// superseded inlined table with no rows goes on the next flush.
#[tokio::test(flavor = "multi_thread")]
async fn sqlite_flush_drops_empty_superseded_tables() {
    let (lake, _temp) = sqlite_lake().await;
    let created = lake
        .table_writer(10)
        .write_table("main", "a", &[batch_i(&[1])])
        .await
        .unwrap();
    let table_id = created.table_id;
    let live = lake.registry().await;
    // Official creates an inlined table per schema version at ALTER time, empty.
    for version in 100..103 {
        let name = format!("ducklake_inlined_data_{table_id}_{version}");
        lake.execute(&format!(
            "CREATE TABLE {name}(row_id BIGINT, begin_snapshot BIGINT, end_snapshot BIGINT, i INTEGER)"
        ))
        .await;
        lake.execute(&format!(
            "INSERT INTO ducklake_inlined_data_tables VALUES ({table_id}, '{name}', {version})"
        ))
        .await;
    }
    // The table with the newest version is the current one. The flush empties the
    // superseded table holding the live rows, and drops it with the empty ones.
    let flushed = lake.flush("a").await.unwrap();
    assert_eq!(flushed.records_written, 1);
    let newest = format!("ducklake_inlined_data_{table_id}_102");
    assert!(!live.contains(&newest));
    assert_eq!(lake.registry().await, vec![newest.clone()]);
    assert_eq!(lake.physical_inlined_tables().await, vec![newest]);
    assert_eq!(lake.rows("a", false, None).await, strings(&["1"]));
}

/// Reclaim drops whatever name the registry holds, as official does, quoted as an
/// identifier: a superseded empty table goes with the flush, a dead table's goes
/// with expire, and a name holding a quote and SQL drops only that table.
#[tokio::test(flavor = "multi_thread")]
async fn sqlite_reclaim_drops_any_registered_name() {
    let (lake, _temp) = sqlite_lake().await;
    let inline = lake.table_writer(10);
    let a = inline
        .write_table("main", "a", &[batch_i(&[1])])
        .await
        .unwrap();
    let b = inline
        .write_table("main", "b", &[batch_i(&[2])])
        .await
        .unwrap();
    let superseded = "superseded a";
    let dead = r#"x"; DROP TABLE ducklake_snapshot; --"#;
    // `a`'s extra table is below its live version; `b`'s is its newest.
    for (table_id, name, version) in [(a.table_id, superseded, -1), (b.table_id, dead, 1000)] {
        let quoted = format!("\"{}\"", name.replace('"', "\"\""));
        lake.execute(&format!(
            "CREATE TABLE {quoted}(row_id BIGINT, begin_snapshot BIGINT, end_snapshot BIGINT, i INTEGER)"
        ))
        .await;
        lake.execute(&format!(
            "INSERT INTO ducklake_inlined_data_tables VALUES ({table_id}, '{}', {version})",
            name.replace('\'', "''")
        ))
        .await;
    }
    let exists = |name: &'static str| {
        let lake = &lake;
        async move {
            lake.scalar(&format!(
                "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = '{}'",
                name.replace('\'', "''")
            ))
            .await
        }
    };

    lake.flush("a").await.unwrap();
    assert_eq!(exists(superseded).await, 0);
    assert_eq!(exists(dead).await, 1);

    lake.drop_table("b").await;
    assert!(lake.expire_all().await > 0);
    assert_eq!(exists(dead).await, 0);
    assert!(lake.scalar("SELECT COUNT(*) FROM ducklake_snapshot").await > 0);
    assert_eq!(lake.rows("a", false, None).await, strings(&["1"]));
}

#[cfg(feature = "write-mysql")]
mod mysql {
    use super::*;
    use testcontainers::ContainerAsync;
    use testcontainers::runners::AsyncRunner;
    use testcontainers_modules::mysql::Mysql;

    async fn mysql_lake() -> (Lake, ContainerAsync<Mysql>) {
        let container = Mysql::default().start().await.unwrap();
        let port = container.get_host_port_ipv4(3306).await.unwrap();
        let url = format!("mysql://root@127.0.0.1:{port}/test");
        let writer = Arc::new(
            datafusion_ducklake::MySqlMetadataWriter::new_with_init(&url)
                .await
                .unwrap(),
        );
        let data = Arc::new(TempDir::new().unwrap());
        writer.set_data_path(data.path().to_str().unwrap()).unwrap();
        let raw = Raw::Mysql(sqlx::MySqlPool::connect(&url).await.unwrap());
        let lake = Lake {
            writer: writer.clone(),
            kind: Kind::Mysql {
                writer,
                url,
            },
            raw,
            _data: data,
        };
        (lake, container)
    }

    #[tokio::test(flavor = "multi_thread")]
    #[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
    async fn mysql_dropped_table_storage_is_reclaimed() {
        let (lake, _container) = mysql_lake().await;
        dropped_table_storage_is_reclaimed(&lake).await;
        let dead = lake
            .scalar("SELECT COUNT(*) FROM ducklake_table WHERE end_snapshot IS NOT NULL")
            .await;
        assert_eq!(dead, 0);
    }

    #[tokio::test(flavor = "multi_thread")]
    #[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
    async fn mysql_superseded_table_is_dropped_by_the_flush_that_empties_it() {
        let (lake, _container) = mysql_lake().await;
        superseded_table_is_dropped_by_the_flush_that_empties_it(&lake).await;
    }

    /// MySQL commits DDL implicitly, so an expire is not atomic: it commits the
    /// metadata first and drops the inlined storage after. An expire that fails
    /// between the two — here a writer whose user may not `DROP` — is finished by the
    /// next call, one with no snapshot left to expire.
    #[tokio::test(flavor = "multi_thread")]
    #[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
    async fn mysql_interrupted_expire_is_finished_by_the_next() {
        let (lake, _container) = mysql_lake().await;
        let inline = lake.table_writer(10);
        let t = inline
            .write_table("main", "t", &[batch_i(&[1, 2, 3])])
            .await
            .unwrap();
        inline
            .append_table("main", "t", &[batch_ij(&[(4, 40)])])
            .await
            .unwrap();
        let parquet_rows = (100..110).map(|i| (i, i)).collect::<Vec<_>>();
        lake.table_writer(0)
            .append_table("main", "t", &[batch_ij(&parquet_rows)])
            .await
            .unwrap();
        lake.inline_delete_parquet_rows("t", &[0]).await;
        inline
            .write_table("main", "keep", &[batch_i(&[7])])
            .await
            .unwrap();
        lake.drop_table("t").await;
        let strays = lake.strays().await;
        let storage = lake.physical_without(&strays).await;
        assert_eq!(storage.len(), 4, "{storage:?}");
        let count = |table: &str| {
            format!(
                "SELECT COUNT(*) FROM {table} WHERE table_id = {}",
                t.table_id
            )
        };
        assert!(lake.scalar(&count("ducklake_column")).await > 0);
        assert!(lake.scalar(&count("ducklake_data_file")).await > 0);

        // A writer whose user can do everything but DROP: its expire commits the
        // metadata transaction and then fails at the first physical drop.
        lake.execute("CREATE USER 'nodrop'@'%' IDENTIFIED BY 'nodrop'")
            .await;
        lake.execute(
            "GRANT SELECT, INSERT, UPDATE, DELETE, CREATE, ALTER, INDEX ON test.* \
             TO 'nodrop'@'%'",
        )
        .await;
        let Kind::Mysql {
            url,
            ..
        } = &lake.kind
        else {
            unreachable!()
        };
        let restricted_url = url.replace("mysql://root@", "mysql://nodrop:nodrop@");
        let restricted = datafusion_ducklake::MySqlMetadataWriter::new(&restricted_url)
            .await
            .unwrap();
        let criteria = ExpireCriteria::OlderThan(Utc::now() + Duration::minutes(1));
        let error = restricted.expire_snapshots(criteria).unwrap_err();
        assert!(
            error.to_string().to_ascii_lowercase().contains("drop"),
            "{error}"
        );

        // The crash point: snapshots and the dead table's column and file rows are
        // gone; its `ducklake_table` and registry rows and its inlined storage are not.
        assert_eq!(
            lake.scalar("SELECT COUNT(*) FROM ducklake_snapshot").await,
            1
        );
        assert_eq!(lake.scalar(&count("ducklake_column")).await, 0);
        assert_eq!(lake.scalar(&count("ducklake_data_file")).await, 0);
        assert_eq!(lake.scalar(&count("ducklake_table")).await, 1);
        assert_eq!(lake.scalar(&count("ducklake_inlined_data_tables")).await, 2);
        assert_eq!(lake.physical_without(&strays).await, storage);

        assert_eq!(lake.expire_all().await, 0);
        let physical = lake.physical_without(&strays).await;
        assert_eq!(physical.len(), 1, "{physical:?}");
        assert_eq!(lake.registry().await, physical);
        for table in [
            "ducklake_table",
            "ducklake_column",
            "ducklake_data_file",
            "ducklake_inlined_data_tables",
        ] {
            assert_eq!(lake.scalar(&count(table)).await, 0, "{table}");
        }
        assert_eq!(lake.rows("keep", false, None).await, strings(&["7"]));
    }

    /// A catalog this crate's MySQL writer created before data inlining has no
    /// `ducklake_inlined_data_tables`; expire and the superseded-table drop reclaim
    /// nothing there instead of failing.
    #[tokio::test(flavor = "multi_thread")]
    #[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
    async fn mysql_reclaim_without_the_registry() {
        let (lake, _container) = mysql_lake().await;
        let parquet = lake.table_writer(0);
        parquet
            .write_table("main", "t", &[batch_i(&[1, 2])])
            .await
            .unwrap();
        parquet
            .write_table("main", "keep", &[batch_i(&[7])])
            .await
            .unwrap();
        lake.drop_table("t").await;
        lake.execute("DROP TABLE ducklake_inlined_data_tables")
            .await;

        assert!(lake.expire_all().await > 0);
        lake.writer.drop_empty_superseded_inlined_tables().unwrap();
        assert_eq!(
            lake.scalar("SELECT COUNT(*) FROM ducklake_table WHERE end_snapshot IS NOT NULL")
                .await,
            0
        );
        assert_eq!(lake.rows("keep", false, None).await, strings(&["7"]));
    }
}

#[cfg(feature = "write-postgres")]
mod postgres {
    use super::*;
    use datafusion_ducklake::multicatalog::{MulticatalogManager, initialize_multicatalog_schema};
    use testcontainers::ContainerAsync;
    use testcontainers::runners::AsyncRunner;
    use testcontainers_modules::postgres::Postgres;

    async fn start() -> (sqlx::PgPool, String, ContainerAsync<Postgres>) {
        let container = Postgres::default().start().await.unwrap();
        let port = container.get_host_port_ipv4(5432).await.unwrap();
        let url = format!("postgresql://postgres:postgres@127.0.0.1:{port}/postgres");
        let pool = sqlx::PgPool::connect(&url).await.unwrap();
        (pool, url, container)
    }

    async fn catalog_lake(pool: &sqlx::PgPool, catalog: &str) -> Lake {
        let manager = MulticatalogManager::new(pool.clone());
        let catalog_id = manager.create_catalog(catalog).await.unwrap();
        let writer = Arc::new(manager.writer(catalog_id).await.unwrap());
        let data = Arc::new(TempDir::new().unwrap());
        writer.set_data_path(data.path().to_str().unwrap()).unwrap();
        Lake {
            writer,
            kind: Kind::Multicatalog {
                manager,
                catalog: catalog.to_string(),
                catalog_id,
            },
            raw: Raw::Postgres(pool.clone()),
            _data: data,
        }
    }

    async fn multicatalog_lake() -> (Lake, ContainerAsync<Postgres>) {
        let (pool, _url, container) = start().await;
        initialize_multicatalog_schema(&pool).await.unwrap();
        (catalog_lake(&pool, "lake").await, container)
    }

    #[tokio::test(flavor = "multi_thread")]
    #[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
    async fn multicatalog_dropped_table_storage_is_reclaimed() {
        let (lake, _container) = multicatalog_lake().await;
        dropped_table_storage_is_reclaimed(&lake).await;
    }

    #[tokio::test(flavor = "multi_thread")]
    #[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
    async fn multicatalog_superseded_table_is_dropped_by_the_flush_that_empties_it() {
        let (lake, _container) = multicatalog_lake().await;
        superseded_table_is_dropped_by_the_flush_that_empties_it(&lake).await;
    }

    /// Expire and flush reach only their own catalog's inlined storage, and dropping a
    /// catalog drops all of it in its transaction.
    #[tokio::test(flavor = "multi_thread")]
    #[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
    async fn multicatalog_reclaim_is_scoped_and_drop_catalog_drops_storage() {
        let (pool, _url, _container) = start().await;
        initialize_multicatalog_schema(&pool).await.unwrap();
        let a = catalog_lake(&pool, "a").await;
        let b = catalog_lake(&pool, "b").await;

        // b: an inline-deleted row and an empty superseded table.
        let b_inline = b.table_writer(10);
        let b_t = b_inline
            .write_table("main", "t", &[batch_i(&[1, 2, 3])])
            .await
            .unwrap();
        b.sql("DELETE FROM lake.main.t WHERE i = 2").await;
        let b_live = b.registry().await;
        let b_superseded = format!("ducklake_inlined_data_{}_0", b_t.table_id);
        b.execute(&format!(
            "CREATE TABLE {b_superseded}(row_id BIGINT, begin_snapshot BIGINT, end_snapshot BIGINT, i INTEGER)"
        ))
        .await;
        b.execute(&format!(
            "INSERT INTO ducklake_inlined_data_tables VALUES ({}, '{b_superseded}', 0)",
            b_t.table_id
        ))
        .await;

        // a: inlined data, an inlined delete table, and its own expire + flush.
        let a_inline = a.table_writer(10);
        a_inline
            .write_table("main", "t", &[batch_i(&[5])])
            .await
            .unwrap();
        let parquet_rows = (100..110).map(|i| (i, i)).collect::<Vec<_>>();
        a.table_writer(0)
            .append_table("main", "t", &[batch_ij(&parquet_rows)])
            .await
            .unwrap();
        let a_delete = a.inline_delete_parquet_rows("t", &[0]).await;
        a.expire_all().await;
        assert!(a.flush("t").await.is_some());

        let b_count = format!("SELECT COUNT(*) FROM {}", b_live[0]);
        assert_eq!(
            b.scalar(&b_count).await,
            3,
            "a's expire and flush left b's rows alone"
        );
        let physical = b.physical_inlined_tables().await;
        assert!(
            physical.contains(&b_superseded),
            "a's flush left b's tables alone"
        );
        assert_eq!(b.rows("t", false, None).await, strings(&["1", "3"]));

        let a_storage = a
            .strings(&format!(
                "SELECT table_name FROM ducklake_inlined_data_tables
                 WHERE table_id = {}",
                a.table_id("t").await
            ))
            .await;
        assert_eq!(a_storage.len(), 1);
        let Kind::Multicatalog {
            manager,
            ..
        } = &a.kind
        else {
            unreachable!()
        };
        assert!(manager.drop_catalog("a").await.unwrap());

        let physical = b.physical_inlined_tables().await;
        assert!(!physical.contains(&a_storage[0]), "{physical:?}");
        assert!(!physical.contains(&a_delete), "{physical:?}");
        let mut expected = b_live.clone();
        expected.push(b_superseded.clone());
        expected.sort();
        assert_eq!(physical, expected);
        assert_eq!(b.registry().await, expected);
        assert_eq!(b.rows("t", false, None).await, strings(&["1", "3"]));
    }

    /// A store bootstrapped before staged commits has no
    /// `ducklake_inlined_data_tables`; expire and `drop_catalog` reclaim no inlined
    /// storage there instead of failing.
    #[tokio::test(flavor = "multi_thread")]
    #[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
    async fn multicatalog_reclaim_without_the_registry() {
        let (pool, _url, _container) = start().await;
        initialize_multicatalog_schema(&pool).await.unwrap();
        let lake = catalog_lake(&pool, "lake").await;
        let parquet = lake.table_writer(0);
        parquet
            .write_table("main", "t", &[batch_i(&[1, 2])])
            .await
            .unwrap();
        parquet
            .write_table("main", "keep", &[batch_i(&[7])])
            .await
            .unwrap();
        lake.drop_table("t").await;
        lake.execute("DROP TABLE ducklake_inlined_data_tables")
            .await;

        assert!(lake.expire_all().await > 0);
        assert_eq!(lake.rows("keep", false, None).await, strings(&["7"]));
        let Kind::Multicatalog {
            manager,
            ..
        } = &lake.kind
        else {
            unreachable!()
        };
        assert!(manager.drop_catalog("lake").await.unwrap());
    }

    /// The single-catalog layout has no expire; its flush drops the superseded tables
    /// it empties as official's does.
    #[tokio::test(flavor = "multi_thread")]
    #[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
    async fn postgres_single_flush_drops_empty_superseded_tables() {
        let (pool, url, _container) = start().await;
        let writer = Arc::new(
            datafusion_ducklake::PostgresSingleCatalogMetadataWriter::new_with_init(&url)
                .await
                .unwrap(),
        );
        let data = Arc::new(TempDir::new().unwrap());
        writer.set_data_path(data.path().to_str().unwrap()).unwrap();
        let lake = Lake {
            writer,
            kind: Kind::PostgresSingle {
                url,
            },
            raw: Raw::Postgres(pool),
            _data: data,
        };
        // The tables are an official writer's: a superseded empty one, a superseded
        // one with a live row, and the current (empty) one.
        let created = lake
            .table_writer(0)
            .write_table("main", "a", &[batch_i(&[1])])
            .await
            .unwrap();
        let table_id = created.table_id;
        let snapshot = created.snapshot_id;
        let name = |version: i64| format!("ducklake_inlined_data_{table_id}_{version}");
        for version in [0, 5, 7] {
            lake.execute(&format!(
                "CREATE TABLE {}(row_id BIGINT, begin_snapshot BIGINT, end_snapshot BIGINT, i INTEGER)",
                name(version)
            ))
            .await;
            lake.execute(&format!(
                "INSERT INTO ducklake_inlined_data_tables VALUES ({table_id}, '{}', {version})",
                name(version)
            ))
            .await;
        }
        lake.execute(&format!(
            "INSERT INTO {} VALUES (1, {snapshot}, NULL, 9)",
            name(5)
        ))
        .await;
        assert_eq!(lake.rows("a", false, None).await, strings(&["1", "9"]));

        // The flush moves the live row to Parquet and empties its table; both
        // superseded tables are then empty and go, the current one stays.
        let flushed = lake.flush("a").await.unwrap();
        assert_eq!(flushed.records_written, 1);
        let kept = vec![name(7)];
        assert_eq!(lake.registry().await, kept);
        assert_eq!(lake.physical_inlined_tables().await, kept);
        assert_eq!(lake.rows("a", false, None).await, strings(&["1", "9"]));
    }
}

#[cfg(all(feature = "write-duckdb", feature = "metadata-duckdb"))]
mod duckdb_catalog {
    use super::*;
    use datafusion_ducklake::{DuckdbMetadataProvider, DuckdbMetadataWriter};
    use std::path::Path;

    fn official(path: &Path, data: &Path) -> duckdb::Connection {
        crate::common::ensure_ducklake_installed();
        let connection = duckdb::Connection::open_in_memory().unwrap();
        connection
            .execute_batch(&format!(
                "LOAD ducklake;
                 ATTACH 'ducklake:{}' AS lake (DATA_PATH '{}', METADATA_CATALOG 'meta',
                     DATA_INLINING_ROW_LIMIT 10);",
                path.display(),
                data.display()
            ))
            .unwrap();
        connection
    }

    fn strings_of(connection: &duckdb::Connection, sql: &str) -> Vec<String> {
        let mut statement = connection.prepare(sql).unwrap();
        let mut values = statement
            .query_map([], |row| row.get::<_, String>(0))
            .unwrap()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap();
        values.sort();
        values
    }

    const PHYSICAL: &str = "SELECT table_name FROM duckdb_tables()
        WHERE database_name = 'meta' AND table_name LIKE 'ducklake_inlined_%'
          AND table_name <> 'ducklake_inlined_data_tables'";

    /// Every scenario on a catalog official DuckDB wrote: its inlined tables per schema
    /// version (created at ALTER time), its inlined delete table, a dropped table and an
    /// inline-deleted row. This crate expires it and drops its empty superseded tables,
    /// and official reads and writes it afterwards. The inline-deleted row stays, as it
    /// does under official 1.5.5's expire.
    #[tokio::test(flavor = "multi_thread")]
    async fn duckdb_official_catalog_inline_storage_is_reclaimed() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("meta.ducklake");
        let data = temp.path().join("data");
        {
            let connection = official(&path, &data);
            connection
                .execute_batch(
                    "CREATE TABLE lake.t AS SELECT range::INTEGER AS i FROM range(100);
                     DELETE FROM lake.t WHERE i IN (3, 5, 7);
                     INSERT INTO lake.t VALUES (200), (201);
                     ALTER TABLE lake.t ADD COLUMN j INTEGER;
                     INSERT INTO lake.t VALUES (202, 1);
                     DROP TABLE lake.t;
                     CREATE TABLE lake.t1(i INTEGER);
                     INSERT INTO lake.t1 VALUES (1), (2), (3);
                     DELETE FROM lake.t1 WHERE i = 2;
                     ALTER TABLE lake.t1 ADD COLUMN j INTEGER;
                     ALTER TABLE lake.t1 ADD COLUMN k INTEGER;",
                )
                .unwrap();
            let physical = strings_of(&connection, PHYSICAL);
            assert!(
                physical.contains(&"ducklake_inlined_delete_1".to_string()),
                "{physical:?}"
            );
            assert!(physical.len() >= 5, "{physical:?}");
        }

        let writer = Arc::new(
            DuckdbMetadataWriter::new_with_init(path.to_string_lossy().into_owned()).unwrap(),
        );
        assert!(
            !writer
                .expire_snapshots(ExpireCriteria::OlderThan(Utc::now() + Duration::minutes(1)))
                .unwrap()
                .is_empty()
        );
        // The dead table's storage is gone; t1's tables remain.
        let provider = Arc::new(DuckdbMetadataProvider::new(path.to_string_lossy()).unwrap());
        let snapshot = provider.get_current_snapshot().unwrap();
        let schema = provider
            .get_schema_by_name("main", snapshot)
            .unwrap()
            .unwrap();
        let t1 = provider
            .get_table_by_name(schema.schema_id, "t1", snapshot)
            .unwrap()
            .unwrap();
        let columns = provider.get_table_structure(t1.table_id, snapshot).unwrap();
        let inlined = provider
            .get_inlined_data_with_row_ids(t1.table_id, snapshot, &columns)
            .unwrap();
        assert_eq!(
            inlined
                .iter()
                .map(|data| data.batch.num_rows())
                .sum::<usize>(),
            2
        );
        // The step official runs after a flush commits: the empty superseded ALTER-time
        // table goes; the first version's table holds the live rows and stays. (This
        // crate's DuckDB writer does not commit to a catalog official created, so its
        // flush is not exercised here.)
        writer.drop_empty_superseded_inlined_tables().unwrap();
        drop(provider);
        drop(writer);

        let connection = official(&path, &data);
        let physical = strings_of(&connection, PHYSICAL);
        let registry = strings_of(
            &connection,
            "SELECT table_name FROM meta.ducklake_inlined_data_tables",
        );
        assert_eq!(physical, registry);
        assert_eq!(
            registry,
            vec![
                format!("ducklake_inlined_data_{}_4", t1.table_id),
                format!("ducklake_inlined_data_{}_6", t1.table_id),
            ],
            "the first version keeps its live rows; the newest is current",
        );
        connection
            .execute_batch("INSERT INTO lake.t1 VALUES (6, 60, 600)")
            .unwrap();
        assert_eq!(
            strings_of(
                &connection,
                "SELECT concat_ws('|', i, coalesce(j::VARCHAR, 'NULL'), coalesce(k::VARCHAR, 'NULL'))
                 FROM lake.t1"
            ),
            strings(&["1|NULL|NULL", "3|NULL|NULL", "6|60|600"])
        );
    }
    /// A catalog this crate's DuckDB writer created before data inlining has no
    /// `ducklake_inlined_data_tables`; expire and the superseded-table drop reclaim
    /// nothing there instead of failing.
    #[tokio::test(flavor = "multi_thread")]
    async fn duckdb_reclaim_without_the_registry() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("meta.ducklake");
        let data = temp.path().join("data");
        std::fs::create_dir_all(&data).unwrap();
        {
            let writer = Arc::new(
                DuckdbMetadataWriter::new_with_init(path.to_string_lossy().into_owned()).unwrap(),
            );
            writer.set_data_path(data.to_str().unwrap()).unwrap();
            let parquet =
                DuckLakeTableWriter::new(writer.clone(), Arc::new(LocalFileSystem::new())).unwrap();
            parquet
                .write_table("main", "t", &[batch_i(&[1, 2])])
                .await
                .unwrap();
            parquet
                .write_table("main", "keep", &[batch_i(&[7])])
                .await
                .unwrap();
        }
        {
            // Drop `t` at a new snapshot, then remove the registry, leaving the
            // layout a pre-inlining writer created.
            let connection = duckdb::Connection::open(&path).unwrap();
            connection
                .execute_batch(
                    "INSERT INTO ducklake_snapshot (snapshot_id, schema_version)
                     SELECT MAX(snapshot_id) + 1, MAX(schema_version) + 1 FROM ducklake_snapshot;
                     UPDATE ducklake_table SET end_snapshot = (SELECT MAX(snapshot_id) FROM ducklake_snapshot)
                     WHERE table_name = 't' AND end_snapshot IS NULL;
                     UPDATE ducklake_data_file SET end_snapshot = (SELECT MAX(snapshot_id) FROM ducklake_snapshot)
                     WHERE table_id = (SELECT table_id FROM ducklake_table WHERE table_name = 't')
                       AND end_snapshot IS NULL;
                     DROP TABLE ducklake_inlined_data_tables;",
                )
                .unwrap();
        }
        let writer =
            Arc::new(DuckdbMetadataWriter::new(path.to_string_lossy().into_owned()).unwrap());
        assert!(
            !writer
                .expire_snapshots(ExpireCriteria::OlderThan(Utc::now() + Duration::minutes(1)))
                .unwrap()
                .is_empty()
        );
        writer.drop_empty_superseded_inlined_tables().unwrap();
        let provider: Arc<dyn MetadataProvider> = Arc::new(writer.metadata_provider());
        let snapshot = provider.get_current_snapshot().unwrap();
        let schema = provider
            .get_schema_by_name("main", snapshot)
            .unwrap()
            .unwrap();
        assert!(
            provider
                .get_table_by_name(schema.schema_id, "t", snapshot)
                .unwrap()
                .is_none()
        );
        let catalog = DuckLakeCatalog::with_writer(provider, writer.clone()).unwrap();
        let ctx = SessionContext::new();
        ctx.register_catalog("lake", Arc::new(catalog));
        let batches = ctx
            .sql("SELECT i FROM lake.main.keep")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 1);
    }
}
