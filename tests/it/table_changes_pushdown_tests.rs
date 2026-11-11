//! Filters pushed into the change feeds, and the feeds' inlined rows read in
//! pages.
//!
//! A filter over the table's columns narrows what `ducklake_table_insertions`
//! (and the reported rows of `ducklake_table_changes`) read: data files by
//! their catalog statistics, row groups by the Parquet reader, inlined rows in
//! the catalog query. None of it may change an answer: every feed, filtered,
//! must return exactly its unfiltered rows filtered. That is checked here over
//! seeded random histories (inlined and file inserts, updates that move the
//! time column, deletes, flushes, merges of adjacent files and rewrites, and
//! schema changes: a widened, an added, a dropped and re-added and, on SQLite,
//! a renamed column; with NaNs and strings longer than a stored bound), for
//! every window, on SQLite and on PostgreSQL (the multi-catalog layout), with
//! the filtered side reading its inlined rows a few rows per page and the
//! Parquet reader filtering rows itself.
//!
//! The rest pins what the pushdown is for: pruned files are not read, the
//! feed nodes report the bytes their hidden scans read, and inlined rows past
//! one page, across physical inlined tables, are streamed through a catalog
//! read pinned to the plan's state (or read at planning where no read can be
//! pinned, as on a server that cannot export a snapshot), so a flush that
//! commits between planning and execution changes no answer, and the pinned
//! read's connection goes back to the pool however the read ends.

#![cfg(all(feature = "write-sqlite", feature = "metadata-sqlite"))]

use std::sync::Arc;

use arrow::record_batch::RecordBatch;
use arrow::util::display::array_value_to_string;
use datafusion::datasource::MemTable;
use datafusion::datasource::physical_plan::{FileScanConfig, FileSource, ParquetSource};
use datafusion::datasource::source::DataSourceExec;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_ducklake::table_changes::InsertedRowsExec;
use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeTable, DuckLakeWriteOptions, MergeOptions, MetadataProvider,
    MetadataWriter, RewriteOptions, execute_ducklake_sql, register_ducklake_functions,
};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

#[derive(Clone)]
struct Lake {
    writer: Arc<dyn MetadataWriter>,
    provider: Arc<dyn MetadataProvider>,
    /// The provider's connection pool.
    pool: ProviderPool,
    /// Inserts of at most this many rows are inlined.
    limit: usize,
}

#[derive(Clone)]
enum ProviderPool {
    Sqlite(sqlx::SqlitePool),
    #[cfg(feature = "write-postgres")]
    Postgres(sqlx::PgPool),
}

impl ProviderPool {
    /// The connections taken from the pool and not given back.
    fn in_use(&self) -> usize {
        match self {
            Self::Sqlite(pool) => pool.size() as usize - pool.num_idle(),
            #[cfg(feature = "write-postgres")]
            Self::Postgres(pool) => pool.size() as usize - pool.num_idle(),
        }
    }

    /// Take every connection the pool may hold but one, and hold them until
    /// the returned value drops: a plan's consistent read takes the last, and
    /// no connection is left to pin a read on.
    async fn exhaust(&self) -> Box<dyn std::any::Any + Send> {
        match self {
            Self::Sqlite(pool) => {
                let mut held = Vec::new();
                for _ in 1..pool.options().get_max_connections() {
                    held.push(pool.acquire().await.unwrap());
                }
                Box::new(held)
            },
            #[cfg(feature = "write-postgres")]
            Self::Postgres(pool) => {
                let mut held = Vec::new();
                for _ in 1..pool.options().get_max_connections() {
                    held.push(pool.acquire().await.unwrap());
                }
                Box::new(held)
            },
        }
    }

    /// Wait until `in_use` connections are taken, failing with `what`.
    async fn settles_at(&self, in_use: usize, what: &str) {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        while self.in_use() != in_use {
            assert!(
                std::time::Instant::now() < deadline,
                "{what}: {} connections in use, expected {in_use}",
                self.in_use()
            );
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
    }
}

impl Lake {
    /// A session over the lake: rows per page of inlined rows (the session's
    /// batch size) and whether the Parquet reader filters rows itself.
    fn session_with(
        &self,
        batch_size: usize,
        row_filter: bool,
    ) -> (SessionContext, DuckLakeCatalog) {
        let mut config = SessionConfig::new()
            .with_target_partitions(4)
            .with_batch_size(batch_size);
        config.options_mut().execution.parquet.pushdown_filters = row_filter;
        let ctx = SessionContext::new_with_config(config);
        let make = || {
            DuckLakeCatalog::with_writer(Arc::clone(&self.provider), Arc::clone(&self.writer))
                .unwrap()
                .with_write_options(
                    DuckLakeWriteOptions::default().with_data_inlining_row_limit(self.limit),
                )
        };
        DuckLakeCatalog::register(&ctx, "lake", make());
        register_ducklake_functions(&ctx, Arc::clone(&self.provider));
        (ctx, make())
    }

    fn session(&self) -> (SessionContext, DuckLakeCatalog) {
        self.session_with(8192, false)
    }

    async fn sql(&self, statement: &str) {
        let (ctx, catalog) = self.session();
        execute_ducklake_sql(&ctx, &catalog, statement)
            .await
            .unwrap_or_else(|e| panic!("{statement}: {e}"))
            .collect()
            .await
            .unwrap_or_else(|e| panic!("{statement}: {e}"));
    }

    async fn table(&self) -> (SessionContext, DuckLakeTable) {
        let (ctx, _) = self.session();
        let provider = ctx
            .catalog("lake")
            .unwrap()
            .schema("main")
            .unwrap()
            .table("t")
            .await
            .unwrap()
            .unwrap();
        let table = (provider.as_ref() as &dyn std::any::Any)
            .downcast_ref::<DuckLakeTable>()
            .expect("a DuckLakeTable")
            .clone();
        (ctx, table)
    }

    fn head(&self) -> i64 {
        self.provider.get_current_snapshot().unwrap()
    }
}

/// A SQLite catalog in WAL mode, where a feed pins its read of the pages.
async fn sqlite_lake(dir: &tempfile::TempDir, limit: usize) -> Lake {
    sqlite_lake_in(dir, limit, true).await
}

async fn sqlite_lake_in(dir: &tempfile::TempDir, limit: usize, wal: bool) -> Lake {
    use datafusion_ducklake::{SqliteMetadataProvider, SqliteMetadataWriter};
    let data = dir.path().join("data");
    std::fs::create_dir_all(&data).unwrap();
    let url = format!("sqlite:{}?mode=rwc", dir.path().join("c.db").display());
    let writer = SqliteMetadataWriter::new_with_init(&url).await.unwrap();
    writer.set_data_path(data.to_str().unwrap()).unwrap();
    let snapshot = writer.create_snapshot().unwrap();
    writer.get_or_create_schema("main", None, snapshot).unwrap();
    // WAL mode is a property of the database file, which every connection
    // opened after this uses.
    let mode = if wal {
        "wal"
    } else {
        "delete"
    };
    let pool = sqlx::SqlitePool::connect(&url).await.unwrap();
    let set: String =
        sqlx::query_scalar(sqlx::AssertSqlSafe(format!("PRAGMA journal_mode={mode}")))
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(set, mode);
    pool.close().await;
    let provider = SqliteMetadataProvider::new(&url).await.unwrap();
    Lake {
        writer: Arc::new(writer),
        pool: ProviderPool::Sqlite(provider.pool.clone()),
        provider: Arc::new(provider),
        limit,
    }
}

#[cfg(feature = "write-postgres")]
async fn postgres_lake(port: u16, database: &str, dir: &tempfile::TempDir, limit: usize) -> Lake {
    postgres_lake_with(port, database, dir, limit, &[]).await
}

/// [`postgres_lake`], running `setup` in the new database before anything
/// connects to it.
#[cfg(feature = "write-postgres")]
async fn postgres_lake_with(
    port: u16,
    database: &str,
    dir: &tempfile::TempDir,
    limit: usize,
    setup: &[&str],
) -> Lake {
    use datafusion_ducklake::{MulticatalogManager, MulticatalogProvider, PostgresMetadataWriter};
    let admin = sqlx::PgPool::connect(&format!(
        "postgresql://postgres:postgres@127.0.0.1:{port}/postgres"
    ))
    .await
    .unwrap();
    sqlx::query(sqlx::AssertSqlSafe(format!("CREATE DATABASE {database}")))
        .execute(&admin)
        .await
        .unwrap();
    admin.close().await;
    let url = format!("postgresql://postgres:postgres@127.0.0.1:{port}/{database}");
    if !setup.is_empty() {
        let pool = sqlx::PgPool::connect(&url).await.unwrap();
        for statement in setup {
            sqlx::raw_sql(sqlx::AssertSqlSafe(*statement))
                .execute(&pool)
                .await
                .unwrap();
        }
        pool.close().await;
    }
    let data = dir.path().join("data");
    std::fs::create_dir_all(&data).unwrap();
    let pool = sqlx::PgPool::connect(&url).await.unwrap();
    datafusion_ducklake::initialize_multicatalog_schema(&pool)
        .await
        .unwrap();
    let catalog = MulticatalogManager::new(pool.clone())
        .create_catalog("lake")
        .await
        .unwrap();
    let writer = PostgresMetadataWriter::with_pool(pool.clone(), catalog)
        .await
        .unwrap();
    writer.set_data_path(data.to_str().unwrap()).unwrap();
    let snapshot = writer.create_snapshot().unwrap();
    writer.get_or_create_schema("main", None, snapshot).unwrap();
    Lake {
        writer: Arc::new(writer),
        provider: Arc::new(
            MulticatalogProvider::with_pool(pool.clone(), "lake")
                .await
                .unwrap(),
        ),
        pool: ProviderPool::Postgres(pool),
        limit,
    }
}

/// A single-catalog PostgreSQL lake in the database `database`. Its writer
/// inlines no row.
#[cfg(feature = "write-postgres")]
async fn postgres_single_lake(port: u16, database: &str, dir: &tempfile::TempDir) -> Lake {
    use datafusion_ducklake::{PostgresMetadataProvider, PostgresSingleCatalogMetadataWriter};
    let admin = sqlx::PgPool::connect(&format!(
        "postgresql://postgres:postgres@127.0.0.1:{port}/postgres"
    ))
    .await
    .unwrap();
    sqlx::query(sqlx::AssertSqlSafe(format!("CREATE DATABASE {database}")))
        .execute(&admin)
        .await
        .unwrap();
    admin.close().await;
    let data = dir.path().join("data");
    std::fs::create_dir_all(&data).unwrap();
    let url = format!("postgresql://postgres:postgres@127.0.0.1:{port}/{database}");
    let writer = PostgresSingleCatalogMetadataWriter::new_with_init(&url)
        .await
        .unwrap();
    writer.set_data_path(data.to_str().unwrap()).unwrap();
    let snapshot = writer.create_snapshot().unwrap();
    writer.get_or_create_schema("main", None, snapshot).unwrap();
    let pool = sqlx::PgPool::connect(&url).await.unwrap();
    Lake {
        writer: Arc::new(writer),
        provider: Arc::new(PostgresMetadataProvider::from_pool(pool.clone())),
        pool: ProviderPool::Postgres(pool),
        limit: 0,
    }
}

#[cfg(feature = "write-postgres")]
async fn postgres() -> (
    testcontainers::ContainerAsync<testcontainers_modules::postgres::Postgres>,
    u16,
) {
    use testcontainers::runners::AsyncRunner;
    let container = testcontainers_modules::postgres::Postgres::default()
        .start()
        .await
        .unwrap();
    let port = container.get_host_port_ipv4(5432).await.unwrap();
    (container, port)
}

fn canonical(batches: &[RecordBatch]) -> Vec<String> {
    let mut out = Vec::new();
    for batch in batches {
        for row in 0..batch.num_rows() {
            let cells: Vec<String> = batch
                .columns()
                .iter()
                .map(|c| array_value_to_string(c, row).unwrap())
                .collect();
            out.push(cells.join("|"));
        }
    }
    out.sort();
    out
}

async fn rows(ctx: &SessionContext, sql: &str) -> Result<Vec<String>, String> {
    let frame = ctx.sql(sql).await.map_err(|e| e.to_string())?;
    let batches = frame.collect().await.map_err(|e| e.to_string())?;
    Ok(canonical(&batches))
}

/// A string longer than the 256 bytes a catalog or Parquet string bound
/// keeps, so the stored bounds of a column holding it are truncated.
fn long_string(tag: u64) -> String {
    format!("{}{tag}", "x".repeat(300))
}

/// The columns of `t` now, as the lake's table reports them.
async fn table_schema(lake: &Lake) -> arrow::datatypes::SchemaRef {
    let (_, table) = lake.table().await;
    datafusion::catalog::TableProvider::schema(&table)
}

/// Append `n` rows to `t` as a batch of `schema`, whose columns may differ
/// from the table's: a column the table lacks is added, one the batch lacks
/// is dropped, as a schema-changing append does. The ids continue from
/// `next_id`; inlined when `n` is within the lake's limit.
async fn append_as(
    lake: &Lake,
    schema: arrow::datatypes::SchemaRef,
    n: usize,
    rng: &mut Rng,
    next_id: &mut i64,
) {
    use arrow::array::{ArrayRef, Float64Array, Int32Array, Int64Array, StringArray};
    use arrow::datatypes::DataType;
    let ids: Vec<i64> = (0..n)
        .map(|_| {
            *next_id += 1;
            *next_id
        })
        .collect();
    let columns: Vec<ArrayRef> = schema
        .fields()
        .iter()
        .map(|field| -> ArrayRef {
            match (field.name().as_str(), field.data_type()) {
                ("id", _) => Arc::new(Int64Array::from(ids.clone())),
                ("ts", _) => Arc::new(Int64Array::from(
                    ids.iter().map(|id| id / 10).collect::<Vec<_>>(),
                )),
                (_, DataType::Int32) => Arc::new(Int32Array::from(
                    ids.iter()
                        .map(|_| (rng.below(4) != 0).then(|| rng.below(100) as i32))
                        .collect::<Vec<_>>(),
                )),
                (_, DataType::Int64) => Arc::new(Int64Array::from(
                    ids.iter()
                        .map(|_| (rng.below(4) != 0).then(|| rng.below(100) as i64))
                        .collect::<Vec<_>>(),
                )),
                (_, DataType::Float64) => Arc::new(Float64Array::from(
                    ids.iter()
                        .map(|_| match rng.below(5) {
                            0 => None,
                            1 => Some(f64::NAN),
                            _ => Some(rng.below(100) as f64 / 100.0),
                        })
                        .collect::<Vec<_>>(),
                )),
                (_, DataType::Utf8 | DataType::Utf8View) => {
                    let values = StringArray::from(
                        ids.iter()
                            .map(|_| match rng.below(5) {
                                0 => None,
                                1 => Some(long_string(rng.below(7))),
                                _ => Some(format!("s{}", rng.below(7))),
                            })
                            .collect::<Vec<_>>(),
                    );
                    arrow::compute::cast(&values, field.data_type()).unwrap()
                },
                (name, other) => panic!("no values for {name}: {other}"),
            }
        })
        .collect();
    let batch = RecordBatch::try_new(Arc::clone(&schema), columns).unwrap();
    datafusion_ducklake::DuckLakeTableWriter::new(
        Arc::clone(&lake.writer),
        Arc::new(object_store::local::LocalFileSystem::new()),
    )
    .unwrap()
    .with_options(&DuckLakeWriteOptions::default().with_data_inlining_row_limit(lake.limit))
    .append_table("main", "t", &[batch])
    .await
    .unwrap();
}

/// `t`'s columns now without `name`, or with `name` of `data_type` added
/// last.
async fn schema_without(lake: &Lake, name: &str) -> arrow::datatypes::SchemaRef {
    let schema = table_schema(lake).await;
    Arc::new(arrow::datatypes::Schema::new(
        schema
            .fields()
            .iter()
            .filter(|field| field.name() != name)
            .cloned()
            .collect::<Vec<_>>(),
    ))
}

async fn schema_with(
    lake: &Lake,
    name: &str,
    data_type: arrow::datatypes::DataType,
) -> arrow::datatypes::SchemaRef {
    let schema = table_schema(lake).await;
    let mut fields: Vec<_> = schema.fields().iter().cloned().collect();
    fields.push(Arc::new(arrow::datatypes::Field::new(
        name, data_type, true,
    )));
    Arc::new(arrow::datatypes::Schema::new(fields))
}

/// Rename column `from` of `t` to `to` in a SQLite catalog, as a rename
/// commits there: a snapshot of a new schema version that ends the column's
/// row and begins one of the same id under the new name. The writers have no
/// rename, so the test writes it.
async fn rename_sqlite_column(pool: &sqlx::SqlitePool, from: &str, to: &str) {
    let mut tx = pool.begin().await.unwrap();
    let table_id: i64 = sqlx::query_scalar(
        "SELECT table_id FROM ducklake_table WHERE table_name = 't' AND end_snapshot IS NULL",
    )
    .fetch_one(&mut *tx)
    .await
    .unwrap();
    let (snapshot, schema_version): (i64, i64) = sqlx::query_as(
        "INSERT INTO ducklake_snapshot
             (snapshot_id, snapshot_time, schema_version, next_catalog_id, next_file_id)
         SELECT snapshot_id + 1, CURRENT_TIMESTAMP,
                (SELECT MAX(schema_version) + 1 FROM ducklake_snapshot),
                next_catalog_id, next_file_id
         FROM ducklake_snapshot ORDER BY snapshot_id DESC LIMIT 1
         RETURNING snapshot_id, schema_version",
    )
    .fetch_one(&mut *tx)
    .await
    .unwrap();
    sqlx::query("INSERT INTO ducklake_snapshot_changes (snapshot_id, changes_made) VALUES (?, ?)")
        .bind(snapshot)
        .bind(format!("altered_table:{table_id}"))
        .execute(&mut *tx)
        .await
        .unwrap();
    sqlx::query(
        "INSERT INTO ducklake_schema_versions (begin_snapshot, schema_version, table_id)
         VALUES (?, ?, ?)",
    )
    .bind(snapshot)
    .bind(schema_version)
    .bind(table_id)
    .execute(&mut *tx)
    .await
    .unwrap();
    let column_id: i64 = sqlx::query_scalar(
        "SELECT column_id FROM ducklake_column
         WHERE table_id = ? AND column_name = ? AND end_snapshot IS NULL",
    )
    .bind(table_id)
    .bind(from)
    .fetch_one(&mut *tx)
    .await
    .unwrap();
    sqlx::query(
        "UPDATE ducklake_column SET end_snapshot = ?
         WHERE table_id = ? AND column_id = ? AND end_snapshot IS NULL",
    )
    .bind(snapshot)
    .bind(table_id)
    .bind(column_id)
    .execute(&mut *tx)
    .await
    .unwrap();
    sqlx::query(
        "INSERT INTO ducklake_column
             (column_id, begin_snapshot, end_snapshot, table_id, column_order, column_name,
              column_type, nulls_allowed, parent_column, initial_default, default_value,
              default_value_type, default_value_dialect)
         SELECT column_id, ?, NULL, table_id, column_order, ?, column_type, nulls_allowed,
                parent_column, initial_default, default_value, default_value_type,
                default_value_dialect
         FROM ducklake_column WHERE table_id = ? AND column_id = ? AND end_snapshot = ?",
    )
    .bind(snapshot)
    .bind(to)
    .bind(table_id)
    .bind(column_id)
    .bind(snapshot)
    .execute(&mut *tx)
    .await
    .unwrap();
    tx.commit().await.unwrap();
}

/// The id of `t`.
fn table_id(lake: &Lake) -> i64 {
    let head = lake.head();
    let schema = lake
        .provider
        .get_schema_by_name("main", head)
        .unwrap()
        .unwrap();
    lake.provider
        .get_table_by_name(schema.schema_id, "t", head)
        .unwrap()
        .unwrap()
        .table_id
}

/// Insert `n` rows into `t` with SQL, naming the columns, so the statement
/// holds whatever columns were added since: `(id, ts, <val>, s, f)`, where
/// `val` is the current name of that column. `ts` grows with the ids; `val`
/// and `s` hold NULLs, `s` some strings longer than a stored bound, and `f`
/// NULLs and NaNs.
async fn insert_rows(lake: &Lake, val: &str, n: u64, rng: &mut Rng, next_id: &mut i64) {
    let values: Vec<String> = (0..n)
        .map(|_| {
            *next_id += 1;
            let v = match rng.below(5) {
                0 => "NULL".to_string(),
                _ => rng.below(100).to_string(),
            };
            let s = match rng.below(5) {
                0 => "NULL".to_string(),
                1 => format!("'{}'", long_string(rng.below(7))),
                _ => format!("'s{}'", rng.below(7)),
            };
            let f = match rng.below(5) {
                0 => "NULL".to_string(),
                1 => "CAST('NaN' AS DOUBLE)".to_string(),
                _ => format!("{}.{:02}", rng.below(2), rng.below(100)),
            };
            format!("({next_id}, {}, {v}, {s}, {f})", *next_id / 10)
        })
        .collect();
    lake.sql(&format!(
        "INSERT INTO lake.main.t (id, ts, {val}, s, f) VALUES {}",
        values.join(", ")
    ))
    .await;
}

/// The schema changes [`history`] makes, one each, at steps the seed picks.
#[derive(Clone, Copy, Debug)]
enum Alter {
    /// `val` widened from `INTEGER` to `BIGINT`.
    Widen,
    /// A column `extra` added, by an append that holds it.
    AddColumn,
    /// `s` dropped and added again: the same name under a new column id.
    DropAndReadd,
    /// `val` renamed, rows inserted, and renamed back: files whose column
    /// is named apart from the catalog's name. SQLite only (the test writes
    /// the rename); elsewhere a second drop and re-add of `s`.
    RenameAndBack,
}

/// A random history of `t(id, ts, val, s, f)`: `ts` grows with the inserts and
/// is what an incremental reader would filter on; `val`, `s` and `f` hold
/// NULLs, `s` strings longer than a stored bound, `f` NaNs. Besides inserts,
/// updates, deletes, flushes, merges and rewrites, it changes the schema once
/// in each way [`Alter`] lists, so files and inlined tables of several schema
/// versions meet in one window.
async fn history(lake: &Lake, seed: u64, steps: usize) -> i64 {
    let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    lake.sql("CREATE TABLE lake.main.t (id BIGINT, ts BIGINT, val INTEGER, s VARCHAR, f DOUBLE)")
        .await;
    let created = lake.head();
    let alters = [Alter::Widen, Alter::AddColumn, Alter::DropAndReadd, Alter::RenameAndBack];
    let at = |k: usize| (k * steps / alters.len() + 1 + seed as usize) % steps;
    let mut next_id = 0i64;
    for step in 0..steps {
        if let Some(alter) = (0..alters.len())
            .find(|&k| at(k) == step)
            .map(|k| alters[k])
        {
            match (alter, &lake.pool) {
                (Alter::Widen, _) => {
                    lake.writer
                        .promote_column_type(table_id(lake), "val", "int64")
                        .unwrap();
                },
                (Alter::AddColumn, _) => {
                    let n = 1 + rng.below(2 * lake.limit as u64 + 2) as usize;
                    let schema =
                        schema_with(lake, "extra", arrow::datatypes::DataType::Int64).await;
                    append_as(lake, schema, n, &mut rng, &mut next_id).await;
                },
                (Alter::RenameAndBack, ProviderPool::Sqlite(pool)) => {
                    rename_sqlite_column(pool, "val", "val_was").await;
                    // Past the inlining limit: the latest inlined table still
                    // names the column `val`.
                    insert_rows(lake, "val_was", 20 + rng.below(200), &mut rng, &mut next_id).await;
                    rename_sqlite_column(pool, "val_was", "val").await;
                },
                _ => {
                    let n = 1 + rng.below(2 * lake.limit as u64 + 2) as usize;
                    let schema = schema_without(lake, "s").await;
                    append_as(lake, schema, n, &mut rng, &mut next_id).await;
                    let n = 1 + rng.below(2 * lake.limit as u64 + 2) as usize;
                    let schema = schema_with(lake, "s", arrow::datatypes::DataType::Utf8).await;
                    append_as(lake, schema, n, &mut rng, &mut next_id).await;
                },
            }
            continue;
        }
        match rng.below(13) {
            0..=4 => {
                let n = if rng.below(2) == 0 {
                    1 + rng.below(4)
                } else {
                    20 + rng.below(1500)
                };
                insert_rows(lake, "val", n, &mut rng, &mut next_id).await;
            },
            // An update moves rows' timestamps, into or out of a filtered range.
            5 | 6 => {
                let (from, width) = (rng.below(next_id as u64 + 1), 1 + rng.below(300));
                lake.sql(&format!(
                    "UPDATE lake.main.t SET ts = ts + 17, val = val + 1 \
                     WHERE id >= {from} AND id < {}",
                    from + width
                ))
                .await;
            },
            7 => {
                let (from, width) = (rng.below(next_id as u64 + 1), 1 + rng.below(6));
                lake.sql(&format!(
                    "DELETE FROM lake.main.t WHERE id >= {from} AND id < {}",
                    from + width
                ))
                .await;
            },
            8 => lake.sql("CALL ducklake_flush_inlined_data('lake')").await,
            9 | 10 => {
                let (ctx, table) = lake.table().await;
                table
                    .merge_adjacent_files(
                        &ctx.state(),
                        MergeOptions {
                            target_file_size: 1 << 30,
                            max_merged_files: usize::MAX,
                            min_file_size: 0,
                            ..MergeOptions::default()
                        },
                    )
                    .await
                    .unwrap();
            },
            _ => {
                let (ctx, table) = lake.table().await;
                table
                    .rewrite_data_files(
                        &ctx.state(),
                        RewriteOptions {
                            delete_threshold: 0.0,
                            data_file_ids: None,
                        },
                    )
                    .await
                    .unwrap();
            },
        }
    }
    created
}

/// Every feed of every window, filtered, against its unfiltered rows
/// filtered after the fact. The filtered side pages inlined rows three at a
/// time and lets the Parquet reader filter rows. A filter naming a column the
/// window's table lacks (a feed reads the table as of the window's end) is
/// skipped for that window.
async fn compare_filtered(lake: &Lake, seed: u64, created: i64) -> usize {
    let head = lake.head();
    let (oracle, _) = lake.session();
    let (paged, _) = lake.session_with(3, true);
    let mut compared = 0;
    for a in 0..=head {
        for b in a.max(created)..=head {
            let mid = (a + b) / 2;
            let common = vec![
                "ts >= 30".to_string(),
                "ts < 20 AND val > 10".to_string(),
                "s IS NULL".to_string(),
                "s = 's3'".to_string(),
                "val IS NOT NULL AND val < 50".to_string(),
                "id IN (3, 5, 77, 200, 1001)".to_string(),
                "ts BETWEEN 10 AND 40 OR s IS NULL".to_string(),
                format!("snapshot_id >= {mid} AND ts > 5"),
                "rowid > 100 AND val >= 0".to_string(),
                // NaN compares above every number.
                "f > 0.5".to_string(),
                "f <= 0.25 OR f IS NULL".to_string(),
                // Bounds of strings longer than the stored prefix.
                format!("s = '{}'", long_string(3)),
                "s > 's6'".to_string(),
                "extra IS NULL".to_string(),
                "extra > 50".to_string(),
            ];
            for (feed, extra) in [
                ("ducklake_table_insertions", None),
                (
                    "ducklake_table_deletions",
                    Some("change_type = 'delete' AND ts > 3"),
                ),
                (
                    "ducklake_table_changes",
                    Some("change_type = 'update_postimage' AND ts > 10"),
                ),
            ] {
                let source = format!("{feed}('main.t', {a}, {b})");
                let unfiltered = oracle
                    .sql(&format!("SELECT * FROM {source}"))
                    .await
                    .unwrap()
                    .collect()
                    .await
                    .unwrap_or_else(|e| panic!("seed {seed} [{a}, {b}] {feed}: {e}"));
                let schema = oracle
                    .sql(&format!("SELECT * FROM {source}"))
                    .await
                    .unwrap()
                    .schema()
                    .inner()
                    .clone();
                let table = MemTable::try_new(Arc::clone(&schema), vec![unfiltered]).unwrap();
                oracle.deregister_table("expected").unwrap();
                oracle.register_table("expected", Arc::new(table)).unwrap();
                let names = |filter: &str| {
                    ["val", "s", "f", "extra"]
                        .into_iter()
                        .filter(|name| {
                            filter
                                .split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
                                .any(|word| word == *name)
                        })
                        .collect::<Vec<_>>()
                };
                for filter in common.iter().map(String::as_str).chain(extra) {
                    if names(filter)
                        .iter()
                        .any(|name| schema.field_with_name(name).is_err())
                    {
                        continue;
                    }
                    let expected = rows(&oracle, &format!("SELECT * FROM expected WHERE {filter}"))
                        .await
                        .unwrap();
                    let actual = rows(&paged, &format!("SELECT * FROM {source} WHERE {filter}"))
                        .await
                        .unwrap_or_else(|e| panic!("seed {seed} [{a}, {b}] {feed} {filter}: {e}"));
                    assert_eq!(
                        actual, expected,
                        "seed {seed} window [{a}, {b}]: {feed} WHERE {filter}"
                    );
                    compared += 1;
                }
            }
        }
    }
    compared
}

fn seeds(default: u64) -> std::ops::Range<u64> {
    let n: u64 = std::env::var("FEED_SEEDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default);
    0..n
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn filtered_feeds_return_the_unfiltered_rows_filtered_sqlite() {
    let mut compared = 0;
    for seed in seeds(3) {
        let dir = tempfile::tempdir().unwrap();
        let lake = sqlite_lake(&dir, 5).await;
        let created = history(&lake, seed, 14).await;
        compared += compare_filtered(&lake, seed, created).await;
    }
    assert!(compared > 0);
}

#[cfg(feature = "write-postgres")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn filtered_feeds_return_the_unfiltered_rows_filtered_postgres() {
    let (_container, port) = postgres().await;
    let mut compared = 0;
    for seed in seeds(1) {
        let dir = tempfile::tempdir().unwrap();
        let lake = postgres_lake(port, &format!("pushdown_{seed}"), &dir, 5).await;
        let created = history(&lake, seed, 12).await;
        compared += compare_filtered(&lake, seed, created).await;
    }
    assert!(compared > 0);
}

/// Every node of `plan`, children first.
fn nodes(plan: &Arc<dyn ExecutionPlan>) -> Vec<Arc<dyn ExecutionPlan>> {
    let mut out = Vec::new();
    for child in plan.children() {
        out.extend(nodes(child));
    }
    out.push(Arc::clone(plan));
    out
}

/// The `InsertedRowsExec`s of `sql`'s physical plan.
async fn inserted_rows_execs(ctx: &SessionContext, sql: &str) -> Vec<Arc<dyn ExecutionPlan>> {
    let plan = ctx
        .sql(sql)
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    nodes(&plan)
        .into_iter()
        .filter(|node| node.downcast_ref::<InsertedRowsExec>().is_some())
        .collect()
}

fn inserted(node: &Arc<dyn ExecutionPlan>) -> &InsertedRowsExec {
    node.downcast_ref::<InsertedRowsExec>().unwrap()
}

/// The parquet scans under `plan`, hidden ones included.
fn parquet_scans(plan: &Arc<dyn ExecutionPlan>) -> Vec<Arc<dyn ExecutionPlan>> {
    nodes(plan)
        .into_iter()
        .filter(|node| node.downcast_ref::<DataSourceExec>().is_some())
        .collect()
}

fn scan_predicate(scan: &Arc<dyn ExecutionPlan>) -> bool {
    let scan = scan.downcast_ref::<DataSourceExec>().unwrap();
    let config = scan
        .data_source()
        .as_ref()
        .downcast_ref::<FileScanConfig>()
        .unwrap();
    config
        .file_source()
        .as_ref()
        .downcast_ref::<ParquetSource>()
        .unwrap()
        .filter()
        .is_some()
}

/// Eight files, each of its own ten `ts` values: a filter on `ts` that only
/// one file can match reads only that file, on the positional (`rowid`) path
/// and the plain one, and both report the bytes their scans read.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn files_the_statistics_rule_out_are_not_read() {
    let dir = tempfile::tempdir().unwrap();
    let lake = sqlite_lake(&dir, 0).await;
    lake.sql("CREATE TABLE lake.main.t (id BIGINT, ts BIGINT, val BIGINT, s VARCHAR)")
        .await;
    let created = lake.head();
    for file in 0..8i64 {
        let values: Vec<String> = (0..200i64)
            .map(|i| {
                let id = file * 200 + i;
                format!(
                    "({id}, {}, {}, 'payload-{id:08}')",
                    file * 10 + i % 10,
                    i % 13
                )
            })
            .collect();
        lake.sql(&format!(
            "INSERT INTO lake.main.t VALUES {}",
            values.join(", ")
        ))
        .await;
    }
    let half = created + 4;
    let head = lake.head();
    let (ctx, _) = lake.session();

    // The positional path: rowid is read, so the files are scanned inside
    // `InsertedRowsExec`.
    let feed = |a: i64, b: i64| format!("ducklake_table_insertions('main.t', {a}, {b})");
    let all = inserted_rows_execs(
        &ctx,
        &format!("SELECT rowid, id FROM {}", feed(created, head)),
    )
    .await;
    let filtered = inserted_rows_execs(
        &ctx,
        &format!(
            "SELECT rowid, id FROM {} WHERE ts >= 70",
            feed(created, head)
        ),
    )
    .await;
    assert_eq!(all.len(), 1);
    assert_eq!(filtered.len(), 1);
    assert_eq!(inserted(&all[0]).file_scans().len(), 8);
    assert_eq!(
        inserted(&filtered[0]).file_scans().len(),
        1,
        "seven files ruled out"
    );
    assert!(
        inserted(&filtered[0])
            .file_scans()
            .iter()
            .all(|scan| parquet_scans(scan).iter().all(scan_predicate)),
        "the reader is handed the filter"
    );
    assert!(
        inserted(&all[0])
            .file_scans()
            .iter()
            .all(|scan| parquet_scans(scan).iter().all(|scan| !scan_predicate(scan))),
        "no filter, no predicate"
    );

    // The estimate is non-zero and follows the files read. (The filtered
    // scan also reads `ts`, for the filter kept above it.)
    let same_columns = inserted_rows_execs(
        &ctx,
        &format!("SELECT rowid, id, ts FROM {}", feed(created, head)),
    )
    .await;
    let bytes_same = inserted(&same_columns[0]).estimated_scan_bytes();
    let bytes_all = inserted(&all[0]).estimated_scan_bytes();
    let bytes_filtered = inserted(&filtered[0]).estimated_scan_bytes();
    assert!(bytes_filtered > 0);
    assert!(
        bytes_same >= 7 * bytes_filtered && bytes_same <= 9 * bytes_filtered,
        "eight similar files read {bytes_same} bytes, one {bytes_filtered}"
    );
    let first_half = inserted_rows_execs(
        &ctx,
        &format!("SELECT rowid, id FROM {}", feed(created, half)),
    )
    .await;
    let bytes_half = inserted(&first_half[0]).estimated_scan_bytes();
    assert_eq!(inserted(&first_half[0]).file_scans().len(), 4);
    assert!(
        bytes_all >= 2 * bytes_half - bytes_half / 10
            && bytes_all <= 2 * bytes_half + bytes_half / 10,
        "eight files read {bytes_all} bytes, four {bytes_half}"
    );
    // Projecting more columns reads more of each file.
    let wide = inserted_rows_execs(
        &ctx,
        &format!("SELECT rowid, id, ts, val, s FROM {}", feed(created, head)),
    )
    .await;
    assert!(inserted(&wide[0]).estimated_scan_bytes() > bytes_all);

    // The plain path: the scans are visible, and as few.
    let plain = ctx
        .sql(&format!(
            "SELECT id, ts FROM {} WHERE ts >= 70",
            feed(created, head)
        ))
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    assert_eq!(parquet_scans(&plain).len(), 1);
    assert!(parquet_scans(&plain).iter().all(scan_predicate));

    // And the answers are the files' rows that match, nothing pruned away.
    for sql in [
        format!(
            "SELECT rowid, id, ts FROM {} WHERE ts >= 70",
            feed(created, head)
        ),
        format!("SELECT id, ts FROM {} WHERE ts >= 70", feed(created, head)),
        format!(
            "SELECT snapshot_id, rowid, change_type, id FROM ducklake_table_changes('main.t', {created}, {head}) WHERE ts >= 70"
        ),
    ] {
        let got = rows(&ctx, &sql).await.unwrap();
        assert_eq!(got.len(), 200, "{sql}");
    }

    // A deletion in the window: its rows are reported whatever the filter
    // prunes on the insertion side, and an update's images still pair.
    lake.sql("UPDATE lake.main.t SET val = -1 WHERE id = 5")
        .await;
    lake.sql("DELETE FROM lake.main.t WHERE id = 1405").await;
    let head = lake.head();
    let changes = rows(
        &ctx,
        &format!(
            "SELECT change_type, id, ts FROM ducklake_table_changes('main.t', {created}, {head}) \
             WHERE id = 5 OR id = 1405"
        ),
    )
    .await
    .unwrap();
    let pre = changes
        .iter()
        .filter(|r| r.starts_with("update_preimage|5|"))
        .count();
    let post = changes
        .iter()
        .filter(|r| r.starts_with("update_postimage|5|"))
        .count();
    let deleted = changes
        .iter()
        .filter(|r| r.starts_with("delete|1405|"))
        .count();
    assert_eq!((pre, post, deleted), (1, 1, 1), "{changes:?}");
}

/// `t(id, ts)` with `batches` inlined inserts of `rows` rows each.
async fn inlined_lake(lake: &Lake, batches: i64, rows_per: i64) -> i64 {
    lake.sql("CREATE TABLE lake.main.t (id BIGINT, ts BIGINT, val BIGINT, s VARCHAR)")
        .await;
    let created = lake.head();
    for batch in 0..batches {
        let values: Vec<String> = (0..rows_per)
            .map(|i| {
                let id = batch * rows_per + i;
                format!("({id}, {batch}, {i}, 's{i}')")
            })
            .collect();
        lake.sql(&format!(
            "INSERT INTO lake.main.t VALUES {}",
            values.join(", ")
        ))
        .await;
    }
    created
}

fn plan_names(plan: &Arc<dyn ExecutionPlan>) -> Vec<String> {
    nodes(plan).iter().map(|n| n.name().to_string()).collect()
}

async fn paging_reads_every_inlined_row(lake: &Lake) {
    let created = inlined_lake(lake, 12, 3).await;
    // Delete most of them inline, in one statement.
    lake.sql("DELETE FROM lake.main.t WHERE id % 4 <> 0").await;
    let head = lake.head();
    let (whole, _) = lake.session();
    let (paged, _) = lake.session_with(4, false);
    for feed in ["ducklake_table_insertions", "ducklake_table_deletions", "ducklake_table_changes"]
    {
        let sql = format!("SELECT * FROM {feed}('main.t', {created}, {head})");
        let plan = paged
            .sql(&sql)
            .await
            .unwrap()
            .create_physical_plan()
            .await
            .unwrap();
        assert!(
            plan_names(&plan)
                .iter()
                .any(|name| name == "InlinedPagesExec"),
            "{feed} streams the pages past the first: {:?}",
            plan_names(&plan)
        );
        let expected = rows(&whole, &sql).await.unwrap();
        assert!(expected.len() >= 27, "{feed}: {}", expected.len());
        assert_eq!(rows(&paged, &sql).await.unwrap(), expected, "{feed}");
        // The filter narrows the paged read of the insertions alike.
        let filtered = format!("{sql} WHERE ts >= 5 AND s = 's1'");
        assert_eq!(
            rows(&paged, &filtered).await.unwrap(),
            rows(&whole, &filtered).await.unwrap(),
            "{feed} filtered"
        );
    }
    // A window that fits a page is held by the plan, as before.
    let one = created + 1;
    let sql = format!("SELECT * FROM ducklake_table_insertions('main.t', {one}, {one})");
    let plan = paged
        .sql(&sql)
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    assert!(
        !plan_names(&plan)
            .iter()
            .any(|name| name == "InlinedPagesExec")
    );
    assert_eq!(rows(&paged, &sql).await.unwrap().len(), 3);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn inlined_rows_past_one_page_are_streamed_sqlite() {
    let dir = tempfile::tempdir().unwrap();
    paging_reads_every_inlined_row(&sqlite_lake(&dir, 10).await).await;
}

#[cfg(feature = "write-postgres")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn inlined_rows_past_one_page_are_streamed_postgres() {
    let (_container, port) = postgres().await;
    let dir = tempfile::tempdir().unwrap();
    paging_reads_every_inlined_row(&postgres_lake(port, "paging", &dir, 10).await).await;
}

/// The physical inlined-data tables holding row versions `t` wrote in
/// `[start, end]`.
fn inlined_tables_written(lake: &Lake, start: i64, end: i64) -> std::collections::BTreeSet<String> {
    let table_id = table_id(lake);
    let columns = lake
        .provider
        .get_table_structure(table_id, lake.head())
        .unwrap();
    lake.provider
        .read_inlined_rows(
            table_id,
            &columns,
            datafusion_ducklake::metadata_provider::InlinedRowSelection::InsertedBetween {
                start,
                end,
            },
            None,
        )
        .unwrap()
        .into_iter()
        .filter(|rows| !rows.row_ids.is_empty())
        .map(|rows| rows.table_name)
        .collect()
}

/// A window whose inlined row versions sit in several physical inlined
/// tables (a schema change between inlined inserts starts a new one) reads
/// every version once, whatever the page size: pages run across the end of
/// one table into the next, and a page can end exactly where a table does.
async fn paging_spans_inlined_tables(lake: &Lake) {
    let created = inlined_lake(lake, 4, 3).await;
    let mut rng = Rng(7);
    let mut next_id = 1000;
    // A column added by an inlined append: the next table.
    let schema = schema_with(lake, "extra", arrow::datatypes::DataType::Int64).await;
    append_as(lake, schema, 3, &mut rng, &mut next_id).await;
    for batch in 0..3 {
        let values: Vec<String> = (0..3)
            .map(|i| {
                next_id += 1;
                format!("({next_id}, {}, {i}, 'n{i}', {batch})", 10 + batch)
            })
            .collect();
        lake.sql(&format!(
            "INSERT INTO lake.main.t (id, ts, val, s, extra) VALUES {}",
            values.join(", ")
        ))
        .await;
    }
    // And one more, dropping `s`.
    let schema = schema_without(lake, "s").await;
    append_as(lake, schema, 4, &mut rng, &mut next_id).await;
    lake.sql("DELETE FROM lake.main.t WHERE id % 3 = 0").await;
    let head = lake.head();
    let tables = inlined_tables_written(lake, created, head);
    assert!(tables.len() >= 3, "{tables:?}");
    let (whole, _) = lake.session();
    for feed in ["ducklake_table_insertions", "ducklake_table_deletions", "ducklake_table_changes"]
    {
        let sql = format!("SELECT * FROM {feed}('main.t', {created}, {head})");
        let expected = rows(&whole, &sql).await.unwrap();
        assert!(expected.len() >= 8, "{feed}: {expected:?}");
        for page in 1..=7 {
            let (paged, _) = lake.session_with(page, false);
            let plan = physical(&paged, &sql).await.0;
            assert!(streams_pages(&plan), "{feed} page {page}");
            assert_eq!(
                rows(&paged, &sql).await.unwrap(),
                expected,
                "{feed} in pages of {page}"
            );
        }
    }
    lake.pool
        .settles_at(0, "every pinned read gave its connection back")
        .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn paging_spans_inlined_tables_sqlite() {
    let dir = tempfile::tempdir().unwrap();
    paging_spans_inlined_tables(&sqlite_lake(&dir, 10).await).await;
}

#[cfg(feature = "write-postgres")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn paging_spans_inlined_tables_postgres() {
    let (_container, port) = postgres().await;
    let dir = tempfile::tempdir().unwrap();
    paging_spans_inlined_tables(&postgres_lake(port, "spans", &dir, 10).await).await;
}

/// A server that cannot export a snapshot pins no read, and fails nothing:
/// the plan's read is never asked to export (a failed statement would abort
/// its transaction), the feeds read every page at planning, inside it, and
/// answer as they would pinned. Simulated by a `pg_export_snapshot` that
/// raises `feature_not_supported`, found ahead of the built-in one on the
/// search path.
#[cfg(feature = "write-postgres")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_server_that_cannot_export_snapshots_reads_every_page_at_planning() {
    let (_container, port) = postgres().await;
    let dir = tempfile::tempdir().unwrap();
    let lake = postgres_lake_with(
        port,
        "noexport",
        &dir,
        10,
        &[
            "CREATE FUNCTION public.pg_export_snapshot() RETURNS text LANGUAGE plpgsql AS \
             $$ BEGIN RAISE EXCEPTION 'snapshot export is not available' \
             USING ERRCODE = 'feature_not_supported'; END $$",
            "ALTER DATABASE noexport SET search_path TO public, pg_catalog",
        ],
    )
    .await;
    let ProviderPool::Postgres(pool) = &lake.pool else {
        unreachable!("a PostgreSQL lake")
    };
    let refused = sqlx::query_scalar::<_, String>("SELECT pg_export_snapshot()")
        .fetch_one(pool)
        .await
        .unwrap_err()
        .to_string();
    assert!(refused.contains("not available"), "{refused}");

    let sqls = feeds_over_inlined_rows(&lake).await;
    let (whole, _) = lake.session();
    for sql in &sqls {
        let expected = rows(&whole, sql).await.unwrap();
        assert!(expected.len() > 4, "{sql}: {expected:?}");
        let (plan, task) = plan_paged(&lake, sql).await;
        assert!(!streams_pages(&plan), "{sql}: {:?}", plan_names(&plan));
        lake.pool
            .settles_at(0, "a plan that read every page holds no read")
            .await;
        assert_eq!(collected(plan, task).await.unwrap(), expected, "{sql}");
    }
    // The read asked for a pin stays usable.
    let read = lake.provider.begin_consistent_read().unwrap().unwrap();
    let head = read.provider().get_current_snapshot().unwrap();
    assert!(read.provider().pin_consistent_read().unwrap().is_none());
    assert_eq!(read.provider().get_current_snapshot().unwrap(), head);
    read.end().unwrap();
    lake.pool.settles_at(0, "the read ended").await;
}

/// An export that fails for a reason that may pass (here a cancelled
/// statement, SQLSTATE `57014`) pins nothing for that query and is not
/// remembered: once the export works, a later feed pins its read and streams
/// its pages. Simulated by a `pg_export_snapshot` ahead of the built-in one
/// on the search path that raises while a flag is set and otherwise calls the
/// built-in one.
#[cfg(feature = "write-postgres")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_transient_export_failure_is_probed_again() {
    let (_container, port) = postgres().await;
    let dir = tempfile::tempdir().unwrap();
    let lake = postgres_lake_with(
        port,
        "flaky",
        &dir,
        10,
        &[
            "CREATE TABLE public.export_fails (fails boolean NOT NULL)",
            "INSERT INTO public.export_fails VALUES (true)",
            "CREATE FUNCTION public.pg_export_snapshot() RETURNS text LANGUAGE plpgsql AS \
             $$ BEGIN \
                IF (SELECT fails FROM public.export_fails) THEN \
                    RAISE EXCEPTION 'canceling statement' USING ERRCODE = '57014'; \
                END IF; \
                RETURN pg_catalog.pg_export_snapshot(); \
             END $$",
            "ALTER DATABASE flaky SET search_path TO public, pg_catalog",
        ],
    )
    .await;
    let ProviderPool::Postgres(pool) = &lake.pool else {
        unreachable!("a PostgreSQL lake")
    };
    let sqls = feeds_over_inlined_rows(&lake).await;
    let (whole, _) = lake.session();
    let expected = rows(&whole, &sqls[0]).await.unwrap();
    assert!(expected.len() > 4, "{expected:?}");

    let (plan, task) = plan_paged(&lake, &sqls[0]).await;
    assert!(!streams_pages(&plan), "{:?}", plan_names(&plan));
    lake.pool
        .settles_at(0, "a plan that read every page holds no read")
        .await;
    assert_eq!(collected(plan, task).await.unwrap(), expected);

    sqlx::query("UPDATE public.export_fails SET fails = false")
        .execute(pool)
        .await
        .unwrap();
    let (plan, task) = plan_paged(&lake, &sqls[0]).await;
    assert!(streams_pages(&plan), "{:?}", plan_names(&plan));
    assert_eq!(collected(plan, task).await.unwrap(), expected);
}

/// A catalog read on connections that refuse writes (`PRAGMA query_only`)
/// cannot hold its pages in a temporary table, so its feeds read every page
/// at planning, inside the plan's read, and answer as they would pinned.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_query_only_catalog_reads_every_page_at_planning() {
    use datafusion_ducklake::SqliteMetadataProvider;
    use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};
    let dir = tempfile::tempdir().unwrap();
    let written = sqlite_lake(&dir, 10).await;
    let sqls = feeds_over_inlined_rows(&written).await;
    let options = SqliteConnectOptions::new()
        .filename(dir.path().join("c.db"))
        .pragma("query_only", "ON");
    let pool = SqlitePoolOptions::new()
        .max_connections(5)
        .connect_lazy_with(options);
    let lake = Lake {
        provider: Arc::new(SqliteMetadataProvider::from_pool(pool.clone())),
        pool: ProviderPool::Sqlite(pool.clone()),
        ..written.clone()
    };
    let refused = sqlx::query("CREATE TEMP TABLE probe (x)")
        .execute(&pool)
        .await
        .unwrap_err()
        .to_string();
    assert!(refused.contains("readonly"), "{refused}");
    let (whole, _) = written.session();
    for sql in &sqls {
        let expected = rows(&whole, sql).await.unwrap();
        assert!(expected.len() > 4, "{sql}: {expected:?}");
        let (plan, task) = plan_paged(&lake, sql).await;
        assert!(!streams_pages(&plan), "{sql}: {:?}", plan_names(&plan));
        assert_eq!(collected(plan, task).await.unwrap(), expected, "{sql}");
    }
    lake.pool
        .settles_at(0, "a plan that read every page holds no read")
        .await;
}

async fn physical(
    ctx: &SessionContext,
    sql: &str,
) -> (
    Arc<dyn ExecutionPlan>,
    Arc<datafusion::execution::TaskContext>,
) {
    let plan = ctx
        .sql(sql)
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    (plan, ctx.task_ctx())
}

async fn collected(
    plan: Arc<dyn ExecutionPlan>,
    task: Arc<datafusion::execution::TaskContext>,
) -> Result<Vec<String>, String> {
    datafusion::physical_plan::collect(plan, task)
        .await
        .map(|batches| canonical(&batches))
        .map_err(|e| e.to_string())
}

fn streams_pages(plan: &Arc<dyn ExecutionPlan>) -> bool {
    plan_names(plan)
        .iter()
        .any(|name| name == "InlinedPagesExec")
}

/// The three feeds of `t` over a window of inlined inserts, most of them
/// deleted inline, each selection past one page of 4 rows; the changes feed
/// last.
async fn feeds_over_inlined_rows(lake: &Lake) -> Vec<String> {
    let created = inlined_lake(lake, 6, 3).await;
    lake.sql("DELETE FROM lake.main.t WHERE id % 4 <> 0").await;
    let head = lake.head();
    ["ducklake_table_insertions", "ducklake_table_deletions", "ducklake_table_changes"]
        .map(|feed| format!("SELECT * FROM {feed}('main.t', {created}, {head})"))
        .to_vec()
}

/// Plans each feed reading its inlined rows a page of 4 at a time: through
/// `lake`'s provider, unless the test binds one per query.
async fn plan_paged(
    lake: &Lake,
    sql: &str,
) -> (
    Arc<dyn ExecutionPlan>,
    Arc<datafusion::execution::TaskContext>,
) {
    physical(&lake.session_with(4, false).0, sql).await
}

/// A flush that commits between planning and execution moves the window's
/// inlined rows into a Parquet file the plan does not list. The pages the
/// plan reads at execution read the catalog it planned against, so every feed
/// still answers with the window's rows, and the flush is not held up. A
/// pinned read is read once: the plan executed again says to plan again.
async fn a_flush_after_planning_changes_no_answer(
    lake: &Lake,
    plan: impl AsyncFn(
        &Lake,
        &str,
    ) -> (
        Arc<dyn ExecutionPlan>,
        Arc<datafusion::execution::TaskContext>,
    ),
) {
    let sqls = feeds_over_inlined_rows(lake).await;
    let (whole, _) = lake.session();
    let mut planned = Vec::new();
    for sql in &sqls {
        let expected = rows(&whole, sql).await.unwrap();
        assert!(expected.len() > 4, "{sql}: {expected:?}");
        let paged = plan(lake, sql).await;
        assert!(streams_pages(&paged.0), "{sql}: {:?}", plan_names(&paged.0));
        let held = physical(&whole, sql).await;
        assert!(!streams_pages(&held.0), "{sql}");
        planned.push((sql, paged, held, expected));
    }
    // Each paged plan holds its pinned read, and nothing else.
    lake.pool
        .settles_at(planned.len(), "the planned feeds hold their pinned reads")
        .await;
    lake.sql("CALL ducklake_flush_inlined_data('lake')").await;
    for (sql, (paged, paged_task), (held, held_task), expected) in planned {
        // The flush moved the rows out of the inlined tables.
        assert!(!streams_pages(&plan_paged(lake, sql).await.0), "{sql}");
        assert_eq!(
            collected(Arc::clone(&paged), Arc::clone(&paged_task))
                .await
                .unwrap(),
            expected,
            "{sql}"
        );
        assert_eq!(collected(held, held_task).await.unwrap(), expected, "{sql}");
        // Each pages node has given its pinned read up: executed again, it
        // says to plan again.
        for node in nodes(&paged)
            .into_iter()
            .filter(|node| node.name() == "InlinedPagesExec")
        {
            let again = node
                .execute(0, Arc::clone(&paged_task))
                .err()
                .map(|e| e.to_string())
                .unwrap_or_default();
            assert!(again.contains("plan the query again"), "{sql}: {again}");
        }
        // A feed planned after the flush reads the rows from the file.
        assert_eq!(rows(&whole, sql).await.unwrap(), expected, "{sql}");
    }
    lake.pool
        .settles_at(0, "every pinned read gave its connection back")
        .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_flush_after_planning_changes_no_feed_answer_sqlite() {
    let dir = tempfile::tempdir().unwrap();
    a_flush_after_planning_changes_no_answer(&sqlite_lake(&dir, 10).await, plan_paged).await;
}

#[cfg(feature = "write-postgres")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_flush_after_planning_changes_no_feed_answer_postgres() {
    use datafusion_ducklake::{CatalogReadMemo, MulticatalogProvider};
    let (_container, port) = postgres().await;
    let dir = tempfile::tempdir().unwrap();
    a_flush_after_planning_changes_no_answer(
        &postgres_lake(port, "multi", &dir, 10).await,
        plan_paged,
    )
    .await;
    // Planned as a server plans: through a memo bound for the query, on a
    // read session it ends once the plan is built, before it executes.
    let dir = tempfile::tempdir().unwrap();
    a_flush_after_planning_changes_no_answer(
        &postgres_lake(port, "session", &dir, 10).await,
        async |lake: &Lake, sql: &str| {
            let ProviderPool::Postgres(pool) = &lake.pool else {
                unreachable!("a PostgreSQL lake")
            };
            let provider = MulticatalogProvider::with_pool(pool.clone(), "lake")
                .await
                .unwrap();
            let session = provider.begin_read_session().await.unwrap();
            let bound = CatalogReadMemo::new(CatalogReadMemo::DEFAULT_MAX_BYTES)
                .bind(Arc::new(provider.with_read_session(&session).unwrap()))
                .unwrap();
            let planner = Lake {
                provider: Arc::new(bound),
                ..lake.clone()
            };
            let plan = plan_paged(&planner, sql).await;
            session.end().await.unwrap();
            plan
        },
    )
    .await;
}

/// A catalog in rollback-journal mode cannot hold a read beside a writer, so
/// its feeds read every page at planning, inside the plan's read, and hold
/// no catalog read while they execute: a flush after planning is not held up
/// and changes no answer.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_rollback_journal_catalog_reads_every_page_at_planning() {
    let dir = tempfile::tempdir().unwrap();
    let lake = sqlite_lake_in(&dir, 10, false).await;
    let sqls = feeds_over_inlined_rows(&lake).await;
    let (whole, _) = lake.session();
    let mut planned = Vec::new();
    for sql in &sqls {
        let expected = rows(&whole, sql).await.unwrap();
        let paged = plan_paged(&lake, sql).await;
        assert!(!streams_pages(&paged.0), "{sql}");
        planned.push((paged, expected));
    }
    lake.pool
        .settles_at(0, "a plan that read every page holds no read")
        .await;
    lake.sql("CALL ducklake_flush_inlined_data('lake')").await;
    for ((plan, task), expected) in planned {
        assert_eq!(collected(plan, task).await.unwrap(), expected);
    }
}

/// A plan whose pool has no connection free to pin a read on does not wait
/// for one: its feeds read every page at planning, inside the plan's read,
/// answer as they would pinned, and hold no connection once planned. With a
/// connection free again, a feed pins its read and streams its pages.
async fn an_exhausted_pool_reads_every_page_at_planning(lake: &Lake) {
    let sqls = feeds_over_inlined_rows(lake).await;
    let (whole, _) = lake.session();
    let mut expected = Vec::new();
    for sql in &sqls {
        expected.push(rows(&whole, sql).await.unwrap());
    }
    lake.pool.settles_at(0, "before exhausting the pool").await;
    let held = lake.pool.exhaust().await;
    let taken = lake.pool.in_use();
    for (sql, expected) in sqls.iter().zip(&expected) {
        let (plan, task) = plan_paged(lake, sql).await;
        assert!(!streams_pages(&plan), "{sql}: {:?}", plan_names(&plan));
        lake.pool
            .settles_at(taken, "a plan that read every page holds no read")
            .await;
        assert_eq!(&collected(plan, task).await.unwrap(), expected, "{sql}");
    }
    drop(held);
    lake.pool.settles_at(0, "the pool given back").await;
    let (plan, task) = plan_paged(lake, &sqls[0]).await;
    assert!(streams_pages(&plan), "{:?}", plan_names(&plan));
    assert_eq!(collected(plan, task).await.unwrap(), expected[0]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_exhausted_pool_reads_every_page_at_planning_sqlite() {
    let dir = tempfile::tempdir().unwrap();
    an_exhausted_pool_reads_every_page_at_planning(&sqlite_lake(&dir, 10).await).await;
}

#[cfg(feature = "write-postgres")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_exhausted_pool_reads_every_page_at_planning_postgres() {
    let (_container, port) = postgres().await;
    let dir = tempfile::tempdir().unwrap();
    an_exhausted_pool_reads_every_page_at_planning(&postgres_lake(port, "multi", &dir, 10).await)
        .await;
}

/// The pinned read of a feed, shared by both selections of the changes
/// feed, gives its connection back when the plan is dropped unexecuted, when
/// the query is cancelled after one batch (its stream and plan dropped: a
/// pages node the plan had yet to execute holds its handle until the plan
/// drops), and when the last page is read, with the plan still alive.
async fn a_feed_gives_its_pinned_connection_back(lake: &Lake) {
    let sqls = feeds_over_inlined_rows(lake).await;
    let sql = sqls.last().unwrap();
    lake.pool.settles_at(0, "before planning").await;

    let (plan, _) = plan_paged(lake, sql).await;
    lake.pool
        .settles_at(1, "a plan holds one pinned read for both selections")
        .await;
    drop(plan);
    lake.pool
        .settles_at(0, "a plan dropped unexecuted ends its pinned read")
        .await;

    let (plan, task) = plan_paged(lake, sql).await;
    let mut stream = datafusion::physical_plan::execute_stream(Arc::clone(&plan), task).unwrap();
    futures::StreamExt::next(&mut stream)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(lake.pool.in_use(), 1, "the stream holds the pinned read");
    drop(stream);
    drop(plan);
    lake.pool
        .settles_at(0, "a query cancelled mid-read ends its pinned read")
        .await;

    let (plan, task) = plan_paged(lake, sql).await;
    let (whole, _) = lake.session();
    assert_eq!(
        collected(Arc::clone(&plan), task).await.unwrap(),
        rows(&whole, sql).await.unwrap()
    );
    lake.pool
        .settles_at(0, "a read to its last page ends its pinned read")
        .await;
    drop(plan);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_feed_gives_its_pinned_connection_back_sqlite() {
    let dir = tempfile::tempdir().unwrap();
    a_feed_gives_its_pinned_connection_back(&sqlite_lake(&dir, 10).await).await;
}

#[cfg(feature = "write-postgres")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_feed_gives_its_pinned_connection_back_postgres() {
    let (_container, port) = postgres().await;
    let dir = tempfile::tempdir().unwrap();
    a_feed_gives_its_pinned_connection_back(&postgres_lake(port, "multi", &dir, 10).await).await;
}

/// A read pinned to a consistent read sees that read's catalog after the read
/// ends and a writer commits, while the provider sees the commit; it holds
/// one connection until it ends. A provider outside a consistent read pins
/// nothing.
async fn a_pinned_read_keeps_its_snapshot(lake: &Lake) {
    assert!(lake.provider.pin_consistent_read().unwrap().is_none());
    let read = lake.provider.begin_consistent_read().unwrap().unwrap();
    let head = read.provider().get_current_snapshot().unwrap();
    let pinned = read
        .provider()
        .pin_consistent_read()
        .unwrap()
        .expect("a read pinned to the open one");
    read.end().unwrap();
    lake.writer.create_snapshot().unwrap();
    assert_eq!(pinned.provider().get_current_snapshot().unwrap(), head);
    assert!(lake.head() > head);
    lake.pool
        .settles_at(1, "the pinned read holds one connection")
        .await;
    pinned.end().unwrap();
    lake.pool.settles_at(0, "an ended pinned read").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_pinned_read_keeps_its_snapshot_sqlite() {
    let dir = tempfile::tempdir().unwrap();
    let lake = sqlite_lake(&dir, 10).await;
    a_pinned_read_keeps_its_snapshot(&lake).await;
    // SQLite takes the pinned read's snapshot apart, and pins none when a
    // commit falls between the two reads.
    let read = lake.provider.begin_consistent_read().unwrap().unwrap();
    read.provider().get_current_snapshot().unwrap();
    lake.writer.create_snapshot().unwrap();
    assert!(read.provider().pin_consistent_read().unwrap().is_none());
    read.end().unwrap();
    lake.pool.settles_at(0, "a read not pinned").await;
    // Nor in rollback-journal mode.
    let dir = tempfile::tempdir().unwrap();
    let lake = sqlite_lake_in(&dir, 10, false).await;
    let read = lake.provider.begin_consistent_read().unwrap().unwrap();
    read.provider().get_current_snapshot().unwrap();
    assert!(read.provider().pin_consistent_read().unwrap().is_none());
    read.end().unwrap();
}

#[cfg(feature = "write-postgres")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_pinned_read_keeps_its_snapshot_postgres() {
    let (_container, port) = postgres().await;
    let dir = tempfile::tempdir().unwrap();
    a_pinned_read_keeps_its_snapshot(&postgres_lake(port, "multi", &dir, 10).await).await;
    let dir = tempfile::tempdir().unwrap();
    a_pinned_read_keeps_its_snapshot(&postgres_single_lake(port, "single", &dir).await).await;
}
