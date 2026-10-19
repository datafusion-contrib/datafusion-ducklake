//! One inlining configuration for every write path, and row-level APIs that
//! handle inlined rows and inlined deletions without the caller knowing where
//! a row lives.
//!
//! Each scenario runs on every backend with a writer that inlines: SQLite and
//! DuckDB per test, MySQL and multi-catalog PostgreSQL as one test per backend
//! (one container, a fresh catalog per scenario). Single-catalog PostgreSQL,
//! which inlines nothing and stages no row mutation (no keyed mutation, no
//! SQL `UPDATE`), runs the configuration and change-ledger scenarios.
//! Every scenario names the failure mode it guards against.

#![cfg(feature = "write")]
#![cfg(any(feature = "write-sqlite", feature = "write-duckdb"))]

use std::collections::{BTreeMap, HashSet};
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

use arrow::array::{Array, Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::prelude::SessionContext;
use datafusion_ducklake::change_detection::{
    TableDataChanges, last_rows_changed_snapshot, table_data_changes,
};

use datafusion_ducklake::sort::{NullOrder, SortDirection, SortField};
use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeError, DuckLakeTable, DuckLakeTableWriter, DuckLakeWriteOptions,
    InlinedFlushPolicy, MergeOptions, MetadataProvider, MetadataWriter, RewriteOptions,
    RowIdLayout, RowLocation, WriteMode, execute_ducklake_sql,
};
use object_store::local::LocalFileSystem;

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

/// A catalog under test: a writer configured with an inlining limit (or none),
/// a provider over the same catalog, and its data directory.
pub(crate) struct Lake {
    pub(crate) writer: Arc<dyn MetadataWriter>,
    pub(crate) provider: Arc<dyn MetadataProvider>,
    pub(crate) data: PathBuf,
    /// Whether the backend inlines rows (every backend but single-catalog
    /// PostgreSQL).
    pub(crate) inlines: bool,
    pub(crate) _dir: tempfile::TempDir,
}

type LakeFuture = Pin<Box<dyn Future<Output = Lake> + Send>>;

/// Makes a fresh, empty catalog whose writer applies the given limit.
pub(crate) type MakeLake = Arc<dyn Fn(Option<usize>) -> LakeFuture + Send + Sync>;

/// [`table_data_changes`] over a fresh session, unwrapped.
pub(crate) async fn data_changes(
    provider: &Arc<dyn MetadataProvider>,
    table_id: i64,
    after: i64,
    through: i64,
) -> TableDataChanges {
    table_data_changes(
        &SessionContext::new().state(),
        provider,
        table_id,
        after,
        through,
    )
    .await
    .unwrap()
}

/// [`last_rows_changed_snapshot`] over a fresh session, unwrapped.
pub(crate) async fn last_rows_changed(
    provider: &Arc<dyn MetadataProvider>,
    table_id: i64,
    through: i64,
) -> Option<i64> {
    last_rows_changed_snapshot(&SessionContext::new().state(), provider, table_id, through)
        .await
        .unwrap()
}

fn scratch() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::TempDir::new().unwrap();
    let data = dir.path().join("data");
    std::fs::create_dir_all(&data).unwrap();
    (dir, data)
}

#[cfg(feature = "write-sqlite")]
pub(crate) fn sqlite() -> MakeLake {
    Arc::new(|limit| {
        Box::pin(async move {
            let (dir, data) = scratch();
            let conn = format!("sqlite:{}?mode=rwc", dir.path().join("lake.db").display());
            let mut writer = datafusion_ducklake::SqliteMetadataWriter::new_with_init(&conn)
                .await
                .unwrap();
            if let Some(limit) = limit {
                writer = writer.with_data_inlining_row_limit(limit);
            }
            writer.set_data_path(data.to_str().unwrap()).unwrap();
            let provider = datafusion_ducklake::SqliteMetadataProvider::new(&conn)
                .await
                .unwrap();
            Lake {
                writer: Arc::new(writer),
                provider: Arc::new(provider),
                data,
                inlines: true,
                _dir: dir,
            }
        })
    })
}

#[cfg(feature = "write-duckdb")]
pub(crate) fn duckdb() -> MakeLake {
    Arc::new(|limit| {
        Box::pin(async move {
            let (dir, data) = scratch();
            let path = dir.path().join("lake.ducklake");
            let mut writer =
                datafusion_ducklake::DuckdbMetadataWriter::new_with_init(path.to_str().unwrap())
                    .unwrap();
            if let Some(limit) = limit {
                writer = writer.with_data_inlining_row_limit(limit);
            }
            writer.set_data_path(data.to_str().unwrap()).unwrap();
            let provider = writer.metadata_provider();
            Lake {
                writer: Arc::new(writer),
                provider: Arc::new(provider),
                data,
                inlines: true,
                _dir: dir,
            }
        })
    })
}

/// A fresh catalog in a multi-catalog store on `pool` per call.
#[cfg(feature = "write-postgres")]
pub(crate) fn multicatalog(pool: sqlx::PgPool) -> MakeLake {
    let counter = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    Arc::new(move |limit| {
        let pool = pool.clone();
        let counter = Arc::clone(&counter);
        Box::pin(async move {
            let (dir, data) = scratch();
            // `lake0`, `lake1`, ...: the first name another maker on the same
            // store has not taken.
            let manager = datafusion_ducklake::MulticatalogManager::new(pool.clone());
            let taken: HashSet<String> = manager
                .list_catalogs()
                .await
                .unwrap()
                .into_iter()
                .map(|(_, name)| name)
                .collect();
            let name = loop {
                let name = format!(
                    "lake{}",
                    counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
                );
                if !taken.contains(&name) {
                    break name;
                }
            };
            let catalog_id = manager.create_catalog(&name).await.unwrap();
            let mut writer =
                datafusion_ducklake::PostgresMetadataWriter::with_pool(pool.clone(), catalog_id)
                    .await
                    .unwrap();
            if let Some(limit) = limit {
                writer = writer.with_data_inlining_row_limit(limit);
            }
            writer.set_data_path(data.to_str().unwrap()).unwrap();
            let provider = datafusion_ducklake::MulticatalogProvider::with_pool(pool, &name)
                .await
                .unwrap();
            Lake {
                writer: Arc::new(writer),
                provider: Arc::new(provider),
                data,
                inlines: true,
                _dir: dir,
            }
        })
    })
}

/// A fresh database on the MySQL server at `port` per call.
#[cfg(feature = "write-mysql")]
pub(crate) fn mysql(port: u16) -> MakeLake {
    let counter = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    Arc::new(move |limit| {
        let counter = Arc::clone(&counter);
        Box::pin(async move {
            let (dir, data) = scratch();
            let database = format!(
                "lake{}",
                counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
            );
            let admin = sqlx::MySqlPool::connect(&format!("mysql://root@127.0.0.1:{port}/test"))
                .await
                .unwrap();
            sqlx::query(sqlx::AssertSqlSafe(format!("CREATE DATABASE {database}")))
                .execute(&admin)
                .await
                .unwrap();
            admin.close().await;
            let conn = format!("mysql://root@127.0.0.1:{port}/{database}");
            let mut writer = datafusion_ducklake::MySqlMetadataWriter::new_with_init(&conn)
                .await
                .unwrap();
            if let Some(limit) = limit {
                writer = writer.with_data_inlining_row_limit(limit);
            }
            writer.set_data_path(data.to_str().unwrap()).unwrap();
            let provider = datafusion_ducklake::MySqlMetadataProvider::new(&conn)
                .await
                .unwrap();
            Lake {
                writer: Arc::new(writer),
                provider: Arc::new(provider),
                data,
                inlines: true,
                _dir: dir,
            }
        })
    })
}

/// A fresh database on the PostgreSQL server at `port` per call, with the
/// standard single-catalog layout.
#[cfg(feature = "write-postgres")]
pub(crate) fn postgres_single(port: u16) -> MakeLake {
    let counter = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    Arc::new(move |limit| {
        let counter = Arc::clone(&counter);
        Box::pin(async move {
            let (dir, data) = scratch();
            let database = format!(
                "lake{}",
                counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
            );
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
            let conn = format!("postgresql://postgres:postgres@127.0.0.1:{port}/{database}");
            let mut writer =
                datafusion_ducklake::PostgresSingleCatalogMetadataWriter::new_with_init(&conn)
                    .await
                    .unwrap();
            if let Some(limit) = limit {
                writer = writer.with_data_inlining_row_limit(limit);
            }
            writer.set_data_path(data.to_str().unwrap()).unwrap();
            let provider = datafusion_ducklake::PostgresMetadataProvider::new(&conn)
                .await
                .unwrap();
            Lake {
                writer: Arc::new(writer),
                provider: Arc::new(provider),
                data,
                inlines: false,
                _dir: dir,
            }
        })
    })
}

impl Lake {
    pub(crate) fn catalog(&self) -> DuckLakeCatalog {
        DuckLakeCatalog::with_writer(Arc::clone(&self.provider), Arc::clone(&self.writer)).unwrap()
    }

    /// Run `statement` through `execute_ducklake_sql`.
    pub(crate) async fn sql(&self, statement: &str) -> Result<(), String> {
        let ctx = SessionContext::new();
        DuckLakeCatalog::register(&ctx, "lake", self.catalog());
        // Each statement's futures on the heap, so a scenario of many
        // statements keeps a small stack in a debug build.
        let frame = Box::pin(execute_ducklake_sql(&ctx, &self.catalog(), statement))
            .await
            .map_err(|e| e.to_string())?;
        Box::pin(frame.collect()).await.map_err(|e| e.to_string())?;
        Ok(())
    }

    /// `rowid, <columns>` of `main.<table>` at the head (or `snapshot`), one
    /// line per row, ordered by `rowid`.
    pub(crate) async fn rows_at(&self, table: &str, snapshot: Option<i64>) -> Vec<String> {
        let ctx = SessionContext::new();
        let snapshot = match snapshot {
            Some(snapshot) => snapshot,
            None => self.provider.get_current_snapshot().unwrap(),
        };
        let catalog = DuckLakeCatalog::with_snapshot(Arc::clone(&self.provider), snapshot)
            .unwrap()
            .with_row_lineage(true);
        ctx.register_catalog("lake", Arc::new(catalog));
        let frame = Box::pin(ctx.sql(&format!("SELECT * FROM lake.main.{table} ORDER BY rowid")))
            .await
            .unwrap();
        let batches = Box::pin(frame.collect()).await.unwrap();
        lines(&batches, true)
    }

    pub(crate) async fn rows(&self, table: &str) -> Vec<String> {
        self.rows_at(table, None).await
    }

    /// A writable handle on `main.<table>` at the head, and its session.
    pub(crate) async fn table(&self, table: &str) -> (SessionContext, DuckLakeTable) {
        let ctx = SessionContext::new();
        DuckLakeCatalog::register(&ctx, "lake", self.catalog());
        let provider = ctx
            .catalog("lake")
            .unwrap()
            .schema("main")
            .unwrap()
            .table(table)
            .await
            .unwrap()
            .unwrap();
        let table = (provider.as_ref() as &dyn std::any::Any)
            .downcast_ref::<DuckLakeTable>()
            .expect("a DuckLakeTable")
            .clone();
        (ctx, table)
    }

    pub(crate) fn head(&self) -> i64 {
        self.provider.get_current_snapshot().unwrap()
    }

    pub(crate) fn table_id(&self, table: &str) -> i64 {
        let head = self.head();
        let schema = self
            .provider
            .get_schema_by_name("main", head)
            .unwrap()
            .unwrap();
        self.provider
            .get_table_by_name(schema.schema_id, table, head)
            .unwrap()
            .unwrap()
            .table_id
    }

    pub(crate) fn schema_id(&self) -> i64 {
        self.provider
            .get_schema_by_name("main", self.head())
            .unwrap()
            .unwrap()
            .schema_id
    }

    /// Live data files of `table` at the head.
    pub(crate) fn data_files(&self, table: &str) -> usize {
        self.provider
            .get_table_files_for_select(self.table_id(table), self.head())
            .unwrap()
            .len()
    }

    /// Live inlined rows of `table` at the head.
    pub(crate) fn inlined_rows(&self, table: &str) -> usize {
        let table_id = self.table_id(table);
        let head = self.head();
        let columns = self.provider.get_table_structure(table_id, head).unwrap();
        self.provider
            .get_inlined_data_with_row_ids(table_id, head, &columns)
            .unwrap()
            .iter()
            .map(|data| data.row_ids.len())
            .sum()
    }

    /// Inlined deletions of `table`'s Parquet rows visible at the head.
    pub(crate) fn inlined_deletes(&self, table: &str) -> usize {
        self.provider
            .get_inlined_deletes(self.table_id(table), self.head())
            .unwrap()
            .len()
    }

    /// Live data files of `table` with a live delete file.
    pub(crate) fn delete_files(&self, table: &str) -> usize {
        self.provider
            .get_table_files_for_select(self.table_id(table), self.head())
            .unwrap()
            .iter()
            .filter(|file| file.delete_file.is_some())
            .count()
    }

    pub(crate) fn table_writer(&self) -> DuckLakeTableWriter {
        DuckLakeTableWriter::new(Arc::clone(&self.writer), Arc::new(LocalFileSystem::new()))
            .unwrap()
    }

    /// Append `batch` to `main.<table>` with the writer's configuration.
    pub(crate) async fn append(&self, table: &str, batch: &RecordBatch) -> usize {
        let writer = self.table_writer();
        Box::pin(writer.append_table("main", table, std::slice::from_ref(batch)))
            .await
            .unwrap()
            .files_written
    }

    /// Append `batch` to Parquet whatever the limit, as a write of more rows
    /// than it would be.
    pub(crate) async fn append_parquet(&self, table: &str, batch: &RecordBatch) {
        let writer = self
            .table_writer()
            .with_options(&DuckLakeWriteOptions::default().with_data_inlining_row_limit(0));
        let files = Box::pin(writer.append_table("main", table, std::slice::from_ref(batch)))
            .await
            .unwrap()
            .files_written;
        assert!(files > 0, "a Parquet append writes a file");
    }
}

/// Rows as comma-separated cells (NULL as `NULL`); with `rowid_first`, a
/// `rowid` column is moved to the front.
pub(crate) fn lines(batches: &[RecordBatch], rowid_first: bool) -> Vec<String> {
    let mut out = Vec::new();
    for batch in batches {
        let schema = batch.schema();
        let mut order: Vec<usize> = (0..batch.num_columns()).collect();
        if rowid_first && let Ok(rowid) = schema.index_of("rowid") {
            order.retain(|index| *index != rowid);
            order.insert(0, rowid);
        }
        for row in 0..batch.num_rows() {
            let cells = order
                .iter()
                .map(|column| {
                    let array = batch.column(*column);
                    if array.is_null(row) {
                        "NULL".to_string()
                    } else {
                        arrow::util::display::array_value_to_string(array, row).unwrap()
                    }
                })
                .collect::<Vec<_>>();
            out.push(cells.join(","));
        }
    }
    out
}

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, true),
        Field::new("v", DataType::Utf8, true),
    ]))
}

/// Rows `(id, v)` with `v` = `prefix` followed by the id.
pub(crate) fn batch(ids: impl IntoIterator<Item = i64>, prefix: &str) -> RecordBatch {
    let ids: Vec<i64> = ids.into_iter().collect();
    let values: Vec<String> = ids.iter().map(|id| format!("{prefix}{id}")).collect();
    RecordBatch::try_new(
        schema(),
        vec![Arc::new(Int64Array::from(ids)), Arc::new(StringArray::from(values))],
    )
    .unwrap()
}

fn keys(ids: impl IntoIterator<Item = Option<i64>>) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, true)])),
        vec![Arc::new(Int64Array::from(ids.into_iter().collect::<Vec<_>>()))],
    )
    .unwrap()
}

/// `id -> v` of the rows a line list holds (lines `rowid,id,v`).
fn values(rows: &[String]) -> BTreeMap<i64, Vec<String>> {
    let mut out: BTreeMap<i64, Vec<String>> = BTreeMap::new();
    for row in rows {
        let cells: Vec<&str> = row.split(',').collect();
        out.entry(cells[1].parse().unwrap())
            .or_default()
            .push(cells[2].to_string());
    }
    out
}

/// `id -> rowid` of the rows a line list holds.
fn row_ids(rows: &[String]) -> BTreeMap<i64, i64> {
    rows.iter()
        .map(|row| {
            let cells: Vec<&str> = row.split(',').collect();
            (cells[1].parse().unwrap(), cells[0].parse().unwrap())
        })
        .collect()
}

/// Create `main.<table>` with `(name, type)` columns, as `CREATE TABLE` does
/// (the schema too, when new).
pub(crate) fn create_with(
    lake: &Lake,
    table: &str,
    columns: &[(&str, DataType)],
) -> Result<(), String> {
    let columns = columns
        .iter()
        .map(|(name, data_type)| {
            datafusion_ducklake::metadata_writer::ColumnDef::from_arrow(*name, data_type, true)
                .unwrap()
        })
        .collect::<Vec<_>>();
    let setup = lake
        .writer
        .begin_write_transaction("main", table, &columns, WriteMode::Replace)
        .map_err(|e| e.to_string())?;
    lake.writer
        .publish_snapshot(
            setup.table_id,
            "main",
            table,
            setup.snapshot_id,
            WriteMode::Replace,
            setup.base_snapshot_id,
            &columns,
            &setup.field_ids,
        )
        .map_err(|e| e.to_string())?;
    Ok(())
}

pub(crate) async fn create(lake: &Lake, table: &str) {
    create_with(
        lake,
        table,
        &[("id", DataType::Int64), ("v", DataType::Utf8)],
    )
    .unwrap();
}

// ---------------------------------------------------------------------------
// D1: one inlining configuration
// ---------------------------------------------------------------------------

/// A writer configured with a limit of `0` creates a table with a column named
/// like an inlined data table's system column through a direct writer call,
/// gets no inlined data table for it, and writes Parquet on every path. Guards
/// against direct `MetadataWriter` calls falling back to the default `10`
/// (refusing the column) when the caller configured `0`.
pub(crate) async fn writer_limit_reaches_every_path(make: MakeLake) {
    let lake = make(Some(0)).await;
    let reserved = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("row_id", DataType::Int64, true),
            Field::new("v", DataType::Utf8, true),
        ])),
        vec![Arc::new(Int64Array::from(vec![1, 2])), Arc::new(StringArray::from(vec!["a", "b"]))],
    )
    .unwrap();
    assert_eq!(
        lake.append("reserved", &reserved).await,
        1,
        "a writer limit of 0 writes Parquet"
    );
    assert_eq!(
        lake.writer
            .table_data_inlining_row_limit("main", "reserved")
            .unwrap(),
        0
    );
    create(&lake, "t").await;
    lake.sql("INSERT INTO lake.main.t VALUES (1, 'a'), (2, 'b')")
        .await
        .unwrap();
    assert_eq!(lake.data_files("t"), 1, "SQL INSERT writes Parquet");
    let mut session = lake
        .table_writer()
        .begin_write("main", "t", &schema(), WriteMode::Append)
        .unwrap();
    assert!(!session.is_inlining());
    session.write_batch(&batch([3], "c")).unwrap();
    assert_eq!(session.finish().await.unwrap().files_written, 1);
    if lake.inlines {
        assert_eq!(lake.inlined_rows("t"), 0);
    }
    assert_eq!(lake.rows("t").await.len(), 3);

    // With the default limit the same column name is refused, as official
    // refuses it while inlining is enabled.
    let lake = make(None).await;
    let error = lake
        .table_writer()
        .append_table("main", "reserved", std::slice::from_ref(&reserved))
        .await
        .unwrap_err();
    assert!(
        error.to_string().contains("reserved by DuckLake"),
        "unexpected error: {error}"
    );
}

/// The limit resolves table > schema > catalog-wide > writer > `10`, as
/// official's `DataInliningRowLimit` resolves it, and every writer can store
/// each scope.
pub(crate) async fn limit_precedence(make: MakeLake) {
    let lake = make(Some(3)).await;
    create(&lake, "t").await;
    create(&lake, "u").await;
    let limit = |table: &str| {
        lake.writer
            .table_data_inlining_row_limit("main", table)
            .unwrap()
    };
    assert_eq!(limit("t"), 3, "the writer's limit");
    assert_eq!(limit("missing"), 3, "a table to be created");
    lake.writer
        .set_global_setting("data_inlining_row_limit", "7")
        .unwrap();
    assert_eq!(limit("t"), 7, "catalog-wide outranks the writer");
    lake.writer
        .set_schema_setting(lake.schema_id(), "data_inlining_row_limit", "5")
        .unwrap();
    assert_eq!(limit("t"), 5, "schema outranks catalog-wide");
    lake.writer
        .set_table_setting(lake.table_id("t"), "data_inlining_row_limit", "2")
        .unwrap();
    assert_eq!(limit("t"), 2, "table outranks schema");
    assert_eq!(limit("u"), 5);
    let default = make(None).await;
    assert_eq!(
        default
            .writer
            .table_data_inlining_row_limit("main", "t")
            .unwrap(),
        10,
        "official's default"
    );
    let missing = lake
        .writer
        .set_schema_setting(i64::MAX, "data_inlining_row_limit", "1");
    assert!(matches!(missing, Err(DuckLakeError::SchemaNotFound(_))));
}

/// `sort_on_insert` resolves table > schema > catalog-wide (on multi-catalog
/// PostgreSQL, catalog-scoped) > the writer's option > `true`, on only while
/// it reads exactly `true`, as official's `PlanInsert` reads it through
/// `GetConfigOption`; every writer stores and reads each scope. Guards
/// against a backend's `table_setting` missing a scope or ranking the writer's
/// option over a stored one.
pub(crate) async fn sort_on_insert_precedence(make: MakeLake) {
    let lake = make(None).await;
    create(&lake, "t").await;
    create(&lake, "u").await;
    let resolved = |table: &str, option: Option<bool>| {
        let mut options = DuckLakeWriteOptions::default();
        options.sort_on_insert = option;
        lake.table_writer()
            .with_options(&options)
            .table_sort_on_insert("main", table)
            .unwrap()
    };
    assert!(resolved("t", None), "official's default");
    assert!(!resolved("t", Some(false)), "the writer's option");
    assert!(!resolved("missing", Some(false)), "a table to be created");
    lake.writer
        .set_global_setting("sort_on_insert", "false")
        .unwrap();
    assert!(
        !resolved("t", Some(true)),
        "catalog-wide outranks the writer"
    );
    lake.writer
        .set_schema_setting(lake.schema_id(), "sort_on_insert", "true")
        .unwrap();
    assert!(resolved("t", Some(false)), "schema outranks catalog-wide");
    lake.writer
        .set_table_setting(lake.table_id("t"), "sort_on_insert", "false")
        .unwrap();
    assert!(!resolved("t", Some(true)), "table outranks schema");
    assert!(
        resolved("u", Some(false)),
        "another table reads its schema's"
    );
    lake.writer
        .set_table_setting(lake.table_id("t"), "sort_on_insert", "TRUE")
        .unwrap();
    assert!(
        resolved("t", Some(false)),
        "`TRUE` is cast and stored as `true`, as official's set_option stores it"
    );
}

/// Each writer stores a setting as official's `ducklake_set_option` stores the
/// same value (checked with the DuckDB CLI: `CALL ducklake_set_option('l',
/// '<key>', '<value>')`), at table, schema and catalog scope, and refuses the
/// values official refuses. Guards against a stored `TRUE` (which official
/// reads as off) or `0.5` (official stores `0.500000`) diverging from what
/// official writes.
pub(crate) async fn settings_store_as_official(make: MakeLake) {
    let lake = make(None).await;
    create(&lake, "t").await;
    // `u` has no table-scoped setting, so it reads its schema's.
    create(&lake, "u").await;
    let table_id = lake.table_id("t");
    let schema_id = lake.schema_id();
    for (key, value, stored) in [
        ("sort_on_insert", "T", Some("true")),
        ("sort_on_insert", "No", Some("false")),
        ("sort_on_insert", " true ", None),
        ("sort_on_insert", "maybe", None),
        ("auto_compact", "0", Some("false")),
        ("auto_compact", "yes", Some("true")),
        // Official reads `hive_file_pattern` as `GetValue<bool>()` of the
        // `VARCHAR`, which casts through `INT8`: any non-zero integer is on,
        // and `'true'` fails ("Could not convert string 'true' to INT8").
        ("hive_file_pattern", "FALSE", None),
        ("hive_file_pattern", "true", None),
        ("hive_file_pattern", "2", Some("true")),
        ("hive_file_pattern", "0", Some("false")),
        ("hive_file_pattern", "0.4", Some("false")),
        ("hive_file_pattern", "-0.5", Some("true")),
        ("hive_file_pattern", "0x10", Some("true")),
        ("hive_file_pattern", " 0x10 ", None),
        ("hive_file_pattern", "128", None),
        ("hive_file_pattern", "127.5", None),
        ("rewrite_delete_threshold", "0.5", Some("0.500000")),
        ("rewrite_delete_threshold", "0.1234567", Some("0.123457")),
        ("rewrite_delete_threshold", " 0.25 ", Some("0.250000")),
        ("rewrite_delete_threshold", "1", Some("1.000000")),
        ("rewrite_delete_threshold", "1.5", None),
        ("rewrite_delete_threshold", "abc", None),
        ("parquet_compression", "ZSTD", Some("zstd")),
        ("parquet_compression", "lzo", None),
        ("parquet_version", "2", Some("V2")),
        ("parquet_version", "V2", None),
        ("parquet_version", "3", None),
        ("parquet_compression_level", "3.6", Some("4")),
        ("parquet_compression_level", "-1", None),
        ("parquet_row_group_size", "1e3", Some("1000")),
        ("parquet_row_group_size", "0", None),
        ("parquet_row_group_size_bytes", "1.5MB", Some("1500000")),
        ("parquet_row_group_size_bytes", "0 kb", None),
        ("target_file_size", "2 GiB", Some("2147483648")),
        ("target_file_size", "1000", None),
        ("data_inlining_row_limit", "1.5", Some("2")),
        ("data_inlining_row_limit", "-1", None),
    ] {
        type Set<'a> = Box<dyn Fn(&str, &str) -> Result<(), DuckLakeError> + 'a>;
        type Get<'a> = Box<dyn Fn(&str) -> Option<String> + 'a>;
        let scopes: [(&str, Set, Get); 3] = [
            (
                "table",
                Box::new(|key, value| lake.writer.set_table_setting(table_id, key, value)),
                Box::new(|key| lake.writer.table_setting("main", "t", key).unwrap()),
            ),
            (
                "schema",
                Box::new(|key, value| lake.writer.set_schema_setting(schema_id, key, value)),
                Box::new(|key| lake.writer.table_setting("main", "u", key).unwrap()),
            ),
            (
                "catalog",
                Box::new(|key, value| lake.writer.set_global_setting(key, value)),
                Box::new(|key| {
                    lake.provider
                        .get_metadata_settings(None, None)
                        .unwrap()
                        .remove(key)
                }),
            ),
        ];
        for (scope, set, get) in &scopes {
            let result = set(key, value);
            match stored {
                Some(stored) => {
                    result.unwrap_or_else(|e| panic!("{scope} {key} = {value:?}: {e}"));
                    assert_eq!(
                        get(key).as_deref(),
                        Some(stored),
                        "{scope} {key} = {value:?}"
                    );
                },
                None => assert!(result.is_err(), "{scope} {key} = {value:?} is refused"),
            }
        }
    }
    // `require_commit_message` casts through `INT8` too, and a key official
    // does not know is refused ("Unsupported option foo_bar").
    for (value, stored) in [("yes", None), ("2", Some("true")), ("0", Some("false"))] {
        let result = lake
            .writer
            .set_global_setting("require_commit_message", value);
        match stored {
            Some(stored) => {
                result.unwrap();
                assert_eq!(
                    lake.provider
                        .get_metadata_settings(None, None)
                        .unwrap()
                        .remove("require_commit_message")
                        .as_deref(),
                    Some(stored)
                );
            },
            None => assert!(result.is_err(), "require_commit_message = {value:?}"),
        }
    }
    let unknown = lake.writer.set_global_setting("foo_bar", "1").unwrap_err();
    assert!(
        unknown.to_string().contains("Unsupported option foo_bar"),
        "{unknown}"
    );
    // Interval options, catalog-wide: empty or an interval, as official
    // checks them ("delete_older_than is not a valid interval value.").
    for (key, value, ok) in [
        ("expire_older_than", "2 days ago", true),
        ("expire_older_than", "", true),
        ("delete_older_than", "garbage", false),
    ] {
        assert_eq!(
            lake.writer.set_global_setting(key, value).is_ok(),
            ok,
            "{key} = {value:?}"
        );
    }
}

/// A write that drops a column the table's sort order references is refused
/// with official's error ("Cannot drop column "v" - the table is sorted by
/// this column. ..."), the key matched ignoring case, as official refuses
/// `ALTER TABLE ... DROP COLUMN` (checked with the DuckDB CLI on `SET SORTED
/// BY (upper(V))`); a key of another dialect allows the drop. Guards against
/// a Replace leaving a sort key naming no column, which fails every later
/// insert.
pub(crate) async fn sort_key_columns_cannot_be_dropped(make: MakeLake) {
    let lake = make(Some(0)).await;
    create(&lake, "t").await;
    let table_id = lake.table_id("t");
    let key = SortField::column(0, "V", SortDirection::Asc, NullOrder::NullsLast);
    lake.writer
        .set_sort_spec(table_id, std::slice::from_ref(&key))
        .unwrap();
    let error = create_with(&lake, "t", &[("id", DataType::Int64)]).unwrap_err();
    assert!(
        error.contains(
            "Cannot drop column \"v\" - the table is sorted by this column. Reset or change \
             the sort order on this table in order to drop this column"
        ),
        "{error}"
    );
    let mut parquet = DuckLakeWriteOptions::default().with_data_inlining_row_limit(0);
    parquet.sort_on_insert = Some(false);
    let ids_only = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, true)])),
        vec![Arc::new(Int64Array::from(vec![1]))],
    )
    .unwrap();
    let error = lake
        .table_writer()
        .with_options(&parquet)
        .write_table("main", "t", std::slice::from_ref(&ids_only))
        .await
        .unwrap_err()
        .to_string();
    assert!(error.contains("Cannot drop column \"v\""), "{error}");
    assert_eq!(lake.data_files("t"), 0);
    create_with(&lake, "t", &[("v", DataType::Utf8)]).unwrap();
    let foreign = SortField {
        dialect: "spark".to_string(),
        ..key
    };
    lake.writer.set_sort_spec(table_id, &[foreign]).unwrap();
    create_with(&lake, "t", &[("id", DataType::Int64)]).unwrap();
}

/// A table with a reserved column name is created under a schema-scoped `0`,
/// opted out with a table-scoped `0`, and the schema's inlining re-enabled, as
/// official allows; enabling inlining over it without the table's `0` is
/// refused.
pub(crate) async fn reserved_names_under_scoped_limits(make: MakeLake) {
    let lake = make(None).await;
    create(&lake, "t").await;
    let schema_id = lake.schema_id();
    lake.writer
        .set_schema_setting(schema_id, "data_inlining_row_limit", "0")
        .unwrap();
    create_with(
        &lake,
        "r",
        &[("row_id", DataType::Int64), ("v", DataType::Utf8)],
    )
    .unwrap();
    let refused = lake
        .writer
        .set_schema_setting(schema_id, "data_inlining_row_limit", "5");
    assert!(
        refused
            .as_ref()
            .is_err_and(|e| e.to_string().contains("Cannot enable data inlining")),
        "{refused:?}"
    );
    lake.writer
        .set_table_setting(lake.table_id("r"), "data_inlining_row_limit", "0")
        .unwrap();
    lake.writer
        .set_schema_setting(schema_id, "data_inlining_row_limit", "5")
        .unwrap();
    lake.sql("INSERT INTO lake.main.r VALUES (1, 'a')")
        .await
        .unwrap();
    assert_eq!(lake.data_files("r"), 1, "the table's 0 keeps it in Parquet");
    if lake.inlines {
        lake.sql("INSERT INTO lake.main.t VALUES (1, 'a')")
            .await
            .unwrap();
        assert_eq!(lake.inlined_rows("t"), 1, "the schema's 5 inlines");
    }
}

/// A `DuckLakeTableWriter` honours a stored limit over its own option, as
/// official's stored settings outrank its attach option.
pub(crate) async fn stored_limit_outranks_writer_options(make: MakeLake) {
    let lake = make(None).await;
    create(&lake, "t").await;
    lake.writer
        .set_table_setting(lake.table_id("t"), "data_inlining_row_limit", "0")
        .unwrap();
    let files = lake
        .table_writer()
        .with_options(&DuckLakeWriteOptions::default().with_data_inlining_row_limit(100))
        .append_table("main", "t", &[batch(1..=3, "a")])
        .await
        .unwrap()
        .files_written;
    assert_eq!(files, 1);
}

/// Run `statement` (over a table named `t`) against `main.t` opened outside
/// the catalog's write options: read through a read-only catalog, then made
/// writable with [`DuckLakeTable::with_writer`].
async fn direct_sql(lake: &Lake, statement: &str) {
    let ctx = SessionContext::new();
    let catalog = DuckLakeCatalog::with_snapshot(Arc::clone(&lake.provider), lake.head()).unwrap();
    ctx.register_catalog("ro", Arc::new(catalog));
    let provider = ctx
        .catalog("ro")
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
        .clone()
        .with_writer("main".to_string(), Arc::clone(&lake.writer));
    ctx.register_table("t", Arc::new(table)).unwrap();
    ctx.sql(statement).await.unwrap().collect().await.unwrap();
}

/// SQL `DELETE` and `UPDATE` follow the table's scoped limit, whether the
/// table comes from the writable catalog or is opened directly with a writer:
/// with no stored setting and a writer without a limit, official's default of
/// `10` inlines a one-row deletion and a one-row update (official records
/// `inlined_delete` and `inlined_insert`, no delete file); a stored `0`
/// inlines nothing. Guards against a DELETE that reads only the catalog's
/// resolved options (and so writes delete files under the default) and an
/// UPDATE that ignores the stored setting.
pub(crate) async fn sql_mutations_follow_the_scoped_limit(make: MakeLake) {
    for direct in [false, true] {
        for stored_zero in [false, true] {
            let case = format!("direct {direct}, stored 0 {stored_zero}");
            let lake = make(None).await;
            create(&lake, "t").await;
            lake.append_parquet("t", &batch(1..=30, "a")).await;
            if stored_zero {
                lake.writer
                    .set_table_setting(lake.table_id("t"), "data_inlining_row_limit", "0")
                    .unwrap();
            }
            let run = async |statement: &str| {
                if direct {
                    direct_sql(&lake, &statement.replace("{t}", "t")).await;
                } else {
                    lake.sql(&statement.replace("{t}", "lake.main.t"))
                        .await
                        .unwrap();
                }
            };
            run("DELETE FROM {t} WHERE id = 3").await;
            if stored_zero {
                assert_eq!(lake.inlined_deletes("t"), 0, "{case}");
                assert_eq!(lake.delete_files("t"), 1, "{case}");
            } else {
                assert_eq!(lake.inlined_deletes("t"), 1, "{case}");
                assert_eq!(lake.delete_files("t"), 0, "{case}");
            }
            run("UPDATE {t} SET v = 'u' WHERE id = 4").await;
            if stored_zero {
                assert_eq!(lake.inlined_rows("t"), 0, "{case}");
                assert_eq!(lake.inlined_deletes("t"), 0, "{case}");
            } else {
                assert_eq!(lake.inlined_rows("t"), 1, "{case}");
                assert_eq!(lake.inlined_deletes("t"), 2, "{case}");
                assert_eq!(lake.delete_files("t"), 0, "{case}");
            }
            let rows = values(&lake.rows("t").await);
            assert_eq!(rows.len(), 29, "{case}");
            assert!(!rows.contains_key(&3), "{case}");
            assert_eq!(rows[&4], vec!["u".to_string()], "{case}");
        }
    }
}

// ---------------------------------------------------------------------------
// D2: streaming sessions and small deletes inline
// ---------------------------------------------------------------------------

/// A write whose batch has an Arrow `Null`-typed column (all NULL) stores it
/// under the column's catalog type: inlined under the default limit, and read
/// back as NULL, through a one-shot write and a streaming session alike.
/// Guards against an inlined insert that binds such a column by its Arrow type
/// instead of the inlined table's column type.
pub(crate) async fn null_typed_columns_inline(make: MakeLake) {
    let lake = make(None).await;
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, true),
        Field::new("x", DataType::Null, true),
    ]));
    let rows = |id: i64| {
        RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int64Array::from(vec![id])), Arc::new(arrow::array::NullArray::new(1))],
        )
        .unwrap()
    };
    let writer = lake.table_writer();
    writer.append_table("main", "t", &[rows(1)]).await.unwrap();
    let mut session = writer
        .begin_write("main", "t", &schema, WriteMode::Append)
        .unwrap();
    session.write_batch(&rows(2)).unwrap();
    session.finish().await.unwrap();
    if lake.inlines {
        assert_eq!(lake.inlined_rows("t"), 2);
        assert_eq!(lake.data_files("t"), 0);
    }
    let ctx = SessionContext::new();
    let catalog = DuckLakeCatalog::with_snapshot(Arc::clone(&lake.provider), lake.head()).unwrap();
    ctx.register_catalog("lake", Arc::new(catalog));
    let batches = ctx
        .sql("SELECT id, x IS NULL FROM lake.main.t ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(
        lines(&batches, false),
        vec!["1,true".to_string(), "2,true".to_string()]
    );
}

/// An Arrow `Null`-typed batch column written to a NOT NULL column is refused,
/// as official refuses a NULL there, by a one-shot write and a streaming
/// session alike, and nothing is stored. Guards against a validator that
/// counted only physical nulls, which a `Null` array does not have.
pub(crate) async fn null_typed_column_refused_for_not_null(make: MakeLake) {
    let lake = make(None).await;
    let columns = [
        datafusion_ducklake::metadata_writer::ColumnDef::from_arrow("id", &DataType::Int64, true)
            .unwrap(),
        datafusion_ducklake::metadata_writer::ColumnDef::from_arrow("x", &DataType::Utf8, false)
            .unwrap(),
    ];
    let setup = lake
        .writer
        .begin_write_transaction("main", "t", &columns, WriteMode::Replace)
        .unwrap();
    lake.writer
        .publish_snapshot(
            setup.table_id,
            "main",
            "t",
            setup.snapshot_id,
            WriteMode::Replace,
            setup.base_snapshot_id,
            &columns,
            &setup.field_ids,
        )
        .unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, true),
        Field::new("x", DataType::Null, true),
    ]));
    let rows = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int64Array::from(vec![1])), Arc::new(arrow::array::NullArray::new(1))],
    )
    .unwrap();
    let writer = lake.table_writer();
    let error = writer
        .append_table("main", "t", std::slice::from_ref(&rows))
        .await
        .unwrap_err()
        .to_string();
    assert!(error.contains("NOT NULL constraint failed: x"), "{error}");
    let refused = match writer.begin_write("main", "t", &schema, WriteMode::Append) {
        Ok(mut session) => match session.write_batch(&rows) {
            Err(error) => Some(error),
            Ok(()) => session.finish().await.err(),
        },
        Err(error) => Some(error),
    };
    let error = refused.expect("the session refuses the NULLs").to_string();
    assert!(error.contains("NOT NULL constraint failed: x"), "{error}");
    assert!(lake.rows("t").await.is_empty());
}

/// A streaming session on a partitioned table inlines a write of at most the
/// limit's rows, as official's insert into a partitioned table does, and
/// writes a larger one as one file per partition, recording each file's
/// value. Guards against a partitioned session that never inlines, or one
/// that inlines past the limit.
pub(crate) async fn partitioned_session_inlines_small_writes(make: MakeLake) {
    let lake = make(Some(5)).await;
    create(&lake, "t").await;
    lake.writer
        .set_partition_spec(
            lake.table_id("t"),
            &[(
                "v".to_string(),
                datafusion_ducklake::PartitionTransform::Identity,
            )],
        )
        .unwrap();
    let rows = |ids: std::ops::RangeInclusive<i64>| {
        let ids: Vec<i64> = ids.collect();
        let values: Vec<&str> = ids
            .iter()
            .map(|id| {
                if id % 2 == 0 {
                    "a"
                } else {
                    "b"
                }
            })
            .collect();
        RecordBatch::try_new(
            schema(),
            vec![Arc::new(Int64Array::from(ids)), Arc::new(StringArray::from(values))],
        )
        .unwrap()
    };
    let writer = lake.table_writer();
    let mut session = writer
        .begin_write("main", "t", &schema(), WriteMode::Append)
        .unwrap();
    session.write_batch(&rows(1..=3)).unwrap();
    assert!(session.is_inlining());
    let result = session.finish().await.unwrap();
    assert_eq!((result.files_written, result.records_written), (0, 3));
    assert_eq!(lake.inlined_rows("t"), 3);
    assert_eq!(lake.data_files("t"), 0);

    let mut session = writer
        .begin_write("main", "t", &schema(), WriteMode::Append)
        .unwrap();
    session.write_batch(&rows(10..=16)).unwrap();
    let result = session.finish().await.unwrap();
    assert_eq!((result.files_written, result.records_written), (2, 7));
    let files = lake
        .provider
        .get_table_files_for_select(lake.table_id("t"), lake.head())
        .unwrap();
    let mut counts: Vec<i64> = files
        .iter()
        .map(|file| file.max_row_count.unwrap_or(0))
        .collect();
    counts.sort_unstable();
    assert_eq!(counts, vec![3, 4], "one file per partition");
    assert_eq!(lake.inlined_rows("t"), 3);
    assert_eq!(values(&lake.rows("t").await).len(), 10);
}

/// A streaming session of at most the limit's rows commits them inlined, one
/// past it writes every row to Parquet, and a single-file session inlines too.
/// Guards against streaming writes never inlining.
pub(crate) async fn sessions_inline_small_writes(make: MakeLake) {
    let lake = make(Some(5)).await;
    create(&lake, "t").await;
    let writer = lake.table_writer();
    let mut session = writer
        .begin_write("main", "t", &schema(), WriteMode::Append)
        .unwrap();
    session.write_batch(&batch([1, 2], "a")).unwrap();
    session.write_batch(&batch([3, 4, 5], "a")).unwrap();
    assert!(session.is_inlining());
    let result = session.finish().await.unwrap();
    assert_eq!((result.files_written, result.records_written), (0, 5));
    assert_eq!(lake.inlined_rows("t"), 5);
    assert_eq!(lake.data_files("t"), 0);

    let mut session = writer
        .begin_write("main", "t", &schema(), WriteMode::Append)
        .unwrap();
    session.write_batch(&batch([10, 11, 12], "b")).unwrap();
    session.write_batch(&batch([13, 14, 15], "b")).unwrap();
    assert!(!session.is_inlining(), "past the limit");
    let result = session.finish().await.unwrap();
    assert_eq!((result.files_written, result.records_written), (1, 6));
    assert_eq!(lake.inlined_rows("t"), 5);

    let mut single = writer
        .begin_write_single_file("main", "t", &schema(), WriteMode::Append)
        .unwrap();
    single.write_batch(&batch([20], "c")).unwrap();
    assert_eq!(single.finish().await.unwrap().files_written, 0);
    assert_eq!(lake.inlined_rows("t"), 6);
    let rows = lake.rows("t").await;
    assert_eq!(
        values(&rows).keys().copied().collect::<Vec<_>>(),
        vec![1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 20]
    );

    // A replace of at most the limit's rows replaces everything with inlined
    // rows.
    let mut replace = writer
        .begin_write("main", "t", &schema(), WriteMode::Replace)
        .unwrap();
    replace.write_batch(&batch([30, 31], "r")).unwrap();
    replace.finish().await.unwrap();
    assert_eq!(
        values(&lake.rows("t").await)
            .keys()
            .copied()
            .collect::<Vec<_>>(),
        vec![30, 31]
    );
}

/// `finish_with_deletes` commits a small session's rows inlined together with
/// its positional deletes, in one snapshot.
pub(crate) async fn session_with_deletes_inlines(make: MakeLake) {
    let lake = make(Some(5)).await;
    create(&lake, "t").await;
    lake.append_parquet("t", &batch(1..=20, "a")).await;
    let file = lake
        .provider
        .get_table_files_for_select(lake.table_id("t"), lake.head())
        .unwrap()
        .remove(0);
    let writer = lake.table_writer();
    let delete = writer
        .write_delete_file("main", "t", &file.file.path, &[0, 1, 2])
        .await
        .unwrap();
    let head = lake.head();
    let mut session = writer
        .begin_write("main", "t", &schema(), WriteMode::Append)
        .unwrap();
    session.write_batch(&batch([100, 101], "n")).unwrap();
    let result = session
        .finish_with_deletes(&[datafusion_ducklake::DeleteFileEntry {
            data_file_id: file.data_file_id,
            expected_prev_delete_file: None,
            delete,
        }])
        .await
        .unwrap();
    assert_eq!(result.files_written, 0);
    assert_eq!(lake.head(), head + 1, "one snapshot");
    assert_eq!(lake.inlined_rows("t"), 2);
    let ids: Vec<i64> = values(&lake.rows("t").await).keys().copied().collect();
    assert_eq!(ids, (4..=20).chain([100, 101]).collect::<Vec<_>>());
}

/// `delete_positions` records at most the limit's positions of a file inlined
/// and more in a delete file, and `delete_rowids` ends an inlined row; both as
/// official's `DELETE` records them.
pub(crate) async fn small_deletes_inline(make: MakeLake) {
    let lake = make(Some(5)).await;
    create(&lake, "t").await;
    lake.append_parquet("t", &batch(1..=20, "a")).await;
    lake.append("t", &batch([100, 101, 102], "i")).await;
    let file = lake
        .provider
        .get_table_files_for_select(lake.table_id("t"), lake.head())
        .unwrap()
        .remove(0);
    let (ctx, table) = lake.table("t").await;
    let result = table
        .delete_positions(&ctx.state(), vec![(file.data_file_id, vec![0, 1])])
        .await
        .unwrap();
    assert_eq!(result.rows_deleted, 2);
    assert_eq!(lake.inlined_deletes("t"), 2);
    assert_eq!(lake.delete_files("t"), 0);
    let (ctx, table) = lake.table("t").await;
    // Already-deleted positions are skipped.
    let result = table
        .delete_positions(&ctx.state(), vec![(file.data_file_id, (0..9).collect())])
        .await
        .unwrap();
    assert_eq!(result.rows_deleted, 7);
    assert_eq!(
        lake.delete_files("t"),
        1,
        "more than the limit: a delete file"
    );
    let rows = lake.rows("t").await;
    let inlined_rowid = row_ids(&rows)[&101];
    let (ctx, table) = lake.table("t").await;
    let result = table
        .delete_rowids(&ctx.state(), &[inlined_rowid, 9_999_999])
        .await
        .unwrap();
    assert_eq!(result.rows_deleted, 1);
    let ids: Vec<i64> = values(&lake.rows("t").await).keys().copied().collect();
    assert_eq!(ids, (10..=20).chain([100, 102]).collect::<Vec<_>>());
    let (ctx, table) = lake.table("t").await;
    let refused = table
        .delete_positions(&ctx.state(), vec![(file.data_file_id, vec![500])])
        .await;
    assert!(matches!(refused, Err(DuckLakeError::InvalidConfig(_))));
}

// ---------------------------------------------------------------------------
// D3: keyed mutations
// ---------------------------------------------------------------------------

/// Parquet rows `1..=30` (two files) and inlined rows `100..=103`.
async fn keyed_table(make: &MakeLake) -> Lake {
    let lake = make(Some(5)).await;
    create(&lake, "t").await;
    lake.append_parquet("t", &batch(1..=15, "a")).await;
    lake.append_parquet("t", &batch(16..=30, "a")).await;
    lake.append("t", &batch(100..=103, "i")).await;
    assert_eq!(lake.inlined_rows("t"), 4);
    lake
}

/// A keyed delete removes a matching inlined row, which a delete built from
/// the data files alone leaves behind.
pub(crate) async fn keyed_delete_reaches_inlined_rows(make: MakeLake) {
    let lake = keyed_table(&make).await;
    let (ctx, table) = lake.table("t").await;
    // The file-only path: list the files that may hold the key, resolve
    // positions, and find nothing for an inlined row.
    let predicate: Arc<dyn datafusion::physical_expr::PhysicalExpr> =
        Arc::new(datafusion::physical_expr::expressions::BinaryExpr::new(
            Arc::new(datafusion::physical_expr::expressions::Column::new("id", 0)),
            datafusion::logical_expr::Operator::Eq,
            Arc::new(datafusion::physical_expr::expressions::Literal::new(
                datafusion::common::ScalarValue::Int64(Some(101)),
            )),
        ));
    let mut found = 0;
    for file in table.files_matching(&predicate).unwrap() {
        found += table
            .resolve_positions(&ctx.state(), &file.file, Arc::clone(&predicate))
            .await
            .unwrap()
            .len();
    }
    assert_eq!(found, 0, "the file-only path cannot see the inlined row");

    let result = table
        .keyed_mutation(&["id"])
        .delete_by_keys(
            &ctx.state(),
            vec![keys([Some(101), Some(5), Some(77), None])],
        )
        .await
        .unwrap();
    assert_eq!(result.rows_deleted, 2);
    assert!(result.snapshot_id.is_some());
    let ids = values(&lake.rows("t").await);
    assert!(!ids.contains_key(&101) && !ids.contains_key(&5));
    assert_eq!(ids.len(), 32);
    assert_eq!(lake.inlined_deletes("t"), 1, "one Parquet row: inlined");

    // Nothing to delete commits nothing.
    let head = lake.head();
    let (ctx, table) = lake.table("t").await;
    let result = table
        .keyed_mutation(&["id"])
        .delete_by_keys(&ctx.state(), vec![keys([Some(101)])])
        .await
        .unwrap();
    assert_eq!((result.rows_deleted, result.snapshot_id), (0, None));
    assert_eq!(lake.head(), head);
}

/// A keyed upsert replaces a matching inlined row instead of adding a second
/// copy, and appends the rest, in one snapshot.
pub(crate) async fn keyed_upsert_replaces_inlined_rows(make: MakeLake) {
    let lake = keyed_table(&make).await;
    let head = lake.head();
    let (ctx, table) = lake.table("t").await;
    let result = table
        .keyed_mutation(&["id"])
        .upsert_by_keys(&ctx.state(), vec![batch([102, 7, 200], "new")])
        .await
        .unwrap();
    assert_eq!((result.rows_deleted, result.rows_inserted), (2, 3));
    assert_eq!(lake.head(), head + 1, "one snapshot");
    let rows = values(&lake.rows("t").await);
    assert_eq!(
        rows[&102],
        vec!["new102".to_string()],
        "replaced, not duplicated"
    );
    assert_eq!(rows[&7], vec!["new7".to_string()]);
    assert_eq!(rows[&200], vec!["new200".to_string()]);
    assert_eq!(rows.len(), 35);
    assert_eq!(lake.inlined_rows("t"), 6, "3 new rows inlined, 1 ended");

    // More rows than the limit go to Parquet.
    let (ctx, table) = lake.table("t").await;
    let files = lake.data_files("t");
    table
        .keyed_mutation(&["id"])
        .upsert_by_keys(&ctx.state(), vec![batch(25..=40, "big")])
        .await
        .unwrap();
    assert_eq!(lake.data_files("t"), files + 1);
    let rows = values(&lake.rows("t").await);
    assert!((25..=40).all(|id| rows[&id] == vec![format!("big{id}")]));
    assert_eq!(rows.len(), 45);
}

/// A keyed update gives matched rows, Parquet and inlined, the staged values
/// and keeps their row ids; a NULL key and an unmatched key change nothing,
/// and two staged rows with one key are refused.
pub(crate) async fn keyed_update_keeps_row_ids(make: MakeLake) {
    let lake = keyed_table(&make).await;
    let before = row_ids(&lake.rows("t").await);
    let (ctx, table) = lake.table("t").await;
    let staged = RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(Int64Array::from(vec![Some(6), Some(103), None, Some(999)])),
            Arc::new(StringArray::from(vec!["u6", "u103", "unull", "u999"])),
        ],
    )
    .unwrap();
    let result = table
        .keyed_mutation(&["id"])
        .update_by_keys(&ctx.state(), vec![staged])
        .await
        .unwrap();
    assert_eq!(result.rows_updated, 2);
    let rows = lake.rows("t").await;
    let after = row_ids(&rows);
    assert_eq!(after, before, "row ids are kept");
    let rows = values(&rows);
    assert_eq!(rows[&6], vec!["u6".to_string()]);
    assert_eq!(rows[&103], vec!["u103".to_string()]);
    assert!(!rows.contains_key(&999));

    // Staged rows sharing a key: the last one's values win, as a `MERGE`
    // with a small source updates in source order.
    let (ctx, table) = lake.table("t").await;
    let duplicated = RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(Int64Array::from(vec![1, 1])),
            Arc::new(StringArray::from(vec!["first", "last"])),
        ],
    )
    .unwrap();
    let updated = table
        .keyed_mutation(&["id"])
        .update_by_keys(&ctx.state(), vec![duplicated])
        .await
        .unwrap();
    assert_eq!(updated.rows_updated, 1);
    assert_eq!(values(&lake.rows("t").await)[&1], vec!["last".to_string()]);
    let (ctx, table) = lake.table("t").await;
    let refused = table
        .keyed_mutation(&["nope"])
        .delete_by_keys(&ctx.state(), vec![keys([Some(1)])])
        .await;
    assert!(matches!(refused, Err(DuckLakeError::InvalidConfig(_))));
}

/// Keys of several columns match on every column; a partitioned table's
/// upsert writes its rows into their partitions.
pub(crate) async fn keyed_composite_and_partitioned(make: MakeLake) {
    let lake = make(Some(3)).await;
    create_with(
        &lake,
        "p",
        &[("region", DataType::Utf8), ("id", DataType::Int64), ("v", DataType::Utf8)],
    )
    .unwrap();
    lake.sql("ALTER TABLE lake.main.p SET PARTITIONED BY (region)")
        .await
        .unwrap();
    let rows = |regions: &[&str], ids: &[i64], prefix: &str| {
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("region", DataType::Utf8, true),
                Field::new("id", DataType::Int64, true),
                Field::new("v", DataType::Utf8, true),
            ])),
            vec![
                Arc::new(StringArray::from(regions.to_vec())),
                Arc::new(Int64Array::from(ids.to_vec())),
                Arc::new(StringArray::from(
                    ids.iter()
                        .map(|id| format!("{prefix}{id}"))
                        .collect::<Vec<_>>(),
                )),
            ],
        )
        .unwrap()
    };
    lake.append_parquet(
        "p",
        &rows(&["eu", "eu", "us", "us", "us"], &[1, 2, 1, 2, 3], "a"),
    )
    .await;
    let (ctx, table) = lake.table("p").await;
    let result = table
        .keyed_mutation(&["region", "id"])
        .upsert_by_keys(
            &ctx.state(),
            vec![rows(
                &["eu", "us", "ap", "ap", "ap", "eu"],
                &[2, 9, 1, 2, 3, 3],
                "n",
            )],
        )
        .await
        .unwrap();
    assert_eq!((result.rows_deleted, result.rows_inserted), (1, 6));
    let ctx = SessionContext::new();
    ctx.register_catalog("lake", Arc::new(lake.catalog()));
    let got = lines(
        &ctx.sql("SELECT region, id, v FROM lake.main.p ORDER BY region, id")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap(),
        false,
    );
    assert_eq!(
        got,
        vec![
            "ap,1,n1", "ap,2,n2", "ap,3,n3", "eu,1,a1", "eu,2,n2", "eu,3,n3", "us,1,a1", "us,2,a2",
            "us,3,a3", "us,9,n9"
        ]
    );
    let (ctx, table) = lake.table("p").await;
    let result = table
        .keyed_mutation(&["region", "id"])
        .delete_by_keys(&ctx.state(), vec![rows(&["us", "eu"], &[1, 1], "x")])
        .await
        .unwrap();
    assert_eq!(result.rows_deleted, 2);
}

// ---------------------------------------------------------------------------
// D5: change detection
// ---------------------------------------------------------------------------

/// Inlined inserts and deletes count as row changes though no data file
/// changes; a flush changes only where rows live. Guards against change
/// detection built from the data and delete files.
pub(crate) async fn change_detection_sees_inlined_changes(make: MakeLake) {
    let lake = make(Some(5)).await;
    create(&lake, "t").await;
    lake.append_parquet("t", &batch(1..=20, "a")).await;
    let table_id = lake.table_id("t");
    let files = |lake: &Lake| {
        lake.provider
            .get_table_files_for_select(table_id, lake.head())
            .unwrap()
            .into_iter()
            .map(|file| (file.data_file_id, file.delete_file_id))
            .collect::<Vec<_>>()
    };
    let before = lake.head();
    let listed = files(&lake);
    lake.append("t", &batch([100, 101], "i")).await;
    let changes = data_changes(&lake.provider, table_id, before, lake.head()).await;
    if lake.inlines {
        assert_eq!(files(&lake), listed, "the file listing does not change");
        assert!(changes.inlined_inserted);
    }
    assert!(changes.rows_changed());

    if lake.inlines {
        let before = lake.head();
        lake.sql("DELETE FROM lake.main.t WHERE id IN (3, 100)")
            .await
            .unwrap();
        let delete_snapshot = lake.head();
        let changes = data_changes(&lake.provider, table_id, before, lake.head()).await;
        assert!(changes.rows_changed());
        assert!(changes.inlined_deleted && !changes.deleted);
        let before = lake.head();
        let ctx = SessionContext::new();
        lake.catalog()
            .flush_inlined_data(&ctx.state(), None, None)
            .await
            .unwrap();
        // The flush writes a delete file for row 3's inlined deletion and
        // records `deleted_from_table` beside `inline_flush`, as official's
        // does, and as a `DELETE` committed with a flush does: a deletion.
        let changes = data_changes(&lake.provider, table_id, before, lake.head()).await;
        assert!(changes.flushed && changes.deleted, "{changes:?}");
        assert!(changes.rows_changed(), "{changes:?}");
        let flushed_deletions = lake.head();
        assert!(flushed_deletions > delete_snapshot);
        // A flush of inlined rows alone records `inline_flush` alone: it
        // changes only where rows live.
        lake.append("t", &batch([200], "j")).await;
        let inserted = lake.head();
        lake.catalog()
            .flush_inlined_data(&ctx.state(), None, None)
            .await
            .unwrap();
        assert_eq!(
            lake.provider
                .list_table_snapshot_changes(table_id, inserted, lake.head())
                .unwrap(),
            vec![(lake.head(), Some(format!("inline_flush:{table_id}")))]
        );
        let changes = data_changes(&lake.provider, table_id, inserted, lake.head()).await;
        assert!(changes.flushed && !changes.rows_changed(), "{changes:?}");
        assert!(changes.layout_changed());
        assert_eq!(
            last_rows_changed(&lake.provider, table_id, lake.head()).await,
            Some(inserted)
        );
        // A flush of inlined deletions alone records `deleted_from_table`
        // and nothing else, as official's does (checked with the DuckDB CLI),
        // as a `DELETE` does: the ledger cannot tell it from one, so it reads
        // as a deletion.
        lake.sql("DELETE FROM lake.main.t WHERE id = 5")
            .await
            .unwrap();
        assert_eq!(lake.inlined_rows("t"), 0);
        let before = lake.head();
        lake.catalog()
            .flush_inlined_data(&ctx.state(), None, None)
            .await
            .unwrap();
        assert_eq!(
            lake.provider
                .list_table_snapshot_changes(table_id, before, lake.head())
                .unwrap(),
            vec![(lake.head(), Some(format!("deleted_from_table:{table_id}")))]
        );
        let changes = data_changes(&lake.provider, table_id, before, lake.head()).await;
        assert!(changes.deleted && changes.rows_changed(), "{changes:?}");
    }
    // Another table's changes and an empty window report nothing.
    create(&lake, "u").await;
    lake.append("u", &batch([1], "x")).await;
    let head = lake.head();
    let changes = data_changes(&lake.provider, table_id, head - 1, head).await;
    assert!(!changes.any_changed());
    let changes = data_changes(&lake.provider, table_id, head, head).await;
    assert!(!changes.any_changed());
    // The ranged id listing change detection reads agrees with the full one.
    let all: Vec<i64> = lake
        .provider
        .list_snapshots()
        .unwrap()
        .into_iter()
        .map(|snapshot| snapshot.snapshot_id)
        .collect();
    let (from, through) = (all[1], all[all.len() - 2]);
    assert_eq!(
        lake.provider.snapshot_ids_between(from, through).unwrap(),
        all.iter()
            .copied()
            .filter(|id| (from..=through).contains(id))
            .collect::<Vec<_>>()
    );
}

// ---------------------------------------------------------------------------
// D6: rowid lookup
// ---------------------------------------------------------------------------

/// Rows are found by `rowid` in inlined rows, in a file whose rows take their
/// ids from position, and in files that embed their row ids: a flushed file,
/// written in the table's sort order, records a `row_id_start` its positions
/// do not follow. Guards against mapping `rowid` to `row_id_start + position`.
pub(crate) async fn rowid_lookup_everywhere(make: MakeLake) {
    let lake = make(Some(5)).await;
    create(&lake, "t").await;
    lake.append_parquet("t", &batch(1..=10, "a")).await;
    lake.sql("ALTER TABLE lake.main.t SET SORTED BY (v DESC)")
        .await
        .unwrap();
    for chunk in [[20, 21, 22], [23, 24, 25]] {
        lake.append("t", &batch(chunk, "z")).await;
    }
    lake.sql("UPDATE lake.main.t SET v = 'u' || id WHERE id <= 8")
        .await
        .unwrap();
    lake.append("t", &batch([30, 31], "i")).await;
    let ctx = SessionContext::new();
    let flushed_before = lake.data_files("t");
    if lake.inlines {
        // Flush the inlined rows (ids 20-25), but keep 30-31 inlined by
        // inserting them after: flush everything, then insert again.
        lake.catalog()
            .flush_inlined_data(&ctx.state(), None, None)
            .await
            .unwrap();
        assert!(lake.data_files("t") > flushed_before);
        lake.append("t", &batch([40, 41], "k")).await;
        assert_eq!(lake.inlined_rows("t"), 2);
    }
    let rows = lake.rows("t").await;
    let truth: BTreeMap<i64, String> = rows
        .iter()
        .map(|row| {
            let (rowid, rest) = row.split_once(',').unwrap();
            (rowid.parse().unwrap(), rest.to_string())
        })
        .collect();
    let (ctx, table) = lake.table("t").await;
    let state = ctx.state();
    let mut embedded = 0;
    let mut embedded_with_start = 0;
    for file in table.files().unwrap() {
        let layout = table.file_row_id_layout(&state, &file).await.unwrap();
        if layout == Some(RowIdLayout::Embedded) {
            embedded += 1;
            if file.row_id_start.is_some() {
                // A flush records its smallest row id, yet its rows are in the
                // table's sort order: `row_id_start + position` is not a rowid.
                embedded_with_start += 1;
            }
        }
    }
    assert!(
        embedded >= 1,
        "the update (and the flush) wrote embedded row ids"
    );
    if lake.inlines {
        assert!(
            embedded_with_start >= 1,
            "the flush recorded a row_id_start"
        );
    }
    let wanted: Vec<i64> = truth.keys().copied().chain([-1, 1_000_000]).collect();
    let located = table.locate_rowids(&state, &wanted).await.unwrap();
    assert_eq!(located.len(), truth.len(), "every live row, once");
    let ids: HashSet<i64> = located.iter().map(|row| row.row_id).collect();
    assert_eq!(ids, truth.keys().copied().collect::<HashSet<_>>());
    if lake.inlines {
        assert!(
            located
                .iter()
                .any(|row| matches!(row.location, RowLocation::Inlined { .. }))
        );
    }
    let read = table.read_rows_by_rowid(&state, &wanted).await.unwrap();
    let mut got: Vec<String> = lines(&read, true);
    got.sort_by_key(|line| line.split(',').next().unwrap().parse::<i64>().unwrap());
    assert_eq!(got, rows, "the rows read by rowid are the rows scanned");
    // A subset reads only those rows.
    let some: Vec<i64> = truth.keys().copied().step_by(3).collect();
    let read = table.read_rows_by_rowid(&state, &some).await.unwrap();
    assert_eq!(lines(&read, true).len(), some.len());
}

// ---------------------------------------------------------------------------
// D7: maintenance flushes inlined data
// ---------------------------------------------------------------------------

/// A merge flushes the table's inlined data when its policy says so, then
/// merges the flushed files; `never` leaves it, and the default leaves a few
/// fresh rows. A rewrite counts inlined deletions and retires a file they
/// empty, then finds no more work.
pub(crate) async fn maintenance_flushes_inlined_data(make: MakeLake) {
    let lake = make(Some(5)).await;
    create(&lake, "t").await;
    for start in [1, 4, 7, 10] {
        lake.append("t", &batch(start..start + 3, "a")).await;
    }
    assert_eq!(lake.inlined_rows("t"), 12);
    let rows = lake.rows("t").await;
    let merge = MergeOptions::default();

    let (ctx, table) = lake.table("t").await;
    let result = table
        .merge_adjacent_files(&ctx.state(), merge.clone())
        .await
        .unwrap();
    assert!(
        !result.did_work(),
        "the default policy leaves 12 fresh rows"
    );
    assert_eq!(lake.inlined_rows("t"), 12);

    let never = lake.catalog().with_write_options(
        DuckLakeWriteOptions::default().with_inlined_flush_policy(InlinedFlushPolicy::never()),
    );
    let ctx = SessionContext::new();
    DuckLakeCatalog::register(&ctx, "lake", never);
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
        .unwrap()
        .clone();
    assert!(
        !table
            .merge_adjacent_files(&ctx.state(), merge.clone())
            .await
            .unwrap()
            .did_work()
    );

    let always = lake.catalog().with_write_options(
        DuckLakeWriteOptions::default().with_inlined_flush_policy(InlinedFlushPolicy::always()),
    );
    let ctx = SessionContext::new();
    DuckLakeCatalog::register(&ctx, "lake", always);
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
        .unwrap()
        .clone();
    let result = table
        .merge_adjacent_files(&ctx.state(), merge.clone())
        .await
        .unwrap();
    assert!(
        result.did_work() && result.inlined_flush_snapshot.is_some(),
        "{result:?}"
    );
    assert_eq!(lake.inlined_rows("t"), 0);
    assert_eq!(lake.rows("t").await, rows, "a flush changes no row");

    // Inlined deletions count towards a rewrite.
    lake.append_parquet("t", &batch(50..=53, "p")).await;
    lake.sql("DELETE FROM lake.main.t WHERE id >= 50")
        .await
        .unwrap();
    assert_eq!(lake.inlined_deletes("t"), 4);
    let rows = lake.rows("t").await;
    let (ctx, table) = lake.table("t").await;
    let result = table
        .rewrite_data_files(
            &ctx.state(),
            RewriteOptions {
                delete_threshold: 0.5,
                data_file_ids: None,
            },
        )
        .await
        .unwrap();
    assert!(result.files_processed >= 1, "{result:?}");
    assert_eq!(lake.rows("t").await, rows);
    let (ctx, table) = lake.table("t").await;
    let again = table
        .rewrite_data_files(
            &ctx.state(),
            RewriteOptions {
                delete_threshold: 0.5,
                data_file_ids: None,
            },
        )
        .await
        .unwrap();
    assert!(!again.did_work(), "no work loop: {again:?}");
}

/// `SET SORTED BY` and `RESET SORTED BY`, a `RESET` of a table with no sort
/// order too, each commit a snapshot that bumps `schema_version` and records
/// `altered_table:<id>`, as official's do (checked with the DuckDB CLI); the
/// table's inlined rows stay readable and later rows inline beside them.
/// Guards against a sort change carrying `schema_version` forward and a no-op
/// `RESET` committing nothing, where official commits both.
pub(crate) async fn sort_changes_bump_schema_version(make: MakeLake) {
    let lake = make(Some(10)).await;
    create(&lake, "t").await;
    let table_id = lake.table_id("t");
    lake.append("t", &batch([1], "a")).await;
    let version = |snapshot: i64| {
        lake.provider
            .list_snapshots()
            .unwrap()
            .into_iter()
            .find(|row| row.snapshot_id == snapshot)
            .and_then(|row| row.schema_version)
            .unwrap()
    };
    for statement in [
        "ALTER TABLE lake.main.t SET SORTED BY (id)",
        "ALTER TABLE lake.main.t RESET SORTED BY",
        "ALTER TABLE lake.main.t RESET SORTED BY",
    ] {
        let before = lake.head();
        lake.sql(statement).await.unwrap();
        let after = lake.head();
        assert!(after > before, "{statement} commits a snapshot");
        assert_eq!(version(after), version(before) + 1, "{statement}");
        assert_eq!(
            lake.provider
                .list_table_snapshot_changes(table_id, before, after)
                .unwrap(),
            vec![(after, Some(format!("altered_table:{table_id}")))],
            "{statement}"
        );
    }
    lake.append("t", &batch([2], "a")).await;
    assert_eq!(lake.rows("t").await.len(), 2);
    if lake.inlines {
        assert_eq!(lake.inlined_rows("t"), 2);
    }
}

/// After sort-order changes, a SQLite catalog still holds one inlined data
/// table for `t`, as official's does: a sort change adds no
/// `ducklake_schema_versions` row, so no new inlined data table.
#[cfg(feature = "write-sqlite")]
#[tokio::test(flavor = "multi_thread")]
async fn sort_changes_keep_the_inlined_data_table() {
    let lake = sqlite()(Some(10)).await;
    create(&lake, "t").await;
    lake.append("t", &batch([1], "a")).await;
    lake.sql("ALTER TABLE lake.main.t SET SORTED BY (id)")
        .await
        .unwrap();
    lake.append("t", &batch([2], "a")).await;
    lake.sql("ALTER TABLE lake.main.t RESET SORTED BY")
        .await
        .unwrap();
    lake.append("t", &batch([3], "a")).await;
    let url = format!("sqlite:{}", lake._dir.path().join("lake.db").display());
    let pool = sqlx::SqlitePool::connect(&url).await.unwrap();
    let tables: Vec<(String, i64)> =
        sqlx::query_as("SELECT table_name, schema_version FROM ducklake_inlined_data_tables")
            .fetch_all(&pool)
            .await
            .unwrap();
    assert_eq!(tables.len(), 1, "{tables:?}");
    assert_eq!(lake.inlined_rows("t"), 3);
}

/// `merge_adjacent_files` on `main.t` of `lake` under write options carrying
/// `policy`.
async fn merge_under(
    lake: &Lake,
    policy: InlinedFlushPolicy,
) -> datafusion_ducklake::CompactionResult {
    merge_with(
        lake,
        DuckLakeWriteOptions::default().with_inlined_flush_policy(policy),
    )
    .await
}

/// `merge_adjacent_files` on `main.t` of `lake` under `options`.
async fn merge_with(
    lake: &Lake,
    options: DuckLakeWriteOptions,
) -> datafusion_ducklake::CompactionResult {
    let catalog = lake.catalog().with_write_options(options);
    let ctx = SessionContext::new();
    DuckLakeCatalog::register(&ctx, "lake", catalog);
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
        .unwrap()
        .clone();
    table
        .merge_adjacent_files(&ctx.state(), MergeOptions::default())
        .await
        .unwrap()
}

/// The flush an [`InlinedFlushPolicy`] runs before a compaction writes with
/// the compaction's write options, as its merged output does: here a row-group
/// cap and a codec. Guards against a flush that took only the catalog's
/// stored settings.
#[cfg(feature = "write-sqlite")]
#[tokio::test(flavor = "multi_thread")]
async fn policy_flush_uses_the_callers_write_options() {
    let lake = sqlite()(Some(10)).await;
    create(&lake, "t").await;
    lake.append("t", &batch(1..=6, "a")).await;
    assert_eq!(lake.inlined_rows("t"), 6);
    let mut options =
        DuckLakeWriteOptions::default().with_inlined_flush_policy(InlinedFlushPolicy::always());
    options.max_row_group_rows = Some(2);
    options.compression = Some(parquet::basic::Compression::GZIP(Default::default()));
    let merged = merge_with(&lake, options).await;
    assert!(merged.inlined_flush_snapshot.is_some(), "{merged:?}");
    assert_eq!(lake.inlined_rows("t"), 0);
    let files: Vec<_> = walk_files(&lake.data)
        .into_iter()
        .filter(|path| path.extension().is_some_and(|ext| ext == "parquet"))
        .collect();
    assert_eq!(files.len(), 1, "{files:?}");
    let reader =
        parquet::file::reader::SerializedFileReader::new(std::fs::File::open(&files[0]).unwrap())
            .unwrap();
    let metadata = parquet::file::reader::FileReader::metadata(&reader);
    assert_eq!(metadata.num_row_groups(), 3);
    assert!(matches!(
        metadata.row_group(0).column(0).compression(),
        parquet::basic::Compression::GZIP(_)
    ));
    assert_eq!(values(&lake.rows("t").await).len(), 6);
}

/// Every file under `dir`.
fn walk_files(dir: &std::path::Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                out.extend(walk_files(&path));
            } else {
                out.push(path);
            }
        }
    }
    out
}

/// A local store whose uploads fail once `allowed` have started.
struct FailingStore {
    inner: LocalFileSystem,
    allowed: usize,
    started: std::sync::atomic::AtomicUsize,
}

impl FailingStore {
    fn admit(&self) -> object_store::Result<()> {
        if self
            .started
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
            < self.allowed
        {
            Ok(())
        } else {
            Err(object_store::Error::Generic {
                store: "FailingStore",
                source: "injected upload failure".into(),
            })
        }
    }
}

impl std::fmt::Debug for FailingStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FailingStore")
    }
}

impl std::fmt::Display for FailingStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FailingStore")
    }
}

#[async_trait::async_trait]
impl object_store::ObjectStore for FailingStore {
    async fn put_opts(
        &self,
        location: &object_store::path::Path,
        payload: object_store::PutPayload,
        opts: object_store::PutOptions,
    ) -> object_store::Result<object_store::PutResult> {
        self.admit()?;
        self.inner.put_opts(location, payload, opts).await
    }

    async fn put_multipart_opts(
        &self,
        location: &object_store::path::Path,
        opts: object_store::PutMultipartOptions,
    ) -> object_store::Result<Box<dyn object_store::MultipartUpload>> {
        self.admit()?;
        self.inner.put_multipart_opts(location, opts).await
    }

    async fn get_opts(
        &self,
        location: &object_store::path::Path,
        options: object_store::GetOptions,
    ) -> object_store::Result<object_store::GetResult> {
        self.inner.get_opts(location, options).await
    }

    fn delete_stream(
        &self,
        locations: futures::stream::BoxStream<
            'static,
            object_store::Result<object_store::path::Path>,
        >,
    ) -> futures::stream::BoxStream<'static, object_store::Result<object_store::path::Path>> {
        self.inner.delete_stream(locations)
    }

    fn list(
        &self,
        prefix: Option<&object_store::path::Path>,
    ) -> futures::stream::BoxStream<'static, object_store::Result<object_store::ObjectMeta>> {
        self.inner.list(prefix)
    }

    async fn list_with_delimiter(
        &self,
        prefix: Option<&object_store::path::Path>,
    ) -> object_store::Result<object_store::ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }

    async fn copy_opts(
        &self,
        from: &object_store::path::Path,
        to: &object_store::path::Path,
        options: object_store::CopyOptions,
    ) -> object_store::Result<()> {
        self.inner.copy_opts(from, to, options).await
    }
}

/// An upsert whose delete file fails to upload, after its new rows' data
/// file and another file's delete file were uploaded, removes both: nothing
/// it wrote outlives the failure. Guards against orphans left by a failure
/// before the transaction takes the files over.
#[cfg(feature = "write-sqlite")]
#[tokio::test(flavor = "multi_thread")]
async fn failed_upsert_before_staging_removes_its_uploads() {
    let lake = sqlite()(Some(0)).await;
    create(&lake, "t").await;
    lake.append_parquet("t", &batch(1..=10, "a")).await;
    lake.append_parquet("t", &batch(11..=20, "b")).await;
    let before = walk_files(&lake.data);
    assert_eq!(before.len(), 2);
    let (ctx, table) = lake.table("t").await;
    ctx.runtime_env().register_object_store(
        datafusion::execution::object_store::ObjectStoreUrl::local_filesystem().as_ref(),
        Arc::new(FailingStore {
            inner: LocalFileSystem::new(),
            allowed: 2,
            started: std::sync::atomic::AtomicUsize::new(0),
        }),
    );
    let failed = table
        .keyed_mutation(&["id"])
        .with_concurrency(1)
        .upsert_by_keys(&ctx.state(), vec![batch([1, 11, 100], "u")])
        .await;
    assert!(failed.is_err(), "{failed:?}");
    let mut after = walk_files(&lake.data);
    after.sort();
    let mut before = before;
    before.sort();
    assert_eq!(after, before, "the failed upsert left no file behind");
    assert_eq!(lake.data_files("t"), 2);
}

/// The age trigger of an [`InlinedFlushPolicy`] measures an inlined insert's
/// age from its snapshot's time as a UTC instant on every backend: rows
/// inlined a moment ago are not an hour old, and are at least zero seconds
/// old. Guards against reading a `snapshot_time` printed with a UTC offset
/// (`-03`, `+05:30`) as unparseable, or a local wall time as UTC, which made
/// the trigger fire hours early or never.
pub(crate) async fn age_trigger_measures_utc_snapshot_times(make: MakeLake) {
    let lake = make(Some(5)).await;
    create(&lake, "t").await;
    lake.append("t", &batch(1..=3, "a")).await;
    assert_eq!(lake.inlined_rows("t"), 3);
    let age = |age| InlinedFlushPolicy::never().with_max_inlined_age(Some(age));
    let hour = merge_under(&lake, age(std::time::Duration::from_secs(3600))).await;
    assert!(
        !hour.did_work(),
        "rows inlined now are not an hour old: {hour:?}"
    );
    assert_eq!(lake.inlined_rows("t"), 3);
    let zero = merge_under(&lake, age(std::time::Duration::ZERO)).await;
    assert!(
        zero.inlined_flush_snapshot.is_some(),
        "rows inlined now are zero seconds old: {zero:?}"
    );
    assert_eq!(lake.inlined_rows("t"), 0);
}

/// [`age_trigger_measures_utc_snapshot_times`] on a multi-catalog
/// PostgreSQL store through a pool whose sessions run in a time zone west
/// (`-03`) and east (`+05:30`) of UTC; and on single-catalog PostgreSQL,
/// which inlines nothing, a snapshot resolved by time
/// (`DuckLakeCatalog::with_snapshot_at`) with `snapshot_time` as this crate
/// creates it (`TIMESTAMP`) and as official does (`TIMESTAMPTZ`, printed
/// with the session's offset).
#[cfg(feature = "write-postgres")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
#[tokio::test(flavor = "multi_thread")]
async fn age_trigger_on_non_utc_postgres_sessions() {
    use testcontainers::runners::AsyncRunner;
    let container = testcontainers_modules::postgres::Postgres::default()
        .start()
        .await
        .unwrap();
    let port = container.get_host_port_ipv4(5432).await.unwrap();
    let url = format!("postgresql://postgres:postgres@127.0.0.1:{port}/postgres");
    let admin = sqlx::PgPool::connect(&url).await.unwrap();
    datafusion_ducklake::initialize_multicatalog_schema(&admin)
        .await
        .unwrap();
    // sqlx starts every session in UTC; a caller's pool may not.
    let pool_in = async |zone: &'static str, url: &str| {
        let pool = sqlx::postgres::PgPoolOptions::new()
            .after_connect(move |connection, _| {
                Box::pin(async move {
                    sqlx::query(sqlx::AssertSqlSafe(format!("SET TIME ZONE '{zone}'")))
                        .execute(connection)
                        .await
                        .map(|_| ())
                })
            })
            .connect(url)
            .await
            .unwrap();
        let session: String = sqlx::query_scalar("SHOW timezone")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(session, zone);
        pool
    };
    for (index, zone) in ["America/Sao_Paulo", "Asia/Kolkata"]
        .into_iter()
        .enumerate()
    {
        age_trigger_measures_utc_snapshot_times(multicatalog(pool_in(zone, &url).await)).await;

        for column_type in ["TIMESTAMP", "TIMESTAMPTZ"] {
            let database = format!("tz{index}_{}", column_type.to_ascii_lowercase());
            sqlx::query(sqlx::AssertSqlSafe(format!("CREATE DATABASE {database}")))
                .execute(&admin)
                .await
                .unwrap();
            let db_url = format!("postgresql://postgres:postgres@127.0.0.1:{port}/{database}");
            datafusion_ducklake::PostgresSingleCatalogMetadataWriter::new_with_init(&db_url)
                .await
                .unwrap();
            let pool = pool_in(zone, &db_url).await;
            sqlx::query(sqlx::AssertSqlSafe(format!(
                "ALTER TABLE ducklake_snapshot ALTER COLUMN snapshot_time TYPE {column_type}"
            )))
            .execute(&pool)
            .await
            .unwrap();
            let (dir, data) = scratch();
            let writer =
                datafusion_ducklake::PostgresSingleCatalogMetadataWriter::from_pool(pool.clone());
            writer.set_data_path(data.to_str().unwrap()).unwrap();
            let lake = Lake {
                writer: Arc::new(writer),
                provider: Arc::new(datafusion_ducklake::PostgresMetadataProvider::from_pool(
                    pool,
                )),
                data,
                inlines: false,
                _dir: dir,
            };
            create(&lake, "t").await;
            let first = lake.head();
            let between = chrono::Utc::now();
            tokio::time::sleep(std::time::Duration::from_millis(1100)).await;
            create(&lake, "u").await;
            assert!(lake.head() > first);
            // The catalog at `first` holds `t` and not yet `u`.
            let catalog =
                DuckLakeCatalog::with_snapshot_at(Arc::clone(&lake.provider), between).unwrap();
            let mut tables = datafusion::catalog::CatalogProvider::schema(&catalog, "main")
                .unwrap()
                .table_names();
            tables.sort();
            assert_eq!(
                tables,
                vec!["t".to_string()],
                "{zone} {column_type}: the snapshot at or before {between}"
            );
        }
    }
}

/// The age trigger does not count an inlined deletion a flush already
/// moved: after a flush of inlined deletions alone (which records
/// `deleted_from_table` and no `inline_flush`), rows inlined a moment ago
/// are not an hour old though the flushed deletion is two hours old. Guards
/// against measuring the age from a deletion that is no longer inlined.
#[cfg(feature = "write-sqlite")]
#[tokio::test(flavor = "multi_thread")]
async fn age_trigger_skips_flushed_deletions() {
    let lake = sqlite()(Some(5)).await;
    create(&lake, "t").await;
    lake.append_parquet("t", &batch(1..=20, "a")).await;
    lake.sql("DELETE FROM lake.main.t WHERE id = 3")
        .await
        .unwrap();
    let deleted = lake.head();
    let url = format!("sqlite:{}", lake._dir.path().join("lake.db").display());
    let pool = sqlx::SqlitePool::connect(&url).await.unwrap();
    sqlx::query(
        "UPDATE ducklake_snapshot SET snapshot_time = datetime('now', '-2 hours')
         WHERE snapshot_id = ?",
    )
    .bind(deleted)
    .execute(&pool)
    .await
    .unwrap();
    let ctx = SessionContext::new();
    lake.catalog()
        .flush_inlined_data(&ctx.state(), None, None)
        .await
        .unwrap();
    assert_eq!(lake.inlined_deletes("t"), 0);
    lake.append("t", &batch([100], "i")).await;
    assert_eq!(lake.inlined_rows("t"), 1);
    let hour = InlinedFlushPolicy::never()
        .with_max_inlined_age(Some(std::time::Duration::from_secs(3600)));
    let result = merge_under(&lake, hour).await;
    assert!(result.inlined_flush_snapshot.is_none(), "{result:?}");
    assert_eq!(lake.inlined_rows("t"), 1);
}

/// Set the time of snapshot `snapshot` of a SQLite `lake` to two hours ago.
#[cfg(feature = "write-sqlite")]
async fn age_snapshot(lake: &Lake, snapshot: i64) {
    let url = format!("sqlite:{}", lake._dir.path().join("lake.db").display());
    let pool = sqlx::SqlitePool::connect(&url).await.unwrap();
    sqlx::query(
        "UPDATE ducklake_snapshot SET snapshot_time = datetime('now', '-2 hours')
         WHERE snapshot_id = ?",
    )
    .bind(snapshot)
    .execute(&pool)
    .await
    .unwrap();
    pool.close().await;
}

/// The age trigger times inlined rows whose snapshot was expired by the
/// first surviving snapshot after it. Guards against an expired snapshot
/// reading as no age at all, which kept old rows inlined forever.
#[cfg(feature = "write-sqlite")]
#[tokio::test(flavor = "multi_thread")]
async fn age_trigger_times_rows_of_an_expired_snapshot() {
    let lake = sqlite()(Some(5)).await;
    create(&lake, "t").await;
    lake.append("t", &batch(1..=3, "a")).await;
    let inserted = lake.head();
    create(&lake, "u").await;
    let after = lake.head();
    age_snapshot(&lake, inserted).await;
    age_snapshot(&lake, after).await;
    create(&lake, "v").await;
    let url = format!(
        "sqlite:{}?mode=rwc",
        lake._dir.path().join("lake.db").display()
    );
    datafusion_ducklake::SqliteMetadataWriter::new(&url)
        .await
        .unwrap()
        .expire_snapshots(datafusion_ducklake::maintenance::ExpireCriteria::Versions(
            vec![inserted],
        ))
        .unwrap();
    assert_eq!(lake.inlined_rows("t"), 3);
    let hour = InlinedFlushPolicy::never()
        .with_max_inlined_age(Some(std::time::Duration::from_secs(3600)));
    let result = merge_under(&lake, hour).await;
    assert!(result.inlined_flush_snapshot.is_some(), "{result:?}");
    assert_eq!(lake.inlined_rows("t"), 0);
}

/// Inlined deletions of a retired data file that the flush leaves inlined
/// (they cannot merge into the file's delete file history) neither age nor
/// count towards a flush: rows inlined a moment ago are not an hour old,
/// and one row is not two, though the stranded deletion is two hours old.
/// Guards against a trigger that fires on every compaction for deletions no
/// flush moves.
#[cfg(feature = "write-sqlite")]
#[tokio::test(flavor = "multi_thread")]
async fn flush_triggers_skip_deletions_a_flush_leaves_inlined() {
    let lake = sqlite()(Some(5)).await;
    create(&lake, "t").await;
    lake.append_parquet("t", &batch(1..=20, "a")).await;
    let flush = async || {
        let ctx = SessionContext::new();
        lake.catalog()
            .flush_inlined_data(&ctx.state(), None, None)
            .await
            .unwrap();
    };
    // A delete file for the data file.
    lake.sql("DELETE FROM lake.main.t WHERE id = 1")
        .await
        .unwrap();
    flush().await;
    let (ctx, table) = lake.table("t").await;
    let rewritten = table
        .rewrite_data_files(
            &ctx.state(),
            RewriteOptions {
                delete_threshold: 0.0,
                data_file_ids: None,
            },
        )
        .await
        .unwrap();
    assert!(rewritten.files_processed >= 1, "{rewritten:?}");
    // A deletion of the retired file left inlined, dated where its delete
    // file ended: merged into that delete file, it would fall outside it.
    let url = format!("sqlite:{}", lake._dir.path().join("lake.db").display());
    let pool = sqlx::SqlitePool::connect(&url).await.unwrap();
    let (retired, stranded): (i64, i64) = sqlx::query_as(
        "SELECT data_file_id, end_snapshot FROM ducklake_data_file WHERE end_snapshot IS NOT NULL",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    sqlx::query(sqlx::AssertSqlSafe(format!(
        "INSERT INTO ducklake_inlined_delete_{} (file_id, row_id, begin_snapshot) \
         VALUES (?, 5, ?)",
        lake.table_id("t")
    )))
    .bind(retired)
    .bind(stranded)
    .execute(&pool)
    .await
    .unwrap();
    pool.close().await;
    flush().await;
    let head = lake.head();
    let stranded_rows = lake
        .provider
        .get_inlined_file_deletes_for_flush(lake.table_id("t"), head)
        .unwrap();
    assert_eq!(
        stranded_rows.len(),
        1,
        "the retired file's deletion stays inlined: {stranded_rows:?}"
    );
    age_snapshot(&lake, stranded).await;
    lake.append("t", &batch([100], "i")).await;
    let hour = InlinedFlushPolicy::never()
        .with_max_inlined_age(Some(std::time::Duration::from_secs(3600)));
    let result = merge_under(&lake, hour).await;
    assert!(result.inlined_flush_snapshot.is_none(), "{result:?}");
    let two = InlinedFlushPolicy::never().with_min_inlined_rows(Some(2));
    let result = merge_under(&lake, two).await;
    assert!(result.inlined_flush_snapshot.is_none(), "{result:?}");
    assert_eq!(lake.inlined_rows("t"), 1);
}

/// A metadata provider without the flush reads (a third-party one, which
/// keeps the trait's defaults) leaves nothing to flush: compaction under a
/// flush policy runs and flushes nothing. Guards against the policy failing
/// compaction with the defaults' `Unsupported`.
#[cfg(all(feature = "write-sqlite", feature = "metadata-duckdb"))]
#[tokio::test(flavor = "multi_thread")]
async fn flush_policy_over_a_provider_without_flush_reads() {
    let lake = sqlite()(Some(5)).await;
    create(&lake, "t").await;
    lake.append_parquet("t", &batch(1..=20, "a")).await;
    lake.sql("DELETE FROM lake.main.t WHERE id = 3")
        .await
        .unwrap();
    assert_eq!(lake.inlined_deletes("t"), 1);
    let provider: Arc<dyn MetadataProvider> =
        Arc::new(crate::pushdown_row_preservation_tests::PushdownToggle::new(
            Arc::clone(&lake.provider),
            true,
        ));
    let policy = InlinedFlushPolicy::never()
        .with_min_inlined_rows(Some(1))
        .with_max_inlined_age(Some(std::time::Duration::ZERO));
    let catalog = DuckLakeCatalog::with_writer(provider, Arc::clone(&lake.writer))
        .unwrap()
        .with_write_options(DuckLakeWriteOptions::default().with_inlined_flush_policy(policy));
    let ctx = SessionContext::new();
    DuckLakeCatalog::register(&ctx, "lake", catalog);
    let table = ctx
        .catalog("lake")
        .unwrap()
        .schema("main")
        .unwrap()
        .table("t")
        .await
        .unwrap()
        .unwrap();
    let table = (table.as_ref() as &dyn std::any::Any)
        .downcast_ref::<DuckLakeTable>()
        .unwrap()
        .clone();
    let result = table
        .merge_adjacent_files(&ctx.state(), MergeOptions::default())
        .await
        .unwrap();
    assert!(result.inlined_flush_snapshot.is_none(), "{result:?}");
    assert_eq!(lake.inlined_deletes("t"), 1);
}

/// A `sort_on_insert` or `hive_file_pattern` that releases before this one
/// stored as a case variant of `true` (`TRUE`), which they read as on, is
/// rewritten to `true` when a writer initialises the catalog's schema, so it
/// stays on now that both this crate and official read only exactly `true` as
/// on; any other value is left as stored, as it already reads as official
/// reads it. `quote` quotes the `key` and `value` columns; `exec` runs a
/// statement on the catalog database, `reinit` initialises a writer's schema
/// over it and `read` returns the `(key, value, scope)` rows a query selects.
pub(crate) async fn legacy_true_settings_are_canonicalized(
    quote: char,
    exec: impl AsyncFn(String),
    reinit: impl AsyncFn(),
    read: impl AsyncFn(String) -> Vec<(String, String, Option<String>)>,
) {
    let (key, value) = (format!("{quote}key{quote}"), format!("{quote}value{quote}"));
    let keys = "('sort_on_insert', 'hive_file_pattern', 'auto_compact')";
    exec(format!(
        "DELETE FROM ducklake_metadata WHERE {key} IN {keys}"
    ))
    .await;
    exec(format!(
        "INSERT INTO ducklake_metadata ({key}, {value}, scope, scope_id) VALUES \
         ('sort_on_insert', 'TRUE', NULL, NULL), \
         ('hive_file_pattern', 'True', 'table', 1), \
         ('hive_file_pattern', 'true', 'schema', 0), \
         ('sort_on_insert', 'FALSE', 'schema', 0), \
         ('sort_on_insert', 'trúe', 'table', 1), \
         ('auto_compact', 'TRUE', NULL, NULL)"
    ))
    .await;
    reinit().await;
    reinit().await;
    let mut rows = read(format!(
        "SELECT {key}, {value}, scope FROM ducklake_metadata WHERE {key} IN {keys}"
    ))
    .await;
    rows.sort();
    let row = |k: &str, v: &str, scope: Option<&str>| {
        (k.to_string(), v.to_string(), scope.map(str::to_string))
    };
    assert_eq!(
        rows,
        vec![
            row("auto_compact", "TRUE", None),
            row("hive_file_pattern", "true", Some("schema")),
            row("hive_file_pattern", "true", Some("table")),
            row("sort_on_insert", "FALSE", Some("schema")),
            row("sort_on_insert", "true", None),
            // Not a case variant of `true`, though MySQL's default collation
            // compares it equal.
            row("sort_on_insert", "trúe", Some("table")),
        ]
    );
}

#[cfg(feature = "write-sqlite")]
#[tokio::test(flavor = "multi_thread")]
async fn legacy_true_settings_are_canonicalized_sqlite() {
    let lake = sqlite()(None).await;
    let url = format!("sqlite:{}", lake._dir.path().join("lake.db").display());
    let pool = sqlx::SqlitePool::connect(&url).await.unwrap();
    legacy_true_settings_are_canonicalized(
        '"',
        async |statement| {
            sqlx::query(sqlx::AssertSqlSafe(statement))
                .execute(&pool)
                .await
                .unwrap();
        },
        async || lake.writer.initialize_schema().unwrap(),
        async |query| {
            sqlx::query_as(sqlx::AssertSqlSafe(query))
                .fetch_all(&pool)
                .await
                .unwrap()
        },
    )
    .await;
}

#[cfg(feature = "write-duckdb")]
#[tokio::test(flavor = "multi_thread")]
async fn legacy_true_settings_are_canonicalized_duckdb() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("lake.ducklake");
    let path = path.to_str().unwrap().to_string();
    drop(datafusion_ducklake::DuckdbMetadataWriter::new_with_init(path.clone()).unwrap());
    legacy_true_settings_are_canonicalized(
        '"',
        async |statement| {
            duckdb::Connection::open(&path)
                .unwrap()
                .execute_batch(&statement)
                .unwrap();
        },
        async || {
            drop(datafusion_ducklake::DuckdbMetadataWriter::new_with_init(path.clone()).unwrap());
        },
        async |query| {
            let connection = duckdb::Connection::open(&path).unwrap();
            let mut statement = connection.prepare(&query).unwrap();
            statement
                .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))
                .unwrap()
                .collect::<Result<_, _>>()
                .unwrap()
        },
    )
    .await;
}

// ---------------------------------------------------------------------------
// Runs
// ---------------------------------------------------------------------------

/// The scenarios for a backend that inlines and stages row mutations.
/// Each scenario is boxed: inlined into one future, they overflow a test
/// thread's stack in a debug build.
pub(crate) async fn all_scenarios(make: MakeLake) {
    Box::pin(writer_limit_reaches_every_path(Arc::clone(&make))).await;
    Box::pin(limit_precedence(Arc::clone(&make))).await;
    Box::pin(sort_on_insert_precedence(Arc::clone(&make))).await;
    Box::pin(settings_store_as_official(Arc::clone(&make))).await;
    Box::pin(age_trigger_measures_utc_snapshot_times(Arc::clone(&make))).await;
    Box::pin(sort_changes_bump_schema_version(Arc::clone(&make))).await;
    Box::pin(reserved_names_under_scoped_limits(Arc::clone(&make))).await;
    Box::pin(stored_limit_outranks_writer_options(Arc::clone(&make))).await;
    Box::pin(sql_mutations_follow_the_scoped_limit(Arc::clone(&make))).await;
    Box::pin(sessions_inline_small_writes(Arc::clone(&make))).await;
    Box::pin(partitioned_session_inlines_small_writes(Arc::clone(&make))).await;
    Box::pin(null_typed_columns_inline(Arc::clone(&make))).await;
    Box::pin(null_typed_column_refused_for_not_null(Arc::clone(&make))).await;
    Box::pin(session_with_deletes_inlines(Arc::clone(&make))).await;
    Box::pin(small_deletes_inline(Arc::clone(&make))).await;
    Box::pin(keyed_delete_reaches_inlined_rows(Arc::clone(&make))).await;
    Box::pin(keyed_upsert_replaces_inlined_rows(Arc::clone(&make))).await;
    Box::pin(keyed_update_keeps_row_ids(Arc::clone(&make))).await;
    Box::pin(keyed_composite_and_partitioned(Arc::clone(&make))).await;
    Box::pin(change_detection_sees_inlined_changes(Arc::clone(&make))).await;
    Box::pin(rowid_lookup_everywhere(Arc::clone(&make))).await;
    Box::pin(sort_key_columns_cannot_be_dropped(Arc::clone(&make))).await;
    Box::pin(maintenance_flushes_inlined_data(make)).await;
}

macro_rules! per_backend {
    ($($name:ident),* $(,)?) => {
        $(
            #[cfg(feature = "write-sqlite")]
            mod $name {
                #[tokio::test(flavor = "multi_thread")]
                async fn sqlite() {
                    super::$name(super::sqlite()).await;
                }

                #[cfg(feature = "write-duckdb")]
                #[tokio::test(flavor = "multi_thread")]
                async fn duckdb() {
                    super::$name(super::duckdb()).await;
                }
            }
        )*
    };
}

per_backend!(
    writer_limit_reaches_every_path,
    limit_precedence,
    sort_on_insert_precedence,
    settings_store_as_official,
    age_trigger_measures_utc_snapshot_times,
    sort_changes_bump_schema_version,
    reserved_names_under_scoped_limits,
    stored_limit_outranks_writer_options,
    sql_mutations_follow_the_scoped_limit,
    sessions_inline_small_writes,
    partitioned_session_inlines_small_writes,
    null_typed_columns_inline,
    null_typed_column_refused_for_not_null,
    session_with_deletes_inlines,
    small_deletes_inline,
    keyed_delete_reaches_inlined_rows,
    keyed_upsert_replaces_inlined_rows,
    keyed_update_keeps_row_ids,
    keyed_composite_and_partitioned,
    change_detection_sees_inlined_changes,
    rowid_lookup_everywhere,
    sort_key_columns_cannot_be_dropped,
    maintenance_flushes_inlined_data,
);

#[cfg(feature = "write-mysql")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
#[tokio::test(flavor = "multi_thread")]
async fn mysql_scenarios() {
    use testcontainers::runners::AsyncRunner;
    let container = testcontainers_modules::mysql::Mysql::default()
        .start()
        .await
        .unwrap();
    let port = container.get_host_port_ipv4(3306).await.unwrap();
    all_scenarios(mysql(port)).await;
    let admin = sqlx::MySqlPool::connect(&format!("mysql://root@127.0.0.1:{port}/test"))
        .await
        .unwrap();
    sqlx::query("CREATE DATABASE legacy_settings")
        .execute(&admin)
        .await
        .unwrap();
    let url = format!("mysql://root@127.0.0.1:{port}/legacy_settings");
    let writer = datafusion_ducklake::MySqlMetadataWriter::new_with_init(&url)
        .await
        .unwrap();
    let pool = sqlx::MySqlPool::connect(&url).await.unwrap();
    legacy_true_settings_are_canonicalized(
        '`',
        async |statement| {
            sqlx::query(sqlx::AssertSqlSafe(statement))
                .execute(&pool)
                .await
                .unwrap();
        },
        async || writer.initialize_schema().unwrap(),
        async |query| {
            sqlx::query_as(sqlx::AssertSqlSafe(query))
                .fetch_all(&pool)
                .await
                .unwrap()
        },
    )
    .await;
}

/// [`legacy_true_settings_are_canonicalized`] on a PostgreSQL database at
/// `url`, initialised by `reinit`.
#[cfg(feature = "write-postgres")]
async fn legacy_true_settings_are_canonicalized_on_postgres(
    pool: &sqlx::PgPool,
    reinit: impl AsyncFn(),
) {
    legacy_true_settings_are_canonicalized(
        '"',
        async |statement| {
            sqlx::query(sqlx::AssertSqlSafe(statement))
                .execute(pool)
                .await
                .unwrap();
        },
        reinit,
        async |query| {
            sqlx::query_as(sqlx::AssertSqlSafe(query))
                .fetch_all(pool)
                .await
                .unwrap()
        },
    )
    .await;
}

#[cfg(feature = "write-postgres")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
#[tokio::test(flavor = "multi_thread")]
async fn multicatalog_scenarios() {
    use testcontainers::runners::AsyncRunner;
    let container = testcontainers_modules::postgres::Postgres::default()
        .start()
        .await
        .unwrap();
    let port = container.get_host_port_ipv4(5432).await.unwrap();
    let pool = sqlx::PgPool::connect(&format!(
        "postgresql://postgres:postgres@127.0.0.1:{port}/postgres"
    ))
    .await
    .unwrap();
    datafusion_ducklake::initialize_multicatalog_schema(&pool)
        .await
        .unwrap();
    all_scenarios(multicatalog(pool.clone())).await;
    // Through the bootstrap multi-catalog stores run, and a writer's own.
    legacy_true_settings_are_canonicalized_on_postgres(&pool, async || {
        datafusion_ducklake::initialize_multicatalog_schema(&pool)
            .await
            .unwrap();
    })
    .await;
    let catalog_id = datafusion_ducklake::MulticatalogManager::new(pool.clone())
        .create_catalog("legacy_settings")
        .await
        .unwrap();
    let writer = datafusion_ducklake::PostgresMetadataWriter::with_pool(pool.clone(), catalog_id)
        .await
        .unwrap();
    legacy_true_settings_are_canonicalized_on_postgres(&pool, async || {
        writer.initialize_schema().unwrap();
    })
    .await;
}

/// Single-catalog PostgreSQL stores and resolves the limit, refuses reserved
/// names under it, and reports its changes; it inlines nothing. The keyed,
/// position and `rowid` scenarios are not run: they stage row mutations or
/// run SQL `UPDATE`, neither of which its writer supports (both refuse with
/// `Unsupported`).
#[cfg(feature = "write-postgres")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
#[tokio::test(flavor = "multi_thread")]
async fn postgres_single_catalog_scenarios() {
    use testcontainers::runners::AsyncRunner;
    let container = testcontainers_modules::postgres::Postgres::default()
        .start()
        .await
        .unwrap();
    let port = container.get_host_port_ipv4(5432).await.unwrap();
    let admin = sqlx::PgPool::connect(&format!(
        "postgresql://postgres:postgres@127.0.0.1:{port}/postgres"
    ))
    .await
    .unwrap();
    sqlx::query("CREATE DATABASE legacy_settings")
        .execute(&admin)
        .await
        .unwrap();
    let url = format!("postgresql://postgres:postgres@127.0.0.1:{port}/legacy_settings");
    let writer = datafusion_ducklake::PostgresSingleCatalogMetadataWriter::new_with_init(&url)
        .await
        .unwrap();
    let pool = sqlx::PgPool::connect(&url).await.unwrap();
    legacy_true_settings_are_canonicalized_on_postgres(&pool, async || {
        writer.initialize_schema().unwrap();
    })
    .await;
    let make = postgres_single(port);
    Box::pin(writer_limit_reaches_every_path(Arc::clone(&make))).await;
    Box::pin(limit_precedence(Arc::clone(&make))).await;
    Box::pin(sort_on_insert_precedence(Arc::clone(&make))).await;
    Box::pin(settings_store_as_official(Arc::clone(&make))).await;
    Box::pin(sort_changes_bump_schema_version(Arc::clone(&make))).await;
    Box::pin(stored_limit_outranks_writer_options(Arc::clone(&make))).await;
    Box::pin(reserved_names_under_scoped_limits(Arc::clone(&make))).await;
    Box::pin(null_typed_columns_inline(Arc::clone(&make))).await;
    Box::pin(null_typed_column_refused_for_not_null(Arc::clone(&make))).await;
    Box::pin(sort_key_columns_cannot_be_dropped(Arc::clone(&make))).await;
    Box::pin(change_detection_sees_inlined_changes(make)).await;
}

/// A writer configured with `0` creates no inlined data table on any direct
/// call that makes a new schema version (create, add column, type promotion,
/// partitioning), where a default writer creates one each time. Guards against
/// eager empty inlined tables when the caller wants no inlining.
#[cfg(feature = "write-sqlite")]
#[tokio::test(flavor = "multi_thread")]
async fn writer_limit_zero_creates_no_inlined_tables() {
    for (limit, expect_tables) in [(Some(0), false), (None, true)] {
        let lake = sqlite()(limit).await;
        create_with(
            &lake,
            "t",
            &[("id", DataType::Int32), ("v", DataType::Utf8)],
        )
        .unwrap();
        let table_id = lake.table_id("t");
        lake.writer
            .promote_column_type(table_id, "id", "int64")
            .unwrap();
        lake.writer
            .set_partition_spec(
                table_id,
                &[(
                    "v".to_string(),
                    datafusion_ducklake::PartitionTransform::Identity,
                )],
            )
            .unwrap();
        let db = lake._dir.path().join("lake.db");
        let pool = sqlx::SqlitePool::connect(&format!("sqlite:{}", db.display()))
            .await
            .unwrap();
        let tables: i64 = sqlx::query_scalar("SELECT count(*) FROM ducklake_inlined_data_tables")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(
            tables > 0,
            expect_tables,
            "limit {limit:?}: {tables} inlined tables"
        );
    }
}

/// Upstream oracle check: what official DuckLake (the pinned DuckDB) reads
/// from a catalog this crate wrote through the new paths equals what this
/// crate reads.
#[cfg(feature = "write-sqlite")]
#[tokio::test(flavor = "multi_thread")]
async fn official_reads_what_the_new_paths_wrote() {
    let lake = sqlite()(Some(5)).await;
    create(&lake, "t").await;
    let created = lake.head();
    lake.append_parquet("t", &batch(1..=20, "a")).await;
    let writer = lake.table_writer();
    let mut session = writer
        .begin_write("main", "t", &schema(), WriteMode::Append)
        .unwrap();
    session.write_batch(&batch([100, 101, 102], "s")).unwrap();
    session.finish().await.unwrap();
    let (ctx, table) = lake.table("t").await;
    table
        .keyed_mutation(&["id"])
        .upsert_by_keys(&ctx.state(), vec![batch([3, 101, 300], "u")])
        .await
        .unwrap();
    let (ctx, table) = lake.table("t").await;
    table
        .keyed_mutation(&["id"])
        .update_by_keys(&ctx.state(), vec![batch([4, 100], "w")])
        .await
        .unwrap();
    let (ctx, table) = lake.table("t").await;
    table
        .keyed_mutation(&["id"])
        .delete_by_keys(&ctx.state(), vec![keys((10..=18).map(Some))])
        .await
        .unwrap();
    let ours = lake.rows("t").await;
    let db = lake._dir.path().join("lake.db");
    let official = crate::inlined_mutation_differential_tests::duckdb_query(
        &crate::inlined_mutation_differential_tests::Metadata::Sqlite(db.clone()),
        &lake.data,
        "SELECT rowid, id, v FROM lake.main.t ORDER BY rowid",
    );
    assert_eq!(official, ours);

    // Official's snapshot listing reads every snapshot's `changes_made`, and
    // its change feed reports what this crate's does over each snapshot and
    // the whole history (the fork's multi-catalog layout is not one DuckDB
    // attaches, so it is covered by this crate's feed alone).
    use crate::inlined_mutation_differential_tests as differential;
    let head = lake.head();
    let connection = differential::duckdb_connection();
    differential::attach_lake(
        &connection,
        &differential::Metadata::Sqlite(db.clone()),
        &lake.data,
        true,
    );
    let listed = differential::query_lines(
        &connection,
        "SELECT snapshot_id FROM ducklake_snapshots('lake') ORDER BY snapshot_id",
    );
    let pool = sqlx::SqlitePool::connect(&format!("sqlite:{}", db.display()))
        .await
        .unwrap();
    let stored: Vec<i64> =
        sqlx::query_scalar("SELECT snapshot_id FROM ducklake_snapshot ORDER BY snapshot_id")
            .fetch_all(&pool)
            .await
            .unwrap();
    assert_eq!(
        listed,
        stored.iter().map(i64::to_string).collect::<Vec<_>>()
    );
    let ctx = SessionContext::new();
    datafusion_ducklake::register_ducklake_functions(&ctx, Arc::clone(&lake.provider));
    for (from, through) in differential::change_windows(created, head) {
        let official = differential::query_lines(
            &connection,
            &format!(
                "SELECT * FROM ducklake_table_changes('lake', 'main', 't', {from}, {through}) \
                 ORDER BY snapshot_id, rowid, change_type"
            ),
        );
        let batches = ctx
            .sql(&format!(
                "SELECT * FROM ducklake_table_changes('main.t', {from}, {through}) \
                 ORDER BY snapshot_id, rowid, change_type"
            ))
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        assert_eq!(
            official,
            differential::arrow_lines(&batches),
            "window {from}..={through}"
        );
    }
}

// ---------------------------------------------------------------------------
// D3 performance: keyed mutations against the hand-built position-delete path
// ---------------------------------------------------------------------------

/// Times a keyed delete and a keyed upsert against the hand-built path they
/// replace (list the files the keys may be in, resolve positions file by file,
/// write a cumulative delete file per file, commit; for the upsert, append the
/// rows with the deletes through a streaming session), on identical tables of
/// `PERF_ROWS` rows (default 2,000,000) in `PERF_FILES` files (default 20),
/// with `PERF_KEYS` keys (default 2,000) spread over every file, alternating
/// the order over `PERF_ROUNDS` rounds (default 3). Prints each side's times.
#[cfg(feature = "write-sqlite")]
#[ignore = "performance comparison: run explicitly"]
#[tokio::test(flavor = "multi_thread")]
async fn keyed_mutation_performance() {
    let env = |name: &str, default: usize| {
        std::env::var(name)
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(default)
    };
    let rows = env("PERF_ROWS", 2_000_000) as i64;
    let files = env("PERF_FILES", 20) as i64;
    let key_count = env("PERF_KEYS", 2_000) as i64;
    let rounds = env("PERF_ROUNDS", 3);

    // Limit 0: both paths record every deletion in delete files, so they do
    // the same catalog work.
    let mut lakes = Vec::new();
    for _ in 0..2 {
        let lake = sqlite()(Some(0)).await;
        create(&lake, "t").await;
        let per_file = rows / files;
        for file in 0..files {
            lake.append("t", &batch(file * per_file..(file + 1) * per_file, "v"))
                .await;
        }
        lakes.push(lake);
    }
    let mut hand_delete = Vec::new();
    let mut keyed_delete = Vec::new();
    let mut hand_upsert = Vec::new();
    let mut keyed_upsert = Vec::new();
    for round in 0..rounds {
        let stride = rows / key_count;
        let ids: Vec<i64> = (0..key_count)
            .map(|k| k * stride + (round as i64 * 7 + 3) % stride)
            .collect();
        let upserted: Vec<i64> = ids.iter().map(|id| id + 1).collect();
        let order = if round % 2 == 0 {
            [0, 1]
        } else {
            [1, 0]
        };
        for which in order {
            let lake = &lakes[which];
            let started = std::time::Instant::now();
            if which == 0 {
                hand_built_delete(lake, &ids).await;
                hand_delete.push(started.elapsed());
            } else {
                let (ctx, table) = lake.table("t").await;
                table
                    .keyed_mutation(&["id"])
                    .delete_by_keys(&ctx.state(), vec![keys(ids.iter().copied().map(Some))])
                    .await
                    .unwrap();
                keyed_delete.push(started.elapsed());
            }
            let started = std::time::Instant::now();
            if which == 0 {
                hand_built_upsert(lake, &upserted).await;
                hand_upsert.push(started.elapsed());
            } else {
                let (ctx, table) = lake.table("t").await;
                table
                    .keyed_mutation(&["id"])
                    .upsert_by_keys(&ctx.state(), vec![batch(upserted.iter().copied(), "u")])
                    .await
                    .unwrap();
                keyed_upsert.push(started.elapsed());
            }
        }
    }
    assert_eq!(
        lakes[0].rows("t").await.len(),
        lakes[1].rows("t").await.len(),
        "both paths made the same change"
    );
    eprintln!(
        "keyed_mutation_performance: {rows} rows, {files} files, {key_count} keys\n  \
         delete: hand-built {hand_delete:?}\n          keyed      {keyed_delete:?}\n  \
         upsert: hand-built {hand_upsert:?}\n          keyed      {keyed_upsert:?}"
    );
}

/// The hand-built delete: list the files the keys may be in, resolve each
/// file's positions in turn, write each file's cumulative delete file, commit.
#[cfg(feature = "write-sqlite")]
async fn hand_built_delete(lake: &Lake, ids: &[i64]) -> Vec<datafusion_ducklake::DeleteFileEntry> {
    let entries = hand_built_entries(lake, ids).await;
    lake.writer
        .commit_deletes(lake.table_id("t"), "main", "t", lake.head(), &entries, &[])
        .unwrap();
    entries
}

#[cfg(feature = "write-sqlite")]
pub(crate) async fn hand_built_entries(
    lake: &Lake,
    ids: &[i64],
) -> Vec<datafusion_ducklake::DeleteFileEntry> {
    use datafusion::physical_expr::expressions::{Column, InListExpr, lit};
    let (ctx, table) = lake.table("t").await;
    let state = ctx.state();
    let schema = datafusion::catalog::TableProvider::schema(&table);
    let predicate: Arc<dyn datafusion::physical_expr::PhysicalExpr> = Arc::new(
        InListExpr::try_new(
            Arc::new(Column::new("id", 0)),
            ids.iter().map(|id| lit(*id)).collect(),
            false,
            &schema,
        )
        .unwrap(),
    );
    let writer = lake.table_writer();
    let mut entries = Vec::new();
    for file in table.files_matching(&predicate).unwrap() {
        let matched = table
            .resolve_positions(&state, &file.file, Arc::clone(&predicate))
            .await
            .unwrap();
        if matched.is_empty() {
            continue;
        }
        let mut positions: Vec<i64> = matched.into_iter().collect();
        if let Some(delete_file) = &file.delete_file {
            positions.extend(
                table
                    .read_delete_file_positions(&state, delete_file)
                    .await
                    .unwrap(),
            );
        }
        positions.sort_unstable();
        positions.dedup();
        let delete = writer
            .write_delete_file("main", "t", &file.file.path, &positions)
            .await
            .unwrap();
        entries.push(datafusion_ducklake::DeleteFileEntry {
            data_file_id: file.data_file_id,
            expected_prev_delete_file: file.delete_file_id,
            delete,
        });
    }
    entries
}

/// The hand-built upsert: the hand-built deletes, committed with the new rows
/// by a streaming session.
#[cfg(feature = "write-sqlite")]
async fn hand_built_upsert(lake: &Lake, ids: &[i64]) {
    let entries = hand_built_entries(lake, ids).await;
    let mut session = lake
        .table_writer()
        .begin_write("main", "t", &schema(), WriteMode::Append)
        .unwrap();
    session
        .write_batch(&batch(ids.iter().copied(), "u"))
        .unwrap();
    session.finish_with_deletes(&entries).await.unwrap();
}

/// An `UPDATE` of inlined rows of a table with NOT NULL struct fields commits:
/// the staged commit's schema check does not compare a nested field's
/// nullability, which the table's Arrow schema does not carry.
#[cfg(feature = "write-sqlite")]
#[tokio::test(flavor = "multi_thread")]
async fn update_of_inlined_rows_with_not_null_struct_fields() {
    use arrow::array::{Int32Array, StructArray};
    let lake = sqlite()(None).await;
    let fields: Vec<Arc<Field>> = vec![
        Arc::new(Field::new("label", DataType::Utf8, false)),
        Arc::new(Field::new("score", DataType::Int32, false)),
    ];
    let values = StructArray::new(
        fields.clone().into(),
        vec![
            Arc::new(StringArray::from(vec!["one", "two"])),
            Arc::new(Int32Array::from(vec![10, 20])),
        ],
        None,
    );
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("attrs", DataType::Struct(fields.into()), false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(Int32Array::from(vec![1, 2])), Arc::new(values)],
    )
    .unwrap();
    lake.table_writer()
        .write_table("main", "s", &[batch])
        .await
        .unwrap();
    assert_eq!(lake.inlined_rows("s"), 2);
    lake.sql("UPDATE lake.main.s SET id = 5 WHERE id = 1")
        .await
        .unwrap();
    let rows = lake.rows("s").await;
    assert_eq!(rows.len(), 2);
    assert!(rows.iter().any(|row| row.starts_with("0,5,")), "{rows:?}");
}

/// The row-level APIs' futures are `Send`, so a caller can spawn them.
#[cfg(feature = "write-sqlite")]
#[tokio::test(flavor = "multi_thread")]
async fn row_api_futures_can_be_spawned() {
    let lake = Arc::new(sqlite()(Some(5)).await);
    create(&lake, "t").await;
    lake.append_parquet("t", &batch(1..=20, "a")).await;
    let spawned = Arc::clone(&lake);
    tokio::spawn(async move {
        let (ctx, table) = spawned.table("t").await;
        let state = ctx.state();
        table
            .keyed_mutation(&["id"])
            .upsert_by_keys(&state, vec![batch([1, 30], "s")])
            .await
            .unwrap();
        let rowids: Vec<i64> = (0..3).collect();
        table.locate_rowids(&state, &rowids).await.unwrap();
        table.read_rows_by_rowid(&state, &rowids).await.unwrap();
        // The upsert committed since this handle's snapshot: delete through
        // a fresh one.
        let (ctx, table) = spawned.table("t").await;
        table.delete_rowids(&ctx.state(), &rowids).await.unwrap();
    })
    .await
    .unwrap();
}

/// A time-travel bound with a non-ASCII offset (`+a€`, four bytes) is
/// refused as an invalid timestamp, not a panic slicing inside a character.
#[cfg(feature = "write-sqlite")]
#[tokio::test(flavor = "multi_thread")]
async fn non_ascii_snapshot_offset_is_an_error() {
    let lake = sqlite()(Some(5)).await;
    create(&lake, "t").await;
    lake.append("t", &batch(1..=2, "a")).await;
    let ctx = SessionContext::new();
    datafusion_ducklake::register_ducklake_functions(&ctx, Arc::clone(&lake.provider));
    // Spawned, so a panic is reported as a failed join, not a crash.
    let outcome = tokio::spawn(query_owned(ctx.clone())).await;
    let error = outcome
        .expect("no panic")
        .expect_err("an invalid timestamp")
        .to_string();
    assert!(error.contains("is not a valid timestamp"), "{error}");
}

async fn query_owned(ctx: SessionContext) -> datafusion::error::Result<Vec<RecordBatch>> {
    ctx.sql(
        "SELECT * FROM ducklake_table_changes('main.t', '2024-01-01 00:00:00+a€', \
         '2030-01-01 00:00:00')",
    )
    .await?
    .collect()
    .await
}
