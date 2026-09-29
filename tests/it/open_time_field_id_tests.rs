//! A scan resolves each data file's columns when the parquet reader opens the
//! file, as official DuckLake does, so planning reads no data file and a scan that
//! stops early opens only the files it reaches.
//!
//! Official DuckLake binds a file's columns by field id as each reader opens
//! (`DuckLakeMultiFileReader::Bind` / `CreateMapping`). The fixtures here are
//! written by the official extension and read back by both engines, with DuckDB's
//! own answer as the oracle, across the schema evolutions a per-file resolution has
//! to get right: a swapped rename, a widened type, a column added with a default,
//! a struct child renamed and another added, and a file with no field ids at all.

#![cfg(all(feature = "metadata-duckdb", feature = "metadata-sqlite"))]

use std::collections::HashSet;
use std::ops::Range;
use std::path::Path;
use std::sync::{Arc, Mutex};

use arrow::array::{Array, StringArray};
use arrow::record_batch::RecordBatch;
use async_trait::async_trait;
use bytes::Bytes;
use datafusion::parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use datafusion::physical_plan::{ExecutionPlan, collect, displayable};
use datafusion::prelude::*;
use futures::stream::BoxStream;
use object_store::local::LocalFileSystem;
use object_store::path::Path as ObjectPath;
use object_store::{
    CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta, ObjectStore,
    PutMultipartOptions, PutOptions, PutPayload, PutResult,
};
use sqlx::sqlite::SqlitePool;
use tempfile::TempDir;

use datafusion_ducklake::{DuckLakeCatalog, SqliteMetadataProvider};

/// Data files written after the schema evolution, on top of the one written
/// before it.
const LATER_FILES: i32 = 23;

/// Every column cast to text, so both engines render values the same way.
const EVOLVED_QUERY: &str = "SELECT CAST(id AS VARCHAR) AS id_text, \
     CAST(a AS VARCHAR) AS a_text, CAST(b AS VARCHAR) AS b_text, \
     CAST(w AS VARCHAR) AS w_text, CAST(s['xx'] AS VARCHAR) AS xx_text, \
     CAST(s['y'] AS VARCHAR) AS y_text, CAST(s['z'] AS VARCHAR) AS z_text, \
     CAST(d AS VARCHAR) AS d_text FROM {table} ORDER BY id";

/// Records every read that reaches the store, and which objects it touched.
#[derive(Debug)]
struct ReadRecordingStore {
    inner: Arc<dyn ObjectStore>,
    reads: Mutex<Vec<ObjectPath>>,
}

impl ReadRecordingStore {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            inner: Arc::new(LocalFileSystem::new()),
            reads: Mutex::new(Vec::new()),
        })
    }

    fn record(&self, location: &ObjectPath) {
        self.reads.lock().unwrap().push(location.clone());
    }

    /// The reads recorded since the last call, then forgets them.
    fn take(&self) -> Vec<ObjectPath> {
        std::mem::take(&mut *self.reads.lock().unwrap())
    }
}

impl std::fmt::Display for ReadRecordingStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ReadRecordingStore({})", self.inner)
    }
}

#[async_trait]
impl ObjectStore for ReadRecordingStore {
    async fn put_opts(
        &self,
        location: &ObjectPath,
        payload: PutPayload,
        opts: PutOptions,
    ) -> object_store::Result<PutResult> {
        self.inner.put_opts(location, payload, opts).await
    }

    async fn put_multipart_opts(
        &self,
        location: &ObjectPath,
        opts: PutMultipartOptions,
    ) -> object_store::Result<Box<dyn MultipartUpload>> {
        self.inner.put_multipart_opts(location, opts).await
    }

    // `get`, `get_range` and `head` all arrive here through `ObjectStoreExt`.
    async fn get_opts(
        &self,
        location: &ObjectPath,
        options: GetOptions,
    ) -> object_store::Result<GetResult> {
        self.record(location);
        self.inner.get_opts(location, options).await
    }

    async fn get_ranges(
        &self,
        location: &ObjectPath,
        ranges: &[Range<u64>],
    ) -> object_store::Result<Vec<Bytes>> {
        self.record(location);
        self.inner.get_ranges(location, ranges).await
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, object_store::Result<ObjectPath>>,
    ) -> BoxStream<'static, object_store::Result<ObjectPath>> {
        self.inner.delete_stream(locations)
    }

    fn list(
        &self,
        prefix: Option<&ObjectPath>,
    ) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        self.inner.list(prefix)
    }

    async fn list_with_delimiter(
        &self,
        prefix: Option<&ObjectPath>,
    ) -> object_store::Result<ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }

    async fn copy_opts(
        &self,
        from: &ObjectPath,
        to: &ObjectPath,
        options: CopyOptions,
    ) -> object_store::Result<()> {
        self.inner.copy_opts(from, to, options).await
    }
}

fn catalog_path(temp: &TempDir) -> std::path::PathBuf {
    temp.path().join("catalog.db")
}

/// Attach the fixture catalog to a fresh DuckDB connection.
fn duckdb_lake(temp: &TempDir) -> anyhow::Result<duckdb::Connection> {
    let data_path = temp.path().join("data");
    std::fs::create_dir_all(&data_path)?;
    let conn = duckdb::Connection::open_in_memory()?;
    crate::common::ensure_ducklake_installed();
    crate::common::ensure_extension_installed("sqlite");
    conn.execute("LOAD sqlite", [])?;
    conn.execute("LOAD ducklake", [])?;
    conn.execute(
        &format!(
            "ATTACH 'ducklake:sqlite:{}' AS lake (DATA_PATH '{}', DATA_INLINING_ROW_LIMIT 0)",
            catalog_path(temp).display(),
            data_path.display()
        ),
        [],
    )?;
    Ok(conn)
}

/// Rows of `sql` as DuckDB answers them, every column read as text.
fn duckdb_rows(conn: &duckdb::Connection, sql: &str) -> anyhow::Result<Vec<Vec<Option<String>>>> {
    let mut statement = conn.prepare(sql)?;
    let mut rows = statement.query([])?;
    let mut out = Vec::new();
    while let Some(row) = rows.next()? {
        let width = row.as_ref().column_count();
        out.push(
            (0..width)
                .map(|index| row.get::<_, Option<String>>(index))
                .collect::<duckdb::Result<Vec<_>>>()?,
        );
    }
    Ok(out)
}

/// One file written before a swapped rename (`a` ↔ `b`), a widening of `w`, a
/// column `d` added with a default, and a struct child renamed (`s.x` → `s.xx`)
/// beside one added (`s.z`); then [`LATER_FILES`] files written after them.
/// Returns DuckDB's answer to [`EVOLVED_QUERY`].
///
/// Every file holds two rows that differ in every column. A file whose column is
/// constant has equal catalog bounds for it, and DataFusion's parquet opener then
/// replaces the column with that literal before any adapter sees it, which would
/// answer correctly whatever column the adapter picked.
fn create_evolved_table(temp: &TempDir) -> anyhow::Result<Vec<Vec<Option<String>>>> {
    let conn = duckdb_lake(temp)?;
    conn.execute(
        "CREATE TABLE lake.t (id INT, a INT, b VARCHAR, w INT, s STRUCT(x INT, y INT))",
        [],
    )?;
    conn.execute(
        "INSERT INTO lake.t VALUES (1, 10, 'one', 100, {'x': 1000, 'y': 1001}), \
         (101, 11, 'one-b', 101, {'x': 1100, 'y': 1101})",
        [],
    )?;
    conn.execute("ALTER TABLE lake.t RENAME COLUMN a TO tmp", [])?;
    conn.execute("ALTER TABLE lake.t RENAME COLUMN b TO a", [])?;
    conn.execute("ALTER TABLE lake.t RENAME COLUMN tmp TO b", [])?;
    conn.execute("ALTER TABLE lake.t ALTER COLUMN w SET DATA TYPE BIGINT", [])?;
    conn.execute("ALTER TABLE lake.t ADD COLUMN d INT DEFAULT 42", [])?;
    conn.execute("ALTER TABLE lake.t RENAME COLUMN s.x TO xx", [])?;
    conn.execute("ALTER TABLE lake.t ADD COLUMN s.z INT", [])?;
    let later_row = |id: i32| {
        format!(
            "({id}, 'v{id}', {id}, {w}, {{'xx': {id}0, 'y': {id}1, 'z': {id}2}}, {id})",
            w = 5_000_000_000_i64 + i64::from(id),
        )
    };
    for id in 2..2 + LATER_FILES {
        conn.execute(
            &format!(
                "INSERT INTO lake.t (id, a, b, w, s, d) VALUES {}, {}",
                later_row(id),
                later_row(id + 100)
            ),
            [],
        )?;
    }
    let expected = duckdb_rows(&conn, &EVOLVED_QUERY.replace("{table}", "lake.t"))?;
    conn.execute("DETACH lake", [])?;
    Ok(expected)
}

/// A session over the fixture catalog whose local files are read through `store`.
async fn session(
    temp: &TempDir,
    store: &Arc<ReadRecordingStore>,
    config: SessionConfig,
) -> anyhow::Result<SessionContext> {
    let provider =
        SqliteMetadataProvider::new(&format!("sqlite:{}", catalog_path(temp).display())).await?;
    let ctx = SessionContext::new_with_config(config);
    ctx.runtime_env().register_object_store(
        &url::Url::parse("file://")?,
        Arc::clone(store) as Arc<dyn ObjectStore>,
    );
    ctx.register_catalog("ducklake", Arc::new(DuckLakeCatalog::new(provider)?));
    Ok(ctx)
}

/// Every cell of `batches`, read as text.
fn text_rows(batches: &[RecordBatch]) -> Vec<Vec<Option<String>>> {
    let mut rows = Vec::new();
    for batch in batches {
        let columns: Vec<StringArray> = batch
            .columns()
            .iter()
            .map(|column| {
                arrow::compute::cast(column, &arrow::datatypes::DataType::Utf8)
                    .expect("every column is cast to text")
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .expect("cast to Utf8 yields a StringArray")
                    .clone()
            })
            .collect();
        for row in 0..batch.num_rows() {
            rows.push(
                columns
                    .iter()
                    .map(|column| (!column.is_null(row)).then(|| column.value(row).to_string()))
                    .collect(),
            );
        }
    }
    rows
}

async fn physical_plan(ctx: &SessionContext, sql: &str) -> anyhow::Result<Arc<dyn ExecutionPlan>> {
    Ok(ctx.sql(sql).await?.create_physical_plan().await?)
}

fn distinct(reads: &[ObjectPath]) -> HashSet<&ObjectPath> {
    reads.iter().collect()
}

/// Planning a scan over evolved files reads none of them, and executing it reads
/// every file under the schema each one was written with, as DuckDB does.
#[tokio::test]
async fn planning_reads_no_data_file_and_each_file_resolves_as_duckdb_does() -> anyhow::Result<()> {
    let temp = TempDir::new()?;
    let expected = create_evolved_table(&temp)?;
    assert_eq!(expected.len(), 2 * (1 + LATER_FILES as usize));
    // The pre-evolution file is the one every evolution applies to: `a` and `b`
    // come back swapped by identity, `w` widened, `s.xx` under its new name,
    // `s.z` absent, and `d` at its default.
    let text = |value: &str| Some(value.to_string());
    assert_eq!(
        expected[0],
        vec![
            text("1"),
            text("one"),
            text("10"),
            text("100"),
            text("1000"),
            text("1001"),
            None,
            text("42")
        ]
    );

    let store = ReadRecordingStore::new();
    let ctx = session(&temp, &store, SessionConfig::new()).await?;
    let plan = physical_plan(&ctx, &EVOLVED_QUERY.replace("{table}", "ducklake.main.t")).await?;
    assert_eq!(
        store.take(),
        Vec::<ObjectPath>::new(),
        "planning must not read any data file"
    );
    let rendered = displayable(plan.as_ref()).indent(true).to_string();
    assert!(
        !rendered.contains("ColumnRenameExec"),
        "a field-id resolved scan presents the catalog schema itself:\n{rendered}"
    );

    let batches = collect(plan, ctx.task_ctx()).await?;
    assert_eq!(text_rows(&batches), expected);
    assert_eq!(
        distinct(&store.take()).len(),
        1 + LATER_FILES as usize,
        "execution reads every file"
    );
    Ok(())
}

/// A predicate on a renamed column is pushed into the scan and still selects by
/// the column's catalog identity in every file: in the pre-rename file `a` is
/// stored under the name `b`, and the other way round.
#[tokio::test]
async fn a_filter_on_a_swapped_column_matches_duckdb() -> anyhow::Result<()> {
    let temp = TempDir::new()?;
    create_evolved_table(&temp)?;
    let conn = duckdb_lake(&temp)?;
    let sql =
        "SELECT CAST(id AS VARCHAR) AS id_text FROM {table} WHERE b = 10 OR a = 'v5' ORDER BY id";
    let expected = duckdb_rows(&conn, &sql.replace("{table}", "lake.t"))?;
    // `b = 10` holds in the pre-rename file, which stores `b` as `a`, and in the
    // file written for id 10; `a = 'v5'` only in the file written for id 5.
    assert_eq!(
        expected,
        vec![
            vec![Some("1".to_string())],
            vec![Some("5".to_string())],
            vec![Some("10".to_string())]
        ]
    );

    let store = ReadRecordingStore::new();
    let ctx = session(&temp, &store, SessionConfig::new()).await?;
    let batches = ctx
        .sql(&sql.replace("{table}", "ducklake.main.t"))
        .await?
        .collect()
        .await?;
    assert_eq!(text_rows(&batches), expected);
    Ok(())
}

/// `LIMIT 1` over many files opens only the files it reaches.
#[tokio::test]
async fn limit_one_opens_only_the_files_it_needs() -> anyhow::Result<()> {
    let temp = TempDir::new()?;
    create_evolved_table(&temp)?;

    let store = ReadRecordingStore::new();
    let ctx = session(
        &temp,
        &store,
        SessionConfig::new().with_target_partitions(1),
    )
    .await?;
    let plan = physical_plan(&ctx, "SELECT * FROM ducklake.main.t LIMIT 1").await?;
    assert_eq!(store.take(), Vec::<ObjectPath>::new());

    let batches = collect(plan, ctx.task_ctx()).await?;
    assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 1);
    let opened = distinct(&store.take()).len();
    // One partition reads its first file and may already be opening the next.
    assert!(
        (1..=2).contains(&opened),
        "LIMIT 1 opened {opened} of {} files",
        1 + LATER_FILES
    );
    Ok(())
}

/// A file that carries no field ids and no name mapping is matched by name, and
/// sits in the same scan as files resolved by field id.
#[tokio::test]
async fn a_file_without_field_ids_reads_beside_field_id_files() -> anyhow::Result<()> {
    let temp = TempDir::new()?;
    let external = temp.path().join("external.parquet");
    let conn = duckdb_lake(&temp)?;
    conn.execute("CREATE TABLE lake.t (id INT, name VARCHAR)", [])?;
    conn.execute("INSERT INTO lake.t VALUES (1, 'one'), (4, 'four')", [])?;
    conn.execute(
        &format!(
            "COPY (SELECT * FROM (VALUES (2::INT, 'two'), (3::INT, 'three')) v(id, name)) \
             TO '{}' (FORMAT PARQUET)",
            external.display()
        ),
        [],
    )?;
    conn.execute(
        &format!(
            "CALL ducklake_add_data_files('lake', 't', '{}')",
            external.display()
        ),
        [],
    )?;
    conn.execute("DETACH lake", [])?;
    drop(conn);
    // Registration records a name mapping; clearing it leaves a file that only
    // its own column names describe, as a catalog written before name mappings
    // existed would hold.
    clear_name_mappings(&catalog_path(&temp)).await?;
    let footer = ParquetRecordBatchReaderBuilder::try_new(std::fs::File::open(&external)?)?;
    assert!(
        footer
            .parquet_schema()
            .columns()
            .iter()
            .all(|column| !column.self_type().get_basic_info().has_id()),
        "the registered file must carry no field ids"
    );

    let conn = duckdb_lake(&temp)?;
    let sql = "SELECT CAST(id AS VARCHAR) AS id_text, CAST(name AS VARCHAR) AS name_text \
               FROM {table} ORDER BY id";
    let expected = duckdb_rows(&conn, &sql.replace("{table}", "lake.t"))?;
    assert_eq!(expected.len(), 4);

    let store = ReadRecordingStore::new();
    let ctx = session(&temp, &store, SessionConfig::new()).await?;
    let plan = physical_plan(&ctx, &sql.replace("{table}", "ducklake.main.t")).await?;
    assert_eq!(store.take(), Vec::<ObjectPath>::new());
    let batches = collect(plan, ctx.task_ctx()).await?;
    assert_eq!(text_rows(&batches), expected);
    Ok(())
}

async fn clear_name_mappings(catalog: &Path) -> anyhow::Result<()> {
    let pool = SqlitePool::connect(&format!("sqlite:{}", catalog.display())).await?;
    let cleared =
        sqlx::query("UPDATE ducklake_data_file SET mapping_id = NULL WHERE mapping_id IS NOT NULL")
            .execute(&pool)
            .await?
            .rows_affected();
    assert_eq!(cleared, 1, "exactly the registered file carried a mapping");
    pool.close().await;
    Ok(())
}

/// DataFusion's parquet opener coerces a file's binary column to a string type
/// when the scan's schema has a string column of the same name, and the scan's
/// schema carries catalog names. A rename swap between a `VARCHAR` and a `BLOB`
/// puts the file's blob under the current string column's name, so the blob
/// passes through that coercion on its way to the column that owns it. Bytes that
/// are not UTF-8 must still come back unchanged.
#[tokio::test]
async fn a_varchar_and_blob_swap_reads_as_duckdb_does() -> anyhow::Result<()> {
    let temp = TempDir::new()?;
    let conn = duckdb_lake(&temp)?;
    conn.execute("CREATE TABLE lake.t (id INT, a BLOB, b VARCHAR)", [])?;
    conn.execute(
        "INSERT INTO lake.t VALUES (1, '\\xFF\\xFE'::BLOB, 'x'), (2, '\\xFE\\xFF\\x01'::BLOB, 'y')",
        [],
    )?;
    conn.execute("ALTER TABLE lake.t RENAME COLUMN a TO tmp", [])?;
    conn.execute("ALTER TABLE lake.t RENAME COLUMN b TO a", [])?;
    conn.execute("ALTER TABLE lake.t RENAME COLUMN tmp TO b", [])?;
    let expected = duckdb_rows(
        &conn,
        "SELECT CAST(id AS VARCHAR), CAST(a AS VARCHAR), lower(hex(b)) FROM lake.t ORDER BY id",
    )?;
    assert_eq!(
        expected,
        vec![
            vec![Some("1".to_string()), Some("x".to_string()), Some("fffe".to_string())],
            vec![Some("2".to_string()), Some("y".to_string()), Some("feff01".to_string())],
        ]
    );
    conn.execute("DETACH lake", [])?;

    let store = ReadRecordingStore::new();
    let ctx = session(&temp, &store, SessionConfig::new()).await?;
    let batches = ctx
        .sql(
            "SELECT CAST(id AS VARCHAR) AS id_text, CAST(a AS VARCHAR) AS a_text, \
             encode(b, 'hex') AS b_text FROM ducklake.main.t ORDER BY id",
        )
        .await?
        .collect()
        .await?;
    assert_eq!(text_rows(&batches), expected);
    Ok(())
}

/// A session that lets filters reach the parquet reader's row filter, where
/// DataFusion drops the `FilterExec` above a scan that accepted them.
fn pushdown_config() -> SessionConfig {
    SessionConfig::new().set_bool("datafusion.execution.parquet.pushdown_filters", true)
}

/// Rows of `sql` from both engines: DuckDB over `lake.t`, this crate over
/// `ducklake.main.t` with filters pushed into the reader.
async fn both_engines(
    temp: &TempDir,
    duckdb_sql: &str,
    datafusion_sql: &str,
) -> anyhow::Result<(Vec<Vec<Option<String>>>, Vec<Vec<Option<String>>>)> {
    let conn = duckdb_lake(temp)?;
    let expected = duckdb_rows(&conn, &duckdb_sql.replace("{table}", "lake.t"))?;
    conn.execute("DETACH lake", [])?;
    drop(conn);
    let store = ReadRecordingStore::new();
    let ctx = session(temp, &store, pushdown_config()).await?;
    let batches = ctx
        .sql(&datafusion_sql.replace("{table}", "ducklake.main.t"))
        .await?
        .collect()
        .await?;
    Ok((expected, text_rows(&batches)))
}

/// A struct column that never evolved differs from the catalog only by the field
/// ids its file tags each child with. A filter on one of its children is
/// accepted by the reader's row filter, so it has to be evaluated there: the
/// `FilterExec` that would otherwise re-apply it is gone.
#[tokio::test]
async fn a_pushed_filter_on_a_plain_struct_child_keeps_only_matching_rows() -> anyhow::Result<()> {
    let temp = TempDir::new()?;
    let conn = duckdb_lake(&temp)?;
    conn.execute("CREATE TABLE lake.t (id INT, s STRUCT(x INT, y INT))", [])?;
    conn.execute(
        "INSERT INTO lake.t VALUES (1, {'x': 10, 'y': 11}), (2, {'x': 20, 'y': 21})",
        [],
    )?;
    conn.execute("DETACH lake", [])?;
    drop(conn);

    let (expected, got) = both_engines(
        &temp,
        "SELECT CAST(id AS VARCHAR), CAST(s['y'] AS VARCHAR) FROM {table} \
         WHERE s['x'] = 10 ORDER BY id",
        "SELECT CAST(id AS VARCHAR) AS id_text, CAST(s['y'] AS VARCHAR) AS y_text \
         FROM {table} WHERE s['x'] = 10 ORDER BY id",
    )
    .await?;
    assert_eq!(
        expected,
        vec![vec![Some("1".to_string()), Some("11".to_string())]]
    );
    assert_eq!(got, expected);
    Ok(())
}

/// A list column's file names its element `element`, where the catalog's type
/// names it `item`. A filter on the list is accepted by the reader's row filter,
/// so it has to be evaluated there.
#[tokio::test]
async fn a_pushed_filter_on_a_list_column_keeps_only_matching_rows() -> anyhow::Result<()> {
    let temp = TempDir::new()?;
    let conn = duckdb_lake(&temp)?;
    conn.execute("CREATE TABLE lake.t (id INT, l INT[])", [])?;
    conn.execute("INSERT INTO lake.t VALUES (1, [1, 2]), (2, [3])", [])?;
    conn.execute("DETACH lake", [])?;
    drop(conn);

    let (expected, got) = both_engines(
        &temp,
        "SELECT CAST(id AS VARCHAR), CAST(l AS VARCHAR) FROM {table} \
         WHERE list_contains(l, 3) ORDER BY id",
        "SELECT CAST(id AS VARCHAR) AS id_text, CAST(l AS VARCHAR) AS l_text \
         FROM {table} WHERE array_has(l, 3) ORDER BY id",
    )
    .await?;
    assert_eq!(
        expected,
        vec![vec![Some("2".to_string()), Some("[3]".to_string())]]
    );
    assert_eq!(got, expected);
    Ok(())
}

/// A column dropped and re-added under its old name gets a new column id, so a
/// file written before the drop stores a different column under that name. The
/// file's values must not reach the re-added column, nor its statistics, and the
/// column added in between reads its default there.
///
/// The dropped column's values lie outside the re-added column's on both sides,
/// so reading one for the other moves both bounds.
#[tokio::test]
async fn a_readded_column_reads_null_and_its_bounds_match_duckdb() -> anyhow::Result<()> {
    let temp = TempDir::new()?;
    let conn = duckdb_lake(&temp)?;
    conn.execute("CREATE TABLE lake.t (id INT, c INT)", [])?;
    conn.execute("INSERT INTO lake.t VALUES (1, 50), (2, 900)", [])?;
    conn.execute("ALTER TABLE lake.t ADD COLUMN d INT DEFAULT 42", [])?;
    conn.execute("ALTER TABLE lake.t DROP COLUMN c", [])?;
    conn.execute("ALTER TABLE lake.t ADD COLUMN c INT", [])?;
    conn.execute("INSERT INTO lake.t VALUES (3, 5, 100), (4, 6, 400)", [])?;
    conn.execute("DETACH lake", [])?;
    drop(conn);

    let sql = "SELECT CAST(min(c) AS VARCHAR) AS min_c, CAST(max(c) AS VARCHAR) AS max_c, \
               CAST(min(d) AS VARCHAR) AS min_d, CAST(max(d) AS VARCHAR) AS max_d FROM {table}";
    let (expected, got) = both_engines(&temp, sql, sql).await?;
    let text = |value: &str| Some(value.to_string());
    assert_eq!(
        expected,
        vec![vec![text("100"), text("400"), text("5"), text("42")]]
    );
    assert_eq!(got, expected);

    let rows = "SELECT CAST(id AS VARCHAR) AS id_text, CAST(c AS VARCHAR) AS c_text, \
                CAST(d AS VARCHAR) AS d_text FROM {table} ORDER BY id";
    let (expected, got) = both_engines(&temp, rows, rows).await?;
    assert_eq!(expected[0], vec![text("1"), None, text("42")]);
    assert_eq!(got, expected);
    Ok(())
}

/// The same evolution beside a column both files carry. With filters pushed into
/// the reader, the filter DataFusion derives from `min(c)` / `max(c)` while the
/// scan runs reaches every file, and each file must apply it to its own `c`: a
/// row holding a bound must never be skipped.
#[tokio::test]
async fn min_max_over_a_readded_column_keeps_rows_holding_the_bounds() -> anyhow::Result<()> {
    let temp = TempDir::new()?;
    let conn = duckdb_lake(&temp)?;
    conn.execute("CREATE TABLE lake.t (id INT, c INT, e VARCHAR)", [])?;
    conn.execute(
        "INSERT INTO lake.t VALUES (1, 100, 'e1'), (2, 200, 'e2')",
        [],
    )?;
    conn.execute("ALTER TABLE lake.t ADD COLUMN d INT DEFAULT 42", [])?;
    conn.execute("ALTER TABLE lake.t DROP COLUMN c", [])?;
    conn.execute("ALTER TABLE lake.t ADD COLUMN c INT", [])?;
    conn.execute(
        "INSERT INTO lake.t (id, e, d, c) VALUES (3, 'e3', 42, 100), (4, 'e4', 5, 400)",
        [],
    )?;
    conn.execute("DETACH lake", [])?;
    drop(conn);

    let sql = "SELECT CAST(min(c) AS VARCHAR) AS min_c, CAST(max(c) AS VARCHAR) AS max_c, \
               CAST(min(d) AS VARCHAR) AS min_d, CAST(max(d) AS VARCHAR) AS max_d FROM {table}";
    let (expected, got) = both_engines(&temp, sql, sql).await?;
    let text = |value: &str| Some(value.to_string());
    assert_eq!(
        expected,
        vec![vec![text("100"), text("400"), text("5"), text("42")]]
    );
    assert_eq!(got, expected);
    Ok(())
}

/// Rows of `sql` from both engines, this crate once with filters pushed into the
/// reader and once without.
async fn both_engines_both_pushdowns(
    temp: &TempDir,
    duckdb_sql: &str,
    datafusion_sql: &str,
) -> anyhow::Result<Vec<Vec<Option<String>>>> {
    let conn = duckdb_lake(temp)?;
    let expected = duckdb_rows(&conn, &duckdb_sql.replace("{table}", "lake.t"))?;
    conn.execute("DETACH lake", [])?;
    drop(conn);
    for (pushdown, config) in [(false, SessionConfig::new()), (true, pushdown_config())] {
        let store = ReadRecordingStore::new();
        let ctx = session(temp, &store, config).await?;
        let batches = ctx
            .sql(&datafusion_sql.replace("{table}", "ducklake.main.t"))
            .await?
            .collect()
            .await?;
        assert_eq!(
            text_rows(&batches),
            expected,
            "pushdown_filters = {pushdown}: {datafusion_sql}"
        );
    }
    Ok(expected)
}

/// A struct whose children evolved after its first file: `x` renamed to `xx`, and
/// `z` added. A filter on a child selects by the child's identity in every file,
/// whether or not the reader evaluates it.
#[tokio::test]
async fn a_filter_on_an_evolved_struct_child_matches_duckdb() -> anyhow::Result<()> {
    let temp = TempDir::new()?;
    let conn = duckdb_lake(&temp)?;
    conn.execute("CREATE TABLE lake.t (id INT, s STRUCT(x INT, y INT))", [])?;
    conn.execute(
        "INSERT INTO lake.t VALUES (1, {'x': 10, 'y': 11}), (2, {'x': 20, 'y': 21})",
        [],
    )?;
    conn.execute("ALTER TABLE lake.t RENAME COLUMN s.x TO xx", [])?;
    conn.execute("ALTER TABLE lake.t ADD COLUMN s.z INT", [])?;
    conn.execute(
        "INSERT INTO lake.t VALUES (3, {'xx': 30, 'y': 31, 'z': 32}), \
         (4, {'xx': 40, 'y': 41, 'z': 42})",
        [],
    )?;
    conn.execute("DETACH lake", [])?;
    drop(conn);

    let ids = |values: &[&str]| -> Vec<Vec<Option<String>>> {
        values.iter().map(|v| vec![Some(v.to_string())]).collect()
    };
    for (filter, expected) in [
        ("s['xx'] = 10", ids(&["1"])),
        ("s['xx'] = 30", ids(&["3"])),
        ("s['y'] = 21", ids(&["2"])),
        ("s['z'] = 42", ids(&["4"])),
        ("s['z'] IS NULL", ids(&["1", "2"])),
    ] {
        let got = both_engines_both_pushdowns(
            &temp,
            &format!("SELECT CAST(id AS VARCHAR) FROM {{table}} WHERE {filter} ORDER BY id"),
            &format!(
                "SELECT CAST(id AS VARCHAR) AS id_text FROM {{table}} WHERE {filter} ORDER BY id"
            ),
        )
        .await?;
        assert_eq!(got, expected, "{filter}");
    }
    Ok(())
}

/// A struct child added with a default takes that default in files written before
/// it, as a child and inside the whole struct, and a filter on it sees it there.
/// Where the struct holding the child is NULL the child is NULL, as official
/// DuckLake's `remap_struct` gives a defaulted child its parent's validity.
#[tokio::test]
async fn a_struct_child_added_with_a_default_reads_it_in_older_files() -> anyhow::Result<()> {
    let temp = TempDir::new()?;
    let conn = duckdb_lake(&temp)?;
    conn.execute(
        "CREATE TABLE lake.t (id INT, s STRUCT(x INT, n STRUCT(p INT)))",
        [],
    )?;
    conn.execute(
        "INSERT INTO lake.t VALUES (1, {'x': 1, 'n': {'p': 10}}), (2, NULL), \
         (4, {'x': 4, 'n': NULL})",
        [],
    )?;
    conn.execute("ALTER TABLE lake.t ADD COLUMN s.z INT DEFAULT 7", [])?;
    conn.execute(
        "INSERT INTO lake.t VALUES (3, {'x': 3, 'n': {'p': 30}, 'z': 9})",
        [],
    )?;
    conn.execute("DETACH lake", [])?;
    drop(conn);

    let text = |value: &str| Some(value.to_string());
    let children = both_engines_both_pushdowns(
        &temp,
        "SELECT CAST(id AS VARCHAR), CAST(s['z'] AS VARCHAR) FROM {table} ORDER BY id",
        "SELECT CAST(id AS VARCHAR) AS id_text, CAST(s['z'] AS VARCHAR) AS z_text \
         FROM {table} ORDER BY id",
    )
    .await?;
    assert_eq!(
        children,
        vec![
            vec![text("1"), text("7")],
            vec![text("2"), None],
            vec![text("3"), text("9")],
            vec![text("4"), text("7")]
        ]
    );
    for (filter, ids) in
        [("s['z'] = 7", vec!["1", "4"]), ("s['z'] IS NULL", vec!["2"]), ("s['z'] > 8", vec!["3"])]
    {
        let got = both_engines_both_pushdowns(
            &temp,
            &format!("SELECT CAST(id AS VARCHAR) FROM {{table}} WHERE {filter} ORDER BY id"),
            &format!(
                "SELECT CAST(id AS VARCHAR) AS id_text FROM {{table}} WHERE {filter} ORDER BY id"
            ),
        )
        .await?;
        assert_eq!(
            got,
            ids.iter().map(|id| vec![text(id)]).collect::<Vec<_>>(),
            "{filter}"
        );
    }
    let aggregates = both_engines_both_pushdowns(
        &temp,
        "SELECT CAST(count(s['z']) AS VARCHAR), CAST(sum(s['z']) AS VARCHAR) FROM {table}",
        "SELECT CAST(count(s['z']) AS VARCHAR) AS n, CAST(sum(s['z']) AS VARCHAR) AS total \
         FROM {table}",
    )
    .await?;
    assert_eq!(aggregates, vec![vec![text("3"), text("23")]]);

    // The whole struct, read without naming the child.
    for config in [SessionConfig::new(), pushdown_config()] {
        let store = ReadRecordingStore::new();
        let ctx = session(&temp, &store, config).await?;
        let batches = ctx
            .sql("SELECT id, s FROM ducklake.main.t ORDER BY id")
            .await?
            .collect()
            .await?;
        let mut z = Vec::new();
        for batch in &batches {
            let s = batch
                .column(1)
                .as_any()
                .downcast_ref::<arrow::array::StructArray>()
                .unwrap();
            let child = arrow::compute::cast(
                s.column_by_name("z").unwrap(),
                &arrow::datatypes::DataType::Utf8,
            )?;
            let child = child.as_any().downcast_ref::<StringArray>().unwrap();
            // The child array itself, not masked by the struct's validity: a
            // consumer reading the child directly must see NULL under a NULL struct.
            z.extend(
                (0..child.len())
                    .map(|row| (!child.is_null(row)).then(|| child.value(row).to_string())),
            );
        }
        assert_eq!(z, vec![text("7"), None, text("9"), text("7")]);
    }
    Ok(())
}

/// A defaulted grandchild, `s.n.r`, under a struct `n` that is NULL in some rows
/// and whose parent `s` is NULL in others. Official DuckLake's reader gives a
/// defaulted child the validity of the struct directly holding it
/// (`remap_struct`'s `RemapChildVectors`), and that struct is NULL wherever a
/// struct above it is: the grandchild is its default where `n` is valid, and NULL
/// otherwise. The expected rows are pinned from that rule rather than from
/// DuckDB's answer, which disagrees with itself across query forms at this depth.
#[tokio::test]
async fn a_defaulted_grandchild_is_null_under_a_null_struct() -> anyhow::Result<()> {
    let temp = TempDir::new()?;
    let conn = duckdb_lake(&temp)?;
    conn.execute(
        "CREATE TABLE lake.t (id INT, s STRUCT(x INT, n STRUCT(p INT)))",
        [],
    )?;
    conn.execute(
        "INSERT INTO lake.t VALUES (1, {'x': 1, 'n': {'p': 10}}), (2, NULL), \
         (4, {'x': 4, 'n': NULL})",
        [],
    )?;
    conn.execute("ALTER TABLE lake.t ADD COLUMN s.n.r INT DEFAULT 5", [])?;
    conn.execute(
        "INSERT INTO lake.t VALUES (3, {'x': 3, 'n': {'p': 30, 'r': 33}})",
        [],
    )?;
    conn.execute("DETACH lake", [])?;
    drop(conn);

    let text = |value: &str| Some(value.to_string());
    for config in [SessionConfig::new(), pushdown_config()] {
        let store = ReadRecordingStore::new();
        let ctx = session(&temp, &store, config).await?;
        let rows = text_rows(
            &ctx.sql(
                "SELECT CAST(id AS VARCHAR) AS id_text, CAST(s['n']['r'] AS VARCHAR) AS r_text \
                 FROM ducklake.main.t ORDER BY id",
            )
            .await?
            .collect()
            .await?,
        );
        assert_eq!(
            rows,
            vec![
                vec![text("1"), text("5")],
                vec![text("2"), None],
                vec![text("3"), text("33")],
                vec![text("4"), None]
            ]
        );
        let filtered = text_rows(
            &ctx.sql(
                "SELECT CAST(id AS VARCHAR) AS id_text FROM ducklake.main.t \
                 WHERE s['n']['r'] = 5 ORDER BY id",
            )
            .await?
            .collect()
            .await?,
        );
        assert_eq!(filtered, vec![vec![text("1")]]);
    }
    Ok(())
}

/// With filters pushed into the reader, a predicate the reader's row filter may
/// decline in some file stays in a `FilterExec` above the scan, while one it is
/// guaranteed to apply is left to the reader.
#[tokio::test]
async fn a_filter_the_reader_may_skip_stays_above_the_scan() -> anyhow::Result<()> {
    let temp = TempDir::new()?;
    let conn = duckdb_lake(&temp)?;
    conn.execute(
        "CREATE TABLE lake.t (id INT, s STRUCT(x INT, m MAP(VARCHAR, INT)))",
        [],
    )?;
    conn.execute(
        "INSERT INTO lake.t VALUES (1, {'x': 1, 'm': MAP {'k': 13}}), \
         (2, {'x': 2, 'm': MAP {'k': 14}})",
        [],
    )?;
    conn.execute("DETACH lake", [])?;
    drop(conn);

    let store = ReadRecordingStore::new();
    let ctx = session(&temp, &store, pushdown_config()).await?;
    let plan = |sql: &'static str| {
        let ctx = ctx.clone();
        async move {
            let plan = physical_plan(&ctx, sql).await?;
            anyhow::Ok(displayable(plan.as_ref()).indent(true).to_string())
        }
    };
    let through_map = plan("SELECT id FROM ducklake.main.t WHERE s['m']['k'] = 13").await?;
    assert!(
        through_map.contains("FilterExec"),
        "a lookup through a map must stay above the scan:\n{through_map}"
    );
    let through_structs = plan("SELECT id FROM ducklake.main.t WHERE s['x'] = 1").await?;
    assert!(
        !through_structs.contains("FilterExec"),
        "a struct child the reader always applies is left to it:\n{through_structs}"
    );
    Ok(())
}
