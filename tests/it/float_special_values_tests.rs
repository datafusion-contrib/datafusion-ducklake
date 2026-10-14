//! NaN, infinities and negative zero through every write path, on every
//! backend, against official DuckLake (DuckDB 1.5.5).
//!
//! One matrix: each write path (inlined insert, Parquet insert, flush,
//! `merge_adjacent_files`, `UPDATE` of inlined rows, `UPDATE` of Parquet rows)
//! applies the same history with official DuckLake to a SQLite catalog and
//! with this crate to a catalog on each backend. Then
//!
//! - this crate's answers to [`QUERIES`] over its catalog,
//! - DuckDB's answers over the crate's catalog, where DuckDB can attach it,
//! - the table-wide and per-file statistics of `d` and `f` (`contains_nan`
//!   and the bounds)
//!
//! must equal official's, except where a path commits a Parquet file holding
//! only NaN (see `WritePath::nan_only_file`): official narrows the column's
//! table-wide bounds after it and this crate stores no row, so the crate's
//! table-wide statistics must then be empty. One kind of cell is known to differ, from the
//! Parquet footer statistics arrow-rs writes, which are out of scope here:
//! DuckDB over a Parquet file this crate wrote may prune its NaN rows, as the
//! footer leaves NaN out of min/max ([`KNOWN`] lists those queries). The run
//! reports those cells rather than failing. The catalog's per-file bounds keep
//! the sign of the zero written first, as DuckDB's writer does, so they are
//! compared exactly.

#![cfg(all(feature = "write", feature = "metadata-duckdb"))]

use std::path::Path;
use std::sync::Arc;

use datafusion::prelude::SessionContext;
use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeTable, MergeOptions, MetadataProvider, MetadataWriter,
    execute_ducklake_sql,
};

use crate::inlined_mutation_differential_tests::{
    Metadata, attach_lake, duckdb_connection, official, query_lines,
};

/// The rows every path writes: `(id, d, f)`.
const ROWS: [(i32, &str); 6] =
    [(1, "'NaN'"), (2, "'inf'"), (3, "'-inf'"), (4, "'-0.0'"), (5, "'1.5'"), (6, "NULL")];

fn value_sql(text: &str) -> (String, String) {
    if text == "NULL" {
        ("NULL".to_string(), "NULL".to_string())
    } else {
        (
            format!("CAST({text} AS DOUBLE)"),
            format!("CAST({text} AS FLOAT)"),
        )
    }
}

fn insert(ids: std::ops::RangeInclusive<i32>) -> String {
    let rows = ROWS
        .iter()
        .filter(|(id, _)| ids.contains(id))
        .map(|(id, text)| {
            let (d, f) = value_sql(text);
            format!("({id}, {d}, {f})")
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!("INSERT INTO lake.main.t VALUES {rows}")
}

fn insert_only(ids: &[i32]) -> String {
    let rows = ROWS
        .iter()
        .filter(|(id, _)| ids.contains(id))
        .map(|(id, text)| {
            let (d, f) = value_sql(text);
            format!("({id}, {d}, {f})")
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!("INSERT INTO lake.main.t VALUES {rows}")
}

fn zeros() -> String {
    let rows = ROWS
        .iter()
        .map(|(id, _)| format!("({id}, CAST(0.25 AS DOUBLE), CAST(0.25 AS FLOAT))"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("INSERT INTO lake.main.t VALUES {rows}")
}

fn updates() -> Vec<Step> {
    ROWS.iter()
        .map(|(id, text)| {
            let (d, f) = value_sql(text);
            Step::Sql(format!(
                "UPDATE lake.main.t SET d = {d}, f = {f} WHERE id = {id}"
            ))
        })
        .collect()
}

const QUERIES: [&str; 10] = [
    "SELECT id, d, f FROM lake.main.t ORDER BY id",
    "SELECT count(*) FROM lake.main.t WHERE d > 1",
    "SELECT count(*) FROM lake.main.t WHERE d > 2",
    "SELECT count(*) FROM lake.main.t WHERE f > 2",
    "SELECT count(*) FROM lake.main.t WHERE d = 'NaN'",
    "SELECT count(*) FROM lake.main.t WHERE f = 'NaN'",
    "SELECT count(*) FROM lake.main.t WHERE f < 0",
    "SELECT count(*) FROM lake.main.t WHERE d IS NULL",
    "SELECT min(d), max(d), min(f), max(f), count(*) FROM lake.main.t",
    "SELECT id FROM lake.main.t WHERE d = 0 ORDER BY id",
];

/// Queries DuckDB is known to answer differently over Parquet files this crate
/// wrote: NaN rows the footer statistics leave out of min/max.
const KNOWN: [&str; 4] = [
    "SELECT count(*) FROM lake.main.t WHERE d > 2",
    "SELECT count(*) FROM lake.main.t WHERE f > 2",
    "SELECT count(*) FROM lake.main.t WHERE d = 'NaN'",
    "SELECT count(*) FROM lake.main.t WHERE f = 'NaN'",
];

#[derive(Clone)]
enum Step {
    /// A statement both run.
    Sql(String),
    /// `merge_adjacent_files` / `ducklake_merge_adjacent_files`.
    Merge,
}

struct WritePath {
    name: &'static str,
    /// Whether the rows end in Parquet files this crate wrote.
    parquet: bool,
    limit: usize,
    steps: Vec<Step>,
    /// Whether a write commits a Parquet file holding only NaN to a column
    /// with table-wide bounds. The file's statistics have no bounds; official
    /// stores a table-wide row with neither and narrows it to the next write's
    /// bounds, while this crate leaves the column with no row (see
    /// COMPATIBILITY.md). The crate's table-wide statistics are then empty.
    nan_only_file: bool,
}

fn paths() -> Vec<WritePath> {
    let flush = Step::Sql("CALL ducklake_flush_inlined_data('lake')".to_string());
    vec![
        WritePath {
            name: "inlined",
            parquet: false,
            limit: 10,
            nan_only_file: false,
            steps: vec![Step::Sql(insert(1..=6))],
        },
        WritePath {
            name: "parquet",
            parquet: true,
            limit: 0,
            nan_only_file: false,
            steps: vec![Step::Sql(insert(1..=6))],
        },
        WritePath {
            name: "parquet, no infinity",
            parquet: true,
            limit: 0,
            nan_only_file: false,
            steps: vec![Step::Sql(insert_only(&[1, 4, 5, 6]))],
        },
        WritePath {
            name: "flush",
            parquet: true,
            limit: 10,
            nan_only_file: false,
            steps: vec![Step::Sql(insert(1..=6)), flush],
        },
        WritePath {
            name: "merge",
            parquet: true,
            limit: 0,
            nan_only_file: false,
            steps: vec![Step::Sql(insert(1..=3)), Step::Sql(insert(4..=6)), Step::Merge],
        },
        WritePath {
            name: "update inlined",
            parquet: false,
            limit: 10,
            nan_only_file: false,
            steps: std::iter::once(Step::Sql(zeros()))
                .chain(updates())
                .collect(),
        },
        WritePath {
            name: "update parquet",
            parquet: true,
            limit: 0,
            nan_only_file: true,
            steps: std::iter::once(Step::Sql(zeros()))
                .chain(updates())
                .collect(),
        },
    ]
}

/// The crate's writer and provider for one backend.
pub(crate) struct Side {
    pub(crate) writer: Arc<dyn MetadataWriter>,
    pub(crate) provider: Arc<dyn MetadataProvider>,
}

fn table_id(side: &Side) -> i64 {
    let snapshot = side.provider.get_current_snapshot().unwrap();
    let schema = side
        .provider
        .get_schema_by_name("main", snapshot)
        .unwrap()
        .unwrap();
    side.provider
        .get_table_by_name(schema.schema_id, "t", snapshot)
        .unwrap()
        .unwrap()
        .table_id
}

async fn crate_sql(side: &Side, statement: &str) -> Result<(), String> {
    let ctx = SessionContext::new();
    let catalog =
        DuckLakeCatalog::with_writer(Arc::clone(&side.provider), Arc::clone(&side.writer)).unwrap();
    DuckLakeCatalog::register(&ctx, "lake", catalog);
    let handle =
        DuckLakeCatalog::with_writer(Arc::clone(&side.provider), Arc::clone(&side.writer)).unwrap();
    execute_ducklake_sql(&ctx, &handle, statement)
        .await
        .map_err(|e| format!("crate `{statement}`: {e}"))?
        .collect()
        .await
        .map_err(|e| format!("crate `{statement}`: {e}"))?;
    Ok(())
}

async fn crate_merge(side: &Side) -> Result<(), String> {
    let ctx = SessionContext::new();
    let catalog =
        DuckLakeCatalog::with_writer(Arc::clone(&side.provider), Arc::clone(&side.writer)).unwrap();
    ctx.register_catalog("lake", Arc::new(catalog));
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
        .expect("provider is a DuckLakeTable")
        .clone();
    let merged = table
        .merge_adjacent_files(
            &ctx.state(),
            MergeOptions {
                min_file_size: 0,
                ..MergeOptions::default()
            },
        )
        .await
        .map_err(|e| format!("merge: {e}"))?;
    if !merged.did_work() {
        return Err("merge: nothing merged".to_string());
    }
    Ok(())
}

async fn apply_crate(side: &Side, path: &WritePath) -> Result<(), String> {
    // `CREATE TABLE` needs the schema; a first write to another table creates
    // it as each backend lays schemas out.
    let seed = arrow::record_batch::RecordBatch::try_new(
        Arc::new(arrow::datatypes::Schema::new(vec![
            arrow::datatypes::Field::new("id", arrow::datatypes::DataType::Int32, true),
        ])),
        vec![Arc::new(arrow::array::Int32Array::from(vec![0]))],
    )
    .unwrap();
    datafusion_ducklake::DuckLakeTableWriter::new(
        Arc::clone(&side.writer),
        Arc::new(object_store::local::LocalFileSystem::new()),
    )
    .unwrap()
    .append_table("main", "seed", &[seed])
    .await
    .map_err(|e| format!("seed: {e}"))?;
    crate_sql(
        side,
        "CREATE TABLE lake.main.t (id INTEGER, d DOUBLE, f FLOAT)",
    )
    .await?;
    side.writer
        .set_table_setting(
            table_id(side),
            "data_inlining_row_limit",
            &path.limit.to_string(),
        )
        .unwrap();
    for step in &path.steps {
        match step {
            Step::Sql(sql) => crate_sql(side, sql).await?,
            Step::Merge => crate_merge(side).await?,
        }
    }
    Ok(())
}

fn official_statements(path: &WritePath) -> Vec<String> {
    let mut out = vec![
        "CREATE TABLE lake.main.t (id INTEGER, d DOUBLE, f FLOAT)".to_string(),
        format!(
            "CALL lake.set_option('data_inlining_row_limit', {}, table_name => 't')",
            path.limit
        ),
    ];
    for step in &path.steps {
        out.push(match step {
            Step::Sql(sql) => sql.clone(),
            Step::Merge => "CALL ducklake_merge_adjacent_files('lake')".to_string(),
        });
    }
    out
}

fn normalize(lines: Vec<String>) -> Vec<String> {
    lines.into_iter().map(|line| line.to_lowercase()).collect()
}

async fn crate_answers(side: &Side) -> Vec<Vec<String>> {
    let ctx = SessionContext::new();
    let snapshot = side.provider.get_current_snapshot().unwrap();
    let catalog = DuckLakeCatalog::with_snapshot(Arc::clone(&side.provider), snapshot).unwrap();
    ctx.register_catalog("lake", Arc::new(catalog));
    let mut out = Vec::new();
    for sql in QUERIES {
        let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
        let mut lines = Vec::new();
        for batch in &batches {
            for row in 0..batch.num_rows() {
                lines.push(
                    (0..batch.num_columns())
                        .map(|c| {
                            let array = batch.column(c);
                            if arrow::array::Array::is_null(array, row) {
                                "NULL".to_string()
                            } else {
                                arrow::util::display::array_value_to_string(array, row).unwrap()
                            }
                        })
                        .collect::<Vec<_>>()
                        .join(","),
                );
            }
        }
        out.push(normalize(lines));
    }
    out
}

fn duckdb_answers(metadata: &Metadata, data: &Path) -> Vec<Vec<String>> {
    let connection = duckdb_connection();
    attach_lake(&connection, metadata, data, true);
    QUERIES
        .iter()
        .map(|sql| normalize(query_lines(&connection, sql)))
        .collect()
}

/// The statistics of `d` and `f`: the table-wide rows, then each live file's,
/// sorted, as `column, contains_nan, min, max`.
fn statistics(metadata: &Metadata) -> (Vec<String>, Vec<String>) {
    let connection = duckdb_connection();
    metadata.attach_raw(&connection);
    let table = "(SELECT table_id FROM m.ducklake_table \
                  WHERE table_name = 't' AND end_snapshot IS NULL)";
    let name = "(SELECT c.column_name FROM m.ducklake_column c \
                 WHERE c.table_id = s.table_id AND c.column_id = s.column_id \
                 ORDER BY c.begin_snapshot DESC LIMIT 1)";
    let bool_text = |line: String| line.replace("false", "0").replace("true", "1");
    let mut table_rows = query_lines(
        &connection,
        &format!(
            "SELECT {name}, s.contains_nan, s.min_value, s.max_value \
             FROM m.ducklake_table_column_stats s WHERE s.table_id = {table}"
        ),
    )
    .into_iter()
    .map(bool_text)
    .filter(|line| !line.starts_with("id,"))
    .collect::<Vec<_>>();
    table_rows.sort();
    let mut file_rows = query_lines(
        &connection,
        &format!(
            "SELECT {name}, s.contains_nan, s.min_value, s.max_value \
             FROM m.ducklake_file_column_stats s \
             JOIN m.ducklake_data_file f ON f.data_file_id = s.data_file_id \
             WHERE s.table_id = {table} AND f.end_snapshot IS NULL"
        ),
    )
    .into_iter()
    .map(bool_text)
    .filter(|line| !line.starts_with("id,"))
    .collect::<Vec<_>>();
    file_rows.sort();
    (table_rows, file_rows)
}

/// One backend's catalog for one path.
pub(crate) struct Backend<'a> {
    pub(crate) name: &'static str,
    pub(crate) side: Side,
    /// The crate's catalog for DuckDB to read, or `None` when DuckDB 1.5.5
    /// cannot attach this layout.
    pub(crate) duckdb_reads: Option<Metadata>,
    /// The crate's metadata, for its statistics.
    pub(crate) raw: Metadata,
    pub(crate) data: &'a Path,
    /// Whether the backend stores these values inline: the single-catalog
    /// PostgreSQL writer never inlines, and MySQL writes NaN and infinities to
    /// Parquet.
    pub(crate) inlines: bool,
    /// Run between the crate's writes and DuckDB's reads.
    pub(crate) before_duckdb: Option<std::pin::Pin<Box<dyn std::future::Future<Output = ()>>>>,
}

/// Run `path` on `backend` and official, returning `(passing cells, known
/// failing cells, failures)`.
async fn run_cell(
    path: &WritePath,
    dir: &Path,
    backend: Backend<'_>,
) -> (Vec<String>, Vec<String>, Vec<String>) {
    let official_metadata = Metadata::Sqlite(dir.join("official.db"));
    let official_data = dir.join("official_data");
    std::fs::create_dir_all(&official_data).unwrap();
    official(
        &official_metadata,
        &official_data,
        &official_statements(path),
    );
    let expected = duckdb_answers(&official_metadata, &official_data);
    let (official_table, official_files) = statistics(&official_metadata);

    let cell = format!("{} / {}", backend.name, path.name);
    let mut passing = Vec::new();
    let mut known = Vec::new();
    let mut failures = Vec::new();
    if let Err(error) = apply_crate(&backend.side, path).await {
        // A path the backend does not offer (the single-catalog PostgreSQL
        // writer has no UPDATE or compaction) is reported, not failed.
        if error.contains("not supported") {
            eprintln!("SKIPPED {cell}: {error}");
        } else {
            failures.push(format!("{cell}: {error}"));
        }
        return (passing, known, failures);
    }
    let got = crate_answers(&backend.side).await;
    for ((sql, want), got) in QUERIES.iter().zip(&expected).zip(&got) {
        if want == got {
            passing.push(format!("{cell}: crate `{sql}`"));
        } else {
            failures.push(format!(
                "{cell}: the crate answers `{sql}` with {got:?}, official {want:?}"
            ));
        }
    }
    drop(backend.side);
    if let Some(before) = backend.before_duckdb {
        before.await;
    }
    if let Some(metadata) = &backend.duckdb_reads {
        let got = duckdb_answers(metadata, backend.data);
        for ((sql, want), got) in QUERIES.iter().zip(&expected).zip(&got) {
            let is_known = path.parquet && KNOWN.contains(sql);
            match (want == got, is_known) {
                (true, false) => passing.push(format!("{cell}: DuckDB `{sql}`")),
                (true, true) => passing.push(format!(
                    "{cell}: DuckDB `{sql}` (listed as known failing, now passes)"
                )),
                (false, true) => known.push(format!(
                    "{cell}: DuckDB answers `{sql}` with {got:?}, official {want:?} \
                     (Parquet footer statistics leave NaN out)"
                )),
                (false, false) => failures.push(format!(
                    "{cell}: DuckDB answers `{sql}` with {got:?} over the crate's catalog, \
                     official {want:?}"
                )),
            }
        }
    }
    if !backend.inlines && path.limit > 0 {
        // The backend writes these rows to Parquet, so its statistics are a
        // Parquet file's, not the inlined rows' official records.
        eprintln!("SKIPPED {cell}: statistics (the backend does not inline these values)");
        return (passing, known, failures);
    }
    let (table, files) = statistics(&backend.raw);
    let official_table = if path.nan_only_file {
        Vec::new()
    } else {
        official_table
    };
    for (what, got, want) in
        [("table-wide", table, official_table), ("per-file", files, official_files)]
    {
        if got == want {
            passing.push(format!("{cell}: {what} statistics"));
        } else {
            failures.push(format!(
                "{cell}: {what} statistics {got:?}, official {want:?}"
            ));
        }
    }
    (passing, known, failures)
}

/// Run every path on the backend `open` builds in a fresh directory, and
/// require no failure beyond [`KNOWN`].
async fn run_backend<F, Fut>(name: &'static str, inlines: bool, mut open: F)
where
    F: FnMut(&'static str, usize, std::path::PathBuf) -> Fut,
    Fut: std::future::Future<Output = (Side, Option<Metadata>, Metadata, BeforeDuckdb)>,
{
    let mut passing = Vec::new();
    let mut known = Vec::new();
    let mut failures = Vec::new();
    for (index, path) in paths().into_iter().enumerate() {
        let temp = tempfile::TempDir::new().unwrap();
        let dir = temp.path();
        let data = dir.join("crate_data");
        std::fs::create_dir_all(&data).unwrap();
        let (side, duckdb_reads, raw, before_duckdb) =
            open(path.name, index, dir.to_path_buf()).await;
        let (p, k, f) = run_cell(
            &path,
            dir,
            Backend {
                name,
                side,
                duckdb_reads,
                raw,
                data: &data,
                inlines,
                before_duckdb,
            },
        )
        .await;
        passing.extend(p);
        known.extend(k);
        failures.extend(f);
    }
    eprintln!(
        "{name}: {} cells pass, {} known failing, {} failing",
        passing.len(),
        known.len(),
        failures.len()
    );
    for line in &known {
        eprintln!("KNOWN FAILING {line}");
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

type BeforeDuckdb = Option<std::pin::Pin<Box<dyn std::future::Future<Output = ()>>>>;

#[cfg(feature = "write-sqlite")]
#[tokio::test(flavor = "multi_thread")]
async fn sqlite() {
    use datafusion_ducklake::{SqliteMetadataProvider, SqliteMetadataWriter};
    run_backend("sqlite", true, |_, _, dir| async move {
        let path = dir.join("crate.db");
        let url = format!("sqlite:{}?mode=rwc", path.display());
        let writer = SqliteMetadataWriter::new_with_init(&url).await.unwrap();
        writer
            .set_data_path(dir.join("crate_data").to_str().unwrap())
            .unwrap();
        let side = Side {
            writer: Arc::new(writer),
            provider: Arc::new(SqliteMetadataProvider::new(&url).await.unwrap()),
        };
        let metadata = Metadata::Sqlite(path.clone());
        let before: BeforeDuckdb = Some(Box::pin(async move {
            crate::inlined_mutation_differential_tests::make_sqlite_catalog_duckdb_readable(&path)
                .await;
        }));
        (side, Some(metadata.clone()), metadata, before)
    })
    .await;
}

#[cfg(feature = "write-duckdb")]
#[tokio::test(flavor = "multi_thread")]
async fn duckdb() {
    use datafusion_ducklake::DuckdbMetadataWriter;
    run_backend("duckdb", true, |_, _, dir| async move {
        let path = dir.join("crate.duckdb");
        let writer = DuckdbMetadataWriter::new_with_init(path.display().to_string()).unwrap();
        writer
            .set_data_path(dir.join("crate_data").to_str().unwrap())
            .unwrap();
        let provider = writer.metadata_provider();
        let side = Side {
            writer: Arc::new(writer),
            provider: Arc::new(provider),
        };
        let metadata = Metadata::Duckdb(path);
        (side, Some(metadata.clone()), metadata, None)
    })
    .await;
}

#[cfg(feature = "write-mysql")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
#[tokio::test(flavor = "multi_thread")]
async fn mysql() {
    use datafusion_ducklake::{MySqlMetadataProvider, MySqlMetadataWriter};
    use testcontainers::runners::AsyncRunner;
    use testcontainers_modules::mysql::Mysql;
    let container = Mysql::default().start().await.unwrap();
    let port = container.get_host_port_ipv4(3306).await.unwrap();
    let admin = sqlx::MySqlPool::connect(&format!("mysql://root@127.0.0.1:{port}/test"))
        .await
        .unwrap();
    run_backend("mysql", false, |_, index, dir| {
        let admin = admin.clone();
        async move {
            let database = format!("floats_{index}");
            sqlx::query(sqlx::AssertSqlSafe(format!("CREATE DATABASE {database}")))
                .execute(&admin)
                .await
                .unwrap();
            let url = format!("mysql://root@127.0.0.1:{port}/{database}");
            let writer = MySqlMetadataWriter::new_with_init(&url).await.unwrap();
            writer
                .set_data_path(dir.join("crate_data").to_str().unwrap())
                .unwrap();
            let side = Side {
                writer: Arc::new(writer),
                provider: Arc::new(MySqlMetadataProvider::new(&url).await.unwrap()),
            };
            // DuckDB 1.5.5 cannot attach this crate's MySQL catalogs (their
            // `ducklake_snapshot` lacks `next_catalog_id`).
            let metadata = Metadata::Mysql {
                host: "127.0.0.1".to_string(),
                port,
                database,
            };
            (side, None, metadata, None)
        }
    })
    .await;
}

#[cfg(feature = "write-postgres")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
#[tokio::test(flavor = "multi_thread")]
async fn postgres_single_catalog() {
    use datafusion_ducklake::{PostgresMetadataProvider, PostgresSingleCatalogMetadataWriter};
    use testcontainers::runners::AsyncRunner;
    use testcontainers_modules::postgres::Postgres;
    let container = Postgres::default().start().await.unwrap();
    let port = container.get_host_port_ipv4(5432).await.unwrap();
    let admin = sqlx::PgPool::connect(&format!(
        "postgresql://postgres:postgres@127.0.0.1:{port}/postgres"
    ))
    .await
    .unwrap();
    run_backend("postgres single catalog", false, |_, index, dir| {
        let admin = admin.clone();
        async move {
            let database = format!("floats_{index}");
            sqlx::query(sqlx::AssertSqlSafe(format!("CREATE DATABASE {database}")))
                .execute(&admin)
                .await
                .unwrap();
            let url = format!("postgresql://postgres:postgres@127.0.0.1:{port}/{database}");
            let writer = PostgresSingleCatalogMetadataWriter::new_with_init(&url)
                .await
                .unwrap();
            writer
                .set_data_path(dir.join("crate_data").to_str().unwrap())
                .unwrap();
            let side = Side {
                writer: Arc::new(writer),
                provider: Arc::new(PostgresMetadataProvider::new(&url).await.unwrap()),
            };
            // As for MySQL, DuckDB cannot attach this crate's PostgreSQL catalogs.
            let metadata = Metadata::Postgres(format!(
                "host=127.0.0.1 port={port} user=postgres password=postgres dbname={database}"
            ));
            (side, None, metadata, None)
        }
    })
    .await;
}

#[cfg(feature = "multicatalog-postgres")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
#[tokio::test(flavor = "multi_thread")]
async fn multicatalog() {
    use datafusion_ducklake::{MulticatalogManager, MulticatalogProvider, PostgresMetadataWriter};
    use testcontainers::runners::AsyncRunner;
    use testcontainers_modules::postgres::Postgres;
    let container = Postgres::default().start().await.unwrap();
    let port = container.get_host_port_ipv4(5432).await.unwrap();
    let admin = sqlx::PgPool::connect(&format!(
        "postgresql://postgres:postgres@127.0.0.1:{port}/postgres"
    ))
    .await
    .unwrap();
    run_backend("multicatalog", true, |_, index, dir| {
        let admin = admin.clone();
        async move {
            let database = format!("floats_{index}");
            sqlx::query(sqlx::AssertSqlSafe(format!("CREATE DATABASE {database}")))
                .execute(&admin)
                .await
                .unwrap();
            let url = format!("postgresql://postgres:postgres@127.0.0.1:{port}/{database}");
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
            writer
                .set_data_path(dir.join("crate_data").to_str().unwrap())
                .unwrap();
            let side = Side {
                writer: Arc::new(writer),
                provider: Arc::new(MulticatalogProvider::with_pool(pool, "lake").await.unwrap()),
            };
            // The multi-catalog layout has no official reader.
            let metadata = Metadata::Postgres(format!(
                "host=127.0.0.1 port={port} user=postgres password=postgres dbname={database}"
            ));
            (side, None, metadata, None)
        }
    })
    .await;
}

/// An inlined data table an earlier release of this crate created declares its
/// float columns `DOUBLE`, where SQLite turns a NaN into NULL. Its rows still
/// read, and a later write of NaN and `-0.0` to it keeps them, readable by this
/// crate and by DuckDB.
#[cfg(feature = "write-sqlite")]
#[tokio::test(flavor = "multi_thread")]
async fn sqlite_double_inlined_table_from_an_earlier_release() {
    use datafusion_ducklake::{SqliteMetadataProvider, SqliteMetadataWriter};
    let temp = tempfile::TempDir::new().unwrap();
    let path = temp.path().join("crate.db");
    let data = temp.path().join("crate_data");
    std::fs::create_dir_all(&data).unwrap();
    let url = format!("sqlite:{}?mode=rwc", path.display());
    let writer = SqliteMetadataWriter::new_with_init(&url).await.unwrap();
    writer.set_data_path(data.to_str().unwrap()).unwrap();
    let side = Side {
        writer: Arc::new(writer),
        provider: Arc::new(SqliteMetadataProvider::new(&url).await.unwrap()),
    };
    apply_crate(
        &side,
        &WritePath {
            name: "legacy",
            parquet: false,
            limit: 10,
            nan_only_file: false,
            steps: vec![Step::Sql(insert_only(&[2, 3, 5, 6]))],
        },
    )
    .await
    .unwrap();

    // Rebuild the inlined table as earlier releases declared it.
    let pool = sqlx::SqlitePool::connect(&url).await.unwrap();
    let name: String = sqlx::query_scalar(
        "SELECT d.table_name FROM ducklake_inlined_data_tables d \
         JOIN ducklake_table t ON t.table_id = d.table_id WHERE t.table_name = 't'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    let real = |column: &str| {
        format!(
            "CASE {column} WHEN 'inf' THEN 9e999 WHEN '-inf' THEN -9e999 \
             ELSE CAST({column} AS REAL) END"
        )
    };
    let mut tx = pool.begin().await.unwrap();
    for statement in [
        "CREATE TABLE legacy (row_id BIGINT, begin_snapshot BIGINT, end_snapshot BIGINT, \
         id BIGINT, d DOUBLE, f DOUBLE)"
            .to_string(),
        format!(
            "INSERT INTO legacy SELECT row_id, begin_snapshot, end_snapshot, id, {}, {} \
             FROM \"{name}\"",
            real("d"),
            real("f")
        ),
        format!("DROP TABLE \"{name}\""),
        format!("ALTER TABLE legacy RENAME TO \"{name}\""),
    ] {
        sqlx::query(sqlx::AssertSqlSafe(statement))
            .execute(&mut *tx)
            .await
            .unwrap();
    }
    tx.commit().await.unwrap();
    let declared = |pool: sqlx::SqlitePool, name: String| async move {
        sqlx::query_as::<_, (String, String)>(
            "SELECT name, type FROM pragma_table_info(?) WHERE name IN ('d', 'f') ORDER BY name",
        )
        .bind(name)
        .fetch_all(&pool)
        .await
        .unwrap()
    };
    assert_eq!(
        declared(pool.clone(), name.clone()).await,
        vec![("d".to_string(), "DOUBLE".to_string()), ("f".to_string(), "DOUBLE".to_string())]
    );
    assert_eq!(
        crate_answers(&side).await.remove(0),
        vec!["2,inf,inf", "3,-inf,-inf", "5,1.5,1.5", "6,null,null"],
        "the earlier release's rows read"
    );

    crate_sql(&side, &insert_only(&[1, 4])).await.unwrap();
    let expected =
        vec!["1,nan,nan", "2,inf,inf", "3,-inf,-inf", "4,-0.0,-0.0", "5,1.5,1.5", "6,null,null"];
    assert_eq!(
        crate_answers(&side).await.remove(0),
        expected,
        "the crate reads the new rows"
    );
    assert_eq!(
        declared(pool.clone(), name.clone()).await,
        vec![("d".to_string(), "VARCHAR".to_string()), ("f".to_string(), "VARCHAR".to_string())],
        "the write converts the table to official's layout"
    );
    pool.close().await;
    drop(side);

    crate::inlined_mutation_differential_tests::make_sqlite_catalog_duckdb_readable(&path).await;
    assert_eq!(
        duckdb_answers(&Metadata::Sqlite(path.clone()), &data).remove(0),
        expected,
        "DuckDB reads every row"
    );
}

/// Converting an earlier release's inlined table keeps every double and float
/// it holds exactly, including those SQLite's own float-to-text conversions
/// round wrongly.
#[cfg(feature = "write-sqlite")]
#[tokio::test(flavor = "multi_thread")]
async fn sqlite_legacy_inlined_floats_convert_exactly() {
    use datafusion_ducklake::{SqliteMetadataProvider, SqliteMetadataWriter};
    let temp = tempfile::TempDir::new().unwrap();
    let path = temp.path().join("crate.db");
    let data = temp.path().join("crate_data");
    std::fs::create_dir_all(&data).unwrap();
    let url = format!("sqlite:{}?mode=rwc", path.display());
    let writer = SqliteMetadataWriter::new_with_init(&url).await.unwrap();
    writer.set_data_path(data.to_str().unwrap()).unwrap();
    let side = Side {
        writer: Arc::new(writer),
        provider: Arc::new(SqliteMetadataProvider::new(&url).await.unwrap()),
    };
    apply_crate(
        &side,
        &WritePath {
            name: "legacy",
            parquet: false,
            limit: 10,
            nan_only_file: false,
            steps: vec![Step::Sql(insert_only(&[2, 3, 5, 6]))],
        },
    )
    .await
    .unwrap();
    let pool = sqlx::SqlitePool::connect(&url).await.unwrap();
    let name: String = sqlx::query_scalar(
        "SELECT d.table_name FROM ducklake_inlined_data_tables d \
         JOIN ducklake_table t ON t.table_id = d.table_id WHERE t.table_name = 't'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    // `printf('%!.17g')` of SQLite 3.46 spells each of these doubles as text
    // that parses to another one.
    let doubles = [
        (2, -5.303182573328798e307_f64),
        (3, 7.597954524892738e-84),
        (5, -3.0797526843217145e-217),
        (6, 1.2112294575482691e302),
    ];
    let floats = [(2, 3.4028235e38_f32), (3, 1.0e-45), (5, -0.1), (6, 16777217.0)];
    let mut tx = pool.begin().await.unwrap();
    for statement in [
        "CREATE TABLE legacy (row_id BIGINT, begin_snapshot BIGINT, end_snapshot BIGINT, \
         id BIGINT, d DOUBLE, f FLOAT)"
            .to_string(),
        format!(
            "INSERT INTO legacy SELECT row_id, begin_snapshot, end_snapshot, id, NULL, NULL \
             FROM \"{name}\""
        ),
    ] {
        sqlx::query(sqlx::AssertSqlSafe(statement))
            .execute(&mut *tx)
            .await
            .unwrap();
    }
    for ((id, d), (_, f)) in doubles.iter().zip(&floats) {
        sqlx::query("UPDATE legacy SET d = ?, f = ? WHERE id = ?")
            .bind(*d)
            .bind(f64::from(*f))
            .bind(*id)
            .execute(&mut *tx)
            .await
            .unwrap();
    }
    for statement in
        [format!("DROP TABLE \"{name}\""), format!("ALTER TABLE legacy RENAME TO \"{name}\"")]
    {
        sqlx::query(sqlx::AssertSqlSafe(statement))
            .execute(&mut *tx)
            .await
            .unwrap();
    }
    tx.commit().await.unwrap();

    crate_sql(&side, &insert_only(&[1, 4])).await.unwrap();
    let stored: Vec<(i64, String, String)> = sqlx::query_as(sqlx::AssertSqlSafe(format!(
        "SELECT id, d, f FROM \"{name}\" WHERE id IN (2, 3, 5, 6) ORDER BY id"
    )))
    .fetch_all(&pool)
    .await
    .unwrap();
    for (((id, d, f), (_, double)), (_, float)) in stored.iter().zip(&doubles).zip(&floats) {
        assert_eq!(
            d.parse::<f64>().unwrap().to_bits(),
            double.to_bits(),
            "row {id}: d {d}"
        );
        assert_eq!(
            f.parse::<f32>().unwrap().to_bits(),
            float.to_bits(),
            "row {id}: f {f}"
        );
    }
    pool.close().await;
}

/// Converting an earlier release's inlined table whose data column is named
/// `rowid`, which shadows SQLite's own row id, keeps each row's float.
#[cfg(feature = "write-sqlite")]
#[tokio::test(flavor = "multi_thread")]
async fn sqlite_legacy_inlined_floats_convert_beside_a_rowid_column() {
    use datafusion_ducklake::{SqliteMetadataProvider, SqliteMetadataWriter};
    let temp = tempfile::TempDir::new().unwrap();
    let path = temp.path().join("crate.db");
    let data = temp.path().join("crate_data");
    std::fs::create_dir_all(&data).unwrap();
    let url = format!("sqlite:{}?mode=rwc", path.display());
    let writer = SqliteMetadataWriter::new_with_init(&url).await.unwrap();
    writer.set_data_path(data.to_str().unwrap()).unwrap();
    let side = Side {
        writer: Arc::new(writer),
        provider: Arc::new(SqliteMetadataProvider::new(&url).await.unwrap()),
    };
    apply_crate(
        &side,
        &WritePath {
            name: "legacy",
            parquet: false,
            limit: 10,
            nan_only_file: false,
            steps: vec![Step::Sql(insert_only(&[5]))],
        },
    )
    .await
    .unwrap();
    crate_sql(&side, "CREATE TABLE lake.main.r (rowid BIGINT, d DOUBLE)")
        .await
        .unwrap();
    crate_sql(
        &side,
        "INSERT INTO lake.main.r VALUES (7, 1.0), (7, 2.0), (NULL, 3.0)",
    )
    .await
    .unwrap();
    let pool = sqlx::SqlitePool::connect(&url).await.unwrap();
    let name: String = sqlx::query_scalar(
        "SELECT d.table_name FROM ducklake_inlined_data_tables d \
         JOIN ducklake_table t ON t.table_id = d.table_id WHERE t.table_name = 'r'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    let mut tx = pool.begin().await.unwrap();
    for statement in [
        "CREATE TABLE legacy (row_id BIGINT, begin_snapshot BIGINT, end_snapshot BIGINT, \
         \"rowid\" BIGINT, d DOUBLE)"
            .to_string(),
        format!(
            "INSERT INTO legacy SELECT row_id, begin_snapshot, end_snapshot, \"rowid\", \
             CAST(d AS REAL) FROM \"{name}\""
        ),
        format!("DROP TABLE \"{name}\""),
        format!("ALTER TABLE legacy RENAME TO \"{name}\""),
    ] {
        sqlx::query(sqlx::AssertSqlSafe(statement))
            .execute(&mut *tx)
            .await
            .unwrap();
    }
    tx.commit().await.unwrap();

    crate_sql(&side, "INSERT INTO lake.main.r VALUES (8, 4.0)")
        .await
        .unwrap();
    let stored: Vec<(Option<i64>, String)> = sqlx::query_as(sqlx::AssertSqlSafe(format!(
        "SELECT \"rowid\", d FROM \"{name}\" ORDER BY row_id"
    )))
    .fetch_all(&pool)
    .await
    .unwrap();
    assert_eq!(
        stored
            .iter()
            .map(|(id, d)| (*id, d.parse::<f64>().unwrap()))
            .collect::<Vec<_>>(),
        vec![(Some(7), 1.0), (Some(7), 2.0), (None, 3.0), (Some(8), 4.0)]
    );
    pool.close().await;
}

/// Converting an earlier release's inlined table takes nothing from the
/// catalog as SQL: a declared column type and an index's stored SQL that would
/// run statements are neither interpolated nor replayed. `FLOAT` columns are
/// converted as `DOUBLE` ones are.
#[cfg(feature = "write-sqlite")]
#[tokio::test(flavor = "multi_thread")]
async fn sqlite_legacy_inlined_table_rebuild_runs_no_catalog_sql() {
    use datafusion_ducklake::{SqliteMetadataProvider, SqliteMetadataWriter};
    let temp = tempfile::TempDir::new().unwrap();
    let path = temp.path().join("crate.db");
    let data = temp.path().join("crate_data");
    std::fs::create_dir_all(&data).unwrap();
    let url = format!("sqlite:{}?mode=rwc", path.display());
    let writer = SqliteMetadataWriter::new_with_init(&url).await.unwrap();
    writer.set_data_path(data.to_str().unwrap()).unwrap();
    let side = Side {
        writer: Arc::new(writer),
        provider: Arc::new(SqliteMetadataProvider::new(&url).await.unwrap()),
    };
    apply_crate(
        &side,
        &WritePath {
            name: "legacy",
            parquet: false,
            limit: 10,
            nan_only_file: false,
            steps: vec![Step::Sql(insert_only(&[2, 5]))],
        },
    )
    .await
    .unwrap();

    let pool = sqlx::SqlitePool::connect(&url).await.unwrap();
    let name: String = sqlx::query_scalar(
        "SELECT d.table_name FROM ducklake_inlined_data_tables d \
         JOIN ducklake_table t ON t.table_id = d.table_id WHERE t.table_name = 't'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    let mut tx = pool.begin().await.unwrap();
    for statement in [
        // A declared type SQLite reports unquoted: interpolated as is, it
        // would end the definition and create `pwned`.
        "CREATE TABLE legacy (row_id BIGINT, begin_snapshot BIGINT, end_snapshot BIGINT, \
         id \"x\"\" BIGINT); CREATE TABLE pwned(a); --\", d FLOAT, f DOUBLE)"
            .to_string(),
        format!(
            "INSERT INTO legacy SELECT row_id, begin_snapshot, end_snapshot, id, \
             CASE d WHEN 'inf' THEN 9e999 ELSE CAST(d AS REAL) END, \
             CASE f WHEN 'inf' THEN 9e999 ELSE CAST(f AS REAL) END FROM \"{name}\""
        ),
        format!("DROP TABLE \"{name}\""),
        format!("ALTER TABLE legacy RENAME TO \"{name}\""),
        format!("CREATE INDEX evil ON \"{name}\"(d)"),
    ] {
        sqlx::query(sqlx::AssertSqlSafe(statement))
            .execute(&mut *tx)
            .await
            .unwrap();
    }
    tx.commit().await.unwrap();
    // An index whose stored SQL runs a second statement when replayed.
    let mut connection = pool.acquire().await.unwrap();
    for statement in [
        "PRAGMA writable_schema = ON".to_string(),
        format!(
            "UPDATE sqlite_master SET sql = 'CREATE INDEX evil ON \"{name}\"(d); \
             CREATE TABLE pwned2(a)' WHERE type = 'index' AND name = 'evil'"
        ),
        "PRAGMA writable_schema = OFF".to_string(),
    ] {
        sqlx::query(sqlx::AssertSqlSafe(statement))
            .execute(&mut *connection)
            .await
            .unwrap();
    }
    drop(connection);
    pool.close().await;

    crate_sql(&side, &insert_only(&[1, 4])).await.unwrap();
    let pool = sqlx::SqlitePool::connect(&url).await.unwrap();
    let pwned: i64 =
        sqlx::query_scalar("SELECT count(*) FROM sqlite_master WHERE name IN ('pwned', 'pwned2')")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(pwned, 0, "no statement from the catalog ran");
    let declared = sqlx::query_as::<_, (String, String)>(
        "SELECT name, type FROM pragma_table_info(?) WHERE name IN ('d', 'f') ORDER BY name",
    )
    .bind(&name)
    .fetch_all(&pool)
    .await
    .unwrap();
    assert_eq!(
        declared,
        vec![("d".to_string(), "VARCHAR".to_string()), ("f".to_string(), "VARCHAR".to_string())],
        "FLOAT and DOUBLE columns become VARCHAR"
    );
    pool.close().await;
    assert_eq!(
        crate_answers(&side).await.remove(0),
        vec!["1,nan,nan", "2,inf,inf", "4,-0.0,-0.0", "5,1.5,1.5"],
    );
}

/// Dates before the year 1000 and after 9999 (BC ones among them) and
/// timestamps at the ends of their range, written inline on each backend:
/// this crate reads back what it wrote, and DuckDB, where it can attach the
/// catalog, reads them as it reads the same values from a Parquet file.
mod temporal_extremes {
    use super::*;
    use arrow::array::{
        Date32Array, Int32Array, TimestampMicrosecondArray, TimestampNanosecondArray,
    };
    use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
    use arrow::record_batch::RecordBatch;
    use datafusion_ducklake::{DuckLakeTableWriter, DuckLakeWriteOptions};

    fn batch() -> RecordBatch {
        let epoch = chrono::NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
        let days = |y: i32, m: u32, d: u32| {
            i32::try_from(
                chrono::NaiveDate::from_ymd_opt(y, m, d)
                    .unwrap()
                    .signed_duration_since(epoch)
                    .num_days(),
            )
            .unwrap()
        };
        let micros = |y: i32, m: u32, d: u32| i64::from(days(y, m, d)) * 86_400_000_000;
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("id", DataType::Int32, true),
                Field::new("dt", DataType::Date32, true),
                Field::new("ts", DataType::Timestamp(TimeUnit::Microsecond, None), true),
                Field::new("ns", DataType::Timestamp(TimeUnit::Nanosecond, None), true),
            ])),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5, 6])),
                Arc::new(Date32Array::from(vec![
                    Some(days(-43, 3, 15)),
                    Some(days(1, 1, 1)),
                    Some(days(999, 12, 31)),
                    Some(days(9999, 12, 31)),
                    Some(days(12000, 1, 1)),
                    Some(days(2024, 2, 29)),
                ])),
                Arc::new(TimestampMicrosecondArray::from(vec![
                    Some(micros(-100, 1, 1)),
                    Some(micros(1, 1, 1)),
                    Some(micros(10000, 1, 1) - 1),
                    Some(micros(12000, 6, 1) + 123_456),
                    Some(micros(1000, 1, 1)),
                    None,
                ])),
                Arc::new(TimestampNanosecondArray::from(vec![
                    Some(-9_000_000_000_000_000_000),
                    Some(9_000_000_000_000_000_000),
                    Some(0),
                    Some(-1),
                    None,
                    Some(1),
                ])),
            ],
        )
        .unwrap()
    }

    async fn write(side: &Side, table: &str, limit: usize) {
        DuckLakeTableWriter::new(
            Arc::clone(&side.writer),
            Arc::new(object_store::local::LocalFileSystem::new()),
        )
        .unwrap()
        .with_options(&DuckLakeWriteOptions::default().with_data_inlining_row_limit(limit))
        .append_table("main", table, &[batch()])
        .await
        .unwrap();
    }

    async fn read_back(side: &Side, table: &str) -> RecordBatch {
        let ctx = SessionContext::new();
        let snapshot = side.provider.get_current_snapshot().unwrap();
        let catalog = DuckLakeCatalog::with_snapshot(Arc::clone(&side.provider), snapshot).unwrap();
        ctx.register_catalog("lake", Arc::new(catalog));
        let batches = ctx
            .sql(&format!("SELECT * FROM lake.main.{table} ORDER BY id"))
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        arrow::compute::concat_batches(&batches[0].schema(), &batches).unwrap()
    }

    /// Write the values inline (`i`) and to Parquet (`p`), and require both to
    /// read back as written. Returns whether `i` was stored inline.
    async fn check(side: &Side) -> bool {
        write(side, "i", 10).await;
        write(side, "p", 0).await;
        for table in ["i", "p"] {
            let got = read_back(side, table).await;
            assert_eq!(
                got.columns(),
                batch().columns(),
                "table `{table}` reads back what was written"
            );
        }
        let snapshot = side.provider.get_current_snapshot().unwrap();
        let schema = side
            .provider
            .get_schema_by_name("main", snapshot)
            .unwrap()
            .unwrap();
        let table = side
            .provider
            .get_table_by_name(schema.schema_id, "i", snapshot)
            .unwrap()
            .unwrap();
        side.provider
            .get_table_files_for_select(table.table_id, snapshot)
            .unwrap()
            .is_empty()
    }

    fn duckdb_same(metadata: &Metadata, data: &Path) {
        let connection = duckdb_connection();
        attach_lake(&connection, metadata, data, true);
        let inline = query_lines(&connection, "SELECT * FROM lake.main.i ORDER BY id");
        let parquet = query_lines(&connection, "SELECT * FROM lake.main.p ORDER BY id");
        assert_eq!(inline.len(), 6);
        assert_eq!(
            inline, parquet,
            "DuckDB reads the inlined values as the Parquet ones"
        );
    }

    #[cfg(feature = "write-sqlite")]
    #[tokio::test(flavor = "multi_thread")]
    async fn sqlite() {
        use datafusion_ducklake::{SqliteMetadataProvider, SqliteMetadataWriter};
        let temp = tempfile::TempDir::new().unwrap();
        let path = temp.path().join("crate.db");
        let data = temp.path().join("crate_data");
        std::fs::create_dir_all(&data).unwrap();
        let url = format!("sqlite:{}?mode=rwc", path.display());
        let writer = SqliteMetadataWriter::new_with_init(&url).await.unwrap();
        writer.set_data_path(data.to_str().unwrap()).unwrap();
        let side = Side {
            writer: Arc::new(writer),
            provider: Arc::new(SqliteMetadataProvider::new(&url).await.unwrap()),
        };
        assert!(check(&side).await, "SQLite stores these values inline");
        drop(side);
        crate::inlined_mutation_differential_tests::make_sqlite_catalog_duckdb_readable(&path)
            .await;
        duckdb_same(&Metadata::Sqlite(path), &data);
    }

    #[cfg(feature = "write-duckdb")]
    #[tokio::test(flavor = "multi_thread")]
    async fn duckdb() {
        use datafusion_ducklake::DuckdbMetadataWriter;
        let temp = tempfile::TempDir::new().unwrap();
        let path = temp.path().join("crate.duckdb");
        let data = temp.path().join("crate_data");
        std::fs::create_dir_all(&data).unwrap();
        let writer = DuckdbMetadataWriter::new_with_init(path.display().to_string()).unwrap();
        writer.set_data_path(data.to_str().unwrap()).unwrap();
        let provider = writer.metadata_provider();
        let side = Side {
            writer: Arc::new(writer),
            provider: Arc::new(provider),
        };
        assert!(check(&side).await, "DuckDB stores these values inline");
        drop(side);
        duckdb_same(&Metadata::Duckdb(path), &data);
    }

    /// MySQL's `DATE` and `DATETIME(6)` span the years 1000 to 9999, so the
    /// write goes to Parquet.
    #[cfg(feature = "write-mysql")]
    #[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
    #[tokio::test(flavor = "multi_thread")]
    async fn mysql() {
        use datafusion_ducklake::{MySqlMetadataProvider, MySqlMetadataWriter};
        use testcontainers::runners::AsyncRunner;
        use testcontainers_modules::mysql::Mysql;
        let container = Mysql::default().start().await.unwrap();
        let port = container.get_host_port_ipv4(3306).await.unwrap();
        let temp = tempfile::TempDir::new().unwrap();
        let data = temp.path().join("crate_data");
        std::fs::create_dir_all(&data).unwrap();
        let url = format!("mysql://root@127.0.0.1:{port}/test");
        let writer = MySqlMetadataWriter::new_with_init(&url).await.unwrap();
        writer.set_data_path(data.to_str().unwrap()).unwrap();
        let side = Side {
            writer: Arc::new(writer),
            provider: Arc::new(MySqlMetadataProvider::new(&url).await.unwrap()),
        };
        assert!(!check(&side).await, "MySQL writes these values to Parquet");
    }

    #[cfg(feature = "multicatalog-postgres")]
    #[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
    #[tokio::test(flavor = "multi_thread")]
    async fn multicatalog() {
        use datafusion_ducklake::{
            MulticatalogManager, MulticatalogProvider, PostgresMetadataWriter,
        };
        use testcontainers::runners::AsyncRunner;
        use testcontainers_modules::postgres::Postgres;
        let container = Postgres::default().start().await.unwrap();
        let port = container.get_host_port_ipv4(5432).await.unwrap();
        let temp = tempfile::TempDir::new().unwrap();
        let data = temp.path().join("crate_data");
        std::fs::create_dir_all(&data).unwrap();
        let url = format!("postgresql://postgres:postgres@127.0.0.1:{port}/postgres");
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
        let side = Side {
            writer: Arc::new(writer),
            provider: Arc::new(MulticatalogProvider::with_pool(pool, "lake").await.unwrap()),
        };
        assert!(check(&side).await, "PostgreSQL stores these values inline");
    }
}
