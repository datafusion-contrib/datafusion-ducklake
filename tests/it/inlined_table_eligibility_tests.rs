//! Which tables get inlined storage, as official DuckLake decides it.
//!
//! GEOMETRY columns are never inlined: official's `SupportsInlining` refuses
//! the type, nested inside another type too, so a table holding one gets no
//! inlined data table and its rows always go to Parquet. Where it can, official
//! creates the tables, this crate writes to them, and official reads back what
//! it can.
//!
//! A catalog's inlining limit set through `DuckLakeCatalog::with_write_options`
//! applies to the inlined data tables its commits create, as official's
//! `DATA_INLINING_ROW_LIMIT` attach option does: at `0`, none.

#![cfg(all(feature = "write", feature = "metadata-duckdb"))]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{BinaryArray, Int32Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::prelude::SessionContext;
use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeTableWriter, DuckLakeWriteOptions, MetadataProvider, MetadataWriter,
    WriteMode, execute_ducklake_sql,
};
use object_store::local::LocalFileSystem;

use crate::inlined_mutation_differential_tests::{
    Metadata, duckdb_connection, duckdb_query, official, query_lines,
};

/// WKB of `POINT (1 2)`.
const POINT_1_2: &str = "0101000000000000000000F03F0000000000000040";
/// WKB of `POINT (3 4)`.
const POINT_3_4: &str = "010100000000000000000008400000000000001040";

/// The crate's writer and provider over the catalog official created.
struct Side {
    writer: Arc<dyn MetadataWriter>,
    provider: Arc<dyn MetadataProvider>,
}

/// `main.g` (a GEOMETRY column), `main.n` (one nested in a struct) and
/// `main.plain` (none), each with one row, under official's default
/// inlining limit.
fn official_tables() -> Vec<String> {
    [
        "CREATE TABLE lake.g (id INTEGER, shape GEOMETRY)",
        "CREATE TABLE lake.n (id INTEGER, s STRUCT(a INTEGER, p GEOMETRY))",
        "CREATE TABLE lake.plain (id INTEGER, b BLOB)",
        "INSERT INTO lake.g VALUES (1, 'POINT (1 2)'::GEOMETRY)",
        "INSERT INTO lake.n VALUES (1, {'a': 1, 'p': 'POINT (1 2)'::GEOMETRY})",
        "INSERT INTO lake.plain VALUES (1, '\\x01'::BLOB)",
    ]
    .iter()
    .map(ToString::to_string)
    .collect()
}

async fn run_sql(side: &Side, statement: &str) {
    let ctx = SessionContext::new();
    let catalog =
        DuckLakeCatalog::with_writer(Arc::clone(&side.provider), Arc::clone(&side.writer)).unwrap();
    DuckLakeCatalog::register(&ctx, "lake", catalog);
    let handle =
        DuckLakeCatalog::with_writer(Arc::clone(&side.provider), Arc::clone(&side.writer)).unwrap();
    execute_ducklake_sql(&ctx, &handle, statement)
        .await
        .unwrap_or_else(|e| panic!("`{statement}`: {e}"))
        .collect()
        .await
        .unwrap_or_else(|e| panic!("`{statement}`: {e}"));
}

/// `(registered inlined data tables, live data files)` of `table`, read from
/// the metadata database.
fn storage(metadata: &Metadata, table: &str) -> (usize, usize) {
    let connection = duckdb_connection();
    metadata.attach_raw(&connection);
    let table_id = query_lines(
        &connection,
        &format!(
            "SELECT table_id FROM ducklake_table WHERE table_name = '{table}' \
             AND end_snapshot IS NULL"
        ),
    )
    .remove(0);
    let count = |sql: String| -> usize { query_lines(&connection, &sql)[0].parse().unwrap() };
    (
        count(format!(
            "SELECT count(*) FROM ducklake_inlined_data_tables WHERE table_id = {table_id}"
        )),
        count(format!(
            "SELECT count(*) FROM ducklake_data_file WHERE table_id = {table_id} \
             AND end_snapshot IS NULL"
        )),
    )
}

fn geometry_batch(ids: &[i32], wkb: &[u8]) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, true),
            Field::new("shape", DataType::Binary, true),
        ])),
        vec![
            Arc::new(Int32Array::from(ids.to_vec())),
            Arc::new(BinaryArray::from_iter_values(ids.iter().map(|_| wkb))),
        ],
    )
    .unwrap()
}

fn hex(text: &str) -> Vec<u8> {
    (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&text[i..i + 2], 16).unwrap())
        .collect()
}

/// Every inlining path on tables official created: SQL `INSERT`, a staged
/// commit, `UPDATE` (whose new row versions official inlines when small) and
/// a partitioning change (after which official creates the table's inlined
/// data table). None inlines a row of a table with a GEOMETRY column, nested
/// or not, nor creates an inlined data table for one; the table without one
/// still inlines.
async fn geometry_is_never_inlined(metadata: &Metadata, data: &Path, side: &Side) {
    assert_eq!(storage(metadata, "g"), (0, 1), "official inlined nothing");
    assert_eq!(storage(metadata, "n"), (0, 1));
    assert_eq!(storage(metadata, "plain"), (1, 0), "official inlines BLOB");

    run_sql(
        side,
        &format!("INSERT INTO lake.main.g VALUES (2, X'{POINT_3_4}')"),
    )
    .await;
    assert_eq!(storage(metadata, "g"), (0, 2), "SQL INSERT wrote Parquet");

    let transaction_writer =
        DuckLakeTableWriter::new(Arc::clone(&side.writer), Arc::new(LocalFileSystem::new()))
            .unwrap()
            .with_options(&DuckLakeWriteOptions::default().with_data_inlining_row_limit(10));
    let mut transaction = transaction_writer.transaction();
    let staged = geometry_batch(&[3], &hex(POINT_1_2));
    transaction
        .stage_write(
            "main",
            "g",
            staged.schema().as_ref(),
            WriteMode::Append,
            std::slice::from_ref(&staged),
        )
        .await
        .unwrap();
    transaction.commit().await.unwrap();
    assert_eq!(
        storage(metadata, "g"),
        (0, 3),
        "a staged commit wrote Parquet"
    );

    // The deletion of the old version is an inlined file deletion.
    run_sql(side, "UPDATE lake.main.g SET id = id + 10 WHERE id = 1").await;
    assert_eq!(
        storage(metadata, "g"),
        (0, 4),
        "UPDATE wrote its new row version to Parquet"
    );

    run_sql(side, "UPDATE lake.main.n SET id = 5").await;
    assert_eq!(
        storage(metadata, "n"),
        (0, 2),
        "a nested GEOMETRY is refused too"
    );

    run_sql(side, "ALTER TABLE lake.main.g SET PARTITIONED BY (id)").await;
    assert_eq!(storage(metadata, "g").0, 0, "no inlined data table");
    run_sql(
        side,
        &format!("INSERT INTO lake.main.g VALUES (4, X'{POINT_1_2}')"),
    )
    .await;
    assert_eq!(storage(metadata, "g"), (0, 5));

    run_sql(side, "INSERT INTO lake.main.plain VALUES (2, X'02')").await;
    assert_eq!(storage(metadata, "plain"), (1, 0), "BLOB still inlines");

    assert_eq!(
        crate_rows(side, "SELECT id, shape FROM lake.main.g ORDER BY id").await,
        [
            format!("2,{POINT_3_4}"),
            format!("3,{POINT_1_2}"),
            format!("4,{POINT_1_2}"),
            format!("11,{POINT_1_2}"),
        ]
    );
    assert_eq!(
        crate_rows(
            side,
            "SELECT id, s['a'], s['p'] FROM lake.main.n ORDER BY id"
        )
        .await,
        [format!("5,1,{POINT_1_2}")]
    );
    // Official reads the tables without a GEOMETRY column; it cannot cast the
    // BLOB a Parquet file this crate writes for a GEOMETRY column holds.
    fill_snapshot_counters(metadata).await;
    assert_eq!(
        duckdb_query(
            metadata,
            data,
            "SELECT id, hex(b) FROM lake.main.plain ORDER BY id"
        ),
        ["1,01", "2,02"]
    );
}

/// The rows of `sql` read by this crate, columns joined by `,` and binary
/// values in upper-case hex.
async fn crate_rows(side: &Side, sql: &str) -> Vec<String> {
    let ctx = SessionContext::new();
    let snapshot = side.provider.get_current_snapshot().unwrap();
    let catalog = DuckLakeCatalog::with_snapshot(Arc::clone(&side.provider), snapshot).unwrap();
    ctx.register_catalog("lake", Arc::new(catalog));
    let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
    let mut rows = Vec::new();
    for batch in &batches {
        for row in 0..batch.num_rows() {
            rows.push(
                batch
                    .columns()
                    .iter()
                    .map(|column| {
                        match datafusion::scalar::ScalarValue::try_from_array(column, row).unwrap()
                        {
                            datafusion::scalar::ScalarValue::Binary(Some(bytes))
                            | datafusion::scalar::ScalarValue::BinaryView(Some(bytes)) => {
                                bytes.iter().map(|byte| format!("{byte:02X}")).collect()
                            },
                            value => value.to_string(),
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(","),
            );
        }
    }
    rows
}

/// Fill the `next_catalog_id` and `next_file_id` this crate leaves NULL in
/// the snapshots it commits to an official-created catalog (a known gap),
/// which official needs to attach it.
async fn fill_snapshot_counters(metadata: &Metadata) {
    let statement = "UPDATE ducklake_snapshot \
                     SET next_catalog_id = COALESCE(next_catalog_id, 1000), \
                         next_file_id = COALESCE(next_file_id, 1000)";
    match metadata {
        Metadata::Sqlite(path) => {
            let pool = sqlx::SqlitePool::connect(&format!("sqlite:{}", path.display()))
                .await
                .unwrap();
            sqlx::query(statement).execute(&pool).await.unwrap();
            pool.close().await;
        },
        #[cfg(feature = "metadata-postgres")]
        Metadata::Postgres(dsn) => {
            let pool = sqlx::PgPool::connect(&postgres_url(dsn)).await.unwrap();
            sqlx::query(statement).execute(&pool).await.unwrap();
            pool.close().await;
        },
        _ => {},
    }
}

/// The URL of the PostgreSQL database a `key=value` DSN names.
#[cfg(feature = "metadata-postgres")]
fn postgres_url(dsn: &str) -> String {
    let value = |key: &str| {
        dsn.split_whitespace()
            .find_map(|pair| pair.strip_prefix(&format!("{key}=")))
            .unwrap()
            .to_string()
    };
    format!(
        "postgresql://{}:{}@{}:{}/{}",
        value("user"),
        value("password"),
        value("host"),
        value("port"),
        value("dbname")
    )
}

fn scratch() -> (tempfile::TempDir, PathBuf) {
    let temp = tempfile::TempDir::new().unwrap();
    let data = temp.path().join("data");
    std::fs::create_dir_all(&data).unwrap();
    (temp, data)
}

#[cfg(feature = "write-sqlite")]
#[tokio::test(flavor = "multi_thread")]
async fn sqlite_never_inlines_geometry() {
    use datafusion_ducklake::{SqliteMetadataProvider, SqliteMetadataWriter};
    let (temp, data) = scratch();
    let path = temp.path().join("lake.db");
    let metadata = Metadata::Sqlite(path.clone());
    official(&metadata, &data, &official_tables());
    let url = format!("sqlite:{}", path.display());
    let side = Side {
        writer: Arc::new(SqliteMetadataWriter::new_with_init(&url).await.unwrap()),
        provider: Arc::new(SqliteMetadataProvider::new(&url).await.unwrap()),
    };
    geometry_is_never_inlined(&metadata, &data, &side).await;
}

/// Single-catalog PostgreSQL inlines no write; a partitioning change still
/// creates the inlined data table official creates, and none for a table
/// with a GEOMETRY column. (Writing rows to an official-created catalog there
/// is a known gap.)
#[cfg(feature = "write-postgres")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
#[tokio::test(flavor = "multi_thread")]
async fn postgres_single_catalog_creates_no_inlined_table_for_geometry() {
    use datafusion_ducklake::{PostgresMetadataProvider, PostgresSingleCatalogMetadataWriter};
    use testcontainers::runners::AsyncRunner;
    use testcontainers_modules::postgres::Postgres;

    let container = Postgres::default().start().await.unwrap();
    let port = container.get_host_port_ipv4(5432).await.unwrap();
    let (_temp, data) = scratch();
    let metadata = Metadata::Postgres(format!(
        "host=127.0.0.1 port={port} user=postgres password=postgres dbname=postgres"
    ));
    official(&metadata, &data, &official_tables());
    let url = format!("postgresql://postgres:postgres@127.0.0.1:{port}/postgres");
    let side = Side {
        writer: Arc::new(
            PostgresSingleCatalogMetadataWriter::new_with_init(&url)
                .await
                .unwrap(),
        ),
        provider: Arc::new(PostgresMetadataProvider::new(&url).await.unwrap()),
    };
    run_sql(&side, "ALTER TABLE lake.main.g SET PARTITIONED BY (id)").await;
    run_sql(&side, "ALTER TABLE lake.main.plain SET PARTITIONED BY (id)").await;
    assert_eq!(storage(&metadata, "g").0, 0);
    assert_eq!(storage(&metadata, "plain").0, 2);
    fill_snapshot_counters(&metadata).await;
    assert_eq!(
        duckdb_query(
            &metadata,
            &data,
            "SELECT id, shape::VARCHAR FROM lake.main.g ORDER BY id"
        ),
        ["1,POINT (1 2)"]
    );
}

/// Create `main.g` with one row in Parquet.
#[cfg(any(feature = "write-duckdb", feature = "write-mysql", feature = "multicatalog-postgres"))]
async fn create_g(side: &Side) {
    DuckLakeTableWriter::new(Arc::clone(&side.writer), Arc::new(LocalFileSystem::new()))
        .unwrap()
        .with_options(&DuckLakeWriteOptions::default().with_data_inlining_row_limit(0))
        .write_table("main", "g", &[geometry_batch(&[1], &hex(POINT_1_2))])
        .await
        .unwrap();
}

/// Record `main.g`'s `shape` column as GEOMETRY and unregister its inlined
/// data table.
#[cfg(any(feature = "write-duckdb", feature = "write-mysql", feature = "multicatalog-postgres"))]
const RETYPE: [&str; 2] = [
    "UPDATE ducklake_column SET column_type = 'geometry' WHERE column_name = 'shape'",
    "DELETE FROM ducklake_inlined_data_tables",
];

/// Backends where this crate does not write to an official-created catalog
/// (official cannot commit to MySQL or multicatalog stores; DuckDB catalogs
/// it creates are a known gap): the crate creates `main.g` ([`create_g`]),
/// whose column is then recorded as GEOMETRY as official records one
/// ([`RETYPE`]). Rows of the table are written to
/// Parquet by `INSERT` and `UPDATE`, and a partitioning change creates no
/// inlined data table for it.
#[cfg(any(feature = "write-duckdb", feature = "write-mysql", feature = "multicatalog-postgres"))]
async fn crate_side_never_inlines_geometry(side: &Side) {
    let snapshot = side.provider.get_current_snapshot().unwrap();
    let schema = side
        .provider
        .get_schema_by_name("main", snapshot)
        .unwrap()
        .unwrap();
    let table = side
        .provider
        .get_table_by_name(schema.schema_id, "g", snapshot)
        .unwrap()
        .unwrap();
    side.writer
        .set_table_setting(table.table_id, "data_inlining_row_limit", "10")
        .unwrap();
    let inlined_rows = |side: &Side| {
        let head = side.provider.get_current_snapshot().unwrap();
        side.provider
            .get_inlined_data(
                table.table_id,
                head,
                &side
                    .provider
                    .get_table_structure(table.table_id, head)
                    .unwrap(),
            )
            .unwrap()
            .iter()
            .map(RecordBatch::num_rows)
            .sum::<usize>()
    };
    run_sql(
        side,
        &format!("INSERT INTO lake.main.g VALUES (2, X'{POINT_3_4}')"),
    )
    .await;
    assert_eq!(inlined_rows(side), 0, "INSERT wrote Parquet");
    run_sql(side, "UPDATE lake.main.g SET id = 11 WHERE id = 1").await;
    assert_eq!(inlined_rows(side), 0, "UPDATE wrote Parquet");
    run_sql(side, "ALTER TABLE lake.main.g SET PARTITIONED BY (id)").await;
    run_sql(
        side,
        &format!("INSERT INTO lake.main.g VALUES (4, X'{POINT_1_2}')"),
    )
    .await;
    assert_eq!(inlined_rows(side), 0);
    assert_eq!(
        crate_rows(side, "SELECT id, shape FROM lake.main.g ORDER BY id").await,
        [format!("2,{POINT_3_4}"), format!("4,{POINT_1_2}"), format!("11,{POINT_1_2}"),]
    );
}

#[cfg(feature = "write-duckdb")]
#[tokio::test(flavor = "multi_thread")]
async fn duckdb_never_inlines_geometry() {
    use datafusion_ducklake::DuckdbMetadataWriter;
    let (temp, data) = scratch();
    let path = temp.path().join("lake.duckdb");
    let writer = DuckdbMetadataWriter::new_with_init(path.display().to_string()).unwrap();
    writer.set_data_path(data.to_str().unwrap()).unwrap();
    let provider = writer.metadata_provider();
    let side = Side {
        writer: Arc::new(writer),
        provider: Arc::new(provider),
    };
    create_g(&side).await;
    drop(side);
    duckdb::Connection::open(&path)
        .unwrap()
        .execute_batch(&RETYPE.join(";"))
        .unwrap();
    let writer = DuckdbMetadataWriter::new(path.display().to_string()).unwrap();
    let provider = writer.metadata_provider();
    let side = Side {
        writer: Arc::new(writer),
        provider: Arc::new(provider),
    };
    crate_side_never_inlines_geometry(&side).await;
    drop(side);
    let metadata = Metadata::Duckdb(path);
    assert_eq!(storage(&metadata, "g").0, 0);
}

#[cfg(feature = "write-mysql")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
#[tokio::test(flavor = "multi_thread")]
async fn mysql_never_inlines_geometry() {
    use datafusion_ducklake::{MySqlMetadataProvider, MySqlMetadataWriter};
    use testcontainers::runners::AsyncRunner;
    use testcontainers_modules::mysql::Mysql;

    let container = Mysql::default().start().await.unwrap();
    let port = container.get_host_port_ipv4(3306).await.unwrap();
    let url = format!("mysql://root@127.0.0.1:{port}/test");
    let (_temp, data) = scratch();
    let writer = MySqlMetadataWriter::new_with_init(&url).await.unwrap();
    writer.set_data_path(data.to_str().unwrap()).unwrap();
    let side = Side {
        writer: Arc::new(writer),
        provider: Arc::new(MySqlMetadataProvider::new(&url).await.unwrap()),
    };
    let pool = sqlx::MySqlPool::connect(&url).await.unwrap();
    create_g(&side).await;
    for statement in RETYPE {
        sqlx::query(statement).execute(&pool).await.unwrap();
    }
    crate_side_never_inlines_geometry(&side).await;
    let inlined: i64 = sqlx::query_scalar("SELECT count(*) FROM ducklake_inlined_data_tables")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(inlined, 0, "no inlined data table for a GEOMETRY table");
}

#[cfg(feature = "multicatalog-postgres")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
#[tokio::test(flavor = "multi_thread")]
async fn multicatalog_never_inlines_geometry() {
    use datafusion_ducklake::{MulticatalogManager, MulticatalogProvider, PostgresMetadataWriter};
    use testcontainers::runners::AsyncRunner;
    use testcontainers_modules::postgres::Postgres;

    let container = Postgres::default().start().await.unwrap();
    let port = container.get_host_port_ipv4(5432).await.unwrap();
    let url = format!("postgresql://postgres:postgres@127.0.0.1:{port}/postgres");
    let (_temp, data) = scratch();
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
        provider: Arc::new(
            MulticatalogProvider::with_pool(pool.clone(), "lake")
                .await
                .unwrap(),
        ),
    };
    create_g(&side).await;
    for statement in RETYPE {
        sqlx::query(statement).execute(&pool).await.unwrap();
    }
    crate_side_never_inlines_geometry(&side).await;
    let inlined: i64 = sqlx::query_scalar("SELECT count(*) FROM ducklake_inlined_data_tables")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(inlined, 0, "no inlined data table for a GEOMETRY table");
}

/// Through a catalog whose write options set the inlining limit to `0`,
/// `CREATE TABLE` and a partitioning change create no inlined data table, and
/// a column named like an inlined system column is allowed, as official with
/// `DATA_INLINING_ROW_LIMIT 0` behaves. A table writer's own limit of `0`
/// applies where the catalog stores none, as that attach option does: its
/// first write creates no inlined data table either. Each step opens
/// the catalog with `open`; `registered` counts its registered inlined data
/// tables with the catalog closed.
async fn zero_limit_creates_no_inlined_table(
    open: impl AsyncFn() -> Side,
    registered: impl AsyncFn() -> usize,
) {
    let zero = DuckLakeWriteOptions::default().with_data_inlining_row_limit(0);
    let catalog = |side: &Side| {
        DuckLakeCatalog::with_writer(Arc::clone(&side.provider), Arc::clone(&side.writer))
            .unwrap()
            .with_write_options(zero.clone())
    };
    {
        let side = open().await;
        DuckLakeTableWriter::new(Arc::clone(&side.writer), Arc::new(LocalFileSystem::new()))
            .unwrap()
            .with_options(&zero)
            .write_table("main", "u", &[geometry_batch(&[1], &hex(POINT_1_2))])
            .await
            .unwrap();
    }
    assert_eq!(registered().await, 0, "a table writer's first write");
    {
        let side = open().await;
        let ctx = SessionContext::new();
        DuckLakeCatalog::register(&ctx, "lake", catalog(&side));
        ctx.sql("CREATE TABLE lake.main.t (id INT, row_id INT)")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
    }
    assert_eq!(registered().await, 0, "CREATE TABLE");
    {
        let side = open().await;
        let ctx = SessionContext::new();
        DuckLakeCatalog::register(&ctx, "lake", catalog(&side));
        execute_ducklake_sql(
            &ctx,
            &catalog(&side),
            "ALTER TABLE lake.main.t SET PARTITIONED BY (id)",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    }
    assert_eq!(registered().await, 0, "SET PARTITIONED BY");
}

/// A stored table, schema or catalog-wide `data_inlining_row_limit` outranks
/// the limit a catalog's write options set, as official's stored settings
/// outrank its `DATA_INLINING_ROW_LIMIT` attach option: with the options at
/// `10`, a table set to `0` gets no inlined data table from a partitioning
/// change and its inserts go to Parquet, and in a schema, then a catalog, set
/// to `0` `CREATE TABLE` allows a column named like an inlined system column,
/// creates no inlined data table, and its inserts go to Parquet. `count` runs
/// a counting query on the closed catalog, `execute` a statement.
async fn scoped_limit_outranks_write_options(
    open: impl AsyncFn() -> Side,
    count: impl AsyncFn(&str) -> i64,
    execute: impl AsyncFn(&str),
) {
    let ten = DuckLakeWriteOptions::default().with_data_inlining_row_limit(10);
    let catalog = |side: &Side| {
        DuckLakeCatalog::with_writer(Arc::clone(&side.provider), Arc::clone(&side.writer))
            .unwrap()
            .with_write_options(ten.clone())
    };
    let run = async |statement: &str| {
        let side = open().await;
        let ctx = SessionContext::new();
        DuckLakeCatalog::register(&ctx, "lake", catalog(&side));
        execute_ducklake_sql(&ctx, &catalog(&side), statement)
            .await
            .unwrap_or_else(|e| panic!("{statement}: {e}"))
            .collect()
            .await
            .unwrap_or_else(|e| panic!("{statement}: {e}"));
    };
    let registered = "SELECT count(*) FROM ducklake_inlined_data_tables";
    let files = |table: &str| {
        format!(
            "SELECT count(*) FROM ducklake_data_file f JOIN ducklake_table t \
             ON t.table_id = f.table_id AND t.end_snapshot IS NULL \
             WHERE t.table_name = '{table}' AND f.end_snapshot IS NULL"
        )
    };

    {
        let side = open().await;
        DuckLakeTableWriter::new(Arc::clone(&side.writer), Arc::new(LocalFileSystem::new()))
            .unwrap()
            .with_options(&DuckLakeWriteOptions::default().with_data_inlining_row_limit(0))
            .write_table("main", "seed", &[geometry_batch(&[1], &hex(POINT_1_2))])
            .await
            .unwrap();
    }
    let base = count(registered).await;
    run("CREATE TABLE lake.main.t (id INT)").await;
    assert_eq!(
        count(registered).await,
        base + 1,
        "CREATE TABLE under the options"
    );
    let table_id = count("SELECT table_id FROM ducklake_table WHERE table_name = 't'").await;
    open()
        .await
        .writer
        .set_table_setting(table_id, "data_inlining_row_limit", "0")
        .unwrap();
    run("ALTER TABLE lake.main.t SET PARTITIONED BY (id)").await;
    assert_eq!(
        count(registered).await,
        base + 1,
        "partitioning a table set to 0"
    );
    run("INSERT INTO lake.main.t VALUES (1), (2)").await;
    assert_eq!(
        count(&files("t")).await,
        2,
        "an insert into a table set to 0: a Parquet file per partition"
    );

    let schema_id = count("SELECT schema_id FROM ducklake_schema WHERE schema_name = 'main'").await;
    execute(&format!(
        "INSERT INTO ducklake_metadata (key, value, scope, scope_id) \
         VALUES ('data_inlining_row_limit', '0', 'schema', {schema_id})"
    ))
    .await;
    run("CREATE TABLE lake.main.s (id INT, row_id INT)").await;
    assert_eq!(
        count(registered).await,
        base + 1,
        "CREATE TABLE in a schema set to 0"
    );
    run("INSERT INTO lake.main.s VALUES (1, 1), (2, 2)").await;
    assert_eq!(
        count(&files("s")).await,
        1,
        "an insert in a schema set to 0"
    );

    // A stored catalog-wide setting outranks the options too, as official's
    // stored settings outrank its attach option.
    execute("DELETE FROM ducklake_metadata WHERE key = 'data_inlining_row_limit'").await;
    execute(
        "INSERT INTO ducklake_metadata (key, value, scope, scope_id) \
         VALUES ('data_inlining_row_limit', '0', NULL, NULL)",
    )
    .await;
    run("CREATE TABLE lake.main.g (id INT, row_id INT)").await;
    assert_eq!(
        count(registered).await,
        base + 1,
        "CREATE TABLE under a catalog set to 0"
    );
    run("INSERT INTO lake.main.g VALUES (1, 1), (2, 2)").await;
    assert_eq!(
        count(&files("g")).await,
        1,
        "an insert under a catalog set to 0"
    );
}

#[cfg(feature = "write-sqlite")]
#[tokio::test(flavor = "multi_thread")]
async fn sqlite_scoped_limit_outranks_write_options() {
    use datafusion_ducklake::{SqliteMetadataProvider, SqliteMetadataWriter};
    let (temp, data) = scratch();
    let url = format!("sqlite:{}?mode=rwc", temp.path().join("c.db").display());
    SqliteMetadataWriter::new_with_init(&url)
        .await
        .unwrap()
        .set_data_path(data.to_str().unwrap())
        .unwrap();
    let pool = async || sqlx::SqlitePool::connect(&url).await.unwrap();
    scoped_limit_outranks_write_options(
        async || Side {
            writer: Arc::new(SqliteMetadataWriter::new(&url).await.unwrap()),
            provider: Arc::new(SqliteMetadataProvider::new(&url).await.unwrap()),
        },
        async |sql: &str| {
            let pool = pool().await;
            let value: i64 = sqlx::query_scalar(sqlx::AssertSqlSafe(sql.to_string()))
                .fetch_one(&pool)
                .await
                .unwrap();
            pool.close().await;
            value
        },
        async |sql: &str| {
            let pool = pool().await;
            sqlx::query(sqlx::AssertSqlSafe(sql.to_string()))
                .execute(&pool)
                .await
                .unwrap();
            pool.close().await;
        },
    )
    .await;
}

#[cfg(feature = "write-duckdb")]
#[tokio::test(flavor = "multi_thread")]
async fn duckdb_scoped_limit_outranks_write_options() {
    use datafusion_ducklake::DuckdbMetadataWriter;
    let (temp, data) = scratch();
    let path = temp.path().join("c.duckdb").display().to_string();
    DuckdbMetadataWriter::new_with_init(&path)
        .unwrap()
        .set_data_path(data.to_str().unwrap())
        .unwrap();
    scoped_limit_outranks_write_options(
        async || {
            let writer = DuckdbMetadataWriter::new(&path).unwrap();
            let provider = writer.metadata_provider();
            Side {
                writer: Arc::new(writer),
                provider: Arc::new(provider),
            }
        },
        async |sql: &str| {
            duckdb::Connection::open(&path)
                .unwrap()
                .query_row(sql, [], |row| row.get::<_, i64>(0))
                .unwrap()
        },
        async |sql: &str| {
            duckdb::Connection::open(&path)
                .unwrap()
                .execute(sql, [])
                .unwrap();
        },
    )
    .await;
}

#[cfg(feature = "write-sqlite")]
#[tokio::test(flavor = "multi_thread")]
async fn sqlite_zero_limit_creates_no_inlined_table() {
    use datafusion_ducklake::{SqliteMetadataProvider, SqliteMetadataWriter};
    let (temp, data) = scratch();
    let url = format!("sqlite:{}?mode=rwc", temp.path().join("c.db").display());
    SqliteMetadataWriter::new_with_init(&url)
        .await
        .unwrap()
        .set_data_path(data.to_str().unwrap())
        .unwrap();
    zero_limit_creates_no_inlined_table(
        async || Side {
            writer: Arc::new(SqliteMetadataWriter::new(&url).await.unwrap()),
            provider: Arc::new(SqliteMetadataProvider::new(&url).await.unwrap()),
        },
        async || {
            let pool = sqlx::SqlitePool::connect(&url).await.unwrap();
            let count: i64 =
                sqlx::query_scalar("SELECT count(*) FROM ducklake_inlined_data_tables")
                    .fetch_one(&pool)
                    .await
                    .unwrap();
            pool.close().await;
            count as usize
        },
    )
    .await;
}

#[cfg(feature = "write-duckdb")]
#[tokio::test(flavor = "multi_thread")]
async fn duckdb_zero_limit_creates_no_inlined_table() {
    use datafusion_ducklake::DuckdbMetadataWriter;
    let (temp, data) = scratch();
    let path = temp.path().join("c.duckdb").display().to_string();
    DuckdbMetadataWriter::new_with_init(&path)
        .unwrap()
        .set_data_path(data.to_str().unwrap())
        .unwrap();
    zero_limit_creates_no_inlined_table(
        async || {
            let writer = DuckdbMetadataWriter::new(&path).unwrap();
            let provider = writer.metadata_provider();
            Side {
                writer: Arc::new(writer),
                provider: Arc::new(provider),
            }
        },
        async || {
            duckdb::Connection::open(&path)
                .unwrap()
                .query_row(
                    "SELECT count(*) FROM ducklake_inlined_data_tables",
                    [],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap() as usize
        },
    )
    .await;
}

#[cfg(feature = "write-mysql")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
#[tokio::test(flavor = "multi_thread")]
async fn mysql_zero_limit_creates_no_inlined_table() {
    use datafusion_ducklake::{MySqlMetadataProvider, MySqlMetadataWriter};
    use testcontainers::runners::AsyncRunner;
    use testcontainers_modules::mysql::Mysql;

    let container = Mysql::default().start().await.unwrap();
    let port = container.get_host_port_ipv4(3306).await.unwrap();
    let url = format!("mysql://root@127.0.0.1:{port}/test");
    let (_temp, data) = scratch();
    MySqlMetadataWriter::new_with_init(&url)
        .await
        .unwrap()
        .set_data_path(data.to_str().unwrap())
        .unwrap();
    let pool = sqlx::MySqlPool::connect(&url).await.unwrap();
    zero_limit_creates_no_inlined_table(
        async || Side {
            writer: Arc::new(MySqlMetadataWriter::new(&url).await.unwrap()),
            provider: Arc::new(MySqlMetadataProvider::new(&url).await.unwrap()),
        },
        async || {
            let count: i64 =
                sqlx::query_scalar("SELECT count(*) FROM ducklake_inlined_data_tables")
                    .fetch_one(&pool)
                    .await
                    .unwrap();
            count as usize
        },
    )
    .await;
}

#[cfg(feature = "multicatalog-postgres")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
#[tokio::test(flavor = "multi_thread")]
async fn multicatalog_zero_limit_creates_no_inlined_table() {
    use datafusion_ducklake::{MulticatalogManager, MulticatalogProvider, PostgresMetadataWriter};
    use testcontainers::runners::AsyncRunner;
    use testcontainers_modules::postgres::Postgres;

    let container = Postgres::default().start().await.unwrap();
    let port = container.get_host_port_ipv4(5432).await.unwrap();
    let url = format!("postgresql://postgres:postgres@127.0.0.1:{port}/postgres");
    let (_temp, data) = scratch();
    let pool = sqlx::PgPool::connect(&url).await.unwrap();
    datafusion_ducklake::initialize_multicatalog_schema(&pool)
        .await
        .unwrap();
    let catalog = MulticatalogManager::new(pool.clone())
        .create_catalog("lake")
        .await
        .unwrap();
    zero_limit_creates_no_inlined_table(
        async || {
            let writer = PostgresMetadataWriter::with_pool(pool.clone(), catalog)
                .await
                .unwrap();
            writer.set_data_path(data.to_str().unwrap()).unwrap();
            Side {
                writer: Arc::new(writer),
                provider: Arc::new(
                    MulticatalogProvider::with_pool(pool.clone(), "lake")
                        .await
                        .unwrap(),
                ),
            }
        },
        async || {
            let count: i64 =
                sqlx::query_scalar("SELECT count(*) FROM ducklake_inlined_data_tables")
                    .fetch_one(&pool)
                    .await
                    .unwrap();
            count as usize
        },
    )
    .await;
}

/// Single-catalog PostgreSQL inlines no write, but creates the inlined data
/// tables official creates; under an in-process limit of `0`, none.
#[cfg(feature = "write-postgres")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
#[tokio::test(flavor = "multi_thread")]
async fn postgres_single_catalog_zero_limit_creates_no_inlined_table() {
    use datafusion_ducklake::{PostgresMetadataProvider, PostgresSingleCatalogMetadataWriter};
    use testcontainers::runners::AsyncRunner;
    use testcontainers_modules::postgres::Postgres;

    let container = Postgres::default().start().await.unwrap();
    let port = container.get_host_port_ipv4(5432).await.unwrap();
    let url = format!("postgresql://postgres:postgres@127.0.0.1:{port}/postgres");
    let (_temp, data) = scratch();
    PostgresSingleCatalogMetadataWriter::new_with_init(&url)
        .await
        .unwrap()
        .set_data_path(data.to_str().unwrap())
        .unwrap();
    let pool = sqlx::PgPool::connect(&url).await.unwrap();
    zero_limit_creates_no_inlined_table(
        async || Side {
            writer: Arc::new(
                PostgresSingleCatalogMetadataWriter::new(&url)
                    .await
                    .unwrap(),
            ),
            provider: Arc::new(PostgresMetadataProvider::new(&url).await.unwrap()),
        },
        async || {
            let count: i64 =
                sqlx::query_scalar("SELECT count(*) FROM ducklake_inlined_data_tables")
                    .fetch_one(&pool)
                    .await
                    .unwrap();
            count as usize
        },
    )
    .await;
}
