//! Dates and timestamps at and past the ends of DuckDB's types, inlined.
//!
//! DuckDB's `DATE` and `TIMESTAMP` types hold `infinity` and `-infinity` and
//! years far beyond what a calendar library covers. Official DuckLake stores
//! an inlined value as DuckDB's text of it; this crate must store the same
//! text, read official's back, and refuse a value DuckDB has no such type for
//! rather than store anything else.

#![cfg(all(feature = "write-sqlite", feature = "metadata-duckdb"))]

use std::path::Path;
use std::sync::Arc;

use arrow::array::{
    Date32Array, Int32Array, TimestampMicrosecondArray, TimestampMillisecondArray,
    TimestampNanosecondArray, TimestampSecondArray,
};
use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use arrow::record_batch::RecordBatch;
use datafusion::prelude::SessionContext;
use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeTableWriter, DuckLakeWriteOptions, MetadataProvider, MetadataWriter,
    SqliteMetadataProvider, SqliteMetadataWriter,
};

use crate::inlined_mutation_differential_tests::{
    Metadata, attach_lake, duckdb_connection, make_sqlite_catalog_duckdb_readable, query_lines,
};

/// One inlined row's six temporal columns as the backend's text.
type TextRow = (
    Option<String>,
    Option<String>,
    Option<String>,
    Option<String>,
    Option<String>,
    Option<String>,
);

/// A change to a row that makes it hold a value DuckDB cannot.
type Shape = Box<dyn Fn(&mut Row)>;

/// Each row as this crate writes it (`id`, then the Arrow value of `d`, `ts`,
/// `tsn`, `tss`, `tsm`, `tz`) and as official writes it (DuckDB literals).
struct Row {
    id: i32,
    date: Option<i32>,
    micros: Option<i64>,
    nanos: Option<i64>,
    seconds: Option<i64>,
    millis: Option<i64>,
    zoned: Option<i64>,
    literals: [&'static str; 6],
}

fn rows() -> Vec<Row> {
    vec![
        Row {
            id: 1,
            date: Some(i32::MAX),
            micros: Some(i64::MAX),
            nanos: Some(i64::MAX),
            seconds: Some(i64::MAX),
            millis: Some(i64::MAX),
            zoned: Some(i64::MAX),
            literals: ["infinity"; 6],
        },
        Row {
            id: 2,
            date: Some(-i32::MAX),
            micros: Some(-i64::MAX),
            nanos: Some(-i64::MAX),
            seconds: Some(-i64::MAX),
            millis: Some(-i64::MAX),
            zoned: Some(-i64::MAX),
            literals: ["-infinity"; 6],
        },
        Row {
            id: 3,
            date: Some(-2_147_483_646),
            micros: Some(-9_223_372_022_400_000_000),
            nanos: Some(-9_223_286_400_000_000_000),
            seconds: Some(-9_223_372_022_400),
            millis: Some(-9_223_372_022_400_000),
            zoned: Some(-9_223_372_022_400_000_000),
            literals: [
                "5877642-06-25 (BC)",
                "290309-12-22 (BC) 00:00:00",
                "1677-09-22 00:00:00",
                "290309-12-22 (BC) 00:00:00",
                "290309-12-22 (BC) 00:00:00",
                "290309-12-22 (BC) 00:00:00+00",
            ],
        },
        Row {
            id: 4,
            date: Some(2_147_483_646),
            micros: Some(9_223_372_036_854_775_806),
            nanos: Some(9_223_372_036_854_775_806),
            seconds: Some(9_223_372_036_854),
            millis: Some(9_223_372_036_854_775),
            zoned: Some(9_223_372_036_854_775_806),
            literals: [
                "5881580-07-10",
                "294247-01-10 04:00:54.775806",
                "2262-04-11 23:47:16.854775806",
                "294247-01-10 04:00:54",
                "294247-01-10 04:00:54.775",
                "294247-01-10 04:00:54.775806+00",
            ],
        },
        Row {
            id: 5,
            date: Some(108_853_222),
            micros: Some(8_458_224_206_400_123_456),
            nanos: Some(0),
            seconds: Some(-8_582_498_337_600),
            millis: Some(-8_582_498_337_599_877),
            zoned: Some(8_458_224_206_400_123_456),
            literals: [
                "300000-01-01",
                "270000-06-15 12:00:00.123456",
                "1970-01-01 00:00:00",
                "270000-06-15 (BC) 12:00:00",
                "270000-06-15 (BC) 12:00:00.123",
                "270000-06-15 12:00:00.123456+00",
            ],
        },
        Row {
            id: 6,
            date: Some(-110_291_912),
            micros: Some(-8_582_498_337_599_876_544),
            nanos: None,
            seconds: None,
            millis: None,
            zoned: None,
            literals: [
                "300000-01-01 (BC)",
                "270000-06-15 (BC) 12:00:00.123456",
                "NULL",
                "NULL",
                "NULL",
                "NULL",
            ],
        },
    ]
}

const TYPES: [&str; 6] =
    ["DATE", "TIMESTAMP", "TIMESTAMP_NS", "TIMESTAMP_S", "TIMESTAMP_MS", "TIMESTAMPTZ"];

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, true),
        Field::new("d", DataType::Date32, true),
        Field::new("ts", DataType::Timestamp(TimeUnit::Microsecond, None), true),
        Field::new("tsn", DataType::Timestamp(TimeUnit::Nanosecond, None), true),
        Field::new("tss", DataType::Timestamp(TimeUnit::Second, None), true),
        Field::new(
            "tsm",
            DataType::Timestamp(TimeUnit::Millisecond, None),
            true,
        ),
        Field::new(
            "tz",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            true,
        ),
    ]))
}

fn batch(rows: &[Row]) -> RecordBatch {
    let column = |f: fn(&Row) -> Option<i64>| rows.iter().map(f).collect::<Vec<_>>();
    RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(Int32Array::from(
                rows.iter().map(|row| row.id).collect::<Vec<_>>(),
            )),
            Arc::new(Date32Array::from(
                rows.iter().map(|row| row.date).collect::<Vec<_>>(),
            )),
            Arc::new(TimestampMicrosecondArray::from(column(|row| row.micros))),
            Arc::new(TimestampNanosecondArray::from(column(|row| row.nanos))),
            Arc::new(TimestampSecondArray::from(column(|row| row.seconds))),
            Arc::new(TimestampMillisecondArray::from(column(|row| row.millis))),
            Arc::new(TimestampMicrosecondArray::from(column(|row| row.zoned)).with_timezone("UTC")),
        ],
    )
    .unwrap()
}

/// Official DuckLake's catalog of [`rows`], on SQLite metadata.
fn official_catalog(path: &Path, data: &Path) {
    let connection = duckdb_connection();
    connection.execute_batch("SET TimeZone = 'UTC'").unwrap();
    attach_lake(
        &connection,
        &Metadata::Sqlite(path.to_path_buf()),
        data,
        false,
    );
    let columns = ["d", "ts", "tsn", "tss", "tsm", "tz"];
    let definition = columns
        .iter()
        .zip(TYPES)
        .map(|(name, ty)| format!("{name} {ty}"))
        .collect::<Vec<_>>()
        .join(", ");
    connection
        .execute_batch(&format!(
            "CREATE TABLE lake.main.t (id INTEGER, {definition})"
        ))
        .unwrap();
    let values = rows()
        .iter()
        .map(|row| {
            let cells = row
                .literals
                .iter()
                .zip(TYPES)
                .map(|(literal, ty)| match *literal {
                    "NULL" => "NULL".to_string(),
                    literal => format!("CAST('{literal}' AS {ty})"),
                })
                .collect::<Vec<_>>()
                .join(", ");
            format!("({}, {cells})", row.id)
        })
        .collect::<Vec<_>>()
        .join(", ");
    connection
        .execute_batch(&format!("INSERT INTO lake.main.t VALUES {values}"))
        .unwrap();
}

struct CrateLake {
    writer: Arc<dyn MetadataWriter>,
    provider: Arc<dyn MetadataProvider>,
}

async fn crate_lake(path: &Path, data: &Path) -> CrateLake {
    let url = format!("sqlite:{}?mode=rwc", path.display());
    let writer = SqliteMetadataWriter::new_with_init(&url).await.unwrap();
    writer.set_data_path(data.to_str().unwrap()).unwrap();
    CrateLake {
        writer: Arc::new(writer),
        provider: Arc::new(SqliteMetadataProvider::new(&url).await.unwrap()),
    }
}

async fn crate_write(lake: &CrateLake, batch: RecordBatch) -> datafusion_ducklake::Result<()> {
    DuckLakeTableWriter::new(
        Arc::clone(&lake.writer),
        Arc::new(object_store::local::LocalFileSystem::new()),
    )?
    .with_options(&DuckLakeWriteOptions::default().with_data_inlining_row_limit(10))
    .append_table("main", "t", &[batch])
    .await
    .map(|_| ())
}

async fn crate_read(provider: Arc<dyn MetadataProvider>) -> RecordBatch {
    let ctx = SessionContext::new();
    let snapshot = provider.get_current_snapshot().unwrap();
    let catalog = DuckLakeCatalog::with_snapshot(provider, snapshot).unwrap();
    ctx.register_catalog("lake", Arc::new(catalog));
    let batches = ctx
        .sql("SELECT id, d, ts, tsn, tss, tsm, tz FROM lake.main.t ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    arrow::compute::concat_batches(&batches[0].schema(), &batches).unwrap()
}

/// The stored text of `t`'s inlined rows, one line per row in row-id order.
async fn stored_text(path: &Path) -> Vec<String> {
    let pool = sqlx::SqlitePool::connect(&format!("sqlite:{}", path.display()))
        .await
        .unwrap();
    let tables: Vec<String> = sqlx::query_scalar(
        "SELECT r.table_name FROM ducklake_inlined_data_tables r \
         JOIN ducklake_table t ON t.table_id = r.table_id \
         WHERE t.table_name = 't' ORDER BY r.schema_version",
    )
    .fetch_all(&pool)
    .await
    .unwrap();
    let mut lines = Vec::new();
    for table in tables {
        let rows: Vec<TextRow> = sqlx::query_as(sqlx::AssertSqlSafe(format!(
            "SELECT CAST(d AS TEXT), CAST(ts AS TEXT), CAST(tsn AS TEXT), \
                 CAST(tss AS TEXT), CAST(tsm AS TEXT), CAST(tz AS TEXT) \
                 FROM \"{table}\" ORDER BY row_id"
        )))
        .fetch_all(&pool)
        .await
        .unwrap();
        for (a, b, c, d, e, f) in rows {
            let cells = [a, b, c, d, e, f].map(|cell| cell.unwrap_or_else(|| "NULL".to_string()));
            lines.push(cells.join("|"));
        }
    }
    lines
}

/// DuckDB's text of every row of `t` in the lake at `path`.
fn duckdb_rows(path: &Path, data: &Path) -> Vec<String> {
    let connection = duckdb_connection();
    connection.execute_batch("SET TimeZone = 'UTC'").unwrap();
    attach_lake(
        &connection,
        &Metadata::Sqlite(path.to_path_buf()),
        data,
        true,
    );
    query_lines(
        &connection,
        "SELECT id, d::VARCHAR, ts::VARCHAR, tsn::VARCHAR, tss::VARCHAR, tsm::VARCHAR, \
         tz::VARCHAR FROM lake.main.t ORDER BY id",
    )
}

/// This crate stores each value as official does, reads back what it wrote,
/// and reads official's catalog as the same values.
#[tokio::test(flavor = "multi_thread")]
async fn sqlite_inlines_values_beyond_the_calendar_as_official_does() {
    let temp = tempfile::TempDir::new().unwrap();
    let official_path = temp.path().join("official.db");
    let official_data = temp.path().join("official_data");
    std::fs::create_dir_all(&official_data).unwrap();
    official_catalog(&official_path, &official_data);

    let official_provider: Arc<dyn MetadataProvider> = Arc::new(
        SqliteMetadataProvider::new(&format!("sqlite:{}", official_path.display()))
            .await
            .unwrap(),
    );
    let expected = batch(&rows());
    let read = crate_read(official_provider).await;
    assert_eq!(
        read.columns(),
        expected.columns(),
        "the crate reads official's inlined rows as the same values"
    );

    let crate_path = temp.path().join("crate.db");
    let crate_data = temp.path().join("crate_data");
    std::fs::create_dir_all(&crate_data).unwrap();
    let lake = crate_lake(&crate_path, &crate_data).await;
    crate_write(&lake, expected.clone()).await.unwrap();

    let official_text = stored_text(&official_path).await;
    assert_eq!(
        official_text.len(),
        rows().len(),
        "official inlines every row"
    );
    assert_eq!(
        stored_text(&crate_path).await,
        official_text,
        "the crate stores DuckDB's text, as official does"
    );

    let read = crate_read(Arc::clone(&lake.provider)).await;
    assert_eq!(
        read.columns(),
        expected.columns(),
        "the crate reads back what it wrote"
    );

    drop(lake);
    make_sqlite_catalog_duckdb_readable(&crate_path).await;
    assert_eq!(
        duckdb_rows(&crate_path, &crate_data),
        duckdb_rows(&official_path, &official_data),
        "DuckDB reads the crate's rows as official's"
    );
}

/// A value no DuckDB type holds fails the write, which commits nothing.
#[tokio::test(flavor = "multi_thread")]
async fn sqlite_refuses_values_duckdb_cannot_hold() {
    let shapes: Vec<(&str, Shape)> = vec![
        (
            "date i32::MIN",
            Box::new(|row: &mut Row| row.date = Some(i32::MIN)),
        ),
        (
            "timestamp i64::MIN",
            Box::new(|row: &mut Row| row.micros = Some(i64::MIN)),
        ),
        (
            "timestamp below 290309 BC",
            Box::new(|row: &mut Row| row.micros = Some(-9_223_372_022_400_000_001)),
        ),
        (
            "timestamp_ns below 1677-09-22",
            Box::new(|row: &mut Row| row.nanos = Some(-9_223_286_400_000_000_001)),
        ),
        (
            "timestamp_s past 294247",
            Box::new(|row: &mut Row| row.seconds = Some(9_223_372_036_855)),
        ),
        (
            "timestamp_ms past 294247",
            Box::new(|row: &mut Row| row.millis = Some(9_223_372_036_854_776)),
        ),
    ];
    for (name, change) in shapes {
        let temp = tempfile::TempDir::new().unwrap();
        let path = temp.path().join("crate.db");
        let data = temp.path().join("crate_data");
        std::fs::create_dir_all(&data).unwrap();
        let lake = crate_lake(&path, &data).await;
        let mut row = rows().remove(4);
        change(&mut row);
        let error = crate_write(&lake, batch(&[row]))
            .await
            .expect_err(name)
            .to_string();
        assert!(
            error.contains("DuckDB has no"),
            "{name}: unexpected error {error}"
        );
        assert!(
            stored_text(&path).await.is_empty(),
            "{name}: nothing is inlined"
        );
        let snapshot = lake.provider.get_current_snapshot().unwrap();
        let tables = lake.provider.list_schemas(snapshot).unwrap();
        for schema in tables {
            for table in lake
                .provider
                .list_tables(schema.schema_id, snapshot)
                .unwrap()
            {
                assert_eq!(
                    lake.provider
                        .get_table_files_for_select(table.table_id, snapshot)
                        .unwrap()
                        .len(),
                    0,
                    "{name}: no file is committed"
                );
            }
        }
    }
}

#[cfg(any(
    feature = "write-duckdb",
    feature = "write-postgres",
    feature = "multicatalog-postgres"
))]
async fn write_with(
    writer: Arc<dyn MetadataWriter>,
    batch: RecordBatch,
) -> datafusion_ducklake::Result<()> {
    DuckLakeTableWriter::new(
        writer,
        Arc::new(object_store::local::LocalFileSystem::new()),
    )?
    .with_options(&DuckLakeWriteOptions::default().with_data_inlining_row_limit(10))
    .append_table("main", "t", &[batch])
    .await
    .map(|_| ())
}

/// DuckDB metadata stores the values natively: this crate reads them back and
/// DuckDB reads them as it reads official's, and a value no DuckDB type holds
/// is refused there too.
#[cfg(feature = "write-duckdb")]
#[tokio::test(flavor = "multi_thread")]
async fn duckdb_inlines_values_beyond_the_calendar() {
    use datafusion_ducklake::DuckdbMetadataWriter;
    let temp = tempfile::TempDir::new().unwrap();
    let official_path = temp.path().join("official.db");
    let official_data = temp.path().join("official_data");
    std::fs::create_dir_all(&official_data).unwrap();
    official_catalog(&official_path, &official_data);

    let path = temp.path().join("crate.duckdb");
    let data = temp.path().join("crate_data");
    std::fs::create_dir_all(&data).unwrap();
    let expected = batch(&rows());
    {
        let writer = DuckdbMetadataWriter::new_with_init(path.display().to_string()).unwrap();
        writer.set_data_path(data.to_str().unwrap()).unwrap();
        let provider: Arc<dyn MetadataProvider> = Arc::new(writer.metadata_provider());
        let writer: Arc<dyn MetadataWriter> = Arc::new(writer);
        write_with(Arc::clone(&writer), expected.clone())
            .await
            .unwrap();
        let snapshot = provider.get_current_snapshot().unwrap();
        let schema = provider
            .get_schema_by_name("main", snapshot)
            .unwrap()
            .unwrap();
        let table = provider
            .get_table_by_name(schema.schema_id, "t", snapshot)
            .unwrap()
            .unwrap();
        assert!(
            provider
                .get_table_files_for_select(table.table_id, snapshot)
                .unwrap()
                .is_empty(),
            "the rows are inlined"
        );
        let read = crate_read(Arc::clone(&provider)).await;
        assert_eq!(read.columns(), expected.columns());
        let mut row = rows().remove(4);
        row.date = Some(i32::MIN);
        let error = write_with(writer, batch(&[row])).await.unwrap_err();
        assert!(error.to_string().contains("DuckDB has no"), "{error}");
    }
    let connection = duckdb_connection();
    connection.execute_batch("SET TimeZone = 'UTC'").unwrap();
    attach_lake(&connection, &Metadata::Duckdb(path), &data, true);
    assert_eq!(
        query_lines(
            &connection,
            "SELECT id, d::VARCHAR, ts::VARCHAR, tsn::VARCHAR, tss::VARCHAR, tsm::VARCHAR, \
             tz::VARCHAR FROM lake.main.t ORDER BY id",
        ),
        duckdb_rows(&official_path, &official_data),
    );
}

/// The multi-catalog layout stores dates and timestamps as text: DuckDB's
/// text, as official's SQLite catalog holds it.
#[cfg(feature = "multicatalog-postgres")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
#[tokio::test(flavor = "multi_thread")]
async fn multicatalog_inlines_values_beyond_the_calendar_as_text() {
    use datafusion_ducklake::{MulticatalogManager, MulticatalogProvider, PostgresMetadataWriter};
    use testcontainers::runners::AsyncRunner;
    use testcontainers_modules::postgres::Postgres;
    let temp = tempfile::TempDir::new().unwrap();
    let official_path = temp.path().join("official.db");
    let official_data = temp.path().join("official_data");
    std::fs::create_dir_all(&official_data).unwrap();
    official_catalog(&official_path, &official_data);
    let official_text = stored_text(&official_path).await;

    let container = Postgres::default().start().await.unwrap();
    let port = container.get_host_port_ipv4(5432).await.unwrap();
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
    let writer: Arc<dyn MetadataWriter> = Arc::new(writer);
    let expected = batch(&rows());
    write_with(Arc::clone(&writer), expected.clone())
        .await
        .unwrap();
    let provider: Arc<dyn MetadataProvider> = Arc::new(
        MulticatalogProvider::with_pool(pool.clone(), "lake")
            .await
            .unwrap(),
    );
    let read = crate_read(provider).await;
    assert_eq!(read.columns(), expected.columns());

    let tables: Vec<String> = sqlx::query_scalar(
        "SELECT table_name FROM ducklake_inlined_data_tables ORDER BY schema_version",
    )
    .fetch_all(&pool)
    .await
    .unwrap();
    let mut stored = Vec::new();
    for table in tables {
        let rows: Vec<TextRow> = sqlx::query_as(sqlx::AssertSqlSafe(format!(
            "SELECT d::TEXT, ts::TEXT, tsn::TEXT, tss::TEXT, tsm::TEXT, tz::TEXT \
                 FROM \"{table}\" ORDER BY row_id"
        )))
        .fetch_all(&pool)
        .await
        .unwrap();
        for (a, b, c, d, e, f) in rows {
            let cells = [a, b, c, d, e, f].map(|cell| cell.unwrap_or_else(|| "NULL".to_string()));
            stored.push(cells.join("|"));
        }
    }
    assert_eq!(stored, official_text);

    let mut row = rows().remove(4);
    row.micros = Some(i64::MIN);
    let error = write_with(writer, batch(&[row])).await.unwrap_err();
    assert!(error.to_string().contains("DuckDB has no"), "{error}");
}

/// A `TIMETZ` column's inlined data table is `TIME WITH TIME ZONE`, as
/// official's PostgreSQL metadata declares it: one this crate creates for a
/// table official created takes official's inserts and reads them back.
#[cfg(feature = "write-postgres")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
#[tokio::test(flavor = "multi_thread")]
async fn postgres_inlined_timetz_table_is_native() {
    use datafusion_ducklake::{
        PartitionTransform, PostgresMetadataProvider, PostgresSingleCatalogMetadataWriter,
    };
    use testcontainers::runners::AsyncRunner;
    use testcontainers_modules::postgres::Postgres;
    let container = Postgres::default().start().await.unwrap();
    let port = container.get_host_port_ipv4(5432).await.unwrap();
    let url = format!("postgresql://postgres:postgres@127.0.0.1:{port}/postgres");
    let metadata = Metadata::Postgres(format!(
        "host=127.0.0.1 port={port} user=postgres password=postgres dbname=postgres"
    ));
    let temp = tempfile::TempDir::new().unwrap();
    let data = temp.path().join("data");
    std::fs::create_dir_all(&data).unwrap();
    let official = |statement: &str| {
        let connection = duckdb_connection();
        attach_lake(&connection, &metadata, &data, false);
        connection
            .execute_batch(&format!("USE lake; {statement}"))
            .unwrap_or_else(|e| panic!("`{statement}`: {e}"));
    };
    official("CREATE TABLE z (id INTEGER, t TIMETZ)");

    let writer = PostgresSingleCatalogMetadataWriter::new_with_init(&url)
        .await
        .unwrap();
    let provider = PostgresMetadataProvider::new(&url).await.unwrap();
    let snapshot = provider.get_current_snapshot().unwrap();
    let schema = provider
        .get_schema_by_name("main", snapshot)
        .unwrap()
        .unwrap();
    let table = provider
        .get_table_by_name(schema.schema_id, "z", snapshot)
        .unwrap()
        .unwrap();
    writer
        .set_partition_spec(
            table.table_id,
            &[("id".to_string(), PartitionTransform::Identity)],
        )
        .unwrap();
    let pool = sqlx::PgPool::connect(&url).await.unwrap();
    // This crate's snapshots leave official's id allocators NULL.
    sqlx::query(
        "UPDATE ducklake_snapshot SET next_catalog_id = 1000, next_file_id = 1000
         WHERE next_catalog_id IS NULL OR next_file_id IS NULL",
    )
    .execute(&pool)
    .await
    .unwrap();
    let (name, declared): (String, String) = sqlx::query_as(
        "SELECT r.table_name, c.data_type::text
         FROM ducklake_inlined_data_tables r
         JOIN information_schema.columns c ON c.table_name = r.table_name
         WHERE r.table_id = $1 AND c.column_name = 't'
         ORDER BY r.schema_version DESC LIMIT 1",
    )
    .bind(table.table_id)
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(declared, "time with time zone", "{name}");

    official("INSERT INTO z VALUES (1, '12:30:00.5+05:30')");
    let connection = duckdb_connection();
    attach_lake(&connection, &metadata, &data, true);
    assert_eq!(
        query_lines(&connection, "SELECT id, t::VARCHAR FROM lake.main.z"),
        vec!["1,12:30:00.5+05:30".to_string()]
    );

    // This crate reads the native column, and a `BYTEA` one an earlier
    // release created, as the same text.
    let read = |url: String| async move {
        let provider: Arc<dyn MetadataProvider> =
            Arc::new(PostgresMetadataProvider::new(&url).await.unwrap());
        let ctx = SessionContext::new();
        let snapshot = provider.get_current_snapshot().unwrap();
        ctx.register_catalog(
            "lake",
            Arc::new(DuckLakeCatalog::with_snapshot(provider, snapshot).unwrap()),
        );
        let batches = ctx
            .sql("SELECT id, t FROM lake.main.z")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        arrow::util::pretty::pretty_format_batches(&batches)
            .unwrap()
            .to_string()
    };
    let native = read(url.clone()).await;
    assert!(native.contains("12:30:00.5+05:30"), "{native}");
    sqlx::query(sqlx::AssertSqlSafe(format!(
        "ALTER TABLE \"{name}\" ALTER COLUMN t TYPE BYTEA USING convert_to(t::text, 'UTF8')"
    )))
    .execute(&pool)
    .await
    .unwrap();
    assert_eq!(read(url.clone()).await, native);
}
