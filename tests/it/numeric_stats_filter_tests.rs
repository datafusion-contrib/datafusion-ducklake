//! Catalog-side file pruning on numeric columns never drops a matching file.
//!
//! Every backend's statistics filter compares a stored bound — text — against
//! a constant in its own SQL. For a float column the two sides have to be read
//! in a domain where they order exactly as the column's values do: a `FLOAT`
//! bound written `1e+20` names the `f32` closest to 10^20, which is not the
//! `DOUBLE` that text parses to. Comparing them as doubles skips the file
//! holding the very row `c = 1e20::FLOAT` asks for (duckdb/ducklake#1604).
//!
//! Each catalog here holds one value per data file, chosen at the edges where
//! the domains part: values that print in exponent notation, `±FLT_MAX`,
//! `±DBL_MAX`, subnormals, signed zeros, integers past 2^53, `DECIMAL`s
//! wider than a double, and doubles whose shortest text is fixed-point with
//! more digits than a double holds exactly (`0.48024702758086213`), which
//! DuckDB parses as a `DECIMAL` whose conversion to `DOUBLE` is not correctly
//! rounded. Every predicate (a sample by default; set
//! `NUMERIC_STATS_ALL_PREDICATES` for all) runs with the statistics filter
//! honoured and ignored (the `PushdownToggle` of
//! `pushdown_row_preservation_tests`), and the two answers must be identical.
//! The catalogs are written both by official DuckLake and by this crate, on
//! every metadata backend.

#![cfg(all(feature = "metadata-duckdb", feature = "write-sqlite", feature = "write-duckdb"))]

use std::path::Path;
use std::sync::Arc;

use datafusion::prelude::SessionContext;
use datafusion_ducklake::metadata_writer::ColumnDef;
use datafusion_ducklake::{
    DuckLakeCatalog, MetadataProvider, MetadataWriter, WriteMode, execute_ducklake_sql,
};
use tempfile::TempDir;

use crate::pushdown_row_preservation_tests::{run_query, with_official_ducklake};

/// One row per data file: `(r FLOAT, d DOUBLE, dec DECIMAL(38,10), big BIGINT)`
/// as the text both engines parse. `None` is NULL.
type Row = [Option<&'static str>; 4];

/// A row whose cells are owned, for the generated doubles.
type OwnedRow = [Option<String>; 4];

const ROWS: &[Row] = &[
    [
        Some("1e20"),
        Some("1e20"),
        Some("12345678901234567890.1234567891"),
        Some("9223372036854775807"),
    ],
    [
        Some("-1e20"),
        Some("-1e20"),
        Some("-12345678901234567890.1234567891"),
        Some("-9223372036854775808"),
    ],
    [Some("1e-7"), Some("1e-7"), Some("0.0000000001"), Some("9007199254740993")],
    [
        Some("3.4028235e38"),
        Some("1.7976931348623157e308"),
        Some("-0.0000000001"),
        Some("-9007199254740993"),
    ],
    [
        Some("-3.4028235e38"),
        Some("-1.7976931348623157e308"),
        Some("9999999999999999999999999999.9999999999"),
        Some("9007199254740992"),
    ],
    [Some("1e-45"), Some("5e-324"), Some("0.1000000000"), Some("0")],
    [Some("-1e-45"), Some("-5e-324"), Some("0"), Some("1")],
    [Some("1.1754944e-38"), Some("2.2250738585072014e-308"), None, Some("-1")],
    [Some("1e-40"), Some("1e-310"), None, None],
    [Some("0.1"), Some("0.1"), None, None],
    [Some("1.0"), Some("0.30000000000000004"), None, None],
    [Some("3.3"), Some("3.4028234663852886e38"), None, None],
    [Some("1.5e-10"), Some("1.2345678901234567e-300"), None, None],
    [Some("0.0"), Some("0.0"), None, None],
    [Some("-0.0"), Some("-0.0"), None, None],
    [Some("1.1529215e18"), Some("9007199254740992"), None, None],
    [Some("16777216"), Some("1e16"), None, None],
];

/// A multi-row file, so a constant can fall strictly inside one file's bounds.
const MIXED: [Row; 3] = [
    [Some("0.1"), Some("0.1"), Some("0.5"), Some("5")],
    [Some("1e20"), Some("1e20"), Some("7.25"), Some("7")],
    [Some("1e-7"), Some("1e-7"), Some("-3.5"), Some("-3")],
];

/// Doubles whose shortest text is fixed-point and carries 16 or 17 significant
/// digits, so that spliced bare into DuckDB SQL it is a `DECIMAL` wider than a
/// double holds exactly. Picked by hand, then drawn at random from a fixed seed
/// across the magnitudes Rust and DuckDB print without an exponent.
fn fixed_point_doubles() -> Vec<f64> {
    let mut values = vec![
        0.48024702758086213,
        123456.78901234567,
        -0.12345678901234568,
        9.999999999999998,
        1234567890123456.8,
        0.00012345678901234567,
    ];
    // splitmix64, so the draw is the same on every run and platform.
    let mut state: u64 = 0x5eed_d0b1_e5f0_0d42;
    let mut next = || {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    };
    while values.len() < 48 {
        let mantissa = next() >> 12;
        // Binary exponents from 2^-14 to 2^52: printed without an exponent by
        // both Rust's `Display` and DuckDB.
        let exponent = 1023 - 14 + next() % 67;
        let sign = next() & 1;
        let value = f64::from_bits((sign << 63) | (exponent << 52) | mantissa);
        values.push(value);
    }
    values
}

/// Every data-file row: the fixed rows, then one file per generated double.
fn rows() -> Vec<OwnedRow> {
    let mut rows: Vec<OwnedRow> = ROWS
        .iter()
        .map(|row| row.map(|cell| cell.map(str::to_string)))
        .collect();
    for value in fixed_point_doubles() {
        rows.push([None, Some(format!("{value:?}")), None, None]);
    }
    rows
}

const COLUMNS: [(&str, &str, &str); 4] = [
    ("r", "FLOAT", "REAL"),
    ("d", "DOUBLE", "DOUBLE"),
    ("dec", "DECIMAL(38,10)", "DECIMAL(38,10)"),
    ("big", "BIGINT", "BIGINT"),
];

/// Constants probed against each column, beyond the values stored in it: the
/// neighbours a lossy comparison confuses with them.
fn extra_constants(column: usize) -> Vec<String> {
    match column {
        0 => {
            let mut values: Vec<String> = [
                f32::MAX.next_down(),
                1e20f32.next_up(),
                1e20f32.next_down(),
                1e-7f32.next_up(),
                1e-7f32.next_down(),
                2e-45,
                0.1f32.next_up(),
            ]
            .iter()
            .map(|v| format!("{v:?}"))
            .collect();
            values.push("1e21".into());
            values
        },
        1 => {
            let mut values = vec![
                f64::MAX.next_down(),
                1e20f64.next_up(),
                1e20f64.next_down(),
                1e-7f64.next_up(),
                1e-7f64.next_down(),
                1e-323,
                // FLT_MAX's shortest text read as a double, which is not FLT_MAX.
                3.4028235e38,
            ];
            // The neighbours of every few generated doubles: a constant one ulp
            // off a bound must not read as the bound.
            for value in fixed_point_doubles().iter().step_by(4) {
                values.push(value.next_up());
                values.push(value.next_down());
            }
            values.iter().map(|v| format!("{v:?}")).collect()
        },
        2 => vec![
            "12345678901234567890.1234567890".into(),
            "12345678901234567890.1234567892".into(),
            "0.0000000002".into(),
        ],
        _ => vec![
            "9223372036854775806".into(),
            "9007199254740994".into(),
            "-9007199254740994".into(),
        ],
    }
}

fn constants(column: usize) -> Vec<String> {
    let mut values: Vec<String> = rows()
        .iter()
        .filter_map(|row| row[column].clone())
        .chain(
            MIXED
                .iter()
                .filter_map(|row| row[column].map(str::to_string)),
        )
        .collect();
    values.extend(extra_constants(column));
    values.sort();
    values.dedup();
    values
}

/// Every predicate this file can check; [`check`] samples them unless
/// `NUMERIC_STATS_ALL_PREDICATES` is set.
fn predicates() -> Vec<String> {
    let mut predicates = Vec::new();
    for (index, (name, _, cast)) in COLUMNS.iter().enumerate() {
        let constants = constants(index);
        for (position, value) in constants.iter().enumerate() {
            let literal = format!("CAST('{value}' AS {cast})");
            for op in ["=", "<", "<=", ">", ">=", "<>"] {
                predicates.push(format!("{name} {op} {literal}"));
            }
            predicates.push(format!("{name} BETWEEN {literal} AND {literal}"));
            let other = &constants[(position + 1) % constants.len()];
            predicates.push(format!("{name} IN ({literal}, CAST('{other}' AS {cast}))"));
        }
    }
    predicates
}

fn duckdb_row<S: AsRef<str>>(id: usize, row: &[Option<S>; 4]) -> String {
    let cells = row
        .iter()
        .zip(COLUMNS)
        .map(|(value, (_, ty, _))| match value {
            Some(v) => format!("CAST('{}' AS {ty})", v.as_ref()),
            None => format!("CAST(NULL AS {ty})"),
        })
        .collect::<Vec<_>>();
    format!("({id}, {})", cells.join(", "))
}

fn crate_row<S: AsRef<str>>(id: usize, row: &[Option<S>; 4]) -> String {
    let cells = row
        .iter()
        .zip(COLUMNS)
        .map(|(value, (_, _, ty))| match value {
            Some(v) => format!("CAST('{}' AS {ty})", v.as_ref()),
            None => format!("CAST(NULL AS {ty})"),
        })
        .collect::<Vec<_>>();
    format!("({id}, {})", cells.join(", "))
}

/// One `INSERT` per data file, in official DuckLake's SQL.
fn official_statements() -> Vec<String> {
    let columns = COLUMNS
        .iter()
        .map(|(name, ty, _)| format!("{name} {ty}"))
        .collect::<Vec<_>>()
        .join(", ");
    let mut statements = vec![format!("CREATE TABLE lake.t (id BIGINT, {columns})")];
    for (id, row) in rows().iter().enumerate() {
        statements.push(format!("INSERT INTO lake.t VALUES {}", duckdb_row(id, row)));
    }
    let mixed = MIXED
        .iter()
        .enumerate()
        .map(|(i, row)| duckdb_row(100 + i, row))
        .collect::<Vec<_>>();
    statements.push(format!("INSERT INTO lake.t VALUES {}", mixed.join(", ")));
    statements
}

/// Write the fixture through official DuckLake at `attach_target`.
fn official_fixture(attach_target: &str, extensions: &[&str], data: &Path) {
    let statements = official_statements();
    let statements = statements.iter().map(String::as_str).collect::<Vec<_>>();
    with_official_ducklake(attach_target, extensions, data, &statements)
        .expect("official fixture builds");
}

/// Write the fixture through this crate's `writer`, one data file per `INSERT`.
async fn crate_fixture(writer: Arc<dyn MetadataWriter>, provider: Arc<dyn MetadataProvider>) {
    let mut columns =
        vec![ColumnDef::from_arrow("id", &arrow::datatypes::DataType::Int64, true).unwrap()];
    let types = [
        arrow::datatypes::DataType::Float32,
        arrow::datatypes::DataType::Float64,
        arrow::datatypes::DataType::Decimal128(38, 10),
        arrow::datatypes::DataType::Int64,
    ];
    for ((name, _, _), ty) in COLUMNS.iter().zip(&types) {
        columns.push(ColumnDef::from_arrow(*name, ty, true).unwrap());
    }
    let setup = writer
        .begin_write_transaction("main", "t", &columns, WriteMode::Replace)
        .unwrap();
    writer
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
    writer
        .set_table_setting(setup.table_id, "data_inlining_row_limit", "0")
        .unwrap();
    let mut inserts: Vec<String> = rows()
        .iter()
        .enumerate()
        .map(|(id, row)| format!("INSERT INTO lake.main.t VALUES {}", crate_row(id, row)))
        .collect();
    inserts.push(format!(
        "INSERT INTO lake.main.t VALUES {}",
        MIXED
            .iter()
            .enumerate()
            .map(|(i, row)| crate_row(100 + i, row))
            .collect::<Vec<_>>()
            .join(", ")
    ));
    for insert in inserts {
        let ctx = SessionContext::new();
        let catalog =
            DuckLakeCatalog::with_writer(Arc::clone(&provider), Arc::clone(&writer)).unwrap();
        DuckLakeCatalog::register(
            &ctx,
            "lake",
            DuckLakeCatalog::with_writer(Arc::clone(&provider), Arc::clone(&writer)).unwrap(),
        );
        execute_ducklake_sql(&ctx, &catalog, &insert)
            .await
            .unwrap_or_else(|e| panic!("`{insert}`: {e}"))
            .collect()
            .await
            .unwrap_or_else(|e| panic!("`{insert}`: {e}"));
    }
}

/// Run every predicate with the statistics filter honoured and ignored, and
/// fail on any difference.
async fn check(label: &str, provider: Arc<dyn MetadataProvider>) {
    let mut failures = Vec::new();
    let mut pruned = 0;
    let mut non_empty = 0;
    // The default suite runs every fifth predicate, a stride coprime to the
    // eight per constant so the operators rotate across constants;
    // `NUMERIC_STATS_ALL_PREDICATES` runs every one.
    let stride = if std::env::var("NUMERIC_STATS_ALL_PREDICATES").is_ok() {
        1
    } else {
        5
    };
    let predicates = predicates().into_iter().step_by(stride).collect::<Vec<_>>();
    for predicate in &predicates {
        let sql = format!("SELECT id FROM lake.main.t WHERE {predicate}");
        let truth = run_query(&provider, false, &sql)
            .await
            .unwrap_or_else(|e| panic!("{label}: `{sql}` without pushdown: {e}"));
        let got = match run_query(&provider, true, &sql).await {
            Ok(got) => got,
            Err(e) => {
                failures.push(format!("`{predicate}`: error {e}"));
                continue;
            },
        };
        if !truth.rows.is_empty() {
            non_empty += 1;
        }
        if got.listed < truth.listed {
            pruned += 1;
        }
        if got.rows != truth.rows {
            failures.push(format!(
                "`{predicate}`: ids {:?} with pushdown, {:?} without",
                got.rows, truth.rows
            ));
        }
    }
    eprintln!(
        "{label}: {} predicates, {non_empty} non-empty, {pruned} pruned, {} wrong",
        predicates.len(),
        failures.len()
    );
    assert!(
        failures.is_empty(),
        "{label}: {} of {} predicates lost or gained rows:\n{}",
        failures.len(),
        predicates.len(),
        failures.join("\n")
    );
    assert!(pruned > 0, "{label}: no predicate pruned a file");
}

/// Respell the `FLOAT` column's bounds the way a `%.9g` writer spells them.
///
/// Each is still a correctly rounded spelling of the same `f32` (it reads back
/// as that float), but it is not the shortest one, so as a `DOUBLE` it is not
/// the double the constant's shortest text reads as. A filter that compares
/// the two texts as doubles drops the file.
const RESPELL_FLOAT_BOUNDS: &str = "UPDATE ducklake_file_column_stats SET
    min_value = CASE min_value
        WHEN '1e+20' THEN '1.00000002e+20' WHEN '-1e+20' THEN '-1.00000002e+20'
        WHEN '1e-07' THEN '9.99999994e-08' WHEN '3.4028235e+38' THEN '3.40282347e+38'
        WHEN '-3.4028235e+38' THEN '-3.40282347e+38' ELSE min_value END,
    max_value = CASE max_value
        WHEN '1e+20' THEN '1.00000002e+20' WHEN '-1e+20' THEN '-1.00000002e+20'
        WHEN '1e-07' THEN '9.99999994e-08' WHEN '3.4028235e+38' THEN '3.40282347e+38'
        WHEN '-3.4028235e+38' THEN '-3.40282347e+38' ELSE max_value END
    WHERE column_id IN (SELECT column_id FROM ducklake_column WHERE column_name = 'r')";

// ---------------------------------------------------------------------------
// SQLite
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn numeric_stats_filter_sqlite_official() {
    use datafusion_ducklake::SqliteMetadataProvider;
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("official.db");
    official_fixture(
        &format!("ducklake:sqlite:{}", path.display()),
        &["sqlite"],
        &temp.path().join("data"),
    );
    let provider: Arc<dyn MetadataProvider> = Arc::new(
        SqliteMetadataProvider::new(&format!("sqlite:{}", path.display()))
            .await
            .unwrap(),
    );
    check("sqlite/official", provider).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn numeric_stats_filter_sqlite_crate() {
    use datafusion_ducklake::{SqliteMetadataProvider, SqliteMetadataWriter};
    let temp = TempDir::new().unwrap();
    let data = temp.path().join("data");
    std::fs::create_dir_all(&data).unwrap();
    let url = format!("sqlite:{}?mode=rwc", temp.path().join("crate.db").display());
    let writer = SqliteMetadataWriter::new_with_init(&url).await.unwrap();
    writer.set_data_path(data.to_str().unwrap()).unwrap();
    let snapshot = writer.create_snapshot().unwrap();
    writer.get_or_create_schema("main", None, snapshot).unwrap();
    let provider: Arc<dyn MetadataProvider> =
        Arc::new(SqliteMetadataProvider::new(&url).await.unwrap());
    crate_fixture(Arc::new(writer), Arc::clone(&provider)).await;
    check("sqlite/crate", Arc::clone(&provider)).await;

    let pool = sqlx::SqlitePool::connect(&url).await.unwrap();
    sqlx::query(RESPELL_FLOAT_BOUNDS)
        .execute(&pool)
        .await
        .unwrap();
    pool.close().await;
    check("sqlite/crate respelled", provider).await;
}

// ---------------------------------------------------------------------------
// DuckDB
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn numeric_stats_filter_duckdb_official() {
    use datafusion_ducklake::DuckdbMetadataProvider;
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("official.ducklake");
    official_fixture(
        &format!("ducklake:{}", path.display()),
        &[],
        &temp.path().join("data"),
    );
    let provider: Arc<dyn MetadataProvider> =
        Arc::new(DuckdbMetadataProvider::new(path.display().to_string()).unwrap());
    check("duckdb/official", provider).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn numeric_stats_filter_duckdb_crate() {
    use datafusion_ducklake::DuckdbMetadataWriter;
    let temp = TempDir::new().unwrap();
    let data = temp.path().join("data");
    std::fs::create_dir_all(&data).unwrap();
    let path = temp.path().join("crate.duckdb");
    let writer = DuckdbMetadataWriter::new_with_init(path.display().to_string()).unwrap();
    writer.set_data_path(data.to_str().unwrap()).unwrap();
    let snapshot = writer.create_snapshot().unwrap();
    writer.get_or_create_schema("main", None, snapshot).unwrap();
    let provider: Arc<dyn MetadataProvider> = Arc::new(writer.metadata_provider());
    crate_fixture(Arc::new(writer), Arc::clone(&provider)).await;
    check("duckdb/crate", provider).await;
}

// ---------------------------------------------------------------------------
// PostgreSQL
// ---------------------------------------------------------------------------

#[cfg(feature = "write-postgres")]
async fn postgres_database(port: u16, name: &str) -> String {
    let admin = sqlx::PgPool::connect(&format!(
        "postgresql://postgres:postgres@127.0.0.1:{port}/postgres"
    ))
    .await
    .unwrap();
    sqlx::query(sqlx::AssertSqlSafe(format!("CREATE DATABASE {name}")))
        .execute(&admin)
        .await
        .unwrap();
    admin.close().await;
    format!("postgresql://postgres:postgres@127.0.0.1:{port}/{name}")
}

#[cfg(feature = "write-postgres")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
#[tokio::test(flavor = "multi_thread")]
async fn numeric_stats_filter_postgres() {
    use datafusion_ducklake::{PostgresMetadataProvider, PostgresSingleCatalogMetadataWriter};
    use testcontainers::runners::AsyncRunner;
    use testcontainers_modules::postgres::Postgres;

    let container = Postgres::default().start().await.unwrap();
    let port = container.get_host_port_ipv4(5432).await.unwrap();
    let temp = TempDir::new().unwrap();

    let url = postgres_database(port, "official").await;
    official_fixture(
        &format!(
            "ducklake:postgres:host=127.0.0.1 port={port} dbname=official \
             user=postgres password=postgres"
        ),
        &["postgres"],
        &temp.path().join("official_data"),
    );
    let provider: Arc<dyn MetadataProvider> =
        Arc::new(PostgresMetadataProvider::new(&url).await.unwrap());
    check("postgres/official", provider).await;

    let url = postgres_database(port, "ours").await;
    let data = temp.path().join("crate_data");
    std::fs::create_dir_all(&data).unwrap();
    let writer = PostgresSingleCatalogMetadataWriter::new_with_init(&url)
        .await
        .unwrap();
    writer.set_data_path(data.to_str().unwrap()).unwrap();
    let snapshot = writer.create_snapshot().unwrap();
    writer.get_or_create_schema("main", None, snapshot).unwrap();
    let provider: Arc<dyn MetadataProvider> =
        Arc::new(PostgresMetadataProvider::new(&url).await.unwrap());
    crate_fixture(Arc::new(writer), Arc::clone(&provider)).await;
    check("postgres/crate", Arc::clone(&provider)).await;

    let pool = sqlx::PgPool::connect(&url).await.unwrap();
    sqlx::query(RESPELL_FLOAT_BOUNDS)
        .execute(&pool)
        .await
        .unwrap();
    pool.close().await;
    check("postgres/crate respelled", provider).await;
}

// ---------------------------------------------------------------------------
// MySQL
// ---------------------------------------------------------------------------

#[cfg(feature = "write-mysql")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
#[tokio::test(flavor = "multi_thread")]
async fn numeric_stats_filter_mysql() {
    use datafusion_ducklake::{MySqlMetadataProvider, MySqlMetadataWriter};
    use testcontainers::runners::AsyncRunner;
    use testcontainers_modules::mysql::Mysql;

    let container = Mysql::default().start().await.unwrap();
    let port = container.get_host_port_ipv4(3306).await.unwrap();
    let temp = TempDir::new().unwrap();
    let admin = sqlx::MySqlPool::connect(&format!("mysql://root@127.0.0.1:{port}/test"))
        .await
        .unwrap();
    for database in ["official", "ours"] {
        sqlx::query(sqlx::AssertSqlSafe(format!("CREATE DATABASE {database}")))
            .execute(&admin)
            .await
            .unwrap();
    }
    admin.close().await;

    // Official DuckLake cannot write a MySQL catalog (see
    // `transport_catalog_to_mysql`), so its DuckDB catalog is copied across.
    let path = temp.path().join("official.ducklake");
    official_fixture(
        &format!("ducklake:{}", path.display()),
        &[],
        &temp.path().join("official_data"),
    );
    crate::pushdown_row_preservation_tests::transport_catalog_to_mysql(
        &path,
        &format!("host=127.0.0.1 port={port} user=root database=official"),
    )
    .expect("catalog transports to mysql");
    let provider: Arc<dyn MetadataProvider> = Arc::new(
        MySqlMetadataProvider::new(&format!("mysql://root@127.0.0.1:{port}/official"))
            .await
            .unwrap(),
    );
    check("mysql/official", provider).await;

    let url = format!("mysql://root@127.0.0.1:{port}/ours");
    let data = temp.path().join("crate_data");
    std::fs::create_dir_all(&data).unwrap();
    let writer = MySqlMetadataWriter::new_with_init(&url).await.unwrap();
    writer.set_data_path(data.to_str().unwrap()).unwrap();
    let snapshot = writer.create_snapshot().unwrap();
    writer.get_or_create_schema("main", None, snapshot).unwrap();
    let provider: Arc<dyn MetadataProvider> =
        Arc::new(MySqlMetadataProvider::new(&url).await.unwrap());
    crate_fixture(Arc::new(writer), Arc::clone(&provider)).await;
    check("mysql/crate", Arc::clone(&provider)).await;

    let pool = sqlx::MySqlPool::connect(&url).await.unwrap();
    sqlx::query(RESPELL_FLOAT_BOUNDS)
        .execute(&pool)
        .await
        .unwrap();
    pool.close().await;
    check("mysql/crate respelled", provider).await;
}
