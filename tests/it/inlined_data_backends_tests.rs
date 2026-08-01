//! Integration tests for inlined rows on DuckDB, PostgreSQL, and MySQL catalogs.

#![cfg(feature = "metadata-duckdb")]

use std::path::Path;
use std::sync::Arc;

use arrow::array::{Array, BooleanArray, Int32Array, Int64Array, StringViewArray};
use arrow::record_batch::RecordBatch;
use datafusion::datasource::MemTable;
use datafusion::prelude::*;
use datafusion::scalar::ScalarValue;
use datafusion_ducklake::metadata_provider::DuckLakeTableColumn;
use datafusion_ducklake::{DuckLakeCatalog, DuckdbMetadataProvider, MetadataProvider};
use tempfile::TempDir;

use crate::common;

type Row = (Option<i32>, Option<String>, Option<bool>);

fn assert_binary_encodings(batch: &RecordBatch) {
    assert_eq!(
        ScalarValue::try_from_array(batch.column(3), 0).unwrap(),
        ScalarValue::BinaryView(Some(vec![0x00, 0xff]))
    );
    assert_eq!(
        ScalarValue::try_from_array(batch.column(4), 0).unwrap(),
        ScalarValue::FixedSizeBinary(
            16,
            Some(vec![
                0x55, 0x0e, 0x84, 0x00, 0xe2, 0x9b, 0x41, 0xd4, 0xa7, 0x16, 0x44, 0x66, 0x55, 0x44,
                0x00, 0x00,
            ]),
        )
    );
}

fn assert_typed_encodings(batch: &RecordBatch) {
    assert_eq!(
        ScalarValue::try_from_array(batch.column(5), 0).unwrap(),
        ScalarValue::Decimal128(Some(12_345), 10, 2)
    );
    assert_eq!(
        ScalarValue::try_from_array(batch.column(6), 0).unwrap(),
        ScalarValue::Date32(Some(1))
    );
    assert_eq!(
        ScalarValue::try_from_array(batch.column(7), 0).unwrap(),
        ScalarValue::Time64Microsecond(Some(1_000_002))
    );
    assert_eq!(
        ScalarValue::try_from_array(batch.column(8), 0).unwrap(),
        ScalarValue::TimestampMicrosecond(Some(1_000_002), None)
    );
    assert_eq!(
        ScalarValue::try_from_array(batch.column(9), 0).unwrap(),
        ScalarValue::Float64(Some(3.5))
    );
}

fn escaped_path(path: &Path) -> String {
    path.to_string_lossy().replace('\'', "''")
}

fn duckdb_rows(conn: &duckdb::Connection, catalog: &str) -> Vec<Row> {
    let mut statement = conn
        .prepare(&format!(
            "SELECT id, note, active FROM {catalog}.main.items ORDER BY id NULLS LAST"
        ))
        .unwrap();
    statement
        .query_map([], |row| {
            Ok((
                row.get::<_, Option<i32>>(0)?,
                row.get::<_, Option<String>>(1)?,
                row.get::<_, Option<bool>>(2)?,
            ))
        })
        .unwrap()
        .collect::<std::result::Result<Vec<_>, _>>()
        .unwrap()
}

async fn datafusion_rows(ctx: &SessionContext, relation: &str) -> Vec<Row> {
    let batches = ctx
        .sql(&format!(
            "SELECT id, note, active FROM {relation} ORDER BY id NULLS LAST"
        ))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let mut rows = Vec::new();
    for batch in batches {
        let ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let notes = batch
            .column(1)
            .as_any()
            .downcast_ref::<StringViewArray>()
            .unwrap();
        let active = batch
            .column(2)
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap();
        for index in 0..batch.num_rows() {
            rows.push((
                (!ids.is_null(index)).then(|| ids.value(index)),
                (!notes.is_null(index)).then(|| notes.value(index).to_string()),
                (!active.is_null(index)).then(|| active.value(index)),
            ));
        }
    }
    rows
}

async fn datafusion_count(ctx: &SessionContext, relation: &str) -> i64 {
    let batches = ctx
        .sql(&format!("SELECT COUNT(*) FROM {relation}"))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0)
}

#[tokio::test]
async fn duckdb_inlined_rows_match_the_extension() {
    let temp = TempDir::new().unwrap();
    let catalog_path = temp.path().join("inlined.ducklake");
    let data_path = temp.path().join("data");
    let expected = {
        common::ensure_ducklake_installed();
        let conn = duckdb::Connection::open_in_memory().unwrap();
        conn.execute("LOAD ducklake", []).unwrap();
        conn.execute(
            &format!(
                "ATTACH 'ducklake:{}' AS lake (DATA_PATH '{}', DATA_INLINING_ROW_LIMIT 10)",
                escaped_path(&catalog_path),
                escaped_path(&data_path)
            ),
            [],
        )
        .unwrap();
        conn.execute(
            "CREATE TABLE lake.items(
                 id INTEGER, note VARCHAR, active BOOLEAN, payload BLOB, token UUID,
                 amount DECIMAL(10,2), event_date DATE, event_time TIME,
                 event_ts TIMESTAMP, score DOUBLE)",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO lake.items VALUES
                 (7, 'alpha', true, from_hex('00ff'), '550e8400-e29b-41d4-a716-446655440000',
                  123.45, '1970-01-02', '00:00:01.000002',
                  '1970-01-01 00:00:01.000002', 3.5),
                 (11, NULL, false, NULL, NULL, NULL, NULL, NULL, NULL, NULL),
                 (NULL, 'omega', NULL, from_hex('01'), '123e4567-e89b-12d3-a456-426614174000',
                  -7.25, '2025-12-31', '23:59:59', '2025-12-31 23:59:59', -1.25)",
            [],
        )
        .unwrap();
        let rows = duckdb_rows(&conn, "lake");
        let file_count: i64 = conn
            .query_row(
                &format!(
                    "SELECT COUNT(*) FROM glob('{}/*')",
                    escaped_path(&data_path)
                ),
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(file_count, 0, "the oracle insert must be inlined");
        rows
    };

    let provider = DuckdbMetadataProvider::new(catalog_path.to_string_lossy().to_string()).unwrap();
    let snapshot_id = provider.get_current_snapshot().unwrap();
    let schema = provider
        .get_schema_by_name("main", snapshot_id)
        .unwrap()
        .unwrap();
    let table = provider
        .get_table_by_name(schema.schema_id, "items", snapshot_id)
        .unwrap()
        .unwrap();
    let columns = provider
        .get_table_structure(table.table_id, snapshot_id)
        .unwrap();
    let inlined = provider
        .get_inlined_data(table.table_id, snapshot_id, &columns)
        .unwrap();
    assert_binary_encodings(&inlined[0]);
    assert_typed_encodings(&inlined[0]);
    let catalog = DuckLakeCatalog::new(provider).unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("lake", Arc::new(catalog));

    assert_eq!(datafusion_rows(&ctx, "lake.main.items").await, expected);
    assert_eq!(
        datafusion_count(&ctx, "lake.main.items").await,
        expected.len() as i64
    );
}

#[cfg(feature = "metadata-postgres")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
#[tokio::test(flavor = "multi_thread")]
async fn postgres_inlined_rows_follow_the_catalog_encoding() {
    use datafusion_ducklake::PostgresMetadataProvider;
    use sqlx::AssertSqlSafe;
    use sqlx::PgPool;
    use testcontainers::runners::AsyncRunner;
    use testcontainers_modules::postgres::Postgres;

    let container = Postgres::default().start().await.unwrap();
    let port = container.get_host_port_ipv4(5432).await.unwrap();
    let url = format!("postgresql://postgres:postgres@127.0.0.1:{port}/postgres");
    let libpq =
        format!("host=127.0.0.1 port={port} dbname=postgres user=postgres password=postgres");
    let temp = TempDir::new().unwrap();
    let data_path = temp.path().join("data");
    {
        common::ensure_ducklake_installed();
        let conn = duckdb::Connection::open_in_memory().unwrap();
        conn.execute_batch("INSTALL postgres; LOAD postgres; LOAD ducklake;")
            .unwrap();
        conn.execute(
            &format!(
                "ATTACH 'ducklake:postgres:{libpq}' AS lake (DATA_PATH '{}', DATA_INLINING_ROW_LIMIT 0)",
                escaped_path(&data_path)
            ),
            [],
        )
        .unwrap();
        conn.execute(
            "CREATE TABLE lake.items(
                 id INTEGER, note VARCHAR, active BOOLEAN, payload BLOB, token UUID,
                 amount DECIMAL(10,2), event_date DATE, event_time TIME,
                 event_ts TIMESTAMP, score DOUBLE)",
            [],
        )
        .unwrap();
        conn.execute("DETACH lake", []).unwrap();
    }

    let pool = PgPool::connect(&url).await.unwrap();
    let (table_id, snapshot_id, schema_version): (i64, i64, i64) = sqlx::query_as(
        "SELECT t.table_id, s.snapshot_id, s.schema_version
         FROM ducklake_table t CROSS JOIN ducklake_snapshot s
         WHERE t.table_name = 'items'
         ORDER BY s.snapshot_id DESC LIMIT 1",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    let physical = format!("ducklake_inlined_data_{table_id}_{schema_version}");
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS ducklake_inlined_data_tables(
             table_id BIGINT, table_name VARCHAR, schema_version BIGINT)",
    )
    .execute(&pool)
    .await
    .unwrap();
    sqlx::query(AssertSqlSafe(format!(
        "CREATE TABLE {physical}(
             row_id BIGINT, begin_snapshot BIGINT, end_snapshot BIGINT,
             id INTEGER, note BYTEA, active BOOLEAN, payload BYTEA, token UUID,
             amount DECIMAL(10,2), event_date VARCHAR, event_time TIME,
             event_ts VARCHAR, score DOUBLE PRECISION)"
    )))
    .execute(&pool)
    .await
    .unwrap();
    sqlx::query(
        "INSERT INTO ducklake_inlined_data_tables(table_id, table_name, schema_version)
         VALUES ($1, $2, $3)",
    )
    .bind(table_id)
    .bind(&physical)
    .bind(schema_version)
    .execute(&pool)
    .await
    .unwrap();
    sqlx::query(AssertSqlSafe(format!(
        "INSERT INTO {physical} VALUES
             (0, $1, NULL, 7, convert_to('alpha', 'UTF8'), TRUE, decode('00ff', 'hex'),
              '550e8400-e29b-41d4-a716-446655440000', 123.45, '1970-01-02',
              '00:00:01.000002', '1970-01-01 00:00:01.000002', 3.5),
             (1, $1, $1, 11, NULL, FALSE, NULL, NULL, NULL, NULL, NULL, NULL, NULL),
             (2, $1, NULL, NULL, convert_to('omega', 'UTF8'), NULL, decode('01', 'hex'),
              '123e4567-e89b-12d3-a456-426614174000', -7.25, '2025-12-31', '23:59:59',
              '2025-12-31 23:59:59', -1.25)"
    )))
    .bind(snapshot_id)
    .execute(&pool)
    .await
    .unwrap();

    let provider = PostgresMetadataProvider::new(&url).await.unwrap();
    let columns = vec![
        DuckLakeTableColumn::new(1, "id".to_string(), "int32".to_string(), true),
        DuckLakeTableColumn::new(2, "note".to_string(), "varchar".to_string(), true),
        DuckLakeTableColumn::new(3, "active".to_string(), "boolean".to_string(), true),
        DuckLakeTableColumn::new(4, "payload".to_string(), "blob".to_string(), true),
        DuckLakeTableColumn::new(5, "token".to_string(), "uuid".to_string(), true),
        DuckLakeTableColumn::new(6, "amount".to_string(), "decimal(10,2)".to_string(), true),
        DuckLakeTableColumn::new(7, "event_date".to_string(), "date".to_string(), true),
        DuckLakeTableColumn::new(8, "event_time".to_string(), "time".to_string(), true),
        DuckLakeTableColumn::new(9, "event_ts".to_string(), "timestamp".to_string(), true),
        DuckLakeTableColumn::new(10, "score".to_string(), "float64".to_string(), true),
    ];
    let batches = provider
        .get_inlined_data(table_id, snapshot_id, &columns)
        .unwrap();
    assert_binary_encodings(&batches[0]);
    assert_typed_encodings(&batches[0]);
    let table = MemTable::try_new(batches[0].schema(), vec![batches]).unwrap();
    let ctx = SessionContext::new();
    ctx.register_table("items", Arc::new(table)).unwrap();

    let expected = vec![
        (Some(7), Some("alpha".to_string()), Some(true)),
        (None, Some("omega".to_string()), None),
    ];
    assert_eq!(datafusion_rows(&ctx, "items").await, expected);
    assert_eq!(datafusion_count(&ctx, "items").await, expected.len() as i64);
}

#[cfg(all(feature = "metadata-mysql", feature = "write-mysql"))]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
#[tokio::test(flavor = "multi_thread")]
async fn mysql_sqlite_style_inlined_rows_are_visible() {
    use datafusion_ducklake::{MySqlMetadataProvider, MySqlMetadataWriter};
    use sqlx::MySqlPool;
    use testcontainers::runners::AsyncRunner;
    use testcontainers_modules::mysql::Mysql;

    let container = Mysql::default().start().await.unwrap();
    let port = container.get_host_port_ipv4(3306).await.unwrap();
    let url = format!("mysql://root@127.0.0.1:{port}/test");
    MySqlMetadataWriter::new_with_init(&url).await.unwrap();
    let pool = MySqlPool::connect(&url).await.unwrap();

    sqlx::query("INSERT INTO ducklake_snapshot(snapshot_id, schema_version) VALUES (1, 1), (2, 1)")
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query(
        "INSERT INTO ducklake_metadata(`key`, `value`, scope) VALUES ('data_path', '/tmp', NULL)",
    )
    .execute(&pool)
    .await
    .unwrap();
    sqlx::query(
        "INSERT INTO ducklake_schema(schema_id, schema_name, path, path_is_relative, begin_snapshot)
         VALUES (1, 'main', '', TRUE, 1)",
    )
    .execute(&pool)
    .await
    .unwrap();
    sqlx::query(
        "INSERT INTO ducklake_table(table_id, schema_id, table_name, path, path_is_relative, begin_snapshot)
         VALUES (1, 1, 'items', '', TRUE, 1)",
    )
    .execute(&pool)
    .await
    .unwrap();
    for (column_id, column_name, column_type) in [
        (1_i64, "id", "int32"),
        (2, "note", "varchar"),
        (3, "active", "boolean"),
        (4, "payload", "blob"),
        (5, "token", "uuid"),
        (6, "amount", "decimal(10,2)"),
        (7, "event_date", "date"),
        (8, "event_time", "time"),
        (9, "event_ts", "timestamp"),
        (10, "score", "float64"),
    ] {
        sqlx::query(
            "INSERT INTO ducklake_column(
                 column_id, begin_snapshot, table_id, column_order, column_name, column_type,
                 nulls_allowed)
             VALUES (?, 1, 1, ?, ?, ?, TRUE)",
        )
        .bind(column_id)
        .bind(column_id)
        .bind(column_name)
        .bind(column_type)
        .execute(&pool)
        .await
        .unwrap();
    }
    sqlx::query(
        "CREATE TABLE ducklake_inlined_data_tables(
             table_id BIGINT, table_name VARCHAR(255), schema_version BIGINT)",
    )
    .execute(&pool)
    .await
    .unwrap();
    sqlx::query(
        "CREATE TABLE ducklake_inlined_data_1_1(
             row_id BIGINT, begin_snapshot BIGINT, end_snapshot BIGINT,
             id BIGINT, note TEXT, active BIGINT, payload BLOB, token VARCHAR(36),
             amount TEXT, event_date TEXT, event_time TEXT, event_ts TEXT, score TEXT)",
    )
    .execute(&pool)
    .await
    .unwrap();
    sqlx::query(
        "INSERT INTO ducklake_inlined_data_tables VALUES (1, 'ducklake_inlined_data_1_1', 1)",
    )
    .execute(&pool)
    .await
    .unwrap();
    sqlx::query(
        "INSERT INTO ducklake_inlined_data_1_1 VALUES
             (0, 1, NULL, 7, 'alpha', 1, X'00ff', '550e8400-e29b-41d4-a716-446655440000',
              '123.45', '1970-01-02', '00:00:01.000002',
              '1970-01-01 00:00:01.000002', '3.5'),
             (1, 1, 2, 11, NULL, 0, NULL, NULL, NULL, NULL, NULL, NULL, NULL),
             (2, 1, NULL, NULL, 'omega', NULL, X'01',
              '123e4567-e89b-12d3-a456-426614174000', '-7.25', '2025-12-31', '23:59:59',
              '2025-12-31 23:59:59', '-1.25')",
    )
    .execute(&pool)
    .await
    .unwrap();

    let provider = MySqlMetadataProvider::new(&url).await.unwrap();
    let columns = vec![
        DuckLakeTableColumn::new(1, "id".to_string(), "int32".to_string(), true),
        DuckLakeTableColumn::new(2, "note".to_string(), "varchar".to_string(), true),
        DuckLakeTableColumn::new(3, "active".to_string(), "boolean".to_string(), true),
        DuckLakeTableColumn::new(4, "payload".to_string(), "blob".to_string(), true),
        DuckLakeTableColumn::new(5, "token".to_string(), "uuid".to_string(), true),
        DuckLakeTableColumn::new(6, "amount".to_string(), "decimal(10,2)".to_string(), true),
        DuckLakeTableColumn::new(7, "event_date".to_string(), "date".to_string(), true),
        DuckLakeTableColumn::new(8, "event_time".to_string(), "time".to_string(), true),
        DuckLakeTableColumn::new(9, "event_ts".to_string(), "timestamp".to_string(), true),
        DuckLakeTableColumn::new(10, "score".to_string(), "float64".to_string(), true),
    ];
    let batches = provider.get_inlined_data(1, 2, &columns).unwrap();
    assert_binary_encodings(&batches[0]);
    assert_typed_encodings(&batches[0]);
    let catalog = DuckLakeCatalog::new(provider).unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("lake", Arc::new(catalog));

    let expected = vec![
        (Some(7), Some("alpha".to_string()), Some(true)),
        (None, Some("omega".to_string()), None),
    ];
    assert_eq!(datafusion_rows(&ctx, "lake.main.items").await, expected);
    assert_eq!(
        datafusion_count(&ctx, "lake.main.items").await,
        expected.len() as i64
    );
}
