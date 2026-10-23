//! Integration tests for inlined rows on DuckDB, PostgreSQL, and MySQL catalogs.

#![cfg(feature = "metadata-duckdb")]

use std::any::Any;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{Array, BooleanArray, Int32Array, Int64Array, StringViewArray};
use arrow::record_batch::RecordBatch;
#[cfg(feature = "metadata-postgres")]
use datafusion::datasource::MemTable;
use datafusion::prelude::*;
use datafusion::scalar::ScalarValue;
use datafusion_ducklake::inlined_filter::{InlinedComparison, InlinedFilter, InlinedValue};
#[cfg(any(
    feature = "metadata-postgres",
    all(feature = "metadata-mysql", feature = "write-mysql")
))]
use datafusion_ducklake::metadata_provider::DuckLakeTableColumn;
use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeTable, DuckdbMetadataProvider, MetadataProvider,
};
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
    let filtered = provider
        .scan_inlined_data(
            table.table_id,
            snapshot_id,
            &columns,
            Some(&InlinedFilter::Comparison {
                column: "id".to_string(),
                op: InlinedComparison::Eq,
                value: InlinedValue::I64(7),
            }),
        )
        .unwrap();
    assert_eq!(filtered.materialized_row_count, 1);
    let case_mismatch = provider
        .scan_inlined_data(
            table.table_id,
            snapshot_id,
            &columns,
            Some(&InlinedFilter::Comparison {
                column: "note".to_string(),
                op: InlinedComparison::Eq,
                value: InlinedValue::Utf8("ALPHA".to_string()),
            }),
        )
        .unwrap();
    assert_eq!(case_mismatch.materialized_row_count, 0);
    let catalog = DuckLakeCatalog::new(provider).unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("lake", Arc::new(catalog));

    assert_eq!(datafusion_rows(&ctx, "lake.main.items").await, expected);
    assert_eq!(
        datafusion_count(&ctx, "lake.main.items").await,
        expected.len() as i64
    );
    let table_provider = ctx.table_provider("lake.main.items").await.unwrap();
    ctx.register_table("filtered_items", Arc::clone(&table_provider))
        .unwrap();
    assert_eq!(
        datafusion_rows(&ctx, "filtered_items WHERE id = 7").await,
        vec![(Some(7), Some("alpha".to_string()), Some(true))]
    );
    let ducklake_table = (table_provider.as_ref() as &dyn Any)
        .downcast_ref::<DuckLakeTable>()
        .expect("DuckLakeTable");
    assert_eq!(ducklake_table.inlined_materialized_row_count(), 1);
    assert_eq!(
        datafusion_rows(&ctx, "filtered_items WHERE id >= 7").await,
        vec![(Some(7), Some("alpha".to_string()), Some(true)), (Some(11), None, Some(false)),]
    );
    assert_eq!(ducklake_table.inlined_materialized_row_count(), 2);
    assert_eq!(
        datafusion_rows(&ctx, "filtered_items WHERE note IS NULL").await,
        vec![(Some(11), None, Some(false))]
    );
    assert_eq!(ducklake_table.inlined_materialized_row_count(), 1);
    assert_eq!(
        datafusion_rows(&ctx, "filtered_items WHERE note LIKE 'al%'").await,
        vec![(Some(7), Some("alpha".to_string()), Some(true))]
    );
    assert_eq!(ducklake_table.inlined_materialized_row_count(), 1);
    assert_eq!(
        datafusion_rows(&ctx, "filtered_items WHERE starts_with(note, 'al')").await,
        vec![(Some(7), Some("alpha".to_string()), Some(true))]
    );
    assert_eq!(ducklake_table.inlined_materialized_row_count(), 1);
    assert_eq!(
        datafusion_rows(&ctx, "filtered_items WHERE id = 7 OR abs(id) = 11").await,
        vec![(Some(7), Some("alpha".to_string()), Some(true)), (Some(11), None, Some(false)),]
    );
    assert_eq!(ducklake_table.inlined_materialized_row_count(), 3);
}

#[cfg(feature = "metadata-postgres")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
#[tokio::test(flavor = "multi_thread")]
async fn postgres_inlined_rows_decode_catalog_encodings() {
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
        // Initialize the official catalog schema without inline rows, then insert each
        // PostgreSQL physical encoding directly to isolate the decoder from the producer
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
    let filtered = provider
        .scan_inlined_data(
            table_id,
            snapshot_id,
            &columns,
            Some(&InlinedFilter::Comparison {
                column: "note".to_string(),
                op: InlinedComparison::Eq,
                value: InlinedValue::Utf8("alpha".to_string()),
            }),
        )
        .unwrap();
    assert_eq!(filtered.materialized_row_count, 1);
    let case_mismatch = provider
        .scan_inlined_data(
            table_id,
            snapshot_id,
            &columns,
            Some(&InlinedFilter::Comparison {
                column: "note".to_string(),
                op: InlinedComparison::Eq,
                value: InlinedValue::Utf8("ALPHA".to_string()),
            }),
        )
        .unwrap();
    assert_eq!(case_mismatch.materialized_row_count, 0);
    let literal_wildcards = provider
        .scan_inlined_data(
            table_id,
            snapshot_id,
            &columns,
            Some(&InlinedFilter::Prefix {
                column: "note".to_string(),
                prefix: "a%_".to_string(),
            }),
        )
        .unwrap();
    assert_eq!(literal_wildcards.materialized_row_count, 0);
    let range = provider
        .scan_inlined_data(
            table_id,
            snapshot_id,
            &columns,
            Some(&InlinedFilter::Comparison {
                column: "id".to_string(),
                op: InlinedComparison::GtEq,
                value: InlinedValue::I64(7),
            }),
        )
        .unwrap();
    assert_eq!(range.materialized_row_count, 1);
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
        "CREATE TABLE IF NOT EXISTS ducklake_inlined_data_tables(
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
    let filtered = provider
        .scan_inlined_data(
            1,
            2,
            &columns,
            Some(&InlinedFilter::Comparison {
                column: "note".to_string(),
                op: InlinedComparison::Eq,
                value: InlinedValue::Utf8("alpha".to_string()),
            }),
        )
        .unwrap();
    assert_eq!(filtered.materialized_row_count, 1);
    let case_mismatch = provider
        .scan_inlined_data(
            1,
            2,
            &columns,
            Some(&InlinedFilter::Comparison {
                column: "note".to_string(),
                op: InlinedComparison::Eq,
                value: InlinedValue::Utf8("ALPHA".to_string()),
            }),
        )
        .unwrap();
    assert_eq!(case_mismatch.materialized_row_count, 0);
    let literal_wildcards = provider
        .scan_inlined_data(
            1,
            2,
            &columns,
            Some(&InlinedFilter::Prefix {
                column: "note".to_string(),
                prefix: "a%_".to_string(),
            }),
        )
        .unwrap();
    assert_eq!(literal_wildcards.materialized_row_count, 0);
    let range = provider
        .scan_inlined_data(
            1,
            2,
            &columns,
            Some(&InlinedFilter::Comparison {
                column: "id".to_string(),
                op: InlinedComparison::GtEq,
                value: InlinedValue::I64(7),
            }),
        )
        .unwrap();
    assert_eq!(range.materialized_row_count, 1);
    let unsafe_text_range = provider
        .scan_inlined_data(
            1,
            2,
            &columns,
            Some(&InlinedFilter::Comparison {
                column: "score".to_string(),
                op: InlinedComparison::Gt,
                value: InlinedValue::F64(0.0),
            }),
        )
        .unwrap();
    assert_eq!(unsafe_text_range.materialized_row_count, 2);
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

/// Render one cell of a DuckDB or DataFusion result for a cross-engine diff.
#[cfg(feature = "metadata-postgres")]
fn duckdb_text(value: duckdb::types::Value) -> String {
    use duckdb::types::Value;
    match value {
        Value::Null => "NULL".to_string(),
        Value::Int(v) => v.to_string(),
        Value::BigInt(v) => v.to_string(),
        Value::Text(v) => v,
        other => panic!("unexpected duckdb value {other:?}"),
    }
}

/// The change feeds and `rowid` scans over a PostgreSQL catalog whose history
/// DuckDB wrote with inlining on — inlined inserts, updates and deletes of
/// inlined rows, inlined deletions of Parquet rows, and a rename between
/// inlined schema versions — match the extension's row for row.
#[cfg(feature = "metadata-postgres")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
#[tokio::test(flavor = "multi_thread")]
async fn postgres_inlined_change_feeds_match_the_extension() {
    use datafusion_ducklake::{PostgresMetadataProvider, register_ducklake_functions};
    use testcontainers::runners::AsyncRunner;
    use testcontainers_modules::postgres::Postgres;

    let container = Postgres::default().start().await.unwrap();
    let port = container.get_host_port_ipv4(5432).await.unwrap();
    let url = format!("postgresql://postgres:postgres@127.0.0.1:{port}/postgres");
    let libpq =
        format!("host=127.0.0.1 port={port} dbname=postgres user=postgres password=postgres");
    let temp = TempDir::new().unwrap();
    let data_path = temp.path().join("data");
    common::ensure_ducklake_installed();
    let conn = duckdb::Connection::open_in_memory().unwrap();
    conn.execute_batch("INSTALL postgres; LOAD postgres; LOAD ducklake;")
        .unwrap();
    conn.execute(
        &format!(
            "ATTACH 'ducklake:postgres:{libpq}' AS lake (DATA_PATH '{}', DATA_INLINING_ROW_LIMIT 10)",
            escaped_path(&data_path)
        ),
        [],
    )
    .unwrap();
    for statement in [
        "CREATE TABLE lake.t(id INTEGER, name VARCHAR)",
        "INSERT INTO lake.t VALUES (1, 'a'), (2, 'b'), (3, 'c')",
        "UPDATE lake.t SET name = 'B' WHERE id = 2",
        "DELETE FROM lake.t WHERE id = 1",
        "INSERT INTO lake.t SELECT i, 'p' || i FROM range(100, 130) r(i)",
        "DELETE FROM lake.t WHERE id IN (105, 106)",
        "UPDATE lake.t SET name = 'q' WHERE id = 110",
        "ALTER TABLE lake.t RENAME COLUMN name TO label",
        "INSERT INTO lake.t VALUES (4, 'd')",
        "DELETE FROM lake.t WHERE id = 3",
    ] {
        conn.execute_batch(statement).unwrap();
    }
    let last: i64 = conn
        .query_row(
            "SELECT max(snapshot_id) FROM ducklake_snapshots('lake')",
            [],
            |row| row.get(0),
        )
        .unwrap();
    let official_rows = |sql: &str| -> Vec<Vec<String>> {
        let mut stmt = conn.prepare(sql).unwrap();
        let mut rows: Vec<Vec<String>> = stmt
            .query_map([], |row| {
                let mut out = Vec::new();
                let mut i = 0;
                while let Ok(value) = row.get::<usize, duckdb::types::Value>(i) {
                    out.push(duckdb_text(value));
                    i += 1;
                }
                Ok(out)
            })
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        rows.sort();
        rows
    };
    let mut official = Vec::new();
    for (start, end) in [(0, last), (2, 4), (5, 7), (7, last), (last, last)] {
        official.push((
            format!("SELECT * FROM ducklake_table_changes('main.t', {start}, {end})"),
            official_rows(&format!(
                "SELECT * FROM ducklake_table_changes('lake', 'main', 't', {start}, {end})"
            )),
        ));
        official.push((
            format!("SELECT * FROM ducklake_table_insertions('main.t', {start}, {end})"),
            official_rows(&format!(
                "SELECT snapshot_id, rowid, * FROM \
                 ducklake_table_insertions('lake', 'main', 't', {start}, {end})"
            )),
        ));
        official.push((
            format!(
                "SELECT * EXCLUDE (change_type) \
                 FROM ducklake_table_deletions('main.t', {start}, {end})"
            ),
            official_rows(&format!(
                "SELECT snapshot_id, rowid, * FROM \
                 ducklake_table_deletions('lake', 'main', 't', {start}, {end})"
            )),
        ));
    }
    let official_scan = official_rows("SELECT *, rowid FROM lake.t");
    conn.execute("DETACH lake", []).unwrap();

    let provider: Arc<dyn MetadataProvider> =
        Arc::new(PostgresMetadataProvider::new(&url).await.unwrap());
    let ctx = SessionContext::new();
    register_ducklake_functions(&ctx, Arc::clone(&provider));
    let catalog = DuckLakeCatalog::new(PostgresMetadataProvider::new(&url).await.unwrap())
        .unwrap()
        .with_row_lineage(true);
    ctx.register_catalog("ducklake", Arc::new(catalog));
    let crate_rows = |batches: Vec<RecordBatch>| -> Vec<Vec<String>> {
        let mut rows: Vec<Vec<String>> = batches
            .iter()
            .flat_map(|batch| {
                (0..batch.num_rows())
                    .map(|row| {
                        (0..batch.num_columns())
                            .map(|column| {
                                let array = batch.column(column);
                                if array.is_null(row) {
                                    return "NULL".to_string();
                                }
                                datafusion::arrow::util::display::array_value_to_string(array, row)
                                    .unwrap()
                            })
                            .collect()
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        rows.sort();
        rows
    };
    for (sql, expected) in official {
        let batches = ctx.sql(&sql).await.unwrap().collect().await.unwrap();
        assert_eq!(crate_rows(batches), expected, "{sql}");
    }
    let batches = ctx
        .sql("SELECT * FROM ducklake.main.t")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(crate_rows(batches), official_scan);
}

/// Inlined rows written before `ALTER ... SET TYPE` widened a column hold
/// values of the older type; they read as the widened type, as DuckDB reads
/// them, on DuckDB and SQLite catalogs.
#[tokio::test(flavor = "multi_thread")]
async fn inlined_rows_read_through_a_widened_column_type() {
    let temp = TempDir::new().unwrap();
    let data_path = temp.path().join("data");
    common::ensure_ducklake_installed();
    common::ensure_extension_installed("sqlite");
    let conn = duckdb::Connection::open_in_memory().unwrap();
    conn.execute_batch("LOAD ducklake; LOAD sqlite;").unwrap();
    let catalogs = [
        (
            "duck",
            format!("ducklake:{}", escaped_path(&temp.path().join("a.ducklake"))),
        ),
        (
            "lite",
            format!(
                "ducklake:sqlite:{}",
                escaped_path(&temp.path().join("b.db"))
            ),
        ),
    ];
    let mut expected = Vec::new();
    for (name, uri) in &catalogs {
        let data = data_path.join(name);
        conn.execute_batch(&format!(
            "ATTACH '{uri}' AS {name} (DATA_PATH '{}', DATA_INLINING_ROW_LIMIT 10);
             CREATE TABLE {name}.t (i INTEGER, f FLOAT);
             INSERT INTO {name}.t VALUES (1, 0.5), (-2, NULL);
             ALTER TABLE {name}.t ALTER i SET TYPE BIGINT;
             ALTER TABLE {name}.t ALTER f SET TYPE DOUBLE;
             INSERT INTO {name}.t VALUES (10000000000, 1.25);",
            escaped_path(&data)
        ))
        .unwrap();
        let mut statement = conn
            .prepare(&format!("SELECT i, f FROM {name}.t ORDER BY i"))
            .unwrap();
        let rows = statement
            .query_map([], |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, Option<f64>>(1)?))
            })
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        expected.push(rows);
    }
    drop(conn);
    assert_eq!(
        expected[0],
        vec![(-2, None), (1, Some(0.5)), (10_000_000_000, Some(1.25))]
    );

    let mut providers: Vec<Arc<dyn MetadataProvider>> = vec![Arc::new(
        DuckdbMetadataProvider::new(temp.path().join("a.ducklake").to_string_lossy().to_string())
            .unwrap(),
    )];
    #[cfg(feature = "metadata-sqlite")]
    providers.push(Arc::new(
        datafusion_ducklake::SqliteMetadataProvider::new(&format!(
            "sqlite:{}",
            temp.path().join("b.db").display()
        ))
        .await
        .unwrap(),
    ));
    for (index, provider) in providers.into_iter().enumerate() {
        let ctx = SessionContext::new();
        let head = provider.get_current_snapshot().unwrap();
        ctx.register_catalog(
            "lake",
            Arc::new(DuckLakeCatalog::with_snapshot(provider, head).unwrap()),
        );
        let batches = ctx
            .sql("SELECT i, f FROM lake.main.t ORDER BY i")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let mut rows = Vec::new();
        for batch in &batches {
            let i = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            let f = batch
                .column(1)
                .as_any()
                .downcast_ref::<arrow::array::Float64Array>()
                .unwrap();
            for row in 0..batch.num_rows() {
                rows.push((i.value(row), f.is_valid(row).then(|| f.value(row))));
            }
        }
        assert_eq!(rows, expected[index], "catalog {}", catalogs[index].0);
    }
}

/// The nested-schema history every [`nested_evolution`] check replays on a
/// catalog `{c}`: inlined rows written under each shape of a struct, a list of
/// structs, a map of structs and a struct nested in a struct, as fields are
/// added (one with a default), renamed and dropped between the inserts.
const NESTED_EVOLUTION: &[&str] = &[
    "CREATE TABLE {c}.t(id INTEGER, s STRUCT(a INTEGER, b VARCHAR), ls STRUCT(x INTEGER)[], \
     m MAP(VARCHAR, STRUCT(p INTEGER)), nn STRUCT(i STRUCT(q INTEGER)))",
    "INSERT INTO {c}.t VALUES (1, {'a': 1, 'b': 'x'}, [{'x': 1}], MAP {'k': {'p': 1}}, \
     {'i': {'q': 1}}), (2, NULL, [NULL, {'x': NULL}], MAP {'k': NULL}, {'i': NULL})",
    "ALTER TABLE {c}.t ADD COLUMN s.c INTEGER DEFAULT 7",
    "INSERT INTO {c}.t VALUES (3, {'a': 3, 'b': 'z', 'c': NULL}, [], MAP {}, NULL)",
    "ALTER TABLE {c}.t ADD COLUMN ls.element.y INTEGER",
    "ALTER TABLE {c}.t ADD COLUMN m.value.r INTEGER",
    "ALTER TABLE {c}.t ADD COLUMN nn.i.w INTEGER",
    "INSERT INTO {c}.t VALUES (4, {'a': 4, 'b': 'w', 'c': 4}, [{'x': 4, 'y': 4}], \
     MAP {'k': {'p': 4, 'r': 4}}, {'i': {'q': 4, 'w': 4}})",
    "ALTER TABLE {c}.t RENAME COLUMN s.b TO bb",
    "INSERT INTO {c}.t VALUES (5, {'a': 5, 'bb': 'v', 'c': 5}, NULL, NULL, NULL)",
    "ALTER TABLE {c}.t DROP COLUMN s.a",
    "INSERT INTO {c}.t VALUES (6, {'bb': 'u', 'c': 6}, [{'x': 6, 'y': NULL}], \
     MAP {'k': {'p': 6, 'r': 6}}, {'i': {'q': 6, 'w': NULL}})",
];

/// Every row of `sql`'s result, one comma-joined line of displayed cells.
fn displayed_duckdb_rows(conn: &duckdb::Connection, sql: &str) -> Vec<String> {
    let mut statement = conn.prepare(sql).unwrap_or_else(|e| panic!("`{sql}`: {e}"));
    let mut lines = Vec::new();
    for batch in statement.query_arrow([]).unwrap() {
        for row in 0..batch.num_rows() {
            let cells = (0..batch.num_columns())
                .map(|column| {
                    let array = batch.column(column);
                    if array.is_null(row) {
                        "NULL".to_string()
                    } else {
                        duckdb::arrow::util::display::array_value_to_string(array, row).unwrap()
                    }
                })
                .collect::<Vec<_>>();
            lines.push(cells.join(","));
        }
    }
    lines
}

/// [`displayed_duckdb_rows`] for DataFusion batches.
fn displayed_datafusion_rows(batches: &[RecordBatch]) -> Vec<String> {
    let mut lines = Vec::new();
    for batch in batches {
        for row in 0..batch.num_rows() {
            let cells = (0..batch.num_columns())
                .map(|column| {
                    let array = batch.column(column);
                    if array.is_null(row) {
                        "NULL".to_string()
                    } else {
                        datafusion::arrow::util::display::array_value_to_string(array, row).unwrap()
                    }
                })
                .collect::<Vec<_>>();
            lines.push(cells.join(","));
        }
    }
    lines
}

/// Replay [`NESTED_EVOLUTION`] on the attached catalog `catalog` and return,
/// per snapshot from the table's creation on, what DuckDB reads there.
fn nested_evolution(conn: &duckdb::Connection, catalog: &str) -> Vec<(i64, Vec<String>)> {
    let head = || -> i64 {
        conn.query_row(
            &format!("SELECT max(snapshot_id) FROM ducklake_snapshots('{catalog}')"),
            [],
            |row| row.get(0),
        )
        .unwrap()
    };
    // The snapshot that creates the table, after however the catalog began.
    let created = head() + 1;
    for statement in NESTED_EVOLUTION {
        let sql = statement.replace("{c}", catalog);
        conn.execute_batch(&sql)
            .unwrap_or_else(|e| panic!("`{sql}`: {e}"));
    }
    let last: i64 = conn
        .query_row(
            &format!("SELECT max(snapshot_id) FROM ducklake_snapshots('{catalog}')"),
            [],
            |row| row.get(0),
        )
        .unwrap();
    (created..=last)
        .map(|snapshot| {
            (
                snapshot,
                displayed_duckdb_rows(
                    conn,
                    &format!("SELECT * FROM {catalog}.t AT (VERSION => {snapshot}) ORDER BY id"),
                ),
            )
        })
        .collect()
}

/// Require this crate to read every snapshot of `expected` as DuckDB did.
async fn assert_nested_evolution_reads(
    provider: Arc<dyn MetadataProvider>,
    expected: &[(i64, Vec<String>)],
    label: &str,
    table: &str,
) {
    for (snapshot, rows) in expected {
        let ctx = SessionContext::new();
        ctx.register_catalog(
            "lake",
            Arc::new(DuckLakeCatalog::with_snapshot(Arc::clone(&provider), *snapshot).unwrap()),
        );
        let batches = ctx
            .sql(&format!("SELECT * FROM lake.main.{table} ORDER BY id"))
            .await
            .unwrap_or_else(|e| panic!("{label} snapshot {snapshot}: {e}"))
            .collect()
            .await
            .unwrap_or_else(|e| panic!("{label} snapshot {snapshot}: {e}"));
        assert_eq!(
            &displayed_datafusion_rows(&batches),
            rows,
            "{label} snapshot {snapshot}"
        );
    }
}

/// Inlined rows written before a struct (at any depth, and inside a list or a
/// map) gained, renamed or lost a field read with the table's current shape, a
/// field they predate taking its default, at every snapshot, as DuckDB reads
/// them on DuckDB and SQLite catalogs.
#[tokio::test(flavor = "multi_thread")]
async fn inlined_nested_rows_read_across_struct_evolution() {
    let temp = TempDir::new().unwrap();
    common::ensure_ducklake_installed();
    common::ensure_extension_installed("sqlite");
    let conn = duckdb::Connection::open_in_memory().unwrap();
    conn.execute_batch("LOAD ducklake; LOAD sqlite;").unwrap();
    let duck = temp.path().join("a.ducklake");
    let lite = temp.path().join("b.db");
    // This crate writes `lite` below, so it creates it; DuckDB writes it first.
    #[cfg(feature = "write-sqlite")]
    {
        let data = temp.path().join("data").join("lite");
        std::fs::create_dir_all(&data).unwrap();
        common::crate_sqlite_catalog(&format!("sqlite:{}?mode=rwc", lite.display()), &data).await;
    }
    let mut expected = Vec::new();
    for (name, uri) in [
        ("duck", format!("ducklake:{}", escaped_path(&duck))),
        ("lite", format!("ducklake:sqlite:{}", escaped_path(&lite))),
    ] {
        conn.execute_batch(&format!(
            "ATTACH '{uri}' AS {name} (DATA_PATH '{}', DATA_INLINING_ROW_LIMIT 10)",
            escaped_path(&temp.path().join("data").join(name))
        ))
        .unwrap();
        expected.push(nested_evolution(&conn, name));
    }
    drop(conn);
    // DuckDB reads the oldest row's added struct field as its default.
    assert!(
        expected[0].last().unwrap().1[0].contains("{bb: x, c: 7}"),
        "{:?}",
        expected[0].last()
    );

    assert_nested_evolution_reads(
        Arc::new(DuckdbMetadataProvider::new(duck.to_string_lossy().to_string()).unwrap()),
        &expected[0],
        "duckdb",
        "t",
    )
    .await;
    #[cfg(feature = "metadata-sqlite")]
    assert_nested_evolution_reads(
        Arc::new(
            datafusion_ducklake::SqliteMetadataProvider::new(&format!("sqlite:{}", lite.display()))
                .await
                .unwrap(),
        ),
        &expected[1],
        "sqlite",
        "t",
    )
    .await;
    #[cfg(all(feature = "metadata-sqlite", feature = "write-sqlite"))]
    nested_evolution_mutations_match(&temp, &lite).await;
}

/// A history like [`NESTED_EVOLUTION`] for a table `u` whose evolving struct is
/// its last column, then the deletes and updates [`nested_evolution_mutations_match`]
/// runs over its inlined rows of every shape.
#[cfg(all(feature = "metadata-sqlite", feature = "write-sqlite"))]
const NESTED_MUTATION_SETUP: &[&str] = &[
    "CREATE TABLE {c}.u(id INTEGER, s STRUCT(a INTEGER, b VARCHAR, l STRUCT(x INTEGER)[]))",
    "INSERT INTO {c}.u VALUES (1, {'a': 1, 'b': 'x', 'l': [{'x': 1}]}), (2, NULL)",
    "ALTER TABLE {c}.u ADD COLUMN s.l.element.y INTEGER",
    "ALTER TABLE {c}.u ADD COLUMN s.c INTEGER DEFAULT 7",
    "INSERT INTO {c}.u VALUES (3, {'a': 3, 'b': 'z', 'l': [{'x': 3, 'y': NULL}], 'c': NULL})",
    "ALTER TABLE {c}.u RENAME COLUMN s.b TO bb",
    "INSERT INTO {c}.u VALUES (4, {'a': 4, 'bb': 'w', 'l': [], 'c': 4})",
    "ALTER TABLE {c}.u DROP COLUMN s.a",
];

#[cfg(all(feature = "metadata-sqlite", feature = "write-sqlite"))]
const NESTED_MUTATIONS: &[&str] =
    &["DELETE FROM {c}.u WHERE id = 2", "UPDATE {c}.u SET id = id + 100 WHERE id IN (1, 3)"];

/// Delete and update inlined rows of every struct shape, then flush them to
/// Parquet, once by DuckDB on a copy of the SQLite catalog `lite` and once by
/// this crate on `lite` itself: after each step this crate reads both tables of
/// `lite` as DuckDB reads its copy.
#[cfg(all(feature = "metadata-sqlite", feature = "write-sqlite"))]
async fn nested_evolution_mutations_match(temp: &TempDir, lite: &Path) {
    use datafusion_ducklake::{SqliteMetadataProvider, SqliteMetadataWriter, execute_ducklake_sql};

    let conn = duckdb::Connection::open_in_memory().unwrap();
    conn.execute_batch("LOAD ducklake; LOAD sqlite;").unwrap();
    conn.execute_batch(&format!(
        "ATTACH 'ducklake:sqlite:{}' AS ours",
        escaped_path(lite)
    ))
    .unwrap();
    for statement in NESTED_MUTATION_SETUP {
        conn.execute_batch(&statement.replace("{c}", "ours"))
            .unwrap();
    }
    conn.execute_batch("DETACH ours").unwrap();
    let copy = temp.path().join("c.db");
    std::fs::copy(lite, &copy).unwrap();
    conn.execute_batch(&format!(
        "ATTACH 'ducklake:sqlite:{}' AS official",
        escaped_path(&copy)
    ))
    .unwrap();
    let official_head = || -> i64 {
        conn.query_row(
            "SELECT max(snapshot_id) FROM ducklake_snapshots('official')",
            [],
            |row| row.get(0),
        )
        .unwrap()
    };
    let official_rows = |snapshot: i64| -> Vec<Vec<String>> {
        ["t", "u"]
            .iter()
            .map(|table| {
                displayed_duckdb_rows(
                    &conn,
                    &format!(
                        "SELECT * FROM official.{table} AT (VERSION => {snapshot}) ORDER BY id"
                    ),
                )
            })
            .collect()
    };
    let mut expected = vec![official_rows(official_head())];
    for statement in NESTED_MUTATIONS {
        conn.execute_batch(&statement.replace("{c}", "official"))
            .unwrap();
        expected.push(official_rows(official_head()));
    }
    conn.execute_batch("CALL ducklake_flush_inlined_data('official', table_name => 'u')")
        .unwrap();
    expected.push(official_rows(official_head()));
    drop(conn);

    let url = format!("sqlite:{}", lite.display());
    let provider: Arc<dyn MetadataProvider> =
        Arc::new(SqliteMetadataProvider::new(&url).await.unwrap());
    let writable = || async {
        DuckLakeCatalog::with_writer(
            Arc::clone(&provider),
            Arc::new(SqliteMetadataWriter::new(&url).await.unwrap()),
        )
        .unwrap()
    };
    let ctx = SessionContext::new();
    let handle = writable().await;
    DuckLakeCatalog::register(&ctx, "lake", writable().await);
    let mut snapshots = vec![provider.get_current_snapshot().unwrap()];
    for statement in NESTED_MUTATIONS {
        let sql = statement.replace("{c}", "lake.main");
        execute_ducklake_sql(&ctx, &handle, &sql)
            .await
            .unwrap_or_else(|e| panic!("`{sql}`: {e}"))
            .collect()
            .await
            .unwrap_or_else(|e| panic!("`{sql}`: {e}"));
        snapshots.push(provider.get_current_snapshot().unwrap());
    }
    let flushed = handle
        .flush_inlined_data(&ctx.state(), Some("main"), Some("u"))
        .await
        .unwrap();
    assert_eq!(flushed.len(), 1, "{flushed:?}");
    snapshots.push(provider.get_current_snapshot().unwrap());

    for (step, (snapshot, official)) in snapshots.iter().zip(&expected).enumerate() {
        for (table, official) in ["t", "u"].iter().zip(official) {
            assert_nested_evolution_reads(
                Arc::clone(&provider),
                &[(*snapshot, official.clone())],
                &format!("sqlite {table} after step {step}"),
                table,
            )
            .await;
        }
    }
    let after_flush = &expected.last().unwrap()[1];
    assert!(
        after_flush
            .iter()
            .any(|row| row.starts_with("101,{bb: x, l: [{x: 1, y: }], c: 7}")),
        "{after_flush:?}"
    );
}

/// [`inlined_nested_rows_read_across_struct_evolution`] on a PostgreSQL catalog.
#[cfg(feature = "metadata-postgres")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
#[tokio::test(flavor = "multi_thread")]
async fn postgres_inlined_nested_rows_read_across_struct_evolution() {
    use datafusion_ducklake::PostgresMetadataProvider;
    use testcontainers::runners::AsyncRunner;
    use testcontainers_modules::postgres::Postgres;

    let container = Postgres::default().start().await.unwrap();
    let port = container.get_host_port_ipv4(5432).await.unwrap();
    let url = format!("postgresql://postgres:postgres@127.0.0.1:{port}/postgres");
    let libpq =
        format!("host=127.0.0.1 port={port} dbname=postgres user=postgres password=postgres");
    let temp = TempDir::new().unwrap();
    common::ensure_ducklake_installed();
    common::ensure_extension_installed("postgres");
    let conn = duckdb::Connection::open_in_memory().unwrap();
    conn.execute_batch("LOAD postgres; LOAD ducklake;").unwrap();
    conn.execute_batch(&format!(
        "ATTACH 'ducklake:postgres:{libpq}' AS lake (DATA_PATH '{}', DATA_INLINING_ROW_LIMIT 10)",
        escaped_path(&temp.path().join("data"))
    ))
    .unwrap();
    let expected = nested_evolution(&conn, "lake");
    drop(conn);
    assert_nested_evolution_reads(
        Arc::new(PostgresMetadataProvider::new(&url).await.unwrap()),
        &expected,
        "postgres",
        "t",
    )
    .await;
}

/// Inlined struct rows on a MySQL catalog written before a field was added
/// (with a default) and another renamed read in the current shape. Official
/// DuckLake cannot inline a struct on MySQL, so the catalog is built directly
/// in the layout it stores inlined text in, with the schema-version ledger
/// rows each inlined table's registered version is looked up in.
#[cfg(all(feature = "metadata-mysql", feature = "write-mysql"))]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
#[tokio::test(flavor = "multi_thread")]
async fn mysql_inlined_nested_rows_read_across_struct_evolution() {
    use datafusion_ducklake::{MySqlMetadataProvider, MySqlMetadataWriter};
    use sqlx::MySqlPool;
    use testcontainers::runners::AsyncRunner;
    use testcontainers_modules::mysql::Mysql;

    let container = Mysql::default().start().await.unwrap();
    let port = container.get_host_port_ipv4(3306).await.unwrap();
    let url = format!("mysql://root@127.0.0.1:{port}/test");
    MySqlMetadataWriter::new_with_init(&url).await.unwrap();
    let pool = MySqlPool::connect(&url).await.unwrap();
    for statement in [
        "INSERT INTO ducklake_snapshot(snapshot_id, schema_version) \
         VALUES (1, 1), (2, 1), (3, 2), (4, 2)",
        "INSERT INTO ducklake_metadata(`key`, `value`, scope) VALUES ('data_path', '/tmp', NULL)",
        "INSERT INTO ducklake_schema(schema_id, schema_name, path, path_is_relative, begin_snapshot) \
         VALUES (1, 'main', '', TRUE, 1)",
        "INSERT INTO ducklake_table(table_id, schema_id, table_name, path, path_is_relative, \
         begin_snapshot) VALUES (1, 1, 't', '', TRUE, 1)",
        "INSERT INTO ducklake_column(column_id, begin_snapshot, end_snapshot, table_id, \
         column_order, column_name, column_type, initial_default, nulls_allowed, parent_column) \
         VALUES (1, 1, NULL, 1, 1, 'id', 'int32', NULL, TRUE, NULL), \
                (2, 1, NULL, 1, 2, 's', 'struct', NULL, TRUE, NULL), \
                (3, 1, NULL, 1, 3, 'a', 'int32', NULL, TRUE, 2), \
                (4, 1, 3, 1, 4, 'b', 'varchar', NULL, TRUE, 2), \
                (4, 3, NULL, 1, 4, 'bb', 'varchar', NULL, TRUE, 2), \
                (5, 3, NULL, 1, 5, 'c', 'int32', '7', FALSE, 2)",
        "CREATE TABLE IF NOT EXISTS ducklake_inlined_data_tables( \
         table_id BIGINT, table_name VARCHAR(255), schema_version BIGINT)",
        "CREATE TABLE ducklake_inlined_data_1_1(row_id BIGINT, begin_snapshot BIGINT, \
         end_snapshot BIGINT, id BIGINT, s TEXT)",
        "CREATE TABLE ducklake_inlined_data_1_2(row_id BIGINT, begin_snapshot BIGINT, \
         end_snapshot BIGINT, id BIGINT, s TEXT)",
        "INSERT INTO ducklake_inlined_data_tables VALUES \
         (1, 'ducklake_inlined_data_1_1', 1), (1, 'ducklake_inlined_data_1_2', 2)",
        "INSERT INTO ducklake_schema_versions(begin_snapshot, schema_version, table_id) \
         VALUES (1, 1, 1), (3, 2, 1)",
        "INSERT INTO ducklake_inlined_data_1_1 VALUES (0, 2, NULL, 1, '{''a'': 1, ''b'': ''x''}')",
        "INSERT INTO ducklake_inlined_data_1_2 VALUES \
         (1, 4, NULL, 2, '{''a'': 2, ''bb'': ''y'', ''c'': NULL}')",
    ] {
        sqlx::query(sqlx::AssertSqlSafe(statement))
            .execute(&pool)
            .await
            .unwrap_or_else(|e| panic!("`{statement}`: {e}"));
    }
    assert_nested_evolution_reads(
        Arc::new(MySqlMetadataProvider::new(&url).await.unwrap()),
        &[
            (2, vec!["1,{a: 1, b: x}".to_string()]),
            (
                4,
                vec!["1,{a: 1, bb: x, c: 7}".to_string(), "2,{a: 2, bb: y, c: }".to_string()],
            ),
        ],
        "mysql",
        "t",
    )
    .await;
}
