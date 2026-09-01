#[cfg(any(feature = "write-postgres", feature = "write-mysql"))]
use std::sync::Arc;

#[cfg(feature = "write-postgres")]
use arrow::array::types::IntervalMonthDayNano;
#[cfg(any(feature = "write-postgres", feature = "write-mysql"))]
use arrow::array::{
    Array, Date32Array, RecordBatch, Time64MicrosecondArray, TimestampMicrosecondArray,
    TimestampNanosecondArray, UInt64Array,
};
#[cfg(feature = "write-mysql")]
use arrow::array::{Decimal128Array, Float32Array, Float64Array};
#[cfg(feature = "write-postgres")]
use arrow::array::{FixedSizeBinaryArray, IntervalMonthDayNanoArray};
#[cfg(feature = "write-postgres")]
use arrow::datatypes::IntervalUnit;
#[cfg(any(feature = "write-postgres", feature = "write-mysql"))]
use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
#[cfg(any(feature = "write-postgres", feature = "write-mysql"))]
use datafusion_ducklake::inlined_filter::{InlinedComparison, InlinedFilter, InlinedValue};
#[cfg(feature = "write-postgres")]
use datafusion_ducklake::{ColumnDef, SnapshotCommitMetadata, WriteMode};
#[cfg(any(feature = "write-postgres", feature = "write-mysql"))]
use datafusion_ducklake::{
    DuckLakeTableWriter, DuckLakeWriteOptions, MetadataProvider, MetadataWriter,
};
#[cfg(any(feature = "write-postgres", feature = "write-mysql"))]
use object_store::memory::InMemory;
#[cfg(any(feature = "write-postgres", feature = "write-mysql"))]
#[cfg(feature = "write-postgres")]
use sqlx::AssertSqlSafe;
#[cfg(any(feature = "write-postgres", feature = "write-mysql"))]
use tempfile::TempDir;

#[cfg(feature = "write-postgres")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
#[tokio::test(flavor = "multi_thread")]
async fn postgres_native_inlined_types_indexes_and_legacy_migration() {
    use datafusion_ducklake::{
        MulticatalogManager, MulticatalogProvider, PostgresMetadataWriter,
        initialize_multicatalog_schema,
    };
    use sqlx::postgres::PgPoolOptions;
    use testcontainers::runners::AsyncRunner;
    use testcontainers_modules::postgres::Postgres;

    let container = Postgres::default().start().await.unwrap();
    let port = container.get_host_port_ipv4(5432).await.unwrap();
    let url = format!("postgresql://postgres:postgres@127.0.0.1:{port}/postgres");
    let pool = PgPoolOptions::new()
        .max_connections(5)
        .connect(&url)
        .await
        .unwrap();
    initialize_multicatalog_schema(&pool).await.unwrap();
    let catalog_id = MulticatalogManager::new(pool.clone())
        .create_catalog("native_types")
        .await
        .unwrap();
    let temp = TempDir::new().unwrap();
    let writer = Arc::new(
        PostgresMetadataWriter::with_pool(pool.clone(), catalog_id)
            .await
            .unwrap(),
    );
    writer.set_data_path(temp.path().to_str().unwrap()).unwrap();

    let u64_values = [0, i64::MAX as u64 + 1, u64::MAX];
    let u64_batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::UInt64,
            false,
        )])),
        vec![Arc::new(UInt64Array::from(u64_values.to_vec()))],
    )
    .unwrap();
    let u64_result = DuckLakeTableWriter::new(writer.clone(), Arc::new(InMemory::new()))
        .unwrap()
        .with_options(&DuckLakeWriteOptions::default().with_data_inlining_row_limit(3))
        .write_table("main", "u64_values", &[u64_batch])
        .await
        .unwrap();
    assert_eq!(u64_result.files_written, 0);

    let native_table: String = sqlx::query_scalar(
        "SELECT table_name FROM ducklake_inlined_data_tables WHERE table_id = $1",
    )
    .bind(u64_result.table_id)
    .fetch_one(&pool)
    .await
    .unwrap();
    let native_type: String = sqlx::query_scalar(
        "SELECT data_type FROM information_schema.columns \
         WHERE table_name = $1 AND column_name = 'value'",
    )
    .bind(&native_table)
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(native_type, "numeric");

    let legacy_table = format!("ducklake_inlined_data_{}_legacy", u64_result.table_id);
    sqlx::query(AssertSqlSafe(format!(
        "CREATE TABLE \"{legacy_table}\"(\
             row_id BIGINT NOT NULL,\
             begin_snapshot BIGINT NOT NULL,\
             end_snapshot BIGINT,\
             value VARCHAR\
         )"
    )))
    .execute(&pool)
    .await
    .unwrap();
    sqlx::query(AssertSqlSafe(format!(
        "INSERT INTO \"{legacy_table}\" VALUES (100, $1, NULL, '18446744073709551615')"
    )))
    .bind(u64_result.snapshot_id)
    .execute(&pool)
    .await
    .unwrap();
    sqlx::query(
        "INSERT INTO ducklake_inlined_data_tables(table_id, table_name, schema_version) \
         VALUES ($1, $2, 0)",
    )
    .bind(u64_result.table_id)
    .bind(&legacy_table)
    .execute(&pool)
    .await
    .unwrap();

    writer
        .set_inlined_index_columns(u64_result.table_id, &["value".to_string()])
        .unwrap();
    writer.ensure_inlined_indexes(u64_result.table_id).unwrap();
    writer.ensure_inlined_indexes(u64_result.table_id).unwrap();

    let legacy_type: String = sqlx::query_scalar(
        "SELECT data_type FROM information_schema.columns \
         WHERE table_name = $1 AND column_name = 'value'",
    )
    .bind(&legacy_table)
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(legacy_type, "numeric");
    let index_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM pg_indexes \
         WHERE tablename IN ($1, $2) AND indexname IN ($3, $4, $5, $6)",
    )
    .bind(&native_table)
    .bind(&legacy_table)
    .bind(format!("{native_table}_row_id_idx"))
    .bind(format!("{native_table}_value_idx"))
    .bind(format!("{legacy_table}_row_id_idx"))
    .bind(format!("{legacy_table}_value_idx"))
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(index_count, 4);

    let provider = MulticatalogProvider::with_pool_and_id(pool.clone(), catalog_id)
        .await
        .unwrap();
    let columns = provider
        .get_table_structure(u64_result.table_id, u64_result.snapshot_id)
        .unwrap();
    let batches = provider
        .get_inlined_data(u64_result.table_id, u64_result.snapshot_id, &columns)
        .unwrap();
    let mut actual = batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(0)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap()
                .values()
                .iter()
                .copied()
        })
        .collect::<Vec<_>>();
    actual.sort_unstable();
    assert_eq!(actual, vec![0, i64::MAX as u64 + 1, u64::MAX, u64::MAX]);
    let filtered = provider
        .scan_inlined_data(
            u64_result.table_id,
            u64_result.snapshot_id,
            &columns,
            Some(&InlinedFilter::Comparison {
                column: "value".to_string(),
                op: InlinedComparison::GtEq,
                value: InlinedValue::U64(i64::MAX as u64 + 1),
            }),
        )
        .unwrap();
    assert_eq!(filtered.materialized_row_count, 3);

    let mut connection = pool.acquire().await.unwrap();
    sqlx::query("SET enable_seqscan = off")
        .execute(&mut *connection)
        .await
        .unwrap();
    let plan: Vec<String> = sqlx::query_scalar(AssertSqlSafe(format!(
        "EXPLAIN SELECT row_id FROM \"{native_table}\" \
         WHERE value >= CAST($1 AS NUMERIC(20,0))"
    )))
    .bind((i64::MAX as u64 + 1).to_string())
    .fetch_all(&mut *connection)
    .await
    .unwrap();
    assert!(
        plan.iter()
            .any(|line| line.contains(&format!("{native_table}_value_idx")))
    );

    let raw = [0x55_u8; 16];
    let raw_columns = vec![ColumnDef::new("raw", "blob", false).unwrap()];
    let raw_setup = writer
        .begin_write_transaction("main", "raw_values", &raw_columns, WriteMode::Append)
        .unwrap();
    let raw_batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "raw",
            DataType::FixedSizeBinary(16),
            false,
        )])),
        vec![Arc::new(FixedSizeBinaryArray::try_from_iter([raw.as_slice()].into_iter()).unwrap())],
    )
    .unwrap();
    writer
        .register_inlined_data(
            raw_setup.table_id,
            "main",
            "raw_values",
            raw_setup.snapshot_id,
            &[raw_batch],
            WriteMode::Append,
            raw_setup.base_snapshot_id,
            &raw_columns,
            &raw_setup.column_ids,
            &SnapshotCommitMetadata::new(),
            None,
        )
        .unwrap();
    let raw_table: String = sqlx::query_scalar(
        "SELECT table_name FROM ducklake_inlined_data_tables WHERE table_id = $1",
    )
    .bind(raw_setup.table_id)
    .fetch_one(&pool)
    .await
    .unwrap();
    let raw_type: String = sqlx::query_scalar(
        "SELECT data_type FROM information_schema.columns \
         WHERE table_name = $1 AND column_name = 'raw'",
    )
    .bind(&raw_table)
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(raw_type, "bytea");
    let stored_raw: Vec<u8> =
        sqlx::query_scalar(AssertSqlSafe(format!("SELECT raw FROM \"{raw_table}\"")))
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(stored_raw, raw);

    let typed_batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("event_date", DataType::Date32, false),
            Field::new("event_time", DataType::Time64(TimeUnit::Microsecond), false),
            Field::new(
                "event_us",
                DataType::Timestamp(TimeUnit::Microsecond, None),
                false,
            ),
            Field::new(
                "event_ns",
                DataType::Timestamp(TimeUnit::Nanosecond, None),
                false,
            ),
        ])),
        vec![
            Arc::new(Date32Array::from(vec![1])),
            Arc::new(Time64MicrosecondArray::from(vec![1_000_002])),
            Arc::new(TimestampMicrosecondArray::from(vec![1_000_002])),
            Arc::new(TimestampNanosecondArray::from(vec![1_000_002_003])),
        ],
    )
    .unwrap();
    let typed_result = DuckLakeTableWriter::new(writer.clone(), Arc::new(InMemory::new()))
        .unwrap()
        .with_options(&DuckLakeWriteOptions::default().with_data_inlining_row_limit(1))
        .write_table("main", "typed_values", &[typed_batch])
        .await
        .unwrap();
    let typed_table: String = sqlx::query_scalar(
        "SELECT table_name FROM ducklake_inlined_data_tables WHERE table_id = $1",
    )
    .bind(typed_result.table_id)
    .fetch_one(&pool)
    .await
    .unwrap();
    let physical_types: Vec<(String, String)> = sqlx::query_as(
        "SELECT column_name, data_type FROM information_schema.columns \
         WHERE table_name = $1 AND column_name IN \
         ('event_date', 'event_time', 'event_us', 'event_ns') \
         ORDER BY column_name",
    )
    .bind(&typed_table)
    .fetch_all(&pool)
    .await
    .unwrap();
    assert_eq!(
        physical_types,
        vec![
            ("event_date".to_string(), "date".to_string()),
            ("event_ns".to_string(), "bigint".to_string()),
            (
                "event_time".to_string(),
                "time without time zone".to_string(),
            ),
            (
                "event_us".to_string(),
                "timestamp without time zone".to_string(),
            ),
        ]
    );
    let columns = provider
        .get_table_structure(typed_result.table_id, typed_result.snapshot_id)
        .unwrap();
    let batches = provider
        .get_inlined_data(typed_result.table_id, typed_result.snapshot_id, &columns)
        .unwrap();
    assert_eq!(
        batches[0]
            .column(3)
            .as_any()
            .downcast_ref::<TimestampNanosecondArray>()
            .unwrap()
            .value(0),
        1_000_002_003
    );

    let interval_batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Interval(IntervalUnit::MonthDayNano),
            false,
        )])),
        vec![Arc::new(IntervalMonthDayNanoArray::from(vec![
            IntervalMonthDayNano::new(0, 0, 3_001),
        ]))],
    )
    .unwrap();
    let error = DuckLakeTableWriter::new(writer, Arc::new(InMemory::new()))
        .unwrap()
        .with_options(&DuckLakeWriteOptions::default().with_data_inlining_row_limit(1))
        .write_table("main", "submicrosecond_interval", &[interval_batch])
        .await
        .unwrap_err();
    assert!(error.to_string().contains("sub-microsecond"));
}

#[cfg(feature = "write-mysql")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
#[tokio::test(flavor = "multi_thread")]
async fn mysql_native_inlined_types_and_indexes_round_trip() {
    use datafusion_ducklake::{MySqlMetadataProvider, MySqlMetadataWriter};
    use testcontainers::runners::AsyncRunner;
    use testcontainers_modules::mysql::Mysql;

    let container = Mysql::default().start().await.unwrap();
    let port = container.get_host_port_ipv4(3306).await.unwrap();
    let url = format!("mysql://root@127.0.0.1:{port}/test");
    let pool = sqlx::MySqlPool::connect(&url).await.unwrap();
    let writer = Arc::new(MySqlMetadataWriter::new_with_init(&url).await.unwrap());
    let temp = TempDir::new().unwrap();
    writer.set_data_path(temp.path().to_str().unwrap()).unwrap();

    let u64_values = [0, i64::MAX as u64 + 1, u64::MAX];
    let decimal = Decimal128Array::from(vec![123_i128, -456, 789])
        .with_precision_and_scale(20, 2)
        .unwrap();
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("u64_value", DataType::UInt64, false),
            Field::new("float_value", DataType::Float32, false),
            Field::new("double_value", DataType::Float64, false),
            Field::new("decimal_value", DataType::Decimal128(20, 2), false),
            Field::new("date_value", DataType::Date32, false),
            Field::new("time_value", DataType::Time64(TimeUnit::Microsecond), false),
            Field::new(
                "timestamp_us",
                DataType::Timestamp(TimeUnit::Microsecond, None),
                false,
            ),
            Field::new(
                "timestamp_ns",
                DataType::Timestamp(TimeUnit::Nanosecond, None),
                false,
            ),
        ])),
        vec![
            Arc::new(UInt64Array::from(u64_values.to_vec())),
            Arc::new(Float32Array::from(vec![1.5, -2.25, 3.75])),
            Arc::new(Float64Array::from(vec![1.25, -2.5, 3.125])),
            Arc::new(decimal),
            Arc::new(Date32Array::from(vec![1, 2, 3])),
            Arc::new(Time64MicrosecondArray::from(vec![
                1_000_002, 2_000_003, 3_000_004,
            ])),
            // Epoch zero and a post-2038 value prove the DATETIME(6) range;
            // both are outside MySQL TIMESTAMP's domain.
            Arc::new(TimestampMicrosecondArray::from(vec![
                0,
                2_500_000_000_000_000,
                3_000_004,
            ])),
            Arc::new(TimestampNanosecondArray::from(vec![
                1_000_002_003,
                2_000_003_004,
                3_000_004_005,
            ])),
        ],
    )
    .unwrap();
    let result = DuckLakeTableWriter::new(writer.clone(), Arc::new(InMemory::new()))
        .unwrap()
        .with_options(&DuckLakeWriteOptions::default().with_data_inlining_row_limit(3))
        .write_table("main", "native_values", &[batch])
        .await
        .unwrap();
    assert_eq!(result.files_written, 0);

    writer
        .set_inlined_index_columns(
            result.table_id,
            &["u64_value".to_string(), "decimal_value".to_string()],
        )
        .unwrap();
    writer.ensure_inlined_indexes(result.table_id).unwrap();
    writer.ensure_inlined_indexes(result.table_id).unwrap();

    let physical_table: String = sqlx::query_scalar(
        "SELECT table_name FROM ducklake_inlined_data_tables WHERE table_id = ?",
    )
    .bind(result.table_id)
    .fetch_one(&pool)
    .await
    .unwrap();
    let physical_types: Vec<(String, String)> = sqlx::query_as(
        "SELECT column_name, column_type FROM information_schema.columns \
         WHERE table_schema = DATABASE() AND table_name = ? AND column_name IN \
         ('u64_value', 'float_value', 'double_value', 'decimal_value', \
          'date_value', 'time_value', 'timestamp_us', 'timestamp_ns') \
         ORDER BY column_name",
    )
    .bind(&physical_table)
    .fetch_all(&pool)
    .await
    .unwrap();
    assert_eq!(
        physical_types,
        vec![
            ("date_value".to_string(), "date".to_string()),
            ("decimal_value".to_string(), "decimal(20,2)".to_string()),
            ("double_value".to_string(), "double".to_string()),
            ("float_value".to_string(), "float".to_string()),
            ("time_value".to_string(), "time(6)".to_string()),
            ("timestamp_ns".to_string(), "bigint".to_string()),
            ("timestamp_us".to_string(), "datetime(6)".to_string()),
            ("u64_value".to_string(), "bigint unsigned".to_string()),
        ]
    );
    let index_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(DISTINCT index_name) FROM information_schema.statistics \
         WHERE table_schema = DATABASE() AND table_name = ? \
         AND index_name IN (?, ?, ?)",
    )
    .bind(&physical_table)
    .bind(format!("{physical_table}_row_id_idx"))
    .bind(format!("{physical_table}_u64_value_idx"))
    .bind(format!("{physical_table}_decimal_value_idx"))
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(index_count, 3);

    let provider = MySqlMetadataProvider::from_pool(pool);
    let columns = provider
        .get_table_structure(result.table_id, result.snapshot_id)
        .unwrap();
    let batches = provider
        .get_inlined_data(result.table_id, result.snapshot_id, &columns)
        .unwrap();
    assert_eq!(
        batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap()
            .values(),
        &u64_values
    );
    assert_eq!(
        batches[0]
            .column(6)
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .unwrap()
            .values(),
        &[0, 2_500_000_000_000_000, 3_000_004]
    );
    assert_eq!(
        batches[0]
            .column(7)
            .as_any()
            .downcast_ref::<TimestampNanosecondArray>()
            .unwrap()
            .values(),
        &[1_000_002_003, 2_000_003_004, 3_000_004_005]
    );
    let filtered = provider
        .scan_inlined_data(
            result.table_id,
            result.snapshot_id,
            &columns,
            Some(&InlinedFilter::Comparison {
                column: "u64_value".to_string(),
                op: InlinedComparison::GtEq,
                value: InlinedValue::U64(i64::MAX as u64 + 1),
            }),
        )
        .unwrap();
    assert_eq!(filtered.materialized_row_count, 2);
}
