#![cfg(feature = "write-duckdb")]

use std::process::Command;
use std::sync::Arc;

use arrow::array::types::IntervalMonthDayNano;
use arrow::array::{
    Array, ArrayRef, Date32Array, Decimal128Array, FixedSizeBinaryArray, Float32Array,
    Float64Array, Int8Array, Int32Array, IntervalMonthDayNanoArray, LargeBinaryArray,
    LargeStringArray, ListArray, MapArray, StringArray, StringViewArray, StructArray,
    Time64MicrosecondArray, TimestampNanosecondArray, UInt32Array, UInt64Array,
};
use arrow::buffer::{NullBuffer, OffsetBuffer, ScalarBuffer};
use arrow::datatypes::{DataType, Field, Fields, IntervalUnit, Schema, TimeUnit};
use arrow::record_batch::RecordBatch;
use datafusion::common::ScalarValue;
use datafusion_ducklake::inlined_filter::{InlinedComparison, InlinedFilter, InlinedValue};
use datafusion_ducklake::{
    ColumnDef, DuckLakeTableWriter, DuckLakeWriteOptions, DuckdbMetadataProvider,
    DuckdbMetadataWriter, InlinedRowRef, MetadataProvider, MetadataWriter, WriteMode,
};
use object_store::local::LocalFileSystem;
use tempfile::TempDir;

#[tokio::test(flavor = "multi_thread")]
async fn duckdb_writer_persists_native_inlined_rows() {
    let temp = TempDir::new().unwrap();
    let catalog_path = temp.path().join("metadata.duckdb");
    let data_path = temp.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();
    let writer = Arc::new(
        DuckdbMetadataWriter::new_with_init(catalog_path.to_string_lossy().into_owned()).unwrap(),
    );
    writer.set_data_path(data_path.to_str().unwrap()).unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![1, 2])),
            Arc::new(StringArray::from(vec![Some("one"), None])),
        ],
    )
    .unwrap();
    let options = DuckLakeWriteOptions {
        data_inlining_row_limit: Some(2),
        ..Default::default()
    };
    let result = DuckLakeTableWriter::new(writer.clone(), Arc::new(LocalFileSystem::new()))
        .unwrap()
        .with_options(&options)
        .write_table("main", "items", &[batch])
        .await
        .unwrap();
    assert_eq!(result.files_written, 0);
    assert_eq!(result.records_written, 2);

    let connection = duckdb::Connection::open(&catalog_path).unwrap();
    let physical_name: String = connection
        .query_row(
            "SELECT table_name FROM ducklake_inlined_data_tables WHERE table_id = ?",
            duckdb::params![result.table_id],
            |row| row.get(0),
        )
        .unwrap();
    let rows = connection
        .prepare(&format!(
            "SELECT row_id, begin_snapshot, end_snapshot, id, name
             FROM {physical_name} ORDER BY row_id"
        ))
        .unwrap()
        .query_map([], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, Option<i64>>(2)?,
                row.get::<_, i32>(3)?,
                row.get::<_, Option<String>>(4)?,
            ))
        })
        .unwrap()
        .collect::<std::result::Result<Vec<_>, _>>()
        .unwrap();
    let stats: (i64, i64, i64) = connection
        .query_row(
            "SELECT record_count, next_row_id, file_size_bytes
             FROM ducklake_table_stats WHERE table_id = ?",
            duckdb::params![result.table_id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .unwrap();
    assert_eq!(
        rows,
        vec![
            (0, result.snapshot_id, None, 1, Some("one".to_string())),
            (1, result.snapshot_id, None, 2, None),
        ]
    );
    assert_eq!(stats, (2, 2, 0));
}

#[tokio::test(flavor = "multi_thread")]
async fn duckdb_writer_round_trips_supported_scalar_inlined_rows() {
    let temp = TempDir::new().unwrap();
    let catalog_path = temp.path().join("scalar-metadata.duckdb");
    let data_path = temp.path().join("scalar-data");
    std::fs::create_dir_all(&data_path).unwrap();
    let writer = Arc::new(
        DuckdbMetadataWriter::new_with_init(catalog_path.to_string_lossy().into_owned()).unwrap(),
    );
    writer.set_data_path(data_path.to_str().unwrap()).unwrap();
    let decimal = Decimal128Array::from(vec![Some(12_345)])
        .with_precision_and_scale(10, 2)
        .unwrap();
    let timestamp = TimestampNanosecondArray::from(vec![Some(1_000_002)]).with_timezone("UTC");
    let uuid = [
        0x55, 0x0e, 0x84, 0x00, 0xe2, 0x9b, 0x41, 0xd4, 0xa7, 0x16, 0x44, 0x66, 0x55, 0x44, 0x00,
        0x00,
    ];
    let schema = Arc::new(Schema::new(vec![
        Field::new("tiny", DataType::Int8, false),
        Field::new("float", DataType::Float32, false),
        Field::new("double", DataType::Float64, false),
        Field::new("decimal", DataType::Decimal128(10, 2), false),
        Field::new("date", DataType::Date32, false),
        Field::new("time", DataType::Time64(TimeUnit::Microsecond), false),
        Field::new(
            "timestamp",
            DataType::Timestamp(TimeUnit::Nanosecond, Some("UTC".into())),
            false,
        ),
        Field::new(
            "interval",
            DataType::Interval(IntervalUnit::MonthDayNano),
            false,
        ),
        Field::new("large_text", DataType::LargeUtf8, false),
        Field::new("large_binary", DataType::LargeBinary, false),
        Field::new("uuid", DataType::FixedSizeBinary(16), false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int8Array::from(vec![-7])),
            Arc::new(Float32Array::from(vec![1.25])),
            Arc::new(Float64Array::from(vec![-2.5])),
            Arc::new(decimal),
            Arc::new(Date32Array::from(vec![1])),
            Arc::new(Time64MicrosecondArray::from(vec![1_000_002])),
            Arc::new(timestamp),
            Arc::new(IntervalMonthDayNanoArray::from(vec![
                IntervalMonthDayNano::new(1, 2, 3_000),
            ])),
            Arc::new(LargeStringArray::from(vec!["large"])),
            Arc::new(LargeBinaryArray::from(vec![&[0_u8, 0xff][..]])),
            Arc::new(FixedSizeBinaryArray::try_from_iter([uuid.as_slice()].into_iter()).unwrap()),
        ],
    )
    .unwrap();
    let result = DuckLakeTableWriter::new(writer, Arc::new(LocalFileSystem::new()))
        .unwrap()
        .with_options(&DuckLakeWriteOptions {
            data_inlining_row_limit: Some(1),
            ..Default::default()
        })
        .write_table("main", "scalars", &[batch])
        .await
        .unwrap();
    assert_eq!(result.files_written, 0);
    let provider = DuckdbMetadataProvider::new(catalog_path.to_string_lossy()).unwrap();
    let snapshot_id = provider.get_current_snapshot().unwrap();
    let schema = provider
        .get_schema_by_name("main", snapshot_id)
        .unwrap()
        .unwrap();
    let table = provider
        .get_table_by_name(schema.schema_id, "scalars", snapshot_id)
        .unwrap()
        .unwrap();
    let columns = provider
        .get_table_structure(table.table_id, snapshot_id)
        .unwrap();
    let batches = provider
        .get_inlined_data(table.table_id, snapshot_id, &columns)
        .unwrap();
    assert_eq!(batches.len(), 1);
    let expected = vec![
        ScalarValue::Int8(Some(-7)),
        ScalarValue::Float32(Some(1.25)),
        ScalarValue::Float64(Some(-2.5)),
        ScalarValue::Decimal128(Some(12_345), 10, 2),
        ScalarValue::Date32(Some(1)),
        ScalarValue::Time64Microsecond(Some(1_000_002)),
        ScalarValue::TimestampNanosecond(Some(1_000_002), Some("UTC".into())),
        ScalarValue::new_interval_mdn(1, 2, 3_000),
        ScalarValue::Utf8View(Some("large".to_string())),
        ScalarValue::BinaryView(Some(vec![0, 0xff])),
        ScalarValue::FixedSizeBinary(16, Some(uuid.to_vec())),
    ];
    for (index, expected) in expected.into_iter().enumerate() {
        assert_eq!(
            ScalarValue::try_from_array(batches[0].column(index), 0).unwrap(),
            expected,
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn duckdb_writer_rejects_submicrosecond_interval_inlining() {
    let temp = TempDir::new().unwrap();
    let catalog_path = temp.path().join("interval-metadata.duckdb");
    let data_path = temp.path().join("interval-data");
    std::fs::create_dir_all(&data_path).unwrap();
    let writer = Arc::new(
        DuckdbMetadataWriter::new_with_init(catalog_path.to_string_lossy().into_owned()).unwrap(),
    );
    writer.set_data_path(data_path.to_str().unwrap()).unwrap();
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "interval",
            DataType::Interval(IntervalUnit::MonthDayNano),
            false,
        )])),
        vec![Arc::new(IntervalMonthDayNanoArray::from(vec![
            IntervalMonthDayNano::new(0, 0, 3_001),
        ]))],
    )
    .unwrap();

    let error = DuckLakeTableWriter::new(writer, Arc::new(LocalFileSystem::new()))
        .unwrap()
        .with_options(&DuckLakeWriteOptions {
            data_inlining_row_limit: Some(1),
            ..Default::default()
        })
        .write_table("main", "submicrosecond_interval", &[batch])
        .await
        .unwrap_err();
    assert!(error.to_string().contains("sub-microsecond"));
}

#[tokio::test(flavor = "multi_thread")]
async fn duckdb_writer_round_trips_uint64_boundaries_and_filters() {
    let temp = TempDir::new().unwrap();
    let catalog_path = temp.path().join("uint64-metadata.duckdb");
    let data_path = temp.path().join("uint64-data");
    std::fs::create_dir_all(&data_path).unwrap();
    let writer = Arc::new(
        DuckdbMetadataWriter::new_with_init(catalog_path.to_string_lossy().into_owned()).unwrap(),
    );
    writer.set_data_path(data_path.to_str().unwrap()).unwrap();
    let values = [0, i64::MAX as u64 + 1, u64::MAX];
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::UInt64,
            false,
        )])),
        vec![Arc::new(UInt64Array::from(values.to_vec()))],
    )
    .unwrap();
    let result = DuckLakeTableWriter::new(writer.clone(), Arc::new(LocalFileSystem::new()))
        .unwrap()
        .with_options(&DuckLakeWriteOptions {
            data_inlining_row_limit: Some(3),
            ..Default::default()
        })
        .write_table("main", "uint64_values", &[batch])
        .await
        .unwrap();
    assert_eq!(result.files_written, 0);
    writer
        .set_inlined_index_columns(result.table_id, &["value".to_string()])
        .unwrap();
    writer.ensure_inlined_indexes(result.table_id).unwrap();
    writer.ensure_inlined_indexes(result.table_id).unwrap();
    drop(writer);

    let provider = DuckdbMetadataProvider::new(catalog_path.to_string_lossy()).unwrap();
    let snapshot = provider.get_current_snapshot().unwrap();
    let columns = provider
        .get_table_structure(result.table_id, snapshot)
        .unwrap();
    let batches = provider
        .get_inlined_data(result.table_id, snapshot, &columns)
        .unwrap();
    let actual = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap();
    assert_eq!(actual.values(), &values);
    let filtered = provider
        .scan_inlined_data(
            result.table_id,
            snapshot,
            &columns,
            Some(&InlinedFilter::Comparison {
                column: "value".to_string(),
                op: InlinedComparison::GtEq,
                value: InlinedValue::U64(i64::MAX as u64 + 1),
            }),
        )
        .unwrap();
    assert_eq!(filtered.materialized_row_count, 2);
    drop(provider);
    let connection = duckdb::Connection::open(&catalog_path).unwrap();
    let physical: String = connection
        .query_row(
            "SELECT table_name FROM ducklake_inlined_data_tables WHERE table_id = ?",
            duckdb::params![result.table_id],
            |row| row.get(0),
        )
        .unwrap();
    let index_count: i64 = connection
        .query_row(
            "SELECT COUNT(*) FROM duckdb_indexes()
             WHERE table_name = ? AND index_name IN (?, ?)",
            duckdb::params![
                physical,
                format!("{physical}_row_id_idx"),
                format!("{physical}_value_idx"),
            ],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(index_count, 2);
}

#[tokio::test(flavor = "multi_thread")]
async fn duckdb_writer_round_trips_nested_inlined_rows() {
    let temp = TempDir::new().unwrap();
    let catalog_path = temp.path().join("nested-metadata.duckdb");
    let data_path = temp.path().join("nested-data");
    std::fs::create_dir_all(&data_path).unwrap();
    let writer = Arc::new(
        DuckdbMetadataWriter::new_with_init(catalog_path.to_string_lossy().into_owned()).unwrap(),
    );
    writer.set_data_path(data_path.to_str().unwrap()).unwrap();

    let item_fields = Fields::from(vec![
        Field::new("price", DataType::Decimal128(10, 2), true),
        Field::new(
            "event_time",
            DataType::Timestamp(TimeUnit::Nanosecond, Some("UTC".into())),
            true,
        ),
        Field::new("label", DataType::Utf8View, true),
        Field::new("count", DataType::UInt32, true),
        Field::new("order_id", DataType::UInt64, true),
    ]);
    let item_values = StructArray::new(
        item_fields.clone(),
        vec![
            Arc::new(
                Decimal128Array::from(vec![Some(12_345), Some(67_890), None])
                    .with_precision_and_scale(10, 2)
                    .unwrap(),
            ) as ArrayRef,
            Arc::new(
                TimestampNanosecondArray::from(vec![Some(1_000_002), Some(2_000_003), None])
                    .with_timezone("UTC"),
            ) as ArrayRef,
            Arc::new(StringViewArray::from(vec![
                Some("first"),
                Some("a,b'c"),
                None,
            ])) as ArrayRef,
            Arc::new(UInt32Array::from(vec![Some(1), Some(2), None])) as ArrayRef,
            Arc::new(UInt64Array::from(vec![Some(11), Some(22), None])) as ArrayRef,
        ],
        None,
    );
    let list_field = Arc::new(Field::new(
        "item",
        DataType::Struct(item_fields.clone()),
        true,
    ));
    let depths = ListArray::new(
        Arc::clone(&list_field),
        OffsetBuffer::new(ScalarBuffer::from(vec![0_i32, 2, 2, 2, 3])),
        Arc::new(item_values),
        Some(NullBuffer::from(vec![true, true, false, true])),
    );

    let state_fields = Fields::from(vec![
        Field::new("count", DataType::Int32, true),
        Field::new("note", DataType::Utf8View, true),
    ]);
    let state = StructArray::new(
        state_fields.clone(),
        vec![
            Arc::new(Int32Array::from(vec![Some(7), None, None, Some(8)])) as ArrayRef,
            Arc::new(StringViewArray::from(vec![
                Some("set"),
                None,
                None,
                Some("last"),
            ])) as ArrayRef,
        ],
        Some(NullBuffer::from(vec![true, true, false, true])),
    );

    let map_fields = Fields::from(vec![
        Field::new("key", DataType::Utf8View, false),
        Field::new("value", DataType::Int32, true),
    ]);
    let map_entries = StructArray::new(
        map_fields.clone(),
        vec![
            Arc::new(StringViewArray::from(vec!["a", "q'x", "tail"])) as ArrayRef,
            Arc::new(Int32Array::from(vec![Some(10), None, Some(30)])) as ArrayRef,
        ],
        None,
    );
    let map_field = Arc::new(Field::new("entries", DataType::Struct(map_fields), false));
    let attributes = MapArray::new(
        map_field,
        OffsetBuffer::new(ScalarBuffer::from(vec![0_i32, 2, 2, 2, 3])),
        map_entries,
        Some(NullBuffer::from(vec![true, true, false, true])),
        false,
    );

    let schema = Arc::new(Schema::new(vec![
        Field::new("depths", depths.data_type().clone(), true),
        Field::new("state", DataType::Struct(state_fields), true),
        Field::new("attributes", attributes.data_type().clone(), true),
    ]));
    let expected = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(depths), Arc::new(state), Arc::new(attributes)],
    )
    .unwrap();

    let result = DuckLakeTableWriter::new(writer, Arc::new(LocalFileSystem::new()))
        .unwrap()
        .with_options(&DuckLakeWriteOptions {
            data_inlining_row_limit: Some(4),
            ..Default::default()
        })
        .write_table("main", "nested", std::slice::from_ref(&expected))
        .await
        .unwrap();
    assert_eq!(result.files_written, 0);

    let provider = DuckdbMetadataProvider::new(catalog_path.to_string_lossy()).unwrap();
    let snapshot_id = provider.get_current_snapshot().unwrap();
    let catalog_schema = provider
        .get_schema_by_name("main", snapshot_id)
        .unwrap()
        .unwrap();
    let table = provider
        .get_table_by_name(catalog_schema.schema_id, "nested", snapshot_id)
        .unwrap()
        .unwrap();
    let columns = provider
        .get_table_structure(table.table_id, snapshot_id)
        .unwrap();
    let batches = provider
        .get_inlined_data(table.table_id, snapshot_id, &columns)
        .unwrap();
    assert_eq!(batches, vec![expected.clone()]);
    let inlined = provider
        .get_inlined_data_with_row_ids(table.table_id, snapshot_id, &columns)
        .unwrap();
    let deleted_row = InlinedRowRef {
        table_name: inlined[0].table_name.clone(),
        row_id: inlined[0].row_ids[0],
    };
    drop(provider);

    let writer = DuckdbMetadataWriter::new(catalog_path.to_string_lossy().into_owned()).unwrap();
    let deleted = writer
        .commit_inlined_deletes(
            table.table_id,
            "main",
            "nested",
            snapshot_id,
            &[deleted_row],
        )
        .unwrap();
    drop(writer);

    let provider = DuckdbMetadataProvider::new(catalog_path.to_string_lossy()).unwrap();
    let live_after_delete = provider
        .get_inlined_data_with_row_ids(table.table_id, deleted.snapshot_id, &columns)
        .unwrap();
    assert_eq!(
        live_after_delete
            .iter()
            .map(|data| data.batch.num_rows())
            .sum::<usize>(),
        3
    );
    assert_eq!(
        provider
            .get_inlined_data(table.table_id, snapshot_id, &columns)
            .unwrap(),
        vec![expected.clone()]
    );
    drop(provider);

    let writer =
        Arc::new(DuckdbMetadataWriter::new(catalog_path.to_string_lossy().into_owned()).unwrap());
    let table_writer = DuckLakeTableWriter::new(writer, Arc::new(LocalFileSystem::new())).unwrap();
    let flushed = table_writer
        .flush_inlined_data("main", "nested", &live_after_delete, deleted.snapshot_id)
        .await
        .unwrap()
        .unwrap();
    drop(table_writer);

    let provider = DuckdbMetadataProvider::new(catalog_path.to_string_lossy()).unwrap();
    assert!(
        provider
            .get_inlined_data_with_row_ids(table.table_id, flushed.snapshot_id, &columns)
            .unwrap()
            .is_empty()
    );
    let pinned = provider
        .get_inlined_data(table.table_id, snapshot_id, &columns)
        .unwrap();
    assert_eq!(pinned, vec![expected]);
}

#[tokio::test(flavor = "multi_thread")]
async fn duckdb_multi_table_write_commits_nested_inline_rows() {
    let temp = TempDir::new().unwrap();
    let catalog_path = temp.path().join("nested-transaction.duckdb");
    let data_path = temp.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();
    let writer = Arc::new(
        DuckdbMetadataWriter::new_with_init(catalog_path.to_string_lossy().into_owned()).unwrap(),
    );
    writer.set_data_path(data_path.to_str().unwrap()).unwrap();
    let item_fields = Fields::from(vec![Field::new("value", DataType::Int32, false)]);
    let item_field = Arc::new(Field::new(
        "item",
        DataType::Struct(item_fields.clone()),
        false,
    ));
    let schema = Arc::new(Schema::new(vec![Field::new(
        "values",
        DataType::List(Arc::clone(&item_field)),
        false,
    )]));
    let columns =
        vec![ColumnDef::from_arrow("values", schema.field(0).data_type(), false).unwrap()];
    let setup = writer
        .begin_write_transaction("main", "nested", &columns, WriteMode::Append)
        .unwrap();
    writer
        .publish_snapshot(
            setup.table_id,
            "main",
            "nested",
            setup.snapshot_id,
            WriteMode::Append,
            setup.base_snapshot_id,
            &columns,
            &setup.field_ids,
        )
        .unwrap();
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(ListArray::new(
            item_field,
            OffsetBuffer::new(ScalarBuffer::from(vec![0_i32, 2, 3])),
            Arc::new(StructArray::new(
                item_fields,
                vec![Arc::new(Int32Array::from(vec![1, 2, 3])) as ArrayRef],
                None,
            )),
            None,
        ))],
    )
    .unwrap();
    let table_writer = DuckLakeTableWriter::new(writer, Arc::new(LocalFileSystem::new())).unwrap();
    let mut transaction = table_writer.transaction();
    transaction
        .stage_write_with_options(
            "main",
            "nested",
            schema.as_ref(),
            WriteMode::Append,
            std::slice::from_ref(&batch),
            &DuckLakeWriteOptions {
                data_inlining_row_limit: Some(2),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    let committed = transaction.commit().await.unwrap();
    assert_eq!(committed.len(), 1);
    assert_eq!(committed[0].files_written, 0);
    drop(table_writer);

    let provider = DuckdbMetadataProvider::new(catalog_path.to_string_lossy()).unwrap();
    let columns = provider
        .get_table_structure(committed[0].table_id, committed[0].snapshot_id)
        .unwrap();
    assert_eq!(
        provider
            .get_inlined_data_with_row_ids(
                committed[0].table_id,
                committed[0].snapshot_id,
                &columns,
            )
            .unwrap()[0]
            .batch,
        batch,
    );
    drop(provider);

    let metadata_path = catalog_path.to_string_lossy().replace('\'', "''");
    let output = Command::new("duckdb")
        .args([
            "-csv",
            "-noheader",
            ":memory:",
            "-c",
            &format!(
                "LOAD ducklake; ATTACH 'ducklake:{metadata_path}' AS lake; \
                 SELECT COUNT(*) FROM lake.main.nested;"
            ),
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "DuckDB extension attach failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(String::from_utf8(output.stdout).unwrap(), "2\n");
}

#[tokio::test(flavor = "multi_thread")]
async fn duckdb_multi_table_write_commits_parquet_and_inline_rows() {
    let temp = TempDir::new().unwrap();
    let catalog_path = temp.path().join("multi.duckdb");
    let data_path = temp.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();
    let writer = Arc::new(
        DuckdbMetadataWriter::new_with_init(catalog_path.to_string_lossy().into_owned()).unwrap(),
    );
    writer.set_data_path(data_path.to_str().unwrap()).unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true),
    ]));
    let columns = vec![
        ColumnDef::from_arrow("id", &DataType::Int32, false).unwrap(),
        ColumnDef::from_arrow("name", &DataType::Utf8, true).unwrap(),
    ];
    for table_name in ["data", "coverage"] {
        let setup = writer
            .begin_write_transaction("main", table_name, &columns, WriteMode::Append)
            .unwrap();
        writer
            .publish_snapshot(
                setup.table_id,
                "main",
                table_name,
                setup.snapshot_id,
                WriteMode::Append,
                setup.base_snapshot_id,
                &columns,
                &setup.column_ids,
            )
            .unwrap();
    }
    let table_writer =
        DuckLakeTableWriter::new(writer.clone(), Arc::new(LocalFileSystem::new())).unwrap();
    let mut transaction = table_writer.transaction();
    transaction
        .stage_write_with_options(
            "main",
            "data",
            schema.as_ref(),
            WriteMode::Append,
            &[RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int32Array::from(vec![1, 2, 3])),
                    Arc::new(StringArray::from(vec!["one", "two", "three"])),
                ],
            )
            .unwrap()],
            &DuckLakeWriteOptions {
                data_inlining_row_limit: Some(0),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    transaction
        .stage_write_with_options(
            "main",
            "coverage",
            schema.as_ref(),
            WriteMode::Append,
            &[RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(Int32Array::from(vec![1])), Arc::new(StringArray::from(vec!["one"]))],
            )
            .unwrap()],
            &DuckLakeWriteOptions {
                data_inlining_row_limit: Some(1),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let committed = transaction.commit().await.unwrap();

    assert_eq!(committed.len(), 2);
    assert_eq!(committed[0].snapshot_id, committed[1].snapshot_id);
    assert_eq!(committed[0].files_written, 1);
    assert_eq!(committed[1].files_written, 0);
    let connection = duckdb::Connection::open(&catalog_path).unwrap();
    let file_snapshot: i64 = connection
        .query_row(
            "SELECT begin_snapshot FROM ducklake_data_file WHERE table_id = ?",
            duckdb::params![committed[0].table_id],
            |row| row.get(0),
        )
        .unwrap();
    let inline_table: String = connection
        .query_row(
            "SELECT table_name FROM ducklake_inlined_data_tables WHERE table_id = ?",
            duckdb::params![committed[1].table_id],
            |row| row.get(0),
        )
        .unwrap();
    let inline_snapshot: i64 = connection
        .query_row(
            &format!("SELECT begin_snapshot FROM \"{inline_table}\""),
            [],
            |row| row.get(0),
        )
        .unwrap();
    let changes: String = connection
        .query_row(
            "SELECT changes_made FROM ducklake_snapshot_changes WHERE snapshot_id = ?",
            duckdb::params![committed[0].snapshot_id],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(file_snapshot, committed[0].snapshot_id);
    assert_eq!(inline_snapshot, committed[0].snapshot_id);
    assert_eq!(
        changes,
        format!(
            "inserted_into_table:{},inserted_into_table:{}",
            committed[0].table_id, committed[1].table_id
        )
    );
}
