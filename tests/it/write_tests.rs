//! Integration tests for write support.
//!
//! These tests verify that data written using DuckLakeTableWriter can be read back
//! via the existing DuckLakeCatalog read path.

#![cfg(all(feature = "write-sqlite", feature = "metadata-sqlite"))]

use std::sync::Arc;

use arrow::array::{
    Array, BinaryViewArray, BooleanArray, Date32Array, Decimal128Array, Float32Array, Float64Array,
    Int32Array, Int64Array, ListArray, ListBuilder, MapArray, StringArray, StringBuilder,
    StringViewArray, StructArray, TimestampMicrosecondArray, TimestampNanosecondArray, UInt32Array,
    UInt64Array,
};
use arrow::buffer::{OffsetBuffer, ScalarBuffer};
use arrow::datatypes::{DataType, Field, Float32Type, Schema, TimeUnit};
use arrow::record_batch::RecordBatch;
use datafusion::prelude::*;
use object_store::local::LocalFileSystem;
use sqlx::Row;
use tempfile::TempDir;

use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeTableWriter, MetadataWriter, SqliteMetadataProvider,
    SqliteMetadataWriter, WriteMode, register_ducklake_functions,
};

/// Create a local filesystem object store.
fn create_object_store() -> Arc<dyn object_store::ObjectStore> {
    Arc::new(LocalFileSystem::new())
}

/// Create a test environment with a writer and data directory.
async fn create_test_env() -> (SqliteMetadataWriter, TempDir) {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("test.db");
    let data_path = temp_dir.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();

    let conn_str = format!("sqlite:{}?mode=rwc", db_path.display());
    let writer = SqliteMetadataWriter::new_with_init(&conn_str)
        .await
        .unwrap();
    writer.set_data_path(data_path.to_str().unwrap()).unwrap();

    (writer, temp_dir)
}

/// Create a `SessionContext` with a `DuckLakeCatalog`.
async fn create_read_context(temp_dir: &TempDir) -> SessionContext {
    let db_path = temp_dir.path().join("test.db");
    let conn_str = format!("sqlite:{}", db_path.display());

    let provider = SqliteMetadataProvider::new(&conn_str).await.unwrap();
    let catalog = DuckLakeCatalog::new(provider).unwrap();

    let ctx = SessionContext::new();
    ctx.register_catalog("test", Arc::new(catalog));
    ctx
}

fn parquet_schema_fields(path: &std::path::Path) -> Vec<(String, Option<i32>)> {
    fn collect(field: &parquet::schema::types::Type, result: &mut Vec<(String, Option<i32>)>) {
        let info = field.get_basic_info();
        result.push((field.name().to_string(), info.has_id().then(|| info.id())));
        if field.is_group() {
            for child in field.get_fields() {
                collect(child, result);
            }
        }
    }

    let builder =
        datafusion::parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(
            std::fs::File::open(path).unwrap(),
        )
        .unwrap();
    let root = builder
        .metadata()
        .file_metadata()
        .schema_descr()
        .root_schema();
    let mut result = Vec::new();
    for field in root.get_fields() {
        collect(field, &mut result);
    }
    result
}

fn assert_duckdb_extension_reads_depths(catalog_path: &std::path::Path) {
    let conformance_path = catalog_path.with_file_name("duckdb-conformance.db");
    std::fs::copy(catalog_path, &conformance_path).unwrap();

    // This fork's catalog schema predates fields the official DuckLake extension requires. Add
    // those compatibility fields to a disposable catalog copy before testing the
    // nested data layout; none of these statements alter ducklake_column, the
    // source catalog, or the Parquet file.
    let setup = format!(
        "INSTALL sqlite; LOAD sqlite; \
         ATTACH '{}' AS raw (TYPE sqlite); \
         ALTER TABLE raw.ducklake_snapshot ADD COLUMN next_catalog_id BIGINT DEFAULT 0; \
         ALTER TABLE raw.ducklake_snapshot ADD COLUMN next_file_id BIGINT DEFAULT 0; \
         UPDATE raw.ducklake_snapshot SET next_catalog_id = 0, next_file_id = 0; \
         ALTER TABLE raw.ducklake_schema ADD COLUMN schema_uuid VARCHAR; \
         UPDATE raw.ducklake_schema SET \
             schema_uuid = '00000000-0000-0000-0000-000000000001', path = path || '/'; \
         ALTER TABLE raw.ducklake_table ADD COLUMN table_uuid VARCHAR; \
         UPDATE raw.ducklake_table SET \
             table_uuid = '00000000-0000-0000-0000-000000000002', path = path || '/'; \
         CREATE TABLE raw.ducklake_tag( \
             object_id BIGINT, begin_snapshot BIGINT, end_snapshot BIGINT, key VARCHAR, value VARCHAR \
         ); \
         CREATE TABLE raw.ducklake_inlined_data_tables( \
             table_id BIGINT, table_name VARCHAR, schema_version BIGINT \
         ); \
         CREATE TABLE raw.ducklake_column_tag( \
             table_id BIGINT, column_id BIGINT, begin_snapshot BIGINT, end_snapshot BIGINT, \
             key VARCHAR, value VARCHAR \
         ); \
         CREATE TABLE raw.ducklake_column_mapping( \
             mapping_id BIGINT, table_id BIGINT, type VARCHAR \
         ); \
         CREATE TABLE raw.ducklake_name_mapping( \
             mapping_id BIGINT, column_id BIGINT, source_name VARCHAR, \
             target_field_id BIGINT, parent_column BIGINT, is_partition BOOLEAN \
         ); \
             CREATE TABLE raw.ducklake_macro( \
             schema_id BIGINT, macro_id BIGINT, macro_name VARCHAR, \
             begin_snapshot BIGINT, end_snapshot BIGINT \
         ); \
         CREATE TABLE raw.ducklake_macro_impl( \
             macro_id BIGINT, impl_id BIGINT, dialect VARCHAR, sql VARCHAR, type VARCHAR \
         ); \
         CREATE TABLE raw.ducklake_macro_parameters( \
             macro_id BIGINT, impl_id BIGINT, column_id BIGINT, parameter_name VARCHAR, \
             parameter_type VARCHAR, default_value VARCHAR, default_value_type VARCHAR \
         ); \
         ALTER TABLE raw.ducklake_data_file ADD COLUMN file_order BIGINT DEFAULT 0; \
         ALTER TABLE raw.ducklake_data_file ADD COLUMN file_format VARCHAR DEFAULT 'parquet'; \
         ALTER TABLE raw.ducklake_data_file ADD COLUMN partial_file_info VARCHAR; \
         UPDATE raw.ducklake_data_file SET footer_size = NULL; \
         ALTER TABLE raw.ducklake_delete_file ADD COLUMN format VARCHAR DEFAULT 'parquet'; \
         DETACH raw; \
         INSTALL ducklake; LOAD ducklake; \
         ATTACH 'ducklake:{}' AS official (META_TYPE 'sqlite');",
        conformance_path.display(),
        conformance_path.display()
    );
    let connection = duckdb::Connection::open_in_memory().unwrap();
    connection.execute_batch(&setup).unwrap();
    let mut statement = connection
        .prepare(
            "SELECT id, CAST(bids[1].price AS VARCHAR), len(bids)
             FROM official.main.depths ORDER BY id",
        )
        .unwrap();
    let rows = statement
        .query_map([], |row| {
            Ok((
                row.get::<_, i32>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, i64>(2)?,
            ))
        })
        .unwrap()
        .collect::<std::result::Result<Vec<_>, _>>()
        .unwrap();

    assert_eq!(
        rows,
        vec![(1, "101.0000000000000000".to_string(), 2), (2, "99.0000000000000000".to_string(), 1),],
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_write_and_read_basic_types() {
    let (writer, temp_dir) = create_test_env().await;
    let object_store = create_object_store();

    // Create test data with various types
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true),
        Field::new("age", DataType::Int64, true),
        Field::new("score", DataType::Float64, true),
        Field::new("active", DataType::Boolean, true),
    ]));

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(StringArray::from(vec![Some("Alice"), Some("Bob"), None])),
            Arc::new(Int64Array::from(vec![Some(25), Some(30), Some(35)])),
            Arc::new(Float64Array::from(vec![Some(95.5), None, Some(88.0)])),
            Arc::new(BooleanArray::from(vec![
                Some(true),
                Some(false),
                Some(true),
            ])),
        ],
    )
    .unwrap();

    // Write data
    let table_writer = DuckLakeTableWriter::new(Arc::new(writer), object_store).unwrap();
    let result = table_writer
        .write_table("main", "users", &[batch])
        .await
        .unwrap();

    assert_eq!(result.records_written, 3);
    assert_eq!(result.files_written, 1);
    assert!(result.snapshot_id > 0);
    assert!(result.table_id > 0);
    assert!(result.schema_id > 0);

    // Read back via DuckLakeCatalog
    let ctx = create_read_context(&temp_dir).await;
    let df = ctx
        .sql("SELECT * FROM test.main.users ORDER BY id")
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();

    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 3);
    assert_eq!(batches[0].num_columns(), 5);

    // Verify data
    let ids = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    assert_eq!(ids.values(), &[1, 2, 3]);

    // DuckLake string columns scan as Utf8View; cast to Utf8 to assert via StringArray.
    let names_arr = arrow::compute::cast(batches[0].column(1), &DataType::Utf8).unwrap();
    let names = names_arr.as_any().downcast_ref::<StringArray>().unwrap();
    assert_eq!(names.value(0), "Alice");
    assert_eq!(names.value(1), "Bob");
    assert!(names.is_null(2));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_write_read_view_columns_roundtrip() {
    // A string column scans as Utf8View and a blob column as BinaryView, matching
    // DataFusion's schema_force_view_types default. This drives the full
    // write -> scan pipeline with view arrays as input (the layout a write-back
    // produces once the provider serves view types), and asserts the scan returns
    // view arrays end to end, not i32-offset Utf8/Binary.
    let (writer, temp_dir) = create_test_env().await;
    let object_store = create_object_store();

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8View, true),
        Field::new("data", DataType::BinaryView, true),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(StringViewArray::from(vec![
                Some("Alice"),
                Some("Bob"),
                None,
            ])),
            Arc::new(BinaryViewArray::from(vec![
                Some(b"xx".as_ref()),
                Some(b"yyy".as_ref()),
                None,
            ])),
        ],
    )
    .unwrap();

    let table_writer = DuckLakeTableWriter::new(Arc::new(writer), object_store).unwrap();
    table_writer
        .write_table("main", "views", &[batch])
        .await
        .unwrap();

    let ctx = create_read_context(&temp_dir).await;
    let batches = ctx
        .sql("SELECT id, name, data FROM test.main.views ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(batches.len(), 1);
    let batch = &batches[0];

    // The scan must expose the view layouts end to end.
    assert_eq!(batch.schema().field(1).data_type(), &DataType::Utf8View);
    assert_eq!(batch.schema().field(2).data_type(), &DataType::BinaryView);

    let names = batch
        .column(1)
        .as_any()
        .downcast_ref::<StringViewArray>()
        .expect("name should scan as StringViewArray");
    assert_eq!(names.value(0), "Alice");
    assert_eq!(names.value(1), "Bob");
    assert!(names.is_null(2));

    let data = batch
        .column(2)
        .as_any()
        .downcast_ref::<BinaryViewArray>()
        .expect("data should scan as BinaryViewArray");
    assert_eq!(data.value(0), b"xx");
    assert_eq!(data.value(1), b"yyy");
    assert!(data.is_null(2));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_write_read_list_of_string_roundtrip() {
    // A list<varchar> column scans with a Utf8View element. The file is written
    // with a Utf8 list child while the catalog type resolves to List(Utf8View),
    // so the read path recasts the list element (List(Utf8) -> List(Utf8View)) in
    // ColumnRenameExec. Exercises that element recast end to end.
    let (writer, temp_dir) = create_test_env().await;
    let object_store = create_object_store();

    let mut tags_builder = ListBuilder::new(StringBuilder::new());
    tags_builder.values().append_value("a");
    tags_builder.values().append_value("b");
    tags_builder.append(true);
    tags_builder.values().append_value("c");
    tags_builder.append(true);
    let tags = tags_builder.finish();

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("tags", tags.data_type().clone(), true),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(Int32Array::from(vec![1, 2])), Arc::new(tags)],
    )
    .unwrap();

    let table_writer = DuckLakeTableWriter::new(Arc::new(writer), object_store).unwrap();
    table_writer
        .write_table("main", "taglists", &[batch])
        .await
        .unwrap();

    let ctx = create_read_context(&temp_dir).await;
    let batches = ctx
        .sql("SELECT id, tags FROM test.main.taglists ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(batches.len(), 1);
    let batch = &batches[0];

    // The list element scans as Utf8View.
    match batch.schema().field(1).data_type() {
        DataType::List(field) => {
            assert_eq!(
                field.data_type(),
                &DataType::Utf8View,
                "list element should scan as Utf8View"
            );
        },
        other => panic!("expected List, got {other:?}"),
    }

    // Values survive the List(Utf8) -> List(Utf8View) element recast. Read each
    // row's element slice via a Utf8 cast so the assertion is layout-agnostic.
    let tags = batch
        .column(1)
        .as_any()
        .downcast_ref::<ListArray>()
        .expect("tags should be a ListArray");
    let row0 = arrow::compute::cast(&tags.value(0), &DataType::Utf8).unwrap();
    let row0 = row0.as_any().downcast_ref::<StringArray>().unwrap();
    assert_eq!(row0.value(0), "a");
    assert_eq!(row0.value(1), "b");
    let row1 = arrow::compute::cast(&tags.value(1), &DataType::Utf8).unwrap();
    let row1 = row1.as_any().downcast_ref::<StringArray>().unwrap();
    assert_eq!(row1.value(0), "c");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_write_and_read_list_column_roundtrip() {
    // Nested write->read coverage: a `List` column must round-trip its VALUES, not
    // get null-filled by the field-id read mapping. The field-id sits on the
    // top-level node; the List leaf carries none, so a leaf-only field-id lookup
    // would treat the column as absent. Isolates WRITE (raw parquet) vs READ.
    let (writer, temp_dir) = create_test_env().await;
    let object_store = create_object_store();

    let v = arrow::array::ListArray::from_iter_primitive::<Float32Type, _, _>(vec![
        Some(vec![Some(1.0f32), Some(2.0), Some(3.0)]),
        Some(vec![Some(4.0f32), Some(5.0), Some(6.0)]),
        Some(vec![Some(7.0f32), Some(8.0), Some(9.0)]),
    ]);
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("v", v.data_type().clone(), true),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(Int32Array::from(vec![1, 2, 3])), Arc::new(v)],
    )
    .unwrap();

    let table_writer = DuckLakeTableWriter::new(Arc::new(writer), object_store).unwrap();
    let res = table_writer
        .write_table("main", "vecs", &[batch])
        .await
        .unwrap();
    assert_eq!(res.records_written, 3);

    // (1) RAW parquet: did the WRITE persist the values, and do leaf columns carry field-ids?
    let data_dir = temp_dir.path().join("data").join("main").join("vecs");
    let pq = std::fs::read_dir(&data_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.extension().map(|x| x == "parquet").unwrap_or(false))
        .expect("a parquet file was written");
    {
        use datafusion::parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
        let file = std::fs::File::open(&pq).unwrap();
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
        let mut reader = builder.build().unwrap();
        let raw = reader.next().unwrap().unwrap();
        let vidx = raw.schema().index_of("v").unwrap();
        assert_eq!(
            raw.column(vidx).null_count(),
            0,
            "WRITE side: raw parquet must persist v values (no nulls)"
        );
    }

    // (2) ducklake READ-BACK: does the read return the values, or null-fill the List?
    let ctx = create_read_context(&temp_dir).await;
    let batches = ctx
        .sql("SELECT v FROM test.main.vecs")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
    let nulls: usize = batches.iter().map(|b| b.column(0).null_count()).sum();
    assert_eq!(total, 3, "read returns 3 rows");
    assert_eq!(
        nulls, 0,
        "READ side: ducklake must return v VALUES, not null-fill the List column"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_write_and_read_list_struct_column_roundtrip() {
    let (writer, temp_dir) = create_test_env().await;
    let object_store = create_object_store();
    const SCALE: i128 = 10_000_000_000_000_000;
    let level_fields = vec![
        Arc::new(Field::new("price", DataType::Decimal128(38, 16), false)),
        Arc::new(Field::new("size", DataType::Decimal128(38, 16), false)),
        Arc::new(Field::new("count", DataType::UInt32, false)),
        Arc::new(Field::new("order_id", DataType::UInt64, false)),
    ];
    let levels = StructArray::new(
        level_fields.clone().into(),
        vec![
            Arc::new(
                Decimal128Array::from(vec![101 * SCALE, 100 * SCALE, 99 * SCALE])
                    .with_precision_and_scale(38, 16)
                    .unwrap(),
            ),
            Arc::new(
                Decimal128Array::from(vec![7 * SCALE, 8 * SCALE, 9 * SCALE])
                    .with_precision_and_scale(38, 16)
                    .unwrap(),
            ),
            Arc::new(UInt32Array::from(vec![1, 2, 3])),
            Arc::new(UInt64Array::from(vec![11, 12, 13])),
        ],
        None,
    );
    let bids = ListArray::new(
        Arc::new(Field::new(
            "element",
            DataType::Struct(level_fields.into()),
            false,
        )),
        OffsetBuffer::new(ScalarBuffer::from(vec![0_i32, 2, 3])),
        Arc::new(levels),
        None,
    );
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("bids", bids.data_type().clone(), false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(Int32Array::from(vec![1, 2])), Arc::new(bids)],
    )
    .unwrap();

    DuckLakeTableWriter::new(Arc::new(writer), object_store)
        .unwrap()
        .write_table("main", "depths", &[batch])
        .await
        .unwrap();

    assert_duckdb_extension_reads_depths(&temp_dir.path().join("test.db"));

    let ctx = create_read_context(&temp_dir).await;
    let batches = ctx
        .sql("SELECT id, bids FROM test.main.depths ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 2);
    let bids = batches[0]
        .column(1)
        .as_any()
        .downcast_ref::<ListArray>()
        .unwrap();
    assert_eq!(bids.value_offsets(), &[0, 2, 3]);
    let levels = bids
        .values()
        .as_any()
        .downcast_ref::<StructArray>()
        .unwrap();
    assert_eq!(
        levels
            .column_by_name("price")
            .unwrap()
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .unwrap()
            .values(),
        &[101 * SCALE, 100 * SCALE, 99 * SCALE]
    );
    assert_eq!(
        levels
            .column_by_name("order_id")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap()
            .values(),
        &[11, 12, 13]
    );

    let pool = sqlx::SqlitePool::connect(&format!(
        "sqlite:{}",
        temp_dir.path().join("test.db").display()
    ))
    .await
    .unwrap();
    let rows = sqlx::query(
        "SELECT c.column_id, c.column_name, c.column_type, c.parent_column
         FROM ducklake_column c
         JOIN ducklake_table t ON t.table_id = c.table_id
         WHERE t.table_name = 'depths' AND c.end_snapshot IS NULL
         ORDER BY c.column_order",
    )
    .fetch_all(&pool)
    .await
    .unwrap();
    assert_eq!(rows.len(), 7);
    let ids = rows
        .iter()
        .map(|row| row.get::<i64, _>("column_id"))
        .collect::<Vec<_>>();
    let actual = rows
        .iter()
        .map(|row| {
            (
                row.get::<String, _>("column_name"),
                row.get::<String, _>("column_type"),
                row.get::<Option<i64>, _>("parent_column"),
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        actual,
        vec![
            ("id".to_string(), "int32".to_string(), None),
            ("bids".to_string(), "list".to_string(), None),
            ("element".to_string(), "struct".to_string(), Some(ids[1])),
            (
                "price".to_string(),
                "decimal(38, 16)".to_string(),
                Some(ids[2]),
            ),
            (
                "size".to_string(),
                "decimal(38, 16)".to_string(),
                Some(ids[2]),
            ),
            ("count".to_string(), "uint32".to_string(), Some(ids[2])),
            ("order_id".to_string(), "uint64".to_string(), Some(ids[2])),
        ]
    );

    let parquet_path = std::fs::read_dir(temp_dir.path().join("data/main/depths"))
        .unwrap()
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .find(|path| {
            path.extension()
                .is_some_and(|extension| extension == "parquet")
        })
        .unwrap();
    assert_eq!(
        parquet_schema_fields(&parquet_path),
        vec![
            ("id".to_string(), Some(ids[0] as i32)),
            ("bids".to_string(), Some(ids[1] as i32)),
            ("list".to_string(), None),
            ("element".to_string(), Some(ids[2] as i32)),
            ("price".to_string(), Some(ids[3] as i32)),
            ("size".to_string(), Some(ids[4] as i32)),
            ("count".to_string(), Some(ids[5] as i32)),
            ("order_id".to_string(), Some(ids[6] as i32)),
        ]
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_write_and_read_struct_list_and_map_roundtrip() {
    let (writer, temp_dir) = create_test_env().await;
    let values = ListArray::new(
        Arc::new(Field::new("item", DataType::Int32, true)),
        OffsetBuffer::new(ScalarBuffer::from(vec![0_i32, 2, 3])),
        Arc::new(Int32Array::from(vec![10, 20, 30])),
        None,
    );
    let payload_fields = vec![
        Arc::new(Field::new("label", DataType::Utf8, false)),
        Arc::new(Field::new("values", values.data_type().clone(), false)),
    ];
    let payload = StructArray::new(
        payload_fields.into(),
        vec![Arc::new(StringArray::from(vec!["first", "second"])), Arc::new(values)],
        None,
    );
    let entry_fields = vec![
        Arc::new(Field::new("key", DataType::Utf8, false)),
        Arc::new(Field::new("value", DataType::Int32, true)),
    ];
    let entries = StructArray::new(
        entry_fields.clone().into(),
        vec![
            Arc::new(StringArray::from(vec!["a", "b", "c"])),
            Arc::new(Int32Array::from(vec![1, 2, 3])),
        ],
        None,
    );
    let attrs = MapArray::new(
        Arc::new(Field::new(
            "entries",
            DataType::Struct(entry_fields.into()),
            false,
        )),
        OffsetBuffer::new(ScalarBuffer::from(vec![0_i32, 2, 3])),
        entries,
        None,
        false,
    );
    let schema = Arc::new(Schema::new(vec![
        Field::new("payload", payload.data_type().clone(), false),
        Field::new("attrs", attrs.data_type().clone(), true),
    ]));
    let batch = RecordBatch::try_new(schema, vec![Arc::new(payload), Arc::new(attrs)]).unwrap();

    DuckLakeTableWriter::new(Arc::new(writer), create_object_store())
        .unwrap()
        .write_table("main", "nested", &[batch])
        .await
        .unwrap();

    let batches = create_read_context(&temp_dir)
        .await
        .sql("SELECT payload, attrs FROM test.main.nested")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let payload = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<StructArray>()
        .unwrap();
    let values = payload
        .column_by_name("values")
        .unwrap()
        .as_any()
        .downcast_ref::<ListArray>()
        .unwrap();
    assert_eq!(values.value_offsets(), &[0, 2, 3]);
    assert_eq!(
        values
            .values()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap()
            .values(),
        &[10, 20, 30]
    );
    let attrs = batches[0]
        .column(1)
        .as_any()
        .downcast_ref::<MapArray>()
        .unwrap();
    assert_eq!(attrs.value_offsets(), &[0, 2, 3]);
    assert_eq!(
        attrs
            .values()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap()
            .values(),
        &[1, 2, 3]
    );

    let pool = sqlx::SqlitePool::connect(&format!(
        "sqlite:{}",
        temp_dir.path().join("test.db").display()
    ))
    .await
    .unwrap();
    let rows = sqlx::query(
        "SELECT c.column_id, c.column_name, c.column_type, c.parent_column
         FROM ducklake_column c
         JOIN ducklake_table t ON t.table_id = c.table_id
         WHERE t.table_name = 'nested' AND c.end_snapshot IS NULL
         ORDER BY c.column_order",
    )
    .fetch_all(&pool)
    .await
    .unwrap();
    let ids = rows
        .iter()
        .map(|row| row.get::<i64, _>("column_id"))
        .collect::<Vec<_>>();
    let actual = rows
        .iter()
        .map(|row| {
            (
                row.get::<String, _>("column_name"),
                row.get::<String, _>("column_type"),
                row.get::<Option<i64>, _>("parent_column"),
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        actual,
        vec![
            ("payload".to_string(), "struct".to_string(), None),
            ("label".to_string(), "varchar".to_string(), Some(ids[0])),
            ("values".to_string(), "list".to_string(), Some(ids[0])),
            ("element".to_string(), "int32".to_string(), Some(ids[2])),
            ("attrs".to_string(), "map".to_string(), None),
            ("key".to_string(), "varchar".to_string(), Some(ids[4])),
            ("value".to_string(), "int32".to_string(), Some(ids[4])),
        ]
    );

    let parquet_path = std::fs::read_dir(temp_dir.path().join("data/main/nested"))
        .unwrap()
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .find(|path| {
            path.extension()
                .is_some_and(|extension| extension == "parquet")
        })
        .unwrap();
    assert_eq!(
        parquet_schema_fields(&parquet_path),
        vec![
            ("payload".to_string(), Some(ids[0] as i32)),
            ("label".to_string(), Some(ids[1] as i32)),
            ("values".to_string(), Some(ids[2] as i32)),
            ("list".to_string(), None),
            ("element".to_string(), Some(ids[3] as i32)),
            ("attrs".to_string(), Some(ids[4] as i32)),
            ("key_value".to_string(), None),
            ("key".to_string(), Some(ids[5] as i32)),
            ("value".to_string(), Some(ids[6] as i32)),
        ]
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_write_temporal_types() {
    let (writer, temp_dir) = create_test_env().await;
    let object_store = create_object_store();

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("date", DataType::Date32, true),
        Field::new(
            "timestamp",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            true,
        ),
    ]));

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![1, 2])),
            Arc::new(Date32Array::from(vec![Some(19000), Some(19001)])), // Days since epoch
            Arc::new(TimestampMicrosecondArray::from(vec![
                Some(1640000000000000),
                Some(1640000001000000),
            ])),
        ],
    )
    .unwrap();

    let table_writer = DuckLakeTableWriter::new(Arc::new(writer), object_store).unwrap();
    let result = table_writer
        .write_table("main", "events", &[batch])
        .await
        .unwrap();
    assert_eq!(result.records_written, 2);

    // Read back
    let ctx = create_read_context(&temp_dir).await;
    let df = ctx
        .sql("SELECT COUNT(*) as cnt FROM test.main.events")
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let count = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!(count, 2);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_write_multiple_batches() {
    let (writer, temp_dir) = create_test_env().await;
    let object_store = create_object_store();

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("value", DataType::Utf8, true),
    ]));

    let batch1 = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(vec![1, 2])), Arc::new(StringArray::from(vec!["a", "b"]))],
    )
    .unwrap();

    let batch2 = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(vec![3, 4])), Arc::new(StringArray::from(vec!["c", "d"]))],
    )
    .unwrap();

    let table_writer = DuckLakeTableWriter::new(Arc::new(writer), object_store).unwrap();
    let result = table_writer
        .write_table("main", "data", &[batch1, batch2])
        .await
        .unwrap();
    assert_eq!(result.records_written, 4);

    // Read back
    let ctx = create_read_context(&temp_dir).await;
    let df = ctx
        .sql("SELECT COUNT(*) as cnt FROM test.main.data")
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let count = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!(count, 4);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_replace_semantics() {
    let (writer, temp_dir) = create_test_env().await;
    let object_store = create_object_store();

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("value", DataType::Int32, true),
    ]));

    // Write initial data
    let batch1 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(Int32Array::from(vec![100, 200, 300])),
        ],
    )
    .unwrap();

    let table_writer =
        DuckLakeTableWriter::new(Arc::new(writer.clone()), Arc::clone(&object_store)).unwrap();
    table_writer
        .write_table("main", "replace_test", &[batch1])
        .await
        .unwrap();

    // Write replacement data
    let batch2 = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(vec![4, 5])), Arc::new(Int32Array::from(vec![400, 500]))],
    )
    .unwrap();

    let table_writer2 =
        DuckLakeTableWriter::new(Arc::new(writer), Arc::clone(&object_store)).unwrap();
    let result = table_writer2
        .write_table("main", "replace_test", &[batch2])
        .await
        .unwrap();
    assert_eq!(result.records_written, 2);

    // Read back - should only have the replacement data
    let ctx = create_read_context(&temp_dir).await;
    let df = ctx
        .sql("SELECT id, value FROM test.main.replace_test ORDER BY id")
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();

    assert_eq!(batches[0].num_rows(), 2);

    let ids = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    assert_eq!(ids.values(), &[4, 5]);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_append_semantics() {
    let (writer, temp_dir) = create_test_env().await;
    let object_store = create_object_store();

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("value", DataType::Int32, true),
    ]));

    // Write initial data
    let batch1 = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(vec![1, 2])), Arc::new(Int32Array::from(vec![100, 200]))],
    )
    .unwrap();

    let table_writer =
        DuckLakeTableWriter::new(Arc::new(writer.clone()), Arc::clone(&object_store)).unwrap();
    table_writer
        .write_table("main", "append_test", &[batch1])
        .await
        .unwrap();

    // Append more data
    let batch2 = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(vec![3, 4])), Arc::new(Int32Array::from(vec![300, 400]))],
    )
    .unwrap();

    let table_writer2 =
        DuckLakeTableWriter::new(Arc::new(writer), Arc::clone(&object_store)).unwrap();
    let result = table_writer2
        .append_table("main", "append_test", &[batch2])
        .await
        .unwrap();
    assert_eq!(result.records_written, 2);

    // Read back - should have all data
    let ctx = create_read_context(&temp_dir).await;
    let df = ctx
        .sql("SELECT COUNT(*) as cnt FROM test.main.append_test")
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();

    let count = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!(count, 4);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_append_preserves_first_file_values() {
    // Regression: a second append must not orphan the first file's column ids.
    // Reading back must return BOTH batches' actual values, not NULLs.
    let (writer, temp_dir) = create_test_env().await;
    let object_store = create_object_store();

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, true),
        Field::new("value", DataType::Int32, true),
    ]));
    let batch1 = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(vec![1, 2])), Arc::new(Int32Array::from(vec![100, 200]))],
    )
    .unwrap();
    DuckLakeTableWriter::new(Arc::new(writer.clone()), Arc::clone(&object_store))
        .unwrap()
        .write_table("main", "t", &[batch1])
        .await
        .unwrap();

    let batch2 = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(vec![3, 4])), Arc::new(Int32Array::from(vec![300, 400]))],
    )
    .unwrap();
    DuckLakeTableWriter::new(Arc::new(writer), Arc::clone(&object_store))
        .unwrap()
        .append_table("main", "t", &[batch2])
        .await
        .unwrap();

    let ctx = create_read_context(&temp_dir).await;
    let batches = ctx
        .sql("SELECT id, value FROM test.main.t ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let mut got: Vec<(i32, i32)> = Vec::new();
    for b in &batches {
        let ids = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        let vals = b.column(1).as_any().downcast_ref::<Int32Array>().unwrap();
        for i in 0..b.num_rows() {
            assert!(
                !ids.is_null(i) && !vals.is_null(i),
                "append lost a row's values (read back NULL)"
            );
            got.push((ids.value(i), vals.value(i)));
        }
    }
    got.sort();
    assert_eq!(got, vec![(1, 100), (2, 200), (3, 300), (4, 400)]);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_append_list_columns_multi_file() {
    // Regression guard for the per-file field-id read mapping on NESTED columns.
    // A write + an append produce two separate Parquet files; each is matched
    // independently via extract_parquet_field_ids. A List column's field-id lives
    // on its top-level node, so a leaf-only walk would null-fill it in EVERY file.
    // The single-file List roundtrip and the multi-file scalar append tests would
    // both still pass under that bug, so this multi-file List case is the one that
    // actually fails if the field-id walk regresses to leaves.
    use arrow::datatypes::Float32Type;

    let (writer, temp_dir) = create_test_env().await;
    let object_store = create_object_store();

    let v1 = arrow::array::ListArray::from_iter_primitive::<Float32Type, _, _>(vec![
        Some(vec![Some(1.0f32), Some(2.0), Some(3.0)]),
        Some(vec![Some(4.0f32), Some(5.0), Some(6.0)]),
    ]);
    let v2 = arrow::array::ListArray::from_iter_primitive::<Float32Type, _, _>(vec![
        Some(vec![Some(7.0f32), Some(8.0), Some(9.0)]),
        Some(vec![Some(10.0f32), Some(11.0), Some(12.0)]),
    ]);
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, true),
        Field::new("v", v1.data_type().clone(), true),
    ]));

    let batch1 = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(vec![1, 2])), Arc::new(v1)],
    )
    .unwrap();
    DuckLakeTableWriter::new(Arc::new(writer.clone()), Arc::clone(&object_store))
        .unwrap()
        .write_table("main", "vt", &[batch1])
        .await
        .unwrap();

    let batch2 = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(vec![3, 4])), Arc::new(v2)],
    )
    .unwrap();
    DuckLakeTableWriter::new(Arc::new(writer), Arc::clone(&object_store))
        .unwrap()
        .append_table("main", "vt", &[batch2])
        .await
        .unwrap();

    let ctx = create_read_context(&temp_dir).await;
    let batches = ctx
        .sql("SELECT id, v FROM test.main.vt ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
    let v_nulls: usize = batches.iter().map(|b| b.column(1).null_count()).sum();
    assert_eq!(total, 4, "both files' rows must be read");
    assert_eq!(
        v_nulls, 0,
        "List column must survive a multi-file (write + append) read; a leaf-only \
         field-id walk would null-fill it in each file"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_append_migrates_legacy_list_catalog_rows() {
    let (writer, temp_dir) = create_test_env().await;
    let object_store = create_object_store();
    let first_values = ListArray::from_iter_primitive::<Float32Type, _, _>(vec![
        Some(vec![Some(1.0), Some(2.0)]),
        Some(vec![Some(3.0)]),
    ]);
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("values", first_values.data_type().clone(), true),
    ]));
    let first = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(vec![1, 2])), Arc::new(first_values)],
    )
    .unwrap();
    let initial = DuckLakeTableWriter::new(Arc::new(writer.clone()), Arc::clone(&object_store))
        .unwrap()
        .write_table("main", "legacy_list", &[first])
        .await
        .unwrap();

    let pool = sqlx::SqlitePool::connect(&format!(
        "sqlite:{}",
        temp_dir.path().join("test.db").display()
    ))
    .await
    .unwrap();
    let rows = sqlx::query_as::<_, (i64, String)>(
        "SELECT column_id, column_name
         FROM ducklake_column
         WHERE table_id = (SELECT table_id FROM ducklake_table WHERE table_name = 'legacy_list')
           AND end_snapshot IS NULL
         ORDER BY column_order",
    )
    .fetch_all(&pool)
    .await
    .unwrap();
    let list_id = rows[1].0;
    let old_element_id = rows[2].0;
    sqlx::query("DELETE FROM ducklake_column WHERE column_id = ?")
        .bind(old_element_id)
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query("UPDATE ducklake_column SET column_type = 'list<float32>' WHERE column_id = ?")
        .bind(list_id)
        .execute(&pool)
        .await
        .unwrap();

    let second_values =
        ListArray::from_iter_primitive::<Float32Type, _, _>(vec![Some(vec![Some(4.0), Some(5.0)])]);
    let second = RecordBatch::try_new(
        schema,
        vec![Arc::new(Int32Array::from(vec![3])), Arc::new(second_values)],
    )
    .unwrap();
    let migration = DuckLakeTableWriter::new(Arc::new(writer), object_store)
        .unwrap()
        .append_table("main", "legacy_list", &[second])
        .await
        .unwrap();

    let list_versions = sqlx::query_as::<_, (String, i64, Option<i64>)>(
        "SELECT column_type, begin_snapshot, end_snapshot
         FROM ducklake_column
         WHERE table_id = (SELECT table_id FROM ducklake_table WHERE table_name = 'legacy_list')
           AND column_id = ?
         ORDER BY begin_snapshot",
    )
    .bind(list_id)
    .fetch_all(&pool)
    .await
    .unwrap();
    assert_eq!(
        list_versions,
        vec![
            (
                "list<float32>".into(),
                initial.snapshot_id,
                Some(migration.snapshot_id),
            ),
            ("list".into(), migration.snapshot_id, None),
        ]
    );

    let catalog_rows = sqlx::query_as::<_, (i64, String, String, Option<i64>)>(
        "SELECT column_id, column_name, column_type, parent_column
         FROM ducklake_column
         WHERE table_id = (SELECT table_id FROM ducklake_table WHERE table_name = 'legacy_list')
           AND end_snapshot IS NULL
         ORDER BY column_order",
    )
    .fetch_all(&pool)
    .await
    .unwrap();
    assert_eq!(catalog_rows.len(), 3);
    assert_eq!(
        catalog_rows[1],
        (list_id, "values".into(), "list".into(), None)
    );
    assert_eq!(catalog_rows[2].1, "element");
    assert_eq!(catalog_rows[2].2, "float32");
    assert_eq!(catalog_rows[2].3, Some(list_id));
    assert_ne!(catalog_rows[2].0, old_element_id);

    let historical = SessionContext::new();
    let provider = Arc::new(
        SqliteMetadataProvider::new(&format!(
            "sqlite:{}",
            temp_dir.path().join("test.db").display()
        ))
        .await
        .unwrap(),
    );
    register_ducklake_functions(&historical, provider);
    let historical_rows = historical
        .sql(&format!(
            "SELECT id FROM ducklake_table_at('main', 'legacy_list', {}) ORDER BY id",
            initial.snapshot_id
        ))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let historical_ids = historical_rows
        .iter()
        .flat_map(|batch| {
            let ids = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            (0..ids.len())
                .map(|index| ids.value(index))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    assert_eq!(historical_ids, vec![1, 2]);

    let batches = create_read_context(&temp_dir)
        .await
        .sql("SELECT id, \"values\" FROM test.main.legacy_list ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let mut actual = Vec::new();
    for batch in batches {
        let ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let values = batch
            .column(1)
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        for row in 0..batch.num_rows() {
            let value = values.value(row);
            let value = value.as_any().downcast_ref::<Float32Array>().unwrap();
            actual.push((ids.value(row), value.values().to_vec()));
        }
    }
    assert_eq!(
        actual,
        vec![(1, vec![1.0, 2.0]), (2, vec![3.0]), (3, vec![4.0, 5.0])]
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_multiple_tables_same_schema() {
    let (writer, temp_dir) = create_test_env().await;
    let object_store = create_object_store();

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true),
    ]));

    let batch1 = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(vec![1])), Arc::new(StringArray::from(vec!["table1"]))],
    )
    .unwrap();

    let batch2 = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(vec![2])), Arc::new(StringArray::from(vec!["table2"]))],
    )
    .unwrap();

    let table_writer = DuckLakeTableWriter::new(Arc::new(writer), object_store).unwrap();
    table_writer
        .write_table("main", "t1", &[batch1])
        .await
        .unwrap();
    table_writer
        .write_table("main", "t2", &[batch2])
        .await
        .unwrap();

    // Read back both tables
    let ctx = create_read_context(&temp_dir).await;

    let df1 = ctx.sql("SELECT name FROM test.main.t1").await.unwrap();
    let batches1 = df1.collect().await.unwrap();
    let names1_arr = arrow::compute::cast(batches1[0].column(0), &DataType::Utf8).unwrap();
    let names1 = names1_arr.as_any().downcast_ref::<StringArray>().unwrap();
    assert_eq!(names1.value(0), "table1");

    let df2 = ctx.sql("SELECT name FROM test.main.t2").await.unwrap();
    let batches2 = df2.collect().await.unwrap();
    let names2_arr = arrow::compute::cast(batches2[0].column(0), &DataType::Utf8).unwrap();
    let names2 = names2_arr.as_any().downcast_ref::<StringArray>().unwrap();
    assert_eq!(names2.value(0), "table2");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_field_ids_preserved_on_roundtrip() {
    let (writer, temp_dir) = create_test_env().await;
    let object_store = create_object_store();

    let schema = Arc::new(Schema::new(vec![
        Field::new("col_a", DataType::Int32, false),
        Field::new("col_b", DataType::Utf8, true),
    ]));

    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(Int32Array::from(vec![1])), Arc::new(StringArray::from(vec!["test"]))],
    )
    .unwrap();

    let table_writer = DuckLakeTableWriter::new(Arc::new(writer), object_store).unwrap();
    table_writer
        .write_table("main", "field_id_test", &[batch])
        .await
        .unwrap();

    // Find the Parquet file and verify field_ids
    let data_path = temp_dir
        .path()
        .join("data")
        .join("main")
        .join("field_id_test");
    let parquet_files: Vec<_> = std::fs::read_dir(&data_path)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "parquet"))
        .collect();

    assert_eq!(parquet_files.len(), 1);

    // Read Parquet file and check field_ids
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    let file = std::fs::File::open(parquet_files[0].path()).unwrap();
    let reader = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
    let metadata = reader.metadata();

    let schema_descr = metadata.file_metadata().schema_descr();
    let mut field_ids = Vec::new();
    for i in 0..schema_descr.num_columns() {
        let column = schema_descr.column(i);
        let basic_info = column.self_type().get_basic_info();
        if basic_info.has_id() {
            field_ids.push(basic_info.id());
        }
    }

    // Should have field_ids for both columns
    assert_eq!(field_ids.len(), 2);
    // Field IDs should be sequential starting from 1
    assert!(field_ids.contains(&1));
    assert!(field_ids.contains(&2));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_streaming_write_api() {
    let (writer, temp_dir) = create_test_env().await;
    let object_store = create_object_store();

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("value", DataType::Utf8, true),
    ]));

    // Use streaming API
    let table_writer = DuckLakeTableWriter::new(Arc::new(writer), object_store).unwrap();
    let mut session = table_writer
        .begin_write("main", "streaming_test", &schema, WriteMode::Replace)
        .unwrap();

    // Write multiple batches incrementally
    for i in 0..3 {
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![i * 10, i * 10 + 1])),
                Arc::new(StringArray::from(vec![
                    format!("val_{}", i * 10),
                    format!("val_{}", i * 10 + 1),
                ])),
            ],
        )
        .unwrap();
        session.write_batch(&batch).unwrap();
    }

    assert_eq!(session.row_count(), 6); // 3 batches * 2 rows

    let result = session.finish().await.unwrap();
    assert_eq!(result.records_written, 6);
    assert_eq!(result.files_written, 1);

    // Read back via DuckLakeCatalog
    let ctx = create_read_context(&temp_dir).await;
    let df = ctx
        .sql("SELECT COUNT(*) as cnt FROM test.main.streaming_test")
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let count = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!(count, 6);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_streaming_write_list_column() {
    use arrow::datatypes::Int32Type;

    let (writer, temp_dir) = create_test_env().await;
    let object_store = create_object_store();
    let values = ListArray::from_iter_primitive::<Int32Type, _, _>(vec![
        Some(vec![Some(10), Some(11)]),
        Some(vec![Some(20)]),
    ]);
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("items", values.data_type().clone(), true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(vec![1, 2])), Arc::new(values)],
    )
    .unwrap();

    let table_writer = DuckLakeTableWriter::new(Arc::new(writer), object_store).unwrap();
    let mut session = table_writer
        .begin_write("main", "streaming_list", &schema, WriteMode::Replace)
        .unwrap();
    session.write_batch(&batch).unwrap();
    assert_eq!(session.finish().await.unwrap().records_written, 2);

    let batches = create_read_context(&temp_dir)
        .await
        .sql("SELECT id, items FROM test.main.streaming_list ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 2);
    let ids = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    let values = batches[0]
        .column(1)
        .as_any()
        .downcast_ref::<ListArray>()
        .unwrap();
    assert_eq!(ids.values(), &[1, 2]);
    let first = values.value(0);
    let first = first.as_any().downcast_ref::<Int32Array>().unwrap();
    let second = values.value(1);
    let second = second.as_any().downcast_ref::<Int32Array>().unwrap();
    assert_eq!(first.values(), &[10, 11]);
    assert_eq!(second.values(), &[20]);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_streaming_write_to_custom_path() {
    let (writer, temp_dir) = create_test_env().await;
    let object_store = create_object_store();

    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));

    // Use custom path (simulating external storage manager)
    let custom_dir = temp_dir.path().join("data").join("custom").join("location");
    let custom_dir_str = custom_dir.to_str().unwrap().to_string();
    let file_name = "my_data.parquet".to_string();

    let table_writer = DuckLakeTableWriter::new(Arc::new(writer), object_store).unwrap();
    let mut session = table_writer
        .begin_write_to_path(
            "main",
            "custom_path_test",
            &schema,
            &custom_dir_str,
            file_name.clone(),
            WriteMode::Replace,
        )
        .unwrap();

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
    )
    .unwrap();
    session.write_batch(&batch).unwrap();

    let result = session.finish().await.unwrap();
    assert_eq!(result.records_written, 3);

    // Verify file exists at custom path
    assert!(custom_dir.join(&file_name).exists());

    // Read back via DuckLakeCatalog
    let ctx = create_read_context(&temp_dir).await;
    let df = ctx
        .sql("SELECT COUNT(*) as cnt FROM test.main.custom_path_test")
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let count = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!(count, 3);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_streaming_empty_write() {
    let (writer, temp_dir) = create_test_env().await;
    let object_store = create_object_store();

    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));

    let table_writer = DuckLakeTableWriter::new(Arc::new(writer), object_store).unwrap();
    let session = table_writer
        .begin_write("main", "empty_test", &schema, WriteMode::Replace)
        .unwrap();

    // Finish without writing any batches
    let result = session.finish().await.unwrap();
    assert_eq!(result.records_written, 0);
    assert_eq!(result.files_written, 1);

    // Read back - should have 0 rows
    let ctx = create_read_context(&temp_dir).await;
    let df = ctx
        .sql("SELECT COUNT(*) as cnt FROM test.main.empty_test")
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let count = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!(count, 0);
}

// Schema Evolution Tests

#[tokio::test(flavor = "multi_thread")]
async fn test_append_add_nullable_column() {
    let (writer, temp_dir) = create_test_env().await;
    let object_store = create_object_store();

    // Initial schema with 2 columns
    let schema1 = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true),
    ]));

    let batch1 = RecordBatch::try_new(
        schema1.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2])),
            Arc::new(StringArray::from(vec!["Alice", "Bob"])),
        ],
    )
    .unwrap();

    let table_writer =
        DuckLakeTableWriter::new(Arc::new(writer.clone()), Arc::clone(&object_store)).unwrap();
    table_writer
        .write_table("main", "evolve_add", &[batch1])
        .await
        .unwrap();

    // Append with new nullable column
    let schema2 = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true),
        Field::new("age", DataType::Int32, true), // New nullable column
    ]));

    let batch2 = RecordBatch::try_new(
        schema2.clone(),
        vec![
            Arc::new(Int32Array::from(vec![3, 4])),
            Arc::new(StringArray::from(vec!["Charlie", "Diana"])),
            Arc::new(Int32Array::from(vec![30, 40])),
        ],
    )
    .unwrap();

    let table_writer2 =
        DuckLakeTableWriter::new(Arc::new(writer), Arc::clone(&object_store)).unwrap();
    let result = table_writer2
        .append_table("main", "evolve_add", &[batch2])
        .await;
    assert!(result.is_ok(), "Adding nullable column should succeed");
    assert_eq!(result.unwrap().records_written, 2);

    // Read back - should have all 4 rows
    let ctx = create_read_context(&temp_dir).await;
    let df = ctx
        .sql("SELECT COUNT(*) as cnt FROM test.main.evolve_add")
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let count = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!(count, 4);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_append_add_non_nullable_struct_field_reads_old_files_as_null() {
    let (writer, temp_dir) = create_test_env().await;
    let object_store = create_object_store();
    let first_fields = vec![
        Arc::new(Field::new("a", DataType::Int32, false)),
        Arc::new(Field::new("b", DataType::Utf8, true)),
    ];
    let first_payload = StructArray::new(
        first_fields.clone().into(),
        vec![
            Arc::new(Int32Array::from(vec![10, 20])),
            Arc::new(StringArray::from(vec!["first", "second"])),
        ],
        None,
    );
    let first_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("payload", DataType::Struct(first_fields.into()), false),
    ]));
    let first = RecordBatch::try_new(
        first_schema,
        vec![Arc::new(Int32Array::from(vec![1, 2])), Arc::new(first_payload)],
    )
    .unwrap();
    DuckLakeTableWriter::new(Arc::new(writer.clone()), Arc::clone(&object_store))
        .unwrap()
        .write_table("main", "evolve_struct", &[first])
        .await
        .unwrap();

    let second_fields = vec![
        Arc::new(Field::new("a", DataType::Int32, false)),
        Arc::new(Field::new("b", DataType::Utf8, true)),
        Arc::new(Field::new("c", DataType::Int32, false)),
    ];
    let second_payload = StructArray::new(
        second_fields.clone().into(),
        vec![
            Arc::new(Int32Array::from(vec![30])),
            Arc::new(StringArray::from(vec!["third"])),
            Arc::new(Int32Array::from(vec![300])),
        ],
        None,
    );
    let second_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("payload", DataType::Struct(second_fields.into()), false),
    ]));
    let second = RecordBatch::try_new(
        second_schema,
        vec![Arc::new(Int32Array::from(vec![3])), Arc::new(second_payload)],
    )
    .unwrap();
    DuckLakeTableWriter::new(Arc::new(writer), object_store)
        .unwrap()
        .append_table("main", "evolve_struct", &[second])
        .await
        .unwrap();

    let batches = create_read_context(&temp_dir)
        .await
        .sql("SELECT id, payload FROM test.main.evolve_struct ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let mut actual = Vec::new();
    for batch in batches {
        let ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let payload = batch
            .column(1)
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();
        let a = payload
            .column_by_name("a")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let c = payload
            .column_by_name("c")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        for row in 0..batch.num_rows() {
            actual.push((
                ids.value(row),
                a.value(row),
                (!c.is_null(row)).then(|| c.value(row)),
            ));
        }
    }
    assert_eq!(
        actual,
        vec![(1, 10, None), (2, 20, None), (3, 30, Some(300))]
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_append_add_non_nullable_list_struct_field_reads_old_files_as_null() {
    let (writer, temp_dir) = create_test_env().await;
    let object_store = create_object_store();
    let first_fields = vec![Arc::new(Field::new("price", DataType::Int32, false))];
    let first_levels = StructArray::new(
        first_fields.clone().into(),
        vec![Arc::new(Int32Array::from(vec![10, 20]))],
        None,
    );
    let first_bids = ListArray::new(
        Arc::new(Field::new(
            "element",
            DataType::Struct(first_fields.into()),
            false,
        )),
        OffsetBuffer::new(ScalarBuffer::from(vec![0_i32, 2])),
        Arc::new(first_levels),
        None,
    );
    let first_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("bids", first_bids.data_type().clone(), false),
    ]));
    let first = RecordBatch::try_new(
        first_schema,
        vec![Arc::new(Int32Array::from(vec![1])), Arc::new(first_bids)],
    )
    .unwrap();
    DuckLakeTableWriter::new(Arc::new(writer.clone()), Arc::clone(&object_store))
        .unwrap()
        .write_table("main", "evolve_depths", &[first])
        .await
        .unwrap();

    let second_fields = vec![
        Arc::new(Field::new("price", DataType::Int32, false)),
        Arc::new(Field::new("venue", DataType::Utf8, false)),
    ];
    let second_levels = StructArray::new(
        second_fields.clone().into(),
        vec![Arc::new(Int32Array::from(vec![30])), Arc::new(StringArray::from(vec!["X"]))],
        None,
    );
    let second_bids = ListArray::new(
        Arc::new(Field::new(
            "element",
            DataType::Struct(second_fields.into()),
            false,
        )),
        OffsetBuffer::new(ScalarBuffer::from(vec![0_i32, 1])),
        Arc::new(second_levels),
        None,
    );
    let second_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("bids", second_bids.data_type().clone(), false),
    ]));
    let second = RecordBatch::try_new(
        second_schema,
        vec![Arc::new(Int32Array::from(vec![2])), Arc::new(second_bids)],
    )
    .unwrap();
    DuckLakeTableWriter::new(Arc::new(writer), object_store)
        .unwrap()
        .append_table("main", "evolve_depths", &[second])
        .await
        .unwrap();

    let batches = create_read_context(&temp_dir)
        .await
        .sql("SELECT id, bids FROM test.main.evolve_depths ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let mut actual = Vec::new();
    for batch in batches {
        let ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let bids = batch
            .column(1)
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        for row in 0..batch.num_rows() {
            let levels = bids.value(row);
            let levels = levels.as_any().downcast_ref::<StructArray>().unwrap();
            let prices = levels
                .column_by_name("price")
                .unwrap()
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            let venues =
                arrow::compute::cast(levels.column_by_name("venue").unwrap(), &DataType::Utf8)
                    .unwrap();
            let venues = venues.as_any().downcast_ref::<StringArray>().unwrap();
            let values = (0..levels.len())
                .map(|index| {
                    (
                        prices.value(index),
                        (!venues.is_null(index)).then(|| venues.value(index).to_string()),
                    )
                })
                .collect::<Vec<_>>();
            actual.push((ids.value(row), values));
        }
    }
    assert_eq!(
        actual,
        vec![(1, vec![(10, None), (20, None)]), (2, vec![(30, Some("X".to_string()))]),]
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_append_remove_column() {
    let (writer, temp_dir) = create_test_env().await;
    let object_store = create_object_store();

    // Initial schema with 3 columns
    let schema1 = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true),
        Field::new("extra", DataType::Utf8, true),
    ]));

    let batch1 = RecordBatch::try_new(
        schema1.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2])),
            Arc::new(StringArray::from(vec!["Alice", "Bob"])),
            Arc::new(StringArray::from(vec!["x", "y"])),
        ],
    )
    .unwrap();

    let table_writer =
        DuckLakeTableWriter::new(Arc::new(writer.clone()), Arc::clone(&object_store)).unwrap();
    table_writer
        .write_table("main", "evolve_remove", &[batch1])
        .await
        .unwrap();

    // Append without the 'extra' column
    let schema2 = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true),
    ]));

    let batch2 = RecordBatch::try_new(
        schema2.clone(),
        vec![
            Arc::new(Int32Array::from(vec![3, 4])),
            Arc::new(StringArray::from(vec!["Charlie", "Diana"])),
        ],
    )
    .unwrap();

    let table_writer2 =
        DuckLakeTableWriter::new(Arc::new(writer), Arc::clone(&object_store)).unwrap();
    let result = table_writer2
        .append_table("main", "evolve_remove", &[batch2])
        .await;
    assert!(result.is_ok(), "Removing column should succeed");
    assert_eq!(result.unwrap().records_written, 2);

    // Read back - should have all 4 rows
    let ctx = create_read_context(&temp_dir).await;
    let df = ctx
        .sql("SELECT COUNT(*) as cnt FROM test.main.evolve_remove")
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let count = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!(count, 4);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_append_type_mismatch_fails() {
    let (writer, _temp_dir) = create_test_env().await;
    let object_store = create_object_store();

    // Initial schema
    let schema1 = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("value", DataType::Int32, true),
    ]));

    let batch1 = RecordBatch::try_new(
        schema1.clone(),
        vec![Arc::new(Int32Array::from(vec![1, 2])), Arc::new(Int32Array::from(vec![100, 200]))],
    )
    .unwrap();

    let table_writer =
        DuckLakeTableWriter::new(Arc::new(writer.clone()), Arc::clone(&object_store)).unwrap();
    table_writer
        .write_table("main", "evolve_type", &[batch1])
        .await
        .unwrap();

    // Try to append with different type for 'value'
    let schema2 = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("value", DataType::Utf8, true), // Changed from Int32 to Utf8
    ]));

    let batch2 = RecordBatch::try_new(
        schema2.clone(),
        vec![Arc::new(Int32Array::from(vec![3])), Arc::new(StringArray::from(vec!["text"]))],
    )
    .unwrap();

    let table_writer2 =
        DuckLakeTableWriter::new(Arc::new(writer), Arc::clone(&object_store)).unwrap();
    let result = table_writer2
        .append_table("main", "evolve_type", &[batch2])
        .await;
    assert!(result.is_err(), "Type mismatch should fail");
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("type") && err.contains("value"),
        "Error should mention type mismatch for 'value' column: {}",
        err
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_append_non_nullable_column_fails() {
    let (writer, _temp_dir) = create_test_env().await;
    let object_store = create_object_store();

    // Initial schema
    let schema1 = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true),
    ]));

    let batch1 = RecordBatch::try_new(
        schema1.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2])),
            Arc::new(StringArray::from(vec!["Alice", "Bob"])),
        ],
    )
    .unwrap();

    let table_writer =
        DuckLakeTableWriter::new(Arc::new(writer.clone()), Arc::clone(&object_store)).unwrap();
    table_writer
        .write_table("main", "evolve_nonnull", &[batch1])
        .await
        .unwrap();

    // Try to add a non-nullable column
    let schema2 = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true),
        Field::new("required_field", DataType::Int32, false), // New non-nullable column
    ]));

    let batch2 = RecordBatch::try_new(
        schema2.clone(),
        vec![
            Arc::new(Int32Array::from(vec![3])),
            Arc::new(StringArray::from(vec!["Charlie"])),
            Arc::new(Int32Array::from(vec![999])),
        ],
    )
    .unwrap();

    let table_writer2 =
        DuckLakeTableWriter::new(Arc::new(writer), Arc::clone(&object_store)).unwrap();
    let result = table_writer2
        .append_table("main", "evolve_nonnull", &[batch2])
        .await;
    assert!(result.is_err(), "Adding non-nullable column should fail");
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("nullable") && err.contains("required_field"),
        "Error should mention that new column must be nullable: {}",
        err
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_append_reorder_columns() {
    let (writer, temp_dir) = create_test_env().await;
    let object_store = create_object_store();

    // Initial schema: id, name, value
    let schema1 = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true),
        Field::new("value", DataType::Int32, true),
    ]));

    let batch1 = RecordBatch::try_new(
        schema1.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2])),
            Arc::new(StringArray::from(vec!["Alice", "Bob"])),
            Arc::new(Int32Array::from(vec![100, 200])),
        ],
    )
    .unwrap();

    let table_writer =
        DuckLakeTableWriter::new(Arc::new(writer.clone()), Arc::clone(&object_store)).unwrap();
    table_writer
        .write_table("main", "evolve_reorder", &[batch1])
        .await
        .unwrap();

    // Append with reordered columns: value, id, name
    let schema2 = Arc::new(Schema::new(vec![
        Field::new("value", DataType::Int32, true),
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true),
    ]));

    let batch2 = RecordBatch::try_new(
        schema2.clone(),
        vec![
            Arc::new(Int32Array::from(vec![300, 400])),
            Arc::new(Int32Array::from(vec![3, 4])),
            Arc::new(StringArray::from(vec!["Charlie", "Diana"])),
        ],
    )
    .unwrap();

    let table_writer2 =
        DuckLakeTableWriter::new(Arc::new(writer), Arc::clone(&object_store)).unwrap();
    let result = table_writer2
        .append_table("main", "evolve_reorder", &[batch2])
        .await;
    assert!(result.is_ok(), "Reordering columns should succeed");
    assert_eq!(result.unwrap().records_written, 2);

    // Read back - should have all 4 rows
    let ctx = create_read_context(&temp_dir).await;
    let df = ctx
        .sql("SELECT COUNT(*) as cnt FROM test.main.evolve_reorder")
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let count = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!(count, 4);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_zero_column_table_rejected() {
    let (writer, _temp_dir) = create_test_env().await;
    let object_store = create_object_store();

    // An empty Arrow schema (zero columns)
    let schema = Arc::new(Schema::empty());

    let batch = RecordBatch::new_empty(schema.clone());

    let table_writer = DuckLakeTableWriter::new(Arc::new(writer), object_store).unwrap();
    let result = table_writer
        .write_table("main", "empty_cols", &[batch])
        .await;
    assert!(result.is_err(), "Zero-column table should be rejected");
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("at least one column"),
        "Error should mention needing at least one column: {}",
        err
    );
}

/// Write a table larger than `object_store::buffered::BufWriter`'s 10 MiB
/// capacity so `finish()` takes the multipart-upload branch rather than a
/// single PUT, then read it back. This is the path that lifts the object
/// store's single-PUT size ceiling, so exercise it end to end.
#[tokio::test(flavor = "multi_thread")]
async fn test_streaming_write_large_file_uses_multipart() {
    let (writer, temp_dir) = create_test_env().await;
    let object_store = create_object_store();

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("payload", DataType::Int64, false),
    ]));

    // ~1.5M rows * 16 bytes ≈ 24 MB of high-entropy (uncompressed) parquet,
    // comfortably above the 10 MiB BufWriter capacity that triggers multipart.
    const ROWS: i64 = 1_500_000;
    let ids: Vec<i64> = (0..ROWS).collect();
    let payload: Vec<i64> = (0..ROWS).map(|i| i.wrapping_mul(2_654_435_761)).collect();
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int64Array::from(ids)), Arc::new(Int64Array::from(payload))],
    )
    .unwrap();

    let table_writer = DuckLakeTableWriter::new(Arc::new(writer), object_store).unwrap();
    let result = table_writer
        .write_table("main", "big", &[batch])
        .await
        .unwrap();
    assert_eq!(result.records_written, ROWS);
    assert_eq!(result.files_written, 1);

    // Read back: a complete, valid parquet means the multipart upload assembled
    // correctly. Verify both the row count and an aggregate over every row.
    let ctx = create_read_context(&temp_dir).await;
    let batches = ctx
        .sql("SELECT COUNT(*) AS n, SUM(id) AS s FROM test.main.big")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let n = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    let s = batches[0]
        .column(1)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!(n, ROWS);
    assert_eq!(s, ROWS * (ROWS - 1) / 2);
}

/// End-to-end regression for nanosecond tz-aware timestamps.
///
/// A `Timestamp(Nanosecond, Some(tz))` column (the pandas/PyArrow default for
/// tz-aware datetimes) used to be cataloged as µs `timestamptz`, so the served
/// schema disagreed with the physical parquet and the read path silently
/// truncated the sub-microsecond fraction on every scan. This writes ns values
/// with non-zero sub-µs digits and asserts they survive the full
/// write -> catalog -> read round-trip at full precision.
#[tokio::test(flavor = "multi_thread")]
async fn test_write_and_read_nanosecond_timestamptz() {
    let (writer, temp_dir) = create_test_env().await;
    let object_store = create_object_store();

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new(
            "event_ts",
            DataType::Timestamp(TimeUnit::Nanosecond, Some("UTC".into())),
            true,
        ),
    ]));

    // Values with a non-zero sub-microsecond fraction: truncation to µs would
    // zero the last three digits, so exact equality proves no precision loss.
    let ns_values = vec![1_000_000_000_123_456_789i64, 1_700_000_000_987_654_321i64];
    let ts_array = TimestampNanosecondArray::from(ns_values.clone()).with_timezone("UTC");

    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(Int32Array::from(vec![1, 2])), Arc::new(ts_array)],
    )
    .unwrap();

    let table_writer = DuckLakeTableWriter::new(Arc::new(writer), object_store).unwrap();
    table_writer
        .write_table("main", "events", &[batch])
        .await
        .unwrap();

    let ctx = create_read_context(&temp_dir).await;
    let df = ctx
        .sql("SELECT event_ts FROM test.main.events ORDER BY id")
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();

    // Served schema must keep nanosecond precision, not collapse to µs.
    assert_eq!(
        batches[0].schema().field(0).data_type(),
        &DataType::Timestamp(TimeUnit::Nanosecond, Some("UTC".into())),
        "served column must round-trip as nanosecond tz-aware"
    );

    let col = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<TimestampNanosecondArray>()
        .expect("column reads back as TimestampNanosecondArray");
    assert_eq!(
        col.values(),
        ns_values.as_slice(),
        "sub-microsecond fraction must survive the round-trip"
    );
}
