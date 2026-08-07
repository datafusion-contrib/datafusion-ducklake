//! The read schema a nested column is scanned with must declare the field ids
//! the physical file carries.
//!
//! DuckLake tags every semantic node of a nested column with its own field id —
//! a List element, each Struct child, a Map's key and value, at any depth — while
//! the synthetic parquet wrapper groups (`list`, `key_value`) stay untagged. A
//! nested node's `PARQUET:field_id` lives in its parent's Arrow *type*
//! (`DataType::List`/`Struct`/`Map` embed whole `Field`s, metadata included), so
//! it takes part in every array and record-batch type check. A read schema that
//! omits it therefore disagrees with the batches the parquet reader produces from
//! the very file it describes, and a caller pairing the two gets
//! "column types must match schema types".
//!
//! Field ids are the only part of that agreement these tests establish. The read
//! schema is type-identical to the file for a nested column that contains neither
//! a MAP nor a string; two known differences remain outside the scope of this
//! module. Both are older than this fix, and both surface as the same
//! "column types must match schema types" error:
//!
//! - **Map wrapper name** — the read schema keeps the wrapper group the catalog
//!   built, named `entries`, while this crate's writer and DuckLake itself name
//!   it `key_value`. Present since MAP support arrived in #230.
//! - **Nested VARCHAR** — a string is served as `Utf8View` while parquet stores
//!   `Utf8` (#160, predating #230). DataFusion coerces top-level fields only, so
//!   the difference stands for a string inside a List, Struct or Map.
//!
//! This crate's own scan absorbs both — `ColumnRenameExec` relabels the wrapper,
//! and the cast `ParquetOpener` inserts widens the string — so they are invisible
//! to a query and visible only to a caller driving arrow-rs from
//! `build_read_schema_with_field_id_mapping` directly. The fixtures below are
//! integral and float throughout so that neither masks the field-id comparison.

#![cfg(all(feature = "write-sqlite", feature = "metadata-sqlite"))]

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{Array, Float32Array, Int32Array, ListArray, MapArray, StructArray};
use arrow::buffer::{OffsetBuffer, ScalarBuffer};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::parquet::arrow::PARQUET_FIELD_ID_META_KEY;
use datafusion::parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use datafusion::prelude::*;
use object_store::local::LocalFileSystem;
use tempfile::TempDir;

use datafusion_ducklake::metadata_provider::MetadataProvider;
use datafusion_ducklake::types::{
    build_read_schema_with_field_id_mapping, extract_parquet_field_ids,
};
use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeTableWriter, MetadataWriter, SqliteMetadataProvider,
    SqliteMetadataWriter,
};

/// `id INT, v FLOAT[], s STRUCT<a INT, b INT>, m MAP<INT, INT>, nn INT[][]`.
///
/// Every column is integral or float on purpose: the read path serves strings as
/// `Utf8View` while parquet stores `Utf8`, and that deliberate difference would
/// mask the field-id comparison this test is about.
fn nested_batch() -> RecordBatch {
    let v = ListArray::new(
        Arc::new(Field::new("element", DataType::Float32, true)),
        OffsetBuffer::new(ScalarBuffer::from(vec![0_i32, 2, 3])),
        Arc::new(Float32Array::from(vec![1.0_f32, 2.0, 3.0])),
        None,
    );
    let s = StructArray::new(
        vec![
            Arc::new(Field::new("a", DataType::Int32, true)),
            Arc::new(Field::new("b", DataType::Int32, true)),
        ]
        .into(),
        vec![Arc::new(Int32Array::from(vec![10, 20])), Arc::new(Int32Array::from(vec![30, 40]))],
        None,
    );
    let entry_fields = vec![
        Arc::new(Field::new("key", DataType::Int32, false)),
        Arc::new(Field::new("value", DataType::Int32, true)),
    ];
    let m = MapArray::new(
        Arc::new(Field::new(
            "entries",
            DataType::Struct(entry_fields.clone().into()),
            false,
        )),
        OffsetBuffer::new(ScalarBuffer::from(vec![0_i32, 2, 3])),
        StructArray::new(
            entry_fields.into(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(Int32Array::from(vec![7, 8, 9])),
            ],
            None,
        ),
        None,
        false,
    );
    let inner = ListArray::new(
        Arc::new(Field::new("element", DataType::Int32, true)),
        OffsetBuffer::new(ScalarBuffer::from(vec![0_i32, 2, 3, 4])),
        Arc::new(Int32Array::from(vec![1, 2, 3, 4])),
        None,
    );
    let nn = ListArray::new(
        Arc::new(Field::new("element", inner.data_type().clone(), true)),
        OffsetBuffer::new(ScalarBuffer::from(vec![0_i32, 2, 3])),
        Arc::new(inner),
        None,
    );

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("v", v.data_type().clone(), true),
        Field::new("s", s.data_type().clone(), true),
        Field::new("m", m.data_type().clone(), true),
        Field::new("nn", nn.data_type().clone(), true),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![1, 2])),
            Arc::new(v),
            Arc::new(s),
            Arc::new(m),
            Arc::new(nn),
        ],
    )
    .unwrap()
}

/// Write [`nested_batch`] as a DuckLake table and return the catalog directory
/// plus the single parquet file it produced.
async fn write_nested_table() -> (TempDir, std::path::PathBuf) {
    let temp_dir = TempDir::new().unwrap();
    let data_path = temp_dir.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();
    let writer = SqliteMetadataWriter::new_with_init(&format!(
        "sqlite:{}?mode=rwc",
        temp_dir.path().join("test.db").display()
    ))
    .await
    .unwrap();
    writer.set_data_path(data_path.to_str().unwrap()).unwrap();

    DuckLakeTableWriter::new(
        Arc::new(writer),
        Arc::new(LocalFileSystem::new()) as Arc<dyn object_store::ObjectStore>,
    )
    .unwrap()
    .write_table("main", "nested", &[nested_batch()])
    .await
    .unwrap();

    let path = std::fs::read_dir(data_path.join("main").join("nested"))
        .unwrap()
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .find(|path| path.extension().is_some_and(|ext| ext == "parquet"))
        .expect("a parquet file was written");
    (temp_dir, path)
}

/// The crate's read schema for the written file: what it hands the parquet
/// reader as that file's schema.
async fn read_schema_for(
    temp_dir: &TempDir,
    file_schema: &Schema,
    parquet_field_ids: &HashMap<i32, String>,
) -> Schema {
    let provider = SqliteMetadataProvider::new(&format!(
        "sqlite:{}",
        temp_dir.path().join("test.db").display()
    ))
    .await
    .unwrap();
    let snapshot = provider.get_current_snapshot().unwrap();
    let table_id = provider.list_all_tables(snapshot).unwrap()[0]
        .table
        .table_id;
    let columns = provider.get_table_structure(table_id, snapshot).unwrap();
    build_read_schema_with_field_id_mapping(&columns, parquet_field_ids, Some(file_schema))
        .unwrap()
        .0
}

/// Every parquet node, depth first, as `(name, field id)` — `None` for the
/// synthetic wrapper groups DuckLake leaves untagged.
fn parquet_nodes(path: &std::path::Path) -> Vec<(String, Option<i32>)> {
    fn collect(
        node: &datafusion::parquet::schema::types::Type,
        out: &mut Vec<(String, Option<i32>)>,
    ) {
        let info = node.get_basic_info();
        out.push((node.name().to_string(), info.has_id().then(|| info.id())));
        if node.is_group() {
            for child in node.get_fields() {
                collect(child, out);
            }
        }
    }

    let builder =
        ParquetRecordBatchReaderBuilder::try_new(std::fs::File::open(path).unwrap()).unwrap();
    let mut out = Vec::new();
    for field in builder
        .metadata()
        .file_metadata()
        .schema_descr()
        .root_schema()
        .get_fields()
    {
        collect(field, &mut out);
    }
    out
}

/// `(dotted path, field id)` for every nested node of `data_type`, reading the id
/// from `PARQUET:field_id`. Untagged nodes are reported as `None` so a missing id
/// shows up in the comparison instead of vanishing. The Map wrapper is descended
/// through, not recorded: it is a parquet group with no DuckLake column.
fn nested_field_ids(data_type: &DataType) -> Vec<(String, Option<String>)> {
    fn walk(prefix: &str, field: &Field, out: &mut Vec<(String, Option<String>)>) {
        let path = format!("{prefix}.{}", field.name());
        out.push((
            path.clone(),
            field.metadata().get(PARQUET_FIELD_ID_META_KEY).cloned(),
        ));
        collect(&path, field.data_type(), out);
    }
    fn collect(prefix: &str, data_type: &DataType, out: &mut Vec<(String, Option<String>)>) {
        match data_type {
            DataType::List(child)
            | DataType::LargeList(child)
            | DataType::FixedSizeList(child, _) => walk(prefix, child, out),
            DataType::Struct(children) => {
                for child in children {
                    walk(prefix, child, out);
                }
            },
            DataType::Map(entries, _) => collect(prefix, entries.data_type(), out),
            _ => {},
        }
    }

    let mut out = Vec::new();
    collect("", data_type, &mut out);
    out
}

fn schema_nested_field_ids(schema: &Schema) -> Vec<(String, Option<String>)> {
    schema
        .fields()
        .iter()
        .flat_map(|field| {
            nested_field_ids(field.data_type())
                .into_iter()
                .map(|(path, id)| (format!("{}{path}", field.name()), id))
        })
        .collect()
}

/// The written file is the reference: DuckLake numbers every semantic node in one
/// depth-first sequence and leaves the wrapper groups untagged.
#[tokio::test(flavor = "multi_thread")]
async fn written_file_tags_every_nested_node_and_no_wrapper() {
    let (_temp_dir, path) = write_nested_table().await;

    assert_eq!(
        parquet_nodes(&path),
        vec![
            ("id".to_string(), Some(1)),
            ("v".to_string(), Some(2)),
            ("list".to_string(), None),
            ("element".to_string(), Some(3)),
            ("s".to_string(), Some(4)),
            ("a".to_string(), Some(5)),
            ("b".to_string(), Some(6)),
            ("m".to_string(), Some(7)),
            ("key_value".to_string(), None),
            ("key".to_string(), Some(8)),
            ("value".to_string(), Some(9)),
            ("nn".to_string(), Some(10)),
            ("list".to_string(), None),
            ("element".to_string(), Some(11)),
            ("list".to_string(), None),
            ("element".to_string(), Some(12)),
        ],
        "field ids must run depth-first across the whole schema, with the \
         synthetic list/key_value groups untagged"
    );
}

/// Regression: the read schema must declare the same nested field ids the file
/// carries, for List elements, Struct children, Map key/value, and at every depth.
#[tokio::test(flavor = "multi_thread")]
async fn read_schema_declares_the_nested_field_ids_the_file_carries() {
    let (temp_dir, path) = write_nested_table().await;
    let builder =
        ParquetRecordBatchReaderBuilder::try_new(std::fs::File::open(&path).unwrap()).unwrap();
    let file_schema = builder.schema().as_ref().clone();
    let field_ids = extract_parquet_field_ids(builder.metadata());

    let read_schema = read_schema_for(&temp_dir, &file_schema, &field_ids).await;

    assert_eq!(
        schema_nested_field_ids(&read_schema),
        schema_nested_field_ids(&file_schema),
        "the read schema must describe the file's nested nodes, field ids included"
    );
    // Spelled out, so a future change that "fixes" a mismatch by stripping ids
    // from both sides fails here.
    assert_eq!(
        schema_nested_field_ids(&read_schema),
        vec![
            ("v.element".to_string(), Some("3".to_string())),
            ("s.a".to_string(), Some("5".to_string())),
            ("s.b".to_string(), Some("6".to_string())),
            ("m.key".to_string(), Some("8".to_string())),
            ("m.value".to_string(), Some("9".to_string())),
            ("nn.element".to_string(), Some("11".to_string())),
            ("nn.element.element".to_string(), Some("12".to_string())),
        ],
    );

    // The Map wrapper group carries no id in the file and must carry none here.
    let DataType::Map(entries, _) = read_schema.field(3).data_type() else {
        panic!("m must remain a map");
    };
    assert!(
        !entries.metadata().contains_key(PARQUET_FIELD_ID_META_KEY),
        "the map wrapper group has no DuckLake column and must stay untagged"
    );
}

/// Regression, as a caller sees it: batches the parquet reader produces from a
/// data file must validate against the schema the crate describes that file with.
#[tokio::test(flavor = "multi_thread")]
async fn file_batches_validate_against_the_crate_read_schema() {
    let (temp_dir, path) = write_nested_table().await;
    let builder =
        ParquetRecordBatchReaderBuilder::try_new(std::fs::File::open(&path).unwrap()).unwrap();
    let file_schema = builder.schema().as_ref().clone();
    let field_ids = extract_parquet_field_ids(builder.metadata());
    let read_schema = read_schema_for(&temp_dir, &file_schema, &field_ids).await;

    // `m` is excluded: the read schema names the Map wrapper group `entries`
    // where the file names it `key_value`. That difference sits in the Arrow type
    // too, but it is not about field ids and predates them — the scan reconciles
    // it in `ColumnRenameExec`.
    let projection: Vec<usize> = (0..read_schema.fields().len())
        .filter(|index| read_schema.field(*index).name() != "m")
        .collect();
    let projected = Arc::new(read_schema.project(&projection).unwrap());

    let mut reader = builder.build().unwrap();
    let raw = reader.next().unwrap().unwrap();
    let columns: Vec<_> = projection
        .iter()
        .map(|index| Arc::clone(raw.column(*index)))
        .collect();

    RecordBatch::try_new(projected, columns)
        .expect("a batch read from the file must match the file's read schema");
}

/// The other direction: the fix must not push parquet field ids into the table's
/// public schema. The catalog schema stays free of them and the scan keeps
/// emitting batches that validate against it.
#[tokio::test(flavor = "multi_thread")]
async fn scan_output_still_matches_the_catalog_schema() {
    let (temp_dir, _path) = write_nested_table().await;

    let provider = SqliteMetadataProvider::new(&format!(
        "sqlite:{}",
        temp_dir.path().join("test.db").display()
    ))
    .await
    .unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("test", Arc::new(DuckLakeCatalog::new(provider).unwrap()));
    let table = ctx
        .catalog("test")
        .unwrap()
        .schema("main")
        .unwrap()
        .table("nested")
        .await
        .unwrap()
        .unwrap();
    let advertised = table.schema();

    assert!(
        schema_nested_field_ids(&advertised)
            .iter()
            .all(|(_, id)| id.is_none()),
        "the catalog schema is the logical one; storage field ids do not belong \
         in it: {advertised:#?}"
    );

    let batches = ctx
        .sql("SELECT * FROM test.main.nested")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 2);
    for batch in &batches {
        RecordBatch::try_new(Arc::clone(&advertised), batch.columns().to_vec())
            .expect("scan output must validate against the advertised table schema");
    }
}

/// Declaring the ids makes a struct column's read schema differ from the catalog
/// schema by metadata alone, which routes the scan through `ColumnRenameExec`.
/// That must not cost the scan its predicate pushdown: the difference is a
/// relabel, not a conversion.
#[tokio::test(flavor = "multi_thread")]
async fn a_struct_column_keeps_its_predicate_pushdown() {
    let temp_dir = TempDir::new().unwrap();
    let data_path = temp_dir.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();
    let writer = SqliteMetadataWriter::new_with_init(&format!(
        "sqlite:{}?mode=rwc",
        temp_dir.path().join("test.db").display()
    ))
    .await
    .unwrap();
    writer.set_data_path(data_path.to_str().unwrap()).unwrap();

    let s = StructArray::new(
        vec![Arc::new(Field::new("a", DataType::Int32, true))].into(),
        vec![Arc::new(Int32Array::from(vec![10, 20]))],
        None,
    );
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, true),
        Field::new("s", s.data_type().clone(), true),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(Int32Array::from(vec![1, 2])), Arc::new(s)],
    )
    .unwrap();
    DuckLakeTableWriter::new(
        Arc::new(writer),
        Arc::new(LocalFileSystem::new()) as Arc<dyn object_store::ObjectStore>,
    )
    .unwrap()
    .write_table("main", "structs", &[batch])
    .await
    .unwrap();

    let provider = SqliteMetadataProvider::new(&format!(
        "sqlite:{}",
        temp_dir.path().join("test.db").display()
    ))
    .await
    .unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("test", Arc::new(DuckLakeCatalog::new(provider).unwrap()));

    let explained = ctx
        .sql("EXPLAIN SELECT * FROM test.main.structs WHERE id = 2")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let plan = arrow::util::pretty::pretty_format_batches(&explained)
        .unwrap()
        .to_string();
    assert!(
        plan.contains("predicate=id@0 = 2"),
        "the filter must still reach the parquet scan:\n{plan}"
    );

    let rows = ctx
        .sql("SELECT id FROM test.main.structs WHERE id = 2")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(rows.iter().map(RecordBatch::num_rows).sum::<usize>(), 1);
}
