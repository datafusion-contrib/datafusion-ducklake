//! A DuckDB `ATTACH` reads the Parquet files this crate writes.
//!
//! DuckDB resolves a data file as `data_path`, schema path, table path, and file
//! name concatenated without separators, and validates the stored `footer_size`
//! against the file. A catalog that stores slash-less paths or a footer size that
//! includes the 8-byte trailer is unreadable to it, so only a full read through
//! an attach catches either defect.

#![cfg(all(feature = "write-duckdb", feature = "metadata-duckdb"))]

use std::sync::Arc;

use arrow::array::{Int32Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion_ducklake::{DuckLakeTableWriter, DuckdbMetadataWriter, MetadataWriter};
use object_store::local::LocalFileSystem;
use tempfile::TempDir;

#[tokio::test(flavor = "multi_thread")]
async fn duckdb_attach_reads_rows_the_crate_wrote() {
    let temp = TempDir::new().unwrap();
    let database = temp.path().join("catalog.ducklake");
    let database = database.to_str().unwrap().to_string();
    let data_path = temp.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();

    let writer = DuckdbMetadataWriter::new_with_init(&database).unwrap();
    writer.set_data_path(data_path.to_str().unwrap()).unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![7, 11, 13])),
            Arc::new(StringArray::from(vec![Some("a"), None, Some("c")])),
        ],
    )
    .unwrap();
    DuckLakeTableWriter::new(Arc::new(writer), Arc::new(LocalFileSystem::new()))
        .unwrap()
        .write_table("main", "readable", &[batch])
        .await
        .unwrap();

    // DuckDB allows one client per catalog file, and the writer is dropped above.
    let oracle = duckdb::Connection::open_in_memory().unwrap();
    oracle
        .execute_batch(&format!(
            "LOAD ducklake; ATTACH 'ducklake:{database}' AS lake (READ_ONLY);"
        ))
        .unwrap();
    let mut statement = oracle
        .prepare("SELECT id, name FROM lake.main.readable ORDER BY id")
        .unwrap();
    let rows: Vec<(i32, Option<String>)> = statement
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();

    assert_eq!(
        rows,
        vec![(7, Some("a".to_string())), (11, None), (13, Some("c".to_string())),]
    );
}
