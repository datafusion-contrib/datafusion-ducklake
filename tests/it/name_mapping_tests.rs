#![cfg(feature = "metadata-duckdb")]

use std::path::Path;
use std::sync::Arc;

use arrow::array::{Int32Array, StringArray};
use datafusion::prelude::SessionContext;
use datafusion_ducklake::{DuckLakeCatalog, DuckdbMetadataProvider};
use tempfile::TempDir;

#[derive(Debug, PartialEq, Eq)]
struct MappedRow {
    id: i32,
    name: String,
    nested_a: i32,
    nested_b: String,
    part: i32,
}

fn create_name_mapping_catalog(
    catalog_path: &Path,
    data_path: &Path,
) -> anyhow::Result<Vec<MappedRow>> {
    let first_hive_path = data_path.join("part=9");
    let second_hive_path = data_path.join("part=10");
    std::fs::create_dir_all(&first_hive_path)?;
    std::fs::create_dir_all(&second_hive_path)?;
    let first_parquet_path = first_hive_path.join("mapped.parquet");
    let second_parquet_path = second_hive_path.join("mapped.parquet");

    let conn = duckdb::Connection::open_in_memory()?;
    conn.execute("INSTALL ducklake", [])?;
    conn.execute("LOAD ducklake", [])?;
    conn.execute("INSTALL parquet", [])?;
    conn.execute(
        &format!(
            "ATTACH 'ducklake:{}' AS lake (DATA_PATH '{}', DATA_INLINING_ROW_LIMIT 0)",
            catalog_path.display(),
            data_path.display()
        ),
        [],
    )?;
    conn.execute(
        "CREATE TABLE lake.mapped(
            source_id INTEGER,
            source_name VARCHAR,
            nested STRUCT(a INTEGER, b VARCHAR),
            part INTEGER
        )",
        [],
    )?;
    conn.execute(
        &format!(
            "COPY (
                SELECT {{'b': 'nested', 'a': 7}} AS nested,
                       42 AS source_id,
                       'value' AS source_name
             ) TO '{}' (FORMAT PARQUET)",
            first_parquet_path.display()
        ),
        [],
    )?;
    conn.execute(
        &format!(
            "COPY (
                SELECT {{'b': 'second', 'a': 8}} AS nested,
                       43 AS source_id,
                       'next' AS source_name
             ) TO '{}' (FORMAT PARQUET)",
            second_parquet_path.display()
        ),
        [],
    )?;
    conn.execute(
        &format!(
            "CALL ducklake_add_data_files(
                'lake', 'mapped', '{}/**/*.parquet', hive_partitioning => true
            )",
            data_path.display()
        ),
        [],
    )?;
    conn.execute("ALTER TABLE lake.mapped RENAME COLUMN source_id TO id", [])?;
    conn.execute(
        "ALTER TABLE lake.mapped RENAME COLUMN source_name TO name",
        [],
    )?;

    let mut statement = conn.prepare(
        "SELECT id, name, nested.a, nested.b, part
         FROM lake.mapped
         WHERE nested.a >= 7 AND part IN (9, 10)
         ORDER BY id",
    )?;
    let rows = statement
        .query_map([], |row| {
            Ok(MappedRow {
                id: row.get(0)?,
                name: row.get(1)?,
                nested_a: row.get(2)?,
                nested_b: row.get(3)?,
                part: row.get(4)?,
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(rows)
}

#[tokio::test]
async fn add_data_files_name_mapping_matches_duckdb() -> anyhow::Result<()> {
    let temp = TempDir::new()?;
    let catalog_path = temp.path().join("mapping.ducklake");
    let data_path = temp.path().join("data");
    let expected = create_name_mapping_catalog(&catalog_path, &data_path)?;
    assert_eq!(
        expected,
        vec![
            MappedRow {
                id: 42,
                name: "value".to_string(),
                nested_a: 7,
                nested_b: "nested".to_string(),
                part: 9,
            },
            MappedRow {
                id: 43,
                name: "next".to_string(),
                nested_a: 8,
                nested_b: "second".to_string(),
                part: 10,
            },
        ]
    );

    let provider = DuckdbMetadataProvider::new(catalog_path.to_string_lossy())?;
    let catalog = DuckLakeCatalog::new(provider)?;
    let context = SessionContext::new();
    context.register_catalog("ducklake", Arc::new(catalog));

    let batches = context
        .sql(
            "SELECT id, name, nested.a, nested.b, part
             FROM ducklake.main.mapped
             WHERE nested.a >= 7 AND part IN (9, 10)
             ORDER BY id",
        )
        .await?
        .collect()
        .await?;
    let mut actual = Vec::new();
    for batch in batches {
        let id = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let name = arrow::compute::cast(batch.column(1), &arrow::datatypes::DataType::Utf8)?;
        let name = name.as_any().downcast_ref::<StringArray>().unwrap();
        let nested_a = batch
            .column(2)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let nested_b = arrow::compute::cast(batch.column(3), &arrow::datatypes::DataType::Utf8)?;
        let nested_b = nested_b.as_any().downcast_ref::<StringArray>().unwrap();
        let part = batch
            .column(4)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        for row in 0..batch.num_rows() {
            actual.push(MappedRow {
                id: id.value(row),
                name: name.value(row).to_string(),
                nested_a: nested_a.value(row),
                nested_b: nested_b.value(row).to_string(),
                part: part.value(row),
            });
        }
    }
    assert_eq!(actual, expected);
    Ok(())
}
