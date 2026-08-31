#![cfg(feature = "metadata-duckdb")]

use std::path::Path;
use std::sync::Arc;

use arrow::array::{Decimal128Array, Int32Array, StringArray};
use datafusion::logical_expr::Operator;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_expr::expressions::{BinaryExpr, Column, Literal};
use datafusion::prelude::SessionContext;
use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeTable, DuckdbMetadataProvider, MetadataProvider,
    register_ducklake_functions,
};
use rstest::rstest;
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
    let function_provider: Arc<dyn MetadataProvider> =
        Arc::new(DuckdbMetadataProvider::new(catalog_path.to_string_lossy())?);
    register_ducklake_functions(&context, function_provider);

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

    let table_provider = context
        .catalog("ducklake")
        .unwrap()
        .schema("main")
        .unwrap()
        .table("mapped")
        .await?
        .unwrap();
    let table = (table_provider.as_ref() as &dyn std::any::Any)
        .downcast_ref::<DuckLakeTable>()
        .unwrap();
    let mapped_file = table
        .files()?
        .into_iter()
        .find(|file| file.file.path.contains("part=10"))
        .unwrap();
    let predicate: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
        Arc::new(Column::new("part", 3)),
        Operator::Eq,
        Arc::new(Literal::new(datafusion::common::ScalarValue::Int32(Some(
            10,
        )))),
    ));
    let positions = table
        .resolve_positions(&context.state(), &mapped_file.file, predicate)
        .await?;
    assert_eq!(positions, [0].into_iter().collect());

    let batches = context
        .sql(
            "SELECT id, part
             FROM ducklake_table_insertions('main.mapped', 0, 1000)
             ORDER BY id",
        )
        .await?
        .collect()
        .await?;
    let mut changes = Vec::new();
    for batch in batches {
        let id = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let part = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        for row in 0..batch.num_rows() {
            changes.push((id.value(row), part.value(row)));
        }
    }
    assert_eq!(changes, vec![(42, 9), (43, 10)]);

    Ok(())
}

#[rstest]
#[tokio::test]
async fn add_data_files_accepts_duckdb_numeric_hive_literals() -> anyhow::Result<()> {
    let temp = TempDir::new()?;
    let catalog_path = temp.path().join("numeric-mapping.ducklake");
    let data_path = temp.path().join("data");
    let partition_path = data_path.join("p=0x10").join("amount=1e2");
    std::fs::create_dir_all(&partition_path)?;
    let parquet_path = partition_path.join("mapped.parquet");

    let conn = duckdb::Connection::open_in_memory()?;
    conn.execute("INSTALL ducklake", [])?;
    conn.execute("LOAD ducklake", [])?;
    conn.execute(
        &format!(
            "ATTACH 'ducklake:{}' AS lake \
             (DATA_PATH '{}', DATA_INLINING_ROW_LIMIT 0)",
            catalog_path.display(),
            data_path.display()
        ),
        [],
    )?;
    conn.execute(
        "CREATE TABLE lake.numeric_mapped(
            id INTEGER,
            p INTEGER,
            amount DECIMAL(10,2)
        )",
        [],
    )?;
    conn.execute(
        &format!(
            "COPY (SELECT 1 AS id) TO '{}' (FORMAT PARQUET)",
            parquet_path.display()
        ),
        [],
    )?;
    conn.execute(
        &format!(
            "CALL ducklake_add_data_files(
                'lake', 'numeric_mapped', '{}/**/*.parquet', hive_partitioning => true
            )",
            data_path.display()
        ),
        [],
    )?;
    let expected: (i32, String) = conn.query_row(
        "SELECT p, CAST(amount AS VARCHAR) FROM lake.numeric_mapped",
        [],
        |row| Ok((row.get(0)?, row.get(1)?)),
    )?;
    assert_eq!(expected, (16, "100.00".to_string()));
    conn.execute("DETACH lake", [])?;

    let provider = DuckdbMetadataProvider::new(catalog_path.to_string_lossy())?;
    let catalog = DuckLakeCatalog::new(provider)?;
    let context = SessionContext::new();
    context.register_catalog("ducklake", Arc::new(catalog));
    let batches = context
        .sql("SELECT p, amount FROM ducklake.main.numeric_mapped")
        .await?
        .collect()
        .await?;
    assert_eq!(batches.len(), 1);
    let p = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    let amount = batches[0]
        .column(1)
        .as_any()
        .downcast_ref::<Decimal128Array>()
        .unwrap();
    assert_eq!(p.values(), &[16]);
    assert_eq!(amount.values(), &[10_000]);
    Ok(())
}
