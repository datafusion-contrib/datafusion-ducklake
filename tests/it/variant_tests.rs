#![cfg(feature = "metadata-duckdb")]

use std::path::Path;
use std::process::Command;
use std::sync::Arc;

use anyhow::Result;
use arrow::array::{Array, Int32Array};
use datafusion::prelude::*;
#[cfg(feature = "write-duckdb")]
use datafusion_ducklake::DuckdbMetadataWriter;
#[cfg(feature = "write-sqlite")]
use datafusion_ducklake::SqliteMetadataWriter;
use datafusion_ducklake::{DuckLakeCatalog, SqliteMetadataProvider, register_ducklake_functions};
use parquet::variant::{Variant, VariantArray, VariantType};
use tempfile::TempDir;

const VARIANT_STATS_ROWS: &str =
    "SELECT COUNT(*) FROM (SELECT data_file_id, table_id, column_id, variant_path,
     shredded_type, column_size_bytes, value_count, null_count, min_value, max_value,
     contains_nan, extra_stats FROM ducklake_file_variant_stats)";

fn create_variant_catalog(catalog_path: &Path, data_path: &Path) -> Result<()> {
    let version = Command::new("duckdb").arg("--version").output()?;
    assert!(version.status.success());
    assert!(String::from_utf8(version.stdout)?.starts_with("v1.5.5 "));

    let sql = format!(
        "LOAD ducklake;
         INSTALL sqlite;
         LOAD sqlite;
         ATTACH 'ducklake:sqlite:{}' AS lake (DATA_PATH '{}', DATA_INLINING_ROW_LIMIT 0);
         CREATE TABLE lake.main.variants (id INTEGER, v VARIANT);
         SET variant_minimum_shredding_size = -1;
         INSERT INTO lake.main.variants VALUES
             (-3, {{'a': 7, 'nested': {{'name': 'unshredded'}}}}::VARIANT),
             (-2, 'scalar'::VARIANT),
             (-1, NULL);
         SET variant_minimum_shredding_size = 1;
         INSERT INTO lake.main.variants
         SELECT i::INTEGER,
                {{'a': i::BIGINT, 'nested': {{'name': 'row-' || i::VARCHAR}}}}::VARIANT
         FROM range(100) AS rows(i);
         INSERT INTO lake.main.variants
         SELECT i::INTEGER,
                {{'a': i::BIGINT, 'nested': {{'name': 'row-' || i::VARCHAR}}}}::VARIANT
         FROM range(100, 200) AS rows(i);",
        catalog_path.display(),
        data_path.display()
    );
    let output = Command::new("duckdb")
        .args(["-unsigned", "-c", &sql])
        .output()?;
    assert!(
        output.status.success(),
        "DuckDB VARIANT fixture failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(())
}

#[cfg(feature = "write-duckdb")]
#[test]
fn duckdb_writer_initializes_variant_statistics_table() -> Result<()> {
    let temp = TempDir::new()?;
    let catalog_path = temp.path().join("variant.ducklake");
    let catalog_path = catalog_path.to_string_lossy().into_owned();
    let writer = DuckdbMetadataWriter::new_with_init(&catalog_path)?;
    drop(writer);
    let connection = duckdb::Connection::open(&catalog_path)?;
    let count = connection.query_row(VARIANT_STATS_ROWS, [], |row| row.get::<_, i64>(0))?;

    assert_eq!(count, 0);
    Ok(())
}

#[cfg(feature = "write-sqlite")]
#[tokio::test(flavor = "multi_thread")]
async fn sqlite_writer_initializes_variant_statistics_table() -> Result<()> {
    let temp = TempDir::new()?;
    let catalog_path = temp.path().join("variant.sqlite");
    let connection_string = format!("sqlite:{}?mode=rwc", catalog_path.display());
    let writer = SqliteMetadataWriter::new_with_init(&connection_string).await?;
    drop(writer);
    let pool = sqlx::SqlitePool::connect(&connection_string).await?;
    let (count,): (i64,) = sqlx::query_as(VARIANT_STATS_ROWS).fetch_one(&pool).await?;

    assert_eq!(count, 0);
    pool.close().await;
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires the DuckDB 1.5.5 CLI and DuckLake extension"]
async fn duckdb_variant_files_read_unshredded_and_shredded_values() -> Result<()> {
    let temp = TempDir::new()?;
    let catalog_path = temp.path().join("variant.ducklake");
    let data_path = temp.path().join("data");
    std::fs::create_dir(&data_path)?;
    create_variant_catalog(&catalog_path, &data_path)?;

    let conn_str = format!("sqlite:{}?mode=ro", catalog_path.display());
    let pool = sqlx::SqlitePool::connect(&conn_str).await?;
    let variant_stats: Vec<(String, String, i64, i64, String, String)> = sqlx::query_as(
        "SELECT variant_path, shredded_type, value_count, null_count, min_value, max_value
             FROM ducklake_file_variant_stats ORDER BY data_file_id, variant_path",
    )
    .fetch_all(&pool)
    .await?;
    assert_eq!(
        variant_stats,
        vec![
            (
                "\"a\"".to_string(),
                "int64".to_string(),
                100,
                0,
                "0".to_string(),
                "99".to_string(),
            ),
            (
                "\"nested\".\"name\"".to_string(),
                "varchar".to_string(),
                100,
                0,
                "row-0".to_string(),
                "row-99".to_string(),
            ),
            (
                "\"a\"".to_string(),
                "int64".to_string(),
                100,
                0,
                "100".to_string(),
                "199".to_string(),
            ),
            (
                "\"nested\".\"name\"".to_string(),
                "varchar".to_string(),
                100,
                0,
                "row-100".to_string(),
                "row-199".to_string(),
            ),
        ]
    );

    pool.close().await;
    let provider = SqliteMetadataProvider::new(&conn_str).await?;
    let catalog = DuckLakeCatalog::new(provider.clone())?;
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));
    register_ducklake_functions(&ctx, Arc::new(provider));
    let batches = ctx
        .sql("SELECT id, v FROM ducklake.main.variants ORDER BY id")
        .await?
        .collect()
        .await?;

    assert!(
        batches[0]
            .schema()
            .field(1)
            .has_valid_extension_type::<VariantType>()
    );
    let mut actual = Vec::new();
    for batch in &batches {
        let ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let variants = VariantArray::try_new(batch.column(1).as_ref())?;
        for row in 0..batch.num_rows() {
            if variants.is_null(row) {
                actual.push((ids.value(row), true));
                continue;
            }
            let value = variants.value(row);
            let is_null = value == Variant::Null;
            actual.push((ids.value(row), is_null));
            if is_null {
                continue;
            }
            match ids.value(row) {
                -3 => {
                    assert_eq!(value.get_object_field("a"), Some(Variant::from(7_i32)));
                    let nested = value.get_object_field("nested").unwrap();
                    assert_eq!(
                        nested.get_object_field("name"),
                        Some(Variant::from("unshredded"))
                    );
                },
                -2 => assert_eq!(value, Variant::from("scalar")),
                id => {
                    assert_eq!(
                        value.get_object_field("a"),
                        Some(Variant::from(i64::from(id)))
                    );
                    let nested = value.get_object_field("nested").unwrap();
                    let name = format!("row-{id}");
                    assert_eq!(
                        nested.get_object_field("name"),
                        Some(Variant::from(name.as_str()))
                    );
                },
            }
        }
    }

    let mut expected = vec![(-3, false), (-2, false), (-1, true)];
    expected.extend((0..200).map(|id| (id, false)));
    assert_eq!(actual, expected);

    let filtered = ctx
        .sql(
            "SELECT id FROM ducklake.main.variants
             WHERE ducklake_variant_get_int64(v, 'a') >= 150 ORDER BY id",
        )
        .await?;
    let plan = filtered.clone().create_physical_plan().await?;
    let display = datafusion::physical_plan::displayable(plan.as_ref())
        .indent(true)
        .to_string();
    assert_eq!(
        display.matches(".parquet").count(),
        2,
        "variant statistics must prune the disjoint shredded file:\n{display}"
    );
    let filtered = filtered.collect().await?;
    let actual = filtered
        .iter()
        .flat_map(|batch| {
            batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .values()
                .iter()
                .copied()
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    assert_eq!(actual, (150..200).collect::<Vec<_>>());

    Ok(())
}
