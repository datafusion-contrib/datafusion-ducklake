#![cfg(feature = "metadata-duckdb")]

use std::path::Path;
use std::sync::Arc;

use anyhow::Context;
use arrow::util::display::{ArrayFormatter, FormatOptions};
use datafusion::prelude::SessionContext;
#[cfg(feature = "write-duckdb")]
use datafusion_ducklake::DuckdbMetadataWriter;
use datafusion_ducklake::{
    DuckLakeCatalog, DuckdbMetadataProvider, MetadataProvider, register_ducklake_macros,
};
#[cfg(feature = "write-sqlite")]
use datafusion_ducklake::{SqliteMetadataProvider, SqliteMetadataWriter};
use tempfile::TempDir;

fn create_macro_catalog(catalog_path: &Path, data_path: &Path) -> anyhow::Result<()> {
    let connection = duckdb::Connection::open_in_memory()?;
    connection.execute("INSTALL ducklake", [])?;
    connection.execute("LOAD ducklake", [])?;
    connection.execute(
        &format!(
            "ATTACH 'ducklake:{}' AS lake (DATA_PATH '{}', DATA_INLINING_ROW_LIMIT 0)",
            catalog_path.display(),
            data_path.display()
        ),
        [],
    )?;
    connection.execute("USE lake", [])?;
    connection.execute("CREATE TABLE items(id INTEGER, name VARCHAR)", [])?;
    connection.execute(
        "INSERT INTO items VALUES (1, 'one'), (2, 'two'), (3, 'three'), (4, 'four')",
        [],
    )?;
    connection
        .execute(
            "CREATE MACRO add_default(value INTEGER, increment := 5) AS value + increment",
            [],
        )
        .context("create scalar macro")?;
    connection
        .execute(
            "CREATE MACRO filtered(minimum INTEGER := 3) AS TABLE
         SELECT id, name FROM items WHERE id >= minimum",
            [],
        )
        .context("create filtered table macro")?;
    connection
        .execute(
            "CREATE MACRO duckdb_only() AS TABLE SELECT COLUMNS('id') FROM items",
            [],
        )
        .context("create DuckDB-only table macro")?;
    connection.execute("CREATE SCHEMA analytics", [])?;
    connection
        .execute(
            "CREATE MACRO analytics.schema_scalar(value INTEGER) AS value + 1",
            [],
        )
        .context("create schema-qualified scalar macro")?;
    Ok(())
}

fn duckdb_rows(
    catalog_path: &Path,
    data_path: &Path,
    sql: &str,
) -> anyhow::Result<Vec<Vec<String>>> {
    let connection = duckdb::Connection::open_in_memory()?;
    connection.execute("LOAD ducklake", [])?;
    connection.execute(
        &format!(
            "ATTACH 'ducklake:{}' AS lake (DATA_PATH '{}')",
            catalog_path.display(),
            data_path.display()
        ),
        [],
    )?;
    connection.execute("USE lake", [])?;
    let mut statement = connection.prepare(sql)?;
    let rows = statement
        .query_map([], |row| {
            (0..row.as_ref().column_count())
                .map(|column| row.get::<_, String>(column))
                .collect::<Result<Vec<_>, _>>()
        })?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(rows)
}

async fn datafusion_rows(context: &SessionContext, sql: &str) -> anyhow::Result<Vec<Vec<String>>> {
    let batches = context.sql(sql).await?.collect().await?;
    let options = FormatOptions::default();
    let mut rows = Vec::new();
    for batch in batches {
        let formatters = batch
            .columns()
            .iter()
            .map(|array| ArrayFormatter::try_new(array.as_ref(), &options))
            .collect::<Result<Vec<_>, _>>()?;
        for row in 0..batch.num_rows() {
            rows.push(
                formatters
                    .iter()
                    .map(|formatter| formatter.value(row).to_string())
                    .collect(),
            );
        }
    }
    Ok(rows)
}

#[tokio::test]
async fn duckdb_macros_match_scalar_and_table_oracles() -> anyhow::Result<()> {
    let temp = TempDir::new()?;
    let catalog_path = temp.path().join("macros.ducklake");
    let data_path = temp.path().join("data");
    create_macro_catalog(&catalog_path, &data_path)?;

    let provider: Arc<dyn MetadataProvider> =
        Arc::new(DuckdbMetadataProvider::new(catalog_path.to_string_lossy())?);
    let snapshot_id = provider.get_current_snapshot()?;
    let macros = provider.list_macros(0, snapshot_id)?;
    let add_default = macros
        .iter()
        .find(|item| item.macro_name == "add_default")
        .unwrap();
    assert_eq!(add_default.implementations.len(), 1);
    assert_eq!(add_default.implementations[0].parameters.len(), 2);
    assert_eq!(
        add_default.implementations[0].parameters[0].parameter_type,
        "int32"
    );
    assert_eq!(
        add_default.implementations[0].parameters[0].default_value,
        None
    );
    assert_eq!(
        add_default.implementations[0].parameters[1]
            .default_value
            .as_deref(),
        Some("5")
    );

    let catalog = DuckLakeCatalog::with_snapshot(provider.clone(), snapshot_id)?;
    let context = SessionContext::new();
    context.register_catalog("ducklake", Arc::new(catalog));
    register_ducklake_macros(&context, "ducklake", provider).await?;

    for (duckdb_sql, datafusion_sql) in [
        (
            "SELECT CAST(add_default(7) AS VARCHAR)",
            "SELECT CAST(add_default(7) AS VARCHAR)",
        ),
        (
            "SELECT CAST(add_default(7, 2) AS VARCHAR)",
            "SELECT CAST(add_default(7, 2) AS VARCHAR)",
        ),
        (
            "SELECT CAST(analytics.schema_scalar(7) AS VARCHAR)",
            "SELECT CAST(analytics.schema_scalar(7) AS VARCHAR)",
        ),
        (
            "SELECT CAST(id AS VARCHAR) AS id_text, name FROM filtered() ORDER BY id",
            "SELECT CAST(id AS VARCHAR) AS id_text, name FROM filtered() ORDER BY id",
        ),
        (
            "SELECT CAST(id AS VARCHAR) AS id_text, name FROM filtered(2) ORDER BY id",
            "SELECT CAST(id AS VARCHAR) AS id_text, name FROM filtered(2) ORDER BY id",
        ),
    ] {
        assert_eq!(
            datafusion_rows(&context, datafusion_sql).await?,
            duckdb_rows(&catalog_path, &data_path, duckdb_sql)?
        );
    }

    let error = context
        .sql("SELECT * FROM duckdb_only()")
        .await
        .unwrap_err()
        .to_string();
    assert!(error.contains("DuckLake macro 'duckdb_only'"), "{error}");
    assert!(error.contains("dialect 'duckdb'"), "{error}");
    Ok(())
}

#[cfg(feature = "write-duckdb")]
#[tokio::test]
async fn duckdb_writer_initializes_macro_tables() -> anyhow::Result<()> {
    let temp = TempDir::new()?;
    let catalog_path = temp.path().join("macros.ducklake");
    drop(DuckdbMetadataWriter::new_with_init(
        catalog_path.to_string_lossy(),
    )?);

    let provider = DuckdbMetadataProvider::new(catalog_path.to_string_lossy())?;
    assert!(provider.list_macros(0, 0)?.is_empty());
    Ok(())
}

#[cfg(feature = "write-sqlite")]
#[tokio::test(flavor = "multi_thread")]
async fn sqlite_writer_initializes_macro_tables() -> anyhow::Result<()> {
    let temp = TempDir::new()?;
    let catalog_path = temp.path().join("macros.sqlite");
    let connection_string = format!("sqlite:{}?mode=rwc", catalog_path.display());
    drop(SqliteMetadataWriter::new_with_init(&connection_string).await?);

    let provider = SqliteMetadataProvider::new(&connection_string).await?;
    assert!(provider.list_macros(0, 0)?.is_empty());
    Ok(())
}
