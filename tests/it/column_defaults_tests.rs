//! Differential and write-path tests for DuckLake column defaults.

#![cfg(all(feature = "metadata-duckdb", feature = "write-sqlite"))]

use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use arrow::array::{Array, Int32Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::prelude::SessionContext;
use datafusion_ducklake::metadata_provider::MetadataProvider;
use datafusion_ducklake::{
    ColumnDef, DuckLakeCatalog, DuckLakeTableWriter, DuckdbMetadataProvider, MetadataWriter,
    SqliteMetadataProvider, SqliteMetadataWriter, WriteMode,
};
use object_store::local::LocalFileSystem;
use tempfile::TempDir;

use super::common::ensure_ducklake_installed;

fn create_official_default_catalog(path: &Path) -> Result<Vec<(i32, i32)>> {
    let conn = duckdb::Connection::open_in_memory()?;
    ensure_ducklake_installed();
    conn.execute("LOAD ducklake", [])?;
    conn.execute(
        &format!("ATTACH 'ducklake:{}' AS official", path.display()),
        [],
    )?;
    conn.execute("CALL official.set_option('data_inlining_row_limit', 0)", [])?;
    conn.execute("CREATE TABLE official.items (id INTEGER)", [])?;
    conn.execute("INSERT INTO official.items VALUES (1), (2)", [])?;
    conn.execute(
        "ALTER TABLE official.items ADD COLUMN priority INTEGER DEFAULT 7",
        [],
    )?;
    conn.execute("INSERT INTO official.items (id) VALUES (3)", [])?;

    let mut statement = conn.prepare("SELECT id, priority FROM official.items ORDER BY id")?;
    let rows = statement
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(rows)
}

fn collect_rows(batches: &[RecordBatch]) -> Vec<(i32, String)> {
    let mut rows = Vec::new();
    for batch in batches {
        let ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let labels = arrow::compute::cast(batch.column(1), &DataType::Utf8).unwrap();
        let labels = labels.as_any().downcast_ref::<StringArray>().unwrap();
        rows.extend((0..batch.num_rows()).map(|index| {
            assert!(ids.is_valid(index));
            assert!(labels.is_valid(index));
            (ids.value(index), labels.value(index).to_string())
        }));
    }
    rows
}

#[tokio::test]
async fn duckdb_added_default_matches_official_rows() -> Result<()> {
    let temp = TempDir::new()?;
    let catalog_path = temp.path().join("column_defaults.ducklake");
    let official_rows = create_official_default_catalog(&catalog_path)?;

    let provider = DuckdbMetadataProvider::new(catalog_path.to_string_lossy())?;
    let catalog = DuckLakeCatalog::new(provider)?;
    let context = SessionContext::new();
    context.register_catalog("ducklake", Arc::new(catalog));

    let batches = context
        .sql("SELECT id, priority FROM ducklake.main.items ORDER BY id")
        .await?
        .collect()
        .await?;
    let datafusion_rows = batches
        .iter()
        .flat_map(|batch| {
            let ids = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            let priorities = batch
                .column(1)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            (0..batch.num_rows()).map(|index| (ids.value(index), priorities.value(index)))
        })
        .collect::<Vec<_>>();

    assert_eq!(official_rows, vec![(1, 7), (2, 7), (3, 7)]);
    assert_eq!(datafusion_rows, official_rows);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn rust_added_default_fills_old_and_omitted_rows() -> Result<()> {
    let temp = TempDir::new()?;
    let db_path = temp.path().join("column_defaults.sqlite");
    let data_path = temp.path().join("data");
    std::fs::create_dir_all(&data_path)?;
    let connection = format!("sqlite:{}?mode=rwc", db_path.display());

    let writer = Arc::new(SqliteMetadataWriter::new_with_init(&connection).await?);
    writer.set_data_path(data_path.to_str().unwrap())?;
    let object_store = Arc::new(LocalFileSystem::new());
    let table_writer = DuckLakeTableWriter::new(writer.clone(), object_store)?;
    let initial_schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let initial_batch =
        RecordBatch::try_new(initial_schema, vec![Arc::new(Int32Array::from(vec![1]))])?;
    let result = table_writer
        .write_table("main", "items", &[initial_batch])
        .await?;

    let columns = vec![
        ColumnDef::from_arrow("id", &DataType::Int32, false)?,
        ColumnDef::from_arrow("label", &DataType::Utf8, true)?.with_default("new")?,
    ];
    let setup = writer.begin_write_transaction("main", "items", &columns, WriteMode::Append)?;
    let committed = writer.publish_snapshot(
        result.table_id,
        "main",
        "items",
        setup.snapshot_id,
        WriteMode::Append,
        setup.base_snapshot_id,
        &columns,
        &setup.column_ids,
    )?;

    let provider = SqliteMetadataProvider::new(&connection).await?;
    let snapshot = provider.get_current_snapshot()?;
    let stored = provider.get_table_structure(committed.table_id, snapshot)?;
    assert_eq!(stored.len(), 2);
    assert_eq!(stored[1].column_name, "label");
    assert_eq!(stored[1].initial_default.as_deref(), Some("new"));
    assert_eq!(stored[1].default_value.as_deref(), Some("new"));
    assert_eq!(stored[1].default_value_type.as_deref(), Some("literal"));
    assert_eq!(stored[1].default_value_dialect.as_deref(), Some("duckdb"));
    let listed = provider.list_all_columns(snapshot)?;
    let listed_label = listed
        .iter()
        .find(|entry| entry.table_name == "items" && entry.column.column_name == "label")
        .unwrap();
    assert_eq!(listed_label.column.initial_default.as_deref(), Some("new"));
    assert_eq!(listed_label.column.default_value.as_deref(), Some("new"));
    assert_eq!(
        listed_label.column.default_value_type.as_deref(),
        Some("literal")
    );
    assert_eq!(
        listed_label.column.default_value_dialect.as_deref(),
        Some("duckdb")
    );

    let provider = SqliteMetadataProvider::new(&connection).await?;
    let writer = SqliteMetadataWriter::new(&connection).await?;
    let catalog = DuckLakeCatalog::with_writer(Arc::new(provider), Arc::new(writer))?;
    let context = SessionContext::new();
    context.register_catalog("ducklake", Arc::new(catalog));
    context
        .sql("INSERT INTO ducklake.main.items (id) VALUES (2)")
        .await?
        .collect()
        .await?;

    let provider = SqliteMetadataProvider::new(&connection).await?;
    let catalog = DuckLakeCatalog::new(provider)?;
    let context = SessionContext::new();
    context.register_catalog("ducklake", Arc::new(catalog));
    let batches = context
        .sql("SELECT id, label FROM ducklake.main.items ORDER BY id")
        .await?
        .collect()
        .await?;

    assert_eq!(
        collect_rows(&batches),
        vec![(1, "new".to_string()), (2, "new".to_string())]
    );
    Ok(())
}

/// A rowid-projected scan is a *positional* path, and since positional paths
/// gained filter pushdown a predicate on a column the file predates now reaches
/// the parquet reader. There is no such column in the file, so the expression
/// adapter substitutes the DuckLake `initial_default` before pruning — the same
/// value the read path materializes. Getting that wrong prunes away rows that
/// do match.
///
/// The catalog here is written by **official DuckDB DuckLake**, so this is also
/// a differential check: our read of an official catalog must agree with what
/// official itself returns.
#[tokio::test]
async fn positional_scan_pushes_a_predicate_on_a_defaulted_column() -> Result<()> {
    let temp = TempDir::new()?;
    let catalog_path = temp.path().join("column_defaults.ducklake");
    // items(id) rows 1,2 predate `priority INTEGER DEFAULT 7`; row 3 follows it.
    let official_rows = create_official_default_catalog(&catalog_path)?;
    assert_eq!(official_rows, vec![(1, 7), (2, 7), (3, 7)]);

    let provider = DuckdbMetadataProvider::new(catalog_path.to_string_lossy())?;
    let catalog = DuckLakeCatalog::new(provider)?.with_row_lineage(true);
    let context = SessionContext::new();
    context.register_catalog("ducklake", Arc::new(catalog));

    let ids_for = |sql: &'static str| {
        let context = context.clone();
        async move {
            let batches = context.sql(sql).await?.collect().await?;
            let mut ids: Vec<i32> = Vec::new();
            for batch in &batches {
                let column = batch
                    .column(batch.schema().index_of("id")?)
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .expect("id is Int32")
                    .clone();
                ids.extend((0..batch.num_rows()).map(|i| column.value(i)));
            }
            ids.sort_unstable();
            anyhow::Ok(ids)
        }
    };

    // Matches the default: every row qualifies, including the two in the file
    // written before the column existed.
    assert_eq!(
        ids_for("SELECT rowid, id FROM ducklake.main.items WHERE priority = 7").await?,
        vec![1, 2, 3],
        "rows in a file predating the column must match its default"
    );

    // Does not match: pruning the old file away is correct here, and is exactly
    // what a pushed `7 = 9` lets the reader do.
    assert_eq!(
        ids_for("SELECT rowid, id FROM ducklake.main.items WHERE priority = 9").await?,
        Vec::<i32>::new()
    );

    // NULL semantics are unaffected.
    assert_eq!(
        ids_for("SELECT rowid, id FROM ducklake.main.items WHERE priority IS NULL").await?,
        Vec::<i32>::new()
    );
    Ok(())
}
