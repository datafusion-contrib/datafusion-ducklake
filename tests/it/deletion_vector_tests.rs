#![cfg(all(feature = "metadata-sqlite", feature = "write-sqlite"))]

use std::path::Path;
use std::process::Command;
use std::sync::Arc;

use anyhow::Result;
use arrow::array::Int32Array;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::prelude::*;
use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeTableWriter, DuckLakeWriteOptions, MetadataWriter,
    SqliteMetadataProvider, SqliteMetadataWriter,
};
use object_store::local::LocalFileSystem;
use sqlx::Row;
use tempfile::TempDir;

fn create_deletion_vector_catalog(catalog_path: &Path, data_path: &Path) -> Result<()> {
    let version = Command::new("duckdb").arg("--version").output()?;
    assert!(version.status.success());
    assert!(String::from_utf8(version.stdout)?.starts_with("v1.5.5 "));

    let sql = format!(
        "LOAD ducklake;
         INSTALL sqlite;
         LOAD sqlite;
         ATTACH 'ducklake:sqlite:{}' AS lake
             (DATA_PATH '{}', DATA_INLINING_ROW_LIMIT 0);

         CREATE TABLE lake.main.bare AS
             SELECT i::INTEGER AS id FROM range(100) rows(i);
         CALL lake.set_option('write_deletion_vectors', 'true');
         DELETE FROM lake.main.bare WHERE id % 3 = 0;

         CREATE TABLE lake.main.mixed (id INTEGER);
         INSERT INTO lake.main.mixed SELECT i::INTEGER FROM range(100) rows(i);
         INSERT INTO lake.main.mixed SELECT i::INTEGER FROM range(100, 200) rows(i);
         DELETE FROM lake.main.mixed WHERE id < 10;
         CALL lake.set_option('write_deletion_vectors', 'false');
         DELETE FROM lake.main.mixed WHERE id >= 100 AND id < 110;",
        catalog_path.display(),
        data_path.display()
    );
    let output = Command::new("duckdb")
        .args(["-unsigned", "-c", &sql])
        .output()?;
    assert!(
        output.status.success(),
        "DuckDB deletion-vector fixture failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(())
}

async fn query_ids(ctx: &SessionContext, sql: &str) -> Result<Vec<i32>> {
    let batches = ctx.sql(sql).await?.collect().await?;
    Ok(batches
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
        .collect())
}

async fn seed_crate_catalog(temp: &TempDir) -> Result<String> {
    let catalog_path = temp.path().join("crate.ducklake");
    let data_path = temp.path().join("crate-data");
    std::fs::create_dir(&data_path)?;
    let conn_str = format!("sqlite:{}?mode=rwc", catalog_path.display());
    let writer = Arc::new(SqliteMetadataWriter::new_with_init(&conn_str).await?);
    writer.set_data_path(data_path.to_str().unwrap())?;
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    writer.set_global_setting("data_inlining_row_limit", "0")?;
    let first = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
    )?;
    let second = RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![4, 5]))])?;
    let writer = DuckLakeTableWriter::new(writer, Arc::new(LocalFileSystem::new()))?;
    writer.write_table("main", "t", &[first]).await?;
    writer.append_table("main", "t", &[second]).await?;
    Ok(conn_str)
}

async fn crate_write_context(
    conn_str: &str,
    write_deletion_vectors: bool,
) -> Result<SessionContext> {
    let provider = SqliteMetadataProvider::new(conn_str).await?;
    let writer = SqliteMetadataWriter::new(conn_str).await?;
    let options =
        DuckLakeWriteOptions::default().with_write_deletion_vectors(write_deletion_vectors);
    let catalog = DuckLakeCatalog::with_writer(Arc::new(provider), Arc::new(writer))?
        .with_write_options(options);
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));
    Ok(ctx)
}

async fn delete_id(ctx: &SessionContext, id: i32) -> Result<()> {
    let batches = ctx
        .sql(&format!("DELETE FROM ducklake.main.t WHERE id = {id}"))
        .await?
        .collect()
        .await?;
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 1);
    Ok(())
}

async fn write_mixed_crate_history(temp: &TempDir) -> Result<String> {
    let conn_str = seed_crate_catalog(temp).await?;
    delete_id(&crate_write_context(&conn_str, false).await?, 2).await?;
    delete_id(&crate_write_context(&conn_str, true).await?, 4).await?;
    Ok(conn_str)
}

#[tokio::test(flavor = "multi_thread")]
async fn crate_write_option_defaults_to_parquet_and_opt_in_writes_puffin() -> Result<()> {
    let temp = TempDir::new()?;
    let conn_str = write_mixed_crate_history(&temp).await?;

    let pool = sqlx::SqlitePool::connect(&conn_str).await?;
    let rows = sqlx::query(
        "SELECT format, end_snapshot IS NULL AS is_live, delete_count, footer_size
         FROM ducklake_delete_file ORDER BY begin_snapshot",
    )
    .fetch_all(&pool)
    .await?;
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].try_get::<String, _>("format")?, "parquet");
    assert!(rows[0].try_get::<bool, _>("is_live")?);
    assert_eq!(rows[0].try_get::<i64, _>("delete_count")?, 1);
    assert!(rows[0].try_get::<Option<i64>, _>("footer_size")?.is_some());
    assert_eq!(rows[1].try_get::<String, _>("format")?, "puffin");
    assert!(rows[1].try_get::<bool, _>("is_live")?);
    assert_eq!(rows[1].try_get::<i64, _>("delete_count")?, 1);
    assert_eq!(rows[1].try_get::<Option<i64>, _>("footer_size")?, None);
    pool.close().await;

    let provider = SqliteMetadataProvider::new(&conn_str).await?;
    let catalog = DuckLakeCatalog::new(provider)?;
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));
    assert_eq!(
        query_ids(&ctx, "SELECT id FROM ducklake.main.t ORDER BY id").await?,
        vec![1, 3, 5]
    );
    datafusion_ducklake::register_ducklake_functions(
        &ctx,
        Arc::new(SqliteMetadataProvider::new(&conn_str).await?),
    );
    assert_eq!(
        query_ids(
            &ctx,
            "SELECT id FROM ducklake_table_deletions('main.t', 1, 4) ORDER BY id",
        )
        .await?,
        vec![2, 4],
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires the DuckDB 1.5.5 CLI and DuckLake extension"]
async fn duckdb_deletion_vectors_apply_bare_and_mixed_delete_formats() -> Result<()> {
    let temp = TempDir::new()?;
    let catalog_path = temp.path().join("deletion-vectors.ducklake");
    let data_path = temp.path().join("data");
    std::fs::create_dir(&data_path)?;
    create_deletion_vector_catalog(&catalog_path, &data_path)?;

    let conn_str = format!("sqlite:{}?mode=ro", catalog_path.display());
    let pool = sqlx::SqlitePool::connect(&conn_str).await?;
    let formats: Vec<(String, i64, i64)> = sqlx::query_as(
        "SELECT format, delete_count, footer_size
         FROM ducklake_delete_file WHERE end_snapshot IS NULL
         ORDER BY format, delete_count",
    )
    .fetch_all(&pool)
    .await?;
    assert_eq!(
        formats,
        vec![
            ("parquet".to_string(), 10, 787),
            ("puffin".to_string(), 10, 0),
            ("puffin".to_string(), 34, 0),
        ]
    );
    pool.close().await;

    let provider = SqliteMetadataProvider::new(&conn_str).await?;
    let catalog = DuckLakeCatalog::new(provider)?;
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));

    assert_eq!(
        query_ids(&ctx, "SELECT id FROM ducklake.main.bare ORDER BY id").await?,
        (0..100).filter(|id| id % 3 != 0).collect::<Vec<_>>()
    );
    assert_eq!(
        query_ids(&ctx, "SELECT id FROM ducklake.main.mixed ORDER BY id").await?,
        (0..200)
            .filter(|id| !(*id < 10 || (100..110).contains(id)))
            .collect::<Vec<_>>()
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires the DuckDB 1.5.5 CLI and DuckLake extension"]
async fn duckdb_reads_crate_written_deletion_vector() -> Result<()> {
    let temp = TempDir::new()?;
    let catalog_path = temp.path().join("crate.ducklake");
    let conn_str = seed_crate_catalog(&temp).await?;
    SqliteMetadataWriter::new(&conn_str)
        .await?
        .migrate_to_official_layout()?;
    delete_id(&crate_write_context(&conn_str, true).await?, 2).await?;
    let pool = sqlx::SqlitePool::connect(&conn_str).await?;
    let live_format: String =
        sqlx::query_scalar("SELECT format FROM ducklake_delete_file WHERE end_snapshot IS NULL")
            .fetch_one(&pool)
            .await?;
    assert_eq!(live_format, "puffin");
    pool.close().await;

    let sql = format!(
        "LOAD ducklake;
         INSTALL sqlite;
         LOAD sqlite;
         ATTACH 'ducklake:sqlite:{}' AS lake;
         SELECT id FROM lake.main.t ORDER BY id;",
        catalog_path.display()
    );
    let output = Command::new("duckdb")
        .args(["-unsigned", "-csv", "-noheader", "-c", &sql])
        .output()?;
    assert!(
        output.status.success(),
        "DuckDB failed to read the crate deletion vector: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let ids = String::from_utf8(output.stdout)?
        .lines()
        .map(str::parse::<i32>)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    assert_eq!(ids, vec![1, 3, 4, 5]);
    Ok(())
}
