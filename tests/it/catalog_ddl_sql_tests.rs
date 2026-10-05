//! SQL catalog DDL tests for the standard SQLite metadata layout.

#![cfg(all(feature = "write-sqlite", feature = "metadata-sqlite"))]

use std::sync::Arc;

use datafusion::prelude::SessionContext;
use sqlx::sqlite::SqlitePool;
use sqlx::{AssertSqlSafe, Row};
use tempfile::TempDir;

use datafusion_ducklake::metadata_provider::MetadataProvider;
use datafusion_ducklake::{
    ColumnDef, DuckLakeCatalog, MetadataWriter, SqliteMetadataProvider, SqliteMetadataWriter,
    WriteMode, execute_ducklake_sql,
};

struct Harness {
    connection: String,
    writer: SqliteMetadataWriter,
    _temp: TempDir,
}

async fn setup() -> Harness {
    let temp = TempDir::new().unwrap();
    let catalog_path = temp.path().join("catalog.sqlite");
    let data_path = temp.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();
    let connection = format!("sqlite:{}?mode=rwc", catalog_path.display());
    let writer = SqliteMetadataWriter::new_with_init(&connection)
        .await
        .unwrap();
    writer.set_data_path(data_path.to_str().unwrap()).unwrap();
    Harness {
        connection,
        writer,
        _temp: temp,
    }
}

async fn execute(connection: &str, sql: &str) -> datafusion::error::Result<()> {
    let provider = SqliteMetadataProvider::new(connection).await.unwrap();
    let writer = SqliteMetadataWriter::new_with_init(connection)
        .await
        .unwrap();
    let catalog = DuckLakeCatalog::with_writer(Arc::new(provider), Arc::new(writer)).unwrap();
    let ctx = SessionContext::new();
    execute_ducklake_sql(&ctx, &catalog, sql).await?;
    Ok(())
}

async fn head(connection: &str) -> i64 {
    SqliteMetadataProvider::new(connection)
        .await
        .unwrap()
        .get_current_snapshot()
        .unwrap()
}

#[tokio::test(flavor = "multi_thread")]
async fn catalog_ddl_roundtrips_and_preserves_history() {
    let h = setup().await;
    execute(&h.connection, "CREATE SCHEMA analytics")
        .await
        .unwrap();
    let schema_snapshot = head(&h.connection).await;

    execute(&h.connection, "CREATE SCHEMA IF NOT EXISTS analytics")
        .await
        .unwrap();
    assert_eq!(head(&h.connection).await, schema_snapshot);

    let columns = vec![ColumnDef::new("id", "int32", false).unwrap()];
    let setup = h
        .writer
        .begin_write_transaction("analytics", "events", &columns, WriteMode::Replace)
        .unwrap();
    let committed = h
        .writer
        .publish_snapshot(
            setup.table_id,
            "analytics",
            "events",
            setup.snapshot_id,
            WriteMode::Replace,
            setup.base_snapshot_id,
            &columns,
            &setup.column_ids,
        )
        .unwrap();
    h.writer
        .set_partition_spec(
            committed.table_id,
            &[(
                "id".to_string(),
                datafusion_ducklake::partition::PartitionTransform::Identity,
            )],
        )
        .unwrap();

    let before_rename = head(&h.connection).await;
    let err = execute(&h.connection, "DROP SCHEMA analytics")
        .await
        .unwrap_err();
    assert!(err.to_string().contains("not empty"));
    assert_eq!(head(&h.connection).await, before_rename);

    execute(
        &h.connection,
        "ALTER TABLE analytics.events RENAME TO archived_events",
    )
    .await
    .unwrap();
    let rename_snapshot = head(&h.connection).await;
    assert_eq!(rename_snapshot, before_rename + 1);

    let provider = SqliteMetadataProvider::new(&h.connection).await.unwrap();
    let schema = provider
        .get_schema_by_name("analytics", rename_snapshot)
        .unwrap()
        .unwrap();
    assert!(
        provider
            .get_table_by_name(schema.schema_id, "events", rename_snapshot)
            .unwrap()
            .is_none()
    );
    let renamed = provider
        .get_table_by_name(schema.schema_id, "archived_events", rename_snapshot)
        .unwrap()
        .unwrap();
    assert_eq!(renamed.table_id, committed.table_id);
    assert_eq!(renamed.path, "events/");

    let historical_schema = provider
        .get_schema_by_name("analytics", before_rename)
        .unwrap()
        .unwrap();
    let historical = provider
        .get_table_by_name(historical_schema.schema_id, "events", before_rename)
        .unwrap()
        .unwrap();
    assert_eq!(historical.table_id, committed.table_id);
    assert!(
        provider
            .get_table_by_name(
                historical_schema.schema_id,
                "archived_events",
                before_rename,
            )
            .unwrap()
            .is_none()
    );

    let pool = SqlitePool::connect(&h.connection).await.unwrap();
    let column_id: i64 = sqlx::query_scalar(
        "SELECT column_id FROM ducklake_column
         WHERE table_id = ? AND end_snapshot IS NULL",
    )
    .bind(committed.table_id)
    .fetch_one(&pool)
    .await
    .unwrap();
    sqlx::query(
        "INSERT INTO ducklake_data_file
             (data_file_id, table_id, path, path_is_relative, file_size_bytes,
              record_count, begin_snapshot)
         VALUES (101, ?, 'events.parquet', 1, 10, 1, ?)",
    )
    .bind(committed.table_id)
    .bind(rename_snapshot)
    .execute(&pool)
    .await
    .unwrap();
    sqlx::query(
        "INSERT INTO ducklake_delete_file
             (delete_file_id, data_file_id, table_id, path, path_is_relative,
              file_size_bytes, delete_count, begin_snapshot)
         VALUES (201, 101, ?, 'events-delete.parquet', 1, 5, 1, ?)",
    )
    .bind(committed.table_id)
    .bind(rename_snapshot)
    .execute(&pool)
    .await
    .unwrap();
    sqlx::query("INSERT INTO ducklake_tag VALUES (?, ?, NULL, 'comment', 'table')")
        .bind(committed.table_id)
        .bind(rename_snapshot)
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query(
        "INSERT INTO ducklake_column_tag
         VALUES (?, ?, ?, NULL, 'comment', 'column')",
    )
    .bind(committed.table_id)
    .bind(column_id)
    .bind(rename_snapshot)
    .execute(&pool)
    .await
    .unwrap();

    execute(&h.connection, "DROP TABLE analytics.archived_events")
        .await
        .unwrap();
    let drop_snapshot = head(&h.connection).await;
    for table in [
        "ducklake_table",
        "ducklake_partition_info",
        "ducklake_column",
        "ducklake_column_tag",
        "ducklake_data_file",
        "ducklake_delete_file",
    ] {
        let sql = format!(
            "SELECT end_snapshot FROM {table} WHERE table_id = ? ORDER BY begin_snapshot DESC LIMIT 1"
        );
        let ended: i64 = sqlx::query(AssertSqlSafe(sql))
            .bind(committed.table_id)
            .fetch_one(&pool)
            .await
            .unwrap()
            .try_get(0)
            .unwrap();
        assert_eq!(ended, drop_snapshot, "{table}");
    }
    let tag_end: i64 =
        sqlx::query_scalar("SELECT end_snapshot FROM ducklake_tag WHERE object_id = ?")
            .bind(committed.table_id)
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(tag_end, drop_snapshot);

    execute(
        &h.connection,
        "DROP TABLE IF EXISTS analytics.archived_events",
    )
    .await
    .unwrap();
    assert_eq!(head(&h.connection).await, drop_snapshot);

    sqlx::query(
        "INSERT INTO ducklake_view (view_id, schema_id, begin_snapshot) VALUES (301, ?, ?)",
    )
    .bind(schema.schema_id)
    .bind(drop_snapshot)
    .execute(&pool)
    .await
    .unwrap();
    let err = execute(&h.connection, "DROP SCHEMA analytics")
        .await
        .unwrap_err();
    assert!(err.to_string().contains("not empty"));
    assert_eq!(head(&h.connection).await, drop_snapshot);
    sqlx::query("DELETE FROM ducklake_view")
        .execute(&pool)
        .await
        .unwrap();

    sqlx::query(
        "CREATE TABLE ducklake_macro (
             schema_id BIGINT, macro_id BIGINT, macro_name VARCHAR,
             begin_snapshot BIGINT, end_snapshot BIGINT
         )",
    )
    .execute(&pool)
    .await
    .unwrap();
    sqlx::query("INSERT INTO ducklake_macro VALUES (?, 401, 'active_macro', ?, NULL)")
        .bind(schema.schema_id)
        .bind(drop_snapshot)
        .execute(&pool)
        .await
        .unwrap();
    let err = execute(&h.connection, "DROP SCHEMA analytics")
        .await
        .unwrap_err();
    assert!(err.to_string().contains("not empty"));
    assert_eq!(head(&h.connection).await, drop_snapshot);
    sqlx::query("DELETE FROM ducklake_macro")
        .execute(&pool)
        .await
        .unwrap();

    execute(&h.connection, "DROP SCHEMA analytics")
        .await
        .unwrap();
    let schema_drop_snapshot = head(&h.connection).await;
    assert_eq!(schema_drop_snapshot, drop_snapshot + 1);
    execute(&h.connection, "DROP SCHEMA IF EXISTS analytics")
        .await
        .unwrap();
    assert_eq!(head(&h.connection).await, schema_drop_snapshot);
}

#[tokio::test(flavor = "multi_thread")]
async fn duckdb_reads_rust_catalog_ddl() {
    let Ok(cli) = std::env::var("DUCKDB_CLI") else {
        return;
    };
    let temp = TempDir::new().unwrap();
    let catalog_path = temp.path().join("official.ducklake");
    let data_path = temp.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();
    let seed = std::process::Command::new(&cli)
        .arg("-c")
        .arg(format!(
            "INSTALL ducklake; LOAD ducklake; \
             ATTACH 'ducklake:{}' AS lake (DATA_PATH '{}', META_TYPE 'sqlite'); \
             CREATE SCHEMA lake.analytics; \
             CREATE TABLE lake.analytics.events(id INTEGER); \
             INSERT INTO lake.analytics.events VALUES (7); \
             DETACH lake;",
            catalog_path.display(),
            data_path.display()
        ))
        .output()
        .unwrap();
    assert!(
        seed.status.success(),
        "DuckDB seed failed: {}",
        String::from_utf8_lossy(&seed.stderr)
    );

    let connection = format!("sqlite:{}?mode=rwc", catalog_path.display());
    execute(
        &connection,
        "ALTER TABLE analytics.events RENAME TO archived_events",
    )
    .await
    .unwrap();
    let read = std::process::Command::new(&cli)
        .args(["-csv", "-noheader", "-c"])
        .arg(format!(
            "INSTALL ducklake; LOAD ducklake; \
             ATTACH 'ducklake:{}' AS lake (META_TYPE 'sqlite'); \
             SELECT id FROM lake.analytics.archived_events;",
            catalog_path.display()
        ))
        .output()
        .unwrap();
    assert!(
        read.status.success(),
        "DuckDB renamed-table read failed: {}",
        String::from_utf8_lossy(&read.stderr)
    );
    assert_eq!(String::from_utf8(read.stdout).unwrap().trim(), "7");

    execute(&connection, "DROP TABLE analytics.archived_events")
        .await
        .unwrap();
    execute(&connection, "DROP SCHEMA analytics").await.unwrap();
    execute(&connection, "CREATE SCHEMA rust_schema")
        .await
        .unwrap();
    let dropped = std::process::Command::new(&cli)
        .args(["-csv", "-noheader", "-c"])
        .arg(format!(
            "INSTALL ducklake; LOAD ducklake; \
             ATTACH 'ducklake:{}' AS lake (META_TYPE 'sqlite'); \
             SELECT \
                 count(*) FILTER (WHERE schema_name = 'analytics'), \
                 count(*) FILTER (WHERE schema_name = 'rust_schema') \
             FROM duckdb_schemas() WHERE database_name = 'lake';",
            catalog_path.display()
        ))
        .output()
        .unwrap();
    assert!(
        dropped.status.success(),
        "DuckDB dropped-schema read failed: {}",
        String::from_utf8_lossy(&dropped.stderr)
    );
    assert_eq!(String::from_utf8(dropped.stdout).unwrap().trim(), "0,1");
}

#[tokio::test(flavor = "multi_thread")]
async fn legacy_table_primary_key_migrates_without_identity_loss() {
    let temp = TempDir::new().unwrap();
    let catalog_path = temp.path().join("legacy.sqlite");
    let connection = format!("sqlite:{}?mode=rwc", catalog_path.display());
    let pool = SqlitePool::connect(&connection).await.unwrap();
    sqlx::query(
        "CREATE TABLE ducklake_table (
             table_id INTEGER PRIMARY KEY,
             table_uuid VARCHAR,
             schema_id INTEGER NOT NULL,
             table_name VARCHAR NOT NULL,
             path VARCHAR NOT NULL DEFAULT '',
             path_is_relative BOOLEAN NOT NULL DEFAULT 1,
             begin_snapshot INTEGER NOT NULL,
             end_snapshot INTEGER
         )",
    )
    .execute(&pool)
    .await
    .unwrap();
    sqlx::query(
        "INSERT INTO ducklake_table
         VALUES (42, '12345678-1234-5678-1234-567812345678', 7, 'events',
                 'events', 1, 3, NULL)",
    )
    .execute(&pool)
    .await
    .unwrap();
    pool.close().await;

    SqliteMetadataWriter::new_with_init(&connection)
        .await
        .unwrap();
    let pool = SqlitePool::connect(&connection).await.unwrap();
    let row = sqlx::query(
        "SELECT table_id, table_uuid, schema_id, table_name, begin_snapshot
         FROM ducklake_table",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(row.try_get::<i64, _>("table_id").unwrap(), 42);
    assert_eq!(
        row.try_get::<String, _>("table_uuid").unwrap(),
        "12345678-1234-5678-1234-567812345678"
    );
    assert_eq!(row.try_get::<i64, _>("schema_id").unwrap(), 7);
    assert_eq!(row.try_get::<String, _>("table_name").unwrap(), "events");
    assert_eq!(row.try_get::<i64, _>("begin_snapshot").unwrap(), 3);
    let pk: i64 = sqlx::query_scalar(
        "SELECT pk FROM pragma_table_info('ducklake_table') WHERE name = 'table_id'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(pk, 0);
}
