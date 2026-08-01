//! SQL column DDL over the writable DuckLake catalog.

#![cfg(all(feature = "write-sqlite", feature = "metadata-sqlite"))]

use std::sync::Arc;

use arrow::array::{Int32Array, RecordBatch};
use arrow::datatypes::{DataType, Field};
use datafusion::prelude::SessionContext;
use object_store::local::LocalFileSystem;
use sqlx::Row;
use sqlx::sqlite::SqlitePool;
use tempfile::TempDir;

use datafusion_ducklake::{
    ColumnDef, DuckLakeCatalog, DuckLakeTableWriter, MetadataWriter, SqliteMetadataProvider,
    SqliteMetadataWriter, WriteMode, execute_ducklake_sql,
};

struct Env {
    catalog_path: std::path::PathBuf,
    conn_str: String,
    official_oracle: bool,
    _temp: TempDir,
}

async fn setup() -> Env {
    let temp = TempDir::new().unwrap();
    let catalog_path = temp.path().join("catalog.ducklake");
    let data_path = temp.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();
    let conn_str = format!("sqlite:{}?mode=rwc", catalog_path.display());
    let official_oracle = if let Ok(cli) = std::env::var("DUCKDB_CLI") {
        let output = std::process::Command::new(cli)
            .arg("-c")
            .arg(format!(
                "INSTALL ducklake; LOAD ducklake; \
                 ATTACH 'ducklake:{}' AS seed (DATA_PATH '{}', META_TYPE 'sqlite'); \
                 CREATE TABLE seed.events(id INTEGER); \
                 INSERT INTO seed.events VALUES (7); \
                 CREATE TABLE seed.nested(profile STRUCT(city INTEGER)); \
                 DETACH seed;",
                catalog_path.display(),
                data_path.display()
            ))
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "DuckDB seed failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        true
    } else {
        let writer = SqliteMetadataWriter::new_with_init(&conn_str)
            .await
            .unwrap();
        writer.set_data_path(data_path.to_str().unwrap()).unwrap();
        let event_schema = Arc::new(arrow::datatypes::Schema::new(vec![Field::new(
            "id",
            DataType::Int32,
            true,
        )]));
        let event_batch = RecordBatch::try_new(
            event_schema,
            vec![Arc::new(Int32Array::from(vec![Some(7)]))],
        )
        .unwrap();
        DuckLakeTableWriter::new(Arc::new(writer.clone()), Arc::new(LocalFileSystem::new()))
            .unwrap()
            .write_table("main", "events", &[event_batch])
            .await
            .unwrap();
        let profile =
            DataType::Struct(vec![Arc::new(Field::new("city", DataType::Int32, true))].into());
        let nested_columns = vec![ColumnDef::from_arrow("profile", &profile, true).unwrap()];
        let nested = writer
            .begin_write_transaction("main", "nested", &nested_columns, WriteMode::Replace)
            .unwrap();
        writer
            .publish_snapshot(
                nested.table_id,
                "main",
                "nested",
                nested.snapshot_id,
                WriteMode::Replace,
                nested.base_snapshot_id,
                &nested_columns,
                &nested.field_ids,
            )
            .unwrap();
        false
    };

    Env {
        catalog_path,
        conn_str,
        official_oracle,
        _temp: temp,
    }
}

async fn writable_catalog(conn_str: &str) -> (SessionContext, Arc<DuckLakeCatalog>) {
    let writer = SqliteMetadataWriter::new_with_init(conn_str).await.unwrap();
    let provider = SqliteMetadataProvider::new(conn_str).await.unwrap();
    let catalog =
        Arc::new(DuckLakeCatalog::with_writer(Arc::new(provider), Arc::new(writer)).unwrap());
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::clone(&catalog) as _);
    (ctx, catalog)
}

async fn read_context(conn_str: &str) -> SessionContext {
    let provider = SqliteMetadataProvider::new(conn_str).await.unwrap();
    let catalog = DuckLakeCatalog::new(provider).unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));
    ctx
}

#[tokio::test(flavor = "multi_thread")]
async fn column_ddl_roundtrips_and_preserves_field_ids() {
    let env = setup().await;
    let pool = SqlitePool::connect(&env.conn_str).await.unwrap();
    let initial = sqlx::query(
        "SELECT t.table_name, c.column_id, c.column_name
         FROM ducklake_column c JOIN ducklake_table t ON t.table_id = c.table_id
         WHERE c.end_snapshot IS NULL ORDER BY t.table_name, c.column_order",
    )
    .fetch_all(&pool)
    .await
    .unwrap();
    let event_id: i64 = initial
        .iter()
        .find(|row| row.get::<String, _>("table_name") == "events")
        .unwrap()
        .get("column_id");
    let profile_id: i64 = initial
        .iter()
        .find(|row| row.get::<String, _>("column_name") == "profile")
        .unwrap()
        .get("column_id");
    let city_id: i64 = initial
        .iter()
        .find(|row| row.get::<String, _>("column_name") == "city")
        .unwrap()
        .get("column_id");

    let (ctx, catalog) = writable_catalog(&env.conn_str).await;
    for sql in [
        "ALTER TABLE ducklake.main.events ADD COLUMN note VARCHAR",
        "ALTER TABLE ducklake.main.events RENAME COLUMN id TO event_id",
        "ALTER TABLE ducklake.main.events ALTER COLUMN event_id SET DATA TYPE BIGINT",
        "ALTER TABLE ducklake.main.nested ADD COLUMN profile.zip INTEGER",
        "ALTER TABLE ducklake.main.nested ALTER COLUMN profile.zip SET TYPE BIGINT",
        "ALTER TABLE ducklake.main.nested ADD COLUMN profile.address STRUCT(street VARCHAR)",
        "ALTER TABLE ducklake.main.nested DROP COLUMN profile.city",
    ] {
        execute_ducklake_sql(&ctx, &catalog, sql)
            .await
            .unwrap_or_else(|e| panic!("{sql}: {e:?}"));
    }

    let before_rejected: i64 =
        sqlx::query_scalar("SELECT COALESCE(MAX(snapshot_id), 0) FROM ducklake_snapshot")
            .fetch_one(&pool)
            .await
            .unwrap();
    let error = execute_ducklake_sql(
        &ctx,
        &catalog,
        "ALTER TABLE ducklake.main.events ALTER COLUMN event_id TYPE INTEGER",
    )
    .await
    .unwrap_err();
    assert!(error.to_string().contains("Unsupported type change"));
    let after_rejected: i64 =
        sqlx::query_scalar("SELECT COALESCE(MAX(snapshot_id), 0) FROM ducklake_snapshot")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(after_rejected, before_rejected);

    let live = sqlx::query(
        "SELECT t.table_name, c.column_id, c.column_name, c.column_type, c.parent_column
         FROM ducklake_column c JOIN ducklake_table t ON t.table_id = c.table_id
         WHERE c.end_snapshot IS NULL ORDER BY t.table_name, c.column_order",
    )
    .fetch_all(&pool)
    .await
    .unwrap();
    let renamed = live
        .iter()
        .find(|row| row.get::<String, _>("column_name") == "event_id")
        .unwrap();
    assert_eq!(renamed.get::<i64, _>("column_id"), event_id);
    assert_eq!(renamed.get::<String, _>("column_type"), "int64");
    let profile = live
        .iter()
        .find(|row| row.get::<String, _>("column_name") == "profile")
        .unwrap();
    assert_eq!(profile.get::<i64, _>("column_id"), profile_id);
    let zip = live
        .iter()
        .find(|row| row.get::<String, _>("column_name") == "zip")
        .unwrap();
    assert_eq!(zip.get::<Option<i64>, _>("parent_column"), Some(profile_id));
    assert!(
        !live
            .iter()
            .any(|row| row.get::<i64, _>("column_id") == city_id)
    );

    let ctx = read_context(&env.conn_str).await;
    if !env.official_oracle {
        let batches = ctx
            .sql("SELECT event_id, note FROM ducklake.main.events")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let ids = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<arrow::array::Int64Array>()
            .unwrap();
        assert_eq!(ids.values(), &[7]);
        assert_eq!(batches[0].column(1).null_count(), 1);
    }

    let nested = ctx
        .sql("SELECT profile.zip, profile.address.street FROM ducklake.main.nested")
        .await
        .unwrap();
    assert_eq!(nested.schema().fields().len(), 2);

    if env.official_oracle {
        let output = std::process::Command::new(std::env::var("DUCKDB_CLI").unwrap())
            .args(["-csv", "-noheader", "-c"])
            .arg(format!(
                "INSTALL ducklake; LOAD ducklake; \
                 ATTACH 'ducklake:{}' AS official (META_TYPE 'sqlite'); \
                 SELECT event_id, coalesce(note, '<null>') FROM official.main.events; \
                 SELECT profile.zip, profile.address.street FROM official.main.nested;",
                env.catalog_path.display()
            ))
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "DuckDB read failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(String::from_utf8(output.stdout).unwrap().trim(), "7,<null>");
    }
}
