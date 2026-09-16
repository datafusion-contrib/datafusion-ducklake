//! DuckLake comment and tag interoperability tests for SQLite catalogs.

#![cfg(all(feature = "write-sqlite", feature = "metadata-sqlite", feature = "metadata-duckdb"))]

use std::sync::Arc;

use datafusion::prelude::SessionContext;
use datafusion_ducklake::metadata_provider::MetadataProvider;
use datafusion_ducklake::{
    DuckLakeCatalog, MetadataWriter, SqliteMetadataProvider, SqliteMetadataWriter, TagObjectType,
    TagTarget, execute_ducklake_sql,
};
use tempfile::TempDir;

struct Env {
    conn_str: String,
    catalog_path: std::path::PathBuf,
    table_id: i64,
    column_ids: Vec<i64>,
    _temp: TempDir,
}

async fn setup() -> Env {
    let temp = TempDir::new().unwrap();
    let catalog_path = temp.path().join("metadata.sqlite");
    let conn_str = format!("sqlite:{}?mode=rwc", catalog_path.display());
    {
        let conn = attach_sqlite_catalog(&catalog_path);
        conn.execute(
            "CREATE TABLE oracle.main.events (id BIGINT NOT NULL, name VARCHAR)",
            [],
        )
        .unwrap();
    }
    let provider = SqliteMetadataProvider::new(&conn_str).await.unwrap();
    let snapshot = provider.get_current_snapshot().unwrap();
    let schema = provider
        .get_schema_by_name("main", snapshot)
        .unwrap()
        .unwrap();
    let table = provider
        .get_table_by_name(schema.schema_id, "events", snapshot)
        .unwrap()
        .unwrap();
    let columns = provider
        .get_table_structure(table.table_id, snapshot)
        .unwrap();
    Env {
        conn_str,
        catalog_path,
        table_id: table.table_id,
        column_ids: columns.into_iter().map(|column| column.column_id).collect(),
        _temp: temp,
    }
}

static INSTALL_EXTENSIONS: std::sync::Once = std::sync::Once::new();

fn attach_sqlite_catalog(path: &std::path::Path) -> duckdb::Connection {
    INSTALL_EXTENSIONS.call_once(|| {
        let conn = duckdb::Connection::open_in_memory().unwrap();
        conn.execute_batch("INSTALL ducklake; INSTALL sqlite;")
            .unwrap();
    });
    let conn = duckdb::Connection::open_in_memory().unwrap();
    conn.execute_batch("LOAD ducklake; LOAD sqlite;").unwrap();
    let data_path = path.with_extension("data");
    std::fs::create_dir_all(&data_path).unwrap();
    conn.execute(
        &format!(
            "ATTACH 'ducklake:sqlite:{}' AS oracle (DATA_PATH '{}')",
            path.to_string_lossy().replace('\'', "''"),
            data_path.to_string_lossy().replace('\'', "''")
        ),
        [],
    )
    .unwrap();
    conn
}

#[tokio::test(flavor = "multi_thread")]
async fn reads_duckdb_comments_and_exposes_information_schema_tags() {
    let env = setup().await;
    {
        let conn = attach_sqlite_catalog(&env.catalog_path);
        conn.execute(
            "COMMENT ON TABLE oracle.main.events IS 'table from DuckDB'",
            [],
        )
        .unwrap();
        conn.execute(
            "COMMENT ON COLUMN oracle.main.events.name IS 'column from DuckDB'",
            [],
        )
        .unwrap();
        conn.execute(
            "CREATE VIEW oracle.main.event_names AS SELECT name FROM oracle.main.events",
            [],
        )
        .unwrap();
        conn.execute(
            "COMMENT ON VIEW oracle.main.event_names IS 'view from DuckDB'",
            [],
        )
        .unwrap();
    }

    let provider = Arc::new(SqliteMetadataProvider::new(&env.conn_str).await.unwrap());
    let snapshot = provider.get_current_snapshot().unwrap();
    assert_eq!(snapshot, 5);
    assert_eq!(
        provider
            .get_tags(
                TagTarget::Object {
                    object_type: TagObjectType::Table,
                    object_id: env.table_id,
                },
                snapshot,
            )
            .unwrap()[0]
            .value
            .as_deref(),
        Some("table from DuckDB")
    );
    assert_eq!(
        provider
            .get_tags(
                TagTarget::Column {
                    table_id: env.table_id,
                    column_id: env.column_ids[1],
                },
                snapshot,
            )
            .unwrap()[0]
            .value
            .as_deref(),
        Some("column from DuckDB")
    );
    let schema = provider
        .get_schema_by_name("main", snapshot)
        .unwrap()
        .unwrap();
    let view_id = provider
        .get_view_id_by_name(schema.schema_id, "event_names", snapshot)
        .unwrap()
        .unwrap();
    assert_eq!(
        provider
            .get_tags(
                TagTarget::Object {
                    object_type: TagObjectType::View,
                    object_id: view_id,
                },
                snapshot,
            )
            .unwrap()[0]
            .value
            .as_deref(),
        Some("view from DuckDB")
    );

    let catalog = DuckLakeCatalog::new(provider.as_ref().clone()).unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));
    let rows = ctx
        .sql(
            "SELECT table_name, comment FROM ducklake.information_schema.tables
             WHERE table_name = 'events'",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].num_rows(), 1);
    assert_eq!(
        rows[0]
            .column(1)
            .as_any()
            .downcast_ref::<arrow::array::StringArray>()
            .unwrap()
            .value(0),
        "table from DuckDB"
    );
    let tag_rows = ctx
        .sql(
            "SELECT table_id, column_id, key, value
             FROM ducklake.information_schema.column_tags",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(tag_rows.len(), 1);
    assert_eq!(tag_rows[0].num_rows(), 1);
    let view_tags = ctx
        .sql(
            "SELECT object_type, object_id, value
             FROM ducklake.information_schema.object_tags
             WHERE object_type = 'view'",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(view_tags.len(), 1);
    assert_eq!(view_tags[0].num_rows(), 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn sql_comments_round_trip_to_duckdb_and_version_tags() {
    let env = setup().await;
    {
        let writer = Arc::new(
            SqliteMetadataWriter::new_with_init(&env.conn_str)
                .await
                .unwrap(),
        );
        let provider = Arc::new(SqliteMetadataProvider::new(&env.conn_str).await.unwrap());
        let catalog = Arc::new(
            DuckLakeCatalog::with_writer(provider, writer)
                .expect("create writable DuckLake catalog"),
        );
        let ctx = SessionContext::new();
        ctx.register_catalog("ducklake", catalog.clone());
        execute_ducklake_sql(
            &ctx,
            catalog.as_ref(),
            "COMMENT ON TABLE ducklake.main.events IS 'table from Rust'",
        )
        .await
        .unwrap();
        execute_ducklake_sql(
            &ctx,
            catalog.as_ref(),
            "COMMENT ON TABLE ducklake.main.events IS 'table replaced'",
        )
        .await
        .unwrap();
        execute_ducklake_sql(
            &ctx,
            catalog.as_ref(),
            "COMMENT ON COLUMN ducklake.main.events.name IS 'column from Rust'",
        )
        .await
        .unwrap();
        execute_ducklake_sql(
            &ctx,
            catalog.as_ref(),
            "COMMENT ON SCHEMA ducklake.main IS 'schema from Rust'",
        )
        .await
        .unwrap();
        let comments = ctx
            .sql(
                "SELECT
                    (SELECT comment FROM ducklake.information_schema.schemata
                     WHERE schema_name = 'main') AS schema_comment,
                    (SELECT comment FROM ducklake.information_schema.columns
                     WHERE table_name = 'events' AND column_name = 'name') AS column_comment",
            )
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let schema_comment = comments[0]
            .column(0)
            .as_any()
            .downcast_ref::<arrow::array::StringArray>()
            .unwrap();
        let column_comment = comments[0]
            .column(1)
            .as_any()
            .downcast_ref::<arrow::array::StringArray>()
            .unwrap();
        assert_eq!(schema_comment.value(0), "schema from Rust");
        assert_eq!(column_comment.value(0), "column from Rust");
    }

    let conn = attach_sqlite_catalog(&env.catalog_path);
    let table_comment: Option<String> = conn
        .query_row(
            "SELECT comment FROM duckdb_tables()
             WHERE database_name = 'oracle' AND schema_name = 'main'
               AND table_name = 'events'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    let column_comment: Option<String> = conn
        .query_row(
            "SELECT comment FROM duckdb_columns()
             WHERE database_name = 'oracle' AND schema_name = 'main'
               AND table_name = 'events' AND column_name = 'name'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(table_comment.as_deref(), Some("table replaced"));
    assert_eq!(column_comment.as_deref(), Some("column from Rust"));

    let sqlite = sqlx::SqlitePool::connect(&env.conn_str).await.unwrap();
    let versions: Vec<(i64, Option<i64>, Option<String>)> = sqlx::query_as(
        "SELECT begin_snapshot, end_snapshot, value FROM ducklake_tag
             WHERE object_id = ? AND key = 'comment' ORDER BY begin_snapshot",
    )
    .bind(env.table_id)
    .fetch_all(&sqlite)
    .await
    .unwrap();
    assert_eq!(
        versions,
        vec![
            (2, Some(3), Some("table from Rust".to_string())),
            (3, None, Some("table replaced".to_string())),
        ]
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn drop_table_tombstones_table_and_column_tags() {
    let env = setup().await;
    let writer = SqliteMetadataWriter::new_with_init(&env.conn_str)
        .await
        .unwrap();
    let table_tag_snapshot = writer
        .set_tag(
            TagTarget::Object {
                object_type: TagObjectType::Table,
                object_id: env.table_id,
            },
            "owner",
            Some("analytics"),
        )
        .unwrap();
    let column_tag_snapshot = writer
        .set_tag(
            TagTarget::Column {
                table_id: env.table_id,
                column_id: env.column_ids[0],
            },
            "classification",
            Some("internal"),
        )
        .unwrap();
    assert_eq!(table_tag_snapshot, 2);
    assert_eq!(column_tag_snapshot, 3);
    assert!(writer.drop_table("main", "events").unwrap());

    let provider = SqliteMetadataProvider::new(&env.conn_str).await.unwrap();
    let drop_snapshot = provider.get_current_snapshot().unwrap();
    assert_eq!(drop_snapshot, 4);
    assert_eq!(
        provider.list_all_object_tags(drop_snapshot).unwrap(),
        vec![]
    );
    assert_eq!(
        provider.list_all_column_tags(drop_snapshot).unwrap(),
        vec![]
    );
    assert_eq!(
        provider
            .list_all_object_tags(column_tag_snapshot)
            .unwrap()
            .len(),
        1
    );
    assert_eq!(
        provider
            .list_all_column_tags(column_tag_snapshot)
            .unwrap()
            .len(),
        1
    );
}
