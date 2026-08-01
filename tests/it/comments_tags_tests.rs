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
        let comments = ctx
            .sql(
                "SELECT comment FROM ducklake.information_schema.columns
                 WHERE table_name = 'events' AND column_name = 'name'",
            )
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let column_comment = comments[0]
            .column(0)
            .as_any()
            .downcast_ref::<arrow::array::StringArray>()
            .unwrap();
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

    // Each comment is one snapshot that bumps `schema_version` without a ledger row and
    // records an alteration, as official does; DuckDB caches the catalog by that version.
    let snapshots: Vec<(i64, i64, Option<String>)> = sqlx::query_as(
        "SELECT s.snapshot_id, s.schema_version, c.changes_made FROM ducklake_snapshot s
         JOIN ducklake_snapshot_changes c ON c.snapshot_id = s.snapshot_id
         WHERE s.snapshot_id >= 1 ORDER BY s.snapshot_id",
    )
    .fetch_all(&sqlite)
    .await
    .unwrap();
    let altered = Some(format!("altered_table:{}", env.table_id));
    let base_version = snapshots[0].1;
    assert_eq!(
        snapshots[1..].to_vec(),
        vec![
            (2, base_version + 1, altered.clone()),
            (3, base_version + 2, altered.clone()),
            (4, base_version + 3, altered),
        ]
    );
    let ledger_rows: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM ducklake_schema_versions WHERE begin_snapshot >= 2",
    )
    .fetch_one(&sqlite)
    .await
    .unwrap();
    assert_eq!(ledger_rows, 0);
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
            "comment",
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

#[tokio::test(flavor = "multi_thread")]
async fn unsupported_tags_are_refused_and_the_catalog_stays_readable_by_duckdb() {
    let env = setup().await;
    let writer = Arc::new(
        SqliteMetadataWriter::new_with_init(&env.conn_str)
            .await
            .unwrap(),
    );
    let provider = Arc::new(SqliteMetadataProvider::new(&env.conn_str).await.unwrap());
    let head = provider.get_current_snapshot().unwrap();
    let schema = provider.get_schema_by_name("main", head).unwrap().unwrap();
    let catalog = Arc::new(DuckLakeCatalog::with_writer(provider.clone(), writer.clone()).unwrap());
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", catalog.clone());

    let schema_sql = execute_ducklake_sql(
        &ctx,
        catalog.as_ref(),
        "COMMENT ON SCHEMA ducklake.main IS 'x'",
    )
    .await
    .unwrap_err();
    assert!(
        schema_sql
            .to_string()
            .contains("COMMENT ON SCHEMA is not supported"),
        "{schema_sql}"
    );
    writer
        .set_tag(
            TagTarget::Object {
                object_type: TagObjectType::Schema,
                object_id: schema.schema_id,
            },
            "comment",
            Some("x"),
        )
        .unwrap_err();
    let column_key = writer
        .set_tag(
            TagTarget::Column {
                table_id: env.table_id,
                column_id: env.column_ids[0],
            },
            "classification",
            Some("internal"),
        )
        .unwrap_err();
    assert!(
        column_key.to_string().contains("only 'comment'"),
        "{column_key}"
    );
    writer
        .set_tag(
            TagTarget::Object {
                object_type: TagObjectType::Table,
                object_id: env.table_id + 1000,
            },
            "comment",
            Some("x"),
        )
        .unwrap_err();

    assert_eq!(provider.get_current_snapshot().unwrap(), head);
    let conn = attach_sqlite_catalog(&env.catalog_path);
    let rows: i64 = conn
        .query_row("SELECT count(*) FROM oracle.main.events", [], |row| {
            row.get(0)
        })
        .unwrap();
    assert_eq!(rows, 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn attached_duckdb_session_sees_a_comment_written_by_the_crate() {
    let env = setup().await;
    let conn = attach_sqlite_catalog(&env.catalog_path);
    let comment = |conn: &duckdb::Connection| -> Option<String> {
        conn.query_row(
            "SELECT comment FROM duckdb_tables()
             WHERE database_name = 'oracle' AND table_name = 'events'",
            [],
            |row| row.get(0),
        )
        .unwrap()
    };
    assert_eq!(comment(&conn), None);

    let writer = SqliteMetadataWriter::new_with_init(&env.conn_str)
        .await
        .unwrap();
    writer
        .set_tag(
            TagTarget::Object {
                object_type: TagObjectType::Table,
                object_id: env.table_id,
            },
            "comment",
            Some("from crate"),
        )
        .unwrap();

    assert_eq!(comment(&conn).as_deref(), Some("from crate"));
}

#[tokio::test(flavor = "multi_thread")]
async fn expire_snapshots_reclaims_object_tags_and_column_tags_of_dead_tables() {
    let env = setup().await;
    let writer = SqliteMetadataWriter::new_with_init(&env.conn_str)
        .await
        .unwrap();
    let table = TagTarget::Object {
        object_type: TagObjectType::Table,
        object_id: env.table_id,
    };
    let column = TagTarget::Column {
        table_id: env.table_id,
        column_id: env.column_ids[1],
    };
    for (target, value) in [(table, "t1"), (table, "t2"), (column, "c1"), (column, "c2")] {
        writer.set_tag(target, "comment", Some(value)).unwrap();
    }
    let sqlite = sqlx::SqlitePool::connect(&env.conn_str).await.unwrap();
    let object_tags = || async {
        sqlx::query_as::<_, (i64, Option<i64>, Option<String>)>(
            "SELECT begin_snapshot, end_snapshot, value FROM ducklake_tag ORDER BY begin_snapshot",
        )
        .fetch_all(&sqlite)
        .await
        .unwrap()
    };
    let column_tags = || async {
        sqlx::query_as::<_, (i64, Option<i64>, Option<String>)>(
            "SELECT begin_snapshot, end_snapshot, value FROM ducklake_column_tag
             ORDER BY begin_snapshot",
        )
        .fetch_all(&sqlite)
        .await
        .unwrap()
    };
    let expire = |versions: Vec<i64>| {
        writer
            .expire_snapshots(datafusion_ducklake::maintenance::ExpireCriteria::Versions(
                versions,
            ))
            .unwrap()
    };

    // Official reclaims an ended object tag no surviving snapshot covers, but keeps an ended
    // column tag of a live table.
    expire(vec![2, 3, 4]);
    assert_eq!(object_tags().await, vec![(3, None, Some("t2".to_string()))]);
    assert_eq!(
        column_tags().await,
        vec![(4, Some(5), Some("c1".to_string())), (5, None, Some("c2".to_string())),]
    );

    // A dead table takes all of its column tags, ended or live.
    assert!(writer.drop_table("main", "events").unwrap());
    expire(vec![1, 5]);
    assert_eq!(column_tags().await, vec![]);
}

async fn crate_created_catalog(temp: &TempDir) -> String {
    let conn_str = format!(
        "sqlite:{}?mode=rwc",
        temp.path().join("crate.sqlite").display()
    );
    let data = temp.path().join("data");
    std::fs::create_dir(&data).unwrap();
    let writer = Arc::new(
        SqliteMetadataWriter::new_with_init(&conn_str)
            .await
            .unwrap(),
    );
    writer.set_data_path(data.to_str().unwrap()).unwrap();
    let batch = arrow::record_batch::RecordBatch::try_new(
        Arc::new(arrow::datatypes::Schema::new(vec![
            arrow::datatypes::Field::new("id", arrow::datatypes::DataType::Int32, false),
        ])),
        vec![Arc::new(arrow::array::Int32Array::from(vec![1]))],
    )
    .unwrap();
    datafusion_ducklake::DuckLakeTableWriter::new(
        writer,
        Arc::new(object_store::local::LocalFileSystem::new()),
    )
    .unwrap()
    .write_table("main", "t", &[batch])
    .await
    .unwrap();
    conn_str
}

#[tokio::test(flavor = "multi_thread")]
async fn crate_created_catalog_keeps_table_comments_and_has_no_schema_comments() {
    let temp = TempDir::new().unwrap();
    let conn_str = crate_created_catalog(&temp).await;
    let writer = Arc::new(SqliteMetadataWriter::new(&conn_str).await.unwrap());
    let provider = Arc::new(SqliteMetadataProvider::new(&conn_str).await.unwrap());
    let catalog = Arc::new(DuckLakeCatalog::with_writer(provider, writer).unwrap());
    let ctx = SessionContext::new();
    ctx.register_catalog("lake", catalog.clone());

    // On a catalog this crate created, the schema and the table both have id 1.
    execute_ducklake_sql(
        &ctx,
        catalog.as_ref(),
        "COMMENT ON TABLE lake.main.t IS 'T'",
    )
    .await
    .unwrap();
    execute_ducklake_sql(&ctx, catalog.as_ref(), "COMMENT ON SCHEMA lake.main IS 'S'")
        .await
        .unwrap_err();

    let batches = ctx
        .sql("SELECT * FROM lake.information_schema.schemata")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(batches[0].num_columns(), 5);
    let batches = ctx
        .sql("SELECT comment FROM lake.information_schema.tables WHERE table_name = 't'")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let comment = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<arrow::array::StringArray>()
        .unwrap();
    assert_eq!(comment.value(0), "T");
}

#[tokio::test(flavor = "multi_thread")]
async fn catalog_predating_tags_opens_with_new_and_takes_comments_and_drops() {
    let temp = TempDir::new().unwrap();
    let conn_str = crate_created_catalog(&temp).await;
    let sqlite = sqlx::SqlitePool::connect(&conn_str).await.unwrap();
    for table in ["ducklake_tag", "ducklake_column_tag"] {
        sqlx::query(sqlx::AssertSqlSafe(format!("DROP TABLE {table}")))
            .execute(&sqlite)
            .await
            .unwrap();
    }

    let writer = SqliteMetadataWriter::new(&conn_str).await.unwrap();
    let table_id: i64 = sqlx::query_scalar("SELECT table_id FROM ducklake_table")
        .fetch_one(&sqlite)
        .await
        .unwrap();
    writer
        .set_tag(
            TagTarget::Object {
                object_type: TagObjectType::Table,
                object_id: table_id,
            },
            "comment",
            Some("late"),
        )
        .unwrap();
    assert!(writer.drop_table("main", "t").unwrap());
}

#[tokio::test(flavor = "multi_thread")]
async fn a_comment_copies_the_watermarks_of_the_preceding_snapshot() {
    let env = setup().await;
    let writer = SqliteMetadataWriter::new_with_init(&env.conn_str)
        .await
        .unwrap();
    let sqlite = sqlx::SqlitePool::connect(&env.conn_str).await.unwrap();
    let watermark = |snapshot: i64| {
        let sqlite = sqlite.clone();
        async move {
            sqlx::query_as::<_, (Option<i64>, Option<i64>)>(
                "SELECT next_catalog_id, next_file_id FROM ducklake_snapshot
                 WHERE snapshot_id = ?",
            )
            .bind(snapshot)
            .fetch_one(&sqlite)
            .await
            .unwrap()
        }
    };
    let comment = |text: &'static str| {
        writer
            .set_tag(
                TagTarget::Object {
                    object_type: TagObjectType::Table,
                    object_id: env.table_id,
                },
                "comment",
                Some(text),
            )
            .unwrap()
    };
    let base = watermark(1).await;
    assert!(base.0.is_some() && base.1.is_some());

    // A comment after a DuckDB commit keeps its watermark, however many comments follow.
    assert_eq!(watermark(comment("c1")).await, base);
    assert_eq!(watermark(comment("c2")).await, base);

    // The crate allocates a partition id from its own counter and leaves the watermarks NULL.
    // A later comment must copy that NULL, not the stale value from before the allocation.
    let partition = writer
        .set_partition_spec(
            env.table_id,
            &[(
                "name".to_string(),
                datafusion_ducklake::partition::PartitionTransform::Identity,
            )],
        )
        .unwrap();
    assert_eq!(watermark(partition).await, (None, None));
    assert_eq!(watermark(comment("c3")).await, (None, None));
    assert_eq!(watermark(comment("c4")).await, (None, None));
}
