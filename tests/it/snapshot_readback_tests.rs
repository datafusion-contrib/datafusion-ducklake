#![cfg(all(feature = "metadata-duckdb", feature = "metadata-sqlite", feature = "write-sqlite"))]

use std::sync::Arc;

use arrow::array::{Array, Int64Array, StringArray};
use arrow::record_batch::RecordBatch;
use datafusion::prelude::SessionContext;
use datafusion_ducklake::{
    ColumnDef, DuckLakeCatalog, DuckdbMetadataProvider, MetadataProvider, MetadataWriter,
    SqliteMetadataProvider, SqliteMetadataWriter, WriteMode, register_ducklake_functions,
};

use crate::common;

struct ExpectedSnapshot<'a> {
    snapshot_id: i64,
    schema_version: i64,
    changes_made: Option<&'a str>,
    author: Option<&'a str>,
    commit_message: Option<&'a str>,
    commit_extra_info: Option<&'a str>,
}

fn assert_snapshot_row(batch: &RecordBatch, expected: &ExpectedSnapshot<'_>) {
    assert_eq!(batch.num_rows(), 1);
    assert_eq!(batch.num_columns(), 6);

    let snapshot_ids = batch
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let schema_versions = batch
        .column(1)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let changes_made = batch
        .column(2)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let authors = batch
        .column(3)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let commit_messages = batch
        .column(4)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let commit_extra_info = batch
        .column(5)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    assert_eq!(snapshot_ids.value(0), expected.snapshot_id);
    assert_eq!(schema_versions.value(0), expected.schema_version);
    assert_eq!(
        (!changes_made.is_null(0)).then(|| changes_made.value(0)),
        expected.changes_made
    );
    assert_eq!(
        (!authors.is_null(0)).then(|| authors.value(0)),
        expected.author
    );
    assert_eq!(
        (!commit_messages.is_null(0)).then(|| commit_messages.value(0)),
        expected.commit_message
    );
    assert_eq!(
        (!commit_extra_info.is_null(0)).then(|| commit_extra_info.value(0)),
        expected.commit_extra_info
    );
}

async fn query_snapshot(ctx: &SessionContext, sql: &str) -> RecordBatch {
    let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
    assert_eq!(batches.len(), 1);
    batches.into_iter().next().unwrap()
}

#[tokio::test(flavor = "multi_thread")]
async fn duckdb_commit_metadata_is_exposed_by_both_snapshot_surfaces() {
    let temp = tempfile::tempdir().unwrap();
    let catalog_path = temp.path().join("oracle.ducklake");
    common::create_catalog_no_deletes(&catalog_path).unwrap();

    let conn = duckdb::Connection::open_in_memory().unwrap();
    conn.execute("LOAD ducklake", []).unwrap();
    conn.execute(
        &format!("ATTACH 'ducklake:{}' AS oracle", catalog_path.display()),
        [],
    )
    .unwrap();
    conn.execute_batch(
        "BEGIN;
         INSERT INTO oracle.users VALUES (5, 'Grace', 'grace@example.com');
         CALL oracle.set_commit_message(
             'Ada',
             'Add Grace',
             extra_info => '{\"ticket\":42}'
         );
         COMMIT;",
    )
    .unwrap();
    drop(conn);

    let provider = DuckdbMetadataProvider::new(catalog_path.to_string_lossy()).unwrap();
    let snapshots = provider.list_snapshots().unwrap();
    let snapshot = snapshots.last().unwrap();
    let schema = provider
        .list_schemas(snapshot.snapshot_id)
        .unwrap()
        .into_iter()
        .find(|schema| schema.schema_name == "main")
        .unwrap();
    let table = provider
        .list_tables(schema.schema_id, snapshot.snapshot_id)
        .unwrap()
        .into_iter()
        .find(|table| table.table_name == "users")
        .unwrap();
    // DuckDB 1.5.5 inlines the small insert, which registers an inlined data
    // table and therefore advances the schema version.
    let changes_made = format!("inlined_insert:{}", table.table_id);
    let expected = ExpectedSnapshot {
        snapshot_id: snapshot.snapshot_id,
        schema_version: 2,
        changes_made: Some(&changes_made),
        author: Some("Ada"),
        commit_message: Some("Add Grace"),
        commit_extra_info: Some("{\"ticket\":42}"),
    };

    assert_eq!(snapshot.schema_version, expected.schema_version);
    assert_eq!(snapshot.changes_made.as_deref(), expected.changes_made);
    assert_eq!(snapshot.author.as_deref(), expected.author);
    assert_eq!(snapshot.commit_message.as_deref(), expected.commit_message);
    assert_eq!(
        snapshot.commit_extra_info.as_deref(),
        expected.commit_extra_info
    );

    let ctx = SessionContext::new();
    register_ducklake_functions(&ctx, Arc::new(provider.clone()));
    let catalog = DuckLakeCatalog::new(provider).unwrap();
    ctx.register_catalog("lake", Arc::new(catalog));

    let projection = format!(
        "SELECT snapshot_id, schema_version, changes_made, author, commit_message,
                commit_extra_info FROM {{surface}} WHERE snapshot_id = {}",
        snapshot.snapshot_id
    );
    let function_batch = query_snapshot(
        &ctx,
        &projection.replace("{surface}", "ducklake_snapshots()"),
    )
    .await;
    let information_schema_batch = query_snapshot(
        &ctx,
        &projection.replace("{surface}", "lake.information_schema.snapshots"),
    )
    .await;

    assert_snapshot_row(&function_batch, &expected);
    assert_snapshot_row(&information_schema_batch, &expected);
}

#[tokio::test(flavor = "multi_thread")]
async fn crate_commit_without_commit_metadata_preserves_nulls() {
    let temp = tempfile::tempdir().unwrap();
    let catalog_path = temp.path().join("crate.db");
    let connection = format!("sqlite:{}?mode=rwc", catalog_path.display());
    let writer = SqliteMetadataWriter::new_with_init(&connection)
        .await
        .unwrap();
    writer
        .set_data_path(temp.path().to_string_lossy().as_ref())
        .unwrap();

    let columns = vec![ColumnDef::new("id", "int64", false).unwrap()];
    let setup = writer
        .begin_write_transaction("main", "events", &columns, WriteMode::Replace)
        .unwrap();
    let committed = writer
        .publish_snapshot(
            setup.table_id,
            "main",
            "events",
            setup.snapshot_id,
            WriteMode::Replace,
            setup.base_snapshot_id,
            &columns,
            &setup.column_ids,
        )
        .unwrap();
    drop(writer);

    let read_connection = format!("sqlite:{}", catalog_path.display());
    let provider = SqliteMetadataProvider::new(&read_connection).await.unwrap();
    let snapshots = provider.list_snapshots().unwrap();
    assert_eq!(snapshots.len(), 1);
    let snapshot = &snapshots[0];
    let changes_made = format!(
        "created_schema:\"main\",created_table:\"main\".\"events\",inserted_into_table:{}",
        setup.table_id
    );
    let expected = ExpectedSnapshot {
        snapshot_id: committed.snapshot_id,
        schema_version: 1,
        changes_made: Some(&changes_made),
        author: None,
        commit_message: None,
        commit_extra_info: None,
    };

    assert_eq!(snapshot.snapshot_id, expected.snapshot_id);
    assert_eq!(snapshot.schema_version, expected.schema_version);
    assert_eq!(snapshot.changes_made.as_deref(), expected.changes_made);
    assert_eq!(snapshot.author, None);
    assert_eq!(snapshot.commit_message, None);
    assert_eq!(snapshot.commit_extra_info, None);

    let ctx = SessionContext::new();
    register_ducklake_functions(&ctx, Arc::new(provider));
    let batch = query_snapshot(
        &ctx,
        &format!(
            "SELECT snapshot_id, schema_version, changes_made, author, commit_message,
                    commit_extra_info
             FROM ducklake_snapshots() WHERE snapshot_id = {}",
            committed.snapshot_id
        ),
    )
    .await;
    assert_snapshot_row(&batch, &expected);
}
