#![cfg(all(feature = "metadata-duckdb", feature = "metadata-sqlite", feature = "write-sqlite"))]

use std::sync::Arc;

use arrow::array::{Array, Int64Array, ListArray, MapArray, StringArray};
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

    assert_eq!(snapshot.schema_version, Some(expected.schema_version));
    let changes = provider.list_snapshot_changes().unwrap();
    let snapshot = changes
        .iter()
        .find(|change| change.snapshot_id == snapshot.snapshot_id)
        .unwrap();
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
    assert_eq!(snapshot.schema_version, Some(expected.schema_version));
    let changes = provider.list_snapshot_changes().unwrap();
    let snapshot = changes
        .iter()
        .find(|change| change.snapshot_id == snapshot.snapshot_id)
        .unwrap();
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

#[tokio::test(flavor = "multi_thread")]
async fn structured_changes_match_official_duckdb() {
    let temp = tempfile::tempdir().unwrap();
    let catalog_path = temp.path().join("changes.ducklake");
    common::create_catalog_no_deletes(&catalog_path).unwrap();
    let changes = concat!(
        "created_schema:\"Sales,West\",created_schema:\"sales,west\",",
        "created_table:\"main\".\"events\",created_table:\"select\".\"a\"\"b,c.d\",",
        "created_table:\"MAIN\".\"abort\",created_view:\"MIXED\".\"v\",created_scalar_macro:\"main\".\"f\",",
        "created_table_macro:\"main\".\"g\",",
        "dropped_schema:12,dropped_schema:2,dropped_table:3,dropped_view:4,",
        "dropped_scalar_macro:5,dropped_table_macro:6,altered_table:7,altered_view:8,",
        "inserted_into_table:10,inserted_into_table:2,inserted_into_table:10,",
        "deleted_from_table:11,inlined_insert:12,inlined_delete:13,",
        "flushed_inlined:14,inline_flush:15,merge_adjacent:16,rewrite_delete:17,compacted_table:18"
    );
    let conn = duckdb::Connection::open(&catalog_path).unwrap();
    let snapshot_id: i64 = conn
        .query_row(
            "SELECT MAX(snapshot_id) FROM ducklake_snapshot",
            [],
            |row| row.get(0),
        )
        .unwrap();
    conn.execute(
        "UPDATE ducklake_snapshot_changes SET changes_made = ? WHERE snapshot_id = ?",
        duckdb::params![changes, snapshot_id],
    )
    .unwrap();
    drop(conn);

    let conn = duckdb::Connection::open_in_memory().unwrap();
    conn.execute_batch(&format!(
        "LOAD ducklake; ATTACH 'ducklake:{}' AS oracle;",
        catalog_path.display()
    ))
    .unwrap();
    let extension_version: String = conn
        .query_row(
            "SELECT extension_version FROM duckdb_extensions() WHERE extension_name = 'ducklake'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    eprintln!("DuckLake snapshot oracle extension: {extension_version}");
    let mut statement = conn
        .prepare(&format!(
            "SELECT entry.key, unnest(entry.value) FROM (
             SELECT unnest(map_entries(changes)) AS entry FROM ducklake_snapshots('oracle')
             WHERE snapshot_id = {snapshot_id}
         ) ORDER BY 1, 2"
        ))
        .unwrap();
    let expected = statement
        .query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    drop(statement);
    drop(conn);

    let provider = DuckdbMetadataProvider::new(catalog_path.to_string_lossy()).unwrap();
    let ctx = SessionContext::new();
    register_ducklake_functions(&ctx, Arc::new(provider.clone()));
    ctx.register_catalog("lake", Arc::new(DuckLakeCatalog::new(provider).unwrap()));
    for surface in ["ducklake_snapshots()", "lake.information_schema.snapshots"] {
        let batch = query_snapshot(
            &ctx,
            &format!(
                "SELECT changes, changes_made FROM {surface} WHERE snapshot_id = {snapshot_id}"
            ),
        )
        .await;
        assert_eq!(batch.num_rows(), 1);
        assert_eq!(batch.num_columns(), 2);
        let map = batch.column(0).as_any().downcast_ref::<MapArray>().unwrap();
        assert!(!map.is_null(0));
        let entries = map.value(0);
        let keys = entries
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let values = entries
            .column(1)
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        let mut actual = Vec::new();
        for index in 0..entries.len() {
            let values = values.value(index);
            let values = values.as_any().downcast_ref::<StringArray>().unwrap();
            for value in values.iter() {
                actual.push((keys.value(index).to_string(), value.unwrap().to_string()));
            }
        }
        actual.sort();
        assert_eq!(actual, expected);
        assert_eq!(
            batch
                .column(1)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .value(0),
            changes
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn nullable_schema_version_and_missing_ledger_row_remain_visible() {
    let temp = tempfile::tempdir().unwrap();
    let connection = format!(
        "sqlite:{}?mode=rwc",
        temp.path().join("nullable.db").display()
    );
    let pool = sqlx::SqlitePool::connect(&connection).await.unwrap();
    sqlx::raw_sql(
        "CREATE TABLE ducklake_snapshot (snapshot_id INTEGER, snapshot_time TEXT, schema_version INTEGER);
         CREATE TABLE ducklake_snapshot_changes (snapshot_id INTEGER, changes_made TEXT, author TEXT, commit_message TEXT, commit_extra_info TEXT);
         INSERT INTO ducklake_snapshot VALUES (19, '2026-09-22 12:34:56', NULL), (23, '2026-09-22 13:45:01', 7);
         INSERT INTO ducklake_snapshot_changes VALUES (23, '', NULL, NULL, NULL);"
    ).execute(&pool).await.unwrap();
    pool.close().await;
    let provider = SqliteMetadataProvider::new(&connection).await.unwrap();
    let ctx = SessionContext::new();
    register_ducklake_functions(&ctx, Arc::new(provider));
    let batch = query_snapshot(
        &ctx,
        "SELECT * FROM ducklake_snapshots() ORDER BY snapshot_id",
    )
    .await;
    assert_eq!(batch.num_rows(), 2);
    assert_eq!(batch.num_columns(), 8);
    assert_eq!(
        batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap(),
        &Int64Array::from(vec![19, 23])
    );
    assert_eq!(
        batch
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap(),
        &StringArray::from(vec![
            "2026-09-22 12:34:56.000000",
            "2026-09-22 13:45:01.000000"
        ])
    );
    assert_eq!(
        batch
            .column(2)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap(),
        &Int64Array::from(vec![None, Some(7)])
    );
    let maps = batch.column(3).as_any().downcast_ref::<MapArray>().unwrap();
    assert_eq!(maps.null_count(), 0);
    assert_eq!(maps.value_offsets(), &[0, 0, 0]);
    assert_eq!(
        batch
            .column(4)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap(),
        &StringArray::from(vec![None, Some("")])
    );
    for index in 5..8 {
        assert_eq!(
            batch
                .column(index)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap(),
            &StringArray::from(vec![None::<&str>, None])
        );
    }
}
