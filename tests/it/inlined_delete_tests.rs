//! Integration tests for positional deletes stored in metadata catalog tables.

use datafusion_ducklake::MetadataProvider;
use datafusion_ducklake::metadata_provider::DuckLakeInlinedDelete;

#[cfg(feature = "metadata-duckdb")]
mod duckdb_oracle {
    use std::path::Path;
    use std::sync::Arc;

    use arrow::array::{Int32Array, Int64Array};
    use datafusion::prelude::*;
    use datafusion_ducklake::{DuckLakeCatalog, DuckdbMetadataProvider};
    use tempfile::TempDir;

    use super::{DuckLakeInlinedDelete, MetadataProvider};
    use crate::common;

    fn escaped_path(path: &Path) -> String {
        path.to_string_lossy().replace('\'', "''")
    }

    async fn ids(ctx: &SessionContext) -> Vec<i32> {
        let batches = ctx
            .sql("SELECT id FROM lake.main.items ORDER BY id")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        batches
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
            .collect()
    }

    async fn rowids(ctx: &SessionContext) -> Vec<(i64, i32)> {
        let batches = ctx
            .sql("SELECT rowid, id FROM lake.main.items ORDER BY id")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let mut rows = Vec::new();
        for batch in batches {
            let rowids = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            let ids = batch
                .column(1)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            rows.extend(
                rowids
                    .values()
                    .iter()
                    .copied()
                    .zip(ids.values().iter().copied()),
            );
        }
        rows
    }

    async fn count(ctx: &SessionContext) -> i64 {
        let batches = ctx
            .sql("SELECT COUNT(*) FROM lake.main.items")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0)
    }

    #[tokio::test]
    async fn inlined_delete_matches_duckdb_select_count_rowid_and_time_travel() {
        let temp = TempDir::new().unwrap();
        let catalog_path = temp.path().join("deletes.ducklake");
        let data_path = temp.path().join("data");
        let expected = {
            common::ensure_ducklake_installed();
            let conn = duckdb::Connection::open_in_memory().unwrap();
            conn.execute("LOAD ducklake", []).unwrap();
            conn.execute(
                &format!(
                    "ATTACH 'ducklake:{}' AS lake (DATA_PATH '{}', DATA_INLINING_ROW_LIMIT 10)",
                    escaped_path(&catalog_path),
                    escaped_path(&data_path)
                ),
                [],
            )
            .unwrap();
            conn.execute("CREATE TABLE lake.items(id INTEGER)", [])
                .unwrap();
            conn.execute("INSERT INTO lake.items SELECT i FROM range(20) t(i)", [])
                .unwrap();
            conn.execute("DELETE FROM lake.items WHERE id IN (3, 7)", [])
                .unwrap();

            let mut statement = conn
                .prepare("SELECT id FROM lake.main.items ORDER BY id")
                .unwrap();
            statement
                .query_map([], |row| row.get::<_, i32>(0))
                .unwrap()
                .collect::<std::result::Result<Vec<_>, _>>()
                .unwrap()
        };

        // DuckDB 1.5 creates this table directly. The 1.4.1 library pinned by
        // this crate emits a regular positional delete, so normalize only that
        // metadata row into the current specification's equivalent encoding.
        let metadata = duckdb::Connection::open(&catalog_path).unwrap();
        let table_id: i64 = metadata
            .query_row(
                "SELECT table_id FROM ducklake_table WHERE table_name = 'items'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        let current_snapshot: i64 = metadata
            .query_row(
                "SELECT max(snapshot_id) FROM ducklake_snapshot",
                [],
                |row| row.get(0),
            )
            .unwrap();
        let inlined_delete_table = format!("ducklake_inlined_delete_{table_id}");
        let has_inlined_deletes: bool = metadata
            .query_row(
                "SELECT count(*) = 1 FROM information_schema.tables WHERE table_schema = 'main' AND table_name = ?",
                [&inlined_delete_table],
                |row| row.get(0),
            )
            .unwrap();
        if !has_inlined_deletes {
            metadata
                .execute(
                    &format!(
                        "CREATE TABLE {inlined_delete_table}(\
                         file_id BIGINT, row_id BIGINT, begin_snapshot BIGINT)"
                    ),
                    [],
                )
                .unwrap();
            let data_file_id: i64 = metadata
                .query_row(
                    "SELECT data_file_id FROM ducklake_data_file \
                     WHERE table_id = ? AND end_snapshot IS NULL",
                    [table_id],
                    |row| row.get(0),
                )
                .unwrap();
            let inserted = metadata
                .execute(
                    &format!("INSERT INTO {inlined_delete_table} VALUES (?, 3, ?), (?, 7, ?)"),
                    duckdb::params![data_file_id, current_snapshot, data_file_id, current_snapshot],
                )
                .unwrap();
            assert_eq!(inserted, 2);
            let removed = metadata
                .execute(
                    "DELETE FROM ducklake_delete_file WHERE begin_snapshot = ?",
                    [current_snapshot],
                )
                .unwrap();
            assert_eq!(removed, 1);
        }
        drop(metadata);

        let path = catalog_path.to_string_lossy().to_string();
        let provider = DuckdbMetadataProvider::new(&path).unwrap();
        assert_eq!(provider.get_current_snapshot().unwrap(), current_snapshot);
        let schema = provider
            .get_schema_by_name("main", current_snapshot)
            .unwrap()
            .unwrap();
        let table = provider
            .get_table_by_name(schema.schema_id, "items", current_snapshot)
            .unwrap()
            .unwrap();
        let deletes = provider
            .get_inlined_deletes(table.table_id, current_snapshot)
            .unwrap();
        let data_file_id = deletes[0].data_file_id;
        assert_eq!(
            deletes,
            vec![
                DuckLakeInlinedDelete {
                    data_file_id,
                    row_id: 3,
                },
                DuckLakeInlinedDelete {
                    data_file_id,
                    row_id: 7,
                },
            ]
        );

        let catalog = DuckLakeCatalog::new(provider)
            .unwrap()
            .with_row_lineage(true);
        let current = SessionContext::new();
        current.register_catalog("lake", Arc::new(catalog));
        assert_eq!(ids(&current).await, expected);
        assert_eq!(count(&current).await, expected.len() as i64);
        assert_eq!(
            rowids(&current).await,
            expected
                .iter()
                .map(|id| (i64::from(*id), *id))
                .collect::<Vec<_>>()
        );

        let previous_provider = Arc::new(DuckdbMetadataProvider::new(&path).unwrap());
        let previous_catalog = DuckLakeCatalog::with_snapshot(
            previous_provider,
            current_snapshot.checked_sub(1).unwrap(),
        )
        .unwrap();
        let previous = SessionContext::new();
        previous.register_catalog("lake", Arc::new(previous_catalog));
        assert_eq!(ids(&previous).await, (0..20).collect::<Vec<_>>());
        assert_eq!(count(&previous).await, 20);
    }
}

#[cfg(feature = "metadata-sqlite")]
#[tokio::test(flavor = "multi_thread")]
async fn sqlite_inlined_delete_lookup_is_snapshot_aware_and_optional() {
    use datafusion_ducklake::SqliteMetadataProvider;
    use sqlx::SqlitePool;
    use tempfile::TempDir;

    let temp = TempDir::new().unwrap();
    let url = format!(
        "sqlite:{}?mode=rwc",
        temp.path().join("catalog.db").display()
    );
    let pool = SqlitePool::connect(&url).await.unwrap();
    let provider = SqliteMetadataProvider::new(&url).await.unwrap();
    assert!(provider.get_inlined_deletes(5, 2).unwrap().is_empty());
    assert_eq!(
        provider.get_inlined_deletes(-1, 2).unwrap_err().to_string(),
        "Invalid configuration: DuckLake table id must be non-negative, was -1"
    );

    sqlx::query(
        "CREATE TABLE ducklake_inlined_delete_5(
             file_id BIGINT, row_id BIGINT, begin_snapshot BIGINT)",
    )
    .execute(&pool)
    .await
    .unwrap();
    sqlx::query("INSERT INTO ducklake_inlined_delete_5 VALUES (9, 3, 1), (9, 7, 3)")
        .execute(&pool)
        .await
        .unwrap();

    let deletes = provider.get_inlined_deletes(5, 2).unwrap();
    assert_eq!(
        deletes,
        vec![DuckLakeInlinedDelete {
            data_file_id: 9,
            row_id: 3,
        }]
    );
}

#[cfg(feature = "metadata-postgres")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
#[tokio::test(flavor = "multi_thread")]
async fn postgres_inlined_delete_lookup_reads_native_bigints() {
    use datafusion_ducklake::PostgresMetadataProvider;
    use sqlx::PgPool;
    use testcontainers::runners::AsyncRunner;
    use testcontainers_modules::postgres::Postgres;

    let container = Postgres::default().start().await.unwrap();
    let port = container.get_host_port_ipv4(5432).await.unwrap();
    let url = format!("postgresql://postgres:postgres@127.0.0.1:{port}/postgres");
    let pool = PgPool::connect(&url).await.unwrap();
    let provider = PostgresMetadataProvider::new(&url).await.unwrap();
    assert!(provider.get_inlined_deletes(5, 2).unwrap().is_empty());
    sqlx::query(
        "CREATE TABLE ducklake_inlined_delete_5(
             file_id BIGINT, row_id BIGINT, begin_snapshot BIGINT)",
    )
    .execute(&pool)
    .await
    .unwrap();
    sqlx::query("INSERT INTO ducklake_inlined_delete_5 VALUES (9, 3, 1), (9, 7, 3)")
        .execute(&pool)
        .await
        .unwrap();

    let deletes = provider.get_inlined_deletes(5, 2).unwrap();
    assert_eq!(
        deletes,
        vec![DuckLakeInlinedDelete {
            data_file_id: 9,
            row_id: 3,
        }]
    );
}

#[cfg(feature = "metadata-mysql")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
#[tokio::test(flavor = "multi_thread")]
async fn mysql_inlined_delete_lookup_reads_native_bigints() {
    use datafusion_ducklake::MySqlMetadataProvider;
    use sqlx::MySqlPool;
    use testcontainers::runners::AsyncRunner;
    use testcontainers_modules::mysql::Mysql;

    let container = Mysql::default().start().await.unwrap();
    let port = container.get_host_port_ipv4(3306).await.unwrap();
    let url = format!("mysql://root@127.0.0.1:{port}/test");
    let pool = MySqlPool::connect(&url).await.unwrap();
    let provider = MySqlMetadataProvider::new(&url).await.unwrap();
    assert!(provider.get_inlined_deletes(5, 2).unwrap().is_empty());
    sqlx::query(
        "CREATE TABLE ducklake_inlined_delete_5(
             file_id BIGINT, row_id BIGINT, begin_snapshot BIGINT)",
    )
    .execute(&pool)
    .await
    .unwrap();
    sqlx::query("INSERT INTO ducklake_inlined_delete_5 VALUES (9, 3, 1), (9, 7, 3)")
        .execute(&pool)
        .await
        .unwrap();

    let deletes = provider.get_inlined_deletes(5, 2).unwrap();
    assert_eq!(
        deletes,
        vec![DuckLakeInlinedDelete {
            data_file_id: 9,
            row_id: 3,
        }]
    );
}
