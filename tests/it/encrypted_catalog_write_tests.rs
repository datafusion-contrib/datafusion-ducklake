//! Writes into a catalog whose `encrypted` option is `true`. Official DuckLake
//! writes such a catalog's Parquet files with Parquet Modular Encryption; this
//! crate writes no encrypted Parquet, so every path that would write a Parquet
//! file fails with one `Unsupported` error, while inlined writes, which write no
//! Parquet, proceed as official's do.

#![cfg(feature = "write-sqlite")]

use std::sync::Arc;

use arrow::array::{Int32Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::prelude::SessionContext;
use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeError, DuckLakeTable, DuckLakeTableWriter, DuckLakeWriteOptions,
    MergeOptions, MetadataProvider, MetadataWriter, RewriteOptions, SqliteMetadataProvider,
    SqliteMetadataWriter, execute_ducklake_sql,
};
use object_store::local::LocalFileSystem;
use tempfile::TempDir;

const REFUSAL: &str = "the catalog is encrypted, and this crate does not write encrypted Parquet";

struct Lake {
    _temp: TempDir,
    url: String,
    writer: Arc<SqliteMetadataWriter>,
    provider: Arc<dyn MetadataProvider>,
}

fn ids(range: std::ops::Range<i32>) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, true)])),
        vec![Arc::new(Int32Array::from(range.collect::<Vec<_>>()))],
    )
    .unwrap()
}

/// A catalog holding `main.t` with two Parquet files (ids 0..20 and 20..40),
/// then marked encrypted, with inlining limit `limit`.
async fn encrypted_lake(limit: usize) -> Lake {
    let temp = TempDir::new().unwrap();
    let url = format!("sqlite:{}?mode=rwc", temp.path().join("c.db").display());
    let data = temp.path().join("data");
    std::fs::create_dir_all(&data).unwrap();
    let writer = Arc::new(SqliteMetadataWriter::new_with_init(&url).await.unwrap());
    writer.set_data_path(data.to_str().unwrap()).unwrap();
    for batch in [ids(0..20), ids(20..40)] {
        let metadata: Arc<dyn MetadataWriter> = writer.clone();
        DuckLakeTableWriter::new(metadata, Arc::new(LocalFileSystem::new()))
            .unwrap()
            .with_options(&DuckLakeWriteOptions::default().with_data_inlining_row_limit(0))
            .append_table("main", "t", &[batch])
            .await
            .unwrap();
    }
    writer
        .set_global_setting("data_inlining_row_limit", &limit.to_string())
        .unwrap();
    writer.set_global_setting("encrypted", "true").unwrap();
    let provider = Arc::new(SqliteMetadataProvider::new(&url).await.unwrap());
    Lake {
        _temp: temp,
        url,
        writer,
        provider,
    }
}

impl Lake {
    fn session(&self) -> (SessionContext, DuckLakeCatalog) {
        let ctx = SessionContext::new();
        let writer: Arc<dyn MetadataWriter> = self.writer.clone();
        let catalog =
            DuckLakeCatalog::with_writer(Arc::clone(&self.provider), Arc::clone(&writer)).unwrap();
        DuckLakeCatalog::register(&ctx, "lake", catalog);
        let handle = DuckLakeCatalog::with_writer(Arc::clone(&self.provider), writer).unwrap();
        (ctx, handle)
    }

    async fn sql(&self, statement: &str) -> Result<(), String> {
        let (ctx, handle) = self.session();
        match execute_ducklake_sql(&ctx, &handle, statement).await {
            Ok(frame) => frame.collect().await.map(|_| ()).map_err(|e| e.to_string()),
            Err(e) => Err(e.to_string()),
        }
    }

    async fn table(&self) -> (SessionContext, DuckLakeTable) {
        let (ctx, _) = self.session();
        let provider = ctx
            .catalog("lake")
            .unwrap()
            .schema("main")
            .unwrap()
            .table("t")
            .await
            .unwrap()
            .unwrap();
        let table = (provider.as_ref() as &dyn std::any::Any)
            .downcast_ref::<DuckLakeTable>()
            .unwrap()
            .clone();
        (ctx, table)
    }

    fn head(&self) -> i64 {
        self.provider.get_current_snapshot().unwrap()
    }

    async fn data_files(&self) -> i64 {
        let pool = sqlx::SqlitePool::connect(&self.url).await.unwrap();
        sqlx::query_scalar("SELECT count(*) FROM ducklake_data_file")
            .fetch_one(&pool)
            .await
            .unwrap()
    }
}

fn refused(error: &str) {
    assert!(error.contains(REFUSAL), "{error}");
}

/// An insert over the inlining limit is refused, by SQL and by the Rust
/// writer; one within it is inlined.
#[tokio::test(flavor = "multi_thread")]
async fn inserts_over_the_limit_are_refused() {
    let lake = encrypted_lake(10).await;
    let head = lake.head();
    let values = (100..120)
        .map(|id| format!("({id})"))
        .collect::<Vec<_>>()
        .join(", ");
    refused(
        &lake
            .sql(&format!("INSERT INTO lake.main.t VALUES {values}"))
            .await
            .unwrap_err(),
    );
    let metadata: Arc<dyn MetadataWriter> = lake.writer.clone();
    let error = DuckLakeTableWriter::new(metadata, Arc::new(LocalFileSystem::new()))
        .unwrap()
        .with_options(&DuckLakeWriteOptions::default().with_data_inlining_row_limit(0))
        .append_table("main", "t", &[ids(100..102)])
        .await
        .unwrap_err();
    assert!(matches!(error, DuckLakeError::Unsupported(_)), "{error}");
    refused(&error.to_string());
    assert_eq!(lake.head(), head);
    assert_eq!(lake.data_files().await, 2);

    lake.sql("INSERT INTO lake.main.t VALUES (100), (101)")
        .await
        .unwrap();
    assert_eq!(lake.head(), head + 1);
    assert_eq!(lake.data_files().await, 2);
}

/// Flushing inlined rows, merging and rewriting files are refused.
#[tokio::test(flavor = "multi_thread")]
async fn flush_and_compaction_are_refused() {
    let lake = encrypted_lake(10).await;
    lake.sql("INSERT INTO lake.main.t VALUES (100)")
        .await
        .unwrap();
    let head = lake.head();
    refused(
        &lake
            .sql("CALL ducklake_flush_inlined_data('lake')")
            .await
            .unwrap_err(),
    );
    let (ctx, table) = lake.table().await;
    let merged = table
        .merge_adjacent_files(&ctx.state(), MergeOptions::default())
        .await
        .unwrap_err();
    refused(&merged.to_string());
    let rewritten = table
        .rewrite_data_files(
            &ctx.state(),
            RewriteOptions {
                delete_threshold: 0.0,
                data_file_ids: None,
            },
        )
        .await
        .unwrap_err();
    refused(&rewritten.to_string());
    assert_eq!(lake.head(), head);
}

/// A `DELETE` or `UPDATE` that writes a delete file or Parquet rows is refused;
/// one recorded as inlined deletions and inlined rows commits.
#[tokio::test(flavor = "multi_thread")]
async fn deletes_and_updates_writing_parquet_are_refused() {
    let lake = encrypted_lake(0).await;
    let head = lake.head();
    refused(
        &lake
            .sql("DELETE FROM lake.main.t WHERE id < 5")
            .await
            .unwrap_err(),
    );
    refused(
        &lake
            .sql("UPDATE lake.main.t SET id = id + 1000 WHERE id < 5")
            .await
            .unwrap_err(),
    );
    assert_eq!(lake.head(), head);

    let lake = encrypted_lake(10).await;
    let head = lake.head();
    lake.sql("DELETE FROM lake.main.t WHERE id < 5")
        .await
        .unwrap();
    lake.sql("UPDATE lake.main.t SET id = id + 1000 WHERE id = 30")
        .await
        .unwrap();
    assert_eq!(lake.head(), head + 2);
    assert_eq!(lake.data_files().await, 2);
}

/// `checkpoint` skips the phases that write Parquet on an encrypted catalog
/// and still expires snapshots and cleans up files.
#[tokio::test(flavor = "multi_thread")]
async fn checkpoint_skips_parquet_phases() {
    let lake = encrypted_lake(10).await;
    lake.sql("INSERT INTO lake.main.t VALUES (100)")
        .await
        .unwrap();
    lake.writer
        .set_global_setting("expire_older_than", "-1 day")
        .unwrap();
    let (ctx, handle) = lake.session();
    let result = handle.checkpoint(&ctx.state()).await.unwrap();
    assert!(result.flushed.is_empty());
    assert!(result.merged.is_empty());
    assert!(result.rewritten.is_empty());
    assert!(!result.expired.is_empty(), "{result:?}");
    assert_eq!(lake.data_files().await, 2);
}

/// The catalog's table and column rows and the snapshot head, which a refused
/// write must leave as they were.
async fn sqlite_catalog_state(url: &str) -> (i64, i64, i64, i64) {
    let pool = sqlx::SqlitePool::connect(url).await.unwrap();
    let count = |sql: &'static str| {
        let pool = pool.clone();
        async move {
            sqlx::query_scalar::<_, i64>(sql)
                .fetch_one(&pool)
                .await
                .unwrap()
        }
    };
    (
        count("SELECT count(*) FROM ducklake_schema").await,
        count("SELECT count(*) FROM ducklake_table").await,
        count("SELECT count(*) FROM ducklake_column").await,
        count("SELECT COALESCE(MAX(column_id), 0) FROM ducklake_column").await,
    )
}

/// A Parquet write into a table or schema that does not exist yet is refused
/// before anything is set up: no schema, table or column row is left behind
/// and no column id is used.
#[tokio::test(flavor = "multi_thread")]
async fn refused_write_to_a_new_table_leaves_no_rows() {
    let lake = encrypted_lake(10).await;
    let before = sqlite_catalog_state(&lake.url).await;
    let head = lake.head();
    for (schema, table) in [("main", "fresh"), ("other", "fresh")] {
        let metadata: Arc<dyn MetadataWriter> = lake.writer.clone();
        let error = DuckLakeTableWriter::new(metadata, Arc::new(LocalFileSystem::new()))
            .unwrap()
            .with_options(&DuckLakeWriteOptions::default().with_data_inlining_row_limit(0))
            .append_table(schema, table, &[ids(0..3)])
            .await
            .unwrap_err();
        refused(&error.to_string());
    }
    assert_eq!(sqlite_catalog_state(&lake.url).await, before);
    assert_eq!(lake.head(), head);
    // The table can still be created by an inlined write.
    let metadata: Arc<dyn MetadataWriter> = lake.writer.clone();
    DuckLakeTableWriter::new(metadata, Arc::new(LocalFileSystem::new()))
        .unwrap()
        .with_options(&DuckLakeWriteOptions::default().with_data_inlining_row_limit(10))
        .append_table("main", "fresh", &[ids(0..3)])
        .await
        .unwrap();
    assert_eq!(lake.head(), head + 1);
}

/// The catalog is encrypted when its global `encrypted` option is exactly
/// `true`, as official reads it; another value fails like official's attach,
/// and a schema- or table-scoped one is not the catalog's.
#[tokio::test(flavor = "multi_thread")]
async fn encrypted_option_is_global_and_exact() {
    let lake = encrypted_lake(10).await;
    let pool = sqlx::SqlitePool::connect(&lake.url).await.unwrap();
    let set = |value: &'static str| {
        let pool = pool.clone();
        async move {
            sqlx::query("UPDATE ducklake_metadata SET value = ? WHERE key = 'encrypted'")
                .bind(value)
                .execute(&pool)
                .await
                .unwrap();
        }
    };
    assert!(lake.writer.catalog_encrypted().unwrap());
    set("TRUE").await;
    let error = lake.writer.catalog_encrypted().unwrap_err();
    assert!(
        error
            .to_string()
            .contains("Encrypted should be either true or false"),
        "{error}"
    );
    set("false").await;
    assert!(!lake.writer.catalog_encrypted().unwrap());
    sqlx::query(
        "INSERT INTO ducklake_metadata (key, value, scope, scope_id)
         VALUES ('encrypted', 'true', 'table', 1)",
    )
    .execute(&pool)
    .await
    .unwrap();
    assert!(!lake.writer.catalog_encrypted().unwrap());
}

/// Flush, compaction and checkpoint read the `encrypted` option with the
/// writer's exact parse: a value other than `true` or `false` fails them as it fails every
/// other write, where a lenient match would take `TRUE` as encrypted and `yes`
/// as not.
#[tokio::test(flavor = "multi_thread")]
async fn flush_and_compaction_parse_the_option_exactly() {
    let lake = encrypted_lake(10).await;
    lake.sql("INSERT INTO lake.main.t VALUES (100)")
        .await
        .unwrap();
    let pool = sqlx::SqlitePool::connect(&lake.url).await.unwrap();
    for value in ["TRUE", "yes"] {
        sqlx::query("UPDATE ducklake_metadata SET value = ? WHERE key = 'encrypted'")
            .bind(value)
            .execute(&pool)
            .await
            .unwrap();
        let head = lake.head();
        let flushed = lake
            .sql("CALL ducklake_flush_inlined_data('lake')")
            .await
            .unwrap_err();
        assert!(
            flushed.contains("Encrypted should be either true or false"),
            "`{value}`: {flushed}"
        );
        let (ctx, table) = lake.table().await;
        let merged = table
            .merge_adjacent_files(&ctx.state(), MergeOptions::default())
            .await
            .unwrap_err()
            .to_string();
        assert!(
            merged.contains("Encrypted should be either true or false"),
            "`{value}`: {merged}"
        );
        let (ctx, handle) = lake.session();
        let checkpointed = handle
            .checkpoint(&ctx.state())
            .await
            .map(|result| format!("{result:?}"))
            .unwrap_err()
            .to_string();
        assert!(
            checkpointed.contains("Encrypted should be either true or false"),
            "`{value}`: {checkpointed}"
        );
        assert_eq!(lake.head(), head, "`{value}`: nothing committed");
    }
}

#[cfg(feature = "write-mysql")]
mod mysql_backend {
    use super::*;
    use datafusion_ducklake::MySqlMetadataWriter;
    use testcontainers::runners::AsyncRunner;
    use testcontainers_modules::mysql::Mysql;

    /// [`super::refused_write_to_a_new_table_leaves_no_rows`] on MySQL.
    #[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
    #[tokio::test(flavor = "multi_thread")]
    async fn refused_write_to_a_new_table_leaves_no_rows() {
        let container = Mysql::default().start().await.unwrap();
        let port = container.get_host_port_ipv4(3306).await.unwrap();
        let url = format!("mysql://root@127.0.0.1:{port}/test");
        let data = TempDir::new().unwrap();
        let writer = Arc::new(MySqlMetadataWriter::new_with_init(&url).await.unwrap());
        writer.set_data_path(data.path().to_str().unwrap()).unwrap();
        let pool = sqlx::MySqlPool::connect(&url).await.unwrap();
        sqlx::query(
            "INSERT INTO ducklake_metadata (`key`, `value`, scope, scope_id)
             VALUES ('encrypted', 'true', NULL, NULL)",
        )
        .execute(&pool)
        .await
        .unwrap();
        let state = || {
            let pool = pool.clone();
            async move {
                let mut out = Vec::new();
                for sql in [
                    "SELECT count(*) FROM ducklake_schema",
                    "SELECT count(*) FROM ducklake_table",
                    "SELECT count(*) FROM ducklake_column",
                    "SELECT COALESCE(MAX(snapshot_id), 0) FROM ducklake_snapshot",
                ] {
                    out.push(
                        sqlx::query_scalar::<_, i64>(sql)
                            .fetch_one(&pool)
                            .await
                            .unwrap(),
                    );
                }
                out
            }
        };
        let before = state().await;
        for (schema, table) in [("main", "fresh"), ("other", "fresh")] {
            let metadata: Arc<dyn MetadataWriter> = writer.clone();
            let error = DuckLakeTableWriter::new(metadata, Arc::new(LocalFileSystem::new()))
                .unwrap()
                .with_options(&DuckLakeWriteOptions::default().with_data_inlining_row_limit(0))
                .append_table(schema, table, &[ids(0..3)])
                .await
                .unwrap_err();
            refused(&error.to_string());
        }
        assert_eq!(state().await, before);
    }
}
