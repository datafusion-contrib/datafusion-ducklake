#![cfg(all(feature = "write-sqlite", feature = "metadata-duckdb"))]
//! The layout version a writer records, and what opening and migrating do
//! with a catalog an earlier release, a newer release or official DuckDB
//! created. Opening never migrates; `migrate` does, once, and only for a
//! catalog this crate created.

use datafusion_ducklake::catalog_version::{LAYOUT_VERSION, LAYOUT_VERSION_KEY};
use datafusion_ducklake::{DuckLakeError, SqliteMetadataWriter};
use sqlx::SqlitePool;
use tempfile::TempDir;

use crate::common::ensure_extension_installed;

fn url(temp: &TempDir) -> String {
    format!("sqlite:{}?mode=rwc", temp.path().join("lake.db").display())
}

/// The `ducklake_metadata` rows under `key`.
async fn rows(url: &str, key: &str) -> Vec<String> {
    let pool = SqlitePool::connect(url).await.unwrap();
    sqlx::query_scalar("SELECT value FROM ducklake_metadata WHERE key = ?")
        .bind(key)
        .fetch_all(&pool)
        .await
        .unwrap()
}

async fn execute(url: &str, statement: &str) {
    let pool = SqlitePool::connect(url).await.unwrap();
    sqlx::query(sqlx::AssertSqlSafe(statement))
        .execute(&pool)
        .await
        .unwrap();
}

/// Create a catalog with official DuckDB at `temp`, holding `main.t`.
fn official_catalog(temp: &TempDir) {
    ensure_extension_installed("ducklake");
    ensure_extension_installed("sqlite");
    let data = temp.path().join("data");
    std::fs::create_dir_all(&data).unwrap();
    let conn = duckdb::Connection::open_in_memory().unwrap();
    conn.execute_batch(&format!(
        "LOAD ducklake; LOAD sqlite;
         ATTACH 'ducklake:sqlite:{}' AS lake (DATA_PATH '{}/');
         CREATE TABLE lake.t (id INTEGER);
         INSERT INTO lake.t VALUES (1), (2);",
        temp.path().join("lake.db").display(),
        data.display()
    ))
    .unwrap();
}

/// The ids official DuckDB reads from `main.t` of the catalog at `temp`.
fn official_ids(temp: &TempDir) -> Vec<i32> {
    let conn = duckdb::Connection::open_in_memory().unwrap();
    conn.execute_batch(&format!(
        "LOAD ducklake; LOAD sqlite;
         ATTACH 'ducklake:sqlite:{}' AS lake;",
        temp.path().join("lake.db").display()
    ))
    .unwrap();
    let mut statement = conn.prepare("SELECT id FROM lake.t ORDER BY id").unwrap();
    statement
        .query_map([], |row| row.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap()
}

#[tokio::test(flavor = "multi_thread")]
async fn a_new_catalog_records_the_layout_version_and_not_officials() {
    let temp = TempDir::new().unwrap();
    let url = url(&temp);

    drop(SqliteMetadataWriter::new_with_init(&url).await.unwrap());
    SqliteMetadataWriter::new(&url).await.unwrap();

    assert_eq!(
        rows(&url, LAYOUT_VERSION_KEY).await,
        vec![LAYOUT_VERSION.to_string()]
    );
    assert_eq!(rows(&url, "version").await, Vec::<String>::new());
}

#[tokio::test(flavor = "multi_thread")]
async fn an_earlier_release_catalog_is_refused_until_migrated_once() {
    let temp = TempDir::new().unwrap();
    let url = url(&temp);
    {
        use datafusion_ducklake::MetadataWriter;
        let writer = SqliteMetadataWriter::new_with_init(&url).await.unwrap();
        writer.create_snapshot().unwrap();
    }
    execute(
        &url,
        &format!("DELETE FROM ducklake_metadata WHERE key = '{LAYOUT_VERSION_KEY}'"),
    )
    .await;

    let refused = SqliteMetadataWriter::new(&url).await;
    SqliteMetadataWriter::migrate(&url).await.unwrap();
    SqliteMetadataWriter::migrate(&url).await.unwrap();
    let reopened = SqliteMetadataWriter::new(&url).await;

    assert!(
        matches!(&refused, Err(DuckLakeError::MigrationRequired(message))
            if message.contains("SqliteMetadataWriter::migrate")),
        "{refused:?}"
    );
    assert!(reopened.is_ok(), "{reopened:?}");
    assert_eq!(
        rows(&url, LAYOUT_VERSION_KEY).await,
        vec![LAYOUT_VERSION.to_string()]
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_catalog_a_newer_release_wrote_is_refused() {
    let temp = TempDir::new().unwrap();
    let url = url(&temp);
    drop(SqliteMetadataWriter::new_with_init(&url).await.unwrap());
    execute(
        &url,
        &format!(
            "UPDATE ducklake_metadata SET value = '{}' WHERE key = '{LAYOUT_VERSION_KEY}'",
            LAYOUT_VERSION + 1
        ),
    )
    .await;

    let opened = SqliteMetadataWriter::new(&url).await;
    let migrated = SqliteMetadataWriter::migrate(&url).await;

    assert!(
        matches!(opened, Err(DuckLakeError::Unsupported(_))),
        "{opened:?}"
    );
    assert!(
        matches!(migrated, Err(DuckLakeError::Unsupported(_))),
        "{migrated:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn migrating_an_empty_database_is_refused() {
    let temp = TempDir::new().unwrap();

    let migrated = SqliteMetadataWriter::migrate(&url(&temp)).await;

    assert!(
        matches!(migrated, Err(DuckLakeError::InvalidConfig(_))),
        "{migrated:?}"
    );
}

/// Everything a SQLite catalog holds: its schema and its metadata rows.
async fn snapshot_of(url: &str) -> (Vec<String>, Vec<(String, String)>) {
    let pool = SqlitePool::connect(url).await.unwrap();
    let schema =
        sqlx::query_scalar("SELECT sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY 1")
            .fetch_all(&pool)
            .await
            .unwrap();
    let metadata = sqlx::query_as("SELECT key, value FROM ducklake_metadata ORDER BY 1, 2")
        .fetch_all(&pool)
        .await
        .unwrap();
    (schema, metadata)
}

/// A catalog official DuckDB created is refused by opening, initializing and
/// migrating alike, none of which changes it; official still reads it.
#[tokio::test(flavor = "multi_thread")]
async fn an_official_catalog_is_refused_and_left_unchanged() {
    let temp = TempDir::new().unwrap();
    let url = url(&temp);
    official_catalog(&temp);
    let before = snapshot_of(&url).await;

    let opened = SqliteMetadataWriter::new(&url).await;
    let initialized = SqliteMetadataWriter::new_with_init(&url).await;
    let migrated = SqliteMetadataWriter::migrate(&url).await;

    for (what, refused) in
        [("open", opened.map(drop)), ("init", initialized.map(drop)), ("migrate", migrated)]
    {
        assert!(
            matches!(&refused, Err(DuckLakeError::Unsupported(message))
                if message.contains("official DuckDB")),
            "{what}: {refused:?}"
        );
    }
    assert_eq!(snapshot_of(&url).await, before);
    assert_eq!(official_ids(&temp), vec![1, 2]);
}

/// A single-catalog writer refuses a database holding a multicatalog store.
#[tokio::test(flavor = "multi_thread")]
async fn a_single_catalog_writer_refuses_a_multicatalog_store() {
    let temp = TempDir::new().unwrap();
    let url = url(&temp);
    drop(SqliteMetadataWriter::new_with_init(&url).await.unwrap());
    execute(
        &url,
        "INSERT INTO ducklake_metadata (key, value) \
         VALUES ('datafusion_ducklake_multicatalog_version', '1')",
    )
    .await;

    let opened = SqliteMetadataWriter::new(&url).await;

    assert!(
        matches!(opened, Err(DuckLakeError::Unsupported(_))),
        "{opened:?}"
    );
}

/// A catalog whose creation stopped before it recorded its layout version is
/// refused like an earlier release's, and `migrate` completes it.
#[tokio::test(flavor = "multi_thread")]
async fn an_interrupted_creation_is_completed_by_migrate() {
    let temp = TempDir::new().unwrap();
    let url = url(&temp);
    execute(
        &url,
        "CREATE TABLE ducklake_metadata (key VARCHAR NOT NULL, value VARCHAR NOT NULL)",
    )
    .await;

    let initialized = SqliteMetadataWriter::new_with_init(&url).await;
    SqliteMetadataWriter::migrate(&url).await.unwrap();
    let reopened = SqliteMetadataWriter::new(&url).await;

    assert!(
        matches!(initialized, Err(DuckLakeError::MigrationRequired(_))),
        "{initialized:?}"
    );
    assert!(reopened.is_ok(), "{reopened:?}");
    assert_eq!(
        rows(&url, LAYOUT_VERSION_KEY).await,
        vec![LAYOUT_VERSION.to_string()]
    );
}

/// Initializing a catalog an earlier release created, which holds data, is
/// refused like opening it and changes nothing: only `migrate` migrates.
#[tokio::test(flavor = "multi_thread")]
async fn initializing_an_earlier_release_catalog_refuses_and_changes_nothing() {
    let temp = TempDir::new().unwrap();
    let url = url(&temp);
    {
        use datafusion_ducklake::MetadataWriter;
        let writer = SqliteMetadataWriter::new_with_init(&url).await.unwrap();
        writer.create_snapshot().unwrap();
    }
    execute(
        &url,
        &format!("DELETE FROM ducklake_metadata WHERE key = '{LAYOUT_VERSION_KEY}'"),
    )
    .await;
    let before = snapshot_of(&url).await;

    let initialized = SqliteMetadataWriter::new_with_init(&url).await;

    assert!(
        matches!(initialized, Err(DuckLakeError::MigrationRequired(_))),
        "{initialized:?}"
    );
    assert_eq!(snapshot_of(&url).await, before);
}

#[cfg(feature = "write-duckdb")]
#[test]
fn a_duckdb_catalog_from_an_earlier_release_is_refused_until_migrated() {
    use datafusion_ducklake::DuckdbMetadataWriter;
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("lake.ducklake").display().to_string();
    {
        use datafusion_ducklake::MetadataWriter;
        let writer = DuckdbMetadataWriter::new_with_init(path.clone()).unwrap();
        writer.create_snapshot().unwrap();
    }
    duckdb::Connection::open(&path)
        .unwrap()
        .execute(
            "DELETE FROM ducklake_metadata WHERE key = ?",
            [LAYOUT_VERSION_KEY],
        )
        .unwrap();

    let refused = DuckdbMetadataWriter::new(path.clone());
    DuckdbMetadataWriter::migrate(path.clone()).unwrap();
    let reopened = DuckdbMetadataWriter::new(path);

    assert!(
        matches!(refused, Err(DuckLakeError::MigrationRequired(_))),
        "{refused:?}"
    );
    assert!(reopened.is_ok(), "{reopened:?}");
}
