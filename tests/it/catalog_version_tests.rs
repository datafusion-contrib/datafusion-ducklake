#![cfg(all(feature = "write-sqlite", feature = "metadata-duckdb"))]
//! The layout version a writer records, and what opening and migrating do
//! with a catalog an earlier release, a newer release or official DuckDB
//! created. Opening never migrates; `migrate` does, once, and only for a
//! catalog this crate created.

use std::sync::Arc;

use datafusion::prelude::SessionContext;
use datafusion_ducklake::catalog_version::{
    LAYOUT_VERSION, LAYOUT_VERSION_KEY, MIN_VERSION, MIN_VERSION_KEY,
};
use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeError, SqliteMetadataProvider, SqliteMetadataWriter,
};
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

/// Record `version` as the catalog's layout version and `min_version` as its
/// minimum version, or none, as a release at `version` would.
async fn stamp(url: &str, version: u64, min_version: Option<u64>) {
    let pool = SqlitePool::connect(url).await.unwrap();
    for key in [LAYOUT_VERSION_KEY, MIN_VERSION_KEY] {
        sqlx::query("DELETE FROM ducklake_metadata WHERE key = ?")
            .bind(key)
            .execute(&pool)
            .await
            .unwrap();
    }
    let rows = std::iter::once((LAYOUT_VERSION_KEY, version))
        .chain(min_version.map(|min_version| (MIN_VERSION_KEY, min_version)));
    for (key, value) in rows {
        sqlx::query("INSERT INTO ducklake_metadata (key, value) VALUES (?, ?)")
            .bind(key)
            .bind(value.to_string())
            .execute(&pool)
            .await
            .unwrap();
    }
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
    assert_eq!(
        rows(&url, MIN_VERSION_KEY).await,
        vec![MIN_VERSION.to_string()]
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

/// A catalog a newer release stamped with a minimum version above this
/// release's, or with none, is refused by opening, initializing and migrating.
#[tokio::test(flavor = "multi_thread")]
async fn a_catalog_a_newer_release_wrote_is_refused() {
    for min_version in [Some(LAYOUT_VERSION + 1), None] {
        let temp = TempDir::new().unwrap();
        let url = url(&temp);
        drop(SqliteMetadataWriter::new_with_init(&url).await.unwrap());
        stamp(&url, LAYOUT_VERSION + 1, min_version).await;

        let opened = SqliteMetadataWriter::new(&url).await;
        let initialized = SqliteMetadataWriter::new_with_init(&url).await;
        let migrated = SqliteMetadataWriter::migrate(&url).await;

        for (what, refused) in
            [("open", opened.map(drop)), ("init", initialized.map(drop)), ("migrate", migrated)]
        {
            assert!(
                matches!(&refused, Err(DuckLakeError::Unsupported(message))
                    if message.contains("newer than this release writes")),
                "{min_version:?} {what}: {refused:?}"
            );
        }
    }
}

/// A catalog a newer release stamped, whose minimum version admits
/// this release (an addition this release can ignore, such as an index), is
/// opened and written as a current one, so a rolling deploy or a rollback
/// across the newer release keeps writing. Migrating it changes nothing, so
/// this release never records its older version over the newer one (#405).
#[tokio::test(flavor = "multi_thread")]
async fn a_newer_catalog_this_release_may_write_is_written_and_left_at_its_version() {
    use datafusion_ducklake::MetadataWriter;
    let temp = TempDir::new().unwrap();
    let url = url(&temp);
    let data = temp.path().join("data");
    std::fs::create_dir_all(&data).unwrap();
    {
        let writer = SqliteMetadataWriter::new_with_init(&url).await.unwrap();
        writer.set_data_path(data.to_str().unwrap()).unwrap();
        let snapshot_id = writer.create_snapshot().unwrap();
        writer
            .get_or_create_schema("main", None, snapshot_id)
            .unwrap();
    }
    stamp(&url, LAYOUT_VERSION + 1, Some(LAYOUT_VERSION)).await;

    let initialized = SqliteMetadataWriter::new_with_init(&url).await.map(drop);
    let migrated = SqliteMetadataWriter::migrate(&url).await;
    let writer = SqliteMetadataWriter::new(&url).await.unwrap();
    let provider = SqliteMetadataProvider::new(&url).await.unwrap();
    let catalog = DuckLakeCatalog::with_writer(Arc::new(provider), Arc::new(writer)).unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("lake", Arc::new(catalog));
    for statement in
        ["CREATE TABLE lake.main.t (id INT)", "INSERT INTO lake.main.t VALUES (1), (2)"]
    {
        ctx.sql(statement).await.unwrap().collect().await.unwrap();
    }
    let ids = ctx
        .sql("SELECT id FROM lake.main.t ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    assert!(initialized.is_ok(), "{initialized:?}");
    assert!(migrated.is_ok(), "{migrated:?}");
    assert_eq!(
        arrow::util::pretty::pretty_format_batches(&ids)
            .unwrap()
            .to_string(),
        "+----+\n| id |\n+----+\n| 1  |\n| 2  |\n+----+"
    );
    assert_eq!(
        rows(&url, LAYOUT_VERSION_KEY).await,
        vec![(LAYOUT_VERSION + 1).to_string()]
    );
    assert_eq!(
        rows(&url, MIN_VERSION_KEY).await,
        vec![LAYOUT_VERSION.to_string()]
    );
}

/// What each way of reading the catalog at `url` returns: a provider opened
/// with `new` listing its snapshot, and one adopting a pool beginning a
/// consistent read and backing a [`DuckLakeCatalog`].
async fn reads(url: &str) -> Vec<Result<(), DuckLakeError>> {
    use datafusion_ducklake::MetadataProvider;
    let opened = SqliteMetadataProvider::new(url).await.unwrap();
    let adopted = || SqliteMetadataProvider::from_pool(SqlitePool::connect_lazy(url).unwrap());
    vec![
        opened.get_current_snapshot().map(drop),
        adopted().begin_consistent_read().map(drop),
        DuckLakeCatalog::new(adopted()).map(drop),
    ]
}

/// A reader refuses a catalog a newer release wrote whose minimum version is
/// above this release's, or that records none, however its provider was built.
/// It reads every other catalog as it is: a newer one whose minimum version
/// admits this release, and one an earlier release created without a version.
#[tokio::test(flavor = "multi_thread")]
async fn a_reader_refuses_only_a_catalog_whose_minimum_version_is_above_this_release() {
    use datafusion_ducklake::MetadataWriter;
    let temp = TempDir::new().unwrap();
    let url = url(&temp);
    let data = temp.path().join("data");
    std::fs::create_dir_all(&data).unwrap();
    {
        let writer = SqliteMetadataWriter::new_with_init(&url).await.unwrap();
        writer.set_data_path(data.to_str().unwrap()).unwrap();
        writer.create_snapshot().unwrap();
    }

    let mut refused = Vec::new();
    for min_version in [Some(LAYOUT_VERSION + 1), None] {
        stamp(&url, LAYOUT_VERSION + 1, min_version).await;
        refused.extend(reads(&url).await);
    }
    stamp(&url, LAYOUT_VERSION + 1, Some(LAYOUT_VERSION)).await;
    let compatible = reads(&url).await;
    execute(
        &url,
        &format!(
            "DELETE FROM ducklake_metadata WHERE key IN ('{LAYOUT_VERSION_KEY}', '{MIN_VERSION_KEY}')"
        ),
    )
    .await;
    let unversioned = reads(&url).await;

    assert_eq!(refused.len(), 6);
    for read in refused {
        assert!(
            matches!(&read, Err(DuckLakeError::Unsupported(message))
                if message.contains("newer than this release reads")),
            "{read:?}"
        );
    }
    for read in compatible.into_iter().chain(unversioned) {
        assert!(read.is_ok(), "{read:?}");
    }
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

#[cfg(feature = "write-duckdb")]
#[test]
fn a_newer_duckdb_catalog_this_release_may_write_is_written_and_left_at_its_version() {
    use datafusion_ducklake::{
        DuckdbMetadataProvider, DuckdbMetadataWriter, MetadataProvider, MetadataWriter,
    };
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("lake.ducklake").display().to_string();
    drop(DuckdbMetadataWriter::new_with_init(path.clone()).unwrap());
    let stamped = |key: &str| -> Vec<String> {
        let conn = duckdb::Connection::open(&path).unwrap();
        let mut statement = conn
            .prepare("SELECT value FROM ducklake_metadata WHERE key = ?")
            .unwrap();
        statement
            .query_map([key], |row| row.get(0))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap()
    };
    let stamp = |min_version: u64| {
        let conn = duckdb::Connection::open(&path).unwrap();
        conn.execute(
            "DELETE FROM ducklake_metadata WHERE key IN (?, ?)",
            [LAYOUT_VERSION_KEY, MIN_VERSION_KEY],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO ducklake_metadata (key, value) VALUES (?, ?), (?, ?)",
            [
                LAYOUT_VERSION_KEY,
                &(LAYOUT_VERSION + 1).to_string(),
                MIN_VERSION_KEY,
                &min_version.to_string(),
            ],
        )
        .unwrap();
    };
    let created = (stamped(LAYOUT_VERSION_KEY), stamped(MIN_VERSION_KEY));
    stamp(LAYOUT_VERSION);

    let initialized = DuckdbMetadataWriter::new_with_init(path.clone()).map(drop);
    let migrated = DuckdbMetadataWriter::migrate(path.clone());
    let committed = DuckdbMetadataWriter::new(path.clone()).and_then(|w| w.create_snapshot());
    let versions = (stamped(LAYOUT_VERSION_KEY), stamped(MIN_VERSION_KEY));
    let read = || DuckdbMetadataProvider::new(path.clone()).and_then(|p| p.get_current_snapshot());
    let compatible_read = read();
    stamp(LAYOUT_VERSION + 1);
    let refused = DuckdbMetadataWriter::new(path.clone()).map(drop);
    let refused_read = read();

    assert_eq!(
        created,
        (
            vec![LAYOUT_VERSION.to_string()],
            vec![MIN_VERSION.to_string()]
        )
    );
    assert!(initialized.is_ok(), "{initialized:?}");
    assert!(migrated.is_ok(), "{migrated:?}");
    assert!(committed.is_ok(), "{committed:?}");
    assert!(compatible_read.is_ok(), "{compatible_read:?}");
    assert_eq!(
        versions,
        (
            vec![(LAYOUT_VERSION + 1).to_string()],
            vec![LAYOUT_VERSION.to_string()]
        )
    );
    assert!(
        matches!(refused, Err(DuckLakeError::Unsupported(_))),
        "{refused:?}"
    );
    assert!(
        matches!(&refused_read, Err(DuckLakeError::Unsupported(message))
            if message.contains("newer than this release reads")),
        "{refused_read:?}"
    );
}
