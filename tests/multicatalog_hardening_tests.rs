#![cfg(feature = "write-postgres")]
//! Hardening tests for the multicatalog Postgres writer.
//!
//! Covers the correctness gaps called out in PR review:
//! - Concurrent `get_or_create_schema` cannot create duplicate schemas
//! - Concurrent `get_or_create_table` cannot create duplicate tables
//! - Cross-catalog `register_data_file` is rejected
//! - Cross-catalog `end_table_files` is rejected
//! - Cross-catalog `set_columns` is rejected
//! - `get_or_create_table` rejects schema_id from a different catalog
//! - All writers serialize per catalog but run in parallel across catalogs

use std::sync::Arc;

use datafusion_ducklake::metadata_writer::{ColumnDef, DataFileInfo, MetadataWriter, WriteMode};
use datafusion_ducklake::{
    MulticatalogManager, PostgresMetadataWriter, initialize_multicatalog_schema,
};
use sqlx::Row;
use sqlx::postgres::{PgPool, PgPoolOptions};
use testcontainers::ContainerAsync;
use testcontainers::runners::AsyncRunner;
use testcontainers_modules::postgres::Postgres;

async fn spin_up_postgres() -> anyhow::Result<(PgPool, ContainerAsync<Postgres>)> {
    let container = Postgres::default().start().await?;
    let port = container.get_host_port_ipv4(5432).await?;
    let conn = format!("postgresql://postgres:postgres@127.0.0.1:{}/postgres", port);
    let pool = PgPoolOptions::new()
        // Need >=N+1 connections for N concurrent writers + this test's queries
        .max_connections(20)
        .connect(&conn)
        .await?;
    initialize_multicatalog_schema(&pool).await?;
    Ok((pool, container))
}

fn users_cols() -> Vec<ColumnDef> {
    vec![
        ColumnDef::new("id", "int64", false).unwrap(),
        ColumnDef::new("name", "varchar", true).unwrap(),
    ]
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn concurrent_get_or_create_schema_no_duplicates() {
    let (pool, _c) = spin_up_postgres().await.unwrap();
    let mgr = MulticatalogManager::new(pool.clone());
    let catalog_id = mgr.create_catalog("pg_prod").await.unwrap();
    let writer = Arc::new(
        PostgresMetadataWriter::with_pool(pool.clone(), catalog_id)
            .await
            .unwrap(),
    );

    // Allocate a snapshot they all share so the schema row inserts have a valid begin_snapshot.
    let snapshot_id = writer.create_snapshot().unwrap();

    // 10 concurrent writers all racing to create a schema with the same name.
    let mut handles = Vec::new();
    for _ in 0..10 {
        let w = Arc::clone(&writer);
        handles.push(tokio::task::spawn_blocking(move || {
            w.get_or_create_schema("racy", None, snapshot_id)
        }));
    }

    let mut ids = Vec::new();
    let mut created_count = 0;
    for h in handles {
        let (id, was_created) = h.await.unwrap().unwrap();
        ids.push(id);
        if was_created {
            created_count += 1;
        }
    }

    // All callers should see the SAME schema_id, exactly one should report "created".
    let first = ids[0];
    assert!(
        ids.iter().all(|&x| x == first),
        "all callers must see the same schema_id, got {:?}",
        ids
    );
    assert_eq!(
        created_count, 1,
        "exactly one writer should report was_created"
    );

    // And the catalog only has one schema row.
    let n: i64 = sqlx::query(
        "SELECT COUNT(*) FROM ducklake_schema s
         JOIN ducklake_catalog_schema_map m ON m.schema_id = s.schema_id
         WHERE m.catalog_id = $1 AND s.schema_name = 'racy'",
    )
    .bind(catalog_id)
    .fetch_one(&pool)
    .await
    .unwrap()
    .try_get(0)
    .unwrap();
    assert_eq!(n, 1, "should be exactly one 'racy' schema row");
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn concurrent_get_or_create_table_no_duplicates() {
    let (pool, _c) = spin_up_postgres().await.unwrap();
    let mgr = MulticatalogManager::new(pool.clone());
    let catalog_id = mgr.create_catalog("pg_prod").await.unwrap();
    let writer = Arc::new(
        PostgresMetadataWriter::with_pool(pool.clone(), catalog_id)
            .await
            .unwrap(),
    );
    let snapshot_id = writer.create_snapshot().unwrap();
    let (schema_id, _) = writer
        .get_or_create_schema("public", None, snapshot_id)
        .unwrap();

    let mut handles = Vec::new();
    for _ in 0..10 {
        let w = Arc::clone(&writer);
        handles.push(tokio::task::spawn_blocking(move || {
            w.get_or_create_table(schema_id, "users", None, snapshot_id)
        }));
    }

    let mut ids = Vec::new();
    let mut created_count = 0;
    for h in handles {
        let (id, was_created) = h.await.unwrap().unwrap();
        ids.push(id);
        if was_created {
            created_count += 1;
        }
    }

    let first = ids[0];
    assert!(
        ids.iter().all(|&x| x == first),
        "all callers must see the same table_id"
    );
    assert_eq!(created_count, 1);

    let n: i64 = sqlx::query(
        "SELECT COUNT(*) FROM ducklake_table
         WHERE schema_id = $1 AND table_name = 'users'",
    )
    .bind(schema_id)
    .fetch_one(&pool)
    .await
    .unwrap()
    .try_get(0)
    .unwrap();
    assert_eq!(n, 1);
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn different_catalogs_proceed_in_parallel_no_serialization() {
    // The FOR UPDATE lock is per-catalog, so two catalogs writing simultaneously
    // should both succeed. This test just sanity-checks that we don't accidentally
    // serialize across catalogs.
    let (pool, _c) = spin_up_postgres().await.unwrap();
    let mgr = MulticatalogManager::new(pool.clone());
    let cat_a = mgr.create_catalog("cat_a").await.unwrap();
    let cat_b = mgr.create_catalog("cat_b").await.unwrap();

    let wa = Arc::new(
        PostgresMetadataWriter::with_pool(pool.clone(), cat_a)
            .await
            .unwrap(),
    );
    let wb = Arc::new(
        PostgresMetadataWriter::with_pool(pool.clone(), cat_b)
            .await
            .unwrap(),
    );
    wa.set_data_path("/data").unwrap();

    // Both writers run a full begin_write_transaction concurrently.
    let wa_clone = Arc::clone(&wa);
    let wb_clone = Arc::clone(&wb);
    let ha = tokio::task::spawn_blocking(move || {
        wa_clone.begin_write_transaction("public", "users", &users_cols(), WriteMode::Replace)
    });
    let hb = tokio::task::spawn_blocking(move || {
        wb_clone.begin_write_transaction("public", "orders", &users_cols(), WriteMode::Replace)
    });
    let res_a = ha.await.unwrap().unwrap();
    let res_b = hb.await.unwrap().unwrap();

    assert_ne!(res_a.schema_id, res_b.schema_id);
    assert_ne!(res_a.table_id, res_b.table_id);
}

// ── cross-catalog ownership checks ────────────────────────────────────────────

async fn seed_two_catalogs(pool: &PgPool) -> (i64, i64, i64, i64) {
    let mgr = MulticatalogManager::new(pool.clone());
    let cat_a = mgr.create_catalog("cat_a").await.unwrap();
    let cat_b = mgr.create_catalog("cat_b").await.unwrap();
    let wa = PostgresMetadataWriter::with_pool(pool.clone(), cat_a)
        .await
        .unwrap();
    let wb = PostgresMetadataWriter::with_pool(pool.clone(), cat_b)
        .await
        .unwrap();
    wa.set_data_path("/data").unwrap();

    let setup_a = wa
        .begin_write_transaction("public", "users", &users_cols(), WriteMode::Replace)
        .unwrap();
    let setup_b = wb
        .begin_write_transaction("public", "orders", &users_cols(), WriteMode::Replace)
        .unwrap();
    (cat_a, cat_b, setup_a.table_id, setup_b.table_id)
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn register_data_file_rejects_cross_catalog_table_id() {
    let (pool, _c) = spin_up_postgres().await.unwrap();
    let (cat_a, _cat_b, _table_a, table_b) = seed_two_catalogs(&pool).await;
    let wa = PostgresMetadataWriter::with_pool(pool.clone(), cat_a)
        .await
        .unwrap();
    let snap = wa.create_snapshot().unwrap();

    let result = wa.register_data_file(
        table_b, // ← belongs to cat_b
        snap,
        &DataFileInfo::new("evil.parquet", 1024, 1),
    );
    let err = result.expect_err("must reject cross-catalog table_id");
    assert!(
        err.to_string().contains("does not belong to catalog"),
        "expected ownership error, got: {}",
        err
    );

    // And no row was inserted.
    let n: i64 = sqlx::query("SELECT COUNT(*) FROM ducklake_data_file WHERE path = 'evil.parquet'")
        .fetch_one(&pool)
        .await
        .unwrap()
        .try_get(0)
        .unwrap();
    assert_eq!(n, 0);
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn end_table_files_rejects_cross_catalog_table_id() {
    let (pool, _c) = spin_up_postgres().await.unwrap();
    let (cat_a, _cat_b, _table_a, table_b) = seed_two_catalogs(&pool).await;
    let wa = PostgresMetadataWriter::with_pool(pool.clone(), cat_a)
        .await
        .unwrap();
    let snap = wa.create_snapshot().unwrap();

    let result = wa.end_table_files(table_b, snap);
    let err = result.expect_err("must reject cross-catalog table_id");
    assert!(
        err.to_string().contains("does not belong to catalog"),
        "expected ownership error, got: {}",
        err
    );
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn set_columns_rejects_cross_catalog_table_id() {
    let (pool, _c) = spin_up_postgres().await.unwrap();
    let (cat_a, _cat_b, _table_a, table_b) = seed_two_catalogs(&pool).await;
    let wa = PostgresMetadataWriter::with_pool(pool.clone(), cat_a)
        .await
        .unwrap();
    let snap = wa.create_snapshot().unwrap();

    let result = wa.set_columns(table_b, &users_cols(), snap);
    let err = result.expect_err("must reject cross-catalog table_id");
    assert!(
        err.to_string().contains("does not belong to catalog"),
        "expected ownership error, got: {}",
        err
    );
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn get_or_create_table_rejects_cross_catalog_schema_id() {
    let (pool, _c) = spin_up_postgres().await.unwrap();
    let mgr = MulticatalogManager::new(pool.clone());
    let cat_a = mgr.create_catalog("cat_a").await.unwrap();
    let cat_b = mgr.create_catalog("cat_b").await.unwrap();
    let wa = PostgresMetadataWriter::with_pool(pool.clone(), cat_a)
        .await
        .unwrap();
    let wb = PostgresMetadataWriter::with_pool(pool.clone(), cat_b)
        .await
        .unwrap();
    wa.set_data_path("/data").unwrap();

    // Set up a schema in each catalog.
    let snap_a = wa.create_snapshot().unwrap();
    let (schema_a, _) = wa.get_or_create_schema("public", None, snap_a).unwrap();
    let snap_b = wb.create_snapshot().unwrap();
    let (schema_b, _) = wb.get_or_create_schema("public", None, snap_b).unwrap();

    // cat_a's writer must reject schema_b.
    let result = wa.get_or_create_table(schema_b, "users", None, snap_a);
    let err = result.expect_err("must reject cross-catalog schema_id");
    assert!(
        err.to_string().contains("does not belong to catalog"),
        "expected ownership error, got: {}",
        err
    );

    // Own schema works.
    let ok = wa.get_or_create_table(schema_a, "users", None, snap_a);
    assert!(ok.is_ok());
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn set_data_path_rejects_silent_overwrite() {
    let (pool, _c) = spin_up_postgres().await.unwrap();
    let mgr = MulticatalogManager::new(pool.clone());
    let cat = mgr.create_catalog("pg_prod").await.unwrap();
    let w = PostgresMetadataWriter::with_pool(pool.clone(), cat)
        .await
        .unwrap();

    w.set_data_path("/data/a").unwrap();
    // Same value is idempotent.
    w.set_data_path("/data/a").unwrap();
    // Different value rejected.
    let err = w
        .set_data_path("/data/b")
        .expect_err("must reject overwrite");
    assert!(err.to_string().contains("already set"), "got: {}", err);
    // Original value is untouched.
    assert_eq!(w.get_data_path().unwrap(), "/data/a");
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn rollback_leaves_no_orphan_rows() {
    // begin_write_transaction with an invalid column type triggers a rollback
    // mid-write. The snapshot/mapping/schema rows it would have inserted must
    // not survive.
    let (pool, _c) = spin_up_postgres().await.unwrap();
    let mgr = MulticatalogManager::new(pool.clone());
    let cat = mgr.create_catalog("pg_prod").await.unwrap();
    let w = PostgresMetadataWriter::with_pool(pool.clone(), cat)
        .await
        .unwrap();
    w.set_data_path("/data").unwrap();

    let before_snaps: i64 = sqlx::query("SELECT COUNT(*) FROM ducklake_snapshot")
        .fetch_one(&pool)
        .await
        .unwrap()
        .try_get(0)
        .unwrap();
    let before_map: i64 = sqlx::query("SELECT COUNT(*) FROM ducklake_catalog_snapshot_map")
        .fetch_one(&pool)
        .await
        .unwrap()
        .try_get(0)
        .unwrap();

    let _ok = w
        .begin_write_transaction("public", "users", &users_cols(), WriteMode::Replace)
        .unwrap();
    // Incompatible type change in Append fails after the snapshot is inserted.
    let bad_cols = vec![
        ColumnDef::new("id", "varchar", false).unwrap(),
        ColumnDef::new("name", "varchar", true).unwrap(),
    ];
    let snaps_before_fail: i64 = sqlx::query("SELECT COUNT(*) FROM ducklake_snapshot")
        .fetch_one(&pool)
        .await
        .unwrap()
        .try_get(0)
        .unwrap();
    let err = w
        .begin_write_transaction("public", "users", &bad_cols, WriteMode::Append)
        .expect_err("incompatible type change must fail");
    assert!(err.to_string().contains("Schema evolution error"));

    // The failed call must not have left a snapshot behind.
    let snaps_after_fail: i64 = sqlx::query("SELECT COUNT(*) FROM ducklake_snapshot")
        .fetch_one(&pool)
        .await
        .unwrap()
        .try_get(0)
        .unwrap();
    assert_eq!(
        snaps_after_fail, snaps_before_fail,
        "rollback should leave snapshot count unchanged"
    );

    // And no orphan map rows.
    let orphans: i64 = sqlx::query(
        "SELECT COUNT(*) FROM ducklake_catalog_snapshot_map m
         LEFT JOIN ducklake_snapshot s ON s.snapshot_id = m.snapshot_id
         WHERE s.snapshot_id IS NULL",
    )
    .fetch_one(&pool)
    .await
    .unwrap()
    .try_get(0)
    .unwrap();
    assert_eq!(orphans, 0);

    // And the original write's snapshot/map rows survived.
    assert!(
        sqlx::query("SELECT COUNT(*) FROM ducklake_snapshot")
            .fetch_one(&pool)
            .await
            .unwrap()
            .try_get::<i64, _>(0)
            .unwrap()
            > before_snaps,
        "the original (good) write should still be in place"
    );
    let _ = before_map;
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn partial_unique_index_blocks_duplicate_active_table() {
    // The app-level lock prevents this in normal use; verify the index would
    // catch it if someone bypasses the writer with raw SQL.
    let (pool, _c) = spin_up_postgres().await.unwrap();
    let mgr = MulticatalogManager::new(pool.clone());
    let cat = mgr.create_catalog("pg_prod").await.unwrap();
    let w = PostgresMetadataWriter::with_pool(pool.clone(), cat)
        .await
        .unwrap();
    w.set_data_path("/data").unwrap();
    let setup = w
        .begin_write_transaction("public", "users", &users_cols(), WriteMode::Replace)
        .unwrap();

    // Raw SQL insert of a second active row with the same (schema_id, name).
    let err = sqlx::query(
        "INSERT INTO ducklake_table (schema_id, table_name, path, path_is_relative, begin_snapshot)
         VALUES ($1, 'users', 'users', TRUE, $2)",
    )
    .bind(setup.schema_id)
    .bind(setup.snapshot_id)
    .execute(&pool)
    .await
    .expect_err("partial unique index should reject duplicate active table");
    let msg = err.to_string();
    assert!(
        msg.contains("idx_active_table_per_schema")
            || msg.contains("duplicate key")
            || msg.contains("unique"),
        "expected unique-violation error, got: {}",
        msg
    );
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn unknown_catalog_id_errors_clearly_on_lock() {
    // Construct a writer with a bogus catalog_id (skip the manager) and ensure
    // its first mutation surfaces CatalogNotFound rather than a generic SQL error.
    let (pool, _c) = spin_up_postgres().await.unwrap();
    let bogus = PostgresMetadataWriter::with_pool(pool.clone(), 999_999)
        .await
        .unwrap();
    // Snapshot create doesn't take the lock (it's a naked insert + map),
    // but the map insert will succeed because there's no FK to ducklake_catalog.
    // The lock-taking methods are the ones that must reject. begin_write_transaction
    // takes the lock first:
    let result =
        bogus.begin_write_transaction("public", "users", &users_cols(), WriteMode::Replace);
    let err = result.expect_err("bogus catalog_id should error");
    assert!(
        err.to_string().contains("999999")
            || err.to_string().contains("not found")
            || err.to_string().to_lowercase().contains("catalog"),
        "expected catalog-related error, got: {}",
        err
    );
}
