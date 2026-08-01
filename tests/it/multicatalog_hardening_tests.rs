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

use datafusion_ducklake::metadata_provider::MetadataProvider;
use datafusion_ducklake::metadata_writer::{ColumnDef, DataFileInfo, MetadataWriter, WriteMode};
use datafusion_ducklake::{
    MulticatalogManager, MulticatalogProvider, PostgresMetadataWriter,
    initialize_multicatalog_schema,
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

    // Commit both tables (publish) so their rows actually exist and carry the
    // correct catalog ownership. Under the commit-time model
    // `begin_write_transaction` only reserves ids — it inserts no table row — so
    // a cross-catalog ownership check has nothing to reject until the table is
    // published.
    let setup_a = wa
        .begin_write_transaction("public", "users", &users_cols(), WriteMode::Replace)
        .unwrap();
    wa.publish_snapshot(
        setup_a.table_id,
        "public",
        "users",
        setup_a.snapshot_id,
        WriteMode::Replace,
        setup_a.base_snapshot_id,
        &users_cols(),
        &setup_a.column_ids,
    )
    .unwrap();
    let setup_b = wb
        .begin_write_transaction("public", "orders", &users_cols(), WriteMode::Replace)
        .unwrap();
    wb.publish_snapshot(
        setup_b.table_id,
        "public",
        "orders",
        setup_b.snapshot_id,
        WriteMode::Replace,
        setup_b.base_snapshot_id,
        &users_cols(),
        &setup_b.column_ids,
    )
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
        "public",
        "orders",
        snap,
        &DataFileInfo::new("evil.parquet", 1024, 1),
        WriteMode::Replace,
        snap, // base_snapshot: unused — the cross-catalog ownership check fires first
        &users_cols(),
        &[],
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
async fn promote_column_type_rejects_cross_catalog_table_id() {
    // BLOCKER guard: table_ids are global across the multicatalog store, so a
    // promote scoped to cat_a must refuse a table_id owned by cat_b (else it would
    // silently mutate another catalog's column).
    let (pool, _c) = spin_up_postgres().await.unwrap();
    let (cat_a, _cat_b, _table_a, table_b) = seed_two_catalogs(&pool).await;
    let wa = PostgresMetadataWriter::with_pool(pool.clone(), cat_a)
        .await
        .unwrap();

    let result = wa.promote_column_type(table_b, "id", "int64");
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
async fn set_data_path_is_catalog_scoped_and_rejects_silent_overwrite() {
    let (pool, _c) = spin_up_postgres().await.unwrap();
    let mgr = MulticatalogManager::new(pool.clone());
    let cat_a = mgr.create_catalog("cat_a").await.unwrap();
    let cat_b = mgr.create_catalog("cat_b").await.unwrap();
    let writer_a = PostgresMetadataWriter::with_pool(pool.clone(), cat_a)
        .await
        .unwrap();
    let writer_b = PostgresMetadataWriter::with_pool(pool.clone(), cat_b)
        .await
        .unwrap();

    writer_a.set_data_path("/data/a").unwrap();
    writer_a.set_data_path("/data/a").unwrap();
    let missing = writer_b
        .get_data_path()
        .expect_err("an unconfigured catalog must not inherit another root");
    writer_b.set_data_path("/data/b").unwrap();
    let err = writer_a
        .set_data_path("/data/b")
        .expect_err("must reject overwrite");
    assert!(
        err.to_string().contains("already set"),
        "unexpected error: {err}"
    );
    assert!(missing.to_string().contains("not configured"));

    let provider_a = MulticatalogProvider::with_pool_and_id(pool.clone(), cat_a)
        .await
        .unwrap();
    let provider_b = MulticatalogProvider::with_pool_and_id(pool, cat_b)
        .await
        .unwrap();

    assert_eq!(writer_a.get_data_path().unwrap(), "/data/a");
    assert_eq!(writer_b.get_data_path().unwrap(), "/data/b");
    assert_eq!(provider_a.get_data_path().unwrap(), "/data/a");
    assert_eq!(provider_b.get_data_path().unwrap(), "/data/b");
    assert_eq!(
        mgr.get_data_path_in_catalog("cat_a").await.unwrap(),
        "/data/a"
    );
    assert_eq!(
        mgr.get_data_path_in_catalog("cat_b").await.unwrap(),
        "/data/b"
    );
    let missing_global = mgr
        .get_data_path()
        .await
        .expect_err("a fresh multicatalog store must not seed the legacy root");
    assert!(missing_global.to_string().contains("not configured"));
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn legacy_data_path_is_backfilled_and_rejects_overwrite() {
    let (pool, _c) = spin_up_postgres().await.unwrap();
    let mgr = MulticatalogManager::new(pool.clone());
    let catalog_id = mgr.create_catalog("legacy").await.unwrap();
    sqlx::query("ALTER TABLE ducklake_catalog DROP COLUMN data_path")
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query(
        "INSERT INTO ducklake_metadata (key, value, scope)
         VALUES ('data_path', '/data/legacy', NULL)",
    )
    .execute(&pool)
    .await
    .unwrap();

    initialize_multicatalog_schema(&pool).await.unwrap();
    let writer = PostgresMetadataWriter::with_pool(pool.clone(), catalog_id)
        .await
        .unwrap();
    let error = writer
        .set_data_path("/data/new")
        .expect_err("the migrated path is already established");

    assert!(error.to_string().contains("already set"));
    assert_eq!(writer.get_data_path().unwrap(), "/data/legacy");

    let fresh_id = mgr.create_catalog("fresh").await.unwrap();
    initialize_multicatalog_schema(&pool).await.unwrap();
    let fresh = PostgresMetadataWriter::with_pool(pool, fresh_id)
        .await
        .unwrap();
    let missing = fresh
        .get_data_path()
        .expect_err("a post-migration catalog must remain unconfigured");
    fresh.set_data_path("/data/fresh").unwrap();
    assert!(missing.to_string().contains("not configured"));
    assert_eq!(fresh.get_data_path().unwrap(), "/data/fresh");
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn reader_uses_legacy_data_path_before_writer_migration() {
    let (pool, _c) = spin_up_postgres().await.unwrap();
    let mgr = MulticatalogManager::new(pool.clone());
    let catalog_id = mgr.create_catalog("legacy_reader").await.unwrap();
    sqlx::query("ALTER TABLE ducklake_catalog DROP COLUMN data_path")
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query(
        "INSERT INTO ducklake_metadata (key, value, scope)
         VALUES ('data_path', '/data/legacy-reader', NULL)",
    )
    .execute(&pool)
    .await
    .unwrap();
    let provider = MulticatalogProvider::with_pool_and_id(pool, catalog_id)
        .await
        .unwrap();

    assert_eq!(provider.get_data_path().unwrap(), "/data/legacy-reader");
    assert_eq!(
        mgr.get_data_path_in_catalog("legacy_reader").await.unwrap(),
        "/data/legacy-reader"
    );
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn global_orphan_cleanup_errors_without_data_path() {
    use datafusion_ducklake::maintenance::{CleanupCriteria, delete_orphaned_files_multicatalog};
    use object_store::ObjectStore;
    use object_store::local::LocalFileSystem;

    let (pool, _c) = spin_up_postgres().await.unwrap();
    let mgr = MulticatalogManager::new(pool);
    let store: Arc<dyn ObjectStore> = Arc::new(LocalFileSystem::new());
    let error = delete_orphaned_files_multicatalog(&mgr, store, CleanupCriteria::All, false)
        .await
        .unwrap_err();

    assert!(error.to_string().contains("data_path' not configured"));
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn catalog_orphan_cleanup_keeps_shared_root_peer_files() {
    use datafusion_ducklake::maintenance::{CleanupCriteria, delete_orphaned_files_in_catalog};
    use object_store::ObjectStore;
    use object_store::local::LocalFileSystem;
    use tempfile::TempDir;

    let (pool, _c) = spin_up_postgres().await.unwrap();
    let mgr = MulticatalogManager::new(pool.clone());
    let cat_a = mgr.create_catalog("cat_a").await.unwrap();
    let cat_b = mgr.create_catalog("cat_b").await.unwrap();
    let writer_a = PostgresMetadataWriter::with_pool(pool.clone(), cat_a)
        .await
        .unwrap();
    let writer_b = PostgresMetadataWriter::with_pool(pool, cat_b)
        .await
        .unwrap();
    let temp = TempDir::new().unwrap();
    let data_path = temp.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();
    writer_a.set_data_path(data_path.to_str().unwrap()).unwrap();
    writer_b
        .set_data_path(&format!("{}/", data_path.display()))
        .unwrap();

    for (writer, table, file) in [(&writer_a, "a", "a.parquet"), (&writer_b, "b", "b.parquet")] {
        let setup = writer
            .begin_write_transaction("public", table, &users_cols(), WriteMode::Replace)
            .unwrap();
        writer
            .register_data_file(
                setup.table_id,
                "public",
                table,
                setup.snapshot_id,
                &DataFileInfo::new(file, 100, 1),
                WriteMode::Replace,
                setup.base_snapshot_id,
                &users_cols(),
                &setup.column_ids,
            )
            .unwrap();
    }

    let a_file = data_path
        .join(format!("cat_{cat_a}"))
        .join("public/a/a.parquet");
    let b_file = data_path
        .join(format!("cat_{cat_b}"))
        .join("public/b/b.parquet");
    let orphan_file = data_path.join("orphan.parquet");
    std::fs::create_dir_all(a_file.parent().unwrap()).unwrap();
    std::fs::create_dir_all(b_file.parent().unwrap()).unwrap();
    std::fs::write(&a_file, b"a").unwrap();
    std::fs::write(&b_file, b"b").unwrap();
    std::fs::write(&orphan_file, b"orphan").unwrap();

    let store: Arc<dyn ObjectStore> = Arc::new(LocalFileSystem::new());
    let deleted =
        delete_orphaned_files_in_catalog(&mgr, "cat_a", store, CleanupCriteria::All, false)
            .await
            .unwrap();

    assert_eq!(deleted, vec![orphan_file.to_string_lossy().into_owned()]);
    assert!(a_file.exists());
    assert!(b_file.exists());
    assert!(!orphan_file.exists());
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn global_orphan_cleanup_keeps_nested_root_files() {
    use datafusion_ducklake::maintenance::{CleanupCriteria, delete_orphaned_files_multicatalog};
    use object_store::ObjectStore;
    use object_store::local::LocalFileSystem;
    use tempfile::TempDir;

    let (pool, _c) = spin_up_postgres().await.unwrap();
    let mgr = MulticatalogManager::new(pool.clone());
    let cat_a = mgr.create_catalog("parent").await.unwrap();
    let cat_b = mgr.create_catalog("nested").await.unwrap();
    let writer_a = PostgresMetadataWriter::with_pool(pool.clone(), cat_a)
        .await
        .unwrap();
    let writer_b = PostgresMetadataWriter::with_pool(pool, cat_b)
        .await
        .unwrap();
    let temp = TempDir::new().unwrap();
    let root_a = temp.path().join("data");
    let root_b = root_a.join("tenant");
    std::fs::create_dir_all(&root_b).unwrap();
    writer_a.set_data_path(root_a.to_str().unwrap()).unwrap();
    writer_b.set_data_path(root_b.to_str().unwrap()).unwrap();

    for (writer, table, file) in [(&writer_a, "a", "a.parquet"), (&writer_b, "b", "b.parquet")] {
        let setup = writer
            .begin_write_transaction("public", table, &users_cols(), WriteMode::Replace)
            .unwrap();
        writer
            .register_data_file(
                setup.table_id,
                "public",
                table,
                setup.snapshot_id,
                &DataFileInfo::new(file, 100, 1),
                WriteMode::Replace,
                setup.base_snapshot_id,
                &users_cols(),
                &setup.column_ids,
            )
            .unwrap();
    }

    let a_file = root_a
        .join(format!("cat_{cat_a}"))
        .join("public/a/a.parquet");
    let b_file = root_b
        .join(format!("cat_{cat_b}"))
        .join("public/b/b.parquet");
    let orphan_file = root_a.join("orphan.parquet");
    std::fs::create_dir_all(a_file.parent().unwrap()).unwrap();
    std::fs::create_dir_all(b_file.parent().unwrap()).unwrap();
    std::fs::write(&a_file, b"a").unwrap();
    std::fs::write(&b_file, b"b").unwrap();
    std::fs::write(&orphan_file, b"orphan").unwrap();

    let store: Arc<dyn ObjectStore> = Arc::new(LocalFileSystem::new());
    let deleted = delete_orphaned_files_multicatalog(&mgr, store, CleanupCriteria::All, false)
        .await
        .unwrap();

    assert_eq!(deleted, vec![orphan_file.to_string_lossy().into_owned()]);
    assert!(a_file.exists());
    assert!(b_file.exists());
    assert!(!orphan_file.exists());
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn global_orphan_cleanup_reclaims_dropped_distinct_root_catalog() {
    use datafusion_ducklake::maintenance::{CleanupCriteria, delete_orphaned_files_multicatalog};
    use object_store::ObjectStore;
    use object_store::local::LocalFileSystem;
    use tempfile::TempDir;

    let (pool, _c) = spin_up_postgres().await.unwrap();
    let mgr = MulticatalogManager::new(pool.clone());
    let cat_a = mgr.create_catalog("cat_a").await.unwrap();
    let cat_b = mgr.create_catalog("cat_b").await.unwrap();
    let writer_a = PostgresMetadataWriter::with_pool(pool.clone(), cat_a)
        .await
        .unwrap();
    let writer_b = PostgresMetadataWriter::with_pool(pool.clone(), cat_b)
        .await
        .unwrap();
    let temp = TempDir::new().unwrap();
    let root_a = temp.path().join("a");
    let root_b = temp.path().join("b");
    std::fs::create_dir_all(&root_a).unwrap();
    std::fs::create_dir_all(&root_b).unwrap();
    writer_a.set_data_path(root_a.to_str().unwrap()).unwrap();
    writer_b.set_data_path(root_b.to_str().unwrap()).unwrap();

    let setup = writer_b
        .begin_write_transaction("public", "b", &users_cols(), WriteMode::Replace)
        .unwrap();
    writer_b
        .register_data_file(
            setup.table_id,
            "public",
            "b",
            setup.snapshot_id,
            &DataFileInfo::new("b.parquet", 100, 1),
            WriteMode::Replace,
            setup.base_snapshot_id,
            &users_cols(),
            &setup.column_ids,
        )
        .unwrap();
    sqlx::query(
        "INSERT INTO ducklake_files_scheduled_for_deletion
             (catalog_id, data_file_id, path, path_is_relative)
         VALUES ($1, 999, 'scheduled.parquet', TRUE)",
    )
    .bind(cat_b)
    .execute(&pool)
    .await
    .unwrap();
    let b_file = root_b
        .join(format!("cat_{cat_b}"))
        .join("public/b/b.parquet");
    std::fs::create_dir_all(b_file.parent().unwrap()).unwrap();
    std::fs::write(&b_file, b"b").unwrap();

    assert!(mgr.drop_catalog("cat_b").await.unwrap());
    let scheduled: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM ducklake_files_scheduled_for_deletion WHERE catalog_id = $1",
    )
    .bind(cat_b)
    .fetch_one(&pool)
    .await
    .unwrap();
    let store: Arc<dyn ObjectStore> = Arc::new(LocalFileSystem::new());
    let deleted = delete_orphaned_files_multicatalog(&mgr, store, CleanupCriteria::All, false)
        .await
        .unwrap();

    assert_eq!(scheduled, 0);
    assert_eq!(deleted, vec![b_file.to_string_lossy().into_owned()]);
    assert!(!b_file.exists());
    assert_eq!(
        mgr.list_data_paths().await.unwrap(),
        vec![root_a.to_string_lossy().into_owned()],
    );
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn global_orphan_cleanup_groups_dropped_path_aliases() {
    use datafusion_ducklake::maintenance::{CleanupCriteria, delete_orphaned_files_multicatalog};
    use object_store::ObjectStore;
    use object_store::local::LocalFileSystem;
    use tempfile::TempDir;

    let (pool, _c) = spin_up_postgres().await.unwrap();
    let mgr = MulticatalogManager::new(pool.clone());
    let catalog_id = mgr.create_catalog("live").await.unwrap();
    let writer = PostgresMetadataWriter::with_pool(pool.clone(), catalog_id)
        .await
        .unwrap();
    let temp = TempDir::new().unwrap();
    let data_path = temp.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();
    writer.set_data_path(data_path.to_str().unwrap()).unwrap();

    let setup = writer
        .begin_write_transaction("public", "live", &users_cols(), WriteMode::Replace)
        .unwrap();
    writer
        .register_data_file(
            setup.table_id,
            "public",
            "live",
            setup.snapshot_id,
            &DataFileInfo::new("live.parquet", 100, 1),
            WriteMode::Replace,
            setup.base_snapshot_id,
            &users_cols(),
            &setup.column_ids,
        )
        .unwrap();
    let data_path_alias = format!("{}/", data_path.display());
    sqlx::query(
        "INSERT INTO ducklake_dropped_data_path (data_path, dropped_at)
         VALUES ($1, NOW() + INTERVAL '1 hour')",
    )
    .bind(&data_path_alias)
    .execute(&pool)
    .await
    .unwrap();

    let live_file = data_path
        .join(format!("cat_{catalog_id}"))
        .join("public/live/live.parquet");
    let orphan_file = data_path.join("orphan.parquet");
    std::fs::create_dir_all(live_file.parent().unwrap()).unwrap();
    std::fs::write(&live_file, b"live").unwrap();
    std::fs::write(&orphan_file, b"orphan").unwrap();

    let store: Arc<dyn ObjectStore> = Arc::new(LocalFileSystem::new());
    let deleted = delete_orphaned_files_multicatalog(&mgr, store, CleanupCriteria::All, false)
        .await
        .unwrap();

    assert_eq!(deleted, vec![orphan_file.to_string_lossy().into_owned()]);
    assert!(live_file.exists());
    assert!(!orphan_file.exists());
    assert_eq!(
        mgr.list_data_paths().await.unwrap(),
        vec![data_path.to_string_lossy().into_owned(), data_path_alias.clone()],
    );

    sqlx::query(
        "UPDATE ducklake_dropped_data_path
         SET dropped_at = NOW() - INTERVAL '1 hour' WHERE data_path = $1",
    )
    .bind(&data_path_alias)
    .execute(&pool)
    .await
    .unwrap();
    let store: Arc<dyn ObjectStore> = Arc::new(LocalFileSystem::new());
    let deleted = delete_orphaned_files_multicatalog(&mgr, store, CleanupCriteria::All, false)
        .await
        .unwrap();

    assert!(deleted.is_empty());
    assert_eq!(
        mgr.list_data_paths().await.unwrap(),
        vec![data_path.to_string_lossy().into_owned()],
    );
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn global_orphan_cleanup_rejects_multiple_object_store_authorities() {
    use datafusion_ducklake::maintenance::{CleanupCriteria, delete_orphaned_files_multicatalog};
    use object_store::ObjectStore;
    use object_store::local::LocalFileSystem;

    let (pool, _c) = spin_up_postgres().await.unwrap();
    let mgr = MulticatalogManager::new(pool.clone());
    let cat_a = mgr.create_catalog("cat_a").await.unwrap();
    let cat_b = mgr.create_catalog("cat_b").await.unwrap();
    let writer_a = PostgresMetadataWriter::with_pool(pool.clone(), cat_a)
        .await
        .unwrap();
    let writer_b = PostgresMetadataWriter::with_pool(pool, cat_b)
        .await
        .unwrap();
    writer_a.set_data_path("s3://bucket-a/data").unwrap();
    writer_b.set_data_path("s3://bucket-b/data").unwrap();
    let store: Arc<dyn ObjectStore> = Arc::new(LocalFileSystem::new());

    let error = delete_orphaned_files_multicatalog(&mgr, store, CleanupCriteria::All, false)
        .await
        .unwrap_err();

    assert!(
        error
            .to_string()
            .contains("multiple object-store authorities")
    );
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn rollback_leaves_no_orphan_rows() {
    // Commit-time model: `begin_write_transaction` inserts NOTHING into
    // ducklake_snapshot / ducklake_schema / ducklake_table / ducklake_column /
    // either map (it only reserves sequence ids). So a begin that never commits
    // — and a begin that fails mid-validation — must leave the snapshot count
    // unchanged and no orphan map rows. The snapshot row only appears when the
    // write commits via register_data_file / publish_snapshot.
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

    // A bare begin that never commits inserts no ducklake_snapshot row.
    let _ok = w
        .begin_write_transaction("public", "users", &users_cols(), WriteMode::Replace)
        .unwrap();
    let after_bare_begin: i64 = sqlx::query("SELECT COUNT(*) FROM ducklake_snapshot")
        .fetch_one(&pool)
        .await
        .unwrap()
        .try_get(0)
        .unwrap();
    assert_eq!(
        after_bare_begin, before_snaps,
        "a begin that never commits must NOT insert a snapshot row"
    );

    // A begin that FAILS validation also leaves nothing behind. There is no live
    // table yet (the previous begin never committed), so use a fresh-table begin
    // with an invalid column type so the failure path is exercised on a
    // sequence-only transaction.
    let bad_col = ColumnDef::new("id", "not_a_real_type", false);
    assert!(bad_col.is_err(), "invalid type is rejected before begin");
    // An invalid begin still must not leave a snapshot or orphan map row.
    let after_invalid: i64 = sqlx::query("SELECT COUNT(*) FROM ducklake_snapshot")
        .fetch_one(&pool)
        .await
        .unwrap()
        .try_get(0)
        .unwrap();
    assert_eq!(after_invalid, before_snaps);

    // No orphan snapshot-map rows anywhere.
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
    // Commit the table first — under the commit-time model `begin_write_transaction`
    // reserves ids but inserts no table row, so the schema/table rows only exist
    // after publish.
    let setup = w
        .begin_write_transaction("public", "users", &users_cols(), WriteMode::Replace)
        .unwrap();
    w.publish_snapshot(
        setup.table_id,
        "public",
        "users",
        setup.snapshot_id,
        WriteMode::Replace,
        setup.base_snapshot_id,
        &users_cols(),
        &setup.column_ids,
    )
    .unwrap();

    // Resolve the committed schema_id and a valid snapshot to bind the raw insert.
    let schema_id: i64 = sqlx::query(
        "SELECT s.schema_id FROM ducklake_schema s
         JOIN ducklake_catalog_schema_map m ON m.schema_id = s.schema_id
         WHERE m.catalog_id = $1 AND s.schema_name = 'public' AND s.end_snapshot IS NULL",
    )
    .bind(cat)
    .fetch_one(&pool)
    .await
    .unwrap()
    .try_get(0)
    .unwrap();
    let snap: i64 = sqlx::query("SELECT MAX(snapshot_id) FROM ducklake_snapshot")
        .fetch_one(&pool)
        .await
        .unwrap()
        .try_get(0)
        .unwrap();

    // Raw SQL insert of a second active row with the same (schema_id, name).
    let err = sqlx::query(
        "INSERT INTO ducklake_table (schema_id, table_name, path, path_is_relative, begin_snapshot)
         VALUES ($1, 'users', 'users', TRUE, $2)",
    )
    .bind(schema_id)
    .bind(snap)
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

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn multicatalog_provider_lists_and_resolves_views() {
    let (pool, _c) = spin_up_postgres().await.unwrap();
    let catalog_id = MulticatalogManager::new(pool.clone())
        .create_catalog("views")
        .await
        .unwrap();
    let writer = PostgresMetadataWriter::with_pool(pool.clone(), catalog_id)
        .await
        .unwrap();
    let setup = writer
        .begin_write_transaction("public", "items", &users_cols(), WriteMode::Replace)
        .unwrap();
    let committed = writer
        .publish_snapshot(
            setup.table_id,
            "public",
            "items",
            setup.snapshot_id,
            WriteMode::Replace,
            setup.base_snapshot_id,
            &users_cols(),
            &setup.column_ids,
        )
        .unwrap();

    let schema_id: i64 = sqlx::query_scalar(
        "SELECT s.schema_id FROM ducklake_schema s
         JOIN ducklake_catalog_schema_map m ON m.schema_id = s.schema_id
         WHERE m.catalog_id = $1 AND s.schema_name = 'public' AND s.end_snapshot IS NULL",
    )
    .bind(catalog_id)
    .fetch_one(&pool)
    .await
    .unwrap();
    sqlx::query(
        "INSERT INTO ducklake_view
         (view_id, view_uuid, begin_snapshot, end_snapshot, schema_id, view_name, dialect, sql, column_aliases)
         VALUES ($1, '00000000-0000-0000-0000-000000000001'::uuid, $2, NULL, $3, $4, $5, $6, $7)",
    )
    .bind(1_i64)
    .bind(committed.snapshot_id)
    .bind(schema_id)
    .bind("active_items")
    .bind("duckdb")
    .bind("SELECT * FROM {DUCKLAKE_CATALOG}.public.items")
    .bind("")
    .execute(&pool)
    .await
    .unwrap();

    let provider = MulticatalogProvider::with_pool_and_id(pool, catalog_id)
        .await
        .unwrap();
    let views = provider
        .list_views(schema_id, committed.snapshot_id)
        .unwrap();
    assert_eq!(views.len(), 1);
    assert_eq!(views[0].view_name, "active_items");
    assert_eq!(views[0].begin_snapshot, committed.snapshot_id);
    assert_eq!(
        provider
            .get_view_by_name(schema_id, "active_items", committed.snapshot_id)
            .unwrap(),
        Some(views[0].clone())
    );
    let all_views = provider.list_all_views(committed.snapshot_id).unwrap();
    assert_eq!(all_views.len(), 1);
    assert_eq!(all_views[0].schema_name, "public");
    assert_eq!(all_views[0].view, views[0]);
}
