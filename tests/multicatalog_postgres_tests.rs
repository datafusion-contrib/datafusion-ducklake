#![cfg(feature = "write-postgres")]
//! Integration tests for the multicatalog Postgres writer.
//!
//! Covers:
//! - DDL bootstrap idempotency
//! - `MulticatalogManager::create_catalog` semantics
//! - Single-catalog write flow on Postgres
//! - Cross-catalog isolation (writes in catalog A invisible to catalog B)
//! - Per-catalog dense `schema_version` allocation
//! - No orphan mapping rows after writes

use datafusion_ducklake::metadata_writer::{ColumnDef, MetadataWriter, WriteMode};
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
    let conn_str = format!("postgresql://postgres:postgres@127.0.0.1:{}/postgres", port);
    let pool = PgPoolOptions::new()
        .max_connections(5)
        .connect(&conn_str)
        .await?;
    initialize_multicatalog_schema(&pool).await?;
    Ok((pool, container))
}

fn cols() -> Vec<ColumnDef> {
    vec![
        ColumnDef::new("id", "int64", false).unwrap(),
        ColumnDef::new("name", "varchar", true).unwrap(),
    ]
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn initialize_multicatalog_schema_is_idempotent() {
    let (pool, _c) = spin_up_postgres().await.unwrap();
    // Calling again must not error.
    initialize_multicatalog_schema(&pool).await.unwrap();
    initialize_multicatalog_schema(&pool).await.unwrap();

    // schema_version column exists on ducklake_snapshot.
    let row = sqlx::query(
        "SELECT column_name FROM information_schema.columns
         WHERE table_name = 'ducklake_snapshot' AND column_name = 'schema_version'",
    )
    .fetch_optional(&pool)
    .await
    .unwrap();
    assert!(row.is_some(), "schema_version column should exist");

    // All catalog tables exist.
    for table in [
        "ducklake_catalog",
        "ducklake_catalog_snapshot_map",
        "ducklake_catalog_schema_map",
        "ducklake_schema_versions",
    ] {
        let row =
            sqlx::query("SELECT table_name FROM information_schema.tables WHERE table_name = $1")
                .bind(table)
                .fetch_optional(&pool)
                .await
                .unwrap();
        assert!(row.is_some(), "table {} should exist", table);
    }
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn create_catalog_is_idempotent_by_name() {
    let (pool, _c) = spin_up_postgres().await.unwrap();
    let mgr = MulticatalogManager::new(pool);

    let id_a = mgr.create_catalog("pg_prod").await.unwrap();
    let id_b = mgr.create_catalog("pg_prod").await.unwrap();
    assert_eq!(id_a, id_b, "same name should yield same id");

    let id_other = mgr.create_catalog("mysql_prod").await.unwrap();
    assert_ne!(id_a, id_other, "different names get different ids");

    let listed = mgr.list_catalogs().await.unwrap();
    assert_eq!(listed.len(), 2);
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn create_catalog_rejects_empty_name() {
    let (pool, _c) = spin_up_postgres().await.unwrap();
    let mgr = MulticatalogManager::new(pool);
    assert!(mgr.create_catalog("").await.is_err());
    assert!(mgr.create_catalog("   ").await.is_err());
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn single_catalog_ddl_then_dml_assigns_versions() {
    let (pool, _c) = spin_up_postgres().await.unwrap();
    let mgr = MulticatalogManager::new(pool.clone());
    let catalog_id = mgr.create_catalog("pg_prod").await.unwrap();
    let writer = PostgresMetadataWriter::with_pool(pool.clone(), catalog_id)
        .await
        .unwrap();
    writer.set_data_path("/data").unwrap();

    // First commit: DDL (table doesn't exist).
    let setup1 = writer
        .begin_write_transaction("public", "users", &cols(), WriteMode::Replace)
        .unwrap();

    // Second commit: same columns -> DML, carry forward schema_version.
    let setup2 = writer
        .begin_write_transaction("public", "users", &cols(), WriteMode::Replace)
        .unwrap();

    let v1: i64 =
        sqlx::query("SELECT schema_version FROM ducklake_snapshot WHERE snapshot_id = $1")
            .bind(setup1.snapshot_id)
            .fetch_one(&pool)
            .await
            .unwrap()
            .try_get(0)
            .unwrap();
    let v2: i64 =
        sqlx::query("SELECT schema_version FROM ducklake_snapshot WHERE snapshot_id = $1")
            .bind(setup2.snapshot_id)
            .fetch_one(&pool)
            .await
            .unwrap()
            .try_get(0)
            .unwrap();
    assert_eq!(v1, 1, "first DDL ⇒ v1");
    assert_eq!(v2, 1, "DML carries forward ⇒ still v1");

    // ducklake_schema_versions has exactly one row for the DDL commit.
    let count: i64 =
        sqlx::query("SELECT COUNT(*) FROM ducklake_schema_versions WHERE table_id = $1")
            .bind(setup1.table_id)
            .fetch_one(&pool)
            .await
            .unwrap()
            .try_get(0)
            .unwrap();
    assert_eq!(
        count, 1,
        "only the DDL commit records a schema_versions row"
    );

    // Third commit: column added → DDL bump.
    let mut cols_v2 = cols();
    cols_v2.push(ColumnDef::new("age", "int32", true).unwrap());
    let setup3 = writer
        .begin_write_transaction("public", "users", &cols_v2, WriteMode::Replace)
        .unwrap();
    let v3: i64 =
        sqlx::query("SELECT schema_version FROM ducklake_snapshot WHERE snapshot_id = $1")
            .bind(setup3.snapshot_id)
            .fetch_one(&pool)
            .await
            .unwrap()
            .try_get(0)
            .unwrap();
    assert_eq!(v3, 2, "column added ⇒ DDL ⇒ v2");
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn cross_catalog_isolation_same_schema_name() {
    let (pool, _c) = spin_up_postgres().await.unwrap();
    let mgr = MulticatalogManager::new(pool.clone());

    let cat_a = mgr.create_catalog("pg_prod").await.unwrap();
    let cat_b = mgr.create_catalog("mysql_prod").await.unwrap();

    let writer_a = PostgresMetadataWriter::with_pool(pool.clone(), cat_a)
        .await
        .unwrap();
    let writer_b = PostgresMetadataWriter::with_pool(pool.clone(), cat_b)
        .await
        .unwrap();
    writer_a.set_data_path("/data").unwrap();

    let setup_a = writer_a
        .begin_write_transaction("public", "users", &cols(), WriteMode::Replace)
        .unwrap();
    let setup_b = writer_b
        .begin_write_transaction("public", "orders", &cols(), WriteMode::Replace)
        .unwrap();

    // Schemas: two "public" rows, one per catalog, with different schema_ids.
    assert_ne!(setup_a.schema_id, setup_b.schema_id);

    // Catalog A's mapping points only at A's schema.
    let schema_ids_a: Vec<i64> =
        sqlx::query("SELECT schema_id FROM ducklake_catalog_schema_map WHERE catalog_id = $1")
            .bind(cat_a)
            .fetch_all(&pool)
            .await
            .unwrap()
            .into_iter()
            .map(|r| r.try_get(0).unwrap())
            .collect();
    assert_eq!(schema_ids_a, vec![setup_a.schema_id]);

    let schema_ids_b: Vec<i64> =
        sqlx::query("SELECT schema_id FROM ducklake_catalog_schema_map WHERE catalog_id = $1")
            .bind(cat_b)
            .fetch_all(&pool)
            .await
            .unwrap()
            .into_iter()
            .map(|r| r.try_get(0).unwrap())
            .collect();
    assert_eq!(schema_ids_b, vec![setup_b.schema_id]);

    // Each catalog has exactly one snapshot mapping after one write.
    for cat in [cat_a, cat_b] {
        let n: i64 =
            sqlx::query("SELECT COUNT(*) FROM ducklake_catalog_snapshot_map WHERE catalog_id = $1")
                .bind(cat)
                .fetch_one(&pool)
                .await
                .unwrap()
                .try_get(0)
                .unwrap();
        assert_eq!(n, 1, "catalog {} should have 1 snapshot mapping", cat);
    }
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn schema_version_is_per_catalog_dense_under_interleaving() {
    // Reproduces the spec's working-example scenario:
    //   cat_a: DDL(v1), DML(v1), DDL(v2)
    //   cat_b interleaved: DDL(v1), DML(v1)
    let (pool, _c) = spin_up_postgres().await.unwrap();
    let mgr = MulticatalogManager::new(pool.clone());
    let cat_a = mgr.create_catalog("pg_prod").await.unwrap();
    let cat_b = mgr.create_catalog("mysql_prod").await.unwrap();
    let wa = PostgresMetadataWriter::with_pool(pool.clone(), cat_a)
        .await
        .unwrap();
    let wb = PostgresMetadataWriter::with_pool(pool.clone(), cat_b)
        .await
        .unwrap();
    wa.set_data_path("/data").unwrap();

    // cat_a DDL (creates users)
    let a1 = wa
        .begin_write_transaction("public", "users", &cols(), WriteMode::Replace)
        .unwrap();
    // cat_a DML (Replace, same schema)
    let a2 = wa
        .begin_write_transaction("public", "users", &cols(), WriteMode::Replace)
        .unwrap();
    // cat_b DDL (creates orders) — happens in between cat_a's DDLs
    let b1 = wb
        .begin_write_transaction("public", "orders", &cols(), WriteMode::Replace)
        .unwrap();
    // cat_a DDL: adds age column
    let mut cols_v2 = cols();
    cols_v2.push(ColumnDef::new("age", "int32", true).unwrap());
    let a3 = wa
        .begin_write_transaction("public", "users", &cols_v2, WriteMode::Replace)
        .unwrap();
    // cat_b DML
    let b2 = wb
        .begin_write_transaction("public", "orders", &cols(), WriteMode::Replace)
        .unwrap();

    let get_v = |snap_id: i64| {
        let pool = pool.clone();
        async move {
            sqlx::query("SELECT schema_version FROM ducklake_snapshot WHERE snapshot_id = $1")
                .bind(snap_id)
                .fetch_one(&pool)
                .await
                .unwrap()
                .try_get::<i64, _>(0)
                .unwrap()
        }
    };

    assert_eq!(get_v(a1.snapshot_id).await, 1, "cat_a first DDL");
    assert_eq!(get_v(a2.snapshot_id).await, 1, "cat_a DML carries v1");
    assert_eq!(get_v(b1.snapshot_id).await, 1, "cat_b first DDL");
    assert_eq!(get_v(a3.snapshot_id).await, 2, "cat_a column-add DDL");
    assert_eq!(get_v(b2.snapshot_id).await, 1, "cat_b DML carries v1");
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn no_orphan_mapping_rows() {
    let (pool, _c) = spin_up_postgres().await.unwrap();
    let mgr = MulticatalogManager::new(pool.clone());
    let cat = mgr.create_catalog("pg_prod").await.unwrap();
    let w = PostgresMetadataWriter::with_pool(pool.clone(), cat)
        .await
        .unwrap();
    w.set_data_path("/data").unwrap();
    let _ = w
        .begin_write_transaction("public", "users", &cols(), WriteMode::Replace)
        .unwrap();
    let _ = w
        .begin_write_transaction("public", "users", &cols(), WriteMode::Replace)
        .unwrap();

    // Every entry in the maps must point at a real row.
    let orphan_snaps: i64 = sqlx::query(
        "SELECT COUNT(*) FROM ducklake_catalog_snapshot_map m
         LEFT JOIN ducklake_snapshot s ON s.snapshot_id = m.snapshot_id
         WHERE s.snapshot_id IS NULL",
    )
    .fetch_one(&pool)
    .await
    .unwrap()
    .try_get(0)
    .unwrap();
    assert_eq!(orphan_snaps, 0);

    let orphan_schemas: i64 = sqlx::query(
        "SELECT COUNT(*) FROM ducklake_catalog_schema_map m
         LEFT JOIN ducklake_schema s ON s.schema_id = m.schema_id
         WHERE s.schema_id IS NULL",
    )
    .fetch_one(&pool)
    .await
    .unwrap()
    .try_get(0)
    .unwrap();
    assert_eq!(orphan_schemas, 0);

    // Every snapshot created via a writer must have a mapping.
    let unmapped_snaps: i64 = sqlx::query(
        "SELECT COUNT(*) FROM ducklake_snapshot s
         LEFT JOIN ducklake_catalog_snapshot_map m ON m.snapshot_id = s.snapshot_id
         WHERE m.catalog_id IS NULL",
    )
    .fetch_one(&pool)
    .await
    .unwrap()
    .try_get(0)
    .unwrap();
    assert_eq!(unmapped_snaps, 0);
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn register_data_file_records_against_table() {
    use datafusion_ducklake::metadata_writer::DataFileInfo;
    let (pool, _c) = spin_up_postgres().await.unwrap();
    let mgr = MulticatalogManager::new(pool.clone());
    let cat = mgr.create_catalog("pg_prod").await.unwrap();
    let w = PostgresMetadataWriter::with_pool(pool.clone(), cat)
        .await
        .unwrap();
    w.set_data_path("/data").unwrap();
    let setup = w
        .begin_write_transaction("public", "users", &cols(), WriteMode::Replace)
        .unwrap();

    let file = DataFileInfo::new("abc.parquet", 4096, 100).with_footer_size(256);
    let file_id = w
        .register_data_file(setup.table_id, setup.snapshot_id, &file)
        .unwrap();
    assert!(file_id > 0);

    let row = sqlx::query(
        "SELECT path, file_size_bytes, record_count, begin_snapshot
         FROM ducklake_data_file WHERE data_file_id = $1",
    )
    .bind(file_id)
    .fetch_one(&pool)
    .await
    .unwrap();
    let path: String = row.try_get(0).unwrap();
    let size: i64 = row.try_get(1).unwrap();
    let count: i64 = row.try_get(2).unwrap();
    let begin: i64 = row.try_get(3).unwrap();
    assert_eq!(path, "abc.parquet");
    assert_eq!(size, 4096);
    assert_eq!(count, 100);
    assert_eq!(begin, setup.snapshot_id);
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn drop_catalog_returns_false_for_unknown_name() {
    let (pool, _c) = spin_up_postgres().await.unwrap();
    let mgr = MulticatalogManager::new(pool);
    let dropped = mgr.drop_catalog("does_not_exist").await.unwrap();
    assert!(!dropped, "dropping unknown catalog should report false");
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn drop_catalog_rejects_empty_name() {
    let (pool, _c) = spin_up_postgres().await.unwrap();
    let mgr = MulticatalogManager::new(pool);
    assert!(mgr.drop_catalog("").await.is_err());
    assert!(mgr.drop_catalog("   ").await.is_err());
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn drop_catalog_removes_empty_catalog() {
    let (pool, _c) = spin_up_postgres().await.unwrap();
    let mgr = MulticatalogManager::new(pool.clone());
    let _ = mgr.create_catalog("pg_prod").await.unwrap();

    let dropped = mgr.drop_catalog("pg_prod").await.unwrap();
    assert!(dropped, "first drop should report true");

    // No catalog row left.
    assert!(mgr.find_catalog_id("pg_prod").await.unwrap().is_none());

    // Second drop is a no-op.
    let again = mgr.drop_catalog("pg_prod").await.unwrap();
    assert!(!again, "second drop should report false");
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn drop_catalog_removes_populated_catalog() {
    use datafusion_ducklake::metadata_writer::DataFileInfo;

    let (pool, _c) = spin_up_postgres().await.unwrap();
    let mgr = MulticatalogManager::new(pool.clone());
    let cat = mgr.create_catalog("pg_prod").await.unwrap();
    let w = PostgresMetadataWriter::with_pool(pool.clone(), cat)
        .await
        .unwrap();
    w.set_data_path("/data").unwrap();

    // Two tables across one schema, with a data file each + a DDL bump
    // to populate ducklake_schema_versions.
    let s1 = w
        .begin_write_transaction("public", "users", &cols(), WriteMode::Replace)
        .unwrap();
    w.register_data_file(
        s1.table_id,
        s1.snapshot_id,
        &DataFileInfo::new("u.parquet", 1024, 10),
    )
    .unwrap();

    let mut cols_v2 = cols();
    cols_v2.push(ColumnDef::new("age", "int32", true).unwrap());
    let _ = w
        .begin_write_transaction("public", "users", &cols_v2, WriteMode::Replace)
        .unwrap();

    let s_orders = w
        .begin_write_transaction("public", "orders", &cols(), WriteMode::Replace)
        .unwrap();
    w.register_data_file(
        s_orders.table_id,
        s_orders.snapshot_id,
        &DataFileInfo::new("o.parquet", 2048, 20),
    )
    .unwrap();

    // Drop and verify every catalog-scoped table has no rows for this
    // catalog. Iterate so a future column addition can't quietly skip a
    // table.
    let dropped = mgr.drop_catalog("pg_prod").await.unwrap();
    assert!(dropped);

    // Catalog and mapping rows.
    for query in [
        "SELECT COUNT(*) FROM ducklake_catalog",
        "SELECT COUNT(*) FROM ducklake_catalog_schema_map",
        "SELECT COUNT(*) FROM ducklake_catalog_snapshot_map",
    ] {
        let n: i64 = sqlx::query(query)
            .fetch_one(&pool)
            .await
            .unwrap()
            .try_get(0)
            .unwrap();
        assert_eq!(n, 0, "{} should be empty after drop", query);
    }

    // Entities owned by the catalog. With only one catalog, "owned by
    // this catalog" is the same as "any row at all" — global zero is
    // the right post-condition.
    for table in [
        "ducklake_schema",
        "ducklake_table",
        "ducklake_column",
        "ducklake_snapshot",
        "ducklake_data_file",
        "ducklake_delete_file",
        "ducklake_schema_versions",
    ] {
        let n: i64 = sqlx::query(&format!("SELECT COUNT(*) FROM {}", table))
            .fetch_one(&pool)
            .await
            .unwrap()
            .try_get(0)
            .unwrap();
        assert_eq!(n, 0, "{} should be empty after drop", table);
    }
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn drop_catalog_isolates_other_catalogs() {
    use datafusion_ducklake::metadata_writer::DataFileInfo;

    let (pool, _c) = spin_up_postgres().await.unwrap();
    let mgr = MulticatalogManager::new(pool.clone());
    let cat_a = mgr.create_catalog("pg_prod").await.unwrap();
    let cat_b = mgr.create_catalog("mysql_prod").await.unwrap();
    let wa = PostgresMetadataWriter::with_pool(pool.clone(), cat_a)
        .await
        .unwrap();
    let wb = PostgresMetadataWriter::with_pool(pool.clone(), cat_b)
        .await
        .unwrap();
    wa.set_data_path("/data").unwrap();

    // Populate both catalogs.
    let sa = wa
        .begin_write_transaction("public", "users", &cols(), WriteMode::Replace)
        .unwrap();
    wa.register_data_file(
        sa.table_id,
        sa.snapshot_id,
        &DataFileInfo::new("a.parquet", 1024, 10),
    )
    .unwrap();

    let sb = wb
        .begin_write_transaction("public", "orders", &cols(), WriteMode::Replace)
        .unwrap();
    wb.register_data_file(
        sb.table_id,
        sb.snapshot_id,
        &DataFileInfo::new("b.parquet", 2048, 20),
    )
    .unwrap();

    // Drop catalog A. Catalog B's entities must survive.
    let dropped = mgr.drop_catalog("pg_prod").await.unwrap();
    assert!(dropped);

    // Mapping rows: A gone, B intact.
    for (cat_id, expected, label) in
        [(cat_a, 0i64, "cat_a schema_map gone"), (cat_b, 1i64, "cat_b schema_map intact")]
    {
        let n: i64 =
            sqlx::query("SELECT COUNT(*) FROM ducklake_catalog_schema_map WHERE catalog_id = $1")
                .bind(cat_id)
                .fetch_one(&pool)
                .await
                .unwrap()
                .try_get(0)
                .unwrap();
        assert_eq!(n, expected, "{}", label);
    }

    // Catalog B's entities reachable through its mapping rows must still exist.
    let b_schema_count: i64 = sqlx::query(
        "SELECT COUNT(*) FROM ducklake_schema s
         JOIN ducklake_catalog_schema_map m ON m.schema_id = s.schema_id
         WHERE m.catalog_id = $1",
    )
    .bind(cat_b)
    .fetch_one(&pool)
    .await
    .unwrap()
    .try_get(0)
    .unwrap();
    assert_eq!(b_schema_count, 1, "cat_b schema should survive");

    let b_table_count: i64 = sqlx::query(
        "SELECT COUNT(*) FROM ducklake_table t
         JOIN ducklake_catalog_schema_map m ON m.schema_id = t.schema_id
         WHERE m.catalog_id = $1",
    )
    .bind(cat_b)
    .fetch_one(&pool)
    .await
    .unwrap()
    .try_get(0)
    .unwrap();
    assert_eq!(b_table_count, 1, "cat_b table should survive");

    let b_file_count: i64 = sqlx::query(
        "SELECT COUNT(*) FROM ducklake_data_file f
         JOIN ducklake_table t ON t.table_id = f.table_id
         JOIN ducklake_catalog_schema_map m ON m.schema_id = t.schema_id
         WHERE m.catalog_id = $1",
    )
    .bind(cat_b)
    .fetch_one(&pool)
    .await
    .unwrap()
    .try_get(0)
    .unwrap();
    assert_eq!(b_file_count, 1, "cat_b data_file should survive");

    // And the catalog row.
    assert!(mgr.find_catalog_id("pg_prod").await.unwrap().is_none());
    assert_eq!(
        mgr.find_catalog_id("mysql_prod").await.unwrap(),
        Some(cat_b)
    );
}
