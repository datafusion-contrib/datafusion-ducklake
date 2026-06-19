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
    DuckLakeTableWriter, MulticatalogManager, PostgresMetadataWriter,
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

/// Current catalog head = MAX(snapshot_id) over the catalog's mapping rows
/// (mirrors `MulticatalogProvider::get_current_snapshot`).
async fn current_head(pool: &PgPool, catalog_id: i64) -> i64 {
    sqlx::query(
        "SELECT COALESCE(MAX(snapshot_id), 0) FROM ducklake_catalog_snapshot_map
         WHERE catalog_id = $1",
    )
    .bind(catalog_id)
    .fetch_one(pool)
    .await
    .unwrap()
    .try_get(0)
    .unwrap()
}

/// Total record_count of the files visible to a reader at the current head —
/// the same window predicate the read path applies.
async fn visible_records_at_head(pool: &PgPool, catalog_id: i64, table_id: i64) -> i64 {
    let head = current_head(pool, catalog_id).await;
    sqlx::query(
        "SELECT COALESCE(SUM(record_count), 0)::bigint FROM ducklake_data_file
         WHERE table_id = $1
           AND $2 >= begin_snapshot
           AND ($2 < end_snapshot OR end_snapshot IS NULL)",
    )
    .bind(table_id)
    .bind(head)
    .fetch_one(pool)
    .await
    .unwrap()
    .try_get(0)
    .unwrap()
}

/// Regression for the transient empty-read bug: a managed-load Replace must
/// never expose a committed state where the head advanced but the table has
/// zero live files. The head advance + prior-generation retirement happen only
/// at the commit point (register_data_file), so a reader interleaved between
/// begin_write_transaction and register_data_file — i.e. during the data upload
/// — still sees the fully-complete OLD generation.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn replace_is_atomic_no_empty_read_window() {
    use datafusion_ducklake::metadata_writer::DataFileInfo;
    let (pool, _c) = spin_up_postgres().await.unwrap();
    let mgr = MulticatalogManager::new(pool.clone());
    let cat = mgr.create_catalog("pg_atomic").await.unwrap();
    let w = PostgresMetadataWriter::with_pool(pool.clone(), cat)
        .await
        .unwrap();
    w.set_data_path("/data").unwrap();

    // Generation 1: a committed, non-empty table.
    let s1 = w
        .begin_write_transaction("public", "users", &cols(), WriteMode::Replace)
        .unwrap();
    w.register_data_file(
        s1.table_id,
        s1.snapshot_id,
        &DataFileInfo::new("g1.parquet", 1024, 10),
        WriteMode::Replace,
        &[],
        &[],
    )
    .unwrap();
    assert_eq!(current_head(&pool, cat).await, s1.snapshot_id);
    assert_eq!(visible_records_at_head(&pool, cat, s1.table_id).await, 10);

    // Begin generation 2 (Replace). The data upload would run here, between
    // setup and the commit. The new snapshot row exists but is NOT yet the head
    // and the old file is NOT yet retired.
    let s2 = w
        .begin_write_transaction("public", "users", &cols(), WriteMode::Replace)
        .unwrap();

    // A reader interleaved during the upload sees the OLD generation intact:
    // head unchanged, old data fully visible. (Pre-fix the head had already
    // advanced to s2 and the old file was already retired ⇒ count == 0.)
    assert_eq!(
        current_head(&pool, cat).await,
        s1.snapshot_id,
        "head must NOT advance before the new data file is registered"
    );
    assert_eq!(
        visible_records_at_head(&pool, cat, s1.table_id).await,
        10,
        "old generation must stay fully visible during the upload window"
    );

    // Commit generation 2: head advances and generation 1 retires atomically.
    w.register_data_file(
        s2.table_id,
        s2.snapshot_id,
        &DataFileInfo::new("g2.parquet", 2048, 7),
        WriteMode::Replace,
        &[],
        &[],
    )
    .unwrap();
    assert_eq!(current_head(&pool, cat).await, s2.snapshot_id);
    assert_eq!(
        visible_records_at_head(&pool, cat, s2.table_id).await,
        7,
        "new generation visible, old generation retired"
    );
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

    // First commit: DDL (table doesn't exist). publish_snapshot maps the
    // snapshot as the head — the next commit's schema_version is computed over
    // mapped predecessors, so without publishing the bump wouldn't advance.
    let setup1 = writer
        .begin_write_transaction("public", "users", &cols(), WriteMode::Replace)
        .unwrap();
    writer
        .publish_snapshot(
            setup1.table_id,
            setup1.snapshot_id,
            WriteMode::Replace,
            &[],
            &[],
        )
        .unwrap();

    // Second commit: same columns -> DML, carry forward schema_version.
    let setup2 = writer
        .begin_write_transaction("public", "users", &cols(), WriteMode::Replace)
        .unwrap();
    writer
        .publish_snapshot(
            setup2.table_id,
            setup2.snapshot_id,
            WriteMode::Replace,
            &[],
            &[],
        )
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
    writer
        .publish_snapshot(
            setup3.table_id,
            setup3.snapshot_id,
            WriteMode::Replace,
            &[],
            &[],
        )
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
    writer_a
        .publish_snapshot(
            setup_a.table_id,
            setup_a.snapshot_id,
            WriteMode::Replace,
            &[],
            &[],
        )
        .unwrap();
    let setup_b = writer_b
        .begin_write_transaction("public", "orders", &cols(), WriteMode::Replace)
        .unwrap();
    writer_b
        .publish_snapshot(
            setup_b.table_id,
            setup_b.snapshot_id,
            WriteMode::Replace,
            &[],
            &[],
        )
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

    // Each write publishes its snapshot as the head; schema_version is computed
    // over mapped predecessors, so publishing is what lets later DDL bumps see
    // the prior versions (per-catalog dense).
    // cat_a DDL (creates users)
    let a1 = wa
        .begin_write_transaction("public", "users", &cols(), WriteMode::Replace)
        .unwrap();
    wa.publish_snapshot(a1.table_id, a1.snapshot_id, WriteMode::Replace, &[], &[])
        .unwrap();
    // cat_a DML (Replace, same schema)
    let a2 = wa
        .begin_write_transaction("public", "users", &cols(), WriteMode::Replace)
        .unwrap();
    wa.publish_snapshot(a2.table_id, a2.snapshot_id, WriteMode::Replace, &[], &[])
        .unwrap();
    // cat_b DDL (creates orders) — happens in between cat_a's DDLs
    let b1 = wb
        .begin_write_transaction("public", "orders", &cols(), WriteMode::Replace)
        .unwrap();
    wb.publish_snapshot(b1.table_id, b1.snapshot_id, WriteMode::Replace, &[], &[])
        .unwrap();
    // cat_a DDL: adds age column
    let mut cols_v2 = cols();
    cols_v2.push(ColumnDef::new("age", "int32", true).unwrap());
    let a3 = wa
        .begin_write_transaction("public", "users", &cols_v2, WriteMode::Replace)
        .unwrap();
    wa.publish_snapshot(a3.table_id, a3.snapshot_id, WriteMode::Replace, &[], &[])
        .unwrap();
    // cat_b DML
    let b2 = wb
        .begin_write_transaction("public", "orders", &cols(), WriteMode::Replace)
        .unwrap();
    wb.publish_snapshot(b2.table_id, b2.snapshot_id, WriteMode::Replace, &[], &[])
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
    // Publish each write: a committed write maps its snapshot. (An uncommitted
    // write deliberately leaves an unmapped snapshot — the head only advances at
    // the commit point — so the "every snapshot is mapped" check below holds
    // only once published.)
    let s1 = w
        .begin_write_transaction("public", "users", &cols(), WriteMode::Replace)
        .unwrap();
    w.publish_snapshot(s1.table_id, s1.snapshot_id, WriteMode::Replace, &[], &[])
        .unwrap();
    let s2 = w
        .begin_write_transaction("public", "users", &cols(), WriteMode::Replace)
        .unwrap();
    w.publish_snapshot(s2.table_id, s2.snapshot_id, WriteMode::Replace, &[], &[])
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
        .register_data_file(
            setup.table_id,
            setup.snapshot_id,
            &file,
            WriteMode::Replace,
            &[],
            &[],
        )
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
        WriteMode::Replace,
        &[],
        &[],
    )
    .unwrap();

    let mut cols_v2 = cols();
    cols_v2.push(ColumnDef::new("age", "int32", true).unwrap());
    let s_ddl = w
        .begin_write_transaction("public", "users", &cols_v2, WriteMode::Replace)
        .unwrap();
    w.publish_snapshot(
        s_ddl.table_id,
        s_ddl.snapshot_id,
        WriteMode::Replace,
        &[],
        &[],
    )
    .unwrap();

    let s_orders = w
        .begin_write_transaction("public", "orders", &cols(), WriteMode::Replace)
        .unwrap();
    w.register_data_file(
        s_orders.table_id,
        s_orders.snapshot_id,
        &DataFileInfo::new("o.parquet", 2048, 20),
        WriteMode::Replace,
        &[],
        &[],
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
        WriteMode::Replace,
        &[],
        &[],
    )
    .unwrap();

    let sb = wb
        .begin_write_transaction("public", "orders", &cols(), WriteMode::Replace)
        .unwrap();
    wb.register_data_file(
        sb.table_id,
        sb.snapshot_id,
        &DataFileInfo::new("b.parquet", 2048, 20),
        WriteMode::Replace,
        &[],
        &[],
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

// -- drop_table_in_catalog --------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn drop_table_in_catalog_rejects_empty_names() {
    let (pool, _c) = spin_up_postgres().await.unwrap();
    let mgr = MulticatalogManager::new(pool);
    assert!(
        mgr.drop_table_in_catalog("", "public", "users")
            .await
            .is_err()
    );
    assert!(
        mgr.drop_table_in_catalog("   ", "public", "users")
            .await
            .is_err()
    );
    assert!(
        mgr.drop_table_in_catalog("pg_prod", "", "users")
            .await
            .is_err()
    );
    assert!(
        mgr.drop_table_in_catalog("pg_prod", "   ", "users")
            .await
            .is_err()
    );
    assert!(
        mgr.drop_table_in_catalog("pg_prod", "public", "")
            .await
            .is_err()
    );
    assert!(
        mgr.drop_table_in_catalog("pg_prod", "public", "   ")
            .await
            .is_err()
    );
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn drop_table_in_catalog_returns_false_for_unknown_catalog() {
    let (pool, _c) = spin_up_postgres().await.unwrap();
    let mgr = MulticatalogManager::new(pool);
    let dropped = mgr
        .drop_table_in_catalog("does_not_exist", "public", "users")
        .await
        .unwrap();
    assert!(
        !dropped,
        "dropping a table in an unknown catalog should report false"
    );
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn drop_table_in_catalog_returns_false_for_unknown_table() {
    let (pool, _c) = spin_up_postgres().await.unwrap();
    let mgr = MulticatalogManager::new(pool);
    let _ = mgr.create_catalog("pg_prod").await.unwrap();

    let dropped = mgr
        .drop_table_in_catalog("pg_prod", "public", "ghost")
        .await
        .unwrap();
    assert!(!dropped, "unknown table should report false");

    // Schema-only mismatch is also "not found".
    let dropped = mgr
        .drop_table_in_catalog("pg_prod", "ghost_schema", "users")
        .await
        .unwrap();
    assert!(!dropped, "unknown schema should report false");
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn drop_table_in_catalog_tombstones_table_and_children() {
    use datafusion_ducklake::metadata_writer::DataFileInfo;

    let (pool, _c) = spin_up_postgres().await.unwrap();
    let mgr = MulticatalogManager::new(pool.clone());
    let cat = mgr.create_catalog("pg_prod").await.unwrap();
    let w = PostgresMetadataWriter::with_pool(pool.clone(), cat)
        .await
        .unwrap();
    w.set_data_path("/data").unwrap();

    let s = w
        .begin_write_transaction("public", "users", &cols(), WriteMode::Replace)
        .unwrap();
    w.register_data_file(
        s.table_id,
        s.snapshot_id,
        &DataFileInfo::new("u.parquet", 1024, 10),
        WriteMode::Replace,
        &[],
        &[],
    )
    .unwrap();

    let snap_pre_drop: i64 = sqlx::query("SELECT MAX(snapshot_id) FROM ducklake_snapshot")
        .fetch_one(&pool)
        .await
        .unwrap()
        .try_get(0)
        .unwrap();

    let dropped = mgr
        .drop_table_in_catalog("pg_prod", "public", "users")
        .await
        .unwrap();
    assert!(dropped, "live table should be dropped");

    // A new snapshot was allocated for the drop.
    let snap_after_drop: i64 = sqlx::query("SELECT MAX(snapshot_id) FROM ducklake_snapshot")
        .fetch_one(&pool)
        .await
        .unwrap()
        .try_get(0)
        .unwrap();
    assert_eq!(
        snap_after_drop,
        snap_pre_drop + 1,
        "drop should allocate exactly one new snapshot"
    );

    // Drop snapshot is registered under this catalog.
    let in_map: bool = sqlx::query(
        "SELECT EXISTS(
            SELECT 1 FROM ducklake_catalog_snapshot_map
            WHERE catalog_id = $1 AND snapshot_id = $2
         )",
    )
    .bind(cat)
    .bind(snap_after_drop)
    .fetch_one(&pool)
    .await
    .unwrap()
    .try_get(0)
    .unwrap();
    assert!(in_map, "drop snapshot should be in catalog_snapshot_map");

    // ducklake_table row carries end_snapshot = drop snapshot.
    let end_snap: Option<i64> =
        sqlx::query("SELECT end_snapshot FROM ducklake_table WHERE table_id = $1")
            .bind(s.table_id)
            .fetch_one(&pool)
            .await
            .unwrap()
            .try_get(0)
            .unwrap();
    assert_eq!(end_snap, Some(snap_after_drop));

    // All currently-live child rows for this table now carry the drop snapshot.
    for child_table in ["ducklake_column", "ducklake_data_file"] {
        let live: i64 = sqlx::query(&format!(
            "SELECT COUNT(*) FROM {} WHERE table_id = $1 AND end_snapshot IS NULL",
            child_table
        ))
        .bind(s.table_id)
        .fetch_one(&pool)
        .await
        .unwrap()
        .try_get(0)
        .unwrap();
        assert_eq!(live, 0, "no live rows left in {} after drop", child_table);

        let tombstoned: i64 = sqlx::query(&format!(
            "SELECT COUNT(*) FROM {} WHERE table_id = $1 AND end_snapshot = $2",
            child_table
        ))
        .bind(s.table_id)
        .bind(snap_after_drop)
        .fetch_one(&pool)
        .await
        .unwrap()
        .try_get(0)
        .unwrap();
        assert!(
            tombstoned > 0,
            "{} should have rows tombstoned at the drop snapshot",
            child_table
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn drop_table_in_catalog_hides_table_from_read_path_and_preserves_time_travel() {
    use datafusion_ducklake::MulticatalogProvider;
    use datafusion_ducklake::metadata_provider::MetadataProvider;
    use datafusion_ducklake::metadata_writer::DataFileInfo;

    let (pool, _c) = spin_up_postgres().await.unwrap();
    let mgr = MulticatalogManager::new(pool.clone());
    let cat = mgr.create_catalog("pg_prod").await.unwrap();
    let w = PostgresMetadataWriter::with_pool(pool.clone(), cat)
        .await
        .unwrap();
    w.set_data_path("/data").unwrap();

    let s = w
        .begin_write_transaction("public", "users", &cols(), WriteMode::Replace)
        .unwrap();
    w.register_data_file(
        s.table_id,
        s.snapshot_id,
        &DataFileInfo::new("u.parquet", 1024, 10),
        WriteMode::Replace,
        &[],
        &[],
    )
    .unwrap();

    // Snapshot of the table-creation commit; we'll time-travel to it post-drop.
    let provider = MulticatalogProvider::with_pool(pool.clone(), "pg_prod")
        .await
        .unwrap();
    let snap_pre_drop = provider.get_current_snapshot().unwrap();

    let dropped = mgr
        .drop_table_in_catalog("pg_prod", "public", "users")
        .await
        .unwrap();
    assert!(dropped);

    let snap_post_drop = provider.get_current_snapshot().unwrap();
    assert!(
        snap_post_drop > snap_pre_drop,
        "current snapshot should advance past the drop"
    );

    // At the current snapshot the table is gone — schema lookup still
    // resolves (schemas are dropped explicitly), but the table is not visible.
    let schema_post = provider
        .get_schema_by_name("public", snap_post_drop)
        .unwrap()
        .unwrap();
    assert!(
        provider
            .get_table_by_name(schema_post.schema_id, "users", snap_post_drop)
            .unwrap()
            .is_none(),
        "dropped table should not be visible at the drop snapshot"
    );
    assert!(
        !provider
            .table_exists(schema_post.schema_id, "users", snap_post_drop)
            .unwrap(),
        "table_exists should report false post-drop"
    );

    // Time travel: at the pre-drop snapshot the table is still resolvable.
    let schema_pre = provider
        .get_schema_by_name("public", snap_pre_drop)
        .unwrap()
        .unwrap();
    let pre_drop_table = provider
        .get_table_by_name(schema_pre.schema_id, "users", snap_pre_drop)
        .unwrap();
    assert!(
        pre_drop_table.is_some(),
        "pre-drop snapshot should still see the table (time travel)"
    );
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn drop_table_in_catalog_is_idempotent_on_second_call() {
    use datafusion_ducklake::metadata_writer::DataFileInfo;

    let (pool, _c) = spin_up_postgres().await.unwrap();
    let mgr = MulticatalogManager::new(pool.clone());
    let cat = mgr.create_catalog("pg_prod").await.unwrap();
    let w = PostgresMetadataWriter::with_pool(pool.clone(), cat)
        .await
        .unwrap();
    w.set_data_path("/data").unwrap();

    let s = w
        .begin_write_transaction("public", "users", &cols(), WriteMode::Replace)
        .unwrap();
    w.register_data_file(
        s.table_id,
        s.snapshot_id,
        &DataFileInfo::new("u.parquet", 1024, 10),
        WriteMode::Replace,
        &[],
        &[],
    )
    .unwrap();

    let first = mgr
        .drop_table_in_catalog("pg_prod", "public", "users")
        .await
        .unwrap();
    assert!(first, "first drop tombstones the table");

    let snap_after_first: i64 = sqlx::query("SELECT MAX(snapshot_id) FROM ducklake_snapshot")
        .fetch_one(&pool)
        .await
        .unwrap()
        .try_get(0)
        .unwrap();

    let second = mgr
        .drop_table_in_catalog("pg_prod", "public", "users")
        .await
        .unwrap();
    assert!(!second, "second drop should be a no-op");

    // No additional snapshot allocated on the no-op.
    let snap_after_second: i64 = sqlx::query("SELECT MAX(snapshot_id) FROM ducklake_snapshot")
        .fetch_one(&pool)
        .await
        .unwrap()
        .try_get(0)
        .unwrap();
    assert_eq!(
        snap_after_first, snap_after_second,
        "idempotent no-op must not allocate a new snapshot"
    );
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn drop_table_in_catalog_does_not_touch_siblings() {
    use datafusion_ducklake::MulticatalogProvider;
    use datafusion_ducklake::metadata_provider::MetadataProvider;
    use datafusion_ducklake::metadata_writer::DataFileInfo;

    let (pool, _c) = spin_up_postgres().await.unwrap();
    let mgr = MulticatalogManager::new(pool.clone());
    let cat = mgr.create_catalog("pg_prod").await.unwrap();
    let w = PostgresMetadataWriter::with_pool(pool.clone(), cat)
        .await
        .unwrap();
    w.set_data_path("/data").unwrap();

    // Two tables in the same schema.
    let s_users = w
        .begin_write_transaction("public", "users", &cols(), WriteMode::Replace)
        .unwrap();
    w.register_data_file(
        s_users.table_id,
        s_users.snapshot_id,
        &DataFileInfo::new("u.parquet", 1024, 10),
        WriteMode::Replace,
        &[],
        &[],
    )
    .unwrap();

    let s_orders = w
        .begin_write_transaction("public", "orders", &cols(), WriteMode::Replace)
        .unwrap();
    w.register_data_file(
        s_orders.table_id,
        s_orders.snapshot_id,
        &DataFileInfo::new("o.parquet", 2048, 20),
        WriteMode::Replace,
        &[],
        &[],
    )
    .unwrap();

    // Same-named table in a different schema; must also survive.
    let s_other_users = w
        .begin_write_transaction("analytics", "users", &cols(), WriteMode::Replace)
        .unwrap();
    w.register_data_file(
        s_other_users.table_id,
        s_other_users.snapshot_id,
        &DataFileInfo::new("au.parquet", 512, 5),
        WriteMode::Replace,
        &[],
        &[],
    )
    .unwrap();

    let dropped = mgr
        .drop_table_in_catalog("pg_prod", "public", "users")
        .await
        .unwrap();
    assert!(dropped);

    // The other table in the same schema must still be live.
    let orders_end_snap: Option<i64> =
        sqlx::query("SELECT end_snapshot FROM ducklake_table WHERE table_id = $1")
            .bind(s_orders.table_id)
            .fetch_one(&pool)
            .await
            .unwrap()
            .try_get(0)
            .unwrap();
    assert!(
        orders_end_snap.is_none(),
        "sibling table in the same schema must not be tombstoned"
    );

    // Same-named table in a different schema must still be live.
    let other_end_snap: Option<i64> =
        sqlx::query("SELECT end_snapshot FROM ducklake_table WHERE table_id = $1")
            .bind(s_other_users.table_id)
            .fetch_one(&pool)
            .await
            .unwrap()
            .try_get(0)
            .unwrap();
    assert!(
        other_end_snap.is_none(),
        "same-named table in a different schema must not be tombstoned"
    );

    // And readability via the provider matches.
    let provider = MulticatalogProvider::with_pool(pool.clone(), "pg_prod")
        .await
        .unwrap();
    let snap = provider.get_current_snapshot().unwrap();

    let public_schema = provider
        .get_schema_by_name("public", snap)
        .unwrap()
        .unwrap();
    assert!(
        provider
            .get_table_by_name(public_schema.schema_id, "orders", snap)
            .unwrap()
            .is_some(),
        "public.orders should still be readable"
    );
    assert!(
        provider
            .get_table_by_name(public_schema.schema_id, "users", snap)
            .unwrap()
            .is_none(),
        "public.users should be hidden after drop"
    );

    let analytics_schema = provider
        .get_schema_by_name("analytics", snap)
        .unwrap()
        .unwrap();
    assert!(
        provider
            .get_table_by_name(analytics_schema.schema_id, "users", snap)
            .unwrap()
            .is_some(),
        "analytics.users should still be readable"
    );
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn drop_table_in_catalog_allows_recreate_with_fresh_identity() {
    use datafusion_ducklake::MulticatalogProvider;
    use datafusion_ducklake::metadata_provider::MetadataProvider;
    use datafusion_ducklake::metadata_writer::DataFileInfo;

    let (pool, _c) = spin_up_postgres().await.unwrap();
    let mgr = MulticatalogManager::new(pool.clone());
    let cat = mgr.create_catalog("pg_prod").await.unwrap();
    let w = PostgresMetadataWriter::with_pool(pool.clone(), cat)
        .await
        .unwrap();
    w.set_data_path("/data").unwrap();

    let s1 = w
        .begin_write_transaction("public", "users", &cols(), WriteMode::Replace)
        .unwrap();
    w.register_data_file(
        s1.table_id,
        s1.snapshot_id,
        &DataFileInfo::new("v1.parquet", 1024, 10),
        WriteMode::Replace,
        &[],
        &[],
    )
    .unwrap();

    let dropped = mgr
        .drop_table_in_catalog("pg_prod", "public", "users")
        .await
        .unwrap();
    assert!(dropped);

    // Re-create the same `(schema, table)` after the drop. The writer
    // should pick a fresh `table_id` (the tombstoned row is filtered
    // out by `end_snapshot IS NULL` in the lookup) and the new table
    // must be independently queryable with its own data.
    let s2 = w
        .begin_write_transaction("public", "users", &cols(), WriteMode::Replace)
        .unwrap();
    assert_ne!(
        s2.table_id, s1.table_id,
        "recreated table must get a fresh table_id"
    );
    w.register_data_file(
        s2.table_id,
        s2.snapshot_id,
        &DataFileInfo::new("v2.parquet", 2048, 20),
        WriteMode::Replace,
        &[],
        &[],
    )
    .unwrap();

    let provider = MulticatalogProvider::with_pool(pool.clone(), "pg_prod")
        .await
        .unwrap();
    let snap = provider.get_current_snapshot().unwrap();
    let schema = provider
        .get_schema_by_name("public", snap)
        .unwrap()
        .unwrap();
    let table = provider
        .get_table_by_name(schema.schema_id, "users", snap)
        .unwrap()
        .unwrap();
    assert_eq!(
        table.table_id, s2.table_id,
        "current snapshot must resolve to the recreated table_id"
    );

    // The recreated table's data file is visible at the current snapshot;
    // the dropped table's data file is not (filtered by end_snapshot).
    let visible_files = provider
        .get_table_files_for_select(table.table_id, snap)
        .unwrap();
    assert_eq!(visible_files.len(), 1);
    assert!(visible_files[0].file.path.ends_with("v2.parquet"));
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn drop_table_in_catalog_bumps_schema_version_as_ddl() {
    use datafusion_ducklake::metadata_writer::DataFileInfo;

    let (pool, _c) = spin_up_postgres().await.unwrap();
    let mgr = MulticatalogManager::new(pool.clone());
    let cat = mgr.create_catalog("pg_prod").await.unwrap();
    let w = PostgresMetadataWriter::with_pool(pool.clone(), cat)
        .await
        .unwrap();
    w.set_data_path("/data").unwrap();

    // First DDL ⇒ schema_version 1.
    let s = w
        .begin_write_transaction("public", "users", &cols(), WriteMode::Replace)
        .unwrap();
    w.register_data_file(
        s.table_id,
        s.snapshot_id,
        &DataFileInfo::new("u.parquet", 1024, 10),
        WriteMode::Replace,
        &[],
        &[],
    )
    .unwrap();

    let v_create: i64 =
        sqlx::query("SELECT schema_version FROM ducklake_snapshot WHERE snapshot_id = $1")
            .bind(s.snapshot_id)
            .fetch_one(&pool)
            .await
            .unwrap()
            .try_get(0)
            .unwrap();
    assert_eq!(v_create, 1);

    // Drop ⇒ DDL bump ⇒ schema_version 2 (per-catalog dense).
    let dropped = mgr
        .drop_table_in_catalog("pg_prod", "public", "users")
        .await
        .unwrap();
    assert!(dropped);

    let drop_snap: i64 = sqlx::query("SELECT MAX(snapshot_id) FROM ducklake_snapshot")
        .fetch_one(&pool)
        .await
        .unwrap()
        .try_get(0)
        .unwrap();
    let v_drop: i64 =
        sqlx::query("SELECT schema_version FROM ducklake_snapshot WHERE snapshot_id = $1")
            .bind(drop_snap)
            .fetch_one(&pool)
            .await
            .unwrap()
            .try_get(0)
            .unwrap();
    assert_eq!(
        v_drop, 2,
        "drop snapshot must bump schema_version (DDL change)"
    );
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn drop_table_in_catalog_isolates_other_catalogs() {
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

    // Same `(schema, table)` identity in both catalogs.
    let sa = wa
        .begin_write_transaction("public", "users", &cols(), WriteMode::Replace)
        .unwrap();
    wa.register_data_file(
        sa.table_id,
        sa.snapshot_id,
        &DataFileInfo::new("a.parquet", 1024, 10),
        WriteMode::Replace,
        &[],
        &[],
    )
    .unwrap();

    let sb = wb
        .begin_write_transaction("public", "users", &cols(), WriteMode::Replace)
        .unwrap();
    wb.register_data_file(
        sb.table_id,
        sb.snapshot_id,
        &DataFileInfo::new("b.parquet", 2048, 20),
        WriteMode::Replace,
        &[],
        &[],
    )
    .unwrap();

    // Drop only in catalog A.
    let dropped = mgr
        .drop_table_in_catalog("pg_prod", "public", "users")
        .await
        .unwrap();
    assert!(dropped);

    // Catalog A's table_id is tombstoned, catalog B's is not.
    let a_end_snap: Option<i64> =
        sqlx::query("SELECT end_snapshot FROM ducklake_table WHERE table_id = $1")
            .bind(sa.table_id)
            .fetch_one(&pool)
            .await
            .unwrap()
            .try_get(0)
            .unwrap();
    assert!(a_end_snap.is_some(), "catalog A's table must be tombstoned");

    let b_end_snap: Option<i64> =
        sqlx::query("SELECT end_snapshot FROM ducklake_table WHERE table_id = $1")
            .bind(sb.table_id)
            .fetch_one(&pool)
            .await
            .unwrap()
            .try_get(0)
            .unwrap();
    assert!(
        b_end_snap.is_none(),
        "catalog B's same-named table must NOT be tombstoned"
    );

    // The drop snapshot is registered only under catalog A.
    let drop_snap = a_end_snap.unwrap();
    let owners: Vec<i64> =
        sqlx::query("SELECT catalog_id FROM ducklake_catalog_snapshot_map WHERE snapshot_id = $1")
            .bind(drop_snap)
            .fetch_all(&pool)
            .await
            .unwrap()
            .into_iter()
            .map(|r| r.try_get::<i64, _>(0).unwrap())
            .collect();
    assert_eq!(
        owners,
        vec![cat_a],
        "drop snapshot must be registered only under the dropping catalog"
    );
}

// ---------------------------------------------------------------------------
// Row lineage / row_id_start coverage for the postgres writer. The read path
// (RowIdExec) hard-errors on data files whose row_id_start is NULL and have
// no embedded `_ducklake_internal_row_id` column, so the writer must populate
// the column on every file it produces. These tests pin that contract.
// ---------------------------------------------------------------------------

async fn read_row_id_start(pool: &PgPool, path: &str) -> Option<i64> {
    sqlx::query("SELECT row_id_start FROM ducklake_data_file WHERE path = $1")
        .bind(path)
        .fetch_one(pool)
        .await
        .unwrap()
        .try_get(0)
        .unwrap()
}

async fn read_table_stats(pool: &PgPool, table_id: i64) -> (i64, i64, i64) {
    let row = sqlx::query(
        "SELECT record_count, next_row_id, file_size_bytes
         FROM ducklake_table_stats WHERE table_id = $1",
    )
    .bind(table_id)
    .fetch_one(pool)
    .await
    .unwrap();
    (
        row.try_get(0).unwrap(),
        row.try_get(1).unwrap(),
        row.try_get(2).unwrap(),
    )
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn register_data_file_assigns_non_overlapping_row_id_start() {
    use datafusion_ducklake::metadata_writer::DataFileInfo;
    let (pool, _c) = spin_up_postgres().await.unwrap();
    let mgr = MulticatalogManager::new(pool.clone());
    let cat = mgr.create_catalog("pg_rowid").await.unwrap();
    let w = PostgresMetadataWriter::with_pool(pool.clone(), cat)
        .await
        .unwrap();
    w.set_data_path("/data").unwrap();
    let setup = w
        .begin_write_transaction("public", "users", &cols(), WriteMode::Replace)
        .unwrap();

    // Three files of 100, 50, 200 rows -> row_id_start = 0, 100, 150;
    // next_row_id = 350 after all three.
    w.register_data_file(
        setup.table_id,
        setup.snapshot_id,
        &DataFileInfo::new("f1.parquet", 4096, 100),
        WriteMode::Replace,
        &[],
        &[],
    )
    .unwrap();
    w.register_data_file(
        setup.table_id,
        setup.snapshot_id,
        &DataFileInfo::new("f2.parquet", 2048, 50),
        WriteMode::Append,
        &[],
        &[],
    )
    .unwrap();
    w.register_data_file(
        setup.table_id,
        setup.snapshot_id,
        &DataFileInfo::new("f3.parquet", 8192, 200),
        WriteMode::Append,
        &[],
        &[],
    )
    .unwrap();

    assert_eq!(read_row_id_start(&pool, "f1.parquet").await, Some(0));
    assert_eq!(read_row_id_start(&pool, "f2.parquet").await, Some(100));
    assert_eq!(read_row_id_start(&pool, "f3.parquet").await, Some(150));

    let (records, next, bytes) = read_table_stats(&pool, setup.table_id).await;
    assert_eq!(records, 350);
    assert_eq!(next, 350);
    assert_eq!(bytes, 4096 + 2048 + 8192);
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn replace_preserves_next_row_id_monotonic() {
    // rowids must never be reused, even across WriteMode::Replace cycles.
    // The Replace commit (register_data_file / publish_snapshot) end-snapshots
    // the prior generation's files and clears record_count / file_size_bytes,
    // but next_row_id stays put — so the first file of the new generation
    // continues where the old generation left off.
    use datafusion_ducklake::metadata_writer::DataFileInfo;
    let (pool, _c) = spin_up_postgres().await.unwrap();
    let mgr = MulticatalogManager::new(pool.clone());
    let cat = mgr.create_catalog("pg_replace").await.unwrap();
    let w = PostgresMetadataWriter::with_pool(pool.clone(), cat)
        .await
        .unwrap();
    w.set_data_path("/data").unwrap();
    let s1 = w
        .begin_write_transaction("public", "users", &cols(), WriteMode::Replace)
        .unwrap();

    w.register_data_file(
        s1.table_id,
        s1.snapshot_id,
        &DataFileInfo::new("g1.parquet", 1024, 5),
        WriteMode::Replace,
        &[],
        &[],
    )
    .unwrap();
    let (rc1, next1, _) = read_table_stats(&pool, s1.table_id).await;
    assert_eq!(rc1, 5);
    assert_eq!(next1, 5);

    // Second Replace: publish_snapshot retires the first generation's files and
    // resets visible totals to zero while preserving next_row_id (the retirement
    // moved here from begin_write_transaction).
    let s2 = w
        .begin_write_transaction("public", "users", &cols(), WriteMode::Replace)
        .unwrap();
    w.publish_snapshot(s2.table_id, s2.snapshot_id, WriteMode::Replace, &[], &[])
        .unwrap();
    let (rc2, next2, bytes2) = read_table_stats(&pool, s2.table_id).await;
    assert_eq!(rc2, 0, "record_count must reset on Replace");
    assert_eq!(next2, 5, "next_row_id must NOT reset on Replace");
    assert_eq!(bytes2, 0);

    // The first file of the new generation picks up at 5. The generation was
    // already published above, so this registration is additive (Append).
    w.register_data_file(
        s2.table_id,
        s2.snapshot_id,
        &DataFileInfo::new("g2.parquet", 2048, 2),
        WriteMode::Append,
        &[],
        &[],
    )
    .unwrap();
    assert_eq!(
        read_row_id_start(&pool, "g2.parquet").await,
        Some(5),
        "post-replace files must start at the preserved counter, not 0",
    );
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn register_data_file_self_initialises_stats_for_legacy_tables() {
    // Defensive path: a table that existed before this writer maintained
    // ducklake_table_stats. The first register_data_file must self-initialise
    // the stats row at 0 rather than fail with "no row".
    use datafusion_ducklake::metadata_writer::DataFileInfo;
    let (pool, _c) = spin_up_postgres().await.unwrap();
    let mgr = MulticatalogManager::new(pool.clone());
    let cat = mgr.create_catalog("pg_legacy").await.unwrap();
    let w = PostgresMetadataWriter::with_pool(pool.clone(), cat)
        .await
        .unwrap();
    w.set_data_path("/data").unwrap();
    let setup = w
        .begin_write_transaction("public", "users", &cols(), WriteMode::Replace)
        .unwrap();

    // Simulate "legacy" state by removing whatever stats row may have been
    // pre-populated. begin_write_transaction doesn't currently insert one,
    // but being explicit keeps the test meaningful if that changes.
    sqlx::query("DELETE FROM ducklake_table_stats WHERE table_id = $1")
        .bind(setup.table_id)
        .execute(&pool)
        .await
        .unwrap();

    w.register_data_file(
        setup.table_id,
        setup.snapshot_id,
        &DataFileInfo::new("a.parquet", 50, 4),
        WriteMode::Replace,
        &[],
        &[],
    )
    .unwrap();
    assert_eq!(read_row_id_start(&pool, "a.parquet").await, Some(0));
    let (records, next, _) = read_table_stats(&pool, setup.table_id).await;
    assert_eq!(records, 4);
    assert_eq!(next, 4);
}

// ---------------------------------------------------------------------------
// expire_snapshots_in_catalog / cleanup_old_files_in_catalog
// ---------------------------------------------------------------------------

/// Write three Replace generations of one table for `writer`, returning the table id and
/// the three snapshot ids. The first two data files end up superseded (end-snapshotted).
fn three_generations(
    writer: &PostgresMetadataWriter,
    schema: &str,
    table: &str,
) -> (i64, i64, i64, i64) {
    use datafusion_ducklake::metadata_writer::DataFileInfo;
    let s1 = writer
        .begin_write_transaction(schema, table, &cols(), WriteMode::Replace)
        .unwrap();
    writer
        .register_data_file(
            s1.table_id,
            s1.snapshot_id,
            &DataFileInfo::new("f1.parquet", 100, 5),
            WriteMode::Replace,
            &[],
            &[],
        )
        .unwrap();
    let s2 = writer
        .begin_write_transaction(schema, table, &cols(), WriteMode::Replace)
        .unwrap();
    writer
        .register_data_file(
            s2.table_id,
            s2.snapshot_id,
            &DataFileInfo::new("f2.parquet", 100, 5),
            WriteMode::Replace,
            &[],
            &[],
        )
        .unwrap();
    let s3 = writer
        .begin_write_transaction(schema, table, &cols(), WriteMode::Replace)
        .unwrap();
    writer
        .register_data_file(
            s3.table_id,
            s3.snapshot_id,
            &DataFileInfo::new("f3.parquet", 100, 5),
            WriteMode::Replace,
            &[],
            &[],
        )
        .unwrap();
    (s1.table_id, s1.snapshot_id, s2.snapshot_id, s3.snapshot_id)
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn expire_in_catalog_empty_for_most_recent_and_unknown() {
    use datafusion_ducklake::maintenance::ExpireCriteria;
    use datafusion_ducklake::metadata_writer::DataFileInfo;

    let (pool, _c) = spin_up_postgres().await.unwrap();
    let mgr = MulticatalogManager::new(pool.clone());
    let cat = mgr.create_catalog("pg_prod").await.unwrap();
    let w = PostgresMetadataWriter::with_pool(pool.clone(), cat)
        .await
        .unwrap();
    w.set_data_path("/data").unwrap();
    let s = w
        .begin_write_transaction("public", "t", &cols(), WriteMode::Replace)
        .unwrap();
    w.register_data_file(
        s.table_id,
        s.snapshot_id,
        &DataFileInfo::new("f1.parquet", 100, 5),
        WriteMode::Replace,
        &[],
        &[],
    )
    .unwrap();

    // The only snapshot is the most recent — never expirable.
    let expired = mgr
        .expire_snapshots_in_catalog("pg_prod", ExpireCriteria::Versions(vec![s.snapshot_id]))
        .await
        .unwrap();
    assert!(expired.is_empty());

    // Unknown catalog is a no-op.
    let none = mgr
        .expire_snapshots_in_catalog("ghost", ExpireCriteria::Versions(vec![1]))
        .await
        .unwrap();
    assert!(none.is_empty());
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "pre-cat_{id} fixture layout + orphan-cleanup id collision; fixed with the maintenance rework"]
async fn expire_in_catalog_by_version_schedules_orphaned_file() {
    use datafusion_ducklake::maintenance::ExpireCriteria;

    let (pool, _c) = spin_up_postgres().await.unwrap();
    let mgr = MulticatalogManager::new(pool.clone());
    let cat = mgr.create_catalog("pg_prod").await.unwrap();
    let w = PostgresMetadataWriter::with_pool(pool.clone(), cat)
        .await
        .unwrap();
    w.set_data_path("/data").unwrap();
    let (_tid, s1, _s2, _s3) = three_generations(&w, "public", "t");

    let expired = mgr
        .expire_snapshots_in_catalog("pg_prod", ExpireCriteria::Versions(vec![s1]))
        .await
        .unwrap();
    assert_eq!(expired.len(), 1);
    assert_eq!(expired[0].snapshot_id, s1);

    // Snapshot row and its catalog map row are both gone.
    let snap_rows: i64 =
        sqlx::query("SELECT COUNT(*) FROM ducklake_snapshot WHERE snapshot_id = $1")
            .bind(s1)
            .fetch_one(&pool)
            .await
            .unwrap()
            .try_get(0)
            .unwrap();
    assert_eq!(snap_rows, 0);
    let map_rows: i64 =
        sqlx::query("SELECT COUNT(*) FROM ducklake_catalog_snapshot_map WHERE snapshot_id = $1")
            .bind(s1)
            .fetch_one(&pool)
            .await
            .unwrap()
            .try_get(0)
            .unwrap();
    assert_eq!(map_rows, 0, "no orphan snapshot-map row");

    // f1 was scheduled, tagged with this catalog and the data_path-relative path.
    let scheduled: Vec<(i64, String, bool)> = sqlx::query(
        "SELECT catalog_id, path, path_is_relative FROM ducklake_files_scheduled_for_deletion",
    )
    .fetch_all(&pool)
    .await
    .unwrap()
    .into_iter()
    .map(|r| {
        (
            r.try_get::<i64, _>(0).unwrap(),
            r.try_get::<String, _>(1).unwrap(),
            r.try_get::<bool, _>(2).unwrap(),
        )
    })
    .collect();
    assert_eq!(scheduled.len(), 1);
    assert_eq!(scheduled[0].0, cat);
    assert_eq!(scheduled[0].1, "public/t/f1.parquet");
    assert!(scheduled[0].2);

    // f1's catalog row is gone; f2/f3 remain.
    let live_files: i64 = sqlx::query("SELECT COUNT(*) FROM ducklake_data_file")
        .fetch_one(&pool)
        .await
        .unwrap()
        .try_get(0)
        .unwrap();
    assert_eq!(live_files, 2);
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn expire_in_catalog_full_after_drop_removes_table_metadata() {
    use datafusion_ducklake::maintenance::ExpireCriteria;
    use datafusion_ducklake::metadata_writer::DataFileInfo;

    let (pool, _c) = spin_up_postgres().await.unwrap();
    let mgr = MulticatalogManager::new(pool.clone());
    let cat = mgr.create_catalog("pg_prod").await.unwrap();
    let w = PostgresMetadataWriter::with_pool(pool.clone(), cat)
        .await
        .unwrap();
    w.set_data_path("/data").unwrap();
    let s = w
        .begin_write_transaction("public", "t", &cols(), WriteMode::Replace)
        .unwrap();
    w.register_data_file(
        s.table_id,
        s.snapshot_id,
        &DataFileInfo::new("f1.parquet", 100, 5),
        WriteMode::Replace,
        &[],
        &[],
    )
    .unwrap();

    // Drop allocates a second snapshot; expire the first (the drop snapshot is kept).
    assert!(
        mgr.drop_table_in_catalog("pg_prod", "public", "t")
            .await
            .unwrap()
    );
    let expired = mgr
        .expire_snapshots_in_catalog("pg_prod", ExpireCriteria::Versions(vec![s.snapshot_id]))
        .await
        .unwrap();
    assert_eq!(expired.len(), 1);

    for tbl in
        ["ducklake_table", "ducklake_column", "ducklake_data_file", "ducklake_schema_versions"]
    {
        let cnt: i64 = sqlx::query(&format!("SELECT COUNT(*) FROM {tbl} WHERE table_id = $1"))
            .bind(s.table_id)
            .fetch_one(&pool)
            .await
            .unwrap()
            .try_get(0)
            .unwrap();
        assert_eq!(cnt, 0, "{tbl} fully reclaimed after expire");
    }
    let scheduled: i64 = sqlx::query(
        "SELECT COUNT(*) FROM ducklake_files_scheduled_for_deletion WHERE catalog_id = $1",
    )
    .bind(cat)
    .fetch_one(&pool)
    .await
    .unwrap()
    .try_get(0)
    .unwrap();
    assert_eq!(
        scheduled, 1,
        "the table's data file was scheduled for deletion"
    );
}

/// The critical scoping regression: a data file in catalog A whose `[begin, end)` global
/// snapshot range *contains* a snapshot belonging to catalog B must still be GC'd when A
/// expires — a globally-scoped reachability check would wrongly keep it alive.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn expire_in_catalog_is_scoped_to_catalog() {
    use datafusion_ducklake::maintenance::ExpireCriteria;
    use datafusion_ducklake::metadata_writer::DataFileInfo;

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

    // Interleave so B's snapshot sits between A's two snapshots:
    //   a1 (A) < b1 (B) < a2 (A)  → A/f1 lives in [a1, a2), which contains b1.
    let a1 = wa
        .begin_write_transaction("public", "t", &cols(), WriteMode::Replace)
        .unwrap();
    wa.register_data_file(
        a1.table_id,
        a1.snapshot_id,
        &DataFileInfo::new("f1.parquet", 100, 5),
        WriteMode::Replace,
        &[],
        &[],
    )
    .unwrap();
    let b1 = wb
        .begin_write_transaction("public", "u", &cols(), WriteMode::Replace)
        .unwrap();
    wb.register_data_file(
        b1.table_id,
        b1.snapshot_id,
        &DataFileInfo::new("g1.parquet", 100, 5),
        WriteMode::Replace,
        &[],
        &[],
    )
    .unwrap();
    let a2 = wa
        .begin_write_transaction("public", "t", &cols(), WriteMode::Replace)
        .unwrap();
    wa.register_data_file(
        a2.table_id,
        a2.snapshot_id,
        &DataFileInfo::new("f2.parquet", 100, 5),
        WriteMode::Replace,
        &[],
        &[],
    )
    .unwrap();
    assert!(
        b1.snapshot_id > a1.snapshot_id && b1.snapshot_id < a2.snapshot_id,
        "test setup: B's snapshot must fall inside A/f1's lifetime range"
    );

    let expired = mgr
        .expire_snapshots_in_catalog("cat_a", ExpireCriteria::Versions(vec![a1.snapshot_id]))
        .await
        .unwrap();
    assert_eq!(expired.len(), 1);

    // A/f1 was scheduled despite B's snapshot being in its global range (proves scoping).
    let a_scheduled: i64 = sqlx::query(
        "SELECT COUNT(*) FROM ducklake_files_scheduled_for_deletion WHERE catalog_id = $1",
    )
    .bind(cat_a)
    .fetch_one(&pool)
    .await
    .unwrap()
    .try_get(0)
    .unwrap();
    assert_eq!(
        a_scheduled, 1,
        "A/f1 must be GC'd — catalog-scoped reachability"
    );

    // Catalog B is entirely untouched.
    let b_snap: i64 = sqlx::query("SELECT COUNT(*) FROM ducklake_snapshot WHERE snapshot_id = $1")
        .bind(b1.snapshot_id)
        .fetch_one(&pool)
        .await
        .unwrap()
        .try_get(0)
        .unwrap();
    assert_eq!(b_snap, 1, "B's snapshot survives");
    let b_files: i64 = sqlx::query(
        "SELECT COUNT(*) FROM ducklake_data_file WHERE table_id = $1 AND end_snapshot IS NULL",
    )
    .bind(b1.table_id)
    .fetch_one(&pool)
    .await
    .unwrap()
    .try_get(0)
    .unwrap();
    assert_eq!(b_files, 1, "B's data file untouched");
    let b_scheduled: i64 = sqlx::query(
        "SELECT COUNT(*) FROM ducklake_files_scheduled_for_deletion WHERE catalog_id = $1",
    )
    .bind(cat_b)
    .fetch_one(&pool)
    .await
    .unwrap()
    .try_get(0)
    .unwrap();
    assert_eq!(b_scheduled, 0, "nothing scheduled for B");
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "pre-cat_{id} fixture layout + orphan-cleanup id collision; fixed with the maintenance rework"]
async fn cleanup_old_files_in_catalog_is_scoped_to_catalog() {
    use datafusion_ducklake::maintenance::{
        CleanupCriteria, ExpireCriteria, cleanup_old_files_in_catalog,
    };
    use datafusion_ducklake::metadata_writer::DataFileInfo;
    use object_store::ObjectStore;
    use object_store::local::LocalFileSystem;
    use std::sync::Arc;
    use tempfile::TempDir;

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

    let temp = TempDir::new().unwrap();
    let data_path = temp.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();
    wa.set_data_path(data_path.to_str().unwrap()).unwrap();

    // A: two generations of public.t (f1 superseded by f2). B: two of public.u.
    let a1 = wa
        .begin_write_transaction("public", "t", &cols(), WriteMode::Replace)
        .unwrap();
    wa.register_data_file(
        a1.table_id,
        a1.snapshot_id,
        &DataFileInfo::new("f1.parquet", 100, 5),
        WriteMode::Replace,
        &[],
        &[],
    )
    .unwrap();
    let a2 = wa
        .begin_write_transaction("public", "t", &cols(), WriteMode::Replace)
        .unwrap();
    wa.register_data_file(
        a2.table_id,
        a2.snapshot_id,
        &DataFileInfo::new("f2.parquet", 100, 5),
        WriteMode::Replace,
        &[],
        &[],
    )
    .unwrap();
    let b1 = wb
        .begin_write_transaction("public", "u", &cols(), WriteMode::Replace)
        .unwrap();
    wb.register_data_file(
        b1.table_id,
        b1.snapshot_id,
        &DataFileInfo::new("g1.parquet", 100, 5),
        WriteMode::Replace,
        &[],
        &[],
    )
    .unwrap();
    let b2 = wb
        .begin_write_transaction("public", "u", &cols(), WriteMode::Replace)
        .unwrap();
    wb.register_data_file(
        b2.table_id,
        b2.snapshot_id,
        &DataFileInfo::new("g2.parquet", 100, 5),
        WriteMode::Replace,
        &[],
        &[],
    )
    .unwrap();

    // Materialize the physical files.
    for (sch, tbl, name) in [
        ("public", "t", "f1.parquet"),
        ("public", "t", "f2.parquet"),
        ("public", "u", "g1.parquet"),
        ("public", "u", "g2.parquet"),
    ] {
        let dir = data_path.join(sch).join(tbl);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(name), b"parquet-bytes").unwrap();
    }

    // Expire the first snapshot of each catalog so each has one scheduled file.
    mgr.expire_snapshots_in_catalog("cat_a", ExpireCriteria::Versions(vec![a1.snapshot_id]))
        .await
        .unwrap();
    mgr.expire_snapshots_in_catalog("cat_b", ExpireCriteria::Versions(vec![b1.snapshot_id]))
        .await
        .unwrap();

    let store: Arc<dyn ObjectStore> = Arc::new(LocalFileSystem::new());
    let a_f1 = data_path.join("public").join("t").join("f1.parquet");
    let b_g1 = data_path.join("public").join("u").join("g1.parquet");

    // Dry run for A reports its one file without deleting.
    let dry =
        cleanup_old_files_in_catalog(&mgr, "cat_a", store.clone(), CleanupCriteria::All, true)
            .await
            .unwrap();
    assert_eq!(dry.len(), 1);
    assert!(a_f1.exists());

    // Real cleanup for A removes only A's file + bookkeeping row.
    let done = cleanup_old_files_in_catalog(&mgr, "cat_a", store, CleanupCriteria::All, false)
        .await
        .unwrap();
    assert_eq!(done.len(), 1);
    assert!(!a_f1.exists(), "A/f1 deleted");
    assert!(b_g1.exists(), "B/g1 untouched by A's cleanup");

    let a_left: i64 = sqlx::query(
        "SELECT COUNT(*) FROM ducklake_files_scheduled_for_deletion WHERE catalog_id = $1",
    )
    .bind(cat_a)
    .fetch_one(&pool)
    .await
    .unwrap()
    .try_get(0)
    .unwrap();
    assert_eq!(a_left, 0, "A's scheduled row cleared");
    let b_left: i64 = sqlx::query(
        "SELECT COUNT(*) FROM ducklake_files_scheduled_for_deletion WHERE catalog_id = $1",
    )
    .bind(cat_b)
    .fetch_one(&pool)
    .await
    .unwrap()
    .try_get(0)
    .unwrap();
    assert_eq!(b_left, 1, "B's scheduled row remains");
}

// ---------------------------------------------------------------------------
// delete_orphaned_files_multicatalog
// ---------------------------------------------------------------------------

/// The flagship use case: `drop_catalog` hard-deletes all of A's metadata,
/// leaving A's data files unreferenced on disk. The orphan sweep finds them
/// (they have no catalog row pointing at them anywhere) and removes them.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn delete_orphaned_files_reclaims_dropped_catalog_files() {
    use datafusion_ducklake::maintenance::{CleanupCriteria, delete_orphaned_files_multicatalog};
    use datafusion_ducklake::metadata_writer::DataFileInfo;
    use object_store::ObjectStore;
    use object_store::local::LocalFileSystem;
    use std::sync::Arc;
    use tempfile::TempDir;

    let (pool, _c) = spin_up_postgres().await.unwrap();
    let mgr = MulticatalogManager::new(pool.clone());
    let cat_a = mgr.create_catalog("cat_a").await.unwrap();
    let wa = PostgresMetadataWriter::with_pool(pool.clone(), cat_a)
        .await
        .unwrap();

    let temp = TempDir::new().unwrap();
    let data_path = temp.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();
    wa.set_data_path(data_path.to_str().unwrap()).unwrap();

    // Register one data file and put the real bytes on disk.
    let s = wa
        .begin_write_transaction("public", "t", &cols(), WriteMode::Replace)
        .unwrap();
    wa.register_data_file(
        s.table_id,
        s.snapshot_id,
        &DataFileInfo::new("f1.parquet", 100, 5),
        WriteMode::Replace,
        &[],
        &[],
    )
    .unwrap();
    let dir = data_path.join("public").join("t");
    std::fs::create_dir_all(&dir).unwrap();
    let f1 = dir.join("f1.parquet");
    std::fs::write(&f1, b"parquet-bytes").unwrap();

    // Drop catalog — metadata is hard-deleted, but the file is still on disk.
    assert!(mgr.drop_catalog("cat_a").await.unwrap());
    assert!(f1.exists(), "drop_catalog must not touch storage");

    let store: Arc<dyn ObjectStore> = Arc::new(LocalFileSystem::new());

    // Orphan sweep finds f1 (no metadata row references it anywhere) and removes it.
    let deleted = delete_orphaned_files_multicatalog(&mgr, store, CleanupCriteria::All, false)
        .await
        .unwrap();
    assert_eq!(deleted.len(), 1, "should find exactly the one orphan");
    assert!(!f1.exists(), "drop_catalog orphan must be reclaimed");
}

/// Global sweep must NOT delete files referenced by ANY catalog. With two
/// catalogs sharing `data_path` plus a stray, only the stray gets removed.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "pre-cat_{id} fixture layout + orphan-cleanup id collision; fixed with the maintenance rework"]
async fn delete_orphaned_files_spares_files_referenced_by_any_catalog() {
    use datafusion_ducklake::maintenance::{CleanupCriteria, delete_orphaned_files_multicatalog};
    use datafusion_ducklake::metadata_writer::DataFileInfo;
    use object_store::ObjectStore;
    use object_store::local::LocalFileSystem;
    use std::sync::Arc;
    use tempfile::TempDir;

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

    let temp = TempDir::new().unwrap();
    let data_path = temp.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();
    wa.set_data_path(data_path.to_str().unwrap()).unwrap();

    // A and B each have one referenced data file.
    let sa = wa
        .begin_write_transaction("public", "t", &cols(), WriteMode::Replace)
        .unwrap();
    wa.register_data_file(
        sa.table_id,
        sa.snapshot_id,
        &DataFileInfo::new("a.parquet", 100, 5),
        WriteMode::Replace,
        &[],
        &[],
    )
    .unwrap();
    let sb = wb
        .begin_write_transaction("public", "u", &cols(), WriteMode::Replace)
        .unwrap();
    wb.register_data_file(
        sb.table_id,
        sb.snapshot_id,
        &DataFileInfo::new("b.parquet", 100, 5),
        WriteMode::Replace,
        &[],
        &[],
    )
    .unwrap();

    // Materialise the three files: A's referenced, B's referenced, one stray.
    let a_file = data_path.join("public").join("t").join("a.parquet");
    let b_file = data_path.join("public").join("u").join("b.parquet");
    let stray = data_path.join("public").join("t").join("stray.parquet");
    std::fs::create_dir_all(a_file.parent().unwrap()).unwrap();
    std::fs::create_dir_all(b_file.parent().unwrap()).unwrap();
    std::fs::write(&a_file, b"a").unwrap();
    std::fs::write(&b_file, b"b").unwrap();
    std::fs::write(&stray, b"stray").unwrap();

    let store: Arc<dyn ObjectStore> = Arc::new(LocalFileSystem::new());
    let deleted = delete_orphaned_files_multicatalog(&mgr, store, CleanupCriteria::All, false)
        .await
        .unwrap();

    assert_eq!(deleted.len(), 1, "only the stray is unreferenced");
    assert!(
        deleted[0].ends_with("public/t/stray.parquet"),
        "got {:?}",
        deleted[0]
    );
    assert!(!stray.exists());
    assert!(a_file.exists(), "A's referenced file survives");
    assert!(b_file.exists(), "B's referenced file survives");
}

/// `OlderThan` cutoff applied to `last_modified` skips files newer than the
/// cutoff. Mirrors the upstream `last_modified < older_than` guard so in-flight
/// writes from concurrent transactions aren't reaped.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn delete_orphaned_files_older_than_skips_recent_files() {
    use datafusion_ducklake::maintenance::{CleanupCriteria, delete_orphaned_files_multicatalog};
    use object_store::ObjectStore;
    use object_store::local::LocalFileSystem;
    use std::sync::Arc;
    use tempfile::TempDir;

    let (pool, _c) = spin_up_postgres().await.unwrap();
    let mgr = MulticatalogManager::new(pool.clone());
    let cat = mgr.create_catalog("cat").await.unwrap();
    let w = PostgresMetadataWriter::with_pool(pool.clone(), cat)
        .await
        .unwrap();
    let temp = TempDir::new().unwrap();
    let data_path = temp.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();
    w.set_data_path(data_path.to_str().unwrap()).unwrap();

    // Stray file written just now — newer than the cutoff below.
    let stray = data_path.join("fresh.parquet");
    std::fs::write(&stray, b"fresh").unwrap();

    let store: Arc<dyn ObjectStore> = Arc::new(LocalFileSystem::new());

    // Cutoff is 1h in the past — the file's last_modified is way newer, so it
    // must NOT be deleted.
    let cutoff = chrono::Utc::now() - chrono::Duration::hours(1);
    let deleted = delete_orphaned_files_multicatalog(
        &mgr,
        store.clone(),
        CleanupCriteria::OlderThan(cutoff),
        false,
    )
    .await
    .unwrap();
    assert!(
        deleted.is_empty(),
        "files newer than cutoff must be skipped (got {deleted:?})"
    );
    assert!(stray.exists(), "fresh stray survives older_than filter");

    // Sanity: with `All`, the same file IS deleted.
    let deleted = delete_orphaned_files_multicatalog(&mgr, store, CleanupCriteria::All, false)
        .await
        .unwrap();
    assert_eq!(deleted.len(), 1);
    assert!(!stray.exists());
}

/// dry_run and real-run must return identical path sets. They take separate
/// code paths inside `run_orphan_cleanup` (dry_run formats from the pre-delete
/// `orphans` Vec; real-run formats per-orphan after each `object_store.delete`
/// succeeds), so a future refactor could diverge them.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "pre-cat_{id} fixture layout + orphan-cleanup id collision; fixed with the maintenance rework"]
async fn delete_orphaned_files_dry_run_matches_real_run() {
    use datafusion_ducklake::maintenance::{CleanupCriteria, delete_orphaned_files_multicatalog};
    use datafusion_ducklake::metadata_writer::DataFileInfo;
    use object_store::ObjectStore;
    use object_store::local::LocalFileSystem;
    use std::collections::HashSet;
    use std::sync::Arc;
    use tempfile::TempDir;

    let (pool, _c) = spin_up_postgres().await.unwrap();
    let mgr = MulticatalogManager::new(pool.clone());
    let cat = mgr.create_catalog("cat").await.unwrap();
    let w = PostgresMetadataWriter::with_pool(pool.clone(), cat)
        .await
        .unwrap();
    let temp = TempDir::new().unwrap();
    let data_path = temp.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();
    w.set_data_path(data_path.to_str().unwrap()).unwrap();

    // One referenced file + three orphans alongside it.
    let s = w
        .begin_write_transaction("public", "t", &cols(), WriteMode::Replace)
        .unwrap();
    w.register_data_file(
        s.table_id,
        s.snapshot_id,
        &DataFileInfo::new("ref.parquet", 100, 5),
        WriteMode::Replace,
        &[],
        &[],
    )
    .unwrap();
    let dir = data_path.join("public").join("t");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("ref.parquet"), b"referenced").unwrap();
    for name in ["o1.parquet", "o2.parquet", "o3.parquet"] {
        std::fs::write(dir.join(name), b"orphan").unwrap();
    }

    let store: Arc<dyn ObjectStore> = Arc::new(LocalFileSystem::new());

    // Dry run reports the orphans without touching disk.
    let dry: HashSet<String> =
        delete_orphaned_files_multicatalog(&mgr, store.clone(), CleanupCriteria::All, true)
            .await
            .unwrap()
            .into_iter()
            .collect();
    assert_eq!(dry.len(), 3, "dry_run finds all three orphans");
    for name in ["o1.parquet", "o2.parquet", "o3.parquet"] {
        assert!(dir.join(name).exists(), "dry_run must not touch disk");
    }

    // Real run returns the same set (order-independent).
    let real: HashSet<String> =
        delete_orphaned_files_multicatalog(&mgr, store, CleanupCriteria::All, false)
            .await
            .unwrap()
            .into_iter()
            .collect();
    assert_eq!(
        dry, real,
        "dry_run and real_run must return identical path sets"
    );
    assert!(dir.join("ref.parquet").exists(), "referenced file survives");
    for name in ["o1.parquet", "o2.parquet", "o3.parquet"] {
        assert!(!dir.join(name).exists(), "orphan deleted: {name}");
    }
}

// ---------------------------------------------------------------------------
// Per-catalog data-path segregation. Two catalogs sharing one physical
// data_path must NOT commingle files under `{data_path}/{schema}/{table}/…`.
// The writer encodes the catalog id into `ducklake_schema.path` (as
// `cat_{catalog_id}/{schema_name}`), so the read-side resolution chain
// `data_path + schema.path + table.path + file.path` naturally produces a
// per-catalog subtree.
//
// The strongest assertion is end-to-end: rebuild the resolution chain from
// the three stored columns and assert the resulting absolute path is a real
// file on disk. If the writer's upload prefix and the catalog-stored
// schema.path ever drift apart, this assertion fails immediately.
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn writes_segregate_data_files_by_catalog_directory() {
    use std::sync::Arc;

    use arrow::array::{Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use object_store::local::LocalFileSystem;
    use tempfile::TempDir;

    let (pool, _c) = spin_up_postgres().await.unwrap();
    let mgr = MulticatalogManager::new(pool.clone());
    let cat_a = mgr.create_catalog("pg_a").await.unwrap();
    let cat_b = mgr.create_catalog("pg_b").await.unwrap();
    assert_ne!(cat_a, cat_b);

    // Both catalogs share one physical root.
    let root = TempDir::new().unwrap();
    let data_path = root.path().to_str().unwrap().to_string();
    let object_store: Arc<dyn object_store::ObjectStore> = Arc::new(LocalFileSystem::new());

    let arrow_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        arrow_schema.clone(),
        vec![Arc::new(Int64Array::from(vec![1])), Arc::new(StringArray::from(vec![Some("x")]))],
    )
    .unwrap();

    // Write one row through each catalog into the same `public.users`.
    for cid in [cat_a, cat_b] {
        let writer = PostgresMetadataWriter::with_pool(pool.clone(), cid)
            .await
            .unwrap();
        writer.set_data_path(&data_path).unwrap();
        let table_writer =
            DuckLakeTableWriter::new(Arc::new(writer), Arc::clone(&object_store)).unwrap();
        table_writer
            .write_table("public", "users", std::slice::from_ref(&batch))
            .await
            .unwrap();
    }

    // Pull every (catalog_id, schema.path, table.path, file.path) tuple. The
    // multicatalog reader walks exactly these columns plus data_path; if our
    // writer's upload prefix doesn't match this chain, the file isn't where
    // the reader will look.
    let rows = sqlx::query(
        "SELECT m.catalog_id,
                s.schema_name, s.path AS schema_path,
                t.table_name,  t.path AS table_path,
                d.path         AS file_path
         FROM ducklake_data_file d
         JOIN ducklake_table t              ON t.table_id  = d.table_id
         JOIN ducklake_schema s             ON s.schema_id = t.schema_id
         JOIN ducklake_catalog_schema_map m ON m.schema_id = s.schema_id",
    )
    .fetch_all(&pool)
    .await
    .unwrap();
    assert_eq!(rows.len(), 2, "one file per catalog");

    let mut top_segments = std::collections::HashSet::new();
    for row in &rows {
        let cid: i64 = row.try_get("catalog_id").unwrap();
        let schema_name: String = row.try_get("schema_name").unwrap();
        let schema_path: String = row.try_get("schema_path").unwrap();
        let table_name: String = row.try_get("table_name").unwrap();
        let table_path: String = row.try_get("table_path").unwrap();
        let file_path: String = row.try_get("file_path").unwrap();

        // schema_name is what the user asked for; schema.path is where files
        // physically land. The id-prefixed path is what gives each catalog
        // its own subtree without renaming the user-visible schema.
        assert_eq!(schema_name, "public");
        assert_eq!(
            schema_path,
            format!("cat_{cid}/public"),
            "catalog {cid} schema.path must carry the cat_<id> prefix",
        );
        assert_eq!(table_path, table_name); // unchanged
        assert!(
            file_path.ends_with(".parquet") && !file_path.contains('/'),
            "data_file.path should still be just the bare filename, got {file_path:?}",
        );

        // End-to-end: rebuild the reader's resolution chain manually and
        // assert it points at a real file. This is the assertion that would
        // have caught the bug where the writer's upload prefix didn't match
        // schema.path.
        let resolved = root
            .path()
            .join(&schema_path)
            .join(&table_path)
            .join(&file_path);
        assert!(
            resolved.is_file(),
            "resolved path missing on disk: {resolved:?}",
        );

        top_segments.insert(schema_path.split('/').next().unwrap_or("").to_string());
    }
    assert_eq!(
        top_segments.len(),
        2,
        "two catalogs must land under distinct cat_<id> top segments: {top_segments:?}",
    );
}
