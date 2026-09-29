//! Expire, `drop_catalog` and the Postgres orphan purge reclaim catalog-inlined
//! storage: the physical `ducklake_inlined_data_<table_id>_<schema_version>`
//! tables, their `ducklake_inlined_data_tables` registry rows, and the
//! `ducklake_inlined_delete_<table_id>` table.
//!
//! Official DuckLake drops these in `DuckLakeMetadataManager::DeleteSnapshots`
//! once a table is fully expired, and not before: `DropTables` only tombstones
//! the table, so time travel to a snapshot before the drop keeps working until
//! that snapshot is expired. Each test checks both halves: the storage and the
//! old rows stay while a surviving snapshot can read them, and they are gone
//! once none can.

#![cfg(any(
    feature = "write-sqlite",
    feature = "write-duckdb",
    feature = "write-mysql",
    feature = "write-postgres"
))]

use std::sync::Arc;

use arrow::array::{Array, Int32Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::prelude::SessionContext;
use datafusion_ducklake::maintenance::ExpireCriteria;
use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeTableWriter, DuckLakeWriteOptions, MetadataProvider, MetadataWriter,
    WriteResult,
};
use object_store::local::LocalFileSystem;

fn batch(ids: &[i32]) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(ids.to_vec()))]).unwrap()
}

/// Append `ids` to `<schema>.<table>`, inlined into the catalog.
async fn write_inlined(
    writer: Arc<dyn MetadataWriter>,
    schema: &str,
    table: &str,
    ids: &[i32],
) -> WriteResult {
    let result = DuckLakeTableWriter::new(writer, Arc::new(LocalFileSystem::new()))
        .unwrap()
        .with_options(&DuckLakeWriteOptions::default().with_data_inlining_row_limit(100))
        .append_table(schema, table, &[batch(ids)])
        .await
        .unwrap();
    assert_eq!(result.files_written, 0, "the rows must be inlined");
    result
}

/// `id` values of `<schema>.<table>` as of `snapshot`, through the DataFusion read path.
async fn read_ids(
    provider: Arc<dyn MetadataProvider>,
    schema: &str,
    table: &str,
    snapshot: i64,
) -> Vec<i32> {
    let ctx = SessionContext::new();
    ctx.register_catalog(
        "lake",
        Arc::new(DuckLakeCatalog::with_snapshot(provider, snapshot).unwrap()),
    );
    let batches = ctx
        .sql(&format!("SELECT id FROM lake.{schema}.{table} ORDER BY id"))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    batches
        .iter()
        .flat_map(|b| {
            let ids = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
            (0..ids.len()).map(|i| ids.value(i)).collect::<Vec<_>>()
        })
        .collect()
}

fn delete_table(table_id: i64) -> String {
    format!("ducklake_inlined_delete_{table_id}")
}

// ---------------------------------------------------------------------------
// SQLite
// ---------------------------------------------------------------------------

#[cfg(feature = "write-sqlite")]
mod sqlite {
    use super::*;
    use datafusion_ducklake::{SqliteMetadataProvider, SqliteMetadataWriter};
    use sqlx::sqlite::SqlitePool;
    use tempfile::TempDir;

    async fn registry(pool: &SqlitePool, table_id: i64) -> Vec<String> {
        sqlx::query_scalar(
            "SELECT table_name FROM ducklake_inlined_data_tables WHERE table_id = ?
             ORDER BY table_name",
        )
        .bind(table_id)
        .fetch_all(pool)
        .await
        .unwrap()
    }

    async fn exists(pool: &SqlitePool, name: &str) -> bool {
        sqlx::query_scalar::<_, i64>(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = ?",
        )
        .bind(name)
        .fetch_one(pool)
        .await
        .unwrap()
            > 0
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn expire_reclaims_inlined_storage_of_a_fully_expired_table() {
        let temp = TempDir::new().unwrap();
        let conn = format!("sqlite:{}?mode=rwc", temp.path().join("m.db").display());
        let data = temp.path().join("data");
        std::fs::create_dir_all(&data).unwrap();
        let writer = Arc::new(SqliteMetadataWriter::new_with_init(&conn).await.unwrap());
        writer.set_data_path(data.to_str().unwrap()).unwrap();
        let pool = SqlitePool::connect(&conn).await.unwrap();

        // s1: t gets inlined rows; s2: an unrelated live table; s3: t is dropped.
        let t = write_inlined(writer.clone(), "main", "t", &[1, 2, 3]).await;
        let other = write_inlined(writer.clone(), "main", "other", &[9]).await;
        // An inlined-delete table as the official extension creates it.
        sqlx::query(sqlx::AssertSqlSafe(format!(
            "CREATE TABLE {} (file_id BIGINT, row_id BIGINT, begin_snapshot BIGINT)",
            delete_table(t.table_id)
        )))
        .execute(&pool)
        .await
        .unwrap();
        assert!(writer.drop_table("main", "t").unwrap());

        let physical = registry(&pool, t.table_id).await;
        assert_eq!(physical.len(), 1);
        let other_physical = registry(&pool, other.table_id).await;
        assert_eq!(other_physical.len(), 1);
        let provider = || async {
            Arc::new(SqliteMetadataProvider::new(&conn).await.unwrap()) as Arc<dyn MetadataProvider>
        };

        // Expiring s1 leaves s2, which still reads t: nothing of t may go.
        writer
            .expire_snapshots(ExpireCriteria::Versions(vec![t.snapshot_id]))
            .unwrap();
        assert_eq!(registry(&pool, t.table_id).await, physical);
        assert!(exists(&pool, &physical[0]).await);
        assert!(exists(&pool, &delete_table(t.table_id)).await);
        assert_eq!(
            read_ids(provider().await, "main", "t", other.snapshot_id).await,
            vec![1, 2, 3]
        );

        // Expiring s2 leaves no snapshot that can read t: all of its storage goes.
        writer
            .expire_snapshots(ExpireCriteria::Versions(vec![other.snapshot_id]))
            .unwrap();
        assert!(registry(&pool, t.table_id).await.is_empty());
        assert!(!exists(&pool, &physical[0]).await);
        assert!(!exists(&pool, &delete_table(t.table_id)).await);

        // The live table keeps its storage and its rows.
        assert_eq!(registry(&pool, other.table_id).await, other_physical);
        assert!(exists(&pool, &other_physical[0]).await);
        let provider = provider().await;
        let head = provider.get_current_snapshot().unwrap();
        assert_eq!(read_ids(provider, "main", "other", head).await, vec![9]);
    }
}

// ---------------------------------------------------------------------------
// DuckDB
// ---------------------------------------------------------------------------

#[cfg(feature = "write-duckdb")]
mod duckdb_backend {
    use super::*;
    use datafusion_ducklake::{DuckdbMetadataProvider, DuckdbMetadataWriter};
    use tempfile::TempDir;

    fn registry(conn: &duckdb::Connection, table_id: i64) -> Vec<String> {
        let mut statement = conn
            .prepare(
                "SELECT table_name FROM ducklake_inlined_data_tables WHERE table_id = ?
                 ORDER BY table_name",
            )
            .unwrap();
        statement
            .query_map(duckdb::params![table_id], |row| row.get(0))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap()
    }

    fn exists(conn: &duckdb::Connection, name: &str) -> bool {
        conn.query_row(
            "SELECT COUNT(*) > 0 FROM information_schema.tables WHERE table_name = ?",
            duckdb::params![name],
            |row| row.get(0),
        )
        .unwrap()
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn expire_reclaims_inlined_storage_of_a_fully_expired_table() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("m.duckdb").to_string_lossy().into_owned();
        let data = temp.path().join("data");
        std::fs::create_dir_all(&data).unwrap();

        let (t, other, drop_snapshot) = {
            let writer = Arc::new(DuckdbMetadataWriter::new_with_init(path.clone()).unwrap());
            writer.set_data_path(data.to_str().unwrap()).unwrap();
            let t = write_inlined(writer.clone(), "main", "t", &[1, 2, 3]).await;
            let other = write_inlined(writer.clone(), "main", "other", &[9]).await;
            let last = write_inlined(writer.clone(), "main", "other", &[10]).await;
            (t, other, last.snapshot_id)
        };

        // This writer has no DROP TABLE; drop t the way official `DropTables`
        // does (tombstone at a later snapshot), and add the inlined-delete table
        // the official extension creates.
        {
            let conn = duckdb::Connection::open(&path).unwrap();
            conn.execute_batch(&format!(
                "UPDATE ducklake_table SET end_snapshot = {drop_snapshot}
                     WHERE table_id = {id} AND end_snapshot IS NULL;
                 UPDATE ducklake_column SET end_snapshot = {drop_snapshot}
                     WHERE table_id = {id} AND end_snapshot IS NULL;
                 CREATE TABLE {delete} (file_id BIGINT, row_id BIGINT, begin_snapshot BIGINT);",
                id = t.table_id,
                delete = delete_table(t.table_id),
            ))
            .unwrap();
        }

        let (physical, other_physical) = {
            let conn = duckdb::Connection::open(&path).unwrap();
            (registry(&conn, t.table_id), registry(&conn, other.table_id))
        };
        assert_eq!(physical.len(), 1);
        assert_eq!(other_physical.len(), 1);
        let provider = || {
            Arc::new(DuckdbMetadataProvider::new(path.clone()).unwrap())
                as Arc<dyn MetadataProvider>
        };

        // Expiring s1 leaves s2, which still reads t: nothing of t may go.
        DuckdbMetadataWriter::new(path.clone())
            .unwrap()
            .expire_snapshots(ExpireCriteria::Versions(vec![t.snapshot_id]))
            .unwrap();
        {
            let conn = duckdb::Connection::open(&path).unwrap();
            assert_eq!(registry(&conn, t.table_id), physical);
            assert!(exists(&conn, &physical[0]));
            assert!(exists(&conn, &delete_table(t.table_id)));
        }
        assert_eq!(
            read_ids(provider(), "main", "t", other.snapshot_id).await,
            vec![1, 2, 3]
        );

        // Expiring s2 leaves no snapshot that can read t: all of its storage goes.
        DuckdbMetadataWriter::new(path.clone())
            .unwrap()
            .expire_snapshots(ExpireCriteria::Versions(vec![other.snapshot_id]))
            .unwrap();
        {
            let conn = duckdb::Connection::open(&path).unwrap();
            assert!(registry(&conn, t.table_id).is_empty());
            assert!(!exists(&conn, &physical[0]));
            assert!(!exists(&conn, &delete_table(t.table_id)));
            assert_eq!(registry(&conn, other.table_id), other_physical);
            assert!(exists(&conn, &other_physical[0]));
        }
        let provider = provider();
        let head = provider.get_current_snapshot().unwrap();
        assert_eq!(read_ids(provider, "main", "other", head).await, vec![9, 10]);
    }
}

// ---------------------------------------------------------------------------
// MySQL
// ---------------------------------------------------------------------------

#[cfg(feature = "write-mysql")]
mod mysql {
    use super::*;
    use datafusion_ducklake::{MySqlMetadataProvider, MySqlMetadataWriter};
    use sqlx::MySqlPool;
    use tempfile::TempDir;
    use testcontainers::runners::AsyncRunner;
    use testcontainers_modules::mysql::Mysql;

    async fn registry(pool: &MySqlPool, table_id: i64) -> Vec<String> {
        sqlx::query_scalar(
            "SELECT table_name FROM ducklake_inlined_data_tables WHERE table_id = ?
             ORDER BY table_name",
        )
        .bind(table_id)
        .fetch_all(pool)
        .await
        .unwrap()
    }

    async fn exists(pool: &MySqlPool, name: &str) -> bool {
        sqlx::query_scalar::<_, i64>(
            "SELECT COUNT(*) FROM information_schema.tables
             WHERE table_schema = DATABASE() AND table_name = ?",
        )
        .bind(name)
        .fetch_one(pool)
        .await
        .unwrap()
            > 0
    }

    #[tokio::test(flavor = "multi_thread")]
    #[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
    async fn expire_reclaims_inlined_storage_of_a_fully_expired_table() {
        let container = Mysql::default().start().await.unwrap();
        let port = container.get_host_port_ipv4(3306).await.unwrap();
        let conn = format!("mysql://root@127.0.0.1:{port}/test");
        let temp = TempDir::new().unwrap();
        let data = temp.path().join("data");
        std::fs::create_dir_all(&data).unwrap();
        let writer = Arc::new(MySqlMetadataWriter::new_with_init(&conn).await.unwrap());
        writer.set_data_path(data.to_str().unwrap()).unwrap();
        let pool = MySqlPool::connect(&conn).await.unwrap();

        let t = write_inlined(writer.clone(), "main", "t", &[1, 2, 3]).await;
        let other = write_inlined(writer.clone(), "main", "other", &[9]).await;
        let last = write_inlined(writer.clone(), "main", "other", &[10]).await;

        // This writer has no DROP TABLE; drop t the way official `DropTables`
        // does (tombstone at a later snapshot), and add the inlined-delete table
        // the official extension creates.
        for table in ["ducklake_table", "ducklake_column"] {
            sqlx::query(sqlx::AssertSqlSafe(format!(
                "UPDATE {table} SET end_snapshot = ? WHERE table_id = ? AND end_snapshot IS NULL"
            )))
            .bind(last.snapshot_id)
            .bind(t.table_id)
            .execute(&pool)
            .await
            .unwrap();
        }
        sqlx::query(sqlx::AssertSqlSafe(format!(
            "CREATE TABLE {} (file_id BIGINT, row_id BIGINT, begin_snapshot BIGINT)",
            delete_table(t.table_id)
        )))
        .execute(&pool)
        .await
        .unwrap();

        let physical = registry(&pool, t.table_id).await;
        assert_eq!(physical.len(), 1);
        let other_physical = registry(&pool, other.table_id).await;
        assert_eq!(other_physical.len(), 1);
        let provider = || async {
            Arc::new(MySqlMetadataProvider::new(&conn).await.unwrap()) as Arc<dyn MetadataProvider>
        };

        // Expiring s1 leaves s2, which still reads t: nothing of t may go.
        writer
            .expire_snapshots(ExpireCriteria::Versions(vec![t.snapshot_id]))
            .unwrap();
        assert_eq!(registry(&pool, t.table_id).await, physical);
        assert!(exists(&pool, &physical[0]).await);
        assert!(exists(&pool, &delete_table(t.table_id)).await);
        assert_eq!(
            read_ids(provider().await, "main", "t", other.snapshot_id).await,
            vec![1, 2, 3]
        );

        // Expiring s2 leaves no snapshot that can read t: all of its storage goes.
        writer
            .expire_snapshots(ExpireCriteria::Versions(vec![other.snapshot_id]))
            .unwrap();
        assert!(registry(&pool, t.table_id).await.is_empty());
        assert!(!exists(&pool, &physical[0]).await);
        assert!(!exists(&pool, &delete_table(t.table_id)).await);
        assert_eq!(registry(&pool, other.table_id).await, other_physical);
        assert!(exists(&pool, &other_physical[0]).await);
        let provider = provider().await;
        let head = provider.get_current_snapshot().unwrap();
        assert_eq!(read_ids(provider, "main", "other", head).await, vec![9, 10]);
    }

    /// MySQL DDL ends the open transaction, so expire drops the physical tables
    /// after it commits. If that step stops part way, the registry rows are
    /// still there, and the next expire finishes the job.
    #[tokio::test(flavor = "multi_thread")]
    #[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
    async fn expire_finishes_an_interrupted_drop_of_inlined_storage() {
        let container = Mysql::default().start().await.unwrap();
        let port = container.get_host_port_ipv4(3306).await.unwrap();
        let conn = format!("mysql://root@127.0.0.1:{port}/test");
        let temp = TempDir::new().unwrap();
        let data = temp.path().join("data");
        std::fs::create_dir_all(&data).unwrap();
        let writer = Arc::new(MySqlMetadataWriter::new_with_init(&conn).await.unwrap());
        writer.set_data_path(data.to_str().unwrap()).unwrap();
        let pool = MySqlPool::connect(&conn).await.unwrap();

        let t = write_inlined(writer.clone(), "main", "t", &[1]).await;
        let physical = registry(&pool, t.table_id).await;
        assert_eq!(physical.len(), 1);
        // The state an interrupted expire leaves: the table rows are gone, the
        // registry row and the physical table are not.
        for table in ["ducklake_table", "ducklake_column", "ducklake_table_stats"] {
            sqlx::query(sqlx::AssertSqlSafe(format!(
                "DELETE FROM {table} WHERE table_id = ?"
            )))
            .bind(t.table_id)
            .execute(&pool)
            .await
            .unwrap();
        }
        let next = write_inlined(writer.clone(), "main", "other", &[2]).await;
        write_inlined(writer.clone(), "main", "other", &[3]).await;

        writer
            .expire_snapshots(ExpireCriteria::Versions(vec![next.snapshot_id]))
            .unwrap();
        assert!(registry(&pool, t.table_id).await.is_empty());
        assert!(!exists(&pool, &physical[0]).await);
    }
}

// ---------------------------------------------------------------------------
// PostgreSQL (multicatalog)
// ---------------------------------------------------------------------------

#[cfg(feature = "write-postgres")]
mod postgres {
    use super::*;
    use datafusion_ducklake::{
        MulticatalogManager, MulticatalogProvider, PostgresMetadataWriter,
        initialize_multicatalog_schema,
    };
    use sqlx::postgres::{PgPool, PgPoolOptions};
    use tempfile::TempDir;
    use testcontainers::ContainerAsync;
    use testcontainers::runners::AsyncRunner;
    use testcontainers_modules::postgres::Postgres;

    async fn spin_up() -> (PgPool, ContainerAsync<Postgres>) {
        let container = Postgres::default().start().await.unwrap();
        let port = container.get_host_port_ipv4(5432).await.unwrap();
        let pool = PgPoolOptions::new()
            .max_connections(5)
            .connect(&format!(
                "postgresql://postgres:postgres@127.0.0.1:{port}/postgres"
            ))
            .await
            .unwrap();
        initialize_multicatalog_schema(&pool).await.unwrap();
        (pool, container)
    }

    async fn writer(pool: &PgPool, catalog_id: i64, temp: &TempDir) -> Arc<dyn MetadataWriter> {
        let writer = PostgresMetadataWriter::with_pool(pool.clone(), catalog_id)
            .await
            .unwrap();
        writer.set_data_path(temp.path().to_str().unwrap()).unwrap();
        Arc::new(writer)
    }

    async fn provider(pool: &PgPool, catalog_id: i64) -> Arc<dyn MetadataProvider> {
        Arc::new(
            MulticatalogProvider::with_pool_and_id(pool.clone(), catalog_id)
                .await
                .unwrap(),
        )
    }

    async fn registry(pool: &PgPool, table_id: i64) -> Vec<String> {
        sqlx::query_scalar(
            "SELECT table_name FROM ducklake_inlined_data_tables WHERE table_id = $1
             ORDER BY table_name",
        )
        .bind(table_id)
        .fetch_all(pool)
        .await
        .unwrap()
    }

    async fn exists(pool: &PgPool, name: &str) -> bool {
        sqlx::query_scalar("SELECT to_regclass($1) IS NOT NULL")
            .bind(format!("\"{name}\""))
            .fetch_one(pool)
            .await
            .unwrap()
    }

    async fn create_delete_table(pool: &PgPool, table_id: i64) {
        sqlx::query(sqlx::AssertSqlSafe(format!(
            "CREATE TABLE {} (file_id BIGINT, row_id BIGINT, begin_snapshot BIGINT)",
            delete_table(table_id)
        )))
        .execute(pool)
        .await
        .unwrap();
    }

    #[tokio::test(flavor = "multi_thread")]
    #[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
    async fn expire_reclaims_inlined_storage_of_a_fully_expired_table() {
        let (pool, _container) = spin_up().await;
        let manager = MulticatalogManager::new(pool.clone());
        let catalog_id = manager.create_catalog("lake").await.unwrap();
        let temp = TempDir::new().unwrap();
        let w = writer(&pool, catalog_id, &temp).await;

        // s1: t gets inlined rows; s2: an unrelated live table; s3: t is dropped.
        let t = write_inlined(w.clone(), "main", "t", &[1, 2, 3]).await;
        let other = write_inlined(w.clone(), "main", "other", &[9]).await;
        create_delete_table(&pool, t.table_id).await;
        assert!(
            manager
                .drop_table_in_catalog("lake", "main", "t")
                .await
                .unwrap()
        );

        // Dropping the table only tombstones it: its storage stays.
        let physical = registry(&pool, t.table_id).await;
        assert_eq!(physical.len(), 1);
        assert!(exists(&pool, &physical[0]).await);
        let other_physical = registry(&pool, other.table_id).await;
        assert_eq!(other_physical.len(), 1);

        // Expiring s1 leaves s2, which still reads t: nothing of t may go.
        manager
            .expire_snapshots_in_catalog("lake", ExpireCriteria::Versions(vec![t.snapshot_id]))
            .await
            .unwrap();
        assert_eq!(registry(&pool, t.table_id).await, physical);
        assert!(exists(&pool, &physical[0]).await);
        assert!(exists(&pool, &delete_table(t.table_id)).await);
        assert_eq!(
            read_ids(
                provider(&pool, catalog_id).await,
                "main",
                "t",
                other.snapshot_id
            )
            .await,
            vec![1, 2, 3]
        );

        // Expiring s2 leaves no snapshot that can read t: all of its storage goes.
        manager
            .expire_snapshots_in_catalog("lake", ExpireCriteria::Versions(vec![other.snapshot_id]))
            .await
            .unwrap();
        assert!(registry(&pool, t.table_id).await.is_empty());
        assert!(!exists(&pool, &physical[0]).await);
        assert!(!exists(&pool, &delete_table(t.table_id)).await);

        // The live table keeps its storage and its rows.
        assert_eq!(registry(&pool, other.table_id).await, other_physical);
        assert!(exists(&pool, &other_physical[0]).await);
        let p = provider(&pool, catalog_id).await;
        let head = p.get_current_snapshot().unwrap();
        assert_eq!(read_ids(p, "main", "other", head).await, vec![9]);
    }

    #[tokio::test(flavor = "multi_thread")]
    #[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
    async fn drop_catalog_reclaims_inlined_storage_and_spares_other_catalogs() {
        let (pool, _container) = spin_up().await;
        let manager = MulticatalogManager::new(pool.clone());
        let doomed = manager.create_catalog("doomed").await.unwrap();
        let kept = manager.create_catalog("kept").await.unwrap();
        let temp = TempDir::new().unwrap();

        let gone = write_inlined(writer(&pool, doomed, &temp).await, "main", "t", &[1]).await;
        create_delete_table(&pool, gone.table_id).await;
        let stays = write_inlined(writer(&pool, kept, &temp).await, "main", "t", &[2]).await;
        create_delete_table(&pool, stays.table_id).await;
        let gone_physical = registry(&pool, gone.table_id).await;
        let stays_physical = registry(&pool, stays.table_id).await;
        assert_eq!((gone_physical.len(), stays_physical.len()), (1, 1));

        assert!(manager.drop_catalog("doomed").await.unwrap());

        assert!(registry(&pool, gone.table_id).await.is_empty());
        assert!(!exists(&pool, &gone_physical[0]).await);
        assert!(!exists(&pool, &delete_table(gone.table_id)).await);
        assert_eq!(registry(&pool, stays.table_id).await, stays_physical);
        assert!(exists(&pool, &stays_physical[0]).await);
        assert!(exists(&pool, &delete_table(stays.table_id)).await);
        let p = provider(&pool, kept).await;
        let head = p.get_current_snapshot().unwrap();
        assert_eq!(read_ids(p, "main", "t", head).await, vec![2]);
    }

    /// Catalogs written before this fix carry inlined tables whose table rows
    /// expire or `drop_catalog` already deleted. The orphan purge drops them and
    /// leaves the storage of live tables alone.
    #[tokio::test(flavor = "multi_thread")]
    #[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
    async fn purge_drops_inlined_storage_left_by_an_older_version() {
        let (pool, _container) = spin_up().await;
        let manager = MulticatalogManager::new(pool.clone());
        let catalog_id = manager.create_catalog("lake").await.unwrap();
        let temp = TempDir::new().unwrap();
        let w = writer(&pool, catalog_id, &temp).await;
        let live = write_inlined(w.clone(), "main", "live", &[1]).await;
        create_delete_table(&pool, live.table_id).await;
        let live_physical = registry(&pool, live.table_id).await;

        // What an older expire left behind for table 987654: a registry row, its
        // physical table, and its inlined-delete table, but no ducklake_table row.
        let orphan_id = 987_654;
        let orphan_physical = format!("ducklake_inlined_data_{orphan_id}_1");
        sqlx::query(sqlx::AssertSqlSafe(format!(
            "CREATE TABLE {orphan_physical} (row_id BIGINT, begin_snapshot BIGINT,
                 end_snapshot BIGINT, id INTEGER)"
        )))
        .execute(&pool)
        .await
        .unwrap();
        sqlx::query(
            "INSERT INTO ducklake_inlined_data_tables (table_id, table_name, schema_version)
             VALUES ($1, $2, 1)",
        )
        .bind(orphan_id)
        .bind(&orphan_physical)
        .execute(&pool)
        .await
        .unwrap();
        create_delete_table(&pool, orphan_id).await;

        datafusion_ducklake::maintenance::purge_orphaned_metadata_postgres(&pool)
            .await
            .unwrap();

        assert!(registry(&pool, orphan_id).await.is_empty());
        assert!(!exists(&pool, &orphan_physical).await);
        assert!(!exists(&pool, &delete_table(orphan_id)).await);
        assert_eq!(registry(&pool, live.table_id).await, live_physical);
        assert!(exists(&pool, &live_physical[0]).await);
        assert!(exists(&pool, &delete_table(live.table_id)).await);
        let p = provider(&pool, catalog_id).await;
        let head = p.get_current_snapshot().unwrap();
        assert_eq!(read_ids(p, "main", "live", head).await, vec![1]);
    }
}
