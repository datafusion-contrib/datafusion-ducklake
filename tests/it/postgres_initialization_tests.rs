#![cfg(feature = "write-postgres")]
//! Initialization of both PostgreSQL layouts: a current store costs one
//! statement and waits behind no writer, a store missing what initialization
//! guarantees gets it back, and initializations racing on a new store leave
//! every relation in place.

use datafusion_ducklake::metadata_writer::MetadataWriter;
use datafusion_ducklake::{PostgresSingleCatalogMetadataWriter, initialize_multicatalog_schema};
use sqlx::postgres::{PgPool, PgPoolOptions};
use sqlx::{AssertSqlSafe, Executor, Postgres, Transaction};
use testcontainers::runners::AsyncRunner;
use testcontainers::{ContainerAsync, ImageExt};
use testcontainers_modules::postgres::Postgres as PostgresImage;

async fn start() -> (ContainerAsync<PostgresImage>, String) {
    let container = PostgresImage::default().start().await.unwrap();
    let port = container.get_host_port_ipv4(5432).await.unwrap();
    let url = format!("postgresql://postgres:postgres@127.0.0.1:{port}/postgres");
    (container, url)
}

/// A container whose `pg_stat_statements` counts every statement sent to it.
async fn start_counting() -> (ContainerAsync<PostgresImage>, String) {
    let container = PostgresImage::default()
        .with_cmd(["-c", "fsync=off", "-c", "shared_preload_libraries=pg_stat_statements"])
        .start()
        .await
        .unwrap();
    let port = container.get_host_port_ipv4(5432).await.unwrap();
    let url = format!("postgresql://postgres:postgres@127.0.0.1:{port}/postgres");
    let pool = PgPool::connect(&url).await.unwrap();
    sqlx::query("CREATE EXTENSION pg_stat_statements")
        .execute(&pool)
        .await
        .unwrap();
    (container, url)
}

/// Run `operation` and count the statements sent meanwhile, other than this
/// count's own.
async fn statements_sent<F: std::future::Future<Output = ()>>(pool: &PgPool, operation: F) -> i64 {
    sqlx::query("SELECT pg_stat_statements_reset()")
        .execute(pool)
        .await
        .unwrap();
    operation.await;
    sqlx::query_scalar(
        "SELECT COALESCE(SUM(calls), 0)::BIGINT FROM pg_stat_statements
         WHERE query NOT LIKE '%pg_stat_statements%'",
    )
    .fetch_one(pool)
    .await
    .unwrap()
}

async fn pool(url: &str) -> PgPool {
    PgPoolOptions::new()
        .max_connections(20)
        .connect(url)
        .await
        .unwrap()
}

/// A pool whose statements fail after waiting a second for a lock.
async fn impatient_pool(url: &str) -> PgPool {
    PgPoolOptions::new()
        .max_connections(5)
        .after_connect(|connection, _| {
            Box::pin(async move {
                connection.execute("SET lock_timeout = '1s'").await?;
                Ok(())
            })
        })
        .connect(url)
        .await
        .unwrap()
}

/// Every catalog table, and every index that backs no constraint.
async fn relations(pool: &PgPool) -> Vec<(String, bool)> {
    sqlx::query_as(
        "SELECT c.relname::text, c.relkind = 'i'
         FROM pg_class c
         LEFT JOIN pg_index i ON i.indexrelid = c.oid
         LEFT JOIN pg_class t ON t.oid = coalesce(i.indrelid, c.oid)
         WHERE c.relnamespace = 'public'::regnamespace
           AND t.relname LIKE 'ducklake\\_%'
           AND (c.relkind = 'r'
                OR (c.relkind = 'i'
                    AND NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conindid = c.oid)))
         ORDER BY 1",
    )
    .fetch_all(pool)
    .await
    .unwrap()
}

/// Hold the lock a writer holds on every catalog table but `except`, which
/// conflicts with every lock DDL takes, until the transaction ends.
async fn hold_writer_locks_except(pool: &PgPool, except: &str) -> Transaction<'static, Postgres> {
    let tables = relations(pool)
        .await
        .into_iter()
        .filter(|(name, index)| !index && name != except)
        .map(|(name, _)| name)
        .collect::<Vec<_>>();
    let mut tx = pool.begin().await.unwrap();
    sqlx::query(AssertSqlSafe(format!(
        "LOCK TABLE {} IN ROW EXCLUSIVE MODE",
        tables.join(", ")
    )))
    .execute(&mut *tx)
    .await
    .unwrap();
    tx
}

/// How many of `columns` exist, as `(table, column)`.
async fn columns_present(pool: &PgPool, columns: &[(&str, &str)]) -> i64 {
    let (tables, names): (Vec<_>, Vec<_>) = columns
        .iter()
        .map(|(table, column)| (table.to_string(), column.to_string()))
        .unzip();
    sqlx::query_scalar(
        "SELECT count(*) FROM unnest($1::text[], $2::text[]) AS c(tbl, col)
         JOIN pg_attribute a ON a.attrelid = to_regclass(c.tbl) AND a.attname = c.col
         WHERE NOT a.attisdropped",
    )
    .bind(tables)
    .bind(names)
    .fetch_one(pool)
    .await
    .unwrap()
}

async fn reset(pool: &PgPool) {
    sqlx::raw_sql("DROP SCHEMA public CASCADE; CREATE SCHEMA public")
        .execute(pool)
        .await
        .unwrap();
}

async fn initialize_single(pool: &PgPool) -> datafusion_ducklake::Result<()> {
    PostgresSingleCatalogMetadataWriter::from_pool(pool.clone())
        .await?
        .initialize_schema()
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn initializing_a_current_multicatalog_store_waits_behind_no_writer() {
    let (_container, url) = start().await;
    let pool = pool(&url).await;
    initialize_multicatalog_schema(&pool).await.unwrap();
    let held = hold_writer_locks_except(&pool, "").await;

    let result = initialize_multicatalog_schema(&impatient_pool(&url).await).await;
    held.rollback().await.unwrap();

    result.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn initializing_a_current_single_catalog_waits_behind_no_writer() {
    let (_container, url) = start().await;
    let pool = pool(&url).await;
    initialize_single(&pool).await.unwrap();
    let held = hold_writer_locks_except(&pool, "").await;

    let result = initialize_single(&impatient_pool(&url).await).await;
    held.rollback().await.unwrap();

    result.unwrap();
}

/// Initializing a store that lacks one index locks only that index's table,
/// so it proceeds while writers hold every other table.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn upgrading_a_multicatalog_store_locks_only_what_it_changes() {
    let (_container, url) = start().await;
    let pool = pool(&url).await;
    initialize_multicatalog_schema(&pool).await.unwrap();
    sqlx::query("DROP INDEX idx_catalog_schema_map_schema")
        .execute(&pool)
        .await
        .unwrap();
    let held = hold_writer_locks_except(&pool, "ducklake_catalog_schema_map").await;

    let result = initialize_multicatalog_schema(&impatient_pool(&url).await).await;
    held.rollback().await.unwrap();
    let restored: bool =
        sqlx::query_scalar("SELECT to_regclass('idx_catalog_schema_map_schema') IS NOT NULL")
            .fetch_one(&pool)
            .await
            .unwrap();

    result.unwrap();
    assert!(restored);
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn initializing_a_current_multicatalog_store_sends_one_statement() {
    let (_container, url) = start_counting().await;
    let counter = pool(&url).await;
    let initializer = pool(&url).await;
    initialize_multicatalog_schema(&initializer).await.unwrap();

    let sent = statements_sent(&counter, async {
        initialize_multicatalog_schema(&initializer).await.unwrap();
    })
    .await;

    assert_eq!(sent, 1);
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn initializing_a_current_single_catalog_sends_one_statement() {
    let (_container, url) = start_counting().await;
    let counter = pool(&url).await;
    let initializer = pool(&url).await;
    initialize_single(&initializer).await.unwrap();

    let opened = statements_sent(&counter, async {
        PostgresSingleCatalogMetadataWriter::from_pool(initializer.clone())
            .await
            .unwrap();
    })
    .await;
    let writer = PostgresSingleCatalogMetadataWriter::from_pool(initializer.clone())
        .await
        .unwrap();
    let initialized = statements_sent(&counter, async {
        writer.initialize_schema().unwrap();
    })
    .await;

    assert_eq!((opened, initialized), (1, 1));
}

/// Each table and index a new multicatalog store has comes back when it is
/// dropped, so the check that skips initialization covers every one.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn multicatalog_initialization_restores_every_relation() {
    let (_container, url) = start().await;
    let pool = pool(&url).await;
    initialize_multicatalog_schema(&pool).await.unwrap();
    let created = relations(&pool).await;

    let mut missing = Vec::new();
    for (name, index) in &created {
        let kind = if *index {
            "INDEX"
        } else {
            "TABLE"
        };
        sqlx::query(AssertSqlSafe(format!("DROP {kind} {name} CASCADE")))
            .execute(&pool)
            .await
            .unwrap();
        initialize_multicatalog_schema(&pool).await.unwrap();
        if !relations(&pool).await.contains(&(name.clone(), *index)) {
            missing.push(name.clone());
        }
    }

    assert!(created.len() > 25, "{created:?}");
    assert_eq!(missing, Vec::<String>::new());
    assert_eq!(relations(&pool).await, created);
}

/// Each table and index a new single catalog has comes back when it is
/// dropped, so the check that skips initialization covers every one.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn single_catalog_initialization_restores_every_relation() {
    let (_container, url) = start().await;
    let pool = pool(&url).await;
    initialize_single(&pool).await.unwrap();
    let created = relations(&pool).await;

    let mut missing = Vec::new();
    for (name, index) in &created {
        let kind = if *index {
            "INDEX"
        } else {
            "TABLE"
        };
        sqlx::query(AssertSqlSafe(format!("DROP {kind} {name} CASCADE")))
            .execute(&pool)
            .await
            .unwrap();
        initialize_single(&pool).await.unwrap();
        if !relations(&pool).await.contains(&(name.clone(), *index)) {
            missing.push(name.clone());
        }
    }

    assert!(created.len() > 25, "{created:?}");
    assert_eq!(missing, Vec::<String>::new());
    assert_eq!(relations(&pool).await, created);
}

/// A store an earlier release left with a single-column `ducklake_column` key,
/// a `NOT NULL` `changes_made`, an absolute-path index and invalid indexes,
/// as an interrupted `CREATE INDEX CONCURRENTLY` leaves, is brought to the
/// current shape.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn multicatalog_initialization_restores_the_current_shape() {
    let (_container, url) = start().await;
    let pool = pool(&url).await;
    initialize_multicatalog_schema(&pool).await.unwrap();
    sqlx::raw_sql(
        "ALTER TABLE ducklake_column DROP CONSTRAINT ducklake_column_pkey;
         ALTER TABLE ducklake_column ADD PRIMARY KEY (column_id);
         ALTER TABLE ducklake_snapshot_changes ALTER COLUMN changes_made SET NOT NULL;
         CREATE INDEX idx_data_file_absolute_path ON ducklake_data_file (path);
         UPDATE pg_index SET indisvalid = false
         WHERE indexrelid IN ('idx_file_column_stats_table_file_column'::regclass,
                              'idx_ducklake_column_live'::regclass);",
    )
    .execute(&pool)
    .await
    .unwrap();

    initialize_multicatalog_schema(&pool).await.unwrap();
    let (key_columns, changes_not_null, absolute, valid): (i32, bool, bool, bool) = sqlx::query_as(
        "SELECT (SELECT array_length(conkey, 1) FROM pg_constraint
                     WHERE conrelid = 'ducklake_column'::regclass AND contype = 'p'),
                    (SELECT attnotnull FROM pg_attribute
                     WHERE attrelid = 'ducklake_snapshot_changes'::regclass
                       AND attname = 'changes_made'),
                    to_regclass('idx_data_file_absolute_path') IS NOT NULL,
                    (SELECT bool_and(indisvalid) FROM pg_index
                     WHERE indexrelid IN ('idx_file_column_stats_table_file_column'::regclass,
                                          'idx_ducklake_column_live'::regclass))",
    )
    .fetch_one(&pool)
    .await
    .unwrap();

    assert_eq!(key_columns, 3);
    assert!(!changes_not_null);
    assert!(!absolute);
    assert!(valid);
}

/// A single catalog missing a counter, with an unslashed `data_path`, a case
/// variant of `true`, a `NOT NULL` `changes_made` and an invalid index is
/// migrated by `initialize_schema`.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn single_catalog_initialization_migrates_earlier_data() {
    let (_container, url) = start().await;
    let pool = pool(&url).await;
    initialize_single(&pool).await.unwrap();
    sqlx::raw_sql(
        "DELETE FROM ducklake_metadata WHERE key = 'next_snapshot_id';
         INSERT INTO ducklake_metadata (key, value) VALUES
             ('data_path', '/lake'), ('sort_on_insert', 'TRUE');
         ALTER TABLE ducklake_snapshot_changes ALTER COLUMN changes_made SET NOT NULL;
         UPDATE pg_index SET indisvalid = false
         WHERE indexrelid = 'idx_file_partition_value_table_key'::regclass;",
    )
    .execute(&pool)
    .await
    .unwrap();

    initialize_single(&pool).await.unwrap();
    let (counter, data_path, sort_on_insert, changes_not_null, valid): (
        bool,
        String,
        String,
        bool,
        bool,
    ) = sqlx::query_as(
        "SELECT EXISTS (SELECT 1 FROM ducklake_metadata WHERE key = 'next_snapshot_id'),
                (SELECT value FROM ducklake_metadata WHERE key = 'data_path'),
                (SELECT value FROM ducklake_metadata WHERE key = 'sort_on_insert'),
                (SELECT attnotnull FROM pg_attribute
                 WHERE attrelid = 'ducklake_snapshot_changes'::regclass
                   AND attname = 'changes_made'),
                (SELECT indisvalid FROM pg_index
                 WHERE indexrelid = 'idx_file_partition_value_table_key'::regclass)",
    )
    .fetch_one(&pool)
    .await
    .unwrap();

    assert!(counter);
    assert_eq!(data_path, "/lake/");
    assert_eq!(sort_on_insert, "true");
    assert!(!changes_not_null);
    assert!(valid);
}

/// Columns an earlier release lacked, dropped from a current store, come back.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn multicatalog_initialization_restores_added_columns() {
    let (_container, url) = start().await;
    let pool = pool(&url).await;
    initialize_multicatalog_schema(&pool).await.unwrap();
    let added = [
        ("ducklake_metadata", "scope_id"),
        ("ducklake_column", "default_value_dialect"),
        ("ducklake_data_file", "partition_id"),
        (
            "ducklake_files_scheduled_for_deletion",
            "scheduled_by_owner",
        ),
    ];
    for (table, column) in added {
        sqlx::query(AssertSqlSafe(format!(
            "ALTER TABLE {table} DROP COLUMN {column}"
        )))
        .execute(&pool)
        .await
        .unwrap();
    }

    initialize_multicatalog_schema(&pool).await.unwrap();

    assert_eq!(columns_present(&pool, &added).await, added.len() as i64);
}

/// Columns an earlier release lacked, dropped from a current catalog, come
/// back when a writer opens it.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn single_catalog_open_restores_added_columns() {
    let (_container, url) = start().await;
    let pool = pool(&url).await;
    initialize_single(&pool).await.unwrap();
    let added = [
        ("ducklake_snapshot", "next_file_id"),
        ("ducklake_column", "default_value_dialect"),
        ("ducklake_metadata", "scope_id"),
    ];
    for (table, column) in added {
        sqlx::query(AssertSqlSafe(format!(
            "ALTER TABLE {table} DROP COLUMN {column}"
        )))
        .execute(&pool)
        .await
        .unwrap();
    }

    PostgresSingleCatalogMetadataWriter::from_pool(pool.clone())
        .await
        .unwrap();

    assert_eq!(columns_present(&pool, &added).await, added.len() as i64);
}

/// A role that reads and writes the catalog tables of a current store, without
/// owning them or `CREATE` on the schema, initializes it in either layout.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn a_role_without_ddl_privileges_initializes_a_current_store() {
    for multicatalog in [true, false] {
        let (_container, url) = start().await;
        let owner = pool(&url).await;
        if multicatalog {
            initialize_multicatalog_schema(&owner).await.unwrap();
        } else {
            initialize_single(&owner).await.unwrap();
        }
        sqlx::raw_sql(
            "CREATE ROLE writer LOGIN PASSWORD 'writer';
             REVOKE CREATE ON SCHEMA public FROM PUBLIC;
             GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO writer;
             GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO writer",
        )
        .execute(&owner)
        .await
        .unwrap();
        let writer = pool(&url.replace("postgres:postgres@", "writer:writer@")).await;

        let result = if multicatalog {
            initialize_multicatalog_schema(&writer).await
        } else {
            initialize_single(&writer).await
        };

        result.unwrap_or_else(|e| panic!("multicatalog {multicatalog}: {e}"));
    }
}

/// Initializations racing to create a new multicatalog store may fail some of
/// them, as in official DuckLake, but at least one succeeds, the store ends
/// with every relation a clean one has, and one more initialization succeeds.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn concurrent_initializations_of_a_new_multicatalog_store_leave_it_whole() {
    let (_container, url) = start().await;
    let pool = pool(&url).await;
    initialize_multicatalog_schema(&pool).await.unwrap();
    let complete = relations(&pool).await;
    for round in 0..5 {
        reset(&pool).await;
        let results =
            futures::future::join_all((0..12).map(|_| initialize_multicatalog_schema(&pool))).await;
        let failed = results.iter().filter(|result| result.is_err()).count();
        let raced = relations(&pool).await;

        assert!(
            failed < results.len(),
            "round {round}: every initialization failed"
        );
        assert_eq!(raced, complete, "round {round}: {failed} failed");
        initialize_multicatalog_schema(&pool)
            .await
            .unwrap_or_else(|e| panic!("round {round}: {e}"));
    }
}

/// Initializations racing to create a new single catalog may fail some of
/// them, as in official DuckLake, but at least one succeeds, the catalog ends
/// with every relation a clean one has, and one more initialization succeeds.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn concurrent_initializations_of_a_new_single_catalog_leave_it_whole() {
    let (_container, url) = start().await;
    let pool = pool(&url).await;
    initialize_single(&pool).await.unwrap();
    let complete = relations(&pool).await;
    for round in 0..5 {
        reset(&pool).await;
        let tasks = (0..12).map(|_| {
            let pool = pool.clone();
            tokio::spawn(async move { initialize_single(&pool).await })
        });
        let results = futures::future::join_all(tasks).await;
        let failed = results
            .into_iter()
            .map(|result| result.unwrap())
            .filter(|result| result.is_err())
            .count();
        let raced = relations(&pool).await;

        assert!(failed < 12, "round {round}: every initialization failed");
        assert_eq!(raced, complete, "round {round}: {failed} failed");
        initialize_single(&pool)
            .await
            .unwrap_or_else(|e| panic!("round {round}: {e}"));
    }
}
