#![cfg(feature = "write-postgres")]
//! Multi-table commits: `commit_batch_in_tx` / `PostgresMetadataWriter::commit_batch`.
//!
//! Covers:
//! - N tables becoming visible at ONE snapshot, with one catalog-head advance
//! - per-table entry sequencing (`Replace` then `Append`s share the snapshot)
//! - `schema_version` decided once for the batch — the regression a naive
//!   per-table loop introduces
//! - `Truncate` as a metadata-only empty (no dummy data file)
//! - a conflicting entry aborting the whole batch
//! - the caller keeping ownership of the transaction (the embedder property)
//! - request validation

use datafusion_ducklake::metadata_writer::{
    BatchEntry, BatchOp, ColumnDef, DataFileInfo, MetadataWriter, WriteMode,
};
use datafusion_ducklake::{
    MulticatalogManager, PostgresMetadataWriter, commit_batch_in_tx, initialize_multicatalog_schema,
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

/// Columns with one extra field, so a table written with `cols()` and then with
/// these registers as a schema change.
fn cols_widened() -> Vec<ColumnDef> {
    let mut c = cols();
    c.push(ColumnDef::new("email", "varchar", true).unwrap());
    c
}

async fn writer_for(pool: &PgPool, catalog: &str) -> (PostgresMetadataWriter, i64) {
    let mgr = MulticatalogManager::new(pool.clone());
    let catalog_id = mgr.create_catalog(catalog).await.unwrap();
    let writer = PostgresMetadataWriter::with_pool(pool.clone(), catalog_id)
        .await
        .unwrap();
    writer.set_data_path("/data").unwrap();
    (writer, catalog_id)
}

/// Plan one entry the way a caller would: reserve ids against the live catalog,
/// then describe the file to register.
fn entry(
    writer: &PostgresMetadataWriter,
    table: &str,
    op: BatchOp,
    columns: Vec<ColumnDef>,
    file: Option<DataFileInfo>,
) -> BatchEntry {
    let mode = match op {
        BatchOp::Append => WriteMode::Append,
        BatchOp::Replace | BatchOp::Truncate => WriteMode::Replace,
    };
    let setup = writer
        .begin_write_transaction("public", table, &columns, mode)
        .unwrap();
    BatchEntry {
        schema_name: "public".to_string(),
        table_name: table.to_string(),
        table_id_hint: setup.table_id,
        op,
        file,
        columns,
        column_ids: setup.column_ids,
        base_snapshot: setup.base_snapshot_id,
    }
}

fn file(path: &str, records: i64) -> Option<DataFileInfo> {
    Some(DataFileInfo::new(path, 1024, records))
}

async fn head(pool: &PgPool, catalog_id: i64) -> i64 {
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

async fn schema_version_of(pool: &PgPool, snapshot_id: i64) -> i64 {
    sqlx::query("SELECT schema_version FROM ducklake_snapshot WHERE snapshot_id = $1")
        .bind(snapshot_id)
        .fetch_one(pool)
        .await
        .unwrap()
        .try_get(0)
        .unwrap()
}

/// Live data files of a table, as `(path, begin_snapshot)`.
async fn live_files(pool: &PgPool, table_id: i64) -> Vec<(String, i64)> {
    sqlx::query(
        "SELECT path, begin_snapshot FROM ducklake_data_file
         WHERE table_id = $1 AND end_snapshot IS NULL ORDER BY data_file_id",
    )
    .bind(table_id)
    .fetch_all(pool)
    .await
    .unwrap()
    .into_iter()
    .map(|r| (r.try_get(0).unwrap(), r.try_get(1).unwrap()))
    .collect()
}

async fn table_record_count(pool: &PgPool, table_id: i64) -> i64 {
    sqlx::query("SELECT record_count FROM ducklake_table_stats WHERE table_id = $1")
        .bind(table_id)
        .fetch_one(pool)
        .await
        .unwrap()
        .try_get(0)
        .unwrap()
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn batch_commits_many_tables_at_one_snapshot() {
    let (pool, _c) = spin_up_postgres().await.unwrap();
    let (writer, catalog_id) = writer_for(&pool, "prod").await;
    let before = head(&pool, catalog_id).await;

    let entries = vec![
        entry(
            &writer,
            "orders",
            BatchOp::Replace,
            cols(),
            file("orders/f1.parquet", 10),
        ),
        entry(
            &writer,
            "customers",
            BatchOp::Replace,
            cols(),
            file("customers/f1.parquet", 20),
        ),
        entry(
            &writer,
            "events",
            BatchOp::Replace,
            cols(),
            file("events/f1.parquet", 30),
        ),
    ];
    let committed = writer.commit_batch(&entries).unwrap();

    assert_eq!(committed.tables.len(), 3);
    // One snapshot for the whole batch, and it is the new head — a single advance.
    assert_eq!(head(&pool, catalog_id).await, committed.snapshot_id);
    assert!(committed.snapshot_id > before);
    let mapped: i64 = sqlx::query(
        "SELECT COUNT(*) FROM ducklake_catalog_snapshot_map
         WHERE catalog_id = $1 AND snapshot_id > $2",
    )
    .bind(catalog_id)
    .bind(before)
    .fetch_one(&pool)
    .await
    .unwrap()
    .try_get(0)
    .unwrap();
    assert_eq!(mapped, 1, "a batch maps exactly one new snapshot");

    // Every table's file carries the shared snapshot as its begin.
    for table in &committed.tables {
        let files = live_files(&pool, table.table_id).await;
        assert_eq!(
            files.len(),
            1,
            "{} should have one live file",
            table.table_name
        );
        assert_eq!(
            files[0].1, committed.snapshot_id,
            "{} should begin at the batch snapshot",
            table.table_name
        );
    }
    let counts: Vec<i64> = committed.tables.iter().map(|t| t.record_count).collect();
    assert_eq!(
        counts,
        vec![10, 20, 30],
        "reported in first-appearance order"
    );
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn batch_replace_then_appends_share_the_snapshot() {
    let (pool, _c) = spin_up_postgres().await.unwrap();
    let (writer, catalog_id) = writer_for(&pool, "prod").await;

    // Seed a generation to be superseded.
    let seed = vec![entry(
        &writer,
        "orders",
        BatchOp::Replace,
        cols(),
        file("orders/old.parquet", 5),
    )];
    let seeded = writer.commit_batch(&seed).unwrap();
    let table_id = seeded.tables[0].table_id;

    // replace, append, append — all in one batch, one snapshot.
    let setup = writer
        .begin_write_transaction("public", "orders", &cols(), WriteMode::Replace)
        .unwrap();
    let mk = |op: BatchOp, path: &str, records: i64| BatchEntry {
        schema_name: "public".to_string(),
        table_name: "orders".to_string(),
        table_id_hint: setup.table_id,
        op,
        file: file(path, records),
        columns: cols(),
        column_ids: setup.column_ids.clone(),
        base_snapshot: setup.base_snapshot_id,
    };
    let committed = writer
        .commit_batch(&[
            mk(BatchOp::Replace, "orders/f1.parquet", 1),
            mk(BatchOp::Append, "orders/f2.parquet", 2),
            mk(BatchOp::Append, "orders/f3.parquet", 3),
        ])
        .unwrap();

    assert_eq!(
        committed.tables.len(),
        1,
        "three entries on one table report as one table"
    );
    assert_eq!(committed.tables[0].record_count, 6);

    // The prior generation is retired; the batch's own three files survive — the
    // `begin_snapshot < S` guard must not eat them.
    let files = live_files(&pool, table_id).await;
    let paths: Vec<&str> = files.iter().map(|(p, _)| p.as_str()).collect();
    assert_eq!(
        paths,
        vec!["orders/f1.parquet", "orders/f2.parquet", "orders/f3.parquet"]
    );
    assert!(
        files
            .iter()
            .all(|(_, begin)| *begin == committed.snapshot_id)
    );
    let retired: i64 = sqlx::query(
        "SELECT end_snapshot FROM ducklake_data_file
         WHERE table_id = $1 AND path = 'orders/old.parquet'",
    )
    .bind(table_id)
    .fetch_one(&pool)
    .await
    .unwrap()
    .try_get(0)
    .unwrap();
    assert_eq!(retired, committed.snapshot_id);
    assert_eq!(table_record_count(&pool, table_id).await, 6);
    assert_eq!(head(&pool, catalog_id).await, committed.snapshot_id);
}

/// The regression a per-table `schema_version` decision introduces.
///
/// Within one unmapped snapshot every table reads the same `MAX`-over-mapped
/// predecessor, so a DDL table writes `prev + 1` and a DML table ordered after it
/// writes `prev` straight back — losing the bump. The version has to be decided once
/// over the union of tables.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn batch_bumps_schema_version_once_with_ddl_before_dml() {
    let (pool, _c) = spin_up_postgres().await.unwrap();
    let (writer, catalog_id) = writer_for(&pool, "prod").await;

    // Seed `existing` so a same-columns write against it is pure DML.
    let seeded = writer
        .commit_batch(&[entry(
            &writer,
            "existing",
            BatchOp::Replace,
            cols(),
            file("existing/f0.parquet", 1),
        )])
        .unwrap();
    let baseline = schema_version_of(&pool, seeded.snapshot_id).await;

    // DDL first (brand-new table), DML second (same columns as seeded). Ordering the
    // DML last is what exposes a per-table decision.
    let committed = writer
        .commit_batch(&[
            entry(
                &writer,
                "fresh",
                BatchOp::Replace,
                cols(),
                file("fresh/f1.parquet", 1),
            ),
            entry(
                &writer,
                "existing",
                BatchOp::Append,
                cols(),
                file("existing/f1.parquet", 1),
            ),
        ])
        .unwrap();

    assert_eq!(
        committed.schema_version,
        baseline + 1,
        "a batch containing any DDL bumps exactly once, and a later DML entry must \
         not lower it back"
    );
    assert_eq!(
        schema_version_of(&pool, committed.snapshot_id).await,
        baseline + 1,
        "the persisted snapshot must agree with the reported version"
    );

    // Dense: one row for the table that changed, none for the one that didn't.
    let ddl_rows: Vec<(i64, i64)> = sqlx::query(
        "SELECT table_id, schema_version FROM ducklake_schema_versions
         WHERE begin_snapshot = $1",
    )
    .bind(committed.snapshot_id)
    .fetch_all(&pool)
    .await
    .unwrap()
    .into_iter()
    .map(|r| (r.try_get(0).unwrap(), r.try_get(1).unwrap()))
    .collect();
    let fresh = committed
        .tables
        .iter()
        .find(|t| t.table_name == "fresh")
        .unwrap();
    assert_eq!(ddl_rows, vec![(fresh.table_id, baseline + 1)]);
    assert!(fresh.schema_changed);
    assert!(
        !committed
            .tables
            .iter()
            .find(|t| t.table_name == "existing")
            .unwrap()
            .schema_changed
    );
    assert_eq!(head(&pool, catalog_id).await, committed.snapshot_id);
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn batch_carries_schema_version_forward_when_no_table_changed() {
    let (pool, _c) = spin_up_postgres().await.unwrap();
    let (writer, _catalog_id) = writer_for(&pool, "prod").await;

    let seeded = writer
        .commit_batch(&[entry(
            &writer,
            "orders",
            BatchOp::Replace,
            cols(),
            file("orders/f0.parquet", 1),
        )])
        .unwrap();
    let baseline = schema_version_of(&pool, seeded.snapshot_id).await;

    let committed = writer
        .commit_batch(&[entry(
            &writer,
            "orders",
            BatchOp::Append,
            cols(),
            file("orders/f1.parquet", 1),
        )])
        .unwrap();
    assert_eq!(
        committed.schema_version, baseline,
        "a DML-only batch carries the version forward"
    );
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn batch_bumps_once_for_several_changed_tables() {
    let (pool, _c) = spin_up_postgres().await.unwrap();
    let (writer, _catalog_id) = writer_for(&pool, "prod").await;

    // Two brand-new tables in one batch: both DDL, but one version for the snapshot.
    let committed = writer
        .commit_batch(&[
            entry(
                &writer,
                "a",
                BatchOp::Replace,
                cols(),
                file("a/f1.parquet", 1),
            ),
            entry(
                &writer,
                "b",
                BatchOp::Replace,
                cols(),
                file("b/f1.parquet", 1),
            ),
        ])
        .unwrap();
    assert_eq!(committed.schema_version, 1);
    let rows: i64 =
        sqlx::query("SELECT COUNT(*) FROM ducklake_schema_versions WHERE begin_snapshot = $1")
            .bind(committed.snapshot_id)
            .fetch_one(&pool)
            .await
            .unwrap()
            .try_get(0)
            .unwrap();
    assert_eq!(rows, 2, "one row per changed table, sharing the version");

    // A widening batch bumps once more, not once per table.
    let next = writer
        .commit_batch(&[
            entry(
                &writer,
                "a",
                BatchOp::Replace,
                cols_widened(),
                file("a/f2.parquet", 1),
            ),
            entry(
                &writer,
                "b",
                BatchOp::Replace,
                cols_widened(),
                file("b/f2.parquet", 1),
            ),
        ])
        .unwrap();
    assert_eq!(next.schema_version, 2, "dense: exactly one bump");
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn batch_truncate_empties_a_table_without_a_data_file() {
    let (pool, _c) = spin_up_postgres().await.unwrap();
    let (writer, catalog_id) = writer_for(&pool, "prod").await;

    let seeded = writer
        .commit_batch(&[entry(
            &writer,
            "stale",
            BatchOp::Replace,
            cols(),
            file("stale/f0.parquet", 7),
        )])
        .unwrap();
    let table_id = seeded.tables[0].table_id;
    assert_eq!(live_files(&pool, table_id).await.len(), 1);

    // Truncate alongside a normal load, so the batch has a reason to exist.
    let committed = writer
        .commit_batch(&[
            entry(&writer, "stale", BatchOp::Truncate, cols(), None),
            entry(
                &writer,
                "fresh",
                BatchOp::Replace,
                cols(),
                file("fresh/f1.parquet", 3),
            ),
        ])
        .unwrap();

    assert!(
        live_files(&pool, table_id).await.is_empty(),
        "truncate retires every live file"
    );
    assert_eq!(table_record_count(&pool, table_id).await, 0);
    // No dummy zero-row file was registered for the truncated table.
    let files_at_snapshot: i64 = sqlx::query(
        "SELECT COUNT(*) FROM ducklake_data_file WHERE table_id = $1 AND begin_snapshot = $2",
    )
    .bind(table_id)
    .bind(committed.snapshot_id)
    .fetch_one(&pool)
    .await
    .unwrap()
    .try_get(0)
    .unwrap();
    assert_eq!(files_at_snapshot, 0);
    assert_eq!(
        committed
            .tables
            .iter()
            .find(|t| t.table_name == "stale")
            .unwrap()
            .record_count,
        0
    );
    // The table loaded alongside it is unaffected.
    let fresh = committed
        .tables
        .iter()
        .find(|t| t.table_name == "fresh")
        .unwrap();
    assert_eq!(live_files(&pool, fresh.table_id).await.len(), 1);
    assert_eq!(head(&pool, catalog_id).await, committed.snapshot_id);
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn batch_aborts_entirely_when_one_entry_conflicts() {
    let (pool, _c) = spin_up_postgres().await.unwrap();
    let (writer, catalog_id) = writer_for(&pool, "prod").await;

    let seeded = writer
        .commit_batch(&[entry(
            &writer,
            "orders",
            BatchOp::Replace,
            cols(),
            file("orders/f0.parquet", 4),
        )])
        .unwrap();
    let orders_id = seeded.tables[0].table_id;

    // Plan a Replace against a stale base, then move the table on so the base is
    // genuinely behind.
    let mut stale = entry(
        &writer,
        "orders",
        BatchOp::Replace,
        cols(),
        file("orders/f_stale.parquet", 1),
    );
    stale.base_snapshot = seeded.snapshot_id - 1;

    let head_before = head(&pool, catalog_id).await;
    let err = writer
        .commit_batch(&[
            entry(
                &writer,
                "customers",
                BatchOp::Replace,
                cols(),
                file("customers/f1.parquet", 9),
            ),
            stale,
        ])
        .expect_err("a conflicting entry must abort the batch");
    assert!(
        matches!(err, datafusion_ducklake::DuckLakeError::Conflict(_)),
        "expected Conflict, got {err:?}"
    );

    // Nothing from the batch survived — not even the entry that preceded the
    // conflicting one.
    assert_eq!(head(&pool, catalog_id).await, head_before);
    let files = live_files(&pool, orders_id).await;
    assert_eq!(
        files.iter().map(|(p, _)| p.as_str()).collect::<Vec<_>>(),
        vec!["orders/f0.parquet"],
        "the seeded generation is untouched"
    );
    let customers_exists: Option<i64> = sqlx::query_scalar(
        "SELECT t.table_id FROM ducklake_table t WHERE t.table_name = 'customers'",
    )
    .fetch_optional(&pool)
    .await
    .unwrap();
    assert!(
        customers_exists.is_none(),
        "the first entry's table must not exist after the abort"
    );
}

/// The embedder property: the caller's transaction is still theirs afterwards, so
/// their own metadata commits atomically with the DuckLake commit.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn caller_metadata_commits_with_the_batch() {
    let (pool, _c) = spin_up_postgres().await.unwrap();
    let (writer, catalog_id) = writer_for(&pool, "prod").await;
    sqlx::query("CREATE TABLE app_loads (commit_id TEXT PRIMARY KEY, snapshot_id BIGINT NOT NULL)")
        .execute(&pool)
        .await
        .unwrap();

    let entries = vec![entry(
        &writer,
        "orders",
        BatchOp::Replace,
        cols(),
        file("orders/f1.parquet", 12),
    )];

    let mut tx = pool.begin().await.unwrap();
    let committed = commit_batch_in_tx(&mut tx, catalog_id, 30_000, &entries)
        .await
        .unwrap();
    // The caller writes its own bookkeeping in the SAME transaction.
    sqlx::query("INSERT INTO app_loads (commit_id, snapshot_id) VALUES ($1, $2)")
        .bind("load-1")
        .bind(committed.snapshot_id)
        .execute(&mut *tx)
        .await
        .unwrap();
    tx.commit().await.unwrap();

    assert_eq!(head(&pool, catalog_id).await, committed.snapshot_id);
    let recorded: i64 = sqlx::query("SELECT snapshot_id FROM app_loads WHERE commit_id = 'load-1'")
        .fetch_one(&pool)
        .await
        .unwrap()
        .try_get(0)
        .unwrap();
    assert_eq!(recorded, committed.snapshot_id);
}

/// The other half of the same property: rolling back takes the DuckLake commit with
/// it, so there is no committed-but-unrecorded snapshot to reconcile.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn rolling_back_the_caller_transaction_discards_the_batch() {
    let (pool, _c) = spin_up_postgres().await.unwrap();
    let (writer, catalog_id) = writer_for(&pool, "prod").await;
    let head_before = head(&pool, catalog_id).await;

    let entries = vec![entry(
        &writer,
        "orders",
        BatchOp::Replace,
        cols(),
        file("orders/f1.parquet", 12),
    )];

    let mut tx = pool.begin().await.unwrap();
    let committed = commit_batch_in_tx(&mut tx, catalog_id, 30_000, &entries)
        .await
        .unwrap();
    tx.rollback().await.unwrap();

    assert_eq!(
        head(&pool, catalog_id).await,
        head_before,
        "the head must not have advanced"
    );
    let snapshot_exists: Option<i64> =
        sqlx::query_scalar("SELECT snapshot_id FROM ducklake_snapshot WHERE snapshot_id = $1")
            .bind(committed.snapshot_id)
            .fetch_optional(&pool)
            .await
            .unwrap();
    assert!(snapshot_exists.is_none(), "the snapshot row is gone");
    let table_exists: Option<i64> =
        sqlx::query_scalar("SELECT table_id FROM ducklake_table WHERE table_name = 'orders'")
            .fetch_optional(&pool)
            .await
            .unwrap();
    assert!(table_exists.is_none(), "the table was never created");
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn batch_rejects_malformed_requests() {
    let (pool, _c) = spin_up_postgres().await.unwrap();
    let (writer, _catalog_id) = writer_for(&pool, "prod").await;

    assert!(
        writer.commit_batch(&[]).is_err(),
        "an empty batch is rejected"
    );

    // Truncate must not carry a file.
    let mut bad = entry(&writer, "t", BatchOp::Truncate, cols(), None);
    bad.file = file("t/f1.parquet", 1);
    assert!(writer.commit_batch(&[bad]).is_err());

    // Append/Replace require one.
    let missing = entry(&writer, "t", BatchOp::Append, cols(), None);
    assert!(writer.commit_batch(&[missing]).is_err());

    // column_ids must be 1:1 with columns.
    let mut ragged = entry(
        &writer,
        "t",
        BatchOp::Replace,
        cols(),
        file("t/f1.parquet", 1),
    );
    ragged.column_ids.pop();
    assert!(writer.commit_batch(&[ragged]).is_err());

    // Only the FIRST entry for a table may retire it.
    let first = entry(
        &writer,
        "t",
        BatchOp::Replace,
        cols(),
        file("t/f1.parquet", 1),
    );
    let mut second = first.clone();
    second.op = BatchOp::Replace;
    second.file = file("t/f2.parquet", 1);
    assert!(
        writer.commit_batch(&[first.clone(), second]).is_err(),
        "a later Replace on the same table is rejected"
    );

    // Entries for one table must agree on columns.
    let mut divergent = first.clone();
    divergent.op = BatchOp::Append;
    divergent.columns = cols_widened();
    divergent.column_ids.push(999);
    assert!(writer.commit_batch(&[first, divergent]).is_err());

    // A rejected batch writes nothing.
    let orphan: Option<i64> =
        sqlx::query_scalar("SELECT table_id FROM ducklake_table WHERE table_name = 't'")
            .fetch_optional(&pool)
            .await
            .unwrap();
    assert!(orphan.is_none());
}
