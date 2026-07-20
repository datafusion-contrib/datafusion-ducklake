//! Partitioned write tests (SQLite single-catalog).
//!
//! Exercises the full partitioned-INSERT path: set a spec, INSERT via SQL, and
//! verify the crate writes one data file per partition with the correct
//! `partition_id` + `ducklake_file_partition_value` rows, reads them back, and
//! prunes on the partition column.

#![cfg(all(feature = "write-sqlite", feature = "metadata-sqlite"))]

use std::sync::Arc;

use arrow::datatypes::{DataType, TimeUnit};
use datafusion::prelude::*;
use tempfile::TempDir;

use datafusion_ducklake::metadata_provider::MetadataProvider;
use datafusion_ducklake::partition::PartitionTransform;
use datafusion_ducklake::{
    ColumnDef, DuckLakeCatalog, MetadataWriter, SqliteMetadataProvider, SqliteMetadataWriter,
    WriteMode, execute_ducklake_sql,
};

struct Env {
    conn_str: String,
    table_id: i64,
    _temp: TempDir,
}

/// Create a writable SQLite catalog with an `events(id, region, ts)` table that is
/// partitioned by `(region, year(ts))` BEFORE any data is written.
async fn setup() -> Env {
    let temp = TempDir::new().unwrap();
    let db_path = temp.path().join("test.db");
    let data_path = temp.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();
    let conn_str = format!("sqlite:{}?mode=rwc", db_path.display());

    let writer = SqliteMetadataWriter::new_with_init(&conn_str)
        .await
        .unwrap();
    writer.set_data_path(data_path.to_str().unwrap()).unwrap();

    let ts_type = DataType::Timestamp(TimeUnit::Microsecond, None);
    let cols = vec![
        ColumnDef::from_arrow("id", &DataType::Int32, false).unwrap(),
        ColumnDef::from_arrow("region", &DataType::Utf8, true).unwrap(),
        ColumnDef::from_arrow("ts", &ts_type, true).unwrap(),
    ];
    // Create the (empty) table, then set the partition spec — so a catalog opened
    // afterwards pins a snapshot that already has the spec.
    let s = writer
        .begin_write_transaction("main", "events", &cols, WriteMode::Replace)
        .unwrap();
    writer
        .publish_snapshot(
            s.table_id,
            "main",
            "events",
            s.snapshot_id,
            WriteMode::Replace,
            s.base_snapshot_id,
            &cols,
            &s.column_ids,
        )
        .unwrap();
    writer
        .set_partition_spec(
            s.table_id,
            &[
                ("region".to_string(), PartitionTransform::Identity),
                ("ts".to_string(), PartitionTransform::Year),
            ],
        )
        .unwrap();

    Env {
        conn_str,
        table_id: s.table_id,
        _temp: temp,
    }
}

/// Open a fresh writable context (pins the current head).
async fn write_ctx(conn_str: &str) -> SessionContext {
    let writer = SqliteMetadataWriter::new_with_init(conn_str).await.unwrap();
    let provider = SqliteMetadataProvider::new(conn_str).await.unwrap();
    let catalog = DuckLakeCatalog::with_writer(Arc::new(provider), Arc::new(writer)).unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));
    ctx
}

/// Open a fresh read-only context (pins the current head to see prior writes).
async fn read_ctx(conn_str: &str) -> SessionContext {
    let provider = SqliteMetadataProvider::new(conn_str).await.unwrap();
    let catalog = DuckLakeCatalog::new(provider).unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));
    ctx
}

const INSERT_SQL: &str = "INSERT INTO ducklake.main.events \
     SELECT * FROM (VALUES \
        (1, 'us', TIMESTAMP '2023-01-15 10:00:00'), \
        (2, 'us', TIMESTAMP '2024-06-20 12:00:00'), \
        (3, 'eu', TIMESTAMP '2023-03-10 08:00:00'), \
        (4, 'eu', TIMESTAMP '2024-11-05 18:00:00')) AS t(id, region, ts)";

#[tokio::test(flavor = "multi_thread")]
async fn partitioned_insert_writes_one_file_per_partition() {
    let env = setup().await;

    let ctx = write_ctx(&env.conn_str).await;
    let inserted = ctx.sql(INSERT_SQL).await.unwrap().collect().await.unwrap();
    // INSERT reports the number of rows written.
    let count = inserted[0]
        .column(0)
        .as_any()
        .downcast_ref::<arrow::array::UInt64Array>()
        .unwrap()
        .value(0);
    assert_eq!(count, 4, "4 rows inserted");

    // The write should produce four files — one per (region, year) partition —
    // each carrying two partition values.
    let provider = SqliteMetadataProvider::new(&env.conn_str).await.unwrap();
    let snapshot = provider.get_current_snapshot().unwrap();
    let page = provider
        .get_table_file_metadata_page(env.table_id, snapshot, None, 4096)
        .unwrap();
    assert_eq!(page.len(), 4, "one file per (region, year) partition");
    for meta in &page {
        assert!(
            meta.file.partition_id.is_none() || meta.file.partition_id.is_some(),
            "partition_id column readable"
        );
        assert_eq!(
            meta.file.partition_values.len(),
            2,
            "each file has (region, year) values"
        );
    }
    // The distinct (region, year) tuples cover the four expected partitions.
    let mut tuples: Vec<(Option<String>, Option<String>)> = page
        .iter()
        .map(|m| {
            let region = m
                .file
                .partition_values
                .iter()
                .find(|(k, _)| *k == 0)
                .and_then(|(_, v)| v.clone());
            let year = m
                .file
                .partition_values
                .iter()
                .find(|(k, _)| *k == 1)
                .and_then(|(_, v)| v.clone());
            (region, year)
        })
        .collect();
    tuples.sort();
    assert_eq!(
        tuples,
        vec![
            (Some("eu".to_string()), Some("2023".to_string())),
            (Some("eu".to_string()), Some("2024".to_string())),
            (Some("us".to_string()), Some("2023".to_string())),
            (Some("us".to_string()), Some("2024".to_string())),
        ]
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn partitioned_insert_reads_back_correctly_and_prunes() {
    let env = setup().await;
    write_ctx(&env.conn_str)
        .await
        .sql(INSERT_SQL)
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    // Read back all rows.
    let rctx = read_ctx(&env.conn_str).await;
    let all = rctx
        .sql("SELECT id FROM ducklake.main.events ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let total: usize = all.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total, 4);

    // Filter on the partition column: correct rows + pruned plan.
    let us = rctx
        .sql("SELECT id FROM ducklake.main.events WHERE region = 'us'")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let us_total: usize = us.iter().map(|b| b.num_rows()).sum();
    assert_eq!(us_total, 2, "two 'us' rows");

    let plan = rctx
        .sql("SELECT id FROM ducklake.main.events WHERE region = 'us'")
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    let display = datafusion::physical_plan::displayable(plan.as_ref())
        .indent(true)
        .to_string();
    let files = display.matches(".parquet").count();
    // The table is partitioned by (region, year(ts)) → 4 files (us/eu × 2023/2024).
    // Filtering region='us' must prune the two 'eu' files via the identity bound,
    // keeping exactly the two 'us' files (not all 4, and never fewer than 2).
    assert_eq!(
        files, 2,
        "region='us' must prune the two 'eu' partition files, keeping exactly the two 'us' files; got {files}:\n{display}"
    );
}

/// Create the `events` table (no partition spec) and return `(conn_str, table_id, temp)`.
async fn create_events_table_no_spec() -> (String, i64, TempDir) {
    let temp = TempDir::new().unwrap();
    let db_path = temp.path().join("test.db");
    let data_path = temp.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();
    let conn_str = format!("sqlite:{}?mode=rwc", db_path.display());

    let writer = SqliteMetadataWriter::new_with_init(&conn_str)
        .await
        .unwrap();
    writer.set_data_path(data_path.to_str().unwrap()).unwrap();
    let ts_type = DataType::Timestamp(TimeUnit::Microsecond, None);
    let cols = vec![
        ColumnDef::from_arrow("id", &DataType::Int32, false).unwrap(),
        ColumnDef::from_arrow("region", &DataType::Utf8, true).unwrap(),
        ColumnDef::from_arrow("ts", &ts_type, true).unwrap(),
    ];
    let s = writer
        .begin_write_transaction("main", "events", &cols, WriteMode::Replace)
        .unwrap();
    writer
        .publish_snapshot(
            s.table_id,
            "main",
            "events",
            s.snapshot_id,
            WriteMode::Replace,
            s.base_snapshot_id,
            &cols,
            &s.column_ids,
        )
        .unwrap();
    (conn_str, s.table_id, temp)
}

async fn writable_catalog(conn_str: &str) -> (SessionContext, Arc<DuckLakeCatalog>) {
    let writer = SqliteMetadataWriter::new_with_init(conn_str).await.unwrap();
    let provider = SqliteMetadataProvider::new(conn_str).await.unwrap();
    let catalog =
        Arc::new(DuckLakeCatalog::with_writer(Arc::new(provider), Arc::new(writer)).unwrap());
    let ctx = SessionContext::new();
    ctx.register_catalog(
        "ducklake",
        Arc::clone(&catalog) as Arc<dyn datafusion::catalog::CatalogProvider>,
    );
    (ctx, catalog)
}

#[tokio::test(flavor = "multi_thread")]
async fn sql_hook_set_and_reset_partitioned_by() {
    let (conn_str, table_id, _temp) = create_events_table_no_spec().await;
    let (ctx, catalog) = writable_catalog(&conn_str).await;

    // SET PARTITIONED BY via the SQL hook.
    execute_ducklake_sql(
        &ctx,
        &catalog,
        "ALTER TABLE ducklake.main.events SET PARTITIONED BY (region, year(ts))",
    )
    .await
    .unwrap();

    let p = SqliteMetadataProvider::new(&conn_str).await.unwrap();
    let snap = p.get_current_snapshot().unwrap();
    let spec = p
        .get_partition_spec(table_id, snap)
        .unwrap()
        .expect("spec set via SQL hook");
    assert_eq!(spec.columns.len(), 2);
    assert_eq!(spec.columns[0].transform, PartitionTransform::Identity);
    assert_eq!(spec.columns[1].transform, PartitionTransform::Year);

    // RESET PARTITIONED BY removes it.
    execute_ducklake_sql(
        &ctx,
        &catalog,
        "ALTER TABLE ducklake.main.events RESET PARTITIONED BY",
    )
    .await
    .unwrap();
    let p2 = SqliteMetadataProvider::new(&conn_str).await.unwrap();
    let snap2 = p2.get_current_snapshot().unwrap();
    assert!(
        p2.get_partition_spec(table_id, snap2).unwrap().is_none(),
        "spec removed after RESET"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn sql_hook_rejects_unknown_transform() {
    let (conn_str, _table_id, _temp) = create_events_table_no_spec().await;
    let (ctx, catalog) = writable_catalog(&conn_str).await;
    let err = execute_ducklake_sql(
        &ctx,
        &catalog,
        "ALTER TABLE ducklake.main.events SET PARTITIONED BY (bucket(4, region))",
    )
    .await
    .unwrap_err();
    assert!(
        err.to_string().to_lowercase().contains("transform"),
        "expected an unsupported-transform error, got: {err}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn sql_hook_delegates_non_partition_sql() {
    let (conn_str, _table_id, _temp) = create_events_table_no_spec().await;
    let (ctx, catalog) = writable_catalog(&conn_str).await;
    // A plain query flows through to ctx.sql unchanged.
    let batches = execute_ducklake_sql(&ctx, &catalog, "SELECT 1 AS x")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total, 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn insert_stays_partitioned_after_repartition() {
    // Regression: a SECOND partition-spec change (re-partition) must not silently
    // make subsequent INSERTs write unpartitioned files under the live spec.
    let (conn_str, table_id, _temp) = create_events_table_no_spec().await;
    {
        let w = SqliteMetadataWriter::new_with_init(&conn_str)
            .await
            .unwrap();
        // Two generations: region (identity) then year(ts).
        w.set_partition_spec(
            table_id,
            &[("region".to_string(), PartitionTransform::Identity)],
        )
        .unwrap();
        w.set_partition_spec(table_id, &[("ts".to_string(), PartitionTransform::Year)])
            .unwrap();
    }

    write_ctx(&conn_str)
        .await
        .sql(INSERT_SQL)
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    // The INSERT must partition by the LIVE spec (year(ts)) → one file per year,
    // not a single unpartitioned file.
    let provider = SqliteMetadataProvider::new(&conn_str).await.unwrap();
    let snapshot = provider.get_current_snapshot().unwrap();
    let page = provider
        .get_table_file_metadata_page(table_id, snapshot, None, 4096)
        .unwrap();
    assert_eq!(
        page.len(),
        2,
        "re-partitioned INSERT must produce one file per year, not one unpartitioned file"
    );
    let mut years: Vec<String> = page
        .iter()
        .filter_map(|m| m.file.partition_values.first().and_then(|(_, v)| v.clone()))
        .collect();
    years.sort();
    assert_eq!(years, vec!["2023".to_string(), "2024".to_string()]);
}

#[tokio::test(flavor = "multi_thread")]
async fn set_partitioned_then_insert_same_session_partitions() {
    // P1 regression: SET PARTITIONED BY via the SQL hook, then INSERT in the SAME
    // session (the catalog was pinned BEFORE the spec existed) must still partition
    // — the write path resolves the spec at the current head, not the pinned snapshot.
    let (conn_str, table_id, _temp) = create_events_table_no_spec().await;
    let (ctx, catalog) = writable_catalog(&conn_str).await; // pins the pre-spec snapshot
    execute_ducklake_sql(
        &ctx,
        &catalog,
        "ALTER TABLE ducklake.main.events SET PARTITIONED BY (region)",
    )
    .await
    .unwrap();
    ctx.sql(INSERT_SQL).await.unwrap().collect().await.unwrap();

    let provider = SqliteMetadataProvider::new(&conn_str).await.unwrap();
    let snap = provider.get_current_snapshot().unwrap();
    let page = provider
        .get_table_file_metadata_page(table_id, snap, None, 4096)
        .unwrap();
    assert_eq!(
        page.len(),
        2,
        "same-session SET then INSERT must partition by region (one file per region)"
    );
    let live_pid = provider
        .get_partition_spec(table_id, snap)
        .unwrap()
        .expect("live spec present")
        .partition_id;
    for m in &page {
        assert_eq!(
            m.file.partition_id,
            Some(live_pid),
            "file must be stamped with the LIVE partition_id, never a retired one"
        );
        assert_eq!(m.file.partition_values.len(), 1);
    }
    let mut regions: Vec<String> = page
        .iter()
        .filter_map(|m| m.file.partition_values.first().and_then(|(_, v)| v.clone()))
        .collect();
    regions.sort();
    assert_eq!(regions, vec!["eu".to_string(), "us".to_string()]);
}

#[tokio::test(flavor = "multi_thread")]
async fn reset_partitioned_then_insert_same_session_is_unpartitioned() {
    // P1 regression: after RESET, a same-session INSERT must write ONE unpartitioned
    // file with NO partition_id — never a retired partition id from the pinned spec.
    let (conn_str, table_id, _temp) = create_events_table_no_spec().await;
    {
        let w = SqliteMetadataWriter::new_with_init(&conn_str)
            .await
            .unwrap();
        w.set_partition_spec(
            table_id,
            &[("region".to_string(), PartitionTransform::Identity)],
        )
        .unwrap();
    }
    let (ctx, catalog) = writable_catalog(&conn_str).await; // pinned where region-spec is live
    execute_ducklake_sql(
        &ctx,
        &catalog,
        "ALTER TABLE ducklake.main.events RESET PARTITIONED BY",
    )
    .await
    .unwrap();
    ctx.sql(INSERT_SQL).await.unwrap().collect().await.unwrap();

    let provider = SqliteMetadataProvider::new(&conn_str).await.unwrap();
    let snap = provider.get_current_snapshot().unwrap();
    let page = provider
        .get_table_file_metadata_page(table_id, snap, None, 4096)
        .unwrap();
    assert_eq!(
        page.len(),
        1,
        "after RESET, a same-session INSERT is unpartitioned"
    );
    assert!(
        page[0].file.partition_id.is_none(),
        "must not stamp a retired partition_id after RESET"
    );
    assert!(page[0].file.partition_values.is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn concurrent_reset_during_insert_conflicts() {
    // P1 (concurrency): a partition spec retired by a concurrent RESET/SET *after*
    // the insert plan captured it but *before* the insert commits must abort at the
    // commit-time fence — never stamp a retired partition_id into a committed file.
    let (conn_str, table_id, _temp) = create_events_table_no_spec().await;
    {
        let w = SqliteMetadataWriter::new_with_init(&conn_str)
            .await
            .unwrap();
        w.set_partition_spec(
            table_id,
            &[("region".to_string(), PartitionTransform::Identity)],
        )
        .unwrap();
    }
    let (ctx, _catalog) = writable_catalog(&conn_str).await;
    // Build the physical plan now: insert_into captures the LIVE spec (region).
    let plan = ctx
        .sql(INSERT_SQL)
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    // Concurrently retire that spec before the captured plan executes.
    {
        let w = SqliteMetadataWriter::new_with_init(&conn_str)
            .await
            .unwrap();
        w.reset_partition_spec(table_id).unwrap();
    }
    // Executing the captured plan must hit the fence and abort with a conflict.
    let result = datafusion::physical_plan::collect(plan, ctx.task_ctx()).await;
    let err = result.expect_err("insert against a concurrently-retired spec must conflict");
    let msg = err.to_string().to_lowercase();
    assert!(
        msg.contains("partition spec") || msg.contains("concurrent"),
        "expected a partition-spec conflict, got: {err}"
    );
    // Nothing committed: the tx rolled back, so no data files exist.
    let provider = SqliteMetadataProvider::new(&conn_str).await.unwrap();
    let snap = provider.get_current_snapshot().unwrap();
    let page = provider
        .get_table_file_metadata_page(table_id, snap, None, 4096)
        .unwrap();
    assert!(
        page.is_empty(),
        "a conflicting insert must not commit any data files"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn concurrent_set_during_unpartitioned_insert_conflicts() {
    // Inverse P1: an unpartitioned INSERT plan captured while the table had NO spec,
    // then a concurrent SET PARTITIONED BY makes it partitioned before the plan
    // commits. The commit must abort — never leave a partition_id-less file in a
    // now-partitioned table.
    let (conn_str, table_id, _temp) = create_events_table_no_spec().await;
    let (ctx, _catalog) = writable_catalog(&conn_str).await;
    // Build the plan now: table is unpartitioned → partition = None.
    let plan = ctx
        .sql(INSERT_SQL)
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    // Concurrently make the table partitioned.
    {
        let w = SqliteMetadataWriter::new_with_init(&conn_str)
            .await
            .unwrap();
        w.set_partition_spec(
            table_id,
            &[("region".to_string(), PartitionTransform::Identity)],
        )
        .unwrap();
    }
    // Executing the stale unpartitioned plan must hit the singular-commit fence.
    let result = datafusion::physical_plan::collect(plan, ctx.task_ctx()).await;
    let err = result.expect_err("unpartitioned insert into a now-partitioned table must conflict");
    let msg = err.to_string().to_lowercase();
    assert!(
        msg.contains("partition spec") || msg.contains("concurrent"),
        "expected a partition-spec conflict, got: {err}"
    );
    // Nothing committed.
    let provider = SqliteMetadataProvider::new(&conn_str).await.unwrap();
    let snap = provider.get_current_snapshot().unwrap();
    let page = provider
        .get_table_file_metadata_page(table_id, snap, None, 4096)
        .unwrap();
    assert!(
        page.is_empty(),
        "a conflicting unpartitioned insert must not commit any data files"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn empty_overwrite_truncates_partitioned_table() {
    // Regression: an empty INSERT OVERWRITE (0 rows) on a PARTITIONED table must
    // truncate (retire the prior generation) via the single-file path — it must NOT
    // trip the inverse fence. The 0-row truncate marker carries no partition_id, but
    // it also carries no data, so it cannot violate the live-spec invariant.
    let env = setup().await; // partitioned by (region, year(ts))
    write_ctx(&env.conn_str)
        .await
        .sql(INSERT_SQL)
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    // Sanity: 4 rows live across the partitions.
    {
        let p = SqliteMetadataProvider::new(&env.conn_str).await.unwrap();
        let s = p.get_current_snapshot().unwrap();
        assert_eq!(p.get_table_row_count(env.table_id, s).unwrap(), 4);
    }
    // Empty INSERT OVERWRITE (WHERE 1=2 → 0 rows) must truncate, not conflict.
    let ctx = write_ctx(&env.conn_str).await;
    ctx.sql(
        "INSERT OVERWRITE ducklake.main.events \
         SELECT * FROM (VALUES (1, 'us', TIMESTAMP '2023-01-15 10:00:00')) AS t(id, region, ts) \
         WHERE 1 = 2",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();
    // After truncate: 0 live rows.
    let p = SqliteMetadataProvider::new(&env.conn_str).await.unwrap();
    let s = p.get_current_snapshot().unwrap();
    assert_eq!(
        p.get_table_row_count(env.table_id, s).unwrap(),
        0,
        "empty INSERT OVERWRITE must truncate the partitioned table, not conflict"
    );
}
