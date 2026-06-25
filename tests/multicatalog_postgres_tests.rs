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
    // register_data_file returns the committed snapshot id (assigned AT commit).
    let g1_snap = w
        .register_data_file(
            s1.table_id,
            "public",
            "users",
            s1.snapshot_id,
            &DataFileInfo::new("g1.parquet", 1024, 10),
            WriteMode::Replace,
            s1.base_snapshot_id,
            &cols(),
            &s1.column_ids,
        )
        .unwrap()
        .snapshot_id;
    assert_eq!(current_head(&pool, cat).await, g1_snap);
    assert_eq!(visible_records_at_head(&pool, cat, s1.table_id).await, 10);

    // Begin generation 2 (Replace). The data upload would run here, between
    // setup and the commit. Under the commit-time model begin writes NOTHING, so
    // the head is unchanged and the old file is NOT yet retired.
    let s2 = w
        .begin_write_transaction("public", "users", &cols(), WriteMode::Replace)
        .unwrap();

    // A reader interleaved during the upload sees the OLD generation intact:
    // head unchanged, old data fully visible. (Pre-fix the head had already
    // advanced to s2 and the old file was already retired ⇒ count == 0.)
    assert_eq!(
        current_head(&pool, cat).await,
        g1_snap,
        "head must NOT advance before the new data file is registered"
    );
    assert_eq!(
        visible_records_at_head(&pool, cat, s1.table_id).await,
        10,
        "old generation must stay fully visible during the upload window"
    );

    // Commit generation 2: head advances and generation 1 retires atomically.
    let g2_snap = w
        .register_data_file(
            s2.table_id,
            "public",
            "users",
            s2.snapshot_id,
            &DataFileInfo::new("g2.parquet", 2048, 7),
            WriteMode::Replace,
            s2.base_snapshot_id,
            &cols(),
            &s2.column_ids,
        )
        .unwrap()
        .snapshot_id;
    assert_eq!(current_head(&pool, cat).await, g2_snap);
    assert_eq!(
        visible_records_at_head(&pool, cat, s2.table_id).await,
        7,
        "new generation visible, old generation retired"
    );
}

/// Two concurrent same-table Replace writers based on the same generation,
/// committing out of reservation order: the FIRST to commit wins and the second
/// — whose base generation is now stale — aborts with a
/// [`datafusion_ducklake::DuckLakeError::Conflict`] (DuckLake-style optimistic
/// concurrency) instead of silently UNIONing both generations at the head.
///
/// This is the multicatalog-Postgres analog of the SQLite
/// `replace_out_of_order_commit_conflicts` regression. Pre-fix, the upload runs
/// outside the catalog lock and there was no commit-time conflict check, so a
/// later-committing writer registered its file alongside the winner's ⇒ both
/// visible ⇒ doubled rows.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn concurrent_replace_conflicts_no_union() {
    use datafusion_ducklake::metadata_writer::DataFileInfo;
    let (pool, _c) = spin_up_postgres().await.unwrap();
    let mgr = MulticatalogManager::new(pool.clone());
    let cat = mgr.create_catalog("pg_conflict").await.unwrap();
    let w = PostgresMetadataWriter::with_pool(pool.clone(), cat)
        .await
        .unwrap();
    w.set_data_path("/data").unwrap();

    // Generation 0: a committed, non-empty table.
    let s0 = w
        .begin_write_transaction("public", "users", &cols(), WriteMode::Replace)
        .unwrap();
    let gen0_snap = w
        .register_data_file(
            s0.table_id,
            "public",
            "users",
            s0.snapshot_id,
            &DataFileInfo::new("gen0.parquet", 1024, 5),
            WriteMode::Replace,
            s0.base_snapshot_id,
            &cols(),
            &s0.column_ids,
        )
        .unwrap()
        .snapshot_id;
    let tid = s0.table_id;

    // Two Replace writers open their windows on the SAME base generation; their
    // parquet uploads would run concurrently here.
    let w1 = w
        .begin_write_transaction("public", "users", &cols(), WriteMode::Replace)
        .unwrap();
    let w2 = w
        .begin_write_transaction("public", "users", &cols(), WriteMode::Replace)
        .unwrap();
    assert_eq!(w1.base_snapshot_id, gen0_snap);
    assert_eq!(w2.base_snapshot_id, gen0_snap);

    // Commit in the OPPOSITE order: w2 (reserved later) commits first and wins;
    // w1's base is now stale.
    let w2_snap = w
        .register_data_file(
            w2.table_id,
            "public",
            "users",
            w2.snapshot_id,
            &DataFileInfo::new("gen_w2.parquet", 2048, 7),
            WriteMode::Replace,
            w2.base_snapshot_id,
            &cols(),
            &w2.column_ids,
        )
        .unwrap()
        .snapshot_id;
    let w1_result = w.register_data_file(
        w1.table_id,
        "public",
        "users",
        w1.snapshot_id,
        &DataFileInfo::new("gen_w1.parquet", 4096, 3),
        WriteMode::Replace,
        w1.base_snapshot_id,
        &cols(),
        &w1.column_ids,
    );
    assert!(
        matches!(
            w1_result,
            Err(datafusion_ducklake::DuckLakeError::Conflict(_))
        ),
        "the out-of-order (stale-base) Replace must abort with a conflict, got {w1_result:?}"
    );

    // Exactly w2's generation is live — NOT the union of w1 + w2.
    assert_eq!(current_head(&pool, cat).await, w2_snap);
    assert_eq!(
        visible_records_at_head(&pool, cat, tid).await,
        7,
        "only the winning (w2) generation is visible; no union with w1"
    );
    let live_files: i64 = sqlx::query(
        "SELECT COUNT(*) FROM ducklake_data_file WHERE table_id = $1 AND end_snapshot IS NULL",
    )
    .bind(tid)
    .fetch_one(&pool)
    .await
    .unwrap()
    .try_get(0)
    .unwrap();
    assert_eq!(
        live_files, 1,
        "exactly one live file after the conflict (no union)"
    );
}

/// A fileless SAME-SCHEMA Replace (`publish_snapshot` — no data file, no column
/// change) leaves no per-table footprint, so a concurrent data Replace does not
/// conflict with it: they resolve LAST-WRITER-WINS, with no union and no
/// corruption. This matches the SQLite writer (whose conflict check is likewise
/// data-file/column based). A schema-changing or data-bearing Replace *does*
/// leave a footprint and is conflict-checked — see the other `concurrent_replace`
/// tests. (The narrow fileless-same-schema case being last-writer-wins rather
/// than abort is documented in COMPATIBILITY.md.)
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn fileless_same_schema_replace_vs_data_write_resolves_last_writer_wins() {
    use datafusion_ducklake::metadata_writer::DataFileInfo;
    let (pool, _c) = spin_up_postgres().await.unwrap();
    let mgr = MulticatalogManager::new(pool.clone());
    let cat = mgr.create_catalog("pg_fileless").await.unwrap();
    let w = PostgresMetadataWriter::with_pool(pool.clone(), cat)
        .await
        .unwrap();
    w.set_data_path("/data").unwrap();

    // Generation 0: an EMPTY table (fileless — no data file), via publish_snapshot.
    let c0 = w
        .begin_write_transaction("public", "t", &cols(), WriteMode::Replace)
        .unwrap();
    w.publish_snapshot(
        c0.table_id,
        "public",
        "t",
        c0.snapshot_id,
        WriteMode::Replace,
        c0.base_snapshot_id,
        &cols(),
        &c0.column_ids,
    )
    .unwrap();
    let tid = c0.table_id;
    let base_head = current_head(&pool, cat).await;

    // A data writer D opens its window on gen0.
    let d = w
        .begin_write_transaction("public", "t", &cols(), WriteMode::Replace)
        .unwrap();
    assert_eq!(d.base_snapshot_id, base_head);

    // A concurrent fileless SAME-SCHEMA Replace C commits first (no file, no
    // column change → no per-table footprint), advancing the head.
    let c = w
        .begin_write_transaction("public", "t", &cols(), WriteMode::Replace)
        .unwrap();
    w.publish_snapshot(
        c.table_id,
        "public",
        "t",
        c.snapshot_id,
        WriteMode::Replace,
        c.base_snapshot_id,
        &cols(),
        &c.column_ids,
    )
    .unwrap();
    assert!(
        current_head(&pool, cat).await > base_head,
        "the fileless replace advanced the head"
    );

    // D commits its data Replace. C left no per-table footprint, so D does NOT
    // conflict — it is the last writer and wins cleanly (no union, no corruption).
    let d_snap = w
        .register_data_file(
            d.table_id,
            "public",
            "t",
            d.snapshot_id,
            &DataFileInfo::new("d.parquet", 1024, 9),
            WriteMode::Replace,
            d.base_snapshot_id,
            &cols(),
            &d.column_ids,
        )
        .unwrap()
        .snapshot_id;
    assert!(d_snap > 0);

    // Exactly D's generation is live: 9 rows, one file (no union with the empty gen).
    assert_eq!(
        visible_records_at_head(&pool, cat, tid).await,
        9,
        "last writer (D) wins cleanly"
    );
    let live_files: i64 = sqlx::query(
        "SELECT COUNT(*) FROM ducklake_data_file WHERE table_id = $1 AND end_snapshot IS NULL",
    )
    .bind(tid)
    .fetch_one(&pool)
    .await
    .unwrap()
    .try_get(0)
    .unwrap();
    assert_eq!(live_files, 1, "exactly D's one file is live (no union)");
}

/// Regression: a same-schema Replace must NOT re-mint column ids — the kept
/// columns keep their stable `column_id` (== parquet field_id), and no new
/// `ducklake_column` rows are written. (Re-minting every Replace was a bug: it
/// corrupted an in-flight Append's field-ids to all-NULL and bloated
/// ducklake_column. Postgres now matches SQLite's surgical, stable-id model.)
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn same_schema_replace_keeps_stable_column_ids() {
    use datafusion_ducklake::metadata_writer::DataFileInfo;
    let (pool, _c) = spin_up_postgres().await.unwrap();
    let mgr = MulticatalogManager::new(pool.clone());
    let cat = mgr.create_catalog("pg_stableids").await.unwrap();
    let w = PostgresMetadataWriter::with_pool(pool.clone(), cat)
        .await
        .unwrap();
    w.set_data_path("/data").unwrap();

    // Generation 0 with data.
    let g0 = w
        .begin_write_transaction("public", "t", &cols(), WriteMode::Replace)
        .unwrap();
    w.register_data_file(
        g0.table_id,
        "public",
        "t",
        g0.snapshot_id,
        &DataFileInfo::new("g0.parquet", 1024, 5),
        WriteMode::Replace,
        g0.base_snapshot_id,
        &cols(),
        &g0.column_ids,
    )
    .unwrap();
    let tid = g0.table_id;

    // Capture the committed live column ids after gen0.
    let ids_before: Vec<i64> = sqlx::query(
        "SELECT column_id FROM ducklake_column
         WHERE table_id = $1 AND end_snapshot IS NULL ORDER BY column_order",
    )
    .bind(tid)
    .fetch_all(&pool)
    .await
    .unwrap()
    .into_iter()
    .map(|r| r.try_get::<i64, _>(0).unwrap())
    .collect();
    assert_eq!(ids_before.len(), cols().len());

    // A second same-schema Replace.
    let g1 = w
        .begin_write_transaction("public", "t", &cols(), WriteMode::Replace)
        .unwrap();
    assert_eq!(
        g1.column_ids, ids_before,
        "begin must hand back the SAME column ids for a same-schema write"
    );
    w.register_data_file(
        g1.table_id,
        "public",
        "t",
        g1.snapshot_id,
        &DataFileInfo::new("g1.parquet", 1024, 7),
        WriteMode::Replace,
        g1.base_snapshot_id,
        &cols(),
        &g1.column_ids,
    )
    .unwrap();

    // Live column ids are unchanged, and the TOTAL column-row count did not grow
    // (no re-mint): exactly one row per column, still live, same ids.
    let ids_after: Vec<i64> = sqlx::query(
        "SELECT column_id FROM ducklake_column
         WHERE table_id = $1 AND end_snapshot IS NULL ORDER BY column_order",
    )
    .bind(tid)
    .fetch_all(&pool)
    .await
    .unwrap()
    .into_iter()
    .map(|r| r.try_get::<i64, _>(0).unwrap())
    .collect();
    assert_eq!(
        ids_after, ids_before,
        "same-schema Replace must keep stable column ids"
    );
    let total_col_rows: i64 =
        sqlx::query("SELECT COUNT(*) FROM ducklake_column WHERE table_id = $1")
            .bind(tid)
            .fetch_one(&pool)
            .await
            .unwrap()
            .try_get(0)
            .unwrap();
    assert_eq!(
        total_col_rows,
        cols().len() as i64,
        "no new column rows written across same-schema replaces (no re-mint churn)"
    );
}

/// External-review P1: an `Append` whose table did not exist at begin reserves
/// FRESH column ids and bakes them into its parquet field-ids. If another writer
/// creates the table first (with different ids), the append's file would resolve
/// those columns to all-NULL. The field-id-drift check must abort the append with
/// `Conflict` (the caller retries against the committed schema) — two concurrent
/// first-writes must not silently corrupt.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn append_racing_table_creation_aborts_on_field_id_drift() {
    use datafusion_ducklake::metadata_writer::DataFileInfo;
    let (pool, _c) = spin_up_postgres().await.unwrap();
    let mgr = MulticatalogManager::new(pool.clone());
    let cat = mgr.create_catalog("pg_drift").await.unwrap();
    let w = PostgresMetadataWriter::with_pool(pool.clone(), cat)
        .await
        .unwrap();
    w.set_data_path("/data").unwrap();

    // Writer A opens an Append on a table that does NOT exist yet → it reserves
    // fresh column ids and stages a parquet with those field-ids.
    let a = w
        .begin_write_transaction("public", "t", &cols(), WriteMode::Append)
        .unwrap();

    // Writer B creates the table first, with its own (different) column ids.
    let b = w
        .begin_write_transaction("public", "t", &cols(), WriteMode::Replace)
        .unwrap();
    w.register_data_file(
        b.table_id,
        "public",
        "t",
        b.snapshot_id,
        &DataFileInfo::new("b.parquet", 1024, 3),
        WriteMode::Replace,
        b.base_snapshot_id,
        &cols(),
        &b.column_ids,
    )
    .unwrap();
    assert_ne!(
        a.column_ids, b.column_ids,
        "A and B reserved distinct fresh column ids for the new table"
    );

    // A commits its Append on B's now-committed table. Its staged field-ids no
    // longer match the committed columns → must abort rather than NULL-corrupt.
    let a_result = w.register_data_file(
        a.table_id,
        "public",
        "t",
        a.snapshot_id,
        &DataFileInfo::new("a.parquet", 1024, 4),
        WriteMode::Append,
        a.base_snapshot_id,
        &cols(),
        &a.column_ids,
    );
    assert!(
        matches!(
            a_result,
            Err(datafusion_ducklake::DuckLakeError::Conflict(_))
        ),
        "append with stale field-ids must abort with Conflict, got {a_result:?}"
    );
}

/// P1 regression (final-review finding): Postgres allocates the global IDENTITY
/// `snapshot_id` at begin but maps the head only at commit, so a commit to a
/// DIFFERENT table in the same catalog can push a later same-table writer's base
/// past an earlier in-flight writer's (lower) id. A scalar `begin_snapshot > head`
/// check would then miss that earlier writer and silently clobber it. The
/// per-table committed-generation compare-and-swap must still abort the loser.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn concurrent_replace_conflicts_despite_intervening_other_table_commit() {
    use datafusion_ducklake::metadata_writer::DataFileInfo;
    let (pool, _c) = spin_up_postgres().await.unwrap();
    let mgr = MulticatalogManager::new(pool.clone());
    let cat = mgr.create_catalog("pg_genmarker").await.unwrap();
    let w = PostgresMetadataWriter::with_pool(pool.clone(), cat)
        .await
        .unwrap();
    w.set_data_path("/data").unwrap();

    // T1 with a committed generation.
    let t1 = w
        .begin_write_transaction("public", "t1", &cols(), WriteMode::Replace)
        .unwrap();
    w.register_data_file(
        t1.table_id,
        "public",
        "t1",
        t1.snapshot_id,
        &DataFileInfo::new("t1_g0.parquet", 1024, 5),
        WriteMode::Replace,
        t1.base_snapshot_id,
        &cols(),
        &t1.column_ids,
    )
    .unwrap();
    let tid1 = t1.table_id;

    // W_a opens a Replace window on T1 (base = T1's committed generation).
    let wa = w
        .begin_write_transaction("public", "t1", &cols(), WriteMode::Replace)
        .unwrap();

    // An UNRELATED writer commits a Replace on a DIFFERENT table T2 in the same
    // catalog, advancing the catalog head past W_a's still-dormant snapshot id.
    let wb = w
        .begin_write_transaction("public", "t2", &cols(), WriteMode::Replace)
        .unwrap();
    w.register_data_file(
        wb.table_id,
        "public",
        "t2",
        wb.snapshot_id,
        &DataFileInfo::new("t2.parquet", 1024, 3),
        WriteMode::Replace,
        wb.base_snapshot_id,
        &cols(),
        &wb.column_ids,
    )
    .unwrap();

    // W_c opens a Replace window on T1 AFTER the head jumped (its catalog-head
    // base is now above W_a's id — the exact trap a scalar check falls into).
    let wc = w
        .begin_write_transaction("public", "t1", &cols(), WriteMode::Replace)
        .unwrap();

    // W_a commits its Replace on T1 first and wins.
    w.register_data_file(
        wa.table_id,
        "public",
        "t1",
        wa.snapshot_id,
        &DataFileInfo::new("t1_ga.parquet", 1024, 7),
        WriteMode::Replace,
        wa.base_snapshot_id,
        &cols(),
        &wa.column_ids,
    )
    .unwrap();

    // W_c commits on a stale base: W_a changed T1's generation. It MUST abort,
    // even though W_a's snapshot id is below W_c's catalog-head base.
    let wc_result = w.register_data_file(
        wc.table_id,
        "public",
        "t1",
        wc.snapshot_id,
        &DataFileInfo::new("t1_gc.parquet", 1024, 9),
        WriteMode::Replace,
        wc.base_snapshot_id,
        &cols(),
        &wc.column_ids,
    );
    assert!(
        matches!(
            wc_result,
            Err(datafusion_ducklake::DuckLakeError::Conflict(_))
        ),
        "W_c must abort against W_a's committed Replace despite the intervening T2 commit, got {wc_result:?}"
    );

    // T1 shows W_a's generation (7 rows) — W_c did not clobber it.
    assert_eq!(
        visible_records_at_head(&pool, cat, tid1).await,
        7,
        "W_a's generation wins; W_c did not silently clobber it"
    );
}

/// Contract: a `Replace` conflicts with a concurrent same-table `Append` that
/// committed since the Replace began (matching DuckLake's overwrite-vs-insert
/// semantics). `Append` itself is never conflict-checked (it commutes), but it
/// raises the table's generation marker, so a Replace built on the pre-Append
/// generation aborts rather than clobbering the appended rows.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn replace_conflicts_with_concurrent_committed_append() {
    use datafusion_ducklake::metadata_writer::DataFileInfo;
    let (pool, _c) = spin_up_postgres().await.unwrap();
    let mgr = MulticatalogManager::new(pool.clone());
    let cat = mgr.create_catalog("pg_replace_vs_append").await.unwrap();
    let w = PostgresMetadataWriter::with_pool(pool.clone(), cat)
        .await
        .unwrap();
    w.set_data_path("/data").unwrap();

    // Generation 0.
    let g0 = w
        .begin_write_transaction("public", "t", &cols(), WriteMode::Replace)
        .unwrap();
    w.register_data_file(
        g0.table_id,
        "public",
        "t",
        g0.snapshot_id,
        &DataFileInfo::new("g0.parquet", 1024, 5),
        WriteMode::Replace,
        g0.base_snapshot_id,
        &cols(),
        &g0.column_ids,
    )
    .unwrap();
    let tid = g0.table_id;

    // A Replace opens its window on gen0.
    let r = w
        .begin_write_transaction("public", "t", &cols(), WriteMode::Replace)
        .unwrap();

    // A concurrent Append on the same table commits first (Append is not
    // conflict-checked) and raises the table's generation marker.
    let a = w
        .begin_write_transaction("public", "t", &cols(), WriteMode::Append)
        .unwrap();
    w.register_data_file(
        a.table_id,
        "public",
        "t",
        a.snapshot_id,
        &DataFileInfo::new("appended.parquet", 1024, 4),
        WriteMode::Append,
        a.base_snapshot_id,
        &cols(),
        &a.column_ids,
    )
    .unwrap();

    // The Replace now commits on a base that predates the Append → conflict.
    let r_result = w.register_data_file(
        r.table_id,
        "public",
        "t",
        r.snapshot_id,
        &DataFileInfo::new("replaced.parquet", 1024, 9),
        WriteMode::Replace,
        r.base_snapshot_id,
        &cols(),
        &r.column_ids,
    );
    assert!(
        matches!(
            r_result,
            Err(datafusion_ducklake::DuckLakeError::Conflict(_))
        ),
        "Replace must abort against a concurrently-committed Append, got {r_result:?}"
    );

    // The appended generation survives (gen0 5 rows + appended 4 = 9); the
    // Replace did not clobber it.
    assert_eq!(
        visible_records_at_head(&pool, cat, tid).await,
        9,
        "the committed Append survives; the Replace did not clobber it"
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
    // mapped predecessors, so without publishing the bump wouldn't advance. The
    // committed snapshot id is assigned AT the commit, so read it back as the
    // catalog head right after publish (begin's snapshot_id is vestigial now).
    let setup1 = writer
        .begin_write_transaction("public", "users", &cols(), WriteMode::Replace)
        .unwrap();
    writer
        .publish_snapshot(
            setup1.table_id,
            "public",
            "users",
            setup1.snapshot_id,
            WriteMode::Replace,
            setup1.base_snapshot_id,
            &cols(),
            &setup1.column_ids,
        )
        .unwrap();
    let snap1 = current_head(&pool, catalog_id).await;

    // Second commit: same columns -> DML, carry forward schema_version.
    let setup2 = writer
        .begin_write_transaction("public", "users", &cols(), WriteMode::Replace)
        .unwrap();
    writer
        .publish_snapshot(
            setup2.table_id,
            "public",
            "users",
            setup2.snapshot_id,
            WriteMode::Replace,
            setup2.base_snapshot_id,
            &cols(),
            &setup2.column_ids,
        )
        .unwrap();
    let snap2 = current_head(&pool, catalog_id).await;

    let v1: i64 =
        sqlx::query("SELECT schema_version FROM ducklake_snapshot WHERE snapshot_id = $1")
            .bind(snap1)
            .fetch_one(&pool)
            .await
            .unwrap()
            .try_get(0)
            .unwrap();
    let v2: i64 =
        sqlx::query("SELECT schema_version FROM ducklake_snapshot WHERE snapshot_id = $1")
            .bind(snap2)
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
            "public",
            "users",
            setup3.snapshot_id,
            WriteMode::Replace,
            setup3.base_snapshot_id,
            &cols_v2,
            &setup3.column_ids,
        )
        .unwrap();
    let snap3 = current_head(&pool, catalog_id).await;
    let v3: i64 =
        sqlx::query("SELECT schema_version FROM ducklake_snapshot WHERE snapshot_id = $1")
            .bind(snap3)
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
            "public",
            "users",
            setup_a.snapshot_id,
            WriteMode::Replace,
            setup_a.base_snapshot_id,
            &cols(),
            &setup_a.column_ids,
        )
        .unwrap();
    let setup_b = writer_b
        .begin_write_transaction("public", "orders", &cols(), WriteMode::Replace)
        .unwrap();
    writer_b
        .publish_snapshot(
            setup_b.table_id,
            "public",
            "orders",
            setup_b.snapshot_id,
            WriteMode::Replace,
            setup_b.base_snapshot_id,
            &cols(),
            &setup_b.column_ids,
        )
        .unwrap();

    // The committed schema ids are assigned at the commit (begin's setup.schema_id
    // is vestigial for a brand-new schema), so read each catalog's mapped schema
    // id directly. Two "public" rows, one per catalog, with different schema_ids.
    let schema_ids_a: Vec<i64> =
        sqlx::query("SELECT schema_id FROM ducklake_catalog_schema_map WHERE catalog_id = $1")
            .bind(cat_a)
            .fetch_all(&pool)
            .await
            .unwrap()
            .into_iter()
            .map(|r| r.try_get(0).unwrap())
            .collect();
    let schema_ids_b: Vec<i64> =
        sqlx::query("SELECT schema_id FROM ducklake_catalog_schema_map WHERE catalog_id = $1")
            .bind(cat_b)
            .fetch_all(&pool)
            .await
            .unwrap()
            .into_iter()
            .map(|r| r.try_get(0).unwrap())
            .collect();
    // Each catalog maps to exactly one schema, and they are distinct.
    assert_eq!(schema_ids_a.len(), 1);
    assert_eq!(schema_ids_b.len(), 1);
    assert_ne!(schema_ids_a[0], schema_ids_b[0]);

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
    // Committed snapshot ids are assigned AT each publish, so read each catalog's
    // head right after its own publish (begin's snapshot_id is vestigial now).
    // cat_a DDL (creates users)
    let a1 = wa
        .begin_write_transaction("public", "users", &cols(), WriteMode::Replace)
        .unwrap();
    wa.publish_snapshot(
        a1.table_id,
        "public",
        "users",
        a1.snapshot_id,
        WriteMode::Replace,
        a1.base_snapshot_id,
        &cols(),
        &a1.column_ids,
    )
    .unwrap();
    let a1_snap = current_head(&pool, cat_a).await;
    // cat_a DML (Replace, same schema)
    let a2 = wa
        .begin_write_transaction("public", "users", &cols(), WriteMode::Replace)
        .unwrap();
    wa.publish_snapshot(
        a2.table_id,
        "public",
        "users",
        a2.snapshot_id,
        WriteMode::Replace,
        a2.base_snapshot_id,
        &cols(),
        &a2.column_ids,
    )
    .unwrap();
    let a2_snap = current_head(&pool, cat_a).await;
    // cat_b DDL (creates orders) — happens in between cat_a's DDLs
    let b1 = wb
        .begin_write_transaction("public", "orders", &cols(), WriteMode::Replace)
        .unwrap();
    wb.publish_snapshot(
        b1.table_id,
        "public",
        "orders",
        b1.snapshot_id,
        WriteMode::Replace,
        b1.base_snapshot_id,
        &cols(),
        &b1.column_ids,
    )
    .unwrap();
    let b1_snap = current_head(&pool, cat_b).await;
    // cat_a DDL: adds age column
    let mut cols_v2 = cols();
    cols_v2.push(ColumnDef::new("age", "int32", true).unwrap());
    let a3 = wa
        .begin_write_transaction("public", "users", &cols_v2, WriteMode::Replace)
        .unwrap();
    wa.publish_snapshot(
        a3.table_id,
        "public",
        "users",
        a3.snapshot_id,
        WriteMode::Replace,
        a3.base_snapshot_id,
        &cols_v2,
        &a3.column_ids,
    )
    .unwrap();
    let a3_snap = current_head(&pool, cat_a).await;
    // cat_b DML
    let b2 = wb
        .begin_write_transaction("public", "orders", &cols(), WriteMode::Replace)
        .unwrap();
    wb.publish_snapshot(
        b2.table_id,
        "public",
        "orders",
        b2.snapshot_id,
        WriteMode::Replace,
        b2.base_snapshot_id,
        &cols(),
        &b2.column_ids,
    )
    .unwrap();
    let b2_snap = current_head(&pool, cat_b).await;

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

    assert_eq!(get_v(a1_snap).await, 1, "cat_a first DDL");
    assert_eq!(get_v(a2_snap).await, 1, "cat_a DML carries v1");
    assert_eq!(get_v(b1_snap).await, 1, "cat_b first DDL");
    assert_eq!(get_v(a3_snap).await, 2, "cat_a column-add DDL");
    assert_eq!(get_v(b2_snap).await, 1, "cat_b DML carries v1");
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
    w.publish_snapshot(
        s1.table_id,
        "public",
        "users",
        s1.snapshot_id,
        WriteMode::Replace,
        s1.base_snapshot_id,
        &cols(),
        &s1.column_ids,
    )
    .unwrap();
    let s2 = w
        .begin_write_transaction("public", "users", &cols(), WriteMode::Replace)
        .unwrap();
    w.publish_snapshot(
        s2.table_id,
        "public",
        "users",
        s2.snapshot_id,
        WriteMode::Replace,
        s2.base_snapshot_id,
        &cols(),
        &s2.column_ids,
    )
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

/// Regression for the "dormant rows" leak: a writer that has only begun (reserved
/// ids, uploaded nothing, NOT committed) must leave NO metadata visible at the
/// catalog head. Under the commit-time model `begin_write_transaction` inserts no
/// snapshot/schema/table/column rows at all, so a concurrent committed write on a
/// DIFFERENT table can advance the head without exposing the in-flight table.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn dormant_rows_invisible_until_publish() {
    use datafusion_ducklake::MulticatalogProvider;
    use datafusion_ducklake::metadata_provider::MetadataProvider;
    use datafusion_ducklake::metadata_writer::DataFileInfo;

    let (pool, _c) = spin_up_postgres().await.unwrap();
    let mgr = MulticatalogManager::new(pool.clone());
    let cat = mgr.create_catalog("pg_dormant").await.unwrap();
    let w = PostgresMetadataWriter::with_pool(pool.clone(), cat)
        .await
        .unwrap();
    w.set_data_path("/data").unwrap();

    // Writer A begins on T1 but never registers/publishes (in-flight upload).
    let a = w
        .begin_write_transaction("public", "t1", &cols(), WriteMode::Replace)
        .unwrap();

    // Writer B does a full begin + register on a DIFFERENT table T2, committing
    // and advancing the catalog head.
    let b = w
        .begin_write_transaction("public", "t2", &cols(), WriteMode::Replace)
        .unwrap();
    w.register_data_file(
        b.table_id,
        "public",
        "t2",
        b.snapshot_id,
        &DataFileInfo::new("t2.parquet", 1024, 5),
        WriteMode::Replace,
        b.base_snapshot_id,
        &cols(),
        &b.column_ids,
    )
    .unwrap();

    // At the new head, T1 is NOT visible — A's reserved ids inserted no rows.
    let provider = MulticatalogProvider::with_pool_and_id(pool.clone(), cat)
        .await
        .unwrap();
    let head = provider.get_current_snapshot().unwrap();
    let schema = provider
        .get_schema_by_name("public", head)
        .unwrap()
        .unwrap();
    let names: Vec<String> = provider
        .list_tables(schema.schema_id, head)
        .unwrap()
        .into_iter()
        .map(|t| t.table_name)
        .collect();
    assert_eq!(names, vec!["t2"], "only the committed table T2 is visible");
    assert!(
        provider
            .get_table_by_name(schema.schema_id, "t1", head)
            .unwrap()
            .is_none(),
        "in-flight table T1 must be invisible at the head"
    );

    // Dropping A (it never commits) must leave no orphan visible. The reserved
    // table id (a.table_id) inserted no row, so there is nothing to clean up.
    drop(a);
    let live_t1: i64 = sqlx::query(
        "SELECT COUNT(*) FROM ducklake_table WHERE schema_id = $1 AND table_name = 't1'",
    )
    .bind(schema.schema_id)
    .fetch_one(&pool)
    .await
    .unwrap()
    .try_get(0)
    .unwrap();
    assert_eq!(live_t1, 0, "abandoned begin left no table row");
    let head_after = provider.get_current_snapshot().unwrap();
    assert_eq!(head_after, head, "abandoning A did not advance the head");
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
            "public",
            "users",
            setup.snapshot_id,
            &file,
            WriteMode::Replace,
            setup.base_snapshot_id,
            &cols(),
            &setup.column_ids,
        )
        .unwrap()
        .snapshot_id;
    assert!(file_id > 0);

    let row = sqlx::query(
        "SELECT path, file_size_bytes, record_count, begin_snapshot
         FROM ducklake_data_file WHERE begin_snapshot = $1",
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
    assert_eq!(begin, file_id);
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
        "public",
        "users",
        s1.snapshot_id,
        &DataFileInfo::new("u.parquet", 1024, 10),
        WriteMode::Replace,
        s1.base_snapshot_id,
        &cols(),
        &s1.column_ids,
    )
    .unwrap();

    let mut cols_v2 = cols();
    cols_v2.push(ColumnDef::new("age", "int32", true).unwrap());
    let s_ddl = w
        .begin_write_transaction("public", "users", &cols_v2, WriteMode::Replace)
        .unwrap();
    w.publish_snapshot(
        s_ddl.table_id,
        "public",
        "users",
        s_ddl.snapshot_id,
        WriteMode::Replace,
        s_ddl.base_snapshot_id,
        &cols_v2,
        &s_ddl.column_ids,
    )
    .unwrap();

    let s_orders = w
        .begin_write_transaction("public", "orders", &cols(), WriteMode::Replace)
        .unwrap();
    w.register_data_file(
        s_orders.table_id,
        "public",
        "orders",
        s_orders.snapshot_id,
        &DataFileInfo::new("o.parquet", 2048, 20),
        WriteMode::Replace,
        s_orders.base_snapshot_id,
        &cols(),
        &s_orders.column_ids,
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
        "public",
        "users",
        sa.snapshot_id,
        &DataFileInfo::new("a.parquet", 1024, 10),
        WriteMode::Replace,
        sa.base_snapshot_id,
        &cols(),
        &sa.column_ids,
    )
    .unwrap();

    let sb = wb
        .begin_write_transaction("public", "orders", &cols(), WriteMode::Replace)
        .unwrap();
    wb.register_data_file(
        sb.table_id,
        "public",
        "orders",
        sb.snapshot_id,
        &DataFileInfo::new("b.parquet", 2048, 20),
        WriteMode::Replace,
        sb.base_snapshot_id,
        &cols(),
        &sb.column_ids,
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
        "public",
        "users",
        s.snapshot_id,
        &DataFileInfo::new("u.parquet", 1024, 10),
        WriteMode::Replace,
        s.base_snapshot_id,
        &cols(),
        &s.column_ids,
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
        "public",
        "users",
        s.snapshot_id,
        &DataFileInfo::new("u.parquet", 1024, 10),
        WriteMode::Replace,
        s.base_snapshot_id,
        &cols(),
        &s.column_ids,
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
        "public",
        "users",
        s.snapshot_id,
        &DataFileInfo::new("u.parquet", 1024, 10),
        WriteMode::Replace,
        s.base_snapshot_id,
        &cols(),
        &s.column_ids,
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
        "public",
        "users",
        s_users.snapshot_id,
        &DataFileInfo::new("u.parquet", 1024, 10),
        WriteMode::Replace,
        s_users.base_snapshot_id,
        &cols(),
        &s_users.column_ids,
    )
    .unwrap();

    let s_orders = w
        .begin_write_transaction("public", "orders", &cols(), WriteMode::Replace)
        .unwrap();
    w.register_data_file(
        s_orders.table_id,
        "public",
        "orders",
        s_orders.snapshot_id,
        &DataFileInfo::new("o.parquet", 2048, 20),
        WriteMode::Replace,
        s_orders.base_snapshot_id,
        &cols(),
        &s_orders.column_ids,
    )
    .unwrap();

    // Same-named table in a different schema; must also survive.
    let s_other_users = w
        .begin_write_transaction("analytics", "users", &cols(), WriteMode::Replace)
        .unwrap();
    w.register_data_file(
        s_other_users.table_id,
        "analytics",
        "users",
        s_other_users.snapshot_id,
        &DataFileInfo::new("au.parquet", 512, 5),
        WriteMode::Replace,
        s_other_users.base_snapshot_id,
        &cols(),
        &s_other_users.column_ids,
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
        "public",
        "users",
        s1.snapshot_id,
        &DataFileInfo::new("v1.parquet", 1024, 10),
        WriteMode::Replace,
        s1.base_snapshot_id,
        &cols(),
        &s1.column_ids,
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
        "public",
        "users",
        s2.snapshot_id,
        &DataFileInfo::new("v2.parquet", 2048, 20),
        WriteMode::Replace,
        s2.base_snapshot_id,
        &cols(),
        &s2.column_ids,
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
    let create_snap = w
        .register_data_file(
            s.table_id,
            "public",
            "users",
            s.snapshot_id,
            &DataFileInfo::new("u.parquet", 1024, 10),
            WriteMode::Replace,
            s.base_snapshot_id,
            &cols(),
            &s.column_ids,
        )
        .unwrap()
        .snapshot_id;

    let v_create: i64 =
        sqlx::query("SELECT schema_version FROM ducklake_snapshot WHERE snapshot_id = $1")
            .bind(create_snap)
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
        "public",
        "users",
        sa.snapshot_id,
        &DataFileInfo::new("a.parquet", 1024, 10),
        WriteMode::Replace,
        sa.base_snapshot_id,
        &cols(),
        &sa.column_ids,
    )
    .unwrap();

    let sb = wb
        .begin_write_transaction("public", "users", &cols(), WriteMode::Replace)
        .unwrap();
    wb.register_data_file(
        sb.table_id,
        "public",
        "users",
        sb.snapshot_id,
        &DataFileInfo::new("b.parquet", 2048, 20),
        WriteMode::Replace,
        sb.base_snapshot_id,
        &cols(),
        &sb.column_ids,
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
        "public",
        "users",
        setup.snapshot_id,
        &DataFileInfo::new("f1.parquet", 4096, 100),
        WriteMode::Replace,
        setup.base_snapshot_id,
        &cols(),
        &setup.column_ids,
    )
    .unwrap();
    w.register_data_file(
        setup.table_id,
        "public",
        "users",
        setup.snapshot_id,
        &DataFileInfo::new("f2.parquet", 2048, 50),
        WriteMode::Append,
        setup.base_snapshot_id,
        &cols(),
        &setup.column_ids,
    )
    .unwrap();
    w.register_data_file(
        setup.table_id,
        "public",
        "users",
        setup.snapshot_id,
        &DataFileInfo::new("f3.parquet", 8192, 200),
        WriteMode::Append,
        setup.base_snapshot_id,
        &cols(),
        &setup.column_ids,
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
        "public",
        "users",
        s1.snapshot_id,
        &DataFileInfo::new("g1.parquet", 1024, 5),
        WriteMode::Replace,
        s1.base_snapshot_id,
        &cols(),
        &s1.column_ids,
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
    w.publish_snapshot(
        s2.table_id,
        "public",
        "users",
        s2.snapshot_id,
        WriteMode::Replace,
        s2.base_snapshot_id,
        &cols(),
        &s2.column_ids,
    )
    .unwrap();
    let (rc2, next2, bytes2) = read_table_stats(&pool, s2.table_id).await;
    assert_eq!(rc2, 0, "record_count must reset on Replace");
    assert_eq!(next2, 5, "next_row_id must NOT reset on Replace");
    assert_eq!(bytes2, 0);

    // The first file of the new generation picks up at 5. The generation was
    // already published above, so this registration is additive (Append).
    w.register_data_file(
        s2.table_id,
        "public",
        "users",
        s2.snapshot_id,
        &DataFileInfo::new("g2.parquet", 2048, 2),
        WriteMode::Append,
        s2.base_snapshot_id,
        &cols(),
        &s2.column_ids,
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

    w.register_data_file(
        setup.table_id,
        "public",
        "users",
        setup.snapshot_id,
        &DataFileInfo::new("a.parquet", 50, 4),
        WriteMode::Replace,
        setup.base_snapshot_id,
        &cols(),
        &setup.column_ids,
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
    // register_data_file returns the committed snapshot id (assigned at commit).
    let s1 = writer
        .begin_write_transaction(schema, table, &cols(), WriteMode::Replace)
        .unwrap();
    let snap1 = writer
        .register_data_file(
            s1.table_id,
            schema,
            table,
            s1.snapshot_id,
            &DataFileInfo::new("f1.parquet", 100, 5),
            WriteMode::Replace,
            s1.base_snapshot_id,
            &cols(),
            &s1.column_ids,
        )
        .unwrap()
        .snapshot_id;
    let s2 = writer
        .begin_write_transaction(schema, table, &cols(), WriteMode::Replace)
        .unwrap();
    let snap2 = writer
        .register_data_file(
            s2.table_id,
            schema,
            table,
            s2.snapshot_id,
            &DataFileInfo::new("f2.parquet", 100, 5),
            WriteMode::Replace,
            s2.base_snapshot_id,
            &cols(),
            &s2.column_ids,
        )
        .unwrap()
        .snapshot_id;
    let s3 = writer
        .begin_write_transaction(schema, table, &cols(), WriteMode::Replace)
        .unwrap();
    let snap3 = writer
        .register_data_file(
            s3.table_id,
            schema,
            table,
            s3.snapshot_id,
            &DataFileInfo::new("f3.parquet", 100, 5),
            WriteMode::Replace,
            s3.base_snapshot_id,
            &cols(),
            &s3.column_ids,
        )
        .unwrap()
        .snapshot_id;
    (s1.table_id, snap1, snap2, snap3)
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
        "public",
        "t",
        s.snapshot_id,
        &DataFileInfo::new("f1.parquet", 100, 5),
        WriteMode::Replace,
        s.base_snapshot_id,
        &cols(),
        &s.column_ids,
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
    let create_snap = w
        .register_data_file(
            s.table_id,
            "public",
            "t",
            s.snapshot_id,
            &DataFileInfo::new("f1.parquet", 100, 5),
            WriteMode::Replace,
            s.base_snapshot_id,
            &cols(),
            &s.column_ids,
        )
        .unwrap()
        .snapshot_id;

    // Drop allocates a second snapshot; expire the first (the drop snapshot is kept).
    assert!(
        mgr.drop_table_in_catalog("pg_prod", "public", "t")
            .await
            .unwrap()
    );
    let expired = mgr
        .expire_snapshots_in_catalog("pg_prod", ExpireCriteria::Versions(vec![create_snap]))
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
    let a1_snap = wa
        .register_data_file(
            a1.table_id,
            "public",
            "t",
            a1.snapshot_id,
            &DataFileInfo::new("f1.parquet", 100, 5),
            WriteMode::Replace,
            a1.base_snapshot_id,
            &cols(),
            &a1.column_ids,
        )
        .unwrap()
        .snapshot_id;
    let b1 = wb
        .begin_write_transaction("public", "u", &cols(), WriteMode::Replace)
        .unwrap();
    let b1_snap = wb
        .register_data_file(
            b1.table_id,
            "public",
            "u",
            b1.snapshot_id,
            &DataFileInfo::new("g1.parquet", 100, 5),
            WriteMode::Replace,
            b1.base_snapshot_id,
            &cols(),
            &b1.column_ids,
        )
        .unwrap()
        .snapshot_id;
    let a2 = wa
        .begin_write_transaction("public", "t", &cols(), WriteMode::Replace)
        .unwrap();
    let a2_snap = wa
        .register_data_file(
            a2.table_id,
            "public",
            "t",
            a2.snapshot_id,
            &DataFileInfo::new("f2.parquet", 100, 5),
            WriteMode::Replace,
            a2.base_snapshot_id,
            &cols(),
            &a2.column_ids,
        )
        .unwrap()
        .snapshot_id;
    assert!(
        b1_snap > a1_snap && b1_snap < a2_snap,
        "test setup: B's snapshot must fall inside A/f1's lifetime range"
    );

    let expired = mgr
        .expire_snapshots_in_catalog("cat_a", ExpireCriteria::Versions(vec![a1_snap]))
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
        .bind(b1_snap)
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
        "public",
        "t",
        a1.snapshot_id,
        &DataFileInfo::new("f1.parquet", 100, 5),
        WriteMode::Replace,
        a1.base_snapshot_id,
        &cols(),
        &a1.column_ids,
    )
    .unwrap();
    let a2 = wa
        .begin_write_transaction("public", "t", &cols(), WriteMode::Replace)
        .unwrap();
    wa.register_data_file(
        a2.table_id,
        "public",
        "t",
        a2.snapshot_id,
        &DataFileInfo::new("f2.parquet", 100, 5),
        WriteMode::Replace,
        a2.base_snapshot_id,
        &cols(),
        &a2.column_ids,
    )
    .unwrap();
    let b1 = wb
        .begin_write_transaction("public", "u", &cols(), WriteMode::Replace)
        .unwrap();
    wb.register_data_file(
        b1.table_id,
        "public",
        "u",
        b1.snapshot_id,
        &DataFileInfo::new("g1.parquet", 100, 5),
        WriteMode::Replace,
        b1.base_snapshot_id,
        &cols(),
        &b1.column_ids,
    )
    .unwrap();
    let b2 = wb
        .begin_write_transaction("public", "u", &cols(), WriteMode::Replace)
        .unwrap();
    wb.register_data_file(
        b2.table_id,
        "public",
        "u",
        b2.snapshot_id,
        &DataFileInfo::new("g2.parquet", 100, 5),
        WriteMode::Replace,
        b2.base_snapshot_id,
        &cols(),
        &b2.column_ids,
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
        "public",
        "t",
        s.snapshot_id,
        &DataFileInfo::new("f1.parquet", 100, 5),
        WriteMode::Replace,
        s.base_snapshot_id,
        &cols(),
        &s.column_ids,
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
        "public",
        "t",
        sa.snapshot_id,
        &DataFileInfo::new("a.parquet", 100, 5),
        WriteMode::Replace,
        sa.base_snapshot_id,
        &cols(),
        &sa.column_ids,
    )
    .unwrap();
    let sb = wb
        .begin_write_transaction("public", "u", &cols(), WriteMode::Replace)
        .unwrap();
    wb.register_data_file(
        sb.table_id,
        "public",
        "u",
        sb.snapshot_id,
        &DataFileInfo::new("b.parquet", 100, 5),
        WriteMode::Replace,
        sb.base_snapshot_id,
        &cols(),
        &sb.column_ids,
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
        "public",
        "t",
        s.snapshot_id,
        &DataFileInfo::new("ref.parquet", 100, 5),
        WriteMode::Replace,
        s.base_snapshot_id,
        &cols(),
        &s.column_ids,
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

/// End-to-end VALUE round-trip on the multicatalog write path.
///
/// Every other test in this file asserts on catalog rows / `record_count` sums.
/// None of them read the actual data back, so a field-id drift or NULL-fill
/// corruption (the exact failure modes the commit-time rework guards against)
/// could pass them all. This test writes real parquet via `DuckLakeTableWriter`
/// and reads the VALUES back through `DuckLakeCatalog` + DataFusion SQL across
/// three generations:
///   1. initial Replace  -> values round-trip (not NULL)
///   2. second  Replace  -> only the NEW generation is visible (no union/NULL)
///   3. Append           -> rows add to the current generation (values intact)
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn multicatalog_write_read_value_roundtrip() {
    use arrow::array::{Array, Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use datafusion::prelude::SessionContext;
    use datafusion_ducklake::{DuckLakeCatalog, MulticatalogProvider};
    use object_store::ObjectStore;
    use object_store::local::LocalFileSystem;
    use std::sync::Arc;
    use tempfile::TempDir;

    let (pool, _c) = spin_up_postgres().await.unwrap();
    let mgr = MulticatalogManager::new(pool.clone());
    let cat = mgr.create_catalog("rt").await.unwrap();
    let writer = PostgresMetadataWriter::with_pool(pool.clone(), cat)
        .await
        .unwrap();
    let temp = TempDir::new().unwrap();
    let data_path = temp.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();
    writer.set_data_path(data_path.to_str().unwrap()).unwrap();

    let store: Arc<dyn ObjectStore> = Arc::new(LocalFileSystem::new());
    let tw = DuckLakeTableWriter::new(Arc::new(writer), store).unwrap();

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, true),
    ]));
    let mk = |ids: Vec<i64>, names: Vec<&'static str>| {
        RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(ids)) as Arc<dyn arrow::array::Array>,
                Arc::new(StringArray::from(
                    names.into_iter().map(Some).collect::<Vec<_>>(),
                )),
            ],
        )
        .unwrap()
    };

    // Read every row back through the real read path (fresh catalog bound to the
    // current head each call) and materialise (id, name) tuples.
    let read_rows = |pool: PgPool| async move {
        let provider = MulticatalogProvider::with_pool(pool, "rt").await.unwrap();
        let catalog = DuckLakeCatalog::new(provider).unwrap();
        let ctx = SessionContext::new();
        ctx.register_catalog("rt", Arc::new(catalog));
        let batches = ctx
            .sql("SELECT id, name FROM rt.public.t ORDER BY id")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let mut out: Vec<(i64, Option<String>)> = Vec::new();
        for b in &batches {
            let ids = b
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("id column is Int64");
            let names = b
                .column(1)
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("name column is Utf8");
            for i in 0..b.num_rows() {
                let name = if names.is_null(i) {
                    None
                } else {
                    Some(names.value(i).to_string())
                };
                out.push((ids.value(i), name));
            }
        }
        out
    };

    // 1. Initial write (Replace). Values must come back intact, not NULL.
    tw.write_table("public", "t", &[mk(vec![1, 2, 3], vec!["a", "b", "c"])])
        .await
        .unwrap();
    assert_eq!(
        read_rows(pool.clone()).await,
        vec![(1, Some("a".into())), (2, Some("b".into())), (3, Some("c".into())),],
        "initial generation values must round-trip (not NULL / not empty)",
    );

    // 2. Replace with a new generation. ONLY the new rows are visible — the old
    //    files are retired, not unioned, and the new values are intact.
    tw.write_table("public", "t", &[mk(vec![10, 20], vec!["x", "y"])])
        .await
        .unwrap();
    assert_eq!(
        read_rows(pool.clone()).await,
        vec![(10, Some("x".into())), (20, Some("y".into()))],
        "Replace must expose only the new generation (no union, no NULL fill)",
    );

    // 3. Append to the current generation. Rows add; existing values stay intact.
    tw.append_table("public", "t", &[mk(vec![30], vec!["z"])])
        .await
        .unwrap();
    assert_eq!(
        read_rows(pool.clone()).await,
        vec![(10, Some("x".into())), (20, Some("y".into())), (30, Some("z".into())),],
        "Append must add to the current generation with values intact",
    );
}

/// Postgres-multicatalog counterpart of the SQLite bug-C proof: promote a column
/// int32 -> int64, append a value BEYOND the int32 range under the widened type,
/// and read back across the old int32 file + the new int64 file — old values
/// up-cast and the beyond-range value survives. Exercises the composite-PK base
/// model, promote_column_type, and cast-on-read on the priority backend.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn multicatalog_promote_widens_column_and_old_values_read_back() {
    use arrow::array::{Array, Int32Array, Int64Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use datafusion::prelude::SessionContext;
    use datafusion_ducklake::{DuckLakeCatalog, MetadataWriter, MulticatalogProvider};
    use object_store::ObjectStore;
    use object_store::local::LocalFileSystem;
    use std::sync::Arc;
    use tempfile::TempDir;

    let (pool, _c) = spin_up_postgres().await.unwrap();
    let mgr = MulticatalogManager::new(pool.clone());
    let cat = mgr.create_catalog("promote").await.unwrap();
    // One writer, reused across write/promote/append via Arc clones.
    let writer: Arc<dyn MetadataWriter> = Arc::new(
        PostgresMetadataWriter::with_pool(pool.clone(), cat)
            .await
            .unwrap(),
    );
    let temp = TempDir::new().unwrap();
    let data_path = temp.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();
    writer.set_data_path(data_path.to_str().unwrap()).unwrap();
    let store: Arc<dyn ObjectStore> = Arc::new(LocalFileSystem::new());

    // t(id int32) = [1, 2, 3] — physically int32 Parquet.
    let i32_schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let b32 =
        RecordBatch::try_new(i32_schema, vec![Arc::new(Int32Array::from(vec![1, 2, 3]))]).unwrap();
    let res = DuckLakeTableWriter::new(Arc::clone(&writer), Arc::clone(&store))
        .unwrap()
        .write_table("public", "t", &[b32])
        .await
        .unwrap();

    // Promote id int32 -> int64 (schema evolution; no data rewritten).
    writer
        .promote_column_type(res.table_id, "id", "int64")
        .unwrap();

    // Append a value BEYOND the int32 range under the widened type (bug C's value).
    let beyond_i32 = 5_000_000_000_i64;
    let i64_schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
    let b64 = RecordBatch::try_new(
        i64_schema,
        vec![Arc::new(Int64Array::from(vec![beyond_i32]))],
    )
    .unwrap();
    DuckLakeTableWriter::new(Arc::clone(&writer), Arc::clone(&store))
        .unwrap()
        .append_table("public", "t", &[b64])
        .await
        .unwrap();

    // Read back through the multicatalog provider: column is Int64, old file
    // up-casts, beyond-range value intact.
    let provider = MulticatalogProvider::with_pool(pool.clone(), "promote")
        .await
        .unwrap();
    let catalog = DuckLakeCatalog::new(provider).unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("promote", Arc::new(catalog));
    let batches = ctx
        .sql("SELECT id FROM promote.public.t ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(
        batches[0].schema().field(0).data_type(),
        &DataType::Int64,
        "promoted column must read as Int64"
    );
    let mut got: Vec<i64> = Vec::new();
    for b in &batches {
        let ids = b
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("id column should be Int64 after promote");
        got.extend(ids.values().iter().copied());
    }
    got.sort();
    assert_eq!(
        got,
        vec![1, 2, 3, beyond_i32],
        "old int32 file up-casts AND the beyond-i32 value survives across the promote"
    );
}

/// Safe-to-migrate (Postgres): a multicatalog store whose `ducklake_column` is in
/// the LEGACY single-row `column_id` PK shape (as a pre-change version wrote it),
/// holding real data, must convert to the composite PK on the next bootstrap and
/// keep its data — then support a promote. Guards the bug where the migration was
/// wired only into `initialize_schema`, not the multicatalog bootstrap that
/// runtimedb actually calls.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn multicatalog_migrates_legacy_single_pk_to_composite_with_data() {
    use arrow::array::{Array, Int32Array, Int64Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use datafusion::prelude::SessionContext;
    use datafusion_ducklake::{DuckLakeCatalog, MulticatalogProvider};
    use object_store::ObjectStore;
    use object_store::local::LocalFileSystem;
    use std::sync::Arc;
    use tempfile::TempDir;

    let (pool, _c) = spin_up_postgres().await.unwrap();
    let mgr = MulticatalogManager::new(pool.clone());
    let cat = mgr.create_catalog("legacy").await.unwrap(); // catalog_id is i64 (Copy) — reusable

    // Write real data t(id int32) = [1,2,3] (fresh DDL → composite PK).
    {
        let writer = PostgresMetadataWriter::with_pool(pool.clone(), cat)
            .await
            .unwrap();
        let temp = TempDir::new().unwrap();
        let data_path = temp.path().join("data");
        std::fs::create_dir_all(&data_path).unwrap();
        writer.set_data_path(data_path.to_str().unwrap()).unwrap();
        let store: Arc<dyn ObjectStore> = Arc::new(LocalFileSystem::new());
        let s32 = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let b32 =
            RecordBatch::try_new(s32, vec![Arc::new(Int32Array::from(vec![1, 2, 3]))]).unwrap();
        // Keep the temp dir alive for the whole test by leaking it (data files must
        // outlive the read below).
        std::mem::forget(temp);
        DuckLakeTableWriter::new(Arc::new(writer), store)
            .unwrap()
            .write_table("public", "t", &[b32])
            .await
            .unwrap();
    }

    // DOWNGRADE ducklake_column to the legacy single-row column_id PK (simulating
    // a catalog written by a pre-change datafusion-ducklake version).
    sqlx::query("DROP INDEX IF EXISTS idx_ducklake_column_live")
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query("ALTER TABLE ducklake_column DROP CONSTRAINT ducklake_column_pkey")
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query("ALTER TABLE ducklake_column ADD PRIMARY KEY (column_id)")
        .execute(&pool)
        .await
        .unwrap();
    let pk_len: i32 = sqlx::query_scalar(
        "SELECT array_length(conkey, 1) FROM pg_constraint
         WHERE conrelid = 'ducklake_column'::regclass AND contype = 'p'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(pk_len, 1, "downgrade produced the legacy single-column PK");

    // Re-run the multicatalog bootstrap → the migration converts the PK.
    initialize_multicatalog_schema(&pool).await.unwrap();
    let pk_len_after: i32 = sqlx::query_scalar(
        "SELECT array_length(conkey, 1) FROM pg_constraint
         WHERE conrelid = 'ducklake_column'::regclass AND contype = 'p'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(
        pk_len_after, 3,
        "bootstrap migrated the legacy PK to the composite PK"
    );

    // Data preserved — reads through the provider return the original values.
    let read = |pool: PgPool| async move {
        let provider = MulticatalogProvider::with_pool(pool, "legacy")
            .await
            .unwrap();
        let catalog = DuckLakeCatalog::new(provider).unwrap();
        let ctx = SessionContext::new();
        ctx.register_catalog("legacy", Arc::new(catalog));
        ctx.sql("SELECT id FROM legacy.public.t ORDER BY id")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap()
    };
    let batches = read(pool.clone()).await;
    let ids = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    assert_eq!(
        ids.values(),
        &[1, 2, 3],
        "data survived the Postgres migration"
    );

    // The migrated catalog now supports promote (two versioned rows, same id).
    let table_id: i64 =
        sqlx::query_scalar("SELECT table_id FROM ducklake_table WHERE table_name = 't'")
            .fetch_one(&pool)
            .await
            .unwrap();
    let writer2 = PostgresMetadataWriter::with_pool(pool.clone(), cat)
        .await
        .unwrap();
    writer2
        .promote_column_type(table_id, "id", "int64")
        .unwrap();
    let n_versions: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM ducklake_column WHERE table_id = $1 AND column_name = 'id'",
    )
    .bind(table_id)
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(
        n_versions, 2,
        "promote on the migrated catalog leaves two versioned rows"
    );

    let batches2 = read(pool.clone()).await;
    assert_eq!(batches2[0].schema().field(0).data_type(), &DataType::Int64);
    let ids2 = batches2[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(
        ids2.values(),
        &[1, 2, 3],
        "post-migration promote widens + reads intact"
    );
}

/// §5 on Postgres: both Replace and Append must REJECT a column type change
/// (schema evolution must go through promote_column_type, not a data write).
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn multicatalog_replace_and_append_reject_type_change() {
    use arrow::array::{Int32Array, Int64Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use object_store::ObjectStore;
    use object_store::local::LocalFileSystem;
    use std::sync::Arc;
    use tempfile::TempDir;

    let (pool, _c) = spin_up_postgres().await.unwrap();
    let mgr = MulticatalogManager::new(pool.clone());
    let cat = mgr.create_catalog("reject").await.unwrap();
    let temp = TempDir::new().unwrap();
    let data_path = temp.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();
    let store: Arc<dyn ObjectStore> = Arc::new(LocalFileSystem::new());

    let s32 = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let s64 = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
    let b32 = RecordBatch::try_new(s32, vec![Arc::new(Int32Array::from(vec![1, 2, 3]))]).unwrap();

    // Build a fresh table writer (the writer Arc is consumed per call).
    let tw = |store: Arc<dyn ObjectStore>, dp: String| {
        let pool = pool.clone();
        async move {
            let w = PostgresMetadataWriter::with_pool(pool, cat).await.unwrap();
            w.set_data_path(&dp).unwrap();
            DuckLakeTableWriter::new(Arc::new(w), store).unwrap()
        }
    };
    let dp = data_path.to_str().unwrap().to_string();

    // Create t(id int32).
    tw(Arc::clone(&store), dp.clone())
        .await
        .write_table("public", "t", &[b32])
        .await
        .unwrap();

    // Replace with id int64 → rejected.
    let b64a = RecordBatch::try_new(
        s64.clone(),
        vec![Arc::new(Int64Array::from(vec![9_999_999_999]))],
    )
    .unwrap();
    let replace_res = tw(Arc::clone(&store), dp.clone())
        .await
        .write_table("public", "t", &[b64a])
        .await;
    assert!(
        replace_res.is_err(),
        "Replace with a type change must be rejected on Postgres"
    );

    // Append with id int64 → rejected.
    let b64b = RecordBatch::try_new(s64, vec![Arc::new(Int64Array::from(vec![4, 5]))]).unwrap();
    let append_res = tw(Arc::clone(&store), dp.clone())
        .await
        .append_table("public", "t", &[b64b])
        .await;
    assert!(
        append_res.is_err(),
        "Append with a type change must be rejected on Postgres"
    );
}

/// Phase C on Postgres: a promote leaves the multicatalog `ducklake_column` in the
/// upstream versioned shape — two rows sharing the same `column_id`, old retired +
/// new live with the widened type.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn multicatalog_promote_leaves_two_versioned_rows() {
    use arrow::array::Int32Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use object_store::ObjectStore;
    use object_store::local::LocalFileSystem;
    use std::sync::Arc;
    use tempfile::TempDir;

    let (pool, _c) = spin_up_postgres().await.unwrap();
    let mgr = MulticatalogManager::new(pool.clone());
    let cat = mgr.create_catalog("tworows").await.unwrap();
    let temp = TempDir::new().unwrap();
    let data_path = temp.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();
    let store: Arc<dyn ObjectStore> = Arc::new(LocalFileSystem::new());

    let writer = PostgresMetadataWriter::with_pool(pool.clone(), cat)
        .await
        .unwrap();
    writer.set_data_path(data_path.to_str().unwrap()).unwrap();
    let s32 = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let b32 = RecordBatch::try_new(s32, vec![Arc::new(Int32Array::from(vec![1, 2, 3]))]).unwrap();
    let res = DuckLakeTableWriter::new(Arc::new(writer), Arc::clone(&store))
        .unwrap()
        .write_table("public", "t", &[b32])
        .await
        .unwrap();

    let promoter = PostgresMetadataWriter::with_pool(pool.clone(), cat)
        .await
        .unwrap();
    promoter
        .promote_column_type(res.table_id, "id", "int64")
        .unwrap();

    let rows = sqlx::query(
        "SELECT column_id, column_type, end_snapshot FROM ducklake_column
         WHERE table_id = $1 AND column_name = 'id' ORDER BY begin_snapshot",
    )
    .bind(res.table_id)
    .fetch_all(&pool)
    .await
    .unwrap();
    assert_eq!(rows.len(), 2, "promote must leave two versioned rows");
    let cid0: i64 = rows[0].try_get("column_id").unwrap();
    let cid1: i64 = rows[1].try_get("column_id").unwrap();
    let t0: String = rows[0].try_get("column_type").unwrap();
    let t1: String = rows[1].try_get("column_type").unwrap();
    let e0: Option<i64> = rows[0].try_get("end_snapshot").unwrap();
    let e1: Option<i64> = rows[1].try_get("end_snapshot").unwrap();
    assert_eq!(cid0, cid1, "both versions share the same column_id");
    assert_eq!(t0, "int32");
    assert!(e0.is_some(), "old version retired");
    assert_eq!(t1, "int64");
    assert!(e1.is_none(), "new version live");
}

/// Concurrency: two promotes fired at the SAME column at once must not corrupt —
/// exactly one wins, the catalog ends with exactly two versioned rows and exactly
/// one live version (no double-promote, no two-live-rows). Exercises lock_catalog
/// + the partial unique index under real multi-connection contention.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn multicatalog_concurrent_promotes_one_wins_no_corruption() {
    use arrow::array::Int32Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use object_store::ObjectStore;
    use object_store::local::LocalFileSystem;
    use std::sync::Arc;
    use tempfile::TempDir;

    let (pool, _c) = spin_up_postgres().await.unwrap();
    let mgr = MulticatalogManager::new(pool.clone());
    let cat = mgr.create_catalog("race").await.unwrap();
    let temp = TempDir::new().unwrap();
    let data_path = temp.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();
    let store: Arc<dyn ObjectStore> = Arc::new(LocalFileSystem::new());

    let writer = PostgresMetadataWriter::with_pool(pool.clone(), cat)
        .await
        .unwrap();
    writer.set_data_path(data_path.to_str().unwrap()).unwrap();
    let s32 = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let b32 = RecordBatch::try_new(s32, vec![Arc::new(Int32Array::from(vec![1, 2, 3]))]).unwrap();
    let res = DuckLakeTableWriter::new(Arc::new(writer), Arc::clone(&store))
        .unwrap()
        .write_table("public", "t", &[b32])
        .await
        .unwrap();
    let tid = res.table_id;

    // Two independent writers (separate connections) promote the same column at once.
    let w1 = PostgresMetadataWriter::with_pool(pool.clone(), cat)
        .await
        .unwrap();
    let w2 = PostgresMetadataWriter::with_pool(pool.clone(), cat)
        .await
        .unwrap();
    let h1 = tokio::task::spawn_blocking(move || w1.promote_column_type(tid, "id", "int64"));
    let h2 = tokio::task::spawn_blocking(move || w2.promote_column_type(tid, "id", "int64"));
    let (r1, r2) = tokio::join!(h1, h2);
    let r1 = r1.unwrap();
    let r2 = r2.unwrap();

    assert!(
        r1.is_ok() ^ r2.is_ok(),
        "exactly one concurrent promote must succeed; got r1={r1:?} r2={r2:?}"
    );

    let total: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM ducklake_column WHERE table_id = $1 AND column_name = 'id'",
    )
    .bind(tid)
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(
        total, 2,
        "no double-promote: exactly two versioned rows, got {total}"
    );

    let live: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM ducklake_column
         WHERE table_id = $1 AND column_name = 'id' AND end_snapshot IS NULL",
    )
    .bind(tid)
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(
        live, 1,
        "exactly one live version after concurrent promotes, got {live}"
    );
}

/// §14 E2: `schema_version` advances on a schema change (promote) but NOT on a
/// data write (Append) — the spec-native "schema changed" signal consumers rely on.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn multicatalog_schema_version_advances_on_promote_not_data_write() {
    use arrow::array::Int32Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use object_store::ObjectStore;
    use object_store::local::LocalFileSystem;
    use std::sync::Arc;
    use tempfile::TempDir;

    async fn head_sv(pool: &PgPool, cat: i64) -> i64 {
        sqlx::query_scalar(
            "SELECT s.schema_version FROM ducklake_snapshot s
             JOIN ducklake_catalog_snapshot_map m ON m.snapshot_id = s.snapshot_id
             WHERE m.catalog_id = $1 ORDER BY s.snapshot_id DESC LIMIT 1",
        )
        .bind(cat)
        .fetch_one(pool)
        .await
        .unwrap()
    }

    let (pool, _c) = spin_up_postgres().await.unwrap();
    let mgr = MulticatalogManager::new(pool.clone());
    let cat = mgr.create_catalog("schemaver").await.unwrap();
    let temp = TempDir::new().unwrap();
    let data_path = temp.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();
    let store: Arc<dyn ObjectStore> = Arc::new(LocalFileSystem::new());
    let dp = data_path.to_str().unwrap().to_string();
    let s32 = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let mk = |v: Vec<i32>| {
        RecordBatch::try_new(s32.clone(), vec![Arc::new(Int32Array::from(v))]).unwrap()
    };

    // Create table (DDL).
    let w0 = PostgresMetadataWriter::with_pool(pool.clone(), cat)
        .await
        .unwrap();
    w0.set_data_path(&dp).unwrap();
    let res = DuckLakeTableWriter::new(Arc::new(w0), Arc::clone(&store))
        .unwrap()
        .write_table("public", "t", &[mk(vec![1, 2, 3])])
        .await
        .unwrap();
    let sv_create = head_sv(&pool, cat).await;

    // Append (data write, same schema) — must NOT bump schema_version.
    let w1 = PostgresMetadataWriter::with_pool(pool.clone(), cat)
        .await
        .unwrap();
    w1.set_data_path(&dp).unwrap();
    DuckLakeTableWriter::new(Arc::new(w1), Arc::clone(&store))
        .unwrap()
        .append_table("public", "t", &[mk(vec![4, 5])])
        .await
        .unwrap();
    let sv_append = head_sv(&pool, cat).await;
    assert_eq!(
        sv_append, sv_create,
        "a data write (Append) must not bump schema_version"
    );

    // Promote (schema change) — MUST bump schema_version.
    let w2 = PostgresMetadataWriter::with_pool(pool.clone(), cat)
        .await
        .unwrap();
    w2.promote_column_type(res.table_id, "id", "int64").unwrap();
    let sv_promote = head_sv(&pool, cat).await;
    assert!(
        sv_promote > sv_append,
        "a promote must bump schema_version ({sv_append} -> {sv_promote})"
    );
}

/// MAJOR-1 regression: an Append that BEGINS pre-promote (staged int32) and
/// COMMITS after a promote (column now int64) is a benign data write — it must
/// NOT be misclassified as DDL, so schema_version must stay put and no extra
/// ducklake_schema_versions row is written. Values still round-trip.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn multicatalog_append_racing_promote_does_not_bump_schema_version() {
    use arrow::array::{Array, Int32Array, Int64Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use datafusion::prelude::SessionContext;
    use datafusion_ducklake::{DuckLakeCatalog, MulticatalogProvider};
    use object_store::ObjectStore;
    use object_store::local::LocalFileSystem;
    use std::sync::Arc;
    use tempfile::TempDir;

    async fn head_sv(pool: &PgPool, cat: i64) -> i64 {
        sqlx::query_scalar(
            "SELECT s.schema_version FROM ducklake_snapshot s
             JOIN ducklake_catalog_snapshot_map m ON m.snapshot_id = s.snapshot_id
             WHERE m.catalog_id = $1 ORDER BY s.snapshot_id DESC LIMIT 1",
        )
        .bind(cat)
        .fetch_one(pool)
        .await
        .unwrap()
    }

    let (pool, _c) = spin_up_postgres().await.unwrap();
    let mgr = MulticatalogManager::new(pool.clone());
    let cat = mgr.create_catalog("racebump").await.unwrap();
    let temp = TempDir::new().unwrap();
    let data_path = temp.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();
    let store: Arc<dyn ObjectStore> = Arc::new(LocalFileSystem::new());
    let dp = data_path.to_str().unwrap().to_string();
    let s32 = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));

    // t(id int32) = [1,2,3]
    let w0 = PostgresMetadataWriter::with_pool(pool.clone(), cat)
        .await
        .unwrap();
    w0.set_data_path(&dp).unwrap();
    let res = DuckLakeTableWriter::new(Arc::new(w0), Arc::clone(&store))
        .unwrap()
        .write_table(
            "public",
            "t",
            &[RecordBatch::try_new(s32.clone(), vec![Arc::new(Int32Array::from(vec![1, 2, 3]))])
                .unwrap()],
        )
        .await
        .unwrap();

    // Begin an Append session under the (current) int32 schema and stage a batch.
    let wa = PostgresMetadataWriter::with_pool(pool.clone(), cat)
        .await
        .unwrap();
    wa.set_data_path(&dp).unwrap();
    let tw = DuckLakeTableWriter::new(Arc::new(wa), Arc::clone(&store)).unwrap();
    let mut session = tw
        .begin_write("public", "t", s32.as_ref(), WriteMode::Append)
        .unwrap();
    session
        .write_batch(
            &RecordBatch::try_new(s32.clone(), vec![Arc::new(Int32Array::from(vec![4, 5]))])
                .unwrap(),
        )
        .unwrap();

    // A promote commits in between (int32 -> int64): bumps schema_version.
    let wp = PostgresMetadataWriter::with_pool(pool.clone(), cat)
        .await
        .unwrap();
    wp.promote_column_type(res.table_id, "id", "int64").unwrap();
    let sv_after_promote = head_sv(&pool, cat).await;
    let ledger_after_promote: i64 =
        sqlx::query_scalar("SELECT count(*) FROM ducklake_schema_versions WHERE table_id = $1")
            .bind(res.table_id)
            .fetch_one(&pool)
            .await
            .unwrap();

    // Finish the Append AFTER the promote — must NOT bump schema_version.
    session.finish().await.unwrap();
    let sv_after_append = head_sv(&pool, cat).await;
    assert_eq!(
        sv_after_append, sv_after_promote,
        "benign Append racing a promote must NOT bump schema_version ({sv_after_promote} -> {sv_after_append})"
    );

    // The racing Append must NOT add a ducklake_schema_versions ledger row (it's a
    // data write, not DDL — only table-create + the promote are legitimate rows).
    let ledger_after_append: i64 =
        sqlx::query_scalar("SELECT count(*) FROM ducklake_schema_versions WHERE table_id = $1")
            .bind(res.table_id)
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(
        ledger_after_append, ledger_after_promote,
        "the racing Append must not add a schema_versions ledger row"
    );

    // Values round-trip (old + appended, all int64).
    let provider = MulticatalogProvider::with_pool(pool.clone(), "racebump")
        .await
        .unwrap();
    let catalog = DuckLakeCatalog::new(provider).unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("racebump", Arc::new(catalog));
    let batches = ctx
        .sql("SELECT id FROM racebump.public.t ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let mut got: Vec<i64> = Vec::new();
    for b in &batches {
        let ids = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        got.extend(ids.values().iter().copied());
    }
    got.sort();
    assert_eq!(
        got,
        vec![1, 2, 3, 4, 5],
        "all rows present and correct after the race"
    );
}
