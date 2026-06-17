//! Side-by-side comparable demo of the new maintenance flow on the single-catalog
//! (SQLite + LocalFS) write path.
//!
//! Mirrors the scenario in `examples/maintenance_demo.sql` (which drives the same
//! logical sequence through the official DuckDB+DuckLake extension), so the two
//! outputs can be lined up step-by-step.
//!
//! Scenario:
//!   snap 1   CREATE TABLE main.t(id int64, name varchar)   -- DDL
//!   snap 2   write data file f1                            -- INSERT
//!   snap 3   Replace: end-snapshot f1, write f2            -- "DELETE FROM t" + reinsert
//!   snap 4   DROP TABLE main.t                              -- tombstone
//!   expire   versions [2, 3]                                -- snap 4 is most-recent, kept
//!   cleanup  cleanup_all                                    -- physically delete scheduled
//!
//! Run with:
//!     cargo run --no-default-features --features write-sqlite \
//!         --example maintenance_demo

use std::path::Path;
use std::sync::Arc;

use object_store::ObjectStore;
use object_store::local::LocalFileSystem;
use sqlx::Row;
use sqlx::sqlite::SqlitePool;

use datafusion_ducklake::SqliteMetadataWriter;
use datafusion_ducklake::maintenance::{CleanupCriteria, ExpireCriteria, cleanup_old_files_sqlite};
use datafusion_ducklake::metadata_writer::{ColumnDef, DataFileInfo, MetadataWriter, WriteMode};

#[tokio::main(flavor = "multi_thread")]
async fn main() -> anyhow::Result<()> {
    // --- temp catalog + data path ----------------------------------------------------
    let temp = tempfile::TempDir::new()?;
    let db_path = temp.path().join("catalog.db");
    let data_path = temp.path().join("data");
    std::fs::create_dir_all(&data_path)?;
    let conn_str = format!("sqlite:{}?mode=rwc", db_path.display());

    let writer = SqliteMetadataWriter::new_with_init(&conn_str).await?;
    writer.set_data_path(data_path.to_str().unwrap())?;
    let pool = SqlitePool::connect(&conn_str).await?;

    println!("data_path = {}\n", data_path.display());

    // --- snap 1: CREATE TABLE (DDL only — no files yet) ------------------------------
    // begin_write_transaction allocates a snapshot and creates the schema/table/columns.
    // We don't register a data file at this snapshot, so it's purely DDL.
    let s1 = writer.begin_write_transaction(
        "main",
        "t",
        &[ColumnDef::new("id", "int64", false)?, ColumnDef::new("name", "varchar", true)?],
        WriteMode::Replace,
    )?;
    // CREATE TABLE registers no data file; publish the (empty) snapshot so it
    // becomes the head — begin_write_transaction now only RESERVES the id.
    writer.publish_snapshot(
        s1.table_id,
        s1.snapshot_id,
        WriteMode::Replace,
        &cols(),
        &s1.column_ids,
    )?;
    print_state(&pool, &data_path, "Step 1 — CREATE TABLE main.t").await?;

    // --- snap 2: write data file f1 ("INSERT") ---------------------------------------
    let s2 = writer.begin_write_transaction(
        "main",
        "t",
        &cols(),
        WriteMode::Replace, // Replace at snap2 ends nothing (nothing live), then writes f1.
    )?;
    writer.register_data_file(
        s2.table_id,
        s2.snapshot_id,
        &DataFileInfo::new("f1.parquet", 100, 5),
        WriteMode::Replace,
        &cols(),
        &s2.column_ids,
    )?;
    touch_file(&data_path, "main", "t", "f1.parquet")?;
    print_state(&pool, &data_path, "Step 2 — write f1 (INSERT)").await?;

    // --- snap 3: Replace — end-snapshot f1, write f2 ("DELETE FROM t" + reinsert) ----
    let s3 = writer.begin_write_transaction("main", "t", &cols(), WriteMode::Replace)?;
    writer.register_data_file(
        s3.table_id,
        s3.snapshot_id,
        &DataFileInfo::new("f2.parquet", 100, 5),
        WriteMode::Replace,
        &cols(),
        &s3.column_ids,
    )?;
    touch_file(&data_path, "main", "t", "f2.parquet")?;
    print_state(&pool, &data_path, "Step 3 — Replace: end f1, write f2").await?;

    // --- snap 4: DROP TABLE ---------------------------------------------------------
    let dropped = writer.drop_table("main", "t")?;
    println!("drop_table -> {dropped}");
    print_state(&pool, &data_path, "Step 4 — DROP TABLE main.t").await?;

    // --- expire snapshots [2, 3] (snap 4 is most-recent, always kept) ----------------
    let expired = writer.expire_snapshots(ExpireCriteria::Versions(vec![
        s2.snapshot_id,
        s3.snapshot_id,
    ]))?;
    println!("expire_snapshots -> {expired:#?}");
    print_state(&pool, &data_path, "Step 5 — expire versions [2, 3]").await?;

    // --- cleanup: physically delete scheduled files ----------------------------------
    let store: Arc<dyn ObjectStore> = Arc::new(LocalFileSystem::new());
    let dry = cleanup_old_files_sqlite(&writer, store.clone(), CleanupCriteria::All, true).await?;
    println!("cleanup dry_run -> {dry:#?}");
    let done = cleanup_old_files_sqlite(&writer, store, CleanupCriteria::All, false).await?;
    println!("cleanup real    -> {done:#?}");
    print_state(&pool, &data_path, "Step 6 — cleanup_old_files(all)").await?;

    // The s1 binding exists only so the table's first snapshot stays readable in the
    // narrative; it's unused after that.
    let _ = s1;
    Ok(())
}

fn cols() -> Vec<ColumnDef> {
    vec![
        ColumnDef::new("id", "int64", false).unwrap(),
        ColumnDef::new("name", "varchar", true).unwrap(),
    ]
}

fn touch_file(data_path: &Path, schema: &str, table: &str, name: &str) -> anyhow::Result<()> {
    let dir = data_path.join(schema).join(table);
    std::fs::create_dir_all(&dir)?;
    std::fs::write(dir.join(name), b"parquet-bytes")?;
    Ok(())
}

/// Dump everything that's interesting at this point: snapshots, the table row(s),
/// data files, scheduled-for-deletion queue, and what's physically on disk.
async fn print_state(pool: &SqlitePool, data_path: &Path, header: &str) -> anyhow::Result<()> {
    println!("\n──────── {header} ────────");

    println!("ducklake_snapshot:");
    let rows = sqlx::query(
        "SELECT snapshot_id, snapshot_time FROM ducklake_snapshot ORDER BY snapshot_id",
    )
    .fetch_all(pool)
    .await?;
    for r in &rows {
        let id: i64 = r.try_get(0)?;
        let t: String = r.try_get(1)?;
        println!("  snap {id} @ {t}");
    }

    println!("ducklake_table:");
    let rows = sqlx::query(
        "SELECT table_id, table_name, begin_snapshot, end_snapshot FROM ducklake_table",
    )
    .fetch_all(pool)
    .await?;
    for r in &rows {
        let id: i64 = r.try_get(0)?;
        let n: String = r.try_get(1)?;
        let b: i64 = r.try_get(2)?;
        let e: Option<i64> = r.try_get(3)?;
        println!("  table_id={id} name={n} begin={b} end={e:?}");
    }

    println!("ducklake_data_file:");
    let rows = sqlx::query(
        "SELECT data_file_id, path, begin_snapshot, end_snapshot FROM ducklake_data_file ORDER BY data_file_id",
    ).fetch_all(pool).await?;
    if rows.is_empty() {
        println!("  (none)");
    }
    for r in &rows {
        let id: i64 = r.try_get(0)?;
        let p: String = r.try_get(1)?;
        let b: i64 = r.try_get(2)?;
        let e: Option<i64> = r.try_get(3)?;
        println!("  df_id={id} path={p} begin={b} end={e:?}");
    }

    println!("ducklake_files_scheduled_for_deletion:");
    let rows = sqlx::query(
        "SELECT data_file_id, path, schedule_start FROM ducklake_files_scheduled_for_deletion ORDER BY data_file_id",
    ).fetch_all(pool).await?;
    if rows.is_empty() {
        println!("  (empty)");
    }
    for r in &rows {
        let id: i64 = r.try_get(0)?;
        let p: String = r.try_get(1)?;
        let t: String = r.try_get(2)?;
        println!("  df_id={id} path={p} scheduled_at={t}");
    }

    println!("ducklake_table_stats:");
    let rows = sqlx::query("SELECT table_id, record_count, next_row_id FROM ducklake_table_stats")
        .fetch_all(pool)
        .await?;
    if rows.is_empty() {
        println!("  (none)");
    }
    for r in &rows {
        let id: i64 = r.try_get(0)?;
        let rc: i64 = r.try_get(1)?;
        let nr: i64 = r.try_get(2)?;
        println!("  table_id={id} record_count={rc} next_row_id={nr}");
    }

    println!("files on disk under data_path:");
    let mut found = Vec::new();
    walk_files(data_path, &mut found);
    if found.is_empty() {
        println!("  (none)");
    }
    for p in found {
        println!("  {}", p.strip_prefix(data_path).unwrap().display());
    }
    println!();
    Ok(())
}

fn walk_files(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                walk_files(&p, out);
            } else {
                out.push(p);
            }
        }
    }
}
