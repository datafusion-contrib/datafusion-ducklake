//! Step-by-step demo of `delete_orphaned_files_sqlite` against a single-catalog
//! SQLite setup with `LocalFileSystem`. Visualises each scenario the regression
//! tests cover, plus the runtime decisions the sweep makes.
//!
//! Companion to `examples/orphan_cleanup_demo.sql` (drives the equivalent
//! scenarios through the official DuckDB + DuckLake extension). The two
//! outputs line up step-by-step.
//!
//! Scenarios shown:
//!   1. Setup — empty catalog, empty data_path
//!   2. Register one data file → file referenced; orphan sweep is a no-op
//!   3. Drop a stray .parquet on disk → orphan sweep finds + deletes it
//!   4. dry_run reports without deleting; real run matches the dry_run set
//!   5. Non-.parquet files are ignored
//!   6. Nested-directory orphan is found by recursive listing
//!   7. `OlderThan` cutoff skips a freshly-written orphan
//!   8. `All` deletes regardless of age — the dangerous opt-in
//!   9. A row in `ducklake_files_scheduled_for_deletion` is treated as
//!      referenced (must not race ahead of cleanup_old_files)
//!
//! Run with:
//!     cargo run --no-default-features --features write-sqlite \
//!         --example orphan_cleanup_demo

use std::path::{Path, PathBuf};
use std::sync::Arc;

use object_store::ObjectStore;
use object_store::local::LocalFileSystem;
use sqlx::Row;
use sqlx::sqlite::SqlitePool;

use datafusion_ducklake::SqliteMetadataWriter;
use datafusion_ducklake::maintenance::{
    CleanupCriteria, cleanup_old_files_sqlite, delete_orphaned_files_sqlite,
};
use datafusion_ducklake::metadata_writer::{ColumnDef, DataFileInfo, MetadataWriter, WriteMode};

#[tokio::main(flavor = "multi_thread")]
async fn main() -> anyhow::Result<()> {
    let temp = tempfile::TempDir::new()?;
    let db_path = temp.path().join("catalog.db");
    let data_path = temp.path().join("data");
    std::fs::create_dir_all(&data_path)?;
    let conn_str = format!("sqlite:{}?mode=rwc", db_path.display());

    let writer = SqliteMetadataWriter::new_with_init(&conn_str).await?;
    writer.set_data_path(data_path.to_str().unwrap())?;
    let pool = SqlitePool::connect(&conn_str).await?;
    let store: Arc<dyn ObjectStore> = Arc::new(LocalFileSystem::new());

    println!("data_path = {}\n", data_path.display());

    // ------ Step 1: empty state ----------------------------------------------
    print_state(&pool, &data_path, "Step 1 — empty catalog, empty data_path").await?;
    let res =
        delete_orphaned_files_sqlite(&writer, store.clone(), CleanupCriteria::All, false).await?;
    println!("delete_orphaned_files(All) -> {res:?}\n");

    // ------ Step 2: register + materialise one referenced file ---------------
    let s = writer.begin_write_transaction(
        "main",
        "t",
        &[ColumnDef::new("id", "int64", false)?, ColumnDef::new("name", "varchar", true)?],
        WriteMode::Replace,
    )?;
    writer.register_data_file(
        s.table_id,
        s.snapshot_id,
        &DataFileInfo::new("ref.parquet", 100, 5),
        WriteMode::Replace,
        &[ColumnDef::new("id", "int64", false)?, ColumnDef::new("name", "varchar", true)?],
        &s.column_ids,
    )?;
    touch_file(&data_path, "main", "t", "ref.parquet")?;
    print_state(
        &pool,
        &data_path,
        "Step 2 — one referenced file (ref.parquet)",
    )
    .await?;
    let res =
        delete_orphaned_files_sqlite(&writer, store.clone(), CleanupCriteria::All, false).await?;
    println!("delete_orphaned_files(All) -> {res:?}\n");

    // ------ Step 3: drop a stray .parquet ------------------------------------
    touch_file(&data_path, "main", "t", "stray.parquet")?;
    print_state(
        &pool,
        &data_path,
        "Step 3 — stray.parquet added (unreferenced)",
    )
    .await?;
    let dry =
        delete_orphaned_files_sqlite(&writer, store.clone(), CleanupCriteria::All, true).await?;
    println!("delete_orphaned_files(All) dry_run -> {dry:?}");
    let real =
        delete_orphaned_files_sqlite(&writer, store.clone(), CleanupCriteria::All, false).await?;
    println!("delete_orphaned_files(All) real    -> {real:?}");
    assert_eq!(dry, real, "dry_run and real_run must agree");
    print_state(&pool, &data_path, "Step 3 (after) — stray gone").await?;

    // ------ Step 4: non-.parquet file is ignored -----------------------------
    std::fs::write(
        data_path.join("main").join("t").join("README.txt"),
        b"keep me",
    )?;
    touch_file(&data_path, "main", "t", "orphan2.parquet")?;
    print_state(
        &pool,
        &data_path,
        "Step 4 — README.txt (ignored) + orphan2.parquet (orphan)",
    )
    .await?;
    let res =
        delete_orphaned_files_sqlite(&writer, store.clone(), CleanupCriteria::All, false).await?;
    println!("delete_orphaned_files(All) -> {res:?}");
    print_state(
        &pool,
        &data_path,
        "Step 4 (after) — orphan2 gone; README intact",
    )
    .await?;

    // ------ Step 5: nested directory orphan ----------------------------------
    let nested = data_path
        .join("main")
        .join("t")
        .join("year=2024")
        .join("month=01");
    std::fs::create_dir_all(&nested)?;
    std::fs::write(nested.join("part.parquet"), b"orphan-deep")?;
    print_state(
        &pool,
        &data_path,
        "Step 5 — orphan at main/t/year=2024/month=01/",
    )
    .await?;
    let res =
        delete_orphaned_files_sqlite(&writer, store.clone(), CleanupCriteria::All, false).await?;
    println!("delete_orphaned_files(All) -> {res:?}");
    print_state(&pool, &data_path, "Step 5 (after) — nested orphan reaped").await?;

    // ------ Step 6: OlderThan skips a fresh orphan ---------------------------
    touch_file(&data_path, "main", "t", "fresh.parquet")?;
    print_state(&pool, &data_path, "Step 6 — fresh.parquet just written").await?;
    let cutoff = chrono::Utc::now() - chrono::Duration::hours(1);
    let res = delete_orphaned_files_sqlite(
        &writer,
        store.clone(),
        CleanupCriteria::OlderThan(cutoff),
        false,
    )
    .await?;
    println!("delete_orphaned_files(OlderThan(now - 1h)) -> {res:?}   ← in-flight protection");

    // Now `All` does delete it (the operator opt-in).
    let res =
        delete_orphaned_files_sqlite(&writer, store.clone(), CleanupCriteria::All, false).await?;
    println!("delete_orphaned_files(All) -> {res:?}              ← danger-mode opt-in");
    print_state(
        &pool,
        &data_path,
        "Step 6 (after) — fresh.parquet finally deleted",
    )
    .await?;

    // ------ Step 7: a file in scheduled_for_deletion is NOT an orphan --------
    touch_file(&data_path, "main", "t", "scheduled.parquet")?;
    sqlx::query(
        "INSERT INTO ducklake_files_scheduled_for_deletion
             (data_file_id, path, path_is_relative, schedule_start)
         VALUES (?, ?, 1, CURRENT_TIMESTAMP)",
    )
    .bind(42_i64)
    .bind("main/t/scheduled.parquet")
    .execute(&pool)
    .await?;
    print_state(
        &pool,
        &data_path,
        "Step 7 — scheduled.parquet awaiting cleanup_old_files",
    )
    .await?;
    let res =
        delete_orphaned_files_sqlite(&writer, store.clone(), CleanupCriteria::All, false).await?;
    println!("delete_orphaned_files(All) -> {res:?}    ← scheduled rows are treated as referenced");

    // cleanup_old_files is what's supposed to delete it.
    let res = cleanup_old_files_sqlite(&writer, store, CleanupCriteria::All, false).await?;
    println!("cleanup_old_files(All)   -> {res:?}");
    print_state(
        &pool,
        &data_path,
        "Step 7 (after) — cleanup_old_files reaped it",
    )
    .await?;

    Ok(())
}

fn touch_file(data_path: &Path, schema: &str, table: &str, name: &str) -> anyhow::Result<()> {
    let dir = data_path.join(schema).join(table);
    std::fs::create_dir_all(&dir)?;
    std::fs::write(dir.join(name), b"parquet-bytes")?;
    Ok(())
}

async fn print_state(pool: &SqlitePool, data_path: &Path, header: &str) -> anyhow::Result<()> {
    println!("\n──────── {header} ────────");

    println!("ducklake_data_file:");
    let rows = sqlx::query(
        "SELECT data_file_id, path, path_is_relative, end_snapshot FROM ducklake_data_file ORDER BY data_file_id",
    )
    .fetch_all(pool)
    .await?;
    if rows.is_empty() {
        println!("  (none)");
    }
    for r in &rows {
        let id: i64 = r.try_get(0)?;
        let p: String = r.try_get(1)?;
        let rel: i64 = r.try_get(2)?;
        let e: Option<i64> = r.try_get(3)?;
        println!("  df_id={id} path={p} rel={rel} end_snapshot={e:?}");
    }

    println!("ducklake_files_scheduled_for_deletion:");
    let rows = sqlx::query(
        "SELECT data_file_id, path FROM ducklake_files_scheduled_for_deletion ORDER BY data_file_id",
    )
    .fetch_all(pool)
    .await?;
    if rows.is_empty() {
        println!("  (empty)");
    }
    for r in &rows {
        let id: i64 = r.try_get(0)?;
        let p: String = r.try_get(1)?;
        println!("  df_id={id} path={p}");
    }

    let files = walk_files(data_path);
    println!("files on disk under data_path ({} total):", files.len());
    if files.is_empty() {
        println!("  (none)");
    }
    for p in files {
        println!("  {}", p.strip_prefix(data_path).unwrap().display());
    }
    println!();
    Ok(())
}

fn walk_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    fn recurse(dir: &Path, out: &mut Vec<PathBuf>) {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for e in entries.flatten() {
                let p = e.path();
                if p.is_dir() {
                    recurse(&p, out);
                } else {
                    out.push(p);
                }
            }
        }
    }
    recurse(dir, &mut out);
    out.sort();
    out
}
