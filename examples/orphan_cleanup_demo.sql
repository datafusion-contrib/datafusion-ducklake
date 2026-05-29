-- Official DuckDB + DuckLake companion to `examples/orphan_cleanup_demo.rs`.
-- Drives the same scenarios via `ducklake_delete_orphaned_files` so the two
-- outputs can be lined up step-by-step. Behaviour should match modulo cosmetic
-- output formatting (leading slashes, default `older_than`, etc.).
--
-- Run with the locally-built DuckLake extension:
--   rm -rf /tmp/orphan_demo_official && mkdir -p /tmp/orphan_demo_official
--   /Users/tsoap/hotdata/ducklake/build/release/duckdb -unsigned \
--     < examples/orphan_cleanup_demo.sql

.bail on
LOAD '/Users/tsoap/hotdata/ducklake/build/release/extension/ducklake/ducklake.duckdb_extension';

ATTACH 'ducklake:/tmp/orphan_demo_official/catalog.db' AS dl
    (DATA_PATH '/tmp/orphan_demo_official/data', METADATA_CATALOG 'metadata');

.print "data_path = /tmp/orphan_demo_official/data"

-- ── Step 1: empty catalog, empty data_path ──
.print ""
.print "──────── Step 1 — empty catalog, empty data_path ────────"
SELECT path FROM ducklake_delete_orphaned_files('dl', cleanup_all => true, dry_run => true);

-- ── Step 2: one referenced file (INSERT writes the parquet) ──
CREATE TABLE dl.main.t(id BIGINT, name VARCHAR);
INSERT INTO dl.main.t VALUES (1, 'a'), (2, 'b'), (3, 'c'), (4, 'd'), (5, 'e');
.print ""
.print "──────── Step 2 — one referenced file written by INSERT ────────"
SELECT 'data_file' AS rel; SELECT data_file_id, path FROM metadata.ducklake_data_file ORDER BY data_file_id;
SELECT 'files on disk' AS rel; SELECT file FROM glob('/tmp/orphan_demo_official/data/**') ORDER BY file;
SELECT 'orphan sweep (dry_run)' AS rel; SELECT path FROM ducklake_delete_orphaned_files('dl', cleanup_all => true, dry_run => true);

-- ── Step 3: drop a stray .parquet on disk ──
.shell touch /tmp/orphan_demo_official/data/main/t/stray.parquet
.print ""
.print "──────── Step 3 — stray.parquet added (unreferenced) ────────"
SELECT 'files on disk' AS rel; SELECT file FROM glob('/tmp/orphan_demo_official/data/**') ORDER BY file;
SELECT 'orphan sweep (dry_run)' AS rel; SELECT path FROM ducklake_delete_orphaned_files('dl', cleanup_all => true, dry_run => true);
SELECT 'orphan sweep (real)'    AS rel; SELECT path FROM ducklake_delete_orphaned_files('dl', cleanup_all => true);
SELECT 'files on disk (after)' AS rel; SELECT file FROM glob('/tmp/orphan_demo_official/data/**') ORDER BY file;

-- ── Step 4: non-.parquet ignored ──
.shell echo 'keep me' > /tmp/orphan_demo_official/data/main/t/README.txt
.shell touch /tmp/orphan_demo_official/data/main/t/orphan2.parquet
.print ""
.print "──────── Step 4 — README.txt + orphan2.parquet ────────"
SELECT 'files on disk' AS rel; SELECT file FROM glob('/tmp/orphan_demo_official/data/**') ORDER BY file;
SELECT 'orphan sweep' AS rel; SELECT path FROM ducklake_delete_orphaned_files('dl', cleanup_all => true);
SELECT 'files on disk (after)' AS rel; SELECT file FROM glob('/tmp/orphan_demo_official/data/**') ORDER BY file;

-- ── Step 5: nested directory orphan ──
.shell mkdir -p /tmp/orphan_demo_official/data/main/t/year=2024/month=01
.shell touch /tmp/orphan_demo_official/data/main/t/year=2024/month=01/part.parquet
.print ""
.print "──────── Step 5 — orphan at main/t/year=2024/month=01/ ────────"
SELECT 'files on disk' AS rel; SELECT file FROM glob('/tmp/orphan_demo_official/data/**') ORDER BY file;
SELECT 'orphan sweep' AS rel; SELECT path FROM ducklake_delete_orphaned_files('dl', cleanup_all => true);
SELECT 'files on disk (after)' AS rel; SELECT file FROM glob('/tmp/orphan_demo_official/data/**') ORDER BY file;

-- ── Step 6: older_than skips fresh files; cleanup_all overrides ──
.shell touch /tmp/orphan_demo_official/data/main/t/fresh.parquet
.print ""
.print "──────── Step 6 — fresh.parquet just touched ────────"
SELECT 'older_than = NOW() - 1h (skips fresh)' AS rel;
SELECT path FROM ducklake_delete_orphaned_files('dl', older_than => NOW() - INTERVAL 1 HOUR);
SELECT 'cleanup_all = true (deletes fresh)'   AS rel;
SELECT path FROM ducklake_delete_orphaned_files('dl', cleanup_all => true);
SELECT 'files on disk (after)' AS rel; SELECT file FROM glob('/tmp/orphan_demo_official/data/**') ORDER BY file;
