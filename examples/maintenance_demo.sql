-- Side-by-side comparable demo of the official DuckDB+DuckLake maintenance flow.
-- Mirrors `examples/maintenance_demo.rs` (which drives the same logical sequence
-- through our Rust port), so the two outputs can be lined up step-by-step.
--
-- Hardcoded paths under /tmp/maint_demo_official — caller must `rm -rf` first.
-- Run with the locally-built DuckLake extension (HEAD of the ported repo):
--   rm -rf /tmp/maint_demo_official && mkdir -p /tmp/maint_demo_official/data
--   /Users/tsoap/hotdata/ducklake/build/release/duckdb -unsigned \
--     -c "LOAD '/Users/tsoap/hotdata/ducklake/build/release/extension/ducklake/ducklake.duckdb_extension';" \
--     < examples/maintenance_demo.sql

.bail on
LOAD '/Users/tsoap/hotdata/ducklake/build/release/extension/ducklake/ducklake.duckdb_extension';

ATTACH 'ducklake:/tmp/maint_demo_official/catalog.db' AS dl
    (DATA_PATH '/tmp/maint_demo_official/data', METADATA_CATALOG 'metadata');

.print "data_path = /tmp/maint_demo_official/data"
.print ""

-- ── Step 1: CREATE TABLE (DDL only) ──
CREATE TABLE dl.main.t(id BIGINT, name VARCHAR);
.print "──────── Step 1 — CREATE TABLE main.t ────────"
SELECT 'ducklake_snapshot' AS rel; SELECT snapshot_id, snapshot_time FROM metadata.ducklake_snapshot ORDER BY snapshot_id;
SELECT 'ducklake_table' AS rel;    SELECT table_id, table_name, begin_snapshot, end_snapshot FROM metadata.ducklake_table;
SELECT 'ducklake_data_file' AS rel; SELECT data_file_id, path, begin_snapshot, end_snapshot FROM metadata.ducklake_data_file ORDER BY data_file_id;
SELECT 'ducklake_files_scheduled_for_deletion' AS rel; SELECT data_file_id, path, schedule_start FROM metadata.ducklake_files_scheduled_for_deletion ORDER BY data_file_id;
SELECT 'files on disk' AS rel; SELECT file FROM glob('/tmp/maint_demo_official/data/**') ORDER BY file;
.print ""

-- ── Step 2: INSERT rows → writes f1 ──
INSERT INTO dl.main.t VALUES (1, 'a'), (2, 'b'), (3, 'c'), (4, 'd'), (5, 'e');
.print "──────── Step 2 — INSERT 5 rows (f1) ────────"
SELECT snapshot_id, snapshot_time FROM metadata.ducklake_snapshot ORDER BY snapshot_id;
SELECT table_id, table_name, begin_snapshot, end_snapshot FROM metadata.ducklake_table;
SELECT data_file_id, path, begin_snapshot, end_snapshot FROM metadata.ducklake_data_file ORDER BY data_file_id;
SELECT data_file_id, path, schedule_start FROM metadata.ducklake_files_scheduled_for_deletion ORDER BY data_file_id;
SELECT file FROM glob('/tmp/maint_demo_official/data/**') ORDER BY file;
.print ""

-- ── Step 3: DELETE all + INSERT more → ends f1, writes f2 (parallels our Replace) ──
DELETE FROM dl.main.t;
INSERT INTO dl.main.t VALUES (6, 'f'), (7, 'g'), (8, 'h'), (9, 'i'), (10, 'j');
.print "──────── Step 3 — DELETE all + INSERT (f2) ────────"
SELECT snapshot_id, snapshot_time FROM metadata.ducklake_snapshot ORDER BY snapshot_id;
SELECT table_id, table_name, begin_snapshot, end_snapshot FROM metadata.ducklake_table;
SELECT data_file_id, path, begin_snapshot, end_snapshot FROM metadata.ducklake_data_file ORDER BY data_file_id;
SELECT data_file_id, path, schedule_start FROM metadata.ducklake_files_scheduled_for_deletion ORDER BY data_file_id;
SELECT file FROM glob('/tmp/maint_demo_official/data/**') ORDER BY file;
.print ""

-- ── Step 4: DROP TABLE ──
DROP TABLE dl.main.t;
.print "──────── Step 4 — DROP TABLE main.t ────────"
SELECT snapshot_id, snapshot_time FROM metadata.ducklake_snapshot ORDER BY snapshot_id;
SELECT table_id, table_name, begin_snapshot, end_snapshot FROM metadata.ducklake_table;
SELECT data_file_id, path, begin_snapshot, end_snapshot FROM metadata.ducklake_data_file ORDER BY data_file_id;
SELECT data_file_id, path, schedule_start FROM metadata.ducklake_files_scheduled_for_deletion ORDER BY data_file_id;
SELECT file FROM glob('/tmp/maint_demo_official/data/**') ORDER BY file;
.print ""

-- ── Step 5: expire snapshots [2, 3] (snap 4 is most-recent, kept) ──
.print "── expire_snapshots(versions => [2, 3]) ──"
CALL ducklake_expire_snapshots('dl', versions => [2, 3]);
.print ""
.print "──────── Step 5 — after expire ────────"
SELECT snapshot_id, snapshot_time FROM metadata.ducklake_snapshot ORDER BY snapshot_id;
SELECT table_id, table_name, begin_snapshot, end_snapshot FROM metadata.ducklake_table;
SELECT data_file_id, path, begin_snapshot, end_snapshot FROM metadata.ducklake_data_file ORDER BY data_file_id;
SELECT data_file_id, path, schedule_start FROM metadata.ducklake_files_scheduled_for_deletion ORDER BY data_file_id;
SELECT file FROM glob('/tmp/maint_demo_official/data/**') ORDER BY file;
.print ""

-- ── Step 6: cleanup ──
.print "── cleanup_old_files dry_run=true, cleanup_all=true ──"
SELECT path FROM ducklake_cleanup_old_files('dl', cleanup_all => true, dry_run => true);
.print ""
.print "── cleanup_old_files(cleanup_all => true) [real] ──"
CALL ducklake_cleanup_old_files('dl', cleanup_all => true);
.print ""
.print "──────── Step 6 — after cleanup_old_files ────────"
SELECT snapshot_id, snapshot_time FROM metadata.ducklake_snapshot ORDER BY snapshot_id;
SELECT table_id, table_name, begin_snapshot, end_snapshot FROM metadata.ducklake_table;
SELECT data_file_id, path, begin_snapshot, end_snapshot FROM metadata.ducklake_data_file ORDER BY data_file_id;
SELECT data_file_id, path, schedule_start FROM metadata.ducklake_files_scheduled_for_deletion ORDER BY data_file_id;
SELECT file FROM glob('/tmp/maint_demo_official/data/**') ORDER BY file;
