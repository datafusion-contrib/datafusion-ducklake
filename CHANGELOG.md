# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Read DuckDB data inlining (SQLite): inlined INSERT rows (`ducklake_inlined_data_*`) are unioned into scans with snapshot visibility, fixing the silent `SELECT` / `COUNT(*)` undercount (SQLite + scalar types; other backends and the `rowid` path are follow-ups).
- Compaction (`merge_adjacent_files` + `rewrite_data_files`): two triggered maintenance ops on `DuckLakeTable`, each returning a `CompactionResult`. Merges small same-schema-version data files (multi-origin merges become DuckLake partial data files, read per-origin so time travel stays correct) and rewrites files past a deleted-fraction threshold (default `0.95`), keeping live rows and rowids. Commits atomically (`MetadataWriter::commit_compaction`, SQLite + PostgreSQL) and only schedules superseded files for later `cleanup_old_files`. Adds `ducklake_data_file.partial_max` + `ducklake_snapshot_changes` with in-place migrations; see [`examples/compaction_demo.rs`](examples/compaction_demo.rs) (#167).

## [0.4.0] - 2026-07-08

### Added
- Positional delete-file authoring: produce & register DuckLake positional delete files (`DuckLakeTable::resolve_positions`, `DuckLakeTableWriter::write_delete_file`, `MetadataWriter::set_delete_file`), keeping one live cumulative delete file per data file via atomic compare-and-swap. Providers now expose `data_file_id` / `delete_file_id` and `read_delete_file_positions` (SQLite + PostgreSQL) (#154, #155).
- Column type promotion (`MetadataWriter::promote_column_type`): explicit, widening-only schema evolution (e.g. `int32 → int64`) with a stable `column_id` / field-id and no data rewrite (`types::is_promotable`; SQLite + PostgreSQL).
- `schema_version` tracking on the SQLite write path: `ducklake_snapshot.schema_version` + a `ducklake_schema_versions` ledger, bumped on DDL commits — porting the PostgreSQL model; catalogs migrate in place on open (#151).

### Changed
- Upgrade to DataFusion 54 and Arrow/Parquet 58 — no on-disk/spec change (#150).
- Reject implicit column type changes on data writes (`Replace` / `Append`) with an error pointing at `promote_column_type` (previously silently dropped/accepted); alias-only restatements remain no-ops.
- `ducklake_column` allows column versioning (multiple rows per `column_id`) — bare table on SQLite, composite PK + partial unique index on PostgreSQL; catalogs migrate in place on open.

### Fixed
- Concurrent `WriteMode::Replace` on the PostgreSQL multi-catalog path now aborts with `Conflict` instead of unioning generations; converged onto DuckLake's commit-time snapshot model (single-transaction commit, stable field-ids) (#146).
- Nested (`List` / struct / map) columns no longer read back all-NULL — field-ids are read from top-level fields, not Parquet leaves; adds a `List` roundtrip regression test.

## [0.3.1] - 2026-06-23

### Documentation
- Refresh the README, add `COMPATIBILITY.md` documenting the backend/feature matrix, and correct `CLAUDE.md` to reflect read **and** write support. Update the crate-level doc comment accordingly (#144).

## [0.3.0] - 2026-06-22

### Added
- PostgreSQL multi-catalog support: multiple independent catalogs in one metadata store, per-catalog data-file segregation, single-table tombstone drops (`drop_table_in_catalog`), and `row_id_start` projection (#117, #120, #121, #124, #132).
- Row lineage: `rowid` virtual column, opt-in via `DuckLakeCatalog::with_row_lineage(true)`; compatible with DuckDB `UPDATE` / compaction output (#115).
- Maintenance API: single-catalog `DROP TABLE`, `expire_snapshots`, `cleanup_old_files`, and `delete_orphaned_files` (#122, #123).
- Writer tuning: Parquet compression (`with_compression`) and row-group caps (`with_max_row_group_rows` / `with_max_row_group_bytes`) (#126, #128).
- `MetadataProvider::get_table_row_count()`, accounting for delete files (#131).

### Changed
- Stream table writes through a staging file with multipart upload instead of buffering in memory, reducing peak memory (#127).
- CI: gate the single-catalog backend test suite (#139); run on `ubuntu-latest` (#118).

### Fixed
- Correct reads across schema evolution and repeated writes, resolving per-file schema mapping (#140, #141).
- Make `WriteMode::Replace` atomic to close a transient empty-read window (SQLite and general paths) (#135, #138).
- Truncate the table on a zero-row `INSERT OVERWRITE` / Replace (#142).
- Require single-partition input in `DuckLakeInsertExec` (#137).
- Derive `rowid` and delete positions from physical file position (#129).
- Map nanosecond timezone-aware timestamps to `timestamptz_ns` (#133).
- Emit catalog list type for `ARRAY`-backed columns (#125).
- Align `ducklake_column` / `ducklake_data_file` schema with the DuckLake spec (#116).

## [0.2.1] - 2026-05-05

### Added
- `TableProvider::statistics()` on `DuckLakeTable`: `total_byte_size` from cached per-file metadata (mirrors `ducklake_table_info`), marked `Precision::Inexact` since the catalog tracks compressed parquet bytes vs DataFusion's uncompressed Arrow output (#112).

### Changed
- README: revise Discord community link (#111)

## [0.2.0] - 2026-04-22

### Changed
- Upgraded DataFusion 52.2→53, Arrow/Parquet 57→58, object_store 0.12→0.13 (#108)

### Added
- Discord community link in README (#105)

## [0.1.2] - 2026-04-13

### Added
- Allow dynamic linking against system libduckdb (#103)

### Fixed
- Update workflow actions for Node.js 24 compatibility (#100)
- Pin 3rd party GitHub Actions to specific SHAs for supply-chain security (#97, #98, #99)

## [0.1.1] - 2026-04-01

### Added
- Support for list/array column types in DuckLake type mapping (#89)

### Fixed
- Missing `end_snapshot IS NULL` filter in Postgres and MySQL `get_table_structure()` (#88)

### Changed
- Updated transitive dependencies for security fixes (#94)

## [0.1.0] - 2026-03-11

### Changed
- Upgraded DataFusion to 52.2, Arrow/Parquet 57

### Fixed
- Validate catalog entity names to reject empty, control chars, and overlength
- Normalize type aliases and add promotion rules for schema evolution
- Validate record_count metadata to reject negative values
- Reject zero-column table creation
- Validate type strings in ColumnDef constructor to reject invalid types early

## [0.0.7] - 2026-02-24

### Fixed
- Validate numeric metadata casts (footer_size, file_size_bytes) to prevent silent truncation
- Error on missing delete files instead of silent data corruption
- Harden path resolver against path traversal, null bytes, encoded slash bypass, and unicode edge cases
- Validate decimal type string parsing and precision/scale bounds
- Handle empty catalogs where data directory does not yet exist
- Reject column_id values exceeding i32 range

## [0.0.6] - 2026-02-13

### Added
- S3/ObjectStore write support for DuckLake catalogs

### Changed
- Upgraded DataFusion 50→51, Arrow/Parquet 56→57

## [0.0.5] - 2026-02-04

### Added
- Write support with streaming API for DuckLake catalogs (`write` feature flag)
- SQL write support with `INSERT INTO` statements (`write` feature flag)
- Schema evolution support
- TPC-H and TPC-DS benchmarks comparing DuckDB-DuckLake vs DataFusion-DuckLake
- Benchmark test workflow for CI

### Changed
- Reuse DuckDB connection for metadata queries instead of creating new connection per call (performance improvement)

## [0.0.4] - 2026-01-14

### Added
- SQLite metadata provider (`metadata-sqlite` feature flag)
- Delete file CDC support in `ducklake_table_changes()` function

## [0.0.3] - 2026-01-09

### Added
- PostgreSQL metadata provider (`metadata-postgres` feature flag)
- MySQL metadata provider (`metadata-mysql` feature flag)
- Parquet Modular Encryption (PME) support for reading encrypted files (`encryption` feature flag)
- `ducklake_table_changes()` table function returning actual row data from Parquet files
- Feature flags for metadata providers
- SQLLogicTest runner for DuckDB test files

### Fixed
- Empty table queries now return empty results instead of errors
- Snapshot filtering for complete row deletion scenarios
- Column renaming via Parquet field_id → DuckLake column_id mapping
- Pinned rustc version to 1.92.0 for build stability

## [0.0.2] - 2025-12-17

### Added
- DuckDB-style table functions for catalog introspection:
  - `ducklake_snapshots()`, `ducklake_schemas()`, `ducklake_tables()`
  - `ducklake_columns()`, `ducklake_data_files()`, `ducklake_delete_files()`
- Snapshot-pinned catalog ensuring consistent reads across a query session

## [0.0.1] - 2025-10-25

Initial release.

### Added
- Read-only SQL queries against DuckLake catalogs via DataFusion
- Support for local filesystem and S3/MinIO object stores
- Row-level delete support (merge-on-read)
- Filter pushdown to Parquet
- Query-scoped snapshot isolation

[0.3.0]: https://github.com/hotdata-dev/datafusion-ducklake/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/hotdata-dev/datafusion-ducklake/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/hotdata-dev/datafusion-ducklake/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/hotdata-dev/datafusion-ducklake/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/hotdata-dev/datafusion-ducklake/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/hotdata-dev/datafusion-ducklake/compare/v0.0.7...v0.1.0
[0.0.7]: https://github.com/hotdata-dev/datafusion-ducklake/compare/v0.0.6...v0.0.7
[0.0.6]: https://github.com/hotdata-dev/datafusion-ducklake/compare/v0.0.5...v0.0.6
[0.0.5]: https://github.com/hotdata-dev/datafusion-ducklake/compare/v0.0.4...v0.0.5
[0.0.4]: https://github.com/hotdata-dev/datafusion-ducklake/compare/v0.0.3...v0.0.4
[0.0.3]: https://github.com/hotdata-dev/datafusion-ducklake/compare/v0.0.2...v0.0.3
[0.0.2]: https://github.com/hotdata-dev/datafusion-ducklake/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/hotdata-dev/datafusion-ducklake/releases/tag/v0.0.1
