# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.6.0] - 2026-07-20

### Added
- Table partitioning: `SET`/`RESET PARTITIONED BY` (`identity`/`year`/`month`/`day`/`hour`); per-partition files on write (SQLite), file pruning on read (all backends) (#191).
- `ducklake_table_insertions()` — the official insertions feed (#179).
- CDC snapshot bounds accept timestamp strings (#179).
- `retire_appends_since` to roll back a pure-append delta (#182).
- `rowid` emitted by `ducklake_table_changes` / `ducklake_table_deletions` (#180).
- Differential CDC conformance suite vs the official extension (#179).

### Changed
- **BREAKING**: CDC snapshot bounds are inclusive on both ends; paginate with `last + 1` (#179).
- **BREAKING**: CDC output leads with `(snapshot_id, rowid, change_type)` (#179).
- **BREAKING**: `ducklake_table_changes` emits pure deletes as `change_type='delete'` rows (#179).
- **BREAKING**: metadata providers gained a private field — construct via `new()`/`from_pool()`, not struct literals (#192).
- Filters push through pure column renames (#188).
- Scan planning streams file metadata, memoizes capability probes, and stops at a short page (#181, #192).

### Fixed
- DuckDB delete-file window off-by-one double-reported boundary deletions (#179).
- CDC missed changes in compaction-merged files (`partial_max` windows) (#179).
- Cumulative delete files windowed per row, each deletion at its own snapshot (#179).
- `SELECT COUNT(*)` over `ducklake_table_changes` on the insert-only path (#179).

## [0.5.0] - 2026-07-15

### Added
- Read DuckDB data inlining (SQLite).
- Compaction: `merge_adjacent_files` + `rewrite_data_files` (#167).

## [0.4.0] - 2026-07-08

### Added
- Positional delete-file authoring (write path) (#154, #155).
- Column type promotion (`promote_column_type`).
- `schema_version` tracking on SQLite (#151).

### Changed
- Upgrade to DataFusion 54, Arrow/Parquet 58 (#150).
- Reject implicit column type changes on data writes.
- `ducklake_column` supports column versioning.

### Fixed
- Concurrent `Replace` on PostgreSQL multi-catalog aborts on conflict (#146).
- Nested (`List`/struct/map) columns no longer read all-NULL.

## [0.3.1] - 2026-06-23

### Documentation
- Refresh README, add `COMPATIBILITY.md` (#144).

## [0.3.0] - 2026-06-22

### Added
- PostgreSQL multi-catalog support (#117, #120, #121, #124, #132).
- Row lineage (`rowid` virtual column) (#115).
- Maintenance API: `DROP TABLE`, `expire_snapshots`, `cleanup_old_files`, `delete_orphaned_files` (#122, #123).
- Writer tuning: compression + row-group caps (#126, #128).
- `get_table_row_count()`, delete-aware (#131).

### Changed
- Stream writes via staging file + multipart upload (#127).
- CI: gate single-catalog suite (#139); run on `ubuntu-latest` (#118).

### Fixed
- Reads across schema evolution + repeated writes (#140, #141).
- Atomic `WriteMode::Replace` (#135, #138).
- Truncate on zero-row `INSERT OVERWRITE` (#142).
- Single-partition input in `DuckLakeInsertExec` (#137).
- `rowid`/delete positions from physical position (#129).
- Nanosecond tz-aware timestamps to `timestamptz_ns` (#133).
- Catalog list type for `ARRAY` columns (#125).
- Align schema with DuckLake spec (#116).

## [0.2.1] - 2026-05-05

### Added
- `TableProvider::statistics()` — `total_byte_size`, `Inexact` (#112).

### Changed
- README: Discord link (#111).

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
- Pin 3rd party GitHub Actions to specific SHAs (#97, #98, #99)

## [0.1.1] - 2026-04-01

### Added
- List/array column types in DuckLake type mapping (#89)

### Fixed
- Missing `end_snapshot IS NULL` filter in Postgres/MySQL `get_table_structure()` (#88)

### Changed
- Updated transitive dependencies for security fixes (#94)

## [0.1.0] - 2026-03-11

### Changed
- Upgraded DataFusion to 52.2, Arrow/Parquet 57

### Fixed
- Validate catalog entity names
- Normalize type aliases; add schema-evolution promotion rules
- Validate record_count metadata (reject negatives)
- Reject zero-column table creation
- Validate type strings in `ColumnDef` constructor

## [0.0.7] - 2026-02-24

### Fixed
- Validate numeric metadata casts (footer_size, file_size_bytes)
- Error on missing delete files instead of silent corruption
- Harden path resolver against traversal, null bytes, encoded slashes
- Validate decimal type parsing and precision/scale bounds
- Handle empty catalogs where the data directory does not yet exist
- Reject column_id values exceeding i32 range

## [0.0.6] - 2026-02-13

### Added
- S3/ObjectStore write support

### Changed
- Upgraded DataFusion 50→51, Arrow/Parquet 56→57

## [0.0.5] - 2026-02-04

### Added
- Write support with streaming API (`write` feature flag)
- SQL `INSERT INTO` write support (`write` feature flag)
- Schema evolution support
- TPC-H and TPC-DS benchmarks (DuckDB-DuckLake vs DataFusion-DuckLake)
- Benchmark test workflow for CI

### Changed
- Reuse DuckDB connection for metadata queries

## [0.0.4] - 2026-01-14

### Added
- SQLite metadata provider (`metadata-sqlite` feature flag)
- Delete file CDC support in `ducklake_table_changes()`

## [0.0.3] - 2026-01-09

### Added
- PostgreSQL metadata provider (`metadata-postgres` feature flag)
- MySQL metadata provider (`metadata-mysql` feature flag)
- Parquet Modular Encryption (PME) reads (`encryption` feature flag)
- `ducklake_table_changes()` table function
- Feature flags for metadata providers
- SQLLogicTest runner for DuckDB test files

### Fixed
- Empty table queries return empty results instead of errors
- Snapshot filtering for complete row deletion
- Column renaming via Parquet field_id → DuckLake column_id
- Pinned rustc to 1.92.0 for build stability

## [0.0.2] - 2025-12-17

### Added
- Catalog introspection table functions (`ducklake_snapshots()`, `ducklake_schemas()`, `ducklake_tables()`, `ducklake_columns()`, `ducklake_data_files()`, `ducklake_delete_files()`)
- Snapshot-pinned catalog for consistent reads across a session

## [0.0.1] - 2025-10-25

Initial release.

### Added
- Read-only SQL queries against DuckLake catalogs via DataFusion
- Local filesystem and S3/MinIO object stores
- Row-level delete support (merge-on-read)
- Filter pushdown to Parquet
- Query-scoped snapshot isolation

[Unreleased]: https://github.com/hotdata-dev/datafusion-ducklake/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/hotdata-dev/datafusion-ducklake/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/hotdata-dev/datafusion-ducklake/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/hotdata-dev/datafusion-ducklake/compare/v0.3.0...v0.3.1
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
