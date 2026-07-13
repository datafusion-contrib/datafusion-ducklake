# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Compaction (`merge_adjacent_files` + `rewrite_data_files`).** Two explicit, triggered maintenance operations on `DuckLakeTable`, each returning a `CompactionResult` (`files_processed` / `files_created` / `rows_written`). `merge_adjacent_files` coalesces several small data files of one table — of the SAME schema version only, never across a DDL boundary — into fewer larger ones; a merged file spanning more than one origin snapshot is written as a DuckLake **partial data file** (embedding each row's original rowid AND a per-row `_ducklake_internal_snapshot_id` column, with `ducklake_data_file.partial_max` recording the max origin), and reads below `partial_max` filter its rows per-origin so time travel and change feeds stay correct. `rewrite_data_files` rewrites a data file whose deleted fraction exceeds a threshold (default `0.95`, configurable), reading only its live rows (delete-aware) and preserving rowids, then retiring both the old data file and its delete file. Both commit atomically in one snapshot (`MetadataWriter::commit_compaction`, SQLite + PostgreSQL), record `compacted_table:<table_id>` in `ducklake_snapshot_changes`, and only *schedule* superseded files for deletion (reclaimed later by `cleanup_old_files`) — the base-snapshot conflict check makes compaction coexist with concurrent appends. Adds the `ducklake_data_file.partial_max` column (v1.0) and the `ducklake_snapshot_changes` table, with in-place idempotent migrations for existing catalogs. See [`examples/compaction_demo.rs`](examples/compaction_demo.rs) (#167).

## [0.4.0] - 2026-07-08

### Added
- **Positional delete-file authoring (write path).** The crate can now *produce and register* DuckLake positional delete files, not only read them. `DuckLakeTable::resolve_positions` scans a data file and returns the physical row positions matching a predicate (the `pos` values a delete file records); `DuckLakeTableWriter::write_delete_file` writes and uploads the `(file_path, pos)` delete parquet beside the data files it masks; and `MetadataWriter::set_delete_file` (SQLite and PostgreSQL) registers it in a single atomic commit — fenced on the target data file's liveness, compare-and-swapping the currently-live delete file, and keeping at most one live delete file per data file by rewriting it cumulatively. Read providers now surface the catalog `data_file_id` and the live `delete_file_id` on `DuckLakeTableFile`, alongside `DuckLakeTable::files()` and a public `read_delete_file_positions`, so callers can assemble the cumulative (existing ∪ new) position set. This is the write-side primitive for row-level deletes; reads already apply delete files via merge-on-read (#154, #155).
- **Column type promotion (`MetadataWriter::promote_column_type`).** Explicit, widening-only schema evolution (e.g. `int32 → int64`): it retires the live `ducklake_column` row and inserts a new version with the **same** `column_id` (stable Parquet field-id), so files written before and after both resolve correctly and old narrow values are cast up on read — no data rewrite. Allowed widenings are an explicit, lossless set (`types::is_promotable`). Implemented on the SQLite single-catalog and PostgreSQL multi-catalog write paths.
- **`schema_version` tracking on the SQLite single-catalog write path** (#151), porting the validated PostgreSQL model. `ducklake_snapshot` now carries `schema_version` and a `ducklake_schema_versions` ledger table records each schema change; a DDL commit (table create, column add/remove/reorder, type promotion) bumps the per-catalog `schema_version` and writes a ledger row, while a pure data write carries the version forward — mirroring upstream's `if (SchemaChangesMade()) schema_version++`. This resolves the `TODO(schema_version)` left in the SQLite `promote_column_type`. Pre-existing catalogs gain the column in place on open (idempotent, lossless). Like the PostgreSQL layout, the SQLite layout deliberately omits upstream's `next_catalog_id` / `next_file_id` snapshot columns (this library allocates ids from its own counters).

### Changed
- **Upgrade to DataFusion 54 and Arrow/Parquet 58** (#150), adapting the positional-scan paths to DataFusion 54's datasource API (keeping file groups partition-local and delegating the positional morselizer). No on-disk format or DuckLake spec change.
- **A column type change on a data write (`Replace`/`Append`) is now rejected** with a clear error pointing at `promote_column_type` — previously a type change on `Replace` was silently dropped (the catalog kept the old type, corrupting reads) and a widening `Append` was silently accepted. Schema evolution must be explicit, mirroring upstream DuckLake's `ALTER`-vs-`INSERT` separation. An alias-only restatement (`bigint` ≡ `int64`) remains a no-op. To adopt a widened source schema, call `promote_column_type` then write under the new type.
- `ducklake_column` no longer uses a single-row `column_id` primary key, so a column can be versioned (two rows sharing a `column_id`): the SQLite layout matches upstream's bare table (invariants enforced in the writer + tests), the PostgreSQL multi-catalog layout uses a composite PK + a partial unique index. Catalogs written by an earlier version are migrated in place on open (idempotent, lossless).

### Fixed
- **Concurrent `WriteMode::Replace` on the experimental PostgreSQL multi-catalog write path could union conflicting generations instead of rejecting one.** Converged the multi-catalog writer onto DuckLake's commit-time model: a snapshot's id is assigned at commit (commit-ordered), and all of its metadata — snapshot, schema/table, columns, data files, and the published head — is written in a single transaction, so committed-but-unpublished rows are never visible to readers or conflict checks. `Replace` now aborts with a `Conflict` error when another writer published a newer generation of the table since the write began. Column field-ids stay stable across a same-schema `Replace`, and an `Append` racing a table creation aborts rather than producing NULL-filled reads. No on-disk format, DuckLake spec, or public API change (#146).
- **Nested (`List` / struct / map) columns could read back all-NULL.** The read-schema field-id matcher built its `field_id → column` map from Parquet *leaf* columns, but a column's field-id is stamped on its *top-level* field — which for a nested column is the group node, leaving the leaf with no id. The column was therefore treated as absent from the file and null-filled. Field-ids are now read from the top-level fields, so nested columns resolve correctly; scalar columns are unaffected (their top-level field is the leaf). Adds a `List` write→read roundtrip regression test.

## [0.3.1] - 2026-06-23

### Documentation
- Refresh the README, add `COMPATIBILITY.md` documenting the backend/feature matrix, and correct `CLAUDE.md` to reflect read **and** write support. Update the crate-level doc comment accordingly (#144).

## [0.3.0] - 2026-06-22

### Added
- **PostgreSQL multi-catalog support.** Manage and read multiple independent DuckLake catalogs within a single PostgreSQL metadata store, including per-catalog data-file segregation on disk, single-table tombstone drops (`drop_table_in_catalog`), and `row_id_start` projection on reads / population on data-file registration (#117, #120, #121, #124, #132).
- **Row lineage.** `rowid` virtual column exposing DuckLake row IDs, opt-in via `DuckLakeCatalog::with_row_lineage(true)`. Compatible with files produced by DuckDB's `UPDATE` / compaction (#115).
- **Maintenance API.** Single-catalog `DROP TABLE`, `expire_snapshots`, and `cleanup_old_files` for reclaiming superseded data, plus `delete_orphaned_files` for storage-scan reclamation of untracked files (#122, #123).
- **Writer tuning.** Configurable Parquet compression (`DuckLakeTableWriter::with_compression`) and row-group caps by row count and byte size (`with_max_row_group_rows` / `with_max_row_group_bytes`) (#126, #128).
- `MetadataProvider::get_table_row_count()`, which accounts for delete files (#131).

### Changed
- Writer streams table writes through a staging file with multipart upload instead of buffering in memory, reducing peak memory for large writes (#127).
- CI: gate the single-catalog backend test suite and fix/quarantine drifted fixtures (#139); run on `ubuntu-latest` instead of `ubuntu-latest-m` (#118).

### Fixed
- Correct reads across schema evolution and repeated writes, resolving per-file schema mapping for schema-evolving reads (#140, #141).
- Make `WriteMode::Replace` atomic to close a transient empty-read window, for both the single-catalog SQLite path and the general path (#135, #138).
- Truncate the table on a zero-row `INSERT OVERWRITE` / Replace (#142).
- Require single-partition input in `DuckLakeInsertExec` (#137).
- Derive `rowid` and delete positions from physical file position (#129).
- Map nanosecond timezone-aware timestamps to `timestamptz_ns` (#133).
- Emit catalog list type for `ARRAY`-backed columns (#125).
- Align `ducklake_column` / `ducklake_data_file` schema with the DuckLake spec (#116).

## [0.2.1] - 2026-05-05

### Added
- Implement `TableProvider::statistics()` on `DuckLakeTable`, populating `total_byte_size` from per-file metadata cached on the table (#112). Mirrors DuckLake's `ducklake_table_info` aggregate exactly. Marked `Precision::Inexact` since the catalog tracks compressed parquet bytes while DataFusion documents `total_byte_size` as uncompressed Arrow output. Enables cost-based optimisation hints and provides a cheap surface for size-aware consumers (e.g. pre-flight ingest guards).

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