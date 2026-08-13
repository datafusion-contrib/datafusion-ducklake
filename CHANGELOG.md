# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Recursive `list`, `struct`, and `map` columns use standard `ducklake_column` parent links and
  matching Parquet field IDs across reads, writes, schema evolution, rewrites, and compaction.
- `PostgresSingleCatalogMetadataWriter` — writes the **standard, spec-compliant** single-catalog DuckLake layout on PostgreSQL (no `catalog_id` columns, no `ducklake_catalog*` map tables, unscoped relative paths), so Postgres catalogs are interchangeable with the SQLite/MySQL backends and DuckDB's `ducklake` extension. SQL `CREATE TABLE AS SELECT` works on this path, unlike the multicatalog writer (#165).
- Atomic append+delete commits accept SEVERAL appended data files: `MetadataWriter::register_data_files_with_deletes` (and its conditional `_and_commit_metadata` sibling) register N data files plus M positional delete files in ONE snapshot — matching the reference implementation, where one transaction carries both the new data files and the new delete files. A keyed mutation (update/upsert) therefore works on a partitioned table and on a write that rolled past `target_file_size` (#214).
- SQL `UPDATE` on a PARTITIONED table: each rewritten row is routed by its own post-assignment key values, so an assignment that changes a partition key moves the row to its new partition (calendar transforms included). A rewrite spanning several partitions writes one file per partition and commits them all — with the positional deletes — in one snapshot, preserving every row's `rowid` lineage. Rewritten rows adopt the table's live spec, so a table that adopted partitioning after data was written partitions its updated rows (#214).
- `DuckLakeTable::files_matching` — the data files a predicate could match, pruned by exactly the catalog statistics and partition bounds a `SELECT` with the same filter uses. A caller driving its own per-file work (a keyed update, an upsert, a positional delete resolved with `resolve_positions`) no longer has to open every data file in the table to find the ones holding a key. Pruning is fail-open: a file is dropped only when its own statistics prove it cannot match, and files are read in bounded pages so peak memory tracks the result rather than the table.
- `DuckLakeTable::file_has_embedded_rowid` is now public, and available without the write features. It reports where a row's `rowid` comes from — the file's embedded row-id column when it has one, `row_id_start + physical position` otherwise. It is not a precondition for `resolve_positions` or for a keyed mutation; see the `### Fixed` note below.

### Changed
- **BREAKING**: `ducklake_table_changes`, `ducklake_table_insertions` and
  `ducklake_table_deletions` resolve each data file's columns **by field id**, as of the window's
  end snapshot, instead of by current name — matching official DuckLake. Output changes, silently,
  on any table whose columns were renamed or dropped and re-added:
  - a renamed column returned NULL for rows in files written before the rename, and now returns
    those rows' values;
  - a column dropped and re-added under the same name returned the DROPPED column's values for
    rows in files written before the re-add, and now returns NULL there (the re-added column has
    its own field id, and those files predate it);
  - two columns whose names were swapped returned each other's values, and now return their own;
  - a field renamed inside a `STRUCT` behaves the same way as a top-level one.

  The fix is **not retroactive**: it changes what the feeds return from now on, and nothing in the
  catalog was damaged, so no repair tooling is needed — but anything derived from earlier feed
  output on an affected table is still wrong. **Re-derive anything built from change-feed output on
  a table whose columns were renamed, or dropped and re-added.** To find those tables, look for a
  `column_id` with more than one row, or a name shared by two `column_id`s:

  ```sql
  SELECT table_id, column_id, count(*) AS generations
  FROM ducklake_column GROUP BY table_id, column_id HAVING count(*) > 1;

  SELECT table_id, column_name, count(DISTINCT column_id) AS ids
  FROM ducklake_column GROUP BY table_id, column_name HAVING count(DISTINCT column_id) > 1;
  ```

  Field ids come from each file's parquet footer, so the feeds now read one footer per data file —
  the insert-only feed previously read none — and each per-file scan carries one more plan node.
  On a table with **encrypted** files the footers cannot be read, so a feed whose window spans a
  rename or a drop-and-re-add is refused with an error rather than served by name. That refusal is
  conservative and catches some column changes that would have been safe to match by name, and only
  `ducklake_table_changes` / `ducklake_table_insertions` give the explanatory error; see
  COMPATIBILITY.md.
- `TableChangesTable`, `TableInsertionsTable` and `TableDeletionsTable` accept the table's columns
  through a new `with_columns` builder. No signature changed; without it the columns are read from
  the metadata provider on each scan.
- **BREAKING**: PostgreSQL metadata features no longer select a TLS provider. Consumers that need
  TLS must also enable one of `tls-native-tls`, `tls-rustls-aws-lc-rs`, or `tls-rustls-ring`.
  Without one, SQLx rejects connections that require TLS and may try plaintext when `sslmode`
  prefers TLS. No catalog or data migration is needed.
- **BREAKING**: S3 support is no longer selected by the library. Consumers using
  `object_store::aws` must enable `object_store/aws` in their application or register another
  `ObjectStore` implementation with DataFusion. Local filesystem support remains available through
  DataFusion without extra configuration. No catalog or data migration is needed.
- `TableWriteSession::finish_with_deletes` no longer refuses a session that produced more than one appended file; it commits them all in the snapshot that carries the deletes (#214).

### Fixed
- Hive partition directory names are percent-encoded, matching official DuckLake byte for byte.
  The previous mapping replaced every character outside `[A-Za-z0-9-_.:]` with `_`, so a partition
  value containing a space, `+`, or `/` produced a directory name official never writes and which
  could not be decoded back to the value — `ts=2024-01-15_12:30:00_00` where official writes
  `ts=2024-01-15%2012%3A30%3A00%2B00`. Partition key names are now escaped on the same terms, as
  official does. Only newly written files are affected: existing files are located by the path
  recorded in `ducklake_data_file`, and pruning has always used the authoritative
  `ducklake_file_partition_value`, so no catalog was damaged and no repair is needed.
- Timezone-aware timestamp writes now record UTC min/max statistics for file pruning; catalogs
  written before this change remain readable with absent bounds. This also lets an identity
  partition key be a `timestamptz` column, which previously failed every `INSERT` with an
  unsupported-value error because the partition value could not be encoded.
- A keyed `DELETE` or `UPDATE` works on a data file that compaction has rewritten. The filtered
  delete path previously refused such a file outright — a v1 scope limit, documented as though
  position resolution depended on `rowid = row_id_start + physical position`. It does not:
  `resolve_positions` reads a file's true physical row positions, and a delete file's `pos` is a
  physical index, which a rewrite leaves meaningful (the rewritten rows sit at `0..n-1`). What a
  rewrite disturbs is the rowid *sequence* — `rewrite_data_files` drops deleted rows, so the
  survivors' rowids carry holes and the catalog records no `row_id_start` at all, making that
  arithmetic not merely unreliable but uncomputable. DuckLake itself performs keyed mutations
  against merged, reordered and partition-merged files, so this brings the crate in line with the
  reference implementation. The consequence for callers: a table can be compacted and still take
  `DELETE`/`UPDATE`/upsert, which previously required choosing one or the other.
- `types::build_read_schema_with_field_id_mapping` declares the `PARQUET:field_id` of every nested
  node the data file tags — list elements, struct children, map key/value, at any depth. A nested
  node's field id is part of its parent's Arrow type, so a read schema that omitted it disagreed with
  the batches the parquet reader produces from the very file it describes ("column types must match
  schema types"). Callers that pair that schema with arrow-rs themselves hit the error directly;
  scans through this crate's `TableProvider` were unaffected, because DataFusion's parquet opener
  casts a metadata-only difference away. Files without nested field ids (external, or written before
  nested nodes were tagged) are still described without them, and the table's catalog schema stays
  free of storage metadata. Dropping the ids on the way out is a relabel rather than a conversion, so
  a scan over a nested column keeps its filter and limit pushdown.
- A write upgrades legacy single‑row `list<T>` metadata to recursive list and element rows without
  changing the existing list column ID or invalidating historical snapshots.
- Reads null‑fill fields added inside structs, including non‑nullable fields and structs nested in
  lists, while preserving field‑ID‑based nested renames and drops.
- The CDC table functions resolve the TABLE and its schema at the window's end snapshot, not at the
  catalog's current snapshot. A window over a table that was dropped afterwards failed with
  "Table 'main.t' not found in catalog" even though every snapshot in the window still had the
  table; it now returns that window's changes, and a window whose end snapshot is past the drop
  reports that the table does not exist at that snapshot — matching official DuckLake. One window
  changes the other way: a window ending BEFORE the table was created used to resolve the table at
  the current snapshot and return an empty feed, and now reports that the table does not exist at
  that snapshot. Official DuckLake errors on that window too, so this is convergence rather than a
  new restriction (#196).
- A `SELECT` over a table whose struct child was added or renamed by DDL no longer fails with
  "Cannot cast nullable struct field … to non-nullable field". DuckLake records such a child as
  non-nullable while the physical parquet node stays optional, and a file written before the change
  does not carry the child at all; `build_read_schema_with_field_id_mapping` now relaxes nested
  nullability exactly as the sibling `build_arrow_schema` does. Map keys stay non-nullable. The read
  schema is then type-identical to the catalog schema, so the rename layer above the scan is a
  relabel and filters and limits reach the parquet reader's pruning on tables with nested columns.

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
