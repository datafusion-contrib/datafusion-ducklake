# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `DuckLakeTableWriter::with_upload_concurrency` and
  `DuckLakeWriteOptions::upload_concurrency` — how many finished data files a rolling
  or partitioned write uploads at once, defaulting to `DEFAULT_UPLOAD_CONCURRENCY`
  (4). Uploads are collected in write order, so `row_id_start`, per-file statistics
  and partition labels are identical at any setting (#280).

- Add read-only DuckLake views across metadata backends and writer-compatible view metadata (#264).

### Changed

- **BREAKING**: `DuckLakeWriteOptions` gained an `upload_concurrency` field. It is a
  plain `pub` struct, so an exhaustive struct literal — or a destructuring pattern
  without `..` — stops compiling with "missing field `upload_concurrency`". Add
  `..Default::default()` to the literal, or set `upload_concurrency: None` to keep the
  writer default. No catalog or data migration is needed (#280).

  Note this struct has now taken a breaking field addition more than once, and will
  again for the next write option. Marking it `#[non_exhaustive]` does not fix that on
  its own — for a struct it forbids struct expressions from other crates entirely,
  including the `..Default::default()` form every caller currently uses — so a lasting
  fix means pairing it with setter methods on the options type. That is a broader API
  decision than this change should make, and is left to the maintainers.

  ```rust
  let options = DuckLakeWriteOptions {
      target_file_size: Some(target_bytes),
      ..Default::default()
  };
  ```

- A write uploads up to 4 finished data files concurrently instead of one at a time,
  raising peak **write memory roughly fourfold** on the paths that use it.
  `object_store`'s `BufWriter` holds a 10 MiB buffer and keeps up to 8 multipart parts
  in flight, so one upload of a large file peaks near 90 MiB and four near 360 MiB —
  **per write session**, with no process-wide cap, so concurrent sessions multiply it.

  Which paths are affected is narrower than it may appear, and worth checking against
  your own usage rather than assuming. In-crate, the concurrent upload helper is
  reached only from a streaming `begin_write` session (roll enabled) and from a
  partitioned write's sink. SQL `INSERT`, compaction, and an unpartitioned SQL
  `UPDATE` all still upload one file at a time and are unaffected. An embedder that
  drives `begin_write` and streams batches — the shape a bulk load takes — gets both
  the speedup and the memory cost.

  Restore the previous behaviour with `DuckLakeTableWriter::with_upload_concurrency(1)`
  on the writer you construct, or pin the previous release. Note that
  `DuckLakeWriteOptions { upload_concurrency: Some(1), .. }` only takes effect if the
  writer is built through `with_options`. The same multiplier applies to request rate —
  up to 32 in-flight part uploads instead of 8 — worth checking against a rate-limited
  store, gateway or proxy (#280).

- A custom `ObjectStore` implementation now sees several `put_opts` /
  `put_multipart_opts` calls from a single write overlapping in time. An
  implementation that assumed this crate wrote one object at a time — a throttling or
  ordering-sensitive wrapper, or a test double keyed on call sequence — must be made
  concurrency-safe, or the writer set to `upload_concurrency: Some(1)` (#280).

- A write whose uploads fail removes the objects from that batch, issuing `DELETE`
  against the object store to do so. A writer role granted only put/list permissions
  logs `failed to remove a data file after an aborted upload batch` and leaves the
  orphans exactly as before; grant delete on the data prefix (`s3:DeleteObject`) to get
  the cleanup. On a versioned bucket the removal leaves a delete marker rather than
  reclaiming the bytes — including for files whose upload failed and which may never
  have existed, since the cleanup covers every path in the batch. The cleanup covers a failure **within the upload batch** only —
  objects from a write that uploaded everything and then failed to commit metadata are
  still left behind (#280).

- A failing write can surface its error later than before. Uploads already in flight
  when the first error appears are awaited rather than dropped, because dropping a
  future mid-multipart strands upload state that only a bucket lifecycle rule could
  reclaim. Uploads that have not yet started are skipped, so the cost is bounded by
  the concurrency setting rather than by the size of the batch. Cleanup of the objects
  that did land is issued as a single batched delete rather than one request per file.
  The first error in file order is the one returned; any others are logged (#280).

- Staged-file uploads are wrapped in a new `ducklake.upload_staged_files` tracing span,
  and the existing per-file `ducklake.upload_staged_file` spans now overlap in
  wall-clock time. Summing their durations no longer approximates a write's upload
  time (#280).

### Fixed

- An upload whose final flush failed panicked with "Already shut down" instead of
  returning the error. `BufWriter::shutdown` leaves the writer shut down even when it
  fails, and `BufWriter::abort` panics in that state, so the abort the writer ran on
  every upload error turned a recoverable object-store failure — an expired credential,
  a rejected `CompleteMultipartUpload` — into a panicking task. The abort decision now
  lives inside the upload helper, which aborts a copy failure (where it releases the
  multipart upload) and returns a flush failure as `DuckLakeError::Io`. No error type
  or variant changed; code that observed this as a task panic now sees an ordinary
  `Err`. The fix covers every upload the writer performs — data files, delete files and
  the single-file write path — not only the concurrent one (#280).

- `files_matching` prunes a data file that carries a delete file. Its recorded
  bounds are marked inexact for readers that apply deletes, and the pruning view
  uses only exact ones, so the first mutation to write a delete file stopped that
  file contributing usable bounds to every later call — permanently, since
  compaction skips delete-bearing files. Callers of `files_matching` read physical
  rows without applying deletes, where the spec's lower/upper bounds still hold,
  so they are restated as usable on that path alone; scan statistics are
  unchanged. The live row count is withheld rather than restated, being the one
  figure deletes change, and promoting it alongside a physical null count would
  let a file be judged entirely null and dropped while still holding matching
  rows (#276).

- Table scans and `COUNT(*)` include scalar rows inlined in SQLite, DuckDB,
  PostgreSQL, and MySQL metadata catalogs. DuckLake inlines inserts of up to 10
  rows by default, so affected queries previously omitted them without warning;
  unsupported non‑scalar inline values now report how to flush or disable
  inlining (#261).

## [0.7.0] - 2026-08-15

### Added
- Sort order: `ALTER TABLE … SET`/`RESET SORTED BY (col [ASC|DESC] [NULLS FIRST|LAST])`, recorded in `ducklake_sort_info`/`ducklake_sort_expression` and applied to insert, `UPDATE` rewrites, and compaction output so per-file statistics tighten (#206, #211).
- `PostgresSingleCatalogMetadataWriter` — writes the **standard, spec-compliant** single-catalog DuckLake layout on PostgreSQL (no `catalog_id` columns, no `ducklake_catalog*` map tables, unscoped relative paths), so Postgres catalogs are interchangeable with the SQLite/MySQL backends and DuckDB's `ducklake` extension. SQL `CREATE TABLE AS SELECT` works on this path, unlike the multicatalog writer (#231).
- Snapshot time travel: `DuckLakeCatalog::with_snapshot_at` and `ducklake_table_at()` select a snapshot by id or timestamp (#236).
- Recursive `list`, `struct`, and `map` columns use standard `ducklake_column` parent links and matching Parquet field IDs across reads, writes, schema evolution, rewrites, and compaction (#230).
- Partitioned writes on **every** writable backend: compaction, the low-level `write_rows`/`write_table`/`append_table` entry points, and `register_existing_data_file` (promote) all honour the table's live spec, and every commit path is fenced against it (#213).
- Atomic append+delete commits accept SEVERAL appended data files: `MetadataWriter::register_data_files_with_deletes` (and its conditional `_and_commit_metadata` sibling) register N data files plus M positional delete files in ONE snapshot, matching the reference implementation. A keyed mutation therefore works on a partitioned table and on a write that rolled past `target_file_size` (#214, #223).
- SQL `UPDATE` on a PARTITIONED table: each rewritten row is routed by its own post-assignment key values, so an assignment that changes a partition key moves the row to its new partition (calendar transforms included). A rewrite spanning several partitions writes one file per partition and commits them all — with the positional deletes — in one snapshot, preserving every row's `rowid` lineage (#239).
- Snapshot metadata and write preconditions: one DuckLake change row per committed snapshot, optional author/message/opaque extra info, and conditional writes fenced against table generation changes, on SQLite and PostgreSQL (#209).
- A streaming write rolls a new data file once the current one passes `target_file_size` (512 MiB default, floored at 4096 bytes) and commits them all in one snapshot, matching official DuckLake; `begin_write_single_file` opts out for sessions finished with `finish_with_deletes` (#224).
- Targeted rewrites: `rewrite_data_files` accepts caller-selected live files without a delete threshold, and streams sort output through DataFusion's spill-capable operator (#211).
- `DuckLakeTable::files_matching` — the data files a predicate could match, pruned by exactly the catalog statistics and partition bounds a `SELECT` with the same filter uses, so a caller driving its own per-file work no longer has to open every data file to find the ones holding a key. Pruning is fail-open and files are read in bounded pages (#240).
- `DuckLakeTable::file_has_embedded_rowid` is now public, and available without the write features. It reports where a row's `rowid` comes from — the file's embedded row-id column when it has one, `row_id_start + physical position` otherwise (#258).
- `column_size_bytes` is populated per column on write (summed from the parquet footer, no extra I/O), and `compute_column_stats` is exposed for callers that already hold a parsed footer (#201).
- Tracing spans over the write path — `ducklake.begin_write_transaction`, `ducklake.register_data_files`, `ducklake.finalize_snapshot`, `ducklake.write_session_finish`, `ducklake.upload_staged_file` (#252).

### Changed
- **BREAKING**: `ducklake_table_changes`, `ducklake_table_insertions` and `ducklake_table_deletions` resolve each data file's columns **by field id**, as of the window's end snapshot, instead of by current name — matching official DuckLake. Output changes, silently, on any table whose columns were renamed or dropped and re-added: a renamed column returned NULL for rows in files written before the rename and now returns those rows' values; a column dropped and re-added under the same name returned the DROPPED column's values and now returns NULL there; two columns whose names were swapped returned each other's values and now return their own; a field renamed inside a `STRUCT` behaves the same way as a top-level one.

  The fix is **not retroactive**: nothing in the catalog was damaged, so no repair tooling is needed — but anything derived from earlier feed output on an affected table is still wrong. **Re-derive anything built from change-feed output on a table whose columns were renamed, or dropped and re-added.** To find those tables:

  ```sql
  SELECT table_id, column_id, count(*) AS generations
  FROM ducklake_column GROUP BY table_id, column_id HAVING count(*) > 1;

  SELECT table_id, column_name, count(DISTINCT column_id) AS ids
  FROM ducklake_column GROUP BY table_id, column_name HAVING count(DISTINCT column_id) > 1;
  ```

  Field ids come from each file's parquet footer, so the feeds now read one footer per data file — the insert-only feed previously read none. On a table with **encrypted** files the footers cannot be read, so a feed whose window spans a rename or a drop-and-re-add is refused with an error rather than served by name; see COMPATIBILITY.md (#253).
- **BREAKING**: PostgreSQL metadata features no longer select a TLS provider. Consumers that need TLS must also enable one of `tls-native-tls`, `tls-rustls-aws-lc-rs`, or `tls-rustls-ring`. Without one, SQLx rejects connections that require TLS and may try plaintext when `sslmode` prefers TLS. No catalog or data migration is needed (#247).
- **BREAKING**: S3 support is no longer selected by the library. Consumers using `object_store::aws` must enable `object_store/aws` in their application or register another `ObjectStore` implementation with DataFusion. Local filesystem support remains available through DataFusion without extra configuration. No catalog or data migration is needed (#247).
- **BREAKING**: DataFusion is depended on with `default-features = false` (only `parquet`, `recursive_protection`, and `sql`). Consumers relying on a DataFusion feature that used to arrive transitively must enable it themselves (#265).
- **BREAKING**: the minimum supported Rust version is now 1.94, the floor set by sqlx 0.9 (#205).
- `TableWriteSession::finish_with_deletes` no longer refuses a session that produced more than one appended file; it commits them all in the snapshot that carries the deletes (#214).
- `TableChangesTable`, `TableInsertionsTable` and `TableDeletionsTable` accept the table's columns through a new `with_columns` builder. No signature changed; without it the columns are read from the metadata provider on each scan (#253).
- The multicatalog Postgres writer sends per-column statistics as one `UNNEST` insert per table instead of a statement per column, removing a round trip per column from every commit (#252).

### Fixed
- `ducklake_table_deletions` silently missed deletions, or emitted the wrong row's content and rowid, whenever DataFusion parallelized its scans: `DeletedRowsExec` inherited the data scan's partitioning, so the optimizer inserted round-robin repartitions and the per-stream offset counted arrival order rather than physical position. It now reports single partitioning, keeps its internal scans away from the optimizer, and matches deleted rows by true physical position (#178, #200).
- Float pruning is NaN-aware. Catalog float min/max exclude NaN while NaN sorts above every value, so a file whose NaN state was unknown or positive could be wrongly pruned on `x > C` while holding matching rows. Stored float maxima are now gated on `contains_nan = false` at every consumption point, and NaN-unsafe predicates no longer reach the parquet reader's row-group and page pruning (#203).
- `decimal(P)` with `P > 38` maps to `Decimal256` instead of an invalid `Decimal128`, which could panic or truncate on decode; and a parquet file carrying two columns with the same `field_id` drops both from the field-id map — the reader null-fills instead of binding the wrong column on the renamed-column read path (#193, #198, #202).
- Timezone-aware timestamp writes now record UTC min/max statistics for file pruning; catalogs written before this change remain readable with absent bounds (#260).
- A keyed `DELETE` or `UPDATE` works on a data file that compaction has rewritten. The filtered delete path previously refused such a file outright — a v1 scope limit documented as though position resolution depended on `rowid = row_id_start + physical position`. It does not: `resolve_positions` reads a file's true physical row positions, and a delete file's `pos` is a physical index, which a rewrite leaves meaningful. A table can now be compacted and still take `DELETE`/`UPDATE`/upsert, which previously required choosing one or the other (#258).
- PostgreSQL `commit_compaction` persists partition metadata, so a merged or rewritten file of a partitioned table keeps its `partition_id` and `ducklake_file_partition_value` rows. Reads stayed correct because zone maps still prune, which is what made this quiet — partition-value pruning was permanently gone while queries still returned the right rows (#246).
- PostgreSQL `register_data_file_with_deletes` persists partition metadata, so the append+delete (update/upsert) path no longer leaves an appended file that can never be partition-pruned again (#225).
- Multicatalog data paths are scoped per catalog: each catalog's root is stored on the registry and resolved for writer, reader, and maintenance paths, with the global metadata path kept as a migration fallback (#266).
- Pruning survives missing statistics. An absent per-file bound is now a typed null, so one file without statistics no longer makes a column unusable for the whole candidate set; files with unknown bounds are kept and exactly-non-matching files are still dropped (#250).
- Conjunctive pruning predicates are applied repeatedly over bounded pages of file metadata, so partition pruning can expose usable range statistics without loading the full file list (#207).
- A data file the catalog records as holding exactly zero rows is dropped before statistics are consulted, saving a pruning pass and closing the residual case where such a file carries no per-column statistics row at all and defeats pruning on that column for the whole page. Only a recorded count of exactly 0 counts as proof; an unset `record_count` keeps the file (#244).
- `types::build_read_schema_with_field_id_mapping` declares the `PARQUET:field_id` of every nested node the data file tags — list elements, struct children, map key/value, at any depth. A nested node's field id is part of its parent's Arrow type, so a read schema that omitted it disagreed with the batches the parquet reader produces from the very file it describes ("column types must match schema types"). Scans through this crate's `TableProvider` were unaffected; callers pairing that schema with arrow-rs themselves hit the error directly (#249).
- A `SELECT` over a table whose struct child was added or renamed by DDL no longer fails with "Cannot cast nullable struct field … to non-nullable field". DuckLake records such a child as non-nullable while the physical parquet node stays optional; nested nullability is now relaxed exactly as the sibling `build_arrow_schema` does, and map keys stay non-nullable (#253).
- Reads null-fill fields added inside structs, including non-nullable fields and structs nested in lists, while preserving field-ID-based nested renames and drops; and a write upgrades legacy single-row `list<T>` metadata to recursive list and element rows without changing the existing list column ID or invalidating historical snapshots (#230).
- The CDC table functions resolve the TABLE and its schema at the window's end snapshot, not at the catalog's current snapshot. A window over a table that was dropped afterwards failed with "Table 'main.t' not found in catalog" even though every snapshot in the window still had the table; it now returns that window's changes. A window whose end snapshot is past the drop — or before the create — reports that the table does not exist at that snapshot, matching official DuckLake (#196, #253).

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

[Unreleased]: https://github.com/hotdata-dev/datafusion-ducklake/compare/v0.7.0...HEAD
[0.7.0]: https://github.com/hotdata-dev/datafusion-ducklake/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/hotdata-dev/datafusion-ducklake/compare/v0.5.0...v0.6.0
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
