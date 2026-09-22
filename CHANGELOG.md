# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Snapshot SQL listings expose schema version, structured changes, raw tokens,
  and commit metadata (#318).
- `StatsSqlDialect::cte_materialization` lets a dialect declare the statistics
  CTE with its engine's materialization modifier; it defaults to none.
- `MetadataWriter::register_existing_data_files` promotes many existing files, with their delete
  files, in one commit, and can establish the table's partition and sort spec on that same
  snapshot; multicatalog Postgres only (#323).
- `DataFileInfo::with_partition_values` carries a promoted file's partition values for a spec the
  same commit is establishing, whose `partition_id` does not exist yet (#323).
- A batched promote refuses a repeated path, where official's `ducklake_add_data_files`
  deduplicates one: an entry here carries the caller's own metadata and delete file, not a bare
  glob-expanded path, so keeping the first would silently drop the rest (#323, #328).
- `StatsSqlDialect::canonical_integer_text` lets a dialect prove that a stored
  partition value is an integer spelled the one way this crate spells one; it
  defaults to declining, which drops the partition pre-filter for integer keys.
- A file listing is narrowed by identity partition value as well as by
  statistics: a query that pins an identity partition key to a set of values no
  longer reads the statistics of files that key already excludes — roughly a
  twentyfold speed-up on a synthetic PostgreSQL 18 fixture of 8192 files with a
  32-value `IN` predicate (127 ms to 6.8 ms; the ratio reproduces, the absolute
  figures are bound to that fixture). Applies only to a table with a single partition-spec generation, an
  `identity` transform, an equality or `IN` predicate, and a key whose values
  have one textual spelling — string columns, and integer columns whose stored
  spelling the dialect can positively prove canonical. Everything else, including
  a dialect that cannot prove it, pre-filters nothing and is unchanged.
- `ducklake_file_partition_value` gains
  `idx_file_partition_value_table_key (table_id, partition_key_index)`, created
  by every writer's schema bootstrap and by existing catalogs on their next
  initialization.

### Changed

- **BREAKING**: `SnapshotMetadata` adds `schema_version: Option<i64>`; update
  struct literals (#318).
- **BREAKING**: `RenderedColumnFilter` adds
  `cte_materialization: &'static str`; update struct literals.
- **BREAKING**: `RenderedColumnFilter::stats: Vec<&'static str>` becomes
  `projections: Vec<CteProjection>`, which carries the hoisted conversions as
  well as the raw statistics; `RenderedColumnFilter::select_list` renders it.
- **BREAKING**: `StatsFilter` adds `partitions: Vec<PartitionPrefilter>`, filled
  by the new `StatsFilter::with_partition_prefilters` and rendered by
  `StatsFilter::render_partition_prefilters`; update struct literals.
- **BREAKING**: `StatsColumnFilter` adds
  `partition_text: Option<PartitionTextMatch>`, the column's own answer to how
  its values are spelled as text; update struct literals. A partition key is
  pre-filtered only when this and the constant's answer agree, so a constant of
  another type than the key's column declines rather than comparing two
  encodings of different things.
- Snapshot SQL listings expand from two to eight columns; select named columns
  to retain a fixed shape (#318).
- The statistics CTE that narrows a file listing is declared `MATERIALIZED` on
  the engines that accept it (DuckDB, SQLite 3.35+, PostgreSQL 12+), so each
  bound is validated and cast once per file rather than once per comparison per
  file. MySQL has no such modifier and is unchanged.
- The statistics CTE also emits each bound's validity test and cast as an output
  column of its own, so the conversion runs once per file whatever the predicate
  does with it. Materializing the CTE alone did not achieve that: what it
  materialized was the raw text, while the conversion stayed in the join filter.
  Roughly a twentyfold speed-up on the same fixture (2703 ms to 127 ms, with
  the same caveat about absolute figures), and with identical results — the
  emitted statement is the previous one with each conversion replaced by a
  reference to itself.

## [0.8.0] - 2026-09-16

### Added

- Catalog-backed data inlining: supported small writes stay in metadata with stable row IDs and
  snapshot visibility, governed by the inclusive `data_inlining_row_limit` (#272).
- MySQL inlines supported list, struct, and map batches under the row limit (#272).
- Optional declared indexes on inlined tables, without changing their column types; MySQL skips
  them on `LONGTEXT`/`LONGBLOB` columns (#272).
- Catalog-inlined scans push safe equality, range, null, boolean, and prefix filters into
  metadata queries, retaining DataFusion residual filters (#277).
- Pushed-down filters narrow the DuckLake file listing in the catalog query itself, so planning
  a selective scan or keyed mutation no longer lists every live file (#293).
- A pushed-down filter never changes a query's answer, checked by a generated sweep over hostile
  catalog statistics on every backend (#293).
- `ORDER BY col LIMIT n` skips data files whose statistics cannot beat the running Top-N
  boundary, including scans that rename columns or project `rowid` (#316).
- An unfiltered `count(*)`, `min(col)` or `max(col)` folds to a literal from catalog statistics
  (#305, #312).
- Scans and plan-time footer reads share the `RuntimeEnv` Parquet metadata cache (#300).
- SQL `DELETE` commits Parquet-resident and catalog-inlined rows in one snapshot on all four
  write backends, with exact inline-aware truncate counts (#273).
- Scoped DuckLake settings resolve per key with table-over-schema-over-global precedence on
  every backend, governing compression, row groups, rollover, sorting and partition paths (#271).
- Staged writes commit across new or existing tables in one snapshot on all metadata backends;
  empty staging calls publish no snapshot (#274).
- Set supported table-scoped options and coordinate commits on DuckDB, SQLite, MySQL, and both
  PostgreSQL layouts (#274).
- Read nullable snapshot changes and find commits with live files on DuckDB, SQLite, PostgreSQL,
  and MySQL (#274).
- Read-only DuckLake views across metadata backends, with writer-compatible view metadata (#264).
- Per-file DuckLake `map_by_name` mappings across scans, predicate-based mutations, compaction
  rewrites, and change feeds, including typed Hive partition constants (#263).
- Literal column defaults for schema evolution and omitted `INSERT` fields, across metadata
  backends (#259).
- Row lineage and positional deletes take the physical row position from the Parquet reader's
  `row_number` virtual column, so those scans now push predicates into Parquet pruning (#130).
- `DuckLakeTableWriter::with_upload_concurrency` and `DuckLakeWriteOptions::upload_concurrency`
  set how many data files a rolling or partitioned write uploads at once, default 4 (#280).
- `MetadataWriter::register_existing_data_file_with_delete` registers an existing data file with
  an existing positional delete file in one commit; multicatalog Postgres only (#309).
- `DataFileInfo::with_source_row_id_start` carries a registered file's `row_id_start` from its
  source catalog, NULL included, and `with_row_id_floor` lifts the destination's allocator (#313).
- `DataFileInfo::with_owner_catalog` / `DeleteFileInfo::with_owner_catalog` mark a registered
  file as a reference to another catalog's object (#310).
- Multicatalog PostgreSQL indexes `ducklake_file_column_stats` for the planning reads. On a large
  existing catalog, build it with `CREATE INDEX CONCURRENTLY` before upgrading (#294).

### Changed

- **BREAKING**: DataFusion 55 and arrow/parquet 59; consumers must move to the same major
  versions (#282).
- **BREAKING**: default features are now `metadata-sqlite` alone, so a default build no longer
  compiles DuckDB; add `features = ["duckdb-bundled"]` for the DuckDB provider and writer (#304).
- **BREAKING**: `DuckLakeWriteOptions` is non-exhaustive and gains `parquet_version`,
  `auto_compact`, `rewrite_delete_threshold` and `upload_concurrency`; add
  `..Default::default()` to literals (#271, #280).
- **BREAKING**: `DuckLakeFileData` and `DataFileChange` are non-exhaustive and implement
  `Default`; `DataFileChange` gains `mapping_id`, which providers must preserve (#263).
- **BREAKING**: `ScheduledFile`, `DataFileInfo` and `DeleteFileInfo` each gain a public field;
  use the `new()` constructors and builders (#310).
- **BREAKING**: `FileRowNumberExec` and `row_id::row_pos_field` are removed;
  `DeleteFilterExec::try_new` / `RowIdExec::try_new` take a trailing `pos_index: usize` for a
  column carrying `row_id::ROW_NUMBER_EXTENSION_TYPE`; `ROW_POS_COLUMN_NAME` is only a base
  name (#130).
- **BREAKING** (multicatalog Postgres only): a data/delete file row with `owner_catalog_id` set
  references a file that catalog owns, so no reclaim path touches it; the next boot migrates
  additively, existing rows becoming NULL (#309, #310).
- Catalog-backed writes default to Snappy compression, 122,880-row row groups and 512 MB target
  files, changing Parquet layout and file size versus the previous defaults (#271).
- Data inlining stays disabled by default, opt in with `data_inlining_row_limit`; unsupported
  schemas fall back to Parquet, and `UPDATE` and row-lineage scans reject inlined rows (#272).
- Rolling and partitioned writes upload up to 4 files at once, raising peak write memory roughly
  fourfold; `with_upload_concurrency(1)` restores the previous behaviour (#280).
- A custom `ObjectStore` must now be concurrency-safe: one write can issue overlapping
  `put_opts` / `put_multipart_opts` calls (#280).
- A write whose uploads fail issues `DELETE` for that batch's objects, and awaits in-flight
  uploads before returning the first error in file order (#280).
- Staged-file uploads gained a `ducklake.upload_staged_files` span (#280).
- Files carrying a live delete file are now pruned by their statistics, matching official
  DuckLake (#293).
- Compaction reads each merge bin with one parallel scan instead of one execution per source
  file (#279).
- Opening SQLite and multicatalog PostgreSQL writers adds the inline registry required by staged
  writes, replacements, and truncation (#274).
- The multicatalog orphan sweep's cross-catalog arm collects reference rows rather than every
  absolutely-spelled row (#310).
- Bundled DuckDB and the CI CLI move to 1.5.5 (#273).

### Fixed

- Table scans and `COUNT(*)` include scalar rows inlined in SQLite, DuckDB, PostgreSQL and MySQL
  catalogs; unsupported non-scalar inline values report how to flush or disable inlining (#261).
- Inlined positions remain in metadata during `DELETE` and `UPDATE`, and unqualified `DELETE` and
  metadata row counts subtract visible inline positions (#262).
- Delete-file and compaction commits abort when a source file gains a distinct inline position
  after planning, and `merge_adjacent_files` skips files masked by inline positions (#262).
- SQLite and MySQL inline encodings round-trip floats and binary values exactly, and DuckLake
  temporal and binary encodings are preserved in inlined writes (#272).
- Inline commits honour expected-base, commit-metadata and partition-spec fences, preserve
  snapshot-change tokens, and count live rows in DELETE-all (#272).
- MySQL creates inline-table DDL before opening the write transaction, avoiding implicit partial
  commits (#272).
- Reject incomplete inline change feeds, and check the current snapshot before `UPDATE` (#272).
- Return a typed conflict for stale PostgreSQL single-catalog writes (#272).
- Reject non-empty `CREATE TABLE AS SELECT` without publishing metadata; use `CREATE TABLE`
  followed by `INSERT INTO ... SELECT` (#272).
- Float `min_value` and `max_value` are gated on `contains_nan`, so a matching row can no longer
  be pruned away unread by `SELECT` or left behind by `DELETE` (#130).
- Float predicates no longer reach Parquet pruning on a scan of a file carrying a delete file,
  which silently dropped NaN rows (#130).
- The internal physical-position column no longer binds to a catalog column of the same name in
  the CDC feeds (#130).
- Compaction writes with the table's configured Parquet options, instead of re-encoding a merged
  or rewritten file with the writer defaults (#278).
- Compaction re-merges partial files instead of stranding them, so a table taking frequent
  appends no longer accumulates a floor of files no later pass can reduce (#281).
- Expiring snapshots reclaims the per-file and per-commit rows official DuckLake deletes; rows
  orphaned by earlier expires are swept during schema initialization (#295).
- DuckDB compaction no longer skips eligible data files, and retains historical row visibility
  (#273).
- MySQL allocates data and delete file IDs consistently from catalog counters, avoiding
  collisions across append, update, delete, and compaction (#273).
- DuckDB and MySQL mutation flows record `changes_made` entries for every data-modifying
  snapshot (#273).
- A first write after an abandoned staged table commits and seeds table stats instead of failing
  with a permanent `Conflict` (#273).
- Read legacy global schema-version ledgers without assuming per-table provenance (#273).
- Invalid write-only catalog settings no longer block table reads; they fail when a write or
  maintenance operation is planned (#271).
- Legacy two-column `ducklake_metadata` tables migrate both scope columns losslessly (#271).
- Multicatalog catalog-scoped settings deterministically override shared globals (#271).
- ZSTD compression level `0` maps to the Parquet library default; non-ZSTD codecs ignore the
  setting (#271).
- `UPDATE` and `DELETE` retain scoped Parquet options (#271).
- Name-mapped Hive columns retain values through `DELETE`, `UPDATE`, and compaction, and mapped
  CDC reads no longer return NULL after column renames (#263).
- Name-mapped Hive paths follow DuckDB's raw segment parser, including backslash separators and
  invalid multiple-`=` segments; hex integers and exponent-form decimals read compatibly (#263).
- Map keys and nested nullability match Arrow's read-compatible schema requirements (#263).
- Field-ID-less Parquet files and inlined rows apply `initial_default` values, and read schemas
  retain the metadata needed for missing-field adaptation (#263).
- `NULL` sentinels, BLOB decoding, expression-default reads, and legacy schema migration (#259).
- An upload whose final flush failed panicked with "Already shut down" instead of returning the
  error (#280).
- `files_matching` no longer stops pruning a data file once that file carries a delete file
  (#276).
- Multicatalog Postgres: a catalog's own absolute-path file, as `begin_write_to_path` writes, is
  scheduled and reclaimed instead of leaking when it lives outside `data_path` (#310).

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

[Unreleased]: https://github.com/hotdata-dev/datafusion-ducklake/compare/v0.8.0...HEAD
[0.8.0]: https://github.com/hotdata-dev/datafusion-ducklake/compare/v0.7.0...v0.8.0
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
