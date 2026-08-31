# Compatibility & Feature Matrix

This is the authoritative reference for what `datafusion-ducklake` supports: catalog
backends, object stores, types, capabilities, and current limitations. The
[README](README.md) covers getting started; this file covers "does it support X?".

> Status: alpha. APIs and supported surface change as DataFusion and the DuckLake
> spec evolve. See [CHANGELOG.md](CHANGELOG.md) for what shipped when.

---

## Catalog backends

DuckLake stores catalog metadata in a SQL database. Reads are supported on all four
backends below; writes (`INSERT`, `DROP TABLE`, maintenance) are currently implemented
for SQLite and PostgreSQL only. SQL `CREATE TABLE`/CTAS works on the SQLite
single-catalog path; the PostgreSQL path is the experimental multi-catalog layout (see
below) where tables are created via `DuckLakeTableWriter` and then appended to with
`INSERT INTO`.

| Backend    | Read | Write | Multi-catalog | Feature flags                                          |
|------------|:----:|:-----:|:-------------:|--------------------------------------------------------|
| DuckDB     |  ✅  |  ❌   |      ❌       | `metadata-duckdb` (default)                            |
| SQLite     |  ✅  |  ✅   |      ❌       | `metadata-sqlite`, `write-sqlite`                      |
| PostgreSQL |  ✅  |  ✅   |      ✅       | `metadata-postgres`, `write-postgres`, `multicatalog-postgres` |
| MySQL      |  ✅  |  ❌   |      ❌       | `metadata-mysql`                                       |

The listed PostgreSQL features do not select a TLS provider. Plain local connections work without
one. For TLS, also enable one of `tls-native-tls`, `tls-rustls-aws-lc-rs`, or
`tls-rustls-ring`.

PostgreSQL has **two** writers, both behind `write-postgres`:

| Writer | Layout | Spec-compliant | SQL `CREATE TABLE`/CTAS | Read back with |
|--------|--------|:--------------:|:-----------------------:|----------------|
| `PostgresSingleCatalogMetadataWriter` | Standard single-catalog | ✅ | ✅ | `PostgresMetadataProvider` (and any other DuckLake reader, incl. DuckDB) |
| `PostgresMetadataWriter` | Library-specific multi-catalog | ❌ | ❌ | `MulticatalogProvider` only |

Use `PostgresSingleCatalogMetadataWriter` unless you specifically need many
catalogs in one database. It produces the same catalog shape as the SQLite and
MySQL writers: no `catalog_id` columns, no `ducklake_catalog*` map tables, and
unscoped relative paths (`{data_path}/{schema}/{table}/…`).

**Multi-catalog** (PostgreSQL only, **experimental**) lets a single metadata store hold
multiple independent DuckLake catalogs. Reading multiple catalogs requires
`multicatalog-postgres` (`MulticatalogProvider`); creating/managing them requires
`write-postgres` (`MulticatalogManager`).

> ⚠️ The multi-catalog layout is **specific to this library** — it is not part of the
> DuckLake specification and is not (yet) supported or accepted upstream. Catalogs
> written this way are only readable through `MulticatalogProvider`, not as standard
> single-catalog DuckLake stores, so **multi-catalog should be treated as
> experimental** and subject to change. Note also that SQL `CREATE TABLE`/CTAS is not
> available on this path (the first write of a table goes through
> `DuckLakeTableWriter`); `INSERT INTO` works once a table exists.
>
> PostgreSQL writes no longer *require* this path: `PostgresSingleCatalogMetadataWriter`
> writes the standard spec-compliant layout and supports CTAS.

---

## Object stores

| Store                       | Supported | Notes                                              |
|-----------------------------|:---------:|----------------------------------------------------|
| Local filesystem            |    ✅     | Available by default via DataFusion's object store |
| S3-compatible (S3, MinIO)   |    ✅     | Enable `object_store/aws` in the application       |
| Google Cloud Storage        |    ❌     | Not currently wired up                             |
| Azure Blob Storage          |    ❌     | Not currently wired up                             |

Applications configure `object_store` directly for local files, S3, or MinIO, or register another
compatible `ObjectStore` implementation with DataFusion, including an OpenDAL‑backed connector.
DuckLake enables the filesystem and AWS providers only for its own tests and examples.

Enabling `object_store/aws` also enables Ring through its HTTP client. If another dependency enables
AWS‑LC, install the intended process‑wide Rustls `CryptoProvider` before creating TLS clients.

---

## Feature flags

| Feature                  | Description                                                              | Default |
|--------------------------|--------------------------------------------------------------------------|:-------:|
| `metadata-duckdb`        | DuckDB catalog read backend                                              |   ✅    |
| `duckdb-bundled`         | Statically compile & bundle DuckDB (disable for dynamic linking)         |   ✅    |
| `metadata-sqlite`        | SQLite catalog read backend                                              |         |
| `metadata-postgres`      | PostgreSQL catalog read backend                                          |         |
| `metadata-mysql`         | MySQL catalog read backend                                               |         |
| `write`                  | Base write support (INSERT, CTAS, maintenance API); needs a write backend|         |
| `write-sqlite`           | Write to SQLite catalogs (`write` + `metadata-sqlite`)                   |         |
| `write-postgres`         | Write to PostgreSQL catalogs (`write` + `metadata-postgres` + multi-catalog) |     |
| `multicatalog-postgres`  | Read multiple catalogs from one PostgreSQL store                         |         |
| `tls-native-tls`         | Use native TLS for SQLx PostgreSQL connections                           |         |
| `tls-rustls-aws-lc-rs`   | Use Rustls with AWS‑LC for SQLx PostgreSQL connections                   |         |
| `tls-rustls-ring`        | Use Rustls with Ring for SQLx PostgreSQL connections                     |         |
| `encryption`             | Parquet Modular Encryption (PME) reads                                   |         |
| `skip-tests-with-docker` | CI-only: skip tests that require Docker                                  |         |

For dynamic linking against a system `libduckdb`, disable defaults and re-enable just
the read backend: `--no-default-features --features metadata-duckdb` (requires
`libduckdb` installed; set `DUCKDB_LIB_DIR` and `DUCKDB_INCLUDE_DIR`).

---

## Type support

| Category                              |  Status   | Notes                                                        |
| ------------------------------------- | :-------: | ------------------------------------------------------------ |
| Integers / floats / boolean           |    ✅     |                                                              |
| Strings / dates / timestamps          |    ✅     |                                                              |
| Decimal (precision & scale)           |    ✅     |                                                              |
| Geometry                              |    ✅     | Mapped to `Binary` (WKB)                                     |
| Complex / nested (list, struct, map)  | Supported | Recursive types and nullable‑field evolution; no pruning     |

### Column defaults

Parquet scans apply `initial_default` only when a file physically lacks the
column. DataFusion inserts apply literal `default_value` metadata when the SQL
statement omits the column. DuckLake's string sentinel `"NULL"` means no
operational default, and BLOB defaults retain printable bytes while decoding
DuckDB `\xNN` escapes.

DuckDB expression defaults remain metadata-only: they do not prevent table
reads, but DataFusion does not evaluate them. Supply those columns explicitly
on insert. Defaults added to nested children are not materialized yet, and the
table-change, insertion, and deletion functions do not fill defaults for fields
missing from older files.

Writer initialization migrates missing default-metadata columns on SQLite,
MySQL, and PostgreSQL. DuckDB read providers project missing fields as `NULL`.
For a legacy SQLite, MySQL, or PostgreSQL catalog opened through a read-only
provider, run writer initialization with a schema-migration role before reading.

---

## Capabilities

| Capability                                              | Status |
|---------------------------------------------------------|:------:|
| `SELECT` against DuckLake tables                        |   ✅   |
| `INSERT INTO` (table must already exist on the PostgreSQL path) | ✅ |
| `CREATE TABLE AS SELECT` (SQL DDL; SQLite single-catalog only — not on the PostgreSQL multi-catalog path) | 🟧 |
| `DROP TABLE` (via `MetadataWriter`)                     |   ✅   |
| Row-level deletes (Merge-On-Read delete files, read)    |   ✅   |
| SQL `DELETE FROM t [WHERE ...]` (positional deletes + metadata-only truncate; SQLite & PostgreSQL) | ✅ |
| SQL `UPDATE t SET c = e [, ...] [WHERE p]` (rewrite + positional delete, one snapshot; SQLite & PostgreSQL) | ✅ |
| Snapshot-based consistency (bound at catalog creation)  |   ✅   |
| Filter pushdown to Parquet (row-group / page pruning)   |   ✅   |
| Parquet footer size hints (1 read/file instead of 2)    |   ✅   |
| Row lineage (`rowid` virtual column, opt-in)            |   ✅   |
| SQL-queryable `information_schema`                      |   ✅   |
| Read-only DuckLake views on every metadata backend      |   🟧   |
| Table functions (`ducklake_snapshots()`, `ducklake_table_info()`, `ducklake_list_files()`, `ducklake_table_changes()`, `ducklake_table_deletions()`, `ducklake_table_insertions()`) | ✅ |
| Maintenance: expire snapshots, cleanup superseded files, orphan-file reclamation | ✅ |
| Parquet Modular Encryption (PME) reads (feature `encryption`) | ✅ |
| Configurable writer output (compression, row-group sizing) | ✅  |
| Table partitioning — read + file pruning (all backends); `identity` + `year`/`month`/`day`/`hour` transforms (`bucket(N)` tolerated, not pruned) | ✅ |
| Partitioned writes — split into per-partition files in one snapshot, on every writable backend, via `set_partition_spec`/`reset_partition_spec` or `execute_ducklake_sql` (`ALTER TABLE … SET/RESET PARTITIONED BY`). Honoured by SQL `INSERT`, the low-level write entry points, the streaming session, compaction, and promote | ✅ |
| Partitioned `UPDATE` / upsert — the append+delete commit registers every appended file (one per output partition) together with the positional deletes, in one snapshot. A row whose partition-key value changed moves to its NEW partition and keeps its `rowid` lineage | ✅ |
| Multi-catalog (PostgreSQL, **experimental** — library-specific, not in the DuckLake spec) | ✅ |

Maintenance and `DROP TABLE` are driven through the Rust API (`maintenance` module and
`MetadataWriter`), not SQL DDL.

### Views

Every metadata backend reads snapshot-visible rows from `ducklake_view`. Catalogs without that
table expose an empty view set, and each writer creates the official table layout, including
`view_uuid`. Creating, replacing, altering, or dropping views is not supported.

DataFusion plans each stored definition using its recorded dialect. The reader restores DuckLake's
`{DUCKLAKE_CATALOG}` placeholder outside quoted strings and identifiers. For DuckDB
`schema.table` bodies, it resolves only a unique schema visible at the view's `begin_snapshot`
whose `schema_id` remains visible at the requested snapshot, then quotes the canonical schema name.
This supports same-schema, cross-schema, and mixed-case references without letting an external
qualifier retarget a schema created later. Other multipart qualifiers fail with the named view and
dialect error. This disambiguation assumes metadata created through DuckDB's view binder; manually
inserted definitions that violate its ambiguity checks are outside the compatibility guarantee.

A definition that DataFusion cannot plan remains visible in view listings. Such a view has no rows
in DataFusion's `information_schema.columns` until its definition becomes plannable.

View planning uses a private, snapshot-pinned `SessionContext`, rejects DDL, DML, and statement
commands, and propagates the catalog's row-lineage option. Caller-registered UDFs and caller session
settings, including `execution.time_zone`, are not inherited.

---

## Write concurrency

Both write backends use the same **commit-time** model: a write's snapshot id, all its
metadata rows, and its publication are written in **one transaction**, with the snapshot id
assigned at commit (so per-catalog id order == commit order) and nothing visible until that
transaction commits. There are no "dormant" (committed-but-unpublished) rows, so reads never
observe another writer's uncommitted schema, a transient empty table, or a torn generation.
On Postgres multi-catalog the begin step only *reserves* ids (via the IDENTITY sequence) and
reads existing state; it writes nothing.

`WriteMode::Replace` (SQL `INSERT OVERWRITE`, and the first write of a table) is
**abort-on-conflict** under concurrency, matching DuckLake's snapshot isolation:

- **Two concurrent `Replace`s of the same table never silently union.** The first to
  commit wins; the later one — whose base is now stale — aborts with
  `DuckLakeError::Conflict` (retryable by the caller). The check runs at the commit point
  under the catalog lock: a `Replace` aborts if any data file **or** column of the table has
  `begin_snapshot`/`end_snapshot` newer than the catalog head it began on.
- **Column ids are stable** across writes: an unchanged column keeps its `column_id`
  (== parquet field-id); a same-schema `Replace` rewrites no column rows. Only added/removed
  columns are written.

Known edges:

- **`Append` (`INSERT INTO`) is not conflict-checked.** Concurrent appends commute and are
  both retained (matching DuckLake); a *stale* `Append` issued before a concurrent `Replace`
  is not detected. Use `Replace` for overwrite semantics.
- A **fileless same-schema `Replace`** (an empty-table overwrite that writes no data file and
  changes no column) leaves no per-table footprint, so it resolves **last-writer-wins** rather
  than abort-on-conflict (both backends). Data-bearing and schema-changing replaces are
  conflict-checked.
- A **column type change is rejected on a data write** (`Replace` **and** `Append`) — this is
  a **behavior change**: previously a type change on `Replace` was silently dropped (the column
  kept its old type, corrupting reads); it is now a clear error. Schema evolution goes through
  the explicit, widening-only **`promote_column_type`** (it retires the old column version and
  inserts a new one with the **same field-id**, mirroring upstream DuckLake's `ALTER`-vs-`INSERT`
  separation; reads cast old narrow files up to the widened type). A widening refresh should call
  `promote_column_type`, then write under the new type. Add/remove columns on `Replace` still work.
- **Schema evolution is versioned.** A promote leaves two `ducklake_column` rows sharing one
  `column_id` (old retired via `end_snapshot`, new live), matching upstream. On the
  **PostgreSQL multicatalog** layout this is enforced by a composite PK + a partial unique
  index; on the **SQLite single-catalog** layout `ducklake_column` matches upstream's bare
  shape (no PK) and the one-live-version invariant is enforced in the writer + tests. Catalogs
  created by an earlier version are migrated in place on open (idempotent, lossless).
- **`schema_version` is maintained on both write layouts.** SQLite and PostgreSQL both carry
  `schema_version` on `ducklake_snapshot` and a `ducklake_schema_versions` ledger table, bumped
  on a schema change (table create, column add/remove/reorder, type promotion) and carried
  forward on a pure data write — matching upstream's `if (SchemaChangesMade()) schema_version++`.
  Both deliberately omit upstream's `next_catalog_id` / `next_file_id` snapshot columns (this
  library allocates ids from its own counters, never from the snapshot row). This applies to
  **partition ids** too: `ducklake_partition_info.partition_id` is allocated from a
  library-owned counter (`next_partition_id` on SQLite/MySQL, an autoincrement/sequence on
  DuckDB, an IDENTITY on Postgres), *not* from `next_catalog_id` as the current spec assigns
  it. The ids are internally consistent (`ducklake_partition_info` ↔ `ducklake_partition_column`
  ↔ `ducklake_data_file.partition_id` ↔ `ducklake_file_partition_value`), so any reader resolves
  a partitioned table correctly; the caveat is only that a DuckLake/DuckDB *writer* that later
  allocates ids from `next_catalog_id` on the same catalog could collide with these. Because of
  that omission a SQLite catalog is DuckLake-*design*-faithful but **not yet a drop-in DuckDB
  catalog** — DuckDB's writer expects those allocator columns. Full DuckDB write-compat
  (adopting the snapshot allocators uniformly for all ids, partitions included) is a tracked
  follow-up.
- A single `Replace` is assumed to register **one** data file (the current writer path); the
  conflict check is not designed for multiple `register_data_file` calls sharing one base.
- Two concurrent `CREATE TABLE` of the same name on the PostgreSQL multi-catalog path are
  rejected by a unique index, surfacing as a raw database unique-violation rather than a
  clean `Conflict`. A `DROP` racing a write can likewise surface as a raw unique-violation.
- Every commit path is fenced on the table's live partition generation, so a file inconsistent
  with the live spec is never committed — never one stamped with a retired `partition_id`, and
  never a `partition_id`-less file in a partitioned table.
- A `SET`/`RESET PARTITIONED BY` that lands between an `INSERT` being planned and committed aborts
  the `INSERT` with `Conflict`; re-open the catalog and retry, and the retry plans against the new
  spec. The write is never silently re-laid-out under a spec its plan did not see — a concurrent
  layout change is reported, not absorbed.
- The low-level writer entry points have no plan step, so they resolve the live spec inside their
  own write transaction and lay out accordingly; a spec change racing that commit still fences.
- Compaction merges only *within* a partition and carries each output's `partition_id` and values
  over from its sources, including a retired generation — those rows really do have that
  generation's layout, so preserving it keeps them prunable exactly as before.

---

## Limitations

- **Write backends:** DuckDB and MySQL are read-only; writes require SQLite or PostgreSQL.
- **Negative NaN orders differently than in DuckDB.** DuckDB normalizes NaN — `-NaN = NaN`
  is true and either sign sorts above every value. DataFusion compares floats with arrow's
  `total_cmp`, which is sign-sensitive, so `-NaN` sorts *below* every value including
  `-Infinity`. A query like `WHERE x < -1e308` therefore returns `-NaN` rows here and none
  in DuckDB. This is inherited from arrow and affects results, not just pruning; float
  min/max statistics are gated on `contains_nan` in **both** directions to stay consistent
  with it, where official DuckLake gates only the max.
- **No `AS OF` syntax:** select a catalog snapshot by ID with
  `DuckLakeCatalog::with_snapshot`, by UTC timestamp with
  `DuckLakeCatalog::with_snapshot_at`, or select one table per query with
  `ducklake_table_at`. Timestamp selection uses the latest snapshot at or before the
  requested time. For snapshots with the same timestamp, an as‑of or change‑data end
  bound selects the highest ID, while a change‑data start bound selects the lowest ID.
- **One mutation per session, then re-open the catalog.** Because a catalog pins its
  snapshot at creation and never refreshes, a `SessionContext` observes a single generation
  for its lifetime. After an `UPDATE`/`DELETE`/`INSERT` commits, the same session keeps
  reading the pre-mutation state; a `SELECT` won't see the change, a just-inserted row can't
  be updated, and a **second `UPDATE`/`DELETE` re-touching a file the first modified aborts
  with a conflict** (the compare-and-swap that also guards genuine concurrency — it is what
  prevents a stale rewrite from resurrecting superseded rows). Re-open the catalog (or create
  a fresh `SessionContext`) between mutating statements so it binds to the latest snapshot.
- **The change feed is degraded on encrypted (PME) catalogs.**
  `ducklake_table_changes` still works on encrypted catalogs for inserted rows, but a
  range containing an `UPDATE` surfaces its rewritten rows as plain `insert`s rather
  than being correlated into `update_preimage`/`update_postimage`, and `delete` rows
  are missing entirely (the correlated path reads parquet footers/rows the change-feed
  path does not decrypt). A window whose only changes are deletes carries no data file
  to detect encryption from, so it fails at read time on an encrypted catalog rather
  than returning wrong results. Compaction-merged (partial) files whose window overlap
  comes only from `partial_max` are likewise dropped on encrypted catalogs (their
  per-row snapshot column cannot be read). Non-encrypted catalogs emit the full
  official change-set (inserts, deletes, update pre/postimages, merged-file rows at
  their origin snapshots).
- **Change feeds over an encrypted table whose columns evolved are refused.** A change
  feed resolves each data file's columns by field id, read from that file's parquet
  footer, so a column renamed (or dropped and re-added under the same name) after a file
  was written still reads that file's values. The encrypted path holds no key for those
  footers, and the only thing left to match on — the column name — is exactly what the
  rename changed. Rather than hand back another column's values, the feed errors when
  the table's columns changed between the oldest data file in the window and the
  window's end snapshot. A window whose files all predate the change is unaffected, and
  so is reading the table itself. Two details worth knowing:
  - **The check is deliberately conservative, and refuses more than it must.** A column's
    identity is its name plus its resolved type, and the type is what carries a nested
    field's name — so widening a column with `ALTER … TYPE`, or adding or renaming a field
    inside an existing `STRUCT`, also trips it, even though matching those by name would
    have been correct. Adding a new top-level column, and dropping one outright, do not
    trip it: the added name is in no older file, and nobody asks for the dropped one.
  - **Only `ducklake_table_changes` and `ducklake_table_insertions` refuse with that
    explanation.** `ducklake_table_deletions` has no encryption support at all: it reads
    the deleted rows' source data file with no key, so on an encrypted table it fails with
    a raw parquet decryption error rather than a message about column evolution. That feed
    did not work on encrypted tables before either — the failure has moved from read time
    to plan time.
- **Partition pruning covers `identity` + `year` only.** `month`/`day`/`hour`/`bucket(N)`
  partition transforms are read correctly but fail open (files are always kept, never
  mis-dropped); only whole-value (`identity`) and calendar-year ranges prune files.
- **Complex / nested types** have minimal support.
- **DuckDB-encrypted (non-PME) Parquet files** are not supported (only PME).
- **Data inlining: scalar rows are read on every metadata backend.** DuckLake
  inlines `INSERT`s of up to 10 rows into the catalog by default. SQLite,
  DuckDB, PostgreSQL, and MySQL scans honor their snapshot visibility, so
  `SELECT` and `COUNT(*)` include them. Inlined *Parquet‑row* deletes
  (`ducklake_inlined_delete_<table_id>`) are applied by scans, `UPDATE`,
  `DELETE`, and compaction on all four backends; the `rowid` path remains
  unsupported for inlined rows. Non‑scalar inlined columns fail with an
  error that directs users to flush the rows to Parquet or disable inlining at
  write time.
- **The change feed does not surface inlined deletes.** `ducklake_table_changes`
  and `ducklake_table_deletions` read delete *files* added in the window; a
  snapshot whose only change is an inlined Parquet‑row delete emits no `delete`
  rows even though scans at the window's two ends differ. Flush or avoid
  inlined deletes on tables consumed through the change feed.
