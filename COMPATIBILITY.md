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

**Multi-catalog** (PostgreSQL only, **experimental**) lets a single metadata store hold
multiple independent DuckLake catalogs. Reading multiple catalogs requires
`multicatalog-postgres` (`MulticatalogProvider`); creating/managing them requires
`write-postgres` (`MulticatalogManager`).

> ⚠️ The multi-catalog layout is **specific to this library** — it is not part of the
> DuckLake specification and is not (yet) supported or accepted upstream. Catalogs
> written this way are only readable through `MulticatalogProvider`, not as standard
> single-catalog DuckLake stores. PostgreSQL writes currently go through this path, so
> **all PostgreSQL write support should be treated as experimental** and subject to
> change. Note also that SQL `CREATE TABLE`/CTAS is not available on this path (the first
> write of a table goes through `DuckLakeTableWriter`); `INSERT INTO` works once a table
> exists.

---

## Object stores

| Store                       | Supported | Notes                                              |
|-----------------------------|:---------:|----------------------------------------------------|
| Local filesystem            |    ✅     | Available by default via DataFusion's object store |
| S3-compatible (S3, MinIO)   |    ✅     | Register with `RuntimeEnv::register_object_store`  |
| Google Cloud Storage        |    ❌     | Not currently wired up                             |
| Azure Blob Storage          |    ❌     | Not currently wired up                             |

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
| `encryption`             | Parquet Modular Encryption (PME) reads                                   |         |
| `skip-tests-with-docker` | CI-only: skip tests that require Docker                                  |         |

For dynamic linking against a system `libduckdb`, disable defaults and re-enable just
the read backend: `--no-default-features --features metadata-duckdb` (requires
`libduckdb` installed; set `DUCKDB_LIB_DIR` and `DUCKDB_INCLUDE_DIR`).

---

## Type support

| Category                         | Status | Notes                                          |
|----------------------------------|:------:|------------------------------------------------|
| Integers / floats / boolean      |   ✅   |                                                |
| Strings / dates / timestamps     |   ✅   |                                                |
| Decimal (precision & scale)      |   ✅   |                                                |
| Geometry                         |   ✅   | Mapped to `Binary` (WKB)                        |
| Complex / nested (list, struct, map) | 🟧 | Minimal support; many cases return errors      |

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
| Table functions (`ducklake_snapshots()`, `ducklake_table_info()`, `ducklake_list_files()`, `ducklake_table_changes()`, `ducklake_table_deletions()`, `ducklake_table_insertions()`) | ✅ |
| Maintenance: expire snapshots, cleanup superseded files, orphan-file reclamation | ✅ |
| Parquet Modular Encryption (PME) reads (feature `encryption`) | ✅ |
| Configurable writer output (compression, row-group sizing) | ✅  |
| Table partitioning — read + file pruning (all backends); `identity` + `year`/`month`/`day`/`hour` transforms (`bucket(N)` tolerated, not pruned) | ✅ |
| Partitioned writes — split into per-partition files in one snapshot (SQLite; DuckDB/Postgres/MySQL not yet wired), via `set_partition_spec`/`reset_partition_spec` or `execute_ducklake_sql` (`ALTER TABLE … SET/RESET PARTITIONED BY`) | 🟧 |
| Multi-catalog (PostgreSQL, **experimental** — library-specific, not in the DuckLake spec) | ✅ |

Maintenance and `DROP TABLE` are driven through the Rust API (`maintenance` module and
`MetadataWriter`), not SQL DDL.

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
- An `INSERT` reads the table's partition state at plan time and fences on it at commit time,
  in both directions: if a concurrent `SET`/`RESET PARTITIONED BY` changes that state between
  planning and commit, the `INSERT` aborts with `Conflict` (re-open the catalog and retry)
  rather than committing files inconsistent with the live spec — never a file stamped with a
  retired `partition_id` (spec retired mid-flight), and never a `partition_id`-less file in a
  table that became partitioned mid-flight.

---

## Limitations

- **Write backends:** DuckDB and MySQL are read-only; writes require SQLite or PostgreSQL.
- **No SQL-level time travel:** a catalog is bound to a single snapshot. You *can* select
  the snapshot programmatically — `DuckLakeCatalog::with_snapshot(provider, snapshot_id)`
  binds to a specific one, and querying another point in time means creating another
  catalog. What's missing is SQL-level historical querying (`AS OF`) within one handle.
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
- **Partition pruning covers `identity` + `year` only.** `month`/`day`/`hour`/`bucket(N)`
  partition transforms are read correctly but fail open (files are always kept, never
  mis-dropped); only whole-value (`identity`) and calendar-year ranges prune files.
- **Complex / nested types** have minimal support.
- **DuckDB-encrypted (non-PME) Parquet files** are not supported (only PME).
- **Data inlining (SQLite backend): now read.** DuckDB inlines small INSERTs into the
  catalog instead of Parquet; on SQLite these are now unioned into scans (snapshot
  visibility honored), so `SELECT` / `COUNT(*)` are correct. Not yet on the DuckDB /
  PostgreSQL / MySQL backends, for inlined *Parquet-row* deletes, the `rowid` path, or
  non-scalar inlined column types.
