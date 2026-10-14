# Compatibility & Feature Matrix

This is the authoritative reference for what `datafusion-ducklake` supports: catalog
backends, object stores, types, capabilities, and current limitations. The
[README](README.md) covers getting started; this file covers "does it support X?".

> Status: alpha. APIs and supported surface change as DataFusion and the DuckLake
> spec evolve. See [CHANGELOG.md](CHANGELOG.md) for what shipped when.

---

## Catalog backends

DuckLake stores catalog metadata in a SQL database. Reads and writes are supported on all
four backends below. DuckDB, SQLite, and MySQL use the standard single-catalog layout. The
PostgreSQL path uses the experimental multi-catalog layout described below.

| Backend    | Read | Write | Multi-catalog | Feature flags                                          |
|------------|:----:|:-----:|:-------------:|--------------------------------------------------------|
| DuckDB     |  ✅  |  ✅   |      ❌       | `metadata-duckdb` (+ `duckdb-bundled`), `write-duckdb` |
| SQLite     |  ✅  |  ✅   |      ❌       | `metadata-sqlite` (default), `write-sqlite`            |
| PostgreSQL |  ✅  |  ✅   |      ✅       | `metadata-postgres`, `write-postgres`, `multicatalog-postgres` |
| MySQL      |  ✅  |  ✅   |      ❌       | `metadata-mysql`, `write-mysql`                        |

The listed PostgreSQL features do not select a TLS provider. Plain local connections work without
one. For TLS, also enable one of `tls-native-tls`, `tls-rustls-aws-lc-rs`, or
`tls-rustls-ring`.

PostgreSQL has **two** writers, both behind `write-postgres`:

| Writer                                | Layout                         | Spec-compliant   | SQL `CREATE TABLE`        | Read back with                                                           |
| ------------------------------------- | ------------------------------ | :--------------: | :-----------------------: | ------------------------------------------------------------------------ |
| `PostgresSingleCatalogMetadataWriter` | Standard single-catalog        | ✅                | ✅                         | `PostgresMetadataProvider` (and any other DuckLake reader, incl. DuckDB) |
| `PostgresMetadataWriter`              | Library-specific multi-catalog | ❌                | ❌                         | `MulticatalogProvider` only                                              |

Use `PostgresSingleCatalogMetadataWriter` unless you specifically need many
catalogs in one database. It produces the same catalog shape as the SQLite and
MySQL writers: no `catalog_id` columns, no `ducklake_catalog*` map tables, and
unscoped relative paths (`{data_path}/{schema}/{table}/…`). The four spec-layout
writers store `data_path`, schema paths, and table paths with a trailing `/`, as
DuckDB does, and initialization appends it to legacy rows so DuckDB can resolve
crate-written files. The multi-catalog layout keeps its paths as given.

**Multi-catalog** (PostgreSQL only, **experimental**) lets a single metadata store hold
multiple independent DuckLake catalogs. Reading multiple catalogs requires
`multicatalog-postgres` (`MulticatalogProvider`); creating/managing them requires
`write-postgres` (`MulticatalogManager`).

> ⚠️ The multi-catalog layout is **specific to this library** — it is not part of the
> DuckLake specification and is not (yet) supported or accepted upstream. Catalogs
> written this way are only readable through `MulticatalogProvider`, not as standard
> single-catalog DuckLake stores, so **multi-catalog should be treated as
> experimental** and subject to change. Note also that SQL `CREATE TABLE` is not
> available on this path (the first write of a table goes through
> `DuckLakeTableWriter`); `INSERT INTO` works once a table exists.
>
> PostgreSQL writes no longer *require* this path: `PostgresSingleCatalogMetadataWriter`
> writes the standard spec-compliant layout and supports SQL `CREATE TABLE` and `INSERT INTO`.

Multi-catalog file ownership is recorded in dedicated tables. A data or delete file
row with no entry in
`ducklake_catalog_data_file_reference` / `ducklake_catalog_delete_file_reference`
is owned by the catalog holding it, which reclaims the object — including a file it
wrote outside its own `cat_{id}/` layout, which `begin_write_to_path` does. A row with
an entry is a *reference* to the named catalog's file, which is how `register_existing_data_file` and its batched sibling
`register_existing_data_files` (with `with_owner_catalog`) let one catalog share another's
files without copying them. Expire and compaction
delete such a row without scheduling its object, and `cleanup_old_files_in_catalog`
does not delete a scheduled object while a reference in any catalog still names it.
Ownership is deliberately not read off the path spelling: an absolute path says only
where the file is, not whose it is. Official DuckLake has no such rule; its maintenance
commands schedule and delete every path the expiring catalog holds.

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

| Feature                    | Description                                                                       | Default   |
| -------------------------- | --------------------------------------------------------------------------------- | :-------: |
| `metadata-duckdb`          | DuckDB catalog read backend                                                       |           |
| `duckdb-bundled`           | Statically compile & bundle DuckDB (disable for dynamic linking)                  |           |
| `metadata-sqlite`          | SQLite catalog read backend                                                       | ✅         |
| `metadata-postgres`        | PostgreSQL catalog read backend                                                   |           |
| `metadata-mysql`           | MySQL catalog read backend                                                        |           |
| `write`                    | Base write support (INSERT, CREATE TABLE, maintenance API); needs a write backend |           |
| `write-duckdb`             | Write to standard single-catalog DuckDB catalogs                                  |           |
| `write-sqlite`             | Write to SQLite catalogs (`write` + `metadata-sqlite`)                            |           |
| `write-postgres`           | Write to PostgreSQL catalogs (`write` + `metadata-postgres` + multi-catalog)      |           |
| `write-mysql`              | Write to standard single-catalog MySQL catalogs                                   |           |
| `multicatalog-postgres`    | Read multiple catalogs from one PostgreSQL store                                  |           |
| `tls-native-tls`           | Use native TLS for SQLx PostgreSQL connections                                    |           |
| `tls-rustls-aws-lc-rs`     | Use Rustls with AWS‑LC for SQLx PostgreSQL connections                            |           |
| `tls-rustls-ring`          | Use Rustls with Ring for SQLx PostgreSQL connections                              |           |
| `encryption`               | Parquet Modular Encryption (PME) reads                                            |           |
| `skip-tests-with-docker`   | CI-only: skip tests that require Docker                                           |           |

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
| Complex / nested (list, struct, map)  | Supported | Recursive types, nullable evolution, inlining; no pruning       |
| Variant                               |    ❌     | Tables with a `VARIANT` column cannot be read                |

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

### Inlined physical storage

Inlined columns preserve DuckLake's physical encodings:

- **DuckDB:** Native unsigned integers, temporal values, and nested columns.
- **PostgreSQL (both writers):** Decimal text for `UInt64`, text for dates
  and timestamps, native `TIME` and `TIME WITH TIME ZONE`, and `BYTEA` for strings
  and binary values. A `TIMETZ` column an earlier release created as `BYTEA` still
  reads. The standard single-catalog writer does not inline: SQL writes and staged
  write transactions (`DuckLakeWriteTransaction`) go to Parquet.
- **SQLite:** Decimal text for `UInt64`, DuckDB's text for floats (`nan`, `inf`,
  `-0.0`) in `VARCHAR` columns, as official stores them because SQLite turns a NaN
  in a `REAL` column into NULL, text for dates and timestamps, and
  DuckDB-compatible nested literals. Binary leaves use escaped bytes. An inlined
  table an earlier release created with `DOUBLE` float columns still reads, and is
  rebuilt with `VARCHAR` float columns holding each stored value's DuckDB text,
  exactly, on its next inlined insert (a `-0.0` such a `REAL` column held already
  reads as `0.0`).
- **MySQL:** `BIGINT UNSIGNED` for UInt64, `FLOAT`/`DOUBLE` for floats,
  `DECIMAL(p,s)` for Decimal128, `DATE`, `TIME(6)`, and `DATETIME(6)` for
  second-to-microsecond timestamps. `DATETIME(6)` supports years in [1000, 9999]
  without session-time-zone conversion. Nanosecond timestamps use `BIGINT`;
  nested values use text. MySQL has no NaN or infinity and no date outside the
  years 1000 to 9999, so a write holding one goes to Parquet instead of inline.
  Where official would inline it, that write then takes new row ids for its
  Parquet file as an official Parquet write does, so later inserts number their
  rows past them; and an inlined-table error official's inline write would raise
  is not raised. `TIMESTAMPTZ` values are stored as the UTC instant without an
  offset.

Dates and timestamps stored as text use DuckDB's spelling, as official's do: a year
before 1 is marked ` (BC)` (`0044-03-15 (BC)`, `0101-01-01 (BC) 00:00:00`) and a year
after 9999 has its extra digits (`12000-01-01`); readers take both, and Arrow's
spelling (`2024-01-01T00:00:00`) earlier releases wrote. Every date and timestamp
DuckDB can hold inlines on every backend but MySQL (see above), as DuckDB's text where
the backend stores text: years far outside chrono's range, and `infinity` and
`-infinity`, which read back as DuckDB's sentinels (`i32::MAX` days, `i64::MAX`
microseconds, negated for `-infinity`). A value DuckDB cannot hold (`i32::MIN` days,
`i64::MIN` microseconds, a nanosecond timestamp before `1677-09-22`) is
refused with an error rather than stored. Inlined bounds of such values compare by value,
`(BC)` years included.

PostgreSQL maps a 16-byte fixed binary column to `UUID` only when its DuckLake
logical type is `uuid`. Other fixed binary values use `BYTEA`. High-level DuckDB
and PostgreSQL writes
with sub-microsecond intervals take the Parquet path; Arrow 59 cannot write that
interval type to Parquet, so those writes still return an unsupported-type error.

Call `MetadataWriter::set_inlined_index_columns(table_id, columns)` to declare
indexes, then `ensure_inlined_indexes(table_id)` to backfill existing physical
tables. New physical tables apply those declarations. Repeated calls are safe
and preserve column types. No index is created without a declaration. Indexes
are a library extension and are optional; no scan-speed guarantee is implied.
Declarations validate strictly when set. Write and ensure paths skip declared
columns that a later schema change removes or renames.

### Inlined scan filters

Catalog queries apply supported equality, range, null, boolean, and prefix
predicates before materializing inline rows. Supported AND terms can push
independently; OR pushes only when every branch is supported. NOT remains a
residual predicate, and DataFusion reapplies all filters for exact results.

Comparisons require a compatible physical encoding. SQLite UInt64 comparisons
normalize both TEXT and VARCHAR decimal values, including mixed padded and
unpadded rows. Unsupported encodings and missing schema-version columns fall
back to residual filtering. Metadata queries read all data columns; output
projection occurs in DataFusion. Index declarations remain optional.

---

## Capabilities

| Capability                                                                                                                                                                                                                                                                                                                       | Status   |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------: |
| `SELECT` against DuckLake tables                                                                                                                                                                                                                                                                                                 | ✅        |
| `INSERT INTO` (table must already exist on the PostgreSQL path)                                                                                                                                                                                                                                                                  | ✅        |
| Non-empty `CREATE TABLE AS SELECT`                                                                                                                                                                                                                                                                                               | Rejected |
| `DROP TABLE` (via `MetadataWriter`)                                                                                                                                                                                                                                                                                              | ✅        |
| Row-level deletes (Merge-On-Read delete files, read)                                                                                                                                                                                                                                                                             | ✅        |
| SQL `DELETE FROM t [WHERE ...]` (positional deletes, inlined deletions of Parquet rows under the scoped `data_inlining_row_limit`, and inlined-row deletes, mixed in one snapshot; metadata-only truncate; SQLite, DuckDB, MySQL, multi-catalog PostgreSQL)                                                                      | ✅        |
| SQL `UPDATE t SET c = e [, ...] [WHERE p]` (Parquet and inlined rows; new versions keep their row ids, inlined under the scoped `data_inlining_row_limit`; one snapshot; SQLite, DuckDB, MySQL, multi-catalog PostgreSQL)                                                                                                        | ✅        |
| Snapshot-based consistency (bound at catalog creation)                                                                                                                                                                                                                                                                           | ✅        |
| Filter pushdown to Parquet (row-group / page pruning)                                                                                                                                                                                                                                                                            | ✅        |
| Filter pushdown into the catalog file listing — per-column statistics narrow the metadata query, so planning a selective scan or keyed mutation does not list every live file                                                                                                                                                    | ✅        |
| Filter pushdown to catalog-inlined rows (equality, range, null, AND, OR, and prefix)                                                                                                                                                                                                                                             | ✅        |
| Parquet footer size hints (1 read/file instead of 2)                                                                                                                                                                                                                                                                             | ✅        |
| Row lineage (`rowid` virtual column, opt-in)                                                                                                                                                                                                                                                                                     | ✅        |
| SQL-queryable `information_schema`                                                                                                                                                                                                                                                                                               | ✅        |
| Read-only DuckLake views on every metadata backend                                                                                                                                                                                                                                                                               | 🟧       |
| Table functions (`ducklake_snapshots()`, `ducklake_table_info()`, `ducklake_list_files()`, `ducklake_table_changes()`, `ducklake_table_deletions()`, `ducklake_table_insertions()`)                                                                                                                                              | ✅        |
| Maintenance: expire snapshots, cleanup superseded files, orphan-file reclamation                                                                                                                                                                                                                                                 | ✅        |
| Parquet Modular Encryption (PME) reads (feature `encryption`)                                                                                                                                                                                                                                                                    | ✅        |
| Configurable writer output (compression, row-group sizing)                                                                                                                                                                                                                                                                       | ✅        |
| Table partitioning — read + file pruning (all backends); `identity` + `year`/`month`/`day`/`hour` transforms (`bucket(N)` tolerated, not pruned)                                                                                                                                                                                 | ✅        |
| Partitioned writes — split into per-partition files in one snapshot, on every writable backend, via `set_partition_spec`/`reset_partition_spec` or `execute_ducklake_sql` (`ALTER TABLE … SET/RESET PARTITIONED BY`). Honoured by SQL `INSERT`, the low-level write entry points, the streaming session, compaction, and promote | ✅        |
| Partitioned `UPDATE` / upsert — the append+delete commit registers every appended file (one per output partition) together with the positional deletes, in one snapshot. A row whose partition-key value changed moves to its NEW partition and keeps its `rowid` lineage                                                        | ✅        |
| Multi-catalog (PostgreSQL, **experimental** — library-specific, not in the DuckLake spec)                                                                                                                                                                                                                                        | ✅        |

Comments and tags are readable on every metadata backend and writable on SQLite and on both
PostgreSQL layouts. `execute_ducklake_sql` handles `COMMENT ON TABLE`, `VIEW`, and `COLUMN`.
As in official DuckLake, schemas take no comments and a column takes only the `comment` key.
A comment bumps `schema_version` and records `altered_table` or `altered_view`, so attached
DuckDB sessions see it.

Maintenance and `DROP TABLE` are driven through the Rust API (`maintenance` module and
`MetadataWriter`), not SQL DDL.

### Parquet writer defaults

Scoped `parquet_version` settings apply to data files and positional-delete
files written by `INSERT`, `UPDATE`, `DELETE`, and maintenance. When the setting
is absent, this crate retains its existing Parquet V2 default; DuckDB `COPY`
defaults to Parquet V1. Set `parquet_version = V1` explicitly when identical
absent-setting behavior is required.

Dictionary encoding and bloom filters follow DuckDB's writer. A data file buffers its first row
group, counts each top-level column, and keeps a dictionary only when the column's distinct values
number at most a fifth of the row group's rows; only a dictionary-encoded column gets a bloom filter
(false-positive rate 0.01), and an all-NULL column gets neither. arrow-rs fixes a file's encoding
once, so later row groups in the same file follow the first one's choice, except that a dictionary
outgrowing the limit falls back to plain encoding partway. Nested columns keep arrow-rs's defaults.
Existing files are unaffected and read as before.

Because the first row group is buffered before it is written, a data file rolls over at
`target_file_size` only once that row group is complete, as DuckDB's does: a file holds at least one
row group, or the whole write when it is smaller. Each open file holds that row group in memory
until it is written, including every open partition file of a partitioned write. SQL writes use
DuckDB's 122,880-row row group; `DuckLakeTableWriter` without `with_max_row_group_rows` uses
arrow-rs's 1,048,576 rows, so its buffered row group is correspondingly larger.

Scoped writer settings and catalog‑backed data inlining are supported. DuckDB, SQLite, MySQL, and
multi‑catalog PostgreSQL support multi‑table Parquet, inline, and delete commits.

A stored table, schema or catalog‑wide `data_inlining_row_limit` decides whether a table gets an
inlined data table and refuses inlined system column names (`row_id`, ...). Where none is stored,
the writer's own limit does, as official's `DATA_INLINING_ROW_LIMIT` attach option does: the
`DuckLakeCatalog::with_write_options` limit for SQL (`10` when unset), a `DuckLakeTableWriter`'s
own limit for its writes (`0` unless set), and official's default `10` for a direct
`MetadataWriter` call. A SQL `INSERT` inlines under the stored limit, else that writer limit, as
official's insert does. A direct `DuckLakeTableWriter` write is this crate's own: it inlines under
the writer's limit alone, whatever the catalog stores.

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

Non-empty `CREATE TABLE AS SELECT` is rejected before any table metadata is
published. Create an empty table, reopen the catalog to see its committed
snapshot, and use `INSERT INTO ... SELECT` to write rows. CTAS support requires
session and object-store handling that the registration path does not provide.

## Write concurrency

Write commits allocate a snapshot and publish their data and final metadata in
one database transaction. Write setup retains each backend's existing schema
and table reservation behavior; PostgreSQL layouts reserve identifiers without
publishing schema or table rows.

`DuckLakeWriteTransaction` finalizes new or existing tables on DuckDB, SQLite,
MySQL, and both PostgreSQL layouts. Every stage must retain its schema and field
IDs. A conflict rejects the whole commit. Snapshot-column staging validates its
inputs before uploading any objects.

A commit that fails with nothing committed removes the data and delete files it
uploaded, as official DuckLake does; this covers `DuckLakeWriteTransaction` and
every `TableWriteSession` path, but not yet the buffered `write_table` /
`append_table` paths, which leave their files to the orphan sweep on any commit
failure. A failure is classified by phase: anything raised
before `COMMIT`, and a `COMMIT` failure on DuckDB or SQLite, means nothing was
committed. One deliberate divergence: when the `COMMIT` statement itself fails on
PostgreSQL or MySQL (`DuckLakeError::CommitOutcomeUnknown`), the server may have
applied it, so the files are kept and left to the orphan sweep. Official removes
them there too, which can leave a committed snapshot naming deleted objects and
every later read of the table failing. Keeping them costs at most unreferenced
objects until the next orphan sweep past its `older_than` cutoff.

Ordinary append stages commute. Callers can explicitly require an unchanged
base snapshot with `expected_base_snapshot_id`, including for append stages.
That optional precondition is stronger than ordinary append isolation. A deletion
committed since that snapshot fails it too: a delete file written or ended, or an
inlined delete. `flush_inlined_data` counts only inlined deletes: as in official
DuckLake, a flush commits over a concurrent delete of a Parquet row.

Empty transactions, zero-row staging calls, and empty delete sets are no-ops.
They add no result entry or snapshot. Staged inline deletes group row IDs by
physical table, ignore missing identities, and exclude rows inserted in the
same snapshot. Separate checks reject concurrent changes, including inline
deletes and flushes. New inline-delete paths leave the gross metadata record
count unchanged; query counts come from the visible rows.

MySQL resolves the inline schema version while holding the commit's counter
lock. It creates physical inline tables through a separate connection so DDL
cannot implicitly commit the metadata transaction. Zero-row inline stages are
rejected before physical DDL. The new DuckDB staged inline DDL is nullable,
matching SQLite and PostgreSQL; older native DDL remains unchanged.

To move existing inline rows into Parquet, run official DuckLake's flush through
`execute_ducklake_sql`:
`CALL ducklake_flush_inlined_data('lake' [, schema_name => 's'] [, table_name => 't'])`.
The call flushes the catalog passed to `execute_ducklake_sql` when it executes, once,
and returns the result rows. The catalog it names, matched case-insensitively as DuckDB
matches it, must be that catalog as registered in the session; another name fails, as
official fails on a catalog it does not know. The function may be qualified only as
DuckDB resolves it (`main.`, `system.`, `system.main.`); `lake.ducklake_flush_inlined_data`
fails as in official. From Rust, call `DuckLakeCatalog::flush_inlined_data`, or
`DuckLakeTableWriter::flush_table_inlined_data` for one table. All return `schema_name`,
`table_name` and `rows_flushed` for each table that had rows; SQL results carry
`rows_flushed` as official's `HUGEINT`, which DuckDB exports to Arrow as
`Decimal128(38, 0)`. The table function form,
`SELECT * FROM ducklake_flush_inlined_data('lake')`, names its catalog in any case as
well, and is registered only by
`register_ducklake_maintenance_functions`, never by `register_ducklake_functions`:
DataFusion plans a table function as a read, so it would commit even in a session
built with `SQLOptions::with_allow_dml(false)`. DataFusion hands a table function its
arguments without their names, so it takes only the catalog name and flushes every
table.

The flush matches official (DuckDB 1.5.5, with DuckDB 1.5.6's `hive_file_pattern`
fix) on every writable backend. One snapshot covers every table and records
`deleted_from_table:<id>` and `inline_flush:<id>`. Each inlined table becomes
Parquet files: one per partition of the spec in effect when its rows were inserted,
in Hive directories unless `hive_file_pattern` is `false`, or for an unpartitioned
table files rolled over at the table's `target_file_size`. Rows are ordered by the table's current sort order,
expression keys included, and then row id. The files hold deleted rows too, with
each row's row id and insert snapshot embedded; each begins at its earliest row's
snapshot and records `partial_max` and the smallest row id, so time travel is
unchanged. Rows deleted while inlined go to a delete file per data file that
records each deleting snapshot. Inlined deletions of Parquet rows merge into a new
delete file; the replaced one is removed and scheduled for deletion. Flushed rows
are deleted from the inlined tables, `record_count` and `next_row_id` are
unchanged, and after the commit the empty superseded inlined tables of every table
the flush considered are dropped, also when it had nothing to write. Tables whose
`auto_compact` is set to anything but exactly `true` are skipped, as official parses
the option. A concurrent
drop, inlined delete, or flush of the same table, a replaced delete file, or a
retired data file aborts the commit with a conflict. A table that cannot be
flushed fails the whole flush, as official's single transaction does; every
object the attempt wrote is removed, and so are those of a commit the catalog
rejected. The deprecated `DuckLakeTableWriter::flush_inlined_data` keeps its
earlier behaviour: it writes the rows it is given to a new file with new row ids
at the flush snapshot and ends them in the inlined tables, recording
`inline_flush`; a later flush of either kind reads the same values at every
snapshot.

As in official, inlined deletions of a data file retired since (by
`rewrite_data_files`, for example) move too, and a file with no delete file gets one
that records each deletion's snapshot. When the current sort order names a column an
older inlined table's schema version lacks, the flush fails with `InvalidConfig`, as
official's binder rejects the column.

Differences from official:

- When the retired file's delete file has ended, official writes a second delete
  file that never ends beside it (duckdb/ducklake#1406, fixed on upstream main by
  duckdb/ducklake#1489, not in 1.5.6), and time travel to the snapshots both cover reads
  duplicated rows and resurrects the ended file's deletions; its change feeds report
  those deletions again at the rewrite's snapshot, and in every window starting after
  the second delete file begins, dated at the first one's begin snapshot. Repro with
  DuckDB 1.5.5
  on SQLite metadata: with the limit at `0`, insert six rows and delete one; set the
  limit to `10` and delete another; run `ducklake_rewrite_data_files` with
  `delete_threshold => 0.0`, then flush; `AT (VERSION => 4)` goes from
  `[1, 2, 3, 4]` to `[1, 1, 2, 2, 3, 3, 4, 4, 5, 6]`. Here the deletions merge into
  the ended delete file, which the merged file replaces with the same end. When the
  merged file would overlap an earlier delete file of the data file (a chain an older
  writer left), or a deletion is later than the delete file's end, the deletions stay
  inlined and every snapshot reads them correctly; if DuckDB later flushes such a
  catalog, it hits official's bug.
- A position both an inlined deletion and the existing delete file record (official
  commits two concurrent `DELETE`s of one row so) keeps the earlier snapshot in the
  merged file. Official keeps the inlined deletion's, so time travel to the snapshots
  between them reads the row again after the flush. Repro with DuckDB 1.5.5, two
  sessions on one catalog with the limit at `5`: insert 100 rows; in one session
  `BEGIN` and delete row 5; in the other delete rows below 20 (a delete file); commit
  the first; after `CALL ducklake_flush_inlined_data`, the snapshot of the delete
  file reads 81 rows, row 5 among them, where it read 80. On such a catalog either
  flush changes the change feeds: windows holding one of the two snapshots report
  the row's deletion before the flush, and after it only those holding the snapshot
  the flush kept, the earlier here and the inlined deletion's in official.
- A catalog whose `encrypted` option is `true` is refused: flush fails with
  `Unsupported` before writing anything. Official writes encrypted Parquet there; this
  crate writes no encrypted files (see "Encrypted catalogs" below).
- Arrow 59 cannot write `Interval(MonthDayNano)` to Parquet, so a catalog holding
  such inlined values fails the flush with an `Unsupported` error and keeps its
  rows inlined.
- A change feed over a window that spans flushed history reports what it reported
  before the flush. Official's does not, an upstream bug (duckdb/ducklake#1572): after a flush, its
  `ducklake_table_changes` over a window holding two `UPDATE`s of one inlined row
  repeats the first update's rows and reports the second as an `insert`. Windows of
  one snapshot agree. Repro with DuckDB 1.5.5: set the limit to `5`, insert one
  row, update it twice, and compare `ducklake_table_changes('lake', 'main', 't', 3, 4)`
  before and after `CALL ducklake_flush_inlined_data('lake')`.
- Official's change feeds date a deletion that a delete file without per-row
  snapshots (the first delete from the file) records against a data file with
  `partial_max` (one a flush or `merge_adjacent_files` wrote) at the deleted row's
  insert snapshot, and over a window holding the insert report it as an
  `update_preimage`/`update_postimage` pair of equal values; and where the file holds
  a row id twice (a row updated while inlined, then flushed), over a window holding
  the deletions of both, they repeat the update pairs and report the later change as
  an `insert` (duckdb/ducklake#1603). A delete file with per-row snapshots, such as a
  flush's, which begins at its earliest deletion, reads right in every window. This
  crate dates each deletion at its own snapshot. Repro with DuckDB 1.5.5: insert two rows inlined, flush, set the limit to
  `0`, delete one row; `ducklake_table_changes` from snapshot 0 reports an update pair
  at the insert snapshot instead of a delete at the last.

`DuckLakeCatalog::checkpoint` runs official `CHECKPOINT`'s six phases in its order
(`ducklake_checkpoint.cpp`), each committing before the next begins: flush; snapshot expiry
of snapshots older than the `expire_older_than` option, skipped when it is unset;
`merge_adjacent_files` up to each table's `target_file_size`; `rewrite_data_files`
at each table's `rewrite_delete_threshold` (default `0.95`); then removal of the
files scheduled for deletion and of unreferenced files older than the
`delete_older_than` option (default `2 days`). Each phase stays callable on its
own. The single-catalog PostgreSQL writer has no expiry or file cleanup, so
`checkpoint` fails there with `Unsupported` before running any phase. A phase that
fails ends the checkpoint with the earlier phases committed, as official's does; an
option is read where its phase uses it, so an empty `delete_older_than` fails at the
cleanup phase. The merge and rewrite phases record `merge_adjacent:<id>` and
`rewrite_delete:<id>` with no commit message, as official's do. Difference from
official: they commit one snapshot per table, where official commits one for the
whole phase. `rewrite_data_files` writes one file per rewritten file where official packs
the surviving rows into fewer files up to `target_file_size`, so the rows match but the
file layout differs (datafusion-contrib/datafusion-ducklake#377). On an encrypted
catalog `checkpoint` skips the phases that write Parquet (flush, merge, rewrite) and
runs expiry and cleanup. Interval options parse as official parses them, with DuckDB's
`Interval::FromString`: fractions (`'1.5 months'` is one month and 15 days), signed
times (`'-00:30:00'`, the text `set_option` stores for `INTERVAL '-30 minutes'`), a
trailing `ago`, and a bare number of seconds; what DuckDB rejects (`'+1 day'`,
`'.5 days'`) fails. An interval that
overflows fails with `InvalidConfig`. A negative `expire_older_than` is accepted, as in
official (its cutoff is in the future, so every snapshot but the latest can expire).
`merge_adjacent_files` numbers the rows of rowid-adjacent sources from the first
source's `row_id_start`, as official does, including a source that embeds row ids (the
new row versions an `UPDATE` wrote). An updated row's `rowid` then changes, and change
feeds report that update as a `delete` and an `insert` instead of an update pair
(duckdb/ducklake#1605): with the limit at `0`, insert a row, update it, insert another,
and merge; the updated row's `rowid` goes from `0` to `1` in both.

Difference from official: a negative `delete_older_than` is refused with
`InvalidConfig`. Its cutoff would be in the future, so cleanup and orphan deletion
would remove the files of commits still in flight, losing their data. Setting the
row limit to zero affects future writes and does not flush existing rows.

Maintenance reclaims inlined storage as official DuckLake 1.5.5 does. Expire
drops a fully expired table's inlined data tables and
`ducklake_inlined_delete_<table_id>` with its registry rows. Inlined rows that no
surviving snapshot can read stay in tables that are still live, as DuckDB 1.5.5
and 1.5.6 keep them. After a flush
commits, the empty superseded inlined data tables are dropped.
`MulticatalogManager::drop_catalog` drops the catalog's inlined storage. Each step
runs in one transaction, except on MySQL, which commits DDL implicitly: there,
expire drops after its metadata commit, and the next expire finishes an
interrupted one. An interrupted superseded-table drop on MySQL can leave an
empty, unregistered table, as official's can. As in official, a registered name is
dropped whatever it is, quoted as an identifier, and physical tables no registry row
names are never reclaimed. A catalog
created before data inlining has no registry, and these steps reclaim nothing
there. A failure of the post-commit drop is returned as an error although the
flush committed.

SQL writes honor the scoped `data_inlining_row_limit`, which defaults to `10` as
official DuckLake's does; the threshold is inclusive, and a table setting overrides a
schema setting, which overrides the global one. Set it to `0` to write every insert to
Parquet. `DuckLakeTableWriter` inlines only under a limit its options set. Unsupported
schemas or writer capabilities fall back to Parquet. As in official, while inlining is
enabled a table cannot be created, or gain a column, named like an inlined data
table's system column (`row_id`, `begin_snapshot`, `end_snapshot`, and DuckDB's
`_ducklake_internal_row_id` / `_ducklake_internal_snapshot_id`; in a DuckLake 1.1
catalog the `_ducklake_`-prefixed `row_id` / `begin_snapshot` / `end_snapshot`
instead), and enabling inlining (`data_inlining_row_limit` above `0`, at table scope or
catalog-wide) over a table that has such a column fails; a table with one keeps
writing Parquet. This crate has no column rename. Inline inserts, deletes, and flushes retain the merged
`inlined_insert`, `inlined_delete`, and `inline_flush` ledger tokens.

As official does, `CREATE TABLE` and each change of a table's columns (add, drop, type
change, partition change) create its inlined data table for the new schema version, on
every backend, unless the limit is `0` or the columns cannot be inlined, and inlined
inserts go to the table's newest inlined data table. A column change made while the
limit is `0` gets no inlined table, so after the limit is raised official's next inlined
insert fails on the older table's columns (duckdb/ducklake#1606); this crate refuses that
insert too, deliberately, until upstream settles the fix. A MySQL insert holding a value
MySQL's inlined columns cannot hold goes to Parquet instead, inlines nothing, and commits.
`CREATE TABLE` writes no table-wide statistics row; the table's first write does.

SQL `UPDATE` and `DELETE` follow official DuckLake 1.5.5 (`DuckLakeDelete::FlushDelete`)
on SQLite, DuckDB, MySQL, and multi-catalog PostgreSQL, with or without a limit, and
a `DELETE` without `WHERE` deletes every row the same way. A statement's deletions
of one data file that number at most the limit go to
`ducklake_inlined_delete_<table_id>`, which is created on first use. Otherwise a
data file whose rows the deletions and its delete file then cover is ended, its
delete file left as it is; a file with a delete file gets a new one carrying every
position with the snapshot that deleted it, the statement's at its own snapshot, and
the replaced one is removed and scheduled for deletion; and any other file gets a
delete file of the new positions. As in official, deletions a file has in
`ducklake_inlined_delete_<table_id>` do not count towards ending it. An `UPDATE`
ends the old versions of inlined rows and inlines its new row versions, keeping
their row ids, when it writes at most the limit's rows; otherwise they go to Parquet
with embedded row ids. Each statement commits in one snapshot whose `changes_made`
lists `inserted_into_table`, `deleted_from_table`, `inlined_insert`, and
`inlined_delete` in official's order. A statement aborts with a conflict, as
official's `CheckForConflicts` does, when a snapshot committed since it read the
table: an `UPDATE`'s new row versions conflict with a drop, alter, or delete, from
Parquet files or inlined data, of the table; deleting Parquet rows with a drop,
alter, compaction (`merge_adjacent`, `rewrite_delete`, or this crate's
`compacted_table`) or insert; and deleting inlined rows or recording inlined
deletions with a drop, alter, inlined delete, flush or insert. The changes are
checked in that order, before any check of the files the statement touches, so an
`UPDATE` of a table dropped since reports `attempting to insert into table`. Every
commit of a `DELETE` or `UPDATE` checks them, whether staged or through
`MetadataWriter::commit_deletes`, `commit_positional_deletes`,
`commit_inlined_deletes`, `commit_truncate` or the `register_data_file*_with_deletes`
commits; a refused commit removes the files it wrote. Such a conflict is not retried. The single-catalog PostgreSQL
writer supports neither statement.

A `DELETE` that writes or rewrites a delete file commits beside an inlined deletion
of other rows of the same data file committed since it read the table, as official
does: its conflict check never compares the two. An inlined deletion of a row the
`DELETE` deletes too, a position of its delete file or any row of a file it drops,
aborts it with a conflict. Difference from official, which commits both, an upstream
data bug: with a row deleted twice, DuckDB 1.5.5 reads other deleted rows of a file
with a delete file back and miscounts `count(*)`; for a dropped file, its change
feeds lose the drop's deletions in windows starting at the inlined deletion. Through
`MetadataWriter::commit_deletes` and `commit_positional_deletes`, a delete file
whose positions are not recorded (`DeleteFileInfo::with_deleted_positions`, which
`DuckLakeTableWriter::write_delete_file` sets) aborts on any such inlined deletion.
In the other order, a `DELETE` or `UPDATE` recording inlined deletions of a data
file's rows commits beside a delete file the file got since it read the table, as
official does: the commit reads that delete file and retries, within the 10 commit
attempts below. One that deletes a row the statement deletes too, or that cannot be
read (an encrypted one among them), aborts it with a conflict, since
official, which commits both, reads a row deleted by both wrong as above. An
inlined deletion of a data file that compaction (`merge_adjacent_files`,
`rewrite_data_files`) retired since the statement read the table aborts it too.
Difference from official, an upstream data bug: its conflict check lets an inlined
deletion through beside a compaction, recorded against the retired file, so the
compacted file still holds the row and every later snapshot reads it again. An
`UPDATE` aborts on a concurrent inlined deletion of the rows it deletes, as
official's insert check does. Inserts are checked the same way: an append or
inlined insert, staged or through `register_data_file*` and `register_inlined_data`,
aborts when the table was dropped, altered or deleted from, from Parquet files or
inlined data, since its base snapshot; concurrent inserts do not conflict. A SQL
`INSERT`'s base is the snapshot it was planned at. A
positional delete committed through `MetadataWriter::set_delete_file` checks the
deletion rules above, so an insert into the table since its base snapshot aborts it.

Difference from official: a statement that rewrites a delete file dates its
deletions to the id the catalog's next snapshot is expected to take when it writes
the file (the snapshot after the latest; on PostgreSQL, the snapshot sequence's next
value), where official dates them to the snapshot after the one it read
(`FlushDeleteWithSnapshots`). When another snapshot takes that id first, the commit is
refused, the deletions are re-dated to the snapshot after the catalog's latest, the
delete file is written again (the refused attempt's file is removed), and the commit
retried after official's backoff (`100 ms × [0.5, 1) × 1.5^attempt`), for at most
official's default `ducklake_max_retry_count` of 10 attempts, after which it fails
with official's "Exceeded the maximum retry count" error. Commits to other tables or
other files do not abort it; a concurrent rewrite of the same delete file,
or the data file ended or dropped, aborts it with a conflict. On PostgreSQL,
snapshot ids come from a sequence that a refused commit and, on multi-catalog
PostgreSQL, every other catalog's commits draw from too: the commit is refused
unless its snapshot takes the id the deletions are dated to, and a retry dates them
to the sequence's next value. Verified against DuckDB 1.5.5 with PostgreSQL
metadata and two sessions, official is wrong twice here
(`ducklake_delete.cpp`, `ducklake_transaction_state.cpp`), both fixed on upstream
main and not in 1.5.6: it commits at the later snapshot with the deletions still
dated to a snapshot that only inserted into another table, so time travel to it
applies them (duckdb/ducklake#1573, fixed by duckdb/ducklake#1584); and its
conflict check compares a delete file's `begin_snapshot` with the base, which a
rewritten file's earliest deletion precedes, so two concurrent `DELETE`s rewriting
one delete file both commit, leaving two live delete files for the data file,
whose rows scans then return twice (duckdb/ducklake#1481, fixed by
duckdb/ducklake#1555).

Table-wide column statistics (`ducklake_table_column_stats`) are kept as official keeps
them. Each commit of Parquet files or inlined rows merges their statistics into the
stored rows as official's `MergeStats` does: NULL counts and NaN flags combine, unknown
once either side is; values that are all NULL leave the bounds alone; a stored row with
neither bound, which official reads as holding no valid value, takes the write's bounds;
otherwise each bound widens. A table's first write creates a row per field it has
statistics for; after that only existing rows are updated, so a column added later
gets none. Deletes and `merge_adjacent_files` leave the rows unchanged, so a
time-travel read through them still covers rows of files ended since. Bounds compare as
official's `RequiresValueComparison` decides: numbers, booleans and `DATE`, `TIME` and
`TIMESTAMP` values by value, everything else as text. Parquet files this crate writes
record statistics per leaf field of a `STRUCT`, `LIST` or `MAP`, and float bounds that
leave NaN out with `contains_nan` set, as official's do. Inlined bounds are DuckDB's
text of each value (`nan`, `TIME`, escaped `BLOB`, strings of any length), with no NaN
flag; `UUID`, `INTERVAL` and other values this crate has no DuckDB text for leave the
bound unknown. File statistics spell `BOOLEAN` bounds `0`/`1`, blob bounds in hex and
`TIME` bounds as text, as official's Parquet statistics do. A file's string and blob
bounds over 256 bytes are truncated as DuckDB's Parquet writer truncates them: a minimum
to its first 256 bytes (back to a character boundary for a string), a maximum to the
last byte in its first 256 below `0x7F` (`0xFF` for a blob), incremented; a file whose
maximum has no such byte keeps no bounds for the column. A file's string bound holding a
NUL byte is kept as official keeps it, except on PostgreSQL, whose text holds none: there
the file records no bound. Official's `extra_stats` (its
`GEOMETRY` and `VARIANT` statistics) are kept as they are.

`ducklake_table_stats.record_count` counts every row ever inserted, as official's
does: inserts and inlined inserts add their rows, and deletes, truncates, `Replace`'s
retirement of the prior rows, flushes, compaction and `retire_appends_since` leave it.
DuckDB answers `MIN`/`MAX` from the table-wide bounds only while `record_count` equals
the live row count, which holds only when no row has left. This crate's own `count(*)`
comes from the files' counts, and `DuckLakeTable::statistics()` reports the row count
and table-wide bounds as inexact.

A row with neither bound, no NaN and no `extra_stats`, while a live file of the column
is not known to hold only NULLs, is unknown: the next write drops it instead of
narrowing it to its own bounds. Releases up to 0.9 wrote such a row when a live file
was all NULL or had no statistics. A row over only all-NULL files is kept, as official
keeps it.

Releases up to 0.9 left inlined rows out of the table-wide bounds. In a catalog where
they inlined rows, run a flush (or `checkpoint`) once, with each table's `auto_compact`
unset or `true` (the flush skips the others), so the bounds cover those rows.

Differences from official:

- `rewrite_data_files` leaves the statistics and `record_count` as they were. The
  pinned DuckDB 1.5.x recomputes them from the remaining files after a rewrite that
  leaves the table free of deletes (`RecomputeGlobalStatsAfterRewrite`), which narrows
  them, and prunes time-travel scans by them (duckdb/ducklake#1377). Repro with DuckDB
  1.5.5: with the limit at `0`, insert `a` = 1..10 and `(50)` in two files, delete
  `a >= 8`, and run `ducklake_rewrite_data_files` with `delete_threshold => 0`; at the
  snapshot of the second insert `count(*)` is 11 but `WHERE a = 9` finds no row.
  Upstream main fixed the time-travel side (duckdb/ducklake#1378, not in 1.5.x); this
  crate keeps the statistics widen-only, so current and time-travel scans stay
  correct, and DuckDB scans for `MIN`/`MAX` after a rewrite rather than answering
  from the statistics. This crate's reader uses the table-wide rows only at the
  current snapshot and never to prune a scan.
- A committed file that records no statistics for a column (one registered without
  them, or a type this crate does not harvest), or values with no bounds other than
  for NaN, leaves the column with no stored row, which reads as unknown. Official
  stores a row with neither bound for a string holding a NUL byte, for a file
  whose string or blob maximum it cannot truncate, and for a float column once a write
  holds only NaN or NULL values of it; here the column then gets no row. DuckDB 1.5.5 reads that row as
  holding no valid value and narrows it to the next write's bounds, after which it
  prunes rows that match; 1.5.6 keeps the row unbounded and finds them. Repro with
  DuckDB 1.5.5: with the limit at `0`, insert `repeat('é', 300)`, then `'a'`;
  `WHERE c0 > 'b'` finds no row.
- After a column is widened (`promote_column_type`, official's `ALTER ... SET DATA TYPE`),
  DuckDB 1.5.5 drops its bounds at the next statistics merge (a flush, or an insert in a
  session that read the table before) and narrows them on the write after, so it
  prunes rows that match; 1.5.6 keeps them, as this crate does. Repro with DuckDB 1.5.5: inline `0`
  into an `INTEGER` column, widen it to `BIGINT`, flush, then insert `5` to Parquet;
  `WHERE c2 = 0` finds no row.
- When a column is widened while the table's inlining limit is `0`, no inlined table is
  created for the new schema version, and official's later inlined inserts go to the older
  inlined table, whose column keeps the narrower type. On DuckDB metadata a value past that
  type fails official's commit; on SQLite metadata the insert commits and every later read
  of the table fails (not filed upstream). Repro with DuckDB 1.5.5: create an `INTEGER`
  column, set the limit to `0`, insert `1`, widen the column to `BIGINT`, set the limit to
  `10`, and insert `9223372036854775807`. This crate's SQLite catalogs commit the insert and read the older
  table at the widened type. The same holds for a partition spec set while the limit is
  `0`: official's flush writes the rows inlined after it unpartitioned, under the older
  table's layout; this crate's flush writes them under the live spec, as its partition
  fence requires of every file it commits.
- Official compares a `FLOAT` file bound against a constant that prints with an exponent
  (`1e20`, `1e-7`, `FLT_MAX`) as a `DOUBLE`, so it drops the file holding the match and
  `WHERE c = 1e20::FLOAT` finds no row (duckdb/ducklake#1604, open on 1.5.6 and main).
  This crate compares through the range of doubles each `FLOAT` spelling reads as and
  returns the row on every backend.
- Official splices a `DOUBLE` constant into its file listing as its shortest text,
  which for a magnitude DuckDB prints without an exponent is fixed-point
  (`0.48024702758086213`). DuckDB parses that as a `DECIMAL`, whose conversion to
  `DOUBLE` is not correctly rounded past 15 significant digits, so the constant reads
  as a neighbouring double and official drops the file holding the match (not filed
  upstream). Repro with DuckDB 1.5.5: with the limit at `0`, insert
  `'0.48024702758086213'::DOUBLE`, then `5.0`; `WHERE d = '0.48024702758086213'::DOUBLE`
  finds no row. This crate splices the constant with an exponent
  (`4.8024702758086213e-1`), which every catalog engine reads as the double it names.
- When one partitioned insert writes the NULL partition's file before a file holding
  values, DuckDB 1.5.5 merges the NULL file first and keeps neither bound, and a later
  write narrows them. Upstream main keeps the values' bounds, as this crate does. Repro with DuckDB 1.5.5:
  partition by `c1`, insert eleven rows of which one has `c1 = 4` and the rest NULL,
  update another row's `c1` to `-8` inlined; `WHERE c1 > 0` finds no row.
- DuckDB 1.5.5's rewrite recomputes the statistics (above) by aggregating each current
  column over every inlined data table, and fails when an older one lacks a column
  added since (duckdb/ducklake#1490, fixed on upstream main by duckdb/ducklake#1577).
  This crate does not recompute them, so the rewrite commits, as upstream main's does. Repro with
  DuckDB 1.5.5: insert eleven rows to Parquet, delete eight, insert one row inlined,
  add a column, and rewrite with `delete_threshold => 0.0`.
- The same aggregate reads an inlined column's maximum as `VARCHAR`, which DuckDB 1.5.5
  renders for the greatest `TIMESTAMPTZ` one millisecond past it
  (`294247-01-10 04:00:54.776806+00`); the statistics merge cannot cast that text back,
  so the rewrite fails. This crate's rewrite commits. Repro with DuckDB 1.5.5: insert
  twenty rows to Parquet, insert `TIMESTAMPTZ '294247-01-10 04:00:54.775806+00'` inlined,
  delete three rows, and rewrite with `delete_threshold => 0.0`.

Opening SQLite and multicatalog PostgreSQL writers creates a missing inline
registry; standard PostgreSQL staged commits create it on demand. SQLite also
migrates legacy non-null change ledgers.

Snapshot-change readers on all backends preserve NULL changes and snapshots
without ledger rows. `ducklake_snapshots()` and `information_schema.snapshots`
expose nullable `schema_version`, the DuckDB-compatible `changes` map, raw
`changes_made`, and optional commit fields. The existing `timestamp` column
remains UTF-8. `SnapshotMetadata` adds only `schema_version: Option<i64>`;
`list_snapshots()` and time travel do not read the change ledger. The SQL
listings reuse `list_snapshot_changes()`, which requires the current
commit-metadata columns.

Commit metadata lookup searches snapshots with live Parquet files; it is not
general idempotency for inline-only, delete-only, or retired commits.

Table settings require a supported option and a live table. Coordination locks
cover the catalog file on SQLite and DuckDB, and the requested identity on
PostgreSQL and MySQL. MySQL's pre-existing global-setting setter remains
unsupported.

`WriteMode::Replace` (SQL `INSERT OVERWRITE`, and the first write of a table) is
**abort-on-conflict** under concurrency, matching DuckLake's snapshot isolation:

- **Two concurrent `Replace`s of the same table never silently union.** The first to
  commit wins; the later one — whose base is now stale — aborts with
  `DuckLakeError::Conflict` (retryable by the caller). The check runs at the commit point
  under the catalog lock: a `Replace` aborts if any data file **or** delete file of the table
  has `begin_snapshot`/`end_snapshot` newer than the catalog head it began on, or an inlined
  delete of the table was committed after it. The PostgreSQL multi-catalog writer
  (`PostgresMetadataWriter`) also aborts when a column of the table changed since that head;
  the other writers do not check columns. A concurrent `DELETE` therefore aborts
  it, as official DuckLake aborts an insert into a table another transaction deleted from.
- **Column ids are stable** across writes: an unchanged column keeps its `column_id`
  (== parquet field-id); a same-schema `Replace` rewrites no column rows. Only added/removed
  columns are written.

Known edges:

- **`Append` (`INSERT INTO`) is checked as official checks an insert.** Concurrent appends
  commute and are both retained (matching DuckLake); an `Append` whose base predates a
  drop, alter, delete or `Replace` of the table aborts with `DuckLakeError::Conflict`.
- A **fileless same-schema `Replace`** (an empty-table overwrite that writes no data file and
  changes no column) leaves no per-table footprint, so it resolves **last-writer-wins** rather
  than abort-on-conflict (all write backends). Data-bearing and schema-changing replaces are
  conflict-checked.
- A **column type change is rejected on a data write** (`Replace` **and** `Append`) — this is
  a **behavior change**: previously a type change on `Replace` was silently dropped (the column
  kept its old type, corrupting reads); it is now a clear error. Schema evolution goes through
  the explicit, widening-only **`promote_column_type`** (it retires the old column version and
  inserts a new one with the **same field-id**, mirroring upstream DuckLake's `ALTER`-vs-`INSERT`
  separation; reads cast old narrow files up to the widened type). A widening refresh should call
  `promote_column_type`, then write under the new type (the single-catalog PostgreSQL writer has
  none). Add/remove columns on `Replace` still work.
- **Schema evolution is versioned.** A promote leaves two `ducklake_column` rows sharing one
  `column_id` (old retired via `end_snapshot`, new live), matching upstream. On the
  **PostgreSQL multicatalog** layout this is enforced by a composite PK + a partial unique
  index; on the **DuckDB, SQLite, and MySQL single-catalog** layouts, `ducklake_column` matches
  upstream's bare shape (no PK), and the one-live-version invariant is enforced in the writer
  and tests. Catalogs
  created by an earlier version are migrated in place on open (idempotent, lossless).
- **`schema_version` is maintained on every write layout.** DuckDB, SQLite, MySQL, and both
  PostgreSQL layouts carry `schema_version` on `ducklake_snapshot` and a
  `ducklake_schema_versions` ledger table. A schema change bumps it and a pure data write carries
  it forward, matching upstream's `if (SchemaChangesMade()) schema_version++`. These branches
  deliberately retain their existing counter or sequence allocation instead of adding the
  separate Group 9 snapshot-allocator change.
- A single `Replace` is assumed to register **one** data file (the current writer path); the
  conflict check is not designed for multiple `register_data_file` calls sharing one base.
- Two concurrent `CREATE TABLE` of the same name on the PostgreSQL multi-catalog path are
  rejected by a unique index, surfacing as a raw database unique-violation rather than a
  clean `Conflict`. An append racing a `DROP` of its table fails with `Conflict` and
  removes its file.
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

- **Catalog-side filter pushdown declines a few type/backend combinations.** A
  pushed-down filter narrows the file-listing query using per-column statistics,
  but the comparison runs in the catalog engine, so some types decline it and
  fall back to listing every file and pruning in memory: `DECIMAL` bounds on
  SQLite (its only fractional numeric is `REAL`, and a `DECIMAL(38, s)` constant
  can carry more significant digits than a double); `inf` / `-inf` bounds on
  SQLite, which reads either as `0.0`; and float magnitudes with a three-digit
  exponent on PostgreSQL and MySQL, where widening the guard to admit `1e+308`
  would also admit `1e+400`, which both engines silently saturate rather than
  reject. A constant whose type differs from a float column's, a float constant
  on a non-float column, a `DECIMAL` constant of smaller scale than its column,
  and `<>` against a float zero push down nothing either. Declining costs pruning
  only and never changes results.

  A `FLOAT` constant is compared in the double domain against the two midpoints
  to its neighbouring floats, so every correctly rounded spelling of a bound
  (`1e+20`, `1.00000002e+20`) compares as the float it names; a strict comparison
  against a zero keeps both zeros, which the catalog engines compare equal.

  Temporal comparisons are made on the encoded text rather than by casting,
  which is what lets them prune on every PostgreSQL version and at nanosecond
  precision — PostgreSQL and MySQL both store microseconds and round a longer
  fraction, which is not order-preserving. Only the canonical encoding is
  compared: a year outside `0000`-`9999`, which `chrono` writes with a sign
  prefix, and a zone offset other than `+00` are declined rather than
  mis-compared.

  Official DuckLake has no equivalent limits because it evaluates every filter
  inside DuckDB even against a foreign catalog, pulling raw statistics rows out
  of it; this crate queries each catalog natively instead, which removes the
  DuckDB dependency at the cost of that dialect fidelity.

- **An identity partition key also narrows the listing, and declines more than
  it accepts.** A predicate that pins an `identity` partition key to a set of
  values restricts the file listing against `ducklake_file_partition_value`
  directly, without reading any statistics. Official does this only for `bucket`
  keys, where hashing leaves statistics no way to prune at all; for `identity`
  its statistics path is already cheap enough, because it runs inside DuckDB.

  The comparison is made on the stored text, so it is pushed down only where a
  value has exactly one spelling. It declines, and prunes nothing, when: the
  transform is not `identity` (`bucket` hashes; `year` is a range and
  `month` / `day` / `hour` are not order-preserving); the table has more than one
  partition-spec generation in its history, since a live file may carry values
  written under a retired key order; the key's column is not an integer or a
  string, because a float, decimal, boolean or temporal value has renderings
  this crate would not write but another writer might (`5.0` for `5`, `1` for
  `true`, a signed year); the predicate's constant is not of the key column's own
  type; or the catalog engine cannot prove that a stored integer is spelled the
  one way this crate spells one. That last case is why the decline is
  per-backend: the proof is a regular expression on PostgreSQL, MySQL and DuckDB
  and a cast round trip on SQLite, and a dialect without one pre-filters nothing.

  A stored value that is present but not that canonical spelling — `007`, `+7`,
  `7.0`, `0x7`, `7e0`, anything with surrounding whitespace, a non-ASCII digit —
  keeps its file, as do a SQL `NULL` value and a file with no row for the key.
  Declining costs pruning only and never changes results.

- **`ducklake_file_partition_value` carries an index this crate creates.**
  `idx_file_partition_value_table_key (table_id, partition_key_index)` is added
  by every writer's schema bootstrap, and by an existing catalog the next time it
  is initialized. It is the index the pre-filter above reads through; without it
  the query scans every table's partition values rather than one table's. The
  table's columns are untouched, so a catalog carrying it stays readable by
  official DuckLake, which creates no indexes of its own on any metadata table.
  Both indexed columns are fixed-width deliberately — `partition_value` is
  unbounded text, and indexing it would put a value's length between a
  partitioned `INSERT` and success.

- **Partitioned writes record partition values as official does.** Hive directories take
  official's key names (`GetPartitionKeyName`): an identity key its column's name, a
  calendar key `year`, `month`, `day` or `hour`, a repeated name suffixed `_col`. A
  directory's value spells each character outside `[A-Za-z0-9-_.:]` as `_` where official
  percent-encodes it, so the catalog path equals the object key on every object store;
  readers prune by `ducklake_file_partition_value`, never by the directory name. A string
  key's value is recorded verbatim there, NUL bytes and any length included, except on
  PostgreSQL, whose text holds no NUL byte: there a partition value with one fails the write. Calendar transforms cover
  DuckDB's whole date and timestamp range, `(BC)` years included, and record NULL for
  `infinity`. An inlined partitioned insert takes row ids in insert order; a Parquet one
  by partition, in the order partitions first appear.
- **Mapped Hive partition paths follow DuckDB 1.5.5 parsing.** The raw object key is
  split on `/` and `\`; a query marker, newline, or second `=` invalidates that path
  segment. Keys are compared without percent decoding, while values are decoded.
- **Reachable mapped numeric literals match DuckDB.** Hexadecimal integers and
  exponent-form decimals are normalized before Arrow parsing. DuckDB rejects boolean
  words such as `on` and compact dates such as `20240102` while registering a file, so
  this crate's acceptance of those forms cannot affect an official-produced catalog.
  Timestamp-offset comparison remains blocked by the percent-encoded object-key issue
  tracked in #288.
- **Nested mapped Hive partition fields are rejected.** Top-level `is_partition`
  fields are materialized on every read path. DuckDB's writer does not produce nested
  partition mappings, but a hand-written or third-party catalog can; this crate returns
  an explicit error instead of reading a physical value or NULL under that shape.
- **`-0.0` keeps its sign in Parquet files this crate writes.** DuckDB's Parquet writer
  dictionary-encodes `-0.0` and `0.0` as one value, so a file official writes with both
  reads every one of them as the zero written first (duckdb/duckdb#25781); this crate's
  files keep each sign.
- **Negative NaN orders differently than in DuckDB.** DuckDB normalizes NaN — `-NaN = NaN`
  is true and either sign sorts above every value. DataFusion compares floats with arrow's
  `total_cmp`, which is sign-sensitive, so `-NaN` sorts *below* every value including
  `-Infinity`. A query like `WHERE x < -1e308` therefore returns `-NaN` rows here and none
  in DuckDB. This is inherited from arrow and affects results, not just pruning; float
  min/max statistics are gated on `contains_nan` in **both** directions to stay consistent
  with it, where official DuckLake gates only the max.
- **PostgreSQL has standard single-catalog and experimental multi-catalog writers.** DuckDB,
  SQLite, MySQL, and single-catalog PostgreSQL use standard metadata; multicatalog PostgreSQL uses
  the library-specific catalog-id layout.
- **No `AS OF` syntax:** select a catalog snapshot by ID with
  `DuckLakeCatalog::with_snapshot`, by UTC timestamp with
  `DuckLakeCatalog::with_snapshot_at`, or select one table per query with
  `ducklake_table_at`. Timestamp selection uses the latest snapshot at or before the requested
  time. For snapshots with the same timestamp, an as-of or change-data end bound selects the
  highest ID, while a change-data start bound selects the lowest ID.
- **`new` and `with_writer` catalogs read the latest snapshot at each lookup.** A statement
  sees its own commits and other writers' commits, as a new DuckDB transaction does.
  DataFusion looks up each table reference on its own, so a commit between two lookups can put
  one statement on two snapshots. Register the catalog with `DuckLakeCatalog::register`, or call
  `register_snapshot_consistency` on the `SessionContext`, to rebuild the tables of a plan,
  including tables behind a view, at the newest snapshot any table of the same catalog
  resolved; without it, a statement over several tables has no single-snapshot guarantee.
  Snapshots of different catalogs are never compared. A table or view that changed in between
  fails the statement, which is safe to retry. A commit by another writer before execution
  makes an `UPDATE`/`DELETE` fail with a conflict. `with_snapshot` and `with_snapshot_at`
  stay fixed at their snapshot, like attaching with `SNAPSHOT_VERSION`.
- **The change feed is degraded on encrypted (PME) catalogs.**
  `ducklake_table_changes` still works on encrypted catalogs for inserted rows, but a
  range containing an `UPDATE` surfaces its rewritten rows as plain `insert`s rather
  than being correlated into `update_preimage`/`update_postimage`, and `delete` rows
  are missing entirely (the correlated path reads parquet footers/rows the change-feed
  path does not decrypt). A window that holds no data file carries none to detect
  encryption from: when the table has delete files (each of which a window may hold
  deletions of, as official reads them), it fails at read time on an encrypted catalog
  rather than returning wrong results. Compaction-merged (partial) files whose window overlap
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
- **Encrypted catalogs: no Parquet writes.** On a catalog whose global `encrypted` option
  is `true`, every operation that would write a Parquet file fails with `Unsupported`
  before writing it, or creating the table it writes to: an insert over the inlining limit (SQL or `DuckLakeTableWriter`),
  `ducklake_flush_inlined_data`, `merge_adjacent_files`, `rewrite_data_files`, and a
  `DELETE` or `UPDATE` that writes a delete file or Parquet rows. Official writes those
  files with Parquet Modular Encryption, which this crate does not write yet. Writes
  that store rows or deletions inlined in the catalog proceed, as official's do;
  `checkpoint` skips the Parquet phases, and registering existing files writes none. As
  in official, only the exact values `true` and `false` are read; another value fails.
- **`DuckdbMetadataProvider` reads metadata tables from the default schema only.** A DuckDB
  catalog attached with `METADATA_SCHEMA` keeps them elsewhere and cannot be read.
- **Data inlining: supported rows are read on every metadata backend.** The DuckDB
  DuckLake extension inlines `INSERT`s of up to 10 rows by default. SQLite,
  DuckDB, PostgreSQL, and MySQL scans honor their snapshot visibility, so
  `SELECT` and `COUNT(*)` include them, in both the DuckLake 1.0 inlined-table
  layout and 1.1's `_ducklake_`-prefixed system columns. Inlined *Parquet‑row*
  deletes (`ducklake_inlined_delete_<table_id>`) are applied by scans, `UPDATE`,
  `DELETE`, and compaction on all four backends. Row-lineage (`rowid`) scans
  include inlined rows with their stable row ids. Inlined rows resolve columns by
  column id against the schema version their inlined table was written with, so
  a renamed column reads its old values and a column dropped and re-added under
  one name does not. Lists, structs, and maps inline on SQLite,
  DuckDB, MySQL, and the multicatalog PostgreSQL writer when every field passes
  the shared type gate. Unsupported schemas fall back to Parquet. Inlined rows containing
  `Interval(MonthDayNano)` that this crate wrote remain readable, but Arrow 59 cannot
  flush that interval type to Parquet (`ducklake_flush_inlined_data` fails with
  `Unsupported`). `INTERVAL` values official DuckDB inlined into a SQLite catalog (stored
  as DuckDB's interval text) fail the scan with `Unsupported`; flush them with DuckDB
  first. A table with a `HUGEINT` or `UHUGEINT` column does not read, inlined or not.
- **Change feeds read inlined rows as official DuckLake does.** Inlined inserts,
  deletes and UPDATEs of inlined rows (paired into `update_preimage` /
  `update_postimage` by row id), and inlined deletions of Parquet rows
  (`ducklake_inlined_delete_<table_id>`) surface in all three feeds, on every
  metadata backend. The feeds keep official's behaviour for a data file with an
  inlined deletion that is later retired whole (for example by
  `ducklake_rewrite_data_files`). That behaviour is a known upstream data bug,
  matched for now because the fix is not small: a window holding both the inlined
  deletion and the retirement reports only the inlined deletion and the rewritten
  file's rows as inserts, and a window holding only the retirement reports every row
  of the file, the one deleted earlier included, so the earlier deletion is reported
  twice across the two windows. Repro with DuckDB 1.5.5: insert 20 rows to Parquet,
  delete one inlined at snapshot `S`, rewrite with `delete_threshold => 0.0` at
  `S+1`; `ducklake_table_changes` over `[S, S+1]` reports the deletion and 19 spurious
  inserts, and over `[S+1, S+1]` the deletion again with 19 update pairs
  (duckdb/ducklake#1571). When the inlined deletion was an `UPDATE`'s, official reports the
  rewritten file's other rows as update pairs with identical pre- and postimages at the
  rewrite's snapshot (not filed upstream; repro with DuckDB 1.5.5: insert 20 rows to
  Parquet, update two of them inlined, rewrite with `delete_threshold => 0.0`). This
  crate reports the same pairs there, except for a file a flush wrote, where it reports
  no change. As in
  official, whose `ducklake_table_changes` left-joins the window's insertions and deletions
  on snapshot and row id, a row that shares that key with several rows of the other side
  is reported once per such row: a retirement that reports an earlier deletion of a row
  again, at a snapshot that also deletes and re-inserts it, gives two preimages and two
  postimages. A sibling this crate does not match (not filed upstream): when the rewritten
  file also had a delete file, official's `ducklake_table_changes` and
  `ducklake_table_deletions` report each row of an inlined deletion or inlined update
  (pre- and postimage) twice in a window holding both the change and the rewrite's
  snapshot; its `ducklake_table_insertions` and this crate report each once. Repro with
  DuckDB 1.5.5: insert 20 rows to Parquet, delete 12 of them to a delete file, delete one
  row and update another inlined, then rewrite with `delete_threshold => 0.0`. On
  encrypted (PME) catalogs, where the feed reads inserts only, inlined inserts surface as
  `insert` rows; inlined deletes do not surface there.
