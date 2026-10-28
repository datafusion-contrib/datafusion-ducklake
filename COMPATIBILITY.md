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
DuckDB does, and `migrate` appends it to legacy rows so DuckDB can resolve
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

### Catalog layout version and migration

Writers migrate only catalogs this crate created, and only when asked. **This differs from
official DuckLake**, which records the DuckLake version it wrote in `ducklake_metadata.version`
and migrates from it when attached with `AUTOMATIC_MIGRATION`.

- A catalog this crate creates records its own layout version in `ducklake_metadata`, under
  `datafusion_ducklake_version` (single-catalog layouts) or
  `datafusion_ducklake_multicatalog_version` (PostgreSQL multicatalog). It never writes
  official's `version`. The crate's layout versions follow its own releases, not official's.
  Beside it, `datafusion_ducklake_min_version` (or
  `datafusion_ducklake_multicatalog_min_version`) records the oldest layout version whose
  release may still read and write the catalog. A release raises it only for a change an older
  release would read, write or clean up wrongly; an addition older releases can ignore, such
  as an index or a nullable column whose NULL keeps the old meaning, leaves it. It never goes
  down. Official has one `version` that must match exactly; it has no minimum.
- Opening a writer reads those rows and official's `version` and runs no DDL. A catalog at the
  current layout version opens as it is. `initialize_schema` (and `new_with_init`,
  `initialize_multicatalog_schema`, `MulticatalogManager::connect`) creates a catalog in a
  database without one; on an existing catalog it migrates nothing either. A creation that
  stopped before recording the layout version is completed by `migrate`.
- A catalog an earlier release created, or one at an older layout version, is refused with
  `DuckLakeError::MigrationRequired` until the writer's `migrate` function runs:
  `SqliteMetadataWriter::migrate`, `DuckdbMetadataWriter::migrate`,
  `MySqlMetadataWriter::migrate`, `PostgresSingleCatalogMetadataWriter::migrate` /
  `migrate_pool`, or `migrate_multicatalog_schema`.
- A catalog a newer release wrote opens as a current one when its minimum version is at most
  this release's layout version, so a rolling deploy or a rollback across that release keeps
  working; `migrate` leaves it unchanged. It is refused when its minimum version is above this
  release's, or when it records none.
- Writers write and migrate only catalogs this crate created. A catalog official DuckDB created
  (it has `version` and no crate layout version) is refused, and nothing changes it. A
  single-catalog writer also refuses a multicatalog store, and the reverse.
- Official DuckDB can read and write a catalog this crate created (see *Official DuckDB can
  write a catalog this crate created* below), but neither engine migrates the other's
  catalogs: official sees no `version` on one this crate created and applies none of its own
  migrations to it.
- Read providers check the version once, before their first catalog read, as official checks
  once per `ATTACH`; `MulticatalogProvider` checks when it is built. A reader refuses only a
  catalog a newer release wrote whose minimum version is above this release's. A catalog at an
  older or no layout version, or one official DuckDB created, reads as it is, where official
  refuses any version but its own.
- `MulticatalogManager` checks the store as a writer does, once, before the first method that
  changes the store or reads what file cleanup deletes by, however it was built.
- An open provider, writer or manager is not checked again. A release that raises the minimum
  version needs every older release stopped before its `migrate` runs.
- `migrate` is idempotent. On SQLite, DuckDB and MySQL it runs its steps in order and records
  the layout version last, so an interrupted run is completed by running it again; on
  PostgreSQL it applies in one transaction, for both layouts. Two migrations racing on one
  catalog may fail one of them; a retry finds the catalog current. Two processes creating the
  same new catalog at once may see it half-created and be refused once; a retry finds it
  current, or, if the creator stopped, `migrate` completes it.

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
| `restore`                  | `DuckLakeTable::restore_rows_to` (library extension); needs a write backend       |           |
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

A writer's `migrate` adds missing default-metadata columns on SQLite, MySQL, and
PostgreSQL. DuckDB read providers project missing fields as `NULL`. For a legacy
SQLite, MySQL, or PostgreSQL catalog opened through a read-only provider, run the
writer's `migrate` with a schema-migration role before reading.

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
| `CREATE TABLE [IF NOT EXISTS] … AS SELECT` via `execute_ducklake_sql` — rows and table metadata in one snapshot, nullable columns as in DuckDB, an empty result publishes no data file; SQLite, DuckDB, MySQL, and single-catalog PostgreSQL                                                                                     | ✅        |
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
| Keep a table's catalog reads across queries (`CatalogReadMemo`, opt-in, not DuckLake behavior: see Limitations)                                                                                                                                                                                                                   | 🟧       |
| Keep a table's delete positions across scans (`DuckLakeReadOptions`, opt-in)                                                                                                                                                                                                                                                     | ✅        |
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

`DuckLakeWriteOptions::data_page_row_count_limit`, `data_page_size_limit` and `write_batch_size`
set arrow-rs's data page limits for every data file a write, `UPDATE`, flush, merge or rewrite
produces; unset, arrow-rs's defaults apply (20,000 rows, 1 MiB, 1,024 values). Official DuckLake has
no such option, and stores none: they are writer settings, not catalog ones. The files remain
ordinary Parquet with an offset index and page-level statistics, which DuckDB reads as any other.

Scoped writer settings and catalog‑backed data inlining are supported. DuckDB, SQLite, MySQL, and
multi‑catalog PostgreSQL support multi‑table Parquet, inline, and delete commits.

Every write path resolves one `data_inlining_row_limit` per table, as official's
`DataInliningRowLimit` does: the table's stored setting, else its schema's, else the catalog-wide
one, else the limit configured on the writer, as official's `DATA_INLINING_ROW_LIMIT` attach option
applies, else official's default `10`. The writer's limit is the metadata writer's own
(`with_data_inlining_row_limit` on every writer, `MetadataWriter::data_inlining_row_limit`), which
`DuckLakeWriteOptions::with_data_inlining_row_limit` on a `DuckLakeTableWriter` or
`DuckLakeCatalog::with_write_options` overrides. The resolved limit decides whether a write is
inlined (SQL `INSERT`, `UPDATE` and `DELETE`, every `DuckLakeTableWriter` write and streaming
session, keyed mutations), whether a new schema version gets an inlined data table, and whether
inlined system column names (`row_id`, ...) are refused, for direct `MetadataWriter` calls too
(`register_data_file*`, `register_existing_data_files`, `promote_column_type`,
`set_partition_spec`, `set_sort_spec`, ...). `MetadataWriter::table_data_inlining_row_limit`
returns it. `set_table_setting`, `set_schema_setting` and `set_global_setting` store the table,
schema and catalog-wide (on multi-catalog PostgreSQL, catalog-scoped) settings on every writable
backend; a table that needs a column named like a system column is created under a limit of `0`
(the writer's, or its schema's), and this crate stores no setting for it, as official stores none:
official never inlines a table with a column named like a system column (`CanInlineColumns`), so
it creates no inlined data table for it either. Every option is cast and stored as official's
`ducklake_set_option` stores it, and a value it refuses is refused: booleans are cast to
`BOOLEAN` (`'T'`, `'yes'`, `'1'`) and stored as `true`/`false`; `data_inlining_row_limit`,
`parquet_row_group_size` and `parquet_compression_level` are cast to `UBIGINT` (DuckDB's
`VARCHAR` cast: `'1.5'` is `2`, `'1e2'` is `100`, a negative value fails);
`target_file_size` and `parquet_row_group_size_bytes` are parsed as DuckDB memory sizes (a
unit is required) and stored in bytes; `parquet_version` is stored as `V1`/`V2`;
`parquet_compression` is lowercased and checked; `rewrite_delete_threshold` is stored as C++
prints a double (`0.500000`). This crate takes values as text and casts `hive_file_pattern` and
`require_commit_message` as official casts a text value of them: to an integer, stored as `true`
when not zero (`'2'` is on, `'0.4'` is off, `'true'` is refused). `delete_older_than` and
`expire_older_than` must be empty or an interval. Every option official knows is accepted at
catalog, schema and table scope, as official accepts it; an unknown one is refused. Stored
`UBIGINT` values are read with the same cast, booleans are on only for exactly `true`, and
setting keys are stored lowercased, as official stores them. A writer's `migrate` rewrites a
`sort_on_insert` or `hive_file_pattern` that earlier releases stored as a case variant of
`true`, which they read as on, to `true`.

Every insert path reads `sort_on_insert` as official's `PlanInsert` does: the table's stored
setting, else its schema's, else the catalog-wide one (`MetadataWriter::table_setting`), on only
while it reads exactly `true`. Official has no session setting outranking it, so the option a
`DuckLakeTableWriter`, `DuckLakeCatalog::with_write_options` or
`DuckLakeTable::with_write_options` carries applies only where none is stored
(`DuckLakeTableWriter::table_sort_on_insert`). SQL `INSERT` reads it once, when it plans the
statement, as official does: the plan sorts every row when it reads it on and every key is a bare
column, and otherwise the writer lays the rows out at execution under the value the plan read.
Exactly one of them sorts. A streaming session reads it when it begins. `upsert_by_keys` lays out
its new rows as an insert does: official has no upsert, and it is checked against official's
`DELETE` followed by `INSERT`.

A sort key names a column as DuckDB's binder names it: the column spelled exactly as the key,
else the one equal to it ignoring ASCII case, so a key DuckDB stored as typed (`SET SORTED BY
(ID)` over a column `id`) orders every insert, flush and compaction. `execute_ducklake_sql` and
`SortField::column` store a key as DuckDB writes a column reference, as typed and quoted only when
it must be (`"My Col"`, a keyword such as `"name"`). Two columns equal ignoring case, neither
spelled as the key, fail the write (DuckDB refuses such a table). Keys of a dialect other than
exactly `duckdb` are skipped, as official skips them. A key naming a column the table lacks fails
an insert with official's binder error ("Columns in the SET SORTED BY statement were not found in
the DuckLake table. Unmatched columns were: ...") whenever official plans the sort: with
`sort_on_insert` on, or off while inlining is enabled, also when the rows are inlined or there are
none. `set_sort_spec` refuses such a key with the same error. As official refuses `DROP COLUMN`
of a column a sort key references, a write that drops one (a `Replace` without the column) is
refused ("Cannot drop column "v" - the table is sorted by this column. ..."); reset or change
the sort order first. A qualified reference (`t.v`, a struct field `s.x`) names its last part, as
DuckDB's binder names it, and a key this crate cannot parse refuses every drop, since it might
reference the column. A catalog another writer leaves with such a key fails the insert, as does
a key with a qualified reference ("Unexpected qualified column reference - only unqualified
columns are supported"). Each `SET SORTED BY` and `RESET SORTED BY`, a `RESET` of an unsorted
table too, commits a snapshot that bumps `schema_version` and records `altered_table`, with no
`ducklake_schema_versions` row, as official's do.

Sort keys that are expressions (written by DuckDB) are evaluated with DataFusion; where its
semantics differ from DuckDB's (integer division, overflow, type promotion, function behaviour),
rows may be ordered differently inside files. Query results are unaffected. If an expression can't
be evaluated (DataFusion cannot plan it, such as a DuckDB macro or `main.list_value`, or it fails
over the rows being written), rows keep their arrival order (a flush: row id order) and the write
proceeds. A function DuckDB wrote qualified by its default schema (`main.abs(...)`) resolves by its
name, and a key naming a column the table lacks still fails as in official. Bare float column keys
sort as DuckDB sorts them (every `nan` after `inf`, `-0.0` equal to `0.0`). Compaction orders its
output by the same keys, evaluated the same way; when they can't be evaluated it writes the rows in
the order it writes an unsorted table's.

A streaming session (`begin_write`, `begin_write_single_file`) holds its rows while they number at
most the limit and commits them inlined when it finishes there, as official's insert inlines a
statement of at most the limit's rows; the first batch past the limit writes them to Parquet, so a
session holds at most the limit's rows in memory. `finish_with_deletes` commits inlined rows with
its deletes in one snapshot. `begin_write_with_embedded_rowid` and `begin_write_to_path` never
inline: their rows carry caller-chosen row ids or a caller-chosen file. A session over a type or
value the backend cannot inline writes Parquet. Rows a caller registers through
`MetadataWriter::begin_write_transaction` and `register_data_file` are files the caller wrote and
are never inlined.

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

`CREATE TABLE … AS SELECT` runs through `execute_ducklake_sql` when its name,
resolved with the session's default catalog and schema, names the DuckLake
catalog the call runs against; any other CTAS goes to DataFusion. It plans the
query on the session, derives nullable columns from its schema, and commits the
rows and the table metadata in one snapshot through the session's object store
with the same write options as `INSERT`. Columns are named as DuckDB names
them: a column list such as `CREATE TABLE t (a, b) AS …` renames the leading
columns, and without one a repeated name, compared ignoring case, takes a `_1`,
`_2`, … suffix. `IF NOT EXISTS` is a no-op on an existing table or view. A
table another transaction creates under the name while the query runs makes the
CTAS conflict, as official refuses the create. `OR REPLACE` and a column list
with types fail closed. Every column and nested child is nullable, as in
official. `SessionContext::sql` still rejects a non-empty CTAS before
publishing metadata, because DataFusion hands the materialized rows to a
synchronous registration hook that has no object store; an empty CTAS through
it keeps creating the empty table.

## Write concurrency

Write commits allocate a snapshot and publish their data and final metadata in
one database transaction. Write setup reserves the schema and table identifiers
of a new table and the commit inserts their rows, as official DuckLake does, so
a write that fails after setup leaves no row. A commit that finds the name
taken by a table another transaction created conflicts instead of writing into
it.

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
`merge_adjacent_files` (and so `checkpoint`) keeps every row's `rowid`: its output embeds
the row id column, and a merge of rowid-adjacent files also keeps the first file's
`row_id_start`, as official does since duckdb/ducklake#1611. The DuckDB extension the
parity tests pin predates that fix and renumbers the rows of such a merge by position.

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

Every write honors the scoped `data_inlining_row_limit`, which defaults to `10` as
official DuckLake's does; the threshold is inclusive, and a table setting overrides a
schema setting, which overrides the global one, which overrides the writer's limit. Set it
to `0` to write every insert to Parquet. Unsupported schemas or writer capabilities fall back
to Parquet. As in official, while inlining is
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

Migrating a SQLite or multicatalog PostgreSQL catalog creates a missing inline
registry; standard PostgreSQL staged commits create it on demand. Migrating SQLite
also converts legacy non-null change ledgers.

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
  created by an earlier version are migrated in place by the writer's `migrate`
  (idempotent, lossless).
- **`schema_version` is maintained on every write layout.** DuckDB, SQLite, MySQL, and both
  PostgreSQL layouts carry `schema_version` on `ducklake_snapshot` and a
  `ducklake_schema_versions` ledger table. A schema change bumps it and a pure data write carries
  it forward, matching upstream's `if (SchemaChangesMade()) schema_version++`. These branches
  deliberately retain their existing counter or sequence allocation instead of adding the
  separate Group 9 snapshot-allocator change.
- **Official DuckDB can write a catalog this crate created.** It applies to DuckDB, SQLite,
  MySQL, and single-catalog PostgreSQL metadata. The multi-catalog PostgreSQL layout is
  library-specific and stays unattachable by official.
  - *Layout:* `ducklake_schema`, `ducklake_table`, `ducklake_data_file`, and
    `ducklake_delete_file` have official's columns, order and keys, since official inserts into
    them by position and keeps one `ducklake_table` row per version of a table. The writers
    create every table official reads, write `parquet` as a file's format, and give schemas and
    tables v7 UUIDs, rows the migration fills in included.
  - *Ids:* schema, table, partition, and sort ids come from official's `next_catalog_id`
    namespace and file ids from its `next_file_id` one. Each commit raises the watermarks of the
    snapshot it writes past every id in use, and a commit that inlines rows advances
    `next_file_id` once per table, as official's does. Only `migrate` raises those of a
    committed snapshot: a head an earlier writer left without them, so official can attach. A column the
    crate adds to a table takes an id past the table's, as official numbers a table's columns.
  - *Conflicts:* a table id reserved when a write begins is checked at its commit, which
    conflicts when official took it meanwhile, for another table or one of the same name. A write
    whose base snapshot was expired meanwhile commits when it creates the table, which the writer
    that reserved the id knows; an `Append` to a table that existed at its setup and has no row
    left conflicts, as the table may have been dropped meanwhile.
  - *Paths:* a file the crate writes goes to the directory the table's stored path names, which
    a rename by official keeps; a table's first file goes under its schema's stored path, which
    official may have stored under another name.
  - *Migration:* `migrate` only adds what an older catalog lacks, so a release before 0.10
    keeps writing it, and the writers take ids past the counters and sequences older releases
    allocate from. `migrate_to_official_layout` then rebuilds it into official's layout
    (idempotent, lossless, blocking the tables' writers while it runs). On PostgreSQL the rebuilt
    tables keep their owner, grants and indexes, and a table a view depends on is refused; on
    MySQL 8.0 reordering columns copies the table (`ALGORITHM=COPY`); a write it interrupts
    fails with a retryable `Conflict`, and the migration retries a pass a write deadlocked. Afterwards only 0.10 or
    later should write it: an older release fails to create a table (and, on DuckDB, to add a
    file), and its snapshots lack the watermarks official reads until the next 0.10 commit.
  - *Privileges:* opening a writer on a current catalog runs no DDL, so a role that only reads
    and writes the catalog tables can open one. On PostgreSQL, file ids default from the
    `next_file_id` counter (`ducklake_next_file_id()`), which takes no privilege beyond the
    tables'. A catalog not yet migrated keeps the identity sequences a release before 0.10
    takes file ids from; writing it once official wrote it takes `UPDATE` on them, and without
    it the commit names the grant.
  - Official DuckDB 1.5.5's MySQL connector cannot append to an existing table on any MySQL
    catalog, its own included.
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

## Row-level APIs over inlined data

These APIs let a caller change and read rows without knowing whether they are inlined in the
catalog or stored in Parquet; each handles both as official DuckLake's SQL does.

- **Keyed mutations** (`DuckLakeTable::keyed_mutation(keys)`, on SQLite, DuckDB, MySQL and
  multi-catalog PostgreSQL): `delete_by_keys` deletes the rows whose key equals a staged row's
  (`DELETE ... WHERE (k) IN (SELECT k FROM staged)`), `update_by_keys` gives each matched row its
  staged row's values, matched to the table's columns by name, and keeps its `rowid` (`MERGE ...
  WHEN MATCHED THEN UPDATE SET *`; of staged rows sharing a key, the last one's values win, as a
  small `MERGE` source updates in source order, and the row counts once), and `upsert_by_keys`
  deletes the matched rows and appends every staged row in one snapshot. Matches come from
  Parquet files, pruned by the catalog statistics of the key columns and read `concurrency` files
  at a time, and from inlined rows. Deletions are recorded as a SQL `DELETE` records them, new
  rows inlined or written as a SQL `INSERT` or `UPDATE` writes them, and the commit runs the
  conflict checks and retries of a SQL `DELETE` or `UPDATE`. A NULL key matches nothing. Every
  column is found by name, so a `delete_by_keys` batch may hold the key columns alone. These are
  typed Rust APIs, so staged types are strict rather than cast as SQL casts them: each staged key
  and value column must have its table column's Arrow type or a lossless widening of it (a wider
  integer of the same signedness, or a signed one wider than an unsigned; `Float32` to `Float64`;
  a decimal with at least the integer digits and scale; any of the string types for another,
  likewise the binary types and `FixedSizeBinary` to binary; `Date32`/`Date64`, a `Date64` that
  is not a whole day refused; a timestamp of the same time zone at the same or a finer unit;
  lists, structs and maps whose children widen under these rules, field ids and metadata
  ignored). A nullable staged column is accepted for a `NOT NULL` one, a NULL written there
  failing with `NOT NULL constraint failed`. Any other type fails the mutation with
  ``InvalidConfig("column `id`: expected Int64, got Utf8")`` before anything is written. Keys
  compare in the table's type after widening: exactly, and floats numerically at any depth
  (`-0.0` equals `0`, NaN equals NaN), as DuckDB's `DISTINCT` compares them. Refusing a type
  changes no catalog data, so official DuckLake reads every table these APIs write unchanged.
  `upsert_by_keys` lays its rows out as official's insert does under the table's stored
  `sort_on_insert`; `update_by_keys` sorts none, as official's `UPDATE` and `MERGE` do not.
- **Positional deletes** (`DuckLakeTable::delete_positions`, `delete_rowids`): delete given
  positions of live data files, or the live rows with given `rowid`s wherever they are, as a SQL
  `DELETE` records them: at most the limit's positions of a file go to
  `ducklake_inlined_delete_<table_id>`, and a file they empty stays live, as official's does;
  more go to a delete file, which replaces the file's live one carrying every position with its
  snapshot, and a file it would empty is ended instead. `DuckLakeTableWriter::write_delete_file`
  with `MetadataWriter::set_delete_file` still writes a delete file unconditionally.
- **Row lookup by `rowid`** (`DuckLakeTable::locate_rowids`, `read_rows_by_rowid`,
  `read_columns_by_rowid`): where each live row is (inlined, or a data file and position) and the
  rows themselves, reading only the files that hold them. `read_columns_by_rowid` reads only the
  projected columns, under the table's current schema as a scan reads them, and returns each
  row's `rowid` with it, in the order the ids were requested; a row the table does not hold at
  its snapshot is left out, as official's `SELECT ... WHERE rowid IN (...)` leaves it out. A file that embeds `_ducklake_internal_row_id` (a flush, merge, rewrite
  or `UPDATE` output) takes its row ids from that column, as official's reader does, even when
  the catalog records a `row_id_start`; `DuckLakeTable::file_row_id_layout` tells which a file is,
  from its footer, and `DuckLakeTableFile::may_embed_row_ids` gives the catalog's hint.
- **Change detection** (`change_detection::table_data_changes`, `table_rows_changed`,
  `last_rows_changed_snapshot`): what a window of snapshots did to a table, from the
  `changes_made` ledger and the window's snapshot ids, with no file listing, and where the
  ledger cannot tell, this crate's port of official's change feed. `rows_changed` is the
  `ducklake_table_insertions` or `ducklake_table_deletions` feed over the window being
  non-empty, decided from the tokens wherever they can: no token naming the table (or only
  `altered_table`) is no change; an insert or delete token (`inserted_into_table`,
  `deleted_from_table`, `inlined_insert`, `inlined_delete`), `dropped_table` or a token this
  crate does not know is a change without reading data. Only snapshots recording just a flush
  or a compaction (`inline_flush`, `flushed_inlined`, `merge_adjacent`, `rewrite_delete`,
  `compacted_table`) read the feed, each feed up to its first row: a `DELETE` committed after
  official's flush in one transaction records only `inline_flush`, and the feed lists the
  deletion; official's `ducklake_rewrite_data_files` lists the rows it rewrote as `UPDATE`
  pairs, so a rewrite is a change, while a pure flush or merge lists nothing. One case errs
  towards a change: official's flush of inlined deletions of Parquet rows records
  `deleted_from_table`, as a `DELETE` does, with nothing in the feed. `layout_changed` adds
  every flush and compaction from the tokens alone. A snapshot without a
  ledger row counts as a change, and so do snapshots expired from the window
  (`history_incomplete`; found by snapshot id: every id from the window's start to the catalog's
  head must survive, so a gap at the window's end is found too. A multi-catalog store, whose
  catalogs share one id sequence, finds only an expired starting snapshot and an expired first
  snapshot of the table). A window that starts
  after snapshot `0` of a catalog this crate created, which never had one, is complete.
- **Table fork** (`MulticatalogManager::fork_table`, multi-catalog PostgreSQL): a table in another
  catalog of the store holding what the source reads at its snapshot, in one commit, its columns
  copying the source's catalog rows (exact types, nullability, defaults) and its tags and
  comments the source's. Data files
  and delete files are registered by reference with their owning catalog; a file with inlined
  deletions gets a delete file of the fork's own holding every deleted position; the source's
  inlined rows are written to a Parquet file of the fork's own, as a flush writes them (row ids
  and insert snapshots embedded, split by partition), so the fork holds what a flush of the
  source would have without changing the source. Row ids are preserved and the fork's allocator
  starts past the source's. A source opened at an earlier snapshot masks later deletions and
  later rows of partial files with a delete file of the fork's own. Forking into the source's own
  catalog is refused (one catalog would own a file twice), as is a source with a live file of
  another partition spec than its current one, an encrypted file (the key cannot be carried) or a
  file read through a name mapping, and a source with a column tag other than `comment`. A fork
  into an existing table replaces it when its columns have the source's ids (an earlier fork),
  recording changed tags as `set_tag` does; one created there on its own, sharing a column name
  under another id, is refused before anything is written (the fork's files carry the source's
  field ids, which a replace keeps by name). A list column in the legacy one-row catalog shape
  forks into upstream's `list` and `element` rows. Official DuckLake has no fork; its closest
  equivalent, `CREATE TABLE AS SELECT`, copies rows with new row ids.
- **Row restore** (`DuckLakeTable::restore_rows_to(state, snapshot)`, feature `restore`; a
  library extension with no official equivalent): the table's rows become its rows at an
  earlier snapshot in one new snapshot, each restored row keeping its `rowid`, and no snapshot
  is rewritten. It commits what a SQL `UPDATE` commits, with its conflict checks, and also
  aborts on any commit to the table since it was read: rows live now but not then are deleted
  as a `DELETE` deletes them (a data file added since, with more live rows than the scoped
  `data_inlining_row_limit` and none deleted inlined, is ended whole), and rows live then but
  not now are inserted again with their row ids, inlined within that limit, else in Parquet
  files embedding them, in the table's partition layout. It runs on SQLite, DuckDB, MySQL and
  multi-catalog PostgreSQL; single-catalog PostgreSQL refuses it (`Unsupported`), as it refuses
  `UPDATE` and `DELETE`. It is refused when the snapshot is not before the table's
  (`InvalidConfig`), the table did not exist then (`InvalidConfig`), the snapshot has expired
  (`InvalidSnapshot`), or its partition spec changed since (`Unsupported`). Columns added since
  and columns widened since (an integer to a wider one of the same signedness, `FLOAT` to
  `DOUBLE`) stay, and the restored rows are written in them as the table reads them: widened
  values cast, added columns holding their initial default, else NULL. A column dropped,
  renamed, made `NOT NULL` or changed to any other type since (a narrowing, a nested field
  change, or a promotion official allows beyond those, such as `INTEGER` to `DOUBLE`), or one
  added since that is `NOT NULL` with no initial default, refuses it (`Unsupported`). Official
  DuckLake reads the result and its change feed as an `UPDATE`'s; its closest equivalent,
  inserting the rows read `AT (VERSION => ...)`, gives them new row ids.
- **Partition live row counts** (`DuckLakeTable::partition_live_row_counts`; a library
  extension with no official equivalent): the live rows of each requested value of a column
  that is the one identity key of the table's live partition spec (not a floating-point one,
  whose `0.0` and `-0.0` are separate partitions), at the table's snapshot,
  read from the catalog's per-file `record_count` and visible delete file `delete_count` and
  the table's inlined deletions, opening no file. A count is exact or unknown, never
  estimated. Every value is unknown when the snapshot no longer exists, the table has any
  visible inlined row (stricter than official, whose nearest mechanism, the table-wide
  `count(*)` fold over `GetNetDataFileRowCountSql`, adds inlined rows in), or a visible file
  was written under another partition spec, has no value for the key, or has a value not
  spelled as this crate's writer spells it. A value is unknown when it is itself spelled
  otherwise (`02` for an integer 2), or a file of it records no row count, is a merged partial
  file read below its `partial_max`, has several visible delete files, a delete file without a
  valid count, or both a delete file and inlined deletions. Below the catalog head a delete
  file counts only with a `partial_max` at or below the snapshot: one that rewrites an earlier
  delete file records its latest deletion there, but one written by the inlined-data flush
  (here and in official) begins at its earliest deletion and records none. It reads SQLite
  and PostgreSQL catalogs (single- and multi-catalog); other providers, and catalogs without
  `ducklake_delete_file.partial_max`, answer unknown (`Unsupported`). It reads no table
  official does not define and adds no index.
- **Maintenance flush** (`InlinedFlushPolicy`, carried by `DuckLakeWriteOptions`; an opt-in
  addition of this crate): by default `merge_adjacent_files` and `rewrite_data_files` flush
  nothing, as official's `ducklake_merge_adjacent_files` and `ducklake_rewrite_data_files`; its
  `CHECKPOINT` flushes every table first, as `DuckLakeCatalog::checkpoint` does. With a policy
  (`InlinedFlushPolicy::when_due()`: 10,000 inlined rows and deletions, or an hour-old unflushed
  change; or custom triggers) a merge or rewrite first flushes a due table's inlined data, as
  `CALL ducklake_flush_inlined_data` does, then plans over the flushed table;
  `CompactionResult::inlined_flush_snapshot` reports it. The triggers count what the flush
  moves: the inlined rows and the inlined deletions of Parquet rows, not a retired file's
  deletions the flush leaves inlined (see the flush above). The age is measured from the oldest
  of them (its insert or deletion snapshot, or the first surviving snapshot after it when that
  one was expired), each snapshot's time read as a UTC instant whatever the offset the backend
  prints it with; PostgreSQL writers store snapshot times in UTC. A metadata provider without
  the flush reads (a third-party one keeping the trait's defaults) has nothing to flush. A merge leaves out
  a file with inlined deletions (merging it would lose them); `rewrite_data_files` counts them
  towards the deleted fraction and retires a file they empty.

## Limitations

- **`CatalogReadMemo` is not DuckLake behavior.** Official DuckLake lists a scan's files again
  in every transaction, at the snapshot that the transaction pinned. The opt-in memo keeps a
  table's listing, inlined deletions and inlined rows across queries. It serves them only while
  the catalog head (the newest snapshot id) is the one that its bind read at the start of the
  query. This is correct only because every commit that changes a listing also moves the head:
  an insert, a delete, an update, a merge of adjacent files, a rewrite of data files and a flush
  of inlined data. Two cases do not move the head:
  - The low-level `MetadataWriter` API changes the catalog at a snapshot that `create_snapshot`
    made earlier, for example with `register_data_file` or `end_table_files`. A memo filled
    between the two misses the change until the next commit.
  - Expiring snapshots and cleanup commit no snapshot. Official refuses a read at an expired
    snapshot with "No snapshot found". Without a memo, this crate returns no rows for it. With a
    memo, a fill kept before the expire returns the old rows until cleanup deletes their files,
    and then the read fails.

  `DuckLakeTable::scan` has no memo code, and the memo lives in its own provider wrapper
  (`src/catalog_read_memo.rs`).

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
  by every writer's schema bootstrap, and to an existing catalog by its
  migration. It is the index the pre-filter above reads through; without it
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
  `DuckLakeCatalog::with_snapshot_at` (a stored snapshot time with a UTC offset, as PostgreSQL
  and DuckDB print a `TIMESTAMPTZ` outside UTC, is converted to UTC; one without is read as UTC,
  which a MySQL session outside UTC does not print), or select one table per query with
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
- **Change feeds over a merged (partial) file with a damaged per-row snapshot column fail.**
  Official's change feed reads such a file's rows with a NULL
  `_ducklake_internal_snapshot_id` with a NULL `snapshot_id` when the window covers the whole
  file and leaves them out otherwise, and reads a file without the column as written at its
  `begin_snapshot` when the window covers the whole file (failing otherwise). This crate's
  change feed fails on both. Official writers always write the column, without NULLs.
- **Streaming write sessions write Parquet rows in the caller's order.** Official's insert
  sorts them by `SET SORTED BY` (with `sort_on_insert` off, while inlining is enabled); a
  `TableWriteSession` cannot without holding the whole write, so it sorts only the rows it
  inlines. `DuckLakeTableWriter::write_rows`, `append_table` and SQL `INSERT` sort as official.
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
  column id against the schema version their inlined table was registered for
  (`ducklake_inlined_data_tables.schema_version` in `ducklake_schema_versions`,
  else the table's creation, as official's `GetBeginSnapshotForSchemaVersion`), so
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
