# DataFusion-DuckLake

[![crates.io](https://img.shields.io/crates/v/datafusion-ducklake.svg)](https://crates.io/crates/datafusion-ducklake)
[![docs.rs](https://img.shields.io/docsrs/datafusion-ducklake)](https://docs.rs/datafusion-ducklake)
[![CI](https://github.com/hotdata-dev/datafusion-ducklake/actions/workflows/ci.yml/badge.svg)](https://github.com/hotdata-dev/datafusion-ducklake/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Discord](https://img.shields.io/badge/Discord-DataFusion%2BDuckLake-5865F2?logo=discord&logoColor=white)](https://discord.com/channels/885562378132000778/1492192627666321452)

A [DataFusion](https://datafusion.apache.org/) extension for reading and writing
[DuckLake](https://ducklake.select) catalogs. DuckLake is an integrated data lake and
catalog format that stores metadata in a SQL database and data as Parquet files on disk
or object storage.

The goal of this project is to make DuckLake a first-class, Arrow-native lakehouse
format inside DataFusion.

This project is maintained by [Hotdata](https://www.hotdata.dev) with support from the
community. Come talk to us on [the Hotdata Discord](https://discord.gg/cdHczfxxBc).

- 📦 **crates.io:** <https://crates.io/crates/datafusion-ducklake>
- 📖 **API docs:** <https://docs.rs/datafusion-ducklake>
- 🧩 **Feature & backend support:** see [COMPATIBILITY.md](COMPATIBILITY.md)
- 💬 **Project chat:** [DataFusion+DuckLake Discord](https://discord.com/channels/885562378132000778/1492192627666321452) — development and usage discussion
- 🧡 **Meet the team:** [Hotdata Discord](https://discord.gg/cdHczfxxBc)

---

## Quick start

Add the crate:

```bash
cargo add datafusion-ducklake
```

The default build includes the DuckDB catalog backend, statically bundled. Other
backends and write support are opt-in via feature flags — see
[COMPATIBILITY.md](COMPATIBILITY.md) for the full matrix.

```toml
# Cargo.toml — read PostgreSQL catalogs
# (for the experimental multi-catalog write path, use features = ["write-postgres"])
[dependencies]
datafusion-ducklake = { version = "0.5", features = ["metadata-postgres"] }
```

The examples below also use `datafusion`, `object_store`, and `url` directly — add them
to your `[dependencies]` as well (this crate does not re-export them). The write example
additionally uses `sqlx` (with its `postgres` and `runtime-tokio` features) to open the
connection pool.

Run a query against an existing PostgreSQL catalog with the bundled example:

```bash
cargo run --example basic_query --features metadata-postgres -- \
  "postgresql://user:password@localhost:5432/database" "SELECT * FROM main.users"
```

(The example also accepts DuckDB, SQLite, and MySQL connection strings with the matching
`metadata-*` feature — see [COMPATIBILITY.md](COMPATIBILITY.md).)

---

## Reading a catalog

Register a `DuckLakeCatalog` with a `SessionContext` and query it with normal SQL as
`catalog.schema.table`:

```rust,ignore
use datafusion::execution::runtime_env::RuntimeEnv;
use datafusion::prelude::*;
use datafusion_ducklake::{DuckLakeCatalog, PostgresMetadataProvider};
use object_store::ObjectStore;
use object_store::aws::AmazonS3Builder;
use std::sync::Arc;
use url::Url;

// (inside an async fn)
// Read metadata from a PostgreSQL catalog
let provider = PostgresMetadataProvider::new("postgresql://user:pass@localhost:5432/db").await?;

// Register object stores for any non-local data (S3 / MinIO)
let runtime = Arc::new(RuntimeEnv::default());
let s3: Arc<dyn ObjectStore> = Arc::new(
    AmazonS3Builder::new()
        .with_endpoint("http://localhost:9000") // MinIO endpoint
        .with_bucket_name("ducklake-data")
        .with_access_key_id("minioadmin")
        .with_secret_access_key("minioadmin")
        .with_region("us-west-2") // any region works for MinIO
        .with_allow_http(true)    // required for http:// endpoints
        .build()?,
);
runtime.register_object_store(&Url::parse("s3://ducklake-data/")?, s3);

let catalog = DuckLakeCatalog::new(provider)?;
let ctx = SessionContext::new_with_config_rt(
    SessionConfig::new().with_default_catalog_and_schema("ducklake", "main"),
    runtime,
);
ctx.register_catalog("ducklake", Arc::new(catalog));

let df = ctx.sql("SELECT * FROM ducklake.main.my_table").await?;
df.show().await?;
```

---

## Writing a catalog

PostgreSQL writes go through the **experimental multi-catalog layout** described in the
[next section](#multi-catalog-postgresql-experimental) — treat them as a preview. Tables
are created through the writer API; once a table exists, you append to it with SQL
`INSERT INTO`. (SQL `CREATE TABLE` / CTAS is not supported on this path — DataFusion
cannot create the schema, so the first write goes through `DuckLakeTableWriter`.)

```rust,ignore
use datafusion::prelude::*;
use datafusion_ducklake::metadata_writer::MetadataWriter; // set_data_path
use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeTableWriter, MulticatalogManager, MulticatalogProvider,
    PostgresMetadataWriter, initialize_multicatalog_schema,
};
use sqlx::postgres::PgPoolOptions;
use std::sync::Arc;

let pool = PgPoolOptions::new().connect("postgresql://user:pass@localhost:5432/db").await?;

// One-time: bootstrap the multi-catalog tables, then create a named catalog
initialize_multicatalog_schema(&pool).await?;
let catalog_id = MulticatalogManager::new(pool.clone()).create_catalog("my_catalog").await?;

// Create a table by writing the first batch through the table writer
let writer = Arc::new(PostgresMetadataWriter::with_pool(pool.clone(), catalog_id).await?);
writer.set_data_path("/abs/path/to/data")?;
let object_store: Arc<dyn object_store::ObjectStore> =
    Arc::new(object_store::local::LocalFileSystem::new());
let table_writer = DuckLakeTableWriter::new(writer.clone(), object_store)?;
table_writer.write_table("public", "events", &[batch]).await?; // `batch` is your RecordBatch

// Now append with SQL, reading the same catalog back through MulticatalogProvider
let provider = MulticatalogProvider::with_pool(pool.clone(), "my_catalog").await?;
let catalog = DuckLakeCatalog::with_writer(Arc::new(provider), writer)?;
let ctx = SessionContext::new();
ctx.register_catalog("ducklake", Arc::new(catalog));
ctx.sql("INSERT INTO ducklake.public.events VALUES (1, 'a')").await?.collect().await?;
ctx.sql("SELECT count(*) FROM ducklake.public.events").await?.show().await?;
```

Writer output is configurable (Parquet compression, row-group sizing by row count and
byte size). See [`DuckLakeTableWriter`](https://docs.rs/datafusion-ducklake) for the
writer options.

> Writing to a **standard, single-catalog** DuckLake store (the spec-compliant layout) is
> supported today for **SQLite** via `SqliteMetadataWriter` (feature `write-sqlite`),
> where SQL `CREATE TABLE AS SELECT` and `INSERT INTO` both work. See
> [`tests/sql_write_tests.rs`](tests/sql_write_tests.rs).

---

## Multi-catalog (PostgreSQL, experimental)

A single PostgreSQL metadata store can host **multiple independent DuckLake catalogs** —
useful for multi-tenant deployments or keeping many logical lakehouses in one database.

> ⚠️ **Experimental and library-specific.** This multi-catalog layout is **not part of the
> DuckLake specification** and is not (yet) supported or accepted upstream. Catalogs
> written this way are read back only through this crate's `MulticatalogProvider` — they
> are **not** interchangeable with a standard single-catalog DuckLake store. The API and
> on-disk/in-catalog layout may change. PostgreSQL write support currently depends on this
> path, so treat it as a preview.

- **Create and manage** catalogs with `MulticatalogManager` (feature `write-postgres`):
  `initialize_multicatalog_schema` bootstraps the shared tables, then `create_catalog`,
  and `drop_table_in_catalog` manage their contents.
- **Read** a specific catalog with `MulticatalogProvider::with_pool(pool, "name")`
  (feature `multicatalog-postgres`), which plugs into a `DuckLakeCatalog` like any other
  metadata provider.

See [`examples/multicatalog_write.rs`](examples/multicatalog_write.rs) for an end-to-end
walkthrough (bootstrap → create catalogs → write → read back).

---

## Maintenance

The `maintenance` API handles lakehouse upkeep from Rust: expiring old snapshots,
cleaning up superseded files, and reclaiming orphaned files. The concrete entry points
are backend-gated (`write-sqlite` / `write-postgres`). `DROP TABLE` is available through
`MetadataWriter`. See
[`examples/maintenance_demo.rs`](examples/maintenance_demo.rs) and
[`examples/orphan_cleanup_demo.rs`](examples/orphan_cleanup_demo.rs).

### Compaction

Two explicit, triggered operations on `DuckLakeTable` rewrite a table's data files
into a better physical layout without changing its logical rows:

- `merge_adjacent_files(state, MergeOptions)` coalesces several small files (of the
  same schema version) into fewer larger ones. A merged file spanning multiple
  origin snapshots is written as a DuckLake *partial data file* (preserving each
  row's original rowid and origin snapshot), so time travel and change feeds are
  unaffected.
- `rewrite_data_files(state, RewriteOptions)` rewrites a file whose deleted fraction
  exceeds a threshold (default `0.95`), dropping its deleted rows.

Both commit atomically in one snapshot and coexist with concurrent appends; superseded
files are scheduled for deletion and reclaimed later by `cleanup_old_files`. See
[`examples/compaction_demo.rs`](examples/compaction_demo.rs).

---

## Compatibility

For the full breakdown of catalog backends, object stores, types, capabilities, and
current limitations, see **[COMPATIBILITY.md](COMPATIBILITY.md)**.

A few highlights worth knowing up front:

- Reads work on DuckDB, SQLite, PostgreSQL, and MySQL; **writes are SQLite/PostgreSQL only**.
- Object stores: local filesystem and S3-compatible (S3, MinIO).
- Snapshots can be selected programmatically (`DuckLakeCatalog::with_snapshot`), but there
  is no SQL-level time travel (`AS OF`) yet, and no partition-based file pruning.
- Data inlined by DuckDB's ducklake extension is **not read** — see COMPATIBILITY.md for
  the `COUNT(*)` undercount caveat and how to avoid it.

---

## Project status

This project is in alpha and evolving alongside DataFusion and DuckLake. APIs may change
as core abstractions are refined. See [CHANGELOG.md](CHANGELOG.md) for release history.
Feedback, issues, and contributions are welcome.
