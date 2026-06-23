# DataFusion-DuckLake

[![crates.io](https://img.shields.io/crates/v/datafusion-ducklake.svg)](https://crates.io/crates/datafusion-ducklake)
[![docs.rs](https://img.shields.io/docsrs/datafusion-ducklake)](https://docs.rs/datafusion-ducklake)
[![CI](https://github.com/hotdata-dev/datafusion-ducklake/actions/workflows/ci.yml/badge.svg)](https://github.com/hotdata-dev/datafusion-ducklake/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Discord](https://img.shields.io/badge/Discord-Hotdata-5865F2?logo=discord&logoColor=white)](https://discord.gg/cdHczfxxBc)

A [DataFusion](https://datafusion.apache.org/) extension for reading and writing
[DuckLake](https://ducklake.select) catalogs. DuckLake is an integrated data lake and
catalog format that stores metadata in a SQL database and data as Parquet files on disk
or object storage.

The goal of this project is to make DuckLake a first-class, Arrow-native lakehouse
format inside DataFusion.

- 📦 **crates.io:** <https://crates.io/crates/datafusion-ducklake>
- 📖 **API docs:** <https://docs.rs/datafusion-ducklake>
- 🧩 **Feature & backend support:** see [COMPATIBILITY.md](COMPATIBILITY.md)
- 💬 **Discord:** [DataFusion+DuckLake](https://discord.com/channels/885562378132000778/1492192627666321452) · [Hotdata](https://discord.gg/cdHczfxxBc)

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
# Cargo.toml — e.g. read PostgreSQL catalogs and write to SQLite ones
[dependencies]
datafusion-ducklake = { version = "0.3", features = ["metadata-postgres", "write-sqlite"] }
```

The examples below also use `datafusion`, `object_store`, and `url` directly — add them
to your `[dependencies]` as well (this crate does not re-export them).

Run a query against an existing catalog with the bundled example:

```bash
# DuckDB catalog
cargo run --example basic_query -- catalog.db "SELECT * FROM main.users"

# SQLite / PostgreSQL / MySQL catalogs (enable the matching backend feature)
cargo run --example basic_query --features metadata-sqlite -- \
  "sqlite:///path/to/catalog.db" "SELECT * FROM main.users"
```

---

## Reading a catalog

Register a `DuckLakeCatalog` with a `SessionContext` and query it with normal SQL as
`catalog.schema.table`:

```rust,ignore
use datafusion::execution::runtime_env::RuntimeEnv;
use datafusion::prelude::*;
use datafusion_ducklake::{DuckLakeCatalog, DuckdbMetadataProvider};
use object_store::ObjectStore;
use object_store::aws::AmazonS3Builder;
use std::sync::Arc;
use url::Url;

// (inside an async fn)
// Read metadata from a DuckDB catalog
let provider = DuckdbMetadataProvider::new("catalog.db")?;

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

Writes are supported on **SQLite** and **PostgreSQL** catalogs (enable `write-sqlite`
or `write-postgres`). Build the catalog with `with_writer`, then use standard SQL —
`INSERT INTO` and `CREATE TABLE AS SELECT` both work:

```rust,ignore
use datafusion::prelude::*;
// `MetadataWriter` is the trait that provides `set_data_path` / `create_snapshot`
use datafusion_ducklake::{
    DuckLakeCatalog, MetadataWriter, SqliteMetadataProvider, SqliteMetadataWriter,
};
use std::sync::Arc;

let conn = "sqlite:catalog.db?mode=rwc";

// Initialize a writer (creates the DuckLake schema + first snapshot if new)
let writer = SqliteMetadataWriter::new_with_init(conn).await?;
writer.set_data_path("data/")?;
writer.create_snapshot()?;

// A writable catalog pairs a read provider with the writer
let provider = SqliteMetadataProvider::new(conn).await?;
let catalog = DuckLakeCatalog::with_writer(Arc::new(provider), Arc::new(writer))?;

let ctx = SessionContext::new();
ctx.register_catalog("ducklake", Arc::new(catalog));

ctx.sql("CREATE TABLE ducklake.main.events AS SELECT * FROM source").await?.collect().await?;
ctx.sql("INSERT INTO ducklake.main.events VALUES (1, 'a')").await?.collect().await?;
```

Writer output is configurable (Parquet compression, row-group sizing by row count and
byte size). See [`DuckLakeTableWriter`](https://docs.rs/datafusion-ducklake) and the
examples in [`examples/`](examples/).

---

## Multi-catalog (PostgreSQL)

A single PostgreSQL metadata store can host **multiple independent DuckLake catalogs**.
This is useful for multi-tenant deployments or keeping many logical lakehouses in one
database.

- **Read** multiple catalogs with `MulticatalogProvider` (feature `multicatalog-postgres`).
- **Create and manage** them with `MulticatalogManager` (feature `write-postgres`, which
  pulls in multi-catalog support).

See [`examples/multicatalog_write.rs`](examples/multicatalog_write.rs) for an end-to-end
walkthrough.

---

## Maintenance

The `maintenance` API (feature `write`) handles lakehouse upkeep from Rust: expiring old
snapshots, cleaning up superseded files, and reclaiming orphaned files. `DROP TABLE` is
available through `MetadataWriter`. See
[`examples/maintenance_demo.rs`](examples/maintenance_demo.rs) and
[`examples/orphan_cleanup_demo.rs`](examples/orphan_cleanup_demo.rs).

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

## Roadmap

This project is under active development. The list below is forward-looking; for what
has already shipped, see [CHANGELOG.md](CHANGELOG.md).

- **Write:** `UPDATE` and `DELETE` operations
- **Time travel:** SQL-level historical queries (`AS OF`); programmatic snapshot binding
  via `DuckLakeCatalog::with_snapshot` already ships
- **Query planning:** partition-aware file pruning, improved predicate pushdown, fewer
  metadata round-trips during planning
- **Metadata:** optional caching layer to reduce repeated catalog lookups
- **Types:** broader support for complex and nested types
- **Object stores:** Google Cloud Storage and Azure Blob Storage
- **Ergonomics:** cleaner APIs for embedding in other DataFusion-based systems, better
  error messages, more docs and examples

For the most up-to-date view, see the open issues and pull requests.

---

## Project status

This project is in alpha and evolving alongside DataFusion and DuckLake. APIs may change
as core abstractions are refined. Feedback, issues, and contributions are welcome.
