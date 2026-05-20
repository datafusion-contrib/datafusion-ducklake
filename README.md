# DataFusion-DuckLake

A [DataFusion](https://datafusion.apache.org/) extension for querying [DuckLake](https://ducklake.select). DuckLake is an integrated data lake and catalog format that stores metadata in SQL databases and data as Parquet files on disk or object storage.

The goal of this project is to make DuckLake a first-class, Arrow-native lakehouse format inside DataFusion.

If you're interested in joining the development, [join the DataFusion+DuckLake Discord](https://discord.com/channels/885562378132000778/1492192627666321452)

To meet the [Hotdata](https://www.hotdata.dev) team, [Join the Hotdata Discord](https://discord.gg/cdHczfxxBc)

---

## Currently Supported

- Read and write queries against DuckLake catalogs (INSERT INTO support)
- DuckDB, PostgreSQL, MySQL, and SQLite catalog backends
- Local filesystem and S3-compatible object stores (MinIO, S3)
- Snapshot-based consistency
- Basic and decimal types
- Hierarchical path resolution (`data_path`, `schema`, `table`, `file`)
- Delete files for row-level deletion (MOR – Merge-On-Read)
- Parquet Modular Encryption (PME) for reading encrypted Parquet files
- Parquet footer size hints for optimized I/O
- Filter pushdown to Parquet for row group pruning and page-level filtering
- Dynamic metadata lookup (no upfront catalog caching)
- SQL-queryable `information_schema` for catalog metadata (snapshots, schemas, tables, columns, files)
- DuckDB-style table functions: `ducklake_snapshots()`, `ducklake_table_info()`, `ducklake_list_files()`, `ducklake_table_changes()`
- DuckLake row lineage (`rowid` virtual column), opt-in via `DuckLakeCatalog::with_row_lineage(true)`. Writers always populate the lineage counter, matching DuckDB's default; the flag is read-only. Compatible with files produced by DuckDB's `UPDATE` / compaction (embedded `_ducklake_internal_row_id`).

---

## Known Limitations

- Complex types (nested lists, structs, maps) have minimal support
- No partition-based file pruning
- No time travel support
- DuckDB-encrypted Parquet files (non-PME) are not supported
- **Data inlining is not read.** DuckDB's ducklake extension inlines small INSERTs (≤ `ducklake_default_data_inlining_row_limit`, default 10 rows) into the catalog itself rather than parquet files. This crate only reads `ducklake_data_file` rows, so inlined data is invisible — `SELECT COUNT(*)` will silently undercount. If you write through DuckDB and read through this crate, either disable inlining at write time (`SET ducklake_default_data_inlining_row_limit = 0` on every writer connection) or run `COMPACT` before reading. Catalogs written entirely through this crate's `SqliteMetadataWriter` are unaffected — we never inline.

---

## Roadmap

This project is under active development. The roadmap below reflects major areas of work currently underway or planned next. For the most up-to-date view, see the open issues and pull requests in this repository.

### Metadata & Catalog Improvements

- Metadata caching to reduce repeated catalog lookups
- Clear abstraction boundaries between catalog, metadata provider, and execution

### Query Planning & Performance

- Partition-aware file pruning
- Improved predicate pushdown
- Smarter Parquet I/O planning
- Reduced metadata round-trips during planning
- Better alignment with DataFusion optimizer rules

### Write Support

- ~~Initial write support for DuckLake tables~~ (Done in v0.0.5)
- ~~S3/ObjectStore write support~~ (Done in v0.0.6)
- UPDATE and DELETE operations

### Time Travel & Versioning

- Querying historical snapshots
- Explicit snapshot selection

### Type System Expansion

- Improved support for complex and nested types
- Better alignment with DuckDB and DataFusion type semantics

### Stability & Ergonomics

- Expanded test coverage
- Improved error messages and diagnostics
- Cleaner APIs for embedding in other DataFusion-based systems
- Additional documentation and examples

---

## Usage

### Feature Flags

| Feature | Description | Default |
|---------|-------------|---------|
| `metadata-duckdb` | DuckDB catalog backend | ✅ |
| `duckdb-bundled` | Statically compile and bundle DuckDB (disable for dynamic linking) | ✅ |
| `metadata-postgres` | PostgreSQL catalog backend | |
| `metadata-mysql` | MySQL catalog backend | |
| `metadata-sqlite` | SQLite catalog backend | |
| `encryption` | Parquet Modular Encryption (PME) support | |
| `write` | Write support (INSERT INTO) | |

```bash
# DuckDB only (default, bundled build)
cargo build

# DuckDB dynamically linked against a system libduckdb
# (requires libduckdb installed; set DUCKDB_LIB_DIR and DUCKDB_INCLUDE_DIR)
cargo build --no-default-features --features metadata-duckdb

# PostgreSQL only
cargo build --no-default-features --features metadata-postgres

# MySQL only
cargo build --no-default-features --features metadata-mysql

# SQLite only
cargo build --no-default-features --features metadata-sqlite

# All backends
cargo build --features metadata-postgres,metadata-mysql,metadata-sqlite
```

### Example

```bash
# DuckDB catalog
cargo run --example basic_query -- catalog.db "SELECT * FROM main.users"

# PostgreSQL catalog
cargo run --example basic_query --features metadata-postgres -- \
  "postgresql://user:password@localhost:5432/database" "SELECT * FROM main.users"

# MySQL catalog
cargo run --example basic_query --features metadata-mysql -- \
  "mysql://user:password@localhost:3306/database" "SELECT * FROM main.users"

# SQLite catalog
cargo run --example basic_query --features metadata-sqlite -- \
  "sqlite:///path/to/catalog.db" "SELECT * FROM main.users"
```

### Integration
```rust
use datafusion::execution::runtime_env::RuntimeEnv;
use datafusion::prelude::*;
use datafusion_ducklake::{DuckLakeCatalog, DuckdbMetadataProvider};
use std::sync::Arc;

// Create metadata provider
let provider = DuckdbMetadataProvider::new("catalog.db")?;

// Create runtime (register object stores if using S3/MinIO)
let runtime = Arc::new(RuntimeEnv::default());

// Example: Register S3/MinIO object store
let s3: Arc<dyn ObjectStore> = Arc::new(
    AmazonS3Builder::new()
        .with_endpoint("http://localhost:9000") // Your MinIO endpoint
        .with_bucket_name("ducklake-data") // Your bucket name
        .with_access_key_id("minioadmin") // Your credentials
        .with_secret_access_key("minioadmin") // Your credentials
        .with_region("us-west-2") // Any region works for MinIO
        .with_allow_http(true) // Required for http:// endpoints
        .build()?,
    );
runtime.register_object_store(&Url::parse("s3://ducklake-data/")?, s3);

// Create DuckLake catalog
let catalog = DuckLakeCatalog::new(provider)?;

// Create session and register catalog
let ctx = SessionContext::new_with_config_rt(
    SessionConfig::new().with_default_catalog_and_schema("ducklake", "main"),
    runtime
);
ctx.register_catalog("ducklake", Arc::new(catalog));

// Query
let df = ctx.sql("SELECT * FROM ducklake.main.my_table").await?;
df.show().await?;


```
### Project Status

This project is evolving alongside DataFusion and DuckLake. APIs may change as core abstractions are refined.

Feedback, issues, and contributions are welcome.
