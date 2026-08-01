//! Basic DuckLake query example with snapshot isolation
//!
//! This example demonstrates how to:
//! 1. Create a DuckLake catalog from DuckDB, PostgreSQL, MySQL, or SQLite
//! 2. Bind the catalog to a specific snapshot for query consistency
//! 3. Register it with DataFusion
//! 4. Execute a simple SELECT query
//!
//! ## Snapshot Isolation
//!
//! Each DuckLake catalog is bound to a specific snapshot ID at creation time.
//! This guarantees that all queries through that catalog see a consistent view
//! of the data, even if multiple schema/table lookups happen during query planning
//! or if the underlying data changes.
//!
//! To query data at different points in time, create separate catalogs bound to
//! different snapshot IDs.
//!
//! ## Usage
//!
//! With DuckDB catalog:
//! ```bash
//! cargo run --example basic_query catalog.db "SELECT * FROM main.users"
//! ```
//!
//! With PostgreSQL catalog (requires --features metadata-postgres):
//! ```bash
//! cargo run --example basic_query --features metadata-postgres \
//!   "postgresql://user:password@localhost:5432/postgres" \
//!   "SELECT * FROM main.users"
//! ```
//!
//! With MySQL catalog (requires --features metadata-mysql):
//! ```bash
//! cargo run --example basic_query --features metadata-mysql \
//!   "mysql://user:password@localhost:3306/database" \
//!   "SELECT * FROM main.users"
//! ```
//!
//! With SQLite catalog (requires --features metadata-sqlite):
//! ```bash
//! cargo run --example basic_query --features metadata-sqlite \
//!   "sqlite:///path/to/catalog.db" \
//!   "SELECT * FROM main.users"
//! ```

use datafusion::execution::runtime_env::RuntimeEnv;
use datafusion::prelude::*;
#[cfg(feature = "metadata-duckdb")]
use datafusion_ducklake::DuckdbMetadataProvider;
#[cfg(feature = "metadata-mysql")]
use datafusion_ducklake::MySqlMetadataProvider;
#[cfg(feature = "metadata-postgres")]
use datafusion_ducklake::PostgresMetadataProvider;
#[cfg(feature = "metadata-sqlite")]
use datafusion_ducklake::SqliteMetadataProvider;
use datafusion_ducklake::{DuckLakeCatalog, MetadataProvider, register_ducklake_functions};
use object_store::ObjectStore;
use object_store::aws::AmazonS3Builder;
use std::env;
use std::process::exit;
use std::sync::Arc;
use url::Url;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage:");
        eprintln!("  DuckDB:     cargo run --example basic_query catalog.db \"SQL\"");
        eprintln!(
            "  PostgreSQL: cargo run --example basic_query --features metadata-postgres \"postgresql://...\" \"SQL\""
        );
        eprintln!(
            "  MySQL:      cargo run --example basic_query --features metadata-mysql \"mysql://...\" \"SQL\""
        );
        eprintln!(
            "  SQLite:     cargo run --example basic_query --features metadata-sqlite \"sqlite://...\" \"SQL\""
        );
        exit(1);
    }
    let catalog_source = &args[1];
    let sql = &args[2];

    // Detect provider type based on input
    let is_postgres = catalog_source.starts_with("postgresql://");
    let is_mysql = catalog_source.starts_with("mysql://");
    let is_sqlite = catalog_source.starts_with("sqlite:");

    if is_postgres {
        #[cfg(not(feature = "metadata-postgres"))]
        {
            eprintln!("Error: PostgreSQL support requires the 'metadata-postgres' feature");
            eprintln!("Run with: cargo run --example basic_query --features metadata-postgres");
            exit(1);
        }

        #[cfg(feature = "metadata-postgres")]
        {
            println!("Connecting to PostgreSQL catalog: {}", catalog_source);
            let provider = Arc::new(PostgresMetadataProvider::new(catalog_source).await?);
            let snapshot_id = provider.get_current_snapshot()?;
            println!("Current snapshot ID: {}", snapshot_id);
            run_query(provider, snapshot_id, sql).await?;
        }
    } else if is_mysql {
        #[cfg(not(feature = "metadata-mysql"))]
        {
            eprintln!("Error: MySQL support requires the 'metadata-mysql' feature");
            eprintln!("Run with: cargo run --example basic_query --features metadata-mysql");
            exit(1);
        }

        #[cfg(feature = "metadata-mysql")]
        {
            println!("Connecting to MySQL catalog: {}", catalog_source);
            let provider = Arc::new(MySqlMetadataProvider::new(catalog_source).await?);
            let snapshot_id = provider.get_current_snapshot()?;
            println!("Current snapshot ID: {}", snapshot_id);
            run_query(provider, snapshot_id, sql).await?;
        }
    } else if is_sqlite {
        #[cfg(not(feature = "metadata-sqlite"))]
        {
            eprintln!("Error: SQLite support requires the 'metadata-sqlite' feature");
            eprintln!("Run with: cargo run --example basic_query --features metadata-sqlite");
            exit(1);
        }

        #[cfg(feature = "metadata-sqlite")]
        {
            println!("Connecting to SQLite catalog: {}", catalog_source);
            let provider = Arc::new(SqliteMetadataProvider::new(catalog_source).await?);
            let snapshot_id = provider.get_current_snapshot()?;
            println!("Current snapshot ID: {}", snapshot_id);
            run_query(provider, snapshot_id, sql).await?;
        }
    } else {
        #[cfg(feature = "metadata-duckdb")]
        {
            println!("Connecting to DuckDB catalog: {}", catalog_source);
            let provider = Arc::new(DuckdbMetadataProvider::new(catalog_source)?);
            let snapshot_id = provider.get_current_snapshot()?;
            println!("Current snapshot ID: {}", snapshot_id);
            run_query(provider, snapshot_id, sql).await?;
        }

        #[cfg(not(feature = "metadata-duckdb"))]
        {
            eprintln!("Error: DuckDB catalog requires the 'metadata-duckdb' feature");
            exit(1);
        }
    }

    Ok(())
}

async fn run_query(
    provider: Arc<dyn MetadataProvider>,
    snapshot_id: i64,
    sql: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Create runtime and register object stores
    // For MinIO or S3, register the object store with the runtime
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

    // Create the DuckLake catalog bound to the snapshot
    // This ensures all queries through this catalog see consistent data
    // from this specific snapshot, even if the underlying data changes
    let ducklake_catalog = DuckLakeCatalog::with_snapshot(provider.clone(), snapshot_id)?;

    println!("✓ Connected to DuckLake catalog");

    let config = SessionConfig::new().with_default_catalog_and_schema("ducklake", "main");

    // Create DataFusion session context
    let ctx = SessionContext::new_with_config_rt(config, runtime.clone());

    // Register the DuckLake catalog (standard DataFusion pattern)
    DuckLakeCatalog::register(&ctx, "ducklake", ducklake_catalog);

    // Register table functions (ducklake_snapshots, ducklake_table_info, ducklake_list_files)
    register_ducklake_functions(&ctx, provider);

    println!("✓ Registered DuckLake catalog with DataFusion");
    println!(
        "✓ Registered table functions (ducklake_snapshots, ducklake_table_info, ducklake_list_files)"
    );

    // List available schemas
    let catalogs = ctx.catalog_names();
    println!("\nAvailable catalogs: {:?}", catalogs);

    if let Some(catalog) = ctx.catalog("ducklake") {
        let schemas = catalog.schema_names();
        println!("Available schemas in 'ducklake' catalog: {:?}", schemas);

        // List tables in each schema
        for schema_name in &schemas {
            if let Some(schema) = catalog.schema(schema_name) {
                let tables = schema.table_names();
                println!("Available tables in schema '{}': {:?}", schema_name, tables);
            }
        }
    }

    // Execute the query
    println!("\nExecuting query...");
    let df = ctx.sql(sql).await?;

    // Show the query results
    df.show().await?;

    println!("\n✓ Example completed successfully!");

    Ok(())
}
