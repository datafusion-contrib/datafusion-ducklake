#![cfg(feature = "metadata-duckdb")]
//! Integration test for object store support (S3)
//!
//! This test verifies that DataFusion-DuckLake works correctly with object stores
//! by starting an in-process S3 server (see `common::s3`), configuring DuckDB to
//! write directly to S3, and running queries against the remote data.

use crate::common::s3::{REGION, S3Server};
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::prelude::*;
use datafusion_ducklake::{DuckLakeCatalog, DuckdbMetadataProvider};
use object_store::ObjectStore;
use object_store::aws::AmazonS3Builder;
use std::sync::Arc;
use tempfile::TempDir;

/// Helper to create test data using DuckDB with local filesystem
async fn create_local_test_catalog(catalog_path: &str) -> anyhow::Result<()> {
    // Use in-memory database to avoid conflicts
    let conn = duckdb::Connection::open_in_memory()?;

    // Install and load ducklake extension
    crate::common::ensure_ducklake_installed();
    conn.execute("LOAD ducklake;", [])?;

    // Create a test table with some data (local filesystem)
    conn.execute(
        &format!(
            "ATTACH 'ducklake:{}' AS test_catalog (DATA_INLINING_ROW_LIMIT 0);",
            catalog_path
        ),
        [],
    )?;

    conn.execute(
        "CREATE TABLE test_catalog.products (
            id INT,
            name VARCHAR,
            price DECIMAL(10,2),
            in_stock BOOLEAN
        );",
        [],
    )?;

    conn.execute(
        "INSERT INTO test_catalog.products VALUES
            (1, 'Laptop', 999.99, true),
            (2, 'Mouse', 25.50, true),
            (3, 'Keyboard', 75.00, true),
            (4, 'Monitor', 299.99, false),
            (5, 'Webcam', 89.99, true);",
        [],
    )?;

    Ok(())
}

/// Helper to create test data using DuckDB with an S3 data path
async fn create_s3_test_catalog(
    catalog_path: &str,
    s3: &S3Server,
    bucket_name: &str,
    data_path: &str,
) -> anyhow::Result<()> {
    // Use in-memory database to avoid conflicts
    let conn = duckdb::Connection::open_in_memory()?;

    // Install and load ducklake extension
    crate::common::ensure_ducklake_installed();
    conn.execute("LOAD ducklake;", [])?;

    eprintln!("Configuring S3 secret for endpoint: {}", s3.endpoint_url);

    let create_secret_sql = format!(
        "CREATE SECRET s3_secret (
            TYPE S3,
            KEY_ID '{}',
            SECRET '{}',
            REGION '{REGION}',
            ENDPOINT '{}',
            USE_SSL {},
            URL_STYLE 'path',
            URL_COMPATIBILITY_MODE true
        );",
        s3.user,
        s3.password,
        s3.host_port(),
        s3.use_ssl()
    );

    conn.execute(&create_secret_sql, [])?;
    eprintln!("S3 secret created");

    // Load httpfs extension for S3 support
    eprintln!("Loading httpfs extension...");
    crate::common::ensure_extension_installed("httpfs");
    conn.execute("LOAD httpfs;", [])?;
    eprintln!("httpfs loaded");

    // Try setting S3 options directly via SET commands
    eprintln!("Setting S3 configuration via SET commands...");
    conn.execute(&format!("SET s3_endpoint='{}';", s3.host_port()), [])?;
    conn.execute(&format!("SET s3_access_key_id='{}';", s3.user), [])?;
    conn.execute(&format!("SET s3_secret_access_key='{}';", s3.password), [])?;
    conn.execute(&format!("SET s3_use_ssl={};", s3.use_ssl()), [])?;
    conn.execute("SET s3_url_style='path';", [])?;
    conn.execute(&format!("SET s3_region='{REGION}';"), [])?;
    eprintln!("S3 configuration set");

    // Test S3 write capability with a simple table
    eprintln!("Testing S3 write capability...");
    let test_table_sql = "CREATE TABLE test_write AS SELECT 1 as id, 'test' as name;";
    conn.execute(test_table_sql, [])?;

    let test_write_path = format!("s3://{}/test-write.parquet", bucket_name);
    let copy_sql = format!("COPY test_write TO '{}';", test_write_path);
    eprintln!("Attempting to write to S3: {}", copy_sql);

    match conn.execute(&copy_sql, []) {
        Ok(_) => {
            eprintln!("S3 write successful");

            // Try to read it back
            let read_sql = format!("SELECT * FROM read_parquet('{}');", test_write_path);
            match conn.execute(&read_sql, []) {
                Ok(_) => eprintln!("S3 read successful"),
                Err(e) => eprintln!("S3 read failed: {}", e),
            }
        },
        Err(e) => {
            eprintln!("S3 write failed: {}", e);
            return Err(anyhow::anyhow!("Failed to write to S3: {}", e));
        },
    }

    // Attach DuckLake catalog with S3 data path
    let s3_data_path = format!("s3://{}/{}", bucket_name, data_path);
    eprintln!("Attaching catalog with DATA_PATH: {}", s3_data_path);

    let attach_sql = format!(
        "ATTACH 'ducklake:{}' AS test_catalog (DATA_PATH '{}', DATA_INLINING_ROW_LIMIT 0);",
        catalog_path, s3_data_path
    );

    eprintln!("Attach SQL: {}", attach_sql);
    conn.execute(&attach_sql, [])?;
    eprintln!("Catalog attached");

    // Create tables and insert data - DuckDB will write Parquet files directly to S3
    conn.execute(
        "CREATE TABLE test_catalog.products (
            id INT,
            name VARCHAR,
            price DECIMAL(10,2),
            in_stock BOOLEAN
        );",
        [],
    )?;

    conn.execute(
        "INSERT INTO test_catalog.products VALUES
            (1, 'Laptop', 999.99, true),
            (2, 'Mouse', 25.50, true),
            (3, 'Keyboard', 75.00, true),
            (4, 'Monitor', 299.99, false),
            (5, 'Webcam', 89.99, true);",
        [],
    )?;

    // Create a table with delete operations
    conn.execute(
        "CREATE TABLE test_catalog.inventory (
            id INT,
            product_name VARCHAR,
            quantity INT
        );",
        [],
    )?;

    conn.execute(
        "INSERT INTO test_catalog.inventory VALUES
            (1, 'Widget A', 100),
            (2, 'Widget B', 200),
            (3, 'Widget C', 150),
            (4, 'Widget D', 75);",
        [],
    )?;

    // Delete some rows to create delete files on S3
    conn.execute("DELETE FROM test_catalog.inventory WHERE id = 2;", [])?;
    conn.execute("DELETE FROM test_catalog.inventory WHERE id = 4;", [])?;

    Ok(())
}

#[tokio::test]
async fn test_s3_object_store_integration() -> anyhow::Result<()> {
    // Tests DataFusion-DuckLake reading from S3 with DuckDB-created data
    let s3 = S3Server::start();
    eprintln!("S3 server on {}", s3.endpoint_url);

    let bucket_name = s3.create_bucket("test-bucket").await?;
    eprintln!("Bucket '{}' created successfully", bucket_name);

    // Create temporary directory for test catalog metadata
    let temp_dir = TempDir::new()?;
    let catalog_path = temp_dir.path().join("test_catalog.ducklake");
    let catalog_path_str = catalog_path.to_string_lossy().to_string();

    // Generate test data - DuckDB writes directly to S3
    eprintln!("Generating test data on S3...");
    create_s3_test_catalog(&catalog_path_str, &s3, &bucket_name, "ducklake-data/").await?;
    assert_parquet_data_files(
        &catalog_path_str,
        &[("inventory", 1, 4), ("products", 1, 5)],
    )?;
    eprintln!("Test data written to S3");

    // Configure S3 client for DataFusion
    let s3_client: Arc<dyn ObjectStore> = Arc::new(
        AmazonS3Builder::new()
            .with_endpoint(&s3.endpoint_url)
            .with_bucket_name(&bucket_name)
            .with_access_key_id(&s3.user)
            .with_secret_access_key(&s3.password)
            .with_region(REGION)
            .with_allow_http(true)
            .build()?,
    );

    // Register object store with DataFusion runtime
    let runtime = Arc::new(RuntimeEnvBuilder::new().build()?);
    runtime.register_object_store(
        &url::Url::parse(&format!("s3://{bucket_name}"))?,
        s3_client.clone(),
    );

    // Create session context with the runtime
    let session_config = SessionConfig::new();
    let ctx = SessionContext::new_with_config_rt(session_config, runtime);

    // Create DuckLake catalog provider
    let provider = DuckdbMetadataProvider::new(&catalog_path_str)?;
    let catalog = Arc::new(DuckLakeCatalog::new(provider)?);
    ctx.register_catalog("ducklake", catalog);

    // Test 1: Query table without deletes
    eprintln!("Testing query on table without deletes...");
    let df = ctx
        .sql("SELECT * FROM ducklake.main.products ORDER BY id")
        .await?;
    let results = df.collect().await?;

    assert_eq!(results.len(), 1, "Expected 1 batch");
    assert_eq!(results[0].num_rows(), 5, "Expected 5 products");

    // Verify first product
    let id_col = results[0]
        .column(0)
        .as_any()
        .downcast_ref::<arrow::array::Int32Array>()
        .expect("id column should be Int32");
    // DuckLake string columns scan as Utf8View; cast to Utf8 to read via StringArray.
    let name_col_arr =
        arrow::compute::cast(results[0].column(1), &arrow::datatypes::DataType::Utf8)
            .expect("cast name column to Utf8");
    let name_col = name_col_arr
        .as_any()
        .downcast_ref::<arrow::array::StringArray>()
        .expect("name column should be String");

    assert_eq!(id_col.value(0), 1);
    assert_eq!(name_col.value(0), "Laptop");

    eprintln!("Query on table without deletes successful");

    // Test 2: Query table with deletes
    eprintln!("Testing query on table with deletes...");
    let df = ctx
        .sql("SELECT * FROM ducklake.main.inventory ORDER BY id")
        .await?;
    let results = df.collect().await?;

    // Should only return rows 1 and 3 (rows 2 and 4 were deleted)
    let mut total_rows = 0;
    for batch in &results {
        total_rows += batch.num_rows();
    }
    assert_eq!(total_rows, 2, "Expected 2 rows after deletes");

    let id_col = results[0]
        .column(0)
        .as_any()
        .downcast_ref::<arrow::array::Int32Array>()
        .expect("id column should be Int32");

    assert_eq!(id_col.value(0), 1, "First row should have id=1");
    assert_eq!(id_col.value(1), 3, "Second row should have id=3");

    eprintln!("Query on table with deletes successful");

    // Test 3: Aggregation query
    eprintln!("Testing aggregation query...");
    let df = ctx
        .sql("SELECT COUNT(*) as count FROM ducklake.main.products")
        .await?;
    let results = df.collect().await?;

    let count_col = results[0]
        .column(0)
        .as_any()
        .downcast_ref::<arrow::array::Int64Array>()
        .expect("count column should be Int64");

    assert_eq!(count_col.value(0), 5, "Expected count of 5");
    eprintln!("Aggregation query successful");

    // Test 4: Filter query
    eprintln!("Testing filter query...");
    let df = ctx
        .sql("SELECT name, price FROM ducklake.main.products WHERE price > 100 ORDER BY price")
        .await?;
    let results = df.collect().await?;

    assert!(
        results[0].num_rows() >= 2,
        "Expected at least 2 products with price > 100"
    );
    eprintln!("Filter query successful");

    eprintln!("All S3 integration tests passed");

    Ok(())
}

#[tokio::test]
async fn test_local_filesystem_with_s3_style_paths() -> anyhow::Result<()> {
    // This test ensures our path resolution works for local filesystem paths

    let temp_dir = TempDir::new()?;
    let catalog_path = temp_dir.path().join("local_test.ducklake");
    let catalog_path_str = catalog_path.to_string_lossy().to_string();

    // Generate test data locally
    eprintln!("Generating local test data...");
    create_local_test_catalog(&catalog_path_str).await?;
    assert_parquet_data_files(&catalog_path_str, &[("products", 1, 5)])?;

    // Create session context
    let ctx = SessionContext::new();

    // Create DuckLake catalog provider
    let provider = DuckdbMetadataProvider::new(&catalog_path_str)?;
    let catalog = Arc::new(DuckLakeCatalog::new(provider)?);
    ctx.register_catalog("ducklake", catalog);

    // Test basic query
    let df = ctx
        .sql("SELECT COUNT(*) as count FROM ducklake.main.products")
        .await?;
    let results = df.collect().await?;

    let count_col = results[0]
        .column(0)
        .as_any()
        .downcast_ref::<arrow::array::Int64Array>()
        .expect("count column should be Int64");

    assert_eq!(count_col.value(0), 5, "Expected count of 5");

    eprintln!("Local filesystem test passed");

    Ok(())
}

fn assert_parquet_data_files(
    catalog_path: &str,
    expected: &[(&str, i64, i64)],
) -> anyhow::Result<()> {
    let connection = duckdb::Connection::open(catalog_path)?;
    let mut statement = connection.prepare(
        "SELECT t.table_name, COUNT(*), SUM(f.record_count)::BIGINT
         FROM ducklake_data_file f JOIN ducklake_table t ON t.table_id = f.table_id
         WHERE f.end_snapshot IS NULL AND t.end_snapshot IS NULL
         GROUP BY t.table_name ORDER BY t.table_name",
    )?;
    let actual = statement
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, i64>(2)?,
            ))
        })?
        .collect::<Result<Vec<_>, _>>()?;
    let expected = expected
        .iter()
        .map(|(name, files, rows)| (name.to_string(), *files, *rows))
        .collect::<Vec<_>>();
    assert_eq!(actual, expected);
    Ok(())
}
