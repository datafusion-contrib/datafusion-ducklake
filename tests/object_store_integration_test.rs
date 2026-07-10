#![cfg(feature = "metadata-duckdb")]
//! Integration test for object store support (S3/MinIO)
//!
//! This test verifies that DataFusion-DuckLake works correctly with object stores
//! by spinning up a MinIO container, configuring DuckDB to write directly to S3,
//! and running queries against the remote data.

use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::prelude::*;
use datafusion_ducklake::{DuckLakeCatalog, DuckdbMetadataProvider};
use object_store::ObjectStore;
use object_store::aws::AmazonS3Builder;
use std::sync::Arc;
use tempfile::TempDir;
use testcontainers::runners::AsyncRunner;
use testcontainers_modules::minio::MinIO;

/// Helper to create test data using DuckDB with local filesystem
async fn create_local_test_catalog(catalog_path: &str) -> anyhow::Result<()> {
    // Use in-memory database to avoid conflicts
    let conn = duckdb::Connection::open_in_memory()?;

    // Install and load ducklake extension
    conn.execute("INSTALL ducklake;", [])?;
    conn.execute("LOAD ducklake;", [])?;

    // Create a test table with some data (local filesystem)
    conn.execute(
        &format!("ATTACH 'ducklake:{}' AS test_catalog;", catalog_path),
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

/// Helper to create test data using DuckDB with S3/MinIO data path
async fn create_s3_test_catalog(
    catalog_path: &str,
    s3_endpoint: &str,
    bucket_name: &str,
    data_path: &str,
) -> anyhow::Result<()> {
    // Use in-memory database to avoid conflicts
    let conn = duckdb::Connection::open_in_memory()?;

    // Install and load ducklake extension
    conn.execute("INSTALL ducklake;", [])?;
    conn.execute("LOAD ducklake;", [])?;

    eprintln!("Configuring S3 secret for endpoint: {}", s3_endpoint);

    // Create S3 secret for MinIO
    // Extract host:port from endpoint
    let endpoint_without_protocol = s3_endpoint
        .trim_start_matches("http://")
        .trim_start_matches("https://");

    eprintln!("Endpoint without protocol: {}", endpoint_without_protocol);

    let create_secret_sql = format!(
        "CREATE SECRET s3_secret (
            TYPE S3,
            KEY_ID 'minioadmin',
            SECRET 'minioadmin',
            REGION 'us-east-1',
            ENDPOINT '{}',
            USE_SSL false,
            URL_STYLE 'path',
            URL_COMPATIBILITY_MODE true
        );",
        endpoint_without_protocol
    );

    eprintln!("Creating secret with SQL: {}", create_secret_sql);
    conn.execute(&create_secret_sql, [])?;
    eprintln!("S3 secret created");

    // Load httpfs extension for S3 support
    eprintln!("Loading httpfs extension...");
    conn.execute("INSTALL httpfs;", [])?;
    conn.execute("LOAD httpfs;", [])?;
    eprintln!("httpfs loaded");

    // Try setting S3 options directly via SET commands
    eprintln!("Setting S3 configuration via SET commands...");
    conn.execute(
        &format!("SET s3_endpoint='{}';", endpoint_without_protocol),
        [],
    )?;
    conn.execute("SET s3_access_key_id='minioadmin';", [])?;
    conn.execute("SET s3_secret_access_key='minioadmin';", [])?;
    conn.execute("SET s3_use_ssl=false;", [])?;
    conn.execute("SET s3_url_style='path';", [])?;
    conn.execute("SET s3_region='us-east-1';", [])?;
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
        "ATTACH 'ducklake:{}' AS test_catalog (DATA_PATH '{}');",
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
#[cfg_attr(feature = "skip-tests-with-docker", ignore)]
async fn test_minio_object_store_integration() -> anyhow::Result<()> {
    // Tests DataFusion-DuckLake reading from S3/MinIO with DuckDB-created data

    // Skip test if Docker is not available
    if !is_docker_available().await {
        eprintln!("Skipping MinIO integration test: Docker not available");
        return Err(anyhow::anyhow!("MinIO integration tests not available"));
    }

    // Start MinIO container using the testcontainers MinIO module
    let minio = MinIO::default().start().await?;
    let minio_port = minio.get_host_port_ipv4(9000).await?;
    let minio_endpoint = format!("http://127.0.0.1:{}", minio_port);

    eprintln!("MinIO started on {}", minio_endpoint);

    // Create bucket using AWS SDK (following testcontainers-modules pattern)
    eprintln!("Creating test bucket using AWS SDK...");
    use aws_credential_types::Credentials;
    use aws_sdk_s3::config::{Region, SharedCredentialsProvider};

    let creds = Credentials::new("minioadmin", "minioadmin", None, None, "test");
    let s3_config = aws_sdk_s3::Config::builder()
        .endpoint_url(&minio_endpoint)
        .region(Region::new("us-east-1"))
        .credentials_provider(SharedCredentialsProvider::new(creds))
        .force_path_style(true)
        .behavior_version_latest()
        .build();

    let s3_client = aws_sdk_s3::Client::from_conf(s3_config);

    let bucket_name = "test-bucket";
    s3_client.create_bucket().bucket(bucket_name).send().await?;

    eprintln!("Bucket '{}' created successfully", bucket_name);

    // Verify bucket exists by listing
    let buckets = s3_client.list_buckets().send().await?.buckets;
    eprintln!(
        "Found {} bucket(s) in MinIO",
        buckets.as_ref().map_or(0, |b| b.len())
    );

    // Create temporary directory for test catalog metadata
    let temp_dir = TempDir::new()?;
    let catalog_path = temp_dir.path().join("test_catalog.ducklake");
    let catalog_path_str = catalog_path.to_string_lossy().to_string();

    // Generate test data - DuckDB writes directly to S3
    eprintln!("Generating test data on S3...");
    create_s3_test_catalog(
        &catalog_path_str,
        &minio_endpoint,
        "test-bucket",
        "ducklake-data/",
    )
    .await?;
    eprintln!("Test data written to MinIO");

    // Configure S3 client for DataFusion
    let s3_client: Arc<dyn ObjectStore> = Arc::new(
        AmazonS3Builder::new()
            .with_endpoint(&minio_endpoint)
            .with_bucket_name("test-bucket")
            .with_access_key_id("minioadmin")
            .with_secret_access_key("minioadmin")
            .with_region("us-east-1")
            .with_allow_http(true)
            .build()?,
    );

    // Register object store with DataFusion runtime
    let runtime = Arc::new(RuntimeEnvBuilder::new().build()?);
    runtime.register_object_store(&url::Url::parse("s3://test-bucket")?, s3_client.clone());

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

    eprintln!("All MinIO integration tests passed");

    Ok(())
}

/// Check if Docker is available
async fn is_docker_available() -> bool {
    tokio::process::Command::new("docker")
        .arg("--version")
        .output()
        .await
        .is_ok()
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
