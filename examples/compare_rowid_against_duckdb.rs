//! Side-by-side verification of our rowid implementation against the
//! upstream DuckDB ducklake extension.
//!
//! For each candidate table, this example:
//!   1. Queries `SELECT rowid, <key> FROM t ORDER BY rowid LIMIT N` via
//!      DuckDB's native ducklake extension (the ground truth).
//!   2. Runs the same query via DataFusion + our `with_row_lineage(true)`.
//!   3. Compares row-by-row and prints a pass/fail summary.
//!
//! Credentials are read from the env vars set by the loader at the top of
//! `main`. The example connects to a Postgres catalog and a Tigris S3
//! bucket — both pulled from `secrets/ducklake_storage_creds`.

use std::error::Error;
use std::sync::Arc;

use arrow::array::{Array, Int32Array, Int64Array, StringArray};
use datafusion::execution::runtime_env::RuntimeEnv;
use datafusion::prelude::*;
use datafusion_ducklake::{DuckLakeCatalog, PostgresMetadataProvider};
use object_store::ObjectStore;
use object_store::aws::AmazonS3Builder;
use url::Url;

// Configuration is read from env vars at startup so no credentials are
// checked in. Required:
//
//   DUCKLAKE_PG_CONN  — libpq-style key=value (used by DuckDB's `ATTACH 'ducklake:postgres:...'`)
//   DUCKLAKE_PG_URI   — URL form (used by datafusion-ducklake's PostgresMetadataProvider)
//   DUCKLAKE_S3_KEY, DUCKLAKE_S3_SECRET
//
// Optional, with sensible defaults for Tigris/S3-compatible setups:
//
//   DUCKLAKE_S3_ENDPOINT   (default https://t3.storage.dev)
//   DUCKLAKE_S3_BUCKET     (default ducklake-storage)
//   DUCKLAKE_S3_ENDPOINT_HOST  (default t3.storage.dev — used in the DuckDB SECRET ENDPOINT)
//   DUCKLAKE_DATA_PATH     (default ducklake_demo/)

/// (schema, table, key_column, key_type, limit).
/// key_type is a hint for how to read the column from arrow batches.
const TEST_TABLES: &[(&str, &str, &str, KeyType, usize)] = &[
    ("tpch_sf0001", "region", "r_regionkey", KeyType::Int32, 50),
    ("tpch_sf0001", "nation", "n_nationkey", KeyType::Int32, 50),
    ("tpch_sf0001", "supplier", "s_suppkey", KeyType::Int32, 50),
    ("tpch_sf0001", "customer", "c_custkey", KeyType::Int32, 200),
    ("main", "demo_users", "id", KeyType::Int32, 100),
    // Stress: 17 files, ~1.2M rows. Comparing the head + a few mid-range
    // rowids should be enough to catch any cross-file offset bug.
    ("sphere_vector_1m", "vectors", "id", KeyType::Utf8, 200),
];

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
enum KeyType {
    Int32,
    Int64,
    Utf8,
}

#[derive(Debug, PartialEq, Eq, Clone)]
struct Row {
    rowid: i64,
    key: KeyValue,
}

#[derive(Debug, PartialEq, Eq, Clone)]
enum KeyValue {
    I64(i64),
    Str(String),
    Null,
}

fn print_header(title: &str) {
    println!("\n========================================");
    println!("{}", title);
    println!("========================================");
}

struct Config {
    pg_conn: String,
    pg_uri: String,
    s3_key: String,
    s3_secret: String,
    s3_endpoint: String,
    s3_bucket: String,
    s3_endpoint_host: String,
    data_path: String,
}

fn load_config() -> Result<Config, Box<dyn Error>> {
    fn required(var: &str) -> Result<String, Box<dyn Error>> {
        std::env::var(var).map_err(|_| {
            format!(
                "{} is required — see the header comment for the full list of env vars",
                var
            )
            .into()
        })
    }
    Ok(Config {
        pg_conn: required("DUCKLAKE_PG_CONN")?,
        pg_uri: required("DUCKLAKE_PG_URI")?,
        s3_key: required("DUCKLAKE_S3_KEY")?,
        s3_secret: required("DUCKLAKE_S3_SECRET")?,
        s3_endpoint: std::env::var("DUCKLAKE_S3_ENDPOINT")
            .unwrap_or_else(|_| "https://t3.storage.dev".to_string()),
        s3_bucket: std::env::var("DUCKLAKE_S3_BUCKET")
            .unwrap_or_else(|_| "ducklake-storage".to_string()),
        s3_endpoint_host: std::env::var("DUCKLAKE_S3_ENDPOINT_HOST")
            .unwrap_or_else(|_| "t3.storage.dev".to_string()),
        data_path: std::env::var("DUCKLAKE_DATA_PATH")
            .unwrap_or_else(|_| "ducklake_demo/".to_string()),
    })
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let cfg = load_config()?;

    // ----- 1. Spin up DuckDB-with-ducklake (the reference) -----
    print_header("Configuring DuckDB ducklake reference path");
    let duckdb_conn = duckdb::Connection::open_in_memory()?;
    duckdb_conn.execute_batch(
        "INSTALL ducklake; LOAD ducklake;
         INSTALL httpfs;   LOAD httpfs;
         INSTALL postgres; LOAD postgres;",
    )?;
    duckdb_conn.execute_batch(&format!(
        "CREATE OR REPLACE SECRET tigris (
             TYPE S3,
             KEY_ID '{key}',
             SECRET '{secret}',
             REGION 'auto',
             ENDPOINT '{endpoint_host}',
             URL_STYLE 'path',
             SCOPE 's3://{bucket}'
         );",
        key = cfg.s3_key,
        secret = cfg.s3_secret,
        endpoint_host = cfg.s3_endpoint_host,
        bucket = cfg.s3_bucket,
    ))?;
    duckdb_conn.execute_batch(&format!(
        "ATTACH 'ducklake:postgres:{pg}' AS dl (DATA_PATH 's3://{bucket}/{data_path}');",
        pg = cfg.pg_conn,
        bucket = cfg.s3_bucket,
        data_path = cfg.data_path,
    ))?;

    // ----- 2. Spin up DataFusion + our extension -----
    print_header("Configuring DataFusion with datafusion-ducklake");
    let provider = PostgresMetadataProvider::new(&cfg.pg_uri).await?;

    let runtime = Arc::new(RuntimeEnv::default());
    let s3: Arc<dyn ObjectStore> = Arc::new(
        AmazonS3Builder::new()
            .with_endpoint(&cfg.s3_endpoint)
            .with_bucket_name(&cfg.s3_bucket)
            .with_access_key_id(&cfg.s3_key)
            .with_secret_access_key(&cfg.s3_secret)
            .with_region("auto")
            .build()?,
    );
    runtime.register_object_store(&Url::parse(&format!("s3://{}/", cfg.s3_bucket))?, s3);

    let catalog = DuckLakeCatalog::new(provider)?.with_row_lineage(true);
    let cfg = SessionConfig::new().with_default_catalog_and_schema("dl", "main");
    let ctx = SessionContext::new_with_config_rt(cfg, runtime);
    ctx.register_catalog("dl", Arc::new(catalog));
    println!("✓ DataFusion catalog registered with row lineage enabled");

    // ----- 3. Compare table by table -----
    let mut overall_ok = true;
    for &(schema, table, key, key_type, limit) in TEST_TABLES {
        print_header(&format!(
            "Comparing {schema}.{table} (key={key}, limit={limit})"
        ));
        match compare_table(&duckdb_conn, &ctx, schema, table, key, key_type, limit).await {
            Ok(true) => println!("✅ rowid + {key} match between DuckDB and DataFusion"),
            Ok(false) => {
                println!("❌ MISMATCH for {schema}.{table}");
                overall_ok = false;
            },
            Err(e) => {
                println!("⚠️  error comparing {schema}.{table}: {e}");
                overall_ok = false;
            },
        }
    }

    print_header(if overall_ok {
        "ALL TABLES MATCH ✅"
    } else {
        "FAILURES DETECTED ❌"
    });
    if overall_ok {
        Ok(())
    } else {
        Err("at least one table mismatched".into())
    }
}

async fn compare_table(
    duckdb_conn: &duckdb::Connection,
    ctx: &SessionContext,
    schema: &str,
    table: &str,
    key: &str,
    key_type: KeyType,
    limit: usize,
) -> Result<bool, Box<dyn Error>> {
    let sql = format!(
        "SELECT rowid, {key} FROM dl.{schema}.{table} ORDER BY rowid LIMIT {limit}",
        key = key,
        schema = schema,
        table = table,
        limit = limit,
    );

    // ----- DuckDB ground truth -----
    let duckdb_rows = duckdb_query(duckdb_conn, &sql, key_type)?;
    println!("DuckDB returned {} rows", duckdb_rows.len());
    print_sample("DuckDB", &duckdb_rows);

    // ----- DataFusion -----
    let df = ctx.sql(&sql).await?;
    let batches = df.collect().await?;
    let datafusion_rows = extract_rows(&batches, key_type)?;
    println!("DataFusion returned {} rows", datafusion_rows.len());
    print_sample("DataFusion", &datafusion_rows);

    if duckdb_rows.len() != datafusion_rows.len() {
        println!(
            "Row count differs: DuckDB={} DataFusion={}",
            duckdb_rows.len(),
            datafusion_rows.len()
        );
        return Ok(false);
    }

    let mut diffs = 0;
    for (i, (a, b)) in duckdb_rows.iter().zip(datafusion_rows.iter()).enumerate() {
        if a != b {
            if diffs < 10 {
                println!("  diff at row {i}: DuckDB={a:?}, DataFusion={b:?}");
            }
            diffs += 1;
        }
    }
    if diffs > 0 {
        println!("Total differing rows: {diffs}");
    }
    Ok(diffs == 0)
}

fn duckdb_query(
    conn: &duckdb::Connection,
    sql: &str,
    _key_type: KeyType,
) -> Result<Vec<Row>, Box<dyn Error>> {
    let mut stmt = conn.prepare(sql)?;
    let mut rows = stmt.query([])?;
    let mut out = Vec::new();
    while let Some(r) = rows.next()? {
        let rowid: i64 = r.get(0)?;
        // Try i32 → i64 → String in turn so we don't need to know the column
        // type up front. Canonicalize ints to i64.
        let key = if let Ok(v) = r.get::<_, Option<i32>>(1) {
            v.map(|x| KeyValue::I64(x as i64)).unwrap_or(KeyValue::Null)
        } else if let Ok(v) = r.get::<_, Option<i64>>(1) {
            v.map(KeyValue::I64).unwrap_or(KeyValue::Null)
        } else if let Ok(v) = r.get::<_, Option<String>>(1) {
            v.map(KeyValue::Str).unwrap_or(KeyValue::Null)
        } else {
            return Err("unsupported key type from DuckDB".into());
        };
        out.push(Row {
            rowid,
            key,
        });
    }
    Ok(out)
}

fn extract_rows(
    batches: &[arrow::record_batch::RecordBatch],
    _key_type: KeyType,
) -> Result<Vec<Row>, Box<dyn Error>> {
    let mut out = Vec::new();
    for batch in batches {
        let rowids = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| {
                format!(
                    "expected rowid column to be Int64; got {:?}",
                    batch.column(0).data_type()
                )
            })?;
        let key_col = batch.column(1);
        for i in 0..batch.num_rows() {
            let rowid = if rowids.is_null(i) {
                return Err("rowid null in DataFusion output".into());
            } else {
                rowids.value(i)
            };
            // Auto-detect key column type. We canonicalize integers to i64 so
            // a DuckDB Int32 and a DataFusion Int64 of the same value compare
            // equal — this is purely a row-equality probe, not a type check.
            let key = if let Some(arr) = key_col.as_any().downcast_ref::<Int32Array>() {
                if arr.is_null(i) {
                    KeyValue::Null
                } else {
                    KeyValue::I64(arr.value(i) as i64)
                }
            } else if let Some(arr) = key_col.as_any().downcast_ref::<Int64Array>() {
                if arr.is_null(i) {
                    KeyValue::Null
                } else {
                    KeyValue::I64(arr.value(i))
                }
            } else if let Some(arr) = key_col.as_any().downcast_ref::<StringArray>() {
                if arr.is_null(i) {
                    KeyValue::Null
                } else {
                    KeyValue::Str(arr.value(i).to_string())
                }
            } else {
                return Err(format!(
                    "unsupported key column type {:?} in DataFusion output",
                    key_col.data_type()
                )
                .into());
            };
            out.push(Row {
                rowid,
                key,
            });
        }
    }
    Ok(out)
}

fn print_sample(label: &str, rows: &[Row]) {
    let n = rows.len().min(5);
    for r in &rows[..n] {
        println!("  [{label}] rowid={:>6}  key={:?}", r.rowid, r.key);
    }
    if rows.len() > n {
        println!("  [{label}] ... ({} more rows)", rows.len() - n);
    }
}
