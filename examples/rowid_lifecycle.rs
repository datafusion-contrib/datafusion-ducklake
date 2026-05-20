//! Full lifecycle comparison of rowid behavior between the upstream DuckDB
//! ducklake extension and datafusion-ducklake.
//!
//! Writes are issued exclusively through DuckDB (the source of truth);
//! after each mutation step we read the same query through BOTH engines and
//! verify the row sets match exactly.
//!
//! Steps:
//!   1. CREATE TABLE + two INSERT batches (→ 2 data files, distinct row_id_start)
//!   2. UPDATE half the rows  (→ a new file with embedded `_ducklake_internal_row_id`)
//!   3. DELETE one row        (→ a delete file)
//!   4. DELETE all rows       (→ table-wide delete)
//!   5. DROP the table        (cleanup)
//!
//! The table lives at `main._rowid_verify_demo` and is always dropped at the
//! end of the run, including on failure.

use std::error::Error;
use std::sync::Arc;

use arrow::array::{Array, Int32Array, Int64Array, StringArray};
use arrow::record_batch::RecordBatch;
use datafusion::execution::runtime_env::RuntimeEnv;
use datafusion::prelude::*;
use datafusion_ducklake::{DuckLakeCatalog, PostgresMetadataProvider};
use object_store::ObjectStore;
use object_store::aws::AmazonS3Builder;
use url::Url;

// Configuration is read from env vars at startup so no credentials are
// checked in. See compare_rowid_against_duckdb.rs for the full list.

const TABLE: &str = "_rowid_verify_demo";
const QUERY: &str = "SELECT rowid, id, name FROM dl.main._rowid_verify_demo ORDER BY rowid";

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

#[derive(Debug, PartialEq, Eq, Clone)]
struct Row {
    rowid: i64,
    id: i64,
    name: Option<String>,
}

fn header(title: &str) {
    println!("\n========================================");
    println!("{title}");
    println!("========================================");
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let cfg = load_config()?;

    header("Setting up DuckDB ducklake reference path");
    let duckdb_conn = setup_duckdb(&cfg)?;

    header("Setting up S3 runtime for DataFusion");
    let runtime = setup_runtime(&cfg)?;

    // Always run cleanup at the end, success or failure.
    let result = run_lifecycle(&duckdb_conn, &runtime, &cfg).await;
    let _ = drop_table(&duckdb_conn);
    println!("\n(cleanup) ✓ dropped {TABLE}");
    result
}

async fn run_lifecycle(
    duckdb_conn: &duckdb::Connection,
    runtime: &Arc<RuntimeEnv>,
    cfg: &Config,
) -> Result<(), Box<dyn Error>> {
    // Defensive cleanup of any leftover from a prior failed run.
    let _ = drop_table(duckdb_conn);

    let mut all_passed = true;

    // -------------------------------------------------------------------
    // Step 1: CREATE TABLE + two INSERT batches → multi-file synthesized rowid
    // -------------------------------------------------------------------
    header("Step 1: CREATE TABLE + two INSERT batches");
    duckdb_conn.execute_batch(&format!(
        "CREATE TABLE dl.main.{TABLE}(id INTEGER, name VARCHAR);"
    ))?;
    duckdb_conn.execute_batch(&format!(
        "INSERT INTO dl.main.{TABLE} VALUES (1, 'alice'), (2, 'bob'), (3, 'charlie');"
    ))?;
    duckdb_conn.execute_batch(&format!(
        "INSERT INTO dl.main.{TABLE} VALUES (4, 'dave'), (5, 'eve');"
    ))?;
    all_passed &= compare_step("After two INSERTs", duckdb_conn, runtime, cfg).await?;

    // -------------------------------------------------------------------
    // Step 2: UPDATE → exercises embedded `_ducklake_internal_row_id` path
    // -------------------------------------------------------------------
    header("Step 2: UPDATE id IN (1, 3) → file rewrite with embedded rowids");
    duckdb_conn.execute_batch(&format!(
        "UPDATE dl.main.{TABLE} SET name = name || '_v2' WHERE id IN (1, 3);"
    ))?;
    all_passed &= compare_step("After UPDATE", duckdb_conn, runtime, cfg).await?;

    // -------------------------------------------------------------------
    // Step 3: DELETE a single row → exercises DeleteFilterExec
    // -------------------------------------------------------------------
    header("Step 3: DELETE id=4");
    duckdb_conn.execute_batch(&format!("DELETE FROM dl.main.{TABLE} WHERE id = 4;"))?;
    all_passed &= compare_step("After DELETE id=4", duckdb_conn, runtime, cfg).await?;

    // -------------------------------------------------------------------
    // Step 4: DELETE all rows → empty result set
    // -------------------------------------------------------------------
    header("Step 4: DELETE all rows");
    duckdb_conn.execute_batch(&format!("DELETE FROM dl.main.{TABLE};"))?;
    all_passed &= compare_step("After DELETE all", duckdb_conn, runtime, cfg).await?;

    if all_passed {
        Ok(())
    } else {
        Err("at least one lifecycle step mismatched".into())
    }
}

fn drop_table(conn: &duckdb::Connection) -> Result<(), Box<dyn Error>> {
    conn.execute_batch(&format!("DROP TABLE IF EXISTS dl.main.{TABLE};"))?;
    Ok(())
}

async fn compare_step(
    label: &str,
    duckdb_conn: &duckdb::Connection,
    runtime: &Arc<RuntimeEnv>,
    cfg: &Config,
) -> Result<bool, Box<dyn Error>> {
    println!("[{label}]");

    // DuckDB read.
    let duckdb_rows = duckdb_read(duckdb_conn, QUERY)?;
    println!("  DuckDB:     {} rows", duckdb_rows.len());
    print_rows("DuckDB", &duckdb_rows);

    // DataFusion read — fresh catalog so we pick up the latest snapshot the
    // DuckDB writes just produced.
    let datafusion_rows = datafusion_read(runtime, cfg, QUERY).await?;
    println!("  DataFusion: {} rows", datafusion_rows.len());
    print_rows("DataFusion", &datafusion_rows);

    if duckdb_rows == datafusion_rows {
        println!("  ✅ MATCH");
        Ok(true)
    } else {
        println!("  ❌ MISMATCH");
        if duckdb_rows.len() != datafusion_rows.len() {
            println!(
                "    row count differs: DuckDB={} vs DataFusion={}",
                duckdb_rows.len(),
                datafusion_rows.len()
            );
        }
        for (i, (a, b)) in duckdb_rows.iter().zip(datafusion_rows.iter()).enumerate() {
            if a != b {
                println!("    diff at row {i}: DuckDB={a:?}  DataFusion={b:?}");
            }
        }
        Ok(false)
    }
}

fn setup_duckdb(cfg: &Config) -> Result<duckdb::Connection, Box<dyn Error>> {
    let conn = duckdb::Connection::open_in_memory()?;
    conn.execute_batch(
        "INSTALL ducklake; LOAD ducklake;
         INSTALL httpfs;   LOAD httpfs;
         INSTALL postgres; LOAD postgres;",
    )?;
    conn.execute_batch(&format!(
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
    conn.execute_batch(&format!(
        "ATTACH 'ducklake:postgres:{pg}' AS dl (DATA_PATH 's3://{bucket}/{data_path}');",
        pg = cfg.pg_conn,
        bucket = cfg.s3_bucket,
        data_path = cfg.data_path,
    ))?;
    println!("✓ DuckDB attached to ducklake catalog");
    Ok(conn)
}

fn setup_runtime(cfg: &Config) -> Result<Arc<RuntimeEnv>, Box<dyn Error>> {
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
    println!("✓ S3 object store registered");
    Ok(runtime)
}

fn duckdb_read(conn: &duckdb::Connection, sql: &str) -> Result<Vec<Row>, Box<dyn Error>> {
    let mut stmt = conn.prepare(sql)?;
    let mut rows = stmt.query([])?;
    let mut out = Vec::new();
    while let Some(r) = rows.next()? {
        let rowid: i64 = r.get(0)?;
        // id was created as INTEGER; DuckDB hands back i32. Canonicalize i64.
        let id: i64 = if let Ok(v) = r.get::<_, Option<i32>>(1) {
            v.unwrap_or(0) as i64
        } else {
            r.get::<_, Option<i64>>(1)?.unwrap_or(0)
        };
        let name: Option<String> = r.get(2)?;
        out.push(Row {
            rowid,
            id,
            name,
        });
    }
    Ok(out)
}

async fn datafusion_read(
    runtime: &Arc<RuntimeEnv>,
    cfg: &Config,
    sql: &str,
) -> Result<Vec<Row>, Box<dyn Error>> {
    // Build a fresh provider + catalog each call so we get the latest
    // snapshot (the catalog binds snapshot_id at construction time).
    let provider = PostgresMetadataProvider::new(&cfg.pg_uri).await?;
    let catalog = DuckLakeCatalog::new(provider)?.with_row_lineage(true);
    let cfg = SessionConfig::new().with_default_catalog_and_schema("dl", "main");
    let ctx = SessionContext::new_with_config_rt(cfg, runtime.clone());
    ctx.register_catalog("dl", Arc::new(catalog));

    let df = ctx.sql(sql).await?;
    let batches = df.collect().await?;
    rows_from_batches(&batches)
}

fn rows_from_batches(batches: &[RecordBatch]) -> Result<Vec<Row>, Box<dyn Error>> {
    let mut out = Vec::new();
    for batch in batches {
        let rowids = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or("rowid column should be Int64")?;
        let id_col = batch.column(1);
        let name_col = batch.column(2);
        for i in 0..batch.num_rows() {
            if rowids.is_null(i) {
                return Err("got NULL rowid from DataFusion".into());
            }
            let rowid = rowids.value(i);

            // id may come back as Int32 or Int64; coerce to i64.
            let id: i64 = if let Some(arr) = id_col.as_any().downcast_ref::<Int32Array>() {
                if arr.is_null(i) {
                    0
                } else {
                    arr.value(i) as i64
                }
            } else if let Some(arr) = id_col.as_any().downcast_ref::<Int64Array>() {
                if arr.is_null(i) {
                    0
                } else {
                    arr.value(i)
                }
            } else {
                return Err(format!("unexpected id type {:?}", id_col.data_type()).into());
            };

            let name = if let Some(arr) = name_col.as_any().downcast_ref::<StringArray>() {
                if arr.is_null(i) {
                    None
                } else {
                    Some(arr.value(i).to_string())
                }
            } else {
                return Err(format!("unexpected name type {:?}", name_col.data_type()).into());
            };

            out.push(Row {
                rowid,
                id,
                name,
            });
        }
    }
    Ok(out)
}

fn print_rows(label: &str, rows: &[Row]) {
    for r in rows {
        println!(
            "    [{label:>10}] rowid={:>4}  id={:>4}  name={:?}",
            r.rowid, r.id, r.name
        );
    }
    if rows.is_empty() {
        println!("    [{label:>10}] (empty)");
    }
}
