//! End-to-end demo of the multicatalog Postgres writer.
//!
//! Walks through:
//! 1. Bootstrapping the catalog tables in Postgres
//! 2. Creating two catalogs ("pg_prod", "mysql_prod")
//! 3. Writing real Parquet data through each catalog via DuckLakeTableWriter
//! 4. Dumping the resulting catalog rows so you can see multi-catalog isolation
//! 5. Reading the Parquet files back through DataFusion to prove the data is real
//!
//! Usage:
//!   cargo run --example multicatalog_write --no-default-features \
//!     --features write-postgres,metadata-postgres -- <POSTGRES_URL> <DATA_DIR>
//!
//! Example:
//!   cargo run --example multicatalog_write --no-default-features \
//!     --features write-postgres,metadata-postgres -- \
//!     "postgresql://postgres:postgres@127.0.0.1:55432/postgres" /tmp/ducklake-mc

use std::sync::Arc;

use arrow::array::{Float64Array, Int32Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::execution::runtime_env::RuntimeEnv;
use datafusion::prelude::*;
use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeTableWriter, MetadataProvider, MetadataWriter, MulticatalogManager,
    MulticatalogProvider, PostgresMetadataWriter, initialize_multicatalog_schema,
};
use object_store::local::LocalFileSystem;
use sqlx::Row;
use sqlx::postgres::PgPoolOptions;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <POSTGRES_URL> <DATA_DIR>", args[0]);
        std::process::exit(1);
    }
    let pg_url = &args[1];
    let data_dir = std::path::PathBuf::from(&args[2]);
    std::fs::create_dir_all(&data_dir)?;
    let data_dir_str = data_dir.canonicalize()?.to_string_lossy().to_string();

    println!("== Multicatalog Postgres writer demo ==");
    println!("postgres : {}", pg_url);
    println!("data dir : {}", data_dir_str);
    println!();

    // ── Step 1: connect + bootstrap schema ────────────────────────────────────
    let pool = PgPoolOptions::new()
        .max_connections(5)
        .connect(pg_url)
        .await?;
    initialize_multicatalog_schema(&pool).await?;
    println!("✓ schema bootstrapped");

    // ── Step 2: create catalogs ───────────────────────────────────────────────
    let mgr = MulticatalogManager::new(pool.clone());
    let cat_pg = mgr.create_catalog("pg_prod").await?;
    let cat_mysql = mgr.create_catalog("mysql_prod").await?;
    println!("✓ catalogs: pg_prod -> {}, mysql_prod -> {}", cat_pg, cat_mysql);

    // ── Step 3: write through each catalog ────────────────────────────────────
    let object_store: Arc<dyn object_store::ObjectStore> = Arc::new(LocalFileSystem::new());

    // pg_prod.public.users
    let writer_pg = Arc::new(PostgresMetadataWriter::with_pool(pool.clone(), cat_pg).await?);
    writer_pg.set_data_path(&data_dir_str)?;
    let users_batch = build_users_batch();
    let tw_pg = DuckLakeTableWriter::new(writer_pg.clone(), Arc::clone(&object_store))?;
    let users_result = tw_pg
        .write_table("public", "users", std::slice::from_ref(&users_batch))
        .await?;
    println!(
        "✓ wrote pg_prod.public.users — snapshot {}, file count {}, rows {}",
        users_result.snapshot_id, users_result.files_written, users_result.records_written
    );

    // pg_prod.public.users again (DML, same schema) — should carry forward schema_version.
    let users_dml = tw_pg
        .write_table("public", "users", std::slice::from_ref(&users_batch))
        .await?;
    println!(
        "✓ wrote pg_prod.public.users AGAIN (DML) — snapshot {}",
        users_dml.snapshot_id
    );

    // mysql_prod.public.orders
    let writer_mysql =
        Arc::new(PostgresMetadataWriter::with_pool(pool.clone(), cat_mysql).await?);
    let orders_batch = build_orders_batch();
    let tw_mysql = DuckLakeTableWriter::new(writer_mysql.clone(), Arc::clone(&object_store))?;
    let orders_result = tw_mysql
        .write_table("public", "orders", std::slice::from_ref(&orders_batch))
        .await?;
    println!(
        "✓ wrote mysql_prod.public.orders — snapshot {}, file count {}, rows {}",
        orders_result.snapshot_id, orders_result.files_written, orders_result.records_written
    );

    // pg_prod.public.users with an added column — DDL, schema_version bumps.
    let users_v2_batch = build_users_v2_batch();
    let users_v2 = tw_pg
        .write_table("public", "users", std::slice::from_ref(&users_v2_batch))
        .await?;
    println!(
        "✓ wrote pg_prod.public.users WITH age column (DDL) — snapshot {}",
        users_v2.snapshot_id
    );

    // ── Step 4: dump catalog state ────────────────────────────────────────────
    println!();
    println!("== Catalog state ==");
    dump_query(
        &pool,
        "ducklake_catalog",
        "SELECT catalog_id, catalog_name FROM ducklake_catalog ORDER BY catalog_id",
    )
    .await?;
    dump_query(
        &pool,
        "ducklake_catalog_snapshot_map",
        "SELECT catalog_id, snapshot_id FROM ducklake_catalog_snapshot_map ORDER BY catalog_id, snapshot_id",
    )
    .await?;
    dump_query(
        &pool,
        "ducklake_catalog_schema_map",
        "SELECT catalog_id, schema_id FROM ducklake_catalog_schema_map ORDER BY catalog_id",
    )
    .await?;
    dump_query(
        &pool,
        "ducklake_snapshot",
        "SELECT snapshot_id, schema_version FROM ducklake_snapshot ORDER BY snapshot_id",
    )
    .await?;
    dump_query(
        &pool,
        "ducklake_schema",
        "SELECT schema_id, schema_name, path, begin_snapshot, end_snapshot FROM ducklake_schema ORDER BY schema_id",
    )
    .await?;
    dump_query(
        &pool,
        "ducklake_table",
        "SELECT table_id, schema_id, table_name, begin_snapshot, end_snapshot FROM ducklake_table ORDER BY table_id",
    )
    .await?;
    dump_query(
        &pool,
        "ducklake_schema_versions",
        "SELECT begin_snapshot, schema_version, table_id FROM ducklake_schema_versions ORDER BY begin_snapshot",
    )
    .await?;
    dump_query(
        &pool,
        "ducklake_data_file",
        "SELECT data_file_id, table_id, path, record_count, begin_snapshot, end_snapshot FROM ducklake_data_file ORDER BY data_file_id",
    )
    .await?;

    // ── Step 5: read back through the catalog layer ───────────────────────────
    println!();
    println!("== Reading via MulticatalogProvider + DuckLakeCatalog ==");
    println!();

    // Each catalog gets its own MulticatalogProvider and SessionContext, mimicking
    // how RuntimeDB would isolate per-tenant connections.
    read_via_multicatalog(&pool, "pg_prod", "users", "SELECT * FROM users ORDER BY id").await?;
    read_via_multicatalog(
        &pool,
        "mysql_prod",
        "orders",
        "SELECT * FROM orders ORDER BY order_id",
    )
    .await?;

    // Cross-check: pg_prod's catalog should NOT see "orders" (lives in mysql_prod).
    println!("\n  -- cross-catalog leakage check (pg_prod must NOT see 'orders') --");
    let cross = pg_prod_sees_orders(&pool).await?;
    if cross {
        println!("    LEAK! pg_prod can see mysql_prod's table");
    } else {
        println!("    ✓ pg_prod cannot see mysql_prod.orders — isolation works");
    }

    println!("\n✓ end-to-end demo complete");
    Ok(())
}

async fn read_via_multicatalog(
    pool: &sqlx::PgPool,
    catalog_name: &str,
    expected_table: &str,
    sql: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n  -- {} via MulticatalogProvider --", catalog_name);
    let provider = MulticatalogProvider::with_pool(pool.clone(), catalog_name).await?;
    let snapshot = provider.get_current_snapshot()?;
    println!(
        "    catalog_id={}, current snapshot={}",
        provider.catalog_id(),
        snapshot
    );

    let catalog = DuckLakeCatalog::with_snapshot(Arc::new(provider), snapshot)?;
    let runtime = Arc::new(RuntimeEnv::default());
    let config = SessionConfig::new().with_default_catalog_and_schema(catalog_name, "public");
    let ctx = SessionContext::new_with_config_rt(config, runtime);
    ctx.register_catalog(catalog_name, Arc::new(catalog));

    // List what this catalog can see — should be exactly the one table.
    if let Some(cat) = ctx.catalog(catalog_name) {
        for schema_name in cat.schema_names() {
            if schema_name == "information_schema" {
                continue;
            }
            let schema = cat.schema(&schema_name).unwrap();
            println!("    schema {} -> tables {:?}", schema_name, schema.table_names());
        }
    }

    println!("    query: {}", sql);
    let df = ctx.sql(sql).await?;
    df.show().await?;
    let _ = expected_table; // illustrative arg
    Ok(())
}

async fn pg_prod_sees_orders(pool: &sqlx::PgPool) -> Result<bool, Box<dyn std::error::Error>> {
    let provider = MulticatalogProvider::with_pool(pool.clone(), "pg_prod").await?;
    let sn = provider.get_current_snapshot()?;
    let catalog = DuckLakeCatalog::with_snapshot(Arc::new(provider), sn)?;
    let ctx = SessionContext::new();
    ctx.register_catalog("pg_prod", Arc::new(catalog));
    let cat = ctx.catalog("pg_prod").unwrap();
    let schema = match cat.schema("public") {
        Some(s) => s,
        None => return Ok(false),
    };
    Ok(schema.table_names().iter().any(|n| n == "orders"))
}

fn build_users_batch() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(StringArray::from(vec![
                Some("Alice"),
                Some("Bob"),
                Some("Carol"),
            ])),
        ],
    )
    .unwrap()
}

fn build_users_v2_batch() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true),
        Field::new("age", DataType::Int32, true),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(StringArray::from(vec![
                Some("Alice"),
                Some("Bob"),
                Some("Carol"),
            ])),
            Arc::new(Int32Array::from(vec![Some(30), Some(25), None])),
        ],
    )
    .unwrap()
}

fn build_orders_batch() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int64, false),
        Field::new("amount", DataType::Float64, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(vec![100, 101, 102])),
            Arc::new(Float64Array::from(vec![19.99, 4.50, 250.00])),
        ],
    )
    .unwrap()
}

async fn dump_query(
    pool: &sqlx::PgPool,
    label: &str,
    sql: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n  -- {} --", label);
    let rows = sqlx::query(sql).fetch_all(pool).await?;
    if rows.is_empty() {
        println!("    (no rows)");
        return Ok(());
    }
    // Print header from column names of the first row.
    let header: Vec<String> = rows[0]
        .columns()
        .iter()
        .map(|c| sqlx::Column::name(c).to_string())
        .collect();
    println!("    {}", header.join(" | "));
    println!("    {}", "-".repeat(header.iter().map(|s| s.len()).sum::<usize>() + header.len() * 3));
    for row in &rows {
        let cols: Vec<String> = (0..row.len())
            .map(|i| format_col(row, i))
            .collect();
        println!("    {}", cols.join(" | "));
    }
    Ok(())
}

fn format_col(row: &sqlx::postgres::PgRow, i: usize) -> String {
    // Try a few common types; fall back to "<binary>".
    if let Ok(v) = row.try_get::<Option<i64>, _>(i) {
        return v.map(|x| x.to_string()).unwrap_or("NULL".into());
    }
    if let Ok(v) = row.try_get::<Option<i32>, _>(i) {
        return v.map(|x| x.to_string()).unwrap_or("NULL".into());
    }
    if let Ok(v) = row.try_get::<Option<bool>, _>(i) {
        return v.map(|x| x.to_string()).unwrap_or("NULL".into());
    }
    if let Ok(v) = row.try_get::<Option<String>, _>(i) {
        return v.unwrap_or("NULL".into());
    }
    "<unprintable>".into()
}

#[allow(dead_code)]
async fn visible_files_for_catalog(
    pool: &sqlx::PgPool,
    catalog_id: i64,
    table_name: &str,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    // Current snapshot for the catalog
    let cur: i64 = sqlx::query(
        "SELECT COALESCE(MAX(snapshot_id), 0) FROM ducklake_catalog_snapshot_map WHERE catalog_id = $1",
    )
    .bind(catalog_id)
    .fetch_one(pool)
    .await?
    .try_get(0)?;

    let rows = sqlx::query(
        "SELECT f.path FROM ducklake_data_file f
         JOIN ducklake_table t ON t.table_id = f.table_id
         JOIN ducklake_schema s ON s.schema_id = t.schema_id
         JOIN ducklake_catalog_schema_map m ON m.schema_id = s.schema_id
         WHERE m.catalog_id = $1
           AND t.table_name = $2
           AND f.begin_snapshot <= $3
           AND (f.end_snapshot IS NULL OR f.end_snapshot > $3)
         ORDER BY f.path",
    )
    .bind(catalog_id)
    .bind(table_name)
    .bind(cur)
    .fetch_all(pool)
    .await?;

    Ok(rows.into_iter().map(|r| r.try_get(0).unwrap()).collect())
}
