//! Interop helper: open a DuckLake catalog (SQLite metadata) that some other
//! writer produced, print the live sort spec for a table, and run a range query
//! to show file pruning. Used to verify cross-engine compatibility with the
//! official DuckDB DuckLake extension.
//!
//! Usage: cargo run --example dump_sort --features write-sqlite,metadata-sqlite \
//!            -- <sqlite-conn-str> <schema> <table> <filter-sql>

use std::sync::Arc;

use datafusion::prelude::*;
use datafusion_ducklake::metadata_provider::MetadataProvider;
use datafusion_ducklake::{DuckLakeCatalog, SqliteMetadataProvider};

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let conn = &args[1];
    let schema = &args[2];
    let table = &args[3];
    let filter = args.get(4).cloned().unwrap_or_default();

    let provider = SqliteMetadataProvider::new(conn).await.unwrap();
    let snapshot = provider.get_current_snapshot().unwrap();
    let sch = provider
        .get_schema_by_name(schema, snapshot)
        .unwrap()
        .expect("schema");
    let tbl = provider
        .get_table_by_name(sch.schema_id, table, snapshot)
        .unwrap()
        .expect("table");

    match provider.get_sort_spec(tbl.table_id, snapshot).unwrap() {
        Some(spec) => {
            println!("SORT SPEC (sort_id={}):", spec.sort_id);
            for f in &spec.fields {
                println!(
                    "  [{}] {} {:?} {:?}  producible_column={:?}",
                    f.sort_key_index,
                    f.expression,
                    f.direction,
                    f.null_order,
                    f.column_candidate()
                );
            }
        },
        None => println!("SORT SPEC: none"),
    }

    let catalog = DuckLakeCatalog::new(provider).unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("lake", Arc::new(catalog));

    let total = ctx
        .sql(&format!("SELECT count(*) FROM lake.{schema}.{table}"))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    println!("TOTAL ROWS: {:?}", total[0].column(0));

    if !filter.is_empty() {
        let q = format!("SELECT count(*) FROM lake.{schema}.{table} WHERE {filter}");
        let cnt = ctx.sql(&q).await.unwrap().collect().await.unwrap();
        println!("FILTERED ({filter}) ROWS: {:?}", cnt[0].column(0));

        let plan = ctx
            .sql(&format!(
                "SELECT * FROM lake.{schema}.{table} WHERE {filter}"
            ))
            .await
            .unwrap()
            .create_physical_plan()
            .await
            .unwrap();
        let display = datafusion::physical_plan::displayable(plan.as_ref())
            .indent(true)
            .to_string();
        let files = display.matches(".parquet").count();
        println!("FILES SCANNED for [{filter}]: {files}");
    }
}
