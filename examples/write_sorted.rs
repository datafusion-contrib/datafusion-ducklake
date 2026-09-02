//! Interop helper: write a sorted + size-rolled DuckLake table with THIS crate,
//! so the official DuckDB DuckLake extension can read it and prune. Creates an
//! `events(id, val)` table sorted by `val`, then inserts `val` out of order so the
//! sort + rollover must produce several contiguous, non-overlapping files.
//!
//! Usage: cargo run --example write_sorted \
//!          --features write-sqlite,metadata-sqlite -- <dir> <rows> <target_bytes>

use std::sync::Arc;

use arrow::array::{Int32Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::datasource::MemTable;
use datafusion::execution::context::SessionConfig;
use datafusion::prelude::*;
use datafusion_ducklake::metadata_provider::MetadataProvider;
use datafusion_ducklake::{
    ColumnDef, DuckLakeCatalog, DuckLakeWriteOptions, MetadataWriter, SqliteMetadataProvider,
    SqliteMetadataWriter, WriteMode, execute_ducklake_sql,
};

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = std::path::PathBuf::from(&args[1]);
    let rows: i32 = args[2].parse().unwrap();
    let target_bytes: usize = args[3].parse().unwrap();

    let data = dir.join("data");
    std::fs::create_dir_all(&data).unwrap();
    let conn = format!("sqlite:{}?mode=rwc", dir.join("cat.db").display());

    // Create an empty events(id, val) table, then SET SORTED BY (val).
    let writer = SqliteMetadataWriter::new_with_init(&conn).await.unwrap();
    writer.set_data_path(data.to_str().unwrap()).unwrap();
    let cols = vec![
        ColumnDef::from_arrow("id", &DataType::Int32, false).unwrap(),
        ColumnDef::from_arrow("val", &DataType::Int32, true).unwrap(),
    ];
    let s = writer
        .begin_write_transaction("main", "events", &cols, WriteMode::Replace)
        .unwrap();
    writer
        .publish_snapshot(
            s.table_id,
            "main",
            "events",
            s.snapshot_id,
            WriteMode::Replace,
            s.base_snapshot_id,
            &cols,
            &s.column_ids,
        )
        .unwrap();

    let ddl_provider = SqliteMetadataProvider::new(&conn).await.unwrap();
    let ddl_writer = SqliteMetadataWriter::new_with_init(&conn).await.unwrap();
    let ddl_catalog =
        DuckLakeCatalog::with_writer(Arc::new(ddl_provider), Arc::new(ddl_writer)).unwrap();
    let sctx = SessionContext::new();
    execute_ducklake_sql(
        &sctx,
        &ddl_catalog,
        "ALTER TABLE main.events SET SORTED BY (val)",
    )
    .await
    .unwrap();

    // Insert `rows` with val a shuffled permutation of 0..rows, small batches +
    // small target so sort + rollover yields several tight-range files.
    let provider = SqliteMetadataProvider::new(&conn).await.unwrap();
    let iwriter = SqliteMetadataWriter::new_with_init(&conn).await.unwrap();
    let mut options = DuckLakeWriteOptions::default();
    options.target_file_size = Some(target_bytes);
    let catalog = DuckLakeCatalog::with_writer(Arc::new(provider), Arc::new(iwriter))
        .unwrap()
        .with_write_options(options);
    let ctx = SessionContext::new_with_config(SessionConfig::new().with_batch_size(200));
    ctx.register_catalog("lake", Arc::new(catalog));

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("val", DataType::Int32, true),
    ]));
    let ids: Vec<i32> = (0..rows).collect();
    let vals: Vec<i32> = (0..rows)
        .map(|i| ((i as i64 * 7919) % rows as i64) as i32)
        .collect();
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(ids)), Arc::new(Int32Array::from(vals))],
    )
    .unwrap();
    ctx.register_table(
        "src",
        Arc::new(MemTable::try_new(schema, vec![vec![batch]]).unwrap()),
    )
    .unwrap();
    ctx.sql("INSERT INTO lake.main.events SELECT id, val FROM src")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    let rprovider = SqliteMetadataProvider::new(&conn).await.unwrap();
    let snap = rprovider.get_current_snapshot().unwrap();
    let files = rprovider
        .get_table_file_metadata_page(s.table_id, snap, None, 4096)
        .unwrap();
    println!("WROTE {} files for {} rows", files.len(), rows);
    println!("catalog: {conn}");
    println!("data_path: {}", data.display());
}
