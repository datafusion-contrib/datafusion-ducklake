#![cfg(all(feature = "write-sqlite", feature = "metadata-duckdb"))]
//! Data files written through SQL `INSERT` choose each column's encoding the way
//! official DuckLake's writer does: a mostly-distinct column gets no dictionary
//! and no bloom filter, a repeating one gets both. These tests pin that layout,
//! that every value reads back exactly, and that DuckDB reads the files too.

use std::sync::Arc;

use arrow::array::{Array, AsArray};
use arrow::datatypes::{Float64Type, Int64Type};
use bytes::Bytes;
use datafusion::prelude::*;
use datafusion_ducklake::{
    ColumnDef, DuckLakeCatalog, MetadataWriter, SqliteMetadataProvider, SqliteMetadataWriter,
    WriteMode,
};
use parquet::basic::Encoding;
use parquet::file::reader::{FileReader, SerializedFileReader};
use tempfile::TempDir;

const ROWS: i64 = 50_000;

/// A table of `ROWS` rows written by one SQL `INSERT` per range in `inserts`:
/// `id` and `s` unique, `v` unique with NULLs, `lc` ten values with NULLs, `m`
/// seven values. Returns the catalog connection string and the data files.
async fn write_table(dir: &TempDir, inserts: &[(i64, i64)]) -> (String, Vec<std::path::PathBuf>) {
    let data = dir.path().join("data");
    std::fs::create_dir_all(&data).unwrap();
    let conn = format!("sqlite:{}?mode=rwc", dir.path().join("cat.db").display());

    let writer = SqliteMetadataWriter::new_with_init(&conn).await.unwrap();
    writer.set_data_path(data.to_str().unwrap()).unwrap();
    let columns = vec![
        ColumnDef::new("id", "int64", false).unwrap(),
        ColumnDef::new("v", "float64", true).unwrap(),
        ColumnDef::new("s", "varchar", false).unwrap(),
        ColumnDef::new("lc", "varchar", true).unwrap(),
        ColumnDef::new("m", "int64", false).unwrap(),
    ];
    let setup = writer
        .begin_write_transaction("main", "t", &columns, WriteMode::Replace)
        .unwrap();
    writer
        .publish_snapshot(
            setup.table_id,
            "main",
            "t",
            setup.snapshot_id,
            WriteMode::Replace,
            setup.base_snapshot_id,
            &columns,
            &setup.column_ids,
        )
        .unwrap();

    for (lo, hi) in inserts {
        // A fresh catalog per insert, so each one sees the previous commit.
        let ctx = writable_context(&conn).await;
        ctx.sql(&format!(
            "INSERT INTO lake.main.t SELECT \
                 value, \
                 CASE WHEN value % 11 = 0 THEN NULL ELSE CAST(value AS DOUBLE) * 0.25 END, \
                 concat('user_', CAST(value AS VARCHAR)), \
                 CASE WHEN value % 13 = 0 THEN NULL \
                      ELSE concat('cat_', CAST(value % 10 AS VARCHAR)) END, \
                 value % 7 \
             FROM range({lo}, {hi})"
        ))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    }

    let files = parquet_files(dir);
    assert!(!files.is_empty(), "the insert wrote a data file");
    (conn, files)
}

async fn writable_context(conn: &str) -> SessionContext {
    let provider = SqliteMetadataProvider::new(conn).await.unwrap();
    let writer = SqliteMetadataWriter::new_with_init(conn).await.unwrap();
    let catalog = DuckLakeCatalog::with_writer(Arc::new(provider), Arc::new(writer)).unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("lake", Arc::new(catalog));
    ctx
}

fn parquet_files(dir: &TempDir) -> Vec<std::path::PathBuf> {
    let mut files: Vec<_> = walk(&dir.path().join("data"))
        .into_iter()
        .filter(|p| p.extension().is_some_and(|e| e == "parquet"))
        .collect();
    files.sort();
    files
}

/// Every row group of every file in `files`: dictionary and bloom filter on the
/// table's repeating columns only.
fn assert_official_layout(files: &[std::path::PathBuf]) {
    for file in files {
        let reader = SerializedFileReader::new(Bytes::from(std::fs::read(file).unwrap())).unwrap();
        for row_group in reader.metadata().row_groups() {
            for column in row_group.columns() {
                let name = column.column_path().string();
                // Compaction output also embeds internal lineage columns; only the
                // table's own columns are pinned here.
                if !matches!(name.as_str(), "id" | "v" | "s" | "lc" | "m") {
                    continue;
                }
                let dictionary = column.dictionary_page_offset().is_some()
                    || column
                        .encodings()
                        .any(|encoding| encoding == Encoding::RLE_DICTIONARY);
                let bloom = column.bloom_filter_offset().is_some();
                let repeating = matches!(name.as_str(), "lc" | "m");
                assert_eq!(
                    dictionary, repeating,
                    "{name}: dictionary only on repeating columns"
                );
                assert_eq!(
                    bloom, repeating,
                    "{name}: bloom filter only on dictionary columns"
                );
            }
        }
    }
}

fn walk(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            out.extend(walk(&path));
        } else {
            out.push(path);
        }
    }
    out
}

#[tokio::test(flavor = "multi_thread")]
async fn each_column_gets_the_encoding_official_would_choose() {
    let dir = TempDir::new().unwrap();
    let (_conn, files) = write_table(&dir, &[(0, ROWS)]).await;
    assert_official_layout(&files);
}

/// Compaction writes its merged file through its own writer, which also picks the
/// encoding from the file's first batch.
#[tokio::test(flavor = "multi_thread")]
async fn a_merged_file_gets_the_same_encoding_choice() {
    let dir = TempDir::new().unwrap();
    let (conn, before) = write_table(&dir, &[(0, ROWS / 2), (ROWS / 2, ROWS)]).await;
    assert_eq!(before.len(), 2, "two inserts, two files");

    let ctx = writable_context(&conn).await;
    let provider = ctx
        .catalog("lake")
        .unwrap()
        .schema("main")
        .unwrap()
        .table("t")
        .await
        .unwrap()
        .unwrap();
    let table = (provider.as_ref() as &dyn std::any::Any)
        .downcast_ref::<datafusion_ducklake::DuckLakeTable>()
        .expect("provider is a DuckLakeTable");
    let result = table
        .merge_adjacent_files(&ctx.state(), datafusion_ducklake::MergeOptions::default())
        .await
        .unwrap();
    assert_eq!(result.files_created, 1, "the two files merge into one");

    let merged: Vec<_> = parquet_files(&dir)
        .into_iter()
        .filter(|file| !before.contains(file))
        .collect();
    assert_eq!(merged.len(), 1);
    assert_official_layout(&merged);

    let ctx = writable_context(&conn).await;
    let sums = ctx
        .sql("SELECT count(*), sum(id), count(v), count(lc) FROM lake.main.t")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let column = |i: usize| sums[0].column(i).as_primitive::<Int64Type>().value(0);
    assert_eq!(column(0), ROWS);
    assert_eq!(column(1), (0..ROWS).sum::<i64>());
    assert_eq!(column(2), (0..ROWS).filter(|i| i % 11 != 0).count() as i64);
    assert_eq!(column(3), (0..ROWS).filter(|i| i % 13 != 0).count() as i64);
}

#[tokio::test(flavor = "multi_thread")]
async fn every_value_reads_back_and_duckdb_reads_the_files() {
    let dir = TempDir::new().unwrap();
    let (conn, files) = write_table(&dir, &[(0, ROWS)]).await;

    let provider = SqliteMetadataProvider::new(&conn).await.unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("lake", Arc::new(DuckLakeCatalog::new(provider).unwrap()));
    let batches = ctx
        .sql("SELECT id, v, s, lc, m FROM lake.main.t ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    let mut expected_id = 0_i64;
    for batch in &batches {
        let id = batch.column(0).as_primitive::<Int64Type>();
        let v = batch.column(1).as_primitive::<Float64Type>();
        let m = batch.column(4).as_primitive::<Int64Type>();
        let s = arrow::compute::cast(batch.column(2), &arrow::datatypes::DataType::Utf8).unwrap();
        let lc = arrow::compute::cast(batch.column(3), &arrow::datatypes::DataType::Utf8).unwrap();
        let (s, lc) = (s.as_string::<i32>(), lc.as_string::<i32>());
        for row in 0..batch.num_rows() {
            let i = expected_id;
            assert_eq!(id.value(row), i);
            if i % 11 == 0 {
                assert!(v.is_null(row), "v[{i}] is NULL");
            } else {
                assert_eq!(v.value(row), i as f64 * 0.25, "v[{i}]");
            }
            assert_eq!(s.value(row), format!("user_{i}"), "s[{i}]");
            if i % 13 == 0 {
                assert!(lc.is_null(row), "lc[{i}] is NULL");
            } else {
                assert_eq!(lc.value(row), format!("cat_{}", i % 10), "lc[{i}]");
            }
            assert_eq!(m.value(row), i % 7, "m[{i}]");
            expected_id += 1;
        }
    }
    assert_eq!(expected_id, ROWS, "every row reads back");

    // An equality on a value no row group holds: the bloom filter may skip the
    // row groups, and the answer is still exact.
    let absent = ctx
        .sql("SELECT count(*) FROM lake.main.t WHERE lc = 'cat_55'")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(absent[0].column(0).as_primitive::<Int64Type>().value(0), 0);

    // DuckDB's own parquet reader agrees on the same files.
    let list = files
        .iter()
        .map(|f| format!("'{}'", f.display()))
        .collect::<Vec<_>>()
        .join(", ");
    let duck = duckdb::Connection::open_in_memory().unwrap();
    let (count, id_sum, v_count, v_sum, lc_distinct, s_max, m_sum): (
        i64,
        i64,
        i64,
        f64,
        i64,
        String,
        i64,
    ) = duck
        .query_row(
            &format!(
                "SELECT count(*), sum(id)::BIGINT, count(v), sum(v), count(DISTINCT lc), \
                        max(s), sum(m)::BIGINT FROM read_parquet([{list}])"
            ),
            [],
            |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                    row.get(5)?,
                    row.get(6)?,
                ))
            },
        )
        .unwrap();
    let ids = 0..ROWS;
    assert_eq!(count, ROWS);
    assert_eq!(id_sum, ids.clone().sum::<i64>());
    assert_eq!(v_count, ids.clone().filter(|i| i % 11 != 0).count() as i64);
    assert_eq!(
        v_sum,
        ids.clone()
            .filter(|i| i % 11 != 0)
            .map(|i| i as f64 * 0.25)
            .sum::<f64>()
    );
    assert_eq!(lc_distinct, 10);
    assert_eq!(s_max, "user_9999");
    assert_eq!(m_sum, ids.map(|i| i % 7).sum::<i64>());
}

/// A bloom filter may only rule a row group out, never a value it holds: an
/// equality on a present value must find every row, on every type that gets one.
#[tokio::test(flavor = "multi_thread")]
async fn a_bloom_filter_never_hides_a_present_value() {
    let dir = TempDir::new().unwrap();
    let data = dir.path().join("data");
    std::fs::create_dir_all(&data).unwrap();
    let conn = format!("sqlite:{}?mode=rwc", dir.path().join("cat.db").display());
    let writer = SqliteMetadataWriter::new_with_init(&conn).await.unwrap();
    writer.set_data_path(data.to_str().unwrap()).unwrap();
    let columns = vec![
        ColumnDef::new("k32", "int32", false).unwrap(),
        ColumnDef::new("k64", "int64", false).unwrap(),
        ColumnDef::new("ks", "varchar", false).unwrap(),
        ColumnDef::new("kd", "decimal(10,2)", false).unwrap(),
        ColumnDef::new("kdt", "date", false).unwrap(),
    ];
    let setup = writer
        .begin_write_transaction("main", "b", &columns, WriteMode::Replace)
        .unwrap();
    writer
        .publish_snapshot(
            setup.table_id,
            "main",
            "b",
            setup.snapshot_id,
            WriteMode::Replace,
            setup.base_snapshot_id,
            &columns,
            &setup.column_ids,
        )
        .unwrap();
    writable_context(&conn)
        .await
        .sql(&format!(
            "INSERT INTO lake.main.b SELECT \
                 CAST(value % 10 AS INT) AS k32, \
                 value % 10 AS k64, \
                 concat('k_', CAST(value % 10 AS VARCHAR)) AS ks, \
                 CAST(value % 10 AS DECIMAL(10,2)) / 4 AS kd, \
                 make_date(2024, 1, CAST(value % 10 + 1 AS INT)) AS kdt \
             FROM range(0, {ROWS})"
        ))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    // Every column is repeating, so every one carries a bloom filter.
    for file in parquet_files(&dir) {
        let reader = SerializedFileReader::new(Bytes::from(std::fs::read(&file).unwrap())).unwrap();
        for column in reader.metadata().row_group(0).columns() {
            assert!(
                column.bloom_filter_offset().is_some(),
                "{}: bloom filter written",
                column.column_path().string()
            );
        }
    }

    let provider = SqliteMetadataProvider::new(&conn).await.unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("lake", Arc::new(DuckLakeCatalog::new(provider).unwrap()));
    for (predicate, expected) in [
        ("k32 = 7", ROWS / 10),
        ("k64 = 7", ROWS / 10),
        ("ks = 'k_7'", ROWS / 10),
        ("kd = 1.75", ROWS / 10),
        ("kdt = DATE '2024-01-08'", ROWS / 10),
        ("k32 = 70", 0),
        ("ks = 'k_70'", 0),
    ] {
        let batches = ctx
            .sql(&format!(
                "SELECT count(*) FROM lake.main.b WHERE {predicate}"
            ))
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        assert_eq!(
            batches[0].column(0).as_primitive::<Int64Type>().value(0),
            expected,
            "{predicate}"
        );
    }
}

/// A file rolls over only once its first row group is complete, as DuckDB's does:
/// below one row group a write is one file whatever `target_file_size` says, and
/// above it files roll at row-group boundaries.
#[tokio::test(flavor = "multi_thread")]
async fn a_file_rolls_only_at_row_group_boundaries() {
    let schema = Arc::new(arrow::datatypes::Schema::new(vec![
        arrow::datatypes::Field::new("id", arrow::datatypes::DataType::Int64, false),
    ]));
    // 5,000 unique ids in 100-row batches.
    let batches: Vec<_> = (0..50_i64)
        .map(|b| {
            arrow::record_batch::RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(arrow::array::Int64Array::from_iter_values(
                    b * 100..(b + 1) * 100,
                ))],
            )
            .unwrap()
        })
        .collect();

    for (row_group_rows, expected_files) in [(None, 1), (Some(1_000), 5)] {
        let dir = TempDir::new().unwrap();
        let data = dir.path().join("data");
        std::fs::create_dir_all(&data).unwrap();
        let conn = format!("sqlite:{}?mode=rwc", dir.path().join("cat.db").display());
        let writer = SqliteMetadataWriter::new_with_init(&conn).await.unwrap();
        writer.set_data_path(data.to_str().unwrap()).unwrap();
        // Parquet V1 writes unique ids plain, so each 1,000-row row group is well
        // over the target and a file rolls right after it.
        let mut options = datafusion_ducklake::DuckLakeWriteOptions::default();
        options.parquet_version = Some(parquet::file::properties::WriterVersion::PARQUET_1_0);
        options.max_row_group_rows = row_group_rows;
        let table_writer = datafusion_ducklake::DuckLakeTableWriter::new(
            Arc::new(writer),
            Arc::new(object_store::local::LocalFileSystem::new()),
        )
        .unwrap()
        .with_target_file_size(4 * 1024)
        .with_options(&options);
        let result = table_writer
            .write_table("main", "r", &batches)
            .await
            .unwrap();
        assert_eq!(result.records_written, 5_000);
        assert_eq!(
            result.files_written, expected_files,
            "row groups of {row_group_rows:?} rows"
        );
    }
}
