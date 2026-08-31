//! Integration tests for explicit DuckLake compaction on the SQLite backend:
//! `DuckLakeTable::merge_adjacent_files` and `rewrite_data_files`.
//!
//! Compaction rewrites data files, so these assert the load-bearing invariants
//! end-to-end: fewer live files with identical query results, exactly one new
//! snapshot, source files retired + scheduled for deletion, rowid lineage
//! preserved, time travel to a pre-compaction snapshot still returning the
//! original rows, and the same-schema-version merge boundary.

#![cfg(all(feature = "write-sqlite", feature = "metadata-sqlite"))]

use std::sync::Arc;

use arrow::array::{Array, Int32Array, Int64Array, ListArray};
use arrow::datatypes::{DataType, Field, Int32Type, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::execution::memory_pool::FairSpillPool;
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::prelude::*;
use object_store::local::LocalFileSystem;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use sqlx::sqlite::SqlitePool;
use sqlx::{AssertSqlSafe, Row};
use tempfile::TempDir;

use datafusion_ducklake::maintenance::{CleanupCriteria, cleanup_old_files_sqlite};
use datafusion_ducklake::{
    CompactionResult, DuckLakeCatalog, DuckLakeTable, DuckLakeTableWriter, MergeOptions,
    MetadataProvider, MetadataWriter, NullOrder, RewriteOptions, SortDirection, SortField,
    SqliteMetadataProvider, SqliteMetadataWriter,
};

fn two_col_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("val", DataType::Int32, false),
    ]))
}

fn object_store() -> Arc<dyn object_store::ObjectStore> {
    Arc::new(LocalFileSystem::new())
}

fn db_url(temp: &TempDir) -> String {
    format!("sqlite:{}?mode=rwc", temp.path().join("test.db").display())
}

fn ro_url(temp: &TempDir) -> String {
    format!("sqlite:{}", temp.path().join("test.db").display())
}

async fn make_writer(temp: &TempDir) -> SqliteMetadataWriter {
    let data_path = temp.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();
    let writer = SqliteMetadataWriter::new_with_init(&db_url(temp))
        .await
        .unwrap();
    writer.set_data_path(data_path.to_str().unwrap()).unwrap();
    writer
}

fn batch(schema: Arc<Schema>, cols: Vec<Arc<dyn Array>>) -> RecordBatch {
    RecordBatch::try_new(schema, cols).unwrap()
}

/// Seed a fresh `main.t(id, val)` as one data file (Replace on a new table).
async fn seed(temp: &TempDir, ids: Vec<i32>, vals: Vec<i32>) {
    let writer = Arc::new(make_writer(temp).await);
    let b = batch(
        two_col_schema(),
        vec![Arc::new(Int32Array::from(ids)), Arc::new(Int32Array::from(vals))],
    );
    DuckLakeTableWriter::new(writer, object_store())
        .unwrap()
        .write_table("main", "t", &[b])
        .await
        .unwrap();
}

/// Append one more `(id, val)` data file to `main.t`.
async fn append(temp: &TempDir, ids: Vec<i32>, vals: Vec<i32>) {
    let writer = Arc::new(SqliteMetadataWriter::new(&db_url(temp)).await.unwrap());
    let b = batch(
        two_col_schema(),
        vec![Arc::new(Int32Array::from(ids)), Arc::new(Int32Array::from(vals))],
    );
    DuckLakeTableWriter::new(writer, object_store())
        .unwrap()
        .append_table("main", "t", &[b])
        .await
        .unwrap();
}

async fn pool(temp: &TempDir) -> SqlitePool {
    SqlitePool::connect(&ro_url(temp)).await.unwrap()
}

async fn scalar_i64(p: &SqlitePool, sql: &str) -> i64 {
    sqlx::query(AssertSqlSafe(sql))
        .fetch_one(p)
        .await
        .unwrap()
        .try_get::<i64, _>(0)
        .unwrap()
}

async fn opt_i64(p: &SqlitePool, sql: &str) -> Option<i64> {
    sqlx::query(AssertSqlSafe(sql))
        .fetch_one(p)
        .await
        .unwrap()
        .try_get::<Option<i64>, _>(0)
        .unwrap()
}

/// Current live `(id, val)` rows of `main.t`, ascending, through the full read
/// path (which applies any live delete file / embedded-rowid file).
async fn read_rows(temp: &TempDir) -> Vec<(i32, i32)> {
    let provider = SqliteMetadataProvider::new(&ro_url(temp)).await.unwrap();
    rows_via(DuckLakeCatalog::new(provider).unwrap()).await
}

/// `(id, val)` rows of `main.t` as of `snapshot` (time travel).
async fn read_rows_at(temp: &TempDir, snapshot: i64) -> Vec<(i32, i32)> {
    let provider = Arc::new(SqliteMetadataProvider::new(&ro_url(temp)).await.unwrap());
    rows_via(DuckLakeCatalog::with_snapshot(provider, snapshot).unwrap()).await
}

async fn rows_via(catalog: DuckLakeCatalog) -> Vec<(i32, i32)> {
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));
    let batches = ctx
        .sql("SELECT id, val FROM ducklake.main.t ORDER BY id, val")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let mut out = Vec::new();
    for b in &batches {
        let ids = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        let vals = b.column(1).as_any().downcast_ref::<Int32Array>().unwrap();
        for i in 0..b.num_rows() {
            out.push((ids.value(i), vals.value(i)));
        }
    }
    out
}

/// Current live `(id, rowid)` of `main.t`, ascending by id, via a row-lineage
/// catalog — the rowid is each row's DuckLake row-lineage id.
async fn read_id_rowid(temp: &TempDir) -> Vec<(i32, i64)> {
    let provider = SqliteMetadataProvider::new(&ro_url(temp)).await.unwrap();
    let catalog = DuckLakeCatalog::new(provider)
        .unwrap()
        .with_row_lineage(true);
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));
    let batches = ctx
        .sql("SELECT id, rowid FROM ducklake.main.t ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let mut out = Vec::new();
    for b in &batches {
        let ids = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        let rids = b.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
        for i in 0..b.num_rows() {
            out.push((ids.value(i), rids.value(i)));
        }
    }
    out
}

fn file_values(temp: &TempDir, path: &str) -> Vec<i32> {
    let file =
        std::fs::File::open(temp.path().join("data").join("main").join("t").join(path)).unwrap();
    ParquetRecordBatchReaderBuilder::try_new(file)
        .unwrap()
        .build()
        .unwrap()
        .map(|batch| {
            let batch = batch.unwrap();
            batch
                .column_by_name("val")
                .unwrap()
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .values()
                .to_vec()
        })
        .collect::<Vec<_>>()
        .concat()
}

/// One live data file's catalog row, as `merge_reads_a_whole_bin_in_one_pass`
/// reads it back.
#[derive(Debug)]
struct MergedFile {
    path: String,
    partition_id: Option<i64>,
    partition_value: Option<String>,
    begin_snapshot: Option<i64>,
    partial_max: Option<i64>,
}

/// The embedded lineage columns of a compaction output — `(rowid, origin
/// snapshot)` per row, in the file's PHYSICAL order. Read from the parquet
/// itself, so it shows the layout on disk rather than what the read path
/// reconstructs. The snapshot column is absent unless the file is partial.
fn file_lineage(temp: &TempDir, path: &str) -> Vec<(i64, Option<i64>)> {
    let file =
        std::fs::File::open(temp.path().join("data").join("main").join("t").join(path)).unwrap();
    let mut out = Vec::new();
    for batch in ParquetRecordBatchReaderBuilder::try_new(file)
        .unwrap()
        .build()
        .unwrap()
    {
        let batch = batch.unwrap();
        let rowids = batch
            .column_by_name("_ducklake_internal_row_id")
            .expect("a compaction output embeds its rowids")
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .clone();
        let snapshots = batch
            .column_by_name("_ducklake_internal_snapshot_id")
            .map(|column| {
                column
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .clone()
            });
        for i in 0..batch.num_rows() {
            out.push((
                rowids.value(i),
                snapshots.as_ref().map(|column| column.value(i)),
            ));
        }
    }
    out
}

/// Downcast the writable `main.t` provider to a `DuckLakeTable` and run `op` on
/// it (the compaction ops are `DuckLakeTable` methods). A fresh writable catalog
/// is opened so the table binds to the latest snapshot.
async fn with_writable_table<F, Fut>(temp: &TempDir, op: F) -> CompactionResult
where
    F: FnOnce(DuckLakeTable, datafusion::execution::SessionState) -> Fut,
    Fut: std::future::Future<Output = CompactionResult>,
{
    with_writable_table_context(temp, SessionContext::new(), op).await
}

async fn with_writable_table_context<F, Fut>(
    temp: &TempDir,
    ctx: SessionContext,
    op: F,
) -> CompactionResult
where
    F: FnOnce(DuckLakeTable, datafusion::execution::SessionState) -> Fut,
    Fut: std::future::Future<Output = CompactionResult>,
{
    let writer = SqliteMetadataWriter::new(&db_url(temp)).await.unwrap();
    let provider = SqliteMetadataProvider::new(&db_url(temp)).await.unwrap();
    let catalog = DuckLakeCatalog::with_writer(Arc::new(provider), Arc::new(writer)).unwrap();
    ctx.register_catalog("ducklake", Arc::new(catalog));
    let provider = ctx
        .catalog("ducklake")
        .unwrap()
        .schema("main")
        .unwrap()
        .table("t")
        .await
        .unwrap()
        .unwrap();
    let table = (provider.as_ref() as &dyn std::any::Any)
        .downcast_ref::<DuckLakeTable>()
        .expect("provider is a DuckLakeTable")
        .clone();
    op(table, ctx.state()).await
}

async fn run_merge(temp: &TempDir, opts: MergeOptions) -> CompactionResult {
    with_writable_table(temp, |table, state| async move {
        table.merge_adjacent_files(&state, opts).await.unwrap()
    })
    .await
}

#[cfg(feature = "metadata-duckdb")]
fn create_official_mapped_compaction_fixture(temp: &TempDir) -> anyhow::Result<()> {
    let data_path = temp.path().join("data");
    let first_partition = data_path.join("part=7");
    let second_partition = data_path.join("part=8");
    std::fs::create_dir_all(&first_partition)?;
    std::fs::create_dir_all(&second_partition)?;

    let conn = duckdb::Connection::open_in_memory()?;
    crate::common::ensure_ducklake_installed();
    conn.execute("INSTALL sqlite", [])?;
    conn.execute("LOAD sqlite", [])?;
    conn.execute("LOAD ducklake", [])?;
    conn.execute(
        &format!(
            "ATTACH 'ducklake:sqlite:{}' AS lake \
             (DATA_PATH '{}', DATA_INLINING_ROW_LIMIT 0)",
            temp.path().join("test.db").display(),
            data_path.display()
        ),
        [],
    )?;
    conn.execute("CREATE TABLE lake.t(id INTEGER, part INTEGER)", [])?;
    conn.execute(
        &format!(
            "COPY (SELECT 1 AS id) TO '{}' (FORMAT PARQUET)",
            first_partition.join("first.parquet").display()
        ),
        [],
    )?;
    conn.execute(
        &format!(
            "COPY (SELECT 2 AS id) TO '{}' (FORMAT PARQUET)",
            second_partition.join("second.parquet").display()
        ),
        [],
    )?;
    conn.execute(
        &format!(
            "CALL ducklake_add_data_files(\
                 'lake', 't', '{}/**/*.parquet', hive_partitioning => true\
             )",
            data_path.display()
        ),
        [],
    )?;
    conn.execute("DETACH lake", [])?;
    Ok(())
}

#[cfg(feature = "metadata-duckdb")]
async fn read_mapped_rows(temp: &TempDir) -> anyhow::Result<Vec<(i32, i32)>> {
    let provider = SqliteMetadataProvider::new(&ro_url(temp)).await?;
    let catalog = DuckLakeCatalog::new(provider)?;
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));
    let batches = ctx
        .sql("SELECT id, part FROM ducklake.main.t ORDER BY id")
        .await?
        .collect()
        .await?;
    let mut rows = Vec::new();
    for batch in batches {
        let ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let parts = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        rows.extend((0..batch.num_rows()).map(|row| (ids.value(row), parts.value(row))));
    }
    Ok(rows)
}

#[cfg(feature = "metadata-duckdb")]
async fn migrate_pinned_duckdb_fixture(temp: &TempDir) -> anyhow::Result<()> {
    let pool = pool(temp).await;
    SqliteMetadataWriter::new_with_init(&db_url(temp)).await?;
    let schema_version_columns = sqlx::query("PRAGMA table_info(ducklake_schema_versions)")
        .fetch_all(&pool)
        .await?;
    if !schema_version_columns
        .iter()
        .any(|row| row.get::<String, _>(1) == "table_id")
    {
        // The pinned DuckDB 1.4.1 test extension predates the per-table column
        // that released DuckDB 1.5.5 writes; keep the fixture shape equivalent
        sqlx::query("ALTER TABLE ducklake_schema_versions ADD COLUMN table_id INTEGER")
            .execute(&pool)
            .await?;
        sqlx::query(
            "UPDATE ducklake_schema_versions
             SET table_id = (SELECT table_id FROM ducklake_table WHERE table_name = 't')",
        )
        .execute(&pool)
        .await?;
    }
    let data_file_columns = sqlx::query("PRAGMA table_info(ducklake_data_file)")
        .fetch_all(&pool)
        .await?;
    let data_file_id_type = data_file_columns
        .iter()
        .find(|row| row.get::<String, _>(1) == "data_file_id")
        .map(|row| row.get::<String, _>(2))
        .unwrap();
    if data_file_id_type != "INTEGER" {
        // SQLite auto-allocates only an exact INTEGER PRIMARY KEY; normalize the
        // official BIGINT declaration to the crate writer's documented precondition
        sqlx::query("ALTER TABLE ducklake_data_file RENAME TO ducklake_data_file__official")
            .execute(&pool)
            .await?;
        sqlx::query(
            "CREATE TABLE ducklake_data_file (
                data_file_id INTEGER PRIMARY KEY,
                table_id INTEGER NOT NULL,
                path VARCHAR NOT NULL,
                path_is_relative BOOLEAN NOT NULL DEFAULT 1,
                file_size_bytes INTEGER NOT NULL,
                footer_size INTEGER,
                encryption_key VARCHAR,
                record_count INTEGER,
                row_id_start INTEGER,
                mapping_id INTEGER,
                begin_snapshot INTEGER NOT NULL,
                end_snapshot INTEGER,
                partial_max INTEGER,
                partition_id INTEGER
            )",
        )
        .execute(&pool)
        .await?;
        sqlx::query(
            "INSERT INTO ducklake_data_file (
                data_file_id, table_id, path, path_is_relative, file_size_bytes,
                footer_size, encryption_key, record_count, row_id_start, mapping_id,
                begin_snapshot, end_snapshot, partial_max, partition_id
             )
             SELECT data_file_id, table_id, path, path_is_relative, file_size_bytes,
                    footer_size, encryption_key, record_count, row_id_start, mapping_id,
                    begin_snapshot, end_snapshot, partial_max, partition_id
             FROM ducklake_data_file__official",
        )
        .execute(&pool)
        .await?;
        sqlx::query("DROP TABLE ducklake_data_file__official")
            .execute(&pool)
            .await?;
    }
    Ok(())
}

#[cfg(feature = "metadata-duckdb")]
#[tokio::test(flavor = "multi_thread")]
async fn mapped_hive_values_survive_merge_adjacent_files() -> anyhow::Result<()> {
    let temp = TempDir::new()?;
    create_official_mapped_compaction_fixture(&temp)?;
    migrate_pinned_duckdb_fixture(&temp).await?;
    assert_eq!(read_mapped_rows(&temp).await?, vec![(1, 7), (2, 8)]);

    let result = run_merge(&temp, MergeOptions::default()).await;

    assert_eq!(
        result,
        CompactionResult {
            files_processed: 2,
            files_created: 1,
            rows_written: 2,
        }
    );
    assert_eq!(read_mapped_rows(&temp).await?, vec![(1, 7), (2, 8)]);
    Ok(())
}

/// Run a merge with explicit table write options, as a catalog configured for a
/// codec would produce.
async fn run_merge_with_write_options(
    temp: &TempDir,
    write_options: datafusion_ducklake::DuckLakeWriteOptions,
    opts: MergeOptions,
) -> CompactionResult {
    with_writable_table(temp, |table, state| async move {
        table
            .with_write_options(write_options)
            .merge_adjacent_files(&state, opts)
            .await
            .unwrap()
    })
    .await
}

/// The same for a rewrite, so both writer sites are covered.
async fn run_rewrite_with_write_options(
    temp: &TempDir,
    write_options: datafusion_ducklake::DuckLakeWriteOptions,
    opts: RewriteOptions,
) -> CompactionResult {
    with_writable_table(temp, |table, state| async move {
        table
            .with_write_options(write_options)
            .rewrite_data_files(&state, opts)
            .await
            .unwrap()
    })
    .await
}

/// Every column chunk's codec, and the row group count, of the single live file.
fn live_file_parquet_facts(
    temp: &TempDir,
    pool_path: &str,
) -> (Vec<parquet::basic::Compression>, usize) {
    let file = std::fs::File::open(
        temp.path()
            .join("data")
            .join("main")
            .join("t")
            .join(pool_path),
    )
    .unwrap();
    let meta = ParquetRecordBatchReaderBuilder::try_new(file)
        .unwrap()
        .metadata()
        .clone();
    let codecs = meta
        .row_groups()
        .iter()
        .flat_map(|rg| rg.columns().iter().map(|c| c.compression()))
        .collect();
    (codecs, meta.num_row_groups())
}

async fn run_rewrite(temp: &TempDir, opts: RewriteOptions) -> CompactionResult {
    with_writable_table(temp, |table, state| async move {
        table.rewrite_data_files(&state, opts).await.unwrap()
    })
    .await
}

/// A sort spec whose fields are ALL from a foreign dialect must compact unsorted, not
/// fail.
///
/// `producible_columns` filters out any field whose dialect is not `duckdb`, so such a
/// spec leaves no usable keys. Official DuckLake skips those fields and proceeds with
/// whatever remains — nothing, here — so compaction completes and simply writes
/// unsorted. Erroring instead would fail on a catalog official compacts fine.
#[tokio::test(flavor = "multi_thread")]
async fn compaction_ignores_a_foreign_dialect_sort_spec() {
    use datafusion_ducklake::sort::{NullOrder, SortDirection, SortField};

    let temp = TempDir::new().unwrap();
    seed(&temp, vec![1, 2], vec![10, 20]).await;
    append(&temp, vec![3, 4], vec![30, 40]).await;

    let p = pool(&temp).await;
    let table_id = scalar_i64(
        &p,
        "SELECT table_id FROM ducklake_table WHERE table_name = 't'",
    )
    .await;

    // A sort field in another engine's dialect: readable, but not executable here.
    let foreign = SortField {
        sort_key_index: 0,
        expression: "val".to_string(),
        dialect: "spark".to_string(),
        direction: SortDirection::Asc,
        null_order: NullOrder::NullsLast,
    };
    SqliteMetadataWriter::new(&db_url(&temp))
        .await
        .unwrap()
        .set_sort_spec(table_id, &[foreign])
        .unwrap();

    // Sanity: a duckdb-dialect spec on the same table is executable, so the fixture is
    // exercising the dialect filter rather than some unrelated rejection.
    let _ = SortField::column(0, "val", SortDirection::Asc, NullOrder::NullsLast);

    let result = run_merge(
        &temp,
        MergeOptions {
            target_file_size: 1 << 30,
            max_merged_files: 1024,
            min_file_size: 0,
        },
    )
    .await;
    assert!(
        result.did_work(),
        "compaction must complete despite an unexecutable sort spec"
    );
    assert_eq!(
        read_rows(&temp).await,
        vec![(1, 10), (2, 20), (3, 30), (4, 40)]
    );
}

/// Compaction of a PARTITIONED table must merge only within a partition and carry
/// each output's partition assignment over from its sources — official DuckLake
/// groups merge candidates by (schema_version, partition_id, partition_values).
/// Merging across partitions would produce a file belonging to no single partition:
/// unprunable, and unrepresentable in `ducklake_file_partition_value`.
#[tokio::test(flavor = "multi_thread")]
async fn merge_only_within_a_partition_and_preserves_assignment() {
    use datafusion_ducklake::partition::PartitionTransform;
    use datafusion_ducklake::{ColumnDef, MetadataWriter, WriteMode};

    let temp = TempDir::new().unwrap();
    // Create `main.t(id, val)` and partition it by `val` before writing any data.
    let writer = Arc::new(make_writer(&temp).await);
    let cols = vec![
        ColumnDef::from_arrow("id", &DataType::Int32, false).unwrap(),
        ColumnDef::from_arrow("val", &DataType::Int32, false).unwrap(),
    ];
    let table_id = {
        let s = writer
            .begin_write_transaction("main", "t", &cols, WriteMode::Replace)
            .unwrap();
        writer
            .publish_snapshot(
                s.table_id,
                "main",
                "t",
                s.snapshot_id,
                WriteMode::Replace,
                s.base_snapshot_id,
                &cols,
                &s.column_ids,
            )
            .unwrap();
        writer
            .set_partition_spec(
                s.table_id,
                &[("val".to_string(), PartitionTransform::Identity)],
            )
            .unwrap();
        s.table_id
    };

    // Four small appends: two rows in partition val=1, two in val=2. Each append is
    // its own snapshot, and each writes one file per partition it touches.
    for id in [1, 2] {
        append(&temp, vec![id, id + 10], vec![1, 2]).await;
    }

    let p = pool(&temp).await;
    let live_before = scalar_i64(
        &p,
        "SELECT COUNT(*) FROM ducklake_data_file WHERE end_snapshot IS NULL",
    )
    .await;
    assert_eq!(live_before, 4, "two appends x two partitions = four files");

    // Merge with a target large enough to bin everything that is legal to bin.
    let result = run_merge(
        &temp,
        MergeOptions {
            target_file_size: 1 << 30,
            max_merged_files: 1024,
            min_file_size: 0,
        },
    )
    .await;
    assert!(result.did_work(), "the small files must be compacted");

    // Two outputs (one per partition), never one merged across both.
    let live_after: Vec<(Option<i64>, Option<String>)> = sqlx::query(
        "SELECT df.partition_id, fpv.partition_value
         FROM ducklake_data_file AS df
         LEFT JOIN ducklake_file_partition_value AS fpv
           ON fpv.data_file_id = df.data_file_id
         WHERE df.table_id = ? AND df.end_snapshot IS NULL
         ORDER BY fpv.partition_value",
    )
    .bind(table_id)
    .fetch_all(&p)
    .await
    .unwrap()
    .into_iter()
    .map(|r| {
        (
            r.try_get::<Option<i64>, _>(0).unwrap(),
            r.try_get::<Option<String>, _>(1).unwrap(),
        )
    })
    .collect();

    assert_eq!(
        live_after.len(),
        2,
        "one merged file per partition, not one across partitions: {live_after:?}"
    );
    let values: Vec<Option<String>> = live_after.iter().map(|(_, v)| v.clone()).collect();
    assert_eq!(
        values,
        vec![Some("1".to_string()), Some("2".to_string())],
        "each merged file keeps its partition value"
    );
    for (partition_id, _) in &live_after {
        assert!(
            partition_id.is_some(),
            "a merged file of a partitioned table must keep its partition_id"
        );
    }

    // And the rows survive intact, still prunable by the partition column.
    assert_eq!(
        read_rows(&temp).await,
        vec![(1, 1), (2, 1), (11, 2), (12, 2)]
    );
}

/// A merge reads the WHOLE bin with one plan — every source at once, each row's
/// lineage carried as a column of that scan, the shape official DuckLake
/// compaction uses. This pins what that has to produce for a bin far wider than
/// one file: the same rows, the same rowids, the same partition assignment and
/// the same per-row origin snapshots the old file-at-a-time reads produced.
///
/// Not the physical row order: the sources are read concurrently and handed on
/// as they arrive, which is what official does too. Nothing reads that order --
/// every row carries its own rowid and origin snapshot -- so the assertions
/// below compare lineage as a set.
#[tokio::test(flavor = "multi_thread")]
async fn merge_reads_a_whole_bin_in_one_pass() {
    use datafusion_ducklake::partition::PartitionTransform;
    use datafusion_ducklake::{ColumnDef, WriteMode};

    // Wide enough that a serialized read would be doing something visibly
    // different from one plan over the set, and deep enough in snapshots that
    // every source has its own origin.
    const APPENDS: i32 = 12;

    let temp = TempDir::new().unwrap();
    let writer = Arc::new(make_writer(&temp).await);
    let cols = vec![
        ColumnDef::from_arrow("id", &DataType::Int32, false).unwrap(),
        ColumnDef::from_arrow("val", &DataType::Int32, false).unwrap(),
    ];
    let table_id = {
        let s = writer
            .begin_write_transaction("main", "t", &cols, WriteMode::Replace)
            .unwrap();
        writer
            .publish_snapshot(
                s.table_id,
                "main",
                "t",
                s.snapshot_id,
                WriteMode::Replace,
                s.base_snapshot_id,
                &cols,
                &s.column_ids,
            )
            .unwrap();
        writer
            .set_partition_spec(
                s.table_id,
                &[("val".to_string(), PartitionTransform::Identity)],
            )
            .unwrap();
        s.table_id
    };

    // Each append is its own snapshot and touches both partitions, so the merge
    // sees APPENDS x 2 sources spread over APPENDS origin snapshots, binned into
    // one output per partition.
    append(&temp, vec![1, 101], vec![1, 2]).await;
    let p = pool(&temp).await;
    let first_snapshot = scalar_i64(&p, "SELECT MAX(snapshot_id) FROM ducklake_snapshot").await;
    for id in 2..=APPENDS {
        append(&temp, vec![id, id + 100], vec![1, 2]).await;
    }
    let last_snapshot = scalar_i64(&p, "SELECT MAX(snapshot_id) FROM ducklake_snapshot").await;
    assert_eq!(
        scalar_i64(
            &p,
            "SELECT COUNT(*) FROM ducklake_data_file WHERE end_snapshot IS NULL"
        )
        .await,
        i64::from(APPENDS) * 2,
    );

    let rows_before = read_rows(&temp).await;
    let rowids_before = read_id_rowid(&temp).await;

    let result = run_merge(
        &temp,
        MergeOptions {
            target_file_size: 1 << 30,
            max_merged_files: 1024,
            min_file_size: 0,
        },
    )
    .await;

    assert_eq!(
        result,
        CompactionResult {
            files_processed: usize::try_from(APPENDS).unwrap() * 2,
            files_created: 2,
            rows_written: i64::from(APPENDS) * 2,
        },
    );
    assert_eq!(read_rows(&temp).await, rows_before, "same rows");
    assert_eq!(read_id_rowid(&temp).await, rowids_before, "same rowids");

    // One merged file per partition, each keeping its sources' assignment, and
    // each recording the bin's origin span: begin at the MIN origin so time
    // travel back to it still sees the file, partial_max at the MAX.
    let live: Vec<MergedFile> = sqlx::query(
        "SELECT df.path, df.partition_id, fpv.partition_value, df.begin_snapshot, df.partial_max
         FROM ducklake_data_file AS df
         LEFT JOIN ducklake_file_partition_value AS fpv
           ON fpv.data_file_id = df.data_file_id
         WHERE df.table_id = ? AND df.end_snapshot IS NULL
         ORDER BY fpv.partition_value",
    )
    .bind(table_id)
    .fetch_all(&p)
    .await
    .unwrap()
    .into_iter()
    .map(|r| MergedFile {
        path: r.try_get(0).unwrap(),
        partition_id: r.try_get(1).unwrap(),
        partition_value: r.try_get(2).unwrap(),
        begin_snapshot: r.try_get(3).unwrap(),
        partial_max: r.try_get(4).unwrap(),
    })
    .collect();

    assert_eq!(live.len(), 2, "one merged file per partition: {live:?}");
    for (index, file) in live.iter().enumerate() {
        let expected_partition_value = if index == 0 {
            "1"
        } else {
            "2"
        };
        assert!(
            file.partition_id.is_some(),
            "a merged file of a partitioned table keeps its partition_id"
        );
        assert_eq!(
            file.partition_value.as_deref(),
            Some(expected_partition_value),
            "each merged file keeps its sources' partition value"
        );
        assert_eq!(
            file.begin_snapshot,
            Some(first_snapshot),
            "begin_snapshot is the MIN origin of the bin"
        );
        assert_eq!(
            file.partial_max,
            Some(last_snapshot),
            "partial_max is the MAX origin of the bin"
        );

        // Every row keeps its own rowid and the origin snapshot of the file it
        // came from. Read straight from the parquet, so this is the bytes on
        // disk, not what the read path reconstructs.
        //
        // Compared as a set: the bin's sources are read concurrently and handed
        // on as they arrive, so physical order is not fixed -- and nothing reads
        // it, because each row carries its own lineage rather than deriving it
        // from position. Sorting by rowid is what makes the assertion about the
        // pairing, which is the part that must hold.
        let mut lineage = file_lineage(&temp, &file.path);
        lineage.sort_unstable();
        let mut expected: Vec<(i64, Option<i64>)> = (0..APPENDS)
            .map(|append_index| {
                (
                    i64::from(append_index) * 2 + index as i64,
                    Some(first_snapshot + i64::from(append_index)),
                )
            })
            .collect();
        expected.sort_unstable();
        assert_eq!(
            lineage, expected,
            "merged rows keep their rowid and origin snapshot"
        );
    }

    // Time travel to the first append still sees exactly its rows, served by the
    // merged partial file.
    assert_eq!(
        read_rows_at(&temp, first_snapshot).await,
        vec![(1, 1), (101, 2)],
    );
}

/// A delete-driven rewrite of a PARTITIONED table must carry the source file's
/// partition assignment onto the rewritten output. The output holds a subset of one
/// file's rows, so it belongs to exactly that file's partition — official takes the
/// partition from `source_files[0]` on this path too. Without this the rewrite would
/// silently strip the assignment and the rows would stop being prunable.
#[tokio::test(flavor = "multi_thread")]
async fn rewrite_preserves_partition_assignment() {
    use datafusion_ducklake::partition::PartitionTransform;
    use datafusion_ducklake::{ColumnDef, MetadataWriter, WriteMode};

    let temp = TempDir::new().unwrap();
    let writer = Arc::new(make_writer(&temp).await);
    let cols = vec![
        ColumnDef::from_arrow("id", &DataType::Int32, false).unwrap(),
        ColumnDef::from_arrow("val", &DataType::Int32, false).unwrap(),
    ];
    let table_id = {
        let s = writer
            .begin_write_transaction("main", "t", &cols, WriteMode::Replace)
            .unwrap();
        writer
            .publish_snapshot(
                s.table_id,
                "main",
                "t",
                s.snapshot_id,
                WriteMode::Replace,
                s.base_snapshot_id,
                &cols,
                &s.column_ids,
            )
            .unwrap();
        writer
            .set_partition_spec(
                s.table_id,
                &[("val".to_string(), PartitionTransform::Identity)],
            )
            .unwrap();
        s.table_id
    };

    // One partition (val=1) holding 10 rows, so the rewrite has a single source.
    append(&temp, (1..=10).collect(), vec![1; 10]).await;
    let p = pool(&temp).await;
    let live_spec_id = scalar_i64(
        &p,
        "SELECT partition_id FROM ducklake_partition_info WHERE end_snapshot IS NULL",
    )
    .await;

    // Delete 8 of 10 rows, then rewrite past the 0.5 threshold.
    {
        let w = SqliteMetadataWriter::new(&db_url(&temp)).await.unwrap();
        let provider = SqliteMetadataProvider::new(&db_url(&temp)).await.unwrap();
        let catalog = DuckLakeCatalog::with_writer(Arc::new(provider), Arc::new(w)).unwrap();
        let ctx = SessionContext::new();
        ctx.register_catalog("ducklake", Arc::new(catalog));
        ctx.sql("DELETE FROM ducklake.main.t WHERE id <= 8")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
    }
    let result = run_rewrite(
        &temp,
        RewriteOptions {
            delete_threshold: 0.5,
            data_file_ids: None,
        },
    )
    .await;
    assert_eq!(result.files_created, 1, "the live rows are rewritten");

    // The rewritten file keeps the partition it came from.
    let (partition_id, value): (Option<i64>, Option<String>) = sqlx::query(
        "SELECT df.partition_id, fpv.partition_value
         FROM ducklake_data_file AS df
         LEFT JOIN ducklake_file_partition_value AS fpv
           ON fpv.data_file_id = df.data_file_id
         WHERE df.table_id = ? AND df.end_snapshot IS NULL
           AND df.begin_snapshot = (SELECT MAX(snapshot_id) FROM ducklake_snapshot)",
    )
    .bind(table_id)
    .fetch_one(&p)
    .await
    .map(|r| {
        (
            r.try_get::<Option<i64>, _>(0).unwrap(),
            r.try_get::<Option<String>, _>(1).unwrap(),
        )
    })
    .unwrap();
    assert_eq!(
        partition_id,
        Some(live_spec_id),
        "the rewritten file must keep its partition_id"
    );
    assert_eq!(
        value,
        Some("1".to_string()),
        "the rewritten file must keep its partition value"
    );
    // And the surviving rows read back.
    assert_eq!(read_rows(&temp).await, vec![(9, 1), (10, 1)]);
}

#[tokio::test(flavor = "multi_thread")]
async fn rewrite_can_target_explicit_data_files_without_deletes() {
    let temp = TempDir::new().unwrap();
    seed(&temp, vec![1, 2], vec![10, 20]).await;
    append(&temp, vec![3, 4], vec![30, 40]).await;
    let p = pool(&temp).await;
    let table_id = scalar_i64(
        &p,
        "SELECT table_id FROM ducklake_table WHERE table_name = 't'",
    )
    .await;
    let provider = SqliteMetadataProvider::new(&ro_url(&temp)).await.unwrap();
    let snapshot = provider.get_current_snapshot().unwrap();
    let mut before = provider
        .get_table_file_metadata_page(table_id, snapshot, None, 10)
        .unwrap();
    before.sort_by_key(|metadata| metadata.file.data_file_id);
    let selected_id = before[0].file.data_file_id;
    let unaffected_path = before[1].file.file.path.clone();

    assert_eq!(
        run_rewrite(
            &temp,
            RewriteOptions {
                data_file_ids: Some(vec![selected_id]),
                ..RewriteOptions::default()
            },
        )
        .await,
        CompactionResult {
            files_processed: 1,
            files_created: 1,
            rows_written: 2,
        },
    );

    let provider = SqliteMetadataProvider::new(&ro_url(&temp)).await.unwrap();
    let snapshot = provider.get_current_snapshot().unwrap();
    let after = provider
        .get_table_file_metadata_page(table_id, snapshot, None, 10)
        .unwrap();

    assert_eq!(after.len(), 2);
    assert!(
        after
            .iter()
            .any(|metadata| metadata.file.file.path == unaffected_path),
    );
    assert_eq!(
        read_rows(&temp).await,
        vec![(1, 10), (2, 20), (3, 30), (4, 40)]
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn rewrite_does_not_resurrect_inlined_deletes() {
    let temp = TempDir::new().unwrap();
    seed(&temp, vec![1, 2, 3], vec![10, 20, 30]).await;
    crate::inlined_delete_fixture::insert_inlined_deletes_for_only_file(
        &temp.path().join("test.db"),
        &[1],
    )
    .await;
    assert_eq!(read_rows(&temp).await, vec![(1, 10), (3, 30)]);

    assert_eq!(
        run_rewrite(
            &temp,
            RewriteOptions {
                delete_threshold: 0.3,
                ..RewriteOptions::default()
            },
        )
        .await,
        CompactionResult {
            files_processed: 1,
            files_created: 1,
            rows_written: 2,
        },
    );
    assert_eq!(read_rows(&temp).await, vec![(1, 10), (3, 30)]);
}

/// An inlined DELETE that lands between a compaction's plan and commit trips the
/// source-file fence instead of resurrecting the concurrently deleted row.
#[tokio::test(flavor = "multi_thread")]
async fn compaction_conflicts_with_a_concurrent_inlined_delete() {
    use datafusion_ducklake::metadata_writer::SourceRetirement;
    use datafusion_ducklake::{CompactionSourceFile, DuckLakeError};

    let temp = TempDir::new().unwrap();
    seed(&temp, vec![1, 2, 3], vec![10, 20, 30]).await;
    // The plan observed no inlined deletes; a concurrent writer then inlines one.
    let p = pool(&temp).await;
    let base_snapshot = scalar_i64(&p, "SELECT MAX(snapshot_id) FROM ducklake_snapshot").await;
    sqlx::query(
        "INSERT INTO ducklake_snapshot (snapshot_id, snapshot_time, schema_version)
         VALUES (?, CURRENT_TIMESTAMP, 1)",
    )
    .bind(base_snapshot + 1)
    .execute(&p)
    .await
    .unwrap();
    let data_file_id = crate::inlined_delete_fixture::insert_inlined_deletes_for_only_file(
        &temp.path().join("test.db"),
        &[1],
    )
    .await;

    let writer = SqliteMetadataWriter::new(&db_url(&temp)).await.unwrap();
    let table_id = scalar_i64(
        &p,
        "SELECT table_id FROM ducklake_table WHERE table_name = 't'",
    )
    .await;
    let error = writer
        .commit_compaction(
            table_id,
            base_snapshot,
            &[CompactionSourceFile {
                data_file_id,
                delete_file_id: None,
            }],
            &[],
            SourceRetirement::Retire,
        )
        .unwrap_err();
    assert!(
        matches!(error, DuckLakeError::Conflict(_)),
        "expected Conflict, got {error:?}"
    );
}

/// A file whose rows are masked only by inlined deletes has `delete_file_id = NULL`,
/// but merging it with `SourceRetirement::Remove` would erase the masked rows from
/// every snapshot and leave `ducklake_inlined_delete_<id>` rows pointing at a removed
/// file. Such a file must not be a merge candidate; unaffected files still merge.
#[tokio::test(flavor = "multi_thread")]
async fn merge_skips_files_masked_by_inlined_deletes() {
    let temp = TempDir::new().unwrap();
    seed(&temp, vec![1, 2], vec![10, 20]).await;
    let p = pool(&temp).await;
    let seed_snapshot = scalar_i64(&p, "SELECT MAX(snapshot_id) FROM ducklake_snapshot").await;
    append(&temp, vec![3, 4], vec![30, 40]).await;
    append(&temp, vec![5, 6], vec![50, 60]).await;
    let masked_file_id = crate::inlined_delete_fixture::insert_inlined_deletes_for_first_file(
        &temp.path().join("test.db"),
        &[0],
    )
    .await;
    assert_eq!(
        read_rows(&temp).await,
        vec![(2, 20), (3, 30), (4, 40), (5, 50), (6, 60)]
    );

    assert_eq!(
        run_merge(&temp, MergeOptions::default()).await,
        CompactionResult {
            files_processed: 2,
            files_created: 1,
            rows_written: 4,
        },
    );

    let provider = SqliteMetadataProvider::new(&ro_url(&temp)).await.unwrap();
    let snapshot = provider.get_current_snapshot().unwrap();
    let table_id = scalar_i64(
        &p,
        "SELECT table_id FROM ducklake_table WHERE table_name = 't'",
    )
    .await;
    let files = provider
        .get_table_file_metadata_page(table_id, snapshot, None, 10)
        .unwrap();
    assert!(
        files
            .iter()
            .any(|metadata| metadata.file.data_file_id == masked_file_id),
        "the masked file must survive the merge",
    );
    assert_eq!(
        read_rows(&temp).await,
        vec![(2, 20), (3, 30), (4, 40), (5, 50), (6, 60)]
    );
    assert_eq!(
        read_rows_at(&temp, seed_snapshot).await,
        vec![(1, 10), (2, 20)]
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn sorted_merge_under_memory_limit_preserves_rowids_and_snapshot_lineage() {
    const MEMORY_LIMIT: usize = 256 * 1024;
    const ROWS_PER_SNAPSHOT: i32 = 16_384;
    const TOTAL_ROWS: i32 = ROWS_PER_SNAPSHOT * 2;

    let temp = TempDir::new().unwrap();
    seed(
        &temp,
        (0..ROWS_PER_SNAPSHOT).collect(),
        (0..ROWS_PER_SNAPSHOT).collect(),
    )
    .await;
    let p = pool(&temp).await;
    let first_snapshot = scalar_i64(&p, "SELECT MAX(snapshot_id) FROM ducklake_snapshot").await;
    append(
        &temp,
        (ROWS_PER_SNAPSHOT..TOTAL_ROWS).collect(),
        (ROWS_PER_SNAPSHOT..TOTAL_ROWS).collect(),
    )
    .await;
    let table_id = scalar_i64(
        &p,
        "SELECT table_id FROM ducklake_table WHERE table_name = 't'",
    )
    .await;
    SqliteMetadataWriter::new(&db_url(&temp))
        .await
        .unwrap()
        .set_sort_spec(
            table_id,
            &[SortField::column(0, "val", SortDirection::Desc, NullOrder::NullsLast)],
        )
        .unwrap();
    let rowids_before = read_id_rowid(&temp).await;
    assert_eq!(
        usize::try_from(TOTAL_ROWS).unwrap()
            * (2 * std::mem::size_of::<i32>() + std::mem::size_of::<i64>()),
        2 * MEMORY_LIMIT,
    );

    let runtime = RuntimeEnvBuilder::new()
        .with_memory_limit(MEMORY_LIMIT, 1.0)
        .with_temp_file_path(temp.path().join("spill"))
        .build_arc()
        .unwrap();
    let config = SessionConfig::new()
        .with_batch_size(1024)
        .with_sort_spill_reservation_bytes(16 * 1024);
    let ctx = SessionContext::new_with_config_rt(config, runtime);
    let result = with_writable_table_context(&temp, ctx, |table, state| async move {
        table
            .merge_adjacent_files(&state, MergeOptions::default())
            .await
            .unwrap()
    })
    .await;

    let provider = SqliteMetadataProvider::new(&ro_url(&temp)).await.unwrap();
    let snapshot = provider.get_current_snapshot().unwrap();
    let files = provider
        .get_table_file_metadata_page(table_id, snapshot, None, 10)
        .unwrap();
    let expected_first_snapshot = (0..ROWS_PER_SNAPSHOT)
        .map(|value| (value, value))
        .collect::<Vec<_>>();

    assert_eq!(
        result,
        CompactionResult {
            files_processed: 2,
            files_created: 1,
            rows_written: i64::from(TOTAL_ROWS),
        },
    );
    assert_eq!(files.len(), 1);
    assert_eq!(
        file_values(&temp, &files[0].file.file.path),
        (0..TOTAL_ROWS).rev().collect::<Vec<_>>(),
    );
    assert_eq!(read_id_rowid(&temp).await, rowids_before);
    assert_eq!(
        read_rows_at(&temp, first_snapshot).await,
        expected_first_snapshot,
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn merge_harvests_output_stats_and_removes_source_stats() {
    // Compaction must record fresh per-file column stats for the merged output
    // (so it stays prunable) and hard-delete the merged-away sources' stats rows
    // (no orphans) — mirroring official DuckLake.
    let temp = TempDir::new().unwrap();
    seed(&temp, vec![1, 2, 3], vec![10, 20, 30]).await;
    append(&temp, vec![4, 5, 6], vec![40, 50, 60]).await;
    let p = pool(&temp).await;

    // Two source files, each with per-column stats rows.
    assert_eq!(
        scalar_i64(
            &p,
            "SELECT COUNT(*) FROM ducklake_data_file WHERE end_snapshot IS NULL"
        )
        .await,
        2
    );
    assert!(
        scalar_i64(&p, "SELECT COUNT(*) FROM ducklake_file_column_stats").await >= 2,
        "each source file should have stats rows"
    );

    run_merge(&temp, MergeOptions::default()).await;

    // No orphaned stats rows: every remaining stats row points at a live data
    // file (the sources' rows were deleted with their data_file rows).
    assert_eq!(
        scalar_i64(
            &p,
            "SELECT COUNT(*) FROM ducklake_file_column_stats s
             LEFT JOIN ducklake_data_file d ON d.data_file_id = s.data_file_id
             WHERE d.data_file_id IS NULL",
        )
        .await,
        0,
        "merge must delete the retired sources' stats rows"
    );
    // The merged output carries harvested stats spanning both sources: the `id`
    // column's bound is now [1, 6].
    assert_eq!(
        scalar_i64(
            &p,
            "SELECT COUNT(*) FROM ducklake_file_column_stats s
             JOIN ducklake_data_file d ON d.data_file_id = s.data_file_id
             WHERE d.end_snapshot IS NULL AND s.min_value = '1' AND s.max_value = '6'",
        )
        .await,
        1,
        "merged file's id-column zone map should span the union of the sources"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn merge_nested_table_uses_top_level_stats_ids() {
    let temp = TempDir::new().unwrap();
    let values1 = ListArray::from_iter_primitive::<Int32Type, _, _>(vec![
        Some(vec![Some(10), Some(11)]),
        Some(vec![Some(20)]),
    ]);
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("items", values1.data_type().clone(), true),
    ]));
    let writer: Arc<dyn MetadataWriter> = Arc::new(make_writer(&temp).await);
    let batch1 = batch(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(vec![1, 2])), Arc::new(values1)],
    );
    DuckLakeTableWriter::new(Arc::clone(&writer), object_store())
        .unwrap()
        .write_table("main", "t", &[batch1])
        .await
        .unwrap();

    let values2 = ListArray::from_iter_primitive::<Int32Type, _, _>(vec![
        Some(vec![Some(30), Some(31)]),
        Some(vec![Some(40)]),
    ]);
    let batch2 = batch(
        schema,
        vec![Arc::new(Int32Array::from(vec![3, 4])), Arc::new(values2)],
    );
    DuckLakeTableWriter::new(writer, object_store())
        .unwrap()
        .append_table("main", "t", &[batch2])
        .await
        .unwrap();

    assert_eq!(
        run_merge(&temp, MergeOptions::default()).await,
        CompactionResult {
            files_processed: 2,
            files_created: 1,
            rows_written: 4,
        },
    );

    let p = pool(&temp).await;
    assert_eq!(
        scalar_i64(
            &p,
            "SELECT COUNT(*) FROM ducklake_data_file WHERE end_snapshot IS NULL",
        )
        .await,
        1,
    );
    assert_eq!(
        scalar_i64(
            &p,
            "SELECT COUNT(*) FROM ducklake_file_column_stats s
             JOIN ducklake_data_file d ON d.data_file_id = s.data_file_id
             JOIN ducklake_column c ON c.column_id = s.column_id
             WHERE d.end_snapshot IS NULL AND c.parent_column IS NOT NULL",
        )
        .await,
        0,
        "compaction stats must not label an embedded column with a nested field id",
    );

    let provider = SqliteMetadataProvider::new(&ro_url(&temp)).await.unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog(
        "ducklake",
        Arc::new(DuckLakeCatalog::new(provider).unwrap()),
    );
    let batches = ctx
        .sql("SELECT id, items FROM ducklake.main.t ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(batches.len(), 1);
    let ids = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    let values = batches[0]
        .column(1)
        .as_any()
        .downcast_ref::<ListArray>()
        .unwrap();
    assert_eq!(ids.values(), &[1, 2, 3, 4]);
    let expected = [vec![10, 11], vec![20], vec![30, 31], vec![40]];
    for (index, expected) in expected.iter().enumerate() {
        let values = values.value(index);
        let values = values.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(values.values(), expected);
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn merge_coalesces_small_files_preserving_results_rowids_and_time_travel() {
    let temp = TempDir::new().unwrap();
    // Three inserts -> three small data files, all at schema version 1.
    seed(&temp, vec![1, 2], vec![10, 20]).await;
    append(&temp, vec![3, 4], vec![30, 40]).await;
    append(&temp, vec![5, 6], vec![50, 60]).await;

    let p = pool(&temp).await;
    let tid = scalar_i64(&p, "SELECT table_id FROM ducklake_table LIMIT 1").await;
    let pre_snapshot = scalar_i64(&p, "SELECT MAX(snapshot_id) FROM ducklake_snapshot").await;
    let snapshots_before = scalar_i64(&p, "SELECT COUNT(*) FROM ducklake_snapshot").await;
    let live_before = scalar_i64(
        &p,
        "SELECT COUNT(*) FROM ducklake_data_file WHERE end_snapshot IS NULL",
    )
    .await;
    assert_eq!(live_before, 3, "three small files before merge");
    // The oldest source's origin snapshot — the merged partial file must begin
    // here so historical reads back to this point still see it.
    let min_origin = scalar_i64(&p, "SELECT MIN(begin_snapshot) FROM ducklake_data_file").await;

    let rows_before = read_rows(&temp).await;
    let id_rowid_before = read_id_rowid(&temp).await;
    assert_eq!(
        rows_before,
        vec![(1, 10), (2, 20), (3, 30), (4, 40), (5, 50), (6, 60)]
    );

    // Default options: a huge target coalesces all three tiny files into one.
    let result = run_merge(&temp, MergeOptions::default()).await;
    assert_eq!(result.files_processed, 3, "all three sources merged");
    assert_eq!(result.files_created, 1, "into one file");
    assert_eq!(result.rows_written, 6);

    // Exactly one new snapshot.
    let snapshots_after = scalar_i64(&p, "SELECT COUNT(*) FROM ducklake_snapshot").await;
    assert_eq!(
        snapshots_after,
        snapshots_before + 1,
        "exactly one new snapshot"
    );
    let new_snapshot = scalar_i64(&p, "SELECT MAX(snapshot_id) FROM ducklake_snapshot").await;
    assert_eq!(new_snapshot, pre_snapshot + 1);

    // Fewer live files: exactly one, and it is the partial merged file.
    let live_after = scalar_i64(
        &p,
        "SELECT COUNT(*) FROM ducklake_data_file WHERE end_snapshot IS NULL",
    )
    .await;
    assert_eq!(live_after, 1, "one live file after merge");
    let partial_max = opt_i64(
        &p,
        "SELECT partial_max FROM ducklake_data_file WHERE end_snapshot IS NULL",
    )
    .await;
    assert_eq!(
        partial_max,
        Some(pre_snapshot),
        "partial_max = max origin snapshot among merged rows"
    );
    let merged_row_id_start = opt_i64(
        &p,
        "SELECT row_id_start FROM ducklake_data_file WHERE end_snapshot IS NULL",
    )
    .await;
    assert_eq!(
        merged_row_id_start, None,
        "merged file serves rowids inline"
    );
    let merged_begin = scalar_i64(
        &p,
        "SELECT begin_snapshot FROM ducklake_data_file WHERE end_snapshot IS NULL",
    )
    .await;
    assert_eq!(
        merged_begin, min_origin,
        "merged partial file begins at the MIN origin snapshot (visible to history)"
    );

    // The three source rows are REMOVED from the catalog (not just retired) — the
    // partial file now represents them for every snapshot — so only the one
    // merged row remains in ducklake_data_file.
    let total_rows = scalar_i64(&p, "SELECT COUNT(*) FROM ducklake_data_file").await;
    assert_eq!(
        total_rows, 1,
        "source rows removed; only the merged file remains"
    );
    // Their physical files are scheduled for deletion (safe: unreachable now).
    let scheduled = scalar_i64(
        &p,
        "SELECT COUNT(*) FROM ducklake_files_scheduled_for_deletion",
    )
    .await;
    assert_eq!(scheduled, 3, "three source files scheduled for deletion");

    // changes_made records the compaction.
    let changes: String =
        sqlx::query("SELECT changes_made FROM ducklake_snapshot_changes WHERE snapshot_id = ?")
            .bind(new_snapshot)
            .fetch_one(&p)
            .await
            .unwrap()
            .try_get(0)
            .unwrap();
    assert_eq!(changes, format!("compacted_table:{tid}"));

    // Identical query results, and rowid lineage preserved across the rewrite.
    assert_eq!(
        read_rows(&temp).await,
        rows_before,
        "results unchanged by merge"
    );
    assert_eq!(
        read_id_rowid(&temp).await,
        id_rowid_before,
        "rowids preserved across merge"
    );

    // Time travel to the pre-merge snapshot still returns the original rows
    // (the retired source files are only scheduled, not yet deleted).
    assert_eq!(
        read_rows_at(&temp, pre_snapshot).await,
        rows_before,
        "time travel to pre-merge snapshot returns the original rows"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn merge_applies_sqlite_sort_order_to_physical_output() {
    let temp = TempDir::new().unwrap();
    seed(&temp, vec![1, 2], vec![20, 10]).await;
    append(&temp, vec![3, 4], vec![40, 30]).await;
    append(&temp, vec![5, 6], vec![60, 50]).await;
    let p = pool(&temp).await;
    let table_id = scalar_i64(&p, "SELECT table_id FROM ducklake_table LIMIT 1").await;
    let writer = SqliteMetadataWriter::new(&db_url(&temp)).await.unwrap();
    writer
        .set_sort_spec(
            table_id,
            &[SortField::column(0, "val", SortDirection::Desc, NullOrder::NullsLast)],
        )
        .unwrap();

    let result = run_merge(&temp, MergeOptions::default()).await;

    let live_path: String =
        sqlx::query_scalar("SELECT path FROM ducklake_data_file WHERE end_snapshot IS NULL")
            .fetch_one(&p)
            .await
            .unwrap();
    assert_eq!(
        result,
        CompactionResult {
            files_processed: 3,
            files_created: 1,
            rows_written: 6,
        },
    );
    assert_eq!(file_values(&temp, &live_path), vec![60, 50, 40, 30, 20, 10]);
    assert_eq!(
        read_rows(&temp).await,
        vec![(1, 20), (2, 10), (3, 40), (4, 30), (5, 60), (6, 50)],
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn sorted_merge_with_limited_memory_preserves_row_and_snapshot_lineage() {
    const ROWS_PER_FILE: i32 = 30_000;
    const ROW_COUNT: i32 = ROWS_PER_FILE * 3;

    let temp = TempDir::new().unwrap();
    seed(
        &temp,
        (0..ROWS_PER_FILE).collect(),
        (0..ROWS_PER_FILE).map(|id| id * 3).collect(),
    )
    .await;
    let p = pool(&temp).await;
    let first_snapshot = scalar_i64(&p, "SELECT MAX(snapshot_id) FROM ducklake_snapshot").await;
    append(
        &temp,
        (ROWS_PER_FILE..ROWS_PER_FILE * 2).collect(),
        (0..ROWS_PER_FILE).map(|id| id * 3 + 1).collect(),
    )
    .await;
    append(
        &temp,
        (ROWS_PER_FILE * 2..ROW_COUNT).collect(),
        (0..ROWS_PER_FILE).map(|id| id * 3 + 2).collect(),
    )
    .await;
    let table_id = scalar_i64(&p, "SELECT table_id FROM ducklake_table LIMIT 1").await;
    let writer = SqliteMetadataWriter::new(&db_url(&temp)).await.unwrap();
    writer
        .set_sort_spec(
            table_id,
            &[SortField::column(0, "val", SortDirection::Asc, NullOrder::NullsLast)],
        )
        .unwrap();
    let rowids_before = read_id_rowid(&temp).await;

    let spill = TempDir::new().unwrap();
    let runtime = Arc::new(
        RuntimeEnvBuilder::new()
            .with_memory_pool(Arc::new(FairSpillPool::new(1 << 20)))
            .with_temp_file_path(spill.path())
            .build()
            .unwrap(),
    );
    let provider = SqliteMetadataProvider::new(&db_url(&temp)).await.unwrap();
    let writer = SqliteMetadataWriter::new(&db_url(&temp)).await.unwrap();
    let catalog = DuckLakeCatalog::with_writer(Arc::new(provider), Arc::new(writer)).unwrap();
    let config = SessionConfig::new()
        .with_batch_size(1_024)
        .with_sort_spill_reservation_bytes(128 << 10);
    let ctx = SessionContext::new_with_config_rt(config, runtime);
    ctx.register_catalog("ducklake", Arc::new(catalog));
    let provider = ctx
        .catalog("ducklake")
        .unwrap()
        .schema("main")
        .unwrap()
        .table("t")
        .await
        .unwrap()
        .unwrap();
    let table = (provider.as_ref() as &dyn std::any::Any)
        .downcast_ref::<DuckLakeTable>()
        .unwrap();

    let result = table
        .merge_adjacent_files(&ctx.state(), MergeOptions::default())
        .await
        .unwrap();

    let live_path: String =
        sqlx::query_scalar("SELECT path FROM ducklake_data_file WHERE end_snapshot IS NULL")
            .fetch_one(&p)
            .await
            .unwrap();
    assert_eq!(
        result,
        CompactionResult {
            files_processed: 3,
            files_created: 1,
            rows_written: i64::from(ROW_COUNT),
        },
    );
    assert_eq!(
        file_values(&temp, &live_path),
        (0..ROW_COUNT).collect::<Vec<_>>(),
    );
    assert_eq!(read_id_rowid(&temp).await, rowids_before);
    assert_eq!(
        read_rows_at(&temp, first_snapshot).await,
        (0..ROWS_PER_FILE)
            .map(|id| (id, id * 3))
            .collect::<Vec<_>>(),
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn rewrite_drops_deleted_rows_and_retires_data_and_delete_files() {
    let temp = TempDir::new().unwrap();
    // One file of ten rows.
    seed(
        &temp,
        (1..=10).collect(),
        (1..=10).map(|v| v * 10).collect(),
    )
    .await;
    let p = pool(&temp).await;
    let tid = scalar_i64(&p, "SELECT table_id FROM ducklake_table LIMIT 1").await;
    let create_snapshot = scalar_i64(&p, "SELECT MAX(snapshot_id) FROM ducklake_snapshot").await;

    // Delete eight of the ten rows via SQL (a positional delete file).
    {
        let writer = SqliteMetadataWriter::new(&db_url(&temp)).await.unwrap();
        let provider = SqliteMetadataProvider::new(&db_url(&temp)).await.unwrap();
        let catalog = DuckLakeCatalog::with_writer(Arc::new(provider), Arc::new(writer)).unwrap();
        let ctx = SessionContext::new();
        ctx.register_catalog("ducklake", Arc::new(catalog));
        ctx.sql("DELETE FROM ducklake.main.t WHERE id <= 8")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
    }
    let after_delete_snapshot =
        scalar_i64(&p, "SELECT MAX(snapshot_id) FROM ducklake_snapshot").await;
    assert_eq!(
        read_rows(&temp).await,
        vec![(9, 90), (10, 100)],
        "8 of 10 deleted"
    );
    // Sanity: one live data file with a live delete file masking 8 rows.
    assert_eq!(
        scalar_i64(
            &p,
            "SELECT COUNT(*) FROM ducklake_delete_file WHERE end_snapshot IS NULL"
        )
        .await,
        1
    );

    // 8/10 = 0.8 deleted; rewrite with a 0.5 threshold.
    let result = run_rewrite(
        &temp,
        RewriteOptions {
            delete_threshold: 0.5,
            ..RewriteOptions::default()
        },
    )
    .await;
    assert_eq!(result.files_processed, 1);
    assert_eq!(result.files_created, 1);
    assert_eq!(result.rows_written, 2, "only the two live rows rewritten");

    let new_snapshot = scalar_i64(&p, "SELECT MAX(snapshot_id) FROM ducklake_snapshot").await;
    assert_eq!(
        new_snapshot,
        after_delete_snapshot + 1,
        "exactly one new snapshot"
    );

    // Results unchanged.
    assert_eq!(read_rows(&temp).await, vec![(9, 90), (10, 100)]);

    // Exactly one live data file, with record_count = live rows and no live delete file.
    assert_eq!(
        scalar_i64(
            &p,
            "SELECT COUNT(*) FROM ducklake_data_file WHERE end_snapshot IS NULL"
        )
        .await,
        1
    );
    assert_eq!(
        scalar_i64(
            &p,
            "SELECT record_count FROM ducklake_data_file WHERE end_snapshot IS NULL"
        )
        .await,
        2,
        "new file contains only the live rows"
    );
    assert_eq!(
        opt_i64(
            &p,
            "SELECT partial_max FROM ducklake_data_file WHERE end_snapshot IS NULL"
        )
        .await,
        None,
        "a rewrite output is not a partial file"
    );
    assert_eq!(
        scalar_i64(
            &p,
            "SELECT COUNT(*) FROM ducklake_delete_file WHERE end_snapshot IS NULL"
        )
        .await,
        0,
        "no live delete file after rewrite"
    );

    // BOTH the old data file AND its delete file retired at the new snapshot ...
    assert_eq!(
        scalar_i64(
            &p,
            &format!("SELECT COUNT(*) FROM ducklake_data_file WHERE end_snapshot = {new_snapshot}")
        )
        .await,
        1
    );
    assert_eq!(
        scalar_i64(
            &p,
            &format!(
                "SELECT COUNT(*) FROM ducklake_delete_file WHERE end_snapshot = {new_snapshot}"
            )
        )
        .await,
        1
    );
    // ... but NOT scheduled for deletion: a rewrite output holds only the
    // currently-live rows, so the retired source (all ten rows + its delete
    // file) still serves time travel to the pre-rewrite snapshot. It is
    // reclaimed later by expire_snapshots, not at the rewrite.
    assert_eq!(
        scalar_i64(
            &p,
            "SELECT COUNT(*) FROM ducklake_files_scheduled_for_deletion"
        )
        .await,
        0,
        "rewrite sources are retained (not scheduled) for time travel"
    );

    // changes_made records the compaction.
    let changes: String =
        sqlx::query("SELECT changes_made FROM ducklake_snapshot_changes WHERE snapshot_id = ?")
            .bind(new_snapshot)
            .fetch_one(&p)
            .await
            .unwrap()
            .try_get(0)
            .unwrap();
    assert_eq!(changes, format!("compacted_table:{tid}"));

    // Rowid lineage of the surviving rows preserved (row 9 was position 8, row 10 position 9).
    assert_eq!(read_id_rowid(&temp).await, vec![(9, 8), (10, 9)]);

    // Time travel to before the DELETE still returns all ten rows (the original
    // data file is retained, and — unscheduled — survives even a cleanup).
    assert_eq!(read_rows_at(&temp, create_snapshot).await.len(), 10);
}

#[tokio::test(flavor = "multi_thread")]
async fn merge_respects_schema_version_boundary() {
    let temp = TempDir::new().unwrap();
    // Two files at schema version 1.
    seed(&temp, vec![1, 2], vec![10, 20]).await;
    append(&temp, vec![3, 4], vec![30, 40]).await;

    // A DDL (append a batch with an extra column) bumps schema_version to 2 and
    // adds a third file under the new version, without retiring the first two.
    {
        let three_col = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("val", DataType::Int32, false),
            Field::new("note", DataType::Int32, true),
        ]));
        let writer = Arc::new(SqliteMetadataWriter::new(&db_url(&temp)).await.unwrap());
        let b = batch(
            three_col,
            vec![
                Arc::new(Int32Array::from(vec![5, 6])),
                Arc::new(Int32Array::from(vec![50, 60])),
                Arc::new(Int32Array::from(vec![500, 600])),
            ],
        );
        DuckLakeTableWriter::new(writer, object_store())
            .unwrap()
            .append_table("main", "t", &[b])
            .await
            .unwrap();
    }

    let p = pool(&temp).await;
    // Confirm the setup: three live files spanning two schema versions.
    assert_eq!(
        scalar_i64(
            &p,
            "SELECT COUNT(*) FROM ducklake_data_file WHERE end_snapshot IS NULL"
        )
        .await,
        3
    );
    assert_eq!(
        scalar_i64(
            &p,
            "SELECT COUNT(DISTINCT schema_version) FROM ducklake_schema_versions"
        )
        .await,
        2,
        "a DDL bumped the schema version"
    );
    let v2_file = scalar_i64(
        &p,
        "SELECT MAX(data_file_id) FROM ducklake_data_file WHERE end_snapshot IS NULL",
    )
    .await;

    // Merge: only the two same-version files may combine; the newer-version file
    // must be left alone (never merged across the DDL boundary).
    let result = run_merge(&temp, MergeOptions::default()).await;
    assert_eq!(result.files_processed, 2, "only the two v1 files merged");
    assert_eq!(result.files_created, 1);

    // The v2 file is untouched (still live, never scheduled).
    assert_eq!(
        scalar_i64(
            &p,
            &format!(
                "SELECT COUNT(*) FROM ducklake_data_file \
                 WHERE data_file_id = {v2_file} AND end_snapshot IS NULL"
            )
        )
        .await,
        1,
        "the newer-schema-version file was not merged"
    );
    assert_eq!(
        scalar_i64(
            &p,
            "SELECT COUNT(*) FROM ducklake_files_scheduled_for_deletion"
        )
        .await,
        2,
        "only the two v1 source files scheduled"
    );
    // Two live files remain: the merged v1 file and the untouched v2 file.
    assert_eq!(
        scalar_i64(
            &p,
            "SELECT COUNT(*) FROM ducklake_data_file WHERE end_snapshot IS NULL"
        )
        .await,
        2
    );

    // Results are unchanged by the (partial) merge.
    assert_eq!(
        read_rows(&temp).await,
        vec![(1, 10), (2, 20), (3, 30), (4, 40), (5, 50), (6, 60)]
    );
}

/// A partial file IS a merge candidate, and a re-merge must carry every row's
/// OWN origin snapshot through: its scan projects the embedded
/// `_ducklake_internal_snapshot_id` column, so the output keeps the distinct
/// origins rather than collapsing them onto one `begin_snapshot`.
#[tokio::test(flavor = "multi_thread")]
async fn merge_remerges_a_partial_file_preserving_per_row_origins() {
    let temp = TempDir::new().unwrap();
    seed(&temp, vec![1], vec![10]).await; // snapshot 1
    append(&temp, vec![2], vec![20]).await; // snapshot 2
    append(&temp, vec![3], vec![30]).await; // snapshot 3

    // Merge #1 -> partial file P (origins {1,2,3}, begin=1, partial_max=3).
    let r1 = run_merge(&temp, MergeOptions::default()).await;
    assert_eq!(r1.files_created, 1);
    let p = pool(&temp).await;
    assert_eq!(
        opt_i64(
            &p,
            "SELECT partial_max FROM ducklake_data_file WHERE end_snapshot IS NULL"
        )
        .await,
        Some(3),
        "merge produced a partial file"
    );

    append(&temp, vec![4], vec![40]).await; // one more small file, D
    let s_d = scalar_i64(&p, "SELECT MAX(snapshot_id) FROM ducklake_snapshot").await;

    // Merge #2 re-merges P together with D into one file.
    let r2 = run_merge(&temp, MergeOptions::default()).await;
    assert_eq!(r2.files_processed, 2, "P and D merge together");
    assert_eq!(r2.files_created, 1);
    assert_eq!(
        scalar_i64(
            &p,
            "SELECT COUNT(*) FROM ducklake_data_file WHERE end_snapshot IS NULL"
        )
        .await,
        1,
        "the re-merge leaves exactly one live file"
    );

    // The output spans [MIN begin_snapshot, MAX(partial_max ?? begin_snapshot)]
    // of its sources — P's interval [1, 3] unioned with D's point — matching
    // official DuckLake's `GetCompactionChanges`.
    let merged_path: String = sqlx::query(AssertSqlSafe(
        "SELECT path FROM ducklake_data_file WHERE end_snapshot IS NULL",
    ))
    .fetch_one(&p)
    .await
    .unwrap()
    .try_get::<String, _>(0)
    .unwrap();
    assert_eq!(
        opt_i64(
            &p,
            "SELECT begin_snapshot FROM ducklake_data_file WHERE end_snapshot IS NULL"
        )
        .await,
        Some(1),
        "output begins at the minimum origin of its sources"
    );
    assert_eq!(
        opt_i64(
            &p,
            "SELECT partial_max FROM ducklake_data_file WHERE end_snapshot IS NULL"
        )
        .await,
        Some(s_d),
        "output partial_max is the maximum origin any of its rows carries"
    );

    // On disk, P's four rows keep their four DISTINCT origins. Keying the
    // decision on the catalog instead would have stamped P's rows with its
    // single begin_snapshot and left only {1, s_d} here.
    let mut origins: Vec<i64> = file_lineage(&temp, &merged_path)
        .into_iter()
        .map(|(_, origin)| origin.expect("a re-merged output embeds per-row origins"))
        .collect();
    origins.sort_unstable();
    assert_eq!(origins, vec![1, 2, 3, s_d]);

    // Time travel therefore still attributes each row to its own origin.
    assert_eq!(read_rows_at(&temp, 1).await, vec![(1, 10)]);
    assert_eq!(read_rows_at(&temp, 2).await, vec![(1, 10), (2, 20)]);
    assert_eq!(
        read_rows_at(&temp, 3).await,
        vec![(1, 10), (2, 20), (3, 30)]
    );
    assert_eq!(
        read_rows(&temp).await,
        vec![(1, 10), (2, 20), (3, 30), (4, 40)]
    );
}

/// The convergence property, in the shape that actually occurs: appends
/// INTERLEAVED with merge passes. Every merge whose sources span more than one
/// origin snapshot writes a partial file, so on a table taking frequent appends
/// nearly every output is partial. If partial files were excluded from the
/// candidates, each pass would strand its own output and the live file count
/// would climb by one per pass (1, 2, 3, …) with no bound — the partition would
/// accumulate a floor of files nothing can reduce.
///
/// Merging repeatedly on a STATIC table does not exercise this: the first pass
/// consumes everything and later passes have nothing to do.
#[tokio::test(flavor = "multi_thread")]
async fn interleaved_appends_and_merges_keep_the_live_file_count_flat() {
    let temp = TempDir::new().unwrap();
    seed(&temp, vec![0], vec![0]).await;
    let p = pool(&temp).await;

    // Each pass appends one small file and then merges, so every pass after the
    // first has a partial file among its sources.
    let mut origins: Vec<(i64, i32)> = Vec::new();
    for id in 1..=6i32 {
        append(&temp, vec![id], vec![id * 10]).await;
        origins.push((
            scalar_i64(&p, "SELECT MAX(snapshot_id) FROM ducklake_snapshot").await,
            id,
        ));
        let result = run_merge(&temp, MergeOptions::default()).await;
        assert_eq!(
            scalar_i64(
                &p,
                "SELECT COUNT(*) FROM ducklake_data_file WHERE end_snapshot IS NULL"
            )
            .await,
            1,
            "pass {id}: the live file count must not climb"
        );
        assert_eq!(
            result.files_processed, 2,
            "pass {id}: the previous output and the new append merged together"
        );
    }

    // Every row is still attributed to the snapshot that appended it.
    for (snapshot, id) in &origins {
        let expected: Vec<(i32, i32)> = std::iter::once((0, 0))
            .chain((1..=*id).map(|i| (i, i * 10)))
            .collect();
        assert_eq!(
            read_rows_at(&temp, *snapshot).await,
            expected,
            "time travel to snapshot {snapshot} sees exactly the rows appended by then"
        );
    }
}

/// Whether a merge source carries per-row origins is decided by the PHYSICAL
/// presence of the embedded column, never by the catalog's `partial_max`. The
/// two can disagree: a provider that does not read the column substitutes NULL,
/// as does a catalog predating it. Keying off the catalog would re-stamp every
/// row of such a file with one origin and drop the embedded column — history
/// lost irreversibly, since the same commit removes the sources.
#[tokio::test(flavor = "multi_thread")]
async fn merge_reads_per_row_origins_from_the_file_not_the_catalog() {
    let temp = TempDir::new().unwrap();
    seed(&temp, vec![1], vec![10]).await; // snapshot 1
    append(&temp, vec![2], vec![20]).await; // snapshot 2

    // Merge #1 -> partial file P (origins {1, 2}).
    assert_eq!(
        run_merge(&temp, MergeOptions::default())
            .await
            .files_created,
        1
    );

    // Simulate a catalog that lost the field: P still physically embeds its
    // per-row origins, but its catalog row now says it is not partial.
    let writable = SqlitePool::connect(&db_url(&temp)).await.unwrap();
    sqlx::query(AssertSqlSafe(
        "UPDATE ducklake_data_file SET partial_max = NULL WHERE end_snapshot IS NULL",
    ))
    .execute(&writable)
    .await
    .unwrap();

    append(&temp, vec![3], vec![30]).await;
    let p = pool(&temp).await;
    let s3 = scalar_i64(&p, "SELECT MAX(snapshot_id) FROM ducklake_snapshot").await;

    let result = run_merge(&temp, MergeOptions::default()).await;
    assert_eq!(result.files_processed, 2, "P is still a candidate");

    // P's rows kept their own origins, so the output holds three distinct ones.
    let merged_path: String = sqlx::query(AssertSqlSafe(
        "SELECT path FROM ducklake_data_file WHERE end_snapshot IS NULL",
    ))
    .fetch_one(&p)
    .await
    .unwrap()
    .try_get::<String, _>(0)
    .unwrap();
    let mut origins: Vec<i64> = file_lineage(&temp, &merged_path)
        .into_iter()
        .map(|(_, origin)| origin.expect("the output embeds per-row origins"))
        .collect();
    origins.sort_unstable();
    assert_eq!(
        origins,
        vec![1, 2, s3],
        "origins came from P's embedded column, not from its catalog begin_snapshot"
    );

    assert_eq!(read_rows_at(&temp, 1).await, vec![(1, 10)]);
    assert_eq!(read_rows_at(&temp, 2).await, vec![(1, 10), (2, 20)]);
    assert_eq!(read_rows(&temp).await, vec![(1, 10), (2, 20), (3, 30)]);
}

/// Merge must not silently drop a column's data. When a column has been dropped
/// since a file was written, merging that file (output is written at the CURRENT
/// schema) would lose the column — so `merge_adjacent_files` skips such files.
#[tokio::test(flavor = "multi_thread")]
async fn merge_skips_files_whose_columns_were_dropped() {
    let temp = TempDir::new().unwrap();
    // Two (id, val) files at schema version 1.
    seed(&temp, vec![1, 2], vec![10, 20]).await; // snapshot 1 (file A)
    append(&temp, vec![3, 4], vec![30, 40]).await; // snapshot 2 (file B)

    // A DDL that DROPS `val` (append a batch with only `id`) bumps the schema and
    // ends the `val` column; A and B stay live (Append), now at an older version
    // whose schema includes a column absent from the current one.
    {
        let id_only = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let writer = Arc::new(SqliteMetadataWriter::new(&db_url(&temp)).await.unwrap());
        let b = batch(id_only, vec![Arc::new(Int32Array::from(vec![5]))]);
        DuckLakeTableWriter::new(writer, object_store())
            .unwrap()
            .append_table("main", "t", &[b])
            .await
            .unwrap();
    }

    let p = pool(&temp).await;
    assert_eq!(
        scalar_i64(
            &p,
            "SELECT COUNT(*) FROM ducklake_data_file WHERE end_snapshot IS NULL"
        )
        .await,
        3,
        "three live files (A, B at v1; C at v2)"
    );

    // Merge: the v1 group {A, B} carries `val`, which the current schema dropped,
    // so it is skipped (merging would lose `val`); the v2 file is a singleton.
    let result = run_merge(&temp, MergeOptions::default()).await;
    assert_eq!(
        result.files_processed, 0,
        "column-dropping files are not merged"
    );
    assert_eq!(
        scalar_i64(
            &p,
            "SELECT COUNT(*) FROM ducklake_files_scheduled_for_deletion"
        )
        .await,
        0,
        "nothing merged, nothing scheduled"
    );
    assert_eq!(
        scalar_i64(
            &p,
            "SELECT COUNT(*) FROM ducklake_data_file WHERE end_snapshot IS NULL"
        )
        .await,
        3,
        "all three files remain live (A, B not removed)"
    );

    // Time travel to snapshot 1 still returns A's rows WITH `val` intact — proof
    // that A was not merged into a current-schema (val-less) file and removed.
    assert_eq!(read_rows_at(&temp, 1).await, vec![(1, 10), (2, 20)]);
}

/// The durability property: after a merge, physically deleting the retired
/// source files (via `cleanup_old_files`) must NOT break time travel — the
/// merged partial file serves every historical snapshot on its own, via per-row
/// `_ducklake_internal_snapshot_id` filtering. This is the case the pre-fix
/// implementation got wrong (it scheduled sources while the merged file was
/// invisible to historical reads).
#[tokio::test(flavor = "multi_thread")]
async fn merge_partial_file_serves_time_travel_after_sources_are_deleted() {
    let temp = TempDir::new().unwrap();
    seed(&temp, vec![1, 2], vec![10, 20]).await; // snapshot 1
    append(&temp, vec![3, 4], vec![30, 40]).await; // snapshot 2
    append(&temp, vec![5, 6], vec![50, 60]).await; // snapshot 3

    let p = pool(&temp).await;
    let result = run_merge(&temp, MergeOptions::default()).await;
    assert_eq!(result.files_processed, 3);
    assert_eq!(result.files_created, 1);

    // Physically delete the scheduled source parquet files. Afterwards the three
    // ORIGINAL files are gone from disk; only the merged partial file remains.
    let deleted = {
        let writer = SqliteMetadataWriter::new(&db_url(&temp)).await.unwrap();
        cleanup_old_files_sqlite(&writer, object_store(), CleanupCriteria::All, false)
            .await
            .unwrap()
    };
    assert_eq!(
        deleted.len(),
        3,
        "three source parquet files physically deleted"
    );
    assert_eq!(
        scalar_i64(
            &p,
            "SELECT COUNT(*) FROM ducklake_files_scheduled_for_deletion"
        )
        .await,
        0,
        "scheduled rows cleared after cleanup"
    );

    // Time travel is now served ENTIRELY by the merged partial file (the sources
    // no longer exist) via per-row origin-snapshot filtering.
    assert_eq!(
        read_rows_at(&temp, 1).await,
        vec![(1, 10), (2, 20)],
        "as of snapshot 1: only the first insert"
    );
    assert_eq!(
        read_rows_at(&temp, 2).await,
        vec![(1, 10), (2, 20), (3, 30), (4, 40)],
        "as of snapshot 2: first two inserts"
    );
    assert_eq!(
        read_rows_at(&temp, 3).await,
        vec![(1, 10), (2, 20), (3, 30), (4, 40), (5, 50), (6, 60)],
        "as of snapshot 3: all three inserts"
    );
    // The current snapshot still returns everything.
    assert_eq!(read_rows(&temp).await.len(), 6);
}

/// CDC over a crate-compacted catalog: `ducklake_table_changes` must attribute
/// each merged row to its ORIGIN snapshot (via the embedded per-row snapshot
/// column) and honor windows that reach the merged file only through
/// `partial_max`, matching official DuckLake.
#[tokio::test(flavor = "multi_thread")]
async fn cdc_attributes_merged_rows_to_origin_snapshots() {
    let temp = TempDir::new().unwrap();
    seed(&temp, vec![1], vec![10]).await;
    let p = pool(&temp).await;
    let s1 = scalar_i64(&p, "SELECT MAX(snapshot_id) FROM ducklake_snapshot").await;
    append(&temp, vec![2], vec![20]).await;
    let s2 = scalar_i64(&p, "SELECT MAX(snapshot_id) FROM ducklake_snapshot").await;
    append(&temp, vec![3], vec![30]).await;
    let s3 = scalar_i64(&p, "SELECT MAX(snapshot_id) FROM ducklake_snapshot").await;
    run_merge(&temp, MergeOptions::default()).await;

    // The merge produced a single partial file spanning s1..=s3.
    let live_files = scalar_i64(
        &p,
        "SELECT COUNT(*) FROM ducklake_data_file WHERE end_snapshot IS NULL",
    )
    .await;
    assert_eq!(live_files, 1, "merge coalesced to one live file");
    let partial_max = opt_i64(
        &p,
        "SELECT partial_max FROM ducklake_data_file WHERE end_snapshot IS NULL",
    )
    .await;
    assert_eq!(partial_max, Some(s3), "merged file records partial_max");

    // CDC through the sqlite provider.
    let provider = Arc::new(SqliteMetadataProvider::new(&ro_url(&temp)).await.unwrap());
    let catalog =
        DuckLakeCatalog::new(SqliteMetadataProvider::new(&ro_url(&temp)).await.unwrap()).unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));
    datafusion_ducklake::register_ducklake_functions(&ctx, provider);

    let changes = |a: i64, b: i64| {
        let ctx = ctx.clone();
        async move {
            let batches = ctx
                .sql(&format!(
                    "SELECT snapshot_id, rowid, change_type, id, val \
                     FROM ducklake_table_changes('main.t', {a}, {b}) \
                     ORDER BY snapshot_id"
                ))
                .await
                .unwrap()
                .collect()
                .await
                .unwrap();
            let mut rows: Vec<(i64, i64, String, i32)> = Vec::new();
            for batch in &batches {
                let snaps = batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap();
                let rowids = batch
                    .column(1)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap();
                let cts = batch
                    .column(2)
                    .as_any()
                    .downcast_ref::<arrow::array::StringArray>()
                    .unwrap();
                let ids = batch
                    .column(3)
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .unwrap();
                for r in 0..batch.num_rows() {
                    rows.push((
                        snaps.value(r),
                        rowids.value(r),
                        cts.value(r).to_string(),
                        ids.value(r),
                    ));
                }
            }
            rows
        }
    };

    // Full range: every row at its origin snapshot with its original rowid.
    assert_eq!(
        changes(0, 1000).await,
        vec![
            (s1, 0, "insert".to_string(), 1),
            (s2, 1, "insert".to_string(), 2),
            (s3, 2, "insert".to_string(), 3),
        ],
        "merged rows attributed to origin snapshots"
    );
    // Windows reachable only via partial_max (start past the merged file's
    // begin_snapshot).
    assert_eq!(
        changes(s2, s2).await,
        vec![(s2, 1, "insert".to_string(), 2)],
        "single-snapshot window inside the merged file"
    );
    assert_eq!(
        changes(s3, 1000).await,
        vec![(s3, 2, "insert".to_string(), 3)],
        "suffix window inside the merged file"
    );
    // The merge snapshot itself emits nothing.
    assert_eq!(
        changes(s3 + 1, s3 + 1).await,
        vec![],
        "merge emits no CDC events"
    );
}

/// Compaction writes with the table's configured parquet settings, not the
/// format defaults.
///
/// Compaction re-encodes data that already exists, so leaving the writer at its
/// defaults does not merely fail to optimise — it *undoes* what the data was
/// written with. A table written LZ4 with a bounded row group comes back
/// uncompressed and, below a million rows, as one row group nothing can prune
/// into.
///
/// Official DuckLake has no such gap: its compaction builds copy options through
/// the same `DuckLakeInsert::GetCopyOptions` inserts use, so a merged file
/// inherits the catalog's configured `parquet_compression`.
#[tokio::test(flavor = "multi_thread")]
async fn merge_inherits_the_tables_write_options() {
    use datafusion_ducklake::DuckLakeWriteOptions;
    use parquet::basic::Compression;

    let temp = TempDir::new().unwrap();
    seed(&temp, vec![1, 2, 3], vec![10, 20, 30]).await;
    append(&temp, vec![4, 5, 6], vec![40, 50, 60]).await;

    let result = run_merge_with_write_options(
        &temp,
        DuckLakeWriteOptions {
            compression: Some(Compression::LZ4),
            // Two rows per group over six: enough that a single group is
            // unmistakably the writer default rather than a coincidence.
            max_row_group_rows: Some(2),
            ..Default::default()
        },
        MergeOptions::default(),
    )
    .await;
    assert_eq!(result.files_processed, 2);
    assert_eq!(result.files_created, 1);

    let p = pool(&temp).await;
    let live_path: String =
        sqlx::query_scalar("SELECT path FROM ducklake_data_file WHERE end_snapshot IS NULL")
            .fetch_one(&p)
            .await
            .unwrap();
    let (codecs, row_groups) = live_file_parquet_facts(&temp, &live_path);
    assert!(
        row_groups > 1,
        "row group cap ignored: {row_groups} group(s) for 6 rows at 2 per group"
    );
    assert!(
        codecs.iter().all(|c| *c == Compression::LZ4),
        "merged file written {codecs:?}, not the table's configured codec"
    );
}

/// The same for `rewrite_data_files`, which is a second writer construction and
/// would otherwise be free to drift from the merge one.
#[tokio::test(flavor = "multi_thread")]
async fn rewrite_inherits_the_tables_write_options() {
    use datafusion_ducklake::DuckLakeWriteOptions;
    use parquet::basic::Compression;

    let temp = TempDir::new().unwrap();
    seed(&temp, vec![1, 2, 3, 4, 5, 6], vec![10, 20, 30, 40, 50, 60]).await;

    let p = pool(&temp).await;
    let file_id: i64 = sqlx::query_scalar(
        "SELECT data_file_id FROM ducklake_data_file WHERE end_snapshot IS NULL",
    )
    .fetch_one(&p)
    .await
    .unwrap();

    // Naming the file rewrites it regardless of its delete fraction, which is
    // what makes this a rewrite rather than a merge.
    let result = run_rewrite_with_write_options(
        &temp,
        DuckLakeWriteOptions {
            compression: Some(Compression::LZ4),
            max_row_group_rows: Some(2),
            ..Default::default()
        },
        RewriteOptions {
            data_file_ids: Some(vec![file_id]),
            ..Default::default()
        },
    )
    .await;
    assert_eq!(result.files_created, 1, "the named file is rewritten");

    let live_path: String =
        sqlx::query_scalar("SELECT path FROM ducklake_data_file WHERE end_snapshot IS NULL")
            .fetch_one(&p)
            .await
            .unwrap();
    let (codecs, row_groups) = live_file_parquet_facts(&temp, &live_path);
    assert!(
        row_groups > 1,
        "row group cap ignored on rewrite: {row_groups} group(s) for 6 rows"
    );
    assert!(
        codecs.iter().all(|c| *c == Compression::LZ4),
        "rewritten file written {codecs:?}, not the table's configured codec"
    );
}

/// The recorded `partial_max` must be what the merge WROTE, not what the catalog
/// claimed its sources held.
///
/// A source can physically embed origins above the `partial_max` its catalog row
/// reports: providers substitute NULL on catalogs predating the column, the MySQL
/// provider hardcodes it, and the migration that added it NULL-filled every
/// existing row. Deriving the output's bound from those rows records a maximum
/// below one the output physically contains — and `needs_snapshot_filter` only
/// installs the row filter below `partial_max`, so the unfiltered rows are served
/// at snapshots before they existed. The sources are retired in the same commit,
/// so the true bound is gone with them.
///
/// The shape needs the understating source to hold the highest origin of the bin,
/// which is why the first merge deliberately leaves the seed file alone.
#[tokio::test(flavor = "multi_thread")]
async fn a_merge_records_the_origin_it_wrote_not_the_one_the_catalog_claimed() {
    let temp = TempDir::new().unwrap();
    // A big seed so it can be held out of the first merge by size alone.
    let ids: Vec<i32> = (0..4000).collect();
    let vals: Vec<i32> = (0..4000).map(|i| i * 2).collect();
    seed(&temp, ids, vals).await; // snapshot 1, file A
    append(&temp, vec![9001], vec![1]).await; // snapshot 2, file B
    append(&temp, vec![9002], vec![2]).await; // snapshot 3, file C

    let p = pool(&temp).await;
    let small = scalar_i64(
        &p,
        "SELECT MIN(file_size_bytes) FROM ducklake_data_file WHERE end_snapshot IS NULL",
    )
    .await;
    let big = scalar_i64(
        &p,
        "SELECT MAX(file_size_bytes) FROM ducklake_data_file WHERE end_snapshot IS NULL",
    )
    .await;
    assert!(big > small * 2, "the seed must be separable by size");

    // Merge B+C only: A is at or above the target, so it is left alone.
    run_merge(
        &temp,
        MergeOptions {
            target_file_size: big as u64,
            max_merged_files: 1024,
            min_file_size: 0,
        },
    )
    .await;

    // P now embeds origins {2, 3}. Make its catalog row understate that, the way
    // a NULL-filling migration or a provider that cannot read the column does.
    let writable = SqlitePool::connect(&db_url(&temp)).await.unwrap();
    sqlx::query(AssertSqlSafe(
        "UPDATE ducklake_data_file SET partial_max = NULL          WHERE end_snapshot IS NULL AND partial_max IS NOT NULL",
    ))
    .execute(&writable)
    .await
    .unwrap();

    // A (begin 1) + P (begin 2, catalog says not partial, really holds 3).
    let result = run_merge(
        &temp,
        MergeOptions {
            target_file_size: 1 << 30,
            max_merged_files: 1024,
            min_file_size: 0,
        },
    )
    .await;
    assert_eq!(result.files_processed, 2, "A and P merge together");

    let recorded = opt_i64(
        &p,
        "SELECT partial_max FROM ducklake_data_file WHERE end_snapshot IS NULL",
    )
    .await
    .expect("the output is partial: it embeds per-row origins");

    let merged_path: String = sqlx::query(AssertSqlSafe(
        "SELECT path FROM ducklake_data_file WHERE end_snapshot IS NULL",
    ))
    .fetch_one(&p)
    .await
    .unwrap()
    .try_get::<String, _>(0)
    .unwrap();
    let written_max = file_lineage(&temp, &merged_path)
        .into_iter()
        .filter_map(|(_, origin)| origin)
        .max()
        .expect("the output embeds per-row origins");

    assert_eq!(
        recorded, written_max,
        "the catalog understated P (partial_max NULL, begin 2) while it physically \
         held origin {written_max}; the output must record what it wrote, or a read \
         below {written_max} skips the row filter and serves rows that did not exist yet"
    );

    // The bookkeeping is only a proxy. What the bound protects is the read: at a
    // snapshot inside the gap the catalog claimed, the row that arrived later
    // must not be visible. `needs_snapshot_filter` is `snapshot_id < partial_max`,
    // so an understated bound is exactly the case where no filter is installed.
    let at_two = read_rows_at(&temp, 2).await;
    assert!(
        !at_two.iter().any(|(id, _)| *id == 9002),
        "the row appended at snapshot 3 must not be visible at snapshot 2: {at_two:?}"
    );
    assert!(
        at_two.iter().any(|(id, _)| *id == 9001),
        "the row appended at snapshot 2 must still be visible at snapshot 2: {at_two:?}"
    );
}

/// Compaction under a scan split across partitions.
///
/// `merge_adjacent_files` builds one source scan per file in the bin, and each
/// of those now asks for its own byte-range split — so the fan-out is the
/// product, and this is the path where a positional error would be widest.
/// Nothing else in the suite runs compaction with splitting on: the other
/// fixtures are far below `repartition_file_min_size`, so their plans stay
/// single-partition however the config is set.
///
/// Every rowid must survive the merge, and every row must be present exactly
/// once.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn merge_preserves_lineage_when_sources_are_split_across_partitions() {
    let temp = TempDir::new().unwrap();

    // Files large enough that DataFusion will actually split them, and written
    // in small row groups so a split has boundaries to land on.
    const PER_FILE: i32 = 60_000;
    let ids: Vec<i32> = (0..PER_FILE).collect();
    let vals: Vec<i32> = ids.iter().map(|i| i * 2).collect();
    seed(&temp, ids.clone(), vals.clone()).await;
    let ids2: Vec<i32> = (PER_FILE..PER_FILE * 2).collect();
    let vals2: Vec<i32> = ids2.iter().map(|i| i * 2).collect();
    append(&temp, ids2.clone(), vals2.clone()).await;

    let mut cfg = datafusion::config::ConfigOptions::new();
    cfg.execution.target_partitions = 8;
    cfg.optimizer.repartition_file_scans = true;
    cfg.optimizer.repartition_file_min_size = 1;
    let ctx = SessionContext::new_with_config(datafusion::prelude::SessionConfig::from(cfg));

    let result = with_writable_table_context(&temp, ctx, |table, state| async move {
        table
            .merge_adjacent_files(&state, MergeOptions::default())
            .await
            .unwrap()
    })
    .await;
    assert_eq!(
        result.files_processed, 2,
        "both source files must be merged"
    );
    assert!(
        result.files_created > 0,
        "the merge must actually write output"
    );
    assert_eq!(result.rows_written, i64::from(PER_FILE) * 2);

    // Every row survives exactly once, with its value intact.
    let mut rows = read_rows(&temp).await;
    rows.sort_unstable();
    let mut expected: Vec<(i32, i32)> =
        ids.iter().chain(ids2.iter()).map(|&i| (i, i * 2)).collect();
    expected.sort_unstable();
    assert_eq!(
        rows, expected,
        "merge under a split scan must not lose or duplicate rows"
    );

    // And rowid lineage is intact: each id keeps a distinct rowid.
    let id_rowid = read_id_rowid(&temp).await;
    assert_eq!(id_rowid.len(), (PER_FILE * 2) as usize);
    let distinct: std::collections::HashSet<i64> = id_rowid.iter().map(|(_, r)| *r).collect();
    assert_eq!(
        distinct.len(),
        id_rowid.len(),
        "every merged row must keep a distinct rowid"
    );
}
