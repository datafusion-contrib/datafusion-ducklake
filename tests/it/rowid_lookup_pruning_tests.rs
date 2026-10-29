//! `DuckLakeTable::read_columns_by_rowid` through the states a table reaches
//! in use, at a size where files hold several row groups and pages and a read
//! returns several batches: merged unsorted and sorted, rewritten after
//! deletes, with inlined rows, read at an older snapshot, after a column is
//! added or renamed, partitioned, and updated. Every read is checked value for
//! value against a full scan of the table at the same snapshot. A lookup in a
//! file that embeds its rows' `rowid`s must decode only the pages that may
//! hold the ids asked for, and a caller's memory reservation must bound what
//! it holds.

#![cfg(feature = "write-sqlite")]

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use arrow::array::{Array, RecordBatch};
use arrow::datatypes::DataType;
use datafusion::execution::memory_pool::{
    GreedyMemoryPool, MemoryConsumer, MemoryPool, MemoryReservation,
};
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeError, DuckLakeTable, DuckLakeWriteOptions, MergeOptions,
    MetadataProvider, MetadataWriter, RewriteOptions, RowIdReadOptions, SqliteMetadataProvider,
    SqliteMetadataWriter, WriteMode, execute_ducklake_sql,
};
use tempfile::TempDir;

use crate::resolve_positions_projection_tests::ByteCountingStore;

/// Rows per page and per write batch, rows per row group, and rows per
/// insert: each file holds several row groups of many pages, and a scan of
/// one returns several batches.
const PAGE_ROWS: usize = 1_024;
const ROW_GROUP_ROWS: usize = 8_192;
const INSERT_ROWS: i64 = 20_000;

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

fn write_options(inline_limit: usize) -> DuckLakeWriteOptions {
    let mut options = DuckLakeWriteOptions::default().with_data_inlining_row_limit(inline_limit);
    options.max_row_group_rows = Some(ROW_GROUP_ROWS);
    options.data_page_row_count_limit = Some(PAGE_ROWS);
    options.write_batch_size = Some(PAGE_ROWS);
    options
}

struct Lake {
    _temp: TempDir,
    url: String,
    provider: Arc<dyn MetadataProvider>,
    options: DuckLakeWriteOptions,
}

impl Lake {
    /// An empty catalog with `main.t (id BIGINT, k BIGINT, v VARCHAR, p
    /// BIGINT)`,
    /// inlining inserts of at most `inline_limit` rows.
    async fn new(inline_limit: usize) -> Self {
        let temp = TempDir::new().unwrap();
        let data = temp.path().join("data");
        std::fs::create_dir_all(&data).unwrap();
        let url = format!("sqlite:{}?mode=rwc", temp.path().join("lake.db").display());
        let writer = SqliteMetadataWriter::new_with_init(&url).await.unwrap();
        writer.set_data_path(data.to_str().unwrap()).unwrap();
        let provider: Arc<dyn MetadataProvider> =
            Arc::new(SqliteMetadataProvider::new(&url).await.unwrap());
        let lake = Self {
            _temp: temp,
            url,
            provider,
            options: write_options(inline_limit),
        };
        // `main.t`, and the schema with it, as `CREATE TABLE` makes them.
        let columns: Vec<_> = [
            ("id", DataType::Int64),
            ("k", DataType::Int64),
            ("v", DataType::Utf8),
            ("p", DataType::Int64),
        ]
        .iter()
        .map(|(name, data_type)| {
            datafusion_ducklake::metadata_writer::ColumnDef::from_arrow(*name, data_type, true)
                .unwrap()
        })
        .collect();
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
                &setup.field_ids,
            )
            .unwrap();
        lake
    }

    async fn catalog(&self) -> DuckLakeCatalog {
        DuckLakeCatalog::with_writer(
            Arc::clone(&self.provider),
            Arc::new(SqliteMetadataWriter::new(&self.url).await.unwrap()),
        )
        .unwrap()
        .with_write_options(self.options.clone())
    }

    async fn sql(&self, statement: &str) {
        let ctx = SessionContext::new();
        let catalog = self.catalog().await;
        DuckLakeCatalog::register(&ctx, "lake", self.catalog().await);
        Box::pin(execute_ducklake_sql(&ctx, &catalog, statement))
            .await
            .unwrap_or_else(|e| panic!("`{statement}`: {e}"))
            .collect()
            .await
            .unwrap_or_else(|e| panic!("`{statement}`: {e}"));
    }

    /// Insert ids `from..from + rows`, with `k` scattering them, `v` naming
    /// them and `p` their remainder by 4.
    async fn insert(&self, from: i64, rows: i64) {
        self.sql(&format!(
            "INSERT INTO lake.main.t SELECT value AS id, (value * 7919) % 10007 AS k, \
             concat('v-', CAST(value AS VARCHAR)) AS v, value % 4 AS p \
             FROM generate_series({from}, {})",
            from + rows - 1
        ))
        .await;
    }

    fn head(&self) -> i64 {
        self.provider.get_current_snapshot().unwrap()
    }

    /// The writable table at the head, and its session.
    async fn writable(&self) -> (SessionContext, DuckLakeTable) {
        let ctx = SessionContext::new();
        DuckLakeCatalog::register(&ctx, "lake", self.catalog().await);
        let table = downcast(&ctx).await;
        (ctx, table)
    }

    async fn merge(&self) {
        let merged = self.merge_any().await;
        assert!(
            merged.files_created > 0,
            "the merge wrote no file: {merged:?}"
        );
    }

    /// Merge what can be merged.
    async fn merge_any(&self) -> datafusion_ducklake::CompactionResult {
        let (ctx, table) = self.writable().await;
        table
            .merge_adjacent_files(
                &ctx.state(),
                MergeOptions {
                    target_file_size: u64::MAX / 2,
                    ..MergeOptions::default()
                },
            )
            .await
            .unwrap()
    }

    async fn rewrite(&self) {
        let (ctx, table) = self.writable().await;
        let rewritten = table
            .rewrite_data_files(
                &ctx.state(),
                RewriteOptions {
                    delete_threshold: 0.0,
                    data_file_ids: None,
                },
            )
            .await
            .unwrap();
        assert!(rewritten.files_created > 0, "the rewrite wrote no file");
    }

    /// A session reading at `snapshot` through a byte-counting store, and the
    /// table with its `rowid`.
    async fn reader(&self, snapshot: i64) -> Reader {
        reader(&self.provider, snapshot).await
    }
}

/// A session reading `provider`'s catalog at `snapshot` through a
/// byte-counting store, and the table `main.t` with its `rowid`.
async fn reader(provider: &Arc<dyn MetadataProvider>, snapshot: i64) -> Reader {
    {
        let ctx = SessionContext::new_with_config(SessionConfig::new().with_target_partitions(4));
        let store = Arc::new(ByteCountingStore::new());
        ctx.runtime_env().register_object_store(
            &url::Url::parse("file:///").unwrap(),
            Arc::clone(&store) as Arc<dyn object_store::ObjectStore>,
        );
        let catalog = DuckLakeCatalog::with_snapshot(Arc::clone(provider), snapshot)
            .unwrap()
            .with_row_lineage(true);
        ctx.register_catalog("lake", Arc::new(catalog));
        let table = downcast(&ctx).await;
        Reader {
            ctx,
            store,
            table,
        }
    }
}

async fn downcast(ctx: &SessionContext) -> DuckLakeTable {
    let provider = ctx
        .catalog("lake")
        .unwrap()
        .schema("main")
        .unwrap()
        .table("t")
        .await
        .unwrap()
        .unwrap();
    (provider.as_ref() as &dyn std::any::Any)
        .downcast_ref::<DuckLakeTable>()
        .expect("a DuckLakeTable")
        .clone()
}

struct Reader {
    ctx: SessionContext,
    store: Arc<ByteCountingStore>,
    table: DuckLakeTable,
}

/// One cell as text, NULL as `NULL`.
fn cell(array: &dyn Array, row: usize) -> String {
    if array.is_null(row) {
        "NULL".to_string()
    } else {
        arrow::util::display::array_value_to_string(array, row).unwrap()
    }
}

/// `rowid -> every other column` of `batches`.
fn by_rowid(batches: &[RecordBatch]) -> BTreeMap<i64, Vec<String>> {
    let mut out = BTreeMap::new();
    for batch in batches {
        let rowid = batch.schema().index_of("rowid").unwrap();
        for row in 0..batch.num_rows() {
            let id: i64 = cell(batch.column(rowid), row).parse().unwrap();
            let cells = (0..batch.num_columns())
                .filter(|column| *column != rowid)
                .map(|column| cell(batch.column(column), row))
                .collect();
            assert!(out.insert(id, cells).is_none(), "rowid {id} read twice");
        }
    }
    out
}

/// What a lookup of a few ids cost.
#[derive(Debug)]
#[allow(dead_code, reason = "printed")]
struct Cost {
    rows: usize,
    rows_decoded: u64,
    bytes: u64,
}

impl Reader {
    /// `rowid -> every other column` of a full scan.
    async fn truth(&self) -> BTreeMap<i64, Vec<String>> {
        let batches = self
            .ctx
            .sql("SELECT * FROM lake.main.t")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        by_rowid(&batches)
    }

    /// Read every live row by `rowid`, with ids no live row has and repeats
    /// among them, with every column and with each column alone, and check
    /// each value against a full scan. Then read ten ids spread over the
    /// table twice, and return what the second read cost, its footers read.
    async fn check(&self, what: &str) -> Cost {
        let truth = self.truth().await;
        assert!(!truth.is_empty(), "{what}: the table is empty");
        let max = *truth.keys().last().unwrap();
        let mut requested: Vec<i64> = truth.keys().copied().rev().collect();
        let third = requested.len() / 3;
        requested.rotate_left(third);
        let absent: Vec<i64> = (0..=max).filter(|id| !truth.contains_key(id)).collect();
        requested.extend(absent.iter().take(500));
        let repeat = requested[0];
        requested.extend([-1, max + 1, i64::MAX, repeat]);
        let state = self.ctx.state();
        let schema = datafusion::catalog::TableProvider::schema(&self.table);
        let rowid = schema.index_of("rowid").unwrap();

        let read = self
            .table
            .read_columns_by_rowid(&state, &requested, None)
            .await
            .unwrap_or_else(|e| panic!("{what}: {e}"));
        let mut seen = HashSet::new();
        let order: Vec<i64> = requested
            .iter()
            .copied()
            .filter(|id| truth.contains_key(id) && seen.insert(*id))
            .collect();
        assert_eq!(read.row_ids.len(), order.len(), "{what}: rows found");
        assert_eq!(read.row_ids, order, "{what}: rows in the order requested");
        assert_eq!(read.batch.schema().fields(), schema.fields(), "{what}");
        for (row, id) in read.row_ids.iter().enumerate() {
            assert_eq!(cell(read.batch.column(rowid), row), id.to_string());
            let cells: Vec<String> = (0..read.batch.num_columns())
                .filter(|column| *column != rowid)
                .map(|column| cell(read.batch.column(column), row))
                .collect();
            assert_eq!(cells, truth[id], "{what}: rowid {id}");
        }
        for column in 0..schema.fields().len() {
            let one = self
                .table
                .read_columns_by_rowid(&state, &requested, Some(&[column]))
                .await
                .unwrap();
            assert_eq!(one.row_ids, order, "{what}: column {column}");
            for row in 0..one.batch.num_rows() {
                assert_eq!(
                    cell(one.batch.column(0), row),
                    cell(read.batch.column(column), row),
                    "{what}: column {column} alone, rowid {}",
                    one.row_ids[row]
                );
            }
        }

        let ids: Vec<i64> = truth.keys().copied().collect();
        let sample: Vec<i64> = (0..10).map(|i| ids[i * (ids.len() - 1) / 9]).collect();
        self.table
            .read_columns_by_rowid(&state, &sample, None)
            .await
            .unwrap();
        self.store.take();
        let started = Instant::now();
        let few = self
            .table
            .read_columns_by_rowid(&state, &sample, None)
            .await
            .unwrap();
        let elapsed = started.elapsed();
        assert_eq!(few.row_ids, sample, "{what}: the sample");
        let cost = Cost {
            rows: truth.len(),
            rows_decoded: few.rows_decoded,
            bytes: self.store.take(),
        };
        eprintln!("{what}: 10 of {cost:?} in {elapsed:?}");
        cost
    }
}

/// The live data files of `t` at `snapshot`, and how many embed row ids.
async fn files(lake: &Lake, snapshot: i64) -> (usize, usize) {
    let reader = lake.reader(snapshot).await;
    let state = reader.ctx.state();
    let files = reader.table.files().unwrap();
    let mut embedding = 0;
    for file in &files {
        if reader
            .table
            .file_has_embedded_rowid(&state, &file.file)
            .await
            .unwrap()
        {
            embedding += 1;
        }
    }
    (files.len(), embedding)
}

// ---------------------------------------------------------------------------
// Table states
// ---------------------------------------------------------------------------

/// Four inserts merged into one file in insert order: every row of the file
/// embeds its `rowid`, in ascending order. A lookup of ten ids decodes only
/// the pages holding them, of 80,000 rows. Guards against a lookup that
/// decodes a merged file's whole `rowid` column, and against a pruned lookup
/// that misses or misreads a row.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn merged_in_insert_order() {
    let lake = Lake::new(0).await;
    for insert in 0..4 {
        lake.insert(insert * INSERT_ROWS, INSERT_ROWS).await;
    }
    let before = lake.head();
    lake.reader(before).await.check("before the merge").await;
    lake.merge().await;
    assert_eq!(files(&lake, lake.head()).await, (1, 1));
    let cost = lake.reader(lake.head()).await.check("merged").await;
    // Each id's page of the `rowid` column, then its page of each read
    // column: at most two pages per id.
    assert!(
        cost.rows_decoded <= 10 * 2 * PAGE_ROWS as u64,
        "decoded {} rows for 10 ids: {cost:?}",
        cost.rows_decoded
    );
    // The earlier snapshot reads the merged file as a partial file.
    lake.reader(before)
        .await
        .check("before the merge, read after")
        .await;
}

/// Ten inserts of small pages merged into one file of 200,000 rows: three
/// ids are found by decoding a few pages, a few hundred rows, where reading
/// the file's `rowid` column decodes all 200,000. Guards against a lookup
/// whose work grows with the file rather than with the ids asked for.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_lookup_decodes_only_the_pages_it_needs() {
    const SMALL_PAGE_ROWS: usize = 128;
    let mut lake = Lake::new(0).await;
    lake.options.data_page_row_count_limit = Some(SMALL_PAGE_ROWS);
    lake.options.write_batch_size = Some(SMALL_PAGE_ROWS);
    for insert in 0..10 {
        lake.insert(insert * INSERT_ROWS, INSERT_ROWS).await;
    }
    lake.merge().await;
    assert_eq!(files(&lake, lake.head()).await, (1, 1));
    let reader = lake.reader(lake.head()).await;
    let state = reader.ctx.state();
    let wanted = [7, 99_999, 199_998];
    let read = reader
        .table
        .read_columns_by_rowid(&state, &wanted, None)
        .await
        .unwrap();
    assert_eq!(read.row_ids, wanted.to_vec());
    let truth = reader.truth().await;
    for (row, id) in wanted.iter().enumerate() {
        let cells: Vec<String> = (0..4)
            .map(|column| cell(read.batch.column(column), row))
            .collect();
        assert_eq!(cells, truth[id], "rowid {id}");
    }
    // A page of the `rowid` column per id, then the id's row of each
    // column.
    let most = (wanted.len() * SMALL_PAGE_ROWS + wanted.len()) as u64;
    assert!(
        read.rows_decoded <= most,
        "decoded {} rows of 200000 for {} ids",
        read.rows_decoded,
        wanted.len()
    );
}

/// Inserts merged under a sort order on a column that scatters the ids: the
/// merged file holds the rows in `k` order, so each page spans most ids and
/// little can be pruned, but every row reads back right. Guards against
/// pruning that assumes a merged file's rows are in `rowid` order.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn merged_under_a_sort_order() {
    let lake = Lake::new(0).await;
    lake.sql("ALTER TABLE lake.main.t SET SORTED BY (k)").await;
    for insert in 0..3 {
        lake.insert(insert * INSERT_ROWS, INSERT_ROWS).await;
    }
    lake.merge().await;
    assert_eq!(files(&lake, lake.head()).await, (1, 1));
    lake.reader(lake.head())
        .await
        .check("merged sorted by k")
        .await;
}

/// Deletes through delete files, and a rewrite that drops the deleted rows:
/// the rewritten file's ids have holes. Both states read right, the deleted
/// rows missing. Guards against a pruned lookup that loses track of a row's
/// position, which the delete files name.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn deleted_then_rewritten() {
    let lake = Lake::new(0).await;
    for insert in 0..3 {
        lake.insert(insert * INSERT_ROWS, INSERT_ROWS).await;
    }
    lake.merge().await;
    lake.sql("DELETE FROM lake.main.t WHERE id % 3 = 0").await;
    lake.reader(lake.head())
        .await
        .check("merged, deleted")
        .await;
    let deleted = lake.head();
    lake.rewrite().await;
    let cost = lake.reader(lake.head()).await.check("rewritten").await;
    assert_eq!(cost.rows, 40_000);
    assert!(
        cost.rows_decoded <= 10 * 2 * PAGE_ROWS as u64,
        "decoded {} rows for 10 ids",
        cost.rows_decoded
    );
    lake.sql("DELETE FROM lake.main.t WHERE id % 5 = 0").await;
    lake.reader(lake.head())
        .await
        .check("rewritten, deleted")
        .await;
    lake.reader(deleted)
        .await
        .check("deleted, read after")
        .await;
}

/// Inlined rows beside a merged file, an inlined row deleted and an inlined
/// deletion of a file row: rows are found wherever they are. Guards against
/// a lookup that prunes inlined rows by a file's bounds.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn inlined_beside_merged_files() {
    let lake = Lake::new(50).await;
    for insert in 0..2 {
        lake.insert(insert * INSERT_ROWS, INSERT_ROWS).await;
    }
    lake.merge().await;
    lake.insert(1_000_000, 20).await;
    lake.insert(1_000_100, 20).await;
    lake.sql("DELETE FROM lake.main.t WHERE id = 1000005").await;
    lake.sql("DELETE FROM lake.main.t WHERE id = 17").await;
    let reader = lake.reader(lake.head()).await;
    reader.check("inlined beside a merge").await;

    // The inlined rows alone are charged too: a pool too small for them
    // refuses the read, and one that holds them is left holding the result.
    let truth = reader.truth().await;
    let inlined: Vec<i64> = truth
        .iter()
        .filter(|(_, row)| row[0].parse::<i64>().unwrap() >= 1_000_000)
        .map(|(rowid, _)| *rowid)
        .collect();
    assert_eq!(inlined.len(), 39);
    let state = reader.ctx.state();
    let (pool, small) = reservation(64);
    let error = reader
        .table
        .read_columns_by_rowid_with(
            &state,
            &inlined,
            None,
            RowIdReadOptions::default().with_reservation(&small),
        )
        .await
        .unwrap_err();
    assert!(
        matches!(
            &error,
            DuckLakeError::DataFusion(datafusion::error::DataFusionError::ResourcesExhausted(_))
        ),
        "{error:?}"
    );
    assert_eq!(pool.reserved(), 0);
    let (_pool, held) = reservation(1 << 20);
    let read = reader
        .table
        .read_columns_by_rowid_with(
            &state,
            &inlined,
            None,
            RowIdReadOptions::default().with_reservation(&held),
        )
        .await
        .unwrap();
    assert_eq!(read.row_ids, inlined);
    assert_eq!(read.rows_decoded, 0, "no data file is read");
    assert_eq!(held.size(), read.batch.get_array_memory_size());
}

/// Updated rows: each new row version is written to a file that embeds the
/// row's original `rowid`, below the ids of the rows around it, and the old
/// version deleted. Then the files are rewritten without the old versions and
/// merged. Guards against pruning a file by its catalog `row_id_start`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn updated_then_rewritten() {
    let lake = Lake::new(0).await;
    for insert in 0..3 {
        lake.insert(insert * INSERT_ROWS, INSERT_ROWS).await;
    }
    lake.sql("UPDATE lake.main.t SET v = concat('u-', v) WHERE id % 7 = 0")
        .await;
    lake.reader(lake.head()).await.check("updated").await;
    let updated = lake.head();
    lake.rewrite().await;
    lake.merge_any().await;
    let cost = lake
        .reader(lake.head())
        .await
        .check("updated, rewritten")
        .await;
    assert_eq!(cost.rows, 60_000);
    lake.reader(updated)
        .await
        .check("updated, read after")
        .await;
}

/// A table partitioned on `p = id % 4`: each insert writes one file per
/// partition, whose ids interleave with the other partitions', and a merge
/// merges within a partition. Guards against a lookup that misses rows a
/// file's bounds span but its rows skip.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn partitioned() {
    let lake = Lake::new(0).await;
    lake.sql("ALTER TABLE lake.main.t SET PARTITIONED BY (p)")
        .await;
    for insert in 0..3 {
        lake.insert(insert * INSERT_ROWS, INSERT_ROWS).await;
    }
    let (files_before, _) = files(&lake, lake.head()).await;
    assert!(files_before >= 12, "{files_before} files");
    lake.reader(lake.head()).await.check("partitioned").await;
    lake.merge().await;
    lake.reader(lake.head())
        .await
        .check("partitioned, merged")
        .await;
}

/// Tables official DuckDB wrote, in files of several row groups: schema
/// evolution, merges, deletes and a rewrite.
#[cfg(feature = "metadata-duckdb")]
mod official_history {
    use super::*;
    use crate::inlined_mutation_differential_tests::{Metadata, official};

    /// A column added with a default and a column renamed after the files
    /// were written, rows written at the new schema, merged, deleted and
    /// rewritten, each snapshot read after every phase: the old rows read the
    /// default and their values under the new name. Guards against a pruned
    /// read that resolves columns by name or loses defaults, and against
    /// pruning by statistics that official's files record differently.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn columns_added_and_renamed() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("official.db");
        let data = dir.path().join("data");
        std::fs::create_dir_all(&data).unwrap();
        let metadata = Metadata::Sqlite(path.clone());
        let insert = |from: i64, to: i64| {
            format!(
                "INSERT INTO lake.main.t SELECT i, (i * 7919) % 10007, 'v-' || i \
                 FROM range({from}, {to}) r(i)"
            )
        };
        let phases: Vec<Vec<String>> = vec![
            vec![
                "CREATE TABLE lake.main.t (id BIGINT, k BIGINT, v VARCHAR)".to_string(),
                "CALL lake.set_option('data_inlining_row_limit', 0)".to_string(),
                "CALL lake.set_option('parquet_row_group_size', 8192)".to_string(),
                insert(0, 20_000),
            ],
            vec![insert(20_000, 40_000)],
            vec!["CALL ducklake_merge_adjacent_files('lake')".to_string()],
            vec![
                "ALTER TABLE lake.main.t ADD COLUMN n BIGINT DEFAULT 7".to_string(),
                "ALTER TABLE lake.main.t RENAME COLUMN v TO w".to_string(),
                "INSERT INTO lake.main.t SELECT i, i % 11, 'w-' || i, i * 2 \
                 FROM range(40000, 60000) r(i)"
                    .to_string(),
            ],
            vec!["CALL ducklake_merge_adjacent_files('lake')".to_string()],
            vec!["DELETE FROM lake.main.t WHERE id % 9 = 0".to_string()],
            vec![
                "CALL ducklake_rewrite_data_files('lake', 't', delete_threshold => 0)".to_string(),
            ],
        ];
        let old_row = ["5", "9574", "v-5", "7"];
        let mut snapshots = Vec::new();
        for (phase, statements) in phases.iter().enumerate() {
            official(&metadata, &data, statements);
            let provider: Arc<dyn MetadataProvider> = Arc::new(
                SqliteMetadataProvider::new(&format!("sqlite:{}", path.display()))
                    .await
                    .unwrap(),
            );
            snapshots.push(provider.get_current_snapshot().unwrap());
            for (at, &snapshot) in snapshots.iter().enumerate() {
                let reader = reader(&provider, snapshot).await;
                if at >= 3 {
                    let truth = reader.truth().await;
                    assert!(
                        truth.values().any(|row| row == &old_row),
                        "an old row reads the default at snapshot {snapshot}"
                    );
                }
                let cost = reader
                    .check(&format!("official, phase {phase}, snapshot {snapshot}"))
                    .await;
                if at == 0 {
                    // The first insert's rows: after the merge, the first
                    // rows of a file merged with the second insert's, read
                    // at the earlier snapshot. Ten rows of the file, and
                    // their snapshots, are read.
                    assert!(cost.rows_decoded <= 20, "{cost:?}");
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Memory
// ---------------------------------------------------------------------------

/// A reservation from a pool of `limit` bytes.
fn reservation(limit: usize) -> (Arc<dyn MemoryPool>, MemoryReservation) {
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(limit));
    let reservation = MemoryConsumer::new("rowid lookup").register(&pool);
    (pool, reservation)
}

/// With a reservation, a read the pool can hold returns the same rows and
/// leaves its batch charged, and a read it cannot hold fails with the pool's
/// error and leaves nothing charged. Guards against a read that allocates a
/// payload it never charges.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_reservation_bounds_the_read() {
    let lake = Lake::new(0).await;
    for insert in 0..2 {
        lake.insert(insert * INSERT_ROWS, INSERT_ROWS).await;
    }
    lake.merge().await;
    let reader = lake.reader(lake.head()).await;
    let state = reader.ctx.state();
    let all: Vec<i64> = (0..2 * INSERT_ROWS).collect();
    let unbounded = reader
        .table
        .read_columns_by_rowid(&state, &all, None)
        .await
        .unwrap();
    assert_eq!(unbounded.row_ids, all);

    let (pool, held) = reservation(1 << 30);
    let read = reader
        .table
        .read_columns_by_rowid_with(
            &state,
            &all,
            None,
            RowIdReadOptions::default().with_reservation(&held),
        )
        .await
        .unwrap();
    assert_eq!(read.batch, unbounded.batch);
    assert_eq!(held.size(), read.batch.get_array_memory_size());
    assert_eq!(pool.reserved(), held.size());
    held.free();

    let limit = unbounded.batch.get_array_memory_size() / 2;
    let (pool, small) = reservation(limit);
    let error = reader
        .table
        .read_columns_by_rowid_with(
            &state,
            &all,
            None,
            RowIdReadOptions::default().with_reservation(&small),
        )
        .await
        .unwrap_err();
    assert!(
        matches!(
            &error,
            DuckLakeError::DataFusion(datafusion::error::DataFusionError::ResourcesExhausted(_))
        ),
        "{error:?}"
    );
    assert_eq!(small.size(), 0, "a failed read leaves nothing charged");
    assert_eq!(pool.reserved(), 0);

    // A few rows fit.
    let few = reader
        .table
        .read_columns_by_rowid_with(
            &state,
            &[3, 39_000],
            None,
            RowIdReadOptions::default().with_reservation(&small),
        )
        .await
        .unwrap();
    assert_eq!(few.row_ids, vec![3, 39_000]);
    assert_eq!(small.size(), few.batch.get_array_memory_size());
}

// ---------------------------------------------------------------------------
// Measurement
// ---------------------------------------------------------------------------

/// Ten ids out of one merged file of `ROWIDS_MEASURE_ROWS` rows (default
/// 2,000,000), written with the parquet writer's default pages.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "measurement; run with --ignored --nocapture"]
async fn measure_rowid_lookup_on_a_merged_file() {
    let rows: i64 = std::env::var("ROWIDS_MEASURE_ROWS")
        .ok()
        .map(|rows| rows.parse().unwrap())
        .unwrap_or(2_000_000);
    let mut lake = Lake::new(0).await;
    lake.options = DuckLakeWriteOptions::default().with_data_inlining_row_limit(0);
    let inserts = 10;
    for insert in 0..inserts {
        lake.insert(insert * rows / inserts, rows / inserts).await;
    }
    lake.merge().await;
    let reader = lake.reader(lake.head()).await;
    let file = reader.table.files().unwrap()[0].file.file_size_bytes;
    let wanted: Vec<i64> = (0..10).map(|i| 1_234 + i * (rows / 10)).collect();
    let state = reader.ctx.state();
    let first = reader
        .table
        .read_columns_by_rowid(&state, &wanted, None)
        .await
        .unwrap();
    assert_eq!(first.row_ids, wanted);
    reader.store.take();
    let started = Instant::now();
    let got = reader
        .table
        .read_columns_by_rowid(&state, &wanted, None)
        .await
        .unwrap();
    let elapsed = started.elapsed();
    assert_eq!(got.row_ids, wanted);
    eprintln!(
        "{rows} rows in one merged file of {file} bytes: 10 rowids read {} bytes, decoded {} \
         rows in {elapsed:?}",
        reader.store.take(),
        got.rows_decoded
    );
}
