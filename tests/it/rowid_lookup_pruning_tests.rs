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
    DuckLakeCatalog, DuckLakeError, DuckLakeReadOptions, DuckLakeTable, DuckLakeWriteOptions,
    MergeOptions, MetadataProvider, MetadataWriter, RewriteOptions, RowIdReadOptions,
    SqliteMetadataProvider, SqliteMetadataWriter, WriteMode, execute_ducklake_sql,
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
    reader_with(provider, snapshot, DuckLakeReadOptions::default()).await
}

/// [`reader`] with `options` on the catalog.
async fn reader_with(
    provider: &Arc<dyn MetadataProvider>,
    snapshot: i64,
    options: DuckLakeReadOptions,
) -> Reader {
    {
        let ctx = SessionContext::new_with_config(SessionConfig::new().with_target_partitions(4));
        let store = Arc::new(ByteCountingStore::new());
        ctx.runtime_env().register_object_store(
            &url::Url::parse("file:///").unwrap(),
            Arc::clone(&store) as Arc<dyn object_store::ObjectStore>,
        );
        let catalog = DuckLakeCatalog::with_snapshot(Arc::clone(provider), snapshot)
            .unwrap()
            .with_row_lineage(true)
            .with_read_options(options);
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
                if phase + 1 == phases.len() && at + 1 == snapshots.len() {
                    // The rewritten file official wrote, 8,192-row row groups
                    // and no page index: a single id decodes its row group.
                    let truth = reader.truth().await;
                    let ids: Vec<i64> = truth.keys().copied().collect();
                    let state = reader.ctx.state();
                    for id in [ids[3], ids[ids.len() / 2], ids[ids.len() - 2]] {
                        let one = reader
                            .table
                            .read_columns_by_rowid(&state, &[id], None)
                            .await
                            .unwrap();
                        assert_eq!(one.row_ids, vec![id]);
                        assert!(
                            one.rows_decoded <= 2 * 8_192 + 1,
                            "rowid {id}: decoded {} of {} rows",
                            one.rows_decoded,
                            truth.len()
                        );
                    }
                }
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

/// A merged file with 100,000 of its 200,000 rows deleted, its delete file
/// several pages long: a lookup of two rows near the start reads the delete
/// file's first page, not all of it, and with the table's delete memo a
/// second lookup reads no delete row at all. Both find the same rows, the
/// deleted ones missing. Guards against a lookup whose delete-file read grows
/// with the table's deletions rather than with the rows asked for.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_lookup_reads_only_the_deletes_it_needs() {
    let lake = Lake::new(0).await;
    for insert in 0..10 {
        lake.insert(insert * INSERT_ROWS, INSERT_ROWS).await;
    }
    lake.merge().await;
    // About half the rows, scattered: `k` scatters the ids.
    lake.sql("DELETE FROM lake.main.t WHERE k % 2 = 0").await;
    let reader = lake.reader(lake.head()).await;
    let truth = reader.truth().await;
    let deleted = 200_000 - truth.len() as u64;
    assert!(deleted > 90_000, "{deleted} rows deleted");
    // Two live rows and one deleted one, all among the file's first rows.
    let live: Vec<i64> = truth.keys().copied().take(2).collect();
    let gone = (0..).find(|id| !truth.contains_key(id)).unwrap();
    let wanted = [live[0], gone, live[1]];
    let delete_file_bytes = reader.table.files().unwrap()[0]
        .delete_file
        .as_ref()
        .expect("the merged file has a delete file")
        .file_size_bytes as u64;
    let state = reader.ctx.state();
    // The first read takes the footers, which later reads keep.
    reader
        .table
        .read_columns_by_rowid(&state, &wanted, None)
        .await
        .unwrap();
    reader.store.take();
    let read = reader
        .table
        .read_columns_by_rowid(&state, &wanted, None)
        .await
        .unwrap();
    let bytes = reader.store.take();
    let pruned_bytes = bytes;
    assert_eq!(read.row_ids, live);
    assert!(
        bytes * 2 < delete_file_bytes,
        "read {bytes} bytes, the delete file is {delete_file_bytes}"
    );
    assert!(
        read.rows_decoded * 3 < deleted,
        "decoded {} rows, the delete file holds {deleted}",
        read.rows_decoded
    );

    let memoized = reader_with(&lake.provider, lake.head(), DuckLakeReadOptions::memoized()).await;
    let state = memoized.ctx.state();
    let first = memoized
        .table
        .read_columns_by_rowid(&state, &wanted, None)
        .await
        .unwrap();
    assert_eq!(first.row_ids, live);
    assert!(
        first.rows_decoded >= deleted,
        "the memo reads the file whole"
    );
    memoized.store.take();
    let again = memoized
        .table
        .read_columns_by_rowid(&state, &wanted, None)
        .await
        .unwrap();
    let bytes = memoized.store.take();
    assert_eq!(again.row_ids, live);
    assert_eq!(again.batch, first.batch);
    assert!(
        bytes < pruned_bytes,
        "read {bytes} bytes with the positions kept, {pruned_bytes} without"
    );
    assert!(
        again.rows_decoded <= (wanted.len() * (PAGE_ROWS + 1)) as u64,
        "decoded {} rows with the positions kept",
        again.rows_decoded
    );

    // A memo too small for the file reads as without one, on every lookup,
    // rather than reading the file whole each time to keep nothing.
    let tiny = reader_with(
        &lake.provider,
        lake.head(),
        DuckLakeReadOptions::default().with_delete_memo(64 << 10),
    )
    .await;
    let state = tiny.ctx.state();
    for round in 0..2 {
        let over = tiny
            .table
            .read_columns_by_rowid(&state, &wanted, None)
            .await
            .unwrap();
        assert_eq!(over.row_ids, live);
        assert!(
            over.rows_decoded <= read.rows_decoded,
            "round {round}: decoded {} rows through a memo too small, {} without one",
            over.rows_decoded,
            read.rows_decoded
        );
    }

    // A reservation too small for the delete file the memo would keep is
    // refused before that file is read, and nothing stays charged.
    let filling = reader_with(&lake.provider, lake.head(), DuckLakeReadOptions::memoized()).await;
    let state = filling.ctx.state();
    let (pool, small) = reservation(256 << 10);
    let error = filling
        .table
        .read_columns_by_rowid_with(
            &state,
            &wanted,
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
    // Nothing was kept, and so nothing was read whole: the next lookup with
    // room reads the file whole to keep it.
    let (_pool, room) = reservation(1 << 30);
    let kept = filling
        .table
        .read_columns_by_rowid_with(
            &state,
            &wanted,
            None,
            RowIdReadOptions::default().with_reservation(&room),
        )
        .await
        .unwrap();
    assert_eq!(kept.row_ids, live);
    assert!(
        kept.rows_decoded >= deleted,
        "the memo reads the file whole"
    );
    assert_eq!(room.size(), kept.batch.get_array_memory_size());
}

/// A lookup in a file whose delete file is gone from storage fails with the
/// error a scan gives, naming the missing delete file, with and without the
/// delete memo. Guards against the raw storage error from the lookup's own
/// footer read.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_missing_delete_file_is_named() {
    let lake = Lake::new(0).await;
    lake.insert(0, INSERT_ROWS).await;
    lake.sql("DELETE FROM lake.main.t WHERE id % 3 = 0").await;
    let reader = lake.reader(lake.head()).await;
    let name = reader.table.files().unwrap()[0]
        .delete_file
        .as_ref()
        .expect("a delete file")
        .path
        .clone();
    let mut stack = vec![lake._temp.path().join("data")];
    let mut removed = false;
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                stack.push(path);
            } else if path.to_string_lossy().ends_with(name.as_str()) {
                std::fs::remove_file(&path).unwrap();
                removed = true;
            }
        }
    }
    assert!(removed, "removed {name}");
    for options in [DuckLakeReadOptions::default(), DuckLakeReadOptions::memoized()] {
        let reader = reader_with(&lake.provider, lake.head(), options).await;
        let error = reader
            .table
            .read_columns_by_rowid(&reader.ctx.state(), &[1, 2], None)
            .await
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("Delete file") && error.contains("not found"),
            "{error}"
        );
    }
}

// ---------------------------------------------------------------------------
// Read ahead
// ---------------------------------------------------------------------------

/// Two inserts of 512 rows merged under a sort order that scatters the ids:
/// the merged file is one page holding ids 0..1023 out of order. Its bounds
/// span exactly its rows, so the read ahead guesses each id at `id`, and
/// nearly every guess misses. Every row still reads right, and the read
/// leaves only its batch charged. Guards against a read ahead served on its
/// page statistics, which cannot show the rows are in order.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_permuted_page_misguesses_and_reads_right() {
    let lake = Lake::new(0).await;
    lake.sql("ALTER TABLE lake.main.t SET SORTED BY (k)").await;
    lake.insert(0, 512).await;
    lake.insert(512, 512).await;
    lake.merge().await;
    assert_eq!(files(&lake, lake.head()).await, (1, 1));
    let reader = lake.reader(lake.head()).await;
    let in_file_order: Vec<i64> = by_rowid_order(&reader).await;
    assert_eq!(in_file_order.len(), 1024);
    assert_ne!(
        in_file_order,
        (0..1024).collect::<Vec<_>>(),
        "the merged file holds its ids out of order"
    );
    reader.check("one permuted page").await;

    let state = reader.ctx.state();
    let all: Vec<i64> = (0..1024).collect();
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
    assert_eq!(read.row_ids, all);
    assert_eq!(held.size(), read.batch.get_array_memory_size());
    assert_eq!(pool.reserved(), held.size());
}

/// The `rowid`s of `reader`'s table in the order a scan returns them.
async fn by_rowid_order(reader: &Reader) -> Vec<i64> {
    let batches = reader
        .ctx
        .sql("SELECT rowid FROM lake.main.t")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(0)
                .as_any()
                .downcast_ref::<arrow::array::Int64Array>()
                .unwrap()
                .values()
                .to_vec()
        })
        .collect()
}

/// A merged file in insert order, so every guess holds, with rows deleted
/// from the page a lookup reads ahead: the deleted rows the read ahead read
/// are not returned at the head, and are at the snapshot before the delete.
/// Guards against a read ahead that returns every row it read.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_read_ahead_returns_only_located_rows() {
    let lake = Lake::new(0).await;
    lake.insert(0, INSERT_ROWS).await;
    lake.insert(INSERT_ROWS, INSERT_ROWS).await;
    lake.merge().await;
    let merged = lake.head();
    // An insert need not write its rows in `id` order, so the rows to delete
    // are named by the `id` each of `rowid`s 1000..=1010 holds.
    let truth = lake.reader(merged).await.truth().await;
    let ids: Vec<String> = (1000..=1010)
        .map(|rowid| truth[&rowid][0].clone())
        .collect();
    lake.sql(&format!(
        "DELETE FROM lake.main.t WHERE id IN ({})",
        ids.join(", ")
    ))
    .await;
    let wanted = [1000, 1005, 1011, 1500, 30_000];
    for (snapshot, found) in [(lake.head(), vec![1011, 1500, 30_000]), (merged, wanted.to_vec())] {
        let reader = lake.reader(snapshot).await;
        let truth = reader.truth().await;
        let state = reader.ctx.state();
        let (pool, held) = reservation(1 << 30);
        let read = reader
            .table
            .read_columns_by_rowid_with(
                &state,
                &wanted,
                None,
                RowIdReadOptions::default().with_reservation(&held),
            )
            .await
            .unwrap();
        assert_eq!(read.row_ids, found, "at snapshot {snapshot}");
        for (row, id) in found.iter().enumerate() {
            let cells: Vec<String> = (0..4)
                .map(|column| cell(read.batch.column(column), row))
                .collect();
            assert_eq!(cells, truth[id], "rowid {id} at snapshot {snapshot}");
        }
        assert_eq!(held.size(), read.batch.get_array_memory_size());
        assert_eq!(pool.reserved(), held.size());
    }
}

/// A cold lookup of a merged file reads its footer and page index once: the
/// bytes a first read costs over a second are the same for a read of the
/// rows, which reads ahead, as for only locating them, which does not.
/// Guards against the read ahead and the locate each fetching the footer.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_cold_lookup_reads_each_footer_once() {
    let lake = Lake::new(0).await;
    lake.insert(0, INSERT_ROWS).await;
    lake.insert(INSERT_ROWS, INSERT_ROWS).await;
    lake.merge().await;
    assert_eq!(files(&lake, lake.head()).await, (1, 1));
    let wanted = [5, 25_000];

    let reading = lake.reader(lake.head()).await;
    let state = reading.ctx.state();
    reading.store.take();
    let mut read_bytes = Vec::new();
    for _ in 0..2 {
        let read = reading
            .table
            .read_columns_by_rowid(&state, &wanted, None)
            .await
            .unwrap();
        assert_eq!(read.row_ids, wanted.to_vec());
        read_bytes.push(reading.store.take());
    }

    let locating = lake.reader(lake.head()).await;
    let state = locating.ctx.state();
    locating.store.take();
    let mut locate_bytes = Vec::new();
    for _ in 0..2 {
        let found = locating.table.locate_rowids(&state, &wanted).await.unwrap();
        assert_eq!(found.len(), wanted.len());
        locate_bytes.push(locating.store.take());
    }

    assert_eq!(
        read_bytes[0] - read_bytes[1],
        locate_bytes[0] - locate_bytes[1],
        "footer bytes: reading {read_bytes:?}, locating {locate_bytes:?}"
    );
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
    // `ROWIDS_MEASURE_DELETES=1` deletes every third row first, and
    // `ROWIDS_MEASURE_MEMO=1` reads with a 64 MiB delete memo.
    let deletes = std::env::var("ROWIDS_MEASURE_DELETES").is_ok();
    if deletes {
        lake.sql("DELETE FROM lake.main.t WHERE id % 3 = 0").await;
    }
    let options = if std::env::var("ROWIDS_MEASURE_MEMO").is_ok() {
        DuckLakeReadOptions::default().with_delete_memo(64 << 20)
    } else {
        DuckLakeReadOptions::default()
    };
    let reader = reader_with(&lake.provider, lake.head(), options).await;
    let file = reader.table.files().unwrap()[0].file.file_size_bytes;
    let wanted: Vec<i64> = if deletes {
        let truth = reader.truth().await;
        let ids: Vec<i64> = truth.keys().copied().collect();
        (0..10).map(|i| ids[1_234 + i * (ids.len() / 10)]).collect()
    } else {
        (0..10).map(|i| 1_234 + i * (rows / 10)).collect()
    };
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
