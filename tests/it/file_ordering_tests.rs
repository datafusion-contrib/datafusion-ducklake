//! With the experimental `ext_file_order` option, a scan declares its sort
//! order to DataFusion only for files the `ext_ducklake_file_order` sidecar
//! records as verified (see `src/file_ordering.rs`, `src/ext_file_order.rs`),
//! so `ORDER BY ts DESC LIMIT n` stops early on sorted data and stays correct
//! on everything else.
//!
//! Every test checks results against the same query over an in-memory copy of
//! the table, so a declared ordering that is not true fails here as wrong rows,
//! not just as a different plan.

#![cfg(all(feature = "write-sqlite", feature = "metadata-sqlite"))]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{Int64Array, RecordBatch, StringArray, TimestampMicrosecondArray};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit};
use datafusion::datasource::MemTable;
use datafusion::execution::context::SessionConfig;
use datafusion::physical_plan::displayable;
use datafusion::prelude::*;
use object_store::local::LocalFileSystem;
use tempfile::TempDir;

use datafusion_ducklake::ext_file_order::FileOrder;
use datafusion_ducklake::{
    ColumnDef, DuckLakeCatalog, DuckLakeReadOptions, DuckLakeTable, DuckLakeTableWriter,
    DuckLakeWriteOptions, MergeOptions, MetadataWriter, NullOrder, SortDirection, SortField,
    SqliteMetadataProvider, SqliteMetadataWriter, VerifyOrderOptions, WriteMode,
};

const TOP_N: &str = "SELECT ts, level, id FROM ducklake.main.logs ORDER BY ts DESC LIMIT 7";

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("ts", DataType::Timestamp(TimeUnit::Microsecond, None), true),
        Field::new("level", DataType::Utf8, true),
        Field::new("id", DataType::Int64, false),
    ]))
}

/// One row per `ts`, in the order given; `id` = `ts`, `level` cycles.
fn batch(ts: &[Option<i64>]) -> RecordBatch {
    let ids: Vec<i64> = ts.iter().map(|t| t.unwrap_or(-1)).collect();
    RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(TimestampMicrosecondArray::from(ts.to_vec())),
            Arc::new(StringArray::from_iter_values(ids.iter().map(|i| {
                if i % 5 == 0 {
                    "ERROR"
                } else {
                    "INFO"
                }
            }))),
            Arc::new(Int64Array::from(ids)),
        ],
    )
    .unwrap()
}

fn desc(range: std::ops::Range<i64>) -> Vec<Option<i64>> {
    range.rev().map(Some).collect()
}

struct Env {
    temp: TempDir,
    table_id: i64,
}

impl Env {
    fn url(&self) -> String {
        format!(
            "sqlite:{}?mode=rwc",
            self.temp.path().join("test.db").display()
        )
    }

    fn data_path(&self) -> PathBuf {
        self.temp.path().join("data")
    }

    async fn writer(&self) -> SqliteMetadataWriter {
        SqliteMetadataWriter::new_with_init(&self.url())
            .await
            .unwrap()
    }

    /// `ALTER TABLE logs SET SORTED BY (ts DESC NULLS FIRST)` (or `nulls`).
    async fn sort_by_ts_desc(&self, nulls: NullOrder) {
        self.writer()
            .await
            .set_sort_spec(
                self.table_id,
                &[SortField::column(0, "ts", SortDirection::Desc, nulls)],
            )
            .unwrap();
    }

    /// Append `batches` through the streaming session RuntimeDB ingests with:
    /// rows land in the order given, the sort is the caller's job.
    async fn append(&self, batches: &[RecordBatch]) {
        self.append_with(batches, ext_options()).await;
    }

    async fn append_with(&self, batches: &[RecordBatch], options: DuckLakeWriteOptions) {
        let writer = DuckLakeTableWriter::new(
            Arc::new(self.writer().await),
            Arc::new(LocalFileSystem::new()),
        )
        .unwrap()
        .with_options(&options);
        if options.data_inlining_row_limit.is_some() {
            writer.append_table("main", "logs", batches).await.unwrap();
            return;
        }
        let mut session = writer
            .begin_write("main", "logs", schema().as_ref(), WriteMode::Append)
            .unwrap();
        for b in batches {
            session.write_batch(b).unwrap();
        }
        session.finish().await.unwrap();
    }

    /// Live data files (by file name) and the sidecar's verified sort key for
    /// each, `None` when the sidecar has no row (or does not exist).
    async fn orders(&self) -> std::collections::HashMap<String, Option<String>> {
        use sqlx::Row;
        let pool = sqlx::SqlitePool::connect(&self.url()).await.unwrap();
        let has_sidecar: bool = sqlx::query(
            "SELECT count(*) > 0 FROM sqlite_master WHERE name = 'ext_ducklake_file_order'",
        )
        .fetch_one(&pool)
        .await
        .unwrap()
        .get(0);
        let sql = if has_sidecar {
            "SELECT df.path, o.sort_key FROM ducklake_data_file df
             LEFT JOIN ext_ducklake_file_order o
               ON o.table_id = df.table_id AND o.path = df.path
              AND o.file_size_bytes = df.file_size_bytes
             WHERE df.end_snapshot IS NULL"
        } else {
            "SELECT path, NULL FROM ducklake_data_file WHERE end_snapshot IS NULL"
        };
        sqlx::query(sql)
            .fetch_all(&pool)
            .await
            .unwrap()
            .iter()
            .map(|row| {
                let path: String = row.get(0);
                let name = path.rsplit('/').next().unwrap().to_string();
                (name, row.get::<Option<String>, _>(1))
            })
            .collect()
    }

    async fn ctx(&self, target_partitions: usize) -> SessionContext {
        self.ctx_with(target_partitions, true).await
    }

    /// A writable context; `ext_file_order` decides whether its scans read the
    /// sidecar and its writes record to it.
    async fn ctx_with(&self, target_partitions: usize, ext_file_order: bool) -> SessionContext {
        let writer = self.writer().await;
        let provider = SqliteMetadataProvider::new(&self.url()).await.unwrap();
        let catalog = DuckLakeCatalog::with_writer(Arc::new(provider), Arc::new(writer))
            .unwrap()
            .with_read_options(DuckLakeReadOptions::default().with_ext_file_order(ext_file_order))
            .with_write_options(
                DuckLakeWriteOptions::default().with_ext_file_order(ext_file_order),
            );
        let ctx = SessionContext::new_with_config(
            SessionConfig::new()
                .with_target_partitions(target_partitions)
                .with_batch_size(64),
        );
        ctx.register_catalog("ducklake", Arc::new(catalog));
        ctx
    }
}

/// Writer options with the experimental option on.
fn ext_options() -> DuckLakeWriteOptions {
    let mut options = DuckLakeWriteOptions::default();
    options.ext_file_order = Some(true);
    options
}

async fn setup() -> Env {
    let temp = TempDir::new().unwrap();
    let data_path = temp.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();
    let env = Env {
        temp,
        table_id: 0,
    };
    let writer = env.writer().await;
    writer.set_data_path(data_path.to_str().unwrap()).unwrap();
    let cols: Vec<ColumnDef> = schema()
        .fields()
        .iter()
        .map(|f| ColumnDef::from_arrow(f.name(), f.data_type(), f.is_nullable()).unwrap())
        .collect();
    let s = writer
        .begin_write_transaction("main", "logs", &cols, WriteMode::Replace)
        .unwrap();
    writer
        .publish_snapshot(
            s.table_id,
            "main",
            "logs",
            s.snapshot_id,
            WriteMode::Replace,
            s.base_snapshot_id,
            &cols,
            &s.column_ids,
        )
        .unwrap();
    Env {
        table_id: s.table_id,
        ..env
    }
}

fn parquet_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            out.extend(parquet_files(&path));
        } else if path.extension().is_some_and(|e| e == "parquet") {
            out.push(path);
        }
    }
    out
}

/// The physical plan, indented.
async fn plan(ctx: &SessionContext, sql: &str) -> String {
    let plan = ctx
        .sql(sql)
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    displayable(plan.as_ref()).indent(true).to_string()
}

/// Whether the plan sorts above the scan: a `SortExec` (TopK) anywhere means the
/// sort was not satisfied by the scan's declared ordering.
fn sorts(plan: &str) -> bool {
    plan.contains("SortExec")
}

fn declares_ts_desc(plan: &str) -> bool {
    plan.lines()
        .any(|l| l.contains("DataSourceExec") && l.contains("output_ordering=[ts@0 DESC"))
}

/// `sql` over the DuckLake table vs over an in-memory copy of all its rows.
async fn assert_matches_reference(ctx: &SessionContext, sql: &str) {
    let all = ctx
        .sql("SELECT ts, level, id FROM ducklake.main.logs")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let reference = SessionContext::new();
    reference.register_catalog(
        "ducklake",
        Arc::new(datafusion::catalog::MemoryCatalogProvider::new()),
    );
    let schema_provider = Arc::new(datafusion::catalog::MemorySchemaProvider::new());
    reference
        .catalog("ducklake")
        .unwrap()
        .register_schema("main", schema_provider)
        .unwrap();
    let mem = MemTable::try_new(all[0].schema(), vec![all]).unwrap();
    reference
        .register_table("ducklake.main.logs", Arc::new(mem))
        .unwrap();
    let got = ctx.sql(sql).await.unwrap().collect().await.unwrap();
    let want = reference.sql(sql).await.unwrap().collect().await.unwrap();
    assert_eq!(
        arrow::util::pretty::pretty_format_batches(&got)
            .unwrap()
            .to_string(),
        arrow::util::pretty::pretty_format_batches(&want)
            .unwrap()
            .to_string(),
        "{sql}"
    );
}

const QUERIES: &[&str] = &[
    TOP_N,
    "SELECT ts, level, id FROM ducklake.main.logs WHERE level = 'ERROR' ORDER BY ts DESC LIMIT 7",
    "SELECT ts, id FROM ducklake.main.logs ORDER BY ts ASC LIMIT 7",
    "SELECT ts, id FROM ducklake.main.logs ORDER BY ts DESC NULLS LAST LIMIT 7",
    "SELECT id FROM ducklake.main.logs ORDER BY ts DESC LIMIT 7",
    "SELECT date_trunc('hour', ts) AS h, count(*) AS n FROM ducklake.main.logs \
     GROUP BY h ORDER BY h DESC",
];

async fn assert_all_queries_correct(ctx: &SessionContext) {
    for sql in QUERIES {
        assert_matches_reference(ctx, sql).await;
    }
}

/// Live ingest: oldest first, each append sorted newest-first, ranges disjoint.
async fn ingest_disjoint_sorted(env: &Env, appends: i64, rows: i64) {
    for a in 0..appends {
        env.append(&[batch(&desc(a * rows..(a + 1) * rows))]).await;
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn writer_attests_only_rows_it_saw_in_order() {
    let env = setup().await;
    env.sort_by_ts_desc(NullOrder::NullsFirst).await;
    env.append(&[batch(&desc(100..200)), batch(&desc(0..100))])
        .await;
    let first = env.orders().await;
    assert_eq!(first.len(), 1);
    assert_eq!(
        first.values().next().unwrap().as_deref(),
        Some("1:desc:nulls_first")
    );

    // Same table, same live spec, rows out of order across the batch boundary.
    env.append(&[batch(&desc(200..300)), batch(&desc(300..400))])
        .await;
    let orders = env.orders().await;
    assert_eq!(orders.len(), 2);
    assert_eq!(orders.values().filter(|o| o.is_none()).count(), 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn no_sort_spec_no_attestation() {
    let env = setup().await;
    env.append(&[batch(&desc(0..100))]).await;
    assert!(env.orders().await.values().all(Option::is_none));
}

/// Off (the default): nothing is recorded, no sidecar table is created, and
/// scans plan exactly as before. A reader with the option on over a catalog
/// that never had it finds no table and scans unordered.
#[tokio::test(flavor = "multi_thread")]
async fn option_off_records_nothing_and_declares_nothing() {
    let env = setup().await;
    env.sort_by_ts_desc(NullOrder::NullsFirst).await;
    for a in 0..3 {
        env.append_with(
            &[batch(&desc(a * 500..(a + 1) * 500))],
            DuckLakeWriteOptions::default(),
        )
        .await;
    }
    assert!(env.orders().await.values().all(Option::is_none));
    let off = env.ctx_with(4, false).await;
    assert!(sorts(&plan(&off, TOP_N).await));
    assert_all_queries_correct(&off).await;
    let on = env.ctx_with(4, true).await;
    assert!(
        sorts(&plan(&on, TOP_N).await),
        "no sidecar table: nothing to trust"
    );
    assert_all_queries_correct(&on).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn ordering_declared_for_attested_disjoint_files() {
    let env = setup().await;
    env.sort_by_ts_desc(NullOrder::NullsFirst).await;
    ingest_disjoint_sorted(&env, 4, 500).await;
    for partitions in [1, 3, 8] {
        let ctx = env.ctx(partitions).await;
        let p = plan(&ctx, TOP_N).await;
        assert!(declares_ts_desc(&p), "partitions={partitions}\n{p}");
        assert!(
            !sorts(&p),
            "ORDER BY ts DESC LIMIT should need no sort\n{p}"
        );
        // The fetch reaches the scan, each partition stopping at 7 rows; above
        // one partition it passes through the run-ahead buffer to get there.
        let scan = p.lines().find(|l| l.contains("DataSourceExec")).unwrap();
        assert!(scan.contains("limit=7"), "{p}");
        // One verified chain per file here: merged, buffered, fetch in the scan.
        assert!(
            p.contains("MergedChainsExec") || !p.contains("SortPreservingMergeExec"),
            "{p}"
        );
        assert_all_queries_correct(&ctx).await;
    }
}

/// DuckDB's default null order for DESC is NULLS LAST; the SQL query's is NULLS
/// FIRST. With no NULLs in the key both hold, and the plan must see that.
#[tokio::test(flavor = "multi_thread")]
async fn nulls_last_spec_satisfies_default_desc_query() {
    let env = setup().await;
    env.sort_by_ts_desc(NullOrder::NullsLast).await;
    ingest_disjoint_sorted(&env, 3, 300).await;
    let ctx = env.ctx(4).await;
    let p = plan(&ctx, TOP_N).await;
    assert!(!sorts(&p), "{p}");
    assert_all_queries_correct(&ctx).await;
}

/// The EXPLAIN ANALYZE of the newest-N query reads a few batches, not the table.
#[tokio::test(flavor = "multi_thread")]
async fn explain_analyze_shows_early_stop() {
    let env = setup().await;
    env.sort_by_ts_desc(NullOrder::NullsFirst).await;
    ingest_disjoint_sorted(&env, 6, 5_000).await;
    let ctx = env.ctx(1).await;
    let batches = ctx
        .sql(&format!("EXPLAIN ANALYZE {TOP_N}"))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let text = arrow::util::pretty::pretty_format_batches(&batches)
        .unwrap()
        .to_string();
    let scan = text
        .lines()
        .find(|l| l.contains("DataSourceExec"))
        .unwrap_or_else(|| panic!("{text}"));
    let rows: usize = scan
        .split("output_rows=")
        .nth(1)
        .and_then(|r| r.split(|c: char| !c.is_ascii_digit()).next())
        .and_then(|n| n.parse().ok())
        .unwrap_or_else(|| panic!("{scan}"));
    assert!(rows < 1_000, "scan read {rows} of 30000 rows\n{text}");
    assert!(!text.contains("SortExec"), "{text}");
}

/// Live ingest before compaction: appends whose ranges overlap. With one
/// partition the files cannot chain into one non-overlapping group, so nothing
/// is declared; with more, DataFusion's split puts them in separate groups.
#[tokio::test(flavor = "multi_thread")]
async fn overlapping_files() {
    let env = setup().await;
    env.sort_by_ts_desc(NullOrder::NullsFirst).await;
    env.append(&[batch(&desc(0..1_000))]).await;
    env.append(&[batch(
        &desc(500..1_500)
            .into_iter()
            .map(|t| t.map(|t| t * 2 + 1))
            .collect::<Vec<_>>(),
    )])
    .await;
    env.append(&[batch(&desc(900..1_100))]).await;

    let one = env.ctx(1).await;
    let p = plan(&one, TOP_N).await;
    assert!(!declares_ts_desc(&p), "{p}");
    assert!(sorts(&p), "{p}");
    assert_all_queries_correct(&one).await;

    let several = env.ctx(4).await;
    let p = plan(&several, TOP_N).await;
    assert!(declares_ts_desc(&p), "{p}");
    assert_all_queries_correct(&several).await;
}

/// A file written before the sort spec existed (or by another DuckLake client)
/// has no attestation. Give it the NEWEST rows: trusting the catalog spec for it
/// would return the wrong top rows.
#[tokio::test(flavor = "multi_thread")]
async fn file_without_attestation_holds_the_newest_rows() {
    let env = setup().await;
    // Unsorted, and newest.
    let mut newest: Vec<Option<i64>> = (9_000..9_500).map(Some).collect();
    newest.swap(3, 400);
    env.append(&[batch(&newest)]).await;
    env.sort_by_ts_desc(NullOrder::NullsFirst).await;
    ingest_disjoint_sorted(&env, 3, 1_000).await;
    for partitions in [1, 4] {
        let ctx = env.ctx(partitions).await;
        let p = plan(&ctx, TOP_N).await;
        assert!(sorts(&p), "unattested rows must still be sorted\n{p}");
        assert_all_queries_correct(&ctx).await;
    }
}

/// `sort_on_insert` off, or a caller that never sorted: the live spec says
/// sorted, the rows are not. No attestation, correct results.
#[tokio::test(flavor = "multi_thread")]
async fn unsorted_rows_under_a_live_spec() {
    let env = setup().await;
    env.sort_by_ts_desc(NullOrder::NullsFirst).await;
    let rows: Vec<Option<i64>> = (0..2_000).map(|i| Some((i * 7_919) % 2_000)).collect();
    env.append(&[batch(&rows)]).await;
    ingest_disjoint_sorted(&env, 1, 100).await;
    assert_eq!(
        env.orders().await.values().filter(|o| o.is_none()).count(),
        1
    );
    let ctx = env.ctx(3).await;
    assert!(sorts(&plan(&ctx, TOP_N).await));
    assert_all_queries_correct(&ctx).await;
}

/// A file with a delete file scans through the delete filter, apart from the
/// ordered group; deleting the newest rows must not resurrect them.
#[tokio::test(flavor = "multi_thread")]
async fn delete_files() {
    let env = setup().await;
    env.sort_by_ts_desc(NullOrder::NullsFirst).await;
    ingest_disjoint_sorted(&env, 3, 1_000).await;
    let ctx = env.ctx(4).await;
    ctx.sql("DELETE FROM ducklake.main.logs WHERE id >= 2995 OR id = 1500")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let ctx = env.ctx(4).await;
    assert_all_queries_correct(&ctx).await;
    let top = ctx.sql(TOP_N).await.unwrap().collect().await.unwrap();
    let ids = top[0]
        .column(2)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(ids.value(0), 2994, "deleted rows must stay deleted");
}

/// Rows inlined in the catalog are a separate, unordered input; make them the
/// newest.
#[tokio::test(flavor = "multi_thread")]
async fn inlined_rows_are_newest() {
    let env = setup().await;
    env.sort_by_ts_desc(NullOrder::NullsFirst).await;
    ingest_disjoint_sorted(&env, 3, 1_000).await;
    env.append_with(
        &[batch(&[Some(50_000), Some(70_000), Some(60_000)])],
        DuckLakeWriteOptions::default().with_data_inlining_row_limit(10),
    )
    .await;
    for partitions in [1, 4] {
        let ctx = env.ctx(partitions).await;
        // Inlined rows are sorted at plan time and declared, so the union with
        // the verified files is ordered: no sort above it, still early stop.
        let p = plan(&ctx, TOP_N).await;
        assert!(!sorts(&p), "{p}");
        // The fetch reaches the parquet scan through the union with the
        // inlined rows, so it still stops early.
        assert!(
            p.lines().any(|l| l.contains("DataSourceExec")
                && l.contains(".parquet")
                && l.contains("limit=7")),
            "{p}"
        );
        assert_all_queries_correct(&ctx).await;
        let top = ctx.sql(TOP_N).await.unwrap().collect().await.unwrap();
        let ids = top[0]
            .column(2)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(
            (ids.value(0), ids.value(1), ids.value(2)),
            (70_000, 60_000, 50_000)
        );
    }
}

/// Inlined rows mixed with verified files, NULL keys among them, overlapping
/// the files' range; then flushed: the flushed file is verified and recorded
/// like any other write.
#[tokio::test(flavor = "multi_thread")]
async fn inlined_rows_mixed_with_verified_files_and_flushed() {
    let env = setup().await;
    env.sort_by_ts_desc(NullOrder::NullsFirst).await;
    ingest_disjoint_sorted(&env, 3, 1_000).await;
    // Overlapping the files (1500, 2500), newer (9000), and NULL.
    env.append_with(&[batch(&[Some(1_500), None, Some(9_000), Some(2_500)])], {
        let mut options = ext_options();
        options.data_inlining_row_limit = Some(10);
        options
    })
    .await;
    for partitions in [1, 4] {
        assert_all_queries_correct(&env.ctx(partitions).await).await;
    }

    // Flushed through the catalog's flush (`inlined_flush`), which verifies
    // and records the flushed file like any write.
    let catalog = DuckLakeCatalog::with_writer(
        Arc::new(SqliteMetadataProvider::new(&env.url()).await.unwrap()),
        Arc::new(env.writer().await),
    )
    .unwrap()
    .with_read_options(DuckLakeReadOptions::default().with_ext_file_order(true))
    .with_write_options(DuckLakeWriteOptions::default().with_ext_file_order(true));
    let flushed = catalog
        .flush_inlined_data(&SessionContext::new().state(), Some("main"), Some("logs"))
        .await
        .unwrap();
    assert_eq!(flushed.len(), 1, "{flushed:?}");
    // Every live file, the flushed one included, is recorded (the NULL lands
    // first in a DESC NULLS FIRST sort, so the flushed file is in order).
    let orders = env.orders().await;
    assert_eq!(orders.len(), 4);
    assert!(orders.values().all(Option::is_some), "{orders:?}");
    for partitions in [1, 4] {
        assert_all_queries_correct(&env.ctx(partitions).await).await;
    }
}

/// A NULL key sits outside every min/max the cross-file check compares, so a
/// file holding one is scanned unordered (SQL `DESC` puts NULLs first).
#[tokio::test(flavor = "multi_thread")]
async fn null_keys() {
    let env = setup().await;
    env.sort_by_ts_desc(NullOrder::NullsFirst).await;
    // The NULL is in the OLDEST file: chained by min/max, that file is read
    // last, so trusting it would emit the NULL after newer rows.
    let mut with_null = vec![None];
    with_null.extend(desc(0..1_000));
    env.append(&[batch(&with_null)]).await;
    env.append(&[batch(&desc(1_000..2_000))]).await;
    assert!(env.orders().await.values().all(Option::is_some));
    for partitions in [1, 4] {
        let ctx = env.ctx(partitions).await;
        assert_all_queries_correct(&ctx).await;
    }
}

/// Compaction sorts its output, so it attests: overlapping ingest files that
/// declared nothing become one ordered file.
#[tokio::test(flavor = "multi_thread")]
async fn compaction_output_is_attested() {
    let env = setup().await;
    env.sort_by_ts_desc(NullOrder::NullsFirst).await;
    env.append(&[batch(&desc(0..1_000))]).await;
    // Sorted too, so attested, but interleaved with the first file.
    env.append(&[batch(
        &desc(0..1_000)
            .into_iter()
            .map(|t| t.map(|t| t * 2 + 1))
            .collect::<Vec<_>>(),
    )])
    .await;
    assert!(env.orders().await.values().all(Option::is_some));
    let one = env.ctx(1).await;
    assert!(!declares_ts_desc(&plan(&one, TOP_N).await));

    let ctx = env.ctx(1).await;
    let table = ctx
        .catalog("ducklake")
        .unwrap()
        .schema("main")
        .unwrap()
        .table("logs")
        .await
        .unwrap()
        .unwrap();
    let table = (table.as_ref() as &dyn std::any::Any)
        .downcast_ref::<DuckLakeTable>()
        .unwrap()
        .clone();
    table
        .merge_adjacent_files(&ctx.state(), MergeOptions::default())
        .await
        .unwrap();

    let one = env.ctx(1).await;
    let p = plan(&one, TOP_N).await;
    assert!(declares_ts_desc(&p), "{p}");
    assert!(!sorts(&p), "{p}");
    assert_all_queries_correct(&one).await;
}

// ---------------------------------------------------------------------------
// verify_file_orders: record files written before the option, or by others
// ---------------------------------------------------------------------------

async fn table_handle(env: &Env) -> (SessionContext, DuckLakeTable) {
    let ctx = env.ctx(1).await;
    let table = ctx
        .catalog("ducklake")
        .unwrap()
        .schema("main")
        .unwrap()
        .table("logs")
        .await
        .unwrap()
        .unwrap();
    let table = (table.as_ref() as &dyn std::any::Any)
        .downcast_ref::<DuckLakeTable>()
        .unwrap()
        .clone();
    (ctx, table)
}

/// `(rowid, id)` for every live row, by rowid.
async fn rowids(env: &Env) -> Vec<(i64, i64)> {
    let provider = SqliteMetadataProvider::new(&env.url()).await.unwrap();
    let catalog = DuckLakeCatalog::new(provider)
        .unwrap()
        .with_row_lineage(true);
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));
    let batches = ctx
        .sql("SELECT rowid, id FROM ducklake.main.logs ORDER BY rowid")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    batches
        .iter()
        .flat_map(|b| {
            let r = b
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .clone();
            let i = b
                .column(1)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .clone();
            (0..b.num_rows()).map(move |n| (r.value(n), i.value(n)))
        })
        .collect()
}

async fn current_snapshot(env: &Env) -> i64 {
    use datafusion_ducklake::metadata_provider::MetadataProvider;
    SqliteMetadataProvider::new(&env.url())
        .await
        .unwrap()
        .get_current_snapshot()
        .unwrap()
}

#[tokio::test(flavor = "multi_thread")]
async fn verify_records_existing_files_without_rewriting_them() {
    let env = setup().await;
    // Written before the option (and the spec): sorted, unrecorded.
    for a in 0..3 {
        env.append_with(
            &[batch(&desc(a * 1_000..(a + 1) * 1_000))],
            DuckLakeWriteOptions::default(),
        )
        .await;
    }
    env.sort_by_ts_desc(NullOrder::NullsFirst).await;
    assert!(sorts(&plan(&env.ctx(4).await, TOP_N).await));
    let snapshot_before = current_snapshot(&env).await;
    let files_before = parquet_files(&env.data_path());
    let rowids_before = rowids(&env).await;

    let (ctx, table) = table_handle(&env).await;
    let result = table
        .verify_file_orders(&ctx.state(), VerifyOrderOptions::default())
        .await
        .unwrap();
    assert_eq!(
        (
            result.files_verified,
            result.files_recorded,
            result.files_unsorted,
            result.files_skipped
        ),
        (3, 3, 0, 0),
        "{result:?}"
    );
    assert!(result.bytes_read > 0);

    // No snapshot, no new or changed file, same rows and rowids.
    assert_eq!(current_snapshot(&env).await, snapshot_before);
    assert_eq!(parquet_files(&env.data_path()), files_before);
    assert_eq!(rowids(&env).await, rowids_before);
    assert!(env.orders().await.values().all(Option::is_some));
    for partitions in [1, 4] {
        let ctx = env.ctx(partitions).await;
        let p = plan(&ctx, TOP_N).await;
        assert!(declares_ts_desc(&p) && !sorts(&p), "{p}");
        assert_all_queries_correct(&ctx).await;
    }

    // Idempotent: everything is recorded.
    let (ctx, table) = table_handle(&env).await;
    let again = table
        .verify_file_orders(&ctx.state(), VerifyOrderOptions::default())
        .await
        .unwrap();
    assert_eq!(
        (again.files_verified, again.files_skipped),
        (0, 3),
        "{again:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn verify_leaves_unsorted_files_alone() {
    let env = setup().await;
    // The unsorted file holds the newest rows, so recording it would show as
    // wrong top rows (no delete-filtered branch here to keep a sort above it).
    let shuffled: Vec<Option<i64>> = (0..1_000)
        .map(|i| Some(5_000 + (i * 7_919) % 1_000))
        .collect();
    env.append_with(&[batch(&shuffled)], DuckLakeWriteOptions::default())
        .await;
    env.append_with(
        &[batch(&desc(1_000..2_000))],
        DuckLakeWriteOptions::default(),
    )
    .await;
    env.sort_by_ts_desc(NullOrder::NullsFirst).await;

    let (ctx, table) = table_handle(&env).await;
    let result = table
        .verify_file_orders(&ctx.state(), VerifyOrderOptions::default())
        .await
        .unwrap();
    for partitions in [1, 4] {
        assert_all_queries_correct(&env.ctx(partitions).await).await;
    }
    assert_eq!(
        (
            result.files_verified,
            result.files_recorded,
            result.files_unsorted
        ),
        (2, 1, 1),
        "{result:?}"
    );
}

/// A parquet file written outside DuckLake (no field ids), registered as is —
/// a user upload or `ducklake_add_data_files` — is verified by column name.
#[tokio::test(flavor = "multi_thread")]
async fn verify_records_an_external_upload() {
    use datafusion_ducklake::metadata_writer::DataFileInfo;
    let env = setup().await;
    env.sort_by_ts_desc(NullOrder::NullsFirst).await;
    ingest_disjoint_sorted(&env, 1, 1_000).await;

    let path = env
        .data_path()
        .join("main")
        .join("logs")
        .join("upload.parquet");
    let plain = Arc::new(Schema::new(
        schema()
            .fields()
            .iter()
            .map(|f| f.as_ref().clone())
            .collect::<Vec<_>>(),
    ));
    let rows = batch(&desc(1_000..2_000));
    let rows = RecordBatch::try_new(Arc::clone(&plain), rows.columns().to_vec()).unwrap();
    let mut out =
        parquet::arrow::ArrowWriter::try_new(std::fs::File::create(&path).unwrap(), plain, None)
            .unwrap();
    out.write(&rows).unwrap();
    out.close().unwrap();
    let size = std::fs::metadata(&path).unwrap().len() as i64;
    let writer = env.writer().await;
    let cols: Vec<ColumnDef> = schema()
        .fields()
        .iter()
        .map(|f| ColumnDef::from_arrow(f.name(), f.data_type(), f.is_nullable()).unwrap())
        .collect();
    let setup = writer
        .begin_write_transaction("main", "logs", &cols, WriteMode::Append)
        .unwrap();
    writer
        .register_data_file(
            setup.table_id,
            "main",
            "logs",
            setup.snapshot_id,
            &DataFileInfo::new("upload.parquet", size, 1_000).with_column_stats(
                datafusion_ducklake::stats_collect::collect_column_stats(
                    &path,
                    &setup.column_ids,
                    1_000,
                    &[],
                ),
            ),
            WriteMode::Append,
            setup.base_snapshot_id,
            &cols,
            &setup.column_ids,
        )
        .unwrap();
    assert_eq!(env.orders().await.get("upload.parquet"), Some(&None));

    let (ctx, table) = table_handle(&env).await;
    let result = table
        .verify_file_orders(&ctx.state(), VerifyOrderOptions::default())
        .await
        .unwrap();
    assert_eq!(
        (result.files_verified, result.files_recorded),
        (1, 1),
        "{result:?}"
    );
    assert_eq!(
        env.orders().await.get("upload.parquet"),
        Some(&Some("1:desc:nulls_first".to_string()))
    );
    for partitions in [1, 4] {
        let ctx = env.ctx(partitions).await;
        assert!(!sorts(&plan(&ctx, TOP_N).await));
        assert_all_queries_correct(&ctx).await;
    }
}

/// A verify pass racing a compaction: an order for a file the compaction has
/// retired is not recorded (the insert is conditioned on the file being
/// live), and the compacted output is recorded by its own write. A row that
/// did land for a retired file would be harmless, and `prune_file_orders`
/// removes it.
#[tokio::test(flavor = "multi_thread")]
async fn verify_racing_compaction_and_prune() {
    use datafusion_ducklake::metadata_provider::MetadataProvider;
    let env = setup().await;
    for a in 0..2 {
        env.append_with(
            &[batch(&desc(a * 1_000..(a + 1) * 1_000))],
            DuckLakeWriteOptions::default(),
        )
        .await;
    }
    env.sort_by_ts_desc(NullOrder::NullsFirst).await;
    let provider = SqliteMetadataProvider::new(&env.url()).await.unwrap();
    let before: Vec<FileOrder> = provider
        .get_table_files_for_select(env.table_id, provider.get_current_snapshot().unwrap())
        .unwrap()
        .iter()
        .map(|f| FileOrder {
            path: f.file.path.clone(),
            file_size_bytes: f.file.file_size_bytes,
            sort_key: "1:desc:nulls_first".to_string(),
        })
        .collect();
    assert_eq!(before.len(), 2);

    // Compaction retires both unrecorded files; its output is recorded.
    let (ctx, table) = table_handle(&env).await;
    table
        .with_write_options(ext_options())
        .merge_adjacent_files(&ctx.state(), MergeOptions::default())
        .await
        .unwrap();
    let orders = env.orders().await;
    assert_eq!(orders.len(), 1);
    assert!(orders.values().all(Option::is_some));

    // The verify pass read the two files before the merge and records now.
    // The rows are true of those objects, but the merge removed the files
    // from the catalog, so they match nothing.
    let writer = env.writer().await;
    assert_eq!(
        writer
            .record_file_orders(env.table_id, "verify", &before)
            .unwrap(),
        2
    );
    for partitions in [1, 4] {
        assert_all_queries_correct(&env.ctx(partitions).await).await;
    }

    // Rows for paths no file of the table has are pruned; the merge output's
    // row stays.
    assert_eq!(writer.prune_file_orders().unwrap(), 2);
    assert_eq!(sidecar_rows(&env).await, 1);
    assert_eq!(env.orders().await, orders);
}

async fn sidecar_rows(env: &Env) -> i64 {
    use sqlx::Row;
    let pool = sqlx::SqlitePool::connect(&env.url()).await.unwrap();
    sqlx::query("SELECT count(*) FROM ext_ducklake_file_order")
        .fetch_one(&pool)
        .await
        .unwrap()
        .get(0)
}

/// Plan text with the wrapper line dropped and indentation removed, so a plan
/// with `ext_file_order` on compares to the same plan with it off.
fn without_wrapper(plan: &str) -> Vec<String> {
    plan.lines()
        .filter(|l| !l.contains("OrderOnRequestExec"))
        .map(|l| l.trim().to_string())
        .collect()
}

/// A query that does not sort by the table's key plans exactly as with the
/// option off: same file groups and byte-range splits, no buffer, no
/// declared order. Only `ORDER BY ts DESC` gets the ordered plan. Small
/// files are scanned whole; with `repartition_file_min_size` at 1 byte the
/// same files are split into byte ranges across partitions.
#[tokio::test(flavor = "multi_thread")]
async fn queries_without_a_key_sort_plan_as_today() {
    let env = setup().await;
    env.sort_by_ts_desc(NullOrder::NullsFirst).await;
    ingest_disjoint_sorted(&env, 4, 1_000).await;
    let split_into_ranges = |ctx: SessionContext| async move {
        ctx.sql("SET datafusion.optimizer.repartition_file_min_size = 1")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        ctx
    };
    for (partitions, split) in [(1, false), (4, false), (8, false), (4, true), (8, true)] {
        let mut on = env.ctx_with(partitions, true).await;
        let mut off = env.ctx_with(partitions, false).await;
        if split {
            on = split_into_ranges(on).await;
            off = split_into_ranges(off).await;
            let p = plan(&off, "SELECT * FROM ducklake.main.logs").await;
            assert!(p.contains(".parquet:0.."), "byte-range split expected\n{p}");
        }
        for sql in [
            "SELECT date_trunc('hour', ts) AS h, count(*) FROM ducklake.main.logs GROUP BY h",
            "SELECT level, count(*) FROM ducklake.main.logs WHERE ts > TIMESTAMP '1970-01-01 00:00:00.002' GROUP BY level",
            "SELECT * FROM ducklake.main.logs LIMIT 5",
            "SELECT * FROM ducklake.main.logs WHERE level = 'ERROR' LIMIT 5",
            "SELECT * FROM ducklake.main.logs ORDER BY id DESC LIMIT 5",
            "SELECT * FROM ducklake.main.logs ORDER BY ts ASC LIMIT 5",
        ] {
            let deterministic = sql.contains("ORDER BY");
            let with = plan(&on, sql).await;
            let without = plan(&off, sql).await;
            assert_eq!(
                without_wrapper(&with),
                without_wrapper(&without),
                "partitions={partitions} split={split} {sql}\n--- on\n{with}\n--- off\n{without}"
            );
            assert!(!with.contains("output_ordering"), "{with}");
            if deterministic {
                assert_matches_reference(&on, sql).await;
            } else {
                let rows = |ctx: SessionContext| async move {
                    let mut lines: Vec<String> = arrow::util::pretty::pretty_format_batches(
                        &ctx.sql(sql).await.unwrap().collect().await.unwrap(),
                    )
                    .unwrap()
                    .to_string()
                    .lines()
                    .map(str::to_string)
                    .collect();
                    lines.sort();
                    lines
                };
                if sql.contains("LIMIT") {
                    assert_eq!(
                        rows(on.clone()).await.len(),
                        rows(off.clone()).await.len(),
                        "{sql}"
                    );
                } else {
                    assert_eq!(rows(on.clone()).await, rows(off.clone()).await, "{sql}");
                }
            }
        }
        let p = plan(&on, TOP_N).await;
        assert!(declares_ts_desc(&p) && !sorts(&p), "{p}");
    }
}

/// With the sort removed by `PushdownSort`, the chains are merged with a
/// per-chain buffer sized by the session's `sort_pushdown_buffer_capacity`,
/// and the fetch is in each chain's scan.
#[tokio::test(flavor = "multi_thread")]
async fn sort_buffer_follows_datafusion_setting() {
    let env = setup().await;
    env.sort_by_ts_desc(NullOrder::NullsFirst).await;
    ingest_disjoint_sorted(&env, 4, 200).await;
    let ctx = env.ctx(4).await;
    let default = plan(&ctx, TOP_N).await;
    assert!(
        default.contains("MergedChainsExec: capacity=1073741824"),
        "{default}"
    );
    ctx.sql("SET datafusion.execution.sort_pushdown_buffer_capacity = 1048576")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let set = plan(&ctx, TOP_N).await;
    assert!(set.contains("MergedChainsExec: capacity=1048576"), "{set}");
    assert!(
        set.lines()
            .any(|l| l.contains("DataSourceExec") && l.contains("limit=7")),
        "{set}"
    );
    assert_all_queries_correct(&ctx).await;
}

/// Fewer verified chains than partitions (a short window holding few files):
/// the ordered plan splits single-file chains into contiguous byte ranges, as
/// today's plan would, so it reads on as many threads; still ordered.
#[tokio::test(flavor = "multi_thread")]
async fn few_chains_are_split_across_partitions() {
    let env = setup().await;
    env.sort_by_ts_desc(NullOrder::NullsFirst).await;
    ingest_disjoint_sorted(&env, 2, 20_000).await;
    let ctx = env.ctx(8).await;
    ctx.sql("SET datafusion.optimizer.repartition_file_min_size = 1")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let p = plan(&ctx, TOP_N).await;
    let scan = p.lines().find(|l| l.contains("DataSourceExec")).unwrap();
    assert!(
        scan.contains("8 groups") && scan.contains(".parquet:0.."),
        "two chains split into byte ranges across 8 partitions\n{p}"
    );
    assert!(!sorts(&p) && declares_ts_desc(&p), "{p}");
    assert_all_queries_correct(&ctx).await;

    // A filter left above the scan (here: filter pushdown off) is a
    // `FilterExec`, which in DataFusion 55 does not forward the sort request:
    // the query keeps today's plan, a TopK. Correct, no early stop.
    let filtered =
        "SELECT ts, level, id FROM ducklake.main.logs WHERE id % 97 = 3 ORDER BY ts DESC LIMIT 7";
    let p = plan(&ctx, filtered).await;
    assert!(
        p.contains("FilterExec") && sorts(&p) && !declares_ts_desc(&p),
        "{p}"
    );
    assert_matches_reference(&ctx, filtered).await;
}

/// A sidecar row describes one file. A writer without the shared file-id
/// counter (a release before it, or another client allocating SQLite row
/// ids) hands a retired file's `data_file_id` to its next file; the row of
/// the retired, sorted file must not then vouch for the new, unsorted one.
#[tokio::test(flavor = "multi_thread")]
async fn sidecar_row_never_vouches_for_a_file_that_reuses_its_id() {
    let env = setup().await;
    env.sort_by_ts_desc(NullOrder::NullsFirst).await;
    env.append(&[batch(&desc(0..1_000))]).await;
    // Ascending, so NOT in the spec's order: written with the option off.
    let ascending: Vec<Option<i64>> = (1_000..2_000).map(Some).collect();
    env.append_with(&[batch(&ascending)], DuckLakeWriteOptions::default())
        .await;

    // What expiry plus such a writer leave behind: the sorted file's rows are
    // gone, and the unsorted file sits under the sorted file's id.
    let pool = sqlx::SqlitePool::connect(&env.url()).await.unwrap();
    let ids: Vec<i64> =
        sqlx::query_scalar("SELECT data_file_id FROM ducklake_data_file ORDER BY data_file_id")
            .fetch_all(&pool)
            .await
            .unwrap();
    let [sorted_id, unsorted_id] = ids[..] else {
        panic!("two data files expected: {ids:?}");
    };
    for table in ["ducklake_data_file", "ducklake_file_column_stats"] {
        sqlx::query(sqlx::AssertSqlSafe(format!(
            "DELETE FROM {table} WHERE data_file_id = ?"
        )))
        .bind(sorted_id)
        .execute(&pool)
        .await
        .unwrap();
        sqlx::query(sqlx::AssertSqlSafe(format!(
            "UPDATE {table} SET data_file_id = ? WHERE data_file_id = ?"
        )))
        .bind(sorted_id)
        .bind(unsorted_id)
        .execute(&pool)
        .await
        .unwrap();
    }

    for partitions in [1, 4] {
        let ctx = env.ctx(partitions).await;
        assert_all_queries_correct(&ctx).await;
        let p = plan(&ctx, TOP_N).await;
        assert!(sorts(&p) && !declares_ts_desc(&p), "{p}");
    }
}

/// An inner `LIMIT` that `LimitPushdown` (which runs before `PushdownSort`)
/// already moved into the scan bounds the subquery. Replanning the scan for
/// an outer sort must not replace it with the outer fetch, nor turn it into a
/// per-chain limit: the subquery still yields 10 rows, whatever the outer
/// `LIMIT`, layout or partition count.
#[tokio::test(flavor = "multi_thread")]
async fn inner_limit_survives_an_ordered_replan() {
    const INNER: &str = "SELECT ts, id FROM ducklake.main.logs LIMIT 10";
    let queries = [
        format!("SELECT ts, id FROM ({INNER}) ORDER BY ts DESC LIMIT 15"),
        format!("SELECT ts, id FROM ({INNER}) ORDER BY ts DESC LIMIT 5"),
        format!("SELECT ts, id FROM ({INNER}) ORDER BY ts DESC"),
    ];
    let expected_rows = |sql: &str| {
        if sql.ends_with("LIMIT 5") {
            5
        } else {
            10
        }
    };

    // Disjoint files (one chain), overlapping files (several chains), and
    // the same with inlined rows in the union.
    let disjoint = setup().await;
    disjoint.sort_by_ts_desc(NullOrder::NullsFirst).await;
    ingest_disjoint_sorted(&disjoint, 3, 1_000).await;
    let overlapping = setup().await;
    overlapping.sort_by_ts_desc(NullOrder::NullsFirst).await;
    overlapping.append(&[batch(&desc(0..1_000))]).await;
    overlapping.append(&[batch(&desc(500..1_500))]).await;
    overlapping.append(&[batch(&desc(900..1_100))]).await;
    let inlined = setup().await;
    inlined.sort_by_ts_desc(NullOrder::NullsFirst).await;
    ingest_disjoint_sorted(&inlined, 3, 1_000).await;
    inlined
        .append_with(
            &[batch(&[Some(50_000), Some(70_000), Some(60_000)])],
            DuckLakeWriteOptions::default().with_data_inlining_row_limit(10),
        )
        .await;

    for env in [&disjoint, &overlapping, &inlined] {
        for partitions in [1, 4] {
            for ext_file_order in [false, true] {
                let ctx = env.ctx_with(partitions, ext_file_order).await;
                for sql in &queries {
                    let rows = ctx.sql(sql).await.unwrap().collect().await.unwrap();
                    let n: usize = rows.iter().map(RecordBatch::num_rows).sum();
                    assert_eq!(
                        n,
                        expected_rows(sql),
                        "partitions={partitions} ext_file_order={ext_file_order}: {sql}\n{}",
                        plan(&ctx, sql).await
                    );
                }
            }
        }
    }

    // One chain split into byte ranges across 8 partitions.
    let split = setup().await;
    split.sort_by_ts_desc(NullOrder::NullsFirst).await;
    ingest_disjoint_sorted(&split, 2, 20_000).await;
    let ctx = split.ctx(8).await;
    ctx.sql("SET datafusion.optimizer.repartition_file_min_size = 1")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    for sql in &queries {
        let rows = ctx.sql(sql).await.unwrap().collect().await.unwrap();
        let n: usize = rows.iter().map(RecordBatch::num_rows).sum();
        assert_eq!(n, expected_rows(sql), "{sql}\n{}", plan(&ctx, sql).await);
    }
}

/// The verify job checks a file's order as the reader will see it: in the
/// catalog's column type. An upload storing a `TIMESTAMP` key as ISO text is
/// read through a cast, and text with mixed UTC offsets sorts differently as
/// text than as instants, so its text order proves nothing.
#[tokio::test(flavor = "multi_thread")]
async fn verify_checks_order_in_the_catalog_type() {
    use datafusion_ducklake::metadata_writer::DataFileInfo;
    let env = setup().await;
    env.sort_by_ts_desc(NullOrder::NullsFirst).await;

    // Descending as text; as instants the `-01:00` row is an hour later than
    // the rows above it.
    let ts = [
        "1970-01-01T00:00:00.009+00:00",
        "1970-01-01T00:00:00.008-01:00",
        "1970-01-01T00:00:00.007+00:00",
        "1970-01-01T00:00:00.006+00:00",
    ];
    let text = Arc::new(Schema::new(vec![
        Field::new("ts", DataType::Utf8, true),
        Field::new("level", DataType::Utf8, true),
        Field::new("id", DataType::Int64, false),
    ]));
    let rows = RecordBatch::try_new(
        Arc::clone(&text),
        vec![
            Arc::new(StringArray::from(ts.to_vec())),
            Arc::new(StringArray::from(vec!["INFO"; ts.len()])),
            Arc::new(Int64Array::from_iter_values(0..ts.len() as i64)),
        ],
    )
    .unwrap();
    let path = env
        .data_path()
        .join("main")
        .join("logs")
        .join("text.parquet");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let mut out =
        parquet::arrow::ArrowWriter::try_new(std::fs::File::create(&path).unwrap(), text, None)
            .unwrap();
    out.write(&rows).unwrap();
    out.close().unwrap();
    let size = std::fs::metadata(&path).unwrap().len() as i64;
    let writer = env.writer().await;
    let cols: Vec<ColumnDef> = schema()
        .fields()
        .iter()
        .map(|f| ColumnDef::from_arrow(f.name(), f.data_type(), f.is_nullable()).unwrap())
        .collect();
    let setup = writer
        .begin_write_transaction("main", "logs", &cols, WriteMode::Append)
        .unwrap();
    writer
        .register_data_file(
            setup.table_id,
            "main",
            "logs",
            setup.snapshot_id,
            &DataFileInfo::new("text.parquet", size, ts.len() as i64).with_column_stats(
                datafusion_ducklake::stats_collect::collect_column_stats(
                    &path,
                    &setup.column_ids,
                    ts.len() as i64,
                    &[],
                ),
            ),
            WriteMode::Append,
            setup.base_snapshot_id,
            &cols,
            &setup.column_ids,
        )
        .unwrap();

    let (ctx, table) = table_handle(&env).await;
    table
        .verify_file_orders(&ctx.state(), VerifyOrderOptions::default())
        .await
        .unwrap();
    for partitions in [1, 4] {
        assert_all_queries_correct(&env.ctx(partitions).await).await;
    }
    assert_eq!(env.orders().await.get("text.parquet"), Some(&None));
}

/// After `SET SORTED BY` changes the key, compaction sorts its output by the
/// new key and records that; the files it replaced kept the old key. Each
/// query is answered from what is recorded, never from the live spec alone.
#[tokio::test(flavor = "multi_thread")]
async fn sort_spec_change_then_compaction() {
    let env = setup().await;
    env.sort_by_ts_desc(NullOrder::NullsFirst).await;
    ingest_disjoint_sorted(&env, 2, 1_000).await;
    assert!(
        env.orders()
            .await
            .values()
            .all(|o| o.as_deref() == Some("1:desc:nulls_first"))
    );
    // `id` is field 3; ascending by it is ascending by `ts` here.
    env.writer()
        .await
        .set_sort_spec(
            env.table_id,
            &[SortField::column(0, "id", SortDirection::Asc, NullOrder::NullsLast)],
        )
        .unwrap();
    // Before compaction: the old key's rows still describe the files.
    for partitions in [1, 4] {
        assert_all_queries_correct(&env.ctx(partitions).await).await;
    }

    let (ctx, table) = table_handle(&env).await;
    table
        .with_write_options(ext_options())
        .merge_adjacent_files(&ctx.state(), MergeOptions::default())
        .await
        .unwrap();
    let orders = env.orders().await;
    assert_eq!(orders.len(), 1, "{orders:?}");
    assert_eq!(
        orders.values().next().unwrap().as_deref(),
        Some("3:asc:nulls_last")
    );
    let by_id = "SELECT ts, level, id FROM ducklake.main.logs ORDER BY id ASC LIMIT 7";
    for partitions in [1, 4] {
        let ctx = env.ctx(partitions).await;
        assert_all_queries_correct(&ctx).await;
        assert_matches_reference(&ctx, by_id).await;
        let p = plan(&ctx, by_id).await;
        assert!(!sorts(&p), "{p}");
        // The old key no longer describes any file: `ts DESC` sorts again.
        assert!(sorts(&plan(&ctx, TOP_N).await));
    }
}

/// A row describes the object at a path. A file whose name the caller picks
/// (`begin_write_to_path`) can be written again with other bytes after a
/// failed attempt, so the writer records only files it names itself.
#[tokio::test(flavor = "multi_thread")]
async fn caller_named_files_are_not_recorded() {
    let env = setup().await;
    env.sort_by_ts_desc(NullOrder::NullsFirst).await;
    let writer = DuckLakeTableWriter::new(
        Arc::new(env.writer().await),
        Arc::new(LocalFileSystem::new()),
    )
    .unwrap()
    .with_options(&ext_options());
    let dir = env.data_path().join("main").join("logs");
    let mut session = writer
        .begin_write_to_path(
            "main",
            "logs",
            schema().as_ref(),
            dir.to_str().unwrap(),
            "chosen.parquet".to_string(),
            WriteMode::Append,
        )
        .unwrap();
    session.write_batch(&batch(&desc(0..1_000))).unwrap();
    session.finish().await.unwrap();
    assert_eq!(sidecar_rows(&env).await, 0);

    // Recorded by the verify job once committed, like any foreign file.
    let (ctx, table) = table_handle(&env).await;
    let verified = table
        .verify_file_orders(&ctx.state(), VerifyOrderOptions::default())
        .await
        .unwrap();
    assert_eq!(verified.files_recorded, 1, "{verified:?}");
    assert_all_queries_correct(&env.ctx(4).await).await;
}
