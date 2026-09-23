//! A DuckLake scan reads the session's `datafusion.execution.parquet.*`
//! settings, the way a plain parquet scan does.
//!
//! `ParquetSource::new` starts from `TableParquetOptions::default()` and never
//! consults the session afterwards, so a source this crate builds has to be
//! handed the session's options explicitly or the whole `parquet` config section
//! is inert on a DuckLake table while working on a `ListingTable` over the very
//! same files.
//!
//! `max_in_list_size` is the setting these tests use, because it is the one that
//! fails silently: an `IN` list longer than it (20 by default) contributes no
//! bound to statistics pruning, the scan still reports a pruning predicate, and
//! the only visible symptom is that every row group matches. Raising it is the
//! documented way to keep a long key lookup selective, which is exactly the shape
//! of query — fetch N rows by id — that cannot afford a full scan.
//!
//! The pruning test pairs the raised setting with the default so it cannot pass
//! vacuously: the same fixture and the same query must prune under one and not
//! under the other. The plan-time half of the same setting — file pruning from
//! catalog statistics — is covered by
//! `table::tests::a_long_in_list_prunes_files_only_under_a_raised_cap`, because
//! the catalog-side SQL pre-filter already narrows the listing here and would
//! mask it.
//!
//! What reaches the scan is the session's `global` options and nothing else. The
//! `crypto` section beside them belongs to whatever `TableOptions` the embedder
//! installed, and an explicit decryption property there would win over the
//! per-file encryption factory this crate attaches, so the last test asserts it
//! is left behind. Every query-time source is built through
//! `session_parquet_source`, which `clippy.toml` enforces by disallowing
//! `ParquetSource::new` everywhere else — the change feeds build theirs inside
//! `TableChangesExec`/`DeletedRowsExec` at execution time, where no plan walk
//! reaches them.

#![cfg(all(feature = "write-sqlite", feature = "metadata-sqlite"))]

use std::sync::Arc;

use arrow::array::{Int64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::config::{ConfigFileDecryptionProperties, ConfigOptions, ParquetEncryptionOptions};
use datafusion::datasource::physical_plan::{FileScanConfig, ParquetSource};
use datafusion::datasource::source::DataSourceExec;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::prelude::*;
use object_store::local::LocalFileSystem;
use tempfile::TempDir;

use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeTableWriter, MetadataWriter, SqliteMetadataProvider,
    SqliteMetadataWriter,
};

/// Rows per file. At [`ROWS_PER_ROW_GROUP`] this spans ten row groups, so a
/// predicate confined to one of them has nine to prune.
const ROWS_PER_FILE: i64 = 20_000;

/// Rows per parquet row group. The parquet default is 1Mi rows, which would put
/// a whole file in one group and make every pruning assertion vacuous.
const ROWS_PER_ROW_GROUP: usize = 2_000;

/// Longer than DataFusion's default `max_in_list_size` of 20, so the list is
/// exactly what that setting decides the fate of.
const IN_LIST_LEN: i64 = 30;

/// A `max_in_list_size` no default produces, so reading it back off a plan can
/// only mean the session's value arrived there.
const RAISED: usize = 4096;

fn table_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, true)]))
}

fn batch(start: i64, rows: i64) -> RecordBatch {
    RecordBatch::try_new(
        table_schema(),
        vec![Arc::new(Int64Array::from((start..start + rows).collect::<Vec<_>>()))],
    )
    .unwrap()
}

/// Write `files` data files of [`ROWS_PER_FILE`] rows each, with disjoint
/// ascending `id` ranges — one commit per file, so the catalog records a
/// separate `min`/`max` for each.
async fn seed(temp: &TempDir, files: i64) {
    let data_path = temp.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();
    let writer = SqliteMetadataWriter::new_with_init(&format!(
        "sqlite:{}?mode=rwc",
        temp.path().join("test.db").display()
    ))
    .await
    .unwrap();
    writer.set_data_path(data_path.to_str().unwrap()).unwrap();
    let table_writer = DuckLakeTableWriter::new(
        Arc::new(writer),
        Arc::new(LocalFileSystem::new()) as Arc<dyn object_store::ObjectStore>,
    )
    .unwrap()
    .with_max_row_group_rows(ROWS_PER_ROW_GROUP);

    for file in 0..files {
        table_writer
            .append_table("main", "t", &[batch(file * ROWS_PER_FILE, ROWS_PER_FILE)])
            .await
            .unwrap();
    }
}

/// A context whose `max_in_list_size` is `max_in_list_size`, or DataFusion's
/// default when `None`.
async fn ctx_for(temp: &TempDir, max_in_list_size: Option<usize>) -> SessionContext {
    let provider =
        SqliteMetadataProvider::new(&format!("sqlite:{}", temp.path().join("test.db").display()))
            .await
            .unwrap();
    let mut cfg = ConfigOptions::new();
    if let Some(size) = max_in_list_size {
        cfg.execution.parquet.max_in_list_size = size;
    }
    let ctx = SessionContext::new_with_config(SessionConfig::from(cfg));
    ctx.register_catalog(
        "ducklake",
        Arc::new(DuckLakeCatalog::new(provider).unwrap()),
    );
    ctx
}

/// `IN (...)` over [`IN_LIST_LEN`] ids scattered through `range`, non-contiguous
/// so nothing folds them back into a range comparison.
fn in_list(range: std::ops::Range<i64>) -> String {
    let step = (range.end - range.start) / IN_LIST_LEN;
    let ids: Vec<String> = (0..IN_LIST_LEN)
        .map(|i| (range.start + i * step).to_string())
        .collect();
    format!(
        "SELECT id FROM ducklake.main.t WHERE id IN ({})",
        ids.join(", ")
    )
}

async fn analyze(ctx: &SessionContext, sql: &str) -> String {
    let batches = ctx
        .sql(&format!("EXPLAIN ANALYZE {sql}"))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    arrow::util::pretty::pretty_format_batches(&batches)
        .unwrap()
        .to_string()
}

/// Parse a `name=<total> total → <matched> matched` metric out of an
/// `EXPLAIN ANALYZE` blob.
fn pruning_metric(plan: &str, name: &str) -> Option<(usize, usize)> {
    let at = plan.find(&format!("{name}="))? + name.len() + 1;
    let rest = &plan[at..];
    let total: usize = rest
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect::<String>()
        .parse()
        .ok()?;
    let arrow = rest.find('→')? + '→'.len_utf8();
    let matched: usize = rest[arrow..]
        .trim_start()
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect::<String>()
        .parse()
        .ok()?;
    Some((total, matched))
}

/// The reader's row-group pruning answers to the session's setting.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_raised_max_in_list_size_prunes_row_groups() {
    let temp = TempDir::new().unwrap();
    seed(&temp, 1).await;
    // Every id in the list lives in the last of the ten row groups.
    let sql = in_list(18_000..20_000);

    let raised = analyze(&ctx_for(&temp, Some(RAISED)).await, &sql).await;
    let (total, matched) = pruning_metric(&raised, "row_groups_pruned_statistics")
        .unwrap_or_else(|| panic!("no row-group metric in:\n{raised}"));
    assert!(
        total > 1,
        "the fixture must span several row groups:\n{raised}"
    );
    assert_eq!(
        matched, 1,
        "every listed id is in the last row group, got {matched}/{total}:\n{raised}"
    );

    // The pairing that keeps the assertion above honest: at the default of 20
    // the same 30-entry list yields no bound and nothing is pruned, so the test
    // is measuring the setting rather than the fixture.
    let default = analyze(&ctx_for(&temp, None).await, &sql).await;
    let (total, matched) = pruning_metric(&default, "row_groups_pruned_statistics")
        .unwrap_or_else(|| panic!("no row-group metric in:\n{default}"));
    assert_eq!(
        matched, total,
        "a list longer than the default cap prunes no row group:\n{default}"
    );
}

/// Pruning on an `IN` list must not lose a row. The list is deliberately spread
/// across every row group and every file, so a bound applied too aggressively
/// shows up as a missing id rather than as a plan difference.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_raised_max_in_list_size_preserves_results() {
    let temp = TempDir::new().unwrap();
    seed(&temp, 4).await;
    let sql = format!("{} ORDER BY id", in_list(0..80_000));

    let ctx = ctx_for(&temp, Some(RAISED)).await;

    // The list spans every row group, but each entry only lands in one, so the
    // raised cap must still leave most of them behind. Without this the row
    // assertion below would hold just as well over an unpruned scan, and the
    // test would say nothing about the setting.
    let plan = analyze(&ctx, &sql).await;
    let (total, matched) = pruning_metric(&plan, "row_groups_pruned_statistics")
        .unwrap_or_else(|| panic!("no row-group metric in:\n{plan}"));
    assert!(
        matched < total,
        "the raised cap must prune something, got {matched}/{total}:\n{plan}"
    );

    let batches = ctx.sql(&sql).await.unwrap().collect().await.unwrap();
    let found: Vec<i64> = batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .values()
                .to_vec()
        })
        .collect();

    let step = 80_000 / IN_LIST_LEN;
    let expected: Vec<i64> = (0..IN_LIST_LEN).map(|i| i * step).collect();
    assert_eq!(found, expected);
}

/// A scan's `ParquetSource` carries the session's options, read back off the
/// plan rather than inferred from a result.
///
/// The pruning tests above show the setting reaching the reader; this shows it
/// arriving, which is what a new `ParquetSource` site would get wrong.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_scan_carries_the_session_parquet_options() {
    let temp = TempDir::new().unwrap();
    seed(&temp, 1).await;
    let ctx = ctx_for(&temp, Some(RAISED)).await;

    let plan = ctx
        .sql("SELECT id FROM ducklake.main.t")
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();

    let mut found = Vec::new();
    collect_parquet_in_list_sizes(&plan, &mut found);
    assert_eq!(
        found,
        vec![RAISED],
        "the scan must build its ParquetSource with the session's options"
    );
}

/// `max_in_list_size` from every `ParquetSource` in `plan`.
fn collect_parquet_in_list_sizes(plan: &Arc<dyn ExecutionPlan>, out: &mut Vec<usize>) {
    if let Some(source) = plan
        .downcast_ref::<DataSourceExec>()
        .and_then(|exec| exec.data_source().downcast_ref::<FileScanConfig>())
        .and_then(|config| config.file_source().downcast_ref::<ParquetSource>())
    {
        out.push(source.table_parquet_options().global.max_in_list_size);
    }
    for child in plan.children() {
        collect_parquet_in_list_sizes(child, out);
    }
}

/// The embedder's `crypto` section stays out of a DuckLake scan.
///
/// `TableOptions` is the embedder's, and its `crypto` section is read by whoever
/// installed it — a `ListingTable` over their own encrypted files, say. A
/// DuckLake file's key comes from the catalog instead, through the per-file
/// encryption factory the scan attaches, and an explicit decryption property in
/// the session's `TableOptions` wins over that factory. Taking the whole of
/// `default_table_options().parquet` rather than its `global` section would
/// therefore point every DuckLake scan at the embedder's footer key.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_scan_leaves_the_embedder_crypto_options_behind() {
    let temp = TempDir::new().unwrap();
    seed(&temp, 1).await;
    let ctx = ctx_for(&temp, Some(RAISED)).await;
    {
        let state = ctx.state_ref();
        let mut state = state.write();
        let crypto = &mut state.table_options_mut().parquet.crypto;
        crypto.file_decryption = Some(ConfigFileDecryptionProperties {
            // b"0123456789012345"
            footer_key_as_hex: "30313233343536373839303132333435".to_string(),
            ..Default::default()
        });
        crypto.factory_id = Some("an_embedder_factory".to_string());
    }

    let plan = ctx
        .sql("SELECT id FROM ducklake.main.t")
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();

    let mut found = Vec::new();
    collect_parquet_crypto(&plan, &mut found);
    assert_eq!(found.len(), 1, "the fixture is a single scan");
    assert_eq!(
        found[0],
        ParquetEncryptionOptions::default(),
        "the session's crypto section must not reach a DuckLake source"
    );
}

/// The `crypto` section of every `ParquetSource` in `plan`.
fn collect_parquet_crypto(plan: &Arc<dyn ExecutionPlan>, out: &mut Vec<ParquetEncryptionOptions>) {
    if let Some(source) = plan
        .downcast_ref::<DataSourceExec>()
        .and_then(|exec| exec.data_source().downcast_ref::<FileScanConfig>())
        .and_then(|config| config.file_source().downcast_ref::<ParquetSource>())
    {
        out.push(source.table_parquet_options().crypto.clone());
    }
    for child in plan.children() {
        collect_parquet_crypto(child, out);
    }
}
