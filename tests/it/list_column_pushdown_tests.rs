//! A list column must not cost the rest of the table its filter pushdown.
//!
//! Three layers spell a list's element field three different ways and none of
//! them is wrong: the catalog schema says `item`, a name-mapped physical schema
//! says `list`, and arrow-rs reports whatever the parquet file records — which
//! for a spec-conformant writer, this crate's own included, is `element`.
//! `ColumnRenameExec` sits between the reader and the catalog schema to absorb
//! exactly that kind of difference, and it decides pushdown for the whole
//! schema at once. So a single list column spelled one way in the file and
//! another in the catalog used to make the node refuse every parent filter,
//! including one on a plain `BIGINT` that has nothing to do with the list: the
//! predicate stayed above the scan, no row group was pruned, and the query read
//! the table.
//!
//! The scan now reads each file under the catalog's spelling, the way official
//! DuckLake normalizes its reader's columns, so the two sides of the node agree
//! and the comparison is strict again.
//!
//! The fixture is therefore a table with one list column and one scalar column,
//! and the assertions are about the SCALAR predicate. They read the parquet
//! reader's own `row_groups_pruned_statistics` counter rather than the plan
//! text, because a plan can name a predicate it never manages to use. A second
//! fixture adds an identity partition key to the same two columns: partition
//! pruning picks the file before the scan opens it and the rename decides what
//! the scan then does with it, and no other test runs the two in one plan.
//!
//! Every query below projects the list column. Projecting it away is what made
//! the original report confusing — the node's output schema then holds no list,
//! the comparison passes, and pushdown works — so a test that read only `id`
//! would pass on either side of the fix.

#![cfg(all(feature = "write-sqlite", feature = "metadata-sqlite"))]

use std::sync::Arc;

use arrow::array::{Array, Float32Array, Int64Array, ListArray, RecordBatch};
use arrow::buffer::{OffsetBuffer, ScalarBuffer};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::prelude::*;
use object_store::local::LocalFileSystem;
use tempfile::TempDir;

use datafusion_ducklake::partition::PartitionTransform;
use datafusion_ducklake::{
    ColumnDef, DuckLakeCatalog, DuckLakeTableWriter, MetadataWriter, SqliteMetadataProvider,
    SqliteMetadataWriter, WriteMode,
};

/// Rows in the fixture. Enough to span several row groups at
/// [`ROWS_PER_ROW_GROUP`], so pruning has something to bite on.
const ROWS: i64 = 20_000;

/// Rows per parquet row group. The parquet default is 1Mi rows, which would put
/// the whole fixture in one group and make every pruning assertion vacuous.
const ROWS_PER_ROW_GROUP: usize = 2_000;

/// Rows written at a time, and — in the partitioned fixture — rows per
/// partition, so each partition is a data file of several row groups.
const ROWS_PER_PARTITION: i64 = 5_000;

/// `id BIGINT, emb FLOAT[]`, with the list element named `element` — what the
/// parquet LIST convention prescribes and what arrow-rs derives back from the
/// file, while the catalog schema names the same field `item`.
fn table_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, true),
        Field::new(
            "emb",
            DataType::List(Arc::new(Field::new("element", DataType::Float32, true))),
            true,
        ),
    ]))
}

/// `rows` rows starting at `start`: an ascending `id` and a two-element vector.
fn batch(start: i64, rows: i64) -> RecordBatch {
    let ids: Vec<i64> = (start..start + rows).collect();
    let values: Vec<f32> = ids
        .iter()
        .flat_map(|&i| [i as f32, i as f32 + 0.5])
        .collect();
    let offsets: Vec<i32> = (0..=rows as i32).map(|i| i * 2).collect();
    let emb = ListArray::new(
        Arc::new(Field::new("element", DataType::Float32, true)),
        OffsetBuffer::new(ScalarBuffer::from(offsets)),
        Arc::new(Float32Array::from(values)),
        None,
    );
    RecordBatch::try_new(
        table_schema(),
        vec![Arc::new(Int64Array::from(ids)), Arc::new(emb)],
    )
    .unwrap()
}

/// A fresh writable SQLite catalog under `temp`, with its data path prepared.
async fn writer_for(temp: &TempDir) -> SqliteMetadataWriter {
    let data_path = temp.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();
    let writer = SqliteMetadataWriter::new_with_init(&format!(
        "sqlite:{}?mode=rwc",
        temp.path().join("test.db").display()
    ))
    .await
    .unwrap();
    writer.set_data_path(data_path.to_str().unwrap()).unwrap();
    writer
}

/// The writer both fixtures write their parquet through, at
/// [`ROWS_PER_ROW_GROUP`].
fn table_writer(writer: SqliteMetadataWriter) -> DuckLakeTableWriter {
    DuckLakeTableWriter::new(
        Arc::new(writer),
        Arc::new(LocalFileSystem::new()) as Arc<dyn object_store::ObjectStore>,
    )
    .unwrap()
    .with_max_row_group_rows(ROWS_PER_ROW_GROUP)
}

/// Write the fixture as a DuckLake table in a fresh catalog.
async fn seed(temp: &TempDir) {
    let writer = writer_for(temp).await;
    let batches: Vec<RecordBatch> = (0..ROWS / ROWS_PER_PARTITION)
        .map(|chunk| batch(chunk * ROWS_PER_PARTITION, ROWS_PER_PARTITION))
        .collect();
    table_writer(writer)
        .write_table("main", "t", &batches)
        .await
        .unwrap();
}

/// `id BIGINT, part BIGINT, emb FLOAT[]` — [`table_schema`] with an identity
/// partition key beside it.
fn partitioned_table_schema() -> Arc<Schema> {
    let scalar = table_schema();
    Arc::new(Schema::new(vec![
        scalar.field(0).clone(),
        Field::new("part", DataType::Int64, true),
        scalar.field(1).clone(),
    ]))
}

/// One partition's rows: [`batch`] with the `part` key its ids fall under.
fn partitioned_batch(start: i64, rows: i64) -> RecordBatch {
    let scalar = batch(start, rows);
    let part = Int64Array::from(vec![start / ROWS_PER_PARTITION; rows as usize]);
    RecordBatch::try_new(
        partitioned_table_schema(),
        vec![Arc::clone(scalar.column(0)), Arc::new(part), Arc::clone(scalar.column(1))],
    )
    .unwrap()
}

/// Write the same rows partitioned by `part`, one data file per partition.
///
/// The spec has to be live before any data lands, so the table is created empty,
/// given the spec, and only then appended to.
async fn seed_partitioned(temp: &TempDir) {
    let writer = writer_for(temp).await;
    let columns: Vec<ColumnDef> = partitioned_table_schema()
        .fields()
        .iter()
        .map(|field| {
            ColumnDef::from_arrow(field.name(), field.data_type(), field.is_nullable()).unwrap()
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
            // Every field id, not just the top-level ones: `emb`'s element is a
            // catalog column node of its own.
            &setup.field_ids,
        )
        .unwrap();
    writer
        .set_partition_spec(
            setup.table_id,
            &[("part".to_string(), PartitionTransform::Identity)],
        )
        .unwrap();

    let batches: Vec<RecordBatch> = (0..ROWS / ROWS_PER_PARTITION)
        .map(|partition| partitioned_batch(partition * ROWS_PER_PARTITION, ROWS_PER_PARTITION))
        .collect();
    table_writer(writer)
        .append_table("main", "t", &batches)
        .await
        .unwrap();
}

async fn ctx_for(temp: &TempDir) -> SessionContext {
    let provider =
        SqliteMetadataProvider::new(&format!("sqlite:{}", temp.path().join("test.db").display()))
            .await
            .unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog(
        "ducklake",
        Arc::new(DuckLakeCatalog::new(provider).unwrap()),
    );
    ctx
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

/// Parse a plain `name=<count>` counter out of an `EXPLAIN ANALYZE` blob,
/// scaled by the unit suffix a large count is rendered with (`2.00 K`).
fn counter_metric(plan: &str, name: &str) -> Option<f64> {
    let at = plan.find(&format!("{name}="))? + name.len() + 1;
    let rest = &plan[at..];
    let digits: String = rest
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '.')
        .collect();
    let value: f64 = digits.parse().ok()?;
    let scale = match rest[digits.len()..]
        .strip_prefix(' ')
        .and_then(|rest| rest.chars().next())
    {
        Some('K') => 1e3,
        Some('M') => 1e6,
        Some('G') => 1e9,
        _ => 1.0,
    };
    Some(value * scale)
}

/// The defect itself: a selective predicate on the scalar column prunes row
/// groups even though the list column is projected alongside it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_scalar_predicate_prunes_row_groups_beside_a_list_column() {
    let temp = TempDir::new().unwrap();
    seed(&temp).await;
    let ctx = ctx_for(&temp).await;

    let plan = analyze(
        &ctx,
        "SELECT id, emb FROM ducklake.main.t WHERE id >= 19000",
    )
    .await;
    let (total, matched) = pruning_metric(&plan, "row_groups_pruned_statistics")
        .unwrap_or_else(|| panic!("no row-group metric in:\n{plan}"));
    assert!(
        total > 1,
        "the fixture must span several row groups, or pruning proves nothing \
         (got {total}):\n{plan}"
    );
    // `id >= 19000` selects the last 1000 of 20000 rows, i.e. one 2000-row group.
    assert_eq!(
        matched, 1,
        "only the last row group can hold id >= 19000, got {matched}/{total}:\n{plan}"
    );
}

/// The same table, same predicate, with the list column projected away. This
/// path always pruned — the node's output schema holds no list, so the
/// comparison that used to fail never ran — and it is here so a regression shows
/// up as "the list column is what breaks it" rather than as a bare failure.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_scalar_predicate_prunes_row_groups_without_the_list_column() {
    let temp = TempDir::new().unwrap();
    seed(&temp).await;
    let ctx = ctx_for(&temp).await;

    let plan = analyze(&ctx, "SELECT id FROM ducklake.main.t WHERE id >= 19000").await;
    let (total, matched) = pruning_metric(&plan, "row_groups_pruned_statistics")
        .unwrap_or_else(|| panic!("no row-group metric in:\n{plan}"));
    assert!(
        total > 1,
        "the fixture must span several row groups:\n{plan}"
    );
    assert_eq!(matched, 1, "got {matched}/{total}:\n{plan}");
}

/// Read `(id, vector)` out of an `id, emb` result, in the order it arrives.
fn id_and_vectors(batches: &[RecordBatch]) -> Vec<(i64, Vec<f32>)> {
    let mut rows: Vec<(i64, Vec<f32>)> = Vec::new();
    for batch in batches {
        let ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let embeddings = batch
            .column(1)
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        for row in 0..batch.num_rows() {
            let values = embeddings.value(row);
            let values = values.as_any().downcast_ref::<Float32Array>().unwrap();
            rows.push((ids.value(row), values.values().to_vec()));
        }
    }
    rows
}

/// What [`batch`] wrote for every id from `first` to the end of the fixture.
fn expected_tail(first: i64) -> Vec<(i64, Vec<f32>)> {
    (first..ROWS)
        .map(|id| (id, vec![id as f32, id as f32 + 0.5]))
        .collect()
}

/// A pushed-down predicate must not change what comes back, list column
/// included: every surviving row, its whole vector, and the type the provider
/// advertises for it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn pushdown_beside_a_list_column_preserves_results() {
    let temp = TempDir::new().unwrap();
    seed(&temp).await;
    let ctx = ctx_for(&temp).await;
    let sql = "SELECT id, emb FROM ducklake.main.t WHERE id >= 19995 ORDER BY id";

    // The predicate has to reach the reader, or the values below prove only that
    // an unpushed scan reads correctly.
    let plan = analyze(&ctx, sql).await;
    let (total, matched) = pruning_metric(&plan, "row_groups_pruned_statistics")
        .unwrap_or_else(|| panic!("no row-group metric in:\n{plan}"));
    assert_eq!(
        matched, 1,
        "only the last row group can hold id >= 19995, got {matched}/{total}:\n{plan}"
    );

    let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
    assert_eq!(id_and_vectors(&batches), expected_tail(19_995));

    // The provider advertises the catalog spelling whatever the file says.
    let schema = batches[0].schema();
    let DataType::List(element) = schema.field(1).data_type() else {
        panic!("emb must still be a list");
    };
    assert_eq!(element.name(), "item");
}

/// The element name a scan reads under is the catalog's, so the physical plan
/// under the rename already carries the type the provider advertises.
///
/// This is the normalization itself, asserted where it happens rather than
/// through its consequences: official DuckLake brings its reader's columns onto
/// one spelling, and this is where a file's columns arrive on it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_scan_reads_a_list_under_the_catalog_element_name() {
    let temp = TempDir::new().unwrap();
    seed(&temp).await;
    let ctx = ctx_for(&temp).await;

    let plan = ctx
        .sql("SELECT id, emb FROM ducklake.main.t")
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();

    let mut scans = Vec::new();
    collect_scans(&plan, &mut scans);
    assert_eq!(scans.len(), 1, "the fixture is a single-file scan");
    let DataType::List(element) = scans[0].field(1).data_type() else {
        panic!("emb must be a list in the scan schema");
    };
    assert_eq!(
        element.name(),
        "item",
        "the file spells the element `element`; the scan must read it under the \
         catalog's name"
    );
}

/// Collect the output schema of every `DataSourceExec` in `plan`.
fn collect_scans(
    plan: &Arc<dyn datafusion::physical_plan::ExecutionPlan>,
    out: &mut Vec<SchemaRef>,
) {
    if plan.name() == "DataSourceExec" {
        out.push(plan.schema());
    }
    for child in plan.children() {
        collect_scans(child, out);
    }
}

/// A predicate that reads the list's own type must mean the same thing whether
/// it is evaluated inside the reader or above the scan.
///
/// `arrow_typeof` renders a list's element name and the metadata its element
/// field carries, and the `IS NULL` disjunct is what makes the parquet reader
/// accept the whole predicate for row-level evaluation (DataFusion's
/// `supports_list_predicates` in `supported_predicates.rs`, applied by
/// `projection_read_plan`'s `PushdownChecker`), so with pushdown on this runs
/// below the rename. Reading the file's spelling there — or the field ids the
/// file stamps on its elements — would make the comparison false and drop every
/// row inside the reader.
///
/// A list is the only nested type this reaches: the same checker refuses a
/// whole-struct column reference outright, so a struct's children keep the ids
/// the file records without any predicate being able to observe them.
///
/// The value assertion alone does not discriminate: a scan that pushes nothing
/// answers both settings the same way. So it is paired with the reader's own
/// row-filter counters, which exist only for a predicate that reached the
/// reader.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_pushed_down_predicate_reads_the_catalog_element_name() {
    let temp = TempDir::new().unwrap();
    seed(&temp).await;
    let sql = "SELECT id, emb FROM ducklake.main.t \
               WHERE (emb IS NULL OR arrow_typeof(emb) = 'List(Float32)') \
                 AND id >= 19995";

    for pushdown in [false, true] {
        let ctx = ctx_for(&temp).await;
        ctx.sql(&format!(
            "SET datafusion.execution.parquet.pushdown_filters = {pushdown}"
        ))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

        let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
        let rows: usize = batches.iter().map(RecordBatch::num_rows).sum();
        assert_eq!(
            rows,
            5,
            "ids 19995..19999, with parquet filter pushdown {}",
            if pushdown {
                "on"
            } else {
                "off"
            }
        );

        if !pushdown {
            continue;
        }

        // `pushdown_rows_matched` and `pushdown_rows_pruned` count the rows the
        // reader's own row filter admitted and dropped, so they move only for a
        // predicate that got past the rename and into the reader. Without them
        // the loop above proves nothing: a scan that pushes no predicate at all
        // answers both settings the same way.
        let plan = analyze(&ctx, sql).await;
        let matched = counter_metric(&plan, "pushdown_rows_matched")
            .unwrap_or_else(|| panic!("no row-filter metric in:\n{plan}"));
        let pruned = counter_metric(&plan, "pushdown_rows_pruned")
            .unwrap_or_else(|| panic!("no row-filter metric in:\n{plan}"));
        assert_eq!(
            matched, 5.0,
            "the reader's row filter must admit ids 19995..19999, got \
             {matched} matched / {pruned} pruned:\n{plan}"
        );
        assert!(
            pruned > 0.0,
            "the reader's row filter must drop the rest of the surviving row \
             group, got {matched} matched / {pruned} pruned:\n{plan}"
        );
    }
}

/// The same predicate on a table that is partitioned as well as list-bearing.
///
/// Partition pruning narrows the file list from the catalog, before the scan
/// opens anything; the rename decides what the scan does with the file it then
/// opens. Neither sweep meets the other — the partitioned fixtures elsewhere
/// carry no nested column, and the fixture above carries no spec — so this is
/// where the two run in one plan.
///
/// Same shape as [`a_pushed_down_predicate_reads_the_catalog_element_name`]: the
/// values must be untouched at both `pushdown_filters` settings, and the
/// reader's row-filter counters must show the predicate actually reached it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_pushed_down_predicate_reads_the_catalog_element_name_when_partitioned() {
    let temp = TempDir::new().unwrap();
    seed_partitioned(&temp).await;
    let sql = "SELECT id, emb FROM ducklake.main.t \
               WHERE (emb IS NULL OR arrow_typeof(emb) = 'List(Float32)') \
                 AND part = 3 AND id >= 19995 ORDER BY id";

    for pushdown in [false, true] {
        let ctx = ctx_for(&temp).await;
        ctx.sql(&format!(
            "SET datafusion.execution.parquet.pushdown_filters = {pushdown}"
        ))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

        let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
        assert_eq!(
            id_and_vectors(&batches),
            expected_tail(19_995),
            "ids 19995..19999 and their vectors, with parquet filter pushdown {}",
            if pushdown {
                "on"
            } else {
                "off"
            }
        );

        if !pushdown {
            continue;
        }

        let plan = analyze(&ctx, sql).await;
        // The spec has to be live, or this is the unpartitioned fixture again:
        // `part = 3` leaves one 5000-row data file of three row groups, where
        // the whole table would be ten.
        let (groups, _) = pruning_metric(&plan, "row_groups_pruned_statistics")
            .unwrap_or_else(|| panic!("no row-group metric in:\n{plan}"));
        assert_eq!(
            groups, 3,
            "the scan must read one partition's file, got {groups} row \
             groups:\n{plan}"
        );

        let matched = counter_metric(&plan, "pushdown_rows_matched")
            .unwrap_or_else(|| panic!("no row-filter metric in:\n{plan}"));
        let pruned = counter_metric(&plan, "pushdown_rows_pruned")
            .unwrap_or_else(|| panic!("no row-filter metric in:\n{plan}"));
        assert_eq!(
            matched, 5.0,
            "the reader's row filter must admit ids 19995..19999, got \
             {matched} matched / {pruned} pruned:\n{plan}"
        );
        assert!(
            pruned > 0.0,
            "the reader's row filter must drop the rest of the surviving row \
             group, got {matched} matched / {pruned} pruned:\n{plan}"
        );
    }
}
