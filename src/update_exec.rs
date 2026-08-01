//! DuckLake `UPDATE` execution plan.
//!
//! [`DuckLakeUpdateExec`] is the physical operator DataFusion's planner lowers
//! `UPDATE t SET col = expr ... [WHERE p]` onto (via
//! [`TableProvider::update`](datafusion::catalog::TableProvider::update)). All
//! work is deferred to execute time so planning / `EXPLAIN` never mutate data:
//!
//! 1. For each source data file that may hold matching rows, collect its
//!    pre-built positional scan, select the rows matching the predicate, apply
//!    the assignments, and produce rewritten row versions that RETAIN each row's
//!    original rowid (written into NEW data file(s) that embed the rowid column,
//!    so lineage survives the rewrite). On a partitioned table each rewritten row
//!    is routed by its OWN post-assignment key values, so an assignment that
//!    changes a partition key MOVES the row to its new partition; the rewrite then
//!    spans one file per output partition.
//! 2. Resolve, per source file, the cumulative positional delete masking the old
//!    row versions (superseded rows unioned with any already-deleted rows).
//! 3. Commit ATOMICALLY: every appended data file AND every positional delete land
//!    in ONE snapshot via
//!    [`MetadataWriter::register_data_file_with_deletes`] (or, when the rewrite
//!    produced several files,
//!    [`MetadataWriter::register_data_files_with_deletes`]) — driven by
//!    `TableWriteSession::finish_with_deletes`.
//! 4. Yield a single row `count: UInt64` = rows updated.
//!
//! Limitations (shared with [`DuckLakeInsertExec`](crate::insert_exec)):
//! collects matched rows into memory before writing; runs in a single DataFusion
//! output partition.
//!
//! # Session lifecycle (important)
//!
//! A [`DuckLakeCatalog`](crate::DuckLakeCatalog) pins its snapshot at creation
//! and never refreshes it, so a `SessionContext` observes ONE catalog generation
//! for its whole lifetime. An `UPDATE` commits a new snapshot, but the same
//! session keeps reading the old one. Consequences:
//!
//! - A second `UPDATE` in the same session that re-touches a data file modified
//!   by an earlier `UPDATE` (in that same session) aborts with a
//!   [`Conflict`](crate::DuckLakeError::Conflict): it resolves against the pinned
//!   (pre-update) view, so the atomic commit's compare-and-swap disagrees with
//!   the live catalog. This is the same guard that (correctly) rejects a
//!   genuinely concurrent writer, so it is safe (the first update is preserved),
//!   just not retryable in-session.
//! - A `SELECT` after an `UPDATE`/`INSERT` in the same session returns the
//!   pre-mutation rows; a just-inserted row cannot be updated in the same
//!   session (it is invisible to the pinned snapshot).
//!
//! To perform multiple mutations, re-open the catalog (or create a fresh
//! `SessionContext`) between statements so it binds to the latest snapshot.

use std::fmt::{self, Debug};
use std::sync::Arc;

use arrow::array::{ArrayRef, RecordBatch, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::execution::object_store::ObjectStoreUrl;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::{EquivalenceProperties, Partitioning, PhysicalExpr};
use datafusion::physical_plan::apply_expression_roots;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use futures::stream::{self, TryStreamExt};

use crate::compaction::sorted_rewrite_batches;
use crate::metadata_writer::{DeleteFileEntry, MetadataWriter, WriteMode};
use crate::table::{DuckLakeTable, UpdateSourceScan};
use crate::table_writer::{DuckLakeTableWriter, validate_not_null_batches};

/// Schema for the output of update operations (count of rows updated).
fn make_update_count_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new(
        "count",
        DataType::UInt64,
        false,
    )]))
}

/// Execution plan that applies an `UPDATE` to a DuckLake table.
pub struct DuckLakeUpdateExec {
    /// Read-only handle to the target table, used to turn each source file's
    /// collected scan batches into rewritten rows at execute time.
    table: Arc<DuckLakeTable>,
    /// Metadata writer for the atomic append-with-deletes commit.
    writer: Arc<dyn MetadataWriter>,
    schema_name: String,
    table_name: String,
    base_snapshot: i64,
    /// Per-source-file positional read plans (built at plan time).
    scans: Vec<UpdateSourceScan>,
    /// `(physical_column_index, new_value_expr)` for each assigned column.
    assignments: Vec<(usize, Arc<dyn PhysicalExpr>)>,
    /// AND of the WHERE predicates, or `None` to update all rows.
    predicate: Option<Arc<dyn PhysicalExpr>>,
    object_store_url: Arc<ObjectStoreUrl>,
    cache: Arc<PlanProperties>,
}

impl DuckLakeUpdateExec {
    /// Create a new `DuckLakeUpdateExec`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        table: Arc<DuckLakeTable>,
        writer: Arc<dyn MetadataWriter>,
        schema_name: String,
        table_name: String,
        base_snapshot: i64,
        scans: Vec<UpdateSourceScan>,
        assignments: Vec<(usize, Arc<dyn PhysicalExpr>)>,
        predicate: Option<Arc<dyn PhysicalExpr>>,
        object_store_url: Arc<ObjectStoreUrl>,
    ) -> Self {
        let cache = Self::compute_properties();
        Self {
            table,
            writer,
            schema_name,
            table_name,
            base_snapshot,
            scans,
            assignments,
            predicate,
            object_store_url,
            cache,
        }
    }

    fn compute_properties() -> Arc<PlanProperties> {
        Arc::new(PlanProperties::new(
            EquivalenceProperties::new(make_update_count_schema()),
            Partitioning::UnknownPartitioning(1),
            datafusion::physical_plan::execution_plan::EmissionType::Final,
            datafusion::physical_plan::execution_plan::Boundedness::Bounded,
        ))
    }
}

impl Debug for DuckLakeUpdateExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DuckLakeUpdateExec")
            .field("schema_name", &self.schema_name)
            .field("table_name", &self.table_name)
            .field("source_files", &self.scans.len())
            .field("assignments", &self.assignments.len())
            .field("has_predicate", &self.predicate.is_some())
            .finish_non_exhaustive()
    }
}

impl DisplayAs for DuckLakeUpdateExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        match t {
            DisplayFormatType::Default
            | DisplayFormatType::Verbose
            | DisplayFormatType::TreeRender => {
                write!(
                    f,
                    "DuckLakeUpdateExec: schema={}, table={}, assignments={}, where={}",
                    self.schema_name,
                    self.table_name,
                    self.assignments.len(),
                    if self.predicate.is_some() {
                        "yes"
                    } else {
                        "no"
                    }
                )
            },
        }
    }
}

impl ExecutionPlan for DuckLakeUpdateExec {
    fn apply_expressions(
        &self,
        f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> DataFusionResult<TreeNodeRecursion>,
    ) -> DataFusionResult<TreeNodeRecursion> {
        let assignments = self.assignments.iter().map(|(_, expr)| expr);
        apply_expression_roots(assignments.chain(self.predicate.iter()), f)
    }

    fn name(&self) -> &str {
        "DuckLakeUpdateExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.cache
    }

    /// No DataFusion children: the per-file source scans are internal and are
    /// executed directly at execute time, so the optimizer treats this as a
    /// leaf and never rewrites (e.g. repartitions) those scans.
    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        if !children.is_empty() {
            return Err(DataFusionError::Internal(
                "DuckLakeUpdateExec has no children".to_string(),
            ));
        }
        Ok(self)
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(DataFusionError::Internal(format!(
                "DuckLakeUpdateExec only supports partition 0, got {partition}"
            )));
        }

        let table = Arc::clone(&self.table);
        let writer = Arc::clone(&self.writer);
        let schema_name = self.schema_name.clone();
        let table_name = self.table_name.clone();
        let base_snapshot = self.base_snapshot;
        let scans = self.scans.clone();
        let assignments = self.assignments.clone();
        let predicate = self.predicate.clone();
        let object_store_url = self.object_store_url.clone();
        let output_schema = make_update_count_schema();

        let stream = stream::once(async move {
            let object_store = context
                .runtime_env()
                .object_store(object_store_url.as_ref())?;
            let table_writer = DuckLakeTableWriter::new(writer, object_store)
                .map(|writer| writer.with_options(table.write_options()))
                .map_err(|e| DataFusionError::External(Box::new(e)))?;

            // Rewrite each source file's matching rows and author its cumulative
            // positional delete. Delete parquet files are uploaded here; the
            // catalog commit that makes them visible happens once, atomically,
            // below — so a failure before the commit leaves the live snapshot
            // untouched (only orphan objects, cleaned by maintenance).
            let mut updated_batches: Vec<RecordBatch> = Vec::new();
            let mut pending_deletes = Vec::new();
            let mut total_updated: u64 = 0;
            let physical_schema = table.physical_schema();

            for scan in &scans {
                let batches =
                    datafusion::physical_plan::collect(Arc::clone(&scan.scan), context.clone())
                        .await?;
                let out = table.apply_update_to_batches(
                    scan,
                    &batches,
                    predicate.as_ref(),
                    &assignments,
                )?;
                if out.matched_count == 0 {
                    continue;
                }
                total_updated += out.matched_count as u64;
                updated_batches.extend(out.updated_batches);
                pending_deletes.push((
                    scan.data_file_id,
                    scan.delete_file_id,
                    scan.source_path.clone(),
                    out.cumulative_positions,
                ));
            }

            validate_not_null_batches(physical_schema.as_ref(), &updated_batches)
                .map_err(|e| DataFusionError::External(Box::new(e)))?;

            let mut delete_entries: Vec<DeleteFileEntry> =
                Vec::with_capacity(pending_deletes.len());
            for (data_file_id, expected_prev_delete_file, source_path, positions) in pending_deletes
            {
                let delete_info = table_writer
                    .write_delete_file(&schema_name, &table_name, &source_path, &positions)
                    .await
                    .map_err(|e| DataFusionError::External(Box::new(e)))?;
                delete_entries.push(DeleteFileEntry {
                    data_file_id,
                    expected_prev_delete_file,
                    delete: delete_info,
                });
            }

            // No matching rows: genuine no-op, publish nothing.
            if total_updated == 0 {
                let count: ArrayRef = Arc::new(UInt64Array::from(vec![0u64]));
                return Ok(RecordBatch::try_new(output_schema, vec![count])?);
            }

            // Append the rewritten rows (embedding their original rowids) AND
            // apply every positional delete in ONE snapshot.
            let sort_spec = table
                .live_sort_spec()
                .map_err(|e| DataFusionError::External(Box::new(e)))?;
            let ordering = crate::compaction::compaction_ordering(
                physical_schema.as_ref(),
                sort_spec.as_ref(),
            )
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
            let mut updated_batches =
                sorted_rewrite_batches(Arc::clone(&context), updated_batches, ordering.as_ref())
                    .map_err(|e| DataFusionError::External(Box::new(e)))?;
            let mut session = table_writer
                .begin_write_with_embedded_rowid(
                    &schema_name,
                    &table_name,
                    physical_schema.as_ref(),
                    WriteMode::Append,
                )
                .map(|session| session.with_base_snapshot_id(base_snapshot))
                .map_err(|e| DataFusionError::External(Box::new(e)))?;
            while let Some(batch) = updated_batches.try_next().await? {
                session
                    .write_batch(&batch)
                    .map_err(|e| DataFusionError::External(Box::new(e)))?;
            }
            session
                .finish_with_deletes(&delete_entries)
                .await
                .map_err(|e| DataFusionError::External(Box::new(e)))?;

            let count: ArrayRef = Arc::new(UInt64Array::from(vec![total_updated]));
            Ok(RecordBatch::try_new(output_schema, vec![count])?)
        });

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            make_update_count_schema(),
            stream.map_err(|e: DataFusionError| e),
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_update_count_schema() {
        let schema = make_update_count_schema();
        assert_eq!(schema.fields().len(), 1);
        assert_eq!(schema.field(0).name(), "count");
        assert_eq!(schema.field(0).data_type(), &DataType::UInt64);
    }

    /// A keyed `UPDATE` hands its predicate to each source file's reader, so a
    /// multi-row-group file decodes only the row groups that can match.
    #[cfg(all(feature = "write-sqlite", feature = "metadata-sqlite"))]
    mod pruning {
        use std::sync::Arc;

        use arrow::array::{Float64Array, Int32Array, RecordBatch};
        use arrow::datatypes::{DataType, Field, Schema};
        use datafusion::physical_plan::{collect, displayable};
        use datafusion::prelude::SessionContext;
        use object_store::local::LocalFileSystem;

        use super::super::DuckLakeUpdateExec;
        use crate::metadata_writer::MetadataWriter;
        use crate::{
            DuckLakeCatalog, DuckLakeTableWriter, SqliteMetadataProvider, SqliteMetadataWriter,
        };

        const ROWS: i32 = 40;
        const ROW_GROUP: usize = 4;

        /// One data file of `ROWS` rows `(id, score)` in `ROW_GROUP`-row groups.
        async fn seed(dir: &tempfile::TempDir) -> String {
            let conn = format!("sqlite:{}?mode=rwc", dir.path().join("t.db").display());
            let data = dir.path().join("data");
            std::fs::create_dir_all(&data).unwrap();
            let writer = SqliteMetadataWriter::new_with_init(&conn).await.unwrap();
            writer.set_data_path(data.to_str().unwrap()).unwrap();
            let schema = Arc::new(Schema::new(vec![
                Field::new("id", DataType::Int32, false),
                Field::new("score", DataType::Float64, false),
            ]));
            let batch = RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(Int32Array::from_iter_values(0..ROWS)),
                    Arc::new(Float64Array::from_iter_values((0..ROWS).map(f64::from))),
                ],
            )
            .unwrap();
            DuckLakeTableWriter::new(Arc::new(writer), Arc::new(LocalFileSystem::new()))
                .unwrap()
                .with_max_row_group_rows(ROW_GROUP)
                .write_table("main", "t", &[batch])
                .await
                .unwrap();
            conn
        }

        /// Plan `sql` (an UPDATE) and return, per source file, its scan's
        /// display and the number of rows the scan yields.
        async fn source_scans(conn: &str, sql: &str) -> Vec<(String, usize)> {
            let provider = SqliteMetadataProvider::new(conn).await.unwrap();
            let writer = SqliteMetadataWriter::new(conn).await.unwrap();
            let catalog =
                DuckLakeCatalog::with_writer(Arc::new(provider), Arc::new(writer)).unwrap();
            let ctx = SessionContext::new();
            ctx.register_catalog("ducklake", Arc::new(catalog));
            let plan = ctx
                .sql(sql)
                .await
                .unwrap()
                .create_physical_plan()
                .await
                .unwrap();
            fn find(
                plan: &Arc<dyn datafusion::physical_plan::ExecutionPlan>,
            ) -> Option<&DuckLakeUpdateExec> {
                plan.downcast_ref::<DuckLakeUpdateExec>()
                    .or_else(|| plan.children().into_iter().find_map(find))
            }
            let update = find(&plan).unwrap_or_else(|| {
                panic!(
                    "no DuckLakeUpdateExec in\n{}",
                    displayable(plan.as_ref()).indent(true)
                )
            });
            let mut out = Vec::new();
            for s in &update.scans {
                let shown = displayable(s.scan.as_ref()).indent(true).to_string();
                let rows = collect(Arc::clone(&s.scan), ctx.task_ctx())
                    .await
                    .unwrap()
                    .iter()
                    .map(|b| b.num_rows())
                    .sum();
                out.push((shown, rows));
            }
            out
        }

        #[tokio::test(flavor = "multi_thread")]
        async fn keyed_update_reads_only_matching_row_groups() {
            let dir = tempfile::TempDir::new().unwrap();
            let conn = seed(&dir).await;
            let scans =
                source_scans(&conn, "UPDATE ducklake.main.t SET score = 0 WHERE id = 22").await;
            assert_eq!(scans.len(), 1);
            let (shown, rows) = &scans[0];
            assert!(
                shown.contains("predicate="),
                "predicate not pushed into the reader:\n{shown}"
            );
            assert!(
                *rows <= ROW_GROUP,
                "scanned {rows} rows of {ROWS}; expected at most one {ROW_GROUP}-row group"
            );
        }

        /// The `DELETE` guard applies: a float predicate is not pushed (footer
        /// bounds exclude NaN), so the whole file is read, as before.
        #[tokio::test(flavor = "multi_thread")]
        async fn float_predicate_is_not_pushed() {
            let dir = tempfile::TempDir::new().unwrap();
            let conn = seed(&dir).await;
            let scans =
                source_scans(&conn, "UPDATE ducklake.main.t SET id = 0 WHERE score = 22").await;
            let (shown, rows) = &scans[0];
            assert!(
                !shown.contains("predicate="),
                "float predicate was pushed:\n{shown}"
            );
            assert_eq!(*rows, ROWS as usize);
        }
    }
}
