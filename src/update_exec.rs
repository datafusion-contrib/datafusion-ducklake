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
//!    original rowid (written into a NEW data file that embeds the rowid column,
//!    so lineage survives the rewrite).
//! 2. Resolve, per source file, the cumulative positional delete masking the old
//!    row versions (superseded rows unioned with any already-deleted rows).
//! 3. Commit ATOMICALLY: the appended data file AND every positional delete land
//!    in ONE snapshot via
//!    [`MetadataWriter::register_data_file_with_deletes`]
//!    (driven by `TableWriteSession::finish_with_deletes`).
//! 4. Yield a single row `count: UInt64` = rows updated.
//!
//! Limitations (shared with [`DuckLakeInsertExec`](crate::insert_exec)):
//! collects matched rows into memory before writing; single partition only.
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
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::execution::object_store::ObjectStoreUrl;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::{EquivalenceProperties, Partitioning, PhysicalExpr};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use futures::stream::{self, TryStreamExt};

use crate::metadata_writer::{DeleteFileEntry, MetadataWriter, WriteMode};
use crate::table::{DuckLakeTable, UpdateSourceScan};
use crate::table_writer::DuckLakeTableWriter;

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
                .map_err(|e| DataFusionError::External(Box::new(e)))?;

            // Rewrite each source file's matching rows and author its cumulative
            // positional delete. Delete parquet files are uploaded here; the
            // catalog commit that makes them visible happens once, atomically,
            // below — so a failure before the commit leaves the live snapshot
            // untouched (only orphan objects, cleaned by maintenance).
            let mut updated_batches: Vec<RecordBatch> = Vec::new();
            let mut delete_entries: Vec<DeleteFileEntry> = Vec::new();
            let mut total_updated: u64 = 0;

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

                let delete_info = table_writer
                    .write_delete_file(
                        &schema_name,
                        &table_name,
                        &scan.source_path,
                        &out.cumulative_positions,
                    )
                    .await
                    .map_err(|e| DataFusionError::External(Box::new(e)))?;
                delete_entries.push(DeleteFileEntry {
                    data_file_id: scan.data_file_id,
                    expected_prev_delete_file: scan.delete_file_id,
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
            let physical_schema = table.physical_schema();
            let mut session = table_writer
                .begin_write_with_embedded_rowid(
                    &schema_name,
                    &table_name,
                    physical_schema.as_ref(),
                    WriteMode::Append,
                )
                .map_err(|e| DataFusionError::External(Box::new(e)))?;
            for batch in &updated_batches {
                session
                    .write_batch(batch)
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
}
