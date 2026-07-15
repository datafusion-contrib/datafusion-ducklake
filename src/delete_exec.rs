//! DuckLake `DELETE` execution plan.
//!
//! Backs [`crate::table::DuckLakeTable`]'s `TableProvider::delete_from`. When
//! executed, it performs the delete and yields a single `count: UInt64` row
//! (rows affected), mirroring the count contract of
//! [`DuckLakeInsertExec`](crate::insert_exec::DuckLakeInsertExec).
//!
//! Two paths:
//! - **Filtered delete** (a `WHERE` clause): for each live data file, resolve the
//!   physical row positions matching the predicate, union them with the file's
//!   already-deleted positions, write ONE cumulative positional delete file, and
//!   commit every affected file's delete file in a SINGLE snapshot (atomic
//!   multi-file delete). The read path is already delete-aware, so a subsequent
//!   `SELECT` excludes the deleted rows automatically.
//! - **Delete-all** (no `WHERE`): a metadata-only truncate — end every live data
//!   file in one new snapshot. Much cheaper than positional-deleting every row.
//!
//! v1 handles insert-only data files. Files rewritten by an UPDATE/compaction
//! (an embedded row-id column) cannot be resolved by physical position, so the
//! filtered path cleanly errors on them rather than risk mis-deleting.
//!
//! # Session lifecycle (important)
//!
//! A [`DuckLakeCatalog`](crate::DuckLakeCatalog) pins its snapshot at creation
//! and never refreshes it, so a `SessionContext` observes ONE catalog generation
//! for its whole lifetime. A `DELETE` commits a new snapshot, but the same
//! session keeps reading the old one. Consequences:
//!
//! - A second filtered `DELETE` in the same session that re-touches a data file
//!   modified by an earlier `DELETE` (in that same session) aborts with a
//!   [`Conflict`](crate::DuckLakeError::Conflict): it resolves against the pinned
//!   (pre-delete) view, so its compare-and-swap disagrees with the live catalog.
//!   This is the SAME guard that (correctly) rejects a genuinely concurrent
//!   writer — and it is what prevents a stale, non-cumulative delete file from
//!   resurrecting already-deleted rows.
//! - A `SELECT` after a `DELETE` (or `INSERT`) in the same session returns the
//!   pre-mutation rows; a just-inserted row cannot be deleted in the same
//!   session (it is invisible to the pinned snapshot).
//!
//! To perform multiple mutations, re-open the catalog (or create a fresh
//! `SessionContext`) between statements so it binds to the latest snapshot.

use std::collections::HashSet;
use std::fmt::{self, Debug};
use std::sync::Arc;

use arrow::array::{ArrayRef, RecordBatch, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::catalog::Session;
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::execution::object_store::ObjectStoreUrl;
use datafusion::execution::{SendableRecordBatchStream, SessionState, TaskContext};
use datafusion::physical_expr::{EquivalenceProperties, Partitioning, PhysicalExpr};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use futures::stream;

use crate::metadata_writer::{DeleteFileEntry, MetadataWriter};
use crate::table::DuckLakeTable;
use crate::table_writer::DuckLakeTableWriter;

/// Schema for the output of a delete operation: the count of rows deleted.
/// Same shape DataFusion expects from `insert_into`.
fn make_delete_count_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new(
        "count",
        DataType::UInt64,
        false,
    )]))
}

/// Execution plan that deletes rows from a DuckLake table.
///
/// Takes no input stream (unlike insert): the rows to delete are discovered by
/// scanning the table's own data files against `predicate`. `None` predicate
/// means delete ALL rows.
pub struct DuckLakeDeleteExec {
    /// A clone of the target table, used for its reader methods
    /// (`resolve_positions`, `read_delete_file_positions`,
    /// `file_has_embedded_rowid`).
    table: Arc<DuckLakeTable>,
    /// Session captured at plan time. A bare `TaskContext` cannot build physical
    /// exprs or drive the positional sub-plans; `SessionState` is the concrete
    /// `Session` impl that can, so the delete work can run at execute time.
    session_state: SessionState,
    /// Physical predicate over the table's physical columns. `None` => delete
    /// ALL rows (metadata-only truncate).
    predicate: Option<Arc<dyn PhysicalExpr>>,
    /// Metadata writer for the atomic commit.
    writer: Arc<dyn MetadataWriter>,
    /// Object store URL for writing positional delete files.
    object_store_url: Arc<ObjectStoreUrl>,
    schema_name: String,
    table_name: String,
    table_id: i64,
    /// Snapshot the table was opened at (the generation the positions were
    /// resolved against); threaded to the commit for conflict diagnostics.
    base_snapshot: i64,
    cache: Arc<PlanProperties>,
}

impl DuckLakeDeleteExec {
    /// Create a new `DuckLakeDeleteExec`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        table: Arc<DuckLakeTable>,
        session_state: SessionState,
        predicate: Option<Arc<dyn PhysicalExpr>>,
        writer: Arc<dyn MetadataWriter>,
        schema_name: String,
        table_name: String,
        table_id: i64,
        base_snapshot: i64,
        object_store_url: Arc<ObjectStoreUrl>,
    ) -> Self {
        let cache = Self::compute_properties();
        Self {
            table,
            session_state,
            predicate,
            writer,
            object_store_url,
            schema_name,
            table_name,
            table_id,
            base_snapshot,
            cache,
        }
    }

    fn compute_properties() -> Arc<PlanProperties> {
        Arc::new(PlanProperties::new(
            EquivalenceProperties::new(make_delete_count_schema()),
            Partitioning::UnknownPartitioning(1),
            datafusion::physical_plan::execution_plan::EmissionType::Final,
            datafusion::physical_plan::execution_plan::Boundedness::Bounded,
        ))
    }
}

impl Debug for DuckLakeDeleteExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DuckLakeDeleteExec")
            .field("schema_name", &self.schema_name)
            .field("table_name", &self.table_name)
            .field("table_id", &self.table_id)
            .field("delete_all", &self.predicate.is_none())
            .finish_non_exhaustive()
    }
}

impl DisplayAs for DuckLakeDeleteExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        match t {
            DisplayFormatType::Default
            | DisplayFormatType::Verbose
            | DisplayFormatType::TreeRender => {
                write!(
                    f,
                    "DuckLakeDeleteExec: table={}.{}, predicate={}",
                    self.schema_name,
                    self.table_name,
                    if self.predicate.is_some() {
                        "filter"
                    } else {
                        "all-rows"
                    }
                )
            },
        }
    }
}

impl ExecutionPlan for DuckLakeDeleteExec {
    fn name(&self) -> &str {
        "DuckLakeDeleteExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.cache
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        // No input plan: rows to delete are found by scanning the table's files.
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        if !children.is_empty() {
            return Err(DataFusionError::Internal(
                "DuckLakeDeleteExec does not take any children".to_string(),
            ));
        }
        Ok(self)
    }

    fn execute(
        &self,
        partition: usize,
        _context: Arc<TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(DataFusionError::Internal(format!(
                "DuckLakeDeleteExec only supports partition 0, got {}",
                partition
            )));
        }

        let table = Arc::clone(&self.table);
        let session_state = self.session_state.clone();
        let predicate = self.predicate.clone();
        let writer = Arc::clone(&self.writer);
        let object_store_url = Arc::clone(&self.object_store_url);
        let schema_name = self.schema_name.clone();
        let table_name = self.table_name.clone();
        let table_id = self.table_id;
        let base_snapshot = self.base_snapshot;
        let output_schema = make_delete_count_schema();

        let stream = stream::once(async move {
            let deleted = run_delete(
                table,
                &session_state,
                predicate,
                writer,
                object_store_url,
                &schema_name,
                &table_name,
                table_id,
                base_snapshot,
            )
            .await?;
            let count: ArrayRef = Arc::new(UInt64Array::from(vec![deleted]));
            Ok(RecordBatch::try_new(output_schema, vec![count])?)
        });

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            make_delete_count_schema(),
            stream,
        )))
    }
}

/// Run the delete and return the number of rows deleted. See the module docs for
/// the two paths (filtered positional delete vs. delete-all truncate).
#[allow(clippy::too_many_arguments)]
async fn run_delete(
    table: Arc<DuckLakeTable>,
    session_state: &SessionState,
    predicate: Option<Arc<dyn PhysicalExpr>>,
    writer: Arc<dyn MetadataWriter>,
    object_store_url: Arc<ObjectStoreUrl>,
    schema_name: &str,
    table_name: &str,
    table_id: i64,
    base_snapshot: i64,
) -> DataFusionResult<u64> {
    let state: &dyn Session = session_state;
    let table_files = table
        .files()
        .map_err(|error| DataFusionError::External(Box::new(error)))?;

    // Delete-all (no WHERE): metadata-only truncate — end every live data file in
    // one snapshot. Skip the empty table entirely (no-op, no snapshot).
    let predicate = match predicate {
        None => {
            if table_files.is_empty() {
                return Ok(0);
            }
            return writer
                .commit_truncate(table_id, schema_name, table_name, base_snapshot)
                .map_err(|e| DataFusionError::External(Box::new(e)));
        },
        Some(p) => p,
    };

    // Object store for writing delete files — the same store the positional reads
    // resolve against.
    let object_store = state
        .runtime_env()
        .object_store(object_store_url.as_ref())?;
    let table_writer = DuckLakeTableWriter::new(Arc::clone(&writer), object_store)
        .map_err(|e| DataFusionError::External(Box::new(e)))?;

    let mut entries: Vec<DeleteFileEntry> = Vec::new();
    let mut total_deleted: u64 = 0;

    for tf in &table_files {
        // v1 refuses files rewritten by an UPDATE/compaction: their surviving
        // rows carry embedded rowids whose physical order need not match the
        // DuckLake `pos` space, so positional resolution could mis-delete. Clean
        // error, never a silent wrong delete.
        if table.file_has_embedded_rowid(state, &tf.file).await? {
            return Err(DataFusionError::NotImplemented(format!(
                "DELETE on data file '{}' is not supported: the file was rewritten by an \
                 UPDATE or compaction (it embeds a row-id column), and v1 resolves delete \
                 positions only for insert-only files",
                tf.file.path
            )));
        }

        // Physical positions matching the predicate (raw scan; delete files NOT
        // applied — resolution is over the file's own rows).
        let matched = table
            .resolve_positions(state, &tf.file, Arc::clone(&predicate))
            .await?;
        if matched.is_empty() {
            continue;
        }

        // Already-deleted positions for this file (if a delete file is live).
        // Rows already deleted are neither re-counted nor re-deleted.
        let existing = match tf.delete_file {
            Some(ref df) => table.read_delete_file_positions(state, df).await?,
            None => HashSet::new(),
        };

        let newly_deleted = matched.difference(&existing).count() as u64;
        if newly_deleted == 0 {
            // Every matched row was already deleted: nothing changes here.
            continue;
        }

        // Cumulative (prior ∪ new) still-deleted set: at most one delete file is
        // live per data file, so each write carries the full set.
        let mut cumulative: Vec<i64> = existing.union(&matched).copied().collect();
        cumulative.sort_unstable();

        let delete_info = table_writer
            .write_delete_file(schema_name, table_name, &tf.file.path, &cumulative)
            .await
            .map_err(|e| DataFusionError::External(Box::new(e)))?;

        entries.push(DeleteFileEntry {
            data_file_id: tf.data_file_id,
            expected_prev_delete_file: tf.delete_file_id,
            delete: delete_info,
        });
        total_deleted += newly_deleted;
    }

    if entries.is_empty() {
        // Predicate matched nothing new anywhere: no commit, no snapshot.
        return Ok(0);
    }

    // Commit every affected file's cumulative delete file in ONE snapshot
    // (atomic multi-file DELETE). No new data file is appended, so this uses the
    // dedicated delete-only commit rather than `register_data_file_with_deletes`
    // (which requires an appended file).
    writer
        .commit_positional_deletes(table_id, schema_name, table_name, base_snapshot, &entries)
        .map_err(|e| DataFusionError::External(Box::new(e)))?;

    Ok(total_deleted)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delete_count_schema() {
        let schema = make_delete_count_schema();
        assert_eq!(schema.fields().len(), 1);
        assert_eq!(schema.field(0).name(), "count");
        assert_eq!(schema.field(0).data_type(), &DataType::UInt64);
    }
}
