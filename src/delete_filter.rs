//! Custom execution plan for filtering deleted rows
//!
//! Wraps a positional scan and drops rows whose **physical file position**
//! appears in a positional delete file. The physical position is read from the
//! reader-produced position column (see
//! `row_id::positional_table_schema`) — never
//! from stream arrival order — so filtering is correct regardless of how the
//! scan is pruned, filtered, partitioned or merged. The position column is
//! passed through unchanged for any downstream consumer (e.g. `RowIdExec`); the
//! final projection drops it.

use std::collections::HashSet;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow::array::Int64Array;
use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use datafusion::common::config::ConfigOptions;
use datafusion::common::stats::Precision;
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::common::{ColumnStatistics, Statistics};
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::execution::{RecordBatchStream, SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_plan::filter_pushdown::{
    ChildFilterDescription, FilterDescription, FilterPushdownPhase,
};
use datafusion::physical_plan::statistics::{ChildStats, StatisticsArgs};
use datafusion::physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use futures::Stream;

use crate::row_id::ROW_POS_COLUMN_NAME;

/// Custom execution plan that filters out deleted rows by physical position.
#[derive(Debug)]
pub struct DeleteFilterExec {
    /// The input execution plan (carries [`ROW_POS_COLUMN_NAME`]).
    input: Arc<dyn ExecutionPlan>,
    /// Path of the file being scanned (for display).
    file_path: String,
    /// Set of deleted physical row positions for this file (shared across streams).
    deleted_positions: Arc<HashSet<i64>>,
    /// Index of the physical-position column in the input schema.
    pos_index: usize,
    /// The file's recorded `record_count`, when the catalog has one.
    max_row_count: Option<usize>,
    /// Cached plan properties.
    properties: Arc<PlanProperties>,
}

impl DeleteFilterExec {
    /// Build a `DeleteFilterExec` over an input whose column `pos_index` is the
    /// reader-produced physical row position.
    ///
    /// The index is passed in rather than looked up by name because the position
    /// column's name is per-scan (see
    /// `row_id::unique_row_pos_name`), and because
    /// a name lookup would silently bind a user column of the same name.
    pub fn try_new(
        input: Arc<dyn ExecutionPlan>,
        file_path: String,
        deleted_positions: Arc<HashSet<i64>>,
        pos_index: usize,
        max_row_count: Option<usize>,
    ) -> DataFusionResult<Self> {
        let schema = input.schema();
        let field = schema.fields().get(pos_index).ok_or_else(|| {
            DataFusionError::Internal(format!(
                "DeleteFilterExec: position index {pos_index} is out of range for an \
                 input of {} columns",
                schema.fields().len()
            ))
        })?;
        crate::row_id::validate_row_pos_field("DeleteFilterExec", pos_index, field)?;
        // Filtering only drops rows; partitioning/ordering are preserved.
        let properties = input.properties().clone();
        Ok(Self {
            input,
            file_path,
            deleted_positions,
            pos_index,
            max_row_count,
            properties,
        })
    }
}

impl DeleteFilterExec {
    /// How many of `deleted_positions` name a row this scan will actually emit.
    ///
    /// NOT `deleted_positions.len()`. Execution drops a row only when that row's
    /// own position is in the set, and it emits no row at or past `rows` (the
    /// clamp in `filter_batch`). A position outside `0..rows` therefore removes
    /// nothing, and subtracting it would publish a count BELOW what the scan
    /// returns — `count(*)` is answered from that number, so the query would be
    /// wrong rather than slow.
    ///
    /// Such a position arises without any corruption: a delete file's
    /// `file_path` column is documentation this reader ignores, so one delete
    /// file referenced by several data files contributes the others' positions.
    fn deleted_in_range(&self, rows: usize) -> usize {
        self.deleted_positions
            .iter()
            .filter(|position| usize::try_from(**position).is_ok_and(|p| p < rows))
            .count()
    }
}

impl DisplayAs for DeleteFilterExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "DeleteFilterExec: file={}, deletes={}",
            self.file_path,
            self.deleted_positions.len()
        )
    }
}

impl ExecutionPlan for DeleteFilterExec {
    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> DataFusionResult<TreeNodeRecursion>,
    ) -> DataFusionResult<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn name(&self) -> &str {
        "DeleteFilterExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    /// Order-preserving: drops rows but never reorders them.
    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true]
    }

    /// The input's row count less the rows this node removes, so an unfiltered
    /// `count(*)` over a table with deletes is still answered from the catalog
    /// rather than by reading every file — which is what official does, folding
    /// `count(*)` unconditionally and subtracting delete counts independently.
    ///
    /// `deleted_positions` is the authority, not the catalog's `delete_count`.
    /// It is a SET, assembled by merging a file's positional deletes with its
    /// inlined ones, so a row deleted by both is counted once — and it covers
    /// inlined deletes, which leave `delete_file` and `delete_count` NULL and are
    /// therefore invisible to the catalog counters.
    ///
    /// PRECONDITION: every position names a row this file holds. Subtracting the
    /// set's size is only right under that invariant — execution drops a row by
    /// set MEMBERSHIP, so an unmatched position removes nothing at runtime while
    /// still counting here, and `count(*)` would answer below what the scan
    /// emits. `DuckLakeTable::deleted_positions_for_file` establishes it by
    /// dropping positions outside the file's `record_count`, which matters
    /// because a delete file's `file_path` column is ignored and one such file
    /// may be referenced by several data files.
    ///
    /// Column bounds are dropped to unknown. Removing rows can remove the very
    /// row holding an extreme, and nothing here knows which, so an inherited
    /// bound could answer `max(col)` with a value that is no longer present. The
    /// row count survives because it needs no such knowledge: every deleted
    /// position is one fewer row, whichever row it was.
    ///
    /// Implemented on `statistics_from_inputs` rather than the deprecated
    /// `partition_statistics`: that is the method the `StatisticsContext` walk
    /// actually calls, and a child reached by calling `partition_statistics`
    /// directly would answer from the trait default (unknown) whenever it, too,
    /// only implements the new one. Unlike `NanPruningBarrierExec`, no deprecated
    /// override is kept alongside — an out-of-tree caller on the old method gets
    /// `new_unknown` here, which loses the fold but cannot produce a wrong
    /// answer, whereas the barrier's override exists to keep a *safety* property
    /// for those callers.
    fn child_stats_requests(&self, partition: Option<usize>) -> Vec<ChildStats> {
        vec![ChildStats::At(partition)]
    }

    fn statistics_from_inputs(
        &self,
        input_stats: &[Arc<Statistics>],
        args: &StatisticsArgs,
    ) -> DataFusionResult<Arc<Statistics>> {
        let Some(input) = input_stats.first() else {
            return Ok(Arc::new(Statistics::new_unknown(&self.schema())));
        };
        let mut statistics = input.as_ref().clone();
        // Deleted positions belong to the FILE, not to any one output partition,
        // so they cannot be attributed to a single partition's count.
        statistics.num_rows = if args.partition().is_some() {
            statistics.num_rows.to_inexact()
        } else {
            match statistics.num_rows {
                Precision::Exact(rows) => Precision::Exact(rows - self.deleted_in_range(rows)),
                other => other.to_inexact(),
            }
        };
        // The input's byte size counts rows this node drops, and nothing here
        // knows their width, so it can only be an over-estimate from here on —
        // never an exact figure.
        statistics.total_byte_size = statistics.total_byte_size.to_inexact();
        statistics.column_statistics = statistics
            .column_statistics
            .iter()
            .map(|_| ColumnStatistics::new_unknown())
            .collect();
        Ok(Arc::new(statistics))
    }

    /// Forward filter pushdown unchanged: this node's output schema is its input
    /// schema, so a predicate means the same thing on either side of it.
    ///
    /// Soundness rests on `filter(delete(R)) == delete(filter(R))`. Deletion is
    /// keyed by **absolute physical position**, which the parquet reader derives
    /// from row-group offsets in the footer, so dropping non-matching rows in the
    /// reader cannot change which surviving row sits at which position. That is
    /// specifically what reader-produced positions buy: when positions were
    /// synthesized by counting stream arrivals, pruning a single row shifted
    /// every position after it and this forwarding would have been corrupting.
    fn gather_filters_for_pushdown(
        &self,
        _phase: FilterPushdownPhase,
        parent_filters: Vec<Arc<dyn PhysicalExpr>>,
        _config: &ConfigOptions,
    ) -> DataFusionResult<FilterDescription> {
        let child = ChildFilterDescription::from_child(&parent_filters, &self.input)?;
        Ok(FilterDescription::new().with_child(child))
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(DataFusionError::Internal(
                "DeleteFilterExec expects exactly one child".into(),
            ));
        }
        Ok(Arc::new(DeleteFilterExec::try_new(
            children.into_iter().next().unwrap(),
            self.file_path.clone(),
            self.deleted_positions.clone(),
            self.pos_index,
            self.max_row_count,
        )?))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        Ok(Box::pin(DeleteFilterStream {
            input: self.input.execute(partition, context)?,
            deleted_positions: self.deleted_positions.clone(),
            pos_index: self.pos_index,
            max_row_count: self.max_row_count,
        }))
    }
}

/// Stream that filters deleted rows by reading the physical-position column.
struct DeleteFilterStream {
    input: SendableRecordBatchStream,
    deleted_positions: Arc<HashSet<i64>>,
    pos_index: usize,
    max_row_count: Option<usize>,
}

impl Stream for DeleteFilterStream {
    type Item = DataFusionResult<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match Pin::new(&mut self.input).poll_next(cx) {
            Poll::Ready(Some(Ok(batch))) => Poll::Ready(Some(self.filter_batch(&batch))),
            Poll::Ready(Some(Err(e))) => Poll::Ready(Some(Err(e))),
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

impl DeleteFilterStream {
    fn filter_batch(&self, batch: &RecordBatch) -> DataFusionResult<RecordBatch> {
        // Fast path only when there is nothing to do at all. The clamp must not
        // depend on the set being non-empty: today the exec is built only with
        // deletes, but that is the caller's invariant, not this method's.
        if self.deleted_positions.is_empty() && self.max_row_count.is_none() {
            return Ok(batch.clone());
        }

        let pos = batch
            .column(self.pos_index)
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| {
                DataFusionError::Internal(format!("`{ROW_POS_COLUMN_NAME}` column is not Int64"))
            })?;

        let num_rows = batch.num_rows();
        let mut keep_indices: Vec<u32> = Vec::with_capacity(num_rows);
        for i in 0..num_rows {
            let position = pos.value(i);
            // Clamp the data scan to the file's recorded row count, as official
            // does (`DuckLakeDeleteFilter::Filter` limits the scan range from
            // `SetMaxRowCount`). A file holding more rows than the catalog
            // records is corrupt metadata, and the rows past the end are ones no
            // delete position can name — dropping the position instead would let
            // a deleted row reappear.
            let beyond_file = self
                .max_row_count
                .is_some_and(|max| usize::try_from(position).map_or(true, |p| p >= max));
            if beyond_file {
                continue;
            }
            if !self.deleted_positions.contains(&position) {
                keep_indices.push(i as u32);
            }
        }

        if keep_indices.len() == num_rows {
            return Ok(batch.clone());
        }

        use arrow::array::UInt32Array;
        use arrow::compute::take;

        let indices = UInt32Array::from(keep_indices);
        let filtered_columns: DataFusionResult<Vec<_>> = batch
            .columns()
            .iter()
            .map(|col| {
                take(col.as_ref(), &indices, None)
                    .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))
            })
            .collect();

        RecordBatch::try_new(batch.schema(), filtered_columns?)
            .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))
    }
}

impl RecordBatchStream for DeleteFilterStream {
    fn schema(&self) -> SchemaRef {
        self.input.schema()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, ArrayRef, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion::physical_plan::EmptyRecordBatchStream;

    /// Build a batch with a value column and a `__ducklake_row_pos` column.
    fn batch(values: &[i32], positions: &[i64]) -> (SchemaRef, RecordBatch) {
        let schema = Arc::new(Schema::new(vec![
            Arc::new(Field::new("id", DataType::Int32, false)),
            crate::row_id::row_pos_virtual_field(ROW_POS_COLUMN_NAME),
        ]));
        let b = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(values.to_vec())) as ArrayRef,
                Arc::new(Int64Array::from(positions.to_vec())) as ArrayRef,
            ],
        )
        .unwrap();
        (schema, b)
    }

    /// Unclamped: `max_row_count` unset, so only the delete set decides.
    fn stream(schema: SchemaRef, deleted: &[i64]) -> DeleteFilterStream {
        clamped_stream(schema, deleted, None)
    }

    fn clamped_stream(
        schema: SchemaRef,
        deleted: &[i64],
        max_row_count: Option<usize>,
    ) -> DeleteFilterStream {
        DeleteFilterStream {
            input: Box::pin(EmptyRecordBatchStream::new(schema)),
            deleted_positions: Arc::new(deleted.iter().copied().collect::<HashSet<i64>>()),
            pos_index: 1,
            max_row_count,
        }
    }

    /// Rows at or past the file's recorded end are not emitted, as official's
    /// read path does by clamping the scan range. Without this, a delete naming
    /// such a row would have nothing to match and the row would come back.
    #[test]
    fn clamps_rows_past_the_recorded_row_count() {
        let (schema, b) = batch(&[1, 2, 3, 4, 5], &[0, 1, 2, 3, 4]);
        let out = clamped_stream(schema, &[4], Some(3))
            .filter_batch(&b)
            .unwrap();
        assert_eq!(ids(&out), vec![1, 2, 3]);
    }

    /// A negative position cannot name a row, and is dropped rather than kept.
    #[test]
    fn clamps_a_negative_position_when_the_row_count_is_known() {
        let (schema, b) = batch(&[1, 2], &[-1, 0]);
        let out = clamped_stream(schema, &[], Some(2))
            .filter_batch(&b)
            .unwrap();
        assert_eq!(ids(&out), vec![2]);
    }

    fn ids(b: &RecordBatch) -> Vec<i32> {
        b.column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap()
            .values()
            .to_vec()
    }

    #[test]
    fn deletes_row_at_listed_position() {
        // positions [0,1,2,3]; delete position 1 (id=2). 1000 is out of range.
        let (schema, b) = batch(&[1, 2, 3, 4], &[0, 1, 2, 3]);
        let filtered = stream(schema, &[1, 1000]).filter_batch(&b).unwrap();
        assert_eq!(ids(&filtered), vec![1, 3, 4]);
    }

    #[test]
    fn keeps_all_when_no_position_matches() {
        let (schema, b) = batch(&[10, 20, 30], &[0, 1, 2]);
        let filtered = stream(schema, &[1000, 2000]).filter_batch(&b).unwrap();
        assert_eq!(ids(&filtered), vec![10, 20, 30]);
    }

    #[test]
    fn deletes_by_physical_position_not_arrival_order() {
        // Positions are non-contiguous and out of arrival order: this batch
        // holds physical rows {10, 11, 12, 13}. Deleting position 11 must drop
        // the row whose pos==11 (value 200), regardless of its index in the batch.
        let (schema, b) = batch(&[100, 200, 300, 400], &[10, 11, 12, 13]);
        let filtered = stream(schema, &[11, 1000]).filter_batch(&b).unwrap();
        assert_eq!(ids(&filtered), vec![100, 300, 400]);
    }
}
