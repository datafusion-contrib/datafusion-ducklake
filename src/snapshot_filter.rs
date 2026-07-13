//! Per-row time-travel visibility filter for DuckLake **partial data files**.
//!
//! A partial file (produced by `merge_adjacent_files`) carries rows from several
//! origin snapshots, each row tagged with its origin in the embedded
//! `_ducklake_internal_snapshot_id` column, and its `ducklake_data_file` row is
//! visible from the MINIMUM origin snapshot onward. When such a file is read at a
//! historical snapshot `S` below its `partial_max`, rows whose origin is newer
//! than `S` did not yet exist at `S` and must be dropped. [`SnapshotFilterExec`]
//! does exactly that: it reads the embedded snapshot-id column (by name) and
//! keeps only rows whose value is `<= read_snapshot`.
//!
//! The snapshot-id column is passed through unchanged; the final projection
//! ([`ColumnRenameExec`](crate::column_rename::ColumnRenameExec)) drops it since
//! it is not a catalog column. Filtering only removes rows, so partitioning and
//! ordering are preserved.

use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow::array::{Array, Int64Array};
use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::execution::{RecordBatchStream, SendableRecordBatchStream, TaskContext};
use datafusion::physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use futures::Stream;

/// Execution plan that drops rows of a partial file whose embedded origin
/// snapshot is newer than the read snapshot.
#[derive(Debug)]
pub struct SnapshotFilterExec {
    /// Input plan; must carry the embedded snapshot-id column named
    /// [`snapshot_column`](Self::snapshot_column).
    input: Arc<dyn ExecutionPlan>,
    /// Parquet name of the embedded snapshot-id column.
    snapshot_column: String,
    /// Rows are kept iff their origin snapshot is `<= read_snapshot`.
    read_snapshot: i64,
    /// Index of `snapshot_column` in the input schema.
    snapshot_index: usize,
    /// Cached plan properties (schema is unchanged — filtering only drops rows).
    properties: Arc<PlanProperties>,
}

impl SnapshotFilterExec {
    /// Build a `SnapshotFilterExec`. The input must carry a column named
    /// `snapshot_column` (the embedded `_ducklake_internal_snapshot_id`); errors
    /// otherwise.
    pub fn try_new(
        input: Arc<dyn ExecutionPlan>,
        snapshot_column: String,
        read_snapshot: i64,
    ) -> DataFusionResult<Self> {
        let snapshot_index = input.schema().index_of(&snapshot_column).map_err(|_| {
            DataFusionError::Internal(format!(
                "SnapshotFilterExec input is missing the `{snapshot_column}` column"
            ))
        })?;
        let properties = input.properties().clone();
        Ok(Self {
            input,
            snapshot_column,
            read_snapshot,
            snapshot_index,
            properties,
        })
    }
}

impl DisplayAs for SnapshotFilterExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "SnapshotFilterExec: keep origin <= {}",
            self.read_snapshot
        )
    }
}

impl ExecutionPlan for SnapshotFilterExec {
    fn name(&self) -> &str {
        "SnapshotFilterExec"
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

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(DataFusionError::Internal(
                "SnapshotFilterExec expects exactly one child".into(),
            ));
        }
        Ok(Arc::new(SnapshotFilterExec::try_new(
            children.into_iter().next().unwrap(),
            self.snapshot_column.clone(),
            self.read_snapshot,
        )?))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        Ok(Box::pin(SnapshotFilterStream {
            input: self.input.execute(partition, context)?,
            snapshot_column: self.snapshot_column.clone(),
            read_snapshot: self.read_snapshot,
            snapshot_index: self.snapshot_index,
        }))
    }
}

/// Stream that keeps rows whose embedded origin snapshot is `<= read_snapshot`.
struct SnapshotFilterStream {
    input: SendableRecordBatchStream,
    snapshot_column: String,
    read_snapshot: i64,
    snapshot_index: usize,
}

impl Stream for SnapshotFilterStream {
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

impl SnapshotFilterStream {
    fn filter_batch(&self, batch: &RecordBatch) -> DataFusionResult<RecordBatch> {
        let snap = batch
            .column(self.snapshot_index)
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| {
                DataFusionError::Internal(format!("`{}` column is not Int64", self.snapshot_column))
            })?;

        let num_rows = batch.num_rows();
        let mut keep_indices: Vec<u32> = Vec::with_capacity(num_rows);
        for i in 0..num_rows {
            // A NULL origin is treated as visible (defensive; the writer always
            // populates it). Otherwise keep rows whose origin is <= read snapshot.
            if snap.is_null(i) || snap.value(i) <= self.read_snapshot {
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

impl RecordBatchStream for SnapshotFilterStream {
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

    fn batch(values: &[i32], origins: &[i64]) -> (SchemaRef, RecordBatch) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("_ducklake_internal_snapshot_id", DataType::Int64, true),
        ]));
        let b = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(values.to_vec())) as ArrayRef,
                Arc::new(Int64Array::from(origins.to_vec())) as ArrayRef,
            ],
        )
        .unwrap();
        (schema, b)
    }

    fn stream(schema: SchemaRef, read_snapshot: i64) -> SnapshotFilterStream {
        SnapshotFilterStream {
            input: Box::pin(EmptyRecordBatchStream::new(schema)),
            snapshot_column: "_ducklake_internal_snapshot_id".to_string(),
            read_snapshot,
            snapshot_index: 1,
        }
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
    fn keeps_rows_at_or_below_read_snapshot() {
        // Rows tagged with origins 1,2,3; reading at snapshot 2 keeps origins 1,2.
        let (schema, b) = batch(&[10, 20, 30], &[1, 2, 3]);
        let filtered = stream(schema, 2).filter_batch(&b).unwrap();
        assert_eq!(ids(&filtered), vec![10, 20]);
    }

    #[test]
    fn keeps_all_when_read_snapshot_at_or_above_max() {
        let (schema, b) = batch(&[10, 20, 30], &[1, 2, 3]);
        let filtered = stream(schema, 3).filter_batch(&b).unwrap();
        assert_eq!(ids(&filtered), vec![10, 20, 30]);
    }

    #[test]
    fn drops_all_when_read_snapshot_below_min() {
        let (schema, b) = batch(&[10, 20], &[5, 6]);
        let filtered = stream(schema, 1).filter_batch(&b).unwrap();
        assert!(ids(&filtered).is_empty());
    }
}
