//! Custom execution plan for filtering deleted rows
//!
//! Wraps a positional scan and drops rows whose **physical file position**
//! appears in a positional delete file. The physical position is read from the
//! internal [`ROW_POS_COLUMN_NAME`] column materialized by
//! [`FileRowNumberExec`](crate::row_id::FileRowNumberExec) — never from stream
//! arrival order — so filtering is correct regardless of how the scan is
//! partitioned or merged. The position column is passed through unchanged for
//! any downstream consumer (e.g. `RowIdExec`); the final projection drops it.

use std::any::Any;
use std::collections::HashSet;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow::array::Int64Array;
use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::execution::{RecordBatchStream, SendableRecordBatchStream, TaskContext};
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
    /// Index of [`ROW_POS_COLUMN_NAME`] in the input schema.
    pos_index: usize,
    /// Cached plan properties.
    properties: Arc<PlanProperties>,
}

impl DeleteFilterExec {
    /// Build a `DeleteFilterExec`. The input must carry the
    /// [`ROW_POS_COLUMN_NAME`] column (from `FileRowNumberExec`); errors otherwise.
    pub fn try_new(
        input: Arc<dyn ExecutionPlan>,
        file_path: String,
        deleted_positions: Arc<HashSet<i64>>,
    ) -> DataFusionResult<Self> {
        let pos_index = input.schema().index_of(ROW_POS_COLUMN_NAME).map_err(|_| {
            DataFusionError::Internal(format!(
                "DeleteFilterExec input is missing the `{ROW_POS_COLUMN_NAME}` column"
            ))
        })?;
        // Filtering only drops rows; partitioning/ordering are preserved.
        let properties = input.properties().clone();
        Ok(Self {
            input,
            file_path,
            deleted_positions,
            pos_index,
            properties,
        })
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
    fn name(&self) -> &str {
        "DeleteFilterExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
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
                "DeleteFilterExec expects exactly one child".into(),
            ));
        }
        Ok(Arc::new(DeleteFilterExec::try_new(
            children.into_iter().next().unwrap(),
            self.file_path.clone(),
            self.deleted_positions.clone(),
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
        }))
    }
}

/// Stream that filters deleted rows by reading the physical-position column.
struct DeleteFilterStream {
    input: SendableRecordBatchStream,
    deleted_positions: Arc<HashSet<i64>>,
    pos_index: usize,
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
        if self.deleted_positions.is_empty() {
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
            if !self.deleted_positions.contains(&pos.value(i)) {
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
            Field::new("id", DataType::Int32, false),
            crate::row_id::row_pos_field(),
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

    fn stream(schema: SchemaRef, deleted: &[i64]) -> DeleteFilterStream {
        DeleteFilterStream {
            input: Box::pin(EmptyRecordBatchStream::new(schema)),
            deleted_positions: Arc::new(deleted.iter().copied().collect::<HashSet<i64>>()),
            pos_index: 1,
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
