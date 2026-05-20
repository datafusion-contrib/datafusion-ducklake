//! Synthetic `rowid` column injection for DuckLake row lineage.
//!
//! DuckLake assigns each row a globally unique `rowid` BIGINT. For files written
//! by INSERT, the catalog records the file's `row_id_start`, and per-row rowid
//! is `row_id_start + position_in_file`. This module implements an execution
//! plan that appends that synthetic column to each batch streaming out of a
//! per-file Parquet scan, in file order.
//!
//! Limitations vs. the DuckDB extension:
//! - Files written by UPDATE/compaction store rowids inline in the parquet
//!   itself (as `_ducklake_internal_row_id`). This is not yet wired up here;
//!   reads of compacted files will produce computed rowids that no longer
//!   reflect the original lineage. Tracked as a follow-up.

use std::any::Any;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow::array::{ArrayRef, Int64Array};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::execution::{RecordBatchStream, SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, PlanProperties,
};
use futures::Stream;

/// Name of the synthetic rowid column exposed when row lineage is enabled.
pub const ROWID_COLUMN_NAME: &str = "rowid";

/// Iceberg / DuckLake reserved parquet field-id for the row-id column.
/// Matches `MultiFileReader::ROW_ID_FIELD_ID` in DuckDB
/// (`duckdb/src/include/duckdb/common/multi_file/multi_file_reader.hpp`).
/// Files written by `UPDATE` / compaction embed a column tagged with this
/// field-id (typically named `_ducklake_internal_row_id`) so original rowids
/// survive across file rewrites.
pub const ROW_ID_PARQUET_FIELD_ID: i32 = 2_147_483_540;

/// Build the Arrow Field for the rowid column. Nullable so we can emit NULL
/// for files whose catalog row_id_start is unrecorded (e.g. older catalogs).
pub fn rowid_field() -> Field {
    Field::new(ROWID_COLUMN_NAME, DataType::Int64, true)
}

/// Execution plan that appends a synthetic `rowid` BIGINT column to each batch.
///
/// For a row at position `p` within the file, `rowid = row_id_start + p`. The
/// row_id_start is supplied per-file at plan construction. If `row_id_start` is
/// `None`, the column is emitted as all NULL (caller's catalog doesn't track
/// lineage for this file).
///
/// The plan does not change row count or row order, so it composes cleanly with
/// a downstream `DeleteFilterExec` whose internal position cursor stays aligned.
#[derive(Debug)]
pub struct RowIdExec {
    input: Arc<dyn ExecutionPlan>,
    /// File's catalog-recorded `row_id_start`. None ⇒ emit NULL rowids.
    row_id_start: Option<i64>,
    /// Position in the output schema where the rowid column is inserted.
    /// Valid range: 0..=input.schema().fields().len().
    insert_at: usize,
    /// Output schema = input schema with rowid inserted at `insert_at`.
    schema: SchemaRef,
    properties: Arc<PlanProperties>,
}

impl RowIdExec {
    /// Append rowid as the last column. Convenience for the common case.
    pub fn new(input: Arc<dyn ExecutionPlan>, row_id_start: Option<i64>) -> Self {
        let insert_at = input.schema().fields().len();
        Self::new_at(input, row_id_start, insert_at)
    }

    /// Insert rowid at a specific output column position.
    pub fn new_at(
        input: Arc<dyn ExecutionPlan>,
        row_id_start: Option<i64>,
        insert_at: usize,
    ) -> Self {
        let input_schema = input.schema();
        let input_len = input_schema.fields().len();
        let insert_at = insert_at.min(input_len);

        let mut fields: Vec<Arc<Field>> = input_schema.fields().iter().cloned().collect();
        fields.insert(insert_at, Arc::new(rowid_field()));
        let schema = Arc::new(Schema::new(fields));

        let eq_properties = EquivalenceProperties::new(schema.clone());
        let properties = Arc::new(PlanProperties::new(
            eq_properties,
            input.output_partitioning().clone(),
            input.pipeline_behavior(),
            input.boundedness(),
        ));

        Self {
            input,
            row_id_start,
            insert_at,
            schema,
            properties,
        }
    }
}

impl DisplayAs for RowIdExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default
            | DisplayFormatType::Verbose
            | DisplayFormatType::TreeRender => {
                write!(
                    f,
                    "RowIdExec: row_id_start={}",
                    self.row_id_start
                        .map_or_else(|| "NULL".to_string(), |v| v.to_string())
                )
            },
        }
    }
}

impl ExecutionPlan for RowIdExec {
    fn name(&self) -> &str {
        "RowIdExec"
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

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(DataFusionError::Internal(
                "RowIdExec expects exactly one child".into(),
            ));
        }
        Ok(Arc::new(RowIdExec::new_at(
            children.into_iter().next().unwrap(),
            self.row_id_start,
            self.insert_at,
        )))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        Ok(Box::pin(RowIdStream {
            input: self.input.execute(partition, context)?,
            schema: self.schema.clone(),
            row_id_start: self.row_id_start,
            insert_at: self.insert_at,
            cursor: 0,
        }))
    }
}

struct RowIdStream {
    input: SendableRecordBatchStream,
    schema: SchemaRef,
    row_id_start: Option<i64>,
    insert_at: usize,
    /// Position in the file of the first row of the next batch (file-order).
    cursor: i64,
}

impl Stream for RowIdStream {
    type Item = DataFusionResult<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match Pin::new(&mut self.input).poll_next(cx) {
            Poll::Ready(Some(Ok(batch))) => {
                let n = batch.num_rows();
                let rowid_col: ArrayRef = match self.row_id_start {
                    Some(start) => {
                        let mut builder = Int64Array::builder(n);
                        for i in 0..n {
                            builder.append_value(start + self.cursor + i as i64);
                        }
                        Arc::new(builder.finish())
                    },
                    None => {
                        let mut builder = Int64Array::builder(n);
                        for _ in 0..n {
                            builder.append_null();
                        }
                        Arc::new(builder.finish())
                    },
                };
                self.cursor += n as i64;

                let mut cols: Vec<ArrayRef> = batch.columns().to_vec();
                let pos = self.insert_at.min(cols.len());
                cols.insert(pos, rowid_col);
                let out = RecordBatch::try_new(self.schema.clone(), cols)
                    .map_err(|e| DataFusionError::ArrowError(Box::new(e), None));
                Poll::Ready(Some(out))
            },
            Poll::Ready(Some(Err(e))) => Poll::Ready(Some(Err(e))),
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

impl RecordBatchStream for RowIdStream {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, Int32Array};
    use datafusion::datasource::memory::MemorySourceConfig;

    fn small_batch(schema: SchemaRef, values: &[i32]) -> RecordBatch {
        RecordBatch::try_new(
            schema,
            vec![Arc::new(Int32Array::from(values.to_vec())) as ArrayRef],
        )
        .unwrap()
    }

    #[tokio::test]
    async fn appends_sequential_rowids_across_batches() {
        let input_schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int32, false)]));
        let b1 = small_batch(input_schema.clone(), &[10, 20, 30]);
        let b2 = small_batch(input_schema.clone(), &[40, 50]);
        let mem =
            MemorySourceConfig::try_new_exec(&[vec![b1, b2]], input_schema.clone(), None)
                .unwrap();

        let exec = Arc::new(RowIdExec::new(mem, Some(1000)));
        assert_eq!(exec.schema().field(1).name(), ROWID_COLUMN_NAME);

        let ctx = Arc::new(TaskContext::default());
        let mut stream = exec.execute(0, ctx).unwrap();
        use futures::StreamExt;

        let first = stream.next().await.unwrap().unwrap();
        let rowids = first
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .values()
            .to_vec();
        assert_eq!(rowids, vec![1000, 1001, 1002]);

        let second = stream.next().await.unwrap().unwrap();
        let rowids = second
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .values()
            .to_vec();
        assert_eq!(rowids, vec![1003, 1004]);
    }

    #[tokio::test]
    async fn inserts_rowid_at_requested_position() {
        let input_schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
        ]));
        let batch = RecordBatch::try_new(
            input_schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![10, 20])) as ArrayRef,
                Arc::new(Int32Array::from(vec![100, 200])) as ArrayRef,
            ],
        )
        .unwrap();
        let mem =
            MemorySourceConfig::try_new_exec(&[vec![batch]], input_schema.clone(), None)
                .unwrap();

        // Insert rowid at position 1 → schema should be [a, rowid, b]
        let exec = Arc::new(RowIdExec::new_at(mem, Some(500), 1));
        assert_eq!(exec.schema().field(0).name(), "a");
        assert_eq!(exec.schema().field(1).name(), ROWID_COLUMN_NAME);
        assert_eq!(exec.schema().field(2).name(), "b");

        let ctx = Arc::new(TaskContext::default());
        let mut stream = exec.execute(0, ctx).unwrap();
        use futures::StreamExt;

        let out = stream.next().await.unwrap().unwrap();
        let rowids = out
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .values()
            .to_vec();
        assert_eq!(rowids, vec![500, 501]);
        // Verify physical columns still in place around rowid
        let a = out
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap()
            .values()
            .to_vec();
        let b = out
            .column(2)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap()
            .values()
            .to_vec();
        assert_eq!(a, vec![10, 20]);
        assert_eq!(b, vec![100, 200]);
    }

    #[tokio::test]
    async fn emits_null_when_row_id_start_is_none() {
        let input_schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int32, false)]));
        let b = small_batch(input_schema.clone(), &[1, 2]);
        let mem =
            MemorySourceConfig::try_new_exec(&[vec![b]], input_schema.clone(), None).unwrap();

        let exec = Arc::new(RowIdExec::new(mem, None));
        let ctx = Arc::new(TaskContext::default());
        let mut stream = exec.execute(0, ctx).unwrap();
        use futures::StreamExt;

        let batch = stream.next().await.unwrap().unwrap();
        let rowid_col = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(rowid_col.len(), 2);
        assert!(rowid_col.is_null(0));
        assert!(rowid_col.is_null(1));
    }
}
