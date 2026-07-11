//! Physical row position and synthetic `rowid` column injection for DuckLake
//! row lineage.
//!
//! DuckLake assigns each row a globally unique `rowid` BIGINT. For files written
//! by INSERT, the catalog records the file's `row_id_start`, and the per-row
//! rowid is `row_id_start + physical_row_position`, where `physical_row_position`
//! is the row's 0-based position in the physical Parquet file. Positional delete
//! files use the same physical position in their `pos` column.
//!
//! The physical position is **not** derivable from stream arrival order: when
//! DataFusion splits a file across scan partitions and merges them, arrival
//! order no longer matches file order. Instead we:
//!
//! 1. partition the scan on row-group boundaries (so each partition's first
//!    physical row is known — see `table.rs::build_row_group_partitions`), then
//! 2. materialize the physical position as an internal column
//!    ([`ROW_POS_COLUMN_NAME`]) with [`FileRowNumberExec`], seeding each
//!    partition's counter from that partition's starting row.
//!
//! Downstream, [`RowIdExec`] reads that column to compute `rowid`, and
//! `DeleteFilterExec` reads it to filter deleted positions — neither counts
//! stream rows, so both are correct regardless of partitioning or merge order.
//! DataFusion upstream is adding Parquet metadata / virtual-column support
//! (apache/datafusion#20135, apache/datafusion#22026). If a future DataFusion
//! release exposes a Parquet physical `row_number` column, prefer that
//! reader-level source for [`ROW_POS_COLUMN_NAME`]: it can preserve more Parquet
//! pruning while still producing true physical positions.
//!
//! Files written by `UPDATE` / compaction store the original rowids inline in
//! the parquet as a column tagged with [`ROW_ID_PARQUET_FIELD_ID`] (typically
//! named `_ducklake_internal_row_id`). Those files do NOT use [`RowIdExec`] —
//! `DuckLakeTable` reads the embedded column directly via the parquet scan and
//! renames it. See `table.rs::build_exec_for_file_with_rowid`.

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

/// Name of the internal physical-row-position column produced by
/// [`FileRowNumberExec`] and consumed by [`RowIdExec`] / `DeleteFilterExec`.
/// Projected away before the table's output schema (by `ColumnRenameExec`),
/// so it never reaches the user. The double-underscore prefix avoids collisions
/// with real catalog columns.
pub const ROW_POS_COLUMN_NAME: &str = "__ducklake_row_pos";

/// Iceberg / DuckLake reserved parquet field-id for the row-id column.
/// Matches `MultiFileReader::ROW_ID_FIELD_ID` in DuckDB
/// (`duckdb/src/include/duckdb/common/multi_file/multi_file_reader.hpp`).
/// Files written by `UPDATE` / compaction embed a column tagged with this
/// field-id (typically named `_ducklake_internal_row_id`) so original rowids
/// survive across file rewrites.
pub const ROW_ID_PARQUET_FIELD_ID: i32 = 2_147_483_540;

/// Parquet column name our writer uses for the embedded row-id column on files
/// produced by `UPDATE` / compaction. The read path matches the column by its
/// [`ROW_ID_PARQUET_FIELD_ID`] field-id, not this name, so the exact string is
/// cosmetic; we mirror the DuckLake extension's `_ducklake_internal_row_id`.
pub const EMBEDDED_ROW_ID_COLUMN_NAME: &str = "_ducklake_internal_row_id";

/// Build the Arrow [`Field`] for the embedded row-id column written into
/// `UPDATE` / compaction output parquet. Carries the reserved
/// [`ROW_ID_PARQUET_FIELD_ID`] as its `PARQUET:field_id` metadata so a later
/// read detects it (see `table.rs::build_file_read_config`) and serves the
/// original rowids inline rather than synthesizing `row_id_start + position`.
/// Nullable to match the read-side `rowid` field.
pub fn embedded_rowid_field() -> Field {
    let mut metadata = std::collections::HashMap::new();
    metadata.insert(
        "PARQUET:field_id".to_string(),
        ROW_ID_PARQUET_FIELD_ID.to_string(),
    );
    Field::new(EMBEDDED_ROW_ID_COLUMN_NAME, DataType::Int64, true).with_metadata(metadata)
}

/// Build the Arrow Field for the rowid column. Nullable so we can emit NULL
/// for files whose catalog row_id_start is unrecorded (e.g. older catalogs).
pub fn rowid_field() -> Field {
    Field::new(ROWID_COLUMN_NAME, DataType::Int64, true)
}

/// Build the Arrow Field for the internal physical-position column. Non-null:
/// every row has a well-defined physical position.
pub fn row_pos_field() -> Field {
    Field::new(ROW_POS_COLUMN_NAME, DataType::Int64, false)
}

// ---------------------------------------------------------------------------
// FileRowNumberExec — materialize the true physical row position as a column
// ---------------------------------------------------------------------------

/// Execution plan that appends an internal [`ROW_POS_COLUMN_NAME`] BIGINT column
/// holding each row's **true physical position in the file**.
///
/// Correctness rests on a precondition enforced by its construction in
/// `table.rs`: the input is a row-group-aligned, non-repartitionable,
/// non-pruning scan, so partition `p` emits a complete, contiguous, in-order run
/// of physical rows beginning at `partition_starts[p]`. The per-partition cursor
/// then equals the physical position. Once materialized, the column travels with
/// each row, so any reordering above this exec is harmless.
#[derive(Debug)]
pub struct FileRowNumberExec {
    input: Arc<dyn ExecutionPlan>,
    /// Starting physical row position for each input partition (1:1 with the
    /// scan's file groups).
    partition_starts: Arc<Vec<i64>>,
    /// Output schema = input schema with [`ROW_POS_COLUMN_NAME`] appended.
    schema: SchemaRef,
    properties: Arc<PlanProperties>,
}

impl FileRowNumberExec {
    pub fn new(input: Arc<dyn ExecutionPlan>, partition_starts: Vec<i64>) -> Self {
        let mut fields: Vec<Arc<Field>> = input.schema().fields().iter().cloned().collect();
        fields.push(Arc::new(row_pos_field()));
        let schema = Arc::new(Schema::new(fields));

        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(schema.clone()),
            input.output_partitioning().clone(),
            input.pipeline_behavior(),
            input.boundedness(),
        ));

        Self {
            input,
            partition_starts: Arc::new(partition_starts),
            schema,
            properties,
        }
    }
}

impl DisplayAs for FileRowNumberExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "FileRowNumberExec: starts={:?}",
            self.partition_starts.as_ref()
        )
    }
}

impl ExecutionPlan for FileRowNumberExec {
    fn name(&self) -> &str {
        "FileRowNumberExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    /// Order-preserving: appends a column without reordering or dropping rows.
    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true]
    }

    /// Refuse extra input partitioning. Our per-partition seeds are 1:1 with the
    /// scan's row-group-aligned file groups; if `EnforceDistribution` inserted a
    /// round-robin `RepartitionExec` below us to parallelize, the child would
    /// report more partitions than we have seeds (and rows would be shuffled out
    /// of physical order). Returning `false` keeps exactly the scan's partitions.
    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![false]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(DataFusionError::Internal(
                "FileRowNumberExec expects exactly one child".into(),
            ));
        }
        Ok(Arc::new(FileRowNumberExec::new(
            children.into_iter().next().unwrap(),
            self.partition_starts.as_ref().clone(),
        )))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        let start = *self.partition_starts.get(partition).ok_or_else(|| {
            DataFusionError::Internal(format!(
                "FileRowNumberExec: no starting position for partition {partition} \
                 (have {} partitions)",
                self.partition_starts.len()
            ))
        })?;
        Ok(Box::pin(FileRowNumberStream {
            input: self.input.execute(partition, context)?,
            schema: self.schema.clone(),
            cursor: start,
        }))
    }
}

struct FileRowNumberStream {
    input: SendableRecordBatchStream,
    schema: SchemaRef,
    /// Physical position of the first row of the next batch.
    cursor: i64,
}

impl Stream for FileRowNumberStream {
    type Item = DataFusionResult<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match Pin::new(&mut self.input).poll_next(cx) {
            Poll::Ready(Some(Ok(batch))) => {
                let n = batch.num_rows();
                let mut builder = Int64Array::builder(n);
                for i in 0..n {
                    builder.append_value(self.cursor + i as i64);
                }
                self.cursor += n as i64;

                let mut cols: Vec<ArrayRef> = batch.columns().to_vec();
                cols.push(Arc::new(builder.finish()));
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

impl RecordBatchStream for FileRowNumberStream {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

// ---------------------------------------------------------------------------
// RowIdExec — derive rowid from the physical-position column
// ---------------------------------------------------------------------------

/// Execution plan that appends a synthetic `rowid` BIGINT column computed as
/// `row_id_start + __ducklake_row_pos`, reading the position column produced by
/// [`FileRowNumberExec`] (possibly via a `DeleteFilterExec`).
///
/// Stateless w.r.t. row order: it reads a per-row value and appends a per-row
/// value, so it is correct under any partitioning. The position column is passed
/// through unchanged for any downstream consumer; the final projection
/// (`ColumnRenameExec`) drops it. If `row_id_start` is `None` the rowid column is
/// emitted as all-NULL (the per-file plan in `table.rs` hard-errors before
/// reaching here for non-embedded files with no `row_id_start`, so this is a
/// defensive fallback only).
#[derive(Debug)]
pub struct RowIdExec {
    input: Arc<dyn ExecutionPlan>,
    row_id_start: Option<i64>,
    /// Index of [`ROW_POS_COLUMN_NAME`] in the input schema.
    pos_index: usize,
    /// Output schema = input schema with `rowid` appended.
    schema: SchemaRef,
    properties: Arc<PlanProperties>,
}

impl RowIdExec {
    /// Build a `RowIdExec`. The input must carry the [`ROW_POS_COLUMN_NAME`]
    /// column (produced by [`FileRowNumberExec`]); errors otherwise.
    pub fn try_new(
        input: Arc<dyn ExecutionPlan>,
        row_id_start: Option<i64>,
    ) -> DataFusionResult<Self> {
        let input_schema = input.schema();
        let pos_index = input_schema.index_of(ROW_POS_COLUMN_NAME).map_err(|_| {
            DataFusionError::Internal(format!(
                "RowIdExec input is missing the `{ROW_POS_COLUMN_NAME}` column"
            ))
        })?;

        let mut fields: Vec<Arc<Field>> = input_schema.fields().iter().cloned().collect();
        fields.push(Arc::new(rowid_field()));
        let schema = Arc::new(Schema::new(fields));

        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(schema.clone()),
            input.output_partitioning().clone(),
            input.pipeline_behavior(),
            input.boundedness(),
        ));

        Ok(Self {
            input,
            row_id_start,
            pos_index,
            schema,
            properties,
        })
    }
}

impl DisplayAs for RowIdExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "RowIdExec: row_id_start={}",
            self.row_id_start
                .map_or_else(|| "NULL".to_string(), |v| v.to_string())
        )
    }
}

impl ExecutionPlan for RowIdExec {
    fn name(&self) -> &str {
        "RowIdExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    /// Order-preserving column append. No distribution requirement: the rowid
    /// value is computed from the per-row position column, so it is correct
    /// regardless of how the input is partitioned or merged.
    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true]
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
        Ok(Arc::new(RowIdExec::try_new(
            children.into_iter().next().unwrap(),
            self.row_id_start,
        )?))
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
            pos_index: self.pos_index,
        }))
    }
}

struct RowIdStream {
    input: SendableRecordBatchStream,
    schema: SchemaRef,
    row_id_start: Option<i64>,
    pos_index: usize,
}

impl Stream for RowIdStream {
    type Item = DataFusionResult<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match Pin::new(&mut self.input).poll_next(cx) {
            Poll::Ready(Some(Ok(batch))) => {
                let n = batch.num_rows();
                let rowid_col: ArrayRef = match self.row_id_start {
                    Some(start) => {
                        let pos = match batch
                            .column(self.pos_index)
                            .as_any()
                            .downcast_ref::<Int64Array>()
                        {
                            Some(p) => p,
                            None => {
                                return Poll::Ready(Some(Err(DataFusionError::Internal(format!(
                                    "`{ROW_POS_COLUMN_NAME}` column is not Int64"
                                )))));
                            },
                        };
                        let mut builder = Int64Array::builder(n);
                        for i in 0..n {
                            builder.append_value(start + pos.value(i));
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

                let mut cols: Vec<ArrayRef> = batch.columns().to_vec();
                cols.push(rowid_col);
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
    use futures::StreamExt;

    /// Build an input batch shaped like a `FileRowNumberExec` output: a value
    /// column `v` plus the internal `__ducklake_row_pos` column.
    fn batch_with_pos(values: &[i32], positions: &[i64]) -> (SchemaRef, RecordBatch) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("v", DataType::Int32, false),
            row_pos_field(),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(values.to_vec())) as ArrayRef,
                Arc::new(Int64Array::from(positions.to_vec())) as ArrayRef,
            ],
        )
        .unwrap();
        (schema, batch)
    }

    // --- FileRowNumberExec ---

    #[tokio::test]
    async fn file_row_number_seeds_per_partition() {
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int32, false)]));
        let mk = |vals: Vec<i32>| {
            RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(Int32Array::from(vals)) as ArrayRef],
            )
            .unwrap()
        };
        // Two partitions: partition 0 starts at 0, partition 1 starts at 100.
        let mem = MemorySourceConfig::try_new_exec(
            &[vec![mk(vec![1, 2, 3])], vec![mk(vec![4, 5])]],
            schema.clone(),
            None,
        )
        .unwrap();
        let exec = Arc::new(FileRowNumberExec::new(mem, vec![0, 100]));
        assert_eq!(exec.schema().field(1).name(), ROW_POS_COLUMN_NAME);

        let ctx = Arc::new(TaskContext::default());
        let pos_of = |b: &RecordBatch| {
            b.column(1)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .values()
                .to_vec()
        };

        let mut s0 = exec.clone().execute(0, ctx.clone()).unwrap();
        let p0 = s0.next().await.unwrap().unwrap();
        assert_eq!(pos_of(&p0), vec![0, 1, 2]);

        let mut s1 = exec.execute(1, ctx).unwrap();
        let p1 = s1.next().await.unwrap().unwrap();
        assert_eq!(
            pos_of(&p1),
            vec![100, 101],
            "partition 1 seeds at its start"
        );
    }

    #[tokio::test]
    async fn file_row_number_continues_across_batches() {
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int32, false)]));
        let mk = |vals: Vec<i32>| {
            RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(Int32Array::from(vals)) as ArrayRef],
            )
            .unwrap()
        };
        let mem = MemorySourceConfig::try_new_exec(
            &[vec![mk(vec![1, 2, 3]), mk(vec![4, 5])]],
            schema.clone(),
            None,
        )
        .unwrap();
        let exec = Arc::new(FileRowNumberExec::new(mem, vec![10]));
        let ctx = Arc::new(TaskContext::default());
        let mut s = exec.execute(0, ctx).unwrap();

        let pos_of = |b: &RecordBatch| {
            b.column(1)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .values()
                .to_vec()
        };
        assert_eq!(pos_of(&s.next().await.unwrap().unwrap()), vec![10, 11, 12]);
        assert_eq!(pos_of(&s.next().await.unwrap().unwrap()), vec![13, 14]);
    }

    #[tokio::test]
    async fn file_row_number_zero_column_input() {
        // COUNT(*)-style input: zero data columns, only a row count.
        let schema = Arc::new(Schema::new(Vec::<Field>::new()));
        let batch = RecordBatch::try_new_with_options(
            schema.clone(),
            vec![],
            &arrow::record_batch::RecordBatchOptions::new().with_row_count(Some(4)),
        )
        .unwrap();
        let mem = MemorySourceConfig::try_new_exec(&[vec![batch]], schema, None).unwrap();
        let exec = Arc::new(FileRowNumberExec::new(mem, vec![7]));
        let ctx = Arc::new(TaskContext::default());
        let mut s = exec.execute(0, ctx).unwrap();
        let out = s.next().await.unwrap().unwrap();
        assert_eq!(out.num_columns(), 1);
        assert_eq!(
            out.column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .values()
                .to_vec(),
            vec![7, 8, 9, 10]
        );
    }

    // --- RowIdExec ---

    #[tokio::test]
    async fn rowid_is_start_plus_position() {
        // Positions deliberately non-contiguous to prove rowid reads the column
        // rather than counting arrivals.
        let (schema, batch) = batch_with_pos(&[10, 20, 30], &[5, 6, 9]);
        let mem = MemorySourceConfig::try_new_exec(&[vec![batch]], schema, None).unwrap();
        let exec = Arc::new(RowIdExec::try_new(mem, Some(1000)).unwrap());

        // Output appends rowid after the input columns (v, pos, rowid).
        assert_eq!(exec.schema().field(2).name(), ROWID_COLUMN_NAME);

        let ctx = Arc::new(TaskContext::default());
        let mut s = exec.execute(0, ctx).unwrap();
        let out = s.next().await.unwrap().unwrap();
        let rowids = out
            .column(2)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .values()
            .to_vec();
        assert_eq!(rowids, vec![1005, 1006, 1009]);
        // Position column passed through unchanged for downstream consumers.
        assert_eq!(
            out.column(1)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .values()
                .to_vec(),
            vec![5, 6, 9]
        );
    }

    #[tokio::test]
    async fn rowid_null_when_start_is_none() {
        let (schema, batch) = batch_with_pos(&[1, 2], &[0, 1]);
        let mem = MemorySourceConfig::try_new_exec(&[vec![batch]], schema, None).unwrap();
        let exec = Arc::new(RowIdExec::try_new(mem, None).unwrap());
        let ctx = Arc::new(TaskContext::default());
        let mut s = exec.execute(0, ctx).unwrap();
        let out = s.next().await.unwrap().unwrap();
        let rowid = out.column(2).as_any().downcast_ref::<Int64Array>().unwrap();
        assert!(rowid.is_null(0) && rowid.is_null(1));
    }

    #[test]
    fn rowid_errors_without_position_column() {
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int32, false)]));
        let mem = MemorySourceConfig::try_new_exec(&[vec![]], schema, None).unwrap();
        assert!(
            RowIdExec::try_new(mem, Some(0)).is_err(),
            "RowIdExec must require the position column"
        );
    }
}
