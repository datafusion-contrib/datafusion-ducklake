//! Physical row position and synthetic `rowid` column injection for DuckLake
//! row lineage.
//!
//! DuckLake assigns each row a globally unique `rowid` BIGINT. For files written
//! by INSERT, the catalog records the file's `row_id_start`, and the per-row
//! rowid is `row_id_start + physical_row_position`, where `physical_row_position`
//! is the row's 0-based position in the physical Parquet file. Positional delete
//! files use the same physical position in their `pos` column.
//!
//! The physical position is **not** derivable from stream arrival order: once a
//! scan prunes row groups, selects rows, or splits a file across partitions,
//! arrival order no longer tracks file order. It comes instead from the parquet
//! reader itself, as a virtual column carrying the `RowNumber` extension type
//! (see `positional_table_schema`). The reader knows each row group's absolute
//! first row from the footer, so the values stay true physical positions however
//! the scan is pruned, filtered, split or reordered.
//!
//! This mirrors official DuckLake, which computes `rowid` the same way from
//! DuckDB's reader-level `COLUMN_IDENTIFIER_FILE_ROW_NUMBER` virtual column
//! (`ducklake_multi_file_reader.cpp::GetVirtualColumnExpression`).
//!
//! Downstream, [`RowIdExec`] reads that column to compute `rowid`, and
//! `DeleteFilterExec` reads it to filter deleted positions — neither counts
//! stream rows, so both are correct regardless of partitioning or merge order.
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
use arrow::datatypes::{DataType, Field, FieldRef, Fields, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use datafusion::common::config::ConfigOptions;
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::datasource::table_schema::TableSchema;
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::execution::{RecordBatchStream, SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_plan::filter_pushdown::{
    ChildFilterDescription, FilterDescription, FilterPushdownPhase,
};
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, PlanProperties,
};
use futures::Stream;
use parquet::arrow::RowNumber;

/// Name of the synthetic rowid column exposed when row lineage is enabled.
pub const ROWID_COLUMN_NAME: &str = "rowid";

/// Base name of the internal physical-row-position column: a parquet virtual
/// column consumed by [`RowIdExec`] / `DeleteFilterExec`. Projected away before
/// the table's output schema (by `ColumnRenameExec`), so it never reaches the
/// user, and never written to a parquet file or the catalog.
///
/// The double-underscore prefix makes a clash with a real catalog column
/// unlikely but not impossible — DuckLake reserves no names — so a scan whose
/// file already has a column of this name uses a suffixed variant instead. See
/// `unique_row_pos_name`.
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

/// Reserved parquet field-id for the per-row snapshot-id column embedded in a
/// **partial data file** produced by `merge_adjacent_files`. Matches the
/// DuckLake extension's `_ducklake_internal_snapshot_id` field-id
/// (`ROW_ID_PARQUET_FIELD_ID - 1`); the two reserved ids sit adjacent below
/// `i32::MAX` so neither collides with a catalog `column_id` (which are small
/// positive ints assigned from `next_column_id`).
pub const SNAPSHOT_ID_PARQUET_FIELD_ID: i32 = 2_147_483_539;

/// Parquet column name our writer uses for the embedded snapshot-id column on
/// merged partial files. The read path matches the column by its
/// [`SNAPSHOT_ID_PARQUET_FIELD_ID`] field-id, not this name, so the exact
/// string is cosmetic; we mirror the DuckLake extension's
/// `_ducklake_internal_snapshot_id`.
pub const EMBEDDED_SNAPSHOT_ID_COLUMN_NAME: &str = "_ducklake_internal_snapshot_id";

/// Build the Arrow [`Field`] for the embedded per-row snapshot-id column written
/// into a merged partial file (`merge_adjacent_files`). Each value is the
/// snapshot in which that row originally became visible, so time-travel / CDC
/// can still attribute a merged row to its origin snapshot. Carries the reserved
/// [`SNAPSHOT_ID_PARQUET_FIELD_ID`] as its `PARQUET:field_id` metadata; because
/// that id matches no catalog column and is not the rowid id, the standard read
/// path (which maps parquet columns to catalog columns strictly by field-id)
/// simply ignores it. Nullable for parity with the other embedded columns.
pub fn embedded_snapshot_id_field() -> Field {
    let mut metadata = std::collections::HashMap::new();
    metadata.insert(
        "PARQUET:field_id".to_string(),
        SNAPSHOT_ID_PARQUET_FIELD_ID.to_string(),
    );
    Field::new(EMBEDDED_SNAPSHOT_ID_COLUMN_NAME, DataType::Int64, true).with_metadata(metadata)
}

/// Build the Arrow Field for the rowid column. Nullable so we can emit NULL
/// for files whose catalog row_id_start is unrecorded (e.g. older catalogs).
pub fn rowid_field() -> Field {
    Field::new(ROWID_COLUMN_NAME, DataType::Int64, true)
}

/// Build the Arrow field for the internal physical-position column, tagged with
/// the parquet `RowNumber` virtual extension type.
///
/// A field carrying this extension type is what tells DataFusion's parquet
/// opener to have the reader itself produce each row's absolute position in the
/// file (`ArrowReaderOptions::with_virtual_columns`). Values are therefore true
/// physical positions under row-group pruning, row-level selection, byte-range
/// splitting and reverse-order reads alike — the property every consumer of
/// [`ROW_POS_COLUMN_NAME`] depends on.
///
/// `name` is normally [`ROW_POS_COLUMN_NAME`]; see [`unique_row_pos_name`] for
/// why it is not always.
pub(crate) fn row_pos_virtual_field(name: &str) -> FieldRef {
    Arc::new(Field::new(name, DataType::Int64, false).with_extension_type(RowNumber))
}

/// Pick a name for the internal position column that no field of `read_schema`
/// already uses.
///
/// [`ROW_POS_COLUMN_NAME`] is not reserved: DuckLake places no restriction on
/// column names (official DuckLake validates none either), so a table may
/// legitimately have a column called `__ducklake_row_pos`. A duplicate name in
/// the scan's table schema is only a `debug_assert!` in DataFusion, so a release
/// build would silently resolve the wrong column. Suffix until unused, the same
/// way DataFusion disambiguates its own internal row-index column.
pub(crate) fn unique_row_pos_name(read_schema: &Schema) -> String {
    if read_schema.field_with_name(ROW_POS_COLUMN_NAME).is_err() {
        return ROW_POS_COLUMN_NAME.to_string();
    }
    let mut suffix = 1;
    loop {
        let candidate = format!("{ROW_POS_COLUMN_NAME}_{suffix}");
        if read_schema.field_with_name(&candidate).is_err() {
            return candidate;
        }
        suffix += 1;
    }
}

/// Build the scan schema for a *positional* read of `read_schema`: the file's
/// own columns plus the reader-produced physical-position column appended last.
///
/// Returns the [`TableSchema`] to hand [`ParquetSource`] and the position
/// column's index within it, which is what a caller appends to its projection.
///
/// [`ParquetSource`]: datafusion::datasource::physical_plan::ParquetSource
pub(crate) fn positional_table_schema(read_schema: SchemaRef) -> (TableSchema, usize) {
    let name = unique_row_pos_name(read_schema.as_ref());
    let pos_index = read_schema.fields().len();
    let table_schema = TableSchema::builder(read_schema)
        .with_virtual_columns(Fields::from(vec![row_pos_virtual_field(&name)]))
        .build();
    (table_schema, pos_index)
}

// ---------------------------------------------------------------------------
// RowIdExec — derive rowid from the physical-position column
// ---------------------------------------------------------------------------

/// Execution plan that appends a synthetic `rowid` BIGINT column computed as
/// `row_id_start + __ducklake_row_pos`, reading the reader-produced position
/// column (possibly via a `DeleteFilterExec`).
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
    /// Build a `RowIdExec` over an input whose column `pos_index` is the
    /// reader-produced physical row position.
    ///
    /// The index is passed in rather than looked up by name because the
    /// position column's name is per-scan (see `unique_row_pos_name`), and
    /// because a name lookup would silently bind a user column of the same name.
    pub fn try_new(
        input: Arc<dyn ExecutionPlan>,
        row_id_start: Option<i64>,
        pos_index: usize,
    ) -> DataFusionResult<Self> {
        let input_schema = input.schema();
        if input_schema.field(pos_index).data_type() != &DataType::Int64 {
            return Err(DataFusionError::Internal(format!(
                "RowIdExec: column {pos_index} (`{}`) is not the Int64 physical-position column",
                input_schema.field(pos_index).name()
            )));
        }

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
    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> DataFusionResult<TreeNodeRecursion>,
    ) -> DataFusionResult<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

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

    /// Forward filter pushdown for every column the input already has.
    ///
    /// This node appends `rowid` and changes nothing else, so a predicate over
    /// the input's own columns means the same thing above and below it and can
    /// be offered to the child unchanged (indices are identical — `rowid` is
    /// appended last).
    ///
    /// A predicate on `rowid` itself is rejected, and terminally so: `rowid` is
    /// `row_id_start + position`, and DataFusion refuses any pushed predicate
    /// that references a virtual column, which the position column is.
    fn gather_filters_for_pushdown(
        &self,
        _phase: FilterPushdownPhase,
        parent_filters: Vec<Arc<dyn PhysicalExpr>>,
        _config: &ConfigOptions,
    ) -> DataFusionResult<FilterDescription> {
        let allowed = (0..self.input.schema().fields().len()).collect();
        let child = ChildFilterDescription::from_child_with_allowed_indices(
            &parent_filters,
            allowed,
            &self.input,
        )?;
        Ok(FilterDescription::new().with_child(child))
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
            self.pos_index,
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

    /// Build an input batch shaped like a positional scan's output: a value
    /// column `v` plus the reader-produced physical-position column.
    fn batch_with_pos(values: &[i32], positions: &[i64]) -> (SchemaRef, RecordBatch) {
        let schema = Arc::new(Schema::new(vec![
            Arc::new(Field::new("v", DataType::Int32, false)),
            row_pos_virtual_field(ROW_POS_COLUMN_NAME),
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

    // --- RowIdExec ---

    #[tokio::test]
    async fn rowid_is_start_plus_position() {
        // Positions deliberately non-contiguous to prove rowid reads the column
        // rather than counting arrivals.
        let (schema, batch) = batch_with_pos(&[10, 20, 30], &[5, 6, 9]);
        let mem = MemorySourceConfig::try_new_exec(&[vec![batch]], schema, None).unwrap();
        let exec = Arc::new(RowIdExec::try_new(mem, Some(1000), 1).unwrap());

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
        let exec = Arc::new(RowIdExec::try_new(mem, None, 1).unwrap());
        let ctx = Arc::new(TaskContext::default());
        let mut s = exec.execute(0, ctx).unwrap();
        let out = s.next().await.unwrap().unwrap();
        let rowid = out.column(2).as_any().downcast_ref::<Int64Array>().unwrap();
        assert!(rowid.is_null(0) && rowid.is_null(1));
    }

    #[test]
    fn rowid_errors_when_position_column_is_not_int64() {
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int32, false)]));
        let mem = MemorySourceConfig::try_new_exec(&[vec![]], schema, None).unwrap();
        assert!(
            RowIdExec::try_new(mem, Some(0), 0).is_err(),
            "RowIdExec must reject a non-Int64 position column"
        );
    }

    #[test]
    fn row_pos_name_avoids_a_user_column_of_the_same_name() {
        // DuckLake reserves no column names, so a table may legitimately have a
        // column called `__ducklake_row_pos`.
        let plain = Schema::new(vec![Field::new("v", DataType::Int32, false)]);
        assert_eq!(unique_row_pos_name(&plain), ROW_POS_COLUMN_NAME);

        let clash = Schema::new(vec![
            Field::new("v", DataType::Int32, false),
            Field::new(ROW_POS_COLUMN_NAME, DataType::Utf8, true),
        ]);
        assert_eq!(
            unique_row_pos_name(&clash),
            format!("{ROW_POS_COLUMN_NAME}_1")
        );

        let clash_twice = Schema::new(vec![
            Field::new(ROW_POS_COLUMN_NAME, DataType::Utf8, true),
            Field::new(format!("{ROW_POS_COLUMN_NAME}_1"), DataType::Utf8, true),
        ]);
        assert_eq!(
            unique_row_pos_name(&clash_twice),
            format!("{ROW_POS_COLUMN_NAME}_2")
        );
    }

    #[test]
    fn positional_table_schema_appends_the_virtual_column_last() {
        let read_schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Utf8, true),
        ]));
        let (table_schema, pos_index) = positional_table_schema(read_schema);
        assert_eq!(pos_index, 2);
        let full = table_schema.table_schema();
        assert_eq!(full.fields().len(), 3);
        assert_eq!(full.field(2).name(), ROW_POS_COLUMN_NAME);
        assert_eq!(
            full.field(2).extension_type_name(),
            // arrow-rs `RowNumber::NAME`; spelled out so the test pins the wire
            // contract rather than restating whatever the constant happens to be.
            Some("parquet.virtual.row_number"),
            "the reader recognises the position column by its extension type"
        );
    }
}
