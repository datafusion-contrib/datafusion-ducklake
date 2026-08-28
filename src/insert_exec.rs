//! DuckLake INSERT execution plan.
//!
//! Limitations:
//! - Collects all batches into memory before writing (no streaming yet)
//! - Single partition only (partition 0)

use std::fmt::{self, Debug};
use std::sync::Arc;

use arrow::array::{ArrayRef, RecordBatch, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::execution::object_store::ObjectStoreUrl;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_expr::{EquivalenceProperties, Partitioning};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use futures::stream::{self, TryStreamExt};

use crate::metadata_writer::{MetadataWriter, WriteMode};
use crate::table_writer::DuckLakeTableWriter;

// The resolved write-side partition spec lives in `partition` (it is shared with
// the low-level writer paths and the UPDATE rewrite); re-exported here because
// `DuckLakeInsertExec::new` takes it.
pub use crate::partition::{PartitionWriteKey, PartitionWriteSpec};

/// Schema for the output of insert operations (count of rows inserted)
fn make_insert_count_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new(
        "count",
        DataType::UInt64,
        false,
    )]))
}

/// Execution plan that writes input data to a DuckLake table.
pub struct DuckLakeInsertExec {
    input: Arc<dyn ExecutionPlan>,
    writer: Arc<dyn MetadataWriter>,
    schema_name: String,
    table_name: String,
    arrow_schema: SchemaRef,
    write_mode: WriteMode,
    object_store_url: Arc<ObjectStoreUrl>,
    /// When set, the target table is partitioned: input rows are split by the
    /// transformed partition key into one file per partition, all committed in a
    /// single snapshot. `None` for an unpartitioned table (single-file write).
    partition: Option<PartitionWriteSpec>,
    /// Write-layout options applied to the DuckLakeTableWriter built at execute
    /// time (compression, row-group caps, file-rollover target).
    write_options: crate::table_writer::DuckLakeWriteOptions,
    /// The sort order this insert requires of its input (the table's live sort
    /// spec). Declared via `required_input_ordering` so DataFusion's EnforceSorting
    /// keeps the input sorted instead of pruning the SortExec as unused.
    required_ordering: Option<datafusion::physical_expr::LexOrdering>,
    cache: Arc<PlanProperties>,
}

impl DuckLakeInsertExec {
    /// Create a new DuckLakeInsertExec
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input: Arc<dyn ExecutionPlan>,
        writer: Arc<dyn MetadataWriter>,
        schema_name: String,
        table_name: String,
        arrow_schema: SchemaRef,
        write_mode: WriteMode,
        object_store_url: Arc<ObjectStoreUrl>,
        partition: Option<PartitionWriteSpec>,
        write_options: crate::table_writer::DuckLakeWriteOptions,
        required_ordering: Option<datafusion::physical_expr::LexOrdering>,
    ) -> Self {
        let cache = Self::compute_properties();
        Self {
            input,
            writer,
            schema_name,
            table_name,
            arrow_schema,
            write_mode,
            object_store_url,
            partition,
            write_options,
            required_ordering,
            cache,
        }
    }

    fn compute_properties() -> Arc<PlanProperties> {
        Arc::new(PlanProperties::new(
            EquivalenceProperties::new(make_insert_count_schema()),
            Partitioning::UnknownPartitioning(1),
            datafusion::physical_plan::execution_plan::EmissionType::Final,
            datafusion::physical_plan::execution_plan::Boundedness::Bounded,
        ))
    }
}

impl Debug for DuckLakeInsertExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DuckLakeInsertExec")
            .field("schema_name", &self.schema_name)
            .field("table_name", &self.table_name)
            .field("write_mode", &self.write_mode)
            .finish_non_exhaustive()
    }
}

impl DisplayAs for DuckLakeInsertExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        match t {
            DisplayFormatType::Default
            | DisplayFormatType::Verbose
            | DisplayFormatType::TreeRender => {
                write!(
                    f,
                    "DuckLakeInsertExec: schema={}, table={}, mode={:?}",
                    self.schema_name, self.table_name, self.write_mode
                )
            },
        }
    }
}

impl ExecutionPlan for DuckLakeInsertExec {
    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> DataFusionResult<TreeNodeRecursion>,
    ) -> DataFusionResult<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn name(&self) -> &str {
        "DuckLakeInsertExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.cache
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    /// Require all input rows in a single partition.
    ///
    /// `execute` only drives `input.execute(0)`, so without this DataFusion
    /// would feed a multi-partition input (e.g. a parallel scan or aggregation)
    /// straight through and partitions `1..N` would be silently dropped. Asking
    /// for `SinglePartition` makes the optimizer insert a `CoalescePartitionsExec`
    /// that merges every input partition into partition 0 before we read it.
    fn required_input_distribution(&self) -> Vec<datafusion::physical_expr::Distribution> {
        vec![datafusion::physical_expr::Distribution::SinglePartition]
    }

    /// Require the input sorted by the table's live sort order (when one exists),
    /// so rows are laid out sorted within each written file. Declaring it as a hard
    /// requirement stops EnforceSorting from pruning the SortExec `insert_into`
    /// added (a sort with no downstream requirement is otherwise removed).
    fn required_input_ordering(
        &self,
    ) -> Vec<Option<datafusion::physical_expr_common::sort_expr::OrderingRequirements>> {
        vec![self.required_ordering.clone().map(|ordering| {
            datafusion::physical_expr_common::sort_expr::OrderingRequirements::from(ordering)
        })]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(DataFusionError::Plan(
                "DuckLakeInsertExec requires exactly one child".to_string(),
            ));
        }
        Ok(Arc::new(Self::new(
            Arc::clone(&children[0]),
            Arc::clone(&self.writer),
            self.schema_name.clone(),
            self.table_name.clone(),
            Arc::clone(&self.arrow_schema),
            self.write_mode,
            self.object_store_url.clone(),
            self.partition.clone(),
            self.write_options.clone(),
            self.required_ordering.clone(),
        )))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(DataFusionError::Internal(format!(
                "DuckLakeInsertExec only supports partition 0, got {}",
                partition
            )));
        }

        let input = Arc::clone(&self.input);
        let writer = Arc::clone(&self.writer);
        let schema_name = self.schema_name.clone();
        let table_name = self.table_name.clone();
        let arrow_schema = Arc::clone(&self.arrow_schema);
        let write_mode = self.write_mode;
        let object_store_url = self.object_store_url.clone();
        let partition = self.partition.clone();
        let write_options = self.write_options.clone();
        let output_schema = make_insert_count_schema();

        let stream = stream::once(async move {
            let input_stream = input.execute(0, Arc::clone(&context))?;
            let batches: Vec<RecordBatch> = input_stream.try_collect().await?;

            // An empty input is a genuine no-op only for Append. For
            // Replace/Overwrite we must still run the write session so the prior
            // generation is retired (truncated): finish() registers a 0-row file
            // and finalize_snapshot runs the Replace retirement. Returning early
            // here would leave the old rows live while reporting count=0 success.
            if batches.is_empty() && write_mode == WriteMode::Append {
                let count_array: ArrayRef = Arc::new(UInt64Array::from(vec![0u64]));
                return Ok(RecordBatch::try_new(output_schema, vec![count_array])?);
            }

            // Get object store from runtime environment
            let object_store = context
                .runtime_env()
                .object_store(object_store_url.as_ref())?;

            let table_writer = DuckLakeTableWriter::new(writer, object_store)
                .map_err(|e| DataFusionError::External(Box::new(e)))?
                .with_options(&write_options);

            let schema_without_metadata =
                Schema::new(arrow_schema.fields().iter().cloned().collect::<Vec<_>>());

            // Partitioned target: split the input into one file per partition and
            // commit them all in one snapshot. An empty input falls through to the
            // single-file path below (so a Replace still retires the prior gen).
            if let Some(spec) = &partition
                && !batches.is_empty()
            {
                let output_schema_ref: SchemaRef = Arc::new(schema_without_metadata.clone());
                let groups = crate::partition::split_batches_by_partition(
                    &output_schema_ref,
                    &batches,
                    spec,
                )
                .map_err(|e| DataFusionError::External(Box::new(e)))?;
                if !groups.is_empty() {
                    let key_names: Vec<String> = spec.keys.iter().map(|k| k.name.clone()).collect();
                    let result = table_writer
                        .write_partitioned(
                            &schema_name,
                            &table_name,
                            &schema_without_metadata,
                            write_mode,
                            spec.partition_id,
                            &key_names,
                            groups,
                        )
                        .await
                        .map_err(|e| DataFusionError::External(Box::new(e)))?;
                    let count_array: ArrayRef =
                        Arc::new(UInt64Array::from(vec![result.records_written as u64]));
                    return Ok(RecordBatch::try_new(output_schema, vec![count_array])?);
                }
            }

            // Write the (already sorted, if the table has a sort order) rows through
            // the size-rolling writer: it produces one file per target_file_size and
            // commits them in one snapshot. Only for non-empty input — an empty
            // Replace still needs the single-file truncate marker below, and empty
            // Append already returned above.
            //
            // `_unpartitioned_as_planned`: reaching here means the plan resolved the
            // table as unpartitioned. If a SET PARTITIONED BY went live since, the
            // commit fence must reject so the caller re-plans against the new spec —
            // this write must NOT quietly re-lay-out its rows under a spec the plan
            // never saw.
            let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
            if total_rows > 0 {
                let result = table_writer
                    .write_rows_unpartitioned_as_planned(
                        &schema_name,
                        &table_name,
                        &schema_without_metadata,
                        write_mode,
                        &batches,
                    )
                    .await
                    .map_err(|e| DataFusionError::External(Box::new(e)))?;
                let count_array: ArrayRef =
                    Arc::new(UInt64Array::from(vec![result.records_written as u64]));
                return Ok(RecordBatch::try_new(output_schema, vec![count_array])?);
            }

            let mut session = table_writer
                .begin_write(
                    &schema_name,
                    &table_name,
                    &schema_without_metadata,
                    write_mode,
                )
                .map_err(|e| DataFusionError::External(Box::new(e)))?;

            for batch in &batches {
                session
                    .write_batch(batch)
                    .map_err(|e| DataFusionError::External(Box::new(e)))?;
            }

            let row_count = session.row_count() as u64;

            session
                .finish()
                .await
                .map_err(|e| DataFusionError::External(Box::new(e)))?;

            let count_array: ArrayRef = Arc::new(UInt64Array::from(vec![row_count]));
            Ok(RecordBatch::try_new(output_schema, vec![count_array])?)
        });

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            make_insert_count_schema(),
            stream.map_err(|e: DataFusionError| e),
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_count_schema() {
        let schema = make_insert_count_schema();
        assert_eq!(schema.fields().len(), 1);
        assert_eq!(schema.field(0).name(), "count");
        assert_eq!(schema.field(0).data_type(), &DataType::UInt64);
    }
}
