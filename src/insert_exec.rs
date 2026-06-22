//! DuckLake INSERT execution plan.
//!
//! Limitations:
//! - Collects all batches into memory before writing (no streaming yet)
//! - Single partition only (partition 0)

use std::any::Any;
use std::fmt::{self, Debug};
use std::sync::Arc;

use arrow::array::{ArrayRef, RecordBatch, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::execution::object_store::ObjectStoreUrl;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::{EquivalenceProperties, Partitioning};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use futures::stream::{self, TryStreamExt};

use crate::metadata_writer::{MetadataWriter, WriteMode};
use crate::table_writer::DuckLakeTableWriter;

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
    cache: Arc<PlanProperties>,
}

impl DuckLakeInsertExec {
    /// Create a new DuckLakeInsertExec
    pub fn new(
        input: Arc<dyn ExecutionPlan>,
        writer: Arc<dyn MetadataWriter>,
        schema_name: String,
        table_name: String,
        arrow_schema: SchemaRef,
        write_mode: WriteMode,
        object_store_url: Arc<ObjectStoreUrl>,
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
    fn name(&self) -> &str {
        "DuckLakeInsertExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
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
                .map_err(|e| DataFusionError::External(Box::new(e)))?;

            let schema_without_metadata =
                Schema::new(arrow_schema.fields().iter().cloned().collect::<Vec<_>>());

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
