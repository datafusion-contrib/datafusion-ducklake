//! DuckLake INSERT execution plan.
//!
//! Limitations:
//! - Collects all batches into memory before writing (no streaming yet)
//! - Single partition only (partition 0)

use std::fmt::{self, Debug};
use std::sync::Arc;

use arrow::array::{ArrayRef, RecordBatch, UInt32Array, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::common::ScalarValue;
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::execution::object_store::ObjectStoreUrl;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::{EquivalenceProperties, Partitioning};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use futures::stream::{self, TryStreamExt};

use crate::metadata_writer::{MetadataWriter, WriteMode};
use crate::partition::PartitionTransform;
use crate::table_writer::{DuckLakeTableWriter, PartitionGroup};

/// Resolved partition spec for the write path: how `DuckLakeInsertExec` splits
/// incoming rows into per-partition files. Built by `DuckLakeTable::insert_into`
/// from the table's active [`crate::partition::PartitionSpec`].
#[derive(Debug, Clone)]
pub struct PartitionWriteSpec {
    /// The active spec generation (`ducklake_partition_info.partition_id`).
    pub partition_id: i64,
    /// Partition keys, in key order.
    pub keys: Vec<PartitionWriteKey>,
}

/// One partition key resolved for the write path.
#[derive(Debug, Clone)]
pub struct PartitionWriteKey {
    /// Column index in the INSERT input schema.
    pub input_index: usize,
    /// Column name (used only for the readable Hive-style path).
    pub name: String,
    /// Transform applied to the column value to form the partition value.
    pub transform: PartitionTransform,
}

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
            self.partition.clone(),
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

            // Partitioned target: split the input into one file per partition and
            // commit them all in one snapshot. An empty input falls through to the
            // single-file path below (so a Replace still retires the prior gen).
            if let Some(spec) = &partition
                && !batches.is_empty()
            {
                let output_schema_ref: SchemaRef = Arc::new(schema_without_metadata.clone());
                let groups = split_batches_by_partition(&output_schema_ref, &batches, spec)?;
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

/// Apply a partition transform to a whole column array: identity returns the
/// column unchanged; the temporal transforms return an `Int32` calendar component
/// (year/month/day/hour) via Arrow's `date_part`. Only producible transforms are
/// valid here — `DuckLakeTable::insert_into` rejects `bucket`/unknown up front.
fn transform_array(transform: &PartitionTransform, array: &ArrayRef) -> DataFusionResult<ArrayRef> {
    use arrow::compute::{DatePart, date_part};
    let part = match transform {
        PartitionTransform::Identity => return Ok(Arc::clone(array)),
        PartitionTransform::Year => DatePart::Year,
        PartitionTransform::Month => DatePart::Month,
        PartitionTransform::Day => DatePart::Day,
        PartitionTransform::Hour => DatePart::Hour,
        other => {
            return Err(DataFusionError::NotImplemented(format!(
                "partitioned write with transform '{}' is not supported",
                other.to_catalog_string()
            )));
        },
    };
    Ok(date_part(array, part)?)
}

/// Split the input into groups keyed by the tuple of transformed, DuckDB-canonical
/// partition values — one group (one output file) per distinct key. Returns
/// `(values, batches)` per group, where `values[i]` is the encoded value for
/// partition key `i` (`None` for SQL NULL). Rows sharing a key land in the same
/// group regardless of which input batch they came from.
fn split_batches_by_partition(
    output_schema: &SchemaRef,
    batches: &[RecordBatch],
    spec: &PartitionWriteSpec,
) -> DataFusionResult<Vec<PartitionGroup>> {
    use arrow::compute::{concat_batches, take};
    use std::collections::HashMap;

    if batches.is_empty() {
        return Ok(Vec::new());
    }
    let input_schema = batches[0].schema();
    let combined = concat_batches(&input_schema, batches)?;
    let num_rows = combined.num_rows();
    if num_rows == 0 {
        return Ok(Vec::new());
    }

    // Transform each partition-key column once for the whole dataset.
    let mut transformed: Vec<ArrayRef> = Vec::with_capacity(spec.keys.len());
    for key in &spec.keys {
        transformed.push(transform_array(
            &key.transform,
            combined.column(key.input_index),
        )?);
    }

    // Group row indices by the encoded partition-value tuple.
    let mut groups: HashMap<Vec<Option<String>>, Vec<u32>> = HashMap::new();
    for row in 0..num_rows {
        let mut values: Vec<Option<String>> = Vec::with_capacity(spec.keys.len());
        for array in &transformed {
            let scalar = ScalarValue::try_from_array(array, row)?;
            // `encode_scalar` returns `None` for BOTH a genuine SQL NULL and a
            // non-null value of a type it cannot encode. Those must not be
            // conflated: silently mapping an unencodable non-null value to `None`
            // would group every distinct such value into one file with a NULL
            // partition value (data corruption). A NULL is a legitimate partition
            // value; an unencodable non-null value is a hard error.
            let encoded = if scalar.is_null() {
                None
            } else {
                match crate::stats_encode::encode_scalar(&scalar) {
                    Some(encoded) => Some(encoded),
                    None => {
                        return Err(DataFusionError::NotImplemented(format!(
                            "partitioned write: partition-key value of type {} cannot be \
                             encoded; partitioning by this column type is not supported",
                            array.data_type()
                        )));
                    },
                }
            };
            values.push(encoded);
        }
        groups.entry(values).or_default().push(row as u32);
    }

    // Materialize one batch per group via `take` (output uses the clean schema).
    let mut result = Vec::with_capacity(groups.len());
    for (values, indices) in groups {
        let index_array = UInt32Array::from(indices);
        let columns = combined
            .columns()
            .iter()
            .map(|c| take(c, &index_array, None))
            .collect::<Result<Vec<_>, _>>()?;
        let batch = RecordBatch::try_new(output_schema.clone(), columns)?;
        result.push((values, vec![batch]));
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::StringArray;

    #[test]
    fn test_insert_count_schema() {
        let schema = make_insert_count_schema();
        assert_eq!(schema.fields().len(), 1);
        assert_eq!(schema.field(0).name(), "count");
        assert_eq!(schema.field(0).data_type(), &DataType::UInt64);
    }

    fn identity_region_spec() -> PartitionWriteSpec {
        PartitionWriteSpec {
            partition_id: 1,
            keys: vec![PartitionWriteKey {
                input_index: 0,
                name: "region".to_string(),
                transform: PartitionTransform::Identity,
            }],
        }
    }

    #[test]
    fn split_groups_by_identity_and_keeps_null_partition() {
        let schema: SchemaRef = Arc::new(Schema::new(vec![Field::new(
            "region",
            DataType::Utf8,
            true,
        )]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(StringArray::from(vec![Some("us"), None, Some("us")])) as ArrayRef],
        )
        .unwrap();
        let groups = split_batches_by_partition(
            &schema,
            std::slice::from_ref(&batch),
            &identity_region_spec(),
        )
        .unwrap();
        // "us" (2 rows) and a legitimate NULL partition (1 row).
        assert_eq!(groups.len(), 2);
        let total: usize = groups
            .iter()
            .flat_map(|(_, b)| b)
            .map(|b| b.num_rows())
            .sum();
        assert_eq!(total, 3);
        let mut values: Vec<Option<String>> = groups.iter().map(|(v, _)| v[0].clone()).collect();
        values.sort();
        assert_eq!(values, vec![None, Some("us".to_string())]);
    }

    #[test]
    fn split_errors_on_unencodable_non_null_value_instead_of_corrupting() {
        let schema: SchemaRef = Arc::new(Schema::new(vec![Field::new(
            "region",
            DataType::Utf8,
            true,
        )]));
        // A NUL byte makes the value unencodable (encode_scalar returns None) but it
        // is NOT null — it must error, not silently collapse into a NULL partition
        // and commingle with genuinely-null rows.
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(StringArray::from(vec![Some("a\u{0}b")])) as ArrayRef],
        )
        .unwrap();
        let err = split_batches_by_partition(
            &schema,
            std::slice::from_ref(&batch),
            &identity_region_spec(),
        )
        .unwrap_err();
        assert!(
            err.to_string().to_lowercase().contains("encode"),
            "expected an encode error, got: {err}"
        );
    }
}
