//! Custom execution plan for renaming columns
//!
//! This module implements a DataFusion execution plan that wraps a scan
//! and renames columns from their original Parquet names to current DuckLake names.
//! This is needed when columns have been renamed in DuckLake metadata but the
//! Parquet files still have the original column names.

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use datafusion::common::config::ConfigOptions;
use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::execution::{RecordBatchStream, SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_plan::execution_plan::Boundedness;
use datafusion::physical_plan::filter_pushdown::{
    ChildFilterDescription, FilterDescription, FilterPushdownPhase,
};
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, PlanProperties,
};
use futures::Stream;

/// Custom execution plan that renames columns from Parquet file names to current DuckLake names
#[derive(Debug)]
pub struct ColumnRenameExec {
    /// The input execution plan (typically ParquetExec)
    input: Arc<dyn ExecutionPlan>,
    /// Output schema with renamed columns
    output_schema: SchemaRef,
    /// Mapping from old (Parquet) column names to new (DuckLake) column names
    name_mapping: HashMap<String, String>,
    /// Reverse mapping: new name -> old name, for looking up input columns
    reverse_mapping: Arc<HashMap<String, String>>,
    /// Cached plan properties with updated schema
    properties: Arc<PlanProperties>,
}

impl ColumnRenameExec {
    pub fn new(
        input: Arc<dyn ExecutionPlan>,
        output_schema: SchemaRef,
        name_mapping: HashMap<String, String>,
    ) -> Self {
        // PlanProperties must use output schema for DataFusion schema validation
        let eq_props = EquivalenceProperties::new(Arc::clone(&output_schema));
        let properties = Arc::new(PlanProperties::new(
            eq_props,
            input.output_partitioning().clone(),
            input.pipeline_behavior(),
            Boundedness::Bounded,
        ));

        // Pre-compute reverse mapping once (new_name -> old_name)
        let reverse_mapping: HashMap<String, String> = name_mapping
            .iter()
            .map(|(old, new)| (new.clone(), old.clone()))
            .collect();

        Self {
            input,
            output_schema,
            name_mapping,
            reverse_mapping: Arc::new(reverse_mapping),
            properties,
        }
    }

    /// Returns whether this node only renames columns without casting,
    /// projecting, or synthesizing values.
    pub fn is_pure_type_preserving_rename(&self) -> bool {
        let input_schema = self.input.schema();
        input_schema.fields().len() == self.output_schema.fields().len()
            && input_schema
                .fields()
                .iter()
                .zip(self.output_schema.fields())
                .all(|(input, output)| {
                    input.data_type() == output.data_type()
                        && self
                            .reverse_mapping
                            .get(output.name())
                            .map(String::as_str)
                            .unwrap_or(output.name())
                            == input.name()
                })
    }

    /// Remap a predicate over the catalog schema to the physical child schema.
    ///
    /// This is intentionally available only for a pure, type-preserving rename:
    /// pushing predicates through casts can change errors and comparison
    /// semantics, while projections and synthetic columns need richer mapping.
    pub fn remap_filter_to_input(
        &self,
        filter: Arc<dyn PhysicalExpr>,
    ) -> DataFusionResult<Option<Arc<dyn PhysicalExpr>>> {
        if !self.is_pure_type_preserving_rename() {
            return Ok(None);
        }

        let input_schema = self.input.schema();
        let output_schema = Arc::clone(&self.output_schema);
        let mut valid = true;
        let transformed = filter.transform_down(|expr| {
            let Some(column) = expr.downcast_ref::<Column>() else {
                return Ok(Transformed::no(expr));
            };
            let index = column.index();
            if output_schema
                .fields()
                .get(index)
                .is_none_or(|field| field.name() != column.name())
            {
                valid = false;
                return Ok(Transformed::complete(expr));
            }

            Ok(Transformed::yes(Arc::new(Column::new(
                input_schema.field(index).name(),
                index,
            ))))
        })?;

        Ok(valid.then_some(transformed.data))
    }
}

impl DisplayAs for ColumnRenameExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "ColumnRenameExec: renames={}", self.name_mapping.len())
    }
}

impl ExecutionPlan for ColumnRenameExec {
    fn name(&self) -> &str {
        "ColumnRenameExec"
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
                "ColumnRenameExec expects exactly one child".into(),
            ));
        }

        // Must call new() to rebuild properties from new child's partitioning
        Ok(Arc::new(ColumnRenameExec::new(
            Arc::clone(&children[0]),
            Arc::clone(&self.output_schema),
            self.name_mapping.clone(),
        )))
    }

    fn supports_limit_pushdown(&self) -> bool {
        self.is_pure_type_preserving_rename()
    }

    fn gather_filters_for_pushdown(
        &self,
        _phase: FilterPushdownPhase,
        parent_filters: Vec<Arc<dyn PhysicalExpr>>,
        _config: &ConfigOptions,
    ) -> DataFusionResult<FilterDescription> {
        if !self.is_pure_type_preserving_rename() {
            return Ok(FilterDescription::new()
                .with_child(ChildFilterDescription::all_unsupported(&parent_filters)));
        }

        let remapped = parent_filters
            .iter()
            .map(|filter| self.remap_filter_to_input(Arc::clone(filter)))
            .collect::<DataFusionResult<Option<Vec<_>>>>()?;
        let child = match remapped {
            Some(remapped) => ChildFilterDescription::from_child(&remapped, &self.input)?,
            None => ChildFilterDescription::all_unsupported(&parent_filters),
        };
        Ok(FilterDescription::new().with_child(child))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        let input_stream = self.input.execute(partition, context)?;

        Ok(Box::pin(ColumnRenameStream {
            input: input_stream,
            output_schema: Arc::clone(&self.output_schema),
            reverse_mapping: Arc::clone(&self.reverse_mapping),
        }))
    }
}

/// Stream that renames columns in output batches
struct ColumnRenameStream {
    input: SendableRecordBatchStream,
    output_schema: SchemaRef,
    /// Mapping from output column name -> input column name (for renamed columns only)
    reverse_mapping: Arc<HashMap<String, String>>,
}

impl Stream for ColumnRenameStream {
    type Item = DataFusionResult<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match Pin::new(&mut self.input).poll_next(cx) {
            Poll::Ready(Some(Ok(batch))) => {
                let result: DataFusionResult<RecordBatch> =
                    if self.output_schema.fields().is_empty() {
                        // Zero OUTPUT columns (e.g. COUNT(*)): preserve the row count
                        // with an empty schema. This must key off the output schema,
                        // not the input: on positional paths the input still carries
                        // the internal `__ducklake_row_pos` column (1 input column),
                        // yet the output is zero columns and the count must survive.
                        use arrow::record_batch::RecordBatchOptions;
                        let options =
                            RecordBatchOptions::new().with_row_count(Some(batch.num_rows()));
                        RecordBatch::try_new_with_options(
                            Arc::clone(&self.output_schema),
                            vec![],
                            &options,
                        )
                        .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))
                    } else {
                        // Build columns by looking up each output field in the input batch
                        let input_schema = batch.schema();
                        let columns: DataFusionResult<Vec<_>> = self
                            .output_schema
                            .fields()
                            .iter()
                            .map(|output_field| {
                                // Check if this column was renamed (new_name -> old_name)
                                let input_name = self
                                    .reverse_mapping
                                    .get(output_field.name())
                                    .map(|s| s.as_str())
                                    .unwrap_or_else(|| output_field.name().as_str());

                                let idx = input_schema
                                    .index_of(input_name)
                                    .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?;
                                // Coerce the column read from the file back to the
                                // catalog (output) type, keeping the provider
                                // self-consistent: it advertises and emits the catalog
                                // schema regardless of the file's physical Arrow type.
                                // Identical types clone cheaply.
                                coerce_column(batch.column(idx), output_field.data_type())
                            })
                            .collect();

                        columns.and_then(|cols| {
                            record_batch_with_schema(Arc::clone(&self.output_schema), cols)
                                .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))
                        })
                    };

                Poll::Ready(Some(result))
            },
            Poll::Ready(Some(Err(e))) => Poll::Ready(Some(Err(e))),
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

impl RecordBatchStream for ColumnRenameStream {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.output_schema)
    }
}

/// Coerce a column read from a parquet file to the catalog's declared type.
///
/// The physical Arrow type in a file can differ from the catalog type for list
/// columns: a DuckDB `ARRAY` may materialise as `FixedSizeList(N)` while the
/// catalog declares a variable `List`, and externally-written files often carry
/// an empty list child field name (`""`) where the catalog uses `"item"`.
///
/// `arrow::compute::cast` handles the structural value conversion
/// (`FixedSizeList` ↔ `List`, element-type changes) but leaves the list child
/// **field name** as-is, so a pure child-name difference round-trips unchanged
/// and would fail `RecordBatch::try_new`. After casting we therefore re-stamp the
/// array's `DataType` to the target when only nested field metadata differs —
/// the buffer layout is identical, so this is a zero-copy metadata swap.
pub(crate) fn coerce_column(
    col: &arrow::array::ArrayRef,
    target: &arrow::datatypes::DataType,
) -> DataFusionResult<arrow::array::ArrayRef> {
    use arrow::array::Array;

    let is_nested = matches!(
        target,
        arrow::datatypes::DataType::List(_)
            | arrow::datatypes::DataType::LargeList(_)
            | arrow::datatypes::DataType::FixedSizeList(_, _)
            | arrow::datatypes::DataType::Map(_, _)
            | arrow::datatypes::DataType::Struct(_)
    );
    if col.data_type() == target && !is_nested {
        return Ok(Arc::clone(col));
    }

    let casted = if col.data_type() == target {
        Arc::clone(col)
    } else {
        arrow::compute::cast(col, target)
            .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?
    };
    if casted.data_type() == target && !is_nested {
        return Ok(casted);
    }

    array_with_data_type(&casted, target)
        .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))
}

pub(crate) fn array_with_data_type(
    array: &arrow::array::ArrayRef,
    data_type: &arrow::datatypes::DataType,
) -> Result<arrow::array::ArrayRef, arrow::error::ArrowError> {
    use arrow::array::{Array, ArrayData, make_array};
    use arrow::datatypes::DataType;
    use arrow::error::ArrowError;

    fn rewrite(data: &ArrayData, data_type: &DataType) -> Result<ArrayData, ArrowError> {
        let is_nested = matches!(
            data_type,
            DataType::List(_)
                | DataType::LargeList(_)
                | DataType::FixedSizeList(_, _)
                | DataType::Map(_, _)
                | DataType::Struct(_)
        );
        if data.data_type() == data_type && !is_nested {
            return Ok(data.clone());
        }
        let child_types = match data_type {
            DataType::List(field)
            | DataType::LargeList(field)
            | DataType::FixedSizeList(field, _)
            | DataType::Map(field, _) => vec![field.data_type()],
            DataType::Struct(fields) => fields.iter().map(|field| field.data_type()).collect(),
            _ => {
                return Err(ArrowError::InvalidArgumentError(format!(
                    "cannot apply array type {:?} as {data_type:?}",
                    data.data_type(),
                )));
            },
        };
        if child_types.len() != data.child_data().len() {
            return Err(ArrowError::InvalidArgumentError(format!(
                "array type {:?} has {} children, expected {} for {data_type:?}",
                data.data_type(),
                data.child_data().len(),
                child_types.len(),
            )));
        }
        let children = data
            .child_data()
            .iter()
            .zip(child_types)
            .map(|(child, child_type)| rewrite(child, child_type))
            .collect::<Result<Vec<_>, _>>()?;
        data.clone()
            .into_builder()
            .data_type(data_type.clone())
            .child_data(children)
            .build()
    }

    Ok(make_array(rewrite(&array.to_data(), data_type)?))
}

pub(crate) fn record_batch_with_schema(
    schema: arrow::datatypes::SchemaRef,
    columns: Vec<arrow::array::ArrayRef>,
) -> Result<arrow::record_batch::RecordBatch, arrow::error::ArrowError> {
    use arrow::datatypes::{Field, Schema};
    use arrow::record_batch::RecordBatch;

    let fields = schema
        .fields()
        .iter()
        .zip(&columns)
        .map(|(field, column)| {
            Field::new(
                field.name(),
                column.data_type().clone(),
                field.is_nullable(),
            )
            .with_metadata(field.metadata().clone())
        })
        .collect::<Vec<_>>();
    let actual_schema = Arc::new(Schema::new_with_metadata(fields, schema.metadata().clone()));
    RecordBatch::try_new(actual_schema, columns)?.with_schema(schema)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion::physical_plan::EmptyRecordBatchStream;
    use datafusion::physical_plan::empty::EmptyExec;
    use datafusion::physical_plan::filter_pushdown::PushedDown;

    #[test]
    fn test_column_rename_stream_schema() {
        let input_schema = Arc::new(Schema::new(vec![Field::new(
            "old_col",
            DataType::Int32,
            false,
        )]));

        let output_schema = Arc::new(Schema::new(vec![Field::new(
            "new_col",
            DataType::Int32,
            false,
        )]));

        let mut reverse_mapping = HashMap::new();
        reverse_mapping.insert("new_col".to_string(), "old_col".to_string());

        let stream = ColumnRenameStream {
            input: Box::pin(EmptyRecordBatchStream::new(input_schema)),
            output_schema: Arc::clone(&output_schema),
            reverse_mapping: Arc::new(reverse_mapping),
        };

        // The stream should report the output schema
        assert_eq!(stream.schema().field(0).name(), "new_col");
    }

    #[test]
    fn coerce_column_restamps_nested_field_metadata() {
        use arrow::{
            array::{ArrayRef, Int32Array, RecordBatch, StructArray},
            datatypes::Fields,
        };

        let source_fields =
            Fields::from(vec![Arc::new(Field::new("value", DataType::Int32, false))]);
        let source: ArrayRef = Arc::new(StructArray::new(
            source_fields,
            vec![Arc::new(Int32Array::from(vec![1, 2]))],
            None,
        ));
        let target_field = Arc::new(Field::new("value", DataType::Int32, false).with_metadata(
            HashMap::from([("PARQUET:field_id".to_string(), "2".to_string())]),
        ));
        let target = DataType::Struct(Fields::from(vec![target_field]));

        let coerced = coerce_column(&source, &target).unwrap();
        let schema = Arc::new(Schema::new(vec![Field::new(
            "nested",
            target.clone(),
            false,
        )]));
        let batch = RecordBatch::try_new(schema, vec![coerced]).unwrap();

        let DataType::Struct(fields) = batch.column(0).data_type() else {
            panic!("expected struct");
        };
        assert_eq!(
            fields[0].metadata().get("PARQUET:field_id"),
            Some(&"2".to_string())
        );
    }

    #[test]
    fn pure_rename_remaps_filters_and_allows_limit_pushdown() {
        let input_schema = Arc::new(Schema::new(vec![Field::new(
            "old_col",
            DataType::Int32,
            false,
        )]));
        let output_schema = Arc::new(Schema::new(vec![Field::new(
            "new_col",
            DataType::Int32,
            false,
        )]));
        let input: Arc<dyn ExecutionPlan> = Arc::new(EmptyExec::new(input_schema));
        let exec = ColumnRenameExec::new(
            input,
            output_schema,
            HashMap::from([("old_col".to_string(), "new_col".to_string())]),
        );
        let filter: Arc<dyn PhysicalExpr> = Arc::new(Column::new("new_col", 0));

        assert!(exec.is_pure_type_preserving_rename());
        assert!(exec.supports_limit_pushdown());
        let remapped = exec
            .remap_filter_to_input(Arc::clone(&filter))
            .unwrap()
            .unwrap();
        let column = remapped.downcast_ref::<Column>().unwrap();
        assert_eq!(column.name(), "old_col");
        assert_eq!(column.index(), 0);

        let description = exec
            .gather_filters_for_pushdown(
                FilterPushdownPhase::Pre,
                vec![filter],
                &ConfigOptions::new(),
            )
            .unwrap();
        let pushed = description.parent_filters();
        assert!(matches!(pushed[0][0].discriminant, PushedDown::Yes));
        let column = pushed[0][0].predicate.downcast_ref::<Column>().unwrap();
        assert_eq!(column.name(), "old_col");
        assert_eq!(column.index(), 0);
    }

    #[test]
    fn casts_do_not_allow_filter_or_limit_pushdown() {
        let input_schema = Arc::new(Schema::new(vec![Field::new(
            "old_col",
            DataType::Int32,
            false,
        )]));
        let output_schema = Arc::new(Schema::new(vec![Field::new(
            "new_col",
            DataType::Int64,
            false,
        )]));
        let input: Arc<dyn ExecutionPlan> = Arc::new(EmptyExec::new(input_schema));
        let exec = ColumnRenameExec::new(
            input,
            output_schema,
            HashMap::from([("old_col".to_string(), "new_col".to_string())]),
        );
        let filter: Arc<dyn PhysicalExpr> = Arc::new(Column::new("new_col", 0));

        assert!(!exec.is_pure_type_preserving_rename());
        assert!(!exec.supports_limit_pushdown());
        assert!(
            exec.remap_filter_to_input(Arc::clone(&filter))
                .unwrap()
                .is_none()
        );
        let description = exec
            .gather_filters_for_pushdown(
                FilterPushdownPhase::Pre,
                vec![filter],
                &ConfigOptions::new(),
            )
            .unwrap();
        assert!(matches!(
            description.parent_filters()[0][0].discriminant,
            PushedDown::No
        ));
    }
}
