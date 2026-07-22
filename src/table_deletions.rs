//! Table deletions functionality for DuckLake
//!
//! This module provides the `ducklake_table_deletions()` table function that returns
//! the actual deleted rows between snapshots, with CDC metadata columns.
//!
//! For each data file with deletions:
//! 1. Read positions from current delete file (or all positions for full file delete)
//! 2. Subtract positions from previous delete file (if exists)
//! 3. Read the data file and return only the rows at the newly deleted positions
//! 4. Append CDC columns (snapshot_id, change_type='delete')
//!
//! Like [`TableChangesExec`](crate::table_changes), [`DeletedRowsExec`] is a
//! single-partition plan with no DataFusion children: its per-file scans are
//! internal and executed directly — the delete files are fully collected (the
//! position set must be complete before any data row can be classified), then
//! the data file is streamed batch-by-batch through the filter. Deleted rows
//! are matched by TRUE physical file position (`PositionalFileSource` +
//! [`FileRowNumberExec`]) rather than stream arrival order. Exposing the scans
//! as children lets the optimizer repartition them (round-robin or byte-range
//! splits), which desynchronizes the delete-position set from the data rows —
//! deletions were silently missed or mis-attributed (issue #178).

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Int64Array, StringArray, UInt32Array};
use arrow::compute::take;
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use async_trait::async_trait;
use datafusion::catalog::Session;
use datafusion::common::Result as DataFusionResult;
use datafusion::datasource::listing::PartitionedFile;
use datafusion::datasource::physical_plan::{FileGroup, FileScanConfigBuilder, ParquetSource};
use datafusion::datasource::source::DataSourceExec;
use datafusion::datasource::{TableProvider, TableType};
use datafusion::error::DataFusionError;
use datafusion::execution::object_store::ObjectStoreUrl;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_expr::{EquivalenceProperties, PhysicalExpr};
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::union::UnionExec;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, collect,
};
use futures::stream::BoxStream;
use futures::{StreamExt, TryStreamExt};
use object_store::path::Path as ObjectPath;
use parquet::arrow::ParquetRecordBatchStreamBuilder;
use parquet::arrow::async_reader::ParquetObjectReader;

use crate::metadata_provider::{DeleteFileChange, MetadataProvider};
use crate::path_resolver::resolve_path;
use crate::positional_source::PositionalFileSource;
use crate::row_id::{
    FileRowNumberExec, ROW_ID_PARQUET_FIELD_ID, ROW_POS_COLUMN_NAME, SNAPSHOT_ID_PARQUET_FIELD_ID,
};
use crate::table::{validated_file_size, validated_record_count};
use crate::types::extract_parquet_field_ids;

/// Delete file schema: (file_path: VARCHAR, pos: INT64)
fn delete_file_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("file_path", DataType::Utf8, false),
        Field::new("pos", DataType::Int64, false),
    ]))
}

/// TableProvider that exposes deleted rows between snapshots
///
/// For each data file with deletions:
/// 1. Read positions from current delete file (or generate all positions for full file delete)
/// 2. Subtract positions from previous delete file (if exists)
/// 3. Read the data file and filter to only deleted row positions
/// 4. Append snapshot_id and change_type columns
#[derive(Debug)]
pub struct TableDeletionsTable {
    provider: Arc<dyn MetadataProvider>,
    table_id: i64,
    start_snapshot: i64,
    end_snapshot: i64,
    object_store_url: Arc<ObjectStoreUrl>,
    table_path: String,
    /// Original table schema (without CDC columns)
    table_schema: SchemaRef,
    /// Combined schema: snapshot_id + rowid + change_type + table columns
    output_schema: SchemaRef,
}

impl TableDeletionsTable {
    pub fn new(
        provider: Arc<dyn MetadataProvider>,
        table_id: i64,
        start_snapshot: i64,
        end_snapshot: i64,
        object_store_url: Arc<ObjectStoreUrl>,
        table_path: String,
        table_schema: SchemaRef,
    ) -> Self {
        // Build output schema: CDC metadata columns leading — (snapshot_id,
        // rowid, change_type), matching ducklake_table_changes and official
        // DuckLake's column order — then the table columns.
        let mut fields: Vec<Field> = Vec::with_capacity(table_schema.fields().len() + 3);
        fields.push(Field::new("snapshot_id", DataType::Int64, false));
        // rowid is nullable for symmetry with ducklake_table_changes (where it is
        // NULL on encrypted tables); the deletions path always synthesizes a
        // non-null value for the cases it supports.
        fields.push(Field::new("rowid", DataType::Int64, true));
        fields.push(Field::new("change_type", DataType::Utf8, false));
        fields.extend(table_schema.fields().iter().map(|f| f.as_ref().clone()));
        let output_schema = Arc::new(Schema::new(fields));

        Self {
            provider,
            table_id,
            start_snapshot,
            end_snapshot,
            object_store_url,
            table_path,
            table_schema,
            output_schema,
        }
    }

    /// Build execution plan for a single delete file entry. `need_rowid` gates
    /// resolving the rowid (footer probe + embedded read + synthesis); when the
    /// caller projected rowid away, it is emitted as a placeholder and dropped.
    async fn build_exec_for_delete_entry(
        &self,
        state: &dyn Session,
        need_rowid: bool,
        delete_file: &DeleteFileChange,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        // Resolve data file path
        let data_file_path = resolve_path(
            &self.table_path,
            &delete_file.data_file_path,
            delete_file.data_file_path_is_relative,
        )
        .map_err(|e| DataFusionError::External(Box::new(e)))?;

        // A cumulative (current-spec) delete file embeds each row's delete
        // snapshot; deletions are then windowed PER ROW on that column, and no
        // previous-file subtraction is needed (pre-window deletions are simply
        // outside the window). Legacy 2-column delete files keep the
        // delta-vs-previous model, one snapshot per file.
        let snapshot_name = match &delete_file.current_delete_path {
            Some(p) => {
                self.detect_delete_file_snapshot_name(
                    state,
                    p,
                    delete_file.current_delete_path_is_relative.unwrap_or(true),
                )
                .await?
            },
            None => None,
        };
        if snapshot_name.is_none() && delete_file.snapshot_id < self.start_snapshot {
            // Only cumulative files may begin before the window (included via
            // ducklake_delete_file.partial_max); a legacy file here means the
            // catalog is inconsistent, and its rows cannot be windowed.
            return Err(DataFusionError::External(
                format!(
                    "delete file {:?} begins before the query window but carries no embedded \
                     per-row snapshot column; its deletions cannot be attributed",
                    delete_file.current_delete_path
                )
                .into(),
            ));
        }

        // Create scan for current delete file (if exists - None means full file delete)
        let current_delete_exec = if let Some(ref current_path) = delete_file.current_delete_path {
            Some(self.build_delete_file_scan(
                current_path,
                delete_file.current_delete_path_is_relative.unwrap_or(true),
                delete_file.current_delete_file_size_bytes.unwrap_or(0),
                delete_file.current_delete_footer_size.unwrap_or(0),
                &snapshot_name,
            )?)
        } else {
            None
        };

        // Create scan for previous delete file (if exists; not needed in
        // cumulative mode, where the per-row window filter replaces it)
        let previous_delete_exec = match &delete_file.previous_delete_path {
            Some(prev_path) if snapshot_name.is_none() => Some(self.build_delete_file_scan(
                prev_path,
                delete_file.previous_delete_path_is_relative.unwrap_or(true),
                delete_file.previous_delete_file_size_bytes.unwrap_or(0),
                delete_file.previous_delete_footer_size.unwrap_or(0),
                &None,
            )?),
            _ => None,
        };

        // Detect the source file's embedded rowid column (present on UPDATE /
        // compaction outputs). When present, a deleted row's rowid is that
        // embedded value — NOT `row_id_start + position`, which would key the
        // delete differently from the row's insert/update_postimage. Only probe
        // when the rowid is actually requested.
        let table_len = self.table_schema.fields().len();
        let embedded_name = if need_rowid {
            self.detect_embedded_rowid_name(
                state,
                &delete_file.data_file_path,
                delete_file.data_file_path_is_relative,
            )
            .await?
        } else {
            None
        };
        let embedded_col_idx = embedded_name.as_ref().map(|_| table_len);
        // The positional scan appends the physical-position column after the
        // table columns and the optional embedded rowid.
        let pos_col_idx = table_len + usize::from(embedded_name.is_some());

        // Create scan for data file (with the embedded rowid column when present)
        let data_file_exec = self.build_data_file_scan(
            &data_file_path,
            delete_file.data_file_size_bytes,
            delete_file.data_file_footer_size.unwrap_or(0),
            &embedded_name,
        )?;

        // Validate record_count before use — a negative value from corrupt metadata
        // would cause incorrect behavior (e.g., empty ranges in full-file deletes).
        validated_record_count(delete_file.data_record_count, &delete_file.data_file_path)?;

        Ok(Arc::new(DeletedRowsExec::new(DeletionUnit {
            current_delete_scan: current_delete_exec,
            previous_delete_scan: previous_delete_exec,
            data_file_scan: data_file_exec,
            record_count: delete_file.data_record_count,
            snapshot_id: delete_file.snapshot_id,
            row_id_start: delete_file.data_row_id_start,
            table_len,
            embedded_col_idx,
            pos_col_idx,
            need_rowid,
            cumulative: snapshot_name.is_some(),
            window: (self.start_snapshot, self.end_snapshot),
            output_schema: self.output_schema.clone(),
        })))
    }

    /// Read the source file's footer and return the physical name of its embedded
    /// row-id column ([`ROW_ID_PARQUET_FIELD_ID`]) when present. Such a file is an
    /// UPDATE / compaction output whose logical rowids come from that column.
    async fn detect_embedded_rowid_name(
        &self,
        state: &dyn Session,
        path: &str,
        is_relative: bool,
    ) -> DataFusionResult<Option<String>> {
        self.parquet_field_id_name(state, path, is_relative, ROW_ID_PARQUET_FIELD_ID)
            .await
    }

    /// Read a DELETE file's footer and return the physical name of its embedded
    /// per-row snapshot column ([`SNAPSHOT_ID_PARQUET_FIELD_ID`]) when present.
    /// Current-spec delete files are cumulative and carry one; each row's value
    /// is the snapshot at which that position was deleted.
    async fn detect_delete_file_snapshot_name(
        &self,
        state: &dyn Session,
        path: &str,
        is_relative: bool,
    ) -> DataFusionResult<Option<String>> {
        self.parquet_field_id_name(state, path, is_relative, SNAPSHOT_ID_PARQUET_FIELD_ID)
            .await
    }

    async fn parquet_field_id_name(
        &self,
        state: &dyn Session,
        path: &str,
        is_relative: bool,
        field_id: i32,
    ) -> DataFusionResult<Option<String>> {
        let resolved = resolve_path(&self.table_path, path, is_relative)
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        let object_store = state
            .runtime_env()
            .object_store(self.object_store_url.as_ref())?;
        let reader = ParquetObjectReader::new(object_store, ObjectPath::from(resolved.as_str()));
        let builder = ParquetRecordBatchStreamBuilder::new(reader)
            .await
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        let field_ids = extract_parquet_field_ids(builder.metadata());
        Ok(field_ids.get(&field_id).cloned())
    }

    /// Build a ParquetExec for a delete file. When `snapshot_name` is `Some`,
    /// the file's embedded per-row snapshot column is read as a third column.
    fn build_delete_file_scan(
        &self,
        path: &str,
        is_relative: bool,
        size_bytes: i64,
        footer_size: i64,
        snapshot_name: &Option<String>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let resolved_path = resolve_path(&self.table_path, path, is_relative)
            .map_err(|e| DataFusionError::External(Box::new(e)))?;

        let mut pf = PartitionedFile::new(
            &resolved_path,
            validated_file_size(size_bytes, &resolved_path)?,
        );
        if footer_size > 0
            && let Ok(hint) = usize::try_from(footer_size)
        {
            pf = pf.with_metadata_size_hint(hint);
        }

        let schema = match snapshot_name {
            Some(name) => {
                let mut fields: Vec<Field> = delete_file_schema()
                    .fields()
                    .iter()
                    .map(|f| f.as_ref().clone())
                    .collect();
                fields.push(Field::new(name, DataType::Int64, true));
                Arc::new(Schema::new(fields))
            },
            None => delete_file_schema(),
        };
        let builder = FileScanConfigBuilder::new(
            self.object_store_url.as_ref().clone(),
            Arc::new(ParquetSource::new(schema)),
        )
        .with_file_group(FileGroup::new(vec![pf]));

        Ok(DataSourceExec::from_data_source(builder.build()))
    }

    /// Positional scan of the source data file: table columns, the embedded
    /// rowid column when `embedded_name` is `Some`, and the internal
    /// physical-position column ([`ROW_POS_COLUMN_NAME`]). [`PositionalFileSource`]
    /// and [`FileRowNumberExec`] guarantee true physical positions, so deleted
    /// rows are matched to the delete file's `pos` set regardless of how the
    /// file is read (issue #178).
    fn build_data_file_scan(
        &self,
        path: &str,
        size_bytes: i64,
        footer_size: i64,
        embedded_name: &Option<String>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let mut pf = PartitionedFile::new(path, validated_file_size(size_bytes, path)?);
        if footer_size > 0
            && let Ok(hint) = usize::try_from(footer_size)
        {
            pf = pf.with_metadata_size_hint(hint);
        }

        let read_schema = match embedded_name {
            Some(name) => {
                let mut fields: Vec<Field> = self
                    .table_schema
                    .fields()
                    .iter()
                    .map(|f| f.as_ref().clone())
                    .collect();
                fields.push(Field::new(name, DataType::Int64, true));
                Arc::new(Schema::new(fields))
            },
            None => self.table_schema.clone(),
        };

        let source = PositionalFileSource::wrap(Arc::new(ParquetSource::new(read_schema)));
        let builder = FileScanConfigBuilder::new(self.object_store_url.as_ref().clone(), source)
            .with_file_group(FileGroup::new(vec![pf]))
            .with_partitioned_by_file_group(true);
        let scan = DataSourceExec::from_data_source(builder.build());
        Ok(Arc::new(FileRowNumberExec::new(scan, vec![0])))
    }
}

#[async_trait]
impl TableProvider for TableDeletionsTable {
    fn schema(&self) -> SchemaRef {
        self.output_schema.clone()
    }

    fn table_type(&self) -> TableType {
        TableType::View
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        _filters: &[datafusion::prelude::Expr],
        _limit: Option<usize>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        // Get delete files added between snapshots
        let delete_files = self
            .provider
            .get_delete_files_added_between_snapshots(
                self.table_id,
                self.start_snapshot,
                self.end_snapshot,
            )
            .map_err(|e| DataFusionError::External(Box::new(e)))?;

        // Handle empty case
        if delete_files.is_empty() {
            use datafusion::physical_plan::empty::EmptyExec;
            let output_schema = match projection {
                Some(indices) => {
                    let fields: Vec<Field> = indices
                        .iter()
                        .map(|&i| self.output_schema.field(i).clone())
                        .collect();
                    Arc::new(Schema::new(fields))
                },
                None => self.output_schema.clone(),
            };
            return Ok(Arc::new(EmptyExec::new(output_schema)));
        }

        // Does the caller actually want `rowid`? It sits at index 1 among the
        // leading CDC columns. When it is projected away we skip resolving it (no
        // footer probe, no synthesis), so a query like `SELECT id, change_type
        // FROM ducklake_table_deletions(...)` never fails on a file whose rowid
        // cannot be synthesized (no embedded rowid and a NULL row_id_start).
        let need_rowid = projection.is_none_or(|indices| indices.contains(&1));

        // Build execution plan for each delete entry
        let mut execs: Vec<Arc<dyn ExecutionPlan>> = Vec::with_capacity(delete_files.len());
        for delete_file in &delete_files {
            let exec = self
                .build_exec_for_delete_entry(state, need_rowid, delete_file)
                .await?;
            execs.push(exec);
        }

        // Combine with UnionExec if multiple
        let full: Arc<dyn ExecutionPlan> = if execs.len() == 1 {
            execs.into_iter().next().unwrap()
        } else {
            UnionExec::try_new(execs)?
        };

        // The exec emits the full `[snapshot_id, rowid, change_type, table cols]`
        // schema; honor the requested projection with a ProjectionExec on top.
        match projection {
            None => Ok(full),
            Some(indices) => {
                let exprs: Vec<(Arc<dyn PhysicalExpr>, String)> = indices
                    .iter()
                    .map(|&i| {
                        let f = self.output_schema.field(i);
                        (
                            Arc::new(Column::new(f.name(), i)) as Arc<dyn PhysicalExpr>,
                            f.name().to_string(),
                        )
                    })
                    .collect();
                Ok(Arc::new(ProjectionExec::try_new(exprs, full)?))
            },
        }
    }
}

/// The internal scans and parameters needed to extract one delete entry's
/// deleted rows. Cloned into the async extraction on execute.
#[derive(Debug, Clone)]
struct DeletionUnit {
    /// Scan of current delete file (None for full file deletes)
    current_delete_scan: Option<Arc<dyn ExecutionPlan>>,
    /// Scan of previous delete file (if exists; legacy delta mode only)
    previous_delete_scan: Option<Arc<dyn ExecutionPlan>>,
    /// Positional scan of the data file (appends [`ROW_POS_COLUMN_NAME`])
    data_file_scan: Arc<dyn ExecutionPlan>,
    /// Total record count in data file (used for full file deletes)
    record_count: i64,
    /// Snapshot ID for CDC column
    snapshot_id: i64,
    /// First rowid of the data file (`None` if the catalog carries none). Used
    /// to synthesize a deleted row's rowid as `row_id_start + physical position`
    /// when the source file has no embedded rowid.
    row_id_start: Option<i64>,
    /// Number of leading table columns in the data-file batch.
    table_len: usize,
    /// Column index of the embedded rowid in the data-file batch, if present (an
    /// UPDATE / compaction output); its value IS the deleted row's rowid.
    embedded_col_idx: Option<usize>,
    /// Column index of the physical-position column in the data-file batch.
    pos_col_idx: usize,
    /// Whether the rowid column is actually requested. When false, rowid is
    /// emitted as a placeholder (dropped by the projection above) and neither an
    /// embedded rowid nor a row_id_start is required.
    need_rowid: bool,
    /// Whether the current delete file is cumulative (carries an embedded
    /// per-row delete-snapshot column as its third column). Rows are then
    /// windowed per row and emitted at their own delete snapshots.
    cumulative: bool,
    /// The query's inclusive `[start, end]` snapshot window (cumulative mode).
    window: (i64, i64),
    /// Output schema (snapshot_id + rowid + change_type + table columns)
    output_schema: SchemaRef,
}

/// Execution plan that reads deleted rows from a data file
///
/// 1. Reads current delete file to get deleted positions
/// 2. Reads previous delete file to get previously deleted positions (if exists)
/// 3. Computes delta: positions in current but not in previous
/// 4. Reads data file and filters to only include rows at deleted positions
/// 5. Appends CDC columns (snapshot_id, change_type='delete')
///
/// Single partition, no DataFusion children: the delete-position set is global
/// to the data file, so the optimizer must not repartition or split the
/// internal scans (issue #178). Rows are matched by the true physical position
/// appended by the positional data scan, never by arrival order.
#[derive(Debug)]
pub struct DeletedRowsExec {
    unit: DeletionUnit,
    /// Cached plan properties
    properties: Arc<PlanProperties>,
}

impl DeletedRowsExec {
    fn new(unit: DeletionUnit) -> Self {
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(unit.output_schema.clone()),
            datafusion::physical_expr::Partitioning::UnknownPartitioning(1),
            EmissionType::Final,
            Boundedness::Bounded,
        ));
        Self {
            unit,
            properties,
        }
    }
}

impl DisplayAs for DeletedRowsExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default
            | DisplayFormatType::Verbose
            | DisplayFormatType::TreeRender => {
                write!(
                    f,
                    "DeletedRowsExec: snapshot_id={}, full_delete={}, has_previous={}",
                    self.unit.snapshot_id,
                    self.unit.current_delete_scan.is_none(),
                    self.unit.previous_delete_scan.is_some()
                )
            },
        }
    }
}

impl ExecutionPlan for DeletedRowsExec {
    fn name(&self) -> &str {
        "DeletedRowsExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    /// No DataFusion children: the per-file scans are internal and executed
    /// directly, so the optimizer never rewrites them.
    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        if !children.is_empty() {
            return Err(DataFusionError::Internal(
                "DeletedRowsExec has no children".to_string(),
            ));
        }
        Ok(self)
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(DataFusionError::Internal(format!(
                "DeletedRowsExec only supports partition 0, got {partition}"
            )));
        }

        let unit = self.unit.clone();
        let schema = self.unit.output_schema.clone();
        let stream = futures::stream::once(deleted_rows_stream(unit, context)).try_flatten();

        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream)))
    }

    fn schema(&self) -> SchemaRef {
        self.unit.output_schema.clone()
    }
}

/// Collect the deleted position set (the delete files must be read fully
/// before any data row can be classified), then return the data file's
/// batches filtered to the deleted rows, matching by the true physical
/// position appended by the positional scan. The data file itself is
/// streamed batch-by-batch, never materialized whole.
async fn deleted_rows_stream(
    unit: DeletionUnit,
    context: Arc<TaskContext>,
) -> DataFusionResult<BoxStream<'static, DataFusionResult<RecordBatch>>> {
    // 1. Deleted positions, from the current delete file (windowed per row in
    //    cumulative mode) or every position for a full-file delete.
    let mut position_snapshots: HashMap<i64, i64> = HashMap::new();
    let current_positions: HashSet<i64> = match &unit.current_delete_scan {
        Some(scan) => {
            let batches = collect(Arc::clone(scan), context.clone()).await?;
            let mut positions = HashSet::new();
            for batch in &batches {
                if unit.cumulative {
                    extract_windowed_positions(
                        batch,
                        unit.window,
                        &mut positions,
                        &mut position_snapshots,
                    )?;
                } else {
                    positions.extend(extract_positions(batch)?);
                }
            }
            positions
        },
        None => (0..unit.record_count).collect(),
    };

    // 2. Subtract the previous delete file's positions (legacy delta mode; the
    //    per-row window filter replaces this in cumulative mode).
    let deleted_positions: HashSet<i64> = match &unit.previous_delete_scan {
        Some(scan) => {
            let batches = collect(Arc::clone(scan), context.clone()).await?;
            let mut previous = HashSet::new();
            for batch in &batches {
                previous.extend(extract_positions(batch)?);
            }
            current_positions
                .into_iter()
                .filter(|pos| !previous.contains(pos))
                .collect()
        },
        None => current_positions,
    };
    if deleted_positions.is_empty() {
        return Ok(futures::stream::empty().boxed());
    }

    // 3. Stream the data file and keep the rows whose PHYSICAL position is in
    //    the deleted set. The positional scan is a single partition covering
    //    the whole file, so no row can end up out of reach of the position set.
    let data_stream = unit.data_file_scan.execute(0, context)?;
    Ok(data_stream
        .try_filter_map(move |batch| {
            futures::future::ready(filter_batch(
                &unit,
                &batch,
                &deleted_positions,
                &position_snapshots,
            ))
        })
        .boxed())
}

/// Extract positions from a delete file batch (`(file_path, pos)` schema).
fn extract_positions(batch: &RecordBatch) -> DataFusionResult<Vec<i64>> {
    if batch.num_columns() < 2 {
        return Ok(Vec::new());
    }
    let pos_array = batch
        .column(1)
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| DataFusionError::Internal("delete `pos` column is not Int64".to_string()))?;
    Ok(pos_array.values().iter().copied().collect())
}

/// Extract in-window positions AND their per-row delete snapshots from a
/// cumulative delete file batch (`(file_path, pos, snapshot)` schema),
/// recording each kept position's snapshot in `position_snapshots`.
fn extract_windowed_positions(
    batch: &RecordBatch,
    window: (i64, i64),
    positions: &mut HashSet<i64>,
    position_snapshots: &mut HashMap<i64, i64>,
) -> DataFusionResult<()> {
    if batch.num_columns() < 3 {
        return Err(DataFusionError::Internal(
            "cumulative delete file batch is missing its snapshot column".to_string(),
        ));
    }
    let pos = batch
        .column(1)
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| DataFusionError::Internal("delete `pos` column is not Int64".to_string()))?;
    let snaps = batch
        .column(2)
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| {
            DataFusionError::Internal("delete snapshot column is not Int64".to_string())
        })?;
    for i in 0..batch.num_rows() {
        if snaps.is_null(i) {
            return Err(DataFusionError::Internal(
                "cumulative delete file has a NULL per-row snapshot".to_string(),
            ));
        }
        let s = snaps.value(i);
        if s >= window.0 && s <= window.1 {
            let p = pos.value(i);
            positions.insert(p);
            position_snapshots.insert(p, s);
        }
    }
    Ok(())
}

/// Filter a data-file batch to its deleted rows and append the CDC columns.
fn filter_batch(
    unit: &DeletionUnit,
    batch: &RecordBatch,
    deleted_positions: &HashSet<i64>,
    position_snapshots: &HashMap<i64, i64>,
) -> DataFusionResult<Option<RecordBatch>> {
    let num_rows = batch.num_rows();

    // The physical position of each row, appended by the positional scan.
    let pos = batch
        .column(unit.pos_col_idx)
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| {
            DataFusionError::Internal(format!(
                "physical-position column {ROW_POS_COLUMN_NAME} is missing or not Int64"
            ))
        })?;

    // Resolve each deleted row's rowid: the embedded rowid column when the
    // source file has one (an UPDATE / compaction output), else
    // `row_id_start + physical position`.
    let embedded = match unit.embedded_col_idx {
        Some(idx) => Some(
            batch
                .column(idx)
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| {
                    DataFusionError::Internal("embedded rowid column is not Int64".to_string())
                })?,
        ),
        None => None,
    };
    // Require a row_id_start only when a rowid is actually needed and there
    // is no embedded rowid to read.
    let synth_start: Option<i64> = if unit.need_rowid && embedded.is_none() {
        Some(unit.row_id_start.ok_or_else(|| {
            DataFusionError::Internal(
                "cannot synthesize deleted rowid: source file has neither an embedded \
                 rowid nor a row_id_start"
                    .to_string(),
            )
        })?)
    } else {
        None
    };

    // Find which rows in this batch are deleted, capturing each one's rowid
    // and delete snapshot (per-row in cumulative mode, constant otherwise).
    let mut keep_indices: Vec<u32> = Vec::new();
    let mut rowids: Vec<i64> = Vec::new();
    let mut snapshots: Vec<i64> = Vec::new();
    for i in 0..num_rows {
        let physical_pos = pos.value(i);
        if deleted_positions.contains(&physical_pos) {
            keep_indices.push(i as u32);
            // When rowid is projected away, emit a placeholder (dropped by
            // the ProjectionExec above) so no rowid needs synthesizing.
            let rowid = if !unit.need_rowid {
                0
            } else {
                match (embedded, synth_start) {
                    (Some(arr), _) => arr.value(i),
                    (None, Some(start)) => start + physical_pos,
                    // synth_start is Some whenever rowid is needed and there
                    // is no embedded rowid (resolved above).
                    (None, None) => unreachable!("row_id_start resolved above"),
                }
            };
            rowids.push(rowid);
            snapshots.push(if unit.cumulative {
                // Kept positions come from the windowed extraction, so the
                // map always holds them.
                *position_snapshots
                    .get(&physical_pos)
                    .unwrap_or(&unit.snapshot_id)
            } else {
                unit.snapshot_id
            });
        }
    }

    // If no deleted rows in this batch, return None
    if keep_indices.is_empty() {
        return Ok(None);
    }

    // Emit the CDC columns first — snapshot_id, rowid, change_type, the
    // official order — then the deleted rows' TABLE columns (excluding the
    // embedded rowid and position helper columns the scan read).
    let indices = UInt32Array::from(keep_indices.clone());
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(unit.table_len + 3);
    columns.push(Arc::new(Int64Array::from(snapshots)));
    columns.push(Arc::new(Int64Array::from(rowids)));
    columns.push(Arc::new(StringArray::from(vec![
        "delete";
        keep_indices.len()
    ])));

    for col in batch.columns().iter().take(unit.table_len) {
        let filtered = take(col.as_ref(), &indices, None)
            .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?;
        columns.push(filtered);
    }

    RecordBatch::try_new(unit.output_schema.clone(), columns)
        .map(Some)
        .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))
}
