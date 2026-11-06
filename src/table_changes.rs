//! Table changes (CDC) functionality for DuckLake
//!
//! This module provides the `ducklake_table_changes()` table function that returns
//! actual row data from Parquet files with additional CDC metadata columns —
//! inserts, deletes (with the deleted rows' old values), and UPDATEs correlated
//! into `update_preimage`/`update_postimage` pairs, matching official DuckLake.
//!
//! The feed is planned as official's `ducklake_table_changes` macro defines it:
//! the window's insertions ([`InsertedRowsExec`], plus inlined rows) left-joined
//! to its deletions ([`DeletedRowsExec`], plus inlined rows) on
//! `(snapshot_id, rowid)`, and the deletions to the insertions, unioned. Both
//! sides stream; each join holds only the other side's keys, in memory charged
//! to the session's pool.
//!
//! Note: Ordering across files is undefined unless explicitly requested via ORDER BY.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use arrow::array::{Array, ArrayRef, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, FieldRef, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use async_trait::async_trait;
use datafusion::catalog::Session;
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::common::{JoinType, NullEquality, Result as DataFusionResult};
use datafusion::datasource::listing::PartitionedFile;
use datafusion::datasource::physical_plan::{FileGroup, FileScanConfigBuilder, ParquetSource};
use datafusion::datasource::source::DataSourceExec;
use datafusion::datasource::table_schema::TableSchema;
use datafusion::datasource::{TableProvider, TableType};
use datafusion::error::DataFusionError;
use datafusion::execution::object_store::ObjectStoreUrl;
use datafusion::execution::{RecordBatchStream, SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::expressions::{CaseExpr, Column, is_not_null};
use datafusion::physical_expr::{EquivalenceProperties, PhysicalExpr};
use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion::physical_plan::execution_plan::reset_plan_states;
use datafusion::physical_plan::joins::utils::JoinOn;
use datafusion::physical_plan::joins::{HashJoinExec, PartitionMode};
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::physical_plan::union::UnionExec;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, PlanProperties,
};
use futures::{Stream, StreamExt, TryStreamExt};

use crate::column_rename::ColumnRenameExec;
use crate::feed_charge::{
    KeyedInlinedRows, PlanCharge, file_deletes_bytes, inlined_rows_bytes, plan_charge, release,
};
use crate::metadata_provider::{
    DataFileChange, DuckLakeTableColumn, InlinedFileDeleteChange, InlinedRowSelection,
    MetadataProvider, snapshot_change_tokens,
};
use crate::path_resolver::resolve_path;
use crate::row_id::positional_table_schema_reserving;
use crate::table::{
    ParquetFileLayout, cached_parquet_reader_factory, metadata_size_hint, session_parquet_source,
    validated_file_size,
};
use crate::table_deletions::{
    DeletedRowsExec, DeletionSource, FeedFiles, WindowDeletions, change_type_literal,
    keyed_rows_schema, label_rows, project_feed, projected_table_indices,
};
use crate::types::ABSENT_FIELD_PREFIX;
use crate::types::DuckLakeDefaultExprAdapterFactory;

#[cfg(feature = "encryption")]
use crate::encryption::EncryptionFactoryBuilder;
#[cfg(feature = "encryption")]
use datafusion::execution::parquet_encryption::EncryptionFactory;

/// Type of change captured in CDC output.
///
/// [`UpdatePreimage`](ChangeType::UpdatePreimage) /
/// [`UpdatePostimage`](ChangeType::UpdatePostimage) are the paired old/new row
/// versions of an `UPDATE`: `ducklake_table_changes` correlates a same-snapshot
/// delete + insert that share a rowid into this pair (the DuckLake spirit of an
/// update in a change feed) instead of surfacing them as an unrelated delete and
/// insert. The `as_str` values match the DuckLake change-feed spec strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChangeType {
    Insert,
    Delete,
    /// The old version of a row that an `UPDATE` rewrote in this snapshot.
    UpdatePreimage,
    /// The new version of a row that an `UPDATE` rewrote in this snapshot.
    UpdatePostimage,
}

impl ChangeType {
    /// Returns the string representation for Arrow output
    fn as_str(&self) -> &'static str {
        match self {
            ChangeType::Insert => "insert",
            ChangeType::Delete => "delete",
            ChangeType::UpdatePreimage => "update_preimage",
            ChangeType::UpdatePostimage => "update_postimage",
        }
    }
}

impl fmt::Display for ChangeType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Present a per-file CDC scan under the catalog schema: `table_fields` resolved
/// by field id and coerced to their catalog types, then whatever internal columns
/// the scan appended (embedded rowid, embedded per-row snapshot, physical
/// position) passed through unchanged, by name.
///
/// The wrap is decided against a target that does NOT depend on the file, because
/// a union's branches must all end up with the SAME output schema: `UnionExec`
/// validates the field count only, adopts the first branch's names, and panics on
/// a type mismatch. Wrapping "only the files that need renaming" would emit the
/// pre-rename column name for the others.
///
/// Two latent hazards, both shared with the ordinary table scan, which presents
/// its positional reads the same way:
///
/// - A NOT NULL top-level column that an older file does not carry is read as an
///   all-NULL array and then presented under the non-nullable catalog field, which
///   `record_batch_with_schema` rejects. DuckDB's own extension cannot create that
///   state (`ADD COLUMN … NOT NULL` is refused as "Adding columns with constraints
///   not yet supported"), but this crate's writer and third-party implementations
///   can.
/// - The internal column names are not reserved: nothing stops a table column
///   from being called `__ducklake_row_pos`, and nothing stops an older file from
///   physically carrying a column called `__ducklake_absent_field_<id>`. Either
///   makes two fields of one schema share a name, and the by-name lookup then
///   binds the first — reading the table column where the position was meant, or
///   real data where a null-fill was meant.
///
///   The position column is handled: its callers pass the catalog names to
///   `positional_table_schema_reserving`, which suffixes the internal name until
///   it clears both name spaces. The `__ducklake_absent_field_<id>` case, and the
///   embedded rowid / snapshot-id columns (whose names come from the file footer
///   and so cannot be suffixed), are not — binding those by index rather than by
///   name is the durable fix.
pub(crate) fn present_catalog_schema(
    scan: Arc<dyn ExecutionPlan>,
    table_fields: &[FieldRef],
    name_mapping: &HashMap<String, String>,
    constants: &HashMap<String, datafusion::common::ScalarValue>,
) -> Arc<dyn ExecutionPlan> {
    let scan_schema = scan.schema();
    debug_assert!(scan_schema.fields().len() >= table_fields.len());
    let mut fields: Vec<FieldRef> = table_fields.to_vec();
    fields.extend(
        scan_schema
            .fields()
            .iter()
            .skip(table_fields.len())
            .cloned(),
    );
    let output_schema: SchemaRef = Arc::new(Schema::new(fields));
    if !name_mapping.is_empty() || !constants.is_empty() || scan_schema != output_schema {
        Arc::new(ColumnRenameExec::new_with_constants(
            scan,
            output_schema,
            name_mapping.clone(),
            constants.clone(),
        ))
    } else {
        scan
    }
}

/// Refuse a column list that disagrees with the table schema built from it.
///
/// Every CDC read path locates a scan batch's internal columns (embedded rowid,
/// embedded per-row snapshot, physical position) by arithmetic on `table_len`,
/// while the per-file read schema is built from the column list. If the list is
/// LONGER, those indices address data columns and the feed reports a data value
/// as a rowid — wrong answers, no error. Both come from one
/// `get_table_structure` call inside the table functions, so this is
/// unreachable there; the providers' constructors and `with_columns` are
/// public, and this is the boundary check for a caller that pairs a schema with
/// the wrong columns.
pub(crate) fn check_column_count(table_len: usize, columns_len: usize) -> DataFusionResult<()> {
    if table_len != columns_len {
        return Err(DataFusionError::External(
            format!(
                "change feed built with {columns_len} column(s) but a {table_len}-field table \
                 schema; the schema and the column list must describe the same table"
            )
            .into(),
        ));
    }
    Ok(())
}

/// Index of a read-schema column the file actually carries. A CDC-columns-only
/// projection (`SELECT snapshot_id FROM …`, `COUNT(*)`) reads one data column
/// purely for its row count, and a column the file predates is materialised as a
/// null array rather than read from it.
pub(crate) fn row_count_probe_index(read_schema: &Schema) -> usize {
    read_schema
        .fields()
        .iter()
        .position(|f| !f.name().starts_with(ABSENT_FIELD_PREFIX))
        .unwrap_or(0)
}

/// Positions of the CDC metadata columns in the feed's output schema. They
/// LEAD the table columns, matching official DuckLake's `ducklake_table_changes`
/// projection (`SELECT snapshot_id, rowid, change_type, ...`).
const SNAPSHOT_ID_IDX: usize = 0;
const ROWID_IDX: usize = 1;
const CHANGE_TYPE_IDX: usize = 2;
/// Number of CDC metadata columns preceding the table columns.
const CDC_COLS: usize = 3;

/// Custom execution plan that prepends CDC columns (snapshot_id, rowid, change_type) to each batch
///
/// This plan wraps a ParquetExec and appends CDC metadata columns to each output batch.
/// It supports projection pushdown by:
/// - Reading only requested table columns from Parquet
/// - Including only requested CDC columns in output
/// - Optionally skipping input columns entirely when only CDC columns are needed
#[derive(Debug)]
pub struct PrependCDCColumnsExec {
    /// The input execution plan (typically ParquetExec)
    input: Arc<dyn ExecutionPlan>,
    /// Snapshot ID for this file
    snapshot_id: i64,
    /// Change type for this file
    change_type: ChangeType,
    /// Whether to include a rowid column in output. On this insert-only path
    /// rowid cannot be synthesized, so it is emitted as an all-NULL column
    /// (used only for encrypted tables, where the correlated path can't run).
    include_rowid: bool,
    /// Whether to include snapshot_id in output
    include_snapshot_id: bool,
    /// Whether to include change_type in output
    include_change_type: bool,
    /// If true, input columns are dummy (for row count only) and should not be included
    skip_input_columns: bool,
    /// Output schema (projected input schema + requested CDC columns)
    output_schema: SchemaRef,
    /// Cached plan properties with updated schema
    properties: Arc<PlanProperties>,
}

impl PrependCDCColumnsExec {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input: Arc<dyn ExecutionPlan>,
        snapshot_id: i64,
        change_type: ChangeType,
        include_rowid: bool,
        include_snapshot_id: bool,
        include_change_type: bool,
        skip_input_columns: bool,
        output_schema: SchemaRef,
    ) -> Self {
        // Create new equivalence properties with the output schema.
        // We preserve partitioning and execution semantics from input.
        // Note: This resets equivalences which is pessimistic but correct.
        // Future optimization: carry forward equivalences for projected table columns.
        let eq_properties = EquivalenceProperties::new(output_schema.clone());

        let properties = Arc::new(PlanProperties::new(
            eq_properties,
            input.output_partitioning().clone(),
            input.pipeline_behavior(),
            input.boundedness(),
        ));

        Self {
            input,
            snapshot_id,
            change_type,
            include_rowid,
            include_snapshot_id,
            include_change_type,
            skip_input_columns,
            output_schema,
            properties,
        }
    }
}

impl DisplayAs for PrependCDCColumnsExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default
            | DisplayFormatType::Verbose
            | DisplayFormatType::TreeRender => {
                write!(
                    f,
                    "PrependCDCColumnsExec: snapshot_id={}, change_type={}, \
                     include_snapshot={}, include_change={}, skip_input={}",
                    self.snapshot_id,
                    self.change_type,
                    self.include_snapshot_id,
                    self.include_change_type,
                    self.skip_input_columns
                )
            },
        }
    }
}

impl ExecutionPlan for PrependCDCColumnsExec {
    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> DataFusionResult<TreeNodeRecursion>,
    ) -> DataFusionResult<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn name(&self) -> &str {
        "PrependCDCColumnsExec"
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
                "PrependCDCColumnsExec expects exactly one child".into(),
            ));
        }

        Ok(Arc::new(PrependCDCColumnsExec::new(
            children[0].clone(),
            self.snapshot_id,
            self.change_type,
            self.include_rowid,
            self.include_snapshot_id,
            self.include_change_type,
            self.skip_input_columns,
            self.output_schema.clone(),
        )))
    }

    fn schema(&self) -> SchemaRef {
        self.output_schema.clone()
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        let input_stream = self.input.execute(partition, context)?;

        Ok(Box::pin(PrependCDCColumnsStream {
            input: input_stream,
            snapshot_id: self.snapshot_id,
            change_type: self.change_type,
            include_rowid: self.include_rowid,
            include_snapshot_id: self.include_snapshot_id,
            include_change_type: self.include_change_type,
            skip_input_columns: self.skip_input_columns,
            output_schema: self.output_schema.clone(),
        }))
    }
}

/// Stream that appends CDC columns to input batches
struct PrependCDCColumnsStream {
    input: SendableRecordBatchStream,
    snapshot_id: i64,
    change_type: ChangeType,
    include_rowid: bool,
    include_snapshot_id: bool,
    include_change_type: bool,
    skip_input_columns: bool,
    output_schema: SchemaRef,
}

impl Stream for PrependCDCColumnsStream {
    type Item = DataFusionResult<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match Pin::new(&mut self.input).poll_next(cx) {
            Poll::Ready(Some(Ok(batch))) => {
                let result = self.transform_batch(&batch);
                Poll::Ready(Some(result))
            },
            Poll::Ready(Some(Err(e))) => Poll::Ready(Some(Err(e))),
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

impl PrependCDCColumnsStream {
    fn transform_batch(&self, batch: &RecordBatch) -> DataFusionResult<RecordBatch> {
        let num_rows = batch.num_rows();
        let mut columns: Vec<ArrayRef> = Vec::new();

        // Prepend requested CDC columns, in the order snapshot_id, rowid,
        // change_type (official DuckLake order). rowid is all-NULL here: this
        // insert-only path can't synthesize it (used for encrypted tables).
        if self.include_snapshot_id {
            columns.push(Arc::new(Int64Array::from(vec![self.snapshot_id; num_rows])));
        }
        if self.include_rowid {
            columns.push(Arc::new(Int64Array::from(vec![None::<i64>; num_rows])));
        }
        if self.include_change_type {
            columns.push(Arc::new(StringArray::from(vec![
                self.change_type.as_str();
                num_rows
            ])));
        }

        // Then the input columns, unless we're skipping them
        if !self.skip_input_columns {
            columns.extend(batch.columns().iter().cloned());
        }

        // A zero-column projection (e.g. `COUNT(*)`) still needs the row count.
        RecordBatch::try_new_with_options(
            self.output_schema.clone(),
            columns,
            &arrow::record_batch::RecordBatchOptions::new().with_row_count(Some(num_rows)),
        )
        .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))
    }
}

impl RecordBatchStream for PrependCDCColumnsStream {
    fn schema(&self) -> SchemaRef {
        self.output_schema.clone()
    }
}

/// Projection analysis result: maps logical projection to physical components
struct ProjectionInfo {
    /// Table column indices to read from Parquet (in original order)
    table_indices: Vec<usize>,
    /// Whether rowid is requested
    need_rowid: bool,
    /// Whether snapshot_id is requested
    need_snapshot_id: bool,
    /// Whether change_type is requested
    need_change_type: bool,
    /// The projected output schema
    output_schema: SchemaRef,
}

#[derive(Debug)]
pub struct TableChangesTable {
    provider: Arc<dyn MetadataProvider>,
    table_id: i64,
    start_snapshot: i64,
    end_snapshot: i64,
    /// Object store URL for resolving file paths
    object_store_url: Arc<ObjectStoreUrl>,
    /// Table path for resolving relative file paths
    table_path: String,
    /// Original table schema (without CDC columns)
    table_schema: SchemaRef,
    /// Combined schema: table columns + snapshot_id + change_type
    output_schema: SchemaRef,
    /// The table's columns as of `end_snapshot`, carrying the field ids each data
    /// file's columns are resolved by. Set via [`Self::with_columns`]; fetched
    /// from the metadata provider on demand when it is not.
    columns: Option<Arc<Vec<DuckLakeTableColumn>>>,
    /// Per-file read layout, memoized by resolved path. A file inserted and then
    /// deleted inside the window is reached from both sides of the feed, and a
    /// delete's source data file can be reached by several delete records.
    layout_cache: Mutex<HashMap<String, Arc<ParquetFileLayout>>>,
    /// When set, the delete side is never read: every row added in the window
    /// surfaces as `insert` (the `ducklake_table_insertions` feed).
    insertions_only: bool,
}

impl TableChangesTable {
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
        // rowid, change_type), official DuckLake's ducklake_table_changes
        // column order — then the table columns.
        let mut fields: Vec<Field> = Vec::with_capacity(table_schema.fields().len() + CDC_COLS);
        fields.push(Field::new("snapshot_id", DataType::Int64, false));
        // rowid is nullable: it is NULL on encrypted (PME) tables, where the
        // correlated feed cannot decrypt footers to resolve rowids.
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
            columns: None,
            layout_cache: Mutex::new(HashMap::new()),
            insertions_only: false,
        }
    }

    /// Supply the table's columns as of the window's end snapshot.
    ///
    /// A data file records each column under the name it had when the file was
    /// written, tagged with the column's field id, so the feed resolves columns by
    /// field id rather than by name. Without this the columns are fetched from the
    /// metadata provider on each scan; passing them in avoids that query when the
    /// caller already holds them.
    pub fn with_columns(mut self, columns: Vec<DuckLakeTableColumn>) -> Self {
        self.columns = Some(Arc::new(columns));
        self
    }

    /// Turn this feed into `ducklake_table_insertions`: the delete side is
    /// never read, so every row added in the window — plain inserts, UPDATE
    /// postimages, in-window rows of merged partial files — surfaces as
    /// `insert`, matching official DuckLake's insertions feed.
    pub fn insertions_only(mut self) -> Self {
        self.insertions_only = true;
        self
    }

    /// The table's columns as of the window's end snapshot: whatever
    /// [`Self::with_columns`] was given, else a fresh metadata read.
    fn resolve_columns(&self) -> DataFusionResult<Arc<Vec<DuckLakeTableColumn>>> {
        match &self.columns {
            Some(columns) => Ok(Arc::clone(columns)),
            None => {
                let columns = self
                    .provider
                    .get_table_structure(self.table_id, self.end_snapshot)
                    .map_err(|e| DataFusionError::External(Box::new(e)))?;
                Ok(Arc::new(columns))
            },
        }
    }

    /// The view of this feed the per-file planners read.
    fn feed_files(&self) -> FeedFiles<'_> {
        FeedFiles {
            provider: self.provider.as_ref(),
            table_id: self.table_id,
            start_snapshot: self.start_snapshot,
            end_snapshot: self.end_snapshot,
            object_store_url: self.object_store_url.as_ref(),
            table_path: &self.table_path,
            table_schema: &self.table_schema,
            layout_cache: &self.layout_cache,
        }
    }

    /// Analyze projection and split into table columns and CDC columns.
    /// CDC columns lead the table columns in the order `snapshot_id`, `rowid`,
    /// `change_type` (official DuckLake order).
    fn analyze_projection(&self, projection: Option<&Vec<usize>>) -> ProjectionInfo {
        let num_table_cols = self.table_schema.fields().len();

        match projection {
            None => {
                // No projection - read all columns
                ProjectionInfo {
                    table_indices: (0..num_table_cols).collect(),
                    need_rowid: true,
                    need_snapshot_id: true,
                    need_change_type: true,
                    output_schema: self.output_schema.clone(),
                }
            },
            Some(indices) => {
                // Split indices into table columns and CDC columns
                let mut table_indices: Vec<usize> = Vec::new();
                let mut need_rowid = false;
                let mut need_snapshot_id = false;
                let mut need_change_type = false;

                for &idx in indices {
                    match idx {
                        SNAPSHOT_ID_IDX => need_snapshot_id = true,
                        ROWID_IDX => need_rowid = true,
                        CHANGE_TYPE_IDX => need_change_type = true,
                        _ if idx < num_table_cols + CDC_COLS => {
                            table_indices.push(idx - CDC_COLS);
                        },
                        _ => {},
                    }
                }

                // Build projected output schema in the order requested
                let mut fields: Vec<Field> = Vec::with_capacity(indices.len());
                for &idx in indices {
                    fields.push(self.output_schema.field(idx).clone());
                }
                let output_schema = Arc::new(Schema::new(fields));

                ProjectionInfo {
                    table_indices,
                    need_rowid,
                    need_snapshot_id,
                    need_change_type,
                    output_schema,
                }
            },
        }
    }

    /// Build the schema that PrependCDCColumnsExec will output. On this
    /// (encryption-aware, insert-only) path rowid cannot be synthesized, so when
    /// requested it is emitted as a nullable, all-NULL column.
    fn build_cdc_exec_schema(
        &self,
        table_indices: &[usize],
        need_rowid: bool,
        need_snapshot_id: bool,
        need_change_type: bool,
    ) -> SchemaRef {
        let mut fields: Vec<Field> = Vec::with_capacity(table_indices.len() + CDC_COLS);

        if need_snapshot_id {
            fields.push(Field::new("snapshot_id", DataType::Int64, false));
        }
        if need_rowid {
            fields.push(Field::new("rowid", DataType::Int64, true));
        }
        if need_change_type {
            fields.push(Field::new("change_type", DataType::Utf8, false));
        }
        fields.extend(
            table_indices
                .iter()
                .map(|&i| self.table_schema.field(i).clone()),
        );

        Arc::new(Schema::new(fields))
    }

    /// Build a ParquetExec wrapped with PrependCDCColumnsExec for a single file.
    /// `layout` is `None` only for an encrypted file, whose footer this path
    /// cannot decrypt: its columns are then matched by name against the catalog
    /// schema (see the guard in [`TableProvider::scan`]).
    #[cfg(feature = "encryption")]
    async fn build_exec_for_file(
        &self,
        state: &dyn Session,
        data_file: &DataFileChange,
        layout: Option<&ParquetFileLayout>,
        proj_info: &ProjectionInfo,
        encryption_factory: &Option<Arc<dyn EncryptionFactory>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let read_schema = self.file_read_schema(layout);
        let reader_factory = cached_parquet_reader_factory(state, self.object_store_url.as_ref())?;
        let parquet_source = if let Some(factory) = encryption_factory {
            session_parquet_source(state, read_schema)
                .with_encryption_factory(Arc::clone(factory))
                .with_parquet_file_reader_factory(reader_factory)
        } else {
            session_parquet_source(state, read_schema)
                .with_parquet_file_reader_factory(reader_factory)
        };
        self.build_exec_for_file_impl(state, data_file, layout, proj_info, parquet_source)
            .await
    }

    /// Build a ParquetExec wrapped with PrependCDCColumnsExec for a single file
    #[cfg(not(feature = "encryption"))]
    async fn build_exec_for_file(
        &self,
        state: &dyn Session,
        data_file: &DataFileChange,
        layout: Option<&ParquetFileLayout>,
        proj_info: &ProjectionInfo,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let reader_factory = cached_parquet_reader_factory(state, self.object_store_url.as_ref())?;
        let parquet_source = session_parquet_source(state, self.file_read_schema(layout))
            .with_parquet_file_reader_factory(reader_factory);
        self.build_exec_for_file_impl(state, data_file, layout, proj_info, parquet_source)
            .await
    }

    /// The schema to scan one data file's table columns with: the file's own
    /// field-id-resolved read schema, or the catalog schema when there is no
    /// layout (an encrypted file).
    fn file_read_schema(&self, layout: Option<&ParquetFileLayout>) -> SchemaRef {
        match layout {
            Some(layout) => Arc::clone(&layout.read_schema),
            None => Arc::clone(&self.table_schema),
        }
    }

    /// Internal implementation for building a ParquetExec wrapped with PrependCDCColumnsExec
    async fn build_exec_for_file_impl(
        &self,
        _state: &dyn Session,
        data_file: &DataFileChange,
        layout: Option<&ParquetFileLayout>,
        proj_info: &ProjectionInfo,
        parquet_source: ParquetSource,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        // Resolve file path
        let resolved_path = resolve_path(
            &self.table_path,
            &data_file.path,
            data_file.path_is_relative,
        )
        .map_err(|e| DataFusionError::External(Box::new(e)))?;

        // Create PartitionedFile with footer size hint if available
        let mut pf = PartitionedFile::new(
            &resolved_path,
            validated_file_size(data_file.file_size_bytes, &resolved_path)?,
        );
        if let Some(hint) = metadata_size_hint(data_file.footer_size) {
            pf = pf.with_metadata_size_hint(hint);
        }

        // Determine what to read from Parquet
        let parquet_projection = if proj_info.table_indices.is_empty() {
            // Only CDC columns requested - read minimal data for row counts
            Some(vec![row_count_probe_index(&self.file_read_schema(layout))])
        } else {
            Some(proj_info.table_indices.clone())
        };

        // Create file scan config with projection pushdown
        let mut builder = FileScanConfigBuilder::new(
            self.object_store_url.as_ref().clone(),
            Arc::new(parquet_source),
        )
        .with_expr_adapter(Some(Arc::new(DuckLakeDefaultExprAdapterFactory)))
        .with_file_group(FileGroup::new(vec![pf]));

        if let Some(proj) = parquet_projection {
            builder = builder.with_projection_indices(Some(proj))?;
        }

        let file_scan_config = builder.build();

        // Use DataSourceExec directly to preserve our ParquetSource with encryption factory
        let mut parquet_exec: Arc<dyn ExecutionPlan> =
            DataSourceExec::from_data_source(file_scan_config);

        // Determine if we should skip input columns (only CDC columns requested)
        let skip_input_columns = proj_info.table_indices.is_empty();

        // Present the projected catalog fields, so every file's branch of the
        // union below carries the same schema and `PrependCDCColumnsExec` can
        // build batches under it. Nothing to present when the table columns are
        // dropped anyway (a CDC-columns-only projection).
        if !skip_input_columns {
            let table_fields: Vec<FieldRef> = proj_info
                .table_indices
                .iter()
                .map(|&i| Arc::clone(&self.table_schema.fields()[i]))
                .collect();
            let name_mapping = layout.map(|l| l.name_mapping.clone()).unwrap_or_default();
            let constants = layout.map(|l| l.constants.clone()).unwrap_or_default();
            parquet_exec =
                present_catalog_schema(parquet_exec, &table_fields, &name_mapping, &constants);
        }

        // Build output schema for PrependCDCColumnsExec
        let cdc_exec_schema = if skip_input_columns {
            // Only CDC columns - build schema with just those
            let mut fields = Vec::new();
            if proj_info.need_snapshot_id {
                fields.push(Field::new("snapshot_id", DataType::Int64, false));
            }
            if proj_info.need_rowid {
                fields.push(Field::new("rowid", DataType::Int64, true));
            }
            if proj_info.need_change_type {
                fields.push(Field::new("change_type", DataType::Utf8, false));
            }
            Arc::new(Schema::new(fields))
        } else {
            self.build_cdc_exec_schema(
                &proj_info.table_indices,
                proj_info.need_rowid,
                proj_info.need_snapshot_id,
                proj_info.need_change_type,
            )
        };

        Ok(Arc::new(PrependCDCColumnsExec::new(
            parquet_exec,
            data_file.begin_snapshot,
            ChangeType::Insert,
            proj_info.need_rowid,
            proj_info.need_snapshot_id,
            proj_info.need_change_type,
            skip_input_columns,
            cdc_exec_schema,
        )))
    }

    /// The read schema for a data file: its table columns under the physical names
    /// THIS file gives them, plus the embedded rowid / per-row snapshot columns
    /// when present, in that order.
    fn read_schema_with_embedded(
        &self,
        layout: &ParquetFileLayout,
        rowid_name: &Option<String>,
        snapshot_name: &Option<String>,
    ) -> SchemaRef {
        match (rowid_name, snapshot_name) {
            (None, None) => Arc::clone(&layout.read_schema),
            (rowid, snapshot) => {
                let mut fields: Vec<FieldRef> =
                    layout.read_schema.fields().iter().cloned().collect();
                if let Some(name) = rowid {
                    fields.push(Arc::new(Field::new(name, DataType::Int64, true)));
                }
                if let Some(name) = snapshot {
                    fields.push(Arc::new(Field::new(name, DataType::Int64, true)));
                }
                Arc::new(Schema::new(fields))
            },
        }
    }

    /// The read of an inserted data file for the correlated feed, projected to
    /// the table columns at `table_indices`.
    ///
    /// A file with an embedded rowid (an UPDATE / compaction postimage) is
    /// scanned plainly — its rowid IS the embedded column. A plain insert is
    /// scanned positionally when `resolve_rowid` — the parquet reader appends
    /// the physical position — so its rowid can be synthesized as
    /// `row_id_start + position`; otherwise it is a plain scan and its rowid
    /// is not resolved. A merged partial file also reads its embedded per-row
    /// snapshot column.
    ///
    /// The scan's columns are the projected table columns, presented under the
    /// catalog schema, then a row-count probe when nothing else is read, then
    /// the embedded rowid, the embedded snapshot and the position, each when
    /// read.
    fn insert_source(
        &self,
        state: &dyn Session,
        data_file: &DataFileChange,
        layout: &ParquetFileLayout,
        resolve_rowid: bool,
        table_indices: &[usize],
        rewrites: &BTreeSet<i64>,
    ) -> DataFusionResult<InsertSource> {
        let reader_factory = cached_parquet_reader_factory(state, self.object_store_url.as_ref())?;
        let resolved = resolve_path(
            &self.table_path,
            &data_file.path,
            data_file.path_is_relative,
        )
        .map_err(|e| DataFusionError::External(Box::new(e)))?;
        let mut pf = PartitionedFile::new(
            &resolved,
            validated_file_size(data_file.file_size_bytes, &resolved)?,
        );
        if let Some(hint) = metadata_size_hint(data_file.footer_size) {
            pf = pf.with_metadata_size_hint(hint);
        }
        // The per-row snapshot column is read only for partial (merged) files.
        let snapshot_name = if data_file.partial_max.is_some() {
            layout.embedded_snapshot_parquet_name.clone()
        } else {
            None
        };
        let embedded_rowid = &layout.embedded_rowid_parquet_name;
        let read_schema = self.read_schema_with_embedded(layout, embedded_rowid, &snapshot_name);
        let table_len = layout.read_schema.fields().len();

        let mut projection = table_indices.to_vec();
        if table_indices.is_empty() && embedded_rowid.is_none() && snapshot_name.is_none() {
            // A data column for the row count: the position alone is a virtual
            // column, and the reader needs a real one to count.
            projection.push(row_count_probe_index(&layout.read_schema));
        }
        let mut next = table_len;
        let mut take_embedded = || {
            projection.push(next);
            next += 1;
            projection.len() - 1
        };
        let rowid_col = embedded_rowid.as_ref().map(|_| take_embedded());
        let snapshot_col = snapshot_name.as_ref().map(|_| take_embedded());
        let (scan_schema, pos_col): (TableSchema, Option<usize>) =
            if embedded_rowid.is_none() && resolve_rowid {
                // Re-presented under the catalog names below, where
                // `ColumnRenameExec` binds by name; keep the position column
                // clear of those as well as of the file's own names.
                let (schema, pos_index, _) = positional_table_schema_reserving(
                    read_schema,
                    self.table_schema.fields().iter().map(|f| f.name().as_str()),
                );
                projection.push(pos_index);
                (schema, Some(projection.len() - 1))
            } else {
                (read_schema.into(), None)
            };
        let builder = FileScanConfigBuilder::new(
            self.object_store_url.as_ref().clone(),
            Arc::new(
                session_parquet_source(state, scan_schema)
                    .with_parquet_file_reader_factory(reader_factory),
            ),
        )
        .with_expr_adapter(Some(Arc::new(DuckLakeDefaultExprAdapterFactory)))
        .with_file_group(FileGroup::new(vec![pf]))
        .with_projection_indices(Some(projection))?;
        let table_fields: Vec<FieldRef> = table_indices
            .iter()
            .map(|&i| Arc::clone(&self.table_schema.fields()[i]))
            .collect();
        let scan = present_catalog_schema(
            DataSourceExec::from_data_source(builder.build()),
            &table_fields,
            &layout.name_mapping,
            &layout.constants,
        );
        #[cfg(debug_assertions)]
        if let Some(idx) = pos_col {
            let schema = scan.schema();
            let field = schema.fields().get(idx).unwrap_or_else(|| {
                panic!(
                    "pos_col {idx} is out of range for a scan of {} columns",
                    schema.fields().len()
                )
            });
            crate::row_id::validate_row_pos_field("insert_source", idx, field)
                .expect("`pos_col` arithmetic must land on the position column");
        }
        let rowid = match (rowid_col, pos_col) {
            (Some(column), _) => InsertRowid::Embedded(column),
            (None, Some(position)) => InsertRowid::Positional {
                position,
                row_id_start: data_file.row_id_start,
            },
            (None, None) => InsertRowid::Unresolved,
        };
        let excluded_snapshots = match data_file.partial_max {
            Some(partial_max) if partial_max >= data_file.begin_snapshot => rewrites
                .range(data_file.begin_snapshot..=partial_max)
                .copied()
                .collect(),
            _ => Vec::new(),
        };
        Ok(InsertSource {
            scan,
            snapshot_id: data_file.begin_snapshot,
            rowid,
            snapshot_col,
            excluded_snapshots,
        })
    }

    /// Refuse the feed when an encrypted table's columns changed identity between
    /// the oldest data file in the window and the window's end.
    ///
    /// Encrypted files' footers cannot be decrypted on this path, so their columns
    /// can only be matched by name — correct only while every in-window file
    /// records the columns under the names the end snapshot uses. A rename or a
    /// drop-and-re-add breaks that, and a by-name read would then return NULL or
    /// another column's values with no indication anything was wrong. Adding a
    /// column, and dropping one outright, stay safe: the new name is in no older
    /// file (so it null-fills) and the dropped one is asked for by nobody.
    ///
    /// The identity below is coarser than "was a column renamed", so this
    /// refuses strictly more than it has to: a type widened by `ALTER … TYPE`
    /// and a struct child added inside an existing column both change it, and
    /// both would in fact read correctly by name. Erring toward refusal is the
    /// point — the alternative on this path is silently wrong values — but a
    /// finer check (compare names per field id, ignore the rest) would let those
    /// two through if they turn out to matter. `COMPATIBILITY.md` says so.
    #[cfg(feature = "encryption")]
    fn reject_evolved_encrypted_table(
        &self,
        columns: &[DuckLakeTableColumn],
        data_files: &[DataFileChange],
    ) -> DataFusionResult<()> {
        let Some(oldest) = data_files.iter().map(|f| f.begin_snapshot).min() else {
            return Ok(());
        };
        if oldest >= self.end_snapshot {
            return Ok(());
        }
        let then = self
            .provider
            .get_table_structure(self.table_id, oldest)
            .map_err(|e| DataFusionError::External(Box::new(e)))?;

        // A column's identity is its name plus its resolved type — the type is
        // what carries a NESTED field's name, so a renamed struct child shows up
        // here too.
        let identity = |column: &DuckLakeTableColumn| -> DataFusionResult<(String, String)> {
            let data_type = column
                .data_type()
                .map_err(|e| DataFusionError::External(Box::new(e)))?;
            Ok((column.column_name.clone(), format!("{data_type:?}")))
        };
        let mut then_by_id: HashMap<i64, (String, String)> = HashMap::new();
        let mut then_names: HashSet<String> = HashSet::new();
        for column in &then {
            then_by_id.insert(column.column_id, identity(column)?);
            then_names.insert(column.column_name.clone());
        }

        let mut evolved = false;
        for column in columns {
            evolved = match then_by_id.get(&column.column_id) {
                Some(previous) => *previous != identity(column)?,
                // A column id the older snapshot did not have: safe unless its
                // name did exist back then under a different id — that is a drop
                // and re-add, and a by-name read would find the dropped column's
                // data in the older files.
                None => then_names.contains(&column.column_name),
            };
            if evolved {
                break;
            }
        }

        if evolved {
            return Err(DataFusionError::External(
                format!(
                    "table {} has encrypted data files and its columns changed between \
                     snapshot {oldest} and snapshot {}: a column was renamed, or dropped and \
                     re-added under the same name. Change feeds resolve columns by field id, \
                     which requires reading each file's parquet footer, and an encrypted \
                     footer cannot be read here — so the feed would return another column's \
                     values. Query a snapshot window whose files all predate the change, or \
                     read the table directly.",
                    self.table_id, self.end_snapshot
                )
                .into(),
            ));
        }
        Ok(())
    }

    /// Build the correlated change feed as official DuckLake's
    /// `ducklake_table_changes` macro defines it: the window's insertions
    /// left-joined to its deletions on `(snapshot_id, rowid)` — a matched row
    /// is an `update_postimage`, the rest `insert` — and the deletions
    /// left-joined to the insertions — `update_preimage`, else `delete` —
    /// unioned. A row that shares its key with several rows of the other side
    /// is reported once per such row, as official's join reports it.
    ///
    /// Each side streams. The joins build on the other side's keys alone
    /// (`(snapshot_id, rowid)`, read with no table column), in a hash table
    /// charged to the session's memory pool.
    ///
    /// A plain insert (no embedded rowid, not a merged partial file) carries a
    /// rowid first assigned in its own snapshot, which no deletion of that
    /// snapshot holds — a row inserted and deleted in one commit is never
    /// written — so plain inserts skip the join and surface as `insert`; that
    /// also lets them leave their rowid unresolved when it is not projected.
    #[allow(clippy::too_many_arguments)]
    async fn build_correlated_changes(
        &self,
        state: &dyn Session,
        columns: &[DuckLakeTableColumn],
        data_files: &[DataFileChange],
        window_deletions: WindowDeletions,
        rewrites: &BTreeSet<i64>,
        inlined: InlinedChanges,
        layouts: &[Arc<ParquetFileLayout>],
        projection: Option<&Vec<usize>>,
        charge: &PlanCharge,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let table_len = self.table_schema.fields().len();
        // Both come from the same column list, and the index arithmetic of the
        // scans (which column of a scan batch is the embedded rowid, the
        // embedded snapshot, the position) assumes they agree. A disagreement
        // would silently misalign every per-file scan — with more columns than
        // `table_len` the internal-column indices land on DATA columns and the
        // feed emits a data value as a rowid. Unreachable through the table
        // functions, which derive both from one `get_table_structure` call, but
        // the constructor and `with_columns` are public: refuse loudly.
        check_column_count(table_len, columns.len())?;
        // The table columns the caller reads, and whether it reads the rowid
        // column (at ROWID_IDX among the leading CDC columns). When it does
        // not, plain inserts skip the positional scan and rowid synthesis
        // entirely.
        let (table_indices, need_rowid) = projected_table_indices(projection, CDC_COLS, table_len);

        // Inserted files split into those whose rows can pair with a deletion
        // (an embedded rowid, or a merged partial file whose real rowids feed
        // the pairing) and plain inserts.
        let mut plain = Vec::new();
        let mut pairable = Vec::new();
        for (df, layout) in data_files.iter().zip(layouts.iter()) {
            let is_partial = df.partial_max.is_some();
            if is_partial && layout.embedded_snapshot_parquet_name.is_none() {
                return Err(DataFusionError::External(
                    format!(
                        "data file {} is a merged partial file (partial_max set) but carries \
                         no embedded per-row snapshot column; cannot attribute its rows to \
                         snapshots",
                        df.path
                    )
                    .into(),
                ));
            }
            if is_partial || layout.embedded_rowid_parquet_name.is_some() {
                pairable.push((df, layout));
            } else {
                plain.push((df, layout));
            }
        }

        let files = self.feed_files();
        // Every delete in range is read: unmatched ones surface as `delete`
        // rows, and those sharing a (snapshot_id, rowid) with an insertion
        // pair into update preimages.
        let InlinedChanges {
            inserts: inlined_inserts,
            deletes: inlined_deletes,
            file_deletes,
        } = inlined;
        let deletions = Arc::new(
            files
                .deletion_sources(state, columns, &window_deletions, file_deletes, charge)
                .await?,
        );
        // The catalog reads not kept are released: the charge follows what the
        // plan holds.
        drop(window_deletions);
        charge.try_resize(
            inlined_inserts.bytes()
                + inlined_deletes.bytes()
                + deletions
                    .iter()
                    .map(DeletionSource::held_bytes)
                    .sum::<usize>(),
        )?;
        let has_pairable = !pairable.is_empty() || !inlined_inserts.is_empty();
        let has_deletions = !deletions.is_empty() || !inlined_deletes.is_empty();

        let window = (self.start_snapshot, self.end_snapshot);
        let inserted = |files: &[(&DataFileChange, &Arc<ParquetFileLayout>)],
                        resolve_rowid: bool,
                        table_indices: &[usize]|
         -> DataFusionResult<Option<Arc<dyn ExecutionPlan>>> {
            if files.is_empty() {
                return Ok(None);
            }
            let sources = files
                .iter()
                .map(|(df, layout)| {
                    // A partial file's rows carry REAL rowids into the update
                    // correlation, so rowid is resolved for them even when it
                    // is not projected.
                    let resolve = resolve_rowid || df.partial_max.is_some();
                    self.insert_source(state, df, layout, resolve, table_indices, rewrites)
                })
                .collect::<DataFusionResult<Vec<_>>>()?;
            Ok(Some(Arc::new(InsertedRowsExec::new(
                sources,
                keyed_rows_schema(&self.table_schema, table_indices),
                table_indices.len(),
                window,
                state.config().target_partitions(),
            ))))
        };
        // Inserted rows that can pair: the pairable files and the inlined
        // inserts, which carry their row ids (one written by an UPDATE keeps
        // the row id of the version it supersedes).
        let pairable_rows =
            |table_indices: &[usize]| -> DataFusionResult<Option<Arc<dyn ExecutionPlan>>> {
                let mut execs: Vec<Arc<dyn ExecutionPlan>> = Vec::with_capacity(2);
                execs.extend(inserted(&pairable, true, table_indices)?);
                if !inlined_inserts.is_empty() {
                    execs.push(inlined_inserts.exec(table_indices)?);
                }
                union_of(execs)
            };
        // Deleted rows: the positions newly masked in the window and the
        // inlined row versions deleted in it. Their rowids are resolved when
        // output or when an insertion could pair with them.
        let deleted_rows = |table_indices: &[usize],
                            resolve_rowid: bool|
         -> DataFusionResult<Option<Arc<dyn ExecutionPlan>>> {
            let mut execs: Vec<Arc<dyn ExecutionPlan>> = Vec::with_capacity(2);
            if !deletions.is_empty() {
                execs.push(Arc::new(DeletedRowsExec::try_new(
                    state,
                    &files,
                    Arc::clone(&deletions),
                    table_indices,
                    resolve_rowid,
                    charge,
                )?));
            }
            if !inlined_deletes.is_empty() {
                execs.push(inlined_deletes.exec(table_indices)?);
            }
            union_of(execs)
        };

        let mut branches: Vec<Arc<dyn ExecutionPlan>> = Vec::with_capacity(3);
        if let Some(rows) = inserted(&plain, need_rowid, &table_indices)? {
            branches.push(label_rows(rows, ChangeType::Insert.as_str())?);
        }
        if let Some(rows) = pairable_rows(&table_indices)? {
            let keys = if has_deletions {
                deleted_rows(&[], true)?
            } else {
                None
            };
            branches.push(pair_rows(
                rows,
                keys,
                ChangeType::UpdatePostimage,
                ChangeType::Insert,
            )?);
        }
        if let Some(rows) = deleted_rows(&table_indices, need_rowid || has_pairable)? {
            let keys = if has_pairable {
                pairable_rows(&[])?
            } else {
                None
            };
            branches.push(pair_rows(
                rows,
                keys,
                ChangeType::UpdatePreimage,
                ChangeType::Delete,
            )?);
        }
        let Some(changes) = union_of(branches)? else {
            use datafusion::physical_plan::empty::EmptyExec;
            return Ok(Arc::new(EmptyExec::new(
                self.analyze_projection(projection).output_schema,
            )));
        };
        project_feed(
            changes,
            projection,
            &self.output_schema,
            CDC_COLS,
            &table_indices,
        )
    }
}

#[async_trait]
impl TableProvider for TableChangesTable {
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
        // Analyze projection to determine what to read
        let proj_info = self.analyze_projection(projection);

        // The columns as of the window's END snapshot, carrying the field ids the
        // per-file read schemas are built from — official DuckLake resolves a
        // change feed against exactly that generation of the schema.
        let columns = self.resolve_columns()?;

        // The data files and delete files added in the window, and the row
        // versions stored in the catalog rather than in Parquet, read in one
        // consistent read of the catalog
        // (`MetadataProvider::begin_consistent_read`): a flush that commits
        // meanwhile moves inlined rows or deletions into files, and separate
        // reads could see them in both places or in neither.
        let window = (self.start_snapshot, self.end_snapshot);
        // What the plan holds from the catalog is charged to the session's
        // memory pool as it is read, and for as long as the plan holds it.
        let charge = plan_charge(state, "ducklake_table_changes");
        let (data_files, window_deletions, rewrites, inlined) =
            crate::metadata_provider::read_consistently(&self.provider, |provider| {
                let external = |e| DataFusionError::External(Box::new(e));
                // A rewrite's snapshot changes no row: the feed reports neither
                // the files it ends nor the rows it writes.
                let rewrites = rewrite_snapshots(provider.as_ref(), self.table_id, window)?;
                // INSERT changes.
                let mut data_files = provider
                    .get_data_files_added_between_snapshots(
                        self.table_id,
                        self.start_snapshot,
                        self.end_snapshot,
                    )
                    .map_err(external)?;
                data_files.retain(|file| !written_by_rewrite(file, &rewrites));
                // Deletes applied in the window surface as `delete` rows (and
                // pair into update preimages), so they participate in both the
                // empty check and the path decision — a delete-only window is
                // NOT empty. The insertions feed never reads the delete side.
                let window_deletions = if self.insertions_only {
                    WindowDeletions::default()
                } else {
                    WindowDeletions::read(
                        provider.as_ref(),
                        self.table_id,
                        window,
                        &rewrites,
                        &charge,
                    )?
                };
                // Inlined inserts, inlined rows deleted or superseded in the
                // window, and inlined deletions of Parquet-resident rows. Each
                // carries its own row id and snapshot, so none of them depends
                // on scan order.
                let inlined = InlinedChanges::read(
                    provider.as_ref(),
                    self.table_id,
                    &columns,
                    &self.table_schema,
                    (self.start_snapshot, self.end_snapshot),
                    !self.insertions_only,
                    &charge,
                )?;
                Ok::<_, DataFusionError>((data_files, window_deletions, rewrites, inlined))
            })?;

        // Handle empty case
        if data_files.is_empty() && window_deletions.is_empty() && inlined.is_empty() {
            use datafusion::physical_plan::empty::EmptyExec;
            return Ok(Arc::new(EmptyExec::new(proj_info.output_schema)));
        }

        let mut data_files = data_files;

        // Decide whether to take the correlated path (reading delete sources to
        // emit `delete` rows and pair an UPDATE's delete+insert into
        // preimage/postimage). Guards, all cheap and metadata-only:
        //
        //  1. Deletes-present OR rowid-requested: with neither, the window is
        //     plain inserts and needs no correlation — a plain-INSERT catalog
        //     does ZERO per-file parquet footer reads at plan time.
        //  2. Not encrypted: the correlated path reads parquet footers (to detect
        //     the embedded-rowid postimage) and the source rows of deletes, none
        //     of which it can decrypt (the delete-side change record carries no
        //     key). On a PME catalog we therefore stay on the insert-only path
        //     below — which IS encryption-aware — so CDC over inserts never
        //     fails; the tradeoff is that UPDATEs surface as plain inserts and
        //     pure deletes are missing there. See COMPATIBILITY.md. (A window
        //     with no data file carries none to detect encryption from, so
        //     where the table has delete files it fails at read on an
        //     encrypted catalog rather than returning wrong results.)
        let any_encrypted = {
            #[cfg(feature = "encryption")]
            {
                data_files.iter().any(|d| d.encryption_key.is_some())
            }
            #[cfg(not(feature = "encryption"))]
            {
                false
            }
        };

        // Merged partial files whose window overlap comes only from
        // `partial_max` (begin_snapshot before the window) need per-row
        // attribution, which the encrypted insert-only path below cannot do
        // (it cannot read the embedded snapshot column). Drop them there —
        // matching the pre-partial_max behavior — rather than emitting rows
        // with wrong snapshots. See COMPATIBILITY.md. When that leaves no
        // data files at all, the feed is empty (deletes are already
        // documented-missing on encrypted catalogs), not a planning error.
        if any_encrypted {
            data_files.retain(|f| f.begin_snapshot >= self.start_snapshot);
            if data_files.is_empty() && inlined.inserts.is_empty() {
                use datafusion::physical_plan::empty::EmptyExec;
                return Ok(Arc::new(EmptyExec::new(proj_info.output_schema)));
            }
            // Without the footers there are no field ids to resolve columns by,
            // and matching by name is wrong once a column has been renamed or
            // dropped and re-added.
            #[cfg(feature = "encryption")]
            self.reject_evolved_encrypted_table(&columns, &data_files)?;
        }
        let any_partial = data_files.iter().any(|f| f.partial_max.is_some());

        if (proj_info.need_rowid
            || !window_deletions.is_empty()
            || any_partial
            || !inlined.is_empty())
            && !any_encrypted
        {
            let mut layouts: Vec<Arc<ParquetFileLayout>> = Vec::with_capacity(data_files.len());
            for data_file in &data_files {
                layouts.push(
                    self.feed_files()
                        .file_layout(
                            state,
                            &columns,
                            &data_file.path,
                            data_file.path_is_relative,
                            data_file.file_size_bytes,
                            data_file.footer_size,
                            data_file.mapping_id,
                        )
                        .await?,
                );
            }
            return self
                .build_correlated_changes(
                    state,
                    &columns,
                    &data_files,
                    window_deletions,
                    &rewrites,
                    inlined,
                    &layouts,
                    projection,
                    &charge,
                )
                .await;
        }

        // Build encryption factory from file encryption keys (when encryption feature is enabled)
        #[cfg(feature = "encryption")]
        let encryption_factory: Option<Arc<dyn EncryptionFactory>> = {
            let mut builder = EncryptionFactoryBuilder::new();
            for data_file in &data_files {
                let resolved_path = resolve_path(
                    &self.table_path,
                    &data_file.path,
                    data_file.path_is_relative,
                )
                .map_err(|e| DataFusionError::External(Box::new(e)))?;
                builder.add_file(&resolved_path, data_file.encryption_key.as_deref());
            }
            let factory = builder.build();
            if factory.has_encrypted_files() {
                Some(Arc::new(factory) as Arc<dyn EncryptionFactory>)
            } else {
                None
            }
        };

        // Build execution plan for each file with projection pushdown. Each
        // file's columns are resolved by field id — except on an encrypted
        // catalog, whose footers this path cannot read (the query has already
        // been refused above if that table's columns evolved).
        let mut execs: Vec<Arc<dyn ExecutionPlan>> = Vec::with_capacity(data_files.len());
        for data_file in &data_files {
            let layout = if any_encrypted {
                None
            } else {
                Some(
                    self.feed_files()
                        .file_layout(
                            state,
                            &columns,
                            &data_file.path,
                            data_file.path_is_relative,
                            data_file.file_size_bytes,
                            data_file.footer_size,
                            data_file.mapping_id,
                        )
                        .await?,
                )
            };
            #[cfg(feature = "encryption")]
            let exec = self
                .build_exec_for_file(
                    state,
                    data_file,
                    layout.as_deref(),
                    &proj_info,
                    &encryption_factory,
                )
                .await?;
            #[cfg(not(feature = "encryption"))]
            let exec = self
                .build_exec_for_file(state, data_file, layout.as_deref(), &proj_info)
                .await?;
            execs.push(exec);
        }

        // The plan keeps only the inlined inserts of what it read.
        charge.try_resize(inlined.inserts.bytes())?;
        // Inlined inserts carry their row ids and snapshots as data; on this
        // insert-only path they surface as `insert` like every other row.
        if !inlined.inserts.is_empty() {
            let all: Vec<usize> = (0..self.table_schema.fields().len()).collect();
            let rows = label_rows(inlined.inserts.exec(&all)?, ChangeType::Insert.as_str())?;
            // The per-file branches lay their columns out as the requested CDC
            // columns in feed order, then the requested table columns.
            let layout: Vec<usize> = [
                (proj_info.need_snapshot_id, SNAPSHOT_ID_IDX),
                (proj_info.need_rowid, ROWID_IDX),
                (proj_info.need_change_type, CHANGE_TYPE_IDX),
            ]
            .into_iter()
            .filter_map(|(needed, idx)| needed.then_some(idx))
            .chain(proj_info.table_indices.iter().map(|i| i + CDC_COLS))
            .collect();
            let exprs: Vec<(Arc<dyn PhysicalExpr>, String)> = layout
                .iter()
                .map(|&i| {
                    let f = self.output_schema.field(i);
                    (
                        Arc::new(Column::new(f.name(), i)) as Arc<dyn PhysicalExpr>,
                        f.name().to_string(),
                    )
                })
                .collect();
            execs.push(Arc::new(ProjectionExec::try_new(exprs, rows)?));
        }

        // Combine with UnionExec if multiple files
        if execs.len() == 1 {
            Ok(execs.into_iter().next().unwrap())
        } else {
            UnionExec::try_new(execs)
        }
    }
}
// ---------------------------------------------------------------------------
// Correlated change feed (insert / delete / update_preimage / update_postimage)
// ---------------------------------------------------------------------------

/// One plan of `execs` unioned, `None` when there are none.
fn union_of(
    mut execs: Vec<Arc<dyn ExecutionPlan>>,
) -> DataFusionResult<Option<Arc<dyn ExecutionPlan>>> {
    match execs.len() {
        0 => Ok(None),
        1 => Ok(execs.pop()),
        _ => UnionExec::try_new(execs).map(Some),
    }
}

/// `rows` (`[snapshot_id, rowid, table columns]`) labelled `matched` where
/// `keys` (`[snapshot_id, rowid]` of the other side of the feed) holds the
/// row's `(snapshot_id, rowid)`, once per such key, and `unmatched` where it
/// holds none: `rows LEFT JOIN keys`, official's macro. With no `keys`, every
/// row is `unmatched`.
///
/// The join builds on `keys` and probes with `rows`, so only the keys are held,
/// in a hash table charged to the session's memory pool; the rows stream.
fn pair_rows(
    rows: Arc<dyn ExecutionPlan>,
    keys: Option<Arc<dyn ExecutionPlan>>,
    matched: ChangeType,
    unmatched: ChangeType,
) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
    let Some(keys) = keys else {
        return label_rows(rows, unmatched.as_str());
    };
    // A `CollectLeft` join builds on one partition. The plan this returns is
    // executed as it stands too, without the physical optimizer that would
    // otherwise add the coalesce.
    let keys: Arc<dyn ExecutionPlan> = if keys.output_partitioning().partition_count() > 1 {
        Arc::new(CoalescePartitionsExec::new(keys))
    } else {
        keys
    };
    let on: JoinOn = vec![
        (
            Arc::new(Column::new("snapshot_id", 0)) as Arc<dyn PhysicalExpr>,
            Arc::new(Column::new("snapshot_id", 0)) as Arc<dyn PhysicalExpr>,
        ),
        (
            Arc::new(Column::new("rowid", 1)) as Arc<dyn PhysicalExpr>,
            Arc::new(Column::new("rowid", 1)) as Arc<dyn PhysicalExpr>,
        ),
    ];
    let join: Arc<dyn ExecutionPlan> = Arc::new(HashJoinExec::try_new(
        keys,
        rows,
        on,
        None,
        &JoinType::Right,
        None,
        PartitionMode::CollectLeft,
        NullEquality::NullEqualsNothing,
        false,
    )?);
    // `[key snapshot_id, key rowid, snapshot_id, rowid, table columns]`: the
    // key columns are NULL exactly where the row matched no key.
    let schema = join.schema();
    let change_type: Arc<dyn PhysicalExpr> = Arc::new(CaseExpr::try_new(
        None,
        vec![(
            is_not_null(Arc::new(Column::new("snapshot_id", 0)))?,
            change_type_literal(matched.as_str()),
        )],
        Some(change_type_literal(unmatched.as_str())),
    )?);
    let mut exprs: Vec<(Arc<dyn PhysicalExpr>, String)> =
        Vec::with_capacity(schema.fields().len() - 1);
    for (i, field) in schema.fields().iter().enumerate().skip(2) {
        if i == 4 {
            exprs.push((Arc::clone(&change_type), "change_type".to_string()));
        }
        exprs.push((
            Arc::new(Column::new(field.name(), i)) as Arc<dyn PhysicalExpr>,
            field.name().to_string(),
        ));
    }
    if schema.fields().len() == 4 {
        exprs.push((change_type, "change_type".to_string()));
    }
    Ok(Arc::new(ProjectionExec::try_new(exprs, join)?))
}

/// How an inserted file's rows find their rowids.
#[derive(Debug, Clone, Copy)]
enum InsertRowid {
    /// The file's embedded rowid column, at this index of the scan batch.
    Embedded(usize),
    /// `row_id_start + position`, the position at this index of the scan batch.
    Positional {
        position: usize,
        row_id_start: Option<i64>,
    },
    /// Not resolved: the rowid is neither output nor paired on (NULL).
    Unresolved,
}

/// One inserted data file of the window and how to key its rows.
#[derive(Debug)]
struct InsertSource {
    /// Single-partition scan of the file: the projected table columns, then
    /// whatever internal columns [`InsertRowid`] and `snapshot_col` index.
    scan: Arc<dyn ExecutionPlan>,
    /// The file's `begin_snapshot`: every row's snapshot for an ordinary file;
    /// for a merged partial file (`snapshot_col` set) each row's actual
    /// snapshot comes from the embedded column.
    snapshot_id: i64,
    rowid: InsertRowid,
    /// Column index of the embedded per-row snapshot (merged partial files
    /// only). Rows outside the query window are filtered out on read.
    snapshot_col: Option<usize>,
    /// The rewrite snapshots a merged partial file's rows can carry: those
    /// rows were written by a rewrite, which inserted none of them, and are
    /// filtered out on read (official's `snapshot_filter_excluded`).
    excluded_snapshots: Vec<i64>,
}

/// Execution plan that reads the rows a window's data files inserted as
/// `[snapshot_id, rowid, projected table columns]`: each row with the rowid
/// it was inserted under (embedded, or `row_id_start + position`) and the
/// snapshot that inserted it (the file's, or a merged partial file's per-row
/// one, filtered to the window).
///
/// Each partition reads its share of the files one at a time, streaming. No
/// DataFusion children: each file's scan is internal and executed directly,
/// as one partition, so the optimizer never splits a file away from the
/// position column its rowids are synthesized from.
#[derive(Debug)]
pub struct InsertedRowsExec {
    sources: Arc<Vec<InsertSource>>,
    /// Number of projected table columns leading each scan batch.
    table_cols: usize,
    /// The query's `[start, end]` snapshot window (inclusive), used to filter
    /// rows of merged partial files by their embedded per-row snapshot.
    window: (i64, i64),
    schema: SchemaRef,
    partitions: usize,
    properties: Arc<PlanProperties>,
}

impl InsertedRowsExec {
    fn new(
        sources: Vec<InsertSource>,
        schema: SchemaRef,
        table_cols: usize,
        window: (i64, i64),
        target_partitions: usize,
    ) -> Self {
        let partitions = sources.len().min(target_partitions).max(1);
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(Arc::clone(&schema)),
            datafusion::physical_expr::Partitioning::UnknownPartitioning(partitions),
            datafusion::physical_plan::execution_plan::EmissionType::Incremental,
            datafusion::physical_plan::execution_plan::Boundedness::Bounded,
        ));
        Self {
            sources: Arc::new(sources),
            table_cols,
            window,
            schema,
            partitions,
            properties,
        }
    }
}

impl DisplayAs for InsertedRowsExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default
            | DisplayFormatType::Verbose
            | DisplayFormatType::TreeRender => {
                write!(
                    f,
                    "InsertedRowsExec: sources={}, table_columns={}",
                    self.sources.len(),
                    self.table_cols
                )
            },
        }
    }
}

impl ExecutionPlan for InsertedRowsExec {
    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> DataFusionResult<TreeNodeRecursion>,
    ) -> DataFusionResult<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn name(&self) -> &str {
        "InsertedRowsExec"
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
                "InsertedRowsExec has no children".to_string(),
            ));
        }
        Ok(self)
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        if partition >= self.partitions {
            return Err(DataFusionError::Internal(format!(
                "InsertedRowsExec has {} partitions, got {partition}",
                self.partitions
            )));
        }
        let sources = Arc::clone(&self.sources);
        let schema = Arc::clone(&self.schema);
        let table_cols = self.table_cols;
        let window = self.window;
        let indices: Vec<usize> = (partition..sources.len())
            .step_by(self.partitions)
            .collect();
        let stream = futures::stream::iter(indices)
            .map(move |index| {
                let source = &sources[index];
                let partitions = source.scan.output_partitioning().partition_count();
                if partitions != 1 {
                    return Err(DataFusionError::Internal(format!(
                        "insertions data-file scan must be a single partition covering the \
                         whole file, found {partitions}"
                    )));
                }
                // A fresh copy: a `DataSourceExec` keeps its work queue for the
                // life of the plan, and yields nothing when executed again.
                let scan = reset_plan_states(Arc::clone(&source.scan))?;
                let stream = scan.execute(0, Arc::clone(&context))?;
                let sources = Arc::clone(&sources);
                let schema = Arc::clone(&schema);
                Ok(stream.try_filter_map(move |batch| {
                    futures::future::ready(inserted_batch(
                        &sources[index],
                        &batch,
                        table_cols,
                        window,
                        &schema,
                    ))
                }))
            })
            .try_flatten();
        Ok(Box::pin(
            datafusion::physical_plan::stream::RecordBatchStreamAdapter::new(
                Arc::clone(&self.schema),
                stream,
            ),
        ))
    }
}

/// One scan batch of an inserted file as `[snapshot_id, rowid, table
/// columns]`, its rows outside the window dropped (merged partial files).
fn inserted_batch(
    source: &InsertSource,
    batch: &RecordBatch,
    table_cols: usize,
    window: (i64, i64),
    schema: &SchemaRef,
) -> DataFusionResult<Option<RecordBatch>> {
    let n = batch.num_rows();
    if n == 0 {
        return Ok(None);
    }
    let rowid: ArrayRef = match source.rowid {
        InsertRowid::Embedded(column) => {
            Arc::new(int64_column(batch, column, "embedded rowid")?.clone())
        },
        InsertRowid::Positional {
            position,
            row_id_start,
        } => {
            let row_id_start = row_id_start.ok_or_else(|| {
                DataFusionError::Internal(
                    "cannot synthesize rowid: inserted file has neither an embedded rowid nor \
                     a row_id_start"
                        .to_string(),
                )
            })?;
            let position = int64_column(batch, position, "physical position")?;
            Arc::new(position.unary::<_, arrow::datatypes::Int64Type>(|p| row_id_start + p))
        },
        InsertRowid::Unresolved => arrow::array::new_null_array(&DataType::Int64, n),
    };
    let snapshot: ArrayRef = match source.snapshot_col {
        Some(column) => {
            let snapshots = int64_column(batch, column, "embedded snapshot_id")?;
            if snapshots.null_count() > 0 {
                return Err(DataFusionError::Internal(
                    "embedded snapshot_id column contains NULL".to_string(),
                ));
            }
            Arc::new(snapshots.clone())
        },
        None => Arc::new(Int64Array::from_value(source.snapshot_id, n)),
    };
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(table_cols + 2);
    columns.push(Arc::clone(&snapshot));
    columns.push(rowid);
    columns.extend(batch.columns().iter().take(table_cols).cloned());
    let rows = RecordBatch::try_new_with_options(
        Arc::clone(schema),
        columns,
        &arrow::record_batch::RecordBatchOptions::new().with_row_count(Some(n)),
    )?;
    if source.snapshot_col.is_none() {
        return Ok(Some(rows));
    }
    // Merged partial file: each row belongs to its embedded origin snapshot;
    // keep the rows inside the query window that a rewrite did not write.
    let snapshot = snapshot
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("the snapshot column is Int64");
    let excluded = &source.excluded_snapshots;
    let in_window: arrow::array::BooleanArray = snapshot
        .iter()
        .map(|s| s.map(|s| s >= window.0 && s <= window.1 && !excluded.contains(&s)))
        .collect();
    let rows = arrow::compute::filter_record_batch(&rows, &in_window)?;
    Ok((rows.num_rows() > 0).then_some(rows))
}

/// Downcast a batch column to `Int64Array` with a descriptive error.
fn int64_column<'a>(
    batch: &'a RecordBatch,
    idx: usize,
    what: &str,
) -> DataFusionResult<&'a Int64Array> {
    batch
        .column(idx)
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| DataFusionError::Internal(format!("{what} column is not Int64")))
}

/// The inlined row versions one change-feed window reads, charged to the
/// plan's [`PlanCharge`] as each is read.
pub(crate) struct InlinedChanges {
    /// Inlined row versions written in the window.
    pub(crate) inserts: KeyedInlinedRows,
    /// Inlined row versions deleted or superseded in the window.
    pub(crate) deletes: KeyedInlinedRows,
    /// Inlined deletions of Parquet-resident rows made in the window.
    pub(crate) file_deletes: Vec<InlinedFileDeleteChange>,
}

impl InlinedChanges {
    /// Read the window's inlined changes; the delete side only when
    /// `read_deletes`.
    pub(crate) fn read(
        provider: &dyn MetadataProvider,
        table_id: i64,
        columns: &[DuckLakeTableColumn],
        table_schema: &SchemaRef,
        (start, end): (i64, i64),
        read_deletes: bool,
        charge: &PlanCharge,
    ) -> DataFusionResult<Self> {
        let inserts = Self::read_rows(
            provider,
            table_id,
            columns,
            table_schema,
            InlinedRowSelection::InsertedBetween {
                start,
                end,
            },
            charge,
        )?;
        let (deletes, file_deletes) = if read_deletes {
            (
                Self::read_deletes(
                    provider,
                    table_id,
                    columns,
                    table_schema,
                    (start, end),
                    charge,
                )?,
                Self::read_file_deletes(provider, table_id, (start, end), charge)?,
            )
        } else {
            (KeyedInlinedRows::empty(charge), Vec::new())
        };
        Ok(Self {
            inserts,
            deletes,
            file_deletes,
        })
    }

    /// Inlined row versions deleted or superseded in the window.
    pub(crate) fn read_deletes(
        provider: &dyn MetadataProvider,
        table_id: i64,
        columns: &[DuckLakeTableColumn],
        table_schema: &SchemaRef,
        (start, end): (i64, i64),
        charge: &PlanCharge,
    ) -> DataFusionResult<KeyedInlinedRows> {
        Self::read_rows(
            provider,
            table_id,
            columns,
            table_schema,
            InlinedRowSelection::DeletedBetween {
                start,
                end,
            },
            charge,
        )
    }

    /// Inlined deletions of Parquet-resident rows made in the inclusive window.
    pub(crate) fn read_file_deletes(
        provider: &dyn MetadataProvider,
        table_id: i64,
        (start, end): (i64, i64),
        charge: &PlanCharge,
    ) -> DataFusionResult<Vec<InlinedFileDeleteChange>> {
        let changes = provider
            .get_inlined_file_deletes_between(table_id, start, end)
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        charge.try_grow(file_deletes_bytes(&changes))?;
        Ok(changes)
    }

    /// [`Self::read_file_deletes`], of the data files `data_file_ids` only.
    pub(crate) fn read_file_deletes_of_files(
        provider: &dyn MetadataProvider,
        table_id: i64,
        data_file_ids: &[i64],
        (start, end): (i64, i64),
        charge: &PlanCharge,
    ) -> DataFusionResult<Vec<InlinedFileDeleteChange>> {
        let changes = provider
            .get_inlined_file_deletes_of_files_between(table_id, data_file_ids, start, end)
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        charge.try_grow(file_deletes_bytes(&changes))?;
        Ok(changes)
    }

    /// Read the inlined row versions `selection` names, charge them, and key
    /// them; the charge then follows what the keyed rows hold.
    fn read_rows(
        provider: &dyn MetadataProvider,
        table_id: i64,
        columns: &[DuckLakeTableColumn],
        table_schema: &SchemaRef,
        selection: InlinedRowSelection,
        charge: &PlanCharge,
    ) -> DataFusionResult<KeyedInlinedRows> {
        let rows = provider
            .read_inlined_rows(table_id, columns, selection, None)
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        let read = inlined_rows_bytes(&rows);
        charge.try_grow(read)?;
        let keyed = KeyedInlinedRows::new(rows, table_schema, charge)?;
        release(charge, read.saturating_sub(keyed.bytes()));
        Ok(keyed)
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.inserts.is_empty() && self.deletes.is_empty() && self.file_deletes.is_empty()
    }
}

/// The snapshots of the inclusive window that rewrote data files of
/// `table_id` (`rewrite_delete:<table_id>` in the snapshot change ledger), as
/// official DuckLake's `GetTableRewriteSnapshots` reads them (upstream main
/// after the extension this crate pins). A rewrite changes no row, so the
/// change feeds report nothing for it: not the data files it ends, not the
/// files it writes, and not the rows of its outputs a merge later carries.
/// A provider without a change ledger records no rewrite.
pub(crate) fn rewrite_snapshots(
    provider: &dyn MetadataProvider,
    table_id: i64,
    (start, end): (i64, i64),
) -> DataFusionResult<BTreeSet<i64>> {
    if start > end {
        return Ok(BTreeSet::new());
    }
    let changes = match provider.list_table_snapshot_changes(table_id, start.saturating_sub(1), end)
    {
        Ok(changes) => changes,
        Err(crate::DuckLakeError::Unsupported(_)) => return Ok(BTreeSet::new()),
        Err(e) => return Err(DataFusionError::External(Box::new(e))),
    };
    Ok(changes
        .into_iter()
        .filter(|(_, made)| {
            made.as_deref().is_some_and(|made| {
                snapshot_change_tokens(made).any(|token| {
                    token.split_once(':').is_some_and(|(kind, id)| {
                        kind.trim().eq_ignore_ascii_case("rewrite_delete")
                            && id.trim().parse::<i64>() == Ok(table_id)
                    })
                })
            })
        })
        .map(|(snapshot, _)| snapshot)
        .collect())
}

/// Whether a data file of the window was written by a rewrite: an ordinary
/// file that begins at a rewrite snapshot holds rows inserted before it.
pub(crate) fn written_by_rewrite(file: &DataFileChange, rewrites: &BTreeSet<i64>) -> bool {
    file.partial_max.is_none() && rewrites.contains(&file.begin_snapshot)
}

/// One data file's inlined deletions in a window.
pub(crate) struct InlinedFileDeletesOfFile {
    /// The first deletion read for the file, dated at the file's earliest
    /// one; its data-file fields describe it.
    pub(crate) change: InlinedFileDeleteChange,
    /// `(deleted position, deleting snapshot)`, sorted by position.
    pub(crate) positions: Vec<(i64, i64)>,
}

impl InlinedFileDeletesOfFile {
    fn bytes(&self) -> usize {
        self.change.data_file_path.capacity() + self.positions.capacity() * size_of::<(i64, i64)>()
    }
}

/// A window's inlined deletions of Parquet rows, grouped by data file and
/// handed out to the delete records that read each file.
///
/// Official DuckLake attaches a file's inlined deletions to the delete records
/// the window holds for it, and reads an inlined-only file on its own. A data
/// file retired in the window takes its inlined deletions with its retirement
/// alone, which reports them at their own snapshots and every other row of the
/// file at the retirement's (official's `GetTableDeletions`, upstream main
/// after the extension this crate pins). The same file reached by several
/// delete records and not retired takes its inlined deletions once, with the
/// first.
///
/// Grouped by sorting the window's deletions in place, so each file's
/// positions are allocated once at their exact size and charged to the plan's
/// [`PlanCharge`] before they are.
pub(crate) struct InlinedFileDeletes {
    /// Resolved path and deletions of each file, in data-file order; `None`
    /// once a delete record claimed them.
    files: Vec<(String, Option<InlinedFileDeletesOfFile>)>,
    /// Indices into `files`, sorted by path.
    by_path: Vec<usize>,
    /// The data files a delete record of the window retires.
    retired: HashSet<String>,
}

impl InlinedFileDeletes {
    pub(crate) fn group(
        table_path: &str,
        mut changes: Vec<InlinedFileDeleteChange>,
        retired: HashSet<String>,
        charge: &PlanCharge,
    ) -> DataFusionResult<Self> {
        type File = (String, Option<InlinedFileDeletesOfFile>);
        changes.sort_unstable_by_key(|c| (c.data_file_id, c.position, c.snapshot_id));
        let run_len = |start: usize| {
            changes[start..]
                .iter()
                .take_while(|c| c.data_file_id == changes[start].data_file_id)
                .count()
        };
        // Reserved before anything is allocated: each file's slot and sorted
        // index, its positions, and at most a resolved path (the table path, a
        // separator and its own) and a copy of its own.
        let mut file_count = 0;
        let mut estimate = 0;
        let mut start = 0;
        while start < changes.len() {
            let len = run_len(start);
            estimate += size_of::<File>()
                + size_of::<usize>()
                + len * size_of::<(i64, i64)>()
                + table_path.len()
                + 1
                + 2 * changes[start].data_file_path.len();
            start += len;
            file_count += 1;
        }
        charge.try_grow(estimate)?;
        let mut files: Vec<File> = Vec::with_capacity(file_count);
        let mut start = 0;
        while start < changes.len() {
            let run = &changes[start..start + run_len(start)];
            start += run.len();
            let first = &run[0];
            let resolved = resolve_path(
                table_path,
                &first.data_file_path,
                first.data_file_path_is_relative,
            )
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
            let mut positions: Vec<(i64, i64)> =
                run.iter().map(|c| (c.position, c.snapshot_id)).collect();
            // One per position, the latest deletion kept, as a map keyed by
            // position would hold them.
            positions.dedup_by(|later, earlier| {
                let same = later.0 == earlier.0;
                if same {
                    earlier.1 = later.1;
                }
                same
            });
            let mut change = first.clone();
            change.snapshot_id = run
                .iter()
                .map(|c| c.snapshot_id)
                .min()
                .unwrap_or(change.snapshot_id);
            files.push((
                resolved,
                Some(InlinedFileDeletesOfFile {
                    change,
                    positions,
                }),
            ));
        }
        let mut by_path: Vec<usize> = (0..files.len()).collect();
        by_path.sort_unstable_by(|&a, &b| files[a].0.cmp(&files[b].0));
        let grouped = Self {
            files,
            by_path,
            retired,
        };
        // The estimate is an upper bound; the charge follows what is held,
        // and the read's deletions, grouped now, are released.
        let held = grouped.files.capacity() * size_of::<File>()
            + grouped.by_path.capacity() * size_of::<usize>()
            + grouped.held_by_files();
        let read = file_deletes_bytes(&changes);
        drop(changes);
        if held > estimate {
            charge.try_grow(held - estimate)?;
        }
        release(charge, estimate.saturating_sub(held) + read);
        Ok(grouped)
    }

    fn held_by_files(&self) -> usize {
        self.files
            .iter()
            .map(|(path, file)| path.capacity() + file.as_ref().map_or(0, |f| f.bytes()))
            .sum()
    }

    /// The inlined deletions a delete record of the file at `resolved` reads:
    /// all of them for the file's retirement (`retires_file`), none for its
    /// other records when it is retired, and otherwise all of them for its
    /// first record.
    pub(crate) fn claim(&mut self, resolved: &str, retires_file: bool) -> Vec<(i64, i64)> {
        let Ok(at) = self
            .by_path
            .binary_search_by(|&i| self.files[i].0.as_str().cmp(resolved))
        else {
            return Vec::new();
        };
        if !retires_file && self.retired.contains(resolved) {
            return Vec::new();
        }
        let index = self.by_path[at];
        self.files[index]
            .1
            .take()
            .map(|file| file.positions)
            .unwrap_or_default()
    }

    /// The files no delete record claimed, in data-file order.
    pub(crate) fn unclaimed(self) -> Vec<(String, InlinedFileDeletesOfFile)> {
        self.files
            .into_iter()
            .filter_map(|(resolved, file)| file.map(|file| (resolved, file)))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// ducklake_table_insertions
// ---------------------------------------------------------------------------

/// `ducklake_table_insertions`: every row added in the snapshot window —
/// plain inserts, UPDATE postimages, and in-window rows of compaction-merged
/// partial files — with `(snapshot_id, rowid)` leading and NO `change_type`
/// column, matching official DuckLake's insertions feed surface (which has
/// none; it exposes rowid/snapshot_id as virtual columns).
///
/// A thin wrapper over the [`TableChangesTable`] machinery with the delete
/// side disabled: projections are translated to skip the inner feed's
/// `change_type` column.
#[derive(Debug)]
pub struct TableInsertionsTable {
    inner: TableChangesTable,
    output_schema: SchemaRef,
}

impl TableInsertionsTable {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        provider: Arc<dyn MetadataProvider>,
        table_id: i64,
        start_snapshot: i64,
        end_snapshot: i64,
        object_store_url: Arc<ObjectStoreUrl>,
        table_path: String,
        table_schema: SchemaRef,
    ) -> Self {
        let mut fields: Vec<Field> = Vec::with_capacity(table_schema.fields().len() + 2);
        fields.push(Field::new("snapshot_id", DataType::Int64, false));
        fields.push(Field::new("rowid", DataType::Int64, true));
        fields.extend(table_schema.fields().iter().map(|f| f.as_ref().clone()));
        let output_schema = Arc::new(Schema::new(fields));
        let inner = TableChangesTable::new(
            provider,
            table_id,
            start_snapshot,
            end_snapshot,
            object_store_url,
            table_path,
            table_schema,
        )
        .insertions_only();
        Self {
            inner,
            output_schema,
        }
    }

    /// Supply the table's columns as of the window's end snapshot; see
    /// [`TableChangesTable::with_columns`].
    pub fn with_columns(mut self, columns: Vec<DuckLakeTableColumn>) -> Self {
        self.inner = self.inner.with_columns(columns);
        self
    }

    /// Map an index of this feed's schema — `(snapshot_id, rowid, table
    /// columns...)` — onto the inner changes schema, which has `change_type`
    /// at [`CHANGE_TYPE_IDX`].
    fn inner_index(outer: usize) -> usize {
        if outer < CHANGE_TYPE_IDX {
            outer
        } else {
            outer + 1
        }
    }
}

#[async_trait]
impl TableProvider for TableInsertionsTable {
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
        filters: &[datafusion::prelude::Expr],
        limit: Option<usize>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        // The inner feed honors arbitrary projections in requested order, so a
        // translated projection yields exactly this feed's columns.
        let translated: Vec<usize> = match projection {
            Some(indices) => indices.iter().map(|&i| Self::inner_index(i)).collect(),
            None => (0..self.output_schema.fields().len())
                .map(Self::inner_index)
                .collect(),
        };
        self.inner
            .scan(state, Some(&translated), filters, limit)
            .await
    }
}
