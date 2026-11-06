//! Table deletions functionality for DuckLake
//!
//! This module provides the `ducklake_table_deletions()` table function that returns
//! the actual deleted rows between snapshots, with CDC metadata columns, and the
//! deletion side of `ducklake_table_changes`.
//!
//! For each data file with deletions:
//! 1. Read positions from current delete file (or all positions for full file delete)
//! 2. Subtract positions from previous delete file (if exists)
//! 3. Read the data file and return only the rows at the newly deleted positions
//! 4. Prepend `(snapshot_id, rowid)`
//!
//! [`DeletedRowsExec`] reads one data file at a time per partition: the file's
//! delete positions are read whole (the set must be complete before any data
//! row can be classified), charged to the session's memory pool, and the data
//! file is then streamed batch-by-batch through them. Official DuckLake's
//! delete filter holds one file's delete data the same way. Its per-file scans
//! are internal and executed directly, never exposed as children: deleted rows
//! are matched by TRUE physical file position — the column the parquet reader
//! produces — and exposing the scans would let the optimizer split them across
//! partitions away from the position set (issue #178).

use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::{Arc, Mutex};

use arrow::array::{Array, ArrayRef, Int64Array, RecordBatch, UInt32Array};
use arrow::compute::take;
use arrow::datatypes::{DataType, Field, FieldRef, Schema, SchemaRef};
use async_trait::async_trait;
use datafusion::catalog::Session;
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::common::{Result as DataFusionResult, ScalarValue};
use datafusion::datasource::listing::PartitionedFile;
use datafusion::datasource::physical_plan::{FileGroup, FileScanConfigBuilder};
use datafusion::datasource::source::DataSourceExec;
use datafusion::datasource::{TableProvider, TableType};
use datafusion::error::DataFusionError;
use datafusion::execution::memory_pool::{MemoryConsumer, MemoryReservation};
use datafusion::execution::object_store::ObjectStoreUrl;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::expressions::{Column, Literal};
use datafusion::physical_expr::{EquivalenceProperties, PhysicalExpr};
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType, reset_plan_states};
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::union::UnionExec;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, PlanProperties,
};
use futures::stream::BoxStream;
use futures::{StreamExt, TryStreamExt};

use crate::feed_charge::{PlanCharge, plan_charge};
use crate::metadata_provider::{
    DeleteFileChange, DuckLakeTableColumn, InlinedFileDeleteChange, MetadataProvider,
};
use crate::path_resolver::resolve_path;
use crate::row_id::{SNAPSHOT_ID_PARQUET_FIELD_ID, positional_table_schema_reserving};
use crate::table::{
    ParquetFileLayout, apply_name_mapping_to_layout, cached_parquet_reader_factory,
    delete_file_schema, metadata_size_hint, read_parquet_file_layout, read_parquet_footer_facts,
    session_parquet_source, validated_file_size, validated_record_count,
};
use crate::table_changes::{
    InlinedChanges, InlinedFileDeletes, InlinedFileDeletesOfFile, check_column_count,
    present_catalog_schema, rewrite_snapshots, row_count_probe_index,
};
use crate::types::DuckLakeDefaultExprAdapterFactory;

/// The schema every row source of a change feed emits: `snapshot_id`, `rowid`,
/// then the projected table columns. `rowid` is nullable: it is NULL where the
/// caller did not ask for it and nothing pairs on it.
pub(crate) fn keyed_rows_schema(table_schema: &SchemaRef, table_indices: &[usize]) -> SchemaRef {
    let mut fields: Vec<FieldRef> = Vec::with_capacity(table_indices.len() + 2);
    fields.push(Arc::new(Field::new("snapshot_id", DataType::Int64, false)));
    fields.push(Arc::new(Field::new("rowid", DataType::Int64, true)));
    fields.extend(
        table_indices
            .iter()
            .map(|&i| Arc::clone(&table_schema.fields()[i])),
    );
    Arc::new(Schema::new(fields))
}

/// The table columns a feed projection names, in table order, and whether it
/// names `rowid`. The feed's columns are `[snapshot_id, rowid, change_type?,
/// table columns...]`, `cdc_cols` being the number before the table columns.
pub(crate) fn projected_table_indices(
    projection: Option<&Vec<usize>>,
    cdc_cols: usize,
    table_len: usize,
) -> (Vec<usize>, bool) {
    match projection {
        None => ((0..table_len).collect(), true),
        Some(indices) => {
            let mut table: Vec<usize> = indices
                .iter()
                .filter(|&&i| i >= cdc_cols && i < cdc_cols + table_len)
                .map(|&i| i - cdc_cols)
                .collect();
            table.sort_unstable();
            table.dedup();
            (table, indices.contains(&1))
        },
    }
}

/// Present `[snapshot_id, rowid, change_type, projected table columns]` as the
/// feed's projection: `output_schema` is the feed's full schema (CDC columns
/// first, `cdc_cols` of them) and `table_indices` the table columns `input`
/// carries.
pub(crate) fn project_feed(
    input: Arc<dyn ExecutionPlan>,
    projection: Option<&Vec<usize>>,
    output_schema: &SchemaRef,
    cdc_cols: usize,
    table_indices: &[usize],
) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
    let all: Vec<usize>;
    let indices = match projection {
        Some(indices) => indices,
        None => {
            all = (0..output_schema.fields().len()).collect();
            &all
        },
    };
    let exprs = indices
        .iter()
        .map(|&i| {
            let field = output_schema.field(i);
            let input_index = if i < cdc_cols {
                i
            } else {
                let table_index = i - cdc_cols;
                let at = table_indices
                    .iter()
                    .position(|&t| t == table_index)
                    .ok_or_else(|| {
                        DataFusionError::Internal(format!(
                            "change feed projection names table column {table_index}, which \
                             its plan does not carry"
                        ))
                    })?;
                cdc_cols + at
            };
            Ok((
                Arc::new(Column::new(field.name(), input_index)) as Arc<dyn PhysicalExpr>,
                field.name().to_string(),
            ))
        })
        .collect::<DataFusionResult<Vec<_>>>()?;
    Ok(Arc::new(ProjectionExec::try_new(exprs, input)?))
}

/// `[snapshot_id, rowid, table columns]` with a constant `change_type` placed
/// after `rowid`.
pub(crate) fn label_rows(
    input: Arc<dyn ExecutionPlan>,
    change_type: &str,
) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
    let schema = input.schema();
    let mut exprs: Vec<(Arc<dyn PhysicalExpr>, String)> =
        Vec::with_capacity(schema.fields().len() + 1);
    for (i, field) in schema.fields().iter().enumerate() {
        if i == 2 {
            exprs.push((change_type_literal(change_type), "change_type".to_string()));
        }
        exprs.push((
            Arc::new(Column::new(field.name(), i)) as Arc<dyn PhysicalExpr>,
            field.name().to_string(),
        ));
    }
    if schema.fields().len() == 2 {
        exprs.push((change_type_literal(change_type), "change_type".to_string()));
    }
    Ok(Arc::new(ProjectionExec::try_new(exprs, input)?))
}

/// A `change_type` value as a physical literal.
pub(crate) fn change_type_literal(change_type: &str) -> Arc<dyn PhysicalExpr> {
    Arc::new(Literal::new(ScalarValue::Utf8(Some(
        change_type.to_string(),
    ))))
}

/// The delete records a change feed over a window reads, selected as official
/// DuckLake's `GetTableDeletions` selects them (upstream main after the
/// extension this crate pins).
#[derive(Debug, Default)]
pub(crate) struct WindowDeletions {
    /// Every delete file with `begin_snapshot <= end`, and every data file
    /// retired in the window but by a rewrite, which ends data files without
    /// deleting their rows.
    pub(crate) delete_files: Vec<DeleteFileChange>,
    /// The inlined deletions made before the window of the data files the
    /// window retires: the rows they deleted are not deleted again by the
    /// retirement. Sorted by data file and position, and charged to the plan.
    pub(crate) earlier_inlined: Vec<InlinedFileDeleteChange>,
}

impl WindowDeletions {
    /// Read the window's delete records of `table_id`, given the window's
    /// rewrite snapshots ([`rewrite_snapshots`](crate::table_changes::rewrite_snapshots)).
    pub(crate) fn read(
        provider: &dyn MetadataProvider,
        table_id: i64,
        (start, end): (i64, i64),
        rewrites: &BTreeSet<i64>,
        charge: &PlanCharge,
    ) -> DataFusionResult<Self> {
        let external = |e| DataFusionError::External(Box::new(e));
        let mut delete_files = provider
            .get_delete_files_added_between_snapshots(table_id, start, end)
            .map_err(external)?;
        delete_files.retain(|dfc| {
            dfc.current_delete_path.is_some() || !rewrites.contains(&dfc.snapshot_id)
        });
        // As official reads them: of the retired files alone (`file_id IN`).
        let mut retired: Vec<i64> = delete_files
            .iter()
            .filter(|dfc| dfc.current_delete_path.is_none())
            .map(|dfc| dfc.data_file_id)
            .collect();
        retired.sort_unstable();
        retired.dedup();
        let earlier_inlined = if !retired.is_empty() && start > 0 {
            let mut earlier = InlinedChanges::read_file_deletes_of_files(
                provider,
                table_id,
                &retired,
                (0, start - 1),
                charge,
            )?;
            earlier.sort_unstable_by_key(|c| (c.data_file_id, c.position));
            earlier
        } else {
            Vec::new()
        };
        Ok(Self {
            delete_files,
            earlier_inlined,
        })
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.delete_files.is_empty()
    }
}

/// What the change feeds need to plan their per-file reads: the table's place
/// in the store, its schema, and the read layouts already resolved.
pub(crate) struct FeedFiles<'a> {
    pub(crate) provider: &'a dyn MetadataProvider,
    pub(crate) table_id: i64,
    pub(crate) start_snapshot: i64,
    pub(crate) end_snapshot: i64,
    pub(crate) object_store_url: &'a ObjectStoreUrl,
    pub(crate) table_path: &'a str,
    pub(crate) table_schema: &'a SchemaRef,
    /// Per-file read layout, memoized by resolved path: a file inserted and
    /// then deleted inside the window is reached from both sides of the feed,
    /// and several delete records can share a source data file.
    pub(crate) layout_cache: &'a Mutex<HashMap<String, Arc<ParquetFileLayout>>>,
}

impl FeedFiles<'_> {
    pub(crate) fn resolve(&self, path: &str, is_relative: bool) -> DataFusionResult<String> {
        resolve_path(self.table_path, path, is_relative)
            .map_err(|e| DataFusionError::External(Box::new(e)))
    }

    /// The read layout of one data file, memoized by resolved path.
    ///
    /// `size_bytes` and `footer_size` are the catalog's recorded values for the
    /// file, so a cold footer read needs no `head()` and no length probe.
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn file_layout(
        &self,
        state: &dyn Session,
        columns: &[DuckLakeTableColumn],
        path: &str,
        is_relative: bool,
        size_bytes: i64,
        footer_size: Option<i64>,
        mapping_id: Option<i64>,
    ) -> DataFusionResult<Arc<ParquetFileLayout>> {
        let resolved = self.resolve(path, is_relative)?;
        {
            let cache = self.layout_cache.lock().unwrap();
            if let Some(layout) = cache.get(&resolved) {
                return Ok(Arc::clone(layout));
            }
        }
        // No encryption key: the feeds that read layouts never reach an
        // encrypted file, and a footer they cannot decrypt is not readable.
        let layout = read_parquet_file_layout(
            state,
            self.object_store_url,
            &resolved,
            Some(size_bytes),
            footer_size,
            None,
            columns,
            self.table_schema,
        )
        .await?;
        let layout = apply_name_mapping_to_layout(
            self.provider,
            self.table_id,
            self.end_snapshot,
            columns,
            &resolved,
            mapping_id,
            layout,
        )?;
        self.layout_cache
            .lock()
            .unwrap()
            .entry(resolved)
            .or_insert_with(|| Arc::clone(&layout));
        Ok(layout)
    }

    /// Read a DELETE file's footer and return the physical name of its embedded
    /// per-row snapshot column ([`SNAPSHOT_ID_PARQUET_FIELD_ID`]) when present.
    /// Current-spec delete files are cumulative and carry one; each row's value
    /// is the snapshot at which that position was deleted.
    ///
    /// A delete file holds `(file_path, pos[, snapshot])`, none of it table
    /// columns, so it needs the raw footer rather than a read layout.
    async fn delete_file_snapshot_name(
        &self,
        state: &dyn Session,
        resolved: &str,
        size_bytes: Option<i64>,
        footer_size: Option<i64>,
    ) -> DataFusionResult<Option<String>> {
        let facts = read_parquet_footer_facts(
            state,
            self.object_store_url,
            resolved,
            size_bytes,
            footer_size,
            None,
        )
        .await?;
        Ok(facts.field_ids.get(&SNAPSHOT_ID_PARQUET_FIELD_ID).cloned())
    }

    /// The deletions of the window, one [`DeletionSource`] per delete record
    /// and one per data file whose only deletions in the window are inlined.
    ///
    /// Each data file's inlined deletions go to the first delete record of the
    /// file (see [`InlinedFileDeletes`]).
    ///
    /// The positions the sources hold are charged to `charge`, and the
    /// window's inlined deletions, consumed here, are released from it.
    pub(crate) async fn deletion_sources(
        &self,
        state: &dyn Session,
        columns: &[DuckLakeTableColumn],
        window_deletions: &WindowDeletions,
        inlined_file_deletes: Vec<InlinedFileDeleteChange>,
        charge: &PlanCharge,
    ) -> DataFusionResult<Vec<DeletionSource>> {
        // Both come from the same column list; the index arithmetic of the
        // scans assumes they agree, and a disagreement would misalign every
        // scan batch silently — see [`check_column_count`].
        check_column_count(self.table_schema.fields().len(), columns.len())?;
        let delete_files = &window_deletions.delete_files;
        let retired = delete_files
            .iter()
            .filter(|dfc| dfc.current_delete_path.is_none())
            .map(|dfc| self.resolve(&dfc.data_file_path, dfc.data_file_path_is_relative))
            .collect::<DataFusionResult<HashSet<_>>>()?;
        // The rows of a retired file deleted inlined before the window, which
        // its retirement does not delete again: each retired file's run of the
        // sorted deletions, by path.
        let mut earlier: Vec<(String, std::ops::Range<usize>)> = Vec::new();
        let earlier_changes = &window_deletions.earlier_inlined;
        let mut start = 0;
        while start < earlier_changes.len() {
            let first = &earlier_changes[start];
            let len = earlier_changes[start..]
                .iter()
                .take_while(|c| c.data_file_id == first.data_file_id)
                .count();
            let path = self.resolve(&first.data_file_path, first.data_file_path_is_relative)?;
            if retired.contains(&path) {
                earlier.push((path, start..start + len));
            }
            start += len;
        }
        earlier.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        let mut inlined =
            InlinedFileDeletes::group(self.table_path, inlined_file_deletes, retired, charge)?;
        let mut sources = Vec::with_capacity(delete_files.len());
        for dfc in delete_files {
            // Validate record_count before use — a negative value from corrupt
            // metadata would cause incorrect behavior (e.g., empty ranges in
            // full-file deletes).
            validated_record_count(dfc.data_record_count, &dfc.data_file_path)?;
            let data_path = self.resolve(&dfc.data_file_path, dfc.data_file_path_is_relative)?;
            let full_file = dfc.current_delete_path.is_none();
            let inlined_positions = inlined.claim(&data_path, full_file);
            let earlier_inlined =
                match earlier.binary_search_by(|(p, _)| p.as_str().cmp(&data_path)) {
                    Ok(at) if full_file => {
                        let run = std::mem::take(&mut earlier[at].1);
                        charge.try_grow(run.len() * size_of::<i64>())?;
                        earlier_changes[run].iter().map(|c| c.position).collect()
                    },
                    _ => Vec::new(),
                };
            // A cumulative (current-spec) delete file embeds each row's delete
            // snapshot; deletions are then windowed PER ROW on that column, and
            // no previous-file subtraction is needed (pre-window deletions are
            // simply outside the window). Legacy 2-column delete files keep the
            // delta-vs-previous model, one snapshot per file.
            let current = match &dfc.current_delete_path {
                Some(p) => {
                    let resolved =
                        self.resolve(p, dfc.current_delete_path_is_relative.unwrap_or(true))?;
                    let snapshot_name = self
                        .delete_file_snapshot_name(
                            state,
                            &resolved,
                            dfc.current_delete_file_size_bytes,
                            dfc.current_delete_footer_size,
                        )
                        .await?;
                    Some(DeleteFileRef {
                        path: resolved,
                        size_bytes: dfc.current_delete_file_size_bytes.unwrap_or(0),
                        footer_size: dfc.current_delete_footer_size.unwrap_or(0),
                        snapshot_name,
                    })
                },
                None => None,
            };
            let cumulative = current.as_ref().is_some_and(|c| c.snapshot_name.is_some());
            // A delete file without per-row snapshots dates every deletion it
            // adds at its own begin_snapshot, so one beginning before the window
            // adds none in it (official subtracts the latest delete file before
            // the window, which holds them all); only the data file's in-window
            // inlined deletions remain.
            let before_window = !cumulative && dfc.snapshot_id < self.start_snapshot;
            let current = if before_window {
                None
            } else {
                current
            };
            let previous = match &dfc.previous_delete_path {
                Some(p) if !cumulative && !before_window => Some(DeleteFileRef {
                    path: self.resolve(p, dfc.previous_delete_path_is_relative.unwrap_or(true))?,
                    size_bytes: dfc.previous_delete_file_size_bytes.unwrap_or(0),
                    footer_size: dfc.previous_delete_footer_size.unwrap_or(0),
                    snapshot_name: None,
                }),
                _ => None,
            };
            // The source data file can predate the window (and any rename in
            // it), so its columns need resolving by field id too.
            let layout = self
                .file_layout(
                    state,
                    columns,
                    &dfc.data_file_path,
                    dfc.data_file_path_is_relative,
                    dfc.data_file_size_bytes,
                    dfc.data_file_footer_size,
                    dfc.data_mapping_id,
                )
                .await?;
            sources.push(DeletionSource {
                snapshot_id: dfc.snapshot_id,
                data_path,
                data_size_bytes: dfc.data_file_size_bytes,
                data_footer_size: dfc.data_file_footer_size.unwrap_or(0),
                layout,
                current,
                previous,
                cumulative,
                full_file,
                inlined_positions,
                earlier_inlined,
                record_count: dfc.data_record_count,
                row_id_start: dfc.data_row_id_start,
            });
        }
        // Data files whose only deletions in the window are inlined.
        for (data_path, file) in inlined.unclaimed() {
            let InlinedFileDeletesOfFile {
                change,
                positions,
            } = file;
            validated_record_count(change.data_record_count, &change.data_file_path)?;
            let layout = self
                .file_layout(
                    state,
                    columns,
                    &change.data_file_path,
                    change.data_file_path_is_relative,
                    change.data_file_size_bytes,
                    change.data_file_footer_size,
                    change.data_mapping_id,
                )
                .await?;
            sources.push(DeletionSource {
                snapshot_id: change.snapshot_id,
                data_path,
                data_size_bytes: change.data_file_size_bytes,
                data_footer_size: change.data_file_footer_size.unwrap_or(0),
                layout,
                current: None,
                previous: None,
                cumulative: false,
                full_file: false,
                inlined_positions: positions,
                earlier_inlined: Vec::new(),
                record_count: change.data_record_count,
                row_id_start: change.data_row_id_start,
            });
        }
        Ok(sources)
    }

    /// Positional scan of a delete's source data file, projected to the table
    /// columns at `table_indices` (presented under their catalog names), then
    /// a row-count probe when nothing else is read, the embedded rowid column
    /// when `embedded_name` is `Some`, and the reader-produced physical-position
    /// column, last. Returns the scan and the indices of the embedded rowid and
    /// the position in its output.
    ///
    /// Positions come from the parquet reader, so deleted rows are matched to
    /// the delete file's `pos` set regardless of how the file is read — pruned,
    /// filtered, split or reordered (issue #178).
    fn data_file_scan(
        &self,
        state: &dyn Session,
        source: &DeletionSource,
        table_indices: &[usize],
        embedded_name: &Option<String>,
    ) -> DataFusionResult<(Arc<dyn ExecutionPlan>, Option<usize>, usize)> {
        let layout = &source.layout;
        let reader_factory = cached_parquet_reader_factory(state, self.object_store_url)?;
        let mut pf = PartitionedFile::new(
            &source.data_path,
            validated_file_size(source.data_size_bytes, &source.data_path)?,
        );
        if let Some(hint) = metadata_size_hint(Some(source.data_footer_size)) {
            pf = pf.with_metadata_size_hint(hint);
        }
        let table_len = layout.read_schema.fields().len();
        let read_schema = match embedded_name {
            Some(name) => {
                let mut fields: Vec<FieldRef> =
                    layout.read_schema.fields().iter().cloned().collect();
                fields.push(Arc::new(Field::new(name, DataType::Int64, true)));
                Arc::new(Schema::new(fields))
            },
            None => Arc::clone(&layout.read_schema),
        };
        // The scan is re-presented under the catalog names below, and
        // `ColumnRenameExec` binds by name, so the position column's name must
        // clear those too — not only the file's physical ones.
        let (scan_schema, pos_index, _) = positional_table_schema_reserving(
            read_schema,
            self.table_schema.fields().iter().map(|f| f.name().as_str()),
        );
        let mut projection = table_indices.to_vec();
        if table_indices.is_empty() && embedded_name.is_none() {
            // A data column for the row count: the position alone is a
            // virtual column, and the reader needs a real one to count.
            projection.push(row_count_probe_index(&layout.read_schema));
        }
        let embedded_col_idx = embedded_name.as_ref().map(|_| {
            projection.push(table_len);
            projection.len() - 1
        });
        projection.push(pos_index);
        let pos_col_idx = projection.len() - 1;
        let builder = FileScanConfigBuilder::new(
            self.object_store_url.clone(),
            Arc::new(
                session_parquet_source(state, scan_schema)
                    .with_parquet_file_reader_factory(reader_factory),
            ),
        )
        .with_expr_adapter(Some(Arc::new(DuckLakeDefaultExprAdapterFactory)))
        .with_file_group(FileGroup::new(vec![pf]))
        .with_projection_indices(Some(projection))?;
        let scan = DataSourceExec::from_data_source(builder.build());
        let table_fields: Vec<FieldRef> = table_indices
            .iter()
            .map(|&i| Arc::clone(&self.table_schema.fields()[i]))
            .collect();
        let scan =
            present_catalog_schema(scan, &table_fields, &layout.name_mapping, &layout.constants);
        #[cfg(debug_assertions)]
        {
            let schema = scan.schema();
            let field = schema.fields().get(pos_col_idx).unwrap_or_else(|| {
                panic!(
                    "pos_col_idx {pos_col_idx} is out of range for a scan of {} columns",
                    schema.fields().len()
                )
            });
            crate::row_id::validate_row_pos_field("deletion data scan", pos_col_idx, field)
                .expect("`pos_col_idx` arithmetic must land on the position column");
        }
        Ok((scan, embedded_col_idx, pos_col_idx))
    }

    /// Scan of a positional delete file (the standard `(file_path, pos)`
    /// schema, plus its per-row snapshot column when it has one).
    fn delete_file_scan(
        &self,
        state: &dyn Session,
        file: &DeleteFileRef,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let reader_factory = cached_parquet_reader_factory(state, self.object_store_url)?;
        let mut pf = PartitionedFile::new(
            &file.path,
            validated_file_size(file.size_bytes, &file.path)?,
        );
        if let Some(hint) = metadata_size_hint(Some(file.footer_size)) {
            pf = pf.with_metadata_size_hint(hint);
        }
        let schema = match &file.snapshot_name {
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
            self.object_store_url.clone(),
            Arc::new(
                session_parquet_source(state, schema)
                    .with_parquet_file_reader_factory(reader_factory),
            ),
        )
        .with_expr_adapter(Some(Arc::new(DuckLakeDefaultExprAdapterFactory)))
        .with_file_group(FileGroup::new(vec![pf]));
        Ok(DataSourceExec::from_data_source(builder.build()))
    }
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
    /// The table's columns as of `end_snapshot`, carrying the field ids each data
    /// file's columns are resolved by. Set via [`Self::with_columns`]; fetched
    /// from the metadata provider on demand when it is not.
    columns: Option<Arc<Vec<DuckLakeTableColumn>>>,
    /// Per-file read layout, memoized by resolved path: several delete records in
    /// one window can share a source data file.
    layout_cache: Mutex<HashMap<String, Arc<ParquetFileLayout>>>,
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
            columns: None,
            layout_cache: Mutex::new(HashMap::new()),
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
        // The columns as of the window's END snapshot, carrying the field ids the
        // per-file read schemas are built from — official DuckLake resolves a
        // change feed against exactly that generation of the schema.
        let columns = self.resolve_columns()?;

        // The delete files added in the window, and the deletions stored in the
        // catalog (inlined rows deleted or superseded in the window, and
        // inlined deletions of Parquet-resident rows), read in one consistent
        // read of the catalog (`MetadataProvider::begin_consistent_read`): a
        // flush that commits meanwhile moves inlined deletions into delete
        // files, and separate reads could see them in both places or in
        // neither.
        let window = (self.start_snapshot, self.end_snapshot);
        // What the plan holds from the catalog is charged to the session's
        // memory pool as it is read, and for as long as the plan holds it.
        let charge = plan_charge(state, "ducklake_table_deletions");
        let (window_deletions, inlined_rows, inlined_file_deletes) =
            crate::metadata_provider::read_consistently(&self.provider, |provider| {
                let rewrites = rewrite_snapshots(provider.as_ref(), self.table_id, window)?;
                let window_deletions = WindowDeletions::read(
                    provider.as_ref(),
                    self.table_id,
                    window,
                    &rewrites,
                    &charge,
                )?;
                let inlined_rows = InlinedChanges::read_deletes(
                    provider.as_ref(),
                    self.table_id,
                    &columns,
                    &self.table_schema,
                    window,
                    &charge,
                )?;
                let inlined_file_deletes = InlinedChanges::read_file_deletes(
                    provider.as_ref(),
                    self.table_id,
                    window,
                    &charge,
                )?;
                Ok::<_, DataFusionError>((window_deletions, inlined_rows, inlined_file_deletes))
            })?;

        // Handle empty case
        if window_deletions.is_empty() && inlined_rows.is_empty() && inlined_file_deletes.is_empty()
        {
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

        // The table columns the caller reads, and whether it reads `rowid` (at
        // index 1 among the leading CDC columns). When it does not, rowids are
        // not resolved (no embedded read, no synthesis), so a query like
        // `SELECT id, change_type FROM ducklake_table_deletions(...)` never
        // fails on a file whose rowid cannot be synthesized (no embedded rowid
        // and a NULL row_id_start).
        let table_len = self.table_schema.fields().len();
        let (table_indices, need_rowid) = projected_table_indices(projection, 3, table_len);

        let files = self.feed_files();
        let sources = files
            .deletion_sources(
                state,
                &columns,
                &window_deletions,
                inlined_file_deletes,
                &charge,
            )
            .await?;
        // The catalog reads not kept are released: the charge follows what the
        // plan holds.
        drop(window_deletions);
        charge.try_resize(
            inlined_rows.bytes()
                + sources
                    .iter()
                    .map(DeletionSource::held_bytes)
                    .sum::<usize>(),
        )?;
        let mut execs: Vec<Arc<dyn ExecutionPlan>> = Vec::with_capacity(2);
        if !sources.is_empty() {
            execs.push(Arc::new(DeletedRowsExec::try_new(
                state,
                &files,
                Arc::new(sources),
                &table_indices,
                need_rowid,
                &charge,
            )?));
        }
        // Deleted inlined rows carry their row ids and deleting snapshots as
        // data, so they need no positional matching.
        if !inlined_rows.is_empty() {
            execs.push(inlined_rows.exec(&table_indices)?);
        }
        let rows: Arc<dyn ExecutionPlan> = if execs.len() == 1 {
            execs.into_iter().next().unwrap()
        } else {
            UnionExec::try_new(execs)?
        };
        project_feed(
            label_rows(rows, "delete")?,
            projection,
            &self.output_schema,
            3,
            &table_indices,
        )
    }
}

/// A delete file a deletion reads.
#[derive(Debug, Clone)]
pub(crate) struct DeleteFileRef {
    path: String,
    size_bytes: i64,
    footer_size: i64,
    /// The physical name of the file's per-row snapshot column, when it is
    /// cumulative.
    snapshot_name: Option<String>,
}

/// One delete record of the window (or one data file whose only deletions in
/// it are inlined): the source data file and what says which of its rows were
/// deleted in the window.
#[derive(Debug)]
pub(crate) struct DeletionSource {
    /// The delete record's snapshot: every deleted row's, unless the delete
    /// file or the inlined deletions date it per row.
    snapshot_id: i64,
    data_path: String,
    data_size_bytes: i64,
    data_footer_size: i64,
    layout: Arc<ParquetFileLayout>,
    /// The current delete file, or `None` for a full-file delete (or a legacy
    /// delete file that adds nothing in the window).
    current: Option<DeleteFileRef>,
    /// The delete file this one superseded, if any (legacy delta mode only).
    previous: Option<DeleteFileRef>,
    /// Whether the current delete file is cumulative (carries an embedded
    /// per-row delete-snapshot column as its third column). Rows are then
    /// windowed per row and emitted at their own delete snapshots.
    cumulative: bool,
    /// Whether the whole data file was retired at `snapshot_id` (no delete
    /// file), so every position not deleted before is deleted now.
    full_file: bool,
    /// Inlined deletions of the data file's rows made in the window:
    /// `(position, deleting snapshot)`, sorted by position.
    inlined_positions: Vec<(i64, i64)>,
    /// For a full-file delete: the positions deleted inlined before the
    /// window, which the retirement does not delete again.
    earlier_inlined: Vec<i64>,
    /// Total record count in data file (used for full file deletes)
    record_count: i64,
    /// First rowid of the data file (`None` if the catalog carries none). Used
    /// to synthesize a deleted row's rowid as `row_id_start + physical position`
    /// when the source file has no embedded rowid.
    row_id_start: Option<i64>,
}

impl DeletionSource {
    /// The bytes of positions the source holds.
    pub(crate) fn held_bytes(&self) -> usize {
        self.inlined_positions.capacity() * size_of::<(i64, i64)>()
            + self.earlier_inlined.capacity() * size_of::<i64>()
    }
}

/// The scans of one [`DeletionSource`], built for one projection.
#[derive(Debug)]
struct DeletionScan {
    source: usize,
    data_scan: Arc<dyn ExecutionPlan>,
    current_scan: Option<Arc<dyn ExecutionPlan>>,
    previous_scan: Option<Arc<dyn ExecutionPlan>>,
    /// Column index of the embedded rowid in the data-file batch, if read.
    embedded_col_idx: Option<usize>,
    /// Column index of the physical-position column in the data-file batch.
    pos_col_idx: usize,
}

/// Execution plan that reads the deleted rows of a window's deletion sources
/// as `[snapshot_id, rowid, projected table columns]`.
///
/// Each partition reads its share of the sources one at a time:
///
/// 1. the current delete file's positions (windowed per row when cumulative),
///    or every position for a full-file delete,
/// 2. plus the data file's inlined deletions in the window,
/// 3. minus the previous delete file's positions (legacy delta mode),
/// 4. then the data file, streamed, keeping the rows at those positions.
///
/// The position sets are charged to the session's memory pool, so a set too
/// large for it fails the query with a resources error. A full-file delete
/// holds no set.
///
/// No DataFusion children: the delete-position set is global to the data
/// file, so the optimizer must not repartition or split the internal scans
/// (issue #178). Rows are matched by the true physical position appended by
/// the positional data scan, never by arrival order.
#[derive(Debug)]
pub struct DeletedRowsExec {
    sources: Arc<Vec<DeletionSource>>,
    scans: Arc<Vec<DeletionScan>>,
    /// Whether rowids are resolved. When false, rowid is emitted as NULL
    /// (projected away above) and neither an embedded rowid nor a row_id_start
    /// is required.
    resolve_rowid: bool,
    /// Number of projected table columns leading each data-file batch.
    table_cols: usize,
    /// The query's inclusive `[start, end]` snapshot window (cumulative mode).
    window: (i64, i64),
    /// `[snapshot_id, rowid, projected table columns]`.
    schema: SchemaRef,
    partitions: usize,
    properties: Arc<PlanProperties>,
    /// The plan's charge, which holds the sources' positions.
    _charge: PlanCharge,
}

impl DeletedRowsExec {
    /// Plan the reads of `sources` for the table columns at `table_indices`.
    /// The positions the sources hold are charged to `charge`.
    pub(crate) fn try_new(
        state: &dyn Session,
        files: &FeedFiles<'_>,
        sources: Arc<Vec<DeletionSource>>,
        table_indices: &[usize],
        resolve_rowid: bool,
        charge: &PlanCharge,
    ) -> DataFusionResult<Self> {
        let mut scans = Vec::with_capacity(sources.len());
        for (index, source) in sources.iter().enumerate() {
            // The source file's embedded rowid column (present on UPDATE /
            // compaction outputs). When present, a deleted row's rowid is that
            // embedded value — NOT `row_id_start + position`, which would key
            // the delete differently from the row's insert/update_postimage.
            // Only read it when the rowid is resolved.
            let embedded_name = if resolve_rowid {
                source.layout.embedded_rowid_parquet_name.clone()
            } else {
                None
            };
            let (data_scan, embedded_col_idx, pos_col_idx) =
                files.data_file_scan(state, source, table_indices, &embedded_name)?;
            let current_scan = source
                .current
                .as_ref()
                .map(|file| files.delete_file_scan(state, file))
                .transpose()?;
            let previous_scan = source
                .previous
                .as_ref()
                .map(|file| files.delete_file_scan(state, file))
                .transpose()?;
            scans.push(DeletionScan {
                source: index,
                data_scan,
                current_scan,
                previous_scan,
                embedded_col_idx,
                pos_col_idx,
            });
        }
        let schema = keyed_rows_schema(files.table_schema, table_indices);
        let partitions = scans.len().min(state.config().target_partitions()).max(1);
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(Arc::clone(&schema)),
            datafusion::physical_expr::Partitioning::UnknownPartitioning(partitions),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ));
        Ok(Self {
            sources,
            scans: Arc::new(scans),
            resolve_rowid,
            table_cols: table_indices.len(),
            window: (files.start_snapshot, files.end_snapshot),
            schema,
            partitions,
            properties,
            _charge: Arc::clone(charge),
        })
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
                    "DeletedRowsExec: sources={}, table_columns={}, resolve_rowid={}",
                    self.scans.len(),
                    self.table_cols,
                    self.resolve_rowid
                )
            },
        }
    }
}

impl ExecutionPlan for DeletedRowsExec {
    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> DataFusionResult<TreeNodeRecursion>,
    ) -> DataFusionResult<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

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
        if partition >= self.partitions {
            return Err(DataFusionError::Internal(format!(
                "DeletedRowsExec has {} partitions, got {partition}",
                self.partitions
            )));
        }
        let reservation = Arc::new(Mutex::new(
            MemoryConsumer::new(format!("DeletedRowsExec[{partition}]"))
                .register(context.memory_pool()),
        ));
        let read = Arc::new(DeletionRead {
            sources: Arc::clone(&self.sources),
            scans: Arc::clone(&self.scans),
            resolve_rowid: self.resolve_rowid,
            table_cols: self.table_cols,
            window: self.window,
            schema: Arc::clone(&self.schema),
        });
        let indices: Vec<usize> = (partition..self.scans.len())
            .step_by(self.partitions)
            .collect();
        let stream = futures::stream::iter(indices)
            .then(move |index| {
                let read = Arc::clone(&read);
                let context = Arc::clone(&context);
                let reservation = Arc::clone(&reservation);
                async move { read.deleted_rows(index, context, reservation).await }
            })
            .try_flatten();
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&self.schema),
            stream,
        )))
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}

/// The positions of a data file deleted in the window: `(current ∪ inlined) −
/// previous`, where `current` is either a set read from the delete file or,
/// for a full-file delete, every position below the record count.
///
/// Held as sorted vectors rather than hash sets: a vector allocates exactly
/// its capacity, so the reservation that charges it is exact, and it can be
/// reserved before each growth.
pub(crate) struct DeletedPositions {
    /// Every position in `0..n` (a full-file delete).
    all_below: Option<i64>,
    /// Positions deleted at the record's snapshot, sorted.
    positions: Vec<i64>,
    /// `(position, deleting snapshot)` of the positions that carry their own
    /// (cumulative delete files and inlined deletions), sorted by position,
    /// one per position. They take precedence over `positions`.
    dated: Vec<(i64, i64)>,
    /// Positions not deleted, sorted; empty unless `all_below` is set.
    except: Vec<i64>,
}

impl DeletedPositions {
    /// The snapshot that deleted `position` in the window, `None` when the
    /// window did not delete it. A position without a snapshot of its own was
    /// deleted at `record_snapshot`.
    pub(crate) fn snapshot_of(&self, position: i64, record_snapshot: i64) -> Option<i64> {
        if self.except.binary_search(&position).is_ok() {
            return None;
        }
        if let Ok(i) = self.dated.binary_search_by_key(&position, |&(p, _)| p) {
            return Some(self.dated[i].1);
        }
        (self.positions.binary_search(&position).is_ok()
            || self.all_below.is_some_and(|n| (0..n).contains(&position)))
        .then_some(record_snapshot)
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.all_below.is_none_or(|n| n <= 0) && self.positions.is_empty() && self.dated.is_empty()
    }
}

/// Builds one data file's [`DeletedPositions`], charging what it holds to a
/// memory reservation. Each growth of a vector is reserved before it is
/// allocated, for the old and the new allocation together, since a
/// reallocation copies from one to the other.
pub(crate) struct DeletedPositionsBuilder<'a> {
    reservation: &'a Mutex<MemoryReservation>,
    all_below: Option<i64>,
    positions: Vec<i64>,
    dated: Vec<(i64, i64)>,
    except: Vec<i64>,
}

impl<'a> DeletedPositionsBuilder<'a> {
    pub(crate) fn new(reservation: &'a Mutex<MemoryReservation>) -> Self {
        Self {
            reservation,
            all_below: None,
            positions: Vec::new(),
            dated: Vec::new(),
            except: Vec::new(),
        }
    }

    /// The bytes the vectors hold.
    fn held(&self) -> usize {
        (self.positions.capacity() + self.except.capacity()) * size_of::<i64>()
            + self.dated.capacity() * size_of::<(i64, i64)>()
    }

    /// Make room in `vec` for `additional` more items: reserve the held bytes
    /// plus the new allocation first, allocate, then let the reservation fall
    /// back to what is held.
    fn grow<T>(
        reservation: &Mutex<MemoryReservation>,
        held: usize,
        vec: &mut Vec<T>,
        additional: usize,
    ) -> DataFusionResult<()> {
        let needed = vec.len().saturating_add(additional);
        if needed <= vec.capacity() {
            return Ok(());
        }
        let capacity = needed.max(vec.capacity().saturating_mul(2));
        let old_bytes = vec.capacity() * size_of::<T>();
        let new_bytes = capacity.saturating_mul(size_of::<T>());
        let reservation = reservation.lock().unwrap();
        reservation.try_resize(held.saturating_add(new_bytes))?;
        vec.reserve_exact(capacity - vec.len());
        // `reserve_exact` may allocate more than asked; the reservation follows
        // what was allocated.
        let allocated = vec.capacity() * size_of::<T>();
        reservation.try_resize(held - old_bytes + allocated)
    }

    /// Every position below `record_count` is deleted (a full-file delete).
    pub(crate) fn delete_all_below(&mut self, record_count: i64) {
        self.all_below = Some(record_count);
    }

    /// Add the positions of a delete file batch: `(file_path, pos)`, deleted at
    /// the record's snapshot, or with `window` a cumulative file's
    /// `(file_path, pos, snapshot)`, of which the positions deleted in the
    /// window count.
    pub(crate) fn add_delete_batch(
        &mut self,
        batch: &RecordBatch,
        window: Option<(i64, i64)>,
    ) -> DataFusionResult<()> {
        let Some(window) = window else {
            if batch.num_columns() < 2 {
                return Ok(());
            }
            let pos = int64_column(batch, 1, "delete `pos`")?;
            let held = self.held();
            Self::grow(
                self.reservation,
                held,
                &mut self.positions,
                pos.len() - pos.null_count(),
            )?;
            self.positions.extend(pos.iter().flatten());
            return Ok(());
        };
        if batch.num_columns() < 3 {
            return Err(DataFusionError::Internal(
                "cumulative delete file batch is missing its snapshot column".to_string(),
            ));
        }
        let pos = int64_column(batch, 1, "delete `pos`")?;
        let snaps = int64_column(batch, 2, "delete snapshot")?;
        let in_window = |i: usize| -> DataFusionResult<bool> {
            if pos.is_null(i) {
                return Ok(false);
            }
            if snaps.is_null(i) {
                return Err(DataFusionError::Internal(
                    "cumulative delete file has a NULL per-row snapshot".to_string(),
                ));
            }
            Ok((window.0..=window.1).contains(&snaps.value(i)))
        };
        let mut count = 0;
        for i in 0..batch.num_rows() {
            count += usize::from(in_window(i)?);
        }
        let held = self.held();
        Self::grow(self.reservation, held, &mut self.dated, count)?;
        for i in 0..batch.num_rows() {
            if in_window(i)? {
                self.dated.push((pos.value(i), snaps.value(i)));
            }
        }
        Ok(())
    }

    /// Add inlined deletions: position and deleting snapshot. They take
    /// precedence over a delete file's snapshot for the same position.
    pub(crate) fn add_inlined(
        &mut self,
        inlined: impl ExactSizeIterator<Item = (i64, i64)>,
    ) -> DataFusionResult<()> {
        if inlined.len() == 0 {
            return Ok(());
        }
        // The delete file's dated positions, sorted and one per position, so
        // that an inlined deletion can replace the entry of its position.
        sort_dated(&mut self.dated);
        let sorted = self.dated.len();
        let held = self.held();
        Self::grow(self.reservation, held, &mut self.dated, inlined.len())?;
        for (position, snapshot) in inlined {
            match self.dated[..sorted].binary_search_by_key(&position, |&(p, _)| p) {
                Ok(i) => self.dated[i].1 = snapshot,
                Err(_) => self.dated.push((position, snapshot)),
            }
        }
        Ok(())
    }

    /// Subtract the positions of a previous delete file batch.
    pub(crate) fn add_except_batch(&mut self, batch: &RecordBatch) -> DataFusionResult<()> {
        if batch.num_columns() < 2 {
            return Ok(());
        }
        let pos = int64_column(batch, 1, "delete `pos`")?;
        let held = self.held();
        Self::grow(
            self.reservation,
            held,
            &mut self.except,
            pos.len() - pos.null_count(),
        )?;
        self.except.extend(pos.iter().flatten());
        Ok(())
    }

    /// Subtract `positions`.
    pub(crate) fn add_except(
        &mut self,
        positions: impl ExactSizeIterator<Item = i64>,
    ) -> DataFusionResult<()> {
        let held = self.held();
        Self::grow(self.reservation, held, &mut self.except, positions.len())?;
        self.except.extend(positions);
        Ok(())
    }

    pub(crate) fn finish(mut self) -> DataFusionResult<DeletedPositions> {
        self.positions.sort_unstable();
        self.positions.dedup();
        sort_dated(&mut self.dated);
        self.except.sort_unstable();
        self.except.dedup();
        // A position the previous file already held is not deleted now; the
        // explicit sets can drop it, a full-file range keeps the exception.
        if self.all_below.is_none() && !self.except.is_empty() {
            let except = std::mem::take(&mut self.except);
            self.positions.retain(|p| except.binary_search(p).is_err());
            self.dated.retain(|(p, _)| except.binary_search(p).is_err());
            drop(except);
            let held = self.held();
            self.reservation.lock().unwrap().try_resize(held)?;
        }
        Ok(DeletedPositions {
            all_below: self.all_below,
            positions: self.positions,
            dated: self.dated,
            except: self.except,
        })
    }
}

/// Sort `(position, snapshot)` pairs by position, one per position; of two
/// pairs of one position the later snapshot is kept.
fn sort_dated(dated: &mut Vec<(i64, i64)>) {
    dated.sort_unstable();
    dated.dedup_by(|later, earlier| {
        let same = later.0 == earlier.0;
        if same {
            earlier.1 = later.1;
        }
        same
    });
}

/// What every partition of a [`DeletedRowsExec`] reads from.
struct DeletionRead {
    sources: Arc<Vec<DeletionSource>>,
    scans: Arc<Vec<DeletionScan>>,
    resolve_rowid: bool,
    table_cols: usize,
    window: (i64, i64),
    schema: SchemaRef,
}

impl DeletionRead {
    /// Collect the deleted position set of the source scanned by
    /// `scans[index]` (charged to `reservation`), then return its data file's
    /// batches filtered to the deleted rows, matching by the true physical
    /// position appended by the positional scan. The data file itself is
    /// streamed batch-by-batch, never materialized whole.
    async fn deleted_rows(
        self: Arc<Self>,
        index: usize,
        context: Arc<TaskContext>,
        reservation: Arc<Mutex<MemoryReservation>>,
    ) -> DataFusionResult<BoxStream<'static, DataFusionResult<RecordBatch>>> {
        let scan = &self.scans[index];
        let source = &self.sources[scan.source];
        // The previous source's sets are dropped by now.
        reservation.lock().unwrap().free();
        let mut builder = DeletedPositionsBuilder::new(&reservation);

        // 1. Deleted positions, from the current delete file (windowed per
        //    row in cumulative mode) or every position for a full-file delete.
        match &scan.current_scan {
            Some(current) => {
                let mut stream =
                    reset_plan_states(Arc::clone(current))?.execute(0, Arc::clone(&context))?;
                while let Some(batch) = stream.next().await {
                    builder.add_delete_batch(&batch?, source.cumulative.then_some(self.window))?;
                }
            },
            None if source.full_file => builder.delete_all_below(source.record_count),
            None => {},
        }
        builder.add_inlined(source.inlined_positions.iter().copied())?;

        // 2. The previous delete file's positions are subtracted (legacy delta
        //    mode; the per-row window filter replaces this in cumulative mode).
        if let Some(previous) = &scan.previous_scan {
            let mut stream =
                reset_plan_states(Arc::clone(previous))?.execute(0, Arc::clone(&context))?;
            while let Some(batch) = stream.next().await {
                builder.add_except_batch(&batch?)?;
            }
        }
        // 3. A retired file's rows deleted inlined before the window are not
        //    deleted by its retirement.
        builder.add_except(source.earlier_inlined.iter().copied())?;
        let deleted = builder.finish()?;
        if deleted.is_empty() {
            return Ok(futures::stream::empty().boxed());
        }

        // 4. Stream the data file and keep the rows whose PHYSICAL position is
        //    in the deleted set.
        //
        //    Only partition 0 is executed, so the scan MUST have exactly one:
        //    this exec has no DataFusion children (see the module docs), so
        //    nothing repartitions it, and it is built from a single
        //    `FileGroup`. If that ever stops holding, every row in the other
        //    partitions would be silently dropped — a deletion feed that quietly
        //    omits deletions — so refuse rather than under-report.
        let partitions = scan.data_scan.output_partitioning().partition_count();
        if partitions != 1 {
            return Err(DataFusionError::Internal(format!(
                "deletions data-file scan must be a single partition covering the whole \
                 file, found {partitions}"
            )));
        }
        let data = reset_plan_states(Arc::clone(&scan.data_scan))?.execute(0, context)?;
        let read = Arc::clone(&self);
        Ok(data
            .try_filter_map(move |batch| {
                let _held = &reservation;
                futures::future::ready(read.filter_batch(index, &batch, &deleted))
            })
            .boxed())
    }

    /// Filter a data-file batch to its deleted rows, prepending each one's
    /// snapshot and rowid.
    fn filter_batch(
        &self,
        index: usize,
        batch: &RecordBatch,
        deleted: &DeletedPositions,
    ) -> DataFusionResult<Option<RecordBatch>> {
        let scan = &self.scans[index];
        let source = &self.sources[scan.source];
        // The physical position of each row, appended by the positional scan.
        let pos = int64_column(batch, scan.pos_col_idx, "physical-position")?;
        // Resolve each deleted row's rowid: the embedded rowid column when the
        // source file has one (an UPDATE / compaction output), else
        // `row_id_start + physical position`.
        let embedded = scan
            .embedded_col_idx
            .map(|idx| int64_column(batch, idx, "embedded rowid"))
            .transpose()?;
        // Require a row_id_start only when a rowid is actually needed and there
        // is no embedded rowid to read.
        let synth_start: Option<i64> = if self.resolve_rowid && embedded.is_none() {
            Some(source.row_id_start.ok_or_else(|| {
                DataFusionError::Internal(
                    "cannot synthesize deleted rowid: source file has neither an embedded \
                     rowid nor a row_id_start"
                        .to_string(),
                )
            })?)
        } else {
            None
        };

        let mut keep: Vec<u32> = Vec::new();
        let mut rowids: Vec<Option<i64>> = Vec::new();
        let mut row_snapshots: Vec<i64> = Vec::new();
        for i in 0..batch.num_rows() {
            let position = pos.value(i);
            let Some(snapshot) = deleted.snapshot_of(position, source.snapshot_id) else {
                continue;
            };
            keep.push(u32::try_from(i).expect("a batch holds fewer than 2^32 rows"));
            rowids.push(match (embedded, synth_start) {
                (Some(arr), _) => arr.is_valid(i).then(|| arr.value(i)),
                (None, Some(start)) => Some(start + position),
                // Rowid not resolved: projected away above.
                (None, None) => None,
            });
            row_snapshots.push(snapshot);
        }
        if keep.is_empty() {
            return Ok(None);
        }
        let indices = UInt32Array::from(keep);
        let mut columns: Vec<ArrayRef> = Vec::with_capacity(self.table_cols + 2);
        columns.push(Arc::new(Int64Array::from(row_snapshots)));
        columns.push(Arc::new(Int64Array::from(rowids)));
        for column in batch.columns().iter().take(self.table_cols) {
            columns.push(
                take(column.as_ref(), &indices, None)
                    .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?,
            );
        }
        RecordBatch::try_new(Arc::clone(&self.schema), columns)
            .map(Some)
            .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))
    }
}

/// Downcast a batch column to `Int64Array` with a descriptive error.
fn int64_column<'a>(
    batch: &'a RecordBatch,
    idx: usize,
    what: &str,
) -> DataFusionResult<&'a Int64Array> {
    batch
        .columns()
        .get(idx)
        .and_then(|column| column.as_any().downcast_ref::<Int64Array>())
        .ok_or_else(|| DataFusionError::Internal(format!("{what} column is missing or not Int64")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_alloc::{BudgetPool, measure};
    use arrow::array::StringArray;
    use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryPool};

    /// A delete file batch: `(file_path, pos)`, plus each position's snapshot
    /// when `snapshots` is given.
    fn delete_batch(positions: std::ops::Range<i64>, snapshot: Option<i64>) -> RecordBatch {
        let n = positions.end - positions.start;
        let mut fields = vec![
            Field::new("file_path", DataType::Utf8, false),
            Field::new("pos", DataType::Int64, false),
        ];
        let mut columns: Vec<ArrayRef> = vec![
            Arc::new(StringArray::from(vec!["f.parquet"; n as usize])),
            Arc::new(Int64Array::from_iter_values(positions)),
        ];
        if let Some(snapshot) = snapshot {
            fields.push(Field::new("snapshot", DataType::Int64, false));
            columns.push(Arc::new(Int64Array::from(vec![snapshot; n as usize])));
        }
        RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap()
    }

    /// Build one file's deleted positions from `batches` (cumulative when
    /// `window` is given), `inlined` deletions and a previous delete file of
    /// `except`, charged to `reservation`.
    fn build(
        reservation: &Mutex<MemoryReservation>,
        batches: &[RecordBatch],
        window: Option<(i64, i64)>,
        inlined: &[(i64, i64)],
        except: &RecordBatch,
    ) -> DataFusionResult<DeletedPositions> {
        let mut builder = DeletedPositionsBuilder::new(reservation);
        for batch in batches {
            builder.add_delete_batch(batch, window)?;
        }
        builder.add_inlined(inlined.iter().copied())?;
        builder.add_except_batch(except)?;
        builder.add_except([3, 5, 1_000_001].into_iter())?;
        builder.finish()
    }

    fn reservation(pool: Arc<dyn MemoryPool>) -> Mutex<MemoryReservation> {
        let reservation = Mutex::new(MemoryConsumer::new("test").register(&pool));
        // A platform mutex can allocate on its first lock; not the sets' bytes.
        drop(reservation.lock().unwrap());
        reservation
    }

    /// Every byte the position sets allocate is reserved before it is
    /// allocated, the transient copy of a growth included, and the
    /// reservation is not far above what they hold.
    #[test]
    fn delete_positions_are_reserved_before_they_are_allocated() {
        for window in [None, Some((5, 9))] {
            let snapshot = window.map(|_| 6);
            let batches: Vec<RecordBatch> = (0..3)
                .map(|b| delete_batch(b * 30_000..(b + 1) * 30_000 + 7, snapshot))
                .collect();
            let inlined: Vec<(i64, i64)> = (0..5_000).map(|i| (100_000 + i * 3, 8)).collect();
            let except = delete_batch(40_000..50_000, None);

            let budget = Arc::new(BudgetPool::default());
            let charged = reservation(Arc::clone(&budget) as _);
            let (deleted, measured) =
                measure(|| build(&charged, &batches, window, &inlined, &except).unwrap());
            assert!(
                measured.excess <= 0,
                "window {window:?}: allocated {} bytes beyond the reservation (peak {} \
                 allocated, {} reserved)",
                measured.excess,
                measured.peak,
                budget.peak()
            );
            assert!(
                budget.peak() <= 2 * measured.peak + 4096,
                "window {window:?}: reserved {} for a peak of {} allocated",
                budget.peak(),
                measured.peak
            );

            // The sets are right.
            assert_eq!(deleted.snapshot_of(0, 7), Some(snapshot.unwrap_or(7)));
            assert_eq!(deleted.snapshot_of(3, 7), None);
            assert_eq!(deleted.snapshot_of(45_000, 7), None);
            assert_eq!(deleted.snapshot_of(100_003, 7), Some(8));
            assert_eq!(deleted.snapshot_of(100_004, 7), None);
            assert_eq!(deleted.snapshot_of(90_006, 7), Some(snapshot.unwrap_or(7)));
            assert_eq!(deleted.snapshot_of(90_007, 7), None);

            // A pool one byte short of the reservation's peak refuses the set,
            // and one of exactly the peak holds it.
            let short = reservation(Arc::new(GreedyMemoryPool::new(budget.peak() - 1)));
            assert!(matches!(
                build(&short, &batches, window, &inlined, &except),
                Err(DataFusionError::ResourcesExhausted(_))
            ));
            let exact = reservation(Arc::new(GreedyMemoryPool::new(budget.peak())));
            build(&exact, &batches, window, &inlined, &except).unwrap();
        }
    }
}
