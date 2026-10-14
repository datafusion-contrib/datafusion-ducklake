//! Flushing catalog-inlined data to Parquet: the crate's
//! `ducklake_flush_inlined_data`.
//!
//! Mirrors official DuckLake's flush (`ducklake_flush_inlined_data.cpp`):
//!
//! - Every physical inlined-data table of a table is written to Parquet, in files
//!   per partition of the partition spec in effect when its rows were inserted,
//!   rolled over at the table's `target_file_size` and placed in Hive directories
//!   unless `hive_file_pattern` is `false`. The files hold every row the table
//!   holds at the flush snapshot — deleted rows included — with its row id and
//!   insert snapshot embedded (`_ducklake_internal_row_id`,
//!   `_ducklake_internal_snapshot_id`), ordered by the table's current sort order
//!   and then `(row_id, begin_snapshot)`. Each file's catalog row begins at its
//!   smallest embedded snapshot, records the largest as `partial_max` and the
//!   smallest row id as `row_id_start`, so time travel to any earlier snapshot
//!   returns exactly what the inlined rows returned.
//! - Rows deleted while inlined become positions of a delete file per data file,
//!   counted from that file's first row, that records each deleting snapshot and
//!   is registered from the earliest of them.
//! - Inlined deletions of Parquet rows (`ducklake_inlined_delete_<table_id>`)
//!   are merged with the data file's live delete file into a new one that
//!   records each position's snapshot; the replaced delete file is removed and
//!   scheduled for deletion.
//! - One snapshot commits every table, deletes the flushed inlined rows, and
//!   records `deleted_from_table:<id>` / `inline_flush:<id>`; then empty
//!   superseded inlined tables are dropped. A table that cannot be flushed fails
//!   the whole flush, as official's single transaction does, and every object the
//!   attempt wrote is removed (official's `CleanupFiles`); so is every object of
//!   a commit the catalog rejected.
//!
//! Deliberate differences from official:
//!
//! - Inlined deletions of a data file retired since (by `rewrite_data_files`,
//!   say) whose delete file has ended merge into that delete file's history:
//!   the new file replaces the latest one and ends where it ended. Official
//!   writes a second, never-ending delete file beside the ended one, and time
//!   travel to the snapshots both cover then reads duplicated rows and
//!   resurrects the ended file's deletions. Deletions that cannot merge (the
//!   merged file would overlap an earlier delete file, or a deletion is later
//!   than the delete file's end) stay inlined.
//! - A catalog whose `encrypted` option is `true` is refused, as every
//!   Parquet-writing operation is there. In a catalog not marked encrypted,
//!   inlined deletions of an encrypted data or delete file stay inlined (this
//!   crate cannot write encrypted delete files), and reads keep applying them.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, Decimal128Array, Int64Array, RecordBatch, StringArray, UInt32Array,
};
use arrow::compute::{SortColumn, SortOptions, concat_batches, lexsort_to_indices, take};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::catalog::{
    CatalogProvider, Session, TableFunctionArgs, TableFunctionImpl, TableProvider,
};
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::common::{DFSchema, ScalarValue, plan_err};
use datafusion::datasource::TableType;
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::execution::object_store::ObjectStoreUrl;
use datafusion::execution::runtime_env::RuntimeEnv;
use datafusion::execution::{SendableRecordBatchStream, SessionState, TaskContext};
use datafusion::logical_expr::Expr;
use datafusion::physical_expr::{EquivalenceProperties, Partitioning, PhysicalExpr};
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use datafusion::prelude::{SessionConfig, SessionContext};
use futures::StreamExt;
use object_store::ObjectStore;
use object_store::ObjectStoreExt;
use object_store::path::Path as ObjectPath;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use crate::catalog::DuckLakeCatalog;
use crate::metadata_provider::{
    DuckLakeInlinedFileDeleteForFlush, DuckLakeInlinedFlushData, DuckLakeTableColumn,
};
use crate::metadata_writer::{
    InlinedFlushDeleteFile, InlinedFlushFile, InlinedFlushFileDelete, InlinedFlushTable,
};
use crate::partition::{PartitionWriteSpec, hive_subpath, split_batches_by_partition};
use crate::path_resolver::resolve_path;
use crate::row_id::{EMBEDDED_ROW_ID_COLUMN_NAME, EMBEDDED_SNAPSHOT_ID_COLUMN_NAME};
use crate::sort::{NullOrder, SortDirection};
use crate::table::{DELETE_POS_COL, DELETE_SNAPSHOT_COL, DuckLakeTable};
use crate::table_writer::DuckLakeTableWriter;
use crate::{DuckLakeError, Result};

/// The schema DuckDB's DuckLake extension falls back to when a flush names a
/// table without a schema.
const DEFAULT_SCHEMA: &str = "main";

/// Rows per batch the flush feeds its file writer; a file rolls over only at a
/// batch boundary.
const FLUSH_BATCH_ROWS: usize = 2048;

/// One row of `ducklake_flush_inlined_data`'s result: a table whose inlined rows
/// were written to Parquet.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct FlushedInlinedTable {
    /// The table's schema.
    pub schema_name: String,
    /// The table.
    pub table_name: String,
    /// Rows written to Parquet, including rows deleted while inlined (they are
    /// written and masked by a delete file, as official DuckLake does).
    pub rows_flushed: i64,
}

/// The Arrow schema of `ducklake_flush_inlined_data`'s result. `rows_flushed` is
/// official's `HUGEINT` as DuckDB exports it to Arrow, `Decimal128(38, 0)`.
pub fn flush_inlined_data_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("schema_name", DataType::Utf8, false),
        Field::new("table_name", DataType::Utf8, false),
        Field::new("rows_flushed", DataType::Decimal128(38, 0), false),
    ]))
}

/// Render flush results as one batch of [`flush_inlined_data_schema`].
pub fn flushed_inlined_tables_batch(tables: &[FlushedInlinedTable]) -> Result<RecordBatch> {
    Ok(RecordBatch::try_new(
        flush_inlined_data_schema(),
        vec![
            Arc::new(StringArray::from_iter_values(
                tables.iter().map(|table| table.schema_name.as_str()),
            )),
            Arc::new(StringArray::from_iter_values(
                tables.iter().map(|table| table.table_name.as_str()),
            )),
            Arc::new(
                Decimal128Array::from_iter_values(
                    tables.iter().map(|table| i128::from(table.rows_flushed)),
                )
                .with_precision_and_scale(38, 0)?,
            ),
        ],
    )?)
}

/// Where a flush writes its objects.
pub(crate) enum FlushObjectStores {
    /// Each table's store, from the session's registry by the table's URL.
    Session(Arc<RuntimeEnv>),
    /// One store for every table.
    Fixed(Arc<dyn ObjectStore>),
}

impl FlushObjectStores {
    fn for_table(&self, url: &ObjectStoreUrl) -> Result<Arc<dyn ObjectStore>> {
        match self {
            Self::Session(runtime) => Ok(runtime.object_store(url)?),
            Self::Fixed(store) => Ok(Arc::clone(store)),
        }
    }
}

/// What one table contributed to a committed flush.
pub(crate) struct FlushedTableDetail {
    pub(crate) flushed: FlushedInlinedTable,
    pub(crate) table_id: i64,
    pub(crate) schema_id: i64,
    pub(crate) files_written: usize,
}

/// The result of [`DuckLakeCatalog::flush_inlined_data_with`].
pub(crate) struct FlushOutcome {
    /// The flush snapshot, or `None` when nothing was committed.
    pub(crate) snapshot_id: Option<i64>,
    /// Tables that had rows flushed, ordered by schema and table.
    pub(crate) tables: Vec<FlushedTableDetail>,
}

/// Objects a flush attempt has written, removed when the attempt is abandoned.
#[derive(Default)]
struct WrittenObjects {
    objects: Vec<(Arc<dyn ObjectStore>, Vec<ObjectPath>)>,
}

impl WrittenObjects {
    /// Best-effort removal: the flush is already failing, so a delete that also
    /// fails is logged rather than masking its error. A survivor is
    /// unreferenced, so `delete_orphaned_files` reclaims it.
    async fn remove(self) {
        for (store, paths) in self.objects {
            let stream = futures::stream::iter(paths.into_iter().map(Ok)).boxed();
            let mut deletions = store.delete_stream(stream);
            while let Some(outcome) = deletions.next().await {
                match outcome {
                    Ok(_)
                    | Err(object_store::Error::NotFound {
                        ..
                    }) => {},
                    Err(error) => tracing::warn!(
                        %error,
                        "failed to remove an object written by an abandoned inlined-data flush"
                    ),
                }
            }
        }
    }
}

/// Refuse `operation` on a catalog whose `encrypted` option is `true`, as the
/// writer reads it ([`crate::metadata_writer::MetadataWriter::catalog_encrypted`], which fails on a
/// value other than `true` or `false`). Official DuckLake writes the Parquet
/// files of such a catalog with Parquet Modular Encryption; this crate writes
/// no encrypted files, so writing plaintext ones there would break the
/// catalog's guarantee.
pub(crate) fn refuse_encrypted_catalog(
    writer: &dyn crate::metadata_writer::MetadataWriter,
    operation: &str,
) -> Result<()> {
    if writer.catalog_encrypted()? {
        return Err(encrypted_catalog_error(operation));
    }
    Ok(())
}

/// The error every Parquet-writing operation on an encrypted catalog fails
/// with: [`DuckLakeError::Unsupported`], naming the operation.
pub(crate) fn encrypted_catalog_error(operation: &str) -> DuckLakeError {
    DuckLakeError::Unsupported(format!(
        "{operation}: the catalog is encrypted, and this crate does not write encrypted Parquet \
         files"
    ))
}

impl DuckLakeCatalog {
    /// Write the catalog-inlined rows of every table — or of one schema's
    /// tables, or of one table — to Parquet, in one snapshot. The equivalent of
    /// official DuckLake's `ducklake_flush_inlined_data(catalog, schema_name =>,
    /// table_name =>)`; see the [module docs](crate::inlined_flush) for what is
    /// written.
    ///
    /// `table_name` without `schema_name` looks in `main`. Tables whose
    /// `auto_compact` setting is `false` are skipped, as official does. Returns
    /// one entry per table that had rows to flush, ordered by schema and table.
    /// Commits nothing when there is nothing to flush.
    ///
    /// Errors when the catalog is read-only, a named schema or table does not
    /// exist, a concurrent commit conflicts ([`DuckLakeError::Conflict`]),
    /// inlined rows hold a type Parquet cannot store through Arrow (`INTERVAL`,
    /// [`DuckLakeError::Unsupported`]), or the catalog's `encrypted` option is
    /// `true` ([`DuckLakeError::Unsupported`], before anything is written). Any of these fails the whole flush and
    /// removes the objects it wrote. Dropping empty superseded inlined tables
    /// runs after the flush commits, so an error there is returned although the
    /// flush has committed.
    pub async fn flush_inlined_data(
        &self,
        state: &dyn Session,
        schema_name: Option<&str>,
        table_name: Option<&str>,
    ) -> Result<Vec<FlushedInlinedTable>> {
        let outcome = self
            .flush_inlined_data_with(
                FlushObjectStores::Session(Arc::clone(state.runtime_env())),
                schema_name,
                table_name,
                None,
            )
            .await?;
        Ok(outcome
            .tables
            .into_iter()
            .map(|table| table.flushed)
            .collect())
    }

    /// [`Self::flush_inlined_data`] writing through `stores`, reading the inlined
    /// rows at `base_snapshot` (the catalog head when `None`) and committing
    /// against it.
    pub(crate) async fn flush_inlined_data_with(
        &self,
        stores: FlushObjectStores,
        schema_name: Option<&str>,
        table_name: Option<&str>,
        base_snapshot: Option<i64>,
    ) -> Result<FlushOutcome> {
        let writer = self.writer().ok_or_else(|| {
            DuckLakeError::InvalidConfig(
                "flush_inlined_data: catalog is read-only; open it with a writer".to_string(),
            )
        })?;
        let provider = self.provider();
        refuse_encrypted_catalog(writer.as_ref(), "flush_inlined_data")?;
        let base_snapshot = match base_snapshot {
            Some(snapshot) => snapshot,
            None => provider.get_current_snapshot()?,
        };

        // Resolve the target tables at the flush snapshot.
        let mut targets: Vec<(i64, String, String)> = Vec::new();
        match (schema_name, table_name) {
            (schema, Some(table)) => {
                let schema = schema.unwrap_or(DEFAULT_SCHEMA);
                let meta = provider
                    .get_schema_by_name(schema, base_snapshot)?
                    .ok_or_else(|| DuckLakeError::SchemaNotFound(schema.to_string()))?;
                if provider
                    .get_table_by_name(meta.schema_id, table, base_snapshot)?
                    .is_none()
                {
                    return Err(DuckLakeError::TableNotFound(format!("{schema}.{table}")));
                }
                targets.push((meta.schema_id, schema.to_string(), table.to_string()));
            },
            (Some(schema), None) => {
                let meta = provider
                    .get_schema_by_name(schema, base_snapshot)?
                    .ok_or_else(|| DuckLakeError::SchemaNotFound(schema.to_string()))?;
                for table in provider.list_tables(meta.schema_id, base_snapshot)? {
                    targets.push((meta.schema_id, schema.to_string(), table.table_name));
                }
            },
            (None, None) => {
                for schema in provider.list_schemas(base_snapshot)? {
                    for table in provider.list_tables(schema.schema_id, base_snapshot)? {
                        targets.push((
                            schema.schema_id,
                            schema.schema_name.clone(),
                            table.table_name,
                        ));
                    }
                }
            },
        }

        let mut written = WrittenObjects::default();
        let attempt = async {
            let mut plans = Vec::new();
            let mut details = Vec::new();
            let mut considered = false;
            for (schema_id, schema, table) in targets {
                let provider_table = self
                    .schema(&schema)
                    .ok_or_else(|| DuckLakeError::SchemaNotFound(schema.clone()))?
                    .table(&table)
                    .await?
                    .ok_or_else(|| DuckLakeError::TableNotFound(format!("{schema}.{table}")))?;
                let Some(table_provider) = provider_table.downcast_ref::<DuckLakeTable>() else {
                    // A view; it holds no rows.
                    continue;
                };
                if !table_provider.write_options().auto_compact.unwrap_or(true) {
                    continue;
                }
                considered = true;
                let plan = flush_table(
                    table_provider,
                    &writer,
                    &stores,
                    base_snapshot,
                    &mut written,
                )
                .await?;
                if plan.files.is_empty() && plan.file_deletes.is_empty() {
                    continue;
                }
                let rows_flushed = plan
                    .files
                    .iter()
                    .map(|file| file.file.record_count)
                    .sum::<i64>();
                if rows_flushed > 0 {
                    details.push(FlushedTableDetail {
                        flushed: FlushedInlinedTable {
                            schema_name: schema,
                            table_name: table,
                            rows_flushed,
                        },
                        table_id: plan.table_id,
                        schema_id,
                        files_written: plan.files.len(),
                    });
                }
                plans.push(plan);
            }
            Ok::<_, DuckLakeError>((plans, details, considered))
        }
        .await;
        let (plans, mut details, considered) = match attempt {
            Ok(planned) => planned,
            Err(error) => {
                written.remove().await;
                return Err(error);
            },
        };
        if plans.is_empty() {
            // Official's flush commits no snapshot here, and still drops the
            // empty superseded inlined data tables once it has visited a table.
            if considered {
                writer.drop_empty_superseded_inlined_tables()?;
            }
            return Ok(FlushOutcome {
                snapshot_id: None,
                tables: Vec::new(),
            });
        }
        let snapshot_id = match writer.commit_inlined_flush(base_snapshot, &plans) {
            Ok(snapshot_id) => snapshot_id,
            Err(error) => {
                // Remove the objects unless the outcome is unknown: a networked
                // COMMIT that failed may still have applied, and removing the
                // objects then would leave a committed snapshot naming missing
                // files; an unreferenced survivor is reclaimed by
                // `delete_orphaned_files`.
                if crate::table_writer::commit_definitely_rolled_back(&error) {
                    written.remove().await;
                }
                return Err(error);
            },
        };
        writer.drop_empty_superseded_inlined_tables()?;
        details.sort_by(|a, b| {
            (&a.flushed.schema_name, &a.flushed.table_name)
                .cmp(&(&b.flushed.schema_name, &b.flushed.table_name))
        });
        Ok(FlushOutcome {
            snapshot_id: Some(snapshot_id),
            tables: details,
        })
    }
}

/// Write one table's inlined rows and inlined file deletions; the returned
/// plan is committed with the others.
async fn flush_table(
    table: &DuckLakeTable,
    writer: &Arc<dyn crate::metadata_writer::MetadataWriter>,
    stores: &FlushObjectStores,
    base_snapshot: i64,
    written: &mut WrittenObjects,
) -> Result<InlinedFlushTable> {
    let schema_name = table
        .schema_name()
        .ok_or_else(|| DuckLakeError::Internal("writable table has no schema name".to_string()))?;
    let provider = table.provider();
    let table_id = table.table_id();
    let mut plan = InlinedFlushTable {
        table_id,
        ..InlinedFlushTable::default()
    };

    let inlined = provider.get_inlined_data_for_flush(table_id, base_snapshot)?;
    let file_deletes = provider.get_inlined_file_deletes_for_flush(table_id, base_snapshot)?;
    if inlined.is_empty() && file_deletes.is_empty() {
        return Ok(plan);
    }
    table.write_options().validate()?;
    // Fail before writing anything when a value cannot reach Parquet.
    for data in &inlined {
        for field in data.rows.batch.schema().fields() {
            ensure_parquet_writable(field.name(), field.data_type())?;
        }
    }
    let object_store = stores.for_table(table.object_store_url().as_ref())?;
    let table_writer = DuckLakeTableWriter::new(Arc::clone(writer), Arc::clone(&object_store))?
        .with_options(table.write_options());
    let mut paths = Vec::new();

    let result = async {
        if !inlined.is_empty() {
            // Official reads the LATEST sort order for every inlined table.
            let sort_keys = current_sort_keys(table)?;
            for data in &inlined {
                let files = write_inlined_table(
                    &table_writer,
                    table,
                    schema_name,
                    data,
                    sort_keys.as_deref(),
                    &mut paths,
                )
                .await?;
                plan.inlined_tables.push(data.rows.table_name.clone());
                plan.files.extend(files);
            }
        }
        plan.file_deletes = write_file_deletes(
            &table_writer,
            table,
            schema_name,
            object_store.as_ref(),
            file_deletes,
            &mut paths,
        )
        .await?;
        Ok::<_, DuckLakeError>(())
    }
    .await;
    written.objects.push((object_store, paths));
    result?;
    Ok(plan)
}

/// A current sort key, resolved against the table's live columns.
enum FlushSortKey {
    /// A bare column, by its column id.
    Column {
        column_id: i64,
        options: SortOptions,
    },
    /// Any other expression, planned against the live columns, with the column
    /// ids it reads.
    Expression {
        expr: Expr,
        column_ids: Vec<i64>,
        options: SortOptions,
    },
}

/// The planner sort-key expressions are evaluated with. Identifiers keep their
/// case, so a key names its column exactly as the catalog does.
fn sort_key_context() -> SessionContext {
    let mut config = SessionConfig::new();
    config.options_mut().sql_parser.enable_ident_normalization = false;
    SessionContext::new_with_config(config)
}

/// The table's live sort keys, or `None` when it is unsorted. Keys of a dialect
/// other than `duckdb` are skipped, as official does.
fn current_sort_keys(table: &DuckLakeTable) -> Result<Option<Vec<FlushSortKey>>> {
    let Some(spec) = table.live_sort_spec()? else {
        return Ok(None);
    };
    let schema = table.physical_schema();
    let ids = table.top_level_column_ids();
    let df_schema = DFSchema::try_from(schema.as_ref().clone())?;
    let context = sort_key_context();
    let mut keys = Vec::with_capacity(spec.fields.len());
    for field in spec
        .fields
        .iter()
        .filter(|field| field.dialect.eq_ignore_ascii_case("duckdb"))
    {
        let options = SortOptions {
            descending: field.direction == SortDirection::Desc,
            nulls_first: field.null_order == NullOrder::NullsFirst,
        };
        if let Some(name) = field.column_candidate() {
            let index = schema.index_of(&name).map_err(|_| {
                DuckLakeError::InvalidConfig(format!(
                    "DuckLake sort key '{name}' is not a column of the table"
                ))
            })?;
            keys.push(FlushSortKey::Column {
                column_id: ids[index],
                options,
            });
            continue;
        }
        let expr = context
            .parse_sql_expr(&field.expression, &df_schema)
            .map_err(|error| {
                DuckLakeError::Unsupported(format!(
                    "DuckLake sort expression '{}' cannot be evaluated: {error}",
                    field.expression
                ))
            })?;
        let column_ids = expr
            .column_refs()
            .iter()
            .map(|column| {
                schema
                    .index_of(column.name())
                    .map(|index| ids[index])
                    .map_err(|_| {
                        DuckLakeError::InvalidConfig(format!(
                            "DuckLake sort expression '{}' names '{}', which is not a column \
                             of the table",
                            field.expression,
                            column.name()
                        ))
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        keys.push(FlushSortKey::Expression {
            expr,
            column_ids,
            options,
        });
    }
    Ok(Some(keys))
}

/// The sort columns of `keys` over `data`, a batch of an inlined table's
/// columns at its schema version (`version_ids`). A key that reads a column the
/// version lacks fails the flush, as official's binder rejects the column.
fn sort_columns(
    keys: &[FlushSortKey],
    data: &RecordBatch,
    version_ids: &[i64],
    table: &DuckLakeTable,
) -> Result<Vec<SortColumn>> {
    let mut columns = Vec::new();
    let mut renamed: Option<RecordBatch> = None;
    for key in keys {
        match key {
            FlushSortKey::Column {
                column_id,
                options,
            } => {
                let index = version_ids
                    .iter()
                    .position(|id| id == column_id)
                    .ok_or_else(|| missing_sort_column(table, &[*column_id], version_ids))?;
                columns.push(SortColumn {
                    values: Arc::clone(data.column(index)),
                    options: Some(*options),
                });
            },
            FlushSortKey::Expression {
                expr,
                column_ids,
                options,
            } => {
                if !column_ids.iter().all(|id| version_ids.contains(id)) {
                    return Err(missing_sort_column(table, column_ids, version_ids));
                }
                if renamed.is_none() {
                    renamed = Some(under_live_names(data, version_ids, table)?);
                }
                let batch = renamed.as_ref().expect("renamed above");
                let df_schema = DFSchema::try_from(batch.schema().as_ref().clone())?;
                let physical = sort_key_context().create_physical_expr(expr.clone(), &df_schema)?;
                let values = physical.evaluate(batch)?.into_array(batch.num_rows())?;
                columns.push(SortColumn {
                    values,
                    options: Some(*options),
                });
            },
        }
    }
    Ok(columns)
}

/// The error for a sort key reading columns (`key_ids`) an inlined table's
/// schema version (`version_ids`) lacks: official's flush fails with the
/// binder's `Referenced column "<name>" not found`.
fn missing_sort_column(
    table: &DuckLakeTable,
    key_ids: &[i64],
    version_ids: &[i64],
) -> DuckLakeError {
    let schema = table.physical_schema();
    let missing = table
        .top_level_column_ids()
        .into_iter()
        .zip(schema.fields().iter())
        .filter(|(id, _)| key_ids.contains(id) && !version_ids.contains(id))
        .map(|(_, field)| format!("\"{}\"", field.name()))
        .collect::<Vec<_>>()
        .join(", ");
    DuckLakeError::InvalidConfig(format!(
        "cannot flush the inlined data of table '{}': its sort order references column {missing}, \
         which inlined rows of an older schema version do not have (Referenced column not found)",
        table.table_name()
    ))
}

/// `data` with each column named as the table's live column of the same column
/// id, so a sort expression written against the live columns reads it. A column
/// dropped since takes a name no expression can reference.
fn under_live_names(
    data: &RecordBatch,
    version_ids: &[i64],
    table: &DuckLakeTable,
) -> Result<RecordBatch> {
    let live_schema = table.physical_schema();
    let live: HashMap<i64, String> = table
        .top_level_column_ids()
        .into_iter()
        .zip(
            live_schema
                .fields()
                .iter()
                .map(|field| field.name().clone()),
        )
        .collect();
    let fields = data
        .schema()
        .fields()
        .iter()
        .zip(version_ids)
        .map(|(field, id)| {
            let name = live
                .get(id)
                .cloned()
                .unwrap_or_else(|| format!("__ducklake_dropped_column_{id}"));
            field.as_ref().clone().with_name(name)
        })
        .collect::<Vec<_>>();
    Ok(RecordBatch::try_new(
        Arc::new(Schema::new(fields)),
        data.columns().to_vec(),
    )?)
}

/// Fail with a typed error when `data_type` contains an interval Arrow cannot
/// write to Parquet.
fn ensure_parquet_writable(column: &str, data_type: &DataType) -> Result<()> {
    let unsupported = match data_type {
        DataType::Interval(arrow::datatypes::IntervalUnit::MonthDayNano) => true,
        DataType::List(field)
        | DataType::LargeList(field)
        | DataType::FixedSizeList(field, _)
        | DataType::Map(field, _) => {
            return ensure_parquet_writable(column, field.data_type());
        },
        DataType::Struct(fields) => {
            for field in fields {
                ensure_parquet_writable(column, field.data_type())?;
            }
            false
        },
        _ => false,
    };
    if unsupported {
        return Err(DuckLakeError::Unsupported(format!(
            "cannot flush inlined column '{column}': Arrow cannot write INTERVAL \
             (Interval(MonthDayNano)) values to Parquet; the rows stay inlined and readable"
        )));
    }
    Ok(())
}

/// The catalog column ids a file of `columns` carries as parquet field ids, in
/// the depth-first order the writer assigns them.
fn field_ids(columns: &[DuckLakeTableColumn]) -> Vec<i64> {
    let mut ids = Vec::new();
    for column in columns {
        ids.push(column.column_id);
        ids.extend_from_slice(&column.nested_column_ids);
    }
    ids
}

/// Write one inlined table's rows: sorted, split by partition, rolled into files
/// at `target_file_size`, each with a delete file for its rows deleted while
/// inlined.
async fn write_inlined_table(
    table_writer: &DuckLakeTableWriter,
    table: &DuckLakeTable,
    schema_name: &str,
    data: &DuckLakeInlinedFlushData,
    sort_keys: Option<&[FlushSortKey]>,
    written: &mut Vec<ObjectPath>,
) -> Result<Vec<InlinedFlushFile>> {
    let batch = &data.rows.batch;
    let rows = batch.num_rows();
    if data.rows.row_ids.len() != rows
        || data.rows.snapshots.len() != rows
        || data.rows.end_snapshots.len() != rows
    {
        return Err(DuckLakeError::Internal(format!(
            "inlined flush rows of '{}' are misaligned with their row identities",
            data.rows.table_name
        )));
    }
    let data_schema = batch.schema();
    let top_level_ids: Vec<i64> = data.columns.iter().map(|c| c.column_id).collect();

    // [data columns..., rowid, snapshot, end_snapshot]
    let mut fields: Vec<Field> = data_schema
        .fields()
        .iter()
        .map(|field| field.as_ref().clone())
        .collect();
    fields.push(Field::new(
        EMBEDDED_ROW_ID_COLUMN_NAME,
        DataType::Int64,
        true,
    ));
    fields.push(Field::new(
        EMBEDDED_SNAPSHOT_ID_COLUMN_NAME,
        DataType::Int64,
        true,
    ));
    fields.push(Field::new("__ducklake_end_snapshot", DataType::Int64, true));
    let full_schema = Arc::new(Schema::new(fields));
    let data_width = data_schema.fields().len();
    let mut columns: Vec<ArrayRef> = batch.columns().to_vec();
    columns.push(Arc::new(Int64Array::from(data.rows.row_ids.clone())));
    columns.push(Arc::new(Int64Array::from(data.rows.snapshots.clone())));
    columns.push(Arc::new(Int64Array::from(data.rows.end_snapshots.clone())));
    let full = RecordBatch::try_new(Arc::clone(&full_schema), columns)?;

    // ORDER BY <sort keys>, row_id ASC NULLS LAST, begin_snapshot ASC NULLS LAST.
    let mut ordering = match sort_keys {
        Some(keys) => sort_columns(keys, batch, &top_level_ids, table)?,
        None => Vec::new(),
    };
    for index in [data_width, data_width + 1] {
        ordering.push(SortColumn {
            values: Arc::clone(full.column(index)),
            options: Some(SortOptions {
                descending: false,
                nulls_first: false,
            }),
        });
    }
    let order: UInt32Array = lexsort_to_indices(&ordering, None)?;
    let sorted = RecordBatch::try_new(
        Arc::clone(&full_schema),
        full.columns()
            .iter()
            .map(|column| take(column.as_ref(), &order, None))
            .collect::<std::result::Result<Vec<_>, _>>()?,
    )?;

    // The partition spec of the inlined table's schema version.
    let spec = table
        .provider()
        .get_partition_spec(table.table_id(), data.columns_snapshot)?;
    let groups: Vec<(Option<FlushPartition>, RecordBatch)> = match spec {
        None => vec![(None, sorted)],
        Some(spec) => {
            let write_spec = PartitionWriteSpec::resolve(&spec, &top_level_ids, &data_schema)?;
            let key_names = write_spec.key_names();
            split_batches_by_partition(&full_schema, &[sorted], &write_spec)?
                .into_iter()
                .map(|(values, batches)| {
                    Ok((
                        Some(FlushPartition {
                            partition_id: write_spec.partition_id,
                            key_names: key_names.clone(),
                            values,
                        }),
                        concat_batches(&full_schema, &batches)?,
                    ))
                })
                .collect::<Result<Vec<_>>>()?
        },
    };

    let data_column_ids = field_ids(&data.columns);
    // Official's flush takes `hive_file_pattern` like an insert does (DuckDB
    // 1.5.6, ducklake 664400c5); an encrypted catalog is refused before this,
    // so the setting alone decides.
    let hive = table.write_options().hive_file_pattern.unwrap_or(true);
    let mut files = Vec::new();
    for (partition, group) in groups {
        if group.num_rows() == 0 {
            continue;
        }
        let subpath = partition
            .as_ref()
            .filter(|_| hive)
            .map(|partition| hive_subpath(&partition.key_names, &partition.values));
        let file_columns = (0..data_width + 2).collect::<Vec<_>>();
        let mut slices = Vec::new();
        let mut offset = 0;
        while offset < group.num_rows() {
            let length = FLUSH_BATCH_ROWS.min(group.num_rows() - offset);
            slices.push(group.slice(offset, length).project(&file_columns)?);
            offset += length;
        }
        let written_files = table_writer
            .write_rolled_lineage_files(
                schema_name,
                table.table_name(),
                &data_schema,
                &data_column_ids,
                &top_level_ids,
                &slices,
                subpath.as_deref(),
                // Official writes a partitioned flush as one file per
                // partition, rolling over at `target_file_size` only an
                // unpartitioned one (`DuckLakeInsert::GetCopyOptions`).
                partition.is_none(),
                written,
            )
            .await?;

        // Each file holds the next `record_count` rows of the group; its delete
        // positions count from its own first row.
        let mut file_start = 0usize;
        for file in written_files {
            let count = usize::try_from(file.record_count).map_err(|_| {
                DuckLakeError::Internal("flushed file has a negative row count".to_string())
            })?;
            let rows = group.slice(file_start, count);
            file_start += count;
            let row_ids = int64(rows.column(data_width))?;
            let snapshots = int64(rows.column(data_width + 1))?;
            let ends = int64(rows.column(data_width + 2))?;
            let (Some(begin), Some(partial_max), Some(min_row_id)) = (
                arrow::compute::min(snapshots),
                arrow::compute::max(snapshots),
                arrow::compute::min(row_ids),
            ) else {
                return Err(DuckLakeError::Internal(format!(
                    "inlined table '{}' has rows without a row id or snapshot",
                    data.rows.table_name
                )));
            };
            let deletions: Vec<(i64, i64)> = (0..rows.num_rows())
                .filter(|row| ends.is_valid(*row))
                .map(|row| (row as i64, ends.value(row)))
                .collect();
            let mut file = file.with_source_row_id_start(Some(min_row_id));
            if let Some(partition) = &partition {
                file = file.with_partition(
                    partition.partition_id,
                    partition
                        .values
                        .iter()
                        .cloned()
                        .enumerate()
                        .map(|(index, value)| (index as i32, value))
                        .collect(),
                );
            }
            let delete = if deletions.is_empty() {
                None
            } else {
                let data_path =
                    resolve_path(table.table_path(), &file.path, file.path_is_relative)?;
                let delete = table_writer
                    .write_delete_file_with_snapshots(
                        schema_name,
                        table.table_name(),
                        &data_path,
                        &deletions,
                        written,
                    )
                    .await?;
                Some(InlinedFlushDeleteFile {
                    delete,
                    begin_snapshot: deletions.iter().map(|(_, s)| *s).min().unwrap_or(begin),
                    partial_max: None,
                    end_snapshot: None,
                })
            };
            files.push(InlinedFlushFile {
                file,
                begin_snapshot: begin,
                partial_max,
                delete,
            });
        }
        if file_start != group.num_rows() {
            return Err(DuckLakeError::Internal(format!(
                "flushed files of '{}' hold {file_start} of {} rows",
                data.rows.table_name,
                group.num_rows()
            )));
        }
    }
    Ok(files)
}

/// The partition one flushed file belongs to.
struct FlushPartition {
    /// The spec generation in effect when the rows were inserted.
    partition_id: i64,
    /// Key column names, for the Hive directory.
    key_names: Vec<String>,
    /// The file's partition values in key order.
    values: Vec<Option<String>>,
}

fn int64(array: &ArrayRef) -> Result<&Int64Array> {
    array
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| DuckLakeError::Internal("inlined flush column is not Int64".to_string()))
}

/// Merge each data file's inlined deletions with its delete file into a new
/// delete file recording every position's snapshot, as official does.
///
/// A data file retired since keeps its history: with no delete file, its
/// deletions go to a new one, as in official. With delete files, they merge
/// into the latest one, the new file ending where it ended, and replace it.
/// Official writes a second delete file that never ends beside the ended one
/// instead, after which time travel to the snapshots the two share reads
/// duplicated rows and resurrects the ended file's deletions. When the merged
/// file would overlap an earlier delete file, or a deletion falls after the
/// latest one ended, the deletions stay inlined, where reads keep applying them.
async fn write_file_deletes(
    table_writer: &DuckLakeTableWriter,
    table: &DuckLakeTable,
    schema_name: &str,
    object_store: &dyn ObjectStore,
    rows: Vec<DuckLakeInlinedFileDeleteForFlush>,
    written: &mut Vec<ObjectPath>,
) -> Result<Vec<InlinedFlushFileDelete>> {
    let mut by_file: BTreeMap<i64, Vec<DuckLakeInlinedFileDeleteForFlush>> = BTreeMap::new();
    for row in rows {
        by_file.entry(row.data_file_id).or_default().push(row);
    }
    let mut out = Vec::with_capacity(by_file.len());
    for (data_file_id, rows) in by_file {
        let first = &rows[0];
        let encrypted = first.data_file.encryption_key.is_some()
            || first
                .existing_delete
                .as_ref()
                .is_some_and(|(_, file, _)| file.encryption_key.is_some());
        if encrypted {
            tracing::warn!(
                data_file_id,
                "leaving inlined deletions of an encrypted file inlined: writing encrypted \
                 delete files is not supported"
            );
            continue;
        }
        // position -> snapshot that deleted it (the earliest, should both record it)
        let mut deletions: BTreeMap<i64, i64> = BTreeMap::new();
        for row in &rows {
            deletions
                .entry(row.row_id)
                .and_modify(|snapshot| *snapshot = (*snapshot).min(row.begin_snapshot))
                .or_insert(row.begin_snapshot);
        }
        let inlined_last = deletions.values().copied().max().unwrap_or_default();
        let expected_prev = match &first.existing_delete {
            Some((delete_file_id, file, begin_snapshot)) => {
                let path = resolve_path(table.table_path(), &file.path, file.path_is_relative)?;
                for (position, snapshot) in
                    read_delete_file(object_store, &path, *begin_snapshot).await?
                {
                    deletions
                        .entry(position)
                        .and_modify(|current| *current = (*current).min(snapshot))
                        .or_insert(snapshot);
                }
                Some(*delete_file_id)
            },
            None => None,
        };
        let deletions: Vec<(i64, i64)> = deletions.into_iter().collect();
        let begin_snapshot = deletions.iter().map(|(_, s)| *s).min().unwrap_or_default();
        let partial_max = deletions.iter().map(|(_, s)| *s).max();
        // The merged file replaces a retired file's latest delete file only
        // where it alone covers the file's history.
        let end_snapshot = match (first.data_file_end, &first.existing_delete) {
            (Some(_), Some(_)) => {
                let overlaps_earlier = first
                    .earlier_delete_end
                    .is_some_and(|end| end > begin_snapshot);
                let outlives_end = first
                    .existing_delete_end
                    .is_some_and(|end| inlined_last >= end);
                if overlaps_earlier || outlives_end {
                    tracing::warn!(
                        data_file_id,
                        "leaving inlined deletions of a retired data file inlined: they cannot \
                         merge into its delete file history"
                    );
                    continue;
                }
                first.existing_delete_end
            },
            _ => None,
        };
        let data_path = resolve_path(
            table.table_path(),
            &first.data_file.path,
            first.data_file.path_is_relative,
        )?;
        let delete = table_writer
            .write_delete_file_with_snapshots(
                schema_name,
                table.table_name(),
                &data_path,
                &deletions,
                written,
            )
            .await?;
        out.push(InlinedFlushFileDelete {
            data_file_id,
            data_file_end: first.data_file_end,
            expected_prev_delete_file: expected_prev,
            delete: InlinedFlushDeleteFile {
                delete,
                begin_snapshot,
                partial_max,
                end_snapshot,
            },
        });
    }
    Ok(out)
}

/// Read a delete file's positions with the snapshot that deleted each: its
/// `_ducklake_internal_snapshot_id` when it records one, else `begin_snapshot`
/// (the snapshot the file was registered at).
async fn read_delete_file(
    object_store: &dyn ObjectStore,
    path: &str,
    begin_snapshot: i64,
) -> Result<Vec<(i64, i64)>> {
    let bytes = object_store
        .get(&ObjectPath::from(path.trim_start_matches('/')))
        .await?
        .bytes()
        .await?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(bytes)?.build()?;
    let mut out = Vec::new();
    for batch in reader {
        let batch = batch?;
        let schema = batch.schema();
        let positions = int64(batch.column(schema.index_of(DELETE_POS_COL)?))?;
        let snapshots = match schema.index_of(DELETE_SNAPSHOT_COL) {
            Ok(index) => Some(int64(batch.column(index))?),
            Err(_) => None,
        };
        for row in 0..batch.num_rows() {
            if positions.is_null(row) {
                continue;
            }
            let snapshot = snapshots
                .filter(|snapshots| snapshots.is_valid(row))
                .map_or(begin_snapshot, |snapshots| snapshots.value(row));
            out.push((positions.value(row), snapshot));
        }
    }
    Ok(out)
}

/// `ducklake_flush_inlined_data(catalog)`: the table function form of
/// [`DuckLakeCatalog::flush_inlined_data`], registered by
/// [`register_ducklake_maintenance_functions`](crate::register_ducklake_maintenance_functions)
/// only: it writes to the catalog although DataFusion plans it as a read.
///
/// `SELECT * FROM ducklake_flush_inlined_data('lake')` flushes every table of
/// the DuckLake catalog registered as `lake` in the calling session when the
/// query executes, returning `schema_name`, `table_name`, `rows_flushed`.
/// DataFusion hands a table function its arguments without their names, so
/// `schema_name =>` and `table_name =>` cannot be told apart and the function
/// takes only the catalog name, refusing more arguments; to restrict the flush
/// to a schema or table, run
/// `CALL ducklake_flush_inlined_data('lake', schema_name => 'main', table_name => 't')`
/// through [`execute_ducklake_sql`](crate::execute_ducklake_sql).
#[derive(Debug, Default)]
pub struct DucklakeFlushInlinedDataFunction;

impl TableFunctionImpl for DucklakeFlushInlinedDataFunction {
    fn call_with_args(&self, args: TableFunctionArgs) -> DataFusionResult<Arc<dyn TableProvider>> {
        let catalog = match args.exprs() {
            [Expr::Literal(ScalarValue::Utf8(Some(name)), _)]
            | [Expr::Literal(ScalarValue::LargeUtf8(Some(name)), _)]
            | [Expr::Literal(ScalarValue::Utf8View(Some(name)), _)] => name.clone(),
            _ => {
                return plan_err!(
                    "ducklake_flush_inlined_data() takes one argument, the catalog name; \
                     use CALL ducklake_flush_inlined_data(...) through execute_ducklake_sql \
                     for schema_name / table_name"
                );
            },
        };
        let state = args
            .session()
            .as_any()
            .downcast_ref::<SessionState>()
            .ok_or_else(|| {
                DataFusionError::Plan(
                    "ducklake_flush_inlined_data() needs a SessionState session".to_string(),
                )
            })?
            .clone();
        // Resolve now, so an unknown catalog fails at planning.
        resolve_ducklake_catalog(&state, &catalog)?;
        Ok(Arc::new(FlushInlinedDataTable {
            state,
            catalog,
            schema_name: None,
            table_name: None,
        }))
    }
}

/// Resolve `name` to a [`DuckLakeCatalog`] registered in `state`, matched
/// case-insensitively as DuckDB matches a database name (an exact match
/// first).
pub(crate) fn resolve_ducklake_catalog(
    state: &SessionState,
    name: &str,
) -> DataFusionResult<Arc<dyn CatalogProvider>> {
    let list = state.catalog_list();
    let found = list.catalog(name).or_else(|| {
        list.catalog_names()
            .into_iter()
            .find(|registered| registered.eq_ignore_ascii_case(name))
            .and_then(|registered| list.catalog(&registered))
    });
    let catalog = found.ok_or_else(|| {
        DataFusionError::Plan(format!(
            "catalog '{name}' is not registered in this session"
        ))
    })?;
    if catalog.downcast_ref::<DuckLakeCatalog>().is_none() {
        return plan_err!("catalog '{name}' is not a DuckLake catalog");
    }
    Ok(catalog)
}

/// The relation a flush returns; the flush runs when it is scanned and executed.
#[derive(Debug)]
pub(crate) struct FlushInlinedDataTable {
    pub(crate) state: SessionState,
    pub(crate) catalog: String,
    pub(crate) schema_name: Option<String>,
    pub(crate) table_name: Option<String>,
}

#[async_trait::async_trait]
impl TableProvider for FlushInlinedDataTable {
    fn schema(&self) -> SchemaRef {
        flush_inlined_data_schema()
    }

    fn table_type(&self) -> TableType {
        TableType::Temporary
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let projection = projection.cloned();
        let schema = match &projection {
            Some(indices) => Arc::new(flush_inlined_data_schema().project(indices)?),
            None => flush_inlined_data_schema(),
        };
        Ok(Arc::new(FlushInlinedDataExec {
            state: self.state.clone(),
            catalog: self.catalog.clone(),
            schema_name: self.schema_name.clone(),
            table_name: self.table_name.clone(),
            projection,
            properties: Arc::new(PlanProperties::new(
                EquivalenceProperties::new(Arc::clone(&schema)),
                Partitioning::UnknownPartitioning(1),
                EmissionType::Final,
                Boundedness::Bounded,
            )),
            schema,
        }))
    }
}

/// Runs the flush on execution and emits its result rows.
#[derive(Debug)]
struct FlushInlinedDataExec {
    state: SessionState,
    catalog: String,
    schema_name: Option<String>,
    table_name: Option<String>,
    projection: Option<Vec<usize>>,
    schema: SchemaRef,
    properties: Arc<PlanProperties>,
}

impl DisplayAs for FlushInlinedDataExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "FlushInlinedDataExec: catalog={}", self.catalog)
    }
}

impl ExecutionPlan for FlushInlinedDataExec {
    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> DataFusionResult<TreeNodeRecursion>,
    ) -> DataFusionResult<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn name(&self) -> &str {
        "FlushInlinedDataExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        if !children.is_empty() {
            return Err(DataFusionError::Internal(
                "FlushInlinedDataExec takes no children".to_string(),
            ));
        }
        Ok(self)
    }

    fn execute(
        &self,
        partition: usize,
        _context: Arc<TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(DataFusionError::Internal(format!(
                "FlushInlinedDataExec has one partition, asked for {partition}"
            )));
        }
        let state = self.state.clone();
        let catalog = self.catalog.clone();
        let schema_name = self.schema_name.clone();
        let table_name = self.table_name.clone();
        let projection = self.projection.clone();
        let stream = futures::stream::once(async move {
            let provider = resolve_ducklake_catalog(&state, &catalog)?;
            let catalog = provider
                .downcast_ref::<DuckLakeCatalog>()
                .ok_or_else(|| DataFusionError::Internal("not a DuckLake catalog".into()))?;
            let flushed = catalog
                .flush_inlined_data(&state, schema_name.as_deref(), table_name.as_deref())
                .await?;
            let batch = flushed_inlined_tables_batch(&flushed)?;
            Ok(match &projection {
                Some(indices) => batch.project(indices)?,
                None => batch,
            })
        });
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&self.schema),
            stream,
        )))
    }
}
