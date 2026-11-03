//! DuckLake `MERGE INTO` planning and atomic execution.
//!
//! The statement follows official DuckLake's `MERGE` (`ducklake_merge_into.cpp`
//! on DuckDB's `bind_merge_into.cpp`):
//!
//! - each matched pair of target and source rows takes the first `WHEN
//!   MATCHED` clause whose condition holds, each target row without a source
//!   match the first `WHEN NOT MATCHED BY SOURCE` clause, and each source row
//!   without a target match the first `WHEN NOT MATCHED` clause;
//! - all of these clauses together hold at most one `UPDATE` or `DELETE`
//!   action, as official DuckLake allows; any number of `INSERT` and `DO
//!   NOTHING` actions are accepted;
//! - a target row matched by several source rows is updated with the first
//!   of them in source order, or deleted once, and every matched pair counts
//!   in the affected-row count. Official keeps the first version to reach its
//!   update, which follows its join plan: the first source row when the
//!   source is the larger side, often the last otherwise;
//! - a bare `UPDATE` assigns the source columns by position and a bare
//!   `INSERT` inserts them by position; an `INSERT` column list fills the
//!   omitted columns with their defaults.
//!
//! The changes commit in one snapshot through the `UPDATE` commit path
//! ([`commit_row_versions`]): updated rows keep their row ids and are inlined
//! under the scoped `data_inlining_row_limit`, while inserted rows draw new
//! row ids and go to Parquet, as official inlines only the `UPDATE` action.

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Int64Array, UInt64Array};
use arrow::compute::{concat_batches, take_record_batch};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use datafusion::catalog::{CatalogProvider, TableProvider};
use datafusion::common::{Column, DFSchema, ScalarValue, TableReference};
use datafusion::datasource::MemTable;
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::execution::SessionStateBuilder;
use datafusion::logical_expr::conditional_expressions::CaseBuilder;
use datafusion::logical_expr::{Expr as DFExpr, JoinType, cast, lit, when};
use datafusion::prelude::{DataFrame, SessionContext, col};
use datafusion::sql::sqlparser::ast::{
    AssignmentTarget, Expr, Ident, Merge, MergeAction, MergeClauseKind, MergeInsertExpr,
    MergeInsertKind, ObjectName, TableFactor,
};

use crate::catalog::DuckLakeCatalog;
use crate::delete_exec::DeletionTarget;
use crate::metadata_writer::InlinedRowRef;
use crate::row_id::{ROWID_COLUMN_NAME, rowid_field};
use crate::rowid_lookup::RowLocation;
use crate::table::DuckLakeTable;
use crate::table_writer::DuckLakeTableWriter;
use crate::update_exec::{RowVersionCommit, commit_row_versions};

/// The assignment target the SQL hook gives a bare `UPDATE`
/// (`SET "<sentinel>" = "<sentinel>"`), which assigns every column by position.
pub(crate) const MERGE_UPDATE_ALL_SENTINEL: &str = "__ducklake_merge_all__";
/// The column the SQL hook gives a `DO NOTHING` action: an `UPDATE` assigning
/// it, or an `INSERT` naming it, does nothing.
pub(crate) const MERGE_DO_NOTHING_SENTINEL: &str = "__ducklake_merge_nothing__";
/// Each source row's position in the source, so a target row matched by
/// several source rows takes the first one's values.
const SOURCE_ROW_COLUMN: &str = "__ducklake_merge_source_row";
/// The index of the clause a row takes, or NULL when it takes none.
const CLAUSE_COLUMN: &str = "__ducklake_merge_clause";

const SINGLE_ACTION_ERROR: &str =
    "MERGE INTO with DuckLake only supports a single UPDATE/DELETE action currently";

/// What a `WHEN` clause does.
enum Action {
    /// `SET` assignments `(column, value)`, or every column by position when
    /// `None` (a bare `UPDATE`).
    Update(Option<Vec<(String, Expr)>>),
    Delete,
    Insert(Box<MergeInsertExpr>),
    Nothing,
}

struct Clause {
    predicate: Option<Expr>,
    action: Action,
}

/// The statement's clauses by the rows they apply to, in statement order.
struct Clauses {
    matched: Vec<Clause>,
    by_source: Vec<Clause>,
    not_matched: Vec<Clause>,
}

impl Clauses {
    fn mutates(&self) -> bool {
        self.matched
            .iter()
            .chain(&self.by_source)
            .any(|clause| matches!(clause.action, Action::Update(_) | Action::Delete))
    }
}

/// The target rows the statement updates or deletes.
#[derive(Default)]
struct Superseded {
    /// Their row ids.
    rowids: Vec<i64>,
    /// The new versions, `[physical columns..., rowid]`; empty for a delete.
    versions: Vec<RecordBatch>,
    /// The rows the statement counts: every matched pair taking the action.
    counted: u64,
}

/// Execute one parsed DuckLake MERGE and return its affected-row count.
pub(crate) async fn execute_merge(
    ctx: &SessionContext,
    catalog: &DuckLakeCatalog,
    merge: Merge,
) -> DataFusionResult<DataFrame> {
    let (target_name, target_alias) = target_factor(&merge.table)?;
    let (schema_name, table_name) = resolve_schema_table(&target_name)?;
    let source_alias = factor_alias(&merge.source).ok_or_else(|| {
        DataFusionError::Plan("MERGE source must have a table name or alias".to_string())
    })?;
    let clauses = classify_clauses(&merge)?;

    let writer = catalog.writer().ok_or_else(|| {
        DataFusionError::Plan(
            "catalog is read-only; open it with DuckLakeCatalog::with_writer to run MERGE"
                .to_string(),
        )
    })?;
    if clauses.mutates() && !writer.supports_inlined_row_mutations() {
        return Err(DataFusionError::NotImplemented(
            "MERGE INTO with an UPDATE or DELETE action needs a writer that stages row \
             mutations (SQLite, DuckDB, MySQL or multi-catalog PostgreSQL)"
                .to_string(),
        ));
    }
    let table = target_table(catalog, &schema_name, &table_name).await?;
    let physical_schema = table.physical_schema();

    let target = ctx
        .read_table(Arc::new(table.clone().with_row_lineage(true)))?
        .alias(&target_alias)?;
    let (source, source_columns) = numbered_source(ctx, &merge.source, &source_alias).await?;
    let join_schema = target.schema().join(source.schema())?;
    let on = logical_expr(ctx, merge.on.as_ref(), &join_schema)?;

    let mut superseded = Superseded::default();
    if let Some(index) = action_index(&clauses.matched) {
        let joined = target
            .clone()
            .join_on(source.clone(), JoinType::Inner, [on.clone()])?;
        let joined = taking_clause(ctx, joined, &clauses.matched, index)?;
        superseded = match &clauses.matched[index].action {
            Action::Update(assignments) => {
                let projection = update_projection(
                    ctx,
                    joined.schema(),
                    physical_schema.as_ref(),
                    &target_alias,
                    Some((&source_alias, &source_columns)),
                    assignments.as_deref(),
                )?;
                first_versions(
                    joined.select(projection)?.collect().await?,
                    &physical_schema,
                )?
            },
            _ => {
                let batches = joined
                    .select([
                        qualified_col(&target_alias, ROWID_COLUMN_NAME).alias(ROWID_COLUMN_NAME)
                    ])?
                    .collect()
                    .await?;
                deleted_rows(&batches)?
            },
        };
    }
    if let Some(index) = action_index(&clauses.by_source) {
        let unmatched = target
            .clone()
            .join_on(source.clone(), JoinType::LeftAnti, [on.clone()])?;
        let unmatched = taking_clause(ctx, unmatched, &clauses.by_source, index)?;
        superseded = match &clauses.by_source[index].action {
            Action::Update(assignments) => {
                let projection = update_projection(
                    ctx,
                    unmatched.schema(),
                    physical_schema.as_ref(),
                    &target_alias,
                    None,
                    assignments.as_deref(),
                )?;
                let batches = unmatched.select(projection)?.collect().await?;
                let counted = batches.iter().map(|batch| batch.num_rows() as u64).sum();
                Superseded {
                    rowids: batch_rowids(&batches)?,
                    versions: coerce_output_batches(&batches, &physical_schema, true)?,
                    counted,
                }
            },
            _ => {
                let batches = unmatched
                    .select([
                        qualified_col(&target_alias, ROWID_COLUMN_NAME).alias(ROWID_COLUMN_NAME)
                    ])?
                    .collect()
                    .await?;
                deleted_rows(&batches)?
            },
        };
    }

    let inserted = if clauses
        .not_matched
        .iter()
        .any(|clause| matches!(clause.action, Action::Insert(_)))
    {
        let unmatched = source.join_on(target, JoinType::LeftAnti, [on])?;
        let projection = insert_projection(
            ctx,
            &unmatched,
            &table,
            &source_alias,
            &source_columns,
            &clauses.not_matched,
        )?;
        let batches = unmatched
            .select(projection)?
            .filter(col(CLAUSE_COLUMN).is_not_null())?
            .sort(vec![
                col(CLAUSE_COLUMN).sort(true, false),
                col(SOURCE_ROW_COLUMN).sort(true, false),
            ])?
            .collect()
            .await?;
        rows_by_clause(&batches, &physical_schema)?
    } else {
        Vec::new()
    };
    let inserted_count: u64 = inserted
        .iter()
        .flatten()
        .map(|batch| batch.num_rows() as u64)
        .sum();
    let affected = superseded.counted + inserted_count;
    if affected == 0 {
        return count_frame(ctx, 0);
    }

    let state = ctx.state();
    let object_store = ctx
        .runtime_env()
        .object_store(table.object_store_url().as_ref())?;
    let inline_capable = writer.supports_inlined_row_mutations();
    let table_writer = DuckLakeTableWriter::new(writer, object_store)
        .map(|writer| writer.with_options(table.write_options()))
        .map_err(DataFusionError::from)?;
    if !superseded.rowids.is_empty() && superseded.versions.is_empty() && inserted.is_empty() {
        table
            .delete_rowids(&state, &superseded.rowids)
            .await
            .map_err(DataFusionError::from)?;
        return count_frame(ctx, affected);
    }

    let mut positions: HashMap<i64, Vec<i64>> = HashMap::new();
    let mut superseded_inlined = Vec::new();
    let located = table
        .locate_rowids(&state, &superseded.rowids)
        .await
        .map_err(DataFusionError::from)?;
    if located.len() != superseded.rowids.len() {
        return Err(DataFusionError::Execution(format!(
            "MERGE located {} of its {} matched rows",
            located.len(),
            superseded.rowids.len()
        )));
    }
    for row in located {
        match row.location {
            RowLocation::Inlined {
                table_name,
            } => superseded_inlined.push(InlinedRowRef {
                table_name,
                row_id: row.row_id,
            }),
            RowLocation::File {
                data_file_id,
                position,
                ..
            } => positions.entry(data_file_id).or_default().push(position),
        }
    }
    let mut files = Vec::with_capacity(positions.len());
    for file in table.files().map_err(DataFusionError::from)? {
        if let Some(mut file_positions) = positions.remove(&file.data_file_id) {
            // A delete file lists its positions in order: official refuses
            // to read one that does not.
            file_positions.sort_unstable();
            let existing = match &file.delete_file {
                Some(delete_file) => table.read_delete_file_entries(&state, delete_file).await?,
                None => HashMap::new(),
            };
            files.push((file, existing, file_positions));
        }
    }
    let superseded_positions = files
        .iter()
        .map(|(file, existing, file_positions)| {
            (
                DeletionTarget::of_file(file, existing),
                file_positions.clone(),
            )
        })
        .collect();

    // A statement updating or deleting rows was refused above on a writer
    // that cannot stage those; one that only inserts runs on any writer.
    let inline_limit = if inline_capable {
        table_writer
            .table_data_inlining_row_limit(&schema_name, &table_name)
            .map_err(DataFusionError::from)?
    } else {
        0
    };
    commit_row_versions(
        &RowVersionCommit {
            table: &table,
            table_writer: &table_writer,
            context: ctx.task_ctx(),
            schema_name: &schema_name,
            table_name: &table_name,
            base_snapshot: table.snapshot_id(),
            inline_capable,
            inline_limit,
        },
        superseded.versions,
        inserted,
        superseded_positions,
        Vec::new(),
        superseded_inlined,
    )
    .await?;
    count_frame(ctx, affected)
}

/// The writable DuckLake table `schema_name.table_name` of `catalog`.
async fn target_table(
    catalog: &DuckLakeCatalog,
    schema_name: &str,
    table_name: &str,
) -> DataFusionResult<DuckLakeTable> {
    let provider = catalog
        .schema(schema_name)
        .ok_or_else(|| DataFusionError::Plan(format!("schema '{schema_name}' not found")))?
        .table(table_name)
        .await?
        .ok_or_else(|| {
            DataFusionError::Plan(format!("table '{schema_name}.{table_name}' not found"))
        })?;
    provider
        .downcast_ref::<DuckLakeTable>()
        .cloned()
        .ok_or_else(|| {
            DataFusionError::NotImplemented(format!(
                "MERGE target '{schema_name}.{table_name}' is not a DuckLake table"
            ))
        })
}

/// The source rows, read once, each with its position in the source
/// ([`SOURCE_ROW_COLUMN`]), and the names of the source's own columns.
/// The source is read in one partition, so the rows keep their source order.
async fn numbered_source(
    ctx: &SessionContext,
    source: &TableFactor,
    alias: &str,
) -> DataFusionResult<(DataFrame, Vec<String>)> {
    let state = ctx.state();
    let config = state.config().clone().with_target_partitions(1);
    let ordered = SessionContext::new_with_state(
        SessionStateBuilder::new_from_existing(state)
            .with_config(config)
            .build(),
    );
    let frame = ordered.sql(&format!("SELECT * FROM {source}")).await?;
    let source_schema: SchemaRef = Arc::new(frame.schema().as_arrow().clone());
    let columns = source_schema
        .fields()
        .iter()
        .map(|field| field.name().clone())
        .collect();
    let mut fields: Vec<Arc<Field>> = source_schema.fields().iter().cloned().collect();
    fields.push(Arc::new(Field::new(
        SOURCE_ROW_COLUMN,
        DataType::UInt64,
        false,
    )));
    let schema = Arc::new(Schema::new(fields));
    let mut next = 0u64;
    let mut numbered = Vec::new();
    for batch in frame.collect().await? {
        let rows = batch.num_rows() as u64;
        let mut columns = batch.columns().to_vec();
        columns.push(Arc::new(UInt64Array::from_iter_values(next..next + rows)));
        numbered.push(RecordBatch::try_new(Arc::clone(&schema), columns)?);
        next += rows;
    }
    let table = MemTable::try_new(schema, vec![numbered])?;
    Ok((ctx.read_table(Arc::new(table))?.alias(alias)?, columns))
}

fn classify_clauses(merge: &Merge) -> DataFusionResult<Clauses> {
    if merge.output.is_some() {
        return Err(DataFusionError::NotImplemented(
            "MERGE OUTPUT/RETURNING is not supported".to_string(),
        ));
    }
    let mut clauses = Clauses {
        matched: Vec::new(),
        by_source: Vec::new(),
        not_matched: Vec::new(),
    };
    for clause in &merge.clauses {
        let action = match &clause.action {
            MergeAction::Update(update) => {
                if update.update_predicate.is_some() || update.delete_predicate.is_some() {
                    return Err(DataFusionError::NotImplemented(
                        "MERGE UPDATE WHERE/DELETE WHERE is not supported; use WHEN MATCHED AND"
                            .to_string(),
                    ));
                }
                let mut assignments = Vec::with_capacity(update.assignments.len());
                for assignment in &update.assignments {
                    let AssignmentTarget::ColumnName(name) = &assignment.target else {
                        return Err(DataFusionError::NotImplemented(
                            "tuple assignments are not supported in MERGE UPDATE".to_string(),
                        ));
                    };
                    let column = object_last_name(name).ok_or_else(|| {
                        DataFusionError::Plan(
                            "MERGE UPDATE assignment has no column name".to_string(),
                        )
                    })?;
                    assignments.push((column, assignment.value.clone()));
                }
                match assignments.as_slice() {
                    [(column, _)] if column == MERGE_DO_NOTHING_SENTINEL => Action::Nothing,
                    [(column, _)] if column == MERGE_UPDATE_ALL_SENTINEL => Action::Update(None),
                    _ => Action::Update(Some(assignments)),
                }
            },
            MergeAction::Delete {
                ..
            } => Action::Delete,
            MergeAction::Insert(insert) => {
                if insert.insert_predicate.is_some() {
                    return Err(DataFusionError::NotImplemented(
                        "MERGE INSERT WHERE is not supported; use WHEN NOT MATCHED AND".to_string(),
                    ));
                }
                let nothing = matches!(
                    insert.columns.as_slice(),
                    [name] if object_last_name(name).as_deref() == Some(MERGE_DO_NOTHING_SENTINEL)
                );
                if nothing {
                    Action::Nothing
                } else {
                    Action::Insert(Box::new(insert.clone()))
                }
            },
        };
        let kind = &clause.clause_kind;
        let clause = Clause {
            predicate: clause.predicate.clone(),
            action,
        };
        match kind {
            MergeClauseKind::Matched => clauses.matched.push(clause),
            MergeClauseKind::NotMatchedBySource => {
                if matches!(clause.action, Action::Update(None)) {
                    return Err(DataFusionError::Plan(
                        "a bare UPDATE assigns the source row's values, and WHEN NOT MATCHED BY \
                         SOURCE has no source row; name the assignments with SET"
                            .to_string(),
                    ));
                }
                clauses.by_source.push(clause);
            },
            MergeClauseKind::NotMatched | MergeClauseKind::NotMatchedByTarget => {
                clauses.not_matched.push(clause);
            },
        }
    }
    if clauses.matched.is_empty() && clauses.by_source.is_empty() && clauses.not_matched.is_empty()
    {
        return Err(DataFusionError::Plan(
            "MERGE requires at least one WHEN clause".to_string(),
        ));
    }
    let actions = clauses
        .matched
        .iter()
        .chain(&clauses.by_source)
        .filter(|clause| matches!(clause.action, Action::Update(_) | Action::Delete))
        .count();
    if actions > 1 {
        return Err(DataFusionError::NotImplemented(
            SINGLE_ACTION_ERROR.to_string(),
        ));
    }
    Ok(clauses)
}

/// The index of the clause of `clauses` that updates or deletes, if any.
fn action_index(clauses: &[Clause]) -> Option<usize> {
    clauses
        .iter()
        .position(|clause| matches!(clause.action, Action::Update(_) | Action::Delete))
}

/// The rows of `frame` that take clause `index` of `clauses`: those whose
/// first clause with a holding condition is that one.
fn taking_clause(
    ctx: &SessionContext,
    frame: DataFrame,
    clauses: &[Clause],
    index: usize,
) -> DataFusionResult<DataFrame> {
    let chosen = clause_choice(ctx, frame.schema(), clauses)?;
    frame.filter(chosen.eq(lit(index as i64)))
}

/// The index of the first of `clauses` whose condition holds for a row of a
/// frame of `schema`, or NULL when none does.
fn clause_choice(
    ctx: &SessionContext,
    schema: &DFSchema,
    clauses: &[Clause],
) -> DataFusionResult<DFExpr> {
    let mut choice: Option<CaseBuilder> = None;
    for (index, clause) in clauses.iter().enumerate() {
        let condition = match &clause.predicate {
            Some(predicate) => logical_expr(ctx, predicate, schema)?,
            None => lit(true),
        };
        let value = lit(index as i64);
        choice = Some(match choice {
            None => when(condition, value),
            Some(mut builder) => {
                builder.when(condition, value);
                builder
            },
        });
    }
    match choice {
        Some(mut builder) => builder.otherwise(lit(ScalarValue::Int64(None))),
        None => Ok(lit(ScalarValue::Int64(None))),
    }
}

fn target_factor(table: &TableFactor) -> DataFusionResult<(ObjectName, String)> {
    let TableFactor::Table {
        name,
        alias,
        args,
        with_hints,
        version,
        partitions,
        ..
    } = table
    else {
        return Err(DataFusionError::NotImplemented(
            "MERGE target must be a named DuckLake table".to_string(),
        ));
    };
    if args.is_some() || !with_hints.is_empty() || version.is_some() || !partitions.is_empty() {
        return Err(DataFusionError::NotImplemented(
            "MERGE target table arguments, hints, versions, and partitions are not supported"
                .to_string(),
        ));
    }
    let alias = alias
        .as_ref()
        .map(|a| normalize_ident(&a.name))
        .or_else(|| object_last_name(name))
        .ok_or_else(|| DataFusionError::Plan("MERGE target has no table name".to_string()))?;
    Ok((name.clone(), alias))
}

fn factor_alias(factor: &TableFactor) -> Option<String> {
    match factor {
        TableFactor::Table {
            name,
            alias,
            ..
        } => alias
            .as_ref()
            .map(|a| normalize_ident(&a.name))
            .or_else(|| object_last_name(name)),
        TableFactor::Derived {
            alias,
            ..
        } => alias.as_ref().map(|a| normalize_ident(&a.name)),
        _ => None,
    }
}

fn resolve_schema_table(name: &ObjectName) -> DataFusionResult<(String, String)> {
    let parts: Vec<String> = name
        .0
        .iter()
        .filter_map(|part| part.as_ident())
        .map(normalize_ident)
        .collect();
    match parts.as_slice() {
        [table] => Ok(("main".to_string(), table.clone())),
        [schema, table] => Ok((schema.clone(), table.clone())),
        [_catalog, schema, table] => Ok((schema.clone(), table.clone())),
        _ => Err(DataFusionError::Plan(
            "MERGE target must be a table name of 1-3 parts".to_string(),
        )),
    }
}

fn object_last_name(name: &ObjectName) -> Option<String> {
    name.0
        .last()
        .and_then(|part| part.as_ident())
        .map(normalize_ident)
}

fn normalize_ident(ident: &Ident) -> String {
    if ident.quote_style.is_some() {
        ident.value.clone()
    } else {
        ident.value.to_ascii_lowercase()
    }
}

/// The column of `schema` that `name` names: exactly, else ignoring case when
/// one column matches so, as DuckDB resolves an identifier.
fn resolve_column<'a>(schema: &'a Schema, name: &str) -> Option<&'a str> {
    if let Ok(field) = schema.field_with_name(name) {
        return Some(field.name());
    }
    let mut matches = schema
        .fields()
        .iter()
        .filter(|field| field.name().eq_ignore_ascii_case(name));
    match (matches.next(), matches.next()) {
        (Some(field), None) => Some(field.name()),
        _ => None,
    }
}

fn logical_expr(ctx: &SessionContext, expr: &Expr, schema: &DFSchema) -> DataFusionResult<DFExpr> {
    ctx.state().create_logical_expr(&expr.to_string(), schema)
}

fn qualified_col(alias: &str, name: &str) -> DFExpr {
    DFExpr::Column(Column::new(
        Some(TableReference::bare(alias.to_string())),
        name,
    ))
}

/// `expr` cast to `data_type`, failing on a value that does not convert as
/// an `INSERT` of it fails.
fn cast_to(expr: DFExpr, data_type: &DataType) -> DFExpr {
    cast(expr, data_type.clone())
}

/// The new version of each row of a frame of `schema`, `[physical columns...,
/// rowid]` plus, for matched rows, the source row's position: the `SET`
/// assignments over the target's values, or with `assignments` `None` (a bare
/// `UPDATE`) the source's columns by position.
fn update_projection(
    ctx: &SessionContext,
    schema: &DFSchema,
    target_schema: &Schema,
    target_alias: &str,
    source: Option<(&str, &[String])>,
    assignments: Option<&[(String, Expr)]>,
) -> DataFusionResult<Vec<DFExpr>> {
    let mut values: Vec<Option<DFExpr>> = vec![None; target_schema.fields().len()];
    match assignments {
        Some(assignments) => {
            for (column, value) in assignments {
                let name = resolve_column(target_schema, column).ok_or_else(|| {
                    DataFusionError::Plan(format!("MERGE UPDATE targets unknown column '{column}'"))
                })?;
                let index = target_schema.index_of(name)?;
                if values[index].is_some() {
                    return Err(DataFusionError::Plan(format!(
                        "MERGE UPDATE assigns column '{name}' more than once"
                    )));
                }
                values[index] = Some(logical_expr(ctx, value, schema)?);
            }
        },
        None => {
            let (source_alias, source_columns) = source.ok_or_else(|| {
                DataFusionError::Internal("a bare UPDATE needs a source row".to_string())
            })?;
            if source_columns.len() != target_schema.fields().len() {
                return Err(DataFusionError::Plan(format!(
                    "Data provided for UPDATE did not match column count in table - expected \
                     {} columns but got {}",
                    target_schema.fields().len(),
                    source_columns.len()
                )));
            }
            for (value, name) in values.iter_mut().zip(source_columns) {
                *value = Some(qualified_col(source_alias, name));
            }
        },
    }
    let mut projection = Vec::with_capacity(target_schema.fields().len() + 2);
    for (field, value) in target_schema.fields().iter().zip(values) {
        let value = value.unwrap_or_else(|| qualified_col(target_alias, field.name()));
        projection.push(cast_to(value, field.data_type()).alias(field.name()));
    }
    projection.push(qualified_col(target_alias, ROWID_COLUMN_NAME).alias(ROWID_COLUMN_NAME));
    if let Some((source_alias, _)) = source {
        projection.push(qualified_col(source_alias, SOURCE_ROW_COLUMN).alias(SOURCE_ROW_COLUMN));
    }
    Ok(projection)
}

/// The row each unmatched source row inserts, `[physical columns...]`, plus
/// the clause it takes ([`CLAUSE_COLUMN`], NULL when it takes an `INSERT`
/// action of none) and its position in the source.
fn insert_projection(
    ctx: &SessionContext,
    unmatched: &DataFrame,
    table: &DuckLakeTable,
    source_alias: &str,
    source_columns: &[String],
    clauses: &[Clause],
) -> DataFusionResult<Vec<DFExpr>> {
    let schema = unmatched.schema();
    let target_schema = table.physical_schema();
    let chosen = clause_choice(ctx, schema, clauses)?;
    let mut rows = Vec::new();
    for (index, clause) in clauses.iter().enumerate() {
        if let Action::Insert(insert) = &clause.action {
            let values = insert_values(ctx, schema, table, source_alias, source_columns, insert)?;
            rows.push((index, values));
        }
    }
    let inserting = rows
        .iter()
        .map(|(index, _)| lit(*index as i64))
        .collect::<Vec<_>>();
    let mut projection = Vec::with_capacity(target_schema.fields().len() + 2);
    for (column, field) in target_schema.fields().iter().enumerate() {
        let mut value: Option<CaseBuilder> = None;
        for (index, values) in &rows {
            let branch = cast_to(values[column].clone(), field.data_type());
            let condition = chosen.clone().eq(lit(*index as i64));
            value = Some(match value {
                None => when(condition, branch),
                Some(mut builder) => {
                    builder.when(condition, branch);
                    builder
                },
            });
        }
        let null = lit(ScalarValue::try_from(field.data_type())?);
        let value = match value {
            Some(mut builder) => builder.otherwise(null)?,
            None => null,
        };
        projection.push(value.alias(field.name()));
    }
    projection.push(
        when(chosen.clone().in_list(inserting, false), chosen)
            .otherwise(lit(ScalarValue::Int64(None)))?
            .alias(CLAUSE_COLUMN),
    );
    projection.push(qualified_col(source_alias, SOURCE_ROW_COLUMN).alias(SOURCE_ROW_COLUMN));
    Ok(projection)
}

/// The value of each target column an `INSERT` action writes: the source's
/// columns by position for a bare `INSERT`, else its `VALUES` row, a column
/// it omits or gives `DEFAULT` taking the column's default.
fn insert_values(
    ctx: &SessionContext,
    schema: &DFSchema,
    table: &DuckLakeTable,
    source_alias: &str,
    source_columns: &[String],
    insert: &MergeInsertExpr,
) -> DataFusionResult<Vec<DFExpr>> {
    let target_schema = table.physical_schema();
    let width = target_schema.fields().len();
    let default_of = |name: &str| {
        table
            .get_column_default(name)
            .cloned()
            .unwrap_or_else(|| lit(ScalarValue::Null))
    };
    let values = match &insert.kind {
        MergeInsertKind::Row => {
            if !insert.columns.is_empty() {
                return Err(DataFusionError::NotImplemented(
                    "MERGE INSERT with a column list needs VALUES".to_string(),
                ));
            }
            if source_columns.len() != width {
                return Err(DataFusionError::Plan(format!(
                    "table {} has {width} columns but {} values were supplied",
                    table.table_name(),
                    source_columns.len()
                )));
            }
            return Ok(source_columns
                .iter()
                .map(|name| qualified_col(source_alias, name))
                .collect());
        },
        MergeInsertKind::Values(values) => values,
    };
    let [row] = values.rows.as_slice() else {
        return Err(DataFusionError::NotImplemented(
            "MERGE INSERT supports exactly one VALUES row".to_string(),
        ));
    };
    let expressions = &row.content;
    let columns: Vec<&str> = if insert.columns.is_empty() {
        target_schema
            .fields()
            .iter()
            .map(|field| field.name().as_str())
            .collect()
    } else {
        insert
            .columns
            .iter()
            .map(|name| {
                let column = object_last_name(name).ok_or_else(|| {
                    DataFusionError::Plan("MERGE INSERT column has no identifier".to_string())
                })?;
                resolve_column(&target_schema, &column).ok_or_else(|| {
                    DataFusionError::Plan(format!("MERGE INSERT targets unknown column '{column}'"))
                })
            })
            .collect::<DataFusionResult<_>>()?
    };
    if columns.len() != expressions.len() {
        return Err(DataFusionError::Plan(format!(
            "MERGE INSERT names {} columns but supplies {} values",
            columns.len(),
            expressions.len()
        )));
    }
    let mut values: Vec<Option<DFExpr>> = vec![None; width];
    for (column, expression) in columns.into_iter().zip(expressions) {
        let index = target_schema.index_of(column)?;
        if values[index].is_some() {
            return Err(DataFusionError::Plan(format!(
                "MERGE INSERT names column '{column}' more than once"
            )));
        }
        values[index] = Some(if is_default_keyword(expression) {
            default_of(column)
        } else {
            logical_expr(ctx, expression, schema)?
        });
    }
    Ok(target_schema
        .fields()
        .iter()
        .zip(values)
        .map(|(field, value)| value.unwrap_or_else(|| default_of(field.name())))
        .collect())
}

fn is_default_keyword(expr: &Expr) -> bool {
    matches!(expr, Expr::Identifier(ident)
        if ident.quote_style.is_none() && ident.value.eq_ignore_ascii_case("DEFAULT"))
}

/// The matched rows a `DELETE` removes: each target row once, every matched
/// pair counted.
fn deleted_rows(batches: &[RecordBatch]) -> DataFusionResult<Superseded> {
    let mut rowids = batch_rowids(batches)?;
    let counted = rowids.len() as u64;
    rowids.sort_unstable();
    rowids.dedup();
    Ok(Superseded {
        rowids,
        versions: Vec::new(),
        counted,
    })
}

/// The new version of each matched target row from `batches`, `[physical
/// columns..., rowid, source row]`: the one of the first source row matching
/// it, as official keeps the first version reaching its update
/// (`ducklake_update.cpp`).
fn first_versions(batches: Vec<RecordBatch>, schema: &SchemaRef) -> DataFusionResult<Superseded> {
    let Some(first) = batches.first() else {
        return Ok(Superseded::default());
    };
    let combined = concat_batches(&first.schema(), &batches)?;
    let width = schema.fields().len();
    let rowids = rowid_array(combined.column(width))?;
    let source_rows = combined
        .column(width + 1)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .ok_or_else(|| DataFusionError::Internal("MERGE source row is not UInt64".to_string()))?;
    let mut first: HashMap<i64, usize> = HashMap::new();
    for row in 0..combined.num_rows() {
        let rowid = rowids.value(row);
        first
            .entry(rowid)
            .and_modify(|kept| {
                if source_rows.value(row) < source_rows.value(*kept) {
                    *kept = row;
                }
            })
            .or_insert(row);
    }
    let mut kept: Vec<u64> = first.values().map(|row| *row as u64).collect();
    kept.sort_unstable();
    let versions = take_record_batch(&combined, &UInt64Array::from(kept))?
        .project(&(0..=width).collect::<Vec<_>>())?;
    let rowids = batch_rowids(std::slice::from_ref(&versions))?;
    Ok(Superseded {
        rowids,
        versions: coerce_output_batches(&[versions], schema, true)?,
        counted: combined.num_rows() as u64,
    })
}

/// `batches`, inserted rows sorted by the clause they take, split into one
/// group per `INSERT` action with the table's physical fields.
fn rows_by_clause(
    batches: &[RecordBatch],
    schema: &SchemaRef,
) -> DataFusionResult<Vec<Vec<RecordBatch>>> {
    let Some(first) = batches.first() else {
        return Ok(Vec::new());
    };
    let combined = concat_batches(&first.schema(), batches)?;
    let clauses = combined
        .column(schema.fields().len())
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| DataFusionError::Internal("MERGE clause is not Int64".to_string()))?;
    let mut groups = Vec::new();
    let mut start = 0;
    for row in 1..=combined.num_rows() {
        if row == combined.num_rows() || clauses.value(row) != clauses.value(start) {
            let group = combined.slice(start, row - start);
            groups.push(coerce_output_batches(&[group], schema, false)?);
            start = row;
        }
    }
    Ok(groups)
}

fn rowid_array(column: &ArrayRef) -> DataFusionResult<&Int64Array> {
    let array = column
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| DataFusionError::Internal("MERGE rowid is not Int64".to_string()))?;
    if array.null_count() > 0 {
        return Err(DataFusionError::Execution(
            "MERGE matched a row without DuckLake row lineage".to_string(),
        ));
    }
    Ok(array)
}

/// The `rowid` column of `batches`, in order.
fn batch_rowids(batches: &[RecordBatch]) -> DataFusionResult<Vec<i64>> {
    let mut rowids = Vec::new();
    for batch in batches {
        let index = batch.schema().index_of(ROWID_COLUMN_NAME)?;
        rowids.extend(rowid_array(batch.column(index))?.values().iter().copied());
    }
    Ok(rowids)
}

/// `batches` with the table's physical fields, their values already of the
/// fields' types, and a trailing `rowid` when `has_rowid`.
fn coerce_output_batches(
    batches: &[RecordBatch],
    schema: &Schema,
    has_rowid: bool,
) -> DataFusionResult<Vec<RecordBatch>> {
    let mut fields: Vec<Arc<Field>> = schema.fields().iter().cloned().collect();
    if has_rowid {
        fields.push(Arc::new(rowid_field()));
    }
    let output_schema = Arc::new(Schema::new(fields));
    batches
        .iter()
        .filter(|batch| batch.num_rows() > 0)
        .map(|batch| {
            let mut columns = Vec::with_capacity(output_schema.fields().len());
            for (index, field) in schema.fields().iter().enumerate() {
                columns.push(crate::column_rename::coerce_column(
                    batch.column(index),
                    field.data_type(),
                )?);
            }
            if has_rowid {
                columns.push(Arc::clone(batch.column(schema.fields().len())));
            }
            Ok(RecordBatch::try_new(Arc::clone(&output_schema), columns)?)
        })
        .collect()
}

fn count_frame(ctx: &SessionContext, count: u64) -> DataFusionResult<DataFrame> {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "count",
        DataType::UInt64,
        false,
    )]));
    let values: ArrayRef = Arc::new(UInt64Array::from(vec![count]));
    ctx.read_batch(RecordBatch::try_new(schema, vec![values])?)
}
