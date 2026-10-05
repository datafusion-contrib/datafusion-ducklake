//! DuckLake `MERGE INTO` planning and atomic execution.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Int64Array, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::common::{Column, DFSchema, TableReference};
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::logical_expr::{Expr as DFExpr, JoinType};
use datafusion::prelude::{DataFrame, SessionContext};
use datafusion::sql::sqlparser::ast::{
    Assignment, AssignmentTarget, Expr, Ident, Merge, MergeAction, MergeClauseKind,
    MergeInsertExpr, MergeInsertKind, ObjectName, TableFactor,
};
use futures::TryStreamExt;

use crate::catalog::DuckLakeCatalog;
use crate::compaction::{compaction_ordering, sorted_rewrite_batches};
use crate::metadata_writer::{DeleteFileEntry, WriteMode};
use crate::row_id::{ROW_POS_COLUMN_NAME, ROWID_COLUMN_NAME, rowid_field};
use crate::table::{DuckLakeTable, UpdateSourceScan};
use crate::table_writer::DuckLakeTableWriter;

enum MatchedAction {
    Update {
        assignments: Vec<Assignment>,
        predicate: Option<Expr>,
    },
    Delete {
        predicate: Option<Expr>,
    },
}

struct MergeActions {
    matched: Option<MatchedAction>,
    insert: Option<(MergeInsertExpr, Option<Expr>)>,
}

/// Execute one parsed DuckLake MERGE and return its affected-row count.
pub(crate) async fn execute_merge(
    ctx: &SessionContext,
    catalog: &DuckLakeCatalog,
    merge: Merge,
    update_all: bool,
) -> DataFusionResult<DataFrame> {
    let (target_name, target_alias) = target_factor(&merge.table)?;
    let (schema_name, table_name) = resolve_schema_table(&target_name)?;
    let source_alias = factor_alias(&merge.source).ok_or_else(|| {
        DataFusionError::Plan("MERGE source must have a table name or alias".to_string())
    })?;
    let actions = classify_actions(&merge, update_all)?;

    let writer = catalog.writer().ok_or_else(|| {
        DataFusionError::Plan(
            "catalog is read-only; open it with DuckLakeCatalog::with_writer to run MERGE"
                .to_string(),
        )
    })?;
    if !writer.supports_update() {
        return Err(DataFusionError::NotImplemented(
            "MERGE INTO is not supported on this metadata backend".to_string(),
        ));
    }

    let table = catalog
        .table_for_write(&schema_name, &table_name)
        .map_err(DataFusionError::from)?;
    if table
        .live_partition_spec()
        .map_err(DataFusionError::from)?
        .is_some()
    {
        return Err(DataFusionError::NotImplemented(
            "MERGE INTO does not support partitioned target tables currently".to_string(),
        ));
    }

    let target = ctx
        .read_table(Arc::new(table.clone().with_row_lineage(true)))?
        .alias(&target_alias)?;
    let source = ctx
        .sql(&format!("SELECT * FROM {}", merge.source))
        .await?
        .alias(&source_alias)?;
    let join_schema = target.schema().join(source.schema())?;
    let on = logical_expr(ctx, merge.on.as_ref(), &join_schema)?;

    let (updated_batches, matched_rowids) = match actions.matched {
        Some(MatchedAction::Update {
            assignments,
            predicate,
        }) => {
            let joined = target
                .clone()
                .join_on(source.clone(), JoinType::Inner, [on.clone()])?;
            let joined = apply_predicate(ctx, joined, predicate.as_ref())?;
            let projection = update_projection(
                ctx,
                joined.schema(),
                table.physical_schema().as_ref(),
                &target_alias,
                &source_alias,
                &assignments,
            )?;
            let batches = joined.select(projection)?.collect().await?;
            let rowids = unique_rowids(&batches)?;
            let batches = coerce_output_batches(&batches, table.physical_schema().as_ref(), true)?;
            (batches, rowids)
        },
        Some(MatchedAction::Delete {
            predicate,
        }) => {
            let joined = target
                .clone()
                .join_on(source.clone(), JoinType::Inner, [on.clone()])?;
            let joined = apply_predicate(ctx, joined, predicate.as_ref())?;
            let batches = joined
                .select([qualified_col(&target_alias, ROWID_COLUMN_NAME)])?
                .collect()
                .await?;
            (Vec::new(), unique_rowids(&batches)?)
        },
        None => (Vec::new(), HashSet::new()),
    };

    let inserted = match actions.insert {
        Some((insert, predicate)) => {
            let unmatched = source
                .clone()
                .join_on(target.clone(), JoinType::LeftAnti, [on])?;
            let unmatched = apply_predicate(ctx, unmatched, predicate.as_ref())?;
            let projection = insert_projection(
                ctx,
                unmatched.schema(),
                table.physical_schema().as_ref(),
                &source_alias,
                &insert,
            )?;
            unmatched.select(projection)?.collect().await?
        },
        None => Vec::new(),
    };
    let inserted_count = inserted.iter().map(RecordBatch::num_rows).sum::<usize>();
    let affected = matched_rowids.len() + inserted_count;
    if affected == 0 {
        return count_frame(ctx, 0);
    }

    let object_store = ctx
        .runtime_env()
        .object_store(table.object_store_url().as_ref())?;
    let table_writer = DuckLakeTableWriter::new(Arc::clone(&writer), object_store)
        .map_err(DataFusionError::from)?;
    let delete_entries = delete_entries(
        ctx,
        &table,
        &table_writer,
        &schema_name,
        &table_name,
        &matched_rowids,
    )
    .await?;

    let mut output = updated_batches;
    if inserted_count > 0 {
        let count = i64::try_from(inserted_count).map_err(|_| {
            DataFusionError::Execution("MERGE insert row count exceeds i64".to_string())
        })?;
        let start = writer
            .reserve_row_ids(table.table_id(), count)
            .map_err(DataFusionError::from)?;
        output.extend(add_reserved_rowids(
            &inserted,
            table.physical_schema().as_ref(),
            start,
        )?);
    }

    let mut session = table_writer
        .begin_write_with_embedded_rowid(
            &schema_name,
            &table_name,
            table.physical_schema().as_ref(),
            WriteMode::Append,
        )
        .map_err(DataFusionError::from)?;
    if !output.is_empty() {
        let sort_spec = table.live_sort_spec().map_err(DataFusionError::from)?;
        let ordering = compaction_ordering(table.physical_schema().as_ref(), sort_spec.as_ref())
            .map_err(DataFusionError::from)?;
        let mut stream = sorted_rewrite_batches(ctx.task_ctx(), output, ordering.as_ref())
            .map_err(DataFusionError::from)?;
        while let Some(batch) = stream.try_next().await? {
            session.write_batch(&batch).map_err(DataFusionError::from)?;
        }
    }
    session
        .finish_with_deletes(&delete_entries)
        .await
        .map_err(DataFusionError::from)?;
    count_frame(ctx, affected as u64)
}

fn classify_actions(merge: &Merge, update_all: bool) -> DataFusionResult<MergeActions> {
    if merge.output.is_some() {
        return Err(DataFusionError::NotImplemented(
            "MERGE OUTPUT/RETURNING is not supported".to_string(),
        ));
    }
    let mut matched = None;
    let mut insert = None;
    for clause in &merge.clauses {
        match (&clause.clause_kind, &clause.action) {
            (MergeClauseKind::Matched, MergeAction::Update(update)) => {
                if matched.is_some() {
                    return Err(single_matched_action_error());
                }
                if update.update_predicate.is_some() || update.delete_predicate.is_some() {
                    return Err(DataFusionError::NotImplemented(
                        "MERGE UPDATE WHERE/DELETE WHERE is not supported; use WHEN MATCHED AND"
                            .to_string(),
                    ));
                }
                matched = Some(MatchedAction::Update {
                    assignments: if update_all {
                        Vec::new()
                    } else {
                        update.assignments.clone()
                    },
                    predicate: clause.predicate.clone(),
                });
            },
            (
                MergeClauseKind::Matched,
                MergeAction::Delete {
                    ..
                },
            ) => {
                if matched.is_some() {
                    return Err(single_matched_action_error());
                }
                matched = Some(MatchedAction::Delete {
                    predicate: clause.predicate.clone(),
                });
            },
            (
                MergeClauseKind::NotMatched | MergeClauseKind::NotMatchedByTarget,
                MergeAction::Insert(action),
            ) => {
                if insert.is_some() {
                    return Err(DataFusionError::NotImplemented(
                        "MERGE INTO supports one INSERT action".to_string(),
                    ));
                }
                if action.insert_predicate.is_some() {
                    return Err(DataFusionError::NotImplemented(
                        "MERGE INSERT WHERE is not supported; use WHEN NOT MATCHED AND".to_string(),
                    ));
                }
                insert = Some((action.clone(), clause.predicate.clone()));
            },
            _ => {
                return Err(DataFusionError::NotImplemented(format!(
                    "unsupported MERGE clause: {clause}"
                )));
            },
        }
    }
    if matched.is_none() && insert.is_none() {
        return Err(DataFusionError::Plan(
            "MERGE requires at least one supported action".to_string(),
        ));
    }
    Ok(MergeActions {
        matched,
        insert,
    })
}

fn single_matched_action_error() -> DataFusionError {
    DataFusionError::NotImplemented(
        "MERGE INTO with DuckLake only supports a single UPDATE/DELETE action currently; the DuckLake specification permits more"
            .to_string(),
    )
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

fn logical_expr(ctx: &SessionContext, expr: &Expr, schema: &DFSchema) -> DataFusionResult<DFExpr> {
    ctx.state().create_logical_expr(&expr.to_string(), schema)
}

fn apply_predicate(
    ctx: &SessionContext,
    frame: DataFrame,
    predicate: Option<&Expr>,
) -> DataFusionResult<DataFrame> {
    match predicate {
        Some(predicate) => {
            let expr = logical_expr(ctx, predicate, frame.schema())?;
            frame.filter(expr)
        },
        None => Ok(frame),
    }
}

fn qualified_col(alias: &str, name: &str) -> DFExpr {
    DFExpr::Column(Column::new(
        Some(TableReference::bare(alias.to_string())),
        name,
    ))
}

fn update_projection(
    ctx: &SessionContext,
    schema: &DFSchema,
    target_schema: &Schema,
    target_alias: &str,
    source_alias: &str,
    assignments: &[Assignment],
) -> DataFusionResult<Vec<DFExpr>> {
    let mut values = HashMap::new();
    for assignment in assignments {
        let AssignmentTarget::ColumnName(name) = &assignment.target else {
            return Err(DataFusionError::NotImplemented(
                "tuple assignments are not supported in MERGE UPDATE".to_string(),
            ));
        };
        let column = object_last_name(name).ok_or_else(|| {
            DataFusionError::Plan("MERGE UPDATE assignment has no column name".to_string())
        })?;
        if target_schema.field_with_name(&column).is_err() {
            return Err(DataFusionError::Plan(format!(
                "MERGE UPDATE targets unknown column '{column}'"
            )));
        }
        if values
            .insert(
                column.clone(),
                logical_expr(ctx, &assignment.value, schema)?,
            )
            .is_some()
        {
            return Err(DataFusionError::Plan(format!(
                "MERGE UPDATE assigns column '{column}' more than once"
            )));
        }
    }
    let mut projection = Vec::with_capacity(target_schema.fields().len() + 1);
    for field in target_schema.fields() {
        let expr = values.remove(field.name()).unwrap_or_else(|| {
            let source = qualified_col(source_alias, field.name());
            if assignments.is_empty() {
                source
            } else {
                qualified_col(target_alias, field.name())
            }
        });
        projection.push(expr.alias(field.name()));
    }
    projection.push(qualified_col(target_alias, ROWID_COLUMN_NAME).alias(ROWID_COLUMN_NAME));
    Ok(projection)
}

fn insert_projection(
    ctx: &SessionContext,
    schema: &DFSchema,
    target_schema: &Schema,
    source_alias: &str,
    insert: &MergeInsertExpr,
) -> DataFusionResult<Vec<DFExpr>> {
    match &insert.kind {
        MergeInsertKind::Row => {
            if schema.fields().len() != target_schema.fields().len() {
                return Err(DataFusionError::Plan(format!(
                    "MERGE INSERT ROW source has {} columns but target has {}",
                    schema.fields().len(),
                    target_schema.fields().len()
                )));
            }
            Ok(target_schema
                .fields()
                .iter()
                .zip(schema.fields())
                .map(|(target, source)| {
                    qualified_col(source_alias, source.name()).alias(target.name())
                })
                .collect())
        },
        MergeInsertKind::Values(values) => {
            if values.rows.len() != 1 {
                return Err(DataFusionError::NotImplemented(
                    "MERGE INSERT supports exactly one VALUES row".to_string(),
                ));
            }
            let expressions = &values.rows[0].content;
            let columns: Vec<String> = if insert.columns.is_empty() {
                target_schema
                    .fields()
                    .iter()
                    .map(|field| field.name().clone())
                    .collect()
            } else {
                insert
                    .columns
                    .iter()
                    .map(|name| {
                        object_last_name(name).ok_or_else(|| {
                            DataFusionError::Plan(
                                "MERGE INSERT column has no identifier".to_string(),
                            )
                        })
                    })
                    .collect::<DataFusionResult<_>>()?
            };
            if columns.len() != expressions.len() || columns.len() != target_schema.fields().len() {
                return Err(DataFusionError::NotImplemented(
                    "MERGE INSERT must provide every target column exactly once".to_string(),
                ));
            }
            let mut values_by_name = HashMap::new();
            for (column, value) in columns.into_iter().zip(expressions) {
                if target_schema.field_with_name(&column).is_err() {
                    return Err(DataFusionError::Plan(format!(
                        "MERGE INSERT targets unknown column '{column}'"
                    )));
                }
                if values_by_name
                    .insert(column.clone(), logical_expr(ctx, value, schema)?)
                    .is_some()
                {
                    return Err(DataFusionError::Plan(format!(
                        "MERGE INSERT names column '{column}' more than once"
                    )));
                }
            }
            target_schema
                .fields()
                .iter()
                .map(|field| {
                    values_by_name
                        .remove(field.name())
                        .map(|expr| expr.alias(field.name()))
                        .ok_or_else(|| {
                            DataFusionError::Plan(format!(
                                "MERGE INSERT omits target column '{}'",
                                field.name()
                            ))
                        })
                })
                .collect()
        },
    }
}

fn unique_rowids(batches: &[RecordBatch]) -> DataFusionResult<HashSet<i64>> {
    let mut rowids = HashSet::new();
    for batch in batches {
        let index = batch.schema().index_of(ROWID_COLUMN_NAME)?;
        let array = batch
            .column(index)
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| DataFusionError::Internal("MERGE rowid is not Int64".to_string()))?;
        for row in 0..array.len() {
            if array.is_null(row) {
                return Err(DataFusionError::Execution(
                    "MERGE matched a row without DuckLake row lineage".to_string(),
                ));
            }
            let rowid = array.value(row);
            if !rowids.insert(rowid) {
                return Err(DataFusionError::Execution(format!(
                    "MERGE target rowid {rowid} matched more than one source row"
                )));
            }
        }
    }
    Ok(rowids)
}

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
        .map(|batch| {
            let mut columns = Vec::with_capacity(output_schema.fields().len());
            for (index, field) in schema.fields().iter().enumerate() {
                columns.push(crate::column_rename::coerce_column(
                    batch.column(index),
                    field.data_type(),
                )?);
            }
            if has_rowid {
                columns.push(crate::column_rename::coerce_column(
                    batch.column(schema.fields().len()),
                    &DataType::Int64,
                )?);
            }
            Ok(RecordBatch::try_new(Arc::clone(&output_schema), columns)?)
        })
        .collect()
}

fn add_reserved_rowids(
    batches: &[RecordBatch],
    schema: &Schema,
    start: i64,
) -> DataFusionResult<Vec<RecordBatch>> {
    let mut next = start;
    let mut output = coerce_output_batches(batches, schema, false)?;
    let mut fields: Vec<Arc<Field>> = schema.fields().iter().cloned().collect();
    fields.push(Arc::new(rowid_field()));
    let output_schema = Arc::new(Schema::new(fields));
    for batch in &mut output {
        let end = next
            .checked_add(i64::try_from(batch.num_rows()).map_err(|_| {
                DataFusionError::Execution("MERGE batch row count exceeds i64".to_string())
            })?)
            .ok_or_else(|| DataFusionError::Execution("MERGE row ID overflow".to_string()))?;
        let rowids: ArrayRef = Arc::new(Int64Array::from_iter_values(next..end));
        let mut columns = batch.columns().to_vec();
        columns.push(rowids);
        *batch = RecordBatch::try_new(Arc::clone(&output_schema), columns)?;
        next = end;
    }
    Ok(output)
}

async fn delete_entries(
    ctx: &SessionContext,
    table: &DuckLakeTable,
    writer: &DuckLakeTableWriter,
    schema_name: &str,
    table_name: &str,
    rowids: &HashSet<i64>,
) -> DataFusionResult<Vec<DeleteFileEntry>> {
    if rowids.is_empty() {
        return Ok(Vec::new());
    }
    let state = ctx.state();
    let mut found = HashSet::new();
    let mut entries = Vec::new();
    let inlined_deletes = table.inlined_deletes_by_file()?;
    for file in table.files().map_err(DataFusionError::from)? {
        let scan = table
            .build_update_scan(&state, &file, inlined_deletes.get(&file.data_file_id))
            .await?;
        let batches =
            datafusion::physical_plan::collect(Arc::clone(&scan.scan), ctx.task_ctx()).await?;
        let positions = matching_positions(&scan, &batches, rowids, &mut found)?;
        if positions.is_empty() {
            continue;
        }
        let mut cumulative = scan.existing_parquet_deleted.clone();
        cumulative.extend(positions);
        let mut cumulative: Vec<i64> = cumulative.into_iter().collect();
        cumulative.sort_unstable();
        let delete = writer
            .write_delete_file(schema_name, table_name, &scan.source_path, &cumulative)
            .await
            .map_err(DataFusionError::from)?;
        entries.push(DeleteFileEntry {
            data_file_id: scan.data_file_id,
            expected_prev_delete_file: scan.delete_file_id,
            delete,
        });
    }
    if found != *rowids {
        let missing = rowids.difference(&found).copied().collect::<Vec<_>>();
        return Err(DataFusionError::Execution(format!(
            "MERGE could not map matched row IDs to live data files: {missing:?}"
        )));
    }
    Ok(entries)
}

fn matching_positions(
    scan: &UpdateSourceScan,
    batches: &[RecordBatch],
    selected: &HashSet<i64>,
    found: &mut HashSet<i64>,
) -> DataFusionResult<Vec<i64>> {
    let mut positions = Vec::new();
    for batch in batches {
        let row_positions = batch
            .column(scan.pos_index)
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| {
                DataFusionError::Internal(format!("{ROW_POS_COLUMN_NAME} is not Int64"))
            })?;
        let embedded = scan
            .embedded_batch_idx
            .map(|index| {
                batch
                    .column(index)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .ok_or_else(|| {
                        DataFusionError::Internal("embedded rowid is not Int64".to_string())
                    })
            })
            .transpose()?;
        for row in 0..batch.num_rows() {
            let position = row_positions.value(row);
            let rowid = match embedded {
                Some(rowids) => rowids.value(row),
                None => {
                    scan.row_id_start
                        .expect("row_id_start validated by build_update_scan")
                        + position
                },
            };
            if selected.contains(&rowid) {
                found.insert(rowid);
                positions.push(position);
            }
        }
    }
    Ok(positions)
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
