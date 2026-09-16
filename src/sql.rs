//! SQL entry point for DuckLake data-layout DDL (partition + sort order) and
//! `CREATE TABLE … AS SELECT`.
//!
//! DataFusion's SQL parser (sqlparser) does not accept `ALTER TABLE … SET
//! PARTITIONED BY (…)` / `… SET SORTED BY (…)` — it errors at parse time, before
//! any `LogicalPlan` exists, so a custom `QueryPlanner`/analyzer can never
//! intercept it. Instead, [`execute_ducklake_sql`] is a transparent wrapper the
//! caller uses in place of [`SessionContext::sql`]: it recognizes these DDL forms
//! with a tiny hand-rolled parser (reusing DataFusion's bundled sqlparser — no new
//! dependency), dispatches them to the programmatic
//! [`MetadataWriter`] API on the given
//! [`DuckLakeCatalog`], and delegates everything else to `ctx.sql(sql)` unchanged.
//!
//! `CREATE TABLE … AS SELECT` (CTAS) also runs here. DataFusion executes a CTAS
//! by materializing the query into a `MemTable` and handing it to the schema's
//! synchronous `register_table`, which has neither the session's object stores
//! nor a runtime to write data files. The wrapper instead plans the query on
//! `ctx`, derives the table columns from its schema, and writes the rows and the
//! table metadata in one DuckLake snapshot through the session's object store.
//!
//! ```no_run
//! # async fn run(ctx: &datafusion::prelude::SessionContext,
//! #             catalog: &datafusion_ducklake::DuckLakeCatalog) -> datafusion::error::Result<()> {
//! use datafusion_ducklake::execute_ducklake_sql;
//! execute_ducklake_sql(ctx, catalog,
//!     "CREATE TABLE lake.main.recent AS SELECT * FROM lake.main.events WHERE ts > '2026-01-01'").await?;
//! execute_ducklake_sql(ctx, catalog,
//!     "ALTER TABLE lake.main.events SET PARTITIONED BY (region, year(ts))").await?;
//! execute_ducklake_sql(ctx, catalog,
//!     "ALTER TABLE lake.main.events RESET PARTITIONED BY").await?;
//! execute_ducklake_sql(ctx, catalog,
//!     "ALTER TABLE lake.main.events SET SORTED BY (device_id, ts DESC NULLS LAST)").await?;
//! execute_ducklake_sql(ctx, catalog,
//!     "ALTER TABLE lake.main.events RESET SORTED BY").await?;
//! # Ok(()) }
//! ```
//!
//! Supported partition transforms: `identity` (a bare column), `year`, `month`,
//! `day`, `hour`. Supported sort keys: bare columns only (`SORTED BY (a, b DESC)`),
//! each with optional `ASC`/`DESC` and `NULLS FIRST`/`NULLS LAST`. CTAS accepts
//! `IF NOT EXISTS`; `OR REPLACE` and an explicit column list fail closed, while
//! `TEMPORARY` and `EXTERNAL` tables go to DataFusion. Anything else fails closed.

use std::collections::HashSet;
use std::sync::Arc;

use arrow::datatypes::{Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::logical_expr::LogicalPlanBuilder;
use datafusion::prelude::{DataFrame, SessionContext};
use datafusion::sql::parser::Statement;
use datafusion::sql::sqlparser::ast::{
    Expr, Function, FunctionArg, FunctionArgExpr, FunctionArguments, Ident, ObjectName, Query,
    Statement as SqlStatement,
};
use datafusion::sql::sqlparser::dialect::GenericDialect;
use datafusion::sql::sqlparser::keywords::Keyword;
use datafusion::sql::sqlparser::parser::{Parser, ParserError};
use datafusion::sql::sqlparser::tokenizer::Token;

use crate::catalog::DuckLakeCatalog;
use crate::metadata_provider::MetadataProvider;
use crate::metadata_writer::{ColumnDef, MetadataWriter, WriteMode};
use crate::partition::PartitionTransform;
use crate::schema::{publish_empty_table, validate_table_name};
use crate::sort::{NullOrder, SortDirection, SortField};
use crate::table_writer::{DuckLakeTableWriter, DuckLakeWriteOptions, TableWriteOptions};

/// Execute a SQL statement against `ctx`, handling DuckLake partition DDL
/// (`ALTER TABLE … SET/RESET PARTITIONED BY`) and `CREATE TABLE … AS SELECT`
/// directly on `catalog` and delegating everything else to
/// [`SessionContext::sql`].
///
/// For a partition DDL statement this resolves the target table through
/// `catalog`'s provider/writer, applies the change, and returns an empty result
/// set (matching DataFusion's own DDL). For a CTAS it plans the query on `ctx`,
/// derives nullable columns from the query schema (as DuckDB does), and commits
/// the table and its rows in one snapshot through the object store registered on
/// `ctx` for the catalog's data path; the catalog's write options and the
/// schema-scoped catalog settings apply. An empty result publishes the table
/// with no data file. It is fully transparent for every other statement, so
/// callers can route all their SQL through it.
///
/// The DDL targets `catalog` (the 1–3 part table name's catalog segment, if any,
/// is not cross-checked); a read-only catalog yields a clear error.
pub async fn execute_ducklake_sql(
    ctx: &SessionContext,
    catalog: &DuckLakeCatalog,
    sql: &str,
) -> DataFusionResult<DataFrame> {
    match parse_ducklake_statement(sql)? {
        Some(DuckLakeStatement::Ddl(ddl)) => apply_ducklake_ddl(ctx, catalog, ddl).await,
        Some(DuckLakeStatement::CreateTableAs {
            table,
            query,
            if_not_exists,
        }) => {
            let target = resolve_writable_target(catalog, &table)?;
            create_table_as(ctx, catalog, &target, *query, if_not_exists).await?;
            empty_result(ctx)
        },
        None => ctx.sql(sql).await,
    }
}

/// The statements this module handles itself instead of delegating to DataFusion.
enum DuckLakeStatement {
    Ddl(DuckLakeDdl),
    CreateTableAs {
        table: Vec<(String, bool)>,
        query: Box<Query>,
        if_not_exists: bool,
    },
}

/// The DuckLake data-layout DDL statements this module recognizes. `table` holds
/// the raw name parts `(value, is_quoted)` for later identifier normalization.
enum DuckLakeDdl {
    SetPartition {
        table: Vec<(String, bool)>,
        transforms: Vec<(String, PartitionTransform)>,
    },
    ResetPartition {
        table: Vec<(String, bool)>,
    },
    SetSort {
        table: Vec<(String, bool)>,
        fields: Vec<SortField>,
    },
    ResetSort {
        table: Vec<(String, bool)>,
    },
}

fn parse_err(error: ParserError) -> DataFusionError {
    DataFusionError::Plan(format!("partition DDL parse error: {error}"))
}

/// After a recognized partition-DDL statement, reject any trailing input (a lone
/// terminating `;` is allowed) so e.g. `… SET PARTITIONED BY (a) DROP TABLE x` is
/// an error rather than a silently-applied `SET (a)`.
fn expect_statement_end(parser: &mut Parser) -> DataFusionResult<()> {
    let _ = parser.consume_token(&Token::SemiColon);
    let next = parser.peek_token().token;
    if next != Token::EOF {
        return Err(DataFusionError::Plan(format!(
            "unexpected trailing input after DuckLake DDL near '{next}'"
        )));
    }
    Ok(())
}

/// Recognize `ALTER TABLE <name> {SET|RESET} {PARTITIONED|SORTED} BY [...]`.
/// Returns `Ok(None)` when the statement is not DuckLake data-layout DDL (so the
/// caller delegates to `ctx.sql`), `Ok(Some(_))` when it is well-formed, and `Err`
/// when it is clearly our DDL but malformed (so the caller gets a precise error
/// rather than a confusing tokenizer error).
fn parse_ducklake_statement(sql: &str) -> DataFusionResult<Option<DuckLakeStatement>> {
    let dialect = GenericDialect {};
    let mut parser = match Parser::new(&dialect).try_with_sql(sql) {
        Ok(parser) => parser,
        // Let ctx.sql surface the tokenizer error for consistency.
        Err(_) => return Ok(None),
    };

    if parser.peek_keyword(Keyword::CREATE) {
        return parse_create_table_as(&mut parser);
    }
    if !parser.parse_keyword(Keyword::ALTER) || !parser.parse_keyword(Keyword::TABLE) {
        return Ok(None);
    }
    let name = match parser.parse_object_name(false) {
        Ok(name) => name,
        Err(_) => return Ok(None),
    };
    let table = object_name_parts(&name);

    if parser.parse_keyword(Keyword::SET) {
        // Only `SET PARTITIONED BY` / `SET SORTED BY` are ours; any other `SET …`
        // goes to DataFusion.
        if parser.parse_keyword(Keyword::PARTITIONED) {
            expect_by(&mut parser, "SET PARTITIONED")?;
            parser.expect_token(&Token::LParen).map_err(parse_err)?;
            let exprs = parser
                .parse_comma_separated(Parser::parse_expr)
                .map_err(parse_err)?;
            parser.expect_token(&Token::RParen).map_err(parse_err)?;
            let transforms = parse_transforms(exprs)?;
            expect_statement_end(&mut parser)?;
            Ok(Some(DuckLakeStatement::Ddl(DuckLakeDdl::SetPartition {
                table,
                transforms,
            })))
        } else if parser.parse_keyword(Keyword::SORTED) {
            expect_by(&mut parser, "SET SORTED")?;
            parser.expect_token(&Token::LParen).map_err(parse_err)?;
            let fields = parse_sort_keys(&mut parser)?;
            parser.expect_token(&Token::RParen).map_err(parse_err)?;
            expect_statement_end(&mut parser)?;
            Ok(Some(DuckLakeStatement::Ddl(DuckLakeDdl::SetSort {
                table,
                fields,
            })))
        } else {
            Ok(None)
        }
    } else if parser.parse_keyword(Keyword::RESET) {
        if parser.parse_keyword(Keyword::PARTITIONED) {
            expect_by(&mut parser, "RESET PARTITIONED")?;
            expect_statement_end(&mut parser)?;
            Ok(Some(DuckLakeStatement::Ddl(DuckLakeDdl::ResetPartition {
                table,
            })))
        } else if parser.parse_keyword(Keyword::SORTED) {
            expect_by(&mut parser, "RESET SORTED")?;
            expect_statement_end(&mut parser)?;
            Ok(Some(DuckLakeStatement::Ddl(DuckLakeDdl::ResetSort {
                table,
            })))
        } else {
            Ok(None)
        }
    } else {
        Ok(None)
    }
}

/// Recognize `CREATE TABLE [IF NOT EXISTS] name AS <query>`. Any other `CREATE`
/// statement, including one sqlparser cannot parse, goes to DataFusion.
fn parse_create_table_as(parser: &mut Parser) -> DataFusionResult<Option<DuckLakeStatement>> {
    let Ok(SqlStatement::CreateTable(create)) = parser.parse_statement() else {
        return Ok(None);
    };
    let Some(query) = create.query else {
        return Ok(None);
    };
    if create.temporary || create.external {
        return Ok(None);
    }
    if create.or_replace {
        return Err(DataFusionError::NotImplemented(
            "CREATE OR REPLACE TABLE AS SELECT is not supported; DROP TABLE first".to_string(),
        ));
    }
    if !create.columns.is_empty() {
        return Err(DataFusionError::NotImplemented(
            "CREATE TABLE AS SELECT does not accept a column list; alias the query columns instead"
                .to_string(),
        ));
    }
    expect_statement_end(parser)?;
    Ok(Some(DuckLakeStatement::CreateTableAs {
        table: object_name_parts(&create.name),
        query,
        if_not_exists: create.if_not_exists,
    }))
}

/// Consume the mandatory `BY` keyword after a `{SET|RESET} {PARTITIONED|SORTED}`
/// prefix, or return a precise error naming `context`.
fn expect_by(parser: &mut Parser, context: &str) -> DataFusionResult<()> {
    if parser.parse_keyword(Keyword::BY) {
        Ok(())
    } else {
        Err(DataFusionError::Plan(format!(
            "expected BY after {context}"
        )))
    }
}

/// Parse the comma-separated sort-key list of `SET SORTED BY (…)`. Each key is a
/// bare column (v1 scope) with optional `ASC`/`DESC` (default `ASC`) and `NULLS
/// FIRST`/`NULLS LAST` (default `NULLS LAST`, matching DuckDB). Non-column
/// expressions fail closed.
fn parse_sort_keys(parser: &mut Parser) -> DataFusionResult<Vec<SortField>> {
    let mut fields = Vec::new();
    loop {
        let expr = parser.parse_expr().map_err(parse_err)?;
        let Expr::Identifier(ident) = expr else {
            return Err(DataFusionError::Plan(format!(
                "unsupported sort key '{expr}'; only bare column names are supported"
            )));
        };
        let column = normalize_ident(&ident);

        let direction = if parser.parse_keyword(Keyword::ASC) {
            SortDirection::Asc
        } else if parser.parse_keyword(Keyword::DESC) {
            SortDirection::Desc
        } else {
            SortDirection::Asc
        };

        let null_order = if parser.parse_keyword(Keyword::NULLS) {
            if parser.parse_keyword(Keyword::FIRST) {
                NullOrder::NullsFirst
            } else if parser.parse_keyword(Keyword::LAST) {
                NullOrder::NullsLast
            } else {
                return Err(DataFusionError::Plan(
                    "expected FIRST or LAST after NULLS".to_string(),
                ));
            }
        } else {
            NullOrder::NullsLast
        };

        fields.push(SortField::column(
            fields.len() as i32,
            column,
            direction,
            null_order,
        ));

        if !parser.consume_token(&Token::Comma) {
            break;
        }
    }
    if fields.is_empty() {
        return Err(DataFusionError::Plan(
            "SET SORTED BY requires at least one column".to_string(),
        ));
    }
    Ok(fields)
}

/// Turn the parsed partition-key expressions into `(column_name, transform)`
/// pairs. Accepts a bare column (identity) or `year|month|day|hour(col)`.
fn parse_transforms(exprs: Vec<Expr>) -> DataFusionResult<Vec<(String, PartitionTransform)>> {
    if exprs.is_empty() {
        return Err(DataFusionError::Plan(
            "SET PARTITIONED BY requires at least one column".to_string(),
        ));
    }
    let mut out = Vec::with_capacity(exprs.len());
    for expr in exprs {
        let pair = match expr {
            Expr::Identifier(ident) => (normalize_ident(&ident), PartitionTransform::Identity),
            Expr::Function(func) => {
                let fname = func
                    .name
                    .0
                    .last()
                    .and_then(|part| part.as_ident())
                    .map(|ident| ident.value.to_ascii_lowercase())
                    .unwrap_or_default();
                let transform = match fname.as_str() {
                    "year" => PartitionTransform::Year,
                    "month" => PartitionTransform::Month,
                    "day" => PartitionTransform::Day,
                    "hour" => PartitionTransform::Hour,
                    other => {
                        return Err(DataFusionError::Plan(format!(
                            "unsupported partition transform '{other}' \
                             (supported: identity, year, month, day, hour)"
                        )));
                    },
                };
                let column = single_ident_arg(&func).ok_or_else(|| {
                    DataFusionError::Plan(format!(
                        "partition transform '{fname}' expects exactly one column argument"
                    ))
                })?;
                (column, transform)
            },
            other => {
                return Err(DataFusionError::Plan(format!(
                    "unsupported partition key expression '{other}'; \
                     use a column or year()/month()/day()/hour()"
                )));
            },
        };
        out.push(pair);
    }
    Ok(out)
}

/// Extract the single unnamed identifier argument of a transform function, if it
/// has exactly one such argument.
fn single_ident_arg(func: &Function) -> Option<String> {
    match &func.args {
        FunctionArguments::List(list) if list.args.len() == 1 => match &list.args[0] {
            FunctionArg::Unnamed(FunctionArgExpr::Expr(Expr::Identifier(ident))) => {
                Some(normalize_ident(ident))
            },
            _ => None,
        },
        _ => None,
    }
}

/// Normalize an identifier the way DataFusion does by default: an unquoted
/// identifier folds to lowercase; a quoted identifier is kept verbatim.
fn normalize_ident(ident: &Ident) -> String {
    if ident.quote_style.is_some() {
        ident.value.clone()
    } else {
        ident.value.to_ascii_lowercase()
    }
}

/// The name parts of an `ObjectName` as `(value, is_quoted)` (function-valued
/// parts, which cannot appear in a table name here, are skipped).
fn object_name_parts(name: &ObjectName) -> Vec<(String, bool)> {
    name.0
        .iter()
        .filter_map(|part| part.as_ident())
        .map(|ident| (ident.value.clone(), ident.quote_style.is_some()))
        .collect()
}

/// Resolve the raw name parts into `(schema, table)`, applying DataFusion's
/// identifier normalization. A 1-part name defaults to the DuckLake `main`
/// schema; a 3-part name's catalog segment is ignored (the DDL targets the given
/// catalog).
fn resolve_schema_table(parts: &[(String, bool)]) -> DataFusionResult<(String, String)> {
    let norm = |(value, quoted): &(String, bool)| {
        if *quoted {
            value.clone()
        } else {
            value.to_ascii_lowercase()
        }
    };
    match parts {
        [table] => Ok(("main".to_string(), norm(table))),
        [schema, table] => Ok((norm(schema), norm(table))),
        [_catalog, schema, table] => Ok((norm(schema), norm(table))),
        _ => Err(DataFusionError::Plan(
            "DuckLake DDL target must be a table name of 1–3 parts".to_string(),
        )),
    }
}

async fn apply_ducklake_ddl(
    ctx: &SessionContext,
    catalog: &DuckLakeCatalog,
    ddl: DuckLakeDdl,
) -> DataFusionResult<DataFrame> {
    let parts = match &ddl {
        DuckLakeDdl::SetPartition {
            table,
            ..
        }
        | DuckLakeDdl::ResetPartition {
            table,
        }
        | DuckLakeDdl::SetSort {
            table,
            ..
        }
        | DuckLakeDdl::ResetSort {
            table,
        } => table,
    };
    let WritableTarget {
        writer,
        provider,
        snapshot,
        schema_name,
        table_name,
    } = resolve_writable_target(catalog, parts)?;

    let schema = provider
        .get_schema_by_name(&schema_name, snapshot)
        .map_err(DataFusionError::from)?
        .ok_or_else(|| DataFusionError::Plan(format!("schema '{schema_name}' not found")))?;
    let table = provider
        .get_table_by_name(schema.schema_id, &table_name, snapshot)
        .map_err(DataFusionError::from)?
        .ok_or_else(|| DataFusionError::Plan(format!("table '{table_name}' not found")))?;

    match ddl {
        DuckLakeDdl::SetPartition {
            transforms,
            ..
        } => {
            writer
                .set_partition_spec(table.table_id, &transforms)
                .map_err(DataFusionError::from)?;
        },
        DuckLakeDdl::ResetPartition {
            ..
        } => {
            writer
                .reset_partition_spec(table.table_id)
                .map_err(DataFusionError::from)?;
        },
        DuckLakeDdl::SetSort {
            fields,
            ..
        } => {
            writer
                .set_sort_spec(table.table_id, &fields)
                .map_err(DataFusionError::from)?;
        },
        DuckLakeDdl::ResetSort {
            ..
        } => {
            writer
                .reset_sort_spec(table.table_id)
                .map_err(DataFusionError::from)?;
        },
    }

    empty_result(ctx)
}

/// A statement's target resolved on a writable catalog at its current head.
struct WritableTarget {
    writer: Arc<dyn MetadataWriter>,
    provider: Arc<dyn MetadataProvider>,
    snapshot: i64,
    schema_name: String,
    table_name: String,
}

// Statements commit on top of the catalog head, so the target resolves at the
// current snapshot rather than any pinned one.
fn resolve_writable_target(
    catalog: &DuckLakeCatalog,
    parts: &[(String, bool)],
) -> DataFusionResult<WritableTarget> {
    let writer = catalog.writer().ok_or_else(|| {
        DataFusionError::Plan(
            "catalog is read-only; open it with DuckLakeCatalog::with_writer to run \
             DuckLake DDL"
                .to_string(),
        )
    })?;
    let provider = catalog.provider();
    let snapshot = provider
        .get_current_snapshot()
        .map_err(DataFusionError::from)?;
    let (schema_name, table_name) = resolve_schema_table(parts)?;
    Ok(WritableTarget {
        writer,
        provider,
        snapshot,
        schema_name,
        table_name,
    })
}

/// Run `CREATE TABLE [IF NOT EXISTS] schema.table AS query` on the catalog head.
///
/// Planning and the existence check happen before the query executes, so a
/// name clash fails without running it. The rows and the table metadata commit
/// in one snapshot with `head` as the expected base, so a table created
/// concurrently under the same name makes the commit conflict instead of
/// replacing it.
async fn create_table_as(
    ctx: &SessionContext,
    catalog: &DuckLakeCatalog,
    target: &WritableTarget,
    query: Query,
    if_not_exists: bool,
) -> DataFusionResult<()> {
    let WritableTarget {
        writer,
        provider,
        snapshot: head,
        schema_name,
        table_name,
    } = target;
    let head = *head;
    validate_table_name(table_name)?;
    let schema = provider
        .get_schema_by_name(schema_name, head)
        .map_err(DataFusionError::from)?
        .ok_or_else(|| DataFusionError::Plan(format!("schema '{schema_name}' not found")))?;
    if provider
        .get_table_by_name(schema.schema_id, table_name, head)
        .map_err(DataFusionError::from)?
        .is_some()
    {
        if if_not_exists {
            return Ok(());
        }
        return Err(DataFusionError::Plan(format!(
            "Cannot create table '{table_name}': a table with that name already exists"
        )));
    }
    if provider
        .get_view_by_name(schema.schema_id, table_name, head)
        .map_err(DataFusionError::from)?
        .is_some()
    {
        return Err(DataFusionError::Plan(format!(
            "Cannot create table '{table_name}': a view with that name already exists"
        )));
    }

    let state = ctx.state();
    let plan = state
        .statement_to_plan(Statement::Statement(Box::new(SqlStatement::Query(
            Box::new(query),
        ))))
        .await?;
    let arrow_schema = ctas_table_schema(plan.schema().as_arrow())?;
    let columns = arrow_schema
        .fields()
        .iter()
        .map(|field| {
            ColumnDef::from_arrow(field.name(), field.data_type(), field.is_nullable())
                .map_err(DataFusionError::from)
        })
        .collect::<DataFusionResult<Vec<_>>>()?;

    let batches = DataFrame::new(state, plan).collect().await?;
    let batches = batches
        .into_iter()
        .filter(|batch| batch.num_rows() > 0)
        .map(|batch| RecordBatch::try_new(Arc::clone(&arrow_schema), batch.columns().to_vec()))
        .collect::<Result<Vec<_>, _>>()?;
    if batches.is_empty() {
        publish_empty_table(writer.as_ref(), schema_name, table_name, &columns)?;
        return Ok(());
    }

    let settings = provider
        .get_metadata_settings(Some(schema.schema_id), None)
        .map_err(DataFusionError::from)?;
    let write_options = DuckLakeWriteOptions::from_metadata_settings_deferred(&settings)
        .with_overrides(&catalog.write_options());
    write_options.validate().map_err(DataFusionError::from)?;
    let object_store = ctx
        .runtime_env()
        .object_store(catalog.object_store_url().as_ref())?;
    let table_writer = DuckLakeTableWriter::new(Arc::clone(writer), object_store)
        .map_err(DataFusionError::from)?
        .with_options(&write_options);
    let mut transaction = table_writer
        .transaction()
        .with_options(&TableWriteOptions::new().with_expected_base_snapshot_id(head));
    transaction
        .stage_write(
            schema_name,
            table_name,
            &arrow_schema,
            WriteMode::Replace,
            &batches,
        )
        .await
        .map_err(DataFusionError::from)?;
    transaction.commit().await.map_err(DataFusionError::from)?;
    Ok(())
}

/// The schema a CTAS table gets from its query: DuckDB creates every CTAS
/// column nullable regardless of the query's nullability, and field metadata
/// does not belong in the catalog. Duplicate names are rejected here because a
/// join or aliased projection can legally produce them in a query result.
fn ctas_table_schema(query_schema: &Schema) -> DataFusionResult<Arc<Schema>> {
    let mut seen = HashSet::new();
    for field in query_schema.fields() {
        if !seen.insert(field.name().as_str()) {
            return Err(DataFusionError::Plan(format!(
                "CREATE TABLE AS SELECT produced duplicate column name '{}'; alias the columns",
                field.name()
            )));
        }
    }
    let fields = query_schema
        .fields()
        .iter()
        .map(|field| Field::new(field.name(), field.data_type().clone(), true))
        .collect::<Vec<_>>();
    Ok(Arc::new(Schema::new(fields)))
}

/// DDL returns an empty (0-row) result, matching DataFusion's own DDL.
fn empty_result(ctx: &SessionContext) -> DataFusionResult<DataFrame> {
    let plan = LogicalPlanBuilder::empty(false).build()?;
    Ok(DataFrame::new(ctx.state(), plan))
}
