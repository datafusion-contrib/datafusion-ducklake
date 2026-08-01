//! SQL entry point for DuckLake metadata and data-layout DDL.
//!
//! DataFusion's SQL parser (sqlparser) does not accept `ALTER TABLE … SET
//! PARTITIONED BY (…)` / `… SET SORTED BY (…)` — it errors at parse time, before
//! any `LogicalPlan` exists, so a custom `QueryPlanner`/analyzer can never
//! intercept it. Instead, [`execute_ducklake_sql`] is a transparent wrapper the
//! caller uses in place of [`SessionContext::sql`]: it recognizes these DDL forms
//! and `COMMENT ON`, using DataFusion's bundled sqlparser with no new dependency,
//! then dispatches them to the programmatic
//! [`MetadataWriter`](crate::metadata_writer::MetadataWriter) API on the given
//! [`DuckLakeCatalog`], and delegates everything else to `ctx.sql(sql)` unchanged.
//!
//! ```no_run
//! # async fn run(ctx: &datafusion::prelude::SessionContext,
//! #             catalog: &datafusion_ducklake::DuckLakeCatalog) -> datafusion::error::Result<()> {
//! use datafusion_ducklake::execute_ducklake_sql;
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
//! each with optional `ASC`/`DESC` and `NULLS FIRST`/`NULLS LAST`. Anything else
//! fails closed.

use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::logical_expr::LogicalPlanBuilder;
use datafusion::prelude::{DataFrame, SessionContext};
use datafusion::sql::sqlparser::ast::{
    CommentObject, Expr, Function, FunctionArg, FunctionArgExpr, FunctionArguments, Ident,
    ObjectName, Statement,
};
use datafusion::sql::sqlparser::dialect::GenericDialect;
use datafusion::sql::sqlparser::keywords::Keyword;
use datafusion::sql::sqlparser::parser::{Parser, ParserError};
use datafusion::sql::sqlparser::tokenizer::Token;

use crate::catalog::DuckLakeCatalog;
use crate::metadata_provider::{TagObjectType, TagTarget};
use crate::partition::PartitionTransform;
use crate::sort::{NullOrder, SortDirection, SortField};

/// Execute a SQL statement against `ctx`, handling DuckLake comments and layout
/// DDL directly on `catalog` and delegating everything else to
/// [`SessionContext::sql`].
///
/// For a recognized statement this resolves the target through `catalog`'s
/// provider/writer, applies the change, and returns an empty result set matching
/// DataFusion's own DDL. Callers can route all their SQL through it.
///
/// The DDL targets `catalog` (the 1–3 part table name's catalog segment, if any,
/// is not cross-checked); a read-only catalog yields a clear error.
pub async fn execute_ducklake_sql(
    ctx: &SessionContext,
    catalog: &DuckLakeCatalog,
    sql: &str,
) -> DataFusionResult<DataFrame> {
    match parse_ducklake_ddl(sql)? {
        Some(ddl) => apply_ducklake_ddl(ctx, catalog, ddl).await,
        None => ctx.sql(sql).await,
    }
}

/// The DuckLake metadata and data-layout DDL statements this module recognizes.
/// Names hold raw `(value, is_quoted)` parts for identifier normalization.
enum DuckLakeDdl {
    Comment {
        object_type: CommentObject,
        object_name: Vec<(String, bool)>,
        comment: Option<String>,
        if_exists: bool,
    },
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
    DataFusionError::Plan(format!("DuckLake DDL parse error: {error}"))
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

/// Recognize `COMMENT ON` and DuckLake partition/sort `ALTER TABLE` statements.
/// Returns `Ok(None)` for unrelated SQL, `Ok(Some(_))` for recognized DDL, and
/// `Err` for malformed recognized DDL.
fn parse_ducklake_ddl(sql: &str) -> DataFusionResult<Option<DuckLakeDdl>> {
    let dialect = GenericDialect {};
    if sql
        .split_whitespace()
        .next()
        .is_some_and(|word| word.eq_ignore_ascii_case("comment"))
    {
        let mut statements = Parser::parse_sql(&dialect, sql).map_err(parse_err)?;
        if statements.len() != 1 {
            return Err(DataFusionError::Plan(
                "COMMENT ON accepts exactly one statement".to_string(),
            ));
        }
        return match statements.pop().expect("one statement checked above") {
            Statement::Comment {
                object_type,
                object_name,
                comment,
                if_exists,
            } => Ok(Some(DuckLakeDdl::Comment {
                object_type,
                object_name: object_name_parts(&object_name),
                comment,
                if_exists,
            })),
            _ => Ok(None),
        };
    }
    let mut parser = match Parser::new(&dialect).try_with_sql(sql) {
        Ok(parser) => parser,
        // Let ctx.sql surface the tokenizer error for consistency.
        Err(_) => return Ok(None),
    };

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
            Ok(Some(DuckLakeDdl::SetPartition {
                table,
                transforms,
            }))
        } else if parser.parse_keyword(Keyword::SORTED) {
            expect_by(&mut parser, "SET SORTED")?;
            parser.expect_token(&Token::LParen).map_err(parse_err)?;
            let fields = parse_sort_keys(&mut parser)?;
            parser.expect_token(&Token::RParen).map_err(parse_err)?;
            expect_statement_end(&mut parser)?;
            Ok(Some(DuckLakeDdl::SetSort {
                table,
                fields,
            }))
        } else {
            Ok(None)
        }
    } else if parser.parse_keyword(Keyword::RESET) {
        if parser.parse_keyword(Keyword::PARTITIONED) {
            expect_by(&mut parser, "RESET PARTITIONED")?;
            expect_statement_end(&mut parser)?;
            Ok(Some(DuckLakeDdl::ResetPartition {
                table,
            }))
        } else if parser.parse_keyword(Keyword::SORTED) {
            expect_by(&mut parser, "RESET SORTED")?;
            expect_statement_end(&mut parser)?;
            Ok(Some(DuckLakeDdl::ResetSort {
                table,
            }))
        } else {
            Ok(None)
        }
    } else {
        Ok(None)
    }
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
            "partition DDL target must be a table name of 1–3 parts".to_string(),
        )),
    }
}

fn resolve_schema(parts: &[(String, bool)]) -> DataFusionResult<String> {
    let norm = |(value, quoted): &(String, bool)| {
        if *quoted {
            value.clone()
        } else {
            value.to_ascii_lowercase()
        }
    };
    match parts {
        [schema] => Ok(norm(schema)),
        [_catalog, schema] => Ok(norm(schema)),
        _ => Err(DataFusionError::Plan(
            "COMMENT ON SCHEMA target must have 1 or 2 parts".to_string(),
        )),
    }
}

fn resolve_schema_table_column(
    parts: &[(String, bool)],
) -> DataFusionResult<(String, String, String)> {
    let norm = |(value, quoted): &(String, bool)| {
        if *quoted {
            value.clone()
        } else {
            value.to_ascii_lowercase()
        }
    };
    match parts {
        [table, column] => Ok(("main".to_string(), norm(table), norm(column))),
        [schema, table, column] => Ok((norm(schema), norm(table), norm(column))),
        [_catalog, schema, table, column] => Ok((norm(schema), norm(table), norm(column))),
        _ => Err(DataFusionError::Plan(
            "COMMENT ON COLUMN target must have 2 to 4 parts".to_string(),
        )),
    }
}

fn empty_dataframe(ctx: &SessionContext) -> DataFusionResult<DataFrame> {
    let plan = LogicalPlanBuilder::empty(false).build()?;
    Ok(DataFrame::new(ctx.state(), plan))
}

async fn apply_ducklake_ddl(
    ctx: &SessionContext,
    catalog: &DuckLakeCatalog,
    ddl: DuckLakeDdl,
) -> DataFusionResult<DataFrame> {
    let writer = catalog.writer().ok_or_else(|| {
        DataFusionError::Plan(
            "catalog is read-only; open it with DuckLakeCatalog::with_writer to run \
             DuckLake metadata DDL"
                .to_string(),
        )
    })?;
    let provider = catalog.provider();
    // The DDL targets the catalog head (writes commit on top of it), so resolve the
    // table at the current snapshot rather than any pinned one.
    let snapshot = provider
        .get_current_snapshot()
        .map_err(DataFusionError::from)?;

    if let DuckLakeDdl::Comment {
        object_type,
        object_name,
        comment,
        if_exists,
    } = ddl
    {
        let missing = |kind: &str, name: &str| {
            if if_exists {
                Ok(None)
            } else {
                Err(DataFusionError::Plan(format!("{kind} '{name}' not found")))
            }
        };
        let target = match object_type {
            CommentObject::Schema => {
                let schema_name = resolve_schema(&object_name)?;
                match provider
                    .get_schema_by_name(&schema_name, snapshot)
                    .map_err(DataFusionError::from)?
                {
                    Some(schema) => Some(TagTarget::Object {
                        object_type: TagObjectType::Schema,
                        object_id: schema.schema_id,
                    }),
                    None => missing("schema", &schema_name)?,
                }
            },
            CommentObject::Table | CommentObject::View => {
                let (schema_name, object_name) = resolve_schema_table(&object_name)?;
                let Some(schema) = provider
                    .get_schema_by_name(&schema_name, snapshot)
                    .map_err(DataFusionError::from)?
                else {
                    let _ = missing("schema", &schema_name)?;
                    return empty_dataframe(ctx);
                };
                if object_type == CommentObject::Table {
                    match provider
                        .get_table_by_name(schema.schema_id, &object_name, snapshot)
                        .map_err(DataFusionError::from)?
                    {
                        Some(table) => Some(TagTarget::Object {
                            object_type: TagObjectType::Table,
                            object_id: table.table_id,
                        }),
                        None => missing("table", &object_name)?,
                    }
                } else {
                    match provider
                        .get_view_id_by_name(schema.schema_id, &object_name, snapshot)
                        .map_err(DataFusionError::from)?
                    {
                        Some(view_id) => Some(TagTarget::Object {
                            object_type: TagObjectType::View,
                            object_id: view_id,
                        }),
                        None => missing("view", &object_name)?,
                    }
                }
            },
            CommentObject::Column => {
                let (schema_name, table_name, column_name) =
                    resolve_schema_table_column(&object_name)?;
                let Some(schema) = provider
                    .get_schema_by_name(&schema_name, snapshot)
                    .map_err(DataFusionError::from)?
                else {
                    let _ = missing("schema", &schema_name)?;
                    return empty_dataframe(ctx);
                };
                let Some(table) = provider
                    .get_table_by_name(schema.schema_id, &table_name, snapshot)
                    .map_err(DataFusionError::from)?
                else {
                    let _ = missing("table", &table_name)?;
                    return empty_dataframe(ctx);
                };
                match provider
                    .get_table_structure(table.table_id, snapshot)
                    .map_err(DataFusionError::from)?
                    .into_iter()
                    .find(|column| column.column_name == column_name)
                {
                    Some(column) => Some(TagTarget::Column {
                        table_id: table.table_id,
                        column_id: column.column_id,
                    }),
                    None => missing("column", &column_name)?,
                }
            },
            other => {
                return Err(DataFusionError::Plan(format!(
                    "COMMENT ON {other} is not supported for DuckLake catalogs"
                )));
            },
        };
        if let Some(target) = target {
            writer
                .set_tag(target, "comment", comment.as_deref())
                .map_err(DataFusionError::from)?;
        }
        return empty_dataframe(ctx);
    }

    let parts = match &ddl {
        DuckLakeDdl::Comment {
            ..
        } => unreachable!("comments return above"),
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
    let (schema_name, table_name) = resolve_schema_table(parts)?;

    let schema = provider
        .get_schema_by_name(&schema_name, snapshot)
        .map_err(DataFusionError::from)?
        .ok_or_else(|| DataFusionError::Plan(format!("schema '{schema_name}' not found")))?;
    let table = provider
        .get_table_by_name(schema.schema_id, &table_name, snapshot)
        .map_err(DataFusionError::from)?
        .ok_or_else(|| DataFusionError::Plan(format!("table '{table_name}' not found")))?;

    match ddl {
        DuckLakeDdl::Comment {
            ..
        } => unreachable!("comments return above"),
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

    // DDL returns an empty (0-row) result, matching DataFusion's own DDL.
    empty_dataframe(ctx)
}
