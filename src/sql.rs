//! SQL entry point for DuckLake partition DDL.
//!
//! DataFusion's SQL parser (sqlparser) does not accept `ALTER TABLE … SET
//! PARTITIONED BY (…)` — it errors at parse time, before any `LogicalPlan`
//! exists, so a custom `QueryPlanner`/analyzer can never intercept it. Instead,
//! [`execute_ducklake_sql`] is a transparent wrapper the caller uses in place of
//! [`SessionContext::sql`]: it recognizes the two partition-DDL forms with a tiny
//! hand-rolled parser (reusing DataFusion's bundled sqlparser — no new
//! dependency), dispatches them to the programmatic
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
//! # Ok(()) }
//! ```
//!
//! Supported transforms: `identity` (a bare column), `year`, `month`, `day`,
//! `hour`. Anything else fails closed.

use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::logical_expr::LogicalPlanBuilder;
use datafusion::prelude::{DataFrame, SessionContext};
use datafusion::sql::sqlparser::ast::{
    Expr, Function, FunctionArg, FunctionArgExpr, FunctionArguments, Ident, ObjectName,
};
use datafusion::sql::sqlparser::dialect::GenericDialect;
use datafusion::sql::sqlparser::keywords::Keyword;
use datafusion::sql::sqlparser::parser::{Parser, ParserError};
use datafusion::sql::sqlparser::tokenizer::Token;

use crate::catalog::DuckLakeCatalog;
use crate::partition::PartitionTransform;

/// Execute a SQL statement against `ctx`, handling DuckLake partition DDL
/// (`ALTER TABLE … SET/RESET PARTITIONED BY`) directly on `catalog` and delegating
/// everything else to [`SessionContext::sql`].
///
/// For a partition DDL statement this resolves the target table through
/// `catalog`'s provider/writer, applies the change, and returns an empty result
/// set (matching DataFusion's own DDL). It is fully transparent for every other
/// statement, so callers can route all their SQL through it.
///
/// The DDL targets `catalog` (the 1–3 part table name's catalog segment, if any,
/// is not cross-checked); a read-only catalog yields a clear error.
pub async fn execute_ducklake_sql(
    ctx: &SessionContext,
    catalog: &DuckLakeCatalog,
    sql: &str,
) -> DataFusionResult<DataFrame> {
    match parse_partition_ddl(sql)? {
        Some(ddl) => apply_partition_ddl(ctx, catalog, ddl).await,
        None => ctx.sql(sql).await,
    }
}

/// The two partition-DDL statements this module recognizes. `table` holds the raw
/// name parts `(value, is_quoted)` for later identifier normalization.
enum PartitionDdl {
    Set {
        table: Vec<(String, bool)>,
        transforms: Vec<(String, PartitionTransform)>,
    },
    Reset {
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
            "unexpected trailing input after partition DDL near '{next}'"
        )));
    }
    Ok(())
}

/// Recognize `ALTER TABLE <name> {SET|RESET} PARTITIONED BY [...]`. Returns
/// `Ok(None)` when the statement is not partition DDL (so the caller delegates to
/// `ctx.sql`), `Ok(Some(_))` when it is well-formed, and `Err` when it is clearly
/// partition DDL but malformed (so the caller gets a precise error rather than a
/// confusing tokenizer error).
fn parse_partition_ddl(sql: &str) -> DataFusionResult<Option<PartitionDdl>> {
    let dialect = GenericDialect {};
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
        // Only `SET PARTITIONED BY` is ours; any other `SET …` goes to DataFusion.
        if !parser.parse_keyword(Keyword::PARTITIONED) {
            return Ok(None);
        }
        if !parser.parse_keyword(Keyword::BY) {
            return Err(DataFusionError::Plan(
                "expected BY after SET PARTITIONED".to_string(),
            ));
        }
        parser.expect_token(&Token::LParen).map_err(parse_err)?;
        let exprs = parser
            .parse_comma_separated(Parser::parse_expr)
            .map_err(parse_err)?;
        parser.expect_token(&Token::RParen).map_err(parse_err)?;
        let transforms = parse_transforms(exprs)?;
        expect_statement_end(&mut parser)?;
        Ok(Some(PartitionDdl::Set {
            table,
            transforms,
        }))
    } else if parser.parse_keyword(Keyword::RESET) {
        if !parser.parse_keyword(Keyword::PARTITIONED) {
            return Ok(None);
        }
        if !parser.parse_keyword(Keyword::BY) {
            return Err(DataFusionError::Plan(
                "expected BY after RESET PARTITIONED".to_string(),
            ));
        }
        expect_statement_end(&mut parser)?;
        Ok(Some(PartitionDdl::Reset {
            table,
        }))
    } else {
        Ok(None)
    }
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

async fn apply_partition_ddl(
    ctx: &SessionContext,
    catalog: &DuckLakeCatalog,
    ddl: PartitionDdl,
) -> DataFusionResult<DataFrame> {
    let writer = catalog.writer().ok_or_else(|| {
        DataFusionError::Plan(
            "catalog is read-only; open it with DuckLakeCatalog::with_writer to run \
             partition DDL"
                .to_string(),
        )
    })?;
    let provider = catalog.provider();
    // Partition DDL targets the catalog head (writes commit on top of it), so
    // resolve the table at the current snapshot rather than any pinned one.
    let snapshot = provider
        .get_current_snapshot()
        .map_err(DataFusionError::from)?;

    let (parts, transforms) = match ddl {
        PartitionDdl::Set {
            table,
            transforms,
        } => (table, Some(transforms)),
        PartitionDdl::Reset {
            table,
        } => (table, None),
    };
    let (schema_name, table_name) = resolve_schema_table(&parts)?;

    let schema = provider
        .get_schema_by_name(&schema_name, snapshot)
        .map_err(DataFusionError::from)?
        .ok_or_else(|| DataFusionError::Plan(format!("schema '{schema_name}' not found")))?;
    let table = provider
        .get_table_by_name(schema.schema_id, &table_name, snapshot)
        .map_err(DataFusionError::from)?
        .ok_or_else(|| DataFusionError::Plan(format!("table '{table_name}' not found")))?;

    match transforms {
        Some(transforms) => {
            writer
                .set_partition_spec(table.table_id, &transforms)
                .map_err(DataFusionError::from)?;
        },
        None => {
            writer
                .reset_partition_spec(table.table_id)
                .map_err(DataFusionError::from)?;
        },
    }

    // DDL returns an empty (0-row) result, matching DataFusion's own DDL.
    let plan = LogicalPlanBuilder::empty(false).build()?;
    Ok(DataFrame::new(ctx.state(), plan))
}
