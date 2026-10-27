//! SQL entry point for DuckLake metadata and data-layout DDL and
//! `CREATE TABLE … AS SELECT`.
//!
//! DataFusion's SQL parser (sqlparser) does not accept `ALTER TABLE … SET
//! PARTITIONED BY (…)` / `… SET SORTED BY (…)` — it errors at parse time, before
//! any `LogicalPlan` exists, so a custom `QueryPlanner`/analyzer can never
//! intercept it. Instead, [`execute_ducklake_sql`] is a transparent wrapper the
//! caller uses in place of [`SessionContext::sql`]: it recognizes these DDL forms
//! and `COMMENT ON`, using DataFusion's bundled sqlparser with no new dependency,
//! then dispatches them to the programmatic
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
//! It also runs official DuckLake's inlined-data flush,
//! `CALL ducklake_flush_inlined_data('lake' [, schema_name => 's'] [, table_name => 't'])`,
//! on `catalog` when the statement is executed, and returns the flush's result
//! rows (`schema_name`, `table_name`, `rows_flushed`); see
//! [`DuckLakeCatalog::flush_inlined_data`]. The catalog name the call passes,
//! matched case-insensitively as DuckDB matches it, must name `catalog` as
//! registered in the session; the function name may be qualified only as
//! DuckDB resolves it (`main.`, `system.` or `system.main.`).
//!
//! Supported partition transforms: `identity` (a bare column), `year`, `month`,
//! `day`, `hour`. Supported sort keys: bare columns only (`SORTED BY (a, b DESC)`),
//! each with optional `ASC`/`DESC` and `NULLS FIRST`/`NULLS LAST`. CTAS accepts
//! `IF NOT EXISTS` and a column list of names; `OR REPLACE` and a column list
//! with types fail closed, while `TEMPORARY` and `EXTERNAL` tables go to
//! DataFusion. Anything else fails closed.

use std::collections::HashSet;
use std::sync::Arc;

use arrow::datatypes::{Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::catalog::CatalogProvider;
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::logical_expr::LogicalPlanBuilder;
use datafusion::prelude::{DataFrame, SessionContext};
use datafusion::sql::parser::Statement;
use datafusion::sql::sqlparser::ast::{
    CommentObject, Expr, Function, FunctionArg, FunctionArgExpr, FunctionArguments, Ident,
    ObjectName, Query, Statement as SqlStatement, Value,
};
use datafusion::sql::sqlparser::dialect::{DuckDbDialect, GenericDialect};
use datafusion::sql::sqlparser::keywords::Keyword;
use datafusion::sql::sqlparser::parser::{Parser, ParserError};
use datafusion::sql::sqlparser::tokenizer::Token;

use crate::catalog::DuckLakeCatalog;
use crate::metadata_provider::{MetadataProvider, TagObjectType, TagTarget};
use crate::metadata_writer::{
    CatalogNode, ColumnDef, MetadataWriter, WriteMode, catalog_column_defs,
};
use crate::partition::PartitionTransform;
use crate::schema::{
    ExistingTable, publish_empty_table, scoped_write_options, validate_table_name,
};
use crate::sort::{NullOrder, SortDirection, SortField};
use crate::table_writer::{DuckLakeTableWriter, TableWriteOptions};

/// Execute a SQL statement against `ctx`, handling DuckLake comments and layout
/// DDL and `CREATE TABLE … AS SELECT` directly on `catalog` and delegating
/// everything else to [`SessionContext::sql`].
///
/// For a recognized DDL statement this resolves the target through `catalog`'s
/// provider/writer, applies the change, and returns an empty result set matching
/// DataFusion's own DDL; `CALL ducklake_flush_inlined_data(...)` flushes before
/// returning and returns the flush's result rows. For a CTAS it plans the query
/// on `ctx`, derives nullable columns from the query schema (as DuckDB does), and
/// commits the table and its rows in one snapshot through the object store
/// registered on `ctx` for the catalog's data path, with the write options an
/// `INSERT` uses. An empty result publishes the table with no data file. Callers
/// can route all their SQL through it.
///
/// The DDL targets `catalog` (the 1–3 part table name's catalog segment, if any,
/// is not cross-checked); a read-only catalog yields a clear error. A CTAS is
/// handled only when its name, resolved with the session's default catalog and
/// schema, names `catalog`; any other goes to DataFusion.
pub async fn execute_ducklake_sql(
    ctx: &SessionContext,
    catalog: &DuckLakeCatalog,
    sql: &str,
) -> DataFusionResult<DataFrame> {
    if let Some(call) = parse_flush_inlined_data_call(sql)? {
        return flush_inlined_data_call(ctx, catalog, call).await;
    }
    match parse_ducklake_statement(sql)? {
        Some(DuckLakeStatement::Ddl(ddl)) => apply_ducklake_ddl(ctx, catalog, ddl).await,
        Some(DuckLakeStatement::CreateTableAs(create)) => {
            let Some((schema_name, table_name)) = ctas_target(ctx, catalog, &create.table) else {
                return ctx.sql(sql).await;
            };
            if let Some(reason) = create.unsupported {
                return Err(DataFusionError::NotImplemented(reason.to_string()));
            }
            let target = writable_target(catalog, schema_name, table_name)?;
            create_table_as(ctx, catalog, &target, create).await?;
            empty_dataframe(ctx)
        },
        None => ctx.sql(sql).await,
    }
}

/// A `CALL ducklake_flush_inlined_data(...)` statement.
#[derive(Debug, PartialEq, Eq)]
struct FlushInlinedDataCall {
    catalog: String,
    schema_name: Option<String>,
    table_name: Option<String>,
}

/// Recognize `CALL ducklake_flush_inlined_data('catalog' [, schema_name => 's']
/// [, table_name => 't'])`, official DuckLake's signature (`:=` is accepted for
/// `=>`). `Ok(None)` for any other statement.
fn parse_flush_inlined_data_call(sql: &str) -> DataFusionResult<Option<FlushInlinedDataCall>> {
    // The parser skips leading whitespace and comments, as the other hooks do.
    let is_call = Parser::new(&DuckDbDialect {})
        .try_with_sql(sql)
        .is_ok_and(|mut parser| parser.parse_keyword(Keyword::CALL));
    if !is_call {
        return Ok(None);
    }
    let Ok(mut statements) = Parser::parse_sql(&DuckDbDialect {}, sql) else {
        return Ok(None);
    };
    if statements.len() != 1 {
        return Ok(None);
    }
    let SqlStatement::Call(function) = statements.pop().expect("one statement checked above")
    else {
        return Ok(None);
    };
    let parts: Vec<&str> = function
        .name
        .0
        .iter()
        .filter_map(|part| part.as_ident())
        .map(|ident| ident.value.as_str())
        .collect();
    let Some((name, qualifier)) = parts.split_last() else {
        return Ok(None);
    };
    if !name.eq_ignore_ascii_case("ducklake_flush_inlined_data") {
        return Ok(None);
    }
    // DuckDB finds the function in the `system` catalog's `main` schema.
    let qualifier: Vec<String> = qualifier.iter().map(|q| q.to_ascii_lowercase()).collect();
    let resolvable = matches!(
        qualifier
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>()
            .as_slice(),
        [] | ["main"] | ["system"] | ["system", "main"]
    );
    if !resolvable || parts.len() != function.name.0.len() {
        return Err(DataFusionError::Plan(
            "Catalog Error: Table Function with name \"ducklake_flush_inlined_data\" is not in \
             the catalog, but it exists in the ducklake extension."
                .to_string(),
        ));
    }
    let string_arg = |arg: &FunctionArgExpr, what: &str| match arg {
        FunctionArgExpr::Expr(Expr::Value(value)) => match &value.value {
            Value::SingleQuotedString(text) => Ok(text.clone()),
            _ => Err(DataFusionError::Plan(format!(
                "ducklake_flush_inlined_data: {what} must be a string literal"
            ))),
        },
        _ => Err(DataFusionError::Plan(format!(
            "ducklake_flush_inlined_data: {what} must be a string literal"
        ))),
    };
    let args = match &function.args {
        FunctionArguments::List(list) => list.args.as_slice(),
        _ => &[],
    };
    let mut catalog = None;
    let mut schema_name = None;
    let mut table_name = None;
    for arg in args {
        match arg {
            FunctionArg::Unnamed(value) => {
                if catalog.is_some() {
                    return Err(DataFusionError::Plan(
                        "ducklake_flush_inlined_data takes one positional argument, the \
                         catalog name"
                            .to_string(),
                    ));
                }
                catalog = Some(string_arg(value, "the catalog name")?);
            },
            FunctionArg::Named {
                name,
                arg,
                ..
            } => match name.value.to_ascii_lowercase().as_str() {
                "schema_name" => schema_name = Some(string_arg(arg, "schema_name")?),
                "table_name" => table_name = Some(string_arg(arg, "table_name")?),
                other => {
                    return Err(DataFusionError::Plan(format!(
                        "ducklake_flush_inlined_data: unknown parameter '{other}'"
                    )));
                },
            },
            FunctionArg::ExprNamed {
                ..
            } => {
                return Err(DataFusionError::Plan(
                    "ducklake_flush_inlined_data: unsupported argument".to_string(),
                ));
            },
        }
    }
    let catalog = catalog.ok_or_else(|| {
        DataFusionError::Plan("ducklake_flush_inlined_data requires the catalog name".to_string())
    })?;
    Ok(Some(FlushInlinedDataCall {
        catalog,
        schema_name,
        table_name,
    }))
}

/// Run the flush on `catalog` now, once, and return its result rows. The named
/// catalog must be `catalog` as registered in `ctx`: official errors on a
/// catalog it does not know.
async fn flush_inlined_data_call(
    ctx: &SessionContext,
    catalog: &DuckLakeCatalog,
    call: FlushInlinedDataCall,
) -> DataFusionResult<DataFrame> {
    // DuckDB matches a database name case-insensitively.
    let name = ctx
        .catalog_names()
        .into_iter()
        .find(|name| name.eq_ignore_ascii_case(&call.catalog))
        .ok_or_else(|| {
            DataFusionError::Plan(format!(
                "Binder Error: Failed to find attached database \"{}\"",
                call.catalog
            ))
        })?;
    let registered = ctx.catalog(&name).ok_or_else(|| {
        DataFusionError::Plan(format!(
            "Binder Error: Failed to find attached database \"{}\"",
            call.catalog
        ))
    })?;
    if !is_same_catalog(&registered, catalog) {
        return Err(DataFusionError::Plan(format!(
            "ducklake_flush_inlined_data: catalog '{}' is not the DuckLake catalog this \
             statement runs against",
            call.catalog
        )));
    }
    let state = ctx.state();
    let flushed = catalog
        .flush_inlined_data(
            &state,
            call.schema_name.as_deref(),
            call.table_name.as_deref(),
        )
        .await
        .map_err(|error| match error {
            crate::DuckLakeError::SchemaNotFound(schema) => DataFusionError::Plan(format!(
                "Binder Error: Schema \"{schema}\" not found in DuckLakeCatalog \"{name}\""
            )),
            crate::DuckLakeError::TableNotFound(_) => DataFusionError::Plan(format!(
                "Catalog Error: Table with name {} does not exist!",
                call.table_name.as_deref().unwrap_or_default()
            )),
            error => error.into(),
        })?;
    ctx.read_batch(crate::inlined_flush::flushed_inlined_tables_batch(
        &flushed,
    )?)
}

/// The statements this module handles itself instead of delegating to DataFusion.
enum DuckLakeStatement {
    Ddl(DuckLakeDdl),
    CreateTableAs(CreateTableAs),
}

/// A `CREATE TABLE … AS <query>`. It is this module's to run only when its name
/// resolves to the DuckLake catalog; any other goes to DataFusion.
struct CreateTableAs {
    table: Vec<(String, bool)>,
    /// The names of `CREATE TABLE t (a, b) AS …`, normalized.
    columns: Vec<String>,
    query: Box<Query>,
    if_not_exists: bool,
    /// Why the statement is refused when it targets the DuckLake catalog.
    unsupported: Option<&'static str>,
}

const OR_REPLACE_UNSUPPORTED: &str =
    "CREATE OR REPLACE TABLE AS SELECT is not supported; DROP TABLE first";

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

/// Recognize `COMMENT ON`, DuckLake partition/sort `ALTER TABLE` statements, and
/// `CREATE TABLE … AS SELECT`. Returns `Ok(None)` for unrelated SQL, `Ok(Some(_))`
/// for a recognized statement, and `Err` for a malformed recognized one.
fn parse_ducklake_statement(sql: &str) -> DataFusionResult<Option<DuckLakeStatement>> {
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
            SqlStatement::Comment {
                object_type,
                object_name,
                comment,
                if_exists,
            } => Ok(Some(DuckLakeStatement::Ddl(DuckLakeDdl::Comment {
                object_type,
                object_name: object_name_parts(&object_name),
                comment,
                if_exists,
            }))),
            _ => Ok(None),
        };
    }
    let mut parser = match Parser::new(&dialect).try_with_sql(sql) {
        Ok(parser) => parser,
        // Let ctx.sql surface the tokenizer error for consistency.
        Err(_) => return Ok(None),
    };

    if parser.peek_keyword(Keyword::CREATE) {
        return Ok(parse_create_table_as(sql, &mut parser)?.map(DuckLakeStatement::CreateTableAs));
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

/// Recognize `CREATE [OR REPLACE] TABLE [IF NOT EXISTS] name [(column, …)] AS
/// <query>`. Any other `CREATE` statement, including one sqlparser cannot parse,
/// goes to DataFusion.
fn parse_create_table_as(
    sql: &str,
    parser: &mut Parser,
) -> DataFusionResult<Option<CreateTableAs>> {
    if let Some(create) = parse_create_table_as_with_column_names(sql) {
        return Ok(Some(create));
    }
    let Ok(SqlStatement::CreateTable(create)) = parser.parse_statement() else {
        return Ok(None);
    };
    let Some(query) = create.query else {
        return Ok(None);
    };
    if create.temporary || create.external {
        return Ok(None);
    }
    expect_statement_end(parser)?;
    let unsupported = if create.or_replace {
        Some(OR_REPLACE_UNSUPPORTED)
    } else if !create.columns.is_empty() {
        Some("CREATE TABLE AS SELECT does not accept column types; list only the column names")
    } else {
        None
    };
    Ok(Some(CreateTableAs {
        table: object_name_parts(&create.name),
        columns: Vec::new(),
        query,
        if_not_exists: create.if_not_exists,
        unsupported,
    }))
}

/// Recognize `CREATE [OR REPLACE] TABLE [IF NOT EXISTS] name (column, …) AS
/// <query>`, the untyped column list DuckDB accepts. sqlparser reads a column
/// list only with types, so this form is parsed here; `None` for anything else.
fn parse_create_table_as_with_column_names(sql: &str) -> Option<CreateTableAs> {
    let mut parser = Parser::new(&GenericDialect {}).try_with_sql(sql).ok()?;
    if !parser.parse_keyword(Keyword::CREATE) {
        return None;
    }
    let or_replace = parser.parse_keywords(&[Keyword::OR, Keyword::REPLACE]);
    if !parser.parse_keyword(Keyword::TABLE) {
        return None;
    }
    let if_not_exists = parser.parse_keywords(&[Keyword::IF, Keyword::NOT, Keyword::EXISTS]);
    let name = parser.parse_object_name(false).ok()?;
    parser.expect_token(&Token::LParen).ok()?;
    let columns = parser
        .parse_comma_separated(|parser| parser.parse_identifier())
        .ok()?;
    parser.expect_token(&Token::RParen).ok()?;
    if !parser.parse_keyword(Keyword::AS) {
        return None;
    }
    let query = parser.parse_query().ok()?;
    let _ = parser.consume_token(&Token::SemiColon);
    if parser.peek_token().token != Token::EOF {
        return None;
    }
    Some(CreateTableAs {
        table: object_name_parts(&name),
        columns: columns.iter().map(normalize_ident).collect(),
        query,
        if_not_exists,
        unsupported: or_replace.then_some(OR_REPLACE_UNSUPPORTED),
    })
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
        // Stored as typed, as official stores `SET SORTED BY (ID)`; a key
        // binds to its column ignoring case ([`crate::sort::bind_sort_column`]).
        let column = ident.value;

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
            crate::metadata_writer::with_inlining_limit(catalog.data_inlining_row_limit(), || {
                writer.set_partition_spec(table.table_id, &transforms)
            })
            .map_err(DataFusionError::from)?;
        },
        DuckLakeDdl::ResetPartition {
            ..
        } => {
            crate::metadata_writer::with_inlining_limit(catalog.data_inlining_row_limit(), || {
                writer.reset_partition_spec(table.table_id)
            })
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

/// Whether `registered`, a catalog of the session, is `catalog`.
fn is_same_catalog(registered: &Arc<dyn CatalogProvider>, catalog: &DuckLakeCatalog) -> bool {
    registered
        .downcast_ref::<DuckLakeCatalog>()
        .is_some_and(|named| Arc::ptr_eq(&named.provider(), &catalog.provider()))
}

/// The `(schema, table)` a CTAS creates in `catalog`, resolving its name with
/// the session's default catalog and schema as DataFusion does; `None` when the
/// name resolves to another catalog.
fn ctas_target(
    ctx: &SessionContext,
    catalog: &DuckLakeCatalog,
    parts: &[(String, bool)],
) -> Option<(String, String)> {
    let state = ctx.state();
    let defaults = &state.config_options().catalog;
    let norm = |(value, quoted): &(String, bool)| {
        if *quoted {
            value.clone()
        } else {
            value.to_ascii_lowercase()
        }
    };
    let (catalog_name, schema_name, table_name) = match parts {
        [table] => (
            defaults.default_catalog.clone(),
            defaults.default_schema.clone(),
            norm(table),
        ),
        [schema, table] => (defaults.default_catalog.clone(), norm(schema), norm(table)),
        [catalog, schema, table] => (norm(catalog), norm(schema), norm(table)),
        _ => return None,
    };
    let registered = ctx.catalog(&catalog_name)?;
    is_same_catalog(&registered, catalog).then_some((schema_name, table_name))
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
fn writable_target(
    catalog: &DuckLakeCatalog,
    schema_name: String,
    table_name: String,
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
    Ok(WritableTarget {
        writer,
        provider,
        snapshot,
        schema_name,
        table_name,
    })
}

/// Run `CREATE TABLE [IF NOT EXISTS] schema.table [(column, …)] AS query` on the
/// catalog head.
///
/// Planning and the existence check happen before the query executes, so a
/// name clash fails without running it. The rows and the table metadata commit
/// in one snapshot with `head` as the expected base, so a table created
/// concurrently under the same name makes the commit conflict instead of
/// replacing it, as official DuckLake refuses it.
async fn create_table_as(
    ctx: &SessionContext,
    catalog: &DuckLakeCatalog,
    target: &WritableTarget,
    create: CreateTableAs,
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
        if create.if_not_exists {
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
        if create.if_not_exists {
            return Ok(());
        }
        return Err(DataFusionError::Plan(format!(
            "Cannot create table '{table_name}': a view with that name already exists"
        )));
    }

    let state = ctx.state();
    let plan = state
        .statement_to_plan(Statement::Statement(Box::new(SqlStatement::Query(
            create.query,
        ))))
        .await?;
    let arrow_schema = ctas_table_schema(plan.schema().as_arrow(), &create.columns)?;
    let columns = arrow_schema
        .fields()
        .iter()
        .map(|field| ctas_column(field).map_err(DataFusionError::from))
        .collect::<DataFusionResult<Vec<_>>>()?;

    let batches = DataFrame::new(state, plan).collect().await?;
    let batches = batches
        .into_iter()
        .filter(|batch| batch.num_rows() > 0)
        .map(|batch| RecordBatch::try_new(Arc::clone(&arrow_schema), batch.columns().to_vec()))
        .collect::<Result<Vec<_>, _>>()?;
    if batches.is_empty() {
        publish_empty_table(
            writer.as_ref(),
            schema_name,
            table_name,
            &columns,
            catalog.data_inlining_row_limit(),
            ExistingTable::Conflict,
        )?;
        return Ok(());
    }

    let settings = provider
        .get_metadata_settings(Some(schema.schema_id), None)
        .map_err(DataFusionError::from)?;
    let write_options =
        scoped_write_options(&settings, &catalog.write_options(), Some(writer.as_ref()));
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
        .stage_write_with_columns(
            schema_name,
            table_name,
            &arrow_schema,
            &columns,
            WriteMode::Replace,
            &batches,
        )
        .await
        .map_err(DataFusionError::from)?;
    transaction.commit().await.map_err(DataFusionError::from)?;
    Ok(())
}

/// The schema a CTAS table gets from its query, named as DuckDB names it: a
/// column list renames the leading columns and must leave every name unique,
/// and without one a repeated name takes the first free `_1`, `_2`, … suffix.
/// Names compare ignoring case, as DuckDB's do. Every column is nullable,
/// whatever the query's nullability, and field metadata does not belong in the
/// catalog.
fn ctas_table_schema(query_schema: &Schema, columns: &[String]) -> DataFusionResult<Arc<Schema>> {
    if columns.len() > query_schema.fields().len() {
        return Err(DataFusionError::Plan(
            "CREATE TABLE AS SELECT lists more column names than the query returns".to_string(),
        ));
    }
    let mut taken = HashSet::new();
    let mut fields = Vec::with_capacity(query_schema.fields().len());
    for (index, field) in query_schema.fields().iter().enumerate() {
        let name = columns.get(index).unwrap_or(field.name());
        let name = if columns.is_empty() {
            let mut unique = name.clone();
            let mut suffix = 1;
            while taken.contains(&unique.to_ascii_lowercase()) {
                unique = format!("{name}_{suffix}");
                suffix += 1;
            }
            unique
        } else if taken.contains(&name.to_ascii_lowercase()) {
            return Err(DataFusionError::Plan(format!(
                "CREATE TABLE AS SELECT produced duplicate column name '{name}'"
            )));
        } else {
            name.clone()
        };
        taken.insert(name.to_ascii_lowercase());
        fields.push(Field::new(name, field.data_type().clone(), true));
    }
    Ok(Arc::new(Schema::new(fields)))
}

/// The catalog column of `field`, with it and every nested child nullable as
/// official DuckDB records a CTAS column. Arrow keeps a map key non-null, so
/// the children are relaxed in the catalog rows, not in `field`'s type.
fn ctas_column(field: &Field) -> crate::Result<ColumnDef> {
    let column = ColumnDef::from_arrow(field.name(), field.data_type(), true)?;
    let nodes = catalog_column_defs(std::slice::from_ref(&column))?
        .into_iter()
        .map(|def| CatalogNode {
            ducklake_type: def.ducklake_type,
            is_nullable: true,
            initial_default: None,
            default_value: None,
            default_value_type: None,
            default_value_dialect: None,
        })
        .collect();
    ColumnDef::copying(field.name(), field.data_type(), nodes)
}

#[cfg(test)]
mod flush_call_tests {
    use super::*;

    fn call(
        catalog: &str,
        schema_name: Option<&str>,
        table_name: Option<&str>,
    ) -> Option<FlushInlinedDataCall> {
        Some(FlushInlinedDataCall {
            catalog: catalog.to_string(),
            schema_name: schema_name.map(str::to_string),
            table_name: table_name.map(str::to_string),
        })
    }

    #[test]
    fn parses_official_signature() {
        let parse = |sql| parse_flush_inlined_data_call(sql).unwrap();
        assert_eq!(
            parse("CALL ducklake_flush_inlined_data('lake')"),
            call("lake", None, None)
        );
        assert_eq!(
            parse("call DuckLake_Flush_Inlined_Data('lake', table_name => 't');"),
            call("lake", None, Some("t"))
        );
        assert_eq!(
            parse(
                "CALL ducklake_flush_inlined_data('lake', schema_name := 's', table_name => 't')"
            ),
            call("lake", Some("s"), Some("t"))
        );
        assert_eq!(
            parse("-- flush\n  /* all */ CALL ducklake_flush_inlined_data('lake')"),
            call("lake", None, None)
        );
        assert_eq!(parse("CALL other_procedure('lake')"), None);
        assert_eq!(parse("SELECT 1"), None);
    }

    #[test]
    fn rejects_malformed_arguments() {
        for sql in [
            "CALL ducklake_flush_inlined_data()",
            "CALL ducklake_flush_inlined_data('a', 'b')",
            "CALL ducklake_flush_inlined_data(lake)",
            "CALL ducklake_flush_inlined_data('lake', table => 't')",
        ] {
            assert!(parse_flush_inlined_data_call(sql).is_err(), "{sql}");
        }
    }
}
