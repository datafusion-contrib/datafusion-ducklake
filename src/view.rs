use std::cell::RefCell;
use std::ops::ControlFlow;
use std::str::FromStr;
use std::sync::Arc;

use arrow::datatypes::{Schema, SchemaRef};
use datafusion::catalog::Session;
use datafusion::common::Column;
use datafusion::common::config::Dialect;
use datafusion::datasource::{TableProvider, ViewTable};
use datafusion::error::{DataFusionError, Result};
use datafusion::logical_expr::{Expr, LogicalPlan, LogicalPlanBuilder, TableType};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::prelude::{SQLOptions, SessionConfig, SessionContext};
use datafusion::sql::sqlparser::ast::{Ident, ObjectNamePart, visit_relations_mut};
use datafusion::sql::sqlparser::dialect::dialect_from_str;
use datafusion::sql::sqlparser::parser::Parser;
use datafusion::sql::sqlparser::tokenizer::{Token, Tokenizer};

use crate::catalog::DuckLakeCatalog;
use crate::metadata_provider::{MetadataProvider, SchemaMetadata, ViewMetadata};

const VIEW_CATALOG: &str = "__ducklake_view";

#[derive(Debug)]
pub(crate) struct UnplannableViewTable {
    definition: String,
    error: String,
    schema: SchemaRef,
}

impl UnplannableViewTable {
    pub(crate) fn new(definition: String, error: &DataFusionError) -> Self {
        Self {
            definition,
            error: error.to_string(),
            schema: Arc::new(Schema::empty()),
        }
    }
}

#[async_trait::async_trait]
impl TableProvider for UnplannableViewTable {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn table_type(&self) -> TableType {
        TableType::View
    }

    fn get_table_definition(&self) -> Option<&str> {
        Some(&self.definition)
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        _projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Err(DataFusionError::Plan(self.error.clone()))
    }
}

tokio::task_local! {
    static VIEW_PLAN_STACK: RefCell<Vec<i64>>;
}

pub(crate) async fn plan_view(
    view: &ViewMetadata,
    sql: &str,
    provider: Arc<dyn MetadataProvider>,
    snapshot_id: i64,
    schema_name: &str,
    row_lineage: bool,
) -> Result<Arc<dyn TableProvider>> {
    let cycle = VIEW_PLAN_STACK
        .try_with(|stack| stack.borrow().contains(&view.view_id))
        .unwrap_or(false);
    if cycle {
        return Err(view_error(view, "view dependency cycle"));
    }

    if VIEW_PLAN_STACK
        .try_with(|stack| stack.borrow_mut().push(view.view_id))
        .is_ok()
    {
        let result =
            plan_view_inner(view, sql, provider, snapshot_id, schema_name, row_lineage).await;
        VIEW_PLAN_STACK.with(|stack| {
            stack.borrow_mut().pop();
        });
        result
    } else {
        VIEW_PLAN_STACK
            .scope(
                RefCell::new(vec![view.view_id]),
                plan_view_inner(view, sql, provider, snapshot_id, schema_name, row_lineage),
            )
            .await
    }
}

async fn plan_view_inner(
    view: &ViewMetadata,
    sql: &str,
    provider: Arc<dyn MetadataProvider>,
    snapshot_id: i64,
    schema_name: &str,
    row_lineage: bool,
) -> Result<Arc<dyn TableProvider>> {
    let dialect = Dialect::from_str(&view.dialect).map_err(|e| view_error(view, e))?;
    let mut config = SessionConfig::new()
        .with_create_default_catalog_and_schema(false)
        .with_default_catalog_and_schema(VIEW_CATALOG, schema_name);
    config.options_mut().sql_parser.dialect = dialect;

    let context = SessionContext::new_with_config(config);
    let catalog = DuckLakeCatalog::with_snapshot(provider, snapshot_id)
        .map_err(|e| view_error(view, e))?
        .with_row_lineage(row_lineage);
    context.register_catalog(VIEW_CATALOG, Arc::new(catalog));

    let options = SQLOptions::new()
        .with_allow_ddl(false)
        .with_allow_dml(false)
        .with_allow_statements(false);
    let plan = context
        .sql_with_options(sql, options)
        .await
        .map_err(|e| view_error(view, e))?
        .into_unoptimized_plan();
    let plan = apply_aliases(plan, view).map_err(|e| view_error(view, e))?;

    Ok(Arc::new(ViewTable::new(plan, Some(sql.to_string()))))
}

fn apply_aliases(plan: LogicalPlan, view: &ViewMetadata) -> Result<LogicalPlan> {
    let aliases = parse_aliases(view.column_aliases.as_deref().unwrap_or_default())?;
    if aliases.is_empty() {
        return Ok(plan);
    }
    if aliases.len() > plan.schema().fields().len() {
        return Err(DataFusionError::Plan(format!(
            "view defines {} aliases for {} columns",
            aliases.len(),
            plan.schema().fields().len()
        )));
    }

    let expressions = plan
        .schema()
        .fields()
        .iter()
        .enumerate()
        .map(|(index, _)| {
            let column = Expr::Column(Column::from(plan.schema().qualified_field(index)));
            aliases
                .get(index)
                .map_or(column.clone(), |alias| column.alias(alias))
        })
        .collect::<Vec<_>>();
    LogicalPlanBuilder::from(plan).project(expressions)?.build()
}

fn parse_aliases(input: &str) -> Result<Vec<String>> {
    if input.is_empty() {
        return Ok(Vec::new());
    }

    let mut aliases = Vec::new();
    let mut chars = input.chars().peekable();
    loop {
        if chars.next() != Some('"') {
            return Err(DataFusionError::Plan(
                "column_aliases must contain quoted identifiers".to_string(),
            ));
        }

        let mut alias = String::new();
        loop {
            match chars.next() {
                Some('"') if chars.peek() == Some(&'"') => {
                    chars.next();
                    alias.push('"');
                },
                Some('"') => break,
                Some(character) => alias.push(character),
                None => {
                    return Err(DataFusionError::Plan(
                        "column_aliases contains an unterminated identifier".to_string(),
                    ));
                },
            }
        }
        aliases.push(alias);

        match chars.next() {
            Some(',') => {},
            None => break,
            Some(_) => {
                return Err(DataFusionError::Plan(
                    "column_aliases must separate identifiers with commas".to_string(),
                ));
            },
        }
    }
    Ok(aliases)
}

pub(crate) fn resolve_view_definition(
    view: &ViewMetadata,
    provider: &dyn MetadataProvider,
    snapshot_id: i64,
    schema_name: &str,
) -> Result<String> {
    let dialect = Dialect::from_str(&view.dialect).map_err(|e| view_error(view, e))?;
    let (creation_schemas, visible_schemas) = if dialect == Dialect::DuckDB {
        let creation_schemas = provider
            .list_schemas(view.begin_snapshot)
            .map_err(|e| view_error(view, e))?;
        let visible_schemas = if snapshot_id == view.begin_snapshot {
            creation_schemas.clone()
        } else {
            provider
                .list_schemas(snapshot_id)
                .map_err(|e| view_error(view, e))?
        };
        (creation_schemas, visible_schemas)
    } else {
        (Vec::new(), Vec::new())
    };
    resolve_view_sql(
        &view.sql,
        schema_name,
        dialect,
        &creation_schemas,
        &visible_schemas,
    )
    .map_err(|e| view_error(view, e))
}

fn resolve_view_sql(
    sql: &str,
    schema_name: &str,
    dialect: Dialect,
    creation_schemas: &[SchemaMetadata],
    visible_schemas: &[SchemaMetadata],
) -> crate::Result<String> {
    let is_duckdb = dialect == Dialect::DuckDB;
    let dialect = dialect_from_str(dialect.as_ref()).ok_or_else(|| {
        crate::DuckLakeError::Unsupported(format!("SQL dialect '{}'", dialect.as_ref()))
    })?;
    let tokens = match Tokenizer::new(dialect.as_ref(), sql)
        .with_unescape(false)
        .tokenize()
    {
        Ok(tokens) => tokens,
        Err(_) => return Ok(sql.to_string()),
    };

    let quoted_schema = Ident::with_quote('"', schema_name).to_string();
    let mut resolved = String::with_capacity(sql.len() + quoted_schema.len());
    let mut index = 0;
    while index < tokens.len() {
        if is_catalog_placeholder(&tokens, index) {
            resolved.push_str(VIEW_CATALOG);
            resolved.push('.');
            let first_part = next_non_whitespace(&tokens, index + 4);
            let second_part = first_part.and_then(|part| next_non_whitespace(&tokens, part + 1));
            if first_part.is_some_and(|part| matches!(tokens[part], Token::Word(_)))
                && second_part.is_none_or(|part| tokens[part] != Token::Period)
            {
                resolved.push_str(&quoted_schema);
                resolved.push('.');
            }
            index += 4;
            continue;
        }

        if reserved_catalog_qualifier(&tokens, index) {
            return Err(crate::DuckLakeError::Unsupported(format!(
                "reserved view catalog qualifier '{VIEW_CATALOG}'"
            )));
        }
        resolved.push_str(&tokens[index].to_string());
        index += 1;
    }

    let mut statements = match Parser::parse_sql(dialect.as_ref(), &resolved) {
        Ok(statements) => statements,
        Err(_) => return Ok(resolved),
    };
    let mut changed = false;
    if let ControlFlow::Break(relation) = visit_relations_mut(&mut statements, |relation| {
        let parts = relation
            .0
            .iter()
            .map(|part| part.as_ident().cloned())
            .collect::<Option<Vec<_>>>();
        let Some(parts) = parts else {
            return ControlFlow::Break(relation.to_string());
        };
        match parts.as_slice() {
            [_] => ControlFlow::Continue(()),
            [schema, _] if is_duckdb => {
                let canonical = match canonical_schema(schema, creation_schemas, visible_schemas) {
                    Ok(canonical) => canonical,
                    Err(()) => return ControlFlow::Break(relation.to_string()),
                };
                relation
                    .0
                    .insert(0, ObjectNamePart::Identifier(Ident::new(VIEW_CATALOG)));
                relation.0[1] = ObjectNamePart::Identifier(Ident::with_quote('"', canonical));
                changed = true;
                ControlFlow::Continue(())
            },
            [catalog, schema, _]
                if catalog.quote_style.is_none() && catalog.value == VIEW_CATALOG =>
            {
                if is_duckdb {
                    let canonical =
                        match canonical_schema(schema, creation_schemas, visible_schemas) {
                            Ok(canonical) => canonical,
                            Err(()) => return ControlFlow::Break(relation.to_string()),
                        };
                    let canonical = ObjectNamePart::Identifier(Ident::with_quote('"', canonical));
                    if relation.0[1] != canonical {
                        relation.0[1] = canonical;
                        changed = true;
                    }
                }
                ControlFlow::Continue(())
            },
            _ => ControlFlow::Break(relation.to_string()),
        }
    }) {
        return Err(crate::DuckLakeError::Unsupported(format!(
            "ambiguous catalog-qualified relation '{relation}'"
        )));
    }
    if changed {
        Ok(statements
            .into_iter()
            .map(|statement| statement.to_string())
            .collect::<Vec<_>>()
            .join("; "))
    } else {
        Ok(resolved)
    }
}

fn canonical_schema<'a>(
    identifier: &Ident,
    creation_schemas: &[SchemaMetadata],
    visible_schemas: &'a [SchemaMetadata],
) -> std::result::Result<&'a str, ()> {
    let mut creation_matches = creation_schemas
        .iter()
        .filter(|schema| schema.schema_name.eq_ignore_ascii_case(&identifier.value));
    let created = creation_matches.next().ok_or(())?;
    if creation_matches.next().is_some() {
        return Err(());
    }
    let mut visible_matches = visible_schemas
        .iter()
        .filter(|schema| schema.schema_id == created.schema_id);
    let visible = visible_matches.next().ok_or(())?;
    if visible_matches.next().is_some() {
        return Err(());
    }
    visible
        .schema_name
        .eq_ignore_ascii_case(&identifier.value)
        .then_some(visible.schema_name.as_str())
        .ok_or(())
}

fn is_catalog_placeholder(tokens: &[Token], index: usize) -> bool {
    if index
        .checked_sub(1)
        .is_some_and(|previous| matches!(tokens[previous], Token::Word(_)))
    {
        return false;
    }
    matches!(
        tokens.get(index..index + 4),
        Some([Token::LBrace, Token::Word(word), Token::RBrace, Token::Period])
            if word.quote_style.is_none() && word.value == "DUCKLAKE_CATALOG"
    )
}

fn next_non_whitespace(tokens: &[Token], start: usize) -> Option<usize> {
    (start..tokens.len()).find(|index| !matches!(tokens[*index], Token::Whitespace(_)))
}

fn reserved_catalog_qualifier(tokens: &[Token], index: usize) -> bool {
    let Token::Word(word) = &tokens[index] else {
        return false;
    };
    word.value == VIEW_CATALOG
        && next_non_whitespace(tokens, index + 1).is_some_and(|next| tokens[next] == Token::Period)
}

fn view_error(view: &ViewMetadata, error: impl std::fmt::Display) -> DataFusionError {
    DataFusionError::Plan(format!(
        "DuckLake view '{}' with dialect '{}' could not be planned: {error}",
        view.view_name, view.dialect
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::logical_expr::lit;

    fn schema(schema_id: i64, schema_name: &str) -> SchemaMetadata {
        SchemaMetadata {
            schema_id,
            schema_name: schema_name.to_string(),
            path: String::new(),
            path_is_relative: true,
        }
    }

    fn schemas() -> Vec<SchemaMetadata> {
        vec![
            schema(1, "main"),
            schema(2, "s1"),
            schema(3, "MySchema"),
            schema(4, "odd schema\"name"),
        ]
    }

    fn resolve(sql: &str, schema_name: &str) -> crate::Result<String> {
        let schemas = schemas();
        resolve_view_sql(sql, schema_name, Dialect::DuckDB, &schemas, &schemas)
    }

    #[test]
    fn parses_quoted_aliases() {
        assert_eq!(
            parse_aliases(r#""left","embedded""quote","right""#).unwrap(),
            vec!["left", "embedded\"quote", "right"]
        );
    }

    #[test]
    fn resolves_only_unquoted_catalog_placeholders() {
        assert_eq!(
            resolve(
                "SELECT * FROM {DUCKLAKE_CATALOG}.t WHERE note = '{DUCKLAKE_CATALOG}.t'",
                "main",
            )
            .unwrap(),
            "SELECT * FROM __ducklake_view.\"main\".t WHERE note = '{DUCKLAKE_CATALOG}.t'"
        );
        assert_eq!(
            resolve(
                "SELECT my{DUCKLAKE_CATALOG}.id FROM {DUCKLAKE_CATALOG}.main.t AS mydb",
                "main",
            )
            .unwrap(),
            "SELECT my{DUCKLAKE_CATALOG}.id FROM __ducklake_view.main.t AS mydb"
        );
        assert_eq!(
            resolve("SELECT * FROM other_db.main.events", "main")
                .unwrap_err()
                .to_string(),
            "Unsupported feature: ambiguous catalog-qualified relation 'other_db.main.events'"
        );
    }

    #[test]
    fn preserves_placeholders_in_quoted_text_and_comments() {
        let sql = "SELECT '{DUCKLAKE_CATALOG}.literal', $$ {DUCKLAKE_CATALOG}.dollar $$\n\
                   FROM {DUCKLAKE_CATALOG}.items -- don't rewrite {DUCKLAKE_CATALOG}.comment\n";
        let resolved = resolve(sql, "odd schema\"name").unwrap();
        assert!(resolved.contains("'{DUCKLAKE_CATALOG}.literal'"));
        assert!(resolved.contains("$$ {DUCKLAKE_CATALOG}.dollar $$"));
        assert!(resolved.contains("-- don't rewrite {DUCKLAKE_CATALOG}.comment"));
        assert!(resolved.contains("FROM __ducklake_view.\"odd schema\"\"name\".items"));
    }

    #[test]
    fn rejects_reserved_and_legacy_qualifiers() {
        for sql in [
            "SELECT * FROM lake.users",
            "SELECT * FROM {DUCKLAKE_CATALOG}.users JOIN __ducklake_view.audit_log USING (id)",
            "SELECT * FROM {DUCKLAKE_CATALOG}.users JOIN \"__ducklake_view\".audit_log USING (id)",
        ] {
            assert!(resolve(sql, "main").is_err());
        }
    }

    #[test]
    fn resolves_creation_time_schema_qualifiers_and_canonical_case() {
        assert_eq!(
            resolve("SELECT * FROM main.t JOIN s1.u USING (id)", "main").unwrap(),
            "SELECT * FROM __ducklake_view.\"main\".t JOIN __ducklake_view.\"s1\".u USING(id)"
        );
        assert_eq!(
            resolve("SELECT * FROM {DUCKLAKE_CATALOG}.MySchema.m", "main",).unwrap(),
            "SELECT * FROM __ducklake_view.\"MySchema\".m"
        );
    }

    #[test]
    fn rejects_later_schema_collisions_and_recreated_schemas() {
        let creation = schemas();
        let mut visible = schemas();
        visible.push(schema(5, "ext"));
        assert!(
            resolve_view_sql(
                "SELECT * FROM ext.t",
                "main",
                Dialect::DuckDB,
                &creation,
                &visible,
            )
            .is_err()
        );

        let visible = vec![schema(1, "main"), schema(9, "s1")];
        assert!(
            resolve_view_sql(
                "SELECT * FROM s1.u",
                "main",
                Dialect::DuckDB,
                &creation,
                &visible,
            )
            .is_err()
        );
    }

    #[test]
    fn applies_partial_alias_list() {
        let plan = LogicalPlanBuilder::empty(false)
            .project(vec![lit(1).alias("first"), lit(2).alias("second")])
            .unwrap()
            .build()
            .unwrap();
        let view = ViewMetadata {
            view_id: 1,
            schema_id: 1,
            begin_snapshot: 1,
            view_name: "partial".to_string(),
            dialect: "duckdb".to_string(),
            sql: "SELECT 1, 2".to_string(),
            column_aliases: Some("\"renamed\"".to_string()),
        };

        let plan = apply_aliases(plan, &view).unwrap();
        assert_eq!(
            plan.schema()
                .fields()
                .iter()
                .map(|field| field.name().as_str())
                .collect::<Vec<_>>(),
            vec!["renamed", "second"]
        );
    }
}
