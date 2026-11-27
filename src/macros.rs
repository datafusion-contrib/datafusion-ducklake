use std::collections::HashMap;
use std::ops::ControlFlow;
use std::str::FromStr;
use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema};
use datafusion::common::config::Dialect;
use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::common::{DFSchema, DataFusionError, Result};
use datafusion::logical_expr::expr::Cast;
use datafusion::logical_expr::expr_rewriter::FunctionRewrite;
use datafusion::logical_expr::planner::{
    PlannedRelation, RelationPlanner, RelationPlannerContext, RelationPlanning,
};
use datafusion::logical_expr::registry::FunctionRegistry;
use datafusion::logical_expr::{
    ColumnarValue, Expr, LogicalPlan, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature,
    TypeSignature, Volatility,
};
use datafusion::prelude::{SQLOptions, SessionConfig, SessionContext};
use datafusion::sql::sqlparser::ast::{
    Expr as SqlExpr, FunctionArg, FunctionArgExpr, ObjectName, Query, Statement, TableFactor,
    Value, visit_expressions_mut,
};
use datafusion::sql::sqlparser::dialect::DuckDbDialect;
use datafusion::sql::sqlparser::parser::Parser;

use crate::catalog::DuckLakeCatalog;
use crate::metadata_provider::{
    MacroImplementationMetadata, MacroMetadata, MacroParameterMetadata, MetadataProvider,
};
use crate::types::ducklake_to_arrow_type;
use crate::view::{VIEW_CATALOG, resolve_view_sql};

#[derive(Debug, Clone)]
struct Parameter {
    name: String,
    data_type: Option<DataType>,
    default_sql: Option<String>,
}

#[derive(Debug, Clone)]
struct ScalarImplementation {
    impl_id: i64,
    dialect: String,
    parameters: Vec<Parameter>,
    body: std::result::Result<Expr, String>,
    defaults: Vec<Option<std::result::Result<Expr, String>>>,
}

#[derive(Debug, Clone)]
struct TableImplementation {
    impl_id: i64,
    dialect: String,
    parameters: Vec<Parameter>,
    plan: std::result::Result<LogicalPlan, String>,
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct MacroFunction {
    name: String,
    signature: Signature,
}

impl MacroFunction {
    fn new(name: String) -> Self {
        Self {
            name,
            signature: Signature::one_of(
                vec![TypeSignature::Any(0), TypeSignature::VariadicAny],
                Volatility::Immutable,
            ),
        }
    }
}

impl ScalarUDFImpl for MacroFunction {
    fn name(&self) -> &str {
        &self.name
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        Ok(arg_types.first().cloned().unwrap_or(DataType::Null))
    }

    fn invoke_with_args(&self, _args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        Err(DataFusionError::Internal(format!(
            "DuckLake macro '{}' was not expanded during planning",
            self.name
        )))
    }
}

#[derive(Debug)]
struct ScalarMacroRewrite {
    macros: HashMap<String, Vec<ScalarImplementation>>,
}

impl FunctionRewrite for ScalarMacroRewrite {
    fn name(&self) -> &str {
        "ducklake_scalar_macros"
    }

    fn rewrite(
        &self,
        expr: Expr,
        _schema: &DFSchema,
        _config: &datafusion::common::config::ConfigOptions,
    ) -> Result<Transformed<Expr>> {
        let Expr::ScalarFunction(function) = expr else {
            return Ok(Transformed::no(expr));
        };
        let Some(implementations) = self.macros.get(function.name()) else {
            return Ok(Transformed::no(Expr::ScalarFunction(function)));
        };
        let implementation = select_scalar_implementation(implementations, function.args.len())
            .ok_or_else(|| {
                macro_error(
                    function.name(),
                    implementations
                        .first()
                        .map(|item| item.dialect.as_str())
                        .unwrap_or("unknown"),
                    format!(
                        "no implementation accepts {} arguments",
                        function.args.len()
                    ),
                )
            })?;
        let function_name = function.name().to_string();
        let expanded = expand_scalar(function.args, implementation)
            .map_err(|e| macro_error(&function_name, &implementation.dialect, e))?;
        Ok(Transformed::yes(expanded))
    }
}

#[derive(Debug)]
struct TableMacroPlanner {
    catalog_name: String,
    macros: HashMap<(String, String), Vec<TableImplementation>>,
    unqualified: HashMap<String, Option<String>>,
}

impl RelationPlanner for TableMacroPlanner {
    fn plan_relation(
        &self,
        relation: TableFactor,
        context: &mut dyn RelationPlannerContext,
    ) -> Result<RelationPlanning> {
        let original = relation.clone();
        let TableFactor::Table {
            name,
            alias,
            args: Some(args),
            ..
        } = relation
        else {
            return Ok(RelationPlanning::Original(Box::new(relation)));
        };
        let Some((schema_name, macro_name)) = self.macro_key(&name) else {
            return Ok(RelationPlanning::Original(Box::new(original)));
        };
        let implementations = &self.macros[&(schema_name.clone(), macro_name.clone())];
        let (implementation, arguments) = implementations
            .iter()
            .filter_map(|implementation| {
                resolve_logical_arguments(&implementation.parameters, &args.args, context)
                    .ok()
                    .map(|arguments| (implementation, arguments))
            })
            .min_by_key(|(implementation, _)| {
                (
                    implementation.parameters.len() != args.args.len(),
                    implementation
                        .parameters
                        .len()
                        .saturating_sub(args.args.len()),
                    implementation.impl_id,
                )
            })
            .ok_or_else(|| {
                macro_error(
                    &macro_name,
                    implementations
                        .first()
                        .map(|item| item.dialect.as_str())
                        .unwrap_or("unknown"),
                    format!("no implementation accepts {} arguments", args.args.len()),
                )
            })?;
        let plan = implementation
            .plan
            .clone()
            .map_err(|e| macro_error(&macro_name, &implementation.dialect, e))?;
        let plan = substitute_logical_parameters(plan, &implementation.parameters, arguments)
            .map_err(|e| macro_error(&macro_name, &implementation.dialect, e))?;
        Ok(RelationPlanning::Planned(Box::new(PlannedRelation::new(
            plan, alias,
        ))))
    }
}

impl TableMacroPlanner {
    fn macro_key(&self, name: &ObjectName) -> Option<(String, String)> {
        let parts = name
            .0
            .iter()
            .map(|part| part.as_ident().map(|ident| ident.value.clone()))
            .collect::<Option<Vec<_>>>()?;
        match parts.as_slice() {
            [macro_name] => self
                .unqualified
                .get(macro_name)
                .and_then(Clone::clone)
                .map(|schema_name| (schema_name, macro_name.clone())),
            [schema_name, macro_name] => Some((schema_name.clone(), macro_name.clone())),
            [catalog_name, schema_name, macro_name] if catalog_name == &self.catalog_name => {
                Some((schema_name.clone(), macro_name.clone()))
            },
            _ => None,
        }
        .filter(|key| self.macros.contains_key(key))
    }
}

/// Register snapshot-visible DuckLake scalar and table macros in a DataFusion context.
pub async fn register_ducklake_macros(
    context: &SessionContext,
    catalog_name: impl Into<String>,
    provider: Arc<dyn MetadataProvider>,
) -> Result<()> {
    let catalog_name = catalog_name.into();
    let snapshot_id = provider
        .get_current_snapshot()
        .map_err(|e| DataFusionError::External(Box::new(e)))?;
    let schemas = provider
        .list_schemas(snapshot_id)
        .map_err(|e| DataFusionError::External(Box::new(e)))?;
    let mut scalar_macros_by_schema: HashMap<(String, String), Vec<ScalarImplementation>> =
        HashMap::new();
    let mut table_macros: HashMap<(String, String), Vec<TableImplementation>> = HashMap::new();
    let mut unqualified: HashMap<String, Option<String>> = HashMap::new();

    for schema in schemas {
        let macros = provider
            .list_macros(schema.schema_id, snapshot_id)
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        for macro_metadata in macros {
            let macro_name = macro_metadata.macro_name.clone();
            prepare_macro(
                context,
                provider.clone(),
                snapshot_id,
                &schema.schema_name,
                macro_metadata,
                &mut scalar_macros_by_schema,
                &mut table_macros,
            )
            .await;
            unqualified
                .entry(macro_name)
                .and_modify(|schema_name| *schema_name = None)
                .or_insert_with(|| Some(schema.schema_name.clone()));
        }
    }

    let mut scalar_macros = HashMap::new();
    for ((schema_name, macro_name), implementations) in scalar_macros_by_schema {
        scalar_macros.insert(
            format!("{schema_name}.{macro_name}"),
            implementations.clone(),
        );
        scalar_macros.insert(
            format!("{catalog_name}.{schema_name}.{macro_name}"),
            implementations.clone(),
        );
        if unqualified.get(&macro_name).and_then(Clone::clone) == Some(schema_name) {
            scalar_macros.insert(macro_name, implementations);
        }
    }
    for name in scalar_macros.keys() {
        context.register_udf(ScalarUDF::new_from_impl(MacroFunction::new(name.clone())));
    }
    if !scalar_macros.is_empty() {
        let mut registry = context.clone();
        registry.register_function_rewrite(Arc::new(ScalarMacroRewrite {
            macros: scalar_macros,
        }))?;
    }
    if !table_macros.is_empty() {
        context.register_relation_planner(Arc::new(TableMacroPlanner {
            catalog_name,
            macros: table_macros,
            unqualified,
        }))?;
    }
    Ok(())
}

async fn prepare_macro(
    context: &SessionContext,
    provider: Arc<dyn MetadataProvider>,
    snapshot_id: i64,
    schema_name: &str,
    macro_metadata: MacroMetadata,
    scalar_macros: &mut HashMap<(String, String), Vec<ScalarImplementation>>,
    table_macros: &mut HashMap<(String, String), Vec<TableImplementation>>,
) {
    for implementation in macro_metadata.implementations {
        match implementation.macro_type.as_str() {
            "scalar" => scalar_macros
                .entry((schema_name.to_string(), macro_metadata.macro_name.clone()))
                .or_default()
                .push(prepare_scalar(context, implementation)),
            "table" => {
                let implementation =
                    prepare_table(provider.clone(), snapshot_id, schema_name, implementation).await;
                table_macros
                    .entry((schema_name.to_string(), macro_metadata.macro_name.clone()))
                    .or_default()
                    .push(implementation);
            },
            _ => {},
        }
    }
}

fn prepare_scalar(
    context: &SessionContext,
    implementation: MacroImplementationMetadata,
) -> ScalarImplementation {
    let (parameters, type_error) = prepare_parameters(&implementation.parameters);
    let dialect = Dialect::from_str(&implementation.dialect);
    let body = type_error
        .map_or_else(|| dialect.as_ref().map_err(ToString::to_string), Err)
        .and_then(|dialect| {
            let fields = parameters
                .iter()
                .map(|parameter| {
                    Field::new(
                        &parameter.name,
                        parameter.data_type.clone().unwrap_or(DataType::Null),
                        true,
                    )
                })
                .collect::<Vec<_>>();
            let schema = DFSchema::try_from(Schema::new(fields)).map_err(|e| e.to_string())?;
            let state = context.state();
            let expression = state
                .sql_to_expr_with_alias(&implementation.sql, dialect)
                .map_err(|e| e.to_string())?;
            state
                .create_logical_expr_from_sql_expr(expression, &schema)
                .map_err(|e| e.to_string())
        });
    let defaults = parameters
        .iter()
        .map(|parameter| {
            parameter.default_sql.as_ref().map(|sql| {
                dialect
                    .as_ref()
                    .map_err(ToString::to_string)
                    .and_then(|dialect| {
                        let state = context.state();
                        let expression = state
                            .sql_to_expr_with_alias(sql, dialect)
                            .map_err(|e| e.to_string())?;
                        state
                            .create_logical_expr_from_sql_expr(expression, &DFSchema::empty())
                            .map_err(|e| e.to_string())
                    })
            })
        })
        .collect();
    ScalarImplementation {
        impl_id: implementation.impl_id,
        dialect: implementation.dialect,
        parameters,
        body,
        defaults,
    }
}

async fn prepare_table(
    provider: Arc<dyn MetadataProvider>,
    snapshot_id: i64,
    schema_name: &str,
    implementation: MacroImplementationMetadata,
) -> TableImplementation {
    let (parameters, type_error) = prepare_parameters(&implementation.parameters);
    let plan = type_error
        .map_or_else(
            || Dialect::from_str(&implementation.dialect).map_err(|e| e.to_string()),
            Err,
        )
        .and_then(|dialect| {
            if dialect != Dialect::DuckDB {
                return Err("only DuckDB stored macro SQL is supported".to_string());
            }
            let schemas = provider
                .list_schemas(snapshot_id)
                .map_err(|e| e.to_string())?;
            resolve_view_sql(
                &implementation.sql,
                schema_name,
                dialect,
                &schemas,
                &schemas,
            )
            .map_err(|e| e.to_string())
        })
        .and_then(|sql| parse_query(&sql))
        .map(|mut query| {
            substitute_sql_placeholders(&mut query, &parameters);
            query
        });
    let plan = match plan {
        Ok(query) => {
            let dialect = Dialect::from_str(&implementation.dialect).expect("validated dialect");
            let mut config = SessionConfig::new()
                .with_create_default_catalog_and_schema(false)
                .with_default_catalog_and_schema(VIEW_CATALOG, schema_name);
            config.options_mut().sql_parser.dialect = dialect;
            let context = SessionContext::new_with_config(config);
            match DuckLakeCatalog::with_snapshot(provider, snapshot_id) {
                Ok(catalog) => {
                    context.register_catalog(VIEW_CATALOG, Arc::new(catalog));
                    let options = SQLOptions::new()
                        .with_allow_ddl(false)
                        .with_allow_dml(false)
                        .with_allow_statements(false);
                    context
                        .sql_with_options(&query.to_string(), options)
                        .await
                        .map(|frame| frame.into_unoptimized_plan())
                        .map_err(|e| e.to_string())
                },
                Err(e) => Err(e.to_string()),
            }
        },
        Err(e) => Err(e),
    };
    TableImplementation {
        impl_id: implementation.impl_id,
        dialect: implementation.dialect,
        parameters,
        plan,
    }
}

fn prepare_parameters(parameters: &[MacroParameterMetadata]) -> (Vec<Parameter>, Option<String>) {
    let mut type_error = None;
    let parameters = parameters
        .iter()
        .map(|parameter| {
            let data_type = if parameter.parameter_type == "unknown" {
                None
            } else {
                match ducklake_to_arrow_type(&parameter.parameter_type) {
                    Ok(data_type) => Some(data_type),
                    Err(e) => {
                        type_error.get_or_insert_with(|| {
                            format!(
                                "parameter '{}' has unsupported type '{}': {e}",
                                parameter.parameter_name, parameter.parameter_type
                            )
                        });
                        None
                    },
                }
            };
            Parameter {
                name: parameter.parameter_name.clone(),
                data_type,
                default_sql: parameter.default_value.clone(),
            }
        })
        .collect();
    (parameters, type_error)
}

fn parse_query(sql: &str) -> std::result::Result<Box<Query>, String> {
    let mut statements = Parser::parse_sql(&DuckDbDialect {}, sql).map_err(|e| e.to_string())?;
    if statements.len() != 1 {
        return Err("stored SQL must contain one query".to_string());
    }
    match statements.remove(0) {
        Statement::Query(query) => Ok(query),
        _ => Err("stored SQL is not a query".to_string()),
    }
}

fn substitute_sql_placeholders(query: &mut Query, parameters: &[Parameter]) {
    let bindings = parameters
        .iter()
        .enumerate()
        .map(|(index, parameter)| (parameter.name.as_str(), format!("${}", index + 1)))
        .collect::<HashMap<_, _>>();
    let _ = visit_expressions_mut(query, |expression| {
        if let SqlExpr::Identifier(identifier) = expression
            && let Some(placeholder) = bindings.get(identifier.value.as_str())
        {
            *expression = SqlExpr::Value(Value::Placeholder(placeholder.clone()).into());
        }
        ControlFlow::<()>::Continue(())
    });
}

fn select_scalar_implementation(
    implementations: &[ScalarImplementation],
    argument_count: usize,
) -> Option<&ScalarImplementation> {
    implementations
        .iter()
        .filter(|implementation| accepts(&implementation.parameters, argument_count))
        .min_by_key(|implementation| {
            (
                implementation.parameters.len() != argument_count,
                implementation.parameters.len() - argument_count,
                implementation.impl_id,
            )
        })
}

fn accepts(parameters: &[Parameter], argument_count: usize) -> bool {
    argument_count <= parameters.len()
        && parameters[argument_count..]
            .iter()
            .all(|parameter| parameter.default_sql.is_some())
}

fn expand_scalar(arguments: Vec<Expr>, implementation: &ScalarImplementation) -> Result<Expr> {
    let mut bindings = HashMap::new();
    for (index, parameter) in implementation.parameters.iter().enumerate() {
        let argument = if let Some(argument) = arguments.get(index) {
            argument.clone()
        } else {
            implementation.defaults[index]
                .clone()
                .ok_or_else(|| {
                    DataFusionError::Plan(format!("missing argument '{}'", parameter.name))
                })?
                .map_err(DataFusionError::Plan)?
        };
        let argument = parameter
            .data_type
            .as_ref()
            .map_or(argument.clone(), |data_type| {
                Expr::Cast(Cast::new(Box::new(argument), data_type.clone()))
            });
        bindings.insert(parameter.name.clone(), argument);
    }
    implementation
        .body
        .clone()
        .map_err(DataFusionError::Plan)?
        .transform_up(|expr| match expr {
            Expr::Column(column) if column.relation.is_none() => {
                bindings.get(&column.name).cloned().map_or_else(
                    || Ok(Transformed::no(Expr::Column(column))),
                    |expr| Ok(Transformed::yes(expr)),
                )
            },
            expr => Ok(Transformed::no(expr)),
        })
        .map(|transformed| transformed.data)
}

fn resolve_logical_arguments(
    parameters: &[Parameter],
    arguments: &[FunctionArg],
    context: &mut dyn RelationPlannerContext,
) -> std::result::Result<Vec<Expr>, String> {
    let mut resolved = vec![None; parameters.len()];
    let mut positional = 0;
    for argument in arguments {
        let (position, value) = match argument {
            FunctionArg::Unnamed(FunctionArgExpr::Expr(value)) => {
                while resolved.get(positional).is_some_and(Option::is_some) {
                    positional += 1;
                }
                (positional, value.clone())
            },
            FunctionArg::Named {
                name,
                arg: FunctionArgExpr::Expr(value),
                ..
            } => {
                let position = parameters
                    .iter()
                    .position(|parameter| parameter.name == name.value)
                    .ok_or_else(|| format!("unknown argument '{}'", name.value))?;
                (position, value.clone())
            },
            _ => return Err("macro arguments must be scalar expressions".to_string()),
        };
        let Some(slot) = resolved.get_mut(position) else {
            return Err("too many macro arguments".to_string());
        };
        if slot.replace(value).is_some() {
            return Err(format!(
                "argument '{}' was provided twice",
                parameters[position].name
            ));
        }
        positional += 1;
    }
    parameters
        .iter()
        .zip(resolved)
        .map(|(parameter, value)| {
            let expression = match value {
                Some(expression) => expression,
                None => parse_sql_expr(
                    parameter
                        .default_sql
                        .as_deref()
                        .ok_or_else(|| format!("missing argument '{}'", parameter.name))?,
                )?,
            };
            context
                .sql_to_expr(expression, &DFSchema::empty())
                .map_err(|e| e.to_string())
        })
        .collect()
}

fn parse_sql_expr(sql: &str) -> std::result::Result<SqlExpr, String> {
    Parser::new(&DuckDbDialect {})
        .try_with_sql(sql)
        .map_err(|e| e.to_string())?
        .parse_expr()
        .map_err(|e| e.to_string())
}

fn substitute_logical_parameters(
    plan: LogicalPlan,
    parameters: &[Parameter],
    arguments: Vec<Expr>,
) -> Result<LogicalPlan> {
    let mut bindings = HashMap::new();
    for (parameter, argument) in parameters.iter().zip(arguments) {
        let argument = if let Some(data_type) = &parameter.data_type {
            Expr::Cast(Cast::new(Box::new(argument), data_type.clone()))
        } else {
            argument
        };
        bindings.insert(format!("${}", bindings.len() + 1), argument);
    }
    plan.transform_up_with_subqueries(|plan| {
        plan.map_expressions(|expression| {
            expression.transform_up(|expression| match expression {
                Expr::Placeholder(placeholder) => {
                    bindings.get(&placeholder.id).cloned().map_or_else(
                        || Ok(Transformed::no(Expr::Placeholder(placeholder))),
                        |expression| Ok(Transformed::yes(expression)),
                    )
                },
                expression => Ok(Transformed::no(expression)),
            })
        })
    })
    .map(|transformed| transformed.data)
}

fn macro_error(macro_name: &str, dialect: &str, error: impl std::fmt::Display) -> DataFusionError {
    DataFusionError::Plan(format!(
        "DuckLake macro '{macro_name}' with dialect '{dialect}' could not be planned: {error}"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::common::Column;

    #[test]
    fn selects_exact_arity_before_defaulted_implementation() {
        let implementations = vec![
            ScalarImplementation {
                impl_id: 0,
                dialect: "duckdb".to_string(),
                parameters: vec![Parameter {
                    name: "value".to_string(),
                    data_type: None,
                    default_sql: Some("1".to_string()),
                }],
                body: Ok(datafusion::logical_expr::lit(1)),
                defaults: vec![Some(Ok(datafusion::logical_expr::lit(1)))],
            },
            ScalarImplementation {
                impl_id: 1,
                dialect: "duckdb".to_string(),
                parameters: Vec::new(),
                body: Ok(datafusion::logical_expr::lit(2)),
                defaults: Vec::new(),
            },
        ];

        assert_eq!(
            select_scalar_implementation(&implementations, 0)
                .unwrap()
                .impl_id,
            1
        );
    }

    #[test]
    fn expansion_casts_typed_arguments() {
        let implementation = ScalarImplementation {
            impl_id: 0,
            dialect: "duckdb".to_string(),
            parameters: vec![Parameter {
                name: "value".to_string(),
                data_type: Some(DataType::Int32),
                default_sql: None,
            }],
            body: Ok(Expr::Column(Column::from_name("value"))),
            defaults: vec![None],
        };

        assert_eq!(
            expand_scalar(vec![datafusion::logical_expr::lit(7_i64)], &implementation).unwrap(),
            Expr::Cast(Cast::new(
                Box::new(datafusion::logical_expr::lit(7_i64)),
                DataType::Int32,
            ))
        );
    }
}
