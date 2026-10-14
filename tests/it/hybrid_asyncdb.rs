#![cfg(feature = "metadata-duckdb")]
//! Hybrid AsyncDB adapter for running official DuckLake sqllogictests.
//!
//! One in-memory DuckDB connection with the official `ducklake` extension and one
//! DataFusion session share every catalog a test attaches. Each statement runs on
//! exactly one of them, chosen up front from the statement itself:
//!
//! - **DataFusion** (this crate's read path) runs every query whose relations all
//!   resolve to an attached DuckLake catalog. These are the reads under test.
//! - **DuckDB** runs everything else: DDL and DML, `ATTACH`/`DETACH`/`USE`,
//!   `SET`/`PRAGMA`/`CHECKPOINT`/`CALL`, statements DataFusion's parser rejects, and
//!   queries over relations DataFusion does not have — the metadata catalog
//!   (`__ducklake_metadata_<alias>` or the `METADATA_CATALOG` name), `memory` and
//!   other non-DuckLake databases, and table functions this crate does not register
//!   (`glob`, `duckdb_tables`, `ducklake_flush_inlined_data`, ...). Those queries
//!   inspect shared catalog state, so DuckDB answers them for both engines.
//!
//! Transactions: while DuckDB has an open transaction, every statement runs on
//! DuckDB, because its uncommitted state is invisible to any other reader. Once the
//! transaction ends, DataFusion sees the committed result.
//!
//! After every DuckDB statement outside a transaction the adapter re-reads DuckDB's
//! attached databases and its current catalog and schema (`USE`), and rebuilds the
//! DataFusion session: one `DuckLakeCatalog` per attached DuckLake under the same
//! alias, a row-lineage twin `<alias>__rowid` used for queries that reference
//! `rowid`, and this crate's table functions registered per alias as
//! `<alias>__ducklake_<fn>`.
//!
//! Queries are rewritten on the parsed AST before DataFusion plans them, following
//! DuckDB's name resolution: a two-part name `a.t` is catalog `a` when `a` is an
//! attached database, otherwise schema `a` of the current catalog; unqualified names
//! use the current catalog and schema; names match case-insensitively. Official
//! table-function call forms (`ducklake_table_changes('cat', 'schema', 't', a, b)`,
//! `cat.table_changes('t', a, b)`, `ducklake_snapshots('cat')`) and time travel
//! (`t AT (VERSION => n)`) map onto this crate's table functions.
//!
//! DuckDB-only virtual columns (`filename`, `file_row_number`, `file_index`,
//! `snapshot_id`) are not exposed by this crate. A query that names them only as
//! plain projection items runs on both engines: DataFusion with each replaced by
//! `NULL`, DuckDB as written. Every other column (`rowid` among them) must agree
//! between the two, and DuckDB's rows, virtual columns included, are checked
//! against the test. A query that uses them elsewhere fails to plan in
//! DataFusion; when the field DataFusion reports missing is one of them, DuckDB
//! answers it, counted under that reason. A relation the DataFusion catalog does
//! not have is answered by DuckDB too — the statement is then expected to fail
//! in DuckDB as well, and the adapter reports an error if DuckDB finds the
//! relation. Any other planning error fails the query.
//!
//! [`HybridDuckLakeDB::stats`] counts which engine ran each query, so a test file
//! that never reaches the DataFusion read path can be told apart from one that
//! passes.

use datafusion::arrow::array::Array;
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::ScalarValue;
use datafusion::prelude::*;
use datafusion::sql::parser::Statement as DFStatement;
use datafusion::sql::sqlparser::ast::{
    ExcludeSelectItem, Expr as SqlExpr, FunctionArg, FunctionArgExpr, FunctionArguments, Ident,
    ObjectName, ObjectNamePart, Query, SelectItem, SetExpr, Statement as SqlStatement, TableAlias,
    TableFactor, Value as SqlValue, VisitMut, VisitorMut, visit_expressions,
};
use datafusion::sql::sqlparser::dialect::GenericDialect;
use datafusion::sql::sqlparser::parser::Parser;
use datafusion::sql::sqlparser::tokenizer::{Token, Tokenizer};
use datafusion_ducklake::catalog::DuckLakeCatalog;
use datafusion_ducklake::{DuckdbMetadataProvider, MetadataProvider, register_ducklake_functions};
use duckdb::Connection;
use sqllogictest::{AsyncDB, DBOutput, DefaultColumnType};
use std::collections::{BTreeMap, HashSet};
use std::ops::ControlFlow;
use std::sync::{Arc, Mutex};

/// Custom error type for hybrid adapter
#[derive(Debug)]
pub struct HybridError(String);

impl std::fmt::Display for HybridError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Hybrid error: {}", self.0)
    }
}

impl std::error::Error for HybridError {}

impl From<duckdb::Error> for HybridError {
    fn from(e: duckdb::Error) -> Self {
        HybridError(format!("DuckDB error: {}", e))
    }
}

impl From<datafusion::error::DataFusionError> for HybridError {
    fn from(e: datafusion::error::DataFusionError) -> Self {
        HybridError(format!("DataFusion error: {}", e))
    }
}

impl From<datafusion_ducklake::error::DuckLakeError> for HybridError {
    fn from(e: datafusion_ducklake::error::DuckLakeError) -> Self {
        HybridError(format!("DuckLake error: {}", e))
    }
}

/// Prefix of the comment the runner puts before a `statement maybe` statement,
/// followed by the error text that statement may fail with and `*/`.
pub const MAYBE_MARKER: &str = "/*hybrid-maybe:";

/// Prefix of the comment the runner puts before a labelled query with no
/// expected rows, followed by the label and `*/`. DuckDB's runner checks such a
/// query against the first query with the same label; the adapter returns
/// [`LABEL_MATCH`] when the rows agree, or the rows themselves when they differ.
pub const LABEL_MARKER: &str = "/*hybrid-label:";

/// The single value a labelled query returns when its rows match the label's.
pub const LABEL_MATCH: &str = "label-matches";

/// Prefix of a value of a numeric column in the rows the adapter returns: the
/// validator compares such values as numbers (`100` and `100.0`), and others
/// as text.
pub const NUMERIC_TAG: &str = "\u{1}N";

/// Prefix of a value of a boolean column: the validator accepts DuckDB's
/// spellings of the same boolean (`true`, `t`, `1`).
pub const BOOLEAN_TAG: &str = "\u{1}B";

/// The tag of a column of `data_type`: [`NUMERIC_TAG`], [`BOOLEAN_TAG`] or none.
fn column_tag(data_type: &datafusion::arrow::datatypes::DataType) -> &'static str {
    use datafusion::arrow::datatypes::DataType;
    match data_type {
        DataType::Boolean => BOOLEAN_TAG,
        data_type if data_type.is_numeric() => NUMERIC_TAG,
        _ => "",
    }
}

/// `value` without its column tag.
pub fn untagged(value: &str) -> &str {
    value
        .strip_prefix(NUMERIC_TAG)
        .or_else(|| value.strip_prefix(BOOLEAN_TAG))
        .unwrap_or(value)
}

/// DuckDB virtual columns this crate does not expose (it exposes `rowid` only).
const DUCKDB_ONLY_VIRTUAL_COLUMNS: &[&str] =
    &["filename", "file_row_number", "file_index", "snapshot_id"];

/// Suffix of the row-lineage twin registered for every DuckLake catalog.
const ROWID_CATALOG_SUFFIX: &str = "__rowid";

/// `ducklake_table_at` with the `rowid` column, for time travel that reads it.
const TIME_TRAVEL_ROWID_FUNCTION: &str = "ducklake_table_at_rowid";

/// `ducklake_table_at`, its table exposing `rowid`.
#[derive(Debug)]
struct RowLineageTableAt(Arc<dyn datafusion::catalog::TableFunctionImpl>);

impl datafusion::catalog::TableFunctionImpl for RowLineageTableAt {
    fn call_with_args(
        &self,
        args: datafusion::catalog::TableFunctionArgs,
    ) -> datafusion::error::Result<Arc<dyn datafusion::catalog::TableProvider>> {
        let table = self.0.call_with_args(args)?;
        let table = table
            .downcast_ref::<datafusion_ducklake::DuckLakeTable>()
            .ok_or_else(|| {
                datafusion::error::DataFusionError::Internal(
                    "ducklake_table_at did not return a DuckLake table".to_string(),
                )
            })?
            .clone()
            .with_row_lineage(true);
        Ok(Arc::new(table))
    }
}

/// Table function the time-travel token rewrite emits for `t AT (VERSION => n)`;
/// the AST rewrite resolves its name arguments to a `ducklake_table_at` call.
const TIME_TRAVEL_FUNCTION: &str = "__hybrid_at";

/// Which engine ran the statements of one test file.
#[derive(Debug, Default, Clone)]
pub struct RouteStats {
    /// Queries DataFusion ran that read at least one DuckLake table.
    pub datafusion_lake_reads: usize,
    /// Queries DataFusion ran that read no DuckLake table (`SELECT 1`).
    pub datafusion_other_reads: usize,
    /// Queries DuckDB answered, by reason.
    pub duckdb_reads: BTreeMap<String, usize>,
    /// Non-query statements DuckDB ran.
    pub duckdb_statements: usize,
}

impl RouteStats {
    pub fn summary(&self) -> String {
        let duckdb_reads: Vec<String> = self
            .duckdb_reads
            .iter()
            .map(|(reason, count)| format!("{reason}={count}"))
            .collect();
        format!(
            "datafusion lake reads={}, other datafusion reads={}, duckdb statements={}, duckdb reads=[{}]",
            self.datafusion_lake_reads,
            self.datafusion_other_reads,
            self.duckdb_statements,
            duckdb_reads.join(", ")
        )
    }
}

/// A DuckLake catalog attached in DuckDB.
#[derive(Debug, Clone)]
struct Lake {
    /// Alias as DuckDB reports it.
    name: String,
    /// Metadata connection string as DuckDB reports it (`path` of `duckdb_databases()`).
    metadata: String,
    /// Why DataFusion cannot read this catalog; queries over it fail with this.
    unreadable: Option<String>,
}

impl Lake {
    /// Register this catalog, its row-lineage twin, and its table functions.
    fn register(&self, ctx: &SessionContext) -> Result<(), HybridError> {
        let metadata = self
            .metadata
            .strip_prefix("duckdb:")
            .unwrap_or(&self.metadata);
        if ["sqlite:", "postgres:", "mysql:"]
            .iter()
            .any(|prefix| metadata.starts_with(prefix))
        {
            return Err(HybridError(format!(
                "the hybrid runner reads DuckDB metadata only, not {metadata}"
            )));
        }
        let provider = DuckdbMetadataProvider::new(metadata)?;
        ctx.register_catalog(
            &self.name,
            Arc::new(DuckLakeCatalog::new(provider.clone())?),
        );
        ctx.register_catalog(
            format!("{}{ROWID_CATALOG_SUFFIX}", self.name),
            Arc::new(DuckLakeCatalog::new(provider.clone())?.with_row_lineage(true)),
        );
        let functions = SessionContext::new();
        let provider: Arc<dyn MetadataProvider> = Arc::new(provider);
        register_ducklake_functions(&functions, provider);
        for (name, function) in functions.state().table_functions() {
            ctx.register_udtf(&self.function_name(name), Arc::clone(function.function()));
            if name == "ducklake_table_at" {
                ctx.register_udtf(
                    &self.function_name(TIME_TRAVEL_ROWID_FUNCTION),
                    Arc::new(RowLineageTableAt(Arc::clone(function.function()))),
                );
            }
        }
        Ok(())
    }

    fn function_name(&self, function: &str) -> String {
        format!("{}__{function}", self.name.to_lowercase())
    }
}

/// Where a statement runs.
enum Route {
    DuckDb {
        /// Why DuckDB answers a query; `None` for a non-query statement.
        read_reason: Option<String>,
    },
    DataFusion {
        statement: Box<SqlStatement>,
        reads_lake: bool,
        /// `statement` with the DuckDB-only virtual columns it projects replaced
        /// by `NULL`, and those columns, for when DataFusion cannot plan
        /// `statement` because they are missing: DuckDB answers the query, and
        /// DataFusion's rows must match its rows outside these columns. A query
        /// whose columns of those names are real (a CTE or view over a table
        /// function that outputs `snapshot_id`) plans as written.
        nulled: Option<(Box<SqlStatement>, Vec<&'static str>)>,
    },
    /// The query reads a DuckLake catalog DataFusion cannot read.
    Error(String),
}

struct Inner {
    conn: Connection,
    ctx: SessionContext,
    /// Every attached DuckDB database, lower-cased, including the internal
    /// metadata catalogs DuckLake attaches.
    databases: HashSet<String>,
    lakes: Vec<Lake>,
    current_catalog: String,
    current_schema: String,
    in_transaction: bool,
    stats: RouteStats,
    /// Rows of the first query run under each result label.
    labels: BTreeMap<String, Vec<Vec<String>>>,
}

/// Hybrid database adapter: DuckDB for writes and DuckDB-only reads, DataFusion
/// for reads of DuckLake tables. Clones share one state.
#[derive(Clone)]
pub struct HybridDuckLakeDB {
    inner: Arc<Mutex<Inner>>,
}

impl HybridDuckLakeDB {
    pub fn new() -> Result<Self, HybridError> {
        let conn = Connection::open_in_memory()?;
        crate::common::ensure_ducklake_installed();
        conn.execute_batch("LOAD ducklake; LOAD parquet;")?;
        // TIMESTAMPTZ values render in UTC on both engines.
        conn.execute_batch("SET TimeZone = 'UTC'").ok();
        let mut inner = Inner {
            conn,
            ctx: SessionContext::new(),
            databases: HashSet::new(),
            lakes: Vec::new(),
            current_catalog: "memory".to_string(),
            current_schema: "main".to_string(),
            in_transaction: false,
            stats: RouteStats::default(),
            labels: BTreeMap::new(),
        };
        inner.sync()?;
        Ok(Self {
            inner: Arc::new(Mutex::new(inner)),
        })
    }

    /// Install and load DuckDB extension `name` (a `require` directive).
    pub fn load_extension(&self, name: &str) -> Result<(), HybridError> {
        let installed =
            std::panic::catch_unwind(|| crate::common::ensure_extension_installed(name));
        if installed.is_err() {
            return Err(HybridError(format!("extension {name} is not installable")));
        }
        let inner = self.inner.lock().unwrap();
        inner.conn.execute_batch(&format!("LOAD {name}"))?;
        Ok(())
    }

    pub fn stats(&self) -> RouteStats {
        self.inner.lock().unwrap().stats.clone()
    }

    async fn run_statement(&self, sql: &str) -> Result<DBOutput<DefaultColumnType>, HybridError> {
        let route = self.inner.lock().unwrap().route(sql);
        match route {
            Route::Error(error) => Err(HybridError(error)),
            Route::DuckDb {
                read_reason,
            } => {
                let mut inner = self.inner.lock().unwrap();
                match read_reason {
                    Some(reason) => *inner.stats.duckdb_reads.entry(reason).or_default() += 1,
                    None => inner.stats.duckdb_statements += 1,
                }
                inner.execute_duckdb(sql)
            },
            Route::DataFusion {
                statement,
                reads_lake,
                nulled,
            } => {
                let ctx = self.inner.lock().unwrap().ctx.clone();
                let state = ctx.state();
                let mut compared_virtual_columns = Vec::new();
                let mut planned = state
                    .statement_to_plan(DFStatement::Statement(statement))
                    .await;
                if let (Err(error), Some((nulled, columns))) = (&planned, nulled)
                    && missing_virtual_column(&error.to_string()).is_some()
                {
                    planned = state
                        .statement_to_plan(DFStatement::Statement(nulled))
                        .await;
                    compared_virtual_columns = columns;
                }
                let plan = match planned {
                    Ok(plan) => plan,
                    Err(error) => {
                        let message = error.to_string();
                        let reason = if let Some(column) = missing_virtual_column(&message) {
                            format!("virtual column {column}")
                        } else if let Some(function) = unknown_function(&message) {
                            format!("DuckDB function {function}")
                        } else if is_missing_relation(&message) {
                            "relation missing in DataFusion".to_string()
                        } else {
                            return Err(error.into());
                        };
                        let missing_relation = reason.starts_with("relation");
                        let mut inner = self.inner.lock().unwrap();
                        *inner.stats.duckdb_reads.entry(reason).or_default() += 1;
                        let output = inner.execute_duckdb(sql);
                        if missing_relation && output.is_ok() {
                            return Err(HybridError(format!(
                                "DuckDB read a relation DataFusion's catalog does not have: {message}"
                            )));
                        }
                        return output;
                    },
                };
                let column_count = plan.schema().fields().len();
                let batches = ctx.execute_logical_plan(plan).await?.collect().await?;
                {
                    let mut inner = self.inner.lock().unwrap();
                    if reads_lake {
                        inner.stats.datafusion_lake_reads += 1;
                    } else {
                        inner.stats.datafusion_other_reads += 1;
                    }
                }
                if !compared_virtual_columns.is_empty() {
                    return self.compare_with_duckdb(sql, &batches, &compared_virtual_columns);
                }
                Ok(DBOutput::Rows {
                    types: vec![DefaultColumnType::Any; column_count],
                    rows: batches_to_rows(&batches)?,
                })
            },
        }
    }
}

impl HybridDuckLakeDB {
    /// Answer `sql` with DuckDB's rows once DataFusion's `batches` for it agree
    /// with them outside the `virtual_columns` DataFusion projected as `NULL`.
    fn compare_with_duckdb(
        &self,
        sql: &str,
        batches: &[RecordBatch],
        virtual_columns: &[&'static str],
    ) -> Result<DBOutput<DefaultColumnType>, HybridError> {
        let skipped: Vec<usize> = batches
            .first()
            .map(|batch| {
                batch
                    .schema()
                    .fields()
                    .iter()
                    .enumerate()
                    .filter(|(_, field)| {
                        field.data_type() == &datafusion::arrow::datatypes::DataType::Null
                            && virtual_columns
                                .iter()
                                .any(|column| field.name().eq_ignore_ascii_case(column))
                    })
                    .map(|(index, _)| index)
                    .collect()
            })
            .unwrap_or_default();
        let ours = batches_to_rows(batches)?;
        let output = {
            let mut inner = self.inner.lock().unwrap();
            *inner
                .stats
                .duckdb_reads
                .entry(format!(
                    "virtual columns {} (compared)",
                    virtual_columns.join(", ")
                ))
                .or_default() += 1;
            inner.execute_duckdb(sql)?
        };
        let DBOutput::Rows {
            rows: theirs,
            ..
        } = &output
        else {
            return Err(HybridError(format!("DuckDB returned no rows for `{sql}`")));
        };
        let without = |rows: &[Vec<String>]| {
            let mut rows: Vec<Vec<String>> = rows
                .iter()
                .map(|row| {
                    row.iter()
                        .enumerate()
                        .filter(|(index, _)| !skipped.contains(index))
                        .map(|(_, value)| untagged(value).to_string())
                        .collect()
                })
                .collect();
            rows.sort();
            rows
        };
        let (ours, theirs) = (without(&ours), without(theirs));
        if ours != theirs {
            return Err(HybridError(format!(
                "DataFusion and DuckDB disagree outside the DuckDB-only virtual columns \
                 {virtual_columns:?}: DataFusion {ours:?}, DuckDB {theirs:?}"
            )));
        }
        Ok(output)
    }
}

/// Replace every top-level projection item of `statement` that is exactly a
/// DuckDB-only virtual column with `NULL AS <column>`, and return those columns.
/// Nothing is replaced when the statement is not a plain `SELECT` over table
/// scans: a table function such as `ducklake_table_changes` or
/// `ducklake_snapshots` outputs `snapshot_id` as a real column. The result is
/// planned only when the statement as written fails on a missing virtual
/// column, which also covers a CTE or view over such a function.
fn null_projected_virtual_columns(statement: &mut SqlStatement) -> Vec<&'static str> {
    let SqlStatement::Query(query) = statement else {
        return Vec::new();
    };
    let SetExpr::Select(select) = query.body.as_mut() else {
        return Vec::new();
    };
    let base_table = |factor: &TableFactor| {
        matches!(
            factor,
            TableFactor::Table {
                args: None,
                ..
            }
        )
    };
    let scans_tables_only = !select.from.is_empty()
        && select.from.iter().all(|from| {
            base_table(&from.relation) && from.joins.iter().all(|join| base_table(&join.relation))
        });
    if !scans_tables_only {
        return Vec::new();
    }
    let mut replaced = Vec::new();
    for item in &mut select.projection {
        let (ident, alias) = match item {
            SelectItem::UnnamedExpr(SqlExpr::Identifier(ident)) => (ident.clone(), None),
            SelectItem::ExprWithAlias {
                expr: SqlExpr::Identifier(ident),
                alias,
            } => (ident.clone(), Some(alias.clone())),
            _ => continue,
        };
        let Some(column) = DUCKDB_ONLY_VIRTUAL_COLUMNS
            .iter()
            .find(|column| ident.value.eq_ignore_ascii_case(column))
        else {
            continue;
        };
        *item = SelectItem::ExprWithAlias {
            expr: SqlExpr::Value(SqlValue::Null.into()),
            alias: alias.unwrap_or_else(|| Ident::new(*column)),
        };
        replaced.push(*column);
    }
    replaced
}

/// The DuckDB-only virtual column a DataFusion "No field named" planning error
/// names, if the missing field is one.
fn missing_virtual_column(message: &str) -> Option<&'static str> {
    let (_, rest) = message.split_once("No field named ")?;
    let field = rest
        .split(|c: char| c.is_whitespace())
        .next()?
        .trim_end_matches('.')
        .rsplit('.')
        .next()?
        .trim_matches('"');
    DUCKDB_ONLY_VIRTUAL_COLUMNS
        .iter()
        .find(|column| field.eq_ignore_ascii_case(column))
        .copied()
}

impl Inner {
    /// Decide where `sql` runs, rewriting it for DataFusion when it runs there.
    fn route(&self, sql: &str) -> Route {
        let is_query = starts_query(sql);
        let duckdb = |reason: &str| Route::DuckDb {
            read_reason: is_query.then(|| reason.to_string()),
        };
        if self.in_transaction {
            return duckdb("open transaction");
        }
        let rewritten = rewrite_time_travel(sql);
        let Ok(mut statements) = Parser::parse_sql(&GenericDialect {}, &rewritten) else {
            return duckdb("DuckDB-only syntax");
        };
        if statements.len() != 1 || !matches!(statements[0], SqlStatement::Query(_)) {
            return duckdb("not a single query");
        }
        let mut statement = statements.pop().unwrap();
        let identifiers = referenced_identifiers(&statement);
        let mut resolver = Resolver {
            inner: self,
            ctes: HashSet::new(),
            row_lineage: identifiers.contains("rowid"),
            duckdb_reason: None,
            error: None,
            reads_lake: false,
        };
        let _ = statement.visit(&mut resolver);
        if let Some(error) = resolver.error {
            return Route::Error(error);
        }
        if let Some(reason) = resolver.duckdb_reason {
            return duckdb(&reason);
        }
        let mut nulled_statement = statement.clone();
        let columns = null_projected_virtual_columns(&mut nulled_statement);
        Route::DataFusion {
            reads_lake: resolver.reads_lake,
            statement: Box::new(statement),
            nulled: (!columns.is_empty()).then(|| (Box::new(nulled_statement), columns)),
        }
    }

    /// Whether DuckDB resolves `name` as a database. DuckLake's default metadata
    /// catalog `__ducklake_metadata_<alias>` is not listed by `duckdb_databases()`.
    fn is_database(&self, name: &str) -> bool {
        let name = name.to_lowercase();
        self.databases.contains(&name) || name.starts_with("__ducklake_metadata_")
    }

    fn lake(&self, name: &str) -> Option<&Lake> {
        self.lakes
            .iter()
            .find(|lake| lake.name.eq_ignore_ascii_case(name))
    }

    /// Run `sql` on DuckDB, then re-sync DataFusion unless a transaction is open.
    fn execute_duckdb(&mut self, sql: &str) -> Result<DBOutput<DefaultColumnType>, HybridError> {
        // A prepared statement binds in one autocommit transaction and runs in
        // the next, and DuckLake table functions bound that way fail with
        // "Scanning a DuckLake table after the transaction has ended". Outside a
        // transaction, a CALL runs inside an explicit one so it binds and runs
        // in the same transaction, as an unprepared statement does.
        let is_call = sql.trim_start().to_uppercase().starts_with("CALL ");
        let output = if is_call && !self.in_transaction {
            self.conn.execute_batch("BEGIN")?;
            let output = run_duckdb(&self.conn, sql);
            let end = if output.is_ok() {
                "COMMIT"
            } else {
                "ROLLBACK"
            };
            match (output, self.conn.execute_batch(end)) {
                (Ok(_), Err(error)) => Err(error.into()),
                (output, _) => output,
            }
        } else {
            run_duckdb(&self.conn, sql)
        };
        // The C API does not expose DuckDB's autocommit state, so track the
        // transaction statements. A failed COMMIT also ends the transaction.
        match transaction_effect(sql) {
            Some(TransactionEffect::Begin) if output.is_ok() => self.in_transaction = true,
            Some(TransactionEffect::End) => self.in_transaction = false,
            _ => {},
        }
        if !self.in_transaction {
            self.sync()?;
        }
        output
    }

    /// Re-read DuckDB's attached databases and current catalog and schema, and
    /// rebuild the DataFusion session over them.
    fn sync(&mut self) -> Result<(), HybridError> {
        let mut databases = HashSet::new();
        let mut lakes = Vec::new();
        {
            let mut stmt = self.conn.prepare(
                "SELECT database_name, type, coalesce(path, '') FROM duckdb_databases() \
                 ORDER BY database_name",
            )?;
            let mut rows = stmt.query([])?;
            while let Some(row) = rows.next()? {
                let name: String = row.get(0)?;
                let kind: String = row.get(1)?;
                let path: String = row.get(2)?;
                databases.insert(name.to_lowercase());
                if kind == "ducklake" {
                    lakes.push(Lake {
                        name,
                        metadata: path,
                        unreadable: None,
                    });
                }
            }
        }
        let (catalog, schema): (String, String) =
            self.conn
                .query_row("SELECT current_database(), current_schema()", [], |row| {
                    Ok((row.get(0)?, row.get(1)?))
                })?;
        self.databases = databases;
        self.lakes = lakes;
        self.current_catalog = catalog;
        self.current_schema = schema;

        let mut config = SessionConfig::new();
        if let Some(lake) = self.lake(&self.current_catalog) {
            config = config.with_default_catalog_and_schema(&lake.name, &self.current_schema);
        }
        let ctx = SessionContext::new_with_config(config);
        for lake in &mut self.lakes {
            if let Err(error) = lake.register(&ctx) {
                lake.unreadable = Some(error.to_string());
            }
        }
        self.ctx = ctx;
        Ok(())
    }
}

enum TransactionEffect {
    Begin,
    End,
}

/// Whether `sql` opens or ends a DuckDB transaction.
fn transaction_effect(sql: &str) -> Option<TransactionEffect> {
    let upper = sql.trim().trim_end_matches(';').to_uppercase();
    let first = upper.split_whitespace().next().unwrap_or("");
    match first {
        "BEGIN" => Some(TransactionEffect::Begin),
        "START" if upper.split_whitespace().nth(1) == Some("TRANSACTION") => {
            Some(TransactionEffect::Begin)
        },
        "COMMIT" | "END" | "ROLLBACK" | "ABORT" => Some(TransactionEffect::End),
        _ => None,
    }
}

/// Whether `sql` reads like a query (as opposed to DDL, DML, or a setting).
fn starts_query(sql: &str) -> bool {
    let upper = sql.trim_start().trim_start_matches('(').to_uppercase();
    ["SELECT", "WITH", "FROM", "VALUES", "TABLE", "SUMMARIZE", "DESCRIBE", "SHOW"]
        .iter()
        .any(|keyword| {
            upper.starts_with(keyword)
                && !upper[keyword.len()..].starts_with(|c: char| c.is_alphanumeric() || c == '_')
        })
}

/// Execute `sql` on DuckDB and render any result set it returns, DML row counts included.
fn run_duckdb(conn: &Connection, sql: &str) -> Result<DBOutput<DefaultColumnType>, HybridError> {
    let mut stmt = match conn.prepare(sql) {
        Ok(stmt) => stmt,
        Err(error) if error.to_string().contains("multiple statements") => {
            conn.execute_batch(sql)?;
            return Ok(DBOutput::StatementComplete(0));
        },
        Err(error) => return Err(error.into()),
    };
    let mut rows = stmt.query([])?;
    let mut output = Vec::new();
    let mut column_count = 0;
    let mut zoned = Vec::new();
    let mut tags = Vec::new();
    while let Some(row) = rows.next()? {
        if output.is_empty() {
            let stmt = row.as_ref();
            column_count = stmt.column_count();
            zoned = (0..column_count)
                .map(|i| {
                    matches!(
                        stmt.column_type(i),
                        duckdb::arrow::datatypes::DataType::Timestamp(_, Some(_))
                    )
                })
                .collect();
            tags = (0..column_count)
                .map(|i| {
                    use duckdb::arrow::datatypes::DataType;
                    match stmt.column_type(i) {
                        DataType::Boolean => BOOLEAN_TAG,
                        data_type if data_type.is_numeric() => NUMERIC_TAG,
                        _ => "",
                    }
                })
                .collect();
        }
        let mut values = Vec::with_capacity(column_count);
        for (i, zoned) in zoned.iter().enumerate() {
            let value: duckdb::types::Value = row.get(i)?;
            values.push(format!(
                "{}{}",
                tags[i],
                format_duckdb_value(&value, *zoned, true)
            ));
        }
        output.push(values);
    }
    let column_count = match rows.as_ref() {
        Some(stmt) => stmt.column_count(),
        None => column_count,
    };
    if column_count == 0 {
        return Ok(DBOutput::StatementComplete(0));
    }
    Ok(DBOutput::Rows {
        types: vec![DefaultColumnType::Any; column_count],
        rows: output,
    })
}

/// Rewrite DuckDB time travel `name AT (VERSION => e)` / `AT (TIMESTAMP => e)` into
/// `__hybrid_at('part', ..., e)`, at the token level so string literals and comments
/// are untouched. The AST rewrite resolves the name parts.
fn rewrite_time_travel(sql: &str) -> String {
    let Ok(tokens) = Tokenizer::new(&GenericDialect {}, sql).tokenize() else {
        return sql.to_string();
    };
    let significant: Vec<usize> = (0..tokens.len())
        .filter(|&i| !matches!(tokens[i], Token::Whitespace(_)))
        .collect();
    let keyword = |i: usize, keyword: &str| matches!(&tokens[i], Token::Word(w) if w.quote_style.is_none() && w.value.eq_ignore_ascii_case(keyword));
    // Token ranges `[start, end]` and the call that replaces each.
    let mut replacements: Vec<(usize, usize, String)> = Vec::new();
    for (pos, &at) in significant.iter().enumerate() {
        if !keyword(at, "AT") {
            continue;
        }
        let token_after = |offset: usize| significant.get(pos + offset).copied();
        let (Some(lparen), Some(kind), Some(arrow)) =
            (token_after(1), token_after(2), token_after(3))
        else {
            continue;
        };
        if tokens[lparen] != Token::LParen
            || !(keyword(kind, "VERSION") || keyword(kind, "TIMESTAMP"))
            || tokens[arrow] != Token::RArrow
        {
            continue;
        }
        // The name directly before `AT`: Word (. Word)*.
        let mut parts = Vec::new();
        let mut start = None;
        let mut expect_word = true;
        for &t in significant[..pos].iter().rev() {
            match (&tokens[t], expect_word) {
                (Token::Word(w), true) => {
                    parts.push(w.value.clone());
                    start = Some(t);
                    expect_word = false;
                },
                (Token::Period, false) => expect_word = true,
                _ => break,
            }
        }
        let Some(start) = start else {
            continue;
        };
        let mut depth = 0usize;
        let mut end = None;
        for (j, token) in tokens.iter().enumerate().skip(arrow + 1) {
            match token {
                Token::LParen => depth += 1,
                Token::RParen if depth == 0 => {
                    end = Some(j);
                    break;
                },
                Token::RParen => depth -= 1,
                _ => {},
            }
        }
        let Some(end) = end else {
            continue;
        };
        parts.reverse();
        let names: Vec<String> = parts
            .iter()
            .map(|part| format!("'{}'", part.replace('\'', "''")))
            .collect();
        let expr: String = tokens[arrow + 1..end]
            .iter()
            .map(Token::to_string)
            .collect();
        replacements.push((
            start,
            end,
            format!(
                "{TIME_TRAVEL_FUNCTION}({}, {})",
                names.join(", "),
                expr.trim()
            ),
        ));
    }
    if replacements.is_empty() {
        return sql.to_string();
    }
    let mut out = String::new();
    let mut next = 0;
    for (start, end, call) in replacements {
        out.extend(tokens[next..start].iter().map(Token::to_string));
        out.push_str(&call);
        next = end + 1;
    }
    out.extend(tokens[next..].iter().map(Token::to_string));
    out
}

/// The column names `statement` references, lower-cased (the last part of each
/// compound identifier).
fn referenced_identifiers(statement: &SqlStatement) -> HashSet<String> {
    let mut names = HashSet::new();
    let _ = visit_expressions(statement, |expr| {
        let name = match expr {
            SqlExpr::Identifier(ident) => Some(ident),
            SqlExpr::CompoundIdentifier(parts) => parts.last(),
            _ => None,
        };
        if let Some(ident) = name {
            names.insert(ident.value.to_lowercase());
        }
        ControlFlow::<()>::Continue(())
    });
    names
}

/// The function a DataFusion "Invalid function" planning error names, if any.
fn unknown_function(message: &str) -> Option<String> {
    let (_, rest) = message.split_once("Invalid function '")?;
    rest.split_once('\'').map(|(name, _)| name.to_string())
}

fn is_missing_relation(message: &str) -> bool {
    message.contains("Error during planning: table '") && message.contains("' not found")
}

fn quoted(value: &str) -> Ident {
    Ident::with_quote('"', value)
}

fn string_literal(expr: &FunctionArg) -> Option<String> {
    match expr {
        FunctionArg::Unnamed(FunctionArgExpr::Expr(SqlExpr::Value(value))) => match &value.value {
            SqlValue::SingleQuotedString(s) => Some(s.clone()),
            _ => None,
        },
        _ => None,
    }
}

fn string_arg(value: &str) -> FunctionArg {
    FunctionArg::Unnamed(FunctionArgExpr::Expr(SqlExpr::Value(
        SqlValue::SingleQuotedString(value.to_string()).into(),
    )))
}

/// Resolves and rewrites the relations of one query for DataFusion, or records why
/// DuckDB has to answer it.
struct Resolver<'a> {
    inner: &'a Inner,
    ctes: HashSet<String>,
    row_lineage: bool,
    duckdb_reason: Option<String>,
    /// Set when the query reads a DuckLake catalog DataFusion cannot read.
    error: Option<String>,
    reads_lake: bool,
}

/// A relation resolved to a DuckLake table: catalog, schema, and table names as
/// the DataFusion catalog spells them.
struct LakeRelation<'a> {
    lake: &'a Lake,
    schema: String,
    table: String,
}

/// Why a relation does not resolve to a readable DuckLake table.
enum Unresolved {
    /// DuckDB answers the query; the reason is tallied.
    DuckDb(String),
    /// The query fails: it reads a DuckLake catalog DataFusion cannot read.
    Error(String),
}

impl<'a> Resolver<'a> {
    fn fail(&mut self, reason: String) {
        self.duckdb_reason.get_or_insert(reason);
    }

    fn unresolved(&mut self, unresolved: Unresolved) {
        match unresolved {
            Unresolved::DuckDb(reason) => self.fail(reason),
            Unresolved::Error(error) => {
                self.error.get_or_insert(error);
            },
        }
    }

    /// Resolve name parts the way DuckDB does. `Err` is the reason DuckDB answers.
    fn resolve(&self, parts: &[String]) -> Result<LakeRelation<'a>, Unresolved> {
        let inner = self.inner;
        let (catalog, schema, table) = match parts {
            [table] => (
                inner.current_catalog.clone(),
                inner.current_schema.clone(),
                table.clone(),
            ),
            [first, table] if inner.is_database(first) => {
                (first.clone(), "main".to_string(), table.clone())
            },
            [schema, table] => (inner.current_catalog.clone(), schema.clone(), table.clone()),
            [catalog, schema, table] => (catalog.clone(), schema.clone(), table.clone()),
            _ => return Err(Unresolved::DuckDb("unsupported relation name".to_string())),
        };
        if ["information_schema", "pg_catalog"].contains(&schema.to_lowercase().as_str()) {
            return Err(Unresolved::DuckDb("DuckDB system schema".to_string()));
        }
        let Some(lake) = inner.lake(&catalog) else {
            return Err(Unresolved::DuckDb("non-DuckLake relation".to_string()));
        };
        if let Some(error) = &lake.unreadable {
            return Err(Unresolved::Error(format!(
                "DuckLake catalog {} is unreadable: {error}",
                lake.name
            )));
        }
        // DuckDB matches names case-insensitively; DataFusion matches exactly.
        let Some(catalog_provider) = inner.ctx.catalog(&lake.name) else {
            return Err(Unresolved::Error(format!(
                "DuckLake catalog {} is not registered",
                lake.name
            )));
        };
        let schema = match_name(&catalog_provider.schema_names(), &schema);
        let table = match catalog_provider.schema(&schema) {
            Some(schema_provider) => match_name(&schema_provider.table_names(), &table),
            None => table,
        };
        Ok(LakeRelation {
            lake,
            schema,
            table,
        })
    }

    /// Default schema of `lake` for an official call form that names only a table.
    fn default_schema(&self, lake: &Lake) -> String {
        if lake.name.eq_ignore_ascii_case(&self.inner.current_catalog) {
            self.inner.current_schema.clone()
        } else {
            "main".to_string()
        }
    }

    /// Rewrite a table-function call onto this crate's per-catalog functions.
    fn rewrite_function(
        &mut self,
        name: &mut ObjectName,
        args: &mut Vec<FunctionArg>,
        alias: &mut Option<TableAlias>,
    ) {
        let parts = name_parts(name);
        let lower: Vec<String> = parts.iter().map(|p| p.to_lowercase()).collect();
        let inner = self.inner;
        let cdc = ["table_changes", "table_insertions", "table_deletions"];
        let rewritten: Option<(String, Vec<FunctionArg>)> = match lower.as_slice() {
            [f] if f == TIME_TRAVEL_FUNCTION => {
                let names: Option<Vec<String>> =
                    args[..args.len() - 1].iter().map(string_literal).collect();
                let Some(names) = names else {
                    return self.fail("time travel".to_string());
                };
                match self.resolve(&names) {
                    Ok(relation) => {
                        self.reads_lake = true;
                        if alias.is_none() {
                            *alias = Some(TableAlias {
                                explicit: true,
                                name: quoted(&relation.table),
                                columns: vec![],
                                at: None,
                            });
                        }
                        let version = args.last().unwrap().clone();
                        Some((
                            relation.lake.function_name(if self.row_lineage {
                                TIME_TRAVEL_ROWID_FUNCTION
                            } else {
                                "ducklake_table_at"
                            }),
                            vec![
                                string_arg(&relation.schema),
                                string_arg(&relation.table),
                                version,
                            ],
                        ))
                    },
                    Err(unresolved) => return self.unresolved(unresolved),
                }
            },
            [catalog, f] if inner.lake(catalog).is_some() => {
                let lake = inner.lake(catalog).unwrap();
                if cdc.contains(&f.as_str()) && !args.is_empty() {
                    let Some(table) = string_literal(&args[0]) else {
                        return self.fail("table function arguments".to_string());
                    };
                    let table = if table.contains('.') {
                        table
                    } else {
                        format!("{}.{table}", self.default_schema(lake))
                    };
                    let mut new_args = vec![string_arg(&table)];
                    new_args.extend(args[1..].iter().cloned());
                    Some((lake.function_name(&format!("ducklake_{f}")), new_args))
                } else if (f == "snapshots" || f == "table_info") && args.is_empty() {
                    Some((lake.function_name(&format!("ducklake_{f}")), vec![]))
                } else {
                    return self.fail(format!("table function {catalog}.{f}"));
                }
            },
            [f] if cdc.iter().any(|c| f == &format!("ducklake_{c}")) && args.len() == 5 => {
                let strings: Option<Vec<String>> = args[..3].iter().map(string_literal).collect();
                match strings.as_deref() {
                    Some([catalog, schema, table]) if inner.lake(catalog).is_some() => {
                        let lake = inner.lake(catalog).unwrap();
                        let mut new_args = vec![string_arg(&format!("{schema}.{table}"))];
                        new_args.extend(args[3..].iter().cloned());
                        Some((lake.function_name(f), new_args))
                    },
                    _ => return self.fail(format!("table function {f}")),
                }
            },
            [f] if (f == "ducklake_snapshots" || f == "ducklake_table_info") && args.len() == 1 => {
                match string_literal(&args[0])
                    .as_deref()
                    .and_then(|c| inner.lake(c))
                {
                    Some(lake) => Some((lake.function_name(f), vec![])),
                    None => return self.fail(format!("table function {f}")),
                }
            },
            [f] if inner.ctx.state().table_functions().contains_key(f) => None,
            _ => return self.fail(format!("table function {}", lower.join("."))),
        };
        if let Some((function, new_args)) = rewritten {
            self.reads_lake = true;
            *name = ObjectName(vec![ObjectNamePart::Identifier(Ident::new(function))]);
            *args = new_args;
        }
    }
}

/// The name in `names` equal to `wanted`, else the only one equal ignoring case,
/// else `wanted`.
fn match_name(names: &[String], wanted: &str) -> String {
    if names.iter().any(|name| name == wanted) {
        return wanted.to_string();
    }
    let matches: Vec<&String> = names
        .iter()
        .filter(|name| name.eq_ignore_ascii_case(wanted))
        .collect();
    match matches.as_slice() {
        [only] => (*only).clone(),
        _ => wanted.to_string(),
    }
}

fn name_parts(name: &ObjectName) -> Vec<String> {
    name.0
        .iter()
        .map(|part| match part {
            ObjectNamePart::Identifier(ident) => ident.value.clone(),
            ObjectNamePart::Function(function) => function.to_string(),
        })
        .collect()
}

impl VisitorMut for Resolver<'_> {
    type Break = ();

    fn pre_visit_expr(&mut self, expr: &mut SqlExpr) -> ControlFlow<()> {
        // DuckDB's date-part shorthands, spelled as DataFusion's `date_part`.
        if let SqlExpr::Function(function) = expr
            && let [ObjectNamePart::Identifier(name)] = function.name.0.as_slice()
            && let FunctionArguments::List(list) = &mut function.args
            && list.args.len() == 1
        {
            let part = name.value.to_lowercase();
            if ["year", "month", "day", "hour", "minute", "second"].contains(&part.as_str()) {
                list.args.insert(0, string_arg(&part));
                function.name =
                    ObjectName(vec![ObjectNamePart::Identifier(Ident::new("date_part"))]);
            }
        }
        ControlFlow::Continue(())
    }

    fn pre_visit_query(&mut self, query: &mut Query) -> ControlFlow<()> {
        if let Some(with) = &query.with {
            for cte in &with.cte_tables {
                self.ctes.insert(cte.alias.name.value.to_lowercase());
            }
        }
        ControlFlow::Continue(())
    }

    fn post_visit_query(&mut self, query: &mut Query) -> ControlFlow<()> {
        select_star_for_from_first(&mut query.body);
        if self.row_lineage {
            exclude_rowid_from_wildcards(&mut query.body);
        }
        ControlFlow::Continue(())
    }

    fn pre_visit_table_factor(&mut self, factor: &mut TableFactor) -> ControlFlow<()> {
        match factor {
            TableFactor::Table {
                name,
                args: Some(args),
                alias,
                ..
            } => {
                let mut list = std::mem::take(&mut args.args);
                self.rewrite_function(name, &mut list, alias);
                args.args = list;
            },
            TableFactor::Table {
                name,
                args: None,
                version,
                ..
            } => {
                // `FROM 'path/*.parquet'`: DuckDB reads the files directly.
                let file_scan = name.0.iter().any(|part| {
                    matches!(part, ObjectNamePart::Identifier(ident) if ident.quote_style == Some('\''))
                });
                if file_scan {
                    self.fail("file scan".to_string());
                    return ControlFlow::Continue(());
                }
                let parts = name_parts(name);
                if parts.len() == 1 && self.ctes.contains(&parts[0].to_lowercase()) {
                    return ControlFlow::Continue(());
                }
                if version.is_some() {
                    self.fail("table version".to_string());
                    return ControlFlow::Continue(());
                }
                match self.resolve(&parts) {
                    Ok(relation) => {
                        self.reads_lake = true;
                        let catalog = if self.row_lineage {
                            format!("{}{ROWID_CATALOG_SUFFIX}", relation.lake.name)
                        } else {
                            relation.lake.name.clone()
                        };
                        *name = ObjectName(vec![
                            ObjectNamePart::Identifier(quoted(&catalog)),
                            ObjectNamePart::Identifier(quoted(&relation.schema)),
                            ObjectNamePart::Identifier(quoted(&relation.table)),
                        ]);
                    },
                    Err(unresolved) => self.unresolved(unresolved),
                }
            },
            TableFactor::Function {
                name,
                ..
            } => {
                self.fail(format!("table function {name}"));
            },
            _ => {},
        }
        ControlFlow::Continue(())
    }
}

/// DuckDB's FROM-first shorthand (`FROM t`) parses as a SELECT without a
/// projection, which DataFusion plans with zero columns; DuckDB means `SELECT *`.
fn select_star_for_from_first(body: &mut SetExpr) {
    match body {
        SetExpr::Select(select) if select.projection.is_empty() => {
            select
                .projection
                .push(SelectItem::Wildcard(Default::default()));
        },
        SetExpr::SetOperation {
            left,
            right,
            ..
        } => {
            select_star_for_from_first(left);
            select_star_for_from_first(right);
        },
        _ => {},
    }
}

/// Add `EXCLUDE (rowid)` to the wildcards of every SELECT that reads a row-lineage
/// catalog, which exposes `rowid` as an ordinary column; DuckDB's `*` hides it.
fn exclude_rowid_from_wildcards(body: &mut SetExpr) {
    match body {
        SetExpr::Select(select) => {
            let reads_rowid_catalog = select.from.iter().any(|from| {
                std::iter::once(&from.relation)
                    .chain(from.joins.iter().map(|join| &join.relation))
                    .any(|factor| match factor {
                        TableFactor::Table {
                            name,
                            args: None,
                            ..
                        } => name_parts(name)
                            .first()
                            .is_some_and(|catalog| catalog.ends_with(ROWID_CATALOG_SUFFIX)),
                        TableFactor::Table {
                            name,
                            args: Some(_),
                            ..
                        } => name_parts(name).last().is_some_and(|function| {
                            function.ends_with(&format!("__{TIME_TRAVEL_ROWID_FUNCTION}"))
                        }),
                        _ => false,
                    })
            });
            if !reads_rowid_catalog {
                return;
            }
            let rowid = || {
                Some(ExcludeSelectItem::Single(ObjectName(vec![
                    ObjectNamePart::Identifier(Ident::new("rowid")),
                ])))
            };
            for item in &mut select.projection {
                match item {
                    SelectItem::Wildcard(options) | SelectItem::QualifiedWildcard(_, options)
                        if options.opt_exclude.is_none() =>
                    {
                        options.opt_exclude = rowid();
                    },
                    _ => {},
                }
            }
        },
        SetExpr::SetOperation {
            left,
            right,
            ..
        } => {
            exclude_rowid_from_wildcards(left);
            exclude_rowid_from_wildcards(right);
        },
        SetExpr::Query(query) => exclude_rowid_from_wildcards(&mut query.body),
        _ => {},
    }
}

#[async_trait::async_trait]
impl AsyncDB for HybridDuckLakeDB {
    type Error = HybridError;
    type ColumnType = DefaultColumnType;

    async fn run(&mut self, sql: &str) -> Result<DBOutput<Self::ColumnType>, Self::Error> {
        // `statement maybe`: success, or failure with the expected error text.
        let (maybe, sql) = match sql.trim_start().strip_prefix(MAYBE_MARKER) {
            Some(rest) => match rest.split_once("*/") {
                Some((expected, sql)) => (Some(expected.to_string()), sql),
                None => (None, sql),
            },
            None => (None, sql),
        };
        let (label, sql) = match sql.trim_start().strip_prefix(LABEL_MARKER) {
            Some(rest) => match rest.split_once("*/") {
                Some((label, sql)) => (Some(label.to_string()), sql),
                None => (None, sql),
            },
            None => (None, sql),
        };
        let mut result = self.run_statement(sql).await;
        if let (
            Some(label),
            Ok(DBOutput::Rows {
                rows,
                ..
            }),
        ) = (label, &result)
        {
            let mut inner = self.inner.lock().unwrap();
            let plain = rows
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|value| untagged(value).to_string())
                        .collect()
                })
                .collect::<Vec<Vec<String>>>();
            let first = inner.labels.entry(label).or_insert_with(|| plain.clone());
            if *first == plain {
                result = Ok(DBOutput::Rows {
                    types: vec![DefaultColumnType::Any],
                    rows: vec![vec![LABEL_MATCH.to_string()]],
                });
            }
        }
        match (maybe, result) {
            (Some(expected), Err(error)) if error.0.contains(expected.trim()) => {
                Ok(DBOutput::StatementComplete(0))
            },
            (_, result) => result,
        }
    }

    fn engine_name(&self) -> &str {
        "HybridDuckLake(DuckDB+DataFusion)"
    }
}

/// Render DataFusion batches the way DuckDB's sqllogictest runner renders values.
fn batches_to_rows(batches: &[RecordBatch]) -> Result<Vec<Vec<String>>, HybridError> {
    let mut rows = Vec::new();
    for batch in batches {
        for row in 0..batch.num_rows() {
            let mut values = Vec::with_capacity(batch.num_columns());
            for column in batch.columns() {
                let value = ScalarValue::try_from_array(column.as_ref(), row)?;
                values.push(format!(
                    "{}{}",
                    column_tag(column.data_type()),
                    format_scalar(&value, true)
                ));
            }
            rows.push(values);
        }
    }
    Ok(rows)
}

/// DuckDB's rendering of a DOUBLE: shortest round-trip digits, a `.0` on integral
/// values, and a signed two-digit exponent (`1e+20`, `1.5e-07`).
fn format_f64(value: f64) -> String {
    if value.is_nan() {
        return "nan".to_string();
    }
    if value.is_infinite() {
        return if value > 0.0 {
            "inf"
        } else {
            "-inf"
        }
        .to_string();
    }
    format_float_debug(format!("{value:?}"))
}

fn format_f32(value: f32) -> String {
    if value.is_nan() {
        return "nan".to_string();
    }
    if value.is_infinite() {
        return if value > 0.0 {
            "inf"
        } else {
            "-inf"
        }
        .to_string();
    }
    format_float_debug(format!("{value:?}"))
}

fn format_float_debug(debug: String) -> String {
    match debug.split_once('e') {
        Some((mantissa, exponent)) => {
            let (sign, digits) = match exponent.strip_prefix('-') {
                Some(digits) => ('-', digits),
                None => ('+', exponent),
            };
            let mantissa = mantissa.strip_suffix(".0").unwrap_or(mantissa);
            format!("{mantissa}e{sign}{digits:0>2}")
        },
        None => debug,
    }
}

fn format_decimal(unscaled: &str, scale: i8) -> String {
    if scale <= 0 {
        return format!("{unscaled}{}", "0".repeat((-scale) as usize));
    }
    let scale = scale as usize;
    let (sign, digits) = match unscaled.strip_prefix('-') {
        Some(digits) => ("-", digits),
        None => ("", unscaled),
    };
    let digits = format!("{digits:0>width$}", width = scale + 1);
    let (whole, fraction) = digits.split_at(digits.len() - scale);
    format!("{sign}{whole}.{fraction}")
}

fn format_blob(bytes: &[u8]) -> String {
    bytes
        .iter()
        .map(|b| {
            if (0x20..0x7f).contains(b) && *b != b'\\' {
                (*b as char).to_string()
            } else {
                format!("\\x{b:02X}")
            }
        })
        .collect()
}

/// `HH:MM:SS` plus fractional digits with trailing zeros removed.
fn format_time_of_day(nanos_of_day: i64) -> String {
    let seconds = nanos_of_day.div_euclid(1_000_000_000);
    let fraction = nanos_of_day.rem_euclid(1_000_000_000);
    let mut out = format!(
        "{:02}:{:02}:{:02}",
        seconds / 3600,
        (seconds / 60) % 60,
        seconds % 60
    );
    if fraction != 0 {
        let digits = format!("{fraction:09}");
        out.push('.');
        out.push_str(digits.trim_end_matches('0'));
    }
    out
}

/// DuckDB's `infinity` sentinels, as microseconds (timestamps) and days (dates).
const TIMESTAMP_INFINITY_MICROS: i128 = i64::MAX as i128;
const DATE_INFINITY_DAYS: i64 = i32::MAX as i64;

fn format_timestamp(nanos: i128, zoned: bool) -> String {
    if nanos.abs() == TIMESTAMP_INFINITY_MICROS * 1_000 {
        return if nanos > 0 {
            "infinity"
        } else {
            "-infinity"
        }
        .to_string();
    }
    let day_nanos: i128 = 86_400 * 1_000_000_000;
    let days = nanos.div_euclid(day_nanos) as i64;
    let time = nanos.rem_euclid(day_nanos) as i64;
    let mut out = format!("{} {}", format_date(days), format_time_of_day(time));
    if zoned {
        out.push_str("+00");
    }
    out
}

/// A date as DuckDB renders it: `YYYY-MM-DD`, with ` (BC)` before year 1.
fn format_date(days: i64) -> String {
    if days.abs() == DATE_INFINITY_DAYS {
        return if days > 0 {
            "infinity"
        } else {
            "-infinity"
        }
        .to_string();
    }
    // Civil date from days since 1970-01-01 (proleptic Gregorian).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let day_of_era = z.rem_euclid(146_097);
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let mp = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 {
        mp + 3
    } else {
        mp - 9
    };
    let year = year_of_era + era * 400 + i64::from(month <= 2);
    if year <= 0 {
        format!("{:04}-{month:02}-{day:02} (BC)", 1 - year)
    } else {
        format!("{year:04}-{month:02}-{day:02}")
    }
}

fn format_string(value: &str, top: bool) -> String {
    if top && value.is_empty() {
        "(empty)".to_string()
    } else {
        value.to_string()
    }
}

/// Render a DataFusion value as DuckDB's sqllogictest runner renders it.
fn format_scalar(value: &ScalarValue, top: bool) -> String {
    if value.is_null() {
        return "NULL".to_string();
    }
    let elements = |array: &dyn Array| -> Vec<String> {
        (0..array.len())
            .map(|i| match ScalarValue::try_from_array(array, i) {
                Ok(element) => format_scalar(&element, false),
                Err(error) => format!("<{error}>"),
            })
            .collect()
    };
    match value {
        ScalarValue::Boolean(Some(v)) => v.to_string(),
        ScalarValue::Float16(Some(v)) => format_f32(v.to_f32()),
        ScalarValue::Float32(Some(v)) => format_f32(*v),
        ScalarValue::Float64(Some(v)) => format_f64(*v),
        ScalarValue::Decimal128(Some(v), _, scale) => format_decimal(&v.to_string(), *scale),
        ScalarValue::Decimal256(Some(v), _, scale) => format_decimal(&v.to_string(), *scale),
        ScalarValue::Utf8(Some(v))
        | ScalarValue::LargeUtf8(Some(v))
        | ScalarValue::Utf8View(Some(v)) => format_string(v, top),
        // This crate reads DuckLake's `uuid` as FixedSizeBinary(16).
        ScalarValue::FixedSizeBinary(16, Some(v)) if v.len() == 16 => {
            let hex: String = v.iter().map(|b| format!("{b:02x}")).collect();
            format!(
                "{}-{}-{}-{}-{}",
                &hex[0..8],
                &hex[8..12],
                &hex[12..16],
                &hex[16..20],
                &hex[20..32]
            )
        },
        ScalarValue::Binary(Some(v))
        | ScalarValue::LargeBinary(Some(v))
        | ScalarValue::BinaryView(Some(v))
        | ScalarValue::FixedSizeBinary(_, Some(v)) => format_blob(v),
        ScalarValue::Date32(Some(v)) => format_date(i64::from(*v)),
        ScalarValue::Date64(Some(v)) => format_date(v.div_euclid(86_400_000)),
        ScalarValue::Time32Second(Some(v)) => format_time_of_day(i64::from(*v) * 1_000_000_000),
        ScalarValue::Time32Millisecond(Some(v)) => format_time_of_day(i64::from(*v) * 1_000_000),
        ScalarValue::Time64Microsecond(Some(v)) => format_time_of_day(v * 1_000),
        ScalarValue::Time64Nanosecond(Some(v)) => format_time_of_day(*v),
        ScalarValue::TimestampSecond(Some(v), tz) => {
            format_timestamp(i128::from(*v) * 1_000_000_000, tz.is_some())
        },
        ScalarValue::TimestampMillisecond(Some(v), tz) => {
            format_timestamp(i128::from(*v) * 1_000_000, tz.is_some())
        },
        ScalarValue::TimestampMicrosecond(Some(v), tz) => {
            format_timestamp(i128::from(*v) * 1_000, tz.is_some())
        },
        ScalarValue::TimestampNanosecond(Some(v), tz) => {
            format_timestamp(i128::from(*v), tz.is_some())
        },
        ScalarValue::List(array) => format!("[{}]", elements(array.value(0).as_ref()).join(", ")),
        ScalarValue::LargeList(array) => {
            format!("[{}]", elements(array.value(0).as_ref()).join(", "))
        },
        ScalarValue::FixedSizeList(array) => {
            format!("[{}]", elements(array.value(0).as_ref()).join(", "))
        },
        ScalarValue::Struct(array) => {
            let fields: Vec<String> = array
                .fields()
                .iter()
                .zip(array.columns())
                .map(|(field, column)| {
                    let value = ScalarValue::try_from_array(column.as_ref(), 0)
                        .map(|v| format_scalar(&v, false))
                        .unwrap_or_else(|error| format!("<{error}>"));
                    format!("'{}': {value}", field.name())
                })
                .collect();
            format!("{{{}}}", fields.join(", "))
        },
        ScalarValue::Map(array) => {
            let entries = array.value(0);
            let keys = elements(entries.column(0).as_ref());
            let values = elements(entries.column(1).as_ref());
            let pairs: Vec<String> = keys
                .iter()
                .zip(values.iter())
                .map(|(k, v)| format!("{k}={v}"))
                .collect();
            format!("{{{}}}", pairs.join(", "))
        },
        ScalarValue::Dictionary(_, inner) => format_scalar(inner, top),
        other => other.to_string(),
    }
}

/// Render a DuckDB value as DuckDB's sqllogictest runner renders it.
fn format_duckdb_value(value: &duckdb::types::Value, zoned: bool, top: bool) -> String {
    use duckdb::types::{TimeUnit, Value};
    let nanos = |unit: &TimeUnit, v: i64| -> i128 {
        let v = i128::from(v);
        match unit {
            TimeUnit::Second => v * 1_000_000_000,
            TimeUnit::Millisecond => v * 1_000_000,
            TimeUnit::Microsecond => v * 1_000,
            TimeUnit::Nanosecond => v,
        }
    };
    let list = |values: &[Value]| {
        let items: Vec<String> = values
            .iter()
            .map(|v| format_duckdb_value(v, false, false))
            .collect();
        format!("[{}]", items.join(", "))
    };
    match value {
        Value::Null => "NULL".to_string(),
        Value::Boolean(v) => v.to_string(),
        Value::TinyInt(v) => v.to_string(),
        Value::SmallInt(v) => v.to_string(),
        Value::Int(v) => v.to_string(),
        Value::BigInt(v) => v.to_string(),
        Value::HugeInt(v) => v.to_string(),
        Value::UHugeInt(v) => v.to_string(),
        Value::UTinyInt(v) => v.to_string(),
        Value::USmallInt(v) => v.to_string(),
        Value::UInt(v) => v.to_string(),
        Value::UBigInt(v) => v.to_string(),
        Value::Float(v) => format_f32(*v),
        Value::Double(v) => format_f64(*v),
        Value::Decimal(v) => v.to_string(),
        Value::Timestamp(unit, v) => format_timestamp(nanos(unit, *v), zoned),
        Value::Text(v) => format_string(v, top),
        Value::Blob(v) | Value::Geometry(v) => format_blob(v),
        Value::Date32(v) => format_date(i64::from(*v)),
        Value::Time64(unit, v) => format_time_of_day(nanos(unit, *v) as i64),
        Value::Interval {
            months,
            days,
            nanos,
        } => format!("{months} months {days} days {nanos} ns"),
        Value::List(values) | Value::Array(values) => list(values),
        Value::Enum(v) => format_string(v, top),
        Value::Struct(fields) => {
            let fields: Vec<String> = fields
                .iter()
                .map(|(name, v)| format!("'{name}': {}", format_duckdb_value(v, false, false)))
                .collect();
            format!("{{{}}}", fields.join(", "))
        },
        Value::Map(entries) => {
            let pairs: Vec<String> = entries
                .iter()
                .map(|(k, v)| {
                    format!(
                        "{}={}",
                        format_duckdb_value(k, false, false),
                        format_duckdb_value(v, false, false)
                    )
                })
                .collect();
            format!("{{{}}}", pairs.join(", "))
        },
        Value::Union(v) => format_duckdb_value(v, zoned, top),
        other => format!("{other:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Only a missing field that is a DuckDB-only virtual column hands the query
    /// to DuckDB.
    #[test]
    fn missing_virtual_columns() {
        assert_eq!(
            missing_virtual_column("Schema error: No field named filename. Valid fields are a."),
            Some("filename")
        );
        assert_eq!(
            missing_virtual_column("Schema error: No field named t.snapshot_id."),
            Some("snapshot_id")
        );
        assert_eq!(
            missing_virtual_column("Schema error: No field named nope. Valid fields are a."),
            None
        );
    }

    /// Projected virtual columns become `NULL`; other uses stay.
    #[test]
    fn virtual_columns_in_projection_become_null() {
        let mut statement = Parser::parse_sql(
            &GenericDialect {},
            "SELECT rowid, snapshot_id, filename AS f, * FROM t WHERE file_index > 0",
        )
        .unwrap()
        .pop()
        .unwrap();
        assert_eq!(
            null_projected_virtual_columns(&mut statement),
            vec!["snapshot_id", "filename"]
        );
        assert_eq!(
            statement.to_string(),
            "SELECT rowid, NULL AS snapshot_id, NULL AS f, * FROM t WHERE file_index > 0"
        );
    }

    /// A CTE over a table function that outputs `snapshot_id` reads the real
    /// column on DataFusion: only a virtual column DataFusion lacks is nulled.
    #[tokio::test(flavor = "multi_thread")]
    async fn snapshot_id_of_a_cte_over_a_table_function_is_real() {
        let temp = tempfile::TempDir::new().unwrap();
        let mut db = HybridDuckLakeDB::new().unwrap();
        for statement in [
            format!(
                "ATTACH 'ducklake:{}' AS lake (DATA_PATH '{}/')",
                temp.path().join("c.ducklake").display(),
                temp.path().display()
            ),
            "CREATE TABLE lake.t (a INTEGER)".to_string(),
            "INSERT INTO lake.t VALUES (1)".to_string(),
        ] {
            db.run(&statement).await.unwrap();
        }
        let before = db.stats().datafusion_other_reads + db.stats().datafusion_lake_reads;
        let DBOutput::Rows {
            rows,
            ..
        } = db
            .run(
                "WITH s AS (SELECT * FROM ducklake_snapshots('lake')) \
                 SELECT snapshot_id FROM s ORDER BY snapshot_id",
            )
            .await
            .unwrap()
        else {
            panic!("expected rows");
        };
        let ids = rows
            .iter()
            .map(|row| untagged(&row[0]).to_string())
            .collect::<Vec<_>>();
        assert_eq!(ids, vec!["0", "1", "2"], "real snapshot ids, not NULL");
        assert_eq!(
            db.stats().datafusion_other_reads + db.stats().datafusion_lake_reads,
            before + 1,
            "DataFusion answered the query"
        );
        assert!(
            !db.stats()
                .duckdb_reads
                .keys()
                .any(|reason| reason.starts_with("virtual columns")),
            "DataFusion's own snapshot_id was answered, not replaced by NULL: {:?}",
            db.stats().duckdb_reads
        );
    }

    #[test]
    fn test_query_detection() {
        assert!(starts_query("SELECT * FROM foo"));
        assert!(starts_query("  with t AS (SELECT 1) SELECT * FROM t"));
        assert!(starts_query("FROM foo"));
        assert!(!starts_query("INSERT INTO foo VALUES (1)"));
        assert!(!starts_query("CALL ducklake_flush_inlined_data('lake')"));
        assert!(!starts_query("SELECTED"));
    }

    #[test]
    fn test_time_travel_rewrite() {
        assert_eq!(
            rewrite_time_travel("SELECT count(*) FROM ducklake.t AT (VERSION => 2) WHERE a = 1"),
            "SELECT count(*) FROM __hybrid_at('ducklake', 't', 2) WHERE a = 1"
        );
        assert_eq!(
            rewrite_time_travel("SELECT 'x AT (VERSION => 2)' FROM t"),
            "SELECT 'x AT (VERSION => 2)' FROM t"
        );
    }

    #[test]
    fn test_value_formatting() {
        assert_eq!(format_f64(100.0), "100.0");
        assert_eq!(format_f64(0.1), "0.1");
        assert_eq!(format_f64(1e20), "1e+20");
        assert_eq!(format_f64(1.5e-7), "1.5e-07");
        assert_eq!(format_decimal("123450", 3), "123.450");
        assert_eq!(format_decimal("-5", 2), "-0.05");
        assert_eq!(format_blob(b"ab\x01"), "ab\\x01");
        assert_eq!(
            format_timestamp(1_500_000_000, false),
            "1970-01-01 00:00:01.5"
        );
        assert_eq!(
            unknown_function("Error during planning: Invalid function 'stats'."),
            Some("stats".to_string())
        );
    }
}
