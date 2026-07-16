//! User-Defined Table Functions (UDTFs) for DuckLake catalog metadata

use datafusion::catalog::TableFunctionImpl;
use datafusion::common::{Result as DataFusionResult, ScalarValue, plan_err};
use datafusion::datasource::TableProvider;
use datafusion::logical_expr::Expr;
use std::sync::Arc;

use crate::information_schema::{FilesTable, SnapshotsTable, TableInfoTable};
use crate::metadata_provider::MetadataProvider;
use crate::path_resolver::{parse_object_store_url, resolve_path};
use crate::table_changes::{TableChangesTable, TableInsertionsTable};
use crate::table_deletions::TableDeletionsTable;
use crate::types::build_arrow_schema;

#[derive(Debug)]
pub struct DucklakeSnapshotsFunction {
    provider: Arc<dyn MetadataProvider>,
}

impl DucklakeSnapshotsFunction {
    pub fn new(provider: Arc<dyn MetadataProvider>) -> Self {
        Self {
            provider,
        }
    }
}

impl TableFunctionImpl for DucklakeSnapshotsFunction {
    fn call(&self, exprs: &[Expr]) -> DataFusionResult<Arc<dyn TableProvider>> {
        if !exprs.is_empty() {
            return plan_err!("ducklake_snapshots() takes no arguments");
        }

        Ok(Arc::new(SnapshotsTable::new(self.provider.clone())))
    }
}

#[derive(Debug)]
pub struct DucklakeTableInfoFunction {
    provider: Arc<dyn MetadataProvider>,
}

impl DucklakeTableInfoFunction {
    pub fn new(provider: Arc<dyn MetadataProvider>) -> Self {
        Self {
            provider,
        }
    }
}

impl TableFunctionImpl for DucklakeTableInfoFunction {
    fn call(&self, exprs: &[Expr]) -> DataFusionResult<Arc<dyn TableProvider>> {
        if !exprs.is_empty() {
            return plan_err!("ducklake_table_info() takes no arguments");
        }

        Ok(Arc::new(TableInfoTable::new(self.provider.clone())))
    }
}

#[derive(Debug)]
pub struct DucklakeListFilesFunction {
    provider: Arc<dyn MetadataProvider>,
}

impl DucklakeListFilesFunction {
    pub fn new(provider: Arc<dyn MetadataProvider>) -> Self {
        Self {
            provider,
        }
    }
}

impl TableFunctionImpl for DucklakeListFilesFunction {
    fn call(&self, exprs: &[Expr]) -> DataFusionResult<Arc<dyn TableProvider>> {
        if !exprs.is_empty() {
            return plan_err!("ducklake_list_files() takes no arguments");
        }

        Ok(Arc::new(FilesTable::new(self.provider.clone())))
    }
}

#[derive(Debug)]
pub struct DucklakeTableChangesFunction {
    provider: Arc<dyn MetadataProvider>,
}

impl DucklakeTableChangesFunction {
    pub fn new(provider: Arc<dyn MetadataProvider>) -> Self {
        Self {
            provider,
        }
    }
}

/// Everything a CDC table function needs about its target table.
struct CdcTableContext {
    table_id: i64,
    object_store_url: datafusion::execution::object_store::ObjectStoreUrl,
    table_path: String,
    table_schema: Arc<arrow::datatypes::Schema>,
}

/// Split `'schema.table'` (defaulting to schema `main`).
fn parse_table_name(table_name: &str) -> (&str, &str) {
    if let Some(dot_pos) = table_name.find('.') {
        (&table_name[..dot_pos], &table_name[dot_pos + 1..])
    } else {
        ("main", table_name)
    }
}

/// Parse a snapshot-time string as stored by the catalog backends. Accepts
/// `YYYY-MM-DD HH:MM:SS[.fff]` / `YYYY-MM-DDTHH:MM:SS[.fff]` / `YYYY-MM-DD`,
/// with an optional trailing `Z`, ` UTC`, `+00` or `+00:00` (times are UTC).
fn parse_snapshot_timestamp(raw: &str) -> Option<chrono::NaiveDateTime> {
    let mut t = raw.trim();
    for suffix in ["Z", " UTC", "+00:00", "+00"] {
        if let Some(stripped) = t.strip_suffix(suffix) {
            t = stripped.trim();
            break;
        }
    }
    for fmt in ["%Y-%m-%d %H:%M:%S%.f", "%Y-%m-%dT%H:%M:%S%.f"] {
        if let Ok(ts) = chrono::NaiveDateTime::parse_from_str(t, fmt) {
            return Some(ts);
        }
    }
    chrono::NaiveDate::parse_from_str(t, "%Y-%m-%d")
        .ok()
        .and_then(|d| d.and_hms_opt(0, 0, 0))
}

/// Resolve a snapshot bound argument: an integer snapshot id, or a timestamp
/// (a string literal or a `TIMESTAMP '...'` cast) resolved against the
/// catalog's snapshot times — a START bound resolves to the FIRST snapshot
/// at-or-after the timestamp, an END bound to the LAST snapshot at-or-before
/// it, matching official DuckLake's `AT (TIMESTAMP => ...)` semantics.
fn resolve_snapshot_bound(
    provider: &Arc<dyn MetadataProvider>,
    expr: &Expr,
    fn_name: &str,
    arg_name: &str,
    is_start: bool,
) -> DataFusionResult<i64> {
    // `TIMESTAMP '...'` parses as a cast around a literal; unwrap it.
    let mut expr = expr;
    while let Expr::Cast(cast) = expr {
        expr = cast.expr.as_ref();
    }
    let ts: chrono::NaiveDateTime = match expr {
        Expr::Literal(ScalarValue::Int64(Some(v)), _) => return Ok(*v),
        Expr::Literal(ScalarValue::Int32(Some(v)), _) => return Ok(*v as i64),
        Expr::Literal(ScalarValue::Utf8(Some(s)), _) => match parse_snapshot_timestamp(s) {
            Some(ts) => ts,
            None => {
                return plan_err!(
                    "{arg_name} of {fn_name}() is not a valid timestamp: '{s}' \
                     (expected 'YYYY-MM-DD[ HH:MM:SS[.ffffff]]', UTC)"
                );
            },
        },
        Expr::Literal(other, _) => {
            let unit_and_value = match other {
                ScalarValue::TimestampSecond(Some(v), _) => Some((*v, 1_000_000_000i64)),
                ScalarValue::TimestampMillisecond(Some(v), _) => Some((*v, 1_000_000)),
                ScalarValue::TimestampMicrosecond(Some(v), _) => Some((*v, 1_000)),
                ScalarValue::TimestampNanosecond(Some(v), _) => Some((*v, 1)),
                _ => None,
            };
            match unit_and_value {
                Some((v, factor)) => {
                    let nanos = v.saturating_mul(factor);
                    match chrono::DateTime::from_timestamp(
                        nanos.div_euclid(1_000_000_000),
                        (nanos.rem_euclid(1_000_000_000)) as u32,
                    ) {
                        Some(dt) => dt.naive_utc(),
                        None => {
                            return plan_err!(
                                "{arg_name} of {fn_name}() is out of timestamp range"
                            );
                        },
                    }
                },
                None => {
                    return plan_err!(
                        "{arg_name} of {fn_name}() must be an integer snapshot id or a \
                         timestamp literal"
                    );
                },
            }
        },
        _ => {
            return plan_err!(
                "{arg_name} of {fn_name}() must be an integer snapshot id or a timestamp \
                 literal"
            );
        },
    };

    let snapshots = provider
        .list_snapshots()
        .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
    let mut best: Option<i64> = None;
    for s in &snapshots {
        let Some(t) = s.timestamp.as_deref().and_then(parse_snapshot_timestamp) else {
            continue;
        };
        let candidate = if is_start {
            t >= ts
        } else {
            t <= ts
        };
        if candidate {
            best = Some(match best {
                None => s.snapshot_id,
                Some(b) if is_start => b.min(s.snapshot_id),
                Some(b) => b.max(s.snapshot_id),
            });
        }
    }
    best.ok_or_else(|| {
        datafusion::error::DataFusionError::Plan(format!(
            "{fn_name}(): no snapshot {} timestamp {ts}",
            if is_start {
                "at or after"
            } else {
                "at or before"
            },
        ))
    })
}

/// Parse the common `('schema.table', start, end)` argument list of the CDC
/// table functions and resolve the target table.
fn parse_cdc_args(
    provider: &Arc<dyn MetadataProvider>,
    exprs: &[Expr],
    fn_name: &str,
) -> DataFusionResult<(i64, i64, CdcTableContext)> {
    if exprs.len() != 3 {
        return plan_err!(
            "{fn_name}() requires 3 arguments: \
             {fn_name}('schema.table', start_snapshot, end_snapshot)"
        );
    }
    let table_name = match &exprs[0] {
        Expr::Literal(ScalarValue::Utf8(Some(name)), _) => name.clone(),
        _ => {
            return plan_err!(
                "First argument to {fn_name}() must be a string literal \
                 (e.g., 'main.users' or 'users')"
            );
        },
    };
    let start_snapshot =
        resolve_snapshot_bound(provider, &exprs[1], fn_name, "start_snapshot", true)?;
    let end_snapshot = resolve_snapshot_bound(provider, &exprs[2], fn_name, "end_snapshot", false)?;
    if start_snapshot > end_snapshot {
        return plan_err!(
            "start_snapshot ({}) must be less than or equal to end_snapshot ({})",
            start_snapshot,
            end_snapshot
        );
    }

    let (schema_name, table_name_only) = parse_table_name(&table_name);
    let snapshot_id = provider
        .get_current_snapshot()
        .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
    let schema = provider
        .get_schema_by_name(schema_name, snapshot_id)
        .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?
        .ok_or_else(|| {
            datafusion::error::DataFusionError::Plan(format!(
                "Schema '{}' not found in catalog",
                schema_name
            ))
        })?;
    let table = provider
        .get_table_by_name(schema.schema_id, table_name_only, snapshot_id)
        .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?
        .ok_or_else(|| {
            datafusion::error::DataFusionError::Plan(format!(
                "Table '{}.{}' not found in catalog",
                schema_name, table_name_only
            ))
        })?;
    let data_path = provider
        .get_data_path()
        .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
    let (object_store_url, catalog_path) = parse_object_store_url(&data_path)
        .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
    let schema_path = resolve_path(&catalog_path, &schema.path, schema.path_is_relative)
        .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
    let table_path = resolve_path(&schema_path, &table.path, table.path_is_relative)
        .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
    let columns = provider
        .get_table_structure(table.table_id, end_snapshot)
        .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
    let table_schema = Arc::new(
        build_arrow_schema(&columns)
            .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?,
    );

    Ok((
        start_snapshot,
        end_snapshot,
        CdcTableContext {
            table_id: table.table_id,
            object_store_url,
            table_path,
            table_schema,
        },
    ))
}

impl TableFunctionImpl for DucklakeTableChangesFunction {
    fn call(&self, exprs: &[Expr]) -> DataFusionResult<Arc<dyn TableProvider>> {
        let (start_snapshot, end_snapshot, ctx) =
            parse_cdc_args(&self.provider, exprs, "ducklake_table_changes")?;
        Ok(Arc::new(TableChangesTable::new(
            self.provider.clone(),
            ctx.table_id,
            start_snapshot,
            end_snapshot,
            Arc::new(ctx.object_store_url),
            ctx.table_path,
            ctx.table_schema,
        )))
    }
}

#[derive(Debug)]
pub struct DucklakeTableDeletionsFunction {
    provider: Arc<dyn MetadataProvider>,
}

impl DucklakeTableDeletionsFunction {
    pub fn new(provider: Arc<dyn MetadataProvider>) -> Self {
        Self {
            provider,
        }
    }
}

impl TableFunctionImpl for DucklakeTableDeletionsFunction {
    fn call(&self, exprs: &[Expr]) -> DataFusionResult<Arc<dyn TableProvider>> {
        let (start_snapshot, end_snapshot, ctx) =
            parse_cdc_args(&self.provider, exprs, "ducklake_table_deletions")?;
        Ok(Arc::new(TableDeletionsTable::new(
            self.provider.clone(),
            ctx.table_id,
            start_snapshot,
            end_snapshot,
            Arc::new(ctx.object_store_url),
            ctx.table_path,
            ctx.table_schema,
        )))
    }
}

/// `ducklake_table_insertions('schema.table', start, end)`: every row added
/// in the inclusive snapshot window, with `(snapshot_id, rowid)` leading and
/// no `change_type` column — official DuckLake's insertions feed.
#[derive(Debug)]
pub struct DucklakeTableInsertionsFunction {
    provider: Arc<dyn MetadataProvider>,
}

impl DucklakeTableInsertionsFunction {
    pub fn new(provider: Arc<dyn MetadataProvider>) -> Self {
        Self {
            provider,
        }
    }
}

impl TableFunctionImpl for DucklakeTableInsertionsFunction {
    fn call(&self, exprs: &[Expr]) -> DataFusionResult<Arc<dyn TableProvider>> {
        let (start_snapshot, end_snapshot, ctx) =
            parse_cdc_args(&self.provider, exprs, "ducklake_table_insertions")?;
        Ok(Arc::new(TableInsertionsTable::new(
            self.provider.clone(),
            ctx.table_id,
            start_snapshot,
            end_snapshot,
            Arc::new(ctx.object_store_url),
            ctx.table_path,
            ctx.table_schema,
        )))
    }
}

/// Registers all ducklake_*() table functions with a SessionContext.
pub fn register_ducklake_functions(
    ctx: &datafusion::execution::context::SessionContext,
    provider: Arc<dyn MetadataProvider>,
) {
    ctx.register_udtf(
        "ducklake_snapshots",
        Arc::new(DucklakeSnapshotsFunction::new(provider.clone())),
    );
    ctx.register_udtf(
        "ducklake_table_info",
        Arc::new(DucklakeTableInfoFunction::new(provider.clone())),
    );
    ctx.register_udtf(
        "ducklake_list_files",
        Arc::new(DucklakeListFilesFunction::new(provider.clone())),
    );
    ctx.register_udtf(
        "ducklake_table_changes",
        Arc::new(DucklakeTableChangesFunction::new(provider.clone())),
    );
    ctx.register_udtf(
        "ducklake_table_insertions",
        Arc::new(DucklakeTableInsertionsFunction::new(provider.clone())),
    );
    ctx.register_udtf(
        "ducklake_table_deletions",
        Arc::new(DucklakeTableDeletionsFunction::new(provider.clone())),
    );
}

#[cfg(test)]
mod snapshot_bound_tests {
    use super::parse_snapshot_timestamp;

    #[test]
    fn parses_backend_snapshot_time_formats() {
        for raw in [
            "2026-07-16 12:34:56.123456",
            "2026-07-16 12:34:56",
            "2026-07-16T12:34:56.123456",
            "2026-07-16 12:34:56+00",
            "2026-07-16 12:34:56+00:00",
            "2026-07-16 12:34:56 UTC",
            "2026-07-16T12:34:56Z",
        ] {
            assert!(
                parse_snapshot_timestamp(raw).is_some(),
                "failed to parse {raw:?}"
            );
        }
        // A bare date is midnight UTC.
        assert_eq!(
            parse_snapshot_timestamp("2026-07-16").unwrap(),
            parse_snapshot_timestamp("2026-07-16 00:00:00").unwrap()
        );
    }

    #[test]
    fn rejects_garbage() {
        assert!(parse_snapshot_timestamp("not a time").is_none());
        assert!(parse_snapshot_timestamp("").is_none());
    }
}
