//! One snapshot per statement for catalogs that read the latest snapshot.
//!
//! DataFusion looks up each table reference of a statement on its own, so a commit that lands
//! between two lookups would put one statement on two snapshots. Official DuckLake loads the
//! snapshot once per transaction. [`register_snapshot_consistency`] adds an analyzer rule that
//! rebuilds every latest-snapshot table of a plan, and every table of a view planned over one,
//! at the newest snapshot any table of the same catalog resolved.

use std::collections::HashMap;
use std::sync::Arc;

use datafusion::common::config::ConfigOptions;
use datafusion::common::tree_node::{Transformed, TreeNodeRecursion};
use datafusion::common::{Result as DataFusionResult, plan_datafusion_err};
use datafusion::datasource::{TableProvider, provider_as_source, source_as_provider};
use datafusion::execution::context::SessionContext;
use datafusion::logical_expr::{DmlStatement, LogicalPlan, TableSource};
use datafusion::optimizer::AnalyzerRule;

use crate::metadata_provider::ViewMetadata;
use crate::schema::DuckLakeSchema;
use crate::table::DuckLakeTable;

/// Add the snapshot consistency analyzer rule to a session. Without it, each table reference of a
/// statement reads the latest snapshot at its own lookup.
pub fn register_snapshot_consistency(ctx: &SessionContext) {
    let registered = ctx
        .state()
        .analyzer()
        .rules
        .iter()
        .any(|rule| rule.name() == RULE_NAME);
    if !registered {
        ctx.add_analyzer_rule(Arc::new(DuckLakeSnapshotConsistencyRule));
    }
}

const RULE_NAME: &str = "ducklake_snapshot_consistency";

/// A view a table was planned through. Moving the table to another snapshot is valid only if
/// the view is unchanged there.
#[derive(Debug, Clone)]
pub(crate) struct ViewGuard {
    view: ViewMetadata,
}

impl ViewGuard {
    pub(crate) fn new(view: ViewMetadata) -> Self {
        Self {
            view,
        }
    }
}

/// How a table from a latest-snapshot catalog, or from a view planned over one, is rebuilt at
/// another snapshot.
#[derive(Debug)]
pub(crate) struct SnapshotRebind {
    schema: DuckLakeSchema,
    table_name: String,
    guards: Arc<Vec<ViewGuard>>,
}

impl SnapshotRebind {
    pub(crate) fn new(
        schema: DuckLakeSchema,
        table_name: String,
        guards: Arc<Vec<ViewGuard>>,
    ) -> Arc<Self> {
        Arc::new(Self {
            schema,
            table_name,
            guards,
        })
    }

    /// Snapshot IDs are comparable only within one catalog, which is one metadata provider.
    fn catalog_key(&self) -> usize {
        self.schema.provider_key()
    }

    fn table_at(&self, table: &DuckLakeTable, snapshot_id: i64) -> DataFusionResult<DuckLakeTable> {
        let changed = || {
            plan_datafusion_err!(
                "Table '{}' changed between lookups of one statement; retry the statement",
                self.table_name
            )
        };
        for guard in self.guards.iter() {
            let current = self.schema.view_at(&guard.view.view_name, snapshot_id)?;
            if current.as_ref() != Some(&guard.view) {
                return Err(changed());
            }
        }
        match self.schema.table_at(&self.table_name, snapshot_id)? {
            Some(rebuilt)
                if rebuilt.table_id() == table.table_id() && rebuilt.schema() == table.schema() =>
            {
                Ok(rebuilt)
            },
            _ => Err(changed()),
        }
    }
}

#[derive(Debug)]
struct DuckLakeSnapshotConsistencyRule;

impl AnalyzerRule for DuckLakeSnapshotConsistencyRule {
    fn analyze(&self, plan: LogicalPlan, _config: &ConfigOptions) -> DataFusionResult<LogicalPlan> {
        let mut newest: HashMap<usize, i64> = HashMap::new();
        let mut mixed = false;
        plan.apply_with_subqueries(|node| {
            let source = match node {
                LogicalPlan::TableScan(scan) => Some(&scan.source),
                LogicalPlan::Dml(dml) => Some(&dml.target),
                _ => None,
            };
            if let Some((key, snapshot_id)) = source.and_then(rebindable) {
                let entry = newest.entry(key).or_insert(snapshot_id);
                mixed |= *entry != snapshot_id;
                *entry = (*entry).max(snapshot_id);
            }
            Ok(TreeNodeRecursion::Continue)
        })?;
        if !mixed {
            return Ok(plan);
        }

        // A table referenced twice is one source, so it is rebuilt once.
        let mut rebuilt: HashMap<usize, Arc<dyn TableSource>> = HashMap::new();
        let mut rebound =
            |source: &Arc<dyn TableSource>| -> DataFusionResult<Option<Arc<dyn TableSource>>> {
                let Some((key, snapshot_id)) = rebindable(source) else {
                    return Ok(None);
                };
                if snapshot_id == newest[&key] {
                    return Ok(None);
                }
                let id = Arc::as_ptr(source) as *const () as usize;
                if let Some(source) = rebuilt.get(&id) {
                    return Ok(Some(Arc::clone(source)));
                }
                let source = rebuild(source, newest[&key])?;
                rebuilt.insert(id, Arc::clone(&source));
                Ok(Some(source))
            };

        plan.transform_up_with_subqueries(|node| match node {
            LogicalPlan::TableScan(mut scan) => match rebound(&scan.source)? {
                Some(source) => {
                    // The rebuilt table has the same schema, so the scan's projection and
                    // filters stay valid.
                    scan.source = source;
                    Ok(Transformed::yes(LogicalPlan::TableScan(scan)))
                },
                None => Ok(Transformed::no(LogicalPlan::TableScan(scan))),
            },
            LogicalPlan::Dml(dml) => match rebound(&dml.target)? {
                Some(target) => Ok(Transformed::yes(LogicalPlan::Dml(DmlStatement::new(
                    dml.table_name,
                    target,
                    dml.op,
                    dml.input,
                )))),
                None => Ok(Transformed::no(LogicalPlan::Dml(dml))),
            },
            other => Ok(Transformed::no(other)),
        })
        .map(|transformed| transformed.data)
    }

    fn name(&self) -> &str {
        RULE_NAME
    }
}

/// The catalog and snapshot of a table that can be rebuilt, or `None` for any other source.
fn rebindable(source: &Arc<dyn TableSource>) -> Option<(usize, i64)> {
    let provider = source_as_provider(source).ok()?;
    let table = provider.downcast_ref::<DuckLakeTable>()?;
    let rebind = table.snapshot_rebind()?;
    Some((rebind.catalog_key(), table.snapshot_id()))
}

fn rebuild(
    source: &Arc<dyn TableSource>,
    snapshot_id: i64,
) -> DataFusionResult<Arc<dyn TableSource>> {
    let provider = source_as_provider(source)?;
    let table = provider
        .downcast_ref::<DuckLakeTable>()
        .ok_or_else(|| plan_datafusion_err!("expected a DuckLake table"))?;
    let rebind = table
        .snapshot_rebind()
        .ok_or_else(|| plan_datafusion_err!("expected a rebindable DuckLake table"))?;
    let rebuilt = rebind.table_at(table, snapshot_id)?;
    Ok(provider_as_source(
        Arc::new(rebuilt) as Arc<dyn TableProvider>
    ))
}
