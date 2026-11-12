//! Filters pushed into the change feeds' insertion reads.
//!
//! A filter a query puts on `ducklake_table_insertions` (or on the rows
//! `ducklake_table_changes` reports) narrows what the feed reads three ways,
//! each only ever dropping rows the filter rejects:
//!
//! - **data files**, by their catalog column statistics: a file whose recorded
//!   bounds prove no row can match is not read at all;
//! - **row groups and pages**, by handing the filter to each file's Parquet
//!   reader;
//! - **inlined rows**, by narrowing the catalog query that reads them.
//!
//! Every filter stays above the feed as well (the providers declare it
//! `Inexact`), so what the feed returns is exactly what it returned unfiltered,
//! filtered. Only conjuncts over the table's own columns are pushed: the CDC
//! columns (`snapshot_id`, `rowid`, `change_type`) are derived by the feed, not
//! read.
//!
//! The deletion side of a feed is never narrowed by data values: a consumer
//! reconciles insertions against deletions (and `ducklake_table_changes` pairs
//! them into update images), so its deletions, and the keys of the insertions
//! it pairs them with, must stay complete for the window.
//!
//! Pruning data files by catalog statistics is a divergence from official
//! DuckLake (at duckdb/ducklake `d8a1881e`). Official's change feeds scan
//! through the table's multi-file list, which accepts a pushed filter only
//! when the scan type is `SCAN_TABLE` (`ducklake_multi_file_list.cpp:61-62`,
//! `DynamicFilterPushdown`, and `:86`, `ComplexFilterPushdown`): a feed
//! (`SCAN_INSERTIONS`, `SCAN_DELETIONS`) lists every file of its window and
//! filters only in the Parquet reader. Pruning is kept here because it is most
//! of what the pushdown buys: a selective filter over a 30-file window read 1
//! file instead of 30 (2.17 MB instead of 64.65 MB, 6.3 ms instead of
//! 25.2 ms). It changes no answer, since every filter is still applied above
//! the feed. Converging means either official taking the same pruning for its
//! insertion scans or this crate dropping [`FeedFilter::pruned_files`]; see
//! `COMPATIBILITY.md`.

use std::collections::HashSet;
use std::sync::Arc;

use arrow::datatypes::{DataType, Schema, SchemaRef};
use datafusion::catalog::Session;
use datafusion::common::DFSchema;
use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::logical_expr::{Expr, TableProviderFilterPushDown};
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_optimizer::pruning::{PruningPredicate, PruningPredicateBuilder};

use crate::inlined_filter::{InlinedFilter, translate_inlined_filters};
use crate::metadata_provider::{
    DataFileChange, DuckLakeTableColumn, FILE_STATISTICS_ID_CHUNK, MetadataProvider,
};

/// Whether `filter`, over a feed whose table columns are `table_schema` and
/// whose derived columns are named `cdc_columns`, can narrow the feed's reads:
/// it reads at least one column, only table columns, none sharing a derived
/// column's name, and is not volatile.
fn pushable(filter: &Expr, table_schema: &Schema, cdc_columns: &[&str]) -> bool {
    if filter.is_volatile() {
        return false;
    }
    let columns = filter.column_refs();
    !columns.is_empty()
        && columns.iter().all(|column| {
            !cdc_columns.contains(&column.name.as_str())
                && table_schema.field_with_name(&column.name).is_ok()
        })
}

/// How a feed provider answers `supports_filters_pushdown`: `Inexact` for each
/// filter it can narrow its reads with, `Unsupported` otherwise. Either way
/// the filter is applied above the feed.
pub(crate) fn filters_pushdown(
    filters: &[&Expr],
    table_schema: &Schema,
    cdc_columns: &[&str],
) -> Vec<TableProviderFilterPushDown> {
    filters
        .iter()
        .map(|filter| {
            if pushable(filter, table_schema, cdc_columns) {
                TableProviderFilterPushDown::Inexact
            } else {
                TableProviderFilterPushDown::Unsupported
            }
        })
        .collect()
}

/// The filters of one feed scan that narrow its insertion reads.
#[derive(Debug)]
pub(crate) struct FeedFilter {
    /// The pushable conjuncts, bound to the table schema (catalog names, in
    /// catalog order).
    conjuncts: Vec<Arc<dyn PhysicalExpr>>,
    /// One pruning predicate per conjunct that builds one, over the catalog
    /// statistics of the table schema.
    pruning: Vec<PruningPredicate>,
    /// The pushable conjuncts as a filter of the inlined-row query.
    inlined: Option<InlinedFilter>,
    table_schema: SchemaRef,
}

impl FeedFilter {
    /// The part of `filters` the feed can push, or `None` when there is none.
    /// A conjunct that does not convert is left out, which only narrows less.
    pub(crate) fn new(
        state: &dyn Session,
        filters: &[Expr],
        table_schema: &SchemaRef,
        cdc_columns: &[&str],
    ) -> Option<Self> {
        let logical: Vec<Expr> = filters
            .iter()
            .flat_map(datafusion::logical_expr::utils::split_conjunction)
            .filter(|conjunct| pushable(conjunct, table_schema, cdc_columns))
            .cloned()
            .collect();
        if logical.is_empty() {
            return None;
        }
        let df_schema = DFSchema::try_from(table_schema.as_ref().clone()).ok()?;
        let conjuncts: Vec<Arc<dyn PhysicalExpr>> = logical
            .iter()
            .filter_map(
                |conjunct| match state.create_physical_expr(conjunct.clone(), &df_schema) {
                    Ok(physical) => Some(physical),
                    Err(error) => {
                        tracing::debug!(%error, %conjunct, "conjunct not pushed into the change feed");
                        None
                    },
                },
            )
            .collect();
        let max_in_list_size = state.config_options().execution.parquet.max_in_list_size;
        let pruning = conjuncts
            .iter()
            .filter_map(|conjunct| {
                PruningPredicateBuilder::new()
                    .with_file_schema(Arc::clone(table_schema))
                    .with_max_in_list_size(max_in_list_size)
                    .try_build(Arc::clone(conjunct))
                    .ok()
            })
            .filter(|predicate| !predicate.always_true())
            .collect();
        Some(Self {
            conjuncts,
            pruning,
            inlined: translate_inlined_filters(&logical),
            table_schema: Arc::clone(table_schema),
        })
    }

    /// The filter of the inlined-row query.
    pub(crate) fn inlined(&self) -> Option<&InlinedFilter> {
        self.inlined.as_ref()
    }

    /// The ids of the data files of `files` whose catalog statistics prove they
    /// hold no row the filter can match, read through `provider` in bounded
    /// chunks. A file the provider gives no id for is kept.
    ///
    /// Official DuckLake prunes no change feed's files (see the module
    /// documentation): this is the divergence, kept for the files and bytes it
    /// saves.
    pub(crate) fn pruned_files(
        &self,
        provider: &dyn MetadataProvider,
        table_id: i64,
        columns: &[DuckLakeTableColumn],
        files: &[DataFileChange],
    ) -> DataFusionResult<HashSet<i64>> {
        let mut pruned = HashSet::new();
        if self.pruning.is_empty() || columns.len() != self.table_schema.fields().len() {
            return Ok(pruned);
        }
        let identified: Vec<(i64, &DataFileChange)> = files
            .iter()
            .filter_map(|file| file.data_file_id.map(|id| (id, file)))
            .collect();
        for chunk in identified.chunks(FILE_STATISTICS_ID_CHUNK) {
            let ids: Vec<i64> = chunk.iter().map(|(id, _)| *id).collect();
            let statistics = provider
                .get_data_file_column_statistics(table_id, &ids)
                .map_err(|e| DataFusionError::External(Box::new(e)))?;
            pruned.extend(crate::table::feed_files_pruned_by_statistics(
                &self.table_schema,
                columns,
                chunk,
                statistics,
                &self.pruning,
            ));
        }
        Ok(pruned)
    }

    /// The predicate to hand the Parquet reader of a file read under
    /// `read_schema` (whose leading fields are the table columns, in catalog
    /// order, under the names the file gives them): the conjuncts whose every
    /// column the file holds under its catalog name and type, and that read no
    /// floating-point column, or `None` when no conjunct qualifies.
    ///
    /// A column the file names differently would not bind (the reader resolves
    /// a predicate by name), one it lacks is filled in rather than read, and
    /// Parquet float bounds leave out NaN, which a comparison can match; a
    /// conjunct over any of them is left to the filter above the feed.
    pub(crate) fn file_predicate(&self, read_schema: &Schema) -> Option<Arc<dyn PhysicalExpr>> {
        let usable = |index: usize| {
            let (Some(table), Some(read)) = (
                self.table_schema.fields().get(index),
                read_schema.fields().get(index),
            ) else {
                return false;
            };
            !matches!(
                table.data_type(),
                DataType::Float16 | DataType::Float32 | DataType::Float64
            ) && table.name() == read.name()
                && table.data_type() == read.data_type()
        };
        let conjuncts: Vec<Arc<dyn PhysicalExpr>> = self
            .conjuncts
            .iter()
            .filter(|conjunct| {
                let mut ok = true;
                conjunct
                    .apply(|expr| {
                        if let Some(column) = expr.downcast_ref::<Column>()
                            && !usable(column.index())
                        {
                            ok = false;
                            return Ok(TreeNodeRecursion::Stop);
                        }
                        Ok(TreeNodeRecursion::Continue)
                    })
                    .expect("visiting a predicate's columns cannot fail");
                ok
            })
            .cloned()
            .collect();
        datafusion::physical_expr::conjunction_opt(conjuncts)
    }
}
