//! [`PositionalFileSource`]: a `FileSource` wrapper for *positional* scan paths
//! (synthetic rowid and/or positional delete filtering).
//!
//! Physical row positions on these paths are reconstructed by
//! [`FileRowNumberExec`](crate::row_id) from row-group-aligned scan partitions.
//! That is correct **only** while each partition emits a complete, contiguous,
//! in-order run of physical rows. This wrapper enforces that by delegating the
//! actual reading to the inner [`ParquetSource`] while refusing every operation
//! that would drop, reorder, or re-split rows:
//!
//! - byte-range repartitioning (would break row-group alignment),
//! - reader-side filter pushdown (row-group/page/bloom pruning would emit a
//!   sparse subset of rows),
//! - sort / reverse-order pushdown (would read row groups out of physical
//!   order).
//!
//! Order- and cardinality-preserving methods (batch size, column projection)
//! are delegated and **re-wrapped**, so the wrapper is never silently dropped
//! and pruning/repartitioning re-enabled.
//!
//! [`ParquetSource`]: datafusion::datasource::physical_plan::ParquetSource

use std::any::Any;
use std::fmt::{self, Formatter};
use std::sync::Arc;

use datafusion::common::config::ConfigOptions;
use datafusion::datasource::physical_plan::{FileOpener, FileScanConfig, FileSource};
use datafusion::datasource::table_schema::TableSchema;
use datafusion::error::Result as DataFusionResult;
use datafusion::physical_expr::projection::ProjectionExprs;
use datafusion::physical_expr::{EquivalenceProperties, LexOrdering, PhysicalExpr};
use datafusion::physical_expr_common::sort_expr::PhysicalSortExpr;
use datafusion::physical_plan::filter_pushdown::{FilterPushdownPropagation, PushedDown};
use datafusion::physical_plan::metrics::ExecutionPlanMetricsSet;
use datafusion::physical_plan::{DisplayFormatType, SortOrderPushdownResult};
use object_store::ObjectStore;

/// Wraps an inner `FileSource` (a configured `ParquetSource`) so DataFusion
/// cannot repartition, prune, or reorder rows on a positional scan path.
pub(crate) struct PositionalFileSource {
    inner: Arc<dyn FileSource>,
}

impl PositionalFileSource {
    pub(crate) fn wrap(inner: Arc<dyn FileSource>) -> Arc<dyn FileSource> {
        Arc::new(Self {
            inner,
        })
    }

    fn rewrap(&self, src: Arc<dyn FileSource>) -> Arc<dyn FileSource> {
        Arc::new(Self {
            inner: src,
        })
    }
}

impl FileSource for PositionalFileSource {
    // ---- delegate: actual file reading and read-only accessors ----

    fn create_file_opener(
        &self,
        object_store: Arc<dyn ObjectStore>,
        base_config: &FileScanConfig,
        partition: usize,
    ) -> DataFusionResult<Arc<dyn FileOpener>> {
        self.inner
            .create_file_opener(object_store, base_config, partition)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn table_schema(&self) -> &TableSchema {
        self.inner.table_schema()
    }

    fn filter(&self) -> Option<Arc<dyn PhysicalExpr>> {
        self.inner.filter()
    }

    fn projection(&self) -> Option<&ProjectionExprs> {
        self.inner.projection()
    }

    fn metrics(&self) -> &ExecutionPlanMetricsSet {
        self.inner.metrics()
    }

    fn file_type(&self) -> &str {
        self.inner.file_type()
    }

    fn fmt_extra(&self, t: DisplayFormatType, f: &mut Formatter) -> fmt::Result {
        self.inner.fmt_extra(t, f)
    }

    // ---- delegate + re-wrap: order/cardinality preserving ----

    fn with_batch_size(&self, batch_size: usize) -> Arc<dyn FileSource> {
        self.rewrap(self.inner.with_batch_size(batch_size))
    }

    fn try_pushdown_projection(
        &self,
        projection: &ProjectionExprs,
    ) -> DataFusionResult<Option<Arc<dyn FileSource>>> {
        Ok(self
            .inner
            .try_pushdown_projection(projection)?
            .map(|s| self.rewrap(s)))
    }

    // ---- refuse: anything that drops, reorders, or re-splits rows ----

    fn supports_repartitioning(&self) -> bool {
        false
    }

    fn repartitioned(
        &self,
        _target_partitions: usize,
        _repartition_file_min_size: usize,
        _output_ordering: Option<LexOrdering>,
        _config: &FileScanConfig,
    ) -> DataFusionResult<Option<FileScanConfig>> {
        Ok(None)
    }

    fn try_pushdown_filters(
        &self,
        filters: Vec<Arc<dyn PhysicalExpr>>,
        _config: &ConfigOptions,
    ) -> DataFusionResult<FilterPushdownPropagation<Arc<dyn FileSource>>> {
        // Refuse all filter pushdown: a pushed predicate enables row-group/page/
        // bloom pruning, which would emit a sparse subset of rows and corrupt
        // physical-position synthesis. DataFusion keeps a FilterExec above us.
        Ok(FilterPushdownPropagation::with_parent_pushdown_result(
            vec![PushedDown::No; filters.len()],
        ))
    }

    fn try_pushdown_sort(
        &self,
        _order: &[PhysicalSortExpr],
        _eq_properties: &EquivalenceProperties,
    ) -> DataFusionResult<SortOrderPushdownResult<Arc<dyn FileSource>>> {
        // Refuse sort/reverse pushdown: reading row groups out of physical order
        // would break position synthesis. DataFusion keeps a SortExec above us.
        Ok(SortOrderPushdownResult::Unsupported)
    }
}
