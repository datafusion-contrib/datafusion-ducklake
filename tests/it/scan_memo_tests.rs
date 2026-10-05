#![cfg(all(feature = "metadata-duckdb", feature = "metadata-sqlite"))]
//! What a built `DuckLakeTable` reads again on each scan, with and without the
//! memos of `DuckLakeReadOptions` (#338).
//!
//! The catalogs are written by official DuckLake through DuckDB, with their
//! metadata in SQLite so a test can commit while a table is open. A
//! [`CountingProvider`] counts the catalog calls each scan makes, and a
//! [`ReadCountingStore`] counts the reads of each file.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use arrow::record_batch::RecordBatch;
use arrow::util::pretty::pretty_format_batches;
use datafusion::catalog::{CatalogProvider, TableProvider};
use datafusion::prelude::SessionContext;
use datafusion_ducklake::inlined_filter::{InlinedDataScan, InlinedFilter};
use datafusion_ducklake::metadata_provider::{
    ColumnWithTable, DataFileChange, DeleteFileChange, DuckLakeFileMetadata, DuckLakeInlinedData,
    DuckLakeInlinedDelete, DuckLakeNameMapping, DuckLakeStatistics, DuckLakeTableColumn,
    DuckLakeTableField, DuckLakeTableFile, FileWithTable, SchemaMetadata, SnapshotMetadata,
    TableMetadata, TableWithSchema, ViewMetadata, ViewWithSchema,
};
use datafusion_ducklake::stats_filter::StatsFilter;
use datafusion_ducklake::{
    ColumnTag, DuckLakeCatalog, DuckLakeReadOptions, DuckLakeTag, MetadataProvider, ObjectTag,
    Result, SnapshotChangeMetadata, SqliteMetadataProvider, TagTarget,
};
use tempfile::TempDir;

use crate::common;

/// Something to run once, just before the next call of a provider method.
type Hook = (&'static str, Box<dyn FnOnce() + Send>);

/// Forwards every `MetadataProvider` method to `inner` and counts the calls.
#[derive(Clone)]
struct CountingProvider {
    inner: Arc<dyn MetadataProvider>,
    calls: Arc<Mutex<BTreeMap<&'static str, usize>>>,
    hook: Arc<Mutex<Option<Hook>>>,
}

impl std::fmt::Debug for CountingProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CountingProvider")
            .field("calls", &self.calls)
            .finish_non_exhaustive()
    }
}

impl CountingProvider {
    fn new(inner: Arc<dyn MetadataProvider>) -> Self {
        Self {
            inner,
            calls: Arc::new(Mutex::new(BTreeMap::new())),
            hook: Arc::new(Mutex::new(None)),
        }
    }

    fn count(&self, method: &'static str) {
        *self.calls.lock().unwrap().entry(method).or_default() += 1;
        let mut hook = self.hook.lock().unwrap();
        if hook.as_ref().is_some_and(|(before, _)| *before == method) {
            let (_, run) = hook.take().unwrap();
            drop(hook);
            run();
        }
    }

    /// Run `run` once, just before the next call of `method`.
    fn before(&self, method: &'static str, run: impl FnOnce() + Send + 'static) {
        *self.hook.lock().unwrap() = Some((method, Box::new(run)));
    }

    /// The calls made since the last `take`, cleared.
    fn take(&self) -> BTreeMap<&'static str, usize> {
        std::mem::take(&mut *self.calls.lock().unwrap())
    }
}

impl MetadataProvider for CountingProvider {
    fn get_current_snapshot(&self) -> Result<i64> {
        self.count("get_current_snapshot");
        self.inner.get_current_snapshot()
    }
    fn get_data_path(&self) -> Result<String> {
        self.count("get_data_path");
        self.inner.get_data_path()
    }
    fn get_metadata_settings(
        &self,
        schema_id: Option<i64>,
        table_id: Option<i64>,
    ) -> Result<std::collections::HashMap<String, String>> {
        self.count("get_metadata_settings");
        self.inner.get_metadata_settings(schema_id, table_id)
    }
    fn list_snapshots(&self) -> Result<Vec<SnapshotMetadata>> {
        self.count("list_snapshots");
        self.inner.list_snapshots()
    }
    fn list_snapshot_changes(&self) -> Result<Vec<SnapshotChangeMetadata>> {
        self.count("list_snapshot_changes");
        self.inner.list_snapshot_changes()
    }
    fn find_snapshot_by_commit_extra_info(&self, needle: &str) -> Result<Option<i64>> {
        self.count("find_snapshot_by_commit_extra_info");
        self.inner.find_snapshot_by_commit_extra_info(needle)
    }
    fn list_schemas(&self, snapshot_id: i64) -> Result<Vec<SchemaMetadata>> {
        self.count("list_schemas");
        self.inner.list_schemas(snapshot_id)
    }
    fn list_tables(&self, schema_id: i64, snapshot_id: i64) -> Result<Vec<TableMetadata>> {
        self.count("list_tables");
        self.inner.list_tables(schema_id, snapshot_id)
    }
    fn list_views(&self, schema_id: i64, snapshot_id: i64) -> Result<Vec<ViewMetadata>> {
        self.count("list_views");
        self.inner.list_views(schema_id, snapshot_id)
    }
    fn get_table_structure(
        &self,
        table_id: i64,
        snapshot_id: i64,
    ) -> Result<Vec<DuckLakeTableColumn>> {
        self.count("get_table_structure");
        self.inner.get_table_structure(table_id, snapshot_id)
    }
    fn get_table_fields(&self, table_id: i64, snapshot_id: i64) -> Result<Vec<DuckLakeTableField>> {
        self.count("get_table_fields");
        self.inner.get_table_fields(table_id, snapshot_id)
    }
    fn get_name_mapping(&self, mapping_id: i64) -> Result<DuckLakeNameMapping> {
        self.count("get_name_mapping");
        self.inner.get_name_mapping(mapping_id)
    }
    fn get_table_files_for_select(
        &self,
        table_id: i64,
        snapshot_id: i64,
    ) -> Result<Vec<DuckLakeTableFile>> {
        self.count("get_table_files_for_select");
        self.inner.get_table_files_for_select(table_id, snapshot_id)
    }
    fn get_partition_spec(
        &self,
        table_id: i64,
        snapshot_id: i64,
    ) -> Result<Option<datafusion_ducklake::PartitionSpec>> {
        self.count("get_partition_spec");
        self.inner.get_partition_spec(table_id, snapshot_id)
    }
    fn get_sort_spec(
        &self,
        table_id: i64,
        snapshot_id: i64,
    ) -> Result<Option<datafusion_ducklake::SortSpec>> {
        self.count("get_sort_spec");
        self.inner.get_sort_spec(table_id, snapshot_id)
    }
    fn get_table_statistics(&self, table_id: i64, snapshot_id: i64) -> Result<DuckLakeStatistics> {
        self.count("get_table_statistics");
        self.inner.get_table_statistics(table_id, snapshot_id)
    }
    fn get_table_summary_statistics(
        &self,
        table_id: i64,
        snapshot_id: i64,
    ) -> Result<DuckLakeStatistics> {
        self.count("get_table_summary_statistics");
        self.inner
            .get_table_summary_statistics(table_id, snapshot_id)
    }
    fn get_table_file_metadata_page(
        &self,
        table_id: i64,
        snapshot_id: i64,
        after_data_file_id: Option<i64>,
        limit: usize,
    ) -> Result<Vec<DuckLakeFileMetadata>> {
        self.count("get_table_file_metadata_page");
        self.inner
            .get_table_file_metadata_page(table_id, snapshot_id, after_data_file_id, limit)
    }
    fn get_table_file_metadata_page_filtered(
        &self,
        table_id: i64,
        snapshot_id: i64,
        after_data_file_id: Option<i64>,
        limit: usize,
        filter: Option<&StatsFilter>,
    ) -> Result<Vec<DuckLakeFileMetadata>> {
        // Counted apart when the catalog-side statistics filter is passed.
        self.count(if filter.is_some() {
            "get_table_file_metadata_page_filtered(filter)"
        } else {
            "get_table_file_metadata_page_filtered"
        });
        self.inner.get_table_file_metadata_page_filtered(
            table_id,
            snapshot_id,
            after_data_file_id,
            limit,
            filter,
        )
    }
    fn get_inlined_data(
        &self,
        table_id: i64,
        snapshot_id: i64,
        columns: &[DuckLakeTableColumn],
    ) -> Result<Vec<RecordBatch>> {
        self.count("get_inlined_data");
        self.inner.get_inlined_data(table_id, snapshot_id, columns)
    }
    fn scan_inlined_data(
        &self,
        table_id: i64,
        snapshot_id: i64,
        columns: &[DuckLakeTableColumn],
        filter: Option<&InlinedFilter>,
    ) -> Result<InlinedDataScan> {
        self.count("scan_inlined_data");
        self.inner
            .scan_inlined_data(table_id, snapshot_id, columns, filter)
    }
    fn get_inlined_deletes(
        &self,
        table_id: i64,
        snapshot_id: i64,
    ) -> Result<Vec<DuckLakeInlinedDelete>> {
        self.count("get_inlined_deletes");
        self.inner.get_inlined_deletes(table_id, snapshot_id)
    }
    fn get_inlined_data_with_row_ids(
        &self,
        table_id: i64,
        snapshot_id: i64,
        columns: &[DuckLakeTableColumn],
    ) -> Result<Vec<DuckLakeInlinedData>> {
        self.count("get_inlined_data_with_row_ids");
        self.inner
            .get_inlined_data_with_row_ids(table_id, snapshot_id, columns)
    }
    fn get_table_row_count(&self, table_id: i64, snapshot_id: i64) -> Result<u64> {
        self.count("get_table_row_count");
        self.inner.get_table_row_count(table_id, snapshot_id)
    }
    fn get_schema_by_name(&self, name: &str, snapshot_id: i64) -> Result<Option<SchemaMetadata>> {
        self.count("get_schema_by_name");
        self.inner.get_schema_by_name(name, snapshot_id)
    }
    fn get_table_by_name(
        &self,
        schema_id: i64,
        name: &str,
        snapshot_id: i64,
    ) -> Result<Option<TableMetadata>> {
        self.count("get_table_by_name");
        self.inner.get_table_by_name(schema_id, name, snapshot_id)
    }
    fn get_view_by_name(
        &self,
        schema_id: i64,
        name: &str,
        snapshot_id: i64,
    ) -> Result<Option<ViewMetadata>> {
        self.count("get_view_by_name");
        self.inner.get_view_by_name(schema_id, name, snapshot_id)
    }
    fn table_exists(&self, schema_id: i64, name: &str, snapshot_id: i64) -> Result<bool> {
        self.count("table_exists");
        self.inner.table_exists(schema_id, name, snapshot_id)
    }
    fn get_tags(&self, target: TagTarget, snapshot_id: i64) -> Result<Vec<DuckLakeTag>> {
        self.count("get_tags");
        self.inner.get_tags(target, snapshot_id)
    }
    fn get_view_id_by_name(
        &self,
        schema_id: i64,
        name: &str,
        snapshot_id: i64,
    ) -> Result<Option<i64>> {
        self.count("get_view_id_by_name");
        self.inner.get_view_id_by_name(schema_id, name, snapshot_id)
    }
    fn list_all_tables(&self, snapshot_id: i64) -> Result<Vec<TableWithSchema>> {
        self.count("list_all_tables");
        self.inner.list_all_tables(snapshot_id)
    }
    fn list_all_views(&self, snapshot_id: i64) -> Result<Vec<ViewWithSchema>> {
        self.count("list_all_views");
        self.inner.list_all_views(snapshot_id)
    }
    fn list_all_columns(&self, snapshot_id: i64) -> Result<Vec<ColumnWithTable>> {
        self.count("list_all_columns");
        self.inner.list_all_columns(snapshot_id)
    }
    fn list_all_object_tags(&self, snapshot_id: i64) -> Result<Vec<ObjectTag>> {
        self.count("list_all_object_tags");
        self.inner.list_all_object_tags(snapshot_id)
    }
    fn list_all_column_tags(&self, snapshot_id: i64) -> Result<Vec<ColumnTag>> {
        self.count("list_all_column_tags");
        self.inner.list_all_column_tags(snapshot_id)
    }
    fn list_all_files(&self, snapshot_id: i64) -> Result<Vec<FileWithTable>> {
        self.count("list_all_files");
        self.inner.list_all_files(snapshot_id)
    }
    fn get_data_files_added_between_snapshots(
        &self,
        table_id: i64,
        start_snapshot: i64,
        end_snapshot: i64,
    ) -> Result<Vec<DataFileChange>> {
        self.count("get_data_files_added_between_snapshots");
        self.inner
            .get_data_files_added_between_snapshots(table_id, start_snapshot, end_snapshot)
    }
    fn get_delete_files_added_between_snapshots(
        &self,
        table_id: i64,
        start_snapshot: i64,
        end_snapshot: i64,
    ) -> Result<Vec<DeleteFileChange>> {
        self.count("get_delete_files_added_between_snapshots");
        self.inner
            .get_delete_files_added_between_snapshots(table_id, start_snapshot, end_snapshot)
    }
}

/// An object store over the local filesystem that counts the reads of each
/// path, so a test can see which files a scan opened.
#[derive(Debug)]
struct ReadCountingStore {
    inner: Arc<dyn object_store::ObjectStore>,
    reads: Mutex<BTreeMap<String, usize>>,
}

impl ReadCountingStore {
    fn new() -> Self {
        Self {
            inner: Arc::new(object_store::local::LocalFileSystem::new()),
            reads: Mutex::new(BTreeMap::new()),
        }
    }

    fn record(&self, location: &object_store::path::Path) {
        *self
            .reads
            .lock()
            .unwrap()
            .entry(location.to_string())
            .or_default() += 1;
    }

    /// Reads of paths containing `needle` since the last `take`, then clear.
    fn take_reads_of(&self, needle: &str) -> usize {
        std::mem::take(&mut *self.reads.lock().unwrap())
            .into_iter()
            .filter(|(path, _)| path.contains(needle))
            .map(|(_, reads)| reads)
            .sum()
    }
}

impl std::fmt::Display for ReadCountingStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ReadCountingStore({})", self.inner)
    }
}

#[async_trait::async_trait]
impl object_store::ObjectStore for ReadCountingStore {
    async fn put_opts(
        &self,
        location: &object_store::path::Path,
        payload: object_store::PutPayload,
        opts: object_store::PutOptions,
    ) -> object_store::Result<object_store::PutResult> {
        self.inner.put_opts(location, payload, opts).await
    }

    async fn put_multipart_opts(
        &self,
        location: &object_store::path::Path,
        opts: object_store::PutMultipartOptions,
    ) -> object_store::Result<Box<dyn object_store::MultipartUpload>> {
        self.inner.put_multipart_opts(location, opts).await
    }

    async fn get_opts(
        &self,
        location: &object_store::path::Path,
        options: object_store::GetOptions,
    ) -> object_store::Result<object_store::GetResult> {
        self.record(location);
        self.inner.get_opts(location, options).await
    }

    async fn get_ranges(
        &self,
        location: &object_store::path::Path,
        ranges: &[std::ops::Range<u64>],
    ) -> object_store::Result<Vec<bytes::Bytes>> {
        self.record(location);
        self.inner.get_ranges(location, ranges).await
    }

    fn delete_stream(
        &self,
        locations: futures::stream::BoxStream<
            'static,
            object_store::Result<object_store::path::Path>,
        >,
    ) -> futures::stream::BoxStream<'static, object_store::Result<object_store::path::Path>> {
        self.inner.delete_stream(locations)
    }

    fn list(
        &self,
        prefix: Option<&object_store::path::Path>,
    ) -> futures::stream::BoxStream<'static, object_store::Result<object_store::ObjectMeta>> {
        self.inner.list(prefix)
    }

    async fn list_with_delimiter(
        &self,
        prefix: Option<&object_store::path::Path>,
    ) -> object_store::Result<object_store::ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }

    async fn copy_opts(
        &self,
        from: &object_store::path::Path,
        to: &object_store::path::Path,
        options: object_store::CopyOptions,
    ) -> object_store::Result<()> {
        self.inner.copy_opts(from, to, options).await
    }
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// A DuckLake catalog in a temporary directory, its metadata in SQLite and its
/// data in Parquet, written by official DuckLake.
struct Lake {
    dir: TempDir,
}

impl Lake {
    /// A catalog that `statements` set up, with inserts of at most
    /// `inlining_rows` rows inlined into the catalog.
    fn new(inlining_rows: usize, statements: &[&str]) -> anyhow::Result<Self> {
        let lake = Self {
            dir: TempDir::new()?,
        };
        std::fs::create_dir_all(lake.data_path())?;
        lake.run(inlining_rows, statements)?;
        Ok(lake)
    }

    fn catalog_path(&self) -> PathBuf {
        self.dir.path().join("catalog.db")
    }

    fn data_path(&self) -> PathBuf {
        self.dir.path().join("data")
    }

    /// Run `statements` through DuckDB with the catalog attached as `lake`.
    fn run(&self, inlining_rows: usize, statements: &[&str]) -> anyhow::Result<()> {
        run_duckdb(
            &self.catalog_path(),
            &self.data_path(),
            inlining_rows,
            statements,
        )
    }

    /// [`Self::run`], as a closure to hand a [`CountingProvider`] hook.
    fn runner(&self, statements: &'static [&'static str]) -> impl FnOnce() + Send + 'static {
        let (catalog, data) = (self.catalog_path(), self.data_path());
        move || run_duckdb(&catalog, &data, 0, statements).expect("DuckDB statements")
    }

    async fn provider(&self) -> anyhow::Result<CountingProvider> {
        let url = format!("sqlite:{}", self.catalog_path().display());
        Ok(CountingProvider::new(Arc::new(
            SqliteMetadataProvider::new(&url).await?,
        )))
    }
}

fn run_duckdb(
    catalog: &std::path::Path,
    data: &std::path::Path,
    inlining_rows: usize,
    statements: &[&str],
) -> anyhow::Result<()> {
    common::ensure_ducklake_installed();
    common::ensure_extension_installed("sqlite");
    let conn = duckdb::Connection::open_in_memory()?;
    conn.execute("LOAD sqlite", [])?;
    conn.execute("LOAD ducklake", [])?;
    conn.execute(
        &format!(
            "ATTACH 'ducklake:sqlite:{}' AS lake \
             (DATA_PATH '{}', DATA_INLINING_ROW_LIMIT {inlining_rows})",
            catalog.display(),
            data.display()
        ),
        [],
    )?;
    for statement in statements {
        conn.execute(statement, [])?;
    }
    Ok(())
}

/// Table `name` of the catalog behind `counting`, looked up once and registered
/// in a fresh session under the same name, the way a caller that caches built
/// tables holds one. The calls the lookup made are cleared.
async fn cached(
    counting: &CountingProvider,
    options: DuckLakeReadOptions,
    row_lineage: bool,
    name: &str,
) -> anyhow::Result<(SessionContext, Arc<dyn TableProvider>)> {
    let catalog = DuckLakeCatalog::new(counting.clone())?
        .with_read_options(options)
        .with_row_lineage(row_lineage);
    let table = catalog
        .schema("main")
        .expect("schema main")
        .table(name)
        .await?
        .unwrap_or_else(|| panic!("table {name}"));
    let ctx = SessionContext::new();
    ctx.register_table(name, Arc::clone(&table))?;
    counting.take();
    Ok((ctx, table))
}

/// [`cached`] for table `t`.
async fn cached_table(
    counting: &CountingProvider,
    options: DuckLakeReadOptions,
    row_lineage: bool,
) -> anyhow::Result<(SessionContext, Arc<dyn TableProvider>)> {
    cached(counting, options, row_lineage, "t").await
}

async fn query(ctx: &SessionContext, sql: &str) -> anyhow::Result<String> {
    let batches = ctx.sql(sql).await?.collect().await?;
    Ok(pretty_format_batches(&batches)?.to_string())
}

/// `t`'s id and the snapshot the catalog is at.
fn table_at_head(provider: &dyn MetadataProvider) -> anyhow::Result<(i64, i64)> {
    let snapshot = provider.get_current_snapshot()?;
    let schema = provider
        .get_schema_by_name("main", snapshot)?
        .expect("schema main");
    let table = provider
        .get_table_by_name(schema.schema_id, "t", snapshot)?
        .expect("table t");
    Ok((table.table_id, snapshot))
}

/// The calls the first scan of a memoized table makes: the catalog head before
/// and after its reads, and each read once.
fn fill_calls() -> BTreeMap<&'static str, usize> {
    BTreeMap::from([
        ("get_current_snapshot", 2),
        ("get_inlined_deletes", 1),
        ("get_table_file_metadata_page_filtered", 1),
        ("scan_inlined_data", 1),
    ])
}

/// Five rows in one data file, two of them deleted by a delete file.
const WITH_DELETES: &[&str] = &[
    "CREATE TABLE lake.t (id INT, name VARCHAR)",
    "INSERT INTO lake.t VALUES (1, 'a'), (2, 'b'), (3, 'c'), (4, 'd'), (5, 'e')",
    "DELETE FROM lake.t WHERE id IN (2, 4)",
];

/// Three data files, one per insert.
const THREE_FILES: &[&str] = &[
    "CREATE TABLE lake.t (id INT)",
    "INSERT INTO lake.t VALUES (1), (2)",
    "INSERT INTO lake.t VALUES (10), (11)",
    "INSERT INTO lake.t VALUES (20), (21)",
];

/// A Parquet data file of twenty rows, two of its rows deleted by deletions
/// inlined into the catalog, and two more rows inlined into the catalog.
const INLINED: &[&str] = &[
    "CREATE TABLE lake.t (id INT)",
    "INSERT INTO lake.t SELECT i FROM range(20) r(i)",
    "DELETE FROM lake.t WHERE id IN (3, 7)",
    "INSERT INTO lake.t VALUES (100), (101)",
];

// ---------------------------------------------------------------------------
// Catalog memo
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn without_a_memo_each_scan_reads_the_catalog_again() -> anyhow::Result<()> {
    let lake = Lake::new(0, WITH_DELETES)?;
    let counting = lake.provider().await?;
    let (ctx, _) = cached_table(&counting, DuckLakeReadOptions::default(), false).await?;

    for scan in 0..2 {
        query(&ctx, "SELECT * FROM t ORDER BY id").await?;
        let calls = counting.take();
        for method in [
            "get_table_file_metadata_page_filtered",
            "get_inlined_deletes",
            "scan_inlined_data",
            "get_current_snapshot",
        ] {
            assert_eq!(
                calls.get(method),
                Some(&1),
                "scan {scan}, {method}: {calls:?}"
            );
        }
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn a_memoized_table_reads_the_catalog_on_its_first_scan_only() -> anyhow::Result<()> {
    let lake = Lake::new(0, WITH_DELETES)?;
    let plain = lake.provider().await?;
    let (plain_ctx, _) = cached_table(&plain, DuckLakeReadOptions::default(), false).await?;
    let counting = lake.provider().await?;
    let (ctx, _) = cached_table(&counting, DuckLakeReadOptions::memoized(), false).await?;

    let queries = [
        "SELECT * FROM t ORDER BY id",
        "SELECT name FROM t WHERE id > 2 ORDER BY id",
        "SELECT count(*) FROM t",
        // The plan is the same too, so `count(*)` is still answered from the
        // catalog's counts rather than by reading the data.
        "EXPLAIN SELECT count(*) FROM t",
        "SELECT * FROM t ORDER BY id",
    ];
    for (scan, sql) in queries.into_iter().enumerate() {
        assert_eq!(
            query(&ctx, sql).await?,
            query(&plain_ctx, sql).await?,
            "{sql}"
        );
        let calls = counting.take();
        if scan == 0 {
            assert_eq!(calls, fill_calls());
        } else {
            assert!(
                calls.is_empty(),
                "scan {scan} ({sql}) read the catalog: {calls:?}"
            );
        }
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn a_filtered_first_scan_memoizes_the_whole_listing() -> anyhow::Result<()> {
    let lake = Lake::new(0, THREE_FILES)?;
    let filtered = "SELECT id FROM t WHERE id >= 20 ORDER BY id";
    let all = "SELECT id FROM t ORDER BY id";

    // Without a memo, the filter goes to the catalog with the listing.
    let plain = lake.provider().await?;
    let (plain_ctx, _) = cached_table(&plain, DuckLakeReadOptions::default(), false).await?;
    let expected_filtered = query(&plain_ctx, filtered).await?;
    assert_eq!(
        plain
            .take()
            .get("get_table_file_metadata_page_filtered(filter)"),
        Some(&1)
    );

    // With one, the first scan lists every file, without the filter.
    let counting = lake.provider().await?;
    let (ctx, _) = cached_table(&counting, DuckLakeReadOptions::memoized(), false).await?;
    assert_eq!(query(&ctx, filtered).await?, expected_filtered);
    assert_eq!(counting.take(), fill_calls());

    // So the memo serves an unfiltered scan too.
    assert_eq!(query(&ctx, all).await?, query(&plain_ctx, all).await?);
    assert!(counting.take().is_empty());
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn a_table_over_the_memo_budget_reads_the_catalog_on_every_scan() -> anyhow::Result<()> {
    let lake = Lake::new(0, THREE_FILES)?;
    let all = "SELECT id FROM t ORDER BY id";
    let plain = lake.provider().await?;
    let (plain_ctx, _) = cached_table(&plain, DuckLakeReadOptions::default(), false).await?;
    let expected = query(&plain_ctx, all).await?;

    let counting = lake.provider().await?;
    let options = DuckLakeReadOptions::memoized().with_catalog_memo(1);
    let (ctx, _) = cached_table(&counting, options, false).await?;
    // The first scan lists the files for the memo, finds they do not fit, and
    // lists them again for itself. Later scans do not try again.
    for listings in [2, 1] {
        assert_eq!(query(&ctx, all).await?, expected);
        let calls = counting.take();
        assert_eq!(
            calls.get("get_table_file_metadata_page_filtered"),
            Some(&listings),
            "{calls:?}"
        );
        assert_eq!(calls.get("scan_inlined_data"), Some(&1), "{calls:?}");
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn inlined_rows_and_deletions_come_from_the_memo() -> anyhow::Result<()> {
    let lake = Lake::new(10, INLINED)?;
    let plain = lake.provider().await?;
    let (table_id, snapshot) = table_at_head(&plain)?;
    assert_eq!(
        plain.get_inlined_deletes(table_id, snapshot)?.len(),
        2,
        "the fixture inlines its deletions"
    );
    let (plain_ctx, _) = cached_table(&plain, DuckLakeReadOptions::default(), false).await?;
    let counting = lake.provider().await?;
    let (ctx, table) = cached_table(&counting, DuckLakeReadOptions::memoized(), false).await?;
    let table = table
        .downcast_ref::<datafusion_ducklake::DuckLakeTable>()
        .expect("a DuckLake table");

    let queries = [
        "SELECT id FROM t ORDER BY id",
        "SELECT id FROM t WHERE id >= 100 ORDER BY id",
        "SELECT id FROM t WHERE id IN (3, 7, 8) ORDER BY id",
        "SELECT count(*) FROM t",
    ];
    for (scan, sql) in queries.into_iter().enumerate() {
        assert_eq!(
            query(&ctx, sql).await?,
            query(&plain_ctx, sql).await?,
            "{sql}"
        );
        let calls = counting.take();
        assert_eq!(calls.is_empty(), scan > 0, "scan {scan} ({sql}): {calls:?}");
        // The fill reads every visible inlined row, unfiltered, and a scan the
        // memo serves reads none, so the count stays the fill's.
        assert_eq!(table.inlined_materialized_row_count(), 2, "scan {scan}");
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn a_row_lineage_scan_uses_the_memo() -> anyhow::Result<()> {
    let lake = Lake::new(0, WITH_DELETES)?;
    let plain = lake.provider().await?;
    let (plain_ctx, _) = cached_table(&plain, DuckLakeReadOptions::default(), true).await?;
    let counting = lake.provider().await?;
    let (ctx, _) = cached_table(&counting, DuckLakeReadOptions::memoized(), true).await?;

    let sql = "SELECT rowid, id, name FROM t ORDER BY rowid";
    for scan in 0..3 {
        assert_eq!(query(&ctx, sql).await?, query(&plain_ctx, sql).await?);
        let calls = counting.take();
        assert_eq!(calls.is_empty(), scan > 0, "scan {scan}: {calls:?}");
    }
    Ok(())
}

/// A row-lineage scan refuses a table with inlined rows, but only when an
/// inlined row passes the scan's filters. The memo keeps the inlined rows
/// unfiltered, so it must apply the filters before refusing.
#[tokio::test(flavor = "multi_thread")]
async fn a_row_lineage_scan_refuses_only_inlined_rows_its_filters_keep() -> anyhow::Result<()> {
    let lake = Lake::new(10, INLINED)?;
    let plain = lake.provider().await?;
    let (plain_ctx, _) = cached_table(&plain, DuckLakeReadOptions::default(), true).await?;
    let counting = lake.provider().await?;
    let (ctx, _) = cached_table(&counting, DuckLakeReadOptions::memoized(), true).await?;

    let refused = "row-lineage (rowid) scan on a table with inlined rows is not supported";
    for (sql, refuses) in [
        // No inlined row has an id below 50.
        (
            "SELECT rowid, id FROM t WHERE id < 50 ORDER BY rowid",
            false,
        ),
        (
            "SELECT rowid, id FROM t WHERE id >= 100 ORDER BY rowid",
            true,
        ),
        ("SELECT rowid, id FROM t ORDER BY rowid", true),
    ] {
        let memoized = query(&ctx, sql).await;
        let unmemoized = query(&plain_ctx, sql).await;
        match (memoized, unmemoized) {
            (Ok(memoized), Ok(unmemoized)) if !refuses => assert_eq!(memoized, unmemoized),
            (Err(memoized), Err(unmemoized)) if refuses => {
                assert!(memoized.to_string().contains(refused), "{memoized}");
                assert!(unmemoized.to_string().contains(refused), "{unmemoized}");
            },
            (memoized, unmemoized) => {
                panic!("{sql}: memoized {memoized:?}, unmemoized {unmemoized:?}")
            },
        }
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn concurrent_first_scans_agree_and_keep_one_memo() -> anyhow::Result<()> {
    let lake = Lake::new(0, WITH_DELETES)?;
    let all = "SELECT * FROM t ORDER BY id";
    let plain = lake.provider().await?;
    let (plain_ctx, _) = cached_table(&plain, DuckLakeReadOptions::default(), false).await?;
    let expected = query(&plain_ctx, all).await?;

    let counting = lake.provider().await?;
    let (ctx, _) = cached_table(&counting, DuckLakeReadOptions::memoized(), false).await?;
    let (first, second) = tokio::join!(query(&ctx, all), query(&ctx, all));
    assert_eq!(first?, expected);
    assert_eq!(second?, expected);
    counting.take();
    assert_eq!(query(&ctx, all).await?, expected);
    assert!(counting.take().is_empty());
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn a_memoized_table_keeps_reading_its_snapshot() -> anyhow::Result<()> {
    let lake = Lake::new(0, THREE_FILES)?;
    let all = "SELECT id FROM t ORDER BY id";
    let counting = lake.provider().await?;
    let (ctx, _) = cached_table(&counting, DuckLakeReadOptions::memoized(), false).await?;
    let before = query(&ctx, all).await?;

    lake.run(0, &["INSERT INTO lake.t VALUES (30)"])?;

    // The table is bound to its snapshot, memo or not.
    counting.take();
    assert_eq!(query(&ctx, all).await?, before);
    assert!(counting.take().is_empty());
    // A new lookup reads the new snapshot.
    let (fresh, _) = cached_table(&counting, DuckLakeReadOptions::memoized(), false).await?;
    assert!(query(&fresh, all).await?.contains("| 30 |"));
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn a_merge_after_the_first_scan_leaves_the_memoized_rows_unchanged() -> anyhow::Result<()> {
    let lake = Lake::new(0, THREE_FILES)?;
    let all = "SELECT id FROM t ORDER BY id";
    let counting = lake.provider().await?;
    let (table_id, snapshot) = table_at_head(&counting)?;
    let (ctx, _) = cached_table(&counting, DuckLakeReadOptions::memoized(), false).await?;
    let before = query(&ctx, all).await?;
    assert_eq!(
        counting
            .get_table_files_for_select(table_id, snapshot)?
            .len(),
        3
    );

    lake.run(0, &["CALL ducklake_merge_adjacent_files('lake')"])?;

    // The merge changed how the table's snapshot is stored: one merged file
    // now stands for the three.
    assert_eq!(
        counting
            .get_table_files_for_select(table_id, snapshot)?
            .len(),
        1
    );
    counting.take();
    // The memo still lists the three, which are scheduled for deletion but not
    // yet deleted, and they hold the same rows.
    assert_eq!(query(&ctx, all).await?, before);
    assert!(counting.take().is_empty());
    // A table built at the same snapshot now reads the merged file.
    let fixed = DuckLakeCatalog::with_snapshot(Arc::new(lake.provider().await?), snapshot)?;
    let table = fixed.schema("main").unwrap().table("t").await?.unwrap();
    let fixed_ctx = SessionContext::new();
    fixed_ctx.register_table("t", table)?;
    assert_eq!(query(&fixed_ctx, all).await?, before);

    // Once cleanup deletes the replaced files, the memoized table can no
    // longer read them: this is why `DuckLakeReadOptions` says to drop a
    // memoized table before cleanup's grace period passes.
    lake.run(
        0,
        &["CALL ducklake_cleanup_old_files('lake', cleanup_all => true)"],
    )?;
    assert!(query(&ctx, all).await.is_err());
    assert_eq!(query(&fixed_ctx, all).await?, before);
    Ok(())
}

/// A merge that commits while the first scan fills the memo changes what the
/// fill reads partway through. Such a fill serves its scan but is not kept, so
/// the next scan fills the memo again, from the merged layout.
#[tokio::test(flavor = "multi_thread")]
async fn a_commit_during_the_fill_is_not_kept() -> anyhow::Result<()> {
    let lake = Lake::new(0, THREE_FILES)?;
    let all = "SELECT id FROM t ORDER BY id";
    let counting = lake.provider().await?;
    let (ctx, _) = cached_table(&counting, DuckLakeReadOptions::memoized(), false).await?;

    // Between the listing and the inlined reads.
    counting.before(
        "get_inlined_deletes",
        lake.runner(&["CALL ducklake_merge_adjacent_files('lake')"]),
    );
    let before = query(&ctx, all).await?;
    assert_eq!(counting.take(), fill_calls());

    // Not kept: the second scan fills the memo again, and now keeps it.
    assert_eq!(query(&ctx, all).await?, before);
    assert_eq!(counting.take(), fill_calls());
    assert_eq!(query(&ctx, all).await?, before);
    assert!(counting.take().is_empty());

    // The memo holds the merged layout, so cleanup takes nothing it reads.
    lake.run(
        0,
        &["CALL ducklake_cleanup_old_files('lake', cleanup_all => true)"],
    )?;
    assert_eq!(query(&ctx, all).await?, before);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn a_table_behind_a_view_keeps_the_memo() -> anyhow::Result<()> {
    let lake = Lake::new(
        0,
        &[WITH_DELETES, &["CREATE VIEW lake.v AS SELECT id FROM lake.t"]].concat(),
    )?;
    let counting = lake.provider().await?;
    let (ctx, _) = cached(&counting, DuckLakeReadOptions::memoized(), false, "v").await?;

    let sql = "SELECT id FROM v ORDER BY id";
    let expected = query(&ctx, sql).await?;
    assert!(expected.contains("| 5  |"), "{expected}");
    counting.take();
    assert_eq!(query(&ctx, sql).await?, expected);
    assert!(counting.take().is_empty());
    Ok(())
}

/// The memo serves `scan` alone. A mutation lists the files it changes
/// afresh, so it never acts on a layout the catalog has since replaced.
#[tokio::test(flavor = "multi_thread")]
async fn file_listings_for_mutations_read_the_catalog() -> anyhow::Result<()> {
    let lake = Lake::new(0, WITH_DELETES)?;
    let counting = lake.provider().await?;
    let (ctx, table) = cached_table(&counting, DuckLakeReadOptions::memoized(), false).await?;
    let table = table
        .downcast_ref::<datafusion_ducklake::DuckLakeTable>()
        .expect("a DuckLake table");
    query(&ctx, "SELECT * FROM t").await?;
    counting.take();

    assert_eq!(table.files()?.len(), 1);
    assert_eq!(counting.take().get("get_table_files_for_select"), Some(&1));
    let always: Arc<dyn datafusion::physical_expr::PhysicalExpr> = Arc::new(
        datafusion::physical_expr::expressions::Literal::new(true.into()),
    );
    assert_eq!(table.files_matching(&always)?.len(), 1);
    assert!(
        counting
            .take()
            .contains_key("get_table_file_metadata_page_filtered"),
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Delete memo
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn the_delete_memo_reads_each_delete_file_once() -> anyhow::Result<()> {
    let lake = Lake::new(0, WITH_DELETES)?;
    let all = "SELECT * FROM t ORDER BY id";
    for (options, read_again) in [
        (DuckLakeReadOptions::memoized(), false),
        (DuckLakeReadOptions::memoized().with_delete_memo(0), true),
    ] {
        let counting = lake.provider().await?;
        let (ctx, _) = cached_table(&counting, options.clone(), false).await?;
        let store = Arc::new(ReadCountingStore::new());
        ctx.runtime_env().register_object_store(
            &url::Url::parse("file:///")?,
            Arc::clone(&store) as Arc<dyn object_store::ObjectStore>,
        );

        let expected = query(&ctx, all).await?;
        assert!(store.take_reads_of("-delete.parquet") > 0, "{options:?}");
        assert_eq!(query(&ctx, all).await?, expected);
        assert_eq!(
            store.take_reads_of("-delete.parquet") > 0,
            read_again,
            "{options:?}"
        );
    }
    Ok(())
}

/// With the catalog memo off, every scan lists the files again. A second
/// delete from a data file replaces its delete file, even for older snapshots,
/// so a pinned table's later scan lists a new path. The delete memo reads the
/// new file once and keeps it in place of the old one.
#[tokio::test(flavor = "multi_thread")]
async fn the_delete_memo_follows_a_replaced_delete_file() -> anyhow::Result<()> {
    let lake = Lake::new(0, WITH_DELETES)?;
    let all = "SELECT id FROM t ORDER BY id";
    let counting = lake.provider().await?;
    let (table_id, snapshot) = table_at_head(&counting)?;
    let delete_path = || -> anyhow::Result<Option<String>> {
        Ok(counting.get_table_files_for_select(table_id, snapshot)?[0]
            .delete_file
            .as_ref()
            .map(|file| file.path.clone()))
    };
    let options = DuckLakeReadOptions::default().with_delete_memo(1 << 20);
    let fixed = DuckLakeCatalog::with_snapshot(Arc::new(counting.clone()), snapshot)?
        .with_read_options(options);
    let table = fixed.schema("main").unwrap().table("t").await?.unwrap();
    let ctx = SessionContext::new();
    ctx.register_table("t", table)?;
    let store = Arc::new(ReadCountingStore::new());
    ctx.runtime_env().register_object_store(
        &url::Url::parse("file:///")?,
        Arc::clone(&store) as Arc<dyn object_store::ObjectStore>,
    );
    let before = query(&ctx, all).await?;
    assert!(store.take_reads_of("-delete.parquet") > 0);

    // Official writes the new delete file with both deletions and their
    // snapshots, and lists it for the older snapshot in place of the first.
    let first_path = delete_path()?;
    lake.run(0, &["DELETE FROM lake.t WHERE id = 5"])?;
    assert_ne!(delete_path()?, first_path);

    // The table still reads its snapshot, where the later deletion does not
    // apply. Its next scan reads the new file; the scan after that reads none.
    assert_eq!(query(&ctx, all).await?, before);
    assert!(store.take_reads_of("-delete.parquet") > 0);
    assert_eq!(query(&ctx, all).await?, before);
    assert_eq!(store.take_reads_of("-delete.parquet"), 0);
    Ok(())
}
