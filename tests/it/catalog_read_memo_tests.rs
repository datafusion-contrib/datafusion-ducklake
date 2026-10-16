#![cfg(all(feature = "metadata-duckdb", feature = "metadata-sqlite"))]
//! What a kept `DuckLakeTable` reads from the catalog on each query, with and
//! without a `CatalogReadMemo`, and what it reads again from its delete files,
//! with and without the delete memo of `DuckLakeReadOptions` (#338).
//!
//! The catalogs are written by official DuckLake through DuckDB, with their
//! metadata in SQLite, or in PostgreSQL for the tests in [`postgres`], so a
//! test can commit while a table is kept. A [`CountingProvider`] under the
//! memo counts the catalog calls each query makes, and a [`ReadCountingStore`]
//! counts the reads of each file.

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use arrow::record_batch::RecordBatch;
use arrow::util::pretty::pretty_format_batches;
use datafusion::catalog::CatalogProvider;
use datafusion::prelude::SessionContext;
use datafusion_ducklake::inlined_filter::{InlinedDataScan, InlinedFilter};
use datafusion_ducklake::metadata_provider::{
    ColumnWithTable, DataFileChange, DeleteFileChange, DuckLakeFileMetadata, DuckLakeInlinedData,
    DuckLakeInlinedDelete, DuckLakeInlinedRows, DuckLakeNameMapping, DuckLakeStatistics,
    DuckLakeTableColumn, DuckLakeTableField, DuckLakeTableFile, FILE_METADATA_BATCH_SIZE,
    FileWithTable, InlinedFileDeleteChange, InlinedRowSelection, SchemaMetadata, SnapshotMetadata,
    TableMetadata, TableWithSchema, ViewMetadata, ViewWithSchema,
};
use datafusion_ducklake::stats_filter::StatsFilter;
use datafusion_ducklake::{
    CatalogReadMemo, ColumnTag, DuckLakeCatalog, DuckLakeInlinedFileDeleteForFlush,
    DuckLakeInlinedFlushData, DuckLakeReadOptions, DuckLakeTable, DuckLakeTag,
    MemoizedMetadataProvider, MetadataProvider, ObjectTag, Result, SnapshotChangeMetadata,
    SqliteMetadataProvider, TagTarget,
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
    ) -> Result<HashMap<String, String>> {
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
    fn scan_inlined_data_noting_absence(
        &self,
        table_id: i64,
        snapshot_id: i64,
        columns: &[DuckLakeTableColumn],
        filter: Option<&InlinedFilter>,
    ) -> Result<(InlinedDataScan, Option<i64>)> {
        self.count("scan_inlined_data_noting_absence");
        self.inner
            .scan_inlined_data_noting_absence(table_id, snapshot_id, columns, filter)
    }
    fn get_inlined_deletes(
        &self,
        table_id: i64,
        snapshot_id: i64,
    ) -> Result<Vec<DuckLakeInlinedDelete>> {
        self.count("get_inlined_deletes");
        self.inner.get_inlined_deletes(table_id, snapshot_id)
    }
    fn get_delete_file_begin_snapshot(&self, delete_file_id: i64) -> Result<Option<i64>> {
        self.count("get_delete_file_begin_snapshot");
        self.inner.get_delete_file_begin_snapshot(delete_file_id)
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
    fn get_inlined_data_for_flush(
        &self,
        table_id: i64,
        snapshot_id: i64,
    ) -> Result<Vec<DuckLakeInlinedFlushData>> {
        self.count("get_inlined_data_for_flush");
        self.inner.get_inlined_data_for_flush(table_id, snapshot_id)
    }
    fn get_inlined_file_deletes_for_flush(
        &self,
        table_id: i64,
        snapshot_id: i64,
    ) -> Result<Vec<DuckLakeInlinedFileDeleteForFlush>> {
        self.count("get_inlined_file_deletes_for_flush");
        self.inner
            .get_inlined_file_deletes_for_flush(table_id, snapshot_id)
    }
    fn read_inlined_rows(
        &self,
        table_id: i64,
        columns: &[DuckLakeTableColumn],
        selection: InlinedRowSelection,
        filter: Option<&InlinedFilter>,
    ) -> Result<Vec<DuckLakeInlinedRows>> {
        self.count("read_inlined_rows");
        self.inner
            .read_inlined_rows(table_id, columns, selection, filter)
    }
    fn get_inlined_file_deletes_between(
        &self,
        table_id: i64,
        start_snapshot: i64,
        end_snapshot: i64,
    ) -> Result<Vec<InlinedFileDeleteChange>> {
        self.count("get_inlined_file_deletes_between");
        self.inner
            .get_inlined_file_deletes_between(table_id, start_snapshot, end_snapshot)
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

/// A DuckLake catalog in a temporary directory, its data in Parquet and its
/// metadata in SQLite or PostgreSQL, written by official DuckLake.
struct Lake {
    dir: TempDir,
    metadata: Metadata,
}

/// Where a [`Lake`] keeps its metadata.
#[derive(Clone)]
enum Metadata {
    /// A SQLite file in the lake's directory.
    Sqlite,
    /// A database on a PostgreSQL server, by its sqlx URL and its libpq string.
    #[cfg(feature = "metadata-postgres")]
    Postgres {
        url: String,
        libpq: String,
    },
}

impl Lake {
    /// A catalog with SQLite metadata that `statements` set up, with inserts of
    /// at most `inlining_rows` rows inlined into the catalog.
    fn new(inlining_rows: usize, statements: &[&str]) -> anyhow::Result<Self> {
        Self::with_metadata(Metadata::Sqlite, inlining_rows, statements)
    }

    /// [`Self::new`], with its metadata in `metadata`.
    fn with_metadata(
        metadata: Metadata,
        inlining_rows: usize,
        statements: &[&str],
    ) -> anyhow::Result<Self> {
        let lake = Self {
            dir: TempDir::new()?,
            metadata,
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

    /// The DuckDB extension that reads the metadata, and the catalog's
    /// `ATTACH` path.
    fn attach(&self) -> (&'static str, String) {
        match &self.metadata {
            Metadata::Sqlite => (
                "sqlite",
                format!("ducklake:sqlite:{}", self.catalog_path().display()),
            ),
            #[cfg(feature = "metadata-postgres")]
            Metadata::Postgres {
                libpq,
                ..
            } => ("postgres", format!("ducklake:postgres:{libpq}")),
        }
    }

    /// Run `statements` through DuckDB with the catalog attached as `lake`.
    fn run(&self, inlining_rows: usize, statements: &[&str]) -> anyhow::Result<()> {
        let (extension, attach) = self.attach();
        run_duckdb(
            extension,
            &attach,
            &self.data_path(),
            inlining_rows,
            statements,
        )
    }

    /// [`Self::run`], as a closure to hand a [`CountingProvider`] hook.
    fn runner(&self, statements: &'static [&'static str]) -> impl FnOnce() + Send + 'static {
        let ((extension, attach), data) = (self.attach(), self.data_path());
        move || run_duckdb(extension, &attach, &data, 0, statements).expect("DuckDB statements")
    }

    async fn provider(&self) -> anyhow::Result<CountingProvider> {
        let inner: Arc<dyn MetadataProvider> = match &self.metadata {
            Metadata::Sqlite => {
                let url = format!("sqlite:{}", self.catalog_path().display());
                Arc::new(SqliteMetadataProvider::new(&url).await?)
            },
            #[cfg(feature = "metadata-postgres")]
            Metadata::Postgres {
                url,
                ..
            } => Arc::new(datafusion_ducklake::PostgresMetadataProvider::new(url).await?),
        };
        Ok(CountingProvider::new(inner))
    }
}

/// Makes the catalogs of one test: SQLite files, or databases on one
/// PostgreSQL server.
enum Backend {
    Sqlite,
    #[cfg(feature = "metadata-postgres")]
    Postgres(Box<postgres::Server>),
}

impl Backend {
    /// A new catalog that `statements` set up; see [`Lake::new`].
    async fn lake(&self, inlining_rows: usize, statements: &[&str]) -> anyhow::Result<Lake> {
        let metadata = match self {
            Self::Sqlite => Metadata::Sqlite,
            #[cfg(feature = "metadata-postgres")]
            Self::Postgres(server) => server.database().await?,
        };
        Lake::with_metadata(metadata, inlining_rows, statements)
    }
}

fn run_duckdb(
    extension: &str,
    attach: &str,
    data: &std::path::Path,
    inlining_rows: usize,
    statements: &[&str],
) -> anyhow::Result<()> {
    common::ensure_ducklake_installed();
    common::ensure_extension_installed(extension);
    let conn = duckdb::Connection::open_in_memory()?;
    conn.execute(&format!("LOAD {extension}"), [])?;
    conn.execute("LOAD ducklake", [])?;
    conn.execute(
        &format!(
            "ATTACH '{attach}' AS lake \
             (DATA_PATH '{}', DATA_INLINING_ROW_LIMIT {inlining_rows})",
            data.display()
        ),
        [],
    )?;
    for statement in statements {
        conn.execute_batch(statement)?;
    }
    Ok(())
}

/// A caller that builds a table once and keeps it across queries, the way a
/// cache keyed by table and snapshot does. Each query gets the kept table with
/// a provider of its own: the counting provider, bound to the memo once per
/// query when there is one.
struct Caller {
    counting: CountingProvider,
    memo: Option<Arc<CatalogReadMemo>>,
    table: DuckLakeTable,
}

impl Caller {
    /// Table `t` of `lake`, kept, without a memo.
    async fn plain(lake: &Lake) -> anyhow::Result<Self> {
        Self::new(lake, None, false).await
    }

    /// Table `t` of `lake`, kept, with a memo of the default budget.
    async fn memoized(lake: &Lake) -> anyhow::Result<Self> {
        Self::new(
            lake,
            Some(CatalogReadMemo::new(CatalogReadMemo::DEFAULT_MAX_BYTES)),
            false,
        )
        .await
    }

    async fn new(
        lake: &Lake,
        memo: Option<Arc<CatalogReadMemo>>,
        row_lineage: bool,
    ) -> anyhow::Result<Self> {
        let counting = lake.provider().await?;
        let catalog = DuckLakeCatalog::new(counting.clone())?.with_row_lineage(row_lineage);
        let table = catalog
            .schema("main")
            .expect("schema main")
            .table("t")
            .await?
            .expect("table t");
        let table = table
            .downcast_ref::<DuckLakeTable>()
            .expect("a DuckLake table")
            .clone();
        counting.take();
        Ok(Self {
            counting,
            memo,
            table,
        })
    }

    /// The provider of one query.
    fn provider(&self) -> anyhow::Result<Arc<dyn MetadataProvider>> {
        Ok(match &self.memo {
            Some(_) => Arc::new(self.bind()?),
            None => Arc::new(self.counting.clone()),
        })
    }

    /// The counting provider, bound to the memo for one query.
    fn bind(&self) -> anyhow::Result<MemoizedMetadataProvider> {
        let memo = self.memo.as_ref().expect("a memo");
        Ok(memo.bind(Arc::new(self.counting.clone()))?)
    }

    /// The session of one query, with the kept table registered as `t`.
    fn session(&self) -> anyhow::Result<SessionContext> {
        let ctx = SessionContext::new();
        ctx.register_table(
            "t",
            Arc::new(self.table.with_metadata_provider(self.provider()?)),
        )?;
        Ok(ctx)
    }

    async fn query(&self, sql: &str) -> anyhow::Result<String> {
        query(&self.session()?, sql).await
    }

    /// `SELECT count(*)` of `t`, as one query.
    async fn count(&self) -> anyhow::Result<i64> {
        count(&self.session()?).await
    }
}

async fn query(ctx: &SessionContext, sql: &str) -> anyhow::Result<String> {
    let batches = ctx.sql(sql).await?.collect().await?;
    Ok(pretty_format_batches(&batches)?.to_string())
}

/// `SELECT count(*)` of table `t`.
async fn count(ctx: &SessionContext) -> anyhow::Result<i64> {
    let batches = ctx.sql("SELECT count(*) FROM t").await?.collect().await?;
    Ok(batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<arrow::array::Int64Array>()
        .expect("count(*) is Int64")
        .value(0))
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

/// Check that `calls` are those of a query that fills the memo for `t` for
/// the first time: the head at the bind and after the fill, and each read of
/// the fill once. `scan_heads` are the head reads of the query's scans: a scan
/// of a table with delete files reads the head to know whether the catalog's
/// delete counts hold, as without a memo.
///
/// The fill reads the inlined rows with their row ids too, unless the provider
/// showed that the table has none. The PostgreSQL provider can show that, and
/// the SQLite provider cannot.
#[track_caller]
fn assert_fill(mut calls: BTreeMap<&'static str, usize>, scan_heads: usize, context: &str) {
    let with_row_ids = calls.remove("read_inlined_rows");
    assert!(
        with_row_ids.is_none_or(|reads| reads == 1),
        "{context}: {with_row_ids:?}"
    );
    let expected = BTreeMap::from([
        ("get_current_snapshot", 2 + scan_heads),
        ("get_inlined_deletes", 1),
        ("get_table_file_metadata_page_filtered", 1),
        ("get_table_structure", 1),
        ("scan_inlined_data_noting_absence", 1),
    ]);
    assert_eq!(calls, expected, "{context}");
}

/// As [`assert_fill`], for a fill after a commit: it takes the table's columns
/// from the fill before. It reads the inlined rows again unless the provider
/// showed that the table has none, which the PostgreSQL provider can show.
#[track_caller]
fn assert_refill(mut calls: BTreeMap<&'static str, usize>, scan_heads: usize, context: &str) {
    for read in ["read_inlined_rows", "scan_inlined_data_noting_absence"] {
        let reads = calls.remove(read);
        assert!(
            reads.is_none_or(|reads| reads == 1),
            "{context}, {read}: {reads:?}"
        );
    }
    let expected = BTreeMap::from([
        ("get_current_snapshot", 2 + scan_heads),
        ("get_inlined_deletes", 1),
        ("get_table_file_metadata_page_filtered", 1),
    ]);
    assert_eq!(calls, expected, "{context}");
}

/// The calls of a query that the memo serves: the head at the bind, and the
/// head reads of its scans (see [`assert_fill`]).
fn bind_calls(scan_heads: usize) -> BTreeMap<&'static str, usize> {
    BTreeMap::from([("get_current_snapshot", 1 + scan_heads)])
}

const CLEANUP_ALL: &str = "CALL ducklake_cleanup_old_files('lake', cleanup_all => true)";

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
// Catalog read memo
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn without_a_memo_each_query_reads_the_catalog_again() -> anyhow::Result<()> {
    let lake = Lake::new(0, WITH_DELETES)?;
    let caller = Caller::plain(&lake).await?;

    for run in 0..2 {
        caller.query("SELECT * FROM t ORDER BY id").await?;
        assert_eq!(
            caller.counting.take(),
            BTreeMap::from([
                ("get_current_snapshot", 1),
                ("get_inlined_deletes", 1),
                ("get_table_file_metadata_page_filtered", 1),
                ("scan_inlined_data_noting_absence", 1),
            ]),
            "query {run}"
        );
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn a_memo_serves_every_query_after_the_fill() -> anyhow::Result<()> {
    memo_serves_every_query_after_the_fill(&Backend::Sqlite).await
}

async fn memo_serves_every_query_after_the_fill(backend: &Backend) -> anyhow::Result<()> {
    let lake = backend.lake(0, WITH_DELETES).await?;
    let plain = Caller::plain(&lake).await?;
    let memoized = Caller::memoized(&lake).await?;

    let queries = [
        "SELECT * FROM t ORDER BY id",
        "SELECT name FROM t WHERE id > 2 ORDER BY id",
        "SELECT count(*) FROM t",
        // The plan is the same too, so `count(*)` is still answered from the
        // catalog's counts rather than by reading the data.
        "EXPLAIN SELECT count(*) FROM t",
        // Two scans of `t` in one query, under one bind.
        "SELECT a.id FROM t a JOIN t b ON a.id = b.id ORDER BY a.id",
    ];
    for (run, sql) in queries.into_iter().enumerate() {
        assert_eq!(memoized.query(sql).await?, plain.query(sql).await?, "{sql}");
        let scans = if sql.contains("JOIN") {
            2
        } else {
            1
        };
        let calls = memoized.counting.take();
        if run == 0 {
            assert_fill(calls, scans, sql);
        } else {
            assert_eq!(calls, bind_calls(scans), "query {run} ({sql})");
        }
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn a_filtered_first_query_memoizes_the_whole_listing() -> anyhow::Result<()> {
    let lake = Lake::new(0, THREE_FILES)?;
    let filtered = "SELECT id FROM t WHERE id >= 20 ORDER BY id";
    let all = "SELECT id FROM t ORDER BY id";

    // Without a memo, the filter goes to the catalog with the listing.
    let plain = Caller::plain(&lake).await?;
    let expected_filtered = plain.query(filtered).await?;
    assert_eq!(
        plain
            .counting
            .take()
            .get("get_table_file_metadata_page_filtered(filter)"),
        Some(&1)
    );

    // With one, the fill lists every file, without the filter.
    let memoized = Caller::memoized(&lake).await?;
    assert_eq!(memoized.query(filtered).await?, expected_filtered);
    assert_fill(memoized.counting.take(), 0, filtered);

    // So the memo serves an unfiltered query too.
    assert_eq!(memoized.query(all).await?, plain.query(all).await?);
    assert_eq!(memoized.counting.take(), bind_calls(0));
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn a_table_over_the_budget_reads_the_catalog_on_every_query() -> anyhow::Result<()> {
    let lake = Lake::new(0, THREE_FILES)?;
    let all = "SELECT id FROM t ORDER BY id";
    let expected = Caller::plain(&lake).await?.query(all).await?;

    let caller = Caller::new(&lake, Some(CatalogReadMemo::new(1)), false).await?;
    // The first query reads the inlined deletions and the files for the memo,
    // finds that they do not fit, and reads the catalog for itself. Later
    // queries do not try again.
    for (run, reads) in [(0, 2), (1, 1)] {
        assert_eq!(caller.query(all).await?, expected);
        assert_eq!(
            caller.counting.take(),
            BTreeMap::from([
                ("get_current_snapshot", 1),
                ("get_inlined_deletes", reads),
                ("get_table_file_metadata_page_filtered", reads),
                ("scan_inlined_data_noting_absence", 1),
            ]),
            "query {run}"
        );
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn inlined_rows_and_deletions_come_from_the_memo() -> anyhow::Result<()> {
    let lake = Lake::new(10, INLINED)?;
    let plain = Caller::plain(&lake).await?;
    let (table_id, snapshot) = table_at_head(&plain.counting)?;
    assert_eq!(
        plain
            .counting
            .get_inlined_deletes(table_id, snapshot)?
            .len(),
        2,
        "the fixture inlines its deletions"
    );
    let memoized = Caller::memoized(&lake).await?;

    let queries = [
        "SELECT id FROM t ORDER BY id",
        "SELECT id FROM t WHERE id >= 100 ORDER BY id",
        "SELECT id FROM t WHERE id IN (3, 7, 8) ORDER BY id",
        "SELECT count(*) FROM t",
    ];
    for (run, sql) in queries.into_iter().enumerate() {
        assert_eq!(memoized.query(sql).await?, plain.query(sql).await?, "{sql}");
        let calls = memoized.counting.take();
        if run == 0 {
            assert_fill(calls, 0, sql);
        } else {
            assert_eq!(calls, bind_calls(0), "query {run} ({sql})");
        }
        // The scan takes its inlined rows from the memo, so the catalog
        // materialized none for it.
        assert_eq!(memoized.table.inlined_materialized_row_count(), 0, "{sql}");
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn a_row_lineage_scan_uses_the_memo() -> anyhow::Result<()> {
    let lake = Lake::new(0, WITH_DELETES)?;
    let plain = Caller::new(&lake, None, true).await?;
    let memo = CatalogReadMemo::new(CatalogReadMemo::DEFAULT_MAX_BYTES);
    let memoized = Caller::new(&lake, Some(memo), true).await?;

    let sql = "SELECT rowid, id, name FROM t ORDER BY rowid";
    for run in 0..3 {
        assert_eq!(memoized.query(sql).await?, plain.query(sql).await?);
        let calls = memoized.counting.take();
        if run == 0 {
            assert_fill(calls, 1, sql);
        } else {
            assert_eq!(calls, bind_calls(1), "query {run}");
        }
    }
    Ok(())
}

/// A row-lineage scan reads the inlined rows with their row ids. The memo
/// serves them too, so the scan reads nothing from the catalog after the fill,
/// and every read of the scan comes from one fill.
#[tokio::test(flavor = "multi_thread")]
async fn a_row_lineage_scan_takes_inlined_rows_from_the_memo() -> anyhow::Result<()> {
    let lake = Lake::new(10, INLINED)?;
    let plain = Caller::new(&lake, None, true).await?;
    let memo = CatalogReadMemo::new(CatalogReadMemo::DEFAULT_MAX_BYTES);
    let memoized = Caller::new(&lake, Some(memo), true).await?;

    let queries = [
        "SELECT rowid, id FROM t ORDER BY rowid",
        "SELECT rowid, id FROM t WHERE id >= 100 ORDER BY rowid",
        "SELECT rowid, id FROM t WHERE id < 50 ORDER BY rowid",
    ];
    for (run, sql) in queries.into_iter().enumerate() {
        let rows = memoized.query(sql).await?;
        assert_eq!(rows, plain.query(sql).await?, "{sql}");
        let calls = memoized.counting.take();
        if run == 0 {
            assert!(rows.contains("| 101 |"), "{rows}");
            assert_eq!(calls.get("read_inlined_rows"), Some(&1), "{calls:?}");
            assert_fill(calls, 0, sql);
        } else {
            assert_eq!(calls, bind_calls(0), "query {run} ({sql})");
        }
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn concurrent_first_queries_agree_and_keep_one_fill() -> anyhow::Result<()> {
    let lake = Lake::new(0, WITH_DELETES)?;
    let all = "SELECT * FROM t ORDER BY id";
    let expected = Caller::plain(&lake).await?.query(all).await?;

    let memoized = Caller::memoized(&lake).await?;
    let (first, second) = tokio::join!(memoized.query(all), memoized.query(all));
    assert_eq!(first?, expected);
    assert_eq!(second?, expected);
    memoized.counting.take();
    assert_eq!(memoized.query(all).await?, expected);
    assert_eq!(memoized.counting.take(), bind_calls(1));
    Ok(())
}

/// A commit moves the catalog head, so the next query fills the memo again.
/// The kept table stays at its snapshot either way.
#[tokio::test(flavor = "multi_thread")]
async fn a_kept_table_keeps_reading_its_snapshot() -> anyhow::Result<()> {
    let lake = Lake::new(0, THREE_FILES)?;
    let all = "SELECT id FROM t ORDER BY id";
    let memoized = Caller::memoized(&lake).await?;
    let before = memoized.query(all).await?;

    lake.run(0, &["INSERT INTO lake.t VALUES (30)"])?;

    memoized.counting.take();
    assert_eq!(memoized.query(all).await?, before);
    assert_refill(memoized.counting.take(), 0, all);
    assert_eq!(memoized.query(all).await?, before);
    assert_eq!(memoized.counting.take(), bind_calls(0));

    // A catalog built over the bound provider reads the new snapshot, from the
    // same memo.
    let catalog = DuckLakeCatalog::new(memoized.bind()?)?;
    let ctx = SessionContext::new();
    ctx.register_catalog("lake", Arc::new(catalog));
    assert!(
        query(&ctx, "SELECT id FROM lake.main.t ORDER BY id")
            .await?
            .contains("| 30 |")
    );
    Ok(())
}

/// The head is the whole catalog's, so a commit to another table also makes
/// the next query fill the memo again: once, after which the head is still.
#[tokio::test(flavor = "multi_thread")]
async fn a_commit_to_another_table_fills_the_memo_again() -> anyhow::Result<()> {
    commit_to_another_table_fills_the_memo_again(&Backend::Sqlite).await
}

async fn commit_to_another_table_fills_the_memo_again(backend: &Backend) -> anyhow::Result<()> {
    let lake = backend.lake(0, THREE_FILES).await?;
    let all = "SELECT id FROM t ORDER BY id";
    let memoized = Caller::memoized(&lake).await?;
    let before = memoized.query(all).await?;

    lake.run(
        0,
        &["CREATE TABLE lake.u (id INT)", "INSERT INTO lake.u VALUES (1)"],
    )?;

    memoized.counting.take();
    assert_eq!(memoized.query(all).await?, before);
    assert_refill(memoized.counting.take(), 0, all);
    assert_eq!(memoized.query(all).await?, before);
    assert_eq!(memoized.counting.take(), bind_calls(0));
    Ok(())
}

/// Every commit that replaces files a snapshot reads also moves the head, so
/// the next query fills the memo from the new layout. Cleanup then deletes the
/// replaced files at once, and the kept table still reads.
#[tokio::test(flavor = "multi_thread")]
async fn a_replacing_commit_fills_the_memo_again_before_cleanup() -> anyhow::Result<()> {
    replacing_commit_fills_the_memo_again_before_cleanup(&Backend::Sqlite).await
}

async fn replacing_commit_fills_the_memo_again_before_cleanup(
    backend: &Backend,
) -> anyhow::Result<()> {
    for (fixture, inlining_rows, replace) in [
        (THREE_FILES, 0, "CALL ducklake_merge_adjacent_files('lake')"),
        (WITH_DELETES, 0, "DELETE FROM lake.t WHERE id = 5"),
        (
            WITH_DELETES,
            0,
            "CALL ducklake_rewrite_data_files('lake', delete_threshold => 0.0)",
        ),
        (INLINED, 10, "CALL ducklake_flush_inlined_data('lake')"),
    ] {
        let lake = backend.lake(inlining_rows, fixture).await?;
        let all = "SELECT * FROM t ORDER BY 1";
        let memoized = Caller::memoized(&lake).await?;
        let before = memoized.query(all).await?;

        lake.run(inlining_rows, &[replace])?;

        memoized.counting.take();
        assert_eq!(memoized.query(all).await?, before, "{replace}");
        let after_fill = memoized.counting.take();
        assert_eq!(
            after_fill.get("get_table_file_metadata_page_filtered"),
            Some(&1),
            "{replace}: {after_fill:?}"
        );
        lake.run(inlining_rows, &[CLEANUP_ALL])?;
        assert_eq!(memoized.query(all).await?, before, "{replace}");
        // Served from the memo: only heads are read.
        let served = memoized.counting.take();
        assert!(
            served.keys().copied().eq(["get_current_snapshot"]),
            "{replace}: {served:?}"
        );
    }
    Ok(())
}

/// A merge that commits while a query fills the memo changes what the fill
/// reads partway through. Such a fill serves its query but is not kept, so the
/// next query fills the memo again, from the merged layout.
#[tokio::test(flavor = "multi_thread")]
async fn a_commit_during_the_fill_is_not_kept() -> anyhow::Result<()> {
    let lake = Lake::new(0, THREE_FILES)?;
    let all = "SELECT id FROM t ORDER BY id";
    let memoized = Caller::memoized(&lake).await?;

    // Between the listing and the inlined reads.
    memoized.counting.before(
        "get_inlined_deletes",
        lake.runner(&["CALL ducklake_merge_adjacent_files('lake')"]),
    );
    let before = memoized.query(all).await?;
    assert_fill(memoized.counting.take(), 0, all);

    // Not kept: the next query fills the memo again, and now keeps it.
    assert_eq!(memoized.query(all).await?, before);
    assert_refill(memoized.counting.take(), 0, all);
    assert_eq!(memoized.query(all).await?, before);
    assert_eq!(memoized.counting.take(), bind_calls(0));

    // The memo holds the merged layout, so cleanup deletes nothing it reads.
    lake.run(0, &[CLEANUP_ALL])?;
    assert_eq!(memoized.query(all).await?, before);
    Ok(())
}

/// A commit that lands while a query fills the memo can split the fill's
/// reads. Here a DELETE replaces the delete file just before the listing, so
/// the listing holds a delete count that includes a deletion made after the
/// table's snapshot. The scan reads the head after the fill, sees that its
/// snapshot is no longer the current one, and counts by reading the data.
#[tokio::test(flavor = "multi_thread")]
async fn a_commit_during_the_fill_does_not_count_from_the_catalog() -> anyhow::Result<()> {
    commit_during_the_fill_does_not_count_from_the_catalog(&Backend::Sqlite).await
}

async fn commit_during_the_fill_does_not_count_from_the_catalog(
    backend: &Backend,
) -> anyhow::Result<()> {
    let lake = backend.lake(0, WITH_DELETES).await?;
    let memoized = Caller::memoized(&lake).await?;

    memoized.counting.before(
        "get_table_file_metadata_page_filtered",
        lake.runner(&["DELETE FROM lake.t WHERE id = 5"]),
    );
    // Ids 1, 3 and 5 are live at the table's snapshot; the DELETE of 5 came later.
    for run in 0..2 {
        assert_eq!(memoized.count().await?, 3, "query {run}");
    }
    Ok(())
}

/// Two queries bound at one head. Inside the second query's fill, the first
/// query fills and keeps its reads, and then a flush commits: it moves the
/// inlined deletions into a delete file. The second query must not mix the
/// reads of the two fills, which would bring the deleted rows 3 and 7 back.
#[tokio::test(flavor = "multi_thread")]
async fn two_queries_at_one_head_do_not_mix_their_reads() -> anyhow::Result<()> {
    for before in ["get_inlined_deletes", "get_table_structure"] {
        let lake = Lake::new(10, INLINED)?;
        let all = "SELECT id FROM t ORDER BY id";
        let expected = Caller::plain(&lake).await?.query(all).await?;
        let memoized = Caller::memoized(&lake).await?;
        let (table_id, snapshot) = table_at_head(&memoized.counting)?;

        let first = Arc::new(memoized.bind()?);
        let second = memoized.bind()?;
        let flush = lake.runner(&["CALL ducklake_flush_inlined_data('lake')"]);
        let fills = Arc::clone(&first);
        memoized.counting.before(before, move || {
            fills
                .get_inlined_deletes(table_id, snapshot)
                .expect("the first query fills");
            flush();
        });
        let ctx = SessionContext::new();
        ctx.register_table(
            "t",
            Arc::new(memoized.table.with_metadata_provider(Arc::new(second))),
        )?;
        // Before the structure read, the second query's own fill has its
        // listing from before the flush and its inlined rows from after it, and
        // lacks rows 100 and 101. It uses the first query's fill instead.
        assert_eq!(query(&ctx, all).await?, expected, "flush before {before}");
    }
    Ok(())
}

/// A query keeps the reads it picked for a table, also after a bind at a
/// newer head empties the memo between two of its calls.
#[tokio::test(flavor = "multi_thread")]
async fn a_query_keeps_the_reads_it_picked() -> anyhow::Result<()> {
    let lake = Lake::new(0, THREE_FILES)?;
    let memoized = Caller::memoized(&lake).await?;
    let (table_id, snapshot) = table_at_head(&memoized.counting)?;
    memoized.query("SELECT id FROM t").await?;

    let bound = memoized.bind()?;
    assert!(bound.get_inlined_deletes(table_id, snapshot)?.is_empty());
    lake.run(0, &["INSERT INTO lake.t VALUES (30)"])?;
    memoized.bind()?;
    memoized.counting.take();

    let files = bound.get_table_file_metadata_page_filtered(
        table_id,
        snapshot,
        None,
        FILE_METADATA_BATCH_SIZE,
        None,
    )?;
    assert_eq!(files.len(), 3);
    assert_eq!(memoized.counting.take(), BTreeMap::new());
    Ok(())
}

/// A fill that a commit split is not kept, so a query bound before the
/// commit, at the same head, fills again instead of taking it.
#[tokio::test(flavor = "multi_thread")]
async fn a_split_fill_is_not_kept_for_other_queries() -> anyhow::Result<()> {
    let lake = Lake::new(0, THREE_FILES)?;
    let all = "SELECT id FROM t ORDER BY id";
    let memoized = Caller::memoized(&lake).await?;
    let earlier = Arc::new(memoized.bind()?);

    memoized.counting.before(
        "get_table_structure",
        lake.runner(&["CREATE TABLE lake.u (id INT)"]),
    );
    let expected = memoized.query(all).await?;
    memoized.counting.take();

    let ctx = SessionContext::new();
    ctx.register_table(
        "t",
        Arc::new(memoized.table.with_metadata_provider(earlier)),
    )?;
    assert_eq!(query(&ctx, all).await?, expected);
    let calls = memoized.counting.take();
    assert_eq!(
        calls.get("get_table_file_metadata_page_filtered"),
        Some(&1),
        "{calls:?}"
    );
    Ok(())
}

/// A table whose reads fit the budget, but not what another table left of it,
/// uses its fill for one query. Later queries at the same head read the
/// catalog for it, as without a memo, and do not fill it again.
#[tokio::test(flavor = "multi_thread")]
async fn a_table_without_room_reads_the_catalog_until_the_head_moves() -> anyhow::Result<()> {
    let lake = Lake::new(
        0,
        &[
            "CREATE TABLE lake.t (id INT)",
            "INSERT INTO lake.t VALUES (1), (2)",
            "INSERT INTO lake.t VALUES (10), (11)",
            "CREATE TABLE lake.u (id INT)",
            "INSERT INTO lake.u VALUES (1), (2)",
            "INSERT INTO lake.u VALUES (10), (11)",
        ],
    )?;
    let counting = lake.provider().await?;
    let catalog = DuckLakeCatalog::new(counting.clone())?;
    let schema = catalog.schema("main").expect("schema main");
    let mut tables = Vec::new();
    for name in ["t", "u"] {
        let table = schema.table(name).await?.expect("the table");
        tables.push(
            table
                .downcast_ref::<DuckLakeTable>()
                .expect("a DuckLake table")
                .clone(),
        );
    }
    let (t, u) = (&tables[0], &tables[1]);
    let run = async |table: &DuckLakeTable, memo: &Arc<CatalogReadMemo>| {
        let provider = Arc::new(memo.bind(Arc::new(counting.clone()))?);
        let ctx = SessionContext::new();
        ctx.register_table("t", Arc::new(table.with_metadata_provider(provider)))?;
        query(&ctx, "SELECT id FROM t ORDER BY id").await
    };

    // Measure one table's fill, then leave room for one and a half.
    let probe = CatalogReadMemo::new(CatalogReadMemo::DEFAULT_MAX_BYTES);
    let expected = run(t, &probe).await?;
    let one: usize = format!("{probe:?}")
        .split(", bytes: ")
        .nth(1)
        .and_then(|rest| rest.split(',').next())
        .expect("the memo's bytes")
        .parse()?;
    let memo = CatalogReadMemo::new(one + one / 2);
    run(u, &memo).await?;
    counting.take();

    for run_no in 0..3 {
        assert_eq!(run(t, &memo).await?, expected);
        let calls = counting.take();
        if run_no == 0 {
            assert_fill(calls, 0, "the fill without room");
        } else {
            assert_eq!(
                calls,
                BTreeMap::from([
                    ("get_current_snapshot", 1),
                    ("get_inlined_deletes", 1),
                    ("get_table_file_metadata_page_filtered", 1),
                    ("scan_inlined_data_noting_absence", 1),
                ]),
                "query {run_no}"
            );
        }
    }
    Ok(())
}

/// Reads at a snapshot that expire removed. Official DuckLake refuses them
/// ("No snapshot found"). Without a memo, this crate returns no rows. With a
/// memo, a fill kept before the expire returns the old rows until cleanup
/// deletes their files, and then the read fails. See `CatalogReadMemo`.
#[tokio::test(flavor = "multi_thread")]
async fn a_kept_fill_of_an_expired_snapshot_reads_until_cleanup() -> anyhow::Result<()> {
    let lake = Lake::new(
        0,
        &["CREATE TABLE lake.t (id INT)", "INSERT INTO lake.t VALUES (1), (2)"],
    )?;
    let counting = lake.provider().await?;
    let (_, snapshot) = table_at_head(&counting)?;
    let catalog = DuckLakeCatalog::with_snapshot(Arc::new(counting.clone()), snapshot)?;
    let kept = catalog.schema("main").unwrap().table("t").await?.unwrap();
    let kept = kept
        .downcast_ref::<DuckLakeTable>()
        .expect("a DuckLake table")
        .clone();
    let memo = CatalogReadMemo::new(CatalogReadMemo::DEFAULT_MAX_BYTES);
    let read = async |memoized: bool| -> anyhow::Result<String> {
        let provider: Arc<dyn MetadataProvider> = if memoized {
            Arc::new(memo.bind(Arc::new(counting.clone()))?)
        } else {
            Arc::new(counting.clone())
        };
        let ctx = SessionContext::new();
        ctx.register_table("t", Arc::new(kept.with_metadata_provider(provider)))?;
        query(&ctx, "SELECT id FROM t ORDER BY id").await
    };

    lake.run(0, &["DELETE FROM lake.t", "INSERT INTO lake.t VALUES (10)"])?;
    let old_rows = read(true).await?;
    assert!(
        old_rows.contains("| 1  |") && old_rows.contains("| 2  |"),
        "{old_rows}"
    );
    assert_eq!(read(false).await?, old_rows);

    lake.run(
        0,
        &["CALL ducklake_expire_snapshots('lake', older_than => now())"],
    )?;
    assert_eq!(read(true).await?, old_rows);
    let plain = read(false).await?;
    assert!(!plain.contains("| 1 "), "no rows without a memo: {plain}");

    lake.run(0, &[CLEANUP_ALL])?;
    assert!(read(true).await.is_err(), "the kept fill's files are gone");
    Ok(())
}

/// A catalog built over a bound provider for each query takes its tables'
/// scans from the memo too, also behind a view. Its lookups read the catalog.
#[tokio::test(flavor = "multi_thread")]
async fn a_catalog_over_a_bound_provider_uses_the_memo() -> anyhow::Result<()> {
    let lake = Lake::new(
        0,
        &[WITH_DELETES, &["CREATE VIEW lake.v AS SELECT id FROM lake.t"]].concat(),
    )?;
    let counting = lake.provider().await?;
    let memo = CatalogReadMemo::new(CatalogReadMemo::DEFAULT_MAX_BYTES);
    let run = async |sql: &str| -> anyhow::Result<String> {
        let provider = memo.bind(Arc::new(counting.clone()))?;
        let ctx = SessionContext::new();
        ctx.register_catalog("lake", Arc::new(DuckLakeCatalog::new(provider)?));
        query(&ctx, sql).await
    };

    for sql in ["SELECT id FROM lake.main.t ORDER BY id", "SELECT id FROM lake.main.v ORDER BY id"]
    {
        let expected = run(sql).await?;
        assert!(expected.contains("| 5  |"), "{expected}");
        counting.take();
        assert_eq!(run(sql).await?, expected);
        let calls = counting.take();
        for read in [
            "get_inlined_deletes",
            "get_table_file_metadata_page_filtered",
            "scan_inlined_data_noting_absence",
        ] {
            assert!(!calls.contains_key(read), "{sql}: {calls:?}");
        }
    }
    Ok(())
}

/// `count(*)` of a table pinned at an older snapshot, with and without a memo.
///
/// The catalog's delete counts answer `count(*)` only at the current snapshot.
/// A second DELETE from a data file replaces its delete file, for older
/// snapshots too, with one that also holds the newer deletion, so its count
/// over-states the deletions at the older snapshot.
#[tokio::test(flavor = "multi_thread")]
async fn a_table_at_an_older_snapshot_counts_its_own_rows() -> anyhow::Result<()> {
    let lake = Lake::new(0, WITH_DELETES)?;
    let (_, snapshot) = table_at_head(&lake.provider().await?)?;
    lake.run(0, &["DELETE FROM lake.t WHERE id = 5"])?;

    let memo = CatalogReadMemo::new(CatalogReadMemo::DEFAULT_MAX_BYTES);
    for memoized in [false, true] {
        for run in 0..2 {
            let provider: Arc<dyn MetadataProvider> = Arc::new(lake.provider().await?);
            let provider: Arc<dyn MetadataProvider> = if memoized {
                Arc::new(memo.bind(provider)?)
            } else {
                provider
            };
            let fixed = DuckLakeCatalog::with_snapshot(provider, snapshot)?;
            let table = fixed.schema("main").unwrap().table("t").await?.unwrap();
            let ctx = SessionContext::new();
            ctx.register_table("t", table)?;
            // Ids 1, 3 and 5 are live at the snapshot; the DELETE of 5 came later.
            assert_eq!(count(&ctx).await?, 3, "memoized {memoized}, query {run}");
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Delete memo
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn the_delete_memo_reads_each_delete_file_once() -> anyhow::Result<()> {
    let lake = Lake::new(0, WITH_DELETES)?;
    let all = "SELECT * FROM t ORDER BY id";
    for (options, read_again) in
        [(DuckLakeReadOptions::memoized(), false), (DuckLakeReadOptions::default(), true)]
    {
        let catalog =
            DuckLakeCatalog::new(lake.provider().await?)?.with_read_options(options.clone());
        let table = catalog.schema("main").unwrap().table("t").await?.unwrap();
        let ctx = SessionContext::new();
        ctx.register_table("t", table)?;
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

/// Every scan lists the files again. A second delete from a data file replaces
/// its delete file, even for older snapshots, so a pinned table's later scan
/// lists a new path. The delete memo reads the new file once and keeps it in
/// place of the old one.
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

// ---------------------------------------------------------------------------
// PostgreSQL metadata
// ---------------------------------------------------------------------------

/// The catalog read memo on PostgreSQL metadata: the same head checks and fill
/// guards as the SQLite tests above, through the PostgreSQL provider, whose
/// head is the newest snapshot of the catalog.
#[cfg(feature = "metadata-postgres")]
mod postgres {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use testcontainers::ContainerAsync;
    use testcontainers::runners::AsyncRunner;
    use testcontainers_modules::postgres::Postgres;

    use super::*;

    /// A PostgreSQL server in a container, for the catalogs of one test.
    pub(super) struct Server {
        _container: ContainerAsync<Postgres>,
        port: u16,
        databases: AtomicUsize,
    }

    impl Server {
        async fn start() -> anyhow::Result<Self> {
            let container = Postgres::default().start().await?;
            let port = container.get_host_port_ipv4(5432).await?;
            Ok(Self {
                _container: container,
                port,
                databases: AtomicUsize::new(0),
            })
        }

        fn url(&self, database: &str) -> String {
            format!(
                "postgresql://postgres:postgres@127.0.0.1:{}/{database}",
                self.port
            )
        }

        /// A new, empty database on this server, so that each catalog of a
        /// test starts empty.
        pub(super) async fn database(&self) -> anyhow::Result<Metadata> {
            let name = format!("lake_{}", self.databases.fetch_add(1, Ordering::Relaxed));
            let admin = sqlx::PgPool::connect(&self.url("postgres")).await?;
            sqlx::query(sqlx::AssertSqlSafe(format!("CREATE DATABASE {name}")))
                .execute(&admin)
                .await?;
            admin.close().await;
            Ok(Metadata::Postgres {
                url: self.url(&name),
                libpq: format!(
                    "host=127.0.0.1 port={} dbname={name} user=postgres password=postgres",
                    self.port
                ),
            })
        }
    }

    async fn backend() -> anyhow::Result<Backend> {
        Ok(Backend::Postgres(Box::new(Server::start().await?)))
    }

    #[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_memo_serves_every_query_after_the_fill() -> anyhow::Result<()> {
        memo_serves_every_query_after_the_fill(&backend().await?).await
    }

    #[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_commit_to_another_table_fills_the_memo_again() -> anyhow::Result<()> {
        commit_to_another_table_fills_the_memo_again(&backend().await?).await
    }

    #[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_replacing_commit_fills_the_memo_again_before_cleanup() -> anyhow::Result<()> {
        replacing_commit_fills_the_memo_again_before_cleanup(&backend().await?).await
    }

    #[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_commit_during_the_fill_does_not_count_from_the_catalog() -> anyhow::Result<()> {
        commit_during_the_fill_does_not_count_from_the_catalog(&backend().await?).await
    }
}
