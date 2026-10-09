//! Commit guards on the PostgreSQL metadata writers.
//!
//! A guard runs inside each metadata transaction, after its last write and right
//! before `COMMIT`. These tests drive every commit path of both PostgreSQL
//! writers with a guard that refuses, and check that the refusal commits nothing
//! (the metadata database is unchanged, row for row), that a write removes the
//! files it uploaded, and that the error carries the guard's own. They also show
//! the guard runs on the live transaction: it sees the rows the transaction
//! wrote, and a row lock it takes holds until the commit finishes.
//! Docker-gated (testcontainers Postgres).

#![cfg(feature = "write-postgres")]

use std::error::Error;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use arrow::array::{Int32Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::error::DataFusionError;
use datafusion::prelude::*;
use datafusion_ducklake::metadata_writer::{DataFileInfo, PromotedFile};
use datafusion_ducklake::{
    ColumnDef, CommitGuardError, DeleteFileEntry, DuckLakeCatalog, DuckLakeError, DuckLakeTable,
    DuckLakeTableWriter, DuckLakeWriteOptions, InlinedRowRef, MergeOptions, MetadataProvider,
    MetadataWriter, MulticatalogManager, MulticatalogProvider, NullOrder, PartitionTransform,
    PostgresCommitGuard, PostgresMetadataProvider, PostgresMetadataWriter,
    PostgresSingleCatalogMetadataWriter, RewriteOptions, SortDirection, SortField, TagObjectType,
    TagTarget, WriteMode,
};
use object_store::local::LocalFileSystem;
use sqlx::AssertSqlSafe;
use sqlx::postgres::{PgPool, PgPoolOptions};
use tempfile::TempDir;
use testcontainers::ContainerAsync;
use testcontainers::runners::AsyncRunner;
use testcontainers_modules::postgres::Postgres;

type BoxError = Box<dyn Error + Send + Sync>;
type ObjStore = Arc<dyn object_store::ObjectStore>;
/// One writer call of a table of cases.
type Case<'a> = Box<dyn Fn() -> datafusion_ducklake::Result<()> + 'a>;

async fn spin_up_postgres() -> anyhow::Result<(PgPool, String, ContainerAsync<Postgres>)> {
    let container = Postgres::default().start().await?;
    let port = container.get_host_port_ipv4(5432).await?;
    let base = format!("postgresql://postgres:postgres@127.0.0.1:{port}");
    let pool = PgPoolOptions::new()
        .max_connections(5)
        .connect(&format!("{base}/postgres"))
        .await?;
    datafusion_ducklake::initialize_multicatalog_schema(&pool).await?;
    Ok((pool, base, container))
}

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("val", DataType::Int32, false),
    ]))
}

fn batch(ids: &[i32], vals: &[i32]) -> RecordBatch {
    RecordBatch::try_new(
        schema(),
        vec![Arc::new(Int32Array::from(ids.to_vec())), Arc::new(Int32Array::from(vals.to_vec()))],
    )
    .unwrap()
}

fn columns() -> Vec<ColumnDef> {
    vec![
        ColumnDef::new("id", "int32", false).unwrap(),
        ColumnDef::new("val", "int32", false).unwrap(),
    ]
}

/// The error the test guards refuse with.
#[derive(Debug)]
struct LeaseLost;

impl std::fmt::Display for LeaseLost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("writer lease lost")
    }
}

impl Error for LeaseLost {}

/// Approves every commit until armed with [`Self::refuse_after`], then approves
/// that many more and refuses every one after them.
#[derive(Debug)]
struct SwitchGuard {
    calls: AtomicUsize,
    refuse_from: AtomicUsize,
}

impl SwitchGuard {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            calls: AtomicUsize::new(0),
            refuse_from: AtomicUsize::new(usize::MAX),
        })
    }

    fn refuse_after(&self, approvals: usize) {
        self.calls.store(0, Ordering::SeqCst);
        self.refuse_from.store(approvals, Ordering::SeqCst);
    }

    fn approve(&self) {
        self.refuse_from.store(usize::MAX, Ordering::SeqCst);
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

#[async_trait::async_trait]
impl PostgresCommitGuard for SwitchGuard {
    async fn before_commit(&self, _conn: &mut sqlx::PgConnection) -> Result<(), CommitGuardError> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        if call >= self.refuse_from.load(Ordering::SeqCst) {
            Err(Box::new(LeaseLost))
        } else {
            Ok(())
        }
    }
}

/// A catalog with a guarded writer and a seeded table `public.t` of two files,
/// (1,10),(2,20) then (3,30),(4,40).
struct Lake {
    name: String,
    /// The multicatalog catalog id, or `None` for a single-catalog database.
    catalog_id: Option<i64>,
    pool: PgPool,
    url: String,
    data: PathBuf,
    _tmp: TempDir,
    guard: Arc<SwitchGuard>,
    writer: Arc<dyn MetadataWriter>,
    os: ObjStore,
    table_id: i64,
    /// The snapshot of the first seed write.
    first_snapshot: i64,
}

impl Lake {
    async fn multi(pool: &PgPool, base: &str, name: &str) -> Self {
        let catalog_id = MulticatalogManager::new(pool.clone())
            .create_catalog(name)
            .await
            .unwrap();
        let guard = SwitchGuard::new();
        let writer = PostgresMetadataWriter::with_pool(pool.clone(), catalog_id)
            .await
            .unwrap()
            .with_commit_guard(guard.clone());
        Self::seed(
            name,
            Some(catalog_id),
            pool.clone(),
            format!("{base}/postgres"),
            guard,
            Arc::new(writer),
        )
        .await
    }

    async fn single(admin: &PgPool, base: &str, name: &str) -> Self {
        sqlx::query(AssertSqlSafe(format!("CREATE DATABASE {name}")))
            .execute(admin)
            .await
            .unwrap();
        let url = format!("{base}/{name}");
        let guard = SwitchGuard::new();
        let writer = PostgresSingleCatalogMetadataWriter::new_with_init(&url)
            .await
            .unwrap()
            .with_commit_guard(guard.clone());
        let pool = PgPoolOptions::new()
            .max_connections(2)
            .connect(&url)
            .await
            .unwrap();
        Self::seed(name, None, pool, url, guard, Arc::new(writer)).await
    }

    async fn seed(
        name: &str,
        catalog_id: Option<i64>,
        pool: PgPool,
        url: String,
        guard: Arc<SwitchGuard>,
        writer: Arc<dyn MetadataWriter>,
    ) -> Self {
        let tmp = TempDir::new().unwrap();
        let data = tmp.path().join("data");
        std::fs::create_dir_all(&data).unwrap();
        writer.set_data_path(data.to_str().unwrap()).unwrap();
        let os: ObjStore = Arc::new(LocalFileSystem::new());
        let table_writer = DuckLakeTableWriter::new(writer.clone(), os.clone()).unwrap();
        let first = table_writer
            .write_table("public", "t", &[batch(&[1, 2], &[10, 20])])
            .await
            .unwrap();
        table_writer
            .append_table("public", "t", &[batch(&[3, 4], &[30, 40])])
            .await
            .unwrap();
        Self {
            name: name.to_string(),
            catalog_id,
            pool,
            url,
            data,
            _tmp: tmp,
            guard,
            writer,
            os,
            table_id: first.table_id,
            first_snapshot: first.snapshot_id,
        }
    }

    fn table_writer(&self) -> DuckLakeTableWriter {
        DuckLakeTableWriter::new(self.writer.clone(), self.os.clone()).unwrap()
    }

    fn inlining_writer(&self) -> DuckLakeTableWriter {
        let mut options = DuckLakeWriteOptions::default();
        options.data_inlining_row_limit = Some(100);
        self.table_writer().with_options(&options)
    }

    async fn provider(&self) -> Arc<dyn MetadataProvider> {
        match self.catalog_id {
            Some(_) => Arc::new(
                MulticatalogProvider::with_pool(self.pool.clone(), &self.name)
                    .await
                    .unwrap(),
            ),
            None => Arc::new(PostgresMetadataProvider::new(&self.url).await.unwrap()),
        }
    }

    /// A session whose catalog `self.name` writes through the guarded writer.
    async fn context(&self) -> SessionContext {
        let catalog =
            DuckLakeCatalog::with_writer(self.provider().await, self.writer.clone()).unwrap();
        let ctx = SessionContext::new();
        ctx.register_catalog(&self.name, Arc::new(catalog));
        ctx
    }

    async fn sql(&self, sql: &str) -> Result<(), DataFusionError> {
        self.context().await.sql(sql).await?.collect().await?;
        Ok(())
    }

    async fn table(&self, ctx: &SessionContext, name: &str) -> DuckLakeTable {
        let provider = ctx
            .catalog(&self.name)
            .unwrap()
            .schema("public")
            .unwrap()
            .table(name)
            .await
            .unwrap()
            .unwrap();
        (provider.as_ref() as &dyn std::any::Any)
            .downcast_ref::<DuckLakeTable>()
            .expect("provider is a DuckLakeTable")
            .clone()
    }

    async fn rows(&self, table: &str) -> Vec<(i32, i32)> {
        let provider = self.provider().await;
        let head = provider.get_current_snapshot().unwrap();
        let catalog = DuckLakeCatalog::with_snapshot(provider, head).unwrap();
        let ctx = SessionContext::new();
        ctx.register_catalog(&self.name, Arc::new(catalog));
        let batches = ctx
            .sql(&format!(
                "SELECT id, val FROM {}.public.{table} ORDER BY id",
                self.name
            ))
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let mut rows = Vec::new();
        for b in &batches {
            let ids = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
            let vals = b.column(1).as_any().downcast_ref::<Int32Array>().unwrap();
            for i in 0..b.num_rows() {
                rows.push((ids.value(i), vals.value(i)));
            }
        }
        rows
    }

    async fn head(&self) -> i64 {
        match self.catalog_id {
            Some(id) => sqlx::query_scalar(
                "SELECT MAX(snapshot_id) FROM ducklake_catalog_snapshot_map WHERE catalog_id = $1",
            )
            .bind(id)
            .fetch_one(&self.pool)
            .await
            .unwrap(),
            None => sqlx::query_scalar("SELECT MAX(snapshot_id) FROM ducklake_snapshot")
                .fetch_one(&self.pool)
                .await
                .unwrap(),
        }
    }

    async fn live_files(&self) -> Vec<datafusion_ducklake::DuckLakeTableFile> {
        let provider = self.provider().await;
        let head = provider.get_current_snapshot().unwrap();
        provider
            .get_table_files_for_select(self.table_id, head)
            .unwrap()
    }

    /// Everything a refused commit must leave as it was.
    async fn state(&self) -> State {
        State {
            head: self.head().await,
            metadata: metadata_fingerprint(&self.pool).await,
            files: files_under(&self.data),
        }
    }

    /// Assert `result` is the refusal of the guard's `approvals + 1`th commit
    /// since it was armed, and that nothing changed since `before`.
    async fn assert_refused<T, E: Into<BoxError>>(
        &self,
        case: &str,
        before: State,
        approvals: usize,
        result: Result<T, E>,
    ) {
        let error: BoxError = match result {
            Ok(_) => panic!("{case}: committed despite the guard refusing"),
            Err(e) => e.into(),
        };
        let refusal = commit_refusal(error.as_ref())
            .unwrap_or_else(|| panic!("{case}: not a CommitRefused: {error}"));
        assert!(
            refusal
                .source()
                .and_then(|source| source.downcast_ref::<LeaseLost>())
                .is_some(),
            "{case}: the refusal carries the guard's error"
        );
        assert_eq!(
            self.guard.calls(),
            approvals + 1,
            "{case}: the guard ran once on the refused commit, after {approvals} approved"
        );
        let after = self.state().await;
        assert_eq!(after.head, before.head, "{case}: the head did not move");
        assert_eq!(
            after.files, before.files,
            "{case}: uploaded files were removed"
        );
        assert_eq!(
            after.metadata, before.metadata,
            "{case}: the metadata database is unchanged"
        );
        self.guard.approve();
    }
}

struct State {
    head: i64,
    metadata: Vec<(String, String)>,
    files: Vec<PathBuf>,
}

/// One digest per `ducklake*` table of the database, over every row, so any
/// committed write shows. Id counters in `ducklake_metadata` are left out:
/// reserving ids commits them before the commit a guard refuses.
async fn metadata_fingerprint(pool: &PgPool) -> Vec<(String, String)> {
    let tables: Vec<String> = sqlx::query_scalar(
        "SELECT table_name::text FROM information_schema.tables
         WHERE table_schema = 'public' AND table_name LIKE 'ducklake%' ORDER BY 1",
    )
    .fetch_all(pool)
    .await
    .unwrap();
    let mut digests = Vec::with_capacity(tables.len());
    for table in tables {
        let filter = if table == "ducklake_metadata" {
            "WHERE key NOT LIKE 'next\\_%'"
        } else {
            ""
        };
        let digest: String = sqlx::query_scalar(AssertSqlSafe(format!(
            "SELECT COALESCE(md5(string_agg(x::text, '|' ORDER BY x::text)), '')
             FROM \"{table}\" x {filter}"
        )))
        .fetch_one(pool)
        .await
        .unwrap();
        digests.push((table, digest));
    }
    digests
}

fn files_under(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let mut pending = vec![dir.to_path_buf()];
    while let Some(dir) = pending.pop() {
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                pending.push(path);
            } else {
                files.push(path);
            }
        }
    }
    files.sort();
    files
}

/// The `CommitRefused` in `error`'s source chain, including through a
/// DataFusion error wrapping it.
fn commit_refusal<'a>(error: &'a (dyn Error + 'static)) -> Option<&'a DuckLakeError> {
    let mut current = Some(error);
    while let Some(e) = current {
        if let Some(refused @ DuckLakeError::CommitRefused(_)) = e.downcast_ref::<DuckLakeError>() {
            return Some(refused);
        }
        if let Some(DataFusionError::External(inner)) = e.downcast_ref::<DataFusionError>()
            && let Some(refused) = commit_refusal(inner.as_ref())
        {
            return Some(refused);
        }
        current = e.source();
    }
    None
}

/// Every commit path of the multicatalog writer, refused: nothing is committed
/// and the files written for it are removed.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn multicatalog_refusal_commits_nothing_on_every_path() {
    let (pool, base, _c) = spin_up_postgres().await.unwrap();

    // Whole-table writes: the id reservation is approved, the registering commit
    // refused after its files were uploaded.
    let l = Lake::multi(&pool, &base, "append_rows").await;
    let before = l.state().await;
    l.guard.refuse_after(1);
    let r = l
        .table_writer()
        .append_table("public", "t", &[batch(&[5], &[50])])
        .await;
    l.assert_refused("append_table", before, 1, r).await;

    let l = Lake::multi(&pool, &base, "replace_rows").await;
    let before = l.state().await;
    l.guard.refuse_after(1);
    let r = l
        .table_writer()
        .write_table("public", "t", &[batch(&[5], &[50])])
        .await;
    l.assert_refused("write_table", before, 1, r).await;

    let l = Lake::multi(&pool, &base, "partitioned").await;
    l.writer
        .set_partition_spec(
            l.table_id,
            &[("id".to_string(), PartitionTransform::Identity)],
        )
        .unwrap();
    let partition_id = l
        .writer
        .live_partition_spec(l.table_id)
        .unwrap()
        .unwrap()
        .partition_id;
    let before = l.state().await;
    l.guard.refuse_after(1);
    let r = l
        .table_writer()
        .write_partitioned(
            "public",
            "t",
            &schema(),
            WriteMode::Append,
            partition_id,
            &["id".to_string()],
            vec![(vec![Some("5".to_string())], vec![batch(&[5], &[50])])],
        )
        .await;
    l.assert_refused("write_partitioned", before, 1, r).await;

    // A refused reservation stops a write before it uploads anything.
    let l = Lake::multi(&pool, &base, "reservation").await;
    let before = l.state().await;
    l.guard.refuse_after(0);
    let r = l
        .writer
        .begin_write_transaction("public", "t", &columns(), WriteMode::Append);
    l.assert_refused("begin_write_transaction", before, 0, r)
        .await;

    // Streaming sessions: refused at finish.
    for (case, mode) in
        [("session_append", WriteMode::Append), ("session_replace", WriteMode::Replace)]
    {
        let l = Lake::multi(&pool, &base, case).await;
        let before = l.state().await;
        let table_writer = l.table_writer();
        let mut session = table_writer
            .begin_write("public", "t", &schema(), mode)
            .unwrap();
        session.write_batch(&batch(&[5, 6], &[50, 60])).unwrap();
        l.guard.refuse_after(0);
        let r = session.finish().await;
        l.assert_refused(case, before, 0, r).await;
    }

    let l = Lake::multi(&pool, &base, "session_deletes").await;
    let target = l.live_files().await.remove(0);
    let before = l.state().await;
    let table_writer = l.table_writer();
    let delete = table_writer
        .write_delete_file("public", "t", &target.file.path, &[0])
        .await
        .unwrap();
    let mut session = table_writer
        .begin_write("public", "t", &schema(), WriteMode::Append)
        .unwrap();
    session.write_batch(&batch(&[1], &[11])).unwrap();
    l.guard.refuse_after(0);
    let r = session
        .finish_with_deletes(&[DeleteFileEntry {
            data_file_id: target.data_file_id,
            expected_prev_delete_file: target.delete_file_id,
            delete,
        }])
        .await;
    l.assert_refused("finish_with_deletes", before, 0, r).await;

    // SQL DML through the catalog.
    for (case, approvals, sql) in [
        ("sql_insert", 1, "INSERT INTO {}.public.t VALUES (5, 50)"),
        (
            "sql_update",
            1,
            "UPDATE {}.public.t SET val = 11 WHERE id = 1",
        ),
        ("sql_delete", 0, "DELETE FROM {}.public.t WHERE id = 1"),
        ("sql_truncate", 0, "DELETE FROM {}.public.t"),
    ] {
        let l = Lake::multi(&pool, &base, case).await;
        let before = l.state().await;
        l.guard.refuse_after(approvals);
        let r = l.sql(&sql.replace("{}", case)).await;
        l.assert_refused(case, before, approvals, r).await;
    }

    let l = Lake::multi(&pool, &base, "sql_create").await;
    let before = l.state().await;
    l.guard.refuse_after(0);
    let r = l.sql("CREATE TABLE sql_create.public.n (a INT)").await;
    l.assert_refused("sql_create_table", before, 0, r).await;

    // A multi-table transaction: both stages reserved, the one commit refused.
    let l = Lake::multi(&pool, &base, "multi_table").await;
    let before = l.state().await;
    let table_writer = l.table_writer();
    let mut transaction = table_writer.transaction();
    transaction
        .stage_write(
            "public",
            "t",
            &schema(),
            WriteMode::Append,
            &[batch(&[5], &[50])],
        )
        .await
        .unwrap();
    transaction
        .stage_write(
            "public",
            "u",
            &schema(),
            WriteMode::Replace,
            &[batch(&[1], &[1])],
        )
        .await
        .unwrap();
    l.guard.refuse_after(0);
    let r = transaction.commit().await;
    l.assert_refused("commit_multi_table", before, 0, r).await;

    // Inlined data: registering it, flushing it to a file, and deleting from it.
    let l = Lake::multi(&pool, &base, "inline_register").await;
    let before = l.state().await;
    l.guard.refuse_after(1);
    let r = l
        .inlining_writer()
        .append_table("public", "inl", &[batch(&[1], &[10])])
        .await;
    l.assert_refused("register_inlined_data", before, 1, r)
        .await;

    let l = Lake::multi(&pool, &base, "inline_flush").await;
    let inlined = l
        .inlining_writer()
        .append_table("public", "inl", &[batch(&[1, 2], &[10, 20])])
        .await
        .unwrap();
    let provider = l.provider().await;
    let table_columns = provider
        .get_table_structure(inlined.table_id, inlined.snapshot_id)
        .unwrap();
    let inlined_rows = provider
        .get_inlined_data_with_row_ids(inlined.table_id, inlined.snapshot_id, &table_columns)
        .unwrap();
    let before = l.state().await;
    l.guard.refuse_after(1);
    let r = l
        .table_writer()
        .flush_inlined_data("public", "inl", &inlined_rows, inlined.snapshot_id)
        .await;
    l.assert_refused("flush_inlined_data", before, 1, r).await;

    let before = l.state().await;
    let rows: Vec<InlinedRowRef> = inlined_rows
        .iter()
        .flat_map(|data| {
            data.row_ids.iter().map(|row_id| InlinedRowRef {
                table_name: data.table_name.clone(),
                row_id: *row_id,
            })
        })
        .collect();
    l.guard.refuse_after(0);
    let r = l.writer.commit_inlined_deletes(
        inlined.table_id,
        "public",
        "inl",
        inlined.snapshot_id,
        &rows,
    );
    l.assert_refused("commit_inlined_deletes", before, 0, r)
        .await;

    // Compaction.
    let l = Lake::multi(&pool, &base, "merge").await;
    let before = l.state().await;
    let ctx = l.context().await;
    let table = l.table(&ctx, "t").await;
    l.guard.refuse_after(0);
    let r = table
        .merge_adjacent_files(&ctx.state(), MergeOptions::default())
        .await;
    l.assert_refused("merge_adjacent_files", before, 0, r).await;

    let l = Lake::multi(&pool, &base, "rewrite").await;
    l.sql("DELETE FROM rewrite.public.t WHERE id = 1")
        .await
        .unwrap();
    let before = l.state().await;
    let ctx = l.context().await;
    let table = l.table(&ctx, "t").await;
    l.guard.refuse_after(0);
    let r = table
        .rewrite_data_files(
            &ctx.state(),
            RewriteOptions {
                delete_threshold: 0.0,
                ..RewriteOptions::default()
            },
        )
        .await;
    l.assert_refused("rewrite_data_files", before, 0, r).await;

    // Direct delete commits. The delete file is the caller's, written before.
    let l = Lake::multi(&pool, &base, "positional").await;
    let target = l.live_files().await.remove(0);
    let delete = l
        .table_writer()
        .write_delete_file("public", "t", &target.file.path, &[0])
        .await
        .unwrap();
    let head = l.head().await;
    let before = l.state().await;
    l.guard.refuse_after(0);
    let r = l.writer.commit_positional_deletes(
        l.table_id,
        "public",
        "t",
        head,
        &[DeleteFileEntry {
            data_file_id: target.data_file_id,
            expected_prev_delete_file: target.delete_file_id,
            delete: delete.clone(),
        }],
    );
    l.assert_refused("commit_positional_deletes", before, 0, r)
        .await;
    let before = l.state().await;
    l.guard.refuse_after(0);
    let r = l.writer.set_delete_file(
        l.table_id,
        "public",
        "t",
        head,
        target.data_file_id,
        target.delete_file_id,
        head,
        &delete,
    );
    l.assert_refused("set_delete_file", before, 0, r).await;

    // Registering existing files, one and several.
    let l = Lake::multi(&pool, &base, "existing").await;
    let ids = [100_i64, 200];
    let before = l.state().await;
    l.guard.refuse_after(0);
    let r = l.writer.register_existing_data_file(
        "public",
        "adopted",
        &columns(),
        &ids,
        &DataFileInfo::new("a.parquet", 1024, 3),
        WriteMode::Replace,
    );
    l.assert_refused("register_existing_data_file", before, 0, r)
        .await;
    let before = l.state().await;
    l.guard.refuse_after(0);
    let r = l.writer.register_existing_data_files(
        "public",
        "adopted",
        &columns(),
        &ids,
        &[
            PromotedFile::new(DataFileInfo::new("a.parquet", 1024, 3)),
            PromotedFile::new(DataFileInfo::new("b.parquet", 1024, 3)),
        ],
        None,
        WriteMode::Replace,
    );
    l.assert_refused("register_existing_data_files", before, 0, r)
        .await;

    // Schema, layout and setting changes, each one transaction.
    let l = Lake::multi(&pool, &base, "ddl").await;
    let head = l.head().await;
    let public_id = l
        .provider()
        .await
        .get_schema_by_name("public", head)
        .unwrap()
        .unwrap()
        .schema_id;
    let mut extended = columns();
    extended.push(ColumnDef::new("extra", "int32", true).unwrap());
    let sort = [SortField::column(0, "val", SortDirection::Desc, NullOrder::NullsLast)];
    let identity = [("id".to_string(), PartitionTransform::Identity)];
    let tag = TagTarget::Object {
        object_type: TagObjectType::Table,
        object_id: l.table_id,
    };
    let data_path = l.data.to_str().unwrap().to_string();
    let w = &l.writer;
    let t = l.table_id;
    let cases: Vec<(&str, Case<'_>)> = vec![
        (
            "promote_column_type",
            Box::new(|| w.promote_column_type(t, "val", "int64").map(drop)),
        ),
        (
            "set_columns",
            Box::new(|| w.set_columns(t, &extended, head).map(drop)),
        ),
        (
            "get_or_create_schema (new)",
            Box::new(|| w.get_or_create_schema("s2", None, head).map(drop)),
        ),
        (
            "get_or_create_schema (existing)",
            Box::new(|| w.get_or_create_schema("public", None, head).map(drop)),
        ),
        (
            "get_or_create_table (new)",
            Box::new(|| w.get_or_create_table(public_id, "t2", None, head).map(drop)),
        ),
        (
            "create_snapshot",
            Box::new(|| w.create_snapshot().map(drop)),
        ),
        (
            "set_partition_spec",
            Box::new(|| w.set_partition_spec(t, &identity).map(drop)),
        ),
        (
            "set_sort_spec",
            Box::new(|| w.set_sort_spec(t, &sort).map(drop)),
        ),
        (
            "set_tag",
            Box::new(|| w.set_tag(tag, "comment", Some("c")).map(drop)),
        ),
        (
            "set_table_setting",
            Box::new(|| w.set_table_setting(t, "target_file_size", "1000")),
        ),
        (
            "set_global_setting",
            Box::new(|| w.set_global_setting("data_inlining_row_limit", "5")),
        ),
        (
            "set_inlined_index_columns",
            Box::new(|| w.set_inlined_index_columns(t, &["id".to_string()])),
        ),
        (
            "ensure_inlined_indexes",
            Box::new(|| w.ensure_inlined_indexes(t)),
        ),
        (
            "end_table_files",
            Box::new(|| w.end_table_files(t, head).map(drop)),
        ),
        ("set_data_path", Box::new(|| w.set_data_path(&data_path))),
        (
            "retire_appends_since",
            Box::new(|| w.retire_appends_since(t, l.first_snapshot).map(drop)),
        ),
        (
            "commit_truncate",
            Box::new(|| w.commit_truncate(t, "public", "t", head).map(drop)),
        ),
    ];
    for (case, op) in cases {
        let before = l.state().await;
        l.guard.refuse_after(0);
        l.assert_refused(case, before, 0, op()).await;
    }

    // Resetting a layout refuses like setting one.
    w.set_partition_spec(t, &identity).unwrap();
    w.set_sort_spec(t, &sort).unwrap();
    for (case, op) in [
        (
            "reset_partition_spec",
            Box::new(|| w.reset_partition_spec(t)) as Box<dyn Fn() -> _>,
        ),
        ("reset_sort_spec", Box::new(|| w.reset_sort_spec(t))),
    ] {
        let before = l.state().await;
        l.guard.refuse_after(0);
        l.assert_refused(case, before, 0, op()).await;
    }
}

/// `PostgresMetadataWriter::drop_table` drops through the writer's guard, and
/// otherwise matches `MulticatalogManager::drop_table_in_catalog`.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn writer_drop_table_is_guarded() {
    let (pool, base, _c) = spin_up_postgres().await.unwrap();
    let l = Lake::multi(&pool, &base, "dropped").await;
    let writer = PostgresMetadataWriter::with_pool(pool.clone(), l.catalog_id.unwrap())
        .await
        .unwrap()
        .with_commit_guard(l.guard.clone());

    let before = l.state().await;
    l.guard.refuse_after(0);
    let r = writer.drop_table("public", "t");
    l.assert_refused("drop_table", before, 0, r).await;
    assert_eq!(l.rows("t").await, vec![(1, 10), (2, 20), (3, 30), (4, 40)]);

    let head = l.head().await;
    assert!(
        writer.drop_table("public", "t").unwrap(),
        "the live table is dropped"
    );
    assert!(l.head().await > head, "the drop commits a snapshot");
    let provider = l.provider().await;
    let snapshot = provider.get_current_snapshot().unwrap();
    let public = provider
        .get_schema_by_name("public", snapshot)
        .unwrap()
        .unwrap();
    assert!(
        provider
            .get_table_by_name(public.schema_id, "t", snapshot)
            .unwrap()
            .is_none(),
        "the table is gone at the head"
    );
    assert!(
        !writer.drop_table("public", "t").unwrap(),
        "dropping it again is a no-op"
    );
}

/// The commit paths of the single-catalog writer, refused.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn single_catalog_refusal_commits_nothing() {
    let (admin, base, _c) = spin_up_postgres().await.unwrap();

    let l = Lake::single(&admin, &base, "single_append").await;
    let before = l.state().await;
    l.guard.refuse_after(1);
    let r = l
        .table_writer()
        .append_table("public", "t", &[batch(&[5], &[50])])
        .await;
    l.assert_refused("append_table", before, 1, r).await;

    let l = Lake::single(&admin, &base, "single_session").await;
    let table_writer = l.table_writer();
    let mut session = table_writer
        .begin_write("public", "t", &schema(), WriteMode::Replace)
        .unwrap();
    let before = l.state().await;
    session.write_batch(&batch(&[5], &[50])).unwrap();
    l.guard.refuse_after(0);
    let r = session.finish().await;
    l.assert_refused("session finish", before, 0, r).await;

    let l = Lake::single(&admin, &base, "single_multi").await;
    let before = l.state().await;
    let table_writer = l.table_writer();
    let mut transaction = table_writer.transaction();
    transaction
        .stage_write(
            "public",
            "t",
            &schema(),
            WriteMode::Append,
            &[batch(&[5], &[50])],
        )
        .await
        .unwrap();
    transaction
        .stage_write(
            "public",
            "u",
            &schema(),
            WriteMode::Replace,
            &[batch(&[1], &[1])],
        )
        .await
        .unwrap();
    l.guard.refuse_after(0);
    let r = transaction.commit().await;
    l.assert_refused("commit_multi_table", before, 0, r).await;

    let l = Lake::single(&admin, &base, "single_ddl").await;
    let w = &l.writer;
    let t = l.table_id;
    let identity = [("id".to_string(), PartitionTransform::Identity)];
    let tag = TagTarget::Object {
        object_type: TagObjectType::Table,
        object_id: t,
    };
    let sort = [SortField::column(0, "val", SortDirection::Desc, NullOrder::NullsLast)];
    let cases: Vec<(&str, Case<'_>)> = vec![
        (
            "begin_write_transaction",
            Box::new(|| {
                w.begin_write_transaction("public", "t", &columns(), WriteMode::Append)
                    .map(drop)
            }),
        ),
        (
            "set_sort_spec",
            Box::new(|| w.set_sort_spec(t, &sort).map(drop)),
        ),
        (
            "set_global_setting",
            Box::new(|| w.set_global_setting("data_inlining_row_limit", "5")),
        ),
        (
            "set_partition_spec",
            Box::new(|| w.set_partition_spec(t, &identity).map(drop)),
        ),
        (
            "set_tag",
            Box::new(|| w.set_tag(tag, "comment", Some("c")).map(drop)),
        ),
        (
            "set_table_setting",
            Box::new(|| w.set_table_setting(t, "target_file_size", "1000")),
        ),
        (
            "create_snapshot",
            Box::new(|| w.create_snapshot().map(drop)),
        ),
    ];
    for (case, op) in cases {
        let before = l.state().await;
        l.guard.refuse_after(0);
        l.assert_refused(case, before, 0, op()).await;
    }
}

/// With a guard that approves, writes commit and read back as usual, through
/// both writers.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn approving_guard_commits_as_usual() {
    let (pool, base, _c) = spin_up_postgres().await.unwrap();
    let multi = Lake::multi(&pool, &base, "approved").await;
    let single = Lake::single(&pool, &base, "approved_single").await;
    for l in [&multi, &single] {
        let seeded = l.guard.calls();
        assert!(seeded > 0, "the guard ran on the seed writes");
        l.table_writer()
            .append_table("public", "t", &[batch(&[5], &[50])])
            .await
            .unwrap();
        assert!(l.guard.calls() > seeded, "the guard ran on the append");
        assert_eq!(
            l.rows("t").await,
            vec![(1, 10), (2, 20), (3, 30), (4, 40), (5, 50)]
        );
    }
    multi
        .sql("UPDATE approved.public.t SET val = 11 WHERE id = 1")
        .await
        .unwrap();
    multi
        .sql("DELETE FROM approved.public.t WHERE id = 2")
        .await
        .unwrap();
    assert_eq!(
        multi.rows("t").await,
        vec![(1, 11), (3, 30), (4, 40), (5, 50)]
    );
}

/// Records, from inside the transaction, the catalog head it sees, and the head
/// a separate connection sees at the same moment.
#[derive(Debug)]
struct ObservingGuard {
    pool: PgPool,
    catalog_id: i64,
    seen: Mutex<Vec<(i64, i64)>>,
}

#[async_trait::async_trait]
impl PostgresCommitGuard for ObservingGuard {
    async fn before_commit(&self, conn: &mut sqlx::PgConnection) -> Result<(), CommitGuardError> {
        let sql = "SELECT COALESCE(MAX(snapshot_id), 0) FROM ducklake_catalog_snapshot_map
                   WHERE catalog_id = $1";
        let inside: i64 = sqlx::query_scalar(sql)
            .bind(self.catalog_id)
            .fetch_one(&mut *conn)
            .await?;
        let outside: i64 = sqlx::query_scalar(sql)
            .bind(self.catalog_id)
            .fetch_one(&self.pool)
            .await?;
        self.seen.lock().unwrap().push((inside, outside));
        Ok(())
    }
}

/// The guard runs on the transaction's own connection after its writes: it
/// sees the snapshot the commit is about to publish, which no other connection
/// sees yet.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn guard_sees_the_uncommitted_transaction() {
    let (pool, base, _c) = spin_up_postgres().await.unwrap();
    let l = Lake::multi(&pool, &base, "observed").await;
    let guard = Arc::new(ObservingGuard {
        pool: pool.clone(),
        catalog_id: l.catalog_id.unwrap(),
        seen: Mutex::new(Vec::new()),
    });
    let writer = PostgresMetadataWriter::with_pool(pool.clone(), l.catalog_id.unwrap())
        .await
        .unwrap()
        .with_commit_guard(guard.clone());
    let head = l.head().await;
    let written = DuckLakeTableWriter::new(Arc::new(writer), l.os.clone())
        .unwrap()
        .append_table("public", "t", &[batch(&[5], &[50])])
        .await
        .unwrap();
    let (inside, outside) = *guard.seen.lock().unwrap().last().unwrap();
    assert_eq!(inside, written.snapshot_id, "the guard sees the new head");
    assert_eq!(outside, head, "no other connection sees it before COMMIT");
    assert!(written.snapshot_id > head);
}

/// Refuses unless `writer_lease` names `holder`, read `FOR SHARE`. When `hold`
/// is set, it signals `locked` and waits on `hold` before approving, so a test
/// can act while the guard's lock is held.
#[derive(Debug)]
struct LeaseGuard {
    holder: String,
    locked: Arc<tokio::sync::Notify>,
    hold: Mutex<Option<tokio::sync::oneshot::Receiver<()>>>,
}

#[async_trait::async_trait]
impl PostgresCommitGuard for LeaseGuard {
    async fn before_commit(&self, conn: &mut sqlx::PgConnection) -> Result<(), CommitGuardError> {
        let holder: Option<String> =
            sqlx::query_scalar("SELECT holder FROM writer_lease WHERE id = 1 FOR SHARE")
                .fetch_optional(&mut *conn)
                .await?;
        if holder.as_deref() != Some(self.holder.as_str()) {
            return Err(Box::new(LeaseLost));
        }
        let hold = self.hold.lock().unwrap().take();
        if let Some(hold) = hold {
            self.locked.notify_one();
            hold.await?;
        }
        Ok(())
    }
}

/// A row the guard reads `FOR SHARE` stays locked until the commit finishes, so
/// a takeover that updates it waits for the commit, and the next commit sees the
/// takeover and is refused.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn guard_row_lock_holds_until_commit() {
    let (pool, base, _c) = spin_up_postgres().await.unwrap();
    let l = Lake::multi(&pool, &base, "leased").await;
    sqlx::query("CREATE TABLE writer_lease (id INT PRIMARY KEY, holder TEXT NOT NULL)")
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query("INSERT INTO writer_lease VALUES (1, 'a')")
        .execute(&pool)
        .await
        .unwrap();

    let locked = Arc::new(tokio::sync::Notify::new());
    let guard = Arc::new(LeaseGuard {
        holder: "a".to_string(),
        locked: locked.clone(),
        hold: Mutex::new(None),
    });
    let writer = Arc::new(
        PostgresMetadataWriter::with_pool(pool.clone(), l.catalog_id.unwrap())
            .await
            .unwrap()
            .with_commit_guard(guard.clone()),
    );
    let table_writer = DuckLakeTableWriter::new(writer.clone(), l.os.clone()).unwrap();
    let mut session = table_writer
        .begin_write("public", "t", &schema(), WriteMode::Append)
        .unwrap();
    session.write_batch(&batch(&[5], &[50])).unwrap();
    // Held from here, so the held guard call is the commit's.
    let (release, hold) = tokio::sync::oneshot::channel();
    *guard.hold.lock().unwrap() = Some(hold);
    let write = tokio::spawn(session.finish());
    tokio::time::timeout(Duration::from_secs(60), locked.notified())
        .await
        .expect("the commit never reached the guard");

    let takeover_pool = pool.clone();
    let takeover = tokio::spawn(async move {
        sqlx::query("UPDATE writer_lease SET holder = 'b' WHERE id = 1")
            .execute(&takeover_pool)
            .await
    });
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        let waiting: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM pg_stat_activity
             WHERE query LIKE 'UPDATE writer_lease%' AND wait_event_type = 'Lock'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        if waiting > 0 {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "the takeover never waited on the lease row"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert!(!takeover.is_finished(), "the takeover waits for the commit");

    release.send(()).unwrap();
    let written = write.await.unwrap().unwrap();
    takeover.await.unwrap().unwrap();
    assert_eq!(l.head().await, written.snapshot_id);
    assert_eq!(
        l.rows("t").await,
        vec![(1, 10), (2, 20), (3, 30), (4, 40), (5, 50)]
    );

    // The lease moved on: the next commit is refused and changes nothing.
    let head = l.head().await;
    let files = files_under(&l.data);
    let error = table_writer
        .append_table("public", "t", &[batch(&[6], &[60])])
        .await
        .unwrap_err();
    assert!(
        matches!(&error, DuckLakeError::CommitRefused(source) if source.is::<LeaseLost>()),
        "refused with the guard's error: {error}"
    );
    assert_eq!(l.head().await, head);
    assert_eq!(files_under(&l.data), files);
}
