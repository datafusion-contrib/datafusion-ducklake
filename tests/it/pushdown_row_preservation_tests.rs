#![cfg(feature = "metadata-duckdb")]
//! Mechanical proof that catalog filter pushdown never changes a query's answer.
//!
//! # The property
//!
//! > Enabling pushdown must never change a query's answer.
//!
//! For every catalog state and every predicate this file generates, the same
//! SQL is run twice against the same catalog and the same Parquet:
//!
//! 1. with [`MetadataProvider::get_table_file_metadata_page_filtered`] ignoring
//!    its filter and delegating to the unfiltered page — the behaviour before
//!    this mechanism existed, and the ground truth, since the in-memory
//!    [`datafusion::physical_optimizer::pruning::PruningPredicate`] that then
//!    prunes has always been correct;
//! 2. with it honoured.
//!
//! The two row sets must be *identical*. Fewer rows under (2) is row loss;
//! more rows means the unfiltered path is wrong. That oracle is stronger than
//! comparing the catalog SQL against the in-memory pruning path, which is
//! deliberately blunt and would forbid legitimate improvements.
//!
//! # Two kinds of hostile catalog, and only one of them is a bug
//!
//! The property above holds unconditionally over a catalog whose statistics are
//! *true*, and over one whose statistics are text no conforming writer produces
//! — which must be declined, so the answer cannot move. It cannot hold over a
//! catalog whose statistics are well-formed and simply false about their files:
//! a `value_count` of 0 beside thirteen live values, a `DOUBLE` bound of `inf`
//! on a file holding `42.5`. Nothing prunes correctly from a lie, official
//! DuckLake included. [`Fidelity`] draws that line mechanically and says why;
//! the false-fact states stay in the sweep under the weaker assertions that are
//! actually true of them.
//!
//! A filter that silently does nothing would pass that alone, so each pair also
//! asserts the filtered listing returned **no more** catalog rows than the
//! unfiltered one, and the run reports how many pairs actually pruned. A clean
//! result is only worth its coverage, so the summary printed at the end of each
//! backend says how much of the space was explored and how much of it was
//! non-vacuous (predicate answers non-empty, listing actually narrowed).
//!
//! # Why a wrapper rather than a flag
//!
//! [`PushdownToggle`] is a `MetadataProvider` that forwards every method to a
//! real one and differs in exactly one: whether the filtered page honours its
//! filter. Nothing in the crate is modified, and the two runs are otherwise
//! byte-identical — same provider, same catalog rows, same files on disk.
//!
//! # What is generated
//!
//! **Real data first.** The fixture is written by official DuckLake through a
//! real DuckDB `ATTACH 'ducklake:...'`, so the Parquet holds genuine values and
//! `ducklake_file_column_stats` holds statistics a real writer produced. That
//! half proves normal operation: the unfiltered answer is a genuine oracle
//! because it is computed from the data.
//!
//! **Then poisoned statistics.** Every bug this feature's reviews found lived in
//! a statistics row no writer of this crate produces — a `nan` bound, a `today`
//! bound, a rounding timestamp. So the second half rewrites
//! `ducklake_file_column_stats` by raw SQL to simulate a catalog written by
//! another tool, and re-runs the same predicates. The Parquet is untouched, so
//! the true answer never moves; only what the catalog claims about it does.
//!
//! # Backends
//!
//! SQLite carries the bulk and needs no Docker. DuckDB runs the same sweep
//! against its native catalog. PostgreSQL and MySQL run a representative subset
//! through testcontainers — the dialects differ most on exactly these hazards,
//! since none of them has `TRY_CAST` and each mis-handles a malformed stat
//! differently.
//!
//! # Runtime
//!
//! The default tests run a representative slice. `#[ignore]`d tests named
//! `..._exhaustive` run the full cross product; run them with
//! `cargo test -- --ignored pushdown_row_preservation`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrow::record_batch::RecordBatch;
use arrow::util::display::{ArrayFormatter, FormatOptions};
use datafusion::catalog::CatalogProvider;
use datafusion::prelude::SessionContext;
use datafusion_ducklake::metadata_provider::{
    ColumnWithTable, DataFileChange, DeleteFileChange, DuckLakeFileMetadata, DuckLakeInlinedDelete,
    DuckLakeNameMapping, DuckLakeStatistics, DuckLakeTableColumn, DuckLakeTableField,
    DuckLakeTableFile, FileWithTable, SchemaMetadata, SnapshotMetadata, TableMetadata,
    TableWithSchema, ViewMetadata, ViewWithSchema,
};
use datafusion_ducklake::stats_filter::StatsFilter;
use datafusion_ducklake::{DuckLakeCatalog, MetadataProvider, Result};
use tempfile::TempDir;

use crate::common;

// ---------------------------------------------------------------------------
// The toggle: one `MetadataProvider` method's worth of difference
// ---------------------------------------------------------------------------

/// Forwards every `MetadataProvider` method to `inner`, and honours the
/// statistics filter only when `enabled`.
///
/// With `enabled == false` the filtered page falls back to
/// [`MetadataProvider::get_table_file_metadata_page`] — which is exactly the
/// trait's own default, i.e. the behaviour of a provider that predates the
/// mechanism. Every other method, including the unfiltered page, is the real
/// provider's.
///
/// `listed` accumulates the number of catalog rows the listing handed back
/// across the whole query, which is what makes "the filter actually filtered"
/// observable.
#[derive(Debug, Clone)]
struct PushdownToggle {
    inner: Arc<dyn MetadataProvider>,
    enabled: bool,
    listed: Arc<AtomicUsize>,
}

impl PushdownToggle {
    fn new(inner: Arc<dyn MetadataProvider>, enabled: bool) -> Self {
        Self {
            inner,
            enabled,
            listed: Arc::new(AtomicUsize::new(0)),
        }
    }
}

impl MetadataProvider for PushdownToggle {
    // The one method that differs.
    fn get_table_file_metadata_page_filtered(
        &self,
        table_id: i64,
        snapshot_id: i64,
        after_data_file_id: Option<i64>,
        limit: usize,
        filter: Option<&StatsFilter>,
    ) -> Result<Vec<DuckLakeFileMetadata>> {
        let page = if self.enabled {
            self.inner.get_table_file_metadata_page_filtered(
                table_id,
                snapshot_id,
                after_data_file_id,
                limit,
                filter,
            )?
        } else {
            self.inner.get_table_file_metadata_page(
                table_id,
                snapshot_id,
                after_data_file_id,
                limit,
            )?
        };
        self.listed.fetch_add(page.len(), Ordering::Relaxed);
        Ok(page)
    }

    // Everything else is pure delegation.
    fn get_current_snapshot(&self) -> Result<i64> {
        self.inner.get_current_snapshot()
    }
    fn get_data_path(&self) -> Result<String> {
        self.inner.get_data_path()
    }
    fn list_snapshots(&self) -> Result<Vec<SnapshotMetadata>> {
        self.inner.list_snapshots()
    }
    fn list_schemas(&self, snapshot_id: i64) -> Result<Vec<SchemaMetadata>> {
        self.inner.list_schemas(snapshot_id)
    }
    fn list_tables(&self, schema_id: i64, snapshot_id: i64) -> Result<Vec<TableMetadata>> {
        self.inner.list_tables(schema_id, snapshot_id)
    }
    fn list_views(&self, schema_id: i64, snapshot_id: i64) -> Result<Vec<ViewMetadata>> {
        self.inner.list_views(schema_id, snapshot_id)
    }
    fn get_table_structure(
        &self,
        table_id: i64,
        snapshot_id: i64,
    ) -> Result<Vec<DuckLakeTableColumn>> {
        self.inner.get_table_structure(table_id, snapshot_id)
    }
    fn get_table_fields(&self, table_id: i64, snapshot_id: i64) -> Result<Vec<DuckLakeTableField>> {
        self.inner.get_table_fields(table_id, snapshot_id)
    }
    fn get_name_mapping(&self, mapping_id: i64) -> Result<DuckLakeNameMapping> {
        self.inner.get_name_mapping(mapping_id)
    }
    fn get_table_files_for_select(
        &self,
        table_id: i64,
        snapshot_id: i64,
    ) -> Result<Vec<DuckLakeTableFile>> {
        self.inner.get_table_files_for_select(table_id, snapshot_id)
    }
    fn get_partition_spec(
        &self,
        table_id: i64,
        snapshot_id: i64,
    ) -> Result<Option<datafusion_ducklake::PartitionSpec>> {
        self.inner.get_partition_spec(table_id, snapshot_id)
    }
    fn get_sort_spec(
        &self,
        table_id: i64,
        snapshot_id: i64,
    ) -> Result<Option<datafusion_ducklake::SortSpec>> {
        self.inner.get_sort_spec(table_id, snapshot_id)
    }
    fn get_table_statistics(&self, table_id: i64, snapshot_id: i64) -> Result<DuckLakeStatistics> {
        self.inner.get_table_statistics(table_id, snapshot_id)
    }
    fn get_table_summary_statistics(
        &self,
        table_id: i64,
        snapshot_id: i64,
    ) -> Result<DuckLakeStatistics> {
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
        self.inner
            .get_table_file_metadata_page(table_id, snapshot_id, after_data_file_id, limit)
    }
    fn get_inlined_data(
        &self,
        table_id: i64,
        snapshot_id: i64,
        columns: &[DuckLakeTableColumn],
    ) -> Result<Vec<RecordBatch>> {
        self.inner.get_inlined_data(table_id, snapshot_id, columns)
    }
    fn get_inlined_deletes(
        &self,
        table_id: i64,
        snapshot_id: i64,
    ) -> Result<Vec<DuckLakeInlinedDelete>> {
        self.inner.get_inlined_deletes(table_id, snapshot_id)
    }
    fn get_table_row_count(&self, table_id: i64, snapshot_id: i64) -> Result<u64> {
        self.inner.get_table_row_count(table_id, snapshot_id)
    }
    fn get_schema_by_name(&self, name: &str, snapshot_id: i64) -> Result<Option<SchemaMetadata>> {
        self.inner.get_schema_by_name(name, snapshot_id)
    }
    fn get_table_by_name(
        &self,
        schema_id: i64,
        name: &str,
        snapshot_id: i64,
    ) -> Result<Option<TableMetadata>> {
        self.inner.get_table_by_name(schema_id, name, snapshot_id)
    }
    fn get_view_by_name(
        &self,
        schema_id: i64,
        name: &str,
        snapshot_id: i64,
    ) -> Result<Option<ViewMetadata>> {
        self.inner.get_view_by_name(schema_id, name, snapshot_id)
    }
    fn table_exists(&self, schema_id: i64, name: &str, snapshot_id: i64) -> Result<bool> {
        self.inner.table_exists(schema_id, name, snapshot_id)
    }
    fn list_all_tables(&self, snapshot_id: i64) -> Result<Vec<TableWithSchema>> {
        self.inner.list_all_tables(snapshot_id)
    }
    fn list_all_views(&self, snapshot_id: i64) -> Result<Vec<ViewWithSchema>> {
        self.inner.list_all_views(snapshot_id)
    }
    fn list_all_columns(&self, snapshot_id: i64) -> Result<Vec<ColumnWithTable>> {
        self.inner.list_all_columns(snapshot_id)
    }
    fn list_all_files(&self, snapshot_id: i64) -> Result<Vec<FileWithTable>> {
        self.inner.list_all_files(snapshot_id)
    }
    fn get_data_files_added_between_snapshots(
        &self,
        table_id: i64,
        start_snapshot: i64,
        end_snapshot: i64,
    ) -> Result<Vec<DataFileChange>> {
        self.inner
            .get_data_files_added_between_snapshots(table_id, start_snapshot, end_snapshot)
    }
    fn get_delete_files_added_between_snapshots(
        &self,
        table_id: i64,
        start_snapshot: i64,
        end_snapshot: i64,
    ) -> Result<Vec<DeleteFileChange>> {
        self.inner
            .get_delete_files_added_between_snapshots(table_id, start_snapshot, end_snapshot)
    }
}

// ---------------------------------------------------------------------------
// The fixture: real Parquet written by official DuckLake
// ---------------------------------------------------------------------------

/// One column of every type the encoder has a canonical text for, so that every
/// `try_cast` branch and every text-domain comparison is reachable.
const CREATE_TABLE: &str = "CREATE TABLE lake.t(
    id INTEGER,
    f DOUBLE,
    dec DECIMAL(18, 4),
    s VARCHAR,
    d DATE,
    ts_s TIMESTAMP_S,
    ts_ms TIMESTAMP_MS,
    ts_us TIMESTAMP,
    ts_ns TIMESTAMP_NS,
    tsz TIMESTAMPTZ,
    b BOOLEAN
)";

/// One `INSERT` per data file. Seven files, each chosen for a different shape of
/// statistics: two ordinary disjoint ranges, one with NULLs scattered through
/// it, one entirely NULL (so the writer records no bounds at all), one carrying
/// a NaN (so `contains_nan` is set for real), one single-row file (the shape
/// `NOT (min = C AND max = C)` can prune), and one whose string is longer than
/// the encoder will store.
const INSERTS: [&str; 7] = [
    // File 1: ids 1..3.
    "INSERT INTO lake.t VALUES
     (1, 1.5, 10.25, 'alpha', DATE '2020-01-01',
      '2020-01-01 00:00:00'::TIMESTAMP_S, '2020-01-01 00:00:00.123'::TIMESTAMP_MS,
      '2020-01-01 00:00:00.123456'::TIMESTAMP, '2020-01-01 00:00:00.123456789'::TIMESTAMP_NS,
      '2020-01-01 00:00:00+00'::TIMESTAMPTZ, true),
     (2, 2.5, 20.5, 'beta', DATE '2020-01-05',
      '2020-01-05 01:02:03'::TIMESTAMP_S, '2020-01-05 01:02:03.456'::TIMESTAMP_MS,
      '2020-01-05 01:02:03.456789'::TIMESTAMP, '2020-01-05 01:02:03.456789012'::TIMESTAMP_NS,
      '2020-01-05 01:02:03+00'::TIMESTAMPTZ, false),
     (3, 3.5, 30.75, 'gamma', DATE '2020-01-09',
      '2020-01-09 02:03:04'::TIMESTAMP_S, '2020-01-09 02:03:04.789'::TIMESTAMP_MS,
      '2020-01-09 02:03:04.789012'::TIMESTAMP, '2020-01-09 02:03:04.789012345'::TIMESTAMP_NS,
      '2020-01-09 02:03:04+00'::TIMESTAMPTZ, true)",
    // File 2: ids 11..13. The file most poison scenarios are aimed at.
    "INSERT INTO lake.t VALUES
     (11, 11.5, 110.25, 'kappa', DATE '2021-03-01',
      '2021-03-01 00:00:00'::TIMESTAMP_S, '2021-03-01 00:00:00.100'::TIMESTAMP_MS,
      '2021-03-01 00:00:00.100001'::TIMESTAMP, '2021-03-01 00:00:00.100000001'::TIMESTAMP_NS,
      '2021-03-01 00:00:00+00'::TIMESTAMPTZ, false),
     (12, 12.5, 120.5, 'lambda', DATE '2021-03-05',
      '2021-03-05 01:02:03'::TIMESTAMP_S, '2021-03-05 01:02:03.200'::TIMESTAMP_MS,
      '2021-03-05 01:02:03.200002'::TIMESTAMP, '2021-03-05 01:02:03.200000002'::TIMESTAMP_NS,
      '2021-03-05 01:02:03+00'::TIMESTAMPTZ, true),
     (13, 13.5, 130.75, 'mu', DATE '2021-03-09',
      '2021-03-09 02:03:04'::TIMESTAMP_S, '2021-03-09 02:03:04.300'::TIMESTAMP_MS,
      '2021-03-09 02:03:04.300003'::TIMESTAMP, '2021-03-09 02:03:04.300000003'::TIMESTAMP_NS,
      '2021-03-09 02:03:04+00'::TIMESTAMPTZ, false)",
    // File 3: ids 21..23, NULLs scattered so `null_count` is positive and the
    // bounds still exist.
    "INSERT INTO lake.t VALUES
     (21, NULL, 210.25, NULL, DATE '2022-06-01',
      '2022-06-01 00:00:00'::TIMESTAMP_S, NULL,
      '2022-06-01 00:00:00.000001'::TIMESTAMP, NULL,
      '2022-06-01 00:00:00+00'::TIMESTAMPTZ, NULL),
     (22, 22.5, NULL, 'nu', NULL,
      NULL, '2022-06-05 01:02:03.500'::TIMESTAMP_MS,
      NULL, '2022-06-05 01:02:03.500000005'::TIMESTAMP_NS,
      NULL, true),
     (23, 23.5, 230.75, 'xi', DATE '2022-06-09',
      '2022-06-09 02:03:04'::TIMESTAMP_S, '2022-06-09 02:03:04.600'::TIMESTAMP_MS,
      '2022-06-09 02:03:04.600006'::TIMESTAMP, '2022-06-09 02:03:04.600000006'::TIMESTAMP_NS,
      '2022-06-09 02:03:04+00'::TIMESTAMPTZ, false)",
    // File 4: every column but `id` is NULL, so the writer records no bounds and
    // `value_count` is 0. This is the file the `value_count` guard exists for.
    "INSERT INTO lake.t VALUES
     (31, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL)",
    // File 5: a real NaN, so `contains_nan` is set by a real writer.
    "INSERT INTO lake.t VALUES
     (41, 'NaN'::DOUBLE, 410.25, 'omicron', DATE '2023-09-01',
      '2023-09-01 00:00:00'::TIMESTAMP_S, '2023-09-01 00:00:00.700'::TIMESTAMP_MS,
      '2023-09-01 00:00:00.700007'::TIMESTAMP, '2023-09-01 00:00:00.700000007'::TIMESTAMP_NS,
      '2023-09-01 00:00:00+00'::TIMESTAMPTZ, true),
     (42, 42.5, 420.5, 'pi', DATE '2023-09-05',
      '2023-09-05 01:02:03'::TIMESTAMP_S, '2023-09-05 01:02:03.800'::TIMESTAMP_MS,
      '2023-09-05 01:02:03.800008'::TIMESTAMP, '2023-09-05 01:02:03.800000008'::TIMESTAMP_NS,
      '2023-09-05 01:02:03+00'::TIMESTAMPTZ, false)",
    // File 6: a single row, so min = max on every column, and a string holding
    // both a quote and a backslash — the two characters a dialect can give a
    // second meaning inside a literal.
    "INSERT INTO lake.t VALUES
     (51, 51.5, 510.25, 'quote''and\\backslash', DATE '2024-12-31',
      '2024-12-31 23:59:59'::TIMESTAMP_S, '2024-12-31 23:59:59.900'::TIMESTAMP_MS,
      '2024-12-31 23:59:59.900009'::TIMESTAMP, '2024-12-31 23:59:59.900000009'::TIMESTAMP_NS,
      '2024-12-31 23:59:59+00'::TIMESTAMPTZ, true)",
    // File 7: a string past the length a bound is stored at, so the recorded
    // bound is a truncation of the real value rather than the value.
    "INSERT INTO lake.t VALUES
     (61, 61.5, 610.25, repeat('z', 3000), DATE '2025-02-14',
      '2025-02-14 12:00:00'::TIMESTAMP_S, '2025-02-14 12:00:00.010'::TIMESTAMP_MS,
      '2025-02-14 12:00:00.010010'::TIMESTAMP, '2025-02-14 12:00:00.010000010'::TIMESTAMP_NS,
      '2025-02-14 12:00:00+00'::TIMESTAMPTZ, NULL)",
];

/// Run `statements` against `attach_target` through official DuckLake.
///
/// `DATA_INLINING_ROW_LIMIT 0` keeps the single-row `INSERT`s data files rather
/// than catalog-inlined rows; inlined data is a separate read path that this
/// mechanism does not touch. `TimeZone` is pinned so the `TIMESTAMPTZ` bounds
/// are written in UTC regardless of where the suite runs.
fn with_official_ducklake(
    attach_target: &str,
    extensions: &[&str],
    data_path: &Path,
    statements: &[&str],
) -> anyhow::Result<()> {
    common::ensure_ducklake_installed();
    for extension in extensions {
        common::ensure_extension_installed(extension);
    }
    std::fs::create_dir_all(data_path)?;

    let conn = duckdb::Connection::open_in_memory()?;
    for extension in extensions {
        conn.execute(&format!("LOAD {extension};"), [])?;
    }
    conn.execute("LOAD ducklake;", [])?;
    conn.execute("SET TimeZone='UTC';", [])?;
    conn.execute(
        &format!(
            "ATTACH '{attach_target}' AS lake (DATA_PATH '{}', DATA_INLINING_ROW_LIMIT 0);",
            data_path.display()
        ),
        [],
    )?;
    for statement in statements {
        conn.execute(statement, [])?;
    }
    conn.execute("DETACH lake;", [])?;
    Ok(())
}

fn build_fixture(attach_target: &str, extensions: &[&str], data_path: &Path) -> anyhow::Result<()> {
    let mut statements = vec![CREATE_TABLE];
    statements.extend_from_slice(&INSERTS);
    with_official_ducklake(attach_target, extensions, data_path, &statements)
}

// ---------------------------------------------------------------------------
// Catalog stores: the four dialects, behind the two operations this test needs
// ---------------------------------------------------------------------------

/// Where the catalog lives, and how to rewrite its statistics.
///
/// The test needs exactly two things from a backend: open a `MetadataProvider`
/// on it, and run arbitrary DDL/DML against `ducklake_file_column_stats`.
enum CatalogStore {
    DuckDb(PathBuf),
    #[cfg(feature = "metadata-sqlite")]
    Sqlite(String),
    #[cfg(feature = "metadata-postgres")]
    Postgres(String),
    #[cfg(feature = "metadata-mysql")]
    MySql(String),
}

impl CatalogStore {
    fn label(&self) -> &'static str {
        match self {
            Self::DuckDb(_) => "duckdb",
            #[cfg(feature = "metadata-sqlite")]
            Self::Sqlite(_) => "sqlite",
            #[cfg(feature = "metadata-postgres")]
            Self::Postgres(_) => "postgres",
            #[cfg(feature = "metadata-mysql")]
            Self::MySql(_) => "mysql",
        }
    }

    /// Run one statement, failing the test if it errors.
    async fn exec(&self, sql: &str) {
        self.try_exec(sql)
            .await
            .unwrap_or_else(|e| panic!("{}: `{sql}`: {e}", self.label()));
    }

    /// Run several statements over one connection.
    ///
    /// Restoring the statistics table and then poisoning it is three or four
    /// statements per scenario, times a couple of hundred scenarios. On DuckDB
    /// each connection opens and checkpoints the catalog file, so reusing one
    /// is most of this sweep's runtime.
    async fn exec_all(&self, statements: &[String]) {
        match self {
            Self::DuckDb(path) => {
                let conn = duckdb::Connection::open(path)
                    .unwrap_or_else(|e| panic!("duckdb: open {}: {e}", path.display()));
                for sql in statements {
                    conn.execute_batch(sql)
                        .unwrap_or_else(|e| panic!("duckdb: `{sql}`: {e}"));
                }
            },
            _ => {
                for sql in statements {
                    self.exec(sql).await;
                }
            },
        }
    }

    async fn try_exec(&self, sql: &str) -> anyhow::Result<()> {
        match self {
            Self::DuckDb(path) => {
                let conn = duckdb::Connection::open(path)?;
                conn.execute_batch(sql)?;
                Ok(())
            },
            #[cfg(feature = "metadata-sqlite")]
            Self::Sqlite(url) => {
                let pool = sqlx::SqlitePool::connect(url).await?;
                let mut conn = pool.acquire().await?;
                let result = sqlx::query(sqlx::AssertSqlSafe(sql.to_string()))
                    .execute(&mut *conn)
                    .await
                    .map(|_| ());
                drop(conn);
                pool.close().await;
                Ok(result?)
            },
            #[cfg(feature = "metadata-postgres")]
            Self::Postgres(url) => {
                let pool = sqlx::PgPool::connect(url).await?;
                let mut conn = pool.acquire().await?;
                let result = sqlx::query(sqlx::AssertSqlSafe(sql.to_string()))
                    .execute(&mut *conn)
                    .await
                    .map(|_| ());
                drop(conn);
                pool.close().await;
                Ok(result?)
            },
            #[cfg(feature = "metadata-mysql")]
            Self::MySql(url) => {
                let pool = sqlx::MySqlPool::connect(url).await?;
                let mut conn = pool.acquire().await?;
                let result = sqlx::query(sqlx::AssertSqlSafe(sql.to_string()))
                    .execute(&mut *conn)
                    .await
                    .map(|_| ());
                drop(conn);
                pool.close().await;
                Ok(result?)
            },
        }
    }

    async fn open(&self) -> Arc<dyn MetadataProvider> {
        match self {
            Self::DuckDb(path) => Arc::new(
                datafusion_ducklake::DuckdbMetadataProvider::new(
                    path.to_string_lossy().to_string(),
                )
                .expect("duckdb provider"),
            ) as Arc<dyn MetadataProvider>,
            #[cfg(feature = "metadata-sqlite")]
            Self::Sqlite(url) => Arc::new(
                datafusion_ducklake::SqliteMetadataProvider::new(url)
                    .await
                    .expect("sqlite provider"),
            ) as Arc<dyn MetadataProvider>,
            #[cfg(feature = "metadata-postgres")]
            Self::Postgres(url) => Arc::new(
                datafusion_ducklake::PostgresMetadataProvider::new(url)
                    .await
                    .expect("postgres provider"),
            ) as Arc<dyn MetadataProvider>,
            #[cfg(feature = "metadata-mysql")]
            Self::MySql(url) => Arc::new(
                datafusion_ducklake::MySqlMetadataProvider::new(url)
                    .await
                    .expect("mysql provider"),
            ) as Arc<dyn MetadataProvider>,
        }
    }

    /// Bring a catalog written by the DuckLake extension that ships with
    /// `duckdb` 1.4.1 — the version this fixture is written by — up to the shape
    /// a released DuckDB 1.5.x writes.
    ///
    /// Only `DuckdbMetadataProvider` probes for the newer catalog columns. The
    /// three SQL providers select `ducklake_column.default_value_type` /
    /// `default_value_dialect` and `ducklake_schema_versions.table_id`
    /// unconditionally, so on the older shape they fail the query outright.
    /// `official_pushdown_parity_tests` and `compaction_sqlite_tests` top up the
    /// same two tables for the same reason. Nothing here touches
    /// `ducklake_data_file` or `ducklake_file_column_stats`: every statistic the
    /// sweep reads is exactly as official wrote it.
    async fn migrate(&self) {
        for statement in [
            "ALTER TABLE ducklake_column ADD COLUMN default_value_type VARCHAR(255)",
            "ALTER TABLE ducklake_column ADD COLUMN default_value_dialect VARCHAR(255)",
            "ALTER TABLE ducklake_schema_versions ADD COLUMN table_id BIGINT",
            "UPDATE ducklake_schema_versions SET table_id =
             (SELECT table_id FROM ducklake_table WHERE table_name = 't')",
        ] {
            // A column a newer extension already wrote is not an error here.
            let _ = self.try_exec(statement).await;
        }
    }

    /// `contains_nan` is a real boolean on DuckDB and PostgreSQL and an integer
    /// on SQLite and MySQL.
    fn bool_literal(&self, value: bool) -> &'static str {
        match self {
            Self::DuckDb(_) => {
                if value {
                    "true"
                } else {
                    "false"
                }
            },
            #[cfg(feature = "metadata-postgres")]
            Self::Postgres(_) => {
                if value {
                    "true"
                } else {
                    "false"
                }
            },
            #[cfg(feature = "metadata-sqlite")]
            Self::Sqlite(_) => {
                if value {
                    "1"
                } else {
                    "0"
                }
            },
            #[cfg(feature = "metadata-mysql")]
            Self::MySql(_) => {
                if value {
                    "1"
                } else {
                    "0"
                }
            },
        }
    }

    /// A SQL string literal for the poison text.
    ///
    /// MySQL gives `\` a second meaning inside a quoted string unless
    /// `NO_BACKSLASH_ESCAPES` is set, so it has to be doubled there and nowhere
    /// else — the same asymmetry the MySQL dialect's `quote_literal` override
    /// carries.
    fn quote(&self, text: &str) -> String {
        let escaped = text.replace('\'', "''");
        match self {
            #[cfg(feature = "metadata-mysql")]
            Self::MySql(_) => format!("'{}'", escaped.replace('\\', "\\\\")),
            _ => format!("'{escaped}'"),
        }
    }
}

// ---------------------------------------------------------------------------
// Predicates
// ---------------------------------------------------------------------------

/// A predicate, tagged with the column whose statistics it reads.
///
/// The tag is what pairs a predicate with the poison scenarios that can affect
/// it: poisoning `f`'s bounds is only detectable through a predicate on `f`.
/// `MULTI` predicates span columns and are run against every scenario.
struct Predicate {
    tag: &'static str,
    sql: String,
}

const MULTI: &str = "*";

/// Every column's operator sweep plus the cross-column combinations.
///
/// Timestamp literals go through `arrow_cast` so the constant carries the
/// column's exact type: a bare `TIMESTAMP '...'` is nanoseconds in DataFusion,
/// which would cast the *column* instead and leave the subject something other
/// than a bare column reference — no pushdown at all, and no coverage.
fn predicates() -> Vec<Predicate> {
    let mut all: Vec<Predicate> = Vec::new();
    macro_rules! push {
        ($tag:expr, $sql:expr) => {
            all.push(Predicate {
                tag: $tag,
                sql: $sql.to_string(),
            })
        };
    }

    // Integer.
    push!("id", "id = 12");
    push!("id", "id <> 12");
    push!("id", "id < 12");
    push!("id", "id <= 12");
    push!("id", "id > 12");
    push!("id", "id >= 12");
    push!("id", "id IN (2, 12, 22, 51)");
    push!("id", "id IS NULL");
    push!("id", "id IS NOT NULL");

    // Float, including the NaN-bearing file.
    push!("f", "f = 12.5");
    push!("f", "f <> 12.5");
    push!("f", "f < 12.5");
    push!("f", "f <= 12.5");
    push!("f", "f > 12.5");
    push!("f", "f >= 12.5");
    push!("f", "f IN (2.5, 12.5, 42.5)");
    push!("f", "f IS NULL");
    push!("f", "f IS NOT NULL");

    // Decimal.
    push!("dec", "dec = arrow_cast('120.5', 'Decimal128(18, 4)')");
    push!("dec", "dec <> arrow_cast('120.5', 'Decimal128(18, 4)')");
    push!("dec", "dec < arrow_cast('120.5', 'Decimal128(18, 4)')");
    push!("dec", "dec <= arrow_cast('120.5', 'Decimal128(18, 4)')");
    push!("dec", "dec > arrow_cast('120.5', 'Decimal128(18, 4)')");
    push!("dec", "dec >= arrow_cast('120.5', 'Decimal128(18, 4)')");
    push!(
        "dec",
        "dec IN (arrow_cast('20.5', 'Decimal128(18, 4)'), arrow_cast('120.5', 'Decimal128(18, 4)'))"
    );
    push!("dec", "dec IS NULL");
    push!("dec", "dec IS NOT NULL");

    // String: the raw, uncast, collation-sensitive path.
    push!("s", "s = 'lambda'");
    push!("s", "s <> 'lambda'");
    push!("s", "s < 'lambda'");
    push!("s", "s <= 'lambda'");
    push!("s", "s > 'lambda'");
    push!("s", "s >= 'lambda'");
    push!("s", "s IN ('beta', 'lambda', 'xi')");
    push!("s", "s = 'quote''and\\backslash'");
    push!("s", "s IS NULL");
    push!("s", "s IS NOT NULL");

    // Date.
    push!("d", "d = DATE '2021-03-05'");
    push!("d", "d <> DATE '2021-03-05'");
    push!("d", "d < DATE '2021-03-05'");
    push!("d", "d <= DATE '2021-03-05'");
    push!("d", "d > DATE '2021-03-05'");
    push!("d", "d >= DATE '2021-03-05'");
    push!("d", "d IN (DATE '2020-01-05', DATE '2021-03-05')");
    push!("d", "d IS NULL");
    push!("d", "d IS NOT NULL");

    // Timestamps at every unit, and one with a zone.
    for (tag, arrow_type, value) in [
        ("ts_s", "Timestamp(Second, None)", "2021-03-05T01:02:03"),
        (
            "ts_ms",
            "Timestamp(Millisecond, None)",
            "2021-03-05T01:02:03.200",
        ),
        (
            "ts_us",
            "Timestamp(Microsecond, None)",
            "2021-03-05T01:02:03.200002",
        ),
        (
            "ts_ns",
            "Timestamp(Nanosecond, None)",
            "2021-03-05T01:02:03.200000002",
        ),
        (
            "tsz",
            "Timestamp(Microsecond, Some(\"UTC\"))",
            "2021-03-05T01:02:03Z",
        ),
    ] {
        let literal = format!("arrow_cast('{value}', '{arrow_type}')");
        for op in ["=", "<>", "<", "<=", ">", ">="] {
            push!(tag, format!("{tag} {op} {literal}"));
        }
        push!(tag, format!("{tag} IS NULL"));
        push!(tag, format!("{tag} IS NOT NULL"));
    }

    // Boolean.
    push!("b", "b = true");
    push!("b", "b <> true");
    push!("b", "b = false");
    push!("b", "b IS NULL");
    push!("b", "b IS NOT NULL");

    // Cross-column AND / OR, and the shapes G10 is about.
    push!(MULTI, "id = 12 AND f > 2.0");
    push!(MULTI, "id = 12 OR f = 42.5");
    push!(MULTI, "s = 'lambda' AND d >= DATE '2021-01-01'");
    push!(MULTI, "d < DATE '2021-01-01' OR id > 40");
    push!(MULTI, "(id < 5 AND f > 1.0) OR s IS NULL");
    push!(MULTI, "id IN (1, 51) AND b = true");
    push!(MULTI, "f = 12.5 OR dec IS NULL");
    push!(MULTI, "NOT (id = 12)");
    push!(MULTI, "id > 5 AND id < 30 AND s <> 'nu'");
    push!(MULTI, "id = 12 OR upper(s) = 'NU'");

    all
}

// ---------------------------------------------------------------------------
// The two kinds of hostile statistic, and why only one of them is a bug
// ---------------------------------------------------------------------------

/// What a poisoned statistic is doing wrong, which decides what may be asserted
/// about it.
///
/// Not every catalog a foreign tool could write is one pruning can survive, and
/// conflating the two would either hide bugs or manufacture them. The line is:
///
/// > Is every statistic in this state a value [`datafusion_ducklake::stats_encode`]
/// > writes, *and* one the comparison path is contracted to use?
///
/// **No — [`Fidelity::HostileEncoding`].** The statistic is text no writer of
/// this crate produces (`nan` for a float, `today` for a date, `2020-02-31`,
/// a padded ` 2020-01-01 `, a `.50` fraction the encoder trims), or a row that
/// is absent or incomplete. `stats_filter`'s module docs commit to declining
/// exactly these — "every dialect admits only the shapes `stats_encode` writes",
/// because an engine's input function is permissive by design and will read
/// `nan`, `epoch` or `0x10` into a real value that then prunes. `table.rs`'s
/// `parse_statistic_scalar` makes the same promise on the in-memory side, and
/// `column_stats_tests` pins it. So the file must be kept, the answer must not
/// move, and **the full property applies**.
///
/// **Yes — [`Fidelity::FalseFact`].** The statistic is well-formed and simply
/// lies: `value_count = 0` on a file with thirteen non-NULL values,
/// `min = max = 'infinity'` on a `VARCHAR` column holding `alpha`,
/// `min = max = 'inf'` on a `DOUBLE` column holding `42.5`. Nothing can defend
/// against this. `value_count` is defined by the DuckLake format as the count of
/// non-NULL values for that column in that file; a writer that sets it to 0 with
/// rows present has broken the format, and official DuckLake prunes those files
/// too — its `(value_count IS NULL OR value_count > 0)` guard is the one this
/// crate ports. If a recorded bound may be false about its file then
/// statistics-based pruning is impossible in principle, which is the whole
/// mechanism under test.
///
/// These states are kept rather than deleted, because they are still worth
/// something: they assert what *is* true of a lying catalog. Pushdown may drop
/// rows, but it must not raise, must not invent a row the unfiltered path did
/// not return, and must not widen the listing. See [`check_pair`].
///
/// The classification is mechanical — [`is_canonical_for`] — not a list of the
/// cases that happened to fail, which is the distinction between splitting a
/// property and weakening one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Fidelity {
    /// The catalog holds something no conforming writer wrote. Must fail open.
    HostileEncoding,
    /// The catalog holds a well-formed statistic that is false about its file.
    FalseFact,
}

impl Fidelity {
    fn label(self) -> &'static str {
        match self {
            Self::HostileEncoding => "hostile-encoding",
            Self::FalseFact => "false-fact",
        }
    }
}

/// A column's type, as far as deciding what text is a canonical encoding for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ColumnKind {
    Integer,
    Float,
    Decimal,
    Text,
    Date,
    Timestamp,
    TimestampTz,
    Boolean,
}

fn column_kind(tag: &str) -> ColumnKind {
    match tag {
        "id" => ColumnKind::Integer,
        "f" => ColumnKind::Float,
        "dec" => ColumnKind::Decimal,
        "s" => ColumnKind::Text,
        "d" => ColumnKind::Date,
        "ts_s" | "ts_ms" | "ts_us" | "ts_ns" => ColumnKind::Timestamp,
        "tsz" => ColumnKind::TimestampTz,
        "b" => ColumnKind::Boolean,
        other => panic!("no column kind for tag `{other}`"),
    }
}

/// Whether `text` is a bound `stats_encode` could have written for a column of
/// `kind`, *and* one the comparison path undertakes to use.
///
/// The second half matters for the temporal types. `chrono` renders a year past
/// 9999 as `+12921-08-18` and one before the common era as `-0044-03-15`, so
/// those are reachable encodings — but `stats_filter` declines them rather than
/// mis-ordering them (a sign prefix sorts below every digit), and a declined
/// stat must keep its file. Treating them as hostile is therefore the stricter
/// classification, which is the one to take when the two readings disagree.
fn is_canonical_for(kind: ColumnKind, text: &str) -> bool {
    match kind {
        // `stats_encode` passes `Utf8` through verbatim, so any string at all is
        // one this column could truthfully carry — including `infinity`, `nan`
        // and the empty string. A raw text bound is compared byte-wise with no
        // parsing, so there is nothing here that could be mis-read.
        ColumnKind::Text => true,
        ColumnKind::Boolean => matches!(text, "true" | "false"),
        ColumnKind::Integer => is_plain_integer(text),
        ColumnKind::Decimal => is_plain_decimal(text),
        // The encoder writes `inf` / `-inf` for the infinities and has no text
        // for NaN at all, so `nan` and `NaN` are hostile while `inf` is not.
        ColumnKind::Float => matches!(text, "inf" | "-inf") || is_plain_decimal(text),
        // Shape, not calendar validity. A bound is not required to be a value
        // any row holds — the DuckLake format says `min_value` / `max_value`
        // "do not have to be exact" — so `2020-02-31` is a well-ordered *bound*
        // even though no row can equal it: as text it sorts between
        // `2020-02-30` and `2020-03-01`, exactly where a February 31st would.
        // A file whose earliest date is 2020-03-01 may legitimately record it
        // as a loose minimum, and pruning on it is correct.
        //
        // So an impossible calendar day is a false *fact* when it is wrong
        // about its file, not a hostile *encoding*: nothing mis-parses and
        // nothing mis-orders. What must stay hostile is a spelling the encoder
        // never emits — `' 2020-01-01 '`, `2020/01/01`, `20200101` — because
        // those can be perfectly truthful about a file and still be read as a
        // different value than the bytes say. MySQL's `DATE` parser normalises
        // all three, which is the bug this sweep found.
        ColumnKind::Date => matches_date_shape(text),
        ColumnKind::Timestamp => is_canonical_timestamp_text(text, ""),
        ColumnKind::TimestampTz => is_canonical_timestamp_text(text, "+00"),
    }
}

fn is_plain_integer(text: &str) -> bool {
    let digits = text.strip_prefix('-').unwrap_or(text);
    !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit())
}

fn is_plain_decimal(text: &str) -> bool {
    let body = text.strip_prefix('-').unwrap_or(text);
    let (whole, fraction) = body.split_once('.').unwrap_or((body, "1"));
    !whole.is_empty()
        && !fraction.is_empty()
        && whole.bytes().all(|b| b.is_ascii_digit())
        && fraction.bytes().all(|b| b.is_ascii_digit())
}

/// Whether `text` is the fixed-width `YYYY-MM-DD` shape `stats_encode` writes,
/// regardless of whether that day exists.
///
/// Fixed width is what makes the byte-wise comparison sound: every field is
/// zero-padded to the same length, so lexicographic order is chronological
/// order. An out-of-range field still sorts where a loose bound belongs, which
/// is why calendar validity is not checked here — see [`is_canonical_for`].
fn matches_date_shape(text: &str) -> bool {
    let bytes = text.as_bytes();
    bytes.len() == 10
        && bytes[4] == b'-'
        && bytes[7] == b'-'
        && [0, 1, 2, 3, 5, 6, 8, 9]
            .iter()
            .all(|&i| bytes[i].is_ascii_digit())
}

/// Whether `text` is the `YYYY-MM-DD HH:MM:SS[.f]` shape `stats_encode` writes,
/// plus `suffix`.
///
/// Shape only, for the reason [`is_canonical_for`] gives for dates: a bound need
/// not be a value, so `2020-02-31 25:00:00` is still a well-ordered loose bound
/// and pruning on it is correct.
///
/// The fraction is the exception, and it is the one rule here that is about
/// *ordering* rather than spelling. `stats_encode` trims trailing zeros, so it
/// never writes `.50`; and as text `.50` sorts above `.5` while naming the same
/// instant. A bound spelled that way would be read as later than it is, so it is
/// hostile — the same category as ` 2020-01-01 `, and for the same reason.
fn is_canonical_timestamp_text(text: &str, suffix: &str) -> bool {
    let Some(body) = text.strip_suffix(suffix) else {
        return false;
    };
    let Some((date, time)) = body.split_once(' ') else {
        return false;
    };
    if !matches_date_shape(date) {
        return false;
    }
    let clock = match time.split_once('.') {
        Some((clock, fraction)) => {
            // A separator with nothing after it, a non-digit, or a trailing
            // zero is not a fraction this encoder produces.
            if fraction.is_empty()
                || fraction.ends_with('0')
                || !fraction.bytes().all(|b| b.is_ascii_digit())
            {
                return false;
            }
            clock
        },
        None => time,
    };
    let parts: Vec<&str> = clock.split(':').collect();
    parts.len() == 3
        && parts
            .iter()
            .all(|part| part.len() == 2 && part.bytes().all(|b| b.is_ascii_digit()))
}

// ---------------------------------------------------------------------------
// Poison scenarios
// ---------------------------------------------------------------------------

/// Text no writer of this crate produces, simulating a catalog written by
/// another tool.
///
/// Each of these is accepted by at least one engine's input function and reads
/// as a value that would then prune: DuckDB reads `nan`, `epoch` and `0x10`;
/// PostgreSQL's `pg_input_is_valid` additionally accepts `today`, `now`,
/// `infinity` and `NaN`. `2020-02-31` is a date no calendar has; `+12921-08-18`
/// and `-0044-03-15` are years chrono signs, which sort below every digit;
/// `2023-11-14 22:13:20.50` has a trailing zero the encoder trims, and
/// `...+01` an offset it never writes.
const FOREIGN_STATS: [(&str, &str); 20] = [
    ("nan", "nan"),
    ("NaN", "NaN"),
    ("inf", "inf"),
    ("-inf", "-inf"),
    ("today", "today"),
    ("now", "now"),
    ("epoch", "epoch"),
    ("infinity", "infinity"),
    ("0x10", "0x10"),
    ("plus_5", "+5"),
    ("padded_date", " 2020-01-01 "),
    ("impossible_date", "2020-02-31"),
    ("year_past_9999", "+12921-08-18"),
    ("year_bce", "-0044-03-15"),
    ("year_zero", "0000-01-01"),
    ("trailing_zero_fraction", "2023-11-14 22:13:20.50"),
    ("offset_plus_one", "2023-11-14 22:13:20+01"),
    ("empty", ""),
    ("quote_and_backslash", "a'b\\c"),
    // Filled in at build time with a 4096-character string.
    ("very_long", ""),
];

/// A rewrite of `ducklake_file_column_stats`.
struct Poison {
    label: String,
    /// The column whose statistics are rewritten, by tag.
    tag: &'static str,
    /// What may be asserted about this state — see [`Fidelity`].
    fidelity: Fidelity,
    statements: Vec<String>,
}

/// Every statistics state, for one column.
///
/// `file` scopes the rewrite: `None` hits every file, which makes the hazard
/// visible on whichever file happens to hold the matching rows; `Some(id)` hits
/// one, which additionally proves the *other* files still prune normally.
fn poisons_for_column(
    store: &CatalogStore,
    tag: &'static str,
    column_id: i64,
    file: Option<i64>,
    exhaustive: bool,
) -> Vec<Poison> {
    let scope = match file {
        Some(id) => format!(" AND data_file_id = {id}"),
        None => String::new(),
    };
    let scope_label = match file {
        Some(_) => "one_file",
        None => "all_files",
    };
    let update = |assignments: &str| {
        vec![format!(
            "UPDATE ducklake_file_column_stats SET {assignments} \
             WHERE column_id = {column_id}{scope}"
        )]
    };
    let mut out = Vec::new();
    let mut push = |name: &str, fidelity: Fidelity, statements: Vec<String>| {
        out.push(Poison {
            label: format!("{tag}/{scope_label}/{name}"),
            tag,
            fidelity,
            statements,
        })
    };

    // Structural states. An absent row and a NULL stat are *incomplete*, not
    // false: they assert nothing, so nothing may be pruned on them and the full
    // property applies. A count or a NaN flag that contradicts the file is a
    // well-formed claim that happens to be a lie, which no pruner can survive —
    // `value_count = 0` beside a non-NULL bound is the DuckLake format's own
    // "this column has no values in this file", and official prunes on it too.
    use Fidelity::{FalseFact, HostileEncoding};
    push(
        "no_stats_row",
        HostileEncoding,
        vec![format!(
            "DELETE FROM ducklake_file_column_stats WHERE column_id = {column_id}{scope}"
        )],
    );
    push("min_null", HostileEncoding, update("min_value = NULL"));
    push("max_null", HostileEncoding, update("max_value = NULL"));
    push(
        "bounds_null",
        HostileEncoding,
        update("min_value = NULL, max_value = NULL"),
    );
    push("value_count_zero", FalseFact, update("value_count = 0"));
    push(
        "value_count_null",
        HostileEncoding,
        update("value_count = NULL"),
    );
    push("null_count_zero", FalseFact, update("null_count = 0"));
    push("null_count_positive", FalseFact, update("null_count = 7"));
    push(
        "null_count_null",
        HostileEncoding,
        update("null_count = NULL"),
    );
    push(
        "contains_nan_null",
        HostileEncoding,
        update("contains_nan = NULL"),
    );
    push(
        "contains_nan_true",
        FalseFact,
        update(&format!("contains_nan = {}", store.bool_literal(true))),
    );
    push(
        "contains_nan_false",
        FalseFact,
        update(&format!("contains_nan = {}", store.bool_literal(false))),
    );

    // Foreign bound text. Both bounds together is the sharp case — it names an
    // interval a naive cast reads as a real one — and min alone leaves a
    // half-valid row, which is where a missing `IS NULL` disjunct would show.
    let long = "y".repeat(4096);
    let kind = column_kind(tag);
    for (name, text) in FOREIGN_STATS {
        let text = if name == "very_long" {
            &long
        } else {
            text
        };
        // The same text is hostile on one column and an ordinary value on
        // another: `infinity` is a string a `VARCHAR` column can hold and is
        // nothing a `DATE` column's encoder writes.
        let fidelity = if is_canonical_for(kind, text) {
            FalseFact
        } else {
            HostileEncoding
        };
        let quoted = store.quote(text);
        push(
            &format!("both_bounds_{name}"),
            fidelity,
            update(&format!("min_value = {quoted}, max_value = {quoted}")),
        );
        if exhaustive || matches!(name, "nan" | "today" | "0x10" | "empty" | "impossible_date") {
            push(
                &format!("min_only_{name}"),
                fidelity,
                update(&format!("min_value = {quoted}")),
            );
        }
    }

    out
}

// ---------------------------------------------------------------------------
// Running one pair
// ---------------------------------------------------------------------------

/// The result of one query: the rows, and how many catalog rows the listing
/// returned to produce them.
struct Run {
    rows: Vec<String>,
    listed: usize,
}

/// Every row of `batches` as a sorted `Vec<String>`, so two answers compare
/// without depending on partition order.
fn rows(batches: &[RecordBatch]) -> Vec<String> {
    let options = FormatOptions::default().with_null("NULL");
    let mut out = Vec::new();
    for batch in batches {
        let formatters: Vec<_> = batch
            .columns()
            .iter()
            .map(|column| {
                ArrayFormatter::try_new(column.as_ref(), &options).expect("array is formattable")
            })
            .collect();
        for row in 0..batch.num_rows() {
            out.push(
                formatters
                    .iter()
                    .map(|f| f.value(row).to_string())
                    .collect::<Vec<_>>()
                    .join("\u{1}"),
            );
        }
    }
    out.sort();
    out
}

async fn run_query(
    inner: &Arc<dyn MetadataProvider>,
    enabled: bool,
    sql: &str,
) -> std::result::Result<Run, String> {
    let toggle = PushdownToggle::new(Arc::clone(inner), enabled);
    let listed = Arc::clone(&toggle.listed);
    let catalog = DuckLakeCatalog::new(toggle).map_err(|e| format!("catalog: {e}"))?;
    let ctx = SessionContext::new();
    ctx.register_catalog("lake", Arc::new(catalog) as Arc<dyn CatalogProvider>);
    let batches = ctx
        .sql(sql)
        .await
        .map_err(|e| format!("plan: {e}"))?
        .collect()
        .await
        .map_err(|e| format!("execute: {e}"))?;
    Ok(Run {
        rows: rows(&batches),
        listed: listed.load(Ordering::Relaxed),
    })
}

/// What one (catalog state, predicate) pair told us.
#[derive(Default)]
struct Tally {
    pairs: usize,
    /// Pairs whose ground-truth answer was non-empty. A pair with an empty
    /// answer cannot witness row loss, so this is the number that matters.
    non_empty: usize,
    /// Pairs where the filtered listing returned strictly fewer catalog rows.
    /// A scenario that never prunes proves nothing about pruning.
    pruned: usize,
    /// Pairs over a [`Fidelity::FalseFact`] state whose answer did change.
    /// Reported, never asserted: a catalog that lies about its files cannot be
    /// pruned faithfully, and this is the size of that blast radius.
    lied_and_lost: usize,
    failures: Vec<String>,
}

impl Tally {
    fn merge(&mut self, other: Tally) {
        self.pairs += other.pairs;
        self.non_empty += other.non_empty;
        self.pruned += other.pruned;
        self.lied_and_lost += other.lied_and_lost;
        self.failures.extend(other.failures);
    }
}

/// The first few rows of a difference, each trimmed. A whole row of an
/// eleven-column fixture is unreadable in a panic message and the identifying
/// prefix is enough to find it.
fn summarize(rows: &[&String]) -> String {
    const SHOWN: usize = 3;
    const WIDTH: usize = 60;
    let mut out: Vec<String> = rows
        .iter()
        .take(SHOWN)
        .map(|row| {
            let flat = row.replace('\u{1}', " | ");
            match flat.char_indices().nth(WIDTH) {
                Some((cut, _)) => format!("{}...", &flat[..cut]),
                None => flat,
            }
        })
        .collect();
    if rows.len() > SHOWN {
        out.push(format!("(+{} more)", rows.len() - SHOWN));
    }
    out.join("; ")
}

/// Run one predicate with the filter off and on, and check what `fidelity`
/// entitles this catalog state to.
///
/// Both runs share the catalog, the Parquet and the provider; the only
/// difference is whether the filtered page honours its filter. So a difference
/// in the two answers is attributable to pushdown and nothing else.
///
/// Over a [`Fidelity::HostileEncoding`] state the full property holds: the rows
/// must be *identical*. Over a [`Fidelity::FalseFact`] state the answer is
/// allowed to shrink — see [`Fidelity`] for why nothing else is possible — but
/// three weaker things still have to be true, and they are not free:
///
/// * neither run raises. A malformed bound must not reach an engine as a cast
///   that aborts the listing query, which is exactly what PostgreSQL does with
///   `CAST('abc' AS double precision)`;
/// * the filtered answer is a *subset* of the unfiltered one. Pushdown may only
///   ever remove files, and the in-memory `PruningPredicate` runs over whatever
///   survives in both runs, so a row appearing only when pushdown is on means
///   the listing returned something it should not have;
/// * the filtered listing is no wider than the unfiltered one.
async fn check_pair(
    inner: &Arc<dyn MetadataProvider>,
    state: &str,
    fidelity: Fidelity,
    predicate: &str,
    tally: &mut Tally,
) {
    let sql = format!("SELECT * FROM lake.main.t WHERE {predicate}");
    let label = format!("state [{state}] predicate [{predicate}]");
    tally.pairs += 1;

    let truth = match run_query(inner, false, &sql).await {
        Ok(run) => run,
        Err(e) => {
            // The oracle itself failed; that is a bug in the unfiltered path or
            // in the fixture, and either way the pair cannot be judged.
            tally
                .failures
                .push(format!("{label}: pushdown DISABLED failed: {e}"));
            return;
        },
    };
    let filtered = match run_query(inner, true, &sql).await {
        Ok(run) => run,
        Err(e) => {
            tally.failures.push(format!(
                "{label}: pushdown ENABLED failed while the disabled run \
                 returned {} rows: {e}",
                truth.rows.len()
            ));
            return;
        },
    };

    if !truth.rows.is_empty() {
        tally.non_empty += 1;
    }
    if filtered.listed < truth.listed {
        tally.pruned += 1;
    }

    let missing: Vec<_> = truth
        .rows
        .iter()
        .filter(|row| !filtered.rows.contains(row))
        .collect();
    let extra: Vec<_> = filtered
        .rows
        .iter()
        .filter(|row| !truth.rows.contains(row))
        .collect();

    match fidelity {
        Fidelity::HostileEncoding if filtered.rows != truth.rows => {
            tally.failures.push(format!(
                "{label} [{}]: pushdown changed the answer — {} rows without it, {} with it\n\
                 \x20   rows LOST by pushdown ({}): {}\n\
                 \x20   rows GAINED by pushdown ({}): {}",
                fidelity.label(),
                truth.rows.len(),
                filtered.rows.len(),
                missing.len(),
                summarize(&missing),
                extra.len(),
                summarize(&extra),
            ));
        },
        Fidelity::HostileEncoding => {},
        Fidelity::FalseFact => {
            if !missing.is_empty() {
                tally.lied_and_lost += 1;
            }
            // The one direction that is still a bug: a lying statistic can make
            // pushdown keep too few files, never too many, so a row that
            // appears only with pushdown on cannot be explained by the lie.
            if !extra.is_empty() {
                tally.failures.push(format!(
                    "{label} [{}]: pushdown ADDED {} rows the unfiltered run did \
                     not return: {}",
                    fidelity.label(),
                    extra.len(),
                    summarize(&extra),
                ));
            }
        },
    }

    if filtered.listed > truth.listed {
        tally.failures.push(format!(
            "{label} [{}]: the filtered listing returned MORE catalog rows ({}) \
             than the unfiltered one ({})",
            fidelity.label(),
            filtered.listed,
            truth.listed
        ));
    }
}

// ---------------------------------------------------------------------------
// The sweep
// ---------------------------------------------------------------------------

/// The tag -> `ducklake_column.column_id` map, taken from the catalog itself
/// rather than assumed.
async fn column_ids(store: &CatalogStore) -> (BTreeMap<String, i64>, Vec<i64>, i64) {
    let provider = store.open().await;
    let snapshot_id = provider.get_current_snapshot().expect("current snapshot");
    let schema = provider
        .get_schema_by_name("main", snapshot_id)
        .expect("schema lookup")
        .expect("`main` exists");
    let table = provider
        .get_table_by_name(schema.schema_id, "t", snapshot_id)
        .expect("table lookup")
        .expect("`t` exists");
    let columns = provider
        .get_table_structure(table.table_id, snapshot_id)
        .expect("table structure");
    let ids = columns
        .iter()
        .map(|c| (c.column_name.clone(), c.column_id))
        .collect();
    let mut files: Vec<i64> = provider
        .get_table_files_for_select(table.table_id, snapshot_id)
        .expect("files")
        .into_iter()
        .map(|f| f.data_file_id)
        .collect();
    files.sort_unstable();
    (ids, files, table.table_id)
}

/// The columns each depth poisons. Representative keeps one column of each
/// shape the lowering treats differently: integer, float (the `contains_nan`
/// gate), raw text (the collation path), date and zoned timestamp (the two
/// text-domain temporal encodings), and boolean.
const REPRESENTATIVE_COLUMNS: [&str; 6] = ["id", "f", "s", "d", "tsz", "b"];
const ALL_COLUMNS: [&str; 11] =
    ["id", "f", "dec", "s", "d", "ts_s", "ts_ms", "ts_us", "ts_ns", "tsz", "b"];

/// Put `ducklake_file_column_stats` back exactly as official wrote it, then
/// apply `then`. One catalog round trip per scenario.
async fn restore_then(store: &CatalogStore, then: &[String]) {
    let mut statements = vec![
        "DELETE FROM ducklake_file_column_stats".to_string(),
        "INSERT INTO ducklake_file_column_stats SELECT * FROM zz_stats_backup".to_string(),
    ];
    statements.extend_from_slice(then);
    store.exec_all(&statements).await;
}

/// Run the whole property sweep against one catalog.
///
/// `exhaustive` selects the full cross product; otherwise a representative
/// slice runs — fewer poisoned columns, fewer `min_only` variants, and the
/// cross-column predicates trimmed.
async fn sweep(store: &CatalogStore, exhaustive: bool) {
    let backend = store.label();
    let started = std::time::Instant::now();
    let (ids, files, _table_id) = column_ids(store).await;
    assert_eq!(files.len(), INSERTS.len(), "{backend}: one file per INSERT");

    // A copy to restore from. Every scenario rewrites the statistics table and
    // then puts it back, so the fixture is written once.
    store
        .exec("CREATE TABLE zz_stats_backup AS SELECT * FROM ducklake_file_column_stats")
        .await;

    let all_predicates = predicates();
    let mut tally = Tally::default();

    // --- Half one: real statistics, real data -----------------------------
    {
        let inner = store.open().await;
        let mut clean = Tally::default();
        for predicate in &all_predicates {
            check_pair(
                &inner,
                "clean",
                Fidelity::HostileEncoding,
                &predicate.sql,
                &mut clean,
            )
            .await;
        }
        println!(
            "{backend}: clean fixture — {} pairs, {} with a non-empty answer, {} that pruned",
            clean.pairs, clean.non_empty, clean.pruned
        );
        tally.merge(clean);
    }

    // --- Half two: statistics a foreign writer could have left ------------
    let columns: &[&str] = if exhaustive {
        &ALL_COLUMNS
    } else {
        &REPRESENTATIVE_COLUMNS
    };
    let scopes: Vec<Option<i64>> = if exhaustive {
        vec![None, Some(files[1])]
    } else {
        vec![None]
    };

    let mut scenarios = 0usize;
    for tag in columns {
        let column_id = *ids
            .get(*tag)
            .unwrap_or_else(|| panic!("{backend}: column `{tag}` in the catalog"));
        for scope in &scopes {
            for poison in poisons_for_column(store, tag, column_id, *scope, exhaustive) {
                restore_then(store, &poison.statements).await;
                scenarios += 1;

                let inner = store.open().await;
                for predicate in &all_predicates {
                    let relevant = predicate.tag == poison.tag
                        || (predicate.tag == MULTI && (exhaustive || predicate.sql.contains(*tag)));
                    if !relevant {
                        continue;
                    }
                    check_pair(
                        &inner,
                        &poison.label,
                        poison.fidelity,
                        &predicate.sql,
                        &mut tally,
                    )
                    .await;
                }
            }
        }
    }

    // --- The whole statistics table gone ----------------------------------
    // A catalog that predates `ducklake_file_column_stats` must still list its
    // files. Joining that table from the listing query would turn a legacy
    // catalog into a hard failure, so this runs last: the restore afterwards
    // rebuilds the table from the backup without its original constraints.
    restore_then(
        store,
        &["DROP TABLE ducklake_file_column_stats".to_string()],
    )
    .await;
    scenarios += 1;
    {
        let inner = store.open().await;
        for predicate in &all_predicates {
            check_pair(
                &inner,
                "no_statistics_table",
                Fidelity::HostileEncoding,
                &predicate.sql,
                &mut tally,
            )
            .await;
        }
    }
    store
        .exec("CREATE TABLE ducklake_file_column_stats AS SELECT * FROM zz_stats_backup")
        .await;

    let elapsed = started.elapsed();
    println!(
        "{backend}: {} catalog states x predicates = {} pairs \
         ({} with a non-empty answer, {} where pushdown narrowed the listing) in {:.1}s",
        scenarios + 1,
        tally.pairs,
        tally.non_empty,
        tally.pruned,
        elapsed.as_secs_f64(),
    );
    // Not a failure — see `Fidelity` — but the number a reader should see, since
    // it is what a catalog whose statistics lie costs.
    println!(
        "{backend}: {} pairs over false-fact states lost rows to pushdown, \
         which is what a well-formed but false statistic buys",
        tally.lied_and_lost
    );

    // A sweep that never pruned and never returned a row would pass vacuously.
    assert!(
        tally.non_empty * 4 > tally.pairs,
        "{backend}: only {} of {} pairs had a non-empty ground-truth answer — \
         the sweep cannot witness row loss",
        tally.non_empty,
        tally.pairs
    );
    assert!(
        tally.pruned > 0,
        "{backend}: pushdown never narrowed the listing in {} pairs — \
         the mechanism is not reachable and this sweep proves nothing",
        tally.pairs
    );
    // Failures group hard by catalog state — one bad statistic shape fails every
    // predicate that reads it — so lead with that census. It is the shape of the
    // answer: which statistic states the SQL filter prunes on and the in-memory
    // path does not.
    let mut by_state: BTreeMap<&str, usize> = BTreeMap::new();
    for failure in &tally.failures {
        let state = failure
            .split_once(']')
            .map_or(failure.as_str(), |(head, _)| {
                head.trim_start_matches("state [")
            });
        *by_state.entry(state).or_default() += 1;
    }
    let census: Vec<String> = by_state
        .iter()
        .map(|(state, count)| format!("  {count:>4} pairs  {state}"))
        .collect();
    assert!(
        tally.failures.is_empty(),
        "{backend}: {} of {} (catalog state, predicate) pairs changed their answer \
         when pushdown was enabled.\n\nBy catalog state:\n{}\n\nDetail:\n\n{}",
        tally.failures.len(),
        tally.pairs,
        census.join("\n"),
        tally.failures.join("\n\n")
    );
}

// ---------------------------------------------------------------------------
// Backends
// ---------------------------------------------------------------------------

#[cfg(feature = "metadata-sqlite")]
async fn sqlite_fixture(temp: &TempDir) -> CatalogStore {
    let catalog_path = temp.path().join("pushdown.db");
    let data_path = temp.path().join("data");
    build_fixture(
        &format!("ducklake:sqlite:{}", catalog_path.display()),
        &["sqlite"],
        &data_path,
    )
    .expect("sqlite fixture builds");
    let store = CatalogStore::Sqlite(format!("sqlite:{}", catalog_path.display()));
    store.migrate().await;
    store
}

/// SQLite carries the bulk: no Docker, and the dialect with the least type
/// system, so every comparison it makes is one it reconstructed in text.
#[cfg(feature = "metadata-sqlite")]
#[tokio::test(flavor = "multi_thread")]
async fn pushdown_row_preservation_sqlite() {
    let temp = TempDir::new().unwrap();
    let store = sqlite_fixture(&temp).await;
    sweep(&store, false).await;
}

/// The full cross product: every column, both scopes, every foreign bound in
/// both the `min`-only and both-bounds shapes, and every cross-column
/// predicate against every scenario.
#[cfg(feature = "metadata-sqlite")]
#[tokio::test(flavor = "multi_thread")]
#[ignore = "exhaustive sweep; run with --ignored"]
async fn pushdown_row_preservation_sqlite_exhaustive() {
    let temp = TempDir::new().unwrap();
    let store = sqlite_fixture(&temp).await;
    sweep(&store, true).await;
}

/// DuckDB's own catalog format, the one official DuckLake writes natively and
/// the only dialect that has a real `TRY_CAST`.
#[tokio::test(flavor = "multi_thread")]
async fn pushdown_row_preservation_duckdb() {
    let temp = TempDir::new().unwrap();
    let catalog_path = temp.path().join("pushdown.ducklake");
    let data_path = temp.path().join("data");
    build_fixture(
        &format!("ducklake:{}", catalog_path.display()),
        &[],
        &data_path,
    )
    .expect("duckdb fixture builds");
    let store = CatalogStore::DuckDb(catalog_path);
    store.migrate().await;
    sweep(&store, false).await;
}

/// PostgreSQL: the dialect where a malformed `CAST` aborts the whole listing
/// query rather than returning a wrong value, so its `try_cast` replacement is
/// the furthest from official's.
#[cfg(feature = "metadata-postgres")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
#[tokio::test(flavor = "multi_thread")]
async fn pushdown_row_preservation_postgres() {
    use testcontainers::runners::AsyncRunner;
    use testcontainers_modules::postgres::Postgres;

    let container = Postgres::default().start().await.unwrap();
    let port = container.get_host_port_ipv4(5432).await.unwrap();
    let url = format!("postgresql://postgres:postgres@127.0.0.1:{port}/postgres");
    let temp = TempDir::new().unwrap();
    let data_path = temp.path().join("data");
    build_fixture(
        &format!(
            "ducklake:postgres:host=127.0.0.1 port={port} dbname=postgres \
             user=postgres password=postgres"
        ),
        &["postgres"],
        &data_path,
    )
    .expect("postgres fixture builds");

    let store = CatalogStore::Postgres(url);
    store.migrate().await;
    sweep(&store, false).await;
}

/// Copy every `ducklake_*` table of the DuckDB catalog at `catalog_path` into
/// the MySQL database `dsn` names.
///
/// MySQL cannot host this fixture directly: DuckLake updates its rollup
/// statistics on commit with an `UPDATE ... JOIN`, which DuckDB's MySQL
/// connector refuses ("only simple deletes are supported"), so the second
/// `INSERT` aborts. Official DuckLake writes the catalog where it can and
/// DuckDB copies the rows across unchanged, which also lands `min_value` /
/// `max_value` in `utf8mb4_0900_ai_ci` `TEXT` columns — the case- and
/// accent-insensitive collation the rendered SQL has to override.
#[cfg(feature = "metadata-mysql")]
fn transport_catalog_to_mysql(catalog_path: &Path, dsn: &str) -> anyhow::Result<()> {
    common::ensure_ducklake_installed();
    common::ensure_extension_installed("mysql");

    let conn = duckdb::Connection::open_in_memory()?;
    conn.execute("LOAD mysql;", [])?;
    conn.execute(
        &format!("ATTACH '{}' AS src (READ_ONLY);", catalog_path.display()),
        [],
    )?;
    conn.execute(&format!("ATTACH '{dsn}' AS dst (TYPE mysql);"), [])?;

    let tables: Vec<String> = {
        let mut statement = conn.prepare(
            "SELECT table_name FROM duckdb_tables()
             WHERE database_name = 'src' ORDER BY table_name",
        )?;
        let rows = statement.query_map([], |row| row.get::<_, String>(0))?;
        rows.collect::<std::result::Result<_, _>>()?
    };
    for table in tables {
        conn.execute(&format!("DROP TABLE IF EXISTS dst.{table};"), [])?;
        conn.execute(
            &format!("CREATE TABLE dst.{table} AS SELECT * FROM src.{table};"),
            [],
        )?;
    }
    Ok(())
}

/// MySQL: the only backend whose default collation would drop a matching file
/// on a raw string comparison.
#[cfg(feature = "metadata-mysql")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
#[tokio::test(flavor = "multi_thread")]
async fn pushdown_row_preservation_mysql() {
    use testcontainers::runners::AsyncRunner;
    use testcontainers_modules::mysql::Mysql;

    let container = Mysql::default().start().await.unwrap();
    let port = container.get_host_port_ipv4(3306).await.unwrap();
    let url = format!("mysql://root@127.0.0.1:{port}/test");
    let dsn = format!("host=127.0.0.1 port={port} user=root database=test");

    let temp = TempDir::new().unwrap();
    let catalog_path = temp.path().join("pushdown.ducklake");
    let data_path = temp.path().join("data");
    build_fixture(
        &format!("ducklake:{}", catalog_path.display()),
        &[],
        &data_path,
    )
    .expect("mysql fixture builds");
    transport_catalog_to_mysql(&catalog_path, &dsn).expect("catalog transports to mysql");

    let store = CatalogStore::MySql(url);
    store.migrate().await;
    sweep(&store, false).await;
}
