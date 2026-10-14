//! Data inlining statistics, checked against what official DuckLake (the
//! pinned DuckDB) writes and reads for the same history.
//!
//! Each test applies one history twice: with DuckDB to a catalog it creates,
//! and with this crate to a DuckDB-backed catalog it creates. The table-wide
//! statistics both leave must agree, and DuckDB, which prunes by them, must
//! answer queries over the crate's catalog as it answers them over its own.

#![cfg(feature = "write-duckdb")]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use datafusion::prelude::SessionContext;
use datafusion_ducklake::{
    DuckLakeCatalog, DuckdbMetadataWriter, MetadataProvider, MetadataWriter, execute_ducklake_sql,
};
use duckdb::arrow::array::Array as _;
use tempfile::TempDir;

/// One catalog (a DuckDB metadata file) and its data directory. A DuckDB
/// metadata file admits one open database per process, so every handle is
/// dropped before the next is opened.
struct Lake {
    metadata: PathBuf,
    data: PathBuf,
}

impl Lake {
    fn new(dir: &Path, name: &str) -> Self {
        let data = dir.join(format!("{name}_data"));
        std::fs::create_dir_all(&data).unwrap();
        Self {
            metadata: dir.join(format!("{name}.ducklake")),
            data,
        }
    }

    fn connection(&self) -> duckdb::Connection {
        crate::common::ensure_ducklake_installed();
        let connection = duckdb::Connection::open_in_memory().unwrap();
        connection
            .execute_batch(&format!(
                "LOAD ducklake; \
                 ATTACH 'ducklake:{}' AS lake (DATA_PATH '{}/'); USE lake;",
                self.metadata.display(),
                self.data.display()
            ))
            .unwrap();
        connection
    }

    /// Run `statement` with official DuckLake.
    fn official(&self, statement: &str) {
        self.connection()
            .execute_batch(statement)
            .unwrap_or_else(|e| panic!("official `{statement}`: {e}"));
    }

    /// Rows of `sql` read by official DuckLake, cells joined by `,`.
    fn query(&self, sql: &str) -> Vec<String> {
        query_lines(&self.connection(), sql)
    }

    /// The table-wide statistics of `main.t`: `column name, contains_null,
    /// contains_nan, min_value, max_value`, booleans as `0`/`1`.
    fn table_column_stats(&self) -> Vec<String> {
        let connection = duckdb::Connection::open_in_memory().unwrap();
        connection
            .execute_batch(&format!(
                "ATTACH '{}' AS m (READ_ONLY);",
                self.metadata.display()
            ))
            .unwrap();
        query_lines(
            &connection,
            "SELECT (SELECT c.column_name FROM m.ducklake_column c \
                     WHERE c.table_id = s.table_id AND c.column_id = s.column_id \
                     ORDER BY c.begin_snapshot DESC LIMIT 1), \
                    contains_null::INT, contains_nan::INT, min_value, max_value \
             FROM m.ducklake_table_column_stats s \
             JOIN m.ducklake_table t ON t.table_id = s.table_id \
             WHERE t.table_name = 't' AND t.end_snapshot IS NULL \
             ORDER BY 1",
        )
    }

    fn crate_side(&self) -> (Arc<dyn MetadataProvider>, Arc<dyn MetadataWriter>) {
        let writer =
            DuckdbMetadataWriter::new_with_init(self.metadata.display().to_string()).unwrap();
        writer.set_data_path(self.data.to_str().unwrap()).unwrap();
        (Arc::new(writer.metadata_provider()), Arc::new(writer))
    }

    /// Run `statement` with this crate.
    async fn with_crate(&self, statement: &str) {
        let (provider, writer) = self.crate_side();
        let ctx = SessionContext::new();
        let catalog =
            DuckLakeCatalog::with_writer(Arc::clone(&provider), Arc::clone(&writer)).unwrap();
        DuckLakeCatalog::register(&ctx, "lake", catalog);
        let handle = DuckLakeCatalog::with_writer(provider, writer).unwrap();
        execute_ducklake_sql(&ctx, &handle, statement)
            .await
            .unwrap_or_else(|e| panic!("crate `{statement}`: {e}"))
            .collect()
            .await
            .unwrap_or_else(|e| panic!("crate `{statement}`: {e}"));
    }

    /// Set `main.t`'s inlining limit with this crate.
    fn crate_limit(&self, limit: usize) {
        let (provider, writer) = self.crate_side();
        let snapshot = provider.get_current_snapshot().unwrap();
        let schema = provider
            .get_schema_by_name("main", snapshot)
            .unwrap()
            .unwrap();
        let table = provider
            .get_table_by_name(schema.schema_id, "t", snapshot)
            .unwrap()
            .unwrap();
        writer
            .set_table_setting(
                table.table_id,
                "data_inlining_row_limit",
                &limit.to_string(),
            )
            .unwrap();
    }
}

fn query_lines(connection: &duckdb::Connection, sql: &str) -> Vec<String> {
    let mut statement = connection
        .prepare(sql)
        .unwrap_or_else(|e| panic!("`{sql}`: {e}"));
    let mut lines = Vec::new();
    for batch in statement.query_arrow([]).unwrap() {
        for row in 0..batch.num_rows() {
            let cells = (0..batch.num_columns())
                .map(|column| {
                    let array = batch.column(column);
                    if array.is_null(row) {
                        "NULL".to_string()
                    } else {
                        duckdb::arrow::util::display::array_value_to_string(array, row).unwrap()
                    }
                })
                .collect::<Vec<_>>();
            lines.push(cells.join(","));
        }
    }
    lines
}

/// One step of a history, applied to both catalogs.
enum Step {
    /// A statement over `lake.main`, run by DuckDB and by this crate.
    Sql(&'static str),
    /// Set `main.t`'s `data_inlining_row_limit`.
    Limit(usize),
    /// One statement in each dialect: DuckDB's, then this crate's.
    Both(&'static str, &'static str),
    /// A statement DuckDB runs on both catalogs.
    Duck(&'static str),
    /// A statement run directly on both metadata databases.
    Metadata(&'static str),
    /// A statement run directly on the crate's metadata database only, which
    /// leaves the two catalogs' statistics rows incomparable.
    CrateMetadata(&'static str),
}

/// Apply `history` to an official lake and a crate lake under `dir`, then
/// require equal table-wide statistics and equal DuckDB answers to `queries`.
/// The crate's lake is created by this crate, or by DuckDB when the history
/// starts with [`Step::Duck`].
async fn differential(dir: &Path, history: &[Step], queries: &[&str]) {
    differential_with(dir, history, queries, None).await;
}

/// [`differential`], except that when `crate_stats` is given, the crate's
/// table-wide statistics must equal it rather than official's: for a history
/// after which official stores statistics that are not sound.
async fn differential_with(
    dir: &Path,
    history: &[Step],
    queries: &[&str],
    crate_stats: Option<&[&str]>,
) {
    let official = Lake::new(dir, "official");
    let ours = Lake::new(dir, "crate");
    if !matches!(history.first(), Some(Step::Duck(_))) {
        // Official creates `main` with the catalog.
        let (_, writer) = ours.crate_side();
        let snapshot = writer.create_snapshot().unwrap();
        writer.get_or_create_schema("main", None, snapshot).unwrap();
    }
    for step in history {
        match step {
            Step::Duck(sql) => {
                official.official(sql);
                ours.official(sql);
            },
            Step::CrateMetadata(sql) => {
                duckdb::Connection::open(&ours.metadata)
                    .unwrap()
                    .execute_batch(sql)
                    .unwrap();
            },
            Step::Metadata(sql) => {
                for lake in [&official, &ours] {
                    duckdb::Connection::open(&lake.metadata)
                        .unwrap()
                        .execute_batch(sql)
                        .unwrap();
                }
            },
            Step::Sql(sql) => {
                official.official(&sql.replace("lake.main.", ""));
                ours.with_crate(sql).await;
            },
            Step::Both(duckdb, ours_sql) => {
                official.official(duckdb);
                ours.with_crate(ours_sql).await;
            },
            Step::Limit(limit) => {
                official.official(&format!(
                    "CALL lake.set_option('data_inlining_row_limit', {limit}, table_name => 't')"
                ));
                ours.crate_limit(*limit);
            },
        }
    }
    if let Some(expected) = crate_stats {
        assert_eq!(
            ours.table_column_stats(),
            expected,
            "the crate's ducklake_table_column_stats"
        );
    } else if !history
        .iter()
        .any(|step| matches!(step, Step::CrateMetadata(_)))
    {
        assert_eq!(
            ours.table_column_stats(),
            official.table_column_stats(),
            "ducklake_table_column_stats differ (left: crate, right: official)"
        );
    }
    for sql in queries {
        assert_eq!(
            ours.query(sql),
            official.query(sql),
            "DuckDB answers `{sql}` differently on the crate's catalog"
        );
    }
}

/// An all-NULL Parquet file contributes no bounds to the table-wide roll-up,
/// so an inlined write later widens the bounds the other files established.
#[tokio::test(flavor = "multi_thread")]
async fn all_null_parquet_file_keeps_bounds_through_inlined_write() {
    let temp = TempDir::new().unwrap();
    differential(
        temp.path(),
        &[
            Step::Sql("CREATE TABLE lake.main.t (id INTEGER, v VARCHAR)"),
            Step::Limit(0),
            Step::Sql("INSERT INTO lake.main.t VALUES (1, NULL)"),
            Step::Sql(
                "INSERT INTO lake.main.t VALUES (5, 's05'), (6, 's06'), (7, 's07'), \
                 (8, 's08'), (9, 's09'), (10, 's10'), (11, 's11'), (12, 's12'), (13, 's13'), \
                 (14, 's14'), (15, 's15'), (16, 's16')",
            ),
            Step::Limit(10),
            Step::Sql("INSERT INTO lake.main.t VALUES (20, 'zz')"),
        ],
        &[
            "SELECT count(*) FROM t WHERE v = 's05'",
            "SELECT count(*) FROM t WHERE v = 'zz'",
            "SELECT count(*) FROM t WHERE v IS NULL",
        ],
    )
    .await;
}

/// The first write of a table creates its statistics rows, inlined or not;
/// later inlined writes merge into them.
#[tokio::test(flavor = "multi_thread")]
async fn first_inlined_write_creates_statistics() {
    let temp = TempDir::new().unwrap();
    differential(
        temp.path(),
        &[
            Step::Sql("CREATE TABLE lake.main.t (id INTEGER, v VARCHAR, d DOUBLE)"),
            Step::Limit(10),
            Step::Sql("INSERT INTO lake.main.t VALUES (3, 'm', 1.5), (4, NULL, 2.5)"),
            Step::Sql("INSERT INTO lake.main.t VALUES (1, 'a', -1.0)"),
        ],
        &["SELECT count(*) FROM t WHERE v = 'a'", "SELECT count(*) FROM t WHERE d < 0"],
    )
    .await;
}

/// Inlined writes widen the statistics of every leaf field of nested columns,
/// which DuckDB prunes by. DuckDB creates the catalog and writes the Parquet
/// file, whose statistics hold a row per leaf field. Only the statistics are
/// compared: DuckDB fails to attach a DuckDB-created catalog once this crate
/// has written a snapshot to it, a separate interop gap. (`MAP` leaves are covered by
/// `stats_encode`'s unit tests: this crate's SQL has no `MAP` literal DuckDB
/// shares.)
#[tokio::test(flavor = "multi_thread")]
async fn inlined_writes_widen_nested_leaf_statistics() {
    let temp = TempDir::new().unwrap();
    differential(
        temp.path(),
        &[
            Step::Duck(
                "CREATE TABLE t (id INTEGER, s STRUCT(a INTEGER, b VARCHAR), l INTEGER[]); \
                 CALL lake.set_option('data_inlining_row_limit', 0, table_name => 't'); \
                 INSERT INTO t VALUES (1, {'a': 1, 'b': 'b'}, [5, 6]); \
                 CALL lake.set_option('data_inlining_row_limit', 10, table_name => 't');",
            ),
            Step::Both(
                "INSERT INTO t VALUES (2, {'a': 999, 'b': 'zzz'}, [700])",
                "INSERT INTO lake.main.t VALUES (2, named_struct('a', 999, 'b', 'zzz'), [700])",
            ),
        ],
        &[],
    )
    .await;
}

/// Inlined bounds are DuckDB's text of each value, compared as official
/// compares them: booleans by value, NaN above every number, `TIME` and
/// `BLOB` as text.
#[tokio::test(flavor = "multi_thread")]
async fn inlined_bounds_of_boolean_nan_time_and_blob() {
    let temp = TempDir::new().unwrap();
    differential(
        temp.path(),
        &[
            Step::Both(
                "CREATE TABLE t (b BOOLEAN, d DOUBLE, tm TIME, bl BLOB)",
                "CREATE TABLE lake.main.t (b BOOLEAN, d DOUBLE, tm TIME, bl BYTEA)",
            ),
            Step::Limit(0),
            Step::Both(
                "INSERT INTO t VALUES (true, 1.5, '12:34:56.5', '\\x01'::BLOB)",
                "INSERT INTO lake.main.t VALUES (true, 1.5, CAST('12:34:56.5' AS TIME), \
                 X'01')",
            ),
            Step::Limit(10),
            Step::Both(
                "INSERT INTO t VALUES (false, 'nan', '01:00:00', '\\xAB\\x00z'::BLOB), \
                 (NULL, 2.5, '23:00:00.25', 'a\\x5Cb'::BLOB)",
                "INSERT INTO lake.main.t VALUES (false, CAST('nan' AS DOUBLE), \
                 CAST('01:00:00' AS TIME), X'AB007A'), \
                 (NULL, 2.5, CAST('23:00:00.25' AS TIME), X'615C62')",
            ),
        ],
        &[
            "SELECT count(*) FROM t WHERE b = false",
            "SELECT count(*) FROM t WHERE tm = '01:00:00'",
            "SELECT count(*) FROM t WHERE isnan(d)",
        ],
    )
    .await;
}

/// Once a table has statistics, an inlined write only updates the column rows
/// it has, as official keys its inlined statistics on `ducklake_table_stats`.
/// A table whose files left no column rows (a writer that records none) keeps
/// none, rather than gaining bounds that cover only the inlined rows and so
/// prune the files' rows.
#[tokio::test(flavor = "multi_thread")]
async fn inlined_write_keeps_missing_column_statistics_missing() {
    let temp = TempDir::new().unwrap();
    differential(
        temp.path(),
        &[
            Step::Sql("CREATE TABLE lake.main.t (id INTEGER, v VARCHAR)"),
            Step::Limit(0),
            Step::Sql("INSERT INTO lake.main.t VALUES (1, 'a'), (2, 'b')"),
            Step::Metadata("DELETE FROM ducklake_table_column_stats"),
            Step::Limit(10),
            Step::Sql("INSERT INTO lake.main.t VALUES (3, 'z')"),
        ],
        &["SELECT count(*) FROM t WHERE v = 'a'", "SELECT count(*) FROM t WHERE id = 1"],
    )
    .await;
}

/// A Parquet file whose column is all NULL leaves the stored bounds as they
/// were, so a later inlined write widens them rather than replacing them.
#[tokio::test(flavor = "multi_thread")]
async fn all_null_parquet_file_after_inlined_rows_keeps_their_bounds() {
    let temp = TempDir::new().unwrap();
    differential(
        temp.path(),
        &[
            Step::Sql("CREATE TABLE lake.main.t (a INTEGER)"),
            Step::Limit(10),
            Step::Sql("INSERT INTO lake.main.t VALUES (1), (2), (3)"),
            Step::Limit(0),
            Step::Both(
                "INSERT INTO t SELECT NULL FROM range(20)",
                "INSERT INTO lake.main.t SELECT CAST(NULL AS INTEGER) FROM range(20)",
            ),
            Step::Limit(10),
            Step::Sql("INSERT INTO lake.main.t VALUES (10)"),
        ],
        &[
            "SELECT count(*) FROM t WHERE a = 1",
            "SELECT a FROM t WHERE a IS NOT NULL ORDER BY a",
            "SELECT count(*) FROM t WHERE a IS NULL",
        ],
    )
    .await;
}

/// A column added after the table has statistics gets no statistics row:
/// official only updates the rows a table already has. Files written before
/// the column existed hold no value of it, and must not lead to bounds that
/// prune the rows reading its default.
#[tokio::test(flavor = "multi_thread")]
async fn added_column_gets_no_statistics_row() {
    let temp = TempDir::new().unwrap();
    differential(
        temp.path(),
        &[
            // Official creates an inlined table with the table under a limit
            // above zero, and then fails to inline into it once a column was
            // added; a global limit of 0 creates none.
            Step::Sql("CREATE TABLE lake.main.other (x INTEGER)"),
            Step::Metadata(
                "INSERT INTO ducklake_metadata VALUES ('data_inlining_row_limit', '0', NULL, NULL)",
            ),
            Step::Sql("CREATE TABLE lake.main.t (a INTEGER)"),
            Step::Sql("INSERT INTO lake.main.t VALUES (1), (2)"),
            Step::Duck("ALTER TABLE t ADD COLUMN c INTEGER DEFAULT 42"),
            Step::Sql("INSERT INTO lake.main.t VALUES (3, 7)"),
            Step::Limit(10),
            Step::Sql("INSERT INTO lake.main.t VALUES (4, 8)"),
        ],
        &[
            "SELECT count(*) FROM t WHERE c = 42",
            "SELECT count(*) FROM t WHERE c = 7",
            "SELECT * FROM t ORDER BY a",
        ],
    )
    .await;
}

/// A Parquet file the crate writes records statistics per leaf field of a
/// nested column, as official does, so an inlined write widens them.
#[tokio::test(flavor = "multi_thread")]
async fn crate_parquet_nested_leaf_statistics_survive_inlined_write() {
    let temp = TempDir::new().unwrap();
    differential(
        temp.path(),
        &[
            Step::Both(
                "CREATE TABLE t (id INTEGER, s STRUCT(a INTEGER, b VARCHAR), l INTEGER[])",
                "CREATE TABLE lake.main.t (id INTEGER, s STRUCT<a INTEGER, b VARCHAR>, \
                 l INTEGER[])",
            ),
            Step::Limit(0),
            Step::Both(
                "INSERT INTO t VALUES (1, {'a': 500, 'b': 'b'}, [5, 6]), (2, NULL, NULL)",
                "INSERT INTO lake.main.t VALUES (1, named_struct('a', 500, 'b', 'b'), [5, 6]), \
                 (2, NULL, NULL)",
            ),
            Step::Limit(10),
            Step::Both(
                "INSERT INTO t VALUES (3, {'a': 999, 'b': 'zzz'}, [700])",
                "INSERT INTO lake.main.t VALUES (3, named_struct('a', 999, 'b', 'zzz'), [700])",
            ),
        ],
        &[
            "SELECT id FROM t WHERE s.a = 500",
            "SELECT id FROM t WHERE s.b = 'b'",
            "SELECT id FROM t WHERE list_contains(l, 5)",
            "SELECT id, s.a, s.b FROM t ORDER BY id",
        ],
    )
    .await;
}

/// Floating-point columns: a Parquet file holding NaN records no bounds, an
/// all-NULL file none either, and inlined rows then merge into what official
/// would store.
#[tokio::test(flavor = "multi_thread")]
async fn crate_parquet_float_nan_and_all_null_files() {
    let temp = TempDir::new().unwrap();
    differential(
        temp.path(),
        &[
            Step::Sql("CREATE TABLE lake.main.t (id INTEGER, d DOUBLE, f REAL)"),
            Step::Limit(0),
            Step::Sql(
                "INSERT INTO lake.main.t VALUES (1, 1.5, CAST(2.5 AS REAL)), \
                 (2, -3.0, CAST(-1.0 AS REAL))",
            ),
            Step::Both(
                "INSERT INTO t VALUES (3, 'nan'::DOUBLE, 'nan'::REAL), (4, 7.0, 8.0)",
                "INSERT INTO lake.main.t VALUES (3, CAST('nan' AS DOUBLE), \
                 CAST('nan' AS REAL)), (4, 7.0, CAST(8.0 AS REAL))",
            ),
            Step::Sql("INSERT INTO lake.main.t VALUES (5, NULL, NULL)"),
            Step::Limit(10),
            Step::Sql("INSERT INTO lake.main.t VALUES (6, 0.5, CAST(0.5 AS REAL))"),
            Step::Limit(0),
            Step::Sql("INSERT INTO lake.main.t VALUES (7, 100.0, CAST(100.0 AS REAL))"),
        ],
        &[
            "SELECT id FROM t WHERE d = 1.5 ORDER BY id",
            "SELECT id FROM t WHERE d = -3.0 ORDER BY id",
            "SELECT id FROM t WHERE d > 50 AND NOT isnan(d) ORDER BY id",
            "SELECT id FROM t WHERE f = 8.0 ORDER BY id",
            "SELECT id FROM t WHERE isnan(d) ORDER BY id",
            "SELECT id, d, f FROM t ORDER BY id",
        ],
    )
    .await;
}

/// Boolean columns compare their bounds by value across Parquet files, an
/// all-NULL file and inlined rows.
#[tokio::test(flavor = "multi_thread")]
async fn crate_parquet_boolean_bounds() {
    let temp = TempDir::new().unwrap();
    differential(
        temp.path(),
        &[
            Step::Sql("CREATE TABLE lake.main.t (id INTEGER, b BOOLEAN)"),
            Step::Limit(10),
            Step::Sql("INSERT INTO lake.main.t VALUES (1, true)"),
            Step::Limit(0),
            Step::Sql("INSERT INTO lake.main.t VALUES (2, NULL), (3, NULL)"),
            Step::Limit(10),
            Step::Sql("INSERT INTO lake.main.t VALUES (4, false)"),
            Step::Limit(0),
            Step::Sql("INSERT INTO lake.main.t VALUES (5, true)"),
        ],
        &[
            "SELECT id FROM t WHERE b ORDER BY id",
            "SELECT id FROM t WHERE NOT b ORDER BY id",
            "SELECT id FROM t WHERE b IS NULL ORDER BY id",
        ],
    )
    .await;
}

/// Deleting rows leaves the table-wide statistics as they were; a later
/// inlined write merges into them.
#[tokio::test(flavor = "multi_thread")]
async fn deletes_leave_statistics_unchanged() {
    let temp = TempDir::new().unwrap();
    differential(
        temp.path(),
        &[
            Step::Sql("CREATE TABLE lake.main.t (a INTEGER, v VARCHAR)"),
            Step::Limit(0),
            Step::Sql("INSERT INTO lake.main.t VALUES (1, 'a'), (2, 'b'), (3, 'c')"),
            Step::Sql("INSERT INTO lake.main.t VALUES (NULL, NULL)"),
            Step::Sql("DELETE FROM lake.main.t WHERE a = 1"),
            Step::Limit(10),
            Step::Sql("INSERT INTO lake.main.t VALUES (9, 'z')"),
            Step::Sql("DELETE FROM lake.main.t WHERE a = 9"),
        ],
        &[
            "SELECT a FROM t WHERE a >= 2 ORDER BY a",
            "SELECT a FROM t WHERE v = 'b'",
            "SELECT count(*) FROM t WHERE a IS NULL",
        ],
    )
    .await;
}

/// A flush merges the flushed files' statistics into the roll-up the inlined
/// rows already widened, as official's does, and a later write widens it.
#[tokio::test(flavor = "multi_thread")]
async fn flush_merges_the_flushed_files_statistics() {
    let temp = TempDir::new().unwrap();
    differential(
        temp.path(),
        &[
            Step::Sql("CREATE TABLE lake.main.t (a INTEGER, v VARCHAR, d DOUBLE)"),
            Step::Limit(10),
            Step::Sql("INSERT INTO lake.main.t VALUES (3, 'm', 1.5), (NULL, NULL, NULL)"),
            Step::Sql("CALL ducklake_flush_inlined_data('lake')"),
            Step::Sql("INSERT INTO lake.main.t VALUES (9, 'z', -2.5)"),
            Step::Limit(0),
            Step::Sql("INSERT INTO lake.main.t VALUES (1, 'a', 0.5)"),
        ],
        &[
            "SELECT a FROM t WHERE a = 3",
            "SELECT a FROM t WHERE v = 'z'",
            "SELECT count(*) FROM t WHERE a IS NULL",
            "SELECT a FROM t WHERE d < 0",
        ],
    )
    .await;
}

/// A roll-up row with neither bound, as releases of this crate up to 0.9
/// stored when a live file held only NULLs, says nothing about the bounds of
/// the other files: a later write must not narrow it to its own values.
#[tokio::test(flavor = "multi_thread")]
async fn legacy_unknown_bounds_are_not_narrowed() {
    let temp = TempDir::new().unwrap();
    differential(
        temp.path(),
        &[
            Step::Sql("CREATE TABLE lake.main.t (a INTEGER)"),
            Step::Limit(0),
            Step::Sql("INSERT INTO lake.main.t VALUES (NULL)"),
            Step::Sql("INSERT INTO lake.main.t VALUES (50), (60)"),
            Step::CrateMetadata(
                "UPDATE ducklake_table_column_stats \
                 SET min_value = NULL, max_value = NULL, contains_nan = NULL",
            ),
            Step::Limit(10),
            Step::Sql("INSERT INTO lake.main.t VALUES (5)"),
        ],
        &[
            "SELECT min(a), max(a), count(a) FROM t",
            "SELECT count(*) FROM t WHERE a = 50",
            "SELECT count(*) FROM t WHERE a = 5",
        ],
    )
    .await;
}

/// `TIMESTAMP` bounds before year 1 compare by the instant they spell, as
/// official casts them to compare: DuckDB marks the era inside the text
/// (`0044-03-15 (BC) 12:30:00.5`), so their bytes do not order them.
#[tokio::test(flavor = "multi_thread")]
async fn inlined_bc_timestamp_bounds_compare_by_value() {
    let temp = TempDir::new().unwrap();
    differential(
        temp.path(),
        &[
            Step::Both(
                "CREATE TABLE t (ts TIMESTAMP, d DATE)",
                "CREATE TABLE lake.main.t (ts TIMESTAMP(6), d DATE)",
            ),
            Step::Both(
                "INSERT INTO t VALUES ('0100-01-01 (BC) 00:00:00', '0100-01-01 (BC)'), \
                 ('0044-03-15 (BC) 12:30:00.5', '0044-03-15 (BC)'), \
                 ('2020-01-01 00:00:00', '2020-01-01')",
                "INSERT INTO lake.main.t VALUES \
                 (arrow_cast(-65291356800000000, 'Timestamp(Microsecond, None)'), \
                  arrow_cast(-755687, 'Date32')), \
                 (arrow_cast(-63517778999500000, 'Timestamp(Microsecond, None)'), \
                  arrow_cast(-735160, 'Date32')), \
                 (arrow_cast(1577836800000000, 'Timestamp(Microsecond, None)'), \
                  DATE '2020-01-01')",
            ),
        ],
        &[
            "SELECT count(*) FROM t WHERE ts < '0001-01-01'",
            "SELECT count(*) FROM t WHERE ts < '0050-01-01 (BC)'",
            "SELECT count(*) FROM t WHERE d < '0050-01-01 (BC)'",
            "SELECT min(ts), max(ts), min(d), max(d) FROM t",
        ],
    )
    .await;
}

/// The `(table_name, schema_version)` rows of `ducklake_inlined_data_tables`.
fn inlined_registry(lake: &Lake) -> Vec<String> {
    let connection = duckdb::Connection::open_in_memory().unwrap();
    connection
        .execute_batch(&format!(
            "ATTACH '{}' AS m (READ_ONLY);",
            lake.metadata.display()
        ))
        .unwrap();
    query_lines(
        &connection,
        "SELECT table_name, schema_version FROM m.ducklake_inlined_data_tables ORDER BY 2",
    )
}

/// Official creates a table's inlined data table when the table is created
/// and again whenever its columns change, so a later inlined insert by any
/// writer finds one with the table's columns. A write of this crate that adds
/// a column is such a change.
#[tokio::test(flavor = "multi_thread")]
async fn inlined_data_tables_follow_column_changes_as_official_creates_them() {
    use arrow::array::Int32Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use datafusion_ducklake::{DuckLakeTableWriter, DuckLakeWriteOptions};
    let temp = TempDir::new().unwrap();
    let official = Lake::new(temp.path(), "official");
    let ours = Lake::new(temp.path(), "crate");
    {
        let (_, writer) = ours.crate_side();
        let snapshot = writer.create_snapshot().unwrap();
        writer.get_or_create_schema("main", None, snapshot).unwrap();
    }
    official.official("CREATE TABLE t (a INTEGER)");
    ours.with_crate("CREATE TABLE lake.main.t (a INT)").await;
    assert_eq!(
        inlined_registry(&ours),
        inlined_registry(&official),
        "at CREATE"
    );
    official.official("INSERT INTO t VALUES (1)");
    ours.with_crate("INSERT INTO lake.main.t VALUES (1)").await;
    assert_eq!(
        inlined_registry(&ours),
        inlined_registry(&official),
        "after inlined INSERT"
    );

    // A Parquet write that adds column `b`.
    official.official(
        "BEGIN; ALTER TABLE t ADD COLUMN b INTEGER; \
         INSERT INTO t SELECT i, i FROM range(20) r(i); COMMIT;",
    );
    {
        let (_, writer) = ours.crate_side();
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Int32, true),
        ]));
        let values = Arc::new(Int32Array::from((0..20).collect::<Vec<_>>()));
        let batch = RecordBatch::try_new(schema, vec![values.clone(), values]).unwrap();
        // Official's default limit of 10: the 20 rows go to Parquet.
        DuckLakeTableWriter::new(
            writer,
            Arc::new(object_store::local::LocalFileSystem::new()),
        )
        .unwrap()
        .with_options(&DuckLakeWriteOptions::default().with_data_inlining_row_limit(10))
        .append_table("main", "t", &[batch])
        .await
        .unwrap();
    }
    assert_eq!(
        inlined_registry(&ours),
        inlined_registry(&official),
        "after the column-adding write"
    );

    for lake in [&official, &ours] {
        lake.official("INSERT INTO t VALUES (5, 6)");
    }
    assert_eq!(
        ours.query("SELECT a, b FROM t ORDER BY a, b"),
        official.query("SELECT a, b FROM t ORDER BY a, b")
    );
    assert_eq!(inlined_registry(&ours), inlined_registry(&official));
}

/// A flush with no rows to write still drops the empty inlined data tables a
/// later schema version superseded, as official's flush does on the commit
/// that changes nothing.
#[tokio::test(flavor = "multi_thread")]
async fn flush_with_nothing_to_write_drops_empty_superseded_inlined_tables() {
    use arrow::array::Int32Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use datafusion_ducklake::{DuckLakeTableWriter, DuckLakeWriteOptions};
    let temp = TempDir::new().unwrap();
    let official = Lake::new(temp.path(), "official");
    let ours = Lake::new(temp.path(), "crate");
    {
        let (_, writer) = ours.crate_side();
        let snapshot = writer.create_snapshot().unwrap();
        writer.get_or_create_schema("main", None, snapshot).unwrap();
    }
    official.official("CREATE TABLE t (a INTEGER)");
    ours.with_crate("CREATE TABLE lake.main.t (a INT)").await;
    official.official(
        "BEGIN; ALTER TABLE t ADD COLUMN b INTEGER; \
         INSERT INTO t SELECT i, i FROM range(20) r(i); COMMIT;",
    );
    {
        let (_, writer) = ours.crate_side();
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Int32, true),
        ]));
        let values = Arc::new(Int32Array::from((0..20).collect::<Vec<_>>()));
        let batch = RecordBatch::try_new(schema, vec![values.clone(), values]).unwrap();
        // Official's default limit of 10: the 20 rows go to Parquet.
        DuckLakeTableWriter::new(
            writer,
            Arc::new(object_store::local::LocalFileSystem::new()),
        )
        .unwrap()
        .with_options(&DuckLakeWriteOptions::default().with_data_inlining_row_limit(10))
        .append_table("main", "t", &[batch])
        .await
        .unwrap();
    }
    assert_eq!(
        inlined_registry(&ours),
        inlined_registry(&official),
        "before the flush"
    );
    official.official("CALL ducklake_flush_inlined_data('lake')");
    ours.with_crate("CALL ducklake_flush_inlined_data('lake')")
        .await;
    assert_eq!(
        inlined_registry(&ours),
        inlined_registry(&official),
        "after the flush"
    );
    assert_eq!(inlined_registry(&ours).len(), 1);
}

/// A string holding a NUL byte has a bound no catalog stores, so its column
/// gets no table-wide statistics row, whether the string is inlined or in a
/// Parquet file. Official writes a row with neither bound for it, which reads
/// as holding no valid value (see COMPATIBILITY.md).
#[tokio::test(flavor = "multi_thread")]
async fn nul_string_leaves_no_statistics_row_inlined_or_in_parquet() {
    let temp = TempDir::new().unwrap();
    for limit in [10, 0] {
        let ours = Lake::new(temp.path(), &format!("crate_{limit}"));
        {
            let (_, writer) = ours.crate_side();
            let snapshot = writer.create_snapshot().unwrap();
            writer.get_or_create_schema("main", None, snapshot).unwrap();
        }
        ours.with_crate("CREATE TABLE lake.main.t (s VARCHAR, i INT)")
            .await;
        ours.crate_limit(limit);
        ours.with_crate("INSERT INTO lake.main.t VALUES (concat('a', chr(0), 'b'), 1), ('c', 2)")
            .await;
        assert_eq!(
            ours.table_column_stats(),
            vec!["i,0,NULL,1,2".to_string()],
            "limit {limit}"
        );
    }
}

/// Official writes a row with neither bound when a rewrite recomputes the
/// statistics from live files of only NULLs; a later write merges into it.
/// Files the rewrite ended do not make the row unknown.
#[tokio::test(flavor = "multi_thread")]
async fn rewrite_recomputed_null_row_takes_the_next_write() {
    let temp = TempDir::new().unwrap();
    differential(
        temp.path(),
        &[
            Step::Duck("CREATE TABLE t (a INTEGER)"),
            Step::Duck("CALL lake.set_option('data_inlining_row_limit', 0, table_name => 't')"),
            Step::Duck("INSERT INTO t VALUES (1), (2), (NULL)"),
            Step::Duck("DELETE FROM t WHERE a IS NOT NULL"),
            Step::Duck("CALL ducklake_rewrite_data_files('lake', delete_threshold => 0.0)"),
            Step::Limit(10),
            Step::Sql("INSERT INTO lake.main.t VALUES (5)"),
        ],
        &[],
    )
    .await;
}

/// After a Parquet file of only NaN and an inlined NULL, official's row has
/// neither bound and no NaN flag, and official narrows it to the next write's
/// bounds (`d,1,NULL,1.5,1.5`), which leave the NaN row out. The crate stores
/// no row for the column once the NaN-only file commits, so later writes add
/// none.
#[tokio::test(flavor = "multi_thread")]
async fn nan_only_file_then_inlined_null_row_leaves_no_row() {
    let temp = TempDir::new().unwrap();
    differential_with(
        temp.path(),
        &[
            Step::Sql("CREATE TABLE lake.main.t (d DOUBLE)"),
            Step::Limit(0),
            Step::Both(
                "INSERT INTO t VALUES ('nan')",
                "INSERT INTO lake.main.t VALUES (CAST('nan' AS DOUBLE))",
            ),
            Step::Limit(10),
            Step::Sql("INSERT INTO lake.main.t VALUES (NULL)"),
            Step::Sql("INSERT INTO lake.main.t VALUES (1.5)"),
        ],
        &["SELECT count(*) FROM t WHERE d = 1.5"],
        Some(&[]),
    )
    .await;
}
