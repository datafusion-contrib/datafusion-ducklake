//! `UPDATE` and `DELETE` under data inlining, differential against official
//! DuckLake (DuckDB 1.5.5).
//!
//! Each scenario builds the same table twice — with DuckDB in an official
//! catalog and with this crate in its own — then applies one statement history
//! to each. Afterwards both catalogs must hold the same snapshot ledger, inlined
//! data and inlined deletion rows, data and delete files and table record count,
//! and read the same values with the same row ids at every snapshot the
//! statements made. Snapshot, table and data file ids are compared relative to
//! the setup, since each implementation allocates them from its own counters.

#![cfg(all(feature = "write", feature = "metadata-duckdb"))]
#![allow(dead_code)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{Int32Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::prelude::SessionContext;
use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeTableWriter, DuckLakeWriteOptions, MetadataProvider, MetadataWriter,
    execute_ducklake_sql,
};
use duckdb::arrow::array::Array as _;
use object_store::local::LocalFileSystem;

/// Where a catalog's metadata lives.
#[derive(Clone)]
pub(crate) enum Metadata {
    Sqlite(PathBuf),
    Duckdb(PathBuf),
    Mysql {
        host: String,
        port: u16,
        database: String,
    },
    /// A PostgreSQL database, read only through `attach_raw`.
    Postgres(String),
}

impl Metadata {
    /// The `ducklake:` attach string official DuckLake opens it with.
    pub(crate) fn ducklake_uri(&self) -> String {
        match self {
            Self::Sqlite(path) => format!("ducklake:sqlite:{}", path.display()),
            Self::Duckdb(path) => format!("ducklake:{}", path.display()),
            Self::Mysql {
                host,
                port,
                database,
            } => format!("ducklake:mysql:db={database} host={host} port={port} user=root"),
            Self::Postgres(dsn) => format!("ducklake:postgres:{dsn}"),
        }
    }

    /// Attach the metadata database itself as `m` in a DuckDB connection.
    pub(crate) fn attach_raw(&self, connection: &duckdb::Connection) {
        let sql = match self {
            Self::Sqlite(path) => {
                format!("ATTACH '{}' AS m (TYPE sqlite, READ_ONLY)", path.display())
            },
            Self::Duckdb(path) => format!("ATTACH '{}' AS m (READ_ONLY)", path.display()),
            Self::Mysql {
                host,
                port,
                database,
            } => format!(
                "ATTACH 'db={database} host={host} port={port} user=root' AS m \
                 (TYPE mysql, READ_ONLY)"
            ),
            Self::Postgres(dsn) => format!("ATTACH '{dsn}' AS m (TYPE postgres, READ_ONLY)"),
        };
        connection.execute_batch(&sql).unwrap();
        connection.execute_batch("USE m").unwrap();
    }
}

pub(crate) fn duckdb_connection() -> duckdb::Connection {
    crate::common::ensure_ducklake_installed();
    crate::common::ensure_extension_installed("sqlite");
    crate::common::ensure_extension_installed("mysql");
    crate::common::ensure_extension_installed("postgres");
    let connection = duckdb::Connection::open_in_memory().unwrap();
    connection
        .execute_batch("LOAD ducklake; LOAD sqlite; LOAD mysql; LOAD postgres;")
        .unwrap();
    connection
}

pub(crate) fn attach_lake(
    connection: &duckdb::Connection,
    metadata: &Metadata,
    data: &Path,
    read_only: bool,
) {
    connection
        .execute_batch(&format!(
            "ATTACH '{}' AS lake (DATA_PATH '{}/'{})",
            metadata.ducklake_uri(),
            data.display(),
            if read_only {
                ", READ_ONLY"
            } else {
                ""
            }
        ))
        .unwrap();
}

/// Run `statements` with official DuckLake, one transaction each.
pub(crate) fn official(metadata: &Metadata, data: &Path, statements: &[String]) {
    let connection = duckdb_connection();
    attach_lake(&connection, metadata, data, false);
    for statement in statements {
        connection
            .execute_batch(statement)
            .unwrap_or_else(|e| panic!("official `{statement}`: {e}"));
    }
}

fn duckdb_lines(batches: &[duckdb::arrow::record_batch::RecordBatch]) -> Vec<String> {
    let mut lines = Vec::new();
    for batch in batches {
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

pub(crate) fn query_lines(connection: &duckdb::Connection, sql: &str) -> Vec<String> {
    let mut statement = connection
        .prepare(sql)
        .unwrap_or_else(|e| panic!("`{sql}`: {e}"));
    let batches = statement.query_arrow([]).unwrap().collect::<Vec<_>>();
    duckdb_lines(&batches)
}

/// Rows of `sql` run by DuckDB over the lake `metadata` describes, read-only.
pub(crate) fn duckdb_query(metadata: &Metadata, data: &Path, sql: &str) -> Vec<String> {
    let connection = duckdb_connection();
    attach_lake(&connection, metadata, data, true);
    query_lines(&connection, sql)
}

/// `rowid, *` of `main.t` at each of `snapshots`, read by DuckDB.
fn official_values(metadata: &Metadata, data: &Path, snapshots: &[i64]) -> Vec<Vec<String>> {
    let connection = duckdb_connection();
    attach_lake(&connection, metadata, data, true);
    snapshots
        .iter()
        .map(|snapshot| {
            query_lines(
                &connection,
                &format!(
                    "SELECT * FROM (SELECT rowid AS __rowid, * FROM lake.main.t \
                     AT (VERSION => {snapshot})) ORDER BY __rowid"
                ),
            )
        })
        .collect()
}

/// `(data_file_id, delete_count, begin_snapshot, end_snapshot, partial_max)`.
type DeleteFileRow = (i64, i64, i64, Option<i64>, Option<i64>);

/// The raw catalog rows of a lake, before normalization.
#[derive(Default)]
pub(crate) struct RawDump {
    head: i64,
    table_id: i64,
    changes: Vec<(i64, String)>,
    /// Inlined data tables in schema-version order: rows `(row_id, begin, end,
    /// values...)`.
    inlined: Vec<Vec<Vec<String>>>,
    /// `(file_id, row_id, begin_snapshot)`.
    inlined_deletes: Vec<(i64, i64, i64)>,
    /// `(data_file_id, record_count, begin_snapshot, end_snapshot)`.
    data_files: Vec<(i64, i64, i64, Option<i64>)>,
    delete_files: Vec<DeleteFileRow>,
    record_count: i64,
    /// Rows of `ducklake_files_scheduled_for_deletion`.
    scheduled: i64,
    /// `ducklake_table_column_stats` rows: `column rank, contains_null,
    /// contains_nan, min_value, max_value`.
    column_stats: Vec<String>,
}

/// Read a catalog through `query`, which renders each row as comma-separated
/// cells with NULL as `NULL`, over its metadata attached as `m`.
fn raw_dump_with(query: impl Fn(&str) -> Vec<String>) -> RawDump {
    let int = |s: &str| s.parse::<i64>().unwrap();
    let opt = |s: &str| (s != "NULL").then(|| s.parse::<i64>().unwrap());
    let mut dump = RawDump {
        head: int(&query("SELECT max(snapshot_id) FROM m.ducklake_snapshot")[0]),
        table_id: int(&query(
            "SELECT table_id FROM m.ducklake_table WHERE table_name = 't' AND end_snapshot IS NULL",
        )[0]),
        ..RawDump::default()
    };
    for line in query(
        "SELECT snapshot_id, coalesce(changes_made, '') FROM m.ducklake_snapshot_changes \
         ORDER BY snapshot_id",
    ) {
        let (id, changes) = line.split_once(',').unwrap();
        dump.changes.push((int(id), changes.to_string()));
    }
    for name in query(&format!(
        "SELECT table_name FROM m.ducklake_inlined_data_tables WHERE table_id = {} \
         ORDER BY schema_version",
        dump.table_id
    )) {
        let rows = query(&format!("SELECT COLUMNS(*)::VARCHAR FROM m.\"{name}\""))
            .into_iter()
            .map(|line| line.split(',').map(str::to_string).collect())
            .collect::<Vec<Vec<String>>>();
        if !rows.is_empty() {
            dump.inlined.push(rows);
        }
    }
    let deletes = format!("ducklake_inlined_delete_{}", dump.table_id);
    // A MySQL attachment lists the tables of every database on the server, so
    // the lookup is confined to the schema `m.<table>` resolves to.
    if !query(&format!(
        "SELECT table_name FROM duckdb_tables() WHERE database_name = 'm' \
         AND schema_name = current_schema() AND table_name = '{deletes}'"
    ))
    .is_empty()
    {
        for line in query(&format!(
            "SELECT file_id, row_id, begin_snapshot FROM m.\"{deletes}\""
        )) {
            let cells = line.split(',').map(int).collect::<Vec<_>>();
            dump.inlined_deletes.push((cells[0], cells[1], cells[2]));
        }
    }
    for line in query(&format!(
        "SELECT data_file_id, record_count, begin_snapshot, end_snapshot \
         FROM m.ducklake_data_file WHERE table_id = {} AND record_count > 0",
        dump.table_id
    )) {
        let cells = line.split(',').collect::<Vec<_>>();
        dump.data_files
            .push((int(cells[0]), int(cells[1]), int(cells[2]), opt(cells[3])));
    }
    for line in query(&format!(
        "SELECT data_file_id, delete_count, begin_snapshot, end_snapshot, partial_max \
         FROM m.ducklake_delete_file WHERE table_id = {}",
        dump.table_id
    )) {
        let cells = line.split(',').collect::<Vec<_>>();
        dump.delete_files.push((
            int(cells[0]),
            int(cells[1]),
            int(cells[2]),
            opt(cells[3]),
            opt(cells[4]),
        ));
    }
    dump.scheduled = int(&query("SELECT count(*) FROM m.ducklake_files_scheduled_for_deletion")[0]);
    dump.record_count = query(&format!(
        "SELECT max(record_count) FROM m.ducklake_table_stats WHERE table_id = {}",
        dump.table_id
    ))
    .first()
    .map_or(0, |s| int(s));
    // Columns are named by their rank among the table's column ids, since each
    // implementation allocates ids from its own counter.
    dump.column_stats = query(&format!(
        "SELECT (SELECT count(DISTINCT c.column_id) FROM m.ducklake_column c \
                 WHERE c.table_id = s.table_id AND c.column_id < s.column_id), \
                contains_null, contains_nan, min_value, max_value \
         FROM m.ducklake_table_column_stats s WHERE table_id = {} ORDER BY column_id",
        dump.table_id
    ));
    dump
}

/// A catalog dump with ids made relative to the setup: snapshot `s` is `S`
/// when the setup made it and `+n` for the statements' `n`th; a data file is
/// named by its rank among the files holding setup rows, or `new`; the table
/// id is `T`.
fn normalize(dump: &RawDump, setup_head: i64) -> BTreeMap<String, Vec<String>> {
    let snap = |s: i64| {
        if s <= setup_head {
            "S".to_string()
        } else {
            format!("+{}", s - setup_head)
        }
    };
    let opt_snap = |s: Option<i64>| s.map_or("NULL".to_string(), snap);
    // Files whose rows the setup wrote, ranked by first snapshot and size: a
    // flush registers its files under the snapshots of their rows, in an order
    // each implementation picks.
    let mut ranked = dump
        .data_files
        .iter()
        .filter(|f| f.2 <= setup_head)
        .map(|f| (f.2, f.1, f.0))
        .collect::<Vec<_>>();
    ranked.sort_unstable();
    let setup_files = ranked.into_iter().map(|f| f.2).collect::<Vec<_>>();
    let file = |id: i64| {
        setup_files
            .iter()
            .position(|f| *f == id)
            .map_or("new".to_string(), |rank| format!("f{rank}"))
    };
    let table = format!(":{}", dump.table_id);
    let mut out = BTreeMap::new();
    out.insert(
        "changes".to_string(),
        dump.changes
            .iter()
            .filter(|(s, _)| *s > setup_head)
            .map(|(s, c)| {
                let tokens = c
                    .split(',')
                    .map(|token| match token.strip_suffix(&table) {
                        Some(kind) => format!("{kind}:T"),
                        None => token.to_string(),
                    })
                    .collect::<Vec<_>>();
                format!("{} {}", snap(*s), tokens.join(","))
            })
            .collect(),
    );
    let mut inlined = Vec::new();
    for (index, rows) in dump.inlined.iter().enumerate() {
        for row in rows {
            let begin = snap(row[1].parse().unwrap());
            let end = if row[2] == "NULL" {
                "NULL".to_string()
            } else {
                snap(row[2].parse().unwrap())
            };
            inlined.push(format!(
                "table{index} rowid={} {begin}..{end} [{}]",
                row[0],
                row[3..].join(",")
            ));
        }
    }
    inlined.sort();
    out.insert("inlined data".to_string(), inlined);
    let mut deletes = dump
        .inlined_deletes
        .iter()
        .map(|(f, r, b)| format!("{} pos={r} {}", file(*f), snap(*b)))
        .collect::<Vec<_>>();
    deletes.sort();
    out.insert("inlined deletes".to_string(), deletes);
    let mut files = dump
        .data_files
        .iter()
        .map(|(f, n, b, e)| format!("{} rows={n} {}..{}", file(*f), snap(*b), opt_snap(*e)))
        .collect::<Vec<_>>();
    files.sort();
    out.insert("data files".to_string(), files);
    let mut delete_files = dump
        .delete_files
        .iter()
        .map(|(f, n, b, e, max)| {
            format!(
                "{} deletes={n} {}..{} max={}",
                file(*f),
                snap(*b),
                opt_snap(*e),
                opt_snap(*max)
            )
        })
        .collect::<Vec<_>>();
    delete_files.sort();
    out.insert("delete files".to_string(), delete_files);
    out.insert(
        "scheduled for deletion".to_string(),
        vec![dump.scheduled.to_string()],
    );
    out.insert(
        "record count".to_string(),
        vec![dump.record_count.to_string()],
    );
    // A row of all-NULL statistics says no more than no row: neither lets a
    // reader prune. Official never adds a row for a column added once the
    // table's statistics exist; this crate records it as unknown.
    out.insert(
        "table column stats".to_string(),
        dump.column_stats
            .iter()
            .filter(|row| !row.ends_with(",NULL,NULL,NULL,NULL"))
            .map(|row| {
                // Booleans render as `0`/`1` or `false`/`true` by backend.
                row.split(',')
                    .map(|cell| match cell {
                        "false" => "0",
                        "true" => "1",
                        cell => cell,
                    })
                    .collect::<Vec<_>>()
                    .join(",")
            })
            .collect(),
    );
    out
}

/// One step of building a scenario's table with this crate.
#[derive(Clone)]
pub(crate) enum CrateStep {
    /// Append rows through `DuckLakeTableWriter` (creating the table first).
    Append(RecordBatch),
    /// Append rows to another table of `main`, creating the schema.
    AppendOther(&'static str, RecordBatch),
    /// Run a statement through `execute_ducklake_sql`.
    Sql(String),
}

/// The crate's writer and provider for one backend.
pub(crate) struct CrateSide {
    pub(crate) writer: Arc<dyn MetadataWriter>,
    pub(crate) provider: Arc<dyn MetadataProvider>,
}

/// The scoped inlining limit most scenarios use.
const LIMIT: usize = 5;

async fn run_crate_sql(side: &CrateSide, statement: &str) {
    let ctx = SessionContext::new();
    let catalog =
        DuckLakeCatalog::with_writer(Arc::clone(&side.provider), Arc::clone(&side.writer)).unwrap();
    DuckLakeCatalog::register(&ctx, "lake", catalog);
    let handle =
        DuckLakeCatalog::with_writer(Arc::clone(&side.provider), Arc::clone(&side.writer)).unwrap();
    execute_ducklake_sql(&ctx, &handle, statement)
        .await
        .unwrap_or_else(|e| panic!("crate `{statement}`: {e}"))
        .collect()
        .await
        .unwrap_or_else(|e| panic!("crate `{statement}`: {e}"));
}

async fn crate_setup(side: &CrateSide, steps: &[CrateStep], limit: usize) {
    // A writer without global settings (MySQL) takes the limit at table scope
    // once the table exists.
    let mut table_scoped = side
        .writer
        .set_global_setting("data_inlining_row_limit", &limit.to_string())
        .is_err();
    for step in steps {
        match step {
            CrateStep::Append(batch) => {
                DuckLakeTableWriter::new(
                    Arc::clone(&side.writer),
                    Arc::new(LocalFileSystem::new()),
                )
                .unwrap()
                .with_options(&DuckLakeWriteOptions::default().with_data_inlining_row_limit(limit))
                .append_table("main", "t", std::slice::from_ref(batch))
                .await
                .unwrap();
                if table_scoped {
                    let snapshot = side.provider.get_current_snapshot().unwrap();
                    let schema = side
                        .provider
                        .get_schema_by_name("main", snapshot)
                        .unwrap()
                        .unwrap();
                    let table = side
                        .provider
                        .get_table_by_name(schema.schema_id, "t", snapshot)
                        .unwrap()
                        .unwrap();
                    side.writer
                        .set_table_setting(
                            table.table_id,
                            "data_inlining_row_limit",
                            &limit.to_string(),
                        )
                        .unwrap();
                    table_scoped = false;
                }
            },
            CrateStep::AppendOther(table, batch) => {
                DuckLakeTableWriter::new(
                    Arc::clone(&side.writer),
                    Arc::new(LocalFileSystem::new()),
                )
                .unwrap()
                .append_table("main", table, std::slice::from_ref(batch))
                .await
                .unwrap();
            },
            CrateStep::Sql(statement) => run_crate_sql(side, statement).await,
        }
    }
}

/// `rowid, *` of `main.t` at each of `snapshots`, read by this crate.
async fn crate_values(provider: &Arc<dyn MetadataProvider>, snapshots: &[i64]) -> Vec<Vec<String>> {
    let mut out = Vec::new();
    for snapshot in snapshots {
        let ctx = SessionContext::new();
        let catalog = DuckLakeCatalog::with_snapshot(Arc::clone(provider), *snapshot)
            .unwrap()
            .with_row_lineage(true);
        ctx.register_catalog("tt", Arc::new(catalog));
        let batches = ctx
            .sql("SELECT * FROM tt.main.t ORDER BY rowid")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let mut lines = Vec::new();
        for batch in &batches {
            let schema = batch.schema();
            let rowid = schema.index_of("rowid").unwrap();
            let mut order = vec![rowid];
            order.extend((0..batch.num_columns()).filter(|i| *i != rowid));
            for row in 0..batch.num_rows() {
                let cells = order
                    .iter()
                    .map(|column| {
                        let array = batch.column(*column);
                        if arrow::array::Array::is_null(array, row) {
                            "NULL".to_string()
                        } else {
                            arrow::util::display::array_value_to_string(array, row).unwrap()
                        }
                    })
                    .collect::<Vec<_>>();
                lines.push(cells.join(","));
            }
        }
        out.push(lines);
    }
    out
}

/// The change-feed windows compared after `previous`: every single snapshot,
/// and their whole range.
fn change_windows(previous: i64, head: i64) -> Vec<(i64, i64)> {
    let mut windows = (previous + 1..=head).map(|s| (s, s)).collect::<Vec<_>>();
    if head > previous + 1 {
        windows.push((previous + 1, head));
    }
    windows
}

/// Statement indices at which change feeds are compared: before each flush and
/// at the end. After a flush has moved deleted inlined rows to Parquet, neither
/// implementation's feed reports them as official's single-snapshot windows did
/// before the flush; official's own multi-snapshot window over such rows also
/// repeats rows and reports updates as inserts.
fn phases(statements: &[String]) -> Vec<usize> {
    let mut checkpoints = statements
        .iter()
        .enumerate()
        .filter(|(index, statement)| {
            *index > 0 && statement.contains("ducklake_flush_inlined_data")
        })
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    checkpoints.push(statements.len());
    checkpoints
}

/// Change-feed rows with the leading snapshot id made relative to the setup.
fn relative_changes(lines: Vec<String>, setup_head: i64) -> Vec<String> {
    lines
        .into_iter()
        .map(|line| {
            let (snapshot, rest) = line.split_once(',').unwrap();
            format!("+{},{rest}", snapshot.parse::<i64>().unwrap() - setup_head)
        })
        .collect()
}

/// Rows of arrow-rs batches as comma-separated cells, NULL as `NULL`.
fn arrow_lines(batches: &[RecordBatch]) -> Vec<String> {
    let mut lines = Vec::new();
    for batch in batches {
        for row in 0..batch.num_rows() {
            let cells = (0..batch.num_columns())
                .map(|column| {
                    let array = batch.column(column);
                    if arrow::array::Array::is_null(array, row) {
                        "NULL".to_string()
                    } else {
                        arrow::util::display::array_value_to_string(array, row).unwrap()
                    }
                })
                .collect::<Vec<_>>();
            lines.push(cells.join(","));
        }
    }
    lines
}

/// A differential scenario.
pub(crate) struct Scenario {
    pub(crate) name: &'static str,
    /// The table built by official DuckLake.
    pub(crate) official_setup: Vec<String>,
    /// The same table built by this crate.
    pub(crate) crate_setup: Vec<CrateStep>,
    /// Statements both sides apply, one snapshot each.
    pub(crate) statements: Vec<String>,
    /// The scoped `data_inlining_row_limit` the crate's side sets; the official
    /// setup sets the same.
    pub(crate) limit: usize,
    /// Sections whose difference from official is documented: reported, not
    /// asserted.
    pub(crate) known_differences: Vec<&'static str>,
}

/// A crate catalog dump, taken by a backend-specific reader once the crate's
/// handles are dropped (a DuckDB metadata file admits one process lock).
/// Runs one catalog query, rendering each row as comma-separated cells.
type QueryFn = Box<dyn Fn(&str) -> Vec<String>>;

pub(crate) struct RawDumpRows(QueryFn);

/// Dump a catalog whose metadata DuckDB can attach as `m`.
pub(crate) fn duckdb_attached_rows(metadata: Metadata) -> RawDumpRows {
    let connection = duckdb_connection();
    metadata.attach_raw(&connection);
    RawDumpRows(Box::new(move |sql| query_lines(&connection, sql)))
}

/// Run `scenario`: official on `official_metadata`, the crate on the fresh
/// catalog `open_crate` initializes, then compare.
pub(crate) async fn run_scenario<F, Fut>(
    scenario: &Scenario,
    official_metadata: &Metadata,
    dir: &Path,
    open_crate: F,
    crate_dump: impl FnOnce() -> RawDumpRows,
) -> Checked
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = CrateSide>,
{
    let official_data = dir.join("official_data");
    std::fs::create_dir_all(&official_data).unwrap();
    official(official_metadata, &official_data, &scenario.official_setup);
    let raw = |metadata: &Metadata| {
        let connection = duckdb_connection();
        metadata.attach_raw(&connection);
        raw_dump_with(|sql| query_lines(&connection, sql))
    };
    let official_setup_head = raw(official_metadata).head;
    // Change feeds are compared at checkpoints before each flush and at the
    // end, over the snapshots made since the previous checkpoint: a window that
    // reaches back across a flush reads rows the flush moved, which the feeds
    // of both implementations report inconsistently (see `phases`).
    let checkpoints = phases(&scenario.statements);
    let mut expected_changes = Vec::new();
    let mut previous = official_setup_head;
    let mut done = 0;
    for checkpoint in &checkpoints {
        official(
            official_metadata,
            &official_data,
            &scenario.statements[done..*checkpoint],
        );
        done = *checkpoint;
        let head = raw(official_metadata).head;
        let connection = duckdb_connection();
        attach_lake(&connection, official_metadata, &official_data, true);
        for (a, b) in change_windows(previous, head) {
            expected_changes.push(relative_changes(
                query_lines(
                    &connection,
                    &format!(
                        "SELECT * FROM ducklake_table_changes('lake', 'main', 't', {a}, {b}) \
                         ORDER BY snapshot_id, rowid, change_type"
                    ),
                ),
                official_setup_head,
            ));
        }
        previous = head;
    }
    let official_dump = raw(official_metadata);
    let official_snapshots = (official_setup_head..=official_dump.head).collect::<Vec<_>>();
    let expected = official_values(official_metadata, &official_data, &official_snapshots);

    let (crate_setup_head, crate_head, got, got_changes) = {
        let side = open_crate().await;
        crate_setup(&side, &scenario.crate_setup, scenario.limit).await;
        let setup_head = side.provider.get_current_snapshot().unwrap();
        let ctx = SessionContext::new();
        datafusion_ducklake::register_ducklake_functions(&ctx, Arc::clone(&side.provider));
        let mut changes = Vec::new();
        let mut previous = setup_head;
        let mut done = 0;
        for checkpoint in &checkpoints {
            for statement in &scenario.statements[done..*checkpoint] {
                run_crate_sql(&side, statement).await;
            }
            done = *checkpoint;
            let head = side.provider.get_current_snapshot().unwrap();
            for (a, b) in change_windows(previous, head) {
                let batches = ctx
                    .sql(&format!(
                        "SELECT * FROM ducklake_table_changes('main.t', {a}, {b}) \
                         ORDER BY snapshot_id, rowid, change_type"
                    ))
                    .await
                    .unwrap()
                    .collect()
                    .await
                    .unwrap();
                changes.push(relative_changes(arrow_lines(&batches), setup_head));
            }
            previous = head;
        }
        let head = side.provider.get_current_snapshot().unwrap();
        let snapshots = (setup_head..=head).collect::<Vec<_>>();
        (
            setup_head,
            head,
            crate_values(&side.provider, &snapshots).await,
            changes,
        )
    };
    assert_eq!(
        expected_changes.len(),
        got_changes.len(),
        "{}: change-feed windows differ",
        scenario.name
    );
    for (index, (want, got)) in expected_changes.iter().zip(&got_changes).enumerate() {
        assert_eq!(
            want, got,
            "{}: ducklake_table_changes differs in window {index}",
            scenario.name
        );
    }
    assert_eq!(
        official_dump.head - official_setup_head,
        crate_head - crate_setup_head,
        "{}: the statements made a different number of snapshots",
        scenario.name
    );
    for (index, (want, got)) in expected.iter().zip(&got).enumerate() {
        assert_eq!(
            want, got,
            "{}: values differ at snapshot +{index}",
            scenario.name
        );
    }
    assert!(
        expected.iter().any(|rows| !rows.is_empty()),
        "{}: no rows were read",
        scenario.name
    );

    let want = normalize(&official_dump, official_setup_head);
    let rows = crate_dump();
    let got = normalize(&raw_dump_with(|sql| (rows.0)(sql)), crate_setup_head);
    let mut failures = Vec::new();
    for (name, want_rows) in &want {
        let got_rows = &got[name];
        if want_rows == got_rows {
            continue;
        }
        let message = format!(
            "{}: `{name}` differs\nofficial:\n  {}\ncrate:\n  {}",
            scenario.name,
            want_rows.join("\n  "),
            got_rows.join("\n  ")
        );
        if scenario.known_differences.contains(&name.as_str()) {
            eprintln!("known difference — {message}");
        } else {
            failures.push(message);
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
    Checked {
        expected,
        crate_snapshots: (crate_setup_head..=crate_head).collect(),
    }
}

/// What a scenario established: official's values at each snapshot the
/// statements made (setup head first), and the crate's matching snapshots.
pub(crate) struct Checked {
    pub(crate) expected: Vec<Vec<String>>,
    pub(crate) crate_snapshots: Vec<i64>,
}

impl Checked {
    /// Require DuckDB to read `metadata` (the crate's catalog over `data`) as
    /// official's own catalog reads.
    fn duckdb_reads(&self, metadata: &Metadata, data: &Path, name: &str) {
        let got = official_values(metadata, data, &self.crate_snapshots);
        for (index, (want, got)) in self.expected.iter().zip(&got).enumerate() {
            assert_eq!(
                want, got,
                "{name}: DuckDB reads the crate's catalog differently at snapshot +{index}"
            );
        }
    }
}

fn s(statements: &[&str]) -> Vec<String> {
    statements.iter().map(|s| (*s).to_string()).collect()
}

fn id_value(ids: impl IntoIterator<Item = i32>, value: impl Fn(i32) -> String) -> RecordBatch {
    let ids = ids.into_iter().collect::<Vec<_>>();
    let values = ids.iter().map(|id| value(*id)).collect::<Vec<_>>();
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, true),
            Field::new("v", DataType::Utf8, true),
        ])),
        vec![Arc::new(Int32Array::from(ids)), Arc::new(StringArray::from(values))],
    )
    .unwrap()
}

/// Inlined and Parquet rows updated together under and over the limit, inlined
/// rows deleted, the same row updated repeatedly, a flush between updates, and
/// a final `DELETE` without `WHERE`.
pub(crate) fn scenario_basic() -> Scenario {
    Scenario {
        name: "basic",
        official_setup: s(&[
            "CALL lake.set_option('data_inlining_row_limit', 5)",
            "CREATE TABLE lake.main.t (id INTEGER, v VARCHAR)",
            "INSERT INTO lake.main.t VALUES (1, 'a'), (2, 'b'), (3, 'c')",
            "INSERT INTO lake.main.t SELECT i, 'x' || i FROM range(10, 30) r(i)",
        ]),
        crate_setup: vec![
            CrateStep::Append(id_value([1, 2, 3], |id| {
                ["a", "b", "c"][id as usize - 1].to_string()
            })),
            CrateStep::Append(id_value(10..30, |id| format!("x{id}"))),
        ],
        statements: s(&[
            "UPDATE lake.main.t SET v = 'u' WHERE id IN (2, 11, 12)",
            "UPDATE lake.main.t SET v = 'w' WHERE id >= 15",
            "DELETE FROM lake.main.t WHERE id = 13",
            "DELETE FROM lake.main.t WHERE id = 1",
            "UPDATE lake.main.t SET id = id + 100 WHERE id = 3",
            "UPDATE lake.main.t SET v = 'again' WHERE id = 103",
            "UPDATE lake.main.t SET v = 'third' WHERE id = 103",
            "CALL ducklake_flush_inlined_data('lake')",
            "UPDATE lake.main.t SET v = 'post' WHERE id IN (2, 103, 14)",
            "DELETE FROM lake.main.t WHERE id = 10",
            "DELETE FROM lake.main.t",
        ]),
        limit: LIMIT,
        known_differences: vec![],
    }
}

/// A table larger than one batch: inlined deletions spread over row groups, a
/// small update re-inlined, a large one rewritten to Parquet.
pub(crate) fn scenario_large() -> Scenario {
    Scenario {
        name: "large",
        official_setup: s(&[
            "CALL lake.set_option('data_inlining_row_limit', 5)",
            "CREATE TABLE lake.main.t (id INTEGER, v VARCHAR)",
            "INSERT INTO lake.main.t SELECT i, 'x' || i FROM range(0, 20000) r(i)",
        ]),
        crate_setup: vec![CrateStep::Append(id_value(0..20000, |id| format!("x{id}")))],
        statements: s(&[
            "DELETE FROM lake.main.t WHERE id % 5000 = 7",
            "UPDATE lake.main.t SET v = 'small' WHERE id BETWEEN 10000 AND 10003",
            "UPDATE lake.main.t SET v = 'big' WHERE id < 9000 AND id % 2 = 0",
            "DELETE FROM lake.main.t WHERE id BETWEEN 15000 AND 15002",
            "UPDATE lake.main.t SET v = 'again' WHERE id IN (10000, 2, 19999)",
        ]),
        limit: LIMIT,
        known_differences: vec![],
    }
}

/// A partitioned table: deletions per file under the limit, then updates that
/// move rows across partitions, from inlined and Parquet rows together.
pub(crate) fn scenario_partitioned() -> Scenario {
    let rows = |ids: Vec<i32>, value: &dyn Fn(i32) -> String| {
        let values = ids.iter().map(|id| value(*id)).collect::<Vec<_>>();
        let parts = ids.iter().map(|id| id % 2).collect::<Vec<_>>();
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("id", DataType::Int32, true),
                Field::new("p", DataType::Int32, true),
                Field::new("v", DataType::Utf8, true),
            ])),
            vec![
                Arc::new(Int32Array::from(ids)),
                Arc::new(Int32Array::from(parts)),
                Arc::new(StringArray::from(values)),
            ],
        )
        .unwrap()
    };
    Scenario {
        name: "partitioned",
        official_setup: s(&[
            "CALL lake.set_option('data_inlining_row_limit', 5)",
            "CREATE TABLE lake.main.t (id INTEGER, p INTEGER, v VARCHAR)",
            "ALTER TABLE lake.main.t SET PARTITIONED BY (p)",
            "INSERT INTO lake.main.t SELECT i, i % 2, 'y' || i FROM range(0, 20) r(i)",
            "INSERT INTO lake.main.t VALUES (101, 1, 'i')",
        ]),
        crate_setup: vec![
            // `CREATE TABLE` needs the schema, and an empty append would
            // write an empty data file official's `CREATE TABLE` does not.
            CrateStep::AppendOther("seed", id_value([0], |_| "seed".to_string())),
            CrateStep::Sql(
                "CREATE TABLE lake.main.t (id INTEGER, p INTEGER, v VARCHAR)".to_string(),
            ),
            CrateStep::Sql("ALTER TABLE lake.main.t SET PARTITIONED BY (p)".to_string()),
            CrateStep::Append(rows((0..20).collect(), &|id| format!("y{id}"))),
            CrateStep::Append(rows(vec![101], &|_| "i".to_string())),
        ],
        statements: s(&[
            "DELETE FROM lake.main.t WHERE id >= 12 AND id < 20",
            "DELETE FROM lake.main.t WHERE id = 0",
            "UPDATE lake.main.t SET v = 'u', p = 1 - p WHERE id IN (1, 2, 101)",
            "UPDATE lake.main.t SET v = 'many' WHERE id < 12",
            "CALL ducklake_flush_inlined_data('lake')",
        ]),
        limit: LIMIT,
        known_differences: vec![],
    }
}

/// Updates across a schema change: old inlined rows in the previous schema
/// version's table, their new versions in the current one.
pub(crate) fn scenario_schema_evolution() -> Scenario {
    let three = |ids: Vec<i32>, v: Vec<&str>, w: Vec<Option<i32>>| {
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("id", DataType::Int32, true),
                Field::new("v", DataType::Utf8, true),
                Field::new("w", DataType::Int32, true),
            ])),
            vec![
                Arc::new(Int32Array::from(ids)),
                Arc::new(StringArray::from(v)),
                Arc::new(Int32Array::from(w)),
            ],
        )
        .unwrap()
    };
    Scenario {
        name: "schema evolution",
        official_setup: s(&[
            "CALL lake.set_option('data_inlining_row_limit', 5)",
            "CREATE TABLE lake.main.t (id INTEGER, v VARCHAR)",
            "INSERT INTO lake.main.t VALUES (1, 'a'), (2, 'b')",
            "INSERT INTO lake.main.t SELECT i, 'x' || i FROM range(10, 20) r(i)",
            "BEGIN; ALTER TABLE lake.main.t ADD COLUMN w INTEGER; \
             INSERT INTO lake.main.t VALUES (3, 'c', 30); COMMIT",
        ]),
        crate_setup: vec![
            CrateStep::Append(id_value([1, 2], |id| {
                ["a", "b"][id as usize - 1].to_string()
            })),
            CrateStep::Append(id_value(10..20, |id| format!("x{id}"))),
            CrateStep::Append(three(vec![3], vec!["c"], vec![Some(30)])),
        ],
        statements: s(&[
            "UPDATE lake.main.t SET w = id * 10 WHERE id IN (1, 3, 11)",
            "DELETE FROM lake.main.t WHERE id IN (2, 12)",
            "UPDATE lake.main.t SET v = 'z' WHERE w IS NULL",
            "CALL ducklake_flush_inlined_data('lake')",
            "DELETE FROM lake.main.t WHERE id = 3",
        ]),
        limit: LIMIT,
        known_differences: vec![],
    }
}

/// Data files over the limit through their lifecycle under `DELETE` and
/// `UPDATE`, as official's `DuckLakeDelete::FlushDelete` handles them: a
/// filtered `DELETE` or `UPDATE` that empties a file drops it, with or without
/// a delete file; a second delete from a file with a delete file rewrites it
/// with each position's snapshot and removes the old one; and a `DELETE`
/// without `WHERE` writes a delete file for a file with inlined deletions
/// rather than dropping it. Under a limit of `0` the inlined steps write delete
/// files instead.
pub(crate) fn scenario_file_lifecycle(limit: usize) -> Scenario {
    let ranges = [(0, 20), (20, 40), (40, 80), (80, 100), (100, 120), (120, 140)];
    let mut official_setup = s(&[
        &format!("CALL lake.set_option('data_inlining_row_limit', {limit})"),
        "CREATE TABLE lake.main.t (id INTEGER, v VARCHAR)",
    ]);
    official_setup.extend(ranges.iter().map(|(from, to)| {
        format!("INSERT INTO lake.main.t SELECT i, 'x' || i FROM range({from}, {to}) r(i)")
    }));
    Scenario {
        name: if limit == 0 {
            "file lifecycle, limit 0"
        } else {
            "file lifecycle"
        },
        official_setup,
        crate_setup: ranges
            .iter()
            .map(|(from, to)| CrateStep::Append(id_value(*from..*to, |id| format!("x{id}"))))
            .collect(),
        statements: s(&[
            // Empties the first file, which has no delete file.
            "DELETE FROM lake.main.t WHERE id < 20",
            "DELETE FROM lake.main.t WHERE id BETWEEN 20 AND 29",
            // Empties the second file through its delete file.
            "DELETE FROM lake.main.t WHERE id BETWEEN 30 AND 39",
            "DELETE FROM lake.main.t WHERE id BETWEEN 40 AND 49",
            // Deletes from a file with a delete file, twice.
            "DELETE FROM lake.main.t WHERE id BETWEEN 50 AND 59",
            "UPDATE lake.main.t SET v = 'u' WHERE id BETWEEN 60 AND 69",
            // An UPDATE that empties a file.
            "UPDATE lake.main.t SET v = 'z' WHERE id >= 80 AND id < 100",
            "DELETE FROM lake.main.t WHERE id = 70",
            "DELETE FROM lake.main.t WHERE id = 100",
            "DELETE FROM lake.main.t WHERE id = 120",
            "DELETE FROM lake.main.t WHERE id BETWEEN 121 AND 130",
            "DELETE FROM lake.main.t",
        ]),
        limit,
        known_differences: vec![],
    }
}

pub(crate) fn scenarios() -> Vec<Scenario> {
    vec![
        scenario_basic(),
        scenario_large(),
        scenario_partitioned(),
        scenario_schema_evolution(),
        scenario_file_lifecycle(LIMIT),
        scenario_file_lifecycle(0),
    ]
}

/// Add the columns and tables DuckDB requires that this crate's SQLite
/// catalogs omit, so DuckDB can attach the catalog at `path`.
#[cfg(feature = "write-sqlite")]
pub(crate) async fn make_sqlite_catalog_duckdb_readable(path: &Path) {
    let pool = sqlx::SqlitePool::connect(&format!("sqlite:{}", path.display()))
        .await
        .unwrap();
    for statement in [
        "ALTER TABLE ducklake_snapshot ADD COLUMN next_catalog_id BIGINT NOT NULL DEFAULT 1000",
        "ALTER TABLE ducklake_snapshot ADD COLUMN next_file_id BIGINT NOT NULL DEFAULT 1000",
        "ALTER TABLE ducklake_schema ADD COLUMN schema_uuid VARCHAR",
        "UPDATE ducklake_schema SET schema_uuid = '00000000-0000-0000-0000-000000000001'",
        "ALTER TABLE ducklake_table ADD COLUMN table_uuid VARCHAR",
        "UPDATE ducklake_table SET table_uuid = '00000000-0000-0000-0000-000000000002'",
        "ALTER TABLE ducklake_data_file ADD COLUMN file_order BIGINT",
        "ALTER TABLE ducklake_data_file ADD COLUMN file_format VARCHAR",
        "ALTER TABLE ducklake_delete_file ADD COLUMN format VARCHAR",
        "CREATE TABLE ducklake_column_mapping (mapping_id BIGINT, table_id BIGINT, type VARCHAR)",
        "CREATE TABLE ducklake_file_variant_stats (data_file_id BIGINT, table_id BIGINT, column_id BIGINT, variant_path VARCHAR, shredded_type VARCHAR, column_size_bytes BIGINT, value_count BIGINT, null_count BIGINT, min_value VARCHAR, max_value VARCHAR, contains_nan BOOLEAN, extra_stats VARCHAR)",
        "CREATE TABLE ducklake_macro (schema_id BIGINT, macro_id BIGINT, macro_name VARCHAR, begin_snapshot BIGINT, end_snapshot BIGINT)",
        "CREATE TABLE ducklake_macro_impl (macro_id BIGINT, impl_id BIGINT, dialect VARCHAR, sql VARCHAR, type VARCHAR)",
        "CREATE TABLE ducklake_macro_parameters (macro_id BIGINT, impl_id BIGINT, column_id BIGINT, parameter_name VARCHAR, parameter_type VARCHAR, default_value VARCHAR, default_value_type VARCHAR)",
        "CREATE TABLE ducklake_name_mapping (mapping_id BIGINT, column_id BIGINT, source_name VARCHAR, target_field_id BIGINT, parent_column BIGINT, is_partition BOOLEAN)",
        "CREATE TABLE IF NOT EXISTS ducklake_view (view_id BIGINT, view_uuid VARCHAR, begin_snapshot BIGINT, end_snapshot BIGINT, schema_id BIGINT, view_name VARCHAR, dialect VARCHAR, sql VARCHAR, column_aliases VARCHAR)",
    ] {
        sqlx::query(statement).execute(&pool).await.unwrap();
    }
    pool.close().await;
}

/// A scratch directory, kept after the test when `DUCKLAKE_DIFF_DEBUG` is set.
fn scratch_dir() -> (Option<tempfile::TempDir>, PathBuf) {
    let temp = tempfile::TempDir::new().unwrap();
    if std::env::var_os("DUCKLAKE_DIFF_DEBUG").is_some() {
        let path = temp.keep();
        eprintln!("scratch directory {}", path.display());
        (None, path)
    } else {
        let path = temp.path().to_path_buf();
        (Some(temp), path)
    }
}

#[cfg(feature = "write-sqlite")]
mod sqlite {
    use super::*;
    use datafusion_ducklake::{SqliteMetadataProvider, SqliteMetadataWriter};

    async fn run(scenario: Scenario) {
        run_in(scenario).await;
    }

    /// Run `scenario`, returning its scratch directory: the crate's catalog is
    /// `crate.db` over `crate_data`, readable by DuckDB.
    async fn run_in(scenario: Scenario) -> (Option<tempfile::TempDir>, PathBuf) {
        let (temp, dir) = scratch_dir();
        let crate_path = dir.join("crate.db");
        let data = dir.join("crate_data");
        std::fs::create_dir_all(&data).unwrap();
        let url = format!("sqlite:{}?mode=rwc", crate_path.display());
        let checked = run_scenario(
            &scenario,
            &Metadata::Sqlite(dir.join("official.db")),
            &dir,
            || async move {
                let writer = SqliteMetadataWriter::new_with_init(&url).await.unwrap();
                writer.set_data_path(data.to_str().unwrap()).unwrap();
                CrateSide {
                    writer: Arc::new(writer),
                    provider: Arc::new(SqliteMetadataProvider::new(&url).await.unwrap()),
                }
            },
            || duckdb_attached_rows(Metadata::Sqlite(crate_path.clone())),
        )
        .await;
        make_sqlite_catalog_duckdb_readable(&crate_path).await;
        checked.duckdb_reads(
            &Metadata::Sqlite(crate_path.clone()),
            &dir.join("crate_data"),
            scenario.name,
        );
        (temp, dir)
    }

    /// DuckDB prunes files and inlined tables by `ducklake_table_column_stats`,
    /// so the statistics of inlined rows must cover their values.
    #[tokio::test(flavor = "multi_thread")]
    async fn duckdb_finds_inlined_update_values() {
        let (_temp, dir) = run_in(scenario_large()).await;
        let rows = duckdb_query(
            &Metadata::Sqlite(dir.join("crate.db")),
            &dir.join("crate_data"),
            "SELECT count(*) FROM lake.main.t WHERE v = 'again'",
        );
        assert_eq!(rows, vec!["3".to_string()]);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn basic() {
        run(scenario_basic()).await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn large() {
        run(scenario_large()).await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn partitioned() {
        run(scenario_partitioned()).await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn schema_evolution() {
        run(scenario_schema_evolution()).await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn file_lifecycle() {
        run(scenario_file_lifecycle(LIMIT)).await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn file_lifecycle_zero_limit() {
        run(scenario_file_lifecycle(0)).await;
    }
}

/// Statements planned at one snapshot and executed one after another, as two
/// sessions would race.
#[cfg(feature = "write-sqlite")]
mod conflicts {
    use super::*;
    use datafusion::physical_plan::{ExecutionPlan, collect};
    use datafusion_ducklake::{SqliteMetadataProvider, SqliteMetadataWriter};

    async fn side(dir: &Path) -> CrateSide {
        let url = format!("sqlite:{}?mode=rwc", dir.join("c.db").display());
        let data = dir.join("data");
        std::fs::create_dir_all(&data).unwrap();
        let writer = SqliteMetadataWriter::new_with_init(&url).await.unwrap();
        writer.set_data_path(data.to_str().unwrap()).unwrap();
        let side = CrateSide {
            writer: Arc::new(writer),
            provider: Arc::new(SqliteMetadataProvider::new(&url).await.unwrap()),
        };
        crate_setup(
            &side,
            &[
                CrateStep::Append(id_value([1, 2, 3], |id| format!("i{id}"))),
                CrateStep::Append(id_value(10..30, |id| format!("x{id}"))),
            ],
            LIMIT,
        )
        .await;
        side
    }

    async fn plan(side: &CrateSide, sql: &str) -> (SessionContext, Arc<dyn ExecutionPlan>) {
        let ctx = SessionContext::new();
        let catalog =
            DuckLakeCatalog::with_writer(Arc::clone(&side.provider), Arc::clone(&side.writer))
                .unwrap();
        DuckLakeCatalog::register(&ctx, "lake", catalog);
        let plan = ctx
            .sql(sql)
            .await
            .unwrap()
            .create_physical_plan()
            .await
            .unwrap();
        (ctx, plan)
    }

    /// The second of two statements that record inlined deletions — of Parquet
    /// rows or by updating inlined rows — conflicts, as official's
    /// `inlined_delete` check does, and leaves the catalog as the first left it.
    #[tokio::test(flavor = "multi_thread")]
    async fn concurrent_inlined_deletions_conflict() {
        for (first, second) in [
            (
                "DELETE FROM lake.main.t WHERE id = 11",
                "DELETE FROM lake.main.t WHERE id = 12",
            ),
            (
                "UPDATE lake.main.t SET v = 'a' WHERE id = 1",
                "UPDATE lake.main.t SET v = 'b' WHERE id = 11",
            ),
            (
                "DELETE FROM lake.main.t WHERE id = 2",
                "DELETE FROM lake.main.t",
            ),
        ] {
            let temp = tempfile::TempDir::new().unwrap();
            let side = side(temp.path()).await;
            let (ctx_a, plan_a) = plan(&side, first).await;
            let (ctx_b, plan_b) = plan(&side, second).await;
            collect(plan_a, ctx_a.task_ctx()).await.unwrap();
            let head = side.provider.get_current_snapshot().unwrap();
            let error = collect(plan_b, ctx_b.task_ctx()).await.unwrap_err();
            assert!(
                error.to_string().contains("conflict") || error.to_string().contains("changed"),
                "`{second}` after `{first}`: {error}"
            );
            assert_eq!(side.provider.get_current_snapshot().unwrap(), head);
        }
    }

    /// Without a scoped inlining limit, `DELETE` and `UPDATE` keep writing
    /// delete files and Parquet, and inlined rows are still updated in place.
    #[tokio::test(flavor = "multi_thread")]
    async fn zero_limit_writes_delete_files() {
        let temp = tempfile::TempDir::new().unwrap();
        let side = side(temp.path()).await;
        side.writer
            .set_global_setting("data_inlining_row_limit", "0")
            .unwrap();
        run_crate_sql(&side, "DELETE FROM lake.main.t WHERE id = 11").await;
        run_crate_sql(&side, "UPDATE lake.main.t SET v = 'u' WHERE id IN (1, 12)").await;
        let pool =
            sqlx::SqlitePool::connect(&format!("sqlite:{}", temp.path().join("c.db").display()))
                .await
                .unwrap();
        let inlined_deletes: i64 = sqlx::query_scalar(
            "SELECT count(*) FROM sqlite_master WHERE name LIKE 'ducklake_inlined_delete_%'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(inlined_deletes, 0);
        let delete_files: i64 = sqlx::query_scalar(
            "SELECT count(*) FROM ducklake_delete_file WHERE end_snapshot IS NULL",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(delete_files, 1);
        let head = side.provider.get_current_snapshot().unwrap();
        let rows = crate_values(&side.provider, &[head]).await.remove(0);
        assert!(rows.contains(&"0,1,u".to_string()), "{rows:?}");
        assert!(rows.contains(&"5,12,u".to_string()), "{rows:?}");
        assert!(!rows.iter().any(|row| row.starts_with("4,")), "{rows:?}");
    }
}

/// A `DELETE` that rewrites a delete file dates its deletions to the snapshot
/// after the one it read. When other snapshots commit first, it re-dates them
/// to its actual commit snapshot and retries; a concurrent rewrite of the same
/// delete file still conflicts. Official DuckLake (1.5.5) instead commits the
/// deletions dated to the snapshot after its read, and lets two concurrent
/// rewrites of one delete file both commit, leaving two live delete files.
pub(crate) mod redate {
    use super::*;
    use datafusion::physical_plan::{ExecutionPlan, collect};

    async fn plan(side: &CrateSide, sql: &str) -> (SessionContext, Arc<dyn ExecutionPlan>) {
        let ctx = SessionContext::new();
        let catalog =
            DuckLakeCatalog::with_writer(Arc::clone(&side.provider), Arc::clone(&side.writer))
                .unwrap();
        DuckLakeCatalog::register(&ctx, "lake", catalog);
        let plan = ctx
            .sql(sql)
            .await
            .unwrap()
            .create_physical_plan()
            .await
            .unwrap();
        (ctx, plan)
    }

    /// `main.t`: ids 1..3 inlined, 10..29 in one Parquet file, and a delete
    /// file of 10..16 on it.
    pub(crate) async fn setup(side: &CrateSide) {
        crate_setup(
            side,
            &[
                CrateStep::Append(id_value([1, 2, 3], |id| format!("i{id}"))),
                CrateStep::Append(id_value(10..30, |id| format!("x{id}"))),
                CrateStep::Sql("DELETE FROM lake.main.t WHERE id BETWEEN 10 AND 16".to_string()),
            ],
            LIMIT,
        )
        .await;
    }

    fn ids(rows: &[String]) -> Vec<i64> {
        rows.iter()
            .map(|row| row.split(',').nth(1).unwrap().parse().unwrap())
            .collect()
    }

    fn live_delete_files(side: &CrateSide) -> usize {
        let head = side.provider.get_current_snapshot().unwrap();
        let snapshot = side
            .provider
            .get_schema_by_name("main", head)
            .unwrap()
            .unwrap();
        let table = side
            .provider
            .get_table_by_name(snapshot.schema_id, "t", head)
            .unwrap()
            .unwrap();
        side.provider
            .get_table_files_for_select(table.table_id, head)
            .unwrap()
            .iter()
            .filter(|file| file.delete_file.is_some())
            .count()
    }

    /// A commit to another table between the statement's read and its commit:
    /// the statement commits, its deletions dated to its commit snapshot.
    pub(crate) async fn interleaved_commit_to_another_table(side: &CrateSide) {
        setup(side).await;
        let (ctx, plan) = plan(side, "DELETE FROM lake.main.t WHERE id BETWEEN 20 AND 26").await;
        DuckLakeTableWriter::new(Arc::clone(&side.writer), Arc::new(LocalFileSystem::new()))
            .unwrap()
            .append_table("main", "u", &[id_value([1], |_| "u".to_string())])
            .await
            .unwrap();
        let between = side.provider.get_current_snapshot().unwrap();
        collect(plan, ctx.task_ctx()).await.unwrap();
        let head = side.provider.get_current_snapshot().unwrap();
        assert!(head > between);
        let values = crate_values(&side.provider, &[between, head]).await;
        let mut kept = vec![1, 2, 3];
        kept.extend(17..30);
        assert_eq!(ids(&values[0]), kept, "before the DELETE committed");
        kept.retain(|id| !(20..=26).contains(id));
        assert_eq!(ids(&values[1]), kept, "after the DELETE committed");
        assert_eq!(live_delete_files(side), 1);
    }

    /// A change another transaction commits to `main.t` between a statement's
    /// read and its commit.
    #[derive(Clone, Copy, Debug)]
    pub(crate) enum SameTableChange {
        /// Rows inlined into it.
        InlinedInsert,
        /// A Parquet file appended to it.
        ParquetInsert,
        /// A partition spec set on it.
        Alter,
        /// A sort order set on it.
        AlterSort,
        /// Its file with deletes rewritten.
        Compaction,
        /// Its Parquet files merged.
        Merge,
    }

    /// The table a conflict scenario starts from.
    #[derive(Clone, Copy, Debug)]
    pub(crate) enum ConflictSetup {
        /// [`setup`]: inlined rows, and a Parquet file with a delete file,
        /// under the limit [`LIMIT`].
        Inlined,
        /// Two Parquet files of ids 10..49, no deletions, under `limit`.
        Plain(usize),
        /// [`ConflictSetup::Plain`] with ids 10..16 deleted.
        WithDeleteFile(usize),
    }

    async fn conflict_setup(side: &CrateSide, setup_kind: ConflictSetup) {
        let (limit, delete) = match setup_kind {
            ConflictSetup::Inlined => return setup(side).await,
            ConflictSetup::Plain(limit) => (limit, false),
            ConflictSetup::WithDeleteFile(limit) => (limit, true),
        };
        let mut steps = vec![
            CrateStep::Append(id_value(10..30, |id| format!("x{id}"))),
            CrateStep::Append(id_value(30..50, |id| format!("x{id}"))),
        ];
        if delete {
            steps.push(CrateStep::Sql(
                "DELETE FROM lake.main.t WHERE id BETWEEN 10 AND 16".to_string(),
            ));
        }
        crate_setup(side, &steps, limit).await;
    }

    /// Official's conflict check (`CheckForConflicts`): a `DELETE` or `UPDATE`
    /// of a table another transaction inserted into, altered or compacted
    /// since the statement read it aborts, without retrying or committing;
    /// one official lets through commits.
    pub(crate) async fn conflicting_change_to_the_same_table(
        side: &CrateSide,
        statement: &str,
        change: SameTableChange,
        conflicts: bool,
    ) {
        conflicting_change_after(side, ConflictSetup::Inlined, statement, change, conflicts).await;
    }

    /// [`conflicting_change_to_the_same_table`] from `setup_kind`.
    pub(crate) async fn conflicting_change_after(
        side: &CrateSide,
        setup_kind: ConflictSetup,
        statement: &str,
        change: SameTableChange,
        conflicts: bool,
    ) {
        conflict_setup(side, setup_kind).await;
        let (ctx, plan) = plan(side, statement).await;
        apply_change(side, change).await;
        let head = side.provider.get_current_snapshot().unwrap();
        let before = crate_values(&side.provider, &[head]).await;
        let result = collect(plan, ctx.task_ctx()).await;
        if !conflicts {
            result.unwrap_or_else(|e| panic!("{statement} after {change:?}: {e}"));
            assert_eq!(side.provider.get_current_snapshot().unwrap(), head + 1);
            return;
        }
        let error = result.unwrap_err().to_string();
        assert!(
            error.to_lowercase().contains("conflict"),
            "{statement} after {change:?}: {error}"
        );
        assert!(
            !error.contains("Exceeded the maximum retry count"),
            "{statement} after {change:?} was retried: {error}"
        );
        if !matches!(change, SameTableChange::Compaction | SameTableChange::Merge) {
            // Official checks an UPDATE's insert before its deletion.
            let action = if statement.starts_with("UPDATE")
                && matches!(change, SameTableChange::Alter | SameTableChange::AlterSort)
            {
                "insert into table"
            } else {
                "delete from table"
            };
            assert!(
                error.contains(&format!("Transaction conflict - attempting to {action}")),
                "{statement} after {change:?} from {setup_kind:?}: {error}"
            );
        }
        assert_eq!(side.provider.get_current_snapshot().unwrap(), head);
        assert_eq!(crate_values(&side.provider, &[head]).await, before);
    }

    /// Commit `change` to `main.t`, as another transaction.
    pub(crate) async fn apply_change(side: &CrateSide, change: SameTableChange) {
        match change {
            SameTableChange::InlinedInsert => {
                DuckLakeTableWriter::new(
                    Arc::clone(&side.writer),
                    Arc::new(LocalFileSystem::new()),
                )
                .unwrap()
                .with_options(&DuckLakeWriteOptions::default().with_data_inlining_row_limit(LIMIT))
                .append_table("main", "t", &[id_value([90], |_| "n".to_string())])
                .await
                .unwrap();
            },
            SameTableChange::ParquetInsert => {
                DuckLakeTableWriter::new(
                    Arc::clone(&side.writer),
                    Arc::new(LocalFileSystem::new()),
                )
                .unwrap()
                .with_options(&DuckLakeWriteOptions::default().with_data_inlining_row_limit(0))
                .append_table("main", "t", &[id_value(90..100, |_| "n".to_string())])
                .await
                .unwrap();
            },
            SameTableChange::Alter => {
                run_crate_sql(side, "ALTER TABLE lake.main.t SET PARTITIONED BY (id)").await;
            },
            SameTableChange::AlterSort => {
                run_crate_sql(side, "ALTER TABLE lake.main.t SET SORTED BY (id)").await;
            },
            SameTableChange::Compaction => {
                let compact = SessionContext::new();
                let catalog = DuckLakeCatalog::with_writer(
                    Arc::clone(&side.provider),
                    Arc::clone(&side.writer),
                )
                .unwrap();
                DuckLakeCatalog::register(&compact, "lake", catalog);
                let provider = compact
                    .catalog("lake")
                    .unwrap()
                    .schema("main")
                    .unwrap()
                    .table("t")
                    .await
                    .unwrap()
                    .unwrap();
                let table = (provider.as_ref() as &dyn std::any::Any)
                    .downcast_ref::<datafusion_ducklake::DuckLakeTable>()
                    .unwrap()
                    .clone();
                let rewritten = table
                    .rewrite_data_files(
                        &compact.state(),
                        datafusion_ducklake::RewriteOptions {
                            delete_threshold: 0.0,
                            data_file_ids: None,
                        },
                    )
                    .await
                    .unwrap();
                assert!(rewritten.did_work());
            },
            SameTableChange::Merge => {
                let compact = SessionContext::new();
                let catalog = DuckLakeCatalog::with_writer(
                    Arc::clone(&side.provider),
                    Arc::clone(&side.writer),
                )
                .unwrap();
                DuckLakeCatalog::register(&compact, "lake", catalog);
                let provider = compact
                    .catalog("lake")
                    .unwrap()
                    .schema("main")
                    .unwrap()
                    .table("t")
                    .await
                    .unwrap()
                    .unwrap();
                let table = (provider.as_ref() as &dyn std::any::Any)
                    .downcast_ref::<datafusion_ducklake::DuckLakeTable>()
                    .unwrap()
                    .clone();
                let merged = table
                    .merge_adjacent_files(
                        &compact.state(),
                        datafusion_ducklake::MergeOptions::default(),
                    )
                    .await
                    .unwrap();
                assert!(merged.did_work());
            },
        }
    }

    /// An inlined-data flush that `change` interleaves with: the change commits
    /// while the flush writes its first object. With `file_deletion`, the flush
    /// moves an inlined deletion of a Parquet row into a delete file, which
    /// deletes from the table as official's flush does (`AddDeletes`), so a
    /// change official's `delete from table` check refuses aborts it, worded as
    /// `conflict` (`None` when it commits). Otherwise it moves inlined rows.
    pub(crate) async fn flush_after_concurrent_change(
        side: &CrateSide,
        file_deletion: bool,
        change: SameTableChange,
        conflict: Option<&str>,
    ) {
        conflict_setup(side, ConflictSetup::Plain(LIMIT)).await;
        if matches!(change, SameTableChange::Merge) {
            // Files without deletions to merge.
            DuckLakeTableWriter::new(Arc::clone(&side.writer), Arc::new(LocalFileSystem::new()))
                .unwrap()
                .with_options(&DuckLakeWriteOptions::default().with_data_inlining_row_limit(LIMIT))
                .append_table("main", "t", &[id_value(50..70, |id| format!("x{id}"))])
                .await
                .unwrap();
        }
        if file_deletion {
            run_crate_sql(side, "DELETE FROM lake.main.t WHERE id = 12").await;
        } else {
            DuckLakeTableWriter::new(Arc::clone(&side.writer), Arc::new(LocalFileSystem::new()))
                .unwrap()
                .with_options(&DuckLakeWriteOptions::default().with_data_inlining_row_limit(LIMIT))
                .append_table("main", "t", &[id_value([1, 2], |id| format!("i{id}"))])
                .await
                .unwrap();
        }
        let other = Arc::new(CrateSide {
            writer: Arc::clone(&side.writer),
            provider: Arc::clone(&side.provider),
        });
        let fired = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let hook_fired = Arc::clone(&fired);
        let store = Arc::new(EveryPutStore {
            inner: LocalFileSystem::new(),
            armed: std::sync::atomic::AtomicBool::new(true),
            puts: std::sync::atomic::AtomicUsize::new(0),
            hook: Box::new(move || {
                let other = Arc::clone(&other);
                let fired = Arc::clone(&hook_fired);
                Box::pin(async move {
                    if !fired.swap(true, std::sync::atomic::Ordering::SeqCst) {
                        apply_change(&other, change).await;
                    }
                })
            }),
        });
        let ctx = SessionContext::new();
        ctx.register_object_store(
            &url::Url::parse("file:///").unwrap(),
            Arc::clone(&store) as _,
        );
        let catalog =
            DuckLakeCatalog::with_writer(Arc::clone(&side.provider), Arc::clone(&side.writer))
                .unwrap();
        DuckLakeCatalog::register(&ctx, "lake", catalog);
        datafusion_ducklake::register_ducklake_maintenance_functions(&ctx);
        let handle =
            DuckLakeCatalog::with_writer(Arc::clone(&side.provider), Arc::clone(&side.writer))
                .unwrap();
        let result =
            match execute_ducklake_sql(&ctx, &handle, "CALL ducklake_flush_inlined_data('lake')")
                .await
            {
                Ok(frame) => frame.collect().await.map(|_| ()),
                Err(error) => Err(error),
            };
        store
            .armed
            .store(false, std::sync::atomic::Ordering::SeqCst);
        assert!(
            fired.load(std::sync::atomic::Ordering::SeqCst),
            "{change:?} was not interleaved"
        );
        let head = side.provider.get_current_snapshot().unwrap();
        let changes = side
            .provider
            .list_snapshot_changes()
            .unwrap()
            .into_iter()
            .find(|snapshot| snapshot.snapshot_id == head)
            .and_then(|snapshot| snapshot.changes_made)
            .unwrap_or_default();
        match conflict {
            None => {
                result.unwrap_or_else(|e| panic!("flush after {change:?}: {e}"));
                assert!(changes.contains("inline_flush:"), "{changes}");
            },
            Some(described) => {
                let error = result.unwrap_err().to_string();
                assert!(
                    error.contains("Transaction conflict - attempting to delete from table")
                        && error.contains(&format!("but another transaction has {described}")),
                    "flush after {change:?}: {error}"
                );
                assert!(!changes.contains("deleted_from_table:"), "{changes}");
            },
        }
    }

    /// [`flush_after_concurrent_change`]'s cases: official's `delete from
    /// table` check refuses a concurrent insert, alter or merge; its flush
    /// check refuses none of the inserts.
    pub(crate) const FLUSH_CASES: [(bool, SameTableChange, Option<&str>); 6] = [
        (
            true,
            SameTableChange::InlinedInsert,
            Some("inserted into it"),
        ),
        (
            true,
            SameTableChange::ParquetInsert,
            Some("inserted into it"),
        ),
        (true, SameTableChange::Alter, Some("altered it")),
        (true, SameTableChange::Merge, Some("compacted it")),
        (false, SameTableChange::InlinedInsert, None),
        (false, SameTableChange::ParquetInsert, None),
    ];

    /// Statements and same-table changes, and whether official's check
    /// conflicts on them: deleting Parquet rows (here, rewriting a delete file)
    /// conflicts with all four; deleting inlined rows with all but compaction.
    pub(crate) const SAME_TABLE_CASES: [(&str, SameTableChange, bool); 10] = [
        (
            "DELETE FROM lake.main.t WHERE id BETWEEN 20 AND 26",
            SameTableChange::InlinedInsert,
            true,
        ),
        (
            "DELETE FROM lake.main.t WHERE id BETWEEN 20 AND 26",
            SameTableChange::ParquetInsert,
            true,
        ),
        (
            "DELETE FROM lake.main.t WHERE id BETWEEN 20 AND 26",
            SameTableChange::Alter,
            true,
        ),
        (
            "DELETE FROM lake.main.t WHERE id BETWEEN 20 AND 26",
            SameTableChange::Compaction,
            true,
        ),
        (
            "DELETE FROM lake.main.t WHERE id = 2",
            SameTableChange::InlinedInsert,
            true,
        ),
        (
            "DELETE FROM lake.main.t WHERE id = 2",
            SameTableChange::ParquetInsert,
            true,
        ),
        (
            "DELETE FROM lake.main.t WHERE id = 2",
            SameTableChange::Alter,
            true,
        ),
        (
            "DELETE FROM lake.main.t WHERE id = 2",
            SameTableChange::Compaction,
            false,
        ),
        (
            "UPDATE lake.main.t SET v = 'u' WHERE id = 20",
            SameTableChange::InlinedInsert,
            true,
        ),
        (
            "UPDATE lake.main.t SET v = 'u' WHERE id = 20",
            SameTableChange::Alter,
            true,
        ),
    ];

    /// Same-table changes against statements that commit without staging —
    /// the first deletion from a file over the limit, or with the limit 0 —
    /// and against ones rewriting a delete file under the limit 0: official's
    /// check conflicts on every one.
    pub(crate) const DIRECT_CASES: [(ConflictSetup, &str, SameTableChange); 14] = [
        (
            ConflictSetup::Plain(0),
            DELETE_RANGE,
            SameTableChange::InlinedInsert,
        ),
        (
            ConflictSetup::Plain(0),
            DELETE_RANGE,
            SameTableChange::ParquetInsert,
        ),
        (
            ConflictSetup::Plain(0),
            DELETE_RANGE,
            SameTableChange::Alter,
        ),
        (
            ConflictSetup::Plain(0),
            DELETE_RANGE,
            SameTableChange::Merge,
        ),
        (
            ConflictSetup::Plain(LIMIT),
            DELETE_RANGE,
            SameTableChange::ParquetInsert,
        ),
        (
            ConflictSetup::Plain(LIMIT),
            DELETE_RANGE,
            SameTableChange::AlterSort,
        ),
        (
            ConflictSetup::WithDeleteFile(0),
            DELETE_RANGE,
            SameTableChange::ParquetInsert,
        ),
        (
            ConflictSetup::WithDeleteFile(0),
            DELETE_RANGE,
            SameTableChange::Compaction,
        ),
        (
            ConflictSetup::Plain(0),
            UPDATE_RANGE,
            SameTableChange::InlinedInsert,
        ),
        (
            ConflictSetup::Plain(0),
            UPDATE_RANGE,
            SameTableChange::ParquetInsert,
        ),
        (
            ConflictSetup::Plain(0),
            UPDATE_RANGE,
            SameTableChange::AlterSort,
        ),
        (
            ConflictSetup::Plain(LIMIT),
            UPDATE_RANGE,
            SameTableChange::ParquetInsert,
        ),
        (
            ConflictSetup::WithDeleteFile(0),
            UPDATE_RANGE,
            SameTableChange::AlterSort,
        ),
        (
            ConflictSetup::WithDeleteFile(LIMIT),
            UPDATE_RANGE,
            SameTableChange::InlinedInsert,
        ),
    ];

    /// Two `DELETE`s of different Parquet files of one table, both reading
    /// the same snapshot: official's conflict check applies its insert rules
    /// only to a transaction that inserts, so both commit. With
    /// `delete_files`, each file has a delete file the statement rewrites.
    pub(crate) async fn concurrent_deletes_of_different_files(
        side: &CrateSide,
        delete_files: bool,
    ) {
        conflict_setup(side, ConflictSetup::Plain(0)).await;
        let mut kept: Vec<i64> = (10..50).collect();
        if delete_files {
            run_crate_sql(side, "DELETE FROM lake.main.t WHERE id = 10").await;
            run_crate_sql(side, "DELETE FROM lake.main.t WHERE id = 30").await;
            kept.retain(|id| *id != 10 && *id != 30);
        }
        let (first_ctx, first) = plan(side, "DELETE FROM lake.main.t WHERE id = 12").await;
        let (second_ctx, second) = plan(side, "DELETE FROM lake.main.t WHERE id = 40").await;
        let before = side.provider.list_snapshots().unwrap().len();
        collect(first, first_ctx.task_ctx()).await.unwrap();
        collect(second, second_ctx.task_ctx())
            .await
            .expect("a DELETE of another file of the table commits");
        let head = side.provider.get_current_snapshot().unwrap();
        assert_eq!(
            side.provider.list_snapshots().unwrap().len(),
            before + 2,
            "both statements committed"
        );
        kept.retain(|id| *id != 12 && *id != 40);
        let values = crate_values(&side.provider, &[head]).await;
        assert_eq!(ids(&values[0]), kept);
        assert_eq!(live_delete_files(side), 2);
    }

    /// `snapshot:id` (and `change_type:` first, for `ducklake_table_changes`)
    /// of the rows `function` returns over `[a, b]`, read by this crate, sorted.
    async fn feed(side: &CrateSide, function: &str, a: i64, b: i64) -> Vec<String> {
        let ctx = SessionContext::new();
        datafusion_ducklake::register_ducklake_functions(&ctx, Arc::clone(&side.provider));
        let column = if function == "ducklake_table_changes" {
            "concat(change_type, ':', snapshot_id, ':', id)"
        } else {
            "concat(snapshot_id, ':', id)"
        };
        let batches = ctx
            .sql(&format!(
                "SELECT {column} FROM {function}('main.t', {a}, {b}) ORDER BY 1"
            ))
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        arrow_lines(&batches)
    }

    /// A `DELETE` writing a delete file for a Parquet file another transaction
    /// recorded an inlined deletion of since the statement read it. Official
    /// DuckLake's conflict check never compares the two, and both commit; so
    /// does this crate, deletions combining at every snapshot and in the
    /// change feeds. When the inlined deletion deleted a row the delete file
    /// deletes too (`overlap`), the `DELETE` aborts instead: official commits
    /// both, after which it reads back deleted rows of the file. With
    /// `delete_file`, the statement rewrites the file's live delete file.
    pub(crate) async fn delete_beside_concurrent_inlined_deletion(
        side: &CrateSide,
        delete_file: bool,
        overlap: bool,
    ) {
        let setup_kind = if delete_file {
            ConflictSetup::WithDeleteFile(LIMIT)
        } else {
            ConflictSetup::Plain(LIMIT)
        };
        conflict_setup(side, setup_kind).await;
        let base = side.provider.get_current_snapshot().unwrap();
        let (ctx, plan) = plan(side, DELETE_RANGE).await;
        let concurrent = if overlap {
            [22, 40]
        } else {
            [28, 40]
        };
        run_crate_sql(
            side,
            &format!(
                "DELETE FROM lake.main.t WHERE id IN ({}, {})",
                concurrent[0], concurrent[1]
            ),
        )
        .await;
        let between = side.provider.get_current_snapshot().unwrap();
        let snapshots = side.provider.list_snapshots().unwrap().len();
        let mut kept: Vec<i64> = (10..50).collect();
        if delete_file {
            kept.retain(|id| !(10..=16).contains(id));
        }
        let result = collect(plan, ctx.task_ctx()).await;
        let head = side.provider.get_current_snapshot().unwrap();
        let at_between = {
            let mut ids = kept.clone();
            ids.retain(|id| !concurrent.contains(id));
            ids
        };
        if overlap {
            let error = result.unwrap_err().to_string();
            assert!(error.contains("gained an inlined delete"), "{error}");
            assert_eq!(head, between, "the DELETE committed nothing");
            let values = crate_values(&side.provider, &[base, head]).await;
            assert_eq!(ids(&values[0]), kept);
            assert_eq!(ids(&values[1]), at_between);
            return;
        }
        assert_eq!(result.unwrap().len(), 1);
        assert_eq!(
            side.provider.list_snapshots().unwrap().len(),
            snapshots + 1,
            "both statements committed"
        );
        let at_head = {
            let mut ids = at_between.clone();
            ids.retain(|id| !(20..=26).contains(id));
            ids
        };
        let values = crate_values(&side.provider, &[base, between, head]).await;
        assert_eq!(ids(&values[0]), kept, "before either");
        assert_eq!(ids(&values[1]), at_between, "after the inlined deletion");
        assert_eq!(ids(&values[2]), at_head, "after both");

        // `count(*)` at the head comes from catalog counts.
        let ctx = SessionContext::new();
        let catalog =
            DuckLakeCatalog::with_writer(Arc::clone(&side.provider), Arc::clone(&side.writer))
                .unwrap();
        DuckLakeCatalog::register(&ctx, "lake", catalog);
        let counted = ctx
            .sql("SELECT count(*) FROM lake.main.t")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        assert_eq!(arrow_lines(&counted), [at_head.len().to_string()]);

        let sorted = |mut lines: Vec<String>| {
            lines.sort();
            lines
        };
        let range: Vec<String> = (20..=26).map(|id| format!("{head}:{id}")).collect();
        let inlined: Vec<String> = concurrent
            .iter()
            .map(|id| format!("{between}:{id}"))
            .collect();
        let both = [inlined.clone(), range.clone()].concat();
        assert_eq!(
            feed(side, "ducklake_table_deletions", between, between).await,
            sorted(inlined)
        );
        assert_eq!(
            feed(side, "ducklake_table_deletions", head, head).await,
            sorted(range)
        );
        assert_eq!(
            feed(side, "ducklake_table_deletions", between, head).await,
            sorted(both.clone())
        );
        assert_eq!(
            feed(side, "ducklake_table_changes", between, head).await,
            sorted(both.iter().map(|line| format!("delete:{line}")).collect())
        );
    }

    /// A `DELETE` recording an inlined deletion of a Parquet row after another
    /// transaction committed a delete file for the same data file since the
    /// statement read it. Official commits both; so does this crate when the
    /// delete file deletes none of the statement's rows, deletions combining at
    /// every snapshot, in `count(*)` and in the change feed. When it deletes one
    /// of them (`overlap`) the statement aborts: official commits both, after
    /// which it reads deleted rows of the file back and miscounts `count(*)`.
    pub(crate) async fn inlined_deletion_after_concurrent_delete_file(
        side: &CrateSide,
        overlap: bool,
    ) {
        conflict_setup(side, ConflictSetup::Plain(LIMIT)).await;
        let base = side.provider.get_current_snapshot().unwrap();
        let deleted = if overlap {
            [22, 40]
        } else {
            [28, 40]
        };
        let statement = format!(
            "DELETE FROM lake.main.t WHERE id IN ({}, {})",
            deleted[0], deleted[1]
        );
        let (ctx, plan) = plan(side, &statement).await;
        run_crate_sql(side, DELETE_RANGE).await;
        let between = side.provider.get_current_snapshot().unwrap();
        let snapshots = side.provider.list_snapshots().unwrap().len();
        let result = collect(plan, ctx.task_ctx()).await;
        let all: Vec<i64> = (10..50).collect();
        let at_between: Vec<i64> = (10..50).filter(|id| !(20..=26).contains(id)).collect();
        let at_head = if overlap {
            let error = result.unwrap_err().to_string();
            assert!(
                error.contains("got a delete file since snapshot"),
                "{error}"
            );
            assert_eq!(side.provider.get_current_snapshot().unwrap(), between);
            at_between.clone()
        } else {
            result.unwrap();
            // On PostgreSQL the refused attempt draws a snapshot id too.
            assert!(side.provider.get_current_snapshot().unwrap() > between);
            assert_eq!(side.provider.list_snapshots().unwrap().len(), snapshots + 1);
            let mut ids = at_between.clone();
            ids.retain(|id| !deleted.contains(id));
            ids
        };
        let head = side.provider.get_current_snapshot().unwrap();
        let values = crate_values(&side.provider, &[base, between, head]).await;
        assert_eq!(ids(&values[0]), all, "before either");
        assert_eq!(ids(&values[1]), at_between, "after the delete file");
        assert_eq!(ids(&values[2]), at_head, "at the head");
        let ctx = SessionContext::new();
        let catalog =
            DuckLakeCatalog::with_writer(Arc::clone(&side.provider), Arc::clone(&side.writer))
                .unwrap();
        DuckLakeCatalog::register(&ctx, "lake", catalog);
        let counted = ctx
            .sql("SELECT count(*) FROM lake.main.t")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        assert_eq!(arrow_lines(&counted), [at_head.len().to_string()]);
        if !overlap {
            let mut inlined: Vec<String> =
                deleted.iter().map(|id| format!("{head}:{id}")).collect();
            inlined.sort();
            assert_eq!(
                feed(side, "ducklake_table_deletions", head, head).await,
                inlined
            );
        }
    }

    /// A `DELETE` emptying a Parquet file another transaction recorded an
    /// inlined deletion of since the statement read it, which it would drop,
    /// conflicts: official commits both, after which its change feeds lose
    /// the drop's deletions in a window starting at the inlined deletion.
    pub(crate) async fn drop_after_concurrent_inlined_deletion(side: &CrateSide) {
        conflict_setup(side, ConflictSetup::Plain(LIMIT)).await;
        let (ctx, plan) = plan(side, "DELETE FROM lake.main.t WHERE id < 30").await;
        run_crate_sql(side, "DELETE FROM lake.main.t WHERE id IN (28, 40)").await;
        let between = side.provider.get_current_snapshot().unwrap();
        let error = collect(plan, ctx.task_ctx()).await.unwrap_err().to_string();
        assert!(error.contains("gained an inlined delete"), "{error}");
        assert_eq!(side.provider.get_current_snapshot().unwrap(), between);
        let values = crate_values(&side.provider, &[between]).await;
        let kept: Vec<i64> = (10..50).filter(|id| *id != 28 && *id != 40).collect();
        assert_eq!(ids(&values[0]), kept);
    }

    /// `MetadataWriter::commit_deletes` with a delete file
    /// `DuckLakeTableWriter::write_delete_file` wrote: it commits beside a
    /// concurrent inlined deletion of other rows of the file, and conflicts
    /// with one of a row the file deletes (`overlap`) or when the file does
    /// not record its positions (`positions` false).
    pub(crate) async fn direct_delete_beside_concurrent_inlined_deletion(
        side: &CrateSide,
        overlap: bool,
        positions: bool,
    ) {
        conflict_setup(side, ConflictSetup::Plain(LIMIT)).await;
        let base = side.provider.get_current_snapshot().unwrap();
        let schema = side
            .provider
            .get_schema_by_name("main", base)
            .unwrap()
            .unwrap();
        let table = side
            .provider
            .get_table_by_name(schema.schema_id, "t", base)
            .unwrap()
            .unwrap();
        let file = side
            .provider
            .get_table_files_for_select(table.table_id, base)
            .unwrap()
            .into_iter()
            .min_by_key(|file| file.data_file_id)
            .unwrap();
        run_crate_sql(
            side,
            if overlap {
                "DELETE FROM lake.main.t WHERE id IN (22, 40)"
            } else {
                "DELETE FROM lake.main.t WHERE id IN (28, 40)"
            },
        )
        .await;
        let between = side.provider.get_current_snapshot().unwrap();
        // Ids 20..=26 sit at positions 10..=16 of the file of ids 10..29.
        let written =
            DuckLakeTableWriter::new(Arc::clone(&side.writer), Arc::new(LocalFileSystem::new()))
                .unwrap()
                .write_delete_file("main", "t", &file.file.path, &(10..17).collect::<Vec<_>>())
                .await
                .unwrap();
        let name = written.path.clone();
        let delete = if positions {
            written
        } else {
            datafusion_ducklake::DeleteFileInfo::new(
                written.path.clone(),
                written.file_size_bytes,
                written.delete_count,
            )
        };
        let result = side.writer.commit_deletes(
            table.table_id,
            "main",
            "t",
            base,
            &[datafusion_ducklake::DeleteFileEntry {
                data_file_id: file.data_file_id,
                expected_prev_delete_file: None,
                delete,
            }],
            &[],
        );
        let head = side.provider.get_current_snapshot().unwrap();
        if overlap || !positions {
            let error = result.unwrap_err().to_string();
            assert!(error.contains("gained an inlined delete"), "{error}");
            assert_eq!(head, between);
            // The caller owns the object of a delete file its commit refused.
            let mut pending = vec![PathBuf::from(side.writer.get_data_path().unwrap())];
            while let Some(dir) = pending.pop() {
                for entry in std::fs::read_dir(dir).unwrap() {
                    let path = entry.unwrap().path();
                    if path.is_dir() {
                        pending.push(path);
                    } else if path.file_name().is_some_and(|file| file == name.as_str()) {
                        std::fs::remove_file(path).unwrap();
                    }
                }
            }
            return;
        }
        result.unwrap();
        assert!(head > between);
        let values = crate_values(&side.provider, &[head]).await;
        let kept: Vec<i64> = (10..50)
            .filter(|id| *id != 28 && *id != 40 && !(20..=26).contains(id))
            .collect();
        assert_eq!(ids(&values[0]), kept);
    }

    /// An `UPDATE` of a Parquet file another transaction recorded an inlined
    /// deletion of since the statement read it conflicts, as official's insert
    /// check refuses it: committing both would bring back the deleted row as
    /// a new row version.
    pub(crate) async fn update_after_concurrent_inlined_deletion(side: &CrateSide) {
        conflict_setup(side, ConflictSetup::Plain(LIMIT)).await;
        let (ctx, plan) = plan(side, UPDATE_RANGE).await;
        run_crate_sql(side, "DELETE FROM lake.main.t WHERE id IN (22, 40)").await;
        let head = side.provider.get_current_snapshot().unwrap();
        let before = crate_values(&side.provider, &[head]).await;
        let error = collect(plan, ctx.task_ctx()).await.unwrap_err().to_string();
        assert!(
            error.contains("Transaction conflict - attempting to insert into table")
                && error.contains("but another transaction has deleted inlined data from it"),
            "{error}"
        );
        assert_eq!(side.provider.get_current_snapshot().unwrap(), head);
        assert_eq!(crate_values(&side.provider, &[head]).await, before);
    }

    const DELETE_RANGE: &str = "DELETE FROM lake.main.t WHERE id BETWEEN 20 AND 26";
    const UPDATE_RANGE: &str = "UPDATE lake.main.t SET v = 'u' WHERE id BETWEEN 20 AND 26";

    /// One scenario of this module, run on a fresh catalog.
    pub(crate) type RedateScenario =
        Box<dyn for<'a> Fn(&'a CrateSide) -> futures::future::BoxFuture<'a, ()> + Send + Sync>;

    /// Every scenario of this module.
    pub(crate) fn scenarios() -> Vec<RedateScenario> {
        let mut scenarios: Vec<RedateScenario> = vec![
            Box::new(|side| Box::pin(interleaved_commit_to_another_table(side))),
            Box::new(|side| Box::pin(concurrent_rewrites_of_one_delete_file(side))),
            Box::new(|side| Box::pin(concurrent_deletes_of_different_files(side, false))),
            Box::new(|side| Box::pin(concurrent_deletes_of_different_files(side, true))),
            Box::new(|side| Box::pin(update_after_concurrent_inlined_deletion(side))),
            Box::new(|side| Box::pin(drop_after_concurrent_inlined_deletion(side))),
            Box::new(|side| Box::pin(inlined_deletion_after_concurrent_delete_file(side, false))),
            Box::new(|side| Box::pin(inlined_deletion_after_concurrent_delete_file(side, true))),
        ];
        for (overlap, positions) in [(false, true), (true, true), (false, false)] {
            scenarios.push(Box::new(move |side| {
                Box::pin(direct_delete_beside_concurrent_inlined_deletion(
                    side, overlap, positions,
                ))
            }));
        }
        for delete_file in [false, true] {
            for overlap in [false, true] {
                scenarios.push(Box::new(move |side| {
                    Box::pin(delete_beside_concurrent_inlined_deletion(
                        side,
                        delete_file,
                        overlap,
                    ))
                }));
            }
        }
        for (statement, change, conflicts) in SAME_TABLE_CASES {
            scenarios.push(Box::new(move |side| {
                Box::pin(conflicting_change_to_the_same_table(
                    side, statement, change, conflicts,
                ))
            }));
        }
        for (setup_kind, statement, change) in DIRECT_CASES {
            scenarios.push(Box::new(move |side| {
                Box::pin(conflicting_change_after(
                    side, setup_kind, statement, change, true,
                ))
            }));
        }
        for (file_deletion, change, conflict) in FLUSH_CASES {
            scenarios.push(Box::new(move |side| {
                Box::pin(flush_after_concurrent_change(
                    side,
                    file_deletion,
                    change,
                    conflict,
                ))
            }));
        }
        scenarios
    }

    /// A local store that runs `hook` before every object written through it
    /// while armed: each attempt of a statement's commit then loses to a commit
    /// the hook makes.
    struct EveryPutStore {
        inner: object_store::local::LocalFileSystem,
        armed: std::sync::atomic::AtomicBool,
        puts: std::sync::atomic::AtomicUsize,
        hook: Box<dyn Fn() -> futures::future::BoxFuture<'static, ()> + Send + Sync>,
    }

    impl std::fmt::Debug for EveryPutStore {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "EveryPutStore")
        }
    }

    impl std::fmt::Display for EveryPutStore {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "EveryPutStore({})", self.inner)
        }
    }

    impl EveryPutStore {
        async fn before_put(&self) {
            if self.armed.load(std::sync::atomic::Ordering::SeqCst) {
                self.puts.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                (self.hook)().await;
            }
        }
    }

    #[async_trait::async_trait]
    impl object_store::ObjectStore for EveryPutStore {
        async fn put_opts(
            &self,
            location: &object_store::path::Path,
            payload: object_store::PutPayload,
            opts: object_store::PutOptions,
        ) -> object_store::Result<object_store::PutResult> {
            self.before_put().await;
            self.inner.put_opts(location, payload, opts).await
        }

        async fn put_multipart_opts(
            &self,
            location: &object_store::path::Path,
            opts: object_store::PutMultipartOptions,
        ) -> object_store::Result<Box<dyn object_store::MultipartUpload>> {
            self.before_put().await;
            self.inner.put_multipart_opts(location, opts).await
        }

        async fn get_opts(
            &self,
            location: &object_store::path::Path,
            options: object_store::GetOptions,
        ) -> object_store::Result<object_store::GetResult> {
            self.inner.get_opts(location, options).await
        }

        async fn get_ranges(
            &self,
            location: &object_store::path::Path,
            ranges: &[std::ops::Range<u64>],
        ) -> object_store::Result<Vec<bytes::Bytes>> {
            self.inner.get_ranges(location, ranges).await
        }

        fn delete_stream(
            &self,
            locations: futures::stream::BoxStream<
                'static,
                object_store::Result<object_store::path::Path>,
            >,
        ) -> futures::stream::BoxStream<'static, object_store::Result<object_store::path::Path>>
        {
            self.inner.delete_stream(locations)
        }

        fn list(
            &self,
            prefix: Option<&object_store::path::Path>,
        ) -> futures::stream::BoxStream<'static, object_store::Result<object_store::ObjectMeta>>
        {
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

    /// A statement whose every commit attempt loses to a commit to another
    /// table re-dates and retries for official's 10 attempts, then fails with
    /// official's error, leaving no object behind.
    pub(crate) async fn retries_stop_after_ten_attempts(side: &CrateSide) {
        setup(side).await;
        let other = CrateSide {
            writer: Arc::clone(&side.writer),
            provider: Arc::clone(&side.provider),
        };
        let other = Arc::new(other);
        let store = Arc::new(EveryPutStore {
            inner: LocalFileSystem::new(),
            armed: std::sync::atomic::AtomicBool::new(true),
            puts: std::sync::atomic::AtomicUsize::new(0),
            hook: Box::new(move || {
                let other = Arc::clone(&other);
                Box::pin(async move {
                    DuckLakeTableWriter::new(
                        Arc::clone(&other.writer),
                        Arc::new(LocalFileSystem::new()),
                    )
                    .unwrap()
                    .with_options(
                        &DuckLakeWriteOptions::default().with_data_inlining_row_limit(LIMIT),
                    )
                    .append_table("main", "u", &[id_value([1], |_| "u".to_string())])
                    .await
                    .unwrap();
                })
            }),
        });
        let ctx = SessionContext::new();
        ctx.register_object_store(
            &url::Url::parse("file:///").unwrap(),
            Arc::clone(&store) as _,
        );
        let catalog =
            DuckLakeCatalog::with_writer(Arc::clone(&side.provider), Arc::clone(&side.writer))
                .unwrap();
        DuckLakeCatalog::register(&ctx, "lake", catalog);
        let error = ctx
            .sql("DELETE FROM lake.main.t WHERE id BETWEEN 20 AND 26")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap_err()
            .to_string();
        store
            .armed
            .store(false, std::sync::atomic::Ordering::SeqCst);
        assert!(
            error.contains(
                "Failed to commit DuckLake transaction.\nExceeded the maximum retry count of 10 \
                 set by the ducklake_max_retry_count setting.\n. Consider increasing the value \
                 with: e.g., \"SET ducklake_max_retry_count = 100;\"\n"
            ),
            "{error}"
        );
        assert_eq!(
            store.puts.load(std::sync::atomic::Ordering::SeqCst),
            10,
            "one delete file per attempt"
        );
        assert_eq!(live_delete_files(side), 1);
    }

    /// A local store that, at the first object written through it, runs
    /// `hook` before the write: a statement whose write is held there has read
    /// the catalog and not yet committed.
    struct HookStore {
        inner: object_store::local::LocalFileSystem,
        hook: tokio::sync::Mutex<Option<futures::future::BoxFuture<'static, ()>>>,
    }

    impl std::fmt::Debug for HookStore {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "HookStore")
        }
    }

    impl std::fmt::Display for HookStore {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "HookStore({})", self.inner)
        }
    }

    #[async_trait::async_trait]
    impl object_store::ObjectStore for HookStore {
        async fn put_opts(
            &self,
            location: &object_store::path::Path,
            payload: object_store::PutPayload,
            opts: object_store::PutOptions,
        ) -> object_store::Result<object_store::PutResult> {
            let hook = self.hook.lock().await.take();
            if let Some(hook) = hook {
                hook.await;
            }
            self.inner.put_opts(location, payload, opts).await
        }

        async fn put_multipart_opts(
            &self,
            location: &object_store::path::Path,
            opts: object_store::PutMultipartOptions,
        ) -> object_store::Result<Box<dyn object_store::MultipartUpload>> {
            let hook = self.hook.lock().await.take();
            if let Some(hook) = hook {
                hook.await;
            }
            self.inner.put_multipart_opts(location, opts).await
        }

        async fn get_opts(
            &self,
            location: &object_store::path::Path,
            options: object_store::GetOptions,
        ) -> object_store::Result<object_store::GetResult> {
            self.inner.get_opts(location, options).await
        }

        async fn get_ranges(
            &self,
            location: &object_store::path::Path,
            ranges: &[std::ops::Range<u64>],
        ) -> object_store::Result<Vec<bytes::Bytes>> {
            self.inner.get_ranges(location, ranges).await
        }

        fn delete_stream(
            &self,
            locations: futures::stream::BoxStream<
                'static,
                object_store::Result<object_store::path::Path>,
            >,
        ) -> futures::stream::BoxStream<'static, object_store::Result<object_store::path::Path>>
        {
            self.inner.delete_stream(locations)
        }

        fn list(
            &self,
            prefix: Option<&object_store::path::Path>,
        ) -> futures::stream::BoxStream<'static, object_store::Result<object_store::ObjectMeta>>
        {
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

    /// Two statements rewriting the same delete file: the second reads it,
    /// the first commits its own rewrite, then the second commits. The second
    /// conflicts, and one live delete file holds the first's deletions with
    /// every earlier one.
    pub(crate) async fn concurrent_rewrites_of_one_delete_file(side: &CrateSide) {
        setup(side).await;
        let first = CrateSide {
            writer: Arc::clone(&side.writer),
            provider: Arc::clone(&side.provider),
        };
        let store = Arc::new(HookStore {
            inner: LocalFileSystem::new(),
            hook: tokio::sync::Mutex::new(Some(Box::pin(async move {
                run_crate_sql(&first, "DELETE FROM lake.main.t WHERE id BETWEEN 20 AND 26").await;
            }))),
        });
        let ctx = SessionContext::new();
        ctx.register_object_store(&url::Url::parse("file:///").unwrap(), store);
        let catalog =
            DuckLakeCatalog::with_writer(Arc::clone(&side.provider), Arc::clone(&side.writer))
                .unwrap();
        DuckLakeCatalog::register(&ctx, "lake", catalog);
        let before = side.provider.get_current_snapshot().unwrap();
        let error = ctx
            .sql("DELETE FROM lake.main.t WHERE id BETWEEN 17 AND 19 OR id BETWEEN 27 AND 29")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap_err();
        assert!(error.to_string().contains("conflict"), "{error}");
        let head = side.provider.get_current_snapshot().unwrap();
        assert_eq!(head, before + 1, "only the first statement committed");
        assert_eq!(live_delete_files(side), 1);
        let values = crate_values(&side.provider, &[head]).await;
        let mut kept = vec![1, 2, 3, 17, 18, 19];
        kept.extend(27..30);
        assert_eq!(ids(&values[0]), kept);
    }

    /// Every Parquet object under `data` is a data or delete file the catalog
    /// lists or has scheduled for deletion: a refused attempt left none.
    pub(crate) fn no_orphaned_objects(rows: RawDumpRows, data: &Path) {
        let count = |sql: &str| (rows.0)(sql)[0].parse::<usize>().unwrap();
        let listed = count("SELECT count(*) FROM m.ducklake_data_file")
            + count("SELECT count(*) FROM m.ducklake_delete_file")
            + count("SELECT count(*) FROM m.ducklake_files_scheduled_for_deletion");
        let mut objects = 0;
        let mut pending = vec![data.to_path_buf()];
        while let Some(dir) = pending.pop() {
            for entry in std::fs::read_dir(dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    pending.push(path);
                } else if path.extension().is_some_and(|ext| ext == "parquet") {
                    objects += 1;
                }
            }
        }
        assert_eq!(objects, listed);
    }

    #[cfg(feature = "write-sqlite")]
    mod sqlite {
        use super::*;
        use datafusion_ducklake::{SqliteMetadataProvider, SqliteMetadataWriter};

        async fn run(
            scenario: impl for<'a> Fn(&'a CrateSide) -> futures::future::BoxFuture<'a, ()>,
        ) {
            let temp = tempfile::TempDir::new().unwrap();
            let path = temp.path().join("c.db");
            let url = format!("sqlite:{}?mode=rwc", path.display());
            let data = temp.path().join("data");
            std::fs::create_dir_all(&data).unwrap();
            {
                let writer = SqliteMetadataWriter::new_with_init(&url).await.unwrap();
                writer.set_data_path(data.to_str().unwrap()).unwrap();
                let side = CrateSide {
                    writer: Arc::new(writer),
                    provider: Arc::new(SqliteMetadataProvider::new(&url).await.unwrap()),
                };
                scenario(&side).await;
            }
            no_orphaned_objects(duckdb_attached_rows(Metadata::Sqlite(path)), &data);
        }

        #[tokio::test(flavor = "multi_thread")]
        async fn interleaved_commit_to_another_table() {
            run(|side| Box::pin(super::interleaved_commit_to_another_table(side))).await;
        }

        #[tokio::test(flavor = "multi_thread")]
        async fn concurrent_rewrites_of_one_delete_file() {
            run(|side| Box::pin(super::concurrent_rewrites_of_one_delete_file(side))).await;
        }

        #[tokio::test(flavor = "multi_thread")]
        async fn retries_stop_after_ten_attempts() {
            run(|side| Box::pin(super::retries_stop_after_ten_attempts(side))).await;
        }

        #[tokio::test(flavor = "multi_thread")]
        async fn concurrent_deletes_of_different_files() {
            for delete_files in [false, true] {
                run(move |side| {
                    Box::pin(super::concurrent_deletes_of_different_files(
                        side,
                        delete_files,
                    ))
                })
                .await;
            }
        }

        #[tokio::test(flavor = "multi_thread")]
        async fn delete_beside_concurrent_inlined_deletion() {
            for delete_file in [false, true] {
                for overlap in [false, true] {
                    run(move |side| {
                        Box::pin(super::delete_beside_concurrent_inlined_deletion(
                            side,
                            delete_file,
                            overlap,
                        ))
                    })
                    .await;
                }
            }
        }

        #[tokio::test(flavor = "multi_thread")]
        async fn update_after_concurrent_inlined_deletion() {
            run(|side| Box::pin(super::update_after_concurrent_inlined_deletion(side))).await;
        }

        #[tokio::test(flavor = "multi_thread")]
        async fn inlined_deletion_after_concurrent_delete_file() {
            for overlap in [false, true] {
                run(move |side| {
                    Box::pin(super::inlined_deletion_after_concurrent_delete_file(
                        side, overlap,
                    ))
                })
                .await;
            }
        }

        #[tokio::test(flavor = "multi_thread")]
        async fn drop_after_concurrent_inlined_deletion() {
            run(|side| Box::pin(super::drop_after_concurrent_inlined_deletion(side))).await;
        }

        #[tokio::test(flavor = "multi_thread")]
        async fn direct_delete_beside_concurrent_inlined_deletion() {
            for (overlap, positions) in [(false, true), (true, true), (false, false)] {
                run(move |side| {
                    Box::pin(super::direct_delete_beside_concurrent_inlined_deletion(
                        side, overlap, positions,
                    ))
                })
                .await;
            }
        }

        #[tokio::test(flavor = "multi_thread")]
        async fn same_table_changes_conflict() {
            for (statement, change, conflicts) in super::SAME_TABLE_CASES {
                run(move |side| {
                    Box::pin(super::conflicting_change_to_the_same_table(
                        side, statement, change, conflicts,
                    ))
                })
                .await;
            }
        }

        /// A `DELETE` or `UPDATE` of a table another transaction dropped since
        /// the statement read it conflicts, an `UPDATE` reporting its insert
        /// first, as official checks it.
        #[tokio::test(flavor = "multi_thread")]
        async fn dropped_table_conflicts() {
            use datafusion::physical_plan::collect;
            let cases = [
                (ConflictSetup::Plain(0), DELETE_RANGE, "delete from table"),
                (ConflictSetup::Plain(0), UPDATE_RANGE, "insert into table"),
                (
                    ConflictSetup::WithDeleteFile(0),
                    DELETE_RANGE,
                    "delete from table",
                ),
                (
                    ConflictSetup::Inlined,
                    "DELETE FROM lake.main.t WHERE id = 2",
                    "delete from table",
                ),
                (
                    ConflictSetup::Inlined,
                    "UPDATE lake.main.t SET v = 'u' WHERE id = 20",
                    "insert into table",
                ),
            ];
            for (setup_kind, statement, action) in cases {
                let temp = tempfile::TempDir::new().unwrap();
                let url = format!("sqlite:{}?mode=rwc", temp.path().join("c.db").display());
                let data = temp.path().join("data");
                std::fs::create_dir_all(&data).unwrap();
                let writer = Arc::new(SqliteMetadataWriter::new_with_init(&url).await.unwrap());
                writer.set_data_path(data.to_str().unwrap()).unwrap();
                let side = CrateSide {
                    writer: Arc::clone(&writer) as Arc<dyn MetadataWriter>,
                    provider: Arc::new(SqliteMetadataProvider::new(&url).await.unwrap()),
                };
                super::conflict_setup(&side, setup_kind).await;
                let (ctx, plan) = super::plan(&side, statement).await;
                assert!(writer.drop_table("main", "t").unwrap());
                let head = side.provider.get_current_snapshot().unwrap();
                let error = collect(plan, ctx.task_ctx())
                    .await
                    .map(|_| ())
                    .unwrap_err()
                    .to_string();
                assert!(
                    error.contains(&format!("Transaction conflict - attempting to {action}"))
                        && error.contains("but another transaction has dropped it"),
                    "{statement} from {setup_kind:?}: {error}"
                );
                assert_eq!(side.provider.get_current_snapshot().unwrap(), head);
            }
        }

        #[tokio::test(flavor = "multi_thread")]
        async fn same_table_changes_conflict_without_staging() {
            for (setup_kind, statement, change) in super::DIRECT_CASES {
                run(move |side| {
                    Box::pin(super::conflicting_change_after(
                        side, setup_kind, statement, change, true,
                    ))
                })
                .await;
            }
        }

        #[tokio::test(flavor = "multi_thread")]
        async fn flush_conflicts_with_concurrent_changes() {
            for (file_deletion, change, conflict) in super::FLUSH_CASES {
                run(move |side| {
                    Box::pin(super::flush_after_concurrent_change(
                        side,
                        file_deletion,
                        change,
                        conflict,
                    ))
                })
                .await;
            }
        }
    }

    #[cfg(feature = "write-duckdb")]
    mod duckdb_backend {
        use super::*;
        use datafusion_ducklake::DuckdbMetadataWriter;

        async fn run(
            scenario: impl for<'a> Fn(&'a CrateSide) -> futures::future::BoxFuture<'a, ()>,
        ) {
            let temp = tempfile::TempDir::new().unwrap();
            let path = temp.path().join("c.duckdb");
            let data = temp.path().join("data");
            std::fs::create_dir_all(&data).unwrap();
            {
                let writer =
                    DuckdbMetadataWriter::new_with_init(path.display().to_string()).unwrap();
                writer.set_data_path(data.to_str().unwrap()).unwrap();
                let provider = writer.metadata_provider();
                let side = CrateSide {
                    writer: Arc::new(writer),
                    provider: Arc::new(provider),
                };
                scenario(&side).await;
            }
            no_orphaned_objects(duckdb_attached_rows(Metadata::Duckdb(path)), &data);
        }

        #[tokio::test(flavor = "multi_thread")]
        async fn interleaved_commit_to_another_table() {
            run(|side| Box::pin(super::interleaved_commit_to_another_table(side))).await;
        }

        #[tokio::test(flavor = "multi_thread")]
        async fn concurrent_rewrites_of_one_delete_file() {
            run(|side| Box::pin(super::concurrent_rewrites_of_one_delete_file(side))).await;
        }

        #[tokio::test(flavor = "multi_thread")]
        async fn retries_stop_after_ten_attempts() {
            run(|side| Box::pin(super::retries_stop_after_ten_attempts(side))).await;
        }

        #[tokio::test(flavor = "multi_thread")]
        async fn concurrent_deletes_of_different_files() {
            for delete_files in [false, true] {
                run(move |side| {
                    Box::pin(super::concurrent_deletes_of_different_files(
                        side,
                        delete_files,
                    ))
                })
                .await;
            }
        }

        #[tokio::test(flavor = "multi_thread")]
        async fn delete_beside_concurrent_inlined_deletion() {
            for delete_file in [false, true] {
                for overlap in [false, true] {
                    run(move |side| {
                        Box::pin(super::delete_beside_concurrent_inlined_deletion(
                            side,
                            delete_file,
                            overlap,
                        ))
                    })
                    .await;
                }
            }
        }

        #[tokio::test(flavor = "multi_thread")]
        async fn update_after_concurrent_inlined_deletion() {
            run(|side| Box::pin(super::update_after_concurrent_inlined_deletion(side))).await;
        }

        #[tokio::test(flavor = "multi_thread")]
        async fn inlined_deletion_after_concurrent_delete_file() {
            for overlap in [false, true] {
                run(move |side| {
                    Box::pin(super::inlined_deletion_after_concurrent_delete_file(
                        side, overlap,
                    ))
                })
                .await;
            }
        }

        #[tokio::test(flavor = "multi_thread")]
        async fn drop_after_concurrent_inlined_deletion() {
            run(|side| Box::pin(super::drop_after_concurrent_inlined_deletion(side))).await;
        }

        #[tokio::test(flavor = "multi_thread")]
        async fn direct_delete_beside_concurrent_inlined_deletion() {
            for (overlap, positions) in [(false, true), (true, true), (false, false)] {
                run(move |side| {
                    Box::pin(super::direct_delete_beside_concurrent_inlined_deletion(
                        side, overlap, positions,
                    ))
                })
                .await;
            }
        }

        #[tokio::test(flavor = "multi_thread")]
        async fn same_table_changes_conflict() {
            for (statement, change, conflicts) in super::SAME_TABLE_CASES {
                run(move |side| {
                    Box::pin(super::conflicting_change_to_the_same_table(
                        side, statement, change, conflicts,
                    ))
                })
                .await;
            }
        }

        #[tokio::test(flavor = "multi_thread")]
        async fn same_table_changes_conflict_without_staging() {
            for (setup_kind, statement, change) in super::DIRECT_CASES {
                run(move |side| {
                    Box::pin(super::conflicting_change_after(
                        side, setup_kind, statement, change, true,
                    ))
                })
                .await;
            }
        }

        #[tokio::test(flavor = "multi_thread")]
        async fn flush_conflicts_with_concurrent_changes() {
            for (file_deletion, change, conflict) in super::FLUSH_CASES {
                run(move |side| {
                    Box::pin(super::flush_after_concurrent_change(
                        side,
                        file_deletion,
                        change,
                        conflict,
                    ))
                })
                .await;
            }
        }
    }

    #[cfg(feature = "write-mysql")]
    mod mysql_backend {
        use super::*;
        use datafusion_ducklake::{MySqlMetadataProvider, MySqlMetadataWriter};
        use testcontainers::runners::AsyncRunner;
        use testcontainers_modules::mysql::Mysql;

        #[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
        #[tokio::test(flavor = "multi_thread")]
        async fn scenarios() {
            let container = Mysql::default().start().await.unwrap();
            let port = container.get_host_port_ipv4(3306).await.unwrap();
            let admin = format!("mysql://root@127.0.0.1:{port}/test");
            let pool = sqlx::MySqlPool::connect(&admin).await.unwrap();
            let scenarios = super::scenarios();
            for (index, scenario) in scenarios.into_iter().enumerate() {
                let database = format!("redate_{index}");
                sqlx::query(sqlx::AssertSqlSafe(format!("CREATE DATABASE {database}")))
                    .execute(&pool)
                    .await
                    .unwrap();
                let url = format!("mysql://root@127.0.0.1:{port}/{database}");
                let temp = tempfile::TempDir::new().unwrap();
                let data = temp.path().join("data");
                std::fs::create_dir_all(&data).unwrap();
                {
                    let writer = MySqlMetadataWriter::new_with_init(&url).await.unwrap();
                    writer.set_data_path(data.to_str().unwrap()).unwrap();
                    let side = CrateSide {
                        writer: Arc::new(writer),
                        provider: Arc::new(MySqlMetadataProvider::new(&url).await.unwrap()),
                    };
                    scenario(&side).await;
                }
                no_orphaned_objects(
                    duckdb_attached_rows(Metadata::Mysql {
                        host: "127.0.0.1".to_string(),
                        port,
                        database,
                    }),
                    &data,
                );
            }
        }
    }

    #[cfg(feature = "multicatalog-postgres")]
    mod multicatalog_backend {
        use super::*;
        use datafusion_ducklake::{
            MulticatalogManager, MulticatalogProvider, PostgresMetadataWriter,
        };
        use testcontainers::runners::AsyncRunner;
        use testcontainers_modules::postgres::Postgres;

        #[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
        #[tokio::test(flavor = "multi_thread")]
        async fn scenarios() {
            let container = Postgres::default().start().await.unwrap();
            let port = container.get_host_port_ipv4(5432).await.unwrap();
            let admin = format!("postgresql://postgres:postgres@127.0.0.1:{port}/postgres");
            let admin_pool = sqlx::PgPool::connect(&admin).await.unwrap();
            let scenarios = super::scenarios();
            for (index, scenario) in scenarios.into_iter().enumerate() {
                let database = format!("redate_{index}");
                sqlx::query(sqlx::AssertSqlSafe(format!("CREATE DATABASE {database}")))
                    .execute(&admin_pool)
                    .await
                    .unwrap();
                let url = format!("postgresql://postgres:postgres@127.0.0.1:{port}/{database}");
                let temp = tempfile::TempDir::new().unwrap();
                let data = temp.path().join("data");
                std::fs::create_dir_all(&data).unwrap();
                {
                    let pool = sqlx::PgPool::connect(&url).await.unwrap();
                    datafusion_ducklake::initialize_multicatalog_schema(&pool)
                        .await
                        .unwrap();
                    let manager = MulticatalogManager::new(pool.clone());
                    // Another catalog's snapshots take ids from the same
                    // sequence as this one's.
                    manager.create_catalog("other").await.unwrap();
                    let catalog = manager.create_catalog("lake").await.unwrap();
                    let writer = PostgresMetadataWriter::with_pool(pool.clone(), catalog)
                        .await
                        .unwrap();
                    writer.set_data_path(data.to_str().unwrap()).unwrap();
                    let side = CrateSide {
                        writer: Arc::new(writer),
                        provider: Arc::new(
                            MulticatalogProvider::with_pool(pool, "lake").await.unwrap(),
                        ),
                    };
                    scenario(&side).await;
                }
                no_orphaned_objects(
                    duckdb_attached_rows(Metadata::Postgres(format!(
                        "host=127.0.0.1 port={port} user=postgres password=postgres \
                         dbname={database}"
                    ))),
                    &data,
                );
            }
        }

        /// Another catalog of the store commits while a `DELETE` rewriting a
        /// delete file runs: the `DELETE` dates its deletions to the id its
        /// snapshot takes, so time travel and the change feed of that
        /// snapshot hold them.
        #[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
        #[tokio::test(flavor = "multi_thread")]
        async fn rewritten_deletions_take_the_snapshot_id_another_catalog_left() {
            let container = Postgres::default().start().await.unwrap();
            let port = container.get_host_port_ipv4(5432).await.unwrap();
            let url = format!("postgresql://postgres:postgres@127.0.0.1:{port}/postgres");
            let temp = tempfile::TempDir::new().unwrap();
            let data = temp.path().join("data");
            std::fs::create_dir_all(&data).unwrap();
            let pool = sqlx::PgPool::connect(&url).await.unwrap();
            datafusion_ducklake::initialize_multicatalog_schema(&pool)
                .await
                .unwrap();
            let manager = MulticatalogManager::new(pool.clone());
            let other_catalog = manager.create_catalog("other").await.unwrap();
            let catalog = manager.create_catalog("lake").await.unwrap();
            let side_for = async |catalog, name| {
                let writer = PostgresMetadataWriter::with_pool(pool.clone(), catalog)
                    .await
                    .unwrap();
                writer.set_data_path(data.to_str().unwrap()).unwrap();
                CrateSide {
                    writer: Arc::new(writer),
                    provider: Arc::new(
                        MulticatalogProvider::with_pool(pool.clone(), name)
                            .await
                            .unwrap(),
                    ),
                }
            };
            let side = side_for(catalog, "lake").await;
            let other = side_for(other_catalog, "other").await;
            super::conflict_setup(&side, ConflictSetup::WithDeleteFile(LIMIT)).await;
            let (ctx, plan) = super::plan(&side, DELETE_RANGE).await;
            DuckLakeTableWriter::new(Arc::clone(&other.writer), Arc::new(LocalFileSystem::new()))
                .unwrap()
                .append_table("main", "u", &[id_value([1], |_| "u".to_string())])
                .await
                .unwrap();
            collect(plan, ctx.task_ctx()).await.unwrap();
            let head = side.provider.get_current_snapshot().unwrap();
            let expected: Vec<String> = (20..=26).map(|id| format!("{head}:{id}")).collect();
            assert_eq!(
                super::feed(&side, "ducklake_table_deletions", head, head).await,
                expected
            );
            let previous = side
                .provider
                .list_snapshots()
                .unwrap()
                .iter()
                .map(|snapshot| snapshot.snapshot_id)
                .filter(|id| *id < head)
                .max()
                .unwrap();
            let values = crate_values(&side.provider, &[previous, head]).await;
            assert!(ids(&values[0]).contains(&20));
            assert!(!ids(&values[1]).contains(&20));
        }

        /// A `DELETE` rewriting a delete file after another catalog of the
        /// store took snapshot ids dates its deletions to the id its snapshot
        /// takes on the first attempt: it writes one delete file and commits
        /// without a retry.
        #[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
        #[tokio::test(flavor = "multi_thread")]
        async fn rewritten_deletions_take_the_next_snapshot_id_first() {
            let container = Postgres::default().start().await.unwrap();
            let port = container.get_host_port_ipv4(5432).await.unwrap();
            let url = format!("postgresql://postgres:postgres@127.0.0.1:{port}/postgres");
            let temp = tempfile::TempDir::new().unwrap();
            let data = temp.path().join("data");
            std::fs::create_dir_all(&data).unwrap();
            let pool = sqlx::PgPool::connect(&url).await.unwrap();
            datafusion_ducklake::initialize_multicatalog_schema(&pool)
                .await
                .unwrap();
            let manager = MulticatalogManager::new(pool.clone());
            let other_catalog = manager.create_catalog("other").await.unwrap();
            let catalog = manager.create_catalog("lake").await.unwrap();
            let side_for = async |catalog, name| {
                let writer = PostgresMetadataWriter::with_pool(pool.clone(), catalog)
                    .await
                    .unwrap();
                writer.set_data_path(data.to_str().unwrap()).unwrap();
                CrateSide {
                    writer: Arc::new(writer),
                    provider: Arc::new(
                        MulticatalogProvider::with_pool(pool.clone(), name)
                            .await
                            .unwrap(),
                    ),
                }
            };
            let side = side_for(catalog, "lake").await;
            let other = side_for(other_catalog, "other").await;
            super::conflict_setup(&side, ConflictSetup::WithDeleteFile(LIMIT)).await;
            for id in 0..3 {
                DuckLakeTableWriter::new(
                    Arc::clone(&other.writer),
                    Arc::new(LocalFileSystem::new()),
                )
                .unwrap()
                .append_table("main", "u", &[id_value([id], |_| "u".to_string())])
                .await
                .unwrap();
            }
            let store = Arc::new(super::EveryPutStore {
                inner: LocalFileSystem::new(),
                armed: std::sync::atomic::AtomicBool::new(true),
                puts: std::sync::atomic::AtomicUsize::new(0),
                hook: Box::new(|| Box::pin(async {})),
            });
            let ctx = SessionContext::new();
            ctx.register_object_store(
                &url::Url::parse("file:///").unwrap(),
                Arc::clone(&store) as _,
            );
            let lake =
                DuckLakeCatalog::with_writer(Arc::clone(&side.provider), Arc::clone(&side.writer))
                    .unwrap();
            DuckLakeCatalog::register(&ctx, "lake", lake);
            ctx.sql(DELETE_RANGE)
                .await
                .unwrap()
                .collect()
                .await
                .unwrap();
            assert_eq!(
                store.puts.load(std::sync::atomic::Ordering::SeqCst),
                1,
                "one delete file, written once"
            );
            let head = side.provider.get_current_snapshot().unwrap();
            let expected: Vec<String> = (20..=26).map(|id| format!("{head}:{id}")).collect();
            assert_eq!(
                super::feed(&side, "ducklake_table_deletions", head, head).await,
                expected
            );
        }
    }
}

#[cfg(feature = "write-duckdb")]
mod duckdb_backend {
    use super::*;
    use datafusion_ducklake::DuckdbMetadataWriter;

    async fn run(scenario: Scenario) {
        run_in(scenario).await;
    }

    /// Run `scenario`, returning its scratch directory: the crate's catalog is
    /// `crate.duckdb` over `crate_data`.
    async fn run_in(scenario: Scenario) -> (Option<tempfile::TempDir>, PathBuf) {
        let (temp, dir) = scratch_dir();
        let crate_path = dir.join("crate.duckdb");
        let data = dir.join("crate_data");
        std::fs::create_dir_all(&data).unwrap();
        let path = crate_path.clone();
        let checked = run_scenario(
            &scenario,
            &Metadata::Duckdb(dir.join("official.ducklake")),
            &dir,
            || async move {
                let writer =
                    DuckdbMetadataWriter::new_with_init(path.display().to_string()).unwrap();
                writer.set_data_path(data.to_str().unwrap()).unwrap();
                let provider = writer.metadata_provider();
                CrateSide {
                    writer: Arc::new(writer),
                    provider: Arc::new(provider),
                }
            },
            || duckdb_attached_rows(Metadata::Duckdb(crate_path.clone())),
        )
        .await;
        checked.duckdb_reads(
            &Metadata::Duckdb(crate_path.clone()),
            &dir.join("crate_data"),
            scenario.name,
        );
        (temp, dir)
    }

    /// See `sqlite::duckdb_finds_inlined_update_values`.
    #[tokio::test(flavor = "multi_thread")]
    async fn duckdb_finds_inlined_update_values() {
        let (_temp, dir) = run_in(scenario_large()).await;
        let rows = duckdb_query(
            &Metadata::Duckdb(dir.join("crate.duckdb")),
            &dir.join("crate_data"),
            "SELECT count(*) FROM lake.main.t WHERE v = 'again'",
        );
        assert_eq!(rows, vec!["3".to_string()]);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn basic() {
        run(scenario_basic()).await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn large() {
        run(scenario_large()).await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn partitioned() {
        run(scenario_partitioned()).await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn schema_evolution() {
        run(scenario_schema_evolution()).await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn file_lifecycle() {
        run(scenario_file_lifecycle(LIMIT)).await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn file_lifecycle_zero_limit() {
        run(scenario_file_lifecycle(0)).await;
    }
}

#[cfg(feature = "write-mysql")]
mod mysql_backend {
    use super::*;
    use datafusion_ducklake::{MySqlMetadataProvider, MySqlMetadataWriter};
    use testcontainers::runners::AsyncRunner;
    use testcontainers_modules::mysql::Mysql;

    #[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
    #[tokio::test(flavor = "multi_thread")]
    async fn scenarios_match_official() {
        let container = Mysql::default().start().await.unwrap();
        let port = container.get_host_port_ipv4(3306).await.unwrap();
        let admin = format!("mysql://root@127.0.0.1:{port}/test");
        let pool = sqlx::MySqlPool::connect(&admin).await.unwrap();
        for (index, scenario) in scenarios().into_iter().enumerate() {
            let (_temp, dir) = scratch_dir();
            let crate_db = format!("crate_{index}");
            sqlx::query(sqlx::AssertSqlSafe(format!("CREATE DATABASE {crate_db}")))
                .execute(&pool)
                .await
                .unwrap();
            let data = dir.join("crate_data");
            std::fs::create_dir_all(&data).unwrap();
            let url = format!("mysql://root@127.0.0.1:{port}/{crate_db}");
            let crate_metadata = Metadata::Mysql {
                host: "127.0.0.1".to_string(),
                port,
                database: crate_db.clone(),
            };
            // DuckDB's MySQL connector cannot commit DuckLake's metadata updates
            // ("only simple deletes ... are supported"), so the oracle runs on
            // SQLite metadata.
            run_scenario(
                &scenario,
                &Metadata::Sqlite(dir.join("official.db")),
                &dir,
                || async move {
                    let writer = MySqlMetadataWriter::new_with_init(&url).await.unwrap();
                    writer.set_data_path(data.to_str().unwrap()).unwrap();
                    CrateSide {
                        writer: Arc::new(writer),
                        provider: Arc::new(MySqlMetadataProvider::new(&url).await.unwrap()),
                    }
                },
                || duckdb_attached_rows(crate_metadata),
            )
            .await;
        }
    }
}

#[cfg(feature = "multicatalog-postgres")]
mod multicatalog_backend {
    use super::*;
    use datafusion_ducklake::{MulticatalogManager, MulticatalogProvider, PostgresMetadataWriter};
    use testcontainers::runners::AsyncRunner;
    use testcontainers_modules::postgres::Postgres;

    /// The multi-catalog layout has no official reader, so official runs on
    /// SQLite metadata and the crate's catalog is compared through DuckDB's
    /// PostgreSQL scanner.
    #[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
    #[tokio::test(flavor = "multi_thread")]
    async fn scenarios_match_official() {
        let container = Postgres::default().start().await.unwrap();
        let port = container.get_host_port_ipv4(5432).await.unwrap();
        for (index, scenario) in scenarios().into_iter().enumerate() {
            let admin = format!("postgresql://postgres:postgres@127.0.0.1:{port}/postgres");
            let admin_pool = sqlx::PgPool::connect(&admin).await.unwrap();
            let database = format!("lake_{index}");
            sqlx::query(sqlx::AssertSqlSafe(format!("CREATE DATABASE {database}")))
                .execute(&admin_pool)
                .await
                .unwrap();
            let url = format!("postgresql://postgres:postgres@127.0.0.1:{port}/{database}");
            let dsn = format!(
                "host=127.0.0.1 port={port} user=postgres password=postgres dbname={database}"
            );
            let (_temp, dir) = scratch_dir();
            let data = dir.join("crate_data");
            std::fs::create_dir_all(&data).unwrap();
            run_scenario(
                &scenario,
                &Metadata::Sqlite(dir.join("official.db")),
                &dir,
                || async move {
                    let pool = sqlx::PgPool::connect(&url).await.unwrap();
                    datafusion_ducklake::initialize_multicatalog_schema(&pool)
                        .await
                        .unwrap();
                    let catalog = MulticatalogManager::new(pool.clone())
                        .create_catalog("lake")
                        .await
                        .unwrap();
                    let writer = PostgresMetadataWriter::with_pool(pool.clone(), catalog)
                        .await
                        .unwrap();
                    writer.set_data_path(data.to_str().unwrap()).unwrap();
                    CrateSide {
                        writer: Arc::new(writer),
                        provider: Arc::new(
                            MulticatalogProvider::with_pool(pool, "lake").await.unwrap(),
                        ),
                    }
                },
                || duckdb_attached_rows(Metadata::Postgres(dsn)),
            )
            .await;
        }
    }
}
