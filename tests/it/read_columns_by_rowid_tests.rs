//! `DuckLakeTable::read_columns_by_rowid`: the projected columns of rows found
//! by `rowid`, wherever each row lives, at the table's snapshot.
//!
//! Two kinds of history:
//!
//! - **Official's.** DuckDB writes a table through schema evolution (a rename,
//!   a widened type, a column added with a default), data inlining, every kind
//!   of deletion, an `UPDATE`, a merge and a flush. At each snapshot the rows
//!   read by `rowid` must be the rows official itself reads with
//!   `SELECT rowid, * ... AT (VERSION => s)`, value for value.
//! - **This crate's.** The same kinds of rows, written by this crate on SQLite
//!   and on a multi-catalog PostgreSQL store, checked against the crate's own
//!   `rowid` scan, for order independence, missing ids and projections.

#![cfg(all(feature = "write-sqlite", feature = "metadata-duckdb"))]

use std::collections::{BTreeMap, HashSet};
use std::path::Path;
use std::sync::Arc;

use arrow::array::{Array, RecordBatch};
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeError, DuckLakeTable, MetadataProvider, RowLocation, RowsByRowId,
    SqliteMetadataProvider,
};

use crate::inlined_mutation_differential_tests::{Metadata, duckdb_query, official};
use crate::inlining_abstraction_tests::{MakeLake, batch, create, sqlite};

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

/// `main.<table>` read at `snapshot`, with its `rowid` column.
async fn table_at(
    provider: &Arc<dyn MetadataProvider>,
    table: &str,
    snapshot: i64,
) -> DuckLakeTable {
    let ctx = SessionContext::new();
    let catalog = DuckLakeCatalog::with_snapshot(Arc::clone(provider), snapshot)
        .unwrap()
        .with_row_lineage(true);
    ctx.register_catalog("lake", Arc::new(catalog));
    let provider = ctx
        .catalog("lake")
        .unwrap()
        .schema("main")
        .unwrap()
        .table(table)
        .await
        .unwrap()
        .unwrap();
    (provider.as_ref() as &dyn std::any::Any)
        .downcast_ref::<DuckLakeTable>()
        .expect("a DuckLakeTable")
        .clone()
}

/// A session reading at most `target_partitions` files at once.
fn session(target_partitions: usize) -> SessionContext {
    SessionContext::new_with_config(SessionConfig::new().with_target_partitions(target_partitions))
}

/// Each row of `batch` as its cells, in column order.
fn cells(batch: &RecordBatch) -> Vec<Vec<String>> {
    let mut out = Vec::with_capacity(batch.num_rows());
    for row in 0..batch.num_rows() {
        out.push(
            batch
                .columns()
                .iter()
                .map(|array| {
                    if array.is_null(row) {
                        "NULL".to_string()
                    } else {
                        arrow::util::display::array_value_to_string(array, row).unwrap()
                    }
                })
                .collect(),
        );
    }
    out
}

/// `rowid -> the rest of the line` of `rowid,...` lines.
fn by_rowid(rows: &[String]) -> BTreeMap<i64, String> {
    rows.iter()
        .map(|row| {
            let (rowid, rest) = row.split_once(',').unwrap();
            (rowid.parse().unwrap(), rest.to_string())
        })
        .collect()
}

/// The ids of `requested` found in `truth`, each once, in the order first
/// requested.
fn expected_order(requested: &[i64], truth: &BTreeMap<i64, String>) -> Vec<i64> {
    let mut seen = HashSet::new();
    requested
        .iter()
        .copied()
        .filter(|row_id| truth.contains_key(row_id) && seen.insert(*row_id))
        .collect()
}

/// The indices of `columns` in `table`'s schema.
fn projection(table: &DuckLakeTable, columns: &[&str]) -> Vec<usize> {
    let schema = datafusion::catalog::TableProvider::schema(table);
    columns
        .iter()
        .map(|column| schema.index_of(column).unwrap())
        .collect()
}

/// Read `requested` with every column, and check the rows against `truth`
/// (`rowid -> every column but rowid, comma-joined`): the found rows in the
/// order first requested, each with its values, and every other id missing.
/// Then read each column alone, `rowid` alone and nothing, and check each
/// against the full read.
async fn check_reads(
    table: &DuckLakeTable,
    requested: &[i64],
    truth: &BTreeMap<i64, String>,
    what: &str,
) -> RowsByRowId {
    let state = session(4).state();
    let read = table
        .read_columns_by_rowid(&state, requested, None)
        .await
        .unwrap_or_else(|e| panic!("{what}: {e}"));
    let order = expected_order(requested, truth);
    assert_eq!(read.row_ids, order, "{what}: rows in the order requested");
    assert_eq!(read.batch.num_rows(), order.len(), "{what}");
    let schema = datafusion::catalog::TableProvider::schema(table);
    assert_eq!(read.batch.schema().fields(), schema.fields(), "{what}");
    let rowid_index = schema.index_of("rowid").unwrap();
    let full = cells(&read.batch);
    for (row_id, row) in read.row_ids.iter().zip(&full) {
        assert_eq!(row[rowid_index], row_id.to_string(), "{what}: rowid column");
        let values: Vec<&str> = row
            .iter()
            .enumerate()
            .filter(|(index, _)| *index != rowid_index)
            .map(|(_, cell)| cell.as_str())
            .collect();
        assert_eq!(values.join(","), truth[row_id], "{what}: rowid {row_id}");
    }
    let missing: Vec<i64> = {
        let mut seen = HashSet::new();
        requested
            .iter()
            .copied()
            .filter(|row_id| !truth.contains_key(row_id) && seen.insert(*row_id))
            .collect()
    };
    assert_eq!(read.missing(requested), missing, "{what}: missing ids");

    for index in 0..schema.fields().len() {
        let one = table
            .read_columns_by_rowid(&state, requested, Some(&[index]))
            .await
            .unwrap();
        assert_eq!(one.row_ids, order, "{what}: column {index}");
        assert_eq!(one.batch.schema().field(0), schema.field(index));
        let got: Vec<String> = cells(&one.batch)
            .into_iter()
            .map(|row| row[0].clone())
            .collect();
        let want: Vec<String> = full.iter().map(|row| row[index].clone()).collect();
        assert_eq!(got, want, "{what}: column {index} alone");
    }
    let none = table
        .read_columns_by_rowid(&state, requested, Some(&[]))
        .await
        .unwrap();
    assert_eq!(none.row_ids, order, "{what}: no column");
    assert_eq!(none.batch.num_columns(), 0);
    assert_eq!(none.batch.num_rows(), order.len());
    read
}

// ---------------------------------------------------------------------------
// Official's history
// ---------------------------------------------------------------------------

/// The SQLite catalog official wrote, read by this crate.
async fn sqlite_provider(path: &Path) -> Arc<dyn MetadataProvider> {
    Arc::new(
        SqliteMetadataProvider::new(&format!("sqlite:{}", path.display()))
            .await
            .unwrap(),
    )
}

/// `rowid -> *` of `main.t` at `snapshot`, as official reads it.
fn official_rows(metadata: &Metadata, data: &Path, snapshot: i64) -> BTreeMap<i64, String> {
    by_rowid(&duckdb_query(
        metadata,
        data,
        &format!("SELECT rowid, * FROM lake.main.t AT (VERSION => {snapshot}) ORDER BY rowid"),
    ))
}

/// Run `statements` with official, and return the snapshot they leave.
async fn official_phase(
    metadata: &Metadata,
    path: &Path,
    data: &Path,
    statements: &[String],
) -> i64 {
    official(metadata, data, statements);
    sqlite_provider(path).await.get_current_snapshot().unwrap()
}

/// Through official's history of a table, the rows read by `rowid` at every
/// snapshot are the rows official reads there: file rows, inlined rows of two
/// schema versions, rows behind an inlined deletion, a delete file or a
/// deleted inlined row left out, an updated row's new version, the later rows
/// of a merged file left out at an earlier snapshot, and flushed rows read
/// from the file that embeds their ids. A renamed column reads its old
/// values, a widened one its narrower stored values, and a column the rows
/// predate its default. Guards against a lookup that reads physical columns
/// by name, skips inlined rows, or applies visibility differently from a scan.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn rows_by_rowid_match_official_through_its_history() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("official.db");
    let data = dir.path().join("data");
    std::fs::create_dir_all(&data).unwrap();
    let metadata = Metadata::Sqlite(path.clone());

    let added = dir.path().join("added");
    std::fs::create_dir_all(&added).unwrap();
    let added = added.join("added.parquet");
    let phases: Vec<Vec<String>> = [
        // Two files of one insert each, and a third the next phase's insert
        // is merged with.
        vec![
            "CREATE TABLE lake.main.t (id INTEGER, v VARCHAR, emb FLOAT[], s STRUCT(a INTEGER))"
                .to_string(),
            "CALL lake.set_option('data_inlining_row_limit', 10)".to_string(),
            "INSERT INTO lake.main.t SELECT i, 'a' || i, [i::FLOAT, i::FLOAT + 0.5], {'a': i} \
             FROM range(0, 40) r(i)"
                .to_string(),
            "INSERT INTO lake.main.t (id, v, emb) SELECT i, 'b' || i, [i::FLOAT, i::FLOAT + 0.5] \
             FROM range(40, 80) r(i)"
                .to_string(),
            "INSERT INTO lake.main.t (id, v, emb) SELECT i, 'm' || i, [i::FLOAT] \
             FROM range(500, 520) r(i)"
                .to_string(),
        ],
        vec![
            "INSERT INTO lake.main.t (id, v, emb) SELECT i, 'm' || i, [i::FLOAT] \
             FROM range(520, 540) r(i)"
                .to_string(),
            // Inlined rows.
            "INSERT INTO lake.main.t VALUES (100, 'i100', [1.5], {'a': 1}), \
             (101, 'i101', NULL, NULL), (102, NULL, [2.5, 3.5], NULL)"
                .to_string(),
            // A file registered without field ids, read through a name
            // mapping.
            format!(
                "COPY (SELECT (600 + i)::INTEGER AS id, 'x' || i AS v, [i::FLOAT] AS emb, \
                 {{'a': i::INTEGER}} AS s FROM range(12) r(i)) TO '{}' (FORMAT PARQUET)",
                added.display()
            ),
            format!(
                "CALL ducklake_add_data_files('lake', 't', '{}')",
                added.display()
            ),
        ],
        // An inlined deletion of a file row, a delete file, a deleted inlined
        // row, and an updated file row.
        vec![
            "DELETE FROM lake.main.t WHERE id = 3".to_string(),
            "DELETE FROM lake.main.t WHERE id BETWEEN 50 AND 69".to_string(),
            "DELETE FROM lake.main.t WHERE id = 101".to_string(),
            "UPDATE lake.main.t SET v = 'u45' WHERE id = 45".to_string(),
        ],
        // Schema evolution, and rows inlined and in a file at the new schema.
        vec![
            "ALTER TABLE lake.main.t RENAME COLUMN v TO w".to_string(),
            "ALTER TABLE lake.main.t ALTER COLUMN id TYPE BIGINT".to_string(),
            "ALTER TABLE lake.main.t ADD COLUMN n INTEGER DEFAULT 7".to_string(),
            "ALTER TABLE lake.main.t ADD COLUMN s.c INTEGER DEFAULT 9".to_string(),
            "INSERT INTO lake.main.t (id, w, emb, n) VALUES (200, 'i200', [4.5], 8), \
             (201, 'i201', NULL, NULL)"
                .to_string(),
            "INSERT INTO lake.main.t SELECT i, 'c' || i, [i::FLOAT], {'a': i, 'c': 1}, i \
             FROM range(300, 340) r(i)"
                .to_string(),
            "DELETE FROM lake.main.t WHERE id = 305".to_string(),
        ],
        vec!["CALL ducklake_merge_adjacent_files('lake')".to_string()],
        vec!["CALL ducklake_flush_inlined_data('lake')".to_string()],
    ]
    .into();
    let mut marks = Vec::new();
    let mut saw_inlined = false;
    let mut saw_partial = false;
    for statements in &phases {
        marks.push(official_phase(&metadata, &path, &data, statements).await);
        // Every snapshot so far: the one just made, and the earlier ones as
        // a merge or a flush since left them.
        let provider = sqlite_provider(&path).await;
        for &snapshot in &marks {
            let truth = official_rows(&metadata, &data, snapshot);
            assert!(!truth.is_empty());
            let table = table_at(&provider, "t", snapshot).await;
            saw_partial |= table
                .files()
                .unwrap()
                .iter()
                .any(|file| file.partial_max.is_some_and(|max| max > snapshot));
            let max = *truth.keys().last().unwrap();
            // Every id up to past the last, in an order that interleaves
            // files and inlined rows, with repeats and ids no row has.
            let mut requested: Vec<i64> = (0..=max + 3).rev().collect();
            let third = requested.len() / 3;
            requested.rotate_left(third);
            requested.extend([-1, 7, 7, i64::MAX]);
            saw_inlined |= table
                .locate_rowids(&session(1).state(), &requested)
                .await
                .unwrap()
                .iter()
                .any(|row| matches!(row.location, RowLocation::Inlined { .. }));
            check_reads(&table, &requested, &truth, &format!("snapshot {snapshot}")).await;
        }
    }
    assert!(saw_inlined, "some snapshot holds inlined rows");
    assert!(
        saw_partial,
        "some snapshot reads a merged file holding later rows"
    );
}

// ---------------------------------------------------------------------------
// This crate's history
// ---------------------------------------------------------------------------

/// Rows written by this crate: two files, inlined rows, an inlined deletion
/// of a file row, a delete file, a deleted inlined row and an updated row.
/// The rows read by `rowid` are the rows the `rowid` scan reads, in the order
/// requested at any read concurrency, the deleted ones and ids no row has
/// missing; a projection out of range is refused. Guards against returning
/// rows in the order their reads finish, and against a deleted row read back.
pub(crate) async fn reads_rows_wherever_they_are(make: MakeLake) {
    let lake = make(Some(5)).await;
    create(&lake, "t").await;
    lake.append_parquet("t", &batch(1..=10, "a")).await;
    lake.append_parquet("t", &batch(11..=20, "b")).await;
    lake.append("t", &batch([30, 31, 32], "i")).await;
    let before = by_rowid(&lake.rows("t").await);
    let id_of = |line: &String| line.split(',').next().unwrap().parse::<i64>().unwrap();
    for statement in [
        "DELETE FROM lake.main.t WHERE id = 3",
        "DELETE FROM lake.main.t WHERE id >= 12 AND id <= 18",
        "DELETE FROM lake.main.t WHERE id = 31",
        "UPDATE lake.main.t SET v = 'u' || id WHERE id = 5",
    ] {
        lake.sql(statement).await.unwrap();
    }
    assert!(lake.inlined_rows("t") >= 2, "inlined rows remain");
    assert!(
        lake.inlined_deletes("t") >= 1,
        "a file row's deletion is inlined"
    );
    assert!(lake.delete_files("t") >= 1, "a delete file is live");
    let truth = by_rowid(&lake.rows("t").await);
    let deleted: Vec<i64> = before
        .iter()
        .filter(|(_, line)| [3, 12, 13, 14, 15, 16, 17, 18, 31].contains(&id_of(line)))
        .map(|(row_id, _)| *row_id)
        .collect();
    assert_eq!(deleted.len(), 9);
    assert!(deleted.iter().all(|row_id| !truth.contains_key(row_id)));

    let table = table_at(&lake.provider, "t", lake.head()).await;
    // Inlined rows between file rows, a deleted id, an id no row has, and a
    // repeat.
    let mut requested: Vec<i64> = truth.keys().copied().rev().collect();
    requested.rotate_left(2);
    requested.splice(1..1, deleted.iter().copied());
    let repeat = requested[0];
    requested.extend([1_000_000, repeat]);
    let read = check_reads(&table, &requested, &truth, "crate").await;
    assert_eq!(read.missing(&requested), {
        let mut missing = deleted.clone();
        missing.push(1_000_000);
        missing
    });
    // The order requested, however many reads run at once.
    for partitions in [1, 2, 16] {
        let again = table
            .read_columns_by_rowid(&session(partitions).state(), &requested, None)
            .await
            .unwrap();
        assert_eq!(again.row_ids, read.row_ids, "{partitions} at once");
        assert_eq!(again.batch, read.batch, "{partitions} at once");
    }
    // A reversed request reads the rows reversed.
    let reversed: Vec<i64> = read.row_ids.iter().rev().copied().collect();
    let back = table
        .read_columns_by_rowid(&session(4).state(), &reversed, Some(&[1, 0]))
        .await
        .unwrap();
    assert_eq!(back.row_ids, reversed);
    let empty = table
        .read_columns_by_rowid(&session(4).state(), &[], None)
        .await
        .unwrap();
    assert!(empty.row_ids.is_empty());
    assert_eq!(empty.batch.num_rows(), 0);
    let error = table
        .read_columns_by_rowid(&session(4).state(), &requested, Some(&[0, 3]))
        .await
        .unwrap_err();
    assert!(
        matches!(&error, DuckLakeError::InvalidConfig(message) if message.contains("projection index 3")),
        "{error}"
    );
    // The projection names columns by their place in the table's schema.
    assert_eq!(projection(&table, &["v", "rowid"]), vec![1, 2]);
}

/// A table read at an earlier snapshot finds the rows it held then: not a
/// row inserted later, inlined or merged into a file with earlier rows, and
/// still a row deleted later. Guards against reading rows by position at any
/// snapshot.
pub(crate) async fn reads_rows_at_the_tables_snapshot(make: MakeLake) {
    let lake = make(Some(5)).await;
    create(&lake, "t").await;
    lake.append_parquet("t", &batch(1..=8, "a")).await;
    let first = lake.head();
    let early = by_rowid(&lake.rows("t").await);
    lake.append_parquet("t", &batch(9..=16, "b")).await;
    lake.append("t", &batch([30, 31], "i")).await;
    lake.sql("DELETE FROM lake.main.t WHERE id = 2")
        .await
        .unwrap();
    let (ctx, table) = lake.table("t").await;
    let merged = table
        .merge_adjacent_files(&ctx.state(), datafusion_ducklake::MergeOptions::default())
        .await
        .unwrap();
    let later = by_rowid(&lake.rows("t").await);
    assert!(later.len() > early.len());
    let table = table_at(&lake.provider, "t", first).await;
    if merged.files_created > 0 {
        assert!(
            table
                .files()
                .unwrap()
                .iter()
                .any(|file| file.partial_max.is_some_and(|max| max > first)),
            "the merged file is read at the first snapshot"
        );
    }
    let all: Vec<i64> = early.keys().chain(later.keys()).copied().collect();
    let read = check_reads(&table, &all, &early, "first snapshot").await;
    assert_eq!(
        read.row_ids.len(),
        early.len(),
        "the later rows are missing"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn sqlite_reads_rows_wherever_they_are() {
    reads_rows_wherever_they_are(sqlite()).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn sqlite_reads_rows_at_the_tables_snapshot() {
    reads_rows_at_the_tables_snapshot(sqlite()).await;
}

#[cfg(feature = "write-postgres")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn multicatalog_postgres_reads_rows_by_rowid() {
    use testcontainers::runners::AsyncRunner;
    let container = testcontainers_modules::postgres::Postgres::default()
        .start()
        .await
        .unwrap();
    let port = container.get_host_port_ipv4(5432).await.unwrap();
    let pool = sqlx::PgPool::connect(&format!(
        "postgresql://postgres:postgres@127.0.0.1:{port}/postgres"
    ))
    .await
    .unwrap();
    datafusion_ducklake::initialize_multicatalog_schema(&pool)
        .await
        .unwrap();
    let make = crate::inlining_abstraction_tests::multicatalog(pool);
    Box::pin(reads_rows_wherever_they_are(Arc::clone(&make))).await;
    Box::pin(reads_rows_at_the_tables_snapshot(make)).await;
}
