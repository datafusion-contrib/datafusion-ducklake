//! The change feeds follow official DuckLake's main branch, ahead of the
//! extension this crate pins, for three cases of its `GetTableDeletions` and
//! `GetTableInsertions`:
//!
//! - a data file dropped in the snapshot that also wrote a delete file for it
//!   (a transaction that deletes part of a file, then the rest) reports each
//!   row once: the drop skips the rows of every delete file up to and
//!   including its own snapshot;
//! - a data file retired whole reports its rows deleted inlined in the window
//!   at their own snapshots and its other rows at the retirement, and not the
//!   rows deleted inlined before the window;
//! - a rewrite (`ducklake_rewrite_data_files`) changes no row, so the feeds
//!   report nothing for it: not the files it ends, not the files it writes,
//!   and not the rows of its outputs a merge later carries.
//!
//! The expected rows are those of official's own tests at its main branch:
//! `test/sql/table_changes/table_changes_delete_rest_of_file.test`,
//! `table_changes_rewrite_data_files.test`,
//! `table_changes_rewrite_merged_twice.test` and
//! `test/sql/issues/issue_1571_rewritten_inlined_delete_changes.test`. Each
//! history is written by official DuckDB into DuckDB, SQLite and PostgreSQL
//! metadata and read by this crate, and the rewrite histories are also written
//! by this crate's own writers (SQLite, and PostgreSQL in the multi-catalog
//! layout). Official's `table_changes_delete_rest_of_file_flush.test` is not
//! here: the extension the tests run fails its history (a flush of inlined
//! deletions and two `DELETE`s in one transaction) with an internal error, and
//! this crate's writers commit no such transaction.

#![cfg(all(feature = "write-sqlite", feature = "metadata-sqlite", feature = "metadata-duckdb"))]

use std::path::Path;
use std::sync::Arc;

use arrow::util::display::array_value_to_string;
use datafusion::prelude::SessionContext;
use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeTable, DuckLakeWriteOptions, DuckdbMetadataProvider, MergeOptions,
    MetadataProvider, MetadataWriter, RewriteOptions, execute_ducklake_sql,
    register_ducklake_functions,
};

use crate::inlined_mutation_differential_tests::{Metadata, duckdb_connection};

/// Sorted rows of `sql`, cells joined by `|`.
async fn rows(ctx: &SessionContext, sql: &str) -> Vec<String> {
    let batches = ctx
        .sql(sql)
        .await
        .unwrap_or_else(|e| panic!("`{sql}`: {e}"))
        .collect()
        .await
        .unwrap_or_else(|e| panic!("`{sql}`: {e}"));
    let mut out = Vec::new();
    for batch in &batches {
        for row in 0..batch.num_rows() {
            let cells: Vec<String> = batch
                .columns()
                .iter()
                .map(|column| array_value_to_string(column, row).unwrap())
                .collect();
            out.push(cells.join("|"));
        }
    }
    out.sort();
    out
}

fn expect(rows: &[&str]) -> Vec<String> {
    let mut rows: Vec<String> = rows.iter().map(|row| row.to_string()).collect();
    rows.sort();
    rows
}

/// `SELECT {select} FROM {function}('main.{table}', {a}, {b})`, sorted.
async fn feed(
    ctx: &SessionContext,
    function: &str,
    table: &str,
    (a, b): (i64, i64),
    select: &str,
) -> Vec<String> {
    rows(
        ctx,
        &format!("SELECT {select} FROM ducklake_{function}('main.{table}', {a}, {b})"),
    )
    .await
}

/// `snapshot_id|change_type|count` of a window's changes.
async fn change_counts(ctx: &SessionContext, table: &str, window: (i64, i64)) -> Vec<String> {
    let (a, b) = window;
    rows(
        ctx,
        &format!(
            "SELECT snapshot_id, change_type, count(*) \
             FROM ducklake_table_changes('main.{table}', {a}, {b}) GROUP BY 1, 2"
        ),
    )
    .await
}

/// `snapshot_id|count|min(id)|max(id)` of a window's deletions.
async fn deletion_counts(ctx: &SessionContext, table: &str, window: (i64, i64)) -> Vec<String> {
    let (a, b) = window;
    rows(
        ctx,
        &format!(
            "SELECT snapshot_id, count(*), min(id), max(id) \
             FROM ducklake_table_deletions('main.{table}', {a}, {b}) GROUP BY 1"
        ),
    )
    .await
}

// ---------------------------------------------------------------------------
// Histories written by official DuckDB
// ---------------------------------------------------------------------------

/// An official DuckDB session over a lake attached as `dl`.
struct Official {
    connection: duckdb::Connection,
}

impl Official {
    fn attach(metadata: &Metadata, data: &Path, inlining_row_limit: usize) -> Self {
        let connection = duckdb_connection();
        connection
            .execute_batch(&format!(
                "ATTACH '{}' AS dl (DATA_PATH '{}/', DATA_INLINING_ROW_LIMIT {inlining_row_limit})",
                metadata.ducklake_uri(),
                data.display()
            ))
            .unwrap();
        Self {
            connection,
        }
    }

    fn run(&self, statements: &str) {
        self.connection
            .execute_batch(statements)
            .unwrap_or_else(|e| panic!("official `{statements}`: {e}"));
    }

    /// The latest snapshot.
    fn head(&self) -> i64 {
        self.connection
            .query_row(
                "SELECT max(snapshot_id) FROM ducklake_snapshots('dl')",
                [],
                |row| row.get(0),
            )
            .unwrap()
    }
}

/// This crate's metadata provider over the lake official wrote.
async fn crate_provider(metadata: &Metadata) -> Arc<dyn MetadataProvider> {
    match metadata {
        Metadata::Duckdb(path) => {
            Arc::new(DuckdbMetadataProvider::new(path.to_str().unwrap()).unwrap())
        },
        Metadata::Sqlite(path) => Arc::new(
            datafusion_ducklake::SqliteMetadataProvider::new(&format!("sqlite:{}", path.display()))
                .await
                .unwrap(),
        ),
        #[cfg(feature = "metadata-postgres")]
        Metadata::Postgres(dsn) => {
            let url = dsn_url(dsn);
            Arc::new(
                datafusion_ducklake::PostgresMetadataProvider::new(&url)
                    .await
                    .unwrap(),
            )
        },
        _ => unreachable!("not a metadata backend these tests write"),
    }
}

/// This crate reading the lake official wrote.
async fn crate_reader(metadata: &Metadata) -> SessionContext {
    let provider = crate_provider(metadata).await;
    let snapshot = provider.get_current_snapshot().unwrap();
    let catalog = DuckLakeCatalog::with_snapshot(Arc::clone(&provider), snapshot).unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));
    register_ducklake_functions(&ctx, provider);
    ctx
}

/// The `postgresql://` URL of a libpq `key=value` DSN.
#[cfg(feature = "metadata-postgres")]
fn dsn_url(dsn: &str) -> String {
    let field = |key: &str| {
        dsn.split_whitespace()
            .find_map(|pair| pair.strip_prefix(&format!("{key}=")))
            .unwrap()
            .to_string()
    };
    format!(
        "postgresql://{}:{}@{}:{}/{}",
        field("user"),
        field("password"),
        field("host"),
        field("port"),
        field("dbname")
    )
}

/// `table_changes_delete_rest_of_file.test`: a transaction that deletes part
/// of a data file and then the rest reports every deleted row once.
async fn delete_rest_of_file(metadata: &Metadata, data: &Path) {
    let (s1, s2, s3) = {
        let official = Official::attach(metadata, data, 0);
        official.run("CREATE TABLE dl.t AS SELECT * FROM (VALUES (1, 'a'), (2, 'b')) v(id, name)");
        let s1 = official.head();
        official.run("BEGIN; DELETE FROM dl.t WHERE id = 1; DELETE FROM dl.t WHERE id = 2; COMMIT");
        official.run(
            "CREATE TABLE dl.m AS SELECT * FROM (VALUES (1, 'a'), (2, 'b')) v(id, name);
             CREATE TABLE src AS SELECT * FROM (VALUES (1, 'A'), (3, 'c')) v(id, name);",
        );
        let s2 = official.head();
        official.run(
            "BEGIN;
             MERGE INTO dl.m AS t USING src AS s ON t.id = s.id
                 WHEN MATCHED THEN UPDATE SET name = s.name
                 WHEN NOT MATCHED THEN INSERT VALUES (s.id, s.name);
             MERGE INTO dl.m AS t USING src AS s ON t.id = s.id
                 WHEN NOT MATCHED BY SOURCE THEN DELETE;
             COMMIT",
        );
        official.run(
            "CREATE TABLE dl.c AS SELECT i AS id FROM range(4) t(i);
             DELETE FROM dl.c WHERE id = 0;",
        );
        let s3 = official.head();
        official
            .run("BEGIN; DELETE FROM dl.c WHERE id = 1; DELETE FROM dl.c WHERE id >= 2; COMMIT");
        (s1, s2, s3)
    };
    let ctx = crate_reader(metadata).await;
    let changes = "rowid, change_type, id, name";
    assert_eq!(
        feed(&ctx, "table_changes", "t", (s1 + 1, s1 + 1), changes).await,
        expect(&["0|delete|1|a", "1|delete|2|b"]),
    );
    assert_eq!(
        feed(&ctx, "table_changes", "m", (s2 + 1, s2 + 1), changes).await,
        expect(&[
            "0|update_postimage|1|A",
            "0|update_preimage|1|a",
            "1|delete|2|b",
            "3|insert|3|c",
        ]),
    );
    assert_eq!(
        feed(&ctx, "table_changes", "c", (s3, s3), "rowid, change_type").await,
        expect(&["0|delete"]),
    );
    assert_eq!(
        feed(
            &ctx,
            "table_changes",
            "c",
            (s3 + 1, s3 + 1),
            "rowid, change_type"
        )
        .await,
        expect(&["1|delete", "2|delete", "3|delete"]),
    );
    // The deletions feed agrees.
    assert_eq!(
        feed(
            &ctx,
            "table_deletions",
            "c",
            (s3, s3 + 1),
            "snapshot_id, rowid"
        )
        .await,
        expect(&[
            &format!("{s3}|0"),
            &format!("{}|1", s3 + 1),
            &format!("{}|2", s3 + 1),
            &format!("{}|3", s3 + 1),
        ]),
    );
}

/// `table_changes_rewrite_data_files.test`: a rewrite of data files changes no
/// rows, so the change feed reports nothing for it.
async fn rewrite_data_files(metadata: &Metadata, data: &Path) {
    let (inserted, rewrite, merge, update, rewrite2) = {
        let official = Official::attach(metadata, data, 2);
        official.run(
            "CREATE TABLE dl.t (id INTEGER);
             INSERT INTO dl.t SELECT range FROM range(10);
             DELETE FROM dl.t WHERE id < 3;
             DELETE FROM dl.t WHERE id = 3;",
        );
        official.run("CALL ducklake_rewrite_data_files('dl', 't', delete_threshold => 0)");
        let rewrite = official.head();
        official.run("INSERT INTO dl.t SELECT range FROM range(100, 103)");
        let inserted = official.head();
        official.run("CALL ducklake_merge_adjacent_files('dl')");
        let merge = official.head();
        official.run("UPDATE dl.t SET id = 50 WHERE id = 5");
        let update = official.head();
        official.run("DELETE FROM dl.t WHERE id = 6");
        official.run("CALL ducklake_rewrite_data_files('dl', 't', delete_threshold => 0)");
        let rewrite2 = official.head();
        assert_eq!(
            (rewrite, inserted, merge, update, rewrite2),
            (5, 6, 7, 8, 10),
            "the snapshots of official's test"
        );
        (inserted, rewrite, merge, update, rewrite2)
    };
    let ctx = crate_reader(metadata).await;
    let all = "snapshot_id, rowid, change_type, id";
    for function in ["table_insertions", "table_deletions", "table_changes"] {
        assert!(
            feed(&ctx, function, "t", (rewrite, rewrite), "*")
                .await
                .is_empty(),
            "{function} of the rewrite"
        );
    }
    let deletes = ["3|0|delete|0", "3|1|delete|1", "3|2|delete|2", "4|3|delete|3"];
    let mut whole: Vec<String> = (0..10).map(|i| format!("2|{i}|insert|{i}")).collect();
    whole.extend(deletes.iter().map(|row| row.to_string()));
    whole.sort();
    assert_eq!(
        feed(&ctx, "table_changes", "t", (0, rewrite), all).await,
        whole
    );
    assert_eq!(
        feed(&ctx, "table_changes", "t", (3, rewrite), all).await,
        expect(&deletes)
    );
    // The merge of the rewritten file with a newer one.
    assert_eq!(
        feed(&ctx, "table_insertions", "t", (rewrite, merge), "rowid, id").await,
        expect(&["10|100", "11|101", "12|102"]),
    );
    assert_eq!(
        feed(&ctx, "table_insertions", "t", (0, merge), "count(*)").await,
        expect(&["13"]),
    );
    assert_eq!(
        feed(
            &ctx,
            "table_changes",
            "t",
            (0, merge),
            "count(*) FILTER (WHERE change_type LIKE 'update%')"
        )
        .await,
        expect(&["0"]),
    );
    // Changes after the rewrite are reported as before.
    assert_eq!(
        feed(&ctx, "table_changes", "t", (update, update + 1), all).await,
        expect(&["8|5|update_postimage|50", "8|5|update_preimage|5", "9|6|delete|6"]),
    );
    assert!(
        feed(&ctx, "table_changes", "t", (rewrite2, rewrite2), "*")
            .await
            .is_empty()
    );
    assert_eq!(
        feed(&ctx, "table_changes", "t", (inserted, rewrite2), all).await,
        expect(&[
            "6|10|insert|100",
            "6|11|insert|101",
            "6|12|insert|102",
            "8|5|update_postimage|50",
            "8|5|update_preimage|5",
            "9|6|delete|6",
        ]),
    );
}

/// `table_changes_rewrite_merged_twice.test`: the rows of two rewrites merged
/// into one file are not changes.
async fn rewrite_merged_twice(metadata: &Metadata, data: &Path) {
    let merge = {
        let official = Official::attach(metadata, data, 0);
        official.run(
            "CREATE TABLE dl.t (id INTEGER);
             INSERT INTO dl.t SELECT range FROM range(10);
             INSERT INTO dl.t SELECT range FROM range(10, 20);
             DELETE FROM dl.t WHERE id = 0;",
        );
        official.run("CALL ducklake_rewrite_data_files('dl', 't', delete_threshold => 0)");
        official.run("DELETE FROM dl.t WHERE id = 10");
        official.run("CALL ducklake_rewrite_data_files('dl', 't', delete_threshold => 0)");
        official.run("CALL ducklake_merge_adjacent_files('dl')");
        official.head()
    };
    assert_eq!(merge, 8, "the snapshots of official's test");
    let ctx = crate_reader(metadata).await;
    assert_eq!(
        change_counts(&ctx, "t", (0, 8)).await,
        expect(&["2|insert|10", "3|insert|10", "4|delete|1", "6|delete|1"]),
    );
    assert_eq!(
        feed(&ctx, "table_insertions", "t", (5, 8), "count(*)").await,
        expect(&["0"]),
    );
}

/// `issue_1571_rewritten_inlined_delete_changes.test`: rewriting a data file
/// with inlined deletes keeps the deletions and changes of every snapshot
/// range.
async fn rewritten_inlined_deletes(metadata: &Metadata, data: &Path) {
    {
        let official = Official::attach(metadata, data, 5);
        official.run(
            "CREATE TABLE dl.t (id INTEGER);
             INSERT INTO dl.t SELECT range FROM range(20);
             DELETE FROM dl.t WHERE id = 0;",
        );
        official.run("CALL ducklake_rewrite_data_files('dl', 't', delete_threshold => 0)");
        official.run(
            "CREATE TABLE dl.u (id INTEGER);
             INSERT INTO dl.u SELECT range FROM range(20);
             DELETE FROM dl.u WHERE id < 12;
             DELETE FROM dl.u WHERE id = 12;
             DELETE FROM dl.u WHERE id = 13;",
        );
        official.run("CALL ducklake_rewrite_data_files('dl', 'u', delete_threshold => 0)");
        assert_eq!(official.head(), 10, "the snapshots of official's test");
    }
    let ctx = crate_reader(metadata).await;
    assert!(deletion_counts(&ctx, "t", (4, 4)).await.is_empty());
    assert_eq!(
        deletion_counts(&ctx, "t", (3, 4)).await,
        expect(&["3|1|0|0"])
    );
    assert_eq!(
        change_counts(&ctx, "t", (3, 4)).await,
        expect(&["3|delete|1"])
    );
    assert!(change_counts(&ctx, "t", (4, 4)).await.is_empty());
    assert_eq!(
        deletion_counts(&ctx, "u", (7, 10)).await,
        expect(&["7|12|0|11", "8|1|12|12", "9|1|13|13"]),
    );
    assert_eq!(
        deletion_counts(&ctx, "u", (9, 10)).await,
        expect(&["9|1|13|13"])
    );
    assert_eq!(
        change_counts(&ctx, "u", (0, 10)).await,
        expect(&["6|insert|20", "7|delete|12", "8|delete|1", "9|delete|1"]),
    );
    // Applying the whole change feed gives the current rows.
    assert_eq!(
        feed(
            &ctx,
            "table_changes",
            "u",
            (0, 10),
            "sum(CASE WHEN change_type IN ('insert', 'update_postimage') THEN 1 ELSE -1 END)"
        )
        .await,
        expect(&["6"]),
    );
}

/// Inlined deletions of two rows of a data file, then a `DELETE` of the rest
/// (inlining limit 5). Official has no test of this history, and the
/// extension the tests run does not retire the file whole: the last `DELETE`
/// writes a delete file of the other 18 rows, its inlined deletions not
/// counted. The expected rows follow official's main-branch code:
///
/// - `GetTableDeletions` (`src/storage/ducklake_metadata_manager.cpp`) reads
///   the delete file as an entry of the window when it begins in it, or when
///   the file has inlined deletions in the window; attaches to each entry the
///   file's inlined deletions in the window; and, for a file with inlined
///   deletions in the window and no delete file begun by its end, reads an
///   entry of those deletions alone.
/// - `DuckLakeDeleteFilter::Initialize`
///   (`src/storage/ducklake_delete_filter.cpp`) scans a delete file's rows at
///   its snapshot (the file records none per row), and each inlined deletion
///   of the window at its own.
///
/// So every row is deleted once, at the snapshot that deleted it. The
/// catalog's own read of the inlined deletions, of every data file or of some
/// (`file_id IN`), is checked against the same history.
async fn inlined_deletes_then_delete_rest(metadata: &Metadata, data: &Path) {
    let (inserted, d1, d2, d3) = {
        let official = Official::attach(metadata, data, 5);
        official.run(
            "CREATE TABLE dl.t (id INTEGER);
             INSERT INTO dl.t SELECT range FROM range(20);",
        );
        let inserted = official.head();
        official.run("DELETE FROM dl.t WHERE id = 0");
        let d1 = official.head();
        official.run("DELETE FROM dl.t WHERE id = 1");
        let d2 = official.head();
        official.run("DELETE FROM dl.t");
        let d3 = official.head();
        // The history has the shape it is for: the file is live, with a
        // delete file of 18 rows at the last snapshot and two inlined
        // deletions before it.
        let count = |sql: &str| -> i64 {
            official
                .connection
                .query_row(sql, [], |row| row.get(0))
                .unwrap_or_else(|e| panic!("official `{sql}`: {e}"))
        };
        assert_eq!(
            count(
                "SELECT count(*) FROM __ducklake_metadata_dl.ducklake_data_file \
                 WHERE end_snapshot IS NULL"
            ),
            1
        );
        assert_eq!(
            count(&format!(
                "SELECT sum(delete_count) FROM __ducklake_metadata_dl.ducklake_delete_file \
                 WHERE begin_snapshot = {d3}"
            )),
            18
        );
        assert_eq!(
            count(
                "SELECT count(*) FROM dl.snapshots() \
                 WHERE map_contains(changes, 'inlined_delete')"
            ),
            2
        );
        (inserted, d1, d2, d3)
    };
    let ctx = crate_reader(metadata).await;
    let at = |snapshot: i64, rows: &str| format!("{snapshot}|{rows}");
    assert_eq!(
        deletion_counts(&ctx, "t", (d3, d3)).await,
        vec![at(d3, "18|2|19")]
    );
    assert_eq!(
        deletion_counts(&ctx, "t", (d2, d3)).await,
        expect(&[&at(d2, "1|1|1"), &at(d3, "18|2|19")]),
    );
    assert_eq!(
        deletion_counts(&ctx, "t", (inserted, d3)).await,
        expect(&[&at(d1, "1|0|0"), &at(d2, "1|1|1"), &at(d3, "18|2|19")]),
    );
    assert_eq!(
        deletion_counts(&ctx, "t", (d1, d2)).await,
        expect(&[&at(d1, "1|0|0"), &at(d2, "1|1|1")]),
    );
    assert_eq!(
        deletion_counts(&ctx, "t", (d2, d2)).await,
        vec![at(d2, "1|1|1")]
    );
    assert_eq!(
        change_counts(&ctx, "t", (0, d3)).await,
        expect(&[
            &format!("{inserted}|insert|20"),
            &format!("{d1}|delete|1"),
            &format!("{d2}|delete|1"),
            &format!("{d3}|delete|18"),
        ]),
    );

    // The catalog read of the inlined deletions, whole and by data file.
    let provider = crate_provider(metadata).await;
    let schema = provider.get_schema_by_name("main", d3).unwrap().unwrap();
    let table_id = provider
        .get_table_by_name(schema.schema_id, "t", d3)
        .unwrap()
        .unwrap()
        .table_id;
    let all = provider
        .get_inlined_file_deletes_between(table_id, 0, d3)
        .unwrap();
    assert_eq!(
        all.iter()
            .map(|c| (c.position, c.snapshot_id))
            .collect::<Vec<_>>(),
        vec![(0, d1), (1, d2)]
    );
    let file = all[0].data_file_id;
    let of_files = |ids: &[i64], window: (i64, i64)| {
        provider
            .get_inlined_file_deletes_of_files_between(table_id, ids, window.0, window.1)
            .unwrap()
    };
    assert_eq!(of_files(&[file], (0, d3)), all);
    assert_eq!(of_files(&[file + 1, file], (0, d3)), all);
    assert_eq!(of_files(&[file], (0, d1)), all[..1].to_vec());
    assert!(of_files(&[file + 1], (0, d3)).is_empty());
    assert!(of_files(&[], (0, d3)).is_empty());
}

fn data_dir(dir: &Path, name: &str) -> std::path::PathBuf {
    let data = dir.join(format!("{name}_data"));
    std::fs::create_dir_all(&data).unwrap();
    data
}

/// One official history on each metadata backend official writes.
macro_rules! official_history {
    ($name:ident) => {
        mod $name {
            use super::*;

            #[tokio::test(flavor = "multi_thread")]
            async fn duckdb_metadata() {
                let dir = tempfile::tempdir().unwrap();
                let metadata = Metadata::Duckdb(dir.path().join("lake.ducklake"));
                super::$name(&metadata, &data_dir(dir.path(), "lake")).await;
            }

            #[tokio::test(flavor = "multi_thread")]
            async fn sqlite_metadata() {
                let dir = tempfile::tempdir().unwrap();
                let metadata = Metadata::Sqlite(dir.path().join("lake.sqlite"));
                super::$name(&metadata, &data_dir(dir.path(), "lake")).await;
            }

            #[cfg(feature = "metadata-postgres")]
            #[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
            #[tokio::test(flavor = "multi_thread")]
            async fn postgres_metadata() {
                use testcontainers::runners::AsyncRunner;
                let container = testcontainers_modules::postgres::Postgres::default()
                    .start()
                    .await
                    .unwrap();
                let port = container.get_host_port_ipv4(5432).await.unwrap();
                let dir = tempfile::tempdir().unwrap();
                let metadata = Metadata::Postgres(format!(
                    "host=127.0.0.1 port={port} user=postgres password=postgres dbname=postgres"
                ));
                super::$name(&metadata, &data_dir(dir.path(), "lake")).await;
            }
        }
    };
}

official_history!(delete_rest_of_file);
official_history!(rewrite_data_files);
official_history!(rewrite_merged_twice);
official_history!(rewritten_inlined_deletes);
official_history!(inlined_deletes_then_delete_rest);

// ---------------------------------------------------------------------------
// Histories written by this crate
// ---------------------------------------------------------------------------

/// A lake this crate writes, with data inlining up to `limit` rows.
struct Lake {
    writer: Arc<dyn MetadataWriter>,
    provider: Arc<dyn MetadataProvider>,
    limit: usize,
    /// What the lake lives in: its directory, and its container.
    _keep: Box<dyn std::any::Any + Send>,
}

impl Lake {
    fn session(&self) -> (SessionContext, DuckLakeCatalog) {
        let ctx = SessionContext::new();
        let make = || {
            DuckLakeCatalog::with_writer(Arc::clone(&self.provider), Arc::clone(&self.writer))
                .unwrap()
                .with_write_options(
                    DuckLakeWriteOptions::default().with_data_inlining_row_limit(self.limit),
                )
        };
        DuckLakeCatalog::register(&ctx, "lake", make());
        register_ducklake_functions(&ctx, Arc::clone(&self.provider));
        (ctx, make())
    }

    /// Run `statement` and return the snapshot it made.
    async fn sql(&self, statement: &str) -> i64 {
        let (ctx, catalog) = self.session();
        execute_ducklake_sql(&ctx, &catalog, statement)
            .await
            .unwrap_or_else(|e| panic!("{statement}: {e}"))
            .collect()
            .await
            .unwrap_or_else(|e| panic!("{statement}: {e}"));
        self.head()
    }

    fn head(&self) -> i64 {
        self.provider.get_current_snapshot().unwrap()
    }

    async fn table(&self, name: &str) -> (SessionContext, DuckLakeTable) {
        let (ctx, _) = self.session();
        let provider = ctx
            .catalog("lake")
            .unwrap()
            .schema("main")
            .unwrap()
            .table(name)
            .await
            .unwrap()
            .unwrap();
        let table = (provider.as_ref() as &dyn std::any::Any)
            .downcast_ref::<DuckLakeTable>()
            .expect("a DuckLakeTable")
            .clone();
        (ctx, table)
    }

    /// `ducklake_rewrite_data_files` with `delete_threshold => 0`; returns its
    /// snapshot.
    async fn rewrite(&self, name: &str) -> i64 {
        let (ctx, table) = self.table(name).await;
        let result = table
            .rewrite_data_files(
                &ctx.state(),
                RewriteOptions {
                    delete_threshold: 0.0,
                    data_file_ids: None,
                },
            )
            .await
            .unwrap();
        assert!(result.did_work(), "nothing rewritten: {result:?}");
        self.head()
    }

    /// `ducklake_merge_adjacent_files`; returns its snapshot.
    async fn merge(&self, name: &str) -> i64 {
        let (ctx, table) = self.table(name).await;
        let result = table
            .merge_adjacent_files(
                &ctx.state(),
                MergeOptions {
                    target_file_size: 1 << 30,
                    max_merged_files: usize::MAX,
                    min_file_size: 0,
                    ..MergeOptions::default()
                },
            )
            .await
            .unwrap();
        assert!(result.did_work(), "nothing merged: {result:?}");
        self.head()
    }

    /// The number of the table's live data files.
    fn data_files(&self, name: &str) -> usize {
        let head = self.head();
        let schema = self
            .provider
            .get_schema_by_name("main", head)
            .unwrap()
            .unwrap();
        let table = self
            .provider
            .get_table_by_name(schema.schema_id, name, head)
            .unwrap()
            .unwrap();
        self.provider
            .get_table_files_for_select(table.table_id, head)
            .unwrap()
            .len()
    }
}

// The rewrite histories of official's tests, written by this crate. Snapshot
// ids are this catalog's own, so each window is spelled by the steps that
// bound it.

/// `issue_1571_rewritten_inlined_delete_changes.test`, table `t` (inlining
/// limit 5).
async fn crate_rewritten_inlined_delete(lake_t: Lake) {
    lake_t.sql("CREATE TABLE lake.main.t (id INTEGER)").await;
    let inserted = lake_t
        .sql("INSERT INTO lake.main.t SELECT CAST(value AS INTEGER) FROM range(20)")
        .await;
    let deleted = lake_t.sql("DELETE FROM lake.main.t WHERE id = 0").await;
    let rewrite = lake_t.rewrite("t").await;
    let (ctx, _) = lake_t.session();
    assert!(
        deletion_counts(&ctx, "t", (rewrite, rewrite))
            .await
            .is_empty()
    );
    assert_eq!(
        deletion_counts(&ctx, "t", (deleted, rewrite)).await,
        vec![format!("{deleted}|1|0|0")]
    );
    assert_eq!(
        change_counts(&ctx, "t", (deleted, rewrite)).await,
        vec![format!("{deleted}|delete|1")]
    );
    assert!(
        change_counts(&ctx, "t", (rewrite, rewrite))
            .await
            .is_empty()
    );
    assert_eq!(
        change_counts(&ctx, "t", (inserted, rewrite)).await,
        expect(&[&format!("{inserted}|insert|20"), &format!("{deleted}|delete|1"),]),
    );
}

/// `issue_1571_rewritten_inlined_delete_changes.test`, table `u`: a delete
/// file and inlined deletions before and inside the range (inlining limit 5).
async fn crate_rewritten_delete_file_and_inlined_deletes(lake_u: Lake) {
    lake_u.sql("CREATE TABLE lake.main.u (id INTEGER)").await;
    let inserted = lake_u
        .sql("INSERT INTO lake.main.u SELECT CAST(value AS INTEGER) FROM range(20)")
        .await;
    let d1 = lake_u.sql("DELETE FROM lake.main.u WHERE id < 12").await;
    let d2 = lake_u.sql("DELETE FROM lake.main.u WHERE id = 12").await;
    let d3 = lake_u.sql("DELETE FROM lake.main.u WHERE id = 13").await;
    let rewrite = lake_u.rewrite("u").await;
    let (ctx, _) = lake_u.session();
    assert_eq!(
        deletion_counts(&ctx, "u", (d1, rewrite)).await,
        expect(&[&format!("{d1}|12|0|11"), &format!("{d2}|1|12|12"), &format!("{d3}|1|13|13"),]),
    );
    assert_eq!(
        deletion_counts(&ctx, "u", (d3, rewrite)).await,
        vec![format!("{d3}|1|13|13")]
    );
    assert_eq!(
        change_counts(&ctx, "u", (0, rewrite)).await,
        expect(&[
            &format!("{inserted}|insert|20"),
            &format!("{d1}|delete|12"),
            &format!("{d2}|delete|1"),
            &format!("{d3}|delete|1"),
        ]),
    );
}

/// `table_changes_rewrite_data_files.test` (inlining limit 2).
async fn crate_rewrite_data_files(lake_r: Lake) {
    lake_r.sql("CREATE TABLE lake.main.t (id INTEGER)").await;
    let inserted = lake_r
        .sql("INSERT INTO lake.main.t SELECT CAST(value AS INTEGER) FROM range(10)")
        .await;
    let d1 = lake_r.sql("DELETE FROM lake.main.t WHERE id < 3").await;
    let d2 = lake_r.sql("DELETE FROM lake.main.t WHERE id = 3").await;
    let rewrite = lake_r.rewrite("t").await;
    let (ctx, _) = lake_r.session();
    for function in ["table_insertions", "table_deletions", "table_changes"] {
        assert!(
            feed(&ctx, function, "t", (rewrite, rewrite), "*")
                .await
                .is_empty(),
            "{function} of the rewrite"
        );
    }
    let all = "snapshot_id, rowid, change_type, id";
    let deletes = [
        format!("{d1}|0|delete|0"),
        format!("{d1}|1|delete|1"),
        format!("{d1}|2|delete|2"),
        format!("{d2}|3|delete|3"),
    ];
    let mut whole: Vec<String> = (0..10)
        .map(|i| format!("{inserted}|{i}|insert|{i}"))
        .collect();
    whole.extend(deletes.iter().cloned());
    whole.sort();
    assert_eq!(
        feed(&ctx, "table_changes", "t", (0, rewrite), all).await,
        whole
    );
    let mut sorted = deletes.to_vec();
    sorted.sort();
    assert_eq!(
        feed(&ctx, "table_changes", "t", (d1, rewrite), all).await,
        sorted
    );
    let inserted2 = lake_r
        .sql("INSERT INTO lake.main.t SELECT CAST(value AS INTEGER) FROM range(100, 103)")
        .await;
    let merge = lake_r.merge("t").await;
    assert_eq!(lake_r.data_files("t"), 1, "the merge makes one file");
    let (ctx, _) = lake_r.session();
    assert_eq!(
        feed(&ctx, "table_insertions", "t", (rewrite, merge), "rowid, id").await,
        expect(&["10|100", "11|101", "12|102"]),
    );
    assert_eq!(
        feed(&ctx, "table_insertions", "t", (0, merge), "count(*)").await,
        expect(&["13"]),
    );
    assert_eq!(
        feed(
            &ctx,
            "table_changes",
            "t",
            (0, merge),
            "count(*) FILTER (WHERE change_type LIKE 'update%')"
        )
        .await,
        expect(&["0"]),
    );
    let update = lake_r
        .sql("UPDATE lake.main.t SET id = 50 WHERE id = 5")
        .await;
    let delete = lake_r.sql("DELETE FROM lake.main.t WHERE id = 6").await;
    let rewrite2 = lake_r.rewrite("t").await;
    let (ctx, _) = lake_r.session();
    let later = [
        format!("{update}|5|update_postimage|50"),
        format!("{update}|5|update_preimage|5"),
        format!("{delete}|6|delete|6"),
    ];
    let mut sorted = later.to_vec();
    sorted.sort();
    assert_eq!(
        feed(&ctx, "table_changes", "t", (update, delete), all).await,
        sorted
    );
    assert!(
        feed(&ctx, "table_changes", "t", (rewrite2, rewrite2), "*")
            .await
            .is_empty()
    );
    let mut window: Vec<String> = (10..13)
        .map(|i| format!("{inserted2}|{i}|insert|{}", 90 + i))
        .collect();
    window.extend(later.iter().cloned());
    window.sort();
    assert_eq!(
        feed(&ctx, "table_changes", "t", (inserted2, rewrite2), all).await,
        window
    );
    assert_eq!(
        rows(&ctx, "SELECT id FROM lake.main.t").await,
        expect(&["100", "101", "102", "4", "50", "7", "8", "9"]),
    );
}

/// `table_changes_rewrite_merged_twice.test` (no inlining).
async fn crate_rewrite_merged_twice(lake_m: Lake) {
    lake_m.sql("CREATE TABLE lake.main.t (id INTEGER)").await;
    let i1 = lake_m
        .sql("INSERT INTO lake.main.t SELECT CAST(value AS INTEGER) FROM range(10)")
        .await;
    let i2 = lake_m
        .sql("INSERT INTO lake.main.t SELECT CAST(value AS INTEGER) FROM range(10, 20)")
        .await;
    let d1 = lake_m.sql("DELETE FROM lake.main.t WHERE id = 0").await;
    let r1 = lake_m.rewrite("t").await;
    let d2 = lake_m.sql("DELETE FROM lake.main.t WHERE id = 10").await;
    lake_m.rewrite("t").await;
    let merge = lake_m.merge("t").await;
    assert_eq!(lake_m.data_files("t"), 1, "the merge makes one file");
    let (ctx, _) = lake_m.session();
    assert_eq!(
        change_counts(&ctx, "t", (0, merge)).await,
        expect(&[
            &format!("{i1}|insert|10"),
            &format!("{i2}|insert|10"),
            &format!("{d1}|delete|1"),
            &format!("{d2}|delete|1"),
        ]),
    );
    assert_eq!(
        feed(&ctx, "table_insertions", "t", (r1, merge), "count(*)").await,
        expect(&["0"]),
    );
}

/// A data file retired whole (here by an overwrite) after inlined deletions of
/// its rows: the retirement deletes every other row of the file, and a window
/// holding it reports the inlined deletions it holds at their own snapshots
/// and none made before it (official's `GetTableDeletions` on its main
/// branch, duckdb/ducklake#1571).
/// (Inlining limit 5.)
async fn crate_retirement_after_inlined_deletes(lake: Lake) {
    lake.sql("CREATE TABLE lake.main.t (id INTEGER)").await;
    let inserted = lake
        .sql("INSERT INTO lake.main.t SELECT CAST(value AS INTEGER) FROM range(20)")
        .await;
    let d1 = lake.sql("DELETE FROM lake.main.t WHERE id = 0").await;
    let d2 = lake.sql("DELETE FROM lake.main.t WHERE id = 1").await;
    let replaced = lake
        .sql("INSERT OVERWRITE lake.main.t VALUES (100), (101)")
        .await;
    let (ctx, _) = lake.session();
    assert_eq!(
        deletion_counts(&ctx, "t", (replaced, replaced)).await,
        vec![format!("{replaced}|18|2|19")]
    );
    assert_eq!(
        deletion_counts(&ctx, "t", (d2, replaced)).await,
        expect(&[&format!("{d2}|1|1|1"), &format!("{replaced}|18|2|19")]),
    );
    assert_eq!(
        deletion_counts(&ctx, "t", (inserted, replaced)).await,
        expect(&[&format!("{d1}|1|0|0"), &format!("{d2}|1|1|1"), &format!("{replaced}|18|2|19"),]),
    );
    // Every row of the file is deleted once over the whole history.
    assert_eq!(
        rows(
            &ctx,
            &format!(
                "SELECT change_type, count(*), count(DISTINCT rowid) \
                 FROM ducklake_table_changes('main.t', 0, {replaced}) GROUP BY change_type"
            ),
        )
        .await,
        expect(&["delete|20|20", "insert|22|22"]),
    );
}

async fn sqlite_lake(limit: usize) -> Lake {
    use datafusion_ducklake::{SqliteMetadataProvider, SqliteMetadataWriter};
    let dir = tempfile::tempdir().unwrap();
    let data = data_dir(dir.path(), "lake");
    let url = format!("sqlite:{}?mode=rwc", dir.path().join("c.db").display());
    let writer = SqliteMetadataWriter::new_with_init(&url).await.unwrap();
    writer.set_data_path(data.to_str().unwrap()).unwrap();
    let snapshot = writer.create_snapshot().unwrap();
    writer.get_or_create_schema("main", None, snapshot).unwrap();
    Lake {
        writer: Arc::new(writer),
        provider: Arc::new(SqliteMetadataProvider::new(&url).await.unwrap()),
        limit,
        _keep: Box::new(dir),
    }
}

/// A lake in the PostgreSQL multi-catalog layout, in a container of its own.
#[cfg(feature = "write-postgres")]
async fn multicatalog_lake(limit: usize) -> Lake {
    use datafusion_ducklake::{MulticatalogManager, MulticatalogProvider, PostgresMetadataWriter};
    use testcontainers::runners::AsyncRunner;
    let container = testcontainers_modules::postgres::Postgres::default()
        .start()
        .await
        .unwrap();
    let port = container.get_host_port_ipv4(5432).await.unwrap();
    let url = format!("postgresql://postgres:postgres@127.0.0.1:{port}/postgres");
    let pool = sqlx::PgPool::connect(&url).await.unwrap();
    datafusion_ducklake::initialize_multicatalog_schema(&pool)
        .await
        .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let data = data_dir(dir.path(), "lake");
    let catalog = MulticatalogManager::new(pool.clone())
        .create_catalog("lake")
        .await
        .unwrap();
    let writer = PostgresMetadataWriter::with_pool(pool.clone(), catalog)
        .await
        .unwrap();
    writer.set_data_path(data.to_str().unwrap()).unwrap();
    let snapshot = writer.create_snapshot().unwrap();
    writer.get_or_create_schema("main", None, snapshot).unwrap();
    Lake {
        writer: Arc::new(writer),
        provider: Arc::new(MulticatalogProvider::with_pool(pool, "lake").await.unwrap()),
        limit,
        _keep: Box::new((container, dir)),
    }
}

/// One crate-written history on SQLite and on the PostgreSQL multi-catalog
/// layout.
macro_rules! crate_history {
    ($name:ident, $limit:expr) => {
        mod $name {
            use super::*;

            #[tokio::test(flavor = "multi_thread")]
            async fn sqlite() {
                super::$name(sqlite_lake($limit).await).await;
            }

            #[cfg(feature = "write-postgres")]
            #[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
            #[tokio::test(flavor = "multi_thread")]
            async fn postgres_multicatalog() {
                super::$name(multicatalog_lake($limit).await).await;
            }
        }
    };
}

crate_history!(crate_rewritten_inlined_delete, 5);
crate_history!(crate_rewritten_delete_file_and_inlined_deletes, 5);
crate_history!(crate_rewrite_data_files, 2);
crate_history!(crate_rewrite_merged_twice, 0);
crate_history!(crate_retirement_after_inlined_deletes, 5);
