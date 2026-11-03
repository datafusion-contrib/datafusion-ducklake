//! Integration tests for DuckLake `MERGE INTO` on SQLite metadata.
//!
//! The differential cases run each statement with official DuckLake and with
//! this crate on copies of one catalog official seeds, then compare what each
//! leaves: the affected-row count, every row with its `rowid` as official
//! reads it (so official must be able to read what the crate wrote), the
//! snapshot changes, the new data and delete files, and the table's row-id
//! counter.

#![cfg(all(feature = "write-sqlite", feature = "metadata-sqlite"))]

use std::sync::Arc;

use arrow::array::{Float64Array, Int32Array, StringArray, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::catalog::CatalogProvider;
use datafusion::prelude::{SessionConfig, SessionContext};
use object_store::local::LocalFileSystem;
use rstest::rstest;
use sqlx::sqlite::SqlitePool;
use tempfile::TempDir;

use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeTableWriter, MetadataWriter, SqliteMetadataProvider,
    SqliteMetadataWriter, execute_ducklake_sql,
};

fn table_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("salary", DataType::Float64, false),
    ]))
}

/// A catalog holding `main.people` (1, John, 92000) and (2, Anna, 100000),
/// written by the crate.
async fn seed_table(temp_dir: &TempDir) -> String {
    let db_path = temp_dir.path().join("test.db");
    let data_path = temp_dir.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();
    let conn_str = format!("sqlite:{}?mode=rwc", db_path.display());
    let writer = Arc::new(
        SqliteMetadataWriter::new_with_init(&conn_str)
            .await
            .unwrap(),
    );
    writer.set_data_path(data_path.to_str().unwrap()).unwrap();
    let seed = RecordBatch::try_new(
        table_schema(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2])),
            Arc::new(StringArray::from(vec!["John", "Anna"])),
            Arc::new(Float64Array::from(vec![92_000.0, 100_000.0])),
        ],
    )
    .unwrap();
    DuckLakeTableWriter::new(writer, Arc::new(LocalFileSystem::new()))
        .unwrap()
        .write_table("main", "people", &[seed])
        .await
        .unwrap();
    conn_str
}

async fn writable_catalog(
    conn_str: &str,
    config: SessionConfig,
) -> (SessionContext, Arc<DuckLakeCatalog>) {
    let writer = SqliteMetadataWriter::new(conn_str).await.unwrap();
    let provider = SqliteMetadataProvider::new(conn_str).await.unwrap();
    let catalog =
        Arc::new(DuckLakeCatalog::with_writer(Arc::new(provider), Arc::new(writer)).unwrap());
    let ctx = SessionContext::new_with_config(config);
    ctx.register_catalog("lake", Arc::clone(&catalog) as Arc<dyn CatalogProvider>);
    (ctx, catalog)
}

async fn run_count(ctx: &SessionContext, catalog: &DuckLakeCatalog, sql: &str) -> u64 {
    let batches = execute_ducklake_sql(ctx, catalog, sql)
        .await
        .unwrap_or_else(|e| panic!("crate `{sql}`: {e}"))
        .collect()
        .await
        .unwrap();
    batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap()
        .value(0)
}

async fn snapshot_id(conn_str: &str) -> i64 {
    let pool = SqlitePool::connect(conn_str).await.unwrap();
    sqlx::query_scalar("SELECT COALESCE(MAX(snapshot_id), 0) FROM ducklake_snapshot")
        .fetch_one(&pool)
        .await
        .unwrap()
}

/// Each statement is refused before anything is written.
#[rstest]
#[case::two_update_or_delete_actions(
    "MERGE INTO lake.main.people AS p
     USING (VALUES (1, 'John', 105000.0)) AS s(id, name, salary)
     ON p.id = s.id
     WHEN MATCHED AND p.salary < 100000.0 THEN UPDATE
     WHEN MATCHED THEN DELETE",
    "This feature is not implemented: MERGE INTO with DuckLake only supports a single \
     UPDATE/DELETE action currently"
)]
#[case::matched_and_by_source_actions(
    "MERGE INTO lake.main.people AS p
     USING (VALUES (1, 'John', 105000.0)) AS s(id, name, salary)
     ON p.id = s.id
     WHEN MATCHED THEN DELETE
     WHEN NOT MATCHED BY SOURCE THEN DELETE",
    "This feature is not implemented: MERGE INTO with DuckLake only supports a single \
     UPDATE/DELETE action currently"
)]
#[case::bare_update_with_an_extra_source_column(
    "MERGE INTO lake.main.people AS p
     USING (VALUES (1, 'John', 105000.0, 7)) AS s(id, name, salary, extra)
     ON p.id = s.id
     WHEN MATCHED THEN UPDATE",
    "Error during planning: Data provided for UPDATE did not match column count in table - \
     expected 3 columns but got 4"
)]
#[case::bare_insert_with_an_extra_source_column(
    "MERGE INTO lake.main.people AS p
     USING (VALUES (9, 'Zoe', 1.0, 7)) AS s(id, name, salary, extra)
     ON p.id = s.id
     WHEN NOT MATCHED THEN INSERT",
    "Error during planning: table people has 3 columns but 4 values were supplied"
)]
#[case::bare_update_by_source(
    "MERGE INTO lake.main.people AS p
     USING (VALUES (1, 'John', 105000.0)) AS s(id, name, salary)
     ON p.id = s.id
     WHEN NOT MATCHED BY SOURCE THEN UPDATE",
    "Error during planning: a bare UPDATE assigns the source row's values, and WHEN NOT \
     MATCHED BY SOURCE has no source row; name the assignments with SET"
)]
#[tokio::test(flavor = "multi_thread")]
async fn merge_refuses_before_writing(#[case] sql: &str, #[case] expected: &str) {
    let temp_dir = TempDir::new().unwrap();
    let conn_str = seed_table(&temp_dir).await;
    let before = snapshot_id(&conn_str).await;
    let (ctx, catalog) = writable_catalog(&conn_str, SessionConfig::new()).await;

    let error = execute_ducklake_sql(&ctx, &catalog, sql).await.unwrap_err();

    assert_eq!(error.to_string(), expected);
    assert_eq!(snapshot_id(&conn_str).await, before);
}

/// A source value that does not convert to its column's type fails the
/// statement, as an `INSERT` of it fails, rather than writing NULL.
#[rstest]
#[case::integer_out_of_range(
    "MERGE INTO lake.main.people AS p
     USING (VALUES (3000000000, 'John', 1.0)) AS s(id, name, salary)
     ON p.id = s.id
     WHEN NOT MATCHED THEN INSERT",
    "Can't cast value 3000000000 to type Int32"
)]
#[case::text_as_a_number(
    "MERGE INTO lake.main.people AS p
     USING (VALUES (1, 'John', 'abc')) AS s(id, name, salary)
     ON p.id = s.id
     WHEN MATCHED THEN UPDATE",
    "Cannot cast string 'abc' to value of Float64 type"
)]
#[tokio::test(flavor = "multi_thread")]
async fn merge_fails_on_a_value_that_does_not_convert(#[case] sql: &str, #[case] expected: &str) {
    let temp_dir = TempDir::new().unwrap();
    let conn_str = seed_table(&temp_dir).await;
    let before = snapshot_id(&conn_str).await;
    let (ctx, catalog) = writable_catalog(&conn_str, SessionConfig::new()).await;

    let error = match execute_ducklake_sql(&ctx, &catalog, sql).await {
        Ok(frame) => frame.collect().await.unwrap_err(),
        Err(error) => error,
    };

    assert!(
        error.to_string().contains(expected),
        "unexpected error: {error}"
    );
    assert_eq!(snapshot_id(&conn_str).await, before);
}

/// Only a statement starting with `MERGE` is tokenized by the hook; any other
/// goes to DataFusion in the session's dialect.
#[tokio::test(flavor = "multi_thread")]
async fn non_merge_sql_keeps_the_session_dialect() {
    let temp_dir = TempDir::new().unwrap();
    let conn_str = seed_table(&temp_dir).await;
    let mut config = SessionConfig::new();
    config.options_mut().sql_parser.dialect = "mysql".parse().unwrap();
    let (ctx, catalog) = writable_catalog(&conn_str, config).await;

    let batches = execute_ducklake_sql(&ctx, &catalog, r"SELECT 'it\'s' AS s")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    let value = arrow::util::display::array_value_to_string(batches[0].column(0), 0).unwrap();
    assert_eq!(value, "it's");
}

#[cfg(feature = "metadata-duckdb")]
mod differential {
    use std::path::Path;

    use super::*;
    use crate::inlined_mutation_differential_tests::{
        Metadata, attach_lake, duckdb_connection, query_lines,
    };

    const TABLE: &str = "CREATE TABLE lake.main.t (id INTEGER, a INTEGER DEFAULT 7, b VARCHAR)";
    const SEED: &str = "INSERT INTO lake.main.t VALUES (1, 10, 'x'), (2, 20, 'y'), (3, 30, 'z')";
    const PARQUET: &str = "CALL lake.set_option('data_inlining_row_limit', 0)";
    const UPSERT: &str = "MERGE INTO lake.main.t USING (VALUES (1, 100, 'p'), (5, 50, 'q'))
        AS s(id, a, b) ON t.id = s.id
        WHEN MATCHED THEN UPDATE WHEN NOT MATCHED THEN INSERT";

    /// What a MERGE left in the catalog.
    #[derive(Debug, PartialEq)]
    struct Outcome {
        count: u64,
        rows: Vec<String>,
        changes: Vec<String>,
        data_files: Vec<String>,
        delete_files: Vec<String>,
        table_stats: Vec<String>,
    }

    fn outcome(metadata: &Metadata, data: &Path, head: i64, count: u64) -> Outcome {
        let lake = duckdb_connection();
        attach_lake(&lake, metadata, data, true);
        let rows = query_lines(
            &lake,
            "SELECT rowid, id, a, b FROM lake.main.t ORDER BY rowid",
        );
        let raw = duckdb_connection();
        metadata.attach_raw(&raw);
        let changes = query_lines(
            &raw,
            &format!(
                "SELECT changes_made FROM ducklake_snapshot_changes
                 WHERE snapshot_id > {head} ORDER BY snapshot_id"
            ),
        );
        let data_files = query_lines(
            &raw,
            &format!(
                "SELECT f.row_id_start, f.record_count,
                        (SELECT string_agg(v.partition_value, '/' ORDER BY v.partition_key_index)
                         FROM ducklake_file_partition_value v
                         WHERE v.data_file_id = f.data_file_id)
                 FROM ducklake_data_file f
                 WHERE f.begin_snapshot > {head}
                 ORDER BY ALL"
            ),
        );
        let delete_files = query_lines(
            &raw,
            &format!(
                "SELECT delete_count FROM ducklake_delete_file
                 WHERE begin_snapshot > {head} ORDER BY ALL"
            ),
        );
        let table_stats = query_lines(
            &raw,
            "SELECT next_row_id, record_count FROM ducklake_table_stats",
        );
        Outcome {
            count,
            rows,
            changes,
            data_files,
            delete_files,
            table_stats,
        }
    }

    /// Run `merge` with official DuckLake and with the crate on copies of a
    /// catalog the crate creates and official fills with `seed`; returns the crate's outcome, then
    /// official's.
    async fn run(seed: &[&str], merge: &str) -> (Outcome, Outcome) {
        let temp = TempDir::new().unwrap();
        let data = temp.path().join("data");
        std::fs::create_dir_all(&data).unwrap();
        let crate_path = temp.path().join("crate.sqlite");
        let official_path = temp.path().join("official.sqlite");
        let crate_metadata = Metadata::Sqlite(crate_path.clone());
        let official_metadata = Metadata::Sqlite(official_path.clone());
        let conn_str = format!("sqlite:{}?mode=rwc", crate_path.display());
        // The crate writes only catalogs it created; official seeds this one.
        let writer = SqliteMetadataWriter::new_with_init(&conn_str)
            .await
            .unwrap();
        writer.set_data_path(data.to_str().unwrap()).unwrap();
        let snapshot = writer.create_snapshot().unwrap();
        writer.get_or_create_schema("main", None, snapshot).unwrap();
        drop(writer);
        {
            let lake = duckdb_connection();
            attach_lake(&lake, &crate_metadata, &data, false);
            for statement in seed {
                lake.execute_batch(statement)
                    .unwrap_or_else(|e| panic!("seed `{statement}`: {e}"));
            }
        }
        std::fs::copy(&crate_path, &official_path).unwrap();
        let head = snapshot_id(&conn_str).await;

        let official_count = {
            let lake = duckdb_connection();
            attach_lake(&lake, &official_metadata, &data, false);
            lake.execute(merge, [])
                .unwrap_or_else(|e| panic!("official `{merge}`: {e}")) as u64
        };
        let crate_count = {
            let (ctx, catalog) = writable_catalog(&conn_str, SessionConfig::new()).await;
            run_count(&ctx, &catalog, merge).await
        };

        (
            outcome(&crate_metadata, &data, head, crate_count),
            outcome(&official_metadata, &data, head, official_count),
        )
    }

    #[rstest]
    #[case::upsert_inlined(&[TABLE, SEED], UPSERT)]
    #[case::upsert_parquet(&[PARQUET, TABLE, SEED], UPSERT)]
    #[case::insert_only(
        &[TABLE, SEED],
        "MERGE INTO lake.main.t USING (VALUES (9, 90, 'n')) AS s(id, a, b) ON t.id = s.id
         WHEN NOT MATCHED THEN INSERT"
    )]
    #[case::rows_of_one_file_updated_out_of_order(
        &[
            PARQUET,
            TABLE,
            "INSERT INTO lake.main.t VALUES (1, 10, 'x'), (2, 20, 'y'), (3, 30, 'z'), (4, 40, 'w')",
        ],
        "MERGE INTO lake.main.t USING (VALUES (4, 400), (1, 100), (3, 300)) AS s(id, a)
         ON t.id = s.id WHEN MATCHED THEN UPDATE SET a = s.a"
    )]
    #[case::bare_update_by_position(
        &[TABLE, SEED],
        "MERGE INTO lake.main.t USING (VALUES (1, 200, 'p'), (8, 800, 'e')) AS s(id, b, a)
         ON t.id = s.id WHEN MATCHED THEN UPDATE WHEN NOT MATCHED THEN INSERT"
    )]
    #[case::set_column_named_in_another_case(
        &["CREATE TABLE lake.main.t (id INTEGER, \"A\" INTEGER, b VARCHAR)", SEED],
        "MERGE INTO lake.main.t USING (VALUES (2, 21)) AS s(id, v) ON t.id = s.id
         WHEN MATCHED THEN UPDATE SET a = s.v"
    )]
    #[case::source_rows_matching_one_target_row_update(
        &[TABLE, SEED],
        "MERGE INTO lake.main.t USING (VALUES (1, 100, 'p'), (1, 200, 'q'), (9, 1, 'n'), (8, 2, 'm'))
         AS s(id, a, b) ON t.id = s.id
         WHEN MATCHED THEN UPDATE WHEN NOT MATCHED THEN INSERT"
    )]
    #[case::source_rows_matching_one_target_row_first_wins(
        &[TABLE, SEED],
        "MERGE INTO lake.main.t
         USING (VALUES (1, 1, 'a'), (2, 1, 'b'), (1, 2, 'c'), (2, 2, 'd'), (1, 3, 'e'))
         AS s(id, a, b) ON t.id = s.id WHEN MATCHED THEN UPDATE"
    )]
    #[case::source_rows_matching_one_target_row_large_source(
        &[TABLE, SEED],
        "MERGE INTO lake.main.t
         USING (SELECT (i % 3 + 1)::INTEGER AS id, i::INTEGER AS a FROM range(20000) AS r(i)) AS s
         ON t.id = s.id WHEN MATCHED THEN UPDATE SET a = s.a"
    )]
    #[case::source_rows_matching_one_target_row_delete(
        &[PARQUET, TABLE, SEED],
        "MERGE INTO lake.main.t USING (VALUES (1), (1)) AS s(id) ON t.id = s.id
         WHEN MATCHED THEN DELETE"
    )]
    #[case::not_matched_by_source_delete(
        &[TABLE, SEED],
        "MERGE INTO lake.main.t USING (VALUES (1)) AS s(id) ON t.id = s.id
         WHEN NOT MATCHED BY SOURCE AND t.id = 2 THEN DELETE"
    )]
    #[case::not_matched_by_source_update_and_insert(
        &[PARQUET, TABLE, SEED],
        "MERGE INTO lake.main.t USING (VALUES (1, 100, 'p'), (8, 80, 'e')) AS s(id, a, b)
         ON t.id = s.id
         WHEN NOT MATCHED BY SOURCE THEN UPDATE SET a = t.a + 1
         WHEN NOT MATCHED THEN INSERT"
    )]
    #[case::inserts_with_defaults_and_do_nothing(
        &[TABLE, SEED],
        "MERGE INTO lake.main.t
         USING (VALUES (4, 40, 'w'), (5, 50, 'v'), (6, 60, 'u'), (7, 70, 'r')) AS s(id, a, b)
         ON t.id = s.id
         WHEN NOT MATCHED AND s.id = 4 THEN INSERT (id, b) VALUES (s.id, s.b)
         WHEN NOT MATCHED AND s.id = 5 THEN DO NOTHING
         WHEN NOT MATCHED AND s.id = 6 THEN INSERT VALUES (s.id, DEFAULT, s.b)
         WHEN NOT MATCHED THEN INSERT"
    )]
    #[case::source_aliased_source(
        &[TABLE, SEED],
        "MERGE INTO lake.main.t USING (VALUES (9, 9, 'q')) AS source(id, a, b)
         ON t.id = source.id
         WHEN NOT MATCHED AND source.id = 8 THEN DO NOTHING WHEN NOT MATCHED THEN INSERT"
    )]
    #[case::matched_do_nothing_before_update(
        &[TABLE, SEED],
        "MERGE INTO lake.main.t USING (VALUES (1, 100, 'p'), (2, 1, 'q')) AS s(id, a, b)
         ON t.id = s.id
         WHEN MATCHED AND s.a > 50 THEN DO NOTHING WHEN MATCHED THEN UPDATE"
    )]
    #[case::delete_and_insert_inlined(
        &[TABLE, SEED],
        "MERGE INTO lake.main.t USING (VALUES (1, 100, 'p'), (5, 50, 'q')) AS s(id, a, b)
         ON t.id = s.id WHEN MATCHED THEN DELETE WHEN NOT MATCHED THEN INSERT"
    )]
    #[case::partitioned_target(
        &[PARQUET, TABLE, "ALTER TABLE lake.main.t SET PARTITIONED BY (b)", SEED],
        "MERGE INTO lake.main.t USING (VALUES (1, 100, 'y'), (5, 50, 'q')) AS s(id, a, b)
         ON t.id = s.id WHEN MATCHED THEN UPDATE WHEN NOT MATCHED THEN INSERT"
    )]
    #[case::inlined_and_parquet_rows(
        &[
            TABLE,
            "CALL lake.set_option('data_inlining_row_limit', 2)",
            SEED,
            "INSERT INTO lake.main.t VALUES (4, 40, 'w')",
        ],
        "MERGE INTO lake.main.t USING (VALUES (1, 100, 'p'), (4, 400, 'o')) AS s(id, a, b)
         ON t.id = s.id WHEN MATCHED THEN UPDATE"
    )]
    #[case::file_with_a_delete_file(
        &[
            PARQUET,
            TABLE,
            "INSERT INTO lake.main.t VALUES (1, 10, 'x'), (2, 20, 'y'), (3, 30, 'z'), (4, 40, 'w')",
            "DELETE FROM lake.main.t WHERE id = 2",
        ],
        "MERGE INTO lake.main.t USING (VALUES (3, 300, 'p'), (1, 100, 'o')) AS s(id, a, b)
         ON t.id = s.id WHEN MATCHED THEN UPDATE"
    )]
    #[case::several_files(
        &[PARQUET, TABLE, SEED, "INSERT INTO lake.main.t VALUES (4, 40, 'w'), (5, 50, 'v')"],
        "MERGE INTO lake.main.t USING (VALUES (5, 500, 'p'), (2, 200, 'o')) AS s(id, a, b)
         ON t.id = s.id WHEN MATCHED THEN UPDATE"
    )]
    #[case::null_keys(
        &[TABLE, SEED, "INSERT INTO lake.main.t VALUES (NULL, 0, 'n')"],
        "MERGE INTO lake.main.t USING (VALUES (NULL::INTEGER, 100, 'p'), (1, 11, 'q'))
         AS s(id, a, b) ON t.id = s.id
         WHEN MATCHED THEN DELETE WHEN NOT MATCHED THEN INSERT"
    )]
    #[case::no_op(
        &[TABLE, SEED],
        "MERGE INTO lake.main.t USING (VALUES (1, 100, 'p'), (5, 1, 'q')) AS s(id, a, b)
         ON t.id = s.id
         WHEN MATCHED AND s.a < 0 THEN UPDATE WHEN NOT MATCHED AND s.a < 0 THEN INSERT"
    )]
    #[case::source_of_several_batches(
        &[PARQUET, TABLE, SEED],
        "MERGE INTO lake.main.t USING (
             SELECT * FROM (VALUES (1, 100, 'p'), (6, 60, 'u')) AS u(id, a, b)
             UNION ALL
             SELECT * FROM (VALUES (2, 200, 'q'), (3, 300, 'r')) AS v(id, a, b)
         ) AS s ON t.id = s.id
         WHEN MATCHED THEN UPDATE WHEN NOT MATCHED THEN INSERT"
    )]
    #[tokio::test(flavor = "multi_thread")]
    async fn merge_matches_official_ducklake(#[case] seed: &[&str], #[case] merge: &str) {
        let (actual, expected) = run(seed, merge).await;

        assert_eq!(actual, expected);
    }

    /// Official writes inserted rows unsorted, in its join's output order, which
    /// the crate does not reproduce; the crate keeps the source order.
    #[rstest]
    #[tokio::test(flavor = "multi_thread")]
    async fn merge_inserts_into_a_sorted_table_unsorted() {
        let (actual, expected) = run(
            &[PARQUET, TABLE, "ALTER TABLE lake.main.t SET SORTED BY (id)", SEED],
            "MERGE INTO lake.main.t USING (VALUES (5), (3), (4), (1), (2)) AS v(id)
             ON t.id = v.id + 10 WHEN NOT MATCHED THEN INSERT (id) VALUES (v.id)",
        )
        .await;

        assert_eq!(
            actual.rows,
            [
                "0,1,10,x",
                "1,2,20,y",
                "2,3,30,z",
                "3,5,7,NULL",
                "4,3,7,NULL",
                "5,4,7,NULL",
                "6,1,7,NULL",
                "7,2,7,NULL",
            ]
        );
        assert_eq!(
            Outcome {
                rows: Vec::new(),
                ..actual
            },
            Outcome {
                rows: Vec::new(),
                ..expected
            }
        );
    }
}
