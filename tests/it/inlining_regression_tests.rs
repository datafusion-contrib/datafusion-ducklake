//! Regressions in the inlining abstraction's row APIs, its compaction flush,
//! change detection and the column metadata data writes leave behind, each
//! against what official DuckLake (DuckDB 1.5.5) does.

#![cfg(feature = "write-sqlite")]
#![cfg(feature = "metadata-duckdb")]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{Float64Array, Int32Array, Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::prelude::SessionContext;

use datafusion_ducklake::maintenance::ExpireCriteria;
use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeTable, DuckLakeWriteOptions, InlinedFlushPolicy, MergeOptions,
    MetadataProvider, MetadataWriter, RewriteOptions, SqliteMetadataProvider, SqliteMetadataWriter,
    execute_ducklake_sql,
};

use crate::inlined_mutation_differential_tests::{Metadata, duckdb_query, official, query_lines};
use crate::inlining_abstraction_tests::{
    Lake, batch, create, create_with, data_changes, last_rows_changed, lines, sqlite,
};

/// The SQLite connection string of a lake made by [`sqlite`].
fn sqlite_url(lake: &Lake) -> String {
    format!(
        "sqlite:{}?mode=rwc",
        lake.data.parent().unwrap().join("lake.db").display()
    )
}

async fn raw(lake: &Lake, statement: &str) {
    let pool = sqlx::SqlitePool::connect(&sqlite_url(lake)).await.unwrap();
    sqlx::query(sqlx::AssertSqlSafe(statement.to_string()))
        .execute(&pool)
        .await
        .unwrap();
    pool.close().await;
}

/// The handle of `main.<table>` from a catalog with `options`.
async fn table_with(
    lake: &Lake,
    table: &str,
    options: DuckLakeWriteOptions,
) -> (SessionContext, DuckLakeTable) {
    let ctx = SessionContext::new();
    DuckLakeCatalog::register(&ctx, "lake", lake.catalog().with_write_options(options));
    let provider = ctx
        .catalog("lake")
        .unwrap()
        .schema("main")
        .unwrap()
        .table(table)
        .await
        .unwrap()
        .unwrap();
    let table = (provider.as_ref() as &dyn std::any::Any)
        .downcast_ref::<DuckLakeTable>()
        .unwrap()
        .clone();
    (ctx, table)
}

/// The handle of `main.<table>` read at `snapshot`, writable.
async fn table_at(lake: &Lake, table: &str, snapshot: i64) -> (SessionContext, DuckLakeTable) {
    let ctx = SessionContext::new();
    let catalog = DuckLakeCatalog::with_snapshot(Arc::clone(&lake.provider), snapshot).unwrap();
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
    let table = (provider.as_ref() as &dyn std::any::Any)
        .downcast_ref::<DuckLakeTable>()
        .unwrap()
        .clone();
    (ctx, table)
}

// ---------------------------------------------------------------------------
// Compaction leaves inlined data, as official's does
// ---------------------------------------------------------------------------

/// Official's `ducklake_merge_adjacent_files` and `ducklake_rewrite_data_files`
/// never flush inlined data: by default neither does a merge or rewrite here,
/// even when the inlined rows are hours old. The policy flushes only when a
/// caller opts in. Guards against a compaction pass that flushes by default.
#[tokio::test(flavor = "multi_thread")]
async fn compaction_leaves_inlined_rows_by_default() {
    let lake = sqlite()(Some(5)).await;
    create(&lake, "t").await;
    lake.append("t", &batch(1..=3, "a")).await;
    lake.append_parquet("t", &batch(10..=12, "p")).await;
    lake.append_parquet("t", &batch(13..=15, "p")).await;
    // The inlined rows are two hours old: past the opt-in policy's age trigger.
    raw(
        &lake,
        "UPDATE ducklake_snapshot SET snapshot_time = datetime(snapshot_time, '-2 hours')",
    )
    .await;
    let rows = lake.rows("t").await;

    let (ctx, table) = lake.table("t").await;
    let merged = table
        .merge_adjacent_files(&ctx.state(), MergeOptions::default())
        .await
        .unwrap();
    assert_eq!(merged.inlined_flush_snapshot, None, "{merged:?}");
    assert_eq!(lake.inlined_rows("t"), 3, "a merge leaves inlined rows");
    let (ctx, table) = lake.table("t").await;
    let rewritten = table
        .rewrite_data_files(
            &ctx.state(),
            RewriteOptions {
                delete_threshold: 0.0,
                data_file_ids: None,
            },
        )
        .await
        .unwrap();
    assert_eq!(rewritten.inlined_flush_snapshot, None, "{rewritten:?}");
    assert!(!rewritten.did_work(), "nothing to rewrite: {rewritten:?}");
    assert_eq!(lake.inlined_rows("t"), 3, "a rewrite leaves inlined rows");
    assert_eq!(lake.rows("t").await, rows);

    // Opting in flushes the hour-old rows first.
    let (ctx, table) = table_with(
        &lake,
        "t",
        DuckLakeWriteOptions::default().with_inlined_flush_policy(InlinedFlushPolicy::when_due()),
    )
    .await;
    let merged = table
        .merge_adjacent_files(&ctx.state(), MergeOptions::default())
        .await
        .unwrap();
    assert!(merged.inlined_flush_snapshot.is_some(), "{merged:?}");
    assert_eq!(lake.inlined_rows("t"), 0);
    assert_eq!(lake.rows("t").await, rows);
}

// ---------------------------------------------------------------------------
// Inlined tables read with their registered schema version's columns
// ---------------------------------------------------------------------------

/// An official catalog whose inlined table was registered before a column
/// rename that made no new one (the table's limit was `0` then), and whose
/// only row was inserted after it: official binds the table's columns at its
/// registered schema version, where the column is still `a`, and reads `y`.
/// Guards against picking the version from the oldest row's snapshot, where
/// the column is `b`, which the inlined table does not hold.
#[tokio::test(flavor = "multi_thread")]
async fn inlined_table_reads_with_its_registered_version() {
    let temp = tempfile::TempDir::new().unwrap();
    let catalog = temp.path().join("c.ducklake");
    let data = temp.path().join("data");
    std::fs::create_dir_all(&data).unwrap();
    let metadata = Metadata::Duckdb(catalog.clone());
    official(
        &metadata,
        &data,
        &[
            "CREATE TABLE lake.t(id INTEGER, a VARCHAR)".to_string(),
            "CALL lake.set_option('data_inlining_row_limit', 0, table_name => 't')".to_string(),
            "ALTER TABLE lake.t RENAME a TO b".to_string(),
            "CALL lake.set_option('data_inlining_row_limit', 10, table_name => 't')".to_string(),
            "INSERT INTO lake.t VALUES (2, 'y')".to_string(),
        ],
    );
    assert_eq!(
        duckdb_query(&metadata, &data, "SELECT * FROM lake.t"),
        vec!["2,y"]
    );
    let provider =
        datafusion_ducklake::DuckdbMetadataProvider::new(catalog.to_str().unwrap()).unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("lake", Arc::new(DuckLakeCatalog::new(provider).unwrap()));
    let batches = ctx
        .sql("SELECT * FROM lake.main.t")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(lines(&batches, false), vec!["2,y"]);
}

// ---------------------------------------------------------------------------
// Data writes on an official table with a struct field added
// ---------------------------------------------------------------------------

/// The `ducklake_column` rows of table `t`, every version.
fn column_rows(metadata: &Metadata) -> Vec<String> {
    let connection = crate::inlined_mutation_differential_tests::duckdb_connection();
    metadata.attach_raw(&connection);
    query_lines(
        &connection,
        "SELECT column_id, begin_snapshot, end_snapshot, column_order, column_name, \
                column_type, nulls_allowed, parent_column, initial_default, default_value \
         FROM ducklake_column ORDER BY column_id, begin_snapshot",
    )
}

/// An official SQLite catalog with `t(id, s STRUCT(a), z)` whose struct got a
/// field `c` after `z` existed, so `c` is numbered after `z`.
fn official_struct_table(temp: &Path) -> (Metadata, PathBuf) {
    let path = temp.join("c.db");
    let data = temp.join("data");
    std::fs::create_dir_all(&data).unwrap();
    let metadata = Metadata::Sqlite(path);
    official(
        &metadata,
        &data,
        &[
            "CREATE TABLE lake.t(id INTEGER, s STRUCT(a INTEGER), z INTEGER)".to_string(),
            "INSERT INTO lake.t SELECT i, {'a': i}, i FROM range(1, 31) r(i)".to_string(),
            "INSERT INTO lake.t VALUES (100, {'a': 100}, 100)".to_string(),
            "ALTER TABLE lake.t ADD COLUMN s.c INTEGER".to_string(),
        ],
    );
    (metadata, data)
}

/// A crate writer and catalog over an official SQLite catalog.
async fn crate_side(
    metadata: &Metadata,
    data: &Path,
) -> (Arc<dyn MetadataWriter>, Arc<dyn MetadataProvider>) {
    let Metadata::Sqlite(path) = metadata else {
        unreachable!()
    };
    let url = format!("sqlite:{}?mode=rwc", path.display());
    // The catalog keeps official's data path.
    let _ = data;
    let writer = SqliteMetadataWriter::new_with_init(&url).await.unwrap();
    let provider = SqliteMetadataProvider::new(&url).await.unwrap();
    (Arc::new(writer), Arc::new(provider))
}

async fn crate_sql(
    writer: &Arc<dyn MetadataWriter>,
    provider: &Arc<dyn MetadataProvider>,
    sql: &str,
) {
    let catalog =
        || DuckLakeCatalog::with_writer(Arc::clone(provider), Arc::clone(writer)).unwrap();
    let ctx = SessionContext::new();
    DuckLakeCatalog::register(&ctx, "lake", catalog());
    execute_ducklake_sql(&ctx, &catalog(), sql)
        .await
        .unwrap_or_else(|e| panic!("`{sql}`: {e}"))
        .collect()
        .await
        .unwrap_or_else(|e| panic!("`{sql}`: {e}"));
}

/// `DELETE`, `UPDATE`, `INSERT` and a keyed upsert commit on an official
/// table whose struct field was added after a later column (old bug: the
/// staged-schema check compared columns by position and refused them with
/// "table schema changed after staging"), and leave its `ducklake_column`
/// rows as official wrote them, as official's data writes do (old bug: a data
/// write renumbered `column_order` in place).
#[tokio::test(flavor = "multi_thread")]
async fn data_writes_keep_official_column_metadata() {
    let temp = tempfile::TempDir::new().unwrap();
    let (metadata, data) = official_struct_table(temp.path());
    let before = column_rows(&metadata);
    let (writer, provider) = crate_side(&metadata, &data).await;

    crate_sql(
        &writer,
        &provider,
        "DELETE FROM lake.main.t WHERE id IN (2, 100)",
    )
    .await;
    crate_sql(&writer, &provider, "DELETE FROM lake.main.t WHERE id > 20").await;
    crate_sql(
        &writer,
        &provider,
        "UPDATE lake.main.t SET z = z + 1000 WHERE id = 3",
    )
    .await;
    crate_sql(
        &writer,
        &provider,
        "INSERT INTO lake.main.t SELECT 200, named_struct('a', 7, 'c', 8), 9",
    )
    .await;

    let catalog = DuckLakeCatalog::with_writer(Arc::clone(&provider), Arc::clone(&writer)).unwrap();
    let ctx = SessionContext::new();
    DuckLakeCatalog::register(&ctx, "lake", catalog);
    let handle = ctx
        .catalog("lake")
        .unwrap()
        .schema("main")
        .unwrap()
        .table("t")
        .await
        .unwrap()
        .unwrap();
    let table = (handle.as_ref() as &dyn std::any::Any)
        .downcast_ref::<DuckLakeTable>()
        .unwrap()
        .clone();
    let staged = ctx
        .sql("SELECT id, s, CAST(z + 1 AS INT) AS z FROM lake.main.t WHERE id IN (4, 5)")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    table
        .keyed_mutation(&["id"])
        .upsert_by_keys(&ctx.state(), staged)
        .await
        .unwrap();

    assert_eq!(
        column_rows(&metadata),
        before,
        "data writes leave ducklake_column"
    );
    let read = ctx
        .sql("SELECT id, s['a'], s['c'], z FROM lake.main.t ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let read = lines(&read, false);
    assert_eq!(
        read,
        vec![
            "1,1,NULL,1",
            "3,3,NULL,1003",
            "4,4,NULL,5",
            "5,5,NULL,6",
            "6,6,NULL,6",
            "7,7,NULL,7",
            "8,8,NULL,8",
            "9,9,NULL,9",
            "10,10,NULL,10",
            "11,11,NULL,11",
            "12,12,NULL,12",
            "13,13,NULL,13",
            "14,14,NULL,14",
            "15,15,NULL,15",
            "16,16,NULL,16",
            "17,17,NULL,17",
            "18,18,NULL,18",
            "19,19,NULL,19",
            "20,20,NULL,20",
            "200,7,8,9",
        ]
    );
}

// ---------------------------------------------------------------------------
// Keyed mutations
// ---------------------------------------------------------------------------

fn vid_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("v", DataType::Utf8, true),
        Field::new("id", DataType::Int64, true),
    ]))
}

/// Rows `(v, id)`.
fn vid(ids: impl IntoIterator<Item = i64>, prefix: &str) -> RecordBatch {
    let ids: Vec<i64> = ids.into_iter().collect();
    let values: Vec<String> = ids.iter().map(|id| format!("{prefix}{id}")).collect();
    RecordBatch::try_new(
        vid_schema(),
        vec![Arc::new(StringArray::from(values)), Arc::new(Int64Array::from(ids))],
    )
    .unwrap()
}

/// `(id, v)` columns of `batch`, reordered.
fn idv(batch: &RecordBatch) -> RecordBatch {
    batch.project(&[1, 0]).unwrap()
}

/// `id -> v` of the table's rows.
async fn by_id(lake: &Lake, table: &str) -> BTreeMap<i64, String> {
    lake.rows(table)
        .await
        .iter()
        .map(|row| {
            let cells: Vec<&str> = row.split(',').collect();
            (cells[2].parse().unwrap(), cells[1].to_string())
        })
        .collect()
}

/// Key columns are found by name, never by position: a `(v, id)` table keyed
/// on `id` takes a batch of the key alone, and every operation takes its
/// columns in any order. Guards against reading the key at the table's
/// position of the key column in a staged batch.
#[tokio::test(flavor = "multi_thread")]
async fn keyed_mutations_find_keys_by_name() {
    let lake = sqlite()(Some(5)).await;
    create_with(
        &lake,
        "t",
        &[("v", DataType::Utf8), ("id", DataType::Int64)],
    )
    .unwrap();
    lake.append_parquet("t", &vid(1..=10, "a")).await;
    lake.append("t", &vid(20..=22, "i")).await;

    let keys = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, true)])),
        vec![Arc::new(Int64Array::from(vec![2, 21]))],
    )
    .unwrap();
    let (ctx, table) = lake.table("t").await;
    let deleted = table
        .keyed_mutation(&["id"])
        .delete_by_keys(&ctx.state(), vec![keys])
        .await
        .unwrap();
    assert_eq!(deleted.rows_deleted, 2);

    let (ctx, table) = lake.table("t").await;
    let deleted = table
        .keyed_mutation(&["id"])
        .delete_by_keys(&ctx.state(), vec![idv(&vid([3, 22], ""))])
        .await
        .unwrap();
    assert_eq!(deleted.rows_deleted, 2, "a reordered batch deletes by name");

    let (ctx, table) = lake.table("t").await;
    let updated = table
        .keyed_mutation(&["id"])
        .update_by_keys(&ctx.state(), vec![idv(&vid([4, 20], "u"))])
        .await
        .unwrap();
    assert_eq!(updated.rows_updated, 2);
    let (ctx, table) = lake.table("t").await;
    let upserted = table
        .keyed_mutation(&["id"])
        .upsert_by_keys(&ctx.state(), vec![idv(&vid([5, 30], "w"))])
        .await
        .unwrap();
    assert_eq!((upserted.rows_deleted, upserted.rows_inserted), (1, 2));

    let rows = by_id(&lake, "t").await;
    let mut expected: BTreeMap<i64, String> = (1..=10).map(|id| (id, format!("a{id}"))).collect();
    expected.remove(&2);
    expected.remove(&3);
    expected.insert(4, "u4".to_string());
    expected.insert(5, "w5".to_string());
    expected.insert(20, "u20".to_string());
    expected.insert(30, "w30".to_string());
    assert_eq!(rows, expected);
}

fn int_rows(ids: &[i32], values: &[i32]) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, true),
            Field::new("v", DataType::Int32, true),
        ])),
        vec![Arc::new(Int32Array::from(ids.to_vec())), Arc::new(Int32Array::from(values.to_vec()))],
    )
    .unwrap()
}

fn float_rows(ids: &[f64], values: &[f64]) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Float64, true),
            Field::new("v", DataType::Float64, true),
        ])),
        vec![
            Arc::new(Float64Array::from(ids.to_vec())),
            Arc::new(Float64Array::from(values.to_vec())),
        ],
    )
    .unwrap()
}

fn bigint_rows(ids: &[i64], values: &[i64]) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, true),
            Field::new("v", DataType::Int64, true),
        ])),
        vec![Arc::new(Int64Array::from(ids.to_vec())), Arc::new(Int64Array::from(values.to_vec()))],
    )
    .unwrap()
}

/// `(id, v)` of an `INTEGER` table, sorted.
async fn int_table(lake: &Lake) -> Vec<String> {
    let mut rows: Vec<String> = lake
        .rows("t")
        .await
        .iter()
        .map(|row| row.split_once(',').unwrap().1.to_string())
        .collect();
    rows.sort();
    rows
}

/// The strict staged-type contract: a key or value of a type that does not
/// widen to the table column's without loss (a wider or float integer, a
/// string, a decimal, for an `INTEGER` column) fails every keyed mutation
/// with one typed error naming the column and both types, before anything is
/// written; nothing commits. Guards against converting staged values (lossy
/// Arrow casts, or DuckDB's text casts) instead of refusing them.
#[tokio::test(flavor = "multi_thread")]
async fn keyed_mutations_refuse_lossy_staged_types() {
    use arrow::array::{ArrayRef, Decimal128Array};
    let lake = int_lake().await;
    let head = lake.head();
    let int32 = |values: Vec<i32>| Arc::new(Int32Array::from(values)) as ArrayRef;
    let cases: Vec<(&str, RecordBatch, &str)> = vec![
        (
            "a BIGINT key",
            bigint_rows(&[1], &[1]),
            "column `id`: expected Int32, got Int64",
        ),
        (
            "a DOUBLE key",
            float_rows(&[1.0], &[1.0]),
            "column `id`: expected Int32, got Float64",
        ),
        (
            "a VARCHAR key",
            named(vec![
                ("id", Arc::new(StringArray::from(vec!["1"]))),
                ("v", int32(vec![1])),
            ]),
            "column `id`: expected Int32, got Utf8",
        ),
        (
            "a DECIMAL key",
            named(vec![
                (
                    "id",
                    Arc::new(
                        Decimal128Array::from(vec![10])
                            .with_precision_and_scale(3, 1)
                            .unwrap(),
                    ),
                ),
                ("v", int32(vec![1])),
            ]),
            "column `id`: expected Int32, got Decimal128(3, 1)",
        ),
        (
            "a BIGINT value",
            named(vec![
                ("id", int32(vec![1])),
                ("v", Arc::new(Int64Array::from(vec![1]))),
            ]),
            "column `v`: expected Int32, got Int64",
        ),
        (
            "a VARCHAR value",
            named(vec![
                ("id", int32(vec![1])),
                ("v", Arc::new(StringArray::from(vec!["7"]))),
            ]),
            "column `v`: expected Int32, got Utf8",
        ),
    ];
    for (what, rows, expected) in cases {
        let value_only = expected.starts_with("column `v`");
        for operation in ["delete", "update", "upsert"] {
            if value_only && operation == "delete" {
                // A delete reads the key columns alone.
                continue;
            }
            let (ctx, table) = lake.table("t").await;
            let mutation = table.keyed_mutation(&["id"]);
            // A batch of the right types first: refusing the second must leave
            // the first unwritten too.
            let batches = vec![int_rows(&[2], &[22]), rows.clone()];
            let refused = match operation {
                "delete" => mutation.delete_by_keys(&ctx.state(), batches).await,
                "update" => mutation.update_by_keys(&ctx.state(), batches).await,
                _ => mutation.upsert_by_keys(&ctx.state(), batches).await,
            };
            match refused {
                Err(datafusion_ducklake::DuckLakeError::InvalidConfig(message)) => {
                    assert_eq!(message, expected, "{operation} of {what}");
                },
                other => panic!("{operation} of {what}: {other:?}"),
            }
            assert_eq!(lake.head(), head, "{operation} of {what} committed");
        }
    }
    assert_eq!(
        int_table(&lake).await,
        vec!["0,0", "1,10", "2,20", "3,30", "4,40"]
    );
}

/// Staged columns of a type that widens to the table column's without loss
/// are accepted by every keyed mutation, keys compared in the table's type:
/// `SMALLINT` keys and `TINYINT` values for `INTEGER` columns, a
/// `Utf8View`/`LargeUtf8` string for a `VARCHAR` key, a `DECIMAL(3,1)` for a
/// `DECIMAL(10,2)` and a `FLOAT` for a `DOUBLE` key (`-0.0` equal to `0`,
/// NaN to NaN, as DuckDB compares them).
#[tokio::test(flavor = "multi_thread")]
async fn keyed_mutations_accept_lossless_widenings() {
    use arrow::array::{
        ArrayRef, Decimal128Array, Float32Array, Int8Array, Int16Array, LargeStringArray,
        StringViewArray,
    };
    let lake = int_lake().await;
    let narrow = |ids: Vec<i16>, values: Vec<i8>| {
        named(vec![
            ("id", Arc::new(Int16Array::from(ids)) as ArrayRef),
            ("v", Arc::new(Int8Array::from(values))),
        ])
    };
    let (ctx, table) = lake.table("t").await;
    let updated = table
        .keyed_mutation(&["id"])
        .update_by_keys(&ctx.state(), vec![narrow(vec![1, 9], vec![11, 99])])
        .await
        .unwrap();
    assert_eq!(updated.rows_updated, 1);
    let (ctx, table) = lake.table("t").await;
    let upserted = table
        .keyed_mutation(&["id"])
        .upsert_by_keys(&ctx.state(), vec![narrow(vec![2, 5], vec![-2, 50])])
        .await
        .unwrap();
    assert_eq!((upserted.rows_deleted, upserted.rows_inserted), (1, 2));
    let (ctx, table) = lake.table("t").await;
    let deleted = table
        .keyed_mutation(&["id"])
        .delete_by_keys(
            &ctx.state(),
            vec![named(vec![("id", Arc::new(Int16Array::from(vec![3, 4])))])],
        )
        .await
        .unwrap();
    assert_eq!(deleted.rows_deleted, 2);
    assert_eq!(int_table(&lake).await, vec!["0,0", "1,11", "2,-2", "5,50"]);

    // String views and large strings for a VARCHAR key.
    let lake = sqlite()(Some(0)).await;
    create_with(&lake, "s", &[("k", DataType::Utf8), ("v", DataType::Utf8)]).unwrap();
    lake.append_parquet(
        "s",
        &named(vec![
            ("k", Arc::new(StringArray::from(vec!["a", "b", "c"]))),
            ("v", Arc::new(StringArray::from(vec!["1", "2", "3"]))),
        ]),
    )
    .await;
    let (ctx, table) = lake.table("s").await;
    let updated = table
        .keyed_mutation(&["k"])
        .update_by_keys(
            &ctx.state(),
            vec![named(vec![
                ("k", Arc::new(StringViewArray::from(vec!["b"])) as ArrayRef),
                ("v", Arc::new(LargeStringArray::from(vec!["two"]))),
            ])],
        )
        .await
        .unwrap();
    assert_eq!(updated.rows_updated, 1);

    // A narrower decimal and a FLOAT key for a DOUBLE column.
    let lake = sqlite()(Some(0)).await;
    create_with(
        &lake,
        "d",
        &[("f", DataType::Float64), ("m", DataType::Decimal128(10, 2))],
    )
    .unwrap();
    let decimals = |values: Vec<i128>, precision: u8, scale: i8| {
        Arc::new(
            Decimal128Array::from(values)
                .with_precision_and_scale(precision, scale)
                .unwrap(),
        ) as ArrayRef
    };
    lake.append_parquet(
        "d",
        &named(vec![
            ("f", Arc::new(Float64Array::from(vec![0.0, f64::NAN, 1.5]))),
            ("m", decimals(vec![100, 200, 300], 10, 2)),
        ]),
    )
    .await;
    let (ctx, table) = lake.table("d").await;
    let updated = table
        .keyed_mutation(&["f"])
        .update_by_keys(
            &ctx.state(),
            vec![named(vec![
                (
                    "f",
                    Arc::new(Float32Array::from(vec![-0.0, f32::NAN])) as ArrayRef,
                ),
                ("m", decimals(vec![15, 25], 3, 1)),
            ])],
        )
        .await
        .unwrap();
    assert_eq!(updated.rows_updated, 2, "-0.0 = 0 and NaN = NaN");
    let mut values: Vec<String> = lake
        .rows("d")
        .await
        .iter()
        .map(|row| row.split(',').nth(2).unwrap().to_string())
        .collect();
    values.sort();
    assert_eq!(values, vec!["1.50", "2.50", "3.00"]);
}

/// An upsert's new rows take the table's `SET SORTED BY` order, inlined or in
/// Parquet, as official's insert sorts them. Guards against appending them in
/// staged order.
#[tokio::test(flavor = "multi_thread")]
async fn upsert_sorts_new_rows() {
    for (limit, rows) in [(5, 4), (2, 4)] {
        let lake = sqlite()(Some(limit)).await;
        create(&lake, "t").await;
        lake.sql("ALTER TABLE lake.main.t SET SORTED BY (v DESC)")
            .await
            .unwrap();
        let (ctx, table) = lake.table("t").await;
        table
            .keyed_mutation(&["id"])
            .upsert_by_keys(&ctx.state(), vec![batch(1..=rows, "v")])
            .await
            .unwrap();
        assert_eq!(lake.inlined_rows("t") > 0, rows <= limit as i64);
        let in_rowid_order: Vec<String> = lake
            .rows("t")
            .await
            .iter()
            .map(|row| row.split(',').nth(2).unwrap().to_string())
            .collect();
        assert_eq!(
            in_rowid_order,
            vec!["v4", "v3", "v2", "v1"],
            "limit {limit}: rows written in sort order"
        );
    }
}

// ---------------------------------------------------------------------------
// rowid lookup at an earlier snapshot
// ---------------------------------------------------------------------------

/// A merge of adjacent files keeps positional row ids and records the
/// snapshot of each row; read at a snapshot between the two inserts, the
/// merged file holds rows not inserted yet. `locate_rowids` and
/// `read_rows_by_rowid` leave them out, as the scan does (`delete_rowids`
/// locates rows with `locate_rowids`). Guards against mapping
/// `rowid - row_id_start` to a row at any snapshot.
#[tokio::test(flavor = "multi_thread")]
async fn rowid_lookup_at_an_earlier_snapshot_skips_later_rows() {
    let lake = sqlite()(Some(0)).await;
    create(&lake, "t").await;
    lake.append_parquet("t", &batch(1..=3, "a")).await;
    let first = lake.head();
    lake.append_parquet("t", &batch(4..=6, "b")).await;
    let (ctx, table) = lake.table("t").await;
    let merged = table
        .merge_adjacent_files(&ctx.state(), MergeOptions::default())
        .await
        .unwrap();
    assert_eq!(merged.files_created, 1, "{merged:?}");

    let early = lake.rows_at("t", Some(first)).await;
    assert_eq!(early.len(), 3);
    let (ctx, table) = table_at(&lake, "t", first).await;
    let files = table.files().unwrap();
    assert!(
        files
            .iter()
            .any(|file| file.partial_max.is_some_and(|max| max > first)),
        "the merged file is visible at the first snapshot: {files:?}"
    );
    let ids: Vec<i64> = (0..6).collect();
    let mut located: Vec<i64> = table
        .locate_rowids(&ctx.state(), &ids)
        .await
        .unwrap()
        .into_iter()
        .map(|row| row.row_id)
        .collect();
    located.sort_unstable();
    let expected: Vec<i64> = early
        .iter()
        .map(|row| row.split(',').next().unwrap().parse().unwrap())
        .collect();
    assert_eq!(located, expected);
    let read = table.read_rows_by_rowid(&ctx.state(), &ids).await.unwrap();
    let mut read = lines(&read, true);
    read.sort();
    assert_eq!(read, early);
}

/// Look up only `row_id` in `main.t` of a catalog read by `provider`: where
/// it is and its row, as `id,v` lines.
async fn look_up_one(
    provider: SqliteMetadataProvider,
    row_id: i64,
) -> (usize, Vec<String>, SessionContext, DuckLakeTable) {
    let ctx = SessionContext::new();
    ctx.register_catalog("lake", Arc::new(DuckLakeCatalog::new(provider).unwrap()));
    let provider = ctx
        .catalog("lake")
        .unwrap()
        .schema("main")
        .unwrap()
        .table("t")
        .await
        .unwrap()
        .unwrap();
    let table = (provider.as_ref() as &dyn std::any::Any)
        .downcast_ref::<DuckLakeTable>()
        .unwrap()
        .clone();
    let located = table
        .locate_rowids(&ctx.state(), &[row_id])
        .await
        .unwrap()
        .len();
    let read = table
        .read_rows_by_rowid(&ctx.state(), &[row_id])
        .await
        .unwrap();
    (located, lines(&read, true), ctx, table)
}

/// An `UPDATE` writes the new row version to a file that embeds the row's
/// original `rowid` under a new, higher `row_id_start`. Looking up only that
/// low `rowid` finds the row there, in a file this crate's `UPDATE` wrote and
/// in one official's wrote, and `delete_rowids` deletes it. Guards against
/// skipping every file whose `row_id_start` exceeds the requested ids before
/// reading whether it embeds them.
#[tokio::test(flavor = "multi_thread")]
async fn rowid_lookup_finds_an_updated_row_below_the_files_row_id_start() {
    // This crate's UPDATE.
    let lake = sqlite()(Some(0)).await;
    create(&lake, "t").await;
    lake.append_parquet("t", &batch(1..=10, "a")).await;
    lake.sql("UPDATE lake.main.t SET v = 'u3' WHERE id = 3")
        .await
        .unwrap();
    let updated: Vec<String> = lake
        .rows("t")
        .await
        .into_iter()
        .filter(|row| row.ends_with(",u3"))
        .collect();
    assert_eq!(updated.len(), 1);
    let row_id: i64 = updated[0].split(',').next().unwrap().parse().unwrap();
    let (ctx, table) = lake.table("t").await;
    let starts: Vec<Option<i64>> = table
        .files()
        .unwrap()
        .iter()
        .map(|file| file.row_id_start)
        .collect();
    assert!(
        starts
            .iter()
            .any(|start| start.is_some_and(|start| start > row_id)),
        "the update's file starts past the row's id: {starts:?}"
    );
    let located = table.locate_rowids(&ctx.state(), &[row_id]).await.unwrap();
    assert_eq!(located.len(), 1, "{located:?}");
    let read = table
        .read_rows_by_rowid(&ctx.state(), &[row_id])
        .await
        .unwrap();
    assert_eq!(lines(&read, true), updated);
    let deleted = table.delete_rowids(&ctx.state(), &[row_id]).await.unwrap();
    assert_eq!(deleted.rows_deleted, 1);
    assert_eq!(lake.rows("t").await.len(), 9);

    // Official's UPDATE.
    let temp = tempfile::TempDir::new().unwrap();
    let path = temp.path().join("o.db");
    let data = temp.path().join("data");
    std::fs::create_dir_all(&data).unwrap();
    let metadata = Metadata::Sqlite(path.clone());
    official(
        &metadata,
        &data,
        &[
            "CALL lake.set_option('data_inlining_row_limit', 0)".to_string(),
            "CREATE TABLE lake.t(id BIGINT, v VARCHAR)".to_string(),
            "INSERT INTO lake.t SELECT i, 'a' || i FROM range(1, 11) r(i)".to_string(),
            "UPDATE lake.t SET v = 'u3' WHERE id = 3".to_string(),
        ],
    );
    let expected = duckdb_query(
        &metadata,
        &data,
        "SELECT rowid, id, v FROM lake.t WHERE id = 3",
    );
    assert_eq!(expected.len(), 1);
    let row_id: i64 = expected[0].split(',').next().unwrap().parse().unwrap();
    let provider = SqliteMetadataProvider::new(&format!("sqlite:{}", path.display()))
        .await
        .unwrap();
    let (located, read, _, table) = look_up_one(provider, row_id).await;
    let starts: Vec<Option<i64>> = table
        .files()
        .unwrap()
        .iter()
        .map(|file| file.row_id_start)
        .collect();
    assert!(
        starts
            .iter()
            .any(|start| start.is_some_and(|start| start > row_id)),
        "official's update file starts past the row's id: {starts:?}"
    );
    assert_eq!(located, 1);
    assert_eq!(read, expected);
}

// ---------------------------------------------------------------------------
// Change detection across expired snapshots
// ---------------------------------------------------------------------------

/// Snapshots expired from a window leave no ledger rows: the window's
/// history is incomplete and reads as changed, whether or not the snapshot it
/// starts after was expired too. Guards against reporting "unchanged" for a
/// window whose insert was expired.
#[tokio::test(flavor = "multi_thread")]
async fn change_detection_sees_expired_snapshots() {
    let lake = sqlite()(Some(5)).await;
    create(&lake, "t").await;
    create(&lake, "u").await;
    lake.append("t", &batch([1], "a")).await;
    let table_id = lake.table_id("t");
    let cached = lake.head();
    lake.append("t", &batch([2], "b")).await;
    let insert = lake.head();
    lake.append("u", &batch([1], "x")).await;
    let head = lake.head();
    let before = data_changes(&lake.provider, table_id, cached, head).await;
    assert!(before.rows_changed() && !before.history_incomplete);

    let writer = SqliteMetadataWriter::new(&sqlite_url(&lake)).await.unwrap();
    writer
        .expire_snapshots(ExpireCriteria::Versions(vec![insert]))
        .unwrap();
    let changes = data_changes(&lake.provider, table_id, cached, head).await;
    assert!(changes.history_incomplete, "{changes:?}");
    assert!(changes.rows_changed());
    assert!(
        last_rows_changed(&lake.provider, table_id, head)
            .await
            .is_some_and(|snapshot| snapshot > insert),
        "the last change may be anywhere up to the expired snapshot"
    );

    // Expiring the snapshot the window starts after leaves it incomplete.
    writer
        .expire_snapshots(ExpireCriteria::Versions(vec![cached]))
        .unwrap();
    lake.append("u", &batch([2], "x")).await;
    let changes = data_changes(&lake.provider, table_id, cached, lake.head()).await;
    assert!(
        changes.history_incomplete && changes.rows_changed(),
        "{changes:?}"
    );
    // A window after the gap is complete.
    let changes = data_changes(&lake.provider, table_id, head, lake.head()).await;
    assert!(!changes.rows_changed(), "{changes:?}");
}

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

/// `data_inlining_row_limit` takes what official's `ducklake_set_option`
/// accepts, a string cast to `UBIGINT` as DuckDB casts it, and stores the
/// result canonically (each pair checked with the DuckDB CLI). Guards against
/// refusing `'1.5'` (stored `2`) and storing a value every later write fails
/// to read.
#[tokio::test(flavor = "multi_thread")]
async fn inlining_limit_setting_is_cast_as_official() {
    let lake = sqlite()(None).await;
    create(&lake, "t").await;
    let table_id = lake.table_id("t");
    let schema_id = lake.schema_id();
    for bad in
        ["-1", "", "ten", "-0.4", "0o17", "1__0", "18446744073709551616", "1e20", "0x", "1 2"]
    {
        assert!(
            lake.writer
                .set_table_setting(table_id, "data_inlining_row_limit", bad)
                .is_err(),
            "{bad:?} is refused"
        );
        assert!(
            lake.writer
                .set_schema_setting(schema_id, "data_inlining_row_limit", bad)
                .is_err()
        );
        assert!(
            lake.writer
                .set_global_setting("data_inlining_row_limit", bad)
                .is_err()
        );
    }
    for (value, stored) in [
        ("1.5", 2),
        ("1e2", 100),
        ("1_000", 1000),
        ("0x10", 16),
        ("0b101", 5),
        ("-0", 0),
        (" +07 ", 7),
        ("0.5", 1),
        ("1.4", 1),
        ("12.50e-1", 1),
        ("1.5e1", 15),
    ] {
        lake.writer
            .set_table_setting(table_id, "data_inlining_row_limit", value)
            .unwrap();
        assert_eq!(
            lake.writer
                .table_data_inlining_row_limit("main", "t")
                .unwrap(),
            stored,
            "{value:?}"
        );
    }
    lake.writer
        .set_table_setting(table_id, "data_inlining_row_limit", "7")
        .unwrap();
    lake.append("t", &batch(1..=7, "a")).await;
    assert_eq!(lake.inlined_rows("t"), 7);
}

// ---------------------------------------------------------------------------
// Change detection at the end of a window
// ---------------------------------------------------------------------------

/// Snapshots expired just before the window's end leave a gap the window's
/// own ids cannot show: the history is incomplete. A window starting at
/// snapshot `0` of a catalog this crate created, which never had one, is
/// complete. Guards against reading snapshot ids only up to the window's end.
#[tokio::test(flavor = "multi_thread")]
async fn change_detection_sees_snapshots_expired_at_the_window_end() {
    let lake = sqlite()(Some(5)).await;
    create(&lake, "t").await;
    create(&lake, "u").await;
    lake.append("t", &batch([1], "a")).await;
    let table_id = lake.table_id("t");
    let cached = lake.head();
    lake.append("u", &batch([1], "x")).await;
    lake.append("t", &batch([2], "b")).await;
    let through = lake.head();
    lake.append("u", &batch([2], "x")).await;

    let writer = SqliteMetadataWriter::new(&sqlite_url(&lake)).await.unwrap();
    writer
        .expire_snapshots(ExpireCriteria::Versions(vec![through - 1, through]))
        .unwrap();
    let changes = data_changes(&lake.provider, table_id, cached, through).await;
    assert!(
        changes.history_incomplete && changes.rows_changed(),
        "{changes:?}"
    );
    assert_eq!(changes.last_rows_changed_snapshot, Some(through));
}

/// A whole-history window (`after_snapshot` negative) through a snapshot
/// whose history was all expired has no snapshot id left to read: its history
/// is incomplete and its rows may have changed anywhere up to its end. Guards
/// against reporting such a window unchanged and complete.
#[tokio::test(flavor = "multi_thread")]
async fn change_detection_of_a_wholly_expired_history_is_incomplete() {
    let lake = sqlite()(Some(5)).await;
    create(&lake, "t").await;
    lake.append("t", &batch([1], "a")).await;
    let table_id = lake.table_id("t");
    let through = lake.head();
    create(&lake, "u").await;
    let writer = SqliteMetadataWriter::new(&sqlite_url(&lake)).await.unwrap();
    writer
        .expire_snapshots(ExpireCriteria::Versions((1..=through).collect()))
        .unwrap();
    assert!(
        lake.provider
            .list_snapshots()
            .unwrap()
            .iter()
            .all(|snapshot| snapshot.snapshot_id > through)
    );
    let changes = data_changes(&lake.provider, table_id, -1, through).await;
    assert!(
        changes.history_incomplete && changes.rows_changed(),
        "{changes:?}"
    );
    assert_eq!(
        last_rows_changed(&lake.provider, table_id, through).await,
        Some(through)
    );
}

/// This crate's writers number a catalog's first snapshot `1`: a window
/// starting after snapshot `0`, which never existed, is complete. Guards
/// against reading a missing starting snapshot as an expired one.
#[tokio::test(flavor = "multi_thread")]
async fn change_detection_from_snapshot_zero_of_a_crate_catalog_is_complete() {
    let lake = sqlite()(Some(5)).await;
    create(&lake, "t").await;
    lake.append("t", &batch([1], "a")).await;
    let table_id = lake.table_id("t");
    let first = lake.provider.list_snapshots().unwrap()[0].snapshot_id;
    assert_eq!(first, 1);
    let changes = data_changes(&lake.provider, table_id, 0, lake.head()).await;
    assert!(!changes.history_incomplete, "{changes:?}");
    assert!(changes.rows_changed());
}

// ---------------------------------------------------------------------------
// Keyed mutations: duplicates and nested keys
// ---------------------------------------------------------------------------

/// A batch of named columns.
fn named(columns: Vec<(&str, arrow::array::ArrayRef)>) -> RecordBatch {
    RecordBatch::try_from_iter(columns).unwrap()
}

/// `(id INTEGER, v INTEGER)` holding `(0, 0)`, `(1, 10)` ... `(4, 40)` in
/// Parquet.
async fn int_lake() -> crate::inlining_abstraction_tests::Lake {
    let lake = sqlite()(Some(0)).await;
    create_with(
        &lake,
        "t",
        &[("id", DataType::Int32), ("v", DataType::Int32)],
    )
    .unwrap();
    lake.append_parquet("t", &int_rows(&[0, 1, 2, 3, 4], &[0, 10, 20, 30, 40]))
        .await;
    lake
}

/// Several staged rows with one key: official's `MERGE ... WHEN MATCHED THEN
/// UPDATE` accepts them and the row takes the last one's values (checked with
/// the DuckDB CLI on a small source), and the row is updated once. Guards
/// against refusing duplicate keys.
#[tokio::test(flavor = "multi_thread")]
async fn keyed_update_takes_the_last_duplicate() {
    let lake = int_lake().await;
    let (ctx, table) = lake.table("t").await;
    let updated = table
        .keyed_mutation(&["id"])
        .update_by_keys(
            &ctx.state(),
            vec![int_rows(&[1, 1, 2], &[11, 12, 21]), int_rows(&[1], &[13])],
        )
        .await
        .unwrap();
    assert_eq!(updated.rows_updated, 2);
    assert_eq!(
        int_table(&lake).await,
        vec!["0,0", "1,13", "2,21", "3,30", "4,40"]
    );
}

/// A nested key column compares by value whatever field ids its fields carry:
/// a staged struct without the table's field-id metadata matches. Guards
/// against comparing nested types with their metadata.
#[tokio::test(flavor = "multi_thread")]
async fn keyed_mutations_match_nested_keys_without_field_ids() {
    let fields: arrow::datatypes::Fields = vec![Field::new("a", DataType::Int64, true)].into();
    let lake = sqlite()(Some(0)).await;
    create_with(
        &lake,
        "t",
        &[("s", DataType::Struct(fields.clone())), ("v", DataType::Utf8)],
    )
    .unwrap();
    let rows = |keys: Vec<i64>, values: Vec<&str>| {
        named(vec![
            (
                "s",
                Arc::new(arrow::array::StructArray::new(
                    fields.clone(),
                    vec![Arc::new(Int64Array::from(keys))],
                    None,
                )) as arrow::array::ArrayRef,
            ),
            ("v", Arc::new(StringArray::from(values))),
        ])
    };
    lake.append_parquet("t", &rows(vec![1, 2, 3], vec!["a", "b", "c"]))
        .await;
    // Two staged batches whose struct fields differ only in their metadata.
    let tagged: arrow::datatypes::Fields = vec![
        Field::new("a", DataType::Int64, true).with_metadata(
            [("PARQUET:field_id".to_string(), "99".to_string())]
                .into_iter()
                .collect(),
        ),
    ]
    .into();
    let tagged_rows = named(vec![
        (
            "s",
            Arc::new(arrow::array::StructArray::new(
                tagged,
                vec![Arc::new(Int64Array::from(vec![9]))],
                None,
            )) as arrow::array::ArrayRef,
        ),
        ("v", Arc::new(StringArray::from(vec!["nine"]))),
    ]);
    let (ctx, table) = lake.table("t").await;
    let updated = table
        .keyed_mutation(&["s"])
        .update_by_keys(&ctx.state(), vec![tagged_rows, rows(vec![2], vec!["B"])])
        .await
        .unwrap();
    assert_eq!(updated.rows_updated, 1);
    let (ctx, table) = lake.table("t").await;
    let deleted = table
        .keyed_mutation(&["s"])
        .delete_by_keys(&ctx.state(), vec![rows(vec![3], vec![""])])
        .await
        .unwrap();
    assert_eq!(deleted.rows_deleted, 1);
    let (ctx, table) = lake.table("t").await;
    let upserted = table
        .keyed_mutation(&["s"])
        .upsert_by_keys(&ctx.state(), vec![rows(vec![1], vec!["A"])])
        .await
        .unwrap();
    assert_eq!(upserted.rows_deleted, 1);
    let mut values: Vec<String> = lake
        .rows("t")
        .await
        .iter()
        .map(|row| row.rsplit(',').next().unwrap().to_string())
        .collect();
    values.sort();
    assert_eq!(values, vec!["A", "B"]);

    // A list key too.
    use arrow::array::ListArray;
    use arrow::datatypes::Int64Type;
    let lake = sqlite()(Some(0)).await;
    let list_type = DataType::List(Arc::new(Field::new_list_field(DataType::Int64, true)));
    create_with(&lake, "t", &[("l", list_type), ("v", DataType::Utf8)]).unwrap();
    let rows = |keys: Vec<i64>, values: Vec<&str>| {
        named(vec![
            (
                "l",
                Arc::new(ListArray::from_iter_primitive::<Int64Type, _, _>(
                    keys.into_iter().map(|key| Some(vec![Some(key)])),
                )) as arrow::array::ArrayRef,
            ),
            ("v", Arc::new(StringArray::from(values))),
        ])
    };
    lake.append_parquet("t", &rows(vec![1, 2, 3], vec!["a", "b", "c"]))
        .await;
    let (ctx, table) = lake.table("t").await;
    let updated = table
        .keyed_mutation(&["l"])
        .update_by_keys(&ctx.state(), vec![rows(vec![2], vec!["B"])])
        .await
        .unwrap();
    assert_eq!(updated.rows_updated, 1);
    let (ctx, table) = lake.table("t").await;
    let deleted = table
        .keyed_mutation(&["l"])
        .delete_by_keys(&ctx.state(), vec![rows(vec![3], vec![""])])
        .await
        .unwrap();
    assert_eq!(deleted.rows_deleted, 1);
}

/// A NULL staged for a `NOT NULL` column fails a keyed upsert or update with
/// the crate's typed error, naming the column, as every other write path
/// does. Guards against an Arrow error from building the table's schema first.
#[tokio::test(flavor = "multi_thread")]
async fn keyed_mutations_refuse_nulls_with_a_typed_error() {
    use datafusion_ducklake::WriteMode;
    use datafusion_ducklake::metadata_writer::ColumnDef;
    let lake = sqlite()(Some(5)).await;
    let columns = vec![
        ColumnDef::from_arrow("id", &DataType::Int64, true).unwrap(),
        ColumnDef::from_arrow("v", &DataType::Utf8, false).unwrap(),
    ];
    let setup = lake
        .writer
        .begin_write_transaction("main", "t", &columns, WriteMode::Replace)
        .unwrap();
    lake.writer
        .publish_snapshot(
            setup.table_id,
            "main",
            "t",
            setup.snapshot_id,
            WriteMode::Replace,
            setup.base_snapshot_id,
            &columns,
            &setup.field_ids,
        )
        .unwrap();
    let not_null = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, true),
        Field::new("v", DataType::Utf8, false),
    ]));
    lake.append(
        "t",
        &RecordBatch::try_new(
            not_null,
            vec![
                Arc::new(Int64Array::from(vec![1, 2])),
                Arc::new(StringArray::from(vec!["a", "b"])),
            ],
        )
        .unwrap(),
    )
    .await;
    let nulls = named(vec![
        ("id", Arc::new(Int64Array::from(vec![1, 7]))),
        ("v", Arc::new(StringArray::from(vec![None::<&str>, None]))),
    ]);
    for update in [false, true] {
        let (ctx, table) = lake.table("t").await;
        let mutation = table.keyed_mutation(&["id"]);
        let refused = if update {
            mutation
                .update_by_keys(&ctx.state(), vec![nulls.clone()])
                .await
        } else {
            mutation
                .upsert_by_keys(&ctx.state(), vec![nulls.clone()])
                .await
        };
        match refused {
            Err(datafusion_ducklake::DuckLakeError::InvalidConfig(message)) => {
                assert!(message.contains("NOT NULL"), "{message}");
            },
            other => panic!("update {update}: {other:?}"),
        }
    }
    // A nullable staged column holding no NULL is accepted for it.
    let (ctx, table) = lake.table("t").await;
    let upserted = table
        .keyed_mutation(&["id"])
        .upsert_by_keys(
            &ctx.state(),
            vec![named(vec![
                ("id", Arc::new(Int64Array::from(vec![3]))),
                ("v", Arc::new(StringArray::from(vec![Some("c")]))),
            ])],
        )
        .await
        .unwrap();
    assert_eq!(upserted.rows_inserted, 1);
}

// ---------------------------------------------------------------------------
// Upsert order with sort_on_insert off
// ---------------------------------------------------------------------------

/// With `sort_on_insert` off, official's insert leaves inlined rows in their
/// order, sorts rows written to Parquet while inlining is enabled, and leaves
/// them when it is not (checked with the DuckDB CLI); an upsert's new rows
/// follow the same rules, from the stored setting.
#[tokio::test(flavor = "multi_thread")]
async fn upsert_without_sort_on_insert_sorts_as_official() {
    for (limit, expected) in [
        (5, ["v1", "v2", "v3", "v4"]),
        (2, ["v4", "v3", "v2", "v1"]),
        (0, ["v1", "v2", "v3", "v4"]),
    ] {
        let lake = sqlite()(Some(limit)).await;
        create(&lake, "t").await;
        lake.sql("ALTER TABLE lake.main.t SET SORTED BY (v DESC)")
            .await
            .unwrap();
        lake.writer
            .set_global_setting("sort_on_insert", "false")
            .unwrap();
        let (ctx, table) = lake.table("t").await;
        table
            .keyed_mutation(&["id"])
            .upsert_by_keys(&ctx.state(), vec![batch(1..=4, "v")])
            .await
            .unwrap();
        let in_rowid_order: Vec<String> = lake
            .rows("t")
            .await
            .iter()
            .map(|row| row.split(',').nth(2).unwrap().to_string())
            .collect();
        assert_eq!(in_rowid_order, expected, "limit {limit}");
    }
}

// ---------------------------------------------------------------------------
// rowid lookup at an earlier snapshot: embedded row ids and deletes
// ---------------------------------------------------------------------------

/// `delete_rowids` finds rows by their embedded ids in flushed files written
/// in sort order, whose positions do not follow the ids, and
/// deletes exactly them. A handle opened before rows were inserted since
/// cannot delete: its commit conflicts with the insert, as official's
/// `DELETE` does, and nothing is deleted. Guards against deleting
/// `row_id_start + position`.
#[tokio::test(flavor = "multi_thread")]
async fn delete_rowids_in_the_embedded_layout() {
    let lake = sqlite()(Some(5)).await;
    create(&lake, "t").await;
    lake.sql("ALTER TABLE lake.main.t SET SORTED BY (v DESC)")
        .await
        .unwrap();
    for (ids, prefix) in [(1..=3, "a"), (4..=6, "b")] {
        lake.append("t", &batch(ids, prefix)).await;
        let ctx = SessionContext::new();
        lake.catalog()
            .flush_inlined_data(&ctx.state(), None, None)
            .await
            .unwrap();
    }
    let (ctx, table) = lake.table("t").await;
    let files = table.files().unwrap();
    assert_eq!(files.len(), 2);
    for file in &files {
        assert_eq!(
            table.file_row_id_layout(&ctx.state(), file).await.unwrap(),
            Some(datafusion_ducklake::RowIdLayout::Embedded)
        );
    }
    let before = lake.rows("t").await;
    // Row ids 1 and 4: ids 2 and 5.
    let deleted = table
        .delete_rowids(&ctx.state(), &[1, 4, 99])
        .await
        .unwrap();
    assert_eq!(deleted.rows_deleted, 2);
    let after = lake.rows("t").await;
    let expected: Vec<String> = before
        .into_iter()
        .filter(|row| !row.starts_with("1,") && !row.starts_with("4,"))
        .collect();
    assert_eq!(after, expected);

    let (ctx, stale) = lake.table("t").await;
    lake.append("t", &batch([7], "c")).await;
    let refused = stale.delete_rowids(&ctx.state(), &[0, 6]).await;
    assert!(
        matches!(
            refused,
            Err(datafusion_ducklake::DuckLakeError::Conflict(_))
        ),
        "{refused:?}"
    );
    assert_eq!(lake.rows("t").await.len(), after.len() + 1);
}

// ---------------------------------------------------------------------------
// Inlined tables without their schema-version ledger rows
// ---------------------------------------------------------------------------

/// Official binds an inlined table at its table's first snapshot when the
/// ledger has no row for its registered schema version
/// (`GetBeginSnapshotForTable`), as on a catalog upgraded from before
/// `ducklake_schema_versions.table_id`, whose rows without it are deleted.
/// This crate reads what official reads there.
#[tokio::test(flavor = "multi_thread")]
async fn inlined_table_without_ledger_rows_reads_as_official() {
    let temp = tempfile::TempDir::new().unwrap();
    let catalog = temp.path().join("c.ducklake");
    let data = temp.path().join("data");
    std::fs::create_dir_all(&data).unwrap();
    let metadata = Metadata::Duckdb(catalog.clone());
    official(
        &metadata,
        &data,
        &[
            "CREATE TABLE lake.t(id INTEGER, a VARCHAR)".to_string(),
            "INSERT INTO lake.t VALUES (1, 'x')".to_string(),
            "ALTER TABLE lake.t RENAME a TO b".to_string(),
            "INSERT INTO lake.t VALUES (2, 'y')".to_string(),
        ],
    );
    {
        let connection = duckdb::Connection::open(&catalog).unwrap();
        connection
            .execute_batch("DELETE FROM ducklake_schema_versions")
            .unwrap();
    }
    let expected = duckdb_query(&metadata, &data, "SELECT * FROM lake.t ORDER BY id");
    assert_eq!(expected.len(), 2, "{expected:?}");
    let provider =
        datafusion_ducklake::DuckdbMetadataProvider::new(catalog.to_str().unwrap()).unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("lake", Arc::new(DuckLakeCatalog::new(provider).unwrap()));
    let batches = ctx
        .sql("SELECT * FROM lake.main.t ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(lines(&batches, false), expected);
}

// ---------------------------------------------------------------------------
// column_order
// ---------------------------------------------------------------------------

/// `(column_id, column_order, name, parent)` of `t`'s live columns.
async fn live_column_orders(
    lake: &crate::inlining_abstraction_tests::Lake,
) -> Vec<(i64, i64, String)> {
    let pool = sqlx::SqlitePool::connect(&sqlite_url(lake)).await.unwrap();
    let rows: Vec<(i64, i64, String)> = sqlx::query_as(
        "SELECT column_id, column_order, column_name FROM ducklake_column
         WHERE end_snapshot IS NULL
           AND table_id = (SELECT table_id FROM ducklake_table
                           WHERE table_name = 't' AND end_snapshot IS NULL)
         ORDER BY column_id",
    )
    .fetch_all(&pool)
    .await
    .unwrap();
    pool.close().await;
    rows
}

/// A column added after another was dropped, and a struct field added, take
/// their column id as `column_order`, as official numbers every column
/// (checked with the DuckDB CLI); the live columns keep theirs. Guards against
/// numbering a new column after the highest live order.
#[tokio::test(flavor = "multi_thread")]
async fn new_columns_take_their_id_as_order() {
    let lake = sqlite()(Some(0)).await;
    let s1: arrow::datatypes::Fields = vec![Field::new("x", DataType::Int64, true)].into();
    let s2: arrow::datatypes::Fields =
        vec![Field::new("x", DataType::Int64, true), Field::new("y", DataType::Int64, true)].into();
    create_with(
        &lake,
        "t",
        &[("a", DataType::Int64), ("s", DataType::Struct(s1.clone())), ("c", DataType::Int64)],
    )
    .unwrap();
    create_with(
        &lake,
        "t",
        &[("a", DataType::Int64), ("s", DataType::Struct(s1))],
    )
    .unwrap();
    create_with(
        &lake,
        "t",
        &[("a", DataType::Int64), ("s", DataType::Struct(s2)), ("d", DataType::Int64)],
    )
    .unwrap();
    let columns = live_column_orders(&lake).await;
    assert_eq!(columns.len(), 5, "{columns:?}");
    for (id, order, name) in &columns {
        assert_eq!(id, order, "{name}: {columns:?}");
    }
}

// ---------------------------------------------------------------------------
// PostgreSQL catalogs without the schema-version ledger
// ---------------------------------------------------------------------------

/// A PostgreSQL catalog without `ducklake_schema_versions` (made before it
/// existed) binds each inlined table at its table's first snapshot, on a
/// single-catalog and a multi-catalog store, for a scan and a flush, as the
/// SQLite and DuckDB providers do. Guards against a query naming the absent
/// ledger.
#[cfg(feature = "write-postgres")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
#[tokio::test(flavor = "multi_thread")]
async fn postgres_inlined_rows_read_without_the_ledger() {
    use datafusion_ducklake::{MulticatalogProvider, PostgresMetadataProvider};
    use testcontainers::runners::AsyncRunner;
    let container = testcontainers_modules::postgres::Postgres::default()
        .start()
        .await
        .unwrap();
    let port = container.get_host_port_ipv4(5432).await.unwrap();
    let url = format!("postgresql://postgres:postgres@127.0.0.1:{port}/postgres");
    let metadata = Metadata::Postgres(format!(
        "host=127.0.0.1 port={port} user=postgres password=postgres dbname=postgres"
    ));
    let temp = tempfile::TempDir::new().unwrap();
    let data = temp.path().join("data");
    std::fs::create_dir_all(&data).unwrap();
    official(
        &metadata,
        &data,
        &[
            "CREATE TABLE lake.t(id INTEGER, a VARCHAR)".to_string(),
            "INSERT INTO lake.t VALUES (1, 'x')".to_string(),
            "INSERT INTO lake.t VALUES (2, 'y')".to_string(),
        ],
    );
    let pool = sqlx::PgPool::connect(&url).await.unwrap();
    sqlx::query("DROP TABLE ducklake_schema_versions")
        .execute(&pool)
        .await
        .unwrap();

    async fn read(provider: Arc<dyn MetadataProvider>) -> Vec<String> {
        let ctx = SessionContext::new();
        let head = provider.get_current_snapshot().unwrap();
        ctx.register_catalog(
            "lake",
            Arc::new(DuckLakeCatalog::with_snapshot(provider, head).unwrap()),
        );
        let batches = ctx
            .sql("SELECT * FROM lake.main.t ORDER BY id")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        lines(&batches, false)
    }
    fn flushed(provider: &dyn MetadataProvider) -> usize {
        let head = provider.get_current_snapshot().unwrap();
        let schema = provider.get_schema_by_name("main", head).unwrap().unwrap();
        let table = provider
            .get_table_by_name(schema.schema_id, "t", head)
            .unwrap()
            .unwrap();
        provider
            .get_inlined_data_for_flush(table.table_id, head)
            .unwrap()
            .iter()
            .map(|data| data.rows.batch.num_rows())
            .sum()
    }
    let provider: Arc<dyn MetadataProvider> =
        Arc::new(PostgresMetadataProvider::new(&url).await.unwrap());
    assert_eq!(read(Arc::clone(&provider)).await, vec!["1,x", "2,y"]);
    assert_eq!(flushed(provider.as_ref()), 2);

    // A multi-catalog store in its own database.
    sqlx::query("CREATE DATABASE store")
        .execute(&pool)
        .await
        .unwrap();
    let store = sqlx::PgPool::connect(&format!(
        "postgresql://postgres:postgres@127.0.0.1:{port}/store"
    ))
    .await
    .unwrap();
    datafusion_ducklake::initialize_multicatalog_schema(&store)
        .await
        .unwrap();
    let lake = crate::inlining_abstraction_tests::multicatalog(store.clone())(Some(5)).await;
    create(&lake, "t").await;
    lake.append("t", &batch(1..=2, "a")).await;
    assert_eq!(lake.inlined_rows("t"), 2);
    sqlx::query("DROP TABLE ducklake_schema_versions")
        .execute(&store)
        .await
        .unwrap();
    let provider: Arc<dyn MetadataProvider> = Arc::new(
        MulticatalogProvider::with_pool(store, "lake0")
            .await
            .unwrap(),
    );
    assert_eq!(read(Arc::clone(&provider)).await, vec!["1,a1", "2,a2"]);
    assert_eq!(flushed(provider.as_ref()), 2);
}

// ---------------------------------------------------------------------------
// Column metadata on every writer
// ---------------------------------------------------------------------------

/// A metadata writer and provider over one catalog.
type Opened = (Arc<dyn MetadataWriter>, Arc<dyn MetadataProvider>);

/// Commit `columns` as table `t`'s schema with no data, as `CREATE TABLE` does.
fn define_t(writer: &Arc<dyn MetadataWriter>, columns: &[(&str, DataType)]) {
    use datafusion_ducklake::WriteMode;
    use datafusion_ducklake::metadata_writer::ColumnDef;
    let columns: Vec<ColumnDef> = columns
        .iter()
        .map(|(name, data_type)| ColumnDef::from_arrow(*name, data_type, true).unwrap())
        .collect();
    let setup = writer
        .begin_write_transaction("main", "t", &columns, WriteMode::Replace)
        .unwrap();
    writer
        .publish_snapshot(
            setup.table_id,
            "main",
            "t",
            setup.snapshot_id,
            WriteMode::Replace,
            setup.base_snapshot_id,
            &columns,
            &setup.field_ids,
        )
        .unwrap();
}

/// The id of table `t` at the head.
fn t_id(provider: &Arc<dyn MetadataProvider>) -> i64 {
    let head = provider.get_current_snapshot().unwrap();
    let schema = provider.get_schema_by_name("main", head).unwrap().unwrap();
    provider
        .get_table_by_name(schema.schema_id, "t", head)
        .unwrap()
        .unwrap()
        .table_id
}

/// On each writer, a table shaped as official leaves one (a struct field
/// numbered after a later column, a column retyped into two versioned
/// `ducklake_column` rows): data writes (`DELETE`, `UPDATE`, `INSERT`, a
/// keyed upsert, where the writer supports each) commit and leave every
/// `ducklake_column` row as it was (old bugs: the staged-schema check
/// compared columns by position and refused them; a data write renumbered
/// `column_order`), and another type promotion, where the writer promotes,
/// records one new version of that column's row, keeping its id and
/// `column_order`. With `prebuilt`, the caller built that table.
async fn data_writes_keep_columns_on(
    metadata: &Metadata,
    data: &Path,
    prebuilt: bool,
    open: impl Fn() -> futures::future::BoxFuture<'static, Opened>,
) {
    let fields_a: arrow::datatypes::Fields = vec![Field::new("a", DataType::Int32, true)].into();
    let fields_ac: arrow::datatypes::Fields =
        vec![Field::new("a", DataType::Int32, true), Field::new("c", DataType::Int32, true)].into();
    let mutations;
    let promotes;
    {
        let (writer, provider) = open().await;
        mutations = writer.supports_inlined_row_mutations();
        promotes = !prebuilt;
        if !prebuilt {
            writer.set_data_path(data.to_str().unwrap()).unwrap();
            define_t(
                &writer,
                &[
                    ("id", DataType::Int32),
                    ("s", DataType::Struct(fields_a)),
                    ("z", DataType::Int32),
                ],
            );
            define_t(
                &writer,
                &[
                    ("id", DataType::Int32),
                    ("s", DataType::Struct(fields_ac)),
                    ("z", DataType::Int32),
                ],
            );
            for statement in [
                "INSERT INTO lake.main.t SELECT CAST(value AS INT) AS id, named_struct('a', CAST(value AS INT), 'c', CAST(NULL AS INT)) AS s, CAST(value AS INT) AS z FROM generate_series(1, 30)",
                "INSERT INTO lake.main.t SELECT 100 AS id, named_struct('a', 100, 'c', CAST(NULL AS INT)) AS s, 100 AS z",
            ] {
                crate_sql(&writer, &provider, statement).await;
            }
            writer
                .promote_column_type(t_id(&provider), "z", "int64")
                .unwrap();
        }
    }
    let before = column_rows(metadata);
    assert_eq!(before.len(), 6, "{before:?}");
    {
        let (writer, provider) = open().await;
        let mut statements = vec![
            "INSERT INTO lake.main.t SELECT 200 AS id, named_struct('a', 7, 'c', 8) AS s, 9 AS z",
        ];
        if mutations {
            statements.push("DELETE FROM lake.main.t WHERE id IN (2, 100)");
            statements.push("UPDATE lake.main.t SET z = z + 1000 WHERE id = 3");
        }
        for statement in statements {
            crate_sql(&writer, &provider, statement).await;
        }
        if mutations {
            let catalog =
                DuckLakeCatalog::with_writer(Arc::clone(&provider), Arc::clone(&writer)).unwrap();
            let ctx = SessionContext::new();
            DuckLakeCatalog::register(&ctx, "lake", catalog);
            let handle = ctx
                .catalog("lake")
                .unwrap()
                .schema("main")
                .unwrap()
                .table("t")
                .await
                .unwrap()
                .unwrap();
            let table = (handle.as_ref() as &dyn std::any::Any)
                .downcast_ref::<DuckLakeTable>()
                .unwrap()
                .clone();
            let staged = ctx
                .sql("SELECT id, s, CAST(z + 1 AS INT) AS z FROM lake.main.t WHERE id IN (4, 5)")
                .await
                .unwrap()
                .collect()
                .await
                .unwrap();
            table
                .keyed_mutation(&["id"])
                .upsert_by_keys(&ctx.state(), staged)
                .await
                .unwrap();
        }
    }
    assert_eq!(
        column_rows(metadata),
        before,
        "data writes leave ducklake_column"
    );

    if promotes {
        {
            let (writer, provider) = open().await;
            writer
                .promote_column_type(t_id(&provider), "id", "int64")
                .unwrap();
        }
        let after = column_rows(metadata);
        let added: Vec<&String> = after.iter().filter(|row| !before.contains(row)).collect();
        let ended: Vec<&String> = before.iter().filter(|row| !after.contains(row)).collect();
        assert_eq!(
            added.len(),
            2,
            "the old row ends, a new one begins: {added:?}"
        );
        assert_eq!(ended.len(), 1, "{ended:?}");
        let fields = |row: &str| {
            let cells: Vec<String> = row.split(',').map(str::to_string).collect();
            (cells[0].clone(), cells[3].clone(), cells[4].clone())
        };
        assert!(
            added.iter().all(|row| fields(row) == fields(ended[0])),
            "{added:?}"
        );
    }
    let expected = if mutations {
        "30,200"
    } else {
        "32,200"
    };
    let (writer, provider) = open().await;
    let ctx = SessionContext::new();
    DuckLakeCatalog::register(
        &ctx,
        "lake",
        DuckLakeCatalog::with_writer(provider, writer).unwrap(),
    );
    let read = ctx
        .sql("SELECT count(*), max(id) FROM lake.main.t")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(lines(&read, false), vec![expected]);
}

#[cfg(feature = "write-duckdb")]
#[tokio::test(flavor = "multi_thread")]
async fn data_writes_keep_columns_duckdb() {
    let temp = tempfile::TempDir::new().unwrap();
    let catalog = temp.path().join("c.ducklake");
    let data = temp.path().join("data");
    std::fs::create_dir_all(&data).unwrap();
    let path = catalog.to_str().unwrap().to_string();
    datafusion_ducklake::DuckdbMetadataWriter::new_with_init(path.clone()).unwrap();
    data_writes_keep_columns_on(
        &Metadata::Duckdb(catalog.clone()),
        &data,
        false,
        move || {
            let path = path.clone();
            Box::pin(async move {
                let writer = datafusion_ducklake::DuckdbMetadataWriter::new(path).unwrap();
                let provider = writer.metadata_provider();
                (
                    Arc::new(writer) as Arc<dyn MetadataWriter>,
                    Arc::new(provider) as Arc<dyn MetadataProvider>,
                )
            })
        },
    )
    .await;
}

#[cfg(feature = "write-postgres")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
#[tokio::test(flavor = "multi_thread")]
async fn data_writes_keep_columns_postgres() {
    use testcontainers::runners::AsyncRunner;
    let container = testcontainers_modules::postgres::Postgres::default()
        .start()
        .await
        .unwrap();
    let port = container.get_host_port_ipv4(5432).await.unwrap();
    let url = format!("postgresql://postgres:postgres@127.0.0.1:{port}/postgres");
    let metadata = Metadata::Postgres(format!(
        "host=127.0.0.1 port={port} user=postgres password=postgres dbname=postgres"
    ));
    let temp = tempfile::TempDir::new().unwrap();
    let data = temp.path().join("data");
    std::fs::create_dir_all(&data).unwrap();
    datafusion_ducklake::PostgresSingleCatalogMetadataWriter::new_with_init(&url)
        .await
        .unwrap();
    // The single-catalog writer promotes no type: the table's shape is built
    // as the other writers build it, and `z` retyped as official's `ALTER`
    // records it, ending its row and adding one with the same id.
    {
        let writer: Arc<dyn MetadataWriter> = Arc::new(
            datafusion_ducklake::PostgresSingleCatalogMetadataWriter::new(&url)
                .await
                .unwrap(),
        );
        let provider: Arc<dyn MetadataProvider> = Arc::new(
            datafusion_ducklake::PostgresMetadataProvider::new(&url)
                .await
                .unwrap(),
        );
        writer.set_data_path(data.to_str().unwrap()).unwrap();
        let fields_a: arrow::datatypes::Fields =
            vec![Field::new("a", DataType::Int32, true)].into();
        let fields_ac: arrow::datatypes::Fields =
            vec![Field::new("a", DataType::Int32, true), Field::new("c", DataType::Int32, true)]
                .into();
        for fields in [fields_a, fields_ac] {
            define_t(
                &writer,
                &[("id", DataType::Int32), ("s", DataType::Struct(fields)), ("z", DataType::Int32)],
            );
        }
        for statement in [
            "INSERT INTO lake.main.t SELECT CAST(value AS INT) AS id, named_struct('a', CAST(value AS INT), 'c', CAST(NULL AS INT)) AS s, CAST(value AS INT) AS z FROM generate_series(1, 30)",
            "INSERT INTO lake.main.t SELECT 100 AS id, named_struct('a', 100, 'c', CAST(NULL AS INT)) AS s, 100 AS z",
        ] {
            crate_sql(&writer, &provider, statement).await;
        }
        let pool = sqlx::PgPool::connect(&url).await.unwrap();
        let head = provider.get_current_snapshot().unwrap();
        sqlx::query(
            "INSERT INTO ducklake_column
                 (column_id, begin_snapshot, end_snapshot, table_id, column_order,
                  column_name, column_type, initial_default, default_value, nulls_allowed,
                  parent_column, default_value_type, default_value_dialect)
             SELECT column_id, $1, NULL, table_id, column_order, column_name, 'int64',
                    initial_default, default_value, nulls_allowed, parent_column,
                    default_value_type, default_value_dialect
             FROM ducklake_column WHERE column_name = 'z' AND end_snapshot IS NULL",
        )
        .bind(head)
        .execute(&pool)
        .await
        .unwrap();
        sqlx::query(
            "UPDATE ducklake_column SET end_snapshot = $1
             WHERE column_name = 'z' AND end_snapshot IS NULL AND begin_snapshot < $1",
        )
        .bind(head)
        .execute(&pool)
        .await
        .unwrap();
    }
    data_writes_keep_columns_on(&metadata, &data, true, move || {
        let url = url.clone();
        Box::pin(async move {
            let writer = datafusion_ducklake::PostgresSingleCatalogMetadataWriter::new(&url)
                .await
                .unwrap();
            let provider = datafusion_ducklake::PostgresMetadataProvider::new(&url)
                .await
                .unwrap();
            (
                Arc::new(writer) as Arc<dyn MetadataWriter>,
                Arc::new(provider) as Arc<dyn MetadataProvider>,
            )
        })
    })
    .await;
}

#[cfg(feature = "write-mysql")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
#[tokio::test(flavor = "multi_thread")]
async fn data_writes_keep_columns_mysql() {
    use testcontainers::runners::AsyncRunner;
    let container = testcontainers_modules::mysql::Mysql::default()
        .start()
        .await
        .unwrap();
    let port = container.get_host_port_ipv4(3306).await.unwrap();
    let url = format!("mysql://root@127.0.0.1:{port}/test");
    let metadata = Metadata::Mysql {
        host: "127.0.0.1".to_string(),
        port,
        database: "test".to_string(),
    };
    let temp = tempfile::TempDir::new().unwrap();
    let data = temp.path().join("data");
    std::fs::create_dir_all(&data).unwrap();
    datafusion_ducklake::MySqlMetadataWriter::new_with_init(&url)
        .await
        .unwrap();
    data_writes_keep_columns_on(&metadata, &data, false, move || {
        let url = url.clone();
        Box::pin(async move {
            let writer = datafusion_ducklake::MySqlMetadataWriter::new(&url)
                .await
                .unwrap();
            let provider = datafusion_ducklake::MySqlMetadataProvider::new(&url)
                .await
                .unwrap();
            (
                Arc::new(writer) as Arc<dyn MetadataWriter>,
                Arc::new(provider) as Arc<dyn MetadataProvider>,
            )
        })
    })
    .await;
}

// ---------------------------------------------------------------------------
// Inserts in the table's sort order
// ---------------------------------------------------------------------------

/// The paths that insert rows into a table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InsertPath {
    /// `append_table` (`write_rows`).
    Rows,
    /// A transaction's `stage_write`.
    Staged,
    /// A streaming session, finished holding its rows.
    Session,
    /// A streaming session holding its rows, finished with deletes of an
    /// earlier Parquet row.
    SessionWithDeletes,
    /// `upsert_by_keys`.
    Upsert,
    /// SQL `INSERT`.
    Sql,
    /// `write_partitioned`, on a partitioned table.
    Partitioned,
    /// SQL `INSERT`, on a partitioned table.
    SqlPartitioned,
    /// `append_table`, on a partitioned table.
    RowsPartitioned,
}

/// Rows `(id, v)` with every `v` `"p"`: one partition when partitioned by `v`.
fn one_partition(ids: &[i64]) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, true),
            Field::new("v", DataType::Utf8, true),
        ])),
        vec![
            Arc::new(Int64Array::from(ids.to_vec())),
            Arc::new(StringArray::from(vec!["p"; ids.len()])),
        ],
    )
    .unwrap()
}

/// The ids of `main.t`, in `rowid` order.
async fn ids_in_rowid_order(lake: &crate::inlining_abstraction_tests::Lake) -> Vec<i64> {
    lake.rows("t")
        .await
        .iter()
        .map(|row| row.split(',').nth(1).unwrap().parse().unwrap())
        .collect()
}

/// Insert `ids` into `main.t` through `path`, with `writer_sort_on_insert` as
/// the writer's own option when given.
async fn insert_through(
    lake: &crate::inlining_abstraction_tests::Lake,
    path: InsertPath,
    writer_sort_on_insert: Option<bool>,
    ids: &[i64],
) {
    try_insert_through(lake, path, writer_sort_on_insert, ids)
        .await
        .unwrap();
}

/// [`insert_through`], returning the write's error. No `ids` writes no row:
/// SQL inserts from an empty `SELECT`, a session finishes empty, and
/// `append_table` writes an empty batch.
async fn try_insert_through(
    lake: &crate::inlining_abstraction_tests::Lake,
    path: InsertPath,
    writer_sort_on_insert: Option<bool>,
    ids: &[i64],
) -> Result<(), String> {
    let mut options = DuckLakeWriteOptions::default();
    options.sort_on_insert = writer_sort_on_insert;
    let writer = lake.table_writer().with_options(&options);
    let rows = one_partition(ids);
    let schema = rows.schema();
    match path {
        InsertPath::Rows | InsertPath::RowsPartitioned => {
            writer
                .append_table("main", "t", std::slice::from_ref(&rows))
                .await
                .map_err(|e| e.to_string())?;
        },
        InsertPath::Staged => {
            let mut transaction = writer.transaction();
            transaction
                .stage_write(
                    "main",
                    "t",
                    &schema,
                    datafusion_ducklake::WriteMode::Append,
                    std::slice::from_ref(&rows),
                )
                .await
                .map_err(|e| e.to_string())?;
            transaction.commit().await.map_err(|e| e.to_string())?;
        },
        InsertPath::Session => {
            let mut session = writer
                .begin_write("main", "t", &schema, datafusion_ducklake::WriteMode::Append)
                .map_err(|e| e.to_string())?;
            session.write_batch(&rows).map_err(|e| e.to_string())?;
            session.finish().await.map_err(|e| e.to_string())?;
        },
        InsertPath::SessionWithDeletes => {
            let entries = crate::inlining_abstraction_tests::hand_built_entries(lake, &[100]).await;
            assert_eq!(entries.len(), 1);
            let mut session = writer
                .begin_write("main", "t", &schema, datafusion_ducklake::WriteMode::Append)
                .map_err(|e| e.to_string())?;
            session.write_batch(&rows).map_err(|e| e.to_string())?;
            session
                .finish_with_deletes(&entries)
                .await
                .map_err(|e| e.to_string())?;
        },
        InsertPath::Upsert => {
            let (ctx, table) = lake.table("t").await;
            let table = match writer_sort_on_insert {
                Some(_) => table.with_write_options(options.clone()),
                None => table,
            };
            table
                .keyed_mutation(&["id"])
                .upsert_by_keys(&ctx.state(), vec![rows.clone()])
                .await
                .map_err(|e| e.to_string())?;
        },
        InsertPath::Sql | InsertPath::SqlPartitioned => {
            if ids.is_empty() {
                return lake
                    .sql("INSERT INTO lake.main.t SELECT * FROM lake.main.t WHERE false")
                    .await;
            }
            let values = ids
                .iter()
                .map(|id| format!("({id}, 'p')"))
                .collect::<Vec<_>>()
                .join(", ");
            lake.sql(&format!("INSERT INTO lake.main.t VALUES {values}"))
                .await?;
        },
        InsertPath::Partitioned => {
            let partition_id = lake
                .writer
                .live_partition_spec(lake.table_id("t"))
                .unwrap()
                .unwrap()
                .partition_id;
            writer
                .write_partitioned(
                    "main",
                    "t",
                    &schema,
                    datafusion_ducklake::WriteMode::Append,
                    partition_id,
                    &["v".to_string()],
                    vec![(vec![Some("p".to_string())], vec![rows.clone()])],
                )
                .await
                .map_err(|e| e.to_string())?;
        },
    }
    Ok(())
}

/// Every insert path lays its rows out as official's `PlanInsert` does
/// (`ducklake_insert.cpp`) under the table's STORED `sort_on_insert`, each
/// case checked with the DuckDB CLI on a table `SET SORTED BY (k)` given
/// `(3), (1), (2)`: with `sort_on_insert` on, the rows are sorted whether
/// inlined (limit 5) or written to Parquet (limit 2, or 0); with it off,
/// inlined rows keep their order, rows written to Parquet while inlining is
/// enabled (limit 2) are sorted, and with inlining off (limit 0) nothing is.
/// The setting is stored at table scope over the opposite catalog-wide one,
/// and no writer option is passed, as official reads only the stored option
/// (`GetConfigOption`). A streaming session sorts the rows it inlines,
/// whether it finishes with deletes or not; its Parquet rows stream in the
/// caller's order. Guards against a direct writer ignoring the stored
/// setting, inlining unsorted rows, skipping the Parquet sort with
/// `sort_on_insert` off, and a direct partitioned write leaving its files
/// unsorted.
#[tokio::test(flavor = "multi_thread")]
async fn inserts_sort_as_official() {
    use InsertPath::*;
    for path in [
        Rows,
        Staged,
        Session,
        SessionWithDeletes,
        Upsert,
        Sql,
        Partitioned,
        SqlPartitioned,
        RowsPartitioned,
    ] {
        for sort_on_insert in [true, false] {
            for limit in [5usize, 2, 0] {
                if matches!(path, Session | SessionWithDeletes) && limit != 5 {
                    continue;
                }
                let case = format!("{path:?}, sort_on_insert {sort_on_insert}, limit {limit}");
                let lake = sorted_table(limit, sort_on_insert, path).await;
                insert_through(&lake, path, None, &[3, 1, 2]).await;
                assert_eq!(
                    lake.inlined_rows("t"),
                    if limit == 5 {
                        3
                    } else {
                        0
                    },
                    "{case}"
                );
                let expected = if sort_on_insert || limit == 2 {
                    vec![1, 2, 3]
                } else {
                    vec![3, 1, 2]
                };
                assert_eq!(ids_in_rowid_order(&lake).await, expected, "{case}");
            }
        }
    }
}

/// `main.t(id, v)` of a lake with a writer limit of `limit`, `SET SORTED BY
/// (id)` (and `PARTITIONED BY (v)` for a partitioned `path`), with
/// `sort_on_insert` stored at table scope over the opposite catalog-wide
/// setting. For `SessionWithDeletes` it holds one Parquet row, `100`, the
/// session's deletes remove.
async fn sorted_table(
    limit: usize,
    sort_on_insert: bool,
    path: InsertPath,
) -> crate::inlining_abstraction_tests::Lake {
    use InsertPath::*;
    let lake = sqlite()(Some(limit)).await;
    create(&lake, "t").await;
    if path == SessionWithDeletes {
        let parquet = DuckLakeWriteOptions::default().with_data_inlining_row_limit(0);
        lake.table_writer()
            .with_options(&parquet)
            .append_table("main", "t", &[one_partition(&[100])])
            .await
            .unwrap();
    }
    lake.sql("ALTER TABLE lake.main.t SET SORTED BY (id)")
        .await
        .unwrap();
    if matches!(path, Partitioned | SqlPartitioned | RowsPartitioned) {
        lake.sql("ALTER TABLE lake.main.t SET PARTITIONED BY (v)")
            .await
            .unwrap();
    }
    lake.writer
        .set_global_setting("sort_on_insert", &(!sort_on_insert).to_string())
        .unwrap();
    lake.writer
        .set_table_setting(
            lake.table_id("t"),
            "sort_on_insert",
            &sort_on_insert.to_string(),
        )
        .unwrap();
    lake
}

/// A writer's own `sort_on_insert` applies only where the catalog stores
/// none, as official has no session setting outranking the stored option:
/// with `false` stored, a writer passing `true` inlines `(3), (1), (2)` in
/// their order; with nothing stored, a writer passing `false` does too, and
/// one passing nothing sorts them.
#[tokio::test(flavor = "multi_thread")]
async fn stored_sort_on_insert_outranks_the_writers() {
    use InsertPath::*;
    for path in [Rows, Staged, Session, Upsert, Sql] {
        for (stored, option, sorted) in
            [(Some(false), Some(true), false), (None, Some(false), false), (None, None, true)]
        {
            let case = format!("{path:?}, stored {stored:?}, option {option:?}");
            let lake = sqlite()(Some(5)).await;
            create(&lake, "t").await;
            lake.sql("ALTER TABLE lake.main.t SET SORTED BY (id)")
                .await
                .unwrap();
            if let Some(stored) = stored {
                lake.writer
                    .set_schema_setting(lake.schema_id(), "sort_on_insert", &stored.to_string())
                    .unwrap();
            }
            if path == Sql {
                let mut options = DuckLakeWriteOptions::default();
                options.sort_on_insert = option;
                let ctx = SessionContext::new();
                ctx.register_catalog("lake", Arc::new(lake.catalog().with_write_options(options)));
                ctx.sql("INSERT INTO lake.main.t VALUES (3, 'p'), (1, 'p'), (2, 'p')")
                    .await
                    .unwrap()
                    .collect()
                    .await
                    .unwrap();
            } else {
                insert_through(&lake, path, option, &[3, 1, 2]).await;
            }
            assert_eq!(lake.inlined_rows("t"), 3, "{case}");
            let expected = if sorted {
                vec![1, 2, 3]
            } else {
                vec![3, 1, 2]
            };
            assert_eq!(ids_in_rowid_order(&lake).await, expected, "{case}");
        }
    }
}

/// The `id`s of each live data file of `main.t`, in file order, files by
/// `data_file_id`.
fn file_ids(lake: &crate::inlining_abstraction_tests::Lake) -> Vec<Vec<i64>> {
    let mut files = lake
        .provider
        .get_table_files_for_select(lake.table_id("t"), lake.head())
        .unwrap();
    files.sort_by_key(|file| file.data_file_id);
    files
        .iter()
        .map(|file| {
            let name = Path::new(&file.file.path).file_name().unwrap().to_owned();
            let path = find_file(&lake.data, &name).expect("data file on disk");
            let reader = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(
                std::fs::File::open(path).unwrap(),
            )
            .unwrap()
            .build()
            .unwrap();
            reader
                .flat_map(|batch| {
                    let batch = batch.unwrap();
                    let ids = batch
                        .column_by_name("id")
                        .unwrap()
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .unwrap()
                        .clone();
                    ids.values().to_vec()
                })
                .collect()
        })
        .collect()
}

fn find_file(dir: &Path, name: &std::ffi::OsStr) -> Option<PathBuf> {
    for entry in std::fs::read_dir(dir).ok()? {
        let path = entry.ok()?.path();
        if path.is_dir() {
            if let Some(found) = find_file(&path, name) {
                return Some(found);
            }
        } else if path.file_name() == Some(name) {
            return Some(path);
        }
    }
    None
}

/// Official's flush orders the rows by the table's sort order whatever
/// `sort_on_insert` says (checked with the DuckDB CLI: rows inlined unsorted
/// under `sort_on_insert` `false` flush sorted), and so do
/// `DuckLakeCatalog::flush_inlined_data` and the deprecated
/// `DuckLakeTableWriter::flush_inlined_data`. Guards against the deprecated
/// flush leaving them in insert order.
#[tokio::test(flavor = "multi_thread")]
#[allow(deprecated)]
async fn flush_sorts_whatever_sort_on_insert_says() {
    for deprecated in [false, true] {
        let lake = sqlite()(Some(5)).await;
        create(&lake, "t").await;
        lake.sql("ALTER TABLE lake.main.t SET SORTED BY (id)")
            .await
            .unwrap();
        lake.writer
            .set_table_setting(lake.table_id("t"), "sort_on_insert", "false")
            .unwrap();
        lake.sql("INSERT INTO lake.main.t VALUES (3, 'p'), (1, 'p'), (2, 'p')")
            .await
            .unwrap();
        assert_eq!(ids_in_rowid_order(&lake).await, vec![3, 1, 2]);
        if deprecated {
            let head = lake.head();
            let table_id = lake.table_id("t");
            let columns = lake.provider.get_table_structure(table_id, head).unwrap();
            let inlined = lake
                .provider
                .get_inlined_data_with_row_ids(table_id, head, &columns)
                .unwrap();
            lake.table_writer()
                .flush_inlined_data("main", "t", &inlined, head)
                .await
                .unwrap()
                .unwrap();
        } else {
            let ctx = SessionContext::new();
            lake.catalog()
                .flush_inlined_data(&ctx.state(), None, None)
                .await
                .unwrap();
        }
        assert_eq!(lake.inlined_rows("t"), 0, "deprecated {deprecated}");
        assert_eq!(
            file_ids(&lake),
            vec![vec![1, 2, 3]],
            "deprecated {deprecated}"
        );
    }
}

/// Official's `UPDATE` plans no sort (`DuckLakeCatalog::PlanUpdate`): on a
/// table `SET SORTED BY (v)` holding `(1,'e') .. (5,'a')` in one Parquet file,
/// `UPDATE u SET v = CAST(id AS VARCHAR)` rewrites the rows in the order it
/// read them, ids `[5, 4, 3, 2, 1]` (checked with the DuckDB CLI). SQL
/// `UPDATE` and `update_by_keys` do the same. Guards against sorting the new
/// versions.
#[tokio::test(flavor = "multi_thread")]
async fn updates_do_not_sort() {
    for keyed in [false, true] {
        let lake = sqlite()(Some(0)).await;
        create(&lake, "t").await;
        lake.sql("ALTER TABLE lake.main.t SET SORTED BY (v)")
            .await
            .unwrap();
        lake.sql("INSERT INTO lake.main.t VALUES (1, 'e'), (2, 'd'), (3, 'c'), (4, 'b'), (5, 'a')")
            .await
            .unwrap();
        assert_eq!(file_ids(&lake), vec![vec![5, 4, 3, 2, 1]]);
        if keyed {
            let (ctx, table) = lake.table("t").await;
            let ids = [1i64, 2, 3, 4, 5];
            let staged = RecordBatch::try_new(
                one_partition(&ids).schema(),
                vec![
                    Arc::new(Int64Array::from(ids.to_vec())),
                    Arc::new(StringArray::from(
                        ids.iter().map(ToString::to_string).collect::<Vec<_>>(),
                    )),
                ],
            )
            .unwrap();
            let result = table
                .keyed_mutation(&["id"])
                .update_by_keys(&ctx.state(), vec![staged])
                .await
                .unwrap();
            assert_eq!(result.rows_updated, 5);
        } else {
            lake.sql("UPDATE lake.main.t SET v = CAST(id AS VARCHAR)")
                .await
                .unwrap();
        }
        assert_eq!(
            file_ids(&lake).last().unwrap(),
            &vec![5, 4, 3, 2, 1],
            "keyed {keyed}"
        );
    }
}

/// A sort key that is an expression orders an insert as official's does
/// (checked with the DuckDB CLI on `SET SORTED BY ((id * 7) % 10)` given
/// `(3), (1), (2), (9), (8)`): with `sort_on_insert` on, the inlined rows are
/// `[3, 9, 2, 8, 1]`; with it off, rows written to Parquet while inlining is
/// enabled (limit 2) are too, and inlined rows keep their order. SQL `INSERT`
/// and a direct write agree. Guards against SQL refusing the expression and
/// a direct write leaving the rows unsorted.
#[tokio::test(flavor = "multi_thread")]
async fn expression_sort_keys_order_inserts() {
    for sql in [false, true] {
        for (sort_on_insert, limit, expected) in [
            (true, 5usize, vec![3, 9, 2, 8, 1]),
            (false, 2, vec![3, 9, 2, 8, 1]),
            (false, 5, vec![3, 1, 2, 9, 8]),
        ] {
            let case = format!("sql {sql}, sort_on_insert {sort_on_insert}, limit {limit}");
            let lake = sqlite()(Some(limit)).await;
            create(&lake, "t").await;
            lake.sql("ALTER TABLE lake.main.t SET SORTED BY (id)")
                .await
                .unwrap();
            raw(
                &lake,
                "UPDATE ducklake_sort_expression SET expression = '((id * 7) % 10)'",
            )
            .await;
            lake.writer
                .set_table_setting(
                    lake.table_id("t"),
                    "sort_on_insert",
                    &sort_on_insert.to_string(),
                )
                .unwrap();
            if sql {
                lake.sql("INSERT INTO lake.main.t VALUES (3, 'p'), (1, 'p'), (2, 'p'), (9, 'p'), (8, 'p')")
                    .await
                    .unwrap();
            } else {
                insert_through(&lake, InsertPath::Rows, None, &[3, 1, 2, 9, 8]).await;
            }
            assert_eq!(ids_in_rowid_order(&lake).await, expected, "{case}");
        }
    }
}

/// Flush `main.t`'s inlined rows through `DuckLakeCatalog::flush_inlined_data`,
/// or the deprecated `DuckLakeTableWriter::flush_inlined_data`.
#[allow(deprecated)]
async fn flush_t(lake: &Lake, deprecated: bool) {
    if deprecated {
        let head = lake.head();
        let table_id = lake.table_id("t");
        let columns = lake.provider.get_table_structure(table_id, head).unwrap();
        let inlined = lake
            .provider
            .get_inlined_data_with_row_ids(table_id, head, &columns)
            .unwrap();
        lake.table_writer()
            .flush_inlined_data("main", "t", &inlined, head)
            .await
            .unwrap()
            .unwrap();
    } else {
        let ctx = SessionContext::new();
        lake.catalog()
            .flush_inlined_data(&ctx.state(), None, None)
            .await
            .unwrap();
    }
}

/// `main.t(id, v)` of a lake with limit `limit`, holding `(3), (1), (2)`
/// inlined unsorted (`sort_on_insert` off), its live sort key stored as
/// `expression`.
async fn inlined_under_sort_key(limit: usize, expression: &str) -> Lake {
    let lake = sqlite()(Some(limit)).await;
    create(&lake, "t").await;
    lake.sql("ALTER TABLE lake.main.t SET SORTED BY (id)")
        .await
        .unwrap();
    lake.writer
        .set_table_setting(lake.table_id("t"), "sort_on_insert", "false")
        .unwrap();
    lake.sql("INSERT INTO lake.main.t VALUES (3, 'p'), (1, 'p'), (2, 'p')")
        .await
        .unwrap();
    assert_eq!(lake.inlined_rows("t"), 3);
    raw(
        &lake,
        &format!("UPDATE ducklake_sort_expression SET expression = '{expression}'"),
    )
    .await;
    lake
}

/// DuckDB stores `SET SORTED BY (ID)` as typed and binds it to column `id`
/// case-insensitively (checked with the DuckDB CLI: on `t(id, v)` with the key
/// `ID DESC`, `(1), (3), (2)` inserted to Parquet read `3, 2, 1`). Every insert
/// path, both flushes and compaction order by such a key, and by `id` over a
/// column named `ID`. Guards against matching sort keys case-sensitively,
/// which fails every write to a table whose sort order DuckDB wrote.
#[tokio::test(flavor = "multi_thread")]
async fn sort_keys_bind_case_insensitively() {
    use InsertPath::*;
    for path in [
        Rows,
        Staged,
        Session,
        SessionWithDeletes,
        Upsert,
        Sql,
        Partitioned,
        SqlPartitioned,
        RowsPartitioned,
    ] {
        for limit in [5usize, 0] {
            if matches!(path, Session | SessionWithDeletes) && limit != 5 {
                continue;
            }
            let case = format!("{path:?}, limit {limit}");
            let lake = sorted_table(limit, true, path).await;
            raw(
                &lake,
                "UPDATE ducklake_sort_expression SET expression = 'ID'",
            )
            .await;
            insert_through(&lake, path, None, &[3, 1, 2]).await;
            assert_eq!(ids_in_rowid_order(&lake).await, vec![1, 2, 3], "{case}");
        }
    }
    for deprecated in [false, true] {
        let lake = inlined_under_sort_key(5, "ID").await;
        flush_t(&lake, deprecated).await;
        assert_eq!(
            file_ids(&lake),
            vec![vec![1, 2, 3]],
            "flush, deprecated {deprecated}"
        );
    }
    // An expression's columns bind the same way (`((ID * 7) % 10)` orders
    // `(3), (1), (2), (9), (8)` as `3, 9, 2, 8, 1`, checked with the DuckDB CLI).
    for path in [InsertPath::Rows, InsertPath::Sql] {
        let lake = sorted_table(5, true, path).await;
        raw(
            &lake,
            "UPDATE ducklake_sort_expression SET expression = '((ID * 7) % 10)'",
        )
        .await;
        insert_through(&lake, path, None, &[3, 1, 2, 9, 8]).await;
        assert_eq!(
            ids_in_rowid_order(&lake).await,
            vec![3, 9, 2, 8, 1],
            "expression, {path:?}"
        );
    }
    let lake = inlined_under_sort_key(5, "((ID * 7) % 10)").await;
    flush_t(&lake, false).await;
    assert_eq!(file_ids(&lake), vec![vec![3, 2, 1]], "expression, flush");
    // Compaction merges two unsorted files into one in the key's order.
    let lake = sqlite()(Some(0)).await;
    create(&lake, "t").await;
    lake.append_parquet("t", &one_partition(&[3, 1])).await;
    lake.append_parquet("t", &one_partition(&[2])).await;
    lake.sql("ALTER TABLE lake.main.t SET SORTED BY (id)")
        .await
        .unwrap();
    raw(
        &lake,
        "UPDATE ducklake_sort_expression SET expression = 'ID'",
    )
    .await;
    let (ctx, table) = lake.table("t").await;
    let merged = table
        .merge_adjacent_files(&ctx.state(), MergeOptions::default())
        .await
        .unwrap();
    assert!(merged.did_work(), "{merged:?}");
    assert_eq!(file_ids(&lake), vec![vec![1, 2, 3]], "compaction");

    // The reverse: a column named `ID`, a key stored as `id`.
    for sql in [false, true] {
        let lake = sqlite()(Some(5)).await;
        create_with(
            &lake,
            "t",
            &[("ID", DataType::Int64), ("v", DataType::Utf8)],
        )
        .unwrap();
        lake.sql("ALTER TABLE lake.main.t SET SORTED BY (\"ID\")")
            .await
            .unwrap();
        raw(
            &lake,
            "UPDATE ducklake_sort_expression SET expression = 'id'",
        )
        .await;
        if sql {
            lake.sql("INSERT INTO lake.main.t VALUES (3, 'p'), (1, 'p'), (2, 'p')")
                .await
                .unwrap();
        } else {
            let rows = one_partition(&[3, 1, 2]);
            let rows = RecordBatch::try_new(
                Arc::new(Schema::new(vec![
                    Field::new("ID", DataType::Int64, true),
                    Field::new("v", DataType::Utf8, true),
                ])),
                rows.columns().to_vec(),
            )
            .unwrap();
            lake.table_writer()
                .append_table("main", "t", &[rows])
                .await
                .unwrap();
        }
        assert_eq!(
            ids_in_rowid_order(&lake).await,
            vec![1, 2, 3],
            "column ID, sql {sql}"
        );
    }
}

/// A key matching two columns that differ only in case (possible in a
/// catalog this crate wrote; DuckDB's binder refuses such a table, "duplicate
/// column name") binds to the one spelled exactly as the key, and fails the
/// write when neither is. Guards against sorting by an arbitrary one of them.
#[tokio::test(flavor = "multi_thread")]
async fn sort_key_matching_columns_by_case_alone() {
    for (key, expected) in [("ID", Ok(vec![2, 3, 1])), ("Id", Err("ambiguous"))] {
        // Limit `0`: SQLite refuses an inlined data table with both columns.
        let lake = sqlite()(Some(0)).await;
        create_with(
            &lake,
            "t",
            &[("id", DataType::Int64), ("ID", DataType::Int64)],
        )
        .unwrap();
        lake.sql("ALTER TABLE lake.main.t SET SORTED BY (id)")
            .await
            .unwrap();
        raw(
            &lake,
            &format!("UPDATE ducklake_sort_expression SET expression = '{key}'"),
        )
        .await;
        let rows = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("id", DataType::Int64, true),
                Field::new("ID", DataType::Int64, true),
            ])),
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3])),
                Arc::new(Int64Array::from(vec![3, 1, 2])),
            ],
        )
        .unwrap();
        let written = lake.table_writer().append_table("main", "t", &[rows]).await;
        match expected {
            Ok(expected) => {
                written.unwrap();
                assert_eq!(ids_in_rowid_order(&lake).await, expected, "key {key}");
            },
            Err(message) => {
                let error = written.unwrap_err();
                assert!(
                    matches!(&error, datafusion_ducklake::DuckLakeError::InvalidConfig(m)
                        if m.contains(message)),
                    "key {key}: {error}"
                );
            },
        }
    }
}

/// SQL `INSERT` sorts under the `sort_on_insert` the catalog stores, as
/// official's `PlanInsert` reads it, and exactly one of the plan and the
/// writer sorts. A table handle whose own option disagrees with the stored
/// setting (`DuckLakeTable::with_write_options`) does not change the layout:
/// stored `true` sorts the inlined `(3), (1), (2)`, stored `false` keeps their
/// order. The setting is read once, when the statement is planned, as
/// official reads it in `PlanInsert`: one changed between planning and
/// execution does not change the layout. Guards against the plan sorting
/// under the handle's option while the writer, assuming the plan sorted,
/// skips its own sort, and against the writer reading the setting again.
#[tokio::test(flavor = "multi_thread")]
async fn sql_insert_sorts_under_the_stored_sort_on_insert() {
    for (stored, option, expected) in [(true, false, vec![1, 2, 3]), (false, true, vec![3, 1, 2])] {
        let case = format!("stored {stored}, handle option {option}");
        let lake = sqlite()(Some(5)).await;
        create(&lake, "t").await;
        lake.sql("ALTER TABLE lake.main.t SET SORTED BY (id)")
            .await
            .unwrap();
        lake.writer
            .set_table_setting(lake.table_id("t"), "sort_on_insert", &stored.to_string())
            .unwrap();
        let (_, table) = lake.table("t").await;
        let mut options = DuckLakeWriteOptions::default();
        options.sort_on_insert = Some(option);
        let ctx = SessionContext::new();
        ctx.register_table("h", Arc::new(table.with_write_options(options)))
            .unwrap();
        ctx.sql("INSERT INTO h VALUES (3, 'p'), (1, 'p'), (2, 'p')")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        assert_eq!(lake.inlined_rows("t"), 3, "{case}");
        assert_eq!(ids_in_rowid_order(&lake).await, expected, "{case}");
    }
    for (planned, executed) in [(false, true), (true, false)] {
        let case = format!("planned under {planned}, executed under {executed}");
        let expected = if planned {
            vec![1, 2, 3]
        } else {
            vec![3, 1, 2]
        };
        let lake = sqlite()(Some(5)).await;
        create(&lake, "t").await;
        lake.sql("ALTER TABLE lake.main.t SET SORTED BY (id)")
            .await
            .unwrap();
        let set = |on: bool| {
            lake.writer
                .set_table_setting(lake.table_id("t"), "sort_on_insert", &on.to_string())
                .unwrap()
        };
        set(planned);
        let ctx = SessionContext::new();
        DuckLakeCatalog::register(&ctx, "lake", lake.catalog());
        let plan = ctx
            .sql("INSERT INTO lake.main.t VALUES (3, 'p'), (1, 'p'), (2, 'p')")
            .await
            .unwrap()
            .create_physical_plan()
            .await
            .unwrap();
        set(executed);
        datafusion::physical_plan::collect(plan, ctx.task_ctx())
            .await
            .unwrap();
        assert_eq!(lake.inlined_rows("t"), 3, "{case}");
        assert_eq!(ids_in_rowid_order(&lake).await, expected, "{case}");
    }
}

/// Official skips a sort key whose dialect is not exactly `duckdb`
/// (`field.dialect != "duckdb"` in `ducklake_compaction_functions.cpp` and
/// `ducklake_sort_data.cpp`), so a key of dialect `DuckDB` orders nothing.
/// Guards against matching the dialect ignoring case.
#[tokio::test(flavor = "multi_thread")]
async fn sort_key_dialect_matches_exactly() {
    for sql in [false, true] {
        let lake = sorted_table(5, true, InsertPath::Rows).await;
        raw(
            &lake,
            "UPDATE ducklake_sort_expression SET dialect = 'DuckDB'",
        )
        .await;
        let path = if sql {
            InsertPath::Sql
        } else {
            InsertPath::Rows
        };
        insert_through(&lake, path, None, &[3, 1, 2]).await;
        assert_eq!(ids_in_rowid_order(&lake).await, vec![3, 1, 2], "sql {sql}");
    }
    let lake = inlined_under_sort_key(5, "id").await;
    raw(
        &lake,
        "UPDATE ducklake_sort_expression SET dialect = 'DuckDB'",
    )
    .await;
    flush_t(&lake, false).await;
    assert_eq!(file_ids(&lake), vec![vec![3, 1, 2]], "flush");
}

/// A sort expression DataFusion cannot plan (a DuckDB macro) lays the rows
/// out in the caller's order instead of failing the write: the order is
/// layout only. This diverges from official for `no_such_macro(id)`, whose
/// binder fails the insert (a macro the connection lacks). Guards against
/// failing a write over a key only DuckDB evaluates.
#[tokio::test(flavor = "multi_thread")]
async fn unevaluable_sort_expressions_keep_the_callers_order() {
    for expression in ["no_such_macro(id)"] {
        for path in [InsertPath::Rows, InsertPath::Sql, InsertPath::Session] {
            let case = format!("{expression}, {path:?}");
            let lake = sorted_table(5, true, path).await;
            raw(
                &lake,
                &format!("UPDATE ducklake_sort_expression SET expression = '{expression}'"),
            )
            .await;
            insert_through(&lake, path, None, &[3, 1, 2]).await;
            assert_eq!(ids_in_rowid_order(&lake).await, vec![3, 1, 2], "{case}");
        }
        let lake = inlined_under_sort_key(5, expression).await;
        flush_t(&lake, false).await;
        assert_eq!(file_ids(&lake), vec![vec![3, 1, 2]], "{expression}, flush");
    }
}

/// A sort key naming a column the table lacks fails every insert path with
/// official's binder error (`ValidateSortExpressionColumns`: "Columns in the
/// SET SORTED BY statement were not found in the DuckLake table. Unmatched
/// columns were: gone") exactly when official plans the sort: with
/// `sort_on_insert` on, or off while inlining is enabled, however many rows
/// the insert holds, inlined or none (checked with the DuckDB CLI on `INSERT
/// ... VALUES (3, 'p')` and an `INSERT ... SELECT` of no rows, under limits
/// `0` and `10`). With it off and inlining disabled the insert succeeds.
/// Guards against validating only the rows a write sorts, skipping an empty
/// write, and failing a write official completes.
#[tokio::test(flavor = "multi_thread")]
async fn missing_sort_key_column_fails_the_write() {
    use InsertPath::*;
    let cases = [
        (Rows, &[3, 1, 2][..]),
        (Staged, &[3, 1, 2]),
        (Session, &[3, 1, 2]),
        (Upsert, &[3, 1, 2]),
        (Sql, &[3, 1, 2]),
        (Partitioned, &[3, 1, 2]),
        (SqlPartitioned, &[3, 1, 2]),
        (Rows, &[]),
        (Session, &[]),
        (Sql, &[]),
    ];
    for (path, ids) in cases {
        for sort_on_insert in [true, false] {
            for limit in [0usize, 5] {
                let case = format!(
                    "{path:?}, {} rows, sort_on_insert {sort_on_insert}, limit {limit}",
                    ids.len()
                );
                let lake = sorted_table(limit, sort_on_insert, path).await;
                raw(
                    &lake,
                    "UPDATE ducklake_sort_expression SET expression = 'gone'",
                )
                .await;
                let result = try_insert_through(&lake, path, None, ids).await;
                if sort_on_insert || limit > 0 {
                    let error = result.expect_err(&case);
                    assert!(
                        error.contains(
                            "Columns in the SET SORTED BY statement were not found in the \
                             DuckLake table. Unmatched columns were: gone"
                        ),
                        "{case}: {error}"
                    );
                    assert_eq!(lake.data_files("t"), 0, "{case}");
                    assert_eq!(lake.inlined_rows("t"), 0, "{case}");
                } else {
                    result.expect(&case);
                    assert_eq!(ids_in_rowid_order(&lake).await, ids.to_vec(), "{case}");
                }
            }
        }
    }
}

/// `SET SORTED BY` stores a key as DuckDB writes a column reference: quoted
/// when the name needs it (`"My Col"`) or is a keyword (`"name"`), bare
/// otherwise (checked with the DuckDB CLI: `SET SORTED BY ("My Col", name)`
/// stores `"My Col"` and `"name"`), and an insert sorts by it. Guards against
/// storing `My Col` unquoted, which DuckDB cannot parse and this crate
/// treated as an unplannable expression, leaving the rows unsorted.
#[tokio::test(flavor = "multi_thread")]
async fn quoted_sort_keys_are_stored_as_duckdb_writes_them() {
    let lake = sqlite()(Some(0)).await;
    create_with(
        &lake,
        "q",
        &[("My Col", DataType::Int64), ("name", DataType::Utf8)],
    )
    .unwrap();
    lake.sql(r#"ALTER TABLE lake.main.q SET SORTED BY ("My Col" DESC, name)"#)
        .await
        .unwrap();
    let spec = lake
        .writer
        .live_sort_spec(lake.table_id("q"))
        .unwrap()
        .unwrap();
    let stored: Vec<&str> = spec.fields.iter().map(|f| f.expression.as_str()).collect();
    assert_eq!(stored, [r#""My Col""#, r#""name""#]);
    let rows = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("My Col", DataType::Int64, true),
            Field::new("name", DataType::Utf8, true),
        ])),
        vec![
            Arc::new(Int64Array::from(vec![1, 3, 2])),
            Arc::new(StringArray::from(vec!["a", "b", "c"])),
        ],
    )
    .unwrap();
    lake.table_writer()
        .append_table("main", "q", &[rows])
        .await
        .unwrap();
    let ids: Vec<String> = lake
        .rows("q")
        .await
        .iter()
        .map(|row| row.split(',').nth(1).unwrap().to_string())
        .collect();
    assert_eq!(ids, ["3", "2", "1"]);
}

/// DuckDB writes some sort-key functions qualified by its default schema
/// (`SET SORTED BY (main.abs(id - 2))` stores `main.abs((id - 2))`, and
/// `extract(day FROM ts)` stores `main.date_part('day', ts)`); an insert
/// sorts by such a key as official does (checked with the DuckDB CLI: `(3),
/// (1), (2)` inserted to Parquet read `2, 3, 1`). Guards against DataFusion
/// failing to resolve `main.abs`, which left the rows in the caller's order.
#[tokio::test(flavor = "multi_thread")]
async fn main_qualified_sort_functions_order_inserts() {
    for path in [InsertPath::Rows, InsertPath::Sql, InsertPath::Session] {
        let lake = sorted_table(0, true, path).await;
        raw(
            &lake,
            "UPDATE ducklake_sort_expression SET expression = 'main.abs((id - 2))'",
        )
        .await;
        insert_through(&lake, path, None, &[3, 1, 2]).await;
        let expected = if path == InsertPath::Session {
            // A session streams its Parquet rows in the caller's order.
            vec![3, 1, 2]
        } else {
            vec![2, 3, 1]
        };
        assert_eq!(ids_in_rowid_order(&lake).await, expected, "{path:?}");
    }
}

/// The deprecated flush writes the rows to Parquet whatever inlining limit
/// is stored: a stored positive limit does not inline them again. Guards
/// against the flush ending the inlined rows only to inline them anew.
#[tokio::test(flavor = "multi_thread")]
async fn deprecated_flush_writes_parquet_under_a_stored_limit() {
    let lake = sqlite()(Some(5)).await;
    create(&lake, "t").await;
    lake.writer
        .set_table_setting(lake.table_id("t"), "data_inlining_row_limit", "10")
        .unwrap();
    lake.sql("INSERT INTO lake.main.t VALUES (3, 'p'), (1, 'p'), (2, 'p')")
        .await
        .unwrap();
    assert_eq!(lake.inlined_rows("t"), 3);
    flush_t(&lake, true).await;
    assert_eq!(lake.inlined_rows("t"), 0);
    assert_eq!(lake.data_files("t"), 1);
    assert_eq!(ids_in_rowid_order(&lake).await.len(), 3);
}

/// A data file whose catalog row records no `record_count` (official's
/// column is nullable) bounds `delete_positions` by its footer's row count:
/// a position past its rows is refused, as for a file with a count. Guards
/// against recording a deletion of a row the file does not have.
#[tokio::test(flavor = "multi_thread")]
async fn delete_positions_bounded_by_the_footer_without_a_record_count() {
    let lake = sqlite()(Some(5)).await;
    create(&lake, "t").await;
    lake.append_parquet("t", &batch(1..=20, "a")).await;
    raw(&lake, "UPDATE ducklake_data_file SET record_count = NULL").await;
    let file = lake
        .provider
        .get_table_files_for_select(lake.table_id("t"), lake.head())
        .unwrap()
        .remove(0);
    assert_eq!(file.max_row_count, None);
    let (ctx, table) = lake.table("t").await;
    let refused = table
        .delete_positions(&ctx.state(), vec![(file.data_file_id, vec![20])])
        .await
        .unwrap_err();
    assert!(
        matches!(&refused, datafusion_ducklake::DuckLakeError::InvalidConfig(message)
            if message.contains("outside")),
        "{refused}"
    );
    assert_eq!(lake.inlined_deletes("t"), 0);
    let deleted = table
        .delete_positions(&ctx.state(), vec![(file.data_file_id, vec![19])])
        .await
        .unwrap();
    assert_eq!(deleted.rows_deleted, 1);
}

/// `delete_positions` naming one data file twice is refused as such, and
/// deletes nothing. Guards against a second entry reported as a file that is
/// not live.
#[tokio::test(flavor = "multi_thread")]
async fn delete_positions_refuses_a_file_given_twice() {
    let lake = sqlite()(Some(5)).await;
    create(&lake, "t").await;
    lake.append_parquet("t", &batch(1..=20, "a")).await;
    let file = lake
        .provider
        .get_table_files_for_select(lake.table_id("t"), lake.head())
        .unwrap()
        .remove(0);
    let (ctx, table) = lake.table("t").await;
    let refused = table
        .delete_positions(
            &ctx.state(),
            vec![(file.data_file_id, vec![1]), (file.data_file_id, vec![2])],
        )
        .await
        .unwrap_err();
    assert!(
        matches!(&refused, datafusion_ducklake::DuckLakeError::InvalidConfig(message)
            if message.contains("more than once")),
        "{refused}"
    );
    assert_eq!(lake.inlined_deletes("t"), 0);
    assert_eq!(lake.delete_files("t"), 0);
}

// ---------------------------------------------------------------------------
// Partial files read at an earlier snapshot
// ---------------------------------------------------------------------------

/// An official SQLite catalog whose `t(k)` holds `(1), (2)` and `(3), (4)`,
/// inserted at two snapshots and merged into one partial file, with the
/// file's Parquet rewritten by `rewrite`. Returns the metadata, the data
/// path, the SQLite URL and the first insert's snapshot.
async fn official_partial_file(
    temp: &Path,
    rewrite: impl FnOnce(RecordBatch) -> RecordBatch,
) -> (Metadata, PathBuf, String, i64) {
    let path = temp.join("c.db");
    let data = temp.join("data");
    std::fs::create_dir_all(&data).unwrap();
    let metadata = Metadata::Sqlite(path.clone());
    official(
        &metadata,
        &data,
        &[
            "CALL lake.set_option('data_inlining_row_limit', '0')".to_string(),
            "CREATE TABLE lake.t(k INTEGER)".to_string(),
            "INSERT INTO lake.t VALUES (1), (2)".to_string(),
            "INSERT INTO lake.t VALUES (3), (4)".to_string(),
            "CALL ducklake_merge_adjacent_files('lake')".to_string(),
        ],
    );
    let url = format!("sqlite:{}?mode=rwc", path.display());
    let pool = sqlx::SqlitePool::connect(&url).await.unwrap();
    let (file_id, file, begin, partial_max): (i64, String, i64, Option<i64>) = sqlx::query_as(
        "SELECT data_file_id, path, begin_snapshot, partial_max FROM ducklake_data_file
         WHERE end_snapshot IS NULL",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert!(partial_max.is_some_and(|max| max > begin));
    let file = data.join("main").join("t").join(file);
    let reader = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(
        std::fs::File::open(&file).unwrap(),
    )
    .unwrap()
    .build()
    .unwrap();
    let batches = reader.collect::<Result<Vec<_>, _>>().unwrap();
    let batch = arrow::compute::concat_batches(&batches[0].schema(), &batches).unwrap();
    let batch = rewrite(batch);
    let mut writer = parquet::arrow::ArrowWriter::try_new(
        std::fs::File::create(&file).unwrap(),
        batch.schema(),
        None,
    )
    .unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    let size = std::fs::metadata(&file).unwrap().len() as i64;
    sqlx::query(
        "UPDATE ducklake_data_file SET file_size_bytes = ?, footer_size = NULL
         WHERE data_file_id = ?",
    )
    .bind(size)
    .bind(file_id)
    .execute(&pool)
    .await
    .unwrap();
    pool.close().await;
    (metadata, data, url, begin)
}

/// The handle of `main.t` of the catalog at `url`, read at `snapshot` with
/// row lineage.
async fn partial_table(url: &str, snapshot: i64) -> (SessionContext, DuckLakeTable) {
    let provider: Arc<dyn MetadataProvider> =
        Arc::new(SqliteMetadataProvider::new(url).await.unwrap());
    let ctx = SessionContext::new();
    let catalog = DuckLakeCatalog::with_snapshot(provider, snapshot)
        .unwrap()
        .with_row_lineage(true);
    ctx.register_catalog("lake", Arc::new(catalog));
    let provider = ctx
        .catalog("lake")
        .unwrap()
        .schema("main")
        .unwrap()
        .table("t")
        .await
        .unwrap()
        .unwrap();
    let table = (provider.as_ref() as &dyn std::any::Any)
        .downcast_ref::<DuckLakeTable>()
        .unwrap()
        .clone();
    (ctx, table)
}

/// A row of a partial file whose `_ducklake_internal_snapshot_id` is NULL
/// fails official's `<=` snapshot filter, so a read at a snapshot before the
/// file's last leaves it out (checked with the DuckDB CLI), while a read at or
/// after its last applies no filter and shows it. The scan, `locate_rowids`
/// and `read_rows_by_rowid` agree. Guards against treating a NULL origin as
/// visible.
#[tokio::test(flavor = "multi_thread")]
async fn partial_file_rows_with_a_null_snapshot_are_hidden() {
    let temp = tempfile::TempDir::new().unwrap();
    let (metadata, data, url, first) = official_partial_file(temp.path(), |batch| {
        let k = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap()
            .clone();
        let snapshots = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let nulled: Int64Array = (0..batch.num_rows())
            .map(|row| (k.value(row) != 1).then(|| snapshots.value(row)))
            .collect();
        RecordBatch::try_new(
            batch.schema(),
            vec![batch.column(0).clone(), Arc::new(nulled)],
        )
        .unwrap()
    })
    .await;
    for snapshot in [first, first + 1] {
        let theirs = duckdb_query(
            &metadata,
            &data,
            &format!("SELECT rowid, k FROM lake.t AT (VERSION => {snapshot}) ORDER BY rowid"),
        );
        let (ctx, table) = partial_table(&url, snapshot).await;
        let ours = lines(
            &ctx.sql("SELECT rowid, k FROM lake.main.t ORDER BY rowid")
                .await
                .unwrap()
                .collect()
                .await
                .unwrap(),
            true,
        );
        assert_eq!(ours, theirs, "snapshot {snapshot}");
        let visible = theirs.iter().any(|row| row == "0,1");
        assert_eq!(visible, snapshot > first, "snapshot {snapshot}: {theirs:?}");
        let located = table.locate_rowids(&ctx.state(), &[0, 1]).await.unwrap();
        let mut ids: Vec<i64> = located.iter().map(|row| row.row_id).collect();
        ids.sort_unstable();
        assert_eq!(
            ids,
            if visible {
                vec![0, 1]
            } else {
                vec![1]
            },
            "snapshot {snapshot}"
        );
        let read = table.read_rows_by_rowid(&ctx.state(), &[0]).await.unwrap();
        let rows: usize = read.iter().map(RecordBatch::num_rows).sum();
        assert_eq!(rows, usize::from(visible), "snapshot {snapshot}");
    }
}

/// A partial file embedding row ids but no `_ducklake_internal_snapshot_id`
/// column cannot be read at a snapshot before its last: official's reader
/// fails ("snapshot column was not present in file"), and so do the scan and
/// `locate_rowids`. Guards against `locate_rowids` reporting every row of
/// such a file as live.
#[tokio::test(flavor = "multi_thread")]
async fn embedded_partial_file_without_snapshots_fails_closed() {
    let temp = tempfile::TempDir::new().unwrap();
    let (_, _, url, first) = official_partial_file(temp.path(), |batch| {
        let k_field = batch.schema().field(0).clone();
        let row_id = Field::new("_ducklake_internal_row_id", DataType::Int64, true).with_metadata(
            [(
                "PARQUET:field_id".to_string(),
                datafusion_ducklake::row_id::ROW_ID_PARQUET_FIELD_ID.to_string(),
            )]
            .into(),
        );
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![k_field, row_id])),
            vec![
                batch.column(0).clone(),
                Arc::new(Int64Array::from_iter_values(0..batch.num_rows() as i64)),
            ],
        )
        .unwrap()
    })
    .await;
    let (ctx, table) = partial_table(&url, first).await;
    let file = &table.files().unwrap()[0];
    assert_eq!(
        table.file_row_id_layout(&ctx.state(), file).await.unwrap(),
        Some(datafusion_ducklake::RowIdLayout::Embedded)
    );
    let scan = ctx
        .sql("SELECT rowid, k FROM lake.main.t")
        .await
        .unwrap()
        .collect()
        .await;
    assert!(scan.is_err(), "{scan:?}");
    let located = table.locate_rowids(&ctx.state(), &[0, 1]).await;
    assert!(
        located
            .as_ref()
            .is_err_and(|e| e.to_string().contains("_ducklake_internal_snapshot_id")),
        "{located:?}"
    );
}

// ---------------------------------------------------------------------------
// rowid lookup without a recorded row count
// ---------------------------------------------------------------------------

/// A file whose catalog `record_count` is NULL (a foreign catalog that
/// omits it) holds the rows its footer records: a `rowid` past them names no
/// row, and `delete_rowids` deletes only the rows that exist. Guards against
/// locating ids at positions the file does not have, which wrote phantom
/// positions into a delete file.
#[tokio::test(flavor = "multi_thread")]
async fn rowid_lookup_without_a_record_count_reads_the_footer() {
    let lake = sqlite()(Some(0)).await;
    create(&lake, "t").await;
    lake.append_parquet("t", &batch(1..=3, "a")).await;
    raw(&lake, "UPDATE ducklake_data_file SET record_count = NULL").await;
    let (ctx, table) = lake.table("t").await;
    assert_eq!(table.files().unwrap()[0].max_row_count, None);
    let located = table
        .locate_rowids(&ctx.state(), &[0, 2, 3, 100])
        .await
        .unwrap();
    let mut ids: Vec<i64> = located.iter().map(|row| row.row_id).collect();
    ids.sort_unstable();
    assert_eq!(ids, vec![0, 2]);
    let deleted = table
        .delete_rowids(&ctx.state(), &[1, 3, 100])
        .await
        .unwrap();
    assert_eq!(deleted.rows_deleted, 1);
    let ids: Vec<String> = lake
        .rows("t")
        .await
        .iter()
        .map(|row| row.split(',').nth(1).unwrap().to_string())
        .collect();
    assert_eq!(ids, vec!["1", "3"]);
}

// ---------------------------------------------------------------------------
// delete_rowids on official's UPDATE output
// ---------------------------------------------------------------------------

/// Official's `UPDATE` writes the new row versions to a file that embeds
/// their row ids under a new `row_id_start`. `delete_rowids` deletes them by
/// those ids, and official reads the result as this crate does. Guards
/// against locating an updated row at `row_id_start + position`.
#[tokio::test(flavor = "multi_thread")]
async fn delete_rowids_on_an_official_update_file() {
    let temp = tempfile::TempDir::new().unwrap();
    let path = temp.path().join("c.db");
    let data = temp.path().join("data");
    std::fs::create_dir_all(&data).unwrap();
    let metadata = Metadata::Sqlite(path.clone());
    official(
        &metadata,
        &data,
        &[
            "CALL lake.set_option('data_inlining_row_limit', '0')".to_string(),
            "CREATE TABLE lake.t(id INTEGER, v INTEGER)".to_string(),
            "INSERT INTO lake.t SELECT i, i FROM range(10) r(i)".to_string(),
            "UPDATE lake.t SET v = v + 100 WHERE id IN (3, 7)".to_string(),
        ],
    );
    let (writer, provider) = crate_side(&metadata, &data).await;
    let catalog = DuckLakeCatalog::with_writer(Arc::clone(&provider), writer).unwrap();
    let ctx = SessionContext::new();
    DuckLakeCatalog::register(&ctx, "lake", catalog);
    let handle = ctx
        .catalog("lake")
        .unwrap()
        .schema("main")
        .unwrap()
        .table("t")
        .await
        .unwrap()
        .unwrap();
    let table = (handle.as_ref() as &dyn std::any::Any)
        .downcast_ref::<DuckLakeTable>()
        .unwrap()
        .clone();
    let mut embedded = 0;
    for file in table.files().unwrap() {
        if table.file_row_id_layout(&ctx.state(), &file).await.unwrap()
            == Some(datafusion_ducklake::RowIdLayout::Embedded)
        {
            embedded += 1;
        }
    }
    assert_eq!(
        embedded, 1,
        "official's UPDATE wrote one file embedding row ids"
    );
    // Row ids 3 and 7 are the updated rows; 0 is an original one.
    let deleted = table.delete_rowids(&ctx.state(), &[0, 3, 7]).await.unwrap();
    assert_eq!(deleted.rows_deleted, 3);
    // The snapshots this crate's writer adds leave official's id counters
    // NULL: carry them forward so official attaches.
    let pool = sqlx::SqlitePool::connect(&format!("sqlite:{}", path.display()))
        .await
        .unwrap();
    for column in ["next_catalog_id", "next_file_id"] {
        sqlx::query(sqlx::AssertSqlSafe(format!(
            "UPDATE ducklake_snapshot SET {column} = \
             (SELECT MAX({column}) FROM ducklake_snapshot) + 100 WHERE {column} IS NULL"
        )))
        .execute(&pool)
        .await
        .unwrap();
    }
    pool.close().await;
    let theirs = duckdb_query(
        &metadata,
        &data,
        "SELECT rowid, id, v FROM lake.t ORDER BY rowid",
    );
    assert_eq!(
        theirs,
        vec!["1,1,1", "2,2,2", "4,4,4", "5,5,5", "6,6,6", "8,8,8", "9,9,9"]
    );
}

// ---------------------------------------------------------------------------
// Keyed mutations: long key lists, wide decimals, partial dates
// ---------------------------------------------------------------------------

/// A delete of more keys than DataFusion's pruning rewrites as an `IN` list
/// prunes files by the keys' bounds and deletes exactly the keyed rows of a
/// table of several files, leaving the files past the bounds untouched.
#[tokio::test(flavor = "multi_thread")]
async fn keyed_delete_of_a_long_key_list_over_many_files() {
    let lake = sqlite()(Some(0)).await;
    create(&lake, "t").await;
    for file in 0..5 {
        lake.append_parquet("t", &batch(file * 10..file * 10 + 10, "a"))
            .await;
    }
    let keys = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, true)])),
        vec![Arc::new(Int64Array::from_iter_values(3..28))],
    )
    .unwrap();
    let (ctx, table) = lake.table("t").await;
    let result = table
        .keyed_mutation(&["id"])
        .delete_by_keys(&ctx.state(), vec![keys])
        .await
        .unwrap();
    assert_eq!(result.rows_deleted, 25, "{result:?}");
    let ids: Vec<i64> = lake
        .rows("t")
        .await
        .iter()
        .map(|row| row.split(',').nth(1).unwrap().parse().unwrap())
        .collect();
    assert_eq!(ids, (0..3).chain(28..50).collect::<Vec<_>>());
    // Files 0 to 2 hold the keys; 3 and 4 lie past them.
    assert_eq!(
        lake.data_files("t"),
        4,
        "the wholly deleted file 1 is ended"
    );
    assert_eq!(lake.delete_files("t"), 2);
}

/// Keys staged as `Decimal256` match the table's `DECIMAL(12, 2)` rows by
/// value.
#[tokio::test(flavor = "multi_thread")]
async fn keyed_mutations_match_decimal256_keys() {
    use arrow::array::{Decimal128Array, Decimal256Array};
    use arrow::datatypes::i256;
    let lake = sqlite()(Some(5)).await;
    create_with(
        &lake,
        "t",
        &[("id", DataType::Decimal128(12, 2)), ("v", DataType::Utf8)],
    )
    .unwrap();
    let rows = named(vec![
        (
            "id",
            Arc::new(
                Decimal128Array::from(vec![100, 250, 399])
                    .with_precision_and_scale(12, 2)
                    .unwrap(),
            ),
        ),
        ("v", Arc::new(StringArray::from(vec!["a", "b", "c"]))),
    ]);
    lake.append_parquet("t", &rows).await;
    lake.append("t", &rows.slice(0, 1)).await;
    let keys = named(vec![(
        "id",
        Arc::new(
            Decimal256Array::from(vec![i256::from(100), i256::from(250)])
                .with_precision_and_scale(10, 2)
                .unwrap(),
        ),
    )]);
    let (ctx, table) = lake.table("t").await;
    let result = table
        .keyed_mutation(&["id"])
        .delete_by_keys(&ctx.state(), vec![keys])
        .await
        .unwrap();
    assert_eq!(result.rows_deleted, 3, "{result:?}");
    let left: Vec<String> = lake
        .rows("t")
        .await
        .iter()
        .map(|row| row.split_once(',').unwrap().1.to_string())
        .collect();
    assert_eq!(left, vec!["3.99,c"]);
}

/// A staged `Date64` widens to the table's `DATE` only when it is a whole
/// day: one that is not is refused before anything is written, where a cast
/// would truncate it to its day. Guards against writing a value the caller
/// did not stage.
#[tokio::test(flavor = "multi_thread")]
async fn keyed_mutations_refuse_a_partial_date64() {
    use arrow::array::{Date32Array, Date64Array};
    const DAY_MS: i64 = 86_400_000;
    let lake = sqlite()(Some(0)).await;
    create_with(
        &lake,
        "t",
        &[("id", DataType::Int64), ("d", DataType::Date32)],
    )
    .unwrap();
    lake.append_parquet(
        "t",
        &named(vec![
            ("id", Arc::new(Int64Array::from(vec![1]))),
            ("d", Arc::new(Date32Array::from(vec![0]))),
        ]),
    )
    .await;
    let staged = |ms: i64| {
        named(vec![
            ("id", Arc::new(Int64Array::from(vec![1]))),
            ("d", Arc::new(Date64Array::from(vec![ms]))),
        ])
    };
    let head = lake.head();
    let (ctx, table) = lake.table("t").await;
    let refused = table
        .keyed_mutation(&["id"])
        .update_by_keys(&ctx.state(), vec![staged(2 * DAY_MS + 1)])
        .await;
    assert!(
        matches!(&refused, Err(datafusion_ducklake::DuckLakeError::InvalidConfig(message)) if message.contains("whole day")),
        "{refused:?}"
    );
    assert_eq!(lake.head(), head, "nothing is written");
    let (ctx, table) = lake.table("t").await;
    table
        .keyed_mutation(&["id"])
        .update_by_keys(&ctx.state(), vec![staged(2 * DAY_MS)])
        .await
        .unwrap();
    assert_eq!(lake.rows("t").await, vec!["0,1,1970-01-03"]);
}

// ---------------------------------------------------------------------------
// Multi-catalog: change detection and the inlining limit
// ---------------------------------------------------------------------------

/// On a multi-catalog store, whose catalogs share one snapshot id sequence,
/// a window over a table whose first snapshots were expired is incomplete:
/// the whole history, and a window starting at a surviving snapshot before
/// the table was created. The table's catalog-scoped inlining limit outranks
/// a catalog-wide one stored without a scope when a write is sized, as it
/// does at the commit. Guards against reporting such a window unchanged and
/// complete, and against the two reading the limits in different orders.
#[cfg(feature = "write-postgres")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
#[tokio::test(flavor = "multi_thread")]
async fn multicatalog_regressions() {
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
    let make = crate::inlining_abstraction_tests::multicatalog(pool.clone());

    // "lake0": the first catalog `make` creates.
    let lake = make(Some(5)).await;
    create(&lake, "w").await;
    let initial = lake.head();
    create(&lake, "t").await;
    let created = lake.head();
    lake.append("t", &batch([1], "a")).await;
    let inserted = lake.head();
    create(&lake, "u").await;
    let head = lake.head();
    let table_id = lake.table_id("t");
    datafusion_ducklake::MulticatalogManager::new(pool.clone())
        .expire_snapshots_in_catalog("lake0", ExpireCriteria::Versions(vec![created, inserted]))
        .await
        .unwrap();
    let ids: Vec<i64> = lake
        .provider
        .list_snapshots()
        .unwrap()
        .iter()
        .map(|snapshot| snapshot.snapshot_id)
        .collect();
    assert!(
        ids.contains(&initial) && !ids.contains(&created) && !ids.contains(&inserted),
        "{ids:?}"
    );
    for after in [-1, initial] {
        let changes = data_changes(&lake.provider, table_id, after, head).await;
        assert!(
            changes.history_incomplete && changes.rows_changed(),
            "after {after}: {changes:?}"
        );
        assert_eq!(changes.last_rows_changed_snapshot, Some(head));
    }
    assert_eq!(
        last_rows_changed(&lake.provider, table_id, head).await,
        Some(head)
    );
    // A window after the gap is complete.
    lake.append("u", &batch([1], "x")).await;
    let changes = data_changes(&lake.provider, table_id, head, lake.head()).await;
    assert!(
        !changes.history_incomplete && !changes.rows_changed(),
        "{changes:?}"
    );

    let lake = make(None).await;
    create(&lake, "t").await;
    sqlx::query(
        "INSERT INTO ducklake_metadata (key, value, scope, scope_id)
         VALUES ('data_inlining_row_limit', '2', NULL, NULL)",
    )
    .execute(&pool)
    .await
    .unwrap();
    lake.writer
        .set_global_setting("data_inlining_row_limit", "7")
        .unwrap();
    assert_eq!(
        lake.writer
            .table_data_inlining_row_limit("main", "t")
            .unwrap(),
        7
    );
    lake.append("t", &batch(1..=5, "a")).await;
    assert_eq!(lake.inlined_rows("t"), 5);
}

/// Keyed mutations and `rowid` lookups over 10 000 Parquet rows in two files
/// of 2 048-row row groups, run one file at a time and several in parallel
/// (`with_concurrency`), agree with a model: rows past a file's first row
/// group and past 8 192 rows (a scan batch) are found, updated and deleted
/// at the right positions, and keep their row ids. Guards against positions
/// or row ids computed per row group or per batch rather than per file,
/// which only rows past the first of either expose.
#[tokio::test(flavor = "multi_thread")]
async fn keyed_and_rowid_apis_across_row_groups() {
    for concurrency in [1, 4] {
        let lake = sqlite()(Some(0)).await;
        create(&lake, "t").await;
        let mut options = DuckLakeWriteOptions::default().with_data_inlining_row_limit(0);
        options.max_row_group_rows = Some(2048);
        for range in [0..5000i64, 5000..10_000] {
            lake.table_writer()
                .with_options(&options)
                .append_table("main", "t", &[batch(range, "a")])
                .await
                .unwrap();
        }
        assert_eq!(lake.data_files("t"), 2);
        for file in lake
            .provider
            .get_table_files_for_select(lake.table_id("t"), lake.head())
            .unwrap()
        {
            let name = Path::new(&file.file.path).file_name().unwrap().to_owned();
            let path = find_file(&lake.data, &name).unwrap();
            let reader = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(
                std::fs::File::open(path).unwrap(),
            )
            .unwrap();
            assert_eq!(reader.metadata().num_row_groups(), 3, "2 048-row groups");
        }
        let mut model: BTreeMap<i64, String> =
            (0..10_000).map(|id| (id, format!("a{id}"))).collect();
        let row_id_of = |rows: &[String], id: i64| -> i64 {
            rows.iter()
                .find(|row| row.split(',').nth(1) == Some(&id.to_string()))
                .map(|row| row.split(',').next().unwrap().parse().unwrap())
                .unwrap()
        };
        let original = lake.rows("t").await;

        let doomed = [1, 2047, 2048, 4999, 5000, 8191, 8192, 8193, 9999];
        let (ctx, table) = lake.table("t").await;
        let deleted = table
            .keyed_mutation(&["id"])
            .with_concurrency(concurrency)
            .delete_by_keys(&ctx.state(), vec![batch(doomed, "")])
            .await
            .unwrap();
        assert_eq!(
            deleted.rows_deleted,
            doomed.len() as u64,
            "concurrency {concurrency}"
        );
        for id in doomed {
            model.remove(&id);
        }

        let updated_ids = [3000, 4096, 9000];
        let (ctx, table) = lake.table("t").await;
        let updated = table
            .keyed_mutation(&["id"])
            .with_concurrency(concurrency)
            .update_by_keys(&ctx.state(), vec![batch(updated_ids, "u")])
            .await
            .unwrap();
        assert_eq!(updated.rows_updated, updated_ids.len() as u64);
        for id in updated_ids {
            model.insert(id, format!("u{id}"));
        }
        let rows = lake.rows("t").await;
        for id in updated_ids {
            assert_eq!(
                row_id_of(&rows, id),
                row_id_of(&original, id),
                "id {id} keeps its rowid"
            );
        }

        // Look rows up by rowid, past the first row group and batch.
        let (ctx, table) = lake.table("t").await;
        let wanted = [6000, 9000, 9998].map(|id| row_id_of(&rows, id));
        let read = table
            .read_rows_by_rowid(&ctx.state(), &wanted)
            .await
            .unwrap();
        let mut read = lines(&read, true);
        read.sort();
        let mut expected: Vec<String> = [6000, 9000, 9998]
            .iter()
            .map(|id| format!("{},{id},{}", row_id_of(&rows, *id), model[id]))
            .collect();
        expected.sort();
        assert_eq!(read, expected);
        let (ctx, table) = lake.table("t").await;
        let by_rowid = table
            .delete_rowids(
                &ctx.state(),
                &[row_id_of(&rows, 6000), row_id_of(&rows, 9998)],
            )
            .await
            .unwrap();
        assert_eq!(by_rowid.rows_deleted, 2);
        model.remove(&6000);
        model.remove(&9998);

        let rows = lake.rows("t").await;
        let read: BTreeMap<i64, String> = rows
            .iter()
            .map(|row| {
                let mut parts = row.split(',');
                parts.next();
                let id = parts.next().unwrap().parse().unwrap();
                (id, parts.next().unwrap().to_string())
            })
            .collect();
        assert_eq!(read, model, "concurrency {concurrency}");
        assert_eq!(rows.len(), model.len());
    }
}

// ---------------------------------------------------------------------------
// Change detection never proves a deletion away
// ---------------------------------------------------------------------------

/// The `(snapshot, changes_made)` ledger rows of table `table_id`.
fn ledger(provider: &dyn MetadataProvider, table_id: i64) -> Vec<(i64, String)> {
    let head = provider.get_current_snapshot().unwrap();
    provider
        .list_table_snapshot_changes(table_id, -1, head)
        .unwrap()
        .into_iter()
        .map(|(snapshot, changes)| (snapshot, changes.unwrap_or_default()))
        .collect()
}

/// A `DELETE` committed in one transaction with a flush records
/// `deleted_from_table` beside `inline_flush`, as a flush that moves inlined
/// deletions into delete files does; official's later flush then writes one
/// delete file dated to the first deletion, so the table's row counts from
/// catalog counts match around the `DELETE` (checked with the DuckDB CLI:
/// the `(4, 5]` window's change feed lists the ten deleted rows). The window
/// reports the deletion. A snapshot recording `inline_flush` alone moved rows
/// and reports a layout change only. Guards against proving a snapshot that
/// recorded `deleted_from_table` a pure move from equal row counts.
#[tokio::test(flavor = "multi_thread")]
async fn change_detection_counts_a_delete_committed_with_a_flush() {
    let temp = tempfile::TempDir::new().unwrap();
    let path = temp.path().join("n.db");
    let data = temp.path().join("data");
    std::fs::create_dir_all(&data).unwrap();
    let metadata = Metadata::Sqlite(path.clone());
    official(
        &metadata,
        &data,
        &[
            "CALL lake.set_option('data_inlining_row_limit', 5)".to_string(),
            "CREATE TABLE lake.h(i INTEGER)".to_string(),
            "INSERT INTO lake.h SELECT range FROM range(40)".to_string(),
            "DELETE FROM lake.h WHERE i >= 30".to_string(),
            "INSERT INTO lake.h VALUES (100)".to_string(),
            "BEGIN; DELETE FROM lake.h WHERE i >= 10 AND i < 20; \
             CALL ducklake_flush_inlined_data('lake', table_name => 'h'); COMMIT;"
                .to_string(),
            "DELETE FROM lake.h WHERE i = 2".to_string(),
            "CALL ducklake_flush_inlined_data('lake', table_name => 'h')".to_string(),
            "INSERT INTO lake.h VALUES (200)".to_string(),
            "CALL ducklake_flush_inlined_data('lake', table_name => 'h')".to_string(),
        ],
    );
    let (_, provider) = crate_side(&metadata, &data).await;
    let head = provider.get_current_snapshot().unwrap();
    let schema = provider.get_schema_by_name("main", head).unwrap().unwrap();
    let table_id = provider
        .get_table_by_name(schema.schema_id, "h", head)
        .unwrap()
        .unwrap()
        .table_id;
    let t = table_id;
    let ledger = ledger(provider.as_ref(), table_id);
    let snapshot_of = |made: &str| {
        ledger
            .iter()
            .filter(|(_, changes)| changes == made)
            .map(|(snapshot, _)| *snapshot)
            .collect::<Vec<_>>()
    };
    let [delete_with_flush] = snapshot_of(&format!("deleted_from_table:{t},inline_flush:{t}"))[..]
    else {
        panic!("{ledger:?}");
    };
    let [flush_only] = snapshot_of(&format!("inline_flush:{t}"))[..] else {
        panic!("{ledger:?}");
    };
    // Official dated the last flush's delete file to the first deletion, so
    // the catalog counts around the `DELETE` match.
    assert_eq!(
        provider
            .get_table_row_count(table_id, delete_with_flush - 1)
            .ok(),
        provider
            .get_table_row_count(table_id, delete_with_flush)
            .ok(),
    );

    let changes = data_changes(
        &provider,
        table_id,
        delete_with_flush - 1,
        delete_with_flush,
    )
    .await;
    assert!(changes.deleted && changes.flushed, "{changes:?}");
    assert!(changes.rows_changed(), "{changes:?}");
    assert_eq!(changes.last_rows_changed_snapshot, Some(delete_with_flush));

    let changes = data_changes(&provider, table_id, flush_only - 1, flush_only).await;
    assert!(changes.flushed && !changes.rows_changed(), "{changes:?}");
    assert!(changes.layout_changed(), "{changes:?}");
    assert_eq!(changes.last_rows_changed_snapshot, None);
}

/// Change detection on snapshots whose tokens record only a flush or a
/// compaction answers what official's change feed answers over them, and a
/// pure flush or merge is no row change. Each window is one snapshot of an
/// official catalog; the feed is official's `ducklake_table_changes`.
///
/// - A `DELETE` committed in one transaction after official's flush records
///   only `inline_flush` (checked with the DuckDB CLI); the feed lists the
///   deletion, so it is a deletion;
/// - a flush of inlined rows alone and a merge of adjacent files list
///   nothing;
/// - official's `ducklake_rewrite_data_files` lists every row it rewrote as
///   an `UPDATE` pair, so it is a row change.
#[tokio::test(flavor = "multi_thread")]
async fn change_detection_follows_the_change_feed() {
    let temp = tempfile::TempDir::new().unwrap();
    let path = temp.path().join("f.db");
    let data = temp.path().join("data");
    std::fs::create_dir_all(&data).unwrap();
    let metadata = Metadata::Sqlite(path.clone());
    official(
        &metadata,
        &data,
        &[
            "CALL lake.set_option('data_inlining_row_limit', 10)".to_string(),
            "CREATE TABLE lake.t(a INTEGER)".to_string(),
            "INSERT INTO lake.t VALUES (1), (2), (3)".to_string(),
            "BEGIN; CALL ducklake_flush_inlined_data('lake'); \
             DELETE FROM lake.t WHERE a = 1; COMMIT;"
                .to_string(),
            "INSERT INTO lake.t VALUES (4), (5), (6)".to_string(),
            "CALL ducklake_flush_inlined_data('lake')".to_string(),
            "INSERT INTO lake.t SELECT range FROM range(100, 120)".to_string(),
            "INSERT INTO lake.t SELECT range FROM range(200, 220)".to_string(),
            "CALL ducklake_merge_adjacent_files('lake')".to_string(),
            "DELETE FROM lake.t WHERE a IN (100, 101, 200)".to_string(),
            "CALL ducklake_flush_inlined_data('lake')".to_string(),
            "CALL ducklake_rewrite_data_files('lake', delete_threshold => 0.0)".to_string(),
        ],
    );
    let (_, provider) = crate_side(&metadata, &data).await;
    let head = provider.get_current_snapshot().unwrap();
    let schema = provider.get_schema_by_name("main", head).unwrap().unwrap();
    let t = provider
        .get_table_by_name(schema.schema_id, "t", head)
        .unwrap()
        .unwrap()
        .table_id;
    let ledger = ledger(provider.as_ref(), t);
    let feed_rows = |snapshot: i64| -> i64 {
        duckdb_query(
            &metadata,
            &data,
            &format!("SELECT count(*) FROM ducklake_table_changes('lake', 'main', 't', {snapshot}, {snapshot})"),
        )[0]
        .parse()
        .unwrap()
    };
    let layout_only = |made: &str| {
        made.split(',').all(|token| {
            [
                "inline_flush:",
                "flushed_inlined:",
                "merge_adjacent:",
                "rewrite_delete:",
                "compacted_table:",
            ]
            .iter()
            .any(|kind| token.starts_with(kind))
        })
    };
    let mut seen = Vec::new();
    for (snapshot, made) in &ledger {
        let changes = data_changes(&provider, t, snapshot - 1, *snapshot).await;
        let feed = feed_rows(*snapshot);
        if feed > 0 {
            assert!(changes.rows_changed(), "{snapshot} `{made}`: {changes:?}");
        }
        if layout_only(made) {
            assert_eq!(
                changes.rows_changed(),
                feed > 0,
                "{snapshot} `{made}`: feed {feed}, {changes:?}"
            );
            assert!(changes.layout_changed(), "{snapshot} `{made}`: {changes:?}");
            seen.push((made.split(':').next().unwrap().to_string(), feed > 0));
        }
    }
    // The flush committed with the `DELETE`, the pure flush, the merge and the
    // rewrite, in ledger order.
    assert_eq!(
        seen,
        [
            ("inline_flush".to_string(), true),
            ("inline_flush".to_string(), false),
            ("merge_adjacent".to_string(), false),
            ("rewrite_delete".to_string(), true),
        ],
        "{ledger:?}"
    );
    let [(flush_with_delete, _), ..] = ledger
        .iter()
        .filter(|(_, made)| layout_only(made))
        .cloned()
        .collect::<Vec<_>>()[..]
    else {
        unreachable!()
    };
    let changes = data_changes(&provider, t, flush_with_delete - 1, flush_with_delete).await;
    assert!(changes.deleted && changes.flushed, "{changes:?}");
    assert_eq!(changes.last_rows_changed_snapshot, Some(flush_with_delete));
    // Over the whole history, the latest row change is the rewrite.
    assert_eq!(last_rows_changed(&provider, t, head).await, Some(head));
}

// ---------------------------------------------------------------------------
// A NaN rewrite_delete_threshold, as official reads it
// ---------------------------------------------------------------------------

/// An official SQLite catalog of three Parquet files, the first two with
/// deletions, its `rewrite_delete_threshold` stored as `nan`.
fn nan_threshold_catalog(dir: &Path) -> (Metadata, PathBuf) {
    let data = dir.join("data");
    std::fs::create_dir_all(&data).unwrap();
    let metadata = Metadata::Sqlite(dir.join("c.db"));
    official(
        &metadata,
        &data,
        &[
            "CALL lake.set_option('data_inlining_row_limit', 0)".to_string(),
            "CREATE TABLE lake.t(i INTEGER)".to_string(),
            "INSERT INTO lake.t SELECT range FROM range(100)".to_string(),
            "INSERT INTO lake.t SELECT range FROM range(100, 200)".to_string(),
            "INSERT INTO lake.t SELECT range FROM range(200, 300)".to_string(),
            "DELETE FROM lake.t WHERE i < 10".to_string(),
            "DELETE FROM lake.t WHERE i = 150".to_string(),
            "CALL lake.set_option('rewrite_delete_threshold', 'nan')".to_string(),
        ],
    );
    (metadata, data)
}

/// The ids of the data files a rewrite ended.
fn ended_data_files(metadata: &Metadata) -> Vec<String> {
    let connection = crate::inlined_mutation_differential_tests::duckdb_connection();
    metadata.attach_raw(&connection);
    query_lines(
        &connection,
        "SELECT data_file_id FROM ducklake_data_file WHERE end_snapshot IS NOT NULL ORDER BY 1",
    )
}

/// `count(*), sum(i)` of `main.t`, read by this crate (official cannot read a
/// SQLite catalog this crate's writer committed to: its snapshots carry no
/// allocator watermarks).
async fn crate_content(provider: &Arc<dyn MetadataProvider>) -> Vec<String> {
    let ctx = SessionContext::new();
    let head = provider.get_current_snapshot().unwrap();
    ctx.register_catalog(
        "lake",
        Arc::new(DuckLakeCatalog::with_snapshot(Arc::clone(provider), head).unwrap()),
    );
    let batches = ctx
        .sql("SELECT count(*), sum(i) FROM lake.main.t")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    lines(&batches, false)
}

/// A `rewrite_delete_threshold` of `nan` rewrites every data file with
/// deletions, as official's does: `ducklake_set_option` stores `nan` (its
/// range check `threshold < 0 || threshold > 1` passes a NaN) and its rewrite
/// keeps a file only when `delete_ratio < threshold`, false for a NaN
/// (checked with the DuckDB CLI). `CHECKPOINT` reads the same option. Guards
/// against refusing the stored NaN and selecting with `ratio >= threshold`.
#[tokio::test(flavor = "multi_thread")]
async fn nan_rewrite_delete_threshold_rewrites_every_file_with_deletions() {
    let official_dir = tempfile::TempDir::new().unwrap();
    let (metadata, data) = nan_threshold_catalog(official_dir.path());
    official(
        &metadata,
        &data,
        &["CALL ducklake_rewrite_data_files('lake')".to_string()],
    );
    let expected = ended_data_files(&metadata);
    assert_eq!(expected, vec!["0", "1"], "official rewrites both files");
    let content = duckdb_query(&metadata, &data, "SELECT count(*), sum(i) FROM lake.t");

    let ours_dir = tempfile::TempDir::new().unwrap();
    let (metadata, data) = nan_threshold_catalog(ours_dir.path());
    let (writer, provider) = crate_side(&metadata, &data).await;
    let ctx = SessionContext::new();
    let catalog = DuckLakeCatalog::with_writer(Arc::clone(&provider), Arc::clone(&writer)).unwrap();
    DuckLakeCatalog::register(&ctx, "lake", catalog);
    let handle = ctx
        .catalog("lake")
        .unwrap()
        .schema("main")
        .unwrap()
        .table("t")
        .await
        .unwrap()
        .unwrap();
    let table = (handle.as_ref() as &dyn std::any::Any)
        .downcast_ref::<DuckLakeTable>()
        .unwrap()
        .clone();
    let rewritten = table
        .rewrite_data_files(&ctx.state(), RewriteOptions::default())
        .await
        .unwrap();
    assert!(rewritten.did_work(), "{rewritten:?}");
    assert_eq!(ended_data_files(&metadata), expected);
    assert_eq!(crate_content(&provider).await, content);

    let checkpoint_dir = tempfile::TempDir::new().unwrap();
    let (metadata, data) = nan_threshold_catalog(checkpoint_dir.path());
    let (writer, provider) = crate_side(&metadata, &data).await;
    let ctx = SessionContext::new();
    let catalog = DuckLakeCatalog::with_writer(Arc::clone(&provider), Arc::clone(&writer)).unwrap();
    catalog.checkpoint(&ctx.state()).await.unwrap();
    let ended = ended_data_files(&metadata);
    assert!(
        ended.contains(&"0".to_string()) && ended.contains(&"1".to_string()),
        "{ended:?}"
    );
    assert_eq!(crate_content(&provider).await, content);
}
