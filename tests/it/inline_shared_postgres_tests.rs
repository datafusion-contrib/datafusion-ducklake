//! Catalog-inlined rows and deletions on the multicatalog PostgreSQL backend
//! live in two shared tables, `ducklake_inlined_row` and
//! `ducklake_inlined_file_delete`. These tests check that inline writes of any
//! number of tables and schema versions create no relation, that rows written
//! under an older schema version read correctly, that a store with the earlier
//! per-table layout is migrated on open, and print the latency of a small
//! inline write and read. Docker-gated (testcontainers Postgres).

#![cfg(feature = "write-postgres")]

use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow::array::{
    Array, BinaryArray, BooleanArray, Date32Array, Decimal128Array, Float64Array, Int32Array,
    Int64Array, RecordBatch, StringArray, TimestampMicrosecondArray,
};
use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use datafusion::prelude::*;
use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeTableWriter, DuckLakeWriteOptions, MetadataProvider, MetadataWriter,
    MulticatalogManager, PostgresMetadataWriter,
};
use object_store::local::LocalFileSystem;
use sqlx::postgres::PgPool;

use super::sql_update_inline_postgres_tests::{CAT, Lake, batch};

/// An Int32 batch with the named nullable columns (the first is `id`).
fn int_batch(names: &[&str], rows: &[&[Option<i32>]]) -> RecordBatch {
    let schema = Arc::new(Schema::new(
        names
            .iter()
            .enumerate()
            .map(|(index, name)| Field::new(*name, DataType::Int32, index > 0))
            .collect::<Vec<_>>(),
    ));
    let columns = (0..names.len())
        .map(|index| {
            Arc::new(Int32Array::from(
                rows.iter().map(|row| row[index]).collect::<Vec<_>>(),
            )) as _
        })
        .collect();
    RecordBatch::try_new(schema, columns).unwrap()
}

/// Rows of `sql` against the catalog at `snapshot` (or the head), each column
/// cast to Int64.
async fn query(lake: &Lake, snapshot: Option<i64>, sql: &str) -> Vec<Vec<Option<i64>>> {
    let provider = lake.provider().await;
    let snapshot = match snapshot {
        Some(snapshot) => snapshot,
        None => provider.get_current_snapshot().unwrap(),
    };
    let ctx = SessionContext::new();
    ctx.register_catalog(
        CAT,
        Arc::new(DuckLakeCatalog::with_snapshot(provider, snapshot).unwrap()),
    );
    let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
    let mut out = Vec::new();
    for batch in &batches {
        let columns = batch
            .columns()
            .iter()
            .map(|column| arrow::compute::cast(column, &DataType::Int64).unwrap())
            .collect::<Vec<_>>();
        for row in 0..batch.num_rows() {
            out.push(
                columns
                    .iter()
                    .map(|column| {
                        let column = column.as_any().downcast_ref::<Int64Array>().unwrap();
                        column.is_valid(row).then(|| column.value(row))
                    })
                    .collect(),
            );
        }
    }
    out
}

fn rows(values: &[&[Option<i64>]]) -> Vec<Vec<Option<i64>>> {
    values.iter().map(|row| row.to_vec()).collect()
}

async fn relation_count(pool: &PgPool) -> i64 {
    sqlx::query_scalar("SELECT COUNT(*) FROM pg_class")
        .fetch_one(pool)
        .await
        .unwrap()
}

/// Inline inserts, schema changes, inline UPDATEs and DELETEs, and inlined
/// deletions of Parquet rows, over several tables of several catalogs, add no
/// relation to the database: every row lands in the two shared tables.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn inline_writes_create_no_relations() {
    const TABLES: usize = 4;
    let lake = Lake::new().await;
    let before = relation_count(&lake.pool).await;

    for table in 0..TABLES {
        let name = format!("t{table}");
        let writer = lake.table_writer(10).await;
        // Schema version 1, then 2 (ADD COLUMN), then 3 (DROP COLUMN).
        writer
            .append_table(
                "public",
                &name,
                &[int_batch(&["id", "val"], &[&[Some(1), Some(10)]])],
            )
            .await
            .unwrap();
        writer
            .append_table(
                "public",
                &name,
                &[int_batch(&["id", "val", "extra"], &[&[Some(2), Some(20), Some(200)]])],
            )
            .await
            .unwrap();
        writer
            .append_table(
                "public",
                &name,
                &[int_batch(&["id", "extra"], &[&[Some(3), Some(300)]])],
            )
            .await
            .unwrap();
        assert_eq!(
            lake.exec(
                &format!("UPDATE {CAT}.public.{name} SET extra = 7 WHERE id = 1"),
                10
            )
            .await,
            1
        );
        assert_eq!(
            lake.exec(&format!("DELETE FROM {CAT}.public.{name} WHERE id = 2"), 10)
                .await,
            1
        );
        assert_eq!(
            query(
                &lake,
                None,
                &format!("SELECT id, extra FROM {CAT}.public.{name} ORDER BY id")
            )
            .await,
            rows(&[&[Some(1), Some(7)], &[Some(3), Some(300)]])
        );
    }
    // An inlined deletion of a Parquet row.
    lake.table_writer(0)
        .await
        .append_table(
            "public",
            "p",
            &[int_batch(&["id", "val"], &[&[Some(1), Some(1)], &[Some(2), Some(2)]])],
        )
        .await
        .unwrap();
    assert_eq!(
        lake.exec(&format!("DELETE FROM {CAT}.public.p WHERE id = 1"), 10)
            .await,
        1
    );
    // Another catalog's writer.
    let other = MulticatalogManager::new(lake.pool.clone())
        .create_catalog("other")
        .await
        .unwrap();
    let other_writer = PostgresMetadataWriter::with_pool(lake.pool.clone(), other)
        .await
        .unwrap();
    other_writer
        .set_data_path(lake.data.to_str().unwrap())
        .unwrap();
    DuckLakeTableWriter::new(Arc::new(other_writer), Arc::new(LocalFileSystem::new()))
        .unwrap()
        .with_options(&DuckLakeWriteOptions::default().with_data_inlining_row_limit(10))
        .append_table("public", "t", &[batch(vec![1], vec![1])])
        .await
        .unwrap();

    let file_deletes: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM ducklake_inlined_file_delete")
        .fetch_one(&lake.pool)
        .await
        .unwrap();
    assert_eq!(file_deletes, 1);
    let schema_versions: i64 = sqlx::query_scalar(
        "SELECT COUNT(DISTINCT (table_id, schema_version)) FROM ducklake_inlined_row",
    )
    .fetch_one(&lake.pool)
    .await
    .unwrap();
    assert!(schema_versions >= 3 * TABLES as i64, "{schema_versions}");
    assert_eq!(relation_count(&lake.pool).await, before);
    let legacy: i64 = sqlx::query_scalar(
        r"SELECT COUNT(*) FROM pg_class
          WHERE relname LIKE 'ducklake\_inlined\_delete\_%'
             OR (relname LIKE 'ducklake\_inlined\_data\_%'
                 AND relname <> 'ducklake_inlined_data_tables')",
    )
    .fetch_one(&lake.pool)
    .await
    .unwrap();
    assert_eq!(legacy, 0);
}

/// Rows written under older schema versions read correctly after columns are
/// added, dropped and re-added: a value belongs to its column id, not to its
/// name, and a column a row predates reads as NULL. Time travel reads each
/// snapshot's own schema, and a flush to Parquet keeps the rows.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn rows_of_older_schema_versions_read_correctly() {
    let lake = Lake::new().await;
    let writer = lake.table_writer(10).await;
    writer
        .append_table(
            "public",
            "t",
            &[int_batch(
                &["id", "val"],
                &[&[Some(1), Some(10)], &[Some(2), Some(20)]],
            )],
        )
        .await
        .unwrap();
    let first = lake.head().await;
    writer
        .append_table(
            "public",
            "t",
            &[int_batch(&["id", "val", "extra"], &[&[Some(3), Some(30), Some(300)]])],
        )
        .await
        .unwrap();
    // An inline UPDATE of a row of the first schema version.
    assert_eq!(
        lake.exec(
            &format!("UPDATE {CAT}.public.t SET extra = 7 WHERE id = 1"),
            10
        )
        .await,
        1
    );
    let before_drop = lake.head().await;
    // DROP COLUMN val and extra, then ADD COLUMN val again (a new column id).
    writer
        .append_table("public", "t", &[int_batch(&["id"], &[&[Some(4)]])])
        .await
        .unwrap();
    writer
        .append_table(
            "public",
            "t",
            &[int_batch(&["id", "val"], &[&[Some(5), Some(55)]])],
        )
        .await
        .unwrap();
    assert_eq!(lake.live_files().await, (0, 0));

    let head_rows = rows(&[
        &[Some(1), None],
        &[Some(2), None],
        &[Some(3), None],
        &[Some(4), None],
        &[Some(5), Some(55)],
    ]);
    let sql = format!("SELECT id, val FROM {CAT}.public.t ORDER BY id");
    assert_eq!(query(&lake, None, &sql).await, head_rows);
    assert_eq!(
        query(
            &lake,
            Some(before_drop),
            &format!("SELECT id, val, extra FROM {CAT}.public.t ORDER BY id")
        )
        .await,
        rows(&[
            &[Some(1), Some(10), Some(7)],
            &[Some(2), Some(20), None],
            &[Some(3), Some(30), Some(300)],
        ])
    );
    assert_eq!(
        query(&lake, Some(first), &sql).await,
        rows(&[&[Some(1), Some(10)], &[Some(2), Some(20)]])
    );
    assert_eq!(
        query(&lake, None, &format!("SELECT COUNT(*) FROM {CAT}.public.t")).await,
        rows(&[&[Some(5)]])
    );
    // Updating a row of an older version writes it under the current one.
    assert_eq!(
        lake.exec(
            &format!("UPDATE {CAT}.public.t SET val = 22 WHERE id = 2"),
            10
        )
        .await,
        1
    );
    let mut expected = head_rows.clone();
    expected[1][1] = Some(22);
    assert_eq!(query(&lake, None, &sql).await, expected);
    lake.flush().await;
    assert_eq!(lake.live_files().await.0, 1);
    assert_eq!(query(&lake, None, &sql).await, expected);
}

/// A store written with the per-table layout (`ducklake_inlined_data_<id>_<sv>`
/// registered in `ducklake_inlined_data_tables`, and
/// `ducklake_inlined_delete_<id>`) is migrated on open: rows and deletions move
/// to the shared tables, the relations are dropped, and reads, UPDATE and
/// DELETE work on the moved rows. Concurrent and repeated initialization is
/// safe.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn per_table_layout_is_migrated_on_open() {
    let lake = Lake::new().await;
    let names = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true),
    ]));
    let parquet = RecordBatch::try_new(
        names,
        vec![
            Arc::new(Int32Array::from(vec![1, 2])),
            Arc::new(StringArray::from(vec![Some("a"), Some("b")])),
        ],
    )
    .unwrap();
    lake.table_writer(0)
        .await
        .append_table("public", "m", &[parquet])
        .await
        .unwrap();
    let table_id = lake.table_id("m").await;
    let head = lake.head().await;
    let schema_version: i64 =
        sqlx::query_scalar("SELECT schema_version FROM ducklake_snapshot WHERE snapshot_id = $1")
            .bind(head)
            .fetch_one(&lake.pool)
            .await
            .unwrap();
    let file_id: i64 =
        sqlx::query_scalar("SELECT data_file_id FROM ducklake_data_file WHERE table_id = $1")
            .bind(table_id)
            .fetch_one(&lake.pool)
            .await
            .unwrap();

    // What an earlier version wrote: an inlined data table per schema version
    // (strings as BYTEA) and an inlined delete table.
    let data_table = format!("ducklake_inlined_data_{table_id}_{schema_version}");
    let delete_table = format!("ducklake_inlined_delete_{table_id}");
    for statement in [
        format!(
            "CREATE TABLE \"{data_table}\" (row_id BIGINT, begin_snapshot BIGINT, \
             end_snapshot BIGINT, id INTEGER, name BYTEA)"
        ),
        format!(
            "INSERT INTO \"{data_table}\" VALUES \
             (2, {head}, NULL, 3, 'c\\000z'::BYTEA), (3, {head}, NULL, 4, NULL), \
             (4, {head}, {head}, 5, 'gone')"
        ),
        format!(
            "INSERT INTO ducklake_inlined_data_tables VALUES \
             ({table_id}, '{data_table}', {schema_version})"
        ),
        format!(
            "CREATE TABLE \"{delete_table}\" (file_id BIGINT, row_id BIGINT, begin_snapshot BIGINT)"
        ),
        format!("INSERT INTO \"{delete_table}\" VALUES ({file_id}, 0, {head})"),
        format!("UPDATE ducklake_table_stats SET next_row_id = 5 WHERE table_id = {table_id}"),
    ] {
        sqlx::query(sqlx::AssertSqlSafe(statement))
            .execute(&lake.pool)
            .await
            .unwrap();
    }

    let (first, second) = tokio::join!(
        datafusion_ducklake::initialize_multicatalog_schema(&lake.pool),
        datafusion_ducklake::initialize_multicatalog_schema(&lake.pool),
    );
    first.unwrap();
    second.unwrap();
    datafusion_ducklake::initialize_multicatalog_schema(&lake.pool)
        .await
        .unwrap();
    for relation in [&data_table, &delete_table] {
        let exists: bool = sqlx::query_scalar("SELECT to_regclass($1) IS NOT NULL")
            .bind(relation)
            .fetch_one(&lake.pool)
            .await
            .unwrap();
        assert!(!exists, "{relation} was not dropped");
    }
    let registered: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM ducklake_inlined_data_tables")
        .fetch_one(&lake.pool)
        .await
        .unwrap();
    assert_eq!(registered, 0);
    let moved: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM ducklake_inlined_row WHERE table_id = $1")
            .bind(table_id)
            .fetch_one(&lake.pool)
            .await
            .unwrap();
    assert_eq!(moved, 3, "the ended row version moves too");

    let read = |ctx: SessionContext| async move {
        let batches = ctx
            .sql(&format!("SELECT id, name FROM {CAT}.public.m ORDER BY id"))
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let mut out = Vec::new();
        for batch in &batches {
            let ids = arrow::compute::cast(batch.column(0), &DataType::Int32).unwrap();
            let ids = ids.as_any().downcast_ref::<Int32Array>().unwrap();
            let names = arrow::compute::cast(batch.column(1), &DataType::Utf8).unwrap();
            let names = names.as_any().downcast_ref::<StringArray>().unwrap();
            for row in 0..batch.num_rows() {
                out.push((
                    ids.value(row),
                    names.is_valid(row).then(|| names.value(row).to_string()),
                ));
            }
        }
        out
    };
    assert_eq!(
        read(lake.ctx(10).await).await,
        vec![(2, Some("b".to_string())), (3, Some("c\0z".to_string())), (4, None)]
    );
    assert_eq!(
        lake.exec(
            &format!("UPDATE {CAT}.public.m SET name = 'd' WHERE id = 4"),
            10
        )
        .await,
        1
    );
    assert_eq!(
        lake.exec(
            &format!("DELETE FROM {CAT}.public.m WHERE id IN (2, 3)"),
            10
        )
        .await,
        2
    );
    assert_eq!(
        read(lake.ctx(10).await).await,
        vec![(4, Some("d".to_string()))]
    );
    assert_eq!(lake.live_files().await, (1, 0));
    let writer = lake.writer().await;
    let backlog = writer.tables_with_inlined_file_deletes().unwrap();
    assert_eq!(backlog.len(), 1);
    assert_eq!(backlog[0].rows, 2);
}

fn median(samples: &mut [Duration]) -> Duration {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

/// Prints the median latency of a one-row inline append, a one-row inline
/// `UPDATE`, and a read of the table's inlined rows through the provider.
/// Run with `--nocapture` to see the figures; it asserts only correctness.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn inline_write_and_read_latency() {
    const ITERATIONS: usize = 40;
    let lake = Lake::new().await;
    lake.seed(batch((0..10).collect(), (0..10).collect()), 100)
        .await;
    let table_writer = lake.table_writer(1_000).await;

    let mut append = Vec::with_capacity(ITERATIONS);
    for index in 0..ITERATIONS {
        let id = 100 + index as i32;
        let started = Instant::now();
        table_writer
            .append_table("public", "t", &[batch(vec![id], vec![id])])
            .await
            .unwrap();
        append.push(started.elapsed());
    }

    let mut update = Vec::with_capacity(ITERATIONS);
    for _ in 0..ITERATIONS {
        let started = Instant::now();
        let changed = lake
            .exec(
                &format!("UPDATE {CAT}.public.t SET val = val + 1 WHERE id = 3"),
                1_000,
            )
            .await;
        update.push(started.elapsed());
        assert_eq!(changed, 1);
    }

    let provider = lake.provider().await;
    let head = provider.get_current_snapshot().unwrap();
    let schema = provider
        .get_schema_by_name("public", head)
        .unwrap()
        .unwrap();
    let table = provider
        .get_table_by_name(schema.schema_id, "t", head)
        .unwrap()
        .unwrap();
    let columns = provider.get_table_structure(table.table_id, head).unwrap();
    let mut read = Vec::with_capacity(ITERATIONS);
    for _ in 0..ITERATIONS {
        let started = Instant::now();
        let rows: usize = provider
            .get_inlined_data(table.table_id, head, &columns)
            .unwrap()
            .iter()
            .map(RecordBatch::num_rows)
            .sum();
        read.push(started.elapsed());
        assert_eq!(rows, 10 + ITERATIONS);
    }
    assert_eq!(lake.live_files().await, (0, 0));

    // A table of mixed types, to show the cost of the value encoding.
    let wide = mixed_batch(0..50);
    let mixed_writer = lake.table_writer(1_000).await;
    let mut mixed_append = Vec::with_capacity(ITERATIONS);
    for index in 0..ITERATIONS {
        let started = Instant::now();
        mixed_writer
            .append_table(
                "public",
                "mixed",
                &[mixed_batch(index as i64..index as i64 + 1)],
            )
            .await
            .unwrap();
        mixed_append.push(started.elapsed());
    }
    mixed_writer
        .append_table("public", "mixed", &[wide])
        .await
        .unwrap();
    let head = provider.get_current_snapshot().unwrap();
    let mixed = provider
        .get_table_by_name(schema.schema_id, "mixed", head)
        .unwrap()
        .unwrap();
    let mixed_columns = provider.get_table_structure(mixed.table_id, head).unwrap();
    let mut mixed_read = Vec::with_capacity(ITERATIONS);
    for _ in 0..ITERATIONS {
        let started = Instant::now();
        let rows: usize = provider
            .get_inlined_data(mixed.table_id, head, &mixed_columns)
            .unwrap()
            .iter()
            .map(RecordBatch::num_rows)
            .sum();
        mixed_read.push(started.elapsed());
        assert_eq!(rows, 50 + ITERATIONS);
    }
    assert_eq!(lake.live_files().await, (0, 0));
    println!(
        "inline latency (median of {ITERATIONS}): append 1 row {:?}, UPDATE 1 row {:?}, \
         read {} rows {:?}; 8 mixed columns: append 1 row {:?}, read {} rows {:?}",
        median(&mut append),
        median(&mut update),
        10 + ITERATIONS,
        median(&mut read),
        median(&mut mixed_append),
        50 + ITERATIONS,
        median(&mut mixed_read),
    );
}

fn mixed_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("ratio", DataType::Float64, true),
        Field::new("name", DataType::Utf8, true),
        Field::new("amount", DataType::Decimal128(18, 3), true),
        Field::new("day", DataType::Date32, true),
        Field::new(
            "at",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            true,
        ),
        Field::new("flag", DataType::Boolean, true),
        Field::new("raw", DataType::Binary, true),
    ]))
}

fn mixed_batch(ids: std::ops::Range<i64>) -> RecordBatch {
    let ids: Vec<i64> = ids.collect();
    RecordBatch::try_new(
        mixed_schema(),
        vec![
            Arc::new(Int64Array::from(ids.clone())),
            Arc::new(Float64Array::from(
                ids.iter().map(|id| *id as f64 / 3.0).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                ids.iter()
                    .map(|id| format!("name {id}"))
                    .collect::<Vec<_>>(),
            )),
            Arc::new(
                Decimal128Array::from(ids.iter().map(|id| *id as i128 * 1_001).collect::<Vec<_>>())
                    .with_precision_and_scale(18, 3)
                    .unwrap(),
            ),
            Arc::new(Date32Array::from(
                ids.iter().map(|id| 19_000 + *id as i32).collect::<Vec<_>>(),
            )),
            Arc::new(
                TimestampMicrosecondArray::from(
                    ids.iter()
                        .map(|id| 1_700_000_000_000_000 + *id)
                        .collect::<Vec<_>>(),
                )
                .with_timezone("UTC"),
            ),
            Arc::new(BooleanArray::from(
                ids.iter().map(|id| id % 2 == 0).collect::<Vec<_>>(),
            )),
            Arc::new(BinaryArray::from_iter_values(
                ids.iter().map(|id| id.to_le_bytes()),
            )),
        ],
    )
    .unwrap()
}

async fn stored_rows(pool: &PgPool, table_id: i64) -> (i64, i64) {
    sqlx::query_as(
        "SELECT COUNT(*), COUNT(*) FILTER (WHERE end_snapshot IS NULL)
         FROM ducklake_inlined_row WHERE table_id = $1",
    )
    .bind(table_id)
    .fetch_one(pool)
    .await
    .unwrap()
}

/// Expiring snapshots reclaims the row versions no surviving snapshot sees
/// and every row of a dropped table; dropping a catalog reclaims all of its
/// inlined rows and deletions, and the orphan purge those whose table is gone.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn expire_and_drop_reclaim_shared_rows() {
    use datafusion_ducklake::maintenance::ExpireCriteria;

    let lake = Lake::new().await;
    let manager = MulticatalogManager::new(lake.pool.clone());
    let everything = || ExpireCriteria::OlderThan(chrono::Utc::now() + chrono::Duration::days(1));
    lake.seed(batch(vec![1, 2], vec![10, 20]), 10).await;
    for value in [11, 12] {
        assert_eq!(
            lake.exec(
                &format!("UPDATE {CAT}.public.t SET val = {value} WHERE id = 1"),
                10
            )
            .await,
            1
        );
    }
    let t = lake.table_id("t").await;
    assert_eq!(stored_rows(&lake.pool, t).await, (4, 2));
    manager
        .expire_snapshots_in_catalog(CAT, everything())
        .await
        .unwrap();
    assert_eq!(stored_rows(&lake.pool, t).await, (2, 2));
    assert_eq!(lake.rows(None).await, vec![(1, Some(12)), (2, Some(20))]);

    // A dropped table's rows go once no snapshot sees the table.
    lake.table_writer(10)
        .await
        .append_table("public", "gone", &[batch(vec![1], vec![1])])
        .await
        .unwrap();
    let gone = lake.table_id("gone").await;
    assert!(
        manager
            .drop_table_in_catalog(CAT, "public", "gone")
            .await
            .unwrap()
    );
    assert_eq!(stored_rows(&lake.pool, gone).await, (1, 0));
    manager
        .expire_snapshots_in_catalog(CAT, everything())
        .await
        .unwrap();
    assert_eq!(stored_rows(&lake.pool, gone).await, (0, 0));
    assert_eq!(stored_rows(&lake.pool, t).await, (2, 2));

    // An inlined deletion of a Parquet row, then drop the whole catalog.
    lake.table_writer(0)
        .await
        .append_table("public", "p", &[batch(vec![1, 2], vec![1, 2])])
        .await
        .unwrap();
    assert_eq!(
        lake.exec(&format!("DELETE FROM {CAT}.public.p WHERE id = 1"), 10)
            .await,
        1
    );
    let deletes = || async {
        sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM ducklake_inlined_file_delete")
            .fetch_one(&lake.pool)
            .await
            .unwrap()
    };
    assert_eq!(deletes().await, 1);
    assert!(manager.drop_catalog(CAT).await.unwrap());
    assert_eq!(stored_rows(&lake.pool, t).await, (0, 0));
    assert_eq!(deletes().await, 0);

    // Rows whose table has no catalog row left are purged.
    sqlx::query(
        "INSERT INTO ducklake_inlined_row VALUES (987654, 0, 1, NULL, 1, '\\x0100000000'::BYTEA)",
    )
    .execute(&lake.pool)
    .await
    .unwrap();
    sqlx::query("INSERT INTO ducklake_inlined_file_delete VALUES (987654, 1, 0, 1)")
        .execute(&lake.pool)
        .await
        .unwrap();
    datafusion_ducklake::maintenance::purge_orphaned_metadata_postgres(&lake.pool)
        .await
        .unwrap();
    assert_eq!(stored_rows(&lake.pool, 987_654).await, (0, 0));
    assert_eq!(deletes().await, 0);
}
