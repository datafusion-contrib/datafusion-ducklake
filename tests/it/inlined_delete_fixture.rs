#![cfg(all(feature = "write-sqlite", feature = "metadata-sqlite"))]

use std::path::Path;

use sqlx::sqlite::SqlitePool;
use sqlx::{AssertSqlSafe, Row};

pub(crate) async fn insert_inlined_deletes_for_first_file(db_path: &Path, row_ids: &[i64]) -> i64 {
    let pool = SqlitePool::connect(&format!("sqlite:{}?mode=rwc", db_path.display()))
        .await
        .unwrap();
    let row = sqlx::query(
        "SELECT t.table_id, MIN(f.data_file_id), MAX(s.snapshot_id) \
         FROM ducklake_table t \
         JOIN ducklake_data_file f ON f.table_id = t.table_id \
         CROSS JOIN ducklake_snapshot s \
         WHERE t.table_name = 't' AND t.end_snapshot IS NULL AND f.end_snapshot IS NULL",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    let table_id: i64 = row.try_get(0).unwrap();
    let data_file_id: i64 = row.try_get(1).unwrap();
    let snapshot_id: i64 = row.try_get(2).unwrap();
    insert_inlined_deletes(&pool, table_id, data_file_id, snapshot_id, row_ids).await;
    data_file_id
}

pub(crate) async fn insert_inlined_deletes_for_only_file(db_path: &Path, row_ids: &[i64]) -> i64 {
    let pool = SqlitePool::connect(&format!("sqlite:{}?mode=rwc", db_path.display()))
        .await
        .unwrap();
    let rows = sqlx::query(
        "SELECT t.table_id, f.data_file_id, max(s.snapshot_id) \
         FROM ducklake_table t \
         JOIN ducklake_data_file f ON f.table_id = t.table_id \
         CROSS JOIN ducklake_snapshot s \
         WHERE t.table_name = 't' AND t.end_snapshot IS NULL AND f.end_snapshot IS NULL \
         GROUP BY t.table_id, f.data_file_id",
    )
    .fetch_all(&pool)
    .await
    .unwrap();
    assert_eq!(rows.len(), 1);
    let row = &rows[0];
    let table_id: i64 = row.try_get(0).unwrap();
    let data_file_id: i64 = row.try_get(1).unwrap();
    let snapshot_id: i64 = row.try_get(2).unwrap();
    insert_inlined_deletes(&pool, table_id, data_file_id, snapshot_id, row_ids).await;
    data_file_id
}

async fn insert_inlined_deletes(
    pool: &SqlitePool,
    table_id: i64,
    data_file_id: i64,
    snapshot_id: i64,
    row_ids: &[i64],
) {
    let table = format!("ducklake_inlined_delete_{table_id}");

    let create_sql =
        format!("CREATE TABLE {table}(file_id BIGINT, row_id BIGINT, begin_snapshot BIGINT)");
    sqlx::query(AssertSqlSafe(create_sql.as_str()))
        .execute(pool)
        .await
        .unwrap();
    for row_id in row_ids {
        let insert_sql = format!("INSERT INTO {table} VALUES (?, ?, ?)");
        sqlx::query(AssertSqlSafe(insert_sql.as_str()))
            .bind(data_file_id)
            .bind(row_id)
            .bind(snapshot_id)
            .execute(pool)
            .await
            .unwrap();
    }
}
