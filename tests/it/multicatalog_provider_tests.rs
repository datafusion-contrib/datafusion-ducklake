#![cfg(all(feature = "multicatalog-postgres", feature = "write-postgres"))]
//! Integration tests for the multicatalog Postgres reader.
//!
//! These set up a populated catalog via the writer side, then verify the
//! reader's catalog-scoping with full isolation between catalogs.

use arrow::datatypes::{DataType, Field, Schema};
use datafusion::logical_expr::Operator;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_expr::expressions::{BinaryExpr, Column as PhysColumn, lit};
use datafusion_ducklake::metadata_provider::MetadataProvider;
use datafusion_ducklake::metadata_provider::{DuckLakeFileMetadata, DuckLakeTableColumn};
use datafusion_ducklake::metadata_writer::{ColumnDef, DataFileInfo, MetadataWriter, WriteMode};
use datafusion_ducklake::stats_filter::{StatsFilter, lower_predicate};
use datafusion_ducklake::{
    MulticatalogManager, MulticatalogProvider, PostgresMetadataWriter,
    initialize_multicatalog_schema,
};
use sqlx::postgres::{PgPool, PgPoolOptions};
use std::sync::Arc;
use testcontainers::ContainerAsync;
use testcontainers::runners::AsyncRunner;
use testcontainers_modules::postgres::Postgres;

async fn spin_up_postgres() -> anyhow::Result<(PgPool, ContainerAsync<Postgres>)> {
    let container = Postgres::default().start().await?;
    let port = container.get_host_port_ipv4(5432).await?;
    let conn = format!("postgresql://postgres:postgres@127.0.0.1:{}/postgres", port);
    let pool = PgPoolOptions::new()
        .max_connections(5)
        .connect(&conn)
        .await?;
    initialize_multicatalog_schema(&pool).await?;
    Ok((pool, container))
}

#[tokio::test(flavor = "multi_thread")]
async fn name_mapping_is_read_from_shared_metadata() -> anyhow::Result<()> {
    let (pool, _container) = spin_up_postgres().await?;
    sqlx::query(
        "CREATE TABLE ducklake_column_mapping (
            mapping_id BIGINT PRIMARY KEY,
            table_id BIGINT NOT NULL,
            type TEXT NOT NULL
        )",
    )
    .execute(&pool)
    .await?;
    sqlx::query(
        "CREATE TABLE ducklake_name_mapping (
            mapping_id BIGINT NOT NULL,
            column_id BIGINT NOT NULL,
            source_name TEXT NOT NULL,
            target_field_id BIGINT NOT NULL,
            parent_column BIGINT,
            is_partition BOOLEAN NOT NULL
        )",
    )
    .execute(&pool)
    .await?;
    sqlx::query(
        "INSERT INTO ducklake_column_mapping(mapping_id, table_id, type)
         VALUES (7, 11, 'map_by_name')",
    )
    .execute(&pool)
    .await?;
    sqlx::query(
        "INSERT INTO ducklake_name_mapping(
            mapping_id, column_id, source_name, target_field_id, parent_column, is_partition
         ) VALUES
            (7, 1, 'nested', 3, NULL, false),
            (7, 2, 'child', 4, 1, false),
            (7, 3, 'part', 5, NULL, true)",
    )
    .execute(&pool)
    .await?;
    let provider = MulticatalogProvider::with_pool_and_id(pool, 1).await?;

    let mapping = provider.get_name_mapping(7)?;
    assert_eq!(mapping.mapping_id, 7);
    assert_eq!(mapping.table_id, 11);
    assert_eq!(mapping.mapping_type, "map_by_name");
    assert_eq!(mapping.entries.len(), 3);
    assert!(mapping.entries[1].is_partition);
    assert_eq!(mapping.entries[2].parent_column, Some(1));
    Ok(())
}

fn users_cols() -> Vec<ColumnDef> {
    vec![
        ColumnDef::new("id", "int64", false).unwrap(),
        ColumnDef::new("name", "varchar", true).unwrap(),
    ]
}

fn orders_cols() -> Vec<ColumnDef> {
    vec![
        ColumnDef::new("order_id", "int64", false).unwrap(),
        ColumnDef::new("amount", "float64", false).unwrap(),
    ]
}

/// Set up the working-example world: pg_prod has users (3 snapshots: DDL, DML, DDL),
/// mysql_prod has orders (1 snapshot: DDL).
async fn seed_two_catalogs(pool: &PgPool) -> anyhow::Result<(i64, i64)> {
    let mgr = MulticatalogManager::new(pool.clone());
    let cat_pg = mgr.create_catalog("pg_prod").await?;
    let cat_mysql = mgr.create_catalog("mysql_prod").await?;
    let wa = PostgresMetadataWriter::with_pool(pool.clone(), cat_pg).await?;
    let wb = PostgresMetadataWriter::with_pool(pool.clone(), cat_mysql).await?;
    wa.set_data_path("/data")?;

    // Columns are written at the commit now (multicatalog Postgres), so the
    // register must carry the table/column generation, not `&[]`.
    let reg = |w: &PostgresMetadataWriter,
               table: &str,
               cols: &[ColumnDef],
               setup: datafusion_ducklake::metadata_writer::WriteSetupResult,
               fname: &str| {
        w.register_data_file(
            setup.table_id,
            "public",
            table,
            setup.snapshot_id,
            &DataFileInfo::new(fname, 1024, 3),
            WriteMode::Replace,
            setup.base_snapshot_id,
            cols,
            &setup.column_ids,
        )
        .unwrap();
    };

    // pg_prod: DDL, then DML
    let s1 = wa.begin_write_transaction("public", "users", &users_cols(), WriteMode::Replace)?;
    reg(&wa, "users", &users_cols(), s1, "users-a.parquet");
    let s2 = wa.begin_write_transaction("public", "users", &users_cols(), WriteMode::Replace)?;
    reg(&wa, "users", &users_cols(), s2, "users-b.parquet");
    // mysql_prod: DDL
    let s3 = wb.begin_write_transaction("public", "orders", &orders_cols(), WriteMode::Replace)?;
    reg(&wb, "orders", &orders_cols(), s3, "orders-a.parquet");
    // pg_prod: column-add DDL
    let mut v2 = users_cols();
    v2.push(ColumnDef::new("age", "int32", true).unwrap());
    let s4 = wa.begin_write_transaction("public", "users", &v2, WriteMode::Replace)?;
    reg(&wa, "users", &v2, s4, "users-c.parquet");
    Ok((cat_pg, cat_mysql))
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn with_pool_resolves_known_catalog_and_errors_on_unknown() {
    let (pool, _c) = spin_up_postgres().await.unwrap();
    let _ = seed_two_catalogs(&pool).await.unwrap();

    let p = MulticatalogProvider::with_pool(pool.clone(), "pg_prod")
        .await
        .unwrap();
    assert!(p.catalog_id() > 0);

    let err = MulticatalogProvider::with_pool(pool.clone(), "ghost_catalog").await;
    assert!(err.is_err(), "unknown catalog should error");
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn current_snapshot_is_catalog_scoped() {
    let (pool, _c) = spin_up_postgres().await.unwrap();
    let (_, _) = seed_two_catalogs(&pool).await.unwrap();

    let pa = MulticatalogProvider::with_pool(pool.clone(), "pg_prod")
        .await
        .unwrap();
    let pb = MulticatalogProvider::with_pool(pool.clone(), "mysql_prod")
        .await
        .unwrap();

    // pg_prod has snapshots 1, 2, 4 ⇒ max = 4
    // mysql_prod has snapshot 3 ⇒ max = 3
    assert_eq!(pa.get_current_snapshot().unwrap(), 4);
    assert_eq!(pb.get_current_snapshot().unwrap(), 3);
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn list_snapshots_is_catalog_scoped() {
    let (pool, _c) = spin_up_postgres().await.unwrap();
    let _ = seed_two_catalogs(&pool).await.unwrap();

    let pa = MulticatalogProvider::with_pool(pool.clone(), "pg_prod")
        .await
        .unwrap();
    let pb = MulticatalogProvider::with_pool(pool.clone(), "mysql_prod")
        .await
        .unwrap();

    let ids_a: Vec<i64> = pa
        .list_snapshots()
        .unwrap()
        .into_iter()
        .map(|s| s.snapshot_id)
        .collect();
    let ids_b: Vec<i64> = pb
        .list_snapshots()
        .unwrap()
        .into_iter()
        .map(|s| s.snapshot_id)
        .collect();
    assert_eq!(ids_a, vec![1, 2, 4]);
    assert_eq!(ids_b, vec![3]);
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn schemas_isolated_by_catalog_despite_same_name() {
    let (pool, _c) = spin_up_postgres().await.unwrap();
    let _ = seed_two_catalogs(&pool).await.unwrap();

    let pa = MulticatalogProvider::with_pool(pool.clone(), "pg_prod")
        .await
        .unwrap();
    let pb = MulticatalogProvider::with_pool(pool.clone(), "mysql_prod")
        .await
        .unwrap();

    let sn_a = pa.get_current_snapshot().unwrap();
    let sn_b = pb.get_current_snapshot().unwrap();

    // Each catalog sees exactly one schema named "public", with different ids.
    let schemas_a = pa.list_schemas(sn_a).unwrap();
    let schemas_b = pb.list_schemas(sn_b).unwrap();
    assert_eq!(schemas_a.len(), 1);
    assert_eq!(schemas_b.len(), 1);
    assert_eq!(schemas_a[0].schema_name, "public");
    assert_eq!(schemas_b[0].schema_name, "public");
    assert_ne!(
        schemas_a[0].schema_id, schemas_b[0].schema_id,
        "same name, different schema_ids (catalog discrimination)"
    );

    // get_schema_by_name agrees.
    let a = pa.get_schema_by_name("public", sn_a).unwrap().unwrap();
    let b = pb.get_schema_by_name("public", sn_b).unwrap().unwrap();
    assert_eq!(a.schema_id, schemas_a[0].schema_id);
    assert_eq!(b.schema_id, schemas_b[0].schema_id);

    // Catalog A should not see a schema from B's snapshot, and vice versa.
    // pg_prod doesn't have snapshot 3 — its schema lookup should still only
    // surface schemas it owns.
    let cross = pa.get_schema_by_name("public", sn_b).unwrap();
    // At snapshot 3 the pg_prod schema was visible (begin_snapshot=1),
    // so this returns the pg_prod schema — that's correct behavior.
    // The catalog scoping prevented us from seeing mysql_prod's schema.
    assert_eq!(cross.unwrap().schema_id, a.schema_id);
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn tables_visible_only_through_owning_catalog() {
    let (pool, _c) = spin_up_postgres().await.unwrap();
    let _ = seed_two_catalogs(&pool).await.unwrap();

    let pa = MulticatalogProvider::with_pool(pool.clone(), "pg_prod")
        .await
        .unwrap();
    let pb = MulticatalogProvider::with_pool(pool.clone(), "mysql_prod")
        .await
        .unwrap();
    let sn_a = pa.get_current_snapshot().unwrap();
    let sn_b = pb.get_current_snapshot().unwrap();

    let schema_a = pa.get_schema_by_name("public", sn_a).unwrap().unwrap();
    let schema_b = pb.get_schema_by_name("public", sn_b).unwrap().unwrap();

    // Each catalog's schema only has its own table.
    let tables_a = pa.list_tables(schema_a.schema_id, sn_a).unwrap();
    let tables_b = pb.list_tables(schema_b.schema_id, sn_b).unwrap();
    assert_eq!(tables_a.len(), 1);
    assert_eq!(tables_a[0].table_name, "users");
    assert_eq!(tables_b.len(), 1);
    assert_eq!(tables_b[0].table_name, "orders");

    // get_table_by_name agrees and table_exists confirms.
    assert!(pa.table_exists(schema_a.schema_id, "users", sn_a).unwrap());
    assert!(!pa.table_exists(schema_a.schema_id, "orders", sn_a).unwrap());
    assert!(pb.table_exists(schema_b.schema_id, "orders", sn_b).unwrap());
    assert!(!pb.table_exists(schema_b.schema_id, "users", sn_b).unwrap());
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn table_structure_reflects_latest_columns_after_ddl() {
    let (pool, _c) = spin_up_postgres().await.unwrap();
    let _ = seed_two_catalogs(&pool).await.unwrap();

    let pa = MulticatalogProvider::with_pool(pool.clone(), "pg_prod")
        .await
        .unwrap();
    let sn = pa.get_current_snapshot().unwrap();
    let schema = pa.get_schema_by_name("public", sn).unwrap().unwrap();
    let table = pa
        .get_table_by_name(schema.schema_id, "users", sn)
        .unwrap()
        .unwrap();

    let cols = pa.get_table_structure(table.table_id, sn).unwrap();
    // After the column-add DDL: id, name, age
    assert_eq!(cols.len(), 3);
    assert_eq!(cols[0].column_name, "id");
    assert_eq!(cols[1].column_name, "name");
    assert_eq!(cols[2].column_name, "age");
    assert_eq!(cols[2].column_type, "int32");
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn list_all_tables_bulk_is_catalog_scoped() {
    let (pool, _c) = spin_up_postgres().await.unwrap();
    let _ = seed_two_catalogs(&pool).await.unwrap();

    let pa = MulticatalogProvider::with_pool(pool.clone(), "pg_prod")
        .await
        .unwrap();
    let pb = MulticatalogProvider::with_pool(pool.clone(), "mysql_prod")
        .await
        .unwrap();

    let names_a: Vec<String> = pa
        .list_all_tables(pa.get_current_snapshot().unwrap())
        .unwrap()
        .into_iter()
        .map(|t| t.table.table_name)
        .collect();
    let names_b: Vec<String> = pb
        .list_all_tables(pb.get_current_snapshot().unwrap())
        .unwrap()
        .into_iter()
        .map(|t| t.table.table_name)
        .collect();
    assert_eq!(names_a, vec!["users"]);
    assert_eq!(names_b, vec!["orders"]);
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn list_all_columns_bulk_is_catalog_scoped() {
    let (pool, _c) = spin_up_postgres().await.unwrap();
    let _ = seed_two_catalogs(&pool).await.unwrap();

    let pa = MulticatalogProvider::with_pool(pool.clone(), "pg_prod")
        .await
        .unwrap();
    let pb = MulticatalogProvider::with_pool(pool.clone(), "mysql_prod")
        .await
        .unwrap();

    let cols_a = pa
        .list_all_columns(pa.get_current_snapshot().unwrap())
        .unwrap();
    let cols_b = pb
        .list_all_columns(pb.get_current_snapshot().unwrap())
        .unwrap();

    // pg_prod sees only users (3 cols: id, name, age)
    assert!(cols_a.iter().all(|c| c.table_name == "users"));
    assert_eq!(cols_a.len(), 3);

    // mysql_prod sees only orders (2 cols: order_id, amount)
    assert!(cols_b.iter().all(|c| c.table_name == "orders"));
    assert_eq!(cols_b.len(), 2);
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn list_all_files_bulk_is_catalog_scoped() {
    let (pool, _c) = spin_up_postgres().await.unwrap();
    let _ = seed_two_catalogs(&pool).await.unwrap();

    let pa = MulticatalogProvider::with_pool(pool.clone(), "pg_prod")
        .await
        .unwrap();
    let pb = MulticatalogProvider::with_pool(pool.clone(), "mysql_prod")
        .await
        .unwrap();

    let files_a = pa
        .list_all_files(pa.get_current_snapshot().unwrap())
        .unwrap();
    let files_b = pb
        .list_all_files(pb.get_current_snapshot().unwrap())
        .unwrap();

    // pg_prod sees only users files visible at snapshot 4 (just one)
    assert!(files_a.iter().all(|f| f.table_name == "users"));
    assert_eq!(files_a.len(), 1);

    // mysql_prod sees only orders files visible at snapshot 3 (just one)
    assert!(files_b.iter().all(|f| f.table_name == "orders"));
    assert_eq!(files_b.len(), 1);
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn get_table_files_for_select_returns_visible_files_at_snapshot() {
    let (pool, _c) = spin_up_postgres().await.unwrap();
    let _ = seed_two_catalogs(&pool).await.unwrap();

    let pa = MulticatalogProvider::with_pool(pool.clone(), "pg_prod")
        .await
        .unwrap();
    let sn = pa.get_current_snapshot().unwrap();
    let schema = pa.get_schema_by_name("public", sn).unwrap().unwrap();
    let table = pa
        .get_table_by_name(schema.schema_id, "users", sn)
        .unwrap()
        .unwrap();

    let files = pa.get_table_files_for_select(table.table_id, sn).unwrap();
    // At snapshot 4 only the latest data file is visible (older ones end_snapshot'd).
    assert_eq!(files.len(), 1);

    // At snapshot 1, the first file is visible.
    let files_at_1 = pa.get_table_files_for_select(table.table_id, 1).unwrap();
    assert_eq!(files_at_1.len(), 1);
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn get_table_files_for_select_returns_row_id_start_and_record_count() {
    // The DuckLake row-lineage read path (RowIdExec) needs row_id_start to
    // reconstruct rowids. The multicatalog reader used to drop these columns
    // on the floor, so any consumer using row lineage saw the file as if no
    // row_id_start were set — matching the single-catalog reader closes the
    // gap.
    let (pool, _c) = spin_up_postgres().await.unwrap();
    let (cat_pg, _) = seed_two_catalogs(&pool).await.unwrap();

    let pa = MulticatalogProvider::with_pool_and_id(pool.clone(), cat_pg)
        .await
        .unwrap();
    let sn = pa.get_current_snapshot().unwrap();
    let schema = pa.get_schema_by_name("public", sn).unwrap().unwrap();
    let table = pa
        .get_table_by_name(schema.schema_id, "users", sn)
        .unwrap()
        .unwrap();

    let files = pa.get_table_files_for_select(table.table_id, sn).unwrap();
    assert_eq!(files.len(), 1);
    let f = &files[0];
    // Three appends of 3 rows each (users-a, -b, -c) → the latest visible
    // file (users-c) starts at offset 6 and carries 3 rows.
    assert_eq!(f.row_id_start, Some(6), "row_id_start must be projected");
    assert_eq!(
        f.max_row_count,
        Some(3),
        "record_count must surface as max_row_count"
    );
    assert_eq!(f.snapshot_id, Some(sn), "snapshot_id is the query snapshot");
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn schema_capability_probe_memoized_after_first_select() {
    let (pool, _c) = spin_up_postgres().await.unwrap();
    let _ = seed_two_catalogs(&pool).await.unwrap();

    let pa = MulticatalogProvider::with_pool(pool.clone(), "pg_prod")
        .await
        .unwrap();
    let sn = pa.get_current_snapshot().unwrap();
    let schema = pa.get_schema_by_name("public", sn).unwrap().unwrap();
    let table = pa
        .get_table_by_name(schema.schema_id, "users", sn)
        .unwrap()
        .unwrap();

    // The multicatalog bootstrap schema is fully migrated, so the first scan
    // probes once (all capabilities true) and memoizes; the second scan reuses
    // the memo and must return identical results.
    assert!(!pa.schema_capabilities_cached());
    let first = pa.get_table_files_for_select(table.table_id, sn).unwrap();
    assert!(
        pa.schema_capabilities_cached(),
        "an all-true probe result must be cached after the first call"
    );
    let second = pa.get_table_files_for_select(table.table_id, sn).unwrap();
    assert_eq!(format!("{first:?}"), format!("{second:?}"));

    // Clones share the memo (the cell is Arc-shared).
    assert!(pa.clone().schema_capabilities_cached());
}

// ---------------------------------------------------------------------------
// Filter pushdown into the file-listing SQL
// ---------------------------------------------------------------------------

/// One catalog holding one table and five live files, each exercising a
/// different case of the statistics join. Returns the provider, the table id,
/// the snapshot to read at, and the file ids in listing order:
///
/// | file | statistics row       | must be         |
/// |------|----------------------|-----------------|
/// | 0    | none at all          | kept, LEFT JOIN |
/// | 1    | min `0`, max `10`    | kept (matches)  |
/// | 2    | min `100`, max `200` | pruned          |
/// | 3    | min `not-a-number`   | kept, fail-open |
/// | 4    | min/max NULL         | kept, fail-open |
///
/// The pruned file sits between two matching ones, which is what lets the
/// keyset-pagination assertion fail if the filter is ever applied after the
/// fetch instead of inside the query.
async fn seed_filter_catalog(
    pool: &PgPool,
) -> anyhow::Result<(MulticatalogProvider, i64, i64, Vec<i64>)> {
    let cols = vec![ColumnDef::new("id", "int64", true).unwrap()];
    let manager = MulticatalogManager::new(pool.clone());
    let catalog_id = manager.create_catalog("filter_cat").await?;
    let writer = PostgresMetadataWriter::with_pool(pool.clone(), catalog_id).await?;
    writer.set_data_path("/data")?;
    let setup = writer.begin_write_transaction("public", "t", &cols, WriteMode::Replace)?;
    writer.register_data_file(
        setup.table_id,
        "public",
        "t",
        setup.snapshot_id,
        &DataFileInfo::new("f0.parquet", 1024, 3),
        WriteMode::Replace,
        setup.base_snapshot_id,
        &cols,
        &setup.column_ids,
    )?;

    let provider = MulticatalogProvider::with_pool_and_id(pool.clone(), catalog_id).await?;
    let snapshot_id = provider.get_current_snapshot()?;
    let table_id = setup.table_id;
    let column_id = provider.get_table_structure(table_id, snapshot_id)?[0].column_id;

    let mut file_ids: Vec<i64> = sqlx::query_scalar(
        "SELECT data_file_id FROM ducklake_data_file WHERE table_id = $1 ORDER BY data_file_id",
    )
    .bind(table_id)
    .fetch_all(pool)
    .await?;
    // `data_file_id` is an identity column, so the ids are assigned here rather
    // than chosen; they come back in insertion order.
    for path in ["f1.parquet", "f2.parquet", "f3.parquet", "f4.parquet"] {
        file_ids.push(
            sqlx::query_scalar(
                "INSERT INTO ducklake_data_file
                     (table_id, path, path_is_relative, file_size_bytes, footer_size,
                      record_count, row_id_start, begin_snapshot)
                 VALUES ($1, $2, true, 1024, 1, 3, 0, $3)
                 RETURNING data_file_id",
            )
            .bind(table_id)
            .bind(path)
            .bind(snapshot_id)
            .fetch_one(pool)
            .await?,
        );
    }

    for (file_index, min_value, max_value) in [
        (1usize, Some("0"), Some("10")),
        (2, Some("100"), Some("200")),
        (3, Some("not-a-number"), Some("also-not-a-number")),
        (4, None, None),
    ] {
        sqlx::query(
            "INSERT INTO ducklake_file_column_stats
                 (data_file_id, table_id, column_id, column_size_bytes, value_count,
                  null_count, min_value, max_value, contains_nan)
             VALUES ($1, $2, $3, 8, 100, 0, $4, $5, NULL)",
        )
        .bind(file_ids[file_index])
        .bind(table_id)
        .bind(column_id)
        .bind(min_value)
        .bind(max_value)
        .execute(pool)
        .await?;
    }
    Ok((provider, table_id, snapshot_id, file_ids))
}

/// `id > 5 AND id < 10` over the seeded table's only column.
fn seeded_filter(column_id: i64) -> StatsFilter {
    let schema = Schema::new(vec![Field::new("id", DataType::Int64, true)]);
    let columns =
        [DuckLakeTableColumn::new(column_id, "id".to_string(), "int64".to_string(), true)];
    let id = Arc::new(PhysColumn::new("id", 0)) as Arc<dyn PhysicalExpr>;
    let predicate = Arc::new(BinaryExpr::new(
        Arc::new(BinaryExpr::new(Arc::clone(&id), Operator::Gt, lit(5i64))),
        Operator::And,
        Arc::new(BinaryExpr::new(id, Operator::Lt, lit(10i64))),
    )) as Arc<dyn PhysicalExpr>;
    lower_predicate(&predicate, &schema, &columns).expect("predicate lowers")
}

/// The multicatalog reader prunes on the same terms as the single-catalog one:
/// only a file whose recorded range proves it cannot match, never one whose
/// statistics are absent, incomplete or unreadable. A malformed `min_value`
/// would abort the query outright under a plain `CAST`, so this covers both
/// halves of the fail-open contract.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn file_listing_filter_prunes_only_on_provable_statistics() {
    let (pool, _c) = spin_up_postgres().await.unwrap();
    let (provider, table_id, snapshot_id, ids) = seed_filter_catalog(&pool).await.unwrap();
    let filter =
        seeded_filter(provider.get_table_structure(table_id, snapshot_id).unwrap()[0].column_id);

    let unfiltered = provider
        .get_table_file_metadata_page(table_id, snapshot_id, None, 4096)
        .unwrap();
    assert_eq!(listed_ids(&unfiltered), ids);

    let filtered = provider
        .get_table_file_metadata_page_filtered(table_id, snapshot_id, None, 4096, Some(&filter))
        .expect("a malformed stat must not fail the listing query");
    assert_eq!(
        listed_ids(&filtered),
        vec![ids[0], ids[1], ids[3], ids[4]],
        "only the file whose range 100..200 excludes 5 < id < 10 may be pruned"
    );
}

/// The filter goes inside the query, ahead of `LIMIT`, so the keyset cursor
/// still reaches every matching file. Applied after the fetch, the page landing
/// on the pruned file would come back empty and iteration would stop there.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn file_listing_filter_paginates_past_pruned_files() {
    let (pool, _c) = spin_up_postgres().await.unwrap();
    let (provider, table_id, snapshot_id, ids) = seed_filter_catalog(&pool).await.unwrap();
    let filter =
        seeded_filter(provider.get_table_structure(table_id, snapshot_id).unwrap()[0].column_id);

    let mut seen = Vec::new();
    let mut cursor = None;
    loop {
        let page = provider
            .get_table_file_metadata_page_filtered(table_id, snapshot_id, cursor, 1, Some(&filter))
            .unwrap();
        if page.is_empty() {
            break;
        }
        seen.extend(listed_ids(&page));
        cursor = seen.last().copied();
    }
    assert_eq!(seen, vec![ids[0], ids[1], ids[3], ids[4]]);
}

/// A catalog with no `ducklake_file_column_stats` cannot be joined against it,
/// so the filter is dropped and every file is listed rather than the scan
/// failing.
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn file_listing_filter_falls_open_on_a_catalog_without_statistics() {
    let (pool, _c) = spin_up_postgres().await.unwrap();
    let (provider, table_id, snapshot_id, ids) = seed_filter_catalog(&pool).await.unwrap();
    let filter =
        seeded_filter(provider.get_table_structure(table_id, snapshot_id).unwrap()[0].column_id);
    sqlx::query("DROP TABLE ducklake_file_column_stats")
        .execute(&pool)
        .await
        .expect("drop the statistics table");

    let filtered = provider
        .get_table_file_metadata_page_filtered(table_id, snapshot_id, None, 4096, Some(&filter))
        .expect("a legacy catalog must still list its files");
    assert_eq!(listed_ids(&filtered), ids);
}

fn listed_ids(files: &[DuckLakeFileMetadata]) -> Vec<i64> {
    files.iter().map(|file| file.file.data_file_id).collect()
}
