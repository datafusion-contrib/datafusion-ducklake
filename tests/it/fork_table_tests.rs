//! `MulticatalogManager::fork_table`: a table forked into another catalog of a
//! multi-catalog PostgreSQL store reads what the source read at its snapshot,
//! inlined rows and inlined deletions included, keeps the source's row ids, and
//! changes independently of it afterwards.
//!
//! A seeded random history (`FORK_FUZZ_SEEDS`, default 4) mutates a source
//! through every write path, forks it at a random step, then mutates both and
//! checks each against a model after every step.
//!
//! Every scenario runs in one test, `fork_table_scenarios`, on one PostgreSQL
//! container, each on catalogs of its own.

#![cfg(feature = "write-postgres")]
#![cfg(feature = "write-sqlite")]

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow::array::{Array as _, Int64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::prelude::SessionContext;
use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeError, DuckLakeTable, MetadataProvider, MulticatalogManager,
};

use crate::inlining_abstraction_tests::{Lake, MakeLake, batch, create, lines, multicatalog};

async fn store() -> (
    testcontainers::ContainerAsync<testcontainers_modules::postgres::Postgres>,
    sqlx::PgPool,
) {
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
    (container, pool)
}

/// The source table as read at `snapshot`.
async fn read_handle(lake: &Lake, snapshot: i64) -> (SessionContext, DuckLakeTable) {
    let ctx = SessionContext::new();
    let catalog = DuckLakeCatalog::with_snapshot(Arc::clone(&lake.provider), snapshot).unwrap();
    ctx.register_catalog("src", Arc::new(catalog));
    let provider = ctx
        .catalog("src")
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

async fn fork(
    manager: &MulticatalogManager,
    source: &Lake,
    snapshot: i64,
    destination: &Lake,
) -> Result<datafusion_ducklake::TableFork, DuckLakeError> {
    let (ctx, table) = read_handle(source, snapshot).await;
    manager
        .fork_table(
            &ctx.state(),
            &table,
            Arc::clone(&destination.writer),
            "main",
            "t",
        )
        .await
}

/// A source with every kind of storage: two Parquet files, one with a delete
/// file and inlined deletions on top of it, and inlined rows.
async fn rich_source(make: &MakeLake) -> Lake {
    let lake = make(Some(5)).await;
    create(&lake, "t").await;
    lake.append_parquet("t", &batch(1..=20, "a")).await;
    lake.append_parquet("t", &batch(30..=40, "b")).await;
    lake.sql("DELETE FROM lake.main.t WHERE id <= 8")
        .await
        .unwrap();
    lake.sql("DELETE FROM lake.main.t WHERE id IN (15, 16)")
        .await
        .unwrap();
    lake.append("t", &batch(100..=102, "i")).await;
    assert_eq!(lake.delete_files("t"), 1);
    assert_eq!(lake.inlined_deletes("t"), 2);
    assert_eq!(lake.inlined_rows("t"), 3);
    lake
}

async fn fork_carries_inlined_rows_and_deletions(pool: sqlx::PgPool) {
    let manager = MulticatalogManager::new(pool.clone());
    let make = multicatalog(pool);
    let source = rich_source(&make).await;
    let early = source.head() - 2;
    let expected = source.rows("t").await;

    // A fork that referenced the source's files alone would read rows 15 and
    // 16 (deleted inlined) and miss 100-102 (inlined rows).
    let destination = make(None).await;
    let forked = fork(&manager, &source, source.head(), &destination)
        .await
        .unwrap();
    assert_eq!(forked.files_referenced, 2);
    assert_eq!(forked.inlined_rows, 3);
    assert_eq!(forked.delete_files_written, 1);
    assert_eq!(
        destination.rows("t").await,
        expected,
        "the fork reads the source's rows with their row ids"
    );

    // Each changes on its own afterwards.
    let (ctx, table) = source.table("t").await;
    table
        .keyed_mutation(&["id"])
        .delete_by_keys(&ctx.state(), vec![batch([100, 30], "")])
        .await
        .unwrap();
    let (ctx, table) = destination.table("t").await;
    table
        .keyed_mutation(&["id"])
        .upsert_by_keys(&ctx.state(), vec![batch([20, 500], "f")])
        .await
        .unwrap();
    let ids = |rows: &[String]| {
        rows.iter()
            .map(|row| row.split(',').nth(1).unwrap().parse::<i64>().unwrap())
            .collect::<Vec<_>>()
    };
    let mut source_ids = ids(&source.rows("t").await);
    source_ids.sort_unstable();
    assert!(!source_ids.contains(&100) && !source_ids.contains(&30) && !source_ids.contains(&500));
    let fork_rows = destination.rows("t").await;
    assert!(fork_rows.iter().any(|row| row.ends_with(",500,f500")));
    assert!(fork_rows.iter().any(|row| row.ends_with(",20,f20")));
    assert!(fork_rows.iter().any(|row| row.ends_with(",100,i100")));

    // The source's maintenance leaves the fork readable.
    let ctx = SessionContext::new();
    source
        .catalog()
        .flush_inlined_data(&ctx.state(), None, None)
        .await
        .unwrap();
    assert_eq!(destination.rows("t").await, fork_rows);

    // A fork of an earlier snapshot reads what the source read then.
    let historical = make(None).await;
    fork(&manager, &source, early, &historical).await.unwrap();
    assert_eq!(
        historical.rows("t").await,
        source.rows_at("t", Some(early)).await
    );

    // A fork into the source's own catalog is refused.
    let refused = fork(&manager, &source, source.head(), &source).await;
    assert!(matches!(refused, Err(DuckLakeError::InvalidConfig(_))));
}

// ---------------------------------------------------------------------------
// Fork fuzz
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// One random change to `lake`'s table `t`, applied to `model` too.
async fn mutate(lake: &Lake, model: &mut BTreeMap<i64, String>, rng: &mut Rng, next: &mut i64) {
    let existing: Vec<i64> = model.keys().copied().collect();
    let pick = |rng: &mut Rng, count: u64| -> Vec<i64> {
        if existing.is_empty() {
            return Vec::new();
        }
        (0..count)
            .map(|_| existing[rng.below(existing.len() as u64) as usize])
            .collect()
    };
    match rng.below(7) {
        0 | 1 => {
            let count = if rng.below(2) == 0 {
                3
            } else {
                12
            };
            let ids: Vec<i64> = (*next..*next + count).collect();
            *next += count;
            let tag = format!("n{}_", rng.below(1000));
            lake.append("t", &batch(ids.iter().copied(), &tag)).await;
            for id in ids {
                model.insert(id, format!("{tag}{id}"));
            }
        },
        2 => {
            let count = 1 + rng.below(8);
            let ids = pick(rng, count);
            let (ctx, table) = lake.table("t").await;
            table
                .keyed_mutation(&["id"])
                .delete_by_keys(&ctx.state(), vec![batch(ids.iter().copied(), "")])
                .await
                .unwrap();
            for id in ids {
                model.remove(&id);
            }
        },
        3 => {
            let count = 1 + rng.below(8);
            let mut ids = pick(rng, count);
            ids.push(*next);
            *next += 1;
            ids.sort_unstable();
            ids.dedup();
            let tag = format!("u{}_", rng.below(1000));
            let (ctx, table) = lake.table("t").await;
            table
                .keyed_mutation(&["id"])
                .upsert_by_keys(&ctx.state(), vec![batch(ids.iter().copied(), &tag)])
                .await
                .unwrap();
            for id in ids {
                model.insert(id, format!("{tag}{id}"));
            }
        },
        4 => {
            let count = 1 + rng.below(4);
            let mut ids = pick(rng, count);
            ids.sort_unstable();
            ids.dedup();
            if ids.is_empty() {
                return;
            }
            let tag = format!("w{}_", rng.below(1000));
            let (ctx, table) = lake.table("t").await;
            table
                .keyed_mutation(&["id"])
                .update_by_keys(&ctx.state(), vec![batch(ids.iter().copied(), &tag)])
                .await
                .unwrap();
            for id in ids {
                model.insert(id, format!("{tag}{id}"));
            }
        },
        5 => {
            if let Some(&id) = existing.first() {
                lake.sql(&format!("DELETE FROM lake.main.t WHERE id <= {}", id + 2))
                    .await
                    .unwrap();
                model.retain(|key, _| *key > id + 2);
            }
        },
        _ => {
            let ctx = SessionContext::new();
            lake.catalog()
                .flush_inlined_data(&ctx.state(), None, None)
                .await
                .unwrap();
        },
    }
}

fn model_lines(rows: &[String]) -> BTreeMap<i64, String> {
    rows.iter()
        .map(|row| {
            let cells: Vec<&str> = row.split(',').collect();
            (cells[1].parse().unwrap(), cells[2].to_string())
        })
        .collect()
}

async fn fork_fuzz(pool: sqlx::PgPool) {
    let seeds: u64 = std::env::var("FORK_FUZZ_SEEDS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(4);
    let manager = MulticatalogManager::new(pool.clone());
    let make = multicatalog(pool);
    for seed in 0..seeds {
        let mut rng = Rng(seed ^ 0x5EED);
        let source = make(Some(5)).await;
        create(&source, "t").await;
        let mut model = BTreeMap::new();
        let mut next = 1;
        let steps = 6 + rng.below(6);
        let fork_at = rng.below(steps);
        let mut forked: Option<(Lake, BTreeMap<i64, String>)> = None;
        for step in 0..steps {
            mutate(&source, &mut model, &mut rng, &mut next).await;
            assert_eq!(
                model_lines(&source.rows("t").await),
                model,
                "seed {seed} step {step}: source"
            );
            if let Some((fork_lake, fork_model)) = forked.as_mut() {
                mutate(fork_lake, fork_model, &mut rng, &mut next).await;
                assert_eq!(
                    model_lines(&fork_lake.rows("t").await),
                    *fork_model,
                    "seed {seed} step {step}: fork"
                );
            }
            if step == fork_at {
                let destination = make(Some(5)).await;
                fork(&manager, &source, source.head(), &destination)
                    .await
                    .unwrap_or_else(|e| panic!("seed {seed}: fork: {e}"));
                assert_eq!(
                    destination.rows("t").await,
                    source.rows("t").await,
                    "seed {seed}: the fork reads the source with its row ids"
                );
                forked = Some((destination, model.clone()));
            }
        }
    }
}

#[allow(dead_code)]
fn _schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, true)]))
}

#[allow(dead_code)]
fn _ids(batch: &RecordBatch) -> Vec<i64> {
    batch
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .values()
        .to_vec()
}

#[allow(dead_code)]
fn _lines(batches: &[RecordBatch]) -> Vec<String> {
    lines(batches, true)
}

#[allow(dead_code)]
fn _provider(lake: &Lake) -> Arc<dyn MetadataProvider> {
    Arc::clone(&lake.provider)
}

// ---------------------------------------------------------------------------
// What a fork carries and refuses
// ---------------------------------------------------------------------------

/// Live `ducklake_column` rows of `table_id`, as `(column_name, column_type,
/// nulls_allowed, initial_default, default_value, default_value_type,
/// default_value_dialect, parent name)`, ordered by name path.
async fn column_metadata(pool: &sqlx::PgPool, table_id: i64) -> Vec<String> {
    sqlx::query_scalar(
        "SELECT format('%s|%s|%s|%s|%s|%s|%s|%s', c.column_name, c.column_type,
                       c.nulls_allowed, c.initial_default, c.default_value,
                       c.default_value_type, c.default_value_dialect, p.column_name)
         FROM ducklake_column c
         LEFT JOIN ducklake_column p
           ON p.table_id = c.table_id AND p.column_id = c.parent_column
          AND p.end_snapshot IS NULL
         WHERE c.table_id = $1 AND c.end_snapshot IS NULL
         ORDER BY c.column_id",
    )
    .bind(table_id)
    .fetch_all(pool)
    .await
    .unwrap()
}

/// A fork's columns are the source's catalog rows (defaults, exact types such
/// as `json`, nested nullability) and it carries the source's comment and
/// tags. Guards against rebuilding the columns from the Arrow schema, which
/// drops all of them.
async fn fork_copies_column_metadata_and_tags(pool: sqlx::PgPool) {
    use datafusion_ducklake::metadata_writer::ColumnDef;
    use datafusion_ducklake::{TagObjectType, TagTarget, WriteMode};
    let manager = MulticatalogManager::new(pool.clone());
    let make = multicatalog(pool.clone());
    let source = make(Some(5)).await;
    let nested = DataType::Struct(vec![Field::new("a", DataType::Int64, true)].into());
    let columns = vec![
        ColumnDef::from_arrow("id", &DataType::Int64, true).unwrap(),
        ColumnDef::from_arrow("v", &DataType::Utf8, true)
            .unwrap()
            .with_default("dflt")
            .unwrap(),
        ColumnDef::from_arrow("s", &nested, true).unwrap(),
    ];
    let setup = source
        .writer
        .begin_write_transaction("main", "t", &columns, WriteMode::Replace)
        .unwrap();
    source
        .writer
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
    let source_id = source.table_id("t");
    // A JSON column and a NOT NULL struct field, which an Arrow schema cannot
    // tell apart from VARCHAR and a nullable field.
    sqlx::query(
        "UPDATE ducklake_column SET column_type = 'json'
         WHERE table_id = $1 AND column_name = 'v'",
    )
    .bind(source_id)
    .execute(&pool)
    .await
    .unwrap();
    sqlx::query(
        "UPDATE ducklake_column SET nulls_allowed = false
         WHERE table_id = $1 AND column_name = 'a'",
    )
    .bind(source_id)
    .execute(&pool)
    .await
    .unwrap();
    source
        .writer
        .set_tag(
            TagTarget::Object {
                object_type: TagObjectType::Table,
                object_id: source_id,
            },
            "comment",
            Some("orders"),
        )
        .unwrap();
    let v_id: i64 = sqlx::query_scalar(
        "SELECT column_id FROM ducklake_column
         WHERE table_id = $1 AND column_name = 'v' AND end_snapshot IS NULL",
    )
    .bind(source_id)
    .fetch_one(&pool)
    .await
    .unwrap();
    source
        .writer
        .set_tag(
            TagTarget::Column {
                table_id: source_id,
                column_id: v_id,
            },
            "comment",
            Some("payload"),
        )
        .unwrap();

    let destination = make(None).await;
    fork(&manager, &source, source.head(), &destination)
        .await
        .unwrap();
    let fork_id = destination.table_id("t");
    assert_eq!(
        column_metadata(&pool, fork_id).await,
        column_metadata(&pool, source_id).await
    );
    let tags = |lake: &Lake, target: TagTarget| {
        lake.provider
            .get_tags(target, lake.head())
            .unwrap()
            .into_iter()
            .map(|tag| (tag.key, tag.value))
            .collect::<Vec<_>>()
    };
    let table_tags = |lake: &Lake, id: i64| {
        tags(
            lake,
            TagTarget::Object {
                object_type: TagObjectType::Table,
                object_id: id,
            },
        )
    };
    assert_eq!(
        table_tags(&destination, fork_id),
        vec![("comment".to_string(), Some("orders".to_string()))]
    );
    assert_eq!(
        tags(
            &destination,
            TagTarget::Column {
                table_id: fork_id,
                column_id: v_id,
            }
        ),
        vec![("comment".to_string(), Some("payload".to_string()))]
    );
}

/// A partitioned source forks, merged files included: the provider's file
/// pages carry each file's partition generation. Guards against a page that
/// reads every file's generation as NULL, which refused every partitioned
/// source.
async fn fork_of_a_partitioned_source(pool: sqlx::PgPool) {
    let manager = MulticatalogManager::new(pool.clone());
    let make = multicatalog(pool);
    let source = make(Some(0)).await;
    create(&source, "t").await;
    source
        .sql("ALTER TABLE lake.main.t SET PARTITIONED BY (v)")
        .await
        .unwrap();
    // Two appends of the same two partitions: two files per partition.
    let partitioned = |ids: [i64; 4]| {
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("id", DataType::Int64, true),
                Field::new("v", DataType::Utf8, true),
            ])),
            vec![
                Arc::new(Int64Array::from(ids.to_vec())),
                Arc::new(arrow::array::StringArray::from(vec!["p", "q", "p", "q"])),
            ],
        )
        .unwrap()
    };
    source.append_parquet("t", &partitioned([1, 2, 3, 4])).await;
    source.append_parquet("t", &partitioned([5, 6, 7, 8])).await;
    assert_eq!(source.data_files("t"), 4);
    let destination = make(None).await;
    fork(&manager, &source, source.head(), &destination)
        .await
        .unwrap();
    assert_eq!(destination.rows("t").await, source.rows("t").await);

    // After a merge too.
    let (ctx, table) = source.table("t").await;
    let merged = table
        .merge_adjacent_files(&ctx.state(), datafusion_ducklake::MergeOptions::default())
        .await
        .unwrap();
    assert_eq!(
        merged.files_created, 2,
        "one file per partition: {merged:?}"
    );
    assert_eq!(source.data_files("t"), 2);
    let after_merge = make(None).await;
    fork(&manager, &source, source.head(), &after_merge)
        .await
        .unwrap();
    assert_eq!(after_merge.rows("t").await, source.rows("t").await);
}

/// A merge of adjacent files keeps positional row ids (no embedded row id
/// column) and records each row's snapshot; forked at a snapshot between the
/// two inserts, the fork masks the merged file's later rows, as the source's
/// scan does. Guards against masking only the rows of files that embed row
/// ids.
async fn historical_fork_of_a_merged_positional_file(pool: sqlx::PgPool) {
    let manager = MulticatalogManager::new(pool.clone());
    let make = multicatalog(pool);
    let source = make(Some(0)).await;
    create(&source, "t").await;
    source.append_parquet("t", &batch(1..=3, "a")).await;
    let first = source.head();
    source.append_parquet("t", &batch(4..=6, "b")).await;
    let (ctx, table) = source.table("t").await;
    let merged = table
        .merge_adjacent_files(&ctx.state(), datafusion_ducklake::MergeOptions::default())
        .await
        .unwrap();
    assert_eq!(merged.files_created, 1, "{merged:?}");
    let expected = source.rows_at("t", Some(first)).await;
    assert_eq!(expected.len(), 3);
    let (ctx, table) = read_handle(&source, first).await;
    let files = table.files().unwrap();
    assert_eq!(files.len(), 1);
    assert!(files[0].partial_max.is_some_and(|max| max > first));
    assert!(matches!(
        table
            .file_row_id_layout(&ctx.state(), &files[0])
            .await
            .unwrap(),
        Some(datafusion_ducklake::RowIdLayout::Positional { .. })
    ));
    let destination = make(None).await;
    let forked = fork(&manager, &source, first, &destination).await.unwrap();
    assert_eq!(forked.delete_files_written, 1, "{forked:?}");
    assert_eq!(destination.rows("t").await, expected);
}

/// A source that dropped a column after a file was written forks: the file's
/// statistics for the dropped column are left behind. Guards against handing
/// the registration statistics for a column the fork does not have.
async fn fork_after_a_dropped_column(pool: sqlx::PgPool) {
    let manager = MulticatalogManager::new(pool.clone());
    let make = multicatalog(pool.clone());
    let source = make(Some(0)).await;
    crate::inlining_abstraction_tests::create_with(
        &source,
        "t",
        &[("id", DataType::Int64), ("v", DataType::Utf8), ("w", DataType::Int64)],
    )
    .unwrap();
    let rows = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, true),
            Field::new("v", DataType::Utf8, true),
            Field::new("w", DataType::Int64, true),
        ])),
        vec![
            Arc::new(Int64Array::from(vec![1, 2])),
            Arc::new(arrow::array::StringArray::from(vec!["a", "b"])),
            Arc::new(Int64Array::from(vec![10, 20])),
        ],
    )
    .unwrap();
    source.append_parquet("t", &rows).await;
    // The catalog state official's `ALTER TABLE t DROP COLUMN w` leaves: the
    // column's row ends, the file written before keeps its statistics.
    sqlx::query(
        "UPDATE ducklake_column SET end_snapshot = $1
         WHERE table_id = $2 AND column_name = 'w'",
    )
    .bind(source.head())
    .bind(source.table_id("t"))
    .execute(&pool)
    .await
    .unwrap();
    let destination = make(None).await;
    fork(&manager, &source, source.head(), &destination)
        .await
        .unwrap();
    assert_eq!(destination.rows("t").await, source.rows("t").await);
}

/// A source with an encrypted file, or one read through a name mapping, is
/// refused with a typed error: a fork carries neither the key nor the
/// mapping.
async fn fork_refuses_encrypted_and_name_mapped_sources(pool: sqlx::PgPool) {
    let manager = MulticatalogManager::new(pool.clone());
    let make = multicatalog(pool.clone());
    for column in ["encryption_key = 'secret'", "mapping_id = 1"] {
        let source = make(Some(0)).await;
        create(&source, "t").await;
        source.append_parquet("t", &batch(1..=3, "a")).await;
        sqlx::query(sqlx::AssertSqlSafe(format!(
            "UPDATE ducklake_data_file SET {column} WHERE table_id = $1"
        )))
        .bind(source.table_id("t"))
        .execute(&pool)
        .await
        .unwrap();
        let destination = make(None).await;
        let refused = fork(&manager, &source, source.head(), &destination).await;
        assert!(
            matches!(refused, Err(DuckLakeError::Unsupported(_))),
            "{column}: {refused:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// Forks into an existing destination
// ---------------------------------------------------------------------------

/// `(schema_version, changes_made)` of snapshot `snapshot`.
async fn snapshot_record(pool: &sqlx::PgPool, snapshot: i64) -> (i64, String) {
    sqlx::query_as(
        "SELECT s.schema_version, COALESCE(c.changes_made, '')
         FROM ducklake_snapshot s
         LEFT JOIN ducklake_snapshot_changes c ON c.snapshot_id = s.snapshot_id
         WHERE s.snapshot_id = $1",
    )
    .bind(snapshot)
    .fetch_one(pool)
    .await
    .unwrap()
}

/// A fork replacing an earlier fork whose tags differ records them as
/// `set_tag` does: the snapshot bumps `schema_version` and records
/// `altered_table`; a fork with the same tags changes neither. A column tag
/// other than `comment`, which `set_tag` refuses, is refused before anything
/// is written. Guards against tag changes no reader's cache sees.
async fn fork_replace_records_tag_changes(pool: sqlx::PgPool) {
    use datafusion_ducklake::{TagObjectType, TagTarget};
    let manager = MulticatalogManager::new(pool.clone());
    let make = multicatalog(pool.clone());
    let source = make(Some(0)).await;
    create(&source, "t").await;
    source.append_parquet("t", &batch(1..=3, "a")).await;
    let source_id = source.table_id("t");
    let comment = |value: &str| {
        source
            .writer
            .set_tag(
                TagTarget::Object {
                    object_type: TagObjectType::Table,
                    object_id: source_id,
                },
                "comment",
                Some(value),
            )
            .unwrap();
    };
    comment("one");
    let destination = make(None).await;
    let first = fork(&manager, &source, source.head(), &destination)
        .await
        .unwrap();
    let fork_id = destination.table_id("t");
    let (first_version, _) = snapshot_record(&pool, first.commit.snapshot_id).await;

    comment("two");
    let second = fork(&manager, &source, source.head(), &destination)
        .await
        .unwrap();
    let (version, changes) = snapshot_record(&pool, second.commit.snapshot_id).await;
    assert_eq!(version, first_version + 1, "{changes}");
    assert!(
        changes.contains(&format!("altered_table:{fork_id}")),
        "{changes}"
    );
    let tags = destination
        .provider
        .get_tags(
            TagTarget::Object {
                object_type: TagObjectType::Table,
                object_id: fork_id,
            },
            destination.head(),
        )
        .unwrap();
    assert_eq!(tags.len(), 1);
    assert_eq!(tags[0].value.as_deref(), Some("two"));

    let third = fork(&manager, &source, source.head(), &destination)
        .await
        .unwrap();
    let (unchanged, changes) = snapshot_record(&pool, third.commit.snapshot_id).await;
    assert_eq!(unchanged, version, "{changes}");
    assert!(!changes.contains("altered_table"), "{changes}");

    let v_id: i64 = sqlx::query_scalar(
        "SELECT column_id FROM ducklake_column
         WHERE table_id = $1 AND column_name = 'v' AND end_snapshot IS NULL",
    )
    .bind(source_id)
    .fetch_one(&pool)
    .await
    .unwrap();
    let catalog_id: i64 = sqlx::query_scalar(
        "SELECT m.catalog_id FROM ducklake_table t
         JOIN ducklake_catalog_schema_map m ON m.schema_id = t.schema_id
         WHERE t.table_id = $1",
    )
    .bind(source_id)
    .fetch_one(&pool)
    .await
    .unwrap();
    sqlx::query(
        "INSERT INTO ducklake_catalog_column_tag
             (catalog_id, table_id, column_id, begin_snapshot, end_snapshot, key, value)
         VALUES ($1, $2, $3, $4, NULL, 'owner', 'x')",
    )
    .bind(catalog_id)
    .bind(source_id)
    .bind(v_id)
    .bind(source.head())
    .execute(&pool)
    .await
    .unwrap();
    let head = destination.head();
    let refused = fork(&manager, &source, source.head(), &destination).await;
    assert!(
        matches!(refused, Err(DuckLakeError::InvalidConfig(_))),
        "{refused:?}"
    );
    assert_eq!(destination.head(), head, "nothing committed");
}

/// A fork into a table created in the destination on its own, sharing a
/// column name with the source but not its column id, is refused up front
/// with a typed error naming the column: the fork's files carry the source's
/// field ids, which `Replace` keeps for a column of the same name, so no
/// retry could succeed. Guards against a `Conflict` asking for a retry.
async fn fork_into_an_independent_destination_is_refused(pool: sqlx::PgPool) {
    let manager = MulticatalogManager::new(pool.clone());
    let make = multicatalog(pool);
    let source = make(Some(0)).await;
    create(&source, "t").await;
    source.append_parquet("t", &batch(1..=3, "a")).await;
    let destination = make(None).await;
    crate::inlining_abstraction_tests::create_with(
        &destination,
        "t",
        &[("v", DataType::Utf8), ("w", DataType::Int64)],
    )
    .unwrap();
    destination
        .append_parquet("t", &{
            RecordBatch::try_new(
                Arc::new(Schema::new(vec![
                    Field::new("v", DataType::Utf8, true),
                    Field::new("w", DataType::Int64, true),
                ])),
                vec![
                    Arc::new(arrow::array::StringArray::from(vec!["d"])),
                    Arc::new(Int64Array::from(vec![1])),
                ],
            )
            .unwrap()
        })
        .await;
    let before = destination.rows("t").await;
    let head = destination.head();
    let refused = fork(&manager, &source, source.head(), &destination).await;
    match refused {
        Err(DuckLakeError::InvalidConfig(message)) => {
            assert!(message.contains("'v'"), "{message}");
        },
        other => panic!("{other:?}"),
    }
    assert_eq!(destination.head(), head);
    assert_eq!(destination.rows("t").await, before);
}

/// A source whose list column has the legacy catalog shape (one
/// `list<...>` row, no `element` row) forks into upstream's shape, as a
/// write to the source would migrate it: a `list` row with an `element`
/// child under a fresh id, which a re-fork into the same table keeps. Guards
/// against an internal error on a catalog node count mismatch, and against a
/// re-fork giving `element` another id, which the destination refuses.
async fn fork_of_a_legacy_list_column(pool: sqlx::PgPool) {
    use arrow::array::ListArray;
    use arrow::datatypes::Int64Type;
    let manager = MulticatalogManager::new(pool.clone());
    let make = multicatalog(pool.clone());
    let source = make(Some(5)).await;
    let values = ListArray::from_iter_primitive::<Int64Type, _, _>(vec![
        Some(vec![Some(1), Some(2)]),
        None,
        Some(vec![Some(3)]),
    ]);
    crate::inlining_abstraction_tests::create_with(
        &source,
        "t",
        &[("id", DataType::Int64), ("l", values.data_type().clone())],
    )
    .unwrap();
    let rows = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, true),
            Field::new("l", values.data_type().clone(), true),
        ])),
        vec![Arc::new(Int64Array::from(vec![1, 2, 3])), Arc::new(values)],
    )
    .unwrap();
    source.append_parquet("t", &rows).await;
    let source_id = source.table_id("t");
    sqlx::query("DELETE FROM ducklake_column WHERE table_id = $1 AND column_name = 'element'")
        .bind(source_id)
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query(
        "UPDATE ducklake_column SET column_type = 'list<int64>'
         WHERE table_id = $1 AND column_name = 'l'",
    )
    .bind(source_id)
    .execute(&pool)
    .await
    .unwrap();
    let expected = source.rows("t").await;
    assert_eq!(expected.len(), 3);

    let destination = make(None).await;
    fork(&manager, &source, source.head(), &destination)
        .await
        .unwrap();
    assert_eq!(destination.rows("t").await, expected);
    let shape: Vec<(String, String, Option<String>)> = sqlx::query_as(
        "SELECT c.column_name, c.column_type, p.column_name
         FROM ducklake_column c
         LEFT JOIN ducklake_column p
           ON p.table_id = c.table_id AND p.column_id = c.parent_column
          AND p.end_snapshot IS NULL
         WHERE c.table_id = $1 AND c.end_snapshot IS NULL
         ORDER BY c.column_order",
    )
    .bind(destination.table_id("t"))
    .fetch_all(&pool)
    .await
    .unwrap();
    assert_eq!(
        shape,
        vec![
            ("id".to_string(), "int64".to_string(), None),
            ("l".to_string(), "list".to_string(), None),
            (
                "element".to_string(),
                "int64".to_string(),
                Some("l".to_string())
            ),
        ]
    );
    // A re-fork into the same table replaces it, its `element` keeping the
    // id the first fork gave it.
    let element_id = |pool: sqlx::PgPool, table_id: i64| async move {
        sqlx::query_scalar::<_, i64>(
            "SELECT column_id FROM ducklake_column
             WHERE table_id = $1 AND column_name = 'element' AND end_snapshot IS NULL",
        )
        .bind(table_id)
        .fetch_one(&pool)
        .await
        .unwrap()
    };
    let first_element = element_id(pool.clone(), destination.table_id("t")).await;
    fork(&manager, &source, source.head(), &destination)
        .await
        .unwrap();
    assert_eq!(destination.rows("t").await, expected);
    assert_eq!(
        element_id(pool.clone(), destination.table_id("t")).await,
        first_element
    );
    // The fork takes new rows.
    let more = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, true),
            Field::new(
                "l",
                DataType::List(Arc::new(Field::new_list_field(DataType::Int64, true))),
                true,
            ),
        ])),
        vec![
            Arc::new(Int64Array::from(vec![4])),
            Arc::new(ListArray::from_iter_primitive::<Int64Type, _, _>(vec![
                Some(vec![Some(9)]),
            ])),
        ],
    )
    .unwrap();
    destination.append_parquet("t", &more).await;
    assert_eq!(destination.rows("t").await.len(), 4);
}

/// A list of structs in the legacy catalog shape (one
/// `list<struct<...>>` row, no `element` or field rows) forks into
/// upstream's shape: a `list` row, a `struct` `element` under it and a row per
/// field under that, each under a fresh id, with the source's rows. Guards
/// against migrating only a list of scalars.
async fn fork_of_a_legacy_list_of_structs(pool: sqlx::PgPool) {
    use arrow::array::{ArrayRef, ListArray, StringArray, StructArray};
    use arrow::buffer::OffsetBuffer;
    use arrow::datatypes::Fields;
    let manager = MulticatalogManager::new(pool.clone());
    let make = multicatalog(pool.clone());
    let source = make(Some(5)).await;
    let fields = Fields::from(vec![
        Field::new("a", DataType::Int64, true),
        Field::new("b", DataType::Utf8, true),
    ]);
    let structs = StructArray::new(
        fields.clone(),
        vec![
            Arc::new(Int64Array::from(vec![Some(1), None, Some(3)])) as ArrayRef,
            Arc::new(StringArray::from(vec![Some("x"), Some("y"), None])),
        ],
        None,
    );
    let item = Arc::new(Field::new("element", DataType::Struct(fields), true));
    let values = ListArray::new(
        Arc::clone(&item),
        OffsetBuffer::from_lengths([2, 0, 1]),
        Arc::new(structs),
        Some(vec![true, false, true].into()),
    );
    crate::inlining_abstraction_tests::create_with(
        &source,
        "t",
        &[("id", DataType::Int64), ("l", values.data_type().clone())],
    )
    .unwrap();
    let rows = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, true),
            Field::new("l", values.data_type().clone(), true),
        ])),
        vec![Arc::new(Int64Array::from(vec![1, 2, 3])), Arc::new(values)],
    )
    .unwrap();
    source.append_parquet("t", &rows).await;
    let source_id = source.table_id("t");
    // A file of the legacy shape tags its top-level columns alone.
    let file: String = sqlx::query_scalar(
        "SELECT path FROM ducklake_data_file WHERE table_id = $1 AND end_snapshot IS NULL",
    )
    .bind(source_id)
    .fetch_one(&pool)
    .await
    .unwrap();
    let file = std::fs::read_dir(&source.data)
        .unwrap()
        .flatten()
        .flat_map(|entry| walk(entry.path()))
        .find(|path| {
            path.to_string_lossy()
                .ends_with(file.trim_start_matches('/'))
        })
        .unwrap();
    let top_level: Vec<i64> = sqlx::query_scalar(
        "SELECT column_id FROM ducklake_column
         WHERE table_id = $1 AND parent_column IS NULL AND end_snapshot IS NULL
         ORDER BY column_order",
    )
    .bind(source_id)
    .fetch_all(&pool)
    .await
    .unwrap();
    write_top_level_tagged(&file, &rows, &top_level);
    sqlx::query(
        "UPDATE ducklake_data_file SET file_size_bytes = $1, footer_size = NULL
         WHERE table_id = $2",
    )
    .bind(std::fs::metadata(&file).unwrap().len() as i64)
    .bind(source_id)
    .execute(&pool)
    .await
    .unwrap();
    sqlx::query("DELETE FROM ducklake_column WHERE table_id = $1 AND parent_column IS NOT NULL")
        .bind(source_id)
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query(
        "UPDATE ducklake_column SET column_type = 'list<struct<a:int64,b:varchar>>'
         WHERE table_id = $1 AND column_name = 'l'",
    )
    .bind(source_id)
    .execute(&pool)
    .await
    .unwrap();
    let expected = source.rows("t").await;
    assert_eq!(expected.len(), 3);

    let destination = make(None).await;
    fork(&manager, &source, source.head(), &destination)
        .await
        .unwrap();
    assert_eq!(destination.rows("t").await, expected);
    let shape: Vec<(String, String, Option<String>)> = sqlx::query_as(
        "SELECT c.column_name, c.column_type, p.column_name
         FROM ducklake_column c
         LEFT JOIN ducklake_column p
           ON p.table_id = c.table_id AND p.column_id = c.parent_column
          AND p.end_snapshot IS NULL
         WHERE c.table_id = $1 AND c.end_snapshot IS NULL
         ORDER BY c.column_id",
    )
    .bind(destination.table_id("t"))
    .fetch_all(&pool)
    .await
    .unwrap();
    let parent = |name: &str| Some(name.to_string());
    assert_eq!(
        shape,
        vec![
            ("id".to_string(), "int64".to_string(), None),
            ("l".to_string(), "list".to_string(), None),
            ("element".to_string(), "struct".to_string(), parent("l")),
            ("a".to_string(), "int64".to_string(), parent("element")),
            ("b".to_string(), "varchar".to_string(), parent("element")),
        ]
    );
    // A re-fork into the same table replaces it and reads the same rows.
    fork(&manager, &source, source.head(), &destination)
        .await
        .unwrap();
    assert_eq!(destination.rows("t").await, expected);
}

/// Every file under `path`.
fn walk(path: std::path::PathBuf) -> Vec<std::path::PathBuf> {
    if path.is_dir() {
        std::fs::read_dir(&path)
            .unwrap()
            .flatten()
            .flat_map(|entry| walk(entry.path()))
            .collect()
    } else {
        vec![path]
    }
}

/// Write `rows` to the Parquet file at `path`, tagging its top-level columns
/// alone with `field_ids`, as a file of a legacy nested column holds them.
fn write_top_level_tagged(path: &std::path::Path, rows: &RecordBatch, field_ids: &[i64]) {
    let fields: Vec<Field> = rows
        .schema()
        .fields()
        .iter()
        .zip(field_ids)
        .map(|(field, id)| {
            field
                .as_ref()
                .clone()
                .with_metadata([("PARQUET:field_id".to_string(), id.to_string())].into())
        })
        .collect();
    let schema = Arc::new(Schema::new(fields));
    let mut writer = parquet::arrow::ArrowWriter::try_new(
        std::fs::File::create(path).unwrap(),
        Arc::clone(&schema),
        None,
    )
    .unwrap();
    writer
        .write(&RecordBatch::try_new(schema, rows.columns().to_vec()).unwrap())
        .unwrap();
    writer.close().unwrap();
}

/// A fork copies the catalog rows of every column shape (a `geometry`, a
/// `timetz`, a list, a map, a struct) and their tags, a nested field's
/// comment among them, from a source holding rows; a later fork replacing it
/// ends the tags the source no longer has.
async fn fork_copies_metadata_of_every_column_shape(pool: sqlx::PgPool) {
    use arrow::array::{BinaryArray, ListArray, MapArray, StructArray};
    use arrow::datatypes::Int64Type;
    use datafusion_ducklake::{TagObjectType, TagTarget};
    let manager = MulticatalogManager::new(pool.clone());
    let make = multicatalog(pool.clone());
    let source = make(Some(0)).await;
    let list =
        ListArray::from_iter_primitive::<Int64Type, _, _>(vec![Some(vec![Some(1)]), Some(vec![])]);
    let map = MapArray::new_from_strings(
        ["a", "b", "c"].into_iter(),
        &Int64Array::from(vec![1, 2, 3]),
        &[0, 2, 3],
    )
    .unwrap();
    let nested_fields: arrow::datatypes::Fields =
        vec![Field::new("a", DataType::Int64, true)].into();
    let nested = StructArray::new(
        nested_fields.clone(),
        vec![Arc::new(Int64Array::from(vec![5, 6]))],
        None,
    );
    let fields = vec![
        Field::new("id", DataType::Int64, true),
        Field::new("g", DataType::Binary, true),
        Field::new("tz", DataType::Utf8, true),
        Field::new("l", list.data_type().clone(), true),
        Field::new("m", map.data_type().clone(), true),
        Field::new("s", DataType::Struct(nested_fields), true),
    ];
    crate::inlining_abstraction_tests::create_with(
        &source,
        "t",
        &fields
            .iter()
            .map(|field| (field.name().as_str(), field.data_type().clone()))
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let rows = RecordBatch::try_new(
        Arc::new(Schema::new(fields)),
        vec![
            Arc::new(Int64Array::from(vec![1, 2])),
            Arc::new(BinaryArray::from(vec![
                &[1u8, 1, 0, 0, 0][..],
                &[1u8, 2, 0, 0, 0][..],
            ])),
            Arc::new(arrow::array::StringArray::from(vec![
                "12:00:00+00",
                "13:30:00+05:30",
            ])),
            Arc::new(list),
            Arc::new(map),
            Arc::new(nested),
        ],
    )
    .unwrap();
    source.append_parquet("t", &rows).await;
    let source_id = source.table_id("t");
    for (column, column_type) in [("g", "geometry"), ("tz", "timetz")] {
        sqlx::query(
            "UPDATE ducklake_column SET column_type = $1
             WHERE table_id = $2 AND column_name = $3",
        )
        .bind(column_type)
        .bind(source_id)
        .bind(column)
        .execute(&pool)
        .await
        .unwrap();
    }
    let column_id = |name: &str| {
        let pool = pool.clone();
        let name = name.to_string();
        async move {
            sqlx::query_scalar::<_, i64>(
                "SELECT column_id FROM ducklake_column
                 WHERE table_id = $1 AND column_name = $2 AND end_snapshot IS NULL",
            )
            .bind(source_id)
            .bind(name)
            .fetch_one(&pool)
            .await
            .unwrap()
        }
    };
    let nested_a = column_id("a").await;
    let map_id = column_id("m").await;
    for (target, value) in [
        (
            TagTarget::Object {
                object_type: TagObjectType::Table,
                object_id: source_id,
            },
            Some("shapes"),
        ),
        (
            TagTarget::Column {
                table_id: source_id,
                column_id: nested_a,
            },
            Some("nested"),
        ),
        (
            TagTarget::Column {
                table_id: source_id,
                column_id: map_id,
            },
            Some("pairs"),
        ),
    ] {
        source.writer.set_tag(target, "comment", value).unwrap();
    }
    let expected = source.rows("t").await;

    let destination = make(None).await;
    fork(&manager, &source, source.head(), &destination)
        .await
        .unwrap();
    let fork_id = destination.table_id("t");
    assert_eq!(
        column_metadata(&pool, fork_id).await,
        column_metadata(&pool, source_id).await
    );
    assert_eq!(destination.rows("t").await, expected);
    let column_tags = |lake: &Lake, table_id: i64| {
        lake.provider
            .list_all_column_tags(lake.head())
            .unwrap()
            .into_iter()
            .filter(|tag| tag.table_id == table_id)
            .map(|tag| (tag.column_id, tag.tag.key, tag.tag.value))
            .collect::<std::collections::BTreeSet<_>>()
    };
    assert_eq!(
        column_tags(&destination, fork_id),
        column_tags(&source, source_id)
    );
    assert_eq!(column_tags(&destination, fork_id).len(), 2);

    // The source drops the map's comment; a fork replacing the first ends it.
    source
        .writer
        .set_tag(
            TagTarget::Column {
                table_id: source_id,
                column_id: map_id,
            },
            "comment",
            None,
        )
        .unwrap();
    fork(&manager, &source, source.head(), &destination)
        .await
        .unwrap();
    let after: Vec<_> = column_tags(&destination, fork_id)
        .into_iter()
        .filter(|(_, _, value)| value.is_some())
        .collect();
    assert_eq!(
        after,
        vec![(nested_a, "comment".to_string(), Some("nested".to_string()))]
    );
    assert_eq!(destination.rows("t").await, expected);
}

/// The change feed of a fork dates each row as a merge of the fork's files
/// would: the source's inlined rows, written to a file of the fork's own that
/// embeds their insert snapshots, by those snapshots, as official reads a
/// file embedding them; the referenced Parquet rows by the fork's snapshot.
/// The fork's file records `row_id_start` and `partial_max` as official's
/// flush does. Guards against a fork's file without `partial_max`, whose rows
/// the feed dated to the fork's snapshot until a merge re-dated them.
async fn fork_change_feed_survives_a_merge(pool: sqlx::PgPool) {
    let manager = MulticatalogManager::new(pool.clone());
    let make = multicatalog(pool.clone());
    let source = make(Some(5)).await;
    create(&source, "t").await;
    source.append_parquet("t", &batch(1..=3, "a")).await;
    source.append("t", &batch([10, 11], "i")).await;
    let first_inlined = source.head();
    source.append("t", &batch([12], "j")).await;
    let second_inlined = source.head();
    let destination = make(None).await;
    let forked = fork(&manager, &source, source.head(), &destination)
        .await
        .unwrap();
    assert_eq!(forked.inlined_files_written, 1);
    let forked_at = destination.head();
    let table_id = destination.table_id("t");
    let own_file: (Option<i64>, Option<i64>, i64) = sqlx::query_as(
        "SELECT row_id_start, partial_max, begin_snapshot FROM ducklake_data_file
         WHERE table_id = $1 AND end_snapshot IS NULL AND partial_max IS NOT NULL",
    )
    .bind(table_id)
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(own_file, (Some(3), Some(second_inlined), forked_at));

    let changes = async |lake: &Lake| {
        let ctx = SessionContext::new();
        datafusion_ducklake::register_ducklake_functions(&ctx, Arc::clone(&lake.provider));
        let batches = ctx
            .sql(&format!(
                "SELECT snapshot_id, rowid, change_type, id \
                 FROM ducklake_table_changes('main.t', 0, {forked_at}) ORDER BY rowid"
            ))
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        lines(&batches, false)
    };
    let before = changes(&destination).await;
    assert_eq!(
        before,
        vec![
            format!("{forked_at},0,insert,1"),
            format!("{forked_at},1,insert,2"),
            format!("{forked_at},2,insert,3"),
            format!("{first_inlined},3,insert,10"),
            format!("{first_inlined},4,insert,11"),
            format!("{second_inlined},5,insert,12"),
        ]
    );
    let (ctx, table) = destination.table("t").await;
    let merged = table
        .merge_adjacent_files(&ctx.state(), datafusion_ducklake::MergeOptions::default())
        .await
        .unwrap();
    assert_eq!(merged.files_processed, 2, "{merged:?}");
    assert_eq!(changes(&destination).await, before);
}

/// A fork's own file of inlined rows with a gap in its row ids (an inlined
/// row deleted before the fork) records no `row_id_start`, so an adjacent
/// merge of the fork's files keeps every row's `rowid`. Guards against a
/// `row_id_start` claiming a contiguous range the file's ids do not fill,
/// which a merge renumbers the rows into.
async fn fork_with_row_id_gaps_keeps_row_ids_through_a_merge(pool: sqlx::PgPool) {
    let manager = MulticatalogManager::new(pool.clone());
    let make = multicatalog(pool.clone());
    let source = make(Some(10)).await;
    create(&source, "t").await;
    source.append_parquet("t", &batch(1..=5, "a")).await;
    source.append("t", &batch(10..=14, "i")).await;
    source
        .sql("DELETE FROM lake.main.t WHERE id = 11")
        .await
        .unwrap();
    let destination = make(None).await;
    fork(&manager, &source, source.head(), &destination)
        .await
        .unwrap();
    let before = destination.rows("t").await;
    assert_eq!(before, source.rows("t").await);
    let (ctx, table) = destination.table("t").await;
    let merged = table
        .merge_adjacent_files(&ctx.state(), datafusion_ducklake::MergeOptions::default())
        .await
        .unwrap();
    assert_eq!(merged.files_processed, 2, "{merged:?}");
    assert_eq!(
        destination.rows("t").await,
        before,
        "every row keeps its rowid"
    );
}

/// A fork's own file of inlined rows whose row ids are contiguous as a set
/// but not in the order the rows were read (an `UPDATE` re-inlined the first
/// row at a later snapshot, keeping its row id) keeps every row's `rowid`,
/// value and change feed through an adjacent merge of the fork's files: the
/// rows are written by `(row_id, begin_snapshot)` as official's flush orders
/// them, and `row_id_start` is recorded only when each position's row id is
/// it plus the position. Guards against checking contiguity as a set over rows
/// in write order, which made a merge renumber the rows by position.
async fn fork_of_updated_inlined_rows_keeps_row_ids_through_a_merge(pool: sqlx::PgPool) {
    let manager = MulticatalogManager::new(pool.clone());
    let make = multicatalog(pool.clone());
    let source = make(Some(10)).await;
    create(&source, "t").await;
    source.append_parquet("t", &batch(10..=12, "a")).await;
    source.append("t", &batch([1, 2, 3], "i")).await;
    source
        .sql("UPDATE lake.main.t SET v = 'u1' WHERE id = 1")
        .await
        .unwrap();
    assert_eq!(source.inlined_rows("t"), 3);
    let expected = source.rows("t").await;
    let destination = make(None).await;
    let forked = fork(&manager, &source, source.head(), &destination)
        .await
        .unwrap();
    assert_eq!(forked.inlined_files_written, 1);
    assert_eq!(destination.rows("t").await, expected);
    let forked_at = destination.head();

    let changes = async |lake: &Lake| {
        let ctx = SessionContext::new();
        datafusion_ducklake::register_ducklake_functions(&ctx, Arc::clone(&lake.provider));
        let batches = ctx
            .sql(&format!(
                "SELECT snapshot_id, rowid, change_type, id, v \
                 FROM ducklake_table_changes('main.t', 0, {forked_at}) ORDER BY rowid"
            ))
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        lines(&batches, false)
    };
    let feed = changes(&destination).await;
    let (ctx, table) = destination.table("t").await;
    let merged = table
        .merge_adjacent_files(&ctx.state(), datafusion_ducklake::MergeOptions::default())
        .await
        .unwrap();
    assert_eq!(merged.files_processed, 2, "{merged:?}");
    assert_eq!(
        destination.rows("t").await,
        expected,
        "every row keeps its rowid and value"
    );
    assert_eq!(changes(&destination).await, feed);
}

/// The `id`s of each live data file of `main.t` in `lake`, in file order,
/// files by `data_file_id`.
fn file_ids(lake: &Lake) -> Vec<Vec<i64>> {
    let mut files = lake
        .provider
        .get_table_files_for_select(lake.table_id("t"), lake.head())
        .unwrap();
    files.sort_by_key(|file| file.data_file_id);
    files
        .iter()
        .map(|file| {
            let name = std::path::Path::new(&file.file.path)
                .file_name()
                .unwrap()
                .to_owned();
            let path = walk(lake.data.clone())
                .into_iter()
                .find(|path| path.file_name() == Some(name.as_os_str()))
                .expect("data file on disk");
            parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(
                std::fs::File::open(path).unwrap(),
            )
            .unwrap()
            .build()
            .unwrap()
            .flat_map(|batch| {
                let batch = batch.unwrap();
                batch
                    .column_by_name("id")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .values()
                    .to_vec()
            })
            .collect()
        })
        .collect()
}

/// A fork writes its own files of the source's inlined rows as official's
/// flush writes them: by the table's sort keys (here `id DESC`, then `v`,
/// inlined in ascending id order with `sort_on_insert` off), then by row id,
/// rolling over into further files at the target file size at the flush's
/// batch boundaries. The fork's files hold the same rows in the same files, with
/// the same `row_id_start`, row count and `partial_max`, as a flush of the
/// source. Guards against writing them by row id alone, and against a NULL
/// `row_id_start` for a file whose rows the sort keys reordered or whose ids
/// the rollover interleaved.
async fn fork_writes_inlined_rows_as_a_flush_sorts_them(pool: sqlx::PgPool) {
    // `id DESC` reverses each file's id range; `v` (`'i1'`, `'i10'`, ...)
    // interleaves the ids of the files, leaving gaps in each one's range.
    let by_v = |id: &i64| format!("i{id}");
    for key in ["id DESC", "v"] {
        let mut order: Vec<i64> = (1..=20_000).collect();
        if key == "v" {
            order.sort_by_key(by_v);
        } else {
            order.reverse();
        }
        fork_writes_inlined_rows_sorted_by(pool.clone(), key, order).await;
    }
}

async fn fork_writes_inlined_rows_sorted_by(pool: sqlx::PgPool, key: &str, order: Vec<i64>) {
    let manager = MulticatalogManager::new(pool.clone());
    let make = multicatalog(pool.clone());
    let source = make(Some(50_000)).await;
    create(&source, "t").await;
    source
        .sql(&format!("ALTER TABLE lake.main.t SET SORTED BY ({key})"))
        .await
        .unwrap();
    let table_id = source.table_id("t");
    source
        .writer
        .set_table_setting(table_id, "sort_on_insert", "false")
        .unwrap();
    source.append("t", &batch(1..=20_000, "i")).await;
    assert_eq!(source.inlined_rows("t"), 20_000);
    let expected = source.rows("t").await;

    // The source with a small target file size and row groups.
    let mut options = datafusion_ducklake::DuckLakeWriteOptions::default();
    options.target_file_size = Some(4096);
    options.max_row_group_rows = Some(1024);
    let catalog = || {
        DuckLakeCatalog::with_writer(Arc::clone(&source.provider), Arc::clone(&source.writer))
            .unwrap()
            .with_write_options(options.clone())
    };
    let ctx = SessionContext::new();
    DuckLakeCatalog::register(&ctx, "src", catalog());
    let handle = ctx
        .catalog("src")
        .unwrap()
        .schema("main")
        .unwrap()
        .table("t")
        .await
        .unwrap()
        .unwrap();
    let handle = (handle.as_ref() as &dyn std::any::Any)
        .downcast_ref::<DuckLakeTable>()
        .unwrap()
        .clone();
    let destination = make(None).await;
    let forked = manager
        .fork_table(
            &ctx.state(),
            &handle,
            Arc::clone(&destination.writer),
            "main",
            "t",
        )
        .await
        .unwrap();
    assert!(forked.inlined_files_written >= 2, "{forked:?}");
    assert_eq!(destination.rows("t").await, expected);
    let forked_files = file_ids(&destination);
    assert_eq!(
        forked_files.concat(),
        order,
        "sorted by {key} across its files"
    );

    catalog()
        .flush_inlined_data(&ctx.state(), None, None)
        .await
        .unwrap();
    assert_eq!(source.inlined_rows("t"), 0);
    assert_eq!(
        file_ids(&source),
        forked_files,
        "{key}: the files a flush writes"
    );
    // Each file's catalog row as the flush records it: `row_id_start` its
    // smallest row id though its rows are not in row id order, its row count
    // and `partial_max`.
    let catalog_rows = |lake: &Lake| {
        let mut files = lake
            .provider
            .get_table_files_for_select(lake.table_id("t"), lake.head())
            .unwrap();
        files.sort_by_key(|file| file.data_file_id);
        files
            .iter()
            .map(|file| (file.row_id_start, file.max_row_count, file.partial_max))
            .collect::<Vec<_>>()
    };
    let flushed = catalog_rows(&source);
    assert!(
        flushed.iter().all(|(start, _, _)| start.is_some()),
        "{key}: {flushed:?}"
    );
    assert_eq!(catalog_rows(&destination), flushed, "{key}");
}

// ---------------------------------------------------------------------------
// Forks against the source's maintenance
// ---------------------------------------------------------------------------

type Hook = futures::future::BoxFuture<'static, ()>;

/// A local store that runs `hook` once, before the first object the fork
/// uploads, so the source's maintenance lands between the fork reading the
/// source and committing.
struct HookedStore {
    inner: object_store::local::LocalFileSystem,
    hook: tokio::sync::Mutex<Option<Hook>>,
}

impl std::fmt::Debug for HookedStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "HookedStore")
    }
}

impl std::fmt::Display for HookedStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "HookedStore")
    }
}

impl HookedStore {
    async fn run_hook(&self) {
        let hook = self.hook.lock().await.take();
        if let Some(hook) = hook {
            hook.await;
        }
    }
}

#[async_trait::async_trait]
impl object_store::ObjectStore for HookedStore {
    async fn put_opts(
        &self,
        location: &object_store::path::Path,
        payload: object_store::PutPayload,
        opts: object_store::PutOptions,
    ) -> object_store::Result<object_store::PutResult> {
        self.run_hook().await;
        self.inner.put_opts(location, payload, opts).await
    }

    async fn put_multipart_opts(
        &self,
        location: &object_store::path::Path,
        opts: object_store::PutMultipartOptions,
    ) -> object_store::Result<Box<dyn object_store::MultipartUpload>> {
        self.run_hook().await;
        self.inner.put_multipart_opts(location, opts).await
    }

    async fn get_opts(
        &self,
        location: &object_store::path::Path,
        options: object_store::GetOptions,
    ) -> object_store::Result<object_store::GetResult> {
        self.inner.get_opts(location, options).await
    }

    fn delete_stream(
        &self,
        locations: futures::stream::BoxStream<
            'static,
            object_store::Result<object_store::path::Path>,
        >,
    ) -> futures::stream::BoxStream<'static, object_store::Result<object_store::path::Path>> {
        self.inner.delete_stream(locations)
    }

    fn list(
        &self,
        prefix: Option<&object_store::path::Path>,
    ) -> futures::stream::BoxStream<'static, object_store::Result<object_store::ObjectMeta>> {
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

async fn catalog_name(
    pool: &sqlx::PgPool,
    writer: &Arc<dyn datafusion_ducklake::MetadataWriter>,
) -> String {
    sqlx::query_scalar("SELECT catalog_name FROM ducklake_catalog WHERE catalog_id = $1")
        .bind(writer.catalog_id().unwrap())
        .fetch_one(pool)
        .await
        .unwrap()
}

/// Replace `main.t` with `rows` in Parquet, ending its files.
async fn replace(writer: &Arc<dyn datafusion_ducklake::MetadataWriter>, rows: RecordBatch) {
    datafusion_ducklake::DuckLakeTableWriter::new(
        Arc::clone(writer),
        Arc::new(object_store::local::LocalFileSystem::new()),
    )
    .unwrap()
    .with_options(
        &datafusion_ducklake::DuckLakeWriteOptions::default().with_data_inlining_row_limit(0),
    )
    .write_table("main", "t", &[rows])
    .await
    .unwrap();
}

/// Expire every snapshot of `writer`'s catalog but its head, then clean up
/// its scheduled files; the paths cleanup deleted.
async fn expire_and_clean(
    pool: &sqlx::PgPool,
    writer: &Arc<dyn datafusion_ducklake::MetadataWriter>,
) -> Vec<String> {
    let manager = MulticatalogManager::new(pool.clone());
    let name = catalog_name(pool, writer).await;
    let snapshots: Vec<i64> = sqlx::query_scalar(
        "SELECT snapshot_id FROM ducklake_catalog_snapshot_map WHERE catalog_id = $1",
    )
    .bind(writer.catalog_id().unwrap())
    .fetch_all(pool)
    .await
    .unwrap();
    manager
        .expire_snapshots_in_catalog(
            &name,
            datafusion_ducklake::maintenance::ExpireCriteria::Versions(snapshots),
        )
        .await
        .unwrap();
    datafusion_ducklake::maintenance::cleanup_old_files_in_catalog(
        &manager,
        &name,
        Arc::new(object_store::local::LocalFileSystem::new()),
        datafusion_ducklake::maintenance::CleanupCriteria::All,
        false,
    )
    .await
    .unwrap()
}

/// The Parquet files under `lake`'s data directory.
fn parquet_files(lake: &Lake) -> Vec<std::path::PathBuf> {
    walk(lake.data.clone())
        .into_iter()
        .filter(|path| path.extension().is_some_and(|ext| ext == "parquet"))
        .collect()
}

/// The source ends a file the fork has read, expires the snapshots that
/// could read it and cleans it up while the fork is writing its own files:
/// the fork's commit refuses with a retryable `Conflict` instead of
/// registering a reference to a deleted object, and removes the files it
/// wrote. Guards against a fork committed after the reclaim, reading a file
/// that no longer exists.
async fn fork_conflicts_with_a_source_reclaim_mid_fork(pool: sqlx::PgPool) {
    let manager = MulticatalogManager::new(pool.clone());
    let make = multicatalog(pool.clone());
    let source = make(Some(5)).await;
    create(&source, "t").await;
    source.append_parquet("t", &batch(1..=20, "a")).await;
    source.append("t", &batch(100..=102, "i")).await;
    let destination = make(None).await;
    let (_, table) = read_handle(&source, source.head()).await;

    let reclaimed = Arc::new(std::sync::Mutex::new(Vec::new()));
    let hook: Hook = {
        let pool = pool.clone();
        let writer = Arc::clone(&source.writer);
        let reclaimed = Arc::clone(&reclaimed);
        Box::pin(async move {
            replace(&writer, batch(50..=52, "r")).await;
            let deleted = expire_and_clean(&pool, &writer).await;
            reclaimed.lock().unwrap().extend(deleted);
        })
    };
    let store = Arc::new(HookedStore {
        inner: object_store::local::LocalFileSystem::new(),
        hook: tokio::sync::Mutex::new(Some(hook)),
    });
    let ctx = SessionContext::new();
    ctx.runtime_env().register_object_store(
        datafusion::execution::object_store::ObjectStoreUrl::local_filesystem().as_ref(),
        store,
    );
    let forked = manager
        .fork_table(
            &ctx.state(),
            &table,
            Arc::clone(&destination.writer),
            "main",
            "t",
        )
        .await;
    assert!(
        !reclaimed.lock().unwrap().is_empty(),
        "the source reclaimed the file the fork read"
    );
    assert!(
        matches!(forked, Err(DuckLakeError::Conflict(_))),
        "{forked:?}"
    );
    assert!(
        parquet_files(&destination).is_empty(),
        "the refused fork removed its own files"
    );

    // A retry forks what the source holds now.
    fork(&manager, &source, source.head(), &destination)
        .await
        .unwrap();
    assert_eq!(destination.rows("t").await, source.rows("t").await);
}

/// A file a fork, and a fork of the fork, reference survives its owner
/// ending, expiring and cleaning it up for as long as either reference
/// stands, and is reclaimed once the last one goes. The fork of the fork
/// names the root as owner, not the catalog it copied the row from.
async fn cleanup_defers_files_forks_and_forks_of_forks_reference(pool: sqlx::PgPool) {
    let manager = MulticatalogManager::new(pool.clone());
    let make = multicatalog(pool.clone());
    let root = make(Some(0)).await;
    create(&root, "t").await;
    root.append_parquet("t", &batch(1..=10, "a")).await;
    let files = parquet_files(&root);
    assert_eq!(files.len(), 1);
    let file = files[0].clone();
    let expected = root.rows("t").await;

    let middle = make(None).await;
    fork(&manager, &root, root.head(), &middle).await.unwrap();
    let leaf = make(None).await;
    fork(&manager, &middle, middle.head(), &leaf).await.unwrap();
    let owners: Vec<i64> = sqlx::query_scalar(
        "SELECT owner_catalog_id FROM ducklake_catalog_data_file_reference r
         JOIN ducklake_data_file f ON f.data_file_id = r.data_file_id
         JOIN ducklake_table t ON t.table_id = f.table_id
         JOIN ducklake_catalog_schema_map m ON m.schema_id = t.schema_id
         WHERE m.catalog_id = $1",
    )
    .bind(leaf.writer.catalog_id().unwrap())
    .fetch_all(&pool)
    .await
    .unwrap();
    assert_eq!(owners, vec![root.writer.catalog_id().unwrap()]);

    replace(&root.writer, batch(50..=52, "r")).await;
    expire_and_clean(&pool, &root.writer).await;
    assert!(file.exists(), "both references defer the reclaim");
    assert_eq!(middle.rows("t").await, expected);
    assert_eq!(leaf.rows("t").await, expected);

    replace(&middle.writer, batch(60..=62, "m")).await;
    expire_and_clean(&pool, &middle.writer).await;
    expire_and_clean(&pool, &root.writer).await;
    assert!(file.exists(), "the fork of the fork defers the reclaim");
    assert_eq!(leaf.rows("t").await, expected);

    replace(&leaf.writer, batch(70..=72, "l")).await;
    expire_and_clean(&pool, &leaf.writer).await;
    let deleted = expire_and_clean(&pool, &root.writer).await;
    assert_eq!(deleted.len(), 1, "{deleted:?}");
    assert!(
        !file.exists(),
        "the owner reclaims it once no reference stands"
    );
}

/// A fork that replaces its table with an empty source's records the rows it
/// removed (`deleted_from_table`), as every Replace does, so change
/// detection reports the snapshot the change feed shows the deletions at.
/// Guards against a fileless Replace that recorded no row change.
async fn fork_replace_from_an_empty_source_records_the_removal(pool: sqlx::PgPool) {
    let manager = MulticatalogManager::new(pool.clone());
    let make = multicatalog(pool.clone());
    let source = make(Some(0)).await;
    create(&source, "t").await;
    source.append_parquet("t", &batch(1..=5, "a")).await;
    let destination = make(None).await;
    fork(&manager, &source, source.head(), &destination)
        .await
        .unwrap();
    assert_eq!(destination.rows("t").await.len(), 5);

    source.sql("DELETE FROM lake.main.t").await.unwrap();
    assert!(source.rows("t").await.is_empty());
    let forked = fork(&manager, &source, source.head(), &destination)
        .await
        .unwrap();
    assert!(destination.rows("t").await.is_empty());
    let table_id = destination.table_id("t");
    let changes: String = sqlx::query_scalar(
        "SELECT changes_made FROM ducklake_snapshot_changes WHERE snapshot_id = $1",
    )
    .bind(forked.commit.snapshot_id)
    .fetch_one(&pool)
    .await
    .unwrap();
    assert!(
        changes
            .split(',')
            .any(|change| change == format!("deleted_from_table:{table_id}")),
        "{changes}"
    );
    assert_eq!(
        crate::inlining_abstraction_tests::last_rows_changed(
            &destination.provider,
            table_id,
            destination.head()
        )
        .await,
        Some(forked.commit.snapshot_id)
    );
}

/// A partitioned source whose only live file is the 0-row file an empty
/// Replace commits (with no partition values) forks: the file holds no row of
/// the source, so the fork leaves it out and reads the source's (empty)
/// content under its partition spec. Guards against refusing the fork as
/// written under another partition spec.
async fn fork_of_a_partitioned_source_emptied_by_a_replace(pool: sqlx::PgPool) {
    let manager = MulticatalogManager::new(pool.clone());
    let make = multicatalog(pool.clone());
    let source = make(None).await;
    create(&source, "t").await;
    source
        .writer
        .set_partition_spec(
            source.table_id("t"),
            &[(
                "v".to_string(),
                datafusion_ducklake::PartitionTransform::Identity,
            )],
        )
        .unwrap();
    source.append_parquet("t", &batch(1..=20, "a")).await;
    source
        .table_writer()
        .write_table("main", "t", &[batch(std::iter::empty(), "a")])
        .await
        .unwrap();
    assert!(source.rows("t").await.is_empty());
    let files = source
        .provider
        .get_table_files_for_select(source.table_id("t"), source.head())
        .unwrap();
    assert!(
        files.iter().all(|file| file.max_row_count == Some(0)),
        "{files:?}"
    );

    let destination = make(None).await;
    let forked = fork(&manager, &source, source.head(), &destination)
        .await
        .unwrap();
    assert_eq!(forked.files_referenced, 0);
    assert!(destination.rows("t").await.is_empty());
    assert!(
        destination
            .provider
            .get_partition_spec(destination.table_id("t"), destination.head())
            .unwrap()
            .is_some(),
        "the fork keeps the source's partition spec"
    );
}

/// Every fork scenario on one PostgreSQL store, a fresh pair of catalogs
/// each.
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
#[tokio::test(flavor = "multi_thread")]
async fn fork_table_scenarios() {
    let (_container, pool) = store().await;
    fork_carries_inlined_rows_and_deletions(pool.clone()).await;
    fork_of_a_partitioned_source(pool.clone()).await;
    historical_fork_of_a_merged_positional_file(pool.clone()).await;
    fork_after_a_dropped_column(pool.clone()).await;
    fork_refuses_encrypted_and_name_mapped_sources(pool.clone()).await;
    fork_into_an_independent_destination_is_refused(pool.clone()).await;
    fork_copies_column_metadata_and_tags(pool.clone()).await;
    fork_replace_records_tag_changes(pool.clone()).await;
    fork_of_a_legacy_list_column(pool.clone()).await;
    fork_of_a_legacy_list_of_structs(pool.clone()).await;
    fork_copies_metadata_of_every_column_shape(pool.clone()).await;
    fork_change_feed_survives_a_merge(pool.clone()).await;
    fork_with_row_id_gaps_keeps_row_ids_through_a_merge(pool.clone()).await;
    fork_of_updated_inlined_rows_keeps_row_ids_through_a_merge(pool.clone()).await;
    fork_writes_inlined_rows_as_a_flush_sorts_them(pool.clone()).await;
    fork_conflicts_with_a_source_reclaim_mid_fork(pool.clone()).await;
    cleanup_defers_files_forks_and_forks_of_forks_reference(pool.clone()).await;
    fork_replace_from_an_empty_source_records_the_removal(pool.clone()).await;
    fork_of_a_partitioned_source_emptied_by_a_replace(pool.clone()).await;
    fork_fuzz(pool).await;
}
