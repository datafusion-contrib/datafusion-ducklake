//! `ducklake_table_changes` is official DuckLake's macro over this crate's own
//! feeds: the window's `ducklake_table_insertions` left-joined to its
//! `ducklake_table_deletions` on `(snapshot_id, rowid)` (`update_postimage`,
//! else `insert`), unioned with the deletions left-joined to the insertions
//! (`update_preimage`, else `delete`).
//!
//! Checked over seeded random histories — inlined and file inserts, updates,
//! deletes, flushes, merges of adjacent files (merged partial files) and
//! rewrites — for every window, on SQLite and on PostgreSQL (the multi-catalog
//! layout, whose writer runs UPDATE and DELETE). The insertions and deletions
//! feeds are diffed against official's elsewhere (`cdc_differential_tests`,
//! `inlining_fuzz_tests`); this pins the pairing that composes them, including
//! the plan's projections and the plain-insert path that skips the join.
//!
//! The same histories are also replayed: each window's feed, applied to the
//! table as it was before the window, must give the table at its end, read
//! from the table itself.

#![cfg(all(feature = "write-sqlite", feature = "metadata-sqlite"))]

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow::util::display::array_value_to_string;
use datafusion::prelude::SessionContext;
use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeTable, DuckLakeWriteOptions, MergeOptions, MetadataProvider,
    MetadataWriter, RewriteOptions, execute_ducklake_sql, register_ducklake_functions,
};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

type Open = Arc<
    dyn Fn() -> std::pin::Pin<
            Box<
                dyn std::future::Future<
                        Output = (Arc<dyn MetadataWriter>, Arc<dyn MetadataProvider>),
                    > + Send,
            >,
        > + Send
        + Sync,
>;

struct Lake {
    open: Open,
    limit: usize,
}

impl Lake {
    async fn session(&self) -> (SessionContext, DuckLakeCatalog, Arc<dyn MetadataProvider>) {
        let (writer, provider) = (self.open)().await;
        let ctx = SessionContext::new();
        let make = || {
            DuckLakeCatalog::with_writer(Arc::clone(&provider), Arc::clone(&writer))
                .unwrap()
                .with_write_options(
                    DuckLakeWriteOptions::default().with_data_inlining_row_limit(self.limit),
                )
        };
        DuckLakeCatalog::register(&ctx, "lake", make());
        register_ducklake_functions(&ctx, Arc::clone(&provider));
        (ctx, make(), provider)
    }

    async fn sql(&self, statement: &str) {
        let (ctx, catalog, _) = self.session().await;
        execute_ducklake_sql(&ctx, &catalog, statement)
            .await
            .unwrap_or_else(|e| panic!("{statement}: {e}"))
            .collect()
            .await
            .unwrap_or_else(|e| panic!("{statement}: {e}"));
    }

    async fn table(&self) -> (SessionContext, DuckLakeTable) {
        let (ctx, _, _) = self.session().await;
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
            .expect("a DuckLakeTable")
            .clone();
        (ctx, table)
    }
}

/// Sorted canonical rows of `sql`, or the error's text.
async fn rows(ctx: &SessionContext, sql: &str) -> Result<Vec<String>, String> {
    let frame = ctx.sql(sql).await.map_err(|e| e.to_string())?;
    let batches = frame.collect().await.map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for batch in &batches {
        for row in 0..batch.num_rows() {
            let cells: Vec<String> = batch
                .columns()
                .iter()
                .map(|c| array_value_to_string(c, row).unwrap())
                .collect();
            out.push(cells.join("|"));
        }
    }
    out.sort();
    Ok(out)
}

async fn run_history(lake: &Lake, seed: u64, steps: usize) -> usize {
    let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    lake.sql("CREATE TABLE lake.main.t (id BIGINT, val BIGINT, s VARCHAR)")
        .await;
    let created = (lake.session().await.2).get_current_snapshot().unwrap();
    let mut next_id = 0i64;
    for _ in 0..steps {
        match rng.below(13) {
            0..=4 => {
                // Small batches inline (below the limit), larger ones are files.
                let n = if rng.below(2) == 0 {
                    1 + rng.below(4)
                } else {
                    20 + rng.below(3000)
                };
                let values: Vec<String> = (0..n)
                    .map(|_| {
                        next_id += 1;
                        format!("({next_id}, {}, 's{}')", rng.below(100), rng.below(7))
                    })
                    .collect();
                lake.sql(&format!(
                    "INSERT INTO lake.main.t VALUES {}",
                    values.join(", ")
                ))
                .await;
            },
            // Updates and deletes of a narrow id range touch few files, leaving
            // the rest delete-free for a merge to combine; a range of at most
            // the inlining limit's rows of a file is deleted inline.
            5 | 6 => {
                let (from, width) = (rng.below(next_id as u64 + 1), 1 + rng.below(400));
                lake.sql(&format!(
                    "UPDATE lake.main.t SET val = val + 1 WHERE id >= {from} AND id < {}",
                    from + width
                ))
                .await;
            },
            7 => {
                let (from, width) = (rng.below(next_id as u64 + 1), 1 + rng.below(6));
                lake.sql(&format!(
                    "DELETE FROM lake.main.t WHERE id >= {from} AND id < {}",
                    from + width
                ))
                .await;
            },
            8 => lake.sql("CALL ducklake_flush_inlined_data('lake')").await,
            9..=11 => {
                let (ctx, table) = lake.table().await;
                table
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
            },
            _ => {
                let (ctx, table) = lake.table().await;
                table
                    .rewrite_data_files(
                        &ctx.state(),
                        RewriteOptions {
                            delete_threshold: 0.0,
                            data_file_ids: None,
                        },
                    )
                    .await
                    .unwrap();
            },
        }
    }

    let (ctx, _, provider) = lake.session().await;
    let head = provider.get_current_snapshot().unwrap();
    let mut compared = replay(&ctx, &provider, seed, created, head).await;
    for a in 0..=head {
        for b in a.max(created)..=head {
            let feed = |name: &str| format!("{name}('main.t', {a}, {b})");
            let (insertions, deletions) = (
                feed("ducklake_table_insertions"),
                feed("ducklake_table_deletions"),
            );
            let composed = format!(
                "SELECT i.snapshot_id, i.rowid, \
                     CASE WHEN d.rowid IS NOT NULL THEN 'update_postimage' ELSE 'insert' END \
                         AS change_type, \
                     i.id, i.val, i.s \
                 FROM {insertions} i LEFT JOIN {deletions} d \
                     ON i.snapshot_id = d.snapshot_id AND i.rowid = d.rowid \
                 UNION ALL \
                 SELECT d.snapshot_id, d.rowid, \
                     CASE WHEN i.rowid IS NOT NULL THEN 'update_preimage' ELSE 'delete' END, \
                     d.id, d.val, d.s \
                 FROM {deletions} d LEFT JOIN {insertions} i \
                     ON d.snapshot_id = i.snapshot_id AND d.rowid = i.rowid"
            );
            let changes = feed("ducklake_table_changes");
            for select in [
                "snapshot_id, rowid, change_type, id, val, s",
                "change_type, id",
                "s, snapshot_id",
                "count(*)",
            ] {
                let feed_sql = format!("SELECT {select} FROM {changes}");
                let macro_sql = format!("SELECT {select} FROM ({composed})");
                let expected = rows(&ctx, &macro_sql).await;
                let actual = rows(&ctx, &feed_sql).await;
                assert_eq!(
                    actual, expected,
                    "seed {seed} window [{a}, {b}]: `{feed_sql}` against the macro"
                );
                compared += 1;
            }
        }
    }
    compared
}

/// The table's rows at `snapshot`, by rowid, read from the table itself.
async fn state_at(
    provider: &Arc<dyn MetadataProvider>,
    created: i64,
    snapshot: i64,
) -> BTreeMap<i64, String> {
    if snapshot < created {
        return BTreeMap::new();
    }
    let ctx = SessionContext::new();
    DuckLakeCatalog::register(
        &ctx,
        "lake",
        DuckLakeCatalog::with_snapshot(Arc::clone(provider), snapshot)
            .unwrap()
            .with_row_lineage(true),
    );
    rows(&ctx, "SELECT rowid, id, val, s FROM lake.main.t")
        .await
        .unwrap_or_else(|e| panic!("reading the table at {snapshot}: {e}"))
        .into_iter()
        .map(|row| {
            let (rowid, values) = row.split_once('|').unwrap();
            (rowid.parse().unwrap(), values.to_string())
        })
        .collect()
}

/// Replay every window's change feed onto the table as it was before the
/// window, snapshot by snapshot, and compare with the table as it is at the
/// window's end: an oracle independent of the feeds' own implementations.
///
/// A deleted row (`delete`, `update_preimage`) must be in the table with the
/// values the feed reports, an added one (`insert`, `update_postimage`) must
/// not be; a snapshot's deletions apply before its additions.
async fn replay(
    ctx: &SessionContext,
    provider: &Arc<dyn MetadataProvider>,
    seed: u64,
    created: i64,
    head: i64,
) -> usize {
    let mut states = Vec::new();
    for snapshot in 0..=head {
        states.push(state_at(provider, created, snapshot).await);
    }
    let mut replayed = 0;
    for a in 0..=head {
        for b in a.max(created)..=head {
            let feed = rows(
                ctx,
                &format!(
                    "SELECT snapshot_id, rowid, change_type, id, val, s \
                     FROM ducklake_table_changes('main.t', {a}, {b})"
                ),
            )
            .await
            .unwrap_or_else(|e| panic!("seed {seed} window [{a}, {b}]: {e}"));
            let mut changes: Vec<(i64, bool, i64, String)> = feed
                .iter()
                .map(|row| {
                    let mut cells = row.splitn(4, '|');
                    let snapshot: i64 = cells.next().unwrap().parse().unwrap();
                    let rowid: i64 = cells.next().unwrap().parse().unwrap();
                    let adds = match cells.next().unwrap() {
                        "insert" | "update_postimage" => true,
                        "delete" | "update_preimage" => false,
                        other => panic!("unknown change type {other}"),
                    };
                    (snapshot, adds, rowid, cells.next().unwrap().to_string())
                })
                .collect();
            changes.sort();
            let mut state = if a == 0 {
                BTreeMap::new()
            } else {
                states[a as usize - 1].clone()
            };
            for (snapshot, adds, rowid, values) in changes {
                assert!(
                    (a..=b).contains(&snapshot),
                    "seed {seed} window [{a}, {b}]: a change at {snapshot}"
                );
                if adds {
                    assert!(
                        state.insert(rowid, values.clone()).is_none(),
                        "seed {seed} window [{a}, {b}]: rowid {rowid} added at {snapshot} \
                         ({values}) is already in the table"
                    );
                } else {
                    assert_eq!(
                        state.remove(&rowid).as_ref(),
                        Some(&values),
                        "seed {seed} window [{a}, {b}]: rowid {rowid} deleted at {snapshot}"
                    );
                }
            }
            assert!(
                state == states[b as usize],
                "seed {seed} window [{a}, {b}]: the replayed feed is not the table at {b}"
            );
            replayed += 1;
        }
    }
    replayed
}

fn seeds(default: u64) -> std::ops::Range<u64> {
    let n: u64 = std::env::var("FEED_SEEDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default);
    0..n
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn table_changes_is_the_macro_over_its_feeds_sqlite() {
    use datafusion_ducklake::{SqliteMetadataProvider, SqliteMetadataWriter};
    let mut compared = 0;
    for seed in seeds(4) {
        let dir = tempfile::tempdir().unwrap();
        let data = dir.path().join("data");
        std::fs::create_dir_all(&data).unwrap();
        let url = format!("sqlite:{}?mode=rwc", dir.path().join("c.db").display());
        let writer = SqliteMetadataWriter::new_with_init(&url).await.unwrap();
        writer.set_data_path(data.to_str().unwrap()).unwrap();
        let snapshot = writer.create_snapshot().unwrap();
        writer.get_or_create_schema("main", None, snapshot).unwrap();
        let writer: Arc<dyn MetadataWriter> = Arc::new(writer);
        let provider: Arc<dyn MetadataProvider> =
            Arc::new(SqliteMetadataProvider::new(&url).await.unwrap());
        let lake = Lake {
            open: Arc::new(move || {
                let pair = (Arc::clone(&writer), Arc::clone(&provider));
                Box::pin(async move { pair })
            }),
            limit: 5,
        };
        compared += run_history(&lake, seed, 18).await;
    }
    assert!(compared > 0);
}

#[cfg(feature = "write-postgres")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn table_changes_is_the_macro_over_its_feeds_postgres() {
    use datafusion_ducklake::{MulticatalogManager, MulticatalogProvider, PostgresMetadataWriter};
    use testcontainers::runners::AsyncRunner;
    let container = testcontainers_modules::postgres::Postgres::default()
        .start()
        .await
        .unwrap();
    let port = container.get_host_port_ipv4(5432).await.unwrap();
    let mut compared = 0;
    for seed in seeds(2) {
        let database = format!("feed_{seed}");
        let admin = sqlx::PgPool::connect(&format!(
            "postgresql://postgres:postgres@127.0.0.1:{port}/postgres"
        ))
        .await
        .unwrap();
        sqlx::query(sqlx::AssertSqlSafe(format!("CREATE DATABASE {database}")))
            .execute(&admin)
            .await
            .unwrap();
        admin.close().await;
        let dir = tempfile::tempdir().unwrap();
        let data = dir.path().join("data");
        std::fs::create_dir_all(&data).unwrap();
        let url = format!("postgresql://postgres:postgres@127.0.0.1:{port}/{database}");
        // The multi-catalog layout: the PostgreSQL writer that runs UPDATE
        // and DELETE.
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
        let snapshot = writer.create_snapshot().unwrap();
        writer.get_or_create_schema("main", None, snapshot).unwrap();
        let writer: Arc<dyn MetadataWriter> = Arc::new(writer);
        let provider: Arc<dyn MetadataProvider> =
            Arc::new(MulticatalogProvider::with_pool(pool, "lake").await.unwrap());
        let lake = Lake {
            open: Arc::new(move || {
                let pair = (Arc::clone(&writer), Arc::clone(&provider));
                Box::pin(async move { pair })
            }),
            limit: 5,
        };
        compared += run_history(&lake, seed, 18).await;
    }
    assert!(compared > 0);
}
