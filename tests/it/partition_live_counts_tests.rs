//! `DuckLakeTable::partition_live_row_counts` against a real scan.
//!
//! Every exact count the metadata reports is compared with
//! `SELECT slot, count(*) ... GROUP BY slot` at the same snapshot, through
//! appends, repeated deletes, updates that rewrite rows within and across
//! partitions, merge and rewrite compaction, time travel, inlined deletions,
//! inlined rows and their flush. Where the catalog cannot account for every
//! row the count must say unknown, with the reason, never a number.
//!
//! Run against SQLite and against the multicatalog Postgres provider
//! (Docker-gated), which share the rules but not the SQL.

#![cfg(any(feature = "write-sqlite", feature = "write-postgres"))]

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use arrow::array::{Array, Int64Array, StringArray};
use arrow::datatypes::DataType;
use datafusion::catalog::CatalogProvider;
use datafusion::prelude::*;
use datafusion_ducklake::partition::PartitionTransform;
use datafusion_ducklake::{
    ColumnDef, DuckLakeCatalog, DuckLakeTable, DuckLakeWriteOptions, MergeOptions,
    MetadataProvider, MetadataWriter, PartitionCountUnknown, PartitionRowCount, RewriteOptions,
    WriteMode,
};
use tempfile::TempDir;

const CATALOG: &str = "lake";

enum Backend {
    #[cfg(feature = "write-sqlite")]
    Sqlite {
        conn: String,
    },
    #[cfg(feature = "write-postgres")]
    Postgres {
        pool: sqlx::PgPool,
        catalog_id: i64,
        _container: Box<testcontainers::ContainerAsync<testcontainers_modules::postgres::Postgres>>,
    },
}

struct Lake {
    backend: Backend,
    data: PathBuf,
    _dir: TempDir,
}

impl Lake {
    #[cfg(feature = "write-sqlite")]
    async fn sqlite() -> Self {
        let dir = TempDir::new().unwrap();
        let data = dir.path().join("data");
        std::fs::create_dir_all(&data).unwrap();
        let conn = format!("sqlite:{}?mode=rwc", dir.path().join("lake.db").display());
        let lake = Self {
            backend: Backend::Sqlite {
                conn,
            },
            data,
            _dir: dir,
        };
        lake.writer(0).await;
        lake
    }

    #[cfg(feature = "write-postgres")]
    async fn postgres() -> Self {
        use testcontainers::runners::AsyncRunner;
        let container = testcontainers_modules::postgres::Postgres::default()
            .start()
            .await
            .unwrap();
        let port = container.get_host_port_ipv4(5432).await.unwrap();
        let pool = sqlx::postgres::PgPoolOptions::new()
            .max_connections(5)
            .connect(&format!(
                "postgresql://postgres:postgres@127.0.0.1:{port}/postgres"
            ))
            .await
            .unwrap();
        datafusion_ducklake::initialize_multicatalog_schema(&pool)
            .await
            .unwrap();
        let catalog_id = datafusion_ducklake::MulticatalogManager::new(pool.clone())
            .create_catalog(CATALOG)
            .await
            .unwrap();
        let dir = TempDir::new().unwrap();
        let data = dir.path().join("data");
        std::fs::create_dir_all(&data).unwrap();
        Self {
            backend: Backend::Postgres {
                pool,
                catalog_id,
                _container: Box::new(container),
            },
            data,
            _dir: dir,
        }
    }

    async fn provider(&self) -> Arc<dyn MetadataProvider> {
        match &self.backend {
            #[cfg(feature = "write-sqlite")]
            Backend::Sqlite {
                conn,
            } => Arc::new(
                datafusion_ducklake::SqliteMetadataProvider::new(conn)
                    .await
                    .unwrap(),
            ),
            #[cfg(feature = "write-postgres")]
            Backend::Postgres {
                pool,
                ..
            } => Arc::new(
                datafusion_ducklake::MulticatalogProvider::with_pool(pool.clone(), CATALOG)
                    .await
                    .unwrap(),
            ),
        }
    }

    /// A metadata writer whose data inlining limit is `inline`.
    async fn writer(&self, inline: usize) -> Arc<dyn MetadataWriter> {
        let writer: Arc<dyn MetadataWriter> = match &self.backend {
            #[cfg(feature = "write-sqlite")]
            Backend::Sqlite {
                conn,
            } => Arc::new(
                datafusion_ducklake::SqliteMetadataWriter::new_with_init(conn)
                    .await
                    .unwrap()
                    .with_data_inlining_row_limit(inline),
            ),
            #[cfg(feature = "write-postgres")]
            Backend::Postgres {
                pool,
                catalog_id,
                ..
            } => Arc::new(
                datafusion_ducklake::PostgresMetadataWriter::with_pool(pool.clone(), *catalog_id)
                    .await
                    .unwrap()
                    .with_data_inlining_row_limit(inline),
            ),
        };
        writer.set_data_path(self.data.to_str().unwrap()).unwrap();
        writer
    }

    async fn head(&self) -> i64 {
        self.provider().await.get_current_snapshot().unwrap()
    }

    /// A writable session at the head, inlining up to `inline` rows.
    async fn session(&self, inline: usize) -> (SessionContext, Arc<DuckLakeCatalog>) {
        let catalog = Arc::new(
            DuckLakeCatalog::with_writer(self.provider().await, self.writer(inline).await)
                .unwrap()
                .with_write_options(
                    DuckLakeWriteOptions::default().with_data_inlining_row_limit(inline),
                ),
        );
        let ctx = SessionContext::new();
        ctx.register_catalog(CATALOG, Arc::clone(&catalog) as Arc<dyn CatalogProvider>);
        (ctx, catalog)
    }

    /// Run one statement in a fresh session at the head.
    async fn sql(&self, inline: usize, sql: &str) {
        let (ctx, _catalog) = self.session(inline).await;
        ctx.sql(sql).await.unwrap().collect().await.unwrap();
    }

    /// A read-only session at `snapshot`, or at the head.
    async fn reader(&self, snapshot: Option<i64>) -> SessionContext {
        let provider = self.provider().await;
        let snapshot = match snapshot {
            Some(snapshot) => snapshot,
            None => provider.get_current_snapshot().unwrap(),
        };
        let catalog = DuckLakeCatalog::with_snapshot(provider, snapshot).unwrap();
        let ctx = SessionContext::new();
        ctx.register_catalog(CATALOG, Arc::new(catalog));
        ctx
    }

    async fn table_in(ctx: &SessionContext, name: &str) -> DuckLakeTable {
        let provider = ctx
            .catalog(CATALOG)
            .unwrap()
            .schema("main")
            .unwrap()
            .table(name)
            .await
            .unwrap()
            .unwrap();
        (provider.as_ref() as &dyn std::any::Any)
            .downcast_ref::<DuckLakeTable>()
            .expect("a DuckLakeTable")
            .clone()
    }

    async fn table(&self, name: &str, snapshot: Option<i64>) -> DuckLakeTable {
        Self::table_in(&self.reader(snapshot).await, name).await
    }

    /// Create `main.<name>(id, slot, val)`, partitioned by `slot` unless
    /// `partitioned` is false.
    async fn create(&self, name: &str, partitioned: bool) -> i64 {
        let columns = vec![
            ColumnDef::from_arrow("id", &DataType::Int32, false).unwrap(),
            ColumnDef::from_arrow("slot", &DataType::Int32, false).unwrap(),
            ColumnDef::from_arrow("val", &DataType::Int32, false).unwrap(),
        ];
        let keys: &[(&str, PartitionTransform)] = if partitioned {
            &[("slot", PartitionTransform::Identity)]
        } else {
            &[]
        };
        self.create_with(name, columns, keys).await
    }

    /// Create `main.<name>` with `columns`, partitioned by `keys` when there
    /// are any.
    async fn create_with(
        &self,
        name: &str,
        columns: Vec<ColumnDef>,
        keys: &[(&str, PartitionTransform)],
    ) -> i64 {
        let writer = self.writer(0).await;
        let write = writer
            .begin_write_transaction("main", name, &columns, WriteMode::Replace)
            .unwrap();
        writer
            .publish_snapshot(
                write.table_id,
                "main",
                name,
                write.snapshot_id,
                WriteMode::Replace,
                write.base_snapshot_id,
                &columns,
                &write.column_ids,
            )
            .unwrap();
        if !keys.is_empty() {
            let keys: Vec<(String, PartitionTransform)> = keys
                .iter()
                .map(|(column, transform)| (column.to_string(), transform.clone()))
                .collect();
            writer.set_partition_spec(write.table_id, &keys).unwrap();
        }
        write.table_id
    }

    async fn partition(&self, table_id: i64) {
        self.writer(0)
            .await
            .set_partition_spec(
                table_id,
                &[("slot".to_string(), PartitionTransform::Identity)],
            )
            .unwrap();
    }

    /// `count(*)` per slot of `name` at `snapshot`, by scanning.
    async fn scan_counts(&self, name: &str, snapshot: Option<i64>) -> BTreeMap<String, u64> {
        self.scan_counts_by(name, "slot", snapshot).await
    }

    /// `count(*)` per non-NULL value of `column` of `name` at `snapshot`, by
    /// scanning.
    async fn scan_counts_by(
        &self,
        name: &str,
        column: &str,
        snapshot: Option<i64>,
    ) -> BTreeMap<String, u64> {
        let ctx = self.reader(snapshot).await;
        let batches = ctx
            .sql(&format!(
                "SELECT CAST({column} AS VARCHAR) AS k, CAST(COUNT(*) AS BIGINT) AS n \
                 FROM {CATALOG}.main.{name} GROUP BY {column}"
            ))
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let mut counts = BTreeMap::new();
        for batch in &batches {
            let slots = arrow::compute::cast(batch.column(0), &DataType::Utf8).unwrap();
            let slots = slots.as_any().downcast_ref::<StringArray>().unwrap();
            let rows = batch
                .column(1)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            for i in 0..batch.num_rows() {
                if slots.is_null(i) {
                    continue;
                }
                counts.insert(slots.value(i).to_string(), rows.value(i) as u64);
            }
        }
        counts
    }

    fn is_postgres(&self) -> bool {
        match &self.backend {
            #[cfg(feature = "write-sqlite")]
            Backend::Sqlite {
                ..
            } => false,
            #[cfg(feature = "write-postgres")]
            Backend::Postgres {
                ..
            } => true,
        }
    }

    /// Run catalog SQL directly, to put the catalog in a state no writer
    /// produces.
    async fn tamper(&self, sql: &str) {
        match &self.backend {
            #[cfg(feature = "write-sqlite")]
            Backend::Sqlite {
                conn,
            } => {
                let pool = sqlx::SqlitePool::connect(conn).await.unwrap();
                sqlx::query(sqlx::AssertSqlSafe(sql))
                    .execute(&pool)
                    .await
                    .unwrap();
            },
            #[cfg(feature = "write-postgres")]
            Backend::Postgres {
                pool,
                ..
            } => {
                sqlx::query(sqlx::AssertSqlSafe(sql))
                    .execute(pool)
                    .await
                    .unwrap();
            },
        }
    }
}

fn strings(values: &[&str]) -> Vec<String> {
    values.iter().map(|value| value.to_string()).collect()
}

/// Compare every exact count of `slot` with a scan at the same snapshot, and
/// return the counts. See [`check_column`].
async fn check(
    lake: &Lake,
    name: &str,
    snapshot: Option<i64>,
    values: &[&str],
    unknown: &[(&str, PartitionCountUnknown)],
) -> BTreeMap<String, PartitionRowCount> {
    check_column(lake, name, "slot", snapshot, values, unknown).await
}

/// Compare every exact count of `column` with a scan at the same snapshot, and
/// return the counts. A value whose count is unknown must be named in
/// `unknown` with the reason given there; one named there must be unknown.
///
/// One unknown is accepted unnamed: below the head, a value with a visible
/// delete file may be `PartialDeleteFile`, as a delete file recording no
/// `partial_max` may begin before deletions it holds.
async fn check_column(
    lake: &Lake,
    name: &str,
    column: &str,
    snapshot: Option<i64>,
    values: &[&str],
    unknown: &[(&str, PartitionCountUnknown)],
) -> BTreeMap<String, PartitionRowCount> {
    let table = lake.table(name, snapshot).await;
    let counts = table
        .partition_live_row_counts(column, &strings(values))
        .unwrap();
    let scanned = lake.scan_counts_by(name, column, snapshot).await;
    let head = lake.head().await;
    let historical = snapshot.is_some_and(|snapshot| snapshot != head);
    let with_deletes: Vec<String> = table
        .files()
        .unwrap()
        .into_iter()
        .filter(|file| file.delete_file.is_some())
        .flat_map(|file| file.partition_values.into_iter().filter_map(|(_, v)| v))
        .collect();
    assert_eq!(counts.len(), values.len(), "{counts:?}");
    for value in values {
        let expected_unknown = unknown
            .iter()
            .find(|(candidate, _)| candidate == value)
            .map(|(_, reason)| *reason);
        match (counts[*value], expected_unknown) {
            (PartitionRowCount::Exact(rows), None) => assert_eq!(
                rows,
                scanned.get(*value).copied().unwrap_or(0),
                "{column} {value} at {snapshot:?}: metadata says {rows}, the scan disagrees \
                 ({scanned:?})"
            ),
            (PartitionRowCount::Unknown(reason), Some(expected)) => {
                assert_eq!(reason, expected, "{column} {value} at {snapshot:?}")
            },
            (PartitionRowCount::Unknown(PartitionCountUnknown::PartialDeleteFile), None)
                if historical && with_deletes.iter().any(|v| v == value) => {},
            (count, expected) => panic!(
                "{column} {value} at {snapshot:?}: got {count:?}, expected unknown \
                 {expected:?} (scan {scanned:?})"
            ),
        }
    }
    counts
}

/// Insert `rows` rows into each of `slots`, ids from `first_id`, in one
/// statement (so one file per slot).
async fn insert(lake: &Lake, name: &str, slots: &[i32], first_id: i32, rows: i32) {
    let mut tuples = Vec::new();
    let mut id = first_id;
    for slot in slots {
        for _ in 0..rows {
            tuples.push(format!("({id}, {slot}, {})", id * 10));
            id += 1;
        }
    }
    lake.sql(
        0,
        &format!(
            "INSERT INTO {CATALOG}.main.{name} SELECT * FROM (VALUES {}) AS v(id, slot, val)",
            tuples.join(", ")
        ),
    )
    .await;
}

/// Run a compaction step on the writable `t` at the head.
async fn compact(lake: &Lake, merge: bool) -> datafusion_ducklake::CompactionResult {
    let (ctx, _catalog) = lake.session(0).await;
    let table = Lake::table_in(&ctx, "t").await;
    if merge {
        table
            .merge_adjacent_files(&ctx.state(), MergeOptions::default())
            .await
            .unwrap()
    } else {
        table
            .rewrite_data_files(
                &ctx.state(),
                RewriteOptions {
                    delete_threshold: 0.0,
                    ..RewriteOptions::default()
                },
            )
            .await
            .unwrap()
    }
}

const SLOTS: &[&str] = &["-1", "0", "1", "2", "3", "4", "99"];

async fn scenario(lake: Lake) {
    use PartitionCountUnknown::*;

    let t_id = lake.create("t", true).await;
    // An empty table: every requested value exact at zero.
    let counts = check(&lake, "t", None, SLOTS, &[]).await;
    assert!(
        counts
            .values()
            .all(|count| *count == PartitionRowCount::Exact(0))
    );

    // Appends: two files per slot for -1..=2, one for 3, a holding-style
    // negative slot among them.
    insert(&lake, "t", &[-1, 0, 1, 2], 0, 6).await;
    let appended_once = lake.head().await;
    insert(&lake, "t", &[-1, 0, 1, 2, 3], 100, 5).await;
    let appended = lake.head().await;
    let counts = check(&lake, "t", None, SLOTS, &[]).await;
    assert_eq!(counts["1"], PartitionRowCount::Exact(11));
    assert_eq!(counts["99"], PartitionRowCount::Exact(0));

    // A delete, then another on the same files (the second rewrites each
    // file's delete file to carry both).
    lake.sql(0, "DELETE FROM lake.main.t WHERE slot = 1 AND id % 3 = 0")
        .await;
    let deleted_once = lake.head().await;
    check(&lake, "t", None, SLOTS, &[]).await;
    lake.sql(0, "DELETE FROM lake.main.t WHERE slot = 1 AND id % 3 = 1")
        .await;
    let deleted_twice = lake.head().await;
    let counts = check(&lake, "t", None, SLOTS, &[]).await;
    assert!(counts["1"].exact().unwrap() < 11, "{counts:?}");

    // Updates: one rewriting rows within their partition, one moving rows to
    // another partition.
    lake.sql(
        0,
        "UPDATE lake.main.t SET val = val + 1 WHERE slot = 2 AND id % 2 = 0",
    )
    .await;
    let updated = lake.head().await;
    check(&lake, "t", None, SLOTS, &[]).await;
    lake.sql(
        0,
        "UPDATE lake.main.t SET slot = 3 WHERE slot = 0 AND id < 9",
    )
    .await;
    let moved = lake.head().await;
    let counts = check(&lake, "t", None, SLOTS, &[]).await;
    assert_eq!(counts["0"], PartitionRowCount::Exact(8), "{counts:?}");
    assert_eq!(counts["3"], PartitionRowCount::Exact(8), "{counts:?}");

    // Time travel over every state so far. No data file is partial yet. The
    // second delete replaced slot 1's delete files with ones that carry each
    // deletion's snapshot, begin at the first delete and record the second
    // as `partial_max`: between the two only a read of them can tell which
    // deletions apply, so slot 1 is unknown there.
    for snapshot in [appended_once, appended, deleted_twice, updated, moved] {
        check(&lake, "t", Some(snapshot), SLOTS, &[]).await;
    }
    check(
        &lake,
        "t",
        Some(deleted_once),
        SLOTS,
        &[("1", PartialDeleteFile)],
    )
    .await;

    // Merge compaction writes partial files: exact at the head, unknown below
    // their `partial_max`, where a scan drops later rows one by one.
    let merged = compact(&lake, true).await;
    assert!(merged.files_created > 0, "{merged:?}");
    let after_merge = lake.head().await;
    check(&lake, "t", None, SLOTS, &[]).await;
    // The slots with a file visible at the first append that a scan there
    // filters row by row.
    let partial: Vec<String> = lake
        .table("t", Some(appended_once))
        .await
        .files()
        .unwrap()
        .into_iter()
        .filter(|file| file.partial_max.is_some_and(|max| max > appended_once))
        .filter_map(|file| file.partition_values.first().and_then(|(_, v)| v.clone()))
        .collect();
    assert!(!partial.is_empty(), "the merge wrote no partial file");
    let historical = lake
        .table("t", Some(appended_once))
        .await
        .partition_live_row_counts("slot", &strings(SLOTS))
        .unwrap();
    let scanned = lake.scan_counts("t", Some(appended_once)).await;
    for value in SLOTS {
        match historical[*value] {
            PartitionRowCount::Exact(rows) => {
                assert!(
                    !partial.contains(&value.to_string()),
                    "slot {value} is partial"
                );
                assert_eq!(
                    rows,
                    scanned.get(*value).copied().unwrap_or(0),
                    "slot {value}"
                );
            },
            PartitionRowCount::Unknown(reason) => {
                assert_eq!(reason, PartialFile, "slot {value}");
                assert!(partial.contains(&value.to_string()), "slot {value}");
            },
        }
    }
    for snapshot in [moved, after_merge] {
        check(&lake, "t", Some(snapshot), SLOTS, &[]).await;
    }

    // Rewrite compaction drops deleted rows into new files.
    compact(&lake, false).await;
    check(&lake, "t", None, SLOTS, &[]).await;

    // Inlined deletions: counted, once, on a file with no delete file.
    lake.sql(10, "DELETE FROM lake.main.t WHERE slot = 3 AND id = 120")
        .await;
    let inlined_delete = lake.head().await;
    assert!(
        !lake
            .provider()
            .await
            .get_inlined_deletes(t_id, inlined_delete)
            .unwrap()
            .is_empty(),
        "the delete was not inlined"
    );
    check(&lake, "t", None, SLOTS, &[]).await;
    // ...and unknown on a file that also has a delete file, where only reading
    // the delete file could rule out an overlap.
    lake.sql(
        0,
        "DELETE FROM lake.main.t WHERE slot = 2 AND id IN (19, 21)",
    )
    .await;
    lake.sql(10, "DELETE FROM lake.main.t WHERE slot = 2 AND id = 23")
        .await;
    let mixed = lake.head().await;
    check(
        &lake,
        "t",
        None,
        SLOTS,
        &[("2", InlinedDeletesWithDeleteFile)],
    )
    .await;

    // Inlined rows: no file counter includes them, so every value is unknown
    // until they are flushed.
    lake.sql(10, "INSERT INTO lake.main.t VALUES (500, 4, 0)")
        .await;
    let inlined_rows = lake.head().await;
    let unknown: Vec<(&str, PartitionCountUnknown)> =
        SLOTS.iter().map(|slot| (*slot, InlinedRows)).collect();
    check(&lake, "t", None, SLOTS, &unknown).await;
    {
        let (ctx, catalog) = lake.session(10).await;
        catalog
            .flush_inlined_data(&ctx.state(), Some("main"), Some("t"))
            .await
            .unwrap();
    }
    let flushed = lake.head().await;
    let counts = check(&lake, "t", None, SLOTS, &[]).await;
    assert_eq!(counts["4"], PartitionRowCount::Exact(1));
    // The flush moves the rows' history with them: a read below it is served
    // by the flushed file, and no inlined row is left to make it unknown.
    let counts = check(&lake, "t", Some(inlined_rows), SLOTS, &[]).await;
    assert_eq!(counts["4"], PartitionRowCount::Exact(1));
    check(&lake, "t", Some(mixed), SLOTS, &[]).await;
    check(&lake, "t", Some(flushed), SLOTS, &[]).await;

    // A column that is not a partition key.
    let not_key = lake
        .table("t", None)
        .await
        .partition_live_row_counts("val", &strings(&["0"]))
        .unwrap();
    assert_eq!(not_key["0"], PartitionRowCount::Unknown(NotAPartitionKey));

    // A file whose row count the catalog does not record: unknown, never 0.
    let table = lake.table("t", None).await;
    let victim = table
        .files()
        .unwrap()
        .into_iter()
        .find(|file| {
            file.partition_values
                .first()
                .and_then(|(_, v)| v.as_deref())
                == Some("-1")
        })
        .expect("a slot -1 file");
    lake.tamper(&format!(
        "UPDATE ducklake_data_file SET record_count = NULL WHERE data_file_id = {}",
        victim.data_file_id
    ))
    .await;
    let counts = lake
        .table("t", None)
        .await
        .partition_live_row_counts("slot", &strings(&["-1", "1"]))
        .unwrap();
    assert_eq!(counts["-1"], PartitionRowCount::Unknown(MissingRecordCount));
    assert!(matches!(counts["1"], PartitionRowCount::Exact(_)));
    lake.tamper(&format!(
        "UPDATE ducklake_data_file SET record_count = {} WHERE data_file_id = {}",
        victim.max_row_count.unwrap(),
        victim.data_file_id
    ))
    .await;
    check(&lake, "t", None, SLOTS, &[]).await;

    // A second visible delete file for one data file: the roll-up must see
    // both rather than count either.
    let victim = lake
        .table("t", None)
        .await
        .files()
        .unwrap()
        .into_iter()
        .find(|file| file.delete_file_id.is_some())
        .expect("a file with a delete file");
    let slot = victim.partition_values[0].1.clone().unwrap();
    let id = if lake.is_postgres() {
        ""
    } else {
        "delete_file_id, "
    };
    let next = if lake.is_postgres() {
        ""
    } else {
        "(SELECT MAX(delete_file_id) + 1 FROM ducklake_delete_file), "
    };
    lake.tamper(&format!(
        "INSERT INTO ducklake_delete_file ({id}data_file_id, table_id, path, \
         path_is_relative, file_size_bytes, footer_size, delete_count, begin_snapshot, \
         end_snapshot, partial_max) \
         SELECT {next}data_file_id, table_id, path, path_is_relative, file_size_bytes, \
         footer_size, delete_count, begin_snapshot, end_snapshot, partial_max \
         FROM ducklake_delete_file WHERE delete_file_id = {}",
        victim.delete_file_id.unwrap()
    ))
    .await;
    let counts = lake
        .table("t", None)
        .await
        .partition_live_row_counts("slot", &strings(&[slot.as_str()]))
        .unwrap();
    assert_eq!(
        counts[&slot],
        PartitionRowCount::Unknown(MultipleDeleteFiles)
    );
    lake.tamper(&format!(
        "DELETE FROM ducklake_delete_file WHERE data_file_id = {} AND delete_file_id <> {}",
        victim.data_file_id,
        victim.delete_file_id.unwrap()
    ))
    .await;
    check(&lake, "t", None, SLOTS, &[]).await;

    // Rows written before the table was partitioned have no partition value,
    // so no value's count can be told.
    let table_id = lake.create("pre", false).await;
    insert(&lake, "pre", &[0, 1], 0, 3).await;
    lake.partition(table_id).await;
    insert(&lake, "pre", &[0, 1], 100, 3).await;
    check(
        &lake,
        "pre",
        None,
        &["0", "1", "2"],
        &[("0", UnattributedFile), ("1", UnattributedFile), ("2", UnattributedFile)],
    )
    .await;
    // Once those rows are deleted, every visible file is attributable again.
    lake.sql(0, "DELETE FROM lake.main.pre WHERE id < 100")
        .await;
    let counts = check(&lake, "pre", None, &["0", "1", "2"], &[]).await;
    assert_eq!(counts["0"], PartitionRowCount::Exact(3));

    // Files written under a retired spec generation are not attributed through
    // the live one, even on the same column.
    let table_id = lake.create("re", true).await;
    insert(&lake, "re", &[0, 1], 0, 3).await;
    let first_spec = lake.head().await;
    lake.partition(table_id).await;
    insert(&lake, "re", &[0, 1], 100, 3).await;
    check(
        &lake,
        "re",
        None,
        &["0", "1"],
        &[("0", UnattributedFile), ("1", UnattributedFile)],
    )
    .await;
    // Before the re-partition every file belongs to the spec live then.
    check(&lake, "re", Some(first_spec), &["0", "1"], &[]).await;
}

/// A delete file the inlined-data flush writes begins at its earliest
/// deletion and records no `partial_max`, so below the head it may hold
/// deletions made after the read snapshot.
async fn flushed_deletes_scenario(lake: Lake) {
    use PartitionCountUnknown::*;

    lake.create("t", true).await;
    lake.sql(
        10,
        "INSERT INTO lake.main.t VALUES (1, 1, 0), (2, 1, 0), (3, 1, 0)",
    )
    .await;
    let inserted = lake.head().await;
    lake.sql(10, "DELETE FROM lake.main.t WHERE id = 1").await;
    let first_delete = lake.head().await;
    lake.sql(10, "DELETE FROM lake.main.t WHERE id = 2").await;
    let second_delete = lake.head().await;
    {
        let (ctx, catalog) = lake.session(10).await;
        catalog
            .flush_inlined_data(&ctx.state(), Some("main"), Some("t"))
            .await
            .unwrap();
    }
    let counts = check(&lake, "t", None, &["1"], &[]).await;
    assert_eq!(counts["1"], PartitionRowCount::Exact(1));
    let counts = check(&lake, "t", Some(inserted), &["1"], &[]).await;
    assert_eq!(counts["1"], PartitionRowCount::Exact(3));
    for snapshot in [first_delete, second_delete] {
        check(
            &lake,
            "t",
            Some(snapshot),
            &["1"],
            &[("1", PartialDeleteFile)],
        )
        .await;
    }
}

/// Keys other than a lone integer identity key.
async fn key_shapes_scenario(lake: Lake) {
    use PartitionCountUnknown::*;

    let timestamp = DataType::Timestamp(arrow::datatypes::TimeUnit::Microsecond, None);
    let columns = || {
        vec![
            ColumnDef::from_arrow("id", &DataType::Int32, false).unwrap(),
            ColumnDef::from_arrow("region", &DataType::Utf8, true).unwrap(),
            ColumnDef::from_arrow("slot", &DataType::Int32, false).unwrap(),
            ColumnDef::from_arrow("ts", &timestamp, false).unwrap(),
        ]
    };
    // Two keys: `slot` is key 1, and `region` a nullable VARCHAR key.
    lake.create_with(
        "two",
        columns(),
        &[("region", PartitionTransform::Identity), ("slot", PartitionTransform::Identity)],
    )
    .await;
    lake.sql(
        0,
        "INSERT INTO lake.main.two SELECT * FROM (VALUES \
            (1, 'it''s a b', 1, TIMESTAMP '2023-01-15 10:00:00'), \
            (2, 'it''s a b', 2, TIMESTAMP '2023-01-15 10:00:00'), \
            (3, 'us', 1, TIMESTAMP '2024-01-15 10:00:00'), \
            (4, CAST(NULL AS VARCHAR), 1, TIMESTAMP '2024-01-15 10:00:00'), \
            (5, CAST(NULL AS VARCHAR), 2, TIMESTAMP '2024-01-15 10:00:00')) AS v(id, region, slot, ts)",
    )
    .await;
    lake.sql(0, "DELETE FROM lake.main.two WHERE id = 3").await;
    let counts = check_column(&lake, "two", "slot", None, &["1", "2", "3"], &[]).await;
    assert_eq!(counts["1"], PartitionRowCount::Exact(2));
    assert_eq!(counts["2"], PartitionRowCount::Exact(2));
    let counts = check_column(
        &lake,
        "two",
        "region",
        None,
        &["it's a b", "us", "eu", "it's"],
        &[],
    )
    .await;
    assert_eq!(counts["it's a b"], PartitionRowCount::Exact(2));
    assert_eq!(counts["us"], PartitionRowCount::Exact(0));
    // An integer spelled other than as it is stored.
    check_column(
        &lake,
        "two",
        "slot",
        None,
        &["01", "+1", " 1", "1"],
        &[("01", NonCanonicalValue), ("+1", NonCanonicalValue), (" 1", NonCanonicalValue)],
    )
    .await;
    // A column that is a key only through a transform.
    lake.create_with("yr", columns(), &[("ts", PartitionTransform::Year)])
        .await;
    lake.sql(
        0,
        "INSERT INTO lake.main.yr VALUES (1, 'us', 1, TIMESTAMP '2023-01-15 10:00:00')",
    )
    .await;
    let counts = lake
        .table("yr", None)
        .await
        .partition_live_row_counts("ts", &strings(&["2023"]))
        .unwrap();
    assert_eq!(counts["2023"], PartitionRowCount::Unknown(NotIdentityKey));
    // A floating-point key: `0.0` and `-0.0` are separate partitions that
    // compare equal, so no value is counted.
    lake.create_with(
        "fl",
        vec![
            ColumnDef::from_arrow("id", &DataType::Int32, false).unwrap(),
            ColumnDef::from_arrow("f", &DataType::Float64, false).unwrap(),
        ],
        &[("f", PartitionTransform::Identity)],
    )
    .await;
    lake.sql(0, "INSERT INTO lake.main.fl VALUES (1, 0.0), (2, -0.0)")
        .await;
    let counts = lake
        .table("fl", None)
        .await
        .partition_live_row_counts("f", &strings(&["0.0", "-0.0"]))
        .unwrap();
    assert_eq!(counts["0.0"], PartitionRowCount::Unknown(FloatKey));
    assert_eq!(counts["-0.0"], PartitionRowCount::Unknown(FloatKey));
    let counts = lake
        .table("yr", None)
        .await
        .partition_live_row_counts("slot", &strings(&["1"]))
        .unwrap();
    assert_eq!(counts["1"], PartitionRowCount::Unknown(NotAPartitionKey));
    // A column that is two keys of the spec.
    lake.create_with(
        "twice",
        columns(),
        &[("slot", PartitionTransform::Identity), ("slot", PartitionTransform::Bucket(4))],
    )
    .await;
    let counts = lake
        .table("twice", None)
        .await
        .partition_live_row_counts("slot", &strings(&["1"]))
        .unwrap();
    assert_eq!(counts["1"], PartitionRowCount::Unknown(AmbiguousKey));
}

#[cfg(feature = "write-sqlite")]
#[tokio::test(flavor = "multi_thread")]
async fn flushed_deletes_sqlite() {
    flushed_deletes_scenario(Lake::sqlite().await).await;
}

#[cfg(feature = "write-postgres")]
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn flushed_deletes_postgres() {
    flushed_deletes_scenario(Lake::postgres().await).await;
}

#[cfg(feature = "write-sqlite")]
#[tokio::test(flavor = "multi_thread")]
async fn key_shapes_sqlite() {
    key_shapes_scenario(Lake::sqlite().await).await;
}

#[cfg(feature = "write-postgres")]
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn key_shapes_postgres() {
    key_shapes_scenario(Lake::postgres().await).await;
}

#[cfg(feature = "write-sqlite")]
#[tokio::test(flavor = "multi_thread")]
async fn partition_counts_match_a_scan_sqlite() {
    scenario(Lake::sqlite().await).await;
}

#[cfg(feature = "write-postgres")]
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn partition_counts_match_a_scan_postgres() {
    scenario(Lake::postgres().await).await;
}
