//! Catalog-scoped Postgres reader for DuckLake multicatalog.
//!
//! Bound to a single `catalog_id` at construction. All queries that touch
//! catalog-discriminated entities (snapshots, schemas, top-level lists) join
//! through `ducklake_catalog_snapshot_map` / `ducklake_catalog_schema_map`.
//! Queries keyed by a globally unique id (`schema_id`, `table_id`) need no
//! extra scoping because the caller already obtained the id through a
//! catalog-scoped lookup.
//!
//! This implementation is standalone; it does not wrap
//! [`crate::PostgresMetadataProvider`] — the existing single-catalog provider
//! is untouched.

use crate::Result;
use crate::metadata_provider::{
    ColumnWithTable, DataFileChange, DeleteFileChange, DuckLakeFileColumnStatistics,
    DuckLakeFileData, DuckLakeFileMetadata, DuckLakeNameMapping, DuckLakeNameMappingEntry,
    DuckLakeStatistics, DuckLakeTableColumn, DuckLakeTableColumnStatistics, DuckLakeTableField,
    DuckLakeTableFile, DuckLakeTableStatistics, FileWithTable, MetadataProvider, SchemaMetadata,
    SnapshotMetadata, TableMetadata, TableWithSchema, ViewMetadata, ViewWithSchema, block_on,
    reconstruct_columns, reconstruct_columns_with_table,
};
use crate::metadata_provider_postgres::{
    PostgresStatsDialect, StatsFilterSql, fetch_data_file_page, stats_filter_sql,
};
use crate::partition::PartitionSpec;
use crate::sort::SortSpec;
use crate::stats_filter::StatsFilter;
use sqlx::AssertSqlSafe;
use sqlx::Row;
use sqlx::postgres::{PgPool, PgPoolOptions, PgRow};
use sqlx::types::chrono::NaiveDateTime;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

fn is_missing_statistics_table(error: &sqlx::Error) -> bool {
    let message = error.to_string().to_ascii_lowercase();
    message.contains("does not exist") || message.contains("undefined table")
}

fn decode_view(row: &PgRow) -> Result<ViewMetadata> {
    Ok(ViewMetadata {
        view_id: row.try_get(0)?,
        schema_id: row.try_get(1)?,
        begin_snapshot: row.try_get(2)?,
        view_name: row.try_get(3)?,
        dialect: row.try_get(4)?,
        sql: row.try_get(5)?,
        column_aliases: row.try_get(6)?,
    })
}

fn decode_table_file(row: &PgRow, snapshot_id: i64) -> Result<DuckLakeTableFile> {
    let delete_file_id: Option<i64> = row.try_get(8)?;
    let (delete_file, delete_count) = if delete_file_id.is_some() {
        (
            Some(DuckLakeFileData {
                path: row.try_get(9)?,
                path_is_relative: row.try_get(10)?,
                file_size_bytes: row.try_get(11)?,
                footer_size: row.try_get(12)?,
                encryption_key: row.try_get(13)?,
                mapping_id: None,
            }),
            row.try_get(14)?,
        )
    } else {
        (None, None)
    };
    Ok(DuckLakeTableFile {
        data_file_id: row.try_get(0)?,
        file: DuckLakeFileData {
            path: row.try_get(1)?,
            path_is_relative: row.try_get(2)?,
            file_size_bytes: row.try_get(3)?,
            footer_size: row.try_get(4)?,
            encryption_key: row.try_get(5)?,
            mapping_id: row.try_get(19).unwrap_or(None),
        },
        delete_file_id,
        delete_file,
        row_id_start: row.try_get(6)?,
        snapshot_id: Some(snapshot_id),
        begin_snapshot: row.try_get(15)?,
        schema_version: row.try_get(17)?,
        partial_max: row.try_get(16)?,
        max_row_count: row.try_get(7)?,
        delete_count,
        // Column 18 is present on the select-path query (which projects
        // `data.partition_id`) and absent on callers that share this decoder
        // without it; `try_get` failing there degrades to `None`, matching the
        // pre-partition behaviour. Per-key values are filled in by the caller.
        partition_id: row.try_get(18).unwrap_or(None),
        partition_values: Vec::new(),
    })
}

const DEFAULT_MAX_CONNECTIONS: u32 = 5;

/// Optional catalog-schema capabilities probed before scan / CDC queries.
///
/// Minimal / pre-v1.0 catalogs may lack the `partial_max` columns and the
/// `ducklake_schema_versions` ledger; the queries degrade the corresponding
/// projections to NULL when a capability is absent.
#[derive(Debug, Clone, Copy)]
struct SchemaCapabilities {
    /// `ducklake_data_file.partial_max` exists.
    data_file_partial_max: bool,
    /// `ducklake_delete_file.partial_max` exists.
    delete_file_partial_max: bool,
    /// The `ducklake_schema_versions` table exists.
    schema_versions: bool,
    /// `ducklake_data_file.partition_id` exists.
    data_file_partition_id: bool,
    /// The `ducklake_view` table exists.
    views: bool,
    /// The server has `pg_input_is_valid` (PostgreSQL 16+), which
    /// [`PostgresStatsDialect`] needs for its exact `TRY_CAST` stand-in.
    ///
    /// A server capability rather than a catalog one, so it is deliberately not
    /// part of [`Self::all`]: gating the memo on it would make an otherwise
    /// fully-migrated catalog on an older server re-probe on every call, and a
    /// stale `false` only costs pruning.
    soft_input_validation: bool,
}

impl SchemaCapabilities {
    fn all(&self) -> bool {
        self.data_file_partial_max
            && self.delete_file_partial_max
            && self.schema_versions
            && self.data_file_partition_id
            && self.views
    }
}

/// Catalog-scoped Postgres metadata reader.
///
/// Construct with [`Self::with_pool`] (name-keyed; resolves to `catalog_id` once
/// at construction) or [`Self::with_pool_and_id`] (id-keyed; skip the lookup).
#[derive(Debug, Clone)]
pub struct MulticatalogProvider {
    pool: PgPool,
    catalog_id: i64,
    // Positive-only memo of the optional-schema capability probes. `Arc` so
    // derived `Clone` shares the cache across provider clones.
    schema_capabilities: Arc<OnceLock<SchemaCapabilities>>,
}

impl MulticatalogProvider {
    /// Build a pool from a connection string, then resolve the catalog by name.
    pub async fn new(connection_string: &str, catalog_name: &str) -> Result<Self> {
        let pool = PgPoolOptions::new()
            .max_connections(DEFAULT_MAX_CONNECTIONS)
            .connect(connection_string)
            .await?;
        Self::with_pool(pool, catalog_name).await
    }

    /// Bind to an existing pool, resolving the catalog by name.
    ///
    /// Returns [`crate::DuckLakeError::CatalogNotFound`] if no row in
    /// `ducklake_catalog` matches `catalog_name`.
    pub async fn with_pool(pool: PgPool, catalog_name: &str) -> Result<Self> {
        let row = sqlx::query("SELECT catalog_id FROM ducklake_catalog WHERE catalog_name = $1")
            .bind(catalog_name)
            .fetch_optional(&pool)
            .await?;
        let catalog_id: i64 = row
            .ok_or_else(|| crate::DuckLakeError::CatalogNotFound(catalog_name.to_string()))?
            .try_get(0)?;
        Ok(Self {
            pool,
            catalog_id,
            schema_capabilities: Arc::new(OnceLock::new()),
        })
    }

    /// Bind to an existing pool with an already-known `catalog_id`. Skips the
    /// name lookup. Caller is responsible for ensuring the id exists.
    pub async fn with_pool_and_id(pool: PgPool, catalog_id: i64) -> Result<Self> {
        Ok(Self {
            pool,
            catalog_id,
            schema_capabilities: Arc::new(OnceLock::new()),
        })
    }

    pub fn catalog_id(&self) -> i64 {
        self.catalog_id
    }

    /// Whether the schema-capability memo is populated. Exposed for tests.
    #[doc(hidden)]
    pub fn schema_capabilities_cached(&self) -> bool {
        self.schema_capabilities.get().is_some()
    }

    /// Returns the catalog's optional-schema capabilities, probing at most
    /// once per provider lifetime on a fully-migrated catalog.
    ///
    /// Cache-positive-only: capability existence is monotonic (migrations only
    /// add columns/tables, never drop them), so an all-`true` answer is an
    /// immutable fact and safe to memoize. A `false` answer is never cached —
    /// the next call re-probes, so a mid-flight catalog upgrade is picked up
    /// on the next call exactly like the previous per-call probing. Concurrent
    /// first calls may each probe once (one statement each) — harmless; a
    /// raced `set` is ignored.
    async fn schema_capabilities(&self) -> Result<SchemaCapabilities> {
        if let Some(caps) = self.schema_capabilities.get() {
            return Ok(*caps);
        }
        let row: (bool, bool, bool, bool, bool, bool) = sqlx::query_as(
            "SELECT
               EXISTS (SELECT 1 FROM information_schema.columns
                       WHERE table_name = 'ducklake_data_file' AND column_name = 'partial_max'),
               EXISTS (SELECT 1 FROM information_schema.columns
                       WHERE table_name = 'ducklake_delete_file' AND column_name = 'partial_max'),
               to_regclass('ducklake_schema_versions') IS NOT NULL,
               EXISTS (SELECT 1 FROM information_schema.columns
                       WHERE table_name = 'ducklake_data_file' AND column_name = 'partition_id'),
               to_regclass('ducklake_view') IS NOT NULL,
               to_regprocedure('pg_input_is_valid(text,text)') IS NOT NULL",
        )
        .fetch_one(&self.pool)
        .await?;
        let caps = SchemaCapabilities {
            data_file_partial_max: row.0,
            delete_file_partial_max: row.1,
            schema_versions: row.2,
            data_file_partition_id: row.3,
            views: row.4,
            soft_input_validation: row.5,
        };
        if caps.all() {
            let _ = self.schema_capabilities.set(caps);
        }
        Ok(caps)
    }

    /// One page of the visible file listing, optionally narrowed inside SQL by
    /// catalog statistics.
    ///
    /// Backs both [`MetadataProvider::get_table_file_metadata_page`] (`filter`
    /// `None`) and [`MetadataProvider::get_table_file_metadata_page_filtered`],
    /// so the paging contract is written once.
    fn file_metadata_page(
        &self,
        table_id: i64,
        snapshot_id: i64,
        after_data_file_id: Option<i64>,
        limit: usize,
        filter: Option<&StatsFilter>,
    ) -> Result<Vec<DuckLakeFileMetadata>> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let limit = i64::try_from(limit).map_err(|_| {
            crate::DuckLakeError::InvalidConfig("file metadata page limit exceeds i64".to_string())
        })?;
        block_on(async {
            let caps = self.schema_capabilities().await?;
            let partial_max_expr = if caps.data_file_partial_max {
                "data.partial_max::bigint"
            } else {
                "NULL::bigint"
            };
            let schema_version_expr = if caps.schema_versions {
                "(SELECT sv.schema_version::bigint
                  FROM ducklake_schema_versions sv
                  WHERE sv.table_id = data.table_id
                    AND sv.begin_snapshot <= data.begin_snapshot
                  ORDER BY sv.begin_snapshot DESC LIMIT 1)"
            } else {
                "NULL::bigint"
            };
            let dialect = PostgresStatsDialect {
                soft_input_validation: caps.soft_input_validation,
            };
            let rendered = filter.and_then(|filter| filter.render(&dialect));
            let stats_sql = rendered
                .as_deref()
                .and_then(|filters| stats_filter_sql(table_id, filters));

            // The statistics conditions go inside the query, ahead of the
            // LIMIT, with the keyset ordering untouched. Filtering a page after
            // fetching it would break the cursor `FileMetadataPages` drives: a
            // page whose candidates all failed the filter would come back
            // empty, which ends the iteration and hides every matching file
            // beyond it.
            let listing_sql = |stats_sql: Option<&StatsFilterSql>| {
                let (with_prefix, joins, conditions) = stats_sql
                    .map(|sql| {
                        (
                            sql.with_prefix.as_str(),
                            sql.joins.as_str(),
                            sql.conditions.as_str(),
                        )
                    })
                    .unwrap_or_default();
                format!(
                    "{with_prefix}SELECT data.data_file_id, data.path, data.path_is_relative,
                        data.file_size_bytes, data.footer_size, data.encryption_key,
                        data.row_id_start, data.record_count,
                        del.delete_file_id, del.path, del.path_is_relative,
                        del.file_size_bytes, del.footer_size, del.encryption_key,
                        del.delete_count, data.begin_snapshot::bigint,
                        {partial_max_expr}, {schema_version_expr},
                        NULL::bigint AS data_partition_id,
                        data.mapping_id::bigint
                 FROM ducklake_data_file AS data
                 LEFT JOIN ducklake_delete_file AS del
                   ON data.data_file_id = del.data_file_id
                  AND del.table_id = $1
                  AND $2 >= del.begin_snapshot
                  AND ($3 < del.end_snapshot OR del.end_snapshot IS NULL){joins}
                 WHERE data.table_id = $4
                   AND $5 >= data.begin_snapshot
                   AND ($6 < data.end_snapshot OR data.end_snapshot IS NULL)
                   AND data.data_file_id > $7{conditions}
                 ORDER BY data.data_file_id
                 LIMIT $8"
                )
            };

            let after = after_data_file_id.unwrap_or(i64::MIN);
            let rows = match fetch_data_file_page(
                &self.pool,
                &listing_sql(stats_sql.as_ref()),
                table_id,
                snapshot_id,
                after,
                limit,
            )
            .await
            {
                Ok(rows) => rows,
                // The filter is advisory, so a catalog the narrowed query
                // cannot run — one predating `ducklake_file_column_stats`,
                // where joining it is a hard error, or one whose stats provoke
                // an error the dialect did not anticipate — still lists its
                // files. Any error retries, not just the missing table: the
                // narrowed query is the only thing this arm adds, and listing
                // every live file is always correct. The retry uses the same
                // parameters, and a failure that is not the filter's fault
                // surfaces from it.
                Err(error) if stats_sql.is_some() => {
                    tracing::debug!(
                        %error,
                        table_id,
                        "statistics-filtered file listing failed; listing every file"
                    );
                    fetch_data_file_page(
                        &self.pool,
                        &listing_sql(None),
                        table_id,
                        snapshot_id,
                        after,
                        limit,
                    )
                    .await?
                },
                Err(error) => return Err(error.into()),
            };
            let files = rows
                .iter()
                .map(|row| decode_table_file(row, snapshot_id))
                .collect::<Result<Vec<_>>>()?;
            let Some(last_data_file_id) = files.last().map(|file| file.data_file_id) else {
                return Ok(Vec::new());
            };
            // A filtered page's ids are sparse within `(after, last]`, so the
            // two enrichment queries below are restricted to the ids actually
            // returned instead of to that whole range — otherwise a selective
            // filter would read the per-column stats of every file it just
            // pruned, which is the cost the filter exists to avoid.
            //
            // The ids are bound as one array parameter rather than spelled out
            // in the SQL. Inlining a page's worth of them gives every page a
            // distinct query string, and sqlx keys its per-connection
            // prepared-statement cache on that string: each page would pay a
            // Parse/Describe round trip and evict the other statements the
            // connection had cached. The listing query's own `$1..$8` are fixed
            // by `fetch_data_file_page` and untouched by this; the two
            // enrichment queries number their parameters independently.
            let page_ids: Option<Vec<i64>> = stats_sql
                .as_ref()
                .map(|_| files.iter().map(|file| file.data_file_id).collect());
            let page_id_filter = |column: &str, parameter: usize| {
                page_ids.as_ref().map_or_else(String::new, |_| {
                    format!("\n                   AND {column} = ANY(${parameter}::bigint[])")
                })
            };
            let statistics_sql = format!(
                "SELECT stats.data_file_id, stats.column_id,
                        stats.column_size_bytes, stats.value_count, stats.null_count,
                        stats.min_value, stats.max_value, stats.contains_nan
                 FROM ducklake_file_column_stats AS stats
                 INNER JOIN ducklake_data_file AS data
                   ON data.data_file_id = stats.data_file_id
                  AND data.table_id = stats.table_id
                 WHERE stats.table_id = $1
                   AND $2 >= data.begin_snapshot
                   AND ($3 < data.end_snapshot OR data.end_snapshot IS NULL)
                   AND stats.data_file_id > $4
                   AND stats.data_file_id <= $5{}
                 ORDER BY stats.data_file_id, stats.column_id",
                page_id_filter("stats.data_file_id", 6)
            );
            let mut statistics_query = sqlx::query(AssertSqlSafe(statistics_sql))
                .bind(table_id)
                .bind(snapshot_id)
                .bind(snapshot_id)
                .bind(after_data_file_id.unwrap_or(i64::MIN))
                .bind(last_data_file_id);
            if let Some(ids) = page_ids.as_deref() {
                statistics_query = statistics_query.bind(ids);
            }
            let statistics = match statistics_query.fetch_all(&self.pool).await {
                Ok(rows) => rows
                    .into_iter()
                    .map(|row| {
                        Ok(DuckLakeFileColumnStatistics {
                            data_file_id: row.try_get(0)?,
                            column_id: row.try_get(1)?,
                            column_size_bytes: row.try_get(2)?,
                            value_count: row.try_get(3)?,
                            null_count: row.try_get(4)?,
                            min_value: row.try_get(5)?,
                            max_value: row.try_get(6)?,
                            contains_nan: row.try_get(7)?,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?,
                Err(error) if is_missing_statistics_table(&error) => Vec::new(),
                Err(error) => return Err(error.into()),
            };
            let mut statistics_by_file: HashMap<i64, Vec<_>> = HashMap::new();
            for statistic in statistics {
                statistics_by_file
                    .entry(statistic.data_file_id)
                    .or_default()
                    .push(statistic);
            }

            // Enrich with per-file partition values (for pruning), scoped to the
            // page's data_file_id range. Keyed by globally-unique ids; no catalog
            // scoping. Missing partition table => no enrichment.
            let mut values_by_file: HashMap<i64, Vec<(i32, Option<String>)>> = HashMap::new();
            let partition_values_sql = format!(
                "SELECT data_file_id, partition_key_index, partition_value
                 FROM ducklake_file_partition_value
                 WHERE table_id = $1 AND data_file_id > $2 AND data_file_id <= $3{}",
                page_id_filter("data_file_id", 4)
            );
            let mut partition_values_query = sqlx::query(AssertSqlSafe(partition_values_sql))
                .bind(table_id)
                .bind(after_data_file_id.unwrap_or(i64::MIN))
                .bind(last_data_file_id);
            if let Some(ids) = page_ids.as_deref() {
                partition_values_query = partition_values_query.bind(ids);
            }
            match partition_values_query.fetch_all(&self.pool).await {
                Ok(rows) => {
                    for row in rows {
                        let data_file_id: i64 = row.try_get(0)?;
                        let key_index: i32 = i32::try_from(row.try_get::<i64, _>(1)?).unwrap_or(0);
                        let value: Option<String> = row.try_get(2)?;
                        values_by_file
                            .entry(data_file_id)
                            .or_default()
                            .push((key_index, value));
                    }
                },
                Err(error) if is_missing_statistics_table(&error) => {},
                Err(error) => return Err(error.into()),
            }

            Ok(files
                .into_iter()
                .map(|mut file| {
                    if let Some(values) = values_by_file.remove(&file.data_file_id) {
                        file.partition_values = values;
                    }
                    DuckLakeFileMetadata {
                        column_statistics: statistics_by_file
                            .remove(&file.data_file_id)
                            .unwrap_or_default(),
                        file,
                    }
                })
                .collect())
        })
    }
}

impl MetadataProvider for MulticatalogProvider {
    fn get_current_snapshot(&self) -> Result<i64> {
        block_on(async {
            let row = sqlx::query(
                "SELECT COALESCE(MAX(snapshot_id), 0)
                 FROM ducklake_catalog_snapshot_map
                 WHERE catalog_id = $1",
            )
            .bind(self.catalog_id)
            .fetch_one(&self.pool)
            .await?;
            Ok(row.try_get(0)?)
        })
    }

    fn get_data_path(&self) -> Result<String> {
        block_on(async {
            let path: Option<String> = if catalog_has_data_path(&self.pool).await? {
                sqlx::query_scalar("SELECT data_path FROM ducklake_catalog WHERE catalog_id = $1")
                    .bind(self.catalog_id)
                    .fetch_one(&self.pool)
                    .await?
            } else {
                sqlx::query_scalar(
                    "SELECT value FROM ducklake_metadata
                     WHERE key = 'data_path' AND scope IS NULL LIMIT 1",
                )
                .fetch_optional(&self.pool)
                .await?
            };

            path.ok_or_else(|| {
                crate::error::DuckLakeError::InvalidConfig(
                    "Missing required catalog metadata: 'data_path' not configured. \
                     The catalog may be uninitialized or corrupted."
                        .to_string(),
                )
            })
        })
    }

    fn list_snapshots(&self) -> Result<Vec<SnapshotMetadata>> {
        block_on(async {
            let rows = sqlx::query(
                "SELECT s.snapshot_id, s.snapshot_time
                 FROM ducklake_snapshot s
                 JOIN ducklake_catalog_snapshot_map m ON m.snapshot_id = s.snapshot_id
                 WHERE m.catalog_id = $1
                 ORDER BY s.snapshot_id",
            )
            .bind(self.catalog_id)
            .fetch_all(&self.pool)
            .await?;

            rows.into_iter()
                .map(|row| {
                    let snapshot_id: i64 = row.try_get(0)?;
                    let timestamp: Option<NaiveDateTime> = row.try_get(1)?;
                    let timestamp_str =
                        timestamp.map(|ts| ts.format("%Y-%m-%d %H:%M:%S%.6f").to_string());
                    Ok(SnapshotMetadata {
                        snapshot_id,
                        timestamp: timestamp_str,
                    })
                })
                .collect()
        })
    }

    fn list_schemas(&self, snapshot_id: i64) -> Result<Vec<SchemaMetadata>> {
        block_on(async {
            let rows = sqlx::query(
                "SELECT s.schema_id, s.schema_name, s.path, s.path_is_relative
                 FROM ducklake_schema s
                 JOIN ducklake_catalog_schema_map m ON m.schema_id = s.schema_id
                 WHERE m.catalog_id = $1
                   AND $2 >= s.begin_snapshot
                   AND ($3 < s.end_snapshot OR s.end_snapshot IS NULL)",
            )
            .bind(self.catalog_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .fetch_all(&self.pool)
            .await?;

            rows.into_iter()
                .map(|row| {
                    Ok(SchemaMetadata {
                        schema_id: row.try_get(0)?,
                        schema_name: row.try_get(1)?,
                        path: row.try_get(2)?,
                        path_is_relative: row.try_get(3)?,
                    })
                })
                .collect()
        })
    }

    fn list_tables(&self, schema_id: i64, snapshot_id: i64) -> Result<Vec<TableMetadata>> {
        // schema_id is globally unique; caller has already resolved it via
        // get_schema_by_name (catalog-scoped). No additional scoping needed.
        block_on(async {
            let rows = sqlx::query(
                "SELECT table_id, table_name, path, path_is_relative
                 FROM ducklake_table
                 WHERE schema_id = $1
                   AND $2 >= begin_snapshot
                   AND ($3 < end_snapshot OR end_snapshot IS NULL)",
            )
            .bind(schema_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .fetch_all(&self.pool)
            .await?;

            rows.into_iter()
                .map(|row| {
                    Ok(TableMetadata {
                        table_id: row.try_get(0)?,
                        table_name: row.try_get(1)?,
                        path: row.try_get(2)?,
                        path_is_relative: row.try_get(3)?,
                    })
                })
                .collect()
        })
    }

    fn list_views(&self, schema_id: i64, snapshot_id: i64) -> Result<Vec<ViewMetadata>> {
        block_on(async {
            if !self.schema_capabilities().await?.views {
                return Ok(Vec::new());
            }
            let rows = sqlx::query(
                "SELECT view_id, schema_id, begin_snapshot, view_name, dialect, sql, column_aliases
                 FROM ducklake_view
                 WHERE schema_id = $1
                   AND $2 >= begin_snapshot
                   AND ($3 < end_snapshot OR end_snapshot IS NULL)",
            )
            .bind(schema_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .fetch_all(&self.pool)
            .await?;

            rows.into_iter().map(|row| decode_view(&row)).collect()
        })
    }

    fn get_table_structure(
        &self,
        table_id: i64,
        snapshot_id: i64,
    ) -> Result<Vec<DuckLakeTableColumn>> {
        // Columns inherit catalog via table_id (a table belongs to exactly one
        // catalog), but must still be SNAPSHOT-scoped like list_tables /
        // list_schemas: reading by `end_snapshot IS NULL` alone leaks a
        // concurrent or aborted writer's begin-time column generation (which
        // commits before the head advances). Match the catalog head window.
        block_on(async {
            let rows = sqlx::query(
                "SELECT column_id, column_name, column_type, nulls_allowed, parent_column,
                        initial_default, default_value, default_value_type, default_value_dialect
                 FROM ducklake_column
                 WHERE table_id = $1
                   AND $2 >= begin_snapshot
                   AND ($3 < end_snapshot OR end_snapshot IS NULL)
                 ORDER BY column_order",
            )
            .bind(table_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .fetch_all(&self.pool)
            .await?;

            let raw: Result<Vec<(DuckLakeTableColumn, Option<i64>)>> = rows
                .into_iter()
                .map(|row| {
                    let nulls_allowed: Option<bool> = row.try_get(3)?;
                    let parent_column: Option<i64> = row.try_get(4)?;
                    Ok((
                        DuckLakeTableColumn::new(
                            row.try_get(0)?,
                            row.try_get(1)?,
                            row.try_get(2)?,
                            nulls_allowed.unwrap_or(true),
                        )
                        .with_defaults(
                            row.try_get(5)?,
                            row.try_get(6)?,
                            row.try_get(7)?,
                            row.try_get(8)?,
                        ),
                        parent_column,
                    ))
                })
                .collect();
            reconstruct_columns(raw?)
        })
    }

    fn get_table_fields(&self, table_id: i64, snapshot_id: i64) -> Result<Vec<DuckLakeTableField>> {
        block_on(async {
            let rows = sqlx::query(
                "SELECT column_id, column_name, column_type, nulls_allowed, parent_column
                 FROM ducklake_column
                 WHERE table_id = $1
                   AND $2 >= begin_snapshot
                   AND ($3 < end_snapshot OR end_snapshot IS NULL)
                 ORDER BY column_order",
            )
            .bind(table_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .fetch_all(&self.pool)
            .await?;
            rows.into_iter()
                .map(|row| {
                    Ok(DuckLakeTableField {
                        column_id: row.try_get(0)?,
                        column_name: row.try_get(1)?,
                        column_type: row.try_get(2)?,
                        is_nullable: row.try_get::<Option<bool>, _>(3)?.unwrap_or(true),
                        parent_column: row.try_get(4)?,
                    })
                })
                .collect()
        })
    }

    fn get_name_mapping(&self, mapping_id: i64) -> Result<DuckLakeNameMapping> {
        block_on(async {
            let rows = sqlx::query(
                "SELECT mapping.mapping_id, mapping.table_id, mapping.type,
                        name.column_id, name.source_name, name.target_field_id,
                        name.parent_column, name.is_partition
                 FROM ducklake_column_mapping AS mapping
                 JOIN ducklake_name_mapping AS name
                   ON name.mapping_id = mapping.mapping_id
                 WHERE mapping.mapping_id = $1
                 ORDER BY name.parent_column NULLS FIRST, name.column_id",
            )
            .bind(mapping_id)
            .fetch_all(&self.pool)
            .await?;
            let first = rows.first().ok_or_else(|| {
                crate::DuckLakeError::InvalidConfig(format!(
                    "DuckLake name mapping {mapping_id} does not exist"
                ))
            })?;
            let mut entries = Vec::new();
            for row in &rows {
                if let Some(column_id) = row.try_get::<Option<i64>, _>(3)? {
                    entries.push(DuckLakeNameMappingEntry {
                        column_id,
                        source_name: row.try_get(4)?,
                        target_field_id: row.try_get(5)?,
                        parent_column: row.try_get(6)?,
                        is_partition: row.try_get::<Option<bool>, _>(7)?.unwrap_or(false),
                    });
                }
            }
            Ok(DuckLakeNameMapping {
                mapping_id: first.try_get(0)?,
                table_id: first.try_get(1)?,
                mapping_type: first.try_get(2)?,
                entries,
            })
        })
    }

    fn get_table_files_for_select(
        &self,
        table_id: i64,
        snapshot_id: i64,
    ) -> Result<Vec<DuckLakeTableFile>> {
        // Files inherit catalog via table_id. No scoping.
        block_on(async {
            // Backward compatibility (mirrors the single-catalog Postgres
            // provider): degrade `partial_max` / `schema_version` to NULL when a
            // catalog predates the v1.0 partial-file column or the
            // `ducklake_schema_versions` ledger. Surfacing these lets compaction
            // (which pairs its writer with THIS reader) select merge candidates
            // and lets time-travel reads filter partial files per-row.
            let caps = self.schema_capabilities().await?;
            let partial_max_expr = if caps.data_file_partial_max {
                "data.partial_max::bigint"
            } else {
                "NULL::bigint"
            };
            // A catalog predating partition support has no `partition_id` column;
            // such a catalog holds no partitioned files either, so NULL is exact.
            let partition_id_expr = if caps.data_file_partition_id {
                "data.partition_id::bigint"
            } else {
                "NULL::bigint"
            };
            let schema_version_expr = if caps.schema_versions {
                "(SELECT sv.schema_version::bigint
                  FROM ducklake_schema_versions sv
                  WHERE sv.table_id = data.table_id
                    AND sv.begin_snapshot <= data.begin_snapshot
                  ORDER BY sv.begin_snapshot DESC
                  LIMIT 1)"
            } else {
                "NULL::bigint"
            };
            let sql = format!(
                "SELECT
                    data.data_file_id,
                    data.path AS data_file_path,
                    data.path_is_relative AS data_path_is_relative,
                    data.file_size_bytes AS data_file_size,
                    data.footer_size AS data_footer_size,
                    data.encryption_key AS data_encryption_key,
                    data.row_id_start AS data_row_id_start,
                    data.record_count AS data_record_count,
                    del.delete_file_id,
                    del.path AS delete_file_path,
                    del.path_is_relative AS delete_path_is_relative,
                    del.file_size_bytes AS delete_file_size,
                    del.footer_size AS delete_footer_size,
                    del.encryption_key AS delete_encryption_key,
                    del.delete_count,
                    data.begin_snapshot::bigint AS data_begin_snapshot,
                    {partial_max_expr} AS data_partial_max,
                    {schema_version_expr} AS data_schema_version,
                    {partition_id_expr} AS data_partition_id,
                    data.mapping_id::bigint AS data_mapping_id
                FROM ducklake_data_file AS data
                LEFT JOIN ducklake_delete_file AS del
                    ON data.data_file_id = del.data_file_id
                    AND del.table_id = $1
                    AND $2 >= del.begin_snapshot
                    AND ($3 < del.end_snapshot OR del.end_snapshot IS NULL)
                WHERE data.table_id = $4
                  AND $5 >= data.begin_snapshot
                  AND ($6 < data.end_snapshot OR data.end_snapshot IS NULL)"
            );
            let rows = sqlx::query(AssertSqlSafe(sql.as_str()))
                .bind(table_id)
                .bind(snapshot_id)
                .bind(snapshot_id)
                .bind(table_id)
                .bind(snapshot_id)
                .bind(snapshot_id)
                .fetch_all(&self.pool)
                .await?;

            let mut files: Vec<DuckLakeTableFile> = rows
                .iter()
                .map(|row| decode_table_file(row, snapshot_id))
                .collect::<Result<Vec<_>>>()?;

            // Enrich with per-file partition values. Compaction reads its candidates
            // through this path and must group by, and preserve, each file's exact
            // partition — without the values it would merge across partitions and
            // strip the assignment. Scoped by the fetched id range; a catalog
            // predating partition support simply yields no rows.
            if let (Some(min), Some(max)) = (
                files.iter().map(|f| f.data_file_id).min(),
                files.iter().map(|f| f.data_file_id).max(),
            ) {
                let mut values_by_file: HashMap<i64, Vec<(i32, Option<String>)>> = HashMap::new();
                match sqlx::query(
                    "SELECT data_file_id, partition_key_index, partition_value
                     FROM ducklake_file_partition_value
                     WHERE table_id = $1 AND data_file_id >= $2 AND data_file_id <= $3",
                )
                .bind(table_id)
                .bind(min)
                .bind(max)
                .fetch_all(&self.pool)
                .await
                {
                    Ok(rows) => {
                        for row in rows {
                            let data_file_id: i64 = row.try_get(0)?;
                            let key_index: i32 =
                                i32::try_from(row.try_get::<i64, _>(1)?).unwrap_or(0);
                            let value: Option<String> = row.try_get(2)?;
                            values_by_file
                                .entry(data_file_id)
                                .or_default()
                                .push((key_index, value));
                        }
                    },
                    Err(error) if is_missing_statistics_table(&error) => {},
                    Err(error) => return Err(error.into()),
                }
                for file in &mut files {
                    if let Some(values) = values_by_file.remove(&file.data_file_id) {
                        file.partition_values = values;
                    }
                }
            }
            Ok(files)
        })
    }

    fn get_partition_spec(&self, table_id: i64, snapshot_id: i64) -> Result<Option<PartitionSpec>> {
        // Keyed by the globally-unique table_id, so no catalog scoping is needed.
        block_on(async {
            let generation_count: i64 = match sqlx::query_scalar(
                "SELECT COUNT(*) FROM ducklake_partition_info WHERE table_id = $1",
            )
            .bind(table_id)
            .fetch_one(&self.pool)
            .await
            {
                Ok(count) => count,
                Err(error) if is_missing_statistics_table(&error) => return Ok(None),
                Err(error) => return Err(error.into()),
            };
            let prune_safe = generation_count == 1;
            let rows = match sqlx::query(
                "SELECT pi.partition_id, pc.partition_key_index, pc.column_id, pc.transform
                 FROM ducklake_partition_info AS pi
                 JOIN ducklake_partition_column AS pc
                   ON pc.partition_id = pi.partition_id AND pc.table_id = pi.table_id
                 WHERE pi.table_id = $1
                   AND $2 >= pi.begin_snapshot
                   AND ($3 < pi.end_snapshot OR pi.end_snapshot IS NULL)
                 ORDER BY pc.partition_key_index",
            )
            .bind(table_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .fetch_all(&self.pool)
            .await
            {
                Ok(rows) => rows,
                Err(error) if is_missing_statistics_table(&error) => return Ok(None),
                Err(error) => return Err(error.into()),
            };
            let parsed = rows
                .iter()
                .map(|row| {
                    Ok::<_, crate::DuckLakeError>((
                        row.try_get::<i64, _>(0)?,
                        i32::try_from(row.try_get::<i64, _>(1)?).unwrap_or(0),
                        row.try_get::<i64, _>(2)?,
                        row.try_get::<String, _>(3)?,
                    ))
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(PartitionSpec::from_rows(parsed, prune_safe))
        })
    }

    fn get_sort_spec(&self, table_id: i64, snapshot_id: i64) -> Result<Option<SortSpec>> {
        // Keyed by the globally unique table_id, so no catalog scoping is needed.
        block_on(async {
            let rows = match sqlx::query(
                "SELECT si.sort_id, se.sort_key_index, se.expression, se.dialect,
                        se.sort_direction, se.null_order
                 FROM ducklake_sort_info AS si
                 JOIN ducklake_sort_expression AS se
                   ON se.sort_id = si.sort_id AND se.table_id = si.table_id
                 WHERE si.table_id = $1
                   AND $2 >= si.begin_snapshot
                   AND ($3 < si.end_snapshot OR si.end_snapshot IS NULL)
                 ORDER BY se.sort_key_index",
            )
            .bind(table_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .fetch_all(&self.pool)
            .await
            {
                Ok(rows) => rows,
                Err(error) if is_missing_statistics_table(&error) => return Ok(None),
                Err(error) => return Err(error.into()),
            };
            let parsed = rows
                .iter()
                .map(|row| {
                    Ok::<_, crate::DuckLakeError>((
                        row.try_get::<i64, _>(0)?,
                        i32::try_from(row.try_get::<i64, _>(1)?).unwrap_or(0),
                        row.try_get::<String, _>(2)?,
                        row.try_get::<String, _>(3)?,
                        row.try_get::<String, _>(4)?,
                        row.try_get::<String, _>(5)?,
                    ))
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(SortSpec::from_rows(parsed))
        })
    }

    fn get_table_file_metadata_page(
        &self,
        table_id: i64,
        snapshot_id: i64,
        after_data_file_id: Option<i64>,
        limit: usize,
    ) -> Result<Vec<DuckLakeFileMetadata>> {
        self.file_metadata_page(table_id, snapshot_id, after_data_file_id, limit, None)
    }

    fn get_table_file_metadata_page_filtered(
        &self,
        table_id: i64,
        snapshot_id: i64,
        after_data_file_id: Option<i64>,
        limit: usize,
        filter: Option<&StatsFilter>,
    ) -> Result<Vec<DuckLakeFileMetadata>> {
        self.file_metadata_page(table_id, snapshot_id, after_data_file_id, limit, filter)
    }

    fn get_table_summary_statistics(
        &self,
        table_id: i64,
        snapshot_id: i64,
    ) -> Result<DuckLakeStatistics> {
        block_on(async {
            let table = match sqlx::query(
                "SELECT record_count, file_size_bytes
                 FROM ducklake_table_stats WHERE table_id = $1",
            )
            .bind(table_id)
            .fetch_optional(&self.pool)
            .await
            {
                Ok(row) => row
                    .map(|row| {
                        Ok::<_, sqlx::Error>(DuckLakeTableStatistics {
                            record_count: row.try_get(0)?,
                            file_size_bytes: row.try_get(1)?,
                        })
                    })
                    .transpose()?,
                Err(error) if is_missing_statistics_table(&error) => None,
                Err(error) => return Err(error.into()),
            };
            let column_sizes = match sqlx::query(
                "SELECT stats.column_id,
                        CASE
                          WHEN COUNT(*) = COUNT(stats.column_size_bytes)
                           AND COUNT(*) = (
                             SELECT COUNT(*) FROM ducklake_data_file visible
                             WHERE visible.table_id = $1
                               AND $2 >= visible.begin_snapshot
                               AND ($3 < visible.end_snapshot OR visible.end_snapshot IS NULL)
                           )
                          THEN CAST(SUM(stats.column_size_bytes) AS BIGINT)
                        END
                 FROM ducklake_file_column_stats stats
                 INNER JOIN ducklake_data_file data
                   ON data.data_file_id = stats.data_file_id
                  AND data.table_id = stats.table_id
                 WHERE stats.table_id = $4
                   AND $5 >= data.begin_snapshot
                   AND ($6 < data.end_snapshot OR data.end_snapshot IS NULL)
                 GROUP BY stats.column_id",
            )
            .bind(table_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .bind(table_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .fetch_all(&self.pool)
            .await
            {
                Ok(rows) => rows
                    .into_iter()
                    .filter_map(|row| match row.try_get::<Option<i64>, _>(1) {
                        Ok(Some(size)) => Some(row.try_get(0).map(|column_id| (column_id, size))),
                        Ok(None) => None,
                        Err(error) => Some(Err(error)),
                    })
                    .collect::<std::result::Result<HashMap<i64, i64>, _>>()?,
                Err(error) if is_missing_statistics_table(&error) => HashMap::new(),
                Err(error) => return Err(error.into()),
            };
            let bounds_are_exact: bool = sqlx::query_scalar(
                "SELECT NOT EXISTS (
                     SELECT 1 FROM ducklake_delete_file
                     WHERE table_id = $1
                       AND $2 >= begin_snapshot
                       AND ($3 < end_snapshot OR end_snapshot IS NULL)
                 )",
            )
            .bind(table_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .fetch_one(&self.pool)
            .await?;
            let columns = match sqlx::query(
                "SELECT column_id, contains_null, min_value, max_value, contains_nan
                 FROM ducklake_table_column_stats WHERE table_id = $1",
            )
            .bind(table_id)
            .fetch_all(&self.pool)
            .await
            {
                Ok(rows) => rows
                    .into_iter()
                    .map(|row| {
                        let column_id = row.try_get(0)?;
                        Ok(DuckLakeTableColumnStatistics {
                            column_id,
                            contains_null: row.try_get(1)?,
                            min_value: row.try_get(2)?,
                            max_value: row.try_get(3)?,
                            contains_nan: row.try_get(4)?,
                            column_size_bytes: column_sizes.get(&column_id).copied(),
                            bounds_are_exact,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?,
                Err(error) if is_missing_statistics_table(&error) => Vec::new(),
                Err(error) => return Err(error.into()),
            };
            Ok(DuckLakeStatistics {
                table,
                columns,
                files: Vec::new(),
            })
        })
    }

    fn get_table_statistics(&self, table_id: i64, snapshot_id: i64) -> Result<DuckLakeStatistics> {
        block_on(async {
            let table = match sqlx::query(
                "SELECT record_count, file_size_bytes
                 FROM ducklake_table_stats WHERE table_id = $1",
            )
            .bind(table_id)
            .fetch_optional(&self.pool)
            .await
            {
                Ok(row) => row
                    .map(|row| {
                        Ok::<_, sqlx::Error>(DuckLakeTableStatistics {
                            record_count: row.try_get(0)?,
                            file_size_bytes: row.try_get(1)?,
                        })
                    })
                    .transpose()?,
                Err(error) if is_missing_statistics_table(&error) => None,
                Err(error) => return Err(error.into()),
            };

            let columns = match sqlx::query(
                "SELECT column_id, contains_null, min_value, max_value, contains_nan
                 FROM ducklake_table_column_stats WHERE table_id = $1",
            )
            .bind(table_id)
            .fetch_all(&self.pool)
            .await
            {
                Ok(rows) => rows
                    .into_iter()
                    .map(|row| {
                        Ok(DuckLakeTableColumnStatistics {
                            column_id: row.try_get(0)?,
                            contains_null: row.try_get(1)?,
                            min_value: row.try_get(2)?,
                            max_value: row.try_get(3)?,
                            contains_nan: row.try_get(4)?,
                            column_size_bytes: None,
                            bounds_are_exact: false,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?,
                Err(error) if is_missing_statistics_table(&error) => Vec::new(),
                Err(error) => return Err(error.into()),
            };

            let files = match sqlx::query(
                "SELECT
                    stats.data_file_id,
                    stats.column_id,
                    stats.column_size_bytes,
                    stats.value_count,
                    stats.null_count,
                    stats.min_value,
                    stats.max_value,
                    stats.contains_nan
                 FROM ducklake_file_column_stats AS stats
                 INNER JOIN ducklake_data_file AS data
                    ON data.data_file_id = stats.data_file_id
                    AND data.table_id = stats.table_id
                 WHERE stats.table_id = $1
                   AND $2 >= data.begin_snapshot
                   AND ($3 < data.end_snapshot OR data.end_snapshot IS NULL)",
            )
            .bind(table_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .fetch_all(&self.pool)
            .await
            {
                Ok(rows) => rows
                    .into_iter()
                    .map(|row| {
                        Ok(DuckLakeFileColumnStatistics {
                            data_file_id: row.try_get(0)?,
                            column_id: row.try_get(1)?,
                            column_size_bytes: row.try_get(2)?,
                            value_count: row.try_get(3)?,
                            null_count: row.try_get(4)?,
                            min_value: row.try_get(5)?,
                            max_value: row.try_get(6)?,
                            contains_nan: row.try_get(7)?,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?,
                Err(error) if is_missing_statistics_table(&error) => Vec::new(),
                Err(error) => return Err(error.into()),
            };

            Ok(DuckLakeStatistics {
                table,
                columns,
                files,
            })
        })
    }

    fn get_schema_by_name(&self, name: &str, snapshot_id: i64) -> Result<Option<SchemaMetadata>> {
        block_on(async {
            let row = sqlx::query(
                "SELECT s.schema_id, s.schema_name, s.path, s.path_is_relative
                 FROM ducklake_schema s
                 JOIN ducklake_catalog_schema_map m ON m.schema_id = s.schema_id
                 WHERE m.catalog_id = $1
                   AND s.schema_name = $2
                   AND $3 >= s.begin_snapshot
                   AND ($4 < s.end_snapshot OR s.end_snapshot IS NULL)",
            )
            .bind(self.catalog_id)
            .bind(name)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .fetch_optional(&self.pool)
            .await?;

            match row {
                Some(r) => Ok(Some(SchemaMetadata {
                    schema_id: r.try_get(0)?,
                    schema_name: r.try_get(1)?,
                    path: r.try_get(2)?,
                    path_is_relative: r.try_get(3)?,
                })),
                None => Ok(None),
            }
        })
    }

    fn get_table_by_name(
        &self,
        schema_id: i64,
        name: &str,
        snapshot_id: i64,
    ) -> Result<Option<TableMetadata>> {
        // schema_id catalog-scoped by caller.
        block_on(async {
            let row = sqlx::query(
                "SELECT table_id, table_name, path, path_is_relative
                 FROM ducklake_table
                 WHERE schema_id = $1
                   AND table_name = $2
                   AND $3 >= begin_snapshot
                   AND ($4 < end_snapshot OR end_snapshot IS NULL)",
            )
            .bind(schema_id)
            .bind(name)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .fetch_optional(&self.pool)
            .await?;

            match row {
                Some(r) => Ok(Some(TableMetadata {
                    table_id: r.try_get(0)?,
                    table_name: r.try_get(1)?,
                    path: r.try_get(2)?,
                    path_is_relative: r.try_get(3)?,
                })),
                None => Ok(None),
            }
        })
    }

    fn get_view_by_name(
        &self,
        schema_id: i64,
        name: &str,
        snapshot_id: i64,
    ) -> Result<Option<ViewMetadata>> {
        block_on(async {
            if !self.schema_capabilities().await?.views {
                return Ok(None);
            }
            let row = sqlx::query(
                "SELECT view_id, schema_id, begin_snapshot, view_name, dialect, sql, column_aliases
                 FROM ducklake_view
                 WHERE schema_id = $1
                   AND view_name = $2
                   AND $3 >= begin_snapshot
                   AND ($4 < end_snapshot OR end_snapshot IS NULL)",
            )
            .bind(schema_id)
            .bind(name)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .fetch_optional(&self.pool)
            .await?;

            row.map(|row| decode_view(&row)).transpose()
        })
    }

    fn table_exists(&self, schema_id: i64, name: &str, snapshot_id: i64) -> Result<bool> {
        block_on(async {
            let row = sqlx::query(
                "SELECT EXISTS(
                    SELECT 1 FROM ducklake_table
                    WHERE schema_id = $1
                      AND table_name = $2
                      AND $3 >= begin_snapshot
                      AND ($4 < end_snapshot OR end_snapshot IS NULL)
                 )",
            )
            .bind(schema_id)
            .bind(name)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .fetch_one(&self.pool)
            .await?;
            Ok(row.try_get(0)?)
        })
    }

    fn list_all_tables(&self, snapshot_id: i64) -> Result<Vec<TableWithSchema>> {
        block_on(async {
            let rows = sqlx::query(
                "SELECT s.schema_name, t.table_id, t.table_name, t.path, t.path_is_relative
                 FROM ducklake_schema s
                 JOIN ducklake_catalog_schema_map m ON m.schema_id = s.schema_id
                 JOIN ducklake_table t ON s.schema_id = t.schema_id
                 WHERE m.catalog_id = $1
                   AND $2 >= s.begin_snapshot
                   AND ($3 < s.end_snapshot OR s.end_snapshot IS NULL)
                   AND $4 >= t.begin_snapshot
                   AND ($5 < t.end_snapshot OR t.end_snapshot IS NULL)
                 ORDER BY s.schema_name, t.table_name",
            )
            .bind(self.catalog_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .fetch_all(&self.pool)
            .await?;

            rows.into_iter()
                .map(|row| {
                    let schema_name: String = row.try_get(0)?;
                    let table = TableMetadata {
                        table_id: row.try_get(1)?,
                        table_name: row.try_get(2)?,
                        path: row.try_get(3)?,
                        path_is_relative: row.try_get(4)?,
                    };
                    Ok(TableWithSchema {
                        schema_name,
                        table,
                    })
                })
                .collect()
        })
    }

    fn list_all_views(&self, snapshot_id: i64) -> Result<Vec<ViewWithSchema>> {
        block_on(async {
            if !self.schema_capabilities().await?.views {
                return Ok(Vec::new());
            }
            let rows = sqlx::query(
                "SELECT s.schema_name, v.view_id, v.schema_id, v.begin_snapshot, v.view_name,
                        v.dialect, v.sql, v.column_aliases
                 FROM ducklake_schema s
                 JOIN ducklake_catalog_schema_map m ON m.schema_id = s.schema_id
                 JOIN ducklake_view v ON s.schema_id = v.schema_id
                 WHERE m.catalog_id = $1
                   AND $2 >= s.begin_snapshot
                   AND ($2 < s.end_snapshot OR s.end_snapshot IS NULL)
                   AND $2 >= v.begin_snapshot
                   AND ($2 < v.end_snapshot OR v.end_snapshot IS NULL)
                 ORDER BY s.schema_name, v.view_name",
            )
            .bind(self.catalog_id)
            .bind(snapshot_id)
            .fetch_all(&self.pool)
            .await?;
            rows.into_iter()
                .map(|row| {
                    Ok(ViewWithSchema {
                        schema_name: row.try_get(0)?,
                        view: ViewMetadata {
                            view_id: row.try_get(1)?,
                            schema_id: row.try_get(2)?,
                            begin_snapshot: row.try_get(3)?,
                            view_name: row.try_get(4)?,
                            dialect: row.try_get(5)?,
                            sql: row.try_get(6)?,
                            column_aliases: row.try_get(7)?,
                        },
                    })
                })
                .collect()
        })
    }

    fn list_all_columns(&self, snapshot_id: i64) -> Result<Vec<ColumnWithTable>> {
        // Note: unlike the single-catalog PostgresMetadataProvider, this filters
        // columns by snapshot range as well — needed for multicatalog because we
        // accumulate column history across catalogs and would otherwise return
        // ended columns alongside current ones.
        block_on(async {
            let rows = sqlx::query(
                "SELECT s.schema_name, t.table_name, c.column_id, c.column_name, c.column_type,
                        c.nulls_allowed, c.parent_column, c.initial_default, c.default_value,
                        c.default_value_type, c.default_value_dialect
                 FROM ducklake_schema s
                 JOIN ducklake_catalog_schema_map m ON m.schema_id = s.schema_id
                 JOIN ducklake_table t ON s.schema_id = t.schema_id
                 JOIN ducklake_column c ON t.table_id = c.table_id
                 WHERE m.catalog_id = $1
                   AND $2 >= s.begin_snapshot
                   AND ($3 < s.end_snapshot OR s.end_snapshot IS NULL)
                   AND $4 >= t.begin_snapshot
                   AND ($5 < t.end_snapshot OR t.end_snapshot IS NULL)
                   AND $6 >= c.begin_snapshot
                   AND ($7 < c.end_snapshot OR c.end_snapshot IS NULL)
                 ORDER BY s.schema_name, t.table_name, c.column_order",
            )
            .bind(self.catalog_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .fetch_all(&self.pool)
            .await?;

            let raw: Result<Vec<(ColumnWithTable, Option<i64>)>> = rows
                .into_iter()
                .map(|row| {
                    let schema_name: String = row.try_get(0)?;
                    let table_name: String = row.try_get(1)?;
                    let nulls_allowed: Option<bool> = row.try_get(5)?;
                    let parent_column: Option<i64> = row.try_get(6)?;
                    let column = DuckLakeTableColumn::new(
                        row.try_get(2)?,
                        row.try_get(3)?,
                        row.try_get(4)?,
                        nulls_allowed.unwrap_or(true),
                    )
                    .with_defaults(
                        row.try_get(7)?,
                        row.try_get(8)?,
                        row.try_get(9)?,
                        row.try_get(10)?,
                    );
                    Ok((
                        ColumnWithTable {
                            schema_name,
                            table_name,
                            column,
                        },
                        parent_column,
                    ))
                })
                .collect();
            reconstruct_columns_with_table(raw?)
        })
    }

    fn list_all_files(&self, snapshot_id: i64) -> Result<Vec<FileWithTable>> {
        block_on(async {
            let rows = sqlx::query(
                "SELECT
                    s.schema_name,
                    t.table_name,
                    data.data_file_id,
                    data.path AS data_file_path,
                    data.path_is_relative AS data_path_is_relative,
                    data.file_size_bytes AS data_file_size,
                    data.footer_size AS data_footer_size,
                    data.encryption_key AS data_encryption_key,
                    del.delete_file_id,
                    del.path AS delete_file_path,
                    del.path_is_relative AS delete_path_is_relative,
                    del.file_size_bytes AS delete_file_size,
                    del.footer_size AS delete_footer_size,
                    del.encryption_key AS delete_encryption_key,
                    del.delete_count
                FROM ducklake_schema s
                JOIN ducklake_catalog_schema_map m ON m.schema_id = s.schema_id
                JOIN ducklake_table t ON s.schema_id = t.schema_id
                JOIN ducklake_data_file data ON t.table_id = data.table_id
                LEFT JOIN ducklake_delete_file del
                    ON data.data_file_id = del.data_file_id
                    AND del.table_id = t.table_id
                    AND $1 >= del.begin_snapshot
                    AND ($2 < del.end_snapshot OR del.end_snapshot IS NULL)
                WHERE m.catalog_id = $3
                  AND $4 >= s.begin_snapshot
                  AND ($5 < s.end_snapshot OR s.end_snapshot IS NULL)
                  AND $6 >= t.begin_snapshot
                  AND ($7 < t.end_snapshot OR t.end_snapshot IS NULL)
                  AND $8 >= data.begin_snapshot
                  AND ($9 < data.end_snapshot OR data.end_snapshot IS NULL)
                ORDER BY s.schema_name, t.table_name, data.path",
            )
            .bind(snapshot_id)
            .bind(snapshot_id)
            .bind(self.catalog_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .bind(snapshot_id)
            .fetch_all(&self.pool)
            .await?;

            rows.into_iter()
                .map(|row| {
                    let data_file = DuckLakeFileData {
                        path: row.try_get(3)?,
                        path_is_relative: row.try_get(4)?,
                        file_size_bytes: row.try_get(5)?,
                        footer_size: row.try_get(6)?,
                        encryption_key: row.try_get(7)?,
                        mapping_id: None,
                    };
                    let delete_file = if row.try_get::<Option<i64>, _>(8)?.is_some() {
                        Some(DuckLakeFileData {
                            path: row.try_get(9)?,
                            path_is_relative: row.try_get(10)?,
                            file_size_bytes: row.try_get(11)?,
                            footer_size: row.try_get(12)?,
                            encryption_key: row.try_get(13)?,
                            mapping_id: None,
                        })
                    } else {
                        None
                    };
                    Ok(FileWithTable {
                        schema_name: row.try_get(0)?,
                        table_name: row.try_get(1)?,
                        file: DuckLakeTableFile {
                            data_file_id: row.try_get(2)?,
                            file: data_file,
                            delete_file_id: row.try_get(8)?,
                            delete_file,
                            row_id_start: None,
                            snapshot_id: None,
                            begin_snapshot: None,
                            schema_version: None,
                            partial_max: None,
                            max_row_count: row.try_get(14)?,
                            delete_count: None,
                            partition_id: None,
                            partition_values: Vec::new(),
                        },
                    })
                })
                .collect()
        })
    }

    fn get_data_files_added_between_snapshots(
        &self,
        table_id: i64,
        start_snapshot: i64,
        end_snapshot: i64,
    ) -> Result<Vec<DataFileChange>> {
        // CDC inherits catalog via table_id. No additional scoping.
        block_on(async {
            // Older catalogs predate `partial_max`; degrade it to NULL there
            // (they cannot contain partial files), matching the probe pattern
            // used by the scan queries above.
            let pm = if self.schema_capabilities().await?.data_file_partial_max {
                "data.partial_max::bigint"
            } else {
                "NULL::bigint"
            };
            let rows = sqlx::query(AssertSqlSafe(format!(
                "SELECT
                    data.begin_snapshot,
                    data.path,
                    data.path_is_relative,
                    data.file_size_bytes,
                    data.footer_size,
                    data.encryption_key,
                    data.row_id_start,
                    {pm},
                    data.mapping_id
                FROM ducklake_data_file AS data
                WHERE data.table_id = $1
                  AND data.begin_snapshot <= $3
                  AND (data.begin_snapshot >= $2
                       OR ({pm} IS NOT NULL AND {pm} >= $2))
                ORDER BY data.begin_snapshot"
            )))
            .bind(table_id)
            .bind(start_snapshot)
            .bind(end_snapshot)
            .fetch_all(&self.pool)
            .await?;

            rows.into_iter()
                .map(|row| {
                    Ok(DataFileChange {
                        begin_snapshot: row.try_get(0)?,
                        path: row.try_get(1)?,
                        path_is_relative: row.try_get(2)?,
                        file_size_bytes: row.try_get(3)?,
                        footer_size: row.try_get(4)?,
                        encryption_key: row.try_get(5)?,
                        row_id_start: row.try_get(6)?,
                        partial_max: row.try_get(7)?,
                        mapping_id: row.try_get(8)?,
                    })
                })
                .collect()
        })
    }

    fn get_delete_files_added_between_snapshots(
        &self,
        table_id: i64,
        start_snapshot: i64,
        end_snapshot: i64,
    ) -> Result<Vec<DeleteFileChange>> {
        // Same shape as the single-catalog PostgresMetadataProvider — inherits
        // catalog via table_id.
        block_on(async {
            // Cumulative (current-spec) delete files can hold in-window deletions
            // even when their begin_snapshot predates the window; included via
            // `ducklake_delete_file.partial_max`. Older catalogs lack the column
            // (and cumulative delete files); degrade it to NULL there.
            let pm = if self.schema_capabilities().await?.delete_file_partial_max {
                "ddf.partial_max::bigint"
            } else {
                "NULL::bigint"
            };
            let rows = sqlx::query(AssertSqlSafe(format!(
                r#"
WITH current_delete AS (
    SELECT
        ddf.data_file_id,
        ddf.begin_snapshot,
        ddf.path,
        ddf.path_is_relative,
        ddf.file_size_bytes,
        ddf.footer_size,
        ddf.encryption_key
    FROM ducklake_delete_file ddf
    WHERE ddf.table_id = $1
      AND ddf.begin_snapshot <= $3
      AND (ddf.begin_snapshot >= $2
           OR ({pm} IS NOT NULL AND {pm} >= $2))
),
data_files AS (
    SELECT df.*
    FROM ducklake_data_file df
    WHERE df.table_id = $1
)
SELECT
    data.path, data.path_is_relative, data.file_size_bytes, data.footer_size,
    data.row_id_start, data.record_count, data.mapping_id,
    current_delete.path, current_delete.path_is_relative,
    current_delete.file_size_bytes, current_delete.footer_size,
    prev.path, prev.path_is_relative, prev.file_size_bytes, prev.footer_size,
    current_delete.begin_snapshot
FROM current_delete
JOIN data_files data USING (data_file_id)
LEFT JOIN LATERAL (
    SELECT ddf.path, ddf.path_is_relative, ddf.file_size_bytes, ddf.footer_size
    FROM ducklake_delete_file ddf
    WHERE ddf.table_id = $1
      AND ddf.data_file_id = current_delete.data_file_id
      AND ddf.begin_snapshot < current_delete.begin_snapshot
    ORDER BY ddf.begin_snapshot DESC
    LIMIT 1
) prev ON true
UNION ALL
SELECT
    data.path, data.path_is_relative, data.file_size_bytes, data.footer_size,
    data.row_id_start, data.record_count, data.mapping_id,
    NULL::VARCHAR, NULL::BOOLEAN, NULL::BIGINT, NULL::BIGINT,
    prev.path, prev.path_is_relative, prev.file_size_bytes, prev.footer_size,
    data.end_snapshot
FROM ducklake_data_file data
LEFT JOIN LATERAL (
    SELECT ddf.path, ddf.path_is_relative, ddf.file_size_bytes, ddf.footer_size
    FROM ducklake_delete_file ddf
    WHERE ddf.table_id = $1
      AND ddf.data_file_id = data.data_file_id
      AND ddf.begin_snapshot < data.end_snapshot
    ORDER BY ddf.begin_snapshot DESC
    LIMIT 1
) prev ON true
WHERE data.table_id = $1
  AND data.end_snapshot >= $2
  AND data.end_snapshot <= $3
"#
            )))
            .bind(table_id)
            .bind(start_snapshot)
            .bind(end_snapshot)
            .fetch_all(&self.pool)
            .await?;

            rows.into_iter()
                .map(|row| {
                    Ok(DeleteFileChange {
                        data_file_path: row.try_get(0)?,
                        data_file_path_is_relative: row.try_get(1)?,
                        data_file_size_bytes: row.try_get(2)?,
                        data_file_footer_size: row.try_get(3)?,
                        data_row_id_start: row.try_get(4)?,
                        data_record_count: row.try_get(5)?,
                        data_mapping_id: row.try_get(6)?,
                        current_delete_path: row.try_get(7)?,
                        current_delete_path_is_relative: row.try_get(8)?,
                        current_delete_file_size_bytes: row.try_get(9)?,
                        current_delete_footer_size: row.try_get(10)?,
                        previous_delete_path: row.try_get(11)?,
                        previous_delete_path_is_relative: row.try_get(12)?,
                        previous_delete_file_size_bytes: row.try_get(13)?,
                        previous_delete_footer_size: row.try_get(14)?,
                        snapshot_id: row.try_get(15)?,
                    })
                })
                .collect()
        })
    }
}

pub(crate) async fn catalog_has_data_path(pool: &PgPool) -> Result<bool> {
    Ok(sqlx::query_scalar(
        "SELECT EXISTS (
             SELECT 1 FROM pg_attribute
             WHERE attrelid = 'ducklake_catalog'::regclass
               AND attname = 'data_path'
               AND NOT attisdropped
         )",
    )
    .fetch_one(pool)
    .await?)
}
